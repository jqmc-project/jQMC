"""Deliver ``sys.settrace`` events to a GDD runtime receipt tracer.

``tests/conftest.py`` observes a governed pytest case by running
:meth:`gdd.engine.module_receipts.RuntimeObservation.tracer` as the thread's
``sys.settrace`` function.  On CPython 3.12 ``sys.settrace`` is a bridge over
``sys.monitoring`` (PEP 669; ``Python/legacy_tracing.c``): it enables call, line,
jump, return and exception events for *every* code object and calls the Python
trace function for every Python call.  A governed scientific test traces JAX
programs, so it pays one Python-level callback per JAX-internal call plus the
de-specialised interpreter for all code -- the dominant cost, which makes the
MPI GFMC twist-decomposition tests several times slower.

The receipt tracer ignores the ``call`` event of every frame that is neither a
test-tree frame, a ``jqmc`` production frame nor a registered comparator frame:
it returns ``None`` without side effects, so such a frame never receives a
local trace function either.  :func:`trace_events` therefore re-implements the
CPython 3.12 bridge (``_PyEval_SetTrace`` / ``trace_trampoline``) for exactly one
trace function on its own ``sys.monitoring`` tool id, and returns
``sys.monitoring.DISABLE`` for the ``call`` event of code whose filename the
caller-supplied predicate classifies as ignorable.  Every event delivered to the
tracer, its arguments, the ``f_trace`` update rule and the per-frame
``f_trace_lines`` / ``f_trace_opcodes`` switches follow the 3.12 sources.

The bridge is used only on CPython 3.12 (the version whose bridge it mirrors;
the CI evidence lanes run it).  Any other interpreter, an unavailable tool id,
or ``GDD_RUNTIME_TRACE_BACKEND=settrace`` uses ``sys.settrace`` unchanged.

The bridge is also used only while no other tracer is active.  The
bridge suspends this thread's legacy trace function with ``sys.settrace(None)``.
If another thread still has a legacy trace or profile function, CPython keeps
the legacy ``sys.settrace`` tool (``PY_MONITORING_SYS_TRACE_ID``) enabled
interpreter-wide.  Once any frame has set ``f_trace_opcodes`` (the receipt
tracer does this for tolerance frames), that tool also has ``INSTRUCTION``
enabled.  CPython 3.12's ``sys_trace_instruction_func``
(``Python/legacy_tracing.c``; verified on 3.12.3 and 3.12.13) then calls this
thread's ``c_tracefunc`` without checking it for ``NULL``, and the process
segfaults.  ``pytest --cov`` creates exactly this situation: coverage's
CTracer is installed in this thread, in the pytest-xdist/execnet threads, and
in the conftest VMA watchdog thread through ``threading.settrace``.
:func:`monitoring_bridge_available` therefore falls back to ``sys.settrace``
when this thread or ``threading`` has a trace or profile function, or when any
``sys.monitoring`` tool id is already in use.
"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import CodeType, FrameType
from typing import Any

TraceFunction = Callable[[FrameType, str, Any], Any]

_BRIDGE_VERSION = (3, 12)
_BACKEND_ENV = "GDD_RUNTIME_TRACE_BACKEND"
# Tool ids 3 and 4 have no reserved meaning (0 debugger, 1 coverage, 2 profiler,
# 5 optimizer); fall back to any free id before giving up on the bridge.
_TOOL_PREFERENCE = (4, 3, 0, 1, 2, 5)


def monitoring_bridge_available() -> bool:
    """Whether :func:`trace_events` would use the ``sys.monitoring`` bridge now."""
    if sys.version_info[:2] != _BRIDGE_VERSION or not hasattr(sys, "monitoring"):
        return False
    if os.environ.get(_BACKEND_ENV, "monitoring") == "settrace":
        return False
    return not other_tracer_active()


def other_tracer_active() -> bool:
    """Whether a trace/profile function or a ``sys.monitoring`` tool is already active.

    A legacy trace or profile function in any other thread makes the bridge
    unsafe on CPython 3.12 (see the module docstring).  Other threads' functions
    cannot be read directly.  Coverage and debuggers install theirs in this
    thread and through ``threading.settrace`` / ``threading.setprofile``, so
    those are the observable signals.  An occupied ``sys.monitoring`` tool id
    (for example ``COVERAGE_CORE=sysmon``) is treated the same way.
    """
    if sys.gettrace() is not None or sys.getprofile() is not None:
        return True
    if threading.gettrace() is not None or threading.getprofile() is not None:
        return True
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is None:
        return False
    return any(monitoring.get_tool(tool) is not None for tool in _TOOL_PREFERENCE)


@contextmanager
def trace_events(tracer: TraceFunction, needs_call_events: Callable[[str], bool]) -> Iterator[str]:
    """Run the block with ``tracer`` receiving this thread's settrace events.

    ``needs_call_events(filename)`` must return ``True`` for every code filename
    whose ``call`` event the tracer can act on (return non-``None`` or record
    anything).  Yields the backend name (``"monitoring"`` or ``"settrace"``).
    The previously installed ``sys.settrace`` function is suspended for the
    block and restored afterwards in both backends.
    """
    previous = sys.gettrace()
    bridge = _MonitoringBridge.acquire(tracer, needs_call_events) if monitoring_bridge_available() else None
    if bridge is None:
        sys.settrace(tracer)
        try:
            yield "settrace"
        finally:
            sys.settrace(previous)
        return
    sys.settrace(None)
    try:
        bridge.start()
        yield "monitoring"
    finally:
        bridge.stop()
        sys.settrace(previous)


class _MonitoringBridge:
    """Per-thread emulation of the CPython 3.12 ``sys.settrace`` bridge."""

    def __init__(self, tool: int, tracer: TraceFunction, needs_call_events: Callable[[str], bool]) -> None:
        mon = sys.monitoring
        self._mon = mon
        self._events = mon.events
        self._disable = mon.DISABLE
        self._tool = tool
        self._tracer = tracer
        self._needs_call_events = needs_call_events
        self._thread = threading.get_ident()
        self._relevant: dict[str, bool] = {}
        # id(code) -> (code, local event mask); the code object is kept alive
        # so an id cannot be reused while its local events are installed.
        self._local: dict[int, tuple[CodeType, int]] = {}
        self._lines: dict[int, tuple[CodeType, dict[int, int]]] = {}
        self._active = False
        E = self._events
        self._line_events = E.LINE | E.JUMP | E.PY_RETURN | E.PY_YIELD | E.STOP_ITERATION

    @classmethod
    def acquire(cls, tracer: TraceFunction, needs_call_events: Callable[[str], bool]) -> _MonitoringBridge | None:
        mon = sys.monitoring
        for tool in _TOOL_PREFERENCE:
            if mon.get_tool(tool) is not None:
                continue
            try:
                mon.use_tool_id(tool, "gdd-runtime-receipts")
            except ValueError:
                continue
            return cls(tool, tracer, needs_call_events)
        return None

    # ------------------------------------------------------------------ setup
    def start(self) -> None:
        mon, E, tool = self._mon, self._events, self._tool
        register = mon.register_callback
        register(tool, E.PY_START, self._on_start)
        register(tool, E.PY_RESUME, self._on_start)
        register(tool, E.PY_THROW, self._on_throw)
        register(tool, E.PY_RETURN, self._on_return)
        register(tool, E.PY_YIELD, self._on_return)
        register(tool, E.PY_UNWIND, self._on_unwind)
        register(tool, E.RAISE, self._on_exception)
        register(tool, E.STOP_ITERATION, self._on_exception)
        register(tool, E.LINE, self._on_line)
        register(tool, E.JUMP, self._on_jump)
        register(tool, E.INSTRUCTION, self._on_instruction)
        self._active = True
        # RAISE / PY_UNWIND / PY_THROW are not local events; they are cheap
        # (exception paths only) and filtered by ``f_trace`` like the bridge.
        mon.set_events(tool, E.PY_START | E.PY_RESUME | E.PY_THROW | E.PY_UNWIND | E.RAISE)

    def stop(self) -> None:
        mon, tool = self._mon, self._tool
        self._active = False
        try:
            mon.set_events(tool, 0)
            for code, _mask in self._local.values():
                mon.set_local_events(tool, code, 0)
            for event in (
                self._events.PY_START,
                self._events.PY_RESUME,
                self._events.PY_THROW,
                self._events.PY_RETURN,
                self._events.PY_YIELD,
                self._events.PY_UNWIND,
                self._events.RAISE,
                self._events.STOP_ITERATION,
                self._events.LINE,
                self._events.JUMP,
                self._events.INSTRUCTION,
            ):
                mon.register_callback(tool, event, None)
        finally:
            self._local.clear()
            mon.free_tool_id(tool)
            # Locations disabled for this tool must fire again for the next
            # observation (a later tool with the same id starts clean).
            mon.restart_events()

    # --------------------------------------------------------------- helpers
    def _is_relevant(self, code: CodeType) -> bool:
        filename = code.co_filename
        relevant = self._relevant.get(filename)
        if relevant is None:
            relevant = bool(self._needs_call_events(filename))
            self._relevant[filename] = relevant
        return relevant

    def _abort(self, frame: FrameType) -> None:
        # trace_trampoline: an exception from the trace function removes the
        # trace function and clears the frame's local trace.
        self._active = False
        self._mon.set_events(self._tool, 0)
        frame.f_trace = None

    def _dispatch(self, frame: FrameType, callback: TraceFunction, event: str, arg: Any) -> None:
        try:
            result = callback(frame, event, arg)
        except BaseException:
            self._abort(frame)
            raise
        if result is not None:
            frame.f_trace = result
        self._sync_local_events(frame)

    def _sync_local_events(self, frame: FrameType) -> None:
        if frame.f_trace is None:
            return
        code = frame.f_code
        wanted = self._line_events
        if frame.f_trace_opcodes:
            wanted |= self._events.INSTRUCTION
        installed = self._local.get(id(code))
        current = 0 if installed is None else installed[1]
        if current | wanted != current:
            mask = current | wanted
            self._mon.set_local_events(self._tool, code, mask)
            self._local[id(code)] = (code, mask)

    def _line_of(self, code: CodeType, offset: int) -> int:
        # _Py_Instrumentation_GetLine: the line of the instruction at ``offset``.
        entry = self._lines.get(id(code))
        if entry is None or entry[0] is not code:
            table: dict[int, int] = {}
            for start, end, line in code.co_lines():
                for position in range(start, end, 2):
                    table[position] = -1 if line is None else line
            entry = (code, table)
            self._lines[id(code)] = entry
        return entry[1].get(offset, -1)

    # ------------------------------------------------------------- callbacks
    def _on_start(self, code: CodeType, offset: int) -> Any:
        # sys_trace_func2 -> trace_trampoline(PyTrace_CALL): the global function.
        if not self._is_relevant(code):
            # The tracer returns None without side effects for this code, and
            # a None result never installs a local trace; nothing is lost.
            return self._disable
        if not self._active or threading.get_ident() != self._thread:
            return None
        self._dispatch(sys._getframe(1), self._tracer, "call", None)
        return None

    def _on_throw(self, code: CodeType, offset: int, exception: BaseException) -> Any:
        # sys_trace_func3 (PY_THROW) -> PyTrace_CALL with None.
        if not self._active or threading.get_ident() != self._thread or not self._is_relevant(code):
            return None
        self._dispatch(sys._getframe(1), self._tracer, "call", None)
        return None

    def _local_callback(self) -> tuple[FrameType, TraceFunction] | None:
        if not self._active or threading.get_ident() != self._thread:
            return None
        frame = sys._getframe(2)
        callback = frame.f_trace
        if callback is None:
            return None
        return frame, callback

    def _on_return(self, code: CodeType, offset: int, value: Any) -> Any:
        # sys_trace_return / sys_trace_yield -> PyTrace_RETURN with the value.
        target = self._local_callback()
        if target is not None:
            self._dispatch(target[0], target[1], "return", value)
        return None

    def _on_unwind(self, code: CodeType, offset: int, exception: BaseException) -> Any:
        # sys_trace_func3 (PY_UNWIND) -> PyTrace_RETURN with None.
        target = self._local_callback()
        if target is not None:
            self._dispatch(target[0], target[1], "return", None)
        return None

    def _on_exception(self, code: CodeType, offset: int, exception: BaseException) -> Any:
        # sys_trace_exception_func (RAISE, STOP_ITERATION).
        target = self._local_callback()
        if target is not None:
            traceback = exception.__traceback__
            self._dispatch(target[0], target[1], "exception", (type(exception), exception, traceback))
        return None

    def _trace_line(self, frame: FrameType, line: int) -> None:
        # trace_line: honour f_trace_lines, skip negative lines, then trampoline.
        if not frame.f_trace_lines or line < 0:
            return
        callback = frame.f_trace
        if callback is None:
            return
        self._dispatch(frame, callback, "line", None)

    def _on_line(self, code: CodeType, line: int) -> Any:
        if not self._active or threading.get_ident() != self._thread:
            return None
        self._trace_line(sys._getframe(1), line)
        return None

    def _on_jump(self, code: CodeType, source: int, destination: int) -> Any:
        # sys_trace_jump_func: only backward jumps within one line emit a line
        # event; every other jump location is disabled (a new line is reported
        # by the LINE event of its first instruction).
        if not self._active or threading.get_ident() != self._thread:
            return None
        if destination > source:
            return self._disable
        to_line = self._line_of(code, destination)
        if to_line != self._line_of(code, source):
            return self._disable
        self._trace_line(sys._getframe(1), to_line)
        return None

    def _on_instruction(self, code: CodeType, offset: int) -> Any:
        # sys_trace_instruction_func: only frames that requested opcodes.
        if not self._active or threading.get_ident() != self._thread:
            return None
        frame = sys._getframe(1)
        if not frame.f_trace_opcodes:
            return None
        callback = frame.f_trace
        if callback is None:
            return None
        self._dispatch(frame, callback, "opcode", None)
        return None
