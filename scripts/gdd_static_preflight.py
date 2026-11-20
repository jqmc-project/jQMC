"""pytest plugin: static-only GDD contract preflight and its numerics guard.

The GDD preflight
runs only the static format and consistency checks. The GDD contract test
files (``scripts/run_gdd_preflight.py::GDD_CONTRACT_TESTS``) hold two kinds of
tests:

* static (S): checks of the GDD ledgers, bindings, receipts and guards that run
  no jqmc numerics. They run in the preflight AND in the main CI matrix.
* science-running (D): evidence-strength checks that execute real jqmc
  numerics (JAX programs or ``jqmc_rs`` kernels), typically injecting a fault
  into production and asserting that the science test turns red. They carry
  ``@pytest.mark.gdd_runs_science`` and run only in the main CI matrix, where
  these files already run in every environment and precision lane.

The classification was MEASURED with ``--gdd-numerics-probe=DIR``.
A test is science-running when, during its setup, call or teardown, at least
one of these signals fires in the pytest process or in an instrumented child
Python process:

1. a JAX trace or compilation event (``jax.monitoring`` events under
   ``/jax/core/compile/`` or ``/jax/compilation_cache/``). JAX caches are
   cleared before every test, so a cached executable cannot run unobserved:
   every JAX computation, including an eager ``jnp`` operation, traces and
   compiles at least one program;
2. a call of any ``jqmc_rs`` native entry point (every public module attribute
   is wrapped before jqmc imports it).

The probe additionally records, as information only, which ``jqmc`` Python
source files a test calls into (``sys.setprofile``).

``--gdd-static-preflight`` is the preflight mode. It deselects the marked
tests, defaults every unmarked test to static (so a new test runs in the
preflight unless explicitly marked), and turns both signals into failures: the
JAX listener and the ``jqmc_rs`` wrappers raise ``StaticPreflightNumericsError``
at the first numerical execution, and the test report is forced to "failed"
even if the test swallowed that exception. Child Python processes inherit the
guard through a generated ``sitecustomize`` on ``PYTHONPATH`` (a child started
with ``-I``/``-S`` or with a scrubbed ``PYTHONPATH`` is not instrumented). At the
end the plugin requires that every selected test produced a report (executed
count equals selected count) and that collection partitioned every collected
contract test into the static and the science-running set.
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

import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import tempfile
import threading
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

MARKER = "gdd_runs_science"
MARKER_HELP = (
    "gdd_runs_science: GDD contract test that executes real jqmc numerics (JAX or jqmc_rs); measured "
    "classification, deselected from the static GDD preflight and run only with the ordinary test suite"
)
GUARD_OPTION = "--gdd-static-preflight"
PROBE_OPTION = "--gdd-numerics-probe"
JAX_EVENT_PREFIXES = ("/jax/core/compile/", "/jax/compilation_cache/")
CHILD_ENV = "GDD_NUMERICS_CHILD"
NODE_ENV = "GDD_NUMERICS_NODE"
SUMMARY_KEY = "gdd_static_preflight"


class StaticPreflightNumericsError(RuntimeError):
    """A test selected as static executed jqmc numerics in the static preflight."""


class _State:
    """Process-wide signal recorder shared by the JAX listener and native wrappers."""

    def __init__(self, mode: str, child_dir: Path | None) -> None:
        self.mode = mode  # "guard" or "probe"
        self.child_dir = child_dir
        self.reset(None)

    def reset(self, node: str | None) -> None:
        self.node = node
        self.jax: Counter[str] = Counter()
        self.native: Counter[str] = Counter()
        self.python: Counter[str] = Counter()
        self.violations: list[str] = []

    def _signal(self, kind: str, name: str) -> None:
        if self.node is None or self.mode == "off":
            return  # Import/collection time outside any test, or a retired child recorder.
        (self.jax if kind == "jax" else self.native)[name] += 1
        if self.mode == "probe" and sys.getprofile() is not None:
            # The test is now known to be science-running; the informational
            # Python-call profile is not needed to classify it, so stop paying
            # for it (keeps long numerical tests at native speed).
            sys.setprofile(None)
            self.python["<profile stopped at first numerical signal>"] += 1
        if self.mode == "guard":
            message = (
                f"GDD static preflight: {self.node} executed jqmc numerics ({kind}: {name}). A test that runs "
                f"JAX or jqmc_rs must carry @pytest.mark.{MARKER}."
            )
            self.violations.append(message)
            raise StaticPreflightNumericsError(message)

    def jax_event(self, event: str, *args: Any, **kwargs: Any) -> None:
        if event.startswith(JAX_EVENT_PREFIXES):
            self._signal("jax", event)

    def native_call(self, name: str) -> None:
        self._signal("native", name)


def _install_jax_listeners(state: _State) -> None:
    import jax.monitoring

    jax.monitoring.register_event_listener(state.jax_event)
    jax.monitoring.register_event_duration_secs_listener(state.jax_event)


def _clear_jax_caches() -> None:
    jax = sys.modules.get("jax")
    if jax is not None:
        jax.clear_caches()


def _wrap_native_module(module: Any, state: _State) -> None:
    """Route every public ``jqmc_rs`` callable through the signal recorder."""
    for name in dir(module):
        if name.startswith("__"):
            continue
        target = getattr(module, name)
        if not callable(target) or getattr(target, "__gdd_native__", None) is not None:
            continue

        def wrapper(*args: Any, __target=target, __name=name, **kwargs: Any) -> Any:
            state.native_call(__name)
            return __target(*args, **kwargs)

        try:
            functools.update_wrapper(wrapper, target)
        except (AttributeError, TypeError):
            pass
        wrapper.__gdd_native__ = target
        setattr(module, name, wrapper)


class _PostImportHook(importlib.abc.MetaPathFinder):
    """Instrument ``jax``/``jqmc_rs`` in a child process only if the child imports them."""

    def __init__(self, state: _State) -> None:
        self.state = state
        self.pending = {"jax", "jqmc_rs"}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.pending:
            return None
        self.pending.discard(fullname)
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        original = spec.loader.exec_module
        state = self.state

        def exec_module(module):
            original(module)
            if fullname == "jax":
                _install_jax_listeners(state)
            else:
                _wrap_native_module(module, state)

        spec.loader.exec_module = exec_module
        return spec


_CHILD_STATE: _State | None = None
CHILD_MODULE = "_gdd_static_preflight_child"


def _retire_child_recorder() -> None:
    """Disable an inherited child recorder in a process that runs the plugin itself.

    An xdist worker is a child of the controller, so it inherits the generated
    ``sitecustomize``. Inside the worker the per-test plugin is the authority;
    the process-wide child recorder (which has no test boundaries) is retired.
    """
    module = sys.modules.get(CHILD_MODULE)
    state = getattr(module, "_CHILD_STATE", None)
    if state is not None:
        state.mode = "off"
    sys.meta_path[:] = [finder for finder in sys.meta_path if type(finder).__name__ != "_PostImportHook"]


def install_child() -> None:
    """Entry point of the generated ``sitecustomize`` of child Python processes."""
    mode, _, directory = os.environ.get(CHILD_ENV, "").partition(":")
    if mode not in {"guard", "probe"} or not directory:
        return
    global _CHILD_STATE
    state = _State(mode, Path(directory))
    # Only a process started while a test runs carries NODE_ENV; one started
    # outside any test (an xdist worker) records nothing.
    state.reset(os.environ.get(NODE_ENV) or None)
    _CHILD_STATE = state
    sys.meta_path.insert(0, _PostImportHook(state))
    if "jax" in sys.modules:
        _install_jax_listeners(state)
    if "jqmc_rs" in sys.modules:
        _wrap_native_module(sys.modules["jqmc_rs"], state)

    import atexit

    def report() -> None:
        if state.mode != "off" and (state.jax or state.native or state.violations):
            record = {
                "node": state.node,
                "pid": os.getpid(),
                "argv": sys.argv,
                "jax": dict(state.jax),
                "native": dict(state.native),
                "violations": state.violations,
            }
            path = state.child_dir / f"child-{os.getpid()}-{uuid.uuid4().hex}.json"
            path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")

    atexit.register(report)


_SITECUSTOMIZE = """\
import importlib.machinery as _machinery
import importlib.util as _util
import os as _os
import sys as _sys

_spec = _util.spec_from_file_location("_gdd_static_preflight_child", {plugin!r})
_module = _util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_module.install_child()
# Also run a sitecustomize this file shadows, if any.
_here = _os.path.dirname(_os.path.abspath(__file__))
_next = _machinery.PathFinder.find_spec(
    "sitecustomize", [p for p in _sys.path if _os.path.abspath(p or ".") != _here]
)
if _next is not None and _next.loader is not None:
    _next.loader.exec_module(_util.module_from_spec(_next))
"""


def _python_profiler(state: _State):
    import jqmc

    jqmc_dir = str(Path(jqmc.__file__).resolve().parent) + os.sep

    def profile(frame, event, arg):
        if event == "call":
            filename = frame.f_code.co_filename
            if filename.startswith(jqmc_dir):
                state.python[filename[len(jqmc_dir) :]] += 1
        elif event == "c_call":
            # Cross-check of the wrappers that also sees methods of native objects.
            owner = getattr(arg, "__self__", None)
            module = getattr(arg, "__module__", None) or getattr(type(owner), "__module__", None)
            if module == "jqmc_rs":
                state.native[f"profile:{getattr(arg, '__qualname__', arg)}"] += 1

    return profile


def pytest_addoption(parser):
    group = parser.getgroup("gdd-static-preflight")
    group.addoption(
        GUARD_OPTION,
        action="store_true",
        default=False,
        help="static GDD preflight: deselect @gdd_runs_science tests and fail any jqmc numerical execution",
    )
    group.addoption(
        PROBE_OPTION,
        default=None,
        metavar="DIR",
        help="measure per-test jqmc numerical execution (JAX events, jqmc_rs calls, jqmc Python calls) into DIR",
    )


class _Plugin:
    def __init__(self, config, mode: str, out_dir: Path | None) -> None:
        self.config = config
        self.mode = mode
        self.out_dir = out_dir
        self.worker = getattr(config, "workerinput", None) is not None
        self.child_dir = Path(tempfile.mkdtemp(prefix="gdd-numerics-children-"))
        self.state = _State(mode, self.child_dir)
        self.records: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.collected: list[str] = []
        self.selected: list[str] = []
        self.deselected: list[str] = []
        self.worker_selected: dict[str, list[str]] = {}
        self.worker_summaries: list[dict[str, Any]] = []
        self.reported: set[str] = set()
        self.problems: list[str] = []

    def install(self) -> None:
        _retire_child_recorder()
        _install_jax_listeners(self.state)
        import jqmc_rs

        _wrap_native_module(jqmc_rs, self.state)
        site_dir = Path(tempfile.mkdtemp(prefix="gdd-numerics-site-"))
        (site_dir / "sitecustomize.py").write_text(
            _SITECUSTOMIZE.format(plugin=str(Path(__file__).resolve())), encoding="utf-8"
        )
        os.environ[CHILD_ENV] = f"{self.mode}:{self.child_dir}"
        existing = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = os.pathsep.join([str(site_dir), *([existing] if existing else [])])

    # -- collection ----------------------------------------------------------
    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, session, config, items):
        if self.mode != "guard":
            return
        science = [item for item in items if item.get_closest_marker(MARKER) is not None]
        static = [item for item in items if item.get_closest_marker(MARKER) is None]
        self.collected = [item.nodeid for item in items]
        self.selected = [item.nodeid for item in static]
        self.deselected = [item.nodeid for item in science]
        if len(science) + len(static) != len(items):
            self.problems.append("classification is not a partition of the collected contract tests")
        if not static:
            self.problems.append("the static preflight selected zero tests")
        if science:
            config.hook.pytest_deselected(items=science)
        items[:] = static
        if self.worker:
            self.config.workeroutput[SUMMARY_KEY] = json.dumps(
                {"collected": len(self.collected), "selected": len(static), "deselected": len(science)}
            )

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):
        self.worker_selected[node.gateway.id] = list(ids)

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node, error):
        summary = getattr(node, "workeroutput", {}).get(SUMMARY_KEY)
        if summary:
            self.worker_summaries.append(json.loads(summary))

    # -- per-test instrumentation -------------------------------------------
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        _clear_jax_caches()
        os.environ[NODE_ENV] = item.nodeid
        self.state.reset(item.nodeid)
        self.current = {"node": item.nodeid, "durations": {}, "outcomes": {}}
        if self.mode == "probe":
            profiler = _python_profiler(self.state)
            threading.setprofile(profiler)
            sys.setprofile(profiler)
        try:
            yield
        finally:
            if self.mode == "probe":
                sys.setprofile(None)
                threading.setprofile(None)
                self.current.update(jax=dict(self.state.jax), native=dict(self.state.native), python=dict(self.state.python))
                self.records.append(self.current)
            self.state.node = None
            self.current = None
            os.environ.pop(NODE_ENV, None)

    def _child_records(self, node: str) -> list[dict[str, Any]]:
        records = []
        for path in sorted(self.child_dir.glob("child-*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if record.get("node") == node:
                records.append(record)
                path.unlink(missing_ok=True)
        return records

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        for child in self._child_records(item.nodeid):
            for name, count in child.get("jax", {}).items():
                self.state.jax[f"child:{name}"] += count
            for name, count in child.get("native", {}).items():
                self.state.native[f"child:{name}"] += count
            if self.mode == "guard":
                self.state.violations.extend(child.get("violations") or [f"child process {child.get('argv')}"])
        if self.current is not None:
            self.current["durations"][report.when] = report.duration
            self.current["outcomes"][report.when] = report.outcome
        if self.mode == "guard" and self.state.violations:
            report.outcome = "failed"
            report.longrepr = "\n".join(dict.fromkeys(self.state.violations))
            self.state.violations = []

    # -- accounting ----------------------------------------------------------
    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
            self.reported.add(report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        if self.mode == "probe":
            if self.out_dir is not None:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                name = getattr(self.config, "workerinput", {}).get("workerid", "main")
                path = self.out_dir / f"probe-{name}-{os.getpid()}.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for record in self.records:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
            return
        if self.worker:
            return
        if self.worker_selected:
            id_sets = {tuple(ids) for ids in self.worker_selected.values()}
            if len(id_sets) != 1:
                self.problems.append("xdist workers selected different static test sets")
            selected = set(next(iter(id_sets)))
            summaries = {json.dumps(summary, sort_keys=True) for summary in self.worker_summaries}
            if len(summaries) != 1:
                self.problems.append(f"xdist workers reported inconsistent classification summaries: {sorted(summaries)}")
            summary = json.loads(next(iter(summaries))) if summaries else {}
            collected, deselected = summary.get("collected", -1), summary.get("deselected", -1)
        else:
            selected = set(self.selected)
            collected, deselected = len(self.collected), len(self.deselected)
        if collected != len(selected) + deselected:
            self.problems.append(
                f"classification incomplete: collected {collected} != static {len(selected)} + science {deselected}"
            )
        missing = sorted(selected - self.reported)
        extra = sorted(self.reported - selected)
        if missing or extra:
            self.problems.append(
                f"executed count {len(self.reported)} != selected count {len(selected)} "
                f"(missing={missing[:20]}, unexpected={extra[:20]})"
            )
        print(
            f"\nGDD static preflight: collected {collected}, static selected {len(selected)}, "
            f"science-running deselected {deselected}, executed {len(self.reported)}",
            flush=True,
        )
        if self.problems:
            for problem in self.problems:
                print(f"GDD static preflight FAILED: {problem}", flush=True)
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_configure(config):
    config.addinivalue_line("markers", MARKER_HELP)
    guard = config.getoption(GUARD_OPTION)
    probe = config.getoption(PROBE_OPTION)
    if guard and probe:
        raise pytest.UsageError(f"{GUARD_OPTION} and {PROBE_OPTION} are mutually exclusive")
    if not guard and not probe:
        return
    plugin = _Plugin(config, "guard" if guard else "probe", Path(probe).resolve() if probe else None)
    plugin.install()
    config.pluginmanager.register(plugin, "gdd-static-preflight-active")
