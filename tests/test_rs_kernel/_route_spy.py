"""The PRODUCTION ROUTE PARITY spy (layer 2 of the two-layer design).

Layer 1 (``test_rs_analytic_matrix.py`` and the ported per-kernel modules) compares
FREE PyO3 kernels against the XLA analytic reference. It answers "is the formula
right?". It cannot answer "does production REACH that formula, with the right
arguments, and reduce the right blocks back out?" -- and that hole has passed two
REAL defects through a green suite:

  * **complex dltmove flag** -- the complex GFMC ``dltmove`` bool was absent from the
    ``run_gfmc_projection_{n,t}_complex`` PyO3 signature and hard-coded ``false``
    inside, so every complex-twist ECP GFMC run silently executed the t-move
    Hamiltonian. The free-function test ``_gfmc_lrdmc_ecp_complex(..., dltmove)``
    stayed green throughout.
  * **unwired complex param-deriv** -- the complex c-axis param-deriv kernels were ported and
    green as free functions while the driver was unwired. The later complex-DTN
    context is intentionally tracked at the same construction seam.

Both are "kernel correct, route broken". This module is the seam that sees them.

WHAT IT DOES
------------
``install_ctx_spy(monkeypatch)`` monkeypatches ``jqmc_rs.build_obs_ctx`` (and, on the
complex-twist param-deriv path, ``jqmc.jqmc_mcmc._RsComplexPdCtx``) so that the object
production actually calls methods on is a recording proxy. PRODUCTION CODE IS NEVER
MODIFIED -- the seam is the module attribute the driver looks up.

Per call it records:

  * the method name;
  * **every argument BOUND TO ITS PARAMETER NAME**, using the PyO3
    ``__text_signature__``, so a positionally-passed ``dltmove`` and a
    keyword-passed one are the same observation, and an argument left at its
    default is distinguishable from one explicitly passed the default value;
  * a structural descriptor of the return value (block set + ORDER, shape, dtype),
    so a dropped block / transposed pair / demoted dtype is visible without
    looking at values.

The ctx CONSTRUCTION arguments are recorded too -- that is where ``want_force`` /
``want_stress`` / ``want_param_deriv`` / ``want_gfmc`` / ``complex`` / ``pbc`` live,
and a ctx built without the block production later asks for is the unwired-driver
failure mode exactly.

WHY BOUND ARGS AND NOT ``*args``
--------------------------------
Recording raw ``*args`` would make "the driver passed 11 positionals" the only observable, which
is precisely the resolution at which the dropped ``dltmove`` flag hid. ``assert_arg_reached`` therefore
takes a PARAMETER NAME and fails when it is absent from the signature, when it was
never passed, or when it silently sat at its declared default while production had
asked for something else.

Tolerances: none live here. Value comparisons belong to the caller and must use the
named tiers from ``jqmc/_setting.py``.
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

import dataclasses
from typing import Any

import numpy as np

#: The arguments whose silent loss produced a real, shipped defect.
#: A route-parity test that exercises a method carrying one of these MUST assert it.
DISCRIMINATING_ARGS = (
    "dltmove",
    "mpc_data",
    "geminal_inv",
    "jastrow_data",
    "want_force",
    "want_stress",
)

_SENTINEL_MISSING = object()


# --------------------------------------------------------------------------- #
# PyO3 signature parsing (``__text_signature__``)
# --------------------------------------------------------------------------- #
def _parse_text_signature(fn) -> tuple[tuple[str, ...], dict[str, Any]] | None:
    """``(param_names, defaults)`` from a PyO3 ``__text_signature__``, or ``None``.

    PyO3 emits e.g. ``($self, r_up, r_dn, alat, ..., random_mesh=False, dltmove=False,
    nn=1, mpc_data=None, want_force=False, want_stress=False)``. These signatures are
    flat (no nested parens / no annotations), so a split on ``", "`` is exact; we assert
    that rather than assume it, because a future nested default would silently mis-bind
    every argument after it.
    """
    ts = getattr(fn, "__text_signature__", None)
    if not ts:
        return None
    body = ts.strip()
    assert body.startswith("(") and body.endswith(")"), f"unexpected __text_signature__ {ts!r}"
    body = body[1:-1]
    if not body.strip():
        return (), {}
    assert "(" not in body and ")" not in body, (
        f"__text_signature__ {ts!r} contains a nested paren; the flat split below would mis-bind arguments"
    )
    names: list[str] = []
    defaults: dict[str, Any] = {}
    for tok in body.split(","):
        tok = tok.strip()
        if not tok or tok in ("$self", "/", "*"):
            continue
        if "=" in tok:
            name, raw = tok.split("=", 1)
            name = name.strip()
            names.append(name)
            defaults[name] = _literal(raw.strip())
        else:
            names.append(tok)
    return tuple(names), defaults


def _literal(raw: str):
    """Best-effort literal for a PyO3 default token (``None`` / ``False`` / ``1`` / ...)."""
    for text, val in (("None", None), ("True", True), ("False", False)):
        if raw == text:
            return val
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _bind(fn, args, kwargs) -> dict[str, Any]:
    """Bind a call to ``fn`` onto parameter names.

    Parameters that were NOT supplied are recorded as ``_SENTINEL_MISSING`` rather
    than filled from their default -- that distinction is the whole point: "production
    never passed ``dltmove``" and "production passed ``dltmove=False``" are different
    facts, and the dropped ``dltmove`` flag was the first indistinguishable from the second.
    """
    parsed = _parse_text_signature(fn)
    if parsed is None:
        # No introspectable signature (a plain Python adapter method): fall back to
        # positional indices so the call is still recorded, and keep the kwargs verbatim.
        bound = {f"arg{i}": v for i, v in enumerate(args)}
        bound.update(kwargs)
        return bound
    names, _defaults = parsed
    bound: dict[str, Any] = {n: _SENTINEL_MISSING for n in names}
    for i, v in enumerate(args):
        if i < len(names):
            bound[names[i]] = v
        else:
            bound[f"extra{i}"] = v
    bound.update(kwargs)
    return bound


def _defaults_of(fn) -> dict[str, Any]:
    parsed = _parse_text_signature(fn)
    return {} if parsed is None else parsed[1]


# --------------------------------------------------------------------------- #
# Return-structure descriptor (block set / ORDER / shape / dtype)
# --------------------------------------------------------------------------- #
def describe_return(v) -> Any:
    """Structural fingerprint of a return value: block names IN ORDER, shape, dtype.

    A dict's keys are kept as an ORDERED tuple (not a set) so a reordered block set is
    a difference; a tuple's length is kept so a dropped element of a ``(alpha, beta,
    gamma)`` DTN triple is a difference; arrays carry shape + dtype so a demoted
    complex->real return is a difference even when the real parts agree.
    """
    if v is None:
        return ("none",)
    if isinstance(v, dict):
        return ("dict", tuple((k, describe_return(v[k])) for k in v))
    if isinstance(v, (tuple, list)):
        return ("seq", len(v), tuple(describe_return(x) for x in v))
    if isinstance(v, (bool, int, float, complex)):
        return ("scalar", type(v).__name__)
    arr = np.asarray(v)
    if arr.ndim == 0:
        return ("scalar", arr.dtype.str)
    return ("array", tuple(arr.shape), arr.dtype.str)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class CtxCall:
    method: str
    bound: dict[str, Any]
    defaults: dict[str, Any]
    ret: Any

    def supplied(self, arg: str) -> bool:
        return arg in self.bound and self.bound[arg] is not _SENTINEL_MISSING

    def value(self, arg: str) -> Any:
        v = self.bound.get(arg, _SENTINEL_MISSING)
        return None if v is _SENTINEL_MISSING else v

    def has_param(self, arg: str) -> bool:
        return arg in self.bound or arg in self.defaults


class RouteSpy:
    """Recorder for every production call onto the native observable context."""

    def __init__(self):
        self.build_calls: list[dict[str, Any]] = []
        self.calls: list[CtxCall] = []
        #: methods reached on the PYTHON complex param-deriv adapter rather than on
        #: ``NativeObsCtx`` -- a genuinely different route, recorded separately so a
        #: test cannot mistake one for the other.
        self.adapter_calls: list[CtxCall] = []

    # -- queries ----------------------------------------------------------------
    def names(self) -> set[str]:
        return {c.method for c in self.calls}

    def all_names(self) -> set[str]:
        return self.names() | {c.method for c in self.adapter_calls}

    @staticmethod
    def _matches(recorded: str, method: str) -> bool:
        """Name match with the rayon-batch alias: a query for
        ``"mcmc_sweep"`` also counts ``"mcmc_sweep_batch"`` -- post-rayon the
        driver issues ONE batched call per step covering every walker, and the
        batch entry is the SAME production route (thin `for_each_walker` fan of
        the per-walker body). Exact ``*_batch`` queries still match exactly, so
        rows that pin the batch entry by name keep their stricter assertion.
        This alias applies to expect AND forbid semantics alike: forbidding the
        per-walker name also forbids its batch twin.
        """
        return recorded == method or recorded == method + "_batch"

    def count(self, method: str) -> int:
        return sum(1 for c in self.calls if self._matches(c.method, method))

    def calls_to(self, method: str) -> list[CtxCall]:
        return [c for c in self.calls if self._matches(c.method, method)]

    def adapter_count(self, method: str) -> int:
        return sum(1 for c in self.adapter_calls if c.method == method)

    def build_count(self) -> int:
        return len(self.build_calls)

    def build_arg(self, arg: str) -> Any:
        assert self.build_calls, "build_obs_ctx was never called -- production did not take the native route"
        v = self.build_calls[-1].get(arg, _SENTINEL_MISSING)
        assert v is not _SENTINEL_MISSING, f"build_obs_ctx was called without {arg!r}; bound={sorted(self.build_calls[-1])}"
        return v

    def summary(self) -> str:
        parts = [f"build_obs_ctx x{len(self.build_calls)}"]
        for n in sorted(self.names()):
            parts.append(f"{n} x{self.count(n)}")
        for n in sorted({c.method for c in self.adapter_calls}):
            parts.append(f"(adapter){n} x{self.adapter_count(n)}")
        return ", ".join(parts) if parts else "<no native calls>"


class _CtxProxy:
    """Recording stand-in for the once-built ``NativeObsCtx``."""

    def __init__(self, inner, spy: RouteSpy, sink: str = "calls"):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_spy", spy)
        object.__setattr__(self, "_sink", sink)

    def __getattr__(self, name):
        inner = object.__getattribute__(self, "_inner")
        spy = object.__getattribute__(self, "_spy")
        sink = object.__getattribute__(self, "_sink")
        attr = getattr(inner, name)
        if not callable(attr) or name.startswith("__"):
            return attr

        def _recorded(*a, **k):
            ret = attr(*a, **k)
            rec = CtxCall(method=name, bound=_bind(attr, a, k), defaults=_defaults_of(attr), ret=describe_return(ret))
            getattr(spy, sink).append(rec)
            return ret

        return _recorded


def install_ctx_spy(monkeypatch, spy_complex_pd_adapter: bool = True) -> RouteSpy:
    """Wrap the object production gets from ``jqmc_rs.build_obs_ctx``.

    ``spy_complex_pd_adapter`` additionally wraps ``jqmc.jqmc_mcmc._RsComplexPdCtx``,
    the PYTHON adapter that serves the complex-twist c-axis param derivatives. Its DTN
    methods delegate to the shared-geminal native context; other families use the free
    complex kernels. Adapter-surface calls land in
    ``spy.adapter_calls``, never in ``spy.calls``, so a route-parity test cannot mistake
    "the free complex kernel ran via a Python shim" for "the native ctx method ran".
    """
    import jqmc_rs

    spy = RouteSpy()
    orig_build = jqmc_rs.build_obs_ctx

    def spy_build(*a, **k):
        inner = orig_build(*a, **k)
        spy.build_calls.append(_bind(orig_build, a, k))
        return _CtxProxy(inner, spy, sink="calls")

    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", spy_build)

    if spy_complex_pd_adapter:
        from jqmc import jqmc_mcmc as _mcmc_mod

        orig_adapter = _mcmc_mod._RsComplexPdCtx

        def spy_adapter(*a, **k):
            return _CtxProxy(orig_adapter(*a, **k), spy, sink="adapter_calls")

        monkeypatch.setattr(_mcmc_mod, "_RsComplexPdCtx", spy_adapter)

    return spy


# --------------------------------------------------------------------------- #
# Assertions (the four route-parity checks)
# --------------------------------------------------------------------------- #
def assert_native_ran(spy: RouteSpy, method: str, *, at_least: int = 1, context: str = ""):
    """CHECK 1 -- the production route actually reached the native ctx method.

    Zero calls means either a capability gate routed the observable to JAX or the
    driver never wired this method at all. Both are the route-broken failure mode.
    """
    got = spy.count(method)
    assert got >= at_least, (
        f"route parity{' [' + context + ']' if context else ''}: expected the production driver to call "
        f"NativeObsCtx.{method}() at least {at_least}x, got {got}. Observed: {spy.summary()}"
    )


def assert_complex_pd_adapter_ran(spy: RouteSpy, method: str, *, at_least: int = 1, context: str = ""):
    """Require the production complex-param-deriv Rust adapter to run.

    Complex Bloch parameter derivatives use ``_RsComplexPdCtx`` as a common surface.
    DTN delegates to the shared native context and other families call exported
    ``jqmc_rs`` complex kernels; both are Rust production routes, but adapter
    calls are deliberately recorded separately from native-context calls so a
    test cannot accidentally accept either route.
    """
    got = spy.adapter_count(method)
    assert got >= at_least, (
        f"route parity{' [' + context + ']' if context else ''}: expected the production driver to call "
        f"_RsComplexPdCtx.{method}() at least {at_least}x, got {got}. Observed: {spy.summary()}"
    )


def assert_no_native_calls(spy: RouteSpy, context: str = ""):
    """Negative form of CHECK 1: assert the native ctx was NOT reached (JAX ran)."""
    assert not spy.calls, (
        f"route parity{' [' + context + ']' if context else ''}: expected NO native ctx call, got {spy.summary()}"
    )


def assert_arg_reached(spy: RouteSpy, method: str, arg: str, expected=_SENTINEL_MISSING, *, context: str = ""):
    """CHECK 2 -- ``arg`` exists on the PyO3 signature, was supplied, and carries
    ``expected`` (when given) rather than silently sitting at its declared default.

    This is the assertion the dropped ``dltmove`` flag needed and did not have. It fails in three distinct
    ways, each with its own message:

      a. the parameter is absent from the PyO3 signature entirely (the dropped-flag root cause:
         ``run_gfmc_projection_t_complex`` had no ``dltmove`` at all);
      b. the parameter exists but the driver never passed it (so the Rust default
         silently governs);
      c. the parameter was passed a value that differs from what production asked for.
    """
    calls = spy.calls_to(method)
    tag = f"{' [' + context + ']' if context else ''}"
    assert calls, f"route parity{tag}: {method}() was never called, so {arg!r} cannot have reached it. {spy.summary()}"
    for i, c in enumerate(calls):
        assert c.has_param(arg), (
            f"route parity{tag}: NativeObsCtx.{method}() has no parameter {arg!r} in its PyO3 signature "
            f"(bound={sorted(k for k in c.bound)}). This is the dropped-flag shape: the wrapper cannot carry the flag, "
            f"so the kernel default governs no matter what the user asked for."
        )
        assert c.supplied(arg), (
            f"route parity{tag}: call {i} of {method}() did not pass {arg!r}; the Rust default "
            f"{c.defaults.get(arg)!r} silently governs."
        )
        if expected is not _SENTINEL_MISSING:
            got = c.value(arg)
            assert bool(got) == bool(expected) if isinstance(expected, bool) else got == expected, (
                f"route parity{tag}: call {i} of {method}() received {arg}={got!r}, expected {expected!r}."
            )


def assert_arg_is_not_none(spy: RouteSpy, method: str, arg: str, *, context: str = ""):
    """CHECK 2, structural variant: the argument arrived carrying actual DATA.

    ``mpc_data`` / ``geminal_inv`` / ``jastrow_data`` all default to ``None``; a driver
    that builds the object and then forgets to thread it produces a green free-kernel
    test and a wrong production number.
    """
    assert_arg_reached(spy, method, arg, context=context)
    for i, c in enumerate(spy.calls_to(method)):
        assert c.value(arg) is not None, (
            f"route parity{' [' + context + ']' if context else ''}: call {i} of {method}() got {arg}=None -- "
            f"production built the object but did not thread it through."
        )


def assert_return_structure(spy: RouteSpy, method: str, expected, *, context: str = ""):
    """CHECK 3 -- block set / ORDER / shape / dtype of the return value."""
    calls = spy.calls_to(method)
    tag = f"{' [' + context + ']' if context else ''}"
    assert calls, f"route parity{tag}: {method}() was never called. {spy.summary()}"
    for i, c in enumerate(calls):
        assert c.ret == expected, f"route parity{tag}: {method}() call {i} returned structure {c.ret!r}, expected {expected!r}"


def assert_return_blocks(spy: RouteSpy, method: str, blocks: tuple[str, ...], *, context: str = ""):
    """CHECK 3, dict form: the returned mapping carries EXACTLY ``blocks``, in order."""
    calls = spy.calls_to(method)
    tag = f"{' [' + context + ']' if context else ''}"
    assert calls, f"route parity{tag}: {method}() was never called. {spy.summary()}"
    for i, c in enumerate(calls):
        assert c.ret[0] == "dict", f"route parity{tag}: {method}() call {i} did not return a dict, got {c.ret[0]}"
        got = tuple(k for k, _ in c.ret[1])
        assert got == blocks, f"route parity{tag}: {method}() call {i} returned blocks {got}, expected {blocks}"
