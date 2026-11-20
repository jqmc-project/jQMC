"""Suite-wide guard: NaN is never agreement in a numpy.testing comparison.

``np.testing.assert_allclose`` defaults to ``equal_nan=True``, and
``assert_array_equal`` / ``assert_array_almost_equal`` / ``assert_almost_equal``
treat same-position NaNs as equal with no way to opt out. When the implementation
under test and its reference share a formula that divides 0/0 (for example a Rust
kernel and the XLA reference, or an analytic derivative and its closed form), both
sides return NaN at the same entries and every such comparison passes. A real
defect (ECP Legendre derivative 0/0 at cos = +-1, NaN GFMC ECP forces) was hidden
exactly this way.

``install()`` (called from ``tests/conftest.py`` at import, before any test module
is imported) replaces the four comparators on the ``numpy.testing`` namespace:

* ``assert_allclose``: when the caller did not pass ``equal_nan``, the original
  numpy function runs with ``equal_nan=False``; a NaN on either side therefore
  fails, and the failure is re-raised as :class:`NaNAgreementError` naming the NaN
  counts. An explicit truthy ``equal_nan`` is accepted only from a call site listed
  in :data:`NAN_AGREEMENT_ALLOWLIST`; anywhere else it raises before comparing.
* ``assert_array_equal`` / ``assert_array_almost_equal`` / ``assert_almost_equal``:
  the original numpy function runs first (so a GDD receipt still records the call
  and a genuine mismatch still reports numpy's own message); if it passed while a
  NaN is present in either operand, :class:`NaNAgreementError` is raised unless the
  call site is allow-listed.

The original numpy function object is always the one that executes, so the GDD
tolerance tracer (``gdd/engine/tolerance_runtime.py``), which identifies comparator
calls by the code object of numpy's private implementation, still records every
authored call site: the wrapper frame sits between the test frame and numpy's frame
and is skipped by the tracer's caller walk.

The allow-list is keyed by ``(repo-relative file, function qualname)`` of the frame
that calls the comparator, with a written reason. It is a literal dict so the static
gate (``scripts/check_no_inline_tolerances.py``) can read it without importing this
module. Adding an entry is a reviewed policy change; ``# tol-ok`` comments do not
apply to it.
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
import inspect
import sys
from pathlib import Path

import numpy as np
import numpy.testing
from numpy.testing._private import utils as _np_testing_utils


REPO_ROOT = Path(__file__).resolve().parents[1]

#: (repo-relative path, function qualname) -> reviewed reason. Keep this a literal dict.
#: Reviewed: the suite contains no call that relies on NaN
#: agreement; the two ``equal_nan=True`` spellings in tests/ are inside source strings
#: of GDD tracer self-tests (test_gdd_ledger.py), not calls, so the list is empty.
NAN_AGREEMENT_ALLOWLIST: dict[tuple[str, str], str] = {}

GUARDED_COMPARATORS = ("assert_allclose", "assert_array_equal", "assert_array_almost_equal", "assert_almost_equal")

ORIGINALS = {name: getattr(_np_testing_utils, name) for name in GUARDED_COMPARATORS}
_ALLCLOSE_SIGNATURE = inspect.signature(ORIGINALS["assert_allclose"])


class NaNAgreementError(AssertionError):
    """A comparison would have accepted NaN as agreement."""


#: Objects that carry no numeric entries and are never recursed into. ``type`` matters:
#: ``np.asarray(np.float64)`` is a 0-d object array holding the class itself, and the
#: class has an ``__array__`` attribute (the unbound method), so recursing into it
#: never terminates (e.g. ``assert_array_equal(x.dtype, np.float64)``).
_NON_NUMERIC_LEAVES = (str, bytes, type, np.dtype)


def _nan_count(value) -> tuple[int, int]:
    """Return ``(nan entries, total entries)``; non-numeric operands count as NaN-free."""
    if isinstance(value, _NON_NUMERIC_LEAVES):
        return 0, 1
    try:
        array = np.asarray(value)
    except Exception:
        return 0, 0
    if array.dtype.kind in "fc":
        return int(np.count_nonzero(np.isnan(array))), int(array.size)
    if array.dtype.kind == "O":
        count = 0
        for item in array.ravel():
            if isinstance(item, (float, complex, np.floating, np.complexfloating)) and np.isnan(item):
                count += 1
            elif isinstance(item, _NON_NUMERIC_LEAVES) or item is value:
                # No numeric entries, or a 0-d object wrapper around the operand itself:
                # recursing would revisit the same object forever.
                continue
            elif isinstance(item, np.ndarray) or hasattr(item, "__array__"):
                count += _nan_count(item)[0]
        return count, int(array.size)
    return 0, int(array.size)


def caller_site(depth: int = 2) -> tuple[str, str]:
    """``(repo-relative file, qualname)`` of the frame that called the guarded comparator."""
    frame = sys._getframe(depth)
    path = Path(frame.f_code.co_filename)
    try:
        relative = path.resolve().relative_to(REPO_ROOT).as_posix()
    except (ValueError, OSError):
        relative = path.as_posix()
    return relative, frame.f_code.co_qualname


def _allowed(site: tuple[str, str]) -> bool:
    return site in NAN_AGREEMENT_ALLOWLIST


def _nan_message(comparator: str, site: tuple[str, str], actual, desired, detail: str) -> str:
    nan_a, size_a = _nan_count(actual)
    nan_d, size_d = _nan_count(desired)
    return (
        f"NaN-agreement guard: numpy.testing.{comparator} at {site[0]}::{site[1]} {detail} "
        f"(NaN in actual: {nan_a} of {size_a}; NaN in desired: {nan_d} of {size_d}). "
        "NaN on both sides at the same entries is not agreement: a shared 0/0 would make the "
        "comparison green while neither side computed a value. Fix the NaN, or add a reviewed "
        "entry to tests/_nan_guard.py:NAN_AGREEMENT_ALLOWLIST."
    )


@functools.wraps(ORIGINALS["assert_allclose"])
def assert_allclose(*args, **kwargs):
    site = caller_site()
    bound = _ALLCLOSE_SIGNATURE.bind(*args, **kwargs)
    if "equal_nan" in bound.arguments and bool(bound.arguments["equal_nan"]):
        if not _allowed(site):
            raise NaNAgreementError(
                f"NaN-agreement guard: explicit equal_nan={bound.arguments['equal_nan']!r} "
                f"at {site[0]}::{site[1]} is not in tests/_nan_guard.py:NAN_AGREEMENT_ALLOWLIST."
            )
        return ORIGINALS["assert_allclose"](*bound.args, **bound.kwargs)
    bound.arguments["equal_nan"] = False
    try:
        return ORIGINALS["assert_allclose"](*bound.args, **bound.kwargs)
    except AssertionError as exc:
        actual, desired = bound.arguments["actual"], bound.arguments["desired"]
        if _nan_count(actual)[0] or _nan_count(desired)[0]:
            raise NaNAgreementError(_nan_message("assert_allclose", site, actual, desired, "saw NaN")) from exc
        raise


def _nan_rejecting(name: str):
    original = ORIGINALS[name]

    @functools.wraps(original)
    def guarded(actual, desired, *args, **kwargs):
        site = caller_site()
        result = original(actual, desired, *args, **kwargs)
        if (_nan_count(actual)[0] or _nan_count(desired)[0]) and not _allowed(site):
            raise NaNAgreementError(_nan_message(name, site, actual, desired, "passed only because NaN == NaN"))
        return result

    return guarded


GUARDS = {
    "assert_allclose": assert_allclose,
    **{name: _nan_rejecting(name) for name in GUARDED_COMPARATORS if name != "assert_allclose"},
}


def install() -> None:
    """Put the guards on the public ``numpy.testing`` namespace (idempotent)."""
    for name, guard in GUARDS.items():
        setattr(numpy.testing, name, guard)


#: Path components that mark third-party installed code even when an environment
#: lives inside the checkout (CI creates its venv at ``$PWD/.venv-ci``).
_INSTALLED_CODE_DIRS = frozenset({"site-packages", "dist-packages"})


def _environment_prefixes() -> list[Path]:
    """Interpreter prefixes that lie strictly inside the repository (an in-tree virtualenv)."""
    prefixes = []
    for raw in {sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix}:
        try:
            prefix = Path(raw).resolve()
            prefix.relative_to(REPO_ROOT)
        except (ValueError, OSError):
            continue
        if prefix != REPO_ROOT:
            prefixes.append(prefix)
    return prefixes


def is_project_module_file(file) -> bool:
    """True only for the project's own source files (tests/, jqmc/, gdd/, scripts/, ...).

    A file must resolve under :data:`REPO_ROOT` and must not be installed third-party
    code: no ``site-packages`` / ``dist-packages`` path component, no directory between
    it and the repository root holding a ``pyvenv.cfg``, and not under an interpreter
    prefix that itself lies inside the repository. Without these exclusions an in-tree
    virtualenv (CI's ``.venv-ci``) makes numpy's own ``numpy.testing._private.utils``
    look like a repository module, its internal comparator globals get replaced by the
    guards, and every numpy-internal comparison is routed through the guard.
    """
    try:
        path = Path(file).resolve()
        relative = path.relative_to(REPO_ROOT)
    except (ValueError, OSError):
        return False
    if _INSTALLED_CODE_DIRS.intersection(relative.parts):
        return False
    for prefix in _environment_prefixes():
        try:
            path.relative_to(prefix)
        except ValueError:
            continue
        return False
    directory = path.parent
    while directory != REPO_ROOT and REPO_ROOT in directory.parents:
        if (directory / "pyvenv.cfg").is_file():
            return False
        directory = directory.parent
    return True


def rebind_stale_references(modules=None) -> list[str]:
    """Rebind module globals still holding an unguarded original; return what was rebound.

    A module that ran ``from numpy.testing import assert_allclose`` before
    :func:`install` holds the original function. Only the project's own modules
    (:func:`is_project_module_file`) are touched; installed third-party code is left
    alone even when its environment lives inside the checkout.
    """
    originals = {id(function): name for name, function in ORIGINALS.items()}
    rebound = []
    for module_name, module in list((modules if modules is not None else sys.modules).items()):
        file = getattr(module, "__file__", None)
        if not file or not is_project_module_file(file):
            continue
        for attribute, value in list(vars(module).items()):
            name = originals.get(id(value))
            if name is not None:
                setattr(module, attribute, GUARDS[name])
                rebound.append(f"{module_name}.{attribute}")
    return rebound
