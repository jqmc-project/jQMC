"""Every real PyO3 entry's arguments must exist on its complex twin.

A real entry argument missing from its complex twin is a silent defect: for example a
``dltmove`` accepted by ``run_gfmc_projection_{n,t}`` but not by ``..._complex`` would
run t-move on every complex GFMC run that requested ``dltmove``. That class of defect is
mechanically detectable -- diff the signatures -- and this test does it.

Scope: PyO3 entries whose name is ``X`` and ``X_complex``. A missing argument is a
defect UNLESS it is listed in ``_KNOWN_ASYMMETRIES`` with the ledger row that owns
it. The allowlist is the port worklist: an entry is deleted when its port lands,
and a NEW asymmetry is a hard failure.
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

import pathlib
import re

import pytest

pytestmark = [pytest.mark.with_chdir]

_LIB_RS = pathlib.Path(__file__).resolve().parents[2] / "jqmc" / "rs" / "src" / "lib.rs"

# MPC is an energy-only observable with no native derivative surface and no PyO3
# entry takes an ``mpc_data`` argument, so real/complex signature parity is
# unconditional.
# No signature mismatch may be hidden here.
_KNOWN_ASYMMETRIES: dict[tuple[str, str], dict[str, str]] = {}

_SIG_RE = re.compile(r'#\[pyo3\((?:name = "([^"]+)", )?signature = \(([^)]*)\)\)\]\s*(?:#\[allow[^\]]*\]\s*)*(?:pub )?fn (\w+)')


def _signatures() -> dict[str, list[str]]:
    src = _LIB_RS.read_text()
    out: dict[str, list[str]] = {}
    for m in _SIG_RE.finditer(src):
        name = m.group(1) or m.group(3)
        out[name] = [p.strip().split("=")[0].strip() for p in m.group(2).split(",") if p.strip()]
    return out


@pytest.mark.light
def test_lib_rs_is_parsable():
    """Guard the guard: if the regex stops matching, every check below goes vacuous."""
    sigs = _signatures()
    assert len(sigs) > 50, f"only {len(sigs)} pyo3 signatures parsed -- the extractor is broken"
    assert any(n.endswith("_complex") for n in sigs), "no complex entries parsed"


@pytest.mark.light
def test_complex_entries_accept_every_real_argument():
    """Complex twins contain the real arguments in the same relative order."""
    sigs = _signatures()
    violations: list[str] = []
    for name in sorted(sigs):
        if not name.endswith("_complex"):
            continue
        base = name[: -len("_complex")]
        if base not in sigs:
            continue
        known = _KNOWN_ASYMMETRIES.get((base, name), {})
        for arg in sigs[base]:
            if arg not in sigs[name] and arg not in known:
                violations.append(
                    f"{name} is missing '{arg}' present on {base} -- thread it through; non-MPC allowlist entries are forbidden"
                )
        shared_real = [arg for arg in sigs[base] if arg in sigs[name]]
        shared_complex = [arg for arg in sigs[name] if arg in sigs[base]]
        if shared_real != shared_complex:
            violations.append(f"{name} reorders shared arguments {shared_complex}; real order is {shared_real}")
    assert violations == [], "real/complex signature asymmetry:\n  " + "\n  ".join(violations)


@pytest.mark.light
def test_known_asymmetry_allowlist_has_no_stale_entries():
    """An allowlisted argument that EXISTS on the complex entry means the port is done: delete the row.

    Without this the allowlist rots into a list of things nobody re-checks -- the
    exact failure mode of a dead fallback guard.
    """
    sigs = _signatures()
    stale: list[str] = []
    for (base, comp), args in _KNOWN_ASYMMETRIES.items():
        if base not in sigs or comp not in sigs:
            stale.append(f"{base}/{comp}: pair no longer exists in lib.rs")
            continue
        for arg in args:
            if arg in sigs[comp]:
                stale.append(f"{comp}: '{arg}' is now present -- remove it from _KNOWN_ASYMMETRIES")
            elif arg not in sigs[base]:
                stale.append(f"{base}: '{arg}' no longer exists on the real entry -- allowlist row is stale")
    assert stale == [], "stale _KNOWN_ASYMMETRIES rows:\n  " + "\n  ".join(stale)


@pytest.mark.light
def test_known_asymmetry_allowlist_is_empty():
    """Any signature omission is blocking port work, never a green allowlist entry,
    so the allowlist is empty."""
    assert _KNOWN_ASYMMETRIES == {}, (
        f"the signature-parity allowlist must be empty; port the arguments instead of allowlisting them: {_KNOWN_ASYMMETRIES}"
    )
