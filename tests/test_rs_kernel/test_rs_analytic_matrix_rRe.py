"""FOUR-AXIS matrix mirror: the r / R / eps sibling of the
c-axis ``test_rs_analytic_matrix.py``, walked over the SAME generated fixture covering array.

``test_rs_analytic_matrix.py`` asserts Rust == XLA-analytic on the ``c`` (parameter) axis over
the 600-node ``AO x Jastrow x ECP x BC x geminal`` covering array. The SAME generated
matrix is asserted for ALL of ``(quantity: ln_Psi/e_L) x (axis: c/r/R/eps)``.
The r (electron-coordinate), R (nuclear / force), and legacy eps axes had only
HAND-CURATED coverage (``test_vmc_forces`` / ``test_vmc_stress`` / ``test_forces*`` / ``test_stress*``
on a fixed set of ``rs_ham`` fixtures) -- they never walked the covering array, so the
``AO x Jastrow x ECP x BC x geminal`` INTERACTIONS on those axes (notably AGPn and both AO- and
MO-representation AGP, which the hand-curated force/stress suites never exercise) went unswept.
This module closes that: it walks ``axis in (r, R, eps) x quantity in (ln_Psi, e_L)`` per cell.

Reference (oracle) = the SAME public production analytic path the c-axis module uses, one axis at
a time: ``jqmc.derivatives.compute_grad_{ln_Psi,e_L}(ham, r_up, r_dn[, RT], {opt_<axis>: True},
analytic=True)[axis]`` -- the ``r`` value is an ``(dr_up, dr_dn)`` pair, ``R`` a per-atom array,
``eps`` a 3x3 affine-response tensor. DUT = the SERVED VMC CONTINUOUS force/affine-response Rust kernels (the same
``vmc_grad_e_L_{ae,ecp,pbc,pbc_complex}`` / ``vmc_grad_ln_psi{,_pbc,_pbc_complex}`` /
``vmc_grad_*_eps_pbc{,_complex}`` kernels whose exact call signatures the ``test_vmc_forces`` /
``test_vmc_stress`` suites establish -- reused here, NOT re-implemented). The MCMC HF force is
``dE_L/dR`` + ``dE_L/dr`` and the Pulay force is ``dln|Psi|/d{R,r}``; the R and r halves ride the
SAME native call (which returns ``(dR, dr_up, dr_dn)``), so the R-axis node reads ``dR`` and the
r-axis node reads ``(dr_up, dr_dn)``. eps rides the strain kernels.

Both the oracle and the native real-path kernels are threaded the SAME sweep SM ``geminal_inv``
(``jqmc_mcmc._geminal_inv_batched``) so the comparison is exact-inverse-vs-exact-inverse (the
complex-twist native kernels rebuild the complex inverse fresh, matching the complex e_L path --
so their oracle still threads the complex inverse; ``test_vmc_forces`` does the same).

Cell universe + builders are IMPORTED (never re-implemented) from the XLA matrix module -- the
IDENTICAL ``_generate_matrix`` / ``_build_ham_for_cell`` / ``_resolve`` / ``_split_cell`` /
``_cell_id`` the c-axis module shares, so every ham flows the EXACT committed fixture.

COVERAGE POLICY: the stored axis token ``eps`` has boundary-dependent semantics: periodic
cells mean physical cell strain, while molecular cells mean the explicitly non-observable
``molecular_affine`` coordinate deformation. Both are Rust/XLA implementation capabilities,
so every generated cell evaluates the XLA oracle and invokes the native kernel. There is no
N/A subtraction. A missing entry point, refused leaf, shape mismatch, or numerical mismatch
fails; there is no skip/xfail/allowlist path. The two known port
gaps exposed this way are:

  * MO-wrapped J3 (``_J3_KIND[ao] == "mo"`` -- composite / plane-wave / B-spline, real OR complex)
    force/stress: the J3-leaf force/stress kernel needs an atom-centered index the MO-wrapped flat /
    composite J3 orbital lacks (measured: ``AOs_{bspline,plane_wave}_data`` has no
    ``_nucleus_index_j3`` / ``_num_ao_kernel``; ``MOs_data`` / ``AOs_composite_data`` mismatch on
    ``num_ao`` / ``aos_data``). The SAME MO-wrapped-J3 exclusion the c-axis module documents in
    ``_J3_OK``. (Single-leaf GTO / pGTO J3 -- ``_J3_KIND in {pgto, complex}`` -- IS served.)
  * dense-AO-AGP (``gem == "AGP"``) on a COMPOSITE leaf: the dense-AGP force/stress is ported only
    for the molecular GTO leaves (measured: the composite geminal's ``AOs_composite_data`` orb has no
    ``mo_coefficients`` / ``aos_data`` the kernel reads). Mirrors the c-axis ``_det_okay`` support:
    ``AGP_data`` is served only on ``bc == mol`` GTO leaves. (dense-AGP molecular GTO -- the only
    other place ``gem == AGP`` is constructible -- IS served.) MO-representation AGP
    (``gem == "AGP_MO"``) follows the ordinary MO orbital-ingestion path while retaining
    the dense-lambda AGP parameterization.

Tolerance has no strict/medium selector. Molecular rows use the dedicated
XLA--Rust consistency policy selected by full/mixed mode. Periodic/twist
force and stress derivatives retain the separately documented propagated-error
floor ``ATOL_FLOOR_MEDIUM`` /
    ``RTOL_FLOOR_MEDIUM`` (2e-5) -- the derivative of the periodic e_L AMPLIFIES the periodic-kinetic
    fp floor (``jqmc/_setting.py``; the SAME constant
    ``test_vmc_forces`` / ``test_vmc_stress`` use for exactly these kernels). Measured worst-case over all served periodic/twist rows =
    1.66e-6 (plane-wave DTN dE_L/dr), well inside; a wiring bug is O(0.1-1), still a strong guard.

A cell/axis whose native-vs-oracle comparison diverges beyond that floor FAILS per element.
The allowlist mappings are required to remain empty. An XLA oracle ``NotImplementedError``
also fails, because otherwise the Rust leg has no reference.

Bridge + jit-assemble caches are cleared per test (the inner-jit LRU is unbounded; walking dozens of
hams in one worker exhausts LLVM mmap). Molecular rows are ``light``; periodic/twist rows ``heavy``
(the suite convention; ``_generate_matrix`` tags each cell).
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

import ast
import pathlib

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import jqmc_rs

from jqmc import derivatives
from jqmc.derivatives import bridge
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.jqmc_mcmc import _geminal_inv_batched, _NV_BASE_MESH, Nv_default

from tests.test_jqmc_kernel._generated_matrix_base import (
    _build_ham_for_cell,
    _cell_id,
    _generate_matrix,
    _resolve,
    _split_cell,
)
from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg
from tests.test_rs_kernel._gdd_rust_adapters import record_rust_matrix_fragment

pytestmark = pytest.mark.with_chdir

_AXES = ("r", "R", "eps")
_QUANTITIES = ("ln_Psi", "e_L")

# ECP non-local mesh, matching the ported VMC force/stress rs tests.
_MESH = _NV_BASE_MESH[Nv_default]
_W = np.asarray(_MESH.weights, dtype=np.float64)
_GP = np.asarray(_MESH.grid_points, dtype=np.float64)

# Tolerance zones (the union the VMC force/stress kernels cross -- identical to
# ``test_vmc_forces`` / ``test_vmc_stress``).
_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap")


def _tol(bc: str) -> tuple[float, float]:
    """Force/stress parity uses the explicit mixed medium envelope everywhere."""
    return get_tolerance_xla_rust_consistency(_ZONES, level="medium")


# --------------------------------------------------------------------------- #
# Config + helpers (identical style to the c-axis module + the ported VMC force rs tests).
# --------------------------------------------------------------------------- #
def _config(ham, seed: int = 1):
    return draw_well_conditioned_cfg(ham, seed=seed, scale=1.0, return_metadata=True)


def _ao_leaf(orb):
    return orb.aos_data if type(orb).__name__ == "MOs_data" else orb


def _is_complex(ham) -> bool:
    # A composite wrapper carries no ``complex`` in its own class name -- its complexity lives in
    # the children (complex-pGTO + complex-PW at a twist), so recurse; a flat leaf reports directly.
    leaf = _ao_leaf(ham.wavefunction_data.geminal_data.orb_data_up_spin)
    if hasattr(leaf, "children"):
        return any("complex" in type(c).__name__ for c in leaf.children)
    return "complex" in type(leaf).__name__


def _geminv(ham, r_up, r_dn, cx: bool):
    _, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(r_up)[None], jnp.asarray(r_dn)[None])
    return np.asarray(ginv[0], dtype=(np.complex128 if cx else np.float64))


# --------------------------------------------------------------------------- #
# (cell-id, axis, quantity) -> one-line reason, for a native-vs-oracle compare that RUNS but
# genuinely DIVERGES. Entries here (and ONLY here) may fail the per-element compare;
# everything else asserts.
#
# EMPTY -- an empirical sweep (492 nodes over the whole covering array) measured
# every SERVED path inside its documented floor (worst 1.66e-6 vs the 2e-5 periodic floor), so
# the r/R/eps kernels have NO known divergence. Same discipline as the c-axis module: a genuinely
# known one must be frozen here with a MEASURED one-line mechanism, and the staleness gate below
# fails the session if a frozen entry stops diverging.
_DIVERGENCE_ALLOWLIST: dict[tuple[str, str, str], str] = {}

_ALLOWLIST_WALKED: set[tuple[str, str, str]] = set()
_ALLOWLIST_FIRED: set[tuple[str, str, str]] = set()

# XLA-analytic ORACLE gap: a NotImplementedError from the reference leg fails, never a skip.
# EMPTY -- measured: the oracle raised on NO matrix cell, including the molecular-affine cells.
# An entry may be added ONLY
# with a measured one-line mechanism; otherwise this is a trap for a future oracle regression.
_XLA_ORACLE_GAP_ALLOWLIST: dict[tuple[str, str, str], str] = {}
_XLA_ORACLE_GAPS: list[tuple[str, str, str, str]] = []


@pytest.mark.light
def test_observable_derivative_matrix_has_no_green_escape_hatch():
    """The production r/R/eps matrix may neither skip/xfail cells nor freeze gaps."""
    tree = ast.parse(pathlib.Path(__file__).read_text())
    escapes = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
        and node.func.attr in {"skip", "xfail"}
    ]
    assert not escapes, f"observable-derivative parity cells can become green at lines {escapes}"
    assert not _DIVERGENCE_ALLOWLIST, "numerical parity gaps must fail, not be allowlisted"
    assert not _XLA_ORACLE_GAP_ALLOWLIST, "XLA oracle gaps must fail, not be allowlisted"


def _record_xla_gap(key: tuple[str, str, str], why: str) -> None:
    _XLA_ORACLE_GAPS.append((*key, why))


@pytest.fixture(scope="session", autouse=True)
def _divergence_allowlist_staleness_gate():
    """FAIL the session if a frozen divergence was walked but never actually diverged (the
    dead-guard failure mode). Per-worker by construction (a key is walked and fires, or not,
    inside one worker), so no cross-process aggregation is needed under xdist.
    """
    yield
    stale = sorted(_ALLOWLIST_WALKED - _ALLOWLIST_FIRED)
    if stale:
        detail = "\n  ".join(f"{cid} {ax}/{qty} :: {_DIVERGENCE_ALLOWLIST.get((cid, ax, qty), '?')}" for cid, ax, qty in stale)
        pytest.fail(
            "STALE _DIVERGENCE_ALLOWLIST entries -- these were compared and did NOT diverge, so the "
            "freeze is no longer earning its place. Delete them (the defect is fixed):\n  " + detail
        )


@pytest.fixture(scope="session", autouse=True)
def _xla_gap_inventory_report():
    yield
    if _XLA_ORACLE_GAPS:
        lines = [
            "",
            "=" * 78,
            "XLA-analytic r/R/eps ORACLE gap inventory -- (cell, axis, quantity) raising",
            "NotImplementedError from the jqmc analytic reference (this xdist worker):",
            "-" * 78,
        ]
        for cid, ax, qty, why in sorted(_XLA_ORACLE_GAPS):
            lines.append(f"  GAP: {cid:44s} {ax:4s} {qty:7s} :: {why}")
        lines.append("=" * 78)
        print("\n".join(lines))


_CURRENT: dict = {}


def _allowlisted() -> str | None:
    return _DIVERGENCE_ALLOWLIST.get((_CURRENT.get("cell_id"), _CURRENT.get("axis"), _CURRENT.get("quantity")))


def _assert_close(pairs, atol: float, rtol: float):
    """``pairs`` = list of (native, oracle, label): assert native == XLA-analytic PER ELEMENT.

    Divergences FAIL (``assert_allclose``, ``atol + rtol*|b|`` per element -- the suite standard).
    A genuinely-known one must be frozen in ``_DIVERGENCE_ALLOWLIST`` (empty). Same mechanics as
    the c-axis module's ``_assert_close``.
    """
    for native, xla, label in pairs:
        a = np.asarray(native)
        b = np.asarray(xla)
        assert a.shape == b.shape, f"{label}: native shape {a.shape} vs XLA shape {b.shape}"
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=f"native vs XLA-analytic {label}")
    record_rust_matrix_fragment(
        cell=_CURRENT["cell"],
        quantity=_CURRENT["quantity"],
        axis=_CURRENT["axis"],
        component_id=_CURRENT["axis"],
        pytest_node_id="tests/test_rs_kernel/test_rs_analytic_matrix_rRe.py::test_rs_analytic_matrix_rRe",
        ham=_CURRENT["ham"],
        r_up=_CURRENT["r_up"],
        r_dn=_CURRENT["r_dn"],
        compared_pairs=tuple((native, xla) for native, xla, _label in pairs),
        zones=_ZONES,
        level="medium",
        atol=atol,
        rtol=rtol,
        generation_attempts=_CURRENT["generation_attempts"],
    )


# --------------------------------------------------------------------------- #
# XLA-analytic oracle for one physical axis: the SAME public production analytic path the c-axis
# module uses (``derivatives.compute_grad_*(..., analytic=True)``), one axis per call.
# --------------------------------------------------------------------------- #
_AXIS_FLAG = {"r": "opt_r", "R": "opt_R", "eps": "opt_eps"}


def _xla_axis(ham, r_up, r_dn, RT, axis, quantity, ginv):
    flags = {_AXIS_FLAG[axis]: True}
    if quantity == "ln_Psi":
        return derivatives.compute_grad_ln_Psi(
            ham, jnp.asarray(r_up), jnp.asarray(r_dn), flags, analytic=True, geminal_inv=jnp.asarray(ginv)
        )[axis]
    return derivatives.compute_grad_e_L(
        ham, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(RT), flags, analytic=True, geminal_inv=jnp.asarray(ginv)
    )[axis]


# --------------------------------------------------------------------------- #
# Native DUT: the served VMC continuous force/stress kernel for (bc, axis, quantity), reusing the
# exact call signatures ``test_vmc_forces`` / ``test_vmc_stress`` establish. The r and R halves
# ride ONE native call returning ``(dR, dr_up, dr_dn)``; eps rides the strain kernels.
# --------------------------------------------------------------------------- #
def _native_axis(ham, r_up, r_dn, RT, axis, quantity, ginv, cx, ecp, bc):
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cp = ham.coulomb_potential_data
    if bc == "mol":
        if axis == "eps":
            # This is NOT physical stress: it is the derivative under the molecular
            # coordinate deformation r,R -> (I+eps)r,(I+eps)R. Keep it as an
            # implementation-parity capability under the upstream storage token ``eps``.
            if quantity == "ln_Psi":
                return jqmc_rs.grad_lnPsi_wrt_strain(gd, jd, r_up, r_dn)

            # Compose existing analytical components without a numerical or XLA fallback.
            # ``withJ`` owns T(with J)+Vbare, ``dtn`` additionally owns the DTN
            # kinetic virial, and ``ecp`` owns T(det)+Vbare+V_ECP. Subtracting the
            # jnone base avoids double-counting when ECP is combined with Jastrow/DTN.
            base = jqmc_rs.grad_eL_wrt_strain(gd, cp, r_up, r_dn)
            has_dtn = getattr(jd, "jastrow_dtn_data", None) is not None
            has_jastrow = any(
                getattr(jd, name, None) is not None
                for name in (
                    "jastrow_one_body_data",
                    "jastrow_two_body_data",
                    "jastrow_three_body_data",
                    "jastrow_dtn_data",
                )
            )
            if has_dtn:
                native = jqmc_rs.grad_eL_wrt_strain_dtn(gd, jd, cp, r_up, r_dn)
            elif has_jastrow:
                native = jqmc_rs.grad_eL_wrt_strain_withJ(gd, jd, cp, r_up, r_dn)
            else:
                native = base
            if ecp:
                native = native + jqmc_rs.grad_eL_wrt_strain_ecp(gd, jd, cp, r_up, r_dn, _W, _GP) - base
            return native
        if quantity == "ln_Psi":
            nR, nu, nd = jqmc_rs.vmc_grad_ln_psi(gd, jd, r_up, r_dn, ginv)
        elif ecp:
            nR, nu, nd = jqmc_rs.vmc_grad_e_L_ecp(gd, jd, cp, r_up, r_dn, _W, _GP, ginv)
        else:
            nR, nu, nd = jqmc_rs.vmc_grad_e_L_ae(gd, jd, cp, r_up, r_dn, ginv)
        return nR if axis == "R" else (nu, nd)
    if bc == "pbc_gamma":
        if axis == "eps":
            if quantity == "ln_Psi":
                return jqmc_rs.vmc_grad_ln_psi_eps_pbc(gd, jd, r_up, r_dn, ginv)
            return jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, r_up, r_dn, _W, _GP @ RT, ginv)
        if quantity == "ln_Psi":
            nR, nu, nd = jqmc_rs.vmc_grad_ln_psi_pbc(gd, jd, r_up, r_dn, ginv)
        else:
            nR, nu, nd = jqmc_rs.vmc_grad_e_L_pbc(gd, jd, cp, r_up, r_dn, _W, _GP @ RT, ginv)
        return nR if axis == "R" else (nu, nd)
    # twist (complex): native kernels rebuild the complex inverse fresh (no ginv arg).
    if axis == "eps":
        if quantity == "ln_Psi":
            return jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(gd, jd, r_up, r_dn)
        return jqmc_rs.vmc_grad_e_L_eps_pbc_complex(gd, jd, cp, r_up, r_dn, _W, _GP @ RT)
    if quantity == "ln_Psi":
        nR, nu, nd = jqmc_rs.vmc_grad_ln_psi_pbc_complex(gd, jd, r_up, r_dn)
    else:
        nR, nu, nd = jqmc_rs.vmc_grad_e_L_pbc_complex(gd, jd, cp, r_up, r_dn, _W, _GP @ RT, None)
    return nR if axis == "R" else (nu, nd)


# --------------------------------------------------------------------------- #
# Parametrization: per (cell, heavy) x axis x quantity.
# --------------------------------------------------------------------------- #
def _cell_params():
    params = []
    for cell, heavy in _generate_matrix():
        mark = pytest.mark.heavy if heavy else pytest.mark.light
        cid = _cell_id(cell)
        _ao, _jas, _ecp, bc, _gem = _split_cell(cell)
        for axis in _AXES:
            for q in _QUANTITIES:
                params.append(pytest.param(tuple(cell), axis, q, marks=mark, id=f"{cid}__{axis}__{q}"))
    return params


_PARAMS = _cell_params()


@pytest.fixture(autouse=True)
def _clear_caches():
    """Per-test teardown: drop the UNBOUNDED bridge inner-jit LRU + the public-entry
    ``_jitted_assemble`` LRU + the XLA compiled programs (the c-axis module does the same).
    """
    yield
    bridge.cache_clear()
    derivatives._jitted_assemble.cache_clear()
    jax.clear_caches()


@pytest.mark.parametrize("cell, axis, quantity", _PARAMS)
def test_rs_analytic_matrix_rRe(cell, axis, quantity):
    _CURRENT.clear()
    _CURRENT.update(cell=tuple(cell), cell_id=_cell_id(cell), axis=axis, quantity=quantity)
    _ao, _jas, _ecp_tok, bc, _gem = _split_cell(cell)

    build = _resolve(cell)
    assert build is not None, f"cell {_cell_id(cell)} has no committed builder (all matrix cells should)"
    stem, with_jastrow, spec, gem_tok = build
    ham = _build_ham_for_cell(stem, with_jastrow, spec, gem_tok)
    r_up, r_dn, generation = _config(ham)
    _CURRENT.update(ham=ham, r_up=r_up, r_dn=r_dn, generation_attempts=generation["attempts"])

    cx = _is_complex(ham)
    ecp = bool(getattr(ham.coulomb_potential_data, "ecp_flag", False))
    ginv = _geminv(ham, r_up, r_dn, cx)
    RT = np.eye(3)
    atol, rtol = _tol(bc)

    # XLA-analytic oracle. A NotImplementedError from the reference is recorded and
    # failed, never a green skip -- only a measured, frozen entry in _XLA_ORACLE_GAP_ALLOWLIST
    # (empty) may still skip.
    try:
        xla = _xla_axis(ham, r_up, r_dn, RT, axis, quantity, ginv)
    except NotImplementedError as e:
        why = str(e).splitlines()[0]
        key = (_CURRENT["cell_id"], axis, quantity)
        _record_xla_gap(key, why)
        pytest.fail(
            f"XLA-ORACLE GAP {key[0]} {axis}/{quantity}: the jqmc XLA-analytic reference raised "
            f"NotImplementedError -- {why}. This is the parity chain's reference leg; without it the "
            "Rust kernel is compared against nothing. Implement it; skip/xfail/allowlist escapes are forbidden."
        )

    native = _native_axis(ham, r_up, r_dn, RT, axis, quantity, ginv, cx, ecp, bc)

    if axis == "r":
        nu, nd = native
        oru, ord_ = xla
        pairs = [(nu, oru, f"{_CURRENT['cell_id']}/{quantity}/dr_up"), (nd, ord_, f"{_CURRENT['cell_id']}/{quantity}/dr_dn")]
    else:
        pairs = [(native, xla, f"{_CURRENT['cell_id']}/{quantity}/d{axis}")]
    _assert_close(pairs, atol, rtol)


@pytest.mark.light
def test_divergence_allowlist_is_covered_by_the_matrix():
    """Every frozen divergence / oracle-gap key must name a (cell, axis, quantity) the matrix
    ACTUALLY walks -- otherwise a typo silently disarms the freeze AND the assertion. With both
    allowlists empty this is a guard on future additions (mirrors the c-axis module's guard).
    """
    walked = {(_cell_id(p.values[0]), p.values[1], p.values[2]) for p in _PARAMS}
    unknown = sorted(k for k in _DIVERGENCE_ALLOWLIST if k not in walked)
    assert not unknown, f"_DIVERGENCE_ALLOWLIST keys not present in the matrix: {unknown}"
    unknown_gap = sorted(k for k in _XLA_ORACLE_GAP_ALLOWLIST if k not in walked)
    assert not unknown_gap, f"_XLA_ORACLE_GAP_ALLOWLIST keys not present in the matrix: {unknown_gap}"


_MAIN_GEOMETRY_REGIME_IDS = (
    "ln_Psi|regime|R01",
    "e_L|regime|R01",
    "ln_Psi|regime|R02",
    "e_L|regime|R02",
    "e_L|regime|R04",
    "e_L|regime|R08",
    "ln_Psi|regime|R10",
    "e_L|regime|R10",
)


@pytest.mark.heavy
@pytest.mark.parametrize("requirement_id", _MAIN_GEOMETRY_REGIME_IDS)
def test_rs_numerical_regime_matches_main_autograd(requirement_id, monkeypatch):
    """Run main's R01/R02/R04/R08/R10 preconditions with Rust as the DUT.

    The upstream evidence function still builds and validates the ordinary,
    ill-conditioned, complex-ratio, rotated-mesh, and additive-channel scenarios.
    Only its analytical half is replaced; its autograd half remains independent.
    """
    from tests.test_jqmc_kernel import test_analytic_coverage_evidence as evidence
    from tests.test_rs_kernel._gdd_rust_adapters import mirror_upstream_receipt

    upstream_pair = evidence._compute_axis_pair
    upstream_assert = evidence._assert_tree_allclose
    native_calls = 0
    pending_bcs = []
    captured_receipts = []
    upstream_case_receipt = evidence.case_receipt

    def capture_case_receipt(**kwargs):
        receipt = upstream_case_receipt(**kwargs)
        captured_receipts.append(receipt)
        return receipt

    def rust_pair(ham, family, axis, r_up, r_dn, RT=None, *, geminal_inv=None):
        nonlocal native_calls
        _analytical, automatic = upstream_pair(
            ham,
            family,
            axis,
            r_up,
            r_dn,
            RT,
            geminal_inv=geminal_inv,
        )
        cx = _is_complex(ham)
        if bool(ham.structure_data.pbc_flag):
            bc = "twist" if cx else "pbc_gamma"
        else:
            bc = "mol"
        rt = np.eye(3) if RT is None else np.asarray(RT)
        ginv = (
            _geminv(ham, np.asarray(r_up), np.asarray(r_dn), cx)
            if geminal_inv is None
            else np.asarray(geminal_inv, dtype=(np.complex128 if cx else np.float64))
        )
        native = _native_axis(
            ham,
            np.asarray(r_up),
            np.asarray(r_dn),
            rt,
            axis,
            "ln_Psi" if family == "ln_Psi" else "e_L",
            ginv,
            cx,
            bool(ham.coulomb_potential_data.ecp_flag),
            bc,
        )
        native_calls += 1
        pending_bcs.append(bc)
        return native, automatic

    def rust_assert(analytical, automatic, label):
        """Return the compared (analytical, automatic) array pairs.

        The upstream helper returns the pair tuple that the GDD receipt layer
        consumes (``observe_allclose``); a replacement that returns ``None``
        makes receipt creation raise, so this monkeypatch MUST honour the same
        contract on both branches.
        """
        assert pending_bcs, f"{label}: comparison ran without a preceding Rust derivative call"
        bc = pending_bcs.pop(0)
        if bc == "mol":
            return upstream_assert(analytical, automatic, label)
        leaves_a = jax.tree_util.tree_leaves(analytical)
        leaves_b = jax.tree_util.tree_leaves(automatic)
        assert len(leaves_a) == len(leaves_b), f"{label}: Rust/autograd tree structures differ"
        atol, rtol = _tol(bc)
        pairs = []
        for index, (leaf_a, leaf_b) in enumerate(zip(leaves_a, leaves_b, strict=True)):
            array_a = np.asarray(leaf_a)
            array_b = np.asarray(leaf_b)
            assert np.all(np.isfinite(array_a)), f"{label} Rust leaf {index} is not finite"
            assert np.all(np.isfinite(array_b)), f"{label} autograd leaf {index} is not finite"
            np.testing.assert_allclose(
                array_a,
                array_b,
                atol=atol,
                rtol=rtol,
                err_msg=f"{label} leaf {index} ({bc} Rust route-parity floor)",
            )
            pairs.append((array_a, array_b))
        return tuple(pairs)

    monkeypatch.setattr(evidence, "_compute_axis_pair", rust_pair)
    monkeypatch.setattr(evidence, "_assert_tree_allclose", rust_assert)
    monkeypatch.setattr(evidence, "case_receipt", capture_case_receipt)
    evidence.test_numerical_regime_evidence(requirement_id)
    assert native_calls > 0, f"{requirement_id}: regime row completed without a Rust derivative call"
    assert not pending_bcs, f"{requirement_id}: {len(pending_bcs)} Rust result(s) were never compared"
    assert len(captured_receipts) == 1
    mirror_upstream_receipt(
        captured_receipts[0],
        component_id="regime",
        pytest_node_id=("tests/test_rs_kernel/test_rs_analytic_matrix_rRe.py::test_rs_numerical_regime_matches_main_autograd"),
    )
