"""Long-chain carried-vs-fresh gate for the Rust LRDMC projection-interval state.

Inside one projection interval the native GFMC drivers keep ONE ``MeshFastState``
(real) or ``MeshFastStateC`` (complex twist) and advance it after every hop
(about 3000 hops per interval at production tau): the Sherman-Morrison ``G^-1``
chain, the carried determinant contraction rows ``winv`` (``DetWdotCache`` /
``DetWdotCacheC``, rank-one advanced with the same factors as ``G^-1``), the
kinetic and ECP DTN candidate tables (pair-corrected ``tabpip`` tables), the DTN
grad/lap cache and the candidate AO tables. This is the Rust twin of
``tests/test_jqmc_kernel/test_lrdmc_mesh_chain_consistency.py``.

``NativeObsCtx.debug_mesh_chain`` runs the PRODUCTION interval machinery with a
fixed (non-identity) kinetic and ECP rotation: every sub-step evaluates the
production ``interval_eval`` (kinetic mesh, ECP mesh, e_L), takes the production
categorical FN hop with a prescribed uniform and applies the production
``advance_after_hop``. At checkpoints it returns the carried state next to a
state built from scratch at the same configuration (``fresh``).

Two hop policies (``policy`` parameter):

* ``categorical``: the realistic production walk (``_N_HOPS`` categorical FN hops
  from a seeded uniform stream, which is longer than the 15-hop chains of the
  driver parity tests);
* ``adversarial``: the 1/rho amplification protocol of the XLA sibling (the
  kinetic candidate of smallest ``|element|`` above the sibling's ``_RHO_FLOOR``,
  from a different electron than the previous hop), which drives cond(G) to
  1e7-1e10. It exercises the per-hop refresh rule of
  ``advance_after_hop`` (the carried ``G^-1`` is screened every hop by the
  Sherman-Morrison probe residual and rebuilt from scratch, with the winv tables, when it exceeds
  ``SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD``). These cases run in full precision
  only: the independent mixed oracle carries fp32
  error above the mixed tolerance at that conditioning; mixed coverage of the
  precision-independent refresh logic comes from the categorical cases.

Every electron starts one lattice vector outside the home cell,
so every distance in the chain needs the minimum image. Cases (all periodic):

* ``si_pgto_real_jdtn``: committed Si diamond cell (8 atoms, 16 up / 16 dn, ccECP,
  real Gamma pGTO), J1 + J2 + DTN;
* ``h8_bspline_real_jdtn`` / ``h8_bspline_twist_jdtn``: committed 8-atom H cell,
  all-electron, 4 up / 4 dn, real Gamma and complex k = 1/4 B-spline, J1 + J2 + DTN;
* ``o_bspline_twist_jdtn``: committed O-lsda k = 1/4 save converted to a complex
  B-spline (4 up / 2 dn, ccECP), J2 + DTN.

The DTN f-term cutoff of the multi-atom cells (``L_f`` = 0.95 of half the
inscribed radius, the production default) is far below the cell, so most atoms
lie outside ``L_f`` of any electron: a neighbour-list restriction of the f-term
loops would skip them (asserted per checkpoint, so the cases are not vacuous).

Verdicts, each precision mode judged within itself:

* carried vs fresh, determinant / mesh quantities (``G^-1``, ``winv`` rows and
  bases, kinetic and ECP mesh elements, e_L, b_L): maximum-normalised (the metric
  of the XLA sibling) within ``get_tolerance_min(zones, "strict")``; on the real
  periodic pGTO case the documented periodic fast-interval floor
  ``get_tolerance_gfmc_fast_pbc`` (the carried moved-electron value column comes
  from the screened fast-cut folded kernel, the fresh one from the
  single-position kernel);
* carried ``winv`` vs ``winv`` rebuilt from the CARRIED ``(det_state, G^-1)``:
  the same pair (single-source property of the two chains);
* DTN candidate tables: candidate positions and candidate e-atom distances
  bitwise (both sides evaluate the same expression), log-ratios element-wise
  within ``get_tolerance("jastrow_ratio", "strict")``; DTN grad/lap cache
  element-wise within ``get_tolerance("jastrow_grad_lap_carry", "strict")``
  (the cache's fp64 carry zone, in both modes);
* Rust carried vs XLA at the same configuration of the same chain, at
  ``_XLA_ANCHORS``: kinetic mesh elements (``compute_discretized_kinetic_energy``),
  ECP mesh values (``compute_ecp_non_local_parts_nearest_neighbors``), e_L and
  b_L (``_compute_V_elements_t``), ``G^-1`` and the DTN grad/lap
  (``compute_grads_and_laplacian_Jastrow_DTN``), within the periodic GFMC
  fast-interval backend pair ``get_tolerance_gfmc_fast_pbc(zones)`` (for e_L plus
  ``cond_2(G) * eps * sum |determinant-ratio terms|`` on the absolute side); the
  fresh state is judged against the same oracle as the fairness control;
* refresh witness: ``jqmc_rs.route_predicate_counters()`` 10/11 count
  exactly one screen decision per hop; every adversarial chain refreshes at
  least once and the categorical chain keeps the carry on most hops; the Rust
  constant equals the drivers' ``SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD``.

The context is built with the active precision policy (as production and
``conftest.py`` configure ``jqmc_rs``), so the mixed run exercises mixed Rust.
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

import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import get_dtype_np, get_tolerance, get_tolerance_min, precision_policy_snapshot
from jqmc._setting import (
    RTOL_LRDMC_STREAMING_MESH_CHAIN,
    SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD,
    get_tolerance_gfmc_fast_pbc,
)
from jqmc.coulomb_potential import (
    NN_default,
    compute_ecp_non_local_parts_nearest_neighbors,
    icosahedron_sym_mesh_Nv12,
)
from jqmc.determinant import compute_geminal_all_elements
from jqmc.jastrow_factor import compute_grads_and_laplacian_Jastrow_DTN
from jqmc.jqmc_gfmc import _compute_V_elements_t
from jqmc.wavefunction import compute_discretized_kinetic_energy
from tests._test_data_cache import qe_example_files_root
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell, _make_dtn
from tests.test_jqmc_kernel.test_lrdmc_mesh_chain_consistency import _RHO_FLOOR
from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg

pytestmark = [pytest.mark.with_chdir]

_ALAT = 0.30
_N_HOPS = 1000
_CHECK_EVERY = 100
_XLA_ANCHORS = (0, _N_HOPS // 2, _N_HOPS)
_SEED = 20261006
# Start-configuration conditioning cap: an input-only fixture filter (the
# 8-atom H cell rarely draws cond(G) below the default cap of
# draw_well_conditioned_cfg); not a comparison budget.
_START_COND_MAX = 1.0e5
# Every electron starts this integer lattice translation away from its drawn
# in-cell position, so the chain runs on unwrapped coordinates.
_START_LATTICE_SHIFT = (1.0, -1.0, 1.0)

# Hop policy of the probe -> its floor argument. "categorical" is the realistic
# production walk; "adversarial" is the 1/rho amplification protocol of the XLA
# sibling (the kinetic candidate of smallest |element| above the sibling's
# _RHO_FLOOR, from a different electron than the previous hop), which drives
# cond(G) to 1e7-1e10 and exercises the per-hop inverse refresh.
_POLICIES = {"categorical": 0.0, "adversarial": _RHO_FLOOR}
# The adversarial chains are judged in full precision only.
_POLICY_PARAMS = [
    pytest.param(
        "adversarial",
        marks=pytest.mark.skip_mixed_precision(
            reason=(
                "Adversarial cond(G) 1e7-1e10 chains: the mixed oracle carries fp32 error above the mixed tolerance; "
                "the parameter-free refresh logic is precision-independent and is judged in full; mixed coverage via "
                "the categorical chains"
            )
        ),
    ),
    "categorical",
]

_MESH_W = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
_MESH_GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)

_HERE = os.path.dirname(os.path.abspath(__file__))
_O_K025_SAVE = os.path.join(qe_example_files_root(), "O-lsda_k025", "out", "o_k025.save")

# Zones crossed by each compared quantity.
_ZONES_INV = ("ao_eval", "det_ratio")
_ZONES_MESH = ("ao_eval", "det_ratio", "jastrow_ratio", "wf_kinetic")
_ZONES_EL = ("ao_eval", "wf_kinetic", "coulomb", "local_energy", "det_ratio", "jastrow_eval", "jastrow_grad_lap", "ewald")
_ZONES_DTN_GRAD = ("jastrow_grad_lap",)
_DET_KEYS = ("ginv", "winv_up", "winv_up_base", "winv_dn", "winv_dn_base")
_MESH_KEYS = ("kinetic_elements", "ecp_values", "e_L", "e_L_meas", "b_L")
_DTN_GRAD_KEYS = ("dtn_grad_up", "dtn_grad_dn", "dtn_lap_up", "dtn_lap_dn")


def _o_lsda_bspline_twist_jdtn(tmp_dir):
    from jqmc._twist_average import load_twist
    from jqmc.jastrow_factor import Jastrow_data
    from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input

    out = os.path.join(tmp_dir, "o_lsda_k025_bspline_dtn.h5")
    convert_qe_to_twist_average_input(
        _O_K025_SAVE, out, upf_paths={"O": os.path.join(_O_K025_SAVE, "O.ccECP.upf")}, kind="bspline"
    )
    ham = load_twist(out, 0)
    jd = ham.wavefunction_data.jastrow_data
    jd = Jastrow_data(
        jastrow_one_body_data=jd.jastrow_one_body_data,
        jastrow_two_body_data=jd.jastrow_two_body_data,
        jastrow_three_body_data=None,
        jastrow_dtn_data=_make_dtn(ham.structure_data, ham.coulomb_potential_data),
    )
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))


# case -> (builder(tmp_dir), complex ctx, carried-vs-fresh family, multi-atom DTN cell)
_CASES = {
    "si_pgto_real_jdtn": (lambda _d: _build_ham_for_cell("silicon.h5", True, {"dtn": True}, "SD"), False, "fast_pbc", True),
    "h8_bspline_real_jdtn": (
        lambda _d: _build_ham_for_cell("QEPROJ:bspline:H_gamma:AE", True, {"dtn": True}, "SD"),
        False,
        "strict",
        True,
    ),
    "h8_bspline_twist_jdtn": (
        lambda _d: _build_ham_for_cell("QEPROJ:bspline:H_k025:AE", True, {"dtn": True}, "SD"),
        True,
        "strict",
        True,
    ),
    "o_bspline_twist_jdtn": (_o_lsda_bspline_twist_jdtn, True, "strict", False),
}


def _rotation(rng):
    a, b, c = rng.uniform(-np.pi, np.pi, 3)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0.0, np.sin(b)], [0.0, 1.0, 0.0], [-np.sin(b), 0.0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0.0], [np.sin(c), np.cos(c), 0.0], [0.0, 0.0, 1.0]])
    return np.ascontiguousarray((rz @ ry @ rx).T)


_CHAIN_CACHE: dict = {}


def _chain(case, policy, mode, tmp_dir):
    """Run the probe once per (case, policy, precision mode) in this process."""
    key = (case, policy, mode)
    if key not in _CHAIN_CACHE:
        _CHAIN_CACHE.clear()
        _CHAIN_CACHE[key] = _run_chain(case, policy, tmp_dir)
    return _CHAIN_CACHE[key]


def _run_chain(case, policy, tmp_dir):
    builder, complex_ctx, _family, _multi = _CASES[case]
    ham = builder(tmp_dir)
    wf = ham.wavefunction_data
    ctx = jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        complex_ctx,
        True,
        False,
        want_force=False,
        want_gfmc=True,
        # The active precision policy (as production and conftest configure
        # jqmc_rs): without it the ctx defaults to full and a mixed run would
        # silently exercise fp64 Rust.
        precision_policy=precision_policy_snapshot(),
    )
    r_up, r_dn = draw_well_conditioned_cfg(ham, seed=_SEED % 997, cond_max=_START_COND_MAX)
    shift = np.asarray(_START_LATTICE_SHIFT) @ np.asarray(ham.structure_data.cell, dtype=np.float64)
    r_up = np.ascontiguousarray(r_up + shift)
    r_dn = np.ascontiguousarray(r_dn + shift)
    rng = np.random.default_rng(_SEED)
    kin_rt = _rotation(rng)
    ecp_rt = _rotation(rng)
    uniforms = rng.random(_N_HOPS)
    checkpoints = sorted(set(range(0, _N_HOPS + 1, _CHECK_EVERY)) | set(_XLA_ANCHORS))
    # Refresh witness: the route-predicate counters of the per-hop
    # interval-inverse screen over exactly this chain (10 = carried G^-1 kept,
    # 11 = G^-1 rebuilt from scratch).
    jqmc_rs.gfmc_route_counters_reset()
    rows = ctx.debug_mesh_chain(
        r_up,
        r_dn,
        _ALAT,
        kin_rt,
        ecp_rt,
        _MESH_W,
        _MESH_GP,
        NN_default,
        False,
        policy,
        _POLICIES[policy],
        list(uniforms),
        _N_HOPS,
        checkpoints,
    )
    assert [row["hop"] for row in rows] == checkpoints
    predicate = jqmc_rs.route_predicate_counters()
    screen = {"carries": int(predicate[10]), "refreshes": int(predicate[11])}
    return ham, kin_rt, ecp_rt, rows, screen


def _mode():
    from jqmc._precision import is_mixed_precision_enabled

    return "mixed" if is_mixed_precision_enabled() else "full"


@pytest.fixture
def _case_tmp(tmp_path_factory):
    return str(tmp_path_factory.mktemp("lrdmc_mesh_chain_rust"))


def _rows(case, policy, tmp_dir):
    return _chain(case, policy, _mode(), tmp_dir)


def _assert_scaled_close(got, ref, atol, rtol, label):
    """Maximum-normalised agreement: ``max|got - ref| <= atol + rtol * max|ref|``."""
    got = np.asarray(got)
    ref = np.asarray(ref)
    assert got.shape == ref.shape, f"{label}: shape {got.shape} != {ref.shape}"
    assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)), f"{label}: non-finite"
    if ref.size == 0:
        return
    divergence = float(np.max(np.abs(got - ref)))
    scale = float(np.max(np.abs(ref)))
    assert divergence <= atol + rtol * scale, f"{label}: max |carried - ref| = {divergence:.3e} at scale {scale:.3e}"


def _family_tolerance(family, zones):
    if family == "fast_pbc":
        return get_tolerance_gfmc_fast_pbc(zones)
    return get_tolerance_min(zones, "strict")


def _min_image_distances(ham, points):
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    atoms = np.asarray(ham.structure_data.positions, dtype=np.float64)
    d = points[:, None, :] - atoms[None, :, :]
    frac = d @ np.linalg.inv(cell)
    d = (frac - np.round(frac)) @ cell
    return np.linalg.norm(d, axis=-1), np.linalg.norm(points[:, None, :] - atoms[None, :, :], axis=-1)


@pytest.mark.heavy
@pytest.mark.parametrize("policy", _POLICY_PARAMS)
@pytest.mark.parametrize("case", sorted(_CASES))
def test_rust_lrdmc_chain_inverse_winv_and_mesh_match_fresh(case, policy, _case_tmp):
    """Carried G^-1, winv rows and mesh elements equal a fresh state along a 1000-hop production chain."""
    ham, _kin_rt, _ecp_rt, rows, screen = _rows(case, policy, _case_tmp)
    _builder, complex_ctx, family, _multi = _CASES[case]
    inv_tol = _family_tolerance(family, _ZONES_INV)
    mesh_tol = _family_tolerance(family, _ZONES_MESH)
    el_tol = _family_tolerance(family, _ZONES_EL)
    has_ecp = bool(ham.coulomb_potential_data.ecp_flag)
    for row in rows:
        carried, fresh = row["carried"], row["fresh"]
        for key in _DET_KEYS:
            assert key in carried and key in fresh, f"{case} hop {row['hop']}: {key} missing"
            assert np.iscomplexobj(carried[key]) == complex_ctx
            _assert_scaled_close(carried[key], fresh[key], *inv_tol, f"{case} hop {row['hop']}: {key} carried vs fresh")
            if key != "ginv":
                _assert_scaled_close(
                    carried[key],
                    carried[key.replace("winv_", "winv_rebuilt_")],
                    *inv_tol,
                    f"{case} hop {row['hop']}: {key} carried vs rebuilt from the carried G^-1",
                )
        assert carried["kinetic_elements"].size == 6 * sum(len(r) for r in (row["r_up"], row["r_dn"]))
        assert (carried["ecp_values"].size > 0) == has_ecp, f"{case}: ECP mesh liveness"
        np.testing.assert_array_equal(carried["ecp_positions"], fresh["ecp_positions"])
        np.testing.assert_array_equal(carried["ecp_electron"], fresh["ecp_electron"])
        # An all-electron cell has no ECP mesh (liveness asserted above): its
        # empty ECP arrays carry no comparison, so they are not handed to the
        # comparator (which would return without comparing anything).
        for key in _MESH_KEYS if has_ecp else tuple(key for key in _MESH_KEYS if key != "ecp_values"):
            tol = el_tol if key.startswith(("e_L", "b_L")) else mesh_tol
            _assert_scaled_close(carried[key], fresh[key], *tol, f"{case} hop {row['hop']}: {key} carried vs fresh")
    # Refresh witness. The Rust screen uses the drivers' constant; it
    # runs on every hop exactly once; the adversarial chain must refresh, and
    # the realistic categorical chain keeps the Sherman-Morrison carry on most
    # hops (the refresh is the exception, not the route).
    assert jqmc_rs.sm_inverse_refresh_residual_threshold() == SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD
    assert screen["carries"] + screen["refreshes"] == _N_HOPS, f"{case} {policy}: per-hop screen count {screen}"
    if policy == "adversarial":
        assert screen["refreshes"] > 0, f"{case}: the adversarial chain never refreshed G^-1 ({screen})"
    else:
        assert screen["refreshes"] < screen["carries"], f"{case}: the categorical chain refreshed on most hops ({screen})"


@pytest.mark.heavy
@pytest.mark.parametrize("policy", _POLICY_PARAMS)
@pytest.mark.parametrize("case", sorted(_CASES))
def test_rust_lrdmc_chain_dtn_tables_match_fresh(case, policy, _case_tmp):
    """Carried kinetic / ECP DTN candidate tables and the DTN grad/lap cache equal a fresh build."""
    ham, _kin_rt, _ecp_rt, rows, _screen = _rows(case, policy, _case_tmp)
    _builder, _complex_ctx, _family, multi_atom = _CASES[case]
    ratio_atol, ratio_rtol = get_tolerance("jastrow_ratio", "strict")
    # The DTN grad/lap cache (seed, per-hop delta, accumulator) is in the fp64
    # ``jastrow_grad_lap_carry`` zone in both modes (#679): an fp32 seed would
    # carry its rounding over the whole interval.
    grad_atol, grad_rtol = get_tolerance("jastrow_grad_lap_carry", "strict")
    dtn = ham.wavefunction_data.jastrow_data.jastrow_dtn_data
    l_f = float(np.max(np.asarray(dtn.L_f, dtype=np.float64)))
    tables = ("dtn_kin", "dtn_ecp") if ham.coulomb_potential_data.ecp_flag else ("dtn_kin",)
    for row in rows:
        carried, fresh = row["carried"], row["fresh"]
        label = f"{case} hop {row['hop']}"
        for table in tables:
            for key in (f"{table}_candidate_pos", f"{table}_cand_r_ei", f"{table}_log_delta"):
                assert key in carried and key in fresh, f"{label}: {key} missing"
            np.testing.assert_array_equal(carried[f"{table}_candidate_pos"], fresh[f"{table}_candidate_pos"], err_msg=label)
            np.testing.assert_array_equal(carried[f"{table}_cand_r_ei"], fresh[f"{table}_cand_r_ei"], err_msg=label)
            got, ref = carried[f"{table}_log_delta"], fresh[f"{table}_log_delta"]
            assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)), f"{label}: {table} non-finite"
            np.testing.assert_allclose(got, ref, atol=ratio_atol, rtol=ratio_rtol, err_msg=f"{label}: {table} log ratios")
        for key in _DTN_GRAD_KEYS:
            assert key in carried and key in fresh, f"{label}: {key} missing"
            got, ref = carried[key], fresh[key]
            assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)), f"{label}: {key} non-finite"
            np.testing.assert_allclose(got, ref, atol=grad_atol, rtol=grad_rtol, err_msg=f"{label}: {key}")
        # Non-vacuity of the f-term neighbour-list regime: the f-term is live
        # (some electron within L_f of an atom), and on the multi-atom cells
        # every electron has atoms beyond L_f; the unwrapped coordinates put
        # every electron beyond L_f of every atom by the plain distance.
        points = np.concatenate([row["r_up"], row["r_dn"]], axis=0)
        d_min, d_plain = _min_image_distances(ham, points)
        assert np.any(d_min < l_f), f"{label}: no electron within L_f of an atom"
        assert np.all(d_plain > l_f), f"{label}: an electron is within L_f of an atom by the plain distance"
        if multi_atom:
            assert np.all(np.sum(d_min > l_f, axis=1) >= 1), f"{label}: an electron has no atom beyond L_f"


@pytest.mark.heavy
@pytest.mark.parametrize("policy", _POLICY_PARAMS)
@pytest.mark.parametrize("case", sorted(_CASES))
def test_rust_lrdmc_chain_matches_xla_anchors(case, policy, _case_tmp):
    """Carried Rust mesh / inverse / DTN state equals XLA at configurations of the same chain."""
    ham, kin_rt, ecp_rt, rows, _screen = _rows(case, policy, _case_tmp)
    _builder, complex_ctx, _family, _multi = _CASES[case]
    wf = ham.wavefunction_data
    cd = ham.coulomb_potential_data
    anchors = [row for row in rows if row["hop"] in _XLA_ANCHORS]
    assert [row["hop"] for row in anchors] == list(_XLA_ANCHORS)
    # Conditioning screen of the XLA sibling (test_lrdmc_mesh_chain_consistency.py):
    # an anchor with cond_2(G) * eps >= RTOL_LRDMC_STREAMING_MESH_CHAIN is outside
    # the domain where the determinant-derived quantities (mesh elements, ECP
    # values, e_L / b_L, G^-1) of two correct evaluations agree to the backend
    # pair: the independent oracle itself moves by that class (Si adversarial,
    # cond 5.7e10: the XLA e_L at the same configuration differs by 1.0 out of
    # 1.4e5 between jax 0.7.2 and 0.10.2, while the carried state is bitwise the
    # fresh one). eps is that of the zone the compared inverse is formed in
    # (det_ratio). The DTN grad/lap does not depend on G and is judged at every
    # anchor.
    inverse_eps = float(np.finfo(get_dtype_np("det_ratio")).eps)
    judged_hops = []
    for row in anchors:
        r_up, r_dn = row["r_up"], row["r_dn"]
        xla_dtn = compute_grads_and_laplacian_Jastrow_DTN(wf.jastrow_data.jastrow_dtn_data, r_up, r_dn)
        for snapshot in ("carried", "fresh"):
            for key, ref in zip(_DTN_GRAD_KEYS, xla_dtn):
                _assert_scaled_close(
                    row[snapshot][key],
                    np.asarray(ref),
                    *get_tolerance_gfmc_fast_pbc(_ZONES_DTN_GRAD),
                    f"{case} hop {row['hop']} {snapshot}: {key} Rust vs XLA",
                )
        g = np.asarray(compute_geminal_all_elements(wf.geminal_data, r_up, r_dn))
        assert np.iscomplexobj(g) == complex_ctx
        cond_times_eps = float(np.linalg.cond(g, p=2) * inverse_eps)
        assert np.isfinite(cond_times_eps), f"{case} hop {row['hop']}: cond_2(G) * eps is not finite"
        if cond_times_eps >= RTOL_LRDMC_STREAMING_MESH_CHAIN:
            continue
        judged_hops.append(row["hop"])
        # Independent XLA references, evaluated from scratch at the anchor
        # configuration (no Rust code, no carried state).
        _, _, elements = compute_discretized_kinetic_energy(_ALAT, wf, r_up, r_dn, kin_rt)
        v_nl_fn = None
        if cd.ecp_flag:
            _, _, v_nl, _ = compute_ecp_non_local_parts_nearest_neighbors(cd, wf, r_up, r_dn, ecp_rt)
            # The interval carries the fixed-node off-diagonal part min(V, 0)
            # of every quadrature candidate (the positive part folds into S').
            v_nl_fn = np.minimum(np.asarray(v_nl).ravel().real, 0.0)
        v_diag, v_nondiag, _ = _compute_V_elements_t(ham, r_up, r_dn, kin_rt, "tmove", _ALAT, ecp_rt)
        el_tol = get_tolerance_gfmc_fast_pbc(_ZONES_EL)
        # e_L is a sum of determinant-ratio terms (kinetic mesh elements and,
        # with an ECP, the non-local mesh values) that cancel to a much smaller
        # total: H8 real adversarial, hop 500: sum |K| = 1.7e4 against
        # |e_L| = 10.5. Each term carries the relative error cond_2(G) * eps of
        # the inverse it is formed from, so two correct evaluations of e_L may
        # differ by cond_2(G) * eps * sum |terms| (first-order forward error);
        # that is added to the absolute side of the backend pair (issue #679).
        # At the well-conditioned anchors it is below 1e-9 and the pair is
        # unchanged.
        ratio_terms = np.sum(np.abs(np.asarray(elements)))
        if cd.ecp_flag:
            ratio_terms += np.sum(np.abs(np.asarray(v_nl)))
        e_l_tol = (el_tol[0] + cond_times_eps * float(ratio_terms), el_tol[1])
        # The carried production state is the verdict; the fresh state built at
        # the same configuration is the fairness control of the same oracle
        # (what any correctly re-anchored state holds).
        for snapshot in ("carried", "fresh"):
            state = row[snapshot]
            label = f"{case} hop {row['hop']} {snapshot}"
            _assert_scaled_close(
                state["kinetic_elements"],
                np.asarray(elements).ravel(),
                *get_tolerance_gfmc_fast_pbc(_ZONES_MESH),
                f"{label}: kinetic elements Rust vs XLA",
            )
            if v_nl_fn is not None:
                _assert_scaled_close(
                    np.real(state["ecp_values"]),
                    v_nl_fn,
                    *get_tolerance_gfmc_fast_pbc(_ZONES_MESH),
                    f"{label}: ECP mesh values Rust vs XLA",
                )
            _assert_scaled_close(
                np.real(state["e_L"]), np.atleast_1d(np.real(v_diag + v_nondiag)), *e_l_tol, f"{label}: e_L Rust vs XLA"
            )
            _assert_scaled_close(np.real(state["b_L"]), np.atleast_1d(np.real(v_nondiag)), *el_tol, f"{label}: b_L Rust vs XLA")
            _assert_scaled_close(
                state["ginv"], np.linalg.inv(g), *get_tolerance_gfmc_fast_pbc(_ZONES_INV), f"{label}: G^-1 Rust vs XLA"
            )
    # Non-vacuity: the start is judged, and so is at least one anchor along the chain.
    assert judged_hops and judged_hops[0] == _XLA_ANCHORS[0], f"{case} {policy}: the start anchor was screened out"
    assert any(hop > _XLA_ANCHORS[0] for hop in judged_hops), (
        f"{case} {policy}: every anchor along the chain was screened out by cond_2(G) * eps (judged {judged_hops})"
    )
