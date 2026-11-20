"""Issue #781: every XLA Jastrow ratio route stays in its own precision zones.

The precision policy (``jqmc/_precision.py``) assigns every Jastrow ratio
``exp(J(R') - J(R))`` to ``jastrow_ratio`` (float64 in both modes), whole Jastrow
values to ``jastrow_eval`` and gradients / Laplacians to ``jastrow_grad_lap``
(both float32 in mixed). Before #781:

* the full-recompute routes formed the ratio from two whole ``compute_Jastrow_part``
  values, i.e. in ``jastrow_eval``: the non-fast discretized kinetic mesh
  (``compute_discretized_kinetic_energy``) and its debug twin, the nearest-neighbour
  ECP kernels (``compute_ecp_non_local_parts_nearest_neighbors`` and
  ``..._holomorphic``, the latter the ``dV_nl/dc`` fallback of
  ``derivatives.assemble._dV_ecp_dc``), the all-pairs ECP kernel and both ECP debug
  kernels, the GFMC_t autograd-fallback ``_compute_V_elements_t``, the dltmove
  verification value ``_dltmove_delta``, the GFMC debug drivers, the up-electron-only
  MCMC kernel ``_update_electron_positions_only_up_electron`` (production on the XLA
  route whenever ``N_dn = 0``) and ``_MCMC_debug``;
* the GFMC ratio kernels read the J3 streaming grad/lap state, whose orbital values,
  ``j3_mat @ aos`` tables and row-sums were rounded and rank-one advanced in
  ``jastrow_grad_lap``;
* the Jastrow variational parameters (J1, J2, J3 ``j_matrix``, DTN independent
  coefficients) were stored in the ``jastrow_eval`` dtype (float32 in mixed) at
  construction, basis conversion, optimizer update and checkpoint load, so the
  float64 zones consumed rounded parameters.

Zone-isolation policy: the full XLA policy except one zone, which takes the active
mode's dtype (float32 in a mixed run; a full run is a same-case consistency check).
Actual and reference are built from independently loaded copies of the same
serialized full input; only the zone policy differs.
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

import contextlib
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc._precision as precision
import jqmc.jqmc_mcmc as jqmc_mcmc
from jqmc import determinant
from jqmc._precision import configure, get_dtype_np, is_mixed_precision_enabled, precision_policy_snapshot
from jqmc._setting import ATOL_REDUCTION_ORDER, RTOL_REDUCTION_ORDER
from jqmc.coulomb_potential import (
    _compute_ecp_non_local_parts_all_pairs_debug,
    _compute_ecp_non_local_parts_nearest_neighbors_debug,
    compute_ecp_non_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors,
    compute_ecp_non_local_parts_nearest_neighbors_holomorphic,
)
from jqmc.derivatives.ecp_force import _dltmove_delta
from jqmc.determinant import compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data, _apply_dtn_dtype_alignment_post_load
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_ratio_state,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    _compute_ratio_Jastrow_part_rank1_update,
    _compute_ratio_Jastrow_part_split_spin,
    _init_grads_laplacian_Jastrow_three_body_streaming_state,
)
from jqmc.jqmc_gfmc import _compute_V_elements_t, _GFMC_n_debug, _GFMC_t_debug
from jqmc.jqmc_mcmc import _MCMC_debug, _update_electron_positions_only_up_electron
from jqmc.wavefunction import (
    _compute_discretized_kinetic_energy_debug,
    compute_discretized_kinetic_energy,
    compute_discretized_kinetic_energy_fast_update,
)
from tests._precision_fidelity import mixed_vs_full
from tests.test_jqmc_kernel.test_gfmc_force_analytic import _build_ham_ecp, _build_ham_ecp_dtn
from tests.test_rs_kernel._native_precision_fidelity import full_source_hamiltonian_factory
from tests.test_rs_kernel._rs_fixtures import rs_ham

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.with_chdir]

_N2 = "N2_ecp_ccpvtz_cart.h5"
_ALAT = 0.3
_THETA = 0.37
_RT = np.array([[np.cos(_THETA), -np.sin(_THETA), 0.0], [np.sin(_THETA), np.cos(_THETA), 0.0], [0.0, 0.0, 1.0]])

_BUILDERS = {
    "n2_j123": lambda: _build_ham_ecp(_N2, "j123ao"),
    "n2_dtn": lambda: _build_ham_ecp_dtn(_N2),
    "li_gamma_j123": lambda: rs_ham("pbc_gamma_sphepgto_j123"),
}


@contextlib.contextmanager
def _full_python():
    """Run the body under the full XLA policy and restore the entry mode."""
    was_mixed = is_mixed_precision_enabled()
    configure("full")
    try:
        yield
    finally:
        configure("mixed" if was_mixed else "full")


@contextlib.contextmanager
def _xla_zone_policy(policy):
    """Install a complete zone policy for the XLA kernels (compiled executables do not key on it)."""
    saved = dict(precision._zone_dtypes)
    jax.clear_caches()
    precision._zone_dtypes.clear()
    precision._zone_dtypes.update(policy)
    try:
        yield
    finally:
        jax.clear_caches()
        precision._zone_dtypes.clear()
        precision._zone_dtypes.update(saved)


def _config(ham, seed):
    """Electrons drawn around the nuclei (molecular) or uniformly in the cell (periodic)."""
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    if bool(ham.structure_data.pbc_flag):
        cell = np.asarray(ham.structure_data.cell, float)
        return rng.random((nu, 3)) @ cell, rng.random((nd, 3)) @ cell
    pos = np.asarray(ham.structure_data.positions, float)
    r = pos[rng.integers(0, len(pos), nu + nd)] + rng.normal(scale=0.9, size=(nu + nd, 3))
    return r[:nu], r[nu:]


def _flat(*xs):
    return np.concatenate([np.ravel(np.real(np.asarray(x, dtype=np.complex128))) for x in xs])


def _route_discretized_kinetic(ham, r_up, r_dn):
    wf = ham.wavefunction_data
    rt = jnp.asarray(_RT)
    out = {"kinetic": _flat(compute_discretized_kinetic_energy(_ALAT, wf, jnp.asarray(r_up), jnp.asarray(r_dn), rt)[2])}
    if not bool(ham.structure_data.pbc_flag):
        # The debug twin uses the unrotated lattice and has no periodic wrap.
        out["kinetic_debug"] = _flat(_compute_discretized_kinetic_energy_debug(_ALAT, wf, r_up, r_dn)[2])
    return out


def _route_ecp_nearest_neighbors(ham, r_up, r_dn):
    cp, wf = ham.coulomb_potential_data, ham.wavefunction_data
    args = (cp, wf, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(_RT))
    out = {
        "ecp_nn": _flat(compute_ecp_non_local_parts_nearest_neighbors(*args)[2]),
        "ecp_nn_holomorphic": _flat(compute_ecp_non_local_parts_nearest_neighbors_holomorphic(*args)[2]),
    }
    if not bool(ham.structure_data.pbc_flag):
        out["ecp_nn_debug"] = _flat(_compute_ecp_non_local_parts_nearest_neighbors_debug(cp, wf, r_up, r_dn, _RT)[2])
    return out


def _route_ecp_all_pairs(ham, r_up, r_dn):
    cp, wf = ham.coulomb_potential_data, ham.wavefunction_data
    return {
        "ecp_all_pairs": _flat(
            compute_ecp_non_local_parts_all_pairs(cp, wf, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(_RT))[2]
        ),
        "ecp_all_pairs_debug": _flat(_compute_ecp_non_local_parts_all_pairs_debug(cp, wf, r_up, r_dn, _RT)[2]),
    }


def _route_gfmc_fallbacks(ham, r_up, r_dn):
    """GFMC_t autograd-fallback V elements (tmove, dltmove) and the dltmove verification value."""
    up, dn, rt = jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(_RT)
    out = {move: _flat(*_compute_V_elements_t(ham, up, dn, rt, move, _ALAT)) for move in ("tmove", "dltmove")}
    out["dltmove_delta"] = _flat(_dltmove_delta(ham, up, dn, rt))
    return out


def _route_gfmc_debug_drivers(ham, r_up, r_dn):
    """Three projections of the GFMC_t / GFMC_n debug drivers (one walker, fixed seed)."""
    out = {}
    common = dict(hamiltonian_data=ham, num_walkers=1, num_gfmc_collect_steps=1, mcmc_seed=3446, alat=_ALAT)
    common.update(random_discretized_mesh=True, non_local_move="tmove")
    for label, driver in (
        ("gfmc_t_debug", _GFMC_t_debug(tau=0.03, **common)),
        ("gfmc_n_debug", _GFMC_n_debug(num_mcmc_per_measurement=2, E_scf=-19.0, **common)),
    ):
        driver.run(num_mcmc_steps=3)
        out[label] = _flat(driver.e_L, driver.w_L)
    return out


_ROUTES = {
    "discretized_kinetic": _route_discretized_kinetic,
    "ecp_nearest_neighbors": _route_ecp_nearest_neighbors,
    "ecp_all_pairs": _route_ecp_all_pairs,
    "gfmc_fallbacks": _route_gfmc_fallbacks,
    "gfmc_debug_drivers": _route_gfmc_debug_drivers,
}

_WHOLE_VALUE_CASES = [
    pytest.param("n2_j123", "discretized_kinetic", marks=pytest.mark.light, id="n2_j123-discretized_kinetic"),
    pytest.param("n2_j123", "ecp_nearest_neighbors", marks=pytest.mark.light, id="n2_j123-ecp_nearest_neighbors"),
    pytest.param("n2_j123", "ecp_all_pairs", marks=pytest.mark.heavy, id="n2_j123-ecp_all_pairs"),
    pytest.param("n2_j123", "gfmc_fallbacks", marks=pytest.mark.heavy, id="n2_j123-gfmc_fallbacks"),
    pytest.param("n2_dtn", "gfmc_fallbacks", marks=pytest.mark.heavy, id="n2_dtn-gfmc_fallbacks"),
    pytest.param("n2_dtn", "ecp_nearest_neighbors", marks=pytest.mark.heavy, id="n2_dtn-ecp_nearest_neighbors"),
    pytest.param("n2_j123", "gfmc_debug_drivers", marks=pytest.mark.heavy, id="n2_j123-gfmc_debug_drivers"),
    pytest.param("li_gamma_j123", "discretized_kinetic", marks=pytest.mark.heavy, id="li_gamma_j123-discretized_kinetic"),
    pytest.param("li_gamma_j123", "ecp_nearest_neighbors", marks=pytest.mark.heavy, id="li_gamma_j123-ecp_nearest_neighbors"),
]


def _zone_isolated_pair(builder, zone, evaluate):
    """``(actual, reference)``: ``zone`` at the active mode's dtype vs the full policy."""
    mode_dtype = precision_policy_snapshot()[zone]
    with _full_python():
        full_policy = precision_policy_snapshot()
        assert set(full_policy.values()) == {"float64"}
        factory = full_source_hamiltonian_factory(builder)
        reference = evaluate(factory())
        with _xla_zone_policy({**full_policy, zone: mode_dtype}):
            actual = evaluate(factory())
    return actual, reference, mode_dtype


def _assert_reduction_order(actual, reference, label):
    """Every output finite, the reference live, actual == reference to reduction order (all mismatches listed)."""
    keys = sorted(set(reference) - {"r_up", "r_dn"})
    assert keys and keys == sorted(set(actual) - {"r_up", "r_dn"})
    mismatches = []
    for key in keys:
        got, ref = np.asarray(actual[key]), np.asarray(reference[key])
        assert got.shape == ref.shape and got.size > 0, (label, key, got.shape, ref.shape)
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)), f"{label} {key}: non-finite"
        assert np.max(np.abs(ref)) > 1.0e-3, f"{label} {key}: inactive output"
        if not np.allclose(got, ref, atol=ATOL_REDUCTION_ORDER, rtol=RTOL_REDUCTION_ORDER):
            mismatches.append(f"{key}: max|diff| {np.max(np.abs(got - ref)):.3e} (max|ref| {np.max(np.abs(ref)):.3e})")
    assert not mismatches, f"{label}:\n" + "\n".join(mismatches)


@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("fixture, route", _WHOLE_VALUE_CASES)
def test_xla_whole_value_jastrow_ratio_routes_do_not_read_jastrow_eval(fixture, route):
    """XLA routes that form ``exp(J(R') - J(R))`` from two whole values ignore ``jastrow_eval``.

    Fixtures: N2 ccECP (``N2_ecp_ccpvtz_cart.h5``) with exp J1/J2 and an AO-leaf J3
    (``n2_j123``) or with the reviewed DTN (``n2_dtn``); bcc Li ccECP at Gamma with J1/J2
    and a periodic pGTO J3 (``li_gamma_j123``). One ``_config(ham, 3)`` configuration,
    kinetic lattice / ECP mesh rotation ``R_z(0.37)``, alat = 0.3, default Nv / NN; the
    GFMC debug drivers run three projections of one walker (seed 3446, tau = 0.03 for
    GFMC_t, two moves per measurement and E_scf = -19 for GFMC_n). Actual policy: full
    except ``jastrow_eval`` at the active mode's dtype. None of these outputs consumes a
    whole Jastrow value, so actual equals the full-policy evaluation of an independently
    loaded copy of the same full input elementwise to reduction order.
    """
    builder = _BUILDERS[fixture]

    def evaluate(ham):
        r_up, r_dn = _config(ham, 3)
        return _ROUTES[route](ham, r_up, r_dn)

    actual, reference, mode_dtype = _zone_isolated_pair(builder, "jastrow_eval", evaluate)
    _assert_reduction_order(actual, reference, f"{fixture}/{route} jastrow_eval={mode_dtype}")


def _only_up_ham():
    """N atom all-electron Cartesian ccpvdz, exp J1/J2, three up electrons on the first MOs, no down electron."""
    from jqmc.trexio_wrapper import read_trexio_file
    from jqmc.wavefunction import Wavefunction_data

    stem = Path(__file__).resolve().parents[1] / "trexio_example_files" / "N_ae_ccpvdz_cart.h5"
    structure_data, _, mos_data, _, _, coulomb_potential_data = read_trexio_file(str(stem), store_tuple=True)
    sd = determinant.SD_data.build(num_electron_up=3, num_electron_dn=0, mos_up=mos_data, mos_dn=mos_data, time_reversal=False)
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.1, structure_data=structure_data, core_electrons=(0.0,)
        ),
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.9),
        jastrow_three_body_data=None,
        jastrow_dtn_data=None,
    )
    return Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=sd),
    )


def _spy_jastrow_values(monkeypatch, module):
    """Record the dtype of every whole Jastrow value ``module`` consumes (its module-level bindings)."""
    seen = []
    for name in ("compute_Jastrow_part", "compute_Jastrow_part_ratio_zone"):
        original = getattr(module, name, None)
        if original is None:
            continue

        def spy(*args, _original=original, **kwargs):
            value = _original(*args, **kwargs)
            seen.append(jnp.result_type(value))
            return value

        monkeypatch.setattr(module, name, spy)
    return seen


def _run_only_up(ham):
    rng = np.random.default_rng(3)
    r_up = jnp.asarray(rng.normal(scale=0.7, size=(3, 3)))
    r_dn = jnp.zeros((0, 3), dtype=jnp.float64)
    gd = ham.wavefunction_data.geminal_data
    geminal = jnp.asarray(compute_geminal_all_elements(gd, r_up, r_dn))
    out = _update_electron_positions_only_up_electron(
        r_up,
        r_dn,
        jax.random.PRNGKey(5),
        4,
        ham,
        1.2,
        1.0e-6,
        jnp.asarray(np.linalg.inv(np.asarray(geminal))),
        geminal,
        determinant._init_det_ratio_streaming_state(gd, r_up, r_dn),
        Jastrow_ratio_state(j3_state=None),
    )
    return {"accepted": int(out[0]), "r_up": np.asarray(out[2])}


def _run_mcmc_debug(ham):
    debug = _MCMC_debug(
        hamiltonian_data=ham,
        Dt=2.0,
        mcmc_seed=34356,
        epsilon_AS=1.0e-6,
        num_walkers=1,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        random_discretized_mesh=True,
    )
    debug.run(num_mcmc_steps=2)
    return {"e_L": np.asarray(debug.e_L, float)}


@pytest.mark.light
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("kernel", ["only_up_electron", "mcmc_debug"])
def test_xla_mcmc_whole_value_accept_ratio_consumes_ratio_zone_jastrow(kernel, monkeypatch):
    """The MCMC accept ratios built from two whole Jastrow values consume ``jastrow_ratio`` values.

    ``_update_electron_positions_only_up_electron`` (four proposals from PRNGKey(5), Dt = 1.2,
    on the all-electron N atom with exp J1/J2 and three up / no down electrons) and
    ``_MCMC_debug`` (two steps of one walker, seed 34356, on ``n2_j123``) form
    ``exp(J(R') - J(R))`` from whole values. A Metropolis decision almost never flips
    at a float32 rounding of that factor, so the check is at the zone boundary: under the
    full policy except ``jastrow_eval`` at the active mode's dtype, every whole Jastrow
    value these kernels consume must carry the ``jastrow_ratio`` dtype (float64), and the
    run must consume at least one. The run itself must be finite.
    """
    mode_dtype = precision_policy_snapshot()["jastrow_eval"]
    module = jqmc_mcmc
    with _full_python():
        full_policy = precision_policy_snapshot()
        ham = _only_up_ham() if kernel == "only_up_electron" else _BUILDERS["n2_j123"]()
        seen = _spy_jastrow_values(monkeypatch, module)
        with _xla_zone_policy({**full_policy, "jastrow_eval": mode_dtype}):
            result = _run_only_up(ham) if kernel == "only_up_electron" else _run_mcmc_debug(ham)
            ratio_dtype = np.dtype(get_dtype_np("jastrow_ratio"))
    assert seen, f"{kernel}: no whole Jastrow value consumed"
    assert all(np.dtype(d) == ratio_dtype for d in seen), (
        f"{kernel}: consumed Jastrow value dtypes {sorted(set(map(str, seen)))}"
    )
    for key, value in result.items():
        assert np.all(np.isfinite(np.asarray(value, float))), f"{kernel} {key}: non-finite"


def _route_j3_streaming(ham, r_up, r_dn):
    """GFMC kinetic mesh and ECP-style one-move ratios fed by the J3 grad/lap streaming state."""
    wf = ham.wavefunction_data
    jd = wf.jastrow_data
    up, dn, rt = jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(_RT)
    state = _init_grads_laplacian_Jastrow_three_body_streaming_state(jd.jastrow_three_body_data, up, dn)
    ainv = jnp.linalg.inv(compute_geminal_all_elements(wf.geminal_data, up, dn))
    out = {
        "kinetic_fast": _flat(compute_discretized_kinetic_energy_fast_update(_ALAT, wf, ainv, up, dn, rt, j3_state=state)[2])
    }
    rng = np.random.default_rng(781)
    mesh_up, mesh_dn = [], []
    for spin, r in ((0, r_up), (1, r_dn)):
        for i in range(len(r)):
            for _ in range(2):
                moved = np.array(r, copy=True)
                moved[i] += rng.normal(scale=0.4, size=3)
                mesh_up.append(moved if spin == 0 else r_up)
                mesh_dn.append(moved if spin == 1 else r_dn)
    mu, md = jnp.asarray(np.asarray(mesh_up)), jnp.asarray(np.asarray(mesh_dn))
    out["rank1"] = _flat(_compute_ratio_Jastrow_part_rank1_update(jd, up, dn, mu, md, j3_state=state))
    n_up = 2 * len(r_up)
    out["split_spin"] = _flat(_compute_ratio_Jastrow_part_split_spin(jd, up, dn, mu[:n_up], md[n_up:], j3_state=state))
    return out


@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize(
    "fixture",
    [pytest.param("n2_j123", marks=pytest.mark.light), pytest.param("li_gamma_j123", marks=pytest.mark.heavy)],
)
def test_xla_j3_streaming_ratio_tables_do_not_read_jastrow_grad_lap(fixture):
    """Ratios fed by the J3 grad/lap streaming state ignore the ``jastrow_grad_lap`` dtype.

    The GFMC drivers pass their J3 grad/lap streaming state
    (``_init_grads_laplacian_Jastrow_three_body_streaming_state``) to the ratio kernels.
    Its orbital values, ``j3_mat @ aos`` tables and row-sums are read by the
    ``jastrow_ratio`` kernels and must not carry the ``jastrow_grad_lap`` rounding. Same
    fixtures, configuration, rotation and alat as above; the one-move mesh moves every
    electron twice by seeded normal steps (``default_rng(781)``, sigma = 0.4 bohr). Actual
    policy: full except ``jastrow_grad_lap`` at the active mode's dtype. The discretized
    kinetic mesh (``compute_discretized_kinetic_energy_fast_update``), the rank-one ratio and
    the split-spin ratio with that state equal the full-policy evaluation of an independently
    loaded copy to reduction order.
    """

    def evaluate(ham):
        r_up, r_dn = _config(ham, 3)
        return _route_j3_streaming(ham, r_up, r_dn)

    actual, reference, mode_dtype = _zone_isolated_pair(_BUILDERS[fixture], "jastrow_grad_lap", evaluate)
    _assert_reduction_order(actual, reference, f"{fixture} jastrow_grad_lap={mode_dtype}")


def _parameter_leaves(jastrow_data):
    """Every variational Jastrow parameter leaf of ``jastrow_data`` (name -> array)."""
    leaves = {}
    if jastrow_data.jastrow_one_body_data is not None:
        leaves["j1"] = jastrow_data.jastrow_one_body_data.jastrow_1b_param
    if jastrow_data.jastrow_two_body_data is not None:
        leaves["j2"] = jastrow_data.jastrow_two_body_data.jastrow_2b_param
    if jastrow_data.jastrow_three_body_data is not None:
        leaves["j3"] = jastrow_data.jastrow_three_body_data.j_matrix
    if jastrow_data.jastrow_dtn_data is not None:
        dtn = jastrow_data.jastrow_dtn_data
        leaves.update(dtn_alpha=dtn.alpha_indep, dtn_beta=dtn.beta_indep, dtn_gamma=dtn.gamma_indep)
    return {k: np.asarray(v) for k, v in leaves.items()}


def _storage_case():
    """Parameter leaves of the reviewed N2 J1/J2/J3 and DTN Jastrows through every storage path."""
    from jqmc.wavefunction import Wavefunction_data

    j123_ham = _build_ham_ecp(_N2, "j123ao")
    dtn_ham = _build_ham_ecp_dtn(_N2)
    out = {}
    rng = np.random.default_rng(781)
    for ham in (j123_ham, dtn_ham):
        wf = ham.wavefunction_data
        jastrow_data = wf.jastrow_data
        out.update({f"build/{k}": v for k, v in _parameter_leaves(jastrow_data).items()})
        updated = jastrow_data
        for block in wf.get_variational_blocks(state_token=("test", 0)):
            if not block.name.startswith(("j1_param", "j2_param", "j3_matrix", "j3_poly")):
                continue
            values = np.asarray(block.values, np.float64)
            step = 1.0e-9 * (1.0 + np.abs(values)) * rng.choice([-1.0, 1.0], size=values.shape)
            updated = updated.apply_block_update(block.apply_update(np.ravel(step), 1.0))
        out.update({f"update/{k}": v for k, v in _parameter_leaves(updated).items()})
    j3 = j123_ham.wavefunction_data.jastrow_data.jastrow_three_body_data
    out["to_cartesian/j3"] = np.asarray(j3.to_cartesian().j_matrix)
    out["random_init/j3"] = np.asarray(
        Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=j3.orb_data, random_init=True, seed=7).j_matrix
    )
    dtn = dtn_ham.wavefunction_data.jastrow_data
    jd = dtn.jastrow_dtn_data
    legacy = jd.replace(
        alpha_indep=np.asarray(jd.alpha_indep, np.float32),
        beta_indep=np.asarray(jd.beta_indep, np.float32),
        gamma_indep=np.asarray(jd.gamma_indep, np.float32),
    )
    legacy_ham = dtn_ham.replace(
        wavefunction_data=Wavefunction_data(
            jastrow_data=dtn.replace(jastrow_dtn_data=legacy), geminal_data=dtn_ham.wavefunction_data.geminal_data
        )
    )
    with tempfile.TemporaryDirectory(prefix="jqmc_781_") as directory:
        path = Path(directory) / "dtn32.h5"
        legacy_ham.save_to_hdf5(path)
        loaded = Hamiltonian_data.load_from_hdf5(path)
    out.update({f"load/{k}": v for k, v in _parameter_leaves(loaded.wavefunction_data.jastrow_data).items()})
    out.update(
        {
            f"align/{k}": v
            for k, v in _parameter_leaves(
                _apply_dtn_dtype_alignment_post_load(legacy_ham).wavefunction_data.jastrow_data
            ).items()
        }
    )
    return out


@pytest.mark.light
def test_jastrow_parameter_storage_is_float64_in_every_mode():
    """Jastrow variational parameters are stored in float64 by construction, in full and mixed.

    The reviewed N2 J1/J2/AO-J3 and DTN Jastrows (``_build_ham_ecp(..., "j123ao")``,
    ``_build_ham_ecp_dtn``): the leaves after construction (``build``), after one optimizer
    update of every J1/J2/J3/DTN block by a seeded relative step of 1e-9
    (``apply_block_update``), after ``to_cartesian`` and a seeded ``random_init`` J3, and after
    an HDF5 round trip of a DTN checkpoint written with float32 coefficients
    (``load_from_hdf5`` and its post-load dtype alignment) must all be float64 under the
    active mode, and in mixed equal, bit for bit, the leaves of an independently rebuilt
    full-precision case. Float32 leaves in mixed would quantize away a 1e-9 relative update
    and every float64 consumer zone would read rounded parameters.
    """
    if is_mixed_precision_enabled():
        result, full = mixed_vs_full(lambda: None, lambda _: _storage_case())
    else:
        result = full = _storage_case()
    assert sorted(result) == sorted(full) and len(result) > 10
    for key in sorted(result):
        got = np.asarray(result[key])
        assert got.dtype == np.float64, f"{key}: stored as {got.dtype}"
        assert got.size and np.all(np.isfinite(got)), key
        np.testing.assert_array_equal(got, np.asarray(full[key]), err_msg=f"{key}: mixed storage differs from full")
