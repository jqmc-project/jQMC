"""Issue #781: no native Jastrow ratio route reads the jastrow_eval zone.

The precision policy (``jqmc/_precision.py``) assigns every Jastrow ratio
``exp(J(R') - J(R))`` to ``jastrow_ratio`` (float64 in both modes) and whole
Jastrow values to ``jastrow_eval`` (float32 in mixed). The native routes below must
not form the ratio from two whole Jastrow values in ``jastrow_eval`` (the
full-recompute ``coulomb_potential::jastrow_value``, the J3 slow fallback of the GFMC
fast mesh state ``MeshFastState::jas_ratios``, and the free periodic ECP strain kernel
``stress_pbc::grad_v_ecp_nonlocal_strain`` with whole DTN and J1/J2 values):

* reference LRDMC e_L / b_L (``_gfmc_lrdmc_ecp`` / ``_gfmc_lrdmc_ae`` with
  ``fast=False``: ``compute_ecp_nonlocal_mesh`` and
  ``compute_discretized_kinetic_elements``), the reference ECP all-pairs sum
  (``compute_ecp_nonlocal_parts_all_pairs``, ``geminal_inv = None``);
* the production GFMC projection, MCMC force / stress and GFMC force / stress of a
  periodic MO-wrapped J3 (no streaming J3 ratio state: J3 slow fallback and the
  ``ecp_nonlocal_force_pbc`` / ``ecp_nonlocal_stress_mesh_pbc`` full-recompute
  fallbacks), and the frozen debug stress oracle
  (``_gfmc_stress_ecp_pbc_withj(..., debug=True)``);
* the free periodic ECP strain ``grad_V_ecp_nonlocal_wrt_strain`` with DTN / J1 / J2.

Zone-isolation policy: every native zone float64 except ``jastrow_eval``, which
takes the active mode's dtype (float32 in a mixed run, so a full run is a
same-case consistency check). None of these outputs consumes a whole Jastrow
value (ratios are ``jastrow_ratio``, drift / Pulay / strain / continuum kinetic
``jastrow_grad_lap``), so actual and reference run the same float64 arithmetic
and must agree elementwise to reduction order. Actual and reference are built from
independently loaded copies of the same serialized full input under the full Python
policy; only the native policy differs. A route that forms the ratio from whole values
in ``jastrow_eval`` fails every case in mixed (e.g. reference LRDMC 3.7e-6, periodic MO-J3 GFMC interval 4.5e-6, MCMC
stress 1.1e-5, free strain 1.6e-8, all-pairs 1.1e-9).
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

import jax
import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import configure, is_mixed_precision_enabled, precision_policy_snapshot
from jqmc._setting import ATOL_REDUCTION_ORDER, RTOL_REDUCTION_ORDER
from tests.test_rs_kernel import test_batch_mesh_per_walker as per_walker
from tests.test_rs_kernel._native_precision_fidelity import full_source_hamiltonian_factory
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_gfmc_projection import _MESH_GP, _MESH_W, _cfg
from tests.test_rs_kernel.test_mcmc_observables import _geminal_inv, _walker_batch

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.with_chdir]

_ALAT = 0.3
_NN = 1
_THETA = 0.37
_RT = np.array([[np.cos(_THETA), -np.sin(_THETA), 0.0], [np.sin(_THETA), np.cos(_THETA), 0.0], [0.0, 0.0, 1.0]])


@contextlib.contextmanager
def _full_python_and_restored_native():
    """Run the body under the full Python policy; restore both backends' entry policies."""
    was_mixed = is_mixed_precision_enabled()
    native_entry = dict(jqmc_rs.native_precision_support())
    configure("full")
    try:
        yield
    finally:
        try:
            configure("mixed" if was_mixed else "full")
        finally:
            jqmc_rs.configure_native_precision(native_entry)


def _ravel(xs):
    return np.concatenate([np.ravel(np.asarray(x, float)) for x in xs])


def _lrdmc_reference(ham, policy):
    """Reference (full-recompute) LRDMC e_L / b_L, kinetic elements and ECP all-pairs sum."""
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    gd, jd = wf.geminal_data, wf.jastrow_data
    r_up, r_dn = (np.ascontiguousarray(x) for x in _cfg(ham, 3))
    out = {"r_up": r_up, "r_dn": r_dn}
    out["kinetic_elements"] = np.asarray(jqmc_rs._gfmc_discretized_kinetic_elements(gd, jd, r_up, r_dn, _ALAT, _RT), float)
    if bool(cp.ecp_flag):
        e, b, _ = jqmc_rs._gfmc_lrdmc_ecp(gd, jd, cp, r_up, r_dn, _ALAT, _RT, _MESH_W, _MESH_GP, _NN, False, False)
        out["lrdmc_e_L_b_L"] = np.array([e, b])
        if bool(ham.structure_data.pbc_flag):
            gp = _MESH_GP @ _RT
            out["ecp_all_pairs"] = np.atleast_1d(
                np.asarray(jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gd, jd, cp, r_up, r_dn, _MESH_W, gp, False, 0), float)
            )
    else:
        out["lrdmc_e_L_b_L"] = np.array(jqmc_rs._gfmc_lrdmc_ae(gd, jd, cp, r_up, r_dn, _ALAT, _RT, False)[:2], float)
    return out


def _gfmc_projection(ham, policy):
    """Production GFMC_n interval (fast mesh state) and the fast LRDMC e_L / b_L."""
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    gd, jd = wf.geminal_data, wf.jastrow_data
    r_up, r_dn = (np.ascontiguousarray(x) for x in _cfg(ham, 3))
    jqmc_rs.gfmc_route_counters_reset()
    ctx = jqmc_rs.build_obs_ctx(
        gd, jd, cp, False, bool(ham.structure_data.pbc_flag), False, want_force=False, want_gfmc=True, precision_policy=policy
    )
    g = ctx.run_gfmc_projection_n(
        r_up, r_dn, 0.30, -1.0, 2, 7, 11, np.zeros((0, 3, 3)), _MESH_W, _MESH_GP, random_mesh=False, dltmove=False, nn=_NN
    )
    slow = int(jqmc_rs.gfmc_route_counters()[12])  # J3 slow fallbacks (gfmc.rs append-only list)
    assert slow > 0, "the periodic MO-wrapped J3 must take the J3 slow fallback"
    e, b, _ = jqmc_rs._gfmc_lrdmc_ecp(gd, jd, cp, r_up, r_dn, _ALAT, _RT, _MESH_W, _MESH_GP, _NN, False, True)
    return {
        "r_up": r_up,
        "r_dn": r_dn,
        "gfmc_n_interval": np.array([float(g["w_L"]), float(g["e_L"]), float(g["V_diag"])]),
        "gfmc_n_r": np.concatenate([np.ravel(g["r_up"]), np.ravel(g["r_dn"])]),
        "fast_lrdmc_e_L_b_L": np.array([e, b]),
    }


def _mesh_force_stress(ham, policy):
    """MCMC force / stress batch, GFMC force (fast and debug) and stress (fast and debug oracle)."""
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    gd, jd = wf.geminal_data, wf.jastrow_data
    nw = per_walker._NW
    rb_up, rb_dn = (np.ascontiguousarray(x) for x in _walker_batch(ham, nw=nw, seed=0))
    rts = per_walker._distinct_rotations(nw)
    ginv = np.stack([_geminal_inv(ham, rb_up[w], rb_dn[w]) for w in range(nw)])
    ctx = jqmc_rs.build_obs_ctx(
        gd, jd, cp, False, True, True, want_force=True, want_gfmc=False, want_param_deriv=False, precision_policy=policy
    )
    gp_all = per_walker._GP @ rts
    out = {"r_up": rb_up, "r_dn": rb_dn}
    out["mcmc_force"] = _ravel(ctx.grad_e_L_pbc_batch(rb_up, rb_dn, per_walker._W, gp_all, ginv))
    out["mcmc_stress"] = np.ravel(np.asarray(ctx.grad_e_L_eps_pbc_batch(rb_up, rb_dn, per_walker._W, gp_all, ginv), float))
    r_up, r_dn = rb_up[0], rb_dn[0]
    for debug in (False, True):
        f = jqmc_rs._gfmc_grad_e_L_ecp_pbc(gd, jd, cp, r_up, r_dn, _ALAT, _RT, _MESH_W, _MESH_GP, _NN, False, debug)
        out[f"gfmc_force_debug{int(debug)}"] = _ravel(f)
        s = jqmc_rs._gfmc_stress_ecp_pbc_withj(gd, jd, cp, r_up, r_dn, _ALAT, _RT, _MESH_W, _MESH_GP, _NN, False, debug)
        out[f"gfmc_stress_debug{int(debug)}"] = np.ravel(np.asarray(s, float))
    return out


def _mesh_force_stress_mol_fallback(ham, policy):
    """Molecular GFMC ECP force / stress through the full-recompute Jastrow fallback.

    ``JQMC_RS_ECPF_SLOW`` (set by the test) disables the one-electron Jastrow mesh
    cache, so ``ecp_nonlocal_force_mol`` takes its ``jas_fallback`` branch, the route
    of a Jastrow leaf the cache does not cover.
    """
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    gd, jd = wf.geminal_data, wf.jastrow_data
    r_up, r_dn = (np.ascontiguousarray(x) for x in _cfg(ham, 3))
    f = jqmc_rs._gfmc_grad_e_L_ecp(gd, jd, cp, r_up, r_dn, _ALAT, _RT, _MESH_W, _MESH_GP, _NN, False)
    s = jqmc_rs._gfmc_stress_ecp_withj(gd, jd, cp, r_up, r_dn, _ALAT, _RT, _MESH_W, _MESH_GP, _NN, False)
    return {
        "r_up": r_up,
        "r_dn": r_dn,
        "gfmc_force_fallback": _ravel(f),
        "gfmc_stress_fallback": np.ravel(np.asarray(s, float)),
    }


def _free_strain(ham, policy):
    """Free periodic ECP non-local strain with the DTN / J1 / J2 wf ratio."""
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    r_up, r_dn = (np.ascontiguousarray(x) for x in _cfg(ham, 3))
    s = jqmc_rs.grad_V_ecp_nonlocal_wrt_strain(wf.geminal_data, cp, r_up, r_dn, _MESH_W, _MESH_GP @ _RT, wf.jastrow_data)
    return {"r_up": r_up, "r_dn": r_dn, "free_ecp_strain": np.ravel(np.asarray(s, float))}


_ROUTES = {
    "lrdmc_reference": _lrdmc_reference,
    "gfmc_projection": _gfmc_projection,
    "mesh_force_stress": _mesh_force_stress,
    "mesh_force_stress_mol_fallback": _mesh_force_stress_mol_fallback,
    "free_strain": _free_strain,
}
_ENV = {"mesh_force_stress_mol_fallback": {"JQMC_RS_ECPF_SLOW": "1"}}

_CASES = [
    pytest.param("mol_ecp_cart_jdtn", "lrdmc_reference", marks=pytest.mark.light, id="mol_ecp_cart_jdtn-lrdmc_reference"),
    pytest.param("mol_cart_sd_j123", "lrdmc_reference", marks=pytest.mark.heavy, id="mol_cart_sd_j123-lrdmc_reference"),
    pytest.param(
        "mol_ecp_sphe_j12",
        "mesh_force_stress_mol_fallback",
        marks=pytest.mark.heavy,
        id="mol_ecp_sphe_j12-mesh_force_stress_mol_fallback",
    ),
    pytest.param(
        "pbc_gamma_sphepgto_jdtn", "lrdmc_reference", marks=pytest.mark.heavy, id="pbc_gamma_sphepgto_jdtn-lrdmc_reference"
    ),
    pytest.param("pbc_gamma_sphepgto_jdtn", "free_strain", marks=pytest.mark.light, id="pbc_gamma_sphepgto_jdtn-free_strain"),
    pytest.param(
        "pbc_gamma_sphepgto_jdtn", "mesh_force_stress", marks=pytest.mark.heavy, id="pbc_gamma_sphepgto_jdtn-mesh_force_stress"
    ),
    pytest.param("pbc_composite_j123", "gfmc_projection", marks=pytest.mark.light, id="pbc_composite_j123-gfmc_projection"),
    pytest.param("pbc_composite_j123", "mesh_force_stress", marks=pytest.mark.heavy, id="pbc_composite_j123-mesh_force_stress"),
    pytest.param("pbc_composite_j123", "lrdmc_reference", marks=pytest.mark.heavy, id="pbc_composite_j123-lrdmc_reference"),
]


@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("pattern, route", _CASES)
def test_native_jastrow_ratio_routes_do_not_read_jastrow_eval(pattern, route, monkeypatch):
    """Native ratio routes with the mode's jastrow_eval dtype equal the float64 rebuild.

    Configurations: ``_cfg(ham, 3)`` (single-walker routes) or the
    ``_walker_batch(seed=0)`` walkers with the production per-walker rotations of
    ``test_batch_mesh_per_walker`` (MCMC batch), mesh rotation ``R_z(0.37)``, Nv = 12
    icosahedral mesh, NN = 1, alat = 0.3; the GFMC_n interval runs two sub-steps from
    key (7, 11) at tau = 0.30, E_ref = -1.0, tmove. The actual policy is all float64
    except ``jastrow_eval`` at the active mode's dtype; the reference is the all-float64
    native policy on an independently loaded copy of the same serialized full input.
    Every output must be finite, the reference live, and actual equal reference
    elementwise to reduction order (``ATOL/RTOL_REDUCTION_ORDER``).
    """
    mode_jastrow_eval = precision_policy_snapshot()["jastrow_eval"]
    evaluate = _ROUTES[route]
    for name, value in _ENV.get(route, {}).items():
        monkeypatch.setenv(name, value)
    with _full_python_and_restored_native():
        full_policy = precision_policy_snapshot()
        assert set(full_policy.values()) == {"float64"}
        actual_policy = {**full_policy, "jastrow_eval": mode_jastrow_eval}
        factory = full_source_hamiltonian_factory(lambda: rs_ham(pattern))
        jqmc_rs.configure_native_precision(full_policy)
        reference = evaluate(factory(), full_policy)
        jqmc_rs.configure_native_precision(actual_policy)
        actual = evaluate(factory(), actual_policy)
    np.testing.assert_array_equal(actual["r_up"], reference["r_up"])
    np.testing.assert_array_equal(actual["r_dn"], reference["r_dn"])
    keys = sorted(set(reference) - {"r_up", "r_dn"})
    assert keys and keys == sorted(set(actual) - {"r_up", "r_dn"})
    mismatches = []
    for key in keys:
        got, ref = np.asarray(actual[key]), np.asarray(reference[key])
        assert got.shape == ref.shape and got.size > 0, (key, got.shape, ref.shape)
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)), f"{pattern}/{route} {key}: non-finite"
        assert np.max(np.abs(ref)) > 1.0e-3, f"{pattern}/{route} {key}: inactive output"
        if not np.allclose(got, ref, atol=ATOL_REDUCTION_ORDER, rtol=RTOL_REDUCTION_ORDER):
            mismatches.append(f"{key}: max|diff| {np.max(np.abs(got - ref)):.3e} (max|ref| {np.max(np.abs(ref)):.3e})")
    # Every mismatching output is reported, one line per route output.
    assert not mismatches, f"{pattern}/{route} jastrow_eval={mode_jastrow_eval} vs float64 rebuild:\n" + "\n".join(mismatches)
