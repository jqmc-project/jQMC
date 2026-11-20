"""The native radial ECP gradient uses the production nearest-projector set."""

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs

from jqmc._precision import get_tolerance_min, get_tolerance_xla_rust_consistency, precision_policy_snapshot
from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors, icosahedron_sym_mesh_Nv12
from jqmc.derivatives.assemble import _dV_ecp_dc
from jqmc.jastrow_factor import Jastrow_data
from tests._ecp_high_l import assert_fixture_has_live_high_l_channel, high_l_hamiltonian
from tests.test_jqmc_kernel._generated_matrix_base import _make_j3
from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]
_ZONES = ("coulomb", "ao_eval", "mo_eval", "det_eval", "jastrow_eval")


def _single_centre_operator(coulomb, atom):
    """Keep one actual ECP operator, leaving orbitals/electrons/structure fixed.

    With one nonlocal centre, NN=1 includes that centre for every electron.
    The sum of the two one-centre operators is therefore the all-pairs
    counterfactual, including the far projectors that production NN=1 excludes.
    This is an operator decomposition, not a changed electronic fixture.
    """
    keep = [i for i, index in enumerate(coulomb.nucleus_index) if int(index) == atom]
    fields = ("nucleus_index", "ang_moms", "exponents", "coefficients", "powers")
    values = {name: tuple(getattr(coulomb, name)[i] for i in keep) for name in fields}
    values["num_ecps"] = len(keep)
    values["max_ang_mom_plus_1"] = tuple(l if i == atom else 0 for i, l in enumerate(coulomb.max_ang_mom_plus_1))
    result = coulomb.replace(**values)
    result.sanity_check()
    assert set(np.asarray(result._nucleus_index_non_local_part).tolist()) == {atom}
    return result


def _radial_leaf(gradient, spin, field):
    orbital = getattr(gradient, spin)
    aos = orbital.aos_data if hasattr(orbital, "aos_data") else orbital
    return np.asarray(getattr(aos.radial, field))


def _case(seed, spread):
    ham = high_l_hamiltonian("Cl2_ecp_ccpvtz_cart.h5")
    wf = ham.wavefunction_data.replace(jastrow_data=Jastrow_data(None, None, None))
    ham = ham.replace(wavefunction_data=wf)
    ham.sanity_check()
    assert_fixture_has_live_high_l_channel(ham)
    geminal = wf.geminal_data
    positions = np.asarray(ham.structure_data.positions)
    assert len(positions) == 2
    mid = 0.5 * (positions[0] + positions[1])
    rng = np.random.default_rng(seed)
    up = mid + rng.normal(scale=spread, size=(geminal.num_electron_up, 3))
    dn = mid + rng.normal(scale=spread, size=(geminal.num_electron_dn, 3))
    nearest = np.argmin(np.linalg.norm(np.concatenate([up, dn])[:, None] - positions[None], axis=2), axis=1)
    assert set(nearest.tolist()) == {0, 1}
    return ham, up, dn


@pytest.mark.parametrize("seed,spread", [(0, 1.2), (1, 1.8), (2, 1.2)])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_native_radial_gradient_preserves_nearest_centre_selection_effect(seed, spread):
    """Retain every reported configuration, including the ill-conditioned seed1.

    Compare the derivative EFFECT of NN selection, native NN1 minus the
    independent all-pairs counterfactual, with the same XLA analytical effect.
    This tests the filter itself without certifying whole-gradient backend
    accuracy at a near-singular geminal. Full/Mixed use unchanged per-element
    parity budgets; neither seeds nor individual radial elements are screened.
    Existing total-gradient parity tests remain unchanged.
    """
    ham, up, dn = _case(seed, spread)
    wf = ham.wavefunction_data
    geminal, coulomb = wf.geminal_data, ham.coulomb_potential_data
    operators = [coulomb, *(_single_centre_operator(coulomb, atom) for atom in (0, 1))]
    weights = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    grid = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    native = [jqmc_rs.grad_eL_ecp_wrt_radial(geminal, wf.jastrow_data, cp, up, dn, weights, grid) for cp in operators]
    analytical = jax.jit(_dV_ecp_dc)
    reference = [
        analytical(ham.replace(coulomb_potential_data=cp), jnp.asarray(up), jnp.asarray(dn), jnp.eye(3)).geminal_data
        for cp in operators
    ]

    def value(cp, nn):
        return np.asarray(compute_ecp_non_local_parts_nearest_neighbors(cp, wf, up, dn, jnp.eye(3), NN=nn)[3])

    parts = [value(cp, 1) for cp in operators[1:]]
    assert all(np.isfinite(part) and part != 0 for part in parts)
    all_pairs_value = value(coulomb, 2)
    part_sum = sum(parts)
    value_atol, value_rtol = get_tolerance_min(_ZONES, "medium")
    np.testing.assert_allclose(all_pairs_value, part_sum, atol=value_atol, rtol=value_rtol)

    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, "medium")
    live = False
    for spin in ("orb_data_up_spin", "orb_data_dn_spin"):
        for index, field in enumerate(("exponents", "coefficients")):
            actual = (
                np.asarray(native[0][spin][index]) - np.asarray(native[1][spin][index]) - np.asarray(native[2][spin][index])
            )
            expected = (
                _radial_leaf(reference[0], spin, field)
                - _radial_leaf(reference[1], spin, field)
                - _radial_leaf(reference[2], spin, field)
            )
            assert actual.size and actual.shape == expected.shape
            assert np.isfinite(actual).all() and np.isfinite(expected).all()
            live |= bool(np.any(np.abs(expected) > atol + rtol * np.abs(expected)))
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=f"{spin} {field}: NN selection effect")
    assert live, "nearest-only and all-pairs derivative operators must be distinguishable"
    jax.clear_caches()


@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_native_ecp_j3_radial_matches_nearest_projector_gradient(monkeypatch):
    """Use both live Cl centres with the production NN=1 mesh and an active J3."""
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    ham = high_l_hamiltonian()
    assert_fixture_has_live_high_l_channel(ham)
    wf = ham.wavefunction_data
    geminal, coulomb = wf.geminal_data, ham.coulomb_potential_data
    j3 = _make_j3(geminal.orb_data_up_spin.aos_data)
    jastrow = wf.jastrow_data.replace(jastrow_three_body_data=j3)
    ham = ham.replace(wavefunction_data=wf.replace(jastrow_data=jastrow))
    ham.sanity_check()

    up, dn = draw_well_conditioned_cfg(ham, seed=2, scale=1.0, cond_max=100.0)
    positions = np.asarray(ham.structure_data.positions)
    distances = np.linalg.norm(np.concatenate([up, dn])[:, None] - positions[None, :], axis=2)
    assert set(np.argmin(distances, axis=1)) == {0, 1}

    policy = precision_policy_snapshot()
    ctx = jqmc_rs.build_obs_ctx(
        geminal,
        jastrow,
        coulomb,
        False,
        False,
        False,
        want_force=False,
        want_gfmc=False,
        want_param_deriv=True,
        precision_policy=policy,
    )
    assert ctx.precision_policy() == policy
    mesh = icosahedron_sym_mesh_Nv12
    native = ctx.pd_el_ecp_j3_basis(up, dn, np.asarray(mesh.weights), np.asarray(mesh.grid_points))

    analytical = jax.jit(_dV_ecp_dc)

    def radial_reference(operator):
        gradient = analytical(ham.replace(coulomb_potential_data=operator), jnp.asarray(up), jnp.asarray(dn), jnp.eye(3))
        return gradient.jastrow_data.jastrow_three_body_data.orb_data.radial

    nearest = radial_reference(coulomb)
    # The sum of both single-centre operators includes every projector for every electron.
    one_centre = [radial_reference(_single_centre_operator(coulomb, atom)) for atom in (0, 1)]
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, "medium")
    for field, native_block in zip(("exponents", "coefficients"), native, strict=True):
        actual = np.asarray(native_block)
        expected = np.asarray(getattr(nearest, field)).ravel()
        all_pairs = sum(np.asarray(getattr(part, field)).ravel() for part in one_centre)
        assert actual.shape == expected.shape == all_pairs.shape and actual.size
        assert np.isfinite(actual).all() and np.isfinite(expected).all() and np.isfinite(all_pairs).all()
        assert np.any(np.abs(all_pairs - expected) > atol + rtol * np.abs(expected)), (
            f"{field}: all-pairs gradient must be distinguishable from nearest-only"
        )
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=f"J3 {field}: nearest-projector gradient")
    jax.clear_caches()
