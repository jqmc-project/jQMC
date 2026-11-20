"""Carried kinetic/ECP candidate caches against same-inverse fresh evaluation."""

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

from jqmc import coulomb_potential as coulomb, determinant, wavefunction
from jqmc._precision import get_tolerance_min
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_KINETIC_PATTERNS = (
    "WF-PATTERN-004",
    "WF-PATTERN-006",
    "WF-PATTERN-008",
    "WF-PATTERN-009",
    "WF-PATTERN-010",
    "WF-PATTERN-013",
    "WF-PATTERN-014",
    "WF-PATTERN-016",
)
_ECP_PATTERNS = ("WF-PATTERN-004", "WF-PATTERN-006", "WF-PATTERN-008", "WF-PATTERN-014")
_COMMITTED = ("mol_ecp_cart_j123", "pbc_gamma_sphepgto_j123", "twist_sphepgto_j123")


def _corrupt(state, channel):
    """Change a real cached field; a fallback that ignores it cannot pass this control."""
    if channel in ("j1", "j2"):
        name = channel + "_candidate_state"
        data = getattr(state, name)
        return state.replace(**{name: data.replace(log_ratio_up=data.log_ratio_up + 0.05)})
    if channel == "dtn":
        data = state.j_dtn_candidate_state
        return state.replace(j_dtn_candidate_state=data.replace(log_delta=data.log_delta + 0.05))
    name = "det_candidate_state" if channel == "det" else "j3_candidate_state"
    data = getattr(state, name)
    field = "orb_up_candidates" if channel == "det" else "aos_up_candidates"
    return state.replace(**{name: data.replace(**{field: getattr(data, field) * 1.01})})


def _chain(ham, family, corruption=None, *, legacy_positions=False):
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    geminal = wf.geminal_data
    rng = np.random.default_rng(0 if legacy_positions else 864)
    n_up, n_dn = geminal.num_electron_up, geminal.num_electron_dn
    if legacy_positions and ham.structure_data.pbc_flag:
        atoms = np.asarray(ham.structure_data.positions)
        r_up = jnp.asarray(np.repeat(atoms, n_up // len(atoms), axis=0) + rng.normal(scale=0.4, size=(n_up, 3)))
        r_dn = jnp.asarray(np.repeat(atoms, n_dn // len(atoms), axis=0) + rng.normal(scale=0.4, size=(n_dn, 3)))
    elif legacy_positions:
        r_up = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n_up, 3)))
        r_dn = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n_dn, 3)))
    else:
        r_up = jnp.asarray(rng.normal(scale=0.6, size=(n_up, 3)))
        r_dn = jnp.asarray(rng.normal(scale=0.6, size=(n_dn, 3)))
    rotation = jnp.eye(3)
    alat = 0.15
    if family == "kinetic":
        state = wavefunction._init_discretized_kinetic_mesh_streaming_state(alat, wf, r_up, r_dn, rotation)
    else:
        assert family == "ecp" and cp.ecp_flag
        state = coulomb._init_ecp_candidate_mesh_streaming_state(cp, wf, r_up, r_dn, rotation, 1, 6)
    if corruption is not None:
        state = _corrupt(state, corruption)
    atol, rtol = get_tolerance_min(("wf_kinetic", "coulomb"), "strict")
    for hop in range(5):
        matrix = determinant.compute_geminal_all_elements(geminal, r_up, r_dn)
        cond_g = float(np.linalg.cond(matrix))
        assert np.isfinite(cond_g)
        # Both routes receive the identical inverse. This isolates candidate
        # cache drift; it does not certify a carried Sherman--Morrison inverse.
        inverse = jnp.linalg.inv(matrix)
        det_state = determinant._init_det_ratio_streaming_state(geminal, r_up, r_dn)
        if family == "kinetic":
            cached = wavefunction.compute_discretized_kinetic_energy_from_streaming_state(
                alat, wf, inverse, r_up, r_dn, state, None, det_state
            )
            fresh = wavefunction.compute_discretized_kinetic_energy_fast_update(
                alat, wf, inverse, r_up, r_dn, rotation, det_ratio_state=det_state
            )
        else:
            kwargs = dict(
                coulomb_potential_data=cp,
                wavefunction_data=wf,
                r_up_carts=r_up,
                r_dn_carts=r_dn,
                RT=rotation,
                A_old_inv=inverse,
                NN=1,
                Nv=6,
                det_ratio_state=det_state,
            )
            cached = coulomb.compute_ecp_non_local_parts_nearest_neighbors_fast_update(**kwargs, candidate_mesh_state=state)
            fresh = coulomb.compute_ecp_non_local_parts_nearest_neighbors_fast_update(**kwargs)
        actual_positions = np.concatenate([np.asarray(value).reshape(-1) for value in cached[:2]])
        reference_positions = np.concatenate([np.asarray(value).reshape(-1) for value in fresh[:2]])
        configuration = f"hop-{hop}"
        np.testing.assert_array_equal(actual_positions, reference_positions, err_msg=configuration)
        values = [("mesh-values", cached[2], fresh[2])]
        cached_total = cached[3] if family == "ecp" else jnp.sum(cached[2])
        fresh_total = fresh[3] if family == "ecp" else jnp.sum(fresh[2])
        values.append(("mesh-total", cached_total, fresh_total))
        for field, observed, expected in values:
            quantity = f"hop-{hop}:{field}"
            actual, reference = np.asarray(observed), np.asarray(expected)
            np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=quantity)
        if hop == 4:
            break
        moved_up = hop % 2 == 0 or n_dn == 0
        index = (hop // 2) % (n_up if moved_up else n_dn)
        shift = jnp.asarray([0.017, -0.013, 0.019]) * (hop + 1)
        next_up = r_up.at[index].add(shift) if moved_up else r_up
        next_dn = r_dn if moved_up else r_dn.at[index].add(shift)
        args = (wf, state, jnp.bool_(moved_up), jnp.int64(index), r_up, r_dn, next_up, next_dn)
        if family == "kinetic":
            state = wavefunction._advance_discretized_kinetic_mesh_streaming_state(*args)
        else:
            state = coulomb._advance_ecp_candidate_mesh_streaming_state(cp, *args, rotation)
        r_up, r_dn = next_up, next_dn
    jax.clear_caches()


@pytest.mark.parametrize("pattern", _KINETIC_PATTERNS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_canonical_kinetic_candidate_cache_values(pattern):
    """Eight synthetic canonical Jastrow fixtures retain every six-point kinetic value through four cache advances.

    No TREXIO file is used. Retain the declared orbital/spin/cell signatures,
    seed-864 normal(scale=0.6) positions, identity rotation and alat=0.15.
    Four fixed displacements alternate spins (up only for an empty down sector).
    No conditioning screen is used. Each hop compares all cached mesh values
    with a fresh candidate evaluation using the identical fresh inverse and
    determinant state. The existing consumer strict tolerance applies in each
    precision separately. This is candidate-cache parity, not inverse stability.
    """
    _chain(build_canonical_hamiltonian(pattern), "kinetic")


@pytest.mark.parametrize("pattern", _ECP_PATTERNS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_canonical_ecp_candidate_cache_values(pattern):
    """Four periodic canonical ECP fixtures preserve every nonlocal weight and total across four cache advances.

    Synthetic inputs have active nonzero projectors, their declared Jastrow and
    spin signatures, seed-864 normal(scale=0.6) positions, Nv=6, NN=1 and identity
    rotation. No TREXIO or conditioning selection. Compare cached versus fresh
    candidates on the same inverse at all five configurations with unchanged
    consumer strict tolerance, independently in full and mixed.
    """
    _chain(build_canonical_hamiltonian(pattern), "ecp")


@pytest.mark.parametrize("pattern", _COMMITTED)
@pytest.mark.parametrize("family", ["kinetic", "ecp"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_committed_j123_candidate_cache_values(pattern, family):
    """H2-ECP, Gamma Li and non-Gamma Li with J123 preserve cached mesh values at five fixed configurations.

    Committed H2_ecp_ccpvdz_cart.h5, lithium.h5 and lithium_k025.h5 supply the
    existing rs_ham SD/J123 fixtures. Seed 864 draws normal(scale=0.6) positions;
    no conditioning rejection. Four fixed displacements alternate spins.
    Kinetic uses alat=0.15; ECP uses Nv=6, NN=1; both use identity rotation.
    Fresh candidates and carried candidates receive the same fresh inverse.
    Entire mesh values and ECP total use the original strict consumer policy.
    """
    _chain(rs_ham(pattern), family)


@pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "silicon.h5"])
@pytest.mark.parametrize("family", ["kinetic", "ecp"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_water_silicon_j123_candidate_cache_values(trexio_file, family):
    """Committed water and silicon with active J123 retain cached mesh values through both-spin updates.

    The existing derivative builder supplies SD/J1-exp/J2-exp/identity-MO-J3 from each
    named TREXIO. PCG64 seed 0 draws uniform(-1,1) molecular positions, or
    normal(scale=0.4) offsets around the repeated periodic atoms. No conditioning
    screen. Four fixed moves, alat=0.15, Nv=6, NN=1 and identity rotation compare
    all five cached/fresh results on identical inverses under consumer strict
    tolerance. Full and mixed are separate obligations.
    """
    ham = _build_ham_for_cell(trexio_file, True, {"j3": "mo"}, "SD")
    _chain(ham, family, legacy_positions=True)


@pytest.mark.parametrize("family", ["kinetic", "ecp"])
@pytest.mark.parametrize("channel", ["det", "j1", "j2", "j3", "dtn"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_each_candidate_channel_is_consumed_and_checked(family, channel):
    """Corrupting each real candidate-cache channel must fail after the identical clean chain passes.

    Canonical 006 supplies all five active caches, nonzero ECP, complex periodic
    orbitals and seed-864 unscreened positions. Modify a stored up-spin orbital
    table or log-ratio table only; do not change the uncached reference. A
    fallback that ignores the supplied cache would fail to detect the fault.
    """
    ham = build_canonical_hamiltonian("WF-PATTERN-006")
    _chain(ham, family)
    with pytest.raises(AssertionError):
        _chain(ham, family, corruption=channel)
    jax.clear_caches()
