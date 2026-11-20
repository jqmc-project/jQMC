"""Cached-J3 ECP ratios after a t-move onto the electron's own quadrature point (issues #652, #692).

Within one GFMC projection interval the ECP quadrature rotation is fixed. After a
t-move the moved electron therefore sits exactly on a point of its own candidate
mesh, and that candidate configuration moves no electron: its Jastrow ratio is
exactly 1. The cached-J3 branches located the moved electron by ``argmax`` over
the displacement mask (index 0 for a no-move configuration) while reading the
cached candidate orbitals of the candidate's own electron, so the ratio was wrong
for every electron index >= 1. This checks the two consumers of the carried ECP
candidate cache against the cache-free evaluation at the post-t-move configuration:
the t-move non-local ECP kernel (split-spin ratio) and the rank-one Jastrow ratio
used by the DLA t-move. The molecular case (issue #692: N2 ccECP with a real
AO-leaf J3, rotated quadrature) also compares both kernels with the brute-force
non-local ECP evaluation.
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

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import coulomb_potential as coulomb, determinant
from jqmc._precision import get_tolerance_min
from jqmc.jastrow_factor import (
    _compute_ratio_Jastrow_part_rank1_update,
    _init_grads_laplacian_Jastrow_three_body_streaming_state,
)
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian
from tests.test_jqmc_kernel.test_gfmc_force_analytic import _build_ham_jastrow, _rot

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_NN = 1
_NV = 6
_MAX_TMOVES = 4


@pytest.mark.parametrize("moved_spin", ["up", "dn"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_ecp_candidate_cache_after_tmove_onto_own_quadrature_point(moved_spin):
    """Electron 1 of a 2 up / 2 dn complex periodic ECP J1/J2/J3/DTN wavefunction t-moves onto its own mesh.

    Canonical WF-PATTERN-006, seed-864 normal(scale=0.6) positions, identity rotation,
    Nv=6, NN=1. Electron index 1 of the chosen spin hops onto a candidate of its own
    carried ECP mesh (the t-move) and the carried mesh is advanced with the same
    rotation; the first candidate whose advanced mesh contains the hopped electron's own
    position (a no-move candidate) is used, and the test fails if none does. Cached and cache-free consumers then receive the same
    fresh inverse: the t-move non-local values and total, and the rank-one Jastrow
    ratio on the ECP mesh, under the existing consumer strict tolerance, separately in
    full and mixed.
    """
    ham = build_canonical_hamiltonian("WF-PATTERN-006")
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    geminal, jastrow = wf.geminal_data, wf.jastrow_data
    assert geminal.num_electron_up >= 2 and geminal.num_electron_dn >= 2 and cp.ecp_flag
    assert jastrow.jastrow_three_body_data is not None
    rng = np.random.default_rng(864)
    r_up = jnp.asarray(rng.normal(scale=0.6, size=(geminal.num_electron_up, 3)))
    r_dn = jnp.asarray(rng.normal(scale=0.6, size=(geminal.num_electron_dn, 3)))
    rotation = jnp.eye(3)
    state = coulomb._init_ecp_candidate_mesh_streaming_state(cp, wf, r_up, r_dn, rotation, _NN, _NV)

    # The t-move: electron 1 of the chosen spin hops onto one of its own candidates and
    # the carried mesh is advanced with the same rotation. Whether the advanced mesh holds
    # the post-hop position fp-exactly depends on rounding, so take the first candidate
    # that does; if none does, t-move onto candidate 0 and try again from there (a chain
    # of t-moves of the same electron, as within one projection interval).
    index = 1
    moved_up = moved_spin == "up"
    found = False
    for _ in range(_MAX_TMOVES):
        candidates = state.candidate_up_positions if moved_up else state.candidate_dn_positions
        for k in range(_NN * _NV):
            target = candidates[index, k]
            next_up = r_up.at[index].set(target) if moved_up else r_up
            next_dn = r_dn if moved_up else r_dn.at[index].set(target)
            advanced = coulomb._advance_ecp_candidate_mesh_streaming_state(
                cp, wf, state, jnp.bool_(moved_up), jnp.int64(index), r_up, r_dn, next_up, next_dn, rotation
            )
            own = np.asarray(advanced.candidate_up_positions if moved_up else advanced.candidate_dn_positions)[index]
            if np.any(np.all(own == np.asarray(target), axis=-1)):
                found = True
                break
            if k == 0:
                first = (advanced, next_up, next_dn)
        if found:
            break
        state, r_up, r_dn = first
    if not found:
        pytest.fail("no t-move target reproduces itself on the advanced mesh: the case would be vacuous")
    state, r_up, r_dn = advanced, next_up, next_dn

    inverse = jnp.linalg.inv(determinant.compute_geminal_all_elements(geminal, r_up, r_dn))
    det_state = determinant._init_det_ratio_streaming_state(geminal, r_up, r_dn)
    j3_state = _init_grads_laplacian_Jastrow_three_body_streaming_state(jastrow.jastrow_three_body_data, r_up, r_dn)
    atol, rtol = get_tolerance_min(("wf_kinetic", "coulomb"), "strict")
    kwargs = dict(
        coulomb_potential_data=cp,
        wavefunction_data=wf,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
        RT=rotation,
        A_old_inv=inverse,
        NN=_NN,
        Nv=_NV,
        det_ratio_state=det_state,
    )
    cached = coulomb.compute_ecp_non_local_parts_nearest_neighbors_fast_update(
        **kwargs, j3_state=j3_state, candidate_mesh_state=state
    )
    fresh = coulomb.compute_ecp_non_local_parts_nearest_neighbors_fast_update(**kwargs)
    for value_index, quantity in ((2, "tmove-mesh-values"), (3, "tmove-mesh-total")):
        np.testing.assert_allclose(
            np.asarray(cached[value_index]),
            np.asarray(fresh[value_index]),
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg=quantity,
        )

    mesh_up, mesh_dn = jnp.asarray(fresh[0]), jnp.asarray(fresh[1])
    ratio_cached = _compute_ratio_Jastrow_part_rank1_update(
        jastrow,
        r_up,
        r_dn,
        mesh_up,
        mesh_dn,
        j3_state=j3_state,
        j1_candidate_state=state.j1_candidate_state,
        j3_candidate_state=state.j3_candidate_state,
        j2_candidate_state=state.j2_candidate_state,
        j_dtn_candidate_state=state.j_dtn_candidate_state,
    )
    ratio_fresh = _compute_ratio_Jastrow_part_rank1_update(jastrow, r_up, r_dn, mesh_up, mesh_dn)
    np.testing.assert_allclose(
        np.asarray(ratio_cached),
        np.asarray(ratio_fresh),
        atol=atol,
        rtol=rtol,
        equal_nan=False,
        err_msg="dltmove-rank1-jastrow-ratio",
    )


_MOL_NN = 1
_MOL_NV = 12


@pytest.mark.parametrize("moved_spin", ["up", "dn"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_molecular_ecp_candidate_cache_with_an_electron_on_a_rotated_quadrature_point(moved_spin):
    """N2 ccECP SD + exp J1/J2 + real AO-leaf J3 with one electron exactly on its own rotated ECP mesh.

    Committed N2_ecp_ccpvtz_cart.h5 with the ``j123ao`` Jastrow of
    test_gfmc_force_analytic (random symmetric J3 block, seed 5), 5 up / 5 dn,
    seed-692 positions nucleus 0 + normal(scale=0.8) (up then dn), the
    test_gfmc_force_analytic seed-692 QR mesh rotation, Nv=12, NN=1. Electron 1 of
    the chosen spin is put on one of its own rotated quadrature points; the first
    point whose freshly built mesh contains that position bit for bit is used
    (a zero-displacement mesh row, as after a t-move in GFMC), else the electron
    moves onto point 0 and the search repeats (at most four moves), and the test
    fails if none does. The cached-state ECP kernel (candidate state and J3
    state initialized at that configuration) must equal the cache-free fast
    update under the existing cached-consumer strict policy, and both must equal
    the brute-force ``compute_ecp_non_local_parts_nearest_neighbors`` (full
    wavefunction ratios) under the existing fast-vs-reference strict policy:
    every mesh value and the total, separately in full and mixed.
    """
    ham = _build_ham_jastrow("N2_ecp_ccpvtz_cart.h5", "j123ao")
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    geminal, jastrow = wf.geminal_data, wf.jastrow_data
    assert geminal.num_electron_up >= 2 and geminal.num_electron_dn >= 2 and cp.ecp_flag
    assert jastrow.jastrow_three_body_data is not None
    rng = np.random.default_rng(692)
    center = np.asarray(ham.structure_data.positions)[0]
    r_up = jnp.asarray(center + rng.normal(scale=0.8, size=(geminal.num_electron_up, 3)))
    r_dn = jnp.asarray(center + rng.normal(scale=0.8, size=(geminal.num_electron_dn, 3)))
    rotation = _rot(692)
    moved_up = moved_spin == "up"
    index = 1

    # Put electron ``index`` on one of its own rotated quadrature points. Whether the
    # mesh rebuilt at that position holds it bit for bit depends on rounding, so take
    # the first point that does; if none does, move onto point 0 and try again from
    # there (a chain of t-moves of the same electron, as within one projection interval).
    state = coulomb._init_ecp_candidate_mesh_streaming_state(cp, wf, r_up, r_dn, rotation, _MOL_NN, _MOL_NV)
    found = False
    for _ in range(_MAX_TMOVES):
        candidates = np.asarray(state.candidate_up_positions if moved_up else state.candidate_dn_positions)[index]
        for k, target in enumerate(candidates):
            next_up = r_up.at[index].set(target) if moved_up else r_up
            next_dn = r_dn if moved_up else r_dn.at[index].set(target)
            trial = coulomb._init_ecp_candidate_mesh_streaming_state(cp, wf, next_up, next_dn, rotation, _MOL_NN, _MOL_NV)
            own = np.asarray(trial.candidate_up_positions if moved_up else trial.candidate_dn_positions)[index]
            if np.any(np.all(own == target, axis=-1)):
                found = True
                break
            if k == 0:
                first = (trial, next_up, next_dn)
        if found:
            break
        state, r_up, r_dn = first
    if not found:
        pytest.fail("no rotated quadrature point reproduces itself on the rebuilt mesh: the case would be vacuous")
    state, r_up, r_dn = trial, next_up, next_dn

    inverse = jnp.linalg.inv(determinant.compute_geminal_all_elements(geminal, r_up, r_dn))
    det_state = determinant._init_det_ratio_streaming_state(geminal, r_up, r_dn)
    j3_state = _init_grads_laplacian_Jastrow_three_body_streaming_state(jastrow.jastrow_three_body_data, r_up, r_dn)
    kwargs = dict(
        coulomb_potential_data=cp,
        wavefunction_data=wf,
        r_up_carts=r_up,
        r_dn_carts=r_dn,
        RT=rotation,
        A_old_inv=inverse,
        NN=_MOL_NN,
        Nv=_MOL_NV,
        det_ratio_state=det_state,
    )
    cached = coulomb.compute_ecp_non_local_parts_nearest_neighbors_fast_update(
        **kwargs, j3_state=j3_state, candidate_mesh_state=state
    )
    fresh = coulomb.compute_ecp_non_local_parts_nearest_neighbors_fast_update(**kwargs)
    brute = coulomb.compute_ecp_non_local_parts_nearest_neighbors(cp, wf, r_up, r_dn, rotation, NN=_MOL_NN, Nv=_MOL_NV)

    # The configuration really has a zero-displacement mesh row.
    mesh_up, mesh_dn = np.asarray(brute[0]), np.asarray(brute[1])
    np.testing.assert_array_equal(mesh_up, np.asarray(fresh[0]))
    np.testing.assert_array_equal(mesh_dn, np.asarray(fresh[1]))
    still = np.all(mesh_up == np.asarray(r_up)[None], axis=(1, 2)) & np.all(mesh_dn == np.asarray(r_dn)[None], axis=(1, 2))
    assert np.count_nonzero(still) >= 1

    atol, rtol = get_tolerance_min(("wf_kinetic", "coulomb"), "strict")
    atol_ref, rtol_ref = get_tolerance_min(("coulomb", "jastrow_eval", "jastrow_ratio"), "strict")
    for value_index, quantity in ((2, "mesh-values"), (3, "mesh-total")):
        np.testing.assert_allclose(
            np.asarray(cached[value_index]),
            np.asarray(fresh[value_index]),
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg=f"cached-vs-cache-free:{quantity}",
        )
        for label, actual in (("cached", cached), ("cache-free", fresh)):
            np.testing.assert_allclose(
                np.asarray(actual[value_index]),
                np.asarray(brute[value_index]),
                atol=atol_ref,
                rtol=rtol_ref,
                equal_nan=False,
                err_msg=f"{label}-vs-brute-force:{quantity}",
            )
