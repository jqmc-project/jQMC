"""Independent wide-lattice checks of the bounded Ewald real-space table."""

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

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import coulomb_potential as cp
from jqmc._precision import get_tolerance
from jqmc._setting import ATOL_PYTHON_FP64_LOOSE
from jqmc.structure import Structure_data

pytestmark = [pytest.mark.without_chdir]


def _integer_box(width):
    return np.array(list(itertools.product(range(-width, width + 1), repeat=3)), dtype=np.int64)


def _inside_cutoff(vectors, cutoff):
    """Return row indices inside the physical sphere, including boundary ties."""
    distances = np.linalg.norm(vectors, axis=1)
    ordered = np.argsort(distances)
    return ordered[: np.searchsorted(distances[ordered], cutoff, side="right")]


@pytest.mark.light
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Geometric image completeness and table size; no electrons."
)
def test_real_images_are_complete_and_bounded():
    """Independent over-wide boxes cover corners, skew cells, slabs and cutoff ties.

    For each test displacement we search every integer triple in [-8,8]^3;
    a singular-value bound certifies that every omitted triple is farther
    than the cutoff. The production list must contain each contributing
    triple. The 24.4-Bohr box with cutoff 10.4 needs at most the seven
    origin/nearest-axis images, rather than the old 171-image table.
    """
    wide = _integer_box(8)
    corners = np.array(list(itertools.product([-0.5, 0.5], repeat=3)))
    fractional = np.vstack((corners, np.zeros((1, 3)), np.random.default_rng(358).uniform(-0.5, 0.5, (40, 3))))
    cells = [
        np.diag([24.4, 24.4, 24.4]),
        np.array([[0.0, 8.0, 8.0], [8.0, 0.0, 8.0], [8.0, 8.0, 0.0]]),
        np.array([[8.0, 0.0, 0.0], [5.0, 6.0, 0.0], [3.0, 2.0, 9.0]]),
        np.diag([6.0, 7.0, 30.0]),
        np.eye(3),
    ]
    for cell in cells:
        cutoff = np.sqrt(3.0) / 2.0 if np.array_equal(cell, np.eye(3)) else 10.4
        translations = wide @ cell
        images, indices = cp._enumerate_real_images(cell, cutoff)
        actual = set(map(tuple, indices.tolist()))
        assert tuple(indices[0]) == (0, 0, 0)
        np.testing.assert_array_equal(images, indices @ cell)
        displacements = fractional @ cell
        # Outside the independent box, |n| >= 9 and |nA| >= sigma_min*9.
        assert (
            9 * np.linalg.svd(cell, compute_uv=False)[-1]
            > cutoff + np.sqrt(np.sum(displacements * displacements, axis=1)).max()
        )
        for displacement in displacements:
            relevant = _inside_cutoff(displacement + translations, cutoff)
            assert set(map(tuple, wide[relevant].tolist())) <= actual
            # Nearest-image representatives are independently obtained from
            # the wide box, not from the production MIC or enumeration.
            nearest = displacement + translations[np.argmin(np.linalg.norm(displacement + translations, axis=1))]
            relevant_nearest = _inside_cutoff(nearest + translations, cutoff)
            assert set(map(tuple, wide[relevant_nearest].tolist())) <= actual
    images, _ = cp._enumerate_real_images(cells[0], 10.4)
    assert len(images) <= 7, "unnecessary real-space images inflate every pair's working array"


def _fixture():
    cell = np.eye(3) * 24.4
    structure = Structure_data.build(
        pbc_flag=True,
        positions=np.array([[0.06, 0.06, 0.06]]) @ cell,
        vec_a=cell[0],
        vec_b=cell[1],
        vec_c=cell[2],
        atomic_numbers=(2,),
        element_symbols=("He",),
        atomic_labels=("He1",),
    )
    potential = cp.Coulomb_potential_data(structure_data=structure, ecp_flag=False)
    return cell, structure, potential


@pytest.mark.light
def test_ewald_opposite_corner_oracles_match_wide_table():
    """Both electron channels and all debug oracles agree with a wide NumPy sum.

    The reference uses an independently enumerated [-6,6]^3 table and the
    independent debug Ewald formulas. Its raw pairs are within one cell,
    so it remains complete even when the debug fold is deliberately removed.
    The short-range/reciprocal formulas and existing strict Ewald budget are
    unchanged. Electrons straddle all three cell faces simultaneously.
    """
    cell, structure, potential = _fixture()
    data = cp.build_ewald_data(structure, potential, eta_override=0.5, n_electron=2)
    assert len(data.real_image_list) == 7
    wide_indices = _integer_box(6)
    wide = data.replace(real_image_list=jnp.asarray(wide_indices @ cell))
    up = jnp.asarray([[0.05, 0.05, 0.05]]) @ jnp.asarray(cell)
    dn = jnp.asarray([[0.95, 0.95, 0.95]]) @ jnp.asarray(cell)
    atol, rtol = get_tolerance("ewald", "strict")
    reference = cp._compute_ewald_potential_ee_debug(up, dn, wide)
    actual = cp.compute_ewald_potential_ee(up, dn, data)
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
    debug = cp._compute_ewald_potential_ee_debug(up, dn, data)
    np.testing.assert_allclose(debug, reference, atol=atol, rtol=rtol)
    for production, oracle, extra in [
        (cp.compute_ewald_potential_ei, cp._compute_ewald_potential_ei_debug, ()),
        (cp.compute_ewald_potential_ei_discretized, cp._compute_ewald_potential_ei_discretized_debug, (0.3,)),
    ]:
        reference = np.asarray(oracle(potential, up, dn, wide, *extra))
        actual = np.asarray(production(potential, up, dn, data, *extra))
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
        debug = np.asarray(oracle(potential, up, dn, data, *extra))
        np.testing.assert_allclose(debug, reference, atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Setup and scalar reload equality of ion-only Ewald tables."
)
def test_ewald_custom_cutoff_and_saved_scalars_preserve_tables():
    """Undersized overrides keep the saved image indices; scalar replay is exact.

    Changing the call's tol during scalar replay must not change the real
    table policy. The singular-value certificate bounds this independent
    wide box also for the legacy envelope.
    """
    cell, structure, potential = _fixture()
    wide = _integer_box(6)
    for cutoff in (None, 0.1):
        data = cp.build_ewald_data(structure, potential, eta_override=0.5, r_real_cut_override=cutoff)
        restored = cp.build_ewald_data(
            structure,
            potential,
            tol=ATOL_PYTHON_FP64_LOOSE,
            eta_override=data.eta,
            r_real_cut_override=data.r_real_cut,
            g_max_override=data.g_max,
        )
        np.testing.assert_array_equal(restored.real_image_indices, data.real_image_indices)
        np.testing.assert_array_equal(restored.real_image_list, data.real_image_list)
        np.testing.assert_array_equal(restored.g_vectors, data.g_vectors)
        np.testing.assert_array_equal(restored.g_factors, data.g_factors)
        assert restored.ii_energy == data.ii_energy
        if cutoff is not None:
            radius = cutoff + sum(np.linalg.norm(row) for row in cell)
            assert 7 * np.linalg.svd(cell, compute_uv=False)[-1] > radius
            reference = set(map(tuple, wide[np.linalg.norm(wide @ cell, axis=1) <= radius].tolist()))
            actual = set(map(tuple, np.asarray(data.real_image_indices).reshape(-1, 3).tolist()))
            assert actual == reference


@pytest.mark.light
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="HDF5 persistence and reconstruction of ion-only Ewald tables."
)
def test_ewald_hdf5_preserves_legacy_real_indices(tmp_path):
    """Actual HDF5 replay preserves the old envelope as well as new tables.

    The existing flattened integer field survives serialization even though
    Cartesian tables are stripped. Scalar-only old files still rebuild.
    """
    from jqmc.hamiltonians import Hamiltonian_data
    from tests.test_jqmc_kernel.test_hamiltonian import _build_pbc_dummy_hamiltonian

    ham = _build_pbc_dummy_hamiltonian("pw", seed=2025)
    data = ham.coulomb_potential_data.ewald_data
    cell = np.asarray(data.structure_data.cell)
    for legacy in (False, True):
        _, indices = cp._enumerate_real_images(cell, data.r_real_cut, legacy_envelope=legacy)
        old = cp.build_ewald_data(
            data.structure_data,
            ham.coulomb_potential_data,
            eta_override=data.eta,
            r_real_cut_override=data.r_real_cut,
            g_max_override=data.g_max,
            real_image_indices_override=tuple(indices.ravel().tolist()),
        )
        original = ham.replace(coulomb_potential_data=ham.coulomb_potential_data.replace(ewald_data=old))
        path = tmp_path / f"ewald-{legacy}.h5"
        original.save_to_hdf5(str(path))
        restored = Hamiltonian_data.load_from_hdf5(str(path)).coulomb_potential_data.ewald_data
        np.testing.assert_array_equal(restored.real_image_indices, old.real_image_indices)
        np.testing.assert_array_equal(restored.real_image_list, old.real_image_list)
        assert restored.ii_energy == old.ii_energy
