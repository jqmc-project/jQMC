"""Unit tests for Structure_data and PBC helpers."""

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

from jqmc._setting import RTOL_NONVACUITY_DISTINCTION
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
from jqmc._precision import get_tolerance, get_tolerance_min
from jqmc._setting import (
    AO_IMAGE_TRUNCATION_ATOL,
    ATOL_BIT_EXACT,
    ATOL_DEBUG_VS_PRODUCTION,
    ATOL_PYTHON_FP64_LOOSE,
    RTOL_BIT_EXACT,
    RTOL_PYTHON_FP64_LOOSE,
)
from jqmc.structure import (
    Structure_data,
    _find_nearest_index_jnp,
    _find_nearest_index_np,
    _find_nearest_nucleus_indices_jnp,
    _find_nearest_nucleus_indices_np,
    _get_min_dist_rel_R_cart_jnp,
    _get_min_dist_rel_R_cart_np,
    _min_image_fractional_parallelepiped,
    _minimum_image_displacement,
    _minimum_image_displacement_debug,
    _primitive_image_list,
    _primitive_image_list_debug,
    _primitive_image_list_indices,
    _primitive_image_list_indices_per_atom,
    _primitive_image_radius,
    normalize_tiling_matrix,
    tile_structure_data,
    tiling_translation_fractions,
)
from tests.conftest import (
    CELL_FIXTURES,
    EXPECTED_N_MAX,
    brute_force_mic,
    make_pbc_structure,
)
from tests._image_list_oracle import image_indices as independent_image_indices, sorted_indices


def _make_pbc_structure():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )
    return Structure_data.build(
        positions=positions,
        pbc_flag=True,
        vec_a=(10.0, 0.0, 0.0),
        vec_b=(0.0, 12.0, 0.0),
        vec_c=(0.0, 0.0, 14.0),
        atomic_numbers=(1, 1, 1),
        element_symbols=("H", "H", "H"),
        atomic_labels=("H1", "H2", "H3"),
    )


def _make_non_pbc_structure():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    return Structure_data(
        positions=positions,
        pbc_flag=False,
        atomic_numbers=(1, 1, 1),
        element_symbols=("H", "H", "H"),
        atomic_labels=("H1", "H2", "H3"),
    )


class TestStructureBasics:
    """Reciprocal lattice + np/jnp consistency (molecular + PBC)."""

    @pytest.mark.light
    def test_reciprocal_lattice_dot_2pi(self):
        """Test that the dot product of the cell and reciprocal cell gives 2pi delta_ij."""
        atol, rtol = get_tolerance("local_energy", "strict")
        structure = _make_pbc_structure()
        recip = structure.recip_cell
        cell = structure.cell
        for i in range(3):
            for j in range(3):
                pair_label = f"{i},{j}"
                dot = np.dot(cell[i], recip[j])
                expected = 2.0 * np.pi if i == j else 0.0
                assert not np.any(np.isnan(np.asarray(dot))), "NaN detected in first argument"
                assert not np.any(np.isnan(np.asarray(expected))), "NaN detected in second argument"
                np.testing.assert_allclose(dot, expected, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_general_integer_tiling_replicates_atoms_and_cell(self):
        """A skew integer tiling enumerates every atom coset and transforms the cell exactly."""
        primitive = Structure_data.build(
            positions=np.array([[0.2, 0.6, 1.2]], dtype=np.float64),
            pbc_flag=True,
            vec_a=(2.0, 0.0, 0.0),
            vec_b=(0.0, 3.0, 0.0),
            vec_c=(0.0, 0.0, 4.0),
            atomic_numbers=(2,),
            element_symbols=("He",),
            atomic_labels=("He",),
        )
        matrix = normalize_tiling_matrix(((1, 1, 0), (1, -1, 0), (0, 0, 1)))
        np.testing.assert_array_equal(normalize_tiling_matrix((1, 1, 0, 1, -1, 0, 0, 0, 1)), matrix)
        translations = tiling_translation_fractions(matrix)
        tiled = tile_structure_data(primitive, matrix)

        assert translations.shape == (2, 3)
        assert np.unique(np.round(translations, 14), axis=0).shape[0] == 2
        np.testing.assert_allclose(tiled.cell, matrix @ primitive.cell, rtol=RTOL_BIT_EXACT, atol=ATOL_BIT_EXACT)
        assert len(tiled.atomic_numbers) == 2
        assert abs(np.linalg.det(tiled.cell)) == pytest.approx(2.0 * abs(np.linalg.det(primitive.cell)))
        primitive_fractional = primitive.positions[0] @ np.linalg.inv(primitive.cell)
        recovered = np.asarray(tiled.positions) @ np.linalg.inv(primitive.cell)
        for row in recovered:
            np.testing.assert_allclose(
                row - primitive_fractional,
                np.rint(row - primitive_fractional),
                rtol=RTOL_BIT_EXACT,
                atol=ATOL_PYTHON_FP64_LOOSE,
            )

    @pytest.mark.light
    def test_np_jnp_consistency_non_pbc(self):
        """Test consistency between NumPy and JAX implementations for non-PBC structures."""
        atol, rtol = get_tolerance("local_energy", "strict")
        structure = _make_non_pbc_structure()
        r_cart = np.array([0.2, 0.0, 0.0])

        assert not np.any(np.isnan(np.asarray(structure._positions_cart_np))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(structure._positions_cart_jnp)))), "NaN detected in second argument"
        np.testing.assert_allclose(
            structure._positions_cart_np,
            np.asarray(structure._positions_cart_jnp),
            atol=atol,
            rtol=rtol,
        )

        idx_np = _find_nearest_index_np(structure, r_cart)
        idx_jnp = int(np.asarray(_find_nearest_index_jnp(structure, r_cart)))
        assert idx_np == idx_jnp

        nearest_np = _find_nearest_nucleus_indices_np(structure, r_cart, 2)
        nearest_jnp = np.asarray(_find_nearest_nucleus_indices_jnp(structure, r_cart, 2))
        np.testing.assert_array_equal(nearest_np, nearest_jnp)

        for i_atom in range(structure.natom):
            atom_label = str(i_atom)
            rel_np = _get_min_dist_rel_R_cart_np(structure, r_cart, i_atom)
            rel_jnp = np.asarray(_get_min_dist_rel_R_cart_jnp(structure, r_cart, i_atom))
            assert not np.any(np.isnan(np.asarray(rel_np))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(rel_jnp))), "NaN detected in second argument"
            np.testing.assert_allclose(rel_np, rel_jnp, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_pbc_minimum_image_and_nearest(self):
        """Test PBC minimum image convention and nearest nucleus finding."""
        atol, rtol = get_tolerance("local_energy", "strict")
        structure = _make_pbc_structure()
        r_cart = np.array([9.1, 0.0, 0.0])

        idx_np = _find_nearest_index_np(structure, r_cart)
        idx_jnp = int(np.asarray(_find_nearest_index_jnp(structure, r_cart)))
        assert idx_np == idx_jnp == 1

        nearest_np = _find_nearest_nucleus_indices_np(structure, r_cart, 2)
        nearest_jnp = np.asarray(_find_nearest_nucleus_indices_jnp(structure, r_cart, 2))
        np.testing.assert_array_equal(nearest_np, nearest_jnp)

        rel_atom0 = _get_min_dist_rel_R_cart_np(structure, r_cart, 0)
        rel_atom1 = _get_min_dist_rel_R_cart_np(structure, r_cart, 1)

        assert not np.any(np.isnan(np.asarray(rel_atom0))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.array([0.9, 0.0, 0.0])))), "NaN detected in second argument"
        np.testing.assert_allclose(
            rel_atom0,
            np.array([0.9, 0.0, 0.0]),
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(rel_atom1))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.array([-0.1, 0.0, 0.0])))), "NaN detected in second argument"
        np.testing.assert_allclose(
            rel_atom1,
            np.array([-0.1, 0.0, 0.0]),
            atol=atol,
            rtol=rtol,
        )

        rel_atom0_jnp = np.asarray(_get_min_dist_rel_R_cart_jnp(structure, r_cart, 0))
        rel_atom1_jnp = np.asarray(_get_min_dist_rel_R_cart_jnp(structure, r_cart, 1))
        assert not np.any(np.isnan(np.asarray(rel_atom0_jnp))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(rel_atom0))), "NaN detected in second argument"
        np.testing.assert_allclose(rel_atom0_jnp, rel_atom0, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(rel_atom1_jnp))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(rel_atom1))), "NaN detected in second argument"
        np.testing.assert_allclose(rel_atom1_jnp, rel_atom1, atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("use_pbc", [False, True])
    def test_find_nearest_index_matches_min_dist_jnp(self, use_pbc):
        """Test that the nearest index found matches the minimum distance calculation."""
        atol, rtol = get_tolerance("local_energy", "strict")
        structure = _make_pbc_structure() if use_pbc else _make_non_pbc_structure()
        r_cart = np.array([9.1, 0.0, 0.0]) if use_pbc else np.array([1.8, 0.1, 0.0])

        idx = int(np.asarray(_find_nearest_index_jnp(structure, r_cart)))
        rel = np.asarray(_get_min_dist_rel_R_cart_jnp(structure, r_cart, idx))
        dist_idx = np.linalg.norm(rel)

        diffs = structure._positions_cart_np - r_cart
        if use_pbc:
            cell = structure.cell
            inv_cell = np.linalg.inv(cell)
            diffs_frac = diffs @ inv_cell
            diffs_frac = diffs_frac - np.round(diffs_frac)
            diffs = diffs_frac @ cell
        dist_all = np.linalg.norm(diffs, axis=1)

        assert not np.any(np.isnan(np.asarray(dist_idx))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.min(dist_all)))), "NaN detected in second argument"
        np.testing.assert_allclose(dist_idx, np.min(dist_all), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# PBC validation -- sanity_check refuses malformed pbc_flag / lattice combos.
# ---------------------------------------------------------------------------


class TestStructurePBCValidation:
    """pbc_flag / lattice combinations -- sanity_check refuses malformed combos."""

    @pytest.mark.light
    def test_pbc_flag_true_without_lattice_vectors_raises(self):
        """``pbc_flag=True`` with empty / wrong-length ``vec_a/b/c`` must fail
        ``sanity_check`` (``__post_init__`` only normalizes the vectors and does
        not derive ``_n_max``, so the dataclass itself constructs).
        """
        s = Structure_data(
            pbc_flag=True,
            positions=np.zeros((1, 3), dtype=np.float64),
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
            # vec_a/b/c default to empty tuples.
        )
        with pytest.raises(ValueError, match="lattice"):
            s.sanity_check()

    @pytest.mark.light
    def test_pbc_flag_false_with_lattice_vectors_raises(self):
        """``pbc_flag=False`` with non-empty lattice vectors must fail
        ``sanity_check``.
        """
        s = Structure_data(
            pbc_flag=False,
            positions=np.zeros((1, 3), dtype=np.float64),
            vec_a=(10.0, 0.0, 0.0),
            vec_b=(0.0, 10.0, 0.0),
            vec_c=(0.0, 0.0, 10.0),
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        with pytest.raises(ValueError, match="empty"):
            s.sanity_check()


# ---------------------------------------------------------------------------
# cell_inradius and _n_max -- formula goldens for the three PBC fixtures.
# ---------------------------------------------------------------------------


class TestStructureLattice:
    """cell_inradius, n_max goldens, position-update invariant."""

    @pytest.mark.light
    @pytest.mark.parametrize("cell_name", list(CELL_FIXTURES))
    def test_cell_inradius_matches_design_formula(self, cell_name):
        """``cell_inradius == V / max_i |T_j x T_k|`` (perpendicular distance
        between opposing parallelepiped faces).
        """
        cell = CELL_FIXTURES[cell_name]
        structure = make_pbc_structure(cell)
        V = float(abs(np.dot(cell[0], np.cross(cell[1], cell[2]))))
        face_areas = [
            float(np.linalg.norm(np.cross(cell[1], cell[2]))),
            float(np.linalg.norm(np.cross(cell[2], cell[0]))),
            float(np.linalg.norm(np.cross(cell[0], cell[1]))),
        ]
        expected = V / max(face_areas)
        assert structure.cell_inradius() == pytest.approx(expected, rel=RTOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    def test_cell_inradius_molecular_raises(self):
        """``cell_inradius`` is undefined for molecular structures."""
        s = _make_non_pbc_structure()
        with pytest.raises(ValueError):
            s.cell_inradius()

    @pytest.mark.light
    @pytest.mark.parametrize("cell_name", list(CELL_FIXTURES))
    def test_n_max_matches_design_goldens(self, cell_name):
        """``Structure_data._n_max`` equals the design goldens
        (cubic_10bohr=2, primitive_fcc_si=3, triclinic_skewed=3).
        """
        structure = make_pbc_structure(CELL_FIXTURES[cell_name])
        assert structure._n_max == EXPECTED_N_MAX[cell_name], (
            f"{cell_name}: got _n_max = {structure._n_max}, expected {EXPECTED_N_MAX[cell_name]}"
        )

    @pytest.mark.light
    @pytest.mark.parametrize("cell_name", list(CELL_FIXTURES))
    def test_n_max_invariant_under_position_update(self, cell_name):
        """``_n_max`` is derived from the lattice alone -- updating positions
        via ``replace(positions=...)`` must leave it unchanged so the JIT
        cache for MIC kernels does not invalidate on position gradients
        (JIT-cache invariant).
        """
        original = make_pbc_structure(CELL_FIXTURES[cell_name])
        n_max_before = original._n_max
        rng = np.random.default_rng(seed=20260603)
        new_positions = rng.uniform(-2.0, 2.0, size=(1, 3)).astype(np.float64)
        moved = original.replace(positions=new_positions)
        assert moved._n_max == n_max_before


# ---------------------------------------------------------------------------
# General Wigner-Seitz minimum-image -- production / debug / brute-force.
# ---------------------------------------------------------------------------


class TestStructureMIC:
    """General-MIC vs brute-force, prod-vs-debug, orthorhombic equivalence,
    oblique distinctness, parallelepiped range.
    """

    @pytest.mark.light
    @pytest.mark.parametrize(
        "cell_name,seed",
        [("cubic_10bohr", 20260603), ("primitive_fcc_si", 20260604), ("triclinic_skewed", 20260605)],
    )
    def test_general_mic_vs_brute_force(self, cell_name, seed):
        """``_minimum_image_displacement`` matches a +/- 5-image brute-force
        enumeration on 100 random probes for cubic / primitive-FCC /
        triclinic cells. The simple parallelepiped wrap is only exact for
        orthorhombic cells; this is the canonical
        correctness check for the general algorithm.
        """
        cell = CELL_FIXTURES[cell_name]
        inv_cell = np.linalg.inv(cell)
        n_max = EXPECTED_N_MAX[cell_name]
        rng = np.random.default_rng(seed=seed)
        atol, rtol = get_tolerance("local_energy", "strict")
        for probe in range(100):
            probe_label = str(probe)
            diff = rng.uniform(-3.0, 3.0, size=3)
            got = np.asarray(
                _minimum_image_displacement(
                    jnp.asarray(diff),
                    jnp.asarray(cell),
                    jnp.asarray(inv_cell),
                    n_max,
                )
            )
            ref = brute_force_mic(diff, cell, n_range=5)
            assert not np.any(np.isnan(got))
            np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("cell_name", list(CELL_FIXTURES))
    def test_general_mic_production_vs_debug(self, cell_name):
        """JIT'd ``_minimum_image_displacement`` and the numpy-loop debug
        sibling ``_minimum_image_displacement_debug`` agree at fp64
        round-off.
        """
        cell = CELL_FIXTURES[cell_name]
        inv_cell = np.linalg.inv(cell)
        n_max = EXPECTED_N_MAX[cell_name]
        rng = np.random.default_rng(seed=20260604)
        atol, rtol = get_tolerance("local_energy", "strict")
        for probe in range(50):
            probe_label = str(probe)
            diff = rng.uniform(-2.0, 2.0, size=3)
            got = np.asarray(
                _minimum_image_displacement(
                    jnp.asarray(diff),
                    jnp.asarray(cell),
                    jnp.asarray(inv_cell),
                    n_max,
                )
            )
            ref = _minimum_image_displacement_debug(diff, cell, inv_cell, n_max)
            np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("cell_name", list(CELL_FIXTURES))
    def test_general_mic_distance_derivatives_away_from_seams(self, cell_name):
        """The shared MIC has the Euclidean gradient/Laplacian on each image branch.

        Random probes include unwrapped displacements, so non-zero image indices
        and oblique-cell branches are exercised. Probability of landing exactly on
        a Wigner--Seitz face is zero; an explicit face probe below documents and
        detects the expected one-sided derivative jump separately.
        """
        cell_np = CELL_FIXTURES[cell_name]
        inv_np = np.linalg.inv(cell_np)
        n_max = EXPECTED_N_MAX[cell_name]
        cell = jnp.asarray(cell_np, dtype=jnp.float64)
        inv_cell = jnp.asarray(inv_np, dtype=jnp.float64)
        rng = np.random.default_rng(seed=20260804)
        atol, rtol = get_tolerance("local_energy", "strict")

        def distance(x):
            return jnp.linalg.norm(_minimum_image_displacement(x, cell, inv_cell, n_max))

        for probe in range(12):
            probe_label = str(probe)
            frac = rng.uniform(-2.5, 2.5, size=3)
            diff_np = frac @ cell_np
            ref = brute_force_mic(diff_np, cell_np, n_range=5)
            radius = float(np.linalg.norm(ref))
            if radius < 1.0e-6:
                continue
            diff = jnp.asarray(diff_np, dtype=jnp.float64)
            value = float(distance(diff))
            grad = np.asarray(jax.grad(distance)(diff))
            lap = float(jnp.trace(jax.hessian(distance)(diff)))
            np.testing.assert_allclose(value, radius, atol=atol, rtol=rtol)
            np.testing.assert_allclose(grad, ref / radius, atol=atol, rtol=rtol)
            np.testing.assert_allclose(lap, 2.0 / radius, atol=atol, rtol=rtol)

            shift = rng.integers(-3, 4, size=3) @ cell_np
            shifted = diff + jnp.asarray(shift, dtype=jnp.float64)
            np.testing.assert_allclose(float(distance(shifted)), value, atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(jax.grad(distance)(shifted)), grad, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_general_mic_face_probe_resolves_expected_gradient_jump(self):
        """Calibration probe: raw MIC distance is continuous but not C1 at a face.

        This is intentional geometry, not a kernel defect. Keeping the explicit
        negative control ensures that the seam probes on cutoff wavefunctions can
        actually resolve the failure mode that caused the DTN energy collapse.
        """
        cell = jnp.asarray(CELL_FIXTURES["cubic_10bohr"], dtype=jnp.float64)
        inv_cell = jnp.linalg.inv(cell)
        n_max = EXPECTED_N_MAX["cubic_10bohr"]
        eps = 1.0e-6

        def distance(x):
            return jnp.linalg.norm(_minimum_image_displacement(x, cell, inv_cell, n_max))

        left = jnp.asarray([5.0 - eps, 0.0, 0.0], dtype=jnp.float64)
        right = jnp.asarray([5.0 + eps, 0.0, 0.0], dtype=jnp.float64)
        np.testing.assert_allclose(
            float(distance(left)),
            float(distance(right)),
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_LOOSE,
        )
        assert float(jax.grad(distance)(left)[0]) > 0.99
        assert float(jax.grad(distance)(right)[0]) < -0.99

    @pytest.mark.light
    @pytest.mark.parametrize("cell_name", list(CELL_FIXTURES))
    def test_structure_ei_and_ii_consumers_match_shared_mic(self, cell_name):
        """The e--i wrappers and i--i nearest-neighbour policy share exact MIC.

        In particular this guards against reintroducing the fractional-rounding
        shortcut in ``_median_nn_distance``; that shortcut disagrees with the
        Wigner--Seitz nearest image on oblique cells and changes DTN
        geometry-derived cutoffs.
        """
        cell = CELL_FIXTURES[cell_name]
        rng = np.random.default_rng(seed=20260805)
        positions = rng.random((5, 3)) @ cell
        structure = make_pbc_structure(
            cell,
            positions=positions,
            atomic_numbers=(1,) * len(positions),
            element_symbols=("H",) * len(positions),
            atomic_labels=tuple(f"H{i + 1}" for i in range(len(positions))),
        )
        inv_cell = np.linalg.inv(cell)
        atol, rtol = get_tolerance("local_energy", "strict")

        point = rng.random(3) @ cell + rng.integers(-3, 4, size=3) @ cell
        distances = []
        for atom in range(len(positions)):
            atom_label = str(atom)
            ref = _minimum_image_displacement_debug(
                positions[atom] - point,
                cell,
                inv_cell,
                structure._n_max,
            )
            got_np = _get_min_dist_rel_R_cart_np(structure, point, atom)
            got_jnp = np.asarray(_get_min_dist_rel_R_cart_jnp(structure, point, atom), dtype=np.float64)
            np.testing.assert_allclose(got_np, ref, atol=atol, rtol=rtol)
            np.testing.assert_allclose(got_jnp, ref, atol=atol, rtol=rtol)
            distances.append(float(np.linalg.norm(ref)))
        assert int(_find_nearest_index_np(structure, point)) == int(np.argmin(distances))
        assert int(_find_nearest_index_jnp(structure, point)) == int(np.argmin(distances))

        nearest = []
        for i in range(len(positions)):
            nearest.append(
                min(
                    np.linalg.norm(
                        _minimum_image_displacement_debug(
                            positions[i] - positions[j],
                            cell,
                            inv_cell,
                            structure._n_max,
                        )
                    )
                    for j in range(len(positions))
                    if i != j
                )
            )
        np.testing.assert_allclose(structure._median_nn_distance(), np.median(nearest), atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_orthorhombic_general_equals_parallelepiped_wrap(self):
        """On the orthorhombic ``cubic_10bohr`` fixture the parallelepiped
        fractional wrap and the general WS MIC coincide (the two methods
        only diverge on oblique cells).
        """
        cell = CELL_FIXTURES["cubic_10bohr"]
        inv_cell = np.linalg.inv(cell)
        n_max = EXPECTED_N_MAX["cubic_10bohr"]
        rng = np.random.default_rng(seed=20260605)
        atol, rtol = get_tolerance("local_energy", "strict")
        for probe in range(20):
            probe_label = str(probe)
            diff = rng.uniform(-2.0, 2.0, size=3)
            general = np.asarray(
                _minimum_image_displacement(
                    jnp.asarray(diff),
                    jnp.asarray(cell),
                    jnp.asarray(inv_cell),
                    n_max,
                )
            )
            para_frac = np.asarray(
                _min_image_fractional_parallelepiped(
                    jnp.asarray(diff),
                    jnp.asarray(inv_cell),
                )
            )
            para_cart = para_frac @ cell
            np.testing.assert_allclose(general, para_cart, atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("cell_name", ["primitive_fcc_si", "triclinic_skewed"])
    def test_parallelepiped_helper_distinct_from_ws_on_oblique(self, cell_name):
        """On oblique cells (FCC primitive, triclinic) the parallelepiped
        wrap and the general WS MIC can disagree; this divergence is why the
        general algorithm is used. Find at least one random
        probe where the two differ.
        """
        cell = CELL_FIXTURES[cell_name]
        inv_cell = np.linalg.inv(cell)
        n_max = EXPECTED_N_MAX[cell_name]
        rng = np.random.default_rng(seed=20260606)
        for _ in range(200):
            diff = rng.uniform(-3.0, 3.0, size=3)
            general = np.asarray(
                _minimum_image_displacement(
                    jnp.asarray(diff),
                    jnp.asarray(cell),
                    jnp.asarray(inv_cell),
                    n_max,
                )
            )
            para_frac = np.asarray(
                _min_image_fractional_parallelepiped(
                    jnp.asarray(diff),
                    jnp.asarray(inv_cell),
                )
            )
            para_cart = para_frac @ cell
            if not np.allclose(general, para_cart, atol=ATOL_DEBUG_VS_PRODUCTION, rtol=RTOL_NONVACUITY_DISTINCTION):
                break
        assert not np.allclose(general, para_cart, atol=ATOL_DEBUG_VS_PRODUCTION, rtol=RTOL_NONVACUITY_DISTINCTION), (
            f"{cell_name}: expected at least one probe where parallelepiped "
            f"wrap differs from general WS MIC; if this is unexpectedly clean "
            f"the reason for the general WS MIC needs to be revisited."
        )
        # Distinctness alone is not the mathematical oracle. Both helpers must
        # also give their defining values on this same selected probe.
        atol, rtol = get_tolerance_min(("local_energy",), level="strict")
        np.testing.assert_allclose(general, brute_force_mic(diff, cell), atol=atol, rtol=rtol, equal_nan=False)
        fractional = np.linalg.solve(cell.T, diff)
        expected_para = (fractional - np.floor(fractional + 0.5)) @ cell
        np.testing.assert_allclose(para_cart, expected_para, atol=atol, rtol=rtol, equal_nan=False)

    @pytest.mark.light
    def test_parallelepiped_fractional_in_unit_box(self):
        """``_min_image_fractional_parallelepiped`` always returns
        fractional coordinates in ``[-1/2, 1/2)`` along each axis.
        """
        cell = CELL_FIXTURES["primitive_fcc_si"]
        inv_cell = np.linalg.inv(cell)
        rng = np.random.default_rng(seed=20260607)
        for _ in range(50):
            diff = rng.uniform(-10.0, 10.0, size=3)
            frac = np.asarray(
                _min_image_fractional_parallelepiped(
                    jnp.asarray(diff),
                    jnp.asarray(inv_cell),
                )
            )
            assert np.all(frac >= -0.5 - ATOL_PYTHON_FP64_LOOSE)
            assert np.all(frac < 0.5 + ATOL_PYTHON_FP64_LOOSE)


# ---------------------------------------------------------------------------
# Primitive image list -- structure on (alpha, L, tol), invariants, defaults.
# ---------------------------------------------------------------------------


class TestStructureImageList:
    """T0 membership, prod-vs-debug, exponent/L scaling, closed-form L=0,
    dtype, default-tol flow.
    """

    @pytest.mark.light
    @pytest.mark.parametrize("cell_name", list(CELL_FIXTURES))
    def test_primitive_image_list_contains_T0(self, cell_name):
        """``T_0 = (0, 0, 0)`` must be a member of the image set (the
        closest-approach distance from ``T_0`` to ``Omega`` is ``0`` which
        is ``<= r_cut`` for any non-trivial truncation tolerance). The
        "T_0 first" ordering is *not* a property of this low-level helper
        -- it is established later by :py:func:`_build_image_expanded_arrays`
        via an explicit sort + insert; here we only check membership.
        """
        cell = CELL_FIXTURES[cell_name]
        structure = make_pbc_structure(cell)
        images = structure.primitive_image_list(exponent=1.0, angular_L=1)
        assert np.any(np.all(images == 0.0, axis=1)), f"{cell_name}: image set must contain T_0 = (0, 0, 0)"
        # This contract is specifically about membership of the origin, not
        # approximate agreement of an entire image array with a reference.
        origin = images[np.argmin(np.sum(images**2, axis=1))]
        expected_origin = np.zeros(3, dtype=images.dtype)
        np.testing.assert_array_equal(origin, expected_origin)

    @pytest.mark.light
    @pytest.mark.parametrize("cell_name", list(CELL_FIXTURES))
    def test_primitive_image_list_production_vs_debug(self, cell_name):
        """``_primitive_image_list`` and ``_primitive_image_list_debug``
        return the same image set (modulo row ordering, which the production
        helper may permute via numpy mask operations).
        """
        cell = CELL_FIXTURES[cell_name]
        prod = _primitive_image_list(cell, exponent=1.0, angular_L=1)
        debug = _primitive_image_list_debug(cell, exponent=1.0, angular_L=1)
        prod_set = {tuple(np.round(row, 9)) for row in prod}
        debug_set = {tuple(np.round(row, 9)) for row in debug}
        assert prod_set == debug_set

    @pytest.mark.light
    @pytest.mark.parametrize(
        "cell_name,exponent,angular_l",
        [
            ("cubic_10bohr", 0.1, 0),
            ("cubic_10bohr", 1.0, 2),
            ("primitive_fcc_si", 1.0, 0),
            ("primitive_fcc_si", 0.1, 1),
            ("triclinic_skewed", 1.0, 0),
            ("triclinic_skewed", 0.1, 2),
        ],
    )
    def test_primitive_image_indices_match_independent_active_set(self, cell_name, exponent, angular_l):
        """Cell-cell image indices equal an independent 27-face closest-approach reference.

        Cubic, primitive FCC and triclinic cells cover selected compact/diffuse
        s and non-s shells. The reference uses an SVD candidate bound and an
        independently solved Gaussian-envelope radius; the integer rows must
        agree exactly, including multiplicity. No electron positions are needed.
        """
        cell = CELL_FIXTURES[cell_name]
        actual = sorted_indices(_primitive_image_list_indices(cell, exponent, angular_l, AO_IMAGE_TRUNCATION_ATOL))
        reference = independent_image_indices(cell, exponent, angular_l, AO_IMAGE_TRUNCATION_ATOL)
        np.testing.assert_array_equal(actual, reference)

    @pytest.mark.light
    @pytest.mark.parametrize("cell_name", list(CELL_FIXTURES))
    @pytest.mark.parametrize(
        "fractional_atom",
        [
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.0, 0.5, 0.5),
            (0.15, 0.4, 0.75),
            (-0.1, 0.4, 0.75),
            (0.15, 1.1, 0.75),
            (-1.0e-12, 0.5, 1.0),
            (-1.1, 0.4, 2.15),
        ],
    )
    def test_per_atom_image_indices_match_independent_active_set(self, cell_name, fractional_atom):
        """Corner, center, face and asymmetric atoms have exactly the independently enumerated images.

        A fixed diffuse s shell in cubic, primitive FCC and triclinic cells
        probes the shifted unit box, independently of the production candidate
        grid and coordinate descent. Atom coordinates are not electron positions.
        """
        cell = CELL_FIXTURES[cell_name]
        actual = sorted_indices(_primitive_image_list_indices_per_atom(cell, 0.1, 0, fractional_atom, AO_IMAGE_TRUNCATION_ATOL))
        reference = independent_image_indices(cell, 0.1, 0, AO_IMAGE_TRUNCATION_ATOL, fractional_atom)
        np.testing.assert_array_equal(actual, reference)

    @pytest.mark.light
    def test_primitive_image_list_grows_with_diffuse_exponent(self):
        """Smaller ``alpha`` -> larger ``r_cut`` -> at least as many images."""
        cell = CELL_FIXTURES["cubic_10bohr"]
        images_compact = _primitive_image_list(cell, exponent=5.0, angular_L=0)
        images_diffuse = _primitive_image_list(cell, exponent=0.1, angular_L=0)
        assert len(images_diffuse) >= len(images_compact)
        np.testing.assert_array_equal(
            sorted_indices(images_compact), independent_image_indices(cell, 5.0, 0, AO_IMAGE_TRUNCATION_ATOL) @ np.asarray(cell)
        )
        np.testing.assert_array_equal(
            sorted_indices(images_diffuse), independent_image_indices(cell, 0.1, 0, AO_IMAGE_TRUNCATION_ATOL) @ np.asarray(cell)
        )

    @pytest.mark.light
    def test_primitive_image_list_grows_with_higher_L(self):
        """Higher ``L`` -> larger ``r_cut`` from the polynomial-peak term ->
        at least as many images. Equality is possible at moderate ``alpha``
        where the polynomial correction is small relative to the lattice
        inradius; strict inequality requires a sufficiently diffuse exponent.
        """
        cell = CELL_FIXTURES["cubic_10bohr"]
        # Moderate alpha: polynomial-correction term is sub-Bohr, so L=0 and L=1
        # both fall inside the same nearest-neighbour shell -> equal counts.
        images_l0_mod = _primitive_image_list(cell, exponent=1.0, angular_L=0)
        images_l1_mod = _primitive_image_list(cell, exponent=1.0, angular_L=1)
        assert len(images_l1_mod) >= len(images_l0_mod)
        # Diffuse alpha: polynomial-correction term reaches the next shell -> strict.
        # Pin ``tol`` (independent of the production default
        # ``AO_IMAGE_TRUNCATION_ATOL``) so the L=1 polynomial correction crosses the
        # next image shell regardless of how the production default is tuned.
        images_l0_diff = _primitive_image_list(cell, exponent=0.05, angular_L=0, tol=ATOL_PYTHON_FP64_LOOSE)
        images_l1_diff = _primitive_image_list(cell, exponent=0.05, angular_L=1, tol=ATOL_PYTHON_FP64_LOOSE)
        assert len(images_l1_diff) > len(images_l0_diff)
        np.testing.assert_array_equal(
            sorted_indices(images_l0_mod), independent_image_indices(cell, 1.0, 0, AO_IMAGE_TRUNCATION_ATOL) @ np.asarray(cell)
        )
        np.testing.assert_array_equal(
            sorted_indices(images_l1_mod), independent_image_indices(cell, 1.0, 1, AO_IMAGE_TRUNCATION_ATOL) @ np.asarray(cell)
        )
        np.testing.assert_array_equal(
            sorted_indices(images_l0_diff), independent_image_indices(cell, 0.05, 0, ATOL_PYTHON_FP64_LOOSE) @ np.asarray(cell)
        )
        np.testing.assert_array_equal(
            sorted_indices(images_l1_diff), independent_image_indices(cell, 0.05, 1, ATOL_PYTHON_FP64_LOOSE) @ np.asarray(cell)
        )

    @pytest.mark.light
    def test_primitive_image_radius_s_orbital_matches_closed_form(self):
        """For ``L = 0`` the truncation radius reduces to
        ``sqrt(-ln(tol) / alpha)`` (the ``max(L, 1)`` guard makes the second
        term vanish).
        """
        r_cut = _primitive_image_radius(exponent=1.0, angular_L=0, tol=ATOL_PYTHON_FP64_LOOSE)
        expected = float(np.sqrt(-np.log(ATOL_PYTHON_FP64_LOOSE) / 1.0))
        assert r_cut == pytest.approx(expected, rel=RTOL_PYTHON_FP64_LOOSE)
        assert np.isfinite(r_cut) and r_cut > 0

    @pytest.mark.light
    def test_primitive_image_list_indices_returns_int64_and_contains_origin(self):
        """``_primitive_image_list_indices`` returns ``int64`` triplets that
        include ``(0, 0, 0)`` somewhere in the set. Consumed by
        :py:func:`_build_image_expanded_arrays` where set-style unions need
        hashable integer keys; ordering is re-established by the consumer.
        """
        cell = CELL_FIXTURES["cubic_10bohr"]
        indices = _primitive_image_list_indices(cell, exponent=1.0, angular_L=0)
        assert indices.dtype == np.int64
        assert np.any(np.all(indices == 0, axis=1)), "image-index set must contain (0, 0, 0)"
        origin = indices[np.argmin(np.sum(indices**2, axis=1))]
        expected_origin = np.zeros(3, dtype=np.int64)
        np.testing.assert_array_equal(origin, expected_origin)

    @pytest.mark.light
    def test_primitive_image_list_default_tol_uses_setting(self):
        """Default ``tol`` of ``primitive_image_list`` is sourced from
        ``jqmc._setting.AO_IMAGE_TRUNCATION_ATOL`` -- changing the setting
        must flow through the helper without per-call updates.
        """
        cell = CELL_FIXTURES["cubic_10bohr"]
        structure = make_pbc_structure(cell)
        images_default = structure.primitive_image_list(exponent=1.0, angular_L=0)
        images_explicit = structure.primitive_image_list(
            exponent=1.0,
            angular_L=0,
            tol=AO_IMAGE_TRUNCATION_ATOL,
        )
        np.testing.assert_array_equal(images_default, images_explicit)


# ---------------------------------------------------------------------------
# Per-atom image truncation -- the floor-breaking ``Omega - R_A`` image set. The cell-cell
# metric (``_primitive_image_list_indices``) keeps all 26 nearest neighbours
# unconditionally (their cell-to-cell closest approach is 0), so the image
# count floors at 27 even for a fully-localised shell. The per-atom metric
# replaces the size-2 box ``[n-1, n+1]`` with the size-1, atom-shifted box
# ``[-n-s_A, 1-n-s_A]``, dropping nearest neighbours an atom genuinely cannot
# reach.
# ---------------------------------------------------------------------------


class TestPerAtomImageList:
    """Per-atom (Omega - R_A) image set: breaks the 27-image floor.

    Spec (measured; 10-Bohr cubic, localised alpha=100,
    r_cut~0.53 Bohr): a centred atom keeps only T_0; a corner atom keeps the
    interior octant ``{0,1}^3`` (8 images); a face atom keeps
    ``{T_0, +face neighbour}`` (2 images). The set always contains T_0, is
    always a subset of the cell-cell set, and recovers the full cell-cell set
    as ``r_cut -> cell`` (diffuse limit).
    """

    CELL = np.diag([10.0, 10.0, 10.0]).astype(np.float64)
    TOL = 1.0e-12

    @staticmethod
    def _set(idx) -> set:
        return {tuple(int(c) for c in row) for row in np.asarray(idx).reshape(-1, 3)}

    @pytest.mark.light
    def test_localized_center_atom_keeps_only_T0(self):
        """Localized center atom keeps only T0."""
        idx = _primitive_image_list_indices_per_atom(
            self.CELL, exponent=100.0, angular_L=0, frac_pos=(0.5, 0.5, 0.5), tol=self.TOL
        )
        assert self._set(idx) == {(0, 0, 0)}

    @pytest.mark.light
    def test_localized_corner_atom_keeps_interior_octant(self):
        """Localized corner atom keeps interior octant."""
        idx = _primitive_image_list_indices_per_atom(
            self.CELL, exponent=100.0, angular_L=0, frac_pos=(0.0, 0.0, 0.0), tol=self.TOL
        )
        expected = {(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)}
        assert self._set(idx) == expected

    @pytest.mark.light
    def test_localized_face_atom_keeps_T0_plus_one_neighbor(self):
        """Localized face atom keeps T0 plus one neighbor."""
        idx = _primitive_image_list_indices_per_atom(
            self.CELL, exponent=100.0, angular_L=0, frac_pos=(0.0, 0.5, 0.5), tol=self.TOL
        )
        assert self._set(idx) == {(0, 0, 0), (1, 0, 0)}

    @pytest.mark.light
    @pytest.mark.parametrize("frac_pos", [(0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (0.1, 0.7, 0.3)])
    @pytest.mark.parametrize("alpha", [100.0, 1.0, 0.3])
    def test_contains_T0_and_subset_of_cellcell(self, frac_pos, alpha):
        """Contains T0 and subset of cellcell."""
        per_atom = self._set(
            _primitive_image_list_indices_per_atom(self.CELL, exponent=alpha, angular_L=1, frac_pos=frac_pos, tol=self.TOL)
        )
        cell_cell = self._set(_primitive_image_list_indices(self.CELL, alpha, 1, tol=self.TOL))
        assert (0, 0, 0) in per_atom, f"T_0 must always be kept (alpha={alpha}, frac={frac_pos})"
        assert per_atom <= cell_cell, f"per-atom set must be a subset of cell-cell (alpha={alpha}, frac={frac_pos})"

    @pytest.mark.light
    def test_diffuse_center_atom_recovers_full_cellcell_set(self):
        """As ``r_cut -> cell`` the per-atom centre set recovers the cell-cell set."""
        per_atom = self._set(
            _primitive_image_list_indices_per_atom(self.CELL, exponent=0.3, angular_L=0, frac_pos=(0.5, 0.5, 0.5), tol=self.TOL)
        )
        cell_cell = self._set(_primitive_image_list_indices(self.CELL, 0.3, 0, tol=self.TOL))
        assert per_atom == cell_cell


def _qe_reference(atom_labels, atom_elements):
    """Minimal QEReference for testing from_qe_reference's element/label split.

    Only the fields from_qe_reference reads (cell, positions, labels, elements)
    carry meaningful values; the wfc/bands are dummy single-band placeholders.
    """
    from jqmc.qe_wrapper import QEBands, QECell, QEReference, QEWavefunction

    n = len(atom_labels)
    cell = QECell(a=np.eye(3) * 10.0)
    wfc = QEWavefunction(
        millers=np.zeros((1, 3), dtype=np.int64),
        coeffs=np.zeros((1, 1), dtype=np.complex128),
        gamma_only=False,
        k_point=np.zeros(3),
    )
    bands = QEBands(eigenvalues=np.zeros(1), occupations=np.ones(1), n_electrons=1.0, fermi_energy=0.0)
    positions = np.array([[float(i), float(i), float(i)] for i in range(n)], dtype=np.float64)
    return QEReference(
        cell=cell,
        atom_labels=tuple(atom_labels),
        atom_positions=positions,
        ecutwfc=10.0,
        fft_grid=(8, 8, 8),
        bands=bands,
        wfc=wfc,
        atom_elements=tuple(atom_elements),
    )


class TestFromQEReferenceElementLabelSplit:
    """from_qe_reference must keep the UPF-resolved element (element_symbols +
    Z) distinct from the raw, possibly-tagged QE species label (atomic_labels),
    so an antiferromagnetic 'O' / 'O1' setup round-trips.
    """

    @pytest.mark.light
    def test_tagged_labels_split_element_and_label(self):
        """Tagged labels split element and label."""
        ref = _qe_reference(atom_labels=("O", "O1"), atom_elements=("O", "O"))
        sd = Structure_data.from_qe_reference(ref)
        assert tuple(sd.element_symbols) == ("O", "O")  # UPF element
        assert tuple(sd.atomic_labels) == ("O", "O1")  # raw QE label preserved
        assert tuple(sd.atomic_numbers) == (8, 8)  # Z from the element

    @pytest.mark.light
    def test_untagged_labels_unchanged(self):
        """Untagged labels unchanged."""
        ref = _qe_reference(atom_labels=("Si", "Si"), atom_elements=("Si", "Si"))
        sd = Structure_data.from_qe_reference(ref)
        assert tuple(sd.element_symbols) == ("Si", "Si")
        assert tuple(sd.atomic_labels) == ("Si", "Si")
        assert tuple(sd.atomic_numbers) == (14, 14)

    @pytest.mark.light
    def test_missing_atom_elements_raises(self):
        """Missing atom elements raises."""
        ref = _qe_reference(atom_labels=("O", "O1"), atom_elements=())
        with pytest.raises(ValueError, match="atom_elements"):
            Structure_data.from_qe_reference(ref)


# ---------------------------------------------------------------------------
# Issue #67: minimum image on a non-reduced cell basis.
# ---------------------------------------------------------------------------
#: Rectangular lattice of the issue #67 reproduction (Bohr).
NON_REDUCED_MIC_LENGTHS = np.array([6.0, 7.0, 8.0])
#: Unimodular basis change (det +1) whose ``a + b + c`` diagonal (6 Bohr) is far
#: shorter than the other body diagonals (33, 69, 53 Bohr), so the old
#: ``a + b + c``-only box (``n_max = 5``) misses minimum images needing ``|n_k| = 6``.
NON_REDUCED_MIC_TRANSFORM = np.array([[6, 1, -1], [-2, 1, 0], [-3, -2, 1]], dtype=np.float64)
#: Displacement from the issue report: the old box returns 5.768 Bohr, the minimum is 5.042 Bohr.
NON_REDUCED_MIC_REPORTED_DISPLACEMENT = np.array([3.7458936435198424, 39.16755006833722, -35.50971359433966])


def non_reduced_mic_displacements(cell: np.ndarray) -> np.ndarray:
    """Reported displacement plus seeded displacements spanning several cells of ``cell``."""
    rng = np.random.default_rng(seed=20261012)
    random = rng.uniform(-2.0, 2.0, size=(3000, 3)) @ cell
    return np.concatenate((NON_REDUCED_MIC_REPORTED_DISPLACEMENT[None, :], random), axis=0)


def orthorhombic_minimum_image(displacements: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Exact minimum image in the rectangular basis ``diag(lengths)`` of the same lattice.

    For an orthogonal basis the shortest lattice-equivalent vector is the per-axis
    wrap to ``[-L_k / 2, L_k / 2]`` (the squared norm separates by axis). No
    production helper and no image box are involved.
    """
    return displacements - np.round(displacements / lengths) * lengths


@pytest.mark.light
def test_minimum_image_in_non_reduced_basis_matches_orthorhombic_reference():
    """XLA and ``_debug`` minimum images on a non-reduced basis are the exact shortest vectors.

    The lattice diag(6, 7, 8) Bohr is written in the basis ``U @ diag(6, 7, 8)``
    (``det U = 1``). The minimum image depends only on the lattice, so its length must
    equal the per-axis wrap in the rectangular basis, and the returned vector must be
    lattice-equivalent to the input. The production consumer
    ``_get_min_dist_rel_R_cart_jnp`` (nearest-nucleus distance) is checked as well.
    """
    lengths = NON_REDUCED_MIC_LENGTHS
    cell = NON_REDUCED_MIC_TRANSFORM @ np.diag(lengths)
    inv_cell = np.linalg.inv(cell)
    structure = make_pbc_structure(cell)
    displacements = non_reduced_mic_displacements(cell)
    reference = orthorhombic_minimum_image(displacements, lengths)
    reference_norm = np.linalg.norm(reference, axis=1)
    atol, rtol = ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE

    # Non-vacuity: some minimum image needs a coefficient beyond the box of the a+b+c diagonal alone.
    wrapped_frac = displacements @ inv_cell
    wrapped = (wrapped_frac - np.round(wrapped_frac)) @ cell
    needed = np.max(np.abs(np.rint((reference - wrapped) @ inv_cell)))
    half_diag = 0.5 * np.linalg.norm(cell.sum(axis=0))
    old_box = int(np.ceil(half_diag / structure.cell_inradius())) + 1
    assert needed > old_box, (needed, old_box)

    xla = np.asarray(
        jax.vmap(lambda d: _minimum_image_displacement(d, jnp.asarray(cell), jnp.asarray(inv_cell), structure._n_max))(
            jnp.asarray(displacements)
        ),
        dtype=np.float64,
    )
    debug = np.stack([_minimum_image_displacement_debug(d, cell, inv_cell, structure._n_max) for d in displacements[:400]])
    nearest = np.stack(
        [np.asarray(_get_min_dist_rel_R_cart_jnp(structure, jnp.asarray(-d), 0), dtype=np.float64) for d in displacements[:400]]
    )
    for label, got in (("xla", xla), ("debug", debug), ("nearest_jnp", nearest)):
        count = len(got)
        np.testing.assert_allclose(np.linalg.norm(got, axis=1), reference_norm[:count], atol=atol, rtol=rtol, err_msg=label)
        shift = (got - displacements[:count]) @ inv_cell
        np.testing.assert_allclose(shift, np.rint(shift), atol=atol, rtol=rtol, err_msg=label)


# ---------------------------------------------------------------------------
# Issue #68: Wigner-Seitz inscribed radius on any cell basis.
# ---------------------------------------------------------------------------
#: Lattices in a reduced (standard) basis, Bohr. FCC and BCC primitive cells and
#: the hexagonal cell are oblique: half their smallest interplanar spacing is
#: below the WS inscribed radius even in this basis.
WS_RADIUS_LATTICES = {
    "orthorhombic_6_7_8": np.diag([6.0, 7.0, 8.0]),
    "fcc_a8": 4.0 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
    "bcc_a8": 4.0 * np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]),
    "hexagonal_a6_c10": np.array([[6.0, 0.0, 0.0], [-3.0, 3.0 * np.sqrt(3.0), 0.0], [0.0, 0.0, 10.0]]),
}
#: Unimodular basis changes (det +1): the same lattice is written as ``U @ L``.
WS_RADIUS_BASIS_CHANGES = {
    "identity": np.eye(3),
    "single_shear": np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "non_reduced_transform": NON_REDUCED_MIC_TRANSFORM,
}


def brute_force_half_shortest_lattice_vector(cell: np.ndarray, n_box: int = 4) -> float:
    """Half the shortest non-zero ``n @ cell`` over ``|n_k| <= n_box`` (independent enumeration).

    The box provably contains the shortest vector ``T``: ``|T| <= l = min_i |b_i|`` and
    ``n_k = T . inv(cell)[:, k]``, so ``|n_k| <= l |inv(cell)[:, k]|``; the assert checks
    that this bound fits in the box. No production helper and no lattice reduction is used.
    """
    cell = np.asarray(cell, dtype=np.float64)
    shortest_row = np.min(np.linalg.norm(cell, axis=1))
    coefficient_bound = shortest_row * np.linalg.norm(np.linalg.inv(cell), axis=0)
    assert np.all(np.floor(coefficient_bound) <= n_box), coefficient_bound
    grid = np.array([n for n in itertools.product(range(-n_box, n_box + 1), repeat=3) if any(n)], dtype=np.float64)
    return 0.5 * float(np.min(np.linalg.norm(grid @ cell, axis=1)))


@pytest.mark.light
def test_wigner_seitz_inscribed_radius_is_half_shortest_lattice_vector_in_any_basis():
    """``_wigner_seitz_inscribed_radius`` is half the shortest lattice vector, for any cell basis.

    Each lattice of :data:`WS_RADIUS_LATTICES` is written in the bases ``U @ L`` of
    :data:`WS_RADIUS_BASIS_CHANGES`. The reference is the brute-force half shortest
    vector of ``L``; it depends only on the lattice. Half the smallest
    interplanar spacing of the given basis is smaller on the oblique lattices
    and on every non-identity basis. An orthorhombic cell in its own basis
    gives ``0.5 / max_j |g_j|`` bit for bit.
    """
    atol, rtol = ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE
    for lattice_name, lattice in WS_RADIUS_LATTICES.items():
        reference = brute_force_half_shortest_lattice_vector(lattice)
        for basis_name, transform in WS_RADIUS_BASIS_CHANGES.items():
            cell = transform @ lattice
            radius = make_pbc_structure(cell)._wigner_seitz_inscribed_radius()
            np.testing.assert_allclose(radius, reference, atol=atol, rtol=rtol, err_msg=f"{lattice_name}/{basis_name}")
    orthorhombic = WS_RADIUS_LATTICES["orthorhombic_6_7_8"]
    legacy = float(0.5 / np.max(np.linalg.norm(np.linalg.inv(orthorhombic), axis=0)))
    assert make_pbc_structure(orthorhombic)._wigner_seitz_inscribed_radius() == legacy
