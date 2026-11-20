"""Unit tests for jqmc.pw_pgto_projection (PW -> pGTO projection)."""

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
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
from jqmc._setting import (
    ATOL_BIT_EXACT,
    ATOL_PW_PROJECTION_FP64,
    ATOL_PW_PROJECTION_FP64_GRID,
    ATOL_PW_PROJECTION_FP64_LOOSE,
    ATOL_PW_PROJECTION_FP64_RELAXED,
    ATOL_PW_PROJECTION_FP64_SPLINE_RESIDUAL,
    ATOL_PYTHON_FP64_LOOSE,
    ATOL_QE_PROJECTION_FIDELITY_PGTO_ONLY_MAX,
    ATOL_QE_PROJECTION_FIDELITY_PGTO_ONLY_MEAN,
    ATOL_QE_PROJECTION_FIDELITY_SPLINE_ONLY,
    ATOL_QE_PROJECTION_FIDELITY_SWEEP_TARGET,
    ATOL_QE_PROJECTION_FIDELITY_TIGHT_SWEEP,
    ATOL_QE_PROJECTION_RECONSTRUCTION_RESIDUAL,
    RTOL_PW_PROJECTION_FP64,
    RTOL_PW_PROJECTION_RESIDUAL_PARITY_FACTOR,
    RTOL_PW_PROJECTION_SPLINE_RESIDUAL_PARITY_FACTOR,
    RTOL_PW_PROJECTION_FP64_GRID,
    RTOL_PW_PROJECTION_FP64_LOOSE,
    RTOL_PW_PROJECTION_FP64_NORM,
    RTOL_PW_PROJECTION_FP64_RELAXED,
    RTOL_PYTHON_FP64_LOOSE,
)
from jqmc.pw_pgto_projection import (
    CartPrimitive,
    build_pgto_primitives,
    eval_primitive_on_grid,
    evaluate_spline_orbitals,
    fit_pw_to_spline,
    grid_points,
    project_pw_to_spline,
    reconstruct_pw_on_grid,
)
from jqmc.qe_wrapper import QECell
from jqmc.upf_wrapper import PrimitiveShell
from tests._test_data_cache import qe_example_files_root

_SAVE_DIR = qe_example_files_root() / "Li_gamma" / "out" / "li.save"
_SAVE_DIR_K025 = qe_example_files_root() / "Li_k025" / "out" / "li_k025.save"
_O_LSDA_K025 = qe_example_files_root() / "O-lsda_k025" / "out" / "o_k025.save"
_O_LSDA_GAMMA = qe_example_files_root() / "O-lsda_gamma" / "out" / "o.save"
_available = (_SAVE_DIR / "data-file-schema.xml").is_file() and (_SAVE_DIR / "wfc1.hdf5").is_file()
requires_qe = pytest.mark.skipif(not _available, reason="QE Li_gamma data not present")

_CLI_FIXTURE = qe_example_files_root() / "Li_gamma" / "out" / "li.save"
_cli_available = (_CLI_FIXTURE / "data-file-schema.xml").is_file() and (_CLI_FIXTURE / "wfc1.hdf5").is_file()
requires_cli_fixture = pytest.mark.skipif(not _cli_available, reason="tests/qe_example_files/Li_gamma not present")


def _toy_reference(cell, millers, fft_grid, ecutwfc=10.0, k_point=None, coeffs=None):
    """Minimal QEReference wrapping a chosen Miller set (no QE files needed)."""
    from jqmc.qe_wrapper import QEBands, QEReference, QEWavefunction

    if coeffs is None:
        coeffs = np.zeros((1, millers.shape[0]), dtype=np.complex128)
        coeffs[0, 0] = 1.0
    if k_point is None:
        k_point = np.zeros(3)
    wfc = QEWavefunction(
        millers=millers.astype(np.int64),
        coeffs=coeffs,
        gamma_only=False,
        k_point=np.asarray(k_point, dtype=np.float64),
    )
    bands = QEBands(eigenvalues=np.zeros(1), occupations=np.ones(1), n_electrons=1.0, fermi_energy=0.0)
    return QEReference(
        cell=cell,
        atom_labels=("X",),
        atom_positions=np.zeros((1, 3)),
        ecutwfc=ecutwfc,
        fft_grid=fft_grid,
        bands=bands,
        wfc=wfc,
    )


class TestPWPGTOProjection:
    @pytest.mark.light
    def test_pw_cutoff_is_invariant_at_plus_minus_pi_boundary(self):
        """The production low-PW handoff must select by physical ``|G+k|``.

        Fractional ``k_x=+0.5`` and ``k_x=-0.5`` are the same Brillouin-zone
        boundary point only when every Miller row is transported by ``+1`` in
        x.  The two representations must retain the same physical wavevectors.
        Changing k without transporting Miller indices is the negative control
        and must remain observably different.
        """
        from jqmc.pw_pgto_projection import pw_gvectors_within

        cell = QECell(a=2.0 * np.pi * np.eye(3))  # reciprocal rows are the Cartesian unit vectors
        # Keep only the two boundary modes. A wider contiguous G set would let
        # the bad representation silently select a neighbouring Miller row and
        # accidentally reconstruct the same physical q set.
        millers_plus = np.array([[-1, 0, 0], [0, 0, 0]])
        shift = np.array([1, 0, 0], dtype=np.int64)
        ref_plus = _toy_reference(cell, millers_plus, (8, 8, 8), k_point=np.array([0.5, 0.0, 0.0]))
        ref_minus = _toy_reference(
            cell,
            millers_plus + shift,
            (8, 8, 8),
            k_point=np.array([-0.5, 0.0, 0.0]),
        )
        ref_bad = _toy_reference(cell, millers_plus, (8, 8, 8), k_point=np.array([-0.5, 0.0, 0.0]))

        def selected_physical_q(reference):
            _millers, g_vectors = pw_gvectors_within(reference, g_cut=1.0)
            q_vectors = g_vectors + reference.wfc.k_point[None, :]
            return sorted(tuple(np.round(row, 14)) for row in q_vectors)

        q_plus = selected_physical_q(ref_plus)
        q_minus = selected_physical_q(ref_minus)
        q_bad = selected_physical_q(ref_bad)
        expected = [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)]

        assert q_plus == expected
        assert q_minus == expected
        assert q_plus == q_minus
        assert q_bad != q_plus

    @pytest.mark.light
    def test_mixed_full_overlap_condition_qr_reduction_matches_direct_matrix(self):
        """Thin-QR full-overlap conditioning must equal direct assembly."""
        from jqmc.pw_pgto_projection import _mixed_full_overlap_condition

        rng = np.random.default_rng(731)
        n_gto, n_pw = 3, 7
        volume = 4.5
        s_gp = 0.1 * (rng.normal(size=(n_gto, n_pw)) + 1j * rng.normal(size=(n_gto, n_pw)))
        core = rng.normal(size=(n_gto, n_gto)) + 1j * rng.normal(size=(n_gto, n_gto))
        schur = core @ core.conj().T + 0.5 * np.eye(n_gto)
        s_gg = schur + (s_gp @ s_gp.conj().T) / volume
        full_s = np.block(
            [
                [s_gg, s_gp],
                [s_gp.conj().T, volume * np.eye(n_pw)],
            ]
        )
        direct = np.linalg.cond(full_s)

        assert _mixed_full_overlap_condition(s_gg, s_gp, volume) == pytest.approx(
            direct,
            rel=RTOL_PW_PROJECTION_FP64,
        )

    @pytest.mark.light
    def test_gamma_generated_overlap_uses_reduced_eigensystem(self, monkeypatch):
        """The emitted Gamma AO condition uses the full spectrum without a dense PW block."""
        from jqmc import pw_pgto_projection as projection

        cell = QECell(a=7.0 * np.eye(3))
        millers = np.array([(i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)])
        ref = _toy_reference(cell, millers, (12, 12, 12))
        prims = [
            CartPrimitive(center=(1.0, 2.0, 3.0), ang=(0, 0, 0), exponent=1.6),
            CartPrimitive(center=(4.0, 3.0, 2.0), ang=(1, 0, 0), exponent=1.2),
        ]
        volume = cell.volume
        reciprocal = projection.build_gto_pw_matrix(ref, prims).conj().T
        index = {tuple(row): i for i, row in enumerate(millers)}
        columns = [reciprocal[:, index[(0, 0, 0)]]]
        for row, i in index.items():
            opposite = tuple(-x for x in row)
            if row > opposite:
                left, right = reciprocal[:, i], reciprocal[:, index[opposite]]
                columns.extend(((left + right) / 2, (left - right) / (2j)))
        cross = projection.cart_primitive_normalisations(prims)[:, None] * np.sqrt(volume) * np.column_stack(columns)
        diagonal = np.full(len(columns), volume / 2)
        diagonal[0] = volume
        s_gg = projection.periodic_pgto_overlap_matrix(ref, prims)
        dense = np.block([[s_gg, cross], [cross.conj().T, np.diag(diagonal)]])
        reference_cond = np.linalg.cond(dense)

        eigvalsh = np.linalg.eigvalsh
        eigensystem_sizes = []

        def bounded_eigvalsh(matrix):
            eigensystem_sizes.append(matrix.shape[0])
            assert matrix.shape[0] <= 2 * len(prims) + 1, "unreduced Gamma overlap eigensystem"
            return eigvalsh(matrix)

        monkeypatch.setattr(np.linalg, "eigvalsh", bounded_eigvalsh)
        observed_cond = projection._generated_pgto_overlap_condition(ref, prims, millers)
        assert eigensystem_sizes
        np.testing.assert_allclose(observed_cond, reference_cond, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64)

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("k_point", "shift"),
        [
            pytest.param((0.2, -0.3, 0.1), (1, -2, 1), id="general-integer-shift"),
            pytest.param((0.5, 0.0, 0.0), (-1, 0, 0), id="plus-minus-pi-boundary"),
        ],
    )
    def test_pw_to_pgto_converters_are_reciprocal_gauge_invariant(self, k_point, shift):
        """Pure pGTO and mixed pGTO+PW conversion preserve a joined k+G shift.

        The source pairs use ``k' = k + s`` and ``m' = m - s`` for an integer
        reciprocal shift ``s``, so every physical ``G+k`` row and its QE
        coefficient are unchanged. Both converter outputs must therefore
        preserve pGTO coefficients and residuals; the mixed route must also
        preserve its selected physical PW rows and coefficients. A k-only
        shift is the negative control and must change the mixed conversion.
        """
        from jqmc.pw_pgto_projection import project_pw_to_mixed, project_pw_to_pgto

        cell = QECell(a=2.0 * np.pi * np.eye(3))
        millers = np.array(
            [[i, j, k] for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)],
            dtype=np.int64,
        )
        k_point = np.asarray(k_point, dtype=np.float64)
        shift = np.asarray(shift, dtype=np.int64)
        row = np.arange(millers.shape[0], dtype=np.float64)
        coeffs = (np.cos(0.17 * row) + 1j * np.sin(0.31 * row))[None, :]
        coeffs /= np.linalg.norm(coeffs)
        ref = _toy_reference(cell, millers, (8, 8, 8), ecutwfc=20.0, k_point=k_point, coeffs=coeffs)
        ref_shifted = _toy_reference(cell, millers - shift, (8, 8, 8), ecutwfc=20.0, k_point=k_point + shift, coeffs=coeffs)
        ref_bad = _toy_reference(cell, millers, (8, 8, 8), ecutwfc=20.0, k_point=k_point + shift, coeffs=coeffs)
        shells = [
            [
                PrimitiveShell(
                    angular_l=0,
                    exponents=(0.3, 1.2, 4.0),
                    coefficients=(1.0, 1.0, 1.0),
                    rel_error=0.0,
                    overlap_cond=1.0,
                )
            ]
        ]

        pgto = project_pw_to_pgto(ref, shells)
        pgto_shifted = project_pw_to_pgto(ref_shifted, shells)
        np.testing.assert_allclose(
            pgto_shifted.residuals, pgto.residuals, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64
        )
        np.testing.assert_allclose(
            pgto_shifted.coefficients,
            pgto.coefficients,
            atol=ATOL_PW_PROJECTION_FP64,
            rtol=RTOL_PW_PROJECTION_FP64,
        )

        mixed = project_pw_to_mixed(ref, shells, g_cut=1.8, drop_diffuse=False)
        mixed_shifted = project_pw_to_mixed(ref_shifted, shells, g_cut=1.8, drop_diffuse=False)
        mixed_bad = project_pw_to_mixed(ref_bad, shells, g_cut=1.8, drop_diffuse=False)

        def ordered_physical_pw(result, twist):
            q_vectors = result.pw_millers.astype(np.float64) + twist[None, :]
            order = np.lexsort((q_vectors[:, 2], q_vectors[:, 1], q_vectors[:, 0]))
            return q_vectors[order], result.pw_coefficients[:, order]

        q_mixed, coeff_mixed = ordered_physical_pw(mixed, k_point)
        q_shifted, coeff_shifted = ordered_physical_pw(mixed_shifted, k_point + shift)
        np.testing.assert_allclose(q_shifted, q_mixed, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64)
        np.testing.assert_allclose(
            mixed_shifted.residuals, mixed.residuals, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64
        )
        np.testing.assert_allclose(
            mixed_shifted.coefficients, mixed.coefficients, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64
        )
        np.testing.assert_allclose(coeff_shifted, coeff_mixed, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64)
        assert abs(float(mixed_bad.residuals[0] - mixed.residuals[0])) > 1.0e-2

    @pytest.mark.light
    def test_truncated_pw_converter_is_reciprocal_gauge_invariant(self):
        """A truncated PW conversion cuts on physical G+k, not Miller-label G.

        Joined integer transport of k and every Miller row must preserve the
        retained rows, coefficients, and discarded spectral weight. The k-only
        negative control must select a different physical subspace.
        """
        from jqmc.pw_pgto_projection import project_pw_to_pw

        cell = QECell(a=2.0 * np.pi * np.eye(3))
        millers = np.array(
            [[i, j, k] for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)],
            dtype=np.int64,
        )
        k_point = np.array([0.2, -0.3, 0.1])
        shift = np.array([1, -2, 1], dtype=np.int64)
        row = np.arange(millers.shape[0], dtype=np.float64)
        coeffs = (np.cos(0.17 * row) + 1j * np.sin(0.31 * row))[None, :]
        coeffs /= np.linalg.norm(coeffs)
        ref = _toy_reference(cell, millers, (8, 8, 8), k_point=k_point, coeffs=coeffs)
        ref_shifted = _toy_reference(cell, millers - shift, (8, 8, 8), k_point=k_point + shift, coeffs=coeffs)
        ref_bad = _toy_reference(cell, millers, (8, 8, 8), k_point=k_point + shift, coeffs=coeffs)

        projected = project_pw_to_pw(ref, g_cut=1.8)
        projected_shifted = project_pw_to_pw(ref_shifted, g_cut=1.8)
        projected_bad = project_pw_to_pw(ref_bad, g_cut=1.8)

        def ordered_physical_pw(result, twist):
            q_vectors = result.pw_millers.astype(np.float64) + twist[None, :]
            order = np.lexsort((q_vectors[:, 2], q_vectors[:, 1], q_vectors[:, 0]))
            return q_vectors[order], result.pw_coefficients[:, order]

        q_projected, coeff_projected = ordered_physical_pw(projected, k_point)
        q_shifted, coeff_shifted = ordered_physical_pw(projected_shifted, k_point + shift)
        np.testing.assert_allclose(q_shifted, q_projected, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64)
        np.testing.assert_array_equal(coeff_shifted, coeff_projected)
        np.testing.assert_allclose(
            projected_shifted.residuals,
            projected.residuals,
            atol=ATOL_PW_PROJECTION_FP64,
            rtol=RTOL_PW_PROJECTION_FP64,
        )
        assert projected_bad.n_pw != projected.n_pw

    @pytest.mark.light
    def test_build_pgto_primitives_counts(self):
        # One atom with an s shell (2 exps) and a p shell (3 exps).
        """Build pgto primitives counts."""
        s = PrimitiveShell(angular_l=0, exponents=(1.0, 3.0), coefficients=(1.0, 1.0), rel_error=0.0, overlap_cond=1.0)
        p = PrimitiveShell(
            angular_l=1, exponents=(0.5, 2.0, 8.0), coefficients=(1.0, 1.0, 1.0), rel_error=0.0, overlap_cond=1.0
        )
        pos = np.zeros((1, 3))
        prims = build_pgto_primitives([[s, p]], pos)
        # s: 1 component * 2 exps = 2; p: 3 components * 3 exps = 9.
        assert len(prims) == 2 + 9
        assert sum(1 for q in prims if q.ang == (0, 0, 0)) == 2
        assert sum(1 for q in prims if sum(q.ang) == 1) == 9

    @pytest.mark.light
    def test_build_pgto_primitives_length_mismatch(self):
        """Build pgto primitives length mismatch."""
        s = PrimitiveShell(angular_l=0, exponents=(1.0,), coefficients=(1.0,), rel_error=0.0, overlap_cond=1.0)
        with pytest.raises(ValueError):
            build_pgto_primitives([[s]], np.zeros((2, 3)))

    @pytest.mark.light
    def test_build_pgto_primitives_is_uncontracted_and_ignores_pswfc_fit_coefficients(self):
        """Build pgto primitives is uncontracted and ignores pswfc fit coefficients."""
        shell_a = PrimitiveShell(
            angular_l=0,
            exponents=(0.5, 2.0),
            coefficients=(1.0, 1.0),
            rel_error=0.0,
            overlap_cond=1.0,
        )
        shell_b = PrimitiveShell(
            angular_l=0,
            exponents=(0.5, 2.0),
            coefficients=(-7.0, 123.0),
            rel_error=0.0,
            overlap_cond=1.0,
        )
        position = np.zeros((1, 3), dtype=np.float64)

        assert build_pgto_primitives([[shell_a]], position) == build_pgto_primitives([[shell_b]], position)

    @pytest.mark.light
    def test_eval_primitive_periodicity(self):
        # A periodic Gaussian must be invariant under a lattice translation.
        """Eval primitive periodicity."""
        a = 6.0 * np.eye(3)
        cell = QECell(a=a)
        grid = grid_points(cell, (8, 8, 8))
        prim = CartPrimitive(center=(1.0, 2.0, 3.0), ang=(0, 0, 0), exponent=1.2)
        vals = eval_primitive_on_grid(cell, prim, grid)
        shifted = CartPrimitive(center=(1.0 + 6.0, 2.0, 3.0), ang=(0, 0, 0), exponent=1.2)
        vals_shift = eval_primitive_on_grid(cell, shifted, grid)
        np.testing.assert_allclose(vals, vals_shift, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    def test_grid_points_shape(self):
        """Grid points shape."""
        cell = QECell(a=4.0 * np.eye(3))
        grid = grid_points(cell, (5, 6, 7))
        assert grid.shape == (5 * 6 * 7, 3)

    @pytest.mark.light
    def test_build_gto_design_matrix_batched_matches_per_primitive(self):
        """The grouped ``build_gto_design_matrix`` (radial reuse across the
        Cartesian angular components of a (center, alpha) group) must agree with
        the simpler per-primitive ``eval_primitive_on_grid`` reference (numpy,
        loops over images one at a time). The test mixes (a) a tight primitive
        (1 image), (b) a very diffuse primitive (hundreds of images), and (c) p / d
        angular components so the polynomial factor and the per-group radial reuse
        are both exercised. Both paths are the real-space ``*_debug`` reference --
        production uses the Fourier projection (:func:`build_gto_pw_matrix`).
        """
        from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL
        from jqmc.pw_pgto_projection import (
            build_gto_design_matrix,
            eval_primitive_on_grid,
        )

        cell = QECell(a=8.0 * np.eye(3))
        grid = grid_points(cell, (12, 12, 12))
        prims = [
            # Tight s: 1 image -> per-image masked path.
            CartPrimitive(center=(2.0, 2.0, 2.0), ang=(0, 0, 0), exponent=4.0),
            # Diffuse s: many images -> batched path.
            CartPrimitive(center=(2.0, 2.0, 2.0), ang=(0, 0, 0), exponent=0.06),
            # Diffuse p, different center: tests poly factor + batched path.
            CartPrimitive(center=(5.0, 5.0, 5.0), ang=(1, 0, 0), exponent=0.3),
            CartPrimitive(center=(5.0, 5.0, 5.0), ang=(0, 1, 0), exponent=0.3),
            CartPrimitive(center=(5.0, 5.0, 5.0), ang=(0, 0, 1), exponent=0.3),
            # Diffuse d (full poly: nx,ny,nz can be 2).
            CartPrimitive(center=(3.0, 5.0, 6.0), ang=(2, 0, 0), exponent=0.15),
            CartPrimitive(center=(3.0, 5.0, 6.0), ang=(1, 1, 0), exponent=0.15),
        ]
        design = build_gto_design_matrix(cell, prims, grid, tol=AO_IMAGE_TRUNCATION_ATOL)
        ref = np.empty((grid.shape[0], len(prims)), dtype=np.float64)
        for i, p in enumerate(prims):
            ref[:, i] = eval_primitive_on_grid(cell, p, grid, tol=AO_IMAGE_TRUNCATION_ATOL)
        # Tolerance reflects the image-sum order differences between the
        # grouped path (per-group accumulation over images) and the naive
        # per-image += accumulation. Both are within float64 round-off but
        # bit-equal is too strict.
        np.testing.assert_allclose(design.real, ref, atol=ATOL_PW_PROJECTION_FP64_LOOSE, rtol=ATOL_PW_PROJECTION_FP64_LOOSE)
        np.testing.assert_allclose(design.imag, 0.0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    @requires_qe
    def test_reconstruct_pw_orbitals_norm(self):
        """Reconstruct pw orbitals norm."""
        from jqmc.pw_pgto_projection import reconstruct_pw_orbitals
        from jqmc.qe_wrapper import load_qe_reference

        ref = load_qe_reference(str(_SAVE_DIR))
        psi = reconstruct_pw_orbitals(ref.wfc, ref.fft_grid)
        dV = ref.cell.volume / psi.shape[0]
        # QE normalises the PW coefficients to sum_G |c_G|^2 = 1; reconstructing
        # psi(r) = sum_G c_G exp(iG.r) without the 1/sqrt(Omega) Bloch factor gives
        # a grid norm equal to the cell volume. (The projection residual is
        # normalisation-invariant, so this overall factor is irrelevant downstream.)
        norms = np.real(np.einsum("ij,ij->j", psi.conj(), psi)) * dV
        # The reference has the explicit shape of the five checked bands (no
        # scalar broadcast), so a fixture with fewer bands fails here.
        np.testing.assert_allclose(
            norms[:5], np.full(5, ref.cell.volume), atol=ATOL_BIT_EXACT, rtol=RTOL_PW_PROJECTION_FP64_NORM
        )

    @pytest.mark.light
    @requires_qe
    def test_project_pw_to_pgto_li_fidelity(self):
        """Project pw to pgto li fidelity."""
        from jqmc.pw_pgto_projection import project_pw_to_pgto
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.upf_wrapper import UPFExponentInit

        ref = load_qe_reference(str(_SAVE_DIR))
        # bcc Li fixture has 2 Li atoms in the conventional cell.
        assert len(ref.atom_labels) == 2
        from jqmc.pw_pgto_projection import build_pgto_primitives

        strat = UPFExponentInit(
            {"li": str(_SAVE_DIR / "Li.ccECP.upf")},
            g_target=float(np.sqrt(2.0 * ref.ecutwfc)),
        )
        shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]
        res = project_pw_to_pgto(ref, shells)
        # n_gto matches the all-primitive enumeration of the shells emitted by
        # UPFExponentInit. The exact count depends on the per-shell exponent
        # list, which includes the always-on tight-primitive extension
        # (jqmc._setting.PGTO_TIGHT_*); don't hardcode it.
        expected_n_gto = len(build_pgto_primitives(shells, ref.atom_positions))
        assert res.n_gto == expected_n_gto
        # The UPF-seeded pGTO basis reproduces the occupied bands well.
        assert res.mean_residual < ATOL_QE_PROJECTION_FIDELITY_PGTO_ONLY_MEAN
        assert res.max_residual < ATOL_QE_PROJECTION_FIDELITY_PGTO_ONLY_MAX
        assert res.residuals.shape == res.band_indices.shape
        # The conversion output: pGTO coefficients over the emitted basis.
        assert res.coefficients.shape == (res.band_indices.size, res.n_gto)
        assert len(res.primitives) == res.n_gto
        assert np.all(np.isfinite(res.coefficients))

    @pytest.mark.light
    @requires_qe
    def test_projection_coefficients_reproduce_residual(self):
        # The returned (primitives, coefficients) must reconstruct each band to the
        # same fidelity the residual reports: 1 - ||proj||^2/||psi||^2 == residual.
        # Production projects in PW space (the QE reciprocal lattice), so the
        # reconstruction is checked there: the Fourier residual is a strictly
        # band-limited quantity -- reconstructing it on the dense real grid would
        # (correctly) report a larger error because the real grid also sees the
        # pGTO's high-G content beyond ecutwfc, which the band-limited projection
        # neither fits nor counts.
        """Projection coefficients reproduce residual."""
        from jqmc.pw_pgto_projection import (
            _psi_pw_matrix,
            build_gto_pw_matrix,
            project_pw_to_pgto,
        )
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.upf_wrapper import UPFExponentInit

        ref = load_qe_reference(str(_SAVE_DIR))
        strat = UPFExponentInit(
            {"li": str(_SAVE_DIR / "Li.ccECP.upf")},
            g_target=float(np.sqrt(2.0 * ref.ecutwfc)),
        )
        shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]
        res = project_pw_to_pgto(ref, shells)

        # PW-space reconstruction: approx[g,n] = sum_mu G[g,mu] c[n,mu].
        G = build_gto_pw_matrix(ref, list(res.primitives))
        psi = _psi_pw_matrix(ref, res.band_indices)
        approx = G @ res.coefficients.T

        # ||psi - P psi||^2 / ||psi||^2 == residual (P psi is the pGTO reconstruction).
        psi_norm2 = np.real(np.einsum("ij,ij->j", psi.conj(), psi))
        diff = psi - approx
        err2 = np.real(np.einsum("ij,ij->j", diff.conj(), diff))
        np.testing.assert_allclose(
            err2 / psi_norm2, res.residuals, atol=ATOL_PW_PROJECTION_FP64_SPLINE_RESIDUAL, rtol=RTOL_PYTHON_FP64_LOOSE
        )

    @pytest.mark.light
    def test_alpha_min_from_gcut_monotone_and_floor(self):
        """Alpha min from gcut monotone and floor."""
        from jqmc.pw_pgto_projection import alpha_min_from_gcut

        # Disabled for non-positive cutoff (keeps every primitive).
        assert alpha_min_from_gcut(0.0) == 0.0
        # Closed form alpha_min = g_cut^2 / (4 ln(1/tol)).
        g, tol = 3.0, 1.0e-3
        assert alpha_min_from_gcut(g, tol) == pytest.approx(g**2 / (4.0 * -np.log(tol)))
        # Monotonically increasing in g_cut: a larger PW shell sheds more diffuse pGTO.
        a1 = alpha_min_from_gcut(2.0)
        a2 = alpha_min_from_gcut(4.0)
        assert a2 > a1 > 0.0

    @pytest.mark.light
    def test_filter_shells_by_alpha_min_drops_diffuse(self):
        """Filter shells by alpha min drops diffuse."""
        from jqmc.pw_pgto_projection import filter_shells_by_alpha_min

        s = PrimitiveShell(
            angular_l=0, exponents=(0.1, 0.5, 4.0), coefficients=(1.0, 1.0, 1.0), rel_error=0.0, overlap_cond=1.0
        )
        p = PrimitiveShell(angular_l=1, exponents=(0.2, 0.3), coefficients=(1.0, 1.0), rel_error=0.0, overlap_cond=1.0)
        out = filter_shells_by_alpha_min([[s, p]], alpha_min=0.5)
        # s keeps only exponents >= 0.5 -> (0.5, 4.0); p loses all -> shell dropped.
        assert len(out[0]) == 1
        assert out[0][0].angular_l == 0
        assert out[0][0].exponents == (0.5, 4.0)
        # alpha_min <= 0 is a no-op copy.
        same = filter_shells_by_alpha_min([[s, p]], alpha_min=0.0)
        assert [sh.exponents for sh in same[0]] == [s.exponents, p.exponents]

    @pytest.mark.light
    def test_build_pw_design_matrix_orthogonal(self):
        # Distinct reciprocal modes are orthogonal over the grid: Phi^H Phi dV = V*I.
        """Build pw design matrix orthogonal."""
        from jqmc.pw_pgto_projection import build_pw_design_matrix

        cell = QECell(a=5.0 * np.eye(3))
        grid = grid_points(cell, (8, 8, 8))
        dV = cell.volume / grid.shape[0]
        b = cell.b
        millers = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 2]])
        g = millers.astype(float) @ b
        phi = build_pw_design_matrix(grid, g)
        gram = (phi.conj().T @ phi) * dV
        np.testing.assert_allclose(
            gram, cell.volume * np.eye(len(millers)), atol=ATOL_PW_PROJECTION_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE
        )

    @pytest.mark.light
    @requires_qe
    def test_optimize_mixed_basis_reduces_pw_and_residual(self):
        """Optimize mixed basis reduces pw and residual.

        The starting point is chosen so the sweep cannot stop at its first trial:
        a deliberately coarse pGTO block (two primitives per UPF
        channel, tight extension only to g = 1 Bohr^-1; its own residual is
        ~8e-4, above the unchanged 1e-4 target) and a g_cut ladder that starts at
        0.2 g_max. The full UPF basis already meets the target without any plane
        wave, so with it the trend loop below would never run.
        """
        from jqmc.pw_pgto_projection import optimize_mixed_basis, project_pw_to_pgto
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.upf_wrapper import UPFExponentInit

        ref = load_qe_reference(str(_SAVE_DIR))
        g_max = float(np.sqrt(2.0 * ref.ecutwfc))
        strat = UPFExponentInit({"li": str(_SAVE_DIR / "Li.ccECP.upf")}, g_target=1.0, n_max=2)
        shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]

        pgto_only = project_pw_to_pgto(ref, shells)
        sweep = optimize_mixed_basis(ref, shells, residual_target=1e-4, g_cut_values=np.geomspace(0.2 * g_max, g_max, 6))

        # Residual decreases overall as the ladder grows. With drop_diffuse=True
        # (legacy default) the per-trial alpha_min varies with g_cut, so the
        # basis CHANGES between trials -- this can introduce small
        # non-monotonic blips (a tighter g_cut sheds a primitive whose
        # contribution exceeds the gain from the new PW shells). For strict
        # monotonicity use fixed_alpha_min (see
        # test_optimize_mixed_basis_fixed_alpha_min_monotone). Here we accept
        # small wiggles and check the trend instead.
        res_seq = [t.mean_residual for t in sweep.trials]
        slack = 0.05  # allow per-step backsteps up to 5% (empirically ~1% on Cu fixture)
        trend_steps = 0
        for i in range(len(res_seq) - 1):
            bound = res_seq[i] * (1.0 + slack) + 1.0e-12
            assert res_seq[i + 1] <= bound, (
                f"residual blip too large at trial {i + 1}: {res_seq[i + 1]:.3e} > {bound:.3e} (prev {res_seq[i]:.3e})"
            )
            trend_steps += 1
        # The trend check above must not be vacuous: the sweep ran at least two trials.
        assert trend_steps >= 1, f"the sweep stopped at its first trial (residuals {res_seq}); the trend loop never ran"
        assert res_seq[-1] <= res_seq[0]  # overall decrease
        pw_seq = [t.n_pw for t in sweep.trials]
        assert all(pw_seq[i] <= pw_seq[i + 1] for i in range(len(pw_seq) - 1))

        # PW reduction: the smallest-g_cut trial appends only a handful of low-G
        # plane waves yet already improves on (or matches) the pGTO-only residual,
        # using far fewer PWs than the full wfc G set.
        first = sweep.trials[0]
        assert 0 < first.n_pw < ref.wfc.n_gvectors
        assert first.mean_residual <= pgto_only.mean_residual + ATOL_PW_PROJECTION_FP64_GRID

        best = sweep.best
        assert best.n_pw > 0
        assert best.mean_residual <= pgto_only.mean_residual + ATOL_PYTHON_FP64_LOOSE
        # Coefficient bookkeeping: pGTO and PW parts both returned, shapes consistent.
        assert best.coefficients.shape == (best.band_indices.size, best.n_gto)
        assert best.pw_coefficients.shape == (best.band_indices.size, best.n_pw)
        assert best.pw_millers.shape == (best.n_pw, 3)
        if sweep.target_met:
            assert best.mean_residual <= ATOL_QE_PROJECTION_FIDELITY_SWEEP_TARGET

    # ---------------------------------------------------------------------------
    # B-spline machinery (grid-free, no QE data required)
    # ---------------------------------------------------------------------------
    @pytest.mark.light
    @pytest.mark.parametrize("order", [3, 5])
    def test_bspline_weights_partition_of_unity(self, order):
        """Bspline weights partition of unity."""
        from jqmc.pw_pgto_projection import _BSPLINE_INT_STENCIL, _bspline_weights

        f = np.linspace(0.0, 1.0, 11)
        w = _bspline_weights(f, order)
        assert w.shape == (11, order + 1)
        # Weights are a partition of unity at every fractional offset.
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
        # At a knot (f=0) they collapse to the integer-sample stencil (the extra
        # trailing control point carries zero weight there).
        expected = np.array(_BSPLINE_INT_STENCIL[order] + (0.0,))
        np.testing.assert_allclose(w[0], expected, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    @pytest.mark.parametrize("order", [3, 5])
    def test_spline_interpolates_at_knots(self, order):
        """Spline interpolates at knots."""
        from jqmc.pw_pgto_projection import bspline_prefilter_periodic, evaluate_spline_orbitals

        shape = (8, 10, 12)
        gx, gy, gz = np.meshgrid(
            np.arange(shape[0]) / shape[0],
            np.arange(shape[1]) / shape[1],
            np.arange(shape[2]) / shape[2],
            indexing="ij",
        )
        # A smooth periodic field (band-limited well below Nyquist).
        field = (np.cos(2 * np.pi * gx) + 0.5 * np.sin(2 * np.pi * gy) + 0.25 * np.cos(2 * np.pi * (gx + gz))).astype(
            np.complex128
        )
        coeffs = bspline_prefilter_periodic(field[None], order)
        recon = evaluate_spline_orbitals(coeffs, shape, order, shift_frac=(0.0, 0.0, 0.0))
        np.testing.assert_allclose(recon[:, 0], field.reshape(-1), atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    @requires_qe
    def test_project_pw_to_pw_exact_and_truncation(self):
        """Project pw to pw exact and truncation."""
        from jqmc.pw_pgto_projection import project_pw_to_pw
        from jqmc.qe_wrapper import load_qe_reference

        ref = load_qe_reference(str(_SAVE_DIR))

        full = project_pw_to_pw(ref, g_cut=None)
        # Keeping every plane wave is exact.
        assert full.n_pw == ref.wfc.n_gvectors
        assert full.n_gto == 0
        np.testing.assert_allclose(full.residuals, 0.0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)

        gnorm = np.linalg.norm(ref.wfc.g_vectors(ref.cell), axis=1)
        g_cut = 0.5 * float(gnorm.max())
        trunc = project_pw_to_pw(ref, g_cut=g_cut)
        assert 0 < trunc.n_pw < ref.wfc.n_gvectors
        assert trunc.pw_coefficients.shape == (trunc.band_indices.size, trunc.n_pw)
        assert trunc.pw_millers.shape == (trunc.n_pw, 3)
        # Residual equals the discarded spectral-power fraction (independent check).
        c = ref.wfc.coeffs[trunc.band_indices]
        keep = gnorm <= g_cut
        expect = 1.0 - np.real(np.einsum("ng,ng->n", c[:, keep].conj(), c[:, keep])) / np.real(
            np.einsum("ng,ng->n", c.conj(), c)
        )
        np.testing.assert_allclose(trunc.residuals, expect, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
        assert trunc.mean_residual > full.mean_residual

    @pytest.mark.light
    @requires_qe
    def test_project_pw_to_spline_fidelity_and_mesh(self):
        """Project pw to spline fidelity and mesh."""
        from jqmc.pw_pgto_projection import project_pw_to_spline
        from jqmc.qe_wrapper import load_qe_reference

        ref = load_qe_reference(str(_SAVE_DIR))

        fine = project_pw_to_spline(ref, mesh_factor=1.0, order=5)
        assert fine.n_gto == 0 and fine.n_pw == 0
        assert fine.spline_order == 5
        assert fine.spline_shape == ref.fft_grid
        assert fine.spline_coefficients.shape == (
            fine.band_indices.size,
            int(np.prod(ref.fft_grid)),
        )
        # Quintic spline on the native mesh resolves the bands well.
        assert fine.mean_residual < ATOL_QE_PROJECTION_FIDELITY_SPLINE_ONLY

        coarse = project_pw_to_spline(ref, mesh_factor=0.6, order=5)
        # Coarsening the grid is the spline analogue of lowering the PW cutoff:
        # accuracy degrades monotonically.
        assert coarse.mean_residual > fine.mean_residual

    @pytest.mark.light
    @requires_qe
    def test_project_pw_to_pgto_spline_additive_beats_parts(self):
        """Project pw to pgto spline additive beats parts."""
        from jqmc.pw_pgto_projection import (
            evaluate_pgto_expansion,
            evaluate_spline_orbitals,
            grid_points,
            project_pw_to_pgto,
            project_pw_to_pgto_spline,
            reconstruct_pw_on_grid,
        )
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.upf_wrapper import UPFExponentInit

        ref = load_qe_reference(str(_SAVE_DIR))
        strat = UPFExponentInit(
            {"li": str(_SAVE_DIR / "Li.ccECP.upf")},
            g_target=float(np.sqrt(2.0 * ref.ecutwfc)),
        )
        shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]

        pgto_only = project_pw_to_pgto(ref, shells)
        mixed = project_pw_to_pgto_spline(ref, shells, mesh_factor=1.0, order=5)

        # Both the pGTO block and a whole-cell spline residual are returned.
        assert mixed.n_gto == pgto_only.n_gto
        assert mixed.spline_order == 5
        assert mixed.spline_shape == ref.fft_grid
        assert mixed.coefficients.shape == (mixed.band_indices.size, mixed.n_gto)
        assert mixed.spline_coefficients.shape == (
            mixed.band_indices.size,
            int(np.prod(ref.fft_grid)),
        )
        # Adding the spline residual strictly improves on the pGTO-only fit and is
        # very accurate overall.
        assert mixed.mean_residual < pgto_only.mean_residual
        assert mixed.mean_residual < ATOL_QE_PROJECTION_FIDELITY_SWEEP_TARGET

        # Independent reconstruction at a knot (no shift): pGTO + spline reproduces
        # the band values (the spline interpolates the residual exactly at knots).
        # The pGTO half MUST be reconstructed in real space (image-summed), which
        # is what jQMC's pGTO leaf does -- the spline now encodes psi minus the
        # full real-space pGTO, so the high-G pGTO tail cancels cleanly. Using the
        # band-limited Fourier IFFT here would leave that tail uncancelled and was
        # the prior self-measurement skew that hid a ~1% per-band orbital loss.
        sp = evaluate_spline_orbitals(
            mixed.spline_coefficients.reshape((mixed.band_indices.size, *mixed.spline_shape)),
            ref.fft_grid,
            mixed.spline_order,
        )
        # The converter emits the unit-cell-normalized band, i.e. the
        # raw ``sum_G c_G e^{iG.r}`` reconstruction divided by sqrt(Omega); the
        # comparison is on absolute values, so the reference carries that factor.
        psi = reconstruct_pw_on_grid(ref.wfc, ref.fft_grid)[:, mixed.band_indices] / np.sqrt(ref.cell.volume)
        grid_fft = grid_points(ref.cell, ref.fft_grid)
        pg = evaluate_pgto_expansion(ref.cell, list(mixed.primitives), mixed.coefficients, grid_fft)
        rel = np.linalg.norm(pg + sp - psi) / np.linalg.norm(psi)
        # Floor is the AO image-truncation tol (AO_IMAGE_TRUNCATION_ATOL=1e-7);
        # the additive pGTO + spline residual still nails the band values an order
        # of magnitude below that.
        assert rel < ATOL_QE_PROJECTION_RECONSTRUCTION_RESIDUAL

        # The alpha_min floor hands the diffuse tail to the spline (fewer pGTO).
        # The exact residual after flooring is fixture-specific (the Cu basis
        # had enough tight shells that floored.mean_residual stayed below 1e-3;
        # the Li basis is dominated by diffuse shells so the floor strips out
        # most of the meaningful pGTO content and the spline alone carries the
        # band -- residual is dominated by the spline approximation rather than
        # the pGTO quality). The shape contract (fewer pGTO) holds either way.
        floored = project_pw_to_pgto_spline(ref, shells, mesh_factor=1.0, order=5, alpha_min=1.0)
        assert floored.n_gto < mixed.n_gto
        assert np.isfinite(floored.mean_residual)

    # ---------------------------------------------------------------------------
    # --ecutwfc auto pathway (overlap-based auto cutoff selection)
    # ---------------------------------------------------------------------------
    @pytest.mark.light
    @requires_qe
    def test_optimize_mixed_basis_fixed_alpha_min_monotone(self):
        """fixed_alpha_min holds the pGTO floor constant -> strict monotone sweep.

        The ``drop_diffuse=True`` path can wiggle the residual because it
        drops diffuse pGTO as ``g_cut`` grows. ``fixed_alpha_min`` is the contract
        underlying ``--ecutwfc auto``: residual must be monotone non-increasing
        along the geometric ``g_cut`` ladder so that first-hit = cheapest
        sufficient PW block. A zero target keeps all six ladder points live
        for this committed Li input; a 1e-4 target would stop after one
        point, making every adjacent-point comparison vacuous. The existing
        1e-12 comparison allowance is unchanged.
        """
        from jqmc.pw_pgto_projection import optimize_mixed_basis
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.upf_wrapper import UPFExponentInit

        ref = load_qe_reference(str(_SAVE_DIR))
        strat = UPFExponentInit(
            {"li": str(_SAVE_DIR / "Li.ccECP.upf")},
            g_target=float(np.sqrt(2.0 * ref.ecutwfc)),
        )
        shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]

        sweep = optimize_mixed_basis(
            ref,
            shells,
            residual_target=0.0,
            n_gcut=6,
            fixed_alpha_min=0.0,
        )
        res_seq = [t.mean_residual for t in sweep.trials]
        # Strictly monotone (with a tiny epsilon for float noise) under fixed alpha_min.
        assert len(res_seq) == 6
        assert np.all(np.isfinite(res_seq))
        for current, following in itertools.pairwise(res_seq):
            assert current >= following - ATOL_PYTHON_FP64_LOOSE
        # All trials retain every pGTO column (fixed_alpha_min=0.0 keeps the lot).
        n_gto_seq = [t.n_gto for t in sweep.trials]
        assert len(n_gto_seq) == len(res_seq)
        for count in n_gto_seq:
            assert count == n_gto_seq[0]
        # The PW count is strictly non-decreasing.
        pw_seq = [t.n_pw for t in sweep.trials]
        assert len(pw_seq) == len(res_seq)
        for current, following in itertools.pairwise(pw_seq):
            assert current <= following

    @pytest.mark.light
    @requires_cli_fixture
    def test_qe2jqmc_ecutwfc_auto_rejects_wrong_kind(self, tmp_path):
        """``ecutwfc='auto'`` is meaningful only for ``kind='pgto+pw'``."""
        from jqmc.jqmc_qe import run_qe2jqmc

        with pytest.raises(ValueError, match="only meaningful for kind='pgto\\+pw'"):
            run_qe2jqmc(
                qe_params={
                    "qe_save_dir": str(_CLI_FIXTURE),
                    "kind": "pw",
                    "upf_paths": [f"Li={_CLI_FIXTURE / 'Li.ccECP.upf'}"],
                    "ecutwfc": "auto",
                },
                jastrow_params={},
                hamiltonian_h5=str(tmp_path / "should_not_be_written.h5"),
            )

    @pytest.mark.light
    @requires_cli_fixture
    def test_qe2jqmc_ecutwfc_auto_end_to_end(self, tmp_path, monkeypatch):
        """End-to-end auto sweep through the new ``jqmc_qe`` entry point.

        The Cu fixture's ``ecutwfc=75 Ha`` cannot reach the default ``1e-4``
        residual within the 30 Ry auto cap (covered by the sibling
        ``test_qe2jqmc_ecutwfc_auto_raises_when_target_unreachable``);
        monkeypatch the target to ``5e-3`` so the sweep clears at the first
        ladder point and we exercise the MET branch.
        """
        from jqmc import _setting
        from jqmc.jqmc_qe import run_qe2jqmc

        monkeypatch.setattr(_setting, "PW_PGTO_AUTO_RESIDUAL_TARGET", 5.0e-3)

        out_file = tmp_path / "cu_auto.h5"
        run_qe2jqmc(
            qe_params={
                "qe_save_dir": str(_CLI_FIXTURE),
                "kind": "pgto+pw",
                "upf_paths": [f"Li={_CLI_FIXTURE / 'Li.ccECP.upf'}"],
                "ecutwfc": "auto",
            },
            jastrow_params={"j2_parameter": 1.0},
            hamiltonian_h5=str(out_file),
        )
        assert out_file.is_file()

    @pytest.mark.light
    @requires_cli_fixture
    def test_qe2jqmc_ecutwfc_auto_raises_when_target_unreachable(self, tmp_path, monkeypatch):
        """Auto sweep cannot meet the residual target within the cap ->
        ``run_qe2jqmc`` must raise ``RuntimeError`` pointing the user at
        manual ``ecutwfc``. Forced by monkeypatching the target below any
        achievable Cu residual.
        """
        from jqmc import _setting
        from jqmc.jqmc_qe import run_qe2jqmc

        monkeypatch.setattr(_setting, "PW_PGTO_AUTO_RESIDUAL_TARGET", 1.0e-20)

        out_file = tmp_path / "should_not_be_written.h5"
        with pytest.raises(RuntimeError, match="overlap residual did not reach"):
            run_qe2jqmc(
                qe_params={
                    "qe_save_dir": str(_CLI_FIXTURE),
                    "kind": "pgto+pw",
                    "upf_paths": [f"Li={_CLI_FIXTURE / 'Li.ccECP.upf'}"],
                    "ecutwfc": "auto",
                },
                jastrow_params={"j2_parameter": 1.0},
                hamiltonian_h5=str(out_file),
            )
        assert not out_file.exists(), "no HDF5 should be written when auto-cutoff fails"

    # ---------------------------------------------------------------------------
    # Schur-complement projector (PW-block-is-V*I exploit)
    # ---------------------------------------------------------------------------
    @pytest.mark.light
    def test_project_from_blocks_matches_full_eigh(self):
        """``_project_from_blocks`` and ``_project_from_S_b`` must agree to round-off.

        Verifies the Schur-complement path mathematically reproduces the legacy
        assemble-then-eigh path on a small randomised problem. The Schur path is
        the production code path for mixed-basis projections; the full path is
        kept only for the pGTO-only kind. Without this equivalence test a
        regression in either would silently corrupt the converter.

        The basis matrices are sampled so that the assembled Gram matrix S =
        Phi^H Phi is genuinely positive semidefinite with the
        ``S_pp = V * I`` block structure -- a randomised
        ``[S_gg | S_gp; S_gp^H | V*I]`` is *not* guaranteed PSD (the Schur
        complement can flip sign), which would let the two paths produce
        different pseudoinverses and break the comparison even when both
        implementations are correct.
        """
        from jqmc.pw_pgto_projection import _project_from_blocks, _project_from_S_b

        rng = np.random.default_rng(20260609)
        n_grid, n_gto, n_pw, n_band = 4000, 12, 200, 6
        V = 17.3

        # Orthonormal PW columns scaled so Phi_pw^H Phi_pw = V * I (matches the
        # _mixed_result_from_design contract: PW columns are orthogonal on the
        # uniform FFT grid up to the cell volume).
        raw_pw = rng.standard_normal((n_grid, n_pw)) + 1j * rng.standard_normal((n_grid, n_pw))
        Q, _ = np.linalg.qr(raw_pw)
        Phi_pw = Q * np.sqrt(V)

        Phi_gto = (rng.standard_normal((n_grid, n_gto)) + 1j * rng.standard_normal((n_grid, n_gto))) * 0.5
        psi = (rng.standard_normal((n_grid, n_band)) + 1j * rng.standard_normal((n_grid, n_band))).astype(np.complex128)

        S_gg = Phi_gto.conj().T @ Phi_gto
        S_gp = Phi_gto.conj().T @ Phi_pw
        b_g = Phi_gto.conj().T @ psi
        b_p = Phi_pw.conj().T @ psi
        psi_norm2 = np.real(np.einsum("ij,ij->j", psi.conj(), psi))

        # Reference path: assemble the full S and call _project_from_S_b.
        n_basis = n_gto + n_pw
        S_full = np.zeros((n_basis, n_basis), dtype=np.complex128)
        S_full[:n_gto, :n_gto] = S_gg
        S_full[:n_gto, n_gto:] = S_gp
        S_full[n_gto:, :n_gto] = S_gp.conj().T
        np.fill_diagonal(S_full[n_gto:, n_gto:], V)
        b_full = np.empty((n_basis, n_band), dtype=np.complex128)
        b_full[:n_gto] = b_g
        b_full[n_gto:] = b_p

        res_ref, coeffs_ref, _, _, _, _ = _project_from_S_b(S_full, b_full, psi_norm2, svd_rcond=1.0e-12)
        coeff_gto_ref = coeffs_ref[:, :n_gto]
        coeff_pw_ref = coeffs_ref[:, n_gto:]

        res_new, coeff_gto_new, coeff_pw_new, _, _, _, _ = _project_from_blocks(
            S_gg,
            S_gp,
            V,
            b_g,
            b_p,
            psi_norm2,
            svd_rcond=1.0e-12,
        )

        np.testing.assert_allclose(res_new, res_ref, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64)
        np.testing.assert_allclose(
            coeff_gto_new, coeff_gto_ref, atol=ATOL_PW_PROJECTION_FP64_GRID, rtol=RTOL_PW_PROJECTION_FP64_GRID
        )
        np.testing.assert_allclose(
            coeff_pw_new, coeff_pw_ref, atol=ATOL_PW_PROJECTION_FP64_GRID, rtol=RTOL_PW_PROJECTION_FP64_GRID
        )

    @pytest.mark.light
    @requires_qe
    def test_mixed_basis_projection_schur_consistency_cu(self):
        """End-to-end consistency on the Cu fixture: residual matches the previous,
        full-eigh-based expectation. Guards against the Schur-path refactor
        silently changing the projection on real wfc data.
        """
        from jqmc.pw_pgto_projection import optimize_mixed_basis
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.upf_wrapper import UPFExponentInit

        ref = load_qe_reference(str(_SAVE_DIR))
        strat = UPFExponentInit(
            {"li": str(_SAVE_DIR / "Li.ccECP.upf")},
            g_target=float(np.sqrt(2.0 * ref.ecutwfc)),
        )
        shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]

        sweep = optimize_mixed_basis(ref, shells, residual_target=1e-4, n_gcut=6)
        best = sweep.best
        # The Cu fixture reaches a machine-zero residual at the largest
        # ladder point (full QE Miller set captured). The Schur and full-eigh
        # paths must both reach the same near-zero residual there.
        assert best.mean_residual < ATOL_QE_PROJECTION_FIDELITY_TIGHT_SWEEP
        assert best.n_pw > 0
        # Projected reconstruction agrees with the original residual within
        # round-off (this is what test_projection_coefficients_reproduce_residual
        # asserts for the pGTO-only path; here we also have a PW block). The
        # reconstruction is done in PW space (the QE reciprocal lattice), the
        # space production projects in.
        from jqmc.pw_pgto_projection import (
            _psi_pw_matrix,
            _wfc_indices_for_millers,
            build_gto_pw_matrix,
        )

        G_gto = build_gto_pw_matrix(ref, list(best.primitives))
        psi = _psi_pw_matrix(ref, best.band_indices)  # (n_pw, n_band)
        approx = G_gto @ best.coefficients.T  # pGTO block
        # PW block: scaled selection E[g, m] = sqrt(Omega) at g == miller m.
        gi = _wfc_indices_for_millers(ref.wfc.millers, best.pw_millers)
        approx[gi, :] += np.sqrt(ref.cell.volume) * best.pw_coefficients.T
        psi_norm2 = np.real(np.einsum("ij,ij->j", psi.conj(), psi))
        diff = psi - approx
        err2 = np.real(np.einsum("ij,ij->j", diff.conj(), diff))
        np.testing.assert_allclose(
            err2 / psi_norm2, best.residuals, atol=ATOL_PW_PROJECTION_FP64_SPLINE_RESIDUAL, rtol=RTOL_PYTHON_FP64_LOOSE
        )

    # ---------------------------------------------------------------------------
    # Fourier-space projection backend
    # ---------------------------------------------------------------------------
    @pytest.mark.light
    def test_build_gto_pw_matrix_matches_fft_of_real(self):
        """Analytic FT (build_gto_pw_matrix) == FFT of the real-space periodic
        Gaussian design matrix sampled at the same Miller indices, up to small
        FFT-grid aliasing. QE-independent equivalence check for the closed-form
        FT (envelope, structure phase, i^n p_n polynomial up to f shells).

        Relation: G_matrix = (sqrt(Omega) / n_grid) * FFT(real phi at millers).
        """
        from jqmc.pw_pgto_projection import (
            _gto_fourier_at_millers,
            build_gto_design_matrix,
            build_gto_pw_matrix,
            grid_points,
        )

        cell = QECell(a=8.0 * np.eye(3))
        grid_shape = (24, 24, 24)
        grid = grid_points(cell, grid_shape)
        n_grid = grid.shape[0]

        prims = [
            CartPrimitive(center=(2.0, 2.0, 2.0), ang=(0, 0, 0), exponent=0.8),
            CartPrimitive(center=(2.0, 2.0, 2.0), ang=(0, 0, 0), exponent=0.3),
            CartPrimitive(center=(5.0, 4.0, 3.0), ang=(1, 0, 0), exponent=0.5),
            CartPrimitive(center=(5.0, 4.0, 3.0), ang=(0, 1, 0), exponent=0.5),
            CartPrimitive(center=(5.0, 4.0, 3.0), ang=(0, 0, 1), exponent=0.5),
            CartPrimitive(center=(3.0, 5.0, 6.0), ang=(2, 0, 0), exponent=0.4),
            CartPrimitive(center=(3.0, 5.0, 6.0), ang=(1, 1, 0), exponent=0.4),
            CartPrimitive(center=(1.0, 1.0, 1.0), ang=(3, 0, 0), exponent=0.6),
        ]
        rng = range(-4, 5)  # within the (24,24,24) Nyquist
        millers = np.array([(i, j, k) for i in rng for j in rng for k in rng], dtype=np.int64)
        ref = _toy_reference(cell, millers, grid_shape)

        G_analytic = build_gto_pw_matrix(ref, prims)
        phi = build_gto_design_matrix(cell, prims, grid, tol=1e-9)
        F = _gto_fourier_at_millers(np.asarray(phi, dtype=np.complex128), grid_shape, millers)
        G_fft = (np.sqrt(cell.volume) / n_grid) * F

        np.testing.assert_allclose(G_analytic, G_fft, atol=ATOL_PW_PROJECTION_FP64_GRID, rtol=RTOL_PW_PROJECTION_FP64_LOOSE)

    @pytest.mark.light
    def test_chunked_fourier_blocks_match_dense(self, monkeypatch):
        """Chunk accumulation reproduces dense G^H G and G^H psi exactly."""
        from jqmc import pw_pgto_projection as projection

        cell = QECell(a=7.0 * np.eye(3))
        millers = np.array(
            [(i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)],
            dtype=np.int64,
        )
        ref = _toy_reference(cell, millers, (12, 12, 12))
        prims = [
            CartPrimitive(center=(1.0, 2.0, 3.0), ang=(0, 0, 0), exponent=0.4),
            CartPrimitive(center=(4.0, 3.0, 2.0), ang=(1, 0, 0), exponent=1.2),
        ]
        rng = np.random.default_rng(20260803)
        psi = rng.normal(size=(millers.shape[0], 3)) + 1j * rng.normal(size=(millers.shape[0], 3))
        dense = projection.build_gto_pw_matrix(ref, prims, millers=millers)

        # Two complex columns permit only three rows per chunk.
        monkeypatch.setattr(projection, "PW_PGTO_FOURIER_CHUNK_BUDGET_BYTES", 3 * 2 * 16)
        gram, rhs = projection.accumulate_gto_pw_blocks(ref, prims, millers, psi=psi, gram=True)

        np.testing.assert_allclose(gram, dense.conj().T @ dense, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64)
        np.testing.assert_allclose(rhs, dense.conj().T @ psi, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64)

    @pytest.mark.light
    @pytest.mark.parametrize("with_spline", [False, True], ids=["pgto", "pgto+bspline"])
    def test_pgto_converters_use_bounded_fourier_blocks(self, monkeypatch, with_spline):
        """Both production fits respect the row budget and match a dense least-squares solve."""
        from jqmc import pw_pgto_projection as projection

        cell = QECell(a=7.0 * np.eye(3))
        millers = np.array([(i, j, 0) for i in range(-2, 3) for j in range(-1, 2)], dtype=np.int64)
        rng = np.random.default_rng(20260803)
        band = rng.normal(size=millers.shape[0]) + 1j * rng.normal(size=millers.shape[0])
        band /= np.linalg.norm(band)
        ref = _toy_reference(cell, millers, (8, 8, 8), k_point=np.array([0.2, 0.0, 0.0]), coeffs=band[None, :])
        shells = [
            [PrimitiveShell(angular_l=0, exponents=(0.45, 1.6), coefficients=(1.0, 1.0), rel_error=0.0, overlap_cond=1.0)]
        ]
        prims = build_pgto_primitives(shells, ref.atom_positions)
        build_fourier = projection.build_gto_pw_matrix
        dense = build_fourier(ref, prims)
        expected_coeffs, _, rank, _ = np.linalg.lstsq(dense, band[:, None], rcond=None)
        assert rank == len(prims)

        chunk_rows = 4
        monkeypatch.setattr(projection, "PW_PGTO_FOURIER_CHUNK_BUDGET_BYTES", chunk_rows * len(prims) * 16)
        blocks = []

        def build_bounded(*args, **kwargs):
            rows = kwargs.get("millers")
            assert rows is not None and len(rows) <= chunk_rows, "converter materialized an oversized Fourier block"
            blocks.append(rows)
            return build_fourier(*args, **kwargs)

        monkeypatch.setattr(projection, "build_gto_pw_matrix", build_bounded)
        convert = projection.project_pw_to_pgto_spline if with_spline else projection.project_pw_to_pgto
        result = convert(ref, shells)

        assert len(blocks) > 1
        np.testing.assert_array_equal(np.concatenate(blocks), millers)
        np.testing.assert_allclose(
            result.coefficients, expected_coeffs.T, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64
        )

    @pytest.mark.light
    @requires_qe
    def test_fourier_vs_real_pgto_cu(self):
        """``fourier`` (default) and ``real`` backends, pGTO-only, on Cu.

        Not expected to match to round-off: the Fourier backend forms the overlap
        on the band-limited QE reciprocal lattice (|G| <= ecutwfc); the real-space
        backend integrates the pGTO self-overlap on the dense FFT grid (up to
        ecutrho), so the high-G pGTO tail differs. The band-limited projection is
        the cleaner one and its residual is no worse than the real-space path.
        """
        from jqmc.pw_pgto_projection import project_pw_to_pgto, project_pw_to_pgto_debug
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.upf_wrapper import UPFExponentInit

        ref = load_qe_reference(str(_SAVE_DIR))
        strat = UPFExponentInit(
            {"li": str(_SAVE_DIR / "Li.ccECP.upf")},
            g_target=float(np.sqrt(2.0 * ref.ecutwfc)),
        )
        shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]

        r_real = project_pw_to_pgto_debug(ref, shells)
        r_four = project_pw_to_pgto(ref, shells)

        assert r_real.n_gto == r_four.n_gto
        assert r_real.coefficients.shape == r_four.coefficients.shape
        assert r_real.mean_residual < ATOL_QE_PROJECTION_FIDELITY_PGTO_ONLY_MEAN
        assert r_four.mean_residual < ATOL_QE_PROJECTION_FIDELITY_PGTO_ONLY_MEAN
        assert (
            r_four.mean_residual
            <= r_real.mean_residual * RTOL_PW_PROJECTION_RESIDUAL_PARITY_FACTOR + ATOL_PW_PROJECTION_FP64_GRID
        )
        np.testing.assert_array_equal(r_real.band_indices, r_four.band_indices)

    @pytest.mark.light
    @requires_qe
    def test_fourier_vs_real_mixed_li(self):
        """Mixed pGTO + low-G PW conversion under both backends on Li.

        Numerically equivalent: the Fourier backend evaluates the pGTO self-overlap
        via :func:`fft_cube_millers` (Plancherel sum over every G in the FFT cube),
        the real-space backend via an FFT-grid Riemann sum -- by Plancherel these
        are the same number up to FP error. The residual norms must agree to
        1e-6; coefficient norms can spread when the LSQ system is ill-conditioned
        (the Li ccECP basis is dominated by diffuse shells that the two backends
        project slightly differently). Pin the residual contract; the
        coefficient spread is a fixture-quality artefact, not a correctness bug.
        """
        from jqmc.pw_pgto_projection import project_pw_to_mixed, project_pw_to_mixed_debug
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.upf_wrapper import UPFExponentInit

        ref = load_qe_reference(str(_SAVE_DIR))
        strat = UPFExponentInit(
            {"li": str(_SAVE_DIR / "Li.ccECP.upf")},
            g_target=float(np.sqrt(2.0 * ref.ecutwfc)),
        )
        shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]
        gnorm = np.linalg.norm(ref.wfc.g_vectors(ref.cell), axis=1)
        g_cut = 0.6 * float(gnorm.max())

        m_real = project_pw_to_mixed_debug(ref, shells, g_cut=g_cut)
        m_four = project_pw_to_mixed(ref, shells, g_cut=g_cut)

        assert m_real.n_gto == m_four.n_gto
        assert m_real.n_pw == m_four.n_pw > 0
        assert m_four.pw_coefficients.shape == (m_four.band_indices.size, m_four.n_pw)
        assert m_four.coefficients.shape == (m_four.band_indices.size, m_four.n_gto)

        def rel(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))

        # Coefficient spread can be much larger when the LSQ system is
        # ill-conditioned (Li ccECP is diffuse-dominated); the design contract
        # is on the residual NORM, which is what jQMC actually consumes
        # downstream. Both residuals must be well below the design target
        # (1e-4) and agree to ~1e-7 absolute (~1% relative noise at this
        # magnitude is expected).
        np.testing.assert_allclose(
            m_four.residuals, m_real.residuals, atol=ATOL_PW_PROJECTION_FP64_RELAXED, rtol=RTOL_PW_PROJECTION_FP64_RELAXED
        )

    @pytest.mark.light
    @requires_qe
    def test_fourier_vs_real_pgto_spline_li(self):
        """Additive pGTO + B-spline conversion under both backends on Li.

        Both paths share the *same* real-space pGTO evaluator for the spline
        residual (``evaluate_pgto_expansion``) -- the only difference is where the
        pGTO LSQ fit lives (band-limited PW for ``fourier``, dense FFT grid for
        ``real``), which gives a small spread in the pGTO coefficients. The
        reported residual measures what jQMC will reconstruct downstream
        (pGTO leaf image-summed in real space + spline), so both paths clear
        the 1e-4 design target with margin; we don't pin a tight
        ratio between them (the spread is fixture-dependent and was a Cu-
        specific constant in earlier revisions).
        """
        from jqmc.pw_pgto_projection import project_pw_to_pgto_spline, project_pw_to_pgto_spline_debug
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.upf_wrapper import UPFExponentInit

        ref = load_qe_reference(str(_SAVE_DIR))
        strat = UPFExponentInit(
            {"li": str(_SAVE_DIR / "Li.ccECP.upf")},
            g_target=float(np.sqrt(2.0 * ref.ecutwfc)),
        )
        shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]

        for mesh_factor in (1.0, 0.6):
            s_real = project_pw_to_pgto_spline_debug(ref, shells, mesh_factor=mesh_factor, order=5)
            s_four = project_pw_to_pgto_spline(ref, shells, mesh_factor=mesh_factor, order=5)

            assert s_real.n_gto == s_four.n_gto
            assert s_real.spline_shape == s_four.spline_shape
            assert s_four.spline_coefficients.shape == s_real.spline_coefficients.shape
            assert s_four.coefficients.shape == (s_four.band_indices.size, s_four.n_gto)
            # Both combined models clear the design target with margin.
            assert s_real.mean_residual < ATOL_QE_PROJECTION_FIDELITY_SWEEP_TARGET
            assert s_four.mean_residual < ATOL_QE_PROJECTION_FIDELITY_SWEEP_TARGET
            # Same-order: the two backends agree on residual to within 2 orders
            # of magnitude at the 1e-9 noise floor. The exact spread is
            # fixture-dependent (Cu fixture in prior revisions hit 2x; the
            # Li ccECP basis spreads more because the LSQ system is more
            # ill-conditioned -- the spline soaks up the gap so the produced
            # leaf is still below the 1e-4 design target).
            assert (
                s_four.mean_residual
                <= s_real.mean_residual * RTOL_PW_PROJECTION_SPLINE_RESIDUAL_PARITY_FACTOR + ATOL_PW_PROJECTION_FP64_GRID
            )
            assert (
                s_real.mean_residual
                <= s_four.mean_residual * RTOL_PW_PROJECTION_SPLINE_RESIDUAL_PARITY_FACTOR + ATOL_PW_PROJECTION_FP64_GRID
            )


class TestSplineFidelityBandChunking:
    """Band-chunked fidelity diagnostic: peak-bounded and numerically faithful.

    The unchunked fidelity evaluation materialised every band on the full
    evaluation grid at once (302 GiB internal gather + three ~216 GiB grid
    arrays for a large production cell) for a per-band-independent diagnostic. The chunk loop
    is bounded by ``PW_SPLINE_FIDELITY_CHUNK_BUDGET_BYTES``; these tests pin
    the numerical contract: FFT / spline-evaluation / band-subset selection are
    BIT-identical per band, and the residuals agree with the single-shot form
    to einsum reduction round-off (the reduction order over grid points depends
    on the column count).
    """

    _SAVE = qe_example_files_root() / "H-ae_gamma" / "out" / "h_gamma.save"

    @pytest.mark.light
    def test_band_subset_reconstruction_is_bit_exact(self):
        """Band subset reconstruction is bit exact."""
        from jqmc.qe_wrapper import load_qe_reference

        ref = load_qe_reference(str(self._SAVE))
        shift = (0.5, 0.5, 0.5)
        full = reconstruct_pw_on_grid(ref.wfc, ref.fft_grid, shift_frac=shift)
        subset = np.array([1, 3])
        sub = reconstruct_pw_on_grid(ref.wfc, ref.fft_grid, shift_frac=shift, band_subset=subset)
        np.testing.assert_array_equal(full[:, subset], sub)

    @pytest.mark.light
    def test_evaluate_spline_orbitals_chunking_is_bit_exact(self, monkeypatch):
        """Evaluate spline orbitals chunking is bit exact."""
        import jqmc.pw_pgto_projection as proj

        rng = np.random.default_rng(0)
        coeffs = rng.standard_normal((5, 12, 12, 12)) + 1j * rng.standard_normal((5, 12, 12, 12))
        monkeypatch.setattr(proj, "PW_SPLINE_FIDELITY_CHUNK_BUDGET_BYTES", 1)  # chunk = 1 band
        chunked = evaluate_spline_orbitals(coeffs, (18, 18, 18), 5, (0.5, 0.5, 0.5))
        monkeypatch.setattr(proj, "PW_SPLINE_FIDELITY_CHUNK_BUDGET_BYTES", 1 << 60)  # one chunk
        single = evaluate_spline_orbitals(coeffs, (18, 18, 18), 5, (0.5, 0.5, 0.5))
        np.testing.assert_array_equal(chunked, single)

    @pytest.mark.light
    def test_fidelity_residuals_chunked_vs_single_shot(self, monkeypatch):
        """Fidelity residuals chunked vs single shot."""
        from jqmc.qe_wrapper import load_qe_reference

        import jqmc.pw_pgto_projection as proj

        ref = load_qe_reference(str(self._SAVE))
        monkeypatch.setattr(proj, "PW_SPLINE_FIDELITY_CHUNK_BUDGET_BYTES", 1)  # chunk = 1 band
        r_chunked = project_pw_to_spline(ref, mesh_factor=0.5, order=5).residuals
        monkeypatch.setattr(proj, "PW_SPLINE_FIDELITY_CHUNK_BUDGET_BYTES", 1 << 60)  # one chunk
        r_single = project_pw_to_spline(ref, mesh_factor=0.5, order=5).residuals
        # Per-band FFT/spline values are bit-identical; only the einsum
        # reduction order over the ~730k grid points differs with the column
        # count. Measured difference ~4e-14 relative on this fixture, so the
        # fp64-ulp-scale RTOL_PYTHON_FP64_LOOSE is the right bound.
        np.testing.assert_allclose(r_chunked, r_single, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)


class TestSplineFitBandChunking:
    """Coefficient-BUILD chunking (``fit_pw_to_spline``) is exactly lossless.

    Distinct from :class:`TestSplineFidelityBandChunking`, which covers the
    diagnostic. This path produces the control-point table every downstream run
    consumes, so the contract is strictly BIT-identical -- not "to round-off":
    band chunking and the ``band_subset`` pre-selection only change how many
    bands share a scatter / IFFT / prefilter call, never the per-band
    arithmetic. (The one variant that did perturb the result, an in-place
    multi-axis ``ifftn``, was rejected for exactly this reason -- see the note
    in ``reconstruct_pw_on_grid``.) Guards ``PW_SPLINE_FIT_CHUNK_BUDGET_BYTES``
    against a future "chunking is only round-off exact" relaxation.
    """

    _SAVE = qe_example_files_root() / "H-ae_gamma" / "out" / "h_gamma.save"

    @pytest.mark.light
    def test_fit_chunked_and_subset_is_bit_exact(self, monkeypatch):
        """Fit chunked and subset is bit exact."""
        from jqmc.qe_wrapper import load_qe_reference

        import jqmc.pw_pgto_projection as proj

        ref = load_qe_reference(str(self._SAVE))
        order = 5
        spline_shape = tuple(max(order + 1, int(round(0.5 * n))) for n in ref.fft_grid)

        monkeypatch.setattr(proj, "PW_SPLINE_FIT_CHUNK_BUDGET_BYTES", 1 << 60)  # one chunk, all bands
        full = fit_pw_to_spline(ref.wfc, spline_shape, order)

        monkeypatch.setattr(proj, "PW_SPLINE_FIT_CHUNK_BUDGET_BYTES", 1)  # chunk = 1 band
        chunked = fit_pw_to_spline(ref.wfc, spline_shape, order)
        np.testing.assert_array_equal(chunked, full)

        subset = np.array([1, 3, 4], dtype=np.int64)
        sub = fit_pw_to_spline(ref.wfc, spline_shape, order, band_subset=subset)
        np.testing.assert_array_equal(sub, full[subset])

    @pytest.mark.light
    def test_spline_coefficients_unchanged_by_the_fit_budget(self, monkeypatch):
        """End-to-end: the projection's stored table is budget-independent."""
        from jqmc.qe_wrapper import load_qe_reference

        import jqmc.pw_pgto_projection as proj

        ref = load_qe_reference(str(self._SAVE))
        monkeypatch.setattr(proj, "PW_SPLINE_FIT_CHUNK_BUDGET_BYTES", 1 << 60)
        single = project_pw_to_spline(ref, mesh_factor=0.5, order=5).spline_coefficients
        monkeypatch.setattr(proj, "PW_SPLINE_FIT_CHUNK_BUDGET_BYTES", 1)
        chunked = project_pw_to_spline(ref, mesh_factor=0.5, order=5).spline_coefficients
        np.testing.assert_array_equal(chunked, single)


class TestPWGSphereTaper:
    """G-sphere taper of the QE plane-wave coefficients (vacuum cells only).

    A sphere-truncated PW expansion rings in vacuum with -1/2 lap(psi)/psi of
    order G_max^2/2 (see ``PW_TAPER_*`` in ``jqmc/_setting.py``). The taper must
    be a monotone C-infinity step in |G+k|, must leave cells without a vacuum
    region untouched (the very same reference object), and in a vacuum cell must
    attenuate only the sphere edge while preserving the Gamma symmetry
    ``c(-G) = conj(c(G))``; the spline fidelity is then judged against the
    tapered bands.
    """

    @staticmethod
    def _stretched_vacuum_reference(reference):
        """Li_gamma with its third lattice vector x4 (~12 bohr nucleus-free) and G_max = max|G|."""
        import dataclasses

        from jqmc.qe_wrapper import QECell

        a = np.array(reference.cell.a, dtype=np.float64).copy()
        a[2] *= 4.0
        cell = QECell(a=a)
        q = np.linalg.norm(reference.wfc.millers.astype(np.float64) @ cell.b, axis=1)
        # ecutwfc marginally below max|G|^2/2 so the outermost shell sits at or beyond g_end.
        return dataclasses.replace(reference, cell=cell, ecutwfc=0.5 * float(q.max() * (1.0 - np.finfo(np.float64).eps)) ** 2)

    @pytest.mark.light
    def test_pw_taper_window_is_a_monotone_smooth_step(self):
        """The window is 1 below g_start, 0 from g_end, monotone, exactly 1/2 at the midpoint, and flat to fp64 at both ends."""
        from jqmc.pw_pgto_projection import pw_taper_window

        g_start, g_end = 8.0, 10.0
        q = np.linspace(0.0, 12.0, 24001)
        w = pw_taper_window(q, g_start, g_end)
        assert np.all(w[q <= g_start] == 1.0)
        assert np.all(w[q >= g_end] == 0.0)
        assert np.all(np.diff(w) <= 0.0)
        ramp = (q > g_start) & (q < g_end)
        assert np.all(w[ramp] <= w[q <= g_start].min()) and np.all(w[ramp] >= w[q >= g_end].max())
        assert np.any((w[ramp] > w[q >= g_end].max()) & (w[ramp] < w[q <= g_start].min()))
        # S(t) + S(1 - t) = 1: at the ramp midpoint f(t) == f(1 - t) exactly, so w is exactly one half.
        assert pw_taper_window([(g_start + g_end) / 2.0], g_start, g_end)[0] == 0.5
        # Every derivative vanishes at both ends: one percent into the ramp the
        # window is already 1 (1 - w underflows) and 0 to far below fp64 eps^2,
        # which no finite-order polynomial step can do.
        width = g_end - g_start
        assert pw_taper_window([g_start + 0.01 * width], g_start, g_end)[0] == 1.0
        assert pw_taper_window([g_end - 0.01 * width], g_start, g_end)[0] < np.finfo(np.float64).eps ** 2
        with pytest.raises(ValueError, match="g_start < g_end"):
            pw_taper_window(q, g_end, g_start)

    @pytest.mark.light
    def test_taper_is_the_identity_without_a_vacuum_region(self):
        """The committed Li gamma save has no nucleus-free point beyond the radius: the same object comes back."""
        from jqmc._setting import PW_TAPER_VACUUM_RADIUS_BOHR
        from jqmc.pw_pgto_projection import max_empty_distance, taper_qe_reference
        from jqmc.qe_wrapper import load_qe_reference

        ref = load_qe_reference(str(_SAVE_DIR))
        assert max_empty_distance(ref.cell.a, ref.atom_positions) <= PW_TAPER_VACUUM_RADIUS_BOHR
        out, meta = taper_qe_reference(ref)
        assert out is ref
        assert meta is None

    @pytest.mark.light
    def test_taper_attenuates_only_the_sphere_edge_and_keeps_gamma_symmetry(self):
        """Stretched vacuum cell: untouched below g_start, zero from g_end, attenuated on the ramp, c(-G) = conj(c(G))."""
        from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, PW_TAPER_START_FRACTION, PW_TAPER_VACUUM_RADIUS_BOHR
        from jqmc.pw_pgto_projection import max_empty_distance, taper_qe_reference
        from jqmc.qe_wrapper import load_qe_reference

        ref = self._stretched_vacuum_reference(load_qe_reference(str(_SAVE_DIR)))
        assert max_empty_distance(ref.cell.a, ref.atom_positions) > PW_TAPER_VACUUM_RADIUS_BOHR
        out, meta = taper_qe_reference(ref)
        assert out is not ref
        assert meta is not None
        g_end = float(np.sqrt(2.0 * ref.ecutwfc))
        g_start = PW_TAPER_START_FRACTION * g_end
        assert meta["pw_taper_g_end"] == g_end
        assert meta["pw_taper_g_start"] == g_start
        q = np.linalg.norm(out.wfc.millers.astype(np.float64) @ out.cell.b, axis=1)
        np.testing.assert_array_equal(out.wfc.coeffs[:, q <= g_start], ref.wfc.coeffs[:, q <= g_start])
        assert np.any(q >= g_end)
        assert np.all(out.wfc.coeffs[:, q >= g_end] == 0.0)
        ramp = (q > g_start) & (q < g_end)
        assert np.any(ramp)
        before = np.abs(ref.wfc.coeffs[:, ramp])
        after = np.abs(out.wfc.coeffs[:, ramp])
        assert np.all(after <= before)
        assert np.any(after < before)
        # The window depends on |G| only, so the expanded gamma_only set keeps c(-G) = conj(c(G)).
        assert out.wfc.gamma_only
        index = {tuple(m): i for i, m in enumerate(out.wfc.millers.tolist())}
        neg = np.array([index[tuple(-int(x) for x in m)] for m in out.wfc.millers.tolist()], dtype=np.int64)
        np.testing.assert_array_equal(out.wfc.coeffs[:, neg], np.conj(out.wfc.coeffs))
        # The reported spectral loss is a real fraction of the removed power, band by band.
        assert 0.0 < meta["pw_taper_spectral_loss_mean_occupied"] <= meta["pw_taper_spectral_loss_max_occupied"] < 1.0
        per_band = np.asarray(meta["pw_taper_spectral_loss_per_band"])
        assert per_band.shape == (ref.wfc.n_bands,)
        power_ref = np.sum(np.abs(ref.wfc.coeffs) ** 2, axis=1)
        power_out = np.sum(np.abs(out.wfc.coeffs) ** 2, axis=1)
        assert np.all(per_band > 0.0) and np.all(power_out < power_ref)
        np.testing.assert_allclose(
            per_band,
            (power_ref - power_out) / power_ref,
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_LOOSE,
        )
        occupied = np.asarray(ref.bands.occupations) > 0.0
        assert meta["pw_taper_spectral_loss_max_occupied"] == float(np.max(per_band[occupied]))
        # Everything else is carried over unchanged.
        assert out.wfc.millers is ref.wfc.millers
        np.testing.assert_array_equal(out.bands.occupations, ref.bands.occupations)
        assert out.ecutwfc == ref.ecutwfc

    @pytest.mark.light
    def test_spline_fidelity_is_judged_against_the_tapered_bands(self):
        """project_pw_to_spline reports the residual against the tapered bands, below that against the raw ones."""
        from jqmc.pw_pgto_projection import (
            _cell_normalization,
            evaluate_spline_orbitals,
            project_pw_to_spline,
            reconstruct_pw_on_grid,
            taper_qe_reference,
        )
        from jqmc.qe_wrapper import load_qe_reference

        ref = self._stretched_vacuum_reference(load_qe_reference(str(_SAVE_DIR)))
        tapered, _ = taper_qe_reference(ref)
        order = 5
        proj = project_pw_to_spline(tapered, mesh_factor=1.0, order=order)
        shift = (0.5, 0.5, 0.5)
        coeffs = np.asarray(proj.spline_coefficients).reshape(-1, *proj.spline_shape)
        spline_vals = evaluate_spline_orbitals(coeffs, ref.fft_grid, order, shift_frac=shift)
        psi_raw = reconstruct_pw_on_grid(ref.wfc, ref.fft_grid, shift_frac=shift, band_subset=proj.band_indices)
        psi_raw *= _cell_normalization(ref)
        diff = spline_vals - psi_raw
        residual_vs_raw = np.real(np.einsum("pn,pn->n", diff.conj(), diff)) / np.real(
            np.einsum("pn,pn->n", psi_raw.conj(), psi_raw)
        )
        # The converter's residual is measured against the tapered bands it fitted; the
        # removed sphere-edge power is NOT reported as a spline error.
        assert np.all(proj.residuals < residual_vs_raw)

    @pytest.mark.light
    @pytest.mark.parametrize("kind, ecutwfc", [("bspline", None), ("pw", 40.0)])
    def test_run_qe2jqmc_tapers_only_vacuum_cells_and_records_it(self, tmp_path, monkeypatch, kind, ecutwfc):
        """run_qe2jqmc leaves the bulk file unmarked; a vacuum cell gets pw_taper_* attrs (edge g_cut for a truncated pw basis) and the tapered table."""
        import h5py

        from jqmc._setting import PW_TAPER_START_FRACTION, PW_TAPER_VACUUM_RADIUS_BOHR
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jqmc_qe import run_qe2jqmc
        from jqmc.pw_pgto_projection import project_pw_to_spline, taper_qe_reference
        from jqmc.qe_wrapper import load_qe_reference

        qe_params = {"qe_save_dir": str(_CLI_FIXTURE), "kind": kind, "mesh_factor": 1.0, "ecutwfc": ecutwfc}
        jastrow_params = {"j2_parameter": 1.0}

        bulk_file = tmp_path / "bulk.h5"
        run_qe2jqmc(qe_params=qe_params, jastrow_params=jastrow_params, hamiltonian_h5=str(bulk_file))
        with h5py.File(bulk_file, "r") as f:
            assert not any(str(key).startswith("pw_taper") for key in f.attrs)

        stretched = self._stretched_vacuum_reference(load_qe_reference(str(_CLI_FIXTURE)))
        monkeypatch.setattr("jqmc.qe_wrapper.load_qe_reference", lambda *args, **kwargs: stretched)
        vacuum_file = tmp_path / "vacuum.h5"
        run_qe2jqmc(qe_params=qe_params, jastrow_params=jastrow_params, hamiltonian_h5=str(vacuum_file))
        # A numeric ecutwfc (Ry) truncates the pw basis at g_cut = sqrt(ecutwfc_Ry); the taper edge follows it.
        g_edge = float(np.sqrt(2.0 * stretched.ecutwfc)) if ecutwfc is None else float(np.sqrt(ecutwfc))
        with h5py.File(vacuum_file, "r") as f:
            assert float(f.attrs["pw_taper_max_empty_distance_bohr"]) > PW_TAPER_VACUUM_RADIUS_BOHR
            assert float(f.attrs["pw_taper_start_fraction"]) == PW_TAPER_START_FRACTION
            assert float(f.attrs["pw_taper_g_end"]) == g_edge
            assert np.asarray(f.attrs["pw_taper_spectral_loss_per_band"]).shape == (stretched.wfc.n_bands,)
        if kind != "bspline":
            return

        tapered, _ = taper_qe_reference(stretched)
        expected = np.asarray(project_pw_to_spline(tapered, mesh_factor=1.0, order=5).spline_coefficients)
        raw = np.asarray(project_pw_to_spline(stretched, mesh_factor=1.0, order=5).spline_coefficients)
        ham = Hamiltonian_data.load_from_hdf5(str(vacuum_file))
        stored = np.asarray(ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data.spline_coefficients)
        np.testing.assert_array_equal(stored.reshape(expected.shape), expected.real)
        assert not np.array_equal(stored.reshape(raw.shape), raw.real)

    @pytest.mark.light
    def test_vacuum_detection_is_exact_and_representation_invariant(self):
        """The covering radius equals the exact value and ignores the origin, the primitive basis and image translations."""
        from jqmc._setting import ATOL_PYTHON_FP64_LOOSE
        from jqmc.pw_pgto_projection import max_empty_distance

        side = 11.6
        cube = side * np.eye(3)
        exact = np.sqrt(3.0) * side / 2.0  # one atom per cubic cell: the body-diagonal midpoint
        shear = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float64) @ cube  # same lattice, other basis
        for cell, atoms in (
            (cube, np.zeros((1, 3))),
            (cube, np.full((1, 3), side / 48.0)),  # shifted origin
            (shear, np.zeros((1, 3))),
            (cube, np.array([[25.0 * side, 0.0, 0.0]])),  # the same periodic atom, unwrapped representative
        ):
            np.testing.assert_allclose(
                max_empty_distance(cell, atoms), exact, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE
            )
        dense = 4.0 * np.eye(3)
        np.testing.assert_allclose(
            max_empty_distance(dense, np.array([[100.0, 0.0, 0.0]])),
            2.0 * np.sqrt(3.0),
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_LOOSE,
        )
        # Two atoms: the far point of a body-centred cell is the midpoint of the octahedral hole.
        bcc = max_empty_distance(cube, np.array([[0.0, 0.0, 0.0], [side / 2.0] * 3]))
        np.testing.assert_allclose(bcc, side * np.sqrt(5.0) / 4.0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    def test_taper_uses_the_physical_wavevector_at_nonzero_twist(self):
        """At k != 0 the window is a function of |G+k|: columns with |G+k| >= g_end vanish and columns with |G| >= g_end > |G+k| do not."""
        from jqmc.pw_pgto_projection import taper_qe_reference
        from jqmc.qe_wrapper import load_qe_reference

        import dataclasses

        ref = self._stretched_vacuum_reference(load_qe_reference(str(_SAVE_DIR_K025)))
        k = np.asarray(ref.wfc.k_point, dtype=np.float64)
        assert np.linalg.norm(k) > 0.0
        # Put the sphere edge just inside the largest PHYSICAL wavevector, so that columns on both
        # sides of the edge in |G| and in |G+k| exist.
        q_max = float(np.linalg.norm(ref.wfc.millers.astype(np.float64) @ ref.cell.b + k[None, :], axis=1).max())
        ref = dataclasses.replace(ref, ecutwfc=0.5 * (q_max * (1.0 - np.finfo(np.float64).eps)) ** 2)
        out, meta = taper_qe_reference(ref)
        assert meta is not None
        g_end = meta["pw_taper_g_end"]
        g = ref.wfc.millers.astype(np.float64) @ ref.cell.b
        q_phys = np.linalg.norm(g + k[None, :], axis=1)
        q_bare = np.linalg.norm(g, axis=1)
        beyond_phys = q_phys >= g_end
        beyond_bare_only = (q_bare >= g_end) & (q_phys < g_end)
        beyond_phys_only = (q_phys >= g_end) & (q_bare < g_end)
        assert np.any(beyond_bare_only) and np.any(beyond_phys_only)
        assert np.all(out.wfc.coeffs[:, beyond_phys] == 0.0)
        assert np.any(out.wfc.coeffs[:, beyond_bare_only] != 0.0)
        assert np.any(ref.wfc.coeffs[:, beyond_phys_only] != 0.0)  # the vanished columns were not trivially zero

    @pytest.mark.light
    def test_taper_restores_the_vacuum_local_kinetic_energy_of_a_known_tail(self):
        """Periodized Gaussian-smoothed exp(-|z|): T = -1/2 Ha exactly away from the source; the sharp sum rings, the tapered sum does not.

        Asserted: the tapered sum is within RTOL_PW_TAPER_VACUUM_KINETIC of -1/2 Ha over the whole
        continuous range z in [1, 8.5] bohr (tail down to 2e-4 of the maximum) and at the sampled
        points z = 8, 10, 12 bohr (3e-4, 4.5e-5, 6e-6), and closer than the sharp sum, which is wrong
        by more than |T| itself at those points. Between z = 10 and 12 the tapered deviation reaches
        26-28 % (not asserted); beyond ~1e-7 of the maximum neither sum bounds the local energy.
        """
        from jqmc._setting import PW_TAPER_START_FRACTION, RTOL_PW_TAPER_VACUUM_KINETIC
        from jqmc.pw_pgto_projection import pw_taper_window

        length, g_max, sigma = 48.0, 20.0, 0.15
        n_max = int(np.floor(length * g_max / (2.0 * np.pi)))
        m = np.arange(-n_max, n_max + 1)
        g = 2.0 * np.pi * m / length
        coeffs = 2.0 * np.exp(-0.5 * sigma**2 * g**2) / (length * (1.0 + g**2))  # Fourier series of the smoothed exp(-|z|)
        window = pw_taper_window(np.abs(g), PW_TAPER_START_FRACTION * g_max, g_max)
        exact_kinetic = -0.5

        def local_kinetic(c, z):
            positive = m > 0
            gp = g[positive].astype(np.longdouble)
            cp = c[positive].astype(np.longdouble)
            phase = np.cos(gp[:, None] * np.asarray(z, dtype=np.longdouble))
            value = np.longdouble(c[m == 0][0]) + 2.0 * np.sum(cp[:, None] * phase, axis=0)
            laplacian = -2.0 * np.sum((cp * gp**2)[:, None] * phase, axis=0)
            return (-0.5 * laplacian / value).astype(np.float64)

        z_scan = np.linspace(1.0, 8.5, 1501)  # exp(-z): 0.37 ... 2e-4 of the maximum
        np.testing.assert_allclose(
            local_kinetic(coeffs * window, z_scan),
            exact_kinetic,
            atol=ATOL_BIT_EXACT,
            rtol=RTOL_PW_TAPER_VACUUM_KINETIC,
        )
        z = np.array([8.0, 10.0, 12.0])  # exp(-z): 3e-4 ... 6e-6 of the maximum
        raw = local_kinetic(coeffs, z)
        tapered = local_kinetic(coeffs * window, z)
        np.testing.assert_allclose(tapered, exact_kinetic, atol=ATOL_BIT_EXACT, rtol=RTOL_PW_TAPER_VACUUM_KINETIC)
        assert np.all(np.abs(raw - exact_kinetic) > np.abs(tapered - exact_kinetic))
        assert np.max(np.abs(raw - exact_kinetic)) > abs(
            exact_kinetic
        )  # the sharp sphere is wrong by more than the value itself

    @pytest.mark.light
    def test_taper_reports_the_exact_removed_power_of_every_band(self):
        """Two-plane-wave Gamma bands with rational powers: the per-band loss is exactly (removed power) / (total power) for each band."""
        from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, PW_TAPER_VACUUM_RADIUS_BOHR
        from jqmc.pw_pgto_projection import max_empty_distance, taper_qe_reference
        from jqmc.qe_wrapper import QEBands, QECell, QEReference, QEWavefunction

        side = 30.0  # one atom per cubic cell: covering radius 26 bohr > PW_TAPER_VACUUM_RADIUS_BOHR
        cell = QECell(a=side * np.eye(3))
        millers = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)
        # Band 0: |c|^2 = 16, 1, 1 -> loss 2/18 = 1/9. Band 1: |c|^2 = 1, 4, 4 -> loss 8/9. Hermitian at Gamma.
        coeffs = np.array([[4.0, 1.0, 1.0], [1.0, 2.0, 2.0]], dtype=np.complex128)
        g_edge = float(np.linalg.norm(millers[1].astype(np.float64) @ cell.b))
        reference = QEReference(
            cell=cell,
            atom_labels=("H",),
            atom_positions=np.zeros((1, 3)),
            ecutwfc=0.5 * (g_edge * (1.0 - np.finfo(np.float64).eps)) ** 2,  # the edge columns sit at or beyond g_end
            fft_grid=(8, 8, 8),
            bands=QEBands(
                eigenvalues=np.array([-1.0, -0.5]), occupations=np.array([1.0, 1.0]), n_electrons=2.0, fermi_energy=None
            ),
            wfc=QEWavefunction(millers=millers, coeffs=coeffs, gamma_only=True, k_point=np.zeros(3)),
            atom_elements=("H",),
        )
        assert max_empty_distance(cell.a, reference.atom_positions) > PW_TAPER_VACUUM_RADIUS_BOHR
        out, meta = taper_qe_reference(reference)
        assert meta is not None
        np.testing.assert_array_equal(out.wfc.coeffs[:, 1:], 0.0)
        np.testing.assert_array_equal(out.wfc.coeffs[:, 0], coeffs[:, 0])
        np.testing.assert_array_equal(meta["pw_taper_spectral_loss_per_band"], np.array([1.0 / 9.0, 8.0 / 9.0]))
        assert meta["pw_taper_spectral_loss_max_occupied"] == 8.0 / 9.0
        np.testing.assert_allclose(
            meta["pw_taper_spectral_loss_mean_occupied"],
            0.5,
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_LOOSE,
        )
        # Only the occupied bands enter the summaries.
        half = taper_qe_reference(dataclasses_replace_bands(reference, np.array([1.0, 0.0])))[1]
        assert half is not None and half["pw_taper_spectral_loss_max_occupied"] == 1.0 / 9.0

    @pytest.mark.light
    def test_pw_zero_cutoff_conversion_keeps_the_constant_plane_wave(self, tmp_path, monkeypatch):
        """Ecutwfc = 0 retains only the constant plane wave: the vacuum cell converts without a taper edge and records no taper."""
        import h5py

        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jqmc_qe import run_qe2jqmc
        from jqmc.pw_pgto_projection import taper_qe_reference
        from jqmc.qe_wrapper import load_qe_reference

        stretched = self._stretched_vacuum_reference(load_qe_reference(str(_CLI_FIXTURE)))
        assert taper_qe_reference(stretched)[1] is not None  # the cell itself is a vacuum cell
        out, meta = taper_qe_reference(stretched, g_end=0.0)
        assert out is stretched and meta is None
        monkeypatch.setattr("jqmc.qe_wrapper.load_qe_reference", lambda *args, **kwargs: stretched)
        out_file = tmp_path / "pw_zero_cutoff.h5"
        run_qe2jqmc(
            qe_params={"qe_save_dir": str(_CLI_FIXTURE), "kind": "pw", "ecutwfc": 0.0},
            jastrow_params={"j2_parameter": 1.0},
            hamiltonian_h5=str(out_file),
        )
        with h5py.File(out_file, "r") as f:
            assert not any(str(key).startswith("pw_taper") for key in f.attrs)
        ham = Hamiltonian_data.load_from_hdf5(str(out_file))
        assert int(ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data.num_ao) == 1

    @pytest.mark.light
    def test_twist_average_records_labeled_taper_summaries_per_channel(self, tmp_path, monkeypatch):
        """The twist-average file keeps one occupied mean/max and one per-band array per (twist, spin) channel; no channel overwrites another."""
        import h5py

        from jqmc import qe_wrapper
        from jqmc.pw_pgto_projection import taper_qe_reference
        from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input

        real_loader = qe_wrapper.load_qe_reference
        monkeypatch.setattr(
            "jqmc.qe_wrapper.load_qe_reference", lambda *a, **k: self._stretched_vacuum_reference(real_loader(*a, **k))
        )
        out = tmp_path / "twist_lsda_taper.h5"
        convert_qe_to_twist_average_input(
            str(_O_LSDA_K025), str(out), upf_paths={"O": str(_O_LSDA_K025 / "O.ccECP.upf")}, kind="bspline", mesh_factor=1.0
        )
        expected = {}
        for spin, label in (("up", "_ik1"), ("dw", "_ik1_dn")):
            _, meta = taper_qe_reference(
                self._stretched_vacuum_reference(real_loader(str(_O_LSDA_K025), ik=1, spin=spin)), label=label
            )
            assert meta is not None
            expected.update(meta)
        with h5py.File(out, "r") as f:
            attrs = dict(f["_meta"].attrs)
        for label in ("_ik1", "_ik1_dn"):
            for stem in ("pw_taper_spectral_loss_mean_occupied", "pw_taper_spectral_loss_max_occupied"):
                assert float(attrs[stem + label]) == expected[stem + label]
            np.testing.assert_array_equal(
                attrs["pw_taper_spectral_loss_per_band" + label], expected["pw_taper_spectral_loss_per_band" + label]
            )
        assert expected["pw_taper_spectral_loss_max_occupied_ik1"] != expected["pw_taper_spectral_loss_max_occupied_ik1_dn"]
        assert "pw_taper_spectral_loss_mean_occupied" not in attrs and "pw_taper_spectral_loss_max_occupied" not in attrs
        assert float(attrs["pw_taper_g_end"]) == expected["pw_taper_g_end"]

    @pytest.mark.light
    def test_truncated_pw_edge_is_applied_to_the_persisted_coefficients(self, tmp_path, monkeypatch):
        """kind=pw with a numeric ecutwfc on a vacuum cell: the stored complex PW coefficients are the raw QE coefficients times w(|G+k|; 0.8 g_cut, g_cut), truncated at g_cut, and compute_MOs reproduces the tapered band values."""
        from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, PW_TAPER_START_FRACTION, RTOL_QE_PW_RECONSTRUCTION_LOOSE
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jqmc_qe import run_qe2jqmc
        from jqmc.molecular_orbital import compute_MOs
        from jqmc.pw_pgto_projection import pw_taper_window
        from jqmc.qe_wrapper import load_qe_reference
        from tests.test_jqmc_kernel.test_qe_lsda import _pw_band_values_in_zones

        stretched = self._stretched_vacuum_reference(load_qe_reference(str(_SAVE_DIR_K025)))
        ecutwfc_ry = 40.0
        g_cut = float(np.sqrt(ecutwfc_ry))
        g_sphere = float(np.sqrt(2.0 * stretched.ecutwfc))
        assert (
            PW_TAPER_START_FRACTION * g_sphere > g_cut > PW_TAPER_START_FRACTION * g_cut
        )  # the requested ramp lies below the QE edge's ramp
        monkeypatch.setattr("jqmc.qe_wrapper.load_qe_reference", lambda *args, **kwargs: stretched)
        out_file = tmp_path / "pw_truncated.h5"
        run_qe2jqmc(
            qe_params={"qe_save_dir": str(_SAVE_DIR_K025), "kind": "pw", "ecutwfc": ecutwfc_ry},
            jastrow_params={"j2_parameter": 1.0},
            hamiltonian_h5=str(out_file),
        )
        orb = Hamiltonian_data.load_from_hdf5(str(out_file)).wavefunction_data.geminal_data.orb_data_up_spin
        aos = orb.aos_data
        stored = np.asarray(orb.mo_coefficients, dtype=np.complex128)  # (n_band, n_ao)
        millers = np.asarray(aos._g_indices, dtype=np.int64).reshape(-1, 3)
        k_cart = np.asarray(stretched.wfc.k_point, dtype=np.float64)
        g_plus_k = millers.astype(np.float64) @ stretched.cell.b + k_cart[None, :]
        q = np.linalg.norm(g_plus_k, axis=1)
        # Every retained plane wave lies inside the requested cutoff, and the ramp below it is populated.
        assert np.all(q <= g_cut)
        ramp = (q > PW_TAPER_START_FRACTION * g_cut) & (q < PW_TAPER_START_FRACTION * g_sphere)
        assert np.any(ramp)
        # Stored = raw QE coefficient x Omega^-1/2 x w(|G+k|; 0.8 g_cut, g_cut), band by band.
        index = {tuple(m): i for i, m in enumerate(stretched.wfc.millers.tolist())}
        raw_cols = np.array([index[tuple(m)] for m in millers.tolist()], dtype=np.int64)
        occupied = np.flatnonzero(np.asarray(stretched.bands.occupations) > 0.0)
        raw = stretched.wfc.coeffs[occupied][:, raw_cols] / np.sqrt(stretched.cell.volume)
        window = pw_taper_window(q, PW_TAPER_START_FRACTION * g_cut, g_cut)
        np.testing.assert_allclose(stored, raw * window[None, :], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
        # The window is exactly 1 to fp64 at the start of the ramp; deeper in the ramp it is below 1.
        assert np.any(window[ramp] < 1.0) and np.any(np.abs(stored[:, ramp]) < np.abs(raw[:, ramp]))
        # The production consumer evaluates exactly those tapered, truncated bands.
        points = np.random.default_rng(5).uniform(0.0, 6.0, size=(16, 3))
        expected = _pw_band_values_in_zones(
            stretched.wfc.coeffs[occupied][:, raw_cols] * window[None, :], g_plus_k, points
        ) / np.sqrt(stretched.cell.volume)
        values = np.asarray(compute_MOs(orb, points))
        assert np.linalg.norm(values - expected) / np.linalg.norm(expected) < RTOL_QE_PW_RECONSTRUCTION_LOOSE

    @pytest.mark.light
    def test_single_k_lsda_records_both_spin_taper_summaries(self, tmp_path, monkeypatch):
        """run_qe2jqmc on an LSDA vacuum cell records the up summaries unlabeled and the down summaries with _dn, each equal to an independent power-sum oracle of its own channel."""
        import h5py

        from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, PW_TAPER_START_FRACTION
        from jqmc.jqmc_qe import run_qe2jqmc
        from jqmc.pw_pgto_projection import pw_taper_window
        from jqmc.qe_wrapper import truncate_qe_reference_bands

        from jqmc import qe_wrapper

        real_loader = qe_wrapper.load_qe_reference
        monkeypatch.setattr(
            "jqmc.qe_wrapper.load_qe_reference", lambda *a, **k: self._stretched_vacuum_reference(real_loader(*a, **k))
        )
        out_file = tmp_path / "lsda_single_k.h5"
        run_qe2jqmc(
            qe_params={
                "qe_save_dir": str(_O_LSDA_GAMMA),
                "kind": "bspline",
                "mesh_factor": 1.0,
                "upf_paths": [f"O={_O_LSDA_GAMMA / 'O.ccECP.upf'}"],
            },
            jastrow_params={"j2_parameter": 1.0},
            hamiltonian_h5=str(out_file),
        )
        with h5py.File(out_file, "r") as f:
            attrs = dict(f.attrs)

        def oracle(spin):
            # The converter keeps each channel's bands through its highest occupied one, then tapers them.
            ref = self._stretched_vacuum_reference(real_loader(str(_O_LSDA_GAMMA), spin=spin))
            ref = truncate_qe_reference_bands(ref, int(ref.bands.occupied_band_indices().max()) + 1)
            q = np.linalg.norm(ref.wfc.millers.astype(np.float64) @ ref.cell.b + np.asarray(ref.wfc.k_point)[None, :], axis=1)
            g_end = float(np.sqrt(2.0 * ref.ecutwfc))
            w = pw_taper_window(q, PW_TAPER_START_FRACTION * g_end, g_end)
            power = np.sum(np.abs(ref.wfc.coeffs) ** 2, axis=1)
            loss = np.sum(np.abs(ref.wfc.coeffs) ** 2 * (1.0 - w * w)[None, :], axis=1) / power
            occ = np.asarray(ref.bands.occupations) > 0.0
            return loss, float(loss[occ].max()), float(np.mean(loss[occ]))

        for spin, label in (("up", ""), ("dw", "_dn")):
            loss, loss_max, loss_mean = oracle(spin)
            np.testing.assert_allclose(
                attrs["pw_taper_spectral_loss_per_band" + label],
                loss,
                atol=ATOL_PYTHON_FP64_LOOSE,
                rtol=RTOL_PYTHON_FP64_LOOSE,
            )
            np.testing.assert_allclose(
                float(attrs["pw_taper_spectral_loss_max_occupied" + label]),
                loss_max,
                atol=ATOL_PYTHON_FP64_LOOSE,
                rtol=RTOL_PYTHON_FP64_LOOSE,
            )
            np.testing.assert_allclose(
                float(attrs["pw_taper_spectral_loss_mean_occupied" + label]),
                loss_mean,
                atol=ATOL_PYTHON_FP64_LOOSE,
                rtol=RTOL_PYTHON_FP64_LOOSE,
            )
        assert attrs["pw_taper_spectral_loss_max_occupied"] != attrs["pw_taper_spectral_loss_max_occupied_dn"]


def dataclasses_replace_bands(reference, occupations):
    """Copy of ``reference`` with other band occupations (module helper: keeps the test body flat)."""
    import dataclasses

    from jqmc.qe_wrapper import QEBands

    bands = reference.bands
    return dataclasses.replace(
        reference,
        bands=QEBands(
            eigenvalues=bands.eigenvalues,
            occupations=np.asarray(occupations, dtype=np.float64),
            n_electrons=bands.n_electrons,
            fermi_energy=bands.fermi_energy,
        ),
    )


#: PR #122 follow-up witness: one tight Cartesian d primitive (alpha 4, about the
#: O ccECP cc-pVQZ d) in a cubic cell, with the atom on the cell's x axis line
#: y = z = edge / 2 at distance ``_SPLINE_LEAF_FACE_GAP`` from the face x = edge.
#: Image (-1, 0, 0) then sits that far outside the face x = 0, beyond the bare
#: envelope radius (about 2.10 Bohr at 1e-7) and inside the normalised leaf radius
#: (about 2.27 Bohr), so the leaf keeps it and the knot (0, edge / 2, edge / 2) sees it.
_SPLINE_LEAF_EDGE = 6.0
_SPLINE_LEAF_FACE_GAP = 2.15
_SPLINE_LEAF_EXPONENT = 4.0
_SPLINE_LEAF_FFT = 24


@pytest.mark.light
def test_pgto_spline_composite_reproduces_band_at_knots_with_leaf_image_set():
    """PGTO leaf + spline of ``project_pw_to_pgto_spline`` equals the PW band at the spline knots.

    The band is the periodic sum of a tight d primitive, given by its analytic
    Fourier coefficients. The composite orbital at the spline knots is the
    emitted pGTO leaf (each normalised AO summed over the images the built leaf
    keeps, read from its ragged layout, times its MO coefficient) plus the
    spline (which interpolates its target at the knots). It must equal the band
    within ``AO_IMAGE_TRUNCATION_ATOL`` per unit normalised MO coefficient: the
    spline target subtracts the pGTO with the same image selection as the leaf.
    With the bare envelope tolerance on the raw primitive, the subtraction drops
    image (-1, 0, 0), which the normalisation-aware leaf keeps, and the
    composite is off by about 8 times that bound at the face knot.
    """
    from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL
    from jqmc.pw_pgto_projection import _cell_normalization, cart_primitive_normalisations, project_pw_to_pgto_spline
    from jqmc.qe_to_jqmc import _build_pgto_leaf_and_mo
    from jqmc.qe_wrapper import QEBands, QEReference, QEWavefunction
    from jqmc.structure import Structure_data

    edge, alpha, n = _SPLINE_LEAF_EDGE, _SPLINE_LEAF_EXPONENT, _SPLINE_LEAF_FFT
    cell = QECell(a=edge * np.eye(3))
    position = np.array([edge - _SPLINE_LEAF_FACE_GAP, 0.5 * edge, 0.5 * edge])
    millers = np.array(list(itertools.product(range(-n // 2 + 1, n // 2), repeat=3)), dtype=np.int64)
    g = millers.astype(np.float64) @ cell.b
    g2 = np.einsum("ij,ij->i", g, g)
    # Fourier transform of x^2 exp(-alpha r^2): (pi/alpha)^(3/2) exp(-G^2 / 4 alpha) (1/(2 alpha) - G_x^2 / (4 alpha^2)).
    transform = (np.pi / alpha) ** 1.5 * np.exp(-g2 / (4.0 * alpha)) * (0.5 / alpha - g[:, 0] ** 2 / (4.0 * alpha**2))
    band = (transform * np.exp(-1j * (g @ position)))[None, :].astype(np.complex128)
    reference = QEReference(
        cell=cell,
        atom_labels=("O",),
        atom_positions=position[None, :],
        ecutwfc=0.5 * float(g2.max()),
        fft_grid=(n, n, n),
        bands=QEBands(eigenvalues=np.zeros(1), occupations=np.ones(1), n_electrons=1.0, fermi_energy=None),
        wfc=QEWavefunction(millers=millers, coeffs=band, gamma_only=False, k_point=np.zeros(3)),
        atom_elements=("O",),
    )
    shells = [
        [PrimitiveShell(angular_l=2, exponents=np.array([alpha]), coefficients=np.ones(1), rel_error=0.0, overlap_cond=1.0)]
    ]
    result = project_pw_to_pgto_spline(reference, shells, mesh_factor=1.0, order=5)

    structure = Structure_data.build(
        pbc_flag=True,
        positions=position[None, :],
        vec_a=tuple(cell.a[0]),
        vec_b=tuple(cell.a[1]),
        vec_c=tuple(cell.a[2]),
        atomic_numbers=(8,),
        element_symbols=("O",),
        atomic_labels=("O",),
    )
    leaf, mo = _build_pgto_leaf_and_mo(result, structure, reference.atom_positions, None)
    images = np.asarray(leaf._image_indices, dtype=np.int64).reshape(-1, 3)
    home = np.asarray(leaf._ragged_kernel_to_home, dtype=np.int64)
    assert (-1, 0, 0) in {tuple(int(v) for v in image) for image in images}, "witness: the leaf must keep image (-1, 0, 0)"

    knots = grid_points(cell, result.spline_shape)
    norms = cart_primitive_normalisations(result.primitives)
    leaf_value = np.zeros(knots.shape[0], dtype=np.float64)
    for image, ao in zip(images, home, strict=True):
        primitive = result.primitives[int(ao)]
        d = knots - (np.asarray(primitive.center) + image.astype(np.float64) @ cell.a)
        polynomial = d[:, 0] ** primitive.ang[0] * d[:, 1] ** primitive.ang[1] * d[:, 2] ** primitive.ang[2]
        leaf_value += mo[0, ao] * norms[ao] * polynomial * np.exp(-primitive.exponent * np.einsum("ij,ij->i", d, d))
    spline_value = evaluate_spline_orbitals(
        result.spline_coefficients.reshape((1, *result.spline_shape)), result.spline_shape, result.spline_order
    )[:, 0]
    band_value = _cell_normalization(reference) * (np.exp(1j * (knots @ g.T)) @ band[0])
    error = np.abs(leaf_value + spline_value - band_value)
    bound = AO_IMAGE_TRUNCATION_ATOL * float(np.sum(np.abs(mo[0])))
    assert float(np.max(error)) <= bound, (
        f"pGTO leaf + spline differs from the PW band by {float(np.max(error)):.3e} at knot "
        f"{knots[int(np.argmax(error))]}, above AO_IMAGE_TRUNCATION_ATOL * sum|MO| = {bound:.3e}: the spline target "
        "subtracts a different pGTO image set than the leaf keeps"
    )
