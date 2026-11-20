"""Unit tests for jqmc.qe_to_jqmc (QE projection -> jqmc AO/MO/Geminal)."""

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

from pathlib import Path

import numpy as np
import pytest

from jqmc._precision import get_complex_dtype_np, get_dtype_np, get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import (
    ATOL_NONVACUITY_DISTINCTION,
    ATOL_PYTHON_FP64_LOOSE,
    RTOL_BIT_EXACT,
    RTOL_NONVACUITY_DISTINCTION,
    RTOL_QE_COMPOSITE_PER_BAND_REGRESSION_GATE,
    RTOL_QE_PW_RECONSTRUCTION_TIGHT,
)
from tests._test_data_cache import qe_example_files_root

pytestmark = [pytest.mark.with_chdir]
_SAVE_DIR = qe_example_files_root() / "Li_gamma" / "out" / "li.save"
_XML = _SAVE_DIR / "data-file-schema.xml"
_WFC = _SAVE_DIR / "wfc1.hdf5"
_UPF = _SAVE_DIR / "Li.ccECP.upf"

_available = _XML.is_file() and _WFC.is_file() and _UPF.is_file()
requires_qe = pytest.mark.skipif(not _available, reason="QE Li_gamma sample not present")


# ---------------------------------------------------------------------------
# Orbital-vs-PW regression: catches conversion bugs that internal projection
# fidelity checks miss because they self-measure (the prior buggy
# ``project_pw_to_pgto_spline`` reported residual ~1e-9 yet the jQMC composite
# orbital differed from PW by ~1.5% per band, leaking ~mHa in MCMC).
#
# For each conversion kind we build the jQMC AO/MO via the public
# ``build_wavefunction_from_projection``, evaluate the occupied MOs at random
# real-space points using production ``compute_MOs``, and compare per-band
# against the independent PW reconstruction up to a per-band
# global scale (VMC energy is invariant under per-orbital rescaling). Any
# composite that does not reproduce psi up to that rescale fails here.
# ---------------------------------------------------------------------------

# Points per block of the zone-dtype plane-wave sum (bounds the phase block).
_PW_BAND_POINT_CHUNK = 4096


def _pw_band_values_in_zones(coeffs, gk_cart, r_carts):
    """Independent QE band sum ``psi_b(r) = sum_G c_{b,G} exp(i (G + k) . r)`` in the production zones.

    Same-precision reference, shared by the QE fidelity tests. The QE
    coefficients ``coeffs`` ``(n_band, n_g)``, the Cartesian ``gk_cart =
    G + k`` ``(n_g, 3)`` and the points ``r_carts`` ``(n_pts, 3)`` are fp64
    input data. Each piece is evaluated in the dtype of the production zone
    that computes it: the phase argument ``(G + k) . r`` is geometry,
    reconstructed in fp64 and then cast to the ``ao_eval`` dtype; the
    plane-wave value ``exp(i theta) = cos(theta) + i sin(theta)`` is evaluated
    in that ``ao_eval`` dtype (the components of the ``ao_eval`` complex
    dtype; complex64 in mixed), as the plane-wave AO leaves evaluate their
    values; the coefficient contraction runs in the ``mo_eval`` dtype, as the
    MO layer contracts its coefficients with ``ao_eval`` AO values. In full
    precision every step is fp64. No production evaluator is called.
    Returns ``(n_band, n_pts)`` in the ``mo_eval`` complex dtype.
    """
    ao_dtype = get_dtype_np("ao_eval")
    mo_dtype = get_dtype_np("mo_eval")
    mo_complex = get_complex_dtype_np("mo_eval")
    c = np.asarray(coeffs).astype(mo_complex)
    c_re, c_im = np.ascontiguousarray(c.real), np.ascontiguousarray(c.imag)
    gk = np.asarray(gk_cart, dtype=np.float64)
    r = np.asarray(r_carts, dtype=np.float64)
    out = np.empty((c.shape[0], r.shape[0]), dtype=mo_complex)
    for start in range(0, r.shape[0], _PW_BAND_POINT_CHUNK):
        theta = (gk @ r[start : start + _PW_BAND_POINT_CHUNK].T).astype(ao_dtype)  # (n_g, n_chunk)
        cos_v = np.cos(theta).astype(mo_dtype)
        sin_v = np.sin(theta).astype(mo_dtype)
        block = out[:, start : start + _PW_BAND_POINT_CHUNK]
        block.real = c_re @ cos_v - c_im @ sin_v
        block.imag = c_re @ sin_v + c_im @ cos_v
    return out


def _pw_reference_at(ref, r_carts, band_idx):
    """PW reconstruction of QE bands at arbitrary Cartesian points (Gamma).

    Evaluates ``psi(r) = sum_G c_G exp(iG.r)`` directly (no FFT grid) for the
    selected band indices, in the production precision zones (see
    :func:`_pw_band_values_in_zones`). Returns ``(n_band, n_probe)``.
    """
    G_vectors = ref.wfc.millers.astype(np.float64) @ ref.cell.b  # (n_g, 3)
    return _pw_band_values_in_zones(ref.wfc.coeffs[band_idx], G_vectors, r_carts)  # (n_band, n_probe), complex


def _per_band_rescale_residual(ref_vals, vals):
    """Per-band: scalar c_b minimising ||vals - c_b * ref||^2; return rel residual.

    Both arrays must be finite, nonempty ``(n_probe, n_band)`` real arrays
    with a nonzero norm in every band. A zero orbital cannot become evidence
    by choosing a zero scale. Normalize each column before fitting to avoid
    underflow or overflow without introducing an absolute residual floor.
    """
    ref_vals, vals = np.asarray(ref_vals), np.asarray(vals)
    assert ref_vals.ndim == vals.ndim == 2, "orbital fidelity: expected two-dimensional arrays"
    assert ref_vals.shape == vals.shape, "orbital fidelity: reference and converted shapes differ"
    assert ref_vals.size > 0, "orbital fidelity: sampled arrays must be nonempty"
    assert np.isrealobj(ref_vals) and np.isrealobj(vals), "orbital fidelity: expected real orbital values"
    assert np.all(np.isfinite(ref_vals)) and np.all(np.isfinite(vals)), "orbital fidelity: nonfinite orbital values"
    n_band = ref_vals.shape[1]
    out = np.empty(n_band, dtype=np.float64)
    for b in range(n_band):
        assert np.any(ref_vals[:, b]), "orbital fidelity: reference band has zero norm"
        assert np.any(vals[:, b]), "orbital fidelity: converted band has zero norm"
        ref_scale = float(np.max(np.abs(ref_vals[:, b])))
        val_scale = float(np.max(np.abs(vals[:, b])))
        x = ref_vals[:, b] / ref_scale
        y = vals[:, b] / val_scale
        denom = float(np.dot(x, x))
        c = float(np.dot(x, y) / denom)
        out[b] = float(np.linalg.norm(y - c * x) / np.linalg.norm(y))
    return out


# ``pgto+pw`` is not in this set. By construction the periodic pGTO image
# sum carries Fourier content beyond the wfc G-set, and the PW block (a
# discrete subset of wfc G's) cannot represent that high-G tail. The
# composite ``pgto(r) + pw(r)`` therefore differs from psi by the high-G
# leak. That difference is the kind's nature, not a conversion bug; the
# proper consistency check for ``pgto+pw`` is the Fourier-vs-real backend
# agreement test ``test_fourier_vs_real_mixed_cu``.
_LOSSLESS_KINDS = ["pw", "bspline", "pgto+bspline"]


# ---------------------------------------------------------------------------
# Wavefunction builders: build_wavefunction_from_projection across kinds.
# ---------------------------------------------------------------------------
class TestQEToJQMCWavefunction:
    @pytest.mark.light
    def test_sd_occupation_truncates_h128_like_nbnd72_to_64_rows_per_twist(self):
        """Projection rank may be 72, but an SD with 64 occupied orbitals per spin stores 64 rows."""
        from jqmc.atomic_orbital import AOs_sphe_gto_data
        from jqmc.molecular_orbital import MOs_data
        from jqmc.qe_to_jqmc import _make_geminal_from_mos
        from jqmc.structure import Structure_data

        nbnd, nocc = 72, 64
        structure = Structure_data(
            pbc_flag=False,
            positions=np.zeros((1, 3)),
            atomic_numbers=(1,),
            element_symbols=("X",),
            atomic_labels=("X",),
        )
        aos = AOs_sphe_gto_data._from_full_basis(
            structure_data=structure,
            nucleus_index=(0,) * nbnd,
            num_ao=nbnd,
            num_ao_prim=nbnd,
            orbital_indices=tuple(range(nbnd)),
            exponents=np.ones(nbnd),
            coefficients=np.ones(nbnd),
            angular_momentums=(0,) * nbnd,
            magnetic_quantum_numbers=(0,) * nbnd,
        )
        occupations = np.concatenate([np.ones(nocc), np.zeros(nbnd - nocc)])

        for twist_scale in (1.0, 2.0):
            mos = MOs_data(
                num_mo=nbnd,
                aos_data=aos,
                mo_coefficients=twist_scale * np.eye(nbnd),
            )
            sd = _make_geminal_from_mos(mos, occupations, 2 * nocc)
            assert sd.orb_data_up_spin.mo_coefficients.shape == (nocc, nbnd)
            assert sd.orb_data_dn_spin.mo_coefficients.shape == (nocc, nbnd)
            assert not hasattr(sd, "lambda_matrix")

    @pytest.mark.light
    def test_sd_restricted_conversion_rejects_too_few_occupied_bands(self):
        """QE conversion must not silently reduce the physical electron count."""
        from jqmc.atomic_orbital import AOs_sphe_gto_data
        from jqmc.molecular_orbital import MOs_data
        from jqmc.qe_to_jqmc import _make_geminal_from_mos
        from jqmc.structure import Structure_data

        nbnd = 3
        structure = Structure_data(
            pbc_flag=False,
            positions=np.zeros((1, 3)),
            atomic_numbers=(1,),
            element_symbols=("X",),
            atomic_labels=("X",),
        )
        aos = AOs_sphe_gto_data._from_full_basis(
            structure_data=structure,
            nucleus_index=(0,) * nbnd,
            num_ao=nbnd,
            num_ao_prim=nbnd,
            orbital_indices=tuple(range(nbnd)),
            exponents=np.ones(nbnd),
            coefficients=np.ones(nbnd),
            angular_momentums=(0,) * nbnd,
            magnetic_quantum_numbers=(0,) * nbnd,
        )
        mos = MOs_data(num_mo=nbnd, aos_data=aos, mo_coefficients=np.eye(nbnd))
        occupations = np.array([1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match=r"only 3 occupied up bands.*ne_up=4"):
            _make_geminal_from_mos(mos, occupations, n_electrons_total=8.0)

    @pytest.mark.light
    @requires_qe
    def test_build_wavefunction_pw(self):
        """kind='pw': PW-only projection -> AOs_plane_wave_data + Geminal_data."""
        from jqmc.atomic_orbital import AOs_plane_wave_data
        from jqmc.determinant import AGP_data, AGPn_data, SD_data, compute_geminal_all_elements
        from jqmc.molecular_orbital import MOs_data, compute_MOs
        from jqmc.pw_pgto_projection import project_pw_to_pw
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.structure import Structure_data

        ref = load_qe_reference(str(_SAVE_DIR))
        proj = project_pw_to_pw(ref, g_cut=None)  # exact PW -> PW
        structure_data, aos_data, geminal_data = build_wavefunction_from_projection(ref, proj, kind="pw")

        # Structure_data: periodic with the QE cell rows.
        assert isinstance(structure_data, Structure_data)
        assert structure_data.pbc_flag is True
        np.testing.assert_array_equal(np.asarray(structure_data.vec_a), ref.cell.a[0])
        np.testing.assert_array_equal(np.asarray(structure_data.vec_b), ref.cell.a[1])
        np.testing.assert_array_equal(np.asarray(structure_data.vec_c), ref.cell.a[2])
        assert tuple(structure_data.element_symbols) == tuple(ref.atom_labels)
        # Li has atomic number 3.
        assert tuple(structure_data.atomic_numbers) == (3,) * len(ref.atom_labels)

        # AO leaf: plane-wave with real cos/sin basis.
        assert isinstance(aos_data, AOs_plane_wave_data)
        # Constant + cos/sin pairs per non-zero G: num_ao approximately n_pw
        # (G=0 contributes 1; (G, -G) pair contributes 2). The total equals n_pw.
        assert aos_data.num_ao == ref.wfc.n_gvectors

        # The QE -> jqmc projection path produces an MO-basis geminal with
        # ``[1]*N_occ + [0]*N_virt`` diagonal lambda; the load-time dispatch routes
        # the closed-shell case to ``SD_data``. ``AGPn_data`` is the target
        # when the loaded diagonal carries non-trivial fractional occupations,
        # which the projection writer does not currently produce but which is
        # reachable on a hand-edited input.
        # (Li.ccECP Z_valence=1 x 2 atoms -> 2 valence electrons -> ne_up=ne_dn=1, RHF.)
        assert isinstance(geminal_data, (AGP_data, AGPn_data, SD_data))
        assert isinstance(geminal_data.orb_data_up_spin, MOs_data)
        assert isinstance(geminal_data.orb_data_dn_spin, MOs_data)
        assert geminal_data.num_electron_up == 1
        assert geminal_data.num_electron_dn == 1

        # MO evaluation should be finite at a few interior points.
        mos = geminal_data.orb_data_up_spin
        rng = np.random.default_rng(0)
        r_probe = rng.uniform(-2.0, 2.0, size=(4, 3)).astype(np.float64)
        mo_vals = np.asarray(compute_MOs(mos, r_probe))
        assert mo_vals.shape == (mos.num_mo, r_probe.shape[0])
        assert np.all(np.isfinite(mo_vals))

        # Geminal matrix at a Slater-fillable configuration.
        r_up = rng.uniform(-2.0, 2.0, size=(geminal_data.num_electron_up, 3)).astype(np.float64)
        r_dn = rng.uniform(-2.0, 2.0, size=(geminal_data.num_electron_dn, 3)).astype(np.float64)
        gem = np.asarray(compute_geminal_all_elements(geminal_data, r_up, r_dn))
        assert gem.shape == (geminal_data.num_electron_up, geminal_data.num_electron_up)
        assert np.all(np.isfinite(gem))

    @pytest.mark.light
    @requires_qe
    def test_build_wavefunction_pw_reproduces_pw_band(self):
        """The plane-wave MO must reproduce the QE band on a grid."""
        from jqmc.molecular_orbital import compute_MOs
        from jqmc.pw_pgto_projection import grid_points, project_pw_to_pw
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection
        from jqmc.qe_wrapper import load_qe_reference

        ref = load_qe_reference(str(_SAVE_DIR))
        proj = project_pw_to_pw(ref, g_cut=None)
        _, _, geminal_data = build_wavefunction_from_projection(ref, proj, kind="pw")

        # Sample on the QE FFT grid (always safe for the stored Miller indices).
        fft_grid = ref.fft_grid
        grid = grid_points(ref.cell, fft_grid)
        mos = geminal_data.orb_data_up_spin
        mo_vals = np.asarray(compute_MOs(mos, grid))  # (n_mo, n_pts)

        # Converted MOs are unit-cell normalized: the raw
        # ``sum_G c_G e^{iG.r}`` reconstruction carries an extra sqrt(Omega), so
        # the absolute reference is divided by it. The reference is the
        # direct band sum at the grid points evaluated in the production zones
        # (plane-wave values in ao_eval, contraction in mo_eval), not the fp64
        # FFT reconstruction; in full precision the two agree to fp64 round-off.
        band_idx = proj.band_indices
        psi_occ = _pw_reference_at(ref, grid, band_idx) / np.sqrt(ref.cell.volume)  # (n_band, n_pts)
        # The PW real basis -> band reconstruction matches the real part of psi.
        # At Gamma the orbital is real up to global phase; expect agreement on Re().
        expected = np.real(psi_occ)  # (n_band, n_pts)
        rel = np.linalg.norm(mo_vals - expected) / max(np.linalg.norm(expected), 1.0e-30)
        reconstruction_rtol = (
            get_tolerance_min(("ao_eval", "mo_eval"), "strict")[1]
            if is_mixed_precision_enabled()
            else RTOL_QE_PW_RECONSTRUCTION_TIGHT
        )
        assert rel < reconstruction_rtol, f"PW MO reconstruction rel error too large: {rel:.3e}"

    @pytest.mark.light
    @requires_qe
    def test_build_wavefunction_bspline(self):
        """kind='bspline': per-band whole-cell spline."""
        from jqmc.atomic_orbital import AOs_bspline_data
        from jqmc.molecular_orbital import compute_MOs
        from jqmc.pw_pgto_projection import project_pw_to_spline
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection
        from jqmc.qe_wrapper import load_qe_reference

        ref = load_qe_reference(str(_SAVE_DIR))
        proj = project_pw_to_spline(ref, mesh_factor=1.0, order=5)
        _, aos_data, geminal_data = build_wavefunction_from_projection(ref, proj, kind="bspline")

        assert isinstance(aos_data, AOs_bspline_data)
        assert aos_data.spline_order == 5
        assert aos_data.spline_shape == ref.fft_grid
        # Each AO IS a band, so num_ao == num_mo == # occupied bands projected.
        n_band = int(proj.band_indices.size)
        assert aos_data.num_ao == n_band
        assert geminal_data.orb_data_up_spin.num_mo == n_band

        # MO values are finite.
        r_carts = np.array([[1.0, 1.5, 2.0], [-0.3, 0.7, 1.2]], dtype=np.float64)
        mo_vals = np.asarray(compute_MOs(geminal_data.orb_data_up_spin, r_carts))
        assert np.all(np.isfinite(mo_vals))

    @pytest.mark.light
    @requires_qe
    def test_build_wavefunction_bspline_force_complex_gamma_is_opt_in(self):
        """Multi-twist storage may encode Gamma as k=(0,0,0) complex.

        The ordinary single-k Gamma path must remain the established real
        representation; only an explicit request from the twist converter may
        select the uniform complex representation.
        """
        from jqmc.atomic_orbital import AOs_bspline_complex_data, AOs_bspline_data
        from jqmc.pw_pgto_projection import project_pw_to_spline
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection
        from jqmc.qe_wrapper import load_qe_reference

        ref = load_qe_reference(str(_SAVE_DIR))
        proj = project_pw_to_spline(ref, mesh_factor=0.5, order=5)

        _, aos_real, _ = build_wavefunction_from_projection(ref, proj, kind="bspline")
        _, aos_complex, gem_complex = build_wavefunction_from_projection(
            ref,
            proj,
            kind="bspline",
            force_complex_gamma=True,
        )

        assert isinstance(aos_real, AOs_bspline_data)
        assert isinstance(aos_complex, AOs_bspline_complex_data)
        assert tuple(aos_complex.k_point) == (0.0, 0.0, 0.0)
        assert np.iscomplexobj(np.asarray(gem_complex.orb_data_up_spin.mo_coefficients))

    @pytest.mark.light
    def test_build_tiled_bspline_wavefunction_unfolds_two_k_blocks_with_implicit_mos(self, tmp_path):
        """A 2x1x1 k mesh becomes one Gamma supercell without a supercell spline grid."""
        from jqmc.atomic_orbital import AOs_bspline_tiled_complex_data
        from jqmc.pw_pgto_projection import ProjectionResult
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction, build_tiled_bspline_wavefunction
        from jqmc.qe_wrapper import QEBands, QECell, QEReference, QEWavefunction

        cell = QECell(a=2.0 * np.eye(3, dtype=np.float64))
        shape = (6, 6, 6)
        rng = np.random.default_rng(194)

        def reference(k_frac):
            k_cart = np.asarray(k_frac, dtype=np.float64) @ cell.b
            return QEReference(
                cell=cell,
                atom_labels=("He",),
                atom_positions=np.array([[0.2, 0.3, 0.4]], dtype=np.float64),
                ecutwfc=10.0,
                fft_grid=shape,
                bands=QEBands(
                    eigenvalues=np.array([-0.5]),
                    occupations=np.array([1.0]),
                    n_electrons=2.0,
                    fermi_energy=0.0,
                ),
                wfc=QEWavefunction(
                    millers=np.array([[0, 0, 0]], dtype=np.int64),
                    coeffs=np.ones((1, 1), dtype=np.complex128),
                    gamma_only=False,
                    k_point=k_cart,
                ),
                atom_elements=("He",),
            )

        def projection(scale):
            coefficients = scale * (rng.normal(size=(1, np.prod(shape))) + 1j * rng.normal(size=(1, np.prod(shape))))
            return ProjectionResult(
                residuals=np.zeros(1),
                band_indices=np.array([0], dtype=np.int64),
                occupations=np.ones(1),
                coefficients=np.empty((1, 0), dtype=np.complex128),
                primitives=(),
                n_gto=0,
                n_pw=0,
                n_dropped=0,
                s_cond=1.0,
                s_cond_full=1.0,
                s_eig_min=1.0,
                spline_coefficients=coefficients,
                spline_shape=shape,
                spline_order=5,
                mesh_factor=1.0,
            )

        refs = [reference((0.0, 0.0, 0.0)), reference((0.5, 0.0, 0.0))]
        projections = [projection(1.0), projection(0.7 - 0.2j)]
        structure, aos, sd = build_tiled_bspline_wavefunction(
            refs,
            projections,
            tiling_matrix=(2, 1, 1),
        )

        assert isinstance(aos, AOs_bspline_tiled_complex_data)
        np.testing.assert_array_equal(structure.cell, np.diag([4.0, 2.0, 2.0]))
        assert len(structure.atomic_numbers) == 2
        assert aos.spline_shape == shape
        assert aos.spline_coefficients.shape == (2, np.prod(shape))
        assert sd.num_electron_up == sd.num_electron_dn == 2
        assert sd.orb_data_up_spin.identity and sd.orb_data_dn_spin.identity
        assert np.asarray(sd.orb_data_up_spin.mo_coefficients).size == 0
        assert sd.orb_data_dn_spin.aos_data.conjugate_coefficients

        hamiltonian = build_hamiltonian_from_wavefunction(
            structure,
            aos,
            sd,
            j1_param=None,
            j2_param=None,
        )
        path = tmp_path / "tiled.h5"
        hamiltonian.save_to_hdf5(path)
        loaded = Hamiltonian_data.load_from_hdf5(path)
        loaded_up = loaded.wavefunction_data.geminal_data.orb_data_up_spin
        assert loaded_up.identity
        assert np.asarray(loaded_up.mo_coefficients).size == 0
        assert isinstance(loaded_up.aos_data, AOs_bspline_tiled_complex_data)
        np.testing.assert_allclose(
            loaded_up.aos_data.primitive_cell_np,
            cell.a,
            rtol=RTOL_BIT_EXACT,
            atol=ATOL_PYTHON_FP64_LOOSE,
        )

    @pytest.mark.light
    def test_build_tiled_bspline_lsda_down_leaf_is_conjugated_beta_partner(self):
        """Tiled LSDA B-splines: the down leaf is beta's own tiled leaf at the opposite twists.

        QE ``O_tiling_sectors`` (LSDA, N_up = 4 > N_dn = 2) at the primitive k
        points (0, 0, 0) and (1/2, 0, 0) unfold to the Gamma 2x1x1 supercell.
        The down leaf must be the ``with_negated_k()`` partner of the tiled
        leaf built from the occupied beta splines at +k: negated primitive k,
        conjugated control points and beta's (not alpha's) payload, so its
        values are the conjugates of the beta values. Constructing the SD runs
        the opposite-twist sanity check.
        """
        from jqmc.atomic_orbital import AOs_bspline_tiled_complex_data
        from jqmc.pw_pgto_projection import project_pw_to_spline
        from jqmc.qe_to_jqmc import build_tiled_bspline_wavefunction
        from jqmc.qe_wrapper import load_qe_reference

        save = _SAVE_DIR.parents[2] / "O_tiling_sectors" / "out" / "o_tiling.save"
        spins = {spin: [load_qe_reference(str(save), ik=ik, spin=spin) for ik in (1, 2)] for spin in ("up", "dw")}
        projections = {spin: [project_pw_to_spline(ref, mesh_factor=0.5) for ref in refs] for spin, refs in spins.items()}
        structure, up_leaf, sd = build_tiled_bspline_wavefunction(
            spins["up"],
            projections["up"],
            tiling_matrix=(2, 1, 1),
            qe_references_dn=spins["dw"],
            projection_results_dn=projections["dw"],
        )
        assert not sd.time_reversal and (sd.num_electron_up, sd.num_electron_dn) == (8, 4)
        dn_leaf = sd.orb_data_dn_spin.aos_data

        def occupied(projection):
            return np.asarray(projection.spline_coefficients)[np.asarray(projection.occupations) > 0.5]

        beta_plus_k = AOs_bspline_tiled_complex_data.build(
            structure_data=structure,
            spline_coefficients_by_k=[occupied(p) for p in projections["dw"]],
            spline_shape=up_leaf.spline_shape,
            tiling_matrix=(2, 1, 1),
            k_points=up_leaf.k_points,
            spline_order=up_leaf.spline_order,
            supercell_k_point=(0.0, 0.0, 0.0),
        )
        np.testing.assert_array_equal(np.asarray(dn_leaf.k_points), -np.asarray(beta_plus_k.k_points))
        assert dn_leaf.conjugate_coefficients and not up_leaf.conjugate_coefficients
        np.testing.assert_array_equal(np.asarray(dn_leaf.spline_coefficients), np.asarray(beta_plus_k.spline_coefficients))
        alpha_rows = np.asarray(up_leaf.spline_coefficients)[: dn_leaf.num_ao]
        assert not np.allclose(
            np.abs(alpha_rows),
            np.abs(np.asarray(dn_leaf.spline_coefficients)),
            atol=ATOL_NONVACUITY_DISTINCTION,
            rtol=RTOL_NONVACUITY_DISTINCTION,
        )
        r = np.random.default_rng(61).uniform(0.0, 1.0, size=(6, 3)) @ np.asarray(structure.cell)
        beta_values = np.asarray(beta_plus_k._compute_AOs(r))
        atol, rtol = get_tolerance_min(("ao_eval", "ao_bloch_eval"), "strict")
        np.testing.assert_allclose(np.asarray(dn_leaf._compute_AOs(r)), np.conj(beta_values), atol=atol, rtol=rtol)
        assert np.max(np.abs(np.imag(beta_values))) > atol

    @pytest.mark.light
    @requires_qe
    def test_build_wavefunction_pgto(self):
        """kind='pgto': UPF-seeded pGTO basis projection."""
        from jqmc.atomic_orbital import AOs_cart_pgto_data
        from jqmc.molecular_orbital import compute_MOs
        from jqmc.pw_pgto_projection import project_pw_to_pgto
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.upf_wrapper import UPFExponentInit

        ref = load_qe_reference(str(_SAVE_DIR))
        strat = UPFExponentInit({"li": str(_UPF)}, g_target=float(np.sqrt(2.0 * ref.ecutwfc)))
        shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]
        proj = project_pw_to_pgto(ref, shells)
        _, aos_data, geminal_data = build_wavefunction_from_projection(ref, proj, kind="pgto")

        assert isinstance(aos_data, AOs_cart_pgto_data)
        # All-primitive: one primitive per molecular AO, so the molecular num_ao
        # matches proj.n_gto. The periodic image expansion is
        # per-shell RAGGED, so the kernel rank is M = _num_ao_kernel (each shell
        # keeps only its own radial's images); with one primitive per kernel-AO,
        # num_ao_prim == _num_ao_kernel -- strictly between num_ao and the old
        # uniform ``n_gto * _n_image`` tiling.
        assert aos_data.num_ao == proj.n_gto
        assert aos_data.num_ao_prim == aos_data._num_ao_kernel
        assert aos_data.num_ao < aos_data._num_ao_kernel <= proj.n_gto * aos_data._n_image

        # MO values finite.
        r_carts = np.array([[1.0, 1.5, 2.0], [-0.3, 0.7, 1.2]], dtype=np.float64)
        mo_vals = np.asarray(compute_MOs(geminal_data.orb_data_up_spin, r_carts))
        assert np.all(np.isfinite(mo_vals))

    @pytest.mark.light
    @requires_qe
    def test_build_wavefunction_pgto_plus_pw(self):
        """kind='pgto+pw': composite of pGTO + low-G PW columns."""
        from jqmc.atomic_orbital import AOs_composite_data
        from jqmc.molecular_orbital import compute_MOs
        from jqmc.pw_pgto_projection import project_pw_to_mixed
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.upf_wrapper import UPFExponentInit

        ref = load_qe_reference(str(_SAVE_DIR))
        strat = UPFExponentInit({"li": str(_UPF)}, g_target=float(np.sqrt(2.0 * ref.ecutwfc)))
        shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]
        g_max = float(np.sqrt(2.0 * ref.ecutwfc))
        proj = project_pw_to_mixed(ref, shells, g_cut=0.5 * g_max, drop_diffuse=False)
        _, aos_data, geminal_data = build_wavefunction_from_projection(ref, proj, kind="pgto+pw")

        assert isinstance(aos_data, AOs_composite_data)
        # MO values finite.
        r_carts = np.array([[1.0, 1.5, 2.0], [-0.3, 0.7, 1.2]], dtype=np.float64)
        mo_vals = np.asarray(compute_MOs(geminal_data.orb_data_up_spin, r_carts))
        assert np.all(np.isfinite(mo_vals))

    @pytest.mark.light
    @requires_qe
    def test_build_wavefunction_pgto_plus_bspline(self):
        """kind='pgto+bspline': composite of pGTO + per-band B-spline residual."""
        from jqmc.atomic_orbital import AOs_composite_data
        from jqmc.molecular_orbital import compute_MOs
        from jqmc.pw_pgto_projection import project_pw_to_pgto_spline
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.upf_wrapper import UPFExponentInit

        ref = load_qe_reference(str(_SAVE_DIR))
        strat = UPFExponentInit({"li": str(_UPF)}, g_target=float(np.sqrt(2.0 * ref.ecutwfc)))
        shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]
        proj = project_pw_to_pgto_spline(ref, shells, mesh_factor=0.5, order=5)
        _, aos_data, geminal_data = build_wavefunction_from_projection(ref, proj, kind="pgto+bspline")

        assert isinstance(aos_data, AOs_composite_data)
        # MO values finite at a few points.
        r_carts = np.array([[1.0, 1.5, 2.0], [-0.3, 0.7, 1.2]], dtype=np.float64)
        mo_vals = np.asarray(compute_MOs(geminal_data.orb_data_up_spin, r_carts))
        assert np.all(np.isfinite(mo_vals))


# ---------------------------------------------------------------------------
# Hamiltonian builder: build_hamiltonian_from_projection + h5 round-trip.
# ---------------------------------------------------------------------------
class TestQEToJQMCHamiltonian:
    @pytest.mark.light
    @requires_qe
    def test_build_hamiltonian_pw_h5_roundtrip(self, tmp_path):
        """kind='pw': Hamiltonian_data save/load round-trip with QE-derived data."""
        from jqmc._setting import get_tol_hdf5_roundtrip
        from jqmc.atomic_orbital import AOs_plane_wave_data
        from jqmc.coulomb_potential import Ewald_data
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.molecular_orbital import MOs_data
        from jqmc.pw_pgto_projection import project_pw_to_pw
        from jqmc.qe_to_jqmc import build_hamiltonian_from_projection
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.upf_wrapper import parse_gaussinfo

        ref = load_qe_reference(str(_SAVE_DIR))
        proj = project_pw_to_pw(ref, g_cut=None)
        # The ECP comes from the save's own UPF (the one jqmc-tool qe
        # auto-discovers). Without an ECP source this pseudopotential save is
        # rejected (issue #99) instead of getting a zero-coefficient placeholder.
        hamiltonian = build_hamiltonian_from_projection(
            ref,
            proj,
            kind="pw",
            j1_param=None,
            j2_param=1.0,
            j2_type="pade",
            ewald_tol=1.0e-8,
            ecp_by_element={"Li": parse_gaussinfo(str(_UPF))},
        )
        hamiltonian.sanity_check()

        # Coulomb / Ewald wiring: pbc_flag iff ewald_data is not None. The QE
        # Li sample uses ccECP (3 protons / 1 valence electron per atom), so
        # every atom carries the UPF's z_core = 2 and ecp_flag is on.
        cp = hamiltonian.coulomb_potential_data
        assert cp.ecp_flag is True
        assert tuple(cp.z_cores) == (2.0,) * len(cp.z_cores)
        assert np.count_nonzero(np.asarray(cp.coefficients)) == cp.num_ecps > 0
        assert cp.structure_data.pbc_flag is True
        assert isinstance(cp.ewald_data, Ewald_data)

        # Wavefunction / Jastrow / Geminal wiring. bcc Li with 2 atoms x 1
        # valence electron = 2 valence electrons total -> RHF ne_up = ne_dn = 1.
        wf = hamiltonian.wavefunction_data
        jd = wf.jastrow_data
        assert jd.jastrow_one_body_data is None
        assert jd.jastrow_two_body_data is not None
        assert float(jd.jastrow_two_body_data.jastrow_2b_param) == pytest.approx(1.0)
        gd = wf.geminal_data
        assert gd.num_electron_up == 1
        assert gd.num_electron_dn == 1
        assert isinstance(gd.orb_data_up_spin, MOs_data)
        assert isinstance(gd.orb_data_up_spin.aos_data, AOs_plane_wave_data)

        # Round-trip through HDF5.
        h5_file = tmp_path / "qe_pw_hamiltonian.h5"
        hamiltonian.save_to_hdf5(str(h5_file))
        assert h5_file.is_file()
        file_size = h5_file.stat().st_size
        assert file_size > 0

        loaded = Hamiltonian_data.load_from_hdf5(str(h5_file))
        loaded.sanity_check()
        atol_hdf5, rtol_hdf5 = get_tol_hdf5_roundtrip("strict")

        # Structure preserved (lattice rows + atom positions).
        np.testing.assert_allclose(
            np.asarray(loaded.structure_data.positions),
            np.asarray(hamiltonian.structure_data.positions),
            atol=atol_hdf5,
            rtol=rtol_hdf5,
        )
        np.testing.assert_allclose(
            np.asarray(loaded.structure_data.vec_a),
            np.asarray(hamiltonian.structure_data.vec_a),
            atol=atol_hdf5,
            rtol=rtol_hdf5,
        )
        np.testing.assert_allclose(
            np.asarray(loaded.structure_data.vec_b),
            np.asarray(hamiltonian.structure_data.vec_b),
            atol=atol_hdf5,
            rtol=rtol_hdf5,
        )
        np.testing.assert_allclose(
            np.asarray(loaded.structure_data.vec_c),
            np.asarray(hamiltonian.structure_data.vec_c),
            atol=atol_hdf5,
            rtol=rtol_hdf5,
        )
        assert loaded.structure_data.pbc_flag is True

        # Coulomb preserved + Ewald rebuilt on load (strict PBC contract).
        assert loaded.coulomb_potential_data.ecp_flag is True
        assert tuple(loaded.coulomb_potential_data.z_cores) == tuple(cp.z_cores)
        assert tuple(loaded.coulomb_potential_data.ang_moms) == tuple(cp.ang_moms)
        assert tuple(loaded.coulomb_potential_data.nucleus_index) == tuple(cp.nucleus_index)
        assert tuple(loaded.coulomb_potential_data.powers) == tuple(cp.powers)
        np.testing.assert_allclose(
            np.asarray(loaded.coulomb_potential_data.exponents),
            np.asarray(cp.exponents),
            atol=atol_hdf5,
            rtol=rtol_hdf5,
        )
        np.testing.assert_allclose(
            np.asarray(loaded.coulomb_potential_data.coefficients),
            np.asarray(cp.coefficients),
            atol=atol_hdf5,
            rtol=rtol_hdf5,
        )
        assert isinstance(loaded.coulomb_potential_data.ewald_data, Ewald_data)

        # Canonical SD preserved: electron counts + occupied MO rows, with no
        # lambda dataset written to the new schema.
        loaded_gd = loaded.wavefunction_data.geminal_data
        assert loaded_gd.num_electron_up == gd.num_electron_up
        assert loaded_gd.num_electron_dn == gd.num_electron_dn
        np.testing.assert_allclose(
            np.asarray(loaded_gd.orb_data_up_spin.mo_coefficients),
            np.asarray(gd.orb_data_up_spin.mo_coefficients),
            atol=atol_hdf5,
            rtol=rtol_hdf5,
        )
        np.testing.assert_allclose(
            np.asarray(loaded_gd.orb_data_dn_spin.mo_coefficients),
            np.asarray(gd.orb_data_dn_spin.mo_coefficients),
            atol=atol_hdf5,
            rtol=rtol_hdf5,
        )
        assert loaded_gd.orb_data_up_spin.num_mo == loaded_gd.num_electron_up
        assert loaded_gd.orb_data_dn_spin.num_mo == loaded_gd.num_electron_dn

        import h5py

        with h5py.File(h5_file, "r") as f:
            sd_group = f["wavefunction_data/geminal_data"]
            assert "sd_representation" not in sd_group.attrs
            assert "lambda_matrix" not in sd_group
            assert "lambda_diag" not in sd_group
            assert "lambda_unpaired" not in sd_group

        # Jastrow preserved (J1 still off, J2 parameter matches).
        loaded_jd = loaded.wavefunction_data.jastrow_data
        assert loaded_jd.jastrow_one_body_data is None
        assert loaded_jd.jastrow_two_body_data is not None
        assert float(loaded_jd.jastrow_two_body_data.jastrow_2b_param) == pytest.approx(
            float(jd.jastrow_two_body_data.jastrow_2b_param)
        )


# ---------------------------------------------------------------------------
# Orbital fidelity: PW-reference reproduction + invalid-kind rejection.
# ---------------------------------------------------------------------------
class TestQEToJQMCOrbitalFidelity:
    @pytest.mark.light
    @requires_qe
    @pytest.mark.parametrize("kind", _LOSSLESS_KINDS)
    def test_jqmc_orbital_matches_pw_reference(self, kind):
        """Composite jQMC MOs reproduce the PW bands up to a per-band scale.

        Lossless conversions must reproduce every occupied band to well under
        1% relative residual after a per-band uniform rescale. Catches a
        projection self-measurement that under-reports orbital error.

        Both sampled reference and converted bands must be finite and nonzero;
        zero output is not an admissible per-band rescaling.

        ``pgto`` (alone) is intentionally excluded: a restricted basis is not
        expected to span psi and would dominate the failure mode rather than
        expose composite-construction bugs.
        """
        from jqmc.molecular_orbital import compute_MOs
        from jqmc.pw_pgto_projection import (
            project_pw_to_pgto_spline,
            project_pw_to_pw,
            project_pw_to_spline,
        )
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.upf_wrapper import UPFExponentInit

        ref = load_qe_reference(str(_SAVE_DIR))
        strat = UPFExponentInit({"li": str(_UPF)}, g_target=float(np.sqrt(2.0 * ref.ecutwfc)))
        shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]

        if kind == "pw":
            proj = project_pw_to_pw(ref, g_cut=None)
        elif kind == "bspline":
            proj = project_pw_to_spline(ref, mesh_factor=1.0, order=5)
        elif kind == "pgto+bspline":
            proj = project_pw_to_pgto_spline(ref, shells, mesh_factor=1.0, order=5)
        else:
            raise AssertionError(kind)

        _, _, geminal_data = build_wavefunction_from_projection(ref, proj, kind=kind)
        mos = geminal_data.orb_data_up_spin

        cell_a = np.asarray(ref.cell.a, dtype=np.float64)
        rng = np.random.default_rng(2026)
        frac = rng.uniform(0.0, 1.0, size=(8, 3))
        r_carts = frac @ cell_a  # (n_probe, 3) Cartesian Bohr

        # jQMC MOs at the probe points (n_band, n_probe).
        jqmc_vals = np.asarray(compute_MOs(mos, r_carts))
        # PW reference: select the same occupied bands the projection consumed.
        pw_ref = _pw_reference_at(ref, r_carts, proj.band_indices)
        # Real bands at Gamma are real up to a global phase; align by the
        # largest-amplitude probe point per band so the comparison is real.
        for b in range(pw_ref.shape[0]):
            k = int(np.argmax(np.abs(pw_ref[b])))
            if np.abs(pw_ref[b, k]) > 0:
                pw_ref[b] = pw_ref[b] * (np.abs(pw_ref[b, k]) / pw_ref[b, k])
        pw_ref_real = np.real(pw_ref)

        rel_resid = _per_band_rescale_residual(pw_ref_real.T, jqmc_vals.T)

        # 0.1% is a generous bound: every working backend tests well under it; the
        # prior pgto+bspline bug was 0.5%-1.7% per band, so 0.1% catches it.
        assert np.max(rel_resid) < RTOL_QE_COMPOSITE_PER_BAND_REGRESSION_GATE, (
            f"kind={kind!r}: jQMC composite differs from PW reference per band: rel_resid={rel_resid}"
        )

    @pytest.mark.light
    def test_invalid_kind_rejected(self):
        """Unknown kinds raise a clear ValueError."""
        from jqmc.pw_pgto_projection import ProjectionResult
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection

        # Build a minimal dummy QEReference / ProjectionResult tuple just to reach
        # the kind validation gate. Use a deliberately-malformed projection result;
        # the validation runs first and raises before any field is read.
        dummy_proj = ProjectionResult(
            residuals=np.zeros(0, dtype=np.float64),
            band_indices=np.zeros(0, dtype=np.int64),
            occupations=np.zeros(0, dtype=np.float64),
            coefficients=np.zeros((0, 0), dtype=np.complex128),
            primitives=(),
            n_gto=0,
            n_pw=0,
            n_dropped=0,
            s_cond=1.0,
            s_cond_full=1.0,
            s_eig_min=1.0,
        )
        with pytest.raises(ValueError):
            build_wavefunction_from_projection(None, dummy_proj, kind="nope")


# ---------------------------------------------------------------------------
# Issue #99: no silent ECP drop when the QE converter resolves no UPF.
#
# Independent reference: the Li ccECP of Bennett et al., J. Chem. Phys. 147,
# 224106 (2017), https://doi.org/10.1063/1.4995643 (Li table), written out by
# hand below (Z_eff = 1, l_max = 0). It equals the [Gaussinfo] block of the
# committed Li.ccECP.upf, but is not parsed from it. Local part, without the
# -Z_eff/r term that the bare Coulomb block carries:
#   V_loc(r) = Z_eff exp(-a r^2) / r + a Z_eff r exp(-b r^2) + g exp(-d r^2),
# non-local s channel: V_s(r) = c exp(-e r^2), applied as
#   V_s(r_iI) (1/4pi) int dOmega Psi(R_I + r_iI Omega) / Psi(r_i).
# ---------------------------------------------------------------------------
_LI_OUT_DIR = _SAVE_DIR.parent  # parent of li.save: the form issue #99 exercises (committed sample)
_LI_TWIST_SAVE_DIR = _SAVE_DIR.parents[2] / "Li_k_twist" / "out" / "li_twist.save"
_LI_CCECP_ZEFF = 1.0
_LI_CCECP_LOCAL = ((15.0, 15.047997), (-1.242730, 1.806054))  # (a, b), (g, d)
_LI_CCECP_S = (6.752868, 1.330248)  # (c, e)
# (ang_mom, exponent, coefficient, power) with jQMC's r^(power-2) convention and
# the local channel tagged ang_mom = l_max + 1 = 1.
_LI_CCECP_TERMS = sorted(
    [
        (1, 15.0, _LI_CCECP_ZEFF, 1),
        (1, 15.047997, 15.0 * _LI_CCECP_ZEFF, 3),
        (1, 1.806054, -1.242730, 2),
        (0, 1.330248, 6.752868, 2),
    ]
)


def _li_ccecp_local(r):
    (a, b), (g, d) = _LI_CCECP_LOCAL
    return _LI_CCECP_ZEFF * np.exp(-a * r * r) / r + a * _LI_CCECP_ZEFF * r * np.exp(-b * r * r) + g * np.exp(-d * r * r)


def _li_ccecp_s(r):
    c, e = _LI_CCECP_S
    return c * np.exp(-e * r * r)


def _icosahedral_directions():
    """12 vertices of the icosahedron with a vertex on +z (cos(theta) = +-1/sqrt(5) rings)."""
    c, s = 1.0 / np.sqrt(5.0), 2.0 / np.sqrt(5.0)
    dirs = [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
    dirs += [(s * np.cos(2.0 * np.pi * k / 5.0), s * np.sin(2.0 * np.pi * k / 5.0), c) for k in range(5)]
    dirs += [(s * np.cos((2.0 * k + 1.0) * np.pi / 5.0), s * np.sin((2.0 * k + 1.0) * np.pi / 5.0), -c) for k in range(5)]
    return np.array(dirs)


def _reference_li_ccecp_energy(hamiltonian, r_up, r_dn):
    """Hand-written Li ccECP energy (local + non-local) at one configuration.

    Each electron sees the local part of every nucleus at its minimum-image
    distance and the non-local s projector of its nearest nucleus image (the
    production NN = 1 choice), integrated with the 12-point icosahedral rule
    of Mitas, Shirley and Ceperley, J. Chem. Phys. 95, 3467 (1991),
    https://doi.org/10.1063/1.460849, in its unrotated orientation (RT = 1).
    Only Psi itself is taken from jQMC.
    """
    from jqmc.wavefunction import evaluate_wavefunction

    st = hamiltonian.coulomb_potential_data.structure_data
    R = np.asarray(st.positions, dtype=np.float64)
    cell = np.array([st.vec_a, st.vec_b, st.vec_c], dtype=np.float64)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.float64) @ cell
    wf = hamiltonian.wavefunction_data
    psi0 = complex(evaluate_wavefunction(wf, r_up, r_dn))
    dirs = _icosahedral_directions()
    e_local = 0.0
    e_nonlocal = 0.0
    for spin, coords in (("up", r_up), ("dn", r_dn)):
        for i in range(coords.shape[0]):
            images = R[:, None, :] + shifts[None, :, :]  # (natom, 27, 3)
            dist = np.linalg.norm(coords[i][None, None, :] - images, axis=-1)
            nearest_image = dist.argmin(axis=1)
            r_min = dist[np.arange(R.shape[0]), nearest_image]
            e_local += float(np.sum(_li_ccecp_local(r_min)))
            atom = int(r_min.argmin())
            centre = images[atom, nearest_image[atom]]
            r = float(r_min[atom])
            ratio = 0.0
            for direction in dirs:
                moved = coords.copy()
                moved[i] = centre + r * direction
                up, dn = (moved, r_dn) if spin == "up" else (r_up, moved)
                ratio += (complex(evaluate_wavefunction(wf, up, dn)) / psi0).real / len(dirs)
            e_nonlocal += float(_li_ccecp_s(r)) * ratio
    return e_local + e_nonlocal


def _li_fixed_configuration(hamiltonian):
    R = np.asarray(hamiltonian.coulomb_potential_data.structure_data.positions, dtype=np.float64)
    r_up = R[0:1] + np.array([[0.3, 0.2, -0.1]])
    r_dn = R[1:2] + np.array([[-0.25, 0.4, 0.15]])
    return r_up, r_dn


def _ecp_terms(coulomb_potential_data, atom):
    idx = [k for k, n in enumerate(coulomb_potential_data.nucleus_index) if int(n) == atom]
    return sorted(
        (
            int(coulomb_potential_data.ang_moms[k]),
            float(coulomb_potential_data.exponents[k]),
            float(coulomb_potential_data.coefficients[k]),
            int(coulomb_potential_data.powers[k]),
        )
        for k in idx
    )


class TestQENoUPFECPGuard:
    @pytest.mark.light
    def test_parent_dir_conversion_keeps_the_ccecp_of_the_save_dir_conversion(self, tmp_path):
        """``qe_save_dir`` = parent of ``li.save`` converts with the Li ccECP.

        ``run_qe2jqmc`` (the route of ``jqmc-tool qe`` and ``QE_Workflow``) is
        run with kind='pw' on the committed Li_gamma save, once with the
        ``li.save`` directory and once with its parent ``out``, both with UPF
        auto-discovery. A zero-coefficient placeholder ECP would lower V by
        about 10 Ha at this configuration. Both files must carry, on every atom, the hand-written
        ccECP table (z_core = 2), the ECP part of V at a fixed configuration
        (V minus the bare Coulomb part of the same block) must equal the
        hand-written local + icosahedral non-local reference, and V and E_L
        must agree between the two conversions.
        """
        from jqmc.coulomb_potential import compute_bare_coulomb_potential, compute_coulomb_potential
        from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
        from jqmc.jqmc_qe import run_qe2jqmc

        hams = {}
        for name, qe_dir in (("save", _SAVE_DIR), ("parent", _LI_OUT_DIR)):
            out = tmp_path / f"ham_{name}.h5"
            run_qe2jqmc(
                {"qe_save_dir": str(qe_dir), "kind": "pw", "upf_paths": [], "ecutwfc": None, "alpha_min": None},
                {"j1_parameter": None, "j2_parameter": 1.0},
                str(out),
            )
            hams[name] = Hamiltonian_data.load_from_hdf5(str(out))

        atol, rtol = get_tolerance_min(("coulomb", "det_ratio", "wf_ratio", "ao_eval", "jastrow_eval"), "strict")
        r_up, r_dn = _li_fixed_configuration(hams["save"])
        RT = np.eye(3)
        values = {}
        for name, ham in hams.items():
            cp = ham.coulomb_potential_data
            assert cp.ecp_flag is True, name
            assert tuple(float(z) for z in cp.z_cores) == (2.0,) * int(cp.structure_data.natom), name
            for atom in range(int(cp.structure_data.natom)):
                terms = _ecp_terms(cp, atom)
                assert [(t[0], t[3]) for t in terms] == [(t[0], t[3]) for t in _LI_CCECP_TERMS], (name, atom, terms)
                np.testing.assert_allclose(
                    np.array([t[1:3] for t in terms]),
                    np.array([t[1:3] for t in _LI_CCECP_TERMS]),
                    atol=atol,
                    rtol=rtol,
                    err_msg=f"{name}: atom {atom} ECP table differs from the Li ccECP",
                )
            v_total = float(
                np.real(
                    compute_coulomb_potential(
                        coulomb_potential_data=cp,
                        r_up_carts=r_up,
                        r_dn_carts=r_dn,
                        RT=RT,
                        wavefunction_data=ham.wavefunction_data,
                    )
                )
            )
            v_bare = float(np.real(compute_bare_coulomb_potential(cp, r_up, r_dn)))
            e_l = float(np.real(compute_local_energy(ham, r_up, r_dn, RT)))
            reference = _reference_li_ccecp_energy(ham, r_up, r_dn)
            np.testing.assert_allclose(
                v_total - v_bare,
                reference,
                atol=atol,
                rtol=rtol,
                err_msg=f"{name}: ECP energy differs from the hand-written Li ccECP reference",
            )
            values[name] = (v_total, e_l)
        np.testing.assert_allclose(
            np.array(values["parent"]),
            np.array(values["save"]),
            atol=atol,
            rtol=rtol,
            err_msg="parent-directory conversion differs from the .save conversion (V, E_L)",
        )

    @pytest.mark.light
    def test_parent_dir_twist_average_conversion_keeps_the_ccecp(self, tmp_path):
        """Twist route: ``twist_average`` on the parent of ``li_twist.save``.

        The shared Coulomb block of the parent-directory conversion must
        carry, on every atom, the hand-written Li ccECP table (z_core = 2) and equal the
        ``li_twist.save`` conversion.
        """
        from jqmc._twist_average import load_twist
        from jqmc.jqmc_qe import run_qe2jqmc

        cps = {}
        for name, qe_dir in (("save", _LI_TWIST_SAVE_DIR), ("parent", _LI_TWIST_SAVE_DIR.parent)):
            out = tmp_path / f"twist_{name}.h5"
            run_qe2jqmc(
                {
                    "qe_save_dir": str(qe_dir),
                    "kind": "pw",
                    "upf_paths": [],
                    "ecutwfc": None,
                    "alpha_min": None,
                    "twist_average": True,
                },
                {"j1_parameter": None, "j2_parameter": 1.0},
                str(out),
            )
            cps[name] = load_twist(str(out), 0).coulomb_potential_data

        atol, rtol = get_tolerance_min(("coulomb",), "strict")
        for name, cp in cps.items():
            assert cp.ecp_flag is True, name
            assert tuple(float(z) for z in cp.z_cores) == (2.0,) * int(cp.structure_data.natom), name
            for atom in range(int(cp.structure_data.natom)):
                terms = _ecp_terms(cp, atom)
                assert [(t[0], t[3]) for t in terms] == [(t[0], t[3]) for t in _LI_CCECP_TERMS], (name, atom, terms)
                np.testing.assert_allclose(
                    np.array([t[1:3] for t in terms]),
                    np.array([t[1:3] for t in _LI_CCECP_TERMS]),
                    atol=atol,
                    rtol=rtol,
                    err_msg=f"{name}: atom {atom} ECP table differs from the Li ccECP",
                )
        assert _ecp_terms(cps["parent"], 0) == _ecp_terms(cps["save"], 0)
        assert _ecp_terms(cps["parent"], 1) == _ecp_terms(cps["save"], 1)

    @pytest.mark.light
    def test_cored_structure_without_ecp_source_is_rejected(self, tmp_path):
        """No ECP source and sum(Z) != n_electron raises; nothing is averaged.

        Builder level, Na/Cl in a 6 Bohr CsCl cell (Z = 11, 17): n_electron =
        16 (ccECP valences 9 + 7) would give one averaged z_core = 6 per atom,
        i.e. Z_eff (5, 11) instead of (9, 7), with zero ECP coefficients;
        n_electron = 15 would give a non-neutral all-electron block. Both
        must raise. The neutral all-electron case (n_electron = 28) is still
        built (ecp_flag False). Public routes on the committed Li_gamma save
        (sum Z = 6, n_electron = 2) without a UPF/ECP source must raise as
        well: ``build_hamiltonian_from_projection`` and
        ``convert_qe_to_twist_average_input(upf_paths=[])``, which must not
        write its output file.
        """
        from jqmc.pw_pgto_projection import project_pw_to_pw
        from jqmc.qe_to_jqmc import (
            _build_qe_coulomb_potential_data,
            build_hamiltonian_from_projection,
            convert_qe_to_twist_average_input,
        )
        from jqmc.qe_wrapper import load_qe_reference
        from jqmc.structure import Structure_data

        a = 6.0
        nacl = Structure_data(
            positions=np.array([[0.0, 0.0, 0.0], [a / 2, a / 2, a / 2]]),
            pbc_flag=True,
            vec_a=[a, 0.0, 0.0],
            vec_b=[0.0, a, 0.0],
            vec_c=[0.0, 0.0, a],
            atomic_numbers=(11, 17),
            element_symbols=("Na", "Cl"),
            atomic_labels=("Na", "Cl"),
        )
        for n_electron in (16, 15):
            with pytest.raises(ValueError, match="no analytic ECP was resolved for Cl, Na"):
                _build_qe_coulomb_potential_data(nacl, n_electron, None, None)
        cp_ae, n_ewald = _build_qe_coulomb_potential_data(nacl, 28, None, None)
        assert cp_ae.ecp_flag is False
        assert n_ewald == 28

        ref = load_qe_reference(str(_SAVE_DIR))
        proj = project_pw_to_pw(ref, g_cut=None)
        with pytest.raises(ValueError, match="no analytic ECP was resolved for Li"):
            build_hamiltonian_from_projection(ref, proj, kind="pw", j1_param=None, j2_param=1.0, ewald_tol=1.0e-8)

        out = tmp_path / "twist_no_upf.h5"
        with pytest.raises(ValueError, match="no analytic ECP was resolved for Li"):
            convert_qe_to_twist_average_input(str(_SAVE_DIR), str(out), upf_paths=[], kind="pw")
        assert not out.exists()

    @pytest.mark.light
    def test_upf_map_missing_a_species_is_rejected_by_name(self, tmp_path):
        """A UPF map without the structure's species raises ValueError naming it.

        ``upf_paths = ['Be=<Li UPF>']`` on the Li_gamma save must not fail with a
        bare ``KeyError: "No UPF path registered for element 'Li'"`` deep in the
        ECP lookup: both the single-k entry (``run_qe2jqmc``) and the twist
        writer must reject the map up front, name the missing species and the
        ``--upf`` option, and write nothing.
        """
        from jqmc.jqmc_qe import run_qe2jqmc
        from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input

        out = tmp_path / "ham_wrong_upf.h5"
        with pytest.raises(ValueError, match=r"No UPF supplied for species Li .*--upf"):
            run_qe2jqmc(
                {"qe_save_dir": str(_SAVE_DIR), "kind": "pw", "upf_paths": [f"Be={_UPF}"], "ecutwfc": None, "alpha_min": None},
                {"j1_parameter": None, "j2_parameter": 1.0},
                str(out),
            )
        assert not out.exists()
        with pytest.raises(ValueError, match=r"No UPF supplied for species Li .*--upf"):
            convert_qe_to_twist_average_input(str(_SAVE_DIR), str(out), upf_paths={"Be": str(_UPF)}, kind="pw")
        assert not out.exists()

    @pytest.mark.light
    @pytest.mark.parametrize("twist_average", [False, True])
    def test_tagged_species_uses_element_upf_key_for_pgto_seeds_and_core_charge(self, tmp_path, twist_average):
        """A tagged species ``Li1`` with the UPF key ``Li`` converts on the single-k and twist routes.

        The Li_gamma save is copied with only its XML species/atom names renamed.
        The accepted label-or-element mapping must reach every UPF lookup: the
        pGTO determinant seeds, the deduced core charge and the ECP.
        """
        import shutil

        from jqmc._twist_average import load_twist
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jqmc_qe import run_qe2jqmc
        from jqmc.upf_wrapper import parse_upf

        save_dir = tmp_path / "li_tagged.save"
        shutil.copytree(_SAVE_DIR, save_dir)
        xml = save_dir / "data-file-schema.xml"
        text = xml.read_text()
        assert text.count('name="Li"') > 0
        xml.write_text(text.replace('name="Li"', 'name="Li1"'))
        out = str(tmp_path / "tagged_pgto.h5")
        run_qe2jqmc(
            {"qe_save_dir": str(save_dir), "kind": "pgto", "twist_average": twist_average, "upf_paths": [f"Li={_UPF}"]},
            {"j1_parameter": None, "j2_parameter": 1.0},
            out,
        )
        ham = load_twist(out, 0) if twist_average else Hamiltonian_data.load_from_hdf5(out)
        ham.sanity_check()
        assert set(ham.structure_data.atomic_labels) == {"Li1"}
        z_core = 3.0 - float(parse_upf(str(_UPF)).z_valence)
        np.testing.assert_array_equal(ham.coulomb_potential_data.z_cores, [z_core] * ham.structure_data.natom)


# ---------------------------------------------------------------------------
# Issue #580: a pGTO fit that cannot represent the occupied bands is rejected
# ---------------------------------------------------------------------------
_O_LSDA_SAVE_DIR = _SAVE_DIR.parents[2] / "O-lsda_gamma" / "out" / "o.save"
_LI_TILING_SAVE_DIR = _SAVE_DIR.parents[2] / "Li_tiling_sectors" / "out" / "li_tiling.save"
_PGTO_SAVES = {
    "o-lsda-gamma": _O_LSDA_SAVE_DIR,
    "li-gamma": _SAVE_DIR,
    "li-k-twist": _LI_TWIST_SAVE_DIR,
    "li-tiling": _LI_TILING_SAVE_DIR,
}


def _run_qe_cli(save_dir, out, extra):
    """``jqmc-tool qe <save> --kind pgto -o <out> <extra>`` through the real Typer command (no stand-ins)."""
    from click.testing import CliRunner

    from jqmc.jqmc_tool import typer_click_qe

    return CliRunner().invoke(typer_click_qe, [str(save_dir), "--kind", "pgto", "-o", str(out), *extra])


_PGTO_REJECTED_CASES = {
    "o-lsda-gamma-cond1.5": ("o-lsda-gamma", ("--max-cond-number", "1.5")),
    "o-lsda-gamma-cond1.5-no-orthonormalize": ("o-lsda-gamma", ("--max-cond-number", "1.5", "--no-orthonormalize")),
    "li-gamma-cond0.5": ("li-gamma", ("--max-cond-number", "0.5")),
    "li-gamma-cond0.5-no-orthonormalize": ("li-gamma", ("--max-cond-number", "0.5", "--no-orthonormalize")),
    "li-k-twist-average-cond0.5": ("li-k-twist", ("--twist-average", "--max-cond-number", "0.5")),
    "li-tiling-211-cond0.5": (
        "li-tiling",
        ("--tiling", "2 1 1", "--supertwist", "0 0 0", "--max-cond-number", "0.5"),
    ),
}


class TestQEPGTOFitFailureRejected:
    @pytest.mark.light
    @pytest.mark.parametrize("case", list(_PGTO_REJECTED_CASES))
    def test_pgto_fit_that_drops_the_occupied_subspace_is_rejected(self, tmp_path, case):
        """``--max-cond-number`` that removes the occupied subspace exits non-zero and writes nothing.

        The canonical-orthogonalisation fit drops every overlap mode with
        eigenvalue below ``lambda_max / max_cond_number``. On the committed O
        LSDA Gamma save ``--max-cond-number 1.5`` keeps 3 modes for 4 occupied
        up bands (the up determinant is identically zero: smallest singular
        value of the sampled MO matrix 1.3e-10 against 0.24 for the default);
        on Li Gamma (and every twist of Li_k_twist, and every k block of the
        2x1x1 tiled Li_tiling_sectors conversion) ``0.5`` < 1 <= cond(S)
        drops all modes and every MO coefficient is 0.0. With and without the
        MO orthonormalization, the public outcome must be a failed command
        (``ValueError`` naming the vanishing determinant) and no output file.
        """
        fixture, extra = _PGTO_REJECTED_CASES[case]
        out = tmp_path / "ham.h5"
        result = _run_qe_cli(_PGTO_SAVES[fixture], out, extra)
        assert result.exit_code != 0, result.output
        assert isinstance(result.exception, ValueError), repr(result.exception)
        assert "occupied" in str(result.exception) and "vanishes identically" in str(result.exception), str(result.exception)
        assert not out.exists()

    @pytest.mark.light
    @pytest.mark.parametrize("fixture", ["o-lsda-gamma", "li-gamma"])
    def test_default_pgto_conversion_writes_a_nonsingular_orthonormal_determinant(self, tmp_path, fixture):
        """Valid control: the default ``jqmc-tool qe --kind pgto`` still converts.

        The command exits 0 and writes the Hamiltonian; for every determinant
        spin channel of the saved file the occupied-MO Gram
        ``conj(C) S C^T`` in the analytic periodic AO overlap ``S``
        (``compute_overlap_matrix``, whose phase and normalization equal the
        independent Fourier Gram of MATH-QE-PGTO-MO-ORTHONORMALIZATION-001)
        is the identity within the fp64 projection pair tolerance, so all its
        singular values are 1 and the occupied determinant is non-singular.
        The rejected inputs above fail exactly this property (Gram rank 3 of
        4 for O, the zero matrix for Li).
        """
        from jqmc._setting import ATOL_PW_PROJECTION_FP64, RTOL_PW_PROJECTION_FP64
        from jqmc.atomic_orbital import compute_overlap_matrix
        from jqmc.hamiltonians import Hamiltonian_data

        out = tmp_path / "ham.h5"
        result = _run_qe_cli(_PGTO_SAVES[fixture], out, ())
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert out.is_file()
        geminal = Hamiltonian_data.load_from_hdf5(str(out)).wavefunction_data.geminal_data
        for channel, mos in (("up", geminal.orb_data_up_spin), ("dn", geminal.orb_data_dn_spin)):
            c = np.asarray(mos.mo_coefficients)
            s = np.asarray(compute_overlap_matrix(mos.aos_data))
            gram = c.conj() @ s @ c.T
            assert gram.shape[0] >= 1, channel
            np.testing.assert_allclose(
                gram,
                np.eye(gram.shape[0]),
                atol=ATOL_PW_PROJECTION_FP64,
                rtol=RTOL_PW_PROJECTION_FP64,
                err_msg=f"{fixture} {channel}: occupied MO Gram is not the identity",
            )
