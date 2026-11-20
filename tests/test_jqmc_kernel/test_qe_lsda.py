"""Unit tests for the LSDA (nspin=2) QE -> jQMC conversion path.

Fixture: ``tests/qe_example_files/O-lsda_gamma`` -- a single ccECP oxygen
atom (Z_valence = 6) in a 10 Bohr cubic box, LDA-PZ, ``nspin = 2`` with
``tot_magnetization = 2`` and fixed occupations (triplet ground state:
``n_up = 4``, ``n_dn = 2``), ``nbnd = 6`` so the per-spin band truncation
(``n_keep = 4 < nbnd``) is exercised. QE writes the two spin channels as
``wfcup1.hdf5`` / ``wfcdw1.hdf5`` and concatenates the per-spin eigenvalue /
occupation arrays inside the single Gamma ``ks_energies`` block.
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

from jqmc._setting import ATOL_NONVACUITY_DISTINCTION, RTOL_NONVACUITY_DISTINCTION
from pathlib import Path

import numpy as np
import pytest

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import (
    ATOL_PYTHON_FP64_LOOSE,
    RTOL_PW_PROJECTION_FP64,
    RTOL_QE_PW_RECONSTRUCTION_LOOSE,
    RTOL_QE_PW_RECONSTRUCTION_TIGHT,
)
from tests._test_data_cache import qe_example_files_root
from tests.test_jqmc_kernel.test_qe_to_jqmc import _pw_band_values_in_zones


def _pw_reconstruction_rtol() -> float:
    """Preserve the fp64 projection gate; admit the active fp32 AO boundary."""
    if is_mixed_precision_enabled():
        return get_tolerance_min(("ao_eval", "mo_eval"), "strict")[1]
    return RTOL_QE_PW_RECONSTRUCTION_TIGHT


pytestmark = [pytest.mark.with_chdir]
_SAVE_DIR = qe_example_files_root() / "O-lsda_gamma" / "out" / "o.save"
_XML = _SAVE_DIR / "data-file-schema.xml"
_WFC_UP = _SAVE_DIR / "wfcup1.hdf5"
_WFC_DW = _SAVE_DIR / "wfcdw1.hdf5"
_UPF = _SAVE_DIR / "O.ccECP.upf"

_available = _XML.is_file() and _WFC_UP.is_file() and _WFC_DW.is_file() and _UPF.is_file()
requires_qe_lsda = pytest.mark.skipif(not _available, reason="QE O-lsda_gamma sample not present")

_N_OCC_UP = 4
_N_OCC_DN = 2
_NBND = 6


# ---------------------------------------------------------------------------
# qe_wrapper: per-spin loading
# ---------------------------------------------------------------------------
class TestQEWrapperLSDA:
    @pytest.mark.light
    @requires_qe_lsda
    def test_lsda_detection(self):
        """Lsda detection."""
        from jqmc.qe_wrapper import load_qe_lsda, qe_save_is_lsda

        assert qe_save_is_lsda(str(_SAVE_DIR)) is True
        assert load_qe_lsda(str(_XML)) is True

    @pytest.mark.light
    @requires_qe_lsda
    def test_per_spin_bands_split(self):
        """Per spin bands split."""
        from jqmc.qe_wrapper import load_qe_bands

        up = load_qe_bands(str(_XML), spin="up")
        dn = load_qe_bands(str(_XML), spin="dw")
        assert up.eigenvalues.size == _NBND
        assert dn.eigenvalues.size == _NBND
        # Fixed occupations: [1]*n_occ + [0]*rest per channel.
        np.testing.assert_array_equal(up.occupations, [1.0] * _N_OCC_UP + [0.0] * (_NBND - _N_OCC_UP))
        np.testing.assert_array_equal(dn.occupations, [1.0] * _N_OCC_DN + [0.0] * (_NBND - _N_OCC_DN))
        # n_electrons stays the TOTAL count in both channels.
        assert up.n_electrons == pytest.approx(6.0)
        assert dn.n_electrons == pytest.approx(6.0)
        # Aufbau within each channel.
        assert np.all(np.diff(up.eigenvalues) > -1.0e-10)
        assert np.all(np.diff(dn.eigenvalues) > -1.0e-10)

    @pytest.mark.light
    @requires_qe_lsda
    def test_spin_selection_errors(self):
        """Spin selection errors."""
        from jqmc.qe_wrapper import load_qe_bands, load_qe_reference

        # LSDA save without a spin choice must not load silently.
        with pytest.raises(ValueError, match="LSDA"):
            load_qe_reference(str(_SAVE_DIR))
        with pytest.raises(ValueError, match="spin='up' or spin='dw'"):
            load_qe_bands(str(_XML), spin=None)
        with pytest.raises(ValueError, match="spin='up' or spin='dw'"):
            load_qe_bands(str(_XML), spin="down")

    @pytest.mark.light
    @requires_qe_lsda
    def test_restricted_save_rejects_spin(self):
        """Restricted save rejects spin."""
        li_save = qe_example_files_root() / "Li_gamma" / "out" / "li.save"
        if not (li_save / "data-file-schema.xml").is_file():
            pytest.skip("Li_gamma sample not present")
        from jqmc.qe_wrapper import load_qe_reference, qe_save_is_lsda

        assert qe_save_is_lsda(str(li_save)) is False
        with pytest.raises(ValueError, match="not an LSDA"):
            load_qe_reference(str(li_save), spin="up")

    @pytest.mark.light
    @requires_qe_lsda
    def test_per_spin_reference_and_truncation(self):
        """Per spin reference and truncation."""
        from jqmc.qe_wrapper import load_qe_reference, truncate_qe_reference_bands

        up = load_qe_reference(str(_SAVE_DIR), spin="up")
        dn = load_qe_reference(str(_SAVE_DIR), spin="dw")
        assert up.wfc.n_bands == _NBND
        assert dn.wfc.n_bands == _NBND
        # The two channels are genuinely different wavefunctions.
        assert not np.allclose(
            np.abs(up.wfc.coeffs[:_N_OCC_DN]),
            np.abs(dn.wfc.coeffs[:_N_OCC_DN]),
            rtol=RTOL_NONVACUITY_DISTINCTION,
            atol=ATOL_NONVACUITY_DISTINCTION,
        )

        t = truncate_qe_reference_bands(dn, _N_OCC_UP)
        assert t.wfc.n_bands == _N_OCC_UP
        assert t.bands.eigenvalues.size == _N_OCC_UP
        np.testing.assert_array_equal(t.bands.occupations, [1.0] * _N_OCC_DN + [0.0] * (_N_OCC_UP - _N_OCC_DN))
        np.testing.assert_array_equal(t.wfc.millers, dn.wfc.millers)
        np.testing.assert_array_equal(t.wfc.coeffs, dn.wfc.coeffs[:_N_OCC_UP])

        with pytest.raises(ValueError, match="out of range"):
            truncate_qe_reference_bands(dn, _NBND + 1)


# ---------------------------------------------------------------------------
# qe_to_jqmc: unrestricted wavefunction assembly
# ---------------------------------------------------------------------------
def _load_truncated_pair():
    from jqmc.qe_wrapper import load_qe_reference, truncate_qe_reference_bands

    up = truncate_qe_reference_bands(load_qe_reference(str(_SAVE_DIR), spin="up"), _N_OCC_UP)
    dn = truncate_qe_reference_bands(load_qe_reference(str(_SAVE_DIR), spin="dw"), _N_OCC_UP)
    return up, dn


class TestQEToJQMCUnrestricted:
    @pytest.mark.light
    @requires_qe_lsda
    def test_build_wavefunction_pw_unrestricted(self):
        """kind='pw' UKS: SD_data with ne_up=4, ne_dn=2, independent dn MOs."""
        from jqmc.determinant import SD_data
        from jqmc.pw_pgto_projection import project_pw_to_pw
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection

        up, dn = _load_truncated_pair()
        proj_up = project_pw_to_pw(up, g_cut=None, occ_only=False)
        proj_dn = project_pw_to_pw(dn, g_cut=None, occ_only=False)
        _, _, geminal = build_wavefunction_from_projection(
            up, proj_up, kind="pw", qe_reference_dn=dn, projection_result_dn=proj_dn
        )
        assert isinstance(geminal, SD_data)
        assert geminal.num_electron_up == _N_OCC_UP
        assert geminal.num_electron_dn == _N_OCC_DN
        assert int(geminal.orb_data_up_spin.num_mo) == _N_OCC_UP
        assert int(geminal.orb_data_dn_spin.num_mo) == _N_OCC_DN
        # Unrestricted: the dn MO block is the beta channel, not conj(alpha).
        mo_up = np.asarray(geminal.orb_data_up_spin.mo_coefficients)
        mo_dn = np.asarray(geminal.orb_data_dn_spin.mo_coefficients)
        assert not np.allclose(
            np.abs(mo_up[:_N_OCC_DN]), np.abs(mo_dn), rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION
        )

    @pytest.mark.light
    @requires_qe_lsda
    def test_pw_mos_reproduce_each_spin_channel(self):
        """Lossless PW conversion: each spin's MOs reproduce that spin's QE bands."""
        from jqmc.molecular_orbital import compute_MOs
        from jqmc.pw_pgto_projection import grid_points, project_pw_to_pw
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection

        up, dn = _load_truncated_pair()
        proj_up = project_pw_to_pw(up, g_cut=None, occ_only=False)
        proj_dn = project_pw_to_pw(dn, g_cut=None, occ_only=False)
        _, _, geminal = build_wavefunction_from_projection(
            up, proj_up, kind="pw", qe_reference_dn=dn, projection_result_dn=proj_dn
        )

        grid = grid_points(up.cell, up.fft_grid)
        for mos, ref in ((geminal.orb_data_up_spin, up), (geminal.orb_data_dn_spin, dn)):
            mo_vals = np.asarray(compute_MOs(mos, grid))  # (n_mo, n_pts)
            # Unit-cell-normalized reference: direct band sum / sqrt(Omega),
            # evaluated in the production zones (see _direct_pw_bands).
            occupied = np.flatnonzero(np.asarray(ref.bands.occupations) > 0.5)
            expected = np.real(_direct_pw_bands(ref, grid, occupied[: int(mos.num_mo)]))
            rel = np.linalg.norm(mo_vals - expected) / max(np.linalg.norm(expected), 1.0e-30)
            assert rel < _pw_reconstruction_rtol(), f"LSDA PW MO reconstruction rel error too large: {rel:.3e}"

    @pytest.mark.light
    @requires_qe_lsda
    def test_source_band_count_mismatch_is_allowed_when_each_spin_is_filled(self):
        """Source band count mismatch is allowed when each spin is filled."""
        from jqmc.pw_pgto_projection import project_pw_to_pw
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection
        from jqmc.qe_wrapper import load_qe_reference, truncate_qe_reference_bands

        up = truncate_qe_reference_bands(load_qe_reference(str(_SAVE_DIR), spin="up"), _N_OCC_UP)
        dn = truncate_qe_reference_bands(load_qe_reference(str(_SAVE_DIR), spin="dw"), _N_OCC_DN)
        proj_up = project_pw_to_pw(up, g_cut=None, occ_only=False)
        proj_dn = project_pw_to_pw(dn, g_cut=None, occ_only=False)
        _, _, geminal = build_wavefunction_from_projection(
            up,
            proj_up,
            kind="pw",
            qe_reference_dn=dn,
            projection_result_dn=proj_dn,
        )
        assert geminal.orb_data_up_spin.num_mo == _N_OCC_UP
        assert geminal.orb_data_dn_spin.num_mo == _N_OCC_DN

    @pytest.mark.light
    @requires_qe_lsda
    def test_dn_kwargs_must_come_together(self):
        """Dn kwargs must come together."""
        from jqmc.pw_pgto_projection import project_pw_to_pw
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection

        up, _ = _load_truncated_pair()
        proj_up = project_pw_to_pw(up, g_cut=None, occ_only=False)
        with pytest.raises(ValueError, match="supplied together"):
            build_wavefunction_from_projection(up, proj_up, kind="pw", qe_reference_dn=up)

    @pytest.mark.light
    @requires_qe_lsda
    def test_fractional_occupations_rejected(self):
        """QMC fixes electron numbers: non-binary band occupations must fail.

        Integer *sums* are not enough -- {0.5, 0.5, 1.0} sums to 2 but has no
        Slater-determinant representation, so each band must be 0 or 1.
        """
        import dataclasses

        from jqmc.pw_pgto_projection import project_pw_to_pw
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection

        up, dn = _load_truncated_pair()
        proj_up = project_pw_to_pw(up, g_cut=None, occ_only=False)
        proj_dn = project_pw_to_pw(dn, g_cut=None, occ_only=False)
        occ_frac = np.asarray(proj_dn.occupations, dtype=np.float64).copy()
        occ_frac[0] = 0.5
        occ_frac[1] = 0.5
        occ_frac[2] = 1.0
        proj_dn_frac = dataclasses.replace(proj_dn, occupations=occ_frac)
        with pytest.raises(ValueError, match="neither 0 nor 1"):
            build_wavefunction_from_projection(up, proj_up, kind="pw", qe_reference_dn=dn, projection_result_dn=proj_dn_frac)

    @pytest.mark.light
    @requires_qe_lsda
    def test_restricted_fractional_occupations_rejected(self):
        """The restricted (nspin=1) path enforces the same integer-filling contract."""
        import dataclasses

        from jqmc.pw_pgto_projection import project_pw_to_pw
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection
        from jqmc.qe_wrapper import load_qe_reference

        li_save = qe_example_files_root() / "Li_gamma" / "out" / "li.save"
        if not (li_save / "data-file-schema.xml").is_file():
            pytest.skip("Li_gamma sample not present")
        ref = load_qe_reference(str(li_save))
        proj = project_pw_to_pw(ref, g_cut=None, occ_only=False)
        occ_frac = np.asarray(proj.occupations, dtype=np.float64).copy()
        occ_frac[0] = 0.7
        occ_frac[1] = 0.3
        proj_frac = dataclasses.replace(proj, occupations=occ_frac)
        with pytest.raises(ValueError, match="neither 0 nor 1"):
            build_wavefunction_from_projection(ref, proj_frac, kind="pw")

    @pytest.mark.light
    def test_check_binary_occupations_tolerance(self):
        """Smeared-but-gapped saves (occ deviations << tol) pass; metals fail."""
        from jqmc.qe_to_jqmc import check_binary_occupations

        # Methfessel-Paxton on a gapped state: ~1e-26 deviations, tiny negatives.
        check_binary_occupations(np.array([1.0, 1.0 - 1e-10, -1e-26, 0.0]))
        with pytest.raises(ValueError, match="neither 0 nor 1"):
            check_binary_occupations(np.array([1.0, 0.5, 0.5, 0.0]), channel="up")

    @pytest.mark.light
    @requires_qe_lsda
    def test_multi_twist_converter_lsda_pgto(self, tmp_path):
        """LSDA multi-twist (kind='pgto'): combined layout carries the two spin channels.

        The fixture is a single-Gamma-twist save, which exercises the full
        writer / loader round-trip: unrestricted template (time_reversal =
        False, ne_up=4 / ne_dn=2 shared via /shared) + per-twist independent
        mo_up / mo_dn blocks.
        """
        from jqmc._twist_average import load_twist, load_twist_average_input
        from jqmc.determinant import SD_data
        from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input

        out = tmp_path / "twist_lsda.h5"
        convert_qe_to_twist_average_input(
            str(_SAVE_DIR),
            str(out),
            upf_paths={"O": str(_UPF)},
            kind="pgto",
        )
        meta = load_twist_average_input(str(out))
        assert meta.n_twist == 1
        ham = load_twist(str(out), 0)
        g = ham.wavefunction_data.geminal_data
        assert isinstance(g, SD_data)
        assert g.num_electron_up == _N_OCC_UP
        assert g.num_electron_dn == _N_OCC_DN
        assert bool(g.time_reversal) is False
        # Projection may use all nbnd internally, but canonical SD storage is
        # occupied-only for every twist.
        assert int(g.orb_data_up_spin.num_mo) == _N_OCC_UP
        assert int(g.orb_data_dn_spin.num_mo) == _N_OCC_DN
        mo_up = np.asarray(g.orb_data_up_spin.mo_coefficients)
        mo_dn = np.asarray(g.orb_data_dn_spin.mo_coefficients)
        assert not np.allclose(
            np.abs(mo_up[:_N_OCC_DN]), np.abs(mo_dn), rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION
        )


# ---------------------------------------------------------------------------
# Non-Gamma LSDA: complex Bloch path (single-k + multi-twist).
# Fixture: same O-atom triplet box at k = (0.25, 0.25, 0.25).
# ---------------------------------------------------------------------------
_SAVE_DIR_K025 = qe_example_files_root() / "O-lsda_k025" / "out" / "o_k025.save"
_UPF_K025 = _SAVE_DIR_K025 / "O.ccECP.upf"
_available_k025 = (
    (_SAVE_DIR_K025 / "data-file-schema.xml").is_file()
    and (_SAVE_DIR_K025 / "wfcup1.hdf5").is_file()
    and (_SAVE_DIR_K025 / "wfcdw1.hdf5").is_file()
)
requires_qe_lsda_k025 = pytest.mark.skipif(not _available_k025, reason="QE O-lsda_k025 sample not present")


def _direct_pw_bands(ref, r_carts, band_idx=None):
    """Direct unit-cell-normalized ``psi(r) = Omega^{-1/2} sum_G c_G e^{i(G+k).r}``, ``(n_band, n_pts)``.

    The converter emits unit-cell-normalized MOs, so the absolute
    reference carries the ``Omega^{-1/2}`` of the QE convention. The band
    sum is evaluated in the production precision zones (phase argument
    reconstructed in fp64 and cast to ``ao_eval``, plane-wave values in
    ``ao_eval``, coefficient contraction and normalization in ``mo_eval``; see
    ``_pw_band_values_in_zones``), so mixed is judged within its own precision.
    """
    G = ref.wfc.millers.astype(np.float64) @ ref.cell.b
    coeffs = ref.wfc.coeffs if band_idx is None else ref.wfc.coeffs[band_idx]
    return _pw_band_values_in_zones(coeffs, G + ref.wfc.k_point, r_carts) / np.sqrt(ref.cell.volume)


class TestQELSDANonGamma:
    @pytest.mark.light
    @requires_qe_lsda_k025
    def test_single_k_uks_pw_orbitals_reproduce_both_channels(self):
        """Lossless 'pw' UKS at k=0.25: up = psi_alpha^{+k}, dn = conj(psi_beta^{+k}).

        Pins the unrestricted down channel: the complex conjugate of the
        independent beta orbitals (their -k partners on the negated-k leaf),
        not the unconjugated beta orbitals and not conj(psi_alpha).
        """
        from jqmc.molecular_orbital import compute_MOs
        from jqmc.pw_pgto_projection import project_pw_to_pw
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection
        from jqmc.qe_wrapper import load_qe_reference, truncate_qe_reference_bands

        up = truncate_qe_reference_bands(load_qe_reference(str(_SAVE_DIR_K025), spin="up"), _N_OCC_UP)
        dn = truncate_qe_reference_bands(load_qe_reference(str(_SAVE_DIR_K025), spin="dw"), _N_OCC_UP)
        proj_up = project_pw_to_pw(up, g_cut=None, occ_only=False)
        proj_dn = project_pw_to_pw(dn, g_cut=None, occ_only=False)
        _, _, gem = build_wavefunction_from_projection(up, proj_up, kind="pw", qe_reference_dn=dn, projection_result_dn=proj_dn)
        rng = np.random.default_rng(11)
        r = rng.uniform(0.0, 9.0, size=(32, 3))
        up_vals = np.asarray(compute_MOs(gem.orb_data_up_spin, r))
        dn_vals = np.asarray(compute_MOs(gem.orb_data_dn_spin, r))
        psi_a = _direct_pw_bands(up, r)
        psi_b = _direct_pw_bands(dn, r)
        psi_a = psi_a[np.flatnonzero(np.asarray(up.bands.occupations) > 0.5)]
        psi_b = psi_b[np.flatnonzero(np.asarray(dn.bands.occupations) > 0.5)]
        psi_b = np.conj(psi_b)
        rel_up = np.linalg.norm(up_vals - psi_a) / np.linalg.norm(psi_a)
        rel_dn = np.linalg.norm(dn_vals - psi_b) / np.linalg.norm(psi_b)
        projection_rtol = (
            get_tolerance_min(("ao_eval", "mo_eval"), "strict")[1] if is_mixed_precision_enabled() else RTOL_PW_PROJECTION_FP64
        )
        assert rel_up < projection_rtol, f"up channel does not reproduce psi_alpha^+k: rel={rel_up:.3e}"
        assert rel_dn < projection_rtol, f"dn channel does not reproduce conj(psi_beta^+k): rel={rel_dn:.3e}"
        rel_plain = np.linalg.norm(dn_vals - np.conj(psi_b)) / np.linalg.norm(psi_b)
        assert rel_plain > projection_rtol, "dn channel is the unconjugated beta orbital"

    @pytest.mark.light
    @requires_qe_lsda_k025
    @pytest.mark.parametrize("kind", ["bspline", "pgto+bspline"])
    def test_multi_twist_lsda_bspline_kinds_round_trip(self, kind, tmp_path):
        """LSDA multi-twist bspline kinds: per-spin child via composite_child_dn.

        Writes the combined file (dn spline payload under composite_child_dn),
        reloads via load_twist and checks BOTH spin channels reproduce their
        QE bands to spline accuracy (the spline child is the bands, so a
        spin-shared child could not pass this). The dn oracle is the complex
        conjugate of the stored beta band (opposite-twist convention).
        """
        from jqmc._twist_average import load_twist
        from jqmc.determinant import SD_data
        from jqmc.molecular_orbital import compute_MOs
        from jqmc.pw_pgto_projection import taper_qe_reference
        from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input
        from jqmc.qe_wrapper import load_qe_reference

        out = tmp_path / f"twist_{kind.replace('+', '_')}.h5"
        convert_qe_to_twist_average_input(
            str(_SAVE_DIR_K025),
            str(out),
            upf_paths={"O": str(_UPF_K025)},
            kind=kind,
            mesh_factor=2.0,
        )
        ham = load_twist(str(out), 0)
        g = ham.wavefunction_data.geminal_data
        assert isinstance(g, SD_data)
        assert g.num_electron_up == _N_OCC_UP
        assert g.num_electron_dn == _N_OCC_DN
        assert bool(g.time_reversal) is False

        # The direct oracle follows the converter's input contract: taper_qe_reference tapers the PW
        # sphere edge only for vacuum cells (here the identity: the 10 bohr box's corner is 8.66 bohr
        # from the O nucleus, below PW_TAPER_VACUUM_RADIUS_BOHR).
        up, _ = taper_qe_reference(load_qe_reference(str(_SAVE_DIR_K025), spin="up"))
        dn, _ = taper_qe_reference(load_qe_reference(str(_SAVE_DIR_K025), spin="dw"))
        rng = np.random.default_rng(11)
        r = rng.uniform(0.0, 9.0, size=(32, 3))
        up_vals = np.asarray(compute_MOs(g.orb_data_up_spin, r))
        dn_vals = np.asarray(compute_MOs(g.orb_data_dn_spin, r))
        psi_a = _direct_pw_bands(up, r)
        psi_b = _direct_pw_bands(dn, r)
        psi_a = psi_a[np.flatnonzero(np.asarray(up.bands.occupations) > 0.5)]
        psi_b = psi_b[np.flatnonzero(np.asarray(dn.bands.occupations) > 0.5)]
        psi_b = np.conj(psi_b)
        rel_up = np.linalg.norm(up_vals - psi_a) / np.linalg.norm(psi_a)
        rel_dn = np.linalg.norm(dn_vals - psi_b) / np.linalg.norm(psi_b)
        assert np.linalg.norm(dn_vals - np.conj(psi_b)) / np.linalg.norm(psi_b) > RTOL_QE_PW_RECONSTRUCTION_LOOSE
        assert rel_up < RTOL_QE_PW_RECONSTRUCTION_LOOSE, f"{kind}: up channel rel={rel_up:.3e}"
        assert rel_dn < RTOL_QE_PW_RECONSTRUCTION_LOOSE, f"{kind}: dn channel rel={rel_dn:.3e}"
        # The two spin channels are genuinely different bands.
        assert not np.allclose(
            np.abs(up_vals[:_N_OCC_DN]),
            np.abs(dn_vals[:_N_OCC_DN]),
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_NONVACUITY_DISTINCTION,
        )


# ---------------------------------------------------------------------------
# run_qe2jqmc: end-to-end TOML-path conversion per kind
# ---------------------------------------------------------------------------
class TestRunQE2JQMCLSDA:
    @pytest.mark.light
    @requires_qe_lsda
    def test_fixed_10ry_pgto_pw_runs_exponent_optimisation(self, tmp_path, monkeypatch):
        """A numeric cutoff optimises shared LSDA exponents before projection."""
        from jqmc import pgto_exponent_optimizer
        from jqmc.jqmc_qe import run_qe2jqmc

        original = pgto_exponent_optimizer.optimize_pgto_exponents
        calls = []

        def recording_optimizer(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append((kwargs, result))
            return result

        monkeypatch.setattr(pgto_exponent_optimizer, "optimize_pgto_exponents", recording_optimizer)
        out = tmp_path / "ham_pgto_pw_fixed_10ry.h5"
        run_qe2jqmc(
            {
                "qe_save_dir": str(_SAVE_DIR),
                "kind": "pgto+pw",
                "upf_paths": [],
                "ecutwfc": 10.0,
                "alpha_min": None,
                # The unbounded optimiser (exponent-tail bound off);
                # the bounded default path on this fixture is covered by
                # test_pgto_exponent_tail_bound.py.
                "exp_tail_eps": None,
            },
            {"j1_parameter": None, "j2_parameter": 1.0},
            str(out),
        )

        assert out.is_file()
        assert len(calls) == 1
        assert calls[0][0]["exp_tail_eps"] is None
        kwargs, result = calls[0]
        assert kwargs["fixed_mixed_overlap"] is False
        assert kwargs["deployed_fixed_mixed_overlap"] is True
        assert kwargs["g_cut"] == pytest.approx(np.sqrt(10.0))
        assert result.accepted
        assert result.deployed_optimized_spilling < result.deployed_initial_spilling

    @pytest.mark.light
    @requires_qe_lsda
    @pytest.mark.parametrize("kind", ["pw", "bspline", "pgto", "pgto+pw", "pgto+bspline"])
    def test_lsda_conversion_end_to_end(self, kind, tmp_path):
        """Lsda conversion end to end."""
        from jqmc.determinant import SD_data
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jqmc_qe import run_qe2jqmc

        out = tmp_path / f"ham_{kind.replace('+', '_')}.h5"
        qe_params = {
            "qe_save_dir": str(_SAVE_DIR),
            "kind": kind,
            "upf_paths": [],
            "ecutwfc": None,
            "alpha_min": None,
        }
        jastrow_params = {"j1_parameter": None, "j2_parameter": 1.0}
        run_qe2jqmc(qe_params, jastrow_params, str(out))

        h = Hamiltonian_data.load_from_hdf5(str(out))
        g = h.wavefunction_data.geminal_data
        assert isinstance(g, SD_data)
        assert g.num_electron_up == _N_OCC_UP
        assert g.num_electron_dn == _N_OCC_DN
        assert int(g.orb_data_up_spin.num_mo) == _N_OCC_UP
        assert int(g.orb_data_dn_spin.num_mo) == _N_OCC_DN
        # ECP oxygen: z_core = 2 (1s frozen).
        assert h.coulomb_potential_data.ecp_flag is True
        np.testing.assert_array_equal(np.asarray(h.coulomb_potential_data.z_cores), [2.0])
