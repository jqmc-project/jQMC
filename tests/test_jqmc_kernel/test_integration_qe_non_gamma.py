"""QE single non-Gamma twist conversion tests.

The QE pGTO projection at a single non-Gamma k-point must

* lift the previous ``k_in_crystal == Gamma`` stub in
  :py:func:`jqmc.pw_pgto_projection.build_gto_pw_matrix` by shifting the
  design-matrix evaluation from ``G`` to ``q = G + k``;
* read the QE-stored ``xk`` (1/bohr Cartesian) and convert it to fractional
  reciprocal-lattice coordinates for :py:attr:`Structure_data.k_point`;
* dispatch the pGTO AO leaf on ``Structure_data.k_point`` -- at a non-Gamma
  twist the leaf must be ``AOs_cart_pgto_complex_data`` (Bloch image-phase
  factors live on the leaf) and the projected MO coefficients must carry
  ``complex128`` end-to-end into ``MOs_data``;
* enforce Convention B: ``MOs_data_dn = conj(MOs_data_up)`` while canonical
  SD stores occupied complex MO rows directly and has no ``lambda_matrix``.

Fixture: ``Li_k025/out/li_k025.save`` (bcc Li at fractional k =
(1/4, 1/4, 1/4), weight 2.0). The converged QE SCF total energy is
~ -0.54 Ha (see ``data-file-schema.xml`` ``<etot>``). Smoke MCMC run
confirms the chain flows through the complex Bloch path without firing
``ComplexWarning`` (autouse guard in ``conftest.py``).
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
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.atomic_orbital import (
    AOs_bspline_complex_data,
    AOs_cart_pgto_complex_data,
    AOs_cart_pgto_data,
    AOs_composite_data,
    AOs_plane_wave_complex_data,
    is_complex_aos,
)
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import (
    QE_NON_GAMMA_FIXTURE_K_NORM_UPPER,
    ATOL_PYTHON_FP64_LOOSE,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_PW_PROJECTION_FP64,
    RTOL_QE_PW_RECONSTRUCTION_LOOSE,
)
from tests._test_data_cache import qe_example_files_root
from tests.test_jqmc_kernel.test_qe_to_jqmc import _pw_band_values_in_zones

_FIXTURE_DIR = qe_example_files_root() / "Li_k025" / "out" / "li_k025.save"
_UPF_PATH = _FIXTURE_DIR / "Li.ccECP.upf"
_available = (
    (_FIXTURE_DIR / "data-file-schema.xml").is_file() and (_FIXTURE_DIR / "wfc1.hdf5").is_file() and _UPF_PATH.is_file()
)

requires_k025 = pytest.mark.skipif(not _available, reason="Li_k025 QE fixture not present")


@pytest.fixture(scope="module")
def qe_reference():
    from jqmc.qe_wrapper import load_qe_reference

    return load_qe_reference(str(_FIXTURE_DIR))


@pytest.fixture(scope="module")
def pgto_shells(qe_reference):
    from jqmc.upf_wrapper import UPFExponentInit

    g_target = float(np.sqrt(2.0 * qe_reference.ecutwfc))
    strategy = UPFExponentInit({"Li": str(_UPF_PATH)}, g_target=g_target)
    return [strategy.shells_for_element(lbl) for lbl in qe_reference.atom_labels]


# ---------- xk handling + Structure_data.k_point conversion ----------


class TestQENonGamma:
    @pytest.mark.heavy
    @requires_k025
    def test_wfc_k_point_is_nonzero_cartesian(self, qe_reference):
        """``QEWavefunction.k_point`` is the QE-stored ``xk`` (1/bohr Cartesian).
        Pins that the fixture is at a non-Gamma twist and that the loader
        preserves the magnitude. For Li_k025 the IBZ point is fractional
        (1/4, 1/4, 1/4) and the bcc Li conventional cell has alat = 6.6329
        Bohr, so |xk| ~ (sqrt(3)/4) * 2*pi/alat ~ 0.41 1/bohr."""
        xk = np.asarray(qe_reference.wfc.k_point)
        assert xk.shape == (3,)
        assert not np.allclose(xk, 0.0, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION)
        assert 0.30 < float(np.linalg.norm(xk)) < QE_NON_GAMMA_FIXTURE_K_NORM_UPPER

    @pytest.mark.heavy
    @requires_k025
    @pytest.mark.parametrize("cell_frame", ["as_stored", "rigidly_rotated"])
    def test_aos_data_k_point_fractional_round_trip(self, qe_reference, cell_frame):
        """``_build_structure_data`` returns the fractional reciprocal-lattice
        k_point alongside the cell. Conversion ``xk_cart = k_frac @ cell.b``
        must round-trip back to the stored ``xk``. For Li_k025 the IBZ point
        is fractional (1/4, 1/4, 1/4) in the simple-cubic reciprocal basis.

        The ``rigidly_rotated`` case rotates the whole crystal (cell rows,
        atoms and the Cartesian ``xk``) by a generic rotation. That leaves
        the physics, and therefore the fractional twist given in the QE input
        (``K_POINTS {crystal}`` 0.25 0.25 0.25), unchanged, but makes
        ``cell.a`` / ``cell.b`` non-symmetric matrices -- the stored cubic
        cell is diagonal, so it cannot tell ``inv(cell.b)`` from
        ``inv(cell.b.T)``. The Bloch phase ``xk . a_i = 2 pi k_frac_i``
        gives a second, convention-free reference.

        ``k_point`` does not live on
        ``Structure_data``; the loader returns it as a side value that the
        AO-leaf builders attach to their complex Bloch leaves.
        """
        from dataclasses import replace

        from scipy.spatial.transform import Rotation

        from jqmc.qe_to_jqmc import _build_structure_data

        reference = qe_reference
        if cell_frame == "rigidly_rotated":
            rot = Rotation.from_rotvec(0.7 * np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)).as_matrix()
            reference = replace(
                qe_reference,
                cell=replace(qe_reference.cell, a=np.asarray(qe_reference.cell.a) @ rot.T),
                atom_positions=np.asarray(qe_reference.atom_positions) @ rot.T,
                wfc=replace(qe_reference.wfc, k_point=np.asarray(qe_reference.wfc.k_point) @ rot.T),
            )
            assert not np.allclose(
                reference.cell.b, reference.cell.b.T, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION
            )

        sd, k_point_frac = _build_structure_data(reference)
        assert sd.pbc_flag is True
        assert k_point_frac is not None
        k_frac = np.asarray(k_point_frac, dtype=np.float64)
        xk_back = k_frac @ reference.cell.b
        np.testing.assert_allclose(
            xk_back, np.asarray(reference.wfc.k_point), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE
        )
        # Bloch phase per direct lattice vector: xk . a_i = 2 pi k_frac_i.
        np.testing.assert_allclose(
            np.asarray(reference.cell.a) @ np.asarray(reference.wfc.k_point) / (2.0 * np.pi),
            k_frac,
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_LOOSE,
        )
        # Each fractional component is 1/4 (the QE IBZ point is
        # (1/4, 1/4, 1/4) in the cubic reciprocal basis).
        np.testing.assert_allclose(np.abs(k_frac), 0.25, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)

    # ---------- build_gto_pw_matrix q = G + k shift ----------
    @pytest.mark.heavy
    @requires_k025
    def test_build_gto_pw_matrix_lifts_gamma_stub(self, qe_reference, pgto_shells):
        """At ``k_cart != Gamma`` the design matrix is evaluated at
        ``q = G + k``. Pins that the call does not raise
        ``NotImplementedError`` and
        the returned matrix is genuinely complex (the structure phase
        ``exp(-i q . R)`` carries a non-trivial Im).
        """
        from jqmc.pw_pgto_projection import build_gto_pw_matrix, build_pgto_primitives

        prims = build_pgto_primitives(pgto_shells, qe_reference.atom_positions)
        G_full = build_gto_pw_matrix(qe_reference, prims)
        assert G_full.dtype == np.complex128
        assert G_full.shape[1] == len(prims)
        # Im should be non-trivial since the structure phase at G + k is not
        # a pure real / pure imaginary alignment.
        assert float(np.max(np.abs(G_full.imag))) > 1e-3

    @pytest.mark.heavy
    @requires_k025
    def test_build_gto_pw_matrix_explicit_k_overrides_default(self, qe_reference, pgto_shells):
        """Explicit ``k_cart=0`` recovers the Gamma form (this pins that the
        override short-circuits the ``q = G + k`` arithmetic to ``q = G``)."""
        from jqmc.pw_pgto_projection import build_gto_pw_matrix, build_pgto_primitives

        prims = build_pgto_primitives(pgto_shells, qe_reference.atom_positions)
        G_default = build_gto_pw_matrix(qe_reference, prims)
        G_gamma = build_gto_pw_matrix(qe_reference, prims, k_cart=(0.0, 0.0, 0.0))
        # The two differ -- the Gamma-overridden one drops the twist shift.
        assert not np.allclose(G_default, G_gamma, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_NONVACUITY_DISTINCTION)

    # ---------- Convention B + complex MO end-to-end ----------
    @pytest.mark.heavy
    @requires_k025
    def test_qe_pgto_projection_dispatches_complex_ao_leaf(self, qe_reference, pgto_shells):
        """End-to-end: at a non-Gamma twist the QE pGTO projection emits an
        ``AOs_cart_pgto_complex_data`` leaf, complex128 MO coefficients,
        and canonical occupied-only SD storage. Convention B
        (``mo_dn = conj(mo_up)``) is enforced by construction.
        """
        from jqmc.pw_pgto_projection import project_pw_to_pgto
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection

        proj = project_pw_to_pgto(qe_reference, pgto_shells)
        assert proj.coefficients.dtype == np.complex128
        # The projection should pick up the genuine non-Gamma Im content.
        # bcc Li at k=(1/4,1/4,1/4) has |Im(coef)|_max ~ 0.76; we just need
        # the bound safely non-zero (a Gamma-stub leak would round to ~ 1e-15).
        assert float(np.max(np.abs(proj.coefficients.imag))) > 1e-2

        sd, aos, gem = build_wavefunction_from_projection(qe_reference, proj, kind="pgto")
        assert isinstance(aos, AOs_cart_pgto_complex_data)
        assert not isinstance(aos, AOs_cart_pgto_data) or type(aos).__name__ == "AOs_cart_pgto_complex_data"
        # k_point lives on the AO leaf, not on Structure_data.
        assert aos.k_point is not None
        mo_up = np.asarray(gem.orb_data_up_spin.mo_coefficients)
        mo_dn = np.asarray(gem.orb_data_dn_spin.mo_coefficients)
        assert mo_up.dtype == np.complex128
        assert mo_dn.dtype == np.complex128
        assert not hasattr(gem, "lambda_matrix")
        # Convention B (mo_dn = conj(mo_up)) is exact by construction.
        np.testing.assert_array_equal(mo_dn, np.conj(mo_up))

    # ---------- pgto+pw composite at non-Gamma ----------
    @pytest.mark.heavy
    @requires_k025
    def test_qe_pgto_plus_pw_composite_at_non_gamma(self, qe_reference, pgto_shells):
        """End-to-end ``kind='pgto+pw'`` at the non-Gamma Li_k025 fixture.

        At a non-Gamma twist the PW basis cannot use the real cos/sin form (the
        Hermitian ``c_{-G} = conj(c_G)`` symmetry only holds at Gamma); it must
        use the genuinely complex ``e^{i(G+k) . r}`` basis carried by
        :class:`AOs_plane_wave_complex_data`. This test pins:

        * the composite AO leaf is recognised as complex by ``is_complex_aos``,
        * the pGTO child is :class:`AOs_cart_pgto_complex_data`,
        * the PW child is :class:`AOs_plane_wave_complex_data`,
        * MO coefficients are complex128 with genuine non-Gamma Im content,
        * Convention B (``mo_dn = conj(mo_up)``) holds and canonical SD
          stores no ``lambda`` object.

        Regression guard: before the complex-PW landing this combination raised
        ``ValueError: real AO leaf requires real mo_coefficients`` deep inside
        :py:meth:`MOs_data.sanity_check` because the composite's real-PW child
        silently dropped the wavefunction's imaginary part.
        """
        from jqmc.pw_pgto_projection import project_pw_to_mixed
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection

        g_max = float(np.sqrt(2.0 * qe_reference.ecutwfc))
        proj = project_pw_to_mixed(qe_reference, pgto_shells, g_cut=0.5 * g_max, drop_diffuse=False)
        sd, aos, gem = build_wavefunction_from_projection(qe_reference, proj, kind="pgto+pw")

        # k_point lives on the k-dependent composite child
        # (PW in slot 1; pgto in slot 0 is k-independent), not on Structure_data.
        assert isinstance(aos, AOs_composite_data)
        assert aos.children[1].k_point is not None
        assert is_complex_aos(aos)
        child_types = {type(c).__name__ for c in aos.children}
        assert "AOs_cart_pgto_complex_data" in child_types
        assert "AOs_plane_wave_complex_data" in child_types

        mo_up = np.asarray(gem.orb_data_up_spin.mo_coefficients)
        mo_dn = np.asarray(gem.orb_data_dn_spin.mo_coefficients)
        assert mo_up.dtype == np.complex128
        assert mo_dn.dtype == np.complex128
        assert not hasattr(gem, "lambda_matrix")
        # Genuine non-Gamma Im content (a Gamma-stub leak would round to ~ 1e-15).
        assert float(np.max(np.abs(mo_up.imag))) > 1e-2
        # Convention B (mo_dn = conj(mo_up)) is exact by construction.
        np.testing.assert_array_equal(mo_dn, np.conj(mo_up))

    # ---------- pgto+bspline composite at non-Gamma ----------
    @pytest.mark.heavy
    @requires_k025
    def test_qe_pgto_plus_bspline_composite_at_non_gamma(self, qe_reference, pgto_shells):
        """End-to-end ``kind='pgto+bspline'`` at Li_k025 (non-Gamma).

        The B-spline stores the cell-periodic part ``u_k(r)`` of each band. At
        Gamma the bands are real and :class:`AOs_bspline_data` (real control
        points) applies; at non-Gamma the AC step in the projection yields
        complex control points that must NOT be silently real-cast. This pins:

        * the composite is recognised as complex by ``is_complex_aos``,
        * the spline child is :class:`AOs_bspline_complex_data` (not the real
          :class:`AOs_bspline_data`),
        * MO coefficients are complex128 with genuine non-Gamma Im content,
        * Convention B (``mo_dn = conj(mo_up)``) holds and canonical SD
          stores no ``lambda`` object.

        Regression guard: building this combination through the real
        ``AOs_bspline_data.build`` would discard the imaginary part and give a
        real-stub leaf with a corrupted wavefunction; ``AOs_bspline_data.build``
        raises on non-negligible imag, so a Gamma vs non-Gamma dispatch error
        fails loudly.
        """
        from jqmc.pw_pgto_projection import project_pw_to_pgto_spline
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection

        proj = project_pw_to_pgto_spline(qe_reference, pgto_shells, mesh_factor=0.5, order=5)
        sd, aos, gem = build_wavefunction_from_projection(qe_reference, proj, kind="pgto+bspline")

        # k_point lives on the k-dependent composite child
        # (bspline in slot 1; pgto in slot 0 is k-independent).
        assert isinstance(aos, AOs_composite_data)
        assert aos.children[1].k_point is not None
        assert is_complex_aos(aos)
        child_types = {type(c).__name__ for c in aos.children}
        assert "AOs_cart_pgto_complex_data" in child_types
        assert "AOs_bspline_complex_data" in child_types

        mo_up = np.asarray(gem.orb_data_up_spin.mo_coefficients)
        mo_dn = np.asarray(gem.orb_data_dn_spin.mo_coefficients)
        assert mo_up.dtype == np.complex128
        assert mo_dn.dtype == np.complex128
        assert not hasattr(gem, "lambda_matrix")
        assert float(np.max(np.abs(mo_up.imag))) > 1e-2
        np.testing.assert_array_equal(mo_dn, np.conj(mo_up))

    # ---------- Single-basis (pw / bspline) at non-Gamma (single-k) ----------
    @pytest.mark.heavy
    @requires_k025
    def test_qe_pw_only_at_non_gamma(self, qe_reference):
        """``kind='pw'`` at a non-Gamma twist dispatches to the complex PW leaf.

        The single-basis PW path goes through ``_build_pw_leaf_and_mo``,
        which branches on ``aos_data.k_point``. At Li_k025 (non-Gamma)
        the result must be :class:`AOs_plane_wave_complex_data` carrying the
        raw QE Fourier amplitudes as complex128 MO coefficients.
        """
        from jqmc.pw_pgto_projection import project_pw_to_pw
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection

        proj = project_pw_to_pw(qe_reference)
        sd, aos, gem = build_wavefunction_from_projection(qe_reference, proj, kind="pw")
        # k_point lives on the AO leaf.
        assert aos.k_point is not None
        assert isinstance(aos, AOs_plane_wave_complex_data)
        assert is_complex_aos(aos)
        mo_up = np.asarray(gem.orb_data_up_spin.mo_coefficients)
        mo_dn = np.asarray(gem.orb_data_dn_spin.mo_coefficients)
        assert mo_up.dtype == np.complex128
        assert mo_dn.dtype == np.complex128
        assert not hasattr(gem, "lambda_matrix")
        assert float(np.max(np.abs(mo_up.imag))) > 1e-2
        np.testing.assert_array_equal(mo_dn, np.conj(mo_up))

    @pytest.mark.heavy
    @requires_k025
    def test_qe_bspline_only_at_non_gamma(self, qe_reference):
        """``kind='bspline'`` at a non-Gamma twist dispatches to the complex B-spline leaf.

        The MO matrix is identity (one spline AO per band), so the per-twist
        Bloch content lives entirely in the complex ``spline_coefficients``
        of the leaf rather than in the MO block. Pins that the leaf is
        :class:`AOs_bspline_complex_data` carrying ``complex128`` coefficient
        cubes with non-trivial imaginary part.
        """
        from jqmc.pw_pgto_projection import project_pw_to_spline
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection

        proj = project_pw_to_spline(qe_reference, mesh_factor=0.5, order=5)
        sd, aos, gem = build_wavefunction_from_projection(qe_reference, proj, kind="bspline")
        # k_point lives on the AO leaf.
        assert aos.k_point is not None
        assert isinstance(aos, AOs_bspline_complex_data)
        assert is_complex_aos(aos)
        coeffs = np.asarray(aos.spline_coefficients)
        assert coeffs.dtype == np.complex128
        assert float(np.max(np.abs(coeffs.imag))) > 1e-2
        # MO block is the identity (one spline AO per band).
        mo_up = np.asarray(gem.orb_data_up_spin.mo_coefficients)
        np.testing.assert_allclose(
            mo_up,
            np.eye(mo_up.shape[0], mo_up.shape[1], dtype=mo_up.dtype),
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_LOOSE,
        )

    @pytest.mark.heavy
    @requires_k025
    @pytest.mark.parametrize("kind", ["pw", "bspline"])
    def test_dn_channel_is_tr_pair_of_up_at_non_gamma(self, qe_reference, kind):
        """Restricted dn channel at non-Gamma = conj(psi^{+k}) (BCS / TR pair).

        Regression guard for the ``with_negated_k`` gap on the complex PW /
        B-spline leaves: ``_make_geminal_from_mos`` builds the dn channel as
        ``(conj(C^k), leaf.with_negated_k())``; before the leaves implemented
        ``with_negated_k`` the fallback silently reused the +k leaf, so the
        dn orbital evaluated ``sum conj(c_G) e^{i(G+k).r}`` -- neither
        ``phi^{+k}`` nor ``phi^{-k}`` (measured rel. deviation ~1.6 at
        Li_k025). Checks both spin channels against the direct PW sum,
        evaluated in the production precision zones (plane-wave values
        in ``ao_eval``, contraction in ``mo_eval``; ``_pw_band_values_in_zones``).
        """
        from jqmc.molecular_orbital import compute_MOs
        from jqmc.pw_pgto_projection import project_pw_to_pw, project_pw_to_spline
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection

        if kind == "pw":
            proj = project_pw_to_pw(qe_reference, g_cut=None)
            tol = (
                get_tolerance_min(("ao_eval", "mo_eval"), "strict")[1]
                if is_mixed_precision_enabled()
                else RTOL_PW_PROJECTION_FP64
            )
        else:
            proj = project_pw_to_spline(qe_reference, mesh_factor=2.0, order=5)
            tol = RTOL_QE_PW_RECONSTRUCTION_LOOSE
        _, _, gem = build_wavefunction_from_projection(qe_reference, proj, kind=kind)

        rng = np.random.default_rng(3)
        r = rng.uniform(0.0, 6.0, size=(32, 3))
        G = qe_reference.wfc.millers.astype(np.float64) @ qe_reference.cell.b
        # Unit-cell-normalized QE band: Omega^{-1/2} sum_G c_G e^{i(G+k).r}.
        psi = _pw_band_values_in_zones(qe_reference.wfc.coeffs[proj.band_indices], G + qe_reference.wfc.k_point, r)
        psi = psi / np.sqrt(qe_reference.cell.volume)
        up_vals = np.asarray(compute_MOs(gem.orb_data_up_spin, r))
        dn_vals = np.asarray(compute_MOs(gem.orb_data_dn_spin, r))
        rel_up = np.linalg.norm(up_vals - psi) / np.linalg.norm(psi)
        rel_dn = np.linalg.norm(dn_vals - np.conj(psi)) / np.linalg.norm(psi)
        assert rel_up < tol, f"{kind}: up channel vs psi^+k rel={rel_up:.3e}"
        assert rel_dn < tol, f"{kind}: dn channel vs conj(psi^+k) rel={rel_dn:.3e}"

    @pytest.mark.heavy
    @pytest.mark.parametrize("kind", ["pgto", "pgto+pw", "pgto+bspline"])
    @pytest.mark.parametrize("r_carts", [((0.21, 0.32, 0.43), (1.73, -0.24, 2.57), (-1.31, 2.41, 0.26))])
    def test_projected_pgto_down_channel_is_time_reversed(self, qe_reference, pgto_shells, kind, r_carts):
        """Converted pGTO and both composites reverse every child and the complete MO.

        The committed Li_k025 QE/UPF input is projected at non-Gamma and
        evaluated at three explicit off-lattice points. Each down AO block
        and the complete down MO must equal the conjugated up value with
        the existing strict AO/MO tolerance in the same precision mode.
        This tests time reversal of the projected orbital, not fidelity of
        a finite pGTO projection to the original infinite PW expansion.
        """
        from jqmc.atomic_orbital import compute_AOs
        from jqmc.molecular_orbital import compute_MOs
        from jqmc.pw_pgto_projection import project_pw_to_mixed, project_pw_to_pgto, project_pw_to_pgto_spline
        from jqmc.qe_to_jqmc import build_wavefunction_from_projection

        if kind == "pgto":
            projection = project_pw_to_pgto(qe_reference, pgto_shells)
        elif kind == "pgto+pw":
            g_max = float(np.sqrt(2.0 * qe_reference.ecutwfc))
            projection = project_pw_to_mixed(qe_reference, pgto_shells, g_cut=0.5 * g_max, drop_diffuse=False)
        else:
            projection = project_pw_to_pgto_spline(qe_reference, pgto_shells, mesh_factor=0.5, order=5)
        _, _, geminal = build_wavefunction_from_projection(qe_reference, projection, kind=kind)
        up = geminal.orb_data_up_spin
        down = geminal.orb_data_dn_spin
        assert up.num_mo == down.num_mo and down.num_mo > 0
        r_carts = np.asarray(r_carts)
        atol, rtol = get_tolerance_min(("ao_eval", "mo_eval"), "strict")
        up_children = up.aos_data.children if isinstance(up.aos_data, AOs_composite_data) else (up.aos_data,)
        down_children = down.aos_data.children if isinstance(down.aos_data, AOs_composite_data) else (down.aos_data,)
        assert len(up_children) == len(down_children) == (1 if kind == "pgto" else 2)
        for up_child, down_child in zip(up_children, down_children, strict=True):
            assert type(up_child) is type(down_child)
            up_values = np.asarray(compute_AOs(up_child, r_carts))
            down_values = np.asarray(compute_AOs(down_child, r_carts))
            # A real/zero block would hide a missing phase reversal.
            assert not np.allclose(up_values, np.conj(up_values), atol=atol, rtol=rtol)
            np.testing.assert_allclose(down_values, np.conj(up_values), atol=atol, rtol=rtol)
        up_mos = np.asarray(compute_MOs(up, r_carts))
        down_mos = np.asarray(compute_MOs(down, r_carts))
        assert not np.allclose(up_mos, np.conj(up_mos), atol=atol, rtol=rtol)
        np.testing.assert_allclose(down_mos, np.conj(up_mos), atol=atol, rtol=rtol)

    # ---------- Multi-twist composite (single-k round-trip) ----------

    @requires_k025
    @pytest.mark.parametrize(
        "kind,expected_child_type,leaf_is_composite",
        [
            ("pgto+pw", AOs_plane_wave_complex_data, True),
            ("pgto+bspline", AOs_bspline_complex_data, True),
            ("pw", AOs_plane_wave_complex_data, False),
            ("bspline", AOs_bspline_complex_data, False),
        ],
    )
    @pytest.mark.heavy
    def test_convert_qe_to_twist_average_input_composite_round_trip(
        self, tmp_path, kind, expected_child_type, leaf_is_composite
    ):
        """Multi-twist driver supports the composite kinds end-to-end.

        Uses the single-k ``Li_k025`` fixture (n_twist=1) as the minimum
        workload to exercise the dispatch + per-twist composite-child
        save / load round-trip.
        Verifies:

        * the driver accepts ``kind="pgto+pw"`` / ``"pgto+bspline"``,
        * ``/_meta`` and ``/twists/twist_*`` are written with the per-twist
          ``composite_child`` group carrying the k-dependent child state,
        * ``load_twist`` rebuilds the composite with the right child type and
          ``is_complex_aos`` returns ``True`` on the result,
        * ``Hamiltonian_data.sanity_check`` passes on the loaded twist.

        The Li_k_twist fixture (the full 8-point unfolded MP grid) is covered
        in ``test_integration_qe_multi_twist.py``; the contract here is
        identical at n_twist=1.
        """
        from jqmc._twist_average import load_twist, load_twist_average_input
        from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input
        from jqmc.upf_wrapper import parse_gaussinfo

        ecp = parse_gaussinfo(str(_UPF_PATH))
        out_path = str(tmp_path / f"{kind.replace('+', '_')}_mt.h5")
        convert_qe_to_twist_average_input(
            qe_save_dir=str(_FIXTURE_DIR),
            output_path=out_path,
            upf_paths={"Li": str(_UPF_PATH)},
            kind=kind,
            ecp_by_element={"Li": ecp},
            j1_param=None,
            j2_param=None,
        )

        meta = load_twist_average_input(out_path)
        assert meta.n_twist == 1
        assert meta.weights[0] == pytest.approx(2.0, abs=ATOL_PYTHON_FP64_LOOSE)

        ham = load_twist(out_path, 0)
        ham.sanity_check()
        aos = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
        assert is_complex_aos(aos)
        if leaf_is_composite:
            assert isinstance(aos, AOs_composite_data)
            assert any(isinstance(c, expected_child_type) for c in aos.children), (
                f"expected one child of type {expected_child_type.__name__}, got {[type(c).__name__ for c in aos.children]}"
            )
        else:
            assert isinstance(aos, expected_child_type), (
                f"expected bare {expected_child_type.__name__}, got {type(aos).__name__}"
            )

        mo_up = np.asarray(ham.wavefunction_data.geminal_data.orb_data_up_spin.mo_coefficients)
        mo_dn = np.asarray(ham.wavefunction_data.geminal_data.orb_data_dn_spin.mo_coefficients)
        assert mo_up.dtype == np.complex128
        assert mo_dn.dtype == np.complex128
        assert not hasattr(ham.wavefunction_data.geminal_data, "lambda_matrix")
        np.testing.assert_array_equal(mo_dn, np.conj(mo_up))

    @requires_k025
    @pytest.mark.heavy
    def test_twist_fixed_cutoff_runs_one_shared_exponent_optimisation(self, tmp_path, monkeypatch):
        """Numeric multi-twist conversion optimises once before the k loop."""
        from types import SimpleNamespace

        from jqmc import pgto_exponent_optimizer
        from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input

        calls = []

        def fake_optimizer(references, shells_per_atom, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                shells_per_atom=shells_per_atom,
                accepted=False,
                deployed_initial_spilling=1.0e-3,
                deployed_optimized_spilling=1.1e-3,
            )

        monkeypatch.setattr(pgto_exponent_optimizer, "optimize_pgto_exponents", fake_optimizer)
        out_path = str(tmp_path / "fixed_twist_shared_exponents.h5")
        convert_qe_to_twist_average_input(
            qe_save_dir=str(_FIXTURE_DIR),
            output_path=out_path,
            upf_paths={"Li": str(_UPF_PATH)},
            kind="pgto+pw",
            ecutwfc=10.0,
            j1_param=None,
            j2_param=None,
        )

        assert len(calls) == 1
        assert calls[0]["fixed_mixed_overlap"] is False
        assert calls[0]["deployed_fixed_mixed_overlap"] is True
        assert calls[0]["g_cut"] == pytest.approx(np.sqrt(10.0))
        assert Path(out_path).is_file()

    @requires_k025
    @pytest.mark.heavy
    def test_twist_auto_runs_shared_exponent_optimisation_before_k_loop(self, tmp_path, monkeypatch):
        """Auto conversion adopts one shared exponent set before projections."""
        from types import SimpleNamespace

        from jqmc import pgto_exponent_optimizer, pw_pgto_projection
        from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input

        events = []
        real_sweep = pw_pgto_projection.optimize_mixed_basis
        sweep_calls = 0
        trial_g_cut = float(np.sqrt(7.5))

        def fake_sweep(*args, **kwargs):
            nonlocal sweep_calls
            sweep_calls += 1
            events.append(f"sweep-{sweep_calls}")
            if sweep_calls == 1:
                trial = SimpleNamespace(
                    mean_residual=1.0e-3,
                    alpha_min=0.1,
                    g_cut=trial_g_cut,
                )
                return SimpleNamespace(trials=[trial], best=SimpleNamespace(g_cut=np.sqrt(15.0)))
            if sweep_calls == 2:
                return SimpleNamespace(
                    target_met=True,
                    best=SimpleNamespace(g_cut=trial_g_cut, mean_residual=1.0e-6),
                )
            return real_sweep(*args, **kwargs)

        def fake_optimizer(references, shells_per_atom, **kwargs):
            events.append("optimise")
            return SimpleNamespace(shells_per_atom=shells_per_atom, accepted=True)

        monkeypatch.setattr(pw_pgto_projection, "optimize_mixed_basis", fake_sweep)
        monkeypatch.setattr(pgto_exponent_optimizer, "optimize_pgto_exponents", fake_optimizer)
        out_path = str(tmp_path / "auto_twist_shared_exponents.h5")
        convert_qe_to_twist_average_input(
            qe_save_dir=str(_FIXTURE_DIR),
            output_path=out_path,
            upf_paths={"Li": str(_UPF_PATH)},
            kind="pgto+pw",
            ecutwfc="auto",
            j1_param=None,
            j2_param=None,
        )

        assert events[:3] == ["sweep-1", "optimise", "sweep-2"]
        assert events.count("optimise") == 1
        assert Path(out_path).is_file()

    # ---------- MCMC smoke (complex Bloch e_L chain) ----------
    @pytest.mark.heavy
    @requires_k025
    def test_qe_non_gamma_mcmc_smoke_no_complex_contamination(self, qe_reference, pgto_shells):
        """Smoke check: a short MCMC chain at the QE non-Gamma twist runs
        end-to-end without firing ``ComplexWarning`` (caught by the
        ``conftest._strict_complex_cast_guard`` autouse fixture). Returns
        a finite real ``E_mean`` after the ``.real`` projection.

        This is a systematic contamination guard, not a milestone -- 30 steps
        is far too few to compare against the QE SCF total energy
        (~ -0.54 Ha for Li_k025; see ``<etot>`` in ``data-file-schema.xml``).
        """
        from jqmc.jqmc_mcmc import MCMC
        from jqmc.pw_pgto_projection import project_pw_to_pgto
        from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction, build_wavefunction_from_projection
        from jqmc.upf_wrapper import parse_gaussinfo

        proj = project_pw_to_pgto(qe_reference, pgto_shells)
        sd, aos, gem = build_wavefunction_from_projection(qe_reference, proj, kind="pgto")
        ecp = parse_gaussinfo(str(_UPF_PATH))
        ham = build_hamiltonian_from_wavefunction(
            sd,
            aos,
            gem,
            j1_param=None,
            j2_param=None,
            ecp_by_element={"Li": ecp},
        )
        ham.sanity_check()
        mcmc = MCMC(hamiltonian_data=ham, num_walkers=2, mcmc_seed=12345, epsilon_AS=0.0)
        mcmc.run(num_mcmc_steps=30)
        E, Estd, _, _ = mcmc.get_E(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=5)
        assert np.isfinite(E)
        assert np.isfinite(Estd)

    @pytest.mark.heavy
    @requires_k025
    def test_qe_non_gamma_pgto_plus_bspline_mcmc_smoke(self, qe_reference, pgto_shells):
        """MCMC smoke for ``kind='pgto+bspline'`` at the non-Gamma Li_k025 twist.

        Mirrors :func:`test_qe_non_gamma_mcmc_smoke_no_complex_contamination`
        but exercises the (complex pGTO + complex B-spline) composite so the
        full e_L chain runs against both children, including the per-electron
        ``e^{i k . r}`` Bloch phase that the complex spline kernel applies.
        """
        from jqmc.jqmc_mcmc import MCMC
        from jqmc.pw_pgto_projection import project_pw_to_pgto_spline
        from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction, build_wavefunction_from_projection
        from jqmc.upf_wrapper import parse_gaussinfo

        proj = project_pw_to_pgto_spline(qe_reference, pgto_shells, mesh_factor=0.5, order=5)
        sd, aos, gem = build_wavefunction_from_projection(qe_reference, proj, kind="pgto+bspline")
        assert is_complex_aos(aos)
        ecp = parse_gaussinfo(str(_UPF_PATH))
        ham = build_hamiltonian_from_wavefunction(
            sd,
            aos,
            gem,
            j1_param=None,
            j2_param=None,
            ecp_by_element={"Li": ecp},
        )
        ham.sanity_check()
        mcmc = MCMC(hamiltonian_data=ham, num_walkers=2, mcmc_seed=12345, epsilon_AS=0.0)
        mcmc.run(num_mcmc_steps=30)
        E, Estd, _, _ = mcmc.get_E(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=5)
        assert np.isfinite(E)
        assert np.isfinite(Estd)

    @pytest.mark.heavy
    @requires_k025
    def test_qe_non_gamma_pgto_plus_pw_mcmc_smoke(self, qe_reference, pgto_shells):
        """MCMC smoke for ``kind='pgto+pw'`` at the non-Gamma Li_k025 twist.

        Mirrors :func:`test_qe_non_gamma_mcmc_smoke_no_complex_contamination`
        but exercises the composite (complex pGTO + complex PW) AO leaf so the
        full e_L chain (kinetic, ECP non-local, Ewald) runs against both
        children. The ComplexWarning autouse guard in conftest catches any
        silent complex -> real cast inside the MCMC kernel.
        """
        from jqmc.jqmc_mcmc import MCMC
        from jqmc.pw_pgto_projection import project_pw_to_mixed
        from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction, build_wavefunction_from_projection
        from jqmc.upf_wrapper import parse_gaussinfo

        g_max = float(np.sqrt(2.0 * qe_reference.ecutwfc))
        proj = project_pw_to_mixed(qe_reference, pgto_shells, g_cut=0.5 * g_max, drop_diffuse=False)
        sd, aos, gem = build_wavefunction_from_projection(qe_reference, proj, kind="pgto+pw")
        assert is_complex_aos(aos)
        ecp = parse_gaussinfo(str(_UPF_PATH))
        ham = build_hamiltonian_from_wavefunction(
            sd,
            aos,
            gem,
            j1_param=None,
            j2_param=None,
            ecp_by_element={"Li": ecp},
        )
        ham.sanity_check()
        mcmc = MCMC(hamiltonian_data=ham, num_walkers=2, mcmc_seed=12345, epsilon_AS=0.0)
        mcmc.run(num_mcmc_steps=30)
        E, Estd, _, _ = mcmc.get_E(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=5)
        assert np.isfinite(E)
        assert np.isfinite(Estd)
