"""QE multi-twist end-to-end conversion tests.

The QE multi-k converter enumerates the per-k
``wfcN.hdf5`` files, reads the IBZ weights from
``data-file-schema.xml``'s ``<k_point weight=...>`` attribute, runs the
pGTO projection once per k, and emits the combined HDF5 input
(``/_meta`` + ``/shared`` + ``/twists``; ``/_meta`` + ``/shared`` + ``/twists/twist_*``).

Fixture: ``Li_k_twist/out/li_twist.save`` -- a 2x2x2 shifted Monkhorst-
Pack run on bcc Li with Im-3m symmetry, scf with ``nosym=.true.`` +
``noinv=.true.`` so the 8 shifted k-points are kept UNFOLDED
(otherwise the cubic point group + time-reversal would IBZ-collapse
them to a single representative). Each of the 8 twists carries
weight 0.25 (QE's paramagnetic convention; the explicit weights are
1/8 each but doubled by QE for the spin factor -> 1/4 per stored
twist, sum = 2.0). Mirrors the unfolded pyscf fixture
``tests/trexio_example_files/lithium_twist_222.h5`` (also 8 twists)
so the QE-side and pyscf-side converter codepaths can be cross-
validated on the same mesh.
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

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, QE_MULTI_TWIST_FIXTURE_K_NORM_UPPER
from tests._test_data_cache import qe_example_files_root

_FIXTURE_DIR = qe_example_files_root() / "Li_k_twist" / "out" / "li_twist.save"
_UPF_PATH = _FIXTURE_DIR / "Li.ccECP.upf"
_available = (
    (_FIXTURE_DIR / "data-file-schema.xml").is_file()
    and (_FIXTURE_DIR / "wfc1.hdf5").is_file()
    and (_FIXTURE_DIR / "wfc8.hdf5").is_file()
    and _UPF_PATH.is_file()
)

requires_multitwist = pytest.mark.skipif(not _available, reason="Li_k_twist QE fixture not present")

_GAMMA_FIXTURE_DIR = qe_example_files_root() / "Li_gamma" / "out" / "li.save"
_GAMMA_UPF_PATH = _GAMMA_FIXTURE_DIR / "Li.ccECP.upf"
_gamma_available = (
    (_GAMMA_FIXTURE_DIR / "data-file-schema.xml").is_file()
    and (_GAMMA_FIXTURE_DIR / "wfc1.hdf5").is_file()
    and _GAMMA_UPF_PATH.is_file()
)
requires_gamma = pytest.mark.skipif(not _gamma_available, reason="Li_gamma QE fixture not present")


_J3_POINTS = np.asarray([[0.31, 0.47, 0.13], [-0.52, 0.21, 0.66], [1.4, -0.9, 0.35]])


def assert_twists_share_one_phase_free_j3(path):
    """Every loaded twist passes sanity_check and carries the same lattice-periodic AO J3.

    The J3 basis is evaluated at fixed points: equal values for every twist
    show one shared object, and invariance under a lattice translation shows
    that no Bloch phase remains. Returns the J3 of the first twist.
    """
    from jqmc._precision import get_tolerance_min
    from jqmc._twist_average import load_twist, load_twist_average_input
    from jqmc.hamiltonians import _detile_aos_data_for_save

    meta = load_twist_average_input(path)
    assert meta.n_twist > 1 and len({tuple(k) for k in meta.k_points}) == meta.n_twist
    atol, rtol = get_tolerance_min(("ao_eval",), "strict")
    first = None
    for index in range(meta.n_twist):
        ham = load_twist(path, index)
        ham.sanity_check()
        j3 = ham.wavefunction_data.jastrow_data.jastrow_three_body_data
        assert j3.orb_data.k_point is None
        assert set(_detile_aos_data_for_save(j3.orb_data).nucleus_index) == set(range(ham.structure_data.natom))
        np.testing.assert_array_equal(j3.orb_data.structure_data.cell, ham.structure_data.cell)
        values = np.asarray(j3.compute_orb_api(j3.orb_data, _J3_POINTS))
        translated = _J3_POINTS + np.asarray(ham.structure_data.cell)[[0, 1, 2]].sum(axis=0)
        np.testing.assert_allclose(np.asarray(j3.compute_orb_api(j3.orb_data, translated)), values, atol=atol, rtol=rtol)
        if first is None:
            first, first_values = j3, values
            continue
        assert type(j3.orb_data) is type(first.orb_data)
        np.testing.assert_array_equal(values, first_values)
        np.testing.assert_array_equal(j3.j_matrix, first.j_matrix)
    return first


# ---------- k-point enumeration + per-k QEReference round-trip ----------


class TestQEMultiTwist:
    @pytest.mark.heavy
    @requires_gamma
    @pytest.mark.parametrize("kind", ["bspline", "pgto+bspline"])
    def test_gamma_bspline_twist_roundtrip_with_dtn(self, kind, tmp_path):
        """Gamma B-spline twists use the uniform complex combined encoding.

        The conversion must neither raise a ``.k_point`` AttributeError nor
        reject the real B-spline child.  It also pins that the
        twist-independent DTN Jastrow is attached to the shared Hamiltonian.
        """
        from jqmc._twist_average import load_twist, load_twist_average_input
        from jqmc.atomic_orbital import AOs_bspline_complex_data, AOs_composite_data
        from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input

        out_path = str(tmp_path / f"gamma_{kind.replace('+', '_')}_dtn.h5")
        convert_qe_to_twist_average_input(
            qe_save_dir=str(_GAMMA_FIXTURE_DIR),
            output_path=out_path,
            upf_paths={"Li": str(_GAMMA_UPF_PATH)},
            kind=kind,
            mesh_factor=0.5,
            j1_param=None,
            j2_param=None,
            jastrow_dtn="u_chi_f",
            dtn_c=4,
            dtn_n_u=6,
            dtn_n_chi=6,
            dtn_n_eN=2,
            dtn_n_ee=2,
        )

        meta = load_twist_average_input(out_path)
        assert meta.n_twist == 1
        assert tuple(meta.k_points[0]) == (0.0, 0.0, 0.0)

        ham = load_twist(out_path, 0)
        ham.sanity_check()
        aos = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
        child = aos.children[1] if isinstance(aos, AOs_composite_data) else aos
        assert isinstance(child, AOs_bspline_complex_data)
        assert tuple(child.k_point) == (0.0, 0.0, 0.0)

        dtn = ham.wavefunction_data.jastrow_data.jastrow_dtn_data
        assert dtn is not None
        assert (dtn.use_u, dtn.use_chi, dtn.use_f) == (True, True, True)
        assert dtn.n_order_u == 6
        assert dtn.n_order_chi == 6
        assert dtn.n_order_eN == 2
        assert dtn.n_order_ee == 2
        # Li.ccECP removes two core electrons: z_eff = 3 - 2 = 1.
        np.testing.assert_array_equal(np.asarray(dtn.z_eff_per_species), [1.0])

    @pytest.mark.heavy
    @requires_multitwist
    def test_load_qe_kpoint_list_enumerates_all_twists(self):
        """``load_qe_kpoint_list`` enumerates every ``ks_energies`` block in
        the XML. The Li_k_twist fixture's shifted 2x2x2 MP grid yields 8
        k-points (UNFOLDED via ``nosym=.true. + noinv=.true.`` in the QE
        input). Per-twist weight = 1/4 (QE's paramagnetic 1/n_twist x 2
        spin factor = 2/8 = 1/4); the eight weights sum to 2.0."""
        from jqmc.qe_wrapper import load_qe_kpoint_list

        kpts = load_qe_kpoint_list(str(_FIXTURE_DIR / "data-file-schema.xml"))
        assert len(kpts) == 8
        weights = np.array([w for _, w in kpts])
        assert weights.sum() == pytest.approx(2.0, abs=ATOL_PYTHON_FP64_LOOSE)
        # Uniform unreduced grid -- every weight is 1/4 exactly.
        np.testing.assert_array_equal(weights, np.full(8, 0.25))
        # Each |xk| ~ sqrt(3)/4 * 2*pi/alat with alat = 6.6329 Bohr.
        for xk, _ in kpts:
            assert xk.shape == (3,)
            assert 0.0 < float(np.linalg.norm(xk)) < QE_MULTI_TWIST_FIXTURE_K_NORM_UPPER

    @pytest.mark.heavy
    @requires_multitwist
    def test_load_qe_reference_per_k_matches_xml_kpoint_list(self):
        """``load_qe_reference(save_dir, ik=N)`` reads the N-th wfc file and
        its xk attribute matches the XML's N-th ``<k_point>``. Pins the
        1-based wfc indexing convention -- a Python-style 0-based ``range(n_kpts)``
        would silently drop ``wfc{n_kpts}.hdf5`` and read a non-existent
        ``wfc0.hdf5``. Exercised on the first, mid, and last twists."""
        from jqmc.qe_wrapper import load_qe_kpoint_list, load_qe_reference

        kpts = load_qe_kpoint_list(str(_FIXTURE_DIR / "data-file-schema.xml"))
        for ik in [1, 4, 8]:
            ref = load_qe_reference(str(_FIXTURE_DIR), ik=ik)
            np.testing.assert_array_equal(ref.wfc.k_point, kpts[ik - 1][0])

    # ---------- full convert + combined HDF5 round-trip ----------
    @pytest.mark.heavy
    @requires_multitwist
    def test_convert_qe_to_twist_average_input_writes_combined_with_all_twists(self, tmp_path):
        """End-to-end: ``convert_qe_to_twist_average_input`` writes a combined
        combined HDF5 with one entry per IBZ k-point; the meta header,
        per-twist k_point, weight, and dtype contracts (occupied complex128
        MO rows, no SD lambda, and Convention B
        ``mo_dn == conj(mo_up)``) all hold."""
        from jqmc._twist_average import load_twist, load_twist_average_input
        from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input
        from jqmc.upf_wrapper import parse_gaussinfo

        ecp = parse_gaussinfo(str(_UPF_PATH))
        out_path = str(tmp_path / "ha_twists.h5")
        convert_qe_to_twist_average_input(
            qe_save_dir=str(_FIXTURE_DIR),
            output_path=out_path,
            upf_paths={"Li": str(_UPF_PATH)},
            kind="pgto",
            j1_param=None,
            j2_param=None,
            ecp_by_element={"Li": ecp},
        )

        meta = load_twist_average_input(out_path)
        assert meta.n_twist == 8
        assert meta.bz_weight_sum == pytest.approx(2.0, abs=ATOL_PYTHON_FP64_LOOSE)
        np.testing.assert_array_equal(meta.weights.sum(), 2.0)
        np.testing.assert_array_equal(meta.weights, np.full(8, 0.25))
        # The unfolded 2x2x2 shifted MP grid carries 8 points at fractional
        # (+/-1/4, +/-1/4, +/-1/4); the converter writes each verbatim.
        for k in meta.k_points:
            assert k is not None
            assert len(k) == 3

        # Load every twist; each must (a) satisfy sanity_check, (b) carry
        # exactly the occupied complex MO rows and no SD lambda, and (c) preserve
        # Convention B at the dataset level.
        for ik in range(meta.n_twist):
            ham_k = load_twist(out_path, ik)
            ham_k.sanity_check()
            mo_up = np.asarray(ham_k.wavefunction_data.geminal_data.orb_data_up_spin.mo_coefficients)
            mo_dn = np.asarray(ham_k.wavefunction_data.geminal_data.orb_data_dn_spin.mo_coefficients)
            assert mo_up.dtype == np.complex128
            assert mo_dn.dtype == np.complex128
            assert mo_up.shape[0] == ham_k.wavefunction_data.geminal_data.num_electron_up
            assert mo_dn.shape[0] == ham_k.wavefunction_data.geminal_data.num_electron_dn
            assert not hasattr(ham_k.wavefunction_data.geminal_data, "lambda_matrix")
            # Convention B at the storage layer.
            np.testing.assert_array_equal(mo_dn, np.conj(mo_up))
            # Per-twist contract:
            # The occupied ``num_mo`` is fixed by electron count and uniform
            # across twists. ``num_ao`` may legitimately differ per twist for
            # composite kinds (``pgto+pw`` / ``pgto+bspline``): QE stores a
            # per-k G-set with ``|G+k|^2 < ecutwfc``, so the PW child's
            # ``num_ao_pw`` is k-dependent. Assert occupied-row uniformity;
            # heterogeneous num_ao is handled by the combined layout (each
            # twist's MO array is stored independently) and by the
            # orchestrators (each per-twist MCMC instance carries its own
            # ``aos_data.num_ao``).
            if ik == 0:
                expected_num_mo = mo_up.shape[0]
            else:
                assert mo_up.shape[0] == expected_num_mo, f"twist {ik}: num_mo={mo_up.shape[0]} != expected {expected_num_mo}"

    # ---------- shared AO J3 on the untiled twist route ----------
    @pytest.mark.light
    def test_twist_average_writes_one_shared_phase_free_ao_j3(self, tmp_path):
        """Untiled QE twist averaging builds one independent periodic AO J3 for the cell.

        The plane-wave determinant has no GTO leaf, so the Cartesian J3 basis
        must come from the twist-shared UPF seeds rather than a twist's Bloch
        AO container. Every twist loads that same zero-initialized J3.
        """
        from jqmc.atomic_orbital import AOs_cart_pgto_data
        from jqmc.jqmc_qe import run_qe2jqmc

        out_path = str(tmp_path / "pw_j3_twists.h5")
        run_qe2jqmc(
            {"qe_save_dir": str(_FIXTURE_DIR), "kind": "pw", "twist_average": True},
            {"j2_parameter": 1.0, "j3_basis_type": "ao-small"},
            out_path,
        )
        j3 = assert_twists_share_one_phase_free_j3(out_path)
        assert isinstance(j3.orb_data, AOs_cart_pgto_data)
        assert j3.j_matrix.shape == (j3.orb_num, j3.orb_num + 1)
        np.testing.assert_array_equal(j3.j_matrix, np.zeros_like(j3.j_matrix))

    @pytest.mark.light
    def test_twist_average_ao_j3_resolves_tagged_species_by_element_upf_key(self, tmp_path):
        """A tagged species ``Li1`` with the element UPF key ``Li`` seeds the shared J3 basis.

        The committed save is copied and only its XML species/atom names are
        renamed, so the J3 seed lookup must use the label-or-element mapping
        that the converter accepts for every other UPF use.
        """
        import shutil

        from jqmc.jqmc_qe import run_qe2jqmc

        save_dir = tmp_path / "li_tagged.save"
        shutil.copytree(_FIXTURE_DIR, save_dir)
        xml = save_dir / "data-file-schema.xml"
        text = xml.read_text()
        assert text.count('name="Li"') > 0
        xml.write_text(text.replace('name="Li"', 'name="Li1"'))
        out_path = str(tmp_path / "tagged_j3_twists.h5")
        run_qe2jqmc(
            {"qe_save_dir": str(save_dir), "kind": "pw", "twist_average": True, "upf_paths": [f"Li={_UPF_PATH}"]},
            {"j2_parameter": 1.0, "j3_basis_type": "ao-small"},
            out_path,
        )
        j3 = assert_twists_share_one_phase_free_j3(out_path)
        assert j3.orb_num > 0

    @pytest.mark.light
    def test_twist_average_rejects_mo_j3_before_conversion(self, tmp_path):
        """``j3_basis_type='mo'`` fails with the twist-dependence reason and writes nothing."""
        from jqmc.jqmc_qe import run_qe2jqmc

        out_path = tmp_path / "rejected.h5"
        with pytest.raises(ValueError, match="twist-dependent"):
            run_qe2jqmc(
                {"qe_save_dir": str(_FIXTURE_DIR), "kind": "pw", "twist_average": True},
                {"j2_parameter": 1.0, "j3_basis_type": "mo"},
                str(out_path),
            )
        assert not out_path.exists()

    # ---------- smoke MCMC on a loaded twist (no Im truncation) ----------
    @pytest.mark.heavy
    @requires_multitwist
    def test_loaded_twist_runs_mcmc_smoke(self, tmp_path):
        """Sanity-level guard: an MCMC chain runs end-to-end on a loaded
        non-Gamma twist without firing the ``ComplexWarning`` autouse guard
        in ``conftest.py``. Pins that the combined load -> MCMC chain preserves
        Im along every dtype-zone cast site (the systematic
        contamination guard, applied to the combined-loaded Hamiltonian).

        Energy magnitude is **not** asserted against the QE SCF total --
        bcc Li is a metal and ``VMC ~ E_HF`` is not the contract of this
        short chain. The contract here is
        "the chain flows; the autouse guard would have failed otherwise".
        """
        from jqmc._twist_average import load_twist
        from jqmc.jqmc_mcmc import MCMC
        from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input
        from jqmc.upf_wrapper import parse_gaussinfo

        ecp = parse_gaussinfo(str(_UPF_PATH))
        out_path = str(tmp_path / "ha_twists.h5")
        convert_qe_to_twist_average_input(
            qe_save_dir=str(_FIXTURE_DIR),
            output_path=out_path,
            upf_paths={"Li": str(_UPF_PATH)},
            kind="pgto",
            j1_param=None,
            j2_param=None,
            ecp_by_element={"Li": ecp},
        )

        # Pick a mid-mesh twist (any of 0..7 works; choosing 3 avoids
        # accidental ``ik == 0`` special-casing in the chain).
        ham = load_twist(out_path, 3)
        mcmc = MCMC(hamiltonian_data=ham, num_walkers=2, mcmc_seed=12345, epsilon_AS=0.0)
        mcmc.run(num_mcmc_steps=30)
        E, Estd, _, _ = mcmc.get_E(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=5)
        assert np.isfinite(E)
        assert np.isfinite(Estd)
