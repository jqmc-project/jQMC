"""Unit tests for jqmc.qe_wrapper (QE PW reference HDF5 + XML loader)."""

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

import shutil
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE
from jqmc.qe_wrapper import (
    QECell,
    QEReference,
    _resolve_atom_elements,
    expand_gamma_half,
    load_qe_atoms,
    load_qe_bands,
    load_qe_cell,
    load_qe_ecutwfc,
    load_qe_reference,
    load_qe_species_pseudo,
    load_qe_wfc,
)
from tests._test_data_cache import qe_example_files_root

# Committed gamma-only QE reference (2-atom Li, 8 bands, ecutwfc 30 Ha). The
# loader tests below are generic -- the numbers are just this fixture's
# goldens -- so they run unconditionally: a missing fixture must fail loudly,
# not skip.
_SAVE_DIR = qe_example_files_root() / "Li_gamma" / "out" / "li.save"
_XML = _SAVE_DIR / "data-file-schema.xml"
_WFC = _SAVE_DIR / "wfc1.hdf5"


class TestQEWrapper:
    @pytest.mark.light
    def test_qe_cell_reciprocal_is_dual(self):
        """QE cell reciprocal is dual."""
        a = np.array([[-3.4158, 0.0, 3.4158], [0.0, 3.4158, 3.4158], [-3.4158, 3.4158, 0.0]])
        cell = QECell(a=a)
        # a_i . b_j = 2*pi*delta_ij.
        gram = cell.a @ cell.b.T
        np.testing.assert_allclose(gram, 2.0 * np.pi * np.eye(3), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
        assert cell.volume > 0.0

    @pytest.mark.light
    def test_expand_gamma_half_conjugate_pairs(self):
        """Expand gamma half conjugate pairs."""
        millers = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int64)
        coeffs = np.array([[1.0 + 0j, 2.0 + 1j, 3.0 - 2j]], dtype=np.complex128)
        m_full, c_full = expand_gamma_half(millers, coeffs)
        # M = 2*n_g - 1 (G=0 not duplicated).
        assert m_full.shape == (5, 3)
        assert c_full.shape == (1, 5)
        # The appended partners are -G with conjugate coefficients.
        np.testing.assert_array_equal(m_full[3:], -millers[1:])
        np.testing.assert_array_equal(c_full[0, 3:], np.conj(coeffs[0, 1:]))

    @pytest.mark.light
    def test_load_qe_cell_and_atoms(self):
        """Load QE cell and atoms."""
        cell = load_qe_cell(str(_XML))
        assert cell.a.shape == (3, 3)
        assert cell.volume > 0.0
        labels, pos = load_qe_atoms(str(_XML))
        assert all(lbl == "Li" for lbl in labels)
        assert pos.shape == (len(labels), 3)

    @pytest.mark.light
    def test_load_qe_ecut_and_bands(self):
        """Load QE ecut and bands."""
        ecut = load_qe_ecutwfc(str(_XML))
        assert ecut == pytest.approx(30.0)
        bands = load_qe_bands(str(_XML))
        assert bands.n_electrons == pytest.approx(2.0)
        assert bands.eigenvalues.shape == bands.occupations.shape
        occ_idx = bands.occupied_band_indices()
        assert occ_idx.size >= 1
        assert occ_idx.size <= bands.eigenvalues.size

    @pytest.mark.light
    def test_load_qe_wfc_gamma_expanded(self):
        """Load QE wfc gamma expanded."""
        wfc = load_qe_wfc(str(_WFC))
        assert wfc.gamma_only is True
        assert wfc.n_bands == 8
        # Full G set = 2*igwx - 1 = 2*1151 - 1.
        assert wfc.n_gvectors == 2 * 1151 - 1
        assert wfc.millers.shape == (wfc.n_gvectors, 3)
        assert wfc.coeffs.shape == (8, wfc.n_gvectors)
        # Exactly one G=0.
        assert int(np.sum(np.all(wfc.millers == 0, axis=1))) == 1

    @pytest.mark.light
    def test_load_qe_wfc_real_orbitals_hermitian(self):
        # For a gamma_only real orbital, expanding the half set should make the
        # coefficient array Hermitian-symmetric: c(-G) = conj(c(G)).
        """Load QE wfc real orbitals hermitian."""
        wfc = load_qe_wfc(str(_WFC))
        millers = wfc.millers
        # Map each G to its row index.
        index = {tuple(m): i for i, m in enumerate(millers.tolist())}
        rng = np.random.default_rng(0)
        sample = rng.choice(wfc.n_gvectors, size=20, replace=False)
        for i in sample:
            neg = tuple((-millers[i]).tolist())
            j = index[neg]
            np.testing.assert_array_equal(wfc.coeffs[:, j], np.conj(wfc.coeffs[:, i]))

    @pytest.mark.light
    def test_load_qe_reference_bundle(self):
        """The QE bundle preserves metadata, Gamma and the nonzero reciprocal lattice."""
        ref = load_qe_reference(str(_SAVE_DIR))
        assert isinstance(ref, QEReference)
        assert ref.ecutwfc == pytest.approx(30.0)
        assert ref.wfc.n_bands == 8
        assert len(ref.atom_labels) == ref.atom_positions.shape[0]
        # Gamma membership alone cannot detect mis-scaled nonzero G vectors.
        g = ref.wfc.g_vectors(ref.cell)
        g2_half = 0.5 * np.einsum("ij,ij->i", g, g)
        assert g2_half.min() == pytest.approx(0.0, abs=ATOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(
            g @ ref.cell.a.T,
            2.0 * np.pi * ref.wfc.millers,
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_LOOSE,
        )

    @pytest.mark.light
    def test_load_qe_reference_accepts_parent_and_file(self):
        # Parent directory containing a single *.save.
        """Load QE reference accepts parent and file."""
        ref_parent = load_qe_reference(str(_SAVE_DIR.parent))
        # A file path inside the .save directory.
        ref_file = load_qe_reference(str(_WFC))
        assert ref_parent.wfc.n_gvectors == ref_file.wfc.n_gvectors


# Real O ccECP UPF used to resolve the element behind tagged species labels.
_O_UPF = qe_example_files_root() / "O-lsda_gamma" / "out" / "o.save" / "O.ccECP.upf"
requires_o_upf = pytest.mark.skipif(not _O_UPF.is_file(), reason="O.ccECP.upf fixture not present")

_AFM_XML = (
    "<root><output><atomic_species ntyp='2'>"
    "<species name='O'><pseudo_file>O.ccECP.upf</pseudo_file></species>"
    "<species name='O1'><pseudo_file>O.ccECP.upf</pseudo_file></species>"
    "</atomic_species></output></root>"
)


class TestAntiferromagneticLabels:
    """QE tags magnetically inequivalent sites of one element with distinct
    labels ('O' vs 'O1'); the element must come from each species' UPF
    PP_HEADER, never from parsing the free-form label string.
    """

    @pytest.mark.light
    def test_species_pseudo_map_reads_tagged_labels(self, tmp_path):
        """Species pseudo map reads tagged labels."""
        xml = tmp_path / "data-file-schema.xml"
        xml.write_text(_AFM_XML)
        assert load_qe_species_pseudo(str(xml)) == {"O": "O.ccECP.upf", "O1": "O.ccECP.upf"}

    @pytest.mark.light
    @requires_o_upf
    def test_resolve_atom_elements_via_upf(self, tmp_path):
        """Resolve atom elements via upf."""
        save = tmp_path / "afm.save"
        save.mkdir()
        (save / "data-file-schema.xml").write_text(_AFM_XML)
        shutil.copy(_O_UPF, save / "O.ccECP.upf")
        # Both the plain 'O' and the tagged 'O1' resolve to the element 'O'.
        elems = _resolve_atom_elements(("O", "O1", "O"), str(save / "data-file-schema.xml"), str(save))
        assert elems == ("O", "O", "O")

    @pytest.mark.light
    def test_resolve_atom_elements_missing_species_raises(self, tmp_path):
        """Resolve atom elements missing species raises."""
        xml = tmp_path / "data-file-schema.xml"
        xml.write_text("<root></root>")
        with pytest.raises(ValueError, match="no <pseudo_file>"):
            _resolve_atom_elements(("O1",), str(xml), str(tmp_path))

    @pytest.mark.light
    def test_resolve_atom_elements_missing_upf_file_raises(self, tmp_path):
        """Resolve atom elements missing upf file raises."""
        xml = tmp_path / "data-file-schema.xml"
        xml.write_text(_AFM_XML)  # points at O.ccECP.upf, which is absent here
        with pytest.raises(FileNotFoundError):
            _resolve_atom_elements(("O1",), str(xml), str(tmp_path))


# Committed non-Gamma QE save (bcc Li, k = (1/4, 1/4, 1/4), npol = 1,
# noncolin/spinorbit = false). The noncollinear tests below derive spinor
# variants from a tmp_path copy; the committed files are never modified.
_LI_K025_SAVE = qe_example_files_root() / "Li_k025" / "out" / "li_k025.save"


def _copy_li_k025(tmp_path: Path) -> Path:
    save = tmp_path / "li_k025.save"
    shutil.copytree(_LI_K025_SAVE, save)
    return save


def _widen_wfc_to_spinor(save: Path, npol: int | None) -> int:
    """Rewrite ``wfc1.hdf5`` in QE's noncollinear layout and return igwx.

    A spinor band stores ``npol * igwx`` complex PW columns, the up block
    followed by the down block. ``npol=None`` removes the attribute so only the
    column-count consistency guard can reject the file.
    """
    import h5py

    with h5py.File(save / "wfc1.hdf5", "r+") as f:
        evc = f["evc"][...]
        igwx = int(f["MillerIndices"].shape[0])
        assert evc.shape[1] == 2 * igwx  # npol = 1 layout of the committed fixture
        assert int(np.asarray(f.attrs["npol"]).reshape(-1)[0]) == 1
        spinor = np.concatenate([evc, evc], axis=1) / np.sqrt(2.0)
        del f["evc"]
        f["evc"] = spinor
        if npol is None:
            del f.attrs["npol"]
        else:
            f.attrs["npol"] = np.int32(npol)
    return igwx


def _set_xml_flag_true(save: Path, tag: str) -> None:
    xml = save / "data-file-schema.xml"
    text = xml.read_text()
    assert f"<{tag}>false</{tag}>" in text
    xml.write_text(text.replace(f"<{tag}>false</{tag}>", f"<{tag}>true</{tag}>"))


def _run_qe2jqmc(save: Path, out: Path, kind: str, twist_average: bool = False) -> None:
    from jqmc.jqmc_qe import run_qe2jqmc

    qe_params = {
        "qe_save_dir": str(save),
        "kind": kind,
        "upf_paths": [],
        "twist_average": twist_average,
        "tiling": [],
        "supertwist": None,
    }
    jastrow_params = {"j1_type": "exp", "j2_type": "pade", "j3_basis_type": "none", "jastrow_dtn": "none"}
    run_qe2jqmc(qe_params=qe_params, jastrow_params=jastrow_params, hamiltonian_h5=str(out))


class TestNoncollinearRejection:
    """Noncollinear / spin-orbit (npol = 2) QE saves are outside the supported
    spin-restricted / LSDA domain and must be rejected at load time, before
    any projection keeps only the first spinor component (issue 534).
    """

    @pytest.mark.light
    @pytest.mark.parametrize(
        "kind,twist_average",
        [("pw", False), ("bspline", False), ("pgto", False), ("pgto+pw", False), ("pw", True)],
    )
    def test_npol2_spinor_wfc_rejected_for_every_kind(self, tmp_path, kind, twist_average):
        """An npol=2 spinor wfc file is rejected naming npol for every kind and the twist-average route."""
        save = _copy_li_k025(tmp_path)
        _widen_wfc_to_spinor(save, npol=2)
        out = tmp_path / "hamiltonian_data.h5"
        with pytest.raises(ValueError, match=r"npol=2 \(noncollinear"):
            _run_qe2jqmc(save, out, kind, twist_average=twist_average)
        assert not out.exists()

    @pytest.mark.light
    def test_load_qe_wfc_rejects_spinor_and_column_mismatch(self, tmp_path):
        """load_qe_wfc rejects npol=2 and any evc width inconsistent with MillerIndices."""
        save = _copy_li_k025(tmp_path)
        _widen_wfc_to_spinor(save, npol=2)
        with pytest.raises(ValueError, match=r"npol=2 \(noncollinear"):
            load_qe_wfc(str(save / "wfc1.hdf5"))
        # Without the npol attribute the per-band column count alone
        # (2 * igwx complex columns vs igwx Miller indices) must reject it.
        _widen_wfc_to_spinor(_copy_li_k025(tmp_path / "no_npol"), npol=None)
        with pytest.raises(ValueError, match="MillerIndices"):
            load_qe_wfc(str(tmp_path / "no_npol" / "li_k025.save" / "wfc1.hdf5"))

    @pytest.mark.light
    @pytest.mark.parametrize("flag", ["noncolin", "spinorbit"])
    def test_noncollinear_xml_flag_rejected(self, tmp_path, flag):
        """A save whose XML marks a noncollinear / spin-orbit run is rejected even with an npol=1 wfc file."""
        save = _copy_li_k025(tmp_path)
        _set_xml_flag_true(save, flag)
        with pytest.raises(ValueError, match="noncollinear / spin-orbit save"):
            load_qe_reference(str(save))
        out = tmp_path / "hamiltonian_data.h5"
        with pytest.raises(ValueError, match="noncollinear / spin-orbit save"):
            _run_qe2jqmc(save, out, "pw")
        assert not out.exists()

    @pytest.mark.light
    def test_unmodified_save_copy_still_converts(self, tmp_path):
        """Control: the unmodified npol=1 copy loads with one column pair per Miller index and converts."""
        import h5py

        save = _copy_li_k025(tmp_path)
        with h5py.File(save / "wfc1.hdf5", "r") as f:
            igwx = int(f["MillerIndices"].shape[0])
            n_bands = int(f["evc"].shape[0])
        wfc = load_qe_wfc(str(save / "wfc1.hdf5"))
        assert wfc.gamma_only is False
        assert wfc.coeffs.shape == (n_bands, igwx)
        out = tmp_path / "hamiltonian_data.h5"
        _run_qe2jqmc(save, out, "pw")
        assert out.is_file()
