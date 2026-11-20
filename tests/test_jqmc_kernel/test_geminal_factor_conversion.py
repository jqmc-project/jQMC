"""C2 regressions for complex AGP-to-factor conversions."""

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

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, ATOL_PYTHON_FP64_TIGHT, RTOL_PYTHON_FP64_LOOSE
from jqmc.determinant import AGP_data, AGPn_data, _smetric_polar_retract, to_AGP, to_AGPn
from jqmc.molecular_orbital import MOs_data
from jqmc.trexio_wrapper import read_trexio_file
from tests.test_jqmc_kernel.optimizer_metadata_helpers import make_test_block
from tests.test_jqmc_kernel.test_determinant import _build_minimal_mos_data, _build_small_sphe_aos_for_conversion


def _hermitian_pair(rng: np.random.Generator, n: int) -> np.ndarray:
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return A.conj().T @ A + np.eye(n)


def _sqrt_pair(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(S)
    sqrt = (vectors * np.sqrt(values)) @ vectors.conj().T
    inv_sqrt = (vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T
    return sqrt, inv_sqrt


def _complex_aos_pair():
    geminal = read_trexio_file(
        trexio_file=str(Path(__file__).resolve().parent.parent / "trexio_example_files" / "H2_pbc_ae_k025_sphe.h5"),
        store_tuple=False,
    )[4]
    return geminal.orb_data_up_spin.aos_data, geminal.orb_data_dn_spin.aos_data


def _complex_mos_pair(rng: np.random.Generator, n: int = 3) -> tuple[MOs_data, MOs_data]:
    aos_up, aos_dn = _complex_aos_pair()
    C_up = rng.standard_normal((n, aos_up.num_ao)) + 1j * rng.standard_normal((n, aos_up.num_ao))
    C_dn = rng.standard_normal((n, aos_dn.num_ao)) + 1j * rng.standard_normal((n, aos_dn.num_ao))
    C_up = _smetric_polar_retract(C_up, aos_up)
    C_dn = _smetric_polar_retract(C_dn, aos_dn)
    return (
        MOs_data(num_mo=n, aos_data=aos_up, mo_coefficients=C_up),
        MOs_data(num_mo=n, aos_data=aos_dn, mo_coefficients=C_dn),
    )


def _represented_ao(geminal) -> np.ndarray:
    return np.asarray(AGP_data.convert_from_MOs_to_AOs(to_AGP(geminal)).lambda_matrix)


class TestAOToMOFactorConversion:
    @pytest.mark.parametrize("num_eigenvectors", ["all", 4])
    def test_independent_complex_bilinear_round_trip_with_unpaired(self, monkeypatch, num_eigenvectors):
        """Independent complex bilinear round trip with unpaired."""
        import jqmc.determinant as determinant_module

        rng = np.random.default_rng(2026072601)
        aos_up = _build_minimal_mos_data(num_mo=1, num_ao=4).aos_data
        aos_dn = _build_minimal_mos_data(num_mo=1, num_ao=4).aos_data
        S_up = _hermitian_pair(rng, 4)
        S_dn = _hermitian_pair(rng, 4)
        paired = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        unpaired = rng.standard_normal((4, 1)) + 1j * rng.standard_normal((4, 1))
        source = AGP_data(
            num_electron_up=3,
            num_electron_dn=2,
            orb_data_up_spin=aos_up,
            orb_data_dn_spin=aos_dn,
            lambda_matrix=np.hstack([paired, unpaired]),
            time_reversal=False,
        )

        monkeypatch.setattr(
            determinant_module,
            "compute_overlap_matrix",
            lambda aos: S_up if aos is aos_up else S_dn,
        )
        converted = AGP_data.convert_from_AOs_to_MOs(source, num_eigenvectors=num_eigenvectors)
        round_trip = AGP_data.convert_from_MOs_to_AOs(converted)

        np.testing.assert_allclose(
            round_trip.lambda_matrix, source.lambda_matrix, atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_LOOSE
        )
        C_up = np.asarray(converted.orb_data_up_spin.mo_coefficients)
        C_dn = np.asarray(converted.orb_data_dn_spin.mo_coefficients)
        np.testing.assert_allclose(
            C_up.conj() @ S_up @ C_up.T, np.eye(4), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE
        )
        np.testing.assert_allclose(
            C_dn.conj() @ S_dn @ C_dn.T, np.eye(4), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE
        )

    def test_time_reversal_conversion_retains_signed_occupations(self, monkeypatch):
        """Time reversal conversion retains signed occupations."""
        import jqmc.determinant as determinant_module

        rng = np.random.default_rng(2026072602)
        aos_up = _build_minimal_mos_data(num_mo=1, num_ao=4).aos_data
        aos_dn = _build_minimal_mos_data(num_mo=1, num_ao=4).aos_data
        S_up = _hermitian_pair(rng, 4)
        _, inv_sqrt = _sqrt_pair(S_up)
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))
        expected = np.array([4.0, -3.0, 1.5, -0.25])
        whitened = Q @ np.diag(expected) @ Q.conj().T
        paired = inv_sqrt @ whitened @ inv_sqrt
        source = AGP_data(
            num_electron_up=2,
            num_electron_dn=2,
            orb_data_up_spin=aos_up,
            orb_data_dn_spin=aos_dn,
            lambda_matrix=paired,
            time_reversal=True,
        )

        monkeypatch.setattr(
            determinant_module,
            "compute_overlap_matrix",
            lambda aos: S_up if aos is aos_up else S_up.conj(),
        )
        converted = AGP_data.convert_from_AOs_to_MOs(source, num_eigenvectors="all")
        occupations = np.diag(np.asarray(converted.lambda_matrix))

        np.testing.assert_allclose(occupations, expected, atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(
            converted.orb_data_dn_spin.mo_coefficients,
            np.conj(converted.orb_data_up_spin.mo_coefficients),
            atol=ATOL_PYTHON_FP64_TIGHT,
            rtol=RTOL_PYTHON_FP64_LOOSE,
        )
        np.testing.assert_allclose(
            AGP_data.convert_from_MOs_to_AOs(converted).lambda_matrix,
            source.lambda_matrix,
            atol=ATOL_PYTHON_FP64_TIGHT,
            rtol=RTOL_PYTHON_FP64_LOOSE,
        )

    @pytest.mark.parametrize("bad_kind", ["overlap", "core"])
    def test_declared_time_reversal_preconditions_are_hard_errors(self, monkeypatch, bad_kind):
        """Declared time reversal preconditions are hard errors."""
        import jqmc.determinant as determinant_module

        rng = np.random.default_rng(2026072603)
        aos_up = _build_minimal_mos_data(num_mo=1, num_ao=3).aos_data
        aos_dn = _build_minimal_mos_data(num_mo=1, num_ao=3).aos_data
        S_up = _hermitian_pair(rng, 3)
        S_dn = S_up.conj() if bad_kind == "core" else _hermitian_pair(rng, 3)
        core = _hermitian_pair(rng, 3)
        if bad_kind == "core":
            core[0, 1] += 0.75j
        source = AGP_data(
            num_electron_up=1,
            num_electron_dn=1,
            orb_data_up_spin=aos_up,
            orb_data_dn_spin=aos_dn,
            lambda_matrix=core,
            time_reversal=True,
        )
        monkeypatch.setattr(
            determinant_module,
            "compute_overlap_matrix",
            lambda aos: S_up if aos is aos_up else S_dn,
        )

        with pytest.raises(ValueError, match=r"time.reversal|Hermitian|overlap"):
            AGP_data.convert_from_AOs_to_MOs(source, num_eigenvectors="all")

    def test_truncated_open_shell_retains_spectrum_and_orthogonal_unpaired_rows(self, monkeypatch):
        """Truncated open shell retains spectrum and orthogonal unpaired rows."""
        import jqmc.determinant as determinant_module

        aos_up = _build_minimal_mos_data(num_mo=1, num_ao=5).aos_data
        aos_dn = _build_minimal_mos_data(num_mo=1, num_ao=5).aos_data
        paired = np.diag([5.0, 4.0, 3.0, 2.0, 1.0])
        unpaired = np.array([[0.0], [0.0], [0.0], [1.0], [1.0]])
        source = AGP_data(
            num_electron_up=3,
            num_electron_dn=2,
            orb_data_up_spin=aos_up,
            orb_data_dn_spin=aos_dn,
            lambda_matrix=np.hstack([paired, unpaired]),
            time_reversal=False,
        )
        monkeypatch.setattr(determinant_module, "compute_overlap_matrix", lambda _aos: np.eye(5))

        converted = AGP_data.convert_from_AOs_to_MOs(source, num_eigenvectors=4)
        C_up = np.asarray(converted.orb_data_up_spin.mo_coefficients)
        occupations = np.diag(np.asarray(converted.lambda_matrix)[:, :4])

        np.testing.assert_allclose(C_up.conj() @ C_up.T, np.eye(4), atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(
            occupations, np.array([5.0, 4.0, 3.0, 0.0]), atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_LOOSE
        )

    def test_ao_factorization_rejects_ambiguous_magnitude_cutoff(self, monkeypatch):
        """AO factorization rejects ambiguous magnitude cutoff."""
        import jqmc.determinant as determinant_module

        aos_up = _build_minimal_mos_data(num_mo=1, num_ao=4).aos_data
        aos_dn = _build_minimal_mos_data(num_mo=1, num_ao=4).aos_data
        source = AGP_data(
            num_electron_up=1,
            num_electron_dn=1,
            orb_data_up_spin=aos_up,
            orb_data_dn_spin=aos_dn,
            lambda_matrix=np.diag([3.0, 2.0, -2.0, 1.0]),
            time_reversal=True,
        )
        monkeypatch.setattr(determinant_module, "compute_overlap_matrix", lambda _aos: np.eye(4))

        with pytest.raises(ValueError, match="ambiguous truncation"):
            AGP_data.convert_from_AOs_to_MOs(source, num_eigenvectors=2)

    def test_time_reversal_open_shell_truncation_preserves_unpaired_column(self, monkeypatch):
        """Time reversal open shell truncation preserves unpaired column."""
        import jqmc.determinant as determinant_module

        rng = np.random.default_rng(2026072608)
        aos_up = _build_minimal_mos_data(num_mo=1, num_ao=5).aos_data
        aos_dn = _build_minimal_mos_data(num_mo=1, num_ao=5).aos_data
        S_up = _hermitian_pair(rng, 5)
        _, inv_sqrt = _sqrt_pair(S_up)
        Q, _ = np.linalg.qr(rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5)))
        signed = np.array([5.0, -4.0, 3.0, -2.0, 1.0])
        paired = inv_sqrt @ (Q @ np.diag(signed) @ Q.conj().T) @ inv_sqrt
        unpaired = inv_sqrt @ (Q[:, [3]] + Q[:, [4]])
        source = AGP_data(
            num_electron_up=3,
            num_electron_dn=2,
            orb_data_up_spin=aos_up,
            orb_data_dn_spin=aos_dn,
            lambda_matrix=np.hstack([paired, unpaired]),
            time_reversal=True,
        )
        monkeypatch.setattr(
            determinant_module,
            "compute_overlap_matrix",
            lambda aos: S_up if aos is aos_up else S_up.conj(),
        )

        converted = AGP_data.convert_from_AOs_to_MOs(source, num_eigenvectors=4)
        C_up = np.asarray(converted.orb_data_up_spin.mo_coefficients)
        C_dn = np.asarray(converted.orb_data_dn_spin.mo_coefficients)
        occupations = np.diag(np.asarray(converted.lambda_matrix)[:, :4])
        round_trip = AGP_data.convert_from_MOs_to_AOs(converted)

        np.testing.assert_allclose(
            occupations, np.array([5.0, -4.0, 3.0, 0.0]), atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_LOOSE
        )
        np.testing.assert_allclose(C_dn, np.conj(C_up), atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(
            C_up.conj() @ S_up @ C_up.T, np.eye(4), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE
        )
        np.testing.assert_allclose(
            round_trip.lambda_matrix[:, -1], unpaired[:, 0], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE
        )


class TestMOToAGPnFactorConversion:
    def test_independent_complex_svd_absorption_preserves_dense_geminal(self):
        """Independent complex svd absorption preserves dense geminal."""
        rng = np.random.default_rng(2026072604)
        mos_up, mos_dn = _complex_mos_pair(rng)
        paired = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        source = AGP_data(
            num_electron_up=1,
            num_electron_dn=1,
            orb_data_up_spin=mos_up,
            orb_data_dn_spin=mos_dn,
            lambda_matrix=paired,
            time_reversal=False,
        )

        expected = _represented_ao(source)
        converted = to_AGPn(source)

        np.testing.assert_allclose(
            _represented_ao(converted), expected, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE
        )

    def test_time_reversal_mo_conversion_and_first_update_keep_negative_occupation(self):
        """Time reversal MO conversion and first update keep negative occupation."""
        rng = np.random.default_rng(2026072605)
        mos_up, _ = _complex_mos_pair(rng)
        mos_dn = MOs_data(
            num_mo=mos_up.num_mo,
            aos_data=_complex_aos_pair()[1],
            mo_coefficients=np.conj(mos_up.mo_coefficients),
        )
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
        expected_occupations = np.array([3.0, -2.0, 0.5])
        paired = Q @ np.diag(expected_occupations) @ Q.conj().T
        source = AGP_data(
            num_electron_up=1,
            num_electron_dn=1,
            orb_data_up_spin=mos_up,
            orb_data_dn_spin=mos_dn,
            lambda_matrix=paired,
            time_reversal=True,
        )
        expected = _represented_ao(source)

        converted = to_AGPn(source)
        np.testing.assert_allclose(
            converted.lambda_diag, expected_occupations, atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_LOOSE
        )
        np.testing.assert_allclose(
            _represented_ao(converted), expected, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE
        )

        block = make_test_block(
            name="mo_coefficients_up",
            values=np.asarray(converted.orb_data_up_spin.mo_coefficients),
            field="complex",
            derivative_policy="wirtinger_pair",
            sync_scope="sampler",
        )
        updated = converted.apply_block_update(block)
        np.testing.assert_array_equal(updated.lambda_diag, converted.lambda_diag)
        assert np.any(np.asarray(updated.lambda_diag) < 0.0)

    def test_real_gamma_indefinite_core_and_first_update_keep_signed_occupations(self):
        """Real gamma indefinite core and first update keep signed occupations."""
        rng = np.random.default_rng(2026072607)
        aos = _build_small_sphe_aos_for_conversion()
        C = _smetric_polar_retract(rng.standard_normal((4, aos.num_ao)), aos)
        mos = MOs_data(num_mo=4, aos_data=aos, mo_coefficients=C)
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        expected_occupations = np.array([4.0, -2.5, 1.0, -0.25])
        paired = Q @ np.diag(expected_occupations) @ Q.T
        source = AGP_data(
            num_electron_up=1,
            num_electron_dn=1,
            orb_data_up_spin=mos,
            orb_data_dn_spin=mos,
            lambda_matrix=paired,
            time_reversal=True,
        )
        expected = _represented_ao(source)

        converted = to_AGPn(source)
        np.testing.assert_allclose(
            converted.lambda_diag, expected_occupations, atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_LOOSE
        )
        np.testing.assert_allclose(
            _represented_ao(converted), expected, atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_LOOSE
        )

        block = make_test_block(
            name="mo_coefficients_up",
            values=np.asarray(converted.orb_data_up_spin.mo_coefficients),
            field="real",
            derivative_policy="direct_real",
            sync_scope="sampler",
        )
        updated = converted.apply_block_update(block)
        np.testing.assert_array_equal(updated.lambda_diag, converted.lambda_diag)
        assert np.any(np.asarray(updated.lambda_diag) < 0.0)

        # A signed ``lambda_diag`` step is stored verbatim (no sign folding);
        # numpy reference for the represented AO geminal: C^T diag(lambda) C.
        stepped_occupations = np.array([1.1, -0.6, 0.35, -0.2])
        lambda_block = make_test_block(
            name="lambda_diag",
            values=stepped_occupations,
            field="real",
            derivative_policy="direct_real",
            sync_scope="sampler",
        )
        stepped = updated.apply_block_update(lambda_block)
        np.testing.assert_array_equal(stepped.lambda_diag, stepped_occupations)
        C_stepped = np.asarray(stepped.orb_data_up_spin.mo_coefficients)
        np.testing.assert_allclose(
            _represented_ao(stepped),
            C_stepped.T @ np.diag(stepped_occupations) @ C_stepped,
            atol=ATOL_PYTHON_FP64_TIGHT,
            rtol=RTOL_PYTHON_FP64_LOOSE,
        )

    def test_declared_time_reversal_rejects_mismatched_down_factor(self):
        """Declared time reversal rejects mismatched down factor."""
        rng = np.random.default_rng(2026072606)
        mos_up, mos_dn = _complex_mos_pair(rng)
        source = AGP_data(
            num_electron_up=1,
            num_electron_dn=1,
            orb_data_up_spin=mos_up,
            orb_data_dn_spin=mos_dn,
            lambda_matrix=np.eye(3, dtype=np.complex128),
            time_reversal=True,
        )

        with pytest.raises(ValueError, match=r"time.reversal|conj"):
            to_AGPn(source)

    @pytest.mark.parametrize(
        ("occupations", "unpaired", "target", "message"),
        [
            (np.array([1.0, 0.2, 0.5]), None, 2, "sorted"),
            (np.array([1.0, 0.5, 0.5]), None, 2, "ambiguous"),
            (
                np.array([1.0, 0.4, 0.1]),
                np.array([[1.0], [0.0], [0.25]]),
                2,
                "unpaired",
            ),
        ],
    )
    def test_rank_truncation_preconditions(self, occupations, unpaired, target, message):
        """Rank truncation preconditions."""
        mos = _build_minimal_mos_data(num_mo=3, num_ao=4)
        geminal = AGPn_data.build(
            num_electron_up=2 if unpaired is not None else 1,
            num_electron_dn=1,
            mos_up=mos,
            mos_dn=mos,
            lambda_diag=occupations,
            lambda_unpaired=unpaired,
            time_reversal=False,
        )

        with pytest.raises(ValueError, match=message):
            to_AGPn(geminal, n=target)

    def test_rank_below_electron_count_is_rejected(self):
        """Rank below electron count is rejected."""
        mos = _build_minimal_mos_data(num_mo=3, num_ao=4)
        source = AGP_data(
            num_electron_up=2,
            num_electron_dn=2,
            orb_data_up_spin=mos,
            orb_data_dn_spin=mos,
            lambda_matrix=np.diag([3.0, 2.0, 1.0]),
            time_reversal=False,
        )

        with pytest.raises(ValueError, match=r"rank|num_electron_up"):
            to_AGPn(source, n=1)
