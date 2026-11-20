"""LSDA (spin-dependent) TREXIO spin-channel routing.

Every spin treatment pairs the spin channels at opposite twists: the up
determinant block is the stored alpha channel at ``+k`` and the dn block is
the complex conjugate of the dn channel's OWN orbitals at ``+k``, i.e.
``conj(C)`` on the ``-k`` AO leaf. For the spin-dependent (UHF / kUHF /
LSDA) loader that channel is the stored beta channel, so the dn block is
``conj(C_beta)``; for the restricted loader it is ``conj(C_alpha)``.

Fixture: a synthetic spin-dependent multi-k TREXIO file derived from the
committed restricted ``lithium_twist_222.h5`` (8-twist shifted MP mesh, all
twists genuinely complex): the mo group is doubled into an alpha block
(``mo_spin = 0``, the original rows) and a DISTINCT beta block
(``mo_spin = 1``, the original rows with an AO-diagonal phase twist that no
per-band gauge factor can undo). The Li fixture cell carries 2 valence
electrons, so charge neutrality fixes ``N_up = N_dn = 1``; the spin-dependent
routing keys on ``mo_spin`` alone, and the two channels occupy DIFFERENT
bands (alpha band 0, beta band 1 -- a broken-symmetry UHF layout), so a
dn block built from the alpha channel, from the unconjugated beta rows or
on the ``+k`` leaf is detected.

The companion restricted test pins ``mo_dn = conj(mo_up)`` on the ``-k``
leaf on the original fixture.
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

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np

import pytest

from jqmc._setting import (
    ATOL_NONVACUITY_DISTINCTION,
    GEMINAL_REPRESENTATION_ATOL,
    RTOL_BIT_EXACT,
    RTOL_NONVACUITY_DISTINCTION,
)

pytestmark = [pytest.mark.with_chdir]

_FIXTURE = Path(__file__).parent.parent / "trexio_example_files" / "lithium_twist_222.h5"
_available = _FIXTURE.is_file()
requires_multitwist = pytest.mark.skipif(not _available, reason="lithium_twist_222.h5 TREXIO multi-k fixture not present")

# AO-diagonal phase step for the synthetic beta block. A per-AO-column phase
# changes the orbital as a function of r (it is NOT a per-band gauge), so the
# beta rows are genuinely distinct from the alpha rows and from their own
# conjugates, while remaining exactly reconstructable for the assertions.
_AO_PHASE_STEP = 0.3

# Overlap thresholds for the phase-insensitive row comparison.
_MATCH = 1.0 - 1.0e-9
_CONJ_MARGIN = 0.01


def _row_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """|<a|b>| / (|a||b|) -- 1.0 iff a == b up to a global phase."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _build_lsda_fixture(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """Derive the spin-dependent multi-k TREXIO file from the restricted one.

    Returns ``(path, C_alpha, C_beta, mo_k_point)`` where ``C_alpha`` /
    ``C_beta`` are the full (n_rows, n_ao) complex per-spin MO blocks and
    ``mo_k_point`` is the per-row k index shared by both blocks. Occupations:
    per k, alpha occupies band 0 and beta occupies band 1 (``N_up = N_dn = 1``
    -- the Li fixture cell has 2 valence electrons, so neutrality forbids a
    spin-imbalanced count; ``spin_dependent`` routing keys on ``mo_spin``).
    """
    dst = tmp_path / "lithium_twist_222_lsda.h5"
    shutil.copy(_FIXTURE, dst)
    with h5py.File(dst, "r+") as f:
        mo = f["mo"]
        C = mo["mo_coefficient"][...] + 1j * mo["mo_coefficient_im"][...]
        kidx = mo["mo_k_point"][...]
        energy = mo["mo_energy"][...]
        n_rows, n_ao = C.shape
        phase = np.exp(1j * _AO_PHASE_STEP * np.arange(n_ao))
        C_beta = C * phase[None, :]

        occ_up = np.zeros(n_rows, dtype=np.float64)
        occ_dn = np.zeros(n_rows, dtype=np.float64)
        for ik in sorted(set(int(x) for x in kidx)):
            rows = np.flatnonzero(kidx == ik)
            occ_up[rows[0]] = 1.0
            occ_dn[rows[1]] = 1.0

        C_all = np.concatenate([C, C_beta], axis=0)
        new_data = {
            "mo_coefficient": np.ascontiguousarray(C_all.real),
            "mo_coefficient_im": np.ascontiguousarray(C_all.imag),
            "mo_energy": np.concatenate([energy, energy]),
            "mo_k_point": np.concatenate([kidx, kidx]),
            "mo_occupation": np.concatenate([occ_up, occ_dn]),
            "mo_spin": np.concatenate([np.zeros(n_rows, dtype=np.int64), np.ones(n_rows, dtype=np.int64)]),
        }
        for name, arr in new_data.items():
            del mo[name]
            mo.create_dataset(name, data=arr)
        mo.attrs["mo_num"] = np.int64(2 * n_rows)
        mo.attrs["mo_type"] = np.bytes_(b"UHF")
    return dst, C, C_beta, np.asarray(kidx)


class TestTREXIOLsdaSpinChannels:
    @pytest.mark.heavy
    @requires_multitwist
    def test_lsda_dn_channel_is_conjugated_beta_at_minus_k(self, tmp_path):
        """Spin-dependent load: the dn block is conj(stored beta) on the -k partner leaf.

        The comparison is per-row phase-insensitive (the loader gauge-fixes
        each band) but conj-sensitive: the beta rows are genuinely complex,
        so ``c`` and ``conj(c)`` are distinguishable, and the beta band
        differs from the alpha band, so ``conj(C_alpha)`` cannot pass. The
        dn leaf must be the ``with_negated_k`` partner of the up leaf (twist
        ``-k`` and conjugated image phases), not a relabelled ``+k`` leaf.
        """
        from jqmc.trexio_wrapper import read_trexio_file

        path, C_alpha, C_beta, kidx = _build_lsda_fixture(tmp_path)
        for ik in (1, 7):
            _, _, _, _, gem, _ = read_trexio_file(str(path), store_tuple=True, twist_ik=ik)
            assert bool(gem.time_reversal) is False
            rows = np.flatnonzero(kidx == ik)
            up_leaf = gem.orb_data_up_spin.aos_data
            dn_leaf = gem.orb_data_dn_spin.aos_data

            # up channel: stored alpha band 0 at +k, no conj.
            mo_up = np.asarray(gem.orb_data_up_spin.mo_coefficients)
            assert mo_up.shape[0] == 1
            assert _row_overlap(mo_up[0], C_alpha[rows[0]]) > _MATCH, f"ik={ik}: up row does not match stored alpha band 0"

            # dn channel: conj(stored beta band 1), never alpha.
            mo_dn = np.asarray(gem.orb_data_dn_spin.mo_coefficients)
            assert mo_dn.shape[0] == 1
            ref_dn = C_beta[rows[1]]
            ov_conj = _row_overlap(mo_dn[0], np.conj(ref_dn))
            ov_plain = _row_overlap(mo_dn[0], ref_dn)
            ov_alpha = _row_overlap(mo_dn[0], np.conj(C_alpha[rows[1]]))
            assert ov_conj > _MATCH, (
                f"ik={ik}: dn row is not conj(stored beta) (|<dn|conj(beta)>|={ov_conj:.6f}, |<dn|beta>|={ov_plain:.6f})"
            )
            assert ov_conj > ov_plain + _CONJ_MARGIN
            assert ov_conj > ov_alpha + _CONJ_MARGIN

            # dn AO leaf: the with_negated_k partner of the up leaf.
            kp_up = np.asarray(up_leaf.k_point, dtype=np.float64)
            np.testing.assert_array_equal(np.asarray(dn_leaf.k_point, dtype=np.float64), -kp_up)
            partner = up_leaf.with_negated_k()
            np.testing.assert_allclose(
                np.asarray(dn_leaf._image_phase_factors),
                np.asarray(partner._image_phase_factors),
                atol=GEMINAL_REPRESENTATION_ATOL,
                rtol=RTOL_BIT_EXACT,
            )
            assert not np.allclose(
                np.asarray(dn_leaf._image_phase_factors),
                np.asarray(up_leaf._image_phase_factors),
                atol=ATOL_NONVACUITY_DISTINCTION,
                rtol=RTOL_NONVACUITY_DISTINCTION,
            )

    @pytest.mark.heavy
    @requires_multitwist
    def test_restricted_dn_channel_keeps_convention_b(self):
        """Companion pin: the RESTRICTED path keeps ``mo_dn == conj(mo_up)``
        byte-identically on the ``-k`` AO leaf with ``time_reversal=True``."""
        from jqmc.trexio_wrapper import read_trexio_file

        for ik in (1, 7):
            _, _, _, _, gem, _ = read_trexio_file(str(_FIXTURE), store_tuple=True, twist_ik=ik)
            assert bool(gem.time_reversal) is True
            mo_up = np.asarray(gem.orb_data_up_spin.mo_coefficients)
            mo_dn = np.asarray(gem.orb_data_dn_spin.mo_coefficients)
            np.testing.assert_array_equal(mo_dn, np.conj(mo_up))
            kp_up = np.asarray(gem.orb_data_up_spin.aos_data.k_point, dtype=np.float64)
            kp_dn = np.asarray(gem.orb_data_dn_spin.aos_data.k_point, dtype=np.float64)
            np.testing.assert_array_equal(kp_dn, -kp_up)

    @pytest.mark.heavy
    @requires_multitwist
    def test_twist_converter_lsda_dn_payload_is_conjugated_beta(self, tmp_path):
        """``convert_trexio_to_twist_average_input`` persists conj(beta) and ``load_twist`` puts it on -k.

        The combined ``mo_coefficients_dn`` payload is the conjugate of the stored
        beta orbitals, and ``load_twist`` rebuilds it unchanged on a ``-k``
        dn leaf for the ``time_reversal=False`` geminal.
        """
        from jqmc._twist_average import load_twist
        from jqmc.trexio_wrapper import convert_trexio_to_twist_average_input

        path, _C_alpha, C_beta, kidx = _build_lsda_fixture(tmp_path)
        out = tmp_path / "hamiltonian_data_lsda.h5"
        convert_trexio_to_twist_average_input(str(path), str(out))

        for ik in (1, 7):
            rows = np.flatnonzero(kidx == ik)
            ref_dn = np.conj(C_beta[rows[1]])
            with h5py.File(out, "r") as f:
                stored_dn = np.asarray(f["twists"][f"twist_{ik:03d}"]["mo_coefficients_dn"])
            assert stored_dn.shape[0] == 1
            ov_conj = _row_overlap(stored_dn[0], ref_dn)
            ov_plain = _row_overlap(stored_dn[0], np.conj(ref_dn))
            assert ov_conj > _MATCH, (
                f"twist {ik}: combined mo_coefficients_dn is not conj(stored beta) "
                f"(|<dn|conj(beta)>|={ov_conj:.6f}, |<dn|beta>|={ov_plain:.6f})."
            )
            assert ov_conj > ov_plain + _CONJ_MARGIN

            ham = load_twist(str(out), ik)
            gem = ham.wavefunction_data.geminal_data
            assert bool(gem.time_reversal) is False
            kp_up = np.asarray(gem.orb_data_up_spin.aos_data.k_point, dtype=np.float64)
            kp_dn = np.asarray(gem.orb_data_dn_spin.aos_data.k_point, dtype=np.float64)
            np.testing.assert_array_equal(kp_dn, -kp_up)
            mo_dn_loaded = np.asarray(gem.orb_data_dn_spin.mo_coefficients)
            assert _row_overlap(mo_dn_loaded[0], ref_dn) > _MATCH
