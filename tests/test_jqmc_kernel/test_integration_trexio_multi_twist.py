"""TREXIO multi-twist (4x4x4 unreduced) end-to-end conversion tests.

Enumerate every k-point via TREXIO's
``pbc_k_point`` / ``pbc_k_point_weight`` slots, filter the unified
``mo_coefficient`` (+ ``mo_coefficient_im``) matrix by ``mo_k_point`` per
twist, build a per-twist complex Bloch geminal under Convention B, and
write a combined HDF5 input (``/_meta`` + ``/shared`` + ``/twists``).

Multi-k Gamma policy (this PR): the unfolded mesh's Gamma slice is
**not** collapsed to ``Structure_data.k_point = None``. A multi-k file
routes every twist (Gamma included) through the complex Bloch path so
the per-twist datasets are dtype-uniform and the combined storage stays
shape-uniform. Single-twist Gamma files keep the real-path
dispatch.

Fixture: ``lithium_twist_222.h5`` -- pyscf+TREXIO export of a 2x2x2
unreduced Monkhorst-Pack run on 2-atom Li (ccECP-soft). 8 twists,
uniform weight 1/8 (sum = 1.0 in the TREXIO normalisation), 26 MOs
per k, 1 valence electron per spin. The Cu metal fixture
(``cu_k_twist.save``) covers the QE half of the multi-k policy.
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

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, ATOL_PYTHON_FP64_TIGHT, RTOL_PYTHON_FP64_ZERO
from jqmc.atomic_orbital import AOs_sphe_pgto_complex_data

_FIXTURE = Path(__file__).resolve().parent.parent / "trexio_example_files" / "lithium_twist_222.h5"
_available = _FIXTURE.is_file()

requires_multitwist = pytest.mark.skipif(not _available, reason="lithium_twist_222.h5 TREXIO multi-k fixture not present")


# ---------- k-point enumeration ----------


class TestTREXIOMultiTwist:
    @pytest.mark.heavy
    @requires_multitwist
    def test_read_trexio_kpoint_list_enumerates_8_twists(self):
        """``read_trexio_kpoint_list`` returns the 8 k-points of the 2x2x2
        unreduced mesh; each weight is exactly 1/8 (TREXIO's per-k weight
        convention; the sum is 1.0 here, vs QE's paramagnetic sum-to-2
        convention)."""
        from jqmc.trexio_wrapper import read_trexio_kpoint_list

        kpts = read_trexio_kpoint_list(str(_FIXTURE))
        assert len(kpts) == 8
        weights = np.array([w for _, w in kpts])
        assert weights.sum() == pytest.approx(1.0, abs=ATOL_PYTHON_FP64_LOOSE)
        # Uniform unreduced grid -- every weight is 1/8 exactly.
        np.testing.assert_array_equal(weights, np.full(8, 0.125))
        # k_frac_tuple is a 3-tuple of Python floats per entry.
        for k, _ in kpts:
            assert isinstance(k, tuple) and len(k) == 3
            for x in k:
                assert isinstance(x, float)

    # ---------- per-twist occupied-row read ----------
    @pytest.mark.heavy
    @requires_multitwist
    def test_read_trexio_file_per_twist_returns_uniform_shape(self):
        """A per-k read returns exactly one occupied row at every twist."""
        from jqmc.trexio_wrapper import read_trexio_file

        shapes = set()
        for ik in [0, 1, 3, 5, 7]:
            sd, _, _, _, gem, _ = read_trexio_file(
                trexio_file=str(_FIXTURE),
                store_tuple=True,
                twist_ik=ik,
            )
            mo = np.asarray(gem.orb_data_up_spin.mo_coefficients)
            shapes.add(mo.shape)
            # Multi-k policy: every twist comes
            # back with ``aos_data.k_point`` as a tuple, not None (so the AO
            # dispatches to the complex Bloch leaf uniformly). The 2x2x2
            # shifted MP mesh in this fixture has no Gamma point, but the
            # tuple-not-None contract is universal. ``k_point`` lives on the
            # AO leaf now (Structure_data lost its ``k_point`` field).
            _aos = gem.orb_data_up_spin.aos_data
            assert _aos.k_point is not None
            assert isinstance(_aos.k_point, tuple)
            # Per-twist sanity_check passes with occupied complex MO rows,
            # no SD lambda, a complex Bloch AO leaf, and Convention B.
            gem.sanity_check()
            assert isinstance(_aos, AOs_sphe_pgto_complex_data)
        assert len(shapes) == 1, f"per-twist MO shapes diverged: {shapes}"
        assert shapes.pop() == (1, 26)

    @pytest.mark.heavy
    @requires_multitwist
    def test_read_trexio_file_per_twist_preserves_convention_b(self, tmp_path):
        """Convention B (``mo_dn = conj(mo_up)``) holds at every twist
        storage layer. For the 2x2x2 shifted MP mesh (no Gamma) all
        entries are non-trivially complex, so the conjugate relation is
        exercised at every probed twist.

        The loaded bands are also gauge-fixed: each Bloch band carries an
        arbitrary global phase from the DFT solver, and the loader
        normalises it so that the first non-negligible coefficient of every
        band is real-positive. The reference is the TREXIO file itself: a
        copy whose raw bands are multiplied by known unit phases
        ``exp(i*alpha_b)`` (the same physical state in another DFT gauge)
        must load to the same coefficients as the original.
        """
        import shutil

        import h5py

        from jqmc.trexio_wrapper import read_trexio_file

        rotated = tmp_path / "lithium_twist_222_rotated_gauge.h5"
        shutil.copy(_FIXTURE, rotated)
        with h5py.File(rotated, "r+") as h:
            c_raw = h["mo/mo_coefficient"][()] + 1j * h["mo/mo_coefficient_im"][()]
            alpha = 0.7 + 0.37 * np.arange(c_raw.shape[0])
            c_rot = c_raw * np.exp(1j * alpha)[:, None]
            h["mo/mo_coefficient"][...] = c_rot.real
            h["mo/mo_coefficient_im"][...] = c_rot.imag

        for ik in [0, 1, 3, 7]:
            _, _, _, _, gem, _ = read_trexio_file(
                trexio_file=str(_FIXTURE),
                store_tuple=True,
                twist_ik=ik,
            )
            mo_up = np.asarray(gem.orb_data_up_spin.mo_coefficients)
            mo_dn = np.asarray(gem.orb_data_dn_spin.mo_coefficients)
            np.testing.assert_array_equal(mo_dn, np.conj(mo_up))

            # Gauge fix: first non-negligible coefficient of each band is
            # real-positive.
            pivot = mo_up[np.arange(mo_up.shape[0]), np.argmax(np.abs(mo_up) > 1e-8, axis=1)]
            assert np.all(pivot.real > 0.0), f"ik={ik}: pivot not positive: {pivot}"
            np.testing.assert_allclose(pivot.imag, 0.0, rtol=RTOL_PYTHON_FP64_ZERO, atol=ATOL_PYTHON_FP64_TIGHT)

            # Gauge invariance: the rotated-phase copy loads identically.
            _, _, _, _, gem_rot, _ = read_trexio_file(
                trexio_file=str(rotated),
                store_tuple=True,
                twist_ik=ik,
            )
            np.testing.assert_allclose(
                np.asarray(gem_rot.orb_data_up_spin.mo_coefficients),
                mo_up,
                rtol=RTOL_PYTHON_FP64_ZERO,
                atol=ATOL_PYTHON_FP64_LOOSE,
            )

    # ---------- end-to-end convert + combined round-trip ----------
    @pytest.mark.heavy
    @requires_multitwist
    def test_convert_trexio_to_twist_average_input_full_round_trip(self, tmp_path):
        """End-to-end: ``convert_trexio_to_twist_average_input`` writes a combined
        combined HDF5 with all 8 twists; every twist round-trips via
        ``load_twist`` to a sanity_check-passing Hamiltonian with complex
        occupied complex MO rows, no SD lambda, and Convention B at the
        dataset level. The shape ``(1, 26)`` is uniform across twists."""
        from jqmc._twist_average import load_twist, load_twist_average_input
        from jqmc.trexio_wrapper import convert_trexio_to_twist_average_input

        out_path = str(tmp_path / "ha_twists.h5")
        convert_trexio_to_twist_average_input(
            trexio_file=str(_FIXTURE),
            output_path=out_path,
        )
        meta = load_twist_average_input(out_path)
        assert meta.n_twist == 8
        assert meta.bz_weight_sum == pytest.approx(1.0, abs=ATOL_PYTHON_FP64_LOOSE)

        # Spot-check the corner cases: Gamma slice, first non-Gamma slice,
        # mid-IBZ, and the last twist. Asserting on every twist is overkill
        # at this layer (the per-twist write loop is identical).
        expected_shape = None
        for ik in [0, 1, 3, 7]:
            ham_k = load_twist(out_path, ik)
            ham_k.sanity_check()
            # Multi-k contract: every twist keeps a tuple ``k_point`` (NOT
            # collapsed to ``None``) so the dtype-uniform Bloch path is
            # taken for all entries. The 2x2x2 shifted MP mesh in this
            # fixture has no Gamma point -- every ``k_point`` is at a
            # half-integer; we just assert non-None here. ``k_point`` lives
            # on the AO leaf (Structure_data lost its ``k_point`` field).
            _aos = ham_k.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
            assert _aos.k_point is not None
            assert isinstance(_aos.k_point, tuple)
            mo_up = np.asarray(ham_k.wavefunction_data.geminal_data.orb_data_up_spin.mo_coefficients)
            mo_dn = np.asarray(ham_k.wavefunction_data.geminal_data.orb_data_dn_spin.mo_coefficients)
            assert mo_up.dtype == np.complex128
            assert mo_dn.dtype == np.complex128
            assert not hasattr(ham_k.wavefunction_data.geminal_data, "lambda_matrix")
            np.testing.assert_array_equal(mo_dn, np.conj(mo_up))
            if expected_shape is None:
                expected_shape = mo_up.shape
            else:
                assert mo_up.shape == expected_shape
            assert mo_up.shape == (1, 26)

    # ---------- public CLI and occupation contract ----------
    @pytest.mark.light
    def test_cli_twist_average_matches_converter_and_keeps_jastrow(self, tmp_path):
        """``trexio convert-to --twist-average`` writes the converter's twists and the requested J2."""
        from jqmc._twist_average import load_twist, load_twist_average_input
        from jqmc.jqmc_tool import trexio_convert_to
        from jqmc.trexio_wrapper import convert_trexio_to_twist_average_input

        cli_path = str(tmp_path / "cli.h5")
        direct_path = str(tmp_path / "direct.h5")
        trexio_convert_to(
            trexio_file=str(_FIXTURE),
            hamiltonian_file=cli_path,
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type="none",
            twist_average=True,
        )
        convert_trexio_to_twist_average_input(str(_FIXTURE), direct_path)

        assert load_twist_average_input(cli_path).n_twist == 8
        for ik in [0, 7]:
            ham_cli = load_twist(cli_path, ik)
            ham_direct = load_twist(direct_path, ik)
            assert ham_cli.wavefunction_data.jastrow_data.jastrow_two_body_data is not None
            np.testing.assert_array_equal(
                np.asarray(ham_cli.wavefunction_data.geminal_data.orb_data_up_spin.mo_coefficients),
                np.asarray(ham_direct.wavefunction_data.geminal_data.orb_data_up_spin.mo_coefficients),
            )

    @pytest.mark.light
    def test_cli_twist_average_shares_one_phase_free_ao_j3_in_source_form(self, tmp_path):
        """``--twist-average -j3 ao-small`` keeps the spherical source form and one J3 for every twist."""
        from jqmc.atomic_orbital import AOs_sphe_pgto_data
        from jqmc.jqmc_tool import trexio_convert_to
        from jqmc.trexio_wrapper import read_trexio_file
        from tests.test_jqmc_kernel.test_integration_qe_multi_twist import assert_twists_share_one_phase_free_j3

        out_path = str(tmp_path / "j3_twists.h5")
        trexio_convert_to(
            trexio_file=str(_FIXTURE),
            hamiltonian_file=out_path,
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type="ao-small",
            twist_average=True,
        )
        j3 = assert_twists_share_one_phase_free_j3(out_path)
        source_aos = read_trexio_file(str(_FIXTURE), store_tuple=True, twist_ik=0)[1]
        assert isinstance(source_aos, AOs_sphe_pgto_complex_data)
        assert isinstance(j3.orb_data, AOs_sphe_pgto_data)
        assert set(j3.orb_data.magnetic_quantum_numbers) <= set(source_aos.magnetic_quantum_numbers)

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("j3_basis_type", "ao_conv_to", "message"),
        [("mo", None, "twist-dependent"), ("none", "cart", "--ao-conv-to")],
    )
    def test_cli_twist_average_rejects_unsupported_options(self, tmp_path, j3_basis_type, ao_conv_to, message):
        """MO J3 and AO conversion fail clearly and write no output, as on the QE twist route."""
        import click

        from jqmc.jqmc_tool import trexio_convert_to

        out_path = tmp_path / "rejected.h5"
        with pytest.raises(click.ClickException, match=message):
            trexio_convert_to(
                trexio_file=str(_FIXTURE),
                hamiltonian_file=str(out_path),
                j1_parmeter=None,
                j2_parmeter=1.0,
                j3_basis_type=j3_basis_type,
                ao_conv_to=ao_conv_to,
                twist_average=True,
            )
        assert not out_path.exists()

    @pytest.mark.light
    def test_converter_keeps_gamma_row_of_multi_k_file_complex(self, tmp_path):
        """A Gamma row of a multi-k file loads on the complex Bloch path, like the other twists."""
        import h5py

        from jqmc._twist_average import load_twist
        from jqmc.trexio_wrapper import convert_trexio_to_twist_average_input

        gamma_file = tmp_path / "lithium_twist_222_gamma.h5"
        shutil.copy(_FIXTURE, gamma_file)
        with h5py.File(gamma_file, "r+") as f:
            k_points = f["pbc"]["pbc_k_point"][...]
            k_points[0] = 0.0
            f["pbc"]["pbc_k_point"][...] = k_points

        out_path = str(tmp_path / "gamma.h5")
        convert_trexio_to_twist_average_input(str(gamma_file), out_path)
        ham_gamma = load_twist(out_path, 0)
        ham_gamma.sanity_check()
        orb_up = ham_gamma.wavefunction_data.geminal_data.orb_data_up_spin
        assert orb_up.aos_data.k_point == (0.0, 0.0, 0.0)
        assert np.asarray(orb_up.mo_coefficients).dtype == np.complex128

    @pytest.mark.light
    def test_converter_rejects_single_k_file(self, tmp_path):
        """A single-k periodic file is not silently written as a one-twist input."""
        from jqmc.trexio_wrapper import convert_trexio_to_twist_average_input

        fixture = _FIXTURE.with_name("lithium.h5")
        out_path = tmp_path / "single_k.h5"
        with pytest.raises(ValueError, match="at least two"):
            convert_trexio_to_twist_average_input(str(fixture), str(out_path))
        assert not out_path.exists()

    @pytest.mark.light
    def test_converter_rejects_metallic_k_point_occupations(self, tmp_path):
        """A k-point holding more electrons than the cell count is rejected, not truncated."""
        import h5py

        from jqmc.trexio_wrapper import convert_trexio_to_twist_average_input

        metal_file = tmp_path / "lithium_twist_222_metal.h5"
        shutil.copy(_FIXTURE, metal_file)
        with h5py.File(metal_file, "r+") as f:
            occupation = f["mo"]["mo_occupation"][...]
            k_index = f["mo"]["mo_k_point"][...]
            occupation[np.flatnonzero(k_index == 2)[:2]] = 2.0
            f["mo"]["mo_occupation"][...] = occupation

        out_path = tmp_path / "metal.h5"
        with pytest.raises(ValueError, match="sum to 4"):
            convert_trexio_to_twist_average_input(str(metal_file), str(out_path))
        assert not out_path.exists()

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("k_index", "occupied_rows", "message"),
        [
            # Restricted rows [1, 1, 0, ...]: integer and summing to N_up + N_dn = 2,
            # but under the reader's convention (2 = up + dn, 1 = up only) they hold
            # two up and no dn electrons, not the file's (1, 1).
            (0, [1.0, 1.0], "0 doubly and 2 singly occupied"),
            # [2, 2, NaN]: the finite rows already hold four electrons; a NaN row
            # must not hide that count from the rounding and sum checks.
            (3, [2.0, 2.0, np.nan], "must be finite"),
        ],
    )
    def test_reader_and_converter_reject_wrong_spin_or_nonfinite_occupations(self, tmp_path, k_index, occupied_rows, message):
        """A k-block whose spin populations differ from the file's, or that hides its count behind NaN, is rejected."""
        import h5py

        from jqmc.trexio_wrapper import convert_trexio_to_twist_average_input, read_trexio_file

        bad_file = tmp_path / "lithium_twist_222_bad_occupation.h5"
        shutil.copy(_FIXTURE, bad_file)
        with h5py.File(bad_file, "r+") as f:
            assert (int(f["electron"].attrs["electron_up_num"]), int(f["electron"].attrs["electron_dn_num"])) == (1, 1)
            occupation = f["mo"]["mo_occupation"][...]
            rows = np.flatnonzero(f["mo"]["mo_k_point"][...] == k_index)
            assert np.all(f["mo"]["mo_spin"][...][rows] == 0)
            occupation[rows] = 0.0
            occupation[rows[: len(occupied_rows)]] = occupied_rows
            f["mo"]["mo_occupation"][...] = occupation

        with pytest.raises(ValueError, match=message):
            read_trexio_file(str(bad_file), store_tuple=True, twist_ik=k_index)
        out_path = tmp_path / "bad.h5"
        with pytest.raises(ValueError, match=message):
            convert_trexio_to_twist_average_input(str(bad_file), str(out_path))
        assert not out_path.exists()

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("occupation", "spin", "message"),
        [
            ([1.5, 0.5, 0.0, 0.0], [0, 0, 0, 0], "must be integers"),
            ([3.0, 0.0, 0.0, 0.0], [0, 0, 0, 0], "must be integers"),
            ([2.0, 2.0, 0.0, 0.0], [0, 0, 0, 0], "sum to 4"),
            ([1.0, 1.0, 0.0, 1.0], [0, 0, 1, 1], "sum to 2"),
            ([2.0, 0.0, 1.0, 0.0], [0, 0, 1, 1], "must be integers"),
            ([1.0, 1.0, 0.0, 0.0], [0, 0, 0, 0], "0 doubly and 2 singly occupied"),
            ([2.0, 2.0, np.nan, 0.0], [0, 0, 0, 0], "must be finite"),
            ([1.0, np.nan, 0.0, 1.0], [0, 0, 1, 1], "must be finite"),
            ([np.inf, 0.0, 0.0, 0.0], [0, 0, 0, 0], "must be finite"),
        ],
    )
    def test_periodic_occupation_check_rejects_invalid_blocks(self, occupation, spin, message):
        """Fractional, out-of-range and wrong-count blocks are rejected for restricted and UHF rows."""
        from jqmc.trexio_wrapper import _check_periodic_occupations

        with pytest.raises(ValueError, match=message):
            _check_periodic_occupations(np.array(occupation), np.array(spin), 1, 1, 0)

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("occupation", "spin", "num_ele_up", "num_ele_dn"),
        [
            ([2.0, 0.0], [0, 0], 1, 1),
            ([2.0, 1.0, 0.0], [0, 0, 0], 2, 1),
            ([1.0, 2.0, 0.0], [0, 0, 0], 2, 1),
            ([1.0, 0.0, 0.0, 1.0], [0, 0, 1, 1], 1, 1),
        ],
    )
    def test_periodic_occupation_check_accepts_fixed_blocks(self, occupation, spin, num_ele_up, num_ele_dn):
        """RHF, ROHF (aufbau and non-aufbau) and UHF integer blocks with the file's electron numbers are accepted."""
        from jqmc.trexio_wrapper import _check_periodic_occupations

        _check_periodic_occupations(np.array(occupation), np.array(spin), num_ele_up, num_ele_dn, 0)

    # ---------- smoke MCMC on a loaded non-Gamma twist ----------
    @pytest.mark.heavy
    @requires_multitwist
    def test_loaded_twist_runs_mcmc_smoke(self, tmp_path):
        """Sanity-level guard: an MCMC chain runs end-to-end on a loaded
        non-Gamma TREXIO twist without firing the ``ComplexWarning`` autouse
        guard (``conftest._strict_complex_cast_guard``). Pins that the combined
        load -> MCMC chain on the TREXIO-converted multi-twist input
        preserves Im through every dtype-zone cast site.

        Energy magnitude is not asserted against a SCF reference here.
        Stochastic ``VMC ~ E_HF`` tolerance checks are intentionally absent
        from the unit-test suite (non-deterministic; they live in production
        validation). This test only
        pins that the multi-twist -> combined -> load -> MCMC chain flows.
        """
        from jqmc._twist_average import load_twist
        from jqmc.jqmc_mcmc import MCMC
        from jqmc.trexio_wrapper import convert_trexio_to_twist_average_input

        out_path = str(tmp_path / "ha_twists.h5")
        convert_trexio_to_twist_average_input(
            trexio_file=str(_FIXTURE),
            output_path=out_path,
        )
        # Pick a non-Gamma twist (avoids any accidental ``ik == 0`` special-
        # casing in the chain). For the 2x2x2 unreduced mesh any ``ik >= 1``
        # is non-Gamma; picking the last twist exercises the high-index end
        # of the /twists group as a bonus.
        ham = load_twist(out_path, 7)
        mcmc = MCMC(hamiltonian_data=ham, num_walkers=2, mcmc_seed=12345, epsilon_AS=0.0)
        mcmc.run(num_mcmc_steps=30)
        E, Estd, _, _ = mcmc.get_E(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=5)
        assert np.isfinite(E)
        assert np.isfinite(Estd)
