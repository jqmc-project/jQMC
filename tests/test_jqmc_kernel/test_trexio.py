"""collections of unit tests."""

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
import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.trexio_wrapper import read_trexio_file

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


def _trexio_path(filename: str) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", filename)


class TestTREXIO:
    @pytest.mark.light
    def test_native_uhf_sd_uses_independent_occupied_spin_rows(self):
        """Native TREXIO ``mo_spin={0,1}`` maps to canonical TR-off SD."""
        import trexio

        from jqmc.determinant import SD_data

        fixture = _trexio_path("N_ae_ccpvdz_cart.h5")
        file_r = trexio.File(fixture, mode="r", back_end=trexio.TREXIO_HDF5)
        try:
            mo_spin = np.asarray(trexio.read_mo_spin(file_r))
            mo_occupation = np.asarray(trexio.read_mo_occupation(file_r))
            mo_coefficients = np.asarray(trexio.read_mo_coefficient(file_r))
        finally:
            file_r.close()

        assert set(np.unique(mo_spin)) == {0, 1}
        expected_up = mo_coefficients[(mo_spin == 0) & (mo_occupation > 0.0)]
        expected_dn = mo_coefficients[(mo_spin == 1) & (mo_occupation > 0.0)]

        _, _, mos_up, mos_dn, sd, _ = read_trexio_file(
            trexio_file=fixture,
            store_tuple=True,
        )
        assert isinstance(sd, SD_data)
        assert sd.time_reversal is False
        assert sd.num_occ_up == expected_up.shape[0]
        assert sd.num_occ_dn == expected_dn.shape[0]
        np.testing.assert_array_equal(np.asarray(mos_up.mo_coefficients), expected_up)
        np.testing.assert_array_equal(np.asarray(mos_dn.mo_coefficients), expected_dn)
        assert not np.allclose(
            expected_dn, expected_up[: expected_dn.shape[0]], rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION
        )

    @pytest.mark.light
    @pytest.mark.parametrize("layout", ["somo_before_docc", "aufbau"])
    def test_restricted_open_shell_sd_dn_uses_doubly_occupied_rows(self, tmp_path, layout):
        """Restricted open-shell TREXIO (one MO set, ``N_up > N_dn``): the dn
        determinant is built from the doubly occupied MOs named by
        ``mo_occupation``, whatever their storage order.

        Reference (implementation independent): TREXIO ``mo_occupation`` of a
        restricted file counts electrons per spatial orbital (2 = alpha + beta,
        1 = alpha only), so the beta orbitals are exactly the raw
        ``mo/mo_coefficient`` rows with occupation 2, read straight from HDF5.
        The energy-relevant consequence is checked through the public geminal
        API: a single Slater determinant factorises as
        ``|Psi| = |det Phi_up| |det Phi_dn|``, with each factor evaluated by
        numpy on the reference rows.
        """
        import shutil

        import h5py

        from jqmc._precision import get_tolerance_min
        from jqmc.determinant import SD_data, compute_ln_det_geminal_all_elements
        from jqmc.molecular_orbital import MOs_data, compute_MOs

        occupation = {
            "somo_before_docc": [2.0, 1.0, 2.0, 2.0, 2.0],  # non-aufbau (Delta-SCF / MOM, user-set mo_occ)
            "aufbau": [2.0, 2.0, 2.0, 2.0, 1.0],  # standard producer order
        }[layout]
        num_up, num_dn = 5, 4
        fixture = tmp_path / "N2_rohf.h5"
        shutil.copy(_trexio_path("N2_ecp_ccpvtz_cart.h5"), fixture)
        with h5py.File(fixture, "r+") as f:
            f["electron"].attrs["electron_up_num"] = num_up
            f["electron"].attrs["electron_dn_num"] = num_dn
            f["electron"].attrs["electron_num"] = num_up + num_dn
            occ = np.zeros_like(f["mo/mo_occupation"][...])
            occ[: len(occupation)] = occupation
            f["mo/mo_occupation"][...] = occ
            assert set(np.unique(f["mo/mo_spin"][...])) == {0}
            raw = np.asarray(f["mo/mo_coefficient"][...])

        expected_up_rows = [i for i, o in enumerate(occupation) if o >= 1.0]
        expected_dn_rows = [i for i, o in enumerate(occupation) if o >= 2.0]
        assert len(expected_up_rows) == num_up and len(expected_dn_rows) == num_dn

        _, aos, mos_up, mos_dn, sd, _ = read_trexio_file(trexio_file=str(fixture), store_tuple=True)
        assert isinstance(sd, SD_data)
        assert sd.time_reversal is True
        # The up block holds the occupied rows (any order: a row permutation
        # only flips the sign of det_up); the dn block is exactly the doubly
        # occupied rows.
        loaded_up = np.asarray(mos_up.mo_coefficients)
        up_rows = sorted(int(np.flatnonzero((raw == row).all(axis=1))[0]) for row in loaded_up)
        assert up_rows == expected_up_rows
        np.testing.assert_array_equal(np.asarray(mos_dn.mo_coefficients), raw[expected_dn_rows])

        rng = np.random.default_rng(20261009)
        r_up = rng.normal(scale=1.0, size=(num_up, 3))
        r_dn = rng.normal(scale=1.0, size=(num_dn, 3))

        def ln_abs_det(rows, r):
            phi = np.asarray(compute_MOs(MOs_data(num_mo=len(rows), mo_coefficients=raw[rows], aos_data=aos), r))
            return float(np.log(np.abs(np.linalg.det(phi))))

        ln_ref = ln_abs_det(expected_up_rows, r_up) + ln_abs_det(expected_dn_rows, r_dn)
        ln_jqmc = float(compute_ln_det_geminal_all_elements(sd, r_up, r_dn))
        atol, rtol = get_tolerance_min(("ao_eval", "det_eval"), "strict")
        np.testing.assert_allclose(ln_jqmc, ln_ref, atol=atol, rtol=rtol)

        # Non-vacuity: the leading-N_dn-rows choice is a different determinant.
        leading_rows = list(range(num_dn))
        if leading_rows != expected_dn_rows:
            ln_leading = ln_abs_det(expected_up_rows, r_up) + ln_abs_det(leading_rows, r_dn)
            assert not np.isclose(ln_leading, ln_ref, atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5"])
    def test_read_trexio_files_with_tuple(self, trexio_file: str):
        """Test read_trexio files."""
        read_trexio_file(trexio_file=_trexio_path(trexio_file), store_tuple=True)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5"])
    def test_read_trexio_files_with_list(self, trexio_file: str):
        """Test read_trexio files."""
        read_trexio_file(trexio_file=_trexio_path(trexio_file), store_tuple=False)
        jax.clear_caches()

    # ---------------------------------------------------------------------------
    # PBC TREXIO loader -- smoke tests and contract checks.
    # ---------------------------------------------------------------------------

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["lithium.h5"])
    def test_read_trexio_pbc_files_with_tuple(self, trexio_file: str):
        """Smoke test: periodic TREXIO file loads without error
        (``store_tuple=True``)."""
        read_trexio_file(trexio_file=_trexio_path(trexio_file), store_tuple=True)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["lithium.h5"])
    def test_read_trexio_pbc_files_with_list(self, trexio_file: str):
        """Smoke test: periodic TREXIO file loads without error
        (``store_tuple=False``)."""
        read_trexio_file(trexio_file=_trexio_path(trexio_file), store_tuple=False)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["lithium.h5"])
    def test_read_trexio_pbc_returns_pbc_state(self, trexio_file: str):
        """Periodic fixture loads to a fully PBC-populated tuple:

        * ``structure_data.pbc_flag == True`` with non-degenerate cell;
        * ``aos_data._n_image > 1`` (the loader auto-applies
          ``_apply_image_expansion`` on PBC);
        * ``coulomb_potential_data.ewald_data is not None`` with finite
          ``ii_energy``;
        * every loaded dataclass passes its own ``sanity_check``.
        """
        (
            structure_data,
            aos_data,
            _mos_data_up,
            _mos_data_dn,
            _geminal_data,
            coulomb_potential_data,
        ) = read_trexio_file(trexio_file=_trexio_path(trexio_file), store_tuple=True)

        # Structure
        assert structure_data.pbc_flag is True
        assert len(structure_data.vec_a) == 3
        assert len(structure_data.vec_b) == 3
        assert len(structure_data.vec_c) == 3
        assert float(np.abs(np.linalg.det(structure_data.cell))) > 0.0
        structure_data.sanity_check()

        # AOs auto-expanded
        assert aos_data._n_image > 1, f"PBC TREXIO loader must auto-expand AOs; got _n_image={aos_data._n_image}"
        # pgto.num_ao is the externally-observable molecular AO
        # count. Periodic pgto uses the per-shell RAGGED
        # image layout, so the kernel-rank count is M = len(nucleus_index)
        # (each shell keeps only its own radial's images) -- strictly between
        # num_ao and the old uniform ``_n_image * num_ao`` tiling.
        assert aos_data._num_ao_kernel == len(aos_data.nucleus_index)
        assert aos_data.num_ao < aos_data._num_ao_kernel <= aos_data._n_image * aos_data.num_ao
        aos_data.sanity_check()

        # Coulomb_potential_data carries a populated Ewald_data
        assert coulomb_potential_data.ewald_data is not None
        assert np.isfinite(float(coulomb_potential_data.ewald_data.ii_energy))
        coulomb_potential_data.sanity_check()

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5"])
    def test_read_trexio_molecular_returns_molecular_state(self, trexio_file: str):
        """Molecular fixture loads to the dual contract: ``pbc_flag=False``,
        empty cell vectors, ``aos_data._n_image == 1``, and
        ``coulomb_potential_data.ewald_data is None``. Validates the
        molecular code path through the same loader entry point that the
        PBC fixture exercises above.
        """
        (
            structure_data,
            aos_data,
            _mos_data_up,
            _mos_data_dn,
            _geminal_data,
            coulomb_potential_data,
        ) = read_trexio_file(trexio_file=_trexio_path(trexio_file), store_tuple=True)

        assert structure_data.pbc_flag is False
        assert len(structure_data.vec_a) == 0
        assert len(structure_data.vec_b) == 0
        assert len(structure_data.vec_c) == 0
        structure_data.sanity_check()

        assert aos_data._n_image == 1
        aos_data.sanity_check()

        assert coulomb_potential_data.ewald_data is None
        coulomb_potential_data.sanity_check()

        jax.clear_caches()

    @pytest.mark.parametrize(
        "trexio_file,expected_pbc",
        [
            ("water_ccecp_ccpvqz.h5", False),
            ("lithium.h5", True),
        ],
    )
    @pytest.mark.light
    def test_read_trexio_pbc_ewald_contract(self, trexio_file: str, expected_pbc: bool):
        """Strict tri-condition contract enforced by the loader
        (``Coulomb_potential_data`` field placement):

        .. code::

            structure_data.pbc_flag == True
                <=> aos_data._n_image > 1
                <=> coulomb_potential_data.ewald_data is not None

        Parametrised over one molecular and one periodic fixture so the
        bi-implication is checked on both directions.
        """
        (
            structure_data,
            aos_data,
            *_rest,
            coulomb_potential_data,
        ) = read_trexio_file(trexio_file=_trexio_path(trexio_file), store_tuple=True)
        pbc = structure_data.pbc_flag
        assert pbc is expected_pbc
        assert pbc == (aos_data._n_image > 1), f"{trexio_file}: pbc_flag={pbc} but _n_image={aos_data._n_image}"
        assert pbc == (coulomb_potential_data.ewald_data is not None), (
            f"{trexio_file}: pbc_flag={pbc} but ewald_data={coulomb_potential_data.ewald_data}"
        )
        jax.clear_caches()


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
