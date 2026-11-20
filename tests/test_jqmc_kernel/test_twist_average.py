"""Deterministic tests for the BZ-mean aggregator.

Four non-overlapping invariants checkable on dummy data, no MC required:

* **T1 -- identical-twist replication** (weight normalisation +
  ``W_k**2`` quadrature)
* **T2 -- single-twist constant shift + unequal weights** (per-``k`` binning,
  no pooling; weight-scale invariance)
* **T3 -- dual-path: live MPI vs offline serial** (the offline aggregator
  *is* the trivially-correct reference for the live path; here we exercise
  the pure-math helper that both paths call)
* **T4 -- BZ-averaged force sum-rule** (per-twist forces satisfy
  ``Sum_a F_k[a] = 0``; the weighted combine is linear so the sum-rule
  holds for ``F_TA``)
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

from jqmc._setting import ATOL_NONVACUITY_DISTINCTION, RTOL_NONVACUITY_DISTINCTION

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._setting import (
    ATOL_BIT_EXACT,
    ATOL_PYTHON_FP64_BZ_MEAN,
    ATOL_PYTHON_FP64_LOOSE,
    ATOL_PYTHON_FP64_MEDIUM,
    ATOL_PYTHON_FP64_TIGHT,
    RTOL_BIT_EXACT,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_PYTHON_FP64_ZERO,
    get_tol_hdf5_roundtrip,
)
from jqmc._twist_average import (
    bz_mean_energy,
    bz_mean_force,
    is_twist_average_input,
    normalize_twist_weights,
    split_world_for_twists,
)
from tests._test_data_cache import qe_example_files_root


def _file_route_energy_budget() -> tuple[float, float]:
    """``(atol, rtol)`` for the file-route vs in-memory twist-average energy.

    With one twist the file-path wrapper must be byte identical to the
    in-memory driver (``IMPL-TA-BYTE-IDENTITY-001``): since the fp64 basis
    storage after HDF5 load (#133) both routes run the same arrays and agree
    exactly in both precision modes. The budget is therefore bit-exact and
    mode independent. It is a function, not a module constant, so that no
    tolerance is resolved at import (collection) time, before the autouse
    ``configure_precision`` fixture.
    """
    return ATOL_BIT_EXACT, RTOL_BIT_EXACT


# ---------- normalize_twist_weights edge cases (small but worth pinning) -----------


class TestNormalizeWeights:
    @pytest.mark.light
    def test_normalize_weights_rejects_empty(self):
        """Normalize weights rejects empty."""
        with pytest.raises(ValueError, match="empty"):
            normalize_twist_weights([])

    @pytest.mark.light
    def test_normalize_weights_rejects_negative(self):
        """Normalize weights rejects negative."""
        with pytest.raises(ValueError, match="non-negative"):
            normalize_twist_weights([1.0, -0.5, 2.0])

    @pytest.mark.light
    def test_normalize_weights_rejects_all_zero(self):
        """Normalize weights rejects all zero."""
        with pytest.raises(ValueError, match="positive"):
            normalize_twist_weights([0.0, 0.0, 0.0])

    @pytest.mark.light
    def test_normalize_weights_rejects_nonfinite(self):
        """Normalize weights rejects nonfinite."""
        with pytest.raises(ValueError, match="finite"):
            normalize_twist_weights([1.0, float("nan")])

    @pytest.mark.light
    def test_normalize_weights_sum_is_one(self):
        """Normalize weights sum is one."""
        rng = np.random.default_rng(100)
        w = np.abs(rng.standard_normal(20)) + 0.01
        W = normalize_twist_weights(w)
        assert np.sum(W) == pytest.approx(1.0, abs=ATOL_PYTHON_FP64_TIGHT)
        np.testing.assert_array_less(0.0, W)


# ---------- split_world_for_twists -- single-process MPI ----------


class TestSplitWorldForTwists:
    @pytest.mark.light
    def test_split_world_for_twists_single_process_n_twist_1(self):
        """On a single-process ``COMM_WORLD`` (world_size=1) and ``n_twist=1``
        the split is the identity: intra is a 1-rank group containing this
        rank, inter is also a 1-rank group containing this rank as the lone
        twist root, twist_index=0."""
        from mpi4py import MPI

        out = split_world_for_twists(MPI.COMM_WORLD, n_twist=1)
        assert out.n_twist == 1
        assert out.twist_index == 0
        # intra_comm is a fresh communicator of size 1 (not the same handle
        # as world_comm; ``Split`` always returns a new comm).
        assert out.intra_comm.Get_size() == 1
        assert out.intra_comm.Get_rank() == 0
        # inter_comm is the twist-roots comm; this rank is the sole root
        # (intra rank 0), so it gets a 1-rank inter_comm.
        assert out.inter_comm is not None
        assert out.inter_comm.Get_size() == 1
        # Free the handles to avoid leaking comms across tests.
        out.intra_comm.Free()
        out.inter_comm.Free()

    @pytest.mark.light
    def test_split_world_for_twists_rejects_negative_n_twist(self):
        """Split world for twists rejects negative n twist."""
        from mpi4py import MPI

        with pytest.raises(ValueError, match=r"n_twist=\-1"):
            split_world_for_twists(MPI.COMM_WORLD, n_twist=-1)

    @pytest.mark.light
    @pytest.mark.skip_mpi(reason="deterministic: n_twist=2 does not divide a single-process world")
    def test_split_world_for_twists_rejects_non_divisor(self):
        """``world_size`` must be a multiple of ``n_twist``
        (equal compute per twist). On a single-process world (size=1),
        n_twist=2 is not a divisor and must raise."""
        from mpi4py import MPI

        with pytest.raises(ValueError, match="must be a multiple"):
            split_world_for_twists(MPI.COMM_WORLD, n_twist=2)


# ---------- run_twist_average_from_hamiltonians N_twist=1 byte identity -----------


class TestRunTwistAverageFromHamiltonians:
    @pytest.mark.light
    def test_run_twist_average_n_twist_1_byte_identity_to_plain_mcmc(self):
        """``run_twist_average_from_hamiltonians`` at
        ``N_twist=1`` reproduces a plain ``MCMC`` chain byte-for-byte on a
        fixed seed.

        The orchestrator's N_twist=1 path must be a thin wrapper: same MCMC
        construction, same run, same get_E. Any extra reduction or RNG
        perturbation would show up as a non-zero residual here.
        """
        import os

        from jqmc._twist_average import run_twist_average_from_hamiltonians
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
        from jqmc.jqmc_mcmc import MCMC
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H_ae_ccpvdz_cart.h5")
        (
            structure_data,
            _,
            _mos_data,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(trexio_file=fixture, store_tuple=True)

        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp"),
            jastrow_three_body_data=None,
        )
        jastrow_data.sanity_check()
        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        wavefunction_data.sanity_check()
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )
        hamiltonian_data.sanity_check()

        num_walkers = 2
        num_mcmc_steps = 30
        num_warmup = 10
        num_bins = 5
        seed = 12345

        # Reference: plain MCMC
        ref_mcmc = MCMC(
            hamiltonian_data=hamiltonian_data,
            mcmc_seed=seed,
            num_walkers=num_walkers,
        )
        ref_mcmc.run(num_mcmc_steps=num_mcmc_steps)
        E_ref, sigma_ref, _, _ = ref_mcmc.get_E(
            num_mcmc_warmup_steps=num_warmup,
            num_mcmc_bin_blocks=num_bins,
        )

        # Orchestrator at N_twist = 1 must reproduce E_ref exactly.
        result = run_twist_average_from_hamiltonians(
            hamiltonian_list=[hamiltonian_data],
            weights=[1.0],
            mcmc_seed=seed,
            num_mcmc_steps=num_mcmc_steps,
            num_walkers=num_walkers,
            num_mcmc_warmup_steps=num_warmup,
            num_mcmc_bin_blocks=num_bins,
            compute_force=False,
        )

        assert result.energy.E_TA == pytest.approx(float(E_ref), abs=ATOL_PYTHON_FP64_LOOSE)
        assert result.energy.E_TA_std == pytest.approx(float(sigma_ref), abs=ATOL_PYTHON_FP64_LOOSE)
        assert result.energy.s_twist == pytest.approx(0.0, abs=ATOL_PYTHON_FP64_TIGHT)
        assert result.force is None
        np.testing.assert_array_equal(result.per_twist_E, np.array([float(E_ref)]))
        np.testing.assert_array_equal(result.per_twist_E_std, np.array([float(sigma_ref)]))


# ---------- HDF5 twist-averaged combined input writer + load_twist ----------


class TestTwistAverageInputLayout:
    @pytest.mark.light
    def test_twist_average_input_writer_reader_roundtrip(self, tmp_path):
        """Save + load round-trip on a synthetic 2-twist
        file built from an existing single-twist (Gamma) fixture.

        Pins the layout:
          /_meta {n_twist=2, bz_weight_sum=...}
          /shared (the template Hamiltonian_data)
          /twists/twist_000/{k_point, weight, mo_coefficients_up,
                             mo_coefficients_dn}
          /twists/twist_001/...

        And the meta-only ``load_twist_average_input`` returns the expected
        per-twist k_points + weights.
        """
        import os

        from jqmc._twist_average import (
            load_twist_average_input,
            save_twist_average_input,
        )
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H_ae_ccpvdz_cart.h5")
        (
            structure_data,
            _,
            _mos_data,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(trexio_file=fixture, store_tuple=True)
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp"),
            jastrow_three_body_data=None,
        )
        template_ham = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data),
        )
        template_ham.sanity_check()

        # Build synthetic per-twist arrays. Same shapes as the template
        # geminal_mo_data. Use complex128 throughout to mimic non-Gamma load.
        mo_up_shape = np.asarray(geminal_mo_data.orb_data_up_spin.mo_coefficients).shape
        mo_dn_shape = np.asarray(geminal_mo_data.orb_data_dn_spin.mo_coefficients).shape
        rng = np.random.default_rng(42)
        mo_up_0 = np.asarray(geminal_mo_data.orb_data_up_spin.mo_coefficients, dtype=np.complex128)
        mo_up_1 = mo_up_0 + 0.001 * (rng.standard_normal(mo_up_shape) + 1j * rng.standard_normal(mo_up_shape))
        mo_dn_0 = np.asarray(geminal_mo_data.orb_data_dn_spin.mo_coefficients, dtype=np.complex128)
        if geminal_mo_data.time_reversal:
            mo_dn_1 = np.conj(mo_up_1[: geminal_mo_data.num_electron_dn])
        else:
            mo_dn_1 = mo_dn_0 + 0.001 * (rng.standard_normal(mo_dn_shape) + 1j * rng.standard_normal(mo_dn_shape))

        twist_path = str(tmp_path / "hamiltonian_data_twists.h5")
        save_twist_average_input(
            filepath=twist_path,
            template_hamiltonian=template_ham,
            k_points=[None, (0.25, 0.0, 0.0)],
            weights=[1.0, 1.0],
            mo_coefficients_up_per_twist=[mo_up_0, mo_up_1],
            mo_coefficients_dn_per_twist=[mo_dn_0, mo_dn_1],
        )

        import h5py

        with h5py.File(twist_path, "r") as h5:
            assert "pbc_schema_version" not in h5["_meta"].attrs
            assert "sd_representation" not in h5["shared/wavefunction_data/geminal_data"].attrs
            assert all("geminal_representation" not in twist.attrs for twist in h5["twists"].values())
        meta = load_twist_average_input(twist_path)
        assert meta.n_twist == 2
        assert meta.k_points[0] is None
        assert meta.k_points[1] == (0.25, 0.0, 0.0)
        np.testing.assert_array_equal(meta.weights, np.array([1.0, 1.0]))
        # bz_weight_sum is fp64 metadata (np.sum of the fp64 weights), never a
        # mixed-precision computation, so it is judged at fp64 ulp in both modes.
        assert meta.bz_weight_sum == pytest.approx(2.0, abs=ATOL_PYTHON_FP64_TIGHT)

    @pytest.mark.light
    @pytest.mark.parametrize("spin", ["up", "dn"])
    def test_twist_writer_rejects_noncanonical_sd_mo_rows(self, tmp_path, spin):
        """A new twist file cannot persist virtual rows in canonical SD."""
        import os

        from jqmc._twist_average import save_twist_average_input
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            "H_ae_ccpvdz_cart.h5",
        )
        structure, _, _, _, sd, coulomb = read_trexio_file(
            trexio_file=fixture,
            store_tuple=True,
        )
        template = Hamiltonian_data(
            structure_data=structure,
            coulomb_potential_data=coulomb,
            wavefunction_data=Wavefunction_data(
                jastrow_data=Jastrow_data(),
                geminal_data=sd,
            ),
        )
        mo_up = np.asarray(sd.orb_data_up_spin.mo_coefficients)
        mo_dn = np.asarray(sd.orb_data_dn_spin.mo_coefficients)
        if spin == "up":
            mo_up = np.concatenate(
                [mo_up, np.zeros((1, mo_up.shape[1]), dtype=mo_up.dtype)],
                axis=0,
            )
        else:
            mo_dn = np.concatenate(
                [mo_dn, np.zeros((1, mo_dn.shape[1]), dtype=mo_dn.dtype)],
                axis=0,
            )

        with pytest.raises(ValueError, match=rf"twist 0.*{spin}-spin.*exactly"):
            save_twist_average_input(
                filepath=str(tmp_path / "invalid_sd_twist.h5"),
                template_hamiltonian=template,
                k_points=[None],
                weights=[1.0],
                mo_coefficients_up_per_twist=[mo_up],
                mo_coefficients_dn_per_twist=[mo_dn],
            )

    @pytest.mark.light
    def test_twist_writer_rejects_inconsistent_time_reversal_sd(self, tmp_path):
        """A tied SD twist cannot persist an independent down-spin state."""
        import os

        from jqmc._twist_average import save_twist_average_input
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            "H2_ae_ccpvdz_cart.h5",
        )
        structure, _, _, _, sd, coulomb = read_trexio_file(
            trexio_file=fixture,
            store_tuple=True,
        )
        assert sd.time_reversal
        template = Hamiltonian_data(
            structure_data=structure,
            coulomb_potential_data=coulomb,
            wavefunction_data=Wavefunction_data(
                jastrow_data=Jastrow_data(),
                geminal_data=sd,
            ),
        )
        mo_up = np.asarray(sd.orb_data_up_spin.mo_coefficients)
        inconsistent_dn = np.asarray(sd.orb_data_dn_spin.mo_coefficients) + 0.125

        with pytest.raises(ValueError, match=r"twist 0.*time_reversal=True"):
            save_twist_average_input(
                filepath=str(tmp_path / "invalid_tr_sd_twist.h5"),
                template_hamiltonian=template,
                k_points=[None],
                weights=[1.0],
                mo_coefficients_up_per_twist=[mo_up],
                mo_coefficients_dn_per_twist=[inconsistent_dn],
            )

    @pytest.mark.light
    def test_twist_roundtrip_preserves_independent_complex_sd_spin_channels(self, tmp_path):
        """time_reversal=False keeps distinct complex up/down occupied MOs."""
        import os

        from jqmc._twist_average import load_twist, save_twist_average_input
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            "lithium_k025.h5",
        )
        structure, _, _, _, tied_sd, coulomb = read_trexio_file(
            trexio_file=fixture,
            store_tuple=True,
        )
        mo_up = np.asarray(tied_sd.orb_data_up_spin.mo_coefficients)
        mo_dn = np.asarray(tied_sd.orb_data_dn_spin.mo_coefficients) * np.exp(0.2j)
        independent_sd = tied_sd.replace(
            orb_data_dn_spin=tied_sd.orb_data_dn_spin.replace(
                mo_coefficients=np.ascontiguousarray(mo_dn),
            ),
            time_reversal=False,
        )
        independent_sd.sanity_check()
        assert not np.allclose(mo_dn, np.conj(mo_up), rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION)
        template = Hamiltonian_data(
            structure_data=structure,
            coulomb_potential_data=coulomb,
            wavefunction_data=Wavefunction_data(
                jastrow_data=Jastrow_data(),
                geminal_data=independent_sd,
            ),
        )
        path = str(tmp_path / "independent_complex_sd_twist.h5")
        save_twist_average_input(
            filepath=path,
            template_hamiltonian=template,
            k_points=[(0.25, 0.0, 0.0)],
            weights=[1.0],
            mo_coefficients_up_per_twist=[mo_up],
            mo_coefficients_dn_per_twist=[mo_dn],
        )

        loaded = load_twist(path, 0).wavefunction_data.geminal_data
        assert loaded.time_reversal is False
        np.testing.assert_array_equal(
            np.asarray(loaded.orb_data_up_spin.mo_coefficients),
            mo_up,
        )
        np.testing.assert_array_equal(
            np.asarray(loaded.orb_data_dn_spin.mo_coefficients),
            mo_dn,
        )

    @pytest.mark.light
    def test_multi_twist_roundtrip_preserves_independent_complex_sd_spin_channels(self, tmp_path):
        """Every twist keeps both independent spin channels when TR is off."""
        import os

        from jqmc._twist_average import load_twist, save_twist_average_input
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            "lithium_k025.h5",
        )
        structure, _, _, _, tied_sd, coulomb = read_trexio_file(
            trexio_file=fixture,
            store_tuple=True,
        )
        base_up = np.asarray(tied_sd.orb_data_up_spin.mo_coefficients)
        base_dn = np.asarray(tied_sd.orb_data_dn_spin.mo_coefficients)
        up_per_twist = [
            np.ascontiguousarray(base_up * np.exp(0.11j)),
            np.ascontiguousarray(base_up * np.exp(-0.23j)),
        ]
        dn_per_twist = [
            np.ascontiguousarray(base_dn * np.exp(0.37j)),
            np.ascontiguousarray(base_dn * np.exp(-0.41j)),
        ]
        independent_sd = tied_sd.replace(
            orb_data_up_spin=tied_sd.orb_data_up_spin.replace(
                mo_coefficients=up_per_twist[0],
            ),
            orb_data_dn_spin=tied_sd.orb_data_dn_spin.replace(
                mo_coefficients=dn_per_twist[0],
            ),
            time_reversal=False,
        )
        independent_sd.sanity_check()
        template = Hamiltonian_data(
            structure_data=structure,
            coulomb_potential_data=coulomb,
            wavefunction_data=Wavefunction_data(
                jastrow_data=Jastrow_data(),
                geminal_data=independent_sd,
            ),
        )
        k_points = [(0.25, 0.0, 0.0), (0.125, -0.25, 0.0)]
        path = str(tmp_path / "independent_complex_sd_multi_twist.h5")
        save_twist_average_input(
            filepath=path,
            template_hamiltonian=template,
            k_points=k_points,
            weights=[0.4, 0.6],
            mo_coefficients_up_per_twist=up_per_twist,
            mo_coefficients_dn_per_twist=dn_per_twist,
        )

        for twist_index, (expected_up, expected_dn, expected_k) in enumerate(
            zip(up_per_twist, dn_per_twist, k_points, strict=True)
        ):
            loaded = load_twist(path, twist_index).wavefunction_data.geminal_data
            assert loaded.time_reversal is False
            assert loaded.orb_data_up_spin.aos_data.k_point == expected_k
            # Opposite spin twists for independent channels too: the stored
            # dn payload (conj of the beta channel's +k rows) is rebuilt on
            # the -k leaf, unchanged.
            assert loaded.orb_data_dn_spin.aos_data.k_point == tuple(-c for c in expected_k)
            np.testing.assert_array_equal(
                np.asarray(loaded.orb_data_up_spin.mo_coefficients),
                expected_up,
            )
            np.testing.assert_array_equal(
                np.asarray(loaded.orb_data_dn_spin.mo_coefficients),
                expected_dn,
            )
            assert not np.allclose(
                expected_dn, np.conj(expected_up), rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION
            )

    @pytest.mark.light
    @pytest.mark.parametrize(
        "dataset_name",
        ["lambda_matrix", "lambda_diag", "lambda_unpaired", "lambda_future"],
    )
    def test_twist_loader_rejects_any_lambda_payload_in_canonical_sd(self, tmp_path, dataset_name):
        """An occupied SD cannot contain any lambda payload."""
        import os

        import h5py

        from jqmc._twist_average import load_twist, save_twist_average_input
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            "H_ae_ccpvdz_cart.h5",
        )
        structure, _, _, _, sd, coulomb = read_trexio_file(
            trexio_file=fixture,
            store_tuple=True,
        )
        template = Hamiltonian_data(
            structure_data=structure,
            coulomb_potential_data=coulomb,
            wavefunction_data=Wavefunction_data(
                jastrow_data=Jastrow_data(),
                geminal_data=sd,
            ),
        )
        mo_up = np.asarray(sd.orb_data_up_spin.mo_coefficients)
        mo_dn = np.asarray(sd.orb_data_dn_spin.mo_coefficients)
        path = str(tmp_path / f"canonical_sd_with_{dataset_name}.h5")
        save_twist_average_input(
            filepath=path,
            template_hamiltonian=template,
            k_points=[None],
            weights=[1.0],
            mo_coefficients_up_per_twist=[mo_up],
            mo_coefficients_dn_per_twist=[mo_dn],
        )
        with h5py.File(path, "r+") as h5:
            group = h5["twists/twist_000"]
            assert "geminal_representation" not in group.attrs
            group.create_dataset(dataset_name, data=np.ones((1,), dtype=np.float64))

        with pytest.raises(ValueError, match="not a canonical occupied SD"):
            load_twist(path, 0)

    @pytest.mark.light
    @pytest.mark.parametrize("spin", ["up", "dn"])
    def test_twist_loader_rejects_sd_with_extra_mo_rows(self, tmp_path, spin):
        """An occupied SD cannot contain a forged virtual row."""
        import os

        import h5py

        from jqmc._twist_average import load_twist, save_twist_average_input
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            "H_ae_ccpvdz_cart.h5",
        )
        structure, _, _, _, sd, coulomb = read_trexio_file(
            trexio_file=fixture,
            store_tuple=True,
        )
        template = Hamiltonian_data(
            structure_data=structure,
            coulomb_potential_data=coulomb,
            wavefunction_data=Wavefunction_data(
                jastrow_data=Jastrow_data(),
                geminal_data=sd,
            ),
        )
        mo_up = np.asarray(sd.orb_data_up_spin.mo_coefficients)
        mo_dn = np.asarray(sd.orb_data_dn_spin.mo_coefficients)
        path = str(tmp_path / "forged_sd_twist.h5")
        save_twist_average_input(
            filepath=path,
            template_hamiltonian=template,
            k_points=[None],
            weights=[1.0],
            mo_coefficients_up_per_twist=[mo_up],
            mo_coefficients_dn_per_twist=[mo_dn],
        )
        dataset_name = f"mo_coefficients_{spin}"
        with h5py.File(path, "r+") as h5:
            group = h5["twists/twist_000"]
            values = np.asarray(group[dataset_name])
            del group[dataset_name]
            group.create_dataset(
                dataset_name,
                data=np.concatenate(
                    [values, np.zeros((1, values.shape[1]), dtype=values.dtype)],
                    axis=0,
                ),
            )

        with pytest.raises(ValueError, match=rf"twist 0.*{spin}-spin.*exactly"):
            load_twist(path, 0)

    @pytest.mark.light
    def test_hamiltonian_data_load_from_hdf5_rejects_combined(self):
        """Hamiltonian_data.load_from_hdf5 must reject combined inputs with a
        clear pointer to the twist-averaged loader. Pins the routing.
        """
        import os
        import tempfile

        from jqmc._twist_average import save_twist_average_input
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H_ae_ccpvdz_cart.h5")
        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(trexio_file=fixture, store_tuple=True)
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp"),
            jastrow_three_body_data=None,
        )
        template_ham = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data),
        )

        mo_up = np.asarray(geminal_mo_data.orb_data_up_spin.mo_coefficients, dtype=np.complex128)
        mo_dn = np.asarray(geminal_mo_data.orb_data_dn_spin.mo_coefficients, dtype=np.complex128)

        with tempfile.TemporaryDirectory() as td:
            twist_path = os.path.join(td, "ta_input.h5")
            save_twist_average_input(
                filepath=twist_path,
                template_hamiltonian=template_ham,
                k_points=[None],
                weights=[1.0],
                mo_coefficients_up_per_twist=[mo_up],
                mo_coefficients_dn_per_twist=[mo_dn],
            )
            with pytest.raises(ValueError, match="twist-averaged"):
                Hamiltonian_data.load_from_hdf5(twist_path)

    @pytest.mark.light
    def test_load_twist_returns_per_twist_hamiltonian(self, tmp_path):
        """``load_twist(path, k_idx)`` returns a Hamiltonian whose
        ``aos_data.k_point`` matches the per-twist value and whose
        geminal carries the per-twist occupied MOs. End-to-end pin of the
        layout."""
        import os

        from jqmc._twist_average import load_twist, save_twist_average_input
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H_ae_ccpvdz_cart.h5")
        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(trexio_file=fixture, store_tuple=True)
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp"),
            jastrow_three_body_data=None,
        )
        template_ham = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data),
        )

        # Use REAL arrays for the Gamma-only twist set. The H_ae fixture is
        # molecular (real AO leaf), so the geminal sanity check would reject
        # complex MO; this is intended (twist averaging at Gamma is
        # degenerate -- the writer must still round-trip real arrays).
        mo_up_0 = np.asarray(geminal_mo_data.orb_data_up_spin.mo_coefficients, dtype=np.float64)
        mo_dn_0 = np.asarray(geminal_mo_data.orb_data_dn_spin.mo_coefficients, dtype=np.float64)

        # Two Gamma twists with the same data (placeholder for a real BZ
        # grid -- the non-Gamma case needs a Bloch-pGTO complex AO leaf
        # fixture, which is exercised separately by
        # tests/test_jqmc_kernel/test_complex_pgto.py and the converter tests).
        twist_path = str(tmp_path / "ta_input.h5")
        save_twist_average_input(
            filepath=twist_path,
            template_hamiltonian=template_ham,
            k_points=[None, None],
            weights=[1.0, 1.0],
            mo_coefficients_up_per_twist=[mo_up_0, mo_up_0],
            mo_coefficients_dn_per_twist=[mo_dn_0, mo_dn_0],
        )

        ham_0 = load_twist(twist_path, 0)
        ham_1 = load_twist(twist_path, 1)

        # Both are Gamma. ``k_point`` lives on the AO leaf (or
        # the k-dependent composite child), not on :class:`Structure_data`.
        # On the Gamma path the loader builds a real AO leaf with
        # ``k_point=None``.
        assert ham_0.wavefunction_data.geminal_data.orb_data_up_spin.aos_data.k_point is None
        assert ham_1.wavefunction_data.geminal_data.orb_data_up_spin.aos_data.k_point is None
        # Occupied MOs are restored to the per-twist values (real path).
        np.testing.assert_array_equal(
            np.asarray(ham_1.wavefunction_data.geminal_data.orb_data_up_spin.mo_coefficients),
            mo_up_0,
        )
        assert not hasattr(ham_1.wavefunction_data.geminal_data, "lambda_matrix")

        # And the loaded Hamiltonian passes sanity_check (i.e. all the
        # post-load fixups left it self-consistent).
        ham_0.sanity_check()
        ham_1.sanity_check()

    @pytest.mark.light
    def test_is_twist_average_input_distinguishes_combined_and_single(self, tmp_path):
        """``is_twist_average_input`` returns True only for combined inputs;
        a single-twist Hamiltonian file returns False. Pins the
        CLI auto-detect predicate."""
        import os

        from jqmc._twist_average import save_twist_average_input
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H_ae_ccpvdz_cart.h5")
        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(trexio_file=fixture, store_tuple=True)
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp"),
            jastrow_three_body_data=None,
        )
        ham = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data),
        )

        # Write single-twist file via the existing API.
        single_path = str(tmp_path / "single_input.h5")
        ham.save_to_hdf5(single_path)
        assert is_twist_average_input(single_path) is False

        # Write combined twist-averaged file.
        combined_path = str(tmp_path / "combined_input.h5")
        mo_up = np.asarray(geminal_mo_data.orb_data_up_spin.mo_coefficients, dtype=np.float64)
        mo_dn = np.asarray(geminal_mo_data.orb_data_dn_spin.mo_coefficients, dtype=np.float64)
        save_twist_average_input(
            filepath=combined_path,
            template_hamiltonian=ham,
            k_points=[None],
            weights=[1.0],
            mo_coefficients_up_per_twist=[mo_up],
            mo_coefficients_dn_per_twist=[mo_dn],
        )
        assert is_twist_average_input(combined_path) is True


# ---------- per-twist + shared-state checkpoint helpers ----------


class TestCheckpointHelpers:
    @pytest.mark.light
    def test_twist_checkpoint_path_and_shared_state_path(self):
        """Path helpers produce the canonical filenames."""
        from jqmc._twist_average import shared_state_path, twist_checkpoint_path

        assert twist_checkpoint_path("/tmp/chk", 0) == "/tmp/chk/twist_000.h5"
        assert twist_checkpoint_path("/tmp/chk", 7) == "/tmp/chk/twist_007.h5"
        assert twist_checkpoint_path("/tmp/chk", 123) == "/tmp/chk/twist_123.h5"
        assert shared_state_path("/tmp/chk") == "/tmp/chk/shared_state.h5"

    @pytest.mark.light
    def test_shared_jastrow_round_trip(self, tmp_path):
        """``save_shared_jastrow`` + ``load_shared_jastrow`` round-trip the
        optimised Jastrow exactly (deterministic byte-level equality on the
        leaves)."""
        from jqmc._twist_average import load_shared_jastrow, save_shared_jastrow
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data

        jastrow_in = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.7, jastrow_2b_type="exp"),
            jastrow_three_body_data=None,
        )

        path = str(tmp_path / "shared_state.h5")
        save_shared_jastrow(path, jastrow_in)

        jastrow_out = load_shared_jastrow(path)
        assert type(jastrow_out) is Jastrow_data
        # Persistence of the stored leaf is exact in both modes (the leaf is
        # written and read back at its own dtype), so the guard stays at fp64
        # ulp instead of following the mixed HDF5 round-trip budget.
        assert jastrow_out.jastrow_two_body_data.jastrow_2b_param == pytest.approx(
            jastrow_in.jastrow_two_body_data.jastrow_2b_param, abs=ATOL_PYTHON_FP64_TIGHT
        )
        assert jastrow_out.jastrow_two_body_data.jastrow_2b_type == jastrow_in.jastrow_two_body_data.jastrow_2b_type

    @pytest.mark.light
    def test_load_shared_jastrow_rejects_wrong_payload(self, tmp_path):
        """Loading a file without the ``payload=shared_jastrow`` attr raises
        with a clear message (catches an accidental Hamiltonian or twist
        input mis-pointed at the loader)."""
        import h5py

        from jqmc._twist_average import load_shared_jastrow

        path = str(tmp_path / "wrong_file.h5")
        with h5py.File(path, "w") as f:
            f.attrs["payload"] = "something_else"
        with pytest.raises(ValueError, match="shared_jastrow"):
            load_shared_jastrow(path)

    @pytest.mark.light
    def test_is_twist_run_complete_missing_file_returns_false(self, tmp_path):
        """Missing per-twist file => not complete (caller restarts from scratch)."""
        from jqmc._twist_average import is_twist_run_complete

        assert is_twist_run_complete(str(tmp_path / "nope.h5"), target_steps=100) is False

    @pytest.mark.light
    def test_is_twist_run_complete_counter_thresholds(self, tmp_path):
        """``is_twist_run_complete`` returns True iff
        ``mcmc_counter >= target_steps``. Test with a hand-crafted minimal
        HDF5 file mimicking the driver_config layout."""
        import h5py

        from jqmc._twist_average import is_twist_run_complete

        path = str(tmp_path / "twist_000.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("driver_config")
            grp.create_dataset("mcmc_counter", data=np.int64(120))

        assert is_twist_run_complete(path, target_steps=100) is True
        assert is_twist_run_complete(path, target_steps=120) is True
        assert is_twist_run_complete(path, target_steps=121) is False

    # ---------- CLI auto-dispatch ----------

    @pytest.mark.light
    def test_cli_auto_dispatch_selects_combined_input(self, tmp_path):
        """The production CLI policy distinguishes ordinary and combined inputs."""
        import os

        from jqmc._twist_average import save_twist_average_input
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
        from jqmc.jqmc_cli import _resolve_twist_dispatch
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H_ae_ccpvdz_cart.h5")
        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(trexio_file=fixture, store_tuple=True)
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp"),
            jastrow_three_body_data=None,
        )
        template_ham = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data),
        )
        mo_up = np.asarray(geminal_mo_data.orb_data_up_spin.mo_coefficients, dtype=np.float64)
        mo_dn = np.asarray(geminal_mo_data.orb_data_dn_spin.mo_coefficients, dtype=np.float64)

        # An ordinary single-twist file stays on the single driver.
        single_path = str(tmp_path / "v2.h5")
        template_ham.save_to_hdf5(single_path)
        assert _resolve_twist_dispatch(single_path) == "single"

        # A combined multi-twist file auto-routes to the twist driver.
        combined_path = str(tmp_path / "combined.h5")
        save_twist_average_input(
            filepath=combined_path,
            template_hamiltonian=template_ham,
            k_points=[None, None],
            weights=[1.0, 1.0],
            mo_coefficients_up_per_twist=[mo_up, mo_up],
            mo_coefficients_dn_per_twist=[mo_dn, mo_dn],
        )
        assert _resolve_twist_dispatch(combined_path) == "twist_average"


# ---------- run_twist_average file-path wrapper ----------


class TestFilePathWrapper:
    @pytest.mark.light
    def test_run_twist_average_file_path_wrapper_n_twist_1(self, tmp_path):
        """``run_twist_average(filepath)`` at N_twist=1 produces the same
        ``E_TA`` as the in-memory ``run_twist_average_from_hamiltonians``
        on the same template. End-to-end pin of the loader + driver glue."""
        import os

        from jqmc._twist_average import (
            run_twist_average,
            run_twist_average_from_hamiltonians,
            save_twist_average_input,
        )
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H_ae_ccpvdz_cart.h5")
        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(trexio_file=fixture, store_tuple=True)
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp"),
            jastrow_three_body_data=None,
        )
        template_ham = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data),
        )
        mo_up = np.asarray(geminal_mo_data.orb_data_up_spin.mo_coefficients, dtype=np.float64)
        mo_dn = np.asarray(geminal_mo_data.orb_data_dn_spin.mo_coefficients, dtype=np.float64)

        twist_path = str(tmp_path / "ta_input.h5")
        save_twist_average_input(
            filepath=twist_path,
            template_hamiltonian=template_ham,
            k_points=[None],
            weights=[1.0],
            mo_coefficients_up_per_twist=[mo_up],
            mo_coefficients_dn_per_twist=[mo_dn],
        )

        common = dict(
            mcmc_seed=12345,
            num_mcmc_steps=30,
            num_walkers=2,
            num_mcmc_warmup_steps=10,
            num_mcmc_bin_blocks=5,
            compute_force=False,
        )

        out_file = run_twist_average(twist_input_path=twist_path, **common)
        out_mem = run_twist_average_from_hamiltonians(
            hamiltonian_list=[template_ham],
            weights=[1.0],
            **common,
        )

        # Same algorithm, same seed, same template -- same energy, bit for bit
        # in both precision modes (IMPL-TA-BYTE-IDENTITY-001).
        file_atol, file_rtol = _file_route_energy_budget()
        assert out_file.energy.E_TA == pytest.approx(out_mem.energy.E_TA, abs=file_atol, rel=file_rtol)
        assert out_file.energy.E_TA_std == pytest.approx(out_mem.energy.E_TA_std, abs=file_atol, rel=file_rtol)

    @pytest.mark.light
    def test_file_route_energy_budget_is_strict_and_not_resolved_at_import(self, request):
        """No tolerance of this module
        is resolved at import time, and the file-route vs in-memory energy
        budget is strict and mode independent.

        Two independent detectors:

        1. Static: no statement executed at import time (module or class
           body, decorators, default arguments) calls a ``get_tol*`` /
           ``get_tolerance*`` resolver. Such a binding runs during collection,
           before the autouse ``configure_precision`` fixture, and freezes the
           collection-time mode's budget.
        2. Behavioural: after switching the policy to ``full`` and then to
           ``mixed``, the helper used by the file-route energy comparison
           returns the bit-exact budget in both modes
           (``IMPL-TA-BYTE-IDENTITY-001``: the two routes are byte identical),
           which is never looser than that mode's strict HDF5 round-trip
           budget. A mode-following HDF5 budget fails this in both modes.
        """
        import ast
        import inspect
        import sys as _sys

        from jqmc._precision import configure

        module = _sys.modules[type(self).__module__]
        tree = ast.parse(inspect.getsource(module))

        def _import_time_nodes(stmts: list[ast.stmt]) -> list[ast.AST]:
            # Collect the AST nodes evaluated at import: everything except
            # function bodies (their decorators and defaults do run at import).
            nodes: list[ast.AST] = []
            for stmt in stmts:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    nodes.extend(stmt.decorator_list)
                    nodes.extend(stmt.args.defaults)
                    nodes.extend(d for d in stmt.args.kw_defaults if d is not None)
                elif isinstance(stmt, ast.ClassDef):
                    nodes.extend(stmt.decorator_list)
                    nodes.extend(stmt.bases)
                    nodes.extend(_import_time_nodes(stmt.body))
                else:
                    nodes.append(stmt)
            return nodes

        offenders = []
        for top in _import_time_nodes(tree.body):
            for node in ast.walk(top):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                    if name.startswith("get_tol"):
                        offenders.append(f"line {node.lineno}: {name}(...)")
        assert not offenders, (
            "precision-dependent tolerance resolved at import time, before the precision fixture: " + ", ".join(offenders)
        )

        active_mode = request.config.getoption("--precision-mode")
        request.addfinalizer(lambda: configure(active_mode))
        budgets = {}
        for mode in ("full", "mixed"):
            configure(mode)
            budgets[mode] = _file_route_energy_budget()
            assert budgets[mode] == (ATOL_BIT_EXACT, RTOL_BIT_EXACT), (mode, budgets[mode])
            hdf5_atol, hdf5_rtol = get_tol_hdf5_roundtrip("strict")
            assert budgets[mode][0] <= hdf5_atol and budgets[mode][1] <= hdf5_rtol, mode
        assert budgets["full"] == budgets["mixed"]


# ---------------------------------------------------------------------------
# BZ-mean physical invariants. Aggregator-level property tests on
# ``bz_mean_energy`` / ``bz_mean_force`` -- pure functions, no MCMC run required.
# ---------------------------------------------------------------------------


class TestBZMeanInvariants:
    """Physical-invariant guards for the two-level twist-averaged estimator."""

    # ---------- T1 (identical-twist replication) ----------

    @pytest.mark.light
    def test_T1_identical_twist_replication_energy(self):
        """T1 (energy): N identical twists with the same fixed
        ``(E_bar, sigma)`` and raw weights ``w_k = 1`` => ``E_TA == E_bar``,
        ``sigma(E_TA) == sigma / sqrt(N)``, ``s_twist == 0``.

        Pins (a) the weight normalisation (``Sum_k W_k = 1``), and (b) the
        ``W_k**2`` in the quadrature -- a ``W_k`` (instead of ``W_k**2``) bug
        gives ``sigma`` (instead of ``sigma/sqrt(N)``) and fails here.
        """
        N = 5
        E_bar = 3.5
        sigma_k = 0.04
        out = bz_mean_energy(
            E_per_twist=np.full(N, E_bar),
            sigma_per_twist=np.full(N, sigma_k),
            weights=np.ones(N),
        )
        assert out.E_TA == pytest.approx(E_bar, abs=ATOL_PYTHON_FP64_TIGHT)
        assert out.E_TA_std == pytest.approx(sigma_k / np.sqrt(N), abs=ATOL_PYTHON_FP64_TIGHT)
        assert out.s_twist == pytest.approx(0.0, abs=ATOL_PYTHON_FP64_TIGHT)
        np.testing.assert_allclose(out.W, np.full(N, 1.0 / N), atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_ZERO)

    @pytest.mark.light
    def test_T1_identical_twist_replication_force(self):
        """T1 (force): same algebra, vector observable."""
        N = 4
        n_atoms = 3
        F_bar = np.array([[0.1, 0.0, -0.05], [0.0, 0.2, 0.1], [-0.1, -0.2, -0.05]])
        sigma_k = 0.01 * np.ones_like(F_bar)
        out = bz_mean_force(
            F_per_twist=np.broadcast_to(F_bar, (N, n_atoms, 3)).copy(),
            sigma_per_twist=np.broadcast_to(sigma_k, (N, n_atoms, 3)).copy(),
            weights=np.ones(N),
        )
        np.testing.assert_allclose(out.F_TA, F_bar, atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(out.F_TA_std, sigma_k / np.sqrt(N), atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_ZERO)

    @pytest.mark.light
    def test_bz_mean_force_preserves_complex_generalized_force(self):
        """The generic array combiner must not real-project raw ``gF``.

        Real/complex parameter decoding belongs to the optimizer coordinate
        map.  Taking ``Re`` in the BZ layer would freeze the imaginary degree
        of freedom of every complex variational parameter.
        """
        gF = np.array(
            [
                [1.0 + 2.0j, -0.5 + 0.25j],
                [3.0 - 1.0j, 0.75 + 4.0j],
            ],
            dtype=np.complex128,
        )
        sigma = np.array([[0.2, 0.4], [0.3, 0.1]], dtype=np.float64)
        weights = np.array([3.0, 1.0], dtype=np.float64)

        out = bz_mean_force(gF, sigma, weights)
        W = weights / weights.sum()

        assert np.iscomplexobj(out.F_TA)
        np.testing.assert_allclose(
            out.F_TA, np.einsum("k,kp->p", W, gF), atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            out.F_TA_std,
            np.sqrt(np.einsum("k,kp->p", W**2, sigma**2)),
            atol=ATOL_PYTHON_FP64_TIGHT,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )

    # ---------- T2 (constant shift + unequal weights) ----------

    @pytest.mark.light
    def test_T2_constant_shift_one_twist_unequal_weights(self):
        """T2: add a constant ``C`` to ONE twist's mean. With
        unequal raw weights ``w_k``, the combine must shift ``E_TA`` by
        ``W_k * C`` (the correct weight on the correct twist) AND keep
        ``sigma(E_TA)`` unchanged (the per-``k`` MC error is invariant under
        a deterministic shift of one twist's mean; the unique signature of
        correct per-``k`` binning).
        """
        E_base = np.array([1.0, 2.0, 3.0])
        sigma_k = np.array([0.10, 0.05, 0.20])
        w_raw = np.array([1.0, 2.0, 1.0])  # unequal
        C = 7.0
        target_idx = 1

        out_before = bz_mean_energy(E_base, sigma_k, w_raw)

        E_after = E_base.copy()
        E_after[target_idx] += C
        out_after = bz_mean_energy(E_after, sigma_k, w_raw)

        W = out_before.W
        expected_shift = W[target_idx] * C
        assert out_after.E_TA == pytest.approx(out_before.E_TA + expected_shift, abs=ATOL_PYTHON_FP64_LOOSE)
        # sigma must not move: a deterministic shift of one twist's mean
        # does not change the MC error.
        assert out_after.E_TA_std == pytest.approx(out_before.E_TA_std, abs=ATOL_PYTHON_FP64_TIGHT)

    @pytest.mark.light
    def test_T2_weight_scale_invariance(self):
        """Folded into T2: scaling all raw weights by a constant must not
        change ``E_TA``, ``sigma(E_TA)``, or ``s_twist``. ``normalize_twist_weights``
        divides by the sum, so this is true by construction."""
        E = np.array([0.5, -0.3, 1.2, 0.8])
        sigma_k = np.array([0.05, 0.10, 0.07, 0.02])
        w = np.array([3.0, 1.0, 2.0, 4.0])

        out_1x = bz_mean_energy(E, sigma_k, w)
        out_7x = bz_mean_energy(E, sigma_k, 7.0 * w)

        assert out_7x.E_TA == pytest.approx(out_1x.E_TA, abs=ATOL_PYTHON_FP64_TIGHT)
        assert out_7x.E_TA_std == pytest.approx(out_1x.E_TA_std, abs=ATOL_PYTHON_FP64_TIGHT)
        assert out_7x.s_twist == pytest.approx(out_1x.s_twist, abs=ATOL_PYTHON_FP64_TIGHT)

    # ---------- T3 (dual-path) ----------

    @pytest.mark.light
    def test_T3_dual_path_hand_reference(self):
        """T3: the helper output equals the hand-computed
        reference on a small fixture. This pins the pure-math helper that both the
        live MPI and the offline serial paths call."""
        E = np.array([-1.2, -0.8, -1.5])
        sigma_k = np.array([0.03, 0.05, 0.04])
        w = np.array([2.0, 1.0, 1.0])  # symmetry-reduced IBZ style

        out = bz_mean_energy(E, sigma_k, w)
        W = np.array([0.5, 0.25, 0.25])
        np.testing.assert_allclose(out.W, W, atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_ZERO)

        # Hand reference
        E_TA_ref = float(np.sum(W * E))
        sigma_ref = float(np.sqrt(np.sum(W**2 * sigma_k**2)))
        s_twist_ref = float(np.sqrt(np.sum(W * (E - E_TA_ref) ** 2)))

        assert out.E_TA == pytest.approx(E_TA_ref, abs=ATOL_PYTHON_FP64_BZ_MEAN)
        assert out.E_TA_std == pytest.approx(sigma_ref, abs=ATOL_PYTHON_FP64_BZ_MEAN)
        assert out.s_twist == pytest.approx(s_twist_ref, abs=ATOL_PYTHON_FP64_BZ_MEAN)

    # ---------- T4 (BZ-averaged force sum-rule) ----------

    @pytest.mark.light
    def test_T4_force_sum_rule_holds_under_BZ_combine(self):
        """T4: per-twist forces satisfy ``Sum_a F_k[a] = 0``
        (translational sum-rule). The weighted BZ combine is linear, so
        ``Sum_a F_TA[a] = 0`` must hold exactly."""
        N = 3
        n_atoms = 4
        # Build per-twist forces that respect the sum-rule.
        rng = np.random.default_rng(42)
        F_per_twist = np.zeros((N, n_atoms, 3))
        for k in range(N):
            # Generate (n_atoms-1) random vectors and set the last to negate the sum
            comparison_iteration = str(k)  # noqa: F841 - captured by the GDD iteration receipt
            F = rng.standard_normal((n_atoms - 1, 3))
            F_per_twist[k, :-1] = F
            F_per_twist[k, -1] = -np.sum(F, axis=0)
            # Confirm
            np.testing.assert_allclose(
                np.sum(F_per_twist[k], axis=0),
                np.zeros(3, dtype=np.float64),
                atol=ATOL_PYTHON_FP64_LOOSE,
                rtol=RTOL_PYTHON_FP64_ZERO,
            )
        sigma_k = 0.01 * np.ones_like(F_per_twist)
        w = np.array([3.0, 1.0, 2.0])

        out = bz_mean_force(F_per_twist, sigma_k, w)
        # The BZ-averaged force satisfies the translational sum-rule exactly.
        np.testing.assert_allclose(
            np.sum(out.F_TA, axis=0), np.zeros(3, dtype=np.float64), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )

    # ---------- TestBZMeanInvariants tests ----------

    @pytest.mark.light
    def test_bz_mean_energy_n_twist_1_equals_input(self):
        """At ``N_twist=1`` ``E_TA == E_0`` exactly, ``sigma(E_TA) ==
        sigma_0`` exactly, and ``s_twist == 0`` exactly. The aggregator
        must not perturb a single-twist input."""
        out = bz_mean_energy(E_per_twist=[-1.2345], sigma_per_twist=[0.0017], weights=[1.0])
        assert out.E_TA == pytest.approx(-1.2345, abs=ATOL_PYTHON_FP64_TIGHT)
        assert out.E_TA_std == pytest.approx(0.0017, abs=ATOL_PYTHON_FP64_TIGHT)
        assert out.s_twist == 0.0

    @pytest.mark.light
    def test_bz_mean_energy_n_identical_replicas_equals_single(self):
        """N replicas of the SAME ``(E_k, sigma_k)`` with equal weight
        reproduce the single-twist result for ``E_TA``, give ``sigma`` ==
        ``sigma_0 / sqrt(N)`` (independent MC chains combine in quadrature),
        and ``s_twist == 0``. Pins the "Bloch-replica fold" identity --
        artificially N-tiling an IBZ twist matches the unfolded mesh.
        """
        N = 4
        E0, sig0 = -7.5, 0.04
        out = bz_mean_energy(
            E_per_twist=[E0] * N,
            sigma_per_twist=[sig0] * N,
            weights=[1.0] * N,
        )
        assert out.E_TA == pytest.approx(E0, abs=ATOL_PYTHON_FP64_TIGHT)
        assert out.E_TA_std == pytest.approx(sig0 / np.sqrt(N), abs=ATOL_PYTHON_FP64_TIGHT)
        assert out.s_twist == pytest.approx(0.0, abs=ATOL_PYTHON_FP64_TIGHT)

    # ---------- T2 -- constant-shift covariance ----------

    @pytest.mark.light
    def test_bz_mean_energy_constant_shift_translates_E_TA_only(self):
        """Shifting every ``E_k`` by ``C`` shifts ``E_TA`` by exactly
        ``C`` and leaves both the MC error ``sigma(E_TA)`` and the
        dispersion ``s_twist`` invariant. Verifying covariance under a
        rigid shift -- the canonical defence against accidental subtraction
        of a reference energy.
        """
        rng = np.random.default_rng(7)
        N = 5
        E = rng.standard_normal(N) - 3.0
        sigma = 0.01 + 0.005 * rng.random(N)
        w = rng.random(N) + 0.1

        out = bz_mean_energy(E, sigma, w)
        shift = 42.7
        out_shifted = bz_mean_energy(E + shift, sigma, w)

        assert out_shifted.E_TA == pytest.approx(out.E_TA + shift, abs=ATOL_PYTHON_FP64_LOOSE)
        assert out_shifted.E_TA_std == pytest.approx(out.E_TA_std, abs=ATOL_PYTHON_FP64_BZ_MEAN)
        assert out_shifted.s_twist == pytest.approx(out.s_twist, abs=ATOL_PYTHON_FP64_LOOSE)

    # ---------- T3 -- dual-path consistency ----------

    @pytest.mark.light
    def test_bz_mean_energy_matches_hand_weighted_sum(self):
        """The aggregator output equals the hand-computed weighted sum
        ``Sum_k W_k E_k`` (mean) and ``sqrt(Sum_k W_k^2 sigma_k^2)`` (error).
        Pins the rule that the per-twist sigmas
        combine in quadrature, NOT via a pooled jackknife."""
        rng = np.random.default_rng(11)
        N = 7
        E = rng.standard_normal(N)
        sigma = 0.01 + 0.02 * rng.random(N)
        w_raw = rng.random(N) + 0.05
        W = w_raw / np.sum(w_raw)

        expected_E = np.sum(W * E)
        expected_sigma = np.sqrt(np.sum((W**2) * (sigma**2)))
        out = bz_mean_energy(E, sigma, w_raw)
        np.testing.assert_allclose(out.W, W, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO)
        assert out.E_TA == pytest.approx(expected_E, abs=ATOL_PYTHON_FP64_BZ_MEAN)
        assert out.E_TA_std == pytest.approx(expected_sigma, abs=ATOL_PYTHON_FP64_BZ_MEAN)

    @pytest.mark.light
    def test_bz_mean_energy_s_twist_diagnoses_zero_MC_error_split(self):
        """Toy example: two twists with ``(E_1, E_2) =
        (0, 10)`` at zero MC error and equal weight. Correct behaviour:
        ``E_TA == 5``, ``sigma(E_TA) == 0`` (quadrature combine), and the
        dispersion ``s_twist == 5`` is reported as the physical finite-size
        diagnostic. A pooled jackknife would have inflated ``sigma`` to
        ``~5`` -- the test pins that we do NOT pool.
        """
        out = bz_mean_energy(E_per_twist=[0.0, 10.0], sigma_per_twist=[0.0, 0.0], weights=[1.0, 1.0])
        assert out.E_TA == pytest.approx(5.0, abs=ATOL_PYTHON_FP64_TIGHT)
        assert out.E_TA_std == pytest.approx(0.0, abs=ATOL_PYTHON_FP64_TIGHT)
        assert out.s_twist == pytest.approx(5.0, abs=ATOL_PYTHON_FP64_TIGHT)

    # ---------- T4 -- force sum-rule preservation ----------

    @pytest.mark.light
    def test_bz_mean_force_preserves_sum_rule(self):
        """At a high-symmetry configuration every per-twist force tensor
        already obeys ``sum_a F_k[a, :] == 0``. The BZ mean is a linear
        combination of the per-twist tensors so the BZ-averaged force
        inherits the sum rule exactly. Catches any future non-linear step
        (per-atom abs / norm / clip) at the aggregator layer.
        """
        rng = np.random.default_rng(13)
        N_twist, n_atoms = 4, 3
        per_twist_F = np.zeros((N_twist, n_atoms, 3), dtype=np.float64)
        for k in range(N_twist):
            F = rng.standard_normal((n_atoms, 3))
            # Project out the centre of mass: now sum_a F[a, :] == 0.
            F -= np.mean(F, axis=0, keepdims=True)
            per_twist_F[k] = F
        per_twist_F_std = 0.01 * np.ones_like(per_twist_F)
        weights = rng.random(N_twist) + 0.1

        out = bz_mean_force(per_twist_F, per_twist_F_std, weights)
        sum_a_F_TA = np.sum(out.F_TA, axis=0)
        np.testing.assert_allclose(sum_a_F_TA, np.zeros(3), atol=ATOL_PYTHON_FP64_MEDIUM, rtol=RTOL_PYTHON_FP64_ZERO)

    @pytest.mark.light
    def test_bz_mean_force_matches_hand_weighted_sum(self):
        """``F_TA[a, d]`` equals ``Sum_k W_k F_k[a, d]``, ``sigma(F_TA)[a, d]
        == sqrt(Sum_k W_k^2 sigma_k[a, d]^2)``. The aggregator must not
        re-symmetrise the tensor or fold ``sigma`` through a non-linear
        accumulator (which would break BZ-linearity for downstream
        analyses)."""
        rng = np.random.default_rng(19)
        N_twist, n_atoms = 5, 2
        per_twist_F = rng.standard_normal((N_twist, n_atoms, 3))
        per_twist_F_std = 0.005 + 0.005 * rng.random((N_twist, n_atoms, 3))
        w_raw = rng.random(N_twist) + 0.05
        W = w_raw / np.sum(w_raw)

        expected_F = np.einsum("k,kad->ad", W, per_twist_F)
        expected_sigma = np.sqrt(np.einsum("k,kad->ad", W**2, per_twist_F_std**2))
        out = bz_mean_force(per_twist_F, per_twist_F_std, w_raw)
        np.testing.assert_allclose(out.F_TA, expected_F, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(out.F_TA_std, expected_sigma, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO)


# ---------- N_twist=1 identity for the full estimator quartet ----------


_LI_GAMMA_SAVE = qe_example_files_root() / "Li_gamma" / "out" / "li.save"
_li_gamma_available = (_LI_GAMMA_SAVE / "data-file-schema.xml").is_file() and (_LI_GAMMA_SAVE / "wfc1.hdf5").is_file()
requires_li_gamma = pytest.mark.skipif(not _li_gamma_available, reason="Li_gamma QE fixture not present")


class TestRunTwistAverageEstimatorQuartetIdentity:
    """N_twist=1 identity for every aggregated estimator (E / aF / aT / gF).

    Extends the plain-E byte-identity above to the observables added by
    ``compute_force`` / ``compute_stress`` / ``compute_param_grad``: at
    ``N_twist = 1`` the orchestrator must reproduce a standalone ``MCMC``
    chain's ``get_E`` / ``get_aF`` / ``get_aT`` / ``get_gF`` exactly (same
    seed, same walker count, same derivative caches -> identical samples;
    the aggregation collapses to the identity). A periodic Hamiltonian is
    required because ``get_aT`` is PBC-only; the committed ``Li_gamma`` QE
    fixture is converted in-process via ``run_qe2jqmc`` (kind='pgto').
    """

    @pytest.mark.heavy
    @requires_li_gamma
    def test_n_twist_1_identity_E_aF_aT_gF(self, tmp_path):
        """N twist 1 identity E aF aT gF."""
        from jqmc._twist_average import run_twist_average_from_hamiltonians
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jqmc_mcmc import MCMC
        from jqmc.jqmc_qe import run_qe2jqmc

        ham_file = tmp_path / "li_gamma_pgto.h5"
        run_qe2jqmc(
            {"qe_save_dir": str(_LI_GAMMA_SAVE), "kind": "pgto", "upf_paths": [], "ecutwfc": None, "alpha_min": None},
            {"j1_parameter": None, "j2_parameter": 1.0},
            str(ham_file),
        )
        hamiltonian_data = Hamiltonian_data.load_from_hdf5(str(ham_file))

        num_walkers = 2
        num_mcmc_steps = 30
        num_warmup = 10
        num_bins = 5
        seed = 12345

        # Reference: plain MCMC with the same derivative caches the driver
        # enables for (compute_force | compute_stress, compute_param_grad).
        ref_mcmc = MCMC(
            hamiltonian_data=hamiltonian_data,
            mcmc_seed=seed,
            num_walkers=num_walkers,
            comput_position_deriv=True,
            comput_stress=True,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=True,
        )
        ref_mcmc.run(num_mcmc_steps=num_mcmc_steps)
        E_ref, sigma_ref, _, _ = ref_mcmc.get_E(num_mcmc_warmup_steps=num_warmup, num_mcmc_bin_blocks=num_bins)
        F_ref, F_std_ref = ref_mcmc.get_aF(num_mcmc_warmup_steps=num_warmup, num_mcmc_bin_blocks=num_bins)
        T_ref, T_std_ref = ref_mcmc.get_aT(num_mcmc_warmup_steps=num_warmup, num_mcmc_bin_blocks=num_bins)
        gF_ref, gF_std_ref = ref_mcmc.get_gF(
            num_mcmc_warmup_steps=num_warmup,
            num_mcmc_bin_blocks=num_bins,
            blocks=hamiltonian_data.wavefunction_data.get_variational_blocks(state_token=("test", 0)),
        )

        result = run_twist_average_from_hamiltonians(
            hamiltonian_list=[hamiltonian_data],
            weights=[1.0],
            mcmc_seed=seed,
            num_mcmc_steps=num_mcmc_steps,
            num_walkers=num_walkers,
            num_mcmc_warmup_steps=num_warmup,
            num_mcmc_bin_blocks=num_bins,
            compute_force=True,
            compute_stress=True,
            compute_param_grad=True,
        )

        assert result.energy.E_TA == pytest.approx(float(E_ref), abs=ATOL_PYTHON_FP64_LOOSE)
        assert result.energy.E_TA_std == pytest.approx(float(sigma_ref), abs=ATOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(
            result.force.F_TA, np.asarray(F_ref), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            result.force.F_TA_std, np.asarray(F_std_ref), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            result.stress.F_TA, np.asarray(T_ref), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            result.stress.F_TA_std, np.asarray(T_std_ref), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            result.param_grad.F_TA, np.ravel(np.asarray(gF_ref)), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            result.param_grad.F_TA_std,
            np.ravel(np.asarray(gF_std_ref)),
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        # Per-twist views are the same numbers (N_twist=1).
        np.testing.assert_allclose(
            result.per_twist_T[0], np.asarray(T_ref), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            result.per_twist_gF[0], np.ravel(np.asarray(gF_ref)), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )


class TestTwistAverageWriterStreaming:
    """The incremental writer emits exactly what the batch entry point does.

    The converter cannot hold every twist in memory: one B-spline
    control-point table of a large production cell is 25.2 GiB, so holding
    4 twists keeps ~100 GiB alive. ``TwistAverageWriter`` writes each
    ``/twists/twist_*`` group as soon as it exists and lets the caller drop the
    arrays. Only the ORDER and LIFETIME change, so the file must be identical
    dataset-for-dataset and attr-for-attr -- this test is what makes that a
    contract rather than an intention.
    """

    @pytest.mark.light
    def test_incremental_writer_matches_batch_entry_point(self, tmp_path):
        """Incremental writer matches batch entry point."""
        import os

        import h5py

        from jqmc._twist_average import TwistAverageWriter, save_twist_average_input
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H_ae_ccpvdz_cart.h5")
        structure_data, _, _mos, _, geminal_mo_data, coulomb_potential_data = read_trexio_file(
            trexio_file=fixture, store_tuple=True
        )
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp"),
            jastrow_three_body_data=None,
        )
        template_ham = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data),
        )
        template_ham.sanity_check()

        rng = np.random.default_rng(42)
        up0 = np.asarray(geminal_mo_data.orb_data_up_spin.mo_coefficients, dtype=np.complex128)
        shape = up0.shape
        ups = [up0, up0 + 0.001 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))]
        n_dn = int(geminal_mo_data.num_electron_dn)
        if geminal_mo_data.time_reversal:
            dns = [np.conj(u[:n_dn]) for u in ups]
        else:
            dns = [np.asarray(geminal_mo_data.orb_data_dn_spin.mo_coefficients, dtype=np.complex128)] * 2
        # A per-twist composite child exercises the heavy payload branch.
        children = [("bspline_complex", rng.standard_normal((3, 40)) + 1j * rng.standard_normal((3, 40))) for _ in range(2)]
        k_points = [None, (0.25, 0.0, 0.0)]
        weights = [1.0, 2.0]

        batch_path = str(tmp_path / "batch.h5")
        save_twist_average_input(
            filepath=batch_path,
            template_hamiltonian=template_ham,
            k_points=k_points,
            weights=weights,
            mo_coefficients_up_per_twist=ups,
            mo_coefficients_dn_per_twist=dns,
            composite_child_per_twist=children,
        )

        stream_path = str(tmp_path / "stream.h5")
        with TwistAverageWriter(stream_path, template_ham, len(k_points)) as writer:
            for idx in range(len(k_points)):
                writer.write_twist(idx, k_points[idx], weights[idx], ups[idx], dns[idx], composite_child=children[idx])

        def _walk(f):
            items: dict = {}
            attrs: dict = {"": dict(f.attrs)}

            def _visit(name, obj):
                items[name] = np.asarray(obj) if isinstance(obj, h5py.Dataset) else None
                attrs[name] = dict(obj.attrs)

            f.visititems(_visit)  # must return None to keep walking
            return items, attrs

        with h5py.File(batch_path, "r") as fa, h5py.File(stream_path, "r") as fb:
            items_a, attrs_a = _walk(fa)
            items_b, attrs_b = _walk(fb)
            assert set(items_a) == set(items_b), "HDF5 layout differs between batch and streamed writers"
            for key in sorted(items_a):
                a, b = items_a[key], items_b[key]
                assert (a is None) == (b is None), f"{key}: dataset/group kind differs"
                if a is None:
                    continue
                assert a.shape == b.shape and a.dtype == b.dtype, f"{key}: shape/dtype differs"
                np.testing.assert_array_equal(a, b, err_msg=f"{key}: values differ")
            assert set(attrs_a) == set(attrs_b)
            for key in sorted(attrs_a):
                assert set(attrs_a[key]) == set(attrs_b[key]), f"{key}: attribute set differs"
                for name in attrs_a[key]:
                    np.testing.assert_array_equal(
                        np.asarray(attrs_a[key][name]),
                        np.asarray(attrs_b[key][name]),
                        err_msg=f"{key}.{name}: attribute differs",
                    )

    @pytest.mark.light
    def test_writer_rejects_incomplete_and_duplicate_twists(self, tmp_path):
        """A half-written file must fail loudly, not load as a valid input."""
        import os

        from jqmc._twist_average import TwistAverageWriter
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H_ae_ccpvdz_cart.h5")
        structure_data, _, _mos, _, geminal_mo_data, coulomb_potential_data = read_trexio_file(
            trexio_file=fixture, store_tuple=True
        )
        template_ham = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(
                jastrow_data=Jastrow_data(
                    jastrow_one_body_data=None,
                    jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
                        jastrow_2b_param=1.0, jastrow_2b_type="exp"
                    ),
                    jastrow_three_body_data=None,
                ),
                geminal_data=geminal_mo_data,
            ),
        )
        up = np.asarray(geminal_mo_data.orb_data_up_spin.mo_coefficients, dtype=np.complex128)
        n_dn = int(geminal_mo_data.num_electron_dn)
        dn = (
            np.conj(up[:n_dn])
            if geminal_mo_data.time_reversal
            else np.asarray(geminal_mo_data.orb_data_dn_spin.mo_coefficients, dtype=np.complex128)
        )

        writer = TwistAverageWriter(str(tmp_path / "partial.h5"), template_ham, 2)
        writer.write_twist(0, None, 1.0, up, dn)
        with pytest.raises(ValueError, match="written twice"):
            writer.write_twist(0, None, 1.0, up, dn)
        with pytest.raises(ValueError, match="never written"):
            writer.close()


@pytest.mark.light
def test_twist_averaged_pressure_row_survives_the_production_parser():
    """The BZ-averaged stress section must publish P_hydro WITH an error bar.

    The twist path called ``_log_stress_section`` without ``P_hydro``, so it
    printed a bare value. ``forces_parser.parse_pressure_table`` requires
    ``val(err)`` for every component and returns None when one is missing --
    which drops the ENTIRE pressure record, Px/Py/Pz included, while the run is
    still recorded as completed. The hydrostatic error bar cannot be recovered
    from the diagonal error bars by quadrature (the normal stresses are
    correlated), so it must be carried through the BZ combine.
    """
    import logging

    from jqmc._twist_average import TwistAverageEnergy, TwistAverageForce, TwistAverageResult
    from jqmc.jqmc_cli import _log_twist_average_result
    from jqmc_workflow.forces_parser import parse_pressure_table

    sigma = np.diag([-0.0031, -0.0042, -0.0053])
    sigma_std = np.diag([0.00011, 0.00012, 0.00013])
    result = TwistAverageResult(
        energy=TwistAverageEnergy(E_TA=-1.5, E_TA_std=0.01, s_twist=0.02, W=np.array([0.5, 0.5])),
        force=None,
        per_twist_E=np.array([-1.4, -1.6]),
        per_twist_E_std=np.array([0.01, 0.01]),
        per_twist_F=None,
        per_twist_F_std=None,
        k_points=[None, (0.25, 0.0, 0.0)],
        stress=TwistAverageForce(F_TA=sigma, F_TA_std=sigma_std),
        P_hydro=TwistAverageForce(F_TA=np.array([0.0042]), F_TA_std=np.array([0.00007])),
    )

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("jqmc")
    handler = _Capture()
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        _log_twist_average_result(result, with_force=False)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)

    text = "\n".join(records)
    rows = parse_pressure_table(text)
    assert rows is not None, f"the BZ-averaged pressure table did not survive the production parser:\n{text}"
    for key in ("Px", "Py", "Pz", "P_hydro"):
        assert key in rows, f"{key} missing from the parsed pressure record"
        assert rows[f"{key}_err"] > 0.0, f"{key} was published without a usable error bar"
    # The hydrostatic error bar must be the covariance-aware one, NOT the
    # quadrature of the diagonal bars (which would be larger here).
    quadrature = float(np.sqrt(np.sum(np.diag(sigma_std) ** 2)) / 3.0)
    assert rows["P_hydro_err"] != pytest.approx(quadrature, rel=RTOL_PYTHON_FP64_LOOSE)


@pytest.mark.light
def test_stress_section_refuses_to_publish_pressure_without_an_error_bar():
    """A missing P_hydro replicate set must be a loud refusal, not a bare row."""
    from jqmc.jqmc_cli import _log_stress_section

    sigma = np.diag([-0.001, -0.002, -0.003])
    sigma_std = np.diag([1.0e-5, 1.0e-5, 1.0e-5])
    with pytest.raises(ValueError, match="P_hydro replicates are unavailable"):
        _log_stress_section(sigma, sigma_std, -np.diag(sigma), np.diag(sigma_std), None)
