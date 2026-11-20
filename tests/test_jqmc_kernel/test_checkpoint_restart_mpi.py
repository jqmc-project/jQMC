"""Exact checkpoint preservation on a real two-rank sampling communicator."""

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
from mpi4py import MPI

from jqmc._checkpoint import merge_rank_checkpoints
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.jqmc_mcmc import MCMC
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir, pytest.mark.mpi_only]


@pytest.mark.parametrize("driver_name", ["MCMC", "GFMC_n", "GFMC_t"])
@pytest.mark.gdd_electron_configuration(
    mode="required", position_source="production-driver-generated", reason="Two walkers per rank, six fixed-seed measurements."
)
def test_two_rank_restart_preserves_state_and_reduced_energy(tmp_path, driver_name):
    """Two real MPI ranks restore their own walkers, RNG, history and reduced energy exactly.

    Input is committed H2_ae_ccpvdz_cart.h5: open all-electron H2, Cartesian
    GTO/MO determinant, one up/one down electron, no Jastrow. Seed 12345 uses
    the production rank-derived streams, two walkers per rank and six stored
    measurements. No cond(G) filtering is applied. MCMC and load-balanced GFMC
    use one update per measurement; conventional GFMC uses tau=0.1. GFMC
    collects no historical weight window. All three samplers actually run.

    The same two-rank communicator writes, merges and reloads its checkpoint.
    The two-block, zero-warmup energy-estimator tuple before and after reload
    must be finite and bit-exact, as must each rank's positions and JAX keys.
    All three histories are preserved: rank-local for MCMC, root-owned for GFMC;
    GFMC non-root ranks retain empty storage and compare the broadcast root data.
    This compares identical saved data, never an
    independent chain or a statistical confidence interval. It does not claim
    uninterrupted-versus-resumed future trajectories or different-size restart.
    """
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2, "launch this MPI-only contract with mpirun -np 2"
    rank = comm.Get_rank()
    root = Path(comm.bcast(str(tmp_path) if rank == 0 else None, root=0))
    source = Path(__file__).resolve().parents[1] / "trexio_example_files" / "H2_ae_ccpvdz_cart.h5"
    structure, _, _, _, geminal, coulomb = read_trexio_file(str(source), store_tuple=True)
    hamiltonian = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=geminal),
    )
    driver_class = {"MCMC": MCMC, "GFMC_n": GFMC_n, "GFMC_t": GFMC_t}[driver_name]
    options = {"num_mcmc_per_measurement": 1} if driver_name != "GFMC_t" else {"tau": 0.1}
    if driver_name != "MCMC":
        options["num_gfmc_collect_steps"] = 0
    driver = driver_class(hamiltonian_data=hamiltonian, num_walkers=2, mcmc_seed=12345, mpi_comm_in=comm, **options)
    driver.run(num_mcmc_steps=6)
    energy_before = np.asarray(driver.get_E(num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=2))
    pattern = str(root / "rank-{rank}.h5")
    driver.save_to_hdf5(pattern.format(rank=rank))
    comm.Barrier()
    merged = str(root / "restart.h5")
    if rank == 0:
        merge_rank_checkpoints(merged, mpi_size=2, driver_type=driver_name, hamiltonian_data=hamiltonian, tmp_pattern=pattern)
    comm.Barrier()
    restored = driver_class.load_from_hdf5(merged, mpi_comm_in=comm)
    energy_after = np.asarray(restored.get_E(num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=2))
    assert np.isfinite(energy_before).all()
    assert np.isfinite(energy_after).all()
    np.testing.assert_array_equal(energy_after, energy_before)
    prefix = "_MCMC__" if driver_name == "MCMC" else "_"
    for name in ("latest_r_up_carts", "latest_r_dn_carts", "jax_PRNG_key_list", "stored_e_L", "stored_e_L2", "stored_w_L"):
        before = np.asarray(getattr(driver, prefix + name))
        after = np.asarray(getattr(restored, prefix + name))
        if driver_name != "MCMC" and name.startswith("stored_"):
            if rank != 0:
                assert before.shape == after.shape == (0, 1)
            before = comm.bcast(before if rank == 0 else None, root=0)
            after = comm.bcast(after if rank == 0 else None, root=0)
        assert before.size > 0
        assert np.isfinite(before).all()
        assert np.isfinite(after).all()
        np.testing.assert_array_equal(after, before)
    # Rank-derived streams must remain distinct; a loader reading rank_0 twice
    # must not pass simply because both saved states happened to be identical.
    keys = comm.allgather(np.asarray(getattr(restored, prefix + "jax_PRNG_key_list")))
    assert not np.array_equal(keys[0], keys[1])
    assert restored.mcmc_counter == driver.mcmc_counter == 6
