"""A rank-local invalid fresh determinant prevents every rank checkpoint write."""

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

from mpi4py import MPI
import numpy as np
import pytest

from jqmc.jqmc_mcmc import MCMC
from tests.test_jqmc_kernel.test_integration_checkpoint_mcmc import _build_hamiltonian

pytestmark = [pytest.mark.light, pytest.mark.with_chdir, pytest.mark.mpi_only]


@pytest.mark.parametrize("rust_enabled", [False, True], ids=["xla", "rust"])
@pytest.mark.parametrize("invalid_state", ["zero-determinant", "nonfinite-coordinate"])
def test_checkpoint_save_rejects_rank_local_invalid_fresh_state_collectively(
    tmp_path, monkeypatch, rust_enabled, invalid_state
):
    """Both sampling ranks receive the same failure before creating any rank file."""
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2, "launch this MPI-only contract with mpirun -np 2"
    rank = comm.Get_rank()
    root = Path(comm.bcast(str(tmp_path) if rank == 0 else None, root=0))
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1" if rust_enabled else "0")
    ham = _build_hamiltonian("H2_ae_ccpvdz_cart.h5", "none")
    driver = MCMC(hamiltonian_data=ham, num_walkers=1, mcmc_seed=12345, mpi_comm_in=comm, world_comm_in=comm)
    if rank == 1:
        down = np.asarray(driver._MCMC__latest_r_dn_carts).copy()
        down[0, 0] = [-115.0, 40.0, 15.0] if invalid_state == "zero-determinant" else [np.nan, 0.0, 0.0]
        driver._MCMC__latest_r_dn_carts = down
    target = root / f"rank-{rank}.h5"
    try:
        driver.save_to_hdf5(str(target))
    except FloatingPointError as error:
        message = str(error)
    except BaseException as error:
        message = f"WRONG_EXCEPTION {type(error).__name__}: {error}"
    else:
        message = "NO_EXCEPTION"
    messages = comm.allgather(message)
    assert len(set(messages)) == 1, messages
    assert all(token in messages[0] for token in ("MCMC", "checkpoint save", "sampling_rank=1", "walker=0")), messages
    assert not any(comm.allgather(target.exists())), "No rank may write before the collective verdict"
