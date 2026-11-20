"""Exercise the production driver guard before JAX joins the MPI world."""

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

from functools import partial
from pathlib import Path

import jax
from mpi4py import MPI
import pytest

from jqmc import jqmc_gfmc, jqmc_mcmc
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.light, pytest.mark.with_chdir, pytest.mark.mpi_only]


@pytest.fixture(scope="module", autouse=True)
def _jax_distributed_init():
    """Leave JAX uninitialized in this dedicated two-rank pytest session."""


@pytest.fixture(scope="module")
def hamiltonian():
    """Load the small molecular H2 fixture used by the MPI checkpoint tests."""
    path = Path(__file__).resolve().parents[1] / "trexio_example_files" / "H2_ae_ccpvdz_cart.h5"
    structure, _, _, _, geminal, coulomb = read_trexio_file(str(path), store_tuple=True)
    return Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
    )


@pytest.mark.parametrize("entry_point", ["mcmc", "gfmc-t", "gfmc-n", "optimize"])
def test_driver_rejects_mpi_jax_rank_mismatch(entry_point, hamiltonian, monkeypatch):
    """Both ranks refuse entry before sampling or optimizer setup."""
    world = MPI.COMM_WORLD
    assert world.Get_size() == 2
    assert jax.process_count() == 1

    driver_class, module = {
        "mcmc": (jqmc_mcmc.MCMC, jqmc_mcmc),
        "gfmc-t": (jqmc_gfmc.GFMC_t, jqmc_gfmc),
        "gfmc-n": (jqmc_gfmc.GFMC_n, jqmc_gfmc),
        "optimize": (jqmc_mcmc.MCMC, jqmc_mcmc),
    }[entry_point]
    # A guard regression on just one rank must fail without stranding that rank
    # at a sampling barrier. The real world size remains visible to the guard.
    driver_options = {"num_mcmc_per_measurement": 1} if entry_point == "gfmc-n" else {}
    driver = driver_class(hamiltonian, num_walkers=2, mpi_comm_in=MPI.COMM_SELF, world_comm_in=world, **driver_options)

    def unexpected_start(*args, **kwargs):
        pytest.fail(f"{entry_point} started driver work on rank {world.Get_rank()} despite the MPI/JAX mismatch")

    if entry_point == "optimize":
        monkeypatch.setattr(driver, "configure_optimization", unexpected_start)
        run = partial(driver.run_optimize, num_mcmc_steps=1, dump_wavefunction=False)
    else:
        monkeypatch.setattr(module, "_init_external_control_file", unexpected_start)
        run = partial(driver.run, num_mcmc_steps=1)

    with pytest.raises(RuntimeError, match=r"MPI/JAX rank-count mismatch: mpi_size=2 vs jax\.process_count\(\)=1"):
        run()
