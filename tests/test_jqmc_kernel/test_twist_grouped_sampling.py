"""Exercise twist-averaged MCMC with two sampling ranks per twist."""

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

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from jqmc._setting import ATOL_PYTHON_FP64_BZ_MEAN, RTOL_PYTHON_FP64_ZERO

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]


def _sample_twists(precision, ranks_per_twist, report_path, inputs):
    """Run the production driver and an independently partitioned reference."""
    import jax
    from mpi4py import MPI

    from jqmc._driver_helpers import _initialize_jax_distributed
    from jqmc._precision import configure
    from jqmc._twist_average import run_twist_average_from_hamiltonians
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import Jastrow_data
    from jqmc.jqmc_mcmc import MCMC
    from jqmc.trexio_wrapper import read_trexio_file
    from jqmc.wavefunction import Wavefunction_data

    configure(precision)
    _initialize_jax_distributed()
    world = MPI.COMM_WORLD
    assert world.Get_size() == jax.process_count() == len(inputs) * ranks_per_twist
    rank = world.Get_rank()
    hamiltonians = []
    for path in inputs:
        structure, _, _, _, geminal, coulomb = read_trexio_file(path, store_tuple=True)
        hamiltonians.append(
            Hamiltonian_data(
                structure_data=structure,
                coulomb_potential_data=coulomb,
                wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
            )
        )

    twist, intra_rank = divmod(rank, ranks_per_twist)
    intra = world.Split(color=twist, key=intra_rank)
    seed, steps, walkers = 12345, 20, 2
    estimator_options = dict(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=3)
    reference = MCMC(
        hamiltonians[twist],
        mcmc_seed=(seed, twist),
        num_walkers=walkers,
        num_mcmc_per_measurement=1,
        mpi_comm_in=intra,
        world_comm_in=world,
    )
    reference.run(num_mcmc_steps=steps)
    mean, error, _, _ = reference.get_E(**estimator_options)
    intra.Free()

    result = run_twist_average_from_hamiltonians(
        hamiltonian_list=hamiltonians,
        weights=[0.25, 0.75],
        mcmc_seed=seed,
        num_walkers=walkers,
        num_mcmc_steps=steps,
        mcmc_kwargs={"num_mcmc_per_measurement": 1},
        world_comm=world,
        **estimator_options,
    )
    rank_results = world.gather(
        {
            "rank": rank,
            "reference": [float(mean), float(error)],
            "per_twist_energy": result.per_twist_E.tolist(),
            "per_twist_error": result.per_twist_E_std.tolist(),
            "aggregate": [result.energy.E_TA, result.energy.E_TA_std, result.energy.s_twist],
        },
        root=0,
    )
    if rank == 0:
        report_path.write_text(json.dumps(rank_results, allow_nan=False))
    world.Barrier()
    jax.distributed.shutdown()
    world.Barrier()
    MPI.Finalize()


@pytest.mark.parametrize("ranks_per_twist", [2])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_twist_driver_matches_independently_grouped_chains(tmp_path, request, ranks_per_twist):
    """Four real ranks agree with two standalone chains and the weighted energy formulas."""
    from tests.test_jqmc_kernel.test_integration_cli_twist_dispatch import _MPIRUN, _cli_env, _mpirun_command

    assert _MPIRUN is not None, "an MPI launcher is required for the grouped-twist contract"
    fixtures = Path(__file__).resolve().parents[1] / "trexio_example_files"
    inputs = [fixtures / "H2_pbc_ae_cart.h5", fixtures / "H2_pbc_ae_k025_cart.h5"]
    report = tmp_path / "twist_results.json"
    n_ranks = len(inputs) * ranks_per_twist
    precision = request.config.getoption("--precision-mode")
    xla_flags = os.environ.get("XLA_FLAGS", "")
    env = _cli_env(
        JAX_PLATFORMS="cpu",
        JQMC_FLAG_RUST_ON_CPU="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        XLA_FLAGS=f"{xla_flags} --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1".strip(),
    )
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    launch = subprocess.run(
        _mpirun_command(n_ranks, [sys.executable, __file__, precision, str(ranks_per_twist), str(report), *map(str, inputs)]),
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert launch.returncode == 0, launch.stdout + launch.stderr
    reports = json.loads(report.read_text())
    assert [row["rank"] for row in reports] == list(range(n_ranks))

    chain_summaries = np.asarray([row["reference"] for row in reports])
    standalone = chain_summaries[::ranks_per_twist]
    np.testing.assert_array_equal(chain_summaries, np.repeat(standalone, ranks_per_twist, axis=0))
    energies, errors = standalone.T
    assert np.isfinite(standalone).all() and energies[0] != energies[1] and np.all(errors > 0)
    weights = np.asarray([0.25, 0.75])
    mean = np.dot(weights, energies)
    error = np.linalg.norm(weights * errors)
    spread = np.sqrt(np.dot(weights, (energies - mean) ** 2))
    expected_aggregate = [mean, error, spread]
    for row in reports:
        np.testing.assert_array_equal(row["per_twist_energy"], energies, err_msg=f"rank {row['rank']} twist energies")
        np.testing.assert_array_equal(row["per_twist_error"], errors, err_msg=f"rank {row['rank']} twist errors")
        np.testing.assert_allclose(
            row["aggregate"], expected_aggregate, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
        )


if __name__ == "__main__":
    mode, group_size, output, *input_files = sys.argv[1:]
    _sample_twists(mode, int(group_size), Path(output), input_files)
