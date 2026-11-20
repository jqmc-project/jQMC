"""Actual MPI twist checkpoints consumed by the single-process tool CLI."""

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
from typer.testing import CliRunner

from jqmc import jqmc_tool
from jqmc._precision import get_tolerance_min
from jqmc._twist_average import run_twist_average_from_hamiltonians
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir, pytest.mark.mpi_only]


def _run_driver_and_tool(tmp_path, monkeypatch, *, mcmc_seed):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    root = Path(comm.bcast(str(tmp_path) if rank == 0 else None, root=0))
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    inputs = Path(__file__).resolve().parents[1] / "trexio_example_files"
    hamiltonians = []
    for name in ("H2_pbc_ae_cart.h5", "H2_pbc_ae_k025_cart.h5"):
        structure, _, _, _, geminal, coulomb = read_trexio_file(str(inputs / name), store_tuple=True)
        hamiltonians.append(
            Hamiltonian_data(
                structure_data=structure,
                coulomb_potential_data=coulomb,
                wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
            )
        )
    with monkeypatch.context() as patch:
        patch.chdir(root)
        expected = run_twist_average_from_hamiltonians(
            hamiltonians,
            [0.25, 0.75],
            mcmc_seed=mcmc_seed,
            num_walkers=2,
            num_mcmc_steps=20,
            num_mcmc_warmup_steps=5,
            num_mcmc_bin_blocks=3,
            world_comm=comm,
            compute_force=True,
            compute_stress=True,
            mcmc_kwargs={"num_mcmc_per_measurement": 1},
            checkpoint_dir=str(root / "checkpoints"),
        )
    comm.Barrier()
    observed = {}
    if rank == 0:
        original_energy = jqmc_tool._echo_energy_estimate
        original_force = jqmc_tool._echo_force_pressure_summary

        def capture_energy(estimate):
            observed["energy"] = np.asarray([estimate["E"], estimate["sigma_E"]])
            original_energy(estimate)

        def capture_force(force, force_std, pressure, pressure_std, labels):
            observed["force"] = np.asarray(force)
            observed["force_std"] = np.asarray(force_std)
            observed["pressure"] = np.asarray([pressure, pressure_std])
            original_force(force, force_std, pressure, pressure_std, labels)

        with monkeypatch.context() as patch:
            patch.setattr(jqmc_tool, "_echo_energy_estimate", capture_energy)
            patch.setattr(jqmc_tool, "_echo_force_pressure_summary", capture_force)
            runner = CliRunner()
            for command in ("compute-energy", "compute-force"):
                result = runner.invoke(jqmc_tool.mcmc_app, [command, str(root / "checkpoints"), "-b", "3", "-w", "5"])
                observed[command] = (result.exit_code, result.output, repr(result.exception))
    return expected, comm.bcast(observed, root=0)


@pytest.mark.parametrize("mcmc_seed", [12345])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_tool_values_match_actual_twist_driver(tmp_path, monkeypatch, mcmc_seed):
    """Real tool CLI energy, force and pressure summaries match an actual two-rank driver.

    Committed H2 Gamma/k025 Cartesian pGTO/MO inputs have one up/one down
    electron and no Jastrow. Weights are 1/4 and 3/4. Seed 12345, two walkers
    per twist and twenty one-update measurements use five warmup measurements
    and three bins, without cond(G) filtering. Both per-twist energy errors
    must be nonzero; no error bar supplies the acceptance tolerance.

    Only rank zero runs the real compute-energy/compute-force commands on the
    driver-written checkpoint directory. Capture the unrounded numbers at
    the real formatter boundary and also execute that formatter. Compare
    energy/error, all force/error components and hydrostatic pressure/error
    using the existing same-mode strict zone policy. This verifies the
    driver/checkpoint/tool route, not independent force or stress physics.
    """
    assert MPI.COMM_WORLD.Get_size() == 2
    expected, observed = _run_driver_and_tool(tmp_path, monkeypatch, mcmc_seed=mcmc_seed)
    for command in ("compute-energy", "compute-force"):
        assert observed[command][0] == 0, observed[command]
    assert "E = " in observed["compute-energy"][1]
    assert "Atomic Forces:" in observed["compute-force"][1]
    assert "P_hydro = " in observed["compute-force"][1]
    assert np.isfinite(expected.per_twist_E_std).all() and np.all(expected.per_twist_E_std > 0)
    references = {
        "energy": np.asarray([expected.energy.E_TA, expected.energy.E_TA_std]),
        "force": np.asarray(expected.force.F_TA),
        "force_std": np.asarray(expected.force.F_TA_std),
        "pressure": np.asarray([expected.P_hydro.F_TA.item(), expected.P_hydro.F_TA_std.item()]),
    }
    atol, rtol = get_tolerance_min(("local_energy", "det_grad_lap", "coulomb"), "strict")
    for field in ("energy", "force", "force_std", "pressure"):
        actual, reference = observed[field], references[field]
        assert actual.shape == reference.shape
        assert np.isfinite(actual).all() and np.isfinite(reference).all()
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, err_msg=field)


@pytest.mark.parametrize("fault", ["reversed-energy-weights", "doubled-force", "shifted-pressure"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_tool_twist_verdict_rejects_reduction_corruption(tmp_path, monkeypatch, fault):
    """The real driver/tool oracle rejects changed energy weights, force or pressure.

    First require the unchanged actual-MPI baseline to pass. Then perturb only
    the postprocessor reduction, leaving its file I/O, CLI, formatter, sampler
    and inputs unchanged. The expected failure is the numerical comparison,
    never an unrelated driver exception. Every perturbation runs on both ranks.
    """
    test_tool_values_match_actual_twist_driver(tmp_path / "baseline", monkeypatch, mcmc_seed=12345)
    combine_energy = jqmc_tool._combine_twist_energy_estimates
    combine_force = jqmc_tool._twist_force_pressure_summary

    def corrupt_energy(estimates, weights):
        return combine_energy(estimates, weights[::-1])

    def corrupt_force(*args, **kwargs):
        force, force_std, pressure, pressure_std, labels = combine_force(*args, **kwargs)
        if fault == "doubled-force":
            force = force * 2
        else:
            pressure = pressure + 1
        return force, force_std, pressure, pressure_std, labels

    with monkeypatch.context() as patch:
        if fault == "reversed-energy-weights":
            patch.setattr(jqmc_tool, "_combine_twist_energy_estimates", corrupt_energy)
        else:
            patch.setattr(jqmc_tool, "_twist_force_pressure_summary", corrupt_force)
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            test_tool_values_match_actual_twist_driver(tmp_path / "corrupted", monkeypatch, mcmc_seed=12345)
