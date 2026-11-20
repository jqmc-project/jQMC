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

import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import toml

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._setting import (
    ATOL_REDUCTION_ORDER,
    ATOL_REPORTED_VALUE_SHORT_FORM,
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_WARMUP_STEPS,
    RTOL_REDUCTION_ORDER,
    RTOL_REPORTED_ERROR_SHORT_FORM,
)
from jqmc import jqmc_cli, output_parser
from jqmc.jqmc_cli import _cli
from jqmc.jqmc_tool import (
    lrdmc_compute_energy,
    lrdmc_extrapolate_energy,
    lrdmc_generate_input,
    mcmc_compute_energy,
    mcmc_generate_input,
    trexio_convert_to,
    vmc_generate_input,
)

trexio_files = ["H2_ecp_ccpvtz_cart.h5"]

# These tests manage their own working directory: each juggles ``os.chdir``
# between the repo root (``root_dir = os.getcwd()``) and a per-test
# ``tmp_path``. Opt out of the autouse per-test CWD isolation in conftest so
# ``os.getcwd()`` still returns the repo root they expect.
pytestmark = [pytest.mark.without_chdir]

# Minimal completion oracle: these subprocess-free CLI smokes check more than
# that ``_cli()`` / the ``jqmc_tool`` helpers run without raising. Parse
# the "E = <mean> +- <std> Ha" line every job type logs on success (via
# ``logger.info``/``typer.echo`` -- captured through pytest's ``capsys``
# since verbosity="low" routes the "jqmc" logger to a stdout StreamHandler,
# see jqmc_cli.py's verbosity setup) and require at least one finite energy
# report per run, mirroring test_integration_cli_twist_dispatch.py's
# returncode/output-content/checkpoint-existence oracle.
_ENERGY_RE = re.compile(r"E\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*\+-")


def _assert_finite_energy_reported(text: str, context: str) -> None:
    matches = _ENERGY_RE.findall(text)
    assert matches, f"no 'E = ... +- ...' energy report found in {context} output:\n{text}"
    for raw in matches:
        e = float(raw)
        assert math.isfinite(e), f"non-finite energy {e!r} reported in {context} output:\n{text}"


def _assert_complete_extraction(
    text: str,
    job_type: str,
    context: str,
    *,
    counter: int,
    first_opt_step: int = 1,
    verbosity: str = "low",
    vmc_sampling: tuple[int, int, int] | None = None,
) -> dict:
    """Require that ``jqmc.output_parser`` extracts every field this run must print.

    Output-parsing policy: check that each expected field is present
    and completely extracted, including quantities fixed by the input (the
    cumulative step counter, the optimization step numbers, the checkpoint
    name). Printed values are never compared with references, because they
    change between versions and backends.
    """
    result = output_parser.parse_jqmc_log(text, job_type)
    assert result["program_ended"], f"{context}: 'Program ends' was not parsed"
    # Every printed number is structured (or a documented non-result such as the basis tables).
    assert result["unparsed_numeric_lines"] == [], f"{context}: unparsed lines {result['unparsed_numeric_lines']}"
    environment = result["environment"]
    for name in ("execution_backend", "precision_mode", "started", "ended"):
        assert environment[name], f"{context}: environment {name}"
    # The build information is printed once per process, the zone detail with verbosity = "high".
    if "git commit = " in text:
        assert environment["jqmc_version"] and environment["git_commit"], f"{context}: build information"
    if verbosity == "high":
        assert environment["precision_zones"], f"{context}: precision zones"
        # ``jax.print_environment_info`` echoes every JAX_*/XLA_* variable of the run.
        echoed = sorted(name for name in os.environ if name.startswith(("JAX_", "XLA_")))
        assert sorted(environment["jax_environment_variables"]) == echoed, f"{context}: JAX/XLA environment"
    explicit = result["input_parameters"]["explicit"]
    assert explicit["control"]["job_type"] == job_type, f"{context}: input echo"
    assert job_type in explicit, f"{context}: [{job_type}] input echo"
    system = result["system"]
    assert system["atoms"] and system["num_electron_up"] is not None, f"{context}: system summary"
    assert result["restart_checkpoint"] == "restart.chk", f"{context}: checkpoint line not parsed"
    assert not result["stopped_by_max_time"] and not result["stopped_by_stop_flag"], f"{context}: unexpected stop"
    assert result["incomplete_twists"] == [], f"{context}: no twist was stopped"
    if job_type == "vmc":
        assert result["opt_step_counter"] == result["total_opt_steps"] == counter, f"{context}: optimization counter"
        steps = result["steps"]
        assert [step["step"] for step in steps] == list(range(first_opt_step, counter + 1)), f"{context}: step records"
        for step in steps:
            for name in (
                "energy",
                "energy_error",
                "max_force",
                "max_force_error",
                "signal_to_noise_ratio",
                "acceptance_ratio",
                "avg_walker_weight",
                "num_variational_parameters",
                "num_optimization_coordinates",
            ):
                assert step[name] is not None and math.isfinite(step[name]), f"{context}: step {step['step']} {name}"
            if vmc_sampling is not None:
                sampling = (step["num_mcmc_steps"], step["num_mcmc_warmup_steps"], step["num_mcmc_bin_blocks"])
                assert sampling == vmc_sampling, f"{context}: step {step['step']} sampling"
            if verbosity == "high":
                _assert_complete_optimizer_step(step, f"{context}: step {step['step']}")
        assert result["optimizer"]["name"] and result["optimizer"]["hyperparameters"], f"{context}: optimizer"
        assert "sampling" in result["opt_timing_breakdown"], f"{context}: optimization timing"
        assert result["dumped_hamiltonians"], f"{context}: no dumped Hamiltonian parsed"
        assert result["dumped_hamiltonians"][-1] == f"hamiltonian_data_opt_step_{counter}.h5", f"{context}: last dump"
        assert result["total_opt_time_sec"] is not None, f"{context}: optimization time"
        return result
    for name in ("energy", "energy_error", "variance", "variance_error", "total_time_sec", "net_time_sec"):
        assert result[name] is not None and math.isfinite(result[name]), f"{context}: {name} not extracted"
    assert result["timing_breakdown"], f"{context}: timing breakdown not extracted"
    if job_type == "mcmc":
        assert result["mcmc_counter"] == result["mcmc_counter_target"] == counter, f"{context}: MCMC counter"
        assert result["acceptance_ratio"] is not None and math.isfinite(result["acceptance_ratio"]), f"{context}: acceptance"
        assert result["avg_walker_weight"] is not None and math.isfinite(result["avg_walker_weight"]), f"{context}: weights"
    else:
        assert result["gfmc_counter"] == result["gfmc_counter_target"] == counter, f"{context}: GFMC counter"
        survived = result["survived_walkers_ratio"]
        assert survived is not None and math.isfinite(survived), f"{context}: survived walkers ratio"
        if verbosity == "high" and job_type == "lrdmc-bra":
            assert result["e_scf_history"], f"{context}: E_scf history"
    return result


def _assert_complete_optimizer_step(step: dict, context: str) -> None:
    """The optimizer details printed with ``verbosity = "high"`` are all extracted."""
    asr = step["asr"]
    for name in ("gamma_plus", "gamma_minus", "selected_gamma", "final_gamma", "scale"):
        assert asr.get(name) is not None and math.isfinite(asr[name]), f"{context}: aSR {name}"
    diagnostics = step["diagnostics"]
    for name in ("coordinates", "metric_diagonal", "sr_solve", "sr_direction", "final_update"):
        assert diagnostics.get(name), f"{context}: diagnostics {name}"
    assert diagnostics["sr_solve"]["iterations"] >= 0 and diagnostics["sr_solve"]["converged"] in (True, False)
    # The same layout is printed in three places; the extracted counts must agree.
    batch = step["optimization_batch"]
    assert batch, f"{context}: optimization batch"
    assert diagnostics["coordinates"]["real"] == batch["real_coordinates"], f"{context}: real coordinates"
    assert batch["real_coordinates"] == step["num_optimization_coordinates"], f"{context}: coordinate count line"
    assert diagnostics["coordinates"]["storage"] == batch["storage_parameters"] == step["num_variational_parameters"]
    assert step["top_coordinates"], f"{context}: top coordinates"
    for row in step["top_coordinates"]:
        assert all(math.isfinite(row[name]) for name in ("force", "force_error", "signal_to_noise", "metric_diag"))
    assert step["blocks"] and all(block["update"] for block in step["blocks"]), f"{context}: optimizer blocks"
    assert step["theta"]["size"] == step["num_variational_parameters"], f"{context}: theta size"
    assert step["parameter_updates"], f"{context}: parameter updates"
    assert step["derivative_sources"], f"{context}: derivative sources"
    assert step["sr_solver_path"] and step["optimizer"], f"{context}: solver path / optimizer"


class TestCLI:
    @pytest.mark.heavy
    @pytest.mark.parametrize("trexio_file", trexio_files)
    def test_jqmc_tool_trexio_conversion(self, tmp_path, trexio_file: str):
        """Test the conversion of TREXIO files to Hamiltonian data."""
        from jqmc.hamiltonians import Hamiltonian_data

        hamiltonian_file = os.path.join(tmp_path, "hamiltonian_data.h5")
        trexio_convert_to(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            hamiltonian_file=hamiltonian_file,
            j1_parmeter=1.0,
            j2_parmeter=1.0,
            j3_basis_type="ao-medium",
        )
        assert os.path.isfile(hamiltonian_file), "trexio_convert_to did not write a hamiltonian_data.h5"
        # Minimal oracle: the written file round-trips through the real loader
        # and describes a non-empty, finite structure/wavefunction (not just
        # a file that exists but is empty or garbage).
        loaded = Hamiltonian_data.load_from_hdf5(hamiltonian_file)
        assert loaded.structure_data.positions.shape[0] > 0
        assert np.all(np.isfinite(loaded.structure_data.positions))
        gem = loaded.wavefunction_data.geminal_data
        assert gem.num_electron_up + gem.num_electron_dn > 0

    @pytest.mark.heavy
    @pytest.mark.parametrize("trexio_file", trexio_files)
    def test_jqmc_cli_run_mcmc(self, tmp_path, monkeypatch, capsys, trexio_file: str):
        """Test the MCMC run command."""
        root_dir = os.getcwd()
        # trexio conversion
        os.chdir(root_dir)
        trexio_convert_to(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            hamiltonian_file=os.path.join(tmp_path, "hamiltonian_data.h5"),
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type="ao-small",
        )
        os.chdir(root_dir)

        # generate input
        os.chdir(root_dir)
        mcmc_generate_input(flag=True, filename=os.path.join(tmp_path, "mcmc_input.toml"), exclude_comment=True)
        with open(os.path.join(tmp_path, "mcmc_input.toml")) as f:
            dict_toml = toml.load(f)
            dict_toml["control"]["restart"] = False
            dict_toml["control"]["hamiltonian_h5"] = "hamiltonian_data.h5"
            dict_toml["control"]["restart_chk"] = "restart.chk"
            dict_toml["mcmc"]["num_mcmc_steps"] = 50
            dict_toml["mcmc"]["num_mcmc_bin_blocks"] = 5
            dict_toml["mcmc"]["num_mcmc_warmup_steps"] = 30
            dict_toml["mcmc"]["atomic_force"] = True
        with open(os.path.join(tmp_path, "mcmc_input.toml"), "w") as f:
            toml.dump(dict_toml, f)
        os.chdir(root_dir)

        # run MCMC
        os.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["jqmc", "mcmc_input.toml"])
        _cli()
        os.chdir(root_dir)
        assert os.path.isfile(os.path.join(tmp_path, "restart.chk")), "MCMC run did not write restart.chk"
        mcmc_output = capsys.readouterr().out
        _assert_finite_energy_reported(mcmc_output, "MCMC run")
        assert "Execution backend = XLA." in mcmc_output
        _assert_complete_extraction(mcmc_output, "mcmc", "MCMC run", counter=50)

        # post MCMC
        os.chdir(root_dir)
        mcmc_compute_energy(restart_chk=os.path.join(tmp_path, "restart.chk"), num_mcmc_bin_blocks=5, num_mcmc_warmup_steps=30)
        tool_output = capsys.readouterr().out
        _assert_finite_energy_reported(tool_output, "mcmc_compute_energy (abs path)")
        tool_result = output_parser.parse_tool_energy(tool_output)
        # No MPC estimate and a single-twist checkpoint here: those fields stay None.
        always = ("energy", "energy_error", "num_mpi_ranks", "num_binned_samples")
        assert all(tool_result[name] is not None for name in always), "mcmc_compute_energy: field not extracted"
        assert all(tool_result[name] is None for name in set(tool_result) - set(always)), "mcmc_compute_energy: MPC/twist"
        os.chdir(root_dir)

        os.chdir(tmp_path)
        mcmc_compute_energy(restart_chk="restart.chk", num_mcmc_bin_blocks=5, num_mcmc_warmup_steps=30)
        _assert_finite_energy_reported(capsys.readouterr().out, "mcmc_compute_energy (relative path)")
        os.chdir(root_dir)

        # run MCMC(restart)
        os.chdir(root_dir)
        with open(os.path.join(tmp_path, "mcmc_input.toml")) as f:
            dict_toml = toml.load(f)
            dict_toml["control"]["restart"] = True
            dict_toml["control"]["hamiltonian_h5"] = None
            dict_toml["control"]["restart_chk"] = "restart.chk"
            dict_toml["mcmc"]["num_mcmc_steps"] = 10
            # Frozen: restored from the checkpoint, so the restart still reports forces (#316).
            del dict_toml["mcmc"]["atomic_force"]
        with open(os.path.join(tmp_path, "mcmc_input.toml"), "w") as f:
            toml.dump(dict_toml, f)
        os.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["jqmc", "mcmc_input.toml"])
        _cli()
        os.chdir(root_dir)
        assert os.path.isfile(os.path.join(tmp_path, "restart.chk")), "MCMC restart run did not write restart.chk"
        restart_output = capsys.readouterr().out
        _assert_finite_energy_reported(restart_output, "MCMC restart run")
        _assert_complete_extraction(restart_output, "mcmc", "MCMC restart run", counter=60)
        assert "Atomic Forces:" in restart_output, "MCMC restart run did not report the forces restored from the checkpoint"

    @pytest.mark.heavy
    @pytest.mark.parametrize("trexio_file", trexio_files)
    def test_jqmc_cli_run_vmc(self, tmp_path, monkeypatch, capsys, trexio_file: str):
        """Test the VMC run command."""
        root_dir = os.getcwd()
        # trexio conversion
        os.chdir(root_dir)
        trexio_convert_to(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            hamiltonian_file=os.path.join(tmp_path, "hamiltonian_data.h5"),
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type="ao-small",
        )
        os.chdir(root_dir)

        # generate input
        os.chdir(root_dir)
        vmc_generate_input(flag=True, filename=os.path.join(tmp_path, "vmc_input.toml"), exclude_comment=True)
        with open(os.path.join(tmp_path, "vmc_input.toml")) as f:
            dict_toml = toml.load(f)
            dict_toml["control"]["restart"] = False
            dict_toml["control"]["hamiltonian_h5"] = "hamiltonian_data.h5"
            dict_toml["control"]["restart_chk"] = "restart.chk"
            dict_toml["vmc"]["num_opt_steps"] = 2
            dict_toml["vmc"]["num_mcmc_steps"] = 50
            dict_toml["vmc"]["num_mcmc_bin_blocks"] = 10
            dict_toml["vmc"]["num_mcmc_warmup_steps"] = 0
            dict_toml["vmc"]["optimizer_kwargs"] = {"method": "sr", "delta": 0.01}
            # The first run prints the optimizer diagnostics, the restart the default log.
            dict_toml["control"]["verbosity"] = "high"
        with open(os.path.join(tmp_path, "vmc_input.toml"), "w") as f:
            toml.dump(dict_toml, f)
        os.chdir(root_dir)

        # run VMC
        os.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["jqmc", "vmc_input.toml"])
        _cli()
        os.chdir(root_dir)
        assert os.path.isfile(os.path.join(tmp_path, "restart.chk")), "VMC run did not write restart.chk"
        vmc_output = capsys.readouterr().out
        _assert_finite_energy_reported(vmc_output, "VMC optimization run")
        _assert_complete_extraction(
            vmc_output, "vmc", "VMC optimization run", counter=2, verbosity="high", vmc_sampling=(50, 0, 10)
        )

        # run VMCopt(restart)
        os.chdir(root_dir)
        with open(os.path.join(tmp_path, "vmc_input.toml")) as f:
            dict_toml = toml.load(f)
            dict_toml["control"]["restart"] = True
            dict_toml["control"]["hamiltonian_chk"] = None
            dict_toml["control"]["restart_chk"] = "restart.chk"
            dict_toml["control"]["verbosity"] = "low"
        with open(os.path.join(tmp_path, "vmc_input.toml"), "w") as f:
            toml.dump(dict_toml, f)
        os.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["jqmc", "vmc_input.toml"])
        _cli()
        os.chdir(root_dir)
        assert os.path.isfile(os.path.join(tmp_path, "restart.chk")), "VMC restart run did not write restart.chk"
        restart_output = capsys.readouterr().out
        _assert_finite_energy_reported(restart_output, "VMC restart run")
        _assert_complete_extraction(
            restart_output, "vmc", "VMC restart run", counter=4, first_opt_step=3, vmc_sampling=(50, 0, 10)
        )

    @pytest.mark.heavy
    @pytest.mark.parametrize("trexio_file", trexio_files)
    def test_jqmc_cli_run_lrdmc(self, tmp_path, monkeypatch, capsys, trexio_file: str):
        """Test the LRDMC run command."""
        root_dir = os.getcwd()

        # alat
        # one LRDMC run per alat value (a run takes a single alat)
        alat_list = [0.2, 0.3, 0.4, 0.5]

        # generate input
        os.chdir(root_dir)
        for alat in alat_list:
            tmp_alat_path = os.path.join(tmp_path, str(alat))
            os.makedirs(tmp_alat_path, exist_ok=True)
            # trexio conversion
            os.chdir(root_dir)
            trexio_convert_to(
                trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
                hamiltonian_file=os.path.join(tmp_alat_path, "hamiltonian_data.h5"),
                j1_parmeter=None,
                j2_parmeter=1.0,
                j3_basis_type="ao-small",
            )
            os.chdir(root_dir)
            lrdmc_generate_input(flag=True, filename=os.path.join(tmp_alat_path, "lrdmc_input.toml"), exclude_comment=True)
            with open(os.path.join(tmp_alat_path, "lrdmc_input.toml")) as f:
                dict_toml = toml.load(f)
                dict_toml["control"]["restart"] = False
                dict_toml["control"]["hamiltonian_h5"] = "hamiltonian_data.h5"
                dict_toml["control"]["restart_chk"] = "restart.chk"
                dict_toml["lrdmc-bra"]["num_mcmc_steps"] = 50
                dict_toml["lrdmc-bra"]["alat"] = alat
                # Required, no default (#690).
                dict_toml["lrdmc-bra"]["num_mcmc_per_measurement"] = {0.2: 90, 0.3: 40, 0.4: 22, 0.5: 14}[alat]
                dict_toml["lrdmc-bra"]["num_gfmc_bin_blocks"] = 10
                dict_toml["lrdmc-bra"]["num_gfmc_warmup_steps"] = 30
                dict_toml["lrdmc-bra"]["num_gfmc_collect_steps"] = 5
                dict_toml["lrdmc-bra"]["atomic_force"] = True
            with open(os.path.join(tmp_alat_path, "lrdmc_input.toml"), "w") as f:
                toml.dump(dict_toml, f)
            os.chdir(root_dir)

            # run LRDMC
            os.chdir(tmp_alat_path)
            monkeypatch.setattr(sys, "argv", ["jqmc", "lrdmc_input.toml"])
            _cli()
            os.chdir(root_dir)
            assert os.path.isfile(os.path.join(tmp_alat_path, "restart.chk")), (
                f"LRDMC run (alat={alat}) did not write restart.chk"
            )
            lrdmc_output = capsys.readouterr().out
            _assert_finite_energy_reported(lrdmc_output, f"LRDMC run (alat={alat})")
            _assert_complete_extraction(lrdmc_output, "lrdmc-bra", f"LRDMC run (alat={alat})", counter=50)

        # post LRDMC (each alat)
        for alat in alat_list:
            tmp_alat_path = os.path.join(tmp_path, str(alat))
            os.chdir(root_dir)
            lrdmc_compute_energy(
                restart_chk=os.path.join(tmp_alat_path, "restart.chk"),
                num_gfmc_bin_block=10,
                num_gfmc_warmup_steps=30,
                num_gfmc_collect_steps=5,
            )
            _assert_finite_energy_reported(capsys.readouterr().out, f"lrdmc_compute_energy (alat={alat}, abs path)")
            os.chdir(tmp_alat_path)
            lrdmc_compute_energy(
                restart_chk="restart.chk", num_gfmc_bin_block=10, num_gfmc_warmup_steps=30, num_gfmc_collect_steps=5
            )
            _assert_finite_energy_reported(capsys.readouterr().out, f"lrdmc_compute_energy (alat={alat}, relative path)")
            os.chdir(root_dir)

        # post LRDMC (extrapolation)
        os.chdir(tmp_path)
        restart_chks = [os.path.join(tmp_path, str(alat), "restart.chk") for alat in alat_list]
        lrdmc_extrapolate_energy(
            restart_chks=restart_chks,
            polynomial_order=1,
            plot_graph=False,
            save_graph=None,
            num_gfmc_bin_block=10,
            num_gfmc_warmup_steps=30,
            num_gfmc_collect_steps=5,
        )
        extrapolation_output = capsys.readouterr().out
        _assert_finite_energy_reported(extrapolation_output, "lrdmc_extrapolate_energy")
        extrapolation = output_parser.parse_lrdmc_extrapolation(extrapolation_output)
        assert extrapolation["extrapolated_energy"] is not None and extrapolation["extrapolated_energy_error"] is not None
        assert extrapolation["polynomial_order"] == 1
        assert sorted(row["alat"] for row in extrapolation["per_alat_results"]) == sorted(alat_list)
        os.chdir(root_dir)

        # run LRDMC(restart)
        os.chdir(root_dir)
        for alat in alat_list:
            tmp_alat_path = os.path.join(tmp_path, str(alat))
            with open(os.path.join(tmp_alat_path, "lrdmc_input.toml")) as f:
                dict_toml = toml.load(f)
                dict_toml["control"]["restart"] = True
                dict_toml["control"]["hamiltonian_h5"] = None
                dict_toml["control"]["restart_chk"] = "restart.chk"
                dict_toml["lrdmc-bra"]["num_mcmc_steps"] = 10
                # Frozen: restored from the checkpoint, so the restart still reports forces (#316).
                del dict_toml["lrdmc-bra"]["atomic_force"]
            with open(os.path.join(tmp_alat_path, "lrdmc_input.toml"), "w") as f:
                toml.dump(dict_toml, f)
            os.chdir(tmp_alat_path)
            monkeypatch.setattr(sys, "argv", ["jqmc", "lrdmc_input.toml"])
            _cli()
            os.chdir(root_dir)
            assert os.path.isfile(os.path.join(tmp_alat_path, "restart.chk")), (
                f"LRDMC restart run (alat={alat}) did not write restart.chk"
            )
            restart_output = capsys.readouterr().out
            _assert_finite_energy_reported(restart_output, f"LRDMC restart run (alat={alat})")
            _assert_complete_extraction(restart_output, "lrdmc-bra", f"LRDMC restart run (alat={alat})", counter=60)
            assert "Atomic Forces:" in restart_output, f"LRDMC restart (alat={alat}) did not report the restored forces"


# --------------------------------------------------------------------------- #
# Deterministic source -> text -> consumer oracle for the final CLI emission.
#
# Everything above asserts only that the reported energy is FINITE. A sign flip
# on the headline energy (``E_mean`` -> ``-E_mean``), a mean/std transposition,
# a swapped force component or a wrong unit label all keep the output finite and
# therefore passed the entire registered CLI evidence set (AUD2-005r). Here the
# estimators are replaced by fixed sentinels, so the exact mapping from
# estimator return value to printed field is pinned -- and the force table is
# read back with the REAL downstream consumer
# (``jqmc_workflow.forces_parser``), which is what actually republishes these
# numbers into workflow state.
# --------------------------------------------------------------------------- #

_STUB_E = -1.23456
_STUB_E_STD = 0.00789
_STUB_VAR = 2.34567
_STUB_VAR_STD = 0.00123
_STUB_F = np.array([[0.11, -0.22, 0.33], [-0.44, 0.55, -0.66]], dtype=np.float64)
_STUB_F_STD = np.array([[0.0011, 0.0022, 0.0033], [0.0044, 0.0055, 0.0066]], dtype=np.float64)


class _SentinelMCMC:
    """Stand-in for :class:`jqmc.jqmc_mcmc.MCMC` returning fixed estimator values."""

    def __init__(self, hamiltonian_data=None, **kwargs):
        self.hamiltonian_data = hamiltonian_data
        self.compute_log_WF_param_deriv = False
        self.mcmc_counter = 0

    def run(self, num_mcmc_steps=0, **kwargs):
        # Count the requested steps as a real driver does; the CLI reads the
        # counter to decide whether enough samples exist for the final output.
        self.mcmc_counter += num_mcmc_steps
        return None

    def get_E(self, **kwargs):
        return _STUB_E, _STUB_E_STD, _STUB_VAR, _STUB_VAR_STD

    def get_aF(self, **kwargs):
        return _STUB_F, _STUB_F_STD

    def save_to_hdf5(self, filename):
        # A real (if empty) HDF5 container: the CLI merges the per-rank files
        # after emitting the final output, and this test is about the emission.
        import h5py

        with h5py.File(filename, "w"):
            pass


class TestCLIFinalOutputMapping:
    @pytest.mark.light
    def test_mcmc_final_output_fields_match_the_estimator_values(self, tmp_path, monkeypatch, capsys):
        """Every printed final-output field equals the estimator value it claims.

        RED for: ``E_mean`` -> ``-E_mean`` (the demonstrated escape), mean/std
        transposition, force component transposition, and a wrong Ha / Ha^2
        unit tag -- none of which disturb finiteness.
        """
        from jqmc_workflow.forces_parser import parse_force_table

        root_dir = os.getcwd()
        trexio_convert_to(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ecp_ccpvtz_cart.h5"
            ),
            hamiltonian_file=os.path.join(tmp_path, "hamiltonian_data.h5"),
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type="ao-small",
        )
        os.chdir(root_dir)
        mcmc_generate_input(flag=True, filename=os.path.join(tmp_path, "mcmc_input.toml"), exclude_comment=True)
        with open(os.path.join(tmp_path, "mcmc_input.toml")) as f:
            dict_toml = toml.load(f)
        dict_toml["control"]["restart"] = False
        dict_toml["control"]["hamiltonian_h5"] = "hamiltonian_data.h5"
        dict_toml["control"]["restart_chk"] = "restart.chk"
        dict_toml["mcmc"]["num_mcmc_steps"] = 2
        dict_toml["mcmc"]["num_mcmc_bin_blocks"] = 1
        dict_toml["mcmc"]["num_mcmc_warmup_steps"] = 0
        dict_toml["mcmc"]["atomic_force"] = True
        with open(os.path.join(tmp_path, "mcmc_input.toml"), "w") as f:
            toml.dump(dict_toml, f)

        monkeypatch.setattr(jqmc_cli, "MCMC", _SentinelMCMC)
        # The checkpoint merge consumes real per-rank driver state; this gate is
        # about what the CLI PRINTS, so the merge is stubbed out rather than fed
        # a synthetic driver.
        monkeypatch.setattr(jqmc_cli, "merge_rank_checkpoints", lambda **kwargs: None)
        os.chdir(tmp_path)
        try:
            monkeypatch.setattr(sys, "argv", ["jqmc", "mcmc_input.toml"])
            _cli()
        finally:
            os.chdir(root_dir)
        out = capsys.readouterr().out

        # Scalars: exact text, so sign, magnitude, mean/std order and unit tag
        # are all pinned at once.
        assert f"Total Energy: E = {_STUB_E:.5f} +- {_STUB_E_STD:.5f} Ha." in out, out
        assert f"Variance: Var = {_STUB_VAR:.5f} +- {_STUB_VAR_STD:.5f} Ha^2." in out, out

        # Forces: read back through the real downstream consumer.
        rows = parse_force_table(out)
        assert rows is not None, f"the force table did not survive the production parser:\n{out}"
        assert len(rows) == _STUB_F.shape[0]
        for i, row in enumerate(rows):
            for j, comp in enumerate(("Fx", "Fy", "Fz")):
                assert row[comp] == pytest.approx(_STUB_F[i, j], abs=ATOL_REPORTED_VALUE_SHORT_FORM), (
                    f"atom {i} component {comp}: printed {row[comp]} but the estimator returned {_STUB_F[i, j]}"
                )
                # The short form prints the error to 2 significant digits, so
                # compare relatively -- still tight enough that a swapped or
                # mis-scaled error bar (the components differ by 2-6x) fails.
                assert row[f"{comp}_err"] == pytest.approx(_STUB_F_STD[i, j], rel=RTOL_REPORTED_ERROR_SHORT_FORM), (
                    f"atom {i} component {comp} error bar mismatch"
                )

        # The dict-returning parser used by non-workflow drivers extracts the
        # same final output (presence and completeness only; the values are
        # pinned above against the estimator).
        parsed = output_parser.parse_mcmc_log(out)
        assert parsed["program_ended"], "the end of the run was not parsed"
        for name in ("energy", "energy_error", "variance", "variance_error"):
            assert parsed[name] is not None, f"{name} was not extracted"
        assert parsed["atomic_forces"] is not None and len(parsed["atomic_forces"]) == _STUB_F.shape[0]

    @pytest.mark.heavy
    @pytest.mark.parametrize("job_type", ["lrdmc-bra", "lrdmc-tau"])
    def test_lrdmc_final_output_fields_match_the_checkpoint_recomputation(self, tmp_path, monkeypatch, capsys, job_type):
        """The LRDMC final output reports the estimators of the run it wrote.

        lrdmc-bra and lrdmc-tau print their final output from their own copies
        of the emission code, which the MCMC sentinel test above does not
        reach. A real tiny run is made here, and every printed energy, variance
        and force field is compared with an independent NumPy recomputation
        (rolling weight products and a leave-one-bin-out jackknife) from the raw
        observables the same run stored in its checkpoint. Each printed decimal
        must be the reference rounded at its own last printed digit.

        RED for: ``E_mean`` -> ``-E_mean``, mean/std transposition and a force
        component transposition in either LRDMC print block.
        """
        import h5py
        from jqmc_workflow.forces_parser import parse_force_table

        warmup, bins, collect = GFMC_MIN_WARMUP_STEPS, GFMC_MIN_BIN_BLOCKS, GFMC_MIN_COLLECT_STEPS
        root_dir = os.getcwd()
        trexio_convert_to(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ecp_ccpvtz_cart.h5"
            ),
            hamiltonian_file=os.path.join(tmp_path, "hamiltonian_data.h5"),
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type="ao-small",
        )
        os.chdir(root_dir)
        section = {
            "num_mcmc_steps": warmup + bins + collect + 20,
            "alat": 0.4,
            "non_local_move": "tmove",
            "atomic_force": True,
            "num_gfmc_warmup_steps": warmup,
            "num_gfmc_bin_blocks": bins,
            "num_gfmc_collect_steps": collect,
        }
        section.update({"num_mcmc_per_measurement": 22} if job_type == "lrdmc-bra" else {"tau": 0.05})
        dict_toml = {
            "control": {
                "job_type": job_type,
                "mcmc_seed": 34456,
                "number_of_walkers": 2,
                "max_time": 86400,
                "restart": False,
                "restart_chk": "restart.chk",
                "hamiltonian_h5": "hamiltonian_data.h5",
                "verbosity": "low",
            },
            job_type: section,
        }
        with open(os.path.join(tmp_path, "lrdmc_input.toml"), "w") as f:
            toml.dump(dict_toml, f)

        os.chdir(tmp_path)
        try:
            monkeypatch.setattr(sys, "argv", ["jqmc", "lrdmc_input.toml"])
            _cli()
        finally:
            os.chdir(root_dir)
        out = capsys.readouterr().out

        # Independent reference from the raw stored observables of this run.
        with h5py.File(os.path.join(tmp_path, "restart.chk"), "r") as f:
            ranks = [name for name in f if name.startswith("rank_")]
            assert ranks == ["rank_0"], f"a single-process run must store one rank, found {ranks}"
            obs = {
                name: np.asarray(f[f"rank_0/observables/{name}"][()], dtype=np.float64)
                for name in ("w_L", "e_L", "e_L2", "force_HF", "force_PP", "E_L_force_PP")
            }
        num_rows = obs["w_L"].shape[0] - collect
        # Accumulated weight of row i = product of the bare weights i .. i + collect - 1;
        # the observable of that row is the one measured at i + collect.
        acc_w = np.stack([np.prod(obs["w_L"][i : i + collect], axis=0) for i in range(num_rows)])[warmup:]

        def binned(x):
            x = x[collect:][warmup:]
            weighted = acc_w.reshape(acc_w.shape + (1,) * (x.ndim - acc_w.ndim)) * x
            return np.stack([np.sum(block, axis=(0, 1)) for block in np.array_split(weighted, bins, axis=0)])

        w_b = np.stack([np.sum(block) for block in np.array_split(acc_w, bins, axis=0)])
        w_e_b, w_e2_b = binned(obs["e_L"]), binned(obs["e_L2"])
        w_fhf_b, w_fpp_b, w_efpp_b = binned(obs["force_HF"]), binned(obs["force_PP"]), binned(obs["E_L_force_PP"])

        def leave_one_out(x_b):
            return (np.sum(x_b, axis=0) - x_b) / (np.sum(w_b) - w_b).reshape((bins,) + (1,) * (x_b.ndim - 1))

        e_jk = leave_one_out(w_e_b)
        var_jk = leave_one_out(w_e2_b) - e_jk**2
        force_jk = -leave_one_out(w_fhf_b) - 2.0 * (leave_one_out(w_efpp_b) - e_jk[:, None, None] * leave_one_out(w_fpp_b))

        def jackknife(samples):
            return np.mean(samples, axis=0), np.sqrt(bins - 1) * np.std(samples, axis=0)

        ref_e, ref_e_std = jackknife(e_jk)
        ref_var, ref_var_std = jackknife(var_jk)
        ref_f, ref_f_std = jackknife(force_jk)

        def assert_rounding_of(printed, half_place, reference, what):
            bound = half_place + ATOL_REDUCTION_ORDER + RTOL_REDUCTION_ORDER * abs(reference)
            assert abs(float(printed) - reference) <= bound, (
                f"{job_type} {what}: printed {printed} is not the checkpoint estimator {reference!r} "
                f"rounded at the printed digit\n{out}"
            )

        # Scalars: the fixed-point lines carry their own printed precision.
        for label, unit, ref_mean, ref_std in (
            ("Total Energy: E", "Ha", ref_e, ref_e_std),
            ("Variance: Var", r"Ha\^2", ref_var, ref_var_std),
        ):
            m = re.search(rf"{label} = ([-+]?\d+\.(\d+)) \+- ([-+]?\d+\.(\d+)) {unit}\.", out)
            assert m is not None, f"{job_type}: no '{label} = ... +- ... {unit}.' final output line\n{out}"
            assert_rounding_of(m.group(1), 0.5 * 10.0 ** -len(m.group(2)), ref_mean, f"{label} mean")
            assert_rounding_of(m.group(3), 0.5 * 10.0 ** -len(m.group(4)), ref_std, f"{label} error bar")

        # Forces: values read back by the real downstream consumer; the printed
        # precision of every field is taken from its own short-form token.
        rows = parse_force_table(out)
        assert rows is not None and len(rows) == ref_f.shape[0], f"{job_type}: force table not parsed\n{out}"
        header = out.index("Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)")
        table_lines = [line.split() for line in out[header:].splitlines()[2 : 2 + ref_f.shape[0]]]
        short_form = re.compile(r"^[+-]?\d+(?:\.(\d*))?\((\d+)(?:\.(\d*))?\)(?:[eE]([+-]?\d+))?$")
        for i, (row, tokens) in enumerate(zip(rows, table_lines, strict=True)):
            assert row["label"] == tokens[0]
            for j, comp in enumerate(("Fx", "Fy", "Fz")):
                m = short_form.match(tokens[1 + j])
                assert m is not None, f"{job_type}: unexpected force token {tokens[1 + j]!r}"
                exponent = int(m.group(4) or 0)
                value_place = 10.0 ** (exponent - len(m.group(1) or ""))
                error_place = value_place if m.group(3) is None else 10.0 ** (exponent - len(m.group(3)))
                assert_rounding_of(row[comp], 0.5 * value_place, ref_f[i, j], f"atom {i} {comp}")
                assert_rounding_of(row[f"{comp}_err"], 0.5 * error_place, ref_f_std[i, j], f"atom {i} {comp} error bar")
