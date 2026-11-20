"""Tests for jqmc_workflow.mcmc_workflow.MCMC_Workflow -- MCMC sampling.

Layer 1: construction + configure + _get_command + can_resume_after_completed.
The run() pipeline is integration-scope (real jqmc).

Test class layout:
    TestMCMCWorkflowConstruction -- defaults + explicit args
    TestMCMCConfigure            -- summary dict + auto/fixed mode marker
    TestMCMCGetCommand           -- argv string
    TestMCMCResumeBehaviour      -- can_resume_after_completed
    TestMCMCDependencies         -- FileFrom / ValueFrom integration
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

import asyncio
from types import SimpleNamespace

import pytest

from jqmc_workflow.mcmc_workflow import MCMC_Workflow
from jqmc_workflow.state import WorkflowStatus, add_job
from jqmc_workflow.workflow import Container, FileFrom, ValueFrom

pytestmark = [pytest.mark.with_chdir]


class TestMCMCWorkflowConstruction:
    @pytest.mark.light
    def test_defaults(self):
        wf = MCMC_Workflow()
        assert wf.server_machine_name == "localhost"
        assert wf.hamiltonian_file == "hamiltonian_data.h5"
        assert wf.queue_label == "default"
        assert wf.jobname == "jqmc-mcmc"
        assert wf.num_mcmc_bin_blocks == 5
        assert wf.num_mcmc_warmup_steps == 0
        assert wf.target_error == 0.001
        assert wf.num_mcmc_steps is None  # auto mode
        assert wf.pilot_steps == 100
        assert wf.max_continuation == 1
        assert wf.precision_mode == "full"
        assert wf.mpc is None

    @pytest.mark.light
    def test_explicit_args(self):
        wf = MCMC_Workflow(
            server_machine_name="cluster",
            num_mcmc_steps=5000,
            jobname="my-mcmc",
            number_of_walkers=16,
            target_error=1e-5,
            max_continuation=3,
            atomic_force=True,
            compute_stress=True,
            mpc=False,
        )
        assert wf.server_machine_name == "cluster"
        assert wf.num_mcmc_steps == 5000
        assert wf.atomic_force is True
        assert wf.compute_stress is True
        assert wf.mpc is False

    @pytest.mark.light
    def test_pilot_queue_defaults_to_main(self):
        wf = MCMC_Workflow(queue_label="gpu")
        assert wf.pilot_queue_label == "gpu"


class TestMCMCConfigure:
    @pytest.mark.light
    def test_fixed_mode_marker(self):
        wf = MCMC_Workflow(num_mcmc_steps=1000)
        cfg = wf.configure()
        assert cfg["mode"] == "fixed"

    @pytest.mark.light
    def test_auto_mode_marker(self):
        wf = MCMC_Workflow()
        cfg = wf.configure()
        assert cfg["mode"] == "auto"

    @pytest.mark.light
    def test_summary_includes_key_params(self):
        wf = MCMC_Workflow(jobname="X", target_error=1e-4)
        cfg = wf.configure()
        assert cfg["jobname"] == "X"
        assert cfg["target_error"] == 1e-4
        assert "pilot_steps" in cfg


class TestMCMCGetCommand:
    @pytest.mark.light
    def test_returns_jqmc_input_string(self):
        wf = MCMC_Workflow()
        assert wf._get_command("input.toml") == "jqmc input.toml"


class TestMCMCPostprocessCommand:
    @pytest.mark.light
    def test_compute_energy_uses_self_contained_checkpoint(self, tmp_path, monkeypatch):
        """Workflow post-processing needs no external BZ-weight source."""
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return SimpleNamespace(stdout="E = -5.0 +- 0.001 Ha.\n")

        monkeypatch.setattr("jqmc_workflow.mcmc_workflow.subprocess.run", fake_run)
        workflow = MCMC_Workflow(hamiltonian_file="hamiltonian_data.h5")
        energy, error = workflow._compute_energy("restart.h5", str(tmp_path))
        assert (energy, error) == (-5.0, 0.001)
        assert captured["cmd"] == [
            "jqmc-tool",
            "mcmc",
            "compute-energy",
            "restart.h5",
            "-b",
            "5",
            "-w",
            "0",
        ]
        assert captured["kwargs"]["cwd"] == str(tmp_path)


class TestMCMCResumeBehaviour:
    """can_resume_after_completed allows continuation when target_error not met."""

    @pytest.mark.light
    def test_returns_callable(self, tmp_path):
        wf = MCMC_Workflow()
        # The method takes the project dir; result depends on state file content.
        # Just verify it can be called without raising on an empty dir.
        result = wf.can_resume_after_completed(str(tmp_path))
        assert isinstance(result, bool)

    @pytest.mark.light
    def test_fixed_step_relaunch_rejects_poisoned_fetched_step_before_submission(self, tmp_path):
        """A fetched NaN result must fail before the next continuation is submitted."""
        (tmp_path / "restart.h5").touch()
        (tmp_path / "mcmc_1.out").write_text("Program ends\n", encoding="utf-8")
        add_job(
            str(tmp_path),
            "mcmc_1.toml",
            "mcmc_1.out",
            "job-1",
            "local",
            status="fetched",
            step=1,
        )
        workflow = MCMC_Workflow(num_mcmc_steps=10, max_continuation=2)
        submissions: list[int] = []

        async def fake_submit_and_wait(*_args: object, **kwargs: object) -> None:
            submissions.append(int(kwargs["step"]))

        workflow._submit_and_wait = fake_submit_and_wait
        workflow._compute_energy = lambda *_args, **_kwargs: (float("nan"), 0.1, None, None)

        status, _, values = asyncio.run(workflow._launch_fixed_steps(str(tmp_path)))

        assert status == WorkflowStatus.FAILED
        assert "Non-finite energy" in values["error"]
        assert submissions == []

    @pytest.mark.light
    def test_fixed_step_relaunch_does_not_reuse_previous_energy_when_fetched_result_is_missing(self, tmp_path):
        """A fetched step without an estimate must not reuse the preceding step's energy."""
        (tmp_path / "restart.h5").touch()
        for step in (1, 2):
            (tmp_path / f"mcmc_{step}.out").write_text("Program ends\n", encoding="utf-8")
            add_job(
                str(tmp_path),
                f"mcmc_{step}.toml",
                f"mcmc_{step}.out",
                f"job-{step}",
                "local",
                status="fetched",
                step=step,
            )

        workflow = MCMC_Workflow(num_mcmc_steps=10, max_continuation=3)
        submissions: list[int] = []
        energy_calls = 0

        async def fake_submit_and_wait(*_args: object, **kwargs: object) -> None:
            submissions.append(int(kwargs["step"]))

        def fake_compute_energy(
            *_args: object,
            **_kwargs: object,
        ) -> tuple[float | None, float | None, None, None]:
            nonlocal energy_calls
            energy_calls += 1
            if energy_calls == 1:
                return -1.0, 0.1, None, None
            return None, None, None, None

        workflow._submit_and_wait = fake_submit_and_wait
        workflow._compute_energy = fake_compute_energy

        status, _, values = asyncio.run(workflow._launch_fixed_steps(str(tmp_path)))

        assert status == WorkflowStatus.FAILED
        assert values["error"] == "No energy estimate was obtained from the completed run."
        assert submissions == []

    @pytest.mark.light
    def test_fixed_step_relaunch_validates_finite_fetched_step_then_continues(self, tmp_path):
        """A finite fetched result is validated before the next fixed-step run."""
        (tmp_path / "restart.h5").touch()
        (tmp_path / "mcmc_1.out").write_text("Program ends\n", encoding="utf-8")
        add_job(
            str(tmp_path),
            "mcmc_1.toml",
            "mcmc_1.out",
            "job-1",
            "local",
            status="fetched",
            step=1,
        )
        workflow = MCMC_Workflow(num_mcmc_steps=10, max_continuation=2)
        submissions: list[int] = []
        energy_calls: list[str] = []
        validation_counts_at_submission: list[int] = []

        async def fake_submit_and_wait(*_args: object, **kwargs: object) -> None:
            submissions.append(int(kwargs["step"]))
            validation_counts_at_submission.append(len(energy_calls))

        def fake_compute_energy(
            restart_chk: str,
            *_args: object,
            **_kwargs: object,
        ) -> tuple[float, float, None, None]:
            energy_calls.append(restart_chk)
            return -1.0, 0.1, None, None

        workflow._submit_and_wait = fake_submit_and_wait
        workflow._compute_energy = fake_compute_energy

        status, _, _ = asyncio.run(workflow._launch_fixed_steps(str(tmp_path)))

        assert status == WorkflowStatus.COMPLETED
        assert submissions == [2]
        assert validation_counts_at_submission == [1]


class TestMCMCDependencies:
    @pytest.mark.light
    def test_filefrom_hamiltonian_input(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        c = Container(
            label="mcmc",
            dirname="mcmc",
            input_files=[FileFrom("vmc-opt", "hamiltonian_data_opt_step_9.h5")],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=MCMC_Workflow(),
        )
        assert isinstance(c.input_files[0], FileFrom)
        assert c.rename_input_files[0] == "hamiltonian_data.h5"

    @pytest.mark.light
    def test_dynamic_filename_via_valuefrom(self, tmp_path, monkeypatch):
        """The composed pattern: FileFrom whose filename is a ValueFrom."""
        monkeypatch.chdir(tmp_path)
        c = Container(
            label="mcmc",
            dirname="mcmc",
            input_files=[FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian"))],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=MCMC_Workflow(),
        )
        ff = c.input_files[0]
        assert isinstance(ff, FileFrom)
        assert isinstance(ff.filename, ValueFrom)
        assert ff.filename.key == "optimized_hamiltonian"
