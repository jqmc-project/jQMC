"""Tests for jqmc_workflow.vmc_workflow.VMC_Workflow -- Jastrow/orbital optimisation.

Layer 1: construction + configure + _get_command. The run() pipeline
(submit/poll/fetch/convergence loop) is integration-scope (real jqmc).

Test class layout:
    TestVMCWorkflowConstruction -- defaults + explicit args
    TestVMCConfigure            -- summary dict
    TestVMCGetCommand           -- _get_command returns the right argv string
    TestVMCDependencies         -- FileFrom / ValueFrom integration (via Container)
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

import tomllib
from pathlib import Path

import pytest

from jqmc_workflow.vmc_workflow import VMC_Workflow
from jqmc_workflow.workflow import Container, FileFrom

pytestmark = [pytest.mark.with_chdir]


class TestVMCWorkflowConstruction:
    @pytest.mark.light
    def test_defaults(self):
        wf = VMC_Workflow()
        assert wf.server_machine_name == "localhost"
        assert wf.num_opt_steps == 20
        assert wf.hamiltonian_file == "hamiltonian_data.h5"
        assert wf.queue_label == "default"
        assert wf.jobname == "jqmc-vmc"
        assert wf.number_of_walkers == 4
        assert wf.target_error == 0.001
        assert wf.num_mcmc_steps is None  # auto mode
        assert wf.max_continuation == 1
        assert wf.precision_mode == "full"
        assert wf.mpc is None  # MPC section omitted

    @pytest.mark.light
    def test_explicit_args(self):
        wf = VMC_Workflow(
            server_machine_name="cluster",
            num_opt_steps=50,
            jobname="my-vmc",
            number_of_walkers=8,
            target_error=1e-4,
            num_mcmc_steps=1000,  # fixed-step mode
            max_continuation=3,
            precision_mode="mixed",
            mpc=True,
        )
        assert wf.server_machine_name == "cluster"
        assert wf.num_opt_steps == 50
        assert wf.num_mcmc_steps == 1000
        assert wf.max_continuation == 3
        assert wf.precision_mode == "mixed"
        assert wf.mpc is True

    @pytest.mark.light
    def test_pilot_queue_defaults_to_main(self):
        wf = VMC_Workflow(queue_label="gpu")
        assert wf.pilot_queue_label == "gpu"

    @pytest.mark.light
    def test_pilot_queue_explicit(self):
        wf = VMC_Workflow(queue_label="gpu", pilot_queue_label="cpu-pilot")
        assert wf.pilot_queue_label == "cpu-pilot"

    @pytest.mark.light
    def test_lm_convenience_kwargs_merge_without_mutating_input(self):
        original = {"max_iterations": 25}
        wf = VMC_Workflow(
            optimizer_kwargs=original,
            use_lm=True,
            num_opt_param=10,
            delta=0.15,
            epsilon=0.001,
        )
        assert original == {"max_iterations": 25}
        assert wf.optimizer_kwargs == {
            "max_iterations": 25,
            "use_lm": True,
            "lm_subspace_dim": 10,
            "delta": 0.15,
            "epsilon": 0.001,
        }

    @pytest.mark.light
    def test_lm_convenience_conflict_is_rejected(self):
        with pytest.raises(ValueError, match=r"optimizer_kwargs.*use_lm"):
            VMC_Workflow(optimizer_kwargs={"use_lm": False}, use_lm=True)

    @pytest.mark.light
    def test_lm_convenience_kwargs_reach_generated_toml(self, tmp_path):
        out = tmp_path / "vmc.toml"
        wf = VMC_Workflow(
            num_mcmc_per_measurement=4,
            use_lm=True,
            num_opt_param=10,
            delta=0.15,
            epsilon=0.001,
        )
        wf._generate_input(20, 2, str(out))
        with out.open("rb") as fh:
            cfg = tomllib.load(fh)
        opt = cfg["vmc"]["optimizer_kwargs"]
        assert opt["use_lm"] is True
        assert opt["lm_subspace_dim"] == 10
        # These values are configuration fields round-tripped through TOML, not
        # floating-point results of a numerical kernel, so exact equality is the
        # contract under test.
        assert opt["delta"] == 0.15
        assert opt["epsilon"] == 0.001


class TestVMCConfigure:
    @pytest.mark.light
    def test_fixed_mode_marker(self):
        wf = VMC_Workflow(num_mcmc_steps=100)
        cfg = wf.configure()
        assert cfg["mode"] == "fixed"
        assert cfg["num_mcmc_steps"] == 100

    @pytest.mark.light
    def test_auto_mode_marker(self):
        wf = VMC_Workflow()  # num_mcmc_steps=None
        cfg = wf.configure()
        assert cfg["mode"] == "auto"

    @pytest.mark.light
    def test_summary_includes_key_params(self):
        wf = VMC_Workflow(jobname="X", target_error=1e-5)
        cfg = wf.configure()
        assert cfg["jobname"] == "X"
        assert cfg["target_error"] == 1e-5
        assert "server_machine" in cfg
        assert "num_opt_steps" in cfg


class TestVMCGetCommand:
    @pytest.mark.light
    def test_returns_jqmc_input_string(self):
        wf = VMC_Workflow()
        assert wf._get_command("input_0.toml") == "jqmc input_0.toml"


class TestVMCDependencies:
    """VMC consumed in a Container with upstream FileFrom dependencies."""

    @pytest.mark.light
    def test_filefrom_hamiltonian_input(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        c = Container(
            label="vmc-opt",
            dirname="01_vmc",
            input_files=[FileFrom("wf-build", "hamiltonian_data.h5")],
            workflow=VMC_Workflow(),
        )
        # Container records the dependency for Launcher to resolve.
        assert isinstance(c.input_files[0], FileFrom)
        assert c.input_files[0].label == "wf-build"


class TestOptimizedHamiltonianPublication:
    """What a VMC run hands downstream -- one path for every run.

    jQMC dumps ``hamiltonian_data_opt_step_N.h5`` every ``wf_dump_freq`` steps,
    and under twist averaging that file is a COMPLETE combined Hamiltonian
    (all twists), so the workflow publishes the newest dump regardless of the
    twist count and never assembles a second artifact.
    """

    @pytest.mark.light
    def test_newest_opt_step_dump_is_published(self, tmp_path):
        for step in (2, 12, 9):
            (tmp_path / f"hamiltonian_data_opt_step_{step}.h5").touch()
        assert VMC_Workflow()._latest_optimized_hamiltonian(str(tmp_path)) == "hamiltonian_data_opt_step_12.h5"

    @pytest.mark.light
    def test_missing_opt_step_dump_raises(self, tmp_path):
        """Nothing to publish is an error, not a silent fallback: guessing an
        input file (or one twist of a twist-averaged run) would corrupt the
        next stage."""
        (tmp_path / "hamiltonian_data.h5").touch()
        with pytest.raises(RuntimeError, match="produced no hamiltonian_data_opt_step"):
            VMC_Workflow()._latest_optimized_hamiltonian(str(tmp_path))
