"""Tests for jqmc_workflow.lrdmc_workflow.LRDMC_Workflow -- LRDMC sampling.

Layer 1: construction + validation + job_type selection + configure +
_get_command. The run() pipeline (GFMC_t / GFMC_n drivers) is integration
scope.

Test class layout:
    TestLRDMCWorkflowConstruction -- defaults + explicit args + GFMC minima
    TestLRDMCJobTypeSelection     -- _use_gfmc_n + job_type property
    TestLRDMCStressValidation     -- compute_stress incompatible with GFMC_n
    TestLRDMCConfigure            -- summary dict
    TestLRDMCGetCommand
    TestLRDMCResumeBehaviour      -- can_resume_after_completed
    TestLRDMCDependencies         -- FileFrom / ValueFrom integration
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

from types import SimpleNamespace

import pytest

from jqmc_workflow.lrdmc_workflow import LRDMC_Workflow
from jqmc_workflow.workflow import Container, FileFrom, ValueFrom

pytestmark = [pytest.mark.with_chdir]


# LRDMC's constructor validates GFMC minima (warmup>=25, bin>=10, collect>=5).
# The defaults equal the minima; tests pass them explicitly so the bodies do not
# depend on the default values.
_LRDMC_OK = {
    "num_gfmc_warmup_steps": 25,
    "num_gfmc_bin_blocks": 10,
    "num_gfmc_collect_steps": 5,
}


class TestLRDMCWorkflowConstruction:
    @pytest.mark.light
    def test_defaults_satisfy_gfmc_minima(self):
        """The default constructor is usable: defaults equal the GFMC minima."""
        from jqmc._setting import GFMC_MIN_BIN_BLOCKS, GFMC_MIN_COLLECT_STEPS, GFMC_MIN_WARMUP_STEPS

        wf = LRDMC_Workflow()
        assert wf.num_gfmc_warmup_steps == GFMC_MIN_WARMUP_STEPS
        assert wf.num_gfmc_bin_blocks == GFMC_MIN_BIN_BLOCKS
        assert wf.num_gfmc_collect_steps == GFMC_MIN_COLLECT_STEPS

    @pytest.mark.light
    def test_minimal_valid_construction(self):
        wf = LRDMC_Workflow(**_LRDMC_OK)
        assert wf.server_machine_name == "localhost"
        assert wf.alat == 0.30
        assert wf.hamiltonian_file == "hamiltonian_data.h5"
        assert wf.queue_label == "default"
        assert wf.jobname == "jqmc-lrdmc"
        assert wf.time_projection_tau == 0.10
        assert wf.max_continuation == 1
        assert wf.precision_mode == "full"

    @pytest.mark.light
    def test_explicit_args(self):
        wf = LRDMC_Workflow(
            server_machine_name="cluster",
            alat=0.10,
            jobname="my-lrdmc",
            number_of_walkers=32,
            target_error=1e-5,
            num_gfmc_warmup_steps=50,
            num_gfmc_bin_blocks=20,
            num_gfmc_collect_steps=20,
        )
        assert wf.server_machine_name == "cluster"
        assert wf.alat == 0.10
        assert wf.num_gfmc_warmup_steps == 50

    @pytest.mark.parametrize(
        "kwarg,value,message",
        [
            ("num_gfmc_warmup_steps", 0, "num_gfmc_warmup_steps"),
            ("num_gfmc_bin_blocks", 1, "num_gfmc_bin_blocks"),
            ("num_gfmc_collect_steps", 0, "num_gfmc_collect_steps"),
        ],
    )
    @pytest.mark.light
    def test_gfmc_minimum_validation(self, kwarg, value, message):
        """GFMC params below the minimum thresholds in _setting.py are rejected."""
        # Default values pass validation; need to also pass other minima.
        # Set the offending kwarg below minimum.
        kwargs = {
            "num_gfmc_warmup_steps": 50,
            "num_gfmc_bin_blocks": 20,
            "num_gfmc_collect_steps": 20,
        }
        kwargs[kwarg] = value
        with pytest.raises(ValueError, match=message):
            LRDMC_Workflow(**kwargs)


class TestLRDMCJobTypeSelection:
    @pytest.mark.light
    def test_default_is_gfmc_t_lrdmc_tau(self):
        wf = LRDMC_Workflow(**_LRDMC_OK)
        assert wf._use_gfmc_n is False
        assert wf.job_type == "lrdmc-tau"

    @pytest.mark.light
    def test_target_survived_walkers_switches_to_gfmc_n(self):
        wf = LRDMC_Workflow(target_survived_walkers_ratio=0.95, **_LRDMC_OK)
        assert wf._use_gfmc_n is True
        assert wf.job_type == "lrdmc-bra"

    @pytest.mark.light
    def test_num_projection_per_measurement_switches_to_gfmc_n(self):
        wf = LRDMC_Workflow(num_projection_per_measurement=10, **_LRDMC_OK)
        assert wf._use_gfmc_n is True
        assert wf.job_type == "lrdmc-bra"


class TestLRDMCStressValidation:
    @pytest.mark.light
    def test_compute_stress_with_gfmc_n_ok(self):
        # GFMC_n (lrdmc-bra) now supports stress at parity with GFMC_t.
        wf = LRDMC_Workflow(
            compute_stress=True,
            target_survived_walkers_ratio=0.95,
            **_LRDMC_OK,
        )
        assert wf.compute_stress is True
        assert wf.job_type == "lrdmc-bra"

    @pytest.mark.light
    @pytest.mark.parametrize("gfmc_n", [True, False], ids=["lrdmc-bra", "lrdmc-tau"])
    def test_compute_stress_reaches_generated_input(self, tmp_path, gfmc_n):
        """compute_stress is written to the TOML on BOTH drivers (it was dropped for lrdmc-bra)."""
        import tomllib

        extra = {"target_survived_walkers_ratio": 0.95} if gfmc_n else {}
        wf = LRDMC_Workflow(compute_stress=True, **extra, **_LRDMC_OK)
        path = tmp_path / "lrdmc.toml"
        # GFMC_n has no default projection count (#690): pass an explicit count
        # (40 at alat = 0.30).
        nmpm = {"num_projection_per_measurement": 40} if gfmc_n else {}
        wf._generate_input(num_steps=100, input_file=str(path), **nmpm)
        with open(path, "rb") as stream:
            toml_data = tomllib.load(stream)
        assert toml_data[wf.job_type]["compute_stress"] is True

    @pytest.mark.light
    def test_compute_stress_with_gfmc_t_ok(self):
        wf = LRDMC_Workflow(compute_stress=True, **_LRDMC_OK)
        assert wf.compute_stress is True
        assert wf.job_type == "lrdmc-tau"


class TestLRDMCConfigure:
    @pytest.mark.light
    def test_summary_includes_alat_and_target(self):
        wf = LRDMC_Workflow(alat=0.20, target_error=1e-4, **_LRDMC_OK)
        cfg = wf.configure()
        assert cfg["alat"] == 0.20
        assert cfg["target_error"] == 1e-4
        assert "pilot_steps" in cfg


class TestLRDMCGetCommand:
    @pytest.mark.light
    def test_returns_jqmc_input_string(self):
        wf = LRDMC_Workflow(**_LRDMC_OK)
        assert wf._get_command("input.toml") == "jqmc input.toml"


class TestLRDMCPostprocessCommand:
    @pytest.mark.light
    def test_compute_energy_uses_self_contained_checkpoint(self, tmp_path, monkeypatch):
        """Workflow post-processing needs no external BZ-weight source."""
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return SimpleNamespace(stdout="E = -6.0 +- 0.002 Ha.\n")

        monkeypatch.setattr("jqmc_workflow.lrdmc_workflow.subprocess.run", fake_run)
        workflow = LRDMC_Workflow(hamiltonian_file="hamiltonian_data.h5", **_LRDMC_OK)
        energy, error = workflow._compute_energy("restart.h5", str(tmp_path))
        assert (energy, error) == (-6.0, 0.002)
        assert captured["cmd"] == [
            "jqmc-tool",
            "lrdmc",
            "compute-energy",
            "restart.h5",
            "-b",
            "10",
            "-w",
            "25",
            "-c",
            "5",
        ]
        assert captured["kwargs"]["cwd"] == str(tmp_path)


class TestLRDMCResumeBehaviour:
    @pytest.mark.light
    def test_fixed_step_mode_never_resumes(self, tmp_path):
        wf = LRDMC_Workflow(num_gfmc_projections=1000, **_LRDMC_OK)
        assert wf.can_resume_after_completed(str(tmp_path)) is False

    @pytest.mark.light
    def test_no_state_returns_false(self, tmp_path):
        wf = LRDMC_Workflow(**_LRDMC_OK)
        assert wf.can_resume_after_completed(str(tmp_path)) is False


class TestLRDMCDependencies:
    @pytest.mark.light
    def test_filefrom_hamiltonian_input(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        c = Container(
            label="lrdmc",
            dirname="lrdmc",
            input_files=[FileFrom("vmc", "hamiltonian_data_opt_step_9.h5")],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=LRDMC_Workflow(**_LRDMC_OK),
        )
        assert isinstance(c.input_files[0], FileFrom)

    @pytest.mark.light
    def test_e_scf_can_be_valuefrom(self):
        """E_scf can be wired from an upstream MCMC energy via ValueFrom."""
        wf = LRDMC_Workflow(E_scf=ValueFrom("mcmc", "energy"), **_LRDMC_OK)
        assert isinstance(wf.E_scf, ValueFrom)
        assert wf.E_scf.key == "energy"
