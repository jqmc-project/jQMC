"""Tests for jqmc_workflow._results -- pure dataclass result containers.

Layer 1 (pure unit): no subprocess, no SSH, no filesystem.

These dataclasses store **facts only** (deterministic parser output) -- no
methods, no __post_init__, no validation. Tests cover:
    - all-default construction
    - all-fields-set construction
    - mutable default factories give independent instances
    - equality semantics from @dataclass

Test class layout:
    TestVMCStepData
    TestVMCDiagnosticData
    TestMCMCDiagnosticData
    TestLRDMCDiagnosticData
    TestLRDMCExtDiagnosticData
    TestInputParameters
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

import pytest

from jqmc_workflow._results import (
    Input_Parameters,
    LRDMC_Diagnostic_Data,
    LRDMC_Ext_Diagnostic_Data,
    MCMC_Diagnostic_Data,
    VMC_Diagnostic_Data,
    VMC_Step_Data,
)

pytestmark = [pytest.mark.without_chdir]


class TestVMCStepData:
    @pytest.mark.light
    def test_minimal_construction_requires_step(self):
        d = VMC_Step_Data(step=0)
        assert d.step == 0
        assert d.energy is None
        assert d.energy_error is None
        assert d.timing_breakdown == {}

    @pytest.mark.light
    def test_step_is_only_required_field(self):
        with pytest.raises(TypeError):
            VMC_Step_Data()  # type: ignore[call-arg]

    @pytest.mark.light
    def test_full_construction(self):
        d = VMC_Step_Data(
            step=42,
            energy=-1.123,
            energy_error=0.0012,
            max_force=0.05,
            max_force_error=0.001,
            signal_to_noise_ratio=12.3,
            avg_walker_weight=0.98,
            acceptance_ratio=0.65,
            total_time_sec=120.5,
            precompilation_time_sec=8.0,
            net_time_sec=112.5,
            timing_breakdown={"mcmc_update": 1.5, "e_L": 0.8},
        )
        assert d.step == 42
        assert d.timing_breakdown["mcmc_update"] == 1.5

    @pytest.mark.light
    def test_timing_breakdown_default_is_independent(self):
        """Mutable default_factory: each instance gets its own dict."""
        a = VMC_Step_Data(step=0)
        b = VMC_Step_Data(step=1)
        a.timing_breakdown["x"] = 1.0
        assert b.timing_breakdown == {}

    @pytest.mark.light
    def test_dataclass_equality(self):
        a = VMC_Step_Data(step=5, energy=-1.0)
        b = VMC_Step_Data(step=5, energy=-1.0)
        assert a == b

    @pytest.mark.light
    def test_dataclass_inequality(self):
        assert VMC_Step_Data(step=5) != VMC_Step_Data(step=6)


class TestVMCDiagnosticData:
    @pytest.mark.light
    def test_default_construction(self):
        d = VMC_Diagnostic_Data()
        assert d.steps == []
        assert d.opt_timing_breakdown == {}
        assert d.jax_devices is None
        assert d.stderr_tail == ""

    @pytest.mark.light
    def test_steps_default_is_independent(self):
        a = VMC_Diagnostic_Data()
        b = VMC_Diagnostic_Data()
        a.steps.append(VMC_Step_Data(step=0))
        assert b.steps == []

    @pytest.mark.light
    def test_full_construction(self):
        d = VMC_Diagnostic_Data(
            steps=[VMC_Step_Data(step=0), VMC_Step_Data(step=1)],
            total_opt_steps=100,
            total_opt_time_sec=3600.0,
            opt_timing_breakdown={"mcmc_run": 1000.0},
            optimized_hamiltonian="hamiltonian_data_opt_step_99.h5",
            restart_checkpoint="restart_0.h5",
            num_mpi_processes=4,
            num_walkers_per_process=8,
            jax_backend="cpu",
            jax_devices=["CpuDevice(id=0)"],
            stderr_tail="warning: ...",
        )
        assert len(d.steps) == 2
        assert d.num_mpi_processes == 4


class TestMCMCDiagnosticData:
    @pytest.mark.light
    def test_default_construction(self):
        d = MCMC_Diagnostic_Data()
        assert d.timing_breakdown == {}
        assert d.atomic_forces is None
        assert d.stress_tensor is None
        assert d.pressure is None

    @pytest.mark.light
    def test_timing_breakdown_independent(self):
        a = MCMC_Diagnostic_Data()
        b = MCMC_Diagnostic_Data()
        a.timing_breakdown["e_L"] = 0.5
        assert b.timing_breakdown == {}

    @pytest.mark.light
    def test_pbc_full_construction(self):
        """Full pressure + stress tensor (PBC system)."""
        d = MCMC_Diagnostic_Data(
            acceptance_ratio=0.7,
            energy=-1.5,
            stress_tensor={"sigma_xx": 0.01, "sigma_xx_err": 1e-4},
            pressure={"P_hydro": 0.005, "P_hydro_err": 5e-5},
        )
        assert d.pressure["P_hydro"] == 0.005

    @pytest.mark.light
    def test_molecular_construction_leaves_pbc_fields_none(self):
        """Molecular system: stress / pressure stay None."""
        d = MCMC_Diagnostic_Data(energy=-1.0, energy_error=0.001)
        assert d.stress_tensor is None
        assert d.pressure is None


class TestLRDMCDiagnosticData:
    @pytest.mark.light
    def test_default_construction(self):
        d = LRDMC_Diagnostic_Data()
        # All fields default to None / empty containers.
        assert d.stderr_tail == ""

    @pytest.mark.light
    def test_independent_mutable_defaults(self):
        a = LRDMC_Diagnostic_Data()
        b = LRDMC_Diagnostic_Data()
        # Verify any list / dict default_factory fields are independent.
        for fname in vars(a):
            v = getattr(a, fname)
            if isinstance(v, (list, dict)):
                assert getattr(b, fname) == type(v)(), f"shared mutable default detected for field {fname!r}"


class TestLRDMCExtDiagnosticData:
    @pytest.mark.light
    def test_default_construction(self):
        d = LRDMC_Ext_Diagnostic_Data()
        assert d.extrapolated_energy is None
        assert d.per_alat_results == []
        assert d.stderr_tail == ""

    @pytest.mark.light
    def test_per_alat_results_independent(self):
        a = LRDMC_Ext_Diagnostic_Data()
        b = LRDMC_Ext_Diagnostic_Data()
        a.per_alat_results.append({"alat": 0.1, "energy": -1.0, "energy_error": 0.001})
        assert b.per_alat_results == []

    @pytest.mark.light
    def test_full_construction(self):
        d = LRDMC_Ext_Diagnostic_Data(
            extrapolated_energy=-1.2345,
            extrapolated_energy_error=0.0001,
            per_alat_results=[
                {"alat": 0.1, "energy": -1.230, "energy_error": 0.001},
                {"alat": 0.2, "energy": -1.225, "energy_error": 0.001},
            ],
        )
        assert len(d.per_alat_results) == 2


class TestInputParameters:
    @pytest.mark.light
    def test_default_construction(self):
        p = Input_Parameters()
        assert p.actual_opt_steps is None
        assert p.per_input == []

    @pytest.mark.light
    def test_per_input_independent(self):
        a = Input_Parameters()
        b = Input_Parameters()
        a.per_input.append({"input_file": "x.toml", "job_type": "vmc"})
        assert b.per_input == []

    @pytest.mark.light
    def test_full_construction(self):
        p = Input_Parameters(
            actual_opt_steps=99,
            per_input=[{"input_file": "input_1.toml", "job_type": "vmc", "control": {}, "vmc": {}}],
        )
        assert p.actual_opt_steps == 99
        assert p.per_input[0]["job_type"] == "vmc"
