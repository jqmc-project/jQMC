"""Tests for jqmc_workflow.wf_workflow.WF_Workflow -- TREXIO -> hamiltonian.h5.

Layer 1: construction + CLI command building. The actual ``subprocess.run``
of ``jqmc-tool trexio convert-to`` is deferred to integration tests (heavy
tier) that need a real trexio fixture.

Test class layout:
    TestWFWorkflowConstruction -- defaults + jastrow / DTN parameters
    TestWFBuildCommand         -- argv assembly
    TestWFConfigure            -- configure() summary
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

import pytest

from jqmc_workflow.wf_workflow import WF_Workflow

pytestmark = [pytest.mark.with_chdir]


class TestWFWorkflowConstruction:
    @pytest.mark.light
    def test_defaults(self):
        wf = WF_Workflow()
        assert wf.trexio_file == "trexio.h5"
        assert wf.hamiltonian_file == "hamiltonian_data.h5"
        assert wf.j1_parameter is None
        assert not hasattr(wf, "dtn_ae_cusp")
        assert wf.ao_conv_to is None

    @pytest.mark.light
    def test_explicit_args(self):
        wf = WF_Workflow(
            trexio_file="x.h5",
            hamiltonian_file="ham.h5",
            j1_parameter=0.1,
            j2_parameter=0.2,
            j3_basis_type="ao",
            jastrow_dtn="f",
        )
        assert wf.trexio_file == "x.h5"
        assert wf.j1_parameter == 0.1
        assert wf.j2_parameter == 0.2
        assert wf.jastrow_dtn == "f"

    @pytest.mark.light
    def test_invalid_ao_conv_to_raises(self):
        with pytest.raises(ValueError, match="ao_conv_to must be"):
            WF_Workflow(ao_conv_to="invalid")

    @pytest.mark.light
    @pytest.mark.parametrize("value", ["cart", "sphe", None])
    def test_ao_conv_to_accepts_valid_values(self, value):
        wf = WF_Workflow(ao_conv_to=value)
        assert wf.ao_conv_to == value


class TestWFBuildCommand:
    @pytest.mark.light
    def test_minimal_command(self):
        wf = WF_Workflow()
        cmd = wf._build_command()
        assert cmd[:3] == ["jqmc-tool", "trexio", "convert-to"]
        assert "trexio.h5" in cmd
        assert "-o" in cmd
        assert "hamiltonian_data.h5" in cmd

    @pytest.mark.light
    def test_command_includes_jastrow_flags(self):
        wf = WF_Workflow(j1_parameter=0.1, j2_parameter=0.2, j3_basis_type="ao")
        cmd = wf._build_command()
        assert "-j1" in cmd
        assert "0.1" in cmd
        assert "-j2" in cmd
        assert "0.2" in cmd
        assert "-j3" in cmd
        assert "ao" in cmd

    @pytest.mark.light
    def test_command_includes_dtn_flags(self):
        wf = WF_Workflow(
            jastrow_dtn="f",
            dtn_c=2,
            dtn_n_u=4,
            dtn_cutoff_ratio_u=0.70,
            dtn_cutoff_ratio_chi=0.72,
            dtn_cutoff_ratio_f=0.74,
        )
        cmd = wf._build_command()
        assert "--jastrow-dtn" in cmd
        assert "f" in cmd
        assert "--dtn-C" in cmd
        assert "--dtn-Nu" in cmd
        assert "--dtn-ae-cusp" not in cmd
        assert cmd[cmd.index("--dtn-cutoff-ratio-u") + 1] == "0.7"
        assert cmd[cmd.index("--dtn-cutoff-ratio-chi") + 1] == "0.72"
        assert cmd[cmd.index("--dtn-cutoff-ratio-f") + 1] == "0.74"

    @pytest.mark.light
    def test_ao_conv_to_included(self):
        wf = WF_Workflow(ao_conv_to="cart")
        cmd = wf._build_command()
        assert "--ao-conv-to" in cmd
        assert "cart" in cmd

    @pytest.mark.light
    def test_twist_average_flag(self):
        assert "--twist-average" not in WF_Workflow()._build_command()
        assert "--twist-average" in WF_Workflow(twist_average=True)._build_command()


class TestWFConfigure:
    @pytest.mark.light
    def test_returns_summary_dict(self):
        wf = WF_Workflow(trexio_file="x.h5", j1_parameter=0.5)
        cfg = wf.configure()
        assert cfg["trexio_file"] == "x.h5"
        assert cfg["j1_parameter"] == 0.5
        assert "hamiltonian_file" in cfg

    @pytest.mark.light
    def test_twist_average_in_summary(self):
        assert WF_Workflow().configure()["twist_average"] is False
        assert WF_Workflow(twist_average=True).configure()["twist_average"] is True
