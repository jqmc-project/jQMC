"""Tests for jqmc_workflow.qe_workflow.QE_Workflow -- QE -> hamiltonian.h5.

Layer 1: construction + kind validation + CLI command building. The actual
``subprocess.run`` and ``validate_inputs`` against a real QE save dir are
integration scope.

Test class layout:
    TestQEWorkflowConstruction -- defaults + kind enum validation
    TestQEBuildCommand         -- argv assembly
    TestQEConfigure            -- configure() summary
    TestQEValidateInputs       -- missing save dir / xml branches
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

from jqmc_workflow.qe_workflow import QE_Workflow

pytestmark = [pytest.mark.with_chdir]


class TestQEWorkflowConstruction:
    @pytest.mark.light
    def test_defaults(self):
        wf = QE_Workflow(qe_save_dir="/tmp/li.save")
        assert wf.qe_save_dir == "/tmp/li.save"
        assert wf.kind == "pw"
        assert wf.hamiltonian_file == "hamiltonian_data.h5"
        assert wf.upf_paths == []
        assert wf.ecp_z_cores == []
        assert wf.dtn_cutoff_ratio_u == 0.95
        assert wf.dtn_cutoff_ratio_chi == 0.95
        assert wf.dtn_cutoff_ratio_f == 0.95

    @pytest.mark.light
    @pytest.mark.parametrize("kind", ["pw", "bspline", "pgto", "pgto+pw", "pgto+bspline"])
    def test_valid_kinds_accepted(self, kind):
        wf = QE_Workflow(qe_save_dir="/tmp/x.save", kind=kind)
        assert wf.kind == kind

    @pytest.mark.light
    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="kind must be one of"):
            QE_Workflow(qe_save_dir="/tmp/x.save", kind="invalid_kind")

    @pytest.mark.light
    def test_explicit_args_preserved(self):
        wf = QE_Workflow(
            qe_save_dir="/save",
            kind="pgto",
            upf_paths=["H=H.upf", "C=C.upf"],
            ecutwfc=50.0,
            spline_order=4,
            j1_parameter=0.1,
            ecp_z_cores=[1.0, 4.0],
        )
        assert wf.upf_paths == ["H=H.upf", "C=C.upf"]
        assert wf.ecutwfc == 50.0
        assert wf.spline_order == 4
        assert wf.j1_parameter == 0.1
        assert wf.ecp_z_cores == [1.0, 4.0]

    @pytest.mark.light
    def test_removed_dtn_ae_cusp_argument_is_rejected(self):
        with pytest.raises(TypeError, match="dtn_ae_cusp"):
            QE_Workflow(qe_save_dir="/x", dtn_ae_cusp=True)
        with pytest.raises(TypeError, match="dtn_L_u"):
            QE_Workflow(qe_save_dir="/x", dtn_L_u=2.0)

    @pytest.mark.light
    def test_twist_average_preserved(self):
        wf = QE_Workflow(qe_save_dir="/x", twist_average=True)
        assert wf.twist_average is True

    @pytest.mark.light
    def test_tiling_supports_all_pure_backends_and_twist_average(self):
        for kind in ("pw", "bspline", "pgto", "pgto+pw"):
            wf = QE_Workflow(qe_save_dir="/x", kind=kind, tiling=[2, 1, 1], twist_average=True)
            assert wf.tiling == [2, 1, 1]
            assert "--twist-average" in wf._build_command()
        with pytest.raises(ValueError, match="tiling supports"):
            QE_Workflow(qe_save_dir="/x", kind="pgto+bspline", tiling=[2, 1, 1])
        with pytest.raises(ValueError, match="cannot be combined"):
            QE_Workflow(qe_save_dir="/x", kind="pw", tiling=[2, 1, 1], twist_average=True, supertwist=[0, 0, 0])


class TestQEBuildCommand:
    @pytest.mark.light
    def test_minimal_command(self):
        wf = QE_Workflow(qe_save_dir="/save")
        cmd = wf._build_command()
        assert cmd[:2] == ["jqmc-tool", "qe"]
        assert "/save" in cmd
        assert "-k" in cmd
        assert "pw" in cmd
        assert "-o" in cmd

    @pytest.mark.light
    def test_upf_paths_each_get_u_flag(self):
        wf = QE_Workflow(qe_save_dir="/x", upf_paths=["H=H.upf", "C=C.upf"])
        cmd = wf._build_command()
        assert cmd.count("-u") == 2

    @pytest.mark.light
    def test_optional_numeric_flags(self):
        wf = QE_Workflow(
            qe_save_dir="/x",
            ecutwfc=50.0,
            alpha_min=0.1,
            max_cond_number=1e6,
            mesh_factor=2.0,
            spline_order=4,
        )
        cmd = wf._build_command()
        assert "--ecutwfc" in cmd
        assert "--alpha-min" in cmd
        assert "--max-cond-number" in cmd
        assert "--mesh-factor" in cmd
        assert "--spline-order" in cmd

    @pytest.mark.light
    def test_twist_average_and_dtn_flags(self):
        wf = QE_Workflow(
            qe_save_dir="/x",
            kind="bspline",
            twist_average=True,
            jastrow_dtn="u_chi_f",
            dtn_n_u=6,
            dtn_n_chi=6,
            dtn_n_eN=2,
            dtn_n_ee=2,
        )
        cmd = wf._build_command()
        assert "--twist-average" in cmd
        assert cmd[cmd.index("--jastrow-dtn") + 1] == "u_chi_f"
        assert cmd[cmd.index("--dtn-Nu") + 1] == "6"
        assert cmd[cmd.index("--dtn-Nee") + 1] == "2"
        assert cmd[cmd.index("--dtn-cutoff-ratio-u") + 1] == "0.95"
        assert cmd[cmd.index("--dtn-cutoff-ratio-chi") + 1] == "0.95"
        assert cmd[cmd.index("--dtn-cutoff-ratio-f") + 1] == "0.95"

    @pytest.mark.light
    def test_tiling_flags(self):
        cmd = QE_Workflow(qe_save_dir="/x", kind="bspline", tiling=[2, 1, 1])._build_command()
        assert cmd.count("--tiling") == 1
        first = cmd.index("--tiling")
        assert cmd[first : first + 2] == ["--tiling", "2 1 1"]


class TestQEConfigure:
    @pytest.mark.light
    def test_returns_summary_dict(self):
        wf = QE_Workflow(qe_save_dir="/save", kind="bspline", j1_parameter=0.3)
        cfg = wf.configure()
        assert cfg["qe_save_dir"] == "/save"
        assert cfg["kind"] == "bspline"
        assert cfg["j1_parameter"] == 0.3

    @pytest.mark.light
    def test_twist_average_in_summary(self):
        cfg = QE_Workflow(qe_save_dir="/save", twist_average=True).configure()
        assert cfg["twist_average"] is True


class TestQEValidateInputs:
    @pytest.mark.light
    def test_missing_save_dir_raises(self, tmp_path):
        wf = QE_Workflow(qe_save_dir=str(tmp_path / "nonexistent.save"))
        with pytest.raises(FileNotFoundError):
            wf.validate_inputs(base_dir=str(tmp_path))

    @pytest.mark.light
    def test_missing_xml_raises(self, tmp_path):
        save_dir = tmp_path / "x.save"
        save_dir.mkdir()  # exists but no data-file-schema.xml
        wf = QE_Workflow(qe_save_dir=str(save_dir))
        with pytest.raises(FileNotFoundError):
            wf.validate_inputs(base_dir=str(tmp_path))
