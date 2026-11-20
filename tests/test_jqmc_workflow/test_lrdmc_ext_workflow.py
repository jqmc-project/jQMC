"""Tests for jqmc_workflow.lrdmc_ext_workflow.LRDMC_Ext_Workflow -- a^2->0 extrapolation.

Layer 1: construction + alat_list normalisation + per-alat
num_projection_per_measurement handling. The fan-out of N sub-workflows
under different alat values is integration scope.

Test class layout:
    TestLRDMCExtWorkflowConstruction -- defaults + alat_list + GFMC minima
    TestLRDMCExtNumProjPerMeasurement -- int / dict / list normalisation
    TestLRDMCExtConfigure
    TestLRDMCExtFanOut               -- sub-workflow creation per alat (light)
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

from jqmc_workflow.lrdmc_ext_workflow import LRDMC_Ext_Workflow

pytestmark = [pytest.mark.with_chdir]


# Same GFMC-min validation as LRDMC_Workflow.
_OK = {
    "num_gfmc_warmup_steps": 25,
    "num_gfmc_bin_blocks": 10,
    "num_gfmc_collect_steps": 5,
}


class TestLRDMCExtWorkflowConstruction:
    @pytest.mark.light
    def test_defaults_satisfy_gfmc_minima(self):
        """The default constructor is usable: defaults equal the GFMC minima."""
        from jqmc._setting import GFMC_MIN_BIN_BLOCKS, GFMC_MIN_COLLECT_STEPS, GFMC_MIN_WARMUP_STEPS

        wf = LRDMC_Ext_Workflow()
        assert wf.num_gfmc_warmup_steps == GFMC_MIN_WARMUP_STEPS
        assert wf.num_gfmc_bin_blocks == GFMC_MIN_BIN_BLOCKS
        assert wf.num_gfmc_collect_steps == GFMC_MIN_COLLECT_STEPS

    @pytest.mark.light
    def test_minimal_valid_construction(self):
        wf = LRDMC_Ext_Workflow(**_OK)
        assert wf.server_machine_name == "localhost"
        assert wf.alat_list == [0.5, 0.4, 0.3]
        assert wf.polynomial_order == 2
        assert wf.hamiltonian_file == "hamiltonian_data.h5"
        assert wf.max_continuation == 5
        assert wf.jobname_prefix == "jqmc-lrdmc"

    @pytest.mark.light
    def test_explicit_alat_list(self):
        wf = LRDMC_Ext_Workflow(alat_list=[0.1, 0.2, 0.4, 0.8], **_OK)
        assert wf.alat_list == [0.1, 0.2, 0.4, 0.8]

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
        kwargs = {
            "num_gfmc_warmup_steps": 50,
            "num_gfmc_bin_blocks": 20,
            "num_gfmc_collect_steps": 20,
        }
        kwargs[kwarg] = value
        with pytest.raises(ValueError, match=message):
            LRDMC_Ext_Workflow(**kwargs)


class TestLRDMCExtNumProjPerMeasurement:
    @pytest.mark.light
    def test_none_default(self):
        wf = LRDMC_Ext_Workflow(**_OK)
        assert wf.num_projection_per_measurement is None

    @pytest.mark.light
    def test_int_kept_as_is(self):
        wf = LRDMC_Ext_Workflow(num_projection_per_measurement=10, **_OK)
        assert wf.num_projection_per_measurement == 10

    @pytest.mark.light
    def test_dict_kept_as_is(self):
        wf = LRDMC_Ext_Workflow(
            alat_list=[0.5, 0.3],
            polynomial_order=1,
            num_projection_per_measurement={0.5: 10, 0.3: 20},
            **_OK,
        )
        assert wf.num_projection_per_measurement == {0.5: 10, 0.3: 20}

    @pytest.mark.light
    def test_list_normalised_to_dict(self):
        """The TOML-safe wire form (list of {alat,nmpm}) becomes a dict at construction."""
        wf = LRDMC_Ext_Workflow(
            alat_list=[0.5, 0.3],
            polynomial_order=1,
            num_projection_per_measurement=[
                {"alat": 0.5, "nmpm": 10},
                {"alat": 0.3, "nmpm": 20},
            ],
            **_OK,
        )
        assert wf.num_projection_per_measurement == {0.5: 10, 0.3: 20}

    @pytest.mark.light
    def test_malformed_list_raises(self):
        with pytest.raises(ValueError, match="must be"):
            LRDMC_Ext_Workflow(
                num_projection_per_measurement=[
                    {"bad_key": 0.5, "nmpm": 10},
                ],
                **_OK,
            )


class TestLRDMCExtConfigure:
    @pytest.mark.light
    def test_summary_includes_alat_list_and_order(self):
        wf = LRDMC_Ext_Workflow(alat_list=[0.4, 0.3, 0.2, 0.1], polynomial_order=3, **_OK)
        cfg = wf.configure()
        assert cfg["alat_list"] == [0.4, 0.3, 0.2, 0.1]
        assert cfg["polynomial_order"] == 3
        assert "target_error" in cfg


class TestLRDMCExtFanOut:
    """Each alat produces a sub-Container internally; verify the helper builds them."""

    @pytest.mark.light
    def test_create_subworkflow_per_alat(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        wf = LRDMC_Ext_Workflow(alat_list=[0.5, 0.3], polynomial_order=1, **_OK)
        # The internal method that builds a Container for a given alat is
        # _make_lrdmc_workflow (see lrdmc_ext_workflow.py).  Call it directly
        # and verify it produces a Container with the right alat/dirname
        # passed through. This must not silently degrade to a trivial
        # input-echo assert if the helper is ever renamed again: a missing
        # attribute is a real regression, not an "equivalent" fallback.
        assert hasattr(wf, "_make_lrdmc_workflow"), (
            "LRDMC_Ext_Workflow lost its per-alat Container-building helper "
            "(expected _make_lrdmc_workflow); this test's real verification "
            "would otherwise silently degrade to a trivial input-echo assert"
        )
        sub = wf._make_lrdmc_workflow(alat=0.5)
        assert sub.dirname == "lrdmc_alat_0.500"
        assert sub.workflow.alat == 0.5
