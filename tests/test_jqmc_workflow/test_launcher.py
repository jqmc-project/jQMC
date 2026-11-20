"""Tests for jqmc_workflow.launcher.Launcher -- DAG construction + validation.

Layer 2 (no SSH; Workflow base instances without execution). The actual
async_launch() path is covered by Layer 2 workflow tests once those land
(it needs running workflows which are integration-heavy).

Test class layout:
    TestLauncherConstruction    -- empty + multi-workflow construction
    TestLauncherLabelValidation -- duplicate label detection
    TestLauncherDependencyGraph -- _build_dependency_graph + missing dep validation
    TestLauncherDepCollection   -- _collect_deps recursion (list / dict / FileFrom / ValueFrom)
    TestLauncherSessionState    -- get_session_state / current_job / job_history accessors
    TestIsDependencyHelper      -- the _is_dependency predicate
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

from jqmc_workflow.launcher import Launcher
from jqmc_workflow.workflow import (
    Container,
    FileFrom,
    ValueFrom,
    Workflow,
    _is_dependency,
)

pytestmark = [pytest.mark.with_chdir]


def _make_container(label: str, dirname: str = None, workflow=None) -> Container:
    """Return a minimal Container; uses the abstract base Workflow."""
    return Container(
        label=label,
        dirname=dirname or label,
        input_files=[],
        workflow=workflow or Workflow(),
    )


class TestLauncherConstruction:
    @pytest.mark.light
    def test_empty_workflow_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        launcher = Launcher(workflows=[], log_level="WARNING", draw_graph=False)
        assert launcher.workflows == []
        assert launcher.workflows_by_label == {}
        assert launcher.dependency_dict == {}

    @pytest.mark.light
    def test_none_workflows_treated_as_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        launcher = Launcher(workflows=None, log_level="WARNING")
        assert launcher.workflows == []

    @pytest.mark.light
    def test_multi_workflow_construction(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        a = _make_container("a")
        b = _make_container("b")
        launcher = Launcher(workflows=[a, b], log_level="WARNING")
        assert set(launcher.workflows_by_label) == {"a", "b"}
        assert launcher.dependency_dict == {"a": (), "b": ()}

    @pytest.mark.light
    def test_root_dir_captured(self, tmp_path, monkeypatch):
        import os

        monkeypatch.chdir(tmp_path)
        launcher = Launcher(workflows=[], log_level="WARNING")
        assert os.path.realpath(launcher.root_dir) == os.path.realpath(str(tmp_path))


class TestLauncherLabelValidation:
    @pytest.mark.light
    def test_duplicate_labels_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        a1 = _make_container("dup", dirname="dup1")
        a2 = _make_container("dup", dirname="dup2")
        with pytest.raises(ValueError, match="Duplicate workflow labels"):
            Launcher(workflows=[a1, a2], log_level="WARNING")


class TestLauncherDependencyGraph:
    @pytest.mark.light
    def test_filefrom_dependency_recorded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        upstream = _make_container("upstream")
        downstream = Container(
            label="downstream",
            dirname="downstream",
            input_files=[FileFrom("upstream", "result.h5")],
            workflow=Workflow(),
        )
        launcher = Launcher(workflows=[upstream, downstream], log_level="WARNING")
        assert launcher.dependency_dict["downstream"] == ("upstream",)
        assert launcher.dependency_dict["upstream"] == ()

    @pytest.mark.light
    def test_valuefrom_dependency_recorded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        upstream = _make_container("upstream")
        wf = Workflow()
        wf.trial_energy = ValueFrom("upstream", "energy")
        downstream = Container(label="downstream", dirname="downstream", workflow=wf)
        launcher = Launcher(workflows=[upstream, downstream], log_level="WARNING")
        assert "upstream" in launcher.dependency_dict["downstream"]

    @pytest.mark.light
    def test_undefined_dependency_label_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        downstream = Container(
            label="downstream",
            dirname="downstream",
            input_files=[FileFrom("ghost-workflow", "x.h5")],
            workflow=Workflow(),
        )
        with pytest.raises(ValueError, match="depends on undefined workflows"):
            Launcher(workflows=[downstream], log_level="WARNING")

    @pytest.mark.light
    def test_diamond_dag(self, tmp_path, monkeypatch):
        """A -> B, A -> C, B+C -> D -- all four labels resolved."""
        monkeypatch.chdir(tmp_path)
        a = _make_container("A")
        b = Container(label="B", dirname="B", input_files=[FileFrom("A", "a.h5")], workflow=Workflow())
        c = Container(label="C", dirname="C", input_files=[FileFrom("A", "a.h5")], workflow=Workflow())
        d_wf = Workflow()
        d_wf.up1 = FileFrom("B", "b.h5")
        d_wf.up2 = ValueFrom("C", "energy")
        d = Container(label="D", dirname="D", workflow=d_wf)
        launcher = Launcher(workflows=[a, b, c, d], log_level="WARNING")
        assert set(launcher.dependency_dict["B"]) == {"A"}
        assert set(launcher.dependency_dict["C"]) == {"A"}
        assert set(launcher.dependency_dict["D"]) == {"B", "C"}
        assert launcher.dependency_dict["A"] == ()


class TestLauncherDepCollection:
    """_collect_deps recursion across various container types."""

    @pytest.mark.light
    def test_filefrom_detected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        launcher = Launcher(workflows=[], log_level="WARNING")
        deps = set()
        launcher._collect_deps(FileFrom("up", "x.h5"), deps)
        assert deps == {"up"}

    @pytest.mark.light
    def test_valuefrom_detected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        launcher = Launcher(workflows=[], log_level="WARNING")
        deps = set()
        launcher._collect_deps(ValueFrom("up", "energy"), deps)
        assert deps == {"up"}

    @pytest.mark.light
    def test_primitives_ignored(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        launcher = Launcher(workflows=[], log_level="WARNING")
        deps = set()
        for primitive in ("string", 42, 3.14, True, None):
            launcher._collect_deps(primitive, deps)
        assert deps == set()

    @pytest.mark.light
    def test_list_recursion(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        launcher = Launcher(workflows=[], log_level="WARNING")
        deps = set()
        launcher._collect_deps([FileFrom("a", "x.h5"), "str", FileFrom("b", "y.h5")], deps)
        assert deps == {"a", "b"}

    @pytest.mark.light
    def test_dict_recursion(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        launcher = Launcher(workflows=[], log_level="WARNING")
        deps = set()
        launcher._collect_deps(
            {"k1": FileFrom("a", "x.h5"), "k2": [ValueFrom("b", "v")]},
            deps,
        )
        assert deps == {"a", "b"}

    @pytest.mark.light
    def test_arbitrary_object_attributes_scanned(self, tmp_path, monkeypatch):
        """An object with __dict__ has its attributes recursively scanned."""
        monkeypatch.chdir(tmp_path)
        launcher = Launcher(workflows=[], log_level="WARNING")

        class Holder:
            pass

        holder = Holder()
        holder.first = FileFrom("up1", "x.h5")
        holder.second = "ignored"
        holder.nested = [ValueFrom("up2", "v")]

        deps = set()
        launcher._collect_deps(holder, deps)
        assert deps == {"up1", "up2"}


class TestLauncherSessionState:
    @pytest.mark.light
    def test_get_session_state_returns_dict(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        a = _make_container("a")
        launcher = Launcher(workflows=[a], log_level="WARNING")
        state = launcher.get_session_state()
        assert isinstance(state, dict)

    @pytest.mark.light
    def test_get_current_job_returns_none_pre_launch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        launcher = Launcher(workflows=[_make_container("a")], log_level="WARNING")
        assert launcher.get_current_job() is None

    @pytest.mark.light
    def test_get_job_history_empty_pre_launch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        launcher = Launcher(workflows=[_make_container("a")], log_level="WARNING")
        history = launcher.get_job_history()
        assert isinstance(history, list)
        assert history == []


class TestIsDependencyHelper:
    """_is_dependency is the predicate used by _collect_deps."""

    @pytest.mark.light
    def test_filefrom_is_dependency(self):
        assert _is_dependency(FileFrom("a", "x.h5")) is True

    @pytest.mark.light
    def test_valuefrom_is_dependency(self):
        assert _is_dependency(ValueFrom("a", "v")) is True

    @pytest.mark.light
    @pytest.mark.parametrize("obj", ["str", 42, 3.14, True, None, [], {}])
    def test_primitives_are_not_dependencies(self, obj):
        assert _is_dependency(obj) is False
