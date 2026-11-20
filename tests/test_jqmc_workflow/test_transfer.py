"""Tests for jqmc_workflow._transfer -- Data_transfer in local mode.

Layer 1 (helpers) + Layer 2 (local-server early-return paths, recursive
local cleanup). Remote SFTP paths (put/get to a real server, glob expansion
via sftp.listdir) deferred to Layer 4.

Test class layout:
    TestDataTransferConstruction -- __init__ + local-root resolution
    TestPutObjectsLocal          -- local-mode early-return
    TestGetObjectsLocal          -- local-mode early-return
    TestRemoveObjectsLocal       -- recursive cleanup + protected_basenames
    TestRemoveObjectsErrorPaths  -- work_dir-not-under-root + remote skip
    TestSshClose                 -- no-op
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

import textwrap
from pathlib import Path

import pytest

from jqmc_workflow.machine import close_all_machines
from jqmc_workflow._transfer import Data_transfer

pytestmark = [pytest.mark.with_chdir]


class TestDataTransferConstruction:
    @pytest.mark.light
    def test_basic_construction(self, local_machine_yaml):
        dt = Data_transfer(server_machine_name="local-test")
        assert dt.server_machine.name == "local-test"
        assert dt.safe_mode is False
        assert dt._local_root is not None

    @pytest.mark.light
    def test_local_root_from_localhost_workspace_root(self, local_machine_yaml, tmp_path):
        """_local_root is read from the localhost entry's workspace_root."""
        dt = Data_transfer(server_machine_name="local-test")
        # localhost workspace_root is set to tmp_path/workspace by the fixture.
        assert (tmp_path / "workspace").resolve() == Path(dt._local_root).resolve()

    @pytest.mark.light
    def test_safe_mode_flag_persists(self, local_machine_yaml):
        dt = Data_transfer(server_machine_name="local-test", safe_mode=True)
        assert dt.safe_mode is True

    @pytest.mark.light
    def test_local_root_fallback_when_no_localhost_entry(self, tmp_jqmc_setting_dir, tmp_path):
        """Without a localhost entry, _local_root falls back to expanduser('~')."""
        import os

        (tmp_jqmc_setting_dir / "machine_data.yaml").write_text(
            textwrap.dedent(
                f"""
                local-test:
                  machine_type: local
                  workspace_root: {tmp_path / "ws"}
                  queuing: false
                  jobsubmit: /bin/echo
                  jobcheck: /usr/bin/true
                  jobdel: /usr/bin/true
                  jobnum_index: 1
                """
            ).strip()
            + "\n"
        )
        (tmp_path / "ws").mkdir()
        close_all_machines()
        dt = Data_transfer(server_machine_name="local-test")
        assert dt._local_root == os.path.expanduser("~")


class TestPutObjectsLocal:
    """server is local -> put_objects is a no-op."""

    @pytest.mark.light
    def test_returns_silently_for_local_server(self, local_machine_yaml, tmp_path):
        dt = Data_transfer(server_machine_name="local-test")
        # Should not raise even though we pass nonsense work_dir.
        dt.put_objects(work_dir=str(tmp_path))

    @pytest.mark.light
    def test_with_objects_still_returns_silently(self, local_machine_yaml, tmp_path):
        dt = Data_transfer(server_machine_name="local-test")
        (tmp_path / "x.txt").write_text("x")
        dt.put_objects(from_objects=["x.txt"], work_dir=str(tmp_path))


class TestGetObjectsLocal:
    """server is local -> get_objects is a no-op."""

    @pytest.mark.light
    def test_returns_silently_for_local_server(self, local_machine_yaml, tmp_path):
        dt = Data_transfer(server_machine_name="local-test")
        dt.get_objects(work_dir=str(tmp_path))

    @pytest.mark.light
    def test_with_objects_returns_silently(self, local_machine_yaml, tmp_path):
        dt = Data_transfer(server_machine_name="local-test")
        dt.get_objects(from_objects=["*.h5"], work_dir=str(tmp_path))


class TestRemoveObjectsLocal:
    """Recursive local deletion is exercised even when server is local."""

    @pytest.mark.light
    def test_removes_matching_files_in_top_level(self, local_machine_yaml, tmp_path):
        ws = tmp_path / "workspace" / "job1"
        ws.mkdir(parents=True)
        (ws / "restart.h5").write_text("x")
        (ws / "keep.txt").write_text("k")
        dt = Data_transfer(server_machine_name="local-test")
        dt.remove_objects(patterns=["restart.h5"], work_dir=str(ws))
        assert not (ws / "restart.h5").exists()
        assert (ws / "keep.txt").exists()

    @pytest.mark.light
    def test_removes_matching_files_recursively(self, local_machine_yaml, tmp_path):
        ws = tmp_path / "workspace" / "job1"
        nested = ws / "_pilot" / "_inner"
        nested.mkdir(parents=True)
        (ws / "restart.h5").write_text("x")
        (ws / "_pilot" / "restart.h5").write_text("x")
        (nested / "restart.h5").write_text("x")
        dt = Data_transfer(server_machine_name="local-test")
        dt.remove_objects(patterns=["restart.h5"], work_dir=str(ws))
        assert not (ws / "restart.h5").exists()
        assert not (ws / "_pilot" / "restart.h5").exists()
        assert not (nested / "restart.h5").exists()

    @pytest.mark.light
    def test_glob_pattern_matches_multiple(self, local_machine_yaml, tmp_path):
        ws = tmp_path / "workspace" / "job1"
        ws.mkdir(parents=True)
        for name in ("hamiltonian_opt_1.h5", "hamiltonian_opt_2.h5", "result.h5"):
            (ws / name).write_text("x")
        dt = Data_transfer(server_machine_name="local-test")
        dt.remove_objects(patterns=["hamiltonian_opt*.h5"], work_dir=str(ws))
        assert not (ws / "hamiltonian_opt_1.h5").exists()
        assert not (ws / "hamiltonian_opt_2.h5").exists()
        assert (ws / "result.h5").exists()  # not matched

    @pytest.mark.light
    def test_protected_basename_not_deleted(self, local_machine_yaml, tmp_path):
        ws = tmp_path / "workspace" / "job1"
        ws.mkdir(parents=True)
        (ws / "workflow_state.toml").write_text("[workflow]\n")
        (ws / "other.toml").write_text("[x]\n")
        dt = Data_transfer(server_machine_name="local-test")
        dt.remove_objects(
            patterns=["*.toml"],
            work_dir=str(ws),
            protected_basenames={"workflow_state.toml"},
        )
        assert (ws / "workflow_state.toml").exists()
        assert not (ws / "other.toml").exists()

    @pytest.mark.light
    def test_directories_are_skipped(self, local_machine_yaml, tmp_path):
        """remove_objects only deletes files, not directories matching the pattern."""
        ws = tmp_path / "workspace" / "job1"
        ws.mkdir(parents=True)
        (ws / "data.h5").write_text("x")
        (ws / "data_dir").mkdir()
        dt = Data_transfer(server_machine_name="local-test")
        dt.remove_objects(patterns=["data*"], work_dir=str(ws))
        assert not (ws / "data.h5").exists()
        assert (ws / "data_dir").exists()  # directory survives

    @pytest.mark.light
    def test_no_match_is_noop(self, local_machine_yaml, tmp_path):
        ws = tmp_path / "workspace" / "job1"
        ws.mkdir(parents=True)
        (ws / "keep.txt").write_text("k")
        dt = Data_transfer(server_machine_name="local-test")
        dt.remove_objects(patterns=["nonexistent.h5"], work_dir=str(ws))
        assert (ws / "keep.txt").exists()


class TestSshClose:
    @pytest.mark.light
    def test_is_noop(self, local_machine_yaml):
        """ssh_close on Data_transfer is a no-op (Machine is process-shared)."""
        dt = Data_transfer(server_machine_name="local-test")
        # Should not raise and should not affect machine state.
        dt.ssh_close()
        dt.ssh_close()
