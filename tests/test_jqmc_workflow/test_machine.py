"""Tests for jqmc_workflow.machine -- Machine / Machines_handler / registry.

Layer 1 (construction-time validation) + Layer 2 (local-mode subprocess).

Test class layout:
    TestMachineConstruction    -- yaml load + validation (no subprocess)
    TestMachineProperties      -- machine_data.yaml field accessors
    TestMachineLocalMode       -- machine_type=local, real subprocess
    TestMachineLocalFilesystem -- is_file / is_dir / exist (local)
    TestMachineLocalJobAPI     -- get_job_list / delete_job via local stubs
    TestMachineRegistry        -- get_machine / close_all_machines cache
    TestListMachines           -- machine catalog helper
    TestProbeEnvironment       -- connectivity probe (local only)
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
from types import SimpleNamespace

import pytest

from jqmc_workflow.machine import (
    Machine,
    close_all_machines,
    get_machine,
    list_machines,
    probe_environment,
)

pytestmark = [pytest.mark.with_chdir]


class TestMachineConstruction:
    """yaml load + validation -- no subprocess, no SSH."""

    @pytest.mark.light
    def test_loads_machine_from_yaml(self, local_machine_yaml):
        m = Machine("local-test")
        assert m.name == "local-test"
        assert m.machine_type == "local"

    @pytest.mark.light
    def test_unknown_machine_raises_key_error(self, local_machine_yaml):
        with pytest.raises(KeyError):
            Machine("nonexistent-machine")

    @pytest.mark.light
    def test_missing_yaml_raises(self, tmp_jqmc_setting_dir):
        """The setting dir exists but no machine_data.yaml in it."""
        with pytest.raises(FileNotFoundError):
            Machine("any")

    @pytest.mark.light
    def test_remote_without_ssh_host_raises(self, tmp_jqmc_setting_dir):
        """Validation: machine_type=remote requires an ssh_host field."""
        (tmp_jqmc_setting_dir / "machine_data.yaml").write_text(
            "broken:\n  machine_type: remote\n  queuing: false\n  jobsubmit: x\n  jobcheck: x\n  jobdel: x\n  jobnum_index: 1\n"
        )
        with pytest.raises(KeyError, match="ssh_host"):
            Machine("broken")

    @pytest.mark.light
    def test_bootstrap_when_setting_dir_missing(self, tmp_path, monkeypatch):
        """First-run bootstrap: setting dir absent -> copy template, raise FileNotFoundError."""
        from jqmc_workflow import _config

        absent = tmp_path / "absent_setting"
        _config.reset_config_dir()
        monkeypatch.setattr(_config, "_resolved_config_dir", str(absent))
        with pytest.raises(FileNotFoundError, match="configure"):
            Machine("any")
        assert absent.exists()
        _config.reset_config_dir()


class TestMachineProperties:
    """Read-only accessors for machine_data.yaml fields."""

    @pytest.mark.light
    def test_basic_properties(self, local_machine):
        m = local_machine
        assert m.name == "local-test"
        assert m.machine_type == "local"
        assert m.queuing is False
        assert m.jobsubmit == "/bin/echo"
        assert m.jobcheck == "/usr/bin/true"
        assert m.jobdel == "/usr/bin/true"
        assert m.jobnum_index == 1

    @pytest.mark.light
    def test_workspace_root_set(self, local_machine):
        assert local_machine.workspace_root is not None
        assert Path(local_machine.workspace_root).exists()

    @pytest.mark.light
    def test_username_falls_back_to_os_login(self, local_machine, monkeypatch):
        # Production code falls back to ``os.getlogin()``, which needs a
        # controlling TTY that headless CI runners lack; pin its answer so the
        # fallback is judged in every environment.
        import os

        monkeypatch.setattr(os, "getlogin", lambda: "fallback-login")
        assert local_machine.username == "fallback-login"

    @pytest.mark.light
    def test_username_setter(self, local_machine):
        local_machine.username = "custom"
        assert local_machine.username == "custom"

    @pytest.mark.light
    def test_invalid_machine_type_raises(self, tmp_jqmc_setting_dir):
        """Invalid machine_type is rejected at construction time."""
        (tmp_jqmc_setting_dir / "machine_data.yaml").write_text(
            "bogus:\n  machine_type: WRONG\n  queuing: false\n  jobsubmit: x\n  jobcheck: x\n  jobdel: x\n  jobnum_index: 1\n"
        )
        with pytest.raises(ValueError, match="machine_type must be"):
            Machine("bogus")

    @pytest.mark.light
    def test_missing_required_key_raises(self, local_machine):
        """Accessing a required (no-default) property that's absent raises."""
        local_machine.data.pop("jobcheck", None)
        with pytest.raises(KeyError, match="jobcheck"):
            local_machine.jobcheck

    @pytest.mark.light
    def test_ip_defaults_to_none(self, local_machine):
        assert local_machine.ip is None


class TestMachineLocalMode:
    """machine_type=local -- exercises subprocess on localhost."""

    @pytest.mark.light
    def test_run_simple_echo(self, local_machine):
        stdout, stderr = local_machine.run_command("echo hello")
        assert stdout.strip() == "hello"
        assert stderr == ""

    @pytest.mark.light
    def test_run_command_in_execute_dir(self, local_machine, tmp_path):
        import os

        stdout, _ = local_machine.run_command("pwd", execute_dir=str(tmp_path))
        # macOS resolves /tmp -> /private/tmp, so compare via realpath.
        assert os.path.realpath(stdout.strip()) == os.path.realpath(str(tmp_path))

    @pytest.mark.light
    def test_run_in_missing_dir_raises(self, local_machine, tmp_path):
        with pytest.raises(FileNotFoundError):
            local_machine.run_command("echo x", execute_dir=str(tmp_path / "no_such_dir"))

    @pytest.mark.light
    def test_non_zero_exit_raises(self, local_machine, monkeypatch):
        """A failing local command raises RuntimeError at once, without retry sleeps.

        A non-zero exit status of a local command is deterministic, so the
        runner must not sleep and repeat it; ``time.sleep`` is patched to fail
        loudly if it is ever called.
        """
        import jqmc_workflow.machine as machine_module

        def _no_sleep(_seconds):
            raise AssertionError("a non-zero exit status must not be retried")

        # Replace this module's dependency, not ``time.sleep`` on the shared
        # stdlib module: ``subprocess.run`` and the CI resource monitor use the
        # same module object and must remain free to perform internal polling.
        monkeypatch.setattr(machine_module, "time", SimpleNamespace(sleep=_no_sleep))
        with pytest.raises(RuntimeError, match="rc=3"):
            local_machine.run_command("exit 3")

    @pytest.mark.light
    def test_execute_dir_with_space_is_quoted(self, local_machine, tmp_path):
        """A working directory containing a space is passed to the shell intact."""
        workdir = tmp_path / "dir with space"
        workdir.mkdir()
        (workdir / "hello.txt").write_text("hi\n")
        stdout, _ = local_machine.run_command("cat hello.txt", execute_dir=str(workdir))
        assert stdout.strip() == "hi"

    @pytest.mark.light
    def test_captures_stderr_separately(self, local_machine):
        stdout, stderr = local_machine.run_command("echo out; echo err 1>&2")
        assert "out" in stdout
        assert "err" in stderr


class TestMachineLocalFilesystem:
    """is_file / is_dir / exist on local filesystem."""

    @pytest.mark.light
    def test_is_file_true_for_regular_file(self, local_machine, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("hello")
        assert local_machine.is_file(str(p)) is True

    @pytest.mark.light
    def test_is_file_false_for_directory(self, local_machine, tmp_path):
        assert local_machine.is_file(str(tmp_path)) is False

    @pytest.mark.light
    def test_is_file_false_for_missing(self, local_machine, tmp_path):
        assert local_machine.is_file(str(tmp_path / "no")) is False

    @pytest.mark.light
    def test_is_dir_true_for_directory(self, local_machine, tmp_path):
        assert local_machine.is_dir(str(tmp_path)) is True

    @pytest.mark.light
    def test_is_dir_false_for_file(self, local_machine, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("x")
        assert local_machine.is_dir(str(p)) is False

    @pytest.mark.light
    def test_exist_true_for_file(self, local_machine, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("x")
        assert local_machine.exist(str(p)) is True

    @pytest.mark.light
    def test_exist_true_for_directory(self, local_machine, tmp_path):
        assert local_machine.exist(str(tmp_path)) is True

    @pytest.mark.light
    def test_exist_false_for_missing(self, local_machine, tmp_path):
        assert local_machine.exist(str(tmp_path / "no")) is False


class TestMachineLocalJobAPI:
    """get_job_list / delete_job dispatched through local subprocess."""

    @pytest.mark.light
    def test_get_job_list_runs_jobcheck(self, local_machine):
        """jobcheck=/usr/bin/true returns empty stdout."""
        stdout, _ = local_machine.get_job_list()
        assert stdout == ""

    @pytest.mark.light
    def test_get_job_list_as_text_splits_lines(self, local_machine):
        lines = local_machine.get_job_list_as_text()
        # /usr/bin/true outputs "" -> split returns [""].
        assert lines == [""]

    @pytest.mark.light
    def test_delete_job_quotes_arg(self, local_machine, tmp_jqmc_setting_dir):
        """jobdel=/bin/echo so delete_job(jobid) prints the id back."""
        yaml_path = tmp_jqmc_setting_dir / "machine_data.yaml"
        text = yaml_path.read_text().replace("jobdel: /usr/bin/true", "jobdel: /bin/echo")
        yaml_path.write_text(text)
        close_all_machines()
        m = Machine("local-test")
        out_lines = m.delete_job("12345")
        assert "12345" in out_lines[0]


class TestMachineRegistry:
    """get_machine caches Machine instances; close_all_machines clears the cache."""

    @pytest.mark.light
    def test_get_machine_caches_instance(self, local_machine_yaml, fresh_machine_registry):
        m1 = get_machine("local-test")
        m2 = get_machine("local-test")
        assert m1 is m2

    @pytest.mark.light
    def test_close_all_machines_drops_cache(self, local_machine_yaml, fresh_machine_registry):
        m1 = get_machine("local-test")
        close_all_machines()
        m2 = get_machine("local-test")
        assert m1 is not m2

    @pytest.mark.light
    def test_close_all_machines_safe_on_empty_registry(self, fresh_machine_registry):
        close_all_machines()
        close_all_machines()


class TestListMachines:
    @pytest.mark.light
    def test_returns_summary_for_each_machine(self, local_machine_yaml):
        result = list_machines()
        names = {e["name"] for e in result}
        assert "local-test" in names
        assert "localhost" in names
        entry = next(e for e in result if e["name"] == "local-test")
        assert entry["machine_type"] == "local"
        assert entry["queuing"] is False

    @pytest.mark.light
    def test_returns_empty_when_yaml_missing(self, tmp_jqmc_setting_dir):
        result = list_machines()
        assert result == []

    @pytest.mark.light
    def test_returns_empty_when_yaml_empty(self, tmp_jqmc_setting_dir):
        (tmp_jqmc_setting_dir / "machine_data.yaml").write_text("")
        result = list_machines()
        assert result == []

    @pytest.mark.light
    def test_multiple_machines_listed(self, tmp_jqmc_setting_dir, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (tmp_jqmc_setting_dir / "machine_data.yaml").write_text(
            textwrap.dedent(
                f"""
                a:
                  machine_type: local
                  workspace_root: {ws}
                  queuing: false
                  jobsubmit: /bin/echo
                  jobcheck: /usr/bin/true
                  jobdel: /usr/bin/true
                  jobnum_index: 1
                b:
                  machine_type: local
                  workspace_root: {ws}
                  queuing: true
                  jobsubmit: /bin/echo
                  jobcheck: /usr/bin/true
                  jobdel: /usr/bin/true
                  jobnum_index: 1
                """
            ).strip()
            + "\n"
        )
        names = {e["name"] for e in list_machines()}
        assert names == {"a", "b"}


class TestProbeEnvironment:
    @pytest.mark.light
    def test_local_machine_always_reachable(self, local_machine_yaml):
        result = probe_environment("local-test")
        assert result["machine_name"] == "local-test"
        assert result["machine_type"] == "local"
        assert result["reachable"] is True
        assert "error" not in result

    @pytest.mark.light
    def test_unknown_machine_raises(self, local_machine_yaml):
        with pytest.raises(KeyError):
            probe_environment("nonexistent-machine")
