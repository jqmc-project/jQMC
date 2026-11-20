"""Tests for jqmc_workflow.workflow -- Workflow base, Container, FileFrom, ValueFrom.

Layer 1 (helpers + dataclass behavior) + Layer 2 (Container filesystem prep
without actually running workflows). The async_launch / _submit_and_wait
paths are integration scope and exercised via per-flavor workflow tests later,
except for the queue-limit retry of _submit_and_wait (fake job, no sleeping).

Test class layout:
    TestFileFrom              -- FileFrom dataclass repr/attrs
    TestValueFrom             -- ValueFrom dataclass repr/attrs
    TestWorkflowConstruction  -- base Workflow defaults
    TestWorkflowFilenameHelpers -- _input_filename / _output_filename / _submit_script_name
    TestWorkflowRunIdGenerator -- _new_run_id format + uniqueness
    TestWorkflowEnsureProjectDir
    TestWorkflowGetCommand    -- _get_command raises NotImplementedError
    TestWorkflowConfigureRun  -- configure() + run() base behavior
    TestSubmitAndWaitQueueLimit -- max_job_submit reached: wait and retry; errors still fail
    TestWorkflowCleanupFiles  -- _cleanup_files local mode + protected basenames
    TestContainerConstruction -- Container defaults + dirs
    TestContainerDstBasename  -- _dst_basename rename logic
    TestContainerPrepare      -- _prepare creates dir + copies inputs + writes state
    TestContainerCopyInputFiles -- _copy_input_files raises on missing source
    TestContainerEnsureInputFiles -- _ensure_input_files fills gaps idempotently
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
import os
from types import SimpleNamespace

import pytest

from jqmc_workflow import workflow as workflow_module
from jqmc_workflow.state import STATE_FILENAME, WorkflowStatus, get_job, read_state
from jqmc_workflow.workflow import (
    Container,
    FileFrom,
    ValueFrom,
    Workflow,
)

pytestmark = [pytest.mark.with_chdir]


# -- FileFrom / ValueFrom --------------------------------------------


class TestFileFrom:
    @pytest.mark.light
    def test_attrs(self):
        f = FileFrom("upstream", "result.h5")
        assert f.label == "upstream"
        assert f.filename == "result.h5"

    @pytest.mark.light
    def test_repr_contains_label_and_filename(self):
        r = repr(FileFrom("up", "x.h5"))
        assert "up" in r and "x.h5" in r and "FileFrom" in r

    @pytest.mark.light
    def test_filename_can_be_valuefrom(self):
        """Dynamic filename pattern: FileFrom whose filename is a ValueFrom."""
        f = FileFrom("up", ValueFrom("up", "optimized_hamiltonian"))
        assert isinstance(f.filename, ValueFrom)
        assert f.filename.label == "up"
        assert f.filename.key == "optimized_hamiltonian"


class TestValueFrom:
    @pytest.mark.light
    def test_attrs(self):
        v = ValueFrom("upstream", "energy")
        assert v.label == "upstream"
        assert v.key == "energy"

    @pytest.mark.light
    def test_repr_contains_label_and_key(self):
        r = repr(ValueFrom("up", "e"))
        assert "up" in r and "e" in r and "ValueFrom" in r


# -- Workflow base ---------------------------------------------------


class TestWorkflowConstruction:
    @pytest.mark.light
    def test_defaults(self):
        wf = Workflow()
        assert wf.status == WorkflowStatus.PENDING
        assert wf.output_files == []
        assert wf.output_values == {}
        assert wf.project_dir is None
        assert wf.cleanup_patterns == []

    @pytest.mark.light
    def test_with_project_dir(self, tmp_path):
        wf = Workflow(project_dir=str(tmp_path))
        assert os.path.isabs(wf.project_dir)
        assert os.path.realpath(wf.project_dir) == os.path.realpath(str(tmp_path))

    @pytest.mark.light
    def test_with_cleanup_patterns(self):
        wf = Workflow(cleanup_patterns=["*.h5", "scratch_*"])
        assert wf.cleanup_patterns == ["*.h5", "scratch_*"]

    @pytest.mark.light
    def test_independent_default_lists(self):
        """Default empty list / dict are independent across instances."""
        a = Workflow()
        b = Workflow()
        a.output_files.append("x.o")
        a.output_values["k"] = 1
        assert b.output_files == []
        assert b.output_values == {}


class TestWorkflowFilenameHelpers:
    """Filename helpers depend on self.jobname which subclasses set."""

    @pytest.mark.light
    def test_input_filename_format(self):
        wf = Workflow()
        wf.jobname = "myjob"
        assert wf._input_filename(0, "abc12345") == "input_myjob_0_abc12345.toml"

    @pytest.mark.light
    def test_output_filename_format(self):
        wf = Workflow()
        wf.jobname = "myjob"
        assert wf._output_filename(3, "deadbeef") == "output_myjob_3_deadbeef.out"

    @pytest.mark.light
    def test_submit_script_name_format(self):
        wf = Workflow()
        assert wf._submit_script_name("abc12345") == "submit_abc12345.sh"


class TestWorkflowRunIdGenerator:
    @pytest.mark.light
    def test_returns_8_char_hex(self):
        run_id = Workflow._new_run_id()
        assert len(run_id) == 8
        int(run_id, 16)  # must be valid hex

    @pytest.mark.light
    def test_two_calls_differ(self):
        # 8 hex chars => collision prob ~= 1 / 4 billion.
        assert Workflow._new_run_id() != Workflow._new_run_id()


class TestWorkflowEnsureProjectDir:
    @pytest.mark.light
    def test_resolves_to_cwd_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        wf = Workflow()
        wf._ensure_project_dir()
        assert os.path.realpath(wf.project_dir) == os.path.realpath(str(tmp_path))

    @pytest.mark.light
    def test_keeps_explicit_value(self, tmp_path):
        explicit = tmp_path / "elsewhere"
        explicit.mkdir()
        wf = Workflow(project_dir=str(explicit))
        wf._ensure_project_dir()
        assert os.path.realpath(wf.project_dir) == os.path.realpath(str(explicit))


class TestWorkflowGetCommand:
    @pytest.mark.light
    def test_raises_not_implemented(self):
        wf = Workflow()
        with pytest.raises(NotImplementedError, match="_get_command"):
            wf._get_command("input.toml")


class TestWorkflowConfigureRun:
    @pytest.mark.light
    def test_configure_returns_empty_dict(self):
        wf = Workflow()
        assert wf.configure() == {}

    @pytest.mark.light
    def test_run_returns_status_tuple(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        wf = Workflow()
        status, files, values = asyncio.run(wf.run())
        assert status == WorkflowStatus.PENDING
        assert files == []
        assert values == {}

    @pytest.mark.light
    def test_can_resume_after_completed_default_false(self):
        wf = Workflow()
        assert wf.can_resume_after_completed("/tmp/some/dir") is False


class _QueueLimitedJob:
    """Stand-in for JobSubmission: the queue is full for the first *full_attempts* submissions."""

    max_job_submit = 4

    def __init__(self, full_attempts=0, error=None):
        self.server_machine = SimpleNamespace(queuing=True)
        self.job_stdout = "job.o"
        self.job_stderr = "job.e"
        self.full_attempts = full_attempts
        self.error = error
        self.submit_calls = 0

    def generate_script(self, submission_script, work_dir=None):
        pass

    def job_submit(self, submission_script, from_objects=None, work_dir=None):
        self.submit_calls += 1
        if self.error is not None:
            raise self.error
        if self.submit_calls <= self.full_attempts:
            return False, None
        return True, "1000030"

    def jobcheck(self):
        return False

    def job_acct(self):
        return None

    def fetch_job(self, **kwargs):
        pass

    def _close_ssh(self):
        pass


class TestSubmitAndWaitQueueLimit:
    @staticmethod
    def _workflow(job):
        wf = Workflow()
        wf.server_machine_name = "my-cluster"
        wf.hamiltonian_file = "hamiltonian_data.h5"
        wf.queue_label = "default"
        wf.jobname = "jqmc-wf"
        wf.poll_interval = 7
        wf._make_job = lambda *args, **kwargs: job
        return wf

    @pytest.fixture
    def sleeps(self, monkeypatch):
        """Record asyncio.sleep delays in workflow.py instead of sleeping."""
        delays = []

        async def fake_sleep(delay):
            delays.append(delay)

        monkeypatch.setattr(workflow_module.asyncio, "sleep", fake_sleep)
        return delays

    @pytest.mark.light
    def test_waits_and_retries_until_a_slot_frees(self, tmp_path, sleeps):
        """A full queue delays the submission by poll_interval per attempt; it does not fail."""
        job = _QueueLimitedJob(full_attempts=2)
        wf = self._workflow(job)
        asyncio.run(wf._submit_and_wait("input.toml", "out.o", work_dir=str(tmp_path), step=1, run_id="abc"))
        assert job.submit_calls == 3
        assert sleeps == [wf.poll_interval, wf.poll_interval]
        record = get_job(str(tmp_path), "input.toml")
        assert record["job_id"] == "1000030"
        assert record["queue_label"] == "default"
        assert record["status"] == "fetched"

    @pytest.mark.light
    def test_submission_error_still_fails(self, tmp_path, sleeps):
        """An exception from job_submit is not retried."""
        job = _QueueLimitedJob(error=RuntimeError("qsub: Unknown queue"))
        wf = self._workflow(job)
        with pytest.raises(RuntimeError, match="Unknown queue"):
            asyncio.run(wf._submit_and_wait("input.toml", "out.o", work_dir=str(tmp_path), step=1, run_id="abc"))
        assert job.submit_calls == 1
        assert sleeps == []
        assert get_job(str(tmp_path), "input.toml") == {}


class TestWorkflowCleanupFiles:
    """Local-only cleanup (no server_machine_name set)."""

    @pytest.mark.light
    def test_noop_when_no_patterns(self, tmp_path):
        (tmp_path / "x.h5").write_text("x")
        wf = Workflow(project_dir=str(tmp_path))
        wf._cleanup_files()
        assert (tmp_path / "x.h5").exists()

    @pytest.mark.light
    def test_noop_when_no_project_dir(self):
        wf = Workflow(cleanup_patterns=["*.h5"])
        wf._cleanup_files()  # early return; no error

    @pytest.mark.light
    def test_deletes_matching_files(self, tmp_path):
        for name in ("a.h5", "b.h5", "keep.txt"):
            (tmp_path / name).write_text("x")
        wf = Workflow(project_dir=str(tmp_path), cleanup_patterns=["*.h5"])
        wf._cleanup_files()
        assert not (tmp_path / "a.h5").exists()
        assert not (tmp_path / "b.h5").exists()
        assert (tmp_path / "keep.txt").exists()

    @pytest.mark.light
    def test_recurses_into_subdirs(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.h5").write_text("x")
        wf = Workflow(project_dir=str(tmp_path), cleanup_patterns=["*.h5"])
        wf._cleanup_files()
        assert not (sub / "deep.h5").exists()

    @pytest.mark.light
    def test_protected_workflow_state_not_deleted(self, tmp_path):
        (tmp_path / "workflow_state.toml").write_text("[workflow]\n")
        (tmp_path / "other.toml").write_text("[x]\n")
        wf = Workflow(project_dir=str(tmp_path), cleanup_patterns=["*.toml"])
        wf._cleanup_files()
        assert (tmp_path / "workflow_state.toml").exists()
        assert not (tmp_path / "other.toml").exists()

    @pytest.mark.light
    def test_protected_state_toml_tmp_also_spared(self, tmp_path):
        (tmp_path / "workflow_state.toml.tmp").write_text("[workflow]\n")
        wf = Workflow(project_dir=str(tmp_path), cleanup_patterns=["*.toml.tmp"])
        wf._cleanup_files()
        assert (tmp_path / "workflow_state.toml.tmp").exists()


# -- Container -------------------------------------------------------


class TestContainerConstruction:
    @pytest.mark.light
    def test_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        c = Container()
        assert c.label == "workflow"
        assert c.dirname == "workflow"
        assert c.input_files == []
        assert c.rename_input_files == []
        assert isinstance(c.workflow, Workflow)
        assert c.status == "init"
        assert c.output_files == []
        assert c.output_values == {}

    @pytest.mark.light
    def test_explicit_args(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        wf = Workflow()
        c = Container(
            label="myjob",
            dirname="my_dir",
            input_files=["a.toml", "b.h5"],
            rename_input_files=["renamed_a.toml", None],
            workflow=wf,
        )
        assert c.label == "myjob"
        assert c.dirname == "my_dir"
        assert c.input_files == ["a.toml", "b.h5"]
        assert c.rename_input_files == ["renamed_a.toml", None]
        assert c.workflow is wf

    @pytest.mark.light
    def test_project_dir_is_cwd_plus_dirname(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        c = Container(label="x", dirname="my_proj")
        parent = os.path.realpath(os.path.dirname(c.project_dir))
        assert parent == os.path.realpath(str(tmp_path))
        assert os.path.basename(c.project_dir) == "my_proj"


class TestContainerDstBasename:
    @pytest.mark.parametrize(
        "src,rename_list,index,expected",
        [
            ("/path/to/source.h5", [], 0, "source.h5"),
            ("/path/to/source.h5", ["renamed.h5"], 0, "renamed.h5"),
            ("/path/source.h5", ["a", "b", "c"], 1, "b"),
            ("/path/source.h5", ["a"], 5, "source.h5"),  # out-of-range
            ("/path/source.h5", [None, "renamed.h5"], 0, "source.h5"),
            ("/path/source.h5", [None, "renamed.h5"], 1, "renamed.h5"),
            ("/path/source.h5", ["/elsewhere/foo.h5"], 0, "foo.h5"),  # collapses to basename
        ],
    )
    @pytest.mark.light
    def test_resolution(self, src, rename_list, index, expected):
        assert Container._dst_basename(src, rename_list, index) == expected


class TestContainerPrepare:
    @pytest.mark.light
    def test_creates_project_dir_and_writes_state(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        c = Container(label="lbl", dirname="proj", input_files=[])
        c._prepare()
        proj = tmp_path / "proj"
        assert proj.exists()
        assert (proj / STATE_FILENAME).exists()
        state = read_state(str(proj))
        assert state["workflow"]["label"] == "lbl"
        assert state["workflow"]["status"] == "pending"

    @pytest.mark.light
    def test_copies_input_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("hello")
        c = Container(label="x", dirname="proj", input_files=["src.txt"])
        c._prepare()
        assert (tmp_path / "proj" / "src.txt").read_text() == "hello"

    @pytest.mark.light
    def test_renames_input_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src.txt"
        src.write_text("hello")
        c = Container(
            label="x",
            dirname="proj",
            input_files=["src.txt"],
            rename_input_files=["renamed.txt"],
        )
        c._prepare()
        assert (tmp_path / "proj" / "renamed.txt").read_text() == "hello"
        assert not (tmp_path / "proj" / "src.txt").exists()

    @pytest.mark.light
    def test_short_circuits_when_completed(self, tmp_path, monkeypatch):
        """A pre-existing project dir with status='completed' is not re-prepared."""
        monkeypatch.chdir(tmp_path)
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / STATE_FILENAME).write_text("[workflow]\nlabel = 'x'\ntype = 'W'\nstatus = 'completed'\n")
        c = Container(label="x", dirname="proj", input_files=[])
        c._prepare()
        state = read_state(str(proj))
        assert state["workflow"]["status"] == "completed"


class TestContainerCopyInputFiles:
    @pytest.mark.light
    def test_missing_input_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        proj = tmp_path / "proj"
        proj.mkdir()
        c = Container(label="x", dirname="proj", input_files=["nonexistent.h5"])
        with pytest.raises(FileNotFoundError, match="Required input not found"):
            c._copy_input_files()

    @pytest.mark.light
    def test_copies_directory_recursively(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src_dir = tmp_path / "src_dir"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("a")
        (src_dir / "b.txt").write_text("b")
        proj = tmp_path / "proj"
        proj.mkdir()
        c = Container(label="x", dirname="proj", input_files=["src_dir"])
        c._copy_input_files()
        assert (proj / "src_dir" / "a.txt").read_text() == "a"
        assert (proj / "src_dir" / "b.txt").read_text() == "b"


class TestContainerEnsureInputFiles:
    @pytest.mark.light
    def test_fills_gaps_without_overwriting(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src_a = tmp_path / "a.txt"
        src_a.write_text("original a")
        src_b = tmp_path / "b.txt"
        src_b.write_text("original b")
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "a.txt").write_text("EDITED")
        c = Container(label="x", dirname="proj", input_files=["a.txt", "b.txt"])
        c._ensure_input_files()
        # a.txt left untouched, b.txt filled in.
        assert (proj / "a.txt").read_text() == "EDITED"
        assert (proj / "b.txt").read_text() == "original b"
