"""Tests for jqmc_workflow._job -- JobSubmission and queue helpers.

Layer 1 (helper functions) + Layer 2 (JobSubmission against local-mode
Machine with /bin/echo as jobsubmit). Layer 4 (remote SSH path) deferred.

Test class layout:
    TestLoadQueueData          -- queue_data.toml loader
    TestGetNumMpi              -- mpi count extraction
    TestJobSubmissionConstruction -- __init__ + config validation
    TestGenerateScript         -- template substitution
    TestJobSubmitLocal         -- job_submit in local mode
    TestJobCheck               -- jobcheck for queuing=false
    TestJobNumCheck            -- max_job_submit limit (driver / user scope)
    TestDriverRestart          -- queued jobs of a previous run (pilots included) hold their slots
    TestDeleteJob              -- delete_job dispatch
    TestJobAcct                -- job_acct guard branches
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
import shlex
import textwrap
from pathlib import Path

import pytest
import toml

from jqmc_workflow import workflow as workflow_module
from jqmc_workflow._job import (
    JobSubmission,
    forget_driver_jobs,
    get_num_mpi,
    load_queue_data,
)
from jqmc_workflow.launcher import Launcher
from jqmc_workflow.machine import close_all_machines
from jqmc_workflow.mcmc_workflow import MCMC_Workflow
from jqmc_workflow.state import add_job, create_state, get_job
from jqmc_workflow.workflow import Container, Workflow, WorkflowStatus

pytestmark = [pytest.mark.with_chdir]


class TestLoadQueueData:
    @pytest.mark.light
    def test_loads_section(self, local_machine_with_queue):
        result = load_queue_data("local-test", "default")
        assert result["num_cores"] == 1
        assert result["submit_template"] == "submit_template.sh"

    @pytest.mark.light
    def test_unknown_label_raises(self, local_machine_with_queue):
        with pytest.raises(KeyError, match="queue_label"):
            load_queue_data("local-test", "nonexistent")

    @pytest.mark.light
    def test_missing_file_raises(self, local_machine_yaml, tmp_jqmc_setting_dir):
        """machine entry exists but queue_data.toml does not."""
        # Make a bare machine dir without queue_data.toml.
        (tmp_jqmc_setting_dir / "local-test").mkdir(exist_ok=True)
        with pytest.raises(FileNotFoundError):
            load_queue_data("local-test", "default")


class TestGetNumMpi:
    @pytest.mark.light
    def test_num_cores_takes_priority(self):
        assert get_num_mpi({"num_cores": 16, "mpi_per_node": 1, "nodes": 2}) == 16

    @pytest.mark.light
    def test_mpi_per_node_x_nodes_when_no_num_cores(self):
        assert get_num_mpi({"mpi_per_node": 4, "nodes": 8}) == 32

    @pytest.mark.light
    def test_defaults_to_one(self):
        assert get_num_mpi({}) == 1

    @pytest.mark.light
    def test_partial_keys_use_defaults(self):
        assert get_num_mpi({"mpi_per_node": 4}) == 4  # nodes defaults to 1


class TestJobSubmissionConstruction:
    @pytest.mark.light
    def test_basic_construction(self, local_machine_with_queue):
        js = JobSubmission(server_machine_name="local-test")
        assert js.server_machine.name == "local-test"
        assert js.queue_label == "default"
        assert js.input_file == "input.toml"
        assert js.output_file == "out.o"
        assert js.jobname == "jqmc-wf"
        assert js.command is None
        assert js.max_job_submit == 100

    @pytest.mark.light
    def test_overrides_applied(self, local_machine_with_queue):
        js = JobSubmission(
            server_machine_name="local-test",
            input_file="custom.toml",
            output_file="custom.o",
            jobname="myjob",
            run_id="abc",
            command="mpirun -np 2 jqmc",
        )
        assert js.input_file == "custom.toml"
        assert js.output_file == "custom.o"
        assert js.jobname == "myjob"
        assert js.run_id == "abc"
        assert js.command == "mpirun -np 2 jqmc"
        # stdout/stderr filenames include run_id when set
        assert js.job_stdout == "job_myjob_abc.o"
        assert js.job_stderr == "job_myjob_abc.e"

    @pytest.mark.light
    def test_unknown_queue_label_raises(self, local_machine_with_queue):
        with pytest.raises(KeyError, match="queue_label"):
            JobSubmission(server_machine_name="local-test", queue_label="bogus")

    @pytest.mark.light
    def test_missing_queue_file_raises(self, local_machine_yaml, tmp_jqmc_setting_dir):
        (tmp_jqmc_setting_dir / "local-test").mkdir(exist_ok=True)  # no queue_data.toml
        with pytest.raises(FileNotFoundError):
            JobSubmission(server_machine_name="local-test")


class TestGenerateScript:
    @pytest.mark.light
    def test_substitutes_input_output_and_jobname(self, local_machine_with_queue, tmp_path):
        js = JobSubmission(
            server_machine_name="local-test",
            input_file="i.toml",
            output_file="out.txt",
            jobname="test-job",
            command="/bin/echo hello",
        )
        js.generate_script(submission_script="submit.sh", work_dir=str(tmp_path))
        text = (tmp_path / "submit.sh").read_text()
        assert "/bin/echo hello" in text  # _COMMAND_ substituted
        assert "out.txt" in text  # _OUTPUT_
        assert "test-job" in text  # _JOBNAME_

    @pytest.mark.light
    def test_missing_command_with_command_template_raises(self, local_machine_with_queue, tmp_path):
        """Template uses _COMMAND_ but JobSubmission was given no command."""
        js = JobSubmission(server_machine_name="local-test")  # no command
        with pytest.raises(ValueError, match="_COMMAND_"):
            js.generate_script(submission_script="x.sh", work_dir=str(tmp_path))

    @pytest.mark.light
    def test_queue_specific_vars_substituted(self, local_machine_with_queue, tmp_path, tmp_jqmc_setting_dir):
        """_QUEUE_ and other queue_data keys are substituted into the template."""
        template_path = tmp_jqmc_setting_dir / "local-test" / "submit_template.sh"
        template_path.write_text("queue=_QUEUE_\nnum_cores=_NUM_CORES_\n_COMMAND_\n")
        js = JobSubmission(server_machine_name="local-test", command="jqmc")
        js.generate_script(submission_script="s.sh", work_dir=str(tmp_path))
        text = (tmp_path / "s.sh").read_text()
        assert "queue=default" in text
        assert "num_cores=1" in text


class TestJobSubmitLocal:
    @pytest.mark.light
    def test_submit_with_queuing_false_does_not_get_job_number(self, local_machine_with_queue, tmp_path):
        """queuing=false: submit just runs the script; no job_number captured."""
        js = JobSubmission(server_machine_name="local-test", command="/usr/bin/true")
        js.generate_script(submission_script="submit.sh", work_dir=str(tmp_path))
        ok, jobnum = js.job_submit(submission_script="submit.sh", work_dir=str(tmp_path))
        assert ok is True
        assert jobnum is None
        assert js.job_running is False
        assert js.job_dir == str(tmp_path)
        assert js.job_submit_date is not None

    @pytest.mark.light
    def test_submit_failure_clears_state(self, local_machine_with_queue, tmp_path, tmp_jqmc_setting_dir):
        """A failing jobsubmit propagates as RuntimeError; state is cleared."""
        # Force jobsubmit to a non-existent command so subprocess returns non-zero.
        yaml_path = tmp_jqmc_setting_dir / "machine_data.yaml"
        text = yaml_path.read_text().replace("jobsubmit: /bin/echo", "jobsubmit: /nonexistent/jobsub")
        yaml_path.write_text(text)
        close_all_machines()
        js = JobSubmission(server_machine_name="local-test", command="/usr/bin/true")
        # Compress retry so the test does not sleep 120s.
        js.server_machine.ssh_retry_time = 0
        js.generate_script(submission_script="submit.sh", work_dir=str(tmp_path))
        with pytest.raises(RuntimeError):
            js.job_submit(submission_script="submit.sh", work_dir=str(tmp_path))
        assert js.job_running is False
        assert js.job_number is None


class TestJobCheck:
    @pytest.mark.light
    def test_returns_false_when_queuing_off(self, local_machine_with_queue):
        """queuing=false: jobcheck short-circuits to False."""
        js = JobSubmission(server_machine_name="local-test")
        assert js.jobcheck() is False
        assert js.job_check_last_time is not None

    @pytest.mark.light
    @pytest.mark.parametrize("queue_after_end", ["other-jobs", "empty"])
    def test_empty_listing_does_not_end_a_running_job(self, queued_local_machine, tmp_path, monkeypatch, queue_after_end):
        """A transiently empty qstat listing (exit 0) does not end a running job.

        The scheduler lists job 1000930 while it runs, except for one poll
        whose listing is empty.  The job writes its output and leaves the
        queue only later; the queue then shows another driver's job, or
        nothing at all (what PBS ``qstat`` prints for an empty queue).  The
        workflow must record the job as fetched only after the job has
        ended, so the output written at the end is there; the job's own
        output is the reference.
        """
        qstat_file = queued_local_machine
        job_line = _GROUP_QSTAT_LINE
        project = tmp_path / "B"
        output = project / "out.o"

        real_sleep = asyncio.sleep
        ticks = []

        # Each poll_interval wait is one scheduler tick.  Listing per tick:
        # 1 -> empty (transient), 2 -> job listed, 3 -> job ends.
        async def scheduler_tick(delay):
            ticks.append(delay)
            if len(ticks) > 50:
                raise RuntimeError("the job was polled 50 times after it ended")
            if len(ticks) == 1:
                qstat_file.write_text("")
            elif len(ticks) == 2:
                qstat_file.write_text(job_line + "\n")
            elif len(ticks) == 3:
                output.write_text("Program ends\n")
                qstat_file.write_text(_OTHER_DRIVER_QSTAT_LINE + "\n" if queue_after_end == "other-jobs" else "")
            await real_sleep(0)

        monkeypatch.setattr(workflow_module.asyncio, "sleep", scheduler_tick)
        # The fake qsub only prints the ID; the job is already queued.
        qstat_file.write_text(job_line + "\n")

        container = Container(label="B", dirname="B", workflow=_OneJobWorkflow())
        Launcher(workflows=[container]).launch()

        assert get_job(str(project), "input.toml")["status"] == "fetched"
        assert output.read_text() == "Program ends\n"

    @pytest.mark.light
    def test_submit_output_without_job_id_is_rejected(self, queued_local_machine, tmp_jqmc_setting_dir, tmp_path):
        """A jobnum_index word that is not a job ID fails the submission.

        ``qsub`` prints a warning line before the ID, so word 0 is ``qsub:``.
        No listing ever shows that word, so accepting it would report the
        queued job done at the first poll.  The submission must raise with
        the raw output, which names the queued job.
        """
        qstat_file = queued_local_machine
        qsub = tmp_path / "warn_qsub.sh"
        qsub.write_text(
            f"echo {shlex.quote(_GROUP_QSTAT_LINE)} >> {shlex.quote(str(qstat_file))}\n"
            "echo 'qsub: warning: using default queue'\n"
            "echo 1000930.pbs\n"
        )
        yaml_path = tmp_jqmc_setting_dir / "machine_data.yaml"
        yaml_path.write_text(yaml_path.read_text().replace("jobsubmit: /bin/echo 1000930.pbs", f"jobsubmit: sh {qsub}", 1))
        close_all_machines()

        js = JobSubmission(server_machine_name="local-test", command="/usr/bin/true")
        js.generate_script(submission_script="submit.sh", work_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match=r"(?s)'qsub:'.*1000930\.pbs"):
            js.job_submit(submission_script="submit.sh", work_dir=str(tmp_path))
        assert js.job_number is None


# A PBS ``qstat`` line from a site whose listing prints the GROUP (grp01) in
# the fourth column, not the user (user01), and whose qstat has no ``-u`` option.
_GROUP_QSTAT_LINE = (
    "1000930           mcmc-fp64- QUEUED    grp01      debug           (11/07 16:48:00) --:--:--      (0.5)    1   -"
)
_OTHER_DRIVER_QSTAT_LINE = (
    "1000931           lrdmc-fp64 RUNNING   grp01      debug           (11/07 16:49:00) 00:01:00      (0.5)    1   -"
)


@pytest.fixture
def queued_local_machine(local_machine_with_queue, tmp_jqmc_setting_dir, tmp_path):
    """local-test with queuing=true, a fake qsub and a file-backed qstat.

    ``jobsubmit`` prints ``1000930.pbs`` (PBS style, the listing shows only
    the numeric part); ``jobcheck`` prints the returned file, so a test edits
    the "queue" by rewriting it.  The queue entry is ``max_job_submit = 1`` on
    queue ``debug``.
    """
    qstat_file = tmp_path / "qstat.txt"
    qstat_file.write_text("")
    yaml_path = tmp_jqmc_setting_dir / "machine_data.yaml"
    text = yaml_path.read_text()
    local_block, rest = text.split("localhost:", 1)
    local_block = (
        local_block.replace("queuing: false", "queuing: true")
        .replace("jobsubmit: /bin/echo", "jobsubmit: /bin/echo 1000930.pbs")
        .replace("jobcheck: /usr/bin/true", f"jobcheck: cat {qstat_file}")
        .replace("jobnum_index: 1", "jobnum_index: 0")
    )
    yaml_path.write_text(local_block + "localhost:" + rest)
    queue_path = tmp_jqmc_setting_dir / "local-test" / "queue_data.toml"
    queue_path.write_text(
        textwrap.dedent(
            """
            [default]
            submit_template = "submit_template.sh"
            num_cores = 1
            max_job_submit = 1
            queue = "debug"
            """
        ).strip()
        + "\n"
    )
    close_all_machines()
    forget_driver_jobs()
    yield qstat_file
    forget_driver_jobs()


class TestJobNumCheck:
    @pytest.mark.light
    def test_returns_true_when_queuing_off(self, local_machine_with_queue):
        """queuing=false: jobnum_check always True (no queue limit)."""
        js = JobSubmission(server_machine_name="local-test")
        assert js.jobnum_check() is True

    @pytest.mark.light
    def test_driver_scope_counts_own_jobs_in_group_only_qstat(self, queued_local_machine, tmp_path):
        """max_job_submit holds when qstat prints the group, not the user."""
        qstat_file = queued_local_machine
        js = JobSubmission(server_machine_name="local-test", command="/usr/bin/true")
        assert js.max_job_submit_scope == "driver"
        js.generate_script(submission_script="submit.sh", work_dir=str(tmp_path))

        # Another driver's job on the same account does not take our slot.
        qstat_file.write_text(_OTHER_DRIVER_QSTAT_LINE + "\n")
        assert js.job_submit(submission_script="submit.sh", work_dir=str(tmp_path)) == (True, "1000930.pbs")

        # Our job is listed (numeric part only): the limit of 1 is reached.
        qstat_file.write_text(_GROUP_QSTAT_LINE + "\n" + _OTHER_DRIVER_QSTAT_LINE + "\n")
        second = JobSubmission(server_machine_name="local-test", command="/usr/bin/true")
        assert second.jobnum_check() is False
        assert second.job_submit(submission_script="submit.sh", work_dir=str(tmp_path)) == (False, None)

        # The legacy username regex cannot see the job in this listing.
        second.max_job_submit_scope = "user"
        second.server_machine.username = "user01"
        assert second.jobnum_check() is True
        second.max_job_submit_scope = "driver"

        # Once our job leaves the queue the slot frees.
        qstat_file.write_text(_OTHER_DRIVER_QSTAT_LINE + "\n")
        assert second.jobnum_check() is True

    @pytest.mark.light
    def test_driver_scope_counts_resumed_job(self, queued_local_machine):
        """A job resumed from workflow_state.toml is registered by jobcheck."""
        qstat_file = queued_local_machine
        qstat_file.write_text(_GROUP_QSTAT_LINE + "\n")
        resumed = JobSubmission(server_machine_name="local-test")
        resumed.job_number = "1000930.pbs"
        assert resumed.jobcheck() is True
        assert JobSubmission(server_machine_name="local-test").jobnum_check() is False

        qstat_file.write_text(_OTHER_DRIVER_QSTAT_LINE + "\n")
        assert resumed.jobcheck() is False
        assert JobSubmission(server_machine_name="local-test").jobnum_check() is True

    @pytest.mark.light
    def test_user_scope_keeps_username_queue_match(self, queued_local_machine, tmp_jqmc_setting_dir):
        """max_job_submit_scope = "user" counts the account's lines (legacy behavior)."""
        qstat_file = queued_local_machine
        queue_path = tmp_jqmc_setting_dir / "local-test" / "queue_data.toml"
        queue_path.write_text(queue_path.read_text() + 'max_job_submit_scope = "user"\n')
        js = JobSubmission(server_machine_name="local-test")
        js.server_machine.username = "alice"
        qstat_file.write_text("17.pbs   job1   alice   00:00:01 R debug \n")
        assert js.jobnum_check() is False
        qstat_file.write_text("17.pbs   job1   bob     00:00:01 R debug \n")
        assert js.jobnum_check() is True

    @pytest.mark.light
    def test_unknown_scope_raises(self, queued_local_machine, tmp_jqmc_setting_dir):
        """An unrecognized max_job_submit_scope is rejected at construction."""
        queue_path = tmp_jqmc_setting_dir / "local-test" / "queue_data.toml"
        queue_path.write_text(queue_path.read_text() + 'max_job_submit_scope = "group"\n')
        with pytest.raises(ValueError, match="max_job_submit_scope"):
            JobSubmission(server_machine_name="local-test")


class _OneJobWorkflow(Workflow):
    """Submits one job (or resumes the saved one) on queued local-test."""

    def __init__(self, start_after: asyncio.Event | None = None):
        super().__init__()
        self.server_machine_name = "local-test"
        self.queue_label = "default"
        self.jobname = "jqmc-wf"
        self.hamiltonian_file = "hamiltonian_data.h5"
        self.poll_interval = 7
        self.start_after = start_after

    def _get_command(self, input_file):
        return "/usr/bin/true"

    async def run(self):
        if self.start_after is not None:
            await self.start_after.wait()
        await self._submit_and_wait("input.toml", "out.o", work_dir=self.project_dir, step=1, run_id="r1")
        return WorkflowStatus.COMPLETED, [], {}


class TestDriverRestart:
    @pytest.mark.light
    def test_saved_job_holds_slot_before_its_workflow_polls(
        self, queued_local_machine, tmp_jqmc_setting_dir, tmp_path, monkeypatch
    ):
        """A restarted driver does not submit past max_job_submit while its saved job is queued.

        Workflow A was interrupted with job 1000929 still queued; workflow B
        has not submitted yet.  B tries to submit before A polls its saved
        job.  With max_job_submit = 1, B's job must wait until 1000929 has
        left the queue.
        """
        qstat_file = queued_local_machine
        saved_line = _GROUP_QSTAT_LINE.replace("1000930", "1000929")
        fresh_line = _GROUP_QSTAT_LINE
        qstat_file.write_text(saved_line + "\n" + _OTHER_DRIVER_QSTAT_LINE + "\n")

        # Fake qsub: log the queue as seen at submission, then queue job 1000930.
        submit_log = tmp_path / "submit_log.txt"
        qsub = tmp_path / "fake_qsub.sh"
        qsub.write_text(
            f"{{ echo '== submit'; cat {shlex.quote(str(qstat_file))}; }} >> {shlex.quote(str(submit_log))}\n"
            f"echo {shlex.quote(fresh_line)} >> {shlex.quote(str(qstat_file))}\n"
            "echo 1000930.pbs\n"
        )
        yaml_path = tmp_jqmc_setting_dir / "machine_data.yaml"
        yaml_path.write_text(yaml_path.read_text().replace("jobsubmit: /bin/echo 1000930.pbs", f"jobsubmit: sh {qsub}", 1))
        close_all_machines()

        # Each poll_interval wait is one scheduler tick: the saved job ends at
        # tick 2 and any job submitted after it ends at tick 3 or later.  A
        # job that ends has written its output into its project directory.
        real_sleep = asyncio.sleep
        first_wait = asyncio.Event()
        ticks = []
        output_of = {saved_line: tmp_path / "A" / "out.o", fresh_line: tmp_path / "B" / "out.o"}

        async def scheduler_tick(delay):
            ticks.append(delay)
            first_wait.set()
            ended = [saved_line] if len(ticks) >= 2 else []
            ended += [fresh_line] if len(ticks) >= 3 else []
            listed = qstat_file.read_text().splitlines()
            for line in listed:
                if line in ended:
                    output_of[line].write_text("Program ends\n")
            kept = [line for line in listed if line not in ended]
            qstat_file.write_text("".join(line + "\n" for line in kept))
            await real_sleep(0)

        monkeypatch.setattr(workflow_module.asyncio, "sleep", scheduler_tick)

        project_a = tmp_path / "A"
        project_a.mkdir()
        # Saved the way a driver without the queue_label field wrote it.
        create_state(str(project_a), label="A", workflow_type="_OneJobWorkflow", status="running")
        add_job(
            str(project_a),
            input_file="input.toml",
            output_file="out.o",
            job_id="1000929.pbs",
            server_machine="local-test",
            step=1,
            run_id="r1",
        )
        # A polls its saved job only after B has tried to submit.
        container_a = Container(label="A", dirname="A", workflow=_OneJobWorkflow(start_after=first_wait))
        container_b = Container(label="B", dirname="B", workflow=_OneJobWorkflow())
        Launcher(workflows=[container_a, container_b]).launch()

        submissions = submit_log.read_text().split("== submit\n")[1:]
        assert len(submissions) == 1
        assert "1000929" not in submissions[0]
        assert get_job(str(project_a), "input.toml")["job_id"] == "1000929.pbs"
        assert get_job(str(tmp_path / "B"), "input.toml")["job_id"] == "1000930.pbs"
        for project in (project_a, tmp_path / "B"):
            assert get_job(str(project), "input.toml")["status"] == "fetched"

    @pytest.mark.light
    def test_saved_pilot_job_holds_pilot_queue_slot(self, queued_local_machine, tmp_jqmc_setting_dir, tmp_path):
        """A saved pilot job without queue_label counts against the pilot queue.

        MCMC_Workflow submits its pilot in ``_pilot/`` with
        ``pilot_queue_label``.  The production queue is ``production-queue``,
        the pilot queue is ``debug`` with ``max_job_submit = 1``, and the
        interrupted driver's pilot 1000929 is still queued there.  Before any
        workflow polls it, a new pilot submission must be refused, while the
        production queue stays free.
        """
        qstat_file = queued_local_machine
        queue_path = tmp_jqmc_setting_dir / "local-test" / "queue_data.toml"
        queue_path.write_text(
            textwrap.dedent(
                """
                [default]
                submit_template = "submit_template.sh"
                num_cores = 1
                max_job_submit = 1
                queue = "production-queue"

                [pilot]
                submit_template = "submit_template.sh"
                num_cores = 1
                max_job_submit = 1
                queue = "debug"
                """
            ).strip()
            + "\n"
        )
        saved_line = _GROUP_QSTAT_LINE.replace("1000930", "1000929")
        qstat_file.write_text(saved_line + "\n" + _OTHER_DRIVER_QSTAT_LINE + "\n")

        wf = MCMC_Workflow(server_machine_name="local-test", queue_label="default", pilot_queue_label="pilot")
        container = Container(label="mcmc", dirname="mcmc", workflow=wf)
        pilot_dir = Path(container.project_dir) / "_pilot"
        pilot_dir.mkdir(parents=True)
        create_state(str(container.project_dir), label="mcmc", workflow_type="MCMC_Workflow", status="running")
        # Saved the way a driver without the queue_label field wrote it.
        create_state(str(pilot_dir), label="mcmc-pilot", workflow_type="MCMC_Workflow", status="running")
        add_job(
            str(pilot_dir),
            input_file="input_0.toml",
            output_file="out_0.o",
            job_id="1000929.pbs",
            server_machine="local-test",
            step=0,
            run_id="r0",
        )
        Launcher(workflows=[container])._restore_submitted_jobs()

        assert JobSubmission(server_machine_name="local-test", queue_label="pilot").jobnum_check() is False
        assert JobSubmission(server_machine_name="local-test", queue_label="default").jobnum_check() is True

        # Once the saved pilot leaves the queue the pilot slot frees.
        qstat_file.write_text(_OTHER_DRIVER_QSTAT_LINE + "\n")
        assert JobSubmission(server_machine_name="local-test", queue_label="pilot").jobnum_check() is True


class TestDeleteJob:
    @pytest.mark.light
    def test_runs_jobdel_command(self, local_machine_with_queue, tmp_jqmc_setting_dir):
        """delete_job runs the jobdel binary with the job_number; flips running flag off."""
        # Use /bin/echo as jobdel so we can verify (and have deterministic output).
        yaml_path = tmp_jqmc_setting_dir / "machine_data.yaml"
        text = yaml_path.read_text().replace("jobdel: /usr/bin/true", "jobdel: /bin/echo")
        yaml_path.write_text(text)
        close_all_machines()
        js = JobSubmission(server_machine_name="local-test")
        js.job_number = "12345"
        js.job_running = True
        js.delete_job()
        assert js.job_running is False


class TestJobAcct:
    @pytest.mark.light
    def test_returns_none_when_queuing_off(self, local_machine_with_queue):
        """queuing=false: job_acct is a no-op."""
        js = JobSubmission(server_machine_name="local-test")
        js.job_number = "12345"
        assert js.job_acct() is None

    @pytest.mark.light
    def test_returns_none_when_no_job_number(self, local_machine_with_queue, tmp_jqmc_setting_dir):
        """Even with queuing=true + jobacct configured, no job_number -> None."""
        yaml_path = tmp_jqmc_setting_dir / "machine_data.yaml"
        text = yaml_path.read_text().replace("queuing: false", "queuing: true", 1)  # flip local-test only (first occurrence)
        # Add jobacct field
        text = text.replace(
            "jobnum_index: 1\nlocalhost",
            "jobnum_index: 1\n  jobacct: /bin/echo\nlocalhost",
            1,
        )
        yaml_path.write_text(text)
        close_all_machines()
        js = JobSubmission(server_machine_name="local-test")
        js.job_number = None
        assert js.job_acct() is None
