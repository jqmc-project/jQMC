"""A queued local job that left the queue without its output is a failure, not a fetched run.

The workflow runs on the ``local-test`` machine with ``queuing: true``. The
scheduler is the same kind of stand-in the other jqmc_workflow tests use: a
fake ``jobsubmit`` script and a file-backed ``jobcheck`` listing. The fake
``jobsubmit`` either runs the submit script (the job writes its output and
leaves the queue) or accepts it and drops it without running it (the job
leaves the queue and writes nothing). The listing always shows an unrelated
job of another driver, so it is never empty.

The independent reference is the job's output file on disk: a run is complete
only if its output exists and ends with ``Program ends``. A fixed-step
production loop of three runs must either finish with three such outputs or
fail naming the output of the run that never ran; it must never record that
run as ``fetched`` or continue from it.
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

import shlex
import textwrap

import pytest

from jqmc_workflow._job import forget_driver_jobs
from jqmc_workflow.launcher import Launcher
from jqmc_workflow.machine import close_all_machines
from jqmc_workflow.state import add_job, create_state, read_state, update_job
from jqmc_workflow.workflow import Container, Workflow, WorkflowStatus

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]

_NUM_RUNS = 3
_DROPPED_STEP = 2
# A job of another driver: the listing is never empty and never shows ours.
_OTHER_DRIVER_QSTAT_LINE = "1000999           lrdmc-fp64 RUNNING   grp01      debug           (11/07 16:49:00) 00:01:00"
# Job id of run 2 when it was submitted by an earlier driver process.
_EARLIER_JOB_ID = "1000929.pbs"


class _FixedRunsWorkflow(Workflow):
    """A fixed-step production loop: every continuation run is one queued job."""

    def __init__(self):
        super().__init__()
        self.server_machine_name = "local-test"
        self.queue_label = "default"
        self.jobname = "jqmc-fixed"
        self.hamiltonian_file = "hamiltonian_data.h5"
        self.poll_interval = 7

    def _get_command(self, input_file):
        return "echo 'Program ends'"

    async def run(self):
        outputs = []
        for step in range(1, _NUM_RUNS + 1):
            output = f"out_{step}.o"
            await self._submit_and_wait(f"input_{step}.toml", output, work_dir=self.project_dir, step=step, run_id=f"r{step}")
            outputs.append(output)
        return WorkflowStatus.COMPLETED, outputs, {}


@pytest.fixture
def queued_local_machine(local_machine_with_queue, tmp_jqmc_setting_dir, tmp_path):
    """local-test with queuing=true; returns a function that installs the fake qsub.

    The fake qsub runs in the job directory, prints ``100093<N>.pbs`` for
    the submit script of run ``r<N>`` and runs that script unless the run
    is the dropped one.
    """
    qstat_file = tmp_path / "qstat.txt"
    qstat_file.write_text(_OTHER_DRIVER_QSTAT_LINE + "\n")
    yaml_path = tmp_jqmc_setting_dir / "machine_data.yaml"
    local_block, rest = yaml_path.read_text().split("localhost:", 1)

    def install(drop_run_id):
        qsub = tmp_path / "fake_qsub.sh"
        run_unless_dropped = (
            f'case "$1" in *_{drop_run_id}.sh) ;; *) sh "$1" >/dev/null 2>&1 ;; esac\n'
            if drop_run_id
            else 'sh "$1" >/dev/null 2>&1\n'
        )
        qsub.write_text(run_unless_dropped + 'echo "100093$(echo "$1" | tr -dc 0-9).pbs"\n')
        block = (
            local_block.replace("queuing: false", "queuing: true")
            .replace("jobsubmit: /bin/echo", f"jobsubmit: sh {shlex.quote(str(qsub))}")
            .replace("jobcheck: /usr/bin/true", f"jobcheck: cat {shlex.quote(str(qstat_file))}")
            .replace("jobnum_index: 1", "jobnum_index: 0")
        )
        yaml_path.write_text(block + "localhost:" + rest)
        close_all_machines()

    queue_path = tmp_jqmc_setting_dir / "local-test" / "queue_data.toml"
    queue_path.write_text(
        textwrap.dedent(
            """
            [default]
            submit_template = "submit_template.sh"
            num_cores = 1
            max_job_submit = 4
            queue = "debug"
            """
        ).strip()
        + "\n"
    )
    forget_driver_jobs()
    yield install
    forget_driver_jobs()
    close_all_machines()


def _has_program_ends(path):
    return path.is_file() and path.read_text().rstrip().endswith("Program ends")


@pytest.mark.parametrize("dropped", [False, True], ids=["job-ran", "job-dropped"])
@pytest.mark.parametrize(
    "route",
    ["new-submission", "resumed-submitted", "completed-not-fetched"],
)
def test_job_that_left_queue_without_output_fails_the_stage(queued_local_machine, tmp_path, route, dropped):
    """Run 2 leaves the queue without its output: FAILED naming it, never fetched/COMPLETED."""
    container = Container(label="mcmc", dirname="mc", workflow=_FixedRunsWorkflow())
    project = tmp_path / "mc"
    if route == "new-submission":
        # The scheduler accepts run 2 and drops it before it starts.
        queued_local_machine("r2" if dropped else None)
    else:
        # An earlier driver ran run 1 and submitted run 2, then was stopped.
        # Run 2 has left the queue since; it ran only if its output exists.
        queued_local_machine(None)
        project.mkdir()
        create_state(str(project), label="mcmc", workflow_type="_FixedRunsWorkflow", status="running")
        (project / "out_1.o").write_text("Program ends\n")
        add_job(str(project), "input_1.toml", "out_1.o", "1000928.pbs", "local-test", status="fetched", step=1, run_id="r1")
        add_job(str(project), "input_2.toml", "out_2.o", _EARLIER_JOB_ID, "local-test", step=2, run_id="r2")
        if route == "completed-not-fetched":
            update_job(str(project), "input_2.toml", status="completed")
        if not dropped:
            (project / "out_2.o").write_text("Program ends\n")

    Launcher(workflows=[container]).launch()

    state = read_state(str(project))
    jobs = {job["step"]: job for job in state["jobs"]}
    # Whatever the outcome, a run recorded as fetched has its complete output.
    for job in jobs.values():
        if job["status"] == "fetched":
            assert _has_program_ends(project / job["output_file"]), job

    if not dropped:
        assert container.status == WorkflowStatus.COMPLETED
        assert state["workflow"]["status"] == "completed"
        assert sorted(jobs) == list(range(1, _NUM_RUNS + 1))
        assert all(job["status"] == "fetched" for job in jobs.values())
        return

    dropped_job_id = "1000932.pbs" if route == "new-submission" else _EARLIER_JOB_ID
    assert not (project / f"out_{_DROPPED_STEP}.o").exists()
    assert state["workflow"]["status"] == "failed"
    message = state["error"]["message"]
    assert f"out_{_DROPPED_STEP}.o" in message
    assert dropped_job_id in message
    assert jobs[_DROPPED_STEP]["status"] != "fetched"
    # No continuation was run on top of the run that never ran.
    assert sorted(jobs) == [1, _DROPPED_STEP]
