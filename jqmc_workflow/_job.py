"""Job submission, monitoring, and fetching for jqmc-workflow.

jqmc is the only target binary.  The binary name ("jqmc") is hard-coded;
there is no binary_path, preoption, postoption, or input_redirect.
Job state is managed exclusively via workflow_state.toml.
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

import fnmatch
import os
import re
import shlex
import shutil
import threading
from datetime import datetime
from logging import getLogger

import toml

from ._config import get_config_dir, template_dir
from .machine import Machine, get_machine
from ._transfer import Data_transfer

logger = getLogger("jqmc-workflow").getChild(__name__)

#: Accepted values of the optional ``max_job_submit_scope`` queue key.
#: ``"driver"`` (default) counts the jobs this process submitted and that
#: are still listed by ``jobcheck``; ``"user"`` counts every line of the
#: ``jobcheck`` output that shows both the login name and the queue name.
MAX_JOB_SUBMIT_SCOPES = ("driver", "user")

# Job IDs submitted (or resumed) by this process and not yet seen to
# leave the queue, keyed by ``(machine name, scheduler queue)``.  All
# workflows of one driver share the registry, so ``max_job_submit`` caps
# the driver as a whole.  Other drivers on the same account have their
# own registry and therefore their own cap.
_DRIVER_JOBS: dict[tuple[str, str], set[str]] = {}
_DRIVER_JOBS_LOCK = threading.RLock()


def _job_listed(job_id: str, tokens: set[str]) -> bool:
    """Return True if *job_id* appears as a whitespace token in a job listing.

    PBS may return ``"1000001.pbs"`` from ``qsub`` but show only
    ``"1000001"`` in ``qstat`` (or vice-versa), so both the full ID and its
    numeric prefix (part before the first dot) are compared.  Exact token
    matching avoids substring collisions (e.g. PID 12 vs 12345).
    """
    job_id_full = str(job_id)
    job_id_short = job_id_full.split(".")[0]
    return job_id_full in tokens or job_id_short in tokens


def _driver_jobs_key(machine_name: str, queue_data: dict, queue_label: str) -> tuple[str, str]:
    """Registry key: machine name and scheduler queue (label if no ``queue`` key)."""
    return (machine_name, str(queue_data.get("queue") or queue_label))


def restore_driver_jobs(saved_jobs) -> int:
    """Register jobs saved as ``submitted`` that the scheduler still lists.

    A restarted driver calls this before it admits any new submission,
    so the jobs it submitted before the restart hold their
    ``max_job_submit`` slots even before their workflows resume polling
    them.  The workflows later poll the same IDs with
    :meth:`JobSubmission.jobcheck`, which releases each slot when its job
    is done.

    Args:
        saved_jobs (iterable[tuple[str, str, str]]):
            ``(server_machine_name, queue_label, job_id)`` per saved job.

    Returns:
        int:
            Number of jobs registered.  A job the scheduler no longer
            lists is not registered, and neither is one whose machine or
            queue label is no longer configured; its workflow handles it
            when it resumes.
    """
    restored = 0
    tokens_by_machine: dict[str, set[str]] = {}
    with _DRIVER_JOBS_LOCK:
        for machine_name, queue_label, job_id in saved_jobs:
            try:
                queue_data = load_queue_data(machine_name, queue_label)
                machine = get_machine(machine_name)
            except (KeyError, FileNotFoundError) as e:
                logger.warning(f"Cannot restore job {job_id} on {machine_name} (queue_label={queue_label!r}): {e}")
                continue
            if not machine.queuing:
                continue
            if machine_name not in tokens_by_machine:
                job_list = machine.get_job_list_as_text()
                tokens_by_machine[machine_name] = {t for line in job_list for t in line.split()}
            if _job_listed(job_id, tokens_by_machine[machine_name]):
                _DRIVER_JOBS.setdefault(_driver_jobs_key(machine.name, queue_data, queue_label), set()).add(str(job_id))
                restored += 1
    return restored


def forget_driver_jobs() -> None:
    """Clear the process-wide registry of submitted job IDs.

    The registry only holds job IDs; clearing it never touches the
    scheduler.  Intended for tests and for long-lived processes that run
    several independent drivers one after another.
    """
    with _DRIVER_JOBS_LOCK:
        _DRIVER_JOBS.clear()


def load_queue_data(server_machine_name: str, queue_label: str) -> dict:
    """Load a single queue section from ``queue_data.toml``.

    Args:
        server_machine_name (str):
            Machine name (directory under ``~/.jqmc_setting/``).
        queue_label (str):
            Section key in ``queue_data.toml``.

    Returns:
        dict:
            The TOML table for *queue_label*.
    """
    machine = Machine(server_machine_name)
    cfg = get_config_dir()
    path = os.path.join(cfg, machine.name, "queue_data.toml")
    with open(path) as f:
        data = toml.load(f)
    if queue_label not in data:
        raise KeyError(f"queue_label='{queue_label}' not found in {path}.")
    return data[queue_label]


def get_num_mpi(queue_data: dict) -> int:
    """Extract the number of MPI processes from a queue configuration.

    Tries ``num_cores`` first, then ``mpi_per_node x nodes``.
    Defaults to 1 if neither key is present.
    """
    if "num_cores" in queue_data:
        return int(queue_data["num_cores"])
    mpi_per_node = queue_data.get("mpi_per_node", 1)
    nodes = queue_data.get("nodes", 1)
    return int(mpi_per_node) * int(nodes)


class JobSubmission:
    """Submit, monitor, and fetch a single jqmc job on a remote (or local) machine."""

    #: Consecutive empty ``jobcheck`` listings after which a job counts as done.
    #: An empty listing (exit status 0, no output) is both what PBS ``qstat``
    #: or ``squeue --noheader`` print once the queue is empty and what a busy
    #: scheduler may print transiently while the job still runs; only a
    #: listing that stays empty over this many polls is taken as the former.
    empty_listing_polls = 3

    def __init__(
        self,
        server_machine_name: str,
        input_file: str = "input.toml",
        output_file: str = "out.o",
        queue_label: str = "default",
        jobname: str = "jqmc-wf",
        run_id: str = "",
        command: str | None = None,
    ):
        self.data_transfer = Data_transfer(
            server_machine_name=server_machine_name,
        )
        self.server_machine = self.data_transfer.server_machine

        # Bootstrap config directory if needed
        cfg = get_config_dir()
        if not os.path.isdir(cfg):
            logger.info(f"Bootstrapping config at {cfg}")
            shutil.copytree(template_dir, cfg)
            raise ValueError(f"Please configure {cfg} first.")

        # -- Queue settings ----------------------------------------
        self.queue_label = queue_label
        queue_data_path = os.path.join(
            cfg,
            self.server_machine.name,
            "queue_data.toml",
        )
        try:
            with open(queue_data_path) as f:
                dict_toml = toml.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"{queue_data_path} not found!")

        try:
            self.queue_data = dict_toml[queue_label]
        except KeyError:
            raise KeyError(f"queue_label='{queue_label}' not found in {queue_data_path}.")

        # -- Job template ------------------------------------------
        self.job_submission_template = self.queue_data["submit_template"]

        # -- Job parameters ----------------------------------------
        self.jobname = jobname
        self.run_id = run_id
        self.input_file = input_file
        self.output_file = output_file
        # Executable + args substituted into the template's ``_COMMAND_``
        # placeholder. Each Workflow subclass constructs this via its
        # ``_get_command`` hook so the script line looks like::
        #
        #     mpirun -np <N> <command> > <output> 2>&1
        #
        # ``None`` means the caller did not supply a command (covers
        # legacy local-only flows that don't go through the template).
        self.command = command

        # -- Job state ---------------------------------------------
        self.max_job_submit = self.queue_data.get("max_job_submit", 1000)
        self.max_job_submit_scope = self.queue_data.get("max_job_submit_scope", "driver")
        if self.max_job_submit_scope not in MAX_JOB_SUBMIT_SCOPES:
            raise ValueError(
                f"queue_label='{queue_label}' in {queue_data_path}: max_job_submit_scope="
                f"{self.max_job_submit_scope!r} is not one of {list(MAX_JOB_SUBMIT_SCOPES)}."
            )
        self.job_number = None
        self.job_running = False
        self.job_dir = None
        self.job_submit_date = None
        self.job_check_last_time = None
        self.empty_listings_in_a_row = 0
        self.job_fetch_date = None
        # -- Scheduler stdout/stderr file paths -------------------
        _id_suffix = f"_{run_id}" if run_id else ""
        self.job_stdout: str = f"job_{jobname}{_id_suffix}.o"
        self.job_stderr: str = f"job_{jobname}{_id_suffix}.e"

    # -- Script generation -----------------------------------------

    def generate_script(self, submission_script: str = "submit.sh", *, work_dir=None):
        """Generate job submission script from template + queue_data.toml vars.

        Args:
            submission_script (str):
                Filename of the generated script (basename).
            work_dir (str, optional):
                Directory where the script is written.  When *None*,
                falls back to the current working directory.
        """
        cfg = get_config_dir()
        template_path = os.path.join(
            cfg,
            self.server_machine.name,
            self.job_submission_template,
        )
        with open(template_path) as f:
            lines = f.readlines()

        def replace_kw(lines, keyword, value):
            """Replace _KEYWORD_ placeholders in script lines."""
            buf = [l for l in lines if re.match(f".*{keyword}.*", l)]
            for b in buf:
                idx = lines.index(b)
                lines[idx] = lines[idx].replace(keyword.replace("\\", ""), str(value))
            return lines

        # Standard replacements (jqmc-specific)
        lines = replace_kw(lines, "_INPUT_", self.input_file)
        lines = replace_kw(lines, "_OUTPUT_", self.output_file)
        lines = replace_kw(lines, "_JOBNAME_", self.jobname)
        lines = replace_kw(lines, "_JOB_STDOUT_", self.job_stdout)
        lines = replace_kw(lines, "_JOB_STDERR_", self.job_stderr)
        if self.command is None:
            # Templates that use ``_COMMAND_`` require a command from
            # the calling Workflow. Failing late (when the script runs)
            # produces a confusing PBS error; fail here instead.
            if any("_COMMAND_" in l for l in lines):
                raise ValueError(
                    f"Template {self.job_submission_template} uses _COMMAND_ but "
                    f"no command was supplied to JobSubmission. Set the "
                    f"workflow's _get_command(input_file) hook."
                )
        else:
            lines = replace_kw(lines, "_COMMAND_", self.command)

        # Queue-specific variables from queue_data.toml
        _SKIP_KEYS = {"submit_template", "max_job_submit", "max_job_submit_scope"}
        for key, value in self.queue_data.items():
            if key not in _SKIP_KEYS:
                lines = replace_kw(lines, f"_{key.upper()}_", value)

        script_path = os.path.join(work_dir, submission_script) if work_dir else submission_script
        with open(script_path, "w") as f:
            f.writelines(lines)

    # -- Job submission --------------------------------------------

    def job_submit(self, submission_script: str = "submit.sh", from_objects=None, *, work_dir=None):
        """Submit the job.

        Args:
            submission_script (str):
                Basename of the submit script in *work_dir*.
            from_objects (list[str], optional):
                Basenames of extra files to upload.
            work_dir (str, optional):
                Absolute path to the local job directory.  When *None*,
                falls back to ``os.getcwd()`` for backward compatibility.

        Returns:
            tuple[bool, str | None]:
                ``(True, job_number)`` after a submission.
                ``(False, None)`` when ``max_job_submit`` is reached and
                nothing was uploaded or submitted; the caller may retry
                later.  Any other failure raises.
        """
        from_objects = from_objects or []

        # The limit check and the registration of the new job ID happen
        # under one lock, so concurrent submissions from this process
        # cannot both take the last free slot.
        with _DRIVER_JOBS_LOCK:
            return self._job_submit_locked(submission_script, from_objects, work_dir)

    def _job_submit_locked(self, submission_script, from_objects, work_dir):
        try:
            if not self.jobnum_check():
                logger.info("Max job limit reached; not submitting.")
                self.job_submit_date = None
                self.job_number = None
                self.job_running = False
                return False, self.job_number

            local_cwd = os.path.abspath(work_dir) if work_dir else os.path.abspath(os.getcwd())

            # -- Submit via queuing system or remote submit script --
            command = f"{self.server_machine.jobsubmit} {shlex.quote(submission_script)}"

            if self.server_machine.machine_type == "local":
                server_dir = local_cwd
            else:
                local_root = self.data_transfer._local_root
                server_root = self.server_machine.workspace_root
                if local_root and local_root not in local_cwd:
                    raise ValueError(f"work_dir ({local_cwd}) not under local workspace_root ({local_root}).")
                server_dir = local_cwd.replace(local_root, server_root)
                self.data_transfer.put_objects(from_objects=from_objects, work_dir=local_cwd)

            if self.server_machine.queuing:
                stdout, stderr = self.server_machine.run_command(command=command, execute_dir=server_dir)
                if not stdout:
                    logger.error(f"Empty stdout! stderr={stderr}")
                job_number = stdout.strip().split()[self.server_machine.jobnum_index]
                # Every scheduler job ID (and a PID) contains a digit.  A word
                # without one (e.g. ``qsub:`` of a warning printed before the
                # ID) is never listed by ``jobcheck``, which would report the
                # job done at the first poll while it runs.
                if not any(ch.isdigit() for ch in job_number):
                    raise RuntimeError(
                        f"Cannot read the job ID from the output of '{command}': word "
                        f"{self.server_machine.jobnum_index} (jobnum_index) is {job_number!r}. "
                        f"The job may have been queued; find it with '{self.server_machine.jobcheck}' "
                        f"and check jobnum_index in machine_data.yaml.\nstdout: {stdout.strip()}"
                    )
                self.job_number = job_number
                self.job_running = True
                self._track_job()
            else:
                # Remote non-queuing: still use submit script
                self.server_machine.run_command(command=command, execute_dir=server_dir)
                self.job_number = None
                self.job_running = False

            self.job_dir = server_dir
            self.job_submit_date = datetime.today()
            logger.info(f"  Job submitted (number={self.job_number}).")

            return True, self.job_number

        except Exception as e:
            self.job_number = None
            self.job_running = False
            logger.error(f"Job submission failed: {e}")
            raise
        finally:
            self._close_ssh()

    # -- Job checking ----------------------------------------------

    def jobcheck(self) -> bool:
        """Return True if the job is still running.

        The job is done when a non-empty listing does not show its ID, or
        when the listing has been empty on ``empty_listing_polls``
        consecutive calls; an earlier empty listing returns True.

        The SSH connection is intentionally kept open so that the
        polling loop in ``_submit_and_wait`` reuses a single
        connection instead of opening and closing one per poll.
        The caller's ``finally: _close_ssh()`` handles cleanup.
        """
        self.job_check_last_time = datetime.today()

        if not self.server_machine.queuing:
            return False

        # A failed listing command (non-zero exit status) raises in
        # ``run_command``.  For local bash/ps mode, job_number is a PID.
        job_list = self.server_machine.get_job_list_as_text()
        tokens = {t for line in job_list for t in line.split()}
        if not tokens:
            # Empty listing: confirm it over ``empty_listing_polls``
            # consecutive polls before the job counts as done.
            self.empty_listings_in_a_row += 1
            if self.empty_listings_in_a_row < self.empty_listing_polls:
                logger.warning(
                    f"  jobcheck listing is empty ({self.empty_listings_in_a_row} of {self.empty_listing_polls} "
                    f"polls); Job {self.job_number} is treated as running until the empty listing is confirmed."
                )
                return True
        else:
            self.empty_listings_in_a_row = 0

        if _job_listed(self.job_number, tokens):
            logger.info(f"  Job {self.job_number} is running.")
            self.job_running = True
            # Also registers a job resumed from workflow_state.toml.
            self._track_job()
            flag = True
        else:
            logger.info(f"  Job {self.job_number} is done.")
            self.job_running = False
            self._untrack_job()
            flag = False

        return flag

    def jobnum_check(self) -> bool:
        """Check if we are below the max_job_submit limit.

        What is counted depends on the queue key ``max_job_submit_scope``:

        * ``"driver"`` (default): the job IDs that this process submitted
          (or resumed) for the same machine and scheduler queue and that
          still appear as a token in the ``jobcheck`` output, matched like
          :meth:`jobcheck`.  This needs no knowledge of the scheduler's
          column layout and ignores jobs of other drivers on the same
          account.  IDs no longer listed are dropped from the registry.
          An empty listing is treated as unreliable: the registered IDs
          are counted but kept.
        * ``"user"``: every ``jobcheck`` line that shows the login name and
          the queue name as separate words, i.e. all of the account's jobs
          in that queue.  This requires a ``jobcheck`` command whose
          output contains the user name (e.g. ``qstat`` on most PBS sites,
          ``squeue --noheader``).

        The SSH connection is kept open so that the subsequent
        ``job_submit`` call can reuse it.  The caller's ``finally``
        block handles cleanup.
        """
        if not self.server_machine.queuing:
            return True

        job_list = self.server_machine.get_job_list_as_text()

        if self.max_job_submit_scope == "user":
            queue_name = self.queue_data.get("queue", "")
            username = self.server_machine.username
            count = sum(
                1
                for line in job_list
                if (
                    re.match(f".*\\s{username}\\s.*\\s{queue_name}\\s.*", line)
                    or re.match(f".*\\s{queue_name}\\s.*\\s{username}\\s.*", line)
                )
            )
            logger.info(f"  {count} of your jobs in queue on {self.server_machine.name} (max_job_submit={self.max_job_submit})")
            return count < self.max_job_submit

        tokens = {t for line in job_list for t in line.split()}
        with _DRIVER_JOBS_LOCK:
            tracked = _DRIVER_JOBS.setdefault(self._driver_jobs_key(), set())
            if tokens:
                tracked -= {j for j in tracked if not _job_listed(j, tokens)}
            count = len(tracked)
        logger.info(
            f"  {count} jobs of this driver in queue on {self.server_machine.name} (max_job_submit={self.max_job_submit})"
        )
        return count < self.max_job_submit

    def _driver_jobs_key(self) -> tuple[str, str]:
        """Registry key of this job (see the module-level ``_driver_jobs_key``)."""
        return _driver_jobs_key(self.server_machine.name, self.queue_data, self.queue_label)

    def _track_job(self) -> None:
        if self.job_number is None:
            return
        with _DRIVER_JOBS_LOCK:
            _DRIVER_JOBS.setdefault(self._driver_jobs_key(), set()).add(str(self.job_number))

    def _untrack_job(self) -> None:
        if self.job_number is None:
            return
        with _DRIVER_JOBS_LOCK:
            _DRIVER_JOBS.get(self._driver_jobs_key(), set()).discard(str(self.job_number))

    # -- Fetch results ---------------------------------------------

    def fetch_job(self, from_objects=None, exclude_patterns=None, *, work_dir=None, optional_patterns=None):
        """Fetch job results from the remote machine.

        Args:
            from_objects (list[str], optional):
                Basenames or glob patterns of files to download.
            exclude_patterns (list[str], optional):
                Glob patterns to exclude.
            work_dir (str, optional):
                Absolute path to the local job directory.  When *None*,
                falls back to ``os.getcwd()`` for backward compatibility.
            optional_patterns (list[str], optional):
                Basenames or glob patterns of non-essential files.
                Missing files matching these patterns produce a warning
                instead of an error.

        Raises:
            FileNotFoundError: A required (non-glob, non-optional) entry of
                *from_objects* is missing on the server (remote machine) or
                in *work_dir* (local machine).
        """
        from_objects = from_objects or []
        exclude_patterns = exclude_patterns or []
        optional_patterns = optional_patterns or []

        if self.server_machine.machine_type != "local":
            self.data_transfer.get_objects(
                from_objects=from_objects,
                exclude_patterns=exclude_patterns,
                work_dir=work_dir,
                optional_patterns=optional_patterns,
            )
        else:
            # A local job writes its files into *work_dir* itself, so
            # there is nothing to transfer.  A required (non-glob,
            # non-optional) file that is absent means the job never ran
            # (e.g. the scheduler accepted it and dropped it before it
            # started); fail like the remote transfer does instead of
            # letting the caller record the job as fetched.
            local_cwd = os.path.abspath(work_dir) if work_dir else os.path.abspath(os.getcwd())
            for obj in from_objects:
                if any(c in obj for c in ("*", "?", "[")):
                    continue
                if any(fnmatch.fnmatch(obj, pat) for pat in optional_patterns):
                    continue
                path = os.path.join(local_cwd, obj)
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"{path} not found after job {self.job_number} left the queue on "
                        f"{self.server_machine.name}: the job did not write its output "
                        f"(it may have been dropped or cancelled before it started)."
                    )

        self.job_fetch_date = datetime.today()
        self._close_ssh()

    # -- Delete a running job --------------------------------------

    def delete_job(self):
        self.server_machine.delete_job(jobid=self.job_number)
        self.job_running = False
        self._untrack_job()
        self._close_ssh()

    # -- Job accounting -----------------------------------------

    def job_acct(self) -> tuple[str, str, str] | None:
        """Run the scheduler accounting command and return raw output.

        Reads the ``jobacct`` field from ``machine_data.yaml`` and
        executes ``{jobacct} {job_id}``.  No parsing or flag-injection
        is performed -- the user specifies the complete command with
        flags in the config.

        Returns:
            tuple[str, str, str] | None:
                ``(command, stdout, stderr)`` on success.
                ``None`` if ``jobacct`` is not configured, the machine does
                not use a queuing system, or the command fails.
        """
        if not self.server_machine.queuing:
            return None
        jobacct_cmd = self.server_machine._get("jobacct", default=None)
        if not jobacct_cmd or not self.job_number:
            return None
        try:
            command = f"{jobacct_cmd} {self.job_number}"
            stdout, stderr = self.server_machine.run_command(command)
            return command, stdout, stderr
        except Exception as e:
            logger.warning(f"job_acct failed for job {self.job_number}: {e}")
            return None

    # -- Helper ----------------------------------------------------

    def _close_ssh(self):
        # No-op: the underlying Machine is shared across all
        # JobSubmissions via the process-wide registry in
        # ``_machine.get_machine``. Per-job teardown would force the
        # next ``_submit_and_wait`` to re-authenticate -- which on
        # clusters that require an OTP means a fresh prompt for every
        # step. The transport is closed once at workflow end via
        # :py:func:`close_all_machines`.
        return
