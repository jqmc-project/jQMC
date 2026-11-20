"""Tests for jqmc_workflow.state -- workflow_state.toml persistence + helpers.

Layer 1 (pure unit) + Layer 2 (real filesystem on tmp_path): no subprocess, no SSH.

Test class layout (expanded to cover
the 25 public helpers + 3 enums in this 808-LOC module):
    TestStatusEnums            -- WorkflowStatus / JobStatus / CompletionStatus invariants
    TestCreateState            -- create + reset, preserve [estimation], [[jobs]], [input_fingerprints]
    TestReadState              -- round-trip, missing file, legacy [job] migration
    TestUpdateStatus           -- status update + phase + result_/workflow_ extras
    TestJobHistory             -- add_job / update_job / get_job / get_job_by_step / get_jobs
    TestResultsAndEstimation   -- set_result / set_estimation / get_estimation
    TestValidateCompletion     -- 4 checks (program-ends, non-finite, no energy, threshold)
    TestReconcileFetchedJobs   -- promotion to "fetched" when Program ends + .h5 present
    TestReconcileRecursive     -- walks nested workflow dirs
    TestDiscovery              -- get_all_workflow_statuses + get_workflow_summary
    TestErrorAndAccounting     -- set_error + save_job_accounting
    TestArtifacts              -- register / lineage / registry
    TestInputFingerprints      -- set / get
    TestAtomicWrite            -- _write semantics (no .tmp leftover)
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

import math
from pathlib import Path

import pytest
import toml

from jqmc_workflow.state import (
    STATE_FILENAME,
    VALID_JOB_STATUSES,
    VALID_STATUSES,
    CompletionStatus,
    JobStatus,
    WorkflowStatus,
    _write,
    add_job,
    create_state,
    get_all_workflow_statuses,
    get_artifact_lineage,
    get_artifact_registry,
    get_estimation,
    get_input_fingerprints,
    get_job,
    get_job_by_step,
    get_jobs,
    get_workflow_summary,
    read_state,
    reconcile_fetched_jobs,
    reconcile_fetched_jobs_recursive,
    register_artifact,
    save_job_accounting,
    set_error,
    set_estimation,
    set_input_fingerprints,
    set_result,
    update_job,
    update_status,
    validate_completion,
)

pytestmark = [pytest.mark.with_chdir]


# -- Helpers ---------------------------------------------------------


def _bootstrap(directory: Path, label: str = "test", wf_type: str = "Workflow") -> dict:
    """Create a fresh workflow_state.toml under *directory* and return the dict."""
    return create_state(str(directory), label=label, workflow_type=wf_type)


def _write_program_ends(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("some output\nProgram ends\n")


def _write_no_program_ends(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("some output without the marker\n")


# -- Tests -----------------------------------------------------------


class TestStatusEnums:
    @pytest.mark.light
    def test_workflow_status_values(self):
        assert {s.value for s in WorkflowStatus} == VALID_STATUSES

    @pytest.mark.light
    def test_job_status_values(self):
        assert {s.value for s in JobStatus} == VALID_JOB_STATUSES

    @pytest.mark.light
    def test_completion_status_has_three_members(self):
        assert {s.value for s in CompletionStatus} == {"ok", "failed", "incomplete"}

    @pytest.mark.light
    def test_workflow_status_string_inheritance(self):
        """WorkflowStatus is a str-enum: value == name comparable to plain str."""
        assert WorkflowStatus.PENDING == "pending"


class TestCreateState:
    @pytest.mark.light
    def test_creates_file_with_expected_sections(self, tmp_path):
        state = _bootstrap(tmp_path)
        assert (tmp_path / STATE_FILENAME).exists()
        assert state["workflow"]["label"] == "test"
        assert state["workflow"]["status"] == "pending"
        assert state["jobs"] == []
        assert state["result"] == {}

    @pytest.mark.light
    def test_invalid_status_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid status"):
            create_state(str(tmp_path), label="x", workflow_type="W", status="bogus")

    @pytest.mark.light
    def test_preserves_estimation_on_reset(self, tmp_path):
        _bootstrap(tmp_path)
        set_estimation(str(tmp_path), pilot_steps=100, target_error=0.001)
        create_state(str(tmp_path), label="rerun", workflow_type="W")
        assert get_estimation(str(tmp_path)) == {"pilot_steps": 100, "target_error": 0.001}

    @pytest.mark.light
    def test_preserves_jobs_on_reset(self, tmp_path):
        _bootstrap(tmp_path)
        add_job(str(tmp_path), "input.toml", "out.o", "12345", "local")
        create_state(str(tmp_path), label="rerun", workflow_type="W")
        jobs = get_jobs(str(tmp_path))
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "12345"

    @pytest.mark.light
    def test_preserves_input_fingerprints_on_reset(self, tmp_path):
        _bootstrap(tmp_path)
        set_input_fingerprints(str(tmp_path), {"input.toml": {"size": 100, "mtime": 1.0}})
        create_state(str(tmp_path), label="rerun", workflow_type="W")
        assert get_input_fingerprints(str(tmp_path))["input.toml"]["size"] == 100

    @pytest.mark.light
    def test_created_and_updated_timestamps_present(self, tmp_path):
        state = _bootstrap(tmp_path)
        assert "T" in state["workflow"]["created_at"]
        assert "T" in state["workflow"]["updated_at"]


class TestReadState:
    @pytest.mark.light
    def test_missing_file_returns_empty_dict(self, tmp_path):
        assert read_state(str(tmp_path)) == {}

    @pytest.mark.light
    def test_round_trip(self, tmp_path):
        original = {"workflow": {"label": "x", "status": "pending"}, "jobs": []}
        _write(str(tmp_path), original)
        assert read_state(str(tmp_path)) == original

    @pytest.mark.light
    def test_legacy_single_job_migrated_to_list(self, tmp_path):
        legacy = {"workflow": {"label": "x"}, "job": {"input_file": "a.toml", "status": "submitted"}}
        _write(str(tmp_path), legacy)
        state = read_state(str(tmp_path))
        assert "job" not in state
        assert state["jobs"] == [{"input_file": "a.toml", "status": "submitted"}]
        reloaded = toml.load(tmp_path / STATE_FILENAME)
        assert "job" not in reloaded
        assert "jobs" in reloaded

    @pytest.mark.light
    def test_legacy_empty_job_migrated_to_empty_list(self, tmp_path):
        legacy = {"workflow": {"label": "x"}, "job": {}}
        _write(str(tmp_path), legacy)
        state = read_state(str(tmp_path))
        assert state["jobs"] == []


class TestUpdateStatus:
    @pytest.mark.light
    def test_status_string_accepted(self, tmp_path):
        _bootstrap(tmp_path)
        update_status(str(tmp_path), "running")
        assert read_state(str(tmp_path))["workflow"]["status"] == "running"

    @pytest.mark.light
    def test_workflow_status_enum_accepted(self, tmp_path):
        _bootstrap(tmp_path)
        update_status(str(tmp_path), WorkflowStatus.COMPLETED)
        assert read_state(str(tmp_path))["workflow"]["status"] == "completed"

    @pytest.mark.light
    def test_invalid_status_raises(self, tmp_path):
        _bootstrap(tmp_path)
        with pytest.raises(ValueError, match="Invalid status"):
            update_status(str(tmp_path), "bogus_status")

    @pytest.mark.light
    def test_phase_written_when_provided(self, tmp_path):
        _bootstrap(tmp_path)
        update_status(str(tmp_path), "running", phase="vmc_pilot")
        assert read_state(str(tmp_path))["workflow"]["phase"] == "vmc_pilot"

    @pytest.mark.light
    def test_result_prefix_routes_to_result_section(self, tmp_path):
        _bootstrap(tmp_path)
        update_status(str(tmp_path), "completed", result_energy=-1.234)
        assert read_state(str(tmp_path))["result"]["energy"] == -1.234

    @pytest.mark.light
    def test_non_result_extras_routed_to_workflow_section(self, tmp_path):
        _bootstrap(tmp_path)
        update_status(str(tmp_path), "running", note="probe phase")
        assert read_state(str(tmp_path))["workflow"]["note"] == "probe phase"

    @pytest.mark.light
    def test_works_without_existing_state(self, tmp_path):
        update_status(str(tmp_path), "pending")
        state = read_state(str(tmp_path))
        assert state["workflow"]["status"] == "pending"
        assert "label" in state["workflow"]

    @pytest.mark.light
    def test_phase_enum_value_extracted(self, tmp_path):
        _bootstrap(tmp_path)
        from jqmc_workflow._phase import ScientificPhase

        update_status(str(tmp_path), "running", phase=ScientificPhase.VMC_PILOT)
        assert read_state(str(tmp_path))["workflow"]["phase"] == "vmc_pilot"


class TestJobHistory:
    @pytest.mark.light
    def test_add_job_appends_to_jobs_list(self, tmp_path):
        _bootstrap(tmp_path)
        job = add_job(str(tmp_path), "input.toml", "out.o", "12345", "local")
        assert job["job_id"] == "12345"
        assert job["status"] == "submitted"
        assert get_jobs(str(tmp_path)) == [job]

    @pytest.mark.light
    def test_add_job_with_optional_fields(self, tmp_path):
        _bootstrap(tmp_path)
        job = add_job(
            str(tmp_path),
            "input.toml",
            "out.o",
            "12345",
            "local",
            step=3,
            run_id="abc",
            job_stdout="stdout.txt",
            job_stderr="stderr.txt",
        )
        assert job["step"] == 3
        assert job["run_id"] == "abc"
        assert job["job_stdout"] == "stdout.txt"

    @pytest.mark.light
    def test_add_job_invalid_status_raises(self, tmp_path):
        _bootstrap(tmp_path)
        with pytest.raises(ValueError, match="Invalid job status"):
            add_job(str(tmp_path), "i", "o", "1", "local", status="bogus")

    @pytest.mark.light
    def test_update_job_modifies_last_matching_record(self, tmp_path):
        _bootstrap(tmp_path)
        add_job(str(tmp_path), "input.toml", "out.o", "1", "local")
        add_job(str(tmp_path), "input.toml", "out.o", "2", "local")
        update_job(str(tmp_path), "input.toml", status="completed", completed_at="2026-06-23T10:00:00")
        jobs = get_jobs(str(tmp_path))
        assert jobs[0]["status"] == "submitted"
        assert jobs[1]["status"] == "completed"
        assert jobs[1]["completed_at"] == "2026-06-23T10:00:00"

    @pytest.mark.light
    def test_update_job_no_match_is_noop(self, tmp_path):
        _bootstrap(tmp_path)
        update_job(str(tmp_path), "nope.toml", status="completed")
        assert get_jobs(str(tmp_path)) == []

    @pytest.mark.light
    def test_get_job_returns_latest_match(self, tmp_path):
        _bootstrap(tmp_path)
        add_job(str(tmp_path), "input.toml", "out.o", "1", "local")
        add_job(str(tmp_path), "input.toml", "out.o", "2", "local")
        assert get_job(str(tmp_path), "input.toml")["job_id"] == "2"

    @pytest.mark.light
    def test_get_job_missing_returns_empty_dict(self, tmp_path):
        _bootstrap(tmp_path)
        assert get_job(str(tmp_path), "nope.toml") == {}

    @pytest.mark.light
    def test_get_job_by_step_returns_latest_match(self, tmp_path):
        _bootstrap(tmp_path)
        add_job(str(tmp_path), "input.toml", "out.o", "1", "local", step=1)
        add_job(str(tmp_path), "input.toml", "out.o", "2", "local", step=2)
        assert get_job_by_step(str(tmp_path), 2)["job_id"] == "2"
        assert get_job_by_step(str(tmp_path), 99) == {}

    @pytest.mark.light
    def test_get_jobs_empty_when_no_file(self, tmp_path):
        assert get_jobs(str(tmp_path)) == []


class TestResultsAndEstimation:
    @pytest.mark.light
    def test_set_result_merges_into_result_section(self, tmp_path):
        _bootstrap(tmp_path)
        set_result(str(tmp_path), energy=-1.0, energy_error=0.001)
        set_result(str(tmp_path), atomic_forces=[1.0, 2.0])
        result = read_state(str(tmp_path))["result"]
        assert result["energy"] == -1.0
        assert result["atomic_forces"] == [1.0, 2.0]

    @pytest.mark.light
    def test_set_result_noop_when_no_state(self, tmp_path):
        set_result(str(tmp_path), energy=-1.0)
        assert not (tmp_path / STATE_FILENAME).exists()

    @pytest.mark.light
    def test_set_estimation_round_trip(self, tmp_path):
        _bootstrap(tmp_path)
        set_estimation(str(tmp_path), pilot_steps=100, target_error=1e-3)
        assert get_estimation(str(tmp_path)) == {"pilot_steps": 100, "target_error": 1e-3}

    @pytest.mark.light
    def test_set_estimation_merges_on_repeat_call(self, tmp_path):
        _bootstrap(tmp_path)
        set_estimation(str(tmp_path), pilot_steps=100)
        set_estimation(str(tmp_path), target_error=1e-3)
        assert get_estimation(str(tmp_path)) == {"pilot_steps": 100, "target_error": 1e-3}

    @pytest.mark.light
    def test_get_estimation_empty_when_unset(self, tmp_path):
        _bootstrap(tmp_path)
        assert get_estimation(str(tmp_path)) == {}


class TestValidateCompletion:
    @pytest.mark.light
    def test_ok_when_all_jobs_normal_no_target(self, tmp_path):
        _bootstrap(tmp_path)
        add_job(str(tmp_path), "i.toml", "out.o", "1", "local", status="fetched")
        _write_program_ends(tmp_path / "out.o")
        update_job(str(tmp_path), "i.toml", status="fetched")
        status, msg = validate_completion(str(tmp_path))
        assert status == CompletionStatus.OK
        assert msg == ""

    @pytest.mark.light
    def test_failed_when_program_ends_missing(self, tmp_path):
        _bootstrap(tmp_path)
        add_job(str(tmp_path), "i.toml", "out.o", "1", "local", status="fetched")
        _write_no_program_ends(tmp_path / "out.o")
        update_job(str(tmp_path), "i.toml", status="fetched")
        status, msg = validate_completion(str(tmp_path))
        assert status == CompletionStatus.FAILED
        assert "Abnormal termination" in msg

    @pytest.mark.light
    def test_failed_on_non_finite_energy(self, tmp_path):
        _bootstrap(tmp_path)
        status, msg = validate_completion(str(tmp_path), output_values={"energy": math.inf})
        assert status == CompletionStatus.FAILED
        assert "Non-finite" in msg

    @pytest.mark.light
    def test_failed_on_nan_energy(self, tmp_path):
        _bootstrap(tmp_path)
        status, _ = validate_completion(str(tmp_path), output_values={"energy": math.nan})
        assert status == CompletionStatus.FAILED

    @pytest.mark.light
    def test_failed_when_required_energy_missing(self, tmp_path):
        """Fixed-step MCMC/LRDMC: a completed run without an energy is not a success."""
        _bootstrap(tmp_path)
        status, msg = validate_completion(str(tmp_path), output_values={}, require_energy=True)
        assert status == CompletionStatus.FAILED
        assert "No energy estimate" in msg
        status, _ = validate_completion(str(tmp_path), output_values={"energy": -1.0}, require_energy=True)
        assert status == CompletionStatus.OK

    @pytest.mark.light
    def test_missing_energy_ok_when_not_required(self, tmp_path):
        """Workflows without an energy (WF, QE, VMC) keep the post-hoc OK."""
        _bootstrap(tmp_path)
        status, _ = validate_completion(str(tmp_path), output_values={})
        assert status == CompletionStatus.OK

    @pytest.mark.light
    def test_incomplete_when_target_set_but_no_energy(self, tmp_path):
        _bootstrap(tmp_path)
        status, msg = validate_completion(str(tmp_path), target_error=1e-3)
        assert status == CompletionStatus.INCOMPLETE
        assert "No energy estimate" in msg

    @pytest.mark.light
    def test_incomplete_when_energy_error_above_threshold(self, tmp_path):
        _bootstrap(tmp_path)
        status, _ = validate_completion(
            str(tmp_path),
            output_values={"energy": -1.0, "energy_error": 5e-3},
            target_error=1e-3,
        )
        assert status == CompletionStatus.INCOMPLETE

    @pytest.mark.light
    def test_ok_when_energy_error_meets_threshold(self, tmp_path):
        _bootstrap(tmp_path)
        status, msg = validate_completion(
            str(tmp_path),
            output_values={"energy": -1.0, "energy_error": 1e-4},
            target_error=1e-3,
        )
        assert status == CompletionStatus.OK
        assert "<=" in msg

    @pytest.mark.light
    def test_target_tol_relaxes_threshold(self, tmp_path):
        _bootstrap(tmp_path)
        status, _ = validate_completion(
            str(tmp_path),
            output_values={"energy": -1.0, "energy_error": 1.1e-3},
            target_error=1e-3,
            target_tol=1.2,
        )
        assert status == CompletionStatus.OK

    @pytest.mark.light
    def test_incomplete_when_energy_error_missing(self, tmp_path):
        _bootstrap(tmp_path)
        status, _ = validate_completion(
            str(tmp_path),
            output_values={"energy": -1.0},
            target_error=1e-3,
        )
        assert status == CompletionStatus.INCOMPLETE


class TestReconcileFetchedJobs:
    @pytest.mark.light
    def test_promotes_submitted_to_fetched(self, tmp_path):
        _bootstrap(tmp_path)
        add_job(str(tmp_path), "i.toml", "out.o", "1", "local", status="submitted")
        _write_program_ends(tmp_path / "out.o")
        (tmp_path / "restart.h5").touch()
        n = reconcile_fetched_jobs(str(tmp_path))
        assert n == 1
        job = get_jobs(str(tmp_path))[0]
        assert job["status"] == "fetched"
        assert "fetched_at" in job

    @pytest.mark.light
    def test_skips_when_no_h5_present(self, tmp_path):
        _bootstrap(tmp_path)
        add_job(str(tmp_path), "i.toml", "out.o", "1", "local", status="submitted")
        _write_program_ends(tmp_path / "out.o")
        n = reconcile_fetched_jobs(str(tmp_path))
        assert n == 0
        assert get_jobs(str(tmp_path))[0]["status"] == "submitted"

    @pytest.mark.light
    def test_skips_when_program_ends_absent(self, tmp_path):
        _bootstrap(tmp_path)
        add_job(str(tmp_path), "i.toml", "out.o", "1", "local", status="submitted")
        _write_no_program_ends(tmp_path / "out.o")
        (tmp_path / "restart.h5").touch()
        assert reconcile_fetched_jobs(str(tmp_path)) == 0

    @pytest.mark.light
    def test_skips_already_fetched_or_failed(self, tmp_path):
        _bootstrap(tmp_path)
        add_job(str(tmp_path), "i1.toml", "out1.o", "1", "local", status="fetched")
        add_job(str(tmp_path), "i2.toml", "out2.o", "2", "local", status="failed")
        for o in ("out1.o", "out2.o"):
            _write_program_ends(tmp_path / o)
        (tmp_path / "restart.h5").touch()
        assert reconcile_fetched_jobs(str(tmp_path)) == 0

    @pytest.mark.light
    def test_returns_zero_when_no_jobs(self, tmp_path):
        _bootstrap(tmp_path)
        assert reconcile_fetched_jobs(str(tmp_path)) == 0

    @pytest.mark.light
    def test_returns_zero_when_directory_missing(self, tmp_path):
        assert reconcile_fetched_jobs(str(tmp_path / "nonexistent")) == 0


class TestReconcileRecursive:
    @pytest.mark.light
    def test_walks_into_nested_pilot_dirs(self, tmp_path):
        _bootstrap(tmp_path)
        add_job(str(tmp_path), "i.toml", "out.o", "1", "local", status="submitted")
        _write_program_ends(tmp_path / "out.o")
        (tmp_path / "restart.h5").touch()

        pilot = tmp_path / "_pilot_b"
        pilot.mkdir()
        _bootstrap(pilot, label="pilot", wf_type="MCMC_Pilot")
        add_job(str(pilot), "p_i.toml", "p_out.o", "2", "local", status="submitted")
        _write_program_ends(pilot / "p_out.o")
        (pilot / "restart.h5").touch()

        total = reconcile_fetched_jobs_recursive(str(tmp_path))
        assert total == 2
        assert get_jobs(str(tmp_path))[0]["status"] == "fetched"
        assert get_jobs(str(pilot))[0]["status"] == "fetched"

    @pytest.mark.light
    def test_continues_past_corrupted_state_file(self, tmp_path):
        _bootstrap(tmp_path)
        add_job(str(tmp_path), "i.toml", "out.o", "1", "local", status="submitted")
        _write_program_ends(tmp_path / "out.o")
        (tmp_path / "restart.h5").touch()

        broken = tmp_path / "_pilot_corrupt"
        broken.mkdir()
        (broken / STATE_FILENAME).write_text("not = valid \xff toml ===")

        total = reconcile_fetched_jobs_recursive(str(tmp_path))
        assert total >= 1
        assert get_jobs(str(tmp_path))[0]["status"] == "fetched"


class TestDiscovery:
    @pytest.mark.light
    def test_get_all_workflow_statuses_finds_nested(self, tmp_path):
        _bootstrap(tmp_path, label="root", wf_type="VMC_Workflow")
        sub = tmp_path / "sub"
        sub.mkdir()
        _bootstrap(sub, label="sub", wf_type="MCMC_Workflow")
        results = get_all_workflow_statuses(str(tmp_path))
        labels = {r["label"] for r in results}
        assert {"root", "sub"} <= labels

    @pytest.mark.light
    def test_get_all_workflow_statuses_skips_pilot_subdirs(self, tmp_path):
        _bootstrap(tmp_path, label="root", wf_type="VMC_Workflow")
        pilot = tmp_path / "_pilot_b"
        pilot.mkdir()
        _bootstrap(pilot, label="pilot", wf_type="MCMC_Pilot")
        results = get_all_workflow_statuses(str(tmp_path))
        labels = {r["label"] for r in results}
        assert "pilot" not in labels
        assert "root" in labels

    @pytest.mark.light
    def test_get_all_workflow_statuses_sorted(self, tmp_path):
        for name in ("b", "a", "c"):
            sub = tmp_path / name
            sub.mkdir()
            _bootstrap(sub, label=name, wf_type="W")
        results = get_all_workflow_statuses(str(tmp_path))
        dirs = [r["directory"] for r in results]
        assert dirs == sorted(dirs)

    @pytest.mark.light
    def test_get_workflow_summary_populated(self, tmp_path):
        _bootstrap(tmp_path, label="lbl", wf_type="VMC_Workflow")
        update_status(str(tmp_path), "running", phase="vmc_pilot")
        add_job(str(tmp_path), "i.toml", "o.o", "1", "local")
        s = get_workflow_summary(str(tmp_path))
        assert s["workflow"]["label"] == "lbl"
        assert s["phase"] == "vmc_pilot"
        assert s["num_jobs"] == 1
        assert isinstance(s["allowed_actions"], list)
        assert len(s["allowed_actions"]) > 0

    @pytest.mark.light
    def test_get_workflow_summary_returns_empty_on_missing(self, tmp_path):
        assert get_workflow_summary(str(tmp_path)) == {}

    @pytest.mark.light
    def test_get_workflow_summary_handles_unknown_phase_gracefully(self, tmp_path):
        """A bogus phase falls back to empty allowed_actions, not a crash."""
        _bootstrap(tmp_path)
        update_status(str(tmp_path), "running", phase="totally_bogus")
        s = get_workflow_summary(str(tmp_path))
        assert s["allowed_actions"] == []


class TestErrorAndAccounting:
    @pytest.mark.light
    def test_set_error_writes_error_section(self, tmp_path):
        _bootstrap(tmp_path)
        set_error(str(tmp_path), "boom", traceback="Traceback...", exception_type="RuntimeError")
        err = read_state(str(tmp_path))["error"]
        assert err["message"] == "boom"
        assert err["traceback"] == "Traceback..."
        assert err["exception_type"] == "RuntimeError"

    @pytest.mark.light
    def test_set_error_overwrites_previous(self, tmp_path):
        _bootstrap(tmp_path)
        set_error(str(tmp_path), "first")
        set_error(str(tmp_path), "second")
        assert read_state(str(tmp_path))["error"] == {"message": "second"}

    @pytest.mark.light
    def test_save_job_accounting_with_job_id(self, tmp_path):
        cmd, fname = save_job_accounting(
            str(tmp_path),
            command="qstat 12345",
            stdout="Job state: R\n",
            stderr="",
            job_id="12345",
        )
        assert cmd == "qstat 12345"
        assert fname == "job_accounting_12345.txt"
        text = (tmp_path / fname).read_text()
        assert "qstat 12345" in text
        assert "Job state: R" in text

    @pytest.mark.light
    def test_save_job_accounting_default_filename(self, tmp_path):
        _, fname = save_job_accounting(str(tmp_path), command="ls", stdout="x")
        assert fname == "job_accounting.txt"

    @pytest.mark.light
    def test_save_job_accounting_includes_stderr(self, tmp_path):
        _, fname = save_job_accounting(str(tmp_path), command="x", stdout="out", stderr="err")
        text = (tmp_path / fname).read_text()
        assert "--- stderr ---" in text
        assert "err" in text


class TestArtifacts:
    @pytest.mark.light
    def test_register_and_get_lineage(self, tmp_path):
        _bootstrap(tmp_path)
        register_artifact(
            str(tmp_path),
            filename="hamiltonian.h5",
            produced_by_job="12345",
            upstream=[{"filename": "input.toml", "fingerprint": "abc"}],
        )
        a = get_artifact_lineage(str(tmp_path), "hamiltonian.h5")
        assert a is not None
        assert a["filename"] == "hamiltonian.h5"
        assert a["produced_by_job"] == "12345"
        assert a["upstream"][0]["filename"] == "input.toml"

    @pytest.mark.light
    def test_register_replaces_existing_entry(self, tmp_path):
        _bootstrap(tmp_path)
        register_artifact(str(tmp_path), "out.h5", produced_by_job="1")
        register_artifact(str(tmp_path), "out.h5", produced_by_job="2")
        registry = get_artifact_registry(str(tmp_path))
        assert len(registry) == 1
        assert registry[0]["produced_by_job"] == "2"

    @pytest.mark.light
    def test_get_lineage_returns_none_when_absent(self, tmp_path):
        _bootstrap(tmp_path)
        assert get_artifact_lineage(str(tmp_path), "nope.h5") is None

    @pytest.mark.light
    def test_registry_empty_when_no_artifacts(self, tmp_path):
        _bootstrap(tmp_path)
        assert get_artifact_registry(str(tmp_path)) == []


class TestInputFingerprints:
    @pytest.mark.light
    def test_set_get_round_trip(self, tmp_path):
        _bootstrap(tmp_path)
        fps = {"input.toml": {"size": 100, "mtime": 1.5}, "data.h5": {"size": 4096, "mtime": 2.5}}
        set_input_fingerprints(str(tmp_path), fps)
        assert get_input_fingerprints(str(tmp_path)) == fps

    @pytest.mark.light
    def test_get_empty_when_unset(self, tmp_path):
        _bootstrap(tmp_path)
        assert get_input_fingerprints(str(tmp_path)) == {}

    @pytest.mark.light
    def test_overwrites_previous(self, tmp_path):
        _bootstrap(tmp_path)
        set_input_fingerprints(str(tmp_path), {"a": {"size": 1, "mtime": 1.0}})
        set_input_fingerprints(str(tmp_path), {"b": {"size": 2, "mtime": 2.0}})
        assert get_input_fingerprints(str(tmp_path)) == {"b": {"size": 2, "mtime": 2.0}}


class TestAtomicWrite:
    @pytest.mark.light
    def test_no_tmp_left_behind(self, tmp_path):
        _bootstrap(tmp_path)
        update_status(str(tmp_path), "running")
        assert not (tmp_path / (STATE_FILENAME + ".tmp")).exists()

    @pytest.mark.light
    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        _write(str(nested), {"workflow": {"label": "x"}})
        assert (nested / STATE_FILENAME).exists()
