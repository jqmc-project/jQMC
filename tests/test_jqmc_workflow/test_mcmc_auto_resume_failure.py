"""Automatic MCMC relaunch must validate fetched failures before using old statistics."""

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

import asyncio

import h5py
import pytest

from jqmc_workflow.mcmc_workflow import MCMC_Workflow
from jqmc_workflow.state import WorkflowStatus, add_job, set_estimation

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


@pytest.mark.parametrize("dt", [2.0, "auto"], ids=["fixed-Dt", "auto-Dt"])
@pytest.mark.parametrize("target_reached", [False, True], ids=["continue", "cached-target"])
@pytest.mark.parametrize("failed_continuation", [False, True], ids=["normal", "floating-point-error"])
def test_auto_relaunch_validates_fetched_continuation_before_reusing_checkpoint(
    tmp_path, target_reached, failed_continuation, dt
):
    """An abnormal fetched continuation cannot be hidden by its predecessor's finite checkpoint."""
    work_dir = str(tmp_path)
    # The failed continuation did not replace the last healthy checkpoint.
    with h5py.File(tmp_path / "restart.h5", "w") as checkpoint:
        checkpoint.create_group("rank_0/driver_config").attrs["mcmc_counter"] = 20
    for step in (1, 2):
        output = tmp_path / f"mcmc_{step}.out"
        output.write_text(
            "Traceback (most recent call last):\nFloatingPointError: MCMC restart state rebuild: ln_abs_det\n"
            if step == 2 and failed_continuation
            else "Program ends\n"
        )
        add_job(work_dir, f"mcmc_{step}.toml", output.name, f"job-{step}", "local", status="fetched", step=step)
    error = 0.0001 if target_reached else 0.1
    set_estimation(
        work_dir,
        estimated_steps=10,
        last_energy=-1.0,
        last_energy_error=error,
        accumulated_measurement_steps=20,
        last_num_mcmc_bin_blocks=5,
        last_num_mcmc_warmup_steps=0,
    )
    workflow = MCMC_Workflow(max_continuation=3, target_error=0.001, mpc=False, Dt=dt)
    workflow.project_dir = work_dir
    submissions = []
    calibrations = []

    async def calibrate(_work_dir):
        calibrations.append("Dt")
        workflow.Dt = 2.0

    async def submit(input_file, output_file, **kwargs):
        step = int(kwargs["step"])
        submissions.append(step)
        (tmp_path / output_file).write_text("Program ends\n")
        add_job(work_dir, input_file, output_file, f"job-{step}", "local", status="fetched", step=step)

    workflow._submit_and_wait = submit
    workflow._run_dt_calibration = calibrate
    workflow._compute_energy = lambda *_args, **_kwargs: (-1.0, error, None, None)
    status, _, values = asyncio.run(workflow.run())
    if failed_continuation:
        assert calibrations == []
        assert submissions == []
        assert status == WorkflowStatus.FAILED
        assert "Abnormal termination" in values["error"]
        assert "mcmc_2.out" in values["error"]
    else:
        assert calibrations == (["Dt"] if dt == "auto" else [])
        assert status == WorkflowStatus.COMPLETED
        assert submissions == ([] if target_reached else [3])
