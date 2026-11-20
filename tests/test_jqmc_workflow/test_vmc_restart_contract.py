"""VMC continuation contract across jqmc-workflow and the jQMC CLI.

The workflow owns the continuation TOML while the CLI owns dispatch to the
single-k or twist-averaged optimizer.  These tests pin both sides of that
boundary so a valid ``restart = true`` cannot be silently downgraded to a
fresh twist optimization.
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

import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from jqmc.jqmc_cli import _run_twist_average_dispatch
from jqmc_workflow.vmc_workflow import VMC_Workflow

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]


def test_crystal_gamma_continuation_toml_requests_restart(tmp_path):
    """The single-twist crystal-Gamma route receives an explicit restart request."""
    input_path = tmp_path / "vmc_gamma_restart.toml"
    workflow = VMC_Workflow(num_mcmc_per_measurement=4)

    workflow._generate_input(
        num_mcmc_steps=20,
        num_opt_steps=2,
        input_file=str(input_path),
        restart=True,
        restart_chk="restart.h5",
    )

    with input_path.open("rb") as stream:
        control = tomllib.load(stream)["control"]

    assert control["restart"] is True
    assert control["restart_chk"] == "restart.h5"
    assert control["hamiltonian_h5"] == "hamiltonian_data.h5"


def test_crystal_twist_continuation_forwards_restart_to_optimizer(monkeypatch):
    """The combined twist CLI forwards restart, its source path and the requested proposal count."""
    import jqmc._twist_average as twist_average

    captured = {}

    monkeypatch.setattr(
        twist_average,
        "load_twist_average_input",
        lambda _path: SimpleNamespace(
            n_twist=2,
            k_points=[None, None],
            weights=np.asarray([0.5, 0.5], dtype=np.float64),
            bz_weight_sum=1.0,
        ),
    )

    # The dispatch prints twist 0's Hamiltonian summary before the driver runs.
    monkeypatch.setattr(
        twist_average, "load_twist", lambda _path, _index, **_kwargs: SimpleNamespace(_logger_info=lambda: None)
    )

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(twist_average, "run_twist_average_optimize", fake_optimize)

    vmc = {
        "num_mcmc_steps": 20,
        "num_mcmc_per_measurement": 4,
        "num_opt_steps": 2,
        "wf_dump_freq": 2,
        "num_mcmc_warmup_steps": 0,
        "num_mcmc_bin_blocks": 2,
        "Dt": 1.0,
        "epsilon_AS": 0.0,
        "opt_J1_param": False,
        "opt_J2_param": True,
        "opt_J3_param": False,
        "opt_geminal": False,
        "fix_geminal_rank": 0,
        "opt_J3_basis_exp": False,
        "opt_J3_basis_coeff": False,
        "opt_lambda_basis_exp": False,
        "opt_lambda_basis_coeff": False,
        "opt_JDTN_u_param": False,
        "opt_JDTN_chi_param": False,
        "opt_JDTN_f_param": False,
        "optimizer_kwargs": {"method": "sr", "delta": 0.01},
    }

    _run_twist_average_dispatch(
        job_type="vmc",
        hamiltonian_h5="hamiltonian_data.h5",
        parameters={"vmc": dict(vmc)},
        dict_toml={"vmc": dict(vmc)},
        mcmc_seed=12345,
        number_of_walkers=2,
        mpc=False,
        mpc_cutoff=30.0,
        max_time=3600,
        restart=True,
        restart_chk="restart.h5",
        mpi_comm=object(),
        mpi_rank=0,
    )

    assert captured["checkpoint_dir"] == "restart.h5"
    assert captured["resume"] is True
    assert captured["mcmc_kwargs"].get("num_mcmc_per_measurement") == 4
    # The combined source is required for the complete multi-twist opt-step dump; the
    # driver refuses to write a rank-0-twist-only file without it.
    assert captured["source_hamiltonian_path"] == "hamiltonian_data.h5"


def test_twist_checkpoint_directory_is_discovered_as_the_restart_chk(tmp_path):
    """A twist-averaged run's ``restart.h5`` is a DIRECTORY.

    ``_find_restart_chk`` feeds both the continuation TOML and the
    ``optimized_hamiltonian`` recovery path, and the multi-twist opt-step dumps
    sit next to it as ``*.h5`` files, so the sharded checkpoint directory
    must still win over them.
    """
    (tmp_path / "restart.h5").mkdir()
    (tmp_path / "restart.h5" / "twist_000.h5").touch()
    (tmp_path / "hamiltonian_data.h5").touch()
    (tmp_path / "hamiltonian_data_opt_step_2.h5").touch()

    assert VMC_Workflow()._find_restart_chk(str(tmp_path)) == "restart.h5"


# ---------------------------------------------------------------------------
# Continuation from an EXTERNALLY supplied twist checkpoint
# ---------------------------------------------------------------------------


def _make_twist_checkpoint_dir(root, n_twist: int = 2):
    """Create a ``restart.h5/`` twist checkpoint directory skeleton."""
    chk = root / "restart.h5"
    chk.mkdir()
    (chk / "shared_state.h5").write_bytes(b"")
    for i in range(n_twist):
        (chk / f"twist_{i:03d}.h5").write_bytes(b"")
    return chk


def _run_fixed_steps(workflow, work_dir):
    """Drive ``_launch_fixed_steps`` with the job layer stubbed out."""
    import asyncio

    calls = []

    async def fake_submit_and_wait(input_file, output_file, **kwargs):
        calls.append({"input_file": input_file, **kwargs})
        # The real job leaves the optimizer dump behind; the single-step
        # publication (``handoff_mode="last"``, ``_latest_optimized_hamiltonian``)
        # refuses to continue without it.  These tests pin the continuation
        # contract; the averaged hand-off is covered in test_vmc_averaged_handoff.py.
        (work_dir / "hamiltonian_data_opt_step_300.h5").touch()

    workflow._submit_and_wait = fake_submit_and_wait
    asyncio.run(workflow._launch_fixed_steps(str(work_dir)))
    return calls


def test_external_twist_checkpoint_makes_run1_a_continuation(tmp_path, monkeypatch):
    """``external_restart_chk`` turns production run 1 into a resume.

    Without it the first run always starts from scratch, which on a
    twist-averaged job silently throws away every completed optimisation
    step in the staged checkpoint.
    """
    import h5py

    from jqmc_workflow import vmc_workflow as vmc_mod
    from jqmc_workflow.state import CompletionStatus

    # A non-combined hamiltonian is enough here: the test pins the continuation
    # contract (restart flags + checkpoint upload), not the publication.
    with h5py.File(tmp_path / "hamiltonian_data.h5", "w") as f:
        f.create_dataset("dummy", data=[0])
    _make_twist_checkpoint_dir(tmp_path)

    monkeypatch.setattr(vmc_mod, "validate_completion", lambda *a, **k: (CompletionStatus.OK, ""))

    workflow = VMC_Workflow(
        jobname="vmc-cont",
        handoff_mode="last",
        num_opt_steps=300,
        num_mcmc_steps=40,
        num_mcmc_per_measurement=40,
        external_restart_chk="restart.h5",
    )
    calls = _run_fixed_steps(workflow, tmp_path)

    assert len(calls) == 1
    # The checkpoint directory travels with the job.
    assert calls[0]["extra_from_objects"] == ["restart.h5"]

    with open(tmp_path / calls[0]["input_file"], "rb") as stream:
        toml_data = tomllib.load(stream)
    assert toml_data["control"]["restart"] is True
    assert toml_data["control"]["restart_chk"] == "restart.h5"
    # On restart num_opt_steps counts ADDITIONAL steps on top of the
    # checkpoint's completed i_opt (single-twist and twist routes alike).
    assert toml_data["vmc"]["num_opt_steps"] == 300


def test_without_external_checkpoint_run1_starts_fresh(tmp_path, monkeypatch):
    """Default behaviour is unchanged: run 1 is not a restart."""
    import h5py

    from jqmc_workflow import vmc_workflow as vmc_mod
    from jqmc_workflow.state import CompletionStatus

    with h5py.File(tmp_path / "hamiltonian_data.h5", "w") as f:
        f.create_dataset("dummy", data=[0])

    monkeypatch.setattr(vmc_mod, "validate_completion", lambda *a, **k: (CompletionStatus.OK, ""))

    workflow = VMC_Workflow(
        jobname="vmc-fresh",
        handoff_mode="last",
        num_opt_steps=5,
        num_mcmc_steps=10,
        num_mcmc_per_measurement=40,
    )
    calls = _run_fixed_steps(workflow, tmp_path)

    assert calls[0]["extra_from_objects"] == []
    with open(tmp_path / calls[0]["input_file"], "rb") as stream:
        toml_data = tomllib.load(stream)
    assert toml_data["control"]["restart"] is False


def test_missing_external_checkpoint_fails_before_submission(tmp_path):
    """A configured-but-absent checkpoint must not reach the scheduler."""
    workflow = VMC_Workflow(
        num_opt_steps=300,
        num_mcmc_steps=40,
        num_mcmc_per_measurement=40,
        external_restart_chk="restart.h5",
    )
    with pytest.raises(FileNotFoundError, match="external_restart_chk"):
        workflow._resolve_external_restart_chk(str(tmp_path))
