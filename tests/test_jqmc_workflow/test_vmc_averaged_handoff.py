"""``VMC_Workflow`` hands the average of the last optimization steps downstream (issue #767).

The optimization dumps are real Hamiltonian files: the committed H2 TREXIO
fixture converted by ``jqmc-tool trexio convert-to`` (J1/J2/J3), with a
seeded perturbation of the J2 parameter and of the J3 matrix (symmetric in
its coupling block, as the loader requires) per step, written with h5py.
The averaging is the real ``jqmc-tool vmc average-wf`` subprocess that the
workflow launches. Expected averages are ``np.mean`` over the h5py-read
leaves of exactly the dumps that should be selected; the workflow's own
selection helpers never build an expectation.
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
import shutil
import tomllib
from pathlib import Path

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from jqmc._precision import get_tolerance_min
from jqmc.jqmc_tool import trexio_app
from jqmc_workflow import Container, FileFrom, Launcher, ValueFrom, Workflow
from jqmc_workflow import vmc_workflow as vmc_mod
from jqmc_workflow.state import CompletionStatus, WorkflowStatus
from jqmc_workflow.vmc_workflow import VMC_Workflow

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_TREXIO = Path(__file__).resolve().parents[1] / "trexio_example_files" / "H2_ecp_ccpvtz_cart.h5"
_J2 = "/wavefunction_data/jastrow_data/jastrow_two_body_data/jastrow_2b_param"
_J3 = "/wavefunction_data/jastrow_data/jastrow_three_body_data/j_matrix"
_POSITIONS = "/structure_data/positions"
_AVERAGED = "hamiltonian_data_opt_averaged.h5"


@pytest.fixture(scope="module")
def base_hamiltonian(tmp_path_factory):
    """Real converted Hamiltonian (committed TREXIO fixture, J1/J2/J3)."""
    path = tmp_path_factory.mktemp("vmc_handoff_base") / "hamiltonian_data.h5"
    result = CliRunner().invoke(
        trexio_app, ["convert-to", str(_TREXIO), "-o", str(path), "-j1", "1.0", "-j2", "1.0", "-j3", "ao-small"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    return path


def _write_dumps(base: Path, directory: Path, steps, *, seed: int) -> None:
    """One dump per step: J2 and the J3 matrix take a seeded random step from the base values."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for step in steps:
        path = directory / f"hamiltonian_data_opt_step_{step}.h5"
        shutil.copyfile(base, path)
        with h5py.File(path, "r+") as h5:
            delta = 1.0e-3 * rng.standard_normal(h5[_J3].shape)
            delta[:, :-1] = 0.5 * (delta[:, :-1] + delta[:, :-1].T)
            h5[_J3][...] = h5[_J3][()] + delta
            h5[_J2][...] = h5[_J2][()] + 1.0e-2 * rng.standard_normal()


def _numpy_mean(directory: Path, steps, leaf: str) -> np.ndarray:
    return np.mean([h5py.File(directory / f"hamiltonian_data_opt_step_{step}.h5", "r")[leaf][()] for step in steps], axis=0)


def _assert_average_of(directory: Path, name: str, steps) -> None:
    """The published file holds the NumPy mean of exactly these steps and names them as its sources."""
    atol, rtol = get_tolerance_min(("jastrow_eval",), "strict")
    with h5py.File(directory / name, "r") as h5:
        for leaf in (_J2, _J3):
            np.testing.assert_allclose(h5[leaf][()], _numpy_mean(directory, steps, leaf), atol=atol, rtol=rtol, err_msg=leaf)
        assert list(h5.attrs["opt_step_average_steps"]) == list(steps)
        assert [str(s) for s in h5.attrs["opt_step_average_sources"]] == [f"hamiltonian_data_opt_step_{s}.h5" for s in steps]


def _run_fixed(workflow: VMC_Workflow, work_dir: Path, base: Path, dumps_per_run, *, seed: int = 767, monkeypatch):
    """Drive ``_launch_fixed_steps``; each stubbed job leaves its optimizer dumps behind, as a real job does."""
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(vmc_mod, "validate_completion", lambda *a, **k: (CompletionStatus.OK, ""))
    runs = iter(dumps_per_run)

    async def fake_submit_and_wait(input_file, output_file, **kwargs):
        _write_dumps(base, work_dir, next(runs), seed=seed)

    workflow._submit_and_wait = fake_submit_and_wait
    return asyncio.run(workflow._launch_fixed_steps(str(work_dir)))


def test_default_handoff_averages_the_last_20_dumps_and_reaches_the_sampler(tmp_path, monkeypatch, base_hamiltonian):
    """Default: the last 20 of 25 dumps are averaged, published, recorded and staged as the sampler's input."""
    monkeypatch.chdir(tmp_path)
    workflow = VMC_Workflow(num_opt_steps=25, num_mcmc_steps=10, num_mcmc_per_measurement=4)
    assert (workflow.handoff_mode, workflow.handoff_average_steps, workflow.handoff_allow_fewer) == ("average", 20, False)
    vmc_dir = tmp_path / "vmc"
    status, files, values = _run_fixed(workflow, vmc_dir, base_hamiltonian, [range(1, 26)], monkeypatch=monkeypatch)

    assert status == WorkflowStatus.COMPLETED
    # Every step is dumped: the default dump frequency is the one the average needs.
    toml_data = tomllib.loads(next(vmc_dir.glob("input_*.toml")).read_text())
    assert toml_data["vmc"]["wf_dump_freq"] == 1
    expected_steps = list(range(6, 26))
    assert values["optimized_hamiltonian"] == _AVERAGED and _AVERAGED in files
    assert values["optimized_hamiltonian_handoff"] == "average"
    assert values["optimized_hamiltonian_steps"] == expected_steps
    assert values["optimized_hamiltonian_sources"] == [f"hamiltonian_data_opt_step_{s}.h5" for s in expected_steps]
    _assert_average_of(vmc_dir, _AVERAGED, expected_steps)
    # The window matters: the mean of all 25 dumps is a different wavefunction.
    atol, rtol = get_tolerance_min(("jastrow_eval",), "strict")
    with h5py.File(vmc_dir / _AVERAGED, "r") as h5:
        assert not np.allclose(h5[_J3][()], _numpy_mean(vmc_dir, range(1, 26), _J3), atol=atol, rtol=rtol)

    upstream = Container(label="vmc", dirname="vmc", workflow=Workflow())
    upstream.output_values.update(values)
    downstream = Container(
        label="sample",
        dirname="sample",
        workflow=Workflow(),
        input_files=[FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian"))],
        rename_input_files=["hamiltonian_data.h5"],
    )
    launcher = Launcher(workflows=[upstream, downstream], log_level="WARNING", draw_graph=False)
    launcher._resolve_variables(downstream)
    downstream._prepare()
    consumed = Path(downstream.project_dir) / "hamiltonian_data.h5"
    assert consumed.read_bytes() == (vmc_dir / _AVERAGED).read_bytes()


def test_explicit_handoff_average_steps_selects_that_window(tmp_path, monkeypatch, base_hamiltonian):
    """``handoff_average_steps=5`` averages exactly the last 5 steps, across a continuation's numbering."""
    workflow = VMC_Workflow(
        num_opt_steps=6, num_mcmc_steps=10, num_mcmc_per_measurement=4, max_continuation=2, handoff_average_steps=5
    )
    work_dir = tmp_path / "vmc"
    work_dir.mkdir()
    (work_dir / "restart.h5").write_bytes(b"")
    _, _, values = _run_fixed(workflow, work_dir, base_hamiltonian, [range(1, 7), range(7, 13)], monkeypatch=monkeypatch)
    assert values["optimized_hamiltonian_steps"] == [8, 9, 10, 11, 12]
    _assert_average_of(work_dir, _AVERAGED, [8, 9, 10, 11, 12])


def test_single_step_handoff_publishes_the_last_dump(tmp_path, monkeypatch, base_hamiltonian):
    """``handoff_mode='last'`` keeps the previous hand-off: the newest dump itself, any dump frequency."""
    workflow = VMC_Workflow(
        num_opt_steps=12, num_mcmc_steps=10, num_mcmc_per_measurement=4, handoff_mode="last", wf_dump_freq=4
    )
    work_dir = tmp_path / "vmc"
    _, files, values = _run_fixed(workflow, work_dir, base_hamiltonian, [(4, 8, 12)], monkeypatch=monkeypatch)
    assert values["optimized_hamiltonian"] == "hamiltonian_data_opt_step_12.h5"
    assert values["optimized_hamiltonian_handoff"] == "last"
    assert values["optimized_hamiltonian_sources"] == ["hamiltonian_data_opt_step_12.h5"]
    assert values["optimized_hamiltonian_steps"] == [12]
    assert not (work_dir / _AVERAGED).exists() and _AVERAGED not in files


def test_too_few_dumps_are_refused_unless_explicitly_allowed(tmp_path, monkeypatch, base_hamiltonian):
    """12 dumps for a 20-step average: refused, nothing published; ``handoff_allow_fewer`` averages all 12."""
    _write_dumps(base_hamiltonian, tmp_path, range(1, 13), seed=11)
    with pytest.raises(
        RuntimeError, match=r"only 12 hamiltonian_data_opt_step_\*\.h5 dump\(s\) \(steps 1-12\).*handoff_average_steps=20"
    ):
        VMC_Workflow(num_opt_steps=20)._publish_optimized_hamiltonian(str(tmp_path))
    assert not (tmp_path / _AVERAGED).exists()

    workflow = VMC_Workflow(num_opt_steps=20, handoff_allow_fewer=True)
    assert workflow._publish_optimized_hamiltonian(str(tmp_path)) == _AVERAGED
    assert workflow.output_values["optimized_hamiltonian_steps"] == list(range(1, 13))
    _assert_average_of(tmp_path, _AVERAGED, range(1, 13))


@pytest.mark.parametrize("allow_fewer", [False, True])
def test_dumps_that_skip_steps_are_refused(tmp_path, base_hamiltonian, allow_fewer):
    """Dumps of every 2nd step (``wf_dump_freq = 2`` or deleted dumps) never become a 20-step average."""
    _write_dumps(base_hamiltonian, tmp_path, range(2, 42, 2), seed=12)
    workflow = VMC_Workflow(num_opt_steps=40, handoff_allow_fewer=allow_fewer)
    with pytest.raises(RuntimeError, match="not consecutive optimization steps"):
        workflow._publish_optimized_hamiltonian(str(tmp_path))
    assert not (tmp_path / _AVERAGED).exists()


_TOOL_REFUSALS = {
    "non-finite": r"holds a non-finite value in hamiltonian_data_opt_step_9\.h5",
    "structure": r"atomic structure",
}


@pytest.mark.parametrize("corruption", sorted(_TOOL_REFUSALS))
def test_average_wf_refusal_is_raised_with_the_tool_message(tmp_path, base_hamiltonian, corruption):
    """A refusal by ``jqmc-tool vmc average-wf`` stops the hand-off with its message; no stale average survives."""
    message = _TOOL_REFUSALS[corruption]
    _write_dumps(base_hamiltonian, tmp_path, range(1, 11), seed=13)
    with h5py.File(tmp_path / "hamiltonian_data_opt_step_9.h5", "r+") as h5:
        if corruption == "non-finite":
            h5[_J2][...] = np.nan
        else:
            h5[_POSITIONS][0, 0] += 0.1
    (tmp_path / _AVERAGED).write_bytes(b"stale average of an earlier run")
    workflow = VMC_Workflow(num_opt_steps=10, handoff_average_steps=10)
    with pytest.raises(
        RuntimeError, match=rf"jqmc-tool vmc average-wf refused the last 10 dumps.*exit status 1: Error: .*{message}"
    ):
        workflow._publish_optimized_hamiltonian(str(tmp_path))
    assert not (tmp_path / _AVERAGED).exists()
    assert "optimized_hamiltonian" not in workflow.output_values


def test_failed_run_publishes_no_average(tmp_path, monkeypatch, base_hamiltonian):
    """A run that ``validate_completion`` failed does not average its (possibly diverged) steps."""
    workflow = VMC_Workflow(num_opt_steps=20, num_mcmc_steps=10, num_mcmc_per_measurement=4)
    work_dir = tmp_path / "vmc"
    work_dir.mkdir()
    monkeypatch.setattr(vmc_mod, "validate_completion", lambda *a, **k: (CompletionStatus.FAILED, "diverged"))

    async def fake_submit_and_wait(input_file, output_file, **kwargs):
        _write_dumps(base_hamiltonian, work_dir, range(1, 21), seed=14)

    workflow._submit_and_wait = fake_submit_and_wait
    status, _, values = asyncio.run(workflow._launch_fixed_steps(str(work_dir)))
    assert status == WorkflowStatus.FAILED and values["error"] == "diverged"
    assert "optimized_hamiltonian" not in values and not (work_dir / _AVERAGED).exists()


@pytest.mark.parametrize("handoff_mode,expected_runs", [("average", 2), ("last", 1)])
def test_auto_mode_defers_early_stop_until_the_average_has_its_dumps(
    tmp_path, monkeypatch, base_hamiltonian, handoff_mode, expected_runs
):
    """Converged after 15 steps: the averaged hand-off runs the next continuation to reach 20 dumps."""
    work_dir = tmp_path / "vmc"
    work_dir.mkdir()
    (work_dir / "restart.h5").write_bytes(b"")
    monkeypatch.setattr(vmc_mod, "validate_completion", lambda *a, **k: (CompletionStatus.OK, ""))
    monkeypatch.setattr(vmc_mod, "get_estimation", lambda _wd: {"estimated_mcmc_steps": 10})
    workflow = VMC_Workflow(
        num_opt_steps=15, num_mcmc_per_measurement=4, max_continuation=2, target_snr=4.0, handoff_mode=handoff_mode
    )
    monkeypatch.setattr(workflow, "_check_convergence", lambda *a, **k: (True, True, True))
    runs = iter([range(1, 16), range(16, 31)])
    submitted = []

    async def fake_submit_and_wait(input_file, output_file, **kwargs):
        submitted.append(input_file)
        _write_dumps(base_hamiltonian, work_dir, next(runs), seed=15)

    workflow._submit_and_wait = fake_submit_and_wait
    status, _, values = asyncio.run(workflow._launch_auto(str(work_dir)))
    assert status == WorkflowStatus.COMPLETED
    assert len(submitted) == expected_runs
    if handoff_mode == "average":
        assert values["optimized_hamiltonian_steps"] == list(range(11, 31))
        _assert_average_of(work_dir, _AVERAGED, range(11, 31))
    else:
        assert values["optimized_hamiltonian"] == "hamiltonian_data_opt_step_15.h5"


def test_handoff_settings_are_checked_before_any_job():
    """Settings that make the requested average impossible fail at construction."""
    with pytest.raises(ValueError, match=r"wf_dump_freq=10 dumps only every 10-th step"):
        VMC_Workflow(num_opt_steps=100, wf_dump_freq=10)
    with pytest.raises(ValueError, match=r"num_opt_steps=10 x max_continuation=1 gives at most 10"):
        VMC_Workflow(num_opt_steps=10)
    with pytest.raises(ValueError, match="handoff_mode must be one of"):
        VMC_Workflow(handoff_mode="mean")
    for steps in (0, -3, 2.5, True):
        with pytest.raises(ValueError, match="handoff_average_steps must be a positive integer"):
            VMC_Workflow(handoff_average_steps=steps)
    # Accepted: every step dumped, enough steps over the continuations, an explicit opt-out, or a staged checkpoint.
    VMC_Workflow(num_opt_steps=100, wf_dump_freq=1)
    VMC_Workflow(num_opt_steps=10, max_continuation=2)
    VMC_Workflow(num_opt_steps=10, handoff_allow_fewer=True)
    VMC_Workflow(num_opt_steps=10, handoff_average_steps=10)
    VMC_Workflow(num_opt_steps=10, external_restart_chk="restart.h5")
    VMC_Workflow(num_opt_steps=5, wf_dump_freq=10, handoff_mode="last")
    summary = VMC_Workflow(num_opt_steps=30, handoff_average_steps=25).configure()
    assert (summary["handoff_mode"], summary["handoff_average_steps"], summary["handoff_allow_fewer"]) == ("average", 25, False)
