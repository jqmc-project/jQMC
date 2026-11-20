"""A COMPLETED LRDMC_Ext_Workflow carries the extrapolation over its whole alat_list.

The per-alat LRDMC children are represented by completed child project
directories (``workflow_state.toml`` plus a small GFMC checkpoint), so each
child ``Container`` takes its real "Already completed" short-circuit. The
parent then runs its real collection code and the real
``jqmc-tool lrdmc extrapolate-energy`` subprocess. The per-alat histories have
exactly known bin means and jackknife errors, so the control is checked against
a closed-form Lagrange extrapolation, independent of jqmc-tool's projector.
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

import os
import shutil
import sys

import h5py
import numpy as np
import pytest

from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_ZERO
from jqmc_workflow.lrdmc_ext_workflow import LRDMC_Ext_Workflow
from jqmc_workflow.state import WorkflowStatus, create_state, read_state, update_status
from jqmc_workflow.workflow import Container

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]

_BIN_BLOCKS, _WARMUP, _COLLECT = 10, 25, 5
_BIN_STEPS = 10
# Fixed-step children (num_gfmc_projections set) never ask to resume.
_FIXED_STEP = {
    "num_gfmc_bin_blocks": _BIN_BLOCKS,
    "num_gfmc_warmup_steps": _WARMUP,
    "num_gfmc_collect_steps": _COLLECT,
    "num_projection_per_measurement": 40,
    "num_gfmc_projections": 100,
    "number_of_walkers": 1,
}
_FLUCTUATION = 0.125
# E(a) = E0 + c2 a^2 + c4 a^4 lies exactly on the order-2 polynomial in a^2.
_E0, _C2, _C4 = -1.25, 0.4, -0.9


def _energy(alat):
    return _E0 + _C2 * alat**2 + _C4 * alat**4


def _sigma():
    """Jackknife error of ten equal unit-weight bins whose means are E + (j - 4.5) * s."""
    return _FLUCTUATION * np.std(np.arange(_BIN_BLOCKS), ddof=1) / np.sqrt(_BIN_BLOCKS)


def _write_checkpoint(path, alat, num_steps=None):
    """GFMC checkpoint whose first collect+warmup samples are wrong and must be discarded."""
    centered = np.repeat(np.arange(_BIN_BLOCKS, dtype=np.float64) - 4.5, _BIN_STEPS) * _FLUCTUATION
    history = np.concatenate((np.full(_COLLECT + _WARMUP, 999.0), _energy(alat) + centered))[:, None]
    if num_steps is not None:
        history = history[:num_steps]
    with h5py.File(path, "w") as f:
        meta = f.create_group("_meta")
        meta.attrs["driver_type"] = "GFMC_n"
        meta.attrs["mpi_size"] = 1
        rank = f.create_group("rank_0")
        config = rank.create_group("driver_config")
        config.attrs["alat"] = alat
        config.attrs["num_gfmc_collect_steps"] = _COLLECT
        observables = rank.create_group("observables")
        observables.create_dataset("e_L", data=history)
        observables.create_dataset("w_L", data=np.ones_like(history))


def _write_completed_child(parent_dir, alat, *, checkpoint=True, num_steps=None):
    child = parent_dir / f"lrdmc_alat_{alat:.3f}"
    child.mkdir(parents=True)
    create_state(str(child), label=f"lrdmc-a{alat:.3f}", workflow_type="LRDMC_Workflow", status="pending")
    update_status(
        str(child),
        WorkflowStatus.COMPLETED,
        result_alat=alat,
        result_energy=_energy(alat),
        result_energy_error=float(_sigma()),
        result_restart_chk="restart.h5",
        result_num_projection_per_measurement=40,
    )
    if checkpoint:
        _write_checkpoint(child / "restart.h5", alat, num_steps=num_steps)


def _lagrange_at_zero(alats):
    """Closed-form weights L_i(0) of the polynomial through (a_i^2, E_i), i.e. E0 = sum_i L_i(0) E_i."""
    x = np.asarray(alats) ** 2
    return np.array([np.prod([-x[j] / (x[i] - x[j]) for j in range(len(x)) if j != i]) for i in range(len(x))])


def _launch(tmp_path, workflow):
    container = Container(label="ext", dirname="ext", input_files=["hamiltonian_data.h5"], workflow=workflow)
    status, _files, values = container.launch()
    return status, values, read_state(str(tmp_path / "ext"))


def _assert_failed(status, values, state, *fragments):
    assert status == WorkflowStatus.FAILED, values
    assert "extrapolated_energy" not in values
    assert state["workflow"]["status"] == "failed"
    assert "extrapolated_energy" not in state.get("result", {})
    message = state.get("error", {}).get("message", "")
    for fragment in fragments:
        assert fragment in message, message


@pytest.mark.parametrize("case", ["control", "order-exceeds-spacings", "tool-rejects", "tool-missing", "missing-checkpoint"])
def test_completed_lrdmc_ext_carries_extrapolation_over_every_alat(tmp_path, monkeypatch, case):
    """COMPLETED implies extrapolated_energy from every alat; otherwise FAILED with the cause."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hamiltonian_data.h5").write_bytes(b"placeholder: children are already completed")
    parent = tmp_path / "ext"
    # The real console script installed next to this interpreter; no stand-in.
    monkeypatch.setenv("PATH", os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    assert shutil.which("jqmc-tool") is not None

    if case == "order-exceeds-spacings":
        alats = [0.5, 0.4]
        for alat in alats:
            _write_completed_child(parent, alat)
        # Rejected before any child is created or submitted ...
        with pytest.raises(ValueError, match="polynomial_order 2 needs at least 3 distinct lattice spacings"):
            LRDMC_Ext_Workflow(alat_list=alats, polynomial_order=2, **_FIXED_STEP)
        with pytest.raises(ValueError, match="duplicate lattice spacings"):
            LRDMC_Ext_Workflow(alat_list=[0.5, 0.4, 0.4], polynomial_order=1, **_FIXED_STEP)
        # ... and again at run time when the order arrives later (Launcher ValueFrom resolution uses setattr).
        workflow = LRDMC_Ext_Workflow(alat_list=alats, polynomial_order=1, **_FIXED_STEP)
        workflow.polynomial_order = 2
        _assert_failed(*_launch(tmp_path, workflow), "polynomial_order 2 needs at least 3 distinct lattice spacings")
        return

    alats = [0.5, 0.4, 0.3]
    order = 1 if case == "missing-checkpoint" else 2
    for alat in alats:
        _write_completed_child(
            parent,
            alat,
            checkpoint=not (case == "missing-checkpoint" and alat == 0.3),
            # 35 samples leave fewer than one per bin after collect+warmup: jqmc-tool exits 2.
            num_steps=35 if case == "tool-rejects" and alat == 0.4 else None,
        )
    if case == "tool-missing":
        empty_bin = tmp_path / "empty_bin"
        empty_bin.mkdir()
        monkeypatch.setenv("PATH", str(empty_bin))
    workflow = LRDMC_Ext_Workflow(alat_list=alats, polynomial_order=order, **_FIXED_STEP)
    status, values, state = _launch(tmp_path, workflow)

    if case == "control":
        assert status == WorkflowStatus.COMPLETED, values
        weights = _lagrange_at_zero(alats)
        reference = [weights @ np.array([_energy(a) for a in alats]), _sigma() * np.sqrt(np.sum(weights**2))]
        np.testing.assert_allclose(
            [values["extrapolated_energy"], values["extrapolated_energy_error"]],
            reference,
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        np.testing.assert_allclose(reference[0], _E0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        assert state["workflow"]["status"] == "completed"
        assert state["result"]["extrapolated_energy"] == values["extrapolated_energy"]
        assert [r["alat"] for r in values["per_alat_results"]] == sorted(alats, reverse=True)
    elif case == "tool-rejects":
        _assert_failed(status, values, state, "extrapolate-energy", "too few measurements")
    elif case == "tool-missing":
        _assert_failed(status, values, state, "jqmc-tool", "not found")
    else:
        _assert_failed(status, values, state, "alat=0.300", "restart.h5")
