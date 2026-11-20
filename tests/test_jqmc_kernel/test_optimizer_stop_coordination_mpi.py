"""Every rank of a VMC optimization stops at the same step when the budget is exceeded."""

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

import copy
import faulthandler
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

from mpi4py import MPI
import pytest
import toml

from jqmc import jqmc_cli, jqmc_optimizer
from jqmc._checkpoint import load_rank_checkpoint
from jqmc._precision import is_mixed_precision_enabled
from tests.test_jqmc_kernel.test_cli_restart_values import _write_hamiltonian

pytestmark = [pytest.mark.light, pytest.mark.with_chdir, pytest.mark.mpi_only]

# A rank that leaves the optimization loop alone leaves the others blocked in
# the next collective; the watchdog turns that deadlock into a failed run.
# It writes to the process's real stderr: ``--capture=tee-sys`` (the MPI test
# argv) replaces ``sys.stderr`` with a stream that has no file descriptor.
_DEADLOCK_TIMEOUT_SEC = 600


def _run_vmc_with_rank_clock(root, late, monkeypatch):
    """Run the actual VMC CLI for two optimization steps with ``max_time = 3600``.

    Only the optimization-loop clock is controlled. On a ``late`` rank every
    reading is 10^4 s after the previous one, so that rank alone sees the
    budget exceeded after the first step; on the others it barely advances.
    """
    params = copy.deepcopy(jqmc_cli.cli_parameters["vmc"])
    params.update(
        num_mcmc_steps=8,
        num_mcmc_per_measurement=1,
        num_mcmc_warmup_steps=0,
        num_mcmc_bin_blocks=2,
        num_opt_steps=2,
    )
    for key in params:
        if key.startswith("opt_"):
            params[key] = key == "opt_J2_param"
    params["optimizer_kwargs"].update(method="sr", delta=0.01, use_lm=False)
    payload = {
        "control": {
            "job_type": "vmc",
            "mcmc_seed": 12345,
            "number_of_walkers": 2,
            "hamiltonian_h5": str(root / "hamiltonian.h5"),
            "restart_chk": str(root / "restart.h5"),
            "verbosity": "low",
            "max_time": 3600,
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        "vmc": params,
    }
    input_file = root / "input.toml"
    if MPI.COMM_WORLD.Get_rank() == 0:
        input_file.write_text(toml.dumps(payload), encoding="utf-8")
    MPI.COMM_WORLD.Barrier()
    ticks = 0

    def clock():
        nonlocal ticks
        ticks += 1
        return ticks * (1.0e4 if late else 1e-6)

    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with monkeypatch.context() as patch:
            patch.chdir(root)
            patch.setattr(sys, "argv", ["jqmc", str(input_file)])
            patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
            patch.setattr(jqmc_optimizer, "time", SimpleNamespace(perf_counter=clock))
            jqmc_cli._cli()
    finally:
        for handler in logger.handlers:
            if handler not in handlers and isinstance(handler, logging.FileHandler):
                handler.close()
        logger.handlers, logger.level = handlers, level
    return load_rank_checkpoint(str(root / "restart.h5"), rank=MPI.COMM_WORLD.Get_rank())


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
@pytest.mark.parametrize("late_rank", [0, 1])
def test_vmc_budget_stop_is_decided_once_for_all_ranks(tmp_path, monkeypatch, late_rank):
    """The global root decides the max_time stop and every rank follows it.

    Same committed H2 AE input, Pade J2 only, two walkers per rank and eight
    measurements per SR step as the serial budget test. The rank clocks are
    made to disagree after the first step. If the root sees the budget
    exceeded (``late_rank = 0``), every rank stops after step 1; if only the
    other rank does (``late_rank = 1``), every rank runs both steps. If each rank
    compared its own clock, the ranks would leave the loop at different steps
    and the run would deadlock in the next collective.
    """
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2, "launch this MPI-only contract with mpirun -np 2"
    rank = comm.Get_rank()
    root = Path(comm.bcast(str(tmp_path) if rank == 0 else None, root=0))
    if rank == 0:
        _write_hamiltonian(root)
    comm.Barrier()
    faulthandler.dump_traceback_later(_DEADLOCK_TIMEOUT_SEC, exit=True, file=sys.__stderr__)
    try:
        checkpoint = _run_vmc_with_rank_clock(root, rank == late_rank, monkeypatch)
    finally:
        faulthandler.cancel_dump_traceback_later()
    expected_steps = 1 if late_rank == 0 else 2
    assert comm.allgather(checkpoint["driver_config"]["i_opt"]) == [expected_steps, expected_steps]
    assert (root / f"hamiltonian_data_opt_step_{expected_steps}.h5").is_file()


# ---------------------------------------------------------------------------
# Twist-averaged optimizer (issue #354): the budget and the external stop flag
# end the whole chunked run on every rank at the same step.
# ---------------------------------------------------------------------------

_TWIST_REQUESTED_STEPS = 3


def _run_two_twist_optimize(monkeypatch, *, max_time):
    """Two-twist ``run_twist_average_optimize`` on the real two-rank world, one-step chunks.

    The two committed H2 AE inputs (Cartesian and spherical basis, Pade J1/J2)
    of the twist update-value tests are the two twists, one per rank. Returns
    this rank's chunks as ``(requested, completed, stop reason)``.
    """
    from jqmc._twist_average import run_twist_average_optimize
    from jqmc.jqmc_mcmc import MCMC
    from tests.test_jqmc_kernel import test_twist_optimize_update_value as twist_inputs

    real_run_optimize = MCMC.run_optimize
    chunks = []

    def spy_run_optimize(mcmc, **kwargs):
        before = mcmc.i_opt
        reason = real_run_optimize(mcmc, **kwargs)
        chunks.append((kwargs["num_opt_steps"], mcmc.i_opt - before, reason))
        return reason

    monkeypatch.setattr(MCMC, "run_optimize", spy_run_optimize)
    run_twist_average_optimize(
        [twist_inputs._build_hamiltonian(stem) for stem in twist_inputs._TWIST_FILES],
        list(twist_inputs._RAW_WEIGHTS),
        mcmc_seed=twist_inputs._SEED,
        num_walkers=2,
        world_comm=MPI.COMM_WORLD,
        optimize_kwargs=dict(
            num_mcmc_steps=8,
            num_opt_steps=_TWIST_REQUESTED_STEPS,
            wf_dump_freq=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=2,
            max_time=max_time,
            opt_J1_param=True,
            opt_J2_param=True,
            opt_J3_param=False,
            opt_geminal=False,
            optimizer_kwargs={"method": "sr", "delta": 0.01, "epsilon": 1.0e-3},
        ),
    )
    return chunks


def _assert_same_chunks_on_every_rank(comm, chunks, expected_steps, expected_reason):
    gathered = comm.allgather(chunks)
    assert gathered[0] == gathered[1], f"ranks left the chunked loop differently: {gathered}"
    assert sum(done for _, done, _ in chunks) == expected_steps, f"chunks (requested, done, reason): {chunks}"
    assert chunks[-1][2] == expected_reason, chunks
    assert all(reason is None for _, _, reason in chunks[:-1]), chunks


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
@pytest.mark.parametrize("late_rank", [0, 1])
@pytest.mark.parametrize("clock", ["optimizer", "driver"])
def test_twist_budget_stop_is_decided_once_for_all_ranks_and_chunks(tmp_path, monkeypatch, clock, late_rank):
    """``max_time`` bounds the whole two-twist chunked run, decided on the world root.

    Two twists on two ranks, three requested SR steps in one-step chunks
    (``wf_dump_freq = 1``, the CLI default), ``max_time = 3600``. Only one
    clock is controlled, on one rank: every reading on the ``late`` rank is
    10^4 s after the previous one, on the other it barely advances.
    ``clock = optimizer`` controls the per-chunk clock of ``run_optimize``,
    ``clock = driver`` the twist driver's own budget clock. If the world root
    sees the budget exceeded (``late_rank = 0``), every rank stops after
    step 1; if only the other rank does (``late_rank = 1``), every rank runs
    all three steps. Restarting the budget with a fresh clock per chunk
    would let a late root run all three steps; a per-rank decision would make the ranks leave the loop at different steps and
    deadlock in the next collective (the watchdog turns that into a failure).
    """
    from jqmc import _twist_average as twist_average_module

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2, "launch this MPI-only contract with mpirun -np 2"
    late = comm.Get_rank() == late_rank
    ticks = 0

    def rank_clock():
        nonlocal ticks
        ticks += 1
        return ticks * (1.0e4 if late else 1e-6)

    module = jqmc_optimizer if clock == "optimizer" else twist_average_module
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=rank_clock), raising=clock == "optimizer")
    faulthandler.dump_traceback_later(_DEADLOCK_TIMEOUT_SEC, exit=True, file=sys.__stderr__)
    try:
        chunks = _run_two_twist_optimize(monkeypatch, max_time=3600)
    finally:
        faulthandler.cancel_dump_traceback_later()
    if late_rank == 0:
        _assert_same_chunks_on_every_rank(comm, chunks, 1, "max_time")
    else:
        _assert_same_chunks_on_every_rank(comm, chunks, _TWIST_REQUESTED_STEPS, None)


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_twist_stop_request_ends_every_rank_after_the_same_step(tmp_path, monkeypatch):
    """``stop = true`` in ``external_control_opt.toml`` ends the two-twist run on both ranks.

    Same two-twist, two-rank, one-step-chunk run as the budget test, with the
    default budget. The world root's control file is rewritten to
    ``stop = true`` as soon as the optimizer creates it (the real file and
    reader; see ``tests/_external_stop_request.py``). Both ranks must stop
    after step 1 and not start another chunk; the next chunk must not
    recreate the file with ``stop = false``.
    """
    from tests._external_stop_request import request_external_stop

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2, "launch this MPI-only contract with mpirun -np 2"
    requests = request_external_stop(monkeypatch, jqmc_optimizer, "external_control_opt.toml")
    faulthandler.dump_traceback_later(_DEADLOCK_TIMEOUT_SEC, exit=True, file=sys.__stderr__)
    try:
        chunks = _run_two_twist_optimize(monkeypatch, max_time=86400)
    finally:
        faulthandler.cancel_dump_traceback_later()
    _assert_same_chunks_on_every_rank(comm, chunks, 1, "stop_flag")
    assert requests == {"created": 1, "requested": 1}


@pytest.mark.parametrize("content", ["true", "false", "missing", "malformed", "missing-key"])
def test_external_control_lifecycle_and_root_read_are_shared(tmp_path, content):
    """Only world root creates/removes and reads the TOML; real MPI broadcasts its valid or EAFP result.

    Each rank has a distinct file. The non-root file requests the opposite
    decision, so local reads or a missing broadcast cannot pass by coincidence.
    Missing, partially written and missing-key root files mean no stop request.
    """
    from jqmc import _driver_helpers as helpers

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2, "launch this MPI-only contract with mpirun -np 2"
    rank = comm.Get_rank()
    local = tmp_path / f"rank-{rank}"
    local.mkdir()
    path = local / "control.toml"
    path.write_text("[external_control]\nstop = true\n")
    helpers._init_external_control_file(str(path), comm)
    assert comm.allgather(toml.loads(path.read_text())["external_control"]["stop"]) == [False, True]
    expected = content == "true"
    if rank == 0:
        if content == "missing":
            path.unlink()
        else:
            payload = {
                "true": "[external_control]\nstop = true\n",
                "false": "[external_control]\nstop = false\n",
                "malformed": "[external_control\n",
                "missing-key": "[external_control]\nother = true\n",
            }
            path.write_text(payload[content])
    else:
        path.write_text(toml.dumps({"external_control": {"stop": not expected}}))
    comm.Barrier()
    assert comm.allgather(helpers._read_external_stop_flag(str(path), comm)) == [expected, expected]
    helpers._remove_external_control_file(str(path), comm)
    helpers._remove_external_control_file(str(path), comm)
    assert comm.allgather(path.exists()) == [False, True], "only the world root owns control-file removal"


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_non_twist_mcmc_stop_checkpoint_continues_on_every_rank(tmp_path, monkeypatch):
    """A real root-written stop ends both H2 MCMC ranks after one step, checkpoints and resumes to eight.

    The committed H2 AE Cartesian-MO/J2 input uses two walkers per rank and
    seed 12345. Real shared read/broadcast, per-rank checkpoint merge/load and
    the resumed sampler execute; no MPI method or production stop call is mocked.
    """
    import numpy as np
    from jqmc import jqmc_mcmc
    from jqmc._checkpoint import merge_rank_checkpoints
    from jqmc.hamiltonians import Hamiltonian_data
    from tests._external_stop_request import request_external_stop

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2, "launch this MPI-only contract with mpirun -np 2"
    rank = comm.Get_rank()
    root = Path(comm.bcast(str(tmp_path) if rank == 0 else None, root=0))
    if rank == 0:
        _write_hamiltonian(root)
    comm.Barrier()
    monkeypatch.chdir(root)
    ham = Hamiltonian_data.load_from_hdf5(str(root / "hamiltonian.h5"))
    driver = jqmc_mcmc.MCMC(
        hamiltonian_data=ham,
        num_walkers=2,
        mcmc_seed=12345,
        num_mcmc_per_measurement=1,
        comput_position_deriv=False,
        mpi_comm_in=comm,
    )
    requests = request_external_stop(monkeypatch, jqmc_mcmc, "external_control_mcmc.toml")
    faulthandler.dump_traceback_later(_DEADLOCK_TIMEOUT_SEC, exit=True, file=sys.__stderr__)
    try:
        driver.run(num_mcmc_steps=8)
        assert comm.allgather(driver.mcmc_counter) == [1, 1], "external stop must end every rank at the first step"
        pattern = str(root / "stop-rank-{rank}.h5")
        driver.save_to_hdf5(pattern.format(rank=rank))
        comm.Barrier()
        merged = str(root / "restart.h5")
        if rank == 0:
            merge_rank_checkpoints(merged, mpi_size=2, driver_type="MCMC", hamiltonian_data=ham, tmp_pattern=pattern)
        comm.Barrier()
        restored = jqmc_mcmc.MCMC.load_from_hdf5(merged, mpi_comm_in=comm)
        assert restored.mcmc_counter == 1
        np.testing.assert_array_equal(restored.e_L, driver.e_L)
        restored.run(num_mcmc_steps=7)
        assert comm.allgather(restored.mcmc_counter) == [8, 8]
        assert requests == {"created": 2, "requested": 1}
        assert np.isfinite(np.asarray(restored.e_L)).all()
        assert not (root / "external_control_mcmc.toml").exists()
    finally:
        faulthandler.cancel_dump_traceback_later()
