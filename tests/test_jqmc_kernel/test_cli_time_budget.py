"""Deterministic CLI time-budget interruption and numerical checkpoint continuation."""

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
import logging
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import toml

from jqmc import jqmc_cli, jqmc_gfmc, jqmc_mcmc, jqmc_optimizer, output_parser
from jqmc._checkpoint import load_hamiltonian_from_checkpoint, load_rank_checkpoint
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import GFMC_MIN_BIN_BLOCKS, GFMC_MIN_COLLECT_STEPS, GFMC_MIN_STEPS, GFMC_MIN_WARMUP_STEPS
from tests.test_jqmc_kernel.test_cli_restart_values import _write_hamiltonian

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]


def _run_with_controlled_clock(root, steps, monkeypatch, *, restart=False, stop_after=None):
    """Change only the sampler's clock; execute actual updates, CLI I/O and checkpoints."""
    root.mkdir(exist_ok=True)
    params = copy.deepcopy(jqmc_cli.cli_parameters["mcmc"])
    params.update(
        num_mcmc_steps=steps,
        num_mcmc_per_measurement=1,
        num_mcmc_warmup_steps=0,
        num_mcmc_bin_blocks=2,
    )
    payload = {
        "control": {
            "job_type": "mcmc",
            "mcmc_seed": 12345,
            "number_of_walkers": 2,
            "hamiltonian_h5": str(root.parent / "hamiltonian.h5"),
            "restart": restart,
            "restart_chk": str(root / "restart.h5"),
            "verbosity": "low",
            "max_time": 50,
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        "mcmc": params,
    }
    input_file = root / "input.toml"
    input_file.write_text(toml.dumps(payload), encoding="utf-8")
    real_run = jqmc_mcmc.MCMC.run

    def timed_run(driver, *args, **kwargs):
        assert kwargs["max_time"] == 50, "the explicit TOML budget must reach the real sampler"
        ticks = 0

        def clock():
            nonlocal ticks
            ticks += 1
            # The real sampler preallocates zero-filled buffers. These fixed
            # H2 samples have nonzero energies, checked below, so filled rows
            # give a deterministic clock trigger without replacing any sample.
            filled = np.count_nonzero(np.any(driver.e_L != 0, axis=1))
            expired = stop_after is not None and filled >= stop_after
            return (1000 if expired else 0) + ticks * 1e-6

        with monkeypatch.context() as patch:
            patch.setattr(jqmc_mcmc, "time", SimpleNamespace(perf_counter=clock))
            return real_run(driver, *args, **kwargs)

    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with monkeypatch.context() as patch:
            patch.chdir(root)
            patch.setattr(sys, "argv", ["jqmc", str(input_file)])
            patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
            patch.setattr(jqmc_mcmc.MCMC, "run", timed_run)
            jqmc_cli._cli()
    finally:
        for handler in logger.handlers:
            if handler not in handlers and isinstance(handler, logging.FileHandler):
                handler.close()
        logger.handlers, logger.level = handlers, level
    return load_rank_checkpoint(str(root / "restart.h5"), rank=0)


@pytest.mark.parametrize("stop_after", [4, 6])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_cli_budget_stop_and_restart_matches_continuous_samples(tmp_path, monkeypatch, stop_after):
    """The TOML time budget stops real MCMC early and preserves its numerical continuation.

    Committed H2 AE Cartesian-GTO/MO has one up/one down electron and nonzero
    Pade J2=1, open boundaries, two walkers and seed 12345. Positions are real
    sampler outputs with no cond(G) selection. A controlled clock crosses the
    explicit 50-second budget after four or six of eight one-update measurements;
    no elapsed wall time, sleep or statistical acceptance band is used.

    Require the actual saved counter and buffer length at interruption, then
    resume the remaining measurements. Final positions, RNG and counters equal
    an uninterrupted eight-measurement run exactly. All stored e_L/e_L2/w_L
    entries use the unchanged same-mode strict zone tolerance. The clock is the
    only synthetic execution input: sampling, energies and file I/O are real.
    This is the XLA MCMC CLI route, not native chunking or an optimizer timeout.
    """
    _write_hamiltonian(tmp_path)
    expected = _run_with_controlled_clock(tmp_path / "continuous", 8, monkeypatch)
    partial = _run_with_controlled_clock(tmp_path / "resumed", 8, monkeypatch, stop_after=stop_after)
    np.testing.assert_array_equal(
        partial["driver_config"]["mcmc_counter"], stop_after, err_msg="the time budget must stop before eight measurements"
    )
    assert partial["observables"]["e_L"].shape == (stop_after, 2)
    assert np.all(partial["observables"]["e_L"] != 0), "the clock trigger requires real nonzero energy samples"
    actual = _run_with_controlled_clock(tmp_path / "resumed", 8 - stop_after, monkeypatch, restart=True)
    assert actual["driver_config"]["mcmc_counter"] == expected["driver_config"]["mcmc_counter"] == 8
    for name in ("latest_r_up_carts", "latest_r_dn_carts"):
        assert actual["walker_state"][name].shape == expected["walker_state"][name].shape == (2, 1, 3)
        np.testing.assert_array_equal(actual["walker_state"][name], expected["walker_state"][name])
    np.testing.assert_array_equal(actual["rng_state"]["jax_PRNG_key_list"], expected["rng_state"]["jax_PRNG_key_list"])
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy"), "strict")
    for name in ("e_L", "e_L2", "w_L"):
        observed, reference = actual["observables"][name], expected["observables"][name]
        assert observed.shape == reference.shape == (8, 2)
        assert np.isfinite(observed).all() and np.isfinite(reference).all()
        np.testing.assert_allclose(observed, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("stop_after", [4, 6])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_cli_budget_verdict_rejects_ignored_budget(tmp_path, monkeypatch, stop_after):
    """After a passing real baseline, discarding the TOML budget must fail the saved-step verdict."""
    baseline = tmp_path / "baseline"
    corrupted = tmp_path / "corrupted"
    baseline.mkdir()
    corrupted.mkdir()
    test_cli_budget_stop_and_restart_matches_continuous_samples(baseline, monkeypatch, stop_after)
    real_run = jqmc_mcmc.MCMC.run

    def ignore_budget(driver, *args, **kwargs):
        kwargs["max_time"] = float("inf")
        return real_run(driver, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(jqmc_mcmc.MCMC, "run", ignore_budget)
        with pytest.raises(AssertionError, match="the time budget must stop before eight measurements"):
            test_cli_budget_stop_and_restart_matches_continuous_samples(corrupted, monkeypatch, stop_after)


def _run_vmc_with_budget_clock(root, num_opt_steps, monkeypatch, *, restart=False, expire=False):
    """Run the actual VMC CLI with ``max_time = 50`` and a controlled clock.

    With ``expire``, every clock reading of the sampler and of the optimization
    loop is 1000 s after the previous one, so the budget is already exceeded
    when it is first checked. Otherwise the clock barely advances.
    """
    root.mkdir(exist_ok=True)
    params = copy.deepcopy(jqmc_cli.cli_parameters["vmc"])
    params.update(
        num_mcmc_steps=8,
        num_mcmc_per_measurement=1,
        num_mcmc_warmup_steps=0,
        num_mcmc_bin_blocks=2,
        num_opt_steps=num_opt_steps,
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
            "hamiltonian_h5": str(root.parent / "hamiltonian.h5"),
            "restart": restart,
            "restart_chk": str(root / "restart.h5"),
            "verbosity": "low",
            "max_time": 50,
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        "vmc": params,
    }
    input_file = root / "input.toml"
    input_file.write_text(toml.dumps(payload), encoding="utf-8")
    ticks = 0

    def clock():
        nonlocal ticks
        ticks += 1
        return ticks * (1000.0 if expire else 1e-6)

    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with monkeypatch.context() as patch:
            patch.chdir(root)
            patch.setattr(sys, "argv", ["jqmc", str(input_file)])
            patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
            patch.setattr(jqmc_mcmc, "time", SimpleNamespace(perf_counter=clock))
            patch.setattr(jqmc_optimizer, "time", SimpleNamespace(perf_counter=clock))
            jqmc_cli._cli()
    finally:
        for handler in logger.handlers:
            if handler not in handlers and isinstance(handler, logging.FileHandler):
                handler.close()
        logger.handlers, logger.level = handlers, level
    return load_rank_checkpoint(str(root / "restart.h5"), rank=0)


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_cli_vmc_budget_stops_between_optimization_steps(tmp_path, monkeypatch):
    """The TOML budget stops VMC only between optimization steps and keeps a continuable checkpoint.

    Same committed H2 AE Cartesian-GTO/MO input, Pade J2=1 (the only optimized
    parameter), two walkers, seed 12345 and eight measurements per SR step as
    the restart test. The controlled clock has already crossed the explicit
    50-second budget when it is first read. The first optimization step must
    still sample all eight measurements and update J2; the loop then stops,
    the CLI finishes and writes its checkpoint and hamiltonian_data_opt_step_1.h5.
    Resuming the remaining step must match an uninterrupted
    two-step run: positions, RNG and optimization counter exactly, the J2
    parameter with the same-mode strict zone tolerance.
    """
    _write_hamiltonian(tmp_path)
    expected = _run_vmc_with_budget_clock(tmp_path / "continuous", 2, monkeypatch)
    partial = _run_vmc_with_budget_clock(tmp_path / "resumed", 2, monkeypatch, expire=True)
    assert partial["driver_config"]["i_opt"] == 1, "the budget must stop the optimization after its first step"
    assert (tmp_path / "resumed" / "hamiltonian_data_opt_step_1.h5").is_file()
    actual = _run_vmc_with_budget_clock(tmp_path / "resumed", 1, monkeypatch, restart=True)
    assert actual["driver_config"]["i_opt"] == expected["driver_config"]["i_opt"] == 2
    for name in ("latest_r_up_carts", "latest_r_dn_carts"):
        assert actual["walker_state"][name].shape == expected["walker_state"][name].shape == (2, 1, 3)
        np.testing.assert_array_equal(actual["walker_state"][name], expected["walker_state"][name])
    np.testing.assert_array_equal(actual["rng_state"]["jax_PRNG_key_list"], expected["rng_state"]["jax_PRNG_key_list"])
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy"), "strict")
    observed_j2, expected_j2 = (
        load_hamiltonian_from_checkpoint(
            str(tmp_path / run / "restart.h5")
        ).wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param
        for run in ("resumed", "continuous")
    )
    assert np.isfinite(observed_j2).all() and np.isfinite(expected_j2).all()
    assert np.any(np.asarray(expected_j2) != 1.0), "the J2 optimization must actually update its parameter"
    np.testing.assert_allclose(observed_j2, expected_j2, atol=atol, rtol=rtol)


_STEP_SECONDS = 1000.0


class _BatchLimitKill(Exception):
    """The scheduler kills the job: nothing after this point is written."""


def _run_vmc_with_step_clock(root, num_opt_steps, monkeypatch, capsys, *, max_time, restart=False, batch_limit=None):
    """Run the actual VMC CLI where every optimization step takes 1000 s of virtual time.

    Only the optimization-loop clock is controlled: it reads 1000 s times the
    number of real ``MCMC.sample`` calls made so far, i.e. one optimization
    step costs 1000 s. Sampling, SR updates, CLI I/O and checkpoints are real.
    With ``batch_limit``, a step that would end after it is killed before it
    samples, as a batch scheduler kills a job at its elapsed limit; the CLI
    then writes nothing more. Returns the checkpoint (or ``None`` when no
    restart.h5 exists), the parsed CLI log and the number of steps started.
    """
    root.mkdir(exist_ok=True)
    params = copy.deepcopy(jqmc_cli.cli_parameters["vmc"])
    params.update(
        num_mcmc_steps=8,
        num_mcmc_per_measurement=1,
        num_mcmc_warmup_steps=0,
        num_mcmc_bin_blocks=2,
        num_opt_steps=num_opt_steps,
        wf_dump_freq=10,
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
            "hamiltonian_h5": str(root.parent / "hamiltonian.h5"),
            "restart": restart,
            "restart_chk": str(root / "restart.h5"),
            "verbosity": "low",
            "max_time": max_time,
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        "vmc": params,
    }
    input_file = root / "input.toml"
    input_file.write_text(toml.dumps(payload), encoding="utf-8")
    real_sample = jqmc_mcmc.MCMC.sample
    started = 0

    def timed_sample(sampler, *args, **kwargs):
        nonlocal started
        if batch_limit is not None and (started + 1) * _STEP_SECONDS > batch_limit:
            raise _BatchLimitKill
        started += 1
        return real_sample(sampler, *args, **kwargs)

    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    capsys.readouterr()
    try:
        with monkeypatch.context() as patch:
            patch.chdir(root)
            patch.setattr(sys, "argv", ["jqmc", str(input_file)])
            patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
            patch.setattr(jqmc_mcmc.MCMC, "sample", timed_sample)
            patch.setattr(jqmc_optimizer, "time", SimpleNamespace(perf_counter=lambda: started * _STEP_SECONDS))
            try:
                jqmc_cli._cli()
            except _BatchLimitKill:
                pass
    finally:
        for handler in logger.handlers:
            if handler not in handlers and isinstance(handler, logging.FileHandler):
                handler.close()
        logger.handlers, logger.level = handlers, level
    parsed = output_parser.parse_jqmc_log(capsys.readouterr().out, "vmc")
    checkpoint = load_rank_checkpoint(str(root / "restart.h5"), rank=0) if (root / "restart.h5").is_file() else None
    return checkpoint, parsed, started


@pytest.mark.parametrize("stop_after", [1, 2])
def test_cli_vmc_budget_does_not_start_a_step_that_ends_after_max_time(tmp_path, monkeypatch, capsys, stop_after):
    """A VMC step that would end after max_time is not started; the stop keeps a continuable checkpoint.

    Same committed H2 AE Cartesian-GTO/MO input, Pade J2 only, two walkers,
    seed 12345 and eight measurements per SR step as the other VMC budget
    test, three requested steps and ``wf_dump_freq = 10``. Each step costs
    1000 s of virtual time and ``max_time = 1000 * stop_after + 500``: after
    step ``stop_after`` the next step would end 500 s after max_time, so the
    run must stop there. The batch limit is 300 s after max_time and kills a
    step that would end after it; a run that started that step (comparing
    max_time only with the elapsed time) would be killed with no restart.h5, no
    hamiltonian_data_opt_step_N.h5 and no ``Program ends``. The run finishes normally with both
    files of step ``stop_after``, and resuming the remaining steps matches an
    uninterrupted three-step run under a budget whose prediction never binds
    before the last step: optimization counter, positions and RNG exactly,
    the J2 parameter with the same-mode strict zone tolerance. The
    uninterrupted run (max_time 3500 s: after its last step the mean predicts
    3000 + 1000 s, but no further step follows) must not report a budget stop.
    """
    total_steps = 3
    max_time = _STEP_SECONDS * stop_after + 500.0
    _write_hamiltonian(tmp_path)
    expected, continuous_log, continuous_started = _run_vmc_with_step_clock(
        tmp_path / "continuous", total_steps, monkeypatch, capsys, max_time=3500
    )
    assert continuous_started == total_steps
    assert continuous_log["program_ended"] and not continuous_log["stopped_by_max_time"]

    stopped, stop_log, started = _run_vmc_with_step_clock(
        tmp_path / "resumed", total_steps, monkeypatch, capsys, max_time=max_time, batch_limit=max_time + 300.0
    )
    assert started == stop_after, "the step predicted to end after max_time must not be started"
    assert stopped is not None, "a budget stop must write restart.h5 before the batch limit"
    assert stopped["driver_config"]["i_opt"] == stop_after
    assert (tmp_path / "resumed" / f"hamiltonian_data_opt_step_{stop_after}.h5").is_file()
    assert stop_log["program_ended"] and stop_log["stopped_by_max_time"]

    actual, resumed_log, _ = _run_vmc_with_step_clock(
        tmp_path / "resumed", total_steps - stop_after, monkeypatch, capsys, max_time=3500, restart=True
    )
    assert resumed_log["program_ended"] and not resumed_log["stopped_by_max_time"]
    assert actual["driver_config"]["i_opt"] == expected["driver_config"]["i_opt"] == total_steps
    for name in ("latest_r_up_carts", "latest_r_dn_carts"):
        assert actual["walker_state"][name].shape == expected["walker_state"][name].shape == (2, 1, 3)
        np.testing.assert_array_equal(actual["walker_state"][name], expected["walker_state"][name])
    np.testing.assert_array_equal(actual["rng_state"]["jax_PRNG_key_list"], expected["rng_state"]["jax_PRNG_key_list"])
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy"), "strict")
    observed_j2, expected_j2 = (
        load_hamiltonian_from_checkpoint(
            str(tmp_path / run / "restart.h5")
        ).wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param
        for run in ("resumed", "continuous")
    )
    assert np.isfinite(observed_j2).all() and np.isfinite(expected_j2).all()
    assert np.any(np.asarray(expected_j2) != 1.0), "the J2 optimization must actually update its parameter"
    np.testing.assert_allclose(observed_j2, expected_j2, atol=atol, rtol=rtol)


_GFMC_DRIVERS = {"lrdmc-bra": jqmc_gfmc.GFMC_n, "lrdmc-tau": jqmc_gfmc.GFMC_t}


def _run_gfmc_with_controlled_clock(root, job_type, steps, monkeypatch, *, restart=False, stop_after=None):
    """GFMC analogue of ``_run_with_controlled_clock``: only the propagator's clock is controlled."""
    root.mkdir(exist_ok=True)
    params = copy.deepcopy(jqmc_cli.cli_parameters[job_type])
    params.update(
        num_mcmc_steps=steps,
        num_gfmc_warmup_steps=GFMC_MIN_WARMUP_STEPS,
        num_gfmc_bin_blocks=GFMC_MIN_BIN_BLOCKS,
        num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
    )
    if job_type == "lrdmc-bra":
        params["num_mcmc_per_measurement"] = 1
        if restart:
            # The adaptive shift is restored from the checkpoint (frozen key).
            params.pop("E_scf")
    payload = {
        "control": {
            "job_type": job_type,
            "mcmc_seed": 12345,
            "number_of_walkers": 2,
            "hamiltonian_h5": str(root.parent / "hamiltonian.h5"),
            "restart": restart,
            "restart_chk": str(root / "restart.h5"),
            "verbosity": "low",
            "max_time": 50,
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        job_type: params,
    }
    input_file = root / "input.toml"
    input_file.write_text(toml.dumps(payload), encoding="utf-8")
    driver_class = _GFMC_DRIVERS[job_type]
    real_run = driver_class.run

    def timed_run(driver, *args, **kwargs):
        assert kwargs["max_time"] == 50, "the explicit TOML budget must reach the real propagator"
        ticks = 0

        def clock():
            nonlocal ticks
            ticks += 1
            # The propagator zero-pads the measurement buffer of this run. The
            # fixed H2 energies are nonzero, so the number of filled rows gives a
            # deterministic trigger after a completed branching step.
            filled = np.count_nonzero(np.asarray(driver._stored_e_L)[:, 0] != 0)
            expired = stop_after is not None and filled >= stop_after
            return (1000 if expired else 0) + ticks * 1e-6

        with monkeypatch.context() as patch:
            patch.setattr(jqmc_gfmc, "time", SimpleNamespace(perf_counter=clock))
            return real_run(driver, *args, **kwargs)

    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with monkeypatch.context() as patch:
            patch.chdir(root)
            patch.setattr(sys, "argv", ["jqmc", str(input_file)])
            patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
            patch.setattr(driver_class, "run", timed_run)
            jqmc_cli._cli()
    finally:
        for handler in logger.handlers:
            if handler not in handlers and isinstance(handler, logging.FileHandler):
                handler.close()
        logger.handlers, logger.level = handlers, level
    return load_rank_checkpoint(str(root / "restart.h5"), rank=0)


@pytest.mark.parametrize("job_type", ["lrdmc-bra", "lrdmc-tau"])
@pytest.mark.parametrize("stop_after", [GFMC_MIN_STEPS + 5, GFMC_MIN_STEPS + 20])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_cli_gfmc_budget_stop_and_restart_matches_continuous_samples(tmp_path, monkeypatch, job_type, stop_after):
    """The TOML time budget stops real LRDMC early and keeps every completed branching step.

    Same committed H2 AE Cartesian-GTO/MO input, Pade J2=1, two walkers and seed
    12345 as the MCMC case. A controlled clock crosses the explicit 50-second
    budget right after branching step ``stop_after`` of ``2 * GFMC_MIN_STEPS``;
    no wall time, sleep or statistical band is used. The saved counter must equal
    the number of completed steps, including the step that triggered the stop
    (it has already moved the walkers and drawn its random numbers), so no
    measurement is dropped and the continuation runs no extra projection. Resuming the remaining steps must match an uninterrupted run:
    positions, JAX and branching RNG and counters exactly, e_L/e_L2/w_L (and, for
    lrdmc-bra, E_scf and the G_L/G_e_L histories) with the same-mode strict zone
    tolerance. This is the XLA GFMC CLI route.
    """
    _write_hamiltonian(tmp_path)
    total = 2 * GFMC_MIN_STEPS
    expected = _run_gfmc_with_controlled_clock(tmp_path / "continuous", job_type, total, monkeypatch)
    partial = _run_gfmc_with_controlled_clock(tmp_path / "resumed", job_type, total, monkeypatch, stop_after=stop_after)
    np.testing.assert_array_equal(
        partial["driver_config"]["mcmc_counter"],
        stop_after,
        err_msg="the time budget must stop after, and keep, the completed branching steps",
    )
    actual = _run_gfmc_with_controlled_clock(tmp_path / "resumed", job_type, total - stop_after, monkeypatch, restart=True)
    assert actual["driver_config"]["mcmc_counter"] == expected["driver_config"]["mcmc_counter"] == total
    for name in ("latest_r_up_carts", "latest_r_dn_carts"):
        assert actual["walker_state"][name].shape == expected["walker_state"][name].shape == (2, 1, 3)
        np.testing.assert_array_equal(actual["walker_state"][name], expected["walker_state"][name])
    for name in (
        "jax_PRNG_key_list",
        "branching_rng_keys",
        "branching_rng_position",
        "branching_rng_has_gauss",
        "branching_rng_cached_gaussian",
    ):
        np.testing.assert_array_equal(actual["rng_state"][name], expected["rng_state"][name])
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy"), "strict")
    names = ["e_L", "e_L2", "w_L"] + (["G_L", "G_e_L"] if job_type == "lrdmc-bra" else [])
    for name in names:
        observed, reference = actual["observables"][name], expected["observables"][name]
        assert observed.shape == reference.shape and observed.size > 0
        assert np.isfinite(observed).all() and np.isfinite(reference).all()
        np.testing.assert_allclose(observed, reference, atol=atol, rtol=rtol)
    if job_type == "lrdmc-bra":
        np.testing.assert_allclose(actual["driver_config"]["E_scf"], expected["driver_config"]["E_scf"], atol=atol, rtol=rtol)


def _run_job_with_budget_clock(root, job_type, steps, monkeypatch, *, restart=False, expire=False):
    """Run the actual CLI with ``max_time = 50`` and a controlled clock for MCMC and GFMC.

    With ``expire``, every clock reading is 1000 s after the previous one, so the
    budget is exceeded at the first check of the sampling loop. Otherwise the
    clock barely advances and the budget is never reached.
    """
    root.mkdir(exist_ok=True)
    params = copy.deepcopy(jqmc_cli.cli_parameters[job_type])
    if job_type == "mcmc":
        params.update(num_mcmc_steps=steps, num_mcmc_per_measurement=1, num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=2)
    else:
        params.update(
            num_mcmc_steps=steps,
            num_gfmc_warmup_steps=GFMC_MIN_WARMUP_STEPS,
            num_gfmc_bin_blocks=GFMC_MIN_BIN_BLOCKS,
            num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
        )
        if job_type == "lrdmc-bra":
            params["num_mcmc_per_measurement"] = 1
            if restart:
                # The adaptive shift is restored from the checkpoint (frozen key).
                params.pop("E_scf")
    payload = {
        "control": {
            "job_type": job_type,
            "mcmc_seed": 12345,
            "number_of_walkers": 2,
            "hamiltonian_h5": str(root.parent / "hamiltonian.h5"),
            "restart": restart,
            "restart_chk": str(root / "restart.h5"),
            "verbosity": "low",
            "max_time": 50,
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        job_type: params,
    }
    input_file = root / "input.toml"
    input_file.write_text(toml.dumps(payload), encoding="utf-8")
    ticks = 0

    def clock():
        nonlocal ticks
        ticks += 1
        return ticks * (1000.0 if expire else 1e-6)

    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with monkeypatch.context() as patch:
            patch.chdir(root)
            patch.setattr(sys, "argv", ["jqmc", str(input_file)])
            patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
            patch.setattr(jqmc_mcmc, "time", SimpleNamespace(perf_counter=clock))
            patch.setattr(jqmc_gfmc, "time", SimpleNamespace(perf_counter=clock))
            jqmc_cli._cli()
    finally:
        for handler in logger.handlers:
            if handler not in handlers and isinstance(handler, logging.FileHandler):
                handler.close()
        logger.handlers, logger.level = handlers, level
    return load_rank_checkpoint(str(root / "restart.h5"), rank=0)


@pytest.mark.parametrize("job_type", ["mcmc", "lrdmc-bra", "lrdmc-tau"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_cli_budget_stop_before_enough_samples_keeps_continuation(tmp_path, monkeypatch, job_type):
    """A budget stop before the binned estimators have enough samples still ends normally and continues.

    Same committed H2 AE Cartesian-GTO/MO input, Pade J2=1, two walkers and seed
    12345 as the restart tests. The controlled clock crosses the explicit
    50-second budget at the first check of the sampling loop, so the run stops
    with fewer measurements than the warm-up and bin settings need for the
    final output. The CLI must still finish and write its checkpoint. Resuming the remaining measurements must then match an
    uninterrupted run: positions, RNG and counters exactly, e_L/e_L2/w_L with
    the same-mode strict zone tolerance. Only elapsed time is controlled.
    """
    _write_hamiltonian(tmp_path)
    total = 8 if job_type == "mcmc" else GFMC_MIN_STEPS
    needed = 2 if job_type == "mcmc" else GFMC_MIN_WARMUP_STEPS + GFMC_MIN_BIN_BLOCKS + GFMC_MIN_COLLECT_STEPS
    expected = _run_job_with_budget_clock(tmp_path / "continuous", job_type, total, monkeypatch)
    partial = _run_job_with_budget_clock(tmp_path / "resumed", job_type, total, monkeypatch, expire=True)
    done = int(partial["driver_config"]["mcmc_counter"])
    assert done < needed, "the budget must stop the run before the final output has enough samples"
    actual = _run_job_with_budget_clock(tmp_path / "resumed", job_type, total - done, monkeypatch, restart=True)

    assert actual["driver_config"]["mcmc_counter"] == expected["driver_config"]["mcmc_counter"] == total
    for name in ("latest_r_up_carts", "latest_r_dn_carts"):
        assert actual["walker_state"][name].shape == expected["walker_state"][name].shape == (2, 1, 3)
        np.testing.assert_array_equal(actual["walker_state"][name], expected["walker_state"][name])
    rng_names = ["jax_PRNG_key_list"]
    if job_type.startswith("lrdmc"):
        rng_names += [
            "branching_rng_keys",
            "branching_rng_position",
            "branching_rng_has_gauss",
            "branching_rng_cached_gaussian",
        ]
    for name in rng_names:
        np.testing.assert_array_equal(actual["rng_state"][name], expected["rng_state"][name])
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy"), "strict")
    for name in ("e_L", "e_L2", "w_L"):
        observed, reference = actual["observables"][name], expected["observables"][name]
        assert observed.shape == reference.shape and observed.size > 0
        assert np.isfinite(observed).all() and np.isfinite(reference).all()
        np.testing.assert_allclose(observed, reference, atol=atol, rtol=rtol)
