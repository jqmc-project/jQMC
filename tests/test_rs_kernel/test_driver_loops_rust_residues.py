"""Rust-ON driver loops: the remaining optimizer and CLI restart routes.

``test_driver_loops_rust.py`` already runs ``run_optimize`` (one SR step on H2 AE),
the Python-API checkpoint continuation, the CLI entry and the two-twist average with
the production Rust backend. This module covers two further routes:

1. ``run_optimize`` with two SR steps on the ccECP fixture ``H2_ecp_ccpvdz_cart``
   (exp J1+J2, and J1+J2+J3 with a Cartesian AO J3 matrix optimized too). The
   Rust side consumes the XLA proposal normals of its own threefry key through
   the per-step sweep seam; every other draw and all arithmetic stay native.
   Each consumed batch, each solved update and the applied parameters must equal
   the XLA run through the reviewed ``test_driver_loops_rust._parity`` helper.
2. One Rust ``jqmc_cli._cli()`` mcmc run of N measurements, and a second CLI
   project running N/2 measurements followed by a CLI ``restart = true`` run of
   N/2 more. The resumed checkpoint must equal the uninterrupted one: walker
   positions, threefry keys and the counter exactly, stored e_L / e_L2 / w_L and
   the public energy tuple at the same-mode strict restart tolerance.

Verdicts are deterministic; no sampling-error band is used. Full and mixed are
each judged within their own precision.
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

import contextlib
import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import toml

pytestmark = [pytest.mark.with_chdir]

import jqmc.jqmc_mcmc as mcmc_module
import jqmc.jqmc_optimizer as optimizer_module
from jqmc import jqmc_cli
from jqmc._checkpoint import load_rank_checkpoint
from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import (
    ATOL_PYTHON_FP64_TIGHT,
    MCMC_MIN_BIN_BLOCKS,
    MCMC_MIN_WARMUP_STEPS,
    RTOL_PYTHON_FP64_TIGHT,
)
from jqmc.jqmc_mcmc import MCMC
from tests._finite_allclose import assert_finite_allclose
from tests.test_rs_kernel import test_driver_loops_rust as driver
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.conftest import assert_native_ran, assert_no_native_ran, native_ctx_spy

# --------------------------------------------------------------------------- #
# 1. run_optimize: two SR steps on H2 ccECP, Rust vs XLA on the same inputs
# --------------------------------------------------------------------------- #
_ECP_OPT_CASES = {
    "h2_ecp_exp_j12": ("mol_ecp_cart_j12", False),
    "h2_ecp_exp_j123": ("mol_ecp_cart_j123", True),
}
_ECP_OPT_STEPS = 2


def _all_jastrow_parameters(ham) -> np.ndarray:
    """J1 and J2 scalars followed by the flattened J3 matrix (when present)."""
    jastrow = ham.wavefunction_data.jastrow_data
    parts = [
        np.atleast_1d(np.asarray(jastrow.jastrow_one_body_data.jastrow_1b_param, dtype=np.float64)),
        np.atleast_1d(np.asarray(jastrow.jastrow_two_body_data.jastrow_2b_param, dtype=np.float64)),
    ]
    if jastrow.jastrow_three_body_data is not None:
        parts.append(np.asarray(jastrow.jastrow_three_body_data.j_matrix, dtype=np.float64).ravel())
    return np.concatenate(parts)


def _optimize_two_steps(ham, rust_on: bool, opt_j3: bool) -> dict:
    """Production SR steps; passive spies record each consumed batch, target and applied step."""
    recorded: dict = {"native_param_grads": 0, "steps": []}
    production_native_grads = mcmc_module._rs_native_c_param_grads
    production_target = optimizer_module.compute_real_coordinate_optimization_target
    production_finalize = optimizer_module.finalize_optimization_step

    def recording_target(batch, coordinate_map, **kwargs):
        target = production_target(batch, coordinate_map, **kwargs)
        recorded.update(batch=batch, target=target)
        return target

    def counting_native_grads(*args, **kwargs):
        if kwargs.get("quantity") == "ln_Psi":
            recorded["native_param_grads"] += 1
        return production_native_grads(*args, **kwargs)

    def recording_finalize(target, coordinate_map, **kwargs):
        result = production_finalize(target, coordinate_map, **kwargs)
        recorded["steps"].append(dict(batch=recorded["batch"], coordinate_map=coordinate_map, target=target, step=result))
        return result

    params_before = _all_jastrow_parameters(ham)
    with driver._backend(rust_on), pytest.MonkeyPatch.context() as patch:
        mcmc = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=driver._OPT_SEED,
            num_walkers=driver._OPT_WALKERS,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=False,
        )
        patch.setattr(optimizer_module, "compute_real_coordinate_optimization_target", recording_target)
        patch.setattr(optimizer_module, "finalize_optimization_step", recording_finalize)
        patch.setattr(mcmc_module, "_rs_native_c_param_grads", counting_native_grads)
        seam = driver._inject_xla_inputs() if rust_on else contextlib.nullcontext({"chunk": 0, "sweep": 0})
        with seam as seen, native_ctx_spy() as counts:
            mcmc.run_optimize(
                num_mcmc_steps=driver._OPT_STEPS,
                num_opt_steps=_ECP_OPT_STEPS,
                num_mcmc_warmup_steps=0,
                num_mcmc_bin_blocks=driver._OPT_BIN_BLOCKS,
                optimizer_kwargs={
                    "method": "sr",
                    "delta": driver._OPT_DELTA,
                    "epsilon": driver._OPT_EPSILON,
                    "cg_flag": False,
                },
                opt_J1_param=True,
                opt_J2_param=True,
                opt_J3_param=opt_j3,
                dump_wavefunction=False,
                use_device_optimizer=False,
            )
    return dict(
        steps=recorded["steps"],
        counts=dict(counts),
        native_param_grads=int(recorded["native_param_grads"]),
        seen=dict(seen),
        accepted=int(mcmc._MCMC__accepted_moves),
        keys=np.asarray(mcmc._MCMC__jax_PRNG_key_list),
        r_up=np.asarray(mcmc._MCMC__latest_r_up_carts, dtype=np.float64),
        r_dn=np.asarray(mcmc._MCMC__latest_r_dn_carts, dtype=np.float64),
        params_before=params_before,
        params_after=_all_jastrow_parameters(mcmc.hamiltonian_data),
    )


def _assert_two_step_parity(rust: dict, xla: dict, label: str) -> None:
    assert len(rust["steps"]) == len(xla["steps"]) == _ECP_OPT_STEPS, f"{label}: SR step count"
    for index, (actual, reference) in enumerate(zip(rust["steps"], xla["steps"], strict=True)):
        tag = f"{label}: step {index}"
        driver._parity(actual["batch"].weights, reference["batch"].weights, f"{tag} consumed weights")
        driver._parity(actual["batch"].local_energies, reference["batch"].local_energies, f"{tag} consumed local energies")
        actual_o = driver._host_coordinates(actual["coordinate_map"], actual["batch"], actual["batch"].log_derivative_sources)
        reference_o = driver._host_coordinates(
            reference["coordinate_map"], reference["batch"], reference["batch"].log_derivative_sources
        )
        driver._parity(actual_o, reference_o, f"{tag} log derivatives", zones=driver._DERIV_ZONES)
        driver._parity(actual["target"].q, reference["target"].q, f"{tag} solved SR update q", zones=driver._DERIV_ZONES)
        driver._parity(actual["step"].q, reference["step"].q, f"{tag} applied update", zones=driver._DERIV_ZONES)
    assert rust["accepted"] == xla["accepted"], f"{label}: accepted-move count differs"
    driver._exact(rust["keys"], xla["keys"], f"{label}: threefry keys")
    driver._exact(rust["r_up"], xla["r_up"], f"{label}: final r_up")
    driver._exact(rust["r_dn"], xla["r_dn"], f"{label}: final r_dn")
    driver._parity(rust["params_after"], xla["params_after"], f"{label}: applied Jastrow parameters", zones=driver._DERIV_ZONES)


@pytest.mark.light
@pytest.mark.parametrize("case", sorted(_ECP_OPT_CASES))
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_run_optimize_two_sr_steps_h2_ecp_rust_equals_xla(case):
    """Two ``run_optimize`` SR steps on H2 ccECP with Rust equal the XLA steps on the same inputs.

    H2 ccECP cc-pVDZ Cartesian (open boundary, real, non-local ECP) with exp J1+J2,
    and with J1+J2 plus a Cartesian AO J3 matrix that is optimized too; four
    walkers, 25 measurements per step, seed 20260927, SR delta 0.05, epsilon 1e-3,
    direct solve, host optimizer. The Rust side must run native code, including the
    native log-derivative assembly, and consume the XLA proposal normals of its own
    threefry key through the per-step sweep seam. For each of the two steps the
    consumed weights, local energies, log-derivative coordinates, solved update and
    applied update agree at the medium XLA-Rust tolerance; the final walkers, keys
    and accepted-move count are exact and the final parameters agree. Each applied
    step equals delta times its solved update. A perturbed second-step update must
    fail the same comparison.
    """
    pattern, opt_j3 = _ECP_OPT_CASES[case]
    ham = rs_ham(pattern)
    rust = _optimize_two_steps(ham, rust_on=True, opt_j3=opt_j3)
    xla = _optimize_two_steps(ham, rust_on=False, opt_j3=opt_j3)
    assert_native_ran(rust["counts"], f"{case}: Rust run_optimize")
    assert_no_native_ran(xla["counts"], f"{case}: XLA run_optimize")
    assert rust["native_param_grads"] >= _ECP_OPT_STEPS * driver._OPT_STEPS, f"{case}: native ln Psi derivatives did not run"
    assert xla["native_param_grads"] == 0, f"{case}: the XLA reference reached the native parameter derivatives"
    assert rust["seen"]["sweep"] >= _ECP_OPT_STEPS * driver._OPT_STEPS, f"{case}: sweep seam did not engage: {rust['seen']}"
    for index, step in enumerate(rust["steps"]):
        assert np.any(np.asarray(step["target"].q) != 0), f"{case}: step {index} update is vacuous"
        assert_finite_allclose(
            step["step"].q,
            driver._OPT_DELTA * np.asarray(step["target"].q),
            atol=ATOL_PYTHON_FP64_TIGHT,
            rtol=RTOL_PYTHON_FP64_TIGHT,
            err_msg=f"{case}: step {index} Rust applied step equals delta * q",
        )
    n_j12 = 2
    if opt_j3:
        assert xla["params_after"].size > n_j12, f"{case}: the J3 matrix is absent"
        assert np.any(xla["params_after"][n_j12:] != xla["params_before"][n_j12:]), f"{case}: J3 did not move"
    assert np.any(xla["params_after"][:n_j12] != xla["params_before"][:n_j12]), f"{case}: J1/J2 did not move"
    _assert_two_step_parity(rust, xla, case)
    # Negative control on the same runs: the XLA second update with its sign flipped (a non-zero update,
    # asserted above) fails the same reviewed parity helper; no tolerance is selected here.
    last = xla["steps"][-1]
    q_ref = np.asarray(last["target"].q)
    perturbed_target = type("T", (), {"q": -q_ref})()
    perturbed = dict(xla, steps=[*xla["steps"][:-1], dict(last, target=perturbed_target)])
    with pytest.raises(AssertionError, match="step 1 solved SR update q"):
        _assert_two_step_parity(rust, perturbed, case)


# --------------------------------------------------------------------------- #
# 2. CLI mcmc run + restart with Rust equals the uninterrupted Rust CLI run
# --------------------------------------------------------------------------- #
_CLI_HALF = MCMC_MIN_WARMUP_STEPS + MCMC_MIN_BIN_BLOCKS  # each half passes the fresh-run step check


def _run_cli_mcmc(root: Path, hamiltonian_h5: Path, monkeypatch, *, num_mcmc_steps: int, restart: bool) -> dict:
    """One Rust ``_cli()`` mcmc job in ``root`` (fresh or ``restart = true``); returns the native counts."""
    root.mkdir(parents=True, exist_ok=True)
    params = copy.deepcopy(jqmc_cli.cli_parameters["mcmc"])
    params.update(driver._CLI_MCMC)
    params["num_mcmc_steps"] = int(num_mcmc_steps)
    payload = {
        "control": {
            "job_type": "mcmc",
            "mcmc_seed": driver._CLI_SEED,
            "number_of_walkers": driver._CLI_WALKERS,
            "hamiltonian_h5": str(hamiltonian_h5),
            "restart": bool(restart),
            "restart_chk": str(root / "restart.h5"),
            "verbosity": "low",
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        "mcmc": params,
    }
    input_file = root / ("input_restart.toml" if restart else "input.toml")
    input_file.write_text(toml.dumps(payload))
    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with driver._backend(True), monkeypatch.context() as patch:
            patch.chdir(root)
            patch.setattr(sys, "argv", ["jqmc", str(input_file)])
            patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
            with native_ctx_spy() as counts:
                jqmc_cli._cli()
    finally:
        logger.handlers, logger.level = handlers, level
    return dict(counts)


def _checkpoint_values(path: Path) -> dict:
    state = load_rank_checkpoint(str(path), rank=0)
    mcmc = MCMC.load_from_hdf5(str(path))
    return dict(
        r_up=state["walker_state"]["latest_r_up_carts"],
        r_dn=state["walker_state"]["latest_r_dn_carts"],
        keys=state["rng_state"]["jax_PRNG_key_list"],
        counter=int(state["driver_config"]["mcmc_counter"]),
        e_L=state["observables"]["e_L"],
        e_L2=state["observables"]["e_L2"],
        w_L=state["observables"]["w_L"],
        E=np.asarray(mcmc.get_E(num_mcmc_warmup_steps=MCMC_MIN_WARMUP_STEPS, num_mcmc_bin_blocks=MCMC_MIN_BIN_BLOCKS)),
    )


def _assert_cli_restart(resumed: dict, continuous: dict) -> None:
    assert resumed["counter"] == continuous["counter"] == 2 * _CLI_HALF, "CLI restart: mcmc counter"
    for name in ("r_up", "r_dn", "keys"):
        driver._exact(resumed[name], continuous[name], f"CLI restart: {name}")
    for name in ("e_L", "e_L2", "w_L", "E"):
        driver._restart_close(resumed[name], continuous[name], f"CLI restart: {name}")


@pytest.mark.heavy
@pytest.mark.without_chdir
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_rust_cli_mcmc_restart_equals_uninterrupted_rust_cli_run(tmp_path, monkeypatch):
    """A Rust CLI mcmc run resumed through ``restart = true`` equals the uninterrupted Rust CLI run.

    H2 ccECP cc-pVDZ Cartesian with exp J1+J2, written to HDF5 and read back by the
    CLI; seed 34356, two walkers, Dt = 1.7, epsilon_AS = 1e-5, three moves per
    measurement. The uninterrupted project runs 80 measurements; the split project
    runs 40, then a second ``_cli()`` call with ``restart = true`` and the same TOML
    values runs 40 more from ``restart.h5``. All three CLI calls must run native
    code. Resumed walker positions, threefry keys and the measurement counter are
    exact; stored e_L / e_L2 / w_L and the public energy tuple agree with the
    uninterrupted run at the same-mode strict restart tolerance (nonzero, finite).
    The split first half must differ from the final state, and a shifted resumed
    energy must fail.
    """
    hamiltonian_h5 = driver._write_cli_hamiltonian(tmp_path)
    counts = _run_cli_mcmc(tmp_path / "continuous", hamiltonian_h5, monkeypatch, num_mcmc_steps=2 * _CLI_HALF, restart=False)
    assert_native_ran(counts, "uninterrupted Rust CLI mcmc")
    counts = _run_cli_mcmc(tmp_path / "split", hamiltonian_h5, monkeypatch, num_mcmc_steps=_CLI_HALF, restart=False)
    assert_native_ran(counts, "first-half Rust CLI mcmc")
    first_half = _checkpoint_values(tmp_path / "split" / "restart.h5")
    counts = _run_cli_mcmc(tmp_path / "split", hamiltonian_h5, monkeypatch, num_mcmc_steps=_CLI_HALF, restart=True)
    assert_native_ran(counts, "resumed Rust CLI mcmc")
    continuous = _checkpoint_values(tmp_path / "continuous" / "restart.h5")
    resumed = _checkpoint_values(tmp_path / "split" / "restart.h5")
    assert first_half["counter"] == _CLI_HALF
    assert np.any(first_half["r_up"] != continuous["r_up"]), "the restart leg must actually advance the walkers"
    for name in ("e_L", "e_L2", "w_L"):
        assert np.any(np.asarray(continuous[name]) != 0), f"CLI restart: {name} is identically zero"
    _assert_cli_restart(resumed, continuous)
    shifted = dict(resumed, E=resumed["E"] + np.array([1.0e-3, 0.0, 0.0, 0.0]))
    with pytest.raises(AssertionError, match="CLI restart: E"):
        _assert_cli_restart(shifted, continuous)
