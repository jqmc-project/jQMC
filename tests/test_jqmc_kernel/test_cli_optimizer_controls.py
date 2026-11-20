"""Real TOML optimizer controls reach real SR/LM solvers, without solver substitutes."""

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

import numpy as np
import pytest
import toml

from jqmc import jqmc_cli, jqmc_optimizer
from jqmc._checkpoint import load_hamiltonian_from_checkpoint, load_rank_checkpoint
from jqmc._precision import is_mixed_precision_enabled
from jqmc.jqmc_mcmc import MCMC
from tests.test_jqmc_kernel.test_cli_restart_values import _write_hamiltonian
from tests.test_jqmc_kernel.test_optimizer_real_batch_update import (
    _REVIEWED_SHIPPED_OPTIMIZER_DEFAULTS,
    _assert_shipped_optimizer_default_snapshot,
)

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_shipped_optimizer_toml_defaults_reach_target_and_final_update(tmp_path, monkeypatch):
    """Untouched shipped optimizer defaults survive the real TOML resolver and scale the actual update exactly once.

    Only sampler size knobs are shortened. The committed molecular H2 J2
    fixture runs one actual aSR optimization step. The reviewed configuration
    snapshot is independent of the production table, and passive observations
    check its controls at the real target and finalizer. This input/application
    contract does not replace independent solver-model value tests.
    """
    _write_hamiltonian(tmp_path)
    params = copy.deepcopy(jqmc_cli.cli_parameters["vmc"])
    _assert_shipped_optimizer_default_snapshot(params["optimizer_kwargs"])
    params.update(num_mcmc_steps=8, num_mcmc_per_measurement=1, num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=2, num_opt_steps=1)
    checkpoint = tmp_path / "restart.h5"
    payload = {
        "control": {
            "job_type": "vmc",
            "mcmc_seed": 12345,
            "number_of_walkers": 2,
            "hamiltonian_h5": str(tmp_path / "hamiltonian.h5"),
            "restart": False,
            "restart_chk": str(checkpoint),
            "verbosity": "low",
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        "vmc": params,
    }
    input_file = tmp_path / "input.toml"
    input_file.write_text(toml.dumps(payload), encoding="utf-8")
    targets, steps = [], []
    original_target = jqmc_optimizer.compute_real_coordinate_optimization_target
    original_finalize = jqmc_optimizer.finalize_optimization_step

    def observe_target(*args, **kwargs):
        target = original_target(*args, **kwargs)
        targets.append((dict(kwargs), target))
        return target

    def observe_finalize(target, coordinate_map, **kwargs):
        step = original_finalize(target, coordinate_map, **kwargs)
        steps.append((dict(kwargs), coordinate_map, step))
        return step

    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with monkeypatch.context() as patch:
            patch.chdir(tmp_path)
            patch.setattr(sys, "argv", ["jqmc", str(input_file)])
            patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
            patch.setattr(jqmc_optimizer, "compute_real_coordinate_optimization_target", observe_target)
            patch.setattr(jqmc_optimizer, "finalize_optimization_step", observe_finalize)
            jqmc_cli._cli()
    finally:
        for handler in logger.handlers:
            if handler not in handlers and isinstance(handler, logging.FileHandler):
                handler.close()
        logger.handlers, logger.level = handlers, level
    assert len(targets) == len(steps) == 1
    kwargs, target = targets[0]
    final_kwargs, coordinate_map, step = steps[0]
    observed = dict(
        delta=final_kwargs["learning_rate"],
        epsilon=kwargs["epsilon"],
        cg_flag=kwargs["cg"],
        cg_max_iter=kwargs["cg_max_iter"],
        cg_tol=kwargs["cg_tol"],
        use_lm=kwargs["use_lm"],
        lm_subspace_dim=kwargs["lm_subspace_dim"],
        lm_cond=kwargs["lm_condition_cutoff"],
    )
    expected = {key: value for key, value in _REVIEWED_SHIPPED_OPTIMIZER_DEFAULTS.items() if key != "method"}
    assert observed == expected, "CLI changed or dropped a reviewed optimizer default"
    np.testing.assert_array_equal(
        step.q,
        _REVIEWED_SHIPPED_OPTIMIZER_DEFAULTS["delta"] * np.asarray(target.q),
        err_msg="CLI did not apply the reviewed learning rate exactly once",
    )
    assert [block.name for block in coordinate_map.blocks] == ["j2_param"]
    assert np.any(step.q != 0), "the shipped-default update must be live"
    state = load_rank_checkpoint(str(checkpoint), rank=0)
    assert state["driver_config"]["i_opt"] == 1
    updated = load_hamiltonian_from_checkpoint(str(checkpoint))
    actual = np.asarray(updated.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param)
    expected_parameter = np.asarray(coordinate_map.blocks[0].values) + step.updates[0].delta
    np.testing.assert_array_equal(actual, expected_parameter, err_msg="checkpoint lost the applied default update")


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_shipped_optimizer_toml_defaults_reject_dropped_learning_rate(tmp_path, monkeypatch):
    """Dropping the CLI learning-rate option before the real optimizer fails the reviewed default boundary check."""
    original = MCMC.run_optimize

    def drop_learning_rate(self, *args, **kwargs):
        controls = dict(kwargs["optimizer_kwargs"])
        controls.pop("delta")
        kwargs["optimizer_kwargs"] = controls
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MCMC, "run_optimize", drop_learning_rate)
    with pytest.raises(AssertionError, match="CLI changed or dropped a reviewed optimizer default"):
        test_shipped_optimizer_toml_defaults_reach_target_and_final_update(tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "cg_flag,use_lm,lm_cond",
    [(False, False, 0.001), (True, False, 0.001), (False, True, 0.05)],
    ids=["sr-direct", "sr-conjugate-gradient", "linear-method-nondefault-cutoff"],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_optimizer_toml_controls_reach_real_solvers(tmp_path, monkeypatch, cg_flag, use_lm, lm_cond):
    """Actual VMC TOML selects CG/direct SR and forwards the non-default LM cutoff.

    One real J2-only optimization step uses committed H2 AE Cartesian-GTO/MO,
    one up/one down electron, Pade J2=1, two walkers, seed 12345, eight one-update
    measurements and two bins, without cond(G) selection. The real host solver
    callbacks are observed, not replaced by fake numerical results. Require the
    chosen CG/direct path exactly once, the exact TOML controls at the optimizer
    boundary, and the exact cutoff at the real LM pencil solver when enabled.
    A real updated checkpoint must be written. The non-default cutoff is 0.05,
    not the default 0.001; LM is explicitly active with subspace dimension one.

    This is an input/dispatch contract, not a new numerical solver oracle or a
    statistical convergence claim. Existing independent dense-embedding and
    pencil tests own solver values. Native/device, twist and all flag products
    are not claimed by this serial host fixture. Both global precisions apply.
    """
    _write_hamiltonian(tmp_path)
    params = copy.deepcopy(jqmc_cli.cli_parameters["vmc"])
    params.update(
        num_mcmc_steps=8,
        num_mcmc_per_measurement=1,
        num_mcmc_warmup_steps=0,
        num_mcmc_bin_blocks=2,
        num_opt_steps=1,
    )
    for key in params:
        if key.startswith("opt_"):
            params[key] = key == "opt_J2_param"
    params["optimizer_kwargs"].update(
        method="sr", delta=0.01, cg_flag=cg_flag, use_lm=use_lm, lm_cond=lm_cond, lm_subspace_dim=1
    )
    checkpoint = tmp_path / "restart.h5"
    payload = {
        "control": {
            "job_type": "vmc",
            "mcmc_seed": 12345,
            "number_of_walkers": 2,
            "hamiltonian_h5": str(tmp_path / "hamiltonian.h5"),
            "restart": False,
            "restart_chk": str(checkpoint),
            "verbosity": "low",
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        "vmc": params,
    }
    input_file = tmp_path / "input.toml"
    input_file.write_text(toml.dumps(payload), encoding="utf-8")
    target_arguments, lm_cutoffs = [], []
    solver_calls = {"cg": 0, "direct": 0}
    real_target = jqmc_optimizer.compute_real_coordinate_optimization_target
    real_lm = jqmc_optimizer.solve_lm_real_pencil
    real_cg = jqmc_optimizer._kernel.solve_wide_sr_real_cg
    real_direct = jqmc_optimizer._kernel.solve_wide_sr_real_direct

    def observe_target(*args, **kwargs):
        target_arguments.append([kwargs["cg"], kwargs["use_lm"], kwargs["lm_condition_cutoff"], kwargs["lm_subspace_dim"]])
        return real_target(*args, **kwargs)

    def observe_cg(*args, **kwargs):
        solver_calls["cg"] += 1
        return real_cg(*args, **kwargs)

    def observe_direct(*args, **kwargs):
        solver_calls["direct"] += 1
        return real_direct(*args, **kwargs)

    def observe_lm(*args, **kwargs):
        lm_cutoffs.append(kwargs["condition_cutoff"])
        return real_lm(*args, **kwargs)

    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with monkeypatch.context() as patch:
            patch.chdir(tmp_path)
            patch.setattr(sys, "argv", ["jqmc", str(input_file)])
            patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
            patch.setattr(jqmc_optimizer, "compute_real_coordinate_optimization_target", observe_target)
            patch.setattr(jqmc_optimizer, "solve_lm_real_pencil", observe_lm)
            patch.setattr(jqmc_optimizer._kernel, "solve_wide_sr_real_cg", observe_cg)
            patch.setattr(jqmc_optimizer._kernel, "solve_wide_sr_real_direct", observe_direct)
            jqmc_cli._cli()
    finally:
        for handler in logger.handlers:
            if handler not in handlers and isinstance(handler, logging.FileHandler):
                handler.close()
        logger.handlers, logger.level = handlers, level
    np.testing.assert_array_equal(target_arguments, [[cg_flag, use_lm, lm_cond, 1]])
    observed_calls = [solver_calls["cg"], solver_calls["direct"]]
    expected_calls = [int(cg_flag), int(not cg_flag)]
    np.testing.assert_array_equal(observed_calls, expected_calls, err_msg="SR solver ignored the TOML cg_flag")
    expected_cutoffs = [lm_cond] if use_lm else []
    observed_lm = [len(lm_cutoffs)] + lm_cutoffs
    expected_lm = [int(use_lm)] + expected_cutoffs
    np.testing.assert_array_equal(observed_lm, expected_lm, err_msg="LM solver ignored the TOML lm_cond")
    state = load_rank_checkpoint(str(checkpoint), rank=0)
    assert state["driver_config"]["i_opt"] == 1
    updated = load_hamiltonian_from_checkpoint(str(checkpoint))
    j2 = np.asarray(updated.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param)
    assert np.isfinite(j2).all() and np.any(j2 != 1.0), "the real optimization must update its nonzero J2 parameter"


@pytest.mark.parametrize("fault", ["force-cg", "reset-lm-cutoff"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_optimizer_toml_verdict_rejects_ignored_controls(tmp_path, monkeypatch, fault):
    """After a passing real baseline, overriding CG or LM controls fails the downstream solver observations."""
    baseline, corrupted = tmp_path / "baseline", tmp_path / "corrupted"
    baseline.mkdir()
    corrupted.mkdir()
    use_lm = fault == "reset-lm-cutoff"
    cutoff = 0.05 if use_lm else 0.001
    test_optimizer_toml_controls_reach_real_solvers(baseline, monkeypatch, False, use_lm, cutoff)
    real_target = jqmc_optimizer.compute_real_coordinate_optimization_target

    def ignore_control(*args, **kwargs):
        if fault == "force-cg":
            kwargs["cg"] = True
        else:
            kwargs["lm_condition_cutoff"] = 0.001
        return real_target(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(jqmc_optimizer, "compute_real_coordinate_optimization_target", ignore_control)
        with pytest.raises(AssertionError, match="solver ignored the TOML"):
            test_optimizer_toml_controls_reach_real_solvers(corrupted, monkeypatch, False, use_lm, cutoff)
