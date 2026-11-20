"""Rust-ON driver loops: optimizer step, checkpoint continuation, CLI and twist average.

``flag_rust_on_CPU=True`` is the shipped default, but ``tests/conftest.py`` forces
``JQMC_FLAG_RUST_ON_CPU=0`` outside ``tests/test_rs_kernel``, so the optimizer loop,
the checkpoint round trip, the CLI entry and the twist-average orchestrator were
only ever run on XLA. This module runs each of them with the
production Rust backend, and every Rust side is wrapped in ``native_ctx_spy`` so
the test fails if native code did not run.

1. ``run_optimize`` (one SR step): Rust and XLA runs on the same inputs. The Rust
   sampler normally draws its proposal normals natively (per-step route: numpy), so
   the test-only seam of ``test_mcmc_exotic_leaf_value_parity`` feeds it the XLA
   normals of the same threefry key; every other draw and all arithmetic stay
   native. The walk must be identical, then the batch the optimizer consumed
   (weights, local energies, log-derivative coordinates), the solved update ``q``
   and the applied parameters must agree at ``get_tolerance_xla_rust_consistency``.
2. Checkpoint save, merge, load and continue with Rust equals the uninterrupted Rust
   run: MCMC, GFMC_n and GFMC_t (GFMC_t with one walker).
3. The CLI (``jqmc_cli._cli``) with Rust equals a direct Python-API Rust run built
   with the same non-default TOML values, and a Rust CLI run fed the XLA inputs
   equals the XLA CLI run.
4. ``run_twist_average_from_hamiltonians`` (MCMC) with Rust on two twists (real
   Gamma and complex k = 0.25), one rank per twist emulated in one process through
   the production inter-comm SUM data flow: per-twist values equal standalone Rust
   chains built with the documented per-twist seed, and the BZ values equal the
   documented weighted formula.

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
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import toml

pytestmark = [pytest.mark.with_chdir]

import jqmc.jqmc_mcmc as mcmc_module
import jqmc.jqmc_optimizer as optimizer_module
from jqmc import jqmc_cli
from jqmc._checkpoint import load_rank_checkpoint, merge_rank_checkpoints
from jqmc._precision import get_tolerance_min, get_tolerance_xla_rust_consistency, is_mixed_precision_enabled
from jqmc._setting import (
    ATOL_PYTHON_FP64_BZ_MEAN,
    ATOL_PYTHON_FP64_TIGHT,
    GFMC_MIN_COLLECT_STEPS,
    MCMC_MIN_BIN_BLOCKS,
    MCMC_MIN_STEPS,
    MCMC_MIN_WARMUP_STEPS,
    RTOL_PYTHON_FP64_TIGHT,
    RTOL_PYTHON_FP64_ZERO,
)
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.jqmc_mcmc import MCMC
from jqmc.jqmc_optimizer import DeviceOptimizationBatch
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._finite_allclose import assert_finite_allclose
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.conftest import assert_native_ran, assert_no_native_ran, native_ctx_spy, trexio_path
from tests.test_rs_kernel.test_mcmc_exotic_leaf_value_parity import _inject_xla_inputs

# Zones crossed by an end-to-end sampled value (same set as the MCMC chain parity).
_PARITY_ZONES = ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic", "swct")
# Zones crossed by the per-sample log-derivative sources (param-deriv parity set).
_DERIV_ZONES = ("ao_eval", "mo_eval", "det_eval", "jastrow_eval")
# Same-backend continuation: the restart-route zone set of the XLA restart tests.
_RESTART_ZONES = ("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy")


@contextlib.contextmanager
def _backend(rust_on: bool):
    """Select the backend through the production env override; restore the previous value."""
    previous = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
    if rust_on:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    else:
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
        else:
            os.environ["JQMC_FLAG_RUST_ON_CPU"] = previous


def _exact(actual, reference, label):
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape and actual.size > 0, f"{label}: shape {actual.shape} vs {reference.shape}"
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)), f"{label}: operand not finite"
    np.testing.assert_array_equal(actual, reference, err_msg=label)


def _parity(actual, reference, label, zones=_PARITY_ZONES):
    atol, rtol = get_tolerance_xla_rust_consistency(zones, level="medium")
    assert np.asarray(actual).size > 0, f"{label}: empty"
    assert_finite_allclose(actual, reference, atol=atol, rtol=rtol, err_msg=label)


def _restart_close(actual, reference, label):
    atol, rtol = get_tolerance_min(_RESTART_ZONES, "strict")
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape and actual.size > 0, f"{label}: shape {actual.shape} vs {reference.shape}"
    assert_finite_allclose(actual, reference, atol=atol, rtol=rtol, err_msg=label)


def _h2_ae_pade_j12() -> Hamiltonian_data:
    """H2 all-electron cc-pVDZ, Pade J1 = 1.0 and J2 = 0.5 (the optimizer fixture)."""
    structure_data, _aos, _mos_up, _mos_dn, geminal_mo_data, coulomb_potential_data = read_trexio_file(
        trexio_file=trexio_path("H2_ae_ccpvdz_cart.h5"), store_tuple=True
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=tuple(0.0 for _ in structure_data.atomic_numbers),
            jastrow_1b_type="pade",
        ),
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="pade"),
        jastrow_three_body_data=None,
    )
    ham = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data),
    )
    ham.sanity_check()
    return ham


def _jastrowless_periodic(stem: str) -> Hamiltonian_data:
    structure_data, _aos, _mos_up, _mos_dn, geminal_mo_data, coulomb_potential_data = read_trexio_file(
        trexio_file=trexio_path(stem), store_tuple=True
    )
    ham = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=geminal_mo_data),
    )
    ham.sanity_check()
    return ham


# --------------------------------------------------------------------------- #
# 1. run_optimize: Rust vs XLA on the same inputs
# --------------------------------------------------------------------------- #
_OPT_CASES = {
    "h2_ae_pade_j12": _h2_ae_pade_j12,
    "li_twist_exp_j12": lambda: rs_ham("twist_sphepgto_j12"),
}
_OPT_WALKERS = 4
_OPT_STEPS = 25
_OPT_BIN_BLOCKS = 5
_OPT_SEED = 20260927
_OPT_DELTA = 0.05
_OPT_EPSILON = 1.0e-3


def _jastrow_parameters(ham) -> np.ndarray:
    jastrow = ham.wavefunction_data.jastrow_data
    return np.asarray(
        [
            np.asarray(jastrow.jastrow_one_body_data.jastrow_1b_param),
            np.asarray(jastrow.jastrow_two_body_data.jastrow_2b_param),
        ],
        dtype=np.float64,
    )


def _host_coordinates(coordinate_map, batch, sources) -> np.ndarray:
    if isinstance(batch, DeviceOptimizationBatch):
        return np.asarray(coordinate_map.transform_derivatives_device(sources))
    return np.asarray(coordinate_map.transform_derivatives(sources))


def _optimize_one_step(ham, rust_on: bool, num_opt_steps: int = 1) -> dict:
    """Production SR steps; passive spies record each consumed batch, target and step."""
    recorded: dict = {"native_param_grads": 0, "steps": []}
    production_native_grads = mcmc_module._rs_native_c_param_grads
    production_target = optimizer_module.compute_real_coordinate_optimization_target
    production_finalize = optimizer_module.finalize_optimization_step

    def recording_target(batch, coordinate_map, **kwargs):
        target = production_target(batch, coordinate_map, **kwargs)
        recorded.update(batch=batch, coordinate_map=coordinate_map, target=target)
        return target

    def counting_native_grads(*args, **kwargs):
        if kwargs.get("quantity") == "ln_Psi":
            recorded["native_param_grads"] += 1
        return production_native_grads(*args, **kwargs)

    def recording_finalize(target, coordinate_map, **kwargs):
        result = production_finalize(target, coordinate_map, **kwargs)
        recorded["step"] = result
        recorded["steps"].append(dict(batch=recorded["batch"], coordinate_map=coordinate_map, target=target, step=result))
        return result

    with _backend(rust_on), pytest.MonkeyPatch.context() as patch:
        mcmc = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=_OPT_SEED,
            num_walkers=_OPT_WALKERS,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=False,
        )
        patch.setattr(optimizer_module, "compute_real_coordinate_optimization_target", recording_target)
        patch.setattr(optimizer_module, "finalize_optimization_step", recording_finalize)
        patch.setattr(mcmc_module, "_rs_native_c_param_grads", counting_native_grads)
        seam = _inject_xla_inputs() if rust_on else contextlib.nullcontext({"chunk": 0, "sweep": 0})
        with seam as seen, native_ctx_spy() as counts:
            mcmc.run_optimize(
                num_mcmc_steps=_OPT_STEPS,
                num_opt_steps=num_opt_steps,
                num_mcmc_warmup_steps=0,
                num_mcmc_bin_blocks=_OPT_BIN_BLOCKS,
                optimizer_kwargs={"method": "sr", "delta": _OPT_DELTA, "epsilon": _OPT_EPSILON, "cg_flag": False},
                opt_J1_param=True,
                opt_J2_param=True,
                opt_J3_param=False,
                dump_wavefunction=False,
                use_device_optimizer=False,
            )
    batch = recorded["batch"]
    coordinate_map = recorded["coordinate_map"]
    return dict(
        steps=recorded["steps"],
        counts=dict(counts),
        native_param_grads=int(recorded["native_param_grads"]),
        seen=dict(seen),
        accepted=int(mcmc._MCMC__accepted_moves),
        keys=np.asarray(mcmc._MCMC__jax_PRNG_key_list),
        r_up=np.asarray(mcmc._MCMC__latest_r_up_carts, dtype=np.float64),
        r_dn=np.asarray(mcmc._MCMC__latest_r_dn_carts, dtype=np.float64),
        weights=np.asarray(batch.weights, dtype=np.float64),
        local_energies=np.asarray(batch.local_energies),
        log_derivatives=_host_coordinates(coordinate_map, batch, batch.log_derivative_sources),
        q=np.asarray(recorded["target"].q),
        step_q=np.asarray(recorded["step"].q),
        params_before=_jastrow_parameters(ham),
        params_after=_jastrow_parameters(mcmc.hamiltonian_data),
    )


def _assert_optimize_parity(rust: dict, xla: dict, label: str) -> None:
    assert rust["accepted"] == xla["accepted"], f"{label}: accepted-move count differs"
    _exact(rust["keys"], xla["keys"], f"{label}: threefry keys")
    _exact(rust["r_up"], xla["r_up"], f"{label}: final r_up")
    _exact(rust["r_dn"], xla["r_dn"], f"{label}: final r_dn")
    _parity(rust["weights"], xla["weights"], f"{label}: consumed batch weights")
    _parity(rust["local_energies"], xla["local_energies"], f"{label}: consumed batch local energies")
    _parity(rust["log_derivatives"], xla["log_derivatives"], f"{label}: consumed batch log derivatives", zones=_DERIV_ZONES)
    _parity(rust["q"], xla["q"], f"{label}: solved SR update q", zones=_DERIV_ZONES)
    _parity(rust["params_after"], xla["params_after"], f"{label}: applied J1/J2 parameters", zones=_DERIV_ZONES)


_OPT_CACHE: dict = {}


def _optimize_pair(case: str):
    key = (case, is_mixed_precision_enabled())
    if key not in _OPT_CACHE:
        ham = _OPT_CASES[case]()
        _OPT_CACHE[key] = (_optimize_one_step(ham, rust_on=True), _optimize_one_step(ham, rust_on=False))
    return _OPT_CACHE[key]


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("h2_ae_pade_j12", marks=pytest.mark.light, id="h2_ae_pade_j12"),
        pytest.param("li_twist_exp_j12", marks=pytest.mark.heavy, id="li_twist_exp_j12"),
    ],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_run_optimize_step_rust_equals_xla(case):
    """One ``run_optimize`` SR step with Rust equals the XLA step on the same inputs.

    H2 AE (open boundary, real, Pade J1+J2) and Li k = 0.25 (periodic ccECP, complex,
    exp J1+J2); four walkers, 25 measurements, seed 20260927, SR delta 0.05,
    epsilon 1e-3, direct solve, host optimizer, J1 and J2 optimized. The Rust side
    must run native code, including the native J1/J2 log-derivative assembly, and
    consume the XLA proposal normals of its own threefry key through the per-step
    sweep seam. Accepted moves, keys and final walkers must be byte equal; the
    consumed weights, local energies, log-derivative coordinates, the solved update
    and the applied parameters agree at the medium XLA-Rust tolerance. A perturbed
    update must fail the same comparison.
    """
    rust, xla = _optimize_pair(case)
    assert_native_ran(rust["counts"], f"{case}: Rust run_optimize")
    assert_no_native_ran(xla["counts"], f"{case}: XLA run_optimize")
    assert rust["native_param_grads"] >= _OPT_STEPS, f"{case}: native ln Psi parameter derivatives did not run"
    assert xla["native_param_grads"] == 0, f"{case}: the XLA reference reached the native parameter derivatives"
    if case == "h2_ae_pade_j12":
        # The real path serves them from the native obs ctx (the complex twist uses
        # the free-kernel adapter, which the ctx spy does not see).
        for method in ("pd_ok_j1", "pd_ok_j2"):
            calls = sum(n for name, n in rust["counts"].items() if name.startswith(method))
            assert calls > 0, f"{case}: native {method}* did not run: {rust['counts']}"
    assert rust["seen"]["sweep"] >= _OPT_STEPS, f"{case}: the XLA-normal sweep seam did not engage: {rust['seen']}"
    assert np.any(xla["params_after"] != xla["params_before"]), f"{case}: the SR step did not move the parameters"
    # The applied step is the learning rate times the solved update.
    assert_finite_allclose(
        rust["step_q"],
        _OPT_DELTA * rust["q"],
        atol=ATOL_PYTHON_FP64_TIGHT,
        rtol=RTOL_PYTHON_FP64_TIGHT,
        err_msg=f"{case}: Rust applied step equals delta * q",
    )
    _assert_optimize_parity(rust, xla, case)
    # Negative control on the same runs: the XLA update moved by ten medium tolerances fails.
    atol, rtol = get_tolerance_xla_rust_consistency(_DERIV_ZONES, level="medium")
    perturbed = dict(xla, q=xla["q"] + 10.0 * (atol + rtol * np.abs(xla["q"])))
    with pytest.raises(AssertionError, match="solved SR update q"):
        _assert_optimize_parity(rust, perturbed, case)


# --------------------------------------------------------------------------- #
# 2. Checkpoint save -> load -> continue with Rust
# --------------------------------------------------------------------------- #
_RESTART_SEED = 12345


def _save_merge(driver, ham, tmp_path: Path, driver_type: str) -> str:
    rank_file = str(tmp_path / "._restart_rank0.h5")
    driver.save_to_hdf5(rank_file)
    merged = str(tmp_path / "restart.h5")
    merge_rank_checkpoints(
        output_path=merged,
        mpi_size=1,
        driver_type=driver_type,
        hamiltonian_data=ham,
        tmp_pattern=str(tmp_path / "._restart_rank{rank}.h5"),
    )
    return merged


def _make_restart_driver(kind: str, ham):
    common = dict(hamiltonian_data=ham, mcmc_seed=_RESTART_SEED, random_discretized_mesh=True, comput_position_deriv=True)
    if kind == "MCMC":
        return MCMC(num_walkers=4, Dt=2.0, **common)
    gfmc = dict(num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS, alat=0.30, non_local_move="tmove", **common)
    if kind == "GFMC_n":
        return GFMC_n(num_walkers=2, num_mcmc_per_measurement=3, E_scf=-1.0, **gfmc)
    return GFMC_t(num_walkers=2, tau=0.05, **gfmc)


_RESTART_HALF = {"MCMC": 8, "GFMC_n": 3, "GFMC_t": 3}
_RESTART_FIELDS = {
    "MCMC": ("e_L", "e_L2", "w_L", "force_HF_stored", "force_PP_stored"),
    "GFMC_n": ("_stored_e_L", "_stored_w_L", "_stored_force_HF", "_stored_force_PP", "_stored_E_L_force_PP"),
    "GFMC_t": ("_stored_e_L", "_stored_w_L", "_stored_force_HF", "_stored_force_PP", "_stored_E_L_force_PP"),
}
_RESTART_STATE = {
    "MCMC": ("_MCMC__latest_r_up_carts", "_MCMC__latest_r_dn_carts", "_MCMC__jax_PRNG_key_list"),
    "GFMC_n": ("_latest_r_up_carts", "_latest_r_dn_carts", "_jax_PRNG_key_list"),
    "GFMC_t": ("_latest_r_up_carts", "_latest_r_dn_carts", "_jax_PRNG_key_list"),
}


def _check_restart(continuous, resumed, kind: str) -> None:
    for field in _RESTART_FIELDS[kind]:
        reference = np.asarray(getattr(continuous, field))
        assert np.any(reference != 0), f"{kind}: {field} is identically zero"
        _restart_close(getattr(resumed, field), reference, f"{kind}: {field}")
    for field in _RESTART_STATE[kind]:
        _exact(getattr(resumed, field), getattr(continuous, field), f"{kind}: {field}")
    if kind != "MCMC":
        actual_rng, reference_rng = resumed._branching_rng.get_state(), continuous._branching_rng.get_state()
        assert actual_rng[0] == reference_rng[0]
        _exact(actual_rng[1], reference_rng[1], f"{kind}: branching MT19937 state")
        assert actual_rng[2:] == reference_rng[2:], f"{kind}: branching MT19937 position/gauss cache"


@pytest.mark.heavy
@pytest.mark.parametrize("kind", ["MCMC", "GFMC_n", "GFMC_t"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_rust_checkpoint_continuation_equals_uninterrupted_run(tmp_path, kind):
    """Rust checkpoint save, merge, load and continue equals the uninterrupted Rust run.

    H2 ccECP cc-pVDZ Cartesian with exp J1+J2 (open boundary, real), seed 12345,
    native random mesh, atomic forces on. MCMC: four walkers, 16 measurements
    versus 8 + 8. Both GFMC algorithms use two walkers, 6 branching steps
    versus 3 + 3. Both halves and the reference must run native
    code. Stored energies, weights and force histories agree at the same-mode strict
    restart tolerance (nonzero, finite); walker positions, threefry keys and the
    GFMC branching MT19937 state are exact.
    """
    ham = rs_ham("mol_ecp_cart_j12")
    half = _RESTART_HALF[kind]
    with _backend(True):
        with native_ctx_spy() as counts:
            continuous = _make_restart_driver(kind, ham)
            continuous.run(num_mcmc_steps=2 * half)
        assert_native_ran(counts, f"{kind}: uninterrupted Rust run")
        with native_ctx_spy() as counts:
            interrupted = _make_restart_driver(kind, ham)
            interrupted.run(num_mcmc_steps=half)
        assert_native_ran(counts, f"{kind}: first Rust half")
        checkpoint = _save_merge(interrupted, ham, tmp_path, kind)
        resumed = type(interrupted).load_from_hdf5(checkpoint, rank=0)
        with native_ctx_spy() as counts:
            resumed.run(num_mcmc_steps=half)
        assert_native_ran(counts, f"{kind}: resumed Rust half")
    _check_restart(continuous, resumed, kind)
    # Negative control: a resumed energy history shifted in one entry must fail.
    field = _RESTART_FIELDS[kind][0]
    shifted = SimpleNamespace(**{name: np.array(getattr(resumed, name), copy=True) for name in _RESTART_FIELDS[kind]})
    for name in _RESTART_STATE[kind]:
        setattr(shifted, name, getattr(resumed, name))
    if kind != "MCMC":
        shifted._branching_rng = resumed._branching_rng
    getattr(shifted, field).reshape(-1)[-1] += 1.0e-3
    with pytest.raises(AssertionError, match=f"{kind}: {field}"):
        _check_restart(continuous, shifted, kind)


# Per-step native MCMC route (issue #135). The continuation test above takes the
# native energy/force chunk; parameter derivatives (real) and position derivatives
# at a complex twist keep the per-step ``mcmc_sweep*_batch`` dispatch, whose proposal
# normals are drawn host-side. Each case: fixture, driver flags, the per-step sweep
# method that must run, and the stored fields compared to the uninterrupted run.
_PERSTEP_CASES = {
    "real_param_deriv": ("mol_ecp_cart_j12", dict(comput_log_WF_param_deriv=True), "mcmc_sweep_batch"),
    "complex_twist_force": ("twist_sphepgto_j12", dict(comput_position_deriv=True), "mcmc_sweep_complex_batch"),
}
_PERSTEP_HALF = 2


def _perstep_fields(driver, case: str) -> dict:
    fields = {name: np.asarray(getattr(driver, name)) for name in ("e_L", "e_L2", "w_L")}
    if case == "complex_twist_force":
        fields.update(force_HF=np.asarray(driver.force_HF_stored), force_PP=np.asarray(driver.force_PP_stored))
    else:
        for name, values in driver._MCMC__stored_log_WF_param_grads.items():
            fields[f"param_grads[{name}]"] = np.asarray(values)
    return fields


def _check_perstep_continuation(continuous, resumed, case: str, label: str) -> None:
    for name in _RESTART_STATE["MCMC"]:
        _exact(getattr(resumed, name), getattr(continuous, name), f"{case} {label}: {name}")
    reference, actual = _perstep_fields(continuous, case), _perstep_fields(resumed, case)
    assert sorted(actual) == sorted(reference), f"{case} {label}: stored fields {sorted(actual)} vs {sorted(reference)}"
    for name, values in reference.items():
        assert values.shape[0] == 2 * _PERSTEP_HALF, f"{case} {label}: {name} has {values.shape[0]} steps"
        assert np.any(values != 0), f"{case} {label}: {name} is identically zero"
        _restart_close(actual[name], values, f"{case} {label}: {name}")


@pytest.mark.heavy
@pytest.mark.parametrize("continuation", ["split_calls", "checkpoint"])
@pytest.mark.parametrize("case", list(_PERSTEP_CASES))
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_rust_perstep_continuation_equals_uninterrupted_run(tmp_path, case, continuation):
    """Rust per-step MCMC: run(2)+run(2), or run(2)+checkpoint+run(2), equals run(4).

    ``real_param_deriv``: H2 ccECP cc-pVDZ Cartesian, exp J1+J2, with
    ``comput_log_WF_param_deriv=True``; ``complex_twist_force``: lithium k = 0.25
    twist, J1+J2, with ``comput_position_deriv=True``. Four walkers, seed 12345,
    eight moves per measurement. Every run must call the per-step native sweep
    (``mcmc_sweep_batch`` / ``mcmc_sweep_complex_batch``) once per step and never
    the energy chunk. Reference: the uninterrupted Rust ``run(4)`` of the same
    driver settings. ``split_calls``: two consecutive ``run(2)`` calls on one
    object; ``checkpoint``: ``run(2)`` followed by save, merge, load and ``run(2)``.
    The continuation must reproduce the reference walker positions and threefry
    keys exactly, and its stored local energies, weights and (per case)
    parameter-derivative or force histories at the same-mode strict restart
    tolerance. Full and mixed are judged separately.
    """
    fixture, flags, sweep = _PERSTEP_CASES[case]
    ham = rs_ham(fixture)
    settings = dict(hamiltonian_data=ham, mcmc_seed=_RESTART_SEED, num_walkers=4, num_mcmc_per_measurement=8, Dt=2.0, **flags)

    def run(driver, label, steps_per_call, calls):
        # The native ctx is built once per driver object and reused by later run()
        # calls, so one spy spans every call on that object; one sweep per step.
        with native_ctx_spy() as counts:
            for _ in range(calls):
                driver.run(num_mcmc_steps=steps_per_call)
        assert_native_ran(counts, f"{case}: {label}")
        assert counts[sweep] == calls * steps_per_call, (
            f"{case}: {label} did not take the per-step {sweep} route: {dict(counts)}"
        )
        chunk = {name: n for name, n in counts.items() if "chunk" in name}
        assert not chunk, f"{case}: {label} took the native chunk route: {chunk}"

    with _backend(True):
        continuous = MCMC(**settings)
        run(continuous, "uninterrupted run(4)", 2 * _PERSTEP_HALF, calls=1)
        resumed = MCMC(**settings)
        if continuation == "split_calls":
            run(resumed, "run(2)+run(2)", _PERSTEP_HALF, calls=2)
        else:
            run(resumed, "first half", _PERSTEP_HALF, calls=1)
            resumed = MCMC.load_from_hdf5(_save_merge(resumed, ham, tmp_path, "MCMC"), rank=0)
            run(resumed, "resumed half", _PERSTEP_HALF, calls=1)
    _check_perstep_continuation(continuous, resumed, case, continuation)
    # Negative control: the same continuation with its walkers moved by 1e-6 bohr must fail.
    moved = SimpleNamespace(**{name: getattr(resumed, name) for name in _RESTART_STATE["MCMC"]})
    moved._MCMC__latest_r_up_carts = np.asarray(resumed._MCMC__latest_r_up_carts) + 1.0e-6
    with pytest.raises(AssertionError, match="_MCMC__latest_r_up_carts"):
        _check_perstep_continuation(continuous, moved, case, "moved walkers")


# --------------------------------------------------------------------------- #
# 3. CLI entry with Rust
# --------------------------------------------------------------------------- #
_CLI_SEED = 34356
_CLI_WALKERS = 2
_CLI_MCMC = dict(
    num_mcmc_steps=MCMC_MIN_STEPS,
    num_mcmc_per_measurement=3,
    num_mcmc_warmup_steps=MCMC_MIN_WARMUP_STEPS,
    num_mcmc_bin_blocks=MCMC_MIN_BIN_BLOCKS,
    Dt=1.7,
    epsilon_AS=1.0e-5,
    atomic_force=False,
    compute_stress=False,
    parameter_derivatives=False,
)


def _write_cli_hamiltonian(root: Path) -> Path:
    path = root / "hamiltonian.h5"
    rs_ham("mol_ecp_cart_j12").save_to_hdf5(str(path))
    return path


def _run_cli_mcmc(root: Path, hamiltonian_h5: Path, monkeypatch, *, rust_on: bool, inject: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    params = copy.deepcopy(jqmc_cli.cli_parameters["mcmc"])
    params.update(_CLI_MCMC)
    payload = {
        "control": {
            "job_type": "mcmc",
            "mcmc_seed": _CLI_SEED,
            "number_of_walkers": _CLI_WALKERS,
            "hamiltonian_h5": str(hamiltonian_h5),
            "restart": False,
            "restart_chk": str(root / "restart.h5"),
            "verbosity": "low",
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        "mcmc": params,
    }
    input_file = root / "input.toml"
    input_file.write_text(toml.dumps(payload))
    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with _backend(rust_on), monkeypatch.context() as patch:
            patch.chdir(root)
            patch.setattr(sys, "argv", ["jqmc", str(input_file)])
            patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
            seam = _inject_xla_inputs() if inject else contextlib.nullcontext()
            with seam, native_ctx_spy() as counts:
                jqmc_cli._cli()
    finally:
        logger.handlers, logger.level = handlers, level
    if rust_on:
        assert_native_ran(counts, "Rust CLI mcmc")
    else:
        assert_no_native_ran(counts, "XLA CLI mcmc")
    return dict(
        state=load_rank_checkpoint(str(root / "restart.h5"), rank=0), driver=MCMC.load_from_hdf5(str(root / "restart.h5"))
    )


def _cli_values(run: dict) -> dict:
    state = run["state"]
    return dict(
        r_up=state["walker_state"]["latest_r_up_carts"],
        r_dn=state["walker_state"]["latest_r_dn_carts"],
        keys=state["rng_state"]["jax_PRNG_key_list"],
        e_L=state["observables"]["e_L"],
        w_L=state["observables"]["w_L"],
        E=np.asarray(run["driver"].get_E(num_mcmc_warmup_steps=MCMC_MIN_WARMUP_STEPS, num_mcmc_bin_blocks=MCMC_MIN_BIN_BLOCKS)),
    )


@pytest.mark.heavy
@pytest.mark.without_chdir
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_rust_cli_mcmc_equals_direct_rust_api_run(tmp_path, monkeypatch):
    """The Rust CLI mcmc job equals a direct Python-API Rust MCMC with the same TOML values.

    H2 ccECP cc-pVDZ Cartesian with exp J1+J2, written to HDF5 and read back by the
    CLI; seed 34356, two walkers, 40 measurements, and the non-default values
    Dt = 1.7, epsilon_AS = 1e-5 and three moves per measurement. The direct run
    loads the same HDF5 file and passes the same values. Both must run native code.
    Checkpoint walker positions and threefry keys are exact; stored e_L / w_L and
    the public energy tuple agree at the same-mode strict restart tolerance. A
    shifted energy must fail.
    """
    hamiltonian_h5 = _write_cli_hamiltonian(tmp_path)
    cli = _cli_values(_run_cli_mcmc(tmp_path / "cli", hamiltonian_h5, monkeypatch, rust_on=True))
    with _backend(True), native_ctx_spy() as counts:
        direct = MCMC(
            hamiltonian_data=Hamiltonian_data.load_from_hdf5(str(hamiltonian_h5)),
            Dt=_CLI_MCMC["Dt"],
            mcmc_seed=_CLI_SEED,
            num_walkers=_CLI_WALKERS,
            num_mcmc_per_measurement=_CLI_MCMC["num_mcmc_per_measurement"],
            epsilon_AS=_CLI_MCMC["epsilon_AS"],
            comput_position_deriv=False,
            comput_stress=False,
            comput_log_WF_param_deriv=False,
        )
        direct.run(num_mcmc_steps=_CLI_MCMC["num_mcmc_steps"])
    assert_native_ran(counts, "direct Rust MCMC")
    _exact(cli["r_up"], np.asarray(direct._MCMC__latest_r_up_carts), "CLI vs direct: r_up")
    _exact(cli["r_dn"], np.asarray(direct._MCMC__latest_r_dn_carts), "CLI vs direct: r_dn")
    _exact(cli["keys"], np.asarray(direct._MCMC__jax_PRNG_key_list), "CLI vs direct: keys")
    _restart_close(cli["e_L"], np.asarray(direct.e_L), "CLI vs direct: e_L")
    _restart_close(cli["w_L"], np.asarray(direct.w_L), "CLI vs direct: w_L")
    direct_E = np.asarray(direct.get_E(num_mcmc_warmup_steps=MCMC_MIN_WARMUP_STEPS, num_mcmc_bin_blocks=MCMC_MIN_BIN_BLOCKS))
    _restart_close(cli["E"], direct_E, "CLI vs direct: E")
    with pytest.raises(AssertionError, match="CLI vs direct: E"):
        _restart_close(cli["E"] + np.array([1.0e-3, 0.0, 0.0, 0.0]), direct_E, "CLI vs direct: E")


@pytest.mark.heavy
@pytest.mark.without_chdir
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_rust_cli_mcmc_equals_xla_cli_on_the_same_walk(tmp_path, monkeypatch):
    """The Rust CLI mcmc job fed the XLA inputs equals the XLA CLI job with the same seed.

    Same H2 ccECP J1+J2 input and TOML values as the direct-API test. The Rust CLI
    run consumes the XLA proposal normals and ECP mesh rotations of its own threefry
    keys through the energy-chunk seam; everything else is the production native
    chunk. The walk must be identical (positions and keys exact), and stored e_L /
    w_L and the public energy tuple agree at the medium XLA-Rust tolerance.
    """
    hamiltonian_h5 = _write_cli_hamiltonian(tmp_path)
    xla = _cli_values(_run_cli_mcmc(tmp_path / "xla", hamiltonian_h5, monkeypatch, rust_on=False))
    rust = _cli_values(_run_cli_mcmc(tmp_path / "rust", hamiltonian_h5, monkeypatch, rust_on=True, inject=True))
    for name in ("r_up", "r_dn", "keys"):
        _exact(rust[name], xla[name], f"Rust vs XLA CLI: {name}")
    for name in ("e_L", "w_L", "E"):
        _parity(rust[name], xla[name], f"Rust vs XLA CLI: {name}")


# --------------------------------------------------------------------------- #
# 4. run_twist_average_from_hamiltonians (MCMC) with Rust on two twists
# --------------------------------------------------------------------------- #
_TWIST_WEIGHTS = np.array([1.0, 3.0])
_TWIST_SEED = 20260927
_TWIST_KNOBS = dict(
    num_mcmc_steps=MCMC_MIN_STEPS,
    num_walkers=2,
    num_mcmc_warmup_steps=MCMC_MIN_WARMUP_STEPS,
    num_mcmc_bin_blocks=MCMC_MIN_BIN_BLOCKS,
)


class _RecordingInterComm:
    """Twist-roots comm of the rank owning twist 1: records its Level-2 SUM operands."""

    def __init__(self):
        self.sent = []

    def Get_rank(self):
        return 0

    def allreduce(self, value, op=None):
        self.sent.append(np.array(value, copy=True))
        return np.array(value, copy=True)


class _ReplayInterComm:
    """Twist-roots comm of the rank owning twist 0: adds the recorded twist-1 operands."""

    def __init__(self, remote):
        self._remote = remote
        self.calls = 0

    def Get_rank(self):
        return 0

    def allreduce(self, value, op=None):
        remote = self._remote[self.calls]
        self.calls += 1
        return np.asarray(value) + remote


def _assert_bz_combine(result, raw_weights) -> None:
    W = np.asarray(raw_weights, dtype=np.float64) / np.sum(raw_weights)
    E_k, s_k = result.per_twist_E, result.per_twist_E_std
    tol = dict(atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO)
    assert_finite_allclose(result.energy.E_TA, W @ E_k, err_msg="E_TA", **tol)
    assert_finite_allclose(result.energy.E_TA_std, np.sqrt(W**2 @ s_k**2), err_msg="E_TA_std", **tol)
    assert_finite_allclose(result.energy.s_twist, np.sqrt(W @ (E_k - result.energy.E_TA) ** 2), err_msg="s_twist", **tol)


_TWIST_CACHE: dict = {}


def _two_twist_rust_runs():
    key = is_mixed_precision_enabled()
    if key in _TWIST_CACHE:
        return _TWIST_CACHE[key]
    from mpi4py import MPI

    import jqmc._twist_average as twist_average
    from jqmc._twist_average import _twist_sampler_seed, run_twist_average_from_hamiltonians

    hams = [_jastrowless_periodic("lithium.h5"), _jastrowless_periodic("lithium_k025.h5")]
    world = MPI.COMM_WORLD
    recorder = _RecordingInterComm()

    def _call(twist_index, inter):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                twist_average,
                "split_world_for_twists",
                lambda world_comm, n_twist: SimpleNamespace(intra_comm=world, inter_comm=inter, twist_index=twist_index),
            )
            with native_ctx_spy() as counts:
                result = run_twist_average_from_hamiltonians(
                    hamiltonian_list=hams,
                    weights=_TWIST_WEIGHTS,
                    mcmc_seed=_TWIST_SEED,
                    world_comm=world,
                    **_TWIST_KNOBS,
                )
            assert_native_ran(counts, f"Rust twist-average MCMC, twist {twist_index}")
            return result

    with _backend(True):
        _call(1, recorder)
        replay = _ReplayInterComm(recorder.sent)
        result = _call(0, replay)
        assert replay.calls == len(recorder.sent) > 0, "the Level-2 inter-comm SUMs of both twists must pair up"
        standalone = []
        for k_idx, ham in enumerate(hams):
            with native_ctx_spy() as counts:
                ref = MCMC(
                    hamiltonian_data=ham,
                    mcmc_seed=_twist_sampler_seed(_TWIST_SEED, k_idx, len(hams)),
                    num_walkers=_TWIST_KNOBS["num_walkers"],
                    mpi_comm_in=world,
                    world_comm_in=world,
                )
                ref.run(num_mcmc_steps=_TWIST_KNOBS["num_mcmc_steps"])
            assert_native_ran(counts, f"standalone Rust MCMC, twist {k_idx}")
            E, E_std, _, _ = ref.get_E(
                num_mcmc_warmup_steps=_TWIST_KNOBS["num_mcmc_warmup_steps"],
                num_mcmc_bin_blocks=_TWIST_KNOBS["num_mcmc_bin_blocks"],
            )
            standalone.append((float(E), float(E_std)))
    _TWIST_CACHE[key] = (result, standalone)
    return _TWIST_CACHE[key]


@pytest.mark.heavy
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_rust_two_twist_average_equals_standalone_chains_and_bz_formula():
    """Rust ``run_twist_average_from_hamiltonians`` on two twists equals standalone chains.

    Jastrow-free Li ccECP at real Gamma (lithium.h5) and at complex k = 0.25
    (lithium_k025.h5), raw weights 1:3, two walkers, 40 measurements, seed
    20260927. The world split is replaced so one process runs twist 1 (recording
    its Level-2 SUM operands) and then twist 0 (receiving them), which is the
    production two-rank data flow; the intra comm is the real one-rank COMM_WORLD.
    Every run must execute native code. Per-twist energy and error equal
    standalone Rust MCMC chains built with the documented per-twist seed
    ``_twist_sampler_seed(seed, k, 2)`` exactly; the BZ energy, error and twist
    dispersion equal the documented weighted formulas; a changed weight or a
    shifted per-twist energy must fail that combine check.
    """
    result, standalone = _two_twist_rust_runs()
    for k_idx, (E, E_std) in enumerate(standalone):
        _exact(result.per_twist_E[k_idx], E, f"twist {k_idx} E")
        _exact(result.per_twist_E_std[k_idx], E_std, f"twist {k_idx} E_std")
    assert result.per_twist_E[0] != result.per_twist_E[1], "the two twists must be physically distinct"
    _assert_bz_combine(result, _TWIST_WEIGHTS)
    with pytest.raises(AssertionError, match="E_TA"):
        _assert_bz_combine(result, np.array([1.0, 2.0]))
    shifted = SimpleNamespace(
        per_twist_E=result.per_twist_E + np.array([0.0, 1.0e-3]), per_twist_E_std=result.per_twist_E_std, energy=result.energy
    )
    with pytest.raises(AssertionError, match="E_TA"):
        _assert_bz_combine(shifted, _TWIST_WEIGHTS)
