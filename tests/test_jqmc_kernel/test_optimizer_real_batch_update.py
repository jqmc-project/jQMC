"""Real-batch oracles for one production ``run_optimize`` step.

Each case runs ONE optimization step of ``MCMC.run_optimize`` on H2
(all-electron cc-pVDZ, Pade J1 + J2, fixed seed) with the real sampler and the
real optimizer. Two passive spies record, without changing any value:

* the per-step walker positions (``MCMC.sample`` is replaced by a loop of
  one-step production ``MCMC.sample`` calls that reads the latest positions
  after each call), and
* the batch, coordinate map and target that ``compute_real_coordinate_optimization_target``
  consumed and returned, and the ``finalize_optimization_step`` result.

The checks are deterministic statements about that fixed, stored data:

1. The batch the optimizer consumed equals an independent evaluation at the
   stored configurations: ``e_L`` from ``compute_local_energy``, the log
   derivatives from ``jax.grad`` of ``evaluate_ln_wavefunction`` and (LM) the
   local-energy derivatives from ``jax.grad`` of ``compute_local_energy``.
2. The SR update equals a dense NumPy solve of the documented regularized SR
   system ``(S + epsilon diag(S)) q = f`` with ``S = <dO dO>`` and
   ``f = -2 <dO (e_L - <e_L>)>`` (weighted, centred); the diagonal-preconditioned
   production solve ``(D^-1/2 S D^-1/2 + epsilon I) q_pc = D^-1/2 f``,
   ``q = D^-1/2 q_pc`` is algebraically this system. The LM update equals a dense
   generalized eigensolve of the non-symmetric centred pencil. The applied
   parameter change equals ``delta * q``.
3. With ``opt_J2_param=False`` the J2 parameter is bit-identical after the
   step and the solved system is the J1-only one.
4. The reweighted fixed-sample energy
   ``E(c) = sum_i w_i r_i(c) e_L(x_i; c) / sum_i w_i r_i(c)``,
   ``r_i(c) = |Psi_c(x_i) / Psi_c0(x_i)|^2``, evaluated on the SAME stored
   configurations, is strictly lower at the applied parameters than at the
   starting parameters, and strictly higher at the reflected step.

Negative controls replay the production target builder on the captured batch
with a flipped force and with epsilon dropped, and apply an update to the
frozen block; each must make the corresponding check fail for the intended
reason. No sampling-error band is used anywhere.
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

import copy
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from jqmc import jqmc_miscs
import jqmc.jqmc_optimizer as optimizer_module
from jqmc._precision import get_tolerance, get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import (
    ATOL_PYTHON_FP64_LOOSE,
    OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF,
    RTOL_PYTHON_FP64_LOOSE,
    min_S_diag_abs,
)
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.jqmc_mcmc import MCMC
from jqmc.jqmc_optimizer import DeviceOptimizationBatch
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data, evaluate_ln_wavefunction
from tests.test_jqmc_kernel._exact_sr_reference import exact_centred_moments, exact_sr_solve

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_TREXIO_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ae_ccpvdz_cart.h5")
_J1_START = 1.0
_J2_START = 0.5
_NUM_WALKERS = 4
_NUM_STEPS = 25
_NUM_BIN_BLOCKS = 5
_DELTA = 0.05
_SEED = 20260927
_PARAMETER_NAMES = ("j1_param", "j2_param")

# The batch sources are produced through the Jastrow and local-energy zones;
# the independent evaluation below crosses the same zones.
_SOURCE_ZONES = ("jastrow_eval", "jastrow_grad_lap", "wf_eval", "wf_kinetic", "local_energy")
# The optimizer stores the updated J1/J2 scalars in float64 in both modes
# (jastrow_factor._JASTROW_PARAMETER_STORAGE_DTYPE, issue #781), so the applied
# change equals the fp64 step up to fp64 round-off on the same data.
_PARAMETER_TOLERANCE = (ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE)


@dataclass(frozen=True)
class _Case:
    epsilon: float
    use_device_optimizer: bool
    opt_J2_param: bool
    use_lm: bool
    shipped_defaults: bool = False


_CASES = {
    "sr_eps_small_host": _Case(epsilon=1.0e-3, use_device_optimizer=False, opt_J2_param=True, use_lm=False),
    "sr_eps_large_device": _Case(epsilon=0.2, use_device_optimizer=True, opt_J2_param=True, use_lm=False),
    "sr_j2_frozen_host": _Case(epsilon=0.2, use_device_optimizer=False, opt_J2_param=False, use_lm=False),
    "lm_full_host": _Case(epsilon=1.0e-3, use_device_optimizer=False, opt_J2_param=True, use_lm=True),
}
_SR_CASES = ("sr_eps_small_host", "sr_eps_large_device", "sr_j2_frozen_host")


def _shipped_optimizer_kwargs() -> dict:
    """The optimizer_kwargs every generated user input ships with, read, never re-encoded."""
    return copy.deepcopy(jqmc_miscs.cli_parameters["vmc"]["optimizer_kwargs"])


# Reviewed published configuration, not a computed physics result.
# Deliberate default changes require reviewing this snapshot; never regenerate
# it from the production table inside the test. cg_tol: relative tolerance
# 1e-5 (issue #859).
_REVIEWED_SHIPPED_OPTIMIZER_DEFAULTS = {
    "method": "sr",
    "delta": 0.15,
    "epsilon": 0.001,
    "cg_flag": True,
    "cg_max_iter": 10000,
    "cg_tol": 1e-5,
    "use_lm": True,
    "lm_subspace_dim": 0,
    "lm_cond": 0.001,
}


def _assert_shipped_optimizer_default_snapshot(actual):
    assert actual == _REVIEWED_SHIPPED_OPTIMIZER_DEFAULTS, "shipped optimizer default snapshot changed"
    assert {key: type(value) for key, value in actual.items()} == {
        key: type(value) for key, value in _REVIEWED_SHIPPED_OPTIMIZER_DEFAULTS.items()
    }, "shipped optimizer default types changed"


@pytest.mark.light
def test_shipped_optimizer_defaults_equal_reviewed_snapshot():
    """The published optimizer configuration exactly preserves its reviewed keys, values and scalar types.

    This configuration contract does not pin calculated scientific results or
    claim that these defaults are optimal for every system. Solver correctness
    and option propagation have separate sampled-batch reference tests.
    """
    _assert_shipped_optimizer_default_snapshot(_shipped_optimizer_kwargs())


@pytest.mark.light
@pytest.mark.parametrize(
    "fault",
    [
        "method",
        "delta",
        "epsilon",
        "cg_flag",
        "cg_max_iter",
        "cg_tol",
        "use_lm",
        "lm_subspace_dim",
        "lm_cond",
        "missing-key",
        "extra-key",
        "bool-as-int",
    ],
)
def test_shipped_optimizer_default_snapshot_rejects_drift(fault, monkeypatch):
    """Changing one published default, key or boolean type fails the exact configuration contract."""
    changed = copy.deepcopy(_REVIEWED_SHIPPED_OPTIMIZER_DEFAULTS)
    if fault == "missing-key":
        del changed["delta"]
    elif fault == "extra-key":
        changed["unreviewed_option"] = True
    elif fault == "bool-as-int":
        changed["cg_flag"] = 1
    elif fault == "method":
        changed[fault] = "unreviewed_method"
    elif isinstance(changed[fault], bool):
        changed[fault] = not changed[fault]
    else:
        changed[fault] += 1
    monkeypatch.setitem(jqmc_miscs.cli_parameters["vmc"], "optimizer_kwargs", changed)
    with pytest.raises(AssertionError, match="shipped optimizer default"):
        test_shipped_optimizer_defaults_equal_reviewed_snapshot()


# The shipped-default case takes epsilon / use_lm from the shipped table; the
# remaining fields are the host J1+J2 fixture of the other cases.
_SHIPPED_CASE = "shipped_defaults_host"
# Reviewed public defaults, independent of the optimizer's runtime constants.
_SHIPPED_ASR_GAMMA_MAX = 0.5
_SHIPPED_ASR_FALLBACK_GAMMA = 0.1
_EXTRA_CASES = {
    _SHIPPED_CASE: _Case(
        epsilon=float(_shipped_optimizer_kwargs()["epsilon"]),
        use_device_optimizer=False,
        opt_J2_param=True,
        use_lm=bool(_shipped_optimizer_kwargs()["use_lm"]),
        shipped_defaults=True,
    )
}


@dataclass(frozen=True)
class _Capture:
    case: _Case
    hamiltonian_before: Hamiltonian_data
    batch: object
    coordinate_map: object
    target_kwargs: dict
    target: object
    step: object
    r_up: np.ndarray
    r_dn: np.ndarray
    params_before: np.ndarray
    params_after: np.ndarray
    weights: np.ndarray
    local_energies: np.ndarray
    log_derivatives: np.ndarray
    local_energy_derivatives: np.ndarray | None


_CAPTURE_CACHE: dict[tuple[str, bool], _Capture] = {}


def _build_hamiltonian() -> Hamiltonian_data:
    structure_data, _aos, _mos_up, _mos_dn, geminal_mo_data, coulomb_potential_data = read_trexio_file(
        trexio_file=_TREXIO_FILE, store_tuple=True
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=_J1_START,
            structure_data=structure_data,
            core_electrons=tuple(0.0 for _ in structure_data.atomic_numbers),
            jastrow_1b_type="pade",
        ),
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=_J2_START, jastrow_2b_type="pade"
        ),
        jastrow_three_body_data=None,
    )
    return Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data),
    )


def _jastrow_parameters(hamiltonian_data: Hamiltonian_data) -> np.ndarray:
    jastrow = hamiltonian_data.wavefunction_data.jastrow_data
    return np.asarray(
        [
            np.asarray(jastrow.jastrow_one_body_data.jastrow_1b_param),
            np.asarray(jastrow.jastrow_two_body_data.jastrow_2b_param),
        ]
    )


def _host_coordinates(coordinate_map, batch, sources) -> np.ndarray:
    if isinstance(batch, DeviceOptimizationBatch):
        return np.asarray(coordinate_map.transform_derivatives_device(sources))
    return np.asarray(coordinate_map.transform_derivatives(sources))


def _capture(name: str) -> _Capture:
    """Run (once per process and precision mode) one production optimization step."""
    key = (name, is_mixed_precision_enabled())
    if key in _CAPTURE_CACHE:
        return _CAPTURE_CACHE[key]
    case = _CASES[name] if name in _CASES else _EXTRA_CASES[name]
    hamiltonian_before = _build_hamiltonian()
    mcmc = MCMC(
        hamiltonian_data=hamiltonian_before,
        Dt=2.0,
        mcmc_seed=_SEED,
        num_walkers=_NUM_WALKERS,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=case.use_lm,
    )
    recorded: dict[str, object] = {}
    positions_up: list[np.ndarray] = []
    positions_dn: list[np.ndarray] = []
    production_target = optimizer_module.compute_real_coordinate_optimization_target
    production_finalize = optimizer_module.finalize_optimization_step
    production_sample = mcmc.sample

    def recording_target(batch, coordinate_map, **kwargs):
        target = production_target(batch, coordinate_map, **kwargs)
        recorded.update(batch=batch, coordinate_map=coordinate_map, target_kwargs=dict(kwargs), target=target)
        return target

    def recording_finalize(target, coordinate_map, **kwargs):
        result = production_finalize(target, coordinate_map, **kwargs)
        recorded["step"] = result
        return result

    def recording_sample(*, num_steps, max_time):
        for _ in range(num_steps):
            production_sample(num_steps=1, max_time=max_time)
            positions_up.append(np.asarray(mcmc._MCMC__latest_r_up_carts))
            positions_dn.append(np.asarray(mcmc._MCMC__latest_r_dn_carts))

    if case.shipped_defaults:
        optimizer_kwargs = _shipped_optimizer_kwargs()
    else:
        optimizer_kwargs = {"method": "sr", "delta": _DELTA, "epsilon": case.epsilon, "cg_flag": False}
        if case.use_lm:
            optimizer_kwargs.update(use_lm=True, lm_subspace_dim=-1)
    mcmc.sample = recording_sample
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(optimizer_module, "compute_real_coordinate_optimization_target", recording_target)
        patch.setattr(optimizer_module, "finalize_optimization_step", recording_finalize)
        mcmc.run_optimize(
            num_mcmc_steps=_NUM_STEPS,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=_NUM_BIN_BLOCKS,
            optimizer_kwargs=optimizer_kwargs,
            opt_J1_param=True,
            opt_J2_param=case.opt_J2_param,
            opt_J3_param=False,
            dump_wavefunction=False,
            use_device_optimizer=case.use_device_optimizer,
        )
    batch = recorded["batch"]
    coordinate_map = recorded["coordinate_map"]
    energy_sources = batch.local_energy_derivative_sources
    capture = _Capture(
        case=case,
        hamiltonian_before=hamiltonian_before,
        batch=batch,
        coordinate_map=coordinate_map,
        target_kwargs=recorded["target_kwargs"],
        target=recorded["target"],
        step=recorded["step"],
        r_up=np.stack(positions_up).reshape(_NUM_STEPS * _NUM_WALKERS, *positions_up[0].shape[1:]),
        r_dn=np.stack(positions_dn).reshape(_NUM_STEPS * _NUM_WALKERS, *positions_dn[0].shape[1:]),
        params_before=_jastrow_parameters(hamiltonian_before),
        params_after=_jastrow_parameters(mcmc.hamiltonian_data),
        weights=np.asarray(batch.weights),
        local_energies=np.asarray(batch.local_energies),
        log_derivatives=_host_coordinates(coordinate_map, batch, batch.log_derivative_sources),
        local_energy_derivatives=(None if energy_sources is None else _host_coordinates(coordinate_map, batch, energy_sources)),
    )
    _CAPTURE_CACHE[key] = capture
    return capture


# --------------------------------------------------------------------------- #
# Independent evaluation at the stored configurations
# --------------------------------------------------------------------------- #


def _with_parameters(hamiltonian_data: Hamiltonian_data, parameters) -> Hamiltonian_data:
    wavefunction = hamiltonian_data.wavefunction_data
    jastrow = wavefunction.jastrow_data
    jastrow = jastrow.replace(
        jastrow_one_body_data=jastrow.jastrow_one_body_data.replace(jastrow_1b_param=parameters[0]),
        jastrow_two_body_data=jastrow.jastrow_two_body_data.replace(jastrow_2b_param=parameters[1]),
    )
    return hamiltonian_data.replace(wavefunction_data=wavefunction.replace(jastrow_data=jastrow))


def _independent_evaluator(capture: _Capture):
    base = capture.hamiltonian_before
    rotation = jnp.eye(3)

    def ln_psi(parameters, r_up, r_dn):
        return evaluate_ln_wavefunction(_with_parameters(base, parameters).wavefunction_data, r_up, r_dn)

    def local_energy(parameters, r_up, r_dn):
        return compute_local_energy(_with_parameters(base, parameters), r_up, r_dn, rotation)

    batched = {
        "ln_psi": jax.jit(jax.vmap(ln_psi, in_axes=(None, 0, 0))),
        "local_energy": jax.jit(jax.vmap(local_energy, in_axes=(None, 0, 0))),
        "d_ln_psi": jax.jit(jax.vmap(jax.grad(ln_psi), in_axes=(None, 0, 0))),
        "d_local_energy": jax.jit(jax.vmap(jax.grad(local_energy), in_axes=(None, 0, 0))),
    }

    def evaluate(name, parameters):
        return np.asarray(batched[name](jnp.asarray(parameters, dtype=jnp.float64), capture.r_up, capture.r_dn))

    return evaluate


def _active_columns(case: _Case) -> list[int]:
    return [0, 1] if case.opt_J2_param else [0]


def _fixed_sample_energy(capture: _Capture, evaluate, parameters) -> float:
    """Reweighted energy of the stored configurations at ``parameters``."""
    ln_start = evaluate("ln_psi", capture.params_before)
    ln_new = evaluate("ln_psi", parameters)
    ratio = capture.weights * np.exp(2.0 * (ln_new - ln_start))
    return float(ratio @ evaluate("local_energy", parameters) / np.sum(ratio))


# --------------------------------------------------------------------------- #
# Dense NumPy references
# --------------------------------------------------------------------------- #


def _reference_sr_update(log_derivatives, local_energies, weights, epsilon) -> np.ndarray:
    """Solve ``(S + epsilon diag(S)) q = f`` exactly on the weighted, centred samples.

    Moments and solve are extended precision (``_exact_sr_reference``), so the
    reference carries no ``cond * u`` rounding of its own.
    """
    O = np.real(np.asarray(log_derivatives, dtype=np.complex128))
    e = np.asarray(local_energies, dtype=np.float64)
    S, f = exact_centred_moments(O, e, weights)
    return exact_sr_solve(S, f, epsilon)


def _reference_lm_update(log_derivatives, energy_derivatives, local_energies, weights) -> tuple[np.ndarray, float, float]:
    """Dense generalized eigensolve of the non-symmetric centred LM pencil."""
    O = np.real(np.asarray(log_derivatives, dtype=np.complex128))
    D = np.real(np.asarray(energy_derivatives, dtype=np.complex128))
    e = np.asarray(local_energies, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = np.sum(w)
    e_bar = (w @ e) / total
    dO = O - (w @ O) / total
    D_bar = (w @ D) / total
    dD = D - D_bar
    de = e - e_bar
    S = np.einsum("n,na,nb->ab", w, dO, dO) / total
    cov = np.einsum("n,na,n->a", w, dO, de) / total
    K = np.einsum("n,n,na,nb->ab", w, e, dO, dO) / total
    B = np.einsum("n,na,nb->ab", w, dD, dO) / total
    size = O.shape[1]
    H = np.zeros((size + 1, size + 1))
    metric = np.zeros_like(H)
    H[0, 0] = e_bar
    H[1:, 0] = cov  # ket column <dO_a de>
    H[0, 1:] = cov + D_bar  # bra row <dO_a de> + <de_L/dc_a>
    H[1:, 1:] = K + B.T
    metric[0, 0] = 1.0
    metric[1:, 1:] = S
    inverse_sqrt = 1.0 / np.sqrt(np.diag(S))
    correlation = inverse_sqrt[:, None] * S * inverse_sqrt[None, :]
    assert np.linalg.eigvalsh(correlation)[0] >= OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF, (
        "precondition: no LM coordinate is removed by the conditioning cutoff"
    )
    eigenvalues, vectors = scipy.linalg.eig(H, metric)
    vectors = np.real(vectors)
    norms = np.sqrt(np.einsum("ij,ik,kj->j", vectors, metric, vectors))
    vectors = vectors / norms[None, :]
    weights_ref = np.where(np.abs(np.imag(eigenvalues)) == 0.0, vectors[0] ** 2, -np.inf)
    best = int(np.argmax(weights_ref))
    return vectors[1:, best] / vectors[0, best], float(np.real(eigenvalues[best])), float(weights_ref[best])


# --------------------------------------------------------------------------- #
# Verdict helpers (shared by the positive checks and the negative controls)
# --------------------------------------------------------------------------- #


def _assert_update_matches_reference(actual_q, reference_q, label: str) -> None:
    # Batch arrays and the host/device solve are fp64 in both precision modes,
    # so the comparison is fp64 round-off on the same data.
    np.testing.assert_allclose(
        np.asarray(actual_q),
        np.asarray(reference_q),
        rtol=RTOL_PYTHON_FP64_LOOSE,
        atol=ATOL_PYTHON_FP64_LOOSE,
        err_msg=f"production {label} update differs from the dense reference",
    )


def _assert_frozen_parameters_unchanged(before, after, active_columns) -> None:
    frozen = [index for index in range(len(_PARAMETER_NAMES)) if index not in active_columns]
    for index in frozen:
        assert np.array_equal(np.asarray(after[index]), np.asarray(before[index])), (
            f"frozen block {_PARAMETER_NAMES[index]} moved: {before[index]!r} -> {after[index]!r}"
        )


def _assert_descent(capture: _Capture, evaluate, step) -> tuple[float, float, float]:
    """Assert the fixed-sample energy drops along ``step`` and rises along ``-step``."""
    atol, rtol = get_tolerance_min(_SOURCE_ZONES, "strict")
    start = np.asarray(capture.params_before, dtype=np.float64)
    energy_start = _fixed_sample_energy(capture, evaluate, start)
    energy_forward = _fixed_sample_energy(capture, evaluate, start + step)
    energy_backward = _fixed_sample_energy(capture, evaluate, start - step)
    margin = atol + rtol * abs(energy_start)
    assert energy_forward < energy_start - margin, (
        f"fixed-sample energy did not decrease along the applied step: "
        f"E(c0)={energy_start:.12f}, E(c0+step)={energy_forward:.12f}, margin={margin:.3e}"
    )
    assert energy_backward > energy_start + margin, (
        f"fixed-sample energy did not increase along the reflected step: "
        f"E(c0)={energy_start:.12f}, E(c0-step)={energy_backward:.12f}, margin={margin:.3e}"
    )
    return energy_start, energy_forward, energy_backward


def _applied_step(capture: _Capture) -> np.ndarray:
    return np.asarray(capture.params_after, dtype=np.float64) - np.asarray(capture.params_before, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Positive checks
# --------------------------------------------------------------------------- #


@pytest.mark.light
@pytest.mark.parametrize("name", list(_CASES))
def test_real_batch_sources_match_independent_evaluation(name):
    """The consumed batch equals e_L, dlnPsi/dc (and de_L/dc) evaluated at the stored configurations."""
    capture = _capture(name)
    evaluate = _independent_evaluator(capture)
    columns = _active_columns(capture.case)
    atol, rtol = get_tolerance_min(_SOURCE_ZONES, "strict")

    assert [block.name for block in capture.coordinate_map.blocks] == [_PARAMETER_NAMES[i] for i in columns]
    assert capture.log_derivatives.shape == (_NUM_STEPS * _NUM_WALKERS, len(columns))
    np.testing.assert_allclose(capture.local_energies, evaluate("local_energy", capture.params_before), rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        np.real(capture.log_derivatives), evaluate("d_ln_psi", capture.params_before)[:, columns], rtol=rtol, atol=atol
    )
    np.testing.assert_array_equal(np.imag(capture.log_derivatives), np.zeros_like(np.real(capture.log_derivatives)))
    if capture.case.use_lm:
        np.testing.assert_allclose(
            np.real(capture.local_energy_derivatives),
            evaluate("d_local_energy", capture.params_before)[:, columns],
            rtol=rtol,
            atol=atol,
        )
    assert np.all(capture.weights > 0.0)


@pytest.mark.light
@pytest.mark.parametrize("name", _SR_CASES)
def test_run_optimize_sr_update_matches_dense_recomputation(name):
    """run_optimize's SR target and applied change equal a dense solve of (S + eps diag S) q = f."""
    capture = _capture(name)
    case = capture.case
    columns = _active_columns(case)
    reference_q = _reference_sr_update(capture.log_derivatives, capture.local_energies, capture.weights, case.epsilon)

    # Preconditions: no coordinate falls under the production diagonal floor,
    # the wide route is used, and the batch route is the requested one.
    assert np.all(np.asarray(capture.target.metric_diagonal) > min_S_diag_abs)
    assert capture.target.frozen_coordinate_groups == ()
    assert isinstance(capture.batch, DeviceOptimizationBatch) == case.use_device_optimizer
    assert capture.target_kwargs["epsilon"] == case.epsilon

    _assert_update_matches_reference(capture.target.q, reference_q, "SR")
    _assert_update_matches_reference(capture.step.q, _DELTA * reference_q, "scaled SR")

    atol, rtol = _PARAMETER_TOLERANCE
    np.testing.assert_allclose(_applied_step(capture)[columns], _DELTA * reference_q, rtol=rtol, atol=atol)
    if len(columns) < len(_PARAMETER_NAMES):  # a frozen block exists and must stay bit-identical
        _assert_frozen_parameters_unchanged(capture.params_before, capture.params_after, columns)


@pytest.mark.light
def test_run_optimize_frozen_block_restricts_the_solved_system():
    """With J2 frozen, J2 is bit-identical and q is the J1-only solve, not the J1 entry of the full solve."""
    capture = _capture("sr_j2_frozen_host")
    evaluate = _independent_evaluator(capture)
    both = evaluate("d_ln_psi", capture.params_before)
    restricted = _reference_sr_update(both[:, [0]], capture.local_energies, capture.weights, capture.case.epsilon)
    unrestricted = _reference_sr_update(both, capture.local_energies, capture.weights, capture.case.epsilon)

    # The two candidate systems give distinguishable J1 updates on this batch.
    atol, rtol = get_tolerance_min(_SOURCE_ZONES, "strict")
    assert abs(unrestricted[0] - restricted[0]) > atol + rtol * abs(restricted[0])
    np.testing.assert_allclose(capture.target.q, restricted, rtol=rtol, atol=atol)
    _assert_frozen_parameters_unchanged(capture.params_before, capture.params_after, [0])


@pytest.mark.light
@pytest.mark.parametrize("lm_cond", [0.0, -1.0e-3, np.nan, 1.0])
def test_run_optimize_rejects_lm_condition_cutoff_outside_open_unit_interval(lm_cond):
    """run_optimize rejects lm_cond outside (0, 1) before sampling and leaves the wavefunction unchanged."""
    hamiltonian_before = _build_hamiltonian()
    mcmc = MCMC(
        hamiltonian_data=hamiltonian_before,
        Dt=2.0,
        mcmc_seed=_SEED,
        num_walkers=_NUM_WALKERS,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=True,
    )
    with pytest.raises(ValueError, match=r"lm_cond must be .*open interval \(0, 1\)"):
        mcmc.run_optimize(
            num_mcmc_steps=_NUM_STEPS,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=_NUM_BIN_BLOCKS,
            optimizer_kwargs={"method": "sr", "delta": _DELTA, "use_lm": True, "lm_subspace_dim": -1, "lm_cond": lm_cond},
            opt_J1_param=True,
            opt_J2_param=True,
            opt_J3_param=False,
            dump_wavefunction=False,
        )
    np.testing.assert_array_equal(_jastrow_parameters(mcmc.hamiltonian_data), _jastrow_parameters(hamiltonian_before))


@pytest.mark.light
def test_run_optimize_lm_update_matches_dense_generalized_eigenproblem():
    """run_optimize's full-space LM target equals a dense eigensolve of the centred non-symmetric pencil."""
    capture = _capture("lm_full_host")
    reference_q, reference_energy, reference_weight = _reference_lm_update(
        capture.log_derivatives, capture.local_energy_derivatives, capture.local_energies, capture.weights
    )
    # Preconditions: the LM root was accepted (no aSR fallback) and it is not the SR step.
    assert capture.target.rejection_reason is None
    assert capture.target.accepted_scale == 1.0
    sr_q = _reference_sr_update(capture.log_derivatives, capture.local_energies, capture.weights, capture.case.epsilon)
    assert np.max(np.abs(reference_q - sr_q)) > ATOL_PYTHON_FP64_LOOSE + RTOL_PYTHON_FP64_LOOSE * np.max(np.abs(sr_q))

    _assert_update_matches_reference(capture.target.q, reference_q, "LM")
    _assert_update_matches_reference(capture.target.predicted_energy, reference_energy, "LM energy")
    _assert_update_matches_reference(capture.target.lm_reference_weight, reference_weight, "LM reference weight")
    atol, rtol = _PARAMETER_TOLERANCE
    np.testing.assert_allclose(_applied_step(capture), _DELTA * reference_q, rtol=rtol, atol=atol)


@pytest.mark.light
@pytest.mark.parametrize("name", list(_CASES))
def test_run_optimize_step_lowers_the_fixed_sample_energy(name):
    """The applied step lowers the reweighted energy of the stored configurations; its reflection raises it."""
    capture = _capture(name)
    evaluate = _independent_evaluator(capture)
    _assert_descent(capture, evaluate, _applied_step(capture))


# --------------------------------------------------------------------------- #
# Negative controls
# --------------------------------------------------------------------------- #


def _replay_target(capture: _Capture, **overrides):
    kwargs = dict(capture.target_kwargs)
    kwargs["warm_start"] = None
    kwargs.update(overrides)
    return optimizer_module.compute_real_coordinate_optimization_target(capture.batch, capture.coordinate_map, **kwargs)


@pytest.mark.light
def test_negative_control_flipped_force_fails_update_and_descent_checks(monkeypatch):
    """Negating the SR force in the production system builder fails the update and descent verdicts."""
    capture = _capture("sr_eps_small_host")
    evaluate = _independent_evaluator(capture)
    reference_q = _reference_sr_update(capture.log_derivatives, capture.local_energies, capture.weights, capture.case.epsilon)

    unmutated = _replay_target(capture)
    _assert_update_matches_reference(unmutated.q, reference_q, "SR")

    production_prepare = optimizer_module.prepare_wide_sr_real_system

    def flipped_force(X_q_local, F_local, *, coordinate_force=None, **kwargs):
        return production_prepare(X_q_local, F_local, coordinate_force=-np.asarray(coordinate_force), **kwargs)

    monkeypatch.setattr(optimizer_module, "prepare_wide_sr_real_system", flipped_force)
    mutated = _replay_target(capture)
    with pytest.raises(AssertionError, match="production SR update differs from the dense reference"):
        _assert_update_matches_reference(mutated.q, reference_q, "SR")
    with pytest.raises(AssertionError, match="fixed-sample energy did not decrease along the applied step"):
        _assert_descent(capture, evaluate, _DELTA * np.asarray(mutated.q))


@pytest.mark.light
def test_negative_control_dropped_epsilon_fails_update_check():
    """Solving without the epsilon regularization fails the dense-reference verdict."""
    capture = _capture("sr_eps_large_device")
    reference_q = _reference_sr_update(capture.log_derivatives, capture.local_energies, capture.weights, capture.case.epsilon)
    _assert_update_matches_reference(_replay_target(capture).q, reference_q, "SR")
    without_epsilon = _replay_target(capture, epsilon=0.0)
    with pytest.raises(AssertionError, match="production SR update differs from the dense reference"):
        _assert_update_matches_reference(without_epsilon.q, reference_q, "SR")


@pytest.mark.light
def test_negative_control_update_on_frozen_block_fails_mask_checks():
    """Moving the frozen J2 block, or solving the unrestricted system, fails the mask verdicts."""
    capture = _capture("sr_j2_frozen_host")
    evaluate = _independent_evaluator(capture)
    both = evaluate("d_ln_psi", capture.params_before)
    unrestricted = _reference_sr_update(both, capture.local_energies, capture.weights, capture.case.epsilon)

    moved = np.array(capture.params_after, copy=True)
    moved[1] = moved[1] + np.asarray(_DELTA * unrestricted[1], dtype=moved.dtype)
    with pytest.raises(AssertionError, match="frozen block j2_param moved"):
        _assert_frozen_parameters_unchanged(capture.params_before, moved, [0])
    with pytest.raises(AssertionError, match="production SR update differs from the dense reference"):
        _assert_update_matches_reference(capture.target.q, unrestricted[:1], "SR")


# --------------------------------------------------------------------------- #
# Shipped default optimizer configuration
# --------------------------------------------------------------------------- #


def _reference_asr_update(log_derivatives, energy_derivatives, local_energies, weights, sr_q, gamma_max, fallback_gamma):
    """Accelerated-SR step along the SR direction g, from the stored samples.

    The one-dimensional linear-method model along g uses the centred,
    weighted moments: H_0 = <e>, H_1 = <dO_g de> + <D_g>/2 (mean of the ket
    column <dO_g de> and the bra row <dO_g de> + <D_g>), H_2 = g (K + B^T) g
    with K = <e dO dO>, B = <dD dO>, and S_2 = g S g. The stationary points of
    E(gamma) = (H_0 + 2 gamma H_1 + gamma^2 H_2) / (1 + gamma^2 S_2) are the
    roots of -H_1 S_2 gamma^2 + (H_2 - H_0 S_2) gamma + H_1 = 0; the root with
    the smaller |gamma| (connected to the reference state) is selected, and a
    root beyond gamma_max is replaced by fallback_gamma with the root's sign.
    """
    O = np.real(np.asarray(log_derivatives, dtype=np.complex128))
    D = np.real(np.asarray(energy_derivatives, dtype=np.complex128))
    e = np.asarray(local_energies, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    g = np.asarray(sr_q, dtype=np.float64)
    total = np.sum(w)
    e_bar = (w @ e) / total
    dO = O - (w @ O) / total
    D_bar = (w @ D) / total
    dD = D - D_bar
    de = e - e_bar
    S = np.einsum("n,na,nb->ab", w, dO, dO) / total
    cov = np.einsum("n,na,n->a", w, dO, de) / total
    K = np.einsum("n,n,na,nb->ab", w, e, dO, dO) / total
    B = np.einsum("n,na,nb->ab", w, dD, dO) / total
    H_0 = e_bar
    H_1 = cov @ g + 0.5 * (D_bar @ g)
    H_2 = g @ (K + B.T) @ g
    S_2 = g @ S @ g
    roots = np.roots([-H_1 * S_2, H_2 - H_0 * S_2, H_1])
    assert np.all(np.isreal(roots)), roots
    roots = np.real(roots)
    gamma = float(roots[np.argmin(np.abs(roots))])
    capped = abs(gamma) > gamma_max
    if capped:
        gamma = float(np.copysign(fallback_gamma, gamma))
    energy = (H_0 + 2.0 * gamma * H_1 + gamma * gamma * H_2) / (1.0 + gamma * gamma * S_2)
    return gamma, float(energy), capped


@pytest.mark.light
def test_shipped_default_optimizer_configuration_reaches_the_target():
    """One real run_optimize step with the shipped optimizer_kwargs hands exactly those values to the target.

    The kwargs are a deep copy of jqmc_miscs.cli_parameters["vmc"]["optimizer_kwargs"]
    (not re-encoded here). At the target boundary the SR/CG/LM controls must
    equal the shipped values, and the applied step must be the shipped delta
    times target.q. The shipped combination is SR with CG and use_lm=True,
    lm_subspace_dim=0 (the accelerated-SR scaling of the SR direction).
    """
    shipped = _shipped_optimizer_kwargs()
    assert shipped["method"] == "sr"
    capture = _capture(_SHIPPED_CASE)
    kwargs = capture.target_kwargs
    observed = [
        kwargs["epsilon"],
        kwargs["cg"],
        kwargs["cg_max_iter"],
        kwargs["cg_tol"],
        kwargs["use_lm"],
        kwargs["lm_subspace_dim"],
        kwargs["lm_condition_cutoff"],
    ]
    expected = [
        float(shipped["epsilon"]),
        bool(shipped["cg_flag"]),
        int(shipped["cg_max_iter"]),
        float(shipped["cg_tol"]),
        bool(shipped["use_lm"]),
        int(shipped["lm_subspace_dim"]),
        float(shipped["lm_cond"]),
    ]
    assert observed == expected, f"shipped optimizer controls did not reach the target: {observed} != {expected}"
    _assert_update_matches_reference(capture.step.q, float(shipped["delta"]) * np.asarray(capture.target.q), "scaled step")
    atol, rtol = _PARAMETER_TOLERANCE
    np.testing.assert_allclose(
        _applied_step(capture), float(shipped["delta"]) * np.asarray(capture.target.q), rtol=rtol, atol=atol
    )


@pytest.mark.light
def test_shipped_default_asr_update_matches_dense_recomputation():
    """The shipped-default update equals gamma * q_SR recomputed from the stored samples.

    q_SR is the dense solve of (S + epsilon diag S) q = f with the shipped
    epsilon (production solves it by CG with the shipped cg_max_iter / cg_tol);
    gamma is the reference-connected stationary point of the one-dimensional
    LM model along q_SR, built independently from the stored log and
    local-energy derivatives (see _reference_asr_update). The applied change
    is the shipped delta times gamma q_SR. Preconditions: no coordinate falls
    below the diagonal floor and gamma is not the capped fallback, so the
    model root determines the step. Replaying the same batch with larger
    epsilon exercises the shipped cap and signed fallback independently.
    """
    shipped = _shipped_optimizer_kwargs()
    capture = _capture(_SHIPPED_CASE)
    evaluate = _independent_evaluator(capture)
    atol, rtol = get_tolerance_min(_SOURCE_ZONES, "strict")
    np.testing.assert_allclose(capture.local_energies, evaluate("local_energy", capture.params_before), rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        np.real(capture.log_derivatives), evaluate("d_ln_psi", capture.params_before), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        np.real(capture.local_energy_derivatives), evaluate("d_local_energy", capture.params_before), rtol=rtol, atol=atol
    )
    assert np.all(np.asarray(capture.target.metric_diagonal) > min_S_diag_abs)
    assert capture.target.frozen_coordinate_groups == ()

    sr_q = _reference_sr_update(capture.log_derivatives, capture.local_energies, capture.weights, float(shipped["epsilon"]))
    _assert_update_matches_reference(capture.target.sr_q, sr_q, "shipped-default SR direction")
    gamma, energy, capped = _reference_asr_update(
        capture.log_derivatives,
        capture.local_energy_derivatives,
        capture.local_energies,
        capture.weights,
        sr_q,
        _SHIPPED_ASR_GAMMA_MAX,
        _SHIPPED_ASR_FALLBACK_GAMMA,
    )
    assert not capped, f"precondition: the aSR root {gamma} is not the capped fallback"
    assert capture.target.rejection_reason is None
    _assert_update_matches_reference(capture.target.accepted_scale, gamma, "aSR gamma")
    _assert_update_matches_reference(capture.target.predicted_energy, energy, "aSR predicted energy")
    _assert_update_matches_reference(capture.target.q, gamma * sr_q, "shipped-default aSR")
    param_atol, param_rtol = _PARAMETER_TOLERANCE
    np.testing.assert_allclose(_applied_step(capture), float(shipped["delta"]) * gamma * sr_q, rtol=param_rtol, atol=param_atol)

    # This epsilon puts the independent root beyond the cap on the same H2 batch.
    replay_epsilon = 10.0
    replay_sr_q = _reference_sr_update(capture.log_derivatives, capture.local_energies, capture.weights, replay_epsilon)
    fallback_gamma, fallback_energy, capped = _reference_asr_update(
        capture.log_derivatives,
        capture.local_energy_derivatives,
        capture.local_energies,
        capture.weights,
        replay_sr_q,
        _SHIPPED_ASR_GAMMA_MAX,
        _SHIPPED_ASR_FALLBACK_GAMMA,
    )
    assert capped, "precondition: the replayed root exceeds the documented aSR cap"
    replay = _replay_target(capture, epsilon=replay_epsilon)
    _assert_update_matches_reference(replay.accepted_scale, fallback_gamma, "capped aSR gamma")
    _assert_update_matches_reference(replay.predicted_energy, fallback_energy, "capped aSR predicted energy")
    _assert_update_matches_reference(replay.q, fallback_gamma * replay_sr_q, "shipped-default capped aSR")
    assert capture.target_kwargs["asr_gamma_max"] == _SHIPPED_ASR_GAMMA_MAX
    assert capture.target_kwargs["asr_fallback_gamma"] == _SHIPPED_ASR_FALLBACK_GAMMA


@pytest.mark.light
def test_shipped_default_step_lowers_the_fixed_sample_energy():
    """The shipped-default step lowers the reweighted energy of the stored configurations; its reflection raises it."""
    capture = _capture(_SHIPPED_CASE)
    evaluate = _independent_evaluator(capture)
    _assert_descent(capture, evaluate, _applied_step(capture))


@pytest.mark.light
def test_negative_control_shipped_default_checks_reject_changed_defaults(monkeypatch):
    """A changed shipped delta, or an update without the aSR scale, fails the shipped-default checks."""
    shipped = _shipped_optimizer_kwargs()
    capture = _capture(_SHIPPED_CASE)
    with pytest.raises(AssertionError, match="production scaled step update differs"):
        _assert_update_matches_reference(
            capture.step.q, 2.0 * float(shipped["delta"]) * np.asarray(capture.target.q), "scaled step"
        )
    sr_q = _reference_sr_update(capture.log_derivatives, capture.local_energies, capture.weights, float(shipped["epsilon"]))
    with pytest.raises(AssertionError, match="production shipped-default aSR update differs"):
        _assert_update_matches_reference(capture.target.q, sr_q, "shipped-default aSR")
    # A replay that ignores the shipped use_lm (plain SR) no longer matches the aSR reference.
    plain_sr = _replay_target(capture, use_lm=False)
    gamma, _energy, _capped = _reference_asr_update(
        capture.log_derivatives,
        capture.local_energy_derivatives,
        capture.local_energies,
        capture.weights,
        sr_q,
        _SHIPPED_ASR_GAMMA_MAX,
        _SHIPPED_ASR_FALLBACK_GAMMA,
    )
    with pytest.raises(AssertionError, match="production shipped-default aSR update differs"):
        _assert_update_matches_reference(plain_sr.q, gamma * sr_q, "shipped-default aSR")
