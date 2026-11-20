"""Actual two-rank twist optimizer updates against deterministic sampled-batch references.

The production communicator split and MPI collectives are not replaced. Only
passive spies capture inputs and outputs. Host/device refer to optimizer batch
implementations, not a claim that this CPU test ran on a GPU. Complex cases
optimize real J1/J2 coordinates and avoid unrelated complex-geminal conventions.
The Gamma rank of a complex case publishes real-typed arrays where the twisted
rank's are complex; the target spy checks, before the production collectives,
that every rank transforms on one real/complex route.
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

import dataclasses
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mpi4py import MPI

import jqmc.jqmc_optimizer as optimizer
from jqmc._precision import get_tolerance, get_tolerance_min
from jqmc._setting import (
    OPTIMIZER_DIAGNOSTICS_TOP_COORDINATES,
    ATOL_BIT_EXACT,
    ATOL_PYTHON_FP64_LOOSE,
    ATOL_PYTHON_FP64_TIGHT,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_PYTHON_FP64_ZERO,
)
from jqmc._twist_average import run_twist_average_optimize
from jqmc.jqmc_mcmc import MCMC
from jqmc.output_parser import parse_vmc_steps
from tests._test_data_cache import qe_example_files_root
from tests.test_jqmc_kernel import test_optimizer_real_batch_update as reference
from tests.test_jqmc_kernel import test_twist_optimize_update_value as sampled

pytestmark = [pytest.mark.mpi_only, pytest.mark.with_chdir]

_CASES = (
    "real-sr-host",
    "real-sr-device",
    "real-lm-host",
    "real-lm-device",
    "real-asr-host",
    "real-asr-device",
    "complex-sr-host",
    "complex-lm-device",
    "complex-asr-host",
)
_COMPLEX_FILES = ("H2_pbc_ae_cart.h5", "H2_pbc_ae_k025_cart.h5")


def _capture_mpi_step(case):
    family, method, batch_kind = case.split("-")
    world = MPI.COMM_WORLD
    assert world.size == 2, "this contract requires two actual MPI ranks"
    assert jax.process_count() == world.size
    files = sampled._TWIST_FILES if family == "real" else _COMPLEX_FILES
    hamiltonians = [sampled._build_hamiltonian(stem) for stem in files]
    if family == "complex":
        leaves = [ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data for ham in hamiltonians]
        np.testing.assert_array_equal(leaves[0].k_frac_np, np.zeros(3))
        np.testing.assert_allclose(
            leaves[1].k_frac_np,
            np.asarray([0.25, 0.25, 0.25]),
            atol=ATOL_PYTHON_FP64_TIGHT,
            rtol=RTOL_PYTHON_FP64_ZERO,
            err_msg="committed non-Gamma twist after Cartesian/fractional conversion",
        )
    capture = {"positions_up": [], "positions_dn": []}
    original_target = optimizer.compute_twist_decomposed_real_coordinate_optimization_target
    original_sample = MCMC.sample

    def recording_target(batch, coordinate_map, **kwargs):
        # Every rank must transform on one real/complex route before the
        # production collectives run; a rank-dependent route fails here on
        # every rank (allgathered flags) instead of hanging in a collective.
        world_comm = kwargs["world_comm"]
        local_routes = world_comm.allgather(tuple(optimizer._local_real_route_flags(batch, coordinate_map)))
        routed = optimizer._collective_transform_batch(batch, coordinate_map, world_comm)
        routed_routes = world_comm.allgather(tuple(optimizer._local_real_route_flags(routed, coordinate_map)))
        assert len(set(routed_routes)) == 1, f"ranks transform on different real/complex routes: {routed_routes}"
        target = original_target(batch, coordinate_map, **kwargs)
        capture.update(batch=batch, coordinate_map=coordinate_map, target=target, kwargs=kwargs, local_routes=local_routes)
        return target

    def recording_sample(self, *, num_steps, max_time):
        for _ in range(num_steps):
            original_sample(self, num_steps=1, max_time=max_time)
            capture["positions_up"].append(np.asarray(self._MCMC__latest_r_up_carts))
            capture["positions_dn"].append(np.asarray(self._MCMC__latest_r_dn_carts))

    settings = {"method": "sr", "delta": sampled._DELTA, "epsilon": sampled._EPSILON, "cg_flag": False}
    if method != "sr":
        settings.update(use_lm=True, lm_subspace_dim=-1 if method == "lm" else 0)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(optimizer, "compute_twist_decomposed_real_coordinate_optimization_target", recording_target)
        patch.setattr(MCMC, "sample", recording_sample)
        after = run_twist_average_optimize(
            hamiltonians,
            list(sampled._RAW_WEIGHTS),
            mcmc_seed=sampled._SEED,
            num_walkers=sampled._NUM_WALKERS,
            world_comm=world,
            mcmc_kwargs={"Dt": 2.0},
            optimize_kwargs=dict(
                num_mcmc_steps=sampled._NUM_STEPS,
                num_opt_steps=1,
                num_mcmc_warmup_steps=0,
                num_mcmc_bin_blocks=sampled._NUM_BIN_BLOCKS,
                optimizer_kwargs=settings,
                opt_J1_param=True,
                opt_J2_param=True,
                opt_J3_param=False,
                opt_geminal=False,
                dump_wavefunction=False,
                use_device_optimizer=batch_kind == "device",
            ),
        )
    batch, coordinate_map = capture["batch"], capture["coordinate_map"]
    assert isinstance(batch, optimizer.DeviceOptimizationBatch) == (batch_kind == "device")
    assert capture["kwargs"]["world_comm"].size == 2
    assert capture["kwargs"]["sampler_comm"].size == 1
    assert [block.name for block in coordinate_map.blocks] == ["j1_param", "j2_param"]
    assert capture["target"].frozen_coordinate_groups == ()
    local = dict(
        O=sampled._host_coordinates(coordinate_map, batch, batch.log_derivative_sources),
        D=(
            sampled._host_coordinates(coordinate_map, batch, batch.local_energy_derivative_sources)
            if batch.local_energy_derivative_sources is not None
            else None
        ),
        e=np.asarray(batch.local_energies),
        w=np.asarray(batch.weights),
        before=sampled._jastrow_parameters(hamiltonians[world.rank]),
        after=sampled._jastrow_parameters(after),
    )
    capture.update(local=local, all_data=world.allgather(local), hamiltonian=hamiltonians[world.rank])
    return capture


def _assert_sources(capture, family, method):
    local = capture["local"]
    n_samples = sampled._NUM_STEPS * sampled._NUM_WALKERS
    positions_up = np.stack(capture["positions_up"]).reshape(n_samples, -1, 3)
    positions_dn = np.stack(capture["positions_dn"]).reshape(n_samples, -1, 3)
    base = capture["hamiltonian"]

    def energy_at(parameters, r_up, r_dn):
        return sampled.compute_local_energy(sampled._with_parameters(base, parameters), r_up, r_dn, jnp.eye(3))

    def log_at(parameters, r_up, r_dn):
        return sampled.evaluate_ln_wavefunction(sampled._with_parameters(base, parameters).wavefunction_data, r_up, r_dn)

    parameters = jnp.asarray(local["before"])
    energy_ref = jax.jit(jax.vmap(energy_at, in_axes=(None, 0, 0)))(parameters, positions_up, positions_dn)
    log_ref = jax.jit(jax.vmap(jax.jacfwd(log_at), in_axes=(None, 0, 0)))(parameters, positions_up, positions_dn)
    atol, rtol = get_tolerance_min(sampled._SOURCE_ZONES, "strict")
    np.testing.assert_allclose(local["e"], energy_ref, atol=atol, rtol=rtol, err_msg="MPI twist consumed energy")
    np.testing.assert_allclose(local["O"], log_ref, atol=atol, rtol=rtol, err_msg="MPI twist consumed log derivative")
    if method != "sr":
        derivative_ref = jax.jit(jax.vmap(jax.jacfwd(energy_at), in_axes=(None, 0, 0)))(parameters, positions_up, positions_dn)
        np.testing.assert_allclose(
            local["D"], derivative_ref, atol=atol, rtol=rtol, err_msg="MPI twist consumed energy derivative"
        )
    if family == "complex":
        twisted_energy = capture["all_data"][1]["e"]
        assert np.max(np.abs(np.imag(twisted_energy))) > atol + rtol * np.max(np.abs(twisted_energy)), (
            "complex input must be live"
        )
        # The Gamma rank publishes real-typed arrays where the twisted rank's are
        # complex, so the collective route decision is live in this case.
        assert len(set(capture["local_routes"])) > 1, capture["local_routes"]


def _combined_samples(all_data, bz_weights):
    """Independent per-twist normalization of captured arrays for the fixed-BZ objective."""
    common_energy = sum(
        weight * np.real(data["w"] @ data["e"]) / data["w"].sum() for weight, data in zip(bz_weights, all_data, strict=True)
    )
    Os, Ds, es, ws = [], [], [], []
    for weight, data in zip(bz_weights, all_data, strict=True):
        norm = data["w"].sum()
        Os.append(data["O"] - data["w"] @ data["O"] / norm)
        Ds.append(data["D"] if data["D"] is not None else np.zeros_like(data["O"]))
        es.append(data["e"] - data["w"] @ data["e"] / norm + common_energy)
        ws.append(weight * data["w"] / norm)
    return tuple(map(np.concatenate, (Os, Ds, es, ws)))


def _assert_update(capture, method):
    all_data, target = capture["all_data"], capture["target"]
    weights = np.asarray(sampled._RAW_WEIGHTS) / np.sum(sampled._RAW_WEIGHTS)
    O, D, e, w = _combined_samples(all_data, weights)
    # These shared coordinates are real J1/J2 parameters, not complex geminal parameters.
    assert np.max(np.abs(np.imag(O))) == 0.0
    sr_q = reference._reference_sr_update(O, np.real(e), w, sampled._EPSILON)
    sampled._assert_update_matches(target.sr_q, sr_q, "actual MPI2 SR")
    expected = sr_q
    if method == "lm":
        expected, energy, weight = reference._reference_lm_update(O, D, np.real(e), w)
        assert target.rejection_reason is None, target.rejection_reason
        sampled._assert_update_matches(target.predicted_energy, energy, "actual MPI2 LM energy")
        sampled._assert_update_matches(target.lm_reference_weight, weight, "actual MPI2 LM reference weight")
    elif method == "asr":
        gamma, energy, capped = reference._reference_asr_update(
            O,
            D,
            np.real(e),
            w,
            sr_q,
            capture["kwargs"]["asr_gamma_max"],
            capture["kwargs"]["asr_fallback_gamma"],
        )
        assert not capped, "the reference-connected aSR root must be live and uncapped"
        expected = gamma * sr_q
        sampled._assert_update_matches(target.predicted_energy, energy, "actual MPI2 aSR energy")
    sampled._assert_update_matches(target.q, expected, f"actual MPI2 {method} target")
    atol, rtol = get_tolerance(sampled._PARAMETER_ZONE, "strict")
    local = capture["local"]
    np.testing.assert_allclose(
        local["after"] - local["before"],
        sampled._DELTA * expected,
        atol=atol,
        rtol=rtol,
        err_msg="actual MPI2 applied shared update",
    )
    np.testing.assert_array_equal(all_data[0]["after"], all_data[1]["after"])


@pytest.mark.heavy
@pytest.mark.parametrize("case", _CASES)
def test_actual_mpi_twist_optimizer_matches_sampled_reference(case):
    """Actual MPI2 SR, accepted LM and uncapped aSR updates equal independent same-batch models.

    Named real/complex inputs and host/device batches cover each implementation
    without asserting a full Cartesian universe. Captured analytic sources are
    checked against same-mode primal/AD values. No statistical acceptance band
    or comparison between precision modes is used.
    """
    capture = _capture_mpi_step(case)
    family, method, _batch = case.split("-")
    _assert_sources(capture, family, method)
    _assert_update(capture, method)


@pytest.mark.heavy
def test_actual_mpi_twist_optimizer_rejects_uniform_bz_weights(monkeypatch):
    """Replacing production BZ normalization by equal weights fails the independent MPI2 update reference."""
    monkeypatch.setattr(optimizer, "_normalize_planb_sampler_weight", lambda *_args, **_kwargs: 0.5)
    capture = _capture_mpi_step("real-sr-host")
    with pytest.raises(AssertionError, match="actual MPI2 SR differs from the fixed-BZ dense reference"):
        _assert_update(capture, "sr")


def _jackknife_standard_error(local_energies, weights, num_bin_blocks):
    """Weighted-ratio jackknife error over (bin, walker) blocks, written independently of ``MCMC.get_E``.

    The arrays are (step, walker) grids. Steps are split into ``num_bin_blocks``
    contiguous bins per walker; each leave-one-block-out estimate is a weighted
    energy ratio and the textbook jackknife variance is ``(n - 1) / n`` times
    the sum of squared deviations of the ``n`` leave-one-out estimates.
    """
    energies = np.real(np.asarray(local_energies, dtype=np.complex128))
    weights = np.asarray(weights, dtype=np.float64)
    bins = np.array_split(np.arange(energies.shape[0]), num_bin_blocks)
    bin_weights = np.concatenate([weights[rows].sum(axis=0) for rows in bins])
    bin_weighted_energies = np.concatenate([(weights[rows] * energies[rows]).sum(axis=0) for rows in bins])
    leave_one_out = (bin_weighted_energies.sum() - bin_weighted_energies) / (bin_weights.sum() - bin_weights)
    n = leave_one_out.size
    return float(np.sqrt((n - 1) / n * np.sum((leave_one_out - leave_one_out.mean()) ** 2)))


@pytest.mark.heavy
def test_actual_mpi_twist_lm_gate_uses_bz_error_for_shared_and_twist_error_for_local_pencil():
    """The two-rank LM driver gates the shared pencil with sigma_TA and the twist-local pencil with sigma_k.

    doc/notes/wavefunction_optimization.md ("Per-step twist-averaged energy and
    statistical error") defines sigma_TA = sqrt(sum_k W_k^2 sigma_k^2) for the
    shared Jastrow pencil and the twist's own jackknife sigma_k for its local
    determinant pencil. The actual MPI2 driver (J1/J2 shared, geminal local,
    raw weights 1:3) runs one LM step. Passive spies record the errors the
    driver hands to the twist-decomposed target; they must equal an independent jackknife
    of each twist's consumed (step, walker) energies and its BZ quadrature over
    both ranks. Only the dense LM eigensolve is replaced by a controlled
    candidate H_0 + c with c = 1.5 (sigma_TA + sigma_k), strictly between
    3 sigma_TA and 3 sigma_k, so the production per-scope gate must reject the
    shared candidate (aSR fallback gamma 0.37 times the SR direction) and accept
    the local one (zero candidate). Exchanging the two errors at the driver call
    fails both the identity and the per-scope decision.
    """
    world = MPI.COMM_WORLD
    assert world.size == 2, "this contract requires two actual MPI ranks"
    assert jax.process_count() == world.size
    hamiltonians = [sampled._build_hamiltonian(stem) for stem in sampled._TWIST_FILES]
    capture = {}
    controlled_gamma = 0.37
    original_target = optimizer.compute_twist_decomposed_real_coordinate_optimization_target

    def recording_target(batch, coordinate_map, **kwargs):
        capture.update(
            batch=batch,
            coordinate_map=coordinate_map,
            world_error=float(kwargs["world_energy_standard_error"]),
            sampler_error=float(kwargs["sampler_energy_standard_error"]),
        )
        target = original_target(batch, coordinate_map, **kwargs)
        capture["target"] = target
        return target

    def controlled_lm(pencil, *, condition_cutoff):
        del condition_cutoff
        size = pencil.H.shape[0] - 1
        # Symmetric in the two received errors, so an exchanged pair yields the same candidate.
        margin = 1.5 * (capture["world_error"] + capture["sampler_error"])
        capture.setdefault("lm_calls", 0)
        capture["lm_calls"] += 1
        return optimizer.LMRealResult(
            q=np.zeros(size),
            predicted_energy=pencil.H_0 + margin,
            reference_weight=1.0,
            active_indices=np.arange(size),
        )

    def controlled_asr(*coefficients, gamma_max, fallback_gamma):
        del coefficients, gamma_max, fallback_gamma
        return optimizer.ASRBoundedResult(gamma=controlled_gamma, predicted_energy=-1.0e9, used_fallback=False)

    settings = {
        "method": "sr",
        "delta": sampled._DELTA,
        "epsilon": sampled._EPSILON,
        "cg_flag": False,
        "use_lm": True,
        "lm_subspace_dim": -1,
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(optimizer, "compute_twist_decomposed_real_coordinate_optimization_target", recording_target)
        patch.setattr(optimizer, "solve_lm_real_pencil", controlled_lm)
        patch.setattr(optimizer, "select_asr_gamma_bounded", controlled_asr)
        run_twist_average_optimize(
            hamiltonians,
            list(sampled._RAW_WEIGHTS),
            mcmc_seed=sampled._SEED,
            num_walkers=sampled._NUM_WALKERS,
            world_comm=world,
            mcmc_kwargs={"Dt": 2.0},
            optimize_kwargs=dict(
                num_mcmc_steps=sampled._NUM_STEPS,
                num_opt_steps=1,
                num_mcmc_warmup_steps=0,
                num_mcmc_bin_blocks=sampled._NUM_BIN_BLOCKS,
                optimizer_kwargs=settings,
                opt_J1_param=True,
                opt_J2_param=True,
                opt_J3_param=False,
                opt_geminal=True,
                dump_wavefunction=False,
                use_device_optimizer=False,
            ),
        )

    batch, coordinate_map, target = capture["batch"], capture["coordinate_map"], capture["target"]
    grid = (sampled._NUM_STEPS, sampled._NUM_WALKERS)
    sigma_k = _jackknife_standard_error(
        np.asarray(batch.local_energies).reshape(grid),
        np.asarray(batch.weights).reshape(grid),
        sampled._NUM_BIN_BLOCKS,
    )
    # Collectives first, assertions afterwards, so a failing rank cannot strand its peer.
    all_sigma_k = np.asarray(world.allgather(sigma_k))
    bz_weights = np.asarray(sampled._RAW_WEIGHTS) / np.sum(sampled._RAW_WEIGHTS)
    sigma_ta = float(np.sqrt(np.sum((bz_weights * all_sigma_k) ** 2)))

    scopes = {"world": [], "sampler": []}
    for block, coordinate_slice in zip(coordinate_map.blocks, coordinate_map.block_coordinate_slices, strict=True):
        scopes[block.sync_scope].extend(range(coordinate_slice.start, coordinate_slice.stop))
    shared, local = (np.asarray(scopes[name], dtype=np.int64) for name in ("world", "sampler"))
    assert shared.size and local.size, "the case needs both a shared and a twist-local pencil"
    assert capture["lm_calls"] == 2, "one LM solve per scope"
    # Non-vacuity: the controlled margin separates the two documented thresholds.
    margin = 1.5 * (sigma_ta + sigma_k)
    assert 3.0 * sigma_ta < margin < 3.0 * sigma_k, (sigma_ta, all_sigma_k)

    np.testing.assert_allclose(
        [capture["world_error"], capture["sampler_error"]],
        [sigma_ta, sigma_k],
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_LOOSE,
        err_msg="driver errors handed to the shared (sigma_TA) and twist-local (sigma_k) LM gates",
    )

    sr_q = np.asarray(target.sr_q)
    assert np.any(sr_q[shared] != 0.0) and np.any(sr_q[local] != 0.0), "both SR directions must be live"
    expected_q = np.zeros_like(sr_q)
    expected_q[shared] = controlled_gamma * sr_q[shared]  # 3 sigma_TA < c: shared candidate rejected
    expected_q[local] = 0.0  # c < 3 sigma_k: twist-local candidate accepted
    np.testing.assert_allclose(
        np.asarray(target.q),
        expected_q,
        atol=ATOL_BIT_EXACT,
        rtol=RTOL_PYTHON_FP64_LOOSE,
        err_msg="per-scope LM energy-gate decision (shared rejected, twist-local accepted)",
    )
    assert target.rejection_reason.count("linear-method energy gate failed") == 1, target.rejection_reason


# Factor on twist 1's local forces (and so its signal-to-noise): the two twists
# then have deliberately different local forces and twist 1 leads the merge.
_TWIST1_LOCAL_FORCE_SCALE = 1.0e3


@pytest.mark.heavy
def test_actual_mpi_twist_step_diagnostics_cover_every_twist(caplog):
    """The world root logs the max force, max S/N, top rows and counts over shared and every twist's local coordinates.

    The actual MPI2 driver (two twists, one rank each) runs one SR step with
    shared J1/J2 and twist-local geminal coordinates at verbosity high. A
    passive spy scales twist 1's local forces by a fixed factor, so twist 1's
    local maximum is not on world rank 0. Each rank's target and coordinate
    map are allgathered, and the expected values are computed here from them:
    the max |force| and max S/N over the shared coordinates and both twists'
    local ones, the merged top-N rows labelled ``block[tK:qJ]`` for a local
    coordinate of twist K, one local max S/N line per twist, and the totals
    N_shared + sum_K N_local(K). The per-block lines must give a shared block
    once and a twist-local block once per twist (``block[tK]``) with that
    twist's own force, metric and update values (the step result is captured
    by a passive spy), and the all-twists coordinate line the shared plus every
    twist's real, storage, active and frozen counts from the frozen sets each
    twist used. Rank 0's log, read by jqmc.output_parser, must print exactly
    these.
    """
    world = MPI.COMM_WORLD
    assert world.size == 2, "this contract requires two actual MPI ranks"
    assert jax.process_count() == world.size
    hamiltonians = [sampled._build_hamiltonian(stem) for stem in sampled._TWIST_FILES]
    capture = {}
    original_target = optimizer.compute_twist_decomposed_real_coordinate_optimization_target

    def scaled_target(batch, coordinate_map, **kwargs):
        target = original_target(batch, coordinate_map, **kwargs)
        if world.rank == 1:
            local = [
                index
                for block, coordinate_slice in zip(coordinate_map.blocks, coordinate_map.block_coordinate_slices, strict=True)
                if block.sync_scope == "sampler"
                for index in range(coordinate_slice.start, coordinate_slice.stop)
            ]
            force = np.array(target.coordinate_force, copy=True)
            signal_to_noise = np.array(target.signal_to_noise, copy=True)
            force[local] *= _TWIST1_LOCAL_FORCE_SCALE
            signal_to_noise[local] *= _TWIST1_LOCAL_FORCE_SCALE
            target = dataclasses.replace(target, coordinate_force=force, signal_to_noise=signal_to_noise)
        capture.update(coordinate_map=coordinate_map, target=target)
        return target

    original_finalize = optimizer.finalize_optimization_step

    def recording_finalize(*args, **kwargs):
        capture["step_result"] = original_finalize(*args, **kwargs)
        return capture["step_result"]

    settings = {"method": "sr", "delta": sampled._DELTA, "epsilon": sampled._EPSILON, "cg_flag": False}
    with pytest.MonkeyPatch.context() as patch, caplog.at_level(logging.DEBUG, logger="jqmc"):
        patch.setattr(optimizer, "compute_twist_decomposed_real_coordinate_optimization_target", scaled_target)
        patch.setattr(optimizer, "finalize_optimization_step", recording_finalize)
        run_twist_average_optimize(
            hamiltonians,
            list(sampled._RAW_WEIGHTS),
            mcmc_seed=sampled._SEED,
            num_walkers=sampled._NUM_WALKERS,
            world_comm=world,
            mcmc_kwargs={"Dt": 2.0},
            optimize_kwargs=dict(
                num_mcmc_steps=sampled._NUM_STEPS,
                num_opt_steps=1,
                num_mcmc_warmup_steps=0,
                num_mcmc_bin_blocks=sampled._NUM_BIN_BLOCKS,
                optimizer_kwargs=settings,
                opt_J1_param=True,
                opt_J2_param=True,
                opt_J3_param=False,
                opt_geminal=True,
                dump_wavefunction=False,
                use_device_optimizer=False,
            ),
        )

    coordinate_map, target = capture["coordinate_map"], capture["target"]
    rows = []
    values = {"world": 0, "sampler": 0}
    for block, coordinate_slice in zip(coordinate_map.blocks, coordinate_map.block_coordinate_slices, strict=True):
        values[block.sync_scope] += int(block.size)
        for offset, index in enumerate(range(coordinate_slice.start, coordinate_slice.stop)):
            twist = "" if block.sync_scope == "world" else f"t{world.rank}:"
            rows.append(
                dict(
                    scope=block.sync_scope,
                    label=f"{block.name}[{twist}q{offset}]",
                    force=float(target.coordinate_force[index]),
                    force_std=float(target.coordinate_force_std[index]),
                    signal_to_noise=float(target.signal_to_noise[index]),
                    metric_diag=float(target.metric_diagonal[index]),
                )
            )
    step_result = capture["step_result"]
    frozen = [group for group in target.frozen_coordinate_groups]
    blocks = []
    counts = {scope: dict(real=0, storage=0, frozen=0, groups=0) for scope in ("world", "sampler")}
    for block, coordinate_slice, storage_slice, update in zip(
        coordinate_map.blocks,
        coordinate_map.block_coordinate_slices,
        coordinate_map.block_storage_slices,
        step_result.updates,
        strict=True,
    ):
        indices = set(range(coordinate_slice.start, coordinate_slice.stop))
        block_groups = [group for group in frozen if set(group) <= indices]
        scope_counts = counts[block.sync_scope]
        scope_counts["real"] += len(indices)
        scope_counts["storage"] += storage_slice.stop - storage_slice.start
        scope_counts["frozen"] += sum(len(group) for group in block_groups)
        scope_counts["groups"] += len(block_groups)
        force = np.asarray(target.coordinate_force)[coordinate_slice]
        metric = np.asarray(target.metric_diagonal)[coordinate_slice]
        delta_q = np.asarray(step_result.q)[coordinate_slice]
        delta_p = np.asarray(update.delta)
        blocks.append(
            dict(
                scope=block.sync_scope,
                block=block.name if block.sync_scope == "world" else f"{block.name}[t{world.rank}]",
                values=[
                    np.linalg.norm(force),
                    np.max(np.abs(force)),
                    np.max(np.asarray(target.signal_to_noise)[coordinate_slice]),
                    np.min(metric),
                    np.max(metric),
                    np.linalg.norm(delta_q),
                    np.max(np.abs(delta_q)),
                    np.linalg.norm(delta_p),
                    np.max(np.abs(delta_p)),
                ],
            )
        )
    # Collectives first, assertions afterwards, so a failing rank cannot strand its peer.
    all_rows = world.allgather(rows)
    all_values = world.allgather(values)
    all_blocks = world.allgather(blocks)
    all_counts = world.allgather(counts)
    if world.rank != 0:
        return

    shared = [row for row in all_rows[0] if row["scope"] == "world"]
    local = [[row for row in twist_rows if row["scope"] == "sampler"] for twist_rows in all_rows]
    assert shared and all(local), "the case needs shared and twist-local coordinates on both twists"
    merged = shared + local[0] + local[1]
    twist_maxima = [max(row["signal_to_noise"] for row in twist_rows) for twist_rows in local]
    # Non-vacuity: the merged maximum lies on twist 1, which rank 0's own target does not hold.
    assert twist_maxima[1] > max(row["signal_to_noise"] for row in all_rows[0])
    text = "\n".join(record.getMessage() for record in caplog.records)
    (step,) = parse_vmc_steps(text)

    expected_force = max(merged, key=lambda row: abs(row["force"]))
    expected_value = optimizer._format_value_and_error(expected_force["force"], expected_force["force_std"])
    assert [line for line in text.splitlines() if line.startswith("Max generalized force =")] == [
        f"Max generalized force = {expected_value} Ha/a.u."
    ]
    assert f"{step['signal_to_noise_ratio']:.3f}" == f"{max(row['signal_to_noise'] for row in merged):.3f}"
    assert [(entry["twist"], f"{entry['signal_to_noise']:.3f}") for entry in step["twist_local_max_signal_to_noise"]] == [
        (0, f"{twist_maxima[0]:.3f}"),
        (1, f"{twist_maxima[1]:.3f}"),
    ]

    top_count = min(OPTIMIZER_DIAGNOSTICS_TOP_COORDINATES, len(merged))
    expected_top = sorted(merged, key=lambda row: row["signal_to_noise"], reverse=True)[:top_count]
    assert [row["coordinate"] for row in step["top_coordinates"]] == [row["label"] for row in expected_top]
    assert [f"{row['signal_to_noise']:.6e}" for row in step["top_coordinates"]] == [
        f"{row['signal_to_noise']:.6e}" for row in expected_top
    ]
    assert any("[t1:q" in row["coordinate"] for row in step["top_coordinates"]), "twist 1's local rows must reach rank 0"

    assert step["num_optimization_coordinates"] == len(all_rows[0])
    assert step["num_optimization_coordinates_all_twists"] == len(shared) + len(local[0]) + len(local[1])
    assert step["num_variational_parameters_all_twists"] == (
        all_values[0]["world"] + all_values[0]["sampler"] + all_values[1]["sampler"]
    )

    # Rank 0's map order: a shared block once, a twist-local block once per twist.
    expected_blocks = []
    for position, entry in enumerate(all_blocks[0]):
        expected_blocks.extend([entry] if entry["scope"] == "world" else [twist[position] for twist in all_blocks])
    parsed_blocks = [
        [block[name] for name in ("force_norm", "max_abs_force", "max_signal_to_noise", "metric_diag_min", "metric_diag_max")]
        + [block["update"][name] for name in ("delta_q_norm", "max_abs_delta_q", "delta_p_norm", "max_abs_delta_p")]
        for block in step["blocks"]
    ]
    assert [block["block"] for block in step["blocks"]] == [entry["block"] for entry in expected_blocks]
    assert any(block["block"].endswith("[t1]") for block in step["blocks"]), "twist 1's local blocks must reach rank 0"
    assert [[f"{value:.6e}" for value in values] for values in parsed_blocks] == [
        [f"{value:.6e}" for value in entry["values"]] for entry in expected_blocks
    ]

    shared_counts = all_counts[0]["world"]
    real = shared_counts["real"] + sum(twist["sampler"]["real"] for twist in all_counts)
    frozen = shared_counts["frozen"] + sum(twist["sampler"]["frozen"] for twist in all_counts)
    assert step["diagnostics"]["coordinates_all_twists"] == dict(
        real=real,
        storage=shared_counts["storage"] + sum(twist["sampler"]["storage"] for twist in all_counts),
        active=real - frozen,
        frozen=frozen,
        atomic_groups=shared_counts["groups"] + sum(twist["sampler"]["groups"] for twist in all_counts),
    )
    assert step["diagnostics"]["coordinates"]["real"] == len(all_rows[0])


_QE_LI_K_TWIST = qe_example_files_root() / "Li_k_twist" / "out" / "li_twist.save"


@pytest.mark.heavy
def test_actual_mpi_twist_optimizer_updates_one_converted_shared_ao_j3(tmp_path):
    """A nonzero AO J3 from untiled QE twist conversion is sampled and optimized as one shared block.

    Each rank converts the committed Li_k_twist save (plane-wave determinant,
    ao-small J3) and loads twists 0 and 1, whose determinants carry different
    k. The converted zero J matrix is replaced on both twists by the same
    nonzero symmetric matrix. One actual MPI2 SR step with only J3 enabled
    must change J3, keep it symmetric and leave both twists with identical
    J3 values, while J2 and the twist-local determinants stay unchanged. A
    twist-local J3 update would sample different k and diverge.
    """
    from jqmc._twist_average import load_twist, load_twist_average_input
    from jqmc.jqmc_qe import run_qe2jqmc

    world = MPI.COMM_WORLD
    assert world.size == 2, "this contract requires two actual MPI ranks"
    assert jax.process_count() == world.size
    path = str(tmp_path / "li_pw_j3_twists.h5")
    run_qe2jqmc(
        {"qe_save_dir": str(_QE_LI_K_TWIST), "kind": "pw", "twist_average": True},
        {"j2_parameter": 1.0, "j3_basis_type": "ao-small"},
        path,
    )
    meta = load_twist_average_input(path)
    hamiltonians = []
    for index in (0, 1):
        ham = load_twist(path, index)
        jastrow = ham.wavefunction_data.jastrow_data
        j3 = jastrow.jastrow_three_body_data
        matrix = np.random.default_rng(4417).uniform(-0.02, 0.02, size=j3.j_matrix.shape)
        matrix[:, :-1] = (matrix[:, :-1] + matrix[:, :-1].T) / 2
        jastrow = jastrow.replace(jastrow_three_body_data=j3.replace(j_matrix=matrix))
        hamiltonians.append(ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jastrow)))
    before = hamiltonians[world.rank].wavefunction_data
    after = run_twist_average_optimize(
        hamiltonians,
        list(meta.weights[:2]),
        mcmc_seed=sampled._SEED,
        num_walkers=sampled._NUM_WALKERS,
        world_comm=world,
        mcmc_kwargs={"Dt": 2.0},
        optimize_kwargs=dict(
            num_mcmc_steps=sampled._NUM_STEPS,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=sampled._NUM_BIN_BLOCKS,
            optimizer_kwargs={"method": "sr", "delta": sampled._DELTA, "epsilon": sampled._EPSILON, "cg_flag": False},
            opt_J1_param=False,
            opt_J2_param=False,
            opt_J3_param=True,
            opt_geminal=False,
            dump_wavefunction=False,
        ),
    ).wavefunction_data
    matrix_before = np.asarray(before.jastrow_data.jastrow_three_body_data.j_matrix)
    matrix_after = np.asarray(after.jastrow_data.jastrow_three_body_data.j_matrix)
    # Collectives first, assertions afterwards, so a failing rank cannot strand its peer.
    all_after = world.allgather(matrix_after)
    k_points = world.allgather(tuple(after.geminal_data.orb_data_up_spin.aos_data.k_point))
    assert k_points[0] != k_points[1], "the two twists must sample different determinants"
    assert np.any(matrix_after != matrix_before), "the nonzero shared J3 must receive the SR update"
    np.testing.assert_array_equal(matrix_after[:, :-1], matrix_after[:, :-1].T)
    np.testing.assert_array_equal(all_after[0], all_after[1])
    np.testing.assert_array_equal(
        after.jastrow_data.jastrow_two_body_data.jastrow_2b_param, before.jastrow_data.jastrow_two_body_data.jastrow_2b_param
    )
    np.testing.assert_array_equal(
        after.geminal_data.orb_data_up_spin.mo_coefficients, before.geminal_data.orb_data_up_spin.mo_coefficients
    )
