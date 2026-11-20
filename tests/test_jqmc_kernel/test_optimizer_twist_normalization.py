"""Fixed-BZ objective oracles; run with mpirun -np 2 (or 4) pytest.

The oracle explicitly normalizes each finite-state wavefunction. It does not
use production moment reductions to decide what the gradient should be.
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

import numpy as np
import pytest
from mpi4py import MPI

import jqmc.jqmc_optimizer as opt
from jqmc._setting import ATOL_C9_GAMMA_SR_UPDATE, ATOL_FD_VS_ANALYTIC_TIGHT, ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_ZERO

from .optimizer_metadata_helpers import TEST_STATE_TOKEN, make_test_block
from .test_optimizer_lm_real import _device_batch

pytestmark = [
    pytest.mark.without_chdir,
    pytest.mark.mpi_only,
    pytest.mark.skip_serial(reason="twist normalization needs an MPI world split into two twists"),
]


@pytest.fixture
def twist_comms():
    world = MPI.COMM_WORLD
    assert world.size % 2 == 0, "splitting the world into two twists needs an even number of ranks"
    k = world.rank // (world.size // 2)
    sampler = world.Split(k, world.rank)
    yield world, sampler, k
    sampler.Free()


def _batch(o, e):
    block = make_test_block("shared_c", np.array(0.0), field="real", derivative_policy="direct_real", sync_scope="world")
    blocks = (block,)
    o, e = np.asarray(o), np.asarray(e)

    def source(a):
        return opt.BlockDerivativeSource.build(block=block, direct=a)

    moment = opt.JackknifeBlockMoments.build(block=block, source_sum=source(o), conjugate_source_energy_sum=source(o * e))
    metadata = opt.JackknifeBinMetadata.build(
        blocks=blocks,
        batch_token="finite-twist",
        bin_sample_counts=np.ones(e.size, dtype=int),
        bin_weight_sums=np.ones(e.size),
        bin_weighted_energy_sums=e,
        log_source_moments=(moment,),
    )
    batch = opt.OptimizationBatch.build(
        blocks=blocks,
        batch_token="finite-twist",
        state_token=TEST_STATE_TOKEN,
        weights=np.ones(e.size),
        local_energies=e,
        log_derivative_sources=(source(o),),
        local_energy_derivative_sources=(source(np.zeros_like(o)),),
        jackknife_bin_metadata=metadata,
    )
    return blocks, batch, opt.OptimizationCoordinateMap.from_blocks(blocks)


def _target(o, e, comms, device, method="sr", weight=None):
    world, sampler, _k = comms
    blocks, batch, cmap = _batch(o, e)
    kwargs = {} if weight is None else {"sampler_weight": weight}
    return opt.compute_twist_decomposed_real_coordinate_optimization_target(
        _device_batch(blocks, batch) if device else batch,
        cmap,
        sampler_comm=sampler,
        world_comm=world,
        epsilon=0.01,
        diagonal_floor=1e-12,
        cg=False,
        cg_max_iter=100,
        cg_tol=1e-10,
        use_lm=method != "sr",
        lm_subspace_dim=0 if method == "asr" else 10,
        lm_condition_cutoff=1e-8,
        world_energy_standard_error=0.0,
        sampler_energy_standard_error=0.0,
        **kwargs,
    )


def _energy(c, observables, hamiltonians, weights):
    answer = 0.0
    for o, h, w in zip(observables, hamiltonians, weights, strict=True):
        amplitude = np.exp(c * np.asarray(o))
        answer += w * (amplitude @ (np.asarray(h) * amplitude)) / (amplitude @ amplitude)
    return answer


def _reference(o, h, weights):
    force = 0.0
    metric = 0.0
    for a, b, w in zip(o, h, weights, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        force -= 2 * w * (np.dot(a, b) / a.size - a.mean() * b.mean())
        metric += w * np.mean((a - a.mean()) ** 2)
    return force, metric


@pytest.mark.light
@pytest.mark.parametrize("device", [False, True])
@pytest.mark.parametrize("method", ["sr", "lm", "asr"])
def test_constant_per_twist_energy_has_zero_shared_force_and_update(twist_comms, device, method):
    """Different identity Hamiltonians have an exactly constant normalized objective."""
    o = (np.array([0.0, 2.0]), np.array([2.0, 4.0]))
    h = (np.array([0.0, 0.0]), np.array([2.0, 2.0]))
    k = twist_comms[2]
    for c in (-0.1, 0.0, 0.1):
        comparison_iteration = str(c)  # noqa: F841 - captured by the GDD iteration receipt
        np.testing.assert_allclose(_energy(c, o, h, [0.5, 0.5]), 1.0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    actual = _target(o[k], h[k], twist_comms, device, method)
    # _batch defines one shared real scalar parameter: one optimizer coordinate.
    np.testing.assert_allclose(
        actual.coordinate_force, np.zeros(1, dtype=np.float64), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
    )
    np.testing.assert_allclose(
        actual.metric_diagonal, np.ones(1, dtype=np.float64), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
    )
    np.testing.assert_allclose(
        actual.sr_q, np.zeros(1, dtype=np.float64), atol=ATOL_C9_GAMMA_SR_UPDATE, rtol=RTOL_PYTHON_FP64_ZERO
    )
    np.testing.assert_allclose(
        actual.q, np.zeros(1, dtype=np.float64), atol=ATOL_C9_GAMMA_SR_UPDATE, rtol=RTOL_PYTHON_FP64_ZERO
    )


@pytest.mark.light
@pytest.mark.parametrize("device", [False, True])
@pytest.mark.parametrize("weights", [(0.5, 0.5), (0.2, 0.8), (0.0, 1.0)])
def test_fixed_bz_gradient_matches_normalized_energy_difference(twist_comms, device, weights):
    """Unequal sample counts cannot replace fixed BZ weights in the derivative."""
    o = (np.array([0.0, 2.0]), np.array([2.0, 4.0]))
    h = (np.array([0.0, 1.0]), np.array([2.0, 4.0]))
    k = twist_comms[2]
    force, metric = _reference(o, h, weights)
    step = 1e-4
    # Richardson cancels the central difference's O(step**2) term.
    fd = lambda d: -(_energy(d, o, h, weights) - _energy(-d, o, h, weights)) / (2 * d)
    reference_fd = (4 * fd(step / 2) - fd(step)) / 3
    # The finite-difference oracle has subtraction error; production force
    # and metric are separately checked against the exact formula at 1e-12.
    np.testing.assert_allclose(reference_fd, force, atol=ATOL_FD_VS_ANALYTIC_TIGHT, rtol=RTOL_PYTHON_FP64_ZERO)
    assert abs(force) > 0.5  # An always-zero implementation must not pass.
    actual = _target(np.tile(o[k], k + 1), np.tile(h[k], k + 1), twist_comms, device, weight=weights[k])
    np.testing.assert_allclose(
        actual.coordinate_force, np.asarray([force]), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
    )
    np.testing.assert_allclose(
        actual.metric_diagonal, np.asarray([metric]), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
    )
    np.testing.assert_allclose(
        actual.sr_q, np.asarray([force / (1.01 * metric)]), atol=ATOL_C9_GAMMA_SR_UPDATE, rtol=RTOL_PYTHON_FP64_ZERO
    )


@pytest.mark.light
@pytest.mark.parametrize("device", [False, True])
@pytest.mark.parametrize("method", ["sr", "lm", "asr"])
def test_independent_twist_gauges_do_not_change_update(twist_comms, device, method):
    """Normalization gauges and independent energy origins cannot alter a fixed-BZ update."""
    o = (np.array([0.0, 1.0, 2.0]), np.array([2.0, 3.0, 5.0]))
    h = (np.array([0.0, 0.5, 1.0]), np.array([1.0, 2.0, 4.0]))
    k = twist_comms[2]
    weights = (0.3, 0.7)
    a = _target(o[k], h[k], twist_comms, device, method, weights[k])
    assert np.any(np.asarray(a.q) != 0.0)
    b = _target(o[k] + (7.0, -3.0)[k], h[k] + (11.0, -5.0)[k], twist_comms, device, method, weights[k])
    for name in ("coordinate_force", "coordinate_force_std", "metric_diagonal", "sr_q", "q"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), atol=ATOL_C9_GAMMA_SR_UPDATE, rtol=RTOL_PYTHON_FP64_ZERO)


@pytest.mark.light
@pytest.mark.parametrize("device", [False, True])
def test_shared_force_error_is_quadrature_of_twist_jackknives(twist_comms, device):
    """Compute delete-one-bin references independently inside each sampler."""
    o = (np.array([0.0, 1.0, 3.0]), np.array([2.0, 4.0, 5.0, 7.0]))
    h = (np.array([0.0, 2.0, 1.0]), np.array([3.0, 1.0, 4.0, 2.0]))
    weights = (0.25, 0.75)
    variance = 0.0
    for oi, hi, w in zip(o, h, weights, strict=True):
        oi = np.tile(oi, twist_comms[1].size)
        hi = np.tile(hi, twist_comms[1].size)
        replicas = []
        for i in range(oi.size):
            aa = np.delete(oi, i)
            bb = np.delete(hi, i)
            replicas.append(-2 * (np.mean(aa * bb) - aa.mean() * bb.mean()))
        replicas = np.array(replicas)
        variance += w * w * (oi.size - 1) / oi.size * np.sum((replicas - replicas.mean()) ** 2)
    k = twist_comms[2]
    a = _target(o[k], h[k], twist_comms, device, weight=weights[k])
    np.testing.assert_allclose(
        a.coordinate_force_std, np.asarray([np.sqrt(variance)]), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
    )


@pytest.mark.light
@pytest.mark.parametrize("device", [False, True])
@pytest.mark.parametrize("method", ["sr", "lm", "asr"])
def test_single_twist_planb_identity(twist_comms, device, method):
    """A replicated single distribution retains the ordinary one-twist result."""
    world = twist_comms[0]
    o = np.array([0.0, 1.0, 3.0])
    h = np.array([0.0, 2.0, 1.0])
    a = _target(o, h, (MPI.COMM_SELF, MPI.COMM_SELF, 0), device, method)
    b = _target(o, h, (world, world, 0), device, method)
    for name in ("coordinate_force", "metric_diagonal", "sr_q", "q"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), atol=ATOL_C9_GAMMA_SR_UPDATE, rtol=RTOL_PYTHON_FP64_ZERO)


@pytest.mark.light
@pytest.mark.parametrize("device", [False, True])
def test_twist_driver_produces_the_unequally_weighted_gradient(twist_comms, device, monkeypatch):
    """Dropping W anywhere between the twist driver and target changes the numerical force."""
    import jqmc.jqmc_mcmc as mcmc_module
    from jqmc._twist_average import run_twist_average_optimize

    facade = mcmc_module.MCMC.run_optimize
    o = (np.array([0.0, 2.0]), np.array([2.0, 4.0]))
    h = (np.array([0.0, 1.0]), np.array([2.0, 4.0]))
    expected, _metric = _reference(o, h, [0.2, 0.8])
    observed = []

    class ExactSampler:
        run_optimize = facade

        def __init__(self, hamiltonian_data, mpi_comm_in, world_comm_in, **kwargs):
            self.hamiltonian_data = hamiltonian_data
            self.comm = mpi_comm_in
            self.world_comm = world_comm_in
            self.i_opt = 0

    def consume_config(sampler, config):
        k = sampler.hamiltonian_data
        target = _target(o[k], h[k], (sampler.world_comm, sampler.comm, k), device, weight=config.sampler_weight)
        observed.append(target.coordinate_force)
        sampler.i_opt += config.num_opt_steps

    monkeypatch.setattr(mcmc_module, "MCMC", ExactSampler)
    monkeypatch.setattr(opt, "run_optimize", consume_config)
    run_twist_average_optimize(
        [0, 1],
        [1.0, 4.0],
        mcmc_seed=729,
        num_walkers=2,
        world_comm=twist_comms[0],
        optimize_kwargs={"num_opt_steps": 2, "wf_dump_freq": 1},
    )
    assert len(observed) == 2
    np.testing.assert_allclose(observed, np.full((2, 1), expected), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)


@pytest.mark.light
@pytest.mark.parametrize("device", [False, True])
def test_nonfinite_twist_local_sample_raises_on_every_twist(device, twist_comms):
    """A NaN twist-local SR sample on one twist raises the same ValueError on every rank of every twist, without a hang."""
    from dataclasses import replace

    from .test_optimizer_lm_real import _host_stage_log_sources, _lm_case

    world, sampler, k = twist_comms
    blocks, batch, _O_q, _D_q = _lm_case("mixed", scopes=("world", "sampler", "world"))
    if k == 1:
        # Block p1 is the only twist-local (sampler-scope) block: the shared
        # system of the other twist stays finite.
        poisoned = np.array(batch.log_derivative_sources[1].d_p, copy=True)
        poisoned[3] = np.nan
        sources = list(batch.log_derivative_sources)
        sources[1] = opt.BlockDerivativeSource.build(block=blocks[1], d_p=poisoned)
        batch = replace(batch, log_derivative_sources=tuple(sources))
    if device:
        batch = _host_stage_log_sources(blocks, _device_batch(blocks, batch))
    with pytest.raises(ValueError, match="wide SR sample arrays must be finite"):
        opt.compute_twist_decomposed_real_coordinate_optimization_target(
            batch,
            opt.OptimizationCoordinateMap.from_blocks(blocks),
            sampler_comm=sampler,
            world_comm=world,
            epsilon=0.2,
            diagonal_floor=0.0,
            cg=True,
            cg_max_iter=500,
            cg_tol=1e-12,
            use_lm=False,
            lm_subspace_dim=0,
            lm_condition_cutoff=1e-3,
            world_energy_standard_error=0.0,
            sampler_energy_standard_error=0.0,
        )
