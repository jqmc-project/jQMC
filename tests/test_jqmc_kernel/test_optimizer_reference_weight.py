"""LM acceptance-control propagation, using fixed candidate and sample payloads."""

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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from mpi4py import MPI

import jqmc.jqmc_optimizer as optimizer_module
from jqmc._setting import (
    ATOL_BIT_EXACT,
    OPTIMIZER_DEFAULT_CG_TOL,
    OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF,
    OPTIMIZER_DEFAULT_LM_REFERENCE_WEIGHT_MIN,
    RTOL_PYTHON_FP64_LOOSE,
    atol_consistency,
    rtol_consistency,
)
from jqmc.jqmc_optimizer import OptimizationConfig, OptimizationCoordinateMap, ParameterSelection, run_optimize

from .complex_opt_ground_truth_harness import _build_ham
from .test_optimizer_full_loop_real import _Sampler, _batch, _disable_side_effects
from .test_optimizer_lm_real import _lm_case
from .test_twist_optimize_update_value import _ThreadWorld, _ThreadWorldComm

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize("route", ["ordinary", "planb"])
@pytest.mark.parametrize("lm_subspace_dim", [-1, 1])
@pytest.mark.parametrize("threshold_case", ["omitted", "equality", "reject"])
@pytest.mark.gdd_electron_configuration(mode="not-required", reason="Synthetic optimizer moments and fixed LM candidates only.")
def test_lm_reference_weight_controls_candidate_acceptance(monkeypatch, route, lm_subspace_dim, threshold_case):
    """Both LM targets use the supplied reference-weight threshold, accept equality, and preserve aSR rejection.

    Fixed mixed-real/complex derivative payloads contain world and sampler
    coordinate groups. Full and reduced pencils use a candidate of weight 0.5
    and energy below the unchanged energy limit. Omitting the threshold or
    explicitly selecting 0.75 rejects it; selecting 0.5 accepts it. The solve
    and aSR selection are controlled here to isolate this acceptance rule,
    while production pencil construction, scope handling, and decoding run.
    """
    assert OPTIMIZER_DEFAULT_LM_REFERENCE_WEIGHT_MIN == 0.9
    blocks, batch, _O_q, _D_q = _lm_case("mixed", scopes=("world", "sampler", "world"))
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    fallback_calls = []
    solver_calls = []

    def candidate_lm(pencil, *, condition_cutoff):
        solver_calls.append(pencil.H.shape[0] - 1)
        size = pencil.H.shape[0] - 1
        return optimizer_module.LMRealResult(
            q=np.zeros(size),
            predicted_energy=pencil.H_0 - 1.0,
            reference_weight=0.5,
            active_indices=np.arange(size),
        )

    def selected_asr(*coefficients, gamma_max, fallback_gamma):
        fallback_calls.append(coefficients)
        return optimizer_module.ASRBoundedResult(gamma=0.37, predicted_energy=-123.0, used_fallback=False)

    monkeypatch.setattr(optimizer_module, "solve_lm_real_pencil", candidate_lm)
    monkeypatch.setattr(optimizer_module, "select_asr_gamma_bounded", selected_asr)
    kwargs = dict(
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=False,
        cg_max_iter=500,
        cg_tol=OPTIMIZER_DEFAULT_CG_TOL,
        use_lm=True,
        lm_subspace_dim=lm_subspace_dim,
        lm_condition_cutoff=OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF,
        energy_standard_error=0.1,
    )
    if threshold_case != "omitted":
        kwargs["lm_reference_weight_min"] = 0.5 if threshold_case == "equality" else 0.75
    if route == "ordinary":
        result = optimizer_module.compute_real_coordinate_optimization_target(batch, coordinate_map, warm_start=None, **kwargs)
    else:
        result = optimizer_module.compute_twist_decomposed_real_coordinate_optimization_target(
            batch, coordinate_map, sampler_comm=MPI.COMM_SELF, world_comm=MPI.COMM_SELF, **kwargs
        )

    scope_count = 1 if route == "ordinary" else 2
    assert len(solver_calls) == scope_count
    assert np.all(np.isfinite(result.q))
    assert np.all(np.isfinite(result.sr_q))
    assert np.any(result.sr_q != 0.0)
    assert result.lm_reference_weight == 0.5
    if threshold_case == "equality":
        assert not fallback_calls
        assert result.rejection_reason is None
        expected_scale = 1.0
        np.testing.assert_array_equal(result.q, np.zeros(coordinate_map.coordinate_size))
    else:
        assert len(fallback_calls) == scope_count
        assert result.rejection_reason.count("reference weight") == scope_count
        assert "linear-method energy gate failed" not in result.rejection_reason
        threshold = OPTIMIZER_DEFAULT_LM_REFERENCE_WEIGHT_MIN if threshold_case == "omitted" else 0.75
        assert f"is below {threshold:.8f}" in result.rejection_reason
        expected_scale = 0.37
        np.testing.assert_allclose(result.q, 0.37 * result.sr_q, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(result.accepted_scale, expected_scale, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)


class _ReferenceWeightSampler(_Sampler):
    """Production parameter publisher/update with fixed, non-sampled LM inputs."""

    def configure_optimization(self, *, method, hyperparameters):
        self.recorded_hyperparameters = dict(hyperparameters)

    def optimization_batch(self, *, num_warmup_steps, num_bin_blocks, blocks, need_local_energy_derivatives):
        assert need_local_energy_derivatives
        batch = _batch(tuple(blocks), self.wavefunction_state_token)
        return replace(batch, local_energy_derivative_sources=batch.log_derivative_sources)


@pytest.mark.parametrize("route", ["ordinary", "planb"])
@pytest.mark.parametrize("threshold_case", ["omitted", "equality"])
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="The driver consumes a fixed synthetic batch; its sampler does not generate electron positions."
)
def test_optimizer_driver_forwards_reference_weight(monkeypatch, route, threshold_case):
    """The real driver logs and forwards the LM threshold and applies the resulting J2 update on both dispatch routes.

    The Hamiltonian is the committed non-Gamma H2 AE Cartesian fixture with
    its exponential J2 parameter and independently perturbed spin orbitals.
    Only J2 is optimized. Eight fixed synthetic samples, four bins, and a
    controlled LM candidate replace Monte Carlo sampling and eigensolution.
    The twist-decomposed route runs two sampler groups concurrently through an exchanging world
    communicator, not a patched dispatch predicate. Default 0.9 rejects the
    weight-0.5 candidate; an explicit 0.5 accepts it and changes the actual J2
    parameter. No physical energy or statistical-accuracy claim is made.
    """
    _disable_side_effects(monkeypatch)
    original_ordinary = optimizer_module.compute_real_coordinate_optimization_target
    original_planb = optimizer_module.compute_twist_decomposed_real_coordinate_optimization_target
    captures = []

    def candidate_lm(pencil, *, condition_cutoff):
        size = pencil.H.shape[0] - 1
        return optimizer_module.LMRealResult(
            q=np.full(size, 0.125),
            predicted_energy=pencil.H_0 - 1.0,
            reference_weight=0.5,
            active_indices=np.arange(size),
        )

    def selected_asr(*coefficients, gamma_max, fallback_gamma):
        return optimizer_module.ASRBoundedResult(gamma=0.37, predicted_energy=-123.0, used_fallback=False)

    def recording_ordinary(batch, coordinate_map, **kwargs):
        result = original_ordinary(batch, coordinate_map, **kwargs)
        captures.append(("ordinary", dict(kwargs), result))
        return result

    def recording_planb(batch, coordinate_map, **kwargs):
        result = original_planb(batch, coordinate_map, **kwargs)
        captures.append(("planb", dict(kwargs), result))
        return result

    monkeypatch.setattr(optimizer_module, "solve_lm_real_pencil", candidate_lm)
    monkeypatch.setattr(optimizer_module, "select_asr_gamma_bounded", selected_asr)
    monkeypatch.setattr(optimizer_module, "compute_real_coordinate_optimization_target", recording_ordinary)
    monkeypatch.setattr(optimizer_module, "compute_twist_decomposed_real_coordinate_optimization_target", recording_planb)
    kwargs = dict(method="sr", delta=0.05, epsilon=0.2, cg_flag=False, use_lm=True, lm_subspace_dim=-1)
    if threshold_case == "equality":
        kwargs["lm_reference_weight_min"] = 0.5
    config = OptimizationConfig(
        num_sampling_steps=8,
        num_opt_steps=1,
        num_bin_blocks=4,
        dump_wavefunction=False,
        use_device_optimizer=False,
        parameters=ParameterSelection(
            opt_J1_param=False,
            opt_J2_param=True,
            opt_J3_param=False,
            opt_geminal=False,
            opt_JDTN_u_param=False,
            opt_JDTN_chi_param=False,
            opt_JDTN_f_param=False,
        ),
        optimizer_kwargs=kwargs,
    )
    if route == "ordinary":
        samplers = [_ReferenceWeightSampler(_build_ham("twist"))]
    else:
        world = _ThreadWorld(2)
        samplers = [
            _ReferenceWeightSampler(_build_ham("twist"), comm=MPI.COMM_SELF, world_comm=_ThreadWorldComm(world, rank))
            for rank in range(2)
        ]
    original_parameters = [
        np.asarray(s.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param).copy()
        for s in samplers
    ]
    if route == "ordinary":
        run_optimize(samplers[0], config)
    else:

        def run_sampler(sampler):
            try:
                run_optimize(sampler, replace(config, twist_index=sampler.world_comm.Get_rank()))
            except BaseException:
                world.barrier.abort()
                raise

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_sampler, sampler) for sampler in samplers]
            for future in futures:
                future.result()

    expected_threshold = 0.5 if threshold_case == "equality" else OPTIMIZER_DEFAULT_LM_REFERENCE_WEIGHT_MIN
    assert len(captures) == len(samplers)
    for observed_route, target_kwargs, target in captures:
        assert observed_route == route
        assert target_kwargs["lm_reference_weight_min"] == expected_threshold
        assert np.all(np.isfinite(target.q))
        if threshold_case == "equality":
            assert target.rejection_reason is None
            np.testing.assert_array_equal(target.q, np.asarray([0.125]))
        else:
            assert "reference weight" in target.rejection_reason
            np.testing.assert_allclose(target.q, 0.37 * target.sr_q, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    # Both twist groups consume the same fixed payload, so their shared update
    # must coincide; association never depends on the threads' completion order.
    if route == "planb":
        np.testing.assert_allclose(captures[1][2].q, captures[0][2].q, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    for sampler, before in zip(samplers, original_parameters, strict=True):
        assert sampler.recorded_hyperparameters["lm_reference_weight_min"] == expected_threshold
        assert sampler.optimization_step_index == 1
        assert sampler.wavefunction_state_token == ("c12-full-loop", 1)
        after = np.asarray(sampler.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param)
        assert np.all(np.isfinite(after))
        assert np.any(after != before)
        np.testing.assert_allclose(after, before + 0.05 * captures[0][2].q[0], rtol=rtol_consistency, atol=atol_consistency)
