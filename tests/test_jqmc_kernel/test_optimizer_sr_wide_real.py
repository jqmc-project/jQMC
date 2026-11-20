"""C9 real-coordinate wide SR preparation and solver oracles."""

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

import dataclasses

import numpy as np
import pytest

from jqmc._setting import (
    ATOL_BIT_EXACT,
    ATOL_C9_GAMMA_SR_UPDATE,
    ATOL_NONVACUITY_DISTINCTION,
    ATOL_PYTHON_FP64_LOOSE,
    ATOL_SR_SOLVER_VS_DENSE_SCALE,
    RTOL_C9_GAMMA_SR_UPDATE,
    RTOL_NONVACUITY_DISTINCTION,
    RTOL_PYTHON_FP64_AGGREGATOR_INVARIANT,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_SR_SOLVER_VS_DENSE,
    min_S_diag_abs,
)
from jqmc.jqmc_optimizer import (
    JackknifeBinMetadata,
    OptimizationBatch,
    OptimizationCoordinateMap,
    SolverWarmStart,
    build_sr_coordinate_samples,
    compute_wide_sr_real_target,
    prepare_wide_sr_real_system,
    restore_wide_sr_coordinates,
)
from jqmc.jqmc_optimizer_kernel import (
    _shard_X,
    solve_device_wide_real_cg_chunks_sharded,
    solve_device_wide_real_direct_sharded,
    solve_wide_sr_real_cg,
    solve_wide_sr_real_direct,
)

from .optimizer_metadata_helpers import make_test_block
from .test_optimizer_batch import _batch, _source

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _solve_all_routes(prepared, residual, epsilon):
    direct = solve_wide_sr_real_direct(
        prepared.X_preconditioned_local,
        prepared.b_preconditioned,
        epsilon,
    )
    cg, _cpu_residual, _cpu_iterations, _breakdown = solve_wide_sr_real_cg(
        prepared.X_preconditioned_local,
        prepared.b_preconditioned,
        epsilon,
        max_iter=500,
        tol=ATOL_PYTHON_FP64_LOOSE,
        x0=np.zeros(prepared.active_indices.size),
    )
    device_direct = solve_device_wide_real_direct_sharded(
        _shard_X(prepared.X_preconditioned_local),
        prepared.b_preconditioned,
        epsilon,
    )
    device_cg, _device_residual, _device_iterations, _breakdown = solve_device_wide_real_cg_chunks_sharded(
        (prepared.X_preconditioned_local,),
        prepared.b_preconditioned,
        epsilon,
        max_iter=500,
        tol=ATOL_PYTHON_FP64_LOOSE,
        x0=np.zeros(prepared.active_indices.size),
    )
    scale = max(float(np.max(np.abs(residual))), 1.0)
    for actual in (direct, cg, device_direct, device_cg):
        np.testing.assert_allclose(
            actual,
            residual,
            rtol=RTOL_SR_SOLVER_VS_DENSE,
            atol=ATOL_SR_SOLVER_VS_DENSE_SCALE * scale,
        )
        assert not np.iscomplexobj(actual)
    return direct, cg, device_direct, device_cg


def test_exact_batch_sources_are_transformed_and_centered_once_in_coordinate_order():
    """Exact batch sources are transformed and centered once in coordinate order."""
    blocks, batch = _batch()
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    samples = build_sr_coordinate_samples(batch, coordinate_map)

    O_q = coordinate_map.transform_derivatives(batch.log_derivative_sources)
    weight_sum = np.sum(batch.weights)
    O_bar = np.einsum("n,nq->q", batch.weights, O_q) / weight_sum
    energy_bar = np.real(np.dot(batch.weights, batch.local_energies) / weight_sum)
    expected_X = (np.conj(O_q - O_bar) * np.sqrt(batch.weights)[:, None] / np.sqrt(weight_sum)).T
    expected_F = -2.0 * np.sqrt(batch.weights) * (batch.local_energies - energy_bar) / np.sqrt(weight_sum)
    np.testing.assert_allclose(samples.X_q_local, expected_X, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(samples.F_local, expected_F, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    assert not samples.X_q_local.flags.writeable
    assert not samples.F_local.flags.writeable


@pytest.mark.parametrize("statistics", ["local", "global", "remote"])
@pytest.mark.parametrize("bad_block", ["r", "z"])
def test_nonfinite_centering_statistics_name_the_offending_block(bad_block, statistics):
    """A non-finite SR centering failure names its block(s) and counts.

    ``local``: one sample of one block's source is NaN (single-rank reduction).
    ``global``: the same poisoned batch with caller-reduced global statistics.
    ``remote``: every local source is finite but the reduced weighted sum of the
    block is not (another rank's NaN). The error must name only that block, give
    the non-finite counts, and never repair or hide the value.
    """
    blocks, batch = _batch()
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    bad_index = [block.name for block in blocks].index(bad_block)
    sources = []
    for block, source in zip(blocks, batch.log_derivative_sources, strict=True):
        values = np.array(source.direct if source.direct is not None else source.d_p, copy=True)
        if block.name == bad_block and statistics != "remote":
            values[2] = np.nan
        sources.append(_source(block, values))
    poisoned = dataclasses.replace(batch, log_derivative_sources=tuple(sources))
    kwargs = {}
    if statistics != "local":
        O_q = coordinate_map.transform_derivatives(poisoned.log_derivative_sources)
        weighted_O = np.einsum("n,nq->q", poisoned.weights, O_q)
        if statistics == "remote":
            weighted_O[coordinate_map.block_coordinate_slices[bad_index]] = np.nan
        kwargs = dict(
            total_weight=np.sum(poisoned.weights),
            total_weighted_log_derivatives=weighted_O,
            total_weighted_energy=np.dot(poisoned.weights, poisoned.local_energies),
        )

    with pytest.raises(ValueError, match="SR global centering statistics must be finite") as excinfo:
        build_sr_coordinate_samples(poisoned, coordinate_map, **kwargs)
    message = str(excinfo.value)
    good_block = next(block.name for block in blocks if block.name != bad_block)
    assert f"block {bad_block!r}" in message, message
    assert f"block {good_block!r}" not in message, message
    if statistics == "remote":
        assert "non-finite raw source entries=0/" in message, message
    else:
        assert "non-finite raw source entries=1/" in message, message
        assert "first non-finite sample=2" in message, message
    bad_slice = coordinate_map.block_coordinate_slices[bad_index]
    width = bad_slice.stop - bad_slice.start
    assert f"non-finite weighted sums={width}/{width}" in message, message


@pytest.mark.parametrize("block_index", [0, 1], ids=["all-real", "all-complex"])
def test_internal_target_path_covers_uniform_real_and_complex_manifolds(block_index):
    """Internal target path covers uniform real and complex manifolds."""
    blocks, batch = _batch()
    selected_blocks = (blocks[block_index],)
    selected_metadata = JackknifeBinMetadata.build(
        blocks=selected_blocks,
        batch_token=batch.batch_token,
        bin_sample_counts=batch.jackknife_bin_metadata.bin_sample_counts,
        bin_weight_sums=batch.jackknife_bin_metadata.bin_weight_sums,
        bin_weighted_energy_sums=batch.jackknife_bin_metadata.bin_weighted_energy_sums,
        log_source_moments=(batch.jackknife_bin_metadata.log_source_moments[block_index],),
    )
    selected_batch = OptimizationBatch.build(
        blocks=selected_blocks,
        batch_token=batch.batch_token,
        state_token=batch.state_token,
        weights=batch.weights,
        local_energies=batch.local_energies,
        log_derivative_sources=(batch.log_derivative_sources[block_index],),
        local_energy_derivative_sources=(batch.local_energy_derivative_sources[block_index],),
        jackknife_bin_metadata=selected_metadata,
    )
    coordinate_map = OptimizationCoordinateMap.from_blocks(selected_blocks)
    result = compute_wide_sr_real_target(
        selected_batch,
        coordinate_map,
        epsilon=0.2,
        diagonal_floor=min_S_diag_abs,
        cg=False,
        cg_max_iter=500,
        cg_tol=ATOL_PYTHON_FP64_LOOSE,
        warm_start=None,
    )
    assert not np.iscomplexobj(result.q)
    decoded = coordinate_map.encode(result.q)
    assert decoded.shape == (1,)
    assert np.iscomplexobj(decoded) is (block_index == 1)


@pytest.mark.parametrize("cg", [False, True])
def test_internal_target_path_agrees_across_cpu_device_direct_and_cg(cg):
    """The internal wide target (host direct and CG) agrees with the dense solve of its prepared system."""
    blocks, batch = _batch()
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    result = compute_wide_sr_real_target(
        batch,
        coordinate_map,
        epsilon=0.2,
        diagonal_floor=min_S_diag_abs,
        cg=cg,
        cg_max_iter=500,
        cg_tol=ATOL_PYTHON_FP64_LOOSE,
        warm_start=None,
    )
    metric = np.real(result.prepared.X_preconditioned_local @ result.prepared.X_preconditioned_local.conj().T)
    q_pc = np.linalg.solve(
        metric + 0.2 * np.eye(metric.shape[0]),
        result.prepared.b_preconditioned,
    )
    expected_q = restore_wide_sr_coordinates(q_pc, result.prepared, coordinate_size=coordinate_map.coordinate_size)
    expected_delta = coordinate_map.encode(expected_q)
    np.testing.assert_allclose(
        result.q,
        expected_q,
        rtol=RTOL_SR_SOLVER_VS_DENSE,
        atol=ATOL_SR_SOLVER_VS_DENSE_SCALE * max(float(np.max(np.abs(expected_q))), 1.0),
    )
    np.testing.assert_allclose(
        coordinate_map.encode(result.q), expected_delta, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT
    )
    if cg:
        assert result.warm_start_out is not None
        assert not np.iscomplexobj(result.warm_start_out.values)
    else:
        assert result.warm_start_out is None


def test_matching_wide_warm_start_is_real_and_reused_at_the_same_fingerprint():
    """Matching wide warm start is real and reused at the same fingerprint."""
    blocks, batch = _batch()
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    first = compute_wide_sr_real_target(
        batch,
        coordinate_map,
        epsilon=0.2,
        diagonal_floor=min_S_diag_abs,
        cg=True,
        cg_max_iter=500,
        cg_tol=ATOL_PYTHON_FP64_LOOSE,
        warm_start=None,
    )
    assert first.warm_start_out is not None
    second = compute_wide_sr_real_target(
        batch,
        coordinate_map,
        epsilon=0.2,
        diagonal_floor=min_S_diag_abs,
        cg=True,
        cg_max_iter=500,
        cg_tol=ATOL_PYTHON_FP64_LOOSE,
        warm_start=first.warm_start_out,
    )
    np.testing.assert_allclose(second.q, first.q, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    assert second.num_iterations == 0


def test_incompatible_wide_warm_start_resets_before_real_cg():
    """Incompatible wide warm start resets before real cg."""
    blocks, batch = _batch()
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    incompatible = SolverWarmStart.build(
        values=np.ones(coordinate_map.coordinate_size),
        layout_fingerprint="incompatible-layout",
    )
    result = compute_wide_sr_real_target(
        batch,
        coordinate_map,
        epsilon=0.2,
        diagonal_floor=min_S_diag_abs,
        cg=True,
        cg_max_iter=0,
        cg_tol=ATOL_PYTHON_FP64_LOOSE,
        warm_start=incompatible,
    )
    np.testing.assert_array_equal(result.q, np.zeros_like(result.q))
    assert result.warm_start_out is not None
    assert result.warm_start_out.layout_fingerprint != incompatible.layout_fingerprint


# Single-process unit tests (``skip_mpi``, decided at collection by
# tests/conftest.py): they feed one fixed local X_q into the MPI-aware wide-SR
# solvers, whose cross-rank allreduce double-counts that input under
# ``mpirun -n 2`` -- measured 45% off the single-process oracle,
# deterministically, on both ranks. The 2-rank session exists for the
# cross-rank contract, ``test_wide_real_cpu_mpi_matches_redundant_dense_global_oracle``.


@pytest.mark.skip_mpi(reason="single-process solver unit test; local input would be double-counted across ranks")
def test_subfloor_coordinate_is_removed_before_strongly_correlated_solve():
    """Subfloor coordinate is removed before strongly correlated solve."""
    tiny = 1.0e-8
    X_q = np.asarray(
        [
            [tiny, 0.5 * tiny, -0.25 * tiny],
            [1.0, 0.5, -0.25],
        ],
        dtype=np.complex128,
    )
    F = np.asarray([0.4 + 0.2j, -0.1 + 0.3j, 0.7 - 0.5j])
    prepared = prepare_wide_sr_real_system(
        X_q,
        F,
        coordinate_groups=((0,), (1,)),
        diagonal_floor=min_S_diag_abs,
        sample_counts=(F.size,),
    )

    np.testing.assert_array_equal(prepared.active_indices, np.asarray([1]))
    assert prepared.frozen_coordinate_groups == ((0,),)
    Y = X_q[1:] / np.sqrt(np.real(np.vdot(X_q[1], X_q[1])))
    b = np.real(Y @ F)
    epsilon = 0.2
    expected = np.linalg.solve(np.real(Y @ Y.conj().T) + epsilon * np.eye(1), b)
    _solve_all_routes(prepared, expected, epsilon)

    q = restore_wide_sr_coordinates(expected, prepared, coordinate_size=2)
    assert q[0] == 0.0
    np.testing.assert_allclose(
        q[1],
        expected[0] / np.sqrt(prepared.metric_diagonal[1]),
        rtol=RTOL_PYTHON_FP64_AGGREGATOR_INVARIANT,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )


def test_atomic_complex_pair_freezes_together_when_one_diagonal_is_subfloor():
    """Atomic complex pair freezes together when one diagonal is subfloor."""
    X_q = np.asarray(
        [
            [1.0 + 0.2j, -0.4 + 0.1j, 0.3 - 0.2j],
            [1.0e-8j, -0.5e-8j, 0.25e-8j],
            [0.2 - 0.1j, 0.7 + 0.4j, -0.3 + 0.5j],
        ]
    )
    F = np.asarray([0.3 + 0.6j, -0.2 + 0.4j, 0.5 - 0.1j])
    prepared = prepare_wide_sr_real_system(
        X_q,
        F,
        coordinate_groups=((0, 1), (2,)),
        diagonal_floor=min_S_diag_abs,
        sample_counts=(F.size,),
    )
    assert prepared.frozen_coordinate_groups == ((0, 1),)
    np.testing.assert_array_equal(prepared.active_indices, np.asarray([2]))


def test_mixed_coordinate_decode_matches_real_and_complex_storage_blocks():
    """Mixed coordinate decode matches real and complex storage blocks."""
    blocks = (
        make_test_block(
            "real",
            np.asarray(0.0),
            field="real",
            derivative_policy="direct_real",
            sync_scope="sampler",
        ),
        make_test_block(
            "complex",
            np.asarray(0.0 + 0.0j),
            field="complex",
            derivative_policy="holomorphic",
            sync_scope="sampler",
        ),
    )
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    X_q = np.asarray(
        [
            [0.6 + 0.2j, -0.1 + 0.3j, 0.4 - 0.2j, 0.2 + 0.1j],
            [0.1 - 0.4j, 0.7 + 0.2j, -0.3 + 0.5j, 0.2 - 0.6j],
            [0.5 + 0.3j, -0.2 + 0.4j, 0.6 + 0.1j, -0.4 - 0.2j],
        ]
    )
    F = np.asarray([0.2 + 0.7j, -0.5 + 0.1j, 0.3 - 0.4j, 0.6 + 0.2j])
    prepared = prepare_wide_sr_real_system(
        X_q,
        F,
        coordinate_groups=coordinate_map.coordinate_groups,
        diagonal_floor=min_S_diag_abs,
        sample_counts=(F.size,),
    )
    metric = np.real(prepared.X_preconditioned_local @ prepared.X_preconditioned_local.conj().T)
    q_pc = np.linalg.solve(metric + 0.15 * np.eye(3), prepared.b_preconditioned)
    q = restore_wide_sr_coordinates(q_pc, prepared, coordinate_size=coordinate_map.coordinate_size)
    delta = coordinate_map.encode(q)
    np.testing.assert_allclose(delta, np.asarray([q[0], q[1] + 1.0j * q[2]]), rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)


@pytest.mark.skip_mpi(reason="single-process solver unit test; local input would be double-counted across ranks")
def test_gamma_all_real_matches_frozen_pre_c9_normalized_reference():
    """Gamma all real matches frozen pre c9 normalized reference."""
    X_q = np.asarray(
        [
            [0.4, -0.2, 0.7, 0.1, -0.3],
            [0.1, 0.8, -0.4, 0.6, 0.2],
        ],
        dtype=np.float64,
    )
    F = np.asarray([0.3, -0.5, 0.2, 0.7, -0.1])
    diagonal = np.einsum("qi,qi->q", X_q, X_q)
    X_pc = X_q / np.sqrt(diagonal)[:, None]
    b_pc = X_pc @ F
    epsilon = 0.1
    reference = np.linalg.solve(X_pc @ X_pc.T + epsilon * np.eye(2), b_pc)
    prepared = prepare_wide_sr_real_system(
        X_q,
        F,
        coordinate_groups=((0,), (1,)),
        diagonal_floor=min_S_diag_abs,
        sample_counts=(F.size,),
    )
    actuals = _solve_all_routes(prepared, reference, epsilon)
    for comparison_index, (actual) in enumerate(actuals):
        comparison_iteration = str(comparison_index)  # noqa: F841 - captured by the GDD iteration receipt
        np.testing.assert_allclose(
            actual,
            reference,
            rtol=RTOL_C9_GAMMA_SR_UPDATE,
            atol=ATOL_C9_GAMMA_SR_UPDATE,
        )


def test_complex_samples_are_contracted_before_real_projection_and_cg_rejects_complex_carry():
    """Complex samples are contracted before real projection and cg rejects complex carry."""
    X_q = np.asarray([[0.2 + 0.8j, -0.4 + 0.3j, 0.5 - 0.7j]])
    F = np.asarray([0.6 - 0.2j, -0.1 + 0.9j, 0.3 + 0.5j])
    prepared = prepare_wide_sr_real_system(
        X_q,
        F,
        coordinate_groups=((0,),),
        diagonal_floor=min_S_diag_abs,
        sample_counts=(F.size,),
    )
    expected = np.real(prepared.X_preconditioned_local @ F)
    discarded = np.real(prepared.X_preconditioned_local) @ np.real(F)
    np.testing.assert_allclose(prepared.b_preconditioned, expected, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    assert not np.allclose(expected, discarded, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION)

    complex_x0 = np.zeros(1, dtype=np.complex128)
    with pytest.raises(ValueError, match="real"):
        solve_wide_sr_real_cg(
            prepared.X_preconditioned_local,
            prepared.b_preconditioned,
            0.1,
            max_iter=10,
            tol=ATOL_PYTHON_FP64_LOOSE,
            x0=complex_x0,
        )


@pytest.mark.skip_serial(reason="cross-rank contract; launch via mpirun -np 2")
def test_wide_real_cpu_mpi_matches_redundant_dense_global_oracle():
    """Wide real cpu mpi matches redundant dense global oracle."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2, "the redundant dense oracle is written for exactly two ranks: launch via mpirun -np 2"
    rank = comm.Get_rank()
    X_local = np.asarray(
        [
            [0.4 + 0.1j * (rank + 1), -0.2 + 0.3j],
            [0.1 - 0.2j, 0.7 + 0.1j * rank],
        ]
    )
    F_local = np.asarray([0.3 + 0.2j * rank, -0.4 + 0.1j * (rank + 1)])
    diagonal = comm.allreduce(
        np.real(np.einsum("qi,qi->q", X_local, np.conj(X_local))),
        op=MPI.SUM,
    )
    force = comm.allreduce(np.real(X_local @ F_local), op=MPI.SUM)
    prepared = prepare_wide_sr_real_system(
        X_local,
        F_local,
        coordinate_groups=((0,), (1,)),
        diagonal_floor=min_S_diag_abs,
        sample_counts=tuple(comm.allgather(F_local.size)),
        metric_diagonal=diagonal,
        coordinate_force=force,
    )

    def reduce_sum(values):
        return comm.allreduce(values, op=MPI.SUM)

    epsilon = 0.2
    direct = solve_wide_sr_real_direct(
        prepared.X_preconditioned_local,
        prepared.b_preconditioned,
        epsilon,
        reduce_sum=reduce_sum,
    )
    cg, _residual, _iterations, _breakdown = solve_wide_sr_real_cg(
        prepared.X_preconditioned_local,
        prepared.b_preconditioned,
        epsilon,
        max_iter=500,
        tol=ATOL_PYTHON_FP64_LOOSE,
        x0=np.zeros(2),
        reduce_sum=reduce_sum,
    )
    device_direct = solve_device_wide_real_direct_sharded(
        _shard_X(prepared.X_preconditioned_local),
        prepared.b_preconditioned,
        epsilon,
    )
    device_cg, _device_residual, _device_iterations, _breakdown = solve_device_wide_real_cg_chunks_sharded(
        (prepared.X_preconditioned_local,),
        prepared.b_preconditioned,
        epsilon,
        max_iter=500,
        tol=ATOL_PYTHON_FP64_LOOSE,
        x0=np.zeros(2),
    )
    X_global = np.concatenate(comm.allgather(prepared.X_preconditioned_local), axis=1)
    expected = np.linalg.solve(
        np.real(X_global @ X_global.conj().T) + epsilon * np.eye(2),
        prepared.b_preconditioned,
    )
    for comparison_index, (actual) in enumerate((direct, cg, device_direct, device_cg)):
        comparison_iteration = str(comparison_index)  # noqa: F841 - captured by the GDD iteration receipt
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=RTOL_C9_GAMMA_SR_UPDATE,
            atol=ATOL_C9_GAMMA_SR_UPDATE,
        )


def _wide_cg_samples(rng, field_mode):
    """Random wide SR samples whose dtype is complex in both modes."""
    X = rng.normal(size=(5, 23)) + 0.0j
    if field_mode == "complex":
        X = X + 1.0j * rng.normal(size=X.shape)
    b = rng.normal(size=X.shape[0])
    return X, b


@pytest.mark.parametrize("field_mode", ["complex", "real-valued-complex"])
def test_wide_real_cg_matches_direct_on_complex_and_real_valued_complex_samples(field_mode):
    """Wide real CG matches the direct solve for complex and real-valued complex samples.

    The CG realifies the complex sample matrix once (``[Re X | Im X]``, or
    only ``Re X`` when ``Im X`` is identically zero), so both branches must
    reproduce ``(Re(X X^dagger) + epsilon I)^{-1} b`` of the direct solver.
    """
    rng = np.random.default_rng(20261007)
    X, b = _wide_cg_samples(rng, field_mode)
    assert np.iscomplexobj(X)
    assert bool(np.any(X.imag)) is (field_mode == "complex")
    epsilon = 0.05
    direct = solve_wide_sr_real_direct(X, b, epsilon)
    cg, _residual, iterations, _breakdown = solve_wide_sr_real_cg(
        X,
        b,
        epsilon,
        max_iter=500,
        tol=ATOL_PYTHON_FP64_LOOSE,
        x0=np.zeros(b.size),
    )
    assert iterations > 0
    assert not np.iscomplexobj(cg)
    np.testing.assert_allclose(
        cg,
        direct,
        rtol=RTOL_SR_SOLVER_VS_DENSE,
        atol=ATOL_SR_SOLVER_VS_DENSE_SCALE * max(float(np.max(np.abs(direct))), 1.0),
    )
    if field_mode == "complex":
        # Non-vacuity: dropping Im X would solve a different system.
        real_only = solve_wide_sr_real_direct(np.real(X), b, epsilon)
        assert not np.allclose(real_only, direct, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION)


def test_wide_real_cg_does_not_conjugate_the_samples_per_iteration(monkeypatch):
    """Wide real CG never copies conj(X) inside the iteration loop.

    The old operator ``np.real(X @ (np.conj(X).T @ v))`` materialized a full
    conjugate copy of the local sample matrix on every CG iteration, which
    dominated the per-iteration cost on 128-rank nodes.
    """
    rng = np.random.default_rng(1007)
    X, b = _wide_cg_samples(rng, "complex")
    calls = {"count": 0}
    original_conj = np.conj
    original_conjugate = np.conjugate

    def counting_conj(*args, **kwargs):
        calls["count"] += 1
        return original_conj(*args, **kwargs)

    def counting_conjugate(*args, **kwargs):
        calls["count"] += 1
        return original_conjugate(*args, **kwargs)

    monkeypatch.setattr(np, "conj", counting_conj)
    monkeypatch.setattr(np, "conjugate", counting_conjugate)
    _x, _residual, iterations, _breakdown = solve_wide_sr_real_cg(
        X,
        b,
        0.05,
        max_iter=500,
        tol=ATOL_PYTHON_FP64_LOOSE,
        x0=np.zeros(b.size),
    )
    assert iterations > 2
    assert calls["count"] <= 1


class _RecordingSingleRankComm:
    """Identity-reduction communicator that records which mpi4py API is used."""

    def __init__(self, size=1):
        self._size = size
        self.solver_active = False
        self.object_reductions_in_solver = 0
        self.buffer_reductions_in_solver = 0

    def Get_rank(self):
        return 0

    def Get_size(self):
        return self._size

    def Barrier(self):
        return None

    def allreduce(self, value, op=None):
        if self.solver_active:
            self.object_reductions_in_solver += 1
        return np.array(value, copy=True) if isinstance(value, np.ndarray) else value

    def Allreduce(self, sendbuf, recvbuf, op=None):
        if self.solver_active:
            self.buffer_reductions_in_solver += 1
        send = sendbuf[0] if isinstance(sendbuf, (list, tuple)) else sendbuf
        recv = recvbuf[0] if isinstance(recvbuf, (list, tuple)) else recvbuf
        np.copyto(recv, send)

    def allgather(self, value):
        return [value] * self._size

    def bcast(self, value, root=0):
        return value


@pytest.mark.parametrize("call_site", ["wide-target", "planb-host", "planb-device-staged"])
def test_wide_real_cg_call_sites_reduce_with_buffer_allreduce(monkeypatch, call_site):
    """Every MPI wide-real CG call site sums the operator image with a buffer Allreduce.

    The per-iteration coordinate-vector reduction must use ``comm.Allreduce``
    on NumPy buffers, never mpi4py's pickle-based ``comm.allreduce``.
    """
    import jqmc.jqmc_optimizer as optimizer_module
    import jqmc.jqmc_optimizer_kernel as kernel_module

    from .test_optimizer_lm_real import _device_batch, _lm_case

    comm = _RecordingSingleRankComm(size=2 if call_site == "planb-device-staged" else 1)
    original_cg = kernel_module.solve_wide_sr_real_cg
    solver_calls = {"count": 0, "iterations": 0}

    def tracking_cg(*args, **kwargs):
        solver_calls["count"] += 1
        comm.solver_active = True
        try:
            result = original_cg(*args, **kwargs)
        finally:
            comm.solver_active = False
        solver_calls["iterations"] += result[2]
        return result

    monkeypatch.setattr(kernel_module, "solve_wide_sr_real_cg", tracking_cg)
    blocks, batch, _O_q, _D_q = _lm_case("mixed")
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    solver = dict(epsilon=0.2, diagonal_floor=0.0, cg=True, cg_max_iter=500, cg_tol=ATOL_PYTHON_FP64_LOOSE)
    if call_site == "wide-target":
        compute_wide_sr_real_target(
            batch,
            coordinate_map,
            comm=comm,
            warm_start=None,
            **solver,
        )
    elif call_site == "planb-host":
        optimizer_module._solve_planb_sr_scope(
            batch,
            coordinate_map,
            np.arange(coordinate_map.coordinate_size),
            comm=comm,
            **solver,
        )
    else:
        optimizer_module._solve_planb_sr_scope_device_dense(
            _device_batch(blocks, batch),
            coordinate_map,
            optimizer_module._coordinate_scope_indices(coordinate_map, "sampler"),
            sync_scope="sampler",
            comm=comm,
            all_real=False,
            **solver,
        )
    assert solver_calls["count"] == 1
    assert solver_calls["iterations"] > 0
    assert comm.object_reductions_in_solver == 0
    # One operator application before the loop plus one per iteration.
    assert comm.buffer_reductions_in_solver == solver_calls["iterations"] + 1
