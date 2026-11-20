"""Contiguous device streaming SR (#885).

The streaming device SR centers every bounded sample block but the last
straight into one contiguous coordinate-by-sample matrix and keeps the last
(possibly shorter) block as a tail, so each CG operator pass is one product per
piece. These tests pin the in-place block writer, the temporary-free operator
pieces against the dense oracle, and run-to-run reproducibility.
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
from jax import numpy as jnp
from mpi4py import MPI

from jqmc._setting import (
    ATOL_BIT_EXACT,
    ATOL_C9_GAMMA_SR_UPDATE,
    ATOL_PYTHON_FP64_LOOSE,
    OPTIMIZER_DERIVATIVE_SAMPLE_BATCH,
    RTOL_C9_GAMMA_SR_UPDATE,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_PYTHON_FP64_ZERO,
    min_S_diag_abs,
)
from jqmc.jqmc_optimizer import (
    BlockDerivativeSource,
    JackknifeBinMetadata,
    JackknifeBlockMoments,
    OptimizationBatch,
    OptimizationCoordinateMap,
    compute_device_sr_real_target,
    compute_real_coordinate_optimization_target,
    compute_twist_decomposed_real_coordinate_optimization_target,
)
from jqmc.jqmc_optimizer_kernel import (
    local_device_sr_sample_dtype,
    prepare_local_device_sr_samples,
    prepare_local_device_sr_samples_into,
    solve_local_device_tall_real_cg_chunks,
    solve_local_device_tall_real_direct,
    solve_local_device_wide_real_cg_chunks,
    solve_local_device_wide_real_direct,
)

from ._optimizer_transform_cases import case_blocks, coordinate_map, jackknife_batches
from .optimizer_metadata_helpers import TEST_STATE_TOKEN, make_test_block
from .test_optimizer_lm_real import _device_batch as _device_copy

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_CASE = "j2-j3-sd-complex-source"  # J2 + J3 tie + SD S-Stiefel MO rows, complex sources: 2755 coordinates
# Wide when the 2755 active coordinates are fewer than twice the samples.
# 64: one tail block only; 1100: tall, head of one block + 76-sample tail;
# 1500: wide, one-block head + 476-sample tail; 2600: wide, two-block head
# (an in-place write at a nonzero offset) + 552-sample tail.
_SAMPLE_CASES = {64: "tall", 1100: "tall", 1500: "wide", 2600: "wide"}
# Same well-conditioned shift as the two-rank tall CG oracle: a CG stopping
# residual of ATOL_PYTHON_FP64_LOOSE bounds the CG error at the C9 tolerances.
_EPSILON = 0.2


def _device_batch(samples: int):
    blocks = case_blocks(_CASE)
    _host, device_batch = jackknife_batches(
        blocks, bins=samples, complex_source=True, complex_energy=False, seed=8850 + samples
    )
    return coordinate_map(blocks), device_batch


def _frozen_zero_shift_case():
    """Four equal-weight real samples whose third coordinate has a zero derivative (frozen).

    The active two-coordinate system is well conditioned, so with
    ``epsilon = 0`` its update is exact: ``q = (-2, -4, 0)``.
    """
    O = np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    energies = np.asarray([1.0, -1.0, 2.0, -2.0])
    weights = np.ones(4)
    blocks = tuple(
        make_test_block(
            f"frozen-case-{index}", np.asarray(0.0), field="real", derivative_policy="direct_real", sync_scope="world"
        )
        for index in range(3)
    )

    def source(block, values):
        return BlockDerivativeSource.build(block=block, direct=values)

    moments = tuple(
        JackknifeBlockMoments.build(
            block=block,
            source_sum=source(block, weights * O[:, index]),
            conjugate_source_energy_sum=source(block, weights * O[:, index] * energies),
        )
        for index, block in enumerate(blocks)
    )
    token = "frozen-zero-shift:1"
    metadata = JackknifeBinMetadata.build(
        blocks=blocks,
        batch_token=token,
        bin_sample_counts=np.ones(4, dtype=np.int64),
        bin_weight_sums=weights,
        bin_weighted_energy_sums=weights * energies,
        log_source_moments=moments,
    )
    batch = OptimizationBatch.build(
        blocks=blocks,
        batch_token=token,
        state_token=TEST_STATE_TOKEN,
        weights=weights,
        local_energies=energies,
        log_derivative_sources=tuple(source(block, O[:, index]) for index, block in enumerate(blocks)),
        local_energy_derivative_sources=tuple(source(block, np.zeros(4)) for block in blocks),
        jackknife_bin_metadata=metadata,
    )
    # Independent active-block oracle at epsilon = 0: the preconditioned
    # normal equations of the two active coordinates only.
    centered = (O - weights @ O / weights.sum()) * np.sqrt(weights / weights.sum())[:, None]
    force_samples = -2.0 * np.sqrt(weights / weights.sum()) * (energies - weights @ energies / weights.sum())
    X_active = centered[:, :2].T
    scale = 1.0 / np.sqrt(np.einsum("qi,qi->q", X_active, X_active))
    q_pc = np.linalg.solve((X_active * scale[:, None]) @ (X_active * scale[:, None]).T, scale * (X_active @ force_samples))
    expected = np.zeros(3)
    expected[:2] = q_pc * scale
    return blocks, batch, expected


def test_prepare_into_writes_the_per_block_centered_samples_in_place():
    """Blocks centered into one donated matrix equal the per-block preparation, column for column."""
    rng = np.random.default_rng(8851)
    coordinates, blocks = 7, (5, 3, 4)
    samples = sum(blocks)
    O = rng.standard_normal((samples, coordinates)) + 1.0j * rng.standard_normal((samples, coordinates))
    weights = rng.uniform(0.5, 1.5, samples)
    energies = rng.standard_normal(samples)
    moments = dict(
        total_weight=weights.sum(),
        total_weighted_O=weights @ O,
        total_weighted_energy=weights @ energies,
    )
    dtype = local_device_sr_sample_dtype(jnp.asarray(O[:5]), weights[:5], energies[:5], **moments)
    assert dtype == np.complex128
    X_matrix = jnp.zeros((coordinates, samples), dtype=dtype)
    start = 0
    for count in blocks:
        rows = slice(start, start + count)
        expected = prepare_local_device_sr_samples(jnp.asarray(O[rows]), weights[rows], energies[rows], **moments)
        X_matrix, *actual = prepare_local_device_sr_samples_into(
            X_matrix, start, jnp.asarray(O[rows]), weights[rows], energies[rows], **moments
        )
        np.testing.assert_allclose(
            np.asarray(X_matrix[:, rows]), np.asarray(expected[0]), rtol=RTOL_PYTHON_FP64_ZERO, atol=ATOL_BIT_EXACT
        )
        for value, reference in zip(actual, expected[1:], strict=True):
            np.testing.assert_allclose(
                np.asarray(value), np.asarray(reference), rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_PYTHON_FP64_LOOSE
            )
        start += count


@pytest.mark.parametrize("complex_", [True, False])
def test_temporary_free_chunk_operators_match_direct_and_one_contiguous_piece(complex_):
    """Wide and tall chunk CG on several blocks equals one contiguous piece and the direct solve."""
    rng = np.random.default_rng(8852 + int(complex_))

    def draw(shape):
        values = rng.standard_normal(shape)
        return values + 1.0j * rng.standard_normal(shape) if complex_ else values

    counts = (4, 3, 2)
    chunks = tuple(draw((9, count)) for count in counts)
    forces = tuple(rng.standard_normal(count) for count in counts)
    X = np.concatenate(chunks, axis=1)
    F = np.concatenate(forces)
    b = np.real(X @ F)

    q_direct = solve_local_device_wide_real_direct(jnp.asarray(X), b, _EPSILON)
    tol = ATOL_PYTHON_FP64_LOOSE / np.linalg.norm(b)
    for pieces in (chunks, (X,)):
        q, residual, _iterations, _breakdown = solve_local_device_wide_real_cg_chunks(
            pieces, b, _EPSILON, max_iter=500, tol=tol, x0=np.zeros_like(b)
        )
        assert residual <= ATOL_PYTHON_FP64_LOOSE
        np.testing.assert_allclose(q, q_direct, rtol=RTOL_C9_GAMMA_SR_UPDATE, atol=ATOL_C9_GAMMA_SR_UPDATE)

    q_tall_direct = solve_local_device_tall_real_direct(jnp.asarray(X), jnp.asarray(F), _EPSILON)
    tol = ATOL_PYTHON_FP64_LOOSE / np.linalg.norm(F)
    for pieces, piece_forces in ((chunks, forces), ((X,), (F,))):
        q, residual, *_ = solve_local_device_tall_real_cg_chunks(pieces, piece_forces, _EPSILON, max_iter=500, tol=tol)
        assert residual <= ATOL_PYTHON_FP64_LOOSE
        np.testing.assert_allclose(q, q_tall_direct, rtol=RTOL_C9_GAMMA_SR_UPDATE, atol=ATOL_C9_GAMMA_SR_UPDATE)


@pytest.mark.parametrize("samples", sorted(_SAMPLE_CASES))
def test_streaming_device_sr_matches_dense_oracle_and_is_reproducible(samples):
    """Contiguous streaming CG matches the dense direct oracle and repeats bit for bit."""
    assert OPTIMIZER_DERIVATIVE_SAMPLE_BATCH == 1024, "sample cases assume 1024-sample streaming blocks"
    cmap, batch = _device_batch(samples)
    common = dict(epsilon=_EPSILON, diagonal_floor=min_S_diag_abs, cg_max_iter=100000, warm_start=None)
    regime, dense = compute_device_sr_real_target(batch, cmap, cg=False, cg_tol=1.0e-5, **common)
    assert regime == _SAMPLE_CASES[samples]
    cg_tol = ATOL_PYTHON_FP64_LOOSE / dense.rhs_norm
    runs = [compute_device_sr_real_target(batch, cmap, cg=True, cg_tol=cg_tol, **common) for _ in range(2)]
    for run_regime, result in runs:
        assert run_regime == regime
        assert result.residual_norm <= ATOL_PYTHON_FP64_LOOSE
        np.testing.assert_allclose(result.q, dense.q, rtol=RTOL_C9_GAMMA_SR_UPDATE, atol=ATOL_C9_GAMMA_SR_UPDATE)
        np.testing.assert_allclose(
            result.prepared.metric_diagonal,
            dense.prepared.metric_diagonal,
            rtol=RTOL_PYTHON_FP64_LOOSE,
            atol=ATOL_PYTHON_FP64_LOOSE,
        )
    (_r0, first), (_r1, second) = runs
    assert first.num_iterations == second.num_iterations
    np.testing.assert_array_equal(first.q, second.q)
    assert first.residual_norm == second.residual_norm


def test_streaming_device_sr_frozen_coordinates_keep_contiguous_route():
    """Frozen coordinate groups (indexed active rows) still solve on the contiguous pieces."""
    cmap, batch = _device_batch(1500)
    common = dict(epsilon=_EPSILON, cg_max_iter=100000, warm_start=None)
    _regime, dense = compute_device_sr_real_target(batch, cmap, cg=False, cg_tol=1.0e-5, diagonal_floor=0.0, **common)
    diagonal = np.sort(dense.prepared.metric_diagonal)
    # Freeze the groups of the few smallest diagonals. The floor lies midway in the first
    # clear gap after the 4th smallest, so round-off in the diagonal reduction order
    # cannot move a coordinate across it.
    gap = 3 + int(np.argmax(np.diff(diagonal[3:]) > 1.0e-3 * diagonal[4:]))
    floor = float(0.5 * (diagonal[gap] + diagonal[gap + 1]))
    _regime, dense = compute_device_sr_real_target(batch, cmap, cg=False, cg_tol=1.0e-5, diagonal_floor=floor, **common)
    assert dense.prepared.frozen_coordinate_groups
    _regime, streamed = compute_device_sr_real_target(
        batch, cmap, cg=True, cg_tol=ATOL_PYTHON_FP64_LOOSE / dense.rhs_norm, diagonal_floor=floor, **common
    )
    assert streamed.prepared.frozen_coordinate_groups == dense.prepared.frozen_coordinate_groups
    assert streamed.residual_norm <= ATOL_PYTHON_FP64_LOOSE
    np.testing.assert_allclose(streamed.q, dense.q, rtol=RTOL_C9_GAMMA_SR_UPDATE, atol=ATOL_C9_GAMMA_SR_UPDATE)

    # epsilon = 0 is a valid shift: the zero frozen rows of the full-row
    # matrix must not make the direct solves singular. Device single-k direct
    # (cg=False) and streaming CG (cold and warm started), the public entry,
    # and the twist-decomposed dense device scope all reproduce the
    # independent active-block oracle, with the frozen entry exactly zero.
    blocks, host_batch, expected = _frozen_zero_shift_case()
    frozen_map = OptimizationCoordinateMap.from_blocks(blocks)
    frozen_batch = _device_copy(blocks, host_batch)
    solver = dict(epsilon=0.0, diagonal_floor=min_S_diag_abs, cg_max_iter=100, cg_tol=1.0e-14)
    _regime, direct = compute_device_sr_real_target(frozen_batch, frozen_map, cg=False, warm_start=None, **solver)
    _regime, cold = compute_device_sr_real_target(frozen_batch, frozen_map, cg=True, warm_start=None, **solver)
    _regime, warm = compute_device_sr_real_target(frozen_batch, frozen_map, cg=True, warm_start=cold.warm_start_out, **solver)
    entry = compute_real_coordinate_optimization_target(
        frozen_batch,
        frozen_map,
        comm=MPI.COMM_SELF,
        cg=False,
        warm_start=None,
        use_lm=False,
        lm_subspace_dim=0,
        lm_condition_cutoff=1.0e-3,
        energy_standard_error=0.1,
        **solver,
    )
    twist = compute_twist_decomposed_real_coordinate_optimization_target(
        frozen_batch,
        frozen_map,
        sampler_comm=MPI.COMM_SELF,
        world_comm=MPI.COMM_SELF,
        cg=False,
        use_lm=False,
        lm_subspace_dim=0,
        lm_condition_cutoff=1.0e-3,
        energy_standard_error=0.1,
        **solver,
    )
    assert direct.prepared.frozen_coordinate_groups == ((2,),)
    for label, q in (
        ("direct", direct.q),
        ("cg-cold", cold.q),
        ("cg-warm", warm.q),
        ("entry", entry.q),
        ("twist", twist.sr_q),
    ):
        assert q[2] == 0.0, label
        np.testing.assert_allclose(q, expected, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_PYTHON_FP64_LOOSE, err_msg=label)
