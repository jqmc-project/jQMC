"""C10 real-coordinate tall SR layout and solver oracles."""

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

from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import (
    ATOL_BIT_EXACT,
    ATOL_C9_GAMMA_SR_UPDATE,
    ATOL_NONVACUITY_DISTINCTION,
    ATOL_PYTHON_FP64_LOOSE,
    ATOL_SR_SOLVER_VS_DENSE_SCALE,
    RTOL_C9_GAMMA_SR_UPDATE,
    RTOL_NONVACUITY_DISTINCTION,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_SR_SOLVER_VS_DENSE,
)
from jqmc.jqmc_optimizer import (
    BlockDerivativeSource,
    JackknifeBinMetadata,
    JackknifeBlockMoments,
    OptimizationBatch,
    OptimizationCoordinateMap,
    build_tall_sr_real_layout,
    compute_sr_real_target,
    compute_tall_sr_real_target,
    select_sr_real_regime,
)
from jqmc.jqmc_optimizer_kernel import (
    _shard_X_F,
    solve_device_tall_real_cg_chunks_sharded,
    solve_device_tall_real_direct_sharded,
    solve_tall_sr_real_cg,
    solve_tall_sr_real_direct,
)
from jqmc.wavefunction import ParameterTangentSpec, VariationalParameterBlock, _symmetric_real_linear_tie

from .optimizer_metadata_helpers import TEST_STATE_TOKEN, make_test_block

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _tied_host_batch(n_samples, n_orbitals):
    shape = (n_orbitals, n_orbitals + 1)
    tie = _symmetric_real_linear_tie(shape)
    block = VariationalParameterBlock.build(
        name="j3_matrix",
        values=np.zeros(shape),
        field="real",
        derivative_policy="direct_real",
        sync_scope="world",
        tangent_spec=ParameterTangentSpec.with_linear_tie(tie, state_token=TEST_STATE_TOKEN),
        state_token=TEST_STATE_TOKEN,
    )
    rng = np.random.default_rng(478)
    dtype = np.complex64 if is_mixed_precision_enabled() else np.complex128
    raw = (rng.normal(size=(n_samples, *shape)) + 1j * rng.normal(size=(n_samples, *shape))).astype(dtype)
    weights = 1.0 + rng.random(n_samples)
    energies = rng.normal(size=n_samples) + 1j * rng.normal(size=n_samples)
    token = "sr-host-tie:1"

    def source(values):
        return BlockDerivativeSource.build(block=block, direct=values)

    weighted_energies = (weights * energies)[:, None, None]
    moments = JackknifeBlockMoments.build(
        block=block,
        source_sum=source(np.sum(weights[:, None, None] * raw, axis=0)[None]),
        conjugate_source_energy_sum=source(np.sum(weighted_energies * raw.conj(), axis=0)[None]),
    )
    metadata = JackknifeBinMetadata.build(
        blocks=(block,),
        batch_token=token,
        bin_sample_counts=[n_samples],
        bin_weight_sums=[weights.sum()],
        bin_weighted_energy_sums=[weights @ energies],
        log_source_moments=(moments,),
    )
    batch = OptimizationBatch.build(
        blocks=(block,),
        batch_token=token,
        state_token=TEST_STATE_TOKEN,
        weights=weights,
        local_energies=energies,
        log_derivative_sources=(source(raw),),
        local_energy_derivative_sources=None,
        jackknife_bin_metadata=metadata,
    )
    return batch, OptimizationCoordinateMap.from_blocks((block,)), tie, raw.reshape(n_samples, -1)


@pytest.mark.parametrize("expected_regime", ["wide", "tall"])
def test_host_sr_prepares_once_and_bounds_tied_workspace(expected_regime, monkeypatch):
    """Both host solvers reuse preparation and process tie adjoints in bounded row blocks."""
    from jqmc import jqmc_optimizer as optimizer

    n_samples = 33
    batch, coordinate_map, tie, raw = _tied_host_batch(n_samples, 7 if expected_regime == "wide" else 12)
    original_transform = OptimizationCoordinateMap.transform_derivatives
    original_adjoint = OptimizationCoordinateMap._real_adjoint_block
    original_prepare = optimizer.prepare_wide_sr_real_system
    transforms = []
    row_counts = []
    preparations = []

    def transform(self, sources):
        result = original_transform(self, sources)
        transforms.append(result.shape)
        return result

    def adjoint(self, index, covectors):
        row_counts.append(covectors.shape[0])
        return original_adjoint(self, index, covectors)

    def prepare(samples, force, **kwargs):
        preparations.append(samples.shape)
        return original_prepare(samples, force, **kwargs)

    monkeypatch.setattr(OptimizationCoordinateMap, "transform_derivatives", transform)
    monkeypatch.setattr(OptimizationCoordinateMap, "_real_adjoint_block", adjoint)
    monkeypatch.setattr(optimizer, "prepare_wide_sr_real_system", prepare)
    epsilon = 0.2
    regime, result = optimizer.compute_sr_real_target(
        batch,
        coordinate_map,
        comm=MPI.COMM_SELF,
        epsilon=epsilon,
        diagonal_floor=0.0,
        cg=False,
        cg_max_iter=500,
        cg_tol=ATOL_PYTHON_FP64_LOOSE,
        warm_start=None,
    )
    assert regime == expected_regime
    assert preparations == [(tie.coordinate_size, n_samples)]
    # One full-size transform (the samples); the global mean transforms only
    # the one summed row of the binned moments.
    assert transforms == [(1, tie.coordinate_size), (n_samples, tie.coordinate_size)]
    assert sum(row_counts) == 2 * (1 + n_samples)
    assert max(row_counts) <= 8

    def tie_columns(rows):
        columns = np.zeros((rows.shape[0], tie.coordinate_size), dtype=np.result_type(rows.dtype, tie.coefficients.dtype))
        for source_index, target_index, coefficient in zip(
            tie.output_indices, tie.coordinate_indices, tie.coefficients, strict=True
        ):
            values = rows[:, source_index]
            columns.real[:, target_index] += coefficient * values.real
            columns.imag[:, target_index] += coefficient * values.imag
        return columns

    O = tie_columns(raw)
    weights, energies = batch.weights, batch.local_energies
    weight = weights.sum()
    # The published moment row sum_n w_n raw_n, transformed like any sample row.
    weighted_O = tie_columns(np.sum(weights[:, None] * raw, axis=0)[None])[0]
    centered = (np.conj(O - weighted_O / weight) * np.sqrt(weights)[:, None] / np.sqrt(weight)).T
    force = -2.0 * np.sqrt(weights) * (energies - (weights @ energies).real / weight) / np.sqrt(weight)
    inverse_sqrt = 1.0 / np.sqrt(np.real(np.einsum("qi,qi->q", centered, centered.conj())))
    X = centered * inverse_sqrt[:, None]
    np.testing.assert_array_equal(result.prepared.X_preconditioned_local, X)
    assert result.prepared.X_preconditioned_local.dtype == X.dtype
    expected_q, Z, h = _dense_oracle(X, force, epsilon)
    np.testing.assert_allclose(
        result.q,
        expected_q * inverse_sqrt,
        rtol=RTOL_SR_SOLVER_VS_DENSE,
        atol=ATOL_SR_SOLVER_VS_DENSE_SCALE * max(float(np.max(np.abs(expected_q * inverse_sqrt))), 1.0),
    )
    if regime == "tall":
        layout = result.dual_layout
        np.testing.assert_array_equal(layout.Z_valid_local, Z)
        np.testing.assert_array_equal(layout.h_valid_local, h)
        assert not layout.Z_valid_local.flags.writeable


def _dense_oracle(X, F, epsilon):
    """Independent real embedding; do not call the production layout helper."""
    Z = np.concatenate((np.real(X), np.imag(X)), axis=1)
    h = np.concatenate((np.real(F), -np.imag(F)))
    dual = np.linalg.solve(Z.T @ Z + epsilon * np.eye(Z.shape[1]), h)
    return Z @ dual, Z, h


def _tall_batch(field_mode):
    rng = np.random.default_rng(921)
    fields = {
        "all-real": ("real",) * 7,
        "all-complex": ("complex",) * 4,
        "mixed": ("real",) * 3 + ("complex",) * 2,
    }[field_mode]
    blocks = tuple(
        make_test_block(
            f"p{index}",
            np.asarray(0.0 if field == "real" else 0.0 + 0.0j),
            field=field,
            derivative_policy="direct_real" if field == "real" else "holomorphic",
            sync_scope="sampler",
        )
        for index, field in enumerate(fields)
    )
    raw_values = tuple(rng.normal(size=3) + 1.0j * rng.normal(size=3) for _ in blocks)

    def source(block, values):
        if block.derivative_policy == "direct_real":
            return BlockDerivativeSource.build(block=block, direct=values)
        return BlockDerivativeSource.build(block=block, d_p=values)

    sources = tuple(source(block, values) for block, values in zip(blocks, raw_values))
    energies = np.asarray([-0.7 + 0.2j, 0.4 - 0.3j, 1.1 + 0.5j])
    moments = tuple(
        JackknifeBlockMoments.build(
            block=block,
            source_sum=source(block, values),
            conjugate_source_energy_sum=source(block, np.conj(values) * energies),
        )
        for block, values in zip(blocks, raw_values)
    )
    token = "c10-test:1"
    metadata = JackknifeBinMetadata.build(
        blocks=blocks,
        batch_token=token,
        bin_sample_counts=np.ones(3, dtype=np.int64),
        bin_weight_sums=np.ones(3),
        bin_weighted_energy_sums=energies,
        log_source_moments=moments,
    )
    batch = OptimizationBatch.build(
        blocks=blocks,
        batch_token=token,
        state_token=TEST_STATE_TOKEN,
        weights=np.ones(3),
        local_energies=energies,
        log_derivative_sources=sources,
        local_energy_derivative_sources=sources,
        jackknife_bin_metadata=metadata,
    )
    return blocks, batch


def test_regime_uses_realified_sample_dimension_and_equality_is_tall(monkeypatch):
    """The regime compares the surviving coordinates with the GLOBAL real dual length; equality is tall.

    The real dual has one entry per sample and the complex dual two. The host
    wide preparation stores its matrix for that same global decision: with
    the host Rust route a globally wide system is written sample-major even
    when this rank alone holds fewer samples than coordinates, and a tall
    system stays coordinate-major.
    """
    assert select_sr_real_regime(5, (1, 2), complex_samples=True) == "wide"
    assert select_sr_real_regime(6, (1, 2), complex_samples=True) == "tall"
    assert select_sr_real_regime(7, (1, 2), complex_samples=True) == "tall"
    assert select_sr_real_regime(2, (1, 2), complex_samples=False) == "wide"
    assert select_sr_real_regime(3, (1, 2), complex_samples=False) == "tall"
    assert select_sr_real_regime(5, (1, 2), complex_samples=False) == "tall"

    from jqmc import jqmc_optimizer as optimizer

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    rng = np.random.default_rng(897)
    # Six coordinates and two local samples: this rank alone is tall for both
    # dtypes; two ranks give N = 4 (real tall: 6 >= 4, complex wide: 6 < 8);
    # four ranks give N = 8, wide for both.
    expected = {
        ((2,), False): "tall",
        ((2,), True): "tall",
        ((2, 2), False): "tall",
        ((2, 2), True): "wide",
        ((2, 2, 2, 2), False): "wide",
        ((2, 2, 2, 2), True): "wide",
    }
    for (sample_counts, complex_samples), regime in expected.items():
        X_q = rng.normal(size=(6, 2))
        if complex_samples:
            X_q = X_q + 1j * rng.normal(size=X_q.shape)
        F = rng.normal(size=2)
        label = f"complex={complex_samples}, counts={sample_counts}"
        assert select_sr_real_regime(6, sample_counts, complex_samples=complex_samples) == regime, label
        prepared = optimizer.prepare_wide_sr_real_system(
            X_q,
            F,
            coordinate_groups=[[q] for q in range(6)],
            diagonal_floor=0.0,
            sample_counts=sample_counts,
        )
        storage = prepared.X_preconditioned_local.flags
        assert (storage.f_contiguous and not storage.c_contiguous) == (regime == "wide"), label


def test_local_layout_is_rank_component_sample_order_with_gather_counts():
    """The local tall layout is rank-major, real block then imaginary block, with the gather counts of every rank."""
    X = np.asarray(
        [
            [0.4 + 0.2j, -0.1 + 0.7j],
            [0.3 - 0.5j, 0.8 + 0.1j],
        ]
    )
    F = np.asarray([0.6 - 0.3j, -0.2 + 0.9j])
    layout = build_tall_sr_real_layout(X, F, sample_counts=(2, 3), rank=0)

    expected_Z = np.concatenate((X.real, X.imag), axis=1)
    expected_h = np.concatenate((F.real, -F.imag))
    np.testing.assert_array_equal(layout.Z_valid_local, expected_Z)
    np.testing.assert_array_equal(layout.h_valid_local, expected_h)
    np.testing.assert_array_equal(layout.valid_counts, np.asarray([4, 6]))
    np.testing.assert_array_equal(layout.valid_displacements, np.asarray([0, 4]))


@pytest.mark.parametrize("field", ["real", "complex"], ids=["all-real", "all-complex"])
def test_cpu_and_device_tall_routes_match_independent_dense_embedding(field):
    """Cpu and device tall routes match independent dense embedding."""
    rng = np.random.default_rng(420)
    X = rng.normal(size=(7, 3))
    F = rng.normal(size=3)
    if field == "complex":
        X = X + 1.0j * rng.normal(size=X.shape)
        F = F + 1.0j * rng.normal(size=F.shape)
    epsilon = 0.17
    expected, _Z, _h = _dense_oracle(X, F, epsilon)
    layout = build_tall_sr_real_layout(X, F, sample_counts=(3,), rank=0)

    cpu_direct = solve_tall_sr_real_direct(layout.Z_valid_local, layout.h_valid_local, epsilon)
    cpu_cg, _residual, _iterations, _breakdown = solve_tall_sr_real_cg(
        layout.Z_valid_local,
        layout.h_valid_local,
        epsilon,
        max_iter=500,
        tol=ATOL_PYTHON_FP64_LOOSE,
    )
    # The world-mesh device kernels realify the samples themselves.
    device_direct = solve_device_tall_real_direct_sharded(*_shard_X_F(X, F), epsilon)
    device_cg, _device_residual, _device_iterations, _breakdown = solve_device_tall_real_cg_chunks_sharded(
        (X,),
        (F,),
        epsilon,
        max_iter=500,
        tol=ATOL_PYTHON_FP64_LOOSE,
    )
    for comparison_index, (actual) in enumerate((cpu_direct, cpu_cg, device_direct, device_cg)):
        comparison_iteration = str(comparison_index)  # noqa: F841 - captured by the GDD iteration receipt
        assert not np.iscomplexobj(actual)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=RTOL_C9_GAMMA_SR_UPDATE,
            atol=ATOL_C9_GAMMA_SR_UPDATE,
        )


def test_mixed_complex_cross_term_survives_realification_before_tall_solve():
    """Mixed complex cross term survives realification before tall solve."""
    X = np.asarray(
        [
            [0.2 + 0.8j, -0.4 + 0.3j],
            [0.5 - 0.7j, 0.1 + 0.6j],
            [0.3 + 0.2j, -0.8 - 0.1j],
            [0.7 - 0.5j, 0.2 + 0.9j],
        ]
    )
    F = np.asarray([0.6 - 0.2j, -0.1 + 0.9j])
    expected, Z, h = _dense_oracle(X, F, 0.2)
    discarded = np.linalg.solve(X.real @ X.real.T + 0.2 * np.eye(X.shape[0]), X.real @ F.real)
    actual = solve_tall_sr_real_direct(Z, h, 0.2)
    np.testing.assert_allclose(actual, expected, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    assert not np.allclose(actual, discarded, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION)


def test_tall_routes_preserve_explicit_float32_real_coordinate_dtype():
    """Tall routes preserve explicit float32 real coordinate dtype."""
    rng = np.random.default_rng(512)
    X = (rng.normal(size=(7, 3)) + 1.0j * rng.normal(size=(7, 3))).astype(np.complex64)
    F = (rng.normal(size=3) + 1.0j * rng.normal(size=3)).astype(np.complex64)
    layout = build_tall_sr_real_layout(X, F, sample_counts=(3,), rank=0)
    expected = solve_tall_sr_real_direct(layout.Z_valid_local, layout.h_valid_local, 0.2)
    actual = solve_device_tall_real_direct_sharded(*_shard_X_F(X, F), 0.2)
    assert layout.Z_valid_local.dtype == np.float32
    assert layout.h_valid_local.dtype == np.float32
    assert expected.dtype == np.float32
    assert actual.dtype == np.float32
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=RTOL_SR_SOLVER_VS_DENSE,
        atol=ATOL_SR_SOLVER_VS_DENSE_SCALE * max(float(np.max(np.abs(expected))), 1.0),
    )


def test_tall_solvers_reject_complex_duals():
    """Tall solvers reject a complex dual system."""
    with pytest.raises(ValueError, match="real"):
        solve_tall_sr_real_cg(
            np.ones((3, 2), dtype=np.complex128),
            np.ones(2),
            0.2,
            max_iter=10,
            tol=ATOL_PYTHON_FP64_LOOSE,
        )


@pytest.mark.parametrize("field_mode", ["all-real", "all-complex", "mixed"])
@pytest.mark.parametrize("cg", [False, True])
def test_internal_tall_target_decodes_once_and_never_persists_sample_dual(
    field_mode,
    cg,
):
    """Internal tall target decodes once and never persists sample dual."""
    blocks, batch = _tall_batch(field_mode)
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    result = compute_tall_sr_real_target(
        batch,
        coordinate_map,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=cg,
        cg_max_iter=500,
        cg_tol=ATOL_PYTHON_FP64_LOOSE,
    )
    Z = result.dual_layout.Z_valid_local
    h = result.dual_layout.h_valid_local
    q_pc = solve_tall_sr_real_direct(Z, h, 0.2)
    expected_q = np.zeros(coordinate_map.coordinate_size)
    expected_q[result.prepared.active_indices] = q_pc * result.prepared.inverse_sqrt_diagonal[result.prepared.active_indices]
    np.testing.assert_allclose(
        result.q,
        expected_q,
        rtol=RTOL_C9_GAMMA_SR_UPDATE,
        atol=ATOL_C9_GAMMA_SR_UPDATE,
    )
    np.testing.assert_allclose(
        coordinate_map.encode(result.q), coordinate_map.encode(expected_q), rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT
    )
    assert result.warm_start_out is None


def test_unified_target_selects_tall_from_surviving_real_dimensions():
    """Unified target selects tall from surviving real dimensions."""
    blocks, batch = _tall_batch("mixed")
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    regime, result = compute_sr_real_target(
        batch,
        coordinate_map,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=False,
        cg_max_iter=500,
        cg_tol=ATOL_PYTHON_FP64_LOOSE,
        warm_start=None,
    )
    assert regime == "tall"
    assert result.warm_start_out is None
