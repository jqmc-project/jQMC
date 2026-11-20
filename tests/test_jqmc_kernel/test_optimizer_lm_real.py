"""Centered real-coordinate LM, reduced LM, and aSR oracles."""

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

from dataclasses import replace

import numpy as np
import pytest
import scipy.linalg
from mpi4py import MPI

import jqmc.jqmc_optimizer as optimizer_module
from jqmc._setting import (
    ATOL_BIT_EXACT,
    ATOL_C9_GAMMA_SR_UPDATE,
    ATOL_LM_EIGENVECTOR_S_METRIC_UNITS,
    ATOL_PYTHON_FP64_LOOSE,
    OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF,
    OPTIMIZER_LM_REAL_ROOT_IMAG_RTOL,
    RTOL_C9_GAMMA_SR_UPDATE,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_PYTHON_FP64_ZERO,
    get_lm_correlation_eigenvalue_slack,
    get_lm_eigenvector_s_metric_unit,
)
from jqmc.jqmc_optimizer import (
    BlockDerivativeSource,
    DeviceBlockDerivativeSource,
    DeviceJackknifeBinMetadata,
    DeviceJackknifeBlockMoments,
    DeviceOptimizationBatch,
    JackknifeBinMetadata,
    JackknifeBlockMoments,
    OptimizationBatch,
    OptimizationCoordinateMap,
    asr_coefficients_from_pencil,
    build_lm_real_pencil,
    build_reduced_lm_real_pencil,
    build_reduced_lm_real_pencil_device,
    compute_twist_decomposed_real_coordinate_optimization_target,
    compute_real_coordinate_optimization_target,
    compute_sr_real_target,
    estimate_coordinate_force_jackknife,
    estimate_coordinate_force_jackknife_device,
    reduce_lm_real_pencil,
    select_asr_gamma_bounded,
    select_lm_coordinate_groups,
    solve_lm_real_pencil,
)
from jqmc.jqmc_optimizer_kernel import LMNoRealRootError

from ._optimizer_step_reference import pivoted_cholesky_correlation_survivors
from .optimizer_metadata_helpers import TEST_STATE_TOKEN, make_test_block

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

# Smallest admissible-order LM cutoff for the solver-plumbing cases below;
# ``lm_cond`` must be positive (issue #855), and on these well-conditioned
# samples it removes nothing.
_NEGLIGIBLE_LM_CONDITION_CUTOFF = 1.0e-12


def _source(block, values):
    if block.derivative_policy == "direct_real":
        return BlockDerivativeSource.build(block=block, direct=values)
    return BlockDerivativeSource.build(block=block, d_p=values)


def _lm_case(field_mode, scopes=None):
    fields = {
        "all-real": ("real", "real", "real"),
        "all-complex": ("complex", "complex"),
        "mixed": ("real", "complex", "real"),
    }[field_mode]
    scopes = tuple(scopes) if scopes is not None else ("sampler",) * len(fields)
    blocks = tuple(
        make_test_block(
            f"p{index}",
            np.asarray(0.0 if field == "real" else 0.0 + 0.0j),
            field=field,
            derivative_policy="direct_real" if field == "real" else "holomorphic",
            sync_scope=scope,
        )
        for index, (field, scope) in enumerate(zip(fields, scopes, strict=True))
    )
    rng = np.random.default_rng(340 + len(fields))
    sample_count = 11
    weights = rng.uniform(0.3, 1.7, size=sample_count)
    energies = rng.normal(size=sample_count) + 1.0j * rng.normal(size=sample_count)
    O_storage = tuple(rng.normal(size=sample_count) + 1.0j * rng.normal(size=sample_count) for _ in blocks)
    D_storage = tuple(rng.normal(size=sample_count) + 1.0j * rng.normal(size=sample_count) for _ in blocks)
    O_sources = tuple(_source(block, values) for block, values in zip(blocks, O_storage))
    D_sources = tuple(_source(block, values) for block, values in zip(blocks, D_storage))
    moments = tuple(
        JackknifeBlockMoments.build(
            block=block,
            source_sum=_source(block, weights * values),
            conjugate_source_energy_sum=_source(block, weights * np.conj(values) * energies),
        )
        for block, values in zip(blocks, O_storage)
    )
    token = f"c11-{field_mode}:1"
    metadata = JackknifeBinMetadata.build(
        blocks=blocks,
        batch_token=token,
        bin_sample_counts=np.ones(sample_count, dtype=np.int64),
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
        log_derivative_sources=O_sources,
        local_energy_derivative_sources=D_sources,
        jackknife_bin_metadata=metadata,
    )

    def expand(storage_columns):
        columns = []
        for field, values in zip(fields, storage_columns):
            columns.append(values)
            if field == "complex":
                columns.append(1.0j * values)
        return np.stack(columns, axis=1)

    return blocks, batch, expand(O_storage), expand(D_storage)


def _device_batch(blocks, batch):
    def source(block, values):
        return DeviceBlockDerivativeSource.build(
            block=block,
            direct=values.direct,
            d_p=values.d_p,
            d_pbar=values.d_pbar,
        )

    moments = tuple(
        DeviceJackknifeBlockMoments(
            name=block.name,
            source_sum=source(block, moment.source_sum),
            conjugate_source_energy_sum=source(block, moment.conjugate_source_energy_sum),
        )
        for block, moment in zip(blocks, batch.jackknife_bin_metadata.log_source_moments, strict=True)
    )
    metadata = DeviceJackknifeBinMetadata(
        batch_token=batch.batch_token,
        bin_sample_counts=batch.jackknife_bin_metadata.bin_sample_counts,
        bin_weight_sums=batch.jackknife_bin_metadata.bin_weight_sums,
        bin_weighted_energy_sums=batch.jackknife_bin_metadata.bin_weighted_energy_sums,
        log_source_moments=moments,
    )
    return DeviceOptimizationBatch(
        batch_token=batch.batch_token,
        state_token=batch.state_token,
        weights=batch.weights,
        local_energies=batch.local_energies,
        log_derivative_sources=tuple(source(block, values) for block, values in zip(blocks, batch.log_derivative_sources)),
        local_energy_derivative_sources=tuple(
            source(block, values) for block, values in zip(blocks, batch.local_energy_derivative_sources)
        ),
        jackknife_bin_metadata=metadata,
    )


def _host_stage_log_sources(blocks, batch):
    return replace(
        batch,
        log_derivative_sources=tuple(
            DeviceBlockDerivativeSource.build_host(
                block=block,
                direct=source.direct,
                d_p=source.d_p,
                d_pbar=source.d_pbar,
            )
            for block, source in zip(blocks, batch.log_derivative_sources, strict=True)
        ),
    )


def _centered_oracle(O_q, D_q, weights, energies):
    weight_sum = np.sum(weights)
    O_bar = np.einsum("n,nq->q", weights, O_q) / weight_sum
    D_bar = np.einsum("n,nq->q", weights, D_q) / weight_sum
    e_bar = np.dot(weights, energies) / weight_sum
    dO = O_q - O_bar
    dD = D_q - D_bar
    de = energies - e_bar
    S_c = np.einsum("n,na,nb->ab", weights, np.conj(dO), dO) / weight_sum
    f_c = -2.0 * np.einsum("n,na,n->a", weights, np.conj(dO), de) / weight_sum
    # NON-SYMMETRIC linear method:
    # bra row = unconjugated covariance + D_bar; bulk = K + B^dagger; NO
    # Hermitian averaging of H (doc/notes/wavefunction_optimization.md,
    # lm_storage_augmented_matrices).
    f_bra_cov = -2.0 * np.einsum("n,na,n->a", weights, dO, de) / weight_sum
    K_c = np.einsum("n,n,na,nb->ab", weights, energies, np.conj(dO), dO) / weight_sum
    B_c = np.einsum("n,na,nb->ab", weights, np.conj(dD), dO) / weight_sum
    M_c = K_c + B_c.conj().T
    H_0 = float(np.real(e_bar))
    size = O_q.shape[1]
    H_c = np.zeros((size + 1, size + 1), dtype=np.result_type(O_q, D_q, energies))
    S_aug = np.zeros_like(H_c)
    H_c[0, 0] = H_0
    H_c[1:, 0] = -0.5 * f_c
    H_c[0, 1:] = -0.5 * f_bra_cov + D_bar
    H_c[1:, 1:] = M_c
    S_aug[0, 0] = 1.0
    S_aug[1:, 1:] = S_c
    H_R = np.real(H_c)
    S_R = np.real(S_aug)
    return H_R, 0.5 * (S_R + S_R.T)


@pytest.mark.parametrize("field_mode", ["all-real", "all-complex", "mixed"])
def test_full_lm_pencil_matches_independent_centered_real_oracle(field_mode):
    """Full LM pencil matches independent centered real oracle."""
    blocks, batch, O_q, D_q = _lm_case(field_mode)
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    actual = build_lm_real_pencil(batch, coordinate_map)
    expected_H, expected_S = _centered_oracle(O_q, D_q, batch.weights, batch.local_energies)
    np.testing.assert_allclose(actual.H, expected_H, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(actual.S, expected_S, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    assert not np.iscomplexobj(actual.H)
    assert not np.iscomplexobj(actual.S)


def test_centered_real_pencil_uses_projective_metric_at_twist():
    """Centered real pencil uses projective metric at twist."""
    blocks, batch, O_q, D_q = _lm_case("mixed")
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    pencil = build_lm_real_pencil(batch, coordinate_map)
    weights = batch.weights
    weight_sum = np.sum(weights)
    O_bar = np.einsum("n,nq->q", weights, O_q) / weight_sum
    dO = O_q - O_bar
    centered_S = np.real(np.einsum("n,na,nb->ab", weights, np.conj(dO), dO) / weight_sum)
    assert np.linalg.norm(np.imag(O_bar)) > 0.0
    np.testing.assert_allclose(pencil.S[1:, 1:], centered_S, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
    np.testing.assert_allclose(pencil.conditioning_metric, centered_S, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
    np.testing.assert_array_equal(pencil.S[0, 1:], np.zeros(O_q.shape[1]))


@pytest.mark.parametrize("field_mode", ["all-real", "all-complex", "mixed"])
def test_coordinate_force_jackknife_matches_raw_delete_one_bin_oracle(field_mode):
    """Coordinate force jackknife matches raw delete one bin oracle."""
    blocks, batch, O_q, _D_q = _lm_case(field_mode)
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    force, standard_error = estimate_coordinate_force_jackknife(batch, coordinate_map)

    def direct(indices):
        weights = batch.weights[indices]
        energies = batch.local_energies[indices]
        derivatives = O_q[indices]
        weight_sum = np.sum(weights)
        energy_mean = np.dot(weights, energies) / weight_sum
        derivative_mean = np.einsum("n,nq->q", weights, derivatives) / weight_sum
        return -2.0 * np.real(
            np.einsum(
                "n,nq,n->q",
                weights,
                np.conj(derivatives - derivative_mean),
                energies - energy_mean,
            )
            / weight_sum
        )

    np.testing.assert_allclose(force, direct(np.arange(batch.weights.size)), rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    replicas = np.asarray([direct(np.delete(np.arange(batch.weights.size), index)) for index in range(batch.weights.size)])
    mean = np.mean(replicas, axis=0)
    expected_variance = (replicas.shape[0] - 1.0) / replicas.shape[0] * np.sum(np.square(replicas - mean), axis=0)
    np.testing.assert_allclose(standard_error, np.sqrt(expected_variance), rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)


def _s_metric_relative_difference(vector, reference, metric):
    """Relative S-norm ``||v - v_ref||_S / ||v_ref||_S`` of two augmented LM vectors ``(1, q)``."""
    difference = vector - reference
    return float(np.sqrt(difference @ metric @ difference / (reference @ metric @ reference)))


def test_real_lm_solver_matches_direct_generalized_eigenproblem():
    """Real LM solver matches the direct generalized eigenproblem on its returned active set.

    The reference is an independent dense general eigensolve of the augmented
    pencil restricted to the returned coordinates, with the documented selection
    (numerically real roots, S-normalized right eigenvectors, maximum reference
    weight).  The update is compared through the linearized wavefunction change
    ``|1> + sum_a q_a |O_a>`` in the S metric, with a bound scaled by
    ``kappa_2(S_A)``; energy and reference weight keep the fp64 pair.
    """
    blocks, batch, _O_q, _D_q = _lm_case("mixed")
    pencil = build_lm_real_pencil(batch, OptimizationCoordinateMap.from_blocks(blocks))
    result = solve_lm_real_pencil(pencil, condition_cutoff=OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF)
    augmented = np.concatenate(([0], np.asarray(result.active_indices) + 1))
    H_A = pencil.H[np.ix_(augmented, augmented)]
    S_A = pencil.S[np.ix_(augmented, augmented)]
    eigenvalues_c, eigenvectors_c = scipy.linalg.eig(H_A, S_A)
    imag_scale = np.maximum(np.abs(eigenvalues_c), 1.0)
    is_real_root = np.abs(np.imag(eigenvalues_c)) <= OPTIMIZER_LM_REAL_ROOT_IMAG_RTOL * imag_scale
    eigenvalues = np.real(eigenvalues_c)
    eigenvectors = np.real(eigenvectors_c)
    norms = np.sqrt(np.einsum("ij,ik,kj->j", eigenvectors, S_A, eigenvectors))
    eigenvectors = eigenvectors / norms[np.newaxis, :]
    reference_weights = np.where(is_real_root, np.square(eigenvectors[0]), -np.inf)
    best = min(range(eigenvalues.size), key=lambda index: (-reference_weights[index], eigenvalues[index], index))
    expected = eigenvectors[:, best] / eigenvectors[0, best]
    actual = np.concatenate(([1.0], np.asarray(result.q)[result.active_indices]))
    s_metric_error_units = _s_metric_relative_difference(actual, expected, S_A) / get_lm_eigenvector_s_metric_unit(
        S_A.dtype, S_A.shape[0], np.linalg.cond(S_A)
    )
    np.testing.assert_allclose(s_metric_error_units, 0.0, rtol=RTOL_PYTHON_FP64_ZERO, atol=ATOL_LM_EIGENVECTOR_S_METRIC_UNITS)
    np.testing.assert_allclose(result.predicted_energy, eigenvalues[best], rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(
        result.reference_weight, reference_weights[best], rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT
    )
    inactive = np.setdiff1d(np.arange(result.q.size), result.active_indices)
    np.testing.assert_array_equal(np.asarray(result.q)[inactive], np.zeros(inactive.size))


def _correlation_block(metric, indices):
    """Independent correlation matrix of ``metric`` on ``indices``."""
    sub = metric[np.ix_(indices, indices)]
    scale = 1.0 / np.sqrt(np.diag(sub))
    return scale[:, None] * sub * scale[None, :]


def _augmented_overlap(metric):
    """Centered augmented overlap ``diag(1, metric)`` (reference-orthogonal tangent block)."""
    overlap = np.zeros((metric.shape[0] + 1, metric.shape[0] + 1))
    overlap[0, 0] = 1.0
    overlap[1:, 1:] = metric
    return overlap


def _analytic_rank_deficient_metric(cutoff):
    """Tangent metric with analytically known conditioned survivors ``[0, 1, 2, 3, 6]``.

    Coordinates 0-3 are orthonormal directions ``e0..e3``; 4 duplicates ``e1``; 5 is
    ``(e0 + e2)/sqrt(2)``; 6 is ``0.5 e3 + sqrt(0.75) e4`` (its pair with 3 has
    ``lambda_min = 0.5 > cutoff``); 7 is ``r e0 + sqrt(1 - r^2) e5`` with
    ``1 - r = cutoff / 2`` (its pair with 0 has ``lambda_min = cutoff / 2 < cutoff``).
    Distinct per-coordinate scales make the correlation normalization visible.
    """
    basis = np.eye(6)
    close = 1.0 - 0.5 * cutoff
    columns = [
        basis[0],
        basis[1],
        basis[2],
        basis[3],
        basis[1],
        (basis[0] + basis[2]) / np.sqrt(2.0),
        0.5 * basis[3] + np.sqrt(0.75) * basis[4],
        close * basis[0] + np.sqrt(1.0 - close**2) * basis[5],
    ]
    scaled = np.stack(columns, axis=1) * np.asarray([1.0, 2.0, 0.5, 3.0, 1.5, 0.7, 1.7, 2.6])
    return scaled.T @ scaled, np.asarray([0, 1, 2, 3, 6])


def _assert_conditioned_and_maximal(metric, active, cutoff):
    """``lambda_min(C_A) >= cutoff`` and no removed coordinate can be added back (independent eigvalsh)."""
    slack = get_lm_correlation_eigenvalue_slack(np.float64, metric.shape[0])
    assert np.linalg.eigvalsh(_correlation_block(metric, active))[0] >= cutoff - slack
    for removed in np.setdiff1d(np.arange(metric.shape[0]), active):
        extended = np.sort(np.append(active, removed))
        assert np.linalg.eigvalsh(_correlation_block(metric, extended))[0] <= cutoff + slack, removed


def _solve_with_metric(metric, cutoff, seed=855):
    """Solve a centered pencil whose tangent overlap and conditioning metric are ``metric``."""
    rng = np.random.default_rng(seed)
    overlap = _augmented_overlap(metric)
    hamiltonian = overlap + 0.1 * rng.standard_normal(overlap.shape)
    pencil = optimizer_module.LMRealPencil(H=hamiltonian, S=overlap, H_0=float(hamiltonian[0, 0]))
    return solve_lm_real_pencil(pencil, condition_cutoff=cutoff)


@pytest.mark.parametrize("cutoff", [OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF, 1.0e-2])
def test_lm_conditioning_keeps_analytic_survivors_of_rank_deficient_metric(cutoff):
    """Pivoted-Cholesky conditioning keeps exactly the analytic survivors of a rank-deficient metric.

    Exact duplicates, exact linear combinations and a pair with ``lambda_min``
    below the cutoff are removed; a correlated pair above the cutoff is kept.
    The set also equals the unblocked NumPy pivoted-Cholesky oracle, satisfies
    ``lambda_min(C_A) >= cutoff`` and is maximal under single additions.
    """
    metric, survivors = _analytic_rank_deficient_metric(cutoff)
    result = _solve_with_metric(metric, cutoff)
    np.testing.assert_array_equal(result.active_indices, survivors)
    np.testing.assert_array_equal(result.active_indices, pivoted_cholesky_correlation_survivors(metric, cutoff))
    _assert_conditioned_and_maximal(metric, result.active_indices, cutoff)


def test_lm_conditioning_keeps_the_first_of_exactly_duplicated_coordinates():
    """Exact ties are broken deterministically by the first index: the earlier duplicate survives."""
    columns = np.stack([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], axis=1) * np.asarray([3.0, 0.5, 1.0])
    metric = columns.T @ columns
    result = _solve_with_metric(metric, OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF)
    np.testing.assert_array_equal(result.active_indices, [0, 2])
    np.testing.assert_array_equal(
        result.active_indices, pivoted_cholesky_correlation_survivors(metric, OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF)
    )


@pytest.mark.parametrize("seed", [8550, 8551])
@pytest.mark.parametrize("cutoff", [OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF, 1.0e-2, 0.3])
def test_lm_conditioning_matches_pivoted_cholesky_oracle_on_sampled_rank_deficient_metric(seed, cutoff):
    """On a sampled rank-deficient metric (more coordinates than samples, near-collinear columns,
    larger than one LAPACK block) the conditioned set equals the independent unblocked pivoted-Cholesky
    oracle, satisfies ``lambda_min(C_A) >= cutoff`` and is maximal under single additions."""
    rng = np.random.default_rng(seed)
    samples, independent = 90, 70
    columns = rng.standard_normal((samples, independent))
    combinations = columns[:, :30] @ rng.standard_normal((30, 30))
    near = columns[:, 30:60] + 0.05 * rng.standard_normal((samples, 30))
    derivatives = np.concatenate([columns, combinations, near], axis=1) * rng.uniform(0.2, 5.0, size=130)
    centered = derivatives - derivatives.mean(axis=0)
    metric = centered.T @ centered / samples
    result = _solve_with_metric(metric, cutoff, seed=seed)
    assert 0 < result.active_indices.size < metric.shape[0]
    np.testing.assert_array_equal(result.active_indices, pivoted_cholesky_correlation_survivors(metric, cutoff))
    _assert_conditioned_and_maximal(metric, result.active_indices, cutoff)


def _analytic_lm_pencil(metric, active, lowest_root_has_max_weight, seed=8552):
    """Centered non-symmetric pencil with analytically known eigenpairs on ``active``.

    ``H_A = S_A X Lambda X^-1`` with ``S_A = diag(1, metric[A, A])``: the real
    columns of ``X`` are right eigenvectors with eigenvalues ``-3, -1, 0.5, 2, (4)``,
    the last two columns span the complex pair ``1 +- 0.75 i`` (given the largest
    reference component, so a missing real-root filter would be visible).
    Rows and columns of removed coordinates are filled with unrelated values.
    """
    rng = np.random.default_rng(seed)
    size = active.size + 1
    real_roots = np.asarray([-3.0, -1.0, 0.5, 2.0, 4.0])[: size - 2]
    first = np.asarray([0.3, 2.0, 0.4, 0.5, 0.35])[: size - 2]
    if lowest_root_has_max_weight:
        first[[0, 1]] = first[[1, 0]]
    vectors = np.eye(size) + 0.25 * rng.standard_normal((size, size))
    vectors[0, : size - 2] = first
    vectors[0, size - 2 :] = 1.5
    eigen_block = np.zeros((size, size))
    eigen_block[: size - 2, : size - 2] = np.diag(real_roots)
    eigen_block[size - 2 :, size - 2 :] = [[1.0, 0.75], [-0.75, 1.0]]
    overlap = _augmented_overlap(metric)
    augmented = np.concatenate(([0], active + 1))
    overlap_active = overlap[np.ix_(augmented, augmented)]
    hamiltonian = rng.standard_normal(overlap.shape)
    hamiltonian[np.ix_(augmented, augmented)] = overlap_active @ vectors @ eigen_block @ np.linalg.inv(vectors)
    real_vectors = vectors[:, : size - 2]
    weights = real_vectors[0] ** 2 / np.einsum("ij,ik,kj->j", real_vectors, overlap_active, real_vectors)
    best = int(np.argmax(weights))
    pencil = optimizer_module.LMRealPencil(H=hamiltonian, S=overlap, H_0=float(hamiltonian[0, 0]))
    return pencil, overlap_active, real_vectors[:, best] / real_vectors[0, best], real_roots[best], weights[best], best


def _well_conditioned_metric(size, seed=8553):
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((size, size)) + 3.0 * np.eye(size)
    return factor.T @ factor


@pytest.mark.parametrize("lowest_root_has_max_weight", [False, True])
@pytest.mark.parametrize("metric_case", ["full-rank", "rank-deficient"])
def test_lm_solver_selects_analytic_max_weight_real_eigenpair(metric_case, lowest_root_has_max_weight):
    """The LM update is the maximum-reference-weight real eigenpair of the pencil on the returned set.

    The pencil is constructed from known right eigenvectors, so the reference is
    analytic: the selected root is the real root of largest S-normalized
    reference weight (also when that is not the lowest real root), complex pairs
    are never selected, and removed coordinates carry a zero update.  The update
    is compared in the S metric with a ``kappa_2(S_A)``-scaled bound; energy and
    weight keep the fp64 pair.
    """
    if metric_case == "full-rank":
        metric = _well_conditioned_metric(6)
        survivors = np.arange(6)
    else:
        metric, survivors = _analytic_rank_deficient_metric(OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF)
    pencil, overlap_active, expected, energy, weight, best = _analytic_lm_pencil(metric, survivors, lowest_root_has_max_weight)
    assert (best == 0) == lowest_root_has_max_weight
    result = solve_lm_real_pencil(pencil, condition_cutoff=OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF)
    np.testing.assert_array_equal(result.active_indices, survivors)
    actual = np.concatenate(([1.0], np.asarray(result.q)[survivors]))
    s_metric_error_units = _s_metric_relative_difference(actual, expected, overlap_active) / get_lm_eigenvector_s_metric_unit(
        overlap_active.dtype, overlap_active.shape[0], np.linalg.cond(overlap_active)
    )
    np.testing.assert_allclose(s_metric_error_units, 0.0, rtol=RTOL_PYTHON_FP64_ZERO, atol=ATOL_LM_EIGENVECTOR_S_METRIC_UNITS)
    np.testing.assert_allclose(result.predicted_energy, energy, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(result.reference_weight, weight, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    removed = np.setdiff1d(np.arange(metric.shape[0]), survivors)
    np.testing.assert_array_equal(np.asarray(result.q)[removed], np.zeros(removed.size))


@pytest.mark.parametrize("condition_cutoff", [0.0, -1.0e-3, np.nan, np.inf, 1.0, 1.5])
def test_lm_solver_rejects_condition_cutoff_outside_open_unit_interval(condition_cutoff):
    """``lm_cond`` outside ``(0, 1)`` is rejected with a clear error instead of an ill-posed or empty LM step."""
    blocks, batch, _O_q, _D_q = _lm_case("mixed")
    pencil = build_lm_real_pencil(batch, OptimizationCoordinateMap.from_blocks(blocks))
    with pytest.raises(ValueError, match=r"lm_cond must be .*open interval \(0, 1\)"):
        solve_lm_real_pencil(pencil, condition_cutoff=condition_cutoff)


@pytest.mark.parametrize("field_mode", ["all-real", "all-complex", "mixed"])
def test_centered_lm_parameter_update_matches_original_solver(field_mode):
    """Solver-plumbing continuity: on a SYMMETRIZED pencil the general-eig
    solver must reproduce the original Hermitian solve_linear_method update.

    The production pencil is non-symmetric (it retains the
    finite <de_L/dc>; doc/notes/wavefunction_optimization.md), so this test
    symmetrizes the built pencil first -- it checks the SOLVER equivalence on
    Hermitian input, not the estimator definition."""
    blocks, batch, _O_q, _D_q = _lm_case(field_mode)
    pencil_raw = build_lm_real_pencil(batch, OptimizationCoordinateMap.from_blocks(blocks))
    H_sym = 0.5 * (pencil_raw.H + pencil_raw.H.T)
    pencil = optimizer_module.LMRealPencil(
        H=H_sym,
        S=pencil_raw.S,
        H_0=pencil_raw.H_0,
        conditioning_metric=pencil_raw.conditioning_metric,
    )
    condition_cutoff = 1.0e-12
    current = solve_lm_real_pencil(pencil, condition_cutoff=condition_cutoff)
    original_q, original_energy, original_weight = optimizer_module._kernel.solve_linear_method(
        pencil.H_0,
        -2.0 * pencil.H[1:, 0],
        pencil.S[1:, 1:],
        pencil.H[1:, 1:],
        np.zeros_like(pencil.H[1:, 1:]),
        condition_cutoff,
    )
    np.testing.assert_allclose(current.q, original_q, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(current.predicted_energy, original_energy, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(current.reference_weight, original_weight, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)


def test_lm_solver_rejects_reference_tangent_overlap_cross_terms():
    """LM solver rejects reference tangent overlap cross terms."""
    pencil = optimizer_module.LMRealPencil(
        H=np.eye(2),
        S=np.asarray([[1.0, 0.1], [0.1, 1.0]]),
        H_0=1.0,
    )
    with pytest.raises(ValueError, match="centered overlap"):
        solve_lm_real_pencil(pencil, condition_cutoff=OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF)


@pytest.mark.parametrize("augmented_dimension", [2, 4])
def test_lm_solver_reports_no_real_root_for_all_complex_spectrum(augmented_dimension):
    """A centered pencil with an all-complex spectrum yields no decoded LM update.

    ``H = S^(1/2) R S^(1/2)`` with ``R`` block diagonal in rotation-scaling
    blocks ``[[a, -b], [b, a]]`` (``b != 0``) and a diagonal centered ``S``, so
    every root ``a +- i b`` of ``H x = lambda S x`` is complex; the
    independent dense ``scipy.linalg.eigvals(H, S)`` confirms it.  Every
    coordinate survives conditioning (diagonal ``S``: unit correlation
    matrix).  A complex eigenvector has an arbitrary phase, so the solver must
    report that no real root exists instead of decoding ``Re(v)``.
    """
    blocks = [(-0.7, 0.4), (1.3, 0.25)][: augmented_dimension // 2]
    rotation = scipy.linalg.block_diag(*[np.asarray([[a, -b], [b, a]]) for a, b in blocks])
    overlap_diagonal = np.asarray([1.0, 2.0, 0.5, 3.0])[:augmented_dimension]
    sqrt_overlap = np.diag(np.sqrt(overlap_diagonal))
    H = sqrt_overlap @ rotation @ sqrt_overlap
    S = np.diag(overlap_diagonal)
    roots = scipy.linalg.eigvals(H, S)
    assert np.all(np.abs(np.imag(roots)) > OPTIMIZER_LM_REAL_ROOT_IMAG_RTOL * np.maximum(np.abs(roots), 1.0))
    pencil = optimizer_module.LMRealPencil(H=H, S=S, H_0=float(H[0, 0]))
    with pytest.raises(LMNoRealRootError, match="no numerically real eigenvalue"):
        solve_lm_real_pencil(pencil, condition_cutoff=OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF)


def test_asr_contraction_rejects_cross_terms_above_centered_roundoff_policy():
    """Asr contraction rejects cross terms above centered roundoff policy."""
    H = np.eye(2)
    S = np.asarray([[1.0, 1.0e-10], [1.0e-10, 1.0]])
    with pytest.raises(ValueError, match="centered pencil"):
        asr_coefficients_from_pencil(H, S, np.ones(1))


@pytest.mark.parametrize("condition_cutoff", [_NEGLIGIBLE_LM_CONDITION_CUTOFF, 1.0e-3])
def test_centered_lm_builder_and_update_are_invariant_to_reference_shift(condition_cutoff):
    """Centering must remove arbitrary constant reference components from O."""
    _blocks, batch, O_q, D_q = _lm_case("mixed")
    build = optimizer_module._build_lm_real_pencil_from_coordinate_arrays
    baseline = build(O_q, D_q, batch.weights, batch.local_energies)

    shift = np.asarray([20.0, -15.0, 30.0, -25.0])
    shifted = build(
        O_q + shift[None, :],
        D_q,
        batch.weights,
        batch.local_energies,
    )

    baseline_result = solve_lm_real_pencil(baseline, condition_cutoff=condition_cutoff)
    shifted_result = solve_lm_real_pencil(shifted, condition_cutoff=condition_cutoff)
    np.testing.assert_allclose(shifted.H, baseline.H, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
    np.testing.assert_allclose(shifted.S, baseline.S, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
    np.testing.assert_allclose(
        shifted.conditioning_metric, baseline.conditioning_metric, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE
    )
    np.testing.assert_allclose(shifted_result.q, baseline_result.q, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
    np.testing.assert_allclose(
        shifted_result.predicted_energy,
        baseline_result.predicted_energy,
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_LOOSE,
    )
    np.testing.assert_allclose(
        shifted_result.reference_weight,
        baseline_result.reference_weight,
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_LOOSE,
    )


def test_reduced_precontracted_pencil_matches_explicit_full_congruence():
    """Reduced precontracted pencil matches explicit full congruence."""
    blocks, batch, _O_q, _D_q = _lm_case("mixed")
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    full = build_lm_real_pencil(batch, coordinate_map)
    g = np.linspace(0.2, 0.2 * coordinate_map.coordinate_size, coordinate_map.coordinate_size)
    selected = np.asarray([0, 2, coordinate_map.coordinate_size - 1], dtype=np.int64)
    explicit, A = reduce_lm_real_pencil(full, g, selected)
    precontracted, precontracted_A = build_reduced_lm_real_pencil(batch, coordinate_map, g, selected)
    np.testing.assert_array_equal(precontracted_A, A)
    np.testing.assert_allclose(precontracted.H, explicit.H, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(precontracted.S, explicit.S, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(
        precontracted.conditioning_metric, explicit.conditioning_metric, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT
    )


def test_device_batch_matches_cpu_reference_for_transform_jackknife_sr_and_reduced_lm():
    """Device batch matches cpu reference for transform jackknife SR and reduced LM."""
    blocks, batch, O_q, D_q = _lm_case("mixed")
    device_batch = _device_batch(blocks, batch)
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    np.testing.assert_allclose(
        np.asarray(coordinate_map.transform_derivatives_device(device_batch.log_derivative_sources)),
        O_q,
        rtol=RTOL_PYTHON_FP64_LOOSE,
        atol=ATOL_BIT_EXACT,
    )
    np.testing.assert_allclose(
        np.asarray(coordinate_map.transform_derivatives_device(device_batch.local_energy_derivative_sources)),
        D_q,
        rtol=RTOL_PYTHON_FP64_LOOSE,
        atol=ATOL_BIT_EXACT,
    )
    contraction = np.arange(
        coordinate_map.coordinate_size * 3,
        dtype=float,
    ).reshape(coordinate_map.coordinate_size, 3)
    host_energy_sources = tuple(
        DeviceBlockDerivativeSource(
            name=source.name,
            direct=None if source.direct is None else np.asarray(source.direct),
            d_p=None if source.d_p is None else np.asarray(source.d_p),
            d_pbar=None if source.d_pbar is None else np.asarray(source.d_pbar),
        )
        for source in device_batch.local_energy_derivative_sources
    )
    np.testing.assert_allclose(
        coordinate_map.transform_host_derivatives_reduced(
            host_energy_sources,
            contraction,
            sample_batch_size=2,
        ),
        np.asarray(
            coordinate_map.transform_derivatives_device(
                device_batch.local_energy_derivative_sources,
                right=contraction,
            )
        ),
        rtol=RTOL_PYTHON_FP64_LOOSE,
        atol=ATOL_BIT_EXACT,
    )
    force_cpu, error_cpu = estimate_coordinate_force_jackknife(batch, coordinate_map)
    force_device, error_device = estimate_coordinate_force_jackknife_device(device_batch, coordinate_map)
    np.testing.assert_allclose(force_device, force_cpu, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(error_device, error_cpu, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)

    full_cpu = build_lm_real_pencil(batch, coordinate_map)
    full_device_H, full_device_S, full_device_H_0, full_device_conditioning = optimizer_module._kernel.build_device_lm_pencil(
        coordinate_map.transform_derivatives_device(device_batch.log_derivative_sources),
        coordinate_map.transform_derivatives_device(device_batch.local_energy_derivative_sources),
        device_batch.weights,
        device_batch.local_energies,
    )
    for comparison_index, (actual, expected) in enumerate(
        (
            (full_device_H, full_cpu.H),
            (full_device_S, full_cpu.S),
            (full_device_H_0, full_cpu.H_0),
            (full_device_conditioning, full_cpu.conditioning_metric),
        )
    ):
        comparison_iteration = str(comparison_index)  # noqa: F841 - captured by the GDD iteration receipt
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=RTOL_PYTHON_FP64_ZERO,
            atol=ATOL_PYTHON_FP64_LOOSE,
        )

    reduced_direction = np.linspace(0.2, 0.2 * coordinate_map.coordinate_size, coordinate_map.coordinate_size)
    reduced_selected = np.asarray([0, 2, coordinate_map.coordinate_size - 1], dtype=np.int64)
    reduced_cpu, reduced_cpu_A = build_reduced_lm_real_pencil(
        batch,
        coordinate_map,
        reduced_direction,
        reduced_selected,
    )
    reduced_device, reduced_device_A = build_reduced_lm_real_pencil_device(
        device_batch,
        coordinate_map,
        reduced_direction,
        reduced_selected,
    )
    np.testing.assert_array_equal(reduced_device_A, reduced_cpu_A)
    np.testing.assert_allclose(
        reduced_device.conditioning_metric,
        reduced_cpu.conditioning_metric,
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )

    common = dict(
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=True,
        cg_max_iter=500,
        cg_tol=1.0e-11,
        warm_start=None,
        use_lm=True,
        lm_subspace_dim=2,
        lm_condition_cutoff=_NEGLIGIBLE_LM_CONDITION_CUTOFF,
        energy_standard_error=0.1,
    )
    cpu = compute_real_coordinate_optimization_target(batch, coordinate_map, **common)
    device = compute_real_coordinate_optimization_target(device_batch, coordinate_map, **common)
    np.testing.assert_allclose(
        device.sr_q,
        cpu.sr_q,
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )
    np.testing.assert_allclose(
        device.q,
        cpu.q,
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )
    assert device.selected_coordinate_groups == cpu.selected_coordinate_groups


@pytest.mark.parametrize("lm_subspace_dim", [0, 2])
def test_reduced_lm_routes_do_not_build_the_full_coordinate_pencil(monkeypatch, lm_subspace_dim):
    """Reduced LM routes do not build the full coordinate pencil."""
    blocks, batch, _O_q, _D_q = _lm_case("mixed")
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)

    def reject_full_pencil(*args, **kwargs):
        raise AssertionError("reduced LM/aSR must not build the full coordinate pencil")

    monkeypatch.setattr(optimizer_module, "_build_lm_real_pencil", reject_full_pencil)
    compute_real_coordinate_optimization_target(
        batch,
        coordinate_map,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=False,
        cg_max_iter=500,
        cg_tol=ATOL_PYTHON_FP64_LOOSE,
        warm_start=None,
        use_lm=True,
        lm_subspace_dim=lm_subspace_dim,
        lm_condition_cutoff=_NEGLIGIBLE_LM_CONDITION_CUTOFF,
        energy_standard_error=0.1,
    )


def test_atomic_group_selection_includes_complex_pair_and_boundary_ties():
    """Atomic group selection includes complex pair and boundary ties."""
    scores = np.asarray([3.0, 3.0, 1.0, 4.0, 4.0, 0.5])
    groups = ((0,), (1, 2), (3, 4), (5,))
    selected = select_lm_coordinate_groups(scores, groups, coordinate_budget=3)
    np.testing.assert_array_equal(selected, np.asarray([3, 4, 0, 1, 2]))


def test_single_k_lm_uses_one_combined_coordinate_budget(monkeypatch):
    """Single k LM uses one combined coordinate budget."""
    blocks, batch, _O_q, _D_q = _lm_case("mixed")
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    calls = []
    original = optimizer_module.select_lm_coordinate_groups

    def record_selection(scores, coordinate_groups, *, coordinate_budget):
        calls.append((np.asarray(scores).shape, tuple(coordinate_groups), coordinate_budget))
        return original(scores, coordinate_groups, coordinate_budget=coordinate_budget)

    monkeypatch.setattr(optimizer_module, "select_lm_coordinate_groups", record_selection)
    compute_real_coordinate_optimization_target(
        batch,
        coordinate_map,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=False,
        cg_max_iter=500,
        cg_tol=1.0e-12,
        warm_start=None,
        use_lm=True,
        lm_subspace_dim=2,
        lm_condition_cutoff=_NEGLIGIBLE_LM_CONDITION_CUTOFF,
        energy_standard_error=0.1,
    )

    assert calls == [((coordinate_map.coordinate_size,), coordinate_map.coordinate_groups, 2)]


def test_planb_lm_applies_coordinate_budget_independently_per_scope(monkeypatch):
    """Planb LM applies coordinate budget independently per scope."""
    blocks, batch, _O_q, _D_q = _lm_case("mixed", scopes=("world", "sampler", "world"))
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    calls = []

    class RecordingComm:
        def __init__(self):
            self.broadcasts = []

        def allreduce(self, value, op=None):
            del op
            return value

        def allgather(self, value):
            return [value]

        def Allreduce(self, send, receive, op=None):
            del op
            np.copyto(np.asarray(receive[0]), np.asarray(send[0]))

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def bcast(self, value, root=0):
            self.broadcasts.append((np.asarray(value).copy(), root))
            return value

    world_comm = RecordingComm()
    sampler_comm = RecordingComm()

    def select_first_group(scores, coordinate_groups, *, coordinate_budget):
        calls.append((np.asarray(scores).shape, tuple(coordinate_groups), coordinate_budget))
        return np.asarray(coordinate_groups[0], dtype=np.int64)

    monkeypatch.setattr(optimizer_module, "select_lm_coordinate_groups", select_first_group)
    result = compute_twist_decomposed_real_coordinate_optimization_target(
        batch,
        coordinate_map,
        sampler_comm=sampler_comm,
        world_comm=world_comm,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=False,
        cg_max_iter=500,
        cg_tol=1.0e-12,
        use_lm=True,
        lm_subspace_dim=1,
        lm_condition_cutoff=_NEGLIGIBLE_LM_CONDITION_CUTOFF,
        energy_standard_error=0.1,
    )

    assert calls == [
        ((2,), ((0,), (1,)), 1),
        ((2,), ((0, 1),), 1),
    ]
    assert len(world_comm.broadcasts) == 1
    assert len(sampler_comm.broadcasts) == 1
    assert result.selected_coordinate_groups == ((0,), (1, 2))


def test_planb_lm_scope_selection_uses_root_consensus_when_rank_scores_differ():
    """Planb LM scope selection uses root consensus when rank scores differ."""
    blocks, _batch, _O_q, _D_q = _lm_case("mixed", scopes=("world", "sampler", "world"))
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    shared = optimizer_module._coordinate_scope_indices(coordinate_map, "world")
    broadcast_value = []

    class SequentialBroadcastComm:
        def __init__(self, rank):
            self.rank = rank

        def Get_rank(self):
            return self.rank

        def bcast(self, value, root=0):
            assert root == 0
            if self.rank == root:
                broadcast_value.append(np.asarray(value).copy())
                return value
            return broadcast_value[0]

    root_scores = np.asarray([9.0, 0.0, 0.0, 1.0])
    other_rank_scores = np.asarray([1.0, 0.0, 0.0, 9.0])
    root_selected = optimizer_module._select_planb_lm_scope_coordinates(
        root_scores,
        coordinate_map,
        shared,
        coordinate_budget=1,
        comm=SequentialBroadcastComm(0),
    )
    other_rank_selected = optimizer_module._select_planb_lm_scope_coordinates(
        other_rank_scores,
        coordinate_map,
        shared,
        coordinate_budget=1,
        comm=SequentialBroadcastComm(1),
    )

    np.testing.assert_array_equal(root_selected, np.asarray([0]))
    np.testing.assert_array_equal(other_rank_selected, root_selected)


def test_centered_asr_selects_reference_connected_stationary_root():
    """Centered asr selects reference connected stationary root."""
    H = np.asarray([[-0.8, -0.3, 0.2], [-0.3, 1.4, 0.1], [0.2, 0.1, 0.9]])
    S = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.2, 0.2], [0.0, 0.2, 0.8]])
    g = np.asarray([0.7, -0.4])
    H_0, H_1, H_2, S_2 = asr_coefficients_from_pencil(H, S, g)
    result = select_asr_gamma_bounded(
        H_0,
        H_1,
        H_2,
        S_2,
        gamma_max=0.8,
        fallback_gamma=0.1,
    )
    B = H_2 - H_0 * S_2
    discriminant = B**2 + 4.0 * H_1**2 * S_2
    denominator = -2.0 * H_1 * S_2
    roots = ((-B + np.sqrt(discriminant)) / denominator, (-B - np.sqrt(discriminant)) / denominator)
    expected_gamma = min(roots, key=abs)
    assert expected_gamma > 0.0

    def energy(gamma):
        return (H_0 + 2.0 * gamma * H_1 + gamma * gamma * H_2) / (1.0 + gamma * gamma * S_2)

    np.testing.assert_allclose(result.gamma, expected_gamma, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(
        result.predicted_energy, energy(expected_gamma), rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT
    )
    assert not result.used_fallback


def test_centered_asr_accepts_small_negative_reference_root():
    # H32 Gamma regression data: the lower-energy stationary root is almost a
    # pure tangent state (gamma ~ 1991), while the reference-connected root is
    # small and NEGATIVE.  With the non-symmetric pencil, a finite negative
    # reference-connected root is a LEGITIMATE
    # anti-SR step -- the model has detected that the covariance direction
    # points uphill -- and is accepted as long as |gamma| <= gamma_max;
    # rejecting it as anti-descent and taking the +0.1 fallback is exactly the
    # uphill walk the non-symmetric pencil diagnoses.  The nearly orthogonal gamma ~ 1991 root must still be
    # rejected by the smaller-|gamma| reference-connection rule.
    """Centered asr accepts small negative reference root."""
    coefficients = (-7.157035387508453, -0.00446417544, -0.143272062, 0.00892834934)

    result = select_asr_gamma_bounded(
        *coefficients,
        gamma_max=1.0e6,
        fallback_gamma=0.1,
    )

    H_0, H_1, H_2, S_2 = coefficients
    B = H_2 - H_0 * S_2
    denom = -2.0 * H_1 * S_2
    expected_gamma = (-B - np.sqrt(B**2 + 4.0 * H_1**2 * S_2)) / denom  # smaller-|gamma| root
    assert expected_gamma < 0.0
    np.testing.assert_allclose(result.gamma, expected_gamma, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    assert not result.used_fallback


def test_bounded_centered_asr_rejects_negative_metric():
    """A negative ASR metric selects the declared bounded fallback gamma."""
    result = select_asr_gamma_bounded(
        -1.0,
        -0.2,
        0.5,
        -1.0,
        gamma_max=1.0,
        fallback_gamma=0.1,
    )
    np.testing.assert_allclose(result.gamma, 0.1, rtol=RTOL_C9_GAMMA_SR_UPDATE, atol=ATOL_C9_GAMMA_SR_UPDATE)
    assert result.used_fallback


@pytest.mark.parametrize("field_mode", ["all-real", "all-complex", "mixed"])
def test_complete_plain_sr_target_encodes_one_real_coordinate_update(field_mode):
    """Complete plain SR target encodes one real coordinate update."""
    blocks, batch, _O_q, _D_q = _lm_case(field_mode)
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    _regime, sr = compute_sr_real_target(
        batch,
        coordinate_map,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=False,
        cg_max_iter=500,
        cg_tol=1.0e-12,
        warm_start=None,
    )
    result = compute_real_coordinate_optimization_target(
        batch,
        coordinate_map,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=False,
        cg_max_iter=500,
        cg_tol=1.0e-12,
        warm_start=None,
        use_lm=False,
        lm_subspace_dim=0,
        lm_condition_cutoff=1.0e-3,
        energy_standard_error=0.1,
    )
    np.testing.assert_allclose(result.q, sr.q, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    assert not np.iscomplexobj(result.q)


def test_complete_asr_target_scales_depreconditioned_sr_direction():
    """Complete asr target scales depreconditioned SR direction."""
    blocks, batch, _O_q, _D_q = _lm_case("mixed")
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    result = compute_real_coordinate_optimization_target(
        batch,
        coordinate_map,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=False,
        cg_max_iter=500,
        cg_tol=1.0e-12,
        warm_start=None,
        use_lm=True,
        lm_subspace_dim=0,
        lm_condition_cutoff=1.0e-3,
        energy_standard_error=0.1,
        asr_gamma_max=0.8,
    )
    np.testing.assert_allclose(result.q, result.accepted_scale * result.sr_q, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)


@pytest.mark.parametrize("lm_subspace_dim", [2, -1])
def test_rejected_lm_candidate_uses_asr_fallback(monkeypatch, lm_subspace_dim):
    """Rejected LM candidate uses asr fallback."""
    blocks, batch, _O_q, _D_q = _lm_case("all-real")
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    calls = []

    def rejected_lm(pencil, *, condition_cutoff):
        del condition_cutoff
        size = pencil.H.shape[0] - 1
        return optimizer_module.LMRealResult(
            q=np.zeros(size),
            predicted_energy=pencil.H_0 - 1.0,
            reference_weight=0.5,
            active_indices=np.arange(size),
        )

    def selected_asr(*coefficients, gamma_max, fallback_gamma):
        calls.append((coefficients, gamma_max, fallback_gamma))
        return optimizer_module.ASRBoundedResult(
            gamma=0.37,
            predicted_energy=-123.0,
            used_fallback=False,
        )

    monkeypatch.setattr(optimizer_module, "solve_lm_real_pencil", rejected_lm)
    monkeypatch.setattr(optimizer_module, "select_asr_gamma_bounded", selected_asr)
    result = compute_real_coordinate_optimization_target(
        batch,
        coordinate_map,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=False,
        cg_max_iter=500,
        cg_tol=1.0e-12,
        warm_start=None,
        use_lm=True,
        lm_subspace_dim=lm_subspace_dim,
        lm_condition_cutoff=1.0e-3,
        energy_standard_error=0.1,
        asr_gamma_max=0.8,
        asr_fallback_gamma=0.1,
    )

    assert len(calls) == 1
    np.testing.assert_allclose(result.q, 0.37 * result.sr_q, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(result.accepted_scale, 0.37, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(result.predicted_energy, -123.0, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    assert result.lm_reference_weight == 0.5
    assert "aSR fallback gamma=0.37000000" in result.rejection_reason
    expected_norm = abs(result.accepted_scale) * np.sqrt(max(float(calls[0][0][-1]), 0.0))
    np.testing.assert_allclose(
        result.linearized_wavefunction_norm, expected_norm, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT
    )


def test_planb_rejected_lm_candidate_uses_asr_fallback_in_each_scope(monkeypatch):
    """Planb rejected LM candidate uses asr fallback in each scope."""
    blocks, batch, _O_q, _D_q = _lm_case("mixed", scopes=("world", "sampler", "world"))
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    calls = []

    def rejected_lm(pencil, *, condition_cutoff):
        del condition_cutoff
        size = pencil.H.shape[0] - 1
        return optimizer_module.LMRealResult(
            q=np.zeros(size),
            predicted_energy=pencil.H_0 - 1.0,
            reference_weight=0.5,
            active_indices=np.arange(size),
        )

    def selected_asr(*coefficients, gamma_max, fallback_gamma):
        calls.append((coefficients, gamma_max, fallback_gamma))
        return optimizer_module.ASRBoundedResult(
            gamma=0.37,
            predicted_energy=-123.0,
            used_fallback=False,
        )

    monkeypatch.setattr(optimizer_module, "solve_lm_real_pencil", rejected_lm)
    monkeypatch.setattr(optimizer_module, "select_asr_gamma_bounded", selected_asr)
    result = compute_twist_decomposed_real_coordinate_optimization_target(
        batch,
        coordinate_map,
        sampler_comm=MPI.COMM_SELF,
        world_comm=MPI.COMM_SELF,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=False,
        cg_max_iter=500,
        cg_tol=1.0e-12,
        use_lm=True,
        lm_subspace_dim=2,
        lm_condition_cutoff=1.0e-3,
        energy_standard_error=0.1,
        asr_gamma_max=0.8,
        asr_fallback_gamma=0.1,
    )

    assert len(calls) == 2
    np.testing.assert_allclose(result.q, 0.37 * result.sr_q, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(result.accepted_scale, 0.37, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(result.predicted_energy, -123.0, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    assert result.rejection_reason.count("aSR fallback gamma=0.37000000") == 2


def _all_complex_lm_case(scopes):
    """One real coordinate per block; each block's centered 2x2 LM pencil has a complex root pair.

    The local-energy derivative of block ``a`` is a constant ``c_a``, so it
    only shifts the bra entry ``H[0, a]`` by ``c_a`` (``D - D_bar = 0``).  With
    ``g = H[a, 0]``, ``K = H[a, a]`` and ``s = S[a, a]`` of the ``c_a = 0``
    pencil (independent oracle :func:`_centered_oracle`), ``c_a = -g - kappa / g``
    with ``kappa = s (H_0 - K / s)^2 / 4 + s`` gives ``H[0, a] H[a, 0] = -kappa``,
    i.e. the 2x2 pencil has the roots ``(H_0 + K / s) / 2 +- i``.
    """
    blocks = tuple(
        make_test_block(f"p{index}", np.asarray(0.0), field="real", derivative_policy="direct_real", sync_scope=scope)
        for index, scope in enumerate(scopes)
    )
    rng = np.random.default_rng(882)
    sample_count = 11
    weights = rng.uniform(0.3, 1.7, size=sample_count)
    energies = rng.normal(size=sample_count)
    O_q = rng.normal(size=(sample_count, len(blocks)))
    H_free, S_free = _centered_oracle(O_q, np.zeros_like(O_q), weights, energies)
    D_q = np.empty_like(O_q)
    for coordinate in range(len(blocks)):
        g = H_free[coordinate + 1, 0]
        K = H_free[coordinate + 1, coordinate + 1]
        s = S_free[coordinate + 1, coordinate + 1]
        kappa = s * (H_free[0, 0] - K / s) ** 2 / 4.0 + s
        D_q[:, coordinate] = -g - kappa / g
    O_sources = tuple(_source(block, O_q[:, index]) for index, block in enumerate(blocks))
    D_sources = tuple(_source(block, D_q[:, index]) for index, block in enumerate(blocks))
    moments = tuple(
        JackknifeBlockMoments.build(
            block=block,
            source_sum=_source(block, weights * O_q[:, index]),
            conjugate_source_energy_sum=_source(block, weights * O_q[:, index] * energies),
        )
        for index, block in enumerate(blocks)
    )
    token = "all-complex:1"
    metadata = JackknifeBinMetadata.build(
        blocks=blocks,
        batch_token=token,
        bin_sample_counts=np.ones(sample_count, dtype=np.int64),
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
        log_derivative_sources=O_sources,
        local_energy_derivative_sources=D_sources,
        jackknife_bin_metadata=metadata,
    )
    return blocks, batch, O_q, D_q


def _reference_connected_asr_step(H, S, direction):
    """Independent aSR reference: the symmetric 2x2 projective model on the line ``1 + gamma g``.

    The model ``[[H_0, H_1], [H_1, H_2]] w = E diag(1, S_2) w`` with
    ``H_1 = (H[0, 1:] g + H[1:, 0] g) / 2`` is solved by a dense symmetric
    generalized eigensolve; the reference-connected root is the S-normalized
    eigenvector with the larger reference weight ``w_0^2``.  Returns
    ``(gamma, E)`` with ``gamma = w_1 / w_0``.
    """
    model_H = np.asarray(
        [
            [H[0, 0], 0.5 * (H[0, 1:] @ direction + H[1:, 0] @ direction)],
            [0.5 * (H[0, 1:] @ direction + H[1:, 0] @ direction), direction @ H[1:, 1:] @ direction],
        ]
    )
    model_S = np.diag([1.0, direction @ S[1:, 1:] @ direction])
    energies, vectors = scipy.linalg.eigh(model_H, model_S)
    best = int(np.argmax(np.square(vectors[0])))
    return vectors[1, best] / vectors[0, best], energies[best]


def _assert_complex_root_pair(H, S):
    roots = scipy.linalg.eigvals(H, S)
    assert np.all(np.abs(np.imag(roots)) > OPTIMIZER_LM_REAL_ROOT_IMAG_RTOL * np.maximum(np.abs(roots), 1.0))


_OPEN_LM_GATES = {"energy_standard_error": 1.0e3, "lm_reference_weight_min": 0.0}
_NO_REAL_ROOT_ASR_GAMMA_MAX = 10.0


def test_single_k_lm_without_real_root_takes_asr_rejection_step():
    """Single-k full LM with an all-complex pencil takes the aSR rejection step.

    Both acceptance gates are opened (huge energy error, zero minimum
    reference weight), so only the absence of a numerically real root can
    reject the LM step.  The applied update must be the reference-connected
    aSR step on the SR direction from the independent 2x2 model, the rejection
    reason must be recorded (its "aSR fallback gamma=" form also shows that the
    model root was neither invalid nor capped by ``asr_gamma_max``) and no LM
    reference weight reported.
    """
    blocks, batch, O_q, D_q = _all_complex_lm_case(("sampler",))
    H, S = _centered_oracle(O_q, D_q, batch.weights, batch.local_energies)
    _assert_complex_root_pair(H, S)
    result = compute_real_coordinate_optimization_target(
        batch,
        OptimizationCoordinateMap.from_blocks(blocks),
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=False,
        cg_max_iter=500,
        cg_tol=1.0e-12,
        warm_start=None,
        use_lm=True,
        lm_subspace_dim=-1,
        lm_condition_cutoff=OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF,
        asr_gamma_max=_NO_REAL_ROOT_ASR_GAMMA_MAX,
        asr_fallback_gamma=0.1,
        **_OPEN_LM_GATES,
    )
    gamma, energy = _reference_connected_asr_step(H, S, np.asarray(result.sr_q))
    np.testing.assert_allclose(result.q, gamma * np.asarray(result.sr_q), rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(result.predicted_energy, energy, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    assert result.lm_reference_weight is None
    assert result.rejection_reason.startswith("no numerically real LM root; aSR fallback gamma=")


def test_twist_lm_without_real_root_takes_asr_rejection_step_in_each_scope():
    """Twist-decomposed full LM with all-complex scope pencils takes the aSR step per scope.

    One shared (world) and one twist-local (sampler) coordinate; each scope
    pencil is the 2x2 all-complex pencil of :func:`_all_complex_lm_case`.  With
    both acceptance gates opened, each scope must apply its reference-connected
    aSR step from the independent 2x2 model and record the rejection reason.
    """
    blocks, batch, O_q, D_q = _all_complex_lm_case(("world", "sampler"))
    result = compute_twist_decomposed_real_coordinate_optimization_target(
        batch,
        OptimizationCoordinateMap.from_blocks(blocks),
        sampler_comm=MPI.COMM_SELF,
        world_comm=MPI.COMM_SELF,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=False,
        cg_max_iter=500,
        cg_tol=1.0e-12,
        use_lm=True,
        lm_subspace_dim=-1,
        lm_condition_cutoff=OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF,
        asr_gamma_max=_NO_REAL_ROOT_ASR_GAMMA_MAX,
        asr_fallback_gamma=0.1,
        **_OPEN_LM_GATES,
    )
    sr_q = np.asarray(result.sr_q)
    expected_q = np.empty_like(sr_q)
    expected_energies = []
    for coordinate in range(len(blocks)):
        H, S = _centered_oracle(O_q[:, [coordinate]], D_q[:, [coordinate]], batch.weights, batch.local_energies)
        _assert_complex_root_pair(H, S)
        gamma, energy = _reference_connected_asr_step(H, S, sr_q[[coordinate]])
        expected_q[coordinate] = gamma * sr_q[coordinate]
        expected_energies.append(energy)
    np.testing.assert_allclose(result.q, expected_q, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(
        result.predicted_energy, min(expected_energies), rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT
    )
    assert result.lm_reference_weight is None
    assert result.rejection_reason.count("shared/local scope: no numerically real LM root; aSR fallback gamma=") == 2


def test_planb_lm_energy_gate_uses_bz_error_for_shared_and_local_error_for_sampler(monkeypatch):
    """Planb LM energy gate uses bz error for shared and local error for sampler."""
    blocks, batch, _O_q, _D_q = _lm_case("mixed", scopes=("world", "sampler", "world"))
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)

    class IdentityComm:
        def allreduce(self, value, op=None):
            del op
            return value

        def allgather(self, value):
            return [value]

        def Get_rank(self):
            return 0

        def Allreduce(self, send, receive, op=None):
            del op
            np.copyto(np.asarray(receive[0]), np.asarray(send[0]))

        def Get_size(self):
            return 1

    world_comm = IdentityComm()
    sampler_comm = IdentityComm()

    def marginal_lm(pencil, *, condition_cutoff):
        del condition_cutoff
        size = pencil.H.shape[0] - 1
        return optimizer_module.LMRealResult(
            q=np.zeros(size),
            predicted_energy=pencil.H_0 + 0.15,
            reference_weight=1.0,
            active_indices=np.arange(size),
        )

    def selected_asr(*coefficients, gamma_max, fallback_gamma):
        del coefficients, gamma_max, fallback_gamma
        return optimizer_module.ASRBoundedResult(
            gamma=0.37,
            predicted_energy=-123.0,
            used_fallback=False,
        )

    monkeypatch.setattr(optimizer_module, "solve_lm_real_pencil", marginal_lm)
    monkeypatch.setattr(optimizer_module, "select_asr_gamma_bounded", selected_asr)
    result = compute_twist_decomposed_real_coordinate_optimization_target(
        batch,
        coordinate_map,
        sampler_comm=sampler_comm,
        world_comm=world_comm,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=False,
        cg_max_iter=500,
        cg_tol=1.0e-12,
        use_lm=True,
        lm_subspace_dim=-1,
        lm_condition_cutoff=1.0e-3,
        world_energy_standard_error=0.1,
        sampler_energy_standard_error=0.01,
        asr_gamma_max=0.8,
        asr_fallback_gamma=0.1,
    )

    shared = np.concatenate(
        [
            np.arange(coordinate_slice.start, coordinate_slice.stop)
            for block, coordinate_slice in zip(blocks, coordinate_map.block_coordinate_slices, strict=True)
            if block.sync_scope == "world"
        ]
    )
    local = np.concatenate(
        [
            np.arange(coordinate_slice.start, coordinate_slice.stop)
            for block, coordinate_slice in zip(blocks, coordinate_map.block_coordinate_slices, strict=True)
            if block.sync_scope == "sampler"
        ]
    )
    np.testing.assert_allclose(
        result.q[shared], np.zeros(2, dtype=np.float64), rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT
    )
    np.testing.assert_allclose(result.q[local], 0.37 * result.sr_q[local], rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    assert result.rejection_reason.count("linear-method energy gate failed") == 1


def test_planb_sr_is_exact_world_sampler_block_diagonal_real_system():
    """Planb SR is exact world sampler block diagonal real system."""
    blocks, batch, O_q, _D_q = _lm_case("mixed", scopes=("world", "sampler", "world"))
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    result = compute_twist_decomposed_real_coordinate_optimization_target(
        batch,
        coordinate_map,
        sampler_comm=MPI.COMM_SELF,
        world_comm=MPI.COMM_SELF,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=False,
        cg_max_iter=500,
        cg_tol=1.0e-12,
        use_lm=False,
        lm_subspace_dim=0,
        lm_condition_cutoff=1.0e-3,
        energy_standard_error=0.1,
    )
    weight_sum = np.sum(batch.weights)
    O_bar = np.einsum("n,nq->q", batch.weights, O_q) / weight_sum
    energy_bar = np.real(np.dot(batch.weights, batch.local_energies) / weight_sum)
    X = (np.conj(O_q - O_bar) * np.sqrt(batch.weights)[:, None] / np.sqrt(weight_sum)).T
    F = -2.0 * np.sqrt(batch.weights) * (batch.local_energies - energy_bar) / np.sqrt(weight_sum)
    expected = np.zeros(coordinate_map.coordinate_size)
    shared = np.concatenate(
        [
            np.arange(sl.start, sl.stop)
            for block, sl in zip(blocks, coordinate_map.block_coordinate_slices)
            if block.sync_scope == "world"
        ]
    )
    local = np.concatenate(
        [
            np.arange(sl.start, sl.stop)
            for block, sl in zip(blocks, coordinate_map.block_coordinate_slices)
            if block.sync_scope == "sampler"
        ]
    )
    for indices in (shared, local):
        diagonal = np.real(np.einsum("qi,qi->q", X[indices], np.conj(X[indices])))
        inverse_sqrt = 1.0 / np.sqrt(diagonal)
        X_pc = X[indices] * inverse_sqrt[:, None]
        b_pc = np.real(X[indices] @ F) * inverse_sqrt
        q_pc = np.linalg.solve(np.real(X_pc @ X_pc.conj().T) + 0.2 * np.eye(indices.size), b_pc)
        expected[indices] = q_pc * inverse_sqrt
    np.testing.assert_allclose(result.q, expected, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    cross = np.real(X[shared] @ X[local].conj().T)
    assert np.linalg.norm(cross) > 0.0


@pytest.mark.parametrize("lm_subspace_dim", [0, 2])
def test_planb_device_local_route_matches_host_for_reduced_lm_and_asr(monkeypatch, lm_subspace_dim):
    """Planb device local route matches host for reduced LM and asr."""
    blocks, batch, _O_q, _D_q = _lm_case("mixed", scopes=("world", "sampler", "world"))
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    common = dict(
        sampler_comm=MPI.COMM_SELF,
        world_comm=MPI.COMM_SELF,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=True,
        cg_max_iter=500,
        cg_tol=1.0e-11,
        use_lm=True,
        lm_subspace_dim=lm_subspace_dim,
        lm_condition_cutoff=1.0e-3,
        energy_standard_error=0.1,
    )
    host = compute_twist_decomposed_real_coordinate_optimization_target(batch, coordinate_map, **common)

    def reject_world_mesh(*args, **kwargs):
        raise AssertionError("twist-decomposed local-device route must not enter a world-mesh kernel")

    for name in (
        "prepare_device_sr_samples",
        "solve_device_wide_real_direct_sharded",
        "solve_device_tall_real_direct_sharded",
        "solve_device_wide_real_cg_chunks_sharded",
        "solve_device_tall_real_cg_chunks_sharded",
        "build_device_lm_pencil",
    ):
        monkeypatch.setattr(optimizer_module._kernel, name, reject_world_mesh)
    device = compute_twist_decomposed_real_coordinate_optimization_target(
        _device_batch(blocks, batch), coordinate_map, **common
    )
    np.testing.assert_allclose(
        device.coordinate_force,
        host.coordinate_force,
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    np.testing.assert_allclose(
        device.coordinate_force_std,
        host.coordinate_force_std,
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    np.testing.assert_allclose(
        device.sr_q,
        host.sr_q,
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    np.testing.assert_allclose(
        device.q,
        host.q,
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    assert "device-local" in device.sr_regime


def test_planb_sampler_streaming_sr_matches_unchanged_dense_device_oracle(monkeypatch):
    """Planb sampler streaming SR matches unchanged dense device oracle."""
    blocks, batch, _O_q, _D_q = _lm_case("mixed", scopes=("world", "sampler", "world"))
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    device_batch = _device_batch(blocks, batch)
    indices = optimizer_module._coordinate_scope_indices(coordinate_map, "sampler")
    common = dict(
        sync_scope="sampler",
        comm=MPI.COMM_SELF,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=True,
        cg_max_iter=500,
        cg_tol=1.0e-11,
        all_real=False,
    )
    dense = optimizer_module._solve_planb_sr_scope_device_dense(
        device_batch,
        coordinate_map,
        indices,
        **common,
    )
    staged_batch = _host_stage_log_sources(blocks, device_batch)
    monkeypatch.setattr(optimizer_module, "OPTIMIZER_DERIVATIVE_SAMPLE_BATCH", 3)
    streamed = optimizer_module._solve_planb_sr_scope_device_streaming(
        staged_batch,
        coordinate_map,
        indices,
        **common,
    )
    # Same bound as the tall Stiefel oracle below: two independent CG runs
    # (cg_tol=1e-11) agree only to the CG stopping tolerance, so the pure-atol
    # 1e-12 bound is machine-marginal.
    np.testing.assert_allclose(streamed.q, dense.q, atol=ATOL_C9_GAMMA_SR_UPDATE, rtol=RTOL_C9_GAMMA_SR_UPDATE)
    np.testing.assert_allclose(
        streamed.metric_diagonal,
        dense.metric_diagonal,
        atol=ATOL_C9_GAMMA_SR_UPDATE,
        rtol=RTOL_C9_GAMMA_SR_UPDATE,
    )
    assert streamed.frozen_coordinate_groups == dense.frozen_coordinate_groups


def test_planb_sampler_streaming_tall_sr_matches_dense_complex_stiefel_oracle(monkeypatch):
    """Planb sampler streaming tall SR matches dense complex stiefel oracle."""
    from .test_optimizer_coordinate_map import _stiefel_block

    rng = np.random.default_rng(952)
    block, _whitened = _stiefel_block("mo", rng, rows=3, columns=4, complex_=True)
    sample_count = 7
    weights = rng.uniform(0.4, 1.6, size=sample_count)
    energies = rng.normal(size=sample_count) + 1.0j * rng.normal(size=sample_count)
    d_p = rng.normal(size=(sample_count, *block.shape)) + 1.0j * rng.normal(size=(sample_count, *block.shape))
    d_pbar = rng.normal(size=(sample_count, *block.shape)) + 1.0j * rng.normal(size=(sample_count, *block.shape))

    def source(primary, secondary):
        return BlockDerivativeSource.build(block=block, d_p=primary, d_pbar=secondary)

    log_source = source(d_p, d_pbar)
    energy_source = source(0.3 * d_p, -0.2 * d_pbar)
    moments = JackknifeBlockMoments.build(
        block=block,
        source_sum=source(weights[:, None, None] * d_p, weights[:, None, None] * d_pbar),
        conjugate_source_energy_sum=source(
            (weights * energies)[:, None, None] * np.conj(d_p),
            (weights * energies)[:, None, None] * np.conj(d_pbar),
        ),
    )
    token = "stiefel-planb-tall:1"
    metadata = JackknifeBinMetadata.build(
        blocks=(block,),
        batch_token=token,
        bin_sample_counts=np.ones(sample_count, dtype=np.int64),
        bin_weight_sums=weights,
        bin_weighted_energy_sums=weights * energies,
        log_source_moments=(moments,),
    )
    batch = OptimizationBatch.build(
        blocks=(block,),
        batch_token=token,
        state_token=TEST_STATE_TOKEN,
        weights=weights,
        local_energies=energies,
        log_derivative_sources=(log_source,),
        local_energy_derivative_sources=(energy_source,),
        jackknife_bin_metadata=metadata,
    )
    coordinate_map = OptimizationCoordinateMap.from_blocks((block,))
    device_batch = _device_batch((block,), batch)
    indices = optimizer_module._coordinate_scope_indices(coordinate_map, "sampler")
    common = dict(
        sync_scope="sampler",
        comm=MPI.COMM_SELF,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=True,
        cg_max_iter=500,
        cg_tol=1.0e-11,
        all_real=False,
    )
    dense = optimizer_module._solve_planb_sr_scope_device_dense(device_batch, coordinate_map, indices, **common)
    staged_batch = _host_stage_log_sources((block,), device_batch)
    monkeypatch.setattr(optimizer_module, "OPTIMIZER_DERIVATIVE_SAMPLE_BATCH", 3)
    streamed = optimizer_module._solve_planb_sr_scope_device_streaming(staged_batch, coordinate_map, indices, **common)
    assert "tall" in dense.solver_regime
    assert "tall" in streamed.solver_regime
    # The streamed and dense paths run CG independently (cg_tol=1e-11), so
    # their agreement is bounded by the CG stopping tolerance, not by fp64
    # accumulation: the pure-atol 1e-12 bound sits below that floor and flips
    # with the host's XLA codegen (observed 1.2e-12 on one platform).
    np.testing.assert_allclose(streamed.q, dense.q, atol=ATOL_C9_GAMMA_SR_UPDATE, rtol=RTOL_C9_GAMMA_SR_UPDATE)
    np.testing.assert_allclose(
        streamed.metric_diagonal,
        dense.metric_diagonal,
        atol=ATOL_C9_GAMMA_SR_UPDATE,
        rtol=RTOL_C9_GAMMA_SR_UPDATE,
    )


def test_planb_reduced_lm_host_staged_contraction_matches_dense():
    """Planb reduced LM host staged contraction matches dense."""
    blocks, batch, _O_q, _D_q = _lm_case("mixed", scopes=("world", "sampler", "world"))
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    device_batch = _device_batch(blocks, batch)
    staged_batch = _host_stage_log_sources(blocks, device_batch)
    right = np.arange(coordinate_map.coordinate_size * 3, dtype=np.float64).reshape(coordinate_map.coordinate_size, 3)
    dense = np.asarray(coordinate_map.transform_derivatives_device(device_batch.log_derivative_sources, right=right))
    streamed = optimizer_module._transform_device_derivatives_reduced_streaming(
        staged_batch.log_derivative_sources,
        coordinate_map,
        right,
        sample_count=int(device_batch.weights.size),
        sample_batch_size=3,
        all_real=False,
    )
    np.testing.assert_allclose(
        np.asarray(streamed),
        dense,
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )


def test_single_k_host_staged_device_sr_matches_dense_oracle(monkeypatch):
    """Single-k host-staged streaming device SR CG matches the dense device direct solve."""
    blocks, batch, _O_q, _D_q = _lm_case("mixed")
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    device_batch = _device_batch(blocks, batch)
    dense_regime, dense = optimizer_module._compute_device_sr_real_target_dense(
        device_batch,
        coordinate_map,
        epsilon=0.2,
        diagonal_floor=0.0,
        all_real=False,
    )
    staged_batch = _host_stage_log_sources(blocks, device_batch)
    monkeypatch.setattr(optimizer_module, "OPTIMIZER_DERIVATIVE_SAMPLE_BATCH", 3)
    streamed_regime, streamed = optimizer_module.compute_device_sr_real_target(
        staged_batch,
        coordinate_map,
        comm=MPI.COMM_SELF,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=True,
        cg_max_iter=500,
        cg_tol=1.0e-11,
        warm_start=None,
    )
    assert streamed_regime == dense_regime
    np.testing.assert_allclose(
        streamed.q,
        dense.q,
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    np.testing.assert_allclose(
        streamed.prepared.metric_diagonal,
        dense.prepared.metric_diagonal,
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )


def test_sharded_chunk_cg_kernels_match_local_oracles_on_one_rank():
    """Sharded chunk cg kernels match local oracles on one rank."""
    rng = np.random.default_rng(953)
    wide_chunks = tuple(rng.normal(size=(5, count)) + 1.0j * rng.normal(size=(5, count)) for count in (3, 2))
    b = rng.normal(size=5)
    wide_local = optimizer_module._kernel.solve_local_device_wide_real_cg_chunks(
        wide_chunks,
        b,
        0.2,
        max_iter=500,
        tol=1.0e-11,
        x0=np.zeros_like(b),
    )
    wide_sharded = optimizer_module._kernel.solve_device_wide_real_cg_chunks_sharded(
        wide_chunks,
        b,
        0.2,
        max_iter=500,
        tol=1.0e-11,
        x0=np.zeros_like(b),
    )
    np.testing.assert_allclose(
        wide_sharded[0],
        wide_local[0],
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )

    tall_chunks = tuple(rng.normal(size=(12, count)) + 1.0j * rng.normal(size=(12, count)) for count in (3, 2))
    force_chunks = tuple(rng.normal(size=count) + 1.0j * rng.normal(size=count) for count in (3, 2))
    tall_local = optimizer_module._kernel.solve_local_device_tall_real_cg_chunks(
        tall_chunks,
        force_chunks,
        0.2,
        max_iter=500,
        tol=1.0e-11,
    )
    tall_sharded = optimizer_module._kernel.solve_device_tall_real_cg_chunks_sharded(
        tall_chunks,
        force_chunks,
        0.2,
        max_iter=500,
        tol=1.0e-11,
    )
    np.testing.assert_allclose(
        tall_sharded[0],
        tall_local[0],
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
