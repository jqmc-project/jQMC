"""Real device optimizer route of a real wave function (#886).

A real wave function (real-structure coordinate map, real sources without
``d_pbar``) publishes float64 device coordinate columns, so the device SR
(CG and direct, wide and tall) and LM consumers run in real arithmetic. Each
quantity is compared with the complex route (``all_real=False``), whose
imaginary half is identically zero for these inputs.
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

import dataclasses

import jax
import numpy as np
import pytest

from jqmc import jqmc_optimizer as optimizer_module
from jqmc._setting import (
    ATOL_BIT_EXACT,
    ATOL_C9_GAMMA_SR_UPDATE,
    ATOL_PYTHON_FP64_LOOSE,
    RTOL_C9_GAMMA_SR_UPDATE,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_PYTHON_FP64_ZERO,
    min_S_diag_abs,
)
from jqmc.jqmc_optimizer import (
    DeviceBlockDerivativeSource,
    _collective_transform_route,
    _transform_device_derivatives_reduced_streaming,
    build_reduced_lm_real_pencil_device,
    compute_device_sr_real_target,
    compute_real_coordinate_optimization_target,
    estimate_coordinate_force_jackknife_device,
)

from ._optimizer_transform_cases import case_blocks, coordinate_map, device_sources, jackknife_batches, source_components

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_CASE = "j2-j3-sd-real-source"  # J2 Euclidean + J3 symmetric tie + SD S-Stiefel MO rows: 2755 coordinates
# The SR regime is "wide" when the active coordinates are fewer than the real
# dual length: the samples on the real route, twice the samples on the
# complex route. 64 samples give the tall route on both, 2800 (three
# 1024-sample streaming chunks) the wide route on both.
_SAMPLES = {"tall": 64, "wide": 2800}
# Same well-conditioned shift as the two-rank tall CG oracle, so a CG stopping
# residual of ATOL_PYTHON_FP64_LOOSE bounds the CG error at the C9 tolerances.
_EPSILON = 0.2
_NEGLIGIBLE_LM_CONDITION_CUTOFF = 1.0e-12


def _real_device_batch(regime: str):
    blocks = case_blocks(_CASE)
    samples = _SAMPLES[regime]
    _host, device_batch = jackknife_batches(
        blocks, bins=samples, complex_source=False, complex_energy=False, seed=8860 + samples
    )
    energy_components = source_components(blocks, (samples,), complex_source=False, seed=8870 + samples)
    energy_sources = tuple(
        DeviceBlockDerivativeSource.build_host(block=block, **parts)
        for block, parts in zip(blocks, energy_components, strict=True)
    )
    device_batch = dataclasses.replace(device_batch, local_energy_derivative_sources=energy_sources)
    return blocks, coordinate_map(blocks), device_batch


def _assert_real_float64(value) -> None:
    assert np.dtype(value.dtype) == np.dtype(np.float64), value.dtype


def test_real_wavefunction_batch_selects_real_route_and_complex_batch_does_not():
    """The entry predicate is real for a real wave function and complex for complex sources."""
    _blocks, cmap, batch = _real_device_batch("tall")
    routed, all_real = _collective_transform_route(batch, cmap, None)
    assert all_real and routed is batch
    # The entry decision and the publication's own (rank-local) route give the same columns.
    np.testing.assert_allclose(
        np.asarray(cmap.transform_derivatives_device(batch.log_derivative_sources, all_real=all_real)),
        np.asarray(cmap.transform_derivatives_device(batch.log_derivative_sources)),
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_BIT_EXACT,
    )
    blocks = case_blocks(_CASE)
    _host, complex_batch = jackknife_batches(blocks, bins=8, complex_source=True, complex_energy=True, seed=8861)
    routed, all_real = _collective_transform_route(complex_batch, cmap, None)
    assert not all_real and routed is complex_batch
    with pytest.raises(ValueError, match="all_real=True requires a real publication"):
        cmap.transform_derivatives_device(complex_batch.log_derivative_sources, all_real=True)


def test_real_device_transform_is_float64_and_equals_complex_route_real_part():
    """Real columns equal the complex route's real part; the complex imaginary half is exactly zero."""
    blocks, cmap, batch = _real_device_batch("tall")
    sources = device_sources(blocks, source_components(blocks, (5, 3), complex_source=False, seed=8862))
    real = cmap.transform_derivatives_device(sources, all_real=True)
    np.testing.assert_array_equal(np.asarray(cmap.transform_derivatives_device(sources)), np.asarray(real))
    complex_ = np.asarray(cmap.transform_derivatives_device(sources, all_real=False))
    _assert_real_float64(real)
    assert real.shape == complex_.shape == (5, 3, cmap.coordinate_size)
    np.testing.assert_allclose(np.imag(complex_), 0.0, rtol=RTOL_PYTHON_FP64_ZERO, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(np.asarray(real), np.real(complex_), rtol=RTOL_PYTHON_FP64_ZERO, atol=ATOL_PYTHON_FP64_LOOSE)

    right = np.random.default_rng(8863).standard_normal((cmap.coordinate_size, 4))
    reduced_real = _transform_device_derivatives_reduced_streaming(
        batch.log_derivative_sources, cmap, right, sample_count=int(batch.weights.size), all_real=True
    )
    reduced_complex = _transform_device_derivatives_reduced_streaming(
        batch.log_derivative_sources, cmap, right, sample_count=int(batch.weights.size), all_real=False
    )
    _assert_real_float64(reduced_real)
    np.testing.assert_allclose(
        np.asarray(reduced_real), np.real(np.asarray(reduced_complex)), rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_PYTHON_FP64_LOOSE
    )
    host_reduced = cmap.transform_host_derivatives_reduced(batch.local_energy_derivative_sources, right, all_real=True)
    _assert_real_float64(host_reduced)


def test_real_device_jackknife_matches_complex_route():
    """The device coordinate-force jackknife agrees between the real and complex routes."""
    _blocks, cmap, batch = _real_device_batch("tall")
    force, std = estimate_coordinate_force_jackknife_device(batch, cmap, all_real=True)
    reference_force, reference_std = estimate_coordinate_force_jackknife_device(batch, cmap, all_real=False)
    np.testing.assert_allclose(force, reference_force, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_PYTHON_FP64_LOOSE)
    np.testing.assert_allclose(std, reference_std, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_PYTHON_FP64_LOOSE)


@pytest.mark.parametrize("regime", ["tall", "wide"])
def test_real_device_sr_direct_and_cg_match_complex_route(monkeypatch, regime):
    """Device SR direct (dense) and CG (streaming) on the real route match the complex route.

    Direct solves agree at the C9 SR-update tolerances. CG runs to a stopping
    residual of ATOL_PYTHON_FP64_LOOSE (cg_tol is relative to the right-hand
    side), which bounds its error at the same tolerances.
    """
    _blocks, cmap, batch = _real_device_batch(regime)
    published = []
    original = optimizer_module.OptimizationCoordinateMap.transform_derivatives_device

    def spy(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        published.append(np.dtype(value.dtype))
        return value

    monkeypatch.setattr(optimizer_module.OptimizationCoordinateMap, "transform_derivatives_device", spy)
    common = dict(epsilon=_EPSILON, diagonal_floor=min_S_diag_abs, cg_max_iter=100000, warm_start=None)
    reference_regime, reference = compute_device_sr_real_target(batch, cmap, cg=False, cg_tol=1.0e-5, all_real=False, **common)
    assert reference_regime == regime
    published.clear()
    direct_regime, direct = compute_device_sr_real_target(batch, cmap, cg=False, cg_tol=1.0e-5, all_real=True, **common)
    assert direct_regime == regime
    assert published and all(dtype == np.float64 for dtype in published)
    np.testing.assert_allclose(direct.q, reference.q, rtol=RTOL_C9_GAMMA_SR_UPDATE, atol=ATOL_C9_GAMMA_SR_UPDATE)
    np.testing.assert_allclose(
        direct.prepared.metric_diagonal, reference.prepared.metric_diagonal, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT
    )
    if regime == "tall":
        assert int(direct.dual_layout.valid_counts[0]) == _SAMPLES["tall"]
        assert int(reference.dual_layout.valid_counts[0]) == 2 * _SAMPLES["tall"]

    cg_tol = ATOL_PYTHON_FP64_LOOSE / reference.rhs_norm
    published.clear()
    cg_regime, cg = compute_device_sr_real_target(batch, cmap, cg=True, cg_tol=cg_tol, all_real=True, **common)
    assert cg_regime == regime
    assert published and all(dtype == np.float64 for dtype in published)
    assert cg.residual_norm <= ATOL_PYTHON_FP64_LOOSE
    np.testing.assert_allclose(cg.q, reference.q, rtol=RTOL_C9_GAMMA_SR_UPDATE, atol=ATOL_C9_GAMMA_SR_UPDATE)
    _complex_regime, complex_cg = compute_device_sr_real_target(batch, cmap, cg=True, cg_tol=cg_tol, all_real=False, **common)
    assert complex_cg.residual_norm <= ATOL_PYTHON_FP64_LOOSE
    np.testing.assert_allclose(cg.q, complex_cg.q, rtol=RTOL_C9_GAMMA_SR_UPDATE, atol=ATOL_C9_GAMMA_SR_UPDATE)


def test_real_device_reduced_lm_pencil_matches_complex_route():
    """The reduced device LM pencil is built from float64 O/D and matches the complex route."""
    _blocks, cmap, batch = _real_device_batch("tall")
    rng = np.random.default_rng(8864)
    direction = rng.standard_normal(cmap.coordinate_size)
    selected = np.sort(rng.choice(cmap.coordinate_size, size=6, replace=False))
    pencil, basis = build_reduced_lm_real_pencil_device(batch, cmap, direction, selected, all_real=True)
    reference, reference_basis = build_reduced_lm_real_pencil_device(batch, cmap, direction, selected, all_real=False)
    np.testing.assert_array_equal(basis, reference_basis)
    for name in ("H", "S", "conditioning_metric"):
        np.testing.assert_allclose(
            getattr(pencil, name), getattr(reference, name), rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_PYTHON_FP64_LOOSE
        )
    np.testing.assert_allclose(pencil.H_0, reference.H_0, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_PYTHON_FP64_LOOSE)


@pytest.mark.parametrize("lm_subspace_dim", [0, 4, -1])
def test_real_wavefunction_optimizer_target_takes_real_device_route(monkeypatch, lm_subspace_dim):
    """The production entry decides the real route once; its step matches the forced complex route.

    Covers aSR (subspace 0), reduced LM and full LM with the dense direct SR.
    """
    _blocks, cmap, batch = _real_device_batch("tall")
    decisions = []
    original = optimizer_module.OptimizationCoordinateMap.transform_derivatives_device

    def spy(self, *args, all_real=None, **kwargs):
        decisions.append(all_real)
        return original(self, *args, all_real=all_real, **kwargs)

    monkeypatch.setattr(optimizer_module.OptimizationCoordinateMap, "transform_derivatives_device", spy)
    kwargs = dict(
        epsilon=_EPSILON,
        diagonal_floor=min_S_diag_abs,
        cg=False,
        cg_max_iter=100000,
        cg_tol=1.0e-5,
        warm_start=None,
        use_lm=True,
        lm_subspace_dim=lm_subspace_dim,
        lm_condition_cutoff=_NEGLIGIBLE_LM_CONDITION_CUTOFF,
        energy_standard_error=1.0e-2,
    )
    target = compute_real_coordinate_optimization_target(batch, cmap, **kwargs)
    # The entry passes its one decision explicitly to every device transform.
    assert decisions and all(decision is True for decision in decisions)
    decisions.clear()
    original_route = optimizer_module._collective_transform_route
    monkeypatch.setattr(optimizer_module, "_collective_transform_route", lambda *args: (original_route(*args)[0], False))
    reference = compute_real_coordinate_optimization_target(batch, cmap, **kwargs)
    assert decisions and all(decision is False for decision in decisions)
    assert target.accepted_scale == pytest.approx(reference.accepted_scale, rel=RTOL_PYTHON_FP64_LOOSE, abs=ATOL_BIT_EXACT)
    assert (target.rejection_reason is None) == (reference.rejection_reason is None)
    np.testing.assert_allclose(target.sr_q, reference.sr_q, rtol=RTOL_C9_GAMMA_SR_UPDATE, atol=ATOL_C9_GAMMA_SR_UPDATE)
    np.testing.assert_allclose(target.q, reference.q, rtol=RTOL_C9_GAMMA_SR_UPDATE, atol=ATOL_C9_GAMMA_SR_UPDATE)


def test_real_tall_chunk_cg_kernel_omits_zero_imaginary_block():
    """Real tall chunks solve the S-row dual; complex chunks with a zero imaginary part give the same q."""
    from jqmc.jqmc_optimizer_kernel import solve_local_device_tall_real_cg_chunks, solve_local_device_tall_real_direct

    rng = np.random.default_rng(8865)
    chunks = tuple(rng.standard_normal((12, count)) for count in (3, 2))
    forces = tuple(rng.standard_normal(count) for count in (3, 2))
    h_norm = float(np.linalg.norm(np.concatenate(forces)))
    tol = ATOL_PYTHON_FP64_LOOSE / h_norm
    q_real, residual_real, *_ = solve_local_device_tall_real_cg_chunks(chunks, forces, _EPSILON, max_iter=500, tol=tol)
    q_complex, residual_complex, *_ = solve_local_device_tall_real_cg_chunks(
        tuple(chunk.astype(np.complex128) for chunk in chunks), forces, _EPSILON, max_iter=500, tol=tol
    )
    assert residual_real <= ATOL_PYTHON_FP64_LOOSE and residual_complex <= ATOL_PYTHON_FP64_LOOSE
    X = jax.numpy.asarray(np.concatenate(chunks, axis=1))
    F = jax.numpy.asarray(np.concatenate(forces))
    q_direct = solve_local_device_tall_real_direct(X, F, _EPSILON)
    np.testing.assert_allclose(q_real, q_direct, rtol=RTOL_C9_GAMMA_SR_UPDATE, atol=ATOL_PYTHON_FP64_LOOSE)
    np.testing.assert_allclose(q_real, q_complex, rtol=RTOL_C9_GAMMA_SR_UPDATE, atol=ATOL_PYTHON_FP64_LOOSE)


# solver -> (inner solver it hands the dual RHS to, kernel getters whose compiled cache is cleared,
# chunked inputs, dual split across sample shards)
_TALL_DUAL_ROUTES = {
    "solve_local_device_tall_real_cg": ("_cg_while_loop", ("_get_local_device_tall_real_cg_kernel",), False, False),
    "solve_local_device_tall_real_cg_chunks": (
        "_cg_while_loop",
        ("_get_local_device_tall_real_cg_chunks_kernel",),
        True,
        False,
    ),
    "solve_device_tall_real_cg_chunks_sharded": ("_cg_while_loop_sharded", (), True, True),
    "solve_local_device_tall_real_direct": (
        "_device_refined_direct_solve",
        ("_get_local_device_tall_real_direct_kernel",),
        False,
        False,
    ),
    # The sharded direct solve all-gathers the dual, so it solves the whole one.
    "solve_device_tall_real_direct_sharded": (
        "_device_refined_direct_solve",
        ("_get_device_tall_real_direct_sharded_kernel",),
        False,
        False,
    ),
}


@pytest.mark.parametrize("route", list(_TALL_DUAL_ROUTES))
def test_real_tall_routes_solve_the_sample_row_dual(monkeypatch, route):
    """Resource witness: a real tall system hands an S-row dual to its solver, a complex one 2S.

    A trace-time spy records the length of the dual RHS that each production
    tall route passes to its CG loop or direct solve. Restoring the zero
    imaginary dual block for real samples doubles that length (and quadruples
    a direct operator), which the numerical comparisons cannot see.
    """
    from jqmc import jqmc_optimizer_kernel as kernel_module

    solver_name = route
    inner_name, cached_getters, chunked, per_shard = _TALL_DUAL_ROUTES[route]
    rng = np.random.default_rng(8866)
    counts = (6, 4)
    chunks = tuple(rng.standard_normal((23, count)) for count in counts)
    forces = tuple(rng.standard_normal(count) for count in counts)
    sample_count = sum(counts)
    device_count = int(kernel_module._get_sr_mesh().devices.size)
    if sample_count % device_count or any(count % device_count for count in counts):
        pytest.skip(f"{device_count} devices do not divide the witness sample chunks")
    # Sharded CG loops see their own shard's samples; every other route solves the whole dual.
    local_samples = sample_count // device_count if per_shard else sample_count

    recorded = []
    inner = getattr(kernel_module, inner_name)

    def spy(operator_or_rhs, rhs_or_operator, *args, **kwargs):
        rhs = rhs_or_operator if inner_name == "_device_refined_direct_solve" else operator_or_rhs
        recorded.append(int(rhs.shape[0]))
        return inner(operator_or_rhs, rhs_or_operator, *args, **kwargs)

    monkeypatch.setattr(kernel_module, inner_name, spy)
    solver = getattr(kernel_module, solver_name)

    def dual_length(dtype) -> int:
        # Clear the compiled kernels so the solve is traced (and the spy runs) again.
        for getter in cached_getters:
            monkeypatch.delattr(getattr(kernel_module, getter), "_cached", raising=False)
        monkeypatch.setattr(kernel_module, "_DEVICE_TALL_REAL_CG_CHUNKS_KERNEL_CACHE", {})
        recorded.clear()
        typed_chunks = tuple(jax.numpy.asarray(chunk.astype(dtype)) for chunk in chunks)
        typed_forces = tuple(jax.numpy.asarray(force) for force in forces)
        cg = dict(max_iter=50, tol=1.0e-5) if "_cg" in solver_name else {}
        if chunked:
            solver(typed_chunks, typed_forces, _EPSILON, **cg)
        else:
            solver(jax.numpy.concatenate(typed_chunks, axis=1), jax.numpy.concatenate(typed_forces), _EPSILON, **cg)
        assert recorded, f"{solver_name} did not reach {inner_name}"
        assert len(set(recorded)) == 1
        return recorded[0]

    assert dual_length(np.float64) == local_samples
    assert dual_length(np.complex128) == 2 * local_samples
