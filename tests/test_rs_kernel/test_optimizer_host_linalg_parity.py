"""Rust == XLA parity of the host (CPU/MPI) optimizer linear algebra (issue #815).

The five BLAS-heavy host functions of the optimizer step dispatch to ``jqmc_rs``
under the production CPU rule (``JQMC_FLAG_RUST_ON_CPU`` / ``flag_rust_on_CPU`` on
the JAX cpu backend). Each test selects the route through that public switch
and compares the Rust result with the XLA (JAX) implementation of the same
operation on identical inputs, and with the NumPy/SciPy route:

* ``solve_wide_sr_real_cg`` / ``solve_wide_sr_real_direct`` against the
  world-mesh ``solve_device_wide_real_cg_chunks_sharded`` /
  ``solve_device_wide_real_direct_sharded``;
* ``solve_tall_sr_real_cg`` / ``solve_tall_sr_real_direct`` against the
  world-mesh ``solve_device_tall_real_cg_chunks_sharded`` /
  ``solve_device_tall_real_direct_sharded`` (a real ``Z`` is its own
  realification);
* ``_build_lm_real_pencil_from_coordinate_arrays`` against ``build_device_lm_pencil``.

``solve_lm_real_pencil`` is shared by the CPU and GPU paths (no dispatch) and
stays the SciPy implementation, so it has no Rust route to test here.

A caller reduction ``reduce_sum`` that returns twice its argument is exactly
the sum over two ranks holding the same local samples, so those cases compare
with XLA on the sample axis duplicated. The real two-rank Allreduce runs in
``tests/test_jqmc_kernel/test_optimizer_host_rust_mpi.py``.

The optimizer arrays are float64 in both precision modes (the fp64
log-derivative and local-energy samples), so the strict Rust/XLA pair of the
fp64 zones that produce them applies in full and mixed runs alike. The
``tiled`` shapes give the Rust kernels several row blocks, column strips and
Gram tiles.

The CG systems have six exactly clustered operator eigenvalues, so exact CG
converges in six iterations and the residual falls by orders of magnitude at
each of them. Every route therefore crosses the stopping tolerance at the same
iteration, which makes the iteration count a well-defined comparison. (On a
generic spectrum, CG run to its last iterations amplifies roundoff in the
recurrence residual: NumPy, XLA and Rust then legitimately stop one iteration
apart.) Truncated runs stop at five iterations, before convergence.

``cg_tol`` is relative (issue #859): every SR CG stops once
``||r|| <= cg_tol * ||b||`` for the right-hand side ``b`` it solves (the dual
``h`` for the tall routes). The relative-rule tests below compare every
production CG route (NumPy, Rust and the XLA local, chunked and
sharded-chunked kernels) with a textbook CG on the explicitly assembled operator,
which stops by the absolute rule at the equivalent threshold
``cg_tol * ||b||``, and with the dense solution of the same system.
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

import multiprocessing
import os
import traceback

import numpy as np
import pytest

from jqmc import jqmc_optimizer as optimizer
from jqmc import jqmc_optimizer_kernel as kernel
from jqmc._setting import get_tolerance_xla_rust_consistency

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

# The SR / LM inputs are built from the fp64 ln Psi derivatives and local
# energies, which stay float64 in both precision modes.
_ZONES = ("wf_eval", "local_energy")
_EPSILON = 1.0e-3
_TALL_EPSILON = 0.2
_TIGHT_CG_TOL = 1.0e-8
_MAX_ITER = 2000
_TRUNCATED_ITER = 5
_KINDS = ["real", "complex", "complex-real-valued"]
_WIDE_SHAPES = {"small": (24, 150), "tiled": (300, 2500)}
_TALL_SHAPES = {"small": (40, 18), "tiled": (2600, 280)}
_PENCIL_SHAPES = {"small": (120, 7), "tiled": (600, 270)}
# Shapes with many fixed row chunks per Rust CG operator pass.
_THREAD_WIDE_SHAPE = (640, 4000)
_THREAD_TALL_SHAPE = (6000, 320)
_THREAD_WIDTHS = (1, 4)


def _tolerance() -> dict[str, float]:
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    return {"atol": atol, "rtol": rtol}


def _route(monkeypatch: pytest.MonkeyPatch, rust: bool) -> None:
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1" if rust else "0")


def _two_identical_ranks(values):
    return 2.0 * np.asarray(values)


_CLUSTERED_SQUARED_SINGULAR_VALUES = np.asarray([0.5, 1.0, 1.7, 2.6, 4.0, 6.5])


def _orthonormal_columns(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    return np.linalg.qr(rng.normal(size=(rows, columns)))[0]


def _clustered_wide_inputs(kind: str, order: str, shape: str, seed: int):
    """Wide CG input with ``Re(X X^dagger)`` having six exact eigenvalue clusters."""
    coordinates, samples = _WIDE_SHAPES[shape]
    rng = np.random.default_rng(seed)
    squared = _CLUSTERED_SQUARED_SINGULAR_VALUES[np.arange(coordinates) % _CLUSTERED_SQUARED_SINGULAR_VALUES.size]
    width = 2 * samples if kind == "complex" else samples
    Y = _orthonormal_columns(rng, coordinates, coordinates) @ (
        np.sqrt(squared)[:, None] * _orthonormal_columns(rng, width, coordinates).T
    )
    if kind == "complex":
        X = Y[:, :samples] + 1j * Y[:, samples:]
    elif kind == "complex-real-valued":
        X = Y + 0j
    else:
        X = Y
    X = np.asarray(X, order=order)
    return X, rng.normal(size=coordinates), 0.1 * rng.normal(size=coordinates)


def _wide_inputs(kind: str, order: str, shape: str, seed: int):
    coordinates, samples = _WIDE_SHAPES[shape]
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(coordinates, samples)) + 0.3 * rng.normal(size=(coordinates, 1))
    if kind == "complex":
        X = X + 1j * rng.normal(size=(coordinates, samples))
    elif kind == "complex-real-valued":
        X = X + 0j
    X = np.asarray(X / np.sqrt(samples), order=order)
    return X, rng.normal(size=coordinates), 0.1 * rng.normal(size=coordinates)


def _absolute_cg_tol(rhs: np.ndarray) -> float:
    """Relative cg_tol (issue #859) whose stopping residual is the absolute _TIGHT_CG_TOL."""
    return _TIGHT_CG_TOL / float(np.linalg.norm(rhs))


def _cg_case(start: str, x0: np.ndarray, rhs: np.ndarray) -> dict:
    initial = np.zeros_like(x0) if start == "cold-converged" else x0
    max_iter = _TRUNCATED_ITER if start == "warm-truncated" else _MAX_ITER
    return {"max_iter": max_iter, "tol": _absolute_cg_tol(rhs), "x0": initial}


def _assert_cg_triplet(actual, reference, start: str) -> None:
    tolerance = _tolerance()
    np.testing.assert_allclose(actual[0], reference[0], **tolerance)
    np.testing.assert_allclose(actual[1], reference[1], **tolerance)
    assert actual[2] == reference[2]
    if start == "warm-truncated":
        assert actual[2] == _TRUNCATED_ITER and actual[1] > _TIGHT_CG_TOL
    else:
        assert actual[1] <= _TIGHT_CG_TOL


@pytest.mark.parametrize("shape", sorted(_WIDE_SHAPES))
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("start", ["cold-converged", "warm-converged", "warm-truncated"])
def test_wide_sr_real_cg_rust_matches_xla(monkeypatch, start, kind, order, shape):
    """Wide SR CG: Rust x, residual and iteration count equal XLA and NumPy (cold/warm start, truncation)."""
    X, b, x0 = _clustered_wide_inputs(kind, order, shape, seed=815)
    case = _cg_case(start, x0, b)
    _route(monkeypatch, rust=True)
    rust = kernel.solve_wide_sr_real_cg(X, b, _EPSILON, **case)
    xla = kernel.solve_device_wide_real_cg_chunks_sharded((X,), b, _EPSILON, **case)
    _route(monkeypatch, rust=False)
    numpy_route = kernel.solve_wide_sr_real_cg(X, b, _EPSILON, **case)
    assert rust[0].dtype == numpy_route[0].dtype and rust[0].shape == b.shape
    _assert_cg_triplet(rust, xla, start)
    _assert_cg_triplet(rust, numpy_route, start)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("start", ["warm-converged", "warm-truncated"])
def test_wide_sr_real_cg_rust_reduction_matches_xla(monkeypatch, start, kind):
    """Wide SR CG with a caller reduction equals XLA on the summed (duplicated) sample axis."""
    X, b, x0 = _clustered_wide_inputs(kind, "C", "small", seed=816)
    case = _cg_case(start, x0, b)
    _route(monkeypatch, rust=True)
    rust = kernel.solve_wide_sr_real_cg(X, b, _EPSILON, reduce_sum=_two_identical_ranks, **case)
    xla = kernel.solve_device_wide_real_cg_chunks_sharded((np.concatenate((X, X), axis=1),), b, _EPSILON, **case)
    _route(monkeypatch, rust=False)
    numpy_route = kernel.solve_wide_sr_real_cg(X, b, _EPSILON, reduce_sum=_two_identical_ranks, **case)
    _assert_cg_triplet(rust, xla, start)
    _assert_cg_triplet(rust, numpy_route, start)


@pytest.mark.parametrize("shape", sorted(_WIDE_SHAPES))
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("reduction", ["none", "two-identical-ranks"])
def test_wide_sr_real_direct_rust_matches_xla(monkeypatch, reduction, kind, order, shape):
    """Wide SR direct solve (LU plus refinement): Rust equals XLA and NumPy."""
    X, b, _x0 = _wide_inputs(kind, order, shape, seed=817)
    reduce_sum = None if reduction == "none" else _two_identical_ranks
    X_xla = X if reduce_sum is None else np.concatenate((X, X), axis=1)
    _route(monkeypatch, rust=True)
    rust = kernel.solve_wide_sr_real_direct(X, b, _EPSILON, reduce_sum=reduce_sum)
    xla = kernel.solve_device_wide_real_direct_sharded(kernel._shard_X(X_xla), b, _EPSILON)
    _route(monkeypatch, rust=False)
    numpy_route = kernel.solve_wide_sr_real_direct(X, b, _EPSILON, reduce_sum=reduce_sum)
    tolerance = _tolerance()
    assert rust.dtype == numpy_route.dtype and rust.shape == b.shape
    np.testing.assert_allclose(rust, xla, **tolerance)
    np.testing.assert_allclose(rust, numpy_route, **tolerance)


def _tall_inputs(order: str, shape: str, seed: int, *, clustered: bool):
    coordinates, duals = _TALL_SHAPES[shape]
    rng = np.random.default_rng(seed)
    if clustered:
        squared = _CLUSTERED_SQUARED_SINGULAR_VALUES[np.arange(duals) % _CLUSTERED_SQUARED_SINGULAR_VALUES.size]
        Z = _orthonormal_columns(rng, coordinates, duals) @ (
            np.sqrt(squared)[:, None] * _orthonormal_columns(rng, duals, duals).T
        )
    else:
        Z = rng.normal(size=(coordinates, duals)) / np.sqrt(duals)
    return np.asarray(Z, order=order), rng.normal(size=duals)


@pytest.mark.parametrize("shape", sorted(_TALL_SHAPES))
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("start", ["cold-converged", "cold-truncated"])
def test_tall_sr_real_cg_rust_matches_xla(monkeypatch, start, order, shape):
    """Tall SR CG from the zero dual: Rust primal update, residual and iterations equal XLA and NumPy."""
    Z, h = _tall_inputs(order, shape, seed=818, clustered=True)
    max_iter = _TRUNCATED_ITER if start == "cold-truncated" else _MAX_ITER
    _route(monkeypatch, rust=True)
    tol = _absolute_cg_tol(h)
    rust = kernel.solve_tall_sr_real_cg(Z, h, _TALL_EPSILON, max_iter=max_iter, tol=tol)
    xla = kernel.solve_device_tall_real_cg_chunks_sharded((Z,), (h,), _TALL_EPSILON, max_iter=max_iter, tol=tol)
    _route(monkeypatch, rust=False)
    numpy_route = kernel.solve_tall_sr_real_cg(Z, h, _TALL_EPSILON, max_iter=max_iter, tol=tol)
    start_kind = "warm-truncated" if start == "cold-truncated" else start
    assert rust[0].shape == (Z.shape[0],)
    _assert_cg_triplet(rust, xla, start_kind)
    _assert_cg_triplet(rust, numpy_route, start_kind)


@pytest.mark.parametrize("shape", sorted(_TALL_SHAPES))
@pytest.mark.parametrize("order", ["C", "F"])
def test_tall_sr_real_direct_rust_matches_xla(monkeypatch, order, shape):
    """Tall SR direct push-through solve (LU plus refinement): Rust equals XLA and NumPy."""
    Z, h = _tall_inputs(order, shape, seed=819, clustered=False)
    _route(monkeypatch, rust=True)
    rust = kernel.solve_tall_sr_real_direct(Z, h, _TALL_EPSILON)
    xla = kernel.solve_device_tall_real_direct_sharded(*kernel._shard_X_F(Z, h), _TALL_EPSILON)
    _route(monkeypatch, rust=False)
    numpy_route = kernel.solve_tall_sr_real_direct(Z, h, _TALL_EPSILON)
    tolerance = _tolerance()
    np.testing.assert_allclose(rust, xla, **tolerance)
    np.testing.assert_allclose(rust, numpy_route, **tolerance)


def _pencil_inputs(fields: str, shape: str, seed: int):
    samples, coordinates = _PENCIL_SHAPES[shape]
    rng = np.random.default_rng(seed)
    O = rng.normal(size=(samples, coordinates)) + 0.5
    D = rng.normal(size=(samples, coordinates)) - 0.2
    energies = rng.normal(size=samples) - 7.0 + 0.3 * O[:, 0]
    if fields in ("complex", "complex-energy"):
        O = O + 1j * rng.normal(size=(samples, coordinates))
        D = D + 1j * rng.normal(size=(samples, coordinates))
    if fields == "complex-energy":
        energies = energies + 0.05j * rng.normal(size=samples)
    weights = rng.uniform(0.2, 1.8, size=samples)
    return O, D, weights, energies


@pytest.mark.parametrize("shape", sorted(_PENCIL_SHAPES))
@pytest.mark.parametrize("fields", ["real", "complex", "complex-energy"])
@pytest.mark.parametrize("reduction", ["none", "two-identical-ranks"])
def test_lm_pencil_core_rust_matches_xla(monkeypatch, reduction, fields, shape):
    """LM pencil core: Rust H, S, H_0 and conditioning metric equal XLA and NumPy."""
    O, D, weights, energies = _pencil_inputs(fields, shape, seed=820)
    reduce_sum = None if reduction == "none" else _two_identical_ranks
    _route(monkeypatch, rust=True)
    rust = optimizer._build_lm_real_pencil_from_coordinate_arrays(O, D, weights, energies, reduce_sum=reduce_sum)
    # Two identical ranks: XLA contracts the duplicated sample rows.
    copies = 1 if reduce_sum is None else 2
    H, S, H_0, metric = kernel.build_device_lm_pencil(
        np.concatenate([O] * copies), np.concatenate([D] * copies), np.tile(weights, copies), np.tile(energies, copies)
    )
    _route(monkeypatch, rust=False)
    numpy_route = optimizer._build_lm_real_pencil_from_coordinate_arrays(O, D, weights, energies, reduce_sum=reduce_sum)
    tolerance = _tolerance()
    for actual, xla, reference in (
        (rust.H, H, numpy_route.H),
        (rust.S, S, numpy_route.S),
        (rust.H_0, H_0, numpy_route.H_0),
        (rust.conditioning_metric, metric, numpy_route.conditioning_metric),
    ):
        np.testing.assert_allclose(actual, xla, **tolerance)
        np.testing.assert_allclose(actual, reference, **tolerance)
    assert rust.H.dtype == numpy_route.H.dtype == np.float64


# ---------------------------------------------------------------------------
# Relative CG stopping rule (issue #859)
# ---------------------------------------------------------------------------

# Right-hand-side scales: the relative rule must give the same iterates (up to
# the scale) and the same iteration count for every scale; an absolute rule
# would stop at once for 1e-6 and run to the roundoff floor for 1e6.
_RHS_SCALES = (1.0e-6, 1.0, 1.0e6)
# Iteration at which the tolerance chosen below makes CG stop (6 = the last,
# exact iteration of the six-cluster spectrum).
_STOP_ITERATIONS = (3, 6)
_RELATIVE_SEED = 859
_RELATIVE_SAMPLES = 150


def _reference_cg_residual_norms(operator: np.ndarray, rhs: np.ndarray, iterations: int) -> np.ndarray:
    """Recurrence residual norms of a textbook CG from zero on the explicit operator."""
    x = np.zeros_like(rhs)
    residual = rhs.copy()
    direction = residual.copy()
    residual_sq = float(residual @ residual)
    norms = [np.sqrt(residual_sq)]
    for _ in range(iterations):
        applied = operator @ direction
        alpha = residual_sq / float(direction @ applied)
        x = x + alpha * direction
        residual = residual - alpha * applied
        next_sq = float(residual @ residual)
        direction = residual + (next_sq / residual_sq) * direction
        residual_sq = next_sq
        norms.append(np.sqrt(residual_sq))
    return np.asarray(norms)


def _relative_tolerance_stopping_at(operator: np.ndarray, rhs: np.ndarray, stop: int) -> float:
    """Relative tolerance whose equivalent absolute threshold is first met at iteration ``stop``.

    The threshold is the geometric mean of the smallest residual before ``stop``
    and the residual at ``stop`` of the reference CG, so every route meets it at
    ``stop`` with a relative margin above 10 %, far above roundoff.
    """
    norms = _reference_cg_residual_norms(operator, rhs, stop)
    before, after = float(np.min(norms[:stop])), float(norms[stop])
    assert after < 0.8 * before, "deterministic input must separate the stopping iteration"
    return np.sqrt(before * after) / float(np.linalg.norm(rhs))


def _relative_rule_system(regime: str, kind: str):
    """A clustered-spectrum wide or tall system with its explicit operators.

    Returns the complex/real sample matrix ``X``; the right-hand side of the
    routes' public inputs (``b`` for wide, ``F`` for tall); the right-hand side
    ``rhs`` of the CG system (``b``; the realified dual ``h = [Re F; -Im F]``);
    the explicit CG operator; the explicit system ``(update_operator,
    update_rhs)`` that the returned update solves (the same system for wide; the
    primal ``(Z Z^T + eps I) q = Z h`` for tall); and the map ``to_update``
    from the CG solution to the update (identity; ``Z``), so that
    ``||update residual|| <= ||to_update||_2 * ||CG residual||``.
    """
    rng = np.random.default_rng(_RELATIVE_SEED)
    samples = _RELATIVE_SAMPLES
    if regime == "wide":
        coordinates = 24
        squared = _CLUSTERED_SQUARED_SINGULAR_VALUES[np.arange(coordinates) % _CLUSTERED_SQUARED_SINGULAR_VALUES.size]
        width = 2 * samples if kind == "complex" else samples
        Y = _orthonormal_columns(rng, coordinates, coordinates) @ (
            np.sqrt(squared)[:, None] * _orthonormal_columns(rng, width, coordinates).T
        )
        X = Y[:, :samples] + 1j * Y[:, samples:] if kind == "complex" else Y
        b = rng.normal(size=coordinates)
        operator = np.real(X @ np.conj(X).T) + _EPSILON * np.eye(coordinates)
        return X, b, b, operator, operator, b, np.eye(coordinates)
    coordinates = 400
    duals = 2 * samples
    squared = _CLUSTERED_SQUARED_SINGULAR_VALUES[np.arange(duals) % _CLUSTERED_SQUARED_SINGULAR_VALUES.size]
    Z = _orthonormal_columns(rng, coordinates, duals) @ (np.sqrt(squared)[:, None] * _orthonormal_columns(rng, duals, duals).T)
    if kind == "real":
        Z[:, samples:] = 0.0
    X = Z[:, :samples] + 1j * Z[:, samples:] if kind == "complex" else Z[:, :samples]
    F = rng.normal(size=samples) + (1j * rng.normal(size=samples) if kind == "complex" else 0.0)
    h = np.concatenate((np.real(F), -np.imag(F)))
    operator = Z.T @ Z + _TALL_EPSILON * np.eye(duals)
    # (Z Z^T + eps) Z y - Z h = Z ((Z^T Z + eps) y - h): the primal residual of
    # q = Z y is Z times the dual CG residual.
    primal_operator = Z @ Z.T + _TALL_EPSILON * np.eye(coordinates)
    return X, F, h, operator, primal_operator, Z @ h, Z


def _wide_cg_routes(monkeypatch, epsilon=_EPSILON):
    """Every production wide SR CG route, called as ``route(X, b, tol=..., max_iter=..., x0=...)``."""

    def host(rust: bool):
        def solve(X, b, **case):
            _route(monkeypatch, rust=rust)
            return kernel.solve_wide_sr_real_cg(X, b, epsilon, **case)

        return solve

    def chunks(X):
        return (X[:, : X.shape[1] // 3], X[:, X.shape[1] // 3 :])

    return {
        "numpy": host(False),
        "rust": host(True),
        "xla-local": lambda X, b, **case: kernel.solve_local_device_wide_real_cg(np.asarray(X), b, epsilon, **case),
        "xla-local-chunks": lambda X, b, **case: kernel.solve_local_device_wide_real_cg_chunks(chunks(X), b, epsilon, **case),
        "xla-sharded-chunks": lambda X, b, **case: kernel.solve_device_wide_real_cg_chunks_sharded(
            chunks(X), b, epsilon, **case
        ),
    }


def _tall_cg_routes(monkeypatch, epsilon=_TALL_EPSILON):
    """Every production tall SR CG route, called as ``route(X, F, tol=..., max_iter=...)``."""

    def realified(X, F):
        return np.concatenate((np.real(X), np.imag(X)), axis=1), np.concatenate((np.real(F), -np.imag(F)))

    def host(rust: bool):
        def solve(X, F, **case):
            _route(monkeypatch, rust=rust)
            Z, h = realified(X, F)
            return kernel.solve_tall_sr_real_cg(Z, h, epsilon, **case)

        return solve

    def chunks(X, F):
        split = X.shape[1] // 3
        return (X[:, :split], X[:, split:]), (F[:split], F[split:])

    return {
        "numpy": host(False),
        "rust": host(True),
        "xla-local": lambda X, F, **case: kernel.solve_local_device_tall_real_cg(np.asarray(X), F, epsilon, **case),
        "xla-local-chunks": lambda X, F, **case: kernel.solve_local_device_tall_real_cg_chunks(*chunks(X, F), epsilon, **case),
        "xla-sharded-chunks": lambda X, F, **case: kernel.solve_device_tall_real_cg_chunks_sharded(
            *chunks(X, F), epsilon, **case
        ),
    }


@pytest.mark.parametrize("stop", _STOP_ITERATIONS)
@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("regime", ["wide", "tall"])
def test_sr_cg_relative_stopping_rule_on_every_route(monkeypatch, regime, kind, stop):
    """Every SR CG route stops by ||r|| <= cg_tol*||b||, like the absolute rule at cg_tol*||b||, for any ||b||; Rust == XLA."""
    X, public_rhs, rhs, operator, update_operator, update_rhs, to_update = _relative_rule_system(regime, kind)
    residual_gain = float(np.linalg.norm(to_update, 2))
    tol = _relative_tolerance_stopping_at(operator, rhs, stop)
    exact_update = np.linalg.solve(update_operator, update_rhs)
    smallest_eigenvalue = float(np.linalg.eigvalsh(update_operator)[0])
    routes = _wide_cg_routes(monkeypatch) if regime == "wide" else _tall_cg_routes(monkeypatch)
    tolerance = _tolerance()
    for scale in _RHS_SCALES:
        threshold = tol * scale * float(np.linalg.norm(rhs))
        case = {"tol": tol, "max_iter": _MAX_ITER}
        if regime == "wide":
            case["x0"] = np.zeros_like(rhs)
        results = {name: route(X, scale * public_rhs, **case) for name, route in routes.items()}
        rust_update, rust_residual, *_ = results["rust"]
        for name, (update, residual, iterations, _breakdown) in results.items():
            label = f"{regime}/{kind}/stop={stop}/scale={scale:g}/{name}"
            # The relative rule stops where the absolute rule at tol * ||b|| stops
            # (the reference CG), for every scale of b.
            assert iterations == stop, label
            assert residual <= threshold, label
            # The returned update solves its explicit system within the residual
            # budget, and its error from the dense solution is bounded by that
            # budget over the smallest eigenvalue (the stopping threshold sits
            # more than 10 % above the stopping residual, far above roundoff).
            update_budget = residual_gain * threshold
            update_residual = float(np.linalg.norm(scale * update_rhs - update_operator @ update))
            assert update_residual <= update_budget, label
            error = float(np.linalg.norm(update - scale * exact_update))
            assert error <= update_budget / smallest_eigenvalue, label
            # Rust == XLA == NumPy on the scale-free iterate and residual.
            np.testing.assert_allclose(update / scale, rust_update / scale, **tolerance, err_msg=label)
            np.testing.assert_allclose(residual / scale, rust_residual / scale, **tolerance, err_msg=label)


@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("regime", ["wide", "tall"])
def test_sr_cg_zero_rhs_returns_exact_zero_on_every_route(monkeypatch, regime, kind):
    """b = 0 under the relative rule: every SR CG route returns the exact solution 0 after zero iterations."""
    X, public_rhs, *_system = _relative_rule_system(regime, kind)
    routes = _wide_cg_routes(monkeypatch) if regime == "wide" else _tall_cg_routes(monkeypatch)
    case = {"tol": 1.0e-5, "max_iter": _MAX_ITER}
    if regime == "wide":
        # A nonzero warm start must not leak into the exact zero solution.
        case["x0"] = np.random.default_rng(_RELATIVE_SEED + 1).normal(size=public_rhs.shape)
    for name, route in routes.items():
        update, residual, iterations, _breakdown = route(X, np.zeros_like(public_rhs), **case)
        assert iterations == 0, name
        assert residual == 0.0, name
        np.testing.assert_array_equal(update, np.zeros_like(update), err_msg=name)


@pytest.mark.parametrize("regime", ["wide", "tall"])
def test_sr_cg_breakdown_is_flagged_and_not_counted_on_every_route(monkeypatch, regime):
    """A zero SR operator breaks CG down at its first step: every route flags it, counts no iteration and keeps x0."""
    X, public_rhs, *_system = _relative_rule_system(regime, "complex")
    X = np.zeros_like(X)
    # epsilon = 0 with X = 0 makes p^T A p exactly zero (a breakdown), not a converged or truncated solve.
    routes = _wide_cg_routes(monkeypatch, epsilon=0.0) if regime == "wide" else _tall_cg_routes(monkeypatch, epsilon=0.0)
    case = {"tol": 1.0e-5, "max_iter": _MAX_ITER}
    if regime == "wide":
        case["x0"] = np.zeros_like(public_rhs)
    for name, route in routes.items():
        update, residual, iterations, breakdown = route(X, public_rhs, **case)
        assert breakdown is True, name
        assert iterations == 0, name
        assert residual > 0.0, name
        np.testing.assert_array_equal(update, np.zeros_like(update), err_msg=name)


# ---------------------------------------------------------------------------
# Rayon pool width
# ---------------------------------------------------------------------------


def _thread_width_cases(regime: str):
    """Every wide (kind x storage order) or tall (storage order) Rust CG input."""
    cases = []
    if regime == "tall":
        coordinates, duals = _THREAD_TALL_SHAPE
        for order in ("C", "F"):
            rng = np.random.default_rng(886)
            Z = np.asarray(rng.normal(size=(coordinates, duals)) / np.sqrt(duals), order=order)
            cases.append((order, Z, rng.normal(size=duals), None))
        return cases
    coordinates, samples = _THREAD_WIDE_SHAPE
    for kind in _KINDS:
        for order in ("C", "F"):
            rng = np.random.default_rng(885)
            X = rng.normal(size=(coordinates, samples)) + 0.3 * rng.normal(size=(coordinates, 1))
            if kind == "complex":
                X = X + 1j * rng.normal(size=(coordinates, samples))
            elif kind == "complex-real-valued":
                X = X + 0j
            X = np.asarray(X / np.sqrt(samples), order=order)
            b, x0 = rng.normal(size=coordinates), 0.1 * rng.normal(size=coordinates)
            cases.append((f"{kind}/{order}", X, b, x0))
    return cases


def _thread_width_worker(connection, regime, width):
    try:
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "1"
        os.environ["JQMC_RS_NUM_THREADS"] = str(width)
        import jqmc_rs

        results = {}
        for label, matrix, rhs, x0 in _thread_width_cases(regime):
            tol = _absolute_cg_tol(rhs)
            if regime == "wide":
                result = kernel.solve_wide_sr_real_cg(matrix, rhs, _EPSILON, max_iter=_MAX_ITER, tol=tol, x0=x0)
            else:
                result = kernel.solve_tall_sr_real_cg(matrix, rhs, _TALL_EPSILON, max_iter=_MAX_ITER, tol=tol)
            results[label] = (np.asarray(result[0]), float(result[1]), int(result[2]))
        connection.send(("ok", jqmc_rs.current_num_threads(), results))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _solve_with_pool_width(regime, width):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_thread_width_worker, args=(child, regime, width))
    process.start()
    child.close()
    try:
        assert parent.poll(300), "Rust SR CG pool-width run timed out"
        message = parent.recv()
        process.join(30)
        assert process.exitcode == 0 and message[0] == "ok", message
        assert message[1] == width, "the requested Rayon pool width was not used"
        return message[2]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(30)
        parent.close()


@pytest.mark.parametrize("regime", ["wide", "tall"])
def test_sr_cg_rust_is_bit_identical_across_pool_widths(regime):
    """Rust wide/tall SR CG: same iterations, residual and bitwise solution on one and four Rayon threads.

    Every wide kind (real, complex, complex with zero imaginary part) in C and
    Fortran storage, or the tall route in C/Fortran storage, runs in spawned processes with
    actual one- and four-thread pools on shapes with many fixed row chunks per
    operator pass. A generic spectrum runs CG long enough that any
    width-dependent summation order would change the iterate.
    """
    reference = _solve_with_pool_width(regime, _THREAD_WIDTHS[0])
    actual = _solve_with_pool_width(regime, _THREAD_WIDTHS[1])
    assert set(actual) == set(reference) and len(reference) == (2 * len(_KINDS) if regime == "wide" else 2)
    for label, (x, residual, iterations) in reference.items():
        assert iterations > _TRUNCATED_ITER, label
        assert actual[label][2] == iterations, label
        assert actual[label][1] == residual, label
        np.testing.assert_array_equal(actual[label][0], x, err_msg=label)


# ---------------------------------------------------------------------------
# Operator passes (issue #885)
# ---------------------------------------------------------------------------


def _prepared_wide_cases(monkeypatch):
    """Wide-only production systems from coordinate-major real and complex X_q.

    ``prepare_wide_sr_real_system`` on the host Rust route must give the same
    preconditioned matrix, element for element, as the NumPy route.
    """
    cases = []
    coordinates, samples = _THREAD_WIDE_SHAPE
    for kind in ("real", "complex"):
        rng = np.random.default_rng(887)
        X_q = rng.normal(size=(coordinates, samples)) / np.sqrt(samples)
        if kind == "complex":
            X_q = X_q + 1j * rng.normal(size=(coordinates, samples)) / np.sqrt(samples)
        F = rng.normal(size=samples)
        prepared = {}
        for rust in (False, True):
            _route(monkeypatch, rust=rust)
            prepared[rust] = optimizer.prepare_wide_sr_real_system(
                X_q, F, coordinate_groups=[[q] for q in range(coordinates)], diagonal_floor=0.0, sample_counts=(samples,)
            )
        np.testing.assert_array_equal(prepared[True].X_preconditioned_local, prepared[False].X_preconditioned_local)
        np.testing.assert_array_equal(prepared[True].b_preconditioned, prepared[False].b_preconditioned)
        system = prepared[True]
        cases.append((f"prepared-{kind}", system.X_preconditioned_local, system.b_preconditioned, None))
    return cases


@pytest.mark.parametrize("regime", ["wide", "tall"])
def test_sr_cg_rust_operator_reads_its_matrix_once_per_application(monkeypatch, regime):
    """Rust wide/tall SR CG: every operator application is one pass over its matrix.

    This pass-count check keeps the two-pass operator (``A (A^T v)`` as two
    matrix-vector products) from silently coming back for a wide-only system;
    it is a deliberate exception to the rule that tests do not check routes.
    It counts behavior, not route names:
    ``jqmc_rs`` counts the CG operator applications and the full passes over
    the operator matrix. A wide-only system prepared by
    ``prepare_wide_sr_real_system`` from a real or complex coordinate-major
    ``X_q`` (the layout the host publishes), and every complex wide matrix and
    real sample-major (Fortran) wide matrix handed to the CG directly, must
    make exactly one pass per application. (The wide preparation writes every
    wide system sample-major; a real coordinate-major matrix reaches the wide
    CG only when handed to it directly, and is applied in two passes.) The
    tall CG makes one pass per application plus the final pass that forms
    the primal update. No timing is involved.
    """
    import jqmc_rs

    cases = _thread_width_cases(regime)
    if regime == "wide":
        cases = [case for case in cases if case[0] != "real/C"] + _prepared_wide_cases(monkeypatch)
    _route(monkeypatch, rust=True)
    for label, matrix, rhs, x0 in cases:
        applications0, sweeps0 = jqmc_rs.optimizer_sr_operator_counts()
        tol = _absolute_cg_tol(rhs)
        if regime == "wide":
            x0 = np.zeros_like(rhs) if x0 is None else x0
            iterations = kernel.solve_wide_sr_real_cg(matrix, rhs, _EPSILON, max_iter=_MAX_ITER, tol=tol, x0=x0)[2]
        else:
            iterations = kernel.solve_tall_sr_real_cg(matrix, rhs, _TALL_EPSILON, max_iter=_MAX_ITER, tol=tol)[2]
        applications1, sweeps1 = jqmc_rs.optimizer_sr_operator_counts()
        applications, sweeps = applications1 - applications0, sweeps1 - sweeps0
        assert iterations > _TRUNCATED_ITER and applications >= iterations, label
        assert sweeps == applications + (1 if regime == "tall" else 0), label


def _bits(values: np.ndarray) -> np.ndarray:
    """The raw float64 bits of a real or complex array, in C order."""
    values = np.ascontiguousarray(values)
    return (values.view(np.float64) if np.iscomplexobj(values) else values).view(np.uint64)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("kind", ["real", "complex"])
def test_wide_sr_preconditioned_matrix_rust_equals_numpy(monkeypatch, kind, order):
    """Wide-only SR preparation: the Rust-written sample-major matrix has the NumPy route's exact bits.

    ``prepare_wide_sr_real_system`` on the host Rust route writes the
    preconditioned matrix of a wide-only system with
    ``jqmc_rs.optimizer_sr_precondition_sample_major`` (a parallel version of
    the NumPy gather and scaling). Against the NumPy route
    (``JQMC_FLAG_RUST_ON_CPU=0``) it must have the same dtype, shape and raw
    bits, be stored sample-major and read-only, with frozen coordinate groups
    (a sub-floor singleton and a sub-floor member of a pair) removed, for a
    real and a complex ``X_q`` in C and Fortran storage on a shape with
    several row blocks.
    """
    coordinates, samples = 120, 1500
    rng = np.random.default_rng(889)
    X_q = rng.normal(size=(coordinates, samples)) / np.sqrt(samples)
    if kind == "complex":
        X_q = X_q + 1j * rng.normal(size=(coordinates, samples)) / np.sqrt(samples)
    X_q[[5, 40]] *= 1.0e-6
    X_q = np.asarray(X_q, order=order)
    F = rng.normal(size=samples)
    groups = [[0, 40]] + [[q] for q in range(1, coordinates) if q != 40]
    prepared = {}
    for rust in (False, True):
        _route(monkeypatch, rust=rust)
        prepared[rust] = optimizer.prepare_wide_sr_real_system(
            X_q, F, coordinate_groups=groups, diagonal_floor=1.0e-6, sample_counts=(samples,)
        )
    rust_matrix, numpy_matrix = prepared[True].X_preconditioned_local, prepared[False].X_preconditioned_local
    assert prepared[True].active_indices.size == coordinates - 3
    np.testing.assert_array_equal(prepared[True].active_indices, prepared[False].active_indices)
    assert rust_matrix.dtype == numpy_matrix.dtype and rust_matrix.shape == numpy_matrix.shape
    assert rust_matrix.flags.f_contiguous and not rust_matrix.flags.writeable
    np.testing.assert_array_equal(_bits(rust_matrix), _bits(numpy_matrix))
    np.testing.assert_array_equal(_bits(prepared[True].b_preconditioned), _bits(prepared[False].b_preconditioned))
