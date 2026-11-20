"""Two-rank Rust == XLA parity of the host optimizer reductions (issue #815).

The Rust host kernels keep MPI in Python: the wide SR CG/direct solves and the
LM pencil core call the caller's ``reduce_sum`` (mpi4py Allreduce) at the
points where the NumPy implementation reduces. Under ``mpirun -np 2`` each
rank holds its own sample block; the Rust result with the real Allreduce must
equal the XLA kernel sharded over the two processes (``shard_map`` + ``psum``)
on the same global samples, and the NumPy route with the same reduction.

These tests opt into the Rust route explicitly (``JQMC_FLAG_RUST_ON_CPU=1``),
whatever the environment of the two-rank run. The tolerance is
the strict Rust/XLA pair of the fp64 zones that produce the optimizer samples;
the arrays are float64 in both precision modes. The CG systems have six exact
operator eigenvalue clusters so every route stops at the same iteration (see
``tests/test_rs_kernel/test_optimizer_host_linalg_parity.py``).

``cg_tol`` is relative (issue #859): ``||r|| <= cg_tol * ||b||`` with the
global norm of the solved right-hand side. The two-rank relative-rule test
checks it where the norm is distributed: the wide Allreduce CG (replicated
``b``) and the tall dual CG sharded over the two processes (``h`` split by rank).
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

from jqmc import jqmc_optimizer as optimizer
from jqmc import jqmc_optimizer_kernel as kernel
from jqmc._setting import get_tolerance_xla_rust_consistency

pytestmark = [
    pytest.mark.heavy,
    pytest.mark.without_chdir,
    pytest.mark.skip_serial(reason="two-rank Rust/XLA SR contract; launch via mpirun -np 2"),
]

_ZONES = ("wf_eval", "local_energy")
_EPSILON = 1.0e-3
_CG_TOL = 1.0e-8
_MAX_ITER = 2000
_TRUNCATED_ITER = 5
_COORDINATES = 30
_LOCAL_SAMPLES = 90
_SQUARED_SINGULAR_VALUES = np.asarray([0.5, 1.0, 1.7, 2.6, 4.0, 6.5])


def _comm() -> MPI.Comm:
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2, "the two-rank contract needs exactly two ranks: launch via mpirun -np 2"
    return comm


def _tolerance() -> dict[str, float]:
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    return {"atol": atol, "rtol": rtol}


def _route(monkeypatch: pytest.MonkeyPatch, rust: bool) -> None:
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1" if rust else "0")


def _orthonormal_columns(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    return np.linalg.qr(rng.normal(size=(rows, columns)))[0]


def _local_wide_block(kind: str, rank: int):
    """This rank's sample block of one global clustered-spectrum wide system."""
    rng = np.random.default_rng(8150)
    samples = 2 * _LOCAL_SAMPLES
    squared = _SQUARED_SINGULAR_VALUES[np.arange(_COORDINATES) % _SQUARED_SINGULAR_VALUES.size]
    width = 2 * samples if kind == "complex" else samples
    Y = _orthonormal_columns(rng, _COORDINATES, _COORDINATES) @ (
        np.sqrt(squared)[:, None] * _orthonormal_columns(rng, width, _COORDINATES).T
    )
    X = Y[:, :samples] + 1j * Y[:, samples:] if kind == "complex" else Y
    columns = slice(rank * _LOCAL_SAMPLES, (rank + 1) * _LOCAL_SAMPLES)
    b = rng.normal(size=_COORDINATES)
    x0 = 0.1 * rng.normal(size=_COORDINATES)
    return np.ascontiguousarray(X[:, columns]), b, x0


@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("start", ["cold-converged", "warm-truncated"])
def test_two_rank_wide_sr_real_cg_rust_matches_sharded_xla(monkeypatch, start, kind):
    """Two-rank wide SR CG: Rust with the Allreduce equals sharded XLA and NumPy (x, residual, iterations)."""
    comm = _comm()
    X_local, b, x0 = _local_wide_block(kind, comm.Get_rank())
    case = {
        "max_iter": _TRUNCATED_ITER if start == "warm-truncated" else _MAX_ITER,
        # Relative cg_tol (issue #859) whose stopping residual is the absolute _CG_TOL.
        "tol": _CG_TOL / float(np.linalg.norm(b)),
        "x0": x0 if start == "warm-truncated" else np.zeros_like(b),
    }

    def reduce_sum(values):
        return optimizer._allreduce_sum_array(comm, values)

    _route(monkeypatch, rust=True)
    rust = kernel.solve_wide_sr_real_cg(X_local, b, _EPSILON, reduce_sum=reduce_sum, **case)
    xla = kernel.solve_device_wide_real_cg_chunks_sharded((X_local,), b, _EPSILON, **case)
    _route(monkeypatch, rust=False)
    numpy_route = kernel.solve_wide_sr_real_cg(X_local, b, _EPSILON, reduce_sum=reduce_sum, **case)
    tolerance = _tolerance()
    for reference in (xla, numpy_route):
        np.testing.assert_allclose(rust[0], reference[0], **tolerance)
        np.testing.assert_allclose(rust[1], reference[1], **tolerance)
        assert rust[2] == reference[2]
    if start == "warm-truncated":
        assert rust[2] == _TRUNCATED_ITER and rust[1] > _CG_TOL
    else:
        assert rust[1] <= _CG_TOL
    gathered = comm.allgather(rust[0])
    np.testing.assert_array_equal(gathered[0], gathered[1])


@pytest.mark.parametrize("kind", ["real", "complex"])
def test_two_rank_wide_sr_real_direct_rust_matches_sharded_xla(monkeypatch, kind):
    """Two-rank wide SR direct solve: Rust with the Allreduce equals sharded XLA and NumPy."""
    comm = _comm()
    rng = np.random.default_rng(8151 + comm.Get_rank())
    X_local = rng.normal(size=(_COORDINATES, _LOCAL_SAMPLES)) / np.sqrt(_LOCAL_SAMPLES)
    if kind == "complex":
        X_local = X_local + 1j * rng.normal(size=X_local.shape) / np.sqrt(_LOCAL_SAMPLES)
    b = np.random.default_rng(8153).normal(size=_COORDINATES)

    def reduce_sum(values):
        return comm.allreduce(np.asarray(values), op=MPI.SUM)

    _route(monkeypatch, rust=True)
    rust = kernel.solve_wide_sr_real_direct(X_local, b, _EPSILON, reduce_sum=reduce_sum)
    xla = kernel.solve_device_wide_real_direct_sharded(kernel._shard_X(X_local), b, _EPSILON)
    _route(monkeypatch, rust=False)
    numpy_route = kernel.solve_wide_sr_real_direct(X_local, b, _EPSILON, reduce_sum=reduce_sum)
    tolerance = _tolerance()
    np.testing.assert_allclose(rust, xla, **tolerance)
    np.testing.assert_allclose(rust, numpy_route, **tolerance)


@pytest.mark.parametrize("fields", ["real", "complex"])
def test_two_rank_lm_pencil_core_rust_matches_sharded_xla(monkeypatch, fields):
    """Two-rank LM pencil core: Rust with mpi4py reductions equals the sharded XLA pencil and NumPy."""
    comm = _comm()
    rng = np.random.default_rng(8154 + comm.Get_rank())
    samples, coordinates = 70, 9
    O = rng.normal(size=(samples, coordinates)) + 0.4 * comm.Get_rank()
    D = rng.normal(size=(samples, coordinates))
    energies = rng.normal(size=samples) - 5.0
    if fields == "complex":
        O = O + 1j * rng.normal(size=O.shape)
        D = D + 1j * rng.normal(size=D.shape)
        energies = energies + 0.05j * rng.normal(size=samples)
    weights = rng.uniform(0.3, 1.7, size=samples)

    def reduce_sum(value):
        return comm.allreduce(value, op=MPI.SUM)

    _route(monkeypatch, rust=True)
    rust = optimizer._build_lm_real_pencil_from_coordinate_arrays(O, D, weights, energies, reduce_sum=reduce_sum)
    H, S, H_0, metric = kernel.build_device_lm_pencil(O, D, weights, energies)
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


@pytest.mark.parametrize("complex_on_rank1", ["e", "D-e", "O-D-e"])
def test_two_rank_lm_pencil_with_real_and_complex_ranks_matches_sharded_xla(monkeypatch, complex_on_rank1):
    """Two-rank LM pencil whose ranks hold real- and complex-typed samples equals the sharded XLA pencil.

    Rank 0 holds real-typed (float64) samples, as a real Gamma-point sampler does; rank 1 holds the named arrays
    complex with a nonzero imaginary part, as a complex twist does in the shared twist-decomposed scope. ``e`` keeps
    O and D complex on both ranks (only the energies differ in type), ``D-e`` is the real-Jastrow twist pattern (O
    real-typed on both ranks) and ``O-D-e`` makes every rank-1 array complex. As every optimizer entry does, the
    arrays first take the one collective real/complex route decision (``collective_real_route``; the complex route
    casts a real-typed array to complex on every rank). The Rust and NumPy pencils with the real mpi4py reductions
    must then equal the XLA pencil sharded over the two processes on the same global samples (rank 0's embedded as
    complex with a zero imaginary part), and be identical on both ranks.
    """
    comm = _comm()
    rank = comm.Get_rank()
    rng = np.random.default_rng(8870 + rank)
    samples, coordinates = 70, 9
    arrays = {
        "O": rng.normal(size=(samples, coordinates)) + 0.4 * rank,
        "D": rng.normal(size=(samples, coordinates)),
        "e": rng.normal(size=samples) - 5.0,
    }
    weights = rng.uniform(0.3, 1.7, size=samples)
    complex_names = complex_on_rank1.split("-")
    if complex_on_rank1 == "e":
        arrays["O"] = arrays["O"] + 1j * rng.normal(size=(samples, coordinates))
        arrays["D"] = arrays["D"] + 1j * rng.normal(size=(samples, coordinates))
    if rank == 1:
        for name in complex_names:
            arrays[name] = arrays[name] + 0.3j * rng.normal(size=arrays[name].shape)
        assert all(np.any(np.imag(arrays[name])) for name in complex_names), "rank-1 imaginary parts must be live"
    else:
        assert not any(np.iscomplexobj(arrays[name]) for name in complex_names), "rank 0 must hold real-typed samples"
    O, D, energies = arrays["O"], arrays["D"], arrays["e"]
    local_all_real = not any(np.iscomplexobj(values) for values in (O, D, energies))
    assert not optimizer.collective_real_route(local_all_real, comm)
    O, D, energies = (optimizer._complex_typed(values) for values in (O, D, energies))

    def reduce_sum(value):
        return comm.allreduce(value, op=MPI.SUM)

    _route(monkeypatch, rust=True)
    rust = optimizer._build_lm_real_pencil_from_coordinate_arrays(O, D, weights, energies, reduce_sum=reduce_sum)
    H, S, H_0, metric = kernel.build_device_lm_pencil(
        O.astype(np.complex128), D.astype(np.complex128), weights, energies.astype(np.complex128)
    )
    _route(monkeypatch, rust=False)
    numpy_route = optimizer._build_lm_real_pencil_from_coordinate_arrays(O, D, weights, energies, reduce_sum=reduce_sum)
    tolerance = _tolerance()
    for label, route in (("rust", rust), ("numpy", numpy_route)):
        for name, actual, xla in (
            ("H", route.H, H),
            ("S", route.S, S),
            ("H_0", route.H_0, H_0),
            ("conditioning_metric", route.conditioning_metric, metric),
        ):
            np.testing.assert_allclose(actual, xla, **tolerance, err_msg=f"{complex_on_rank1}/{label}/{name}")
    gathered = comm.allgather((rust.H, numpy_route.H))
    for index in range(2):
        np.testing.assert_array_equal(gathered[0][index], gathered[1][index])


_RHS_SCALES = (1.0e-6, 1.0, 1.0e6)
_RELATIVE_STOP = 3
_TALL_EPSILON = 0.2


def _reference_cg_residual_norms(operator, rhs, iterations):
    """Recurrence residual norms of a textbook CG from zero on the explicit global operator."""
    residual = rhs.copy()
    direction = residual.copy()
    residual_sq = float(residual @ residual)
    norms = [np.sqrt(residual_sq)]
    for _ in range(iterations):
        applied = operator @ direction
        alpha = residual_sq / float(direction @ applied)
        residual = residual - alpha * applied
        next_sq = float(residual @ residual)
        direction = residual + (next_sq / residual_sq) * direction
        residual_sq = next_sq
        norms.append(np.sqrt(residual_sq))
    return np.asarray(norms)


def _relative_tolerance_stopping_at(operator, rhs, stop):
    """Relative tolerance whose absolute threshold cg_tol*||rhs|| the reference CG first meets at ``stop``."""
    norms = _reference_cg_residual_norms(operator, rhs, stop)
    before, after = float(np.min(norms[:stop])), float(norms[stop])
    assert after < 0.8 * before, "deterministic input must separate the stopping iteration"
    return np.sqrt(before * after) / float(np.linalg.norm(rhs))


def _local_tall_block(kind: str, rank: int):
    """This rank's samples of one global clustered-spectrum tall dual system."""
    rng = np.random.default_rng(8590)
    coordinates, local = 120, 20
    samples = 2 * local
    duals = 2 * samples
    squared = _SQUARED_SINGULAR_VALUES[np.arange(duals) % _SQUARED_SINGULAR_VALUES.size]
    Z = _orthonormal_columns(rng, coordinates, duals) @ (np.sqrt(squared)[:, None] * _orthonormal_columns(rng, duals, duals).T)
    if kind == "real":
        Z[:, samples:] = 0.0
    X = Z[:, :samples] + 1j * Z[:, samples:] if kind == "complex" else Z[:, :samples]
    F = rng.normal(size=samples) + (1j * rng.normal(size=samples) if kind == "complex" else 0.0)
    columns = slice(rank * local, (rank + 1) * local)
    return np.ascontiguousarray(X[:, columns]), np.ascontiguousarray(F[columns])


@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("regime", ["wide", "tall"])
def test_two_rank_sr_cg_relative_rule_uses_the_global_rhs_norm(monkeypatch, regime, kind):
    """Two ranks: every SR CG route stops at ||r|| <= cg_tol * ||b|| with the GLOBAL ||b|| for any scale of b; Rust == XLA."""
    comm = _comm()
    rank = comm.Get_rank()
    tolerance = _tolerance()
    if regime == "wide":
        X_local, b, _x0 = _local_wide_block(kind, rank)
        X_global = np.concatenate(comm.allgather(X_local), axis=1)
        operator = np.real(X_global @ np.conj(X_global).T) + _EPSILON * np.eye(_COORDINATES)
        rhs = b

        def reduce_sum(values):
            return optimizer._allreduce_sum_array(comm, values)

        def host(rust):
            def solve(scale, **case):
                _route(monkeypatch, rust=rust)
                return kernel.solve_wide_sr_real_cg(
                    X_local, scale * b, _EPSILON, reduce_sum=reduce_sum, x0=np.zeros_like(b), **case
                )

            return solve

        half = X_local.shape[1] // 2
        routes = {
            "numpy": host(False),
            "rust": host(True),
            "xla-sharded-chunks": lambda scale, **case: kernel.solve_device_wide_real_cg_chunks_sharded(
                (X_local[:, :half], X_local[:, half:]), scale * b, _EPSILON, x0=np.zeros_like(b), **case
            ),
        }
    else:
        X_local, F_local = _local_tall_block(kind, rank)
        # Sharded dual layout: each rank owns [Re F_r; -Im F_r] against [Re X_r, Im X_r].
        Z_blocks = comm.allgather(np.concatenate((np.real(X_local), np.imag(X_local)), axis=1))
        h_blocks = comm.allgather(np.concatenate((np.real(F_local), -np.imag(F_local))))
        Z_global, rhs = np.concatenate(Z_blocks, axis=1), np.concatenate(h_blocks)
        operator = Z_global.T @ Z_global + _TALL_EPSILON * np.eye(rhs.size)

        def host(rust):
            def solve(scale, **case):
                _route(monkeypatch, rust=rust)
                return kernel.solve_tall_sr_real_cg(Z_global, scale * rhs, _TALL_EPSILON, **case)

            return solve

        half = X_local.shape[1] // 2
        routes = {
            "numpy": host(False),
            "rust": host(True),
            "xla-sharded-chunks": lambda scale, **case: kernel.solve_device_tall_real_cg_chunks_sharded(
                (X_local[:, :half], X_local[:, half:]), (scale * F_local[:half], scale * F_local[half:]), _TALL_EPSILON, **case
            ),
        }
    tol = _relative_tolerance_stopping_at(operator, rhs, _RELATIVE_STOP)
    for scale in _RHS_SCALES:
        threshold = tol * scale * float(np.linalg.norm(rhs))
        results = {name: route(scale, tol=tol, max_iter=_MAX_ITER) for name, route in routes.items()}
        rust_update, rust_residual, *_ = results["rust"]
        for name, (update, residual, iterations, _breakdown) in results.items():
            label = f"{regime}/{kind}/scale={scale:g}/{name}"
            assert iterations == _RELATIVE_STOP, label
            assert residual <= threshold, label
            np.testing.assert_allclose(update / scale, rust_update / scale, **tolerance, err_msg=label)
            np.testing.assert_allclose(residual / scale, rust_residual / scale, **tolerance, err_msg=label)
        gathered = comm.allgather(rust_update)
        np.testing.assert_array_equal(gathered[0], gathered[1])
