"""Numerical kernels for wave-function optimization.

This private implementation module owns SR, LM, and aSR numerical kernels plus
their JAX device/sharding machinery. It does not know any sampler class or own
the optimization control flow.
"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os
from collections.abc import Callable, Sequence
from functools import partial
from logging import getLogger
from typing import Any, TypeAlias

import jax
import jax.scipy.linalg
import numpy as np
import numpy.typing as npt
import scipy.linalg
from jax import numpy as jnp
from jax import typing as jnpt

from . import _setting
from ._setting import (
    OPTIMIZER_LM_REAL_ROOT_IMAG_RTOL,
    EPS_zero_division,
    get_optimizer_breakdown_threshold,
    get_optimizer_roundoff_tolerance,
)


logger = getLogger("jqmc").getChild(__name__)

# ``(x, residual_norm, num_iterations, breakdown)`` of one SR CG solve.
HostSolveResult: TypeAlias = tuple[npt.NDArray, float, int, bool]
PreparedSamples: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
ReduceSum: TypeAlias = Callable[[npt.NDArray], jnpt.ArrayLike]
DeviceSolveResult: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
DevicePrepareKernel: TypeAlias = Callable[[jax.Array, jax.Array, jax.Array], PreparedSamples]
LocalDevicePrepareKernel: TypeAlias = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    PreparedSamples,
]
DevicePreconditionKernel: TypeAlias = Callable[[jax.Array, jax.Array], jax.Array]
DeviceWideCGKernel: TypeAlias = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    DeviceSolveResult,
]
DeviceTallCGKernel: TypeAlias = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    DeviceSolveResult,
]
DeviceDirectKernel: TypeAlias = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
DeviceWideChunkCGKernel: TypeAlias = Callable[
    [tuple[jax.Array, ...], jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    DeviceSolveResult,
]
DeviceTallChunkCGKernel: TypeAlias = Callable[
    [tuple[jax.Array, ...], tuple[jax.Array, ...], jax.Array, jax.Array, jax.Array],
    DeviceSolveResult,
]
DeviceLMPencilKernel: TypeAlias = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array],
]
ArrayOperator: TypeAlias = Callable[[jax.Array], jax.Array]
DeviceCGState: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
# Direct SR solves factor an explicitly formed Gram operator (``Re(X X^dagger)``
# in coordinate space, ``Z^T Z`` in sample space) plus ``epsilon I``.  Forming
# that Gram matrix rounds every entry by O(u), and the solve amplifies it by
# cond ~ 1/epsilon, so the plain factor-and-solve answer carries a forward
# error of ~cond * u * |theta| (about 1e-12 at epsilon = 1e-3).  Every direct
# solve therefore applies this many fixed-precision iterative-refinement
# corrections, reusing the LU factor, with the residual evaluated through the
# factored operator ``Y (Y^T v) + epsilon v`` (never the squared matrix).
_SR_DIRECT_REFINEMENT_STEPS = 2

_RUST_FLAG_FALSE_VALUES = ("0", "false", "no", "off", "")


def _host_rust_kernel(symbol: str) -> Callable[..., Any] | None:
    """Return ``jqmc_rs.<symbol>`` when the production CPU dispatch selects Rust.

    The rule is the one the production drivers resolve at their run head
    (``jqmc_mcmc.MCMC.run``): the ``JQMC_FLAG_RUST_ON_CPU`` environment
    override if set, else ``_setting.flag_rust_on_CPU``, and the JAX ``cpu``
    backend. ``None`` keeps the NumPy/SciPy implementation unchanged. When Rust
    is selected but the extension or the kernel is unavailable this raises; it
    never falls back silently.
    """
    env_rust = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
    if env_rust is None:
        rust_requested = _setting.flag_rust_on_CPU
    else:
        rust_requested = env_rust.strip().lower() not in _RUST_FLAG_FALSE_VALUES
    if not rust_requested or jax.default_backend() != "cpu":
        return None
    try:
        import jqmc_rs
    except ImportError as exc:
        raise RuntimeError(
            "flag_rust_on_CPU is enabled on the CPU path but the native extension 'jqmc_rs' "
            "could not be imported. Build it with:\n"
            "    pip install -e .   (from the repository root; needs a Rust toolchain)\n"
            "or set JQMC_FLAG_RUST_ON_CPU=0 to force the NumPy/SciPy optimizer path."
        ) from exc
    kernel = getattr(jqmc_rs, symbol, None)
    if kernel is None:
        raise RuntimeError(
            f"flag_rust_on_CPU is enabled on the CPU path but the installed 'jqmc_rs' has no '{symbol}' "
            "host optimizer kernel. Rebuild it with `pip install -e .` from the repository root, "
            "or set JQMC_FLAG_RUST_ON_CPU=0 to force the NumPy/SciPy optimizer path."
        )
    # rayon's global pool is built once per process; honor JQMC_RS_NUM_THREADS
    # exactly like the drivers (idempotent after the first call).
    jqmc_rs.configure_threads(None)
    return kernel


def _conjugate_if_complex(values: npt.NDArray) -> npt.NDArray:
    """Return ``conj(values)``; a real array is returned as is, without a copy.

    The host optimizer keeps a real wave function's sample-by-coordinate arrays
    real (float64), where conjugation is the identity.
    """
    return np.conj(values) if np.iscomplexobj(values) else values


def _rust_row_major(matrix: npt.NDArray) -> tuple[npt.NDArray, bool, bool]:
    """Present a real/complex matrix to the native kernels without a copy if possible.

    Returns ``(stored, transposed, complex_input)``: ``stored`` is a C-contiguous
    float64 array holding ``matrix`` (``transposed=False``) or ``matrix.T``
    (``transposed=True``, for a Fortran-ordered input); complex input is viewed
    as interleaved float64 pairs along the last axis.
    """
    complex_input = bool(np.iscomplexobj(matrix))
    target = np.complex128 if complex_input else np.float64
    if matrix.dtype != target:
        matrix = matrix.astype(target)
    if matrix.flags.c_contiguous:
        stored, transposed = matrix, False
    elif matrix.flags.f_contiguous:
        stored, transposed = matrix.T, True
    else:
        stored, transposed = np.ascontiguousarray(matrix), False
    if complex_input:
        stored = stored.view(np.float64)
    return stored, transposed, complex_input


def _sample_major_preconditioned_rows(X_q: npt.NDArray, active_indices: npt.NDArray, scale: npt.NDArray) -> npt.NDArray:
    """Return ``X_q[active_indices] * scale[:, None]`` stored sample-major (a Fortran-ordered array).

    The host Rust wide SR CG streams a sample-major matrix in one pass per
    operator application. ``jqmc_rs.optimizer_sr_precondition_sample_major``
    parallelises the single-threaded NumPy pass
    ``np.take(X_q.T, active_indices, axis=1) * scale`` on the Rayon pool with
    the same products, so both give the same bits; NumPy runs it for dtypes
    other than float64/complex128 or when the host Rust route is off. The
    output is allocated with NumPy so the Rust writer fills an untouched
    buffer, avoiding a serial zero-fill of the whole buffer on the calling
    thread.
    """
    native = _host_rust_kernel("optimizer_sr_precondition_sample_major")
    if native is None or X_q.dtype not in (np.float64, np.complex128) or scale.dtype != np.float64:
        selected = np.take(X_q.T, active_indices, axis=1)
        if np.result_type(selected, scale) == selected.dtype:
            return np.multiply(selected, scale, out=selected).T
        return np.multiply(selected, scale).T
    stored, transposed, complex_input = _rust_row_major(X_q)
    values = np.empty((X_q.shape[1], (2 if complex_input else 1) * active_indices.size), dtype=np.float64)
    native(stored, transposed, complex_input, np.ascontiguousarray(active_indices, dtype=np.int64), scale, values)
    return (values.view(np.complex128) if complex_input else values).T


def _rust_vector_reduction(reduce_sum: ReduceSum) -> Callable[[npt.NDArray], npt.NDArray]:
    """Wrap a caller reduction so the native kernels receive float64 arrays."""

    def reduce(values: npt.NDArray) -> npt.NDArray:
        return np.ascontiguousarray(np.asarray(reduce_sum(values)), dtype=np.float64)

    return reduce


def _refine_host_direct_solve(
    factor: tuple[npt.NDArray, npt.NDArray],
    rhs: npt.NDArray,
    solution: npt.NDArray,
    apply_operator: Callable[[npt.NDArray], npt.NDArray],
) -> npt.NDArray:
    """Iteratively refine a host LU solve against the factored operator."""
    for _ in range(_SR_DIRECT_REFINEMENT_STEPS):
        solution = solution + scipy.linalg.lu_solve(factor, rhs - apply_operator(solution), check_finite=False)
    return solution


def _device_refined_direct_solve(
    operator: jax.Array,
    rhs: jax.Array,
    apply_operator: ArrayOperator,
) -> jax.Array:
    """Traced LU solve plus refinement against the factored operator."""
    # A real Gram operator may meet a complex right-hand side (real observable
    # matrix at a twist); LAPACK-style triangular solves need one dtype.
    operator = operator.astype(jnp.result_type(operator, rhs))
    factor = jax.scipy.linalg.lu_factor(operator)
    solution = jax.scipy.linalg.lu_solve(factor, rhs)
    for _ in range(_SR_DIRECT_REFINEMENT_STEPS):
        solution = solution + jax.scipy.linalg.lu_solve(factor, rhs - apply_operator(solution))
    return solution


def validate_lm_condition_cutoff(condition_cutoff: object) -> float:
    """Return ``lm_cond`` as a float after checking its mathematical domain.

    ``lm_cond`` is the smallest eigenvalue admitted in the correlation matrix
    ``C_A`` of the active LM coordinates.  It must lie in the open interval
    ``(0, 1)``: a non-positive cutoff admits a singular (or indefinite) metric
    and therefore an ill-posed pencil, while a correlation matrix has a unit
    diagonal, so ``lambda_min(C_A) <= 1`` for every set, with equality only for
    exactly uncorrelated coordinates.  A cutoff at or above one would therefore
    remove every coordinate and turn each LM step into the zero update.
    """
    if isinstance(condition_cutoff, (bool, np.bool_)):
        raise ValueError(f"lm_cond must be a real number in the open interval (0, 1); got {condition_cutoff!r}.")
    try:
        value = float(condition_cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"lm_cond must be a real number in the open interval (0, 1); got {condition_cutoff!r}.") from exc
    if not np.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError(
            f"lm_cond must be a finite number in the open interval (0, 1); got {condition_cutoff!r}. It is the smallest "
            "eigenvalue kept in the correlation matrix of the LM coordinates: lm_cond <= 0 admits a singular metric, "
            "and lm_cond >= 1 removes every coordinate (a correlation matrix has eigenvalues <= 1)."
        )
    return value


def _lm_correlation_block(metric: npt.NDArray, indices: npt.NDArray) -> npt.NDArray:
    """Return the symmetric correlation matrix of ``metric`` on ``indices`` (unit diagonal)."""
    sub = metric[np.ix_(indices, indices)]
    inv_sqrt = 1.0 / np.sqrt(np.diag(sub))
    correlation = inv_sqrt[:, None] * sub * inv_sqrt[None, :]
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def select_lm_conditioned_coordinates(
    conditioning_metric: npt.NDArray,
    candidates: npt.NDArray,
    condition_cutoff: float,
) -> tuple[npt.NDArray, float]:
    """Select a well-conditioned LM coordinate set by one pivoted Cholesky pass.

    The correlation matrix ``C`` of ``candidates`` is shifted to ``C - c I``
    (``c`` = ``condition_cutoff``) and factorized by LAPACK ``?pstrf``
    (complete diagonal pivoting: the largest residual diagonal is chosen at
    each step; among exact ties the first position in LAPACK's current pivot
    order, so the result is deterministic).  The factorization stops at
    the first residual pivot at or below the round-off threshold
    ``lm_correlation_pivot`` (``n * eps``).  The kept pivots give a positive
    definite ``C_A - c I``, i.e. ``lambda_min(C_A) > c``; every rejected
    coordinate ``j`` has a residual (Schur complement) ``<= n * eps``, and since
    ``lambda_min`` of a bordered positive definite matrix is bounded by its
    Schur complement, ``lambda_min(C_{A+j}) <= c + n * eps``: the set is maximal
    under single additions.  Cost: one ``O(n^2 r)`` factorization (``r`` =
    kept coordinates) instead of one eigendecomposition per removed coordinate.

    Returns the sorted active indices (into the full coordinate axis) and
    ``lambda_min(C_A)`` from one ``eigvalsh`` of the active block (``nan`` if empty).
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.size == 0:
        return candidates, float("nan")
    shifted = _lm_correlation_block(conditioning_metric, candidates)
    shifted[np.diag_indices_from(shifted)] -= condition_cutoff
    threshold = get_optimizer_roundoff_tolerance("lm_correlation_pivot", shifted.dtype, scale=candidates.size)
    (pstrf,) = scipy.linalg.lapack.get_lapack_funcs(("pstrf",), (shifted,))
    _factor, pivots, rank, info = pstrf(shifted, tol=threshold, lower=1, overwrite_a=1)
    if info < 0:
        raise ValueError(f"LM correlation pivoted Cholesky received an invalid argument (LAPACK info={info}).")
    active = np.sort(candidates[np.asarray(pivots[:rank], dtype=np.int64) - 1])
    if active.size == 0:
        return active, float("nan")
    lambda_min = scipy.linalg.eigvalsh(
        _lm_correlation_block(conditioning_metric, active), subset_by_index=[0, 0], check_finite=False
    )
    return active, float(lambda_min[0])


def _solve_lm_cholesky_reduced_pencil(H: npt.NDArray, S: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Return the eigenvalues and right eigenvectors of the pencil ``H x = lambda S x``.

    ``S = L L^T`` (Cholesky); the standard non-symmetric problem
    ``(L^-1 H L^-T) y = lambda y`` is solved by LAPACK ``?geev`` and
    back-transformed, ``x = L^-T y``.
    """
    try:
        factor = scipy.linalg.cholesky(S, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise FloatingPointError(
            "LM overlap of the conditioned active coordinates is not positive definite (Cholesky failed); the "
            "sampled pencil cannot be solved. Increase lm_cond or the number of samples."
        ) from exc
    left = scipy.linalg.solve_triangular(factor, H, lower=True, check_finite=False)
    reduced = scipy.linalg.solve_triangular(factor, left.T, lower=True, check_finite=False).T
    eigenvalues, reduced_vectors = scipy.linalg.eig(reduced, overwrite_a=True, check_finite=False)

    def back_transform(vectors: npt.NDArray) -> npt.NDArray:
        return scipy.linalg.solve_triangular(factor, vectors, lower=True, trans="T", check_finite=False)

    if np.iscomplexobj(reduced_vectors):
        eigenvectors = back_transform(np.real(reduced_vectors)) + 1j * back_transform(np.imag(reduced_vectors))
    else:
        eigenvectors = back_transform(reduced_vectors)
    return eigenvalues, eigenvectors


class LMNoRealRootError(ArithmeticError):
    """The conditioned non-symmetric LM pencil has no numerically real eigenvalue.

    A complex eigenvector is defined only up to a phase, so no real update can
    be decoded from it; the optimizer rejects such an LM step.
    """


def solve_lm_real_pencil(
    H: npt.ArrayLike,
    S: npt.ArrayLike,
    H_0: float,
    *,
    condition_cutoff: float,
    conditioning_metric: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray, float, float, npt.NDArray]:
    """Stabilize and solve one centered real generalized eigenproblem.

    Conditioning: coordinates with a negligible ``conditioning_metric``
    diagonal are dropped, then :func:`select_lm_conditioned_coordinates`
    keeps a set whose correlation matrix has ``lambda_min >= condition_cutoff``.
    Solve: the augmented active pencil is Cholesky-reduced and solved by
    ``?geev`` (:func:`_solve_lm_cholesky_reduced_pencil`); among the numerically
    real roots the S-normalized eigenvector with the largest reference weight
    is selected.

    Raises:
        LMNoRealRootError: no root of the conditioned pencil is numerically
            real; no update is decoded.
    """
    H = np.asarray(H)
    S = np.asarray(S)
    if H.shape != S.shape or H.ndim != 2 or H.shape[0] != H.shape[1] or np.iscomplexobj(H) or np.iscomplexobj(S):
        raise ValueError("LM pencil matrices must be matching real squares.")
    condition_cutoff = validate_lm_condition_cutoff(condition_cutoff)
    coordinate_size = H.shape[0] - 1
    reference_scale = max(float(np.max(np.abs(S), initial=0.0)), 1.0)
    centered_tolerance = get_optimizer_roundoff_tolerance("lm_centered_overlap", S.dtype, scale=reference_scale)
    if (
        not np.isfinite(S[0, 0])
        or S[0, 0] <= 0
        or np.max(np.abs(S[0, 1:]), initial=0.0) > centered_tolerance
        or np.max(np.abs(S[1:, 0]), initial=0.0) > centered_tolerance
    ):
        raise ValueError("LM requires a centered overlap with a reference-orthogonal tangent block.")
    if conditioning_metric is None:
        conditioning_metric = S[1:, 1:]
    conditioning_metric = np.asarray(conditioning_metric)
    if (
        conditioning_metric.shape != (coordinate_size, coordinate_size)
        or np.iscomplexobj(conditioning_metric)
        or not np.all(np.isfinite(conditioning_metric))
    ):
        raise ValueError("LM conditioning metric must be a finite real coordinate square.")
    conditioning_metric = 0.5 * (conditioning_metric + conditioning_metric.T)
    diagonal = np.diag(conditioning_metric)
    logger.debug(
        "  LM conditioning input: coordinates=%d, cutoff=%.6e, diag[min=%.6e, max=%.6e].",
        coordinate_size,
        condition_cutoff,
        float(np.min(diagonal, initial=np.inf)),
        float(np.max(diagonal, initial=-np.inf)),
    )
    scale = max(float(np.max(np.abs(diagonal), initial=0.0)), 1.0)
    diagonal_tolerance = get_optimizer_roundoff_tolerance("lm_active_diagonal", S.dtype, scale=scale)
    diagonal_survivors = np.flatnonzero(np.isfinite(diagonal) & (diagonal > diagonal_tolerance))
    active, final_min_correlation_eigenvalue = select_lm_conditioned_coordinates(
        conditioning_metric, diagonal_survivors, condition_cutoff
    )
    logger.info(
        "  LM conditioning: active=%d/%d, tiny-diagonal removed=%d, correlation removed=%d, "
        "min correlation eigenvalue of the active set=%.6e (cutoff=%.3e).",
        active.size,
        coordinate_size,
        coordinate_size - diagonal_survivors.size,
        diagonal_survivors.size - active.size,
        final_min_correlation_eigenvalue,
        condition_cutoff,
    )
    if active.size == 0:
        logger.debug("  LM solve: no active tangent coordinate; returning the zero update.")
        return np.zeros(coordinate_size, dtype=S.dtype), float(H_0), 0.0, active
    augmented = np.concatenate((np.asarray([0], dtype=np.int64), active + 1))
    H_augmented = H[np.ix_(augmented, augmented)]
    S_augmented = S[np.ix_(augmented, augmented)]
    # NON-SYMMETRIC linear method: H is built element-by-element without
    # Hermitian averaging, so the pencil is a GENERAL real generalized
    # eigenproblem (right eigenvectors).  Under exact sampling H is symmetric
    # in expectation; on an imperfect sample the asymmetry (the finite
    # <de_L/dc>) is the retained information.  The conditioned S block is
    # positive definite, so the pencil is reduced by its Cholesky factor.
    eigenvalues_c, eigenvectors_c = _solve_lm_cholesky_reduced_pencil(H_augmented, S_augmented)
    # Physical roots: (numerically) real eigenvalues. Complex-conjugate pairs
    # can appear for a strongly non-normal sample-level pencil; they carry no
    # usable real update and are excluded from selection. If ALL roots are
    # complex, no update is decoded and the caller rejects the LM step.
    imag_scale = np.maximum(np.abs(eigenvalues_c), 1.0)
    is_real_root = np.abs(np.imag(eigenvalues_c)) <= OPTIMIZER_LM_REAL_ROOT_IMAG_RTOL * imag_scale
    if not np.any(is_real_root):
        logger.warning("  LM: no numerically real eigenvalue in the non-symmetric pencil; no LM update is decoded.")
        raise LMNoRealRootError("LM pencil has no numerically real eigenvalue; no real update can be decoded.")
    eigenvalues = np.real(eigenvalues_c)
    # Normalize the back-transformed right eigenvectors in the S metric so the
    # reference weight keeps its meaning (v[0]**2 for an S-normalized vector).
    eigenvectors = np.real_if_close(eigenvectors_c, tol=1.0e6)
    if np.iscomplexobj(eigenvectors):
        eigenvectors = np.real(eigenvectors)
    norms = np.sqrt(np.maximum(np.einsum("ij,ij->j", eigenvectors, S_augmented @ eigenvectors), 1.0e-300))
    eigenvectors = eigenvectors / norms[np.newaxis, :]
    reference_weights = np.where(is_real_root, np.square(eigenvectors[0]), -np.inf)
    best = sorted(
        range(eigenvalues.size),
        key=lambda index: (-reference_weights[index], eigenvalues[index], index),
    )[0]
    vector = eigenvectors[:, best]
    if not np.isfinite(vector[0]) or vector[0] == 0:
        raise FloatingPointError("LM selected eigenvector has an unusable reference coefficient.")
    q = np.zeros(coordinate_size, dtype=np.result_type(H.dtype, S.dtype))
    q[active] = vector[1:] / vector[0]
    if not np.all(np.isfinite(q)):
        raise FloatingPointError("LM decoded a non-finite real-coordinate update.")
    logger.debug(
        "  LM selected state: reference weight=%.8f, ||q||=%.6e, max|q|=%.6e.",
        float(reference_weights[best]),
        float(np.linalg.norm(q)),
        float(np.max(np.abs(q), initial=0.0)),
    )
    logger.log(5, "  LM active coordinate indices = %s", active)
    logger.log(5, "  LM decoded q = %s", q)
    return q, float(eigenvalues[best]), float(reference_weights[best]), active


def _select_asr_reference_branch(
    H_0: float,
    H_1: float,
    H_2: float,
    S_2: float,
    *,
    fallback_gamma: float,
    gamma_max: float | None,
) -> tuple[float, bool]:
    """Select the stationary root continuously connected to the reference."""
    B = H_2 - H_0 * S_2
    discriminant = B**2 + 4.0 * H_1**2 * S_2
    if discriminant < 0.0:
        logger.warning("aSR: discriminant is negative (%.3e); setting it to zero.", discriminant)
        discriminant = 0.0
    denom = -2.0 * H_1 * S_2
    if abs(denom) < EPS_zero_division:
        logger.warning(
            "aSR: denom = -2*H_1*S_2 ~ 0 (H_1=%.3e, S_2=%.3e); no meaningful optimization direction. Returning gamma = 0.",
            H_1,
            S_2,
        )
        logger.info("aSR: final gamma = 0.000000.")
        return 0.0, True

    sqrt_discriminant = np.sqrt(discriminant)
    gamma_plus = (-B + sqrt_discriminant) / denom
    gamma_minus = (-B - sqrt_discriminant) / denom
    logger.info("aSR: gamma+ = %.6f, gamma- = %.6f.", gamma_plus, gamma_minus)

    # The two roots are the two projective eigenstates on the SR line.  The
    # smaller-|gamma| root has the larger overlap with the current/reference
    # state.  Selecting the lower model energy instead can choose the nearly
    # orthogonal tangent state and generate an arbitrarily large update.
    if abs(gamma_plus) <= abs(gamma_minus):
        selected_label = "gamma+"
        selected_gamma = float(gamma_plus)
    else:
        selected_label = "gamma-"
        selected_gamma = float(gamma_minus)
    logger.info(
        "aSR: selected %s = %.6f (smaller-|gamma|, reference-connected root).",
        selected_label,
        selected_gamma,
    )

    used_fallback = False
    # NEGATIVE roots are legitimate: with the
    # non-symmetric H_1 = C_g + D_g/2 the model can detect that the SR
    # (covariance) direction points UPHILL on the measurable landscape --
    # a periodic slab is the reference case (fine FD confirmed
    # E(p0 - dp) < E(p0) < E(p0 + dp) while the covariance force claimed
    # descent along +dp). Stepping backward along g is then the model-optimal
    # move; only the MAGNITUDE is capped by ``gamma_max``, and the safety
    # fallback inherits the model's SIGN (a forced +fallback would re-create
    # the uphill walk the non-symmetric pencil just diagnosed).
    if not np.isfinite(selected_gamma) or selected_gamma == 0.0:
        logger.warning(
            "aSR: selected root is not finite/nonzero (gamma=%.6f); falling back to plain SR (gamma=%.6f).",
            selected_gamma,
            fallback_gamma,
        )
        selected_gamma = fallback_gamma
        used_fallback = True
    elif gamma_max is not None and abs(selected_gamma) > gamma_max:
        capped = float(np.copysign(fallback_gamma, selected_gamma))
        logger.warning(
            "aSR: selected reference-connected root gamma=%.6f exceeds |gamma_max|=%.6f; "
            "falling back to plain SR with the model's sign (gamma=%.6f).",
            selected_gamma,
            gamma_max,
            capped,
        )
        selected_gamma = capped
        used_fallback = True
    elif selected_gamma > 0.0:
        logger.info("aSR: selected root is in the steepest-descent direction.")
    else:
        logger.info(
            "aSR: selected root is NEGATIVE (anti-SR direction): the non-symmetric "
            "model detects the covariance direction to be uphill; stepping backward."
        )

    logger.info("aSR: final gamma = %.6f.", selected_gamma)
    return selected_gamma, used_fallback


def select_asr_gamma_bounded(
    H_0: float,
    H_1: float,
    H_2: float,
    S_2: float,
    *,
    gamma_max: float,
    fallback_gamma: float,
) -> tuple[float, bool]:
    """Select the reference-connected centered-aSR root with safety fallback."""
    coefficients = np.asarray([H_0, H_1, H_2, S_2, gamma_max, fallback_gamma], dtype=float)
    if not np.all(np.isfinite(coefficients)) or gamma_max <= 0:
        raise ValueError("aSR coefficients and bounds must be finite with gamma_max positive.")
    scale = max(float(np.max(np.abs(coefficients[:4]))), 1.0)
    tolerance = get_optimizer_roundoff_tolerance("asr_model", coefficients.dtype, scale=scale)
    fallback = float(np.clip(fallback_gamma, -gamma_max, gamma_max))

    def energy(gamma: float) -> float | None:
        denominator = 1.0 + gamma * gamma * S_2
        if not np.isfinite(denominator) or denominator <= tolerance:
            return None
        value = (H_0 + 2.0 * gamma * H_1 + gamma * gamma * H_2) / denominator
        return float(value) if np.isfinite(value) else None

    if S_2 < -tolerance:
        logger.warning(
            "aSR: centered metric contraction S_2=%.6e is negative; falling back to plain SR (gamma=%.6f).",
            S_2,
            fallback,
        )
        gamma = fallback
        used_fallback = True
        logger.info("aSR: final gamma = %.6f.", gamma)
    else:
        gamma, used_fallback = _select_asr_reference_branch(
            H_0,
            H_1,
            H_2,
            S_2,
            fallback_gamma=fallback,
            gamma_max=gamma_max,
        )

    predicted_energy = energy(gamma)
    if predicted_energy is None:
        gamma = 0.0
        predicted_energy = float(H_0)
        used_fallback = True
        logger.warning("aSR: selected gamma produced an invalid model denominator; using gamma = 0.")
        logger.info("aSR: final gamma = 0.000000.")
    return gamma, predicted_energy, used_fallback


def compute_asr_gamma(H_0: float, H_1: float, H_2: float, S_2: float) -> float:
    r"""Solve for the optimal gamma in the accelerated SR energy minimization.

    Finds :math:`\gamma` satisfying

    .. math::

        \frac{\partial E_{\alpha+\gamma g}}{\partial \gamma} = 0,

    where :math:`E(\gamma) = (H_0 + 2\gamma H_1 + \gamma^2 H_2) / (1 + \gamma^2 S_2)`.

    Setting the derivative to zero yields the quadratic

    .. math::

        -H_1 S_2 \gamma^2 + (H_2 - H_0 S_2)\gamma + H_1 = 0,

    whose solution is

    .. math::

        \gamma = \frac{H_0 S_2 - H_2 \pm
                 \sqrt{(H_2 - H_0 S_2)^2 + 4 H_1^2 S_2}}
                {-2 H_1 S_2}.

    The root with the smaller absolute value is returned.  This avoids
    spuriously large steps when the quadratic coefficient
    :math:`a = -H_1 S_2` is small,
    which causes one root to diverge while the other stays near zero.

    Args:
        H_0 (float): Current energy :math:`E_{\alpha}`.
        H_1 (float): :math:`H_1 = -\tfrac{1}{2} g^T f`.
        H_2 (float): :math:`H_2 = g^T (B+K) g`.
        S_2 (float): :math:`S_2 = g^T S g = \langle w (g^T \delta O)^2 \rangle_w`.

    Returns:
        float: Optimal :math:`\gamma`.
    """
    gamma, _used_fallback = _select_asr_reference_branch(
        H_0,
        H_1,
        H_2,
        S_2,
        fallback_gamma=0.1,
        gamma_max=None,
    )
    return gamma


def solve_linear_method(
    H_0: float,
    f_vec: npt.NDArray,
    S_matrix: npt.NDArray,
    K_matrix: npt.NDArray,
    B_matrix: npt.NDArray,
    epsilon: float,
) -> tuple[npt.NDArray, float, float]:
    r"""Solve the Linear Method generalized eigenvalue problem.

    Constructs extended matrices :math:`\bar H` and :math:`\bar S` of
    dimension :math:`(p'+1) \times (p'+1)` (after S eigenvalue cutoff)
    and solves :math:`\bar H v = E \bar S v`.

    The eigenvector with the largest :math:`|v_0|^2` is selected, and the
    parameter update is :math:`c_k = v_k / v_0`.  :math:`|v_0|^2` is also
    returned so the caller can guard against updates that fall outside
    the linear regime (small overlap with the current wavefunction).

    Args:
        H_0: Current energy :math:`E_\alpha`.
        f_vec: Generalized force vector, shape ``(p,)``.
        S_matrix: SR matrix, shape ``(p, p)``.
        K_matrix: K matrix, shape ``(p, p)``.
        B_matrix: B matrix, shape ``(p, p)``.
        epsilon: Eigenvalue cutoff for S matrix.

    Returns:
        tuple: ``(c_vec, E_lm, v0_sq_best)`` where ``c_vec`` has shape
        ``(p,)`` (in the original parameter space), ``E_lm`` is the
        selected eigenvalue, and ``v0_sq_best`` is the :math:`|v_0|^2`
        of the selected eigenvector (in [0, 1]).
    """
    p = len(f_vec)

    # ==================================================================
    # Preconditioning (dgelscut algorithm, following TurboRVB)
    #
    # 1. Remove parameters with negligible diag(S)
    # 2. Normalize S to a correlation matrix (diag = 1)
    # 3. Iteratively remove parameters until the minimum eigenvalue
    #    of the correlation matrix >= epsilon (= epsdgel)
    # This guarantees condition number <= 1/epsilon.
    # ==================================================================

    # ---- Step 1: Remove parameters with near-zero diag(S) ----
    # ``dtype_mcmc_np`` follows the input matrix dtype so the c_vec /
    # early-return zero updates come back in the right dtype (complex128
    # at non-Gamma; float64 at Gamma -- bit-identical to the real Gamma
    # path). ``parcut2`` is on the REAL machine epsilon
    # (the threshold lives on ``|diag_S|`` which is real-positive even
    # for complex Hermitian S).
    #
    # Promote over ALL inputs (S / K / B / f), not just ``S``: at a
    # complex twist with a real-``O_k`` block ``S`` is real while ``K``
    # (weighted by the complex ``e_L``) is complex, so the extended
    # ``H_bar = K + B`` and the back-transformed ``c_vec`` are complex
    # even though ``S`` is not -- deriving the dtype from ``S`` alone
    # would silently drop that Im part. ``result_type`` of all-real
    # inputs is real, so the Gamma path is bit-identical.
    dtype_mcmc_np = np.result_type(
        np.asarray(S_matrix).dtype,
        np.asarray(K_matrix).dtype,
        np.asarray(B_matrix).dtype,
        np.asarray(f_vec).dtype,
    )
    _real_eps_dtype = np.float64 if np.issubdtype(dtype_mcmc_np, np.complexfloating) else dtype_mcmc_np
    diag_S = np.diag(S_matrix)
    max_diag_S = np.max(np.abs(diag_S))
    # parcut2 ~ machine_precision^2, effectively only removes exact zeros
    parcut2 = get_optimizer_roundoff_tolerance("lm_active_diagonal", _real_eps_dtype)
    alive = np.abs(diag_S) > parcut2 * max_diag_S
    n_removed_step1 = p - int(np.count_nonzero(alive))
    if n_removed_step1 > 0:
        logger.info("  LM dgelscut: Step 1 removed %d/%d parameters (tiny diag(S)).", n_removed_step1, p)

    if not np.any(alive):
        logger.warning("  LM dgelscut: all parameters removed in Step 1; returning zero update.")
        return np.zeros(p, dtype=dtype_mcmc_np), H_0, 0.0

    # ---- Step 2: Build correlation matrix for alive parameters ----
    # ``D_inv_sqrt`` is real (built from ``|diag_S|`` which is
    # real-positive even for complex Hermitian S);
    # keep it real-typed so the correlation-matrix Hermitian
    # ``C = D D conj(S) D D`` stays well-defined under ``eigh``.
    alive_idx = np.where(alive)[0]
    D_inv_sqrt = np.zeros(p, dtype=_real_eps_dtype)
    D_inv_sqrt[alive_idx] = 1.0 / np.sqrt(np.abs(diag_S[alive_idx]).astype(_real_eps_dtype))

    # ---- Step 3: Iteratively remove parameters until well-conditioned ----
    while True:
        idx = np.where(alive)[0]
        n_alive = len(idx)
        if n_alive == 0:
            logger.warning("  LM dgelscut: all parameters removed; returning zero update.")
            return np.zeros(p, dtype=dtype_mcmc_np), H_0, 0.0

        # Build correlation matrix for current alive set
        D_sub = D_inv_sqrt[idx]  # (n_alive,)
        S_sub = S_matrix[np.ix_(idx, idx)]  # (n_alive, n_alive)
        C = D_sub[:, np.newaxis] * S_sub * D_sub[np.newaxis, :]  # correlation matrix
        np.fill_diagonal(C, 1.0)  # enforce exact 1 on diagonal

        # Eigenvalue decomposition of correlation matrix
        eigvals_C, eigvecs_C = np.linalg.eigh(C)
        min_eigval = eigvals_C[0]  # smallest eigenvalue

        if min_eigval >= epsilon:
            break  # well-conditioned

        # Find the parameter contributing most to the smallest eigenvector
        # (the one with largest |component| in the problematic eigenvector)
        worst_local = int(np.argmax(np.abs(eigvecs_C[:, 0])))
        worst_global = idx[worst_local]
        alive[worst_global] = False
        logger.debug(
            "  LM dgelscut: removing param %d (min eigval=%.3e < eps=%.3e), %d remaining.",
            worst_global,
            min_eigval,
            epsilon,
            int(np.count_nonzero(alive)),
        )

    n_final = int(np.count_nonzero(alive))
    logger.info(
        "  LM dgelscut: %d/%d parameters kept (correlation matrix cond <= %.0f).",
        n_final,
        p,
        1.0 / epsilon,
    )

    # ==================================================================
    # Build LM matrices for the surviving parameters
    # ==================================================================
    idx = np.where(alive)[0]

    # Hermitian-symmetrize H = K + B (B is not Hermitian due to finite-sample
    # noise). ``T -> conj().T`` -- bit-identical on the real path
    # (``conj`` of float64 is identity), and the
    # Hermitian average ``0.5*(H + H^H)`` is the standard Hermitian
    # projector for ``eigh`` consumption.
    H_matrix = K_matrix + B_matrix
    H_matrix = 0.5 * (H_matrix + H_matrix.conj().T)

    # Extract sub-matrices for alive parameters
    S_alive = S_matrix[np.ix_(idx, idx)]
    H_alive = H_matrix[np.ix_(idx, idx)]
    f_alive = f_vec[idx]

    # ---- S-orthonormalization: S = U Lambda U^T, P = U Lambda^{-1/2} ----
    eigvals_S, eigvecs_S = np.linalg.eigh(S_alive)
    # After dgelscut, all eigenvalues should be positive, but clip for safety
    pos_mask = eigvals_S > 0
    U = eigvecs_S[:, pos_mask]
    Lambda = eigvals_S[pos_mask]
    p_prime = len(Lambda)

    if p_prime == 0:
        logger.warning("  LM: no positive S eigenvalues after dgelscut; returning zero update.")
        return np.zeros(p, dtype=dtype_mcmc_np), H_0, 0.0

    # P = U Lambda^{-1/2} (S-orthonormal basis)
    inv_sqrt_Lambda = 1.0 / np.sqrt(Lambda)
    P = U * inv_sqrt_Lambda[np.newaxis, :]  # (n_alive, p')

    # Transform H and f to S-orthonormal basis.
    # ``P.T -> P.conj().T`` -- the S-orthonormal basis change requires
    # the Hermitian conjugate when ``P`` is complex; bit-identical on
    # the real path.
    H_new = P.conj().T @ H_alive @ P  # (p', p') -- should be near-identity S
    f_new = P.conj().T @ f_alive  # (p',)

    # ---- Build extended matrices (p'+1) x (p'+1) ----
    dim = p_prime + 1
    H_bar = np.zeros((dim, dim), dtype=dtype_mcmc_np)
    # S_bar is the identity (S-orthonormal basis).

    H_bar[0, 0] = H_0
    # Orientation of the complex force block in the Linear Method: the KET
    # column is the un-conjugated force,
    #   H_bar[k, 0] = <psi_k|(H - E)|psi_0> = <w conj(dO_k) de> = -f_k / 2,
    # and the bra row is its Hermitian conjugate. The swapped assignment
    # (f in the row, conj(f) in the column) solves the eigenproblem with
    # f -> conj(f) while H_new is NOT conjugated -- silently wrong update
    # direction for complex f. Bit-identical on the real path.
    H_bar[1:, 0] = -0.5 * f_new
    H_bar[0, 1:] = -0.5 * np.conj(f_new)
    H_bar[1:, 1:] = H_new

    # ---- Standard eigenvalue problem (S_bar = I) ----
    eigvals_lm, eigvecs_lm = np.linalg.eigh(H_bar)

    # ---- Select eigenvector with max |v_0|^2 ----
    # |v_0|^2 measures the overlap with the current wavefunction; large
    # overlap means the LM step stays in the linear regime.  The caller
    # is expected to reject the update (e.g. fall back to plain SR) when
    # ``v0_sq_best`` is below its safety threshold -- this routine only
    # surfaces the value, it does not enforce a cutoff.
    # ``|v_0|^2`` is the overlap magnitude squared, which stays
    # real-positive when ``eigvecs_lm`` is complex (``eigh``
    # of a Hermitian H_bar). Bit-identical at Gamma since ``np.abs(x)**2
    # == x**2`` for real ``x``.
    v0_sq = np.abs(eigvecs_lm[0, :]) ** 2
    best_idx = int(np.argmax(v0_sq))
    E_lm = float(eigvals_lm[best_idx])
    v0_sq_best = float(v0_sq[best_idx])

    # Diagnostic
    lowest_idx = 0
    if lowest_idx != best_idx:
        logger.debug(
            "  LM: lowest eigenvalue E_LM = %.6f (|v0|^2 = %.4f), selected (max |v0|^2) E_LM = %.6f (|v0|^2 = %.4f)",
            eigvals_lm[lowest_idx],
            v0_sq[lowest_idx],
            E_lm,
            v0_sq_best,
        )
    else:
        logger.debug("  LM: selected eigenvalue E_LM = %.6f (|v0|^2 = %.4f)", E_lm, v0_sq_best)

    w = eigvecs_lm[:, best_idx]
    w0 = w[0]
    c_new = w[1:] / w0  # (p',) in S-orthonormal basis

    # ---- Back-transform: P @ c_new -> alive parameter space -> full space ----
    c_alive = P @ c_new  # (n_alive,)
    c_vec = np.zeros(p, dtype=dtype_mcmc_np)
    c_vec[idx] = c_alive

    logger.info(
        "  LM: E_LM = %.6f (|v0|^2 = %.4f), ||c|| = %.3e, max|c| = %.3e",
        E_lm,
        v0_sq_best,
        np.linalg.norm(c_vec),
        np.max(np.abs(c_vec)),
    )

    return c_vec, E_lm, v0_sq_best


# ---------------------------------------------------------------------------
# Device-resident SR solvers (JAX-native, sharding + psum / all_gather)
#
# Implements the four SR solve paths:
#   - wide + direct : (X X^T + eps I) y = X F       via psum
#   - wide + CG     :  same system, conjugate gradient with psum'd matvec
#   - tall + direct : (X^T X + eps I) z = F, theta = X z   via all_gather
#   - tall + CG     :  same system, conjugate gradient via psum on
#                      sharded (N_local,) state -- ``X`` stays sharded so
#                      per-rank memory is independent of ``mpi_size``.
#
# Compiled kernels are cached at module level so the JIT cost is paid once
# per process. The same code path runs on:
#   - single-process CPU       (psum is trivial; mesh has 1 device)
#   - multi-process CPU + Gloo (after ``jax.distributed.initialize``)
#   - GPU (single or multi process; psum maps to NVLink / NCCL)
# ---------------------------------------------------------------------------
def _realify_tall_samples(X: jax.Array, F: jax.Array) -> tuple[jax.Array, jax.Array]:
    """In-jit tall realification ``Z = [Re X, Im X]``, ``h = [Re F; -Im F]``.

    Real samples (a real wave function: real ``X`` and ``F``) have an
    identically zero imaginary block whose dual is exactly zero; it is omitted
    and ``(X, F)`` is the realified system itself. The dispatch is on the
    static dtype, so complex samples trace exactly as before.
    """
    if not jnp.iscomplexobj(X) and not jnp.iscomplexobj(F):
        return X, F
    return jnp.concatenate((jnp.real(X), jnp.imag(X)), axis=1), jnp.concatenate((jnp.real(F), -jnp.imag(F)))


def _get_sr_mesh() -> jax.sharding.Mesh:
    """Lazy-built 1-D ``Mesh`` over all visible JAX devices, axis name 'rank'."""
    cached = getattr(_get_sr_mesh, "_cached", None)
    if cached is not None:
        return cached
    from jax.sharding import Mesh

    mesh = Mesh(np.array(jax.devices()), axis_names=("rank",))
    _get_sr_mesh._cached = mesh
    return mesh


def shard_sample_rows(*local_arrays: jnpt.ArrayLike) -> tuple[jax.Array, ...]:
    """Stitch device-local sample-leading arrays into rank-sharded arrays."""
    if jax.process_count() <= 1:
        return tuple(jnp.asarray(array) for array in local_arrays)
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as PSpec

    mesh = _get_sr_mesh()
    return tuple(
        jax.make_array_from_process_local_data(NamedSharding(mesh, PSpec("rank", *([None] * (array.ndim - 1)))), array)
        for array in map(jnp.asarray, local_arrays)
    )


def _get_device_sr_prepare_kernel() -> DevicePrepareKernel:
    """Center and weight SR samples using device collectives only."""
    cached = getattr(_get_device_sr_prepare_kernel, "_cached", None)
    if cached is not None:
        return cached
    from jax.sharding import PartitionSpec as PSpec

    @partial(
        jax.shard_map,
        mesh=_get_sr_mesh(),
        in_specs=(PSpec("rank", None), PSpec("rank"), PSpec("rank")),
        out_specs=(PSpec(None, "rank"), PSpec("rank"), PSpec(), PSpec(), PSpec()),
    )
    def prepare(O: jax.Array, weights: jax.Array, energies: jax.Array) -> PreparedSamples:
        total_weight = jax.lax.psum(jnp.sum(weights), "rank")
        weighted_O = jax.lax.psum(jnp.einsum("n,nq->q", weights, O), "rank")
        weighted_energy = jax.lax.psum(jnp.dot(weights, energies), "rank")
        O_bar = weighted_O / total_weight
        energy_bar = jnp.real(weighted_energy / total_weight)
        sqrt_weight = jnp.sqrt(weights)
        X = (jnp.conj(O - O_bar) * sqrt_weight[:, None] / jnp.sqrt(total_weight)).T
        F = -2.0 * sqrt_weight * (energies - energy_bar) / jnp.sqrt(total_weight)
        diagonal = jax.lax.psum(jnp.real(jnp.einsum("qn,qn->q", X, jnp.conj(X))), "rank")
        force = jax.lax.psum(jnp.real(X @ F), "rank")
        return X, F, diagonal, force, energy_bar

    # The coordinate matrix is dead after centering.  Donate its storage to
    # the equally-sized centered sample matrix instead of keeping both live.
    prepare = jax.jit(prepare, donate_argnums=(0,))
    _get_device_sr_prepare_kernel._cached = prepare
    return prepare


def prepare_device_sr_samples(
    O_local: jnpt.ArrayLike,
    weights_local: jnpt.ArrayLike,
    energies_local: jnpt.ArrayLike,
) -> PreparedSamples:
    """Return sharded centered SR arrays plus replicated small statistics."""
    O, weights, energies = shard_sample_rows(O_local, weights_local, energies_local)
    return _get_device_sr_prepare_kernel()(O, weights, energies)


def _get_local_device_sr_prepare_kernel() -> LocalDevicePrepareKernel:
    """Center one sampler-local SR scope without entering the world mesh."""
    cached = getattr(_get_local_device_sr_prepare_kernel, "_cached", None)
    if cached is not None:
        return cached

    @partial(jax.jit, donate_argnums=(0,))
    def prepare(
        O: jax.Array,
        weights: jax.Array,
        energies: jax.Array,
        total_weight: jax.Array,
        weighted_O: jax.Array,
        weighted_energy: jax.Array,
    ) -> PreparedSamples:
        return _center_local_sr_samples(O, weights, energies, total_weight, weighted_O, weighted_energy)

    _get_local_device_sr_prepare_kernel._cached = prepare
    return prepare


def _center_local_sr_samples(
    O: jax.Array,
    weights: jax.Array,
    energies: jax.Array,
    total_weight: jax.Array,
    weighted_O: jax.Array,
    weighted_energy: jax.Array,
) -> PreparedSamples:
    """Centered, weighted ``(X, F)`` of one local sample block plus its local statistics."""
    O_bar = weighted_O / total_weight
    energy_bar = jnp.real(weighted_energy / total_weight)
    sqrt_weight = jnp.sqrt(weights)
    X = (jnp.conj(O - O_bar) * sqrt_weight[:, None] / jnp.sqrt(total_weight)).T
    F = -2.0 * sqrt_weight * (energies - energy_bar) / jnp.sqrt(total_weight)
    diagonal_local = jnp.real(jnp.einsum("qn,qn->q", X, jnp.conj(X)))
    force_local = jnp.real(X @ F)
    return X, F, diagonal_local, force_local, energy_bar


def _get_local_device_sr_prepare_into_kernel() -> Callable[..., PreparedSamples]:
    """Center one sample block and write its ``X`` into a donated contiguous matrix."""
    cached = getattr(_get_local_device_sr_prepare_into_kernel, "_cached", None)
    if cached is not None:
        return cached

    @partial(jax.jit, donate_argnums=(0,))
    def prepare_into(
        X_matrix: jax.Array,
        start: jax.Array,
        O: jax.Array,
        weights: jax.Array,
        energies: jax.Array,
        total_weight: jax.Array,
        weighted_O: jax.Array,
        weighted_energy: jax.Array,
    ) -> PreparedSamples:
        X, F, diagonal_local, force_local, energy_bar = _center_local_sr_samples(
            O, weights, energies, total_weight, weighted_O, weighted_energy
        )
        zero = jnp.zeros((), dtype=start.dtype)
        return jax.lax.dynamic_update_slice(X_matrix, X, (zero, start)), F, diagonal_local, force_local, energy_bar

    _get_local_device_sr_prepare_into_kernel._cached = prepare_into
    return prepare_into


def local_device_sr_sample_dtype(
    O_local: jax.Array,
    weights_local: jnpt.ArrayLike,
    energies_local: jnpt.ArrayLike,
    *,
    total_weight: jnpt.ArrayLike,
    total_weighted_O: jnpt.ArrayLike,
    total_weighted_energy: jnpt.ArrayLike,
) -> np.dtype:
    """Dtype of the centered ``X`` produced from these operands (shape evaluation only)."""
    shapes = jax.eval_shape(
        _center_local_sr_samples,
        O_local,
        jnp.asarray(weights_local),
        jnp.asarray(energies_local),
        jnp.asarray(total_weight),
        jnp.asarray(total_weighted_O),
        jnp.asarray(total_weighted_energy),
    )
    return np.dtype(shapes[0].dtype)


def prepare_local_device_sr_samples_into(
    X_matrix: jax.Array,
    start: int,
    O_local: jnpt.ArrayLike,
    weights_local: jnpt.ArrayLike,
    energies_local: jnpt.ArrayLike,
    *,
    total_weight: jnpt.ArrayLike,
    total_weighted_O: jnpt.ArrayLike,
    total_weighted_energy: jnpt.ArrayLike,
) -> PreparedSamples:
    """Prepare one sample block straight into columns ``start:`` of a contiguous matrix.

    ``X_matrix`` (coordinates x samples) is donated: the centered block is
    written in place, so building one contiguous matrix costs no second copy
    of the samples. (``O_local`` cannot alias an output and is not donated;
    the caller drops it after the call.) Returns the updated matrix in place
    of the block's ``X``; the block arithmetic is that of
    :func:`prepare_local_device_sr_samples`.
    """
    return _get_local_device_sr_prepare_into_kernel()(
        X_matrix,
        jnp.asarray(start, dtype=jnp.int32),
        jnp.asarray(O_local),
        jnp.asarray(weights_local),
        jnp.asarray(energies_local),
        jnp.asarray(total_weight),
        jnp.asarray(total_weighted_O),
        jnp.asarray(total_weighted_energy),
    )


def prepare_local_device_sr_samples(
    O_local: jnpt.ArrayLike,
    weights_local: jnpt.ArrayLike,
    energies_local: jnpt.ArrayLike,
    *,
    total_weight: jnpt.ArrayLike,
    total_weighted_O: jnpt.ArrayLike,
    total_weighted_energy: jnpt.ArrayLike,
) -> PreparedSamples:
    """Prepare a device-resident local scope using MPI-reduced statistics.

    Unlike :func:`prepare_device_sr_samples`, this function deliberately uses
    no ``shard_map`` or JAX collective.  It is therefore safe when a twist's
    sampler communicator is a strict subset of the JAX world mesh.
    """
    return _get_local_device_sr_prepare_kernel()(
        jnp.asarray(O_local),
        jnp.asarray(weights_local),
        jnp.asarray(energies_local),
        jnp.asarray(total_weight),
        jnp.asarray(total_weighted_O),
        jnp.asarray(total_weighted_energy),
    )


def _get_device_sr_precondition_kernel() -> DevicePreconditionKernel:
    """Scale every SR coordinate row while reusing the input buffer."""
    cached = getattr(_get_device_sr_precondition_kernel, "_cached", None)
    if cached is not None:
        return cached

    @partial(jax.jit, donate_argnums=(0,))
    def precondition(X: jax.Array, inverse_sqrt: jax.Array) -> jax.Array:
        scale = inverse_sqrt[:, None]
        # A frozen row (zero scale) is an exact zero whatever it held, so a
        # non-finite frozen row cannot reach the solve; an active row is the
        # plain product.
        return jnp.where(scale != 0, X * scale, jnp.zeros((), dtype=X.dtype))

    _get_device_sr_precondition_kernel._cached = precondition
    return precondition


def precondition_device_sr_samples(X: jax.Array, inverse_sqrt: jnpt.ArrayLike) -> jax.Array:
    """Row-precondition ``X`` in place of its own buffer (``X`` is donated).

    ``inverse_sqrt`` holds one scale per coordinate row, zero on the rows of
    frozen coordinate groups. The result keeps every row, frozen rows as
    exact zeros, so a frozen group never costs a second full matrix: the
    solvers run on all rows, where a zero row with a zero right-hand side
    entry keeps its solution entry exactly zero.
    """
    return _get_device_sr_precondition_kernel()(X, jnp.asarray(inverse_sqrt))


def _wide_direct_diagonal_shift(
    epsilon: jnpt.ArrayLike, size: int, frozen_rows: npt.ArrayLike | None, dtype: npt.DTypeLike
) -> np.ndarray:
    """Diagonal of the shifted wide operator: ``epsilon`` on every row, ``epsilon + 1`` on frozen rows.

    A frozen row of the preconditioned matrix is an exact zero row with a zero
    right-hand side entry. Its Gram row and column are zero, so with
    ``epsilon = 0`` the full operator would be singular although the active
    block is not. The unit shift makes that artificial equation
    ``(epsilon + 1) q_f = 0`` well-posed with the exact solution ``q_f = 0``
    and leaves the active block, and the physical ``epsilon``, unchanged.
    """
    shift = np.full(int(size), float(epsilon), dtype=dtype)
    if frozen_rows is not None:
        shift[np.asarray(frozen_rows, dtype=np.int64)] += 1.0
    return shift


def solve_device_wide_real_direct_sharded(
    X: jax.Array,
    b: jnpt.ArrayLike,
    epsilon: jnpt.ArrayLike,
    *,
    frozen_rows: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """Run the direct wide solve from an already device-sharded matrix.

    ``frozen_rows`` lists the exact-zero rows of frozen coordinate groups;
    their solution entries are exactly zero (see :func:`_wide_direct_diagonal_shift`).
    """
    dtype = jnp.real(X).dtype
    q = _get_sr_wide_real_direct_kernel()(
        X,
        _replicated_jax_array(jnp.asarray(b, dtype=dtype)),
        _replicated_jax_array(jnp.asarray(_wide_direct_diagonal_shift(epsilon, X.shape[0], frozen_rows, dtype))),
    )
    q.block_until_ready()
    return np.asarray(q)


def _get_local_device_wide_real_cg_kernel() -> DeviceWideCGKernel:
    cached = getattr(_get_local_device_wide_real_cg_kernel, "_cached", None)
    if cached is not None:
        return cached

    @jax.jit
    def solve(
        X: jax.Array,
        b: jax.Array,
        epsilon: jax.Array,
        max_iter: jax.Array,
        tol: jax.Array,
        x0: jax.Array,
    ) -> DeviceSolveResult:
        def apply_operator(vector: jax.Array) -> jax.Array:
            return jnp.real(X @ (jnp.conj(X).T @ vector)) + epsilon * vector

        return _cg_while_loop(b, apply_operator, x0, max_iter, tol, b.dtype)

    _get_local_device_wide_real_cg_kernel._cached = solve
    return solve


def solve_local_device_wide_real_cg(
    X: jax.Array,
    b: jnpt.ArrayLike,
    epsilon: jnpt.ArrayLike,
    *,
    max_iter: int,
    tol: float,
    x0: jnpt.ArrayLike,
) -> HostSolveResult:
    """Solve a sampler-local wide SR system entirely on the local device."""
    dtype = jnp.real(X).dtype
    q, residual, iterations, breakdown = _get_local_device_wide_real_cg_kernel()(
        X,
        jnp.asarray(b, dtype=dtype),
        jnp.asarray(epsilon, dtype=dtype),
        jnp.asarray(max_iter, dtype=jnp.int32),
        jnp.asarray(tol, dtype=dtype),
        jnp.asarray(x0, dtype=dtype),
    )
    q.block_until_ready()
    return np.asarray(q), float(residual), int(iterations), bool(breakdown)


def _get_local_device_wide_real_direct_kernel() -> DeviceDirectKernel:
    cached = getattr(_get_local_device_wide_real_direct_kernel, "_cached", None)
    if cached is not None:
        return cached

    @jax.jit
    def solve(X: jax.Array, b: jax.Array, shift: jax.Array) -> jax.Array:
        Xc = jnp.conj(X).T
        metric = jnp.real(X @ Xc)
        return _device_refined_direct_solve(
            metric + jnp.diag(shift),
            b,
            lambda vector: jnp.real(X @ (Xc @ vector)) + shift * vector,
        )

    _get_local_device_wide_real_direct_kernel._cached = solve
    return solve


def solve_local_device_wide_real_direct(
    X: jax.Array,
    b: jnpt.ArrayLike,
    epsilon: jnpt.ArrayLike,
    *,
    frozen_rows: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """Solve a sampler-local wide SR system directly on the local device.

    ``frozen_rows`` lists the exact-zero rows of frozen coordinate groups;
    their solution entries are exactly zero (see :func:`_wide_direct_diagonal_shift`).
    """
    dtype = jnp.real(X).dtype
    q = _get_local_device_wide_real_direct_kernel()(
        X,
        jnp.asarray(b, dtype=dtype),
        jnp.asarray(_wide_direct_diagonal_shift(epsilon, X.shape[0], frozen_rows, dtype)),
    )
    q.block_until_ready()
    return np.asarray(q)


def _get_local_device_tall_real_cg_kernel() -> DeviceTallCGKernel:
    cached = getattr(_get_local_device_tall_real_cg_kernel, "_cached", None)
    if cached is not None:
        return cached

    @jax.jit
    def solve(
        X: jax.Array,
        F: jax.Array,
        epsilon: jax.Array,
        max_iter: jax.Array,
        tol: jax.Array,
    ) -> DeviceSolveResult:
        Z, h = _realify_tall_samples(X, F)

        def apply_operator(vector: jax.Array) -> jax.Array:
            return Z.T @ (Z @ vector) + epsilon * vector

        dual, residual, iterations, breakdown = _cg_while_loop(
            h,
            apply_operator,
            jnp.zeros_like(h),
            max_iter,
            tol,
            h.dtype,
        )
        return Z @ dual, residual, iterations, breakdown

    _get_local_device_tall_real_cg_kernel._cached = solve
    return solve


def solve_local_device_tall_real_cg(
    X: jax.Array,
    F: jnpt.ArrayLike,
    epsilon: jnpt.ArrayLike,
    *,
    max_iter: int,
    tol: float,
) -> HostSolveResult:
    """Solve a sampler-local tall SR system entirely on the local device."""
    dtype = jnp.real(X).dtype
    q, residual, iterations, breakdown = _get_local_device_tall_real_cg_kernel()(
        X,
        jnp.asarray(F),
        jnp.asarray(epsilon, dtype=dtype),
        jnp.asarray(max_iter, dtype=jnp.int32),
        jnp.asarray(tol, dtype=dtype),
    )
    q.block_until_ready()
    return np.asarray(q), float(residual), int(iterations), bool(breakdown)


def _wide_gram_apply(X: jax.Array, vector: jax.Array) -> jax.Array:
    """``Re(X X^H) v`` for a real vector without a matrix-sized temporary.

    ``X^H v = conj(v^T X)`` for real ``v``; ``v @ X`` contracts the leading
    axis in place, so neither ``conj(X)`` nor a transposed copy of the
    resident sample matrix is formed.
    """
    if not jnp.iscomplexobj(X):
        return X @ (vector @ X)
    return jnp.real(X @ jnp.conj(vector @ X))


def _tall_sample_to_coordinate(X: jax.Array, real_part: jax.Array, imag_part: jax.Array) -> jax.Array:
    """``Re(X) a + Im(X) c`` as one product ``Re(X (a - i c))``, without ``Re(X)``/``Im(X)`` copies."""
    if not jnp.iscomplexobj(X):
        return X @ real_part  # Im(X) = 0
    return jnp.real(X @ (real_part - 1.0j * imag_part))


def _tall_coordinate_to_sample(X: jax.Array, vector: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(Re(X)^T v, Im(X)^T v)`` from one product ``v @ X``, without ``Re(X)``/``Im(X)`` copies."""
    product = vector @ X
    if not jnp.iscomplexobj(X):
        return product, jnp.zeros_like(product)  # Im(X) = 0
    return jnp.real(product), jnp.imag(product)


def _get_local_device_wide_real_cg_chunks_kernel() -> DeviceWideChunkCGKernel:
    cached = getattr(_get_local_device_wide_real_cg_chunks_kernel, "_cached", None)
    if cached is not None:
        return cached

    @jax.jit
    def solve(
        X_chunks: tuple[jax.Array, ...],
        b: jax.Array,
        epsilon: jax.Array,
        max_iter: jax.Array,
        tol: jax.Array,
        x0: jax.Array,
    ) -> DeviceSolveResult:
        def apply_operator(vector: jax.Array) -> jax.Array:
            contributions = tuple(_wide_gram_apply(X, vector) for X in X_chunks)
            return jnp.sum(jnp.stack(contributions), axis=0) + epsilon * vector

        return _cg_while_loop(b, apply_operator, x0, max_iter, tol, b.dtype)

    _get_local_device_wide_real_cg_chunks_kernel._cached = solve
    return solve


def solve_local_device_wide_real_cg_chunks(
    X_chunks: Sequence[jnpt.ArrayLike],
    b: jnpt.ArrayLike,
    epsilon: jnpt.ArrayLike,
    *,
    max_iter: int,
    tol: float,
    x0: jnpt.ArrayLike,
) -> HostSolveResult:
    """Solve a local wide SR system from bounded resident Jacobian chunks."""
    chunks = tuple(jnp.asarray(X) for X in X_chunks)
    if not chunks:
        raise ValueError("chunked wide SR requires at least one Jacobian chunk.")
    coordinate_size = int(chunks[0].shape[0])
    if any(X.ndim != 2 or int(X.shape[0]) != coordinate_size for X in chunks):
        raise ValueError("chunked wide SR Jacobians must share one coordinate axis.")
    dtype = jnp.real(chunks[0]).dtype
    q, residual, iterations, breakdown = _get_local_device_wide_real_cg_chunks_kernel()(
        chunks,
        jnp.asarray(b, dtype=dtype),
        jnp.asarray(epsilon, dtype=dtype),
        jnp.asarray(max_iter, dtype=jnp.int32),
        jnp.asarray(tol, dtype=dtype),
        jnp.asarray(x0, dtype=dtype),
    )
    q.block_until_ready()
    return np.asarray(q), float(residual), int(iterations), bool(breakdown)


def _get_local_device_tall_real_cg_chunks_kernel() -> DeviceTallChunkCGKernel:
    cached = getattr(_get_local_device_tall_real_cg_chunks_kernel, "_cached", None)
    if cached is not None:
        return cached

    @jax.jit
    def solve(
        X_chunks: tuple[jax.Array, ...],
        F_chunks: tuple[jax.Array, ...],
        epsilon: jax.Array,
        max_iter: jax.Array,
        tol: jax.Array,
    ) -> DeviceSolveResult:
        # Real chunks (a real wave function) carry no zero imaginary dual block.
        complex_samples = any(jnp.iscomplexobj(X) for X in X_chunks) or any(jnp.iscomplexobj(F) for F in F_chunks)
        components = 2 if complex_samples else 1
        sample_sizes = tuple(components * int(X.shape[1]) for X in X_chunks)
        offsets = np.cumsum((0, *sample_sizes))
        if complex_samples:
            h = jnp.concatenate(tuple(component for F in F_chunks for component in (jnp.real(F), -jnp.imag(F))))
        else:
            h = jnp.concatenate(tuple(F_chunks))

        def sample_to_coordinate(vector: jax.Array) -> jax.Array:
            contributions = []
            for index, X in enumerate(X_chunks):
                sample_count = int(X.shape[1])
                values = vector[offsets[index] : offsets[index + 1]]
                if complex_samples:
                    contributions.append(_tall_sample_to_coordinate(X, values[:sample_count], values[sample_count:]))
                else:
                    contributions.append(X @ values)
            return jnp.sum(jnp.stack(contributions), axis=0)

        def coordinate_to_sample(vector: jax.Array) -> jax.Array:
            if not complex_samples:
                return jnp.concatenate(tuple(vector @ X for X in X_chunks))
            return jnp.concatenate(tuple(component for X in X_chunks for component in _tall_coordinate_to_sample(X, vector)))

        def apply_operator(vector: jax.Array) -> jax.Array:
            return coordinate_to_sample(sample_to_coordinate(vector)) + epsilon * vector

        dual, residual, iterations, breakdown = _cg_while_loop(
            h,
            apply_operator,
            jnp.zeros_like(h),
            max_iter,
            tol,
            h.dtype,
        )
        return sample_to_coordinate(dual), residual, iterations, breakdown

    _get_local_device_tall_real_cg_chunks_kernel._cached = solve
    return solve


def solve_local_device_tall_real_cg_chunks(
    X_chunks: Sequence[jnpt.ArrayLike],
    F_chunks: Sequence[jnpt.ArrayLike],
    epsilon: jnpt.ArrayLike,
    *,
    max_iter: int,
    tol: float,
) -> HostSolveResult:
    """Solve a local tall SR system from bounded resident Jacobian chunks."""
    chunks = tuple(jnp.asarray(X) for X in X_chunks)
    forces = tuple(jnp.asarray(F) for F in F_chunks)
    if not chunks or len(chunks) != len(forces):
        raise ValueError("chunked tall SR requires matching Jacobian and force chunks.")
    coordinate_size = int(chunks[0].shape[0])
    if any(X.ndim != 2 or int(X.shape[0]) != coordinate_size for X in chunks):
        raise ValueError("chunked tall SR Jacobians must share one coordinate axis.")
    if any(F.ndim != 1 or int(F.size) != int(X.shape[1]) for X, F in zip(chunks, forces, strict=True)):
        raise ValueError("chunked tall SR force chunks must match their sample axes.")
    dtype = jnp.real(chunks[0]).dtype
    q, residual, iterations, breakdown = _get_local_device_tall_real_cg_chunks_kernel()(
        chunks,
        forces,
        jnp.asarray(epsilon, dtype=dtype),
        jnp.asarray(max_iter, dtype=jnp.int32),
        jnp.asarray(tol, dtype=dtype),
    )
    q.block_until_ready()
    return np.asarray(q), float(residual), int(iterations), bool(breakdown)


_DEVICE_WIDE_REAL_CG_CHUNKS_KERNEL_CACHE = {}


def _get_device_wide_real_cg_chunks_sharded_kernel(chunk_count: int) -> DeviceWideChunkCGKernel:
    cached = _DEVICE_WIDE_REAL_CG_CHUNKS_KERNEL_CACHE.get(chunk_count)
    if cached is not None:
        return cached
    from jax.sharding import PartitionSpec as PSpec

    @partial(
        jax.shard_map,
        mesh=_get_sr_mesh(),
        in_specs=(tuple(PSpec(None, "rank") for _ in range(chunk_count)), PSpec(), PSpec(), PSpec(), PSpec(), PSpec()),
        out_specs=(PSpec(), PSpec(), PSpec(), PSpec()),
    )
    def solve(
        X_chunks: tuple[jax.Array, ...],
        b: jax.Array,
        epsilon: jax.Array,
        max_iter: jax.Array,
        tol: jax.Array,
        x0: jax.Array,
    ) -> DeviceSolveResult:
        def apply_operator(vector: jax.Array) -> jax.Array:
            local = jnp.sum(
                jnp.stack(tuple(_wide_gram_apply(X, vector) for X in X_chunks)),
                axis=0,
            )
            return jax.lax.psum(local, "rank") + epsilon * vector

        return _cg_while_loop(b, apply_operator, x0, max_iter, tol, b.dtype)

    compiled = jax.jit(solve)
    _DEVICE_WIDE_REAL_CG_CHUNKS_KERNEL_CACHE[chunk_count] = compiled
    return compiled


def solve_device_wide_real_cg_chunks_sharded(
    X_chunks: Sequence[jnpt.ArrayLike],
    b: jnpt.ArrayLike,
    epsilon: jnpt.ArrayLike,
    *,
    max_iter: int,
    tol: float,
    x0: jnpt.ArrayLike,
) -> HostSolveResult:
    """Solve distributed wide SR from bounded local Jacobian chunks."""
    local_chunks = tuple(jnp.asarray(X) for X in X_chunks)
    if not local_chunks:
        raise ValueError("sharded chunked wide SR requires at least one Jacobian chunk.")
    coordinate_size = int(local_chunks[0].shape[0])
    if any(X.ndim != 2 or int(X.shape[0]) != coordinate_size for X in local_chunks):
        raise ValueError("sharded chunked wide SR Jacobians must share one coordinate axis.")
    chunks = tuple(_shard_X(X) for X in local_chunks)
    dtype = jnp.real(local_chunks[0]).dtype
    q, residual, iterations, breakdown = _get_device_wide_real_cg_chunks_sharded_kernel(len(chunks))(
        chunks,
        _replicated_jax_array(jnp.asarray(b, dtype=dtype)),
        jnp.asarray(epsilon, dtype=dtype),
        jnp.asarray(max_iter, dtype=jnp.int32),
        jnp.asarray(tol, dtype=dtype),
        _replicated_jax_array(jnp.asarray(x0, dtype=dtype)),
    )
    q.block_until_ready()
    return np.asarray(q), float(residual), int(iterations), bool(breakdown)


_DEVICE_TALL_REAL_CG_CHUNKS_KERNEL_CACHE = {}


def _get_device_tall_real_cg_chunks_sharded_kernel(chunk_count: int) -> DeviceTallChunkCGKernel:
    cached = _DEVICE_TALL_REAL_CG_CHUNKS_KERNEL_CACHE.get(chunk_count)
    if cached is not None:
        return cached
    from jax.sharding import PartitionSpec as PSpec

    @partial(
        jax.shard_map,
        mesh=_get_sr_mesh(),
        in_specs=(
            tuple(PSpec(None, "rank") for _ in range(chunk_count)),
            tuple(PSpec("rank") for _ in range(chunk_count)),
            PSpec(),
            PSpec(),
            PSpec(),
        ),
        out_specs=(PSpec(), PSpec(), PSpec(), PSpec()),
    )
    def solve(
        X_chunks: tuple[jax.Array, ...],
        F_chunks: tuple[jax.Array, ...],
        epsilon: jax.Array,
        max_iter: jax.Array,
        tol: jax.Array,
    ) -> DeviceSolveResult:
        # Real chunks (a real wave function) carry no zero imaginary dual block.
        complex_samples = any(jnp.iscomplexobj(X) for X in X_chunks) or any(jnp.iscomplexobj(F) for F in F_chunks)
        components = 2 if complex_samples else 1
        sample_sizes = tuple(components * int(X.shape[1]) for X in X_chunks)
        offsets = np.cumsum((0, *sample_sizes))
        if complex_samples:
            h = jnp.concatenate(tuple(component for F in F_chunks for component in (jnp.real(F), -jnp.imag(F))))
        else:
            h = jnp.concatenate(tuple(F_chunks))

        def sample_to_coordinate(vector: jax.Array) -> jax.Array:
            contributions = []
            for index, X in enumerate(X_chunks):
                sample_count = int(X.shape[1])
                values = vector[offsets[index] : offsets[index + 1]]
                if complex_samples:
                    contributions.append(_tall_sample_to_coordinate(X, values[:sample_count], values[sample_count:]))
                else:
                    contributions.append(X @ values)
            local = jnp.sum(jnp.stack(contributions), axis=0)
            return jax.lax.psum(local, "rank")

        def coordinate_to_sample(vector: jax.Array) -> jax.Array:
            if not complex_samples:
                return jnp.concatenate(tuple(vector @ X for X in X_chunks))
            return jnp.concatenate(tuple(component for X in X_chunks for component in _tall_coordinate_to_sample(X, vector)))

        def apply_operator(vector: jax.Array) -> jax.Array:
            return coordinate_to_sample(sample_to_coordinate(vector)) + epsilon * vector

        dual, residual, iterations, breakdown = _cg_while_loop_sharded(
            h,
            apply_operator,
            jnp.zeros_like(h),
            max_iter,
            tol,
            h.dtype,
            axis_name="rank",
        )
        return sample_to_coordinate(dual), residual, iterations, breakdown

    compiled = jax.jit(solve)
    _DEVICE_TALL_REAL_CG_CHUNKS_KERNEL_CACHE[chunk_count] = compiled
    return compiled


def solve_device_tall_real_cg_chunks_sharded(
    X_chunks: Sequence[jnpt.ArrayLike],
    F_chunks: Sequence[jnpt.ArrayLike],
    epsilon: jnpt.ArrayLike,
    *,
    max_iter: int,
    tol: float,
) -> HostSolveResult:
    """Solve distributed tall SR from bounded local Jacobian chunks."""
    local_chunks = tuple(jnp.asarray(X) for X in X_chunks)
    local_forces = tuple(jnp.asarray(F) for F in F_chunks)
    if not local_chunks or len(local_chunks) != len(local_forces):
        raise ValueError("sharded chunked tall SR requires matching Jacobian and force chunks.")
    coordinate_size = int(local_chunks[0].shape[0])
    if any(X.ndim != 2 or int(X.shape[0]) != coordinate_size for X in local_chunks):
        raise ValueError("sharded chunked tall SR Jacobians must share one coordinate axis.")
    if any(F.ndim != 1 or int(F.size) != int(X.shape[1]) for X, F in zip(local_chunks, local_forces, strict=True)):
        raise ValueError("sharded chunked tall SR force chunks must match their sample axes.")
    sharded = tuple(_shard_X_F(X, F) for X, F in zip(local_chunks, local_forces, strict=True))
    chunks = tuple(values[0] for values in sharded)
    forces = tuple(values[1] for values in sharded)
    dtype = jnp.real(local_chunks[0]).dtype
    q, residual, iterations, breakdown = _get_device_tall_real_cg_chunks_sharded_kernel(len(chunks))(
        chunks,
        forces,
        jnp.asarray(epsilon, dtype=dtype),
        jnp.asarray(max_iter, dtype=jnp.int32),
        jnp.asarray(tol, dtype=dtype),
    )
    q.block_until_ready()
    return np.asarray(q), float(residual), int(iterations), bool(breakdown)


def _get_local_device_tall_real_direct_kernel() -> DeviceDirectKernel:
    cached = getattr(_get_local_device_tall_real_direct_kernel, "_cached", None)
    if cached is not None:
        return cached

    @jax.jit
    def solve(X: jax.Array, F: jax.Array, epsilon: jax.Array) -> jax.Array:
        Z, h = _realify_tall_samples(X, F)
        operator = Z.T @ Z + epsilon * jnp.eye(h.size, dtype=Z.dtype)
        return Z @ _device_refined_direct_solve(operator, h, lambda dual: Z.T @ (Z @ dual) + epsilon * dual)

    _get_local_device_tall_real_direct_kernel._cached = solve
    return solve


def solve_local_device_tall_real_direct(
    X: jax.Array,
    F: jnpt.ArrayLike,
    epsilon: jnpt.ArrayLike,
) -> npt.NDArray:
    """Solve a sampler-local tall SR system directly on the local device."""
    dtype = jnp.real(X).dtype
    q = _get_local_device_tall_real_direct_kernel()(
        X,
        jnp.asarray(F),
        jnp.asarray(epsilon, dtype=dtype),
    )
    q.block_until_ready()
    return np.asarray(q)


def _get_device_lm_pencil_kernel() -> DeviceLMPencilKernel:
    """Build a small reduced LM pencil from sample-sharded O/D columns."""
    cached = getattr(_get_device_lm_pencil_kernel, "_cached", None)
    if cached is not None:
        return cached
    from jax.sharding import PartitionSpec as PSpec

    @partial(
        jax.shard_map,
        mesh=_get_sr_mesh(),
        in_specs=(PSpec("rank", None), PSpec("rank", None), PSpec("rank"), PSpec("rank")),
        out_specs=(PSpec(), PSpec(), PSpec(), PSpec()),
    )
    def build(
        O: jax.Array,
        D: jax.Array,
        weights: jax.Array,
        energies: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        total_weight = jax.lax.psum(jnp.sum(weights), "rank")
        O_bar = jax.lax.psum(jnp.einsum("n,na->a", weights, O), "rank") / total_weight
        D_bar = jax.lax.psum(jnp.einsum("n,na->a", weights, D), "rank") / total_weight
        energy_bar = jax.lax.psum(jnp.dot(weights, energies), "rank") / total_weight
        dO = O - O_bar
        dD = D - D_bar
        de = energies - energy_bar
        S = jax.lax.psum(jnp.einsum("n,na,nb->ab", weights, jnp.conj(dO), dO), "rank") / total_weight
        force = -2.0 * jax.lax.psum(jnp.einsum("n,na,n->a", weights, jnp.conj(dO), de), "rank") / total_weight
        # NON-SYMMETRIC linear method (matches the host builder
        # ``_build_lm_real_pencil_from_coordinate_arrays``): the bra row keeps
        # the UNCONJUGATED covariance plus the mean local-energy derivative
        # D_bar, the bulk keeps K + B^dagger, and no Hermitian averaging is
        # applied to H. This preserves the finite sample-level local-energy
        # derivative carried by the bra row.
        force_bra_cov = -2.0 * jax.lax.psum(jnp.einsum("n,na,n->a", weights, dO, de), "rank") / total_weight
        K = jax.lax.psum(jnp.einsum("n,n,na,nb->ab", weights, energies, jnp.conj(dO), dO), "rank") / total_weight
        B = jax.lax.psum(jnp.einsum("n,na,nb->ab", weights, jnp.conj(dD), dO), "rank") / total_weight
        M = K + B.conj().T
        size = O.shape[1]
        H_c = jnp.zeros((size + 1, size + 1), dtype=jnp.result_type(O, D, energies))
        S_c = jnp.zeros_like(H_c)
        H_c = H_c.at[0, 0].set(energy_bar)
        H_c = H_c.at[1:, 0].set(-0.5 * force)
        H_c = H_c.at[0, 1:].set(-0.5 * force_bra_cov + D_bar)
        H_c = H_c.at[1:, 1:].set(M)
        S_c = S_c.at[0, 0].set(1.0)
        S_c = S_c.at[1:, 1:].set(S)
        H = jnp.real(H_c)
        overlap = jnp.real(S_c)
        conditioning_metric = jnp.real(S)
        conditioning_metric = 0.5 * (conditioning_metric + conditioning_metric.T)
        return H, 0.5 * (overlap + overlap.T), jnp.real(energy_bar), conditioning_metric

    build = jax.jit(build)
    _get_device_lm_pencil_kernel._cached = build
    return build


def build_device_lm_pencil(
    O_local: jnpt.ArrayLike,
    D_local: jnpt.ArrayLike,
    weights_local: jnpt.ArrayLike,
    energies_local: jnpt.ArrayLike,
) -> tuple[npt.NDArray, npt.NDArray, float, npt.NDArray]:
    """Return one host-small LM pencil while O/D contractions stay on devices."""
    O, D, weights, energies = shard_sample_rows(O_local, D_local, weights_local, energies_local)
    H, S, energy, conditioning_metric = _get_device_lm_pencil_kernel()(O, D, weights, energies)
    H.block_until_ready()
    return np.asarray(H), np.asarray(S), float(energy), np.asarray(conditioning_metric)


def _get_device_tall_real_direct_sharded_kernel() -> DeviceDirectKernel:
    cached = getattr(_get_device_tall_real_direct_sharded_kernel, "_cached", None)
    if cached is not None:
        return cached
    from jax.sharding import PartitionSpec as PSpec

    @partial(
        jax.shard_map,
        mesh=_get_sr_mesh(),
        in_specs=(PSpec(None, "rank"), PSpec("rank"), PSpec()),
        out_specs=PSpec(),
        check_vma=False,
    )
    def solve(X: jax.Array, F: jax.Array, epsilon: jax.Array) -> jax.Array:
        Z_local, h_local = _realify_tall_samples(X, F)
        Z = jax.lax.all_gather(Z_local.T, "rank", tiled=True).T
        h = jax.lax.all_gather(h_local, "rank", tiled=True)
        operator = Z.T @ Z + epsilon * jnp.eye(h.size, dtype=Z.dtype)
        return Z @ _device_refined_direct_solve(operator, h, lambda dual: Z.T @ (Z @ dual) + epsilon * dual)

    solve = jax.jit(solve)
    _get_device_tall_real_direct_sharded_kernel._cached = solve
    return solve


def solve_device_tall_real_direct_sharded(
    X: jax.Array,
    F: jax.Array,
    epsilon: jnpt.ArrayLike,
) -> npt.NDArray:
    dtype = jnp.real(X).dtype
    q = _get_device_tall_real_direct_sharded_kernel()(X, F, jnp.asarray(epsilon, dtype=dtype))
    q.block_until_ready()
    return np.asarray(q)


def _cg_while_loop(
    b: jax.Array,
    apply_A: ArrayOperator,
    x0: jax.Array,
    max_iter: jax.Array,
    tol: jax.Array,
    dtype: npt.DTypeLike,
) -> DeviceSolveResult:
    """Plain CG with break-on-breakdown, packed into ``jax.lax.while_loop``.

    Returns ``(x, sqrt(rs), num_iter, breakdown)`` matching the NumPy and
    Rust references: ``breakdown`` is ``True`` when the loop stopped on a
    non-finite or ``|p^H A p| <= tiny`` denominator, and that broken-down
    iteration is not counted in ``num_iter``. The CG vectors are replicated:
    every inner product is taken locally. It runs in the sampler-local jitted
    kernels, where ``apply_A`` is purely local, and inside a ``shard_map``
    whose ``apply_A`` closes over a ``psum`` that returns a replicated vector
    (the wide coordinate-space kernels). Vectors sharded along a mesh axis
    need :func:`_cg_while_loop_sharded` instead.

    Complex-Hermitian safe:
    the CG state vectors follow ``dtype`` (complex128 at a twist) while the
    scalars ``rs`` / ``denom`` use ``vdot`` (conjugate on the first factor)
    + ``real`` so they stay REAL and the norm-based stopping criterion is
    unchanged. ``vdot == dot`` on the real Gamma path (bit-identical).

    ``tol`` is relative (issue #859): the loop stops once
    ``||r||^2 <= (tol*tol) * ||b||^2``, the rule of the NumPy and Rust host
    CGs.  ``b = 0`` starts from its exact solution ``0`` (zero residual, no
    iteration) whatever ``x0``.  The returned residual stays the absolute
    ``||r||``.
    """
    real_dtype = jnp.finfo(dtype).dtype
    tiny = jnp.asarray(get_optimizer_breakdown_threshold(dtype), dtype=real_dtype)
    rhs_sq = jnp.real(jnp.vdot(b, b))
    tol_sq = (tol * tol) * rhs_sq
    x0 = jnp.where(rhs_sq > 0, x0, jnp.zeros_like(x0))

    r0 = b - apply_A(x0)
    rs0 = jnp.real(jnp.vdot(r0, r0))
    state0 = (
        x0,
        r0,
        r0,  # p
        rs0,
        jnp.int32(0),
        jnp.bool_(False),  # breakdown
    )

    def cond(state: DeviceCGState) -> jax.Array:
        _x, _r, _p, rs, k, breakdown = state
        return (k < max_iter) & (rs > tol_sq) & jnp.logical_not(breakdown)

    def body(state: DeviceCGState) -> DeviceCGState:
        x, r, p, rs_old, k, breakdown = state
        Ap = apply_A(p)
        denom = jnp.real(jnp.vdot(p, Ap))  # real for Hermitian A
        new_breakdown = breakdown | jnp.logical_not(jnp.isfinite(denom)) | (jnp.abs(denom) <= tiny)
        safe_denom = jnp.where(new_breakdown, jnp.asarray(1.0, dtype=real_dtype), denom)
        alpha = rs_old / safe_denom
        x_new = jnp.where(new_breakdown, x, x + alpha * p)
        r_new = jnp.where(new_breakdown, r, r - alpha * Ap)
        rs_new_real = jnp.real(jnp.vdot(r_new, r_new))
        rs_new = jnp.where(new_breakdown, rs_old, rs_new_real)
        safe_rs_old = jnp.where(rs_old > 0, rs_old, jnp.asarray(1.0, dtype=real_dtype))
        beta = rs_new / safe_rs_old
        p_new = jnp.where(new_breakdown, p, r_new + beta * p)
        # The broken-down iteration takes no step and is not counted (the
        # NumPy and Rust convention).
        k_new = jnp.where(new_breakdown, k, k + 1)
        return (x_new, r_new, p_new, rs_new, k_new, new_breakdown)

    x_f, _r_f, _p_f, rs_f, k_f, breakdown_f = jax.lax.while_loop(cond, body, state0)
    return x_f, jnp.sqrt(rs_f), k_f, breakdown_f


def _cg_while_loop_sharded(
    b: jax.Array,
    apply_A: ArrayOperator,
    x0: jax.Array,
    max_iter: jax.Array,
    tol: jax.Array,
    dtype: npt.DTypeLike,
    axis_name: str,
) -> DeviceSolveResult:
    """``_cg_while_loop`` variant for vectors sharded along ``axis_name``.

    All vectors (``b``, ``x0`` and the output of ``apply_A``) live in
    the per-rank slice ``(N_local,)``; inner products are summed
    globally with ``jax.lax.psum`` so the loop's convergence check and
    step-size scalars match the replicated (un-sharded) implementation
    bit-for-bit when only the round-off order changes.

    This helper is what makes the "distributed" CG kernel below feasible
    without materialising the full design matrix on every rank.

    Complex-Hermitian safe: same ``vdot`` + ``real`` scalar convention as
    :meth:`_cg_while_loop` (bit-identical on the real Gamma path).

    Relative stopping rule of :meth:`_cg_while_loop` with the GLOBAL
    ``||b||`` of the sharded right-hand side (issue #859).
    """
    real_dtype = jnp.finfo(dtype).dtype
    tiny = jnp.asarray(get_optimizer_breakdown_threshold(dtype), dtype=real_dtype)

    def gdot(a: jax.Array, c: jax.Array) -> jax.Array:
        # Global REAL inner product across the sharded axis (conjugate on
        # the first factor; real for the Hermitian CG scalars).
        return jax.lax.psum(jnp.real(jnp.vdot(a, c)), axis_name)

    rhs_sq = gdot(b, b)
    tol_sq = (tol * tol) * rhs_sq
    x0 = jnp.where(rhs_sq > 0, x0, jnp.zeros_like(x0))

    r0 = b - apply_A(x0)
    rs0 = gdot(r0, r0)
    state0 = (
        x0,
        r0,
        r0,  # p
        rs0,
        jnp.int32(0),
        jnp.bool_(False),  # breakdown
    )

    def cond(state: DeviceCGState) -> jax.Array:
        _x, _r, _p, rs, k, breakdown = state
        return (k < max_iter) & (rs > tol_sq) & jnp.logical_not(breakdown)

    def body(state: DeviceCGState) -> DeviceCGState:
        x, r, p, rs_old, k, breakdown = state
        Ap = apply_A(p)
        denom = gdot(p, Ap)
        new_breakdown = breakdown | jnp.logical_not(jnp.isfinite(denom)) | (jnp.abs(denom) <= tiny)
        safe_denom = jnp.where(new_breakdown, jnp.asarray(1.0, dtype=real_dtype), denom)
        alpha = rs_old / safe_denom
        x_new = jnp.where(new_breakdown, x, x + alpha * p)
        r_new = jnp.where(new_breakdown, r, r - alpha * Ap)
        rs_new_real = gdot(r_new, r_new)
        rs_new = jnp.where(new_breakdown, rs_old, rs_new_real)
        safe_rs_old = jnp.where(rs_old > 0, rs_old, jnp.asarray(1.0, dtype=real_dtype))
        beta = rs_new / safe_rs_old
        p_new = jnp.where(new_breakdown, p, r_new + beta * p)
        # The broken-down iteration takes no step and is not counted (the
        # NumPy and Rust convention).
        k_new = jnp.where(new_breakdown, k, k + 1)
        return (x_new, r_new, p_new, rs_new, k_new, new_breakdown)

    x_f, _r_f, _p_f, rs_f, k_f, breakdown_f = jax.lax.while_loop(cond, body, state0)
    return x_f, jnp.sqrt(rs_f), k_f, breakdown_f


def _validated_wide_real_inputs(
    X_preconditioned_local: npt.ArrayLike,
    b_preconditioned: npt.ArrayLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Shape and dtype checks of a wide real SR system.

    The finiteness of the sample matrix is not re-scanned here: the SR
    preparation established it once for every rank, from the globally reduced
    metric diagonal (a finite diagonal entry bounds every entry of its row;
    see :func:`jqmc.jqmc_optimizer.prepare_wide_sr_real_system`). Only the
    replicated right-hand side is checked.
    """
    X = np.asarray(X_preconditioned_local)
    b = np.asarray(b_preconditioned)
    if X.ndim != 2 or b.shape != (X.shape[0],):
        raise ValueError("wide real SR requires X=(coordinate, sample) and a matching force vector.")
    if np.iscomplexobj(b):
        raise ValueError("wide real SR coordinate force must be real.")
    if not np.all(np.isfinite(b)):
        raise ValueError("wide real SR inputs must be finite.")
    return X, b


def solve_wide_sr_real_direct(
    X_preconditioned_local: npt.ArrayLike,
    b_preconditioned: npt.ArrayLike,
    epsilon: float,
    *,
    reduce_sum: ReduceSum | None = None,
) -> npt.NDArray:
    """Solve the wide real-coordinate SR system on the CPU."""
    X, b = _validated_wide_real_inputs(X_preconditioned_local, b_preconditioned)
    native = _host_rust_kernel("optimizer_wide_sr_real_direct")
    if native is not None:
        stored, transposed, complex_input = _rust_row_major(X)

        def reduce_metric(metric_local: npt.NDArray) -> npt.NDArray:
            metric = np.asarray(reduce_sum(metric_local))
            if metric.shape != (b.size, b.size) or np.iscomplexobj(metric):
                raise ValueError("wide real SR reduction must return a matching real metric.")
            return np.ascontiguousarray(metric, dtype=np.float64)

        theta = native(
            stored,
            transposed,
            complex_input,
            np.ascontiguousarray(b, dtype=np.float64),
            float(epsilon),
            _SR_DIRECT_REFINEMENT_STEPS,
            None if reduce_sum is None else reduce_metric,
            None if reduce_sum is None else _rust_vector_reduction(reduce_sum),
        )
        return np.asarray(theta, dtype=np.result_type(X.real.dtype, b.dtype))
    Xc = _conjugate_if_complex(X).T
    metric_local = np.real(X @ Xc)
    metric = metric_local if reduce_sum is None else np.asarray(reduce_sum(metric_local))
    if metric.shape != (b.size, b.size) or np.iscomplexobj(metric):
        raise ValueError("wide real SR reduction must return a matching real metric.")
    operator = np.array(metric, copy=True)
    operator[np.diag_indices_from(operator)] += epsilon
    factor = scipy.linalg.lu_factor(operator, check_finite=False)

    def apply_operator(vector: npt.NDArray) -> npt.NDArray:
        applied_local = np.real(X @ (Xc @ vector))
        applied = applied_local if reduce_sum is None else np.asarray(reduce_sum(applied_local))
        return applied + epsilon * vector

    theta = scipy.linalg.lu_solve(factor, b, check_finite=False)
    theta = _refine_host_direct_solve(factor, b, theta, apply_operator)
    return np.asarray(theta)


def _relative_cg_tolerance_sq(rhs: npt.NDArray, tol: float) -> tuple[float, bool]:
    """Squared stopping threshold of the relative SR CG rule (issue #859).

    Every SR CG (NumPy, Rust and XLA) stops once ``||r||^2 <= (tol*tol) * ||b||^2``
    for the right-hand side ``b`` of the system it solves.  The second value
    is ``True`` for ``b = 0``: the exact solution is then ``0`` and the solve
    starts there, so its residual is exactly zero and no iteration runs.
    """
    rhs_sq = float(np.dot(rhs, rhs))
    return (float(tol) * float(tol)) * rhs_sq, rhs_sq == 0.0


def solve_wide_sr_real_cg(
    X_preconditioned_local: npt.ArrayLike,
    b_preconditioned: npt.ArrayLike,
    epsilon: float,
    *,
    max_iter: int,
    tol: float,
    x0: npt.ArrayLike,
    reduce_sum: ReduceSum | None = None,
) -> HostSolveResult:
    """Matrix-free CPU CG for the wide real-coordinate SR system.

    The operator is ``Re(X X^dagger) + epsilon I`` acting on real vectors.
    With ``X = Xr + i Xi``, ``Re(X X^dagger) v = Xr (Xr^T v) + Xi (Xi^T v) =
    Y (Y^T v)`` for the real ``Y = [Xr | Xi]`` (``Y = Xr`` when ``Xi`` is
    identically zero, e.g. for every real wavefunction). How one application
    reads ``Y`` depends on the route and the storage of ``X``:

    * host Rust (the CPU default): a complex ``X`` is realified once into
      owned sample-major rows of ``Y``, and a real sample-major (Fortran)
      ``X`` is used in place; either way each application is ONE fused pass
      over ``Y`` (one row per sample: ``sum_n y_n (y_n . v)``). A real
      coordinate-major ``X`` is used in place with two passes per
      application, ``Y^T v`` then ``Y (Y^T v)``. The production wide
      preparation (:func:`jqmc.jqmc_optimizer.prepare_wide_sr_real_system`)
      publishes the sample-major layout for every wide system it prepares;
    * NumPy (``JQMC_FLAG_RUST_ON_CPU=0``): ``Y`` is formed once as a
      C-contiguous matrix and each application runs the two products
      ``Y^T v`` and ``Y (Y^T v)`` into preallocated buffers.

    ``tol`` is relative: the iteration stops once the recurrence
    residual satisfies ``||r|| <= tol * ||b||``.  ``b`` is the
    coordinate-replicated right-hand side, so ``||b||`` is the global norm on
    every rank.  ``b = 0`` returns its exact solution ``0`` after zero
    iterations, whatever ``x0``.  The returned residual is the absolute
    ``||r||`` of the returned iterate's recurrence, not ``||r|| / ||b||``.
    The last value is ``True`` when the loop stopped on a breakdown
    (non-finite or ``|p^T A p| <= tiny`` denominator); the broken-down
    iteration is not counted.
    """
    X, b = _validated_wide_real_inputs(X_preconditioned_local, b_preconditioned)
    initial = np.asarray(x0)
    if np.iscomplexobj(initial) or initial.shape != b.shape:
        raise ValueError("wide real SR CG carry must be a matching real vector.")
    real_dtype = np.result_type(X.real.dtype, b.dtype, initial.dtype)
    native = _host_rust_kernel("optimizer_wide_sr_real_cg")
    if native is not None:
        stored, transposed, complex_input = _rust_row_major(X)
        theta, residual_norm, num_iterations, breakdown = native(
            stored,
            transposed,
            complex_input,
            np.ascontiguousarray(b, dtype=np.float64),
            float(epsilon),
            int(max_iter),
            float(tol),
            float(get_optimizer_breakdown_threshold(real_dtype)),
            np.ascontiguousarray(initial, dtype=np.float64),
            None if reduce_sum is None else _rust_vector_reduction(reduce_sum),
        )
        return np.asarray(theta, dtype=real_dtype), float(residual_norm), int(num_iterations), bool(breakdown)
    b = np.asarray(b, dtype=real_dtype)
    x = np.asarray(initial, dtype=real_dtype).copy()

    if np.iscomplexobj(X) and np.any(X.imag):
        Y = np.ascontiguousarray(np.concatenate([X.real, X.imag], axis=1), dtype=real_dtype)
    else:
        Y = np.ascontiguousarray(X.real, dtype=real_dtype)
    sample_buffer = np.empty(Y.shape[1], dtype=real_dtype)
    coordinate_buffer = np.empty(Y.shape[0], dtype=real_dtype)

    def apply_operator(vector: npt.NDArray) -> npt.NDArray:
        np.dot(Y.T, vector, out=sample_buffer)
        np.dot(Y, sample_buffer, out=coordinate_buffer)
        global_values = coordinate_buffer if reduce_sum is None else np.asarray(reduce_sum(coordinate_buffer))
        return np.asarray(global_values, dtype=real_dtype) + epsilon * vector

    tolerance_sq, start_from_zero = _relative_cg_tolerance_sq(b, tol)
    if start_from_zero:
        x = np.zeros_like(x)
    residual = b - apply_operator(x)
    direction = residual.copy()
    residual_sq = float(np.dot(residual, residual))
    tiny = get_optimizer_breakdown_threshold(real_dtype)
    iterations = 0
    breakdown = False
    while iterations < int(max_iter) and residual_sq > tolerance_sq:
        applied = apply_operator(direction)
        denominator = float(np.dot(direction, applied))
        if not np.isfinite(denominator) or abs(denominator) <= tiny:
            breakdown = True
            break
        alpha = residual_sq / denominator
        x = x + alpha * direction
        residual = residual - alpha * applied
        next_residual_sq = float(np.dot(residual, residual))
        beta = next_residual_sq / residual_sq if residual_sq > 0 else 0.0
        direction = residual + beta * direction
        residual_sq = next_residual_sq
        iterations += 1
    return x, float(np.sqrt(residual_sq)), iterations, breakdown


def _validated_tall_real_inputs(
    Z_local: npt.ArrayLike,
    h_local: npt.ArrayLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Shape and dtype checks of a tall real SR system.

    As for the wide system, the finiteness of ``Z`` was established once by
    the SR preparation from the globally reduced metric diagonal; only the
    dual right-hand side ``h`` is checked here.
    """
    Z = np.asarray(Z_local)
    h = np.asarray(h_local)
    if Z.ndim != 2 or h.ndim != 1 or Z.shape[1] != h.size:
        raise ValueError("tall real SR requires Z=(coordinate, dual) and matching h.")
    if np.iscomplexobj(Z) or np.iscomplexobj(h):
        raise ValueError("tall real SR inputs and dual carry must be real.")
    if not np.all(np.isfinite(h)):
        raise ValueError("tall real SR inputs must be finite.")
    return Z, h


def solve_tall_sr_real_direct(
    Z_local: npt.ArrayLike,
    h_local: npt.ArrayLike,
    epsilon: float,
) -> npt.NDArray:
    """Solve the tall real sample-space system redundantly on the CPU."""
    Z, h = _validated_tall_real_inputs(Z_local, h_local)
    native = _host_rust_kernel("optimizer_tall_sr_real_direct")
    if native is not None:
        stored, transposed, _complex = _rust_row_major(Z)
        q = native(stored, transposed, np.ascontiguousarray(h, dtype=np.float64), float(epsilon), _SR_DIRECT_REFINEMENT_STEPS)
        return np.asarray(q, dtype=np.result_type(Z.dtype, h.dtype))
    dual_operator = Z.T @ Z
    dual_operator[np.diag_indices_from(dual_operator)] += epsilon
    factor = scipy.linalg.lu_factor(dual_operator, check_finite=False)
    dual = scipy.linalg.lu_solve(factor, h, check_finite=False)
    dual = _refine_host_direct_solve(factor, h, dual, lambda vector: Z.T @ (Z @ vector) + epsilon * vector)
    return np.asarray(Z @ dual)


def solve_tall_sr_real_cg(
    Z_local: npt.ArrayLike,
    h_local: npt.ArrayLike,
    epsilon: float,
    *,
    max_iter: int,
    tol: float,
) -> HostSolveResult:
    """Solve tall SR from a batch-scoped, always-zero real dual.

    ``tol`` is relative to the dual right-hand side: the dual CG
    stops once ``||r|| <= tol * ||h||``; ``h = 0`` returns ``0`` after zero
    iterations.  The returned residual is the absolute dual ``||r||``; the
    last value is the breakdown flag of :func:`solve_wide_sr_real_cg`.
    """
    Z, h = _validated_tall_real_inputs(Z_local, h_local)
    real_dtype = np.result_type(Z.dtype, h.dtype)
    native = _host_rust_kernel("optimizer_tall_sr_real_cg")
    if native is not None:
        stored, transposed, _complex = _rust_row_major(Z)
        q, residual_norm, num_iterations, breakdown = native(
            stored,
            transposed,
            np.ascontiguousarray(h, dtype=np.float64),
            float(epsilon),
            int(max_iter),
            float(tol),
            float(get_optimizer_breakdown_threshold(real_dtype)),
        )
        return np.asarray(q, dtype=real_dtype), float(residual_norm), int(num_iterations), bool(breakdown)
    h = np.asarray(h, dtype=real_dtype)
    dual = np.zeros(h.shape, dtype=real_dtype)

    def apply_operator(vector: npt.NDArray) -> npt.NDArray:
        return Z.T @ (Z @ vector) + epsilon * vector

    tolerance_sq, _start_from_zero = _relative_cg_tolerance_sq(h, tol)
    residual = h - apply_operator(dual)
    direction = residual.copy()
    residual_sq = float(np.dot(residual, residual))
    tiny = get_optimizer_breakdown_threshold(real_dtype)
    iterations = 0
    breakdown = False
    while iterations < int(max_iter) and residual_sq > tolerance_sq:
        applied = apply_operator(direction)
        denominator = float(np.dot(direction, applied))
        if not np.isfinite(denominator) or abs(denominator) <= tiny:
            breakdown = True
            break
        alpha = residual_sq / denominator
        dual = dual + alpha * direction
        residual = residual - alpha * applied
        next_residual_sq = float(np.dot(residual, residual))
        beta = next_residual_sq / residual_sq if residual_sq > 0 else 0.0
        direction = residual + beta * direction
        residual_sq = next_residual_sq
        iterations += 1
    return np.asarray(Z @ dual), float(np.sqrt(residual_sq)), iterations, breakdown


def _get_sr_wide_real_direct_kernel() -> DeviceDirectKernel:
    """Device direct solve for a complex sample matrix and real coordinates."""
    cached = getattr(_get_sr_wide_real_direct_kernel, "_cached", None)
    if cached is not None:
        return cached
    from jax.sharding import PartitionSpec as PSpec

    mesh = _get_sr_mesh()

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(PSpec(None, "rank"), PSpec(), PSpec()),
        out_specs=PSpec(),
    )
    def _solve(X: jax.Array, b: jax.Array, shift: jax.Array) -> jax.Array:
        metric = jax.lax.psum(jnp.real(X @ jnp.conj(X).T), "rank")
        operator = metric + jnp.diag(shift)
        Xc = jnp.conj(X).T
        return _device_refined_direct_solve(
            operator,
            b,
            lambda vector: jax.lax.psum(jnp.real(X @ (Xc @ vector)), "rank") + shift * vector,
        )

    _solve = jax.jit(_solve)
    _get_sr_wide_real_direct_kernel._cached = _solve
    return _solve


def _shard_X(X_local: npt.NDArray) -> jax.Array:
    """Shard a host-local coordinate-by-sample matrix along its sample axis."""
    X_jnp = jnp.asarray(X_local)
    if jax.process_count() <= 1:
        return X_jnp

    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as PSpec

    mesh = _get_sr_mesh()
    return jax.make_array_from_process_local_data(NamedSharding(mesh, PSpec(None, "rank")), X_jnp)


def _shard_X_F(X_local: npt.NDArray, F_local: npt.NDArray) -> tuple[jax.Array, jax.Array]:
    """Convert host-local NumPy ``(X, F)`` into shard_map-ready ``jax.Array`` s.

    ``X_local`` is sharded along axis 1 (samples), ``F_local`` along axis 0.
    Single-process: returns plain ``jnp.array`` (1-rank mesh = identity).
    Multi-process: stitches host-local slices into a global ``jax.Array``.
    """
    X_jnp = jnp.asarray(X_local)
    F_jnp = jnp.asarray(F_local)
    if jax.process_count() <= 1:
        return X_jnp, F_jnp

    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as PSpec

    mesh = _get_sr_mesh()
    X_global = jax.make_array_from_process_local_data(NamedSharding(mesh, PSpec(None, "rank")), X_jnp)
    F_global = jax.make_array_from_process_local_data(NamedSharding(mesh, PSpec("rank")), F_jnp)
    return X_global, F_global


def _replicated_jax_array(arr: npt.NDArray) -> jax.Array:
    """Wrap a host-local NumPy array as a replicated ``jax.Array`` across ranks.

    For multi-process runs the ``arr`` must be identical on every rank (e.g.
    a CG warm-start state that was psum'd before being persisted).
    """
    jnp_arr = jnp.asarray(arr)
    if jax.process_count() <= 1:
        return jnp_arr

    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as PSpec

    mesh = _get_sr_mesh()
    return jax.make_array_from_process_local_data(NamedSharding(mesh, PSpec()), jnp_arr)
