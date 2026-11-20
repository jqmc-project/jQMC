"""Exact (extended-precision) dense SR references.

The SR system ``(S + epsilon diag S) q = f`` built from sampled log
derivatives has ``cond ~ 1/epsilon`` (about 1e4 at ``epsilon = 1e-3``). A
reference that forms ``S`` and ``f`` in fp64 and solves with one LU therefore
carries its own forward error of about ``cond * u * |q|`` (~1e-12), which is the
same size as the ``*_PYTHON_FP64_LOOSE`` bound it is compared at. It would
spend the comparison bound on its own rounding.

These helpers form the centred moments, the Gram operators and every residual
in extended precision. The fp64 LU factor serves only as the preconditioner of
iterative refinement. The result is the fp64 rounding of the exact solution
for the given fp64 inputs (measured within 4.4e-16 of a 50-digit mpmath
solve), so a comparison against it measures only the solver under test.

Two extended-precision backends are provided:

* ``double-double`` (the default on every platform): unevaluated sums
  ``hi + lo`` of two fp64 values built from error-free transformations
  (TwoSum, Dekker-split TwoProd), about 106 significand bits. Its residuals
  stay accurate far beyond fp64 output precision even at cond ~ 1e5 (it
  matched a 60-digit mpmath push-through solve exactly at cond 1.2e5).
* ``longdouble``: ``np.longdouble``, available only where it is genuinely
  extended (x87 80-bit on x86-64 Linux, IEEE quad on aarch64 Linux). It is
  kept as an independent cross-check. With 64 significand bits it is limited
  to about ``cond * 1e-19`` relative accuracy.

``JQMC_TEST_EXACT_SR_BACKEND`` (``auto`` = ``double-double`` | ``longdouble`` |
``double-double``) selects the backend. It is a test-only switch used to show
that the two agree.
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

import os

import numpy as np
import scipy.linalg

BACKEND_ENV = "JQMC_TEST_EXACT_SR_BACKEND"
_BACKENDS = ("auto", "longdouble", "double-double")
# Unit roundoff required for the optional longdouble backend to be extended
# (x87 extended: 1.08e-19; IEEE quad: 1.9e-34; fp64: 2.2e-16).
_LONGDOUBLE_EPS_MAX = 1.0e-18
_REFINEMENT_STEPS = 6
# Dekker's splitter 2**27 + 1 for fp64 (53-bit significand).
_DEKKER_SPLITTER = 134217729.0


def exact_backend() -> str:
    """Resolve the extended-precision backend: ``double-double`` unless ``longdouble`` is forced."""
    requested = os.environ.get(BACKEND_ENV, "auto")
    if requested not in _BACKENDS:
        raise ValueError(f"{BACKEND_ENV} must be one of {_BACKENDS}, got {requested!r}")
    extended = float(np.finfo(np.longdouble).eps) < _LONGDOUBLE_EPS_MAX
    if requested == "longdouble":
        assert extended, (
            f"{BACKEND_ENV}=longdouble requested, but np.longdouble is fp64 on this platform "
            f"(eps={float(np.finfo(np.longdouble).eps):.3e})"
        )
        return "longdouble"
    return "double-double"


# --------------------------------------------------------------------------- #
# Error-free transformations and the double-double array
# --------------------------------------------------------------------------- #


def two_sum(a, b):
    """Knuth TwoSum: ``s + err == a + b`` exactly, with ``s = fl(a + b)``."""
    s = a + b
    b_virtual = s - a
    err = (a - (s - b_virtual)) + (b - b_virtual)
    return s, err


def _quick_two_sum(a, b):
    """Dekker FastTwoSum for ``|a| >= |b|``."""
    s = a + b
    return s, b - (s - a)


def _split(a):
    scaled = _DEKKER_SPLITTER * a
    hi = scaled - (scaled - a)
    return hi, a - hi


def two_prod(a, b):
    """Dekker TwoProd: ``p + err == a * b`` exactly, with ``p = fl(a * b)`` (no overflow)."""
    p = a * b
    a_hi, a_lo = _split(a)
    b_hi, b_lo = _split(b)
    err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, err


class DoubleDouble:
    """Array of unevaluated sums ``hi + lo`` (|lo| <= ulp(hi)/2), about 106 significand bits.

    Supports the broadcasting arithmetic, indexing, transposition, reshaping and
    matrix products the SR references need. Inputs that are not
    ``DoubleDouble`` are taken as exact fp64 values.
    """

    __array_ufunc__ = None  # make NumPy operands defer to the reflected operators

    def __init__(self, hi, lo=None):
        self.hi = np.asarray(hi, dtype=np.float64)
        self.lo = np.zeros_like(self.hi) if lo is None else np.asarray(lo, dtype=np.float64)

    @staticmethod
    def lift(value) -> "DoubleDouble":
        return value if isinstance(value, DoubleDouble) else DoubleDouble(value)

    @property
    def shape(self):
        return self.hi.shape

    @property
    def T(self) -> "DoubleDouble":
        return DoubleDouble(self.hi.T, self.lo.T)

    def reshape(self, *shape) -> "DoubleDouble":
        return DoubleDouble(self.hi.reshape(*shape), self.lo.reshape(*shape))

    def __getitem__(self, key) -> "DoubleDouble":
        return DoubleDouble(self.hi[key], self.lo[key])

    def to_float64(self) -> np.ndarray:
        return self.hi + self.lo

    def __neg__(self) -> "DoubleDouble":
        return DoubleDouble(-self.hi, -self.lo)

    def __add__(self, other) -> "DoubleDouble":
        other = DoubleDouble.lift(other)
        s1, s2 = two_sum(self.hi, other.hi)
        t1, t2 = two_sum(self.lo, other.lo)
        s1, s2 = _quick_two_sum(s1, s2 + t1)
        hi, lo = _quick_two_sum(s1, s2 + t2)
        return DoubleDouble(hi, lo)

    __radd__ = __add__

    def __sub__(self, other) -> "DoubleDouble":
        return self + (-DoubleDouble.lift(other))

    def __rsub__(self, other) -> "DoubleDouble":
        return DoubleDouble.lift(other) + (-self)

    def __mul__(self, other) -> "DoubleDouble":
        other = DoubleDouble.lift(other)
        p, err = two_prod(self.hi, other.hi)
        hi, lo = _quick_two_sum(p, err + (self.hi * other.lo + self.lo * other.hi))
        return DoubleDouble(hi, lo)

    __rmul__ = __mul__

    def __truediv__(self, other) -> "DoubleDouble":
        other = DoubleDouble.lift(other)
        q1 = self.hi / other.hi
        remainder = self - other * q1
        q2 = remainder.hi / other.hi
        remainder = remainder - other * q2
        q3 = remainder.hi / other.hi
        q1, q2 = _quick_two_sum(q1, q2)
        return DoubleDouble(q1, q2) + q3

    def __rtruediv__(self, other) -> "DoubleDouble":
        return DoubleDouble.lift(other) / self

    def __matmul__(self, other) -> "DoubleDouble":
        other = DoubleDouble.lift(other)
        vector = other.hi.ndim == 1
        right = other.reshape(-1, 1) if vector else other
        if self.hi.ndim != 2 or right.hi.ndim != 2 or self.shape[1] != right.shape[0]:
            raise ValueError(f"double-double matmul shape mismatch {self.shape} @ {other.shape}")
        total = DoubleDouble(np.zeros((self.shape[0], right.shape[1])))
        for k in range(self.shape[1]):
            total = total + self[:, k : k + 1] * right[k : k + 1, :]
        return total.reshape(-1) if vector else total

    def __rmatmul__(self, other) -> "DoubleDouble":
        return DoubleDouble.lift(other) @ self


# --------------------------------------------------------------------------- #
# Backend-neutral extended-precision helpers
# --------------------------------------------------------------------------- #


def _lift(value, backend: str):
    if backend == "double-double":
        return DoubleDouble.lift(value)
    return np.asarray(value, dtype=np.longdouble)


def to_float64(value) -> np.ndarray:
    """fp64 rounding of an extended-precision array."""
    if isinstance(value, DoubleDouble):
        return value.to_float64()
    return np.asarray(value, dtype=np.float64)


def _backend_of(value) -> str:
    return "double-double" if isinstance(value, DoubleDouble) else "longdouble"


def ext_diag(matrix):
    """Diagonal of an extended-precision square matrix."""
    if isinstance(matrix, DoubleDouble):
        return DoubleDouble(np.diag(matrix.hi), np.diag(matrix.lo))
    return np.diag(matrix)


def _ext_diag_matrix(vector):
    if isinstance(vector, DoubleDouble):
        return DoubleDouble(np.diag(vector.hi), np.diag(vector.lo))
    return np.diag(vector)


def _ext_block(rows):
    if any(isinstance(item, DoubleDouble) for row in rows for item in row):
        lifted = [[DoubleDouble.lift(item) for item in row] for row in rows]
        return DoubleDouble(
            np.block([[item.hi for item in row] for row in lifted]),
            np.block([[item.lo for item in row] for row in lifted]),
        )
    return np.block(rows)


def _ext_concat(parts):
    if any(isinstance(item, DoubleDouble) for item in parts):
        lifted = [DoubleDouble.lift(item) for item in parts]
        return DoubleDouble(np.concatenate([item.hi for item in lifted]), np.concatenate([item.lo for item in lifted]))
    return np.concatenate(parts)


def exact_centred_moments(log_derivatives, local_energies, weights):
    """Extended-precision ``S = Re<dO* dO>`` and ``f = -2 Re<dO* (e - Re<e>)>``.

    Real columns and energies reduce to the real SR formulas. Returns
    extended-precision arrays of the active backend.
    """
    backend = exact_backend()
    O = np.asarray(log_derivatives, dtype=np.complex128)
    e = np.asarray(local_energies, dtype=np.complex128)
    w64 = np.asarray(weights, dtype=np.float64)
    N = w64.size
    w_row = _lift(w64.reshape(1, N), backend)
    w_col = _lift(w64.reshape(N, 1), backend)
    total = w_row @ _lift(np.ones((N, 1)), backend)
    O_re, O_im = _lift(O.real, backend), _lift(O.imag, backend)
    e_re, e_im = _lift(e.real.reshape(N, 1), backend), _lift(e.imag.reshape(N, 1), backend)
    dO_re = O_re - (w_row @ O_re) / total
    dO_im = O_im - (w_row @ O_im) / total
    de_re = e_re - (w_row @ e_re) / total
    S = (dO_re.T @ (w_col * dO_re) + dO_im.T @ (w_col * dO_im)) / total
    f = -2.0 * (dO_re.T @ (w_col * de_re) + dO_im.T @ (w_col * e_im)) / total
    return S, f.reshape(-1)


def exact_solve(operator, rhs):
    """Solve the real system ``operator x = rhs`` exactly for extended-precision inputs.

    fp64 LU plus iterative refinement with an extended-precision residual.
    Returns the extended-precision solution in the operator's backend.
    """
    backend = _backend_of(operator)
    A = _lift(operator, backend)
    b = _lift(rhs, backend)
    factor = scipy.linalg.lu_factor(to_float64(A))
    x = _lift(scipy.linalg.lu_solve(factor, to_float64(b)), backend)
    for _ in range(_REFINEMENT_STEPS):
        x = x + _lift(scipy.linalg.lu_solve(factor, to_float64(b - A @ x)), backend)
    return x


def exact_sr_solve(S, f, epsilon) -> np.ndarray:
    """Exact fp64 rounding of ``(S + epsilon diag S)^{-1} f`` for extended-precision moments."""
    operator = S + epsilon * _ext_diag_matrix(ext_diag(S))
    return to_float64(exact_solve(operator, f))


def _identity(size, backend):
    return _lift(np.eye(size), backend)


def exact_real_wide_gram_solve(X, b, epsilon) -> np.ndarray:
    """Exact ``(Re(X X^dagger) + epsilon I)^{-1} b`` from the fp64 design matrix ``X``."""
    backend = exact_backend()
    X = np.asarray(X, dtype=np.complex128)
    X_re, X_im = _lift(X.real, backend), _lift(X.imag, backend)
    operator = X_re @ X_re.T + X_im @ X_im.T + epsilon * _identity(X.shape[0], backend)
    return to_float64(exact_solve(operator, _lift(np.asarray(b, dtype=np.float64), backend)))


def exact_real_tall_gram_solve(Z, h, epsilon) -> np.ndarray:
    """Exact push-through ``Z (Z^T Z + epsilon I)^{-1} h`` from the fp64 real dual matrix ``Z``."""
    backend = exact_backend()
    Z_ext = _lift(np.asarray(Z, dtype=np.float64), backend)
    operator = Z_ext.T @ Z_ext + epsilon * _identity(Z_ext.shape[1], backend)
    dual = exact_solve(operator, _lift(np.asarray(h, dtype=np.float64), backend))
    return to_float64(Z_ext @ dual)


def exact_regularized_gram_solve(X, F, epsilon) -> np.ndarray:
    """Exact complex ``(X X^dagger + epsilon I)^{-1} X F`` from the fp64 design matrix ``X``.

    The Hermitian system is solved through its real embedding
    ``[[A_re, -A_im], [A_im, A_re]] [x_re; x_im] = [b_re; b_im]``.
    """
    backend = exact_backend()
    X = np.asarray(X, dtype=np.complex128)
    F = np.asarray(F, dtype=np.complex128)
    X_re, X_im = _lift(X.real, backend), _lift(X.imag, backend)
    F_re, F_im = _lift(F.real, backend), _lift(F.imag, backend)
    size = X.shape[0]
    A_re = X_re @ X_re.T + X_im @ X_im.T + epsilon * _identity(size, backend)
    A_im = X_im @ X_re.T - X_re @ X_im.T
    b_re = X_re @ F_re - X_im @ F_im
    b_im = X_re @ F_im + X_im @ F_re
    solution = to_float64(exact_solve(_ext_block([[A_re, -A_im], [A_im, A_re]]), _ext_concat([b_re, b_im])))
    return solution[:size] + 1j * solution[size:]
