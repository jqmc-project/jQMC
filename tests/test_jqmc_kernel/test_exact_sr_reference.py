"""The double-double kernel of the exact SR reference against exact rational arithmetic.

``_exact_sr_reference`` builds the dense SR references from error-free fp64
transformations (TwoSum, Dekker TwoProd) and double-double (``hi + lo``)
arithmetic. On random inputs, each check evaluates the same expression in
``fractions.Fraction`` (exact rationals of the fp64 inputs) and requires:

* TwoSum and TwoProd to be error free, so ``s + err`` equals the exact sum or
  product;
* the double-double sum, difference, product, quotient and matrix product to
  round to the correctly rounded fp64 value of the exact result;
* the centred SR moments and the regularized SR solve to round to the
  correctly rounded fp64 values of the exact moments and the exact solution.

A non-vacuity control checks that plain fp64 moments of the same inputs miss
the correctly rounded values.
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

from fractions import Fraction

import numpy as np
import pytest

from tests.test_jqmc_kernel import _exact_sr_reference as exact_reference
from tests.test_jqmc_kernel._exact_sr_reference import DoubleDouble, two_prod, two_sum

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_SEED = 20261111
_EPSILON = 1.0e-3


def _scaled_normals(rng, size):
    """Random fp64 values spread over many binades."""
    return rng.standard_normal(size) * np.exp2(rng.integers(-30, 30, size))


def _random_double_double(rng, shape):
    hi, lo = two_sum(rng.standard_normal(shape), rng.standard_normal(shape) * 1.0e-17)
    return DoubleDouble(hi, lo)


def _fraction_of(value: DoubleDouble):
    return np.vectorize(lambda hi, lo: Fraction(float(hi)) + Fraction(float(lo)), otypes=[object])(value.hi, value.lo)


def _rounded(fractions_array):
    return np.vectorize(float, otypes=[np.float64])(fractions_array)


def test_two_sum_and_two_prod_are_error_free():
    """TwoSum and Dekker TwoProd return the exact sum and product as an unevaluated pair."""
    rng = np.random.default_rng(_SEED)
    a, b = _scaled_normals(rng, 400), _scaled_normals(rng, 400)
    s, s_err = two_sum(a, b)
    p, p_err = two_prod(a, b)
    exact_sums = [Fraction(x) + Fraction(y) for x, y in zip(a, b)]
    exact_products = [Fraction(x) * Fraction(y) for x, y in zip(a, b)]
    assert [Fraction(x) + Fraction(y) for x, y in zip(s, s_err)] == exact_sums
    assert [Fraction(x) + Fraction(y) for x, y in zip(p, p_err)] == exact_products


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div", "matmul"])
def test_double_double_arithmetic_rounds_to_the_exact_value(operation):
    """Each double-double operation rounds to the correctly rounded fp64 value of the exact result."""
    rng = np.random.default_rng(_SEED)
    x = _random_double_double(rng, (6, 5))
    y = _random_double_double(rng, (6, 5)) if operation != "matmul" else _random_double_double(rng, (5, 4))
    x_exact, y_exact = _fraction_of(x), _fraction_of(y)
    if operation == "add":
        actual, exact = x + y, x_exact + y_exact
    elif operation == "sub":
        actual, exact = x - y, x_exact - y_exact
    elif operation == "mul":
        actual, exact = x * y, x_exact * y_exact
    elif operation == "div":
        actual, exact = x / y, x_exact / y_exact
    else:
        actual, exact = x @ y, np.dot(x_exact, y_exact)
    np.testing.assert_array_equal(actual.to_float64(), _rounded(exact))


def _exact_moments(O, e, w):
    O_re = np.vectorize(Fraction, otypes=[object])(O.real)
    O_im = np.vectorize(Fraction, otypes=[object])(O.imag)
    e_re = np.vectorize(Fraction, otypes=[object])(e.real)
    e_im = np.vectorize(Fraction, otypes=[object])(e.imag)
    w_exact = np.vectorize(Fraction, otypes=[object])(w)
    total = sum(w_exact)
    dO_re = O_re - np.dot(w_exact, O_re) / total
    dO_im = O_im - np.dot(w_exact, O_im) / total
    de_re = e_re - np.dot(w_exact, e_re) / total
    weighted_re = dO_re * w_exact[:, None]
    weighted_im = dO_im * w_exact[:, None]
    S = (np.dot(dO_re.T, weighted_re) + np.dot(dO_im.T, weighted_im)) / total
    f = -2 * (np.dot(weighted_re.T, de_re) + np.dot(weighted_im.T, e_im)) / total
    return S, f


def _exact_linear_solve(A, b):
    """Gauss-Jordan elimination in exact rationals."""
    size = len(b)
    augmented = [list(A[i]) + [b[i]] for i in range(size)]
    for column in range(size):
        pivot = next(row for row in range(column, size) if augmented[row][column] != 0)
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(size):
            if row != column:
                factor = augmented[row][column] / augmented[column][column]
                augmented[row] = [left - factor * right for left, right in zip(augmented[row], augmented[column])]
    return np.array([augmented[i][size] / augmented[i][i] for i in range(size)], dtype=object)


def test_double_double_sr_moments_and_solve_round_to_the_exact_values(monkeypatch):
    """Double-double SR moments and the regularized solve round to the exact rational results."""
    monkeypatch.setenv(exact_reference.BACKEND_ENV, "double-double")
    rng = np.random.default_rng(_SEED)
    samples, coordinates = 9, 4
    # A large common offset makes the centring cancel, so plain fp64 moments lose digits.
    O = 1.0e3 + rng.standard_normal((samples, coordinates)) + 1j * (1.0e3 + rng.standard_normal((samples, coordinates)))
    e = 1.0e2 + rng.standard_normal(samples) + 1j * rng.standard_normal(samples)
    w = rng.uniform(0.5, 1.5, samples)

    S, f = exact_reference.exact_centred_moments(O, e, w)
    S_exact, f_exact = _exact_moments(O, e, w)
    np.testing.assert_array_equal(exact_reference.to_float64(S), _rounded(S_exact))
    np.testing.assert_array_equal(exact_reference.to_float64(f), _rounded(f_exact))

    epsilon = Fraction(_EPSILON)
    operator = S_exact + np.diag([epsilon * S_exact[i, i] for i in range(coordinates)])
    q_exact = _exact_linear_solve(operator, f_exact)
    np.testing.assert_array_equal(exact_reference.exact_sr_solve(S, f, _EPSILON), _rounded(q_exact))

    # Non-vacuity: the plain fp64 moments of the same inputs miss the correctly rounded values.
    total = np.sum(w)
    dO = O - (w @ O) / total
    plain_S = np.real(np.einsum("n,na,nb->ab", w, np.conj(dO), dO)) / total
    assert np.any(plain_S != _rounded(S_exact))
