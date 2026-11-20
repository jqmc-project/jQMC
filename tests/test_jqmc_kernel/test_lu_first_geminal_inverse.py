"""Guarded LU seed/refresh inverses of a walker batch (issues #533, #605).

The carried geminal inverse is seeded and re-anchored by ``_geminal_pinv_batched``: every
walker, on every backend, gets the LU inverse of its power-of-two equilibrated geminal, kept when it
is finite and its scale-free probe residual is at most ``GEMINAL_LU_GUARD_RESIDUAL_THRESHOLD``; any
other walker gets the equilibrated SVD pseudo-inverse. The equilibration exponents are clipped to
``GEMINAL_EQUILIBRATION_MAX_EXPONENT`` so the pseudo-inverse of a finite geminal is finite.

1. Far electrons: on the committed all-electron UHF N geminal (a regular walker, a row- and
   column-graded walker and a column-graded walker) every walker keeps its LU inverse, equal to the
   pseudo-inverse under the det_eval strict pair, and the rank-1 ratios of a 0.3 bohr inward move
   of the far electrons equal the fresh determinant ratios under the det_ratio strict pair.
2. Pathological walkers (an exactly zero row or column, a duplicated row or column, a linearly
   dependent row, a rank N-2 product, a near-duplicate row, an all-zero matrix, a subnormal row):
   the walker gets the pseudo-inverse bit for bit, the output is finite and the healthy walkers of
   the same batch keep their LU inverse bit for bit (real and complex geminals).
3. Overflow: a far up and a far down electron (one row and one column of order 1e-200), or a
   subnormal row, give a finite pseudo-inverse with ``G X G = G``.
4. No false fallback: invertible graded geminals with condition numbers up to 1e10 keep the LU
   inverse, and a row of order 1e-200 keeps the exact LU inverse (its inverse column of order
   1e200 is representable; only the pseudo-inverse clips its equilibration).
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import jqmc_mcmc as mcmc
from jqmc._precision import get_tolerance_min
from jqmc.determinant import _equilibrated_geminal_lu_inverse, _equilibrated_geminal_lu_inverse_and_residual
from tests.test_jqmc_kernel.test_sm_inverse_refresh import _exact_ratio, _far_electron_walkers, _sm_ratio

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_N = 16
_HEALTHY = 7

_pinv_batch = jax.jit(jax.vmap(mcmc._geminal_pinv_from_matrix))
_lu_batch = jax.jit(jax.vmap(_equilibrated_geminal_lu_inverse))


def _random(rng, n, complex_):
    g = rng.normal(size=(n, n))
    if complex_:
        g = g + 1j * rng.normal(size=(n, n))
    return g


def _pathological(name, rng, complex_):
    g = _random(rng, _N, complex_)
    if name == "zero_row":
        g[3] = 0.0
    elif name == "zero_column":
        g[:, 5] = 0.0
    elif name == "duplicate_row":
        g[4] = g[2]
    elif name == "duplicate_column":
        g[:, 6] = g[:, 1]
    elif name == "dependent_row":
        g[4] = 0.3 * g[2] - 1.7 * g[7]
    elif name == "rank_n_minus_2":
        g = _random(rng, _N, complex_)[:, : _N - 2] @ _random(rng, _N, complex_)[: _N - 2, :]
    elif name == "near_duplicate_row":
        g[4] = g[2] * (1.0 + 1.0e-13)
    elif name == "all_zero":
        g = np.zeros_like(g)
    elif name == "subnormal_row":
        g[5] *= 1.0e-310
    else:
        raise AssertionError(name)
    return g


def _relative_gxg(g, x):
    return np.max(np.abs(g @ x @ g - g)) / np.max(np.abs(g))


def test_far_electron_walkers_keep_the_lu_inverse_and_exact_ratios():
    """Graded but invertible walkers keep the LU inverse; it matches the pseudo-inverse and the exact ratios."""
    gem, r_up, r_dn, r_up_moved, r_dn_moved = _far_electron_walkers()
    geminal = np.asarray(mcmc._jit_vmap_geminal_all_elements(gem, jnp.asarray(r_up), jnp.asarray(r_dn)))
    up_moved = np.asarray(mcmc._jit_vmap_geminal_all_elements(gem, jnp.asarray(r_up_moved), jnp.asarray(r_dn)))
    dn_moved = np.asarray(mcmc._jit_vmap_geminal_all_elements(gem, jnp.asarray(r_up), jnp.asarray(r_dn_moved)))
    batch = jnp.asarray(geminal)
    out = np.asarray(mcmc._geminal_pinv_from_matrix_batched(batch))
    np.testing.assert_array_equal(out, np.asarray(_lu_batch(batch)))
    atol, rtol = get_tolerance_min(("det_eval",), "strict")
    np.testing.assert_allclose(out, np.asarray(_pinv_batch(batch)), atol=atol, rtol=rtol)
    atol_r, rtol_r = get_tolerance_min(("det_ratio",), "strict")
    for w in range(geminal.shape[0]):
        for spin, moved in (("up", up_moved), ("dn", dn_moved)):
            got = _sm_ratio(out[w], geminal[w], moved[w], 0, spin)
            np.testing.assert_allclose(got, _exact_ratio(geminal[w], moved[w]), atol=atol_r, rtol=rtol_r, err_msg=f"{w} {spin}")


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize(
    "name",
    [
        "zero_row",
        "zero_column",
        "duplicate_row",
        "duplicate_column",
        "dependent_row",
        "rank_n_minus_2",
        "near_duplicate_row",
        "all_zero",
        "subnormal_row",
    ],
)
def test_pathological_walker_gets_the_pseudo_inverse(name, complex_):
    """A singular or LU-unsafe walker gets the pseudo-inverse; the healthy walkers keep their LU inverse."""
    rng = np.random.default_rng(11)
    healthy = [_random(rng, _N, complex_) for _ in range(_HEALTHY)]
    batch = jnp.asarray(np.stack([_pathological(name, rng, complex_)] + healthy))
    out = np.asarray(mcmc._geminal_pinv_from_matrix_batched(batch))
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[0], np.asarray(_pinv_batch(batch))[0])
    np.testing.assert_array_equal(out[1:], np.asarray(_lu_batch(batch))[1:])


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("case", ["far_up_and_far_down", "subnormal_row"])
def test_pseudo_inverse_of_a_finite_geminal_is_finite(case, complex_):
    """Rows/columns far below the equilibration range give a finite pseudo-inverse with G X G = G."""
    rng = np.random.default_rng(5)
    g = _random(rng, _N, complex_)
    if case == "far_up_and_far_down":
        g[5] *= 1.0e-200
        g[:, 9] *= 1.0e-200
    else:
        g[5] *= 1.0e-310
    x = np.asarray(jax.jit(mcmc._geminal_pinv_from_matrix)(jnp.asarray(g)))
    assert np.isfinite(x).all()
    _, rtol = get_tolerance_min(("det_eval",), "strict")
    assert _relative_gxg(g, x) <= rtol, _relative_gxg(g, x)
    batch_out = np.asarray(mcmc._geminal_pinv_from_matrix_batched(jnp.asarray(g[None])))
    assert np.isfinite(batch_out).all()
    assert _relative_gxg(g, batch_out[0]) <= rtol, _relative_gxg(g, batch_out[0])


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("log10_cond", [2, 6, 10])
def test_invertible_graded_geminals_keep_the_lu_inverse(log10_cond, complex_):
    """Ill-conditioned, row-graded but invertible geminals (cond up to 1e10) never take the fallback."""
    rng = np.random.default_rng(7 + log10_cond)
    mats = []
    for _ in range(8):
        q1, _ = np.linalg.qr(_random(rng, _N, complex_))
        q2, _ = np.linalg.qr(_random(rng, _N, complex_))
        g = (q1 * np.logspace(0, -log10_cond, _N)) @ q2.conj().T
        g *= np.exp2(rng.integers(-300, 1, size=_N))[:, None]
        mats.append(g)
    batch = jnp.asarray(np.stack(mats))
    np.testing.assert_array_equal(np.asarray(mcmc._geminal_pinv_from_matrix_batched(batch)), np.asarray(_lu_batch(batch)))


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
def test_strongly_graded_walker_keeps_the_exact_lu_inverse(complex_):
    """A row of order 1e-200 stays on the LU: its inverse column of order 1e200 is representable and exact."""
    rng = np.random.default_rng(13)
    g = _random(rng, _N, complex_)
    g[5] *= 1.0e-200
    batch = jnp.asarray(np.stack([g] + [_random(rng, _N, complex_) for _ in range(_HEALTHY)]))
    out = np.asarray(mcmc._geminal_pinv_from_matrix_batched(batch))
    np.testing.assert_array_equal(out, np.asarray(_lu_batch(batch)))
    assert np.max(np.abs(out[0][:, 5])) > 1.0e199
    _, residual = jax.jit(_equilibrated_geminal_lu_inverse_and_residual)(jnp.asarray(g))
    _, rtol = get_tolerance_min(("det_eval",), "strict")
    assert float(residual) <= rtol, float(residual)
