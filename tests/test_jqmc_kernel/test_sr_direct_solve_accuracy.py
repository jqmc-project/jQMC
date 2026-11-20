"""Direct SR solves reach the exact solution of their fp64 inputs on an ill-conditioned system.

Every direct SR route factors an explicitly formed Gram operator
(``Re(X X^dagger) + epsilon I`` in coordinate space, ``Z^T Z + epsilon I`` in
sample space). Forming the Gram matrix rounds each entry by O(u), and the
solve amplifies that by ``cond ~ 1/epsilon``. A plain factor-and-solve
therefore carries a forward error of about ``cond * u * |q|``, about 1e-12 at
``epsilon = 1e-3`` on the complex-twist AGP case (issue #784), against the
fixed-sample fp64 bound.

Each route is compared on fixed synthetic SR systems (``epsilon = 1e-3``,
``cond ~ 2-3e4``) with the exact solution of the same fp64 inputs. That
solution comes from extended-precision moments and residuals
(``_exact_sr_reference``). The fixed-sample fp64 pair is the budget. A
witness checks that the plain unrefined Gram solve of the same system
violates that budget, so the fixture stays sensitive to the defect.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE
from jqmc.jqmc_optimizer_kernel import (
    solve_device_tall_real_direct_sharded,
    solve_device_wide_real_direct_sharded,
    solve_local_device_tall_real_direct,
    solve_local_device_wide_real_direct,
    solve_tall_sr_real_direct,
    solve_wide_sr_real_direct,
)
from tests.test_jqmc_kernel._exact_sr_reference import (
    exact_real_tall_gram_solve,
    exact_real_wide_gram_solve,
)

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_EPSILON = 1.0e-3
# (coordinates P, complex samples N): wide P < 2N, tall P >= 2N.
_SHAPES = {"wide": (102, 100), "tall": (160, 50)}
_SPECTRAL_DECAY = {"wide": 0.85, "tall": 0.9}
_SEED = 20261111

_ROUTES = {
    "wide": (
        "host_wide_real",
        "device_wide_real_sharded",
        "local_device_wide_real",
    ),
    "tall": (
        "host_tall_real",
        "device_tall_real_sharded",
        "local_device_tall_real",
    ),
}
_CASES = [(regime, route) for regime, routes in _ROUTES.items() for route in routes]


def _sr_system(regime):
    """Jacobi-preconditioned SR design matrix with a geometric spectrum (cond ~ 2-3e4 at epsilon=1e-3)."""
    P, N = _SHAPES[regime]
    rng = np.random.default_rng(_SEED)
    spectrum = _SPECTRAL_DECAY[regime] ** np.arange(P)
    rotation = np.linalg.qr(rng.standard_normal((P, P)))[0]
    O = (rng.standard_normal((N, P)) + 1j * rng.standard_normal((N, P))) @ (spectrum[:, None] * rotation.T) + 3.0
    w = rng.uniform(0.5, 1.5, N)
    total = np.sum(w)
    dO = O - (w @ O) / total
    X = (np.conj(dO) * np.sqrt(w)[:, None] / np.sqrt(total)).T
    X = X / np.sqrt(np.real(np.einsum("qi,qi->q", X, np.conj(X))))[:, None]
    e = rng.standard_normal(N) + 0.3j * rng.standard_normal(N)
    F = -2.0 * np.sqrt(w) * (e - np.real((w @ e) / total)) / np.sqrt(total)
    b = np.real(X @ F)
    Z = np.concatenate((X.real, X.imag), axis=1)
    h = np.concatenate((F.real, -F.imag))
    return X, F, b, Z, h


def _exact_real_wide(X, b):
    return exact_real_wide_gram_solve(X, b, _EPSILON)


def _exact_real_tall(Z, h):
    return exact_real_tall_gram_solve(Z, h, _EPSILON)


def _solve(route, X, F, b, Z, h):
    if route == "host_wide_real":
        return solve_wide_sr_real_direct(X, b, _EPSILON)
    if route == "device_wide_real_sharded":
        return solve_device_wide_real_direct_sharded(jnp.asarray(X), b, _EPSILON)
    if route == "local_device_wide_real":
        return solve_local_device_wide_real_direct(jnp.asarray(X), b, _EPSILON)
    if route == "host_tall_real":
        return solve_tall_sr_real_direct(Z, h, _EPSILON)
    if route == "device_tall_real_sharded":
        return solve_device_tall_real_direct_sharded(jnp.asarray(X), jnp.asarray(F), _EPSILON)
    if route == "local_device_tall_real":
        return solve_local_device_tall_real_direct(jnp.asarray(X), F, _EPSILON)
    raise AssertionError(route)


def _exact(route, X, F, b, Z, h):
    if route in _ROUTES["wide"]:
        return _exact_real_wide(X, b)
    return _exact_real_tall(Z, h)


@pytest.mark.parametrize(("regime", "route"), _CASES)
def test_direct_sr_solve_matches_exact_solution_of_its_inputs(regime, route):
    """Each direct SR route equals the exact solution of its fp64 inputs on a cond ~ 2-3e4 system."""
    X, F, b, Z, h = _sr_system(regime)
    actual = np.asarray(_solve(route, X, F, b, Z, h))
    exact = _exact(route, X, F, b, Z, h)
    np.testing.assert_allclose(
        actual,
        exact,
        rtol=RTOL_PYTHON_FP64_LOOSE,
        atol=ATOL_PYTHON_FP64_LOOSE,
        err_msg=f"direct SR route {route} misses the exact solution of its fp64 inputs",
    )


@pytest.mark.parametrize("regime", sorted(_SHAPES))
def test_unrefined_gram_solve_violates_the_budget_on_the_fixture(regime):
    """Witness: a plain fp64 Gram-plus-LU solve of the same system misses the fixed-sample budget."""
    X, _F, b, Z, h = _sr_system(regime)
    if regime == "wide":
        plain = np.linalg.solve(np.real(X @ np.conj(X).T) + _EPSILON * np.eye(X.shape[0]), b)
        exact = _exact_real_wide(X, b)
    else:
        plain = Z @ np.linalg.solve(Z.T @ Z + _EPSILON * np.eye(Z.shape[1]), h)
        exact = _exact_real_tall(Z, h)
    assert np.max(np.abs(plain - exact) - (ATOL_PYTHON_FP64_LOOSE + RTOL_PYTHON_FP64_LOOSE * np.abs(exact))) > 0.0, (
        "fixture no longer exposes the cond * u error of an unrefined Gram solve"
    )
