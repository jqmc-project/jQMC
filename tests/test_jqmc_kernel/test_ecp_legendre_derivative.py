"""Non-local ECP projector derivative P_l'(x) on the closed interval [-1, 1].

The analytic ECP force / stress kernels (``jqmc.derivatives.ecp_force``) need P_l'(cos theta)
for every non-local channel.  The recurrence form l (P_{l-1} - x P_l) / (1 - x^2) is 0/0
at cos theta = +-1: an electron lying exactly on a quadrature direction of the current mesh
rotation.  That happens in production right after a GFMC T-move / DLT-move (the electron
hops onto a mesh point and the measurement reuses the same rotation), so with that form every
GFMC force on an ECP with a live l >= 1 channel (Cl2 ccECP) is NaN for the affected walkers.
l = 0 channels never reach the division.

References, all independent of the production table:
* the closed form P_l'(+-1) = (+-1)^(l-1) l (l+1) / 2;
* forward-mode autodiff of the tabulated P_l polynomial itself, on a grid that includes the
  end points and points next to them.

The driver-level consequence (finite GFMC forces on Cl2) is pinned in
``test_driver_ecp_high_l``.
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

pytestmark = [pytest.mark.with_chdir]

from tests._finite_allclose import assert_finite_allclose
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from tests._precision_fidelity import mixed_vs_full
from tests.test_jqmc_kernel._orbital_mixed_fidelity import assert_mixed_orbital_derivative_fidelity
from jqmc.coulomb_potential import _legendre_tablated
from jqmc.derivatives.ecp_force import _leg_jnp

_L_VALUES = tuple(range(7))
_X = np.concatenate(
    [
        np.array([-1.0, 1.0, np.nextafter(1.0, 0.0), np.nextafter(-1.0, 0.0)]),
        1.0 - np.logspace(-15, -1, 8),
        -1.0 + np.logspace(-15, -1, 8),
        np.linspace(-0.95, 0.95, 21),
    ]
)


def _reference_derivative(l, x):
    return np.asarray(jax.vmap(jax.grad(lambda t: _legendre_tablated(l, t)))(jnp.asarray(x, dtype=jnp.float64)))


def _check_derivative(dP, l, x, atol, rtol, label):
    assert_finite_allclose(dP, _reference_derivative(l, x), atol=atol, rtol=rtol, err_msg=f"{label}: P_{l}'(x)")


@pytest.mark.light
@pytest.mark.parametrize("l", _L_VALUES)
def test_legendre_derivative_matches_polynomial_on_closed_interval(l):
    """Both-mode FP64 AD, additional Mixed fidelity, and original endpoint identities."""
    atol, rtol = get_tolerance_min(("coulomb",), "strict")
    x = jnp.asarray(_X, dtype=jnp.float64)
    P, dP = jax.vmap(lambda t: _leg_jnp(l, t))(x)
    P, dP = np.asarray(P), np.asarray(dP)
    assert np.all(np.isfinite(dP)), f"P_{l}' is not finite on [-1, 1]: {dP}"
    assert_finite_allclose(P, np.asarray(jax.vmap(lambda t: _legendre_tablated(l, t))(x)), atol=atol, rtol=rtol)
    end = np.array([(-1.0) ** (l - 1) * l * (l + 1) / 2.0, l * (l + 1) / 2.0]) if l else np.zeros(2)
    assert_finite_allclose(dP[:2], end, atol=atol, rtol=rtol, err_msg=f"P_{l}'(+-1) closed form")
    _check_derivative(dP, l, _X, atol, rtol, "production")
    if is_mixed_precision_enabled():
        mixed, full = mixed_vs_full(
            lambda: jnp.asarray(_X, dtype=jnp.float64),
            lambda points: jax.vmap(lambda t: _leg_jnp(l, t))(points)[1],
        )
        assert_mixed_orbital_derivative_fidelity(mixed, full, f"Legendre P_{l}' mixed vs full")


@pytest.mark.light
@pytest.mark.parametrize("l", [1, 2])
def test_legendre_derivative_check_rejects_the_recurrence_form(l):
    """Negative control: the recurrence form fails the same check at the end points."""
    atol, rtol = get_tolerance_min(("coulomb",), "strict")
    x = jnp.asarray(_X, dtype=jnp.float64)
    old = np.asarray(l * (_legendre_tablated(l - 1, x) - x * _legendre_tablated(l, x)) / (1.0 - x * x))
    assert not np.all(np.isfinite(old)), "the recurrence form must be 0/0 at x = +-1"
    with pytest.raises(AssertionError, match=f"recurrence: P_{l}'"):
        _check_derivative(old, l, _X, atol, rtol, "recurrence")
