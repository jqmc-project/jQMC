"""R7 component tests -- the ``_fast`` geminal_inv reuse plumbing.

The analytic geminal cotangent is ``M = G^{-T}``; the fast path substitutes the
provided ``geminal_inv`` for the internally computed SVD pseudoinverse. These
matrix-level checks confirm (a) the resolver returns the provided inverse and
matches the internal pinv for a well-conditioned G, and (b) the resulting
log-det parameter gradient (fast = provided inverse) is identical to the
non-fast (internal pinv) path and to ``jax.grad`` -- i.e. reuse is exact.
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

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

pytestmark = [pytest.mark.with_chdir]

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance_min
from jqmc.derivatives.fast import geminal_pinv, resolve_geminal_inv


def _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol):
    """Keep the homogeneous-block bound transparent to GDD runtime receipts."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max()


def _random_geminal(n, seed, complex_g):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    if complex_g:
        A = A + 1j * rng.normal(size=(n, n))
    A = A + (n + 1.0) * np.eye(n)  # well-conditioned
    return jnp.asarray(A)


@pytest.mark.light
@pytest.mark.parametrize("complex_g", [False, True])
def test_resolve_uses_provided_and_matches_pinv(complex_g):
    """Resolve uses provided and matches pinv."""
    G = _random_geminal(5, seed=1, complex_g=complex_g)
    provided = geminal_pinv(G)
    # provided is returned verbatim
    assert resolve_geminal_inv(geminal_inv=provided) is provided
    # internal pinv == provided for a well-conditioned G == true inverse
    internal = resolve_geminal_inv(G=G)
    atol, rtol = get_tolerance_min(("det_eval",), "strict")
    np.testing.assert_allclose(np.asarray(internal), np.asarray(provided), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(internal @ G), np.asarray(jnp.eye(5)), atol=atol, rtol=rtol)
    jax.clear_caches()


@pytest.mark.light
def test_resolve_requires_an_argument():
    """Resolve requires an argument."""
    with pytest.raises(ValueError):
        resolve_geminal_inv()


def _fast_logdet_parameter_case(complex_lambda):
    """Rebuild the original PCG64(7) matrix source in the active policy."""
    n = 4
    rng = np.random.default_rng(7)
    A = jnp.asarray(rng.normal(size=(n, n)))
    B = jnp.asarray(rng.normal(size=(n, n)))
    if complex_lambda:
        Lam = jnp.asarray(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    else:
        Lam = jnp.asarray(rng.normal(size=(n, n)))
    Lam = Lam + (n + 1.0) * jnp.eye(n)
    V = jnp.asarray(rng.normal(size=(n, n)) + (1j * rng.normal(size=(n, n)) if complex_lambda else 0.0))
    return A, B, Lam, V


@pytest.mark.light
@pytest.mark.parametrize("complex_lambda", [False, True])
def test_fast_logdet_param_grad_matches_nonfast_and_autograd(complex_lambda):
    """d ln|det G(Lambda)| / dLambda using the provided inverse == internal pinv == jax.grad.

    G = A @ Lambda @ B is the generic (complex-)linear param form the geminal
    construction reduces to; M = G^{-T} gives the un-conjugated holomorphic
    cotangent. Fast (provided inverse) must reproduce non-fast exactly.
    """
    if is_mixed_precision_enabled():
        # This non-AD fast/non-fast parity keeps its original entry-policy pair.
        control_atol, control_rtol = get_tolerance_min(("det_eval",), "strict")

        def analytical(case):
            A, B, Lam, _V = case
            G = A @ Lam @ B
            provided = geminal_pinv(G)
            g_fast = A.T @ resolve_geminal_inv(G=G, geminal_inv=provided).T @ B.T
            g_nonfast = A.T @ resolve_geminal_inv(G=G, geminal_inv=None).T @ B.T
            return g_fast, g_nonfast

        mixed, full = mixed_vs_full(lambda: _fast_logdet_parameter_case(complex_lambda), analytical)
        for route_precision, (g_fast, g_nonfast) in zip(("mixed", "full"), (mixed, full), strict=True):
            np.testing.assert_allclose(np.asarray(g_fast), np.asarray(g_nonfast), atol=control_atol, rtol=control_rtol)
        blocks = mixed_fidelity_blocks(mixed, full)
        atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
        for mixed_block, full_block in blocks:
            _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol)
        corrupted = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            _assert_component_derivative_fidelity(corrupted, blocks[0][1], atol, rtol)
        jax.clear_caches()
        return

    A, B, Lam, V = _fast_logdet_parameter_case(complex_lambda)

    def build(L):
        return A @ L @ B

    G = build(Lam)
    provided = geminal_pinv(G)

    def grad_via(geminal_inv):
        M = resolve_geminal_inv(G=G, geminal_inv=geminal_inv).T  # G^{-T}
        # vjp of G = A @ Lambda @ B w.r.t. Lambda with cotangent M (holomorphic)
        return A.T @ M @ B.T

    g_fast = grad_via(provided)
    g_nonfast = grad_via(None)

    # holomorphic directional oracle along the original source direction V
    _, jvp_val = jax.jvp(lambda L: jnp.log(jnp.linalg.det(build(L))), (Lam,), (V,))

    atol, rtol = get_tolerance_min(("det_eval",), "strict")
    np.testing.assert_allclose(np.asarray(g_fast), np.asarray(g_nonfast), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(jnp.sum(g_fast * V)), np.asarray(jvp_val), atol=atol, rtol=rtol)
    jax.clear_caches()
