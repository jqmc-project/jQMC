"""R3 component tests -- analytic ``d ln|det G| / d(lambda_diag, lambda_unpaired)``
for the AGPn bare-lambda leaf vs. finite difference and vs. ``jax.grad``.

These are *component-level* tests: they exercise the closed-form kernel in
``jqmc.derivatives.geminal`` directly on random orbital/lambda matrices, so they
run independently of the R0 ``assemble.py`` wiring and the ``_auto`` twin. The
matrix-level forward mirrors ``determinant._compute_geminal_all_elements``; the
``jax.grad`` oracle goes through the same ``jnp.linalg.det`` the production
custom-VJP differentiates, so a match here certifies the math.
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
from jqmc.derivatives.geminal import build_geminal, dln_det_dlambda_bare


def _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol):
    """Keep the homogeneous-block bound transparent to GDD runtime receipts."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max()


def _random_case(n, n_up, n_dn, *, complex_unpaired: bool, seed: int):
    """Random (orb_up, orb_dn, lambda_diag, lambda_unpaired) with a well-conditioned G."""
    n_unpaired = n_up - n_dn
    rng = np.random.default_rng(seed)
    orb_up = rng.normal(size=(n, n_up))
    orb_dn = rng.normal(size=(n, n_dn))
    lambda_diag = rng.normal(size=(n,)) + 2.0  # away from zero -> well-conditioned
    if complex_unpaired:
        orb_up = orb_up + 1j * rng.normal(size=(n, n_up))
        orb_dn = orb_dn + 1j * rng.normal(size=(n, n_dn))
        lu = rng.normal(size=(n, n_unpaired)) + 1j * rng.normal(size=(n, n_unpaired))
    else:
        lu = rng.normal(size=(n, n_unpaired))
    return (
        jnp.asarray(orb_up),
        jnp.asarray(orb_dn),
        jnp.asarray(lambda_diag),
        jnp.asarray(lu),
    )


def _logdet(orb_up, orb_dn, lambda_diag, lambda_unpaired):
    G = build_geminal(orb_up, orb_dn, lambda_diag, lambda_unpaired)
    return jnp.log(jnp.abs(jnp.linalg.det(G)))


def _fd_lambda_diag(orb_up, orb_dn, lambda_diag, lambda_unpaired, h):
    ld = np.asarray(lambda_diag, dtype=np.float64)
    out = np.zeros_like(ld)
    for k in range(ld.shape[0]):
        lp, lm = ld.copy(), ld.copy()
        lp[k] += h
        lm[k] -= h
        fp = _logdet(orb_up, orb_dn, jnp.asarray(lp), lambda_unpaired)
        fm = _logdet(orb_up, orb_dn, jnp.asarray(lm), lambda_unpaired)
        out[k] = (np.asarray(fp) - np.asarray(fm)) / (2 * h)
    return out


@pytest.mark.light
@pytest.mark.skip_mixed_precision(
    reason="FD comparison dominated by float32 round-off",
    alternative_guarantee_ref="MATH-DERIV-LOG-WAVEFUNCTION-001",
)
@pytest.mark.parametrize("complex_unpaired", [False, True])
@pytest.mark.parametrize("n,n_up,n_dn", [(5, 3, 2), (6, 4, 1)])
def test_dln_det_dlambda_diag_vs_fd(n, n_up, n_dn, complex_unpaired):
    """lambda_diag derivative matches central finite difference."""
    orb_up, orb_dn, ld, lu = _random_case(n, n_up, n_dn, complex_unpaired=complex_unpaired, seed=11)
    d_diag, _ = dln_det_dlambda_bare(orb_up, orb_dn, ld, lu)
    h = 1e-5
    fd = _fd_lambda_diag(orb_up, orb_dn, ld, lu, h)
    atol, rtol = get_tolerance_min(("det_eval",), "medium")
    np.testing.assert_allclose(np.asarray(d_diag), fd, atol=atol, rtol=rtol)
    jax.clear_caches()


def _holo_logdet(orb_up, orb_dn, lambda_diag, lambda_unpaired):
    """Holomorphic complex log-det (no abs) -- for directional jvp checks."""
    G = build_geminal(orb_up, orb_dn, lambda_diag, lambda_unpaired)
    return jnp.log(jnp.linalg.det(G))


@pytest.mark.light
@pytest.mark.parametrize("complex_unpaired", [False, True])
@pytest.mark.parametrize("n,n_up,n_dn", [(5, 3, 2), (6, 4, 1)])
def test_dln_det_dlambda_unpaired_vs_holomorphic_jvp(n, n_up, n_dn, complex_unpaired):
    """Un-conjugated holomorphic d/d lambda_unpaired (D15): sum(D*V) == jvp along V.

    Full uses the holomorphic complex log-det so the directional derivative is the
    un-ambiguous complex-linear ``sum_{k,j} (orb_up @ M_u)[k,j] V[k,j]`` -- this
    is the un-conjugated O_k convention production uses, not jax.grad's conjugate.
    """
    if is_mixed_precision_enabled():

        def case_factory():
            return _random_case(n, n_up, n_dn, complex_unpaired=complex_unpaired, seed=23)

        def analytical(case):
            _, d_unpaired = dln_det_dlambda_bare(*case)
            return d_unpaired

        mixed, full = mixed_vs_full(case_factory, analytical)
        blocks = mixed_fidelity_blocks(mixed, full)
        atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
        for mixed_block, full_block in blocks:
            _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol)
        corrupted = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            _assert_component_derivative_fidelity(corrupted, blocks[0][1], atol, rtol)
        jax.clear_caches()
        return

    orb_up, orb_dn, ld, lu = _random_case(n, n_up, n_dn, complex_unpaired=complex_unpaired, seed=23)
    _, d_unp = dln_det_dlambda_bare(orb_up, orb_dn, ld, lu)

    rng = np.random.default_rng(99)
    V = rng.normal(size=lu.shape)
    if complex_unpaired:
        V = V + 1j * rng.normal(size=lu.shape)
    V = jnp.asarray(V)

    directional = jnp.sum(d_unp * V)
    _, jvp_val = jax.jvp(lambda x: _holo_logdet(orb_up, orb_dn, ld, x), (lu,), (V,))

    atol, rtol = get_tolerance_min(("det_eval",), "strict")
    np.testing.assert_allclose(np.asarray(directional), np.asarray(jvp_val), atol=atol, rtol=rtol)
    jax.clear_caches()


@pytest.mark.light
def test_closed_shell_unpaired_empty():
    """n_unpaired == 0 returns an empty (n, 0) block and a finite diag gradient."""
    orb_up, orb_dn, ld, lu = _random_case(4, 2, 2, complex_unpaired=False, seed=5)
    d_diag, d_unp = dln_det_dlambda_bare(orb_up, orb_dn, ld, lu)
    assert d_unp.shape == (4, 0)
    assert np.all(np.isfinite(np.asarray(d_diag)))
    jax.clear_caches()
