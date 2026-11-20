"""Analytic ``ln|det G|`` VJP for AGPn bare-lambda parameters.

This module handles open-shell electrons stored directly in
``lambda_unpaired`` and paired occupations stored in ``lambda_diag``. It is
distinct from the representation in which unpaired orbitals are extra MO
coefficient columns.

The geminal matrix is built (determinant.py:_compute_geminal_all_elements) as::

    G_paired   = orb_up.T @ diag(lambda_diag) @ orb_dn      # (N_up, N_dn)
    G_unpaired = orb_up.T @ lambda_unpaired                 # (N_up, n_unpaired)
    G          = hstack([G_paired, G_unpaired])             # (N_up, N_up)

with ``orb_up`` / ``orb_dn`` the MO value matrices of shape ``(n, N_up)`` /
``(n, N_dn)`` (n = num_mo). Differentiating ``ln|det G|``::

    d ln|det G| / dG = (G^{-1})^T =: M            # M_p = M[:, :N_dn], M_u = M[:, N_dn:]

backpropagated through the (complex-)linear geminal construction via the trace
identity ``d ln det G = tr(G^{-1} dG)`` gives the closed forms below. These are
the un-conjugated holomorphic derivatives. The observable assembly applies
``conj(O)`` later in the S/K/f contraction. For a holomorphic ``g``, JAX uses
``jax.grad(Re(g))(c) = dg/dc`` under its complex-gradient convention, so adding
a conjugation here would be invisible at Gamma but wrong at a nonzero twist::

    d/d lambda_diag[k]   = Re( sum_{a,b} orb_up[k,a] M_p[a,b] orb_dn[k,b] )   # lambda_diag is real
    d/d lambda_unpaired  = orb_up @ M_u                                       # holomorphic, complex at twist

``lambda_diag`` is real by construction (occupations are real even on the
complex Bloch path), hence ``Re(...)``.

Both bare-lambda and MO-backed forms use ``AGPn_data``. Assembly distinguishes
them from the active leaf shape and owns capability routing; this module owns
only the derivative formula.
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

from jqmc._setting import get_eps


def build_geminal(orb_up: jax.Array, orb_dn: jax.Array, lambda_diag: jax.Array, lambda_unpaired: jax.Array) -> jax.Array:
    """Mirror of determinant._compute_geminal_all_elements for the AGPn bare-lambda form.

    Args:
        orb_up: MO value matrix at spin-up electrons, shape ``(n, N_up)``.
        orb_dn: MO value matrix at spin-down electrons, shape ``(n, N_dn)``.
        lambda_diag: paired occupation diagonal, shape ``(n,)`` (real).
        lambda_unpaired: open-shell block, shape ``(n, n_unpaired)``.

    Returns:
        Geminal matrix ``G`` of shape ``(N_up, N_up)`` (square: N_up = N_dn + n_unpaired).
    """
    work = jnp.result_type(orb_up.dtype, orb_dn.dtype, jnp.asarray(lambda_diag).dtype, jnp.asarray(lambda_unpaired).dtype)
    orb_up = orb_up.astype(work)
    orb_dn = orb_dn.astype(work)
    diag = jnp.diag(jnp.asarray(lambda_diag).astype(work))
    g_paired = orb_up.T @ (diag @ orb_dn)
    unp = jnp.asarray(lambda_unpaired).astype(work)
    if unp.shape[1] == 0:
        return g_paired
    g_unpaired = orb_up.T @ unp
    return jnp.hstack([g_paired, g_unpaired])


def _geminal_pinv(G: jax.Array) -> jax.Array:
    """Thresholded SVD pseudoinverse matching determinant.py (_ln_det_bwd).

    Singular values below ``rcond_svd * s_max`` are zeroed (avoids 1/~0 NaN on
    near-singular G). Hermitian-consistent for complex G.
    """
    real_dtype = jnp.zeros((), dtype=G.dtype).real.dtype
    eps_rcond = get_eps("rcond_svd", real_dtype)
    U, s, Vt = jnp.linalg.svd(G, full_matrices=False)
    s_inv = jnp.where(s > eps_rcond * s[0], jnp.asarray(1.0, dtype=G.dtype) / s, jnp.asarray(0.0, dtype=G.dtype))
    return (jnp.conj(Vt).T * s_inv[jnp.newaxis, :]) @ jnp.conj(U).T


def dln_det_dlambda_bare(
    orb_up: jax.Array,
    orb_dn: jax.Array,
    lambda_diag: jax.Array,
    lambda_unpaired: jax.Array,
    *,
    geminal_inv: jax.Array | None = None,
    preserve_complex_diag: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Closed-form ``d ln|det G| / d(lambda_diag, lambda_unpaired)``.

    Args:
        orb_up: MO value matrix at spin-up electrons, shape ``(n, N_up)``.
        orb_dn: MO value matrix at spin-down electrons, shape ``(n, N_dn)``.
        lambda_diag: paired occupation diagonal, shape ``(n,)`` (real).
        lambda_unpaired: open-shell block, shape ``(n, n_unpaired)``.
        geminal_inv: optional pre-computed ``G^{-1}`` for the fast path. When
            None, computed via the thresholded SVD pseudoinverse.
        preserve_complex_diag: retain the complex paired log derivative for
            weighted ECP/source contractions. The final real-parameter
            projection belongs after that contraction, not before it.

    Returns:
        ``(d_lambda_diag, d_lambda_unpaired)`` with shapes ``(n,)`` and
        ``(n, n_unpaired)``. ``d_lambda_diag`` is real unless
        ``preserve_complex_diag`` is requested.
    """
    n_dn = orb_dn.shape[1]
    if geminal_inv is None:
        G = build_geminal(orb_up, orb_dn, lambda_diag, lambda_unpaired)
        geminal_inv = _geminal_pinv(G)
    M = geminal_inv.T  # d ln|det G| / dG
    M_p = M[:, :n_dn]
    # d/d lambda_diag[k] = Re( sum_{a,b} orb_up[k,a] M_p[a,b] orb_dn[k,b] )  (lambda_diag real)
    d_lambda_diag = jnp.einsum("ka,ab,kb->k", orb_up, M_p, orb_dn)
    if not preserve_complex_diag:
        d_lambda_diag = jnp.real(d_lambda_diag)
    n_unpaired = jnp.asarray(lambda_unpaired).shape[1]
    if n_unpaired == 0:
        d_lambda_unpaired = jnp.zeros((orb_up.shape[0], 0), dtype=M.dtype)
    else:
        M_u = M[:, n_dn:]
        d_lambda_unpaired = orb_up @ M_u  # unconjugated holomorphic derivative
    return d_lambda_diag, d_lambda_unpaired
