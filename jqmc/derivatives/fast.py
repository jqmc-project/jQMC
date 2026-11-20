"""Reuse a running geminal inverse in analytical derivative kernels.

Fast wavefunction and local-energy evaluators carry the Sherman-Morrison
inverse maintained by the MCMC/GFMC loop, so their derivative kernels do not
recompute ``G^{-1}``.

The analytic gradient already uses ``G^{-1}`` everywhere (the geminal cotangent
is ``M = G^{-T}``; see ``derivatives/geminal.py`` and determinant.py
``_ln_det_fast_bwd``, which uses exactly ``G_bar = g * geminal_inv.T``). So the
"fast" analytic path is the *same* math with the provided inverse substituted
for the internally-computed SVD pseudoinverse. The single plumbing primitive is
``resolve_geminal_inv`` below; every analytic kernel that needs ``G^{-1}`` takes
an optional ``geminal_inv`` and passes it through (e.g.
``geminal.dln_det_dlambda_bare(..., geminal_inv=...)``).

``resolve_geminal_inv`` is the common boundary: it returns the supplied
running inverse or computes the same thresholded SVD pseudoinverse used by a
fresh evaluation.
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


def geminal_pinv(G: jax.Array) -> jax.Array:
    """Thresholded SVD pseudoinverse of G, matching determinant.py (_ln_det_bwd)."""
    real_dtype = jnp.zeros((), dtype=G.dtype).real.dtype
    eps_rcond = get_eps("rcond_svd", real_dtype)
    U, s, Vt = jnp.linalg.svd(G, full_matrices=False)
    s_inv = jnp.where(s > eps_rcond * s[0], jnp.asarray(1.0, dtype=G.dtype) / s, jnp.asarray(0.0, dtype=G.dtype))
    return (jnp.conj(Vt).T * s_inv[jnp.newaxis, :]) @ jnp.conj(U).T


def geminal_value_tables(
    geminal_data: object,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return the orbital value tables ``(orb_up, orb_dn)`` that own ``G`` on a derivative path.

    Fused-table rule: without a supplied inverse, ``G`` and its fresh
    inverse are assembled from the fused value/gradient/Laplacian table
    (``determinant._orbital_value_grad_lap_tables``), the same table the
    determinant drift, the kinetic reverse pass and the rank-one ECP mesh base
    invert.  The table follows the consumer's ratio: the streamed ECP mesh
    weights its fields by the value-only ``det(G'_g) / det(G)`` and resolves a
    value-only base inverse (``one_move_mesh.ratio_table_geminal_inv``).
    The value-only ``compute_orb_api`` table routes AO values through
    ``ao_eval`` (float32 in mixed); an inverse built from it describes a
    different geminal matrix than the derivative tables it is contracted with,
    so sibling axes stop cancelling (``dR + sum dr = 0``).

    With a supplied running inverse the value-only table is returned unchanged:
    that inverse is maintained from the determinant-ratio (value-only) tables.
    """
    if geminal_inv is None:
        from jqmc.determinant import _orbital_value_grad_lap_tables

        orb_up, _, _, orb_dn, _, _ = _orbital_value_grad_lap_tables(geminal_data, r_up, r_dn)
        return orb_up, orb_dn
    return (
        geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up),
        geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn),
    )


def resolve_geminal_inv(G: jax.Array | None = None, geminal_inv: jax.Array | None = None) -> jax.Array:
    """Return the ``G^{-1}`` an analytic kernel should use.

    Fast path (``geminal_inv`` provided): reuse it verbatim -- the running
    Sherman-Morrison inverse, already AS-regularised by the MCMC/GFMC loop.
    Non-fast path: compute the thresholded SVD pseudoinverse of ``G``.

    Raises:
        ValueError: if both ``G`` and ``geminal_inv`` are None.
    """
    if geminal_inv is not None:
        return geminal_inv
    if G is None:
        raise ValueError("resolve_geminal_inv requires either G or geminal_inv.")
    return geminal_pinv(G)
