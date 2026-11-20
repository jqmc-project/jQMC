"""AO-type-specific analytic derivative kernels.

This module provides the AO-leaf derivative pieces composed by the shared VJP
in :mod:`jqmc.derivatives.assemble`:

* Pure plane-wave AO: strain (cell) derivative.
* B-spline AO: strain derivative with zero nuclear-position derivative.
* pGTO and spherical-GTO radial exponent/coefficient derivatives.

Each kernel is verified at component level (vs ``jax.jvp`` of the production
forward and vs finite difference), independently of the shared assembly layer.
JIT and J3-basis capability registration is deferred to the assemble module.

----------------------------------------------------------------------------
Pure plane-wave AO
----------------------------------------------------------------------------
The real Gamma plane-wave basis (atomic_orbital.py:AOs_plane_wave_data) depends
on the lattice ONLY through the reciprocal vectors

    G[a] = 2*pi * inv(cell) @ m[a]            # m[a] = integer Miller triple

(no dependence on nuclear positions R -> dAO/dR = 0; the per-band amplitudes
live in MOs_data, so dAO/dc = 0 for the leaf itself). Hence the sole new
analytic piece for forces/stress is the strain chain, which reduces to the
directional cell derivative below. For a cell perturbation ``dcell``::

    dG = -G @ (inv(cell) @ dcell).T

The wrap-into-cell only shifts the phase by an integer multiple of 2*pi
(``G . (n @ cell) = 2*pi * (m . n)``), whose derivative vanishes, so the kernel
uses the un-wrapped coordinates; component tests draw electrons inside the home
cell where the wrap is the identity.
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

from typing import TypeAlias

import jax
import jax.numpy as jnp

import jax.scipy as jscipy

from jqmc.atomic_orbital import (
    AOs_bspline_complex_data,
    AOs_bspline_data,
    AOs_plane_wave_complex_data,
    AOs_plane_wave_data,
    AOs_sphe_gto_data,
    _compute_S_l_m_and_grad_lap,
    _inv_3x3,
    _reduce_primitives_to_aos,
)

_PlaneWaveData: TypeAlias = AOs_plane_wave_data | AOs_plane_wave_complex_data
_ArrayPair: TypeAlias = tuple[jax.Array, jax.Array]
_KineticFields: TypeAlias = tuple[jax.Array, jax.Array, jax.Array]
_ValueGradLapTangent: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


def _pw_geometry(aos: _PlaneWaveData, dcell: jax.Array) -> _ArrayPair:
    """Return ``(G, dG)`` for a plane-wave leaf along cell perturbation ``dcell``.

    ``G`` has shape ``(num_ao, 3)``; ``dG`` is its directional derivative
    ``-G @ (inv(cell) @ dcell).T``.
    """
    cell = aos.structure_data._cell_jnp
    inv = _inv_3x3(cell)
    G = aos._g_vectors_jnp  # (num_ao, 3)
    dG = -G @ (inv @ dcell).T
    return G, dG


def pw_value_grad_lap_dcell(
    aos: AOs_plane_wave_data, r_carts: jax.Array, dcell: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Directional derivative of the plane-wave (value, grad, lap) along ``dcell``.

    Args:
        aos: ``AOs_plane_wave_data`` leaf.
        r_carts: electron coordinates ``(N_e, 3)`` (inside the home cell).
        dcell: cell perturbation ``(3, 3)`` (rows = d vec_a / d vec_b / d vec_c).

    Returns:
        ``(dval, dgx, dgy, dgz, dlap)``, each ``(num_ao, N_e)`` -- the change in
        value / Cartesian gradients / Laplacian for an infinitesimal step of the
        lattice along ``dcell``. Matches ``jax.jvp`` of
        ``_compute_AOs_value_grad_lap`` w.r.t. the cell.
    """
    G, dG = _pw_geometry(aos, dcell)
    is_sin = aos._phase_is_sin_jnp  # (num_ao,)

    theta = jnp.dot(r_carts, G.T)  # (N_e, num_ao)
    phi = theta - (is_sin * (jnp.pi / 2.0))[None, :]
    dtheta = jnp.dot(r_carts, dG.T)  # (N_e, num_ao)

    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)

    # value = cos(phi); dval = -sin(phi) * dphi   (dphi = dtheta)
    dval = (-sin_phi * dtheta).T

    # grad_k = -sin(phi) * G_k ; product rule with d(-sin(phi)) = -cos(phi)*dphi
    d_dvaldtheta = -cos_phi * dtheta  # d(-sin(phi))
    dvaldtheta = -sin_phi
    dgx = (d_dvaldtheta * G[:, 0][None, :] + dvaldtheta * dG[:, 0][None, :]).T
    dgy = (d_dvaldtheta * G[:, 1][None, :] + dvaldtheta * dG[:, 1][None, :]).T
    dgz = (d_dvaldtheta * G[:, 2][None, :] + dvaldtheta * dG[:, 2][None, :]).T

    # lap = -|G|^2 * value ; d|G|^2 = 2 G.dG
    g2 = jnp.sum(G * G, axis=1)  # (num_ao,)
    dg2 = 2.0 * jnp.sum(G * dG, axis=1)  # (num_ao,)
    val = cos_phi  # (N_e, num_ao)
    dval_ea = -sin_phi * dtheta
    dlap = (-dg2[None, :] * val - g2[None, :] * dval_ea).T

    return dval, dgx, dgy, dgz, dlap


# ----------------------------------------------------------------------------
# R2 -- plane-wave AO ELECTRON-position derivative kernels (kinetic side)
# ----------------------------------------------------------------------------
# The kinetic drift / force / stress contract the per-AO grad, grad-of-laplacian
# and Hessian (the pgto_deriv.pgto_{grad,gradlap,hessian} interface). For the real
# Gamma plane-wave leaf AO_k = cos(phi_k), phi_k = G_k.r - is_sin_k*pi/2, every
# d/dr_d steps the phase (cos->-sin->-cos->sin) and pulls a G_d, so the whole
# tower is closed form: with val=cos(phi), dval=-sin(phi) (= the forward's
# value / dval_dtheta), G2=|G|^2,
#   grad[d]     =  G_d dval
#   hessian[c,d]= -G_c G_d val
#   laplacian   = -G2 val
#   gradlap[d]  = -G2 grad[d] = -G2 G_d dval
# matching AOs_plane_wave_data._compute_AOs_{grad,laplacian} exactly. Returned at
# the native PW rank (num_ao); plane waves are not atom-centred so there is no
# image fold / home-atom scatter (and dAO/dR = 0, handled by the force skip).
def _pw_val_dval(aos: _PlaneWaveData, r_carts: jax.Array) -> _KineticFields:
    """(G, val, dval) for a plane-wave leaf, unifying real Gamma and complex twist so the SAME
    kernels (grad[d]=G_d dval, hess[c,d]=-G_c G_d val, gradlap[d]=-|G|^2 G_d dval) cover both:
      real (cos/sin):  G=g_vectors, phi=G.r - is_sin*pi/2 (cell-wrapped, G.r mod 2pi),
                       val=cos(phi), dval=-sin(phi).
      complex (twist): G=gk=G+k_cart (NOT wrapped -- exp(i k.T)!=1 for a twist), val=exp(i gk.r),
                       dval=i*val  (grad = i gk val = gk_d * (i val)).
    The leaf-aware kinetic paths take Re of the assembled product (holomorphic; no-op for real)."""
    if "complex" in type(aos).__name__:
        gk = jnp.asarray(aos._gk_vectors_jnp, jnp.float64)  # (num_ao,3), G + k_cart
        r = jnp.asarray(r_carts, jnp.float64)  # NO wrap (twist phase e^{i k.T} is not 1)
        val = jnp.exp(1j * (gk @ r.T))  # (num_ao,N) complex
        return gk, val, 1j * val
    from jqmc.atomic_orbital import _wrap_into_cell_if_pbc

    G = jnp.asarray(aos._g_vectors_jnp, jnp.float64)  # (num_ao,3)
    is_sin = jnp.asarray(aos._phase_is_sin_jnp, jnp.float64)  # (num_ao,)
    r = jnp.asarray(_wrap_into_cell_if_pbc(aos.structure_data, jnp.asarray(r_carts, jnp.float64)), jnp.float64)  # (N,3)
    phi = (G @ r.T) - (is_sin * (jnp.pi / 2.0))[:, None]  # (num_ao,N)
    return G, jnp.cos(phi), -jnp.sin(phi)


def pw_grad(aos: AOs_plane_wave_data | AOs_plane_wave_complex_data, r: jax.Array) -> jax.Array:
    """(3, num_ao, N) plane-wave AO gradient: grad[d] = G_d dval."""
    G, _val, dval = _pw_val_dval(aos, r)
    return jnp.einsum("kd,ki->dki", G, dval)


def pw_gradlap(aos: AOs_plane_wave_data | AOs_plane_wave_complex_data, r: jax.Array) -> jax.Array:
    """(3, num_ao, N) grad(laplacian AO): ``-|G|^2 grad = -|G|^2 G_d dval``."""
    G, _val, dval = _pw_val_dval(aos, r)
    g2 = jnp.sum(G * G, axis=1)
    return -jnp.einsum("k,kd,ki->dki", g2, G, dval)


def pw_hessian(aos: AOs_plane_wave_data | AOs_plane_wave_complex_data, r: jax.Array) -> jax.Array:
    """(3, 3, num_ao, N) plane-wave AO Hessian: H[c,d] = -G_c G_d val."""
    G, val, _dval = _pw_val_dval(aos, r)
    return -jnp.einsum("kc,kd,ki->cdki", G, G, val)


# ----------------------------------------------------------------------------
# R1 -- B-spline AO
# ----------------------------------------------------------------------------
# The B-spline band leaf (atomic_orbital.py:AOs_bspline_data) depends on the
# lattice only through the fractional metric ``frac = r @ inv(cell)`` and the
# Cartesian conversions of the separable spline derivatives (``grad_r``,
# Laplacian via ``a_mat = N_i N_j (inv.T inv)``). It carries NO nuclear-position
# dependence (dAO/dR = 0) and ``spline_coefficients`` is never variational
# (dAO/dc = 0). So, as for the plane wave, the only new analytic piece is the
# strain (cell) derivative.
#
# Writing every separable derivative as a single einsum
#   E(a,b,c) = einsum('bpqr,p,q,r->b', cg, W0[a], W1[b], W2[c])
# (W_k[o] = o-th f-derivative weight vector on axis k), the t-space value,
# gradient and Hessian are E000 / {E100,E010,E001} / {E200,...}. A cell step
# dcell perturbs the fractional offset by ``df = (r @ dinv) * ns`` with
# ``dinv = -inv dcell inv`` (floor locally constant), and chains as
#   dval        = grad_t . df
#   d(grad_t)   = H @ df
#   d(grad_r)   = ((H@df)*ns) @ inv.T + (grad_t*ns) @ dinv.T
#   d(lap)      = <d a_mat, H> + <a_mat, dH>      (dH needs 3rd-order weights)
# Hence ``_bspline_d3w`` below (the forward only needs up to the Hessian).


def _bspline_d3w(f: jax.Array, order: int) -> jax.Array:
    """Third f-derivative of the cardinal B-spline weight vector (order 3 or 5).

    Companion to ``AOs_bspline_data._bspline_weights_jax`` (which returns value,
    1st and 2nd derivatives); the Laplacian's cell derivative needs the 3rd.
    """
    if order == 3:
        return jnp.stack([-1.0 + 0.0 * f, 3.0 + 0.0 * f, -3.0 + 0.0 * f, 1.0 + 0.0 * f])
    if order == 5:
        return (
            jnp.stack(
                [
                    -60.0 * (1.0 - f) ** 2,
                    120.0 - 480.0 * f + 300.0 * f**2,
                    720.0 * f - 600.0 * f**2,
                    -120.0 - 480.0 * f + 600.0 * f**2,
                    60.0 + 120.0 * f - 300.0 * f**2,
                    60.0 * f**2,
                ]
            )
            / 120.0
        )
    raise ValueError(f"Unsupported B-spline order {order!r} (use 3 or 5).")


def bspline_value_grad_lap_dcell(
    aos: AOs_bspline_data, r_carts: jax.Array, dcell: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Directional derivative of the B-spline (value, grad, lap) along ``dcell``.

    Args / returns mirror :func:`pw_value_grad_lap_dcell`. Matches ``jax.jvp`` of
    ``AOs_bspline_data._compute_AOs_value_grad_lap`` w.r.t. the cell, away from
    spline knots (where ``floor`` is locally constant).
    """
    order = aos.spline_order
    coeffs = aos._coeffs_cube_jnp  # (n_band, N0, N1, N2)
    cell = aos.structure_data._cell_jnp.astype(jnp.float64)
    inv = _inv_3x3(cell)
    ns = jnp.asarray(aos.spline_shape, dtype=jnp.float64)
    metric = inv.T @ inv
    a_mat = (ns[:, None] * ns[None, :]) * metric
    start = -((order - 1) // 2)
    offsets = jnp.arange(start, start + order + 1, dtype=jnp.int32)

    dinv = -inv @ dcell @ inv
    dmetric = dinv.T @ inv + inv.T @ dinv
    da_mat = (ns[:, None] * ns[None, :]) * dmetric

    def eval_one(r: jax.Array) -> _ValueGradLapTangent:
        frac = r @ inv
        frac = frac - jnp.floor(frac)
        t = frac * ns
        i = jnp.floor(t)
        f = t - i
        i_int = i.astype(jnp.int32)
        df = (r @ dinv) * ns  # cell-perturbation of f (floor const)

        w0, dw0, d20 = aos._bspline_weights_jax(f[0], order)
        w1, dw1, d21 = aos._bspline_weights_jax(f[1], order)
        w2, dw2, d22 = aos._bspline_weights_jax(f[2], order)
        d30 = _bspline_d3w(f[0], order)
        d31 = _bspline_d3w(f[1], order)
        d32 = _bspline_d3w(f[2], order)
        W0 = [w0, dw0, d20, d30]
        W1 = [w1, dw1, d21, d31]
        W2 = [w2, dw2, d22, d32]

        cg = jnp.take(coeffs, i_int[0] + offsets, axis=1, mode="wrap")
        cg = jnp.take(cg, i_int[1] + offsets, axis=2, mode="wrap")
        cg = jnp.take(cg, i_int[2] + offsets, axis=3, mode="wrap")  # (n_band, K, K, K)

        def E(a: int, b: int, c: int) -> jax.Array:
            return jnp.einsum("bpqr,p,q,r->b", cg, W0[a], W1[b], W2[c])

        grad_t = jnp.stack([E(1, 0, 0), E(0, 1, 0), E(0, 0, 1)], axis=1)  # (n_band, 3)
        # symmetric t-Hessian per band (n_band, 3, 3)
        H00, H11, H22 = E(2, 0, 0), E(0, 2, 0), E(0, 0, 2)
        H01, H02, H12 = E(1, 1, 0), E(1, 0, 1), E(0, 1, 1)
        H = jnp.stack(
            [
                jnp.stack([H00, H01, H02], axis=1),
                jnp.stack([H01, H11, H12], axis=1),
                jnp.stack([H02, H12, H22], axis=1),
            ],
            axis=1,
        )  # (n_band, 3, 3)

        # dval = grad_t . df
        dval = grad_t @ df

        # d(grad_t) = H @ df ; d(grad_r) = ((H@df)*ns)@inv.T + (grad_t*ns)@dinv.T
        Hdf = jnp.einsum("bij,j->bi", H, df)  # (n_band, 3)
        dgrad_r = (Hdf * ns[None, :]) @ inv.T + (grad_t * ns[None, :]) @ dinv.T

        # d(lap) = <da_mat, H> + <a_mat, dH>
        # dH[i,j] = sum_k (raise k) df_k  -- third-order einsums
        dH00 = E(3, 0, 0) * df[0] + E(2, 1, 0) * df[1] + E(2, 0, 1) * df[2]
        dH11 = E(1, 2, 0) * df[0] + E(0, 3, 0) * df[1] + E(0, 2, 1) * df[2]
        dH22 = E(1, 0, 2) * df[0] + E(0, 1, 2) * df[1] + E(0, 0, 3) * df[2]
        dH01 = E(2, 1, 0) * df[0] + E(1, 2, 0) * df[1] + E(1, 1, 1) * df[2]
        dH02 = E(2, 0, 1) * df[0] + E(1, 1, 1) * df[1] + E(1, 0, 2) * df[2]
        dH12 = E(1, 1, 1) * df[0] + E(0, 2, 1) * df[1] + E(0, 1, 2) * df[2]

        def contract(
            a: jax.Array,
            H00: jax.Array,
            H11: jax.Array,
            H22: jax.Array,
            H01: jax.Array,
            H02: jax.Array,
            H12: jax.Array,
        ) -> jax.Array:
            return a[0, 0] * H00 + a[1, 1] * H11 + a[2, 2] * H22 + 2.0 * (a[0, 1] * H01 + a[0, 2] * H02 + a[1, 2] * H12)

        dlap = contract(da_mat, H00, H11, H22, H01, H02, H12) + contract(a_mat, dH00, dH11, dH22, dH01, dH02, dH12)
        return dval, dgrad_r[:, 0], dgrad_r[:, 1], dgrad_r[:, 2], dlap

    # vmap over electrons -> (N_e, n_band) per field; transpose to the (n_band, N_e) AO convention.
    return tuple(o.T for o in jax.vmap(eval_one)(jnp.asarray(r_carts, jnp.float64)))


# ----------------------------------------------------------------------------
# R1 -- B-spline AO ELECTRON-position kinetic kernels (grad / grad-of-lap / Hessian)
# ----------------------------------------------------------------------------
# Separable spline AO_b(r) = sum_pqr cg[b,p,q,r] w(f0) w(f1) w(f2), t_k = frac_k N_k,
# f = frac fractional part. The chain rule dt_k/dr_c = N_k inv[c,k] =: M[k,c]:
#   grad_r[b,c]      = sum_k  gt[b,k]      M[k,c]
#   hessian_r[b,c,d] = sum_kl Ht[b,k,l]    M[k,c] M[l,d]
#   gradlap_r[b,d]   = sum_kl a_mat[k,l] dHt[k,l;d],  dHt[k,l;d] = sum_m T3[k,l,m] M[m,d]
# where gt / Ht / T3 are the t-space 1st/2nd/3rd derivatives (E(a,b,c) with d3w) and a_mat is
# CONSTANT here (cell fixed for the electron derivative -- no da_mat strain term). Reuses the exact
# dH_ij(v) machinery of bspline_value_grad_lap_dcell with v = M[:,d] (the dt/dr column).
def _bspline_kinetic_fields(aos: AOs_bspline_data, r_carts: jax.Array) -> _KineticFields:
    """(grad, gradlap, hessian) for a B-spline leaf, matching pgto_deriv.pgto_* shapes:
    grad/gradlap (3, num_ao, N), hessian (3, 3, num_ao, N). EAGER (np at the boundary)."""
    order = aos.spline_order
    coeffs = aos._coeffs_cube_jnp
    cell = aos.structure_data._cell_jnp.astype(jnp.float64)
    inv = _inv_3x3(cell)
    ns = jnp.asarray(aos.spline_shape, dtype=jnp.float64)
    metric = inv.T @ inv
    a_mat = (ns[:, None] * ns[None, :]) * metric
    M = ns[:, None] * inv.T  # (3, 3) M[k,c] = N_k inv[c,k] = dt_k/dr_c
    start = -((order - 1) // 2)
    offsets = jnp.arange(start, start + order + 1, dtype=jnp.int32)

    def eval_one(r: jax.Array) -> _KineticFields:
        frac = r @ inv
        frac = frac - jnp.floor(frac)
        t = frac * ns
        i_int = jnp.floor(t).astype(jnp.int32)
        f = t - jnp.floor(t)
        w0, dw0, d20 = aos._bspline_weights_jax(f[0], order)
        w1, dw1, d21 = aos._bspline_weights_jax(f[1], order)
        w2, dw2, d22 = aos._bspline_weights_jax(f[2], order)
        W0, W1, W2 = (
            [w0, dw0, d20, _bspline_d3w(f[0], order)],
            [w1, dw1, d21, _bspline_d3w(f[1], order)],
            [
                w2,
                dw2,
                d22,
                _bspline_d3w(f[2], order),
            ],
        )
        cg = jnp.take(coeffs, i_int[0] + offsets, axis=1, mode="wrap")
        cg = jnp.take(cg, i_int[1] + offsets, axis=2, mode="wrap")
        cg = jnp.take(cg, i_int[2] + offsets, axis=3, mode="wrap")  # (nb,K,K,K)

        def E(a: int, b: int, c: int) -> jax.Array:
            return jnp.einsum("bpqr,p,q,r->b", cg, W0[a], W1[b], W2[c])

        gt = jnp.stack([E(1, 0, 0), E(0, 1, 0), E(0, 0, 1)], axis=1)  # (nb,3)
        H00, H11, H22, H01, H02, H12 = E(2, 0, 0), E(0, 2, 0), E(0, 0, 2), E(1, 1, 0), E(1, 0, 1), E(0, 1, 1)
        Ht = jnp.stack(
            [
                jnp.stack([H00, H01, H02], 1),
                jnp.stack([H01, H11, H12], 1),
                jnp.stack([H02, H12, H22], 1),
            ],
            axis=1,
        )  # (nb,3,3)
        grad_r = gt @ M  # (nb,3)
        H_r = jnp.einsum("bkl,kc,ld->bcd", Ht, M, M)  # (nb,3,3)

        def contract(
            H00: jax.Array,
            H11: jax.Array,
            H22: jax.Array,
            H01: jax.Array,
            H02: jax.Array,
            H12: jax.Array,
        ) -> jax.Array:
            return (
                a_mat[0, 0] * H00
                + a_mat[1, 1] * H11
                + a_mat[2, 2] * H22
                + 2.0 * (a_mat[0, 1] * H01 + a_mat[0, 2] * H02 + a_mat[1, 2] * H12)
            )

        gl_cols = []
        for d in range(3):
            v = M[:, d]  # (3,) dt_k/dr_d
            dH00 = E(3, 0, 0) * v[0] + E(2, 1, 0) * v[1] + E(2, 0, 1) * v[2]
            dH11 = E(1, 2, 0) * v[0] + E(0, 3, 0) * v[1] + E(0, 2, 1) * v[2]
            dH22 = E(1, 0, 2) * v[0] + E(0, 1, 2) * v[1] + E(0, 0, 3) * v[2]
            dH01 = E(2, 1, 0) * v[0] + E(1, 2, 0) * v[1] + E(1, 1, 1) * v[2]
            dH02 = E(2, 0, 1) * v[0] + E(1, 1, 1) * v[1] + E(1, 0, 2) * v[2]
            dH12 = E(1, 1, 1) * v[0] + E(0, 2, 1) * v[1] + E(0, 1, 2) * v[2]
            gl_cols.append(contract(dH00, dH11, dH22, dH01, dH02, dH12))
        gradlap_r = jnp.stack(gl_cols, axis=1)  # (nb,3)
        return grad_r, H_r, gradlap_r

    grad_r, H_r, gradlap_r = jax.vmap(eval_one)(jnp.asarray(r_carts, jnp.float64))  # (N,nb,3),(N,nb,3,3),(N,nb,3)
    grad = jnp.transpose(grad_r, (2, 1, 0))  # (3, nb, N)
    gradlap = jnp.transpose(gradlap_r, (2, 1, 0))  # (3, nb, N)
    hessian = jnp.transpose(H_r, (2, 3, 1, 0))  # (3, 3, nb, N)
    return grad, gradlap, hessian


# ----------------------------------------------------------------------------
# Complex-twist B-spline AO ELECTRON-position kinetic kernels (grad / grad-of-lap / Hessian)
# ----------------------------------------------------------------------------
# The complex-Bloch B-spline AO is psi_k(r) = e^{i k_cart . r} u_k(r) with u_k a cell-periodic
# separable spline (atomic_orbital.py AOs_bspline_complex_data._eval_u_core / _compute_AOs*). The
# real-Gamma kernel above computes the spatial (grad_u, Hessian Hu, grad-of-lap glu) of u_k via the
# SAME separable-spline E(a,b,c) machinery; the only twist ingredient is the Bloch-phase product
# rule (exactly as ao._pw_val_dval unifies real/complex PW by folding k into gk). With P=e^{i k.r},
# u the spline value, gu=grad u, Hu the Hessian, Lu=lap u, glu=grad(lap u):
#   grad psi_d       = P (gu_d + i k_d u)
#   hess psi_cd      = P (Hu_cd + i k_c gu_d + i k_d gu_c - k_c k_d u)
#   lap psi          = P (Lu + 2 i k.gu - |k|^2 u)
#   gradlap psi_d    = P (glu_d + 2 i (k^T Hu)_d - |k|^2 gu_d + i k_d (Lu + 2 i k.gu - |k|^2 u))
# All downstream kinetic contractions take Re(...) of the assembled product (holomorphic), exactly
# as the real-Gamma / complex-PW analytic kernels. u_k has NO nuclear dependence (dAO/dR=0), and the
# spline control points are non-variational (dAO/dc=0), so this covers r; eps (kinetic stress) and
# ln_Psi drift reuse the SAME assembled fields via kinetic._orbital_side_r_pbc / pbc_stress.
def _bspline_complex_kinetic_fields(aos: AOs_bspline_complex_data, r_carts: jax.Array) -> _KineticFields:
    """(grad, gradlap, hessian) for a complex-twist B-spline leaf, matching the real
    ``_bspline_kinetic_fields`` shapes -- grad/gradlap (3, num_ao, N), hessian (3, 3, num_ao, N) --
    but complex128, carrying the Bloch-phase e^{i k_cart . r} derivative. EAGER."""
    from jqmc.atomic_orbital import AOs_bspline_data

    order = aos.spline_order
    coeffs = aos._coeffs_cube_jnp  # (nb, N0, N1, N2) complex
    cell = aos.structure_data._cell_jnp.astype(jnp.float64)
    inv = _inv_3x3(cell)
    ns = jnp.asarray(aos.spline_shape, dtype=jnp.float64)
    metric = inv.T @ inv
    a_mat = (ns[:, None] * ns[None, :]) * metric
    M = ns[:, None] * inv.T  # (3, 3) M[k,c] = N_k inv[c,k] = dt_k/dr_c
    k_cart = aos._k_cart_jnp  # (3,) cartesian twist
    k2 = jnp.sum(k_cart * k_cart)
    start = -((order - 1) // 2)
    offsets = jnp.arange(start, start + order + 1, dtype=jnp.int32)

    def eval_one(r: jax.Array) -> _KineticFields:
        frac = r @ inv
        frac = frac - jnp.floor(frac)
        t = frac * ns
        i_int = jnp.floor(t).astype(jnp.int32)
        f = t - jnp.floor(t)
        w0, dw0, d20 = AOs_bspline_data._bspline_weights_jax(f[0], order)
        w1, dw1, d21 = AOs_bspline_data._bspline_weights_jax(f[1], order)
        w2, dw2, d22 = AOs_bspline_data._bspline_weights_jax(f[2], order)
        W0 = [w0, dw0, d20, _bspline_d3w(f[0], order)]
        W1 = [w1, dw1, d21, _bspline_d3w(f[1], order)]
        W2 = [w2, dw2, d22, _bspline_d3w(f[2], order)]
        cg = jnp.take(coeffs, i_int[0] + offsets, axis=1, mode="wrap")
        cg = jnp.take(cg, i_int[1] + offsets, axis=2, mode="wrap")
        cg = jnp.take(cg, i_int[2] + offsets, axis=3, mode="wrap")  # (nb,K,K,K) complex

        def E(a: int, b: int, c: int) -> jax.Array:
            return jnp.einsum("bpqr,p,q,r->b", cg, W0[a], W1[b], W2[c])

        u = E(0, 0, 0)  # (nb,) complex
        gt = jnp.stack([E(1, 0, 0), E(0, 1, 0), E(0, 0, 1)], axis=1)  # (nb,3)
        H00, H11, H22, H01, H02, H12 = E(2, 0, 0), E(0, 2, 0), E(0, 0, 2), E(1, 1, 0), E(1, 0, 1), E(0, 1, 1)
        Ht = jnp.stack(
            [
                jnp.stack([H00, H01, H02], 1),
                jnp.stack([H01, H11, H12], 1),
                jnp.stack([H02, H12, H22], 1),
            ],
            axis=1,
        )  # (nb,3,3)
        gu = gt @ M  # (nb,3) grad u
        Hu = jnp.einsum("bkl,kc,ld->bcd", Ht, M, M)  # (nb,3,3) Hessian u

        def contract(
            H00: jax.Array,
            H11: jax.Array,
            H22: jax.Array,
            H01: jax.Array,
            H02: jax.Array,
            H12: jax.Array,
        ) -> jax.Array:
            return (
                a_mat[0, 0] * H00
                + a_mat[1, 1] * H11
                + a_mat[2, 2] * H22
                + 2.0 * (a_mat[0, 1] * H01 + a_mat[0, 2] * H02 + a_mat[1, 2] * H12)
            )

        Lu = contract(H00, H11, H22, H01, H02, H12)  # (nb,) lap u = trace(Hu)
        gl_cols = []
        for d in range(3):
            v = M[:, d]  # (3,) dt_k/dr_d
            dH00 = E(3, 0, 0) * v[0] + E(2, 1, 0) * v[1] + E(2, 0, 1) * v[2]
            dH11 = E(1, 2, 0) * v[0] + E(0, 3, 0) * v[1] + E(0, 2, 1) * v[2]
            dH22 = E(1, 0, 2) * v[0] + E(0, 1, 2) * v[1] + E(0, 0, 3) * v[2]
            dH01 = E(2, 1, 0) * v[0] + E(1, 2, 0) * v[1] + E(1, 1, 1) * v[2]
            dH02 = E(2, 0, 1) * v[0] + E(1, 1, 1) * v[1] + E(1, 0, 2) * v[2]
            dH12 = E(1, 1, 1) * v[0] + E(0, 2, 1) * v[1] + E(0, 1, 2) * v[2]
            gl_cols.append(contract(dH00, dH11, dH22, dH01, dH02, dH12))
        glu = jnp.stack(gl_cols, axis=1)  # (nb,3) grad(lap u)

        phase = jnp.exp(1j * jnp.dot(r, k_cart))  # scalar
        grad_psi = phase * (gu + 1j * u[:, None] * k_cart[None, :])  # (nb,3)
        cross = 1j * (k_cart[None, :, None] * gu[:, None, :] + k_cart[None, None, :] * gu[:, :, None])  # (nb,3,3)
        kk = (k_cart[:, None] * k_cart[None, :])[None, :, :]  # (1,3,3)
        hess_psi = phase * (Hu + cross - u[:, None, None] * kk)  # (nb,3,3)
        kTHu = jnp.einsum("c,bcd->bd", k_cart, Hu)  # (nb,3)
        k_dot_gu = jnp.einsum("c,bc->b", k_cart, gu)  # (nb,)
        lap_bloch = Lu + 2j * k_dot_gu - k2 * u  # (nb,) = lap psi / phase
        gradlap_psi = phase * (glu + 2j * kTHu - k2 * gu + 1j * k_cart[None, :] * lap_bloch[:, None])  # (nb,3)
        return grad_psi, hess_psi, gradlap_psi

    grad_r, H_r, gradlap_r = jax.vmap(eval_one)(jnp.asarray(r_carts, jnp.float64))
    grad = jnp.transpose(grad_r, (2, 1, 0))  # (3, nb, N)
    gradlap = jnp.transpose(gradlap_r, (2, 1, 0))  # (3, nb, N)
    hessian = jnp.transpose(H_r, (2, 3, 1, 0))  # (3, 3, nb, N)
    return grad, gradlap, hessian


def bspline_complex_kinetic_stress_contrib(
    leaf: AOs_bspline_complex_data,
    CM_lap: jax.Array,
    CM_gou: jax.Array,
    r: jax.Array,
) -> jax.Array:
    """Complex-twist B-spline orbital-side kinetic stress dT/deps (3,3), mirroring
    pbc_stress._bspline_kinetic_stress_contrib for the real leaf.

    Under the reciprocal (metric) strain the electron and lattice scale together so frac = r.inv(cell)
    is invariant while k_cart -> F^-T k_cart (as G -> F^-T G for PW). The Bloch-phase AO kinetic fields
    then transform as covariant tensors -- the phase / k terms fold into grad psi and Hess psi exactly
    as k folds into gk for the complex plane wave -- giving::

        d(grad psi_a)/deps_pq = -delta_aq (grad psi)_p
        d(lap psi)/deps_pq    = -2 (Hess psi)[p,q]

    (the value side is 0: psi value is affine-invariant). At k=0 / phase=1 these reduce to the real
    ``_bspline_kinetic_stress_contrib`` (grad psi -> grad u, Hess psi -> Hu) bit-for-bit. Reuses the
    assembled Bloch kinetic fields; the caller takes Re(...)."""
    fields = _bspline_complex_kinetic_fields(leaf, r)
    grad, hessian = fields[0], fields[2]  # (3=p, nb, N), (3=p, 3=q, nb, N)
    lap_term = -2.0 * jnp.einsum("bi,pqbi->pq", jnp.asarray(CM_lap), hessian)  # CM_lap contracted with d(lap)/deps
    gou_term = -jnp.einsum("qbi,pbi->pq", jnp.asarray(CM_gou), grad)  # CM_gou contracted with d(grad)/deps
    return lap_term + gou_term


def bspline_t_fields(aos: AOs_bspline_data, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """t-space spline derivatives (gt, Ht) per electron for the B-spline kinetic STRESS, where only
    the metric (a_mat / inv) responds to the affine strain (frac = r.inv(cell) is invariant). Returns
    gt (N, nb, 3) [d/dt_k] and Ht (N, nb, 3, 3) [d^2/dt_k dt_l]; the spatial conversion M = N_k inv[c,k]
    and its strain derivative are applied in pbc_stress._bspline_kinetic_stress_contrib."""
    order = aos.spline_order
    coeffs = aos._coeffs_cube_jnp
    cell = aos.structure_data._cell_jnp.astype(jnp.float64)
    inv = _inv_3x3(cell)
    ns = jnp.asarray(aos.spline_shape, dtype=jnp.float64)
    start = -((order - 1) // 2)
    offsets = jnp.arange(start, start + order + 1, dtype=jnp.int32)

    def eval_one(r: jax.Array) -> _ArrayPair:
        frac = r @ inv
        frac = frac - jnp.floor(frac)
        t = frac * ns
        i_int = jnp.floor(t).astype(jnp.int32)
        f = t - jnp.floor(t)
        w0, dw0, d20 = aos._bspline_weights_jax(f[0], order)
        w1, dw1, d21 = aos._bspline_weights_jax(f[1], order)
        w2, dw2, d22 = aos._bspline_weights_jax(f[2], order)
        W0, W1, W2 = [w0, dw0, d20], [w1, dw1, d21], [w2, dw2, d22]
        cg = jnp.take(coeffs, i_int[0] + offsets, axis=1, mode="wrap")
        cg = jnp.take(cg, i_int[1] + offsets, axis=2, mode="wrap")
        cg = jnp.take(cg, i_int[2] + offsets, axis=3, mode="wrap")

        def E(a: int, b: int, c: int) -> jax.Array:
            return jnp.einsum("bpqr,p,q,r->b", cg, W0[a], W1[b], W2[c])

        gt = jnp.stack([E(1, 0, 0), E(0, 1, 0), E(0, 0, 1)], axis=1)  # (nb,3)
        H00, H11, H22, H01, H02, H12 = E(2, 0, 0), E(0, 2, 0), E(0, 0, 2), E(1, 1, 0), E(1, 0, 1), E(0, 1, 1)
        Ht = jnp.stack(
            [jnp.stack([H00, H01, H02], 1), jnp.stack([H01, H11, H12], 1), jnp.stack([H02, H12, H22], 1)], axis=1
        )  # (nb,3,3)
        return gt, Ht

    gt, Ht = jax.vmap(eval_one)(jnp.asarray(r_carts, jnp.float64))
    return gt, Ht, inv, ns

    dval, dgx, dgy, dgz, dlap = jax.vmap(eval_one)(r_carts)  # each (N_e, n_band)
    return dval.T, dgx.T, dgy.T, dgz.T, dlap.T


# ----------------------------------------------------------------------------
# R5 -- pgto / sphe radial basis-parameter (exp / coeff) derivative
# ----------------------------------------------------------------------------
# The contracted spherical AO value/grad/lap (atomic_orbital.py:
# _compute_AOs_value_grad_lap_sphe) is c_p times the unit-coefficient primitive
#   U = N_n(Z_p) * exp(-Z_p s^2) * N_l_m * S_l_m
# with s^2 = |r - R_p|^2 and the gaussian normalization N_n(Z). The radial
# leaves (AO_radial_shell_data.exponents / coefficients) are per-shell-prim and
# expand to per-AO-prim via ao_prim_to_shell_prim. autograd already carries
# these gradients structurally (jnp.take VJP); R5 is the closed (Rust-portable)
# form. With kappa = (2l+3)/(4 Z) - s^2 (the d ln N_n/dZ term minus the
# exp term), the per-primitive directional derivatives are
#   d val  = U * (kappa c dZ + dc)
#   d grad = grad(U) * (kappa c dZ + dc) - 2 U (r-R) c dZ
#   d lap  = lap(U)  * (kappa c dZ + dc) + (U (8 Z s^2 - 6)
#                                           - 4 pref (S_grad . (r-R))) c dZ
# reduced over primitives (and contracted) to the AO axis. (pgto adds only a
# param-independent, linear image fold on top, so the same per-prim formulas
# carry through.)


def sphe_basis_param_jvp(
    aos: AOs_sphe_gto_data,
    r_carts: jax.Array,
    d_exponents: jax.Array,
    d_coefficients: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Directional derivative of the sphe AO (value, grad, lap) along radial
    (exp, coeff) perturbations.

    Args:
        aos: ``AOs_sphe_gto_data`` (molecular) leaf.
        r_carts: electron coordinates ``(N_e, 3)``.
        d_exponents: tangent on ``radial.exponents`` (per shell-prim).
        d_coefficients: tangent on ``radial.coefficients`` (per shell-prim).

    Returns:
        ``(dval, dgx, dgy, dgz, dlap)``, each ``(num_ao, N_e)``. Matches
        ``jax.jvp`` of ``_compute_AOs_value_grad_lap`` w.r.t. (exp, coeff).
    """
    dtype = jnp.float64
    R_prim = aos._atomic_center_carts_prim_jnp
    R_uniq = aos._atomic_center_carts_unique_jnp
    r_R = (r_carts[None, :, :] - R_prim[:, None, :]).astype(dtype)  # (num_prim, N_e, 3)
    r_R_uq = (r_carts[None, :, :] - R_uniq[:, None, :]).astype(dtype)
    s2 = jnp.sum(r_R**2, axis=-1)  # (num_prim, N_e)

    c = aos._coefficients_jnp.astype(dtype)  # (num_prim,)
    Z = aos._exponents_jnp.astype(dtype)
    l = aos._angular_momentums_prim_jnp
    m = aos._magnetic_quantum_numbers_prim_jnp
    nucleus_index = aos._nucleus_index_prim_jnp
    l_f = l.astype(dtype)

    fac_l1 = jnp.exp(jscipy.special.gammaln(l_f + 2.0))
    fac_2l2 = jnp.exp(jscipy.special.gammaln(2.0 * l_f + 3.0))
    N_n = jnp.sqrt((2.0 ** (2 * l_f + 3) * fac_l1 * (2 * Z) ** (l_f + 1.5)) / (fac_2l2 * jnp.sqrt(jnp.pi)))
    N_l_m = jnp.sqrt((2 * l_f + 1) / (4 * jnp.pi))
    R_n = jnp.exp(-Z[:, None] * s2)

    n_lm = (int(aos._angular_momentums_prim_np.max()) + 1) ** 2
    S_vals, S_grads, S_laps = _compute_S_l_m_and_grad_lap(r_R_uq, n_lm=n_lm)
    S_vals = S_vals.reshape((S_vals.shape[0] * S_vals.shape[1], S_vals.shape[2]))
    S_grads = S_grads.reshape((S_grads.shape[0] * S_grads.shape[1], S_grads.shape[2], 3))
    glm = l**2 + (m + l)
    idx = nucleus_index * n_lm + glm
    S = S_vals[idx]  # (num_prim, N_e)
    S_grad = S_grads[idx]  # (num_prim, N_e, 3)

    pref = N_n[:, None] * R_n * N_l_m[:, None]  # (num_prim, N_e)
    AOs_dup = pref * S
    grad_dup = AOs_dup[..., None] * (-2.0 * Z[:, None, None] * r_R) + pref[..., None] * S_grad
    grad_S_dot_r = jnp.sum(S_grad * r_R, axis=-1)
    lap_dup = (
        pref * S_laps.reshape((S_laps.shape[0] * S_laps.shape[1], S_laps.shape[2]))[idx]
        + AOs_dup * (4.0 * (Z[:, None] ** 2) * s2 - 6.0 * Z[:, None])
        - 4.0 * Z[:, None] * pref * grad_S_dot_r
    )

    # tangents expanded to per-AO-prim
    a2sp = jnp.asarray(aos.ao_prim_to_shell_prim, dtype=jnp.int32)
    dZ = jnp.asarray(d_exponents, dtype=dtype)[a2sp]  # (num_prim,)
    dc = jnp.asarray(d_coefficients, dtype=dtype)[a2sp] * aos._coefficient_scales_prim_jnp
    kappa = (2.0 * l_f[:, None] + 3.0) / (4.0 * Z[:, None]) - s2  # (num_prim, N_e)
    weighted_dZ = (c * dZ)[:, None]
    base = kappa * weighted_dZ + dc[:, None]  # (num_prim, N_e)

    d_val_dup = AOs_dup * base
    d_grad_dup = grad_dup * base[..., None] - 2.0 * AOs_dup[..., None] * r_R * weighted_dZ[..., None]
    d_lap_dup = lap_dup * base + (AOs_dup * (8.0 * Z[:, None] * s2 - 6.0) - 4.0 * pref * grad_S_dot_r) * weighted_dZ

    dval = _reduce_primitives_to_aos(d_val_dup, aos)
    dgx = _reduce_primitives_to_aos(d_grad_dup[..., 0], aos)
    dgy = _reduce_primitives_to_aos(d_grad_dup[..., 1], aos)
    dgz = _reduce_primitives_to_aos(d_grad_dup[..., 2], aos)
    dlap = _reduce_primitives_to_aos(d_lap_dup, aos)
    return dval, dgx, dgy, dgz, dlap
