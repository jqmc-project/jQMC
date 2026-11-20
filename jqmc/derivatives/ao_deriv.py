"""Autograd-free arbitrary-order derivatives for Cartesian-GTO AOs.

Each contracted Cartesian GTO primitive factorizes as
``N c * [x^nx e^{-Z x^2}][y^ny e^{-Z y^2}][z^nz e^{-Z z^2}]`` (x = coord - center),
so any mixed derivative ``d^ax_x d^ay_y d^az_z`` is a product of 1D Hermite-Gaussian
derivatives ``P_k(t;n,Z) e^{-Z t^2}`` with the recurrence ``P_{k+1} = P_k' - 2Z t P_k``
(closed forms k<=3 below). No autodiff, no finite differences -- exact, Rust-portable.
Normalization comes from the same primitive data as ``compute_AOs*`` so the
value, gradient, and Laplacian conventions are identical.
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

from jqmc.atomic_orbital import (
    AOs_base_data,
    AOs_cart_gto_data,
    _cart_max_polynomial_order,
    _int_pow_unrolled_cart,
    _reduce_primitives_to_aos,
)
from jqmc.molecular_orbital import MOs_data


def orb_to_cart_pieces(
    orb_data: AOs_base_data | MOs_data,
) -> tuple[AOs_cart_gto_data, jax.Array]:
    """Return (cart_aos_data, C) for a geminal orbital leaf, ``orb = C @ AO``."""
    tn = type(orb_data).__name__
    if tn == "MOs_data":
        aos = orb_data.aos_data
        if type(aos).__name__ == "AOs_cart_gto_data":
            return aos, jnp.asarray(orb_data._mo_coefficients_jnp)
        from jqmc.atomic_orbital import _aos_sphe_to_cart, _radial_is_traced

        if not _radial_is_traced(aos):
            oc = orb_data.to_cartesian()  # eager host path -- bit-identical (keeps its fp64 c2s dtype handling)
            return oc.aos_data, jnp.asarray(oc._mo_coefficients_jnp)
        # Traced-ham bridge: ``MOs_data.to_cartesian()`` folds the c2s transform into the MO
        # coefficients as a HOST numpy matmul (``np.asarray(mo_coefficients) @ T``), which raises
        # TracerArrayConversionError here (the MO matrix is a tracer). Do it in ``jnp`` instead:
        # ``_aos_sphe_to_cart`` self-dispatches to its jit-safe twin for the traced exponents and
        # ``T`` is a concrete c2s constant, so ``C_mo @ T`` is the identical coefficient transform.
        C_mo = jnp.asarray(orb_data._mo_coefficients_jnp)
        cart, T = _aos_sphe_to_cart(aos)
        return cart, C_mo @ jnp.asarray(np.asarray(T), dtype=C_mo.dtype)
    if tn == "AOs_cart_gto_data":
        return orb_data, jnp.eye(orb_data.num_ao)
    if tn == "AOs_sphe_gto_data":
        from jqmc.atomic_orbital import _aos_sphe_to_cart

        cart, T = _aos_sphe_to_cart(orb_data)
        return cart, jnp.asarray(np.asarray(T))
    # Defensive guard (currently unreachable): this molecular det cart-conversion is reached only
    # for a MOLECULAR geminal, whose leaves are MOs_data / cart_gto / sphe_gto (handled above).
    # pgto / plane-wave / composite leaves are periodic and route through the PBC orbital side, so a
    # "bare pgto AGP" never lands here. Kept as a guard for any future molecular AO leaf type.
    raise NotImplementedError(f"{tn}: molecular geminal orbital -> cart not implemented for this AO leaf type")


def _safe_pow(t: jax.Array, m: jax.Array) -> jax.Array:
    me = jnp.maximum(m, 0)
    val = t ** me[:, None]
    return jnp.where((m >= 0)[:, None], val, 0.0)


def _P(t: jax.Array, n: jax.Array, Z: jax.Array, k: int) -> jax.Array:
    """1D Hermite-Gaussian derivative polynomial P_k(t; n, Z) (no e^{-Zt^2})."""
    n = n.astype(jnp.float64)
    ni = n.astype(jnp.int32)
    if k == 0:
        return _safe_pow(t, ni)
    if k == 1:
        return n[:, None] * _safe_pow(t, ni - 1) - 2.0 * Z[:, None] * _safe_pow(t, ni + 1)
    if k == 2:
        return (
            n[:, None] * (n - 1)[:, None] * _safe_pow(t, ni - 2)
            - 2.0 * Z[:, None] * (2.0 * n + 1.0)[:, None] * _safe_pow(t, ni)
            + 4.0 * (Z**2)[:, None] * _safe_pow(t, ni + 2)
        )
    if k == 3:
        return (
            n[:, None] * (n - 1)[:, None] * (n - 2)[:, None] * _safe_pow(t, ni - 3)
            - 6.0 * Z[:, None] * (n**2)[:, None] * _safe_pow(t, ni - 1)
            + 12.0 * (Z**2)[:, None] * (n + 1)[:, None] * _safe_pow(t, ni + 1)
            - 8.0 * (Z**3)[:, None] * _safe_pow(t, ni + 3)
        )
    raise ValueError(k)


def _prim_norm(aos: AOs_cart_gto_data) -> jax.Array:
    Z = jnp.asarray(aos._exponents_jnp, jnp.float64)
    l = jnp.asarray(aos._angular_momentums_prim_jnp, jnp.float64)
    fac = jnp.asarray(aos._normalization_factorial_ratio_prim_jnp, jnp.float64)
    N_Z = (2.0 * Z / jnp.pi) ** 1.5 * (8.0 * Z) ** l
    return jnp.sqrt(N_Z * fac)


AxisOrder = tuple[int, int, int]

# Cartesian derivative orders (ax, ay, az) of the standard AO fields.
GRAD_ORDERS: tuple[AxisOrder, ...] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
LAP_ORDERS: tuple[AxisOrder, ...] = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
# grad(laplacian): row c holds the three terms of d_c (d_xx + d_yy + d_zz)
GRADLAP_ORDERS: tuple[tuple[AxisOrder, ...], ...] = (
    ((3, 0, 0), (1, 2, 0), (1, 0, 2)),
    ((2, 1, 0), (0, 3, 0), (0, 1, 2)),
    ((2, 0, 1), (0, 2, 1), (0, 0, 3)),
)
# Hessian [i][j] (symmetric)
HESS_ORDERS: tuple[tuple[AxisOrder, ...], ...] = (
    ((2, 0, 0), (1, 1, 0), (1, 0, 1)),
    ((1, 1, 0), (0, 2, 0), (0, 1, 1)),
    ((1, 0, 1), (0, 1, 1), (0, 0, 2)),
)
GRAD_GRADLAP_HESS_ORDERS: tuple[AxisOrder, ...] = tuple(
    dict.fromkeys(GRAD_ORDERS + sum(GRADLAP_ORDERS, ()) + sum(HESS_ORDERS, ())).keys()
)


def _hermite_z_coeffs(t: jax.Array, n: jax.Array, k: int, pw: dict[int, jax.Array]) -> list[jax.Array]:
    """``P_k(t; n, Z) = sum_j Z**j A[j]``: the Z-polynomial coefficients of :func:`_P` (no
    ``e^{-Z t^2}``) for AO-rank ``t`` (num_ao, N) and ``n`` (num_ao,). ``pw[m] = t**m`` (0 for m < 0)."""
    n = n.astype(jnp.float64)[:, None]
    if k == 0:
        return [pw[0]]
    if k == 1:
        return [n * pw[-1], -2.0 * pw[1]]
    if k == 2:
        return [n * (n - 1.0) * pw[-2], -2.0 * (2.0 * n + 1.0) * pw[0], 4.0 * pw[2]]
    if k == 3:
        return [n * (n - 1.0) * (n - 2.0) * pw[-3], -6.0 * n**2 * pw[-1], 12.0 * (n + 1.0) * pw[1], -8.0 * pw[3]]
    raise ValueError(k)


def ao_deriv_set(
    aos: AOs_cart_gto_data,
    r_carts: jax.Array,
    orders: tuple[AxisOrder, ...],
) -> jax.Array:
    """Cartesian-GTO AO derivatives of every order in ``orders`` in ONE primitive sweep.

    Returns shape ``(len(orders), num_ao_kernel, N)`` (``num_ao_kernel == num_ao`` for a
    molecular leaf; the kernel rank before the image fold for an image-tiled pGTO leaf, see
    :func:`jqmc.derivatives.pgto_deriv.pgto_deriv_set`).

    The primitives of a contracted AO share its center and angular powers, so every derivative
    ``prod_i P_{a_i}(t_i; n_i, Z) e^{-Z r^2}`` is a polynomial of degree ``a_x + a_y + a_z`` in Z
    with AO-level coefficients.  As in the e_L value/grad/lap kernel, only the radial moments
    ``S_j = sum_p N_p c_p Z_p^j e^{-Z_p r^2}`` (j <= 3) are formed at primitive rank and reduced to
    AOs; all orders are then assembled at AO rank from ``S_j`` and the AO-level polynomials.
    """
    orders = tuple(tuple(int(v) for v in o) for o in orders)
    j_max = max(sum(o) for o in orders)
    Z = jnp.asarray(aos._exponents_jnp, jnp.float64)
    c = jnp.asarray(aos._coefficients_jnp, jnp.float64)
    Nrm = _prim_norm(aos)
    R_prim = jnp.asarray(aos._atomic_center_carts_prim_jnp, jnp.float64)
    r = jnp.asarray(r_carts, jnp.float64)
    d_prim = r[None, :, :] - R_prim[:, None, :]
    rad = (Nrm * c)[:, None] * jnp.exp(-Z[:, None] * jnp.sum(d_prim**2, axis=-1))  # (n_prim, N)
    S = [_reduce_primitives_to_aos(rad * Z[:, None] ** j if j else rad, aos) for j in range(j_max + 1)]  # (n_ao, N)

    # AO-rank displacement and angular powers
    d = r[None, :, :] - jnp.asarray(aos._atomic_center_carts_jnp, jnp.float64)[:, None, :]  # (n_ao, N, 3)
    n_axis = [jnp.asarray(np.asarray(p)) for p in (aos.polynominal_order_x, aos.polynominal_order_y, aos.polynominal_order_z)]
    # t**(n+m) by a static unroll (a traced integer exponent array would lower ``**`` to an XLA
    # repeated-squaring while-loop); negative powers only occur with a zero coefficient.
    l_max = _cart_max_polynomial_order(aos)
    coeffs = []
    for i in range(3):
        t, ni = d[..., i], n_axis[i].astype(jnp.int32)
        ks = sorted({o[i] for o in orders})
        pw = {
            m: jnp.where((ni + m >= 0)[:, None], _int_pow_unrolled_cart(t, jnp.maximum(ni + m, 0), l_max + max(ks)), 0.0)
            for m in range(-max(ks), max(ks) + 1)
        }
        coeffs.append({k: _hermite_z_coeffs(t, n_axis[i], k, pw) for k in ks})
    out = []
    for ax, ay, az in orders:
        terms = jnp.zeros_like(S[0])
        for jx, a in enumerate(coeffs[0][ax]):
            for jy, b in enumerate(coeffs[1][ay]):
                for jz, e in enumerate(coeffs[2][az]):
                    terms = terms + S[jx + jy + jz] * (a * b * e)
        out.append(terms)
    return jnp.stack(out, 0)


def fields_from_deriv_set(deriv_set: jax.Array, orders: tuple[AxisOrder, ...]) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(grad (3, nao, N), gradlap (3, nao, N), hessian (3, 3, nao, N)) from a ``*_deriv_set``
    output computed for ``orders`` (which must contain :data:`GRAD_GRADLAP_HESS_ORDERS`)."""
    at = {o: deriv_set[i] for i, o in enumerate(orders)}
    grad = jnp.stack([at[o] for o in GRAD_ORDERS], 0)
    gradlap = jnp.stack([at[a] + at[b] + at[c] for a, b, c in GRADLAP_ORDERS], 0)
    hess = jnp.stack([jnp.stack([at[o] for o in row], 0) for row in HESS_ORDERS], 0)
    return grad, gradlap, hess


def ao_deriv(
    aos: AOs_cart_gto_data,
    r_carts: jax.Array,
    order: AxisOrder,
) -> jax.Array:
    """Cartesian-GTO AO derivative of order (ax,ay,az); shape (num_ao, N)."""
    return ao_deriv_set(aos, r_carts, (order,))[0]


def ao_grad(aos: AOs_cart_gto_data, r: jax.Array) -> jax.Array:
    return ao_deriv_set(aos, r, GRAD_ORDERS)


def ao_laplacian(aos: AOs_cart_gto_data, r: jax.Array) -> jax.Array:
    return jnp.sum(ao_deriv_set(aos, r, LAP_ORDERS), axis=0)


def ao_gradlap(aos: AOs_cart_gto_data, r: jax.Array) -> jax.Array:
    """grad(laplacian AO): (3, num_ao, N) -- the 3rd-order term the kinetic force needs."""
    terms = ao_deriv_set(aos, r, sum(GRADLAP_ORDERS, ()))  # (9, nao, N), row-major (c, term)
    return jnp.sum(terms.reshape((3, 3) + terms.shape[1:]), axis=1)


def ao_hessian(aos: AOs_cart_gto_data, r: jax.Array) -> jax.Array:
    """(3, 3, num_ao, N) symmetric AO Hessian."""
    orders = HESS_ORDERS[0] + (HESS_ORDERS[1][1], HESS_ORDERS[1][2], HESS_ORDERS[2][2])
    at = dict(zip(orders, ao_deriv_set(aos, r, orders)))
    return jnp.stack([jnp.stack([at[o] for o in row], 0) for row in HESS_ORDERS], 0)


def ao_grad_gradlap_hessian(aos: AOs_cart_gto_data, r: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(grad (3, nao, N), gradlap (3, nao, N), hessian (3, 3, nao, N)) in one primitive sweep
    (the 18 distinct orders of :data:`GRAD_GRADLAP_HESS_ORDERS`)."""
    return fields_from_deriv_set(ao_deriv_set(aos, r, GRAD_GRADLAP_HESS_ORDERS), GRAD_GRADLAP_HESS_ORDERS)


_J3Fields = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


def j3_basis_derivs(aos: AOs_base_data | MOs_data, r: jax.Array) -> _J3Fields:
    """(value, grad, laplacian, hessian, gradlap) of the J3 orbital FIELD, sphe->cart aware.

    The three-body Jastrow consumes the real field ``Re(orb)`` (``Jastrow_three_body_data``
    Re-projects ``compute_orb_api``), and ``Re`` commutes with d/dr, so every returned field is
    ``Re`` of the matching derivative of the orbital:

    * a bare AO leaf (GTO / pGTO / plane wave / B-spline / composite): ``Re(d^n AO)``;
    * an MO-backed leaf (``MOs_data``, ``orb = C @ AO``): ``Re(C @ d^n AO)``, contracted on the
      RAW (pre-``Re``) AO tower of :func:`j3_ao_basis_derivs_raw`. For a complex ``C`` over an AO
      leaf with a nonzero ``Im AO`` (a twisted plane-wave / pGTO / B-spline leaf or composite
      child), ``Re(C) @ Re(AO)`` would drop ``-Im(C) @ Im(AO)``, so the projection must
      come after the MO contraction. For a real ``C`` (or a real AO tower) the two coincide.

    The fields are MO-rank for an MO leaf: the downstream kinetic basis contraction
    (basis._basis_contract_kinetic_general) chains the MO-level cotangents back through ``C`` to
    the AO radial params."""
    if type(aos).__name__ == "MOs_data":
        C = jnp.asarray(aos._mo_coefficients_jnp)  # (num_mo, num_ao)
        Av, Ag, Al, Ah, Agl = j3_ao_basis_derivs_raw(aos.aos_data, r)
        return (
            jnp.real(C @ Av),
            jnp.real(jnp.einsum("ma,dan->dmn", C, Ag)),
            jnp.real(C @ Al),
            jnp.real(jnp.einsum("ma,dean->demn", C, Ah)),
            jnp.real(jnp.einsum("ma,dan->dmn", C, Agl)),
        )
    return tuple(jnp.real(f) for f in j3_ao_basis_derivs_raw(aos, r))


def j3_ao_basis_derivs_raw(aos: AOs_base_data, r: jax.Array) -> _J3Fields:
    """RAW (pre-``Re``) field tower (value, grad, laplacian, hessian, gradlap) of any J3 AO leaf.

    Shapes: value/laplacian (num_ao, N), grad/gradlap (3, num_ao, N), hessian (3, 3, num_ao, N).
    Complex for a complex-Bloch (twist) leaf, real otherwise. :func:`j3_basis_derivs` takes ``Re``
    of it for a bare AO J3 and of ``C @`` it for an MO J3.

    * cart / sphe GTO and cart / sphe pGTO: :func:`j3_gto_basis_derivs_raw` (a sphe-pGTO leaf is
      de-tiled -> molecular c2s -> re-expanded, with the complex Bloch promotion for a twist);
    * plane wave (real Gamma or complex twist): ``grad[d] = G_d dval``, ``lap = -|G|^2 val``,
      ``hess[c,d] = -G_c G_d val``, ``gradlap = -|G|^2 grad`` (``ao.pw_*``);
    * B-spline (real Gamma or complex twist): value / Laplacian from the forward separable spline,
      grad / grad-of-Laplacian / Hessian from the shared kinetic kernels (ao._bspline_*_fields), as
      the geminal kinetic path does;
    * composite: the CONCATENATION of the child towers along the AO axis (axis -2 in all five
      shapes), mirroring ``AOs_composite_data._compute_AOs*`` and pgto_deriv.orbital_leaves."""
    tn = type(aos).__name__
    if tn == "AOs_composite_data":
        parts = []
        for child in aos.children:
            ctn = type(child).__name__
            if not ("plane_wave" in ctn or "gto" in ctn or "bspline" in ctn):
                # Defensive guard: every composite child the builder can emit is handled.
                raise NotImplementedError(f"j3_basis_derivs: composite child {ctn} not implemented")
            parts.append(j3_ao_basis_derivs_raw(child, r))
        return tuple(jnp.concatenate([p[i] for p in parts], axis=-2) for i in range(5))
    if "plane_wave" in tn:
        from . import ao as A

        G, val, _dval = A._pw_val_dval(aos, r)
        g2 = jnp.sum(G * G, axis=1)
        return val, A.pw_grad(aos, r), -g2[:, None] * val, A.pw_hessian(aos, r), A.pw_gradlap(aos, r)
    if tn in ("AOs_bspline_data", "AOs_bspline_complex_data"):
        # A B-spline band has no cart-vs-sphe form / radial basis params (dAO/dR = 0, dAO/dc = 0).
        from . import ao as A

        val, _gx, _gy, _gz, lap = aos._compute_AOs_value_grad_lap(r)
        kernel = A._bspline_complex_kinetic_fields if tn == "AOs_bspline_complex_data" else A._bspline_kinetic_fields
        grad, gradlap, hess = kernel(aos, r)
        return val, grad, lap, hess, gradlap
    if tn in ("AOs_cart_gto_data", "AOs_sphe_gto_data") or "pgto" in tn:
        return j3_gto_basis_derivs_raw(aos, r)
    # Defensive guard: every J3-orbital AO leaf type is handled above.
    raise NotImplementedError(f"{tn}: J3-orbital basis -> cart not implemented for this AO leaf type")


def j3_gto_basis_derivs_raw(aos: AOs_base_data, r: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Raw field tower for one atom-centred GTO/pGTO J3 leaf.

    A complex MO J3 needs these pre-``Re`` AO fields after its real MO-rank
    cotangents are mapped back through ``C.T``.  This preserves
    ``Re(C @ AO)`` and avoids the invalid ``Re(C) @ Re(AO)`` lowering.
    """
    tn = type(aos).__name__
    if tn == "AOs_cart_gto_data":
        return (
            ao_deriv(aos, r, (0, 0, 0)),
            ao_grad(aos, r),
            ao_laplacian(aos, r),
            ao_hessian(aos, r),
            ao_gradlap(aos, r),
        )
    if tn == "AOs_sphe_gto_data":
        from jqmc.atomic_orbital import _aos_sphe_to_cart

        cart, transform = _aos_sphe_to_cart(aos)
        transform = jnp.asarray(np.asarray(transform))
        return (
            transform @ ao_deriv(cart, r, (0, 0, 0)),
            jnp.einsum("sc,dcn->dsn", transform, ao_grad(cart, r)),
            transform @ ao_laplacian(cart, r),
            jnp.einsum("sc,decn->desn", transform, ao_hessian(cart, r)),
            jnp.einsum("sc,dcn->dsn", transform, ao_gradlap(cart, r)),
        )
    if "pgto" in tn:
        from . import pgto_deriv as P

        if "sphe" in tn:
            cart, transform = P.sphe_leaf_to_cart(aos)
            transform = jnp.asarray(np.asarray(transform))
            return (
                transform @ P.pgto_value(cart, r),
                jnp.einsum("sc,dcn->dsn", transform, P.pgto_grad(cart, r)),
                transform @ P.pgto_laplacian(cart, r),
                jnp.einsum("sc,decn->desn", transform, P.pgto_hessian(cart, r)),
                jnp.einsum("sc,dcn->dsn", transform, P.pgto_gradlap(cart, r)),
            )
        return (
            P.pgto_value(aos, r),
            P.pgto_grad(aos, r),
            P.pgto_laplacian(aos, r),
            P.pgto_hessian(aos, r),
            P.pgto_gradlap(aos, r),
        )
    raise NotImplementedError(f"j3_gto_basis_derivs_raw: unsupported AO leaf {tn}")
