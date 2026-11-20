"""Basis-parameter gradients for cart-GTO radial exponents / coefficients.

O_k  (d ln|det| / d radial): det value-cotangent . d AO/d radial.
dT/dc (kinetic, WITH Jastrow): the kinetic VJP cotangents (bar_orb on the orbital
VALUE, bar_lap on the Laplacian, bar_gou on the orbital GRADIENT -- the last is the
``-sum gJ.gD`` Jastrow cross term absent in the determinant-only cancellation)
contracted with the value / Laplacian / gradient sensitivities of the cart-GTO
primitive, scattered prim -> shell by ``ao_prim_to_shell_prim``. cart-GTO only.

For a unit-coefficient primitive U (N_p ~ Z^{(2l+3)/4}, a=(2l+3)/(4Z), d=r-R)::

  d(c U)/dZ        = c (a - r^2) U
  d(c lap U)/dZ    = c [a lap U - (r^2 lap U + 4 d.grad U + 6 U)]
  d(c grad_k U)/dZ = c [a grad_k U - (2 d_k U + r^2 grad_k U)]

The coefficient sensitivities are U, lap U and grad U, including at c=0.
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
import numpy as np
from jax import typing as jnpt

from jqmc._precision import get_dtype_jnp
from jqmc.atomic_orbital import AOs_base_data, AOs_cart_gto_data
from jqmc.determinant import AGP_data, AGPn_data, SD_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data
from jqmc.molecular_orbital import MOs_data

from .ao_deriv import _P, _prim_norm
from .fast import geminal_value_tables, resolve_geminal_inv

GeminalData: TypeAlias = AGP_data | AGPn_data | SD_data
RadialGradient: TypeAlias = tuple[jax.Array, jax.Array]
OrbitalData: TypeAlias = AOs_base_data | MOs_data
PrimitiveSensitivities: TypeAlias = tuple[jax.Array, RadialGradient, RadialGradient, RadialGradient]
PrimitiveValueFields: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
J3ValueCotangents: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def _orb_pieces(orb_data: OrbitalData) -> tuple[AOs_base_data, jax.Array | None]:
    if type(orb_data).__name__ == "MOs_data":
        return orb_data.aos_data, jnp.asarray(orb_data._mo_coefficients_jnp)
    return orb_data, None


def _prim_sens(aos: AOs_cart_gto_data, r: jax.Array) -> PrimitiveSensitivities:
    """oi, (value/lap/grad sensitivities) for the cart-GTO primitives. Each spatial (P, N)."""
    Z = jnp.asarray(aos._exponents_jnp, jnp.float64)
    c = jnp.asarray(aos._coefficients_jnp, jnp.float64)
    Nrm = _prim_norm(aos)
    R = jnp.asarray(aos._atomic_center_carts_prim_jnp, jnp.float64)
    oi = jnp.asarray(aos._orbital_indices_np)
    l = jnp.asarray(aos._angular_momentums_prim_jnp, jnp.float64)
    nx = jnp.asarray(np.asarray(aos.polynominal_order_x))[oi]
    ny = jnp.asarray(np.asarray(aos.polynominal_order_y))[oi]
    nz = jnp.asarray(np.asarray(aos.polynominal_order_z))[oi]
    d = jnp.asarray(r, jnp.float64)[None, :, :] - R[:, None, :]
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    r2 = x**2 + y**2 + z**2
    rad = Nrm[:, None] * jnp.exp(-Z[:, None] * r2)

    def D(ax: int, ay: int, az: int) -> jax.Array:
        return rad * _P(x, nx, Z, ax) * _P(y, ny, Z, ay) * _P(z, nz, Z, az)

    sc_val = D(0, 0, 0)
    sc_grad = jnp.stack([D(1, 0, 0), D(0, 1, 0), D(0, 0, 1)], 0)  # (3, P, N)
    sc_lap = D(2, 0, 0) + D(0, 2, 0) + D(0, 0, 2)
    a = ((2.0 * l + 3.0) / (4.0 * Z))[:, None]
    dk = jnp.stack([x, y, z], 0)  # (3, P, N)
    dgrad = (dk * sc_grad).sum(0)  # d . grad U (P, N)
    sZ_val = c[:, None] * (a - r2) * sc_val
    sZ_lap = c[:, None] * ((a - r2) * sc_lap - 4.0 * dgrad - 6.0 * sc_val)
    sZ_grad = c[None, :, None] * ((a - r2)[None] * sc_grad - 2.0 * dk * sc_val[None])
    return oi, (sZ_val, sc_val), (sZ_lap, sc_lap), (sZ_grad, sc_grad)


def _basis_contract(
    orb_data: OrbitalData,
    bar_orb: jax.Array | None,
    bar_lap: jax.Array | None,
    bar_gou: jax.Array | None,
    r: jax.Array,
) -> RadialGradient:
    """Contract orbital-level cotangents (any may be None) with the basis sensitivities
    -> (d/d radial.exponents, d/d radial.coefficients), each shell-level (n_shell_prim,)."""
    aos, C = _orb_pieces(orb_data)
    oi, (sZv, scv), (sZl, scl), (sZg, scg) = _prim_sens(aos, r)
    bo = bar_orb if (C is None or bar_orb is None) else C.T @ bar_orb  # (num_ao, N)
    bl = bar_lap if (C is None or bar_lap is None) else C.T @ bar_lap
    bg = bar_gou if (C is None or bar_gou is None) else jnp.einsum("ma,dmN->daN", C, bar_gou)
    gZ = jnp.zeros(oi.shape[0])
    gc = jnp.zeros(oi.shape[0])
    if bo is not None:
        gZ = gZ + jnp.sum(bo[oi] * sZv, axis=1)
        gc = gc + jnp.sum(bo[oi] * scv, axis=1)
    if bl is not None:
        gZ = gZ + jnp.sum(bl[oi] * sZl, axis=1)
        gc = gc + jnp.sum(bl[oi] * scl, axis=1)
    if bg is not None:
        gZ = gZ + jnp.sum(bg[:, oi] * sZg, axis=(0, 2))
        gc = gc + jnp.sum(bg[:, oi] * scg, axis=(0, 2))
    n_sh = aos.radial.exponents.shape[0]  # static shape (no np.asarray -> jit-safe)
    idx = jnp.asarray(aos.ao_prim_to_shell_prim, jnp.int32)
    return (
        jax.ops.segment_sum(jnp.real(gZ), idx, num_segments=n_sh),
        jax.ops.segment_sum(jnp.real(gc) * aos._coefficient_scales_prim_jnp, idx, num_segments=n_sh),
    )


def _cart_prim_value(aos: AOs_base_data, r: jax.Array) -> PrimitiveValueFields:
    """Per unit-coefficient cart-GTO primitive value (radial x cart-poly) and metadata."""
    Z = jnp.asarray(aos._exponents_jnp, jnp.float64)
    c = jnp.asarray(aos._coefficients_jnp, jnp.float64)
    Nrm = _prim_norm(aos)
    R = jnp.asarray(aos._atomic_center_carts_prim_jnp, jnp.float64)
    oi = jnp.asarray(aos._orbital_indices_np)
    l = jnp.asarray(aos._angular_momentums_prim_jnp, jnp.float64)
    nx = jnp.asarray(np.asarray(aos.polynominal_order_x))[oi]
    ny = jnp.asarray(np.asarray(aos.polynominal_order_y))[oi]
    nz = jnp.asarray(np.asarray(aos.polynominal_order_z))[oi]
    d = jnp.asarray(r, jnp.float64)[None, :, :] - R[:, None, :]
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    r2 = x**2 + y**2 + z**2
    prim = Nrm[:, None] * jnp.exp(-Z[:, None] * r2) * _P(x, nx, Z, 0) * _P(y, ny, Z, 0) * _P(z, nz, Z, 0)
    return oi, prim, r2, Z, c, l


def _sphe_prim_value(aos: AOs_base_data, r: jax.Array) -> PrimitiveValueFields:
    """Per unit-coefficient spherical-GTO primitive value and metadata.

    Keep the normalization, radial Gaussian, and AO-level solid harmonic at
    primitive rank. The same factors apply at image-expanded kernel rank for pGTO.
    """
    import jax.scipy as jscipy

    from jqmc.atomic_orbital import _compute_S_l_m

    Z = jnp.asarray(aos._exponents_jnp, jnp.float64)
    c = jnp.asarray(aos._coefficients_jnp, jnp.float64)
    l = jnp.asarray(aos._angular_momentums_prim_jnp, jnp.float64)
    oi = jnp.asarray(aos._orbital_indices_np)
    R_uq = jnp.asarray(aos._atomic_center_carts_unique_jnp, jnp.float64)
    r_R_uq = jnp.asarray(r, jnp.float64)[None, :, :] - R_uq[:, None, :]  # (n_uq, N, 3)
    fac_l1 = jnp.exp(jscipy.special.gammaln(l + 2.0))
    fac_2l2 = jnp.exp(jscipy.special.gammaln(2.0 * l + 3.0))
    N_n = jnp.sqrt((2.0 ** (2 * l + 3) * fac_l1 * (2 * Z) ** (l + 1.5)) / (fac_2l2 * jnp.sqrt(jnp.pi)))
    N_lm = jnp.sqrt((2 * l + 1) / (4 * jnp.pi))
    r2 = jnp.sum(r_R_uq**2, axis=-1)[jnp.asarray(aos._nucleus_index_prim_jnp)]  # (P, N)
    NR_dup = N_n[:, None] * N_lm[:, None] * jnp.exp(-Z[:, None] * r2)
    n_lm = (int(np.asarray(aos._angular_momentums_prim_np).max()) + 1) ** 2
    max_ml, S = _compute_S_l_m(r_R_uq, n_lm=n_lm)
    S_resh = S.reshape((S.shape[0] * S.shape[1], S.shape[2]), order="F")
    nuc_ao = jnp.asarray(aos._nucleus_index_jnp)
    l_ao = jnp.asarray(np.asarray(aos.angular_momentums), dtype=jnp.int32)
    m_ao = jnp.asarray(np.asarray(aos.magnetic_quantum_numbers), dtype=jnp.int32)
    gidx = nuc_ao * max_ml + l_ao**2 + (m_ao + l_ao)
    S_ao = S_resh[gidx]  # (num_ao_kernel, N)
    prim = NR_dup * S_ao[oi]
    return oi, prim, r2, Z, c, l


def _sphe_value_adjoint_fields(aos: AOs_base_data, r: jax.Array) -> tuple[jax.Array, ...]:
    """Mixed forward-value factors, with angular factors kept at AO rank.

    Follow the forward AO precision boundary: subtract storage-precision
    coordinates before casting their difference to ``ao_eval``. Harmonics are
    evaluated once per distinct (atom, image), using the corresponding primitive
    AND AO inverse indices. This avoids repeating the same harmonic table for
    every shell/angular component of a periodic kernel.

    The radial factor has unit coefficient. Keeping its angular factor separate
    lets the adjoint multiply the AO cotangent before primitive gathering, just
    as the forward path contracts radial primitives before applying harmonics.
    """
    import jax.scipy as jscipy

    from jqmc.atomic_orbital import _compute_S_l_m

    dtype = get_dtype_jnp(aos._value_zone)
    Z = jnp.asarray(aos._exponents_jnp, dtype)
    c = jnp.asarray(aos._coefficients_jnp, dtype)
    l = jnp.asarray(aos._angular_momentums_prim_jnp, dtype)
    oi = jnp.asarray(aos._orbital_indices_np)
    centers = jnp.asarray(aos._uq_center_carts_jnp, jnp.float64)
    displacement = (jnp.asarray(r, jnp.float64)[None, :, :] - centers[:, None, :]).astype(dtype)
    # The radial normalization N_n does not depend on the electron coordinates:
    # form it from the fp64 storage exponents and round it ONCE to the ao_eval
    # zone, as the forward AO kernels do. Assembled from zone-dtype pieces it
    # overflowed float32 (inf d AO/d a and d AO/d c) although N_n and the AO
    # value are representable (issue #111).
    Z_storage = jnp.asarray(aos._exponents_jnp, jnp.float64)
    l_storage = jnp.asarray(aos._angular_momentums_prim_jnp, jnp.float64)
    fac_l1 = jnp.exp(jscipy.special.gammaln(l_storage + 2.0))
    fac_2l2 = jnp.exp(jscipy.special.gammaln(2.0 * l_storage + 3.0))
    N_n = jnp.sqrt(
        (2.0 ** (2 * l_storage + 3) * fac_l1 * (2 * Z_storage) ** (l_storage + 1.5)) / (fac_2l2 * jnp.sqrt(jnp.pi))
    ).astype(dtype)
    N_lm = jnp.sqrt((2 * l + 1) / (4 * jnp.pi))
    r2 = jnp.sum(displacement**2, axis=-1)[jnp.asarray(aos._uq_center_index_prim_jnp)]
    radial = N_n[:, None] * N_lm[:, None] * jnp.exp(-Z[:, None] * r2)
    n_lm = (int(np.asarray(aos._angular_momentums_prim_np).max()) + 1) ** 2
    max_ml, harmonics = _compute_S_l_m(displacement, n_lm=n_lm)
    harmonics = harmonics.reshape((harmonics.shape[0] * harmonics.shape[1], harmonics.shape[2]), order="F")
    centers_ao = jnp.asarray(aos._uq_center_index_ao_jnp)
    l_ao = jnp.asarray(np.asarray(aos.angular_momentums), dtype=jnp.int32)
    m_ao = jnp.asarray(np.asarray(aos.magnetic_quantum_numbers), dtype=jnp.int32)
    angular = harmonics[centers_ao * max_ml + l_ao**2 + (m_ao + l_ao)]
    return oi, radial, r2, Z, c, l, angular


def _sphe_prim_sens(aos: AOs_base_data, r: jax.Array) -> PrimitiveSensitivities:
    """oi, (value/lap/grad sensitivities) for the native SPHERICAL-GTO primitives (kernel rank
    for a sphe-pgto leaf). Each spatial (P, N) / (3, P, N).

    The cart sensitivity closed forms in ``_prim_sens`` are angular-factor AGNOSTIC -- their
    derivation only uses that the angular factor is Z/c-independent (cart polynomial OR real solid
    harmonic), so the SAME formulas apply once the native sphe primitive value / gradient /
    laplacian are assembled. With the unit-coefficient ``pref = N_n N_lm e^{-Z s^2}``,
    ``u = r - R``, ``s^2 = |u|^2``,
    solid harmonic ``S`` (value ``S``, grad ``dS``, lap ``lS`` from ``_compute_S_l_m_and_grad_lap``):
        prim   = pref S
        grad_k = pref dS_k - 2 Z u_k prim
        lap    = pref lS + (4 Z^2 s^2 - 6 Z) prim - 4 Z pref (u.dS)
    These fields are the coefficient sensitivities. Multiply the corresponding
    exponent sensitivities by c, as in ``_prim_sens``."""
    import jax.scipy as jscipy

    from jqmc.atomic_orbital import _compute_S_l_m_and_grad_lap

    Z = jnp.asarray(aos._exponents_jnp, jnp.float64)
    c = jnp.asarray(aos._coefficients_jnp, jnp.float64)
    l = jnp.asarray(aos._angular_momentums_prim_jnp, jnp.float64)
    oi = jnp.asarray(aos._orbital_indices_np)
    R_prim = jnp.asarray(aos._atomic_center_carts_prim_jnp, jnp.float64)
    R_uq = jnp.asarray(aos._atomic_center_carts_unique_jnp, jnp.float64)
    rj = jnp.asarray(r, jnp.float64)
    u = rj[None, :, :] - R_prim[:, None, :]  # (P, N, 3)
    r_R_uq = rj[None, :, :] - R_uq[:, None, :]  # (n_uq, N, 3)
    s2 = jnp.sum(u**2, axis=-1)  # (P, N)

    fac_l1 = jnp.exp(jscipy.special.gammaln(l + 2.0))
    fac_2l2 = jnp.exp(jscipy.special.gammaln(2.0 * l + 3.0))
    N_n = jnp.sqrt((2.0 ** (2 * l + 3) * fac_l1 * (2 * Z) ** (l + 1.5)) / (fac_2l2 * jnp.sqrt(jnp.pi)))
    N_lm = jnp.sqrt((2 * l + 1) / (4 * jnp.pi))
    pref = (N_n * N_lm)[:, None] * jnp.exp(-Z[:, None] * s2)

    n_lm = (int(np.asarray(aos._angular_momentums_prim_np).max()) + 1) ** 2
    Sv, Sg, Sl = _compute_S_l_m_and_grad_lap(r_R_uq, n_lm=n_lm)  # (n_uq,n_lm,N), (n_uq,n_lm,N,3), (n_uq,n_lm,N)
    Sv = Sv.reshape((Sv.shape[0] * Sv.shape[1], Sv.shape[2]))  # C-order: nucleus*n_lm + global_l_m
    Sg = Sg.reshape((Sg.shape[0] * Sg.shape[1], Sg.shape[2], 3))
    Sl = Sl.reshape((Sl.shape[0] * Sl.shape[1], Sl.shape[2]))
    li = jnp.asarray(aos._angular_momentums_prim_jnp)
    mi = jnp.asarray(aos._magnetic_quantum_numbers_prim_jnp)
    gidx = jnp.asarray(aos._nucleus_index_prim_jnp) * n_lm + li**2 + (mi + li)
    S = Sv[gidx]  # (P, N)
    dS = Sg[gidx]  # (P, N, 3)
    lS = Sl[gidx]  # (P, N)

    sc_val = pref * S  # (P, N)
    grad_pn3 = pref[..., None] * dS - 2.0 * Z[:, None, None] * u * sc_val[..., None]  # (P, N, 3)
    sc_grad = jnp.transpose(grad_pn3, (2, 0, 1))  # (3, P, N)
    sc_lap = (
        pref * lS
        + (4.0 * (Z[:, None] ** 2) * s2 - 6.0 * Z[:, None]) * sc_val
        - 4.0 * Z[:, None] * pref * jnp.sum(u * dS, axis=-1)
    )
    dk = jnp.transpose(u, (2, 0, 1))  # (3, P, N)
    dgrad = (dk * sc_grad).sum(0)  # u . grad U (P, N)

    a = ((2.0 * l + 3.0) / (4.0 * Z))[:, None]
    sZ_val = c[:, None] * (a - s2) * sc_val
    sZ_lap = c[:, None] * ((a - s2) * sc_lap - 4.0 * dgrad - 6.0 * sc_val)
    sZ_grad = c[None, :, None] * ((a - s2)[None] * sc_grad - 2.0 * dk * sc_val[None])
    return oi, (sZ_val, sc_val), (sZ_lap, sc_lap), (sZ_grad, sc_grad)


def _pgto_unfold(
    native_aos: AOs_base_data,
    r_j: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """(phase, home, restore-or-None) for un-folding a molecular-AO cotangent to kernel rank --
    the adjoint of the forward image fold (+ wrap-restore for a complex leaf). All STATIC except
    ``restore`` (per-electron, Z/c-independent), so safe under the TRACED-hamiltonian jit bridge."""
    from .pgto_deriv import _is_complex_pgto, _pgto_kernel_image

    _T, phase, home = _pgto_kernel_image(native_aos)  # phase (M,), home (M,) kernel->mol-AO
    restore = native_aos._restore_phase(r_j, "ao_grad_lap") if _is_complex_pgto(native_aos) else None
    return phase, home, restore


def _value_general_leaf(
    native_aos: AOs_base_data,
    C_mo: jax.Array | None,
    bar_orb: jax.Array,
    r: jax.Array,
    *,
    holomorphic: bool = False,
) -> RadialGradient:
    """O_k VALUE-block basis sensitivity for ONE native sphe-gto / cart-pgto / sphe-pgto leaf.

    The determinant O_k needs only the orbital VALUE cotangent, and the radial sensitivity
    ``d AO/dZ = c (a - r2) U`` / ``d AO/dc = U`` for a unit-coefficient primitive U factors out of the Z/c-independent
    angular part (cart polynomial or solid harmonic), so we evaluate per PRIMITIVE on the NATIVE
    leaf -- never sphe->cart, which would change the radial shell rank. For a pgto leaf the radial
    params are SHARED across images: the cart/sphe primitive values are taken at KERNEL rank on the
    wrapped electrons, and the molecular-AO cotangent ``B = C_mo^T bar_orb`` is un-folded to kernel
    rank via the SAME (phase, home) the forward image fold uses plus the wrap-restore per-electron
    column; ``segment_sum`` over ``ao_prim_to_shell_prim`` then accumulates every image onto its
    (native) molecular shell. Returns ``(d/d exponents, d/d coefficients)``.

    ``holomorphic`` (default False): the DEFAULT Re-projects the shell grad -- the O_k every existing
    consumer (direct ln_Psi/c, J3 value, kinetic/stress/PW paths) expects (a real d ln|det|/d radial;
    bit-identical at real Gamma / molecular where the cotangents are already real). Pass ``True`` ONLY
    for the complex-Bloch (twist) ECP non-local dV_nl/dc, which needs the FULL holomorphic O_k =
    Re(...) + i Im(...) (Im = d arg(det)/d radial): it multiplies O_k by the complex per-grid weight
    before taking Re on the product, so real-projecting O_k too early drops the Im(wf_ratio).Im(O_k)
    cross term. Scoped so the shared kernel's output is unchanged for every non-ECP caller."""
    tn = type(native_aos).__name__
    is_pgto = "pgto" in tn
    is_sphe = "sphe" in tn
    r_j = jnp.asarray(r, jnp.float64)
    if is_pgto:
        from jqmc.structure import _wrap_into_cell_if_pbc

        r_eval = _wrap_into_cell_if_pbc(native_aos.structure_data, r_j)
    else:
        r_eval = r_j

    mixed_sphe = is_sphe and get_dtype_jnp(native_aos._value_zone) == jnp.float32
    if mixed_sphe:
        oi, sc, r2, Z, c, l, angular = _sphe_value_adjoint_fields(native_aos, r_eval)
    else:
        oi, sc, r2, Z, c, l = (_sphe_prim_value if is_sphe else _cart_prim_value)(native_aos, r_eval)
    a = ((2.0 * l + 3.0) / (4.0 * Z))[:, None]
    sZ = (a - r2) * (sc * c[:, None])  # d radial / d exponent

    # AO-level cotangent at the NATIVE molecular AO rank: orb = C_mo @ AO_native (C_mo=None => AGP).
    B = bar_orb if C_mo is None else (C_mo.T @ bar_orb)
    if is_pgto:
        phase, home, restore = _pgto_unfold(native_aos, r_j)
        bo = phase[:, None] * B[home, :]  # un-fold (adjoint of the image fold)
        if restore is not None:
            bo = bo * restore[None, :]  # per-electron wrap-restore column
    else:
        bo = B

    if mixed_sphe:
        cotangent_dtype = jnp.result_type(sZ.dtype, jnp.complex64) if jnp.iscomplexobj(bo) else sZ.dtype
        bo = bo.astype(cotangent_dtype) * angular
    gZ = jnp.sum(bo[oi] * sZ, axis=1)
    gc = jnp.sum(bo[oi] * sc, axis=1)
    if mixed_sphe:
        # The transpose of the forward parameter cast restores storage dtype
        # before the primitive/image contributions gather into shared shells.
        gZ = gZ.astype(jnp.result_type(native_aos.radial.exponents.dtype, gZ.dtype))
        gc = gc.astype(jnp.result_type(native_aos.radial.coefficients.dtype, gc.dtype))
    n_sh = native_aos.radial.exponents.shape[0]
    idx = jnp.asarray(native_aos.ao_prim_to_shell_prim, jnp.int32)
    proj = (lambda x: x) if holomorphic else jnp.real  # default Re -> unchanged for every non-ECP caller
    return (
        jax.ops.segment_sum(proj(gZ), idx, num_segments=n_sh),
        jax.ops.segment_sum(proj(gc) * native_aos._coefficient_scales_prim_jnp, idx, num_segments=n_sh),
    )


def _basis_contract_value_general(
    orb_data: OrbitalData,
    bar_orb: jax.Array,
    r: jax.Array,
    *,
    holomorphic: bool = False,
) -> RadialGradient:
    """O_k basis (VALUE block) for a sphe-gto / cart-pgto / sphe-pgto leaf, OR a composite (pgto(+)PW)
    orbital. A composite has no single ``radial``: only its GTO/pGTO children carry basis params (a PW
    / B-spline child contributes nothing), so iterate ``pgto_deriv.orbital_leaves`` and concatenate
    each contributing child's shell gradient in child order (the order ``get_basis_variational_blocks``
    publishes ``child{i}_basis_*`` blocks). Returns ``(d/d exponents, d/d coefficients)``."""
    native_aos, C_mo = _orb_pieces(orb_data)
    if type(native_aos).__name__ == "AOs_composite_data":
        from .pgto_deriv import orbital_leaves

        gZ_parts, gc_parts = [], []
        for leaf, C_leaf, kind in orbital_leaves(orb_data):
            if kind in ("pw", "bspline"):  # no radial basis params
                gZ_parts.append(jnp.zeros(0))
                gc_parts.append(jnp.zeros(0))
                continue
            gZc, gcc = _value_general_leaf(leaf, jnp.asarray(C_leaf), bar_orb, r, holomorphic=holomorphic)
            gZ_parts.append(gZc)
            gc_parts.append(gcc)
        return jnp.concatenate(gZ_parts), jnp.concatenate(gc_parts)
    return _value_general_leaf(native_aos, C_mo, bar_orb, r, holomorphic=holomorphic)


def _kinetic_general_leaf(
    native_aos: AOs_base_data,
    C_mo: jax.Array | None,
    bar_orb: jax.Array | None,
    bar_lap: jax.Array | None,
    bar_gou: jax.Array | None,
    r: jax.Array,
    *,
    holomorphic: bool = False,
) -> RadialGradient:
    """dT/dc basis sensitivity (value + Laplacian + orbital-gradient cotangents) for ONE native leaf
    -- cart-gto, sphe-gto, cart-pgto OR sphe-pgto -- generalizing ``_basis_contract`` past cart-gto.

    Uses the SAME closed-form value/lap/grad primitive sensitivities (``_prim_sens`` for a cart
    angular factor, ``_sphe_prim_sens`` for a solid-harmonic one -- both jit-safe, native-rank, no
    host-side sphe->cart). For a pgto leaf the sensitivities are taken at KERNEL rank on the wrapped
    electrons, and EACH cotangent (orbital / Laplacian / orbital-gradient) is un-folded to kernel rank
    via the same ``(phase, home)`` adjoint + wrap-restore column the value O_k path uses, then
    ``segment_sum`` over ``ao_prim_to_shell_prim`` accumulates the images onto the native shells."""
    tn = type(native_aos).__name__
    is_pgto = "pgto" in tn
    is_sphe = "sphe" in tn
    r_j = jnp.asarray(r, jnp.float64)
    if is_pgto:
        from jqmc.structure import _wrap_into_cell_if_pbc

        r_eval = _wrap_into_cell_if_pbc(native_aos.structure_data, r_j)
    else:
        r_eval = r_j

    oi, (sZv, scv), (sZl, scl), (sZg, scg) = (_sphe_prim_sens if is_sphe else _prim_sens)(native_aos, r_eval)

    # Molecular-AO cotangents (C_mo=None => bare-AO AGP leaf: cotangent already at AO rank).
    bo = bar_orb if (C_mo is None or bar_orb is None) else C_mo.T @ bar_orb  # (num_ao, N)
    bl = bar_lap if (C_mo is None or bar_lap is None) else C_mo.T @ bar_lap
    bg = bar_gou if (C_mo is None or bar_gou is None) else jnp.einsum("ma,dmN->daN", C_mo, bar_gou)  # (3, num_ao, N)
    if is_pgto:
        phase, home, restore = _pgto_unfold(native_aos, r_j)

        def _uf(B: jax.Array | None) -> jax.Array | None:  # (num_ao, N) -> (M, N)
            if B is None:
                return None
            b = phase[:, None] * B[home, :]
            return b if restore is None else b * restore[None, :]

        def _ufg(B: jax.Array | None) -> jax.Array | None:  # (3, num_ao, N) -> (3, M, N)
            if B is None:
                return None
            b = phase[None, :, None] * B[:, home, :]
            return b if restore is None else b * restore[None, None, :]

        bo, bl, bg = _uf(bo), _uf(bl), _ufg(bg)

    gZ = jnp.zeros(oi.shape[0])
    gc = jnp.zeros(oi.shape[0])
    if bo is not None:
        gZ = gZ + jnp.sum(bo[oi] * sZv, axis=1)
        gc = gc + jnp.sum(bo[oi] * scv, axis=1)
    if bl is not None:
        gZ = gZ + jnp.sum(bl[oi] * sZl, axis=1)
        gc = gc + jnp.sum(bl[oi] * scl, axis=1)
    if bg is not None:
        gZ = gZ + jnp.sum(bg[:, oi] * sZg, axis=(0, 2))
        gc = gc + jnp.sum(bg[:, oi] * scg, axis=(0, 2))
    n_sh = native_aos.radial.exponents.shape[0]
    idx = jnp.asarray(native_aos.ao_prim_to_shell_prim, jnp.int32)
    project = (lambda x: x) if holomorphic else jnp.real
    return (
        jax.ops.segment_sum(project(gZ), idx, num_segments=n_sh),
        jax.ops.segment_sum(project(gc) * native_aos._coefficient_scales_prim_jnp, idx, num_segments=n_sh),
    )


def _basis_contract_kinetic_general(
    orb_data: OrbitalData,
    bar_orb: jax.Array | None,
    bar_lap: jax.Array | None,
    bar_gou: jax.Array | None,
    r: jax.Array,
    *,
    holomorphic: bool = False,
) -> RadialGradient:
    """dT/dc basis (value + Laplacian + gradient cotangents) for a sphe-gto / pgto leaf OR a
    composite (pgto(+)PW). Composite: sum is per-child (only GTO/pGTO children carry radial params;
    PW / B-spline contribute zero) -- concatenate each contributing child's shell gradient in child
    order, exactly as the VALUE general path does for O_k."""
    native_aos, C_mo = _orb_pieces(orb_data)
    if type(native_aos).__name__ == "AOs_composite_data":
        from .pgto_deriv import orbital_leaves

        gZ_parts, gc_parts = [], []
        for leaf, C_leaf, kind in orbital_leaves(orb_data):
            if kind in ("pw", "bspline"):
                gZ_parts.append(jnp.zeros(0))
                gc_parts.append(jnp.zeros(0))
                continue
            gZc, gcc = _kinetic_general_leaf(
                leaf,
                jnp.asarray(C_leaf),
                bar_orb,
                bar_lap,
                bar_gou,
                r,
                holomorphic=holomorphic,
            )
            gZ_parts.append(gZc)
            gc_parts.append(gcc)
        return jnp.concatenate(gZ_parts), jnp.concatenate(gc_parts)
    return _kinetic_general_leaf(
        native_aos,
        C_mo,
        bar_orb,
        bar_lap,
        bar_gou,
        r,
        holomorphic=holomorphic,
    )


def _is_plain_cart_leaf(orb_data: OrbitalData) -> bool:
    aos, _ = _orb_pieces(orb_data)
    return type(aos).__name__ == "AOs_cart_gto_data"


def _det_value_cotangents(
    gd: GeminalData,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """d ln|det G| / d orb (value only): (W G^-1) for up, (G^-1 Q)[:N_dn]^T for dn."""
    M_dn, N_dn = gd.orb_num_dn, gd.num_electron_dn
    # Without a running inverse, G comes from the fused value table, as the kinetic
    # dT/d radial reverse pass (kinetic._geminal_reverse) inverts it (#762).
    orb_up, orb_dn = geminal_value_tables(gd, r_up, r_dn, geminal_inv)
    from jqmc.determinant import SD_data

    if isinstance(gd, SD_data):
        G = jnp.hstack([orb_up[:N_dn].T @ orb_dn, orb_up[N_dn:].T])
        G_inv = resolve_geminal_inv(G, geminal_inv)
        bar_orb_up = jnp.concatenate([orb_dn @ G_inv[:N_dn, :], G_inv[N_dn:, :]], axis=0)
        Q = orb_up[:N_dn].T
        bar_orb_dn = (G_inv @ Q)[:N_dn].T
    else:
        lam_p, lam_unp = jnp.hsplit(gd._lambda_matrix_jnp, [M_dn])
        W = jnp.hstack([lam_p @ orb_dn, lam_unp])
        G = orb_up.T @ W
        G_inv = resolve_geminal_inv(G, geminal_inv)  # fast: running SM inverse; else SVD-pinv
        bar_orb_up = W @ G_inv  # W G^-1
        Q = orb_up.T @ lam_p
        bar_orb_dn = (G_inv @ Q)[:N_dn].T
    return bar_orb_up, bar_orb_dn


def grad_ln_det_wrt_basis(
    gd: GeminalData,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    *,
    holomorphic: bool = False,
    geminal_inv: jnpt.ArrayLike | None = None,
) -> tuple[RadialGradient, RadialGradient]:
    """O_k basis: d ln|det| / d radial.{exponents,coefficients} (up, dn).

    Dispatch per orbital leaf: a plain cart-GTO leaf uses ``_basis_contract``
    (also serves the kinetic dT/dc path), while sphe-GTO and (real / complex) pgto leaves fold
    the cart value sensitivities through the c2s / image map in ``_basis_contract_value_sphe_pgto``.

    ``holomorphic`` (default False): Re-project the O_k -- the real d ln|det|/d radial every existing
    consumer (direct ln_Psi/c, stress, PW) expects. Pass ``True`` ONLY from the ECP non-local
    dV_nl/dc (assemble._dV_ecp_dc via _O_k) so the complex-Bloch (twist) O_k keeps its Im part for
    the complex per-grid-weight product. No-op at real Gamma / molecular (cotangents already real).
    """

    def _side(orb_data: OrbitalData, bar: jax.Array, r: jax.Array) -> RadialGradient:
        if _is_plain_cart_leaf(orb_data):
            return _basis_contract(orb_data, bar, None, None, r)
        return _basis_contract_value_general(orb_data, bar, r, holomorphic=holomorphic)

    bu, bd = _det_value_cotangents(gd, r_up, r_dn, geminal_inv=geminal_inv)
    return _side(gd.orb_data_up_spin, bu, r_up), _side(gd.orb_data_dn_spin, bd, r_dn)


def grad_T_wrt_basis(
    gd: GeminalData,
    jastrow_data: Jastrow_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    geminal_inv: jnpt.ArrayLike | None = None,
    *,
    holomorphic: bool = False,
) -> tuple[RadialGradient, RadialGradient]:
    """dT/dc basis (WITH Jastrow): kinetic VJP cotangents (value + Laplacian + orbital
    gradient) contracted with the basis sensitivities.

    Dispatch per orbital leaf, mirroring ``grad_ln_det_wrt_basis``: a plain cart-GTO leaf uses
    ``_basis_contract``, while sphe-GTO, (real / complex) pgto and composite
    (pgto(+)PW) leaves go through ``_basis_contract_kinetic_general`` (native value/lap/grad primitive
    sensitivities folded through the solid-harmonic assembly / image map)."""
    from jqmc.jastrow_factor import compute_grads_and_laplacian_Jastrow_part

    from .kinetic import _geminal_reverse

    def _side(
        orb_data: OrbitalData,
        bar_orb: jax.Array,
        bar_lap: jax.Array,
        bar_gou: jax.Array,
        r: jax.Array,
    ) -> RadialGradient:
        if _is_plain_cart_leaf(orb_data):
            return _basis_contract(orb_data, bar_orb, bar_lap, bar_gou, r)
        return _basis_contract_kinetic_general(
            orb_data,
            bar_orb,
            bar_lap,
            bar_gou,
            r,
            holomorphic=holomorphic,
        )

    gJ_up, gJ_dn, _, _ = compute_grads_and_laplacian_Jastrow_part(jastrow_data, r_up, r_dn)
    b = _geminal_reverse(gd, r_up, r_dn, gJ_up, gJ_dn, geminal_inv=geminal_inv)
    return (
        _side(gd.orb_data_up_spin, b["bar_orb_up"], b["bar_lap_up"], b["bar_gou"], r_up),
        _side(gd.orb_data_dn_spin, b["bar_orb_dn"], b["bar_lap_dn"], b["bar_godn"], r_dn),
    )


# --------------------------------------------------------------------------- #
# J3 three-body-Jastrow ORBITAL radial basis-opt (d/d radial.{exponents,coefficients}
# of the J3 orbital). The J3 orbital is SHARED across both spins, so each derivative
# accumulates the up- AND dn-electron contributions of the single orbital leaf.
# Bare-AO or MO-backed leaf, cart / sphe / real / complex-twist pgto (composite J3 stays on _auto);
# the MO case contracts through the MO coefficients in _basis_contract_*_general.
# --------------------------------------------------------------------------- #
def _j3_value_cotangents(
    j3: Jastrow_three_body_data,
    r_up: jax.Array,
    r_dn: jax.Array,
) -> J3ValueCotangents:
    """J3 VALUE cotangents ``d J3 / d A`` on the orbital value field A = Re(orb(r)).

    J3 = <j_matrix, [grad_M | grad_j1col]> is bilinear in A, and the value gradient equals
    the ``coefA`` field the J3 e-gradient (kinetic._j3_side_r) uses:
        bar_A_up = j1col + M A_up Ku + M^T A_up Ku^T + (M s_dn)   (broadcast over columns)
        bar_A_dn = j1col + M A_dn Kd + M^T A_dn Kd^T + (M^T s_up)
    with Ku/Kd strict-lower ones, s = A@1, M = j_matrix[:, :-1], j1col = j_matrix[:, -1].
    Returns (A_up, A_dn, bar_A_up, bar_A_dn)."""
    orb = j3.orb_data
    A_up = jnp.real(j3.compute_orb_api(orb, r_up))  # (orb_num, N_up)
    A_dn = jnp.real(j3.compute_orb_api(orb, r_dn))
    n_up, n_dn = A_up.shape[1], A_dn.shape[1]
    Ku = jnp.tril(jnp.ones((n_up, n_up)), -1)
    Kd = jnp.tril(jnp.ones((n_dn, n_dn)), -1)
    Jm = j3._j_matrix_jnp
    M = Jm[:, :-1]
    j1 = Jm[:, -1]
    s_up, s_dn = A_up.sum(1), A_dn.sum(1)
    bar_A_up = j1[:, None] + M @ A_up @ Ku + M.T @ A_up @ Ku.T + (M @ s_dn)[:, None]
    bar_A_dn = j1[:, None] + M @ A_dn @ Kd + M.T @ A_dn @ Kd.T + (M.T @ s_up)[:, None]
    return A_up, A_dn, bar_A_up, bar_A_dn


def _side_j3_value(orb: OrbitalData, bar: jax.Array, r: jax.Array) -> RadialGradient:
    """VALUE-block basis contraction for the J3 orbital leaf (bare AO / MO, cart / sphe / pgto).

    Always Re-projected (holomorphic=False): unlike the determinant, the three-body Jastrow consumes
    ``Re(orb)`` (jastrow_factor Re-projects each AO leaf so exp(J3) stays a real correlation factor),
    so ``d J3/d radial`` is a REAL derivative -- there is no Im part to preserve even on the
    complex-twist ECP dV_nl/dc path (the complex per-grid weight's Re is taken by compute_grad_e_L)."""
    if _is_plain_cart_leaf(orb):
        return _basis_contract(orb, bar, None, None, r)
    return _basis_contract_value_general(orb, bar, r)


def _side_j3_kinetic(
    orb: OrbitalData,
    bar_orb: jax.Array,
    bar_lap: jax.Array,
    bar_gou: jax.Array,
    r: jax.Array,
) -> RadialGradient:
    """KINETIC-block basis contraction for the J3 orbital leaf (bare AO / MO, cart / sphe / pgto)."""
    if _is_plain_cart_leaf(orb):
        return _basis_contract(orb, bar_orb, bar_lap, bar_gou, r)
    return _basis_contract_kinetic_general(orb, bar_orb, bar_lap, bar_gou, r)


def grad_J3_wrt_basis(
    jastrow_data: Jastrow_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
) -> RadialGradient:
    """O_k basis block for the three-body Jastrow ORBITAL radial exp/coeff.

    d ln|Psi| / d(J3 radial) = d J3 / d(J3 radial) (the determinant is J3-independent). Contract
    the J3 VALUE cotangent ``bar_A`` with the SHARED orbital's radial value sensitivity for each
    electron set and SUM (the one orbital is evaluated at both r_up and r_dn). Returns the single
    orbital's (d/d exponents, d/d coefficients). Handles a bare-AO or MO-backed J3 leaf (cart /
    sphe / real / complex-twist pgto) via the shared _basis_contract_value_general kernel.

    ALWAYS real (Re-projected): the three-body Jastrow consumes ``Re(orb)``, so d J3/d radial is a
    real derivative -- no holomorphic seam is needed even for the complex-twist ECP dV_nl/dc (that
    path pairs this REAL J3 O_k with the complex per-grid weight and Re's the assembled c-grad; the
    complex holomorphic seam applies only to the DETERMINANT O_k, which consumes the complex orb)."""
    j3 = jastrow_data.jastrow_three_body_data
    orb = j3.orb_data
    _A_up, _A_dn, bar_A_up, bar_A_dn = _j3_value_cotangents(j3, r_up, r_dn)
    gZu, gcu = _side_j3_value(orb, bar_A_up, r_up)
    gZd, gcd = _side_j3_value(orb, bar_A_dn, r_dn)
    return gZu + gZd, gcu + gcd


def grad_T_wrt_J3_basis(
    gd: GeminalData,
    jastrow_data: Jastrow_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    geminal_inv: jnpt.ArrayLike | None = None,
) -> RadialGradient:
    """dT/dc basis block for the three-body Jastrow ORBITAL radial exp/coeff.

    Mirrors kinetic._j3_side_r but keeps the reverse cotangents on the J3 orbital FIELDS
    (value / gradient / Laplacian) instead of collapsing them onto r, then contracts each
    field cotangent with the radial value/grad/lap sensitivities (basis._basis_contract_*).

    With cot = gJ + gD (drift), coefA the J3 value gradient, and per-orbital
    ``bcA = -0.5 lapA - cot.gradA`` (the coefficient the pair/outer value derivative multiplies)::

        bar_lap  = -0.5 coefA                             (cotangent on lap A)
        bar_gou  = -cot (x) coefA                         (cotangent on grad A)
        bar_orb  = M^T bcA Ku^T + M bcA Ku + (M bcA_other.sum) (cotangent on value A;
                    last term = the s = A@1 outer-block cross coupling to the other spin)

    Summed over up + dn (shared orbital). cart/sphe GTO only."""
    from jqmc.jastrow_factor import compute_grads_and_laplacian_Jastrow_part

    from .ao_deriv import j3_basis_derivs
    from .kinetic import _kinetic_det_grad_field

    j3 = jastrow_data.jastrow_three_body_data
    orb = j3.orb_data
    A_up, gO_up, lO_up, _, _ = j3_basis_derivs(orb, r_up)  # (orb,N), (3,orb,N), (orb,N)
    A_dn, gO_dn, lO_dn, _, _ = j3_basis_derivs(orb, r_dn)
    n_up, n_dn = A_up.shape[1], A_dn.shape[1]
    Ku = jnp.tril(jnp.ones((n_up, n_up)), -1)
    Kd = jnp.tril(jnp.ones((n_dn, n_dn)), -1)
    Jm = j3._j_matrix_jnp
    M = Jm[:, :-1]
    j1 = Jm[:, -1]
    s_up, s_dn = A_up.sum(1), A_dn.sum(1)
    coefA_up = j1[:, None] + M @ A_up @ Ku + M.T @ A_up @ Ku.T + (M @ s_dn)[:, None]
    coefA_dn = j1[:, None] + M @ A_dn @ Kd + M.T @ A_dn @ Kd.T + (M.T @ s_up)[:, None]

    gJ_up, gJ_dn, _, _ = compute_grads_and_laplacian_Jastrow_part(jastrow_data, r_up, r_dn)
    gD_up, gD_dn = _kinetic_det_grad_field(gd, r_up, r_dn, geminal_inv=geminal_inv)
    # cot = grad(ln D) + grad(J) is the coefficient of d(grad J)/dc in the kinetic local energy
    # (from lap(ln Psi) + (grad ln Psi)^2). For a COMPLEX-TWIST determinant grad(ln D) is complex,
    # but the three-body Jastrow consumes Re(orb) so grad(J) -- and its basis derivative d(grad J)/dc
    # -- is REAL. e_L takes Re on the WHOLE local energy, so the physically correct coefficient of the
    # real d(grad J)/dc is Re(grad ln D): Re(gD . dgJ/dc) = Re(gD) . dgJ/dc. Keeping the full complex
    # gD here (Re-ing only at the end of _kinetic_general_leaf) leaks a spurious -Im(gD).Im(sens) term
    # -- the holomorphic behavior that is CORRECT for the complex-determinant leaf (lambda-basis dT/dc,
    # which consumes the complex orb) but WRONG for a Re(orb)-consuming J3 leaf. Re-project cot so the
    # shared kinetic-basis kernel gets REAL cotangents; its final Re then contracts Re(phase).Re(bar).
    # sens exactly. No-op at real Gamma / molecular (gD already real): real/pgto J3 paths bit-identical.
    cot_u = jnp.real(jnp.asarray(gJ_up) + gD_up)  # (N_up, 3)
    cot_d = jnp.real(jnp.asarray(gJ_dn) + gD_dn)

    bcA_up = -0.5 * lO_up - jnp.einsum("kd,dak->ak", cot_u, gO_up)  # (orb, N_up)
    bcA_dn = -0.5 * lO_dn - jnp.einsum("kd,dak->ak", cot_d, gO_dn)

    bar_lap_up = -0.5 * coefA_up
    bar_lap_dn = -0.5 * coefA_dn
    bar_gou_up = -jnp.einsum("kd,ak->dak", cot_u, coefA_up)  # (3, orb, N_up)
    bar_gou_dn = -jnp.einsum("kd,ak->dak", cot_d, coefA_dn)
    bar_orb_up = M.T @ bcA_up @ Ku.T + M @ bcA_up @ Ku + (M @ bcA_dn.sum(1))[:, None]
    bar_orb_dn = M.T @ bcA_dn @ Kd.T + M @ bcA_dn @ Kd + (M.T @ bcA_up.sum(1))[:, None]

    gZu, gcu = _side_j3_kinetic(orb, bar_orb_up, bar_lap_up, bar_gou_up, r_up)
    gZd, gcd = _side_j3_kinetic(orb, bar_orb_dn, bar_lap_dn, bar_gou_dn, r_dn)
    return gZu + gZd, gcu + gcd
