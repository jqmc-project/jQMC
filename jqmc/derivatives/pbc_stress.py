r"""Autograd-free periodic Pulay and local-energy strain derivatives.

Provides strain derivatives for periodic pgto and plane-wave wavefunctions used by stress estimators. Affine strain F = I + eps deforms cell vectors, atom positions, electrons, and AO image translations. Plane-wave contributions are zero for a joint affine strain. Detailed strain-derivative and periodic-estimator derivations are work in progress."""

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

from jqmc.atomic_orbital import AOs_base_data
from jqmc.determinant import AGP_data, AGPn_data, SD_data
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_three_body_data, Jastrow_two_body_data
from jqmc.structure import Structure_data, _inv_3x3

from .ao_deriv import ao_grad as _cart_ao_grad
from .ao_deriv import ao_grad_gradlap_hessian as _k_grad_gradlap_hessian
from . import pgto_deriv as P
from .fast import geminal_value_tables, resolve_geminal_inv

_ArrayPair: TypeAlias = tuple[jax.Array, jax.Array]
# One spin's orbital set: an AO/MO leaf, a composite or a TiledOrbitalSet_data.
OrbitalData: TypeAlias = object
_ArrayTriple: TypeAlias = tuple[jax.Array, jax.Array, jax.Array]
_SmoothStrainFields: TypeAlias = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]


def _pgto_stress_contrib(
    aos: AOs_base_data,
    CM: jax.Array,
    r: jax.Array,
    pos: jax.Array,
) -> jax.Array:
    """Det stress (3, 3) from an atom-centered pgto leaf = folded-grad virial (term1) minus the
    kernel-rank image-T correction (term2)."""
    from jqmc.structure import _wrap_into_cell_if_pbc

    from .assemble import _pgto_home_atom

    r_j = jnp.asarray(r, jnp.float64)
    r_w = _wrap_into_cell_if_pbc(aos.structure_data, r_j)
    co = jnp.asarray(CM)  # (n_ao, N)
    R_mu = jnp.asarray(pos)[jnp.asarray(_pgto_home_atom(aos), jnp.int32)]  # (n_ao, 3)
    gA = P.pgto_grad(aos, r)  # (3, n_ao, N) folded
    rRk = r_w[None, :, :] - R_mu[:, None, :]  # (n_ao, N, 3)
    term1 = jnp.einsum("mk,cmk,mkd->cd", co, gA, rRk)
    gK = _cart_ao_grad(aos, r_w)  # (3, M, N) kernel rank
    T, phase, home = P._pgto_kernel_image(aos)  # static lattice-image metadata
    co_k = co[jnp.asarray(home, jnp.int32)]  # (M, N) cotangent gathered to kernel rank
    restore = aos._restore_phase(r_j, "ao_grad_lap") if P._is_complex_pgto(aos) else jnp.ones(r_j.shape[0])
    wgK = gK * jnp.asarray(phase)[None, :, None] * restore[None, None, :]  # (3, M, N)
    term2 = -jnp.einsum("ckn,kn,kd->cd", wgK, co_k, jnp.asarray(T))
    return term1 + term2


def grad_lndet_wrt_strain(
    gd: AGP_data | AGPn_data | SD_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    take_real: bool = True,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    r"""d ln|det G| / d eps (3, 3) for a periodic pgto geminal. take_real=True returns Re (the Pulay stress); False keeps the holomorphic complex form for the ECP wf-ratio (Re on the product).

    `geminal_inv` is the base-position running G^{-1} cotangent field used by the strain contraction. Detailed derivation is work in progress."""
    M_dn, N_dn = gd.orb_num_dn, gd.num_electron_dn
    # Without a running inverse, G comes from the fused value table whose inverse the
    # drift and the d/dR force contract with the same AO gradients (#762).
    orb_up, orb_dn = geminal_value_tables(gd, r_up, r_dn, geminal_inv)
    from jqmc.determinant import SD_data

    if isinstance(gd, SD_data):
        G = jnp.hstack([orb_up[:N_dn].T @ orb_dn, orb_up[N_dn:].T])
        Q = orb_up[:N_dn].T
    else:
        lam_p, lam_unp = jnp.hsplit(gd._lambda_matrix_jnp, [M_dn])
        G = jnp.hstack([orb_up.T @ lam_p @ orb_dn, orb_up.T @ lam_unp])
        W = jnp.hstack([lam_p @ orb_dn, lam_unp])
        Q = orb_up.T @ lam_p
    # Use the thresholded SVD pseudoinverse rather than an LU solve: the
    # antisymmetry-regularized geminal can be rank fragile. Reuse one inverse.
    G_inv = resolve_geminal_inv(G, geminal_inv)
    if isinstance(gd, SD_data):
        bar_orb_up = jnp.concatenate([orb_dn @ G_inv[:N_dn], G_inv[N_dn:]], axis=0)
    else:
        bar_orb_up = W @ G_inv  # d ln|det| / d orb_up
    bar_orb_dn = (G_inv @ Q)[:N_dn].T  # d ln|det| / d orb_dn
    pos = gd.orb_data_up_spin.structure_data._positions_cart_jnp
    out = jnp.zeros((3, 3), dtype=jnp.float64 if take_real else jnp.complex128)
    for orb, bar_orb, r in ((gd.orb_data_up_spin, bar_orb_up, r_up), (gd.orb_data_dn_spin, bar_orb_dn, r_dn)):
        out = out + det_strain_field(orb, bar_orb, r, pos, take_real=take_real)
    return out


def det_strain_field(
    orb: OrbitalData,
    bar_orb: jax.Array,
    r: jax.Array,
    pos: jax.Array,
    take_real: bool = True,
) -> jax.Array:
    r"""One spin's det Pulay-stress field ``sum_{orb, k} bar_orb[orb, k] d orb(r_k) / d eps`` (3, 3).

    ``bar_orb`` (num_orb, N) is the orbital cotangent and ``r`` (N, 3) the electron positions of
    its columns.
    """
    from jqmc.orbital_tiling import TiledOrbitalSet_data

    from .orbital_tiling import tiled_field_vjp

    if isinstance(orb, TiledOrbitalSet_data):
        contribution = tiled_field_vjp(orb, r, bar_orb, wrt="eps")
        return jnp.real(contribution) if take_real else contribution
    out = jnp.zeros((3, 3), dtype=jnp.float64 if take_real else jnp.complex128)
    for leaf, C_leaf, kind in P.orbital_leaves(orb):
        if kind in ("pw", "bspline"):
            continue  # PW / B-spline orbital VALUE is affine-strain-invariant (G.r resp. frac
            # = r.inv(cell) is preserved under r->Fr, cell->F.cell) -> 0 Pulay-stress contribution
        aos, T_mol = P.sphe_leaf_to_cart(leaf)
        CM = (jnp.asarray(C_leaf) @ jnp.asarray(T_mol)).T @ jnp.asarray(bar_orb)  # (num_cart, N)
        s = _pgto_stress_contrib(aos, CM, r, pos)
        out = out + (jnp.real(s) if take_real else s)
    return out


def _dd_dstrain(disp: jax.Array, cell: jax.Array, inv_cell: jax.Array) -> _ArrayPair:
    """(d, dd/deps) of the smooth periodic distance, batched over a leading ``disp`` field (..., 3).
    Fractional s is strain-invariant; the full positive quadratic tensor Q
    gives dd/deps=Q/d, not only diagonal Gram contributions."""
    from jqmc.jastrow_factor import _periodic_sincos_residual

    local = disp - jnp.rint(disp @ inv_cell) @ cell
    theta = jnp.pi * (local @ inv_cell)
    z = jnp.sin(theta) ** 2
    cartesian = local + (_periodic_sincos_residual(theta) / jnp.pi) @ cell
    tensor = cartesian[..., :, None] * cartesian[..., None, :]
    tensor += jnp.einsum("...k,ka,kb->...ab", z * z / jnp.pi**2, cell, cell)
    d = jnp.sqrt(jnp.trace(tensor, axis1=-2, axis2=-1))
    dd = tensor / d[..., None, None]
    return d, dd


def _j1_fp(j1: Jastrow_one_body_data, d: jax.Array) -> jax.Array:
    """J1 radial f'(d) for every atom, batched: d (..., nA) -> (..., nA). Z/c/pref broadcast over nA."""
    Z = jnp.asarray(j1.structure_data.atomic_numbers, jnp.float64) - jnp.asarray(j1.core_electrons, jnp.float64)  # (nA,)
    p = j1.jastrow_1b_param  # (possibly traced) scalar
    c = (2.0 * Z) ** 0.25  # (nA,)
    pref = -((2.0 * Z) ** 0.75) if j1.jastrow_1b_type == "exp" else -2.0 * Z  # (nA,)
    if j1.jastrow_1b_type == "exp":
        return pref * c / 2.0 * jnp.exp(-p * c * d)
    if j1.jastrow_1b_type == "pade":
        return pref / (2.0 * (1.0 + p * c * d) ** 2)
    raise ValueError(j1.jastrow_1b_type)


def _j3_pulay_strain(
    j3: Jastrow_three_body_data,
    r_up: jax.Array,
    r_dn: jax.Array,
) -> jax.Array:
    """J3 Pulay stress d ln|Psi_J3| / d eps (3, 3): the J3 linearized coefficient (gA, as in
    assemble.grad_J3_wrt_R) contracted with the J3-basis VALUE affine field strain (virial +
    image-T), transformed sphe->cart and Re'd. The periodic analogue of the molecular J3 piece of
    _stress_lnPsi (drift.r + force.R), exactly as the det Pulay stress (grad_lndet_wrt_strain)."""
    from .ao_deriv import j3_basis_derivs
    from .assemble import _j3_mo_to_bare_ao

    j3 = _j3_mo_to_bare_ao(j3)  # real MO lowers to AO; a complex MO stays at orbital rank
    aos = j3.orb_data
    A_up = j3_basis_derivs(aos, r_up)[0]  # Re J3 basis value, orbital (sphe / composite) rank
    A_dn = j3_basis_derivs(aos, r_dn)[0]
    gA_up, gA_dn = j3_strain_coefficients(j3, A_up, A_dn)
    return j3_strain_field(aos, gA_up, r_up) + j3_strain_field(aos, gA_dn, r_dn)


def j3_strain_coefficients(
    j3: Jastrow_three_body_data,
    A_up: jax.Array,
    A_dn: jax.Array,
) -> _ArrayPair:
    """J3 linearized coefficients (gA_up (n, N_up), gA_dn (n, N_dn)) of the basis values ``A``:
    ``d J3 = sum_k gA[:, k] . d A[:, k]`` (``j3`` already lowered by ``_j3_mo_to_bare_ao``)."""
    n_up, n_dn = A_up.shape[1], A_dn.shape[1]
    K_up = jnp.tril(jnp.ones((n_up, n_up)), -1)
    K_dn = jnp.tril(jnp.ones((n_dn, n_dn)), -1)
    Jm = j3._j_matrix_jnp
    M = Jm[:, :-1]
    j1col = Jm[:, -1]
    s_up, s_dn = A_up.sum(1), A_dn.sum(1)
    gA_up = j1col[:, None] + M @ A_up @ K_up + M.T @ A_up @ K_up.T + (M @ s_dn)[:, None]
    gA_dn = j1col[:, None] + M @ A_dn @ K_dn + M.T @ A_dn @ K_dn.T + (M.T @ s_up)[:, None]
    return gA_up, gA_dn


def j3_strain_field(aos: OrbitalData, coefA: jax.Array, r: jax.Array) -> jax.Array:
    r"""J3 basis-value strain field ``Re sum_k coefA[:, k] . d A(r_k) / d eps`` (3, 3) for one
    spin's coefficients ``coefA`` (n, N) at positions ``r`` (N, 3); ``aos`` is the lowered J3 basis.
    """
    from jqmc.structure import _wrap_into_cell_if_pbc

    from .assemble import _pgto_home_atom

    out = jnp.zeros((3, 3))
    # Per-AO-leaf strain contraction (a composite descends to its children by AO-row offset, the
    # coupling gA staying at composite rank -- J3 couples across children): a GTO/pGTO child takes
    # the virial + image-T field strain; a plane-wave / B-spline child's VALUE is affine-strain-
    # invariant (G.r resp. frac = r.inv(cell) preserved under r->Fr, cell->F.cell) -> 0 Pulay
    # contribution, exactly as the det Pulay stress (grad_lndet_wrt_strain) skips them.
    # A single bare leaf is one "child" (bit-identical to the legacy path).
    mo_rank = type(aos).__name__ == "MOs_data"
    off = 0
    for leaf, C, kind in P.orbital_leaves(aos):
        n = int(leaf.num_ao)
        if kind in ("pw", "bspline"):
            off += n
            continue
        cart, T_mol = P.sphe_leaf_to_cart(leaf)
        T = jnp.asarray(T_mol)
        pos = jnp.asarray(cart.structure_data.positions, jnp.float64)
        R_mu = pos[jnp.asarray(_pgto_home_atom(cart), jnp.int32)]
        r_j = jnp.asarray(r, jnp.float64)
        r_w = _wrap_into_cell_if_pbc(cart.structure_data, r_j)
        restore = cart._restore_phase(r_j, "ao_grad_lap") if P._is_complex_pgto(cart) else jnp.ones(r_j.shape[0])
        coef_leaf = jnp.asarray(C).T @ coefA if mo_rank else coefA[off : off + n]
        bA = T.T @ coef_leaf
        s = _field_stress(cart, bA, P.pgto_grad(cart, r), _cart_ao_grad(cart, r_w), r_w, R_mu, restore)
        out = out + jnp.real(s)
        off += n
    return out


def j1_point_strain(j1: Jastrow_one_body_data, x: jax.Array) -> jax.Array:
    r"""J1 strain of each electron position ``x`` (..., 3), shape (..., 3, 3)."""
    cell = jnp.asarray(j1.structure_data._cell_jnp, jnp.float64)  # static lattice (constant under jit)
    inv = _inv_3x3(cell)
    R = jnp.asarray(j1.structure_data._positions_cart_jnp, jnp.float64)
    disp = x[..., None, :] - R  # (..., nA, 3)
    d, dd = _dd_dstrain(disp, cell, inv)  # (..., nA), (..., nA, 3, 3)
    fp = _j1_fp(j1, d)  # (..., nA)
    return jnp.einsum("...I,...Iab->...ab", fp, dd)


def j2_pair_strain(j2: Jastrow_two_body_data, disp: jax.Array) -> jax.Array:
    r"""J2 strain of each electron-pair displacement ``disp`` (..., 3), shape (..., 3, 3)."""
    p = j2.jastrow_2b_param
    kind = j2.jastrow_2b_type
    if j2.structure_data.pbc_flag:
        cell = jnp.asarray(j2.structure_data._cell_jnp, jnp.float64)
        d, dd = _dd_dstrain(disp, cell, jnp.linalg.inv(cell))
    else:
        # A cell-less J2 embedded in a periodic determinant remains Euclidean: there is no
        # J2 lattice metric to strain, only the affine electron-position virial.
        from .kinetic import _dist_and_derivs

        d, grad_d, _, _ = _dist_and_derivs(j2.structure_data, disp)
        dd = jnp.einsum("...a,...b->...ab", grad_d, disp)
    hp = 1.0 / (2.0 * (1.0 + p * d) ** 2) if kind == "pade" else 0.5 * jnp.exp(-p * d)
    return hp[..., None, None] * dd


def grad_lnJastrow_wrt_strain(jastrow_data: Jastrow_data, r_up: jax.Array, r_dn: jax.Array) -> jax.Array:
    r"""Strain derivative d(J1 + J2 + J3)/d eps (3, 3) for PBC."""
    all_e = jnp.concatenate([jnp.asarray(r_up, jnp.float64), jnp.asarray(r_dn, jnp.float64)], 0)
    out = jnp.zeros((3, 3))
    j1 = jastrow_data.jastrow_one_body_data
    if j1 is not None:
        out = out + jnp.sum(j1_point_strain(j1, all_e), axis=0)
    j2 = jastrow_data.jastrow_two_body_data
    if j2 is not None:
        ii, jj = jnp.triu_indices(all_e.shape[0], 1)
        out = out + jnp.sum(j2_pair_strain(j2, all_e[ii] - all_e[jj]), axis=0)
    j3 = jastrow_data.jastrow_three_body_data
    if j3 is not None:
        out = out + _j3_pulay_strain(j3, r_up, r_dn)
    return out


def stress_lnPsi_pbc(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    take_real: bool = True,
    geminal_inv: jax.Array | None = None,
    *,
    include_dtn: bool = True,
) -> jax.Array:
    """Periodic Pulay stress d ln|Psi| / d eps = d ln|det|/d eps + d(J1+J2+J3)/d eps (3, 3). take_real=
    False keeps the det part holomorphic (complex Bloch) for the ECP wf-ratio; J1/J2/J3 are real.

    ``geminal_inv`` (fast path): base-position running SM ``G^{-1}`` reused by ``grad_lndet_wrt_strain``."""
    wf = hamiltonian_data.wavefunction_data
    out = grad_lndet_wrt_strain(
        wf.geminal_data, r_up, r_dn, take_real=take_real, geminal_inv=geminal_inv
    ) + grad_lnJastrow_wrt_strain(wf.jastrow_data, r_up, r_dn)
    if include_dtn and wf.jastrow_data.jastrow_dtn_data is not None:
        # DTN min-image displacement virial (real; J1/J2 use Williamson).
        # ``include_dtn=False`` is the production one-move route: its weighted
        # DTN delta is supplied separately by the rank-one geometry kernel.
        from . import dtn

        out = out + dtn.grad_dtn_wrt_strain(wf.jastrow_data.jastrow_dtn_data, r_up, r_dn)
    return out


# --------------------------------------------------------------------------- #
# Periodic kinetic stress d T / d eps -- orbital (determinant) side. The kinetic VJP
# cotangents (bar_orb / bar_lap / bar_gou from _geminal_reverse) contracted with the
# strain response of the AO value / laplacian / gradient fields: each = folded-grad
# affine virial (term1) + kernel-rank image-T correction (term2).
# --------------------------------------------------------------------------- #
def _field_stress(
    aos: AOs_base_data,
    CM: jax.Array,
    gfold: jax.Array,
    gkern: jax.Array,
    r_w: jax.Array,
    R_mu: jax.Array,
    restore: jax.Array,
) -> jax.Array:
    """sum_{mu,k} CM d(field)/d eps for one pgto leaf: term1 virial + term2 image-T. ``field``
    is the AO value (gfold=grad) or laplacian (gfold=gradlap); gkern is the kernel-rank grad."""
    co = jnp.asarray(CM)
    term1 = jnp.einsum("mk,cmk,mkd->cd", co, gfold, r_w[None, :, :] - R_mu[:, None, :])
    T, phase, home = P._pgto_kernel_image(aos)
    wgK = gkern * jnp.asarray(phase)[None, :, None] * restore[None, None, :]  # (3, M, N)
    term2 = -jnp.einsum("ckn,kn,kd->cd", wgK, co[jnp.asarray(home, jnp.int32)], jnp.asarray(T))
    return term1 + term2


def _gou_field_stress(
    aos: AOs_base_data,
    bar_gou: jax.Array,
    Hfold: jax.Array,
    Hkern: jax.Array,
    r_w: jax.Array,
    R_mu: jax.Array,
    restore: jax.Array,
) -> jax.Array:
    """bar_gou . d(grad AO)/d eps: grad(grad AO_d) = Hessian[:, d] -> virial + image-T."""
    bg = jnp.asarray(bar_gou)  # (3=d, mu, k)
    term1 = jnp.einsum("dmk,cdmk,mkp->cp", bg, Hfold, r_w[None, :, :] - R_mu[:, None, :])
    T, phase, home = P._pgto_kernel_image(aos)
    wHk = Hkern * jnp.asarray(phase)[None, None, :, None] * restore[None, None, None, :]  # (3c,3d,M,N)
    term2 = -jnp.einsum("dkn,cdkn,kp->cp", bg[:, jnp.asarray(home, jnp.int32), :], wHk, jnp.asarray(T))
    return term1 + term2


def _pw_kinetic_stress_contrib(
    leaf: AOs_base_data,
    CM_orb: jax.Array,
    CM_lap: jax.Array,
    CM_gou: jax.Array,
    r: jax.Array,
) -> jax.Array:
    """Plane-wave orbital-side kinetic stress dT/deps (3,3). Under the affine stress strain the PW
    AO value cos(G.r) is invariant (G->F^-T G, r->F r), so only the laplacian (-|G|^2 val) and the
    gradient (dval (F^-T G)_a) respond: d(lap)/deps_cd = 2 G_c G_d val, d(gou_a)/deps_cd =
    -delta_ad G_c dval. Contract with the kinetic cotangents (CM_lap value-side, CM_gou grad-side)."""
    from . import ao as A

    G, val, dval = A._pw_val_dval(leaf, r)  # G (num_ao,3); val,dval (num_ao,N)
    SL = jnp.einsum("ki,ki->k", jnp.asarray(CM_lap), val)  # (num_ao,)
    lap_term = 2.0 * jnp.einsum("k,kc,kd->cd", SL, G, G)
    SG = jnp.einsum("dki,ki->dk", jnp.asarray(CM_gou), dval)  # (3, num_ao)
    gou_term = -jnp.einsum("kc,dk->cd", G, SG)
    return lap_term + gou_term


def _bspline_kinetic_stress_contrib(
    leaf: AOs_base_data,
    CM_lap: jax.Array,
    CM_gou: jax.Array,
    r: jax.Array,
) -> jax.Array:
    """B-spline orbital-side kinetic stress dT/deps (3,3). frac=r.inv(cell) is affine-invariant so the
    spline t-space fields (gt, Ht) are strain-invariant; only the metric responds. inv(cell)->F^-T inv
    (same G->F^-T G convention as PW), so dinv_cd[a,b]=-delta_ad inv[c,b] -> dmetric[k,l;c,d]=
    -inv[c,k]inv[d,l]-inv[d,k]inv[c,l] and dM[k,a;c,d]=-delta_ad N_k inv[c,k]:
      lap term = -inv @ (SLn + SLn^T) @ inv^T,  SLn[k,l]=N_k N_l sum_bi CM_lap[b,i] Ht[i,b,k,l]
      gou term = -inv @ (SG2 * N[None,:])^T,    SG2[d,k]=sum_bi CM_gou[d,b,i] gt[i,b,k]
    (value side is 0: B-spline value affine-invariant)."""
    from . import ao as A

    gt, Ht, inv, ns = A.bspline_t_fields(leaf, r)  # (N,nb,3),(N,nb,3,3),(3,3),(3,)
    SL = jnp.einsum("bi,ibkl->kl", jnp.asarray(CM_lap), Ht)
    SLn = SL * ns[:, None] * ns[None, :]
    lap_term = -inv @ (SLn + SLn.T) @ inv.T
    SG2 = jnp.einsum("dbi,ibk->dk", jnp.asarray(CM_gou), gt)  # (3,3)
    gou_term = -inv @ (SG2 * ns[None, :]).T
    return lap_term + gou_term


def grad_T_wrt_strain_orbital(
    gd: AGP_data | AGPn_data | SD_data,
    jastrow_data: Jastrow_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    b: dict[str, jax.Array | None] | None = None,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    r"""Orbital (det) side of the periodic kinetic stress d T / d eps (3, 3). The Jastrow side is added by grad_T_wrt_strain_jastrow.

    `b` (optional): a precomputed kinetic reverse-pass cotangent dict (`bar_orb` / `bar_lap` /
    `bar_gou` per side). When `None` (the default), it is computed internally.
    """
    from jqmc.jastrow_factor import compute_grads_and_laplacian_Jastrow_part
    from jqmc.structure import _wrap_into_cell_if_pbc

    from . import ao as A
    from .assemble import _pgto_home_atom
    from .kinetic import _geminal_reverse

    if b is None:
        gJ_up, gJ_dn, _, _ = compute_grads_and_laplacian_Jastrow_part(jastrow_data, r_up, r_dn)
        b = _geminal_reverse(gd, r_up, r_dn, gJ_up, gJ_dn, geminal_inv=geminal_inv)
    pos = gd.orb_data_up_spin.structure_data._positions_cart_jnp
    out = jnp.zeros((3, 3))
    for orb, bar_orb, bar_lap, bar_gou, r in (
        (gd.orb_data_up_spin, b["bar_orb_up"], b["bar_lap_up"], b["bar_gou"], r_up),
        (gd.orb_data_dn_spin, b["bar_orb_dn"], b["bar_lap_dn"], b["bar_godn"], r_dn),
    ):
        from jqmc.orbital_tiling import TiledOrbitalSet_data
        from .orbital_tiling import tiled_field_vjp

        if isinstance(orb, TiledOrbitalSet_data):
            out = out + jnp.real(tiled_field_vjp(orb, r, bar_orb, bar_gou, bar_lap, wrt="eps"))
            continue
        for leaf, C_leaf, kind in P.orbital_leaves(orb):
            C = jnp.asarray(C_leaf)
            CM_orb = C.T @ jnp.asarray(bar_orb)
            CM_lap = C.T @ jnp.asarray(bar_lap)
            CM_gou = jnp.einsum("ac,dak->dck", C, jnp.asarray(bar_gou))
            if kind == "pw":
                out = out + jnp.real(_pw_kinetic_stress_contrib(leaf, CM_orb, CM_lap, CM_gou, r))
                continue
            if kind == "bspline":
                if "complex" in type(leaf).__name__:  # complex-twist Bloch spline: phase/k-aware kinetic stress
                    out = out + jnp.real(A.bspline_complex_kinetic_stress_contrib(leaf, CM_lap, CM_gou, r))
                else:
                    out = out + jnp.real(_bspline_kinetic_stress_contrib(leaf, CM_lap, CM_gou, r))
                continue
            if kind != "pgto":
                # Defensive guard (currently unreachable): a periodic geminal's orbital leaves are
                # pgto / pw / bspline (all handled above); kind == "gto" is a molecular GTO, which
                # only appears at the mol boundary, never on this PBC kinetic-stress path.
                raise NotImplementedError(
                    f"grad_T_wrt_strain_orbital: AO leaf kind {kind!r} not supported (PBC path expects pgto/pw/bspline)"
                )
            aos, T_mol = P.sphe_leaf_to_cart(leaf)
            Tm = jnp.asarray(T_mol)
            CM_orb = (C @ Tm).T @ jnp.asarray(bar_orb)
            CM_lap = (C @ Tm).T @ jnp.asarray(bar_lap)
            CM_gou = jnp.einsum("ac,dak->dck", C @ Tm, jnp.asarray(bar_gou))
            r_j = jnp.asarray(r, jnp.float64)
            r_w = _wrap_into_cell_if_pbc(aos.structure_data, r_j)
            R_mu = pos[jnp.asarray(_pgto_home_atom(aos), jnp.int32)]
            restore = aos._restore_phase(r_j, "ao_grad_lap") if P._is_complex_pgto(aos) else jnp.ones(r_j.shape[0])
            # One primitive sweep for the folded grad / gradlap / hessian and one for their
            # kernel-rank twins (issue #654), instead of one sweep per field and rank.
            g_f, gl_f, h_f = P.pgto_grad_gradlap_hessian(aos, r)
            g_k, gl_k, h_k = _k_grad_gradlap_hessian(aos, r_w)
            s = _field_stress(aos, CM_orb, g_f, g_k, r_w, R_mu, restore)
            s = s + _field_stress(aos, CM_lap, gl_f, gl_k, r_w, R_mu, restore)
            s = s + _gou_field_stress(aos, CM_gou, h_f, h_k, r_w, R_mu, restore)
            out = out + jnp.real(s)
    return out


def _smooth_strain(disp: jax.Array, cell: jax.Array, inv_cell: jax.Array) -> _SmoothStrainFields:
    """Positive full-Gram geometry and all nine affine-strain derivatives.

    Fractional phases are fixed, while the full cell Gram matrix and inverse
    transform. No coordinate autodiff or rank-five pair Hessian is formed.
    """
    from jqmc.jastrow_factor import _periodic_geometry_components, _periodic_sincos_residual

    B = jnp.asarray(inv_cell, jnp.float64)
    C = jnp.asarray(cell, jnp.float64)
    square, components, trace = _periodic_geometry_components(disp, C, B)
    d = jnp.sqrt(square)
    numerator = jnp.stack(components, axis=-1)
    gd = numerator / d[..., None]
    q = jnp.sum(gd * gd, axis=-1)
    L = (trace - q) / d
    local = disp - jnp.rint(disp @ B) @ C
    theta = jnp.pi * (local @ B)
    sine, cosine = jnp.sin(theta), jnp.cos(theta)
    z, h = sine * sine, sine * cosine
    cartesian = local + (_periodic_sincos_residual(theta) / jnp.pi) @ C
    G, H = C @ C.T, B.T @ B
    D, Hdiag = jnp.diag(G), jnp.diag(H)
    off = G - jnp.diag(D)
    off_h = h @ off
    tensor = cartesian[..., :, None] * cartesian[..., None, :]
    tensor += jnp.einsum("...i,ia,ib->...ab", z * z / jnp.pi**2, C, C)
    dd = tensor / d[..., None, None]

    dG = jnp.einsum("ia,jb->ijab", C, C) + jnp.einsum("ib,ja->ijab", C, C)
    dD = 2 * jnp.einsum("ia,ib->iab", C, C)
    eye = jnp.eye(3, dtype=C.dtype)
    dO = dG - jnp.einsum("ij,iab->ijab", eye, dD)
    dH = -jnp.einsum("ai,bj->ijab", B, B) - jnp.einsum("bi,aj->ijab", B, B)
    dHdiag = -2 * jnp.einsum("ai,bi->iab", B, B)
    d_off_h = jnp.einsum("...j,jiab->...iab", h, dO)

    dm = jnp.einsum("ia,...b->...iab", eye, cartesian)
    dm -= (2 / jnp.pi) * jnp.einsum("...k,...kab,ik->...iab", z, d_off_h, B)
    dm += jnp.einsum("ib,...a->...iab", eye, cartesian - numerator)
    dgd = (dm - gd[..., :, None, None] * dd[..., None, :, :]) / d[..., None, None, None]
    dq = 2 * jnp.einsum("...i,...iab->...ab", gd, dgd)

    dtrace = 2 * jnp.einsum("iab,...i->...ab", dD * Hdiag[:, None, None] + D[:, None, None] * dHdiag, z)
    dtrace += 4 * jnp.einsum("...i,ijab,...j->...ab", z, dO * H[:, :, None, None] + off[:, :, None, None] * dH, z)
    dtrace -= 4 * jnp.einsum("...i,iab,...i->...ab", h, dHdiag, off_h)
    dtrace -= 4 * jnp.einsum("...i,i,...iab->...ab", h, Hdiag, d_off_h)
    dL = (dtrace - dq - L[..., None, None] * dd) / d[..., None, None]
    return d, gd, q, L, dd, dgd, dq, dL


def _smooth_strain_fixed_cell(j2_sd: Structure_data, disp: jax.Array) -> _SmoothStrainFields:
    """Cell-less J2 fallback: only ``disp -> F @ disp`` deforms because there is no
    owned lattice metric. These are the Euclidean affine virials of ``(d, gd, q, L)``."""
    from .force import _smooth_dist_kinematics

    d, gd, q, H, L, gl = _smooth_dist_kinematics(j2_sd, disp)  # batched over leading ... of disp
    dd = jnp.einsum("...a,...b->...ab", gd, disp)  # d d/d eps
    dgd = jnp.einsum("...ia,...b->...iab", H, disp)  # d gd_i/d eps = H[i,a] disp_b
    Hgd = jnp.einsum("...ai,...i->...a", H, gd)  # (H gd)_a
    dq = 2.0 * jnp.einsum("...a,...b->...ab", Hgd, disp)  # d q/d eps = 2 (H gd)_a disp_b
    dL = jnp.einsum("...a,...b->...ab", gl, disp)  # d L/d eps = grad(lap)_a disp_b
    return d, gd, q, L, dd, dgd, dq, dL


def _j1_radial(j1: Jastrow_one_body_data, d: jax.Array) -> _ArrayTriple:
    """(f', f'', f''') of the J1 term vs distance for every atom, batched: d (..., nA) -> (..., nA)."""
    Z = jnp.asarray(j1.structure_data.atomic_numbers, jnp.float64) - jnp.asarray(j1.core_electrons, jnp.float64)  # (nA,)
    p = j1.jastrow_1b_param  # (possibly traced) scalar
    c = (2.0 * Z) ** 0.25  # (nA,)
    pref = -((2.0 * Z) ** 0.75) if j1.jastrow_1b_type == "exp" else -2.0 * Z  # (nA,)
    if j1.jastrow_1b_type == "exp":
        a = p * c
        e = jnp.exp(-a * d)
        return pref * c / 2 * e, -pref * p * c**2 / 2 * e, pref * p**2 * c**3 / 2 * e
    if j1.jastrow_1b_type == "pade":
        u = 1.0 + p * c * d
        return pref / (2 * u**2), -pref * p * c / u**3, 3 * pref * p**2 * c**2 / u**4
    raise ValueError(j1.jastrow_1b_type)


def _j2_radial(p: jax.Array | float, d: jax.Array, kind: str) -> _ArrayTriple:
    if kind == "pade":
        u = 1.0 + p * d
        return 1.0 / (2 * u**2), -p / u**3, 3 * p**2 / u**4
    e = jnp.exp(-p * d)
    return 0.5 * e, -0.5 * p * e, 0.5 * p**2 * e


def _j3_kstress(
    j3: Jastrow_three_body_data,
    cot: jax.Array,
    r_up: jax.Array,
    r_dn: jax.Array,
    w: jax.Array | None = None,
) -> jax.Array:
    """J3 kinetic stress d T_J3 / d eps (3, 3). The J3 cotangents (bar_A / bar_gO / bar_lO, built
    from the Re-projected J3 basis exactly as the J3 force/drift) are contracted with the J3-basis
    AO field strain (value virial + lap virial + grad-of-grad image-T), transformed sphe->cart
    (T_mol) and projected to its real component.

    The field strain re-uses the orbital-side _field_stress / _gou_field_stress on the cart-pgto
    leaf (complex Bloch kept through the kernel image-T term, Re only at the end), so Re(d field/d
    eps) = d Re(field)/d eps -- consistent with the J3 = Re(AO) convention."""
    from jqmc.structure import _wrap_into_cell_if_pbc

    from .assemble import _j3_mo_to_bare_ao, _pgto_home_atom
    from .ao_deriv import j3_basis_derivs

    j3 = _j3_mo_to_bare_ao(j3)  # real MO lowers to AO; a complex MO stays at orbital rank
    aos = j3.orb_data
    A_up, gOu, lOu, _, _ = j3_basis_derivs(aos, r_up)  # Re J3 basis (value/grad/lap), orbital rank
    A_dn, gOd, lOd, _, _ = j3_basis_derivs(aos, r_dn)
    n_up, n_dn = A_up.shape[1], A_dn.shape[1]
    Ku = jnp.tril(jnp.ones((n_up, n_up)), -1)
    Kd = jnp.tril(jnp.ones((n_dn, n_dn)), -1)
    Jm = j3._j_matrix_jnp
    M = Jm[:, :-1]
    j1 = Jm[:, -1]
    s_up, s_dn = A_up.sum(1), A_dn.sum(1)
    coefA_up = j1[:, None] + M @ A_up @ Ku + M.T @ A_up @ Ku.T + (M @ s_dn)[:, None]
    coefA_dn = j1[:, None] + M @ A_dn @ Kd + M.T @ A_dn @ Kd.T + (M.T @ s_up)[:, None]
    cot = jnp.asarray(cot)
    bgu, bgd = -cot[: len(r_up)], -cot[len(r_up) :]  # bar on gJ3 = -(gJ + gD)
    # S' membership weights per owner-electron (1 for the unit-weight VMC path).
    n_up_e = len(r_up)
    if w is None:
        w_u = jnp.ones(n_up_e, jnp.float64)
        w_d = jnp.ones(len(r_dn), jnp.float64)
    else:
        w = jnp.asarray(w, jnp.float64)
        w_u, w_d = w[:n_up_e], w[n_up_e:]

    def side_cot(
        coefA: jax.Array,
        bg: jax.Array,
        gO: jax.Array,
        lO: jax.Array,
        wS: jax.Array,
    ) -> _ArrayTriple:
        # each per-owner-electron (column) seed scaled by the S' weight wS_k -- exactly the
        # weighting pattern of the J3 kinetic drift ``gfmc._j3_side_r_w``.
        bar_gO = jnp.einsum("kd,ak->dak", wS[:, None] * bg, coefA)
        bar_lO = (-0.5) * wS[None, :] * coefA
        bar_coefA = (jnp.einsum("kd,dak->ak", bg, gO) + (-0.5) * lO) * wS[None, :]
        return bar_gO, bar_lO, bar_coefA

    bgO_u, blO_u, bcA_u = side_cot(coefA_up, bgu, gOu, lOu, w_u)
    bgO_d, blO_d, bcA_d = side_cot(coefA_dn, bgd, gOd, lOd, w_d)
    bar_A_up = M.T @ bcA_u @ Ku.T + M @ bcA_u @ Ku + (M @ bcA_d.sum(1))[:, None]
    bar_A_dn = M.T @ bcA_d @ Kd.T + M @ bcA_d @ Kd + (M.T @ bcA_u.sum(1))[:, None]

    out = jnp.zeros((3, 3))
    # Per-AO-leaf field strain (composite descends to children by AO-row offset; the coupling
    # coefA / bar_* above stays at the full composite rank): a GTO/pGTO child de-tiles + c2s to its
    # cart leaf and takes the value/lap virial + grad-of-grad image-T field strain; a plane-wave
    # child's kinetic strain is the closed-form _pw_kinetic_stress_contrib (value affine-invariant;
    # laplacian -|G|^2 val and gradient G_d dval respond through G -> F^-T G), with the SAME
    # cotangent triple (bar_A -> CM_orb, bar_lO -> CM_lap, bar_gO -> CM_gou) the det orbital-side
    # kinetic stress (grad_T_wrt_strain_orbital) feeds it. Single bare leaf == one child.
    mo_rank = type(aos).__name__ == "MOs_data"
    for r, bar_A, bar_gO, bar_lO in ((r_up, bar_A_up, bgO_u, blO_u), (r_dn, bar_A_dn, bgO_d, blO_d)):
        off = 0
        for leaf, C, kind in P.orbital_leaves(aos):
            n = int(leaf.num_ao)
            if mo_rank:
                cT = jnp.asarray(C).T
                bar_A_leaf = cT @ bar_A
                bar_lO_leaf = cT @ bar_lO
                bar_gO_leaf = jnp.einsum("am,dmk->dak", cT, bar_gO)
            else:
                bar_A_leaf = bar_A[off : off + n]
                bar_lO_leaf = bar_lO[off : off + n]
                bar_gO_leaf = bar_gO[:, off : off + n]
            if kind == "pw":
                s = _pw_kinetic_stress_contrib(leaf, bar_A_leaf, bar_lO_leaf, bar_gO_leaf, r)
                out = out + jnp.real(s)
                off += n
                continue
            if kind == "bspline":
                if "complex" in type(leaf).__name__:  # complex-twist Bloch spline J3 leaf: phase/k-aware kinetic stress
                    from . import ao as A

                    s = A.bspline_complex_kinetic_stress_contrib(leaf, bar_lO_leaf, bar_gO_leaf, r)
                else:
                    s = _bspline_kinetic_stress_contrib(leaf, bar_lO_leaf, bar_gO_leaf, r)
                out = out + jnp.real(s)
                off += n
                continue
            cart, T_mol = P.sphe_leaf_to_cart(leaf)  # de-tile + c2s -> cart leaf for the pgto kernels
            T = jnp.asarray(T_mol)  # (n_sphe, n_cart) real c2s
            pos = jnp.asarray(cart.structure_data.positions, jnp.float64)
            R_mu = pos[jnp.asarray(_pgto_home_atom(cart), jnp.int32)]  # (n_cart, 3)
            r_j = jnp.asarray(r, jnp.float64)
            r_w = _wrap_into_cell_if_pbc(cart.structure_data, r_j)
            restore = cart._restore_phase(r_j, "ao_grad_lap") if P._is_complex_pgto(cart) else jnp.ones(r_j.shape[0])
            bA = T.T @ bar_A_leaf
            blO = T.T @ bar_lO_leaf
            bgO = jnp.einsum("sc,dsk->dck", T, bar_gO_leaf)
            g_f, gl_f, h_f = P.pgto_grad_gradlap_hessian(cart, r)  # one sweep (issue #654)
            g_k, gl_k, h_k = _k_grad_gradlap_hessian(cart, r_w)
            s = _field_stress(cart, bA, g_f, g_k, r_w, R_mu, restore)
            s = s + _field_stress(cart, blO, gl_f, gl_k, r_w, R_mu, restore)
            s = s + _gou_field_stress(cart, bgO, h_f, h_k, r_w, R_mu, restore)
            out = out + jnp.real(s)
            off += n
    return out


def grad_T_wrt_strain_jastrow(
    jd: Jastrow_data,
    gd_geminal: AGP_data | AGPn_data | SD_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    w: jax.Array | None = None,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    r"""Jastrow (J1 + J2 + J3) side of the periodic kinetic stress d T_J / d eps (3, 3).

    `w` (optional per-electron weights, length Ne): when `None` (the default), every electron
    contributes with weight 1.
    """
    from jqmc.jastrow_factor import compute_grads_and_laplacian_Jastrow_part

    from .kinetic import _kinetic_det_grad_field

    gJu, gJd, _, _ = compute_grads_and_laplacian_Jastrow_part(jd, r_up, r_dn)
    gDu, gDd = _kinetic_det_grad_field(gd_geminal, r_up, r_dn, geminal_inv=geminal_inv)
    cot = jnp.real(jnp.concatenate([jnp.asarray(gJu) + jnp.asarray(gDu), jnp.asarray(gJd) + jnp.asarray(gDd)], 0))  # (Ne,3)
    all_e = jnp.concatenate([jnp.asarray(r_up, jnp.float64), jnp.asarray(r_dn, jnp.float64)], 0)  # (Ne,3)
    w = jnp.ones(all_e.shape[0], jnp.float64) if w is None else jnp.asarray(w, jnp.float64)
    out = jnp.zeros((3, 3))
    j1 = jd.jastrow_one_body_data
    if j1 is not None:
        cell = jnp.asarray(j1.structure_data._cell_jnp, jnp.float64)
        inv = _inv_3x3(cell)
        R = jnp.asarray(j1.structure_data._positions_cart_jnp, jnp.float64)  # (nA,3)
        disp = all_e[:, None, :] - R[None, :, :]  # (Ne, nA, 3)
        d, gd, q, L, dd, dgd, dq, dL = _smooth_strain(disp, cell, inv)  # batched (Ne,nA,...)
        fp, fpp, fppp = _j1_radial(j1, d)  # (Ne,nA) each
        cot_gd = jnp.einsum("ei,eIi->eI", cot, gd)  # (Ne,nA)
        dgJ = fpp[..., None, None] * dd * cot_gd[..., None, None] + fp[..., None, None] * jnp.einsum("ei,eIiab->eIab", cot, dgd)
        dlap = (
            fppp[..., None, None] * dd * q[..., None, None]
            + fpp[..., None, None] * dq
            + fpp[..., None, None] * dd * L[..., None, None]
            + fp[..., None, None] * dL
        )
        out = out + jnp.einsum("e,eIab->ab", w, -0.5 * dlap - dgJ)
    j2 = jd.jastrow_two_body_data
    if j2 is not None:
        p = j2.jastrow_2b_param
        kind = j2.jastrow_2b_type
        ii, jj = jnp.triu_indices(all_e.shape[0], 1)  # (P,) electron pairs i<j
        disp = all_e[ii] - all_e[jj]  # (P,3)
        if j2.structure_data.pbc_flag:
            cell = jnp.asarray(j2.structure_data._cell_jnp, jnp.float64)
            d, gd, q, L, dd, dgd, dq, dL = _smooth_strain(disp, cell, jnp.linalg.inv(cell))
        else:
            # No owned J2 cell means Euclidean distance; only electron positions strain.
            d, gd, q, L, dd, dgd, dq, dL = _smooth_strain_fixed_cell(j2.structure_data, disp)
        hp, hpp, hppp = _j2_radial(p, d, kind)  # (P,) each
        coti, cotj = cot[ii], cot[jj]  # (P,3)
        ci_gd = jnp.einsum("pk,pk->p", coti, gd)
        cj_gd = jnp.einsum("pk,pk->p", cotj, gd)
        dgJ_i = hpp[..., None, None] * dd * ci_gd[..., None, None] + hp[..., None, None] * jnp.einsum("pk,pkab->pab", coti, dgd)
        dgJ_j = -(
            hpp[..., None, None] * dd * cj_gd[..., None, None] + hp[..., None, None] * jnp.einsum("pk,pkab->pab", cotj, dgd)
        )
        dlap = (
            hppp[..., None, None] * dd * q[..., None, None]
            + hpp[..., None, None] * dq
            + hpp[..., None, None] * dd * L[..., None, None]
            + hp[..., None, None] * dL
        )
        # electron-i share weight w_i, electron-j share weight w_j; the shared -lap term (counted
        # for both electrons, coefficient -1 = 2 x -1/2) gets the pair weight 0.5(w_i + w_j).
        wi, wj = w[ii], w[jj]
        out = out + jnp.einsum(
            "pab->ab", -0.5 * (wi + wj)[:, None, None] * dlap - wi[:, None, None] * dgJ_i - wj[:, None, None] * dgJ_j
        )
    j3 = jd.jastrow_three_body_data
    if j3 is not None:
        out = out + _j3_kstress(j3, cot, r_up, r_dn, w)
    return out


def stress_eL_kinetic_pbc(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    r"""Periodic kinetic stress d T / d eps (3, 3).

    ``geminal_inv``: base-position running SM ``G^{-1}``.
    """
    wf = hamiltonian_data.wavefunction_data
    return grad_T_wrt_strain_orbital(
        wf.geminal_data, wf.jastrow_data, r_up, r_dn, geminal_inv=geminal_inv
    ) + grad_T_wrt_strain_jastrow(wf.jastrow_data, wf.geminal_data, r_up, r_dn, geminal_inv=geminal_inv)
