"""Autograd-free Ewald electrostatic force and strain derivatives.

These kernels replace the molecular bare-Coulomb electron-nucleus and
nucleus-nucleus derivatives for periodic systems. They use only JAX operations
(``erfc``, image meshes, and minimum-image mapping) so they compose inside the
``jit(vmap(...))`` walker batch -- the static lattice tables (T_n, G_m, g_fac, V) are
constant-folded; only the electron field is traced.

The differentiated channel energies are::

    E_ii = 1/2 Sum_{I,J,n}' Z_I Z_J erfc(eta d)/d  +  1/2 Sum_G g_G |S(G)|^2  (+ self/bg, R-indep)
    E_ei = -Sum_{i,I} Z_I [ Sum_n erfc(eta r)/r + Sum_G g_G cos(G.(r_i-R_I)) ]  (+ bg)
    with d=|R_IJ+T_n|, r=|r_i-R_I-T_n|, g'(x) = -2 eta/(sqrt(pi) x) e^{-eta^2 x^2} - erfc(eta x)/x^2.
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
from jax.scipy.special import erfc

from jqmc.coulomb_potential import Coulomb_potential_data, Ewald_data, _ewald_tables_from_cell
from jqmc.structure import _inv_3x3, _minimum_image_displacement

_EwaldTables: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
_EwaldEiPieces: TypeAlias = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
_EwaldEeImages: TypeAlias = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]


def _tables(ewd: Ewald_data, eta: jax.Array) -> _EwaldTables:
    """(T_n, G_m, g_fac, V) static lattice Ewald tables (jnp; cell + eta only)."""
    return _ewald_tables_from_cell(
        jnp.asarray(eta),
        ewd._real_image_indices_jnp,
        ewd._g_indices_jnp,
        jnp.asarray(ewd.structure_data._cell_jnp, jnp.float64),
    )


def _ii_pair_displacements(ewd: Ewald_data, pos: jax.Array) -> jax.Array:
    """Ion-ion pair displacements R_I - R_J folded to the parallelepiped-centred image (N,N,3).

    Mirrors the forward ``_compute_ewald_ii_energy_jax``: the real-space image list is complete only
    for ``|R_I - R_J| <= sum_i |a_i|``, so a nucleus given several cells outside the home cell would
    otherwise lose images (issue #52). ``d - round(d @ inv(cell)) @ cell`` keeps
    ``|d| <= sum_i |a_i| / 2``; the derivative of E_ii w.r.t. R or strain is unchanged by the fold.
    """
    cell = jnp.asarray(ewd.structure_data._cell_jnp, jnp.float64)
    R_IJ = pos[:, None, :] - pos[None, :, :]
    return R_IJ - jnp.round(R_IJ @ _inv_3x3(cell)) @ cell


def _gprime(eta: jax.Array, dd: jax.Array) -> jax.Array:
    """g'(x) of the real-space Ewald kernel g(x)=erfc(eta x)/x."""
    return -2.0 * eta / (jnp.sqrt(jnp.pi) * dd) * jnp.exp(-(eta**2) * dd**2) - erfc(eta * dd) / dd**2


def grad_Eii_wrt_R(cd: Coulomb_potential_data, ewd: Ewald_data) -> jax.Array:
    """Ion-ion Ewald force d E_ii / dR (n_atom, 3)."""
    Z = jnp.asarray(cd._effective_charges, jnp.float64)
    pos = jnp.asarray(ewd.structure_data._positions_cart_jnp, jnp.float64)
    eta = jnp.asarray(ewd.eta, jnp.float64)
    T_n, G_m, g_fac, _ = _tables(ewd, eta)
    # real-space (pair displacement folded first, as in the forward energy)
    v = _ii_pair_displacements(ewd, pos)[:, :, None, :] + T_n[None, None, :, :]  # (N,N,Nn,3)
    d = jnp.linalg.norm(v, axis=-1)
    nz = d > 1e-12
    dd = jnp.where(nz, d, 1.0)
    gp = jnp.where(nz, _gprime(eta, dd), 0.0)
    ZZ = Z[:, None] * Z[None, :]
    f_real = jnp.einsum("ij,ijn,ijnc->ic", ZZ, gp / dd, v)
    # reciprocal-space
    GR = G_m @ pos.T  # (Ng,N)
    S_real = jnp.cos(GR) @ Z
    S_imag = jnp.sin(GR) @ Z
    w = g_fac[:, None] * (S_imag[:, None] * jnp.cos(GR) - S_real[:, None] * jnp.sin(GR))  # (Ng,N)
    f_recip = Z[:, None] * jnp.einsum("gN,gc->Nc", w, G_m)
    return f_real + f_recip


def _ei_pieces(
    cd: Coulomb_potential_data,
    ewd: Ewald_data,
    r_up: jax.Array,
    r_dn: jax.Array,
) -> _EwaldEiPieces:
    Z = jnp.asarray(cd._effective_charges, jnp.float64)
    R = jnp.asarray(ewd.structure_data._positions_cart_jnp, jnp.float64)
    eta = jnp.asarray(ewd.eta, jnp.float64)
    T_n, G_m, g_fac, _ = _tables(ewd, eta)
    r_all = jnp.concatenate([jnp.asarray(r_up), jnp.asarray(r_dn)], 0)
    # Match the forward (compute_ewald_potential_ei): minimum-image r_iI BEFORE the real-space
    # image sum, else the T_n truncation drops the dominant erfc image when an electron sits in a
    # different periodic copy than the nucleus -- negligible on a tiny cell, ~5e-3 on a large one.
    cell = jnp.asarray(ewd.structure_data._cell_jnp, jnp.float64)
    inv_cell = _inv_3x3(cell)
    n_max = ewd.structure_data._n_max
    r_iI_raw = r_all[:, None, :] - R[None, :, :]  # (Ne,NI,3)
    r_iI = jax.vmap(jax.vmap(lambda d: _minimum_image_displacement(d, cell, inv_cell, n_max)))(r_iI_raw)
    w = r_iI[:, :, None, :] + T_n[None, None, :, :]  # (Ne,NI,Nn,3) -- forward uses min-image + T_n
    d = jnp.linalg.norm(w, axis=-1)
    nz = d > 1e-12
    dd = jnp.where(nz, d, 1.0)
    fac = jnp.where(nz, _gprime(eta, dd), 0.0) / dd  # (Ne,NI,Nn)
    # Reciprocal phases as PER-PARTICLE structure factors (G.r_i-G.R_I never combined into the
    # (n_Gvec, Ne, n_atom) tensor). The reciprocal sum is periodic-invariant (G.T_n = 2*pi*int),
    # so the RAW G.r_i / G.R_I phases are used (no min-image), matching the forward and the sibling
    # grad_Eee_wrt_r structure-factor reduction.
    Gr = G_m @ r_all.T  # (n_Gvec, Ne)   G_g . r_i
    GR = G_m @ R.T  # (n_Gvec, n_atom)    G_g . R_I
    return Z, G_m, g_fac, w, fac, Gr, GR


def grad_Eei_wrt_R(
    cd: Coulomb_potential_data,
    ewd: Ewald_data,
    r_up: jax.Array,
    r_dn: jax.Array,
) -> jax.Array:
    """Electron-ion Ewald force d E_ei / dR (n_atom, 3)."""
    Z, G_m, g_fac, w, fac, Gr, GR = _ei_pieces(cd, ewd, r_up, r_dn)
    real = jnp.einsum("I,iIn,iInc->Ic", Z, fac, w)
    # recip = -sum_{g,i} Z_I g_G sin(G.(r_i-R_I)) G, and sum_i sin(G.(r_i-R_I)) reduces over the
    # electron index FIRST via sin(G.(r_i-R_I)) = sin(G.r_i)cos(G.R_I) - cos(G.r_i)sin(G.R_I):
    #   sum_i sin(G.(r_i-R_I)) = Ss_g cos(G.R_I) - Sc_g sin(G.R_I),  Sc/Ss = sum_i cos/sin(G.r_i).
    Sc = jnp.cos(Gr).sum(1)  # (n_Gvec,)
    Ss = jnp.sin(Gr).sum(1)  # (n_Gvec,)
    sumI = Ss[:, None] * jnp.cos(GR) - Sc[:, None] * jnp.sin(GR)  # (n_Gvec, n_atom)
    recip = -jnp.einsum("I,g,gI,gc->Ic", Z, g_fac, sumI, G_m)
    return real + recip


def grad_Eei_wrt_r(
    cd: Coulomb_potential_data,
    ewd: Ewald_data,
    r_up: jax.Array,
    r_dn: jax.Array,
) -> jax.Array:
    """Electron-ion Ewald drift d E_ei / dr (Ne, 3) -- for the periodic e_L/r path."""
    Z, G_m, g_fac, w, fac, Gr, GR = _ei_pieces(cd, ewd, r_up, r_dn)
    real = -jnp.einsum("I,iIn,iInc->ic", Z, fac, w)
    # Per-ELECTRON output -> reduce the ION index first (Z-weighted structure factors):
    #   sum_I Z_I sin(G.(r_i-R_I)) = Zc_g sin(G.r_i) - Zs_g cos(G.r_i),
    #   Zc/Zs = sum_I Z_I cos/sin(G.R_I).  Memory n_Gvec*(Ne+n_atom), never the (g,i,I) triple.
    Zc = jnp.cos(GR) @ Z  # (n_Gvec,)
    Zs = jnp.sin(GR) @ Z  # (n_Gvec,)
    sumr = Zc[:, None] * jnp.sin(Gr) - Zs[:, None] * jnp.cos(Gr)  # (n_Gvec, Ne)
    recip = jnp.einsum("g,gi,gc->ic", g_fac, sumr, G_m)
    return real + recip


def grad_V_ewald_wrt_R(coulomb_data: Coulomb_potential_data, r_up: jax.Array, r_dn: jax.Array) -> jax.Array:
    """Total periodic electrostatic HF force d(E_ii + E_ei)/dR (n_atom, 3)."""
    ewd = coulomb_data.ewald_data
    return grad_Eii_wrt_R(coulomb_data, ewd) + grad_Eei_wrt_R(coulomb_data, ewd, r_up, r_dn)


def _recip_strain_bracket(G_m: jax.Array, g_fac: jax.Array, eta: jax.Array) -> jax.Array:
    """g_G [ -I + G(x)G/(2 eta^2) + 2 G(x)G/|G|^2 ] (Ng,3,3): the reciprocal-space strain
    kernel shared by the ion-ion and electron-ion Ewald stress."""
    G2 = jnp.sum(G_m * G_m, axis=1)
    GG = jnp.einsum("gc,gd->gcd", G_m, G_m)
    return g_fac[:, None, None] * (-jnp.eye(3)[None] + GG / (2.0 * eta**2) + 2.0 * GG / G2[:, None, None])


def grad_Eii_wrt_strain(coulomb_data: Coulomb_potential_data, ewd: Ewald_data) -> jax.Array:
    """Ion-ion Ewald stress d E_ii / d eps (3, 3). Affine strain (G.R invariant)."""
    Z = jnp.asarray(coulomb_data._effective_charges, jnp.float64)
    pos = jnp.asarray(ewd.structure_data._positions_cart_jnp, jnp.float64)
    eta = jnp.asarray(ewd.eta, jnp.float64)
    T_n, G_m, g_fac, V = _tables(ewd, eta)
    v = _ii_pair_displacements(ewd, pos)[:, :, None, :] + T_n[None, None, :, :]
    d = jnp.linalg.norm(v, axis=-1)
    nz = d > 1e-12
    dd = jnp.where(nz, d, 1.0)
    gp = jnp.where(nz, _gprime(eta, dd), 0.0)
    ZZ = Z[:, None] * Z[None, :]
    s_real = 0.5 * jnp.einsum("ij,ijn,ijnc,ijnd->cd", ZZ, gp / dd, v, v)
    GR = G_m @ pos.T
    S2 = (jnp.cos(GR) @ Z) ** 2 + (jnp.sin(GR) @ Z) ** 2
    s_recip = 0.5 * jnp.einsum("g,gcd->cd", S2, _recip_strain_bracket(G_m, g_fac, eta))
    s_bg = (jnp.pi / (2.0 * eta**2 * V)) * jnp.sum(Z) ** 2 * jnp.eye(3)
    return s_real + s_recip + s_bg


def grad_Eei_wrt_strain(
    coulomb_data: Coulomb_potential_data,
    ewd: Ewald_data,
    r_up: jax.Array,
    r_dn: jax.Array,
) -> jax.Array:
    """Electron-ion Ewald stress d E_ei / d eps (3, 3). Electrons + ions + cell deform."""
    Z = jnp.asarray(coulomb_data._effective_charges, jnp.float64)
    R = jnp.asarray(ewd.structure_data._positions_cart_jnp, jnp.float64)
    eta = jnp.asarray(ewd.eta, jnp.float64)
    T_n, G_m, g_fac, V = _tables(ewd, eta)
    r_all = jnp.concatenate([jnp.asarray(r_up), jnp.asarray(r_dn)], 0)
    Ne = r_all.shape[0]
    # Min-image r_iI before the real-space image sum (match the forward; same fix as _ei_pieces).
    cell = jnp.asarray(ewd.structure_data._cell_jnp, jnp.float64)
    inv_cell = _inv_3x3(cell)
    n_max = ewd.structure_data._n_max
    r_iI = jax.vmap(jax.vmap(lambda dd_: _minimum_image_displacement(dd_, cell, inv_cell, n_max)))(
        r_all[:, None, :] - R[None, :, :]
    )
    w = r_iI[:, :, None, :] + T_n[None, None, :, :]
    d = jnp.linalg.norm(w, axis=-1)
    nz = d > 1e-12
    dd = jnp.where(nz, d, 1.0)
    gp = jnp.where(nz, _gprime(eta, dd), 0.0)
    s_real = -jnp.einsum("I,iIn,iInc,iInd->cd", Z, gp / dd, w, w)
    # Both the electron and ion indices are fully reduced -> pre-contract into per-G structure
    # factors (never the (g,i,I) phase tensor). With cos(G.(r_i-R_I)) = cos(G.r_i)cos(G.R_I)
    # + sin(G.r_i)sin(G.R_I): sum_{i,I} Z_I cos(G.(r_i-R_I)) = Sc_g Zc_g + Ss_g Zs_g.
    Gr = G_m @ r_all.T  # (n_Gvec, Ne)
    GR = G_m @ R.T  # (n_Gvec, n_atom)
    Sc = jnp.cos(Gr).sum(1)  # (n_Gvec,)  sum_i cos(G.r_i)
    Ss = jnp.sin(Gr).sum(1)  # (n_Gvec,)  sum_i sin(G.r_i)
    Zc = jnp.cos(GR) @ Z  # (n_Gvec,)     sum_I Z_I cos(G.R_I)
    Zs = jnp.sin(GR) @ Z  # (n_Gvec,)     sum_I Z_I sin(G.R_I)
    cosfac = Sc * Zc + Ss * Zs  # (n_Gvec,) = sum_{i,I} Z_I cos(G.(r_i-R_I))
    s_recip = -jnp.einsum("g,gcd->cd", cosfac, _recip_strain_bracket(G_m, g_fac, eta))
    s_bg = -(jnp.pi / (eta**2 * V)) * Ne * jnp.sum(Z) * jnp.eye(3)
    return s_real + s_recip + s_bg


def _ee_images(ewd: Ewald_data, r_up: jax.Array, r_dn: jax.Array) -> _EwaldEeImages:
    cell = jnp.asarray(ewd.structure_data._cell_jnp, jnp.float64)
    inv_cell = _inv_3x3(cell)
    n_max = ewd.structure_data._n_max
    eta = jnp.asarray(ewd.eta, jnp.float64)
    T_n, G_m, g_fac, V = _tables(ewd, eta)
    r_all = jnp.concatenate([jnp.asarray(r_up), jnp.asarray(r_dn)], 0)
    r_ij = r_all[:, None, :] - r_all[None, :, :]
    r_ij_w = jax.vmap(jax.vmap(lambda d: _minimum_image_displacement(d, cell, inv_cell, n_max)))(r_ij)
    v = r_ij_w[:, :, None, :] + T_n[None, None, :, :]  # (N,N,Nn,3)
    d = jnp.linalg.norm(v, axis=-1)
    nz = d > 1e-12
    dd = jnp.where(nz, d, 1.0)
    gp = jnp.where(nz, _gprime(eta, dd), 0.0)
    return r_all, G_m, g_fac, v, gp, dd, V, eta


def grad_Eee_wrt_r(ewd: Ewald_data, r_up: jax.Array, r_dn: jax.Array) -> jax.Array:
    """Electron-electron Ewald drift d E_ee / d r (Ne, 3) (charges q=-1, q_i q_j = 1)."""
    r_all, G_m, g_fac, v, gp, dd, _, _ = _ee_images(ewd, r_up, r_dn)
    f_real = jnp.einsum("ijn,ijnc->ic", gp / dd, v)
    GR = G_m @ r_all.T
    S_real = jnp.cos(GR).sum(1)
    S_imag = jnp.sin(GR).sum(1)
    w = g_fac[:, None] * (S_imag[:, None] * jnp.cos(GR) - S_real[:, None] * jnp.sin(GR))  # (Ng,N)
    return f_real + jnp.einsum("gN,gc->Nc", w, G_m)


def grad_Eee_wrt_strain(ewd: Ewald_data, r_up: jax.Array, r_dn: jax.Array) -> jax.Array:
    """Electron-electron Ewald stress d E_ee / d eps (3, 3)."""
    r_all, G_m, g_fac, v, gp, dd, V, eta = _ee_images(ewd, r_up, r_dn)
    N = r_all.shape[0]
    s_real = 0.5 * jnp.einsum("ijn,ijnc,ijnd->cd", gp / dd, v, v)
    GR = G_m @ r_all.T
    S2 = jnp.cos(GR).sum(1) ** 2 + jnp.sin(GR).sum(1) ** 2
    s_recip = 0.5 * jnp.einsum("g,gcd->cd", S2, _recip_strain_bracket(G_m, g_fac, eta))
    s_bg = (jnp.pi / (2.0 * eta**2 * V)) * N**2 * jnp.eye(3)
    return s_real + s_recip + s_bg


def grad_V_ewald_wrt_strain(coulomb_data: Coulomb_potential_data, r_up: jax.Array, r_dn: jax.Array) -> jax.Array:
    """Total periodic electrostatic stress d(E_ii + E_ei + E_ee)/d eps (3, 3)."""
    ewd = coulomb_data.ewald_data
    return (
        grad_Eii_wrt_strain(coulomb_data, ewd)
        + grad_Eei_wrt_strain(coulomb_data, ewd, r_up, r_dn)
        + grad_Eee_wrt_strain(ewd, r_up, r_dn)
    )
