"""Hellmann-Feynman force d e_L/dR = dT/dR + dV/dR.

dT/dR: with-Jastrow kinetic force -- the kinetic VJP cotangents (kinetic.py) converted
to nuclear forces by the translation identity, needing the AO 3rd derivative
grad(laplacian) + Hessian (ao_deriv.py). Orbital side (det) + Jastrow side (J1 radial;
J2 has no R; J3 orbital centers). dV/dR contains the classical electrostatic
electron-nucleus and nucleus-nucleus force. ECP non-local and periodic Ewald
contributions are assembled by their dedicated derivative modules.
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

from jqmc.coulomb_potential import Coulomb_potential_data
from jqmc.determinant import AGP_data, AGPn_data, SD_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    compute_grads_and_laplacian_Jastrow_part,
)
from jqmc.structure import Structure_data

from .ao_deriv import ao_grad, ao_gradlap, ao_hessian, j3_basis_derivs, orb_to_cart_pieces
from .kinetic import _geminal_reverse

_GeminalData: TypeAlias = AGP_data | AGPn_data | SD_data
_ReverseCotangents: TypeAlias = dict[str, jax.Array | None]
_ArrayTriple: TypeAlias = tuple[jax.Array, jax.Array, jax.Array]
_SmoothKinematics: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


def _scatter(s_dc: jax.Array, nucleus_index: jax.Array, n_atom: int) -> jax.Array:
    idx = jnp.asarray(nucleus_index, jnp.int32)
    return jax.ops.segment_sum(s_dc.T, idx, num_segments=n_atom)


def _orbital_side(
    gd: _GeminalData,
    r_up: jax.Array,
    r_dn: jax.Array,
    b: _ReverseCotangents,
    n_atom: int,
) -> jax.Array:
    out = jnp.zeros((n_atom, 3))
    sides = (
        ("orb_data_up_spin", b["bar_orb_up"], b["bar_lap_up"], b["bar_gou"], r_up),
        ("orb_data_dn_spin", b["bar_orb_dn"], b["bar_lap_dn"], b["bar_godn"], r_dn),
    )
    for side, bar_orb, bar_lap, bar_gou, r in sides:
        aos, C = orb_to_cart_pieces(getattr(gd, side))
        CM_orb = C.T @ bar_orb
        CM_lap = C.T @ bar_lap
        CM_gou = jnp.einsum("ac,dak->dck", C, bar_gou)
        gA = ao_grad(aos, r)
        gL = ao_gradlap(aos, r)
        H = ao_hessian(aos, r)
        s = -(
            jnp.einsum("ck,dck->dc", CM_orb, gA) + jnp.einsum("ck,dck->dc", CM_lap, gL) + jnp.einsum("eck,deck->dc", CM_gou, H)
        )
        out = out + jnp.real(_scatter(s, aos._nucleus_index_jnp, n_atom))
    return out


def _orbital_side_pbc(
    gd: _GeminalData,
    r_up: jax.Array,
    r_dn: jax.Array,
    b: _ReverseCotangents,
    n_atom: int,
) -> jax.Array:
    """Periodic counterpart of _orbital_side: same translation-identity contraction, but with
    the folded pgto AO grad/gradlap/Hessian (pgto_deriv) and the ragged home-atom scatter.
    EAGER (host-side numpy); the cotangents may be complex (twist) -> the force takes Re."""
    from . import pgto_deriv as P
    from .assemble import _pgto_home_atom

    out = jnp.zeros((n_atom, 3))
    sides = (
        ("orb_data_up_spin", b["bar_orb_up"], b["bar_lap_up"], b["bar_gou"], r_up),
        ("orb_data_dn_spin", b["bar_orb_dn"], b["bar_lap_dn"], b["bar_godn"], r_dn),
    )
    for side, bar_orb, bar_lap, bar_gou, r in sides:
        from jqmc.orbital_tiling import TiledOrbitalSet_data
        from .orbital_tiling import tiled_field_vjp

        orb = getattr(gd, side)
        if isinstance(orb, TiledOrbitalSet_data):
            out = out + jnp.real(tiled_field_vjp(orb, r, bar_orb, bar_gou, bar_lap, wrt="R"))
            continue
        for leaf, C_leaf, kind in P.orbital_leaves(getattr(gd, side)):
            if kind != "pgto":
                continue  # plane-wave / B-spline AOs carry no nuclear dependence -> dAO/dR = 0
            aos, T_mol = P.sphe_leaf_to_cart(leaf)  # de-tile + c2s -> cart leaf for the pgto kernels
            C = jnp.asarray(C_leaf) @ jnp.asarray(T_mol)  # (num_mo, num_cart)
            CM_orb = C.T @ jnp.asarray(bar_orb)  # (num_cart, N)
            CM_lap = C.T @ jnp.asarray(bar_lap)
            CM_gou = jnp.einsum("ac,dak->dck", C, jnp.asarray(bar_gou))  # (3, num_cart, N)
            gA = P.pgto_grad(aos, r)  # (3, num_cart, N)
            gL = P.pgto_gradlap(aos, r)
            H = P.pgto_hessian(aos, r)  # (3, 3, num_cart, N)
            s = -(
                jnp.einsum("ck,dck->dc", CM_orb, gA)
                + jnp.einsum("ck,dck->dc", CM_lap, gL)
                + jnp.einsum("eck,deck->dc", CM_gou, H)
            )  # (3, num_cart)
            # home-atom scatter: each cart-AO's static home atom -> segment_sum (the molecular
            # _scatter, but keyed on the pgto home-atom index instead of _nucleus_index_jnp).
            atoms = jnp.asarray(_pgto_home_atom(aos), jnp.int32)  # (num_cart,)
            out = out + jnp.real(_scatter(s, atoms, n_atom))
    return out


def _j1_radial(j1: Jastrow_one_body_data, dist: jax.Array) -> _ArrayTriple:
    Z = jnp.asarray(j1.structure_data.atomic_numbers) - jnp.asarray(j1.core_electrons)
    p = j1.jastrow_1b_param
    c = ((2.0 * Z) ** 0.25)[None, :]
    pref = (-((2.0 * Z) ** 0.75) if j1.jastrow_1b_type == "exp" else -2.0 * Z)[None, :]
    if j1.jastrow_1b_type == "exp":
        a = p * c
        e = jnp.exp(-a * dist)
        fp = pref * c / 2.0 * e
        fpp = -pref * (p * c**2) / 2.0 * e
        fppp = pref * (p**2 * c**3) / 2.0 * e
    elif j1.jastrow_1b_type == "pade":
        bb = p * c
        u = 1.0 + bb * dist
        fp = pref / (2.0 * u**2)
        fpp = -pref * bb / u**3
        fppp = 3.0 * pref * bb**2 / u**4
    else:
        raise ValueError(j1.jastrow_1b_type)
    hp = fppp + 2.0 * fpp / dist - 2.0 * fp / dist**2
    return fp, fpp, hp


def _j1_side(
    j1: Jastrow_one_body_data,
    r_all: jax.Array,
    bar_gJ: jax.Array,
    bar_lapJ: jax.Array,
    n_atom: int,
) -> jax.Array:
    R = jnp.asarray(j1.structure_data.positions)
    diff = r_all[:, None, :] - R[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)
    u = diff / dist[..., None]
    fp, fpp, hp = _j1_radial(j1, dist)
    bg = jnp.asarray(bar_gJ)
    bu = jnp.einsum("ed,eid->ei", bg, u)
    term_g = -(fpp[..., None] * bu[..., None] * u + (fp / dist)[..., None] * (bg[:, None, :] - bu[..., None] * u))
    term_l = -(jnp.asarray(bar_lapJ)[:, None, None]) * (hp[..., None] * u)
    return jnp.asarray((term_g + term_l).sum(axis=0))


def _j1_radial_np(j1: Jastrow_one_body_data, dist: jax.Array) -> _ArrayTriple:
    """(f', f'', f''') of the J1 radial term vs distance (jnp, periodic J1 force)."""
    Z = jnp.asarray(j1.structure_data.atomic_numbers, jnp.float64) - jnp.asarray(j1.core_electrons, jnp.float64)
    p = j1.jastrow_1b_param  # (possibly traced) scalar -- no host float()
    c = ((2.0 * Z) ** 0.25)[None, :]
    pref = (-((2.0 * Z) ** 0.75) if j1.jastrow_1b_type == "exp" else -2.0 * Z)[None, :]
    if j1.jastrow_1b_type == "exp":
        a = p * c
        e = jnp.exp(-a * dist)
        return pref * c / 2.0 * e, -pref * (p * c**2) / 2.0 * e, pref * (p**2 * c**3) / 2.0 * e
    if j1.jastrow_1b_type == "pade":
        bb = p * c
        u = 1.0 + bb * dist
        return pref / (2.0 * u**2), -pref * bb / u**3, 3.0 * pref * bb**2 / u**4
    raise ValueError(j1.jastrow_1b_type)


def _smooth_dist_kinematics(sd: Structure_data, disp: jax.Array) -> _SmoothKinematics:
    """(d, grad_d, q, Hess_d, lap_d, gradlap_d) of the positive full-Gram periodic distance for
    a displacement field ``disp`` (..., 3), in closed form (autograd-free). grad/Hess/lap are
    d/d(disp). q = |grad_d|^2. Reduces to the Euclidean forms (grad_d=unit, Hess=(I-uu)/d,
    lap=2/d, gradlap=-2u/d^2) near coincidence in every nonsingular cell."""
    if not bool(sd.pbc_flag):
        # Molecular (open-boundary) Jastrow: the smooth periodic distance IS the plain Euclidean
        # distance. A molecular J1/J2 built with no lattice carries an empty cell (``_cell_jnp`` has
        # shape ``(3, 0)``), so the periodic branch's ``_inv_3x3`` would fail; and the FORWARD
        # (``compute_grads_and_laplacian_Jastrow_part``, which dispatches on the Jastrow's OWN
        # ``structure_data.pbc_flag``) uses the Euclidean e-e/e-N distance for such a Jastrow -- so the
        # analytic drift/stress must too. These are exactly the closed-form Euclidean limits above.
        d2 = jnp.sum(disp * disp, axis=-1)
        pos = d2 > 0.0
        dsafe = jnp.sqrt(jnp.where(pos, d2, 1.0))
        d = jnp.where(pos, dsafe, 0.0)
        u = disp / dsafe[..., None]  # grad_d = unit displacement
        eye = jnp.eye(3, dtype=u.dtype)
        Hess = (eye - u[..., :, None] * u[..., None, :]) / dsafe[..., None, None]
        lap_d = 2.0 / dsafe
        gradlap_d = -2.0 * u / (dsafe**2)[..., None]
        q = jnp.sum(u * u, axis=-1)
        return d, u, q, Hess, lap_d, gradlap_d

    from jqmc.structure import _inv_3x3

    cell = jnp.asarray(sd._cell_jnp, jnp.float64)  # static lattice (constant-folded under jit)
    inv = _inv_3x3(cell)  # (3,3) [m,k]
    from jqmc.jastrow_factor import _periodic_geometry_components

    S, numerator, trace = _periodic_geometry_components(disp, cell, inv)
    d = jnp.sqrt(jnp.where(S > 0, S, 1.0))
    dsafe = jnp.where(S > 0, d, 1.0)
    d = jnp.where(S > 0, d, 0.0)
    grad_d = jnp.stack(numerator, axis=-1) / dsafe[..., None]
    q = jnp.sum(grad_d * grad_d, axis=-1)
    local = disp - jnp.rint(disp @ inv) @ cell
    theta = jnp.pi * (local @ inv)
    sine, cosine = jnp.sin(theta), jnp.cos(theta)
    z, h = sine * sine, sine * cosine
    t = 1 - 2 * z
    gram, inverse_gram = cell @ cell.T, inv.T @ inv
    diagonal, inverse_diagonal = jnp.diag(gram), jnp.diag(inverse_gram)
    off = gram - jnp.diag(diagonal)
    off_h = h @ off
    # Half-Hessian of d^2, with the Cartesian identity explicit at the cusp.
    transform = jnp.eye(3, dtype=cell.dtype) - 2 * jnp.einsum("ik,...k,kl->...il", inv, z, cell)
    weights = 2 * diagonal * t * z - 4 * h * off_h
    half_hessian = transform @ jnp.swapaxes(transform, -1, -2)
    half_hessian += jnp.einsum("ik,...k,jk->...ij", inv, weights, inv)
    Hess = (half_hessian - grad_d[..., :, None] * grad_d[..., None, :]) / dsafe[..., None, None]
    lap_d = (trace - q) / dsafe
    trace_theta = 2 * (2 * diagonal * inverse_diagonal - 4) * h
    trace_theta += 16 * h * (z @ (off * inverse_gram))
    trace_theta -= 4 * t * (h @ (off * (inverse_diagonal[:, None] + inverse_diagonal[None, :])))
    grad_trace = jnp.pi * (trace_theta @ inv.T)
    grad_q = 2 * jnp.einsum("...ij,...j->...i", Hess, grad_d)
    gradlap_d = (grad_trace - grad_q - lap_d[..., None] * grad_d) / dsafe[..., None]
    return d, grad_d, q, Hess, lap_d, gradlap_d


def _j1_side_pbc(
    j1: Jastrow_one_body_data,
    r_all: jax.Array,
    bar_gJ: jax.Array,
    bar_lapJ: jax.Array,
    n_atom: int,
) -> jax.Array:
    """Periodic e-N J1 kinetic force (eager). Same translation-identity assembly as _j1_side
    but with the smooth periodic distance's grad/Hessian/laplacian/grad-of-laplacian:
      d gJ1_e/dR = -[f'' grad_d (x) grad_d + f' Hess_d]
      d lapJ1_e/dR = -[f''' q grad_d + 2 f'' Hess_d.grad_d + f'' lap_d grad_d + f' grad(lap_d)]
    contracted with (bar_gJ, bar_lapJ) and summed over electrons. Re for complex (twist)."""
    R = jnp.asarray(j1.structure_data.positions, jnp.float64)
    r = jnp.asarray(r_all, jnp.float64)
    disp = r[:, None, :] - R[None, :, :]  # (Ne, nA, 3) = r - R
    d, gd, q, H, lap, glap = _smooth_dist_kinematics(j1.structure_data, disp)
    fp, fpp, fppp = _j1_radial_np(j1, d)  # (Ne, nA)
    bg = jnp.asarray(bar_gJ)  # (Ne, 3), maybe complex
    bl = jnp.asarray(bar_lapJ)  # (Ne,)
    bg_gd = jnp.einsum("ed,eId->eI", bg, gd)
    bg_H = jnp.einsum("ed,eImd->eIm", bg, H)
    term_g = -(fpp[..., None] * bg_gd[..., None] * gd + fp[..., None] * bg_H)
    H_gd = jnp.einsum("eImn,eIn->eIm", H, gd)
    term_l = -bl[:, None, None] * (
        fppp[..., None] * q[..., None] * gd
        + 2.0 * fpp[..., None] * H_gd
        + fpp[..., None] * lap[..., None] * gd
        + fp[..., None] * glap
    )
    return jnp.real((term_g + term_l).sum(axis=0))  # (nA=n_atom, 3)


def _j3_side(
    j3: Jastrow_three_body_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    bgu: jax.Array,
    bgd: jax.Array,
    blu: jax.Array,
    bld: jax.Array,
    n_atom: int,
) -> jax.Array:
    from .assemble import _j3_mo_to_bare_ao

    j3 = _j3_mo_to_bare_ao(j3)  # real MO lowers to AO; a complex MO stays at orbital rank
    aos = j3.orb_data
    A_up, gOu, lOu, Hu, GLu = j3_basis_derivs(aos, r_up)
    A_dn, gOd, lOd, Hd, GLd = j3_basis_derivs(aos, r_dn)
    n_up, n_dn = A_up.shape[1], A_dn.shape[1]
    Ku = jnp.tril(jnp.ones((n_up, n_up)), -1)
    Kd = jnp.tril(jnp.ones((n_dn, n_dn)), -1)
    Jm = j3._j_matrix_jnp
    M = Jm[:, :-1]
    j1 = Jm[:, -1]
    s_up, s_dn = A_up.sum(1), A_dn.sum(1)
    coefA_up = j1[:, None] + M @ A_up @ Ku + M.T @ A_up @ Ku.T + (M @ s_dn)[:, None]
    coefA_dn = j1[:, None] + M @ A_dn @ Kd + M.T @ A_dn @ Kd.T + (M.T @ s_up)[:, None]
    # The J3 orbital fields are real (Re(C @ AO)), so e_L = Re(...) depends on them only through
    # Re of their cotangents: Re(bar . field) = Re(bar) . field.  The drift cotangent
    # bar_gJ = -(gJ + gD) is complex on a twist (complex determinant drift gD), so project it here,
    # as the J3 kinetic stress (grad_T_wrt_strain_jastrow) and the J3 basis-parameter path
    # (basis.grad_T_wrt_J3_basis) already do.  Keeping it complex is harmless on the AO-rank path
    # (real fields, Re at the end) but WRONG on the complex-MO rank path below, where the cotangent
    # meets the complex C @ dAO and the final Re leaks a spurious -Im(bar) . Im(C dAO) term (#90).
    bgu, bgd = jnp.real(jnp.asarray(bgu)), jnp.real(jnp.asarray(bgd))
    blu, bld = jnp.real(jnp.asarray(blu)), jnp.real(jnp.asarray(bld))

    def side_cot(
        coefA: jax.Array,
        bg: jax.Array,
        bl: jax.Array,
        gO: jax.Array,
        lO: jax.Array,
    ) -> _ArrayTriple:
        bar_gO = jnp.einsum("kd,ak->dak", bg, coefA)
        bar_lO = bl[None, :] * coefA
        bar_coefA = jnp.einsum("kd,dak->ak", bg, gO) + bl[None, :] * lO
        return bar_gO, bar_lO, bar_coefA

    bgO_u, blO_u, bcA_u = side_cot(coefA_up, bgu, blu, gOu, lOu)
    bgO_d, blO_d, bcA_d = side_cot(coefA_dn, bgd, bld, gOd, lOd)
    bar_A_up = M.T @ bcA_u @ Ku.T + M @ bcA_u @ Ku + (M @ bcA_d.sum(1))[:, None]
    bar_A_dn = M.T @ bcA_d @ Kd.T + M @ bcA_d @ Kd + (M.T @ bcA_u.sum(1))[:, None]

    def force_side(
        bar_A: jax.Array,
        bar_gO: jax.Array,
        bar_lO: jax.Array,
        gO: jax.Array,
        H: jax.Array,
        GL: jax.Array,
    ) -> jax.Array:
        return -(
            jnp.einsum("ak,dak->da", bar_A, gO) + jnp.einsum("eak,deak->da", bar_gO, H) + jnp.einsum("ak,dak->da", bar_lO, GL)
        )

    mo_rank = type(aos).__name__ == "MOs_data"
    if not mo_rank:
        su = force_side(bar_A_up, bgO_u, blO_u, gOu, Hu, GLu)
        sd = force_side(bar_A_dn, bgO_d, blO_d, gOd, Hd, GLd)
    # Leaf-aware home-atom scatter: molecular GTO -> _nucleus_index_jnp (per-AO, unchanged);
    # periodic pgto -> the folded (molecular-rank) ragged home atom (the per-image
    # _nucleus_index_jnp is kernel rank and would mismatch the folded j3_basis_derivs rank).
    # A composite descends to its children by AO-row offset: only atom-centred GTO/pGTO children
    # scatter (the translation identity dAO/dR = -dAO/dr holds only for atom-centred AOs);
    # plane-wave / B-spline rows carry NO nuclear dependence -> excluded, exactly as the det force.
    # The coupling (coefA / bar_*) above stays at the full composite rank -- J3 couples across
    # children -- only the per-AO force rows are split. Single leaf == one child (bit-identical).
    from .assemble import _pgto_home_atom
    from .pgto_deriv import orbital_leaves

    out = jnp.zeros((n_atom, 3), dtype=bar_A_up.dtype)
    off = 0
    for leaf, C, kind in orbital_leaves(aos):
        n = int(leaf.num_ao)
        if kind in ("pw", "bspline"):
            off += n
            continue
        idx = _pgto_home_atom(leaf)
        if mo_rank:
            from .ao_deriv import j3_gto_basis_derivs_raw

            def mo_side(
                bar_A: jax.Array,
                bar_gO: jax.Array,
                bar_lO: jax.Array,
                r: jax.Array,
            ) -> jax.Array:
                _, grad, _, hess, gradlap = j3_gto_basis_derivs_raw(leaf, r)
                cT = jnp.asarray(C).T
                ba = cT @ bar_A
                blo = cT @ bar_lO
                bgo = jnp.einsum("am,dmk->dak", cT, bar_gO)
                return -jnp.real(
                    jnp.einsum("ak,dak->da", ba, grad)
                    + jnp.einsum("eak,deak->da", bgo, hess)
                    + jnp.einsum("ak,dak->da", blo, gradlap)
                )

            su_leaf = mo_side(bar_A_up, bgO_u, blO_u, r_up)
            sd_leaf = mo_side(bar_A_dn, bgO_d, blO_d, r_dn)
        else:
            su_leaf = su[:, off : off + n]
            sd_leaf = sd[:, off : off + n]
        out = out + jax.ops.segment_sum(su_leaf.T, idx, num_segments=n_atom)
        out = out + jax.ops.segment_sum(sd_leaf.T, idx, num_segments=n_atom)
        off += n
    # The cotangents are real (projected above); the complex-MO rank path contracts them with the
    # complex C @ dAO, whose real part is the derivative of the real J3 field Re(C @ AO).
    return jnp.real(out)


def grad_T_wrt_R_withJ(
    gd: AGP_data | AGPn_data | SD_data,
    jastrow_data: Jastrow_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    # Reuse the running inverse in both the kinetic reverse pass and drift
    # field so the nuclear-force graph does not lower a fresh SVD-pinv.
    gJ_up, gJ_dn, _, _ = compute_grads_and_laplacian_Jastrow_part(jastrow_data, r_up, r_dn)
    b = _geminal_reverse(gd, r_up, r_dn, gJ_up, gJ_dn, geminal_inv=geminal_inv)
    up_orbital = gd.orb_data_up_spin
    aos_up = up_orbital.aos_data if type(up_orbital).__name__ == "MOs_data" else up_orbital
    n_atom = jnp.asarray(aos_up.structure_data.positions).shape[0]
    pbc = bool(aos_up.structure_data.pbc_flag)
    force = _orbital_side_pbc(gd, r_up, r_dn, b, n_atom) if pbc else _orbital_side(gd, r_up, r_dn, b, n_atom)
    # The reverse pass already formed the determinant drift on the fused
    # kinetic primal.  Reuse it here instead of evaluating the standalone
    # determinant-ratio boundary again.
    gD_up, gD_dn = b["gD_up"], b["gD_dn"]
    gJ = jnp.concatenate([jnp.asarray(gJ_up), jnp.asarray(gJ_dn)], 0)
    gD = jnp.concatenate([jnp.asarray(gD_up), jnp.asarray(gD_dn)], 0)
    bar_gJ = -(gJ + gD)
    bar_lapJ = -0.5 * jnp.ones(len(bar_gJ))
    r_all = jnp.concatenate([jnp.asarray(r_up), jnp.asarray(r_dn)], 0)
    if jastrow_data.jastrow_one_body_data is not None:
        j1side = _j1_side_pbc if pbc else _j1_side
        force = force + j1side(jastrow_data.jastrow_one_body_data, r_all, bar_gJ, bar_lapJ, n_atom)
    if jastrow_data.jastrow_three_body_data is not None:
        n_up = len(r_up)
        force = force + _j3_side(
            jastrow_data.jastrow_three_body_data,
            r_up,
            r_dn,
            bar_gJ[:n_up],
            bar_gJ[n_up:],
            bar_lapJ[:n_up],
            bar_lapJ[n_up:],
            n_atom,
        )
    return force


def grad_Vbare_wrt_r(coulomb_data: Coulomb_potential_data, r_up: jax.Array, r_dn: jax.Array) -> jax.Array:
    """Classical electron-coordinate force dV_bare/dr (e-e repulsion + e-N attraction),
    all-electron. Per electron e: ``sum_{e'!=e} -(r_e-r_e')/|.|^3 + sum_I Z_I (r_e-R_I)/|.|^3``."""
    sd = coulomb_data.structure_data
    R = jnp.asarray(sd.positions)
    Z = jnp.asarray(sd.atomic_numbers)
    zc = np.asarray(coulomb_data.z_cores, dtype=np.float64)
    if zc.size:
        Z = Z - jnp.asarray(zc)
    r_all = jnp.concatenate([jnp.asarray(r_up), jnp.asarray(r_dn)], axis=0)
    d_ee = r_all[:, None, :] - r_all[None, :, :]
    dist = jnp.linalg.norm(d_ee, axis=-1)
    dist = jnp.where(jnp.eye(dist.shape[0], dtype=bool), jnp.inf, dist)
    g_ee = -jnp.einsum("eod,eo->ed", d_ee, dist**-3)
    d_eN = r_all[:, None, :] - R[None, :, :]
    inv3 = jnp.linalg.norm(d_eN, axis=-1) ** -3
    g_eN = jnp.einsum("eId,eI->ed", d_eN, Z[None, :] * inv3)
    return jnp.asarray(g_ee + g_eN)


def grad_V_wrt_R(coulomb_data: Coulomb_potential_data, r_up: jax.Array, r_dn: jax.Array) -> jax.Array:
    """Electrostatic HF force d V / dR. PBC -> Ewald lattice-sum force d(E_ii + E_ei)/dR
    (ewald.grad_V_ewald_wrt_R); molecular -> bare 1/r e-N + N-N force."""
    if coulomb_data.structure_data.pbc_flag:
        from . import ewald

        return ewald.grad_V_ewald_wrt_R(coulomb_data, r_up, r_dn)
    sd = coulomb_data.structure_data
    R = jnp.asarray(sd.positions)
    Z = jnp.asarray(sd.atomic_numbers)
    zc = np.asarray(coulomb_data.z_cores, dtype=np.float64)
    if zc.size:
        Z = Z - jnp.asarray(zc)
    r_all = jnp.concatenate([jnp.asarray(r_up), jnp.asarray(r_dn)], axis=0)
    d_eR = r_all[:, None, :] - R[None, :, :]
    inv3 = jnp.linalg.norm(d_eR, axis=-1)[..., None] ** -3
    grad_eN = -Z[:, None] * jnp.einsum("ead,ea->ad", d_eR, inv3[..., 0])
    d_RR = R[:, None, :] - R[None, :, :]
    dist = jnp.linalg.norm(d_RR, axis=-1)
    dist = jnp.where(jnp.eye(dist.shape[0], dtype=bool), jnp.inf, dist)
    inv3_RR = dist[..., None] ** -3
    ZZ = Z[:, None] * Z[None, :]
    grad_NN = -jnp.einsum("ij,ijd->id", ZZ, d_RR * inv3_RR)
    return jnp.asarray(grad_eN + grad_NN)
