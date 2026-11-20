"""Rank-one fields for a mesh of one-electron-moved configurations.

The kernels are mesh agnostic: callers supply displaced positions for either
the ECP non-local quadrature or the LRDMC six-hop kinetic lattice.

The ECP non-local derivatives need, at every mesh config (one electron ``e``
displaced to a quadrature point ``r'``), the wavefunction ratio and the FULL
``d ln Psi / d{r, R, c}`` fields.  The streamed slow path re-evaluates every
field from scratch per config, including a full all-electron AO evaluation and
MO projection for every ``Ne * Nv`` configuration.

This module computes the same quantities by reusing the
base ``G^{-1}`` (thresholded SVD-pinv, matching the slow kernels) is reused for
every config via the Sherman-Morrison rank-1 update, all per-config work
reduces to ONE-POINT orbital/AO evaluations at the displaced position plus
small contractions, and the mesh dimension enters only through bounded GEMMs
over the one-point stacks.  No mesh-sized AO-derivative tensor is ever live:
the largest mesh arrays are the one-point AO stacks ``(N_AO, Ne*Nv)``.

Derivation sketch (up-move ``e -> r'``; dn-move symmetric).  With
``G' = G + e_e (b' - b_e)^T``, ``b' = W^T p'``, ``t = Ginv^T b'``, ``ratio = t_e``:

    G'^{-1}          = Ginv - Ginv[:,e] s^T / ratio,   s = t - e_e
    drift, up m != e : gD'_m = gD_m - (gou_m . WG[:,e]) t_m / ratio
    drift, moved slot: gD'_e = (gp' . WG[:,e]) / ratio
    drift, dn l      : gD'_l = gD_l + Ginv[l,e] (godn_l . y) / ratio,  y = lam_p^T p' - U^T t
    P' = G'inv A'    = P + Ginv[:,e] (p' - A^T t)^T / ratio            (lambda O_k)
    Mup' = W G'inv   = WG - WG[:,e] s^T / ratio                        (MO O_k / force)
    G'inv U'         = GQ + Ginv[:,e] y^T / ratio                      (dn coeff)

Jastrow channels are additive one-move deltas: J1 = per-electron (existing
kernels re-used at single points), J2 = closed-form pair kernel (pade/exp),
J3 = symmetric AO quadratic form (``dJ3 = j1.d + (Jd).S + d.Jd/2 - (a'+a_e).Jd/2``
with ``d = a' - a_e``); their field deltas are single GEMMs against the
one-point AO stacks.

Scope: molecular and periodic wavefunctions with SD/AGP/AGPn determinants and
J1/J2/J3/DTN. The rank-one
one-point orbital/AO evals are periodic-image-correct through ``compute_orb_api``
/ ``compute_AOs_grad`` (wrap-into-cell + image folding); J2 swaps the Euclidean
pair distance for the Williamson smooth periodic metric (``kinetic._dist_and_derivs``);
J1 point-grads dispatch to the periodic kernel; J3 AO grads fold images; bspline /
pw orbital leaves carry no nuclear/strain dependence (skipped in force/stress).
DTN enters as a bounded one-move value delta. Its drift, parameter, position,
and strain fields use the corresponding analytic kernels. The strain
axis collapses to a single strained-log-ratio reverse pass (:func:`mesh_stress_wf`).
Unsupported J1/J2 forms keep the streamed full-recompute path.
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

from collections.abc import Callable, Mapping
from functools import partial
from logging import getLogger
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax import typing as jnpt

from jqmc.atomic_orbital import AOs_base_data, compute_AOs, compute_AOs_grad
from jqmc.determinant import AGP_data, AGPn_data, SD_data
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_DTN_data,
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    _compute_Jastrow_one_body_electron_contrib_ratio,
    compute_grads_and_laplacian_Jastrow_one_body,
)
from jqmc.molecular_orbital import MOs_data
from jqmc.wavefunction import Wavefunction_data

from .fast import resolve_geminal_inv

logger = getLogger("jqmc").getChild(__name__)

GeminalData: TypeAlias = AGP_data | AGPn_data | SD_data
MeshState: TypeAlias = dict[str, object]
ArrayPair: TypeAlias = tuple[jax.Array, jax.Array]
ArrayQuad: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
RadialKernelPair: TypeAlias = tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]


# --------------------------------------------------------------------------- #
# capability gate
# --------------------------------------------------------------------------- #
# Every capability rejection below keeps a production-reachable cell on the
# streamed full-recompute path (for DTN
# wavefunctions that includes the dense per-config drift stream -- the
# dominant analytic-force execution cost the fused kernels removed). The
# fallback is deliberate, but it must be LOUD, not silent: warn ONCE per
# (site, reason) per process.
_STREAMED_FALLBACK_WARNED: set[str] = set()


def warn_streamed_fallback(site: str, reason: str, jastrow_data: Jastrow_data) -> None:
    """One-time-per-(site, reason) loud classification of a dense-stream cell."""
    key = f"{site}:{reason}"
    if key in _STREAMED_FALLBACK_WARNED:
        return
    _STREAMED_FALLBACK_WARNED.add(key)
    dtn_extra = ""
    if getattr(jastrow_data, "jastrow_dtn_data", None) is not None:
        dtn_extra = " With DTN active this includes the DENSE per-config drift stream (expect ~10x slower force blocks than the fused route)."
    logger.warning(
        f"{site}: rank-one mesh fast path NOT available ({reason}); using the streamed full-recompute production path.{dtn_extra}"
    )


def _fast_reject_reason(jd: Jastrow_data) -> str | None:
    """Shared capability rejections (None = servable). Kept in one place so the
    S-prime (:mod:`fast_mesh`) and ECP quadrature predicates cannot drift."""
    if jd.jastrow_two_body_data is not None and jd.jastrow_two_body_data.jastrow_2b_type not in ("pade", "exp"):
        return f"non-pade/exp J2 kernel '{jd.jastrow_two_body_data.jastrow_2b_type}'"
    if jd.jastrow_one_body_data is not None and jd.jastrow_one_body_data.jastrow_1b_type not in ("pade", "exp"):
        return f"non-pade/exp J1 kernel '{jd.jastrow_one_body_data.jastrow_1b_type}'"
    j3_data = jd.jastrow_three_body_data
    if j3_data is not None:
        from jqmc.atomic_orbital import is_complex_aos

        j3_orbital = j3_data.orb_data
        j3_aos = j3_orbital.aos_data if hasattr(j3_orbital, "aos_data") else j3_orbital
        if bool(is_complex_aos(j3_aos)):
            return "complex Bloch J3 orbital leaf"
    return None


def fast_supported(hamiltonian_data: Hamiltonian_data) -> bool:
    """True when the rank-1 mesh fast path serves this hamiltonian's ratio + drift
    (e_L/r) + Pulay-force (e_L/R) axes.

    Molecular real and periodic wavefunctions with SD/AGP/AGPn determinants,
    J1/J2/J3, and DTN are served. The
    rank-1 Sherman-Morrison det machinery uses only one-point orbital evals, which
    are periodic-image-correct through ``compute_orb_api`` / ``compute_AOs_grad``
    (they wrap into the cell + fold images); J1 point-grads dispatch to the
    periodic kernel; J2 uses the Williamson smooth periodic distance
    (``kinetic._dist_and_derivs``); J3 AO grads fold images.  DTN enters as a
    one-move value-delta whose drift / force / c / eps fields are the autodiff of
    the exact ``jastrow_factor._delta_J_DTN_one_electron_moved`` (see
    :func:`dtn_mesh`) -- bounded (short-range coupling), matches the streamed
    full-recompute to fp.

    Complex-twist orbitals are served because the determinant one-point
    ratio/drift/force/stress stacks are holomorphic (plain transposes, dtype-agnostic)
    and remain holomorphic. The c-axis collapse has its own capability gate in
    :func:`fast_supported_c` for basis/Jastrow parameter contractions.

    Still streamed: non-(pade/exp) J1/J2 kernels.
    """
    try:
        from jqmc.orbital_tiling import TiledOrbitalSet_data

        if isinstance(hamiltonian_data.wavefunction_data.geminal_data.orb_data_up_spin, TiledOrbitalSet_data):
            # Image -> physical-atom maps depend on each moved electron. Use
            # the bounded analytic mesh stream until rank-one center fields
            # have a block-aware consumer; the orbital value path stays tiled.
            return False
        wf = hamiltonian_data.wavefunction_data
        jd = wf.jastrow_data
        # Complex Bloch J3 rationale: the one-move J3 kernels here Re-project
        # the AO stack BEFORE the coupling contraction (j3_base/j3_mesh),
        # which does not commute with the complex leaf's production
        # convention -- pinned red by the curated Li_pbc_twist_J3cplx c-axis
        # cell.  Real-AO J3 at twist IS served.
        reason = _fast_reject_reason(jd)
        if reason is not None:
            warn_streamed_fallback("one_move_mesh.fast_supported (ECP quadrature)", reason, jd)
            return False
        return True
    except Exception:
        # A real bug in this probe must remain distinguishable from
        # "this hamiltonian genuinely isn't rank-1-mesh-servable" (both
        # silently routed to the streamed slow path). Log loud, keep the
        # same fallback (callers rely on this probe never raising).
        logger.warning("one_move_mesh.fast_supported probe raised; falling back to the streamed slow path.", exc_info=True)
        return False


# --------------------------------------------------------------------------- #
# base determinant fields
# --------------------------------------------------------------------------- #
def det_base(
    gd: GeminalData,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    geminal_inv: jnpt.ArrayLike | None = None,
) -> MeshState:
    """Base geminal fields shared by every mesh config (bounded, computed once).

    ``geminal_inv`` is the running Sherman-Morrison ``G^{-1}`` at
    the base walker position. When provided it is reused verbatim (the whole ECP mesh
    then rides SM rank-1 updates off it -- NO fresh SVD-pinv is lowered on the axis)."""
    M_dn = gd.orb_num_dn
    N_dn = gd.num_electron_dn
    if geminal_inv is None:
        # No running inverse to stay consistent with: assemble G from the fused value table,
        # the same one whose fp64 gradients are contracted with G^{-1} below and that the
        # kinetic reverse pass inverts (determinant._orbital_value_grad_lap_tables).  The
        # value-only ``compute_orb_api`` table is ``ao_eval`` (fp32 in mixed); with it, G^{-1}
        # and the fp64 orbital gradients describe two different geminal matrices (#681).
        from jqmc.determinant import _orbital_value_grad_lap_tables

        orb_up, gou, _, orb_dn, godn, _ = _orbital_value_grad_lap_tables(gd, r_up, r_dn)
    else:
        orb_up = gd.compute_orb_api(gd.orb_data_up_spin, r_up)  # (m_up, N_up)
        orb_dn = gd.compute_orb_api(gd.orb_data_dn_spin, r_dn)  # (m_dn, N_dn)
        gou = jnp.stack(gd.compute_orb_grad_api(gd.orb_data_up_spin, r_up), 0)  # (3, m_up, N_up)
        godn = jnp.stack(gd.compute_orb_grad_api(gd.orb_data_dn_spin, r_dn), 0)  # (3, m_dn, N_dn)
    from jqmc.determinant import SD_data

    if isinstance(gd, SD_data):
        lam_p = lam_unp = W = None
        U = orb_up[:N_dn].T
        G = jnp.hstack([U @ orb_dn, orb_up[N_dn:].T])
    else:
        lam_p, lam_unp = jnp.hsplit(gd._lambda_matrix_jnp, [M_dn])
        W = jnp.hstack([lam_p @ orb_dn, lam_unp])  # (m_up, N_up)
        U = orb_up.T @ lam_p  # (N_up, m_dn)
        G = orb_up.T @ W  # (N_up, N_up)
    Ginv = resolve_geminal_inv(G, geminal_inv)  # fast: running SM inverse; else thresholded SVD-pinv
    if isinstance(gd, SD_data):
        WG = jnp.concatenate([orb_dn @ Ginv[:N_dn], Ginv[N_dn:]], axis=0)
    else:
        WG = W @ Ginv  # (m_up, N_up)
    GQ = Ginv @ U  # (N_up, m_dn)
    gD_up = jnp.einsum("dak,ak->kd", gou, WG)  # (N_up, 3) base det drift
    gD_dn = jnp.einsum("dbl,lb->ld", godn, GQ[:N_dn])  # (N_dn, 3)
    return {
        "M_dn": M_dn, "N_dn": N_dn, "orb_up": orb_up, "orb_dn": orb_dn, "gou": gou, "godn": godn,
        "lam_p": lam_p, "lam_unp": lam_unp, "W": W, "U": U, "G": G, "Ginv": Ginv, "WG": WG, "GQ": GQ,
        "gD_up": gD_up, "gD_dn": gD_dn,
    }  # fmt: skip


def _one_point_orb(
    gd: GeminalData,
    orb_data: AOs_base_data | MOs_data,
    pts: jax.Array,
) -> ArrayPair:
    """Orbital values + grads at a stack of points: value (m, K), grad (3, m, K)."""
    v = gd.compute_orb_api(orb_data, pts)
    g = jnp.stack(gd.compute_orb_grad_api(orb_data, pts), 0)
    return v, g


# --------------------------------------------------------------------------- #
# determinant: per-config ratio + drift field (whole-mesh GEMMs)
# --------------------------------------------------------------------------- #
def det_mesh(
    gd: GeminalData,
    base: Mapping[str, object],
    pts_up: jnpt.ArrayLike,
    pts_dn: jnpt.ArrayLike,
) -> MeshState:
    """Rank-1 determinant mesh quantities for up-moves (each up electron to Nv points,
    electron-major) and dn-moves.  Returns a dict with the per-config det ratios,
    the SM auxiliaries (t-stacks etc.), and the one-point orbital stacks.
    """
    n_up = base["gD_up"].shape[0]
    n_dn = base["N_dn"]
    Nv_u = pts_up.shape[0] // max(n_up, 1) if n_up else 0
    Nv_d = pts_dn.shape[0] // max(n_dn, 1) if n_dn else 0
    W, U, Ginv, lam_p = base["W"], base["U"], base["Ginv"], base["lam_p"]
    from jqmc.determinant import SD_data

    is_sd = isinstance(gd, SD_data)

    # --- up-moves ---
    pU, gU = _one_point_orb(gd, gd.orb_data_up_spin, pts_up)  # (m_up, Ku), (3, m_up, Ku)
    if is_sd:
        bU = jnp.concatenate([base["orb_dn"].T @ pU[:n_dn], pU[n_dn:]], axis=0)
    else:
        bU = W.T @ pU  # (N_up, Ku) geminal rows at r'
    tU = Ginv.T @ bU  # (N_up, Ku)
    eU = jnp.repeat(jnp.arange(n_up), Nv_u)  # (Ku,) moved electron per config
    kU = jnp.arange(eU.shape[0])
    ratio_up = tU[eU, kU]  # (Ku,) det ratios
    yU = (pU[:n_dn] if is_sd else lam_p.T @ pU) - U.T @ tU  # (m_dn, Ku)

    # --- dn-moves ---
    pD, gD = _one_point_orb(gd, gd.orb_data_dn_spin, pts_dn)  # (m_dn, Kd), (3, m_dn, Kd)
    cD = U @ pD  # (N_up, Kd) geminal columns at r''
    tD = Ginv @ cD  # (N_up, Kd)
    jD = jnp.repeat(jnp.arange(n_dn), Nv_d)  # (Kd,)
    kD = jnp.arange(jD.shape[0])
    ratio_dn = tD[jD, kD]  # (Kd,)
    if is_sd:
        zD = jnp.concatenate(
            [pD - base["orb_dn"] @ tD[:n_dn], -tD[n_dn:]],
            axis=0,
        )
    else:
        zD = W @ (-tD) + lam_p @ pD  # (m_up, Kd)  z = lam_p pd' - W t'

    return {
        "pU": pU, "gU": gU, "tU": tU, "eU": eU, "ratio_up": ratio_up, "yU": yU,
        "pD": pD, "gD": gD, "tD": tD, "jD": jD, "ratio_dn": ratio_dn, "zD": zD,
    }  # fmt: skip


def det_drift_mesh(base: Mapping[str, object], dm: Mapping[str, object]) -> jax.Array:
    """Full det drift field at every config: (Ku+Kd, Ne, 3), configs up-major then dn."""
    gou, godn, Ginv, WG, GQ = base["gou"], base["godn"], base["Ginv"], base["WG"], base["GQ"]
    gD_up, gD_dn, N_dn = base["gD_up"], base["gD_dn"], base["N_dn"]
    n_up = gD_up.shape[0]
    base_field = jnp.concatenate([gD_up, gD_dn], 0)  # (Ne, 3)

    # ---- up-move configs ----
    tU, eU, rU, yU, gU = dm["tU"], dm["eU"], dm["ratio_up"], dm["yU"], dm["gU"]
    kU = jnp.arange(eU.shape[0])
    # spectator up m: -HUP[e, m, :] * t_m / R,  HUP[e', k, d] = gou_k . WG[:, e']
    HUP = jnp.einsum("dak,ae->ekd", gou, WG)  # (N_up, N_up, 3)
    corr_up = -HUP[eU] * (tU.T / rU[:, None])[:, :, None]  # (Ku, N_up, 3)
    # spectator dn l: + Ginv[l, e] (godn_l . y) / R
    gy = jnp.einsum("dbl,bK->Kld", godn, yU)  # (Ku, N_dn, 3)
    corr_dn = Ginv[:N_dn, :][:, eU].T[:, :, None] * gy / rU[:, None, None]  # (Ku, N_dn, 3)
    O_up_cfg = base_field[None] + jnp.concatenate([corr_up, corr_dn], 1)  # (Ku, Ne, 3)
    # moved slot overwrite: (gp' . WG[:,e]) / R
    moved_up = jnp.einsum("daK,aK->Kd", gU, WG[:, eU]) / rU[:, None]
    O_up_cfg = O_up_cfg.at[kU, eU].set(moved_up)

    # ---- dn-move configs ----
    tD, jDi, rD, zD, gDp = dm["tD"], dm["jD"], dm["ratio_dn"], dm["zD"], dm["gD"]
    kD = jnp.arange(jDi.shape[0])
    # spectator dn m: -HDN[j, m, :] * t'_m / R,  HDN[j', l, d] = godn_l . GQ[j', :]
    HDN = jnp.einsum("dbl,jb->jld", godn, GQ)  # (N_up, N_dn, 3)
    corr_dn2 = -HDN[jDi] * (tD[:N_dn].T / rD[:, None])[:, :, None]  # (Kd, N_dn, 3)
    # spectator up m: + Ginv[j, m] (gou_m . z) / R
    gz = jnp.einsum("dam,aK->Kmd", gou, zD)  # (Kd, N_up, 3)
    corr_up2 = Ginv[jDi, :][:, :, None] * gz / rD[:, None, None]  # (Kd, N_up, 3)
    O_dn_cfg = base_field[None] + jnp.concatenate([corr_up2, corr_dn2], 1)  # (Kd, Ne, 3)
    moved_dn = jnp.einsum("dbK,Kb->Kd", gDp, GQ[jDi]) / rD[:, None]
    O_dn_cfg = O_dn_cfg.at[kD, n_up + jDi].set(moved_dn)

    return jnp.concatenate([O_up_cfg, O_dn_cfg], 0)  # (K, Ne, 3)


# --------------------------------------------------------------------------- #
# Jastrow one-move deltas (value + drift-field), molecular
# --------------------------------------------------------------------------- #
def _j2_kernel(j2d: Jastrow_two_body_data) -> RadialKernelPair:
    """(u(r), u'(r)) closed forms for the production J2 (pade/exp), molecular."""
    b = jnp.asarray(j2d.jastrow_2b_param, jnp.float64)
    if j2d.jastrow_2b_type == "pade":
        return (lambda r: r / (2.0 * (1.0 + b * r)), lambda r: 1.0 / (2.0 * (1.0 + b * r) ** 2))
    # exp: u = (1 - e^{-b r}) / (2 b) = -expm1(-b r) / (2 b) (no cancellation at small r), u' = e^{-b r} / 2
    return (lambda r: -jnp.expm1(-b * r) / (2.0 * b), lambda r: jnp.exp(-b * r) / 2.0)


def _j2_mesh_pairs(
    j2d: Jastrow_two_body_data,
    r_all: jnpt.ArrayLike,
    pts: jnpt.ArrayLike,
    e_idx: jnpt.ArrayLike,
) -> ArrayQuad:
    """J2 one-move pair terms per (config, electron): value delta (K, Ne), the new/old
    pair drift ``g_new``/``g_old`` (K, Ne, 3) and the spectator mask (K, Ne).

    Distance metric = ``kinetic._dist_and_derivs`` (Euclidean molecular, Williamson
    smooth periodic under PBC); the pade/exp kernels are metric-independent, only the
    distance ``d`` and its gradient direction ``grad_d = d(dist)/d(disp)`` change.
    """
    from .kinetic import _dist_and_derivs

    sd = j2d.structure_data
    u, du = _j2_kernel(j2d)
    Ne = r_all.shape[0]
    r_e = r_all[e_idx]  # (K, 3)
    d_new = r_all[None, :, :] - pts[:, None, :]  # (K, Ne, 3) disp = r_m - r'
    d_old = r_all[None, :, :] - r_e[:, None, :]  # (K, Ne, 3) disp = r_m - r_e
    mask = jnp.arange(Ne)[None, :] != e_idx[:, None]  # exclude the moved electron itself
    # The moved electron's own slot has a ZERO ``d_old`` displacement (r_e - r_e == 0). ``_dist_and_derivs``
    # is value-safe there (safe-where d==0 -> 0), but its reverse-mode VJP through ``norm``/Williamson is
    # 0*(0/0) == NaN.  ``j2_mesh`` is not differentiated by any production kernel (the ECP eps stress
    # is the closed-form ``mesh_stress_wf``), but the double-where is kept as a cheap, forward-inert
    # safeguard against any future reverse-mode use and keeps the masked slots well-defined.
    # Replace every masked slot's displacement with a nonzero dummy BEFORE the distance eval (double-where);
    # dJ/corr/moved all mask these slots out, so the forward is bit-identical.
    safe = mask[..., None]
    d_new = jnp.where(safe, d_new, 1.0)
    d_old = jnp.where(safe, d_old, 1.0)
    dist_new, gdir_new, _, _ = _dist_and_derivs(sd, d_new)  # (K,Ne), (K,Ne,3)
    dist_old, gdir_old, _, _ = _dist_and_derivs(sd, d_old)
    if j2d.jastrow_2b_type == "exp":
        b = jnp.asarray(j2d.jastrow_2b_param, jnp.float64)
        e_new = jnp.exp(-b * dist_new)
        e_old = jnp.exp(-b * dist_old)
        # Evaluate the difference directly, sharing both exponentials with the
        # drift.  For nearby distances, e_old - e_new cancels; use
        # e_old * (1 - exp(-dx)) with dx=b*(dist_new-dist_old) instead.
        # The degree-13 series retains small-separation accuracy without two
        # additional expm1 evaluations. Mask its argument before evaluating
        # the polynomial so the inactive branch remains finite for AD.
        dx = b * (dist_new - dist_old)
        small = jnp.abs(dx) < 0.25
        xs = jnp.where(small, dx, 0.0)
        polynomial = jnp.ones_like(xs)
        for degree in range(13, 1, -1):
            polynomial = 1.0 - xs * polynomial / degree
        delta = jnp.where(small, e_old * xs * polynomial, e_old - e_new) / (2.0 * b)
        du_new, du_old = e_new / 2.0, e_old / 2.0
    else:
        delta = u(dist_new) - u(dist_old)
        du_new, du_old = du(dist_new), du(dist_old)
    # spectator m: grad_m[u(dist(r_m, r'))] - grad_m[u(dist(r_m, r_e))]; d(dist)/d(r_m) = +grad_d
    g_new = du_new[..., None] * gdir_new
    g_old = du_old[..., None] * gdir_old
    return delta, g_new, g_old, mask


def j2_mesh(
    j2d: Jastrow_two_body_data,
    r_all: jnpt.ArrayLike,
    pts: jnpt.ArrayLike,
    e_idx: jnpt.ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """J2 one-move deltas: value delta (K,), spectator drift corr (K, Ne, 3), moved-slot
    drift (K, 3).  ``pts`` (K, 3) displaced positions, ``e_idx`` (K,) the moved electron.
    """
    delta, g_new, g_old, mask = _j2_mesh_pairs(j2d, r_all, pts, e_idx)
    dJ = jnp.sum(jnp.where(mask, delta, 0.0), axis=1)  # (K,)
    corr = jnp.where(mask[..., None], g_new - g_old, 0.0)  # (K, Ne, 3)
    # moved slot: sum_m grad_{r'}[u(dist(r_m, r'))] = -sum_m g_new  (d(dist)/d(r') = -grad_d)
    moved = -jnp.sum(jnp.where(mask[..., None], g_new, 0.0), axis=1)  # (K, 3)
    return dJ, corr, moved


def j1_point_values(j1d: Jastrow_one_body_data, pts: jnpt.ArrayLike) -> jax.Array:
    """One electron's J1 contribution at single-electron positions (K,), in ``jastrow_ratio``.

    Callers only form one-move differences ``J1(r') - J1(r_e)`` from these values, i.e. the
    J1 part of the Jastrow log-ratio, which the precision policy assigns to the fp64
    ``jastrow_ratio`` zone.  The forward ``compute_Jastrow_one_body`` belongs to
    ``jastrow_eval`` (fp32 in mixed); subtracting two of its accumulated values would round
    the log-ratio at fp32.  This is the same per-electron kernel the production
    rank-one ratio (``_compute_ratio_Jastrow_part_rank1_update``) uses.
    """
    return jax.vmap(lambda p: _compute_Jastrow_one_body_electron_contrib_ratio(j1d, p))(jnp.asarray(pts))


def j1_point_grads(j1d: Jastrow_one_body_data, pts: jnpt.ArrayLike) -> jax.Array:
    """J1 drift at single-electron positions (K, 3) via the production kernel."""
    empty = jnp.zeros((0, 3), dtype=jnp.float64)

    def _g(p: jax.Array) -> jax.Array:
        gu, _, _, _ = compute_grads_and_laplacian_Jastrow_one_body(j1d, p[None, :], empty)
        return jnp.asarray(gu)[0]

    return jax.vmap(_g)(pts)


def j3_base(
    j3d: Jastrow_three_body_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
) -> MeshState:
    """Base J3 fields: AO matrix A (n, Ne), grads gA (3, n, Ne), the symmetric block J,
    the J1-like column, S = sum_i a_i, JS, V3 = j1 + J(S - a_m) per electron.
    """
    from .assemble import _j3_mo_to_bare_ao

    j3d = _j3_mo_to_bare_ao(j3d)  # MO-backed J3 -> equivalent bare-AO J3 (bit-identical value)
    Jm = j3d._j_matrix_jnp
    j1v = Jm[:, -1]
    J = Jm[:, :-1]
    # Leaf-agnostic J3 orbital value: ``j3d.orb_data`` is the bare AO leaf (MO-backed folded above),
    # so ``compute_AOs`` (dispatches every AOs_base_data subtype, incl. plane-wave / B-spline /
    # composite) is bit-identical to ``compute_orb_api`` for gto/pgto/composite AND additionally
    # serves standalone plane-wave / B-spline J3 leaves that ``compute_orb_api`` does not whitelist.
    A_up = jnp.real(compute_AOs(j3d.orb_data, r_up))
    A_dn = jnp.real(compute_AOs(j3d.orb_data, r_dn))
    A = jnp.concatenate([A_up, A_dn], axis=1)  # (n, Ne)
    gx, gy, gz = compute_AOs_grad(j3d.orb_data, jnp.concatenate([r_up, r_dn], 0))
    gA = jnp.real(jnp.stack([gx, gy, gz], 0))  # (3, n, Ne)
    S = jnp.sum(A, axis=1)  # (n,)
    JA = J @ A  # (n, Ne)
    JS = J @ S  # (n,)
    V3 = j1v[:, None] + JS[:, None] - JA  # (n, Ne): j1 + J(S - a_m)
    return {"J": J, "j1v": j1v, "A": A, "gA": gA, "S": S, "JA": JA, "JS": JS, "V3": V3, "j3d": j3d}


def j3_mesh(
    j3d: Jastrow_three_body_data,
    jb: Mapping[str, object],
    pts: jnpt.ArrayLike,
    e_idx: jnpt.ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """J3 one-move deltas: value delta (K,), spectator drift corr (K, Ne, 3), moved-slot
    drift (K, 3), plus the one-point AO stacks (for the c-axis / force reuse).
    """
    j3d = jb["j3d"]  # bare-AO form (converted in j3_base)
    J, j1v, A, gA, S = jb["J"], jb["j1v"], jb["A"], jb["gA"], jb["S"]
    Ne = A.shape[1]
    Ap = jnp.real(compute_AOs(j3d.orb_data, pts))  # (n, K); bare-AO leaf-agnostic (PW/bspline/composite)
    gxp, gyp, gzp = compute_AOs_grad(j3d.orb_data, pts)
    gAp = jnp.real(jnp.stack([gxp, gyp, gzp], 0))  # (3, n, K)
    a_e = A[:, e_idx]  # (n, K)
    delta = Ap - a_e  # (n, K)
    Jd = J @ delta  # (n, K) -- THE J3 mesh GEMM
    # value: j1.d + (Jd).S + d.Jd/2 - (a' + a_e).Jd/2
    dJ = j1v @ delta + S @ Jd + 0.5 * jnp.einsum("nK,nK->K", delta, Jd) - 0.5 * jnp.einsum("nK,nK->K", Ap + a_e, Jd)
    # spectator m != e: + (da_m . Jd)
    corr = jnp.einsum("dnm,nK->Kmd", gA, Jd)  # (K, Ne, 3); row e overwritten by caller
    mask = jnp.arange(Ne)[None, :] != e_idx[:, None]
    corr = jnp.where(mask[..., None], corr, 0.0)
    # moved slot: da(r') . (j1 + J(S - a_e)) = da(r') . V3[:, e]
    moved = jnp.einsum("dnK,nK->Kd", gAp, jb["V3"][:, e_idx])  # (K, 3)
    return dJ, corr, moved, Ap, gAp, delta, Jd


# --------------------------------------------------------------------------- #
# DTN one-move mesh deltas (closed form -- NO autodiff)
# --------------------------------------------------------------------------- #
def _config_electron_arrays(
    r_up: jax.Array,
    r_dn: jax.Array,
    pts_up: jax.Array,
    pts_dn: jax.Array,
    eU: jax.Array,
    jDl: jax.Array,
) -> ArrayPair:
    """The FULL per-config electron arrays (RU, RD) with exactly one electron moved to its
    quadrature point, up-move configs first then dn-move configs (matches the mesh ordering).
    ``eU`` (Ku,) up-electron index per up-move config; ``jDl`` (Kd,) dn-electron index per
    dn-move config.  Returns ``RU`` (K, n_up, 3), ``RD`` (K, n_dn, 3), K = Ku + Kd.
    """
    n_up, n_dn = r_up.shape[0], r_dn.shape[0]
    Ku, Kd = pts_up.shape[0], pts_dn.shape[0]
    RU_up = jnp.broadcast_to(r_up[None], (Ku, n_up, 3)).at[jnp.arange(Ku), eU].set(pts_up)
    RD_up = jnp.broadcast_to(r_dn[None], (Ku, n_dn, 3))
    RU_dn = jnp.broadcast_to(r_up[None], (Kd, n_up, 3))
    RD_dn = jnp.broadcast_to(r_dn[None], (Kd, n_dn, 3)).at[jnp.arange(Kd), jDl].set(pts_dn)
    return jnp.concatenate([RU_up, RU_dn], 0), jnp.concatenate([RD_up, RD_dn], 0)


def _dtn_drift_probe(dtn_data: Jastrow_DTN_data, ru: jax.Array, rd: jax.Array) -> jax.Array:
    """Per-config DTN drift kernel: the full closed-form DTN drift at ONE mesh config.
    The per-config parity oracle of the batched :func:`dtn_mesh` kernel."""
    from jqmc.jastrow_factor import compute_grads_and_laplacian_Jastrow_DTN

    gup, gdn, _, _ = compute_grads_and_laplacian_Jastrow_DTN(dtn_data, ru, rd)
    return jnp.concatenate([jnp.asarray(gup, jnp.float64), jnp.asarray(gdn, jnp.float64)], 0)


def _dtn_forceR_probe(dtn_data: Jastrow_DTN_data, ru: jax.Array, rd: jax.Array) -> jax.Array:
    """Per-config DTN Pulay-force kernel ``d J_DTN / dR`` at ONE mesh config."""
    from . import dtn as dtn_mod

    return jnp.asarray(dtn_mod.grad_dtn_wrt_R(dtn_data, ru, rd))


@jax.jit
def _dtn_forceR_mesh(
    dtn_data: Jastrow_DTN_data,
    RU: jax.Array,
    RD: jax.Array,
) -> jax.Array:
    """Per-config DTN Pulay force ``(K, n_atom, 3)`` over the one-move mesh configs.

    The bounded ``_mesh_map`` streams with a ``lax.scan`` whose body closes over a
    per-call lambda.  Called eagerly (``assemble.compute_grad_e_L`` outside ``jax.jit``)
    that scan compiled and loaded a new XLA executable on every call (issue #130);
    this module-level ``jax.jit`` reuses one program for repeated calls with the same
    shapes, and is an inlined nested call under an outer ``jax.jit``. The chunk is
    the measured workspace against the DTN-owned per-walker budget."""
    from jqmc._setting import JASTROW_DTN_ONE_MOVE_MESH_BUDGET_BYTES

    from .mesh_stream import _mesh_map

    return _mesh_map(
        lambda ru, rd: _dtn_forceR_probe(dtn_data, ru, rd),
        (RU, RD),
        budget_bytes=JASTROW_DTN_ONE_MOVE_MESH_BUDGET_BYTES,
    )


def dtn_mesh(
    dtn: Jastrow_DTN_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    pts_up: jnpt.ArrayLike,
    pts_dn: jnpt.ArrayLike,
    eU: jnpt.ArrayLike,
    jDl: jnpt.ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """DTN one-move value delta + full drift field for every mesh config (NO autodiff).

    Value: the exact one-move scalar delta ``d = J_DTN(config) - J_DTN(base)`` -- only
    pairs/triples touching the moved electron (short-range, bounded).

    Drift: the FULL per-config DTN drift ``grad_m J_DTN(config)``.  This equals the
    ``base_drift + grad(one-move delta)`` assembly to fp -- both are the true gradient of the SAME
    J_DTN, since ``base_drift[j] + d(delta)/d r_j = grad_j J_DTN(config)`` (the base contribution
    cancels against the delta's) and the moved slot carries ``grad_{r'} J_DTN(config)`` -- but with
    ZERO autodiff (the delta's derivative is served in closed form by the production DTN gradient,
    matching the Rust port which hand-differentiates the same channels).  Returns ``dJ`` (K,) and
    ``O_dtn`` (K, Ne, 3), up-move configs first.
    """
    # A per-config route would evaluate the FULL dense DTN drift ~768 times
    # serially per force call (the dominant analytic-force execution cost) plus
    # a vmapped dense value-delta sweep. BOTH the value deltas and the drift
    # field come from ONE batched rank-one kernel, unconditionally -- the
    # u/chi/f channels are all exact rank-one closed forms over the dense
    # atom-centric partner set, without a neighbor-list state.
    # ``_dtn_drift_probe`` is the per-config parity oracle.
    from .dtn import dtn_drift_mesh_dense

    n_up = r_up.shape[0]
    pts = jnp.concatenate([pts_up, pts_dn], 0)
    e_idx = jnp.concatenate([jnp.asarray(eU, jnp.int32), jnp.asarray(jDl, jnp.int32) + n_up], 0)
    return dtn_drift_mesh_dense(dtn, r_up, r_dn, pts, e_idx)  # (K,), (K, Ne, 3)


# --------------------------------------------------------------------------- #
# assembled mesh fields (ratio + full drift field), molecular fast path
# --------------------------------------------------------------------------- #
def mesh_ratio_and_drift(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    rprime: jnpt.ArrayLike,
    geminal_inv: jnpt.ArrayLike | None = None,
) -> tuple[jax.Array, jax.Array, MeshState, MeshState]:
    """Ratio (Ne, Nv) and O_mesh (Ne, Nv, Ne, 3) = grad ln Psi at every mesh config
    (det + J1 + J2 + J3), matching the streamed slow path's quantities.  ``rprime``
    is the (Ne, Nv, 3) displaced-position mesh (up electrons first).

    ``geminal_inv`` is the base-position running Sherman-Morrison ``G^{-1}`` reused
    by ``det_base`` (the whole rank-1 mesh rides off it -- no fresh SVD-pinv).
    """
    wf = hamiltonian_data.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    r_up = jnp.asarray(r_up, jnp.float64)
    r_dn = jnp.asarray(r_dn, jnp.float64)
    n_up, n_dn = r_up.shape[0], r_dn.shape[0]
    Ne, Nv = rprime.shape[0], rprime.shape[1]
    r_all = jnp.concatenate([r_up, r_dn], 0)
    pts = rprime.reshape(Ne * Nv, 3)
    pts_up = pts[: n_up * Nv]
    pts_dn = pts[n_up * Nv :]
    e_idx = jnp.repeat(jnp.arange(Ne), Nv)  # (K,) moved electron (global index)

    base = det_base(gd, r_up, r_dn, geminal_inv=geminal_inv)
    dm = det_mesh(gd, base, pts_up, pts_dn)
    det_ratio = jnp.concatenate([dm["ratio_up"], dm["ratio_dn"]], 0)  # (K,)
    O = det_drift_mesh(base, dm)  # (K, Ne, 3)

    dJ = jnp.zeros(Ne * Nv, dtype=jnp.float64)
    kK = jnp.arange(Ne * Nv)
    if jd.jastrow_two_body_data is not None:
        d2, c2, m2 = j2_mesh(jd.jastrow_two_body_data, r_all, pts, e_idx)
        from jqmc.jastrow_factor import compute_grads_and_laplacian_Jastrow_two_body

        g2u, g2d, _, _ = compute_grads_and_laplacian_Jastrow_two_body(jd.jastrow_two_body_data, r_up, r_dn)
        base2 = jnp.concatenate([jnp.asarray(g2u), jnp.asarray(g2d)], 0)
        O = O + base2[None] + c2
        O = O.at[kK, e_idx].add(m2 - base2[e_idx] - c2[kK, e_idx])
        dJ = dJ + d2
    if jd.jastrow_three_body_data is not None:
        jb = j3_base(jd.jastrow_three_body_data, r_up, r_dn)
        d3, c3, m3, _, _, _, _ = j3_mesh(jd.jastrow_three_body_data, jb, pts, e_idx)
        base3 = jnp.einsum("dnm,nm->md", jb["gA"], jb["V3"])  # base J3 drift (Ne, 3)
        O = O + base3[None] + c3
        O = O.at[kK, e_idx].add(m3 - base3[e_idx] - c3[kK, e_idx])
        dJ = dJ + d3
    if jd.jastrow_one_body_data is not None:
        j1d = jd.jastrow_one_body_data
        v_pts = j1_point_values(j1d, pts)
        v_base = j1_point_values(j1d, r_all)
        dJ = dJ + v_pts - v_base[e_idx]
        g1_base = j1_point_grads(j1d, r_all)  # (Ne, 3)
        g1_pts = j1_point_grads(j1d, pts)  # (K, 3)
        O = O + g1_base[None]
        O = O.at[kK, e_idx].add(g1_pts - g1_base[e_idx])
    if jd.jastrow_dtn_data is not None:
        dtn = jd.jastrow_dtn_data
        Ku = n_up * Nv
        eU = e_idx[:Ku]  # up electron index (== global for up rows)
        jDl = e_idx[Ku:] - n_up  # dn electron index
        dJd, O_dtn = dtn_mesh(dtn, r_up, r_dn, pts_up, pts_dn, eU, jDl)
        O = O + O_dtn  # O_dtn is the FULL per-config DTN drift (closed-form kernel, no autodiff)
        dJ = dJ + dJd

    ratio = (det_ratio * jnp.exp(dJ)).reshape(Ne, Nv)
    return ratio, O.reshape(Ne, Nv, Ne, 3), base, dm


# --------------------------------------------------------------------------- #
# weighted mesh fields: sum_k w_k field(config_k) without per-config fields
# --------------------------------------------------------------------------- #
# The ECP non-local (and S-prime) derivatives only consume weighted sums over the
# K = Ne * Nv one-move configurations, e.g. ``sum_k w_k O(config_k)``.  Every rank-one
# correction is linear in the moved row/column, so the weighted sum is formed by
# contracting the weights into the rank-one factors first: per-electron segment sums
# and GEMMs replace the (K, Ne, 3) drift and (K, 3, n_AO) Pulay-force fields of
# :func:`mesh_ratio_and_drift` / :func:`mesh_force` (which stay as the dense reference).
# Only the moved electron's own drift ``Oee`` (K, 3) is kept per configuration.
def _per_electron_sum(x: jax.Array, n: int) -> jax.Array:
    """Sum the leading config axis per moved electron: ``out[i] = sum_v x[i * Nv + v]``.

    The mesh is electron-major with ``Nv`` candidates per electron (``e_idx = repeat(arange(n), Nv)``),
    so the per-electron sum is a reshape and a reduction.
    """
    if n == 0:  # empty spin channel (e.g. a fully polarized atom)
        return jnp.zeros((0,) + x.shape[1:], dtype=x.dtype)
    return jnp.sum(x.reshape((n, -1) + x.shape[1:]), axis=1)


def mesh_ratio_state(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    rprime: jnpt.ArrayLike,
    geminal_inv: jnpt.ArrayLike | None = None,
) -> tuple[jax.Array, jax.Array, MeshState]:
    """Ratio (Ne, Nv), the moved electron's own drift ``Oee`` (Ne, Nv, 3), and the
    state the weighted sums :func:`mesh_drift_weighted` / :func:`mesh_force_weighted`
    contract.  Same quantities as :func:`mesh_ratio_and_drift` (``Oee[e, v] ==
    O[e, v, e]``) without the (Ne, Nv, Ne, 3) field.
    """
    wf = hamiltonian_data.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    r_up = jnp.asarray(r_up, jnp.float64)
    r_dn = jnp.asarray(r_dn, jnp.float64)
    n_up, n_dn = r_up.shape[0], r_dn.shape[0]
    Ne, Nv = rprime.shape[0], rprime.shape[1]
    K = Ne * Nv
    r_all = jnp.concatenate([r_up, r_dn], 0)
    pts = rprime.reshape(K, 3)
    pts_up = pts[: n_up * Nv]
    pts_dn = pts[n_up * Nv :]
    e_idx = jnp.repeat(jnp.arange(Ne), Nv)  # (K,) moved electron (global index)

    base = det_base(gd, r_up, r_dn, geminal_inv=geminal_inv)
    dm = det_mesh(gd, base, pts_up, pts_dn)
    det_ratio = jnp.concatenate([dm["ratio_up"], dm["ratio_dn"]], 0)  # (K,)
    # moved slot of the det drift: (gp' . WG[:,e]) / R (up), (gp'' . GQ[j,:]) / R (dn)
    moved_up = jnp.einsum("daK,aK->Kd", dm["gU"], base["WG"][:, dm["eU"]]) / dm["ratio_up"][:, None]
    moved_dn = jnp.einsum("dbK,Kb->Kd", dm["gD"], base["GQ"][dm["jD"]]) / dm["ratio_dn"][:, None]
    oee_det = jnp.concatenate([moved_up, moved_dn], 0)  # (K, 3)
    det_field = jnp.concatenate([base["gD_up"], base["gD_dn"]], 0)  # (Ne, 3) base det drift

    dJ = jnp.zeros(K, dtype=jnp.float64)
    jas_field = jnp.zeros((Ne, 3), dtype=jnp.float64)  # base Jastrow drift (J1+J2+J3)
    oee_jas = jnp.zeros((K, 3), dtype=jnp.float64)  # moved-slot Jastrow drift at r'
    state: MeshState = {"base": base, "dm": dm, "r_up": r_up, "r_dn": r_dn, "r_all": r_all, "pts": pts, "e_idx": e_idx}
    if jd.jastrow_two_body_data is not None:
        d2, _, m2 = j2_mesh(jd.jastrow_two_body_data, r_all, pts, e_idx)
        from jqmc.jastrow_factor import compute_grads_and_laplacian_Jastrow_two_body

        g2u, g2d, _, _ = compute_grads_and_laplacian_Jastrow_two_body(jd.jastrow_two_body_data, r_up, r_dn)
        jas_field = jas_field + jnp.concatenate([jnp.asarray(g2u), jnp.asarray(g2d)], 0)
        oee_jas = oee_jas + m2
        dJ = dJ + d2
    if jd.jastrow_three_body_data is not None:
        jb = j3_base(jd.jastrow_three_body_data, r_up, r_dn)
        d3, _, m3, _, gAp3, _, Jd = j3_mesh(jd.jastrow_three_body_data, jb, pts, e_idx)
        jas_field = jas_field + jnp.einsum("dnm,nm->md", jb["gA"], jb["V3"])
        oee_jas = oee_jas + m3
        dJ = dJ + d3
        state.update({"jb": jb, "j3_Jd": Jd, "j3_gAp": gAp3})
    if jd.jastrow_one_body_data is not None:
        j1d = jd.jastrow_one_body_data
        dJ = dJ + j1_point_values(j1d, pts) - j1_point_values(j1d, r_all)[e_idx]
        jas_field = jas_field + j1_point_grads(j1d, r_all)
        oee_jas = oee_jas + j1_point_grads(j1d, pts)
    oee = oee_det + oee_jas
    if jd.jastrow_dtn_data is not None:
        Ku = n_up * Nv
        dJd, O_dtn = dtn_mesh(jd.jastrow_dtn_data, r_up, r_dn, pts_up, pts_dn, e_idx[:Ku], e_idx[Ku:] - n_up)
        oee = oee + O_dtn[jnp.arange(K), e_idx]
        dJ = dJ + dJd
        state["dtn_O"] = O_dtn  # (K, Ne, 3) full per-config DTN drift (DTN wavefunctions only)

    # ``dJ`` (K,) is the one-move Jastrow log-ratio factor of ``ratio`` (formed from one-move deltas,
    # not whole Jastrow values); the dltmove ECP correction re-weights the same candidates by
    # ``exp(dJ)`` (issue #720).
    state.update({"det_field": det_field, "jas_field": jas_field, "oee_det": oee_det, "oee_nodtn": oee_det + oee_jas, "dJ": dJ})
    ratio = (det_ratio * jnp.exp(dJ)).reshape(Ne, Nv)
    return ratio, oee.reshape(Ne, Nv, 3), state


def _det_drift_weighted(base: Mapping[str, object], dm: Mapping[str, object], w: jax.Array) -> jax.Array:
    """``sum_k w_k [O_det(config_k) - base]`` over the spectator electrons (moved slot
    excluded), (Ne, 3): the spectator terms of :func:`det_drift_mesh` with ``w_k / R_k``
    folded into the rank-one factors.
    """
    gou, godn, Ginv, WG, GQ, N_dn = base["gou"], base["godn"], base["Ginv"], base["WG"], base["GQ"], base["N_dn"]
    n_up = base["gD_up"].shape[0]
    tU, eU, rU, yU = dm["tU"], dm["eU"], dm["ratio_up"], dm["yU"]
    tD, jD, rD, zD = dm["tD"], dm["jD"], dm["ratio_dn"], dm["zD"]
    Ku = eU.shape[0]
    a = w[:Ku] / rU  # (Ku,)
    b = w[Ku:] / rD  # (Kd,)
    # up-moves, spectator up m: -HUP[e, m] t_m / R, HUP[e, m, d] = gou_m . WG[:, e]
    HUP = jnp.einsum("dak,ae->ekd", gou, WG)  # (N_up, N_up, 3)
    A_up = _per_electron_sum((tU * a[None, :]).T, n_up)  # (N_up, N_up): [e, m]
    A_up = jnp.where(jnp.eye(n_up, dtype=bool), 0.0, A_up)
    spec_up = -jnp.einsum("emd,em->md", HUP, A_up)
    # up-moves, spectator dn l: + Ginv[l, e] (godn_l . y) / R
    Y = yU @ (a[:, None] * Ginv[:N_dn, eU].T)  # (m_dn, N_dn)
    spec_dn = jnp.einsum("dbl,bl->ld", godn, Y)
    # dn-moves, spectator dn m: -HDN[j, m] t'_m / R, HDN[j, m, d] = godn_m . GQ[j, :]
    HDN = jnp.einsum("dbl,jb->jld", godn, GQ[:N_dn])  # (N_dn, N_dn, 3)
    A_dn = _per_electron_sum((tD[:N_dn] * b[None, :]).T, N_dn)  # (N_dn, N_dn): [j, m]
    A_dn = jnp.where(jnp.eye(N_dn, dtype=bool), 0.0, A_dn)
    spec_dn = spec_dn - jnp.einsum("jmd,jm->md", HDN, A_dn)
    # dn-moves, spectator up m: + Ginv[j, m] (gou_m . z) / R
    Z = zD @ (b[:, None] * Ginv[jD, :])  # (m_up, N_up)
    spec_up = spec_up + jnp.einsum("dam,am->md", gou, Z)
    return jnp.concatenate([spec_up, spec_dn], 0)


def mesh_drift_weighted(
    hamiltonian_data: Hamiltonian_data,
    state: Mapping[str, object],
    w: jnpt.ArrayLike,
    *,
    det_only: bool = False,
) -> jax.Array:
    """``sum_k w_k O(config_k)`` (Ne, 3) for weights ``w`` (Ne, Nv) or (K,): the
    weighted sum of the :func:`mesh_ratio_and_drift` field ``O`` (``det_only``: of the
    determinant field :func:`det_drift_mesh`).  ``state`` from :func:`mesh_ratio_state`.
    """
    jd = hamiltonian_data.wavefunction_data.jastrow_data
    base, dm, e_idx, r_all, pts = state["base"], state["dm"], state["e_idx"], state["r_all"], state["pts"]
    Ne = r_all.shape[0]
    w = jnp.asarray(w).reshape(-1)
    wsum = jnp.sum(w)
    spectators = _det_drift_weighted(base, dm, w)
    if det_only:
        field, oee = state["det_field"], state["oee_det"]
    else:
        field, oee = state["det_field"] + state["jas_field"], state["oee_nodtn"]
        if jd.jastrow_two_body_data is not None:
            _, g_new, g_old, mask = _j2_mesh_pairs(jd.jastrow_two_body_data, r_all, pts, e_idx)
            spectators = spectators + jnp.einsum("k,kmd->md", w, jnp.where(mask[..., None], g_new - g_old, 0.0))
        if jd.jastrow_three_body_data is not None:
            gA, Jd = state["jb"]["gA"], state["j3_Jd"]
            # spectator m != e_k: da_m . Jd[:, k]  (all m, minus the moved slot)
            spectators = spectators + jnp.einsum("dnm,n->md", gA, Jd @ w)
            spectators = spectators - jnp.einsum("dnm,mn->md", gA, _per_electron_sum((Jd * w[None, :]).T, Ne))
    # base field on every config, then the moved slot replaced by its own drift at r'
    out = wsum * field + spectators + _per_electron_sum(w[:, None] * (oee - field[e_idx]), Ne)
    if not det_only and jd.jastrow_dtn_data is not None:
        out = out + jnp.einsum("k,kmd->md", w, state["dtn_O"])
    return out


# --------------------------------------------------------------------------- #
# eps (strain) axis: collapsed periodic Pulay-stress delta (closed form, no autodiff)
# --------------------------------------------------------------------------- #
def _sm_geminal_inv_aux(
    base: Mapping[str, object],
    dm: Mapping[str, object],
) -> ArrayQuad:
    """Compact rank-1 auxiliaries for the per-config SM geminal-inverse mesh (avoids ever
    materialising the ``(K, N_up, N_up)`` tensor, 7.21 GiB/walker at N_up = 432 -> 923 GiB for 128 walkers;
    A = N_up^2 = 186624 there).  Returns ``(Ginv, U, V, r)``: the shared base inverse ``Ginv``
    ``(N_up, N_up)`` plus per-config rank-1 vectors ``U, V`` ``(K, N_up)`` and pivots ``r`` ``(K,)``
    such that each config's exact rank-1 SM inverse is ``G'^{-1}_k = Ginv - outer(U[k], V[k]) / r[k]``.
    That is a factor ``~N_up/2`` smaller than the full tensor (2*K*N_up + K vs K*N_up^2).

    Up-move config (electron ``e`` -> r'): ROW-e rank-1, ``U = Ginv[:,e]``, ``V = s = t - e_e``.
    Dn-move config (electron ``j`` -> r''): COLUMN-j rank-1, ``U = s' = t' - e_j``, ``V = Ginv[j,:]``.
    Up-move configs first (matching :func:`_config_electron_arrays` / the mesh ordering).  The
    reconstruction ``Ginv - outer(U,V) / r`` reproduces :func:`_sm_geminal_inv_mesh` BIT-IDENTICALLY
    (same multiply operands and division order), so a consumer that reconstructs one config at a time
    inside a bounded map gets exactly the full-tensor result at ``O(chunk * N_up^2)`` live memory.
    """
    Ginv = base["Ginv"]  # (N_up, N_up)
    tU, eU, rU = dm["tU"], dm["eU"], dm["ratio_up"]
    tD, jDi, rD = dm["tD"], dm["jD"], dm["ratio_dn"]
    Ku, Kd = eU.shape[0], jDi.shape[0]
    sU = tU.at[eU, jnp.arange(Ku)].add(-1.0)  # (N_up, Ku)   s = t - e_e
    sD = tD.at[jDi, jnp.arange(Kd)].add(-1.0)  # (N_up, Kd)  s' = t' - e_j
    U = jnp.concatenate([Ginv[:, eU].T, sD.T], 0)  # (K, N_up): up-move Ginv[:,e]; dn-move s'
    V = jnp.concatenate([sU.T, Ginv[jDi, :]], 0)  # (K, N_up): up-move s;          dn-move Ginv[j,:]
    r = jnp.concatenate([rU, rD], 0)  # (K,)
    return Ginv, U, V, r


def _sm_geminal_inv_one(
    Ginv: jax.Array,
    u: jax.Array,
    v: jax.Array,
    r: jax.Array,
) -> jax.Array:
    """Reconstruct ONE config's exact rank-1 SM geminal inverse from the compact auxiliaries
    (:func:`_sm_geminal_inv_aux`): ``Ginv - outer(u, v) / r`` ``(N_up, N_up)``.  Bit-identical to the
    matching slice of :func:`_sm_geminal_inv_mesh`; used inside bounded mesh maps so the full
    ``(K, N_up, N_up)`` tensor is never materialised -- only ``(chunk, N_up, N_up)`` is live."""
    return Ginv - jnp.outer(u, v) / r


def _sm_geminal_inv_mesh(
    base: Mapping[str, object],
    dm: Mapping[str, object],
) -> jax.Array:
    """Per-config geminal inverse ``(G'_g)^{-1}`` for every mesh config via the exact rank-1
    Sherman-Morrison update off the base ``G^{-1}`` (no fresh SVD-pinv).  Up-move config (electron
    ``e`` -> r'): a ROW-e rank-1 update, ``(G')^{-1} = Ginv - Ginv[:,e] s^T / ratio``, ``s = t - e_e``;
    dn-move config (electron ``j`` -> r''): a COLUMN-j rank-1 update,
    ``(G')^{-1} = Ginv - s' Ginv[j,:] / ratio``, ``s' = t' - e_j``.  Returns ``(K, N_up, N_up)``,
    up-move configs first (matching :func:`_config_electron_arrays` / the mesh ordering).

    Built off the compact :func:`_sm_geminal_inv_aux` factors: ``Ginv - outer(U, V) / r`` broadcast
    over the K axis.  This materialises the FULL ``(K, N_up, N_up)`` tensor (7.21 GiB/walker at
    N_up = 432) and serves ONLY the streamed external consumers (:func:`sm_geminal_inv_configs`) whose
    call sites hand the full tensor to their own ``_mesh_map``.  The in-scope fast-path consumer
    (:func:`mesh_stress_wf`) instead reconstructs one config at a time via :func:`_sm_geminal_inv_one`
    inside its bounded map, so it never allocates the full tensor.  The ``outer(U,V)/r`` reassociation
    is bit-identical to a per-block einsum (same operands, same divide order).
    """
    Ginv, U, V, r = _sm_geminal_inv_aux(base, dm)
    return Ginv[None] - (U[:, :, None] * V[:, None, :]) / r[:, None, None]


def ratio_table_geminal_inv(
    gd: GeminalData,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    geminal_inv: jnpt.ArrayLike | None = None,
) -> jax.Array:
    """Base ``G^{-1}`` of the determinant-ratio (value-only) table for the streamed ECP mesh.

    A supplied running inverse is returned unchanged.  Otherwise this is the
    thresholded pseudoinverse of ``G`` assembled from ``compute_orb_api``,
    the table of the streamed mesh ratios ``det(G'_g) / det(G)`` and of every
    consumer that receives a per-config inverse from
    :func:`sm_geminal_inv_configs`.  Those consumers select the value-only
    table because an inverse is supplied.

    The no-inverse :func:`det_base` builds ``G`` from the fused table instead.
    An inverse from that table, updated with a value-only moved row,
    is not the inverse of the ``G'_g`` whose determinant forms the ratio.  In
    mixed precision the two tables differ by fp32 rounding.  Near a node of
    ``G'_g`` the drift ``~ 1/ratio`` amplifies the mismatch, and the weighted
    product ``ratio * O(G'_g)`` no longer cancels its ``1/ratio``.
    """
    if geminal_inv is not None:
        return jnp.asarray(geminal_inv)
    from jqmc.determinant import compute_geminal_all_elements

    from .fast import geminal_pinv

    return geminal_pinv(compute_geminal_all_elements(gd, jnp.asarray(r_up, jnp.float64), jnp.asarray(r_dn, jnp.float64)))


def sm_geminal_inv_configs(
    gd: GeminalData,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    RU: jnpt.ArrayLike,
    RD: jnpt.ArrayLike,
    geminal_inv: jnpt.ArrayLike | None = None,
) -> jax.Array:
    """Per-config displaced geminal inverse ``(G'_g)^{-1}`` (K, N_up, N_up) via the EXACT rank-1
    Sherman-Morrison update off the base ``G^{-1}`` -- NO fresh SVD-pinv (``lapack_[dz]gesdd``) per
    config.  This is the surgical fix that lets the STREAMED (complex-twist / non-(pade,exp)-Jastrow)
    ECP-mesh ``e_L/{c,r,eps}`` paths serve every mesh-displaced config's inverse from the threaded
    base running inverse instead of re-pinning: the ECP non-local move displaces exactly ONE electron,
    a rank-1 row (up-move) / column (dn-move) update of the pairing matrix ``G``, so its inverse is the
    Sherman-Morrison update of the base inverse (pure GEMM, dtype-agnostic -- complex128 twist OK).

    ``RU`` (K, n_up, 3) / ``RD`` (K, n_dn, 3) are the FULL per-config electron arrays with exactly one
    electron displaced, ordered up-move configs first then dn-move (electron-major, Nv minor) -- the
    ordering of both ``compute_ecp_non_local_parts_nearest_neighbors`` (NN=1) and
    ``_config_electron_arrays`` / the ``_ecp_nl_common`` mesh.  The single moved position per config is
    extracted, the base geminal fields are built ONCE (reusing the threaded ``geminal_inv`` -- no base
    pinv either when threaded), and the rank-1 mesh update is applied.  The resulting inverses match the
    per-config fresh ``geminal_pinv(G'_g)`` to roundoff (exact rank-1 update == the fresh inverse).

    Every consumer receives these inverses as a supplied inverse and therefore
    reads the value-only orbital table.  Without a base inverse, the base is
    the value-only :func:`ratio_table_geminal_inv`, so each ``(G'_g)^{-1}``
    inverts the same ``G'_g`` as its consumer and its mesh ratio.  The caller
    must difference the mesh fields against base fields computed with the same
    inverse.
    """
    r_up = jnp.asarray(r_up, jnp.float64)
    r_dn = jnp.asarray(r_dn, jnp.float64)
    n_up, n_dn = r_up.shape[0], r_dn.shape[0]
    Ne = n_up + n_dn
    K = RU.shape[0]
    Nv = K // Ne  # NN=1 mesh: K = Ne * Nv
    Ku = n_up * Nv
    Kd = K - Ku
    eU = jnp.repeat(jnp.arange(n_up), Nv)  # (Ku,) moved up-electron per up-move config
    jDl = jnp.repeat(jnp.arange(n_dn), Nv)  # (Kd,) moved dn-electron per dn-move config
    pts_up = RU[jnp.arange(Ku), eU] if Ku else jnp.zeros((0, 3), jnp.float64)  # (Ku,3) displaced up positions
    pts_dn = RD[Ku + jnp.arange(Kd), jDl] if Kd else jnp.zeros((0, 3), jnp.float64)  # (Kd,3) displaced dn positions
    geminal_inv = ratio_table_geminal_inv(gd, r_up, r_dn, geminal_inv)
    base = det_base(gd, r_up, r_dn, geminal_inv=geminal_inv)  # value-only table, reuses the inverse
    dm = det_mesh(gd, base, pts_up, pts_dn)  # one-point orbital evals + SM auxiliaries (no pinv)
    return _sm_geminal_inv_mesh(base, dm)  # (K, N_up, N_up) exact per-config rank-1 inverse


def mesh_stress_wf(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    rprime: jnpt.ArrayLike,
    W: jnpt.ArrayLike,
    base: Mapping[str, object],
    dm: Mapping[str, object],
    geminal_inv: jnpt.ArrayLike | None = None,
    *,
    include_dtn: bool = True,
) -> jax.Array:
    """Collapsed periodic wf-ratio Pulay-stress ``sum_{e,g} W[e,g] (stress(config)-stress(base))`` (3, 3).

    The ECP eps assembly weights the per-config Pulay-stress delta by ``W = (sum_t Vgrid_nr) * ratio``.
    Each ``stress(config)`` is the CLOSED-FORM periodic Pulay stress ``pbc_stress.stress_lnPsi_pbc``
    (det ``grad_lndet_wrt_strain`` + Jastrow ``grad_lnJastrow_wrt_strain`` + DTN ``grad_dtn_wrt_strain``:
    Williamson lattice-length virial for J1/J2, min-image for DTN, AO virial+image-T for the det/J3 --
    the exact periodic strain conventions, no molecular ``drift.r + force.R`` shortcut).  It is streamed
    at each mesh config (one electron displaced) with the per-config geminal inverse supplied by the
    exact rank-1 Sherman-Morrison update (:func:`_sm_geminal_inv_mesh`) off the base ``G^{-1}`` -- so NO
    fresh SVD-pinv is lowered on ANY config (``resolve_geminal_inv`` reuses the provided inverse), and
    the whole eps axis lowers with ZERO ``lapack_gesdd``.  It is mathematically identical to the
    ``jax.grad`` of the strained log-ratio, which would autodiff through a fresh strained-geminal pinv
    (an eps-axis SVD): both equal ``sum_g W_g (stress(config_g) - stress(base))``.  Real Gamma
    (``fast_supported`` gates out complex twist); the caller Re's the assembled product.
    """
    from . import pbc_stress
    from .mesh_stream import _mesh_map

    r_up = jnp.asarray(r_up, jnp.float64)
    r_dn = jnp.asarray(r_dn, jnp.float64)
    rprime = jnp.asarray(rprime, jnp.float64)
    W = jnp.asarray(W)
    ham = hamiltonian_data
    n_up = r_up.shape[0]
    Ne, Nv = rprime.shape[0], rprime.shape[1]
    Ku = n_up * Nv
    pts = rprime.reshape(Ne * Nv, 3)
    pts_up, pts_dn = pts[:Ku], pts[Ku:]
    e_idx = jnp.repeat(jnp.arange(Ne), Nv)
    eU = e_idx[:Ku]
    jDl = e_idx[Ku:] - n_up

    RU, RD = _config_electron_arrays(r_up, r_dn, pts_up, pts_dn, eU, jDl)  # (K, n_up/n_dn, 3)
    # Stream the compact rank-1 SM factors (U, V (K, N_up); r (K,)) instead of the full
    # (K, N_up, N_up) inverse tensor (7.21 GiB/walker at N_up = 432 -> 923 GiB for 128 walkers, with
    # A = N_up^2 = 186624).  The per-config inverse ``Ginv - outer(U,V)/r`` is reconstructed ONE
    # config at a time INSIDE the bounded map (:func:`_sm_geminal_inv_one`), so only
    # ``(chunk, N_up, N_up)`` is ever live -- bit-identical to the materialised-tensor path.
    Ginv, U, V, r = _sm_geminal_inv_aux(base, dm)

    def _S_cfg(
        ru: jax.Array,
        rd: jax.Array,
        u: jax.Array,
        v: jax.Array,
        rr: jax.Array,
    ) -> jax.Array:
        gi = _sm_geminal_inv_one(Ginv, u, v, rr)  # (N_up, N_up) exact per-config SM inverse (no SVD)
        return pbc_stress.stress_lnPsi_pbc(
            ham,
            ru,
            rd,
            take_real=False,
            geminal_inv=gi,
            include_dtn=include_dtn,
        )

    S_mesh = _mesh_map(
        _S_cfg,
        (RU, RD, U, V, r),
    ).reshape(Ne, Nv, 3, 3)
    S_base = pbc_stress.stress_lnPsi_pbc(
        ham,
        r_up,
        r_dn,
        take_real=False,
        geminal_inv=geminal_inv,
        include_dtn=include_dtn,
    )
    return jnp.einsum("eg,egab->ab", W, S_mesh - S_base[None, None])


def _real_linear_field(field: Callable[[jax.Array], jax.Array], coef: jax.Array) -> jax.Array:
    """Apply a field that is linear over the reals (it takes ``Re`` of its result) to a possibly
    complex coefficient: ``field(Re c) + i field(Im c)`` keeps complex weights holomorphic."""
    if jnp.iscomplexobj(coef):
        return field(jnp.real(coef)) + 1j * field(jnp.imag(coef))
    return field(coef)


def mesh_stress_weighted(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    rprime: jnpt.ArrayLike,
    W: jnpt.ArrayLike,
    base: Mapping[str, object],
    dm: Mapping[str, object],
    *,
    include_dtn: bool = True,
) -> jax.Array:
    """``sum_{e,g} W[e,g] (stress(config)-stress(base))`` (3, 3), the same value as
    :func:`mesh_stress_wf`, with the weights contracted before any field is evaluated.

    Every piece of the periodic Pulay stress ``pbc_stress.stress_lnPsi_pbc`` is a sum of
    independent per-electron-column terms, linear in a per-column coefficient:

    * det: ``det_strain_field(orb, bar_orb, r)`` with the orbital cotangent ``bar_orb``
      (``WG`` up, ``GQ[:N_dn].T`` dn).  A one-electron move changes ``bar_orb`` by a rank-one term
      (the Sherman--Morrison factors of :func:`det_mesh`, as in :func:`mesh_force_weighted`) and
      moves one column to ``r'``, so the weighted sum is one field evaluation per spin at the
      base positions (summed cotangent change) plus one at the mesh points (moved columns);
    * J1 / J2: point and pair terms of the moved electron;
    * J3: the coefficient ``gA`` is linear in the basis values, so the same split applies;
    * DTN: :func:`jqmc.derivatives.dtn.dtn_geometry_mesh_weighted`.

    No per-configuration all-electron evaluation and no per-configuration inverse.
    """
    from . import pbc_stress
    from .ao_deriv import j3_basis_derivs
    from .assemble import _j3_mo_to_bare_ao

    wf = hamiltonian_data.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    r_up = jnp.asarray(r_up, jnp.float64)
    r_dn = jnp.asarray(r_dn, jnp.float64)
    rprime = jnp.asarray(rprime, jnp.float64)
    n_up, n_dn = r_up.shape[0], r_dn.shape[0]
    Ne, Nv = rprime.shape[0], rprime.shape[1]
    K = Ne * Nv
    Ku = n_up * Nv
    pts = rprime.reshape(K, 3)
    pts_up, pts_dn = pts[:Ku], pts[Ku:]
    e_idx = jnp.repeat(jnp.arange(Ne), Nv)
    r_all = jnp.concatenate([r_up, r_dn], 0)
    w = jnp.asarray(W).reshape(K)
    wU, wD = w[:Ku], w[Ku:]
    w_e = _per_electron_sum(w, Ne)  # (Ne,) summed weight of each moved electron

    # ---- det: summed rank-one cotangent change at the base positions + moved columns at r' ----
    Ginv, WG, GQ, N_dn = base["Ginv"], base["WG"], base["GQ"], base["N_dn"]
    tU, eU, rU, yU = dm["tU"], dm["eU"], dm["ratio_up"], dm["yU"]
    tD, jD, rD, zD = dm["tD"], dm["jD"], dm["ratio_dn"], dm["zD"]
    Kd = jD.shape[0]
    a = wU / rU  # (Ku,)
    b = wD / rD  # (Kd,)
    sU = tU.at[eU, jnp.arange(Ku)].add(-1.0)  # s = t - e_e   (N_up, Ku)
    sD = tD.at[jD, jnp.arange(Kd)].add(-1.0)  # s' = t' - e_j (N_up, Kd)
    alpha = WG[:, eU] * a[None, :]  # (m_up, Ku): moved up column's cotangent, weighted
    delta = GQ[jD, :].T * b[None, :]  # (m_dn, Kd): moved dn column's cotangent, weighted
    # up-moves: bar_up -> bar_up - bar_up[:, e] s^T / R (moved column -> bar_up[:, e] / R at r');
    # dn-moves: bar_up -> bar_up + z Ginv[j, :] / R.
    bar_up = -alpha @ sU.T - _per_electron_sum(alpha.T, n_up).T + (zD * b[None, :]) @ Ginv[jD, :]
    # up-moves: bar_dn -> bar_dn + y Ginv[:N_dn, e]^T / R; dn-moves: as the up side.
    bar_dn = (yU * a[None, :]) @ Ginv[:N_dn, eU].T - delta @ sD[:N_dn].T - _per_electron_sum(delta.T, n_dn).T
    pos = gd.orb_data_up_spin.structure_data._positions_cart_jnp
    up, dn = gd.orb_data_up_spin, gd.orb_data_dn_spin
    out = (
        pbc_stress.det_strain_field(up, bar_up, r_up, pos, take_real=False)
        + pbc_stress.det_strain_field(up, alpha, pts_up, pos, take_real=False)
        + pbc_stress.det_strain_field(dn, bar_dn, r_dn, pos, take_real=False)
        + pbc_stress.det_strain_field(dn, delta, pts_dn, pos, take_real=False)
    )

    # ---- J1: the moved electron's point term ----
    if jd.jastrow_one_body_data is not None:
        j1d = jd.jastrow_one_body_data
        out = out + jnp.einsum("k,kab->ab", w, pbc_stress.j1_point_strain(j1d, pts))
        out = out - jnp.einsum("e,eab->ab", w_e, pbc_stress.j1_point_strain(j1d, r_all))

    # ---- J2: the moved electron's pairs (the self pair excluded) ----
    if jd.jastrow_two_body_data is not None:
        j2d = jd.jastrow_two_body_data
        partner = jnp.arange(Ne)[None, :] != e_idx[:, None]  # (K, Ne)
        safe = jnp.asarray([1.0, 0.0, 0.0])  # placeholder displacement for the excluded self pair
        d_new = jnp.where(partner[..., None], pts[:, None, :] - r_all[None, :, :], safe)
        d_old = jnp.where(partner[..., None], r_all[e_idx][:, None, :] - r_all[None, :, :], safe)
        s2 = pbc_stress.j2_pair_strain(j2d, d_new) - pbc_stress.j2_pair_strain(j2d, d_old)  # (K, Ne, 3, 3)
        out = out + jnp.einsum("k,km,kmab->ab", w, partner.astype(jnp.float64), s2)

    # ---- J3: summed coefficient change at the base positions + moved columns at r' ----
    if jd.jastrow_three_body_data is not None:
        j3 = _j3_mo_to_bare_ao(jd.jastrow_three_body_data)
        aos3 = j3.orb_data
        A_up = j3_basis_derivs(aos3, r_up)[0]  # (n, N_up)
        A_dn = j3_basis_derivs(aos3, r_dn)[0]
        Ap_up = j3_basis_derivs(aos3, pts_up)[0]  # (n, Ku)
        Ap_dn = j3_basis_derivs(aos3, pts_dn)[0]  # (n, Kd)
        gA_up, gA_dn = pbc_stress.j3_strain_coefficients(j3, A_up, A_dn)
        wE_up, wE_dn = w_e[:n_up], w_e[n_up:]
        # weighted basis-value change per moved electron: sum_k w_k (A(r'_k) - A(r_e))
        D_up = _per_electron_sum((Ap_up * wU[None, :]).T, n_up).T - A_up * wE_up[None, :]
        D_dn = _per_electron_sum((Ap_dn * wD[None, :]).T, n_dn).T - A_dn * wE_dn[None, :]
        Jm = j3._j_matrix_jnp
        M = Jm[:, :-1]
        K_up = jnp.tril(jnp.ones((n_up, n_up)), -1)
        K_dn = jnp.tril(jnp.ones((n_dn, n_dn)), -1)
        # gA is linear in A (strictly triangular same-spin coupling, so the moved column's own
        # coefficient does not change); the cross-spin term sees the summed change.
        dG_up = M @ D_up @ K_up + M.T @ D_up @ K_up.T + (M @ D_dn.sum(1))[:, None]
        dG_dn = M @ D_dn @ K_dn + M.T @ D_dn @ K_dn.T + (M.T @ D_up.sum(1))[:, None]
        out = out + _real_linear_field(lambda c: pbc_stress.j3_strain_field(aos3, c, r_up), dG_up - gA_up * wE_up[None, :])
        out = out + _real_linear_field(lambda c: pbc_stress.j3_strain_field(aos3, c, pts_up), gA_up[:, eU] * wU[None, :])
        out = out + _real_linear_field(lambda c: pbc_stress.j3_strain_field(aos3, c, r_dn), dG_dn - gA_dn * wE_dn[None, :])
        out = out + _real_linear_field(lambda c: pbc_stress.j3_strain_field(aos3, c, pts_dn), gA_dn[:, jD] * wD[None, :])

    # ---- DTN: weighted one-move strain delta (closed form) ----
    if include_dtn and jd.jastrow_dtn_data is not None:
        from .dtn import dtn_geometry_mesh_weighted

        _, dtn_strain = dtn_geometry_mesh_weighted(
            jd.jastrow_dtn_data, r_up, r_dn, pts, e_idx, w, compute_force=False, compute_strain=True
        )
        out = out + dtn_strain
    return out


# --------------------------------------------------------------------------- #
# force field d ln Psi / dR at every mesh config (molecular)
# --------------------------------------------------------------------------- #
def _orb_pieces(orb_data: AOs_base_data | MOs_data) -> tuple[AOs_base_data, jax.Array]:
    if type(orb_data).__name__ == "MOs_data":
        return orb_data.aos_data, jnp.asarray(orb_data._mo_coefficients_jnp)
    return orb_data, jnp.eye(orb_data.num_ao)


def mesh_force(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    rprime: jnpt.ArrayLike,
    base: Mapping[str, object],
    dm: Mapping[str, object],
    geminal_inv: jnpt.ArrayLike | None = None,
    *,
    include_dtn: bool = True,
) -> jax.Array:
    """OR_mesh (Ne, Nv, n_atom, 3) = d ln Psi / dR at every mesh config: det Pulay
    (rank-1) + J1 (one-point rows) + J3 (AO-center deltas). J2 has no R dependence.
    Reuses ``base``/``dm`` from :func:`mesh_ratio_and_drift`.

    ``geminal_inv`` (fast path): base-position running SM ``G^{-1}`` reused by the base
    ``_force_lnPsi`` baseline (the mesh corrections ride the rank-1 ``base``/``dm``).
    """
    from .assemble import _force_lnPsi, _pgto_home_atom

    wf = hamiltonian_data.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    r_up = jnp.asarray(r_up, jnp.float64)
    r_dn = jnp.asarray(r_dn, jnp.float64)
    n_up, n_dn = r_up.shape[0], r_dn.shape[0]
    Ne, Nv = rprime.shape[0], rprime.shape[1]
    K = Ne * Nv
    pts = rprime.reshape(K, 3)
    pts_up = pts[: n_up * Nv]
    pts_dn = pts[n_up * Nv :]
    e_idx = jnp.repeat(jnp.arange(Ne), Nv)
    r_all = jnp.concatenate([r_up, r_dn], 0)
    n_atom = jnp.asarray(hamiltonian_data.structure_data.positions).shape[0]

    # Complex twist: embed the HOLOMORPHIC base force.  The ECP assembly subtracts the
    # holomorphic ``SV * OR_base`` from ``sum_g Vgrid_g OR_mesh[g]`` with COMPLEX weights;
    # a real-projected embedded base would leave the spurious ``Im(SV) * Im(force_base)``
    # cross term in the Re of the product (Gamma/molecular: Im == 0, bit-identical).
    _holo = jnp.iscomplexobj(base["Ginv"])
    force_base = jnp.asarray(
        _force_lnPsi(hamiltonian_data, r_up, r_dn, take_real=not _holo, geminal_inv=geminal_inv)
    )  # (nA, 3)
    OR = jnp.broadcast_to(force_base[None], (K, n_atom, 3))

    # ---- det Pulay corrections (rank-1, per AO leaf -- mirrors grad_lndet_wrt_R) ----
    from . import pgto_deriv

    Ginv, WG, GQ = base["Ginv"], base["WG"], base["GQ"]
    N_dn = base["N_dn"]
    Ku = n_up * Nv
    Kd = K - Ku
    kUidx = jnp.arange(Ku)
    kDidx = jnp.arange(Kd)
    tU, eU, rU, yU = dm["tU"], dm["eU"], dm["ratio_up"], dm["yU"]
    tD, jDi, rD, zD = dm["tD"], dm["jD"], dm["ratio_dn"], dm["zD"]
    sU = tU.at[eU, kUidx].add(-1.0)  # s = t - e_e   (N_up, Ku)
    sD = tD.at[jDi, kDidx].add(-1.0)  # s' = t' - e_j (N_up, Kd)
    Mdn_coeff = GQ[:N_dn].T  # (m_dn, N_dn)

    def _scatter(sK: jax.Array, atom_idx: jax.Array) -> jax.Array:
        # sK (Kc, 3, nao) -> (Kc, nA, 3)
        return jax.ops.segment_sum(jnp.moveaxis(sK, 1, 2).swapaxes(0, 1), atom_idx, num_segments=n_atom).swapaxes(0, 1)

    dR_up_cfg = jnp.zeros((Ku, n_atom, 3))
    dR_dn_cfg = jnp.zeros((Kd, n_atom, 3))
    # == up-side orbital leaves (cotangent Mup = WG; moved column on up-move configs) ==
    for aos, C, kind in pgto_deriv.orbital_leaves(gd.orb_data_up_spin):
        if kind in ("pw", "bspline"):
            continue  # no nuclear dependence
        C = jnp.asarray(C)
        atom_idx = _pgto_home_atom(aos)
        gxu, gyu, gzu = compute_AOs_grad(aos, r_up)
        gA = jnp.stack([gxu, gyu, gzu], 0)  # (3, nao, N_up)
        CM = C.T @ WG  # (nao, N_up)
        # up-move: CM' = CM - CM[:,e] s^T / R ; gA col e -> ga'(pts_up)
        gxp, gyp, gzp = compute_AOs_grad(aos, pts_up)
        gAp = jnp.stack([gxp, gyp, gzp], 0)  # (3, nao, Ku)
        h = jnp.einsum("dcm,mK->Kdc", gA, sU)  # (Ku, 3, nao)
        dg = gAp.transpose(2, 0, 1) - gA[:, :, eU].transpose(2, 0, 1)  # (Ku, 3, nao)
        u_e = CM[:, eU]  # (nao, Ku)
        s_e = rU - 1.0
        corr_u = (
            CM[:, eU].T[:, None, :] * dg
            - (u_e.T / rU[:, None])[:, None, :] * h
            - ((u_e * s_e[None, :] / rU[None, :]).T)[:, None, :] * dg
        )
        dR_up_cfg = dR_up_cfg - _scatter(corr_u, atom_idx)
        # dn-move: CM' = CM + (C^T z) Ginv[j,:] / R (spectator columns; no column swap)
        HU = jnp.einsum("dcm,jm->dcj", gA, Ginv)  # (3, nao, N_up)
        CZ = C.T @ zD  # (nao, Kd)
        corr_u2 = CZ.T[:, None, :] * (HU[:, :, jDi].transpose(2, 0, 1)) / rD[:, None, None]
        dR_dn_cfg = dR_dn_cfg - _scatter(corr_u2, atom_idx)
    # == dn-side orbital leaves (cotangent Mdn_coeff; moved column on dn-move configs) ==
    for aos, C, kind in pgto_deriv.orbital_leaves(gd.orb_data_dn_spin):
        if kind in ("pw", "bspline"):
            continue
        C = jnp.asarray(C)
        atom_idx = _pgto_home_atom(aos)
        gxd, gyd, gzd = compute_AOs_grad(aos, r_dn)
        gA = jnp.stack([gxd, gyd, gzd], 0)  # (3, nao, N_dn)
        CM = C.T @ Mdn_coeff  # (nao, N_dn)
        # up-move: coeff' = Mdn_coeff + y Ginv[:N_dn,e]^T / R (rank-1; no column swap)
        HD = jnp.einsum("dcl,le->dce", gA, Ginv[:N_dn, :])  # (3, nao, N_up)
        CY = C.T @ yU  # (nao, Ku)
        corr_d = CY.T[:, None, :] * (HD[:, :, eU].transpose(2, 0, 1)) / rU[:, None, None]
        dR_up_cfg = dR_up_cfg - _scatter(corr_d, atom_idx)
        # dn-move: CM' = CM - (C^T GQ[j,:]) s'[:N_dn]^T / R ; gA col j -> ga'(pts_dn)
        gxq, gyq, gzq = compute_AOs_grad(aos, pts_dn)
        gAp = jnp.stack([gxq, gyq, gzq], 0)  # (3, nao, Kd)
        dgd = gAp.transpose(2, 0, 1) - gA[:, :, jDi].transpose(2, 0, 1)  # (Kd, 3, nao)
        hd = jnp.einsum("dcl,lK->Kdc", gA, sD[:N_dn])  # (Kd, 3, nao)
        q_j = C.T @ GQ[jDi].T  # (nao, Kd)
        corr_d2 = (
            CM[:, jDi].T[:, None, :] * dgd
            - (q_j.T / rD[:, None])[:, None, :] * hd
            - ((q_j * (rD - 1.0)[None, :] / rD[None, :]).T)[:, None, :] * dgd
        )
        dR_dn_cfg = dR_dn_cfg - _scatter(corr_d2, atom_idx)

    OR = OR + jnp.concatenate([dR_up_cfg, dR_dn_cfg], 0)

    # ---- J1 rows (moved electron only) ----
    if jd.jastrow_one_body_data is not None:
        from .assemble import grad_J1_wrt_R

        j1d = jd.jastrow_one_body_data
        empty = jnp.zeros((0, 3), dtype=jnp.float64)
        g1R_pts = jax.vmap(lambda p: jnp.asarray(grad_J1_wrt_R(j1d, p[None, :], empty)))(pts)  # (K, nA, 3)
        g1R_base = jax.vmap(lambda p: jnp.asarray(grad_J1_wrt_R(j1d, p[None, :], empty)))(r_all)  # (Ne, nA, 3)
        OR = OR + g1R_pts - g1R_base[e_idx]

    # ---- J3 AO-center rows ----
    if jd.jastrow_three_body_data is not None:
        j3d = jd.jastrow_three_body_data
        jb = j3_base(j3d, r_up, r_dn)
        _, _, _, Ap, gAp3, delta, Jd = j3_mesh(j3d, jb, pts, e_idx)
        # Per-AO home atom + validity mask (a COMPOSITE J3 leaf descends to its children exactly like
        # grad_lndet_wrt_R's leaf loop: plane-wave / B-spline children carry NO nuclear dependence
        # (dAO/dR = 0) so their rows contribute nothing to the Pulay force). Each ``s_all`` row below is
        # exactly one AO's gradient contribution, so zeroing the pw/bspline rows is exact. A single
        # GTO/pgto leaf stays bit-identical (one entry, mask all-ones, atom == _pgto_home_atom).
        _atom_parts, _mask_parts = [], []
        for _ch, _C_ch, _kind in pgto_deriv.orbital_leaves(jb["j3d"].orb_data):
            _n_ch = int(_ch.num_ao)
            if _kind in ("pw", "bspline"):
                _atom_parts.append(jnp.zeros(_n_ch, jnp.int32))
                _mask_parts.append(jnp.zeros(_n_ch, jnp.float64))
            else:
                _atom_parts.append(jnp.asarray(_pgto_home_atom(_ch), jnp.int32))
                _mask_parts.append(jnp.ones(_n_ch, jnp.float64))
        atom_j3 = _atom_parts[0] if len(_atom_parts) == 1 else jnp.concatenate(_atom_parts)
        mask_j3 = _mask_parts[0] if len(_mask_parts) == 1 else jnp.concatenate(_mask_parts)  # (n,)
        # force = -sum_m scatter( gA[:, :, m] * V3'[:, m] ); V3'[:, m!=e] = V3[:, m] + Jd,
        # V3'[:, e](moved at r') = V3[:, e]; gA col e -> gAp3.
        Gsum = jnp.sum(jb["gA"], axis=2)  # (3, n)
        s_all = jnp.einsum("dn,nK->Kdn", Gsum, Jd) - jnp.einsum("dnK,nK->Kdn", jb["gA"][:, :, e_idx], Jd)
        s_all = (
            s_all
            + jnp.einsum("dnK,nK->Kdn", gAp3, jb["V3"][:, e_idx])
            - jnp.einsum("dnK,nK->Kdn", jb["gA"][:, :, e_idx], jb["V3"][:, e_idx])
        )
        s_all = s_all * mask_j3[None, None, :]  # pw / B-spline rows carry no nuclear (dAO/dR = 0)
        dR_j3 = -jax.ops.segment_sum(s_all.transpose(2, 0, 1), atom_j3, num_segments=n_atom).transpose(1, 0, 2)  # (K,nA,3)
        OR = OR + dR_j3

    # ---- DTN AO/atom rows: per-config Pulay force delta d J_DTN/dR (closed-form, no autodiff) ----
    if include_dtn and jd.jastrow_dtn_data is not None:
        from . import dtn as dtn_mod

        # ``force_base`` (via _force_lnPsi) already carries grad_dtn_wrt_R(base); the per-config
        # delta dtnR(cfg) - dtnR(base) holds only the chi leg and the f triples of the moved
        # electron, O(N_e N_atom) per config (issue #558). The dense per-config kernel
        # ``_dtn_forceR_mesh`` minus the base stays the parity oracle.
        OR = OR + dtn_mod.dtn_force_mesh_delta(jd.jastrow_dtn_data, r_up, r_dn, pts, e_idx)  # (K,nA,3)

    return OR.reshape(Ne, Nv, n_atom, 3)


def mesh_force_weighted(
    hamiltonian_data: Hamiltonian_data,
    state: Mapping[str, object],
    w: jnpt.ArrayLike,
    geminal_inv: jnpt.ArrayLike | None = None,
    *,
    include_dtn: bool = True,
) -> jax.Array:
    """``sum_k w_k OR(config_k)`` (n_atom, 3) for weights ``w`` (Ne, Nv) or (K,): the
    weighted sum of the :func:`mesh_force` field, with the weights contracted into the
    rank-one factors (no per-config (K, 3, n_AO) corrections, one atom scatter per AO
    leaf).  ``state`` from :func:`mesh_ratio_state`.

    Each per-config det Pulay correction of :func:`mesh_force` is ``(u_k / R_k) (dg_k - h_k)``
    on the moved spin's leaf (``u_k = CM[:, e_k]``, ``dg_k`` the moved AO-gradient column
    change, ``h_k = gA s_k``) and ``CZ_k HU[:, :, j_k] / R_k`` on the other spin's leaf;
    both are linear in per-config vectors, so ``sum_k w_k`` turns into segment sums over
    the moved electron and GEMMs over the configurations.
    """
    from . import pgto_deriv
    from .assemble import _force_lnPsi, _pgto_home_atom

    wf = hamiltonian_data.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    base, dm, e_idx, pts = state["base"], state["dm"], state["e_idx"], state["pts"]
    r_up, r_dn = state["r_up"], state["r_dn"]
    n_up, n_dn = r_up.shape[0], r_dn.shape[0]
    Ne = n_up + n_dn
    n_atom = jnp.asarray(hamiltonian_data.structure_data.positions).shape[0]
    w = jnp.asarray(w).reshape(-1)
    wsum = jnp.sum(w)

    _holo = jnp.iscomplexobj(base["Ginv"])  # holomorphic base force at a twist (see mesh_force)
    force_base = jnp.asarray(_force_lnPsi(hamiltonian_data, r_up, r_dn, take_real=not _holo, geminal_inv=geminal_inv))
    out = wsum * force_base

    Ginv, WG, GQ, N_dn = base["Ginv"], base["WG"], base["GQ"], base["N_dn"]
    tU, eU, rU, yU = dm["tU"], dm["eU"], dm["ratio_up"], dm["yU"]
    tD, jD, rD, zD = dm["tD"], dm["jD"], dm["ratio_dn"], dm["zD"]
    Ku = eU.shape[0]
    Kd = jD.shape[0]
    pts_up, pts_dn = pts[:Ku], pts[Ku:]
    a = w[:Ku] / rU  # (Ku,)
    b = w[Ku:] / rD  # (Kd,)
    sU = tU.at[eU, jnp.arange(Ku)].add(-1.0)  # s = t - e_e   (N_up, Ku)
    sD = tD.at[jD, jnp.arange(Kd)].add(-1.0)  # s' = t' - e_j (N_up, Kd)

    def _scatter(S: jax.Array, atom_idx: jax.Array) -> jax.Array:
        # S (3, nao) -> (nA, 3)
        return jax.ops.segment_sum(S.T, atom_idx, num_segments=n_atom)

    # == up-side orbital leaves (cotangent WG; moved column on up-move configs) ==
    for aos, C, kind in pgto_deriv.orbital_leaves(gd.orb_data_up_spin):
        if kind in ("pw", "bspline"):
            continue  # no nuclear dependence
        C = jnp.asarray(C)
        atom_idx = _pgto_home_atom(aos)
        gA = jnp.stack(compute_AOs_grad(aos, r_up), 0)  # (3, nao, N_up)
        gAp = jnp.stack(compute_AOs_grad(aos, pts_up), 0)  # (3, nao, Ku)
        CM = C.T @ WG  # (nao, N_up)
        # up-moves: sum_k (w_k / R_k) CM[:, e_k] (gAp_k - gA[:, :, e_k] - gA s_k)
        alpha = CM[:, eU] * a[None, :]  # (nao, Ku)
        S = (
            jnp.einsum("dck,ck->dc", gAp, alpha)
            - jnp.einsum("dce,ec->dc", gA, _per_electron_sum(alpha.T, n_up))
            - jnp.einsum("dcm,mc->dc", gA, sU @ alpha.T)
        )
        # dn-moves: sum_k (w_k / R_k) CZ_k HU[:, :, j_k],  HU[d, c, j] = sum_m gA[d, c, m] Ginv[j, m]
        CZ = C.T @ zD  # (nao, Kd)
        S = S + jnp.einsum("dcm,cm->dc", gA, (CZ * b[None, :]) @ Ginv[jD, :])
        out = out - _scatter(S, atom_idx)
    # == dn-side orbital leaves (cotangent GQ[:N_dn].T; moved column on dn-move configs) ==
    for aos, C, kind in pgto_deriv.orbital_leaves(gd.orb_data_dn_spin):
        if kind in ("pw", "bspline"):
            continue
        C = jnp.asarray(C)
        atom_idx = _pgto_home_atom(aos)
        gA = jnp.stack(compute_AOs_grad(aos, r_dn), 0)  # (3, nao, N_dn)
        CM = C.T @ GQ[:N_dn].T  # (nao, N_dn)
        # up-moves: sum_k (w_k / R_k) CY_k HD[:, :, e_k],  HD[d, c, e] = sum_l gA[d, c, l] Ginv[l, e]
        CY = C.T @ yU  # (nao, Ku)
        S = jnp.einsum("dcl,cl->dc", gA, (CY * a[None, :]) @ Ginv[:N_dn, eU].T)
        # dn-moves: sum_k (w_k / R_k) CM[:, j_k] (gAp_k - gA[:, :, j_k] - gA s'_k)
        gAp = jnp.stack(compute_AOs_grad(aos, pts_dn), 0)  # (3, nao, Kd)
        delta = CM[:, jD] * b[None, :]  # (nao, Kd)
        S = S + (
            jnp.einsum("dck,ck->dc", gAp, delta)
            - jnp.einsum("dcl,lc->dc", gA, _per_electron_sum(delta.T, N_dn))
            - jnp.einsum("dcl,lc->dc", gA, sD[:N_dn] @ delta.T)
        )
        out = out - _scatter(S, atom_idx)

    # ---- J1 rows (moved electron only) ----
    if jd.jastrow_one_body_data is not None:
        from .assemble import grad_J1_wrt_R

        j1d = jd.jastrow_one_body_data
        empty = jnp.zeros((0, 3), dtype=jnp.float64)
        r_all = state["r_all"]
        g1R_pts = jax.vmap(lambda p: jnp.asarray(grad_J1_wrt_R(j1d, p[None, :], empty)))(pts)  # (K, nA, 3)
        g1R_base = jax.vmap(lambda p: jnp.asarray(grad_J1_wrt_R(j1d, p[None, :], empty)))(r_all)  # (Ne, nA, 3)
        out = out + jnp.einsum("k,kad->ad", w, g1R_pts) - jnp.einsum("e,ead->ad", _per_electron_sum(w, Ne), g1R_base)

    # ---- J3 AO-center rows: the weighted sum of mesh_force's ``s_all`` ----
    if jd.jastrow_three_body_data is not None:
        jb, Jd, gAp3 = state["jb"], state["j3_Jd"], state["j3_gAp"]
        gA, V3 = jb["gA"], jb["V3"]
        _atom_parts, _mask_parts = [], []
        for _ch, _C_ch, _kind in pgto_deriv.orbital_leaves(jb["j3d"].orb_data):
            _n_ch = int(_ch.num_ao)
            if _kind in ("pw", "bspline"):
                _atom_parts.append(jnp.zeros(_n_ch, jnp.int32))
                _mask_parts.append(jnp.zeros(_n_ch, jnp.float64))
            else:
                _atom_parts.append(jnp.asarray(_pgto_home_atom(_ch), jnp.int32))
                _mask_parts.append(jnp.ones(_n_ch, jnp.float64))
        atom_j3 = _atom_parts[0] if len(_atom_parts) == 1 else jnp.concatenate(_atom_parts)
        mask_j3 = _mask_parts[0] if len(_mask_parts) == 1 else jnp.concatenate(_mask_parts)  # (n,)
        w_e = _per_electron_sum(w, Ne)  # (Ne,)
        S3 = (
            jnp.sum(gA, axis=2) * (Jd @ w)[None, :]
            - jnp.einsum("dne,en->dn", gA, _per_electron_sum((Jd * w[None, :]).T, Ne))
            + jnp.einsum("dnk,nk->dn", gAp3, V3[:, e_idx] * w[None, :])
            - jnp.einsum("dne,ne->dn", gA, V3 * w_e[None, :])
        )
        out = out - jax.ops.segment_sum((S3 * mask_j3[None, :]).T, atom_j3, num_segments=n_atom)

    # ---- DTN AO/atom rows: weighted one-move Pulay-force delta (closed form) ----
    if include_dtn and jd.jastrow_dtn_data is not None:
        from .dtn import dtn_geometry_mesh_weighted

        dR_dtn, _ = dtn_geometry_mesh_weighted(
            jd.jastrow_dtn_data, r_up, r_dn, pts, e_idx, w, compute_force=True, compute_strain=False
        )
        out = out + dR_dtn
    return out


# --------------------------------------------------------------------------- #
# c-axis: collapsed weighted O_k mesh sum  dV_nl/dc = sum_g W_g [O_k(g) - O_k(base)]
# --------------------------------------------------------------------------- #
def fast_supported_c(hamiltonian_data: Hamiltonian_data) -> bool:
    """The c-axis collapse additionally requires the standard lambda_matrix geminal
    leaf (AGPn bare-lambda keeps the streamed path).  Basis-opt blocks (radial
    exponent/coefficient) ARE collapsed (via the effective-cotangent reuse of the
    existing _basis_contract kernels).

    Complex twists are served because every block is a linear
    contraction of the (complex) per-grid weight with rank-1 O_k deltas -- pure GEMMs,
    holomorphic with complex orbital values -- and the basis-opt block routes through
    ``_value_general_leaf(holomorphic=True)`` (its designed twist entry).  The caller
    (``compute_grad_e_L``) takes Re on the assembled c-grad, exactly as the streamed
    holomorphic path does.
    """
    if not fast_supported(hamiltonian_data):
        return False
    gd = hamiltonian_data.wavefunction_data.geminal_data
    jd = hamiltonian_data.wavefunction_data.jastrow_data
    if "lambda_diag" in gd.__dataclass_fields__:
        return False
    from .assemble import _j3_has_opt_basis

    if _j3_has_opt_basis(jd) and type(jd.jastrow_three_body_data.orb_data).__name__ == "MOs_data":
        # J3-basis collapse assembles its cotangents in the BARE-AO basis; an MO-backed
        # J3 leaf's basis contraction expects MO-rank cotangents -> streamed path.
        return False
    return True


def dV_ecp_dc_fast(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    c: Mapping[str, object],
) -> Wavefunction_data:
    """Collapsed-GEMM ``d V_ecp(non-local) / dc``: since ``sumV = sum_g W_g``, the
    weighted mesh identity reduces to ``sum_g W_g [O_k(r'_g) - O_k(r)]`` and only the
    RANK-1 per-config O_k deltas survive -- each block's weighted sum collapses into a
    handful of GEMMs over the one-point orbital/AO stacks.  ``c`` is the
    ``ecp_force._ecp_nl_common`` dict (fast path: carries base/dm/rprime/Vgrid).
    """
    from .assemble import (
        _exp_jastrow_param_kernel,
        _set_geminal_lambda_grad,
        _set_geminal_mo_grad,
        grad_J1_wrt_param,
    )

    wf = hamiltonian_data.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    r_up = jnp.asarray(r_up, jnp.float64)
    r_dn = jnp.asarray(r_dn, jnp.float64)
    n_up, n_dn = r_up.shape[0], r_dn.shape[0]
    base, dm = c["fast_base"], c["fast_dm"]
    rprime = c["rprime"]
    Ne, Nv = rprime.shape[0], rprime.shape[1]
    K = Ne * Nv
    Ku = n_up * Nv
    pts = rprime.reshape(K, 3)
    pts_up = pts[:Ku]
    pts_dn = pts[Ku:]
    e_idx = jnp.repeat(jnp.arange(Ne), Nv)
    r_all = jnp.concatenate([r_up, r_dn], 0)
    Wg = jnp.einsum("etg->eg", c["Vgrid"]).reshape(K)  # per-config full weight (ratio included)
    wU, wD = Wg[:Ku], Wg[Ku:]

    from .assemble import _c_grad_template

    grad_wf = _c_grad_template(wf)  # static orbital tables zero-size (issue #562)

    # ---- determinant blocks ----
    Ginv, WG, GQ = base["Ginv"], base["WG"], base["GQ"]
    orb_up, orb_dn = base["orb_up"], base["orb_dn"]
    N_dn = base["N_dn"]
    tU, eU, rU, yU = dm["tU"], dm["eU"], dm["ratio_up"], dm["yU"]
    tD, jDi, rD, zD, pU, pD = dm["tD"], dm["jD"], dm["ratio_dn"], dm["zD"], dm["pU"], dm["pD"]
    kDidx = jnp.arange(K - Ku)
    sD = tD.at[jDi, kDidx].add(-1.0)  # s' = t' - e_j (N_up, Kd)

    # lambda block: A = orb_up^T (N_up, m_up), B = orb_dn, P = Ginv A
    P = Ginv @ orb_up.T  # (N_up, m_up)
    Wv = (pU - orb_up @ tU) / rU[None, :]  # (m_up, Ku) w-vectors
    Bg = orb_dn @ Ginv[:N_dn, :]  # (m_dn, N_up)
    d_paired = (Wv * wU[None, :]) @ Bg[:, eU].T  # (m_up, m_dn) up-move sum
    d_unp = (Wv * wU[None, :]) @ Ginv[N_dn:, :][:, eU].T  # (m_up, N_up - N_dn)
    kap = orb_up @ Ginv[jDi, :].T  # (m_up, Kd)
    dbD = pD - orb_dn[:, jDi]  # (m_dn, Kd)
    Bs = orb_dn @ sD[:N_dn]  # (m_dn, Kd)
    d_paired = d_paired + (P[jDi, :].T * wD[None, :]) @ dbD.T
    d_paired = d_paired - (kap * (wD / rD)[None, :]) @ Bs.T
    d_paired = d_paired - (kap * (wD * (rD - 1.0) / rD)[None, :]) @ dbD.T
    d_unp = d_unp - (kap * (wD / rD)[None, :]) @ sD[N_dn:].T
    g_gd = _set_geminal_lambda_grad(grad_wf.geminal_data, gd, jnp.hstack([d_paired, d_unp]))

    # MO block (SD / AGPn-MO orbital leaves)
    if type(gd.orb_data_up_spin).__name__ == "MOs_data":
        aos_up = gd.orb_data_up_spin.aos_data
        aos_dn = gd.orb_data_dn_spin.aos_data
        Au = compute_AOs(aos_up, r_up)  # (nao_up, N_up)
        Ad = compute_AOs(aos_dn, r_dn)  # (nao_dn, N_dn)
        AuP = compute_AOs(aos_up, pts_up)  # (nao_up, Ku)
        AdP = compute_AOs(aos_dn, pts_dn)  # (nao_dn, Kd)
        Vau = (AuP - Au @ tU) / rU[None, :]  # (nao_up, Ku)
        dCu = (WG[:, eU] * wU[None, :]) @ Vau.T  # (m_up, nao_up)
        AuGj = Au @ Ginv[jDi, :].T  # (nao_up, Kd)
        dCu = dCu + (zD * (wD / rD)[None, :]) @ AuGj.T
        Mdn_coeff = GQ[:N_dn].T  # (m_dn, N_dn)
        AdGe = Ad @ Ginv[:N_dn, :][:, eU]  # (nao_dn, Ku)
        dCd = (yU * (wU / rU)[None, :]) @ AdGe.T  # (m_dn, nao_dn)
        dad = AdP - Ad[:, jDi]  # (nao_dn, Kd)
        Ads = Ad @ sD[:N_dn]  # (nao_dn, Kd)
        dCd = dCd + (Mdn_coeff[:, jDi] * wD[None, :]) @ dad.T
        dCd = dCd - (GQ[jDi].T * (wD / rD)[None, :]) @ Ads.T
        dCd = dCd - (GQ[jDi].T * (wD * (rD - 1.0) / rD)[None, :]) @ dad.T
        g_gd = _set_geminal_mo_grad(g_gd, gd, dCu, dCd)

    # ---- Jastrow blocks ----
    g_jd = grad_wf.jastrow_data
    if jd.jastrow_one_body_data is not None:
        j1d = jd.jastrow_one_body_data
        empty = jnp.zeros((0, 3), dtype=jnp.float64)
        gp_pts = jax.vmap(lambda p: grad_J1_wrt_param(j1d, p[None, :], empty))(pts)  # (K,)
        gp_base = jax.vmap(lambda p: grad_J1_wrt_param(j1d, p[None, :], empty))(r_all)  # (Ne,)
        gj1 = jnp.sum(Wg * (gp_pts - gp_base[e_idx]))
        g_jd = g_jd.replace(jastrow_one_body_data=g_jd.jastrow_one_body_data.replace(jastrow_1b_param=gj1))
    if jd.jastrow_two_body_data is not None:
        from .kinetic import _dist_and_derivs

        j2d = jd.jastrow_two_body_data
        p = j2d.jastrow_2b_param
        # Williamson smooth periodic distance under PBC (Euclidean molecular); the
        # d(u)/d(param) form is metric-independent, only the distance changes.
        d_new = _dist_and_derivs(j2d.structure_data, r_all[None, :, :] - pts[:, None, :])[0]  # (K, Ne)
        d_old = _dist_and_derivs(j2d.structure_data, r_all[None, :, :] - r_all[e_idx][:, None, :])[0]
        mask = jnp.arange(Ne)[None, :] != e_idx[:, None]
        dn_s = jnp.where(mask, d_new, 1.0)
        do_s = jnp.where(mask, d_old, 1.0)
        if j2d.jastrow_2b_type == "pade":
            f = lambda d: -(d**2) / (2.0 * (1.0 + p * d) ** 2)  # noqa: E731
        else:  # exp
            f = lambda d: _exp_jastrow_param_kernel(p, p * d)  # noqa: E731
        dp2 = jnp.sum(jnp.where(mask, f(dn_s) - f(do_s), 0.0), axis=1)  # (K,)
        gj2 = jnp.sum(Wg * dp2)
        g_jd = g_jd.replace(jastrow_two_body_data=g_jd.jastrow_two_body_data.replace(jastrow_2b_param=gj2))
    if jd.jastrow_three_body_data is not None:
        j3d = jd.jastrow_three_body_data
        jb = j3_base(j3d, r_up, r_dn)
        j3c = jb["j3d"]
        A = jb["A"]  # (n, Ne)
        A_up, A_dn = A[:, :n_up], A[:, n_up:]
        Ap = jnp.real(compute_AOs(j3c.orb_data, pts))  # (n, K); bare-AO leaf-agnostic (PW/bspline/composite)
        delta = Ap - A[:, e_idx]  # (n, K)
        s_up = jnp.sum(A_up, axis=1)
        s_dn = jnp.sum(A_dn, axis=1)
        # same-spin prefix (H_e = sum_{i<e}) / suffix (T_e = sum_{j>e}) partner sums
        cs_up = jnp.cumsum(A_up, axis=1)
        H_up = cs_up - A_up  # (n, n_up): sum_{i<e}
        T_up = s_up[:, None] - cs_up  # (n, n_up): sum_{j>e}
        cs_dn = jnp.cumsum(A_dn, axis=1)
        H_dn = cs_dn - A_dn
        T_dn = s_dn[:, None] - cs_dn
        dU, dD = delta[:, :Ku], delta[:, Ku:]
        eUl = e_idx[:Ku]  # up electron index
        jDl = e_idx[Ku:] - n_up  # dn electron index
        # up-move: dgrad_M = delta (x) (T_up[e] + s_dn) + H_up[e] (x) delta
        right_u = T_up[:, eUl] + s_dn[:, None]  # (n, Ku)
        dM = (dU * wU[None, :]) @ right_u.T + (H_up[:, eUl] * wU[None, :]) @ dU.T
        # dn-move: dgrad_M = delta (x) T_dn[j] + (H_dn[j] + s_up) (x) delta
        left_d = H_dn[:, jDl] + s_up[:, None]  # (n, Kd)
        dM = dM + (dD * wD[None, :]) @ T_dn[:, jDl].T + (left_d * wD[None, :]) @ dD.T
        dj1col = delta @ Wg  # (n,)
        gj3 = jnp.hstack([dM, dj1col[:, None]])
        g_jd = g_jd.replace(jastrow_three_body_data=g_jd.jastrow_three_body_data.replace(j_matrix=gj3))
    if jd.jastrow_dtn_data is not None:
        # DTN param grad: sum_g W_g [O_k(config) - O_k(base)], O_k = d J_DTN/d{alpha,beta,gamma}_indep.
        # J_DTN is parameter-linear, so the config-minus-base O_k holds only the moved electron's
        # pairs, legs and triples: ``dtn_param_mesh_weighted`` sums them with the weights in
        # O(N_e N_atom) per config (issue #566). The dense per-config ``grad_dtn_wrt_param`` minus
        # the base stays the parity oracle.
        from . import dtn as dtn_mod

        updates = dtn_mod.dtn_param_mesh_weighted(jd.jastrow_dtn_data, r_up, r_dn, pts, e_idx, Wg)
        g_dtn = g_jd.jastrow_dtn_data.replace(**updates)
        g_jd = g_jd.replace(jastrow_dtn_data=g_dtn)

    # ---- basis-opt blocks (radial exponents / coefficients) ----
    # The per-config basis O_k is the LINEAR contraction of the radial sensitivity field
    # with the orbital-value cotangent (bar_up = WG, bar_dn = Mdn_coeff; J3: bar_A = V3).
    # With one electron moved the cotangent shifts by a RANK-1 delta and one sensitivity
    # column moves to r', so the weighted mesh sum collapses to TWO extra calls of the
    # existing contraction kernels per side: one at the base positions with an
    # assembled effective cotangent  M_eff = sum_g W_g [Dbar(g) - e_{e_g}(Dbar + bar')(g)],
    # and one at the mesh POINT stack with the weighted moved-column cotangents.
    from .assemble import _geminal_has_opt_basis, _j3_has_opt_basis, _set_basis_grad, _set_j3_basis_grad

    if _geminal_has_opt_basis(gd):
        from .basis import _basis_contract, _basis_contract_value_general, _is_plain_cart_leaf

        # Complex-Bloch twist: the det basis O_k delta must stay HOLOMORPHIC (it multiplies the
        # complex per-grid weight before the caller's final Re -- _value_general_leaf's designed
        # twist entry).  Real Gamma / molecular keep the (bit-identical) real projection.
        _holo_basis = jnp.iscomplexobj(c["ratio"])

        def _side_det(
            orb_data: AOs_base_data | MOs_data,
            bar: jax.Array,
            r: jax.Array,
        ) -> object:
            if _is_plain_cart_leaf(orb_data) and not _holo_basis:
                return _basis_contract(orb_data, bar, None, None, r)
            return _basis_contract_value_general(orb_data, bar, r, holomorphic=_holo_basis)

        def _add_pair(a: object, b: object) -> object:
            return jax.tree_util.tree_map(lambda x, y: x + y, a, b)

        kUidx2 = jnp.arange(Ku)
        sU_full = tU.at[eU, kUidx2].add(-1.0)  # s = t - e_e (N_up, Ku)
        # Per config:  D(basis) = sum_i sens(r_i) . Dbar[:,i]  +  [sens(r') - sens(r_e)] . bar'[:,e]
        # up-side: Dbar = -WG[:,e] s^T / R (up-moves) + z Ginv[j,:] / R (dn-moves);
        # moved-column bar'[:,e] = WG[:,e]/R.
        M_eff_up = -(WG[:, eU] * (wU / rU)[None, :]) @ sU_full.T + (zD * (wD / rD)[None, :]) @ Ginv[jDi, :]
        colw_u = jax.ops.segment_sum(wU / rU, eU, num_segments=n_up)  # (n_up,) -sens(r_e).bar'[:,e]
        M_eff_up = M_eff_up - WG * colw_u[None, :]
        barp_up = WG[:, eU] * (wU / rU)[None, :]  # (m_up, Ku) weighted moved-column cotangent
        det_basis = _add_pair(_side_det(gd.orb_data_up_spin, M_eff_up, r_up), _side_det(gd.orb_data_up_spin, barp_up, pts_up))
        # dn-side: Dbar = y Ginv[:N_dn,e]^T / R (up-moves) - GQ[j,:] s'[:N_dn]^T / R (dn-moves);
        # moved-column bar'[:,j] = GQ[j,:]/R.
        M_eff_dn = (yU * (wU / rU)[None, :]) @ Ginv[:N_dn, :][:, eU].T - (GQ[jDi].T * (wD / rD)[None, :]) @ sD[:N_dn].T
        colw_d = jax.ops.segment_sum(wD / rD, jDi, num_segments=n_dn)  # (n_dn,) -sens(r_j).bar'[:,j]
        M_eff_dn = M_eff_dn - GQ[:N_dn].T * colw_d[None, :]
        barp_dn = GQ[jDi].T * (wD / rD)[None, :]  # (m_dn, Kd)
        det_basis_dn = _add_pair(
            _side_det(gd.orb_data_dn_spin, M_eff_dn, r_dn), _side_det(gd.orb_data_dn_spin, barp_dn, pts_dn)
        )
        g_gd = _set_basis_grad(g_gd, det_basis, det_basis_dn)

    if _j3_has_opt_basis(jd):
        from .basis import _side_j3_value

        j3d = jd.jastrow_three_body_data
        jb = j3_base(j3d, r_up, r_dn)
        j3c = jb["j3d"]
        A, V3, J = jb["A"], jb["V3"], jb["J"]
        Ap = jnp.real(compute_AOs(j3c.orb_data, pts))  # bare-AO leaf-agnostic (PW/bspline/composite)
        delta = Ap - A[:, e_idx]
        Jd = J @ delta  # (n, K)
        JdW = Jd @ Wg  # (n,) total
        part = jax.ops.segment_sum((Jd * Wg[None, :]).T, e_idx, num_segments=Ne).T  # (n, Ne) per-e partials
        wtot_e = jax.ops.segment_sum(Wg, e_idx, num_segments=Ne)  # (Ne,)
        M_eff = JdW[:, None] - part - V3 * wtot_e[None, :]  # (n, Ne)
        barp = V3[:, e_idx] * Wg[None, :]  # (n, K)
        orb = j3c.orb_data
        pieces = [
            _side_j3_value(orb, M_eff[:, :n_up], r_up),
            _side_j3_value(orb, M_eff[:, n_up:], r_dn),
            _side_j3_value(orb, barp[:, :Ku], pts_up),
            _side_j3_value(orb, barp[:, Ku:], pts_dn),
        ]
        gZ = sum(p[0] for p in pieces)
        gc = sum(p[1] for p in pieces)
        g_jd = _set_j3_basis_grad(g_jd, jd, gZ, gc)

    return grad_wf.replace(geminal_data=g_gd, jastrow_data=g_jd)
