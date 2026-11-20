"""Analytic kinetic-energy VJPs for parameters, positions, and strain.

With a Jastrow factor the determinant-only ``(grad lnPsi)^2`` cancellation no
longer holds, so

    T = -1/2 sum lap_J - 1/2 sum lap_orb.WG - 1/2 sum gJ^2 - sum gJ.gD

(WG = W G^-1, gD = determinant per-electron gradient field). For a geminal param only
the trace term + cross term ``-sum gJ.gD`` depend on it (gD uses only the 1st-order
orbital gradient -> no 3rd derivatives, so dT/dc is tractable). For a Jastrow param,
``f_c = dJ/dc`` and ``dT/dc = -1/2 sum lap_r f_c - sum (gJ+gD).grad_r f_c``.

The kernels below evaluate these expressions directly without autodiff.
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

from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp

from jqmc.atomic_orbital import AOs_base_data, compute_AOs, compute_AOs_value_grad_lap
from jqmc.derivatives.fast import resolve_geminal_inv
from jqmc.determinant import AGP_data, AGPn_data, SD_data, _orbital_value_grad_lap_tables
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    compute_grads_and_laplacian_Jastrow_part,
)
from jqmc.structure import Structure_data

_GeminalData: TypeAlias = AGP_data | AGPn_data | SD_data
_ReverseCotangents: TypeAlias = dict[str, jax.Array | None]
_ArrayPair: TypeAlias = tuple[jax.Array, jax.Array]
_ArrayTriple: TypeAlias = tuple[jax.Array, jax.Array, jax.Array]
_ArrayQuad: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
_PeriodicLeafKind: TypeAlias = Literal["pgto", "pw", "bspline"]


def _sd_apply_W(orb_dn: jax.Array, right: jax.Array, n_dn: int) -> jax.Array:
    """Apply the direct-SD orbital factor ``W`` without forming a selector."""
    return jnp.concatenate([orb_dn @ right[:n_dn], right[n_dn:]], axis=0)


def _sd_apply_WT(orb_dn: jax.Array, right: jax.Array, n_dn: int) -> jax.Array:
    """Apply ``W.T`` for the direct occupied-SD factorization."""
    return jnp.concatenate([orb_dn.T @ right[:n_dn], right[n_dn:]], axis=0)


def _sd_geminal_reverse(
    gd: SD_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    gJ_up: jax.Array,
    gJ_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> _ReverseCotangents:
    """Analytic kinetic reverse pass for the occupied-orbital SD form."""
    n_dn = gd.num_electron_dn
    orb_up_fused, gou, lap_up, orb_dn_fused, godn, lap_dn = _orbital_value_grad_lap_tables(gd, r_up, r_dn)
    if geminal_inv is None:
        orb_up_inverse, orb_dn_inverse = orb_up_fused, orb_dn_fused
    else:
        # The MCMC/GFMC running inverse is maintained from the value-only
        # determinant-ratio tables.  Keep that inverse route separate from
        # the fused value/grad/lap tables used by the kinetic forward.
        orb_up_inverse = gd.compute_orb_api(gd.orb_data_up_spin, r_up)
        orb_dn_inverse = gd.compute_orb_api(gd.orb_data_dn_spin, r_dn)
    G = jnp.hstack([orb_up_inverse[:n_dn].T @ orb_dn_inverse, orb_up_inverse[n_dn:].T])
    Ginv = resolve_geminal_inv(G, geminal_inv)
    Q_fused = orb_up_fused[:n_dn].T
    WG = _sd_apply_W(orb_dn_fused, Ginv, n_dn)
    GQ = Ginv @ Q_fused
    bar_lap_up = -0.5 * WG
    bar_lap_dn = -0.5 * GQ[:n_dn].T
    bar_WG = -0.5 * lap_up - jnp.einsum("kd,dak->ak", gJ_up, gou)
    bar_gou = -jnp.einsum("kd,ak->dak", gJ_up, WG)
    bar_GQ_rows = -0.5 * lap_dn.T - jnp.einsum("ld,dbl->lb", gJ_dn, godn)
    bar_GQ = jnp.zeros((G.shape[0], n_dn), dtype=bar_GQ_rows.dtype).at[:n_dn].set(bar_GQ_rows)
    bar_godn = -jnp.einsum("ld,lb->dbl", gJ_dn, GQ[:n_dn])
    bar_Ginv = _sd_apply_WT(orb_dn_fused, bar_WG, n_dn) + bar_GQ @ Q_fused.T
    bar_W_direct = bar_WG @ Ginv.T
    bar_Q_direct = Ginv.T @ bar_GQ
    bar_G = -Ginv.T @ bar_Ginv @ Ginv.T
    bar_W_inverse = orb_up_inverse @ bar_G
    bar_orb_up_inverse = _sd_apply_W(orb_dn_inverse, bar_G.T, n_dn)
    bar_orb_dn_inverse = bar_W_inverse[:n_dn, :n_dn]
    bar_orb_up_fused = jnp.concatenate([bar_Q_direct.T, jnp.zeros_like(bar_orb_up_inverse[n_dn:])], axis=0)
    bar_orb_dn_fused = bar_W_direct[:n_dn, :n_dn]
    gD_up = jnp.einsum("dak,ak->kd", gou, WG)
    gD_dn = jnp.einsum("dbl,lb->ld", godn, GQ[:n_dn])
    return dict(
        bar_orb_up=bar_orb_up_inverse + bar_orb_up_fused,
        bar_orb_dn=bar_orb_dn_inverse + bar_orb_dn_fused,
        bar_orb_up_inverse=bar_orb_up_inverse,
        bar_orb_dn_inverse=bar_orb_dn_inverse,
        bar_orb_up_fused=bar_orb_up_fused,
        bar_orb_dn_fused=bar_orb_dn_fused,
        bar_lap_up=bar_lap_up,
        bar_lap_dn=bar_lap_dn,
        bar_gou=bar_gou,
        bar_godn=bar_godn,
        gD_up=gD_up,
        gD_dn=gD_dn,
        bar_lam_p=None,
        bar_lam_unp=None,
    )


def _dist_and_derivs(structure_data: Structure_data, disp: jax.Array) -> _ArrayQuad:
    """(d, grad_d, gd2, lap_d) for a Cartesian displacement field ``disp`` (..., 3).

    Molecular: Euclidean -- d = |disp|, grad_d = unit, |grad_d|^2 = 1, lap_d = 2/d.
    PBC: positive full-Gram periodic geometry, with closed-form gradient and
    Laplacian. The Cartesian leading term retains the Euclidean cusp also
    in nonorthogonal cells. No coordinate autodiff is used. ``pbc_flag`` is
    static (pytree_node=False), so the molecular branch compiles unaffected. A safe d
    (=1 where d==0) is used for the divisions; callers mask coincident points anyway."""
    if not structure_data.pbc_flag:
        d = jnp.linalg.norm(disp, axis=-1)
        dsafe = jnp.where(d > 0.0, d, 1.0)
        grad_d = disp / dsafe[..., None]
        gd2 = jnp.sum(grad_d * grad_d, axis=-1)
        return d, grad_d, gd2, 2.0 / dsafe

    from jqmc.structure import _inv_3x3

    cell = jnp.asarray(structure_data._cell_jnp, jnp.float64)
    inv_cell = _inv_3x3(cell)  # (3, 3) [m, k]
    from jqmc.jastrow_factor import _periodic_geometry_components

    S, numerator, trace = _periodic_geometry_components(disp, cell, inv_cell)
    Ssafe = jnp.where(S > 0.0, S, 1.0)
    d = jnp.where(S > 0.0, jnp.sqrt(Ssafe), 0.0)
    dsafe = jnp.where(S > 0.0, jnp.sqrt(Ssafe), 1.0)
    grad_d = jnp.stack(numerator, axis=-1) / dsafe[..., None]
    gd2 = jnp.sum(grad_d * grad_d, axis=-1)
    lap_d = (trace - gd2) / dsafe
    return d, grad_d, gd2, lap_d


def _geminal_reverse(
    gd: _GeminalData,
    r_up: jax.Array,
    r_dn: jax.Array,
    gJ_up: jax.Array,
    gJ_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> _ReverseCotangents:
    """Common reverse pass -> cotangents on (orb, lap, gou, godn) and (lam_p, lam_unp)."""
    from jqmc.determinant import SD_data

    if isinstance(gd, SD_data):
        return _sd_geminal_reverse(gd, r_up, r_dn, gJ_up, gJ_dn, geminal_inv=geminal_inv)
    M_dn, N_dn = gd.orb_num_dn, gd.num_electron_dn
    orb_up_fused, gou, lap_up, orb_dn_fused, godn, lap_dn = _orbital_value_grad_lap_tables(gd, r_up, r_dn)
    if geminal_inv is None:
        orb_up_inverse, orb_dn_inverse = orb_up_fused, orb_dn_fused
    else:
        orb_up_inverse = gd.compute_orb_api(gd.orb_data_up_spin, r_up)
        orb_dn_inverse = gd.compute_orb_api(gd.orb_data_dn_spin, r_dn)
    lam_p, lam_unp = jnp.hsplit(gd._lambda_matrix_jnp, [M_dn])
    W_fused = jnp.hstack([lam_p @ orb_dn_fused, lam_unp])
    W_inverse = jnp.hstack([lam_p @ orb_dn_inverse, lam_unp])
    G = orb_up_inverse.T @ W_inverse
    Ginv = resolve_geminal_inv(G, geminal_inv)  # fast: running SM inverse; else SVD-pinv
    Q_fused = orb_up_fused.T @ lam_p
    WG = W_fused @ Ginv
    GQ = Ginv @ Q_fused
    bar_lap_up = -0.5 * WG
    bar_lap_dn = -0.5 * GQ[:N_dn].T
    bar_WG = -0.5 * lap_up - jnp.einsum("kd,dak->ak", gJ_up, gou)
    bar_gou = -jnp.einsum("kd,ak->dak", gJ_up, WG)
    bar_GQ_rows = -0.5 * lap_dn.T - jnp.einsum("ld,dbl->lb", gJ_dn, godn)
    bar_GQ = jnp.zeros((G.shape[0], M_dn), dtype=bar_GQ_rows.dtype).at[:N_dn].set(bar_GQ_rows)
    bar_godn = -jnp.einsum("ld,lb->dbl", gJ_dn, GQ[:N_dn])
    bar_Ginv = W_fused.T @ bar_WG + bar_GQ @ Q_fused.T
    bar_W_direct = bar_WG @ Ginv.T
    bar_Q_direct = Ginv.T @ bar_GQ
    bar_G = -Ginv.T @ bar_Ginv @ Ginv.T
    bar_W_inverse = orb_up_inverse @ bar_G
    bar_orb_up_inverse = W_inverse @ bar_G.T
    bar_orb_dn_inverse = lam_p.T @ bar_W_inverse[:, :N_dn]
    bar_orb_up_fused = lam_p @ bar_Q_direct.T
    bar_orb_dn_fused = lam_p.T @ bar_W_direct[:, :N_dn]
    bar_lam_p = (
        bar_W_direct[:, :N_dn] @ orb_dn_fused.T + orb_up_fused @ bar_Q_direct + bar_W_inverse[:, :N_dn] @ orb_dn_inverse.T
    )
    bar_lam_unp = bar_W_direct[:, N_dn:] + bar_W_inverse[:, N_dn:]
    gD_up = jnp.einsum("dak,ak->kd", gou, WG)
    gD_dn = jnp.einsum("dbl,lb->ld", godn, GQ[:N_dn])
    return dict(
        bar_orb_up=bar_orb_up_inverse + bar_orb_up_fused,
        bar_orb_dn=bar_orb_dn_inverse + bar_orb_dn_fused,
        bar_orb_up_inverse=bar_orb_up_inverse,
        bar_orb_dn_inverse=bar_orb_dn_inverse,
        bar_orb_up_fused=bar_orb_up_fused,
        bar_orb_dn_fused=bar_orb_dn_fused,
        bar_lap_up=bar_lap_up,
        bar_lap_dn=bar_lap_dn,
        bar_gou=bar_gou,
        bar_godn=bar_godn,
        gD_up=gD_up,
        gD_dn=gD_dn,
        bar_lam_p=bar_lam_p,
        bar_lam_unp=bar_lam_unp,
    )


def grad_T_wrt_lambda_withJ(
    gd: AGP_data | AGPn_data,
    jastrow_data: Jastrow_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    from jqmc.determinant import AGPn_data, SD_data

    if isinstance(gd, SD_data):
        raise TypeError("canonical SD_data has no lambda parameter block")
    # A closed-shell AGPn with exactly one paired natural orbital per
    # electron has
    #
    #   det(O_up.T @ diag(lambda) @ O_dn)
    #     = det(O_up.T) prod(lambda) det(O_dn).
    #
    # Each occupation therefore changes only the configuration-independent
    # normalization and the local energy derivative is identically zero,
    # with or without a Jastrow.  Return that invariant directly instead of
    # forming an ill-conditioned reverse-pass cancellation.
    if (
        isinstance(gd, AGPn_data)
        and gd.lambda_unpaired.size == 0
        and int(gd.orb_num_up) == int(r_up.shape[0])
        and int(gd.orb_num_dn) == int(r_dn.shape[0])
    ):
        return jnp.zeros_like(gd._lambda_matrix_jnp)
    gJ_up, gJ_dn, _, _ = compute_grads_and_laplacian_Jastrow_part(jastrow_data, r_up, r_dn)
    b = _geminal_reverse(gd, r_up, r_dn, gJ_up, gJ_dn, geminal_inv=geminal_inv)
    return jnp.hstack([b["bar_lam_p"], b["bar_lam_unp"]])


def grad_T_wrt_mo_withJ(
    gd: AGPn_data | SD_data,
    jastrow_data: Jastrow_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    gJ_up, gJ_dn, _, _ = compute_grads_and_laplacian_Jastrow_part(jastrow_data, r_up, r_dn)
    b = _geminal_reverse(gd, r_up, r_dn, gJ_up, gJ_dn, geminal_inv=geminal_inv)

    def to_C(
        side: Literal["orb_data_up_spin", "orb_data_dn_spin"],
        bar_orb_inverse: jax.Array,
        bar_orb_fused: jax.Array,
        bar_lap: jax.Array,
        bar_g: jax.Array,
        r: jax.Array,
    ) -> jax.Array:
        aos = getattr(gd, side).aos_data
        AO_fused, gx, gy, gz, lapAO = compute_AOs_value_grad_lap(aos, r)
        gAO = jnp.stack((gx, gy, gz), 0)
        value_term = bar_orb_fused @ AO_fused.T
        if geminal_inv is None:
            value_term = value_term + bar_orb_inverse @ AO_fused.T
        else:
            value_term = value_term + bar_orb_inverse @ compute_AOs(aos, r).T
        return value_term + bar_lap @ lapAO.T + jnp.einsum("dak,dnk->an", bar_g, gAO)

    C_up = to_C(
        "orb_data_up_spin",
        b["bar_orb_up_inverse"],
        b["bar_orb_up_fused"],
        b["bar_lap_up"],
        b["bar_gou"],
        r_up,
    )
    C_dn = to_C(
        "orb_data_dn_spin",
        b["bar_orb_dn_inverse"],
        b["bar_orb_dn_fused"],
        b["bar_lap_dn"],
        b["bar_godn"],
        r_dn,
    )
    return C_up, C_dn


def _det_grad_field(
    gd: _GeminalData,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> _ArrayPair:
    """Determinant per-electron gradient gD_up (N_up,3), gD_dn (N_dn,3)."""
    from jqmc.determinant import SD_data

    M_dn, N_dn = gd.orb_num_dn, gd.num_electron_dn
    # This field differentiates determinant ratios (including the ECP dltmove
    # reweight), not the kinetic fused value/grad/lap primal.  Its value and
    # gradient must therefore stay on the same boundary as the ECP mesh base
    # (``one_move_mesh.det_base``): the running inverse's value-only ratio
    # tables when one is supplied, else the fused table whose fp64 gradients
    # are contracted with G^{-1} (#681).
    if geminal_inv is None:
        orb_up, gou, _, orb_dn, godn, _ = _orbital_value_grad_lap_tables(gd, r_up, r_dn)
    else:
        orb_up = gd.compute_orb_api(gd.orb_data_up_spin, r_up)
        orb_dn = gd.compute_orb_api(gd.orb_data_dn_spin, r_dn)
        gou = jnp.stack(gd.compute_orb_grad_api(gd.orb_data_up_spin, r_up), 0)
        godn = jnp.stack(gd.compute_orb_grad_api(gd.orb_data_dn_spin, r_dn), 0)
    if isinstance(gd, SD_data):
        G = jnp.hstack([orb_up[:N_dn].T @ orb_dn, orb_up[N_dn:].T])
        Q = orb_up[:N_dn].T
    else:
        lam_p, lam_unp = jnp.hsplit(gd._lambda_matrix_jnp, [M_dn])
        W = jnp.hstack([lam_p @ orb_dn, lam_unp])
        G = orb_up.T @ W
        Q = orb_up.T @ lam_p
    Ginv = resolve_geminal_inv(G, geminal_inv)  # fast: running SM inverse; else SVD-pinv
    WG = _sd_apply_W(orb_dn, Ginv, N_dn) if isinstance(gd, SD_data) else W @ Ginv
    GQ = Ginv @ Q
    gD_up = jnp.einsum("dak,ak->kd", gou, WG)
    gD_dn = jnp.einsum("dbl,lb->ld", godn, GQ[:N_dn])
    return gD_up, gD_dn


def _kinetic_det_grad_field(
    gd: _GeminalData,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> _ArrayPair:
    """Determinant drift on the fused kinetic primal.

    The kinetic forward obtains its orbital values, gradients, and Laplacians
    from one fused evaluator.  In mixed precision its value table is not
    numerically interchangeable with the standalone determinant-ratio table
    used by :func:`_det_grad_field`.  Jastrow cross terms in the kinetic
    derivative must therefore build ``gD`` from this fused table.  A supplied
    running inverse is reused directly; no additional factorization is added.
    """
    from jqmc.determinant import SD_data

    M_dn, N_dn = gd.orb_num_dn, gd.num_electron_dn
    orb_up, gou, _lap_up, orb_dn, godn, _lap_dn = _orbital_value_grad_lap_tables(gd, r_up, r_dn)
    if isinstance(gd, SD_data):
        G = jnp.hstack([orb_up[:N_dn].T @ orb_dn, orb_up[N_dn:].T])
        Q = orb_up[:N_dn].T
        Ginv = resolve_geminal_inv(G, geminal_inv)
        WG = _sd_apply_W(orb_dn, Ginv, N_dn)
    else:
        lam_p, lam_unp = jnp.hsplit(gd._lambda_matrix_jnp, [M_dn])
        W = jnp.hstack([lam_p @ orb_dn, lam_unp])
        G = orb_up.T @ W
        Q = orb_up.T @ lam_p
        Ginv = resolve_geminal_inv(G, geminal_inv)
        WG = W @ Ginv
    GQ = Ginv @ Q
    gD_up = jnp.einsum("dak,ak->kd", gou, WG)
    gD_dn = jnp.einsum("dbl,lb->ld", godn, GQ[:N_dn])
    return gD_up, gD_dn


def _orbital_side_r(
    gd: _GeminalData,
    b: _ReverseCotangents,
    r_up: jax.Array,
    r_dn: jax.Array,
) -> _ArrayPair:
    """Per-electron orbital-side dT_det/dr (+ -gJ.gD cross via bar_gou): the kinetic
    VJP cotangents (b from _geminal_reverse) contracted with the AO grad / grad-of-Lap
    / Hessian. Returns (gou_up (N_up,3), gou_dn (N_dn,3))."""
    from .ao_deriv import ao_grad_gradlap_hessian, orb_to_cart_pieces

    res = {}
    for side, bar_orb, bar_lap, bar_gou, r in (
        ("orb_data_up_spin", b["bar_orb_up"], b["bar_lap_up"], b["bar_gou"], r_up),
        ("orb_data_dn_spin", b["bar_orb_dn"], b["bar_lap_dn"], b["bar_godn"], r_dn),
    ):
        aos, C = orb_to_cart_pieces(getattr(gd, side))
        CM_orb = C.T @ bar_orb
        CM_lap = C.T @ bar_lap
        CM_gou = jnp.einsum("ac,dak->dck", C, bar_gou)
        # grad (3, num_ao, N), grad-of-Laplacian (3, num_ao, N), Hessian (3, 3, num_ao, N): one primitive sweep
        gA, gL, H = ao_grad_gradlap_hessian(aos, r)
        res[side] = (
            jnp.einsum("bk,cbk->kc", CM_orb, gA) + jnp.einsum("bk,cbk->kc", CM_lap, gL) + jnp.einsum("dbk,cdbk->kc", CM_gou, H)
        )
    return res["orb_data_up_spin"], res["orb_data_dn_spin"]


def _j2_hr(p: jax.Array | float, r: jax.Array, kind: str) -> _ArrayTriple:
    """Pair-function radial derivs h', h'', h''' of u(r) (so h'=du/dr)."""
    if kind == "pade":
        u = 1.0 + p * r
        return 1.0 / (2.0 * u**2), -p / u**3, 3.0 * p**2 / u**4
    if kind == "exp":
        e = jnp.exp(-p * r)
        return 0.5 * e, -0.5 * p * e, 0.5 * p**2 * e
    raise ValueError(kind)


def _j1_fr(j1: Jastrow_one_body_data, dist: jax.Array) -> _ArrayTriple:
    """One-body radial derivs f', f'', f''' (per electron x atom)."""
    Z = jnp.asarray(j1.structure_data.atomic_numbers) - jnp.asarray(j1.core_electrons)
    p = j1.jastrow_1b_param
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


def _leaf_ao_kinetic_kernels(
    leaf: AOs_base_data,
    C_leaf: jax.Array,
    kind: _PeriodicLeafKind,
    r: jax.Array,
) -> _ArrayQuad:
    """(C, gA, gL, H) for one AO leaf: the cotangent map C (num_mo->leaf AO rank) and the per-AO
    grad / grad-of-laplacian / Hessian for the kinetic contraction. pgto -> de-tile+c2s to the cart
    leaf (pgto_deriv kernels); plane-wave -> closed-form kernels (ao.pw_*), native rank, no fold."""
    from . import pgto_deriv as P

    if kind == "pgto":
        aos, T_mol = P.sphe_leaf_to_cart(leaf)
        gA, gL, H = P.pgto_grad_gradlap_hessian(aos, r)  # one primitive sweep + one image fold
        return jnp.asarray(C_leaf) @ jnp.asarray(T_mol), gA, gL, H
    if kind == "pw":
        from . import ao as A

        return (
            jnp.asarray(C_leaf),
            A.pw_grad(leaf, r),
            A.pw_gradlap(leaf, r),
            A.pw_hessian(leaf, r),
        )
    if kind == "bspline":
        from . import ao as A

        if "complex" in type(leaf).__name__:  # complex-twist Bloch spline: phase-carrying kinetic fields
            grad, gradlap, hess = A._bspline_complex_kinetic_fields(leaf, r)
        else:
            grad, gradlap, hess = A._bspline_kinetic_fields(leaf, r)
        return jnp.asarray(C_leaf), grad, gradlap, hess
    # Defensive guard (currently unreachable): on the periodic orbital side ``orbital_leaves`` yields
    # only pgto / pw / bspline kinds (all handled above); kind == "gto" is molecular and never reaches
    # this periodic path. Kept for any future AO leaf kind.
    raise NotImplementedError(f"_leaf_ao_kinetic_kernels: AO leaf kind {kind!r} not supported")


def _orbital_side_r_pbc(
    gd: _GeminalData,
    b: _ReverseCotangents,
    r_up: jax.Array,
    r_dn: jax.Array,
) -> _ArrayPair:
    """Periodic counterpart of _orbital_side_r: per-electron orbital-side dT_det/dr using the per-AO
    grad/grad-of-laplacian/Hessian (no nucleus scatter -- this is d/dr). Leaf-aware (pgto + PW
    composites). EAGER."""
    from . import pgto_deriv as P

    res = {}
    for side, bar_orb, bar_lap, bar_gou, r in (
        ("orb_data_up_spin", b["bar_orb_up"], b["bar_lap_up"], b["bar_gou"], r_up),
        ("orb_data_dn_spin", b["bar_orb_dn"], b["bar_lap_dn"], b["bar_godn"], r_dn),
    ):
        from jqmc.orbital_tiling import TiledOrbitalSet_data
        from .orbital_tiling import tiled_field_vjp

        orb = getattr(gd, side)
        if isinstance(orb, TiledOrbitalSet_data):
            res[side] = jnp.real(tiled_field_vjp(orb, r, bar_orb, bar_gou, bar_lap, wrt="r"))
            continue
        acc = 0.0
        for leaf, C_leaf, kind in P.orbital_leaves(getattr(gd, side)):
            C, gA, gL, H = _leaf_ao_kinetic_kernels(leaf, C_leaf, kind, r)
            CM_orb = C.T @ jnp.asarray(bar_orb)
            CM_lap = C.T @ jnp.asarray(bar_lap)
            CM_gou = jnp.einsum("ac,dak->dck", C, jnp.asarray(bar_gou))
            acc = (
                acc
                + jnp.einsum("bk,cbk->kc", CM_orb, gA)
                + jnp.einsum("bk,cbk->kc", CM_lap, gL)
                + jnp.einsum("dbk,cdbk->kc", CM_gou, H)
            )
        res[side] = jnp.real(acc)
    return res["orb_data_up_spin"], res["orb_data_dn_spin"]


def _j2_drift_pbc(
    j2: Jastrow_two_body_data,
    r_all: jax.Array,
    cot: jax.Array,
    Ne: int,
) -> jax.Array:
    """Periodic e-e J2 drift dT_J2/dr (Ne, 3): the euclidean radial Hessian/grad-of-laplacian
    replaced by the smooth periodic distance ones (force._smooth_dist_kinematics)."""
    from .force import _smooth_dist_kinematics

    p = j2.jastrow_2b_param  # keep as a (possibly traced) scalar -- no host float()
    ra = jnp.asarray(r_all, jnp.float64)
    disp = ra[None, :, :] - ra[:, None, :]  # (Ne, Ne, 3) = r_k - r_e
    d, gd, q, Hd, lapd, gld = _smooth_dist_kinematics(j2.structure_data, disp)
    off = ~jnp.eye(Ne, dtype=bool)
    dsafe = jnp.where(off, d, 1.0)
    h1, h2, h3 = _j2_hr(p, dsafe, j2.jastrow_2b_type)
    Hess = h2[..., None, None] * (gd[..., :, None] * gd[..., None, :]) + h1[..., None, None] * Hd  # (Ne,Ne,3,3)
    Hd_gd = jnp.einsum("ekmn,ekn->ekm", Hd, gd)
    Lpterm = (h3 * q)[..., None] * gd + 2.0 * h2[..., None] * Hd_gd + (h2 * lapd)[..., None] * gd + h1[..., None] * gld
    m = off[..., None]
    out = (Lpterm * m).sum(axis=1)
    cotn = jnp.real(jnp.asarray(cot))
    dcot = (cotn[None, :, :] - cotn[:, None, :]) * m
    out = out + jnp.einsum("ekc,ekcd->ed", dcot, Hess)
    return out


def _j1_drift_pbc(j1: Jastrow_one_body_data, r_all: jax.Array, cot: jax.Array) -> jax.Array:
    """Periodic e-N J1 drift dT_J1/dr (Ne, 3)."""
    from .force import _smooth_dist_kinematics

    ra = jnp.asarray(r_all, jnp.float64)
    R = jnp.asarray(j1.structure_data.positions, jnp.float64)
    disp = ra[:, None, :] - R[None, :, :]  # (Ne, nA, 3) = r_e - R_I
    d, gd, q, Hd, lapd, gld = _smooth_dist_kinematics(j1.structure_data, disp)
    f1, f2, f3 = _j1_fr(j1, d)  # (Ne, nA)
    Hess = (f2[..., None, None] * (gd[..., :, None] * gd[..., None, :]) + f1[..., None, None] * Hd).sum(axis=1)  # (Ne,3,3)
    Hd_gd = jnp.einsum("eImn,eIn->eIm", Hd, gd)
    Lpterm = (f3 * q)[..., None] * gd + 2.0 * f2[..., None] * Hd_gd + (f2 * lapd)[..., None] * gd + f1[..., None] * gld
    dlap = Lpterm.sum(axis=1)  # (Ne, 3)
    cotn = jnp.real(jnp.asarray(cot))
    out = -0.5 * dlap - jnp.einsum("ec,ecd->ed", cotn, Hess)
    return out


def grad_T_wrt_r(
    gd: AGP_data | AGPn_data | SD_data,
    jastrow_data: Jastrow_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """Full with-Jastrow electron-coordinate kinetic gradient dT/dr (Ne, 3).

    This is the kinetic force expression as a direct per-electron gradient: the
    orbital response plus ``-0.5 sum d lap_J/dr - sum (gJ+gD).d gJ/dr``. J1 and
    J2 are accumulated here and J3 is added when present.

    ``geminal_inv`` is the running Sherman-Morrison inverse at the base walker.
    Both the kinetic reverse and determinant drift reuse it, avoiding a fresh
    SVD pseudoinverse."""
    gJ_up, gJ_dn, _, _ = compute_grads_and_laplacian_Jastrow_part(jastrow_data, r_up, r_dn)
    b = _geminal_reverse(gd, r_up, r_dn, gJ_up, gJ_dn, geminal_inv=geminal_inv)
    od_up = gd.orb_data_up_spin
    aos0 = od_up.aos_data if type(od_up).__name__ == "MOs_data" else od_up
    pbc = bool(aos0.structure_data.pbc_flag)
    ou, od = _orbital_side_r_pbc(gd, b, r_up, r_dn) if pbc else _orbital_side_r(gd, b, r_up, r_dn)
    dT = jnp.concatenate([jnp.asarray(ou), jnp.asarray(od)], axis=0)
    # Reuse the fused-primal determinant drift already formed by the reverse
    # pass.  Besides matching the production kinetic formula in mixed mode,
    # this avoids an extra standalone value+gradient AO evaluation.
    gD_up, gD_dn = b["gD_up"], b["gD_dn"]
    gJ = jnp.concatenate([jnp.asarray(gJ_up), jnp.asarray(gJ_dn)], axis=0)
    gD = jnp.concatenate([jnp.asarray(gD_up), jnp.asarray(gD_dn)], axis=0)
    cot = gJ + gD  # (Ne, 3)
    r_all = jnp.concatenate([jnp.asarray(r_up), jnp.asarray(r_dn)], axis=0)
    Ne = len(r_all)
    eye3 = jnp.eye(3)

    j2 = jastrow_data.jastrow_two_body_data
    if j2 is not None and not pbc:
        p = j2.jastrow_2b_param
        diff = r_all[None, :, :] - r_all[:, None, :]  # diff[e,k] = r_k - r_e
        dist = jnp.linalg.norm(diff, axis=-1)
        off = ~jnp.eye(Ne, dtype=bool)
        dsafe = jnp.where(off, dist, 1.0)
        u = diff / dsafe[..., None]
        h1, h2, h3 = _j2_hr(p, dsafe, j2.jastrow_2b_type)
        uu = u[..., :, None] * u[..., None, :]  # (Ne,Ne,3,3)
        Hess = h2[..., None, None] * uu + (h1 / dsafe)[..., None, None] * (eye3 - uu)
        Lp = h3 + 2.0 * h2 / dsafe - 2.0 * h1 / dsafe**2
        m = off[..., None]
        dT = dT + (Lp[..., None] * u * m).sum(axis=1)  # 2 x 0.5 Lp u per pair
        dcot = (cot[None, :, :] - cot[:, None, :]) * m  # cot[k] - cot[e]
        dT = dT + jnp.einsum("ekc,ekcd->ed", dcot, Hess)
    elif j2 is not None and pbc:
        dT = dT + _j2_drift_pbc(j2, r_all, cot, Ne)

    j1 = jastrow_data.jastrow_one_body_data
    if j1 is not None and not pbc:
        R = jnp.asarray(j1.structure_data.positions)
        diff = r_all[:, None, :] - R[None, :, :]  # (Ne, nA, 3)
        dist = jnp.linalg.norm(diff, axis=-1)
        u = diff / dist[..., None]
        f1, f2, f3 = _j1_fr(j1, dist)
        Lp = f3 + 2.0 * f2 / dist - 2.0 * f1 / dist**2
        uu = u[..., :, None] * u[..., None, :]
        Hess = (f2[..., None, None] * uu + (f1 / dist)[..., None, None] * (eye3 - uu)).sum(axis=1)
        dlap = jnp.einsum("eI,eIc->ec", Lp, u)
        dT = dT + (-0.5 * dlap - jnp.einsum("ec,ecd->ed", cot, Hess))
    elif j1 is not None and pbc:
        dT = dT + _j1_drift_pbc(j1, r_all, cot)

    j3 = jastrow_data.jastrow_three_body_data
    if j3 is not None:
        dT = dT + _j3_side_r(j3, cot, r_up, r_dn)
    return dT


def _j3_side_r(
    j3: Jastrow_three_body_data,
    cot: jax.Array,
    r_up: jax.Array,
    r_dn: jax.Array,
) -> jax.Array:
    """Return per-electron ``dT_J3/dr`` from the J3 gradient/laplacian VJP."""
    from .ao_deriv import j3_basis_derivs

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
    n_up_e = len(r_up)
    cot_u = jnp.asarray(cot[:n_up_e])
    cot_d = jnp.asarray(cot[n_up_e:])
    de_blocks = []
    for coefA, cotS, gO, lO, H, GL, K in (
        (coefA_up, cot_u, gOu, lOu, Hu, GLu, Ku),
        (coefA_dn, cot_d, gOd, lOd, Hd, GLd, Kd),
    ):
        bar_lO = -0.5 * coefA
        bar_gO = -jnp.einsum("kd,ak->dak", cotS, coefA)
        bar_A_self = (
            M.T @ (-0.5 * lO - jnp.einsum("kd,dak->ak", cotS, gO)) @ K.T
            + M @ (-0.5 * lO - jnp.einsum("kd,dak->ak", cotS, gO)) @ K
        )
        de_blocks.append(
            jnp.einsum("ae,cae->ec", bar_A_self, gO)
            + jnp.einsum("dae,cdae->ec", bar_gO, H)
            + jnp.einsum("ae,cae->ec", bar_lO, GL)
        )
    out = jnp.concatenate(de_blocks, axis=0)  # up block then dn block
    bcA_u = -0.5 * lOu - jnp.einsum("kd,dak->ak", cot_u, gOu)
    bcA_d = -0.5 * lOd - jnp.einsum("kd,dak->ak", cot_d, gOd)
    bar_s_dn = M.T @ bcA_u.sum(1)
    bar_s_up = M @ bcA_d.sum(1)
    out = out.at[n_up_e:].add(jnp.einsum("a,cae->ec", bar_s_dn, gOd))
    out = out.at[:n_up_e].add(jnp.einsum("a,cae->ec", bar_s_up, gOu))
    return jnp.real(out)  # complex-twist: cot is complex while the Re-projected J3 fields are real -> Re


def grad_T_wrt_J2(
    gd: AGP_data | AGPn_data | SD_data,
    jastrow_data: Jastrow_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    j2 = jastrow_data.jastrow_two_body_data
    p = j2.jastrow_2b_param
    r_up = jnp.asarray(r_up)
    r_dn = jnp.asarray(r_dn)
    r_all = jnp.concatenate([r_up, r_dn], axis=0)
    Ne = len(r_all)
    diff = r_all[:, None, :] - r_all[None, :, :]
    dist, grad_d, gd2, lap_d = _dist_and_derivs(j2.structure_data, diff)
    eye = jnp.eye(Ne, dtype=bool)
    safe = jnp.where(eye, 1.0, dist)
    # gp = f_c'(r), gpp = f_c''(r) of the O_k-field pair function f_c = d u / dp.
    if j2.jastrow_2b_type == "pade":  # u = r/2 (1+pr)^-1 -> f_c = -r^2/(2(1+pr)^2)
        u = 1.0 + p * safe
        gp = -safe / u**3
        gpp = (2.0 * p * safe - 1.0) / u**4
    elif j2.jastrow_2b_type == "exp":  # u = (1-e^{-pr})/(2p) -> f_c' = -(r/2)e^{-pr}
        e = jnp.exp(-p * safe)
        gp = -(safe / 2.0) * e
        gpp = 0.5 * (p * safe - 1.0) * e
    else:
        raise ValueError(j2.jastrow_2b_type)
    gp = jnp.where(eye, 0.0, gp)
    gpp = jnp.where(eye, 0.0, gpp)
    # grad_r f_c = f_c'(d) grad_d ; lap_r f_c = f_c''(d) |grad_d|^2 + f_c'(d) lap_d
    grad_f = jnp.sum(gp[..., None] * grad_d, axis=1)
    lap_f = (gpp * gd2 + gp * lap_d).sum(axis=1)
    gJ_up, gJ_dn, _, _ = compute_grads_and_laplacian_Jastrow_part(jastrow_data, r_up, r_dn)
    gD_up, gD_dn = _kinetic_det_grad_field(gd, r_up, r_dn, geminal_inv=geminal_inv)
    gJ = jnp.concatenate([jnp.asarray(gJ_up), jnp.asarray(gJ_dn)], axis=0)
    gD = jnp.concatenate([jnp.asarray(gD_up), jnp.asarray(gD_dn)], axis=0)
    return -0.5 * lap_f.sum() - jnp.sum((gJ + gD) * grad_f)


def grad_T_wrt_J1(
    gd: AGP_data | AGPn_data | SD_data,
    jastrow_data: Jastrow_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    j1 = jastrow_data.jastrow_one_body_data
    R = jnp.asarray(j1.structure_data.positions)
    Z = jnp.asarray(j1.structure_data.atomic_numbers) - jnp.asarray(j1.core_electrons)
    p = j1.jastrow_1b_param
    c = (2.0 * Z) ** 0.25
    pref = -((2.0 * Z) ** 0.75) if j1.jastrow_1b_type == "exp" else -2.0 * Z
    r_up = jnp.asarray(r_up)
    r_dn = jnp.asarray(r_dn)
    r_all = jnp.concatenate([r_up, r_dn], axis=0)
    diff = r_all[:, None, :] - R[None, :, :]  # (Ne, nA, 3)
    dist, grad_d, gd2, lap_d = _dist_and_derivs(j1.structure_data, diff)
    a = p * c[None, :]
    if j1.jastrow_1b_type == "exp":
        e = jnp.exp(-a * dist)
        dterm_p = -pref[None, :] * (c[None, :] ** 2 / 2.0) * dist * e
        dterm_pp = -pref[None, :] * (c[None, :] ** 2 / 2.0) * e * (1.0 - a * dist)
    elif j1.jastrow_1b_type == "pade":
        A = -pref[None, :] * c[None, :] / 2.0
        u = 1.0 + a * dist
        dterm_p = 2.0 * A * dist / u**3
        dterm_pp = 2.0 * A * (1.0 - 2.0 * a * dist) / u**4
    else:
        raise ValueError(j1.jastrow_1b_type)
    # grad_r f_c = f_c'(d) grad_d ; lap_r f_c = f_c''(d) |grad_d|^2 + f_c'(d) lap_d
    grad_f = jnp.sum(dterm_p[..., None] * grad_d, axis=1)  # (Ne, 3)
    lap_f = (dterm_pp * gd2 + dterm_p * lap_d).sum(axis=1)  # (Ne,)
    gJ_up, gJ_dn, _, _ = compute_grads_and_laplacian_Jastrow_part(jastrow_data, r_up, r_dn)
    gD_up, gD_dn = _kinetic_det_grad_field(gd, r_up, r_dn, geminal_inv=geminal_inv)
    gJ = jnp.concatenate([jnp.asarray(gJ_up), jnp.asarray(gJ_dn)], axis=0)
    gD = jnp.concatenate([jnp.asarray(gD_up), jnp.asarray(gD_dn)], axis=0)
    return -0.5 * lap_f.sum() - jnp.sum((gJ + gD) * grad_f)


def grad_T_wrt_J3(
    gd: AGP_data | AGPn_data | SD_data,
    jastrow_data: Jastrow_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """Return ``dT/d j_matrix`` for the three-body Jastrow.

    J3 is LINEAR in ``j_matrix = [M | j1col]``, so its O_k field equals the J3
    log-derivative blocks (assemble.grad_J3_wrt_jmatrix)::

        f_M[a,b] = (A_up K2 A_up^T + A_dn K2 A_dn^T + outer(s_up,s_dn))[a,b]  (K2=strict upper)
        f_j1[a]  = s_up[a] + s_dn[a]                                          (s = A@1)

    with A = J3 orbital basis (orb_num, N). The same scalar-Jastrow-param reduction
    ``dT/dc = -1/2 sum_k lap(f_c) - sum_k (gJ+gD).grad(f_c)`` then contracts the AO
    grad/Laplacian of these bilinear fields with the kinetic cotangent ``h = -(gJ+gD)``
    and ``-1/2``. Each electron factor in a pair is a distinct electron, so ``lap_k``
    hits only one factor (no ``|grad|^2`` cross term). Uses ``j3_basis_derivs`` (sphe/cart/pgto-aware, real-
    projected) for value/grad/Laplacian, matching ``_j3_side_r`` conventions."""
    from .ao_deriv import j3_basis_derivs

    j3 = jastrow_data.jastrow_three_body_data
    aos = j3.orb_data
    A_up, GA_up, LA_up, _, _ = j3_basis_derivs(aos, r_up)  # (orb,N_up), (3,orb,N_up), (orb,N_up)
    A_dn, GA_dn, LA_dn, _, _ = j3_basis_derivs(aos, r_dn)
    n_up, n_dn = A_up.shape[1], A_dn.shape[1]
    K2u = jnp.triu(jnp.ones((n_up, n_up)), k=1)  # strict upper: i<j
    K2d = jnp.triu(jnp.ones((n_dn, n_dn)), k=1)
    s_up, s_dn = A_up.sum(1), A_dn.sum(1)

    gJ_up, gJ_dn, _, _ = compute_grads_and_laplacian_Jastrow_part(jastrow_data, r_up, r_dn)
    gD_up, gD_dn = _kinetic_det_grad_field(gd, r_up, r_dn, geminal_inv=geminal_inv)
    h_up = -(jnp.asarray(gJ_up) + gD_up)  # cotangent on grad(f), (N_up,3)
    h_dn = -(jnp.asarray(gJ_dn) + gD_dn)

    # per-electron contraction with the orbital gradient: (orb, N)
    hGA_up = jnp.einsum("kd,dak->ak", h_up, GA_up)
    hGA_dn = jnp.einsum("kd,dak->ak", h_dn, GA_dn)

    def block(coeff_up: jax.Array, coeff_dn: jax.Array) -> jax.Array:
        """(orb,orb) M-block from per-electron up/dn coefficients ``coeff`` standing in
        for the grad- or laplacian-contracted orbital factors."""
        uu = coeff_up @ (K2u @ A_up.T) + (A_up @ K2u) @ coeff_up.T
        dd = coeff_dn @ (K2d @ A_dn.T) + (A_dn @ K2d) @ coeff_dn.T
        ud = jnp.outer(coeff_up.sum(1), s_dn) + jnp.outer(s_up, coeff_dn.sum(1))
        return uu + dd + ud

    M_grad = block(hGA_up, hGA_dn) - 0.5 * block(LA_up, LA_dn)
    j1_grad = (hGA_up.sum(1) + hGA_dn.sum(1)) - 0.5 * (LA_up.sum(1) + LA_dn.sum(1))
    return jnp.hstack([M_grad, j1_grad[:, None]])
