"""ECP (pseudopotential) nuclear force d V_ecp / dR -- local + non-local, NN-truncated.

Matches the production forward (NN=1 nearest nucleus per electron, Nv quadrature)::

  V_local  = sum_{e}            sum_t coef_t r_eI^{p_t-2} exp(-exp_t r_eI^2)   (l == l_max)
  V_nonloc = sum_{e,g,l<l_max} V_l(r_eI) (2l+1) P_l(cos g) w_g Psi(r'_g)/Psi(r),  r'_g = R_I + r_eI v_g

The nuclear derivatives are::

  d V_local/dR_I = -sum_e V_local'(r_eI) u_eI            (u_eI=(r_e-R_I)/r_eI)
  d V_nonloc/dR_I splits:
    (a) wf-ratio (orbital/Jastrow centers move): sum_g Vgrid_g O_R(r'_g) - sumV O_R(r)   [O_R = d lnPsi/dR]
    (b1) mesh-motion (r'_g moves with R):        sum_g Vgrid_g (O_e - u_eI (v_g . O_e))   [O = drift d lnPsi/dr]
    (b2) projector/weight geometry:  -sum_{g,l} (2l+1) w_g ratio_g [V_l' P_l u_eI + V_l P_l' (vhat - cos u_eI)/r_eI]

Eager (host-side) evaluation.
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

from collections.abc import Callable
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp

from jqmc.coulomb_potential import (
    Nv_default,
    _Mesh,
    _find_nearest_ecp_nucleus_indices_jnp,
    _legendre_tablated,
    icosahedron_sym_mesh_Nv12,
    octahedron_sym_mesh_Nv6,
    octahedron_sym_mesh_Nv18,
    tetrahedron_sym_mesh_Nv4,
)
from jqmc.determinant import AGP_data, AGPn_data, SD_data, compute_det_geminal_all_elements
from jqmc.jastrow_factor import Jastrow_data, _compute_ratio_Jastrow_part_split_spin, compute_Jastrow_part_ratio_zone
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.structure import _get_min_dist_rel_R_cart_jnp

# Bounded mesh-config streaming primitives live in ``mesh_stream`` (they are
# estimator/potential-agnostic and shared with one_move_mesh / gfmc / assemble; the
# ECP quadrature mesh is only one of their users). The ECP force kernels below
# use only these two.
from .mesh_stream import _mesh_map

_GeminalData: TypeAlias = AGP_data | AGPn_data | SD_data
_ArrayPair: TypeAlias = tuple[jax.Array, jax.Array]
_EcpCommon: TypeAlias = dict[str, object]
_DltMoveTerms: TypeAlias = tuple[
    _EcpCommon,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]


def _legendre_derivative_tablated(l: jax.Array | int, x: jax.Array) -> jax.Array:
    """Tabulated P_l'(x) for l = 0..6, a polynomial valid on the closed interval [-1, 1].

    The recurrence form l (P_{l-1} - x P_l) / (1 - x^2) is 0/0 at x = +-1, which is
    reached whenever an electron lies exactly on a quadrature direction of the current
    mesh rotation (e.g. right after a GFMC T-move / DLT-move onto a mesh point), and it
    loses all digits by cancellation next to it for l >= 2.
    """
    conditions = [l == 0, l == 1, l == 2, l == 3, l == 4, l == 5, l == 6]
    dP = [
        0.0 * x,
        1.0 + 0.0 * x,
        3.0 * x,
        1.0 / 2.0 * (15.0 * x**2 - 3.0),
        1.0 / 2.0 * (35.0 * x**3 - 15.0 * x),
        1.0 / 8.0 * (315.0 * x**4 - 210.0 * x**2 + 15.0),
        1.0 / 8.0 * (693.0 * x**5 - 630.0 * x**3 + 105.0 * x),
    ]
    return jnp.select(conditions, dP, jnp.nan)


def _leg_jnp(l: jax.Array | int, x: jax.Array) -> _ArrayPair:
    """(P_l(x), P_l'(x)) via the jnp tabulated Legendre (n<=6); jit/vmap-safe. dP_0 = 0."""
    Pl = _legendre_tablated(l, x)
    dPl = _legendre_derivative_tablated(l, x)
    return Pl, dPl


def _mesh(Nv: int) -> _Mesh:
    return {
        4: tetrahedron_sym_mesh_Nv4,
        6: octahedron_sym_mesh_Nv6,
        12: icosahedron_sym_mesh_Nv12,
        18: octahedron_sym_mesh_Nv18,
    }[Nv]


def grad_V_ecp_wrt_R(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array | None = None,
    Nv: int | None = None,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """Total ECP nuclear force d V_ecp / dR (n_atom, 3), NN=1. Fully vectorized (jit/vmap-safe).
    Local virial scattered to atoms; non-local = projector geometry + mesh-motion scattered to the
    nearest atom, plus the wf-ratio force d lnPsi/dR (_force_lnPsi at the mesh) over ALL atoms.

    ``geminal_inv`` (fast path): base-position running SM ``G^{-1}`` reused by the base mesh setup +
    the base wf-ratio force (mesh-config re-evals use their own inverse).
    """
    from .assemble import _force_lnPsi

    c = _ecp_nl_common(hamiltonian_data, r_up, r_dn, RT, Nv, geminal_inv=geminal_inv)
    geminal_inv = c["geminal_inv"]  # streamed mesh: the value-only base inverse its fields use
    n_atom, Ne = c["n_atom"], c["Ne"]
    r_up = jnp.asarray(r_up, jnp.float64)
    r_dn = jnp.asarray(r_dn, jnp.float64)
    Nv = c["ratio"].shape[1]
    # wf-ratio force d lnPsi/dR (n_atom,3) at base + each mesh config (holomorphic; Re at the END).
    OR_base = _force_lnPsi(hamiltonian_data, r_up, r_dn, take_real=False, geminal_inv=geminal_inv)
    Vgrid = c["Vgrid"]
    Vg_sg = jnp.sum(Vgrid, axis=1)  # (Ne,Nv)
    SV = jnp.sum(Vgrid)
    if c["fast"]:  # Rank-one force field contracted with the config weights (issue #596).
        from . import one_move_mesh

        OR_w = one_move_mesh.mesh_force_weighted(hamiltonian_data, c["mesh_state"], Vg_sg, geminal_inv=geminal_inv)
    else:
        from . import one_move_mesh

        # Per-config displaced geminal inverse via EXACT rank-1 Sherman-Morrison off the base (threaded)
        # inverse -- ECP move = rank-1 row/column update of G, no fresh SVD-pinv per config.  Threading it
        # into the streamed wf-ratio Pulay force ``_force_lnPsi`` (-> grad_lndet_wrt_R) keeps the streamed
        # complex-twist e_L/R path free of ``lapack_[dz]gesdd``.
        ginv_cfg = one_move_mesh.sm_geminal_inv_configs(c["gd"], r_up, r_dn, c["RU"], c["RD"], geminal_inv=geminal_inv)
        OR_mesh = _mesh_map(
            lambda ru, rd, gi: _force_lnPsi(hamiltonian_data, ru, rd, take_real=False, geminal_inv=gi),
            (c["RU"], c["RD"], ginv_cfg),
        ).reshape(Ne, Nv, n_atom, 3)
        OR_w = jnp.einsum("eg,egac->ac", Vg_sg, OR_mesh)

    # local: virial -Vloc'(r) u per (electron, local-term), scattered to the term's atom.
    loc_per_t = -jnp.einsum("et,etc->tc", c["Vlocp"], c["u_loc"])  # (Tloc,3)
    dVdR_loc = jax.ops.segment_sum(loc_per_t, c["nuc_loc"], num_segments=n_atom)
    # non-local projector geometry + mesh-motion -> scattered to the nearest atom of each electron.
    proj_e = -(
        jnp.einsum("etg,et,etg->e", c["pref"], c["Vlp_t"], c["Pl"])[:, None] * c["u_e"]
        + jnp.einsum("etg,et,etg,egc->ec", c["pref"], c["Vl_t"], c["dPl"], c["dcth"])
    )
    mesh_e = jnp.einsum("eg,egc->ec", Vg_sg, c["Oee"]) - jnp.einsum("eg,eg,ec->ec", Vg_sg, c["vg_Oee"], c["u_e"])
    dVdR_nl = jax.ops.segment_sum(proj_e + mesh_e, c["i_atom_e"], num_segments=n_atom)
    # wf-ratio: sum_g Vgrid O_R(mesh) - sumV O_R(base), over ALL atoms.
    term_wf = OR_w - SV * OR_base
    return jnp.real(dVdR_loc + dVdR_nl + term_wf)


def _O_drift(
    gd: _GeminalData,
    jd: Jastrow_data,
    ru: jax.Array,
    rd: jax.Array,
    take_real: bool = True,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """Full electron-gradient field O = grad lnPsi = grad ln det + grad J. take_real -> Re (the
    drift); False keeps the holomorphic complex form for the ECP wf-ratio (Re on the product).

    ``geminal_inv`` (fast path): base-position running SM ``G^{-1}`` -- valid ONLY when ``ru``/``rd``
    are the base walker coordinates (the mesh-config re-evals pass None; different G there).
    """
    from jqmc.jastrow_factor import compute_grads_and_laplacian_Jastrow_part

    from .kinetic import _det_grad_field

    gu, gdn = _det_grad_field(gd, ru, rd, geminal_inv=geminal_inv)
    O = jnp.concatenate([gu, gdn], 0)
    if any(
        getattr(jd, f) is not None
        for f in ("jastrow_one_body_data", "jastrow_two_body_data", "jastrow_three_body_data", "jastrow_dtn_data")
    ):
        gju, gjd, _, _ = compute_grads_and_laplacian_Jastrow_part(jd, ru, rd)
        O = O + jnp.concatenate([gju, gjd], 0)
    return jnp.real(O) if take_real else O


def _ecp_nl_common(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array | None,
    Nv: int | None,
    geminal_inv: jax.Array | None = None,
) -> _EcpCommon:
    """Shared vectorized ECP non-local setup for the R/r/strain derivatives: nearest-nucleus mesh
    batch + ratios + drift field + radial/angular tables. Returns a dict of jnp arrays. All wf
    re-evaluations are `vmap`'d over the (electron x grid) mesh-config batch (one-hot scatter), so
    the whole thing composes inside `jit(vmap(...))`. Ratios/O are holomorphic (Re applied by the
    caller at the END of the assembled product -- complex-Bloch twist).

    ``c["geminal_inv"]`` is the base inverse the mesh fields were built with.  On the rank-one
    route it is the argument unchanged.  On the streamed route without a supplied inverse it is
    the value-only ratio-table inverse (``one_move_mesh.ratio_table_geminal_inv``).  Callers pass
    it to the base fields they difference against the mesh (``O``, ``dlnPsi/dR``, ``dlnPsi/deps``).
    """
    cp = hamiltonian_data.coulomb_potential_data
    wf = hamiltonian_data.wavefunction_data
    gd, jd, sd = wf.geminal_data, wf.jastrow_data, cp.structure_data
    Nv = Nv_default if Nv is None else Nv
    RT = jnp.eye(3) if RT is None else jnp.asarray(RT, jnp.float64)
    msh = _mesh(Nv)
    weights = jnp.asarray(msh.weights, jnp.float64)  # (Nv,)
    grid = jnp.asarray(msh.grid_points, jnp.float64) @ RT  # (Nv,3)
    vhat = grid / jnp.linalg.norm(grid, axis=1)[:, None]  # (Nv,3)
    r_up = jnp.asarray(r_up, jnp.float64)
    r_dn = jnp.asarray(r_dn, jnp.float64)
    n_up, n_dn = r_up.shape[0], r_dn.shape[0]
    Ne = n_up + n_dn
    n_atom = sd.natom
    all_e = jnp.concatenate([r_up, r_dn], 0)  # (Ne,3)
    eidx = jnp.arange(Ne)

    det_den = compute_det_geminal_all_elements(gd, r_up, r_dn)

    def _rel_ei(rc: jax.Array, i: jax.Array) -> jax.Array:
        return _get_min_dist_rel_R_cart_jnp(structure_data=sd, r_cart=rc, i_atom=i, dtype=jnp.float64)

    rels_all = jax.vmap(lambda rc: jax.vmap(lambda i: _rel_ei(rc, i))(jnp.arange(n_atom)))(all_e)  # (Ne,n_atom,3)

    # local terms (flat; atom = nuc_loc[t])
    nuc_loc = jnp.asarray(cp._nucleus_index_local_part)  # (Tloc,)
    exp_loc = jnp.asarray(cp._exponents_local_part, jnp.float64)
    coef_loc = jnp.asarray(cp._coefficients_local_part, jnp.float64)
    pow_loc = jnp.asarray(cp._powers_local_part, jnp.float64)
    rel_loc = rels_all[:, nuc_loc, :]  # (Ne,Tloc,3)
    r_loc = jnp.linalg.norm(rel_loc, axis=-1)  # (Ne,Tloc)
    u_loc = -rel_loc / r_loc[..., None]
    Vlocp = (
        coef_loc
        * ((pow_loc - 2.0) * r_loc ** (pow_loc - 3.0) - 2.0 * exp_loc * r_loc ** (pow_loc - 1.0))
        * jnp.exp(-exp_loc * r_loc**2)
    )  # (Ne,Tloc)

    # non-local: NN=1 nearest ECP nucleus (one carrying a non-local channel) per electron + mesh-config batch
    i_atom_e = jax.vmap(
        lambda electron_position: _find_nearest_ecp_nucleus_indices_jnp(
            coulomb_potential_data=cp,
            r_cart=electron_position,
            NN=1,
        )[0]
    )(all_e)  # (Ne,)
    rel_e = jax.vmap(_rel_ei)(all_e, i_atom_e)  # (Ne,3)
    r_eI = jnp.linalg.norm(rel_e, axis=-1)  # (Ne,)
    u_e = -rel_e / r_eI[:, None]  # (Ne,3)
    # Match coulomb_potential._build_mesh_for_spin's production coordinate
    # construction exactly: r + (rel + |rel| grid).  The algebraically
    # equivalent (r + rel) + |rel| grid differs at the final fp64 bit; radial pGTO parameter derivatives can amplify that coordinate
    # discrepancy near cancellations even though values remain unchanged.
    rprime = all_e[:, None, :] + (rel_e[:, None, :] + r_eI[:, None, None] * grid[None, :, :])  # (Ne,Nv,3)
    up_oh = jax.nn.one_hot(jnp.where(eidx < n_up, eidx, n_up), n_up)  # (Ne,n_up); dn-electron rows = 0
    dn_oh = jax.nn.one_hot(jnp.where(eidx >= n_up, eidx - n_up, n_dn), n_dn)  # (Ne,n_dn); up-electron rows = 0
    ru_mesh = r_up[None, None, :, :] + up_oh[:, None, :, None] * (rprime[:, :, None, :] - r_up[None, None, :, :])
    rd_mesh = r_dn[None, None, :, :] + dn_oh[:, None, :, None] * (rprime[:, :, None, :] - r_dn[None, None, :, :])
    # Explicit leading dim ``Ne * Nv`` (NOT ``-1``): a fully spin-polarised channel (e.g. H atom,
    # ``n_dn = 0``) makes ``prod(n_dn, 3) = 0``, so ``reshape(-1, n_dn, 3)`` raises
    # ``ZeroDivisionError`` in JAX's ``-1`` inference. ``Ne * Nv`` is identical for non-empty channels.
    RU = ru_mesh.reshape(Ne * Nv, n_up, 3)  # (Ne*Nv, n_up, 3)
    RD = rd_mesh.reshape(Ne * Nv, n_dn, 3)

    # Molecular-real rank-one fast path: ratio + drift field via the
    # base G^{-1} Sherman-Morrison update and ONE-POINT orbital evals at the displaced
    # positions -- no per-config all-electron rebuild.  Unsupported cases (PBC / complex
    # twist / DTN) keep the streamed full-recompute (bounded by _mesh_map, not yet fast).
    from . import one_move_mesh

    _fast = one_move_mesh.fast_supported(hamiltonian_data)
    if _fast:
        # Weighted-sum state (issue #596): consumers take sum_k W_k O(config_k) through
        # _weighted_drift / _weighted_force instead of a (Ne, Nv, Ne, 3) field.
        ratio, Oee, _ms = one_move_mesh.mesh_ratio_state(hamiltonian_data, r_up, r_dn, rprime, geminal_inv=geminal_inv)
        O_mesh = None
        _fb, _fd = _ms["base"], _ms["dm"]
        jas_ratio = None  # fast path: the Jastrow factor is exp(mesh_state["dJ"])
    else:
        # One base inverse for the streamed mesh and every base field its
        # callers difference against it: the value-only table of ``det_den`` /
        # ``det_xp`` (``one_move_mesh.ratio_table_geminal_inv``).  A supplied
        # running inverse is unchanged.
        geminal_inv = one_move_mesh.ratio_table_geminal_inv(gd, r_up, r_dn, geminal_inv)
        det_xp = _mesh_map(
            lambda ru, rd: compute_det_geminal_all_elements(gd, ru, rd),
            (RU, RD),
        )
        # Jastrow ratio exp(J(R') - J(R)) in the fp64 jastrow_ratio zone (issue #765): the same
        # one-move split-spin kernel as the production ECP e_L ratio (``coulomb_potential``), whose
        # up block is the first ``n_up * Nv`` configs (one up electron moved) and dn block the rest.
        # A difference of two whole ``compute_Jastrow_part`` values would round in jastrow_eval
        # (fp32 in mixed).
        jas_ratio = _compute_ratio_Jastrow_part_split_spin(
            jastrow_data=jd,
            old_r_up_carts=r_up,
            old_r_dn_carts=r_dn,
            new_r_up_shifted=RU[: n_up * Nv],
            new_r_dn_shifted=RD[n_up * Nv :],
        ).reshape(Ne, Nv)
        ratio = (det_xp / det_den).reshape(Ne, Nv) * jas_ratio  # (Ne,Nv)
        # Per-config displaced geminal inverse via EXACT rank-1 Sherman-Morrison off the base (threaded)
        # inverse -- the ECP move is a rank-1 row/column update of G, so no fresh SVD-pinv per config.
        # Threading it into the streamed drift ``_O_drift`` (-> _det_grad_field) keeps the streamed
        # (complex-twist / non-(pade,exp)) e_L/r path free of ``lapack_[dz]gesdd``.
        ginv_cfg = one_move_mesh.sm_geminal_inv_configs(gd, r_up, r_dn, RU, RD, geminal_inv=geminal_inv)
        O_mesh = _mesh_map(
            lambda ru, rd, gi: _O_drift(gd, jd, ru, rd, take_real=False, geminal_inv=gi),
            (RU, RD, ginv_cfg),
        ).reshape(Ne, Nv, Ne, 3)
        _fb = _fd = _ms = None
        Oee = O_mesh[eidx, :, eidx, :]  # (Ne,Nv,3): the moved electron's own drift

    cth = jnp.einsum("ec,gc->eg", u_e, vhat)  # (Ne,Nv)
    dcth = (vhat[None, :, :] - cth[:, :, None] * u_e[:, None, :]) / r_eI[:, None, None]  # (Ne,Nv,3)

    ang_all, exp_all, coef_all, pow_all = cp._padded_parameters_tuple  # (n_atom,T) non-local, coef=0 pads
    ang_e = jnp.asarray(ang_all)[i_atom_e]
    exp_e = jnp.asarray(exp_all, jnp.float64)[i_atom_e]
    coef_e = jnp.asarray(coef_all, jnp.float64)[i_atom_e]
    pow_e = jnp.asarray(pow_all, jnp.float64)[i_atom_e]
    r = r_eI[:, None]  # (Ne,1)
    Vl_t = r**-2.0 * coef_e * r**pow_e * jnp.exp(-exp_e * r**2)  # (Ne,T)
    Vlp_t = coef_e * ((pow_e - 2.0) * r ** (pow_e - 3.0) - 2.0 * exp_e * r ** (pow_e - 1.0)) * jnp.exp(-exp_e * r**2)
    twol1 = 2.0 * ang_e + 1.0  # (Ne,T)
    Pl, dPl = jax.vmap(lambda lr, xr: jax.vmap(lambda l: jax.vmap(lambda x: _leg_jnp(l, x))(xr))(lr))(ang_e, cth)  # (Ne,T,Nv)
    Vgrid_nr = (Vl_t * twol1)[:, :, None] * Pl * weights[None, None, :]  # (Ne,T,Nv) angular, NO ratio
    Vgrid = Vgrid_nr * ratio[:, None, :]  # (Ne,T,Nv)
    vg_Oee = jnp.einsum("gc,egc->eg", grid, Oee)
    pref = twol1[:, :, None] * weights[None, None, :] * ratio[:, None, :]  # (Ne,T,Nv)
    return {
        "gd": gd, "jd": jd, "sd": sd, "RU": RU, "RD": RD, "n_atom": n_atom, "Ne": Ne, "eidx": eidx,
        "grid": grid, "weights": weights, "ratio": ratio, "rel_loc": rel_loc, "r_loc": r_loc,
        "u_loc": u_loc, "Vlocp": Vlocp, "nuc_loc": nuc_loc, "i_atom_e": i_atom_e, "rel_e": rel_e,
        "r_eI": r_eI, "u_e": u_e, "cth": cth, "dcth": dcth, "Vl_t": Vl_t, "Vlp_t": Vlp_t, "twol1": twol1,
        "Pl": Pl, "dPl": dPl, "Vgrid": Vgrid, "Vgrid_nr": Vgrid_nr, "O_mesh": O_mesh, "Oee": Oee,
        "vg_Oee": vg_Oee, "pref": pref, "rprime": rprime, "fast": _fast, "fast_base": _fb, "fast_dm": _fd,
        "mesh_state": _ms, "jas_ratio": jas_ratio, "geminal_inv": geminal_inv,
    }  # fmt: skip


def _weighted_drift(hamiltonian_data: Hamiltonian_data, c: _EcpCommon, W: jax.Array) -> jax.Array:
    """``sum_{e,g} W[e,g] O(config_{e,g})`` (Ne, 3) of the :func:`_ecp_nl_common` mesh: the
    rank-one weighted sum on the fast path (issue #596), else the streamed field."""
    if c["fast"]:
        from . import one_move_mesh

        return one_move_mesh.mesh_drift_weighted(hamiltonian_data, c["mesh_state"], W)
    return jnp.einsum("eg,egmc->mc", W, c["O_mesh"])


def grad_V_ecp_wrt_r(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array | None = None,
    Nv: int | None = None,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """D V_ecp / d r per electron (Ne, 3) -- local + non-local, NN=1, min-image. Per ECP term
    (electron e, atom I, l, grid g): explicit projector/weight geometry (e only), mesh-numerator via
    the drift field O=grad lnPsi at the mesh config (i!=e direct; i==e the Jacobian u (v_g.O[e])),
    and the denominator -sumV O(r). Fully vectorized (jit/vmap-safe).

    ``geminal_inv`` (fast path): base-position running SM ``G^{-1}`` reused by the base mesh setup +
    the base denominator drift (the mesh-config re-evals use their own inverse).
    """
    c = _ecp_nl_common(hamiltonian_data, r_up, r_dn, RT, Nv, geminal_inv=geminal_inv)
    geminal_inv = c["geminal_inv"]  # streamed mesh: the value-only base inverse its fields use
    O_base = _O_drift(
        c["gd"],
        c["jd"],
        jnp.asarray(r_up, jnp.float64),
        jnp.asarray(r_dn, jnp.float64),
        take_real=False,
        geminal_inv=geminal_inv,
    )
    Vgrid, Vg_sg = c["Vgrid"], jnp.sum(c["Vgrid"], axis=1)
    SV = jnp.sum(Vgrid)
    dV_loc = jnp.einsum("et,etc->ec", c["Vlocp"], c["u_loc"])  # local virial per electron
    term_all = _weighted_drift(hamiltonian_data, c, Vg_sg)  # mesh-numerator (all electrons)
    term_den = -SV * O_base  # denominator
    term_self = jnp.einsum("eg,eg,ec->ec", Vg_sg, c["vg_Oee"], c["u_e"]) - jnp.einsum("eg,egc->ec", Vg_sg, c["Oee"])
    proj = jnp.einsum("etg,et,etg->e", c["pref"], c["Vlp_t"], c["Pl"])[:, None] * c["u_e"] + jnp.einsum(
        "etg,et,etg,egc->ec", c["pref"], c["Vl_t"], c["dPl"], c["dcth"]
    )
    return jnp.real(dV_loc + term_all + term_den + term_self + proj)


def _lndet_drift(
    gd: _GeminalData,
    ru: jax.Array,
    rd: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """det-only per-electron drift ``d ln det / d r`` concatenated ((Ne, 3), HOLOMORPHIC).

    Kept complex on purpose.  These fields are consumed ONLY by the ``dltmove`` delta assembly,
    which multiplies them by the complex determinant ratio; ``Re`` belongs at the END of that
    product because ``Re(A)*Re(B) != Re(A*B)``.  Re-projecting here would drop the Im.Im
    cross term, which is invisible at Gamma (everything real there) -- see the ``_ecp_nl_common``
    contract: "Ratios/O are holomorphic (Re applied by the caller at the END)".
    """
    from .kinetic import _det_grad_field

    gu, gdn = _det_grad_field(gd, ru, rd, geminal_inv=geminal_inv)
    return jnp.concatenate([jnp.asarray(gu), jnp.asarray(gdn)], 0)


def _force_lndet(
    gd: _GeminalData,
    ru: jax.Array,
    rd: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """det-only Pulay ``d ln det / dR`` ((n_atom, 3), HOLOMORPHIC); the P1 partner of
    ``_force_lnPsi``.  Complex for the same reason as :func:`_lndet_drift` -- ``Re`` is taken by
    the caller once the field has been multiplied by the complex ratio."""
    from .assemble import grad_lndet_wrt_R

    return jnp.asarray(grad_lndet_wrt_R(gd, ru, rd, take_real=False, geminal_inv=geminal_inv))


def _stress_lndet_mol(
    gd: _GeminalData,
    ru: jax.Array,
    rd: jax.Array,
    R: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """det-only MOLECULAR affine-strain virial ``d ln det / d eps`` ((3, 3), HOLOMORPHIC): the
    determinant Pulay stress with NO Jastrow -- the P1 (det-only) partner of ``_stress_lnPsi``.
    Complex because its two input fields are (see :func:`_lndet_drift`); the molecular case is
    real by construction (no twist), so this is a dtype widening there, not a value change.
    """
    drift = _lndet_drift(gd, ru, rd, geminal_inv=geminal_inv)
    dR = _force_lndet(gd, ru, rd, geminal_inv=geminal_inv)
    r_all = jnp.concatenate([jnp.asarray(ru), jnp.asarray(rd)], 0)
    return jnp.einsum("ec,ed->cd", drift, r_all) + jnp.einsum("ic,id->cd", dR, jnp.asarray(R))


def _whole_value_jastrow_ratio(c: _EcpCommon, ru0: jax.Array, rd0: jax.Array) -> jax.Array:
    """``exp(J(R'_g) - J(R))`` (Ne, Nv) from two whole Jastrow values per candidate.

    Kept for the differentiable verification value :func:`_dltmove_delta` (the DTN one-move mesh
    kernel is not autodiff-safe).  Both whole values come from
    :func:`~jqmc.jastrow_factor.compute_Jastrow_part_ratio_zone`, i.e. the ``jastrow_ratio`` zone
    that owns the ratio, not ``jastrow_eval`` (fp32 in mixed), so this factor matches the production
    routes (``c["mesh_state"]["dJ"]`` on the rank-one route, ``c["jas_ratio"]`` on the streamed
    route; issues #720 / #765 / #781).
    """
    jd = c["jd"]
    J_den = compute_Jastrow_part_ratio_zone(jd, ru0, rd0)
    J_xp = _mesh_map(
        lambda ru, rd: compute_Jastrow_part_ratio_zone(jd, ru, rd),
        (c["RU"], c["RD"]),
    )
    return jnp.real(jnp.exp(J_xp - J_den).reshape(c["Ne"], c["ratio"].shape[1]))


def _dltmove_delta_terms(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array | None,
    Nv: int | None,
) -> _DltMoveTerms:
    """Shared per-candidate scalars for the ``dltmove`` ECP non-local correction (MOLECULAR): the
    det-only / full ratios, the angular sum, the SP mask, and the assembly weights.  Returns
    ``(c, det_ratio, jr, ang, m, W1, W2, rat_ang)`` where ``c`` is the :func:`_ecp_nl_common` dict.

    COMPLEX-TWIST CONVENTION (fixed phase).  Two different projections live here and they must not
    be confused:

    * the AMPLITUDE the walk clamps is ``Re(det_xp/det_den)`` -- ``Re`` at the SOURCE, per candidate
      (``_compute_V_elements_t`` / ``nn_ecp(project_real=True)``).  ``det_ratio`` therefore stays
      REAL and feeds the SP mask ``m`` and the (real) angular-geometry weight ``rat_ang``;
    * the WF-RATIO DERIVATIVE chain ``d ln ratio/dX`` is holomorphic, so the assembly weights
      ``W1``/``W2`` keep the COMPLEX ``det_ratio`` and are contracted against the complex drift /
      Pulay fields, with ``Re`` taken once at the end of ``grad_V_ecp_nl_dltmove``.

    Using the Re'd ``det_ratio`` in ``W1``/``W2`` computes ``Re(A)*Re(B)`` where the derivative is
    ``Re(A*B)``; the dropped Im.Im cross term is exactly zero at Gamma, so only a twist shows it.
    """
    c = _ecp_nl_common(hamiltonian_data, r_up, r_dn, RT, Nv)
    gd = c["gd"]
    Ne, Nv = c["Ne"], c["ratio"].shape[1]
    ru0 = jnp.asarray(r_up, jnp.float64)
    rd0 = jnp.asarray(r_dn, jnp.float64)
    det_den = compute_det_geminal_all_elements(gd, ru0, rd0)
    det_xp = _mesh_map(
        lambda ru, rd: compute_det_geminal_all_elements(gd, ru, rd),
        (c["RU"], c["RD"]),
    )
    det_ratio_c = (det_xp / det_den).reshape(Ne, Nv)  # HOLOMORPHIC (derivative chain)
    det_ratio = jnp.real(det_ratio_c)  # fixed-phase amplitude (mask + real angular geometry)
    if c["fast"]:
        # Jastrow ratio J(R') - J(R) in the fp64 jastrow_ratio zone: the rank-one mesh state's
        # moved-electron log-ratio, i.e. the SAME Jastrow factor as the tmove ratio
        # ``c["ratio"] = det_ratio * exp(dJ)`` whose SP terms P2 removes.  A difference of two
        # whole ``compute_Jastrow_part`` values would round in the jastrow_eval zone (fp32 in
        # mixed), issue #720.
        jr = jnp.real(jnp.exp(c["mesh_state"]["dJ"]).reshape(Ne, Nv))
    else:
        # Streamed (non-rank-one) route: the same fp64 jastrow_ratio factor as its ``c["ratio"]``
        # (the one-move split-spin ratio kernel, issue #765).
        jr = jnp.real(c["jas_ratio"])
    ang = jnp.einsum("etg->eg", c["Vgrid_nr"])  # angular sum (no ratio)
    v_det = ang * det_ratio
    m = (v_det >= 0.0).astype(jnp.float64)  # SP mask (max(.,0) support), on the Re'd amplitude
    W1 = m * ang * det_ratio_c  # = m ang det_ratio   (P1: det-only, SP-masked, COMPLEX)
    W2 = W1 * jr  # = m ang det_ratio jr  (P2: full, SP-masked, COMPLEX)
    rat_ang = m * det_ratio * (1.0 - jr)  # angular-geometry ratio weight (= ratio_P1 - ratio_P2)
    return c, det_ratio, jr, ang, m, W1, W2, rat_ang


def _dltmove_delta(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array | None = None,
    Nv: int | None = None,
) -> jax.Array:
    """Scalar ``dltmove``-vs-``tmove`` ECP non-local correction (MOLECULAR):

        delta = sum_g max(V^det_g, 0) (1 - jr_g),  V^det_g = ang_g det_ratio_g,  jr_g = exp(J_mesh - J_ref)

    Here ``jr`` is formed from whole Jastrow values on every route (:func:`_whole_value_jastrow_ratio`),
    the definition this verification value is differentiated with (autodiff oracles in the tests, also
    through DTN, whose one-move mesh kernel is not autodiff-safe).  Both whole values are evaluated in
    the fp64 ``jastrow_ratio`` zone (issue #781), like the mesh log-ratio of the production gradient
    :func:`grad_V_ecp_nl_dltmove` (issue #720).

    The driver's ``dltmove`` fixed-node-clamps the DETERMINANT-only amplitude ``V^det`` and re-weights
    ONLY the accepted (FN, ``V^det < 0``) hops by the Jastrow ratio, while the localised (SP,
    ``V^det >= 0``) part goes to the diagonal WITHOUT the Jastrow ratio.  The full ``dltmove`` ``e_L``
    equals the ``tmove`` ``e_L`` plus this ``delta`` (see gfmc.grad_e_L_wrt_positions).  Provided as
    a closed-form value for verification / debugging; the production path uses the closed-form
    gradient :func:`grad_V_ecp_nl_dltmove`.
    """
    c, det_ratio, _, ang, _, _, _, _ = _dltmove_delta_terms(hamiltonian_data, r_up, r_dn, RT, Nv)
    jr = _whole_value_jastrow_ratio(c, jnp.asarray(r_up, jnp.float64), jnp.asarray(r_dn, jnp.float64))
    v_det = ang * det_ratio
    return jnp.sum(jnp.maximum(v_det, 0.0) * (1.0 - jr))


def grad_V_ecp_nl_dltmove(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array | None = None,
    Nv: int | None = None,
    *,
    wrt: Literal["r", "R", "eps"],
) -> jax.Array:
    """Closed-form derivative of the MOLECULAR ``dltmove`` ECP non-local correction ``delta`` (see
    :func:`_dltmove_delta`) w.r.t. ``wrt in {"r", "R", "eps"}``.

    ``delta = P1 - P2`` with ``P1 = sum_{SP} ang det_ratio`` (det-only ratio, SP-masked) and
    ``P2 = sum_{SP} ang det_ratio jr`` (full ratio, SP-masked), ``SP = {V^det >= 0}``.  Each is an
    ECP-non-local-like sum, so this reuses the verified ``tmove`` non-local force/stress assembly
    (``grad_V_ecp_wrt_{r,R,strain}``) with (a) the SP mask on the per-candidate ratio and (b) the O
    field / Pulay-stress matched to the ratio's Jastrow inclusion (det-only for P1, full for P2).
    The SP-mask step function is measure-zero-differentiable (matching ``jnp.maximum``'s subgradient);
    the angular geometry (Jastrow-independent) is weighted by ``ratio_P1 - ratio_P2 = det_ratio(1-jr)``.
    Returns the SAME axes as ``grad_V_ecp_wrt_{r,R,strain}`` -- (Ne,3)/(n_atom,3)/(3,3).
    """
    from .assemble import _force_lnPsi, _stress_lnPsi

    c, det_ratio, jr, ang, m, W1, W2, rat_ang = _dltmove_delta_terms(hamiltonian_data, r_up, r_dn, RT, Nv)
    gd, jd = c["gd"], c["jd"]
    Ne, n_atom = c["Ne"], c["n_atom"]
    Nv = c["ratio"].shape[1]
    RU, RD = c["RU"], c["RD"]
    u_e, i_atom_e = c["u_e"], c["i_atom_e"]
    grid, weights = c["grid"], c["weights"]
    Vl_t, Vlp_t, Pl, dPl, dcth, twol1 = c["Vl_t"], c["Vlp_t"], c["Pl"], c["dPl"], c["dcth"], c["twol1"]
    ru0 = jnp.asarray(r_up, jnp.float64)
    rd0 = jnp.asarray(r_dn, jnp.float64)
    # Streamed route: the full drift ``c["Oee"]`` rides the value-only per-config inverses of
    # ``det_ratio``'s table, so every streamed det-only / full field and its base use the same
    # inverses (``one_move_mesh.ratio_table_geminal_inv``).  The rank-one route keeps its own
    # no-inverse fields (``base_inv`` stays the caller's, ``None``).
    base_inv = c["geminal_inv"]
    if c["fast"]:

        def _per_config(field: Callable[[jax.Array, jax.Array, jax.Array | None], jax.Array]) -> jax.Array:
            return _mesh_map(lambda ru, rd: field(ru, rd, None), (RU, RD))

    else:
        from . import one_move_mesh

        ginv_cfg = one_move_mesh.sm_geminal_inv_configs(gd, ru0, rd0, RU, RD, geminal_inv=base_inv)

        def _per_config(field: Callable[[jax.Array, jax.Array, jax.Array | None], jax.Array]) -> jax.Array:
            return _mesh_map(field, (RU, RD, ginv_cfg))

    # det-only + full drift O-fields, at each mesh config + base.  The delta assembly differences
    # them (``W1 O_det - W2 O_full`` with ``W2 = W1 jr``), so the det-only drift must come from the
    # SAME route as the determinant part of the full drift: on the rank-one fast path that is the
    # Sherman-Morrison determinant drift (``one_move_mesh.mesh_drift_weighted(det_only=True)``, the
    # weighted ``det_drift_mesh``) that the full weighted drift adds the Jastrow drift to.
    # A separate full recompute rounds differently (fp32 AOs in mixed precision); near a node,
    # where the drift is large, the two roundings no longer cancel and the force loses translation
    # invariance.
    if c["fast"]:
        from . import one_move_mesh

        ms = c["mesh_state"]
        Oee_det = ms["oee_det"].reshape(Ne, Nv, 3)

        def _wdrift_det(W: jax.Array) -> jax.Array:
            return one_move_mesh.mesh_drift_weighted(hamiltonian_data, ms, W, det_only=True)

    else:
        O_det_mesh = _per_config(lambda ru, rd, gi: _lndet_drift(gd, ru, rd, geminal_inv=gi)).reshape(Ne, Nv, Ne, 3)
        eidx = jnp.arange(Ne)
        Oee_det = O_det_mesh[eidx, :, eidx, :]  # (Ne,Nv,3)

        def _wdrift_det(W: jax.Array) -> jax.Array:
            return jnp.einsum("eg,egmc->mc", W, O_det_mesh)

    O_det_base = _lndet_drift(gd, ru0, rd0, geminal_inv=base_inv)
    # HOLOMORPHIC base drift: the full mesh drift (``_weighted_drift``) is holomorphic by the
    # `_ecp_nl_common` contract, so the base it is differenced against must be too --
    # a Re'd base against a complex mesh is not even a consistent difference.
    O_full_base = _O_drift(gd, jd, ru0, rd0, take_real=False, geminal_inv=base_inv)
    Oee_full = c["Oee"]
    vg_det = jnp.einsum("gc,egc->eg", grid, Oee_det)
    vg_full = c["vg_Oee"]

    pref_ang = twol1[:, :, None] * weights[None, None, :] * rat_ang[:, None, :]  # (Ne,T,Nv)

    def _ratio_drift(
        W: jax.Array,
        term_all: jax.Array,
        O_base: jax.Array,
        Oee: jax.Array,
        vg: jax.Array,
    ) -> jax.Array:
        SV = jnp.sum(W)
        term_den = -SV * O_base
        term_self = jnp.einsum("eg,eg,ec->ec", W, vg, u_e) - jnp.einsum("eg,egc->ec", W, Oee)
        return term_all + term_den + term_self

    if wrt == "r":
        proj = jnp.einsum("etg,et,etg->e", pref_ang, Vlp_t, Pl)[:, None] * u_e + jnp.einsum(
            "etg,et,etg,egc->ec", pref_ang, Vl_t, dPl, dcth
        )
        dr = (
            _ratio_drift(W1, _wdrift_det(W1), O_det_base, Oee_det, vg_det)
            - _ratio_drift(W2, _weighted_drift(hamiltonian_data, c, W2), O_full_base, Oee_full, vg_full)
            + proj
        )
        return jnp.real(dr)

    if wrt == "R":
        OR_det_mesh = _per_config(lambda ru, rd, gi: _force_lndet(gd, ru, rd, geminal_inv=gi)).reshape(Ne, Nv, n_atom, 3)
        # take_real=False: holomorphic Pulay, matching `_force_lndet` -- `Re` is taken once on
        # the assembled product below (`jnp.real(dVdR_nl + term_wf)`).
        OR_full_mesh = _per_config(
            lambda ru, rd, gi: _force_lnPsi(hamiltonian_data, ru, rd, take_real=False, geminal_inv=gi)
        ).reshape(Ne, Nv, n_atom, 3)
        OR_det_base = _force_lndet(gd, ru0, rd0, geminal_inv=base_inv)
        OR_full_base = jnp.asarray(_force_lnPsi(hamiltonian_data, r_up, r_dn, take_real=False, geminal_inv=base_inv))
        proj_e = -(
            jnp.einsum("etg,et,etg->e", pref_ang, Vlp_t, Pl)[:, None] * u_e
            + jnp.einsum("etg,et,etg,egc->ec", pref_ang, Vl_t, dPl, dcth)
        )

        def _mesh_e(W: jax.Array, Oee: jax.Array, vg: jax.Array) -> jax.Array:
            return jnp.einsum("eg,egc->ec", W, Oee) - jnp.einsum("eg,eg,ec->ec", W, vg, u_e)

        def _wf_R(W: jax.Array, OR_mesh: jax.Array, OR_base: jax.Array) -> jax.Array:
            return jnp.einsum("eg,egac->ac", W, OR_mesh) - jnp.sum(W) * OR_base

        mesh_e = _mesh_e(W1, Oee_det, vg_det) - _mesh_e(W2, Oee_full, vg_full)
        dVdR_nl = jax.ops.segment_sum(proj_e + mesh_e, i_atom_e, num_segments=n_atom)
        term_wf = _wf_R(W1, OR_det_mesh, OR_det_base) - _wf_R(W2, OR_full_mesh, OR_full_base)
        return jnp.real(dVdR_nl + term_wf)

    if wrt == "eps":
        R = jnp.asarray(hamiltonian_data.structure_data.positions, jnp.float64)
        rel_e, r_eI = c["rel_e"], c["r_eI"]
        I3 = jnp.eye(3)
        # geometry strain (rel -> F rel, quadrature directions v_g fixed; mesh radius r_eI = |F rel|).
        drel = jnp.einsum("ea,eb->eab", rel_e, rel_e) / r_eI[:, None, None]
        du = (
            -jnp.einsum("ca,eb->ecab", I3, rel_e) / r_eI[:, None, None, None]
            + jnp.einsum("ec,ea,eb->ecab", rel_e, rel_e, rel_e) / (r_eI**3)[:, None, None, None]
        )
        # Quadrature directions stay fixed under the strain: only u_e moves.
        vhat = grid / jnp.linalg.norm(grid, axis=1)[:, None]
        dcth_s = jnp.einsum("ecab,gc->egab", du, vhat)  # (Ne,Nv,3,3)

        S_det_base = _stress_lndet_mol(gd, ru0, rd0, R, geminal_inv=base_inv)
        S_det_mesh = _per_config(lambda ru, rd, gi: _stress_lndet_mol(gd, ru, rd, R, geminal_inv=gi)).reshape(Ne, Nv, 3, 3)
        S_full_base = jnp.asarray(_stress_lnPsi(hamiltonian_data, r_up, r_dn, geminal_inv=base_inv))
        S_full_mesh = _per_config(lambda ru, rd, gi: _stress_lnPsi(hamiltonian_data, ru, rd, geminal_inv=gi)).reshape(
            Ne, Nv, 3, 3
        )
        # Fixed directions: the moved electron sits at F R + |F rel| v_g = F r'_g - r_eI eps v_g, so the
        # affine config stress S(cfg) is corrected by -r_eI O(r'_g)[e] (x) v_g.
        dln_det = (
            (S_det_mesh - S_det_base[None, None])
            + drel[:, None, :, :] * vg_det[:, :, None, None]
            - jnp.einsum("e,egc,gd->egcd", r_eI, Oee_det, grid)
        )
        dln_full = (
            (S_full_mesh - S_full_base[None, None])
            + drel[:, None, :, :] * vg_full[:, :, None, None]
            - jnp.einsum("e,egc,gd->egcd", r_eI, Oee_full, grid)
        )

        coef_tg = twol1[:, :, None] * weights[None, None, :]
        dVgrid = coef_tg[..., None, None] * (
            (Vlp_t[:, :, None] * Pl)[..., None, None] * drel[:, None, None, :, :]
            + (Vl_t[:, :, None] * dPl)[..., None, None] * dcth_s[:, None, :, :, :]
        )  # (Ne,T,Nv,3,3)
        term1 = jnp.einsum("etgab,eg->ab", dVgrid, rat_ang)  # ANG geometry, delta-ratio-weighted
        term2 = jnp.einsum("eg,egab->ab", W1, dln_det) - jnp.einsum("eg,egab->ab", W2, dln_full)  # wf-ratio
        return jnp.real(term1 + term2)

    raise ValueError(f"grad_V_ecp_nl_dltmove: unknown wrt={wrt!r}")


def grad_V_ecp_wrt_strain(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array | None = None,
    Nv: int | None = None,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """D V_ecp / d eps (3, 3) -- the ECP electrostatic stress (PBC). The affine strain deforms the
    electrons, atoms and CELL; the quadrature DIRECTIONS v_g stay fixed, so every strained quadrature
    point lies on the sphere of the strained electron-nucleus distance and the rotation average of this
    derivative is the exact strain derivative of the angular integral (carrying v_g -> F v_g
    would move the points off that sphere and bias the stress). LOCAL = affine virial
    V_l'(r_eI) rel(x)rel/r_eI. NON-LOCAL = sum_g [dV_grid/deps ratio + V_grid ratio d ln ratio/deps],
    where the geometry strain dV_grid/deps uses d r_eI/deps = rel(x)rel/r_eI and d cos(theta)/deps
    (rel -> F rel, v_g fixed), and d ln ratio/deps = [stress_lnPsi_pbc(r'_g) - stress_lnPsi_pbc(r)]
    (full periodic Pulay stress at the mesh and base configs -- this carries the AFFINE electron strain)
    PLUS the non-affine mesh-radius correction (rel(x)rel/r_eI)(O(r'_g)[e].v_g) MINUS the affine
    over-motion r_eI O(r'_g)[e] (x) v_g of the moved electron.
    Verified vs jacrev(compute_ecp_coulomb_potential) with a fixed rotation.
    """
    from . import pbc_stress

    c = _ecp_nl_common(hamiltonian_data, r_up, r_dn, RT, Nv, geminal_inv=geminal_inv)
    geminal_inv = c["geminal_inv"]  # streamed mesh: the value-only base inverse its fields use
    Ne = c["Ne"]
    r_up = jnp.asarray(r_up, jnp.float64)
    r_dn = jnp.asarray(r_dn, jnp.float64)
    Nv = c["ratio"].shape[1]
    I3 = jnp.eye(3)
    grid, weights = c["grid"], c["weights"]
    rel_e, r_eI, u_e = c["rel_e"], c["r_eI"], c["u_e"]
    ratio, vg_Oee = c["ratio"], c["vg_Oee"]

    # local ECP stress: affine virial sum_{e,t} V_l'(r) rel(x)rel / r.
    stress_loc = jnp.einsum("et,eta,etb->ab", c["Vlocp"] / c["r_loc"], c["rel_loc"], c["rel_loc"])

    # geometry strain derivatives (rel -> F rel, quadrature directions v_g fixed; r_eI = |F rel|).
    drel = jnp.einsum("ea,eb->eab", rel_e, rel_e) / r_eI[:, None, None]  # (Ne,3,3) d r_eI/d eps
    du = (
        -jnp.einsum("ca,eb->ecab", I3, rel_e) / r_eI[:, None, None, None]
        + jnp.einsum("ec,ea,eb->ecab", rel_e, rel_e, rel_e) / (r_eI**3)[:, None, None, None]
    )  # (Ne,3,3,3)
    # Quadrature directions stay fixed under the strain: only u_e moves.
    vhat = grid / jnp.linalg.norm(grid, axis=1)[:, None]  # (Nv,3)
    dcth = jnp.einsum("ecab,gc->egab", du, vhat)  # (Ne,Nv,3,3)

    # per-term geometry strain (dVgrid) + wf-ratio strain (V_grid d ln ratio). V_grid here EXCLUDES ratio.
    Vl_t, Vlp_t, Pl, dPl, twol1 = c["Vl_t"], c["Vlp_t"], c["Pl"], c["dPl"], c["twol1"]
    coef_tg = twol1[:, :, None] * weights[None, None, :]  # (Ne,T,Nv) = (2l+1) w_g
    Vgrid_nr = (Vl_t * twol1)[:, :, None] * Pl * weights[None, None, :]  # (Ne,T,Nv)
    dVgrid = coef_tg[..., None, None] * (
        (Vlp_t[:, :, None] * Pl)[..., None, None] * drel[:, None, None, :, :]
        + (Vl_t[:, :, None] * dPl)[..., None, None] * dcth[:, None, :, :, :]
    )  # (Ne,T,Nv,3,3)
    term1 = jnp.einsum("etgab,eg->ab", dVgrid, ratio)

    # wf-ratio Pulay stress: sum_{e,g} W[e,g] [ (stress(config)-stress(base)) + drel*vg_Oee ], where
    # W = (sum_t Vgrid_nr) * ratio is the per-config weight.  The non-affine mesh-radius term
    # (drel*vg_Oee) is always closed-form; on the fast path the config stress delta is contracted with
    # the weights before any field is evaluated (one_move_mesh.mesh_stress_weighted, issue #622), else
    # the streamed per-config recompute.
    Wc = jnp.einsum("etg->eg", Vgrid_nr) * ratio  # (Ne,Nv)
    # Fixed quadrature directions: the strained mesh point is F R + |F rel| v_g =
    # F r'_g - r_eI eps v_g (+ the radius term), so remove the affine over-motion r_eI eps v_g of
    # the moved electron from the affine config Pulay stress.
    term2_radius = jnp.einsum("eg,eab->ab", Wc * vg_Oee, drel) - jnp.einsum("eg,e,egc,gd->cd", Wc, r_eI, c["Oee"], grid)
    if c["fast"]:  # Weighted rank-one periodic Pulay-stress delta (PBC + DTN), issue #622.
        from . import one_move_mesh

        term2_stress = one_move_mesh.mesh_stress_weighted(
            hamiltonian_data, r_up, r_dn, c["rprime"], Wc, c["fast_base"], c["fast_dm"]
        )
    else:
        from . import one_move_mesh

        # periodic Pulay stress d lnPsi/d eps (3,3) at base + each mesh config (holomorphic; Re at END).
        stress_base = pbc_stress.stress_lnPsi_pbc(hamiltonian_data, r_up, r_dn, take_real=False, geminal_inv=geminal_inv)
        # Per-config displaced geminal inverse via EXACT rank-1 Sherman-Morrison off the base (threaded)
        # inverse -- ECP move = rank-1 row/column update of G, no fresh SVD-pinv per config.  Threading it
        # into stress_lnPsi_pbc (-> grad_lndet_wrt_strain, which uses G^{-1} as a constant cotangent field)
        # keeps the streamed complex-twist e_L/eps path free of ``lapack_[dz]gesdd``.
        ginv_cfg = one_move_mesh.sm_geminal_inv_configs(c["gd"], r_up, r_dn, c["RU"], c["RD"], geminal_inv=geminal_inv)
        stress_mesh = _mesh_map(
            lambda ru, rd, gi: pbc_stress.stress_lnPsi_pbc(hamiltonian_data, ru, rd, take_real=False, geminal_inv=gi),
            (c["RU"], c["RD"], ginv_cfg),
        ).reshape(Ne, Nv, 3, 3)
        term2_stress = jnp.einsum("eg,egab->ab", Wc, stress_mesh - stress_base[None, None])
    return jnp.real(stress_loc + term1 + term2_radius + term2_stress)


def grad_V_ecp_wrt_strain_mol(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array | None = None,
    Nv: int | None = None,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """MOLECULAR ECP electrostatic stress d V_ecp / d eps (3, 3).

    Mirror of the PBC :func:`grad_V_ecp_wrt_strain` -- LOCAL affine virial ``V_l'(r) rel(x)rel/r`` +
    NON-LOCAL geometry strain (``dVgrid/deps`` from ``d r_eI/deps`` and ``d cos theta/deps`` with
    ``rel -> F rel`` and the quadrature directions ``v_g`` fixed) + NON-LOCAL wf-ratio Pulay stress
    (including the fixed-direction mesh term) -- but with the periodic
    Pulay stress (``pbc_stress.stress_lnPsi_pbc`` / the ``one_move_mesh.mesh_stress_wf`` autodiff through
    the strained-geminal pinv) replaced by the MOLECULAR affine virial in CLOSED FORM.

    The per-config molecular Pulay stress is ``_stress_lnPsi(cfg) = drift(cfg) . r_cfg + dR(cfg) . R``.
    Both the config drift field ``O_mesh`` (already reused by e_L/r) and the
    config Pulay-force field ``OR_mesh`` (already reused by e_L/R) are the SAME rank-1 Sherman-Morrison
    fields the R/r axes ride off the base ``geminal_inv`` -- so ``S(cfg)`` is assembled directly by
    contracting them with the config electron positions (base ``r_all`` with the moved slot at ``r'_g``)
    and the nuclei ``R``.  No autodiff, no per-config SVD-pinv: with the running ``geminal_inv`` threaded
    the whole axis lowers with zero ``lapack_dgesdd`` calls. The non-affine mesh-radius
    correction ``(rel(x)rel/r_eI)(O(r'_g)[e] . v_g)`` and the geometry-strain ``term1`` are identical to
    the PBC kernel.  Verified vs ``_auto`` (strained forward) to fp.
    """
    from .assemble import _force_lnPsi

    c = _ecp_nl_common(hamiltonian_data, r_up, r_dn, RT, Nv, geminal_inv=geminal_inv)
    geminal_inv = c["geminal_inv"]  # streamed mesh: the value-only base inverse its fields use
    Ne, n_atom = c["Ne"], c["n_atom"]
    r_up = jnp.asarray(r_up, jnp.float64)
    r_dn = jnp.asarray(r_dn, jnp.float64)
    Nv = c["ratio"].shape[1]
    I3 = jnp.eye(3)
    grid, weights = c["grid"], c["weights"]
    rel_e, r_eI, u_e = c["rel_e"], c["r_eI"], c["u_e"]
    ratio, vg_Oee = c["ratio"], c["vg_Oee"]
    R = jnp.asarray(hamiltonian_data.structure_data.positions, jnp.float64)  # (n_atom, 3)
    r_all = jnp.concatenate([r_up, r_dn], 0)  # (Ne, 3)

    # local ECP stress: affine virial sum_{e,t} V_l'(r) rel(x)rel / r (identical to the PBC kernel).
    stress_loc = jnp.einsum("et,eta,etb->ab", c["Vlocp"] / c["r_loc"], c["rel_loc"], c["rel_loc"])

    # non-local geometry strain (rel -> F rel, quadrature directions v_g fixed; r_eI = |F rel|).
    drel = jnp.einsum("ea,eb->eab", rel_e, rel_e) / r_eI[:, None, None]  # (Ne,3,3) d r_eI/d eps
    du = (
        -jnp.einsum("ca,eb->ecab", I3, rel_e) / r_eI[:, None, None, None]
        + jnp.einsum("ec,ea,eb->ecab", rel_e, rel_e, rel_e) / (r_eI**3)[:, None, None, None]
    )  # (Ne,3,3,3)
    # Quadrature directions stay fixed under the strain: only u_e moves.
    vhat = grid / jnp.linalg.norm(grid, axis=1)[:, None]  # (Nv,3)
    dcth = jnp.einsum("ecab,gc->egab", du, vhat)  # (Ne,Nv,3,3)

    Vl_t, Vlp_t, Pl, dPl, twol1 = c["Vl_t"], c["Vlp_t"], c["Pl"], c["dPl"], c["twol1"]
    coef_tg = twol1[:, :, None] * weights[None, None, :]  # (Ne,T,Nv) = (2l+1) w_g
    Vgrid_nr = c["Vgrid_nr"]  # (Ne,T,Nv) angular, NO ratio
    dVgrid = coef_tg[..., None, None] * (
        (Vlp_t[:, :, None] * Pl)[..., None, None] * drel[:, None, None, :, :]
        + (Vl_t[:, :, None] * dPl)[..., None, None] * dcth[:, None, :, :, :]
    )  # (Ne,T,Nv,3,3)
    term1 = jnp.einsum("etgab,eg->ab", dVgrid, ratio)  # non-local geometry strain

    # wf-ratio MOLECULAR Pulay stress (closed form). W = (sum_t Vgrid_nr) * ratio is the per-config
    # weight; the non-affine mesh-radius term (drel * vg_Oee) is closed-form (== the PBC kernel), and the
    # Contract the weighted drift/force (sum_g W_g O, sum_g W_g OR) with electron/nuclear positions
    # to obtain the weighted Pulay-stress difference sum_g W_g [S(cfg) - S(base)].
    Wc = jnp.einsum("etg->eg", Vgrid_nr) * ratio  # (Ne,Nv)
    # Fixed quadrature directions: the strained mesh point is F R + |F rel| v_g =
    # F r'_g - r_eI eps v_g (+ the radius term), so remove the affine over-motion r_eI eps v_g of
    # the moved electron from the affine config Pulay stress.
    term2_radius = jnp.einsum("eg,eab->ab", Wc * vg_Oee, drel) - jnp.einsum("eg,e,egc,gd->cd", Wc, r_eI, c["Oee"], grid)

    Oee = c["Oee"]  # (Ne,Nv,3) the moved electron's own drift = O_mesh[e,:,e,:]
    O_base = _O_drift(c["gd"], c["jd"], r_up, r_dn, take_real=True, geminal_inv=geminal_inv)  # (Ne,3)
    OR_base = jnp.asarray(_force_lnPsi(hamiltonian_data, r_up, r_dn, take_real=True, geminal_inv=geminal_inv))  # (nA,3)
    if c["fast"]:  # rank-1 config Pulay force contracted with the config weights (issue #596)
        from . import one_move_mesh

        OR_w = one_move_mesh.mesh_force_weighted(hamiltonian_data, c["mesh_state"], Wc, geminal_inv=geminal_inv)
    else:  # streamed per-config Pulay force (non-(pade/exp) Jastrow etc.)
        from . import one_move_mesh

        # Per-config displaced geminal inverse via EXACT rank-1 Sherman-Morrison off the base (threaded)
        # inverse (ECP move = rank-1 update of G, no fresh SVD-pinv per config) -- drops the streamed
        # ``lapack_[dz]gesdd`` on this molecular e_L/eps mesh Pulay-force term when geminal_inv is threaded.
        ginv_cfg = one_move_mesh.sm_geminal_inv_configs(c["gd"], r_up, r_dn, c["RU"], c["RD"], geminal_inv=geminal_inv)
        OR_mesh = _mesh_map(
            lambda ru, rd, gi: _force_lnPsi(hamiltonian_data, ru, rd, take_real=True, geminal_inv=gi),
            (c["RU"], c["RD"], ginv_cfg),
        ).reshape(Ne, Nv, n_atom, 3)
        OR_w = jnp.einsum("eg,egic->ic", Wc, OR_mesh)

    # S(cfg) = drift(cfg) . r_cfg + dR(cfg) . R, with r_cfg = r_all except the moved slot e -> r'_g.
    # drift part = sum_m O_mesh[m] . r_all[m] + Oee . (r'_g - r_e) (the moved slot's affine position).
    disp = c["rprime"] - r_all[:, None, :]  # (Ne,Nv,3) moved-electron displacement r'_g - r_e
    SV = jnp.sum(Wc)
    drift_cfg = jnp.einsum("mc,md->cd", _weighted_drift(hamiltonian_data, c, Wc), r_all) + jnp.einsum(
        "eg,egc,egd->cd", Wc, Oee, disp
    )
    force_cfg = jnp.einsum("ic,id->cd", OR_w, R)
    S_base = jnp.einsum("mc,md->cd", O_base, r_all) + jnp.einsum("ic,id->cd", OR_base, R)
    term2_stress = drift_cfg + force_cfg - SV * S_base
    return jnp.real(stress_loc + term1 + term2_radius + term2_stress)
