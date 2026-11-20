"""Rank-one weighted derivatives for single-electron candidate meshes.

Given an ``(Ne, Nv, 3)`` candidate mesh and bare weights ``W`` with shape
``(Ne, Nv)``, this module computes reduced weighted derivatives. Each candidate
configuration moves one electron; the mesh is electron-major, candidate-minor,
with the up-spin electron block first. This layout is shared by ECP non-local
quadrature and the LRDMC/GFMC six-hop S-prime mesh. The reduced derivative is

    sum_{e,v} W[e,v] * [ d lnPsi(config_{e,v}) / dq  -  d lnPsi(base) / dq ]

for positions, returned as ``(dR (nA, 3), dr (Ne, 3))``, and for strain,
returned as ``(3, 3)``. It avoids evaluating the full wave-function derivative
at every candidate configuration. The per-config work uses :mod:`one_move_mesh`
(``det_base`` / ``det_mesh`` / ``det_drift_mesh`` / ``mesh_force`` /
``mesh_stress_weighted`` and the Jastrow one-move deltas): base AO/geminal tables are
built once, each candidate needs one-point orbital evaluation plus at most
``O(Ne^2)`` candidate algebra, and large table products are shared GEMMs.

For complex twists, the determinant machinery is holomorphic: it uses plain
transposes without conjugation and preserves the complex Bloch-orbital dtype.
Jastrow channels remain real. The caller keeps ``W`` complex and projects the
assembled product to its real part at the observable boundary, preserving the
``Re(W O)`` convention.

The :func:`supported` predicate requires closed-form move deltas for active J1
and J2 channels. Complex twists are supported here. ECP quadrature uses its own
real-only capability predicate. Unsupported inputs remain on the streamed
full-recompute route.
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

from logging import getLogger
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax import typing as jnpt

from jqmc.hamiltonians import Hamiltonian_data

from . import one_move_mesh

logger = getLogger("jqmc").getChild(__name__)

MoveState: TypeAlias = dict[str, object]


def supported(hamiltonian_data: Hamiltonian_data) -> bool:
    """True when rank-one weighted deltas support the Hamiltonian.

    J1 and J2 must use the production Pade or exponential move-delta kernels
    when present. J3 (bare-AO or MO-backed, folded), DTN, SD/AGP/AGPn
    determinants, and molecular or periodic cells are supported. Complex-twist
    orbitals are also supported because the determinant one-point stacks are
    holomorphic. ECP quadrature applies a separate real-only capability check.
    """
    try:
        jd = hamiltonian_data.wavefunction_data.jastrow_data
        # complex Bloch J3 leaf: streamed (see one_move_mesh.fast_supported)
        reason = one_move_mesh._fast_reject_reason(jd)
        if reason is not None:
            one_move_mesh.warn_streamed_fallback("fast_mesh.supported (GFMC S-prime mesh)", reason, jd)
            return False
        return True
    except Exception:
        logger.warning("fast_mesh.supported probe raised; keeping the streamed slow path.", exc_info=True)
        return False


def build_move_state(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    rprime: jnpt.ArrayLike,
    geminal_inv: jnpt.ArrayLike | None = None,
) -> MoveState:
    """Shared primal for the reduced adjoints: per-config ratio, the moved electron's
    own drift ``Oee`` ``(Ne, Nv, 3)``, and the reusable rank-one state
    (:func:`one_move_mesh.mesh_ratio_state`: one ``det_base`` + ``det_mesh``; the force and
    strain adjoints contract it with their weights, so a combined force and stress request
    builds the primal once).

    ``rprime`` is the ``(Ne, Nv, 3)`` candidate mesh (up electrons first).
    ``geminal_inv``: the driver's running SM ``G^{-1}`` at the base positions
    (no fresh SVD-pinv anywhere on the mesh when threaded).
    """
    ratio, Oee, mesh = one_move_mesh.mesh_ratio_state(hamiltonian_data, r_up, r_dn, rprime, geminal_inv=geminal_inv)
    return {"ratio": ratio, "Oee": Oee, "mesh": mesh, "base": mesh["base"], "dm": mesh["dm"], "rprime": jnp.asarray(rprime)}


def _base_drift(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    holomorphic: bool,
    geminal_inv: jnpt.ArrayLike | None = None,
) -> jax.Array:
    """Base ``d lnPsi/dr`` ``(Ne, 3)`` -- holomorphic (complex det part kept) or the
    real sampling drift, matching the channel kernels embedded in the ``O`` field."""
    if holomorphic:
        from .gfmc import _drift_lnPsi_holo

        du, dd = _drift_lnPsi_holo(hamiltonian_data, r_up, r_dn, geminal_inv=geminal_inv)
    else:
        from .assemble import _drift_lnPsi

        du, dd = _drift_lnPsi(hamiltonian_data, r_up, r_dn, geminal_inv=geminal_inv)
    return jnp.concatenate([jnp.asarray(du), jnp.asarray(dd)], axis=0)


def _shared_OR(
    hamiltonian_data: Hamiltonian_data,
    state: MoveState,
    W: jax.Array,
    geminal_inv: jnpt.ArrayLike | None = None,
) -> jax.Array:
    """Return ``sum W OR`` ``(nA, 3)`` of the rank-one ``mesh_force`` field (DTN rows
    excluded), cached on the move state.

    The cache is weight-specific like the DTN cache: a fused force and strain caller
    contracts the same weights once.
    """
    OR_w = state.get("OR_w")
    if OR_w is None:
        OR_w = one_move_mesh.mesh_force_weighted(hamiltonian_data, state["mesh"], W, geminal_inv=geminal_inv, include_dtn=False)
        state["OR_w"] = OR_w
    return OR_w


def _prepare_dtn_geometry_delta(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    state: MoveState,
    W: jnpt.ArrayLike,
    *,
    compute_force: bool,
    compute_strain: bool,
) -> tuple[jax.Array | None, jax.Array | None]:
    """Compute/cache reduced one-move DTN geometry deltas on ``state``.

    The cache is weight-specific; a move state has exactly one consumer weight
    field in production.  The fused force+stress caller asks for both outputs
    before either generic mesh path runs, while single-observable callers lower
    only the requested channel.
    """
    need_force = compute_force and "dtn_dR_delta" not in state
    need_strain = compute_strain and "dtn_strain_delta" not in state
    if not (need_force or need_strain):
        return state.get("dtn_dR_delta"), state.get("dtn_strain_delta")

    dtn_data = hamiltonian_data.wavefunction_data.jastrow_data.jastrow_dtn_data
    weight_dtype = jnp.asarray(W).dtype
    n_atom = int(hamiltonian_data.structure_data.positions.shape[0])
    if dtn_data is None:
        if need_force:
            state["dtn_dR_delta"] = jnp.zeros((n_atom, 3), weight_dtype)
        if need_strain:
            state["dtn_strain_delta"] = jnp.zeros((3, 3), weight_dtype)
        return state.get("dtn_dR_delta"), state.get("dtn_strain_delta")

    from .dtn import dtn_geometry_mesh_weighted

    rprime = jnp.asarray(state["rprime"])
    ne, nv = int(rprime.shape[0]), int(rprime.shape[1])
    pts = rprime.reshape(ne * nv, 3)
    e_idx = jnp.repeat(jnp.arange(ne, dtype=jnp.int32), nv)
    dR, strain = dtn_geometry_mesh_weighted(
        dtn_data,
        r_up,
        r_dn,
        pts,
        e_idx,
        jnp.asarray(W).reshape(ne * nv),
        compute_force=need_force,
        compute_strain=need_strain,
    )
    if need_force:
        state["dtn_dR_delta"] = dR
    if need_strain:
        state["dtn_strain_delta"] = strain
    return state.get("dtn_dR_delta"), state.get("dtn_strain_delta")


def weighted_position_delta(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    state: MoveState,
    W: jnpt.ArrayLike,
    geminal_inv: jnpt.ArrayLike | None = None,
    holomorphic: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Reduced position adjoint ``(dR (nA,3), dr (Ne,3))`` =
    ``sum_{e,v} W[e,v] [ (d lnPsi/d{R,r})(config) - (d lnPsi/d{R,r})(base) ]``.

    ``dr`` contracts the rank-one drift field ``O`` (which embeds the base drift
    channel-by-channel) against ``W`` and subtracts the ``sum(W)``-weighted base
    drift; ``dR`` contracts the rank-one ``mesh_force`` field the same way.  The
    real base ``mesh_force`` embeds cancels in the delta, so the assembled result
    is the exact holomorphic delta at a twist (``W`` complex, Re at the caller).
    """
    from .assemble import _force_lnPsi

    W = jnp.asarray(W)

    Wsum = jnp.sum(W)
    drift_base = _base_drift(hamiltonian_data, r_up, r_dn, holomorphic, geminal_inv=geminal_inv)
    dr = one_move_mesh.mesh_drift_weighted(hamiltonian_data, state["mesh"], W) - Wsum * drift_base

    OR_w = _shared_OR(hamiltonian_data, state, W, geminal_inv=geminal_inv)
    # ``mesh_force`` embeds the base force MATCHING its dtype (holomorphic complex at a
    # twist, real otherwise); subtract the SAME base so the delta is exactly
    # ``sum W * (force(config) - force(base))`` -- the rank-one corrections are the exact
    # (holomorphic when complex) per-config deltas.
    _embed_holo = jnp.iscomplexobj(state["base"]["Ginv"])
    force_base = jnp.asarray(_force_lnPsi(hamiltonian_data, r_up, r_dn, take_real=not _embed_holo, geminal_inv=geminal_inv))
    dR = OR_w - Wsum * force_base
    dtn_dR, _ = _prepare_dtn_geometry_delta(
        hamiltonian_data,
        r_up,
        r_dn,
        state,
        W,
        compute_force=True,
        compute_strain=False,
    )
    dR = dR + dtn_dR
    return dR, dr


def weighted_strain_delta(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    state: MoveState,
    W: jnpt.ArrayLike,
    geminal_inv: jnpt.ArrayLike | None = None,
    holomorphic: bool = False,
) -> jax.Array:
    """Reduced strain adjoint ``(3, 3)`` =
    ``sum_{e,v} W[e,v] [ (d lnPsi/d eps)(config) - (d lnPsi/d eps)(base) ]``.

    PERIODIC: the full periodic Pulay stress delta via :func:`one_move_mesh.mesh_stress_weighted`
    (the weighted sum of the closed-form ``stress_lnPsi_pbc`` deltas, contracted with the
    weights before any field is evaluated; cell/image terms carried).  MOLECULAR: the affine
    virial assembled from the shared rank-one drift/force fields -- the config electron
    positions differ from the base only in the moved row, so the virial is the base
    contraction plus a moved-row correction (no per-config field recompute).
    """
    from .assemble import _force_lnPsi

    W = jnp.asarray(W)
    base, dm, rprime = state["base"], state["dm"], state["rprime"]

    if hamiltonian_data.structure_data.pbc_flag:
        generic = one_move_mesh.mesh_stress_weighted(
            hamiltonian_data,
            r_up,
            r_dn,
            rprime,
            W,
            base,
            dm,
            include_dtn=False,
        )
        _, dtn_strain = _prepare_dtn_geometry_delta(
            hamiltonian_data,
            r_up,
            r_dn,
            state,
            W,
            compute_force=False,
            compute_strain=True,
        )
        return generic + dtn_strain

    # molecular affine virial (twist implies PBC, so this branch is always real).
    r_all = jnp.concatenate([jnp.asarray(r_up), jnp.asarray(r_dn)], axis=0)
    R = jnp.asarray(hamiltonian_data.structure_data.positions)
    Wsum = jnp.sum(W)

    OR_w = _shared_OR(hamiltonian_data, state, W, geminal_inv=geminal_inv)
    drift_base = _base_drift(hamiltonian_data, r_up, r_dn, False, geminal_inv=geminal_inv)
    force_base = jnp.asarray(_force_lnPsi(hamiltonian_data, r_up, r_dn, take_real=True, geminal_inv=geminal_inv))

    # sum_c Wc sum_j drift(cfg)_j (x) cfgpos_j : cfgpos == r_all except the moved row e,
    # which sits at rprime[e,v] = r_all[e] + shift -- one diagonal-slice correction.
    O_moved = state["Oee"]  # (Ne, Nv, 3): the moved electron's own drift per config
    shift = rprime - r_all[:, None, :]  # (Ne, Nv, 3)
    O_w = one_move_mesh.mesh_drift_weighted(hamiltonian_data, state["mesh"], W)  # (Ne, 3)
    term_e = jnp.einsum("jc,jd->cd", O_w, r_all) + jnp.einsum("ev,evc,evd->cd", W, O_moved, shift)
    term_I = jnp.einsum("Ic,Id->cd", OR_w, R)
    base_e = Wsum * jnp.einsum("ec,ed->cd", drift_base, r_all)
    base_I = Wsum * jnp.einsum("Ic,Id->cd", force_base, R)
    # ``O`` already carries the DTN electron-drift delta.  ``OR`` deliberately
    # omits only the DTN nuclear delta, so restore its affine nuclear virial.
    dtn_dR, _ = _prepare_dtn_geometry_delta(
        hamiltonian_data,
        r_up,
        r_dn,
        state,
        W,
        compute_force=True,
        compute_strain=False,
    )
    return term_e + term_I - base_e - base_I + jnp.einsum("Ic,Id->cd", dtn_dR, R)
