"""Analytic position and strain derivatives of the LRDMC measurement energy.

LRDMC propagation uses the clipped lattice Hamiltonian, while the novar=0
mixed estimator measures the continuous local energy.  Force and stress are
therefore derivatives of that continuous measurement energy; no derivative of
the propagation-only sign-flip regularizer or its S-prime set appears here.

The shared kernels support molecular and periodic all-electron/ECP
Hamiltonians, real Gamma and complex twists, and polynomial or DTN Jastrows.
GFMC_t and GFMC_n differ only in their propagation weights and delegate to the
same measurement-derivative implementation.
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
from functools import cache
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jax import typing as jnpt

from jqmc.coulomb_potential import compute_coulomb_potential_ei_discretized_dispatch, compute_coulomb_potential_ei_dispatch
from jqmc._route_probe import route_site
from jqmc.determinant import AGP_data, AGPn_data, SD_data
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    compute_grads_and_laplacian_Jastrow_part,
)
from jqmc.wavefunction import (
    compute_discretized_kinetic_energy,
    compute_discretized_kinetic_energy_fast_update,
    compute_kinetic_energy_all_elements,
    compute_kinetic_energy_all_elements_fast_update,
)

from . import assemble
from .kinetic import _det_grad_field

NonLocalMove: TypeAlias = Literal["tmove", "dltmove"]
GFMCQuantity: TypeAlias = Literal["pos", "strain", "both"]
GeminalData: TypeAlias = AGP_data | AGPn_data | SD_data
PositionGradients: TypeAlias = tuple[jax.Array, jax.Array, jax.Array]
PositionAndStrainGradients: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
ArrayPair: TypeAlias = tuple[jax.Array, jax.Array]
GFMCGradientResult: TypeAlias = PositionGradients | jax.Array | PositionAndStrainGradients
GFMCNumericKernel: TypeAlias = Callable[
    [Hamiltonian_data, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None],
    GFMCGradientResult,
]


def _assert_scope(hamiltonian_data: Hamiltonian_data) -> None:
    """Document the analytic GFMC derivative scope.

    The route accepts molecular all-electron/ECP and periodic all-electron/ECP
    Hamiltonians at real Gamma or a complex twist, with no Jastrow, polynomial
    J1/J2/J3, or DTN.  The differentiated quantity is always the continuous
    novar=0 measurement energy.  Propagation-only lattice clipping is outside
    this derivative boundary.

    The function is retained as the single documented capability boundary; no
    currently constructible Hamiltonian in this vocabulary is excluded.
    """
    del hamiltonian_data


def _validate_non_local_move(hamiltonian_data: Hamiltonian_data, non_local_move: str | None) -> None:
    """Validate the ECP move vocabulary at the public boundary.

    The shared numerical body only has an extra term for ``dltmove``; without
    this guard an unknown value silently took the ``tmove`` path.  ``None`` is
    the production representation for an all-electron Hamiltonian, where no
    non-local ECP move exists; ECP Hamiltonians must select ``tmove`` or
    ``dltmove`` explicitly.  Keep the vocabulary check outside JIT, alongside
    :func:`_assert_scope`.
    """
    ecp_flag = bool(getattr(hamiltonian_data.coulomb_potential_data, "ecp_flag", False))
    if non_local_move in ("tmove", "dltmove") or (non_local_move is None and not ecp_flag):
        return
    expected = "'tmove' or 'dltmove'" if ecp_flag else "None, 'tmove', or 'dltmove'"
    raise ValueError(f"unsupported non_local_move={non_local_move!r}; expected {expected}")


def _is_pbc(hamiltonian_data: Hamiltonian_data) -> bool:
    return bool(hamiltonian_data.structure_data.pbc_flag)


def _has_poly_jastrow(jd: Jastrow_data) -> bool:
    """True iff an analytic-polynomial Jastrow (J1/J2/J3) is present (the DTN channel is handled
    separately -- it is NOT a poly-side kinetic term).
    """
    return any(
        getattr(jd, f, None) is not None for f in ("jastrow_one_body_data", "jastrow_two_body_data", "jastrow_three_body_data")
    )


def _has_dtn(jd: Jastrow_data) -> bool:
    """True iff a DTN (cutoff-poly u/chi/f) Jastrow channel is present."""
    return getattr(jd, "jastrow_dtn_data", None) is not None


def _has_any_jastrow(jd: Jastrow_data) -> bool:
    """True iff any polynomial or DTN Jastrow channel is present."""
    return _has_poly_jastrow(jd) or _has_dtn(jd)


def _drift_lnPsi_holo(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> ArrayPair:
    """Return the unprojected complex ``d ln Psi / dr`` for a twisted cell.

    The retained rank-one weighted-delta kernel in :mod:`fast_mesh` contracts
    complex candidate weights with this field before taking the observable's
    real part.  Using the real sampling drift here would lose the
    ``-Im(weight) * Im(drift)`` contribution.  Polynomial and DTN Jastrows are
    real and are therefore added directly to the holomorphic determinant
    drift.

    This low-level helper has no measurement-side S-prime correction; the
    novar=0 GFMC force/stress route does not call it.
    """
    wf = hamiltonian_data.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    gdu, gdd = _det_grad_field(gd, r_up, r_dn, geminal_inv=geminal_inv)
    du, dd = jnp.asarray(gdu), jnp.asarray(gdd)
    if _has_any_jastrow(jd):
        gju, gjd, _, _ = compute_grads_and_laplacian_Jastrow_part(jd, r_up, r_dn)
        du = du + jnp.asarray(gju)
        dd = dd + jnp.asarray(gjd)
    return du, dd


def _dltmove_delta_active(
    hamiltonian_data: Hamiltonian_data,
    non_local_move: NonLocalMove | None,
) -> bool:
    """True iff the ``dltmove`` ECP non-local Jastrow-reweight correction is live for this config.

    The LRDMC ``dltmove`` estimator forms a DIFFERENT ``e_L`` from ``tmove``: the fixed-node-clamped
    non-local hop uses the DETERMINANT-only amplitude and re-weights only the accepted (FN) part by
    the Jastrow ratio ``exp(J_mesh - J_ref)``, so ``e_L^dlt = e_L^t + delta`` with
    ``delta = sum_g max(V^det_g, 0)(1 - jr_g)`` (see :func:`grad_e_L_wrt_positions`).  The
    correction vanishes when there is no Jastrow (``jr = 1`` -> ``dltmove == tmove``) or no ECP.

    Molecular and periodic-Gamma systems use the closed-form ``delta`` force
    (``ecp_force.grad_V_ecp_nl_dltmove``
    ``r``/``R`` branches) composes for BOTH -- its shared setup ``_ecp_nl_common`` is already min-image /
    periodic-aware, so the same kernel serves PBC. The
    ``delta`` STRESS uses the molecular ``ecp_force.grad_V_ecp_nl_dltmove(wrt="eps")`` for molecular and
    the periodic :func:`_dltmove_delta_strain_pbc` for PBC, where explicit cell
    terms are required.
    """
    if non_local_move != "dltmove":
        return False
    cp = hamiltonian_data.coulomb_potential_data
    jd = hamiltonian_data.wavefunction_data.jastrow_data
    return getattr(cp, "ecp_flag", False) and _has_any_jastrow(jd)


def _sprime_data(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array,
    alat: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> ArrayPair:
    """Independent oracle for the propagation S' membership and hopping matrix.

    This helper is not part of the novar=0 force/stress route.  Coverage tests
    compare it with the membership derived from the live projection state, so
    propagation's sign-flip predicates cannot drift from their independent
    recomputation.

    ``geminal_inv`` (fast path): the base-position running SM ``G^{-1}`` (must equal
    ``G(r_up, r_dn)^{-1}`` exactly, which the driver guarantees).  When provided, the two forward
    kinetic evaluators (continuous ``Tc`` and the 6-hop discretised mesh) use their fast-update
    variants so this membership setup lowers with NO fresh SVD-pinv; ``None`` (equivalence harness)
    keeps the plain forward recompute.
    """
    wf = hamiltonian_data.wavefunction_data
    if geminal_inv is not None:
        Tc_up, Tc_dn = compute_kinetic_energy_all_elements_fast_update(wf, r_up, r_dn, geminal_inv)
        # Keep the oracle on the same fused DTN lattice evaluator as propagation.
        with route_site("force"):
            _, _, elem = compute_discretized_kinetic_energy_fast_update(alat, wf, geminal_inv, r_up, r_dn, RT)
    else:
        Tc_up, Tc_dn = compute_kinetic_energy_all_elements(wf, r_up, r_dn)
        with route_site("force"):
            _, _, elem = compute_discretized_kinetic_energy(alat, wf, r_up, r_dn, RT)
    # Keep hopping elements complex at a twist; membership follows the same
    # real-primary reduction as the live projection state.
    elem_all = elem.reshape(-1, 6)
    sign_flip = jnp.any(jnp.real(elem_all) >= 0, axis=1)
    ndk = jnp.sum(elem_all + 1.0 / (2.0 * alat**2), axis=1)  # S_i + 3/a^2 (complex at twist)
    ei_up, ei_dn = compute_coulomb_potential_ei_dispatch(hamiltonian_data.coulomb_potential_data, r_up, r_dn)
    eid_up, eid_dn = compute_coulomb_potential_ei_discretized_dispatch(
        hamiltonian_data.coulomb_potential_data, r_up, r_dn, alat
    )
    ei = jnp.concatenate([jnp.asarray(ei_up), jnp.asarray(ei_dn)])
    eid = jnp.concatenate([jnp.asarray(eid_up), jnp.asarray(eid_dn)])
    Tc = jnp.concatenate([jnp.asarray(Tc_up), jnp.asarray(Tc_dn)])
    zv = jnp.real(ei + Tc - ndk)
    # ``opt_i`` selects the discretised branch when sign_flip AND ``ei_used_i > zv_i``.  The
    # forward uses ``ei_used = eid`` (all-electron) but ``ei_used = ei`` (continuous) for ECP
    # (jqmc_gfmc._compute_V_elements_t lines ~1559/1576) -- so the S' membership comparison must
    # switch potential accordingly.  Consequently for ECP the ``eid - ei`` Coulomb S' correction
    # vanishes identically (see grad_e_L_wrt_positions).
    ei_used = ei if getattr(hamiltonian_data.coulomb_potential_data, "ecp_flag", False) else eid
    w = (jnp.asarray(sign_flip, dtype=bool) & (jnp.real(ei_used) > jnp.real(zv))).astype(jnp.float64)
    return w, elem_all


def _dltmove_delta_strain_pbc(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """PERIODIC-Gamma analogue of ``ecp_force.grad_V_ecp_nl_dltmove(wrt="eps")`` (the molecular
    ``dltmove`` ECP non-local Jastrow-reweight stress correction ``d(delta)/d eps``).

    ``delta = P1 - P2`` with ``P1 = sum_{SP} ang det_ratio`` (det-only ratio) and
    ``P2 = sum_{SP} ang det_ratio jr`` (full ratio), ``SP = {V^det >= 0}`` -- exactly the molecular
    decomposition (see ``ecp_force.grad_V_ecp_nl_dltmove``).  The molecular kernel's shared per-candidate
    setup (``_ecp_nl_common`` -> ``_dltmove_delta_terms``) is ALREADY min-image / periodic-aware, so its
    ``r``/``R`` FORCE branches compose for PBC verbatim; ONLY the ``eps`` branch's wavefunction-ratio
    Pulay stress uses the MOLECULAR affine virial (``_stress_lndet_mol`` / ``_stress_lnPsi``), which
    misses the periodic cell terms.  This function reproduces the molecular ``eps`` assembly (geometry
    strain ``term1`` + wf-ratio Pulay ``term2``) but swaps the two wf-ratio Pulay stresses for the FULL
    PERIODIC Pulay stress ``pbc_stress.stress_lnPsi_pbc`` (det-only via a Jastrow-stripped ham for P1;
    the full ham for P2), keeping the (identical) affine geometry + non-affine mesh-radius terms.
    Verified vs ``jax.jacrev`` of the strained discretised ``dltmove`` e_L on lithium (PBC ECP, j12 /
    j123ao): max abs 1.1e-13 / 1.6e-13.  Returns (3, 3).

    ``geminal_inv`` (fast path): base-position running SM ``G^{-1}`` reused by the base periodic Pulay
    stress (det-only + full) and the ``_ecp_nl_common`` base setup (the mesh-config re-evals form their
    own inverse).
    """
    from . import pbc_stress
    from .ecp_force import _dltmove_delta_terms, _lndet_drift
    from .mesh_stream import _mesh_map

    c, det_ratio, jr, ang, m, W1, W2, rat_ang = _dltmove_delta_terms(hamiltonian_data, r_up, r_dn, RT, None)
    gd = c["gd"]
    Ne = c["Ne"]
    Nv = c["ratio"].shape[1]
    RU, RD = c["RU"], c["RD"]
    grid, weights = c["grid"], c["weights"]
    rel_e, r_eI, u_e = c["rel_e"], c["r_eI"], c["u_e"]
    Vl_t, Vlp_t, Pl, dPl, twol1 = c["Vl_t"], c["Vlp_t"], c["Pl"], c["dPl"], c["twol1"]
    ru0 = jnp.asarray(r_up, jnp.float64)
    rd0 = jnp.asarray(r_dn, jnp.float64)
    eidx = jnp.arange(Ne)
    # Streamed route: ``c["Oee"]`` and ``det_ratio`` share the value-only per-config inverses
    # (``one_move_mesh.ratio_table_geminal_inv``); every streamed mesh field and its base take
    # the same inverses.  The rank-one route keeps its fields unchanged.
    if c["fast"]:
        base_inv = geminal_inv

        def _per_config(field):
            return _mesh_map(lambda ru, rd: field(ru, rd, None), (RU, RD))

    else:
        from . import one_move_mesh

        base_inv = c["geminal_inv"] if geminal_inv is None else geminal_inv
        ginv_cfg = one_move_mesh.sm_geminal_inv_configs(gd, ru0, rd0, RU, RD, geminal_inv=base_inv)

        def _per_config(field):
            return _mesh_map(field, (RU, RD, ginv_cfg))

    # det-only + full drift O-fields at each mesh config; Oee = the moved electron's own drift.
    O_det_mesh = _per_config(lambda ru, rd, gi: _lndet_drift(gd, ru, rd, geminal_inv=gi)).reshape(Ne, Nv, Ne, 3)
    Oee_det = O_det_mesh[eidx, :, eidx, :]  # (Ne,Nv,3)
    vg_det = jnp.einsum("gc,egc->eg", grid, Oee_det)
    vg_full = c["vg_Oee"]

    # geometry strain (rel -> F rel with the quadrature directions v_g held fixed; mesh radius
    # r_eI = |F rel| non-affine) -- IDENTICAL to the molecular eps branch AND the periodic tmove
    # ECP-nl stress (min-image rel strains affinely; quadrature directions stay fixed).
    I3 = jnp.eye(3)
    drel = jnp.einsum("ea,eb->eab", rel_e, rel_e) / r_eI[:, None, None]
    du = (
        -jnp.einsum("ca,eb->ecab", I3, rel_e) / r_eI[:, None, None, None]
        + jnp.einsum("ec,ea,eb->ecab", rel_e, rel_e, rel_e) / (r_eI**3)[:, None, None, None]
    )
    vhat = grid / jnp.linalg.norm(grid, axis=1)[:, None]
    dcth_s = jnp.einsum("ecab,gc->egab", du, vhat)  # (Ne,Nv,3,3)

    # PERIODIC Pulay stress (cell terms carried): det-only via the Jastrow-stripped ham (P1), full ham
    # (P2).  The base uses the shared running inverse; the mesh configs re-evaluate at displaced coords.
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import Jastrow_data
    from jqmc.wavefunction import Wavefunction_data

    bare_jastrow = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=None,
        jastrow_dtn_data=None,
    )
    bare_wavefunction = Wavefunction_data(
        jastrow_data=bare_jastrow,
        geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
    )
    ham0 = Hamiltonian_data(
        structure_data=hamiltonian_data.structure_data,
        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
        wavefunction_data=bare_wavefunction,
    )
    S_det_base = pbc_stress.stress_lnPsi_pbc(ham0, ru0, rd0, take_real=False, geminal_inv=base_inv)
    S_det_mesh = _per_config(
        lambda ru, rd, gi: pbc_stress.stress_lnPsi_pbc(ham0, ru, rd, take_real=False, geminal_inv=gi)
    ).reshape(Ne, Nv, 3, 3)
    S_full_base = pbc_stress.stress_lnPsi_pbc(hamiltonian_data, ru0, rd0, take_real=False, geminal_inv=base_inv)
    S_full_mesh = _per_config(
        lambda ru, rd, gi: pbc_stress.stress_lnPsi_pbc(hamiltonian_data, ru, rd, take_real=False, geminal_inv=gi)
    ).reshape(Ne, Nv, 3, 3)
    # Fixed directions: the moved electron sits at F R + |F rel| v_g = F r'_g - r_eI eps v_g, so the
    # affine config stress is corrected by -r_eI O(r'_g)[e] (x) v_g.
    dln_det = (
        (S_det_mesh - S_det_base[None, None])
        + drel[:, None, :, :] * vg_det[:, :, None, None]
        - jnp.einsum("e,egc,gd->egcd", r_eI, Oee_det, grid)
    )
    dln_full = (
        (S_full_mesh - S_full_base[None, None])
        + drel[:, None, :, :] * vg_full[:, :, None, None]
        - jnp.einsum("e,egc,gd->egcd", r_eI, c["Oee"], grid)
    )

    coef_tg = twol1[:, :, None] * weights[None, None, :]
    dVgrid = coef_tg[..., None, None] * (
        (Vlp_t[:, :, None] * Pl)[..., None, None] * drel[:, None, None, :, :]
        + (Vl_t[:, :, None] * dPl)[..., None, None] * dcth_s[:, None, :, :, :]
    )  # (Ne,T,Nv,3,3)
    term1 = jnp.einsum("etgab,eg->ab", dVgrid, rat_ang)  # ANG geometry, delta-ratio-weighted
    term2 = jnp.einsum("eg,egab->ab", W1, dln_det) - jnp.einsum("eg,egab->ab", W2, dln_full)  # wf-ratio Pulay
    return jnp.real(term1 + term2)


# --------------------------------------------------------------------------- #
# Run the GFMC measurement-force and stress kernels under a cached ``jax.jit``
# for eager callers. Without the outer JIT, each JAX operation in the derivative
# can compile separately; production already runs the same graph through its
# ``jit(vmap)`` bridge with the Hamiltonian and lattice spacing traced.
#
# The SCOPE / capability guards (``_assert_scope`` + the dltmove-PBC and molecular-ECP-DTN-stress
# NotImplementedError raises) stay OUTSIDE jit / at Python level (they read ham STRUCTURE and raise,
# so they cannot be traced -- and out-of-scope configs must never attempt a trace); the jit boundary
# is AROUND the numeric ``_num`` body only.  The STATIC (graph-shaping) ``non_local_move`` string is
# captured via the lru_cache key so repeated eager calls REUSE the compiled
# program (2nd call = 0 compiles); the ham pytree / r_up / r_dn / RT / alat are
# TRACED jit arguments (the ham's static AO leaves live in the treedef, so distinct AO/pbc/ecp
# configs retrace automatically -- exactly the production jit(vmap) bridge).
# --------------------------------------------------------------------------- #
@cache
def _jitted_gfmc_kernel(
    which: GFMCQuantity,
    non_local_move: NonLocalMove | None,
) -> GFMCNumericKernel:
    """Cached JIT wrapper for a numeric GFMC measurement-derivative kernel."""
    impl = {"pos": _grad_e_L_wrt_positions_num, "strain": _grad_e_L_wrt_strain_num, "both": _grad_e_L_wrt_both_num}[which]

    def _f(
        hamiltonian_data: Hamiltonian_data,
        r_up: jax.Array,
        r_dn: jax.Array,
        RT: jax.Array,
        alat: jax.Array,
        geminal_inv: jax.Array | None,
    ) -> GFMCGradientResult:
        # ``geminal_inv`` is a TRACED jit arg (an array or None) -- NOT part of the lru_cache key.
        return impl(
            hamiltonian_data,
            r_up,
            r_dn,
            RT,
            alat,
            non_local_move=non_local_move,
            geminal_inv=geminal_inv,
        )

    return jax.jit(_f)


def _dispatch_gfmc(
    which: GFMCQuantity,
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array,
    alat: jax.Array,
    non_local_move: NonLocalMove | None,
    geminal_inv: jax.Array | None = None,
) -> GFMCGradientResult:
    """Dispatch a numeric GFMC measurement derivative through the cached JIT wrapper."""
    from . import _capability

    impl = {"pos": _grad_e_L_wrt_positions_num, "strain": _grad_e_L_wrt_strain_num, "both": _grad_e_L_wrt_both_num}[which]
    if not _capability.jit_ao_serveable(hamiltonian_data):
        return impl(
            hamiltonian_data,
            r_up,
            r_dn,
            RT,
            alat,
            non_local_move=non_local_move,
            geminal_inv=geminal_inv,
        )
    return _jitted_gfmc_kernel(which, non_local_move)(hamiltonian_data, r_up, r_dn, RT, alat, geminal_inv)


def grad_e_L_wrt_strain(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    alat: jnpt.ArrayLike,
    non_local_move: NonLocalMove = "tmove",
    geminal_inv: jnpt.ArrayLike | None = None,
) -> jax.Array:
    """Shared GFMC/LRDMC HF stress of the novar=0 measurement local energy."""
    # Periodic-Gamma dltmove plus Jastrow stress is analytic: the r/R delta
    # branches use the min-image ECP force geometry and the strain delta uses
    # ``_dltmove_delta_strain_pbc``. Molecular ECP+DTN stress is also closed form:
    # ``_stress_eL``, ``grad_V_ecp_wrt_strain_mol``, and
    # ``grad_T_dtn_wrt_strain`` compose with the analytic S-prime correction.
    return _dispatch_gfmc(
        "strain",
        hamiltonian_data,
        r_up,
        r_dn,
        RT,
        alat,
        non_local_move,
        geminal_inv=geminal_inv,
    )


def _grad_e_L_wrt_strain_num(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array,
    alat: jax.Array,
    non_local_move: NonLocalMove = "tmove",
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """Numeric SHARED GFMC/LRDMC HF stress body -- jit-served via :func:`_jitted_gfmc_kernel`.  The
    scope / capability guards live in the public estimator wrappers (:func:`grad_e_L_t_wrt_strain` /
    :func:`grad_e_L_n_wrt_strain`; they read ham STRUCTURE and raise, so they cannot be traced).

    ``geminal_inv`` (fast path): the driver's running SM ``G^{-1}`` at the BASE walker positions,
    threaded into every base-position geminal reverse so the kernel lowers with NO fresh SVD-pinv.
    """
    # (1) continuous with-Jastrow local-energy stress: d e_L^cont / d eps.
    stress = jnp.asarray(assemble.compute_grad_e_L(hamiltonian_data, r_up, r_dn, RT, wrt="eps", geminal_inv=geminal_inv))

    # novar=0 split: forces/stress differentiate the MEASUREMENT (continuum)
    # local energy. The S'-only terms (d(eid-ei), d S_i, -d Tc_i on the
    # sign-flip clip set) belong to the clipped PROPAGATION composition and
    # are not differentiated; the mixed-average estimator has no dw/dX term,
    # so no clipped-H derivative enters anywhere (steps (2)-(4) are empty).

    # (5) dltmove ECP non-local Jastrow-reweight correction  d(delta)/d eps (ECP + Jastrow).
    # Molecular: ``ecp_force.grad_V_ecp_nl_dltmove(wrt="eps")`` (affine-virial wf-ratio Pulay).
    # Periodic-Gamma: :func:`_dltmove_delta_strain_pbc` (the same delta but with the FULL periodic Pulay
    # stress carrying the cell terms -- the molecular affine virial is wrong for PBC).
    if _dltmove_delta_active(hamiltonian_data, non_local_move):
        if _is_pbc(hamiltonian_data):
            stress = stress + _dltmove_delta_strain_pbc(hamiltonian_data, r_up, r_dn, RT, geminal_inv=geminal_inv)
        else:
            from . import ecp_force

            stress = stress + jnp.asarray(ecp_force.grad_V_ecp_nl_dltmove(hamiltonian_data, r_up, r_dn, RT, wrt="eps"))

    return stress


def _grad_e_L_wrt_both_num(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array,
    alat: jax.Array,
    non_local_move: NonLocalMove = "tmove",
    geminal_inv: jax.Array | None = None,
) -> PositionAndStrainGradients:
    """Numeric fused force and stress body for the continuous measurement energy."""
    nu = int(r_up.shape[0])
    pbc = _is_pbc(hamiltonian_data)

    # (1) continuous local-energy force + stress in one program, enabling CSE.
    dR = jnp.asarray(assemble.compute_grad_e_L(hamiltonian_data, r_up, r_dn, RT, wrt="R", geminal_inv=geminal_inv))
    dru, drd = assemble.compute_grad_e_L(hamiltonian_data, r_up, r_dn, RT, wrt="r", geminal_inv=geminal_inv)
    dr = jnp.concatenate([jnp.asarray(dru), jnp.asarray(drd)], axis=0)
    stress = jnp.asarray(assemble.compute_grad_e_L(hamiltonian_data, r_up, r_dn, RT, wrt="eps", geminal_inv=geminal_inv))

    # novar=0 split: forces/stress differentiate the MEASUREMENT (continuum)
    # local energy. The S'-only terms belong to the clipped PROPAGATION
    # composition and are not differentiated; the mixed-average estimator
    # has no dw/dX term, so no clipped-H derivative enters (steps (2)-(4) are empty).

    # (5) dltmove ECP non-local Jastrow-reweight correction.
    if _dltmove_delta_active(hamiltonian_data, non_local_move):
        from . import ecp_force

        dR = dR + jnp.asarray(ecp_force.grad_V_ecp_nl_dltmove(hamiltonian_data, r_up, r_dn, RT, wrt="R"))
        dr = dr + jnp.asarray(ecp_force.grad_V_ecp_nl_dltmove(hamiltonian_data, r_up, r_dn, RT, wrt="r"))
        if pbc:
            stress = stress + _dltmove_delta_strain_pbc(hamiltonian_data, r_up, r_dn, RT, geminal_inv=geminal_inv)
        else:
            stress = stress + jnp.asarray(ecp_force.grad_V_ecp_nl_dltmove(hamiltonian_data, r_up, r_dn, RT, wrt="eps"))

    return dR, dr[:nu], dr[nu:], stress


def grad_e_L_wrt_positions_and_strain(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    alat: jnpt.ArrayLike,
    non_local_move: NonLocalMove = "tmove",
    geminal_inv: jnpt.ArrayLike | None = None,
) -> PositionAndStrainGradients:
    """Fused force and stress for the shared novar=0 measurement local energy."""
    return _dispatch_gfmc(
        "both",
        hamiltonian_data,
        r_up,
        r_dn,
        RT,
        alat,
        non_local_move,
        geminal_inv=geminal_inv,
    )


def grad_e_L_t_wrt_positions_and_strain(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    alat: jnpt.ArrayLike,
    non_local_move: NonLocalMove = "tmove",
    geminal_inv: jnpt.ArrayLike | None = None,
) -> PositionAndStrainGradients:
    """GFMC_t fused force and stress of the novar=0 measurement local energy."""
    _assert_scope(hamiltonian_data)
    _validate_non_local_move(hamiltonian_data, non_local_move)
    return grad_e_L_wrt_positions_and_strain(
        hamiltonian_data,
        r_up,
        r_dn,
        RT,
        alat,
        non_local_move=non_local_move,
        geminal_inv=geminal_inv,
    )


def grad_e_L_wrt_positions(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    alat: jnpt.ArrayLike,
    non_local_move: NonLocalMove = "tmove",
    geminal_inv: jnpt.ArrayLike | None = None,
) -> PositionGradients:
    """Shared GFMC/LRDMC HF force of the novar=0 measurement local energy."""
    # Periodic-Gamma dltmove plus Jastrow force is analytic: the closed-form
    # ``grad_V_ecp_nl_dltmove`` r/R branches use min-image periodic geometry.
    return _dispatch_gfmc(
        "pos",
        hamiltonian_data,
        r_up,
        r_dn,
        RT,
        alat,
        non_local_move,
        geminal_inv=geminal_inv,
    )


def _grad_e_L_wrt_positions_num(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array,
    alat: jax.Array,
    non_local_move: NonLocalMove = "tmove",
    geminal_inv: jax.Array | None = None,
) -> PositionGradients:
    """Numeric HF force body for the continuous measurement local energy."""
    nu = int(r_up.shape[0])

    # (1) continuous with-Jastrow local-energy force: d e_L^cont / d{R, r}.
    dR = jnp.asarray(assemble.compute_grad_e_L(hamiltonian_data, r_up, r_dn, RT, wrt="R", geminal_inv=geminal_inv))
    dru, drd = assemble.compute_grad_e_L(hamiltonian_data, r_up, r_dn, RT, wrt="r", geminal_inv=geminal_inv)
    dr = jnp.concatenate([jnp.asarray(dru), jnp.asarray(drd)], axis=0)

    # novar=0 split: forces/stress differentiate the MEASUREMENT (continuum)
    # local energy. The S'-only terms belong to the clipped PROPAGATION
    # composition and are not differentiated; the mixed-average estimator
    # has no dw/dX term, so no clipped-H derivative enters (steps (2)-(4) are empty).

    # (5) dltmove ECP non-local Jastrow-reweight correction  d(delta)/dX (molecular ECP + Jastrow).
    # ``e_L^dlt = e_L^t + delta`` (delta = sum_g max(V^det_g, 0)(1 - jr_g)); its closed-form force is
    # the difference of two SP-masked ECP-non-local reductions (det-only P1 minus full P2).  jnone /
    # tmove -> no correction (delta == 0).
    if _dltmove_delta_active(hamiltonian_data, non_local_move):
        from . import ecp_force

        dR = dR + jnp.asarray(ecp_force.grad_V_ecp_nl_dltmove(hamiltonian_data, r_up, r_dn, RT, wrt="R"))
        dr = dr + jnp.asarray(ecp_force.grad_V_ecp_nl_dltmove(hamiltonian_data, r_up, r_dn, RT, wrt="r"))

    return dR, dr[:nu], dr[nu:]


# ---------------------------------------------------------------------------- #
# GFMC_t public entry points: scope-guarded delegates to the shared
# novar=0 measurement derivative.
# ---------------------------------------------------------------------------- #
def grad_e_L_t_wrt_positions(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    alat: jnpt.ArrayLike,
    non_local_move: NonLocalMove = "tmove",
    geminal_inv: jnpt.ArrayLike | None = None,
) -> PositionGradients:
    """GFMC_t HF force of the novar=0 measurement local energy."""
    _assert_scope(hamiltonian_data)
    _validate_non_local_move(hamiltonian_data, non_local_move)
    return grad_e_L_wrt_positions(
        hamiltonian_data,
        r_up,
        r_dn,
        RT,
        alat,
        non_local_move=non_local_move,
        geminal_inv=geminal_inv,
    )


def grad_e_L_t_wrt_strain(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    alat: jnpt.ArrayLike,
    non_local_move: NonLocalMove = "tmove",
    geminal_inv: jnpt.ArrayLike | None = None,
) -> jax.Array:
    """GFMC_t HF stress of the novar=0 measurement local energy."""
    _assert_scope(hamiltonian_data)
    _validate_non_local_move(hamiltonian_data, non_local_move)
    return grad_e_L_wrt_strain(
        hamiltonian_data,
        r_up,
        r_dn,
        RT,
        alat,
        non_local_move=non_local_move,
        geminal_inv=geminal_inv,
    )


# ---------------------------------------------------------------------------- #
# GFMC_n / LRDMC number-estimator (a.k.a. "n" / lrdmc-bra).
#
# CRUX (measured against the source, jqmc_gfmc.py):  GFMC_n and GFMC_t differ ONLY at the
# DRIVER level, in how the branching weight ``w_L`` is formed -- GFMC_t projects with the
# continuous-time factor built directly from ``w_L`` (see the "GFMC_t uses w_L directly (no
# V_diag_E division unlike GFMC_n)" note near jqmc_gfmc.py:2286), whereas GFMC_n uses the
# fixed-``n`` number projection with a ``V_diag / E`` division.  The differentiated
# Hellmann--Feynman LOCAL ENERGY is *identical*:
#
#   * ``jqmc_gfmc._compute_V_elements_n`` (the ``_n`` forward, use_fast_update=False path)
#     is line-for-line ``_compute_V_elements_t`` -- same ``V_diag`` (3N/2a^2 + discretised
#     bare Coulomb + ECP-local + kinetic-SP + ECP-SP) and same ``V_nondiag`` (fixed-node
#     kinetic + ECP non-local), so ``_compute_local_energy_n(..., use_fast_update=False) ==
#     _compute_local_energy_t``.
#   * The ``use_fast_update=True`` branch only swaps the geminal inverse for an SVD
#     pseudo-inverse in the hot move loop; it is VALUE-identical and is NOT the path the
#     force/stress differentiate -- the production GFMC_n force kernel is invoked with
#     ``use_fast_update=False`` (jqmc_gfmc.py:7012-7021) and the strained-energy stress
#     kernel hard-codes ``use_fast_update=False`` (jqmc_gfmc.py:6776).
#
# Both estimators measure the same continuous local energy, so GFMC_n delegates
# to the same derivative implementation as GFMC_t. Scope is guarded here.
# ---------------------------------------------------------------------------- #
def grad_e_L_n_wrt_positions_and_strain(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    alat: jnpt.ArrayLike,
    non_local_move: NonLocalMove = "tmove",
    geminal_inv: jnpt.ArrayLike | None = None,
) -> PositionAndStrainGradients:
    """GFMC_n fused force and stress of the novar=0 measurement local energy."""
    _assert_scope(hamiltonian_data)
    _validate_non_local_move(hamiltonian_data, non_local_move)
    return grad_e_L_wrt_positions_and_strain(
        hamiltonian_data,
        r_up,
        r_dn,
        RT,
        alat,
        non_local_move=non_local_move,
        geminal_inv=geminal_inv,
    )


def grad_e_L_n_wrt_positions(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    alat: jnpt.ArrayLike,
    non_local_move: NonLocalMove = "tmove",
    geminal_inv: jnpt.ArrayLike | None = None,
) -> PositionGradients:
    """GFMC_n HF force of the novar=0 measurement local energy."""
    _assert_scope(hamiltonian_data)
    _validate_non_local_move(hamiltonian_data, non_local_move)
    return grad_e_L_wrt_positions(
        hamiltonian_data,
        r_up,
        r_dn,
        RT,
        alat,
        non_local_move=non_local_move,
        geminal_inv=geminal_inv,
    )


def grad_e_L_n_wrt_strain(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    alat: jnpt.ArrayLike,
    non_local_move: NonLocalMove = "tmove",
    geminal_inv: jnpt.ArrayLike | None = None,
) -> jax.Array:
    """GFMC_n HF stress of the novar=0 measurement local energy."""
    _assert_scope(hamiltonian_data)
    _validate_non_local_move(hamiltonian_data, non_local_move)
    return grad_e_L_wrt_strain(
        hamiltonian_data,
        r_up,
        r_dn,
        RT,
        alat,
        non_local_move=non_local_move,
        geminal_inv=geminal_inv,
    )
