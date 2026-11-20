"""Production parity for GFMC/LRDMC projection as a state machine, XLA vs Rust.

Scope of this file (distinct from the neighbouring GFMC suites):

* ``test_gfmc.py`` pins the DIAGONAL e_L kernels at one configuration.
* ``test_gfmc_chain_modes.py`` pins the DRIVER-level chain-mode gate + statistics.
* THIS file walks the projection SUB-STEP BY SUB-STEP with the randomness pinned:
  discretized kinetic elements -> sign-flip determination -> S' (the SP fold) ->
  branch weight -> non-local ECP move (tmove / DLA-``dltmove``) -> the walker /
  weight / tau state as it stands BEFORE reconfiguration, for a multi-step
  interval, and forces every branch of that machine to be reached by a test.

Reference (XLA is the reference).  ``_projection_t_core`` /
``_projection_n`` are closures built inside ``GFMC_{t,n}.run()``, so they cannot be
imported; the module-level helpers below transcribe those closures ONE-TO-ONE onto
the SAME XLA kernels the closures call (``compute_discretized_kinetic_energy``,
``compute_kinetic_energy_all_elements``, the coulomb dispatches,
``compute_ecp_non_local_parts_nearest_neighbors``, ``compute_Jastrow_part``) -- the
established convention of ``test_gfmc.py``, extended from "one diagonal e_L" to
"the whole step transition + the interval loop".  The transcription is itself
pinned against the production XLA driver by
``test_projection_reference_machine_matches_production_driver``: if the
transcription ever drifts from ``_projection_t_core`` that test goes red, so the
oracle cannot silently soften.

Randomness is pinned by construction: the native interval drivers replay jqmc's
``_split_step_keys`` threefry stream, so the reference machine draws its uniforms
from ``jax.random.split`` / ``jax.random.uniform`` on the SAME key and every
sub-step decision (escape time, categorical hop) is deterministic on both sides.
``random_discretized_mesh`` is covered both off (R^T = I) and on.  The injected
GFMC_n cases pin an arbitrary non-identity matrix independently of the sampler;
the native-random cases additionally cover the production Threefry-to-Haar route
and the projection-scoped fast caches under distinct kinetic/ECP orientations.

GFMC_t 2-mode note: GFMC_t is fast-only by
design because its escape-time sub-step COUNT is fp-dependent.  That is NOT tested
as a defect here; what IS tested is that at a pinned key the native GFMC_t interval
reproduces the reference machine's sub-step count, weight and walker -- which it
does on every cell below.
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
from jqmc._setting import ATOL_NONVACUITY_DISTINCTION, RTOL_NONVACUITY_DISTINCTION

import contextlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import precision_policy_snapshot

pytestmark = [pytest.mark.with_chdir]

from jqmc._precision import get_tolerance_min, get_tolerance_xla_rust_consistency
from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_ZERO
from jqmc.coulomb_potential import (
    compute_coulomb_potential_ee_dispatch as _ee_d,
    compute_coulomb_potential_ei_discretized_dispatch as _eid_d,
    compute_coulomb_potential_ei_dispatch as _ei_d,
    compute_coulomb_potential_ii_dispatch as _ii_d,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors as _nn_ecp,
    icosahedron_sym_mesh_Nv12,
    NN_default,
)
from jqmc.jastrow_factor import compute_Jastrow_part
from jqmc.jqmc_gfmc import (
    _compute_V_elements_t,
    _generate_rotation_matrix_t,
    _projection_t_core,
    _sample_lrdmc_rotation_matrix,
    _split_lrdmc_rotation_keys,
    compute_geminal_all_elements,
    get_eps,
)
from jqmc.wavefunction import (
    compute_discretized_kinetic_energy,
    compute_kinetic_energy_all_elements,
    compute_kinetic_energy_all_elements_fast_update,
)

# jit-wrapped production single-substep core. static: arg 12
# random_discretized_mesh (bool) and arg 13 non_local_move (str). mpc defaults
# (False/None/None) are baked as trace-time constants -- MPC off.
_jit_projection_t_core = jax.jit(_projection_t_core, static_argnums=(12, 13))

from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel.conftest import assert_native_ran, assert_no_native_ran, native_ctx_spy

# Precision zones. The projection step starts at AO values, then crosses the
# discretized kinetic ratios (``det_ratio``), continuum kinetic, Coulomb/ECP
# diagonal and e_L assembly; periodic rows add Ewald.  ``ao_eval`` must remain
# explicit even without Jastrow: it is fp32 in mixed mode and feeds every
# determinant ratio/local-energy aggregate in this module.
_EL_ZONES = ("ao_eval", "wf_kinetic", "coulomb", "local_energy", "det_ratio")
_EL_ZONES_J = _EL_ZONES + ("jastrow_eval", "jastrow_grad_lap")
_EL_ZONES_PBC = _EL_ZONES_J + ("ewald",)

_MESH_W = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
_MESH_GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Configuration builders
# --------------------------------------------------------------------------- #
def _cfg(ham, seed: int, scale: float = 0.6, cell_shift=None):
    """Deterministic (r_up, r_dn). Periodic hams draw inside the cell; ``cell_shift``
    (integer lattice translations) produces the FAR / unwrapped-PBC configurations."""
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    if bool(ham.structure_data.pbc_flag):
        cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
        ru = np.asarray(rng.random((nu, 3)) @ cell, dtype=np.float64)
        rd = np.asarray(rng.random((nd, 3)) @ cell, dtype=np.float64)
        if cell_shift is not None:
            ru = ru + np.asarray(cell_shift, dtype=np.float64) @ cell
            rd = rd + np.asarray(cell_shift, dtype=np.float64) @ cell
        return ru, rd
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    iu = rng.integers(0, R.shape[0], nu)
    idn = rng.integers(0, R.shape[0], nd)
    return (
        R[iu] + rng.normal(scale=scale, size=(nu, 3)),
        R[idn] + rng.normal(scale=scale, size=(nd, 3)),
    )


def _rt_stack(n: int, seed: int = 5):
    """A deterministic stack of ``n`` non-identity mesh rotations ``R^T`` (the
    ``random_discretized_mesh`` degree of freedom, supplied as the INJECTED
    bitexact-mode stack so the XLA trig never enters the comparison)."""
    rng = np.random.default_rng(seed)

    def _rx(t):
        return np.array([[1.0, 0.0, 0.0], [0.0, np.cos(t), -np.sin(t)], [0.0, np.sin(t), np.cos(t)]])

    def _ry(t):
        return np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])

    def _rz(t):
        return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])

    out = []
    for _ in range(n):
        a, b, c = rng.uniform(-np.pi, np.pi, 3)
        out.append((_rz(c) @ _ry(b) @ _rx(a)).T)
    return np.asarray(out, dtype=np.float64)


def _ctx(ham, complex_: bool = False):
    """The production ``NativeObsCtx`` the GFMC drivers build (``want_gfmc=True``)."""
    wf = ham.wavefunction_data
    return jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        complex_,
        bool(ham.structure_data.pbc_flag),
        False,
        want_force=False,
        want_gfmc=True,
        precision_policy=precision_policy_snapshot(),
    )


# --------------------------------------------------------------------------- #
# XLA reference machine: one projection SUB-STEP (transcription of
# jqmc_gfmc.py ``_projection_t_core`` / ``_projection_n``'s shared body).
# --------------------------------------------------------------------------- #
def _xla_step(ham, r_up, r_dn, alat, RT, non_local_move, ecp_RT=None):
    """Return every intermediate of ONE projection sub-step, computed with XLA kernels.

    Keys: ``elements`` (6N discretized kinetic elements), ``sign_flip`` (per-electron
    ``any(el >= 0)``), ``sp_kin`` / ``sp_ecp`` (the S' fold), ``nondiag`` (per-electron
    ``sum_6 (el + 1/(2 alat^2))``), ``e_L``, ``b_L`` (``non_diagonal_sum_hamiltonian``),
    ``p_list`` (FN move amplitudes, kinetic ++ ECP), ``mesh_up`` / ``mesh_dn``.
    """
    wf = ham.wavefunction_data
    cd = ham.coulomb_potential_data
    if ecp_RT is None:
        ecp_RT = RT
    nu, nd = len(r_up), len(r_dn)

    mesh_u, mesh_d, el = compute_discretized_kinetic_energy(alat, wf, r_up, r_dn, RT)
    el = np.asarray(el).real
    mesh_u = np.asarray(mesh_u)
    mesh_d = np.asarray(mesh_d)
    fn = np.minimum(el, 0.0)
    sp_kin = float(np.maximum(el, 0.0).sum())
    ela = el.reshape(-1, 6)
    nondiag = (ela + 1.0 / (2.0 * alat**2)).sum(1)
    sflip = np.any(ela >= 0.0, axis=1)

    _kc = compute_kinetic_energy_all_elements(wf, r_up, r_dn)
    if isinstance(_kc, tuple):
        kc = np.concatenate([np.asarray(_kc[0]).ravel(), np.asarray(_kc[1]).ravel()]).real
    else:
        kc = np.asarray(_kc).ravel().real

    elel = float(np.real(_ee_d(coulomb_potential_data=cd, r_up_carts=r_up, r_dn_carts=r_dn)))
    iiv = float(np.real(_ii_d(coulomb_potential_data=cd)))
    ei_u, ei_d = _ei_d(coulomb_potential_data=cd, r_up_carts=r_up, r_dn_carts=r_dn)
    ei = np.concatenate([np.asarray(ei_u), np.asarray(ei_d)]).real
    zv = ei + kc - nondiag
    if cd.ecp_flag:
        # parcutg=1: no alat truncation for ECP -> the max-arg is the BARE el-ion.
        ei_arg = ei
    else:
        d_u, d_d = _eid_d(coulomb_potential_data=cd, r_up_carts=r_up, r_dn_carts=r_dn, alat=alat)
        ei_arg = np.concatenate([np.asarray(d_u), np.asarray(d_d)]).real
    opt = np.where(sflip, np.maximum(zv, ei_arg), zv)
    diag = 3.0 / alat**2 * (nu + nd) + (elel + iiv + float(opt.sum()))

    if cd.ecp_flag:
        ecp_local = float(
            np.real(compute_ecp_local_parts_all_pairs(coulomb_potential_data=cd, r_up_carts=r_up, r_dn_carts=r_dn))
        )
        det_only = non_local_move == "dltmove"
        m_u, m_d, v_nl, _ = _nn_ecp(cd, wf, r_up, r_dn, ecp_RT, flag_determinant_only=det_only)
        v_nl = np.asarray(v_nl).real
        m_u = np.asarray(m_u)
        m_d = np.asarray(m_d)
        v_fn = np.minimum(v_nl, 0.0)
        sp_ecp = float(np.maximum(v_nl, 0.0).sum())
        if det_only:
            # DLA: SP folds the DETERMINANT-only amplitude; only the FN off-diagonal
            # is multiplied by the separate Jastrow ratio (jqmc_gfmc.py:1065).
            jd = wf.jastrow_data
            j_base = float(compute_Jastrow_part(jd, r_up, r_dn))
            ratio = np.array(
                [float(np.exp(float(compute_Jastrow_part(jd, m_u[m], m_d[m])) - j_base)) for m in range(len(v_nl))]
            )
            v_fn = v_fn * ratio
        p_list = np.concatenate([fn.ravel(), v_fn.ravel()])
        mesh_up = np.concatenate([mesh_u, m_u], axis=0)
        mesh_dn = np.concatenate([mesh_d, m_d], axis=0)
    else:
        sp_ecp = 0.0
        p_list = fn.ravel()
        mesh_up, mesh_dn = mesh_u, mesh_d

    # e_L / b_L from the PRODUCTION closure the driver's projection step runs:
    # _compute_V_elements_t returns (V_diag, V_nondiag) with
    # V_nondiag == b_L (the FN off-diagonal sum) and e_L == V_diag + V_nondiag. The
    # leaf intermediates above (elements, sign_flip, S', the FN move mesh / p_list)
    # remain the leaf references the Rust component recovery and the categorical hop
    # replay are pinned against; the diagonal e_L assembly is not transcribed.
    V_diag, V_nondiag, _ = _compute_V_elements_t(ham, r_up, r_dn, RT, non_local_move, alat, ecp_RT)
    e_L = float(np.real(V_diag) + np.real(V_nondiag))
    b_L = float(np.real(V_nondiag))

    # novar=0 split: the MEASUREMENT (continuum) local energy -- bare el-ion +
    # continuum kinetic, no lattice terms, no clip; ECP nonlocal kept as-is.
    e_L_meas = elel + iiv + float(np.sum(ei + kc))
    if cd.ecp_flag:
        e_L_meas += ecp_local + sp_ecp + float(np.sum(v_fn))

    return {
        "elements": el,
        "sign_flip": sflip,
        "sp_kin": sp_kin,
        "sp_ecp": sp_ecp,
        "nondiag": nondiag,
        "e_L": e_L,
        "e_L_meas": e_L_meas,
        "b_L": b_L,
        "p_list": p_list,
        "mesh_up": mesh_up,
        "mesh_dn": mesh_dn,
    }


def _hop(step, u):
    """jqmc's categorical FN hop: ``k = searchsorted(cumsum(p/p.sum()), u)`` with JAX's
    clamp-at-index semantics; returns ``(k, new_r_up, new_r_dn)``."""
    p = step["p_list"]
    cdf = np.cumsum(p / p.sum())
    k = min(int(np.searchsorted(cdf, u)), len(cdf) - 1)
    return k, np.asarray(step["mesh_up"][k]), np.asarray(step["mesh_dn"][k])


def _uniform(key):
    return float(jax.random.uniform(key, minval=0.0, maxval=1.0))


def _xla_interval_n(
    ham,
    r_up,
    r_dn,
    alat,
    e_scf,
    num_per_meas,
    key,
    non_local_move,
    rts=None,
    *,
    couple_ecp_to_kinetic=False,
):
    """GFMC_n interval: ``num_per_meas`` fixed sub-steps of ``w_L *= -b_L/(V_diag - E_scf)``
    + FN hop, then the final-config measurement at the LAST sub-step's rotation."""
    w_L = 1.0
    trace = []
    last_rt = np.eye(3)
    interval_ecp_rt = np.eye(3)
    for i in range(num_per_meas):
        key, rot = jax.random.split(key)
        key, mv = jax.random.split(key)
        rt = np.eye(3) if rts is None else rts[i]
        if rts is not None and i == 0:
            _, ecp_key = _split_lrdmc_rotation_keys(rot)
            interval_ecp_rt = rt if couple_ecp_to_kinetic else np.asarray(_sample_lrdmc_rotation_matrix(ecp_key, True).T)
        last_rt = rt
        st = _xla_step(ham, r_up, r_dn, alat, rt, non_local_move, interval_ecp_rt)
        w_L *= 1.0 / ((st["e_L"] - st["b_L"]) - e_scf) * (-st["b_L"])
        k, r_up, r_dn = _hop(st, _uniform(mv))
        trace.append(
            {"e_L": st["e_L"], "b_L": st["b_L"], "w_L": w_L, "k": k, "r_up": r_up.copy(), "r_dn": r_dn.copy(), "step": st}
        )
    fin = _xla_step(ham, r_up, r_dn, alat, last_rt, non_local_move, interval_ecp_rt)
    return {
        "w_L": w_L,
        # novar=0 split: measurement e_L; V_diag keeps the propagation composition.
        "e_L": fin["e_L_meas"],
        "e_L_propagation": fin["e_L"],
        "V_diag": fin["e_L"] - fin["b_L"],
        "r_up": r_up,
        "r_dn": r_dn,
        "key": key,
        "trace": trace,
    }


def _xla_interval_t(
    ham,
    r_up,
    r_dn,
    alat,
    tau,
    key,
    non_local_move,
    random_mesh=False,
    *,
    couple_ecp_to_kinetic=False,
):
    """GFMC_t interval: escape-time loop ``tau_update = min(tau_left, ln(1-xi)/b_L)``,
    ``w_L *= exp(-tau_update e_L)``, FN hop only while projection time remains."""
    w_L = 1.0
    tau_left = tau
    n_sub = 0
    e_last = 0.0
    trace = []
    interval_rt = np.eye(3)
    interval_ecp_rt = np.eye(3)
    while tau_left > 0.0:
        n_sub += 1
        key, rot = jax.random.split(key)
        if random_mesh and n_sub == 1:
            kinetic_key, ecp_key = _split_lrdmc_rotation_keys(rot)
            interval_rt = np.asarray(_sample_lrdmc_rotation_matrix(kinetic_key, True).T)
            interval_ecp_rt = (
                interval_rt if couple_ecp_to_kinetic else np.asarray(_sample_lrdmc_rotation_matrix(ecp_key, True).T)
            )
        st = _xla_step(ham, r_up, r_dn, alat, interval_rt, non_local_move, interval_ecp_rt)
        e_last = st["e_L_meas"]
        key, xik = jax.random.split(key)
        tau_update = min(tau_left, float(np.log(1.0 - _uniform(xik)) / st["b_L"]))
        w_L *= float(np.exp(-tau_update * st["e_L"]))
        tau_left -= tau_update
        key, mv = jax.random.split(key)
        u = _uniform(mv)  # drawn unconditionally (jqmc split order), consumed only if tau remains
        if tau_left > 0.0:
            k, r_up, r_dn = _hop(st, u)
        else:
            k = None
        trace.append({"e_L": st["e_L"], "b_L": st["b_L"], "w_L": w_L, "tau_left": tau_left, "k": k, "step": st})
    return {"w_L": w_L, "e_L": e_last, "r_up": r_up, "r_dn": r_dn, "key": key, "n_substeps": n_sub, "trace": trace}


# --------------------------------------------------------------------------- #
# COMPLEX (Bloch twist) reference machine: identical structure, everything kept
# complex (JAX-lexicographic min/max/>=) with the fixed-phase Re-projection at
# exactly the three real decision points of ``_projection_t_core``.
# --------------------------------------------------------------------------- #
def _xla_step_complex(ham, r_up, r_dn, alat, RT, non_local_move="tmove"):
    wf = ham.wavefunction_data
    cd = ham.coulomb_potential_data
    nu, nd = len(r_up), len(r_dn)
    mesh_u, mesh_d, el = compute_discretized_kinetic_energy(alat, wf, r_up, r_dn, RT)
    el = jnp.asarray(el).ravel()
    fn = jnp.minimum(el, 0.0)
    sp = jnp.sum(jnp.maximum(el, 0.0))
    ela = el.reshape(-1, 6)
    nondiag = jnp.sum(ela + 1.0 / (2.0 * alat**2), axis=1)
    sflip = jnp.any(ela >= 0, axis=1)
    kc = jnp.asarray(compute_kinetic_energy_all_elements(wf, r_up, r_dn)).ravel()
    ei_u, ei_d = _ei_d(coulomb_potential_data=cd, r_up_carts=r_up, r_dn_carts=r_dn)
    ei = jnp.concatenate([jnp.asarray(ei_u), jnp.asarray(ei_d)])
    zv = ei + kc - nondiag
    opt = jnp.where(sflip, jnp.maximum(zv, ei), zv)
    elel = complex(_ee_d(coulomb_potential_data=cd, r_up_carts=r_up, r_dn_carts=r_dn))
    iiv = complex(_ii_d(coulomb_potential_data=cd))
    diag = 3.0 / alat**2 * (nu + nd) + (elel + iiv + complex(jnp.sum(opt)))
    ecp_local = complex(compute_ecp_local_parts_all_pairs(coulomb_potential_data=cd, r_up_carts=r_up, r_dn_carts=r_dn))
    # FORWARD (fixed-phase, per-element Re) convention.
    # This reference machine uses the XLA production convention (project_real=True; XLA's
    # sum V is purely real), not the un-projected (holomorphic) variant: an oracle
    # transcribed from the implementation under test would be a weak oracle.
    # tmove: full (det*Jastrow) fixed-phase ratio. dltmove (DLA): det-ONLY ratio folds
    # into SP; only the FN off-diagonal carries the separate Jastrow ratio
    # (jqmc_gfmc.py:2067). project_real=True -> the mesh weight is already Re-projected.
    det_only = non_local_move == "dltmove"
    m_u, m_d, v_nl, _ = _nn_ecp(cd, wf, r_up, r_dn, RT, flag_determinant_only=det_only)
    v_nl = jnp.real(jnp.asarray(v_nl))
    v_fn = jnp.minimum(v_nl, 0.0)
    v_sp = jnp.sum(jnp.maximum(v_nl, 0.0))
    if det_only:
        jd = wf.jastrow_data
        j_base = compute_Jastrow_part(jd, r_up, r_dn)
        ratio = jnp.asarray(
            [jnp.exp(compute_Jastrow_part(jd, np.asarray(m_u)[m], np.asarray(m_d)[m]) - j_base) for m in range(len(v_nl))]
        )
        v_fn = v_fn * ratio
    # e_L / b_L from the PRODUCTION closure. The complex-twist
    # diagonal assembly (JAX-lexicographic min/max, fixed-phase Re-projection of the
    # ECP mesh inside nn_ecp) is _compute_V_elements_t's own; the leaf p_list / mesh /
    # sign_flip above remain the categorical-hop references.
    V_diag, V_nondiag, _ = _compute_V_elements_t(ham, r_up, r_dn, RT, non_local_move, alat)
    e_L = complex(V_diag + V_nondiag)
    b_L = complex(V_nondiag)
    # novar=0 split: MEASUREMENT (continuum) local energy (complex twin).
    e_L_meas = elel + iiv + complex(jnp.sum(ei + kc)) + ecp_local + complex(v_sp) + complex(jnp.sum(v_fn))
    return {
        "e_L": e_L,
        "e_L_meas": e_L_meas,
        "b_L": b_L,
        "p_list": np.asarray(jnp.real(jnp.concatenate([fn, v_fn]))),
        "mesh_up": np.concatenate([np.asarray(mesh_u), np.asarray(m_u)], axis=0),
        "mesh_dn": np.concatenate([np.asarray(mesh_d), np.asarray(m_d)], axis=0),
        "sign_flip": np.asarray(sflip),
    }


def _xla_interval_t_complex(ham, r_up, r_dn, alat, tau, key):
    w_L = 1.0
    tau_left = tau
    n_sub = 0
    e_last = 0j
    while tau_left > 0.0:
        n_sub += 1
        key, _rot = jax.random.split(key)
        st = _xla_step_complex(ham, r_up, r_dn, alat, np.eye(3))
        e_last = st["e_L_meas"]
        key, xik = jax.random.split(key)
        tau_update = min(tau_left, float(np.log(1.0 - _uniform(xik)) / st["b_L"].real))
        w_L *= float(np.exp(-tau_update * st["e_L"].real))
        tau_left -= tau_update
        key, mv = jax.random.split(key)
        u = _uniform(mv)
        if tau_left > 0.0:
            _k, r_up, r_dn = _hop(st, u)
    return {"w_L": w_L, "e_L": e_last, "r_up": r_up, "r_dn": r_dn, "key": key, "n_substeps": n_sub}


def _xla_interval_n_complex(ham, r_up, r_dn, alat, e_scf, num_per_meas, key):
    w_L = 1.0
    for _ in range(num_per_meas):
        key, _rot = jax.random.split(key)
        key, mv = jax.random.split(key)
        st = _xla_step_complex(ham, r_up, r_dn, alat, np.eye(3))
        w_L *= 1.0 / ((st["e_L"].real - st["b_L"].real) - e_scf) * (-st["b_L"].real)
        _k, r_up, r_dn = _hop(st, _uniform(mv))
    fin = _xla_step_complex(ham, r_up, r_dn, alat, np.eye(3))
    return {
        "w_L": w_L,
        # novar=0 split: measurement e_L; V_diag keeps the propagation composition.
        "e_L": fin["e_L_meas"],
        "V_diag": fin["e_L"].real - fin["b_L"].real,
        "r_up": r_up,
        "r_dn": r_dn,
        "key": key,
        "n_substeps": num_per_meas,
    }


# --------------------------------------------------------------------------- #
# Rust step / interval adapters
# --------------------------------------------------------------------------- #
def _rs_step_eL(ham, r_up, r_dn, alat, RT, non_local_move, fast: bool):
    wf = ham.wavefunction_data
    gd, jd = wf.geminal_data, wf.jastrow_data
    cd = ham.coulomb_potential_data
    if cd.ecp_flag:
        e_L, b_L, _n = jqmc_rs._gfmc_lrdmc_ecp(
            gd, jd, cd, r_up, r_dn, alat, RT, _MESH_W, _MESH_GP, 1, non_local_move == "dltmove", fast
        )
    else:
        e_L, b_L = jqmc_rs._gfmc_lrdmc_ae(gd, jd, cd, r_up, r_dn, alat, RT, fast)
    return float(e_L), float(b_L)


def _assert_close(got, ref, atol, rtol, msg):
    assert abs(got - ref) <= atol + rtol * abs(ref), (msg, got, ref, abs(got - ref))


def _gfmc_projection_tolerance(zones):
    """Projection aggregates use the established full=strict/mixed=medium policy."""
    return get_tolerance_xla_rust_consistency(zones, level="medium")


# --------------------------------------------------------------------------- #
# Cell table. ``_CELL_ZONES``/``route`` pick the named tolerance pair; ``shift`` marks the
# FAR (unwrapped, several lattice vectors out) periodic rows.
# --------------------------------------------------------------------------- #
_CELLS = {
    # molecular all-electron (no ECP -> the discretized el-ion max-arg branch, parcutg=2)
    "mol_ae_sphe": dict(pattern="mol_sphe_sd_jnone", nlm="tmove", route="strict", seed=1),
    "mol_ae_j123": dict(pattern="mol_cart_sd_j123", nlm="tmove", route="strict", seed=1),
    "mol_ae_jdtn": dict(pattern="mol_cart_sd_jdtn", nlm="tmove", route="strict", seed=1),
    # molecular ECP: both non-local move conventions (parcutg=1 max-arg)
    "mol_ecp_tmove": dict(pattern="mol_ecp_cart_j12", nlm="tmove", route="strict", seed=3),
    "mol_ecp_dltmove": dict(pattern="mol_ecp_cart_j12", nlm="dltmove", route="strict", seed=3),
    "mol_ecp_jdtn": dict(pattern="mol_ecp_cart_jdtn", nlm="tmove", route="strict", seed=3),
    # production spherical determinant: exercises the Sphe candidate AO batch,
    # stateful rank-1 chain, and projection-scoped kinetic/ECP mesh caches.
    "mol_ecp_sphe": dict(pattern="mol_ecp_sphe_j12", nlm="tmove", route="strict", seed=3),
    # periodic real-Gamma ECP, in-cell and FAR (unwrapped by 3-4 lattice vectors)
    "pbc_gamma": dict(pattern="pbc_gamma_sphepgto_j12", nlm="tmove", route="medium", seed=2),
    "pbc_gamma_jdtn": dict(pattern="pbc_gamma_sphepgto_jdtn", nlm="tmove", route="medium", seed=2),
    # Periodic raw-AO pGTO J3, the ``pbc && PeriodicPgto`` disjunct
    # of gfmc.rs ``mesh_fast_state::j3_state_ok``; the interval test reads counter 12 == 0.
    "pbc_gamma_j123": dict(pattern="pbc_gamma_sphepgto_j123", nlm="tmove", route="medium", seed=2),
    "pbc_gamma_far": dict(
        pattern="pbc_gamma_sphepgto_j12",
        nlm="tmove",
        route="medium",
        seed=2,
        shift=[[3.0, -2.0, 4.0]],
    ),
}


# One explicit zone list per cell (never their union). The table is a literal so
# each case's list is read statically by its per-case executable check.
_CELL_ZONES = {
    "mol_ae_sphe": _EL_ZONES,
    "mol_ae_j123": _EL_ZONES_J,
    "mol_ae_jdtn": _EL_ZONES_J,
    "mol_ecp_tmove": _EL_ZONES_J,
    "mol_ecp_dltmove": _EL_ZONES_J,
    "mol_ecp_jdtn": _EL_ZONES_J,
    "mol_ecp_sphe": _EL_ZONES_J,
    "pbc_gamma": _EL_ZONES_PBC,
    "pbc_gamma_jdtn": _EL_ZONES_PBC,
    "pbc_gamma_j123": _EL_ZONES_PBC,
    "pbc_gamma_far": _EL_ZONES_PBC,
}


def _cell_state(name):
    c = _CELLS[name]
    ham = rs_ham(c["pattern"])
    r_up, r_dn = _cfg(ham, c["seed"], cell_shift=c.get("shift"))
    return ham, r_up, r_dn, c["nlm"]


# =========================================================================== #
# 1. Per-sub-step component parity: kinetic elements, sign flip, S', b_L, e_L
# =========================================================================== #
@pytest.mark.light
@pytest.mark.parametrize("cell", list(_CELLS))
@pytest.mark.parametrize("alat", [0.30, 1.00])
def test_step_components_parity(cell, alat):
    """Every intermediate of ONE projection sub-step, XLA vs Rust, at two alat.

    The Rust kernels return ``(e_L, b_L)`` and the raw ``elements``; the sign-flip
    vector, the per-electron ``nondiag`` fold and S' are pure functions of
    ``elements`` on BOTH sides (identical formula in ``gfmc.rs`` and
    ``_projection_t_core``), so element-wise parity + ``(e_L, b_L)`` parity pins the
    whole diagonal/mesh block including the ZV ``where(sign_flip, max(zv, ei), zv)``
    selection.
    """
    ham, r_up, r_dn, nlm = _cell_state(cell)
    atol, rtol = _gfmc_projection_tolerance(_CELL_ZONES[cell])
    wf = ham.wavefunction_data
    ref = _xla_step(ham, r_up, r_dn, alat, np.eye(3), nlm)
    els = np.asarray(jqmc_rs._gfmc_discretized_kinetic_elements(wf.geminal_data, wf.jastrow_data, r_up, r_dn, alat, np.eye(3)))
    assert els.shape == ref["elements"].shape
    assert np.allclose(els, ref["elements"], atol=atol, rtol=rtol), np.abs(els - ref["elements"]).max()

    # sign-flip determination and the S' fold, recomputed from each side's OWN elements
    rs_flip = np.any(els.reshape(-1, 6) >= 0.0, axis=1)
    assert np.array_equal(rs_flip, ref["sign_flip"]), (rs_flip, ref["sign_flip"])
    _assert_close(float(np.maximum(els, 0.0).sum()), ref["sp_kin"], atol, rtol, "S'_kinetic")
    rs_nondiag = (els.reshape(-1, 6) + 1.0 / (2.0 * alat**2)).sum(1)
    # JCP 163, 194117 (2025), Eq. (D4): six directed hops carry +1/(2a^2) each (3/a^2 total).
    stencil_background = rs_nondiag - els.reshape(-1, 6).sum(1)
    expected_background = np.full(stencil_background.shape, 3.0 / alat**2)  # equal shapes, no scalar broadcast
    assert np.allclose(
        stencil_background,
        expected_background,
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    assert np.allclose(rs_nondiag, ref["nondiag"], atol=atol, rtol=rtol)

    e_L, b_L = _rs_step_eL(ham, r_up, r_dn, alat, np.eye(3), nlm, fast=False)
    _assert_close(e_L, ref["e_L"], atol, rtol, "e_L")
    _assert_close(b_L, ref["b_L"], atol, rtol, "b_L (non_diagonal_sum_hamiltonian)")


@pytest.mark.light
@pytest.mark.parametrize("cell", ["mol_ae_sphe", "mol_ecp_tmove", "mol_ecp_dltmove", "pbc_gamma"])
def test_step_components_parity_rotated_mesh(cell):
    """Same, with a NON-identity mesh rotation R^T (the ``random_discretized_mesh``
    degree of freedom, injected deterministically instead of drawn)."""
    ham, r_up, r_dn, nlm = _cell_state(cell)
    atol, rtol = _gfmc_projection_tolerance(_CELL_ZONES[cell])
    wf = ham.wavefunction_data
    for rt in _rt_stack(2, seed=11):
        ref = _xla_step(ham, r_up, r_dn, 0.40, rt, nlm)
        els = np.asarray(jqmc_rs._gfmc_discretized_kinetic_elements(wf.geminal_data, wf.jastrow_data, r_up, r_dn, 0.40, rt))
        assert np.allclose(els, ref["elements"], atol=atol, rtol=rtol), np.abs(els - ref["elements"]).max()
        e_L, b_L = _rs_step_eL(ham, r_up, r_dn, 0.40, rt, nlm, fast=False)
        _assert_close(e_L, ref["e_L"], atol, rtol, "e_L(rotated mesh)")
        _assert_close(b_L, ref["b_L"], atol, rtol, "b_L(rotated mesh)")


# =========================================================================== #
# 2. Forced branches of the step machine
# =========================================================================== #
# (pattern, seed, scale, alat, expected branch) -- each row was selected by scanning
# committed fixtures until the branch fired, and the test ASSERTS the branch fired
# (so a fixture/kernel change that silently stops reaching it turns the row red).
_FLIP_ROWS = [
    ("none", "mol_cart_sd_jnone", 2, 1.5, 0.30),  # no electron sign-flips -> ZV arm only
    ("mixed", "mol_cart_sd_jnone", 0, 0.5, 0.30),  # electron-wise mask: some flip, some not
    ("mixed_large_alat", "mol_cart_sd_jnone", 3, 0.5, 2.00),  # alat variation changes the mask
    ("all", "mol_ecp_sphe_j12", 0, 0.5, 2.00),  # every electron sign-flips -> max arm only
]


@pytest.mark.light
@pytest.mark.parametrize("tag,pattern,seed,scale,alat", _FLIP_ROWS)
def test_sign_flip_branches_forced_and_parity(tag, pattern, seed, scale, alat):
    """Force the sign-flip determination into each of its states (none / electron-wise
    mixed / all) and check XLA==Rust there. ``sign_flip`` selects the per-electron
    el-ion ZV arm (``max(zv, ei)`` vs ``zv``), so a mis-set flag moves e_L."""
    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    nlm = "tmove"
    # Both committed fixtures carry a Jastrow_data container, including jnone.
    zones = _EL_ZONES_J
    assert zones == (_EL_ZONES_J if ham.wavefunction_data.jastrow_data is not None else _EL_ZONES)
    atol, rtol = _gfmc_projection_tolerance(zones)
    r_up, r_dn = _cfg(ham, seed, scale=scale)
    ref = _xla_step(ham, r_up, r_dn, alat, np.eye(3), nlm)
    n_flip = int(ref["sign_flip"].sum())
    n_el = int(ref["sign_flip"].size)
    if tag == "none":
        assert n_flip == 0, (tag, n_flip, n_el)
        assert ref["sp_kin"] == 0.0, ref["sp_kin"]  # S' EMPTY: all elements < 0
    elif tag == "all":
        assert n_flip == n_el, (tag, n_flip, n_el)
        assert ref["sp_kin"] > 0.0
    else:
        assert 0 < n_flip < n_el, (tag, n_flip, n_el)
        assert ref["sp_kin"] > 0.0  # S' NON-EMPTY

    els = np.asarray(jqmc_rs._gfmc_discretized_kinetic_elements(wf.geminal_data, wf.jastrow_data, r_up, r_dn, alat, np.eye(3)))
    assert np.array_equal(np.any(els.reshape(-1, 6) >= 0.0, axis=1), ref["sign_flip"])
    e_L, b_L = _rs_step_eL(ham, r_up, r_dn, alat, np.eye(3), nlm, fast=False)
    _assert_close(e_L, ref["e_L"], atol, rtol, f"e_L[{tag}]")
    _assert_close(b_L, ref["b_L"], atol, rtol, f"b_L[{tag}]")


@pytest.mark.light
@pytest.mark.parametrize("seed", [1, 3])
def test_ecp_nonlocal_sprime_nonempty_separates_tmove_and_dltmove(seed):
    """The ECP S' branch. ``tmove`` folds ``max(V_det * J_ratio, 0)`` into S' and takes
    ``min(V_det * J_ratio, 0)`` off-diagonal; ``dltmove`` (DLA) folds ``max(V_det, 0)``
    and multiplies ONLY the off-diagonal by the Jastrow ratio. The two coincide
    identically when every ``V_nonlocal < 0`` (empty ECP S'), so this row FORCES a
    configuration with positive mesh amplitudes -- otherwise a dltmove/tmove swap in
    the Rust kernel would be invisible.
    """
    ham = rs_ham("mol_ecp_sphe_j12")
    wf = ham.wavefunction_data
    cd = ham.coulomb_potential_data
    atol, rtol = _gfmc_projection_tolerance(_EL_ZONES_J)
    r_up, r_dn = _cfg(ham, seed, scale=0.8)
    _, _, v_nl, _ = _nn_ecp(cd, wf, r_up, r_dn, jnp.eye(3), flag_determinant_only=True)
    assert (np.asarray(v_nl).real > 0.0).any(), "cell does not force a non-empty ECP S'"

    e_t = _xla_step(ham, r_up, r_dn, 0.30, np.eye(3), "tmove")
    e_d = _xla_step(ham, r_up, r_dn, 0.30, np.eye(3), "dltmove")
    assert e_t["sp_ecp"] > 0.0 and e_d["sp_ecp"] > 0.0
    assert abs(e_t["e_L"] - e_d["e_L"]) > atol + rtol * abs(e_t["e_L"]), "tmove/dltmove are degenerate here"

    for nlm, ref in (("tmove", e_t), ("dltmove", e_d)):
        got_e, got_b = _rs_step_eL(ham, r_up, r_dn, 0.30, np.eye(3), nlm, fast=False)
        _assert_close(got_e, ref["e_L"], atol, rtol, f"e_L[{nlm}]")
        _assert_close(got_b, ref["b_L"], atol, rtol, f"b_L[{nlm}]")


# =========================================================================== #
# 3. rank-1 / FAST state vs the FULL-RECOMPUTE reference (exit criterion 3)
# =========================================================================== #
@pytest.mark.light
@pytest.mark.parametrize("cell", list(_CELLS))
@pytest.mark.parametrize("alat", [0.30, 1.00])
def test_fast_state_equals_full_recompute(cell, alat):
    """``gfmc_lrdmc_{ae,ecp}(fast=True)`` shares ONE Sherman-Morrison base state across
    the kinetic hop mesh AND the ECP T-move quadrature; ``fast=False`` recomputes every
    mesh point from scratch. Both must produce the same (e_L, b_L) -- and both must
    match XLA, so a shared regression cannot hide behind agreeing with itself."""
    ham, r_up, r_dn, nlm = _cell_state(cell)
    atol, rtol = _gfmc_projection_tolerance(_CELL_ZONES[cell])
    ref = _xla_step(ham, r_up, r_dn, alat, np.eye(3), nlm)
    slow_e, slow_b = _rs_step_eL(ham, r_up, r_dn, alat, np.eye(3), nlm, fast=False)
    fast_e, fast_b = _rs_step_eL(ham, r_up, r_dn, alat, np.eye(3), nlm, fast=True)
    _assert_close(fast_e, slow_e, atol, rtol, "fast vs full-recompute e_L")
    _assert_close(fast_b, slow_b, atol, rtol, "fast vs full-recompute b_L")
    _assert_close(fast_e, ref["e_L"], atol, rtol, "fast e_L vs XLA")
    _assert_close(fast_b, ref["b_L"], atol, rtol, "fast b_L vs XLA")


@pytest.mark.light
@pytest.mark.parametrize("alat", [0.30, 1.00])
def test_fast_state_equals_full_recompute_cart_pbc(alat):
    """CARTESIAN periodic coverage of the WF fast path.

    The registered ``pbc_*`` GFMC cells are all spherical-pGTO (``lithium.h5``),
    but the converter's production output is CARTESIAN periodic. This pins the Cart n_image>1 leaf on
    the periodic WF fast path (per-basis candidate/derivative routes): fast (Sherman-Morrison + periodic
    candidate tables) == full-recompute == XLA at the periodic consistency
    envelope. Bare-AE (no ECP) so the AE + kinetic-mesh Cart cache is
    exercised; the ECP-mesh periodic cache is pinned by the ``pbc_gamma``
    (Sphe) cells and the Cart affine fold by the molecular Cart cells.
    Route engagement is observable via ``jqmc_rs.gfmc_route_counters()``
    (periodic-pGTO build/refresh indices); the interval-level liveness pins
    live with the route-counter suite, not here.
    """
    ham = _build_ham_for_cell("H2_pbc_ae_cart.h5", True, None, "SD")
    aos = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
    assert int(getattr(aos, "_n_image", 1)) > 1, "cart PBC fixture must be image-tiled"
    r_up, r_dn = _cfg(ham, seed=2)
    atol, rtol = get_tolerance_xla_rust_consistency(_EL_ZONES_PBC, level="strict")
    route_atol, route_rtol = get_tolerance_min(_EL_ZONES_PBC, level="strict")
    ref = _xla_step(ham, r_up, r_dn, alat, np.eye(3), "tmove")

    slow_e, slow_b = _rs_step_eL(ham, r_up, r_dn, alat, np.eye(3), "tmove", fast=False)
    fast_e, fast_b = _rs_step_eL(ham, r_up, r_dn, alat, np.eye(3), "tmove", fast=True)

    _assert_close(fast_e, slow_e, route_atol, route_rtol, "cart-pbc fast vs full-recompute e_L")
    _assert_close(fast_b, slow_b, route_atol, route_rtol, "cart-pbc fast vs full-recompute b_L")
    _assert_close(fast_e, ref["e_L"], atol, rtol, "cart-pbc fast e_L vs XLA")
    _assert_close(fast_b, ref["b_L"], atol, rtol, "cart-pbc fast b_L vs XLA")


@pytest.mark.light
@pytest.mark.parametrize("alat", [0.30, 1.00])
def test_fast_state_equals_full_recompute_bspline_pbc(alat):
    """B-SPLINE (blip) periodic coverage of the WF fast path.

    The pure B-spline leaf rides the WF fast path (per-basis B-spline
    candidate/derivative routes). Pins
    fast (Sherman-Morrison + spline candidate tables) == full-recompute == XLA
    at the periodic consistency envelope on ``QEPROJ:bspline:Li_gamma:ECP``
    (real Gamma, ECP -> BOTH the kinetic and the ECP T-move meshes engage).
    Route engagement is observable via ``jqmc_rs.gfmc_route_counters()``
    (B-spline build/refresh indices); the interval-level liveness pins live
    with the route-counter suite, not here.
    """
    from jqmc.atomic_orbital import AOs_bspline_data

    ham = rs_ham("bspline_gamma_jnone")
    orb = ham.wavefunction_data.geminal_data.orb_data_up_spin
    assert isinstance(orb.aos_data, AOs_bspline_data), "fixture must carry a pure B-spline leaf"
    r_up, r_dn = _cfg(ham, seed=2)
    atol, rtol = get_tolerance_xla_rust_consistency(_EL_ZONES_PBC, level="strict")
    route_atol, route_rtol = get_tolerance_min(_EL_ZONES_PBC, level="strict")
    ref = _xla_step(ham, r_up, r_dn, alat, np.eye(3), "tmove")

    slow_e, slow_b = _rs_step_eL(ham, r_up, r_dn, alat, np.eye(3), "tmove", fast=False)
    fast_e, fast_b = _rs_step_eL(ham, r_up, r_dn, alat, np.eye(3), "tmove", fast=True)

    _assert_close(fast_e, slow_e, route_atol, route_rtol, "bspline-pbc fast vs full-recompute e_L")
    _assert_close(fast_b, slow_b, route_atol, route_rtol, "bspline-pbc fast vs full-recompute b_L")
    _assert_close(fast_e, ref["e_L"], atol, rtol, "bspline-pbc fast e_L vs XLA")
    _assert_close(fast_b, ref["b_L"], atol, rtol, "bspline-pbc fast b_L vs XLA")


# The ECP mesh is routed through the per-basis EcpAoCache; ECP-mesh engagement is
# counted by ``jqmc_rs.gfmc_route_counters()``.


# =========================================================================== #
# 4. Multi-step interval parity with pinned randomness (exit criterion 1)
# =========================================================================== #
_INTERVAL_CELLS = [
    "mol_ae_sphe",
    "mol_ae_j123",
    "mol_ae_jdtn",
    "mol_ecp_tmove",
    "mol_ecp_dltmove",
    "mol_ecp_jdtn",
    "mol_ecp_sphe",
    "pbc_gamma",
    "pbc_gamma_jdtn",
    "pbc_gamma_j123",
    "pbc_gamma_far",
]


@pytest.mark.light
@pytest.mark.parametrize("cell", _INTERVAL_CELLS)
def test_multistep_projection_n_parity(cell):
    """GFMC_n: the native interval driver vs the XLA reference machine, sub-step by
    sub-step (the interval is re-run for every prefix length 1..3, so the state
    BEFORE reconfiguration is compared at each sub-step, not just at the end).

    Compared per prefix: the branch weight ``w_L`` (the running product of
    ``-b_L/(V_diag - E_scf)``), the measurement ``e_L`` and ``V_diag`` at the
    post-hop configuration, the walker positions (which mesh point the categorical
    hop selected), and the final threefry key (the RNG contract).
    """
    ham, r_up, r_dn, nlm = _cell_state(cell)
    atol, rtol = _gfmc_projection_tolerance(_CELL_ZONES[cell])
    ctx = _ctx(ham)
    key = jax.random.PRNGKey(7)
    jqmc_rs.gfmc_route_counters_reset()
    for nsub in (1, 2, 3):
        comparison_iteration = str(nsub)  # noqa: F841 - captured by the GDD iteration receipt
        ref = _xla_interval_n(ham, r_up, r_dn, 0.30, -1.0, nsub, key, nlm)
        got = ctx.run_gfmc_projection_n(
            r_up,
            r_dn,
            0.30,
            -1.0,
            nsub,
            int(key[0]),
            int(key[1]),
            np.zeros((0, 3, 3)),
            _MESH_W,
            _MESH_GP,
            random_mesh=False,
            dltmove=(nlm == "dltmove"),
            nn=1,
        )
        _assert_close(float(got["w_L"]), ref["w_L"], atol, rtol, f"w_L[{cell},nsub={nsub}]")
        _assert_close(float(got["e_L"]), ref["e_L"], atol, rtol, f"e_L[{cell},nsub={nsub}]")
        _assert_close(float(got["V_diag"]), ref["V_diag"], atol, rtol, f"V_diag[{cell},nsub={nsub}]")
        assert np.allclose(np.asarray(got["r_up"]), ref["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        assert np.allclose(np.asarray(got["r_dn"]), ref["r_dn"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        assert int(got["n_substeps"]) == nsub
        assert list(np.asarray(got["final_key"], dtype=np.uint32)) == list(np.asarray(ref["key"], dtype=np.uint32))
        # the walk must actually move the walker (otherwise the hop branch is vacuous)
        assert not np.allclose(
            ref["r_up"], r_up, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION
        ) or not np.allclose(ref["r_dn"], r_dn, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION)

    counters = jqmc_rs.gfmc_route_counters()
    winv = {name: counters[idx] for name, idx in {"builds": 56, "advances": 57, "kin_hits": 58, "fallbacks": 60}.items()}
    assert all(winv[name] for name in ("builds", "advances", "kin_hits")) and winv["fallbacks"] == 0, (cell, winv)
    # IMPL-RS-GFMC-ECP-ZERO-GROUP-SKIP-001: only live (non-zero-amplitude) ECP
    # groups evaluate ratios; the H2 ccECP fixture's groups are all exactly zero.
    ecp_live, ecp_zero = counters[68], counters[69]
    if ham.coulomb_potential_data.ecp_flag:
        assert ecp_live + ecp_zero > 0, (cell, "ECP mesh did not run", ecp_live, ecp_zero)
        assert counters[59] == ecp_live, (cell, "live ECP groups did not consume maintained winv", counters[59], ecp_live)

    pattern = _CELLS[cell]["pattern"]
    if "jdtn" in pattern:
        kinetic_dtn = {name: counters[idx] for name, idx in {"builds": 45, "advances": 46, "hits": 47}.items()}
        assert all(kinetic_dtn.values()), (cell, kinetic_dtn)
        if ham.coulomb_potential_data.ecp_flag:
            ecp_dtn = {name: counters[idx] for name, idx in {"builds": 61, "advances": 62, "hits": 63}.items()}
            assert ecp_dtn["builds"] and ecp_dtn["advances"] and (ecp_dtn["hits"] > 0) == (ecp_live > 0), (
                cell,
                ecp_dtn,
                ecp_live,
            )
    jd = ham.wavefunction_data.jastrow_data
    if jd.jastrow_one_body_data is not None or jd.jastrow_two_body_data is not None:
        kinetic_j12 = {name: counters[idx] for name, idx in {"builds": 48, "advances": 49, "hits": 50, "fallbacks": 51}.items()}
        assert all(kinetic_j12[name] for name in ("builds", "advances", "hits")) and kinetic_j12["fallbacks"] == 0, (
            cell,
            kinetic_j12,
        )
        if ham.coulomb_potential_data.ecp_flag:
            ecp_j12 = {name: counters[idx] for name, idx in {"builds": 52, "advances": 53, "hits": 54, "fallbacks": 55}.items()}
            assert ecp_j12["builds"] and ecp_j12["advances"] and ecp_j12["hits"] == ecp_live and ecp_j12["fallbacks"] == 0, (
                cell,
                ecp_j12,
                ecp_live,
            )
    if jd.jastrow_three_body_data is not None:
        assert counters[12] == 0, (cell, "J3 candidate table fell back", counters[12])


# Declared slow routes. gfmc.rs ``mesh_fast_state::j3_state_ok``
# requires ``orb.mo_coefficients.is_none()``: a J3 whose basis is MO-wrapped carries no
# streaming J3 ratio state and every J3 candidate ratio takes the counted slow fallback
# (route counter 12, GFMC_J3_SLOW_FALLBACKS). Because the kinetic candidate cache is
# ``cache_ok = reuse_fixed_mesh && det_up_fast && det_dn_fast && j3_fast``, the J1/J2
# tabulated candidate ratios fall back with it (route counter 51,
# GFMC_J12_TABPIP_KIN_FALLBACKS). This is a same-observable performance route by
# design, not a capability gap. The declaration is executable: the J3-MO cell
# must take it (counter 12 > 0) and must still match the XLA machine sub-step by
# sub-step; a Rust change that streams J3-MO without updating this row turns it red.
# A MOLECULAR MO-wrapped J3 is lowered to raw AO at the GFMC parse and
# streams (pinned by test_gfmc_mo_j3_lowering.py), so the declared cell is the
# periodic MO-wrapped pGTO(+)PW composite J3, which the lowering leaves unchanged.
_DECLARED_SLOW_ROUTES = {
    "j3_mo_periodic": dict(
        stem="QE:Li_gamma_pgto+pw",
        spec={"j3": "mo"},
        counters=(12, 51),
        reason="periodic MO-wrapped J3 basis: no streaming J3 ratio state (j3_state_ok requires mo_coefficients None; "
        "the raw-AO lowering is molecular-only), so the kinetic candidate cache and its J1/J2 table are bypassed",
    ),
}


@pytest.mark.light
@pytest.mark.parametrize("route", sorted(_DECLARED_SLOW_ROUTES))
def test_declared_slow_route_is_taken_and_matches_xla(route):
    """A declared slow Rust route fires on its fixture and is value-identical to XLA.

    j3_mo_periodic: the real-Gamma Li pGTO(+)PW composite SD with J1/J2 and an
    MO-wrapped J3 (the generic ``j3="mo"`` builder); a molecular MO-wrapped J3
    does not take this route (raw-AO lowering). GFMC_n interval prefixes 1..3 at a pinned key:
    w_L, e_L and V_diag within the same policy as the interval cells, walkers and
    the final threefry key exact, the walk non-trivial. The declared counter
    must be positive (the slow route is the one taken), the carried-state
    fallbacks of the interval test must stay silent.
    """
    decl = _DECLARED_SLOW_ROUTES[route]
    ham = _build_ham_for_cell(decl["stem"], True, decl["spec"], "SD")
    j3 = ham.wavefunction_data.jastrow_data.jastrow_three_body_data
    assert j3 is not None and getattr(j3.orb_data, "mo_coefficients", None) is not None, route
    r_up, r_dn = _cfg(ham, 1)
    atol, rtol = _gfmc_projection_tolerance(_EL_ZONES_J)
    ctx = _ctx(ham)
    key = jax.random.PRNGKey(7)
    jqmc_rs.gfmc_route_counters_reset()
    for nsub in (1, 2, 3):
        ref = _xla_interval_n(ham, r_up, r_dn, 0.30, -1.0, nsub, key, "tmove")
        got = ctx.run_gfmc_projection_n(
            r_up,
            r_dn,
            0.30,
            -1.0,
            nsub,
            int(key[0]),
            int(key[1]),
            np.zeros((0, 3, 3)),
            _MESH_W,
            _MESH_GP,
            random_mesh=False,
            dltmove=False,
            nn=1,
        )
        _assert_close(float(got["w_L"]), ref["w_L"], atol, rtol, f"w_L[{route},nsub={nsub}]")
        _assert_close(float(got["e_L"]), ref["e_L"], atol, rtol, f"e_L[{route},nsub={nsub}]")
        _assert_close(float(got["V_diag"]), ref["V_diag"], atol, rtol, f"V_diag[{route},nsub={nsub}]")
        assert np.allclose(np.asarray(got["r_up"]), ref["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        assert np.allclose(np.asarray(got["r_dn"]), ref["r_dn"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        assert list(np.asarray(got["final_key"], dtype=np.uint32)) == list(np.asarray(ref["key"], dtype=np.uint32))
        assert not np.allclose(
            ref["r_up"], r_up, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION
        ) or not np.allclose(ref["r_dn"], r_dn, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION)
    counters = jqmc_rs.gfmc_route_counters()
    taken = {12: counters[12], 51: counters[51]}
    assert tuple(taken) == decl["counters"], (route, "declared counter set changed", decl["counters"])
    assert all(taken.values()), (route, "declared slow route was not taken", taken, decl["reason"])
    assert counters[60] == 0, (route, "carried winv fell back", counters[60])


# Complex determinant derivative-cache counters (gfmc.rs ``note_deriv_event_c``) sit
# exactly nine slots after the matching complex candidate-table counters
# (``note_candidate_event_c``): pgto 15/16 -> 24/25, bspline 17/18 -> 26/27,
# pw / pgto+pw / pgto+bspline 19/20 -> 28/29.
_CPLX_DERIV_OFFSET = 9
_CPLX_DERIV_PAIRS = ((24, 25), (26, 27), (28, 29))


def _assert_interval_state_route_witnesses(counters, ham, complex_path, build_idx, refresh_idx, pattern):
    """Route counters of the interval state, each read by this test.

    Real path (gfmc.rs ``fast_mol_kinetic_ok`` / ``fast_generic_det_gradlap_cell``):
    a molecular MO-backed geminal takes the molecular fast-kinetic state ([1]),
    anything else the reduced state ([2]); every real geminal off the molecular
    MO route (periodic leaves, molecular raw-AO AGP / AGPn) builds the
    basis-generic det grad/lap cache ([13]). Complex path: the carried complex
    state is built ([14]) and never declined ([23] CPLX_STATE_GATE_MISS == 0),
    the per-class complex det derivative cache builds and refreshes, the other
    classes stay silent, and the counted slow route ([30]) never fires.
    """
    counters = list(counters)
    if complex_path:
        assert counters[14] > 0 and counters[23] == 0, (
            f"{pattern}: complex fast state not carried (CPLX_STATE_INTERVALS={counters[14]}, "
            f"CPLX_STATE_GATE_MISS={counters[23]})"
        )
        own = (build_idx + _CPLX_DERIV_OFFSET, refresh_idx + _CPLX_DERIV_OFFSET)
        assert own in _CPLX_DERIV_PAIRS, (pattern, own)
        for pair in _CPLX_DERIV_PAIRS:
            got = (counters[pair[0]], counters[pair[1]])
            if pair == own:
                assert got[0] > 0 and got[1] > 0, f"{pattern}: complex det derivative cache {pair} did not build/refresh: {got}"
            else:
                assert got == (0, 0), f"{pattern}: foreign complex derivative class {pair} fired: {got}"
        assert counters[30] == 0, f"{pattern}: complex det derivative took the slow route ({counters[30]})"
        assert (counters[1], counters[2], counters[13]) == (0, 0, 0), (pattern, "real state counters on complex path")
    else:
        pbc = bool(ham.structure_data.pbc_flag)
        mo_backed = type(ham.wavefunction_data.geminal_data.orb_data_up_spin).__name__ == "MOs_data"
        molecular_fast = (not pbc) and mo_backed
        got = (counters[1] > 0, counters[2] > 0)
        assert got == (molecular_fast, not molecular_fast), (
            f"{pattern}: real interval state route: DET_FAST={counters[1]}, REDUCED_STATE={counters[2]}, "
            f"expected molecular_fast={molecular_fast}"
        )
        # Every real geminal off the molecular MO route builds the basis-generic
        # det grad/lap cache: periodic leaves and the molecular
        # raw-AO AGP / AGPn.
        assert (counters[13] > 0) == (not molecular_fast), (
            f"{pattern}: DET_GRAD_GENERIC_BUILDS={counters[13]} with molecular_fast={molecular_fast}"
        )
        assert all(counters[i] == 0 for i in (14, 23, *range(24, 31))), (pattern, "complex counters on real path")


_WINV_AO_CASES = [
    # determinant representation x molecular AO convention
    pytest.param("mol_cart_sd_jnone", False, 3, 4, id="real-cart-sd"),
    pytest.param("mol_sphe_sd_jnone", False, 3, 4, id="real-sphe-sd"),
    pytest.param("mol_cart_agp_jnone", False, 3, 4, id="real-cart-agp-raw-ao"),
    pytest.param("mol_cart_agpn_jnone", False, 3, 4, id="real-cart-agpn-raw-ao"),
    # real periodic AO leaves and both composite families
    pytest.param("pbc_gamma_sphepgto_jnone", False, 5, 6, id="real-pgto"),
    pytest.param("pw_gamma_jnone", False, 9, 10, id="real-pw"),
    pytest.param("bspline_gamma_jnone", False, 7, 8, id="real-bspline"),
    pytest.param("pbc_composite_jnone", False, 9, 10, id="real-pgto-pw"),
    pytest.param("pgto_bspline_gamma_jnone", False, 9, 10, id="real-pgto-bspline"),
    # non-Gamma complex twins
    pytest.param("twist_sphepgto_jnone", True, 15, 16, id="complex-pgto"),
    pytest.param("pw_twist_jnone", True, 19, 20, id="complex-pw"),
    pytest.param("bspline_twist_jnone", True, 17, 18, id="complex-bspline"),
    pytest.param("pgto_pw_twist_jnone", True, 19, 20, id="complex-pgto-pw"),
    pytest.param("pgto_bspline_twist_jnone", True, 19, 20, id="complex-pgto-bspline"),
]


_WINV_ZONES = {
    "mol_cart_sd_jnone": _EL_ZONES,
    "mol_sphe_sd_jnone": _EL_ZONES,
    "mol_cart_agp_jnone": _EL_ZONES,
    "mol_cart_agpn_jnone": _EL_ZONES,
    "pbc_gamma_sphepgto_jnone": _EL_ZONES_PBC,
    "pw_gamma_jnone": _EL_ZONES_PBC,
    "bspline_gamma_jnone": _EL_ZONES_PBC,
    "pbc_composite_jnone": _EL_ZONES_PBC,
    "pgto_bspline_gamma_jnone": _EL_ZONES_PBC,
    "twist_sphepgto_jnone": _EL_ZONES_PBC,
    "pw_twist_jnone": _EL_ZONES_PBC,
    "bspline_twist_jnone": _EL_ZONES_PBC,
    "pgto_pw_twist_jnone": _EL_ZONES_PBC,
    "pgto_bspline_twist_jnone": _EL_ZONES_PBC,
}


@pytest.mark.light
@pytest.mark.parametrize(("pattern", "complex_path", "build_idx", "refresh_idx"), _WINV_AO_CASES)
def test_winv_covers_supported_determinant_and_ao_types(pattern, complex_path, build_idx, refresh_idx):
    """Every supported determinant/AO leaf uses maintained winv in a live hop chain.

    This is deliberately route-observable as well as numerical: one three-hop
    native interval must agree with XLA, build and rank-one advance winv, serve
    both kinetic and (when present) ECP meshes, and avoid the counted fallback.
    """
    ham = rs_ham(pattern)
    r_up, r_dn = _cfg(ham, 17)
    key = jax.random.PRNGKey(29)
    zones = _WINV_ZONES[pattern]
    assert zones == (_EL_ZONES_PBC if ham.structure_data.pbc_flag else _EL_ZONES)
    atol, rtol = _gfmc_projection_tolerance(zones)
    jqmc_rs.gfmc_route_counters_reset()
    if complex_path:
        ctx = _ctx(ham, complex_=True)
        ref = _xla_interval_n_complex(ham, r_up, r_dn, 0.30, -1.0, 3, key)
        got = ctx.run_gfmc_projection_n_complex(
            r_up,
            r_dn,
            0.30,
            -1.0,
            3,
            int(key[0]),
            int(key[1]),
            np.zeros((0, 3, 3)),
            _MESH_W,
            _MESH_GP,
            random_mesh=False,
            nn=1,
        )
        _assert_close(complex(got["e_L"]).real, ref["e_L"].real, atol, rtol, f"Re(e_L)[{pattern}]")
        _assert_close(complex(got["e_L"]).imag, ref["e_L"].imag, atol, rtol, f"Im(e_L)[{pattern}]")
        slow_idx = 21
    else:
        ctx = _ctx(ham)
        ref = _xla_interval_n(ham, r_up, r_dn, 0.30, -1.0, 3, key, "tmove")
        got = ctx.run_gfmc_projection_n(
            r_up,
            r_dn,
            0.30,
            -1.0,
            3,
            int(key[0]),
            int(key[1]),
            np.zeros((0, 3, 3)),
            _MESH_W,
            _MESH_GP,
            random_mesh=False,
            nn=1,
        )
        _assert_close(float(got["e_L"]), ref["e_L"], atol, rtol, f"e_L[{pattern}]")
        slow_idx = 11
    _assert_close(float(got["w_L"]), ref["w_L"], atol, rtol, f"w_L[{pattern}]")
    assert np.allclose(np.asarray(got["r_up"]), ref["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    assert np.allclose(np.asarray(got["r_dn"]), ref["r_dn"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)

    counters = jqmc_rs.gfmc_route_counters()
    _assert_interval_state_route_witnesses(counters, ham, complex_path, build_idx, refresh_idx, pattern)
    # Negative control: a complex fast-state gate miss / a real reduced-state
    # misroute must turn the witness red (the helper is not vacuous).
    mutated = list(counters)
    if complex_path:
        mutated[14], mutated[23] = 0, 1
        expected_message = "complex fast state"
    else:
        mutated[1], mutated[2] = mutated[2], mutated[1]
        expected_message = "real interval state route"
    with pytest.raises(AssertionError, match=expected_message):
        _assert_interval_state_route_witnesses(mutated, ham, complex_path, build_idx, refresh_idx, pattern)
    assert counters[build_idx] > 0 and counters[refresh_idx] > 0, (pattern, build_idx, refresh_idx, counters)
    assert counters[slow_idx] == 0, (pattern, "candidate slow path", counters[slow_idx])
    # IMPL-RS-GFMC-MOL-CANDIDATE-EXACT-001: molecular MO-space determinant
    # candidate batches take the lane-tiled projection (index 66), never the
    # dense fallback (index 67); non-molecular leaves reach neither molecular
    # MO kernel (an AGP raw-AO leaf may, through its MO-wrapped spin, or not).
    mo_tiled, mo_dense = counters[66], counters[67]
    if pattern.startswith("mol_"):
        assert mo_dense == 0, (pattern, "molecular MO candidate dense fallback", mo_tiled, mo_dense)
        if pattern in ("mol_cart_sd_jnone", "mol_sphe_sd_jnone"):
            assert mo_tiled > 0, (pattern, "molecular MO candidate route", mo_tiled, mo_dense)
    else:
        assert mo_tiled == 0 and mo_dense == 0, (pattern, "molecular MO candidate route", mo_tiled, mo_dense)
    winv = {
        name: counters[idx]
        for name, idx in {"builds": 56, "advances": 57, "kin_hits": 58, "ecp_hits": 59, "fallbacks": 60}.items()
    }
    assert all(winv[name] for name in ("builds", "advances", "kin_hits")) and winv["fallbacks"] == 0, (pattern, winv)
    if ham.coulomb_potential_data.ecp_flag:
        assert winv["ecp_hits"] > 0, (pattern, winv)
    else:
        assert winv["ecp_hits"] == 0, (pattern, winv)


@pytest.mark.light
@pytest.mark.parametrize("cell", ["mol_ecp_tmove", "mol_ecp_dltmove", "mol_ecp_sphe"])
def test_multistep_projection_n_injected_rotations(cell):
    """GFMC_n projection-cadence contract with injected XLA rotations.

    The bridge supplies a per-substep RT stack so the PRNG/key
    contract stays explicit, but the native interval must consume only its first RT
    and hold it fixed until the next projection interval. The ECP quadrature uses
    the independent child-key rotation. The water row exercises the production
    spherical-ECP path in addition to the H2 Cartesian tmove/dltmove rows.
    """
    ham, r_up, r_dn, nlm = _cell_state(cell)
    atol, rtol = _gfmc_projection_tolerance(_CELL_ZONES[cell])
    ctx = _ctx(ham)
    key = jax.random.PRNGKey(21)
    drawn_rts = _rt_stack(3, seed=5)
    interval_rts = np.repeat(drawn_rts[:1], len(drawn_rts), axis=0)
    ref = _xla_interval_n(ham, r_up, r_dn, 0.30, -1.0, len(drawn_rts), key, nlm, rts=interval_rts)
    got = ctx.run_gfmc_projection_n(
        r_up,
        r_dn,
        0.30,
        -1.0,
        len(drawn_rts),
        int(key[0]),
        int(key[1]),
        drawn_rts,
        _MESH_W,
        _MESH_GP,
        random_mesh=True,
        dltmove=(nlm == "dltmove"),
        nn=1,
    )
    _assert_close(float(got["w_L"]), ref["w_L"], atol, rtol, "w_L(injected RTs)")
    _assert_close(float(got["e_L"]), ref["e_L"], atol, rtol, "e_L(injected RTs)")
    assert np.allclose(np.asarray(got["r_up"]), ref["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    assert np.allclose(np.asarray(got["r_dn"]), ref["r_dn"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    assert list(np.asarray(got["final_key"], dtype=np.uint32)) == list(np.asarray(ref["key"], dtype=np.uint32))


@pytest.mark.light
@pytest.mark.parametrize("cell", _INTERVAL_CELLS)
def test_multistep_projection_n_native_random_rotation(cell):
    """GFMC_n native fast cache matches XLA with random mesh rotation enabled.

    This is deliberately distinct from the injected-rotation route: an empty RT
    stack selects the production Rust sampler and projection-scoped candidate
    caches.  The reference derives the same interval-fixed kinetic/ECP rotations
    from the shared Threefry key.  Prefixes 1..3 make the first post-hop cache
    refresh observable; a one-step-only test cannot detect a stale or cross-wired
    candidate table.
    """
    ham, r_up, r_dn, nlm = _cell_state(cell)
    atol, rtol = _gfmc_projection_tolerance(_CELL_ZONES[cell])
    ctx = _ctx(ham)
    key = jax.random.PRNGKey(0)
    _, rotation_key = jax.random.split(key)
    kinetic_key, _ = _split_lrdmc_rotation_keys(rotation_key)
    interval_rt = np.asarray(_sample_lrdmc_rotation_matrix(kinetic_key, True).T)

    for nsub in (1, 2, 3):
        comparison_iteration = str(nsub)  # noqa: F841 - captured by the GDD iteration receipt
        interval_rts = np.repeat(interval_rt[None, :, :], nsub, axis=0)
        ref = _xla_interval_n(ham, r_up, r_dn, 0.30, -1.0, nsub, key, nlm, rts=interval_rts)
        got = ctx.run_gfmc_projection_n(
            r_up,
            r_dn,
            0.30,
            -1.0,
            nsub,
            int(key[0]),
            int(key[1]),
            np.zeros((0, 3, 3)),
            _MESH_W,
            _MESH_GP,
            random_mesh=True,
            dltmove=(nlm == "dltmove"),
            nn=1,
        )
        _assert_close(float(got["w_L"]), ref["w_L"], atol, rtol, f"w_L(native RT,{cell},nsub={nsub})")
        _assert_close(float(got["e_L"]), ref["e_L"], atol, rtol, f"e_L(native RT,{cell},nsub={nsub})")
        _assert_close(float(got["V_diag"]), ref["V_diag"], atol, rtol, f"V_diag(native RT,{cell},nsub={nsub})")
        assert np.allclose(np.asarray(got["r_up"]), ref["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        assert np.allclose(np.asarray(got["r_dn"]), ref["r_dn"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        assert list(np.asarray(got["final_key"], dtype=np.uint32)) == list(np.asarray(ref["key"], dtype=np.uint32))


@pytest.mark.light
@pytest.mark.parametrize("cell", _INTERVAL_CELLS)
def test_multistep_projection_t_interval_rotation(cell):
    """GFMC_t native interval matches XLA's two fixed, independent rotations.

    Kinetic and ECP orientations are separately derived and held for the whole
    projection interval. The water row exercises the production spherical-ECP
    route in addition to the H2 Cartesian tmove/dltmove rows.
    """
    ham, r_up, r_dn, nlm = _cell_state(cell)
    atol, rtol = _gfmc_projection_tolerance(_CELL_ZONES[cell])
    ctx = _ctx(ham)
    # GFMC_t's escape-count boundary is not a bitexact cross-backend contract;
    # this pinned two-substep stream exercises interval reuse without landing
    # on a later categorical/escape fp boundary.
    # The water row uses a two-substep interval with one well-separated ECP hop,
    # away from a cross-backend escape-count boundary in both precisions.
    key = jax.random.PRNGKey(2 if cell == "mol_ecp_sphe" else 5)
    tau = 0.006 if cell == "mol_ecp_sphe" else 0.05
    ref = _xla_interval_t(ham, r_up, r_dn, 0.30, tau, key, nlm, random_mesh=True)
    got = ctx.run_gfmc_projection_t(
        r_up,
        r_dn,
        0.30,
        tau,
        int(key[0]),
        int(key[1]),
        _MESH_W,
        _MESH_GP,
        random_mesh=True,
        dltmove=(nlm == "dltmove"),
        nn=1,
    )
    assert int(got["n_substeps"]) == ref["n_substeps"], (cell, got["n_substeps"], ref["n_substeps"])
    _assert_close(float(got["w_L"]), ref["w_L"], atol, rtol, f"w_L(interval RT,{cell})")
    _assert_close(float(got["e_L"]), ref["e_L"], atol, rtol, f"e_L(interval RT,{cell})")
    assert np.allclose(np.asarray(got["r_up"]), ref["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    assert np.allclose(np.asarray(got["r_dn"]), ref["r_dn"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    assert list(np.asarray(got["final_key"], dtype=np.uint32)) == list(np.asarray(ref["key"], dtype=np.uint32))


@pytest.mark.light
@pytest.mark.parametrize(
    ("cell", "non_local_move"),
    [("mol_ecp_sphe", "tmove"), ("mol_ecp_sphe", "dltmove")],
    ids=["water-tmove", "water-dltmove"],
)
def test_native_projection_n_rejects_shared_kinetic_ecp_rotation(cell, non_local_move):
    """GFMC_n native routing uses an ECP orientation independent of kinetic RT.

    On the committed spherical-ECP water fixture, coupling the two orientations
    changes weight, measurement energy, and the exported walker beyond the same
    full/mixed comparison policy used for Rust/XLA parity. The native interval must
    match the independent-orientation reference for both tmove and dltmove.
    """
    ham, r_up, r_dn, _declared_move = _cell_state(cell)
    atol, rtol = _gfmc_projection_tolerance(_CELL_ZONES[cell])
    ctx = _ctx(ham)
    key = jax.random.PRNGKey(21)
    drawn_rts = _rt_stack(3, seed=5)
    interval_rts = np.repeat(drawn_rts[:1], len(drawn_rts), axis=0)
    ref = _xla_interval_n(
        ham,
        r_up,
        r_dn,
        0.30,
        -1.0,
        len(drawn_rts),
        key,
        non_local_move,
        rts=interval_rts,
    )
    coupled = _xla_interval_n(
        ham,
        r_up,
        r_dn,
        0.30,
        -1.0,
        len(drawn_rts),
        key,
        non_local_move,
        rts=interval_rts,
        couple_ecp_to_kinetic=True,
    )
    assert not np.isclose(ref["w_L"], coupled["w_L"], atol=atol, rtol=rtol)
    assert not np.isclose(ref["e_L"], coupled["e_L"], atol=atol, rtol=rtol)
    assert not np.allclose(ref["r_up"], coupled["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)

    got = ctx.run_gfmc_projection_n(
        r_up,
        r_dn,
        0.30,
        -1.0,
        len(drawn_rts),
        int(key[0]),
        int(key[1]),
        drawn_rts,
        _MESH_W,
        _MESH_GP,
        random_mesh=True,
        dltmove=(non_local_move == "dltmove"),
        nn=1,
    )
    _assert_close(float(got["w_L"]), ref["w_L"], atol, rtol, f"w_L({non_local_move})")
    _assert_close(float(got["e_L"]), ref["e_L"], atol, rtol, f"e_L({non_local_move})")
    assert np.allclose(np.asarray(got["r_up"]), ref["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    assert np.allclose(np.asarray(got["r_dn"]), ref["r_dn"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    assert list(np.asarray(got["final_key"], dtype=np.uint32)) == list(np.asarray(ref["key"], dtype=np.uint32))


@pytest.mark.light
@pytest.mark.parametrize(
    ("cell", "non_local_move"),
    [("mol_ecp_sphe", "tmove"), ("mol_ecp_sphe", "dltmove")],
    ids=["water-tmove", "water-dltmove"],
)
def test_native_projection_t_rejects_shared_kinetic_ecp_rotation(cell, non_local_move):
    """GFMC_t native routing uses fixed, independent kinetic and ECP rotations.

    A pinned two-substep water interval has one ECP hop far from its escape-count
    boundary. Sharing the kinetic orientation changes its measurement energy and
    exported walker under both precision policies. The native interval must instead
    match the independent-orientation XLA reference for tmove and dltmove.
    """
    ham, r_up, r_dn, _declared_move = _cell_state(cell)
    atol, rtol = _gfmc_projection_tolerance(_CELL_ZONES[cell])
    ctx = _ctx(ham)
    key = jax.random.PRNGKey(2)
    tau = 0.006
    ref = _xla_interval_t(ham, r_up, r_dn, 0.30, tau, key, non_local_move, random_mesh=True)
    coupled = _xla_interval_t(
        ham,
        r_up,
        r_dn,
        0.30,
        tau,
        key,
        non_local_move,
        random_mesh=True,
        couple_ecp_to_kinetic=True,
    )
    assert ref["n_substeps"] == coupled["n_substeps"] == 2
    assert not np.isclose(ref["e_L"], coupled["e_L"], atol=atol, rtol=rtol)
    assert not np.allclose(ref["r_up"], coupled["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)

    got = ctx.run_gfmc_projection_t(
        r_up,
        r_dn,
        0.30,
        tau,
        int(key[0]),
        int(key[1]),
        _MESH_W,
        _MESH_GP,
        random_mesh=True,
        dltmove=(non_local_move == "dltmove"),
        nn=1,
    )
    assert int(got["n_substeps"]) == ref["n_substeps"]
    _assert_close(float(got["w_L"]), ref["w_L"], atol, rtol, f"w_L({non_local_move})")
    _assert_close(float(got["e_L"]), ref["e_L"], atol, rtol, f"e_L({non_local_move})")
    assert np.allclose(np.asarray(got["r_up"]), ref["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    assert np.allclose(np.asarray(got["r_dn"]), ref["r_dn"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    assert list(np.asarray(got["final_key"], dtype=np.uint32)) == list(np.asarray(ref["key"], dtype=np.uint32))


@pytest.mark.light
@pytest.mark.parametrize("cell", _INTERVAL_CELLS)
def test_multistep_projection_t_parity(cell):
    """GFMC_t: escape-time interval, native vs the XLA reference machine at a pinned key.

    GFMC_t is fast-only BY DESIGN (its sub-step COUNT is an fp function of
    ``log(1-xi)/b_L``) -- that design is not
    tested here as a defect. What IS tested is that with the randomness pinned the
    native interval walks the SAME machine: same sub-step count, same escape-time
    weight product, same measurement e_L and same walker.
    """
    ham, r_up, r_dn, nlm = _cell_state(cell)
    atol, rtol = _gfmc_projection_tolerance(_CELL_ZONES[cell])
    ctx = _ctx(ham)
    key = jax.random.PRNGKey(7)
    for tau in (0.05, 0.20):
        comparison_iteration = str(tau)  # noqa: F841 - captured by the GDD iteration receipt
        ref = _xla_interval_t(ham, r_up, r_dn, 0.30, tau, key, nlm)
        got = ctx.run_gfmc_projection_t(
            r_up,
            r_dn,
            0.30,
            tau,
            int(key[0]),
            int(key[1]),
            _MESH_W,
            _MESH_GP,
            random_mesh=False,
            dltmove=(nlm == "dltmove"),
            nn=1,
        )
        assert int(got["n_substeps"]) == ref["n_substeps"], (cell, tau, got["n_substeps"], ref["n_substeps"])
        _assert_close(float(got["w_L"]), ref["w_L"], atol, rtol, f"w_L[{cell},tau={tau}]")
        _assert_close(float(got["e_L"]), ref["e_L"], atol, rtol, f"e_L[{cell},tau={tau}]")
        assert np.allclose(np.asarray(got["r_up"]), ref["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        assert np.allclose(np.asarray(got["r_dn"]), ref["r_dn"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        # tau_left must be exactly exhausted on the reference side
        assert ref["trace"][-1]["tau_left"] <= 0.0


@pytest.mark.light
@pytest.mark.parametrize("tau", [1.0e-6, 1.0e-4])
def test_projection_t_tau_exhaust_leaves_walker_unmoved(tau):
    """The "all proposals rejected" branch: when ``tau_update == tau_left`` the sub-step
    consumes the remaining projection time and jqmc's
    ``jnp.where(tau_left <= 0.0, r, proposed)`` (Rust: ``if tau_left > 0 {hop}``) keeps
    the OLD configuration even though a categorical index was drawn. Forced with a tau
    far below the escape time so the very first sub-step exhausts it."""
    ham = rs_ham("mol_sphe_sd_jnone")
    atol, rtol = _gfmc_projection_tolerance(_EL_ZONES)
    ctx = _ctx(ham)
    r_up, r_dn = _cfg(ham, 0, scale=0.5)
    key = jax.random.PRNGKey(3)
    ref = _xla_interval_t(ham, r_up, r_dn, 0.30, tau, key, "tmove")
    got = ctx.run_gfmc_projection_t(r_up, r_dn, 0.30, tau, int(key[0]), int(key[1]), _MESH_W, _MESH_GP, random_mesh=False, nn=1)
    assert ref["n_substeps"] == 1 and int(got["n_substeps"]) == 1
    assert ref["trace"][0]["k"] is None  # hop drawn but not applied
    assert np.array_equal(ref["r_up"], r_up) and np.array_equal(ref["r_dn"], r_dn)
    assert np.array_equal(np.asarray(got["r_up"]), r_up), "native applied a hop after tau exhausted"
    assert np.array_equal(np.asarray(got["r_dn"]), r_dn)
    _assert_close(float(got["w_L"]), ref["w_L"], atol, rtol, "w_L(tau exhausted)")
    _assert_close(float(got["e_L"]), ref["e_L"], atol, rtol, "e_L(tau exhausted)")


# =========================================================================== #
# 5. Complex Bloch twist (k != Gamma), in-cell and far PBC
# =========================================================================== #
@pytest.mark.light
@pytest.mark.parametrize(
    "pattern",
    [
        "twist_sphepgto_jnone",
        "twist_sphepgto_j12",
        "twist_sphepgto_j12_pade",
        "twist_sphepgto_j123",
        "twist_sphepgto_jdtn",
    ],
)
@pytest.mark.parametrize("far", [False, True])
def test_multistep_projection_complex_twist_parity(pattern, far):
    """Complex-twist projection interval (both drivers), native vs the complex XLA
    reference machine. The fixed-phase Re-projection enters at exactly three points --
    the escape denominator ``Re(b_L)``, the weight exponent ``Re(e_L)`` / ``b_x``, and
    the FN categorical over ``Re(min(.,0))`` -- and the measurement e_L stays complex."""
    ham = rs_ham(pattern)
    atol, rtol = _gfmc_projection_tolerance(_EL_ZONES_PBC)
    ctx = _ctx(ham, complex_=True)
    r_up, r_dn = _cfg(ham, 4, cell_shift=[[2.0, -3.0, 1.0]] if far else None)
    key = jax.random.PRNGKey(13)
    jqmc_rs.gfmc_route_counters_reset()

    ref_t = _xla_interval_t_complex(ham, r_up, r_dn, 0.30, 0.10, key)
    got_t = ctx.run_gfmc_projection_t_complex(
        r_up, r_dn, 0.30, 0.10, int(key[0]), int(key[1]), _MESH_W, _MESH_GP, random_mesh=False, nn=1
    )
    assert int(got_t["n_substeps"]) == ref_t["n_substeps"]
    _assert_close(float(got_t["w_L"]), ref_t["w_L"], atol, rtol, "complex w_L (t)")
    _assert_close(complex(got_t["e_L"]).real, ref_t["e_L"].real, atol, rtol, "complex Re(e_L) (t)")
    _assert_close(complex(got_t["e_L"]).imag, ref_t["e_L"].imag, atol, rtol, "complex Im(e_L) (t)")
    assert np.allclose(np.asarray(got_t["r_up"]), ref_t["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)

    ref_n = _xla_interval_n_complex(ham, r_up, r_dn, 0.30, -1.0, 3, key)
    got_n = ctx.run_gfmc_projection_n_complex(
        r_up,
        r_dn,
        0.30,
        -1.0,
        3,
        int(key[0]),
        int(key[1]),
        np.zeros((0, 3, 3)),  # empty injected_rts -> FAST chain, as the real twin's tests pass
        _MESH_W,
        _MESH_GP,
        random_mesh=False,
        nn=1,
    )
    _assert_close(float(got_n["w_L"]), ref_n["w_L"], atol, rtol, "complex w_L (n)")
    _assert_close(complex(got_n["e_L"]).real, ref_n["e_L"].real, atol, rtol, "complex Re(e_L) (n)")
    _assert_close(complex(got_n["e_L"]).imag, ref_n["e_L"].imag, atol, rtol, "complex Im(e_L) (n)")
    _assert_close(float(got_n["V_diag"]), ref_n["V_diag"], atol, rtol, "complex V_diag (n)")
    assert np.allclose(np.asarray(got_n["r_up"]), ref_n["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    assert np.allclose(np.asarray(got_n["r_dn"]), ref_n["r_dn"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    if pattern == "twist_sphepgto_jdtn":
        counters = jqmc_rs.gfmc_route_counters()
        tabpip = {name: counters[idx] for name, idx in {"builds": 45, "advances": 46, "hits": 47}.items()}
        assert all(tabpip.values()), f"complex DTN candidate table did not build/advance/serve rows: {tabpip}"
        n_e = r_up.shape[0] + r_dn.shape[0]
        assert tabpip["hits"] % n_e == 0, f"DTN table did not serve complete all-electron kinetic meshes: {tabpip}"
        assert ham.coulomb_potential_data.ecp_flag, "DTN ECP liveness row must carry a non-local ECP"
        ecp_tabpip = {name: counters[idx] for name, idx in {"builds": 61, "advances": 62, "hits": 63}.items()}
        assert all(ecp_tabpip.values()), f"complex ECP DTN table did not build/advance/serve rows: {ecp_tabpip}"
    counters = jqmc_rs.gfmc_route_counters()
    winv = {
        name: counters[idx]
        for name, idx in {"builds": 56, "advances": 57, "kin_hits": 58, "ecp_hits": 59, "fallbacks": 60}.items()
    }
    assert all(winv[name] for name in ("builds", "advances", "kin_hits", "ecp_hits")) and winv["fallbacks"] == 0, winv
    if ham.wavefunction_data.jastrow_data.jastrow_three_body_data is not None:
        assert counters[22] == 0, (pattern, far, "complex J3 candidate table fell back", counters[22])
    if "j12" in pattern:
        counters = jqmc_rs.gfmc_route_counters()
        kin = {name: counters[idx] for name, idx in {"builds": 48, "advances": 49, "hits": 50, "fallbacks": 51}.items()}
        ecp = {name: counters[idx] for name, idx in {"builds": 52, "advances": 53, "hits": 54, "fallbacks": 55}.items()}
        assert all(kin[name] for name in ("builds", "advances", "hits")) and kin["fallbacks"] == 0, kin
        assert all(ecp[name] for name in ("builds", "advances", "hits")) and ecp["fallbacks"] == 0, ecp


# =========================================================================== #
# 6. The reference machine is itself pinned against the production XLA driver
# =========================================================================== #
@contextlib.contextmanager
def _backend_env(mode):
    """Select pure XLA or the single production Rust backend for this block."""
    _prev_flag = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
    if mode == "jax":
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    else:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    try:
        yield
    finally:
        if _prev_flag is None:
            os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
        else:
            os.environ["JQMC_FLAG_RUST_ON_CPU"] = _prev_flag


def _run_driver(ham, mode, *, nlm="tmove", tau=0.05, steps=3, seed=34, walkers=1):
    """One short production GFMC_t run in ``mode`` in {"jax", "fast"}.

    The run is spied and the BACKEND IS ASSERTED: a ``fast`` side built purely by
    popping ``JQMC_FLAG_RUST_ON_CPU`` and trusting the default to be True would leave
    nothing verifying that the native GFMC interval actually ran. A flipped
    default or a silent gate fallback would make the ``jax`` and ``fast`` runs identical
    XLA runs, and the XLA-vs-native rows in this module would pass without ever exercising
    Rust. Runs that raise (the unsupported-``non_local_move`` error cells) or that request
    no steps never reach the assertion.
    """
    with _backend_env(mode):
        from jqmc.jqmc_gfmc import GFMC_t

        g = GFMC_t(
            hamiltonian_data=ham,
            num_walkers=walkers,
            num_gfmc_collect_steps=2,
            mcmc_seed=seed,
            alat=0.30,
            non_local_move=nlm,
            tau=tau,
            random_discretized_mesh=False,
        )
        if steps > 0:
            with native_ctx_spy() as counts:
                g.run(num_mcmc_steps=steps)
            if mode == "jax":
                assert_no_native_ran(counts, f"GFMC projection backend 'jax' (nlm={nlm})")
            else:
                assert_native_ran(counts, f"GFMC projection backend {mode!r} (nlm={nlm})")
        return g


@pytest.mark.heavy
@pytest.mark.parametrize(
    "pattern,nlm,zones",
    [
        ("mol_sphe_sd_jnone", "tmove", _EL_ZONES),
        ("mol_ecp_cart_jnone", "tmove", _EL_ZONES),
        ("mol_ecp_cart_j12", "dltmove", _EL_ZONES_J),
    ],
)
def test_projection_reference_machine_matches_production_driver(pattern, nlm, zones):
    """Anti-drift pin for the ORACLE: the module's ``_xla_step`` must equal the real
    ``_projection_t_core``.

    Isolated by making the interval EXACTLY one sub-step with no hop: a tau far below
    the escape time means ``tau_update == tau_left`` immediately, so the driver's stored
    ``e_L`` is the continuum measurement energy evaluated at the driver's OWN initial
    walker with ``R^T = I``, while ``w_L`` is exactly ``exp(-tau e_L^propagation)``.
    Both definitions are recomputed separately by the reference machine. If the
    transcription ever drifts from the closure (a changed ZV arm, S' fold, DLA
    convention, measurement composition, ...) this row goes red before any
    XLA-vs-Rust row can be fooled by a matching-but-wrong oracle. The dedicated
    non-degenerate split test below proves that the two definitions are not merely
    interchangeable on a zero-variance fixture.
    """
    ham = rs_ham(pattern)
    atol, rtol = _gfmc_projection_tolerance(zones)
    tau = 1.0e-6
    with _backend_env("jax"):
        g = _run_driver(ham, "jax", nlm=nlm, tau=tau, steps=0, walkers=1)
        r_up0 = np.asarray(g._latest_r_up_carts, dtype=np.float64)[0].copy()
        r_dn0 = np.asarray(g._latest_r_dn_carts, dtype=np.float64)[0].copy()
        g.run(num_mcmc_steps=1)
    e_driver = float(np.asarray(g._stored_e_L).ravel()[0])
    w_driver = float(np.asarray(g._stored_w_L).ravel()[0])
    # tau exhausted on the first sub-step -> the walker must be untouched
    assert np.array_equal(np.asarray(g._latest_r_up_carts, dtype=np.float64)[0], r_up0)
    assert np.array_equal(np.asarray(g._latest_r_dn_carts, dtype=np.float64)[0], r_dn0)
    ref = _xla_step(ham, r_up0, r_dn0, 0.30, np.eye(3), nlm)
    _assert_close(e_driver, ref["e_L_meas"], atol, rtol, f"oracle drift: driver measurement e_L [{pattern},{nlm}]")
    _assert_close(w_driver, float(np.exp(-tau * ref["e_L"])), atol, rtol, "oracle drift: driver w_L")
    # The Rust single-step kernel exposes the propagation energy, not the stored
    # continuum measurement.  Pin that distinct contract as well.
    rs_e, _rs_b = _rs_step_eL(ham, r_up0, r_dn0, 0.30, np.eye(3), nlm, fast=True)
    _assert_close(rs_e, ref["e_L"], atol, rtol, "native step propagation e_L vs _xla_step")


@pytest.mark.heavy
@pytest.mark.parametrize("algorithm", ["conventional", "load-balanced"])
@pytest.mark.parametrize("backend", ["xla", "rust"])
def test_novar0_propagation_and_measurement_energies_are_routed_separately(algorithm, backend):
    """At novar=0, both production GFMC algorithms and both backends must use the
    clipped lattice local energy only for propagation, while storing the continuum
    local energy as the measurement observable.

    The committed ``tests/trexio_example_files/N_ae_ccpvdz_cart.h5`` fixture is
    evaluated at seed 0, scale 0.5, ``alat=0.30``, and ``tau=0.001``. This forces a mixed
    per-electron sign-flip mask and separates the two local energies by about 21 Ha,
    so swapping the routes cannot pass through a zero-variance coincidence. The
    XLA replay gives the expected measurement and propagation weight, using the
    production closure for the propagation energy. The same GDD guarantee also
    requires ``test_novar0_propagation_matches_independent_leaf_assembly`` to
    anchor that closure against independently summed leaf terms at the initial
    and first-hop states. This routing test rejects the two deliberately swapped
    formulas and proves that the requested production backend interval ran.
    """
    ham = _build_ham_for_cell("N_ae_ccpvdz_cart.h5", False, None, "SD")
    r_up0, r_dn0 = _cfg(ham, seed=0, scale=0.5)
    alat = 0.30
    # The fixed-key first escape time is about 0.00429, so 0.001 guarantees one
    # no-hop sub-step while keeping the two exponential weights well separated
    # even under the centrally declared mixed-precision tolerance.
    tau = 1.0e-3
    e_scf = -1.0
    atol, rtol = _gfmc_projection_tolerance(_EL_ZONES)
    initial = _xla_step(ham, r_up0, r_dn0, alat, np.eye(3), "tmove")

    n_flip = int(np.count_nonzero(initial["sign_flip"]))
    assert 0 < n_flip < int(initial["sign_flip"].size), "fixture must force the mixed sign-flip branch"
    assert not np.isclose(initial["e_L"], initial["e_L_meas"], atol=atol, rtol=rtol), (
        "propagation and measurement energies are degenerate; the split test is vacuous",
        initial["e_L"],
        initial["e_L_meas"],
    )

    env_mode = "jax" if backend == "xla" else "fast"
    with _backend_env(env_mode):
        if algorithm == "conventional":
            from jqmc.jqmc_gfmc import GFMC_t

            g = GFMC_t(
                hamiltonian_data=ham,
                num_walkers=1,
                num_gfmc_collect_steps=2,
                mcmc_seed=34,
                tau=tau,
                alat=alat,
                random_discretized_mesh=False,
                non_local_move="tmove",
            )
            g._latest_r_up_carts = jnp.asarray(r_up0[np.newaxis, ...])
            g._latest_r_dn_carts = jnp.asarray(r_dn0[np.newaxis, ...])
            expected_measurement = initial["e_L_meas"]
            expected_propagation = initial["e_L"]
            expected_weight = float(np.exp(-tau * expected_propagation))
            wrong_weight = float(np.exp(-tau * expected_measurement))
            expected_r_up = r_up0
            expected_r_dn = r_dn0
            with native_ctx_spy() as counts:
                g.run(num_mcmc_steps=1)
        else:
            from jqmc.jqmc_gfmc import GFMC_n

            g = GFMC_n(
                hamiltonian_data=ham,
                num_walkers=1,
                num_mcmc_per_measurement=1,
                num_gfmc_collect_steps=2,
                mcmc_seed=34,
                E_scf=e_scf,
                alat=alat,
                random_discretized_mesh=False,
                non_local_move="tmove",
            )
            g._latest_r_up_carts = jnp.asarray(r_up0[np.newaxis, ...])
            g._latest_r_dn_carts = jnp.asarray(r_dn0[np.newaxis, ...])
            initial_key = np.asarray(g._jax_PRNG_key_list[0], dtype=np.uint32).copy()
            ref = _xla_interval_n(ham, r_up0, r_dn0, alat, e_scf, 1, initial_key, "tmove")
            expected_measurement = ref["e_L"]
            expected_propagation = ref["e_L_propagation"]
            expected_weight = ref["w_L"]
            wrong_v_diag = initial["e_L_meas"] - initial["b_L"]
            wrong_weight = (-initial["b_L"]) / (wrong_v_diag - e_scf)
            expected_r_up = ref["r_up"]
            expected_r_dn = ref["r_dn"]
            with native_ctx_spy() as counts:
                g.run(num_mcmc_steps=1)

    assert_backend_route = assert_no_native_ran if backend == "xla" else assert_native_ran
    assert_backend_route(counts, f"GFMC {algorithm} novar=0 split {backend} route")

    e_driver = float(np.real(np.asarray(g._stored_e_L).ravel()[0]))
    w_driver = float(np.asarray(g._stored_w_L).ravel()[0])
    np.testing.assert_allclose(np.asarray(g._latest_r_up_carts)[0], expected_r_up, atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(g._latest_r_dn_carts)[0], expected_r_dn, atol=atol, rtol=rtol)
    assert not np.isclose(expected_measurement, expected_propagation, atol=atol, rtol=rtol), (
        "final measurement and propagation energies must remain distinguishable",
        expected_measurement,
        expected_propagation,
    )
    assert not np.isclose(expected_weight, wrong_weight, atol=atol, rtol=rtol), (
        "propagation and measurement weight formulas are degenerate",
        expected_weight,
        wrong_weight,
    )
    np.testing.assert_allclose(e_driver, expected_measurement, atol=atol, rtol=rtol)
    np.testing.assert_allclose(w_driver, expected_weight, atol=atol, rtol=rtol)
    assert not np.isclose(e_driver, expected_propagation, atol=atol, rtol=rtol)
    assert not np.isclose(w_driver, wrong_weight, atol=atol, rtol=rtol)


# =========================================================================== #
# 6b. Fixed-snapshot native-vs-PRODUCTION shadow oracle for GFMC_t.
#
# The native GFMC_t interval had no deterministic PRODUCTION-vs-PRODUCTION oracle:
# bitexact GFMC_t is JAX-vs-JAX (native runs zero times), fast GFMC_t is a 5-sigma
# statistical band, and the only thing tying a native interval to XLA was the
# replica machine of section 1-4 above. A byte-exact native GFMC_t chain is not
# achievable because the escape-time sub-step COUNT is fp-dependent, which is why a chain-level comparison cannot close
# the gap. This oracle side-steps that: it advances the native chain once to a
# measurement-point snapshot, then drives BOTH the native interval AND a loop of
# the hoisted PRODUCTION ``_projection_t_core`` (the exact closure the driver's
# projection step runs) from the SAME snapshot and the SAME pinned key -- so the
# sub-step count is deterministic and identical on both sides, and the divergence,
# if any, is a real defect rather than the documented fp escape-count drift.
# =========================================================================== #
def _production_interval_t(ham, r_up, r_dn, alat, tau, key, non_local_move):
    """One GFMC_t escape-time interval driven by the PRODUCTION ``_projection_t_core``.

    Loops the hoisted closure (legacy path: continuum kinetic recomputed each sub-step
    from the running SM inverse, ``j3_state=None``, ``det_ratio_state=None``,
    ``random_discretized_mesh=False``) exactly as ``_projection_t`` does inside
    ``GFMC_t.run()``. Returns the final ``(w_L, e_L, r_up, r_dn, key)`` and the
    sub-step count.
    """
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    geminal = compute_geminal_all_elements(geminal_data=gd, r_up_carts=r_up, r_dn_carts=r_dn)
    U, s, Vt = jnp.linalg.svd(geminal, full_matrices=False)
    eps = get_eps("rcond_svd", jnp.float64)
    s_inv = jnp.where(s > eps * s[0], 1.0 / s, 0.0)
    A_inv = (jnp.conj(Vt).T * s_inv[jnp.newaxis, :]) @ jnp.conj(U).T

    pc = 0
    tau_left = jnp.asarray(tau, dtype=jnp.float64)
    w_L = jnp.asarray(1.0, dtype=jnp.float64)
    r_up = jnp.asarray(r_up)
    r_dn = jnp.asarray(r_dn)
    n_sub = 0
    e_last = 0.0 + 0.0j
    interval_RT = jnp.eye(3, dtype=jnp.float64)
    interval_ecp_RT = jnp.eye(3, dtype=jnp.float64)
    while float(jnp.real(tau_left)) > 0.0:
        n_sub += 1
        ke_up, ke_dn = compute_kinetic_energy_all_elements_fast_update(
            wavefunction_data=wf, r_up_carts=r_up, r_dn_carts=r_dn, geminal_inverse=A_inv
        )
        # ``pot_state`` stays at its default (None) so this oracle loop is the
        # FULL-RECOMPUTE route (IMPL-GFMC-INCREMENTAL-TWIN-001 debug twin); the
        # native interval it is compared against runs the incremental
        # potential, and the walk (sub-step count / key / positions) is
        # route-invariant by construction.
        (e_L, pc, tau_left, w_L, r_up, r_dn, A_inv, key, _RT, _ecp_RT, _hu, _ui, _di, _pot) = _jit_projection_t_core(
            pc,
            tau_left,
            w_L,
            r_up,
            r_dn,
            A_inv,
            key,
            ke_up,
            ke_dn,
            None,
            interval_RT,
            interval_ecp_RT,
            False,
            non_local_move,
            alat,
            ham,
        )
        interval_RT = _RT
        interval_ecp_RT = _ecp_RT
        e_last = complex(e_L)
    return {
        "w_L": float(jnp.real(w_L)),
        "e_L": e_last,
        "r_up": np.asarray(r_up),
        "r_dn": np.asarray(r_dn),
        "key": np.asarray(key),
        "n_substeps": n_sub,
    }


@pytest.mark.light
@pytest.mark.parametrize("cell", ["mol_ae_sphe", "mol_ecp_tmove", "mol_ecp_dltmove", "pbc_gamma"])
def test_gfmc_t_fixed_snapshot_native_vs_production_core(cell):
    """Native GFMC_t interval == a loop of the PRODUCTION ``_projection_t_core``
    on a FIXED measurement-point snapshot.

    The snapshot is produced by advancing the native chain ONE interval from the
    deterministic config, so the compared interval starts from a state the projector
    actually visits. From that snapshot, at a pinned key, the native interval and the
    production-core loop must agree on: the sub-step COUNT (deterministic here because
    both consume the same escape-time draws), the escape-time weight product ``w_L``,
    the measurement ``e_L``, the walker, and the final threefry key.
    """
    ham, r_up, r_dn, nlm = _cell_state(cell)
    atol, rtol = _gfmc_projection_tolerance(_CELL_ZONES[cell])
    ctx = _ctx(ham)
    alat, tau = 0.30, 0.20
    dlt = nlm == "dltmove"

    # advance one native interval -> measurement-point snapshot
    k0 = jax.random.PRNGKey(101)
    adv = ctx.run_gfmc_projection_t(
        r_up, r_dn, alat, tau, int(k0[0]), int(k0[1]), _MESH_W, _MESH_GP, random_mesh=False, dltmove=dlt, nn=1
    )
    snap_up = np.ascontiguousarray(np.asarray(adv["r_up"], dtype=np.float64))
    snap_dn = np.ascontiguousarray(np.asarray(adv["r_dn"], dtype=np.float64))

    key = jax.random.PRNGKey(202)
    nat = ctx.run_gfmc_projection_t(
        snap_up, snap_dn, alat, tau, int(key[0]), int(key[1]), _MESH_W, _MESH_GP, random_mesh=False, dltmove=dlt, nn=1
    )
    prod = _production_interval_t(ham, snap_up, snap_dn, alat, tau, key, nlm)

    assert int(nat["n_substeps"]) == prod["n_substeps"], (cell, nat["n_substeps"], prod["n_substeps"])
    _assert_close(float(nat["w_L"]), prod["w_L"], atol, rtol, f"w_L[{cell}]")
    _assert_close(complex(nat["e_L"]).real, prod["e_L"].real, atol, rtol, f"e_L[{cell}]")
    assert np.allclose(np.asarray(nat["r_up"]), prod["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    assert np.allclose(np.asarray(nat["r_dn"]), prod["r_dn"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    assert list(np.asarray(nat["final_key"], dtype=np.uint32)) == list(np.asarray(prod["key"], dtype=np.uint32)), (
        cell,
        "final threefry key diverged",
    )


# =========================================================================== #
# 7. ERROR cell (NOT a comparison cell): unsupported ``non_local_move``
# =========================================================================== #
@pytest.mark.light
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="the invalid non_local_move option is rejected before walker initialization"
)
def test_non_local_move_unsupported_raises_on_xla():
    """XLA reference behaviour: an ECP run whose ``non_local_move`` is neither
    ``tmove`` nor ``dltmove`` (including ``None``) raises ``NotImplementedError``
    from ``_projection_t_core``. This is the specified ERROR cell."""
    ham = rs_ham("mol_ecp_cart_jnone")
    for nlm in (None, "bogus"):
        with pytest.raises(NotImplementedError):
            _run_driver(ham, "jax", nlm=nlm, steps=1)


@pytest.mark.light
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="the invalid non_local_move option is rejected before walker initialization"
)
def test_non_local_move_unsupported_raises_on_native_chunk():
    """Non local move unsupported raises on native chunk."""
    ham = rs_ham("mol_ecp_cart_jnone")
    for nlm in (None, "bogus"):
        with pytest.raises(NotImplementedError):
            _run_driver(ham, "fast", nlm=nlm, steps=1)


def _rs_step_eL_complex(ham, r_up, r_dn, alat, RT, non_local_move, nn):
    """One complex-twist ECP projection sub-step e_L (native), tmove or dltmove."""
    wf = ham.wavefunction_data
    gd, jd = wf.geminal_data, wf.jastrow_data
    cd = ham.coulomb_potential_data
    e_re, e_im, _b_re, _b_im, _n = jqmc_rs._gfmc_lrdmc_ecp_complex(
        gd, jd, cd, r_up, r_dn, alat, RT, _MESH_W, _MESH_GP, nn, non_local_move == "dltmove"
    )
    return complex(e_re, e_im)


@pytest.mark.heavy
def test_complex_twist_chunk_honours_dltmove():
    """Twist ECP: the DLA ``dltmove`` fold runs in the complex
    chunk (``gfmc_lrdmc_ecp_complex``): SP folds ``max(Re(det_only), 0)``, the FN
    off-diagonal is ``min(Re(det_only), 0) * Jastrow_ratio``, with the fixed-phase Re at
    the source on the det-only ratio. ``_resolve_native_gfmc_chunk`` accepts complex
    ``dltmove``.

    Asserts (a) XLA honours the tmove/dltmove distinction on the complex path (the DLA
    fold is not a no-op), and (b) the native complex e_L reproduces the XLA complex e_L
    for BOTH tmove and dltmove at a fixed twist config (a per-config parity)."""
    ham = rs_ham("twist_sphepgto_j12")
    # Seed 11 makes the physical tmove/dltmove split robustly larger than the
    # mixed backend-parity envelope (about 33x on JAX 0.7.2); a seed whose
    # nonzero difference is smaller than medium is not a useful liveness fixture.
    r_up, r_dn = _cfg(ham, seed=11)
    alat, RT, nn = 0.30, np.eye(3), NN_default
    xt = _xla_step_complex(ham, r_up, r_dn, alat, RT, non_local_move="tmove")["e_L"]
    xd = _xla_step_complex(ham, r_up, r_dn, alat, RT, non_local_move="dltmove")["e_L"]
    atol, rtol = get_tolerance_xla_rust_consistency(_EL_ZONES_PBC, level="medium")
    assert abs(xt.real - xd.real) > atol + rtol * abs(xt.real), "XLA fixture does not separate tmove/dltmove"
    nt = _rs_step_eL_complex(ham, r_up, r_dn, alat, RT, "tmove", nn)
    nd = _rs_step_eL_complex(ham, r_up, r_dn, alat, RT, "dltmove", nn)
    _assert_close(nt.real, xt.real, atol, rtol, "native complex tmove e_L != XLA")
    _assert_close(nd.real, xd.real, atol, rtol, "native complex dltmove e_L != XLA (DLA fold port)")
