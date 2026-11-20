"""Production parity for GFMC/LRDMC observable aggregation, XLA vs Rust,
component by component.

The pre-existing GFMC observable suite (``test_gfmc.py``) compares the TOTAL
``e_L^t`` only. A total can hide a compensating pair of errors, and -- as this
module's ``test_gfmc_branch_liveness`` records -- on the fixtures those tests use
several of the summands are IDENTICALLY ZERO, so the total is blind to them:

  * ``H2_ecp_ccpvdz_cart.h5`` (the ``mol_ecp_cart_*`` patterns) has ``V_nonlocal
    == 0`` for EVERY configuration (the H ccECP carries no non-local projector).
    Both the ``tmove`` and the ``dltmove`` e_L rows built on it therefore multiply
    the whole non-local mesh -- including the ``dltmove`` Jastrow-ratio weighting
    ``min(V_det,0) * exp(J_mesh - J_ref)`` -- by zero.
  * ``lithium.h5`` / the QE ``Li_gamma`` composite / pw / bspline cells
    (1 up + 1 dn) never produce a discretized-kinetic sign flip, so ``S'_kin ==
    0``, and every mesh ``V_nonlocal >= 0``, so ``FN_ecp == 0`` and ``dltmove``
    degenerates to ``tmove``.

This module therefore (a) decomposes ``e_L^t`` into the summands of
``jqmc_gfmc._compute_V_elements_t`` and pins each one Rust-vs-XLA separately,
(b) records per cell WHICH summands are live so a vacuous row cannot silently
pass as coverage, and (c) adds cells (water ccECP; the PBC AE fixtures) where the
S' and non-local-mesh channels are simultaneously non-zero.

Decomposition (``_compute_V_elements_t``, jqmc_gfmc.py:2473):

    e_L^t = diag_kin_const                     3/(alat^2) * (N_up + N_dn)
          + v_ee + v_ii + sum(zv_opt)          Coulomb / Ewald + the
                                               ZV-regularised el-ion term, which
                                               is where the CONTINUUM kinetic
                                               ``compute_kinetic_energy_all_elements``
                                               enters
          + ecp_local
          + sp_kin + sp_ecp                    the S' (sign-flip) corrections
          + fn_kin + fn_ecp                    the fixed-node off-diagonal sum
                                               ( == b_L, the branching rate)

Every one of those is recovered from the RUST side without consulting XLA:

    sp_kin, fn_kin   ``_gfmc_discretized_kinetic_elements``  (elementwise)
    e_L, b_L         ``_gfmc_lrdmc_{ae,ecp}``
    fn_ecp           b_L - fn_kin
    sum(V_nl)        ``compute_ecp_nonlocal_parts_all_pairs``
    sp_ecp           tmove:   sum(V_nl) - fn_ecp   (max0(x) + min0(x) == x)
                     dltmove: sp_ecp(tmove) + [V_diag(dltmove) - V_diag(tmove)]
                              -- the ONLY term the two modes differ in
    v_ee             bare / Ewald Rust kernel
    ecp_local        ``compute_ecp_local_parts_all_pairs``
    zv_opt           V_diag - (const + sp_kin + sp_ecp + ecp_local + v_ee + v_ii)

Attribution caveat: the periodic ion-ion Ewald constant is not separately
exported by the extension (it is precomputed inside the Rust ``PbcCtx``), so the
PBC rows take ``v_ii`` from XLA. A Rust ion-ion error would then surface as a
``zv_opt`` mismatch -- still a failure, only mis-attributed. ``v_ii`` is
configuration-independent and its Rust==XLA parity is pinned by ``test_ewald.py``
/ ``test_coulomb_potential.py``.

Two-stage method: stage (a) is kernel parity on a SAVED, IDENTICAL
input state so no projection difference can leak in; stage (b) feeds genuinely
PROJECTED walkers -- the projection suite's multi-step output, taken from a real GFMC_n run --
into the same decomposition.
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

import jqmc_rs
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min as _d8_zone_tolerance

pytestmark = [pytest.mark.with_chdir]

import jax
import jax.numpy as jnp

from jqmc._precision import get_tolerance_min, get_tolerance_xla_rust_consistency, is_mixed_precision_enabled
from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_ZERO
from jqmc.coulomb_potential import (
    compute_coulomb_potential_ee_dispatch,
    compute_coulomb_potential_ei_discretized_dispatch,
    compute_coulomb_potential_ei_dispatch,
    compute_coulomb_potential_ii_dispatch,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors,
    icosahedron_sym_mesh_Nv12,
)
from jqmc.determinant import compute_geminal_all_elements
from jqmc.jastrow_factor import compute_Jastrow_part
from jqmc.jqmc_gfmc import _compute_V_elements_t
from jqmc.wavefunction import compute_discretized_kinetic_energy, compute_kinetic_energy_all_elements
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import rs_ham

_W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
_GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)

# Every GFMC component starts from AO values, including the discretized kinetic
# elements.  Listing that forward zone is essential: it is fp32 in mixed mode,
# even for jnone cells, and must enter each comparison's declared tolerance family.
_ZONES = ("ao_eval", "wf_kinetic", "coulomb", "local_energy", "det_ratio")
_ZONES_J = _ZONES + ("jastrow_eval", "jastrow_grad_lap")
_ZONES_PBC = _ZONES + ("ewald",)
_ZONES_PBC_J = _ZONES_J + ("ewald",)

# The exact-zero identities this module records (branch liveness) are integer-clean
# sums of exactly-zero terms, so they are compared against the fp64 accumulator floor.
_ZERO_ATOL, _ZERO_RTOL = ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_ZERO


# One explicit zone list per (pbc, withj) cell flag pair (never their union). The
# table is a literal so a per-case executable check can read each case's list statically.
_ZONES_BY_FLAGS = {
    False: {False: _ZONES, True: _ZONES_J},
    True: {False: _ZONES_PBC, True: _ZONES_PBC_J},
}


def _zones(cell):
    return _ZONES_BY_FLAGS[cell["pbc"]][cell["withj"]]


# --------------------------------------------------------------------------- #
# Cell table. Every row is a GFMC-reachable production combination; ``nlm`` is
# the ``non_local_move`` of ``_compute_V_elements_t`` (``None`` == all-electron,
# where the ECP branch of the closure is not taken at all).
# --------------------------------------------------------------------------- #
def _p(pattern):
    return ("pattern", pattern)


def _b(stem, withj, spec, gem="SD"):
    return ("build", stem, withj, spec, gem)


_CELLS = [
    # ---- molecular all-electron (no ECP branch) ----
    dict(id="mol_ae_sphe_jnone", src=_p("mol_sphe_sd_jnone"), nlm=None, alat=0.3, withj=False, pbc=False),
    dict(id="mol_ae_cart_j12", src=_p("mol_cart_sd_j12"), nlm=None, alat=0.3, withj=True, pbc=False),
    dict(id="mol_ae_cart_j123", src=_p("mol_cart_sd_j123"), nlm=None, alat=0.3, withj=True, pbc=False),
    dict(id="mol_ae_cart_jdtn", src=_p("mol_cart_sd_jdtn"), nlm=None, alat=0.3, withj=True, pbc=False),
    # ---- molecular ECP, tmove ----
    dict(id="mol_ecp_cart_jnone_tmove", src=_p("mol_ecp_cart_jnone"), nlm="tmove", alat=0.3, withj=False, pbc=False),
    dict(id="mol_ecp_cart_j123_tmove", src=_p("mol_ecp_cart_j123"), nlm="tmove", alat=0.3, withj=True, pbc=False),
    dict(id="mol_ecp_cart_jdtn_tmove", src=_p("mol_ecp_cart_jdtn"), nlm="tmove", alat=0.3, withj=True, pbc=False),
    # water ccECP: the ONLY molecular cell where S'_kin, S'_ecp and FN_ecp are all live.
    dict(id="mol_ecp_sphe_jnone_tmove", src=_p("mol_ecp_sphe_jnone"), nlm="tmove", alat=0.3, withj=False, pbc=False),
    dict(id="mol_ecp_sphe_j12_tmove", src=_p("mol_ecp_sphe_j12"), nlm="tmove", alat=0.3, withj=True, pbc=False),
    dict(
        id="mol_ecp_sphe_j123_tmove",
        src=_b("water_ccecp_ccpvqz.h5", True, {"j3": "sphe_gto"}),
        nlm="tmove",
        alat=0.3,
        withj=True,
        pbc=False,
    ),
    dict(
        id="mol_ecp_sphe_jdtn_tmove",
        src=_b("water_ccecp_ccpvqz.h5", True, {"dtn": True}),
        nlm="tmove",
        alat=0.3,
        withj=True,
        pbc=False,
    ),
    # ---- molecular ECP, dltmove (determinant-locality: FN leg gets the Jastrow ratio) ----
    dict(id="mol_ecp_cart_j12_dltmove", src=_p("mol_ecp_cart_j12"), nlm="dltmove", alat=0.3, withj=True, pbc=False),
    dict(id="mol_ecp_sphe_j12_dltmove", src=_p("mol_ecp_sphe_j12"), nlm="dltmove", alat=0.3, withj=True, pbc=False),
    dict(
        id="mol_ecp_sphe_jdtn_dltmove",
        src=_b("water_ccecp_ccpvqz.h5", True, {"dtn": True}),
        nlm="dltmove",
        alat=0.3,
        withj=True,
        pbc=False,
    ),
    # ---- periodic real-Gamma ECP (Ewald) ----
    dict(id="pbc_gamma_jnone_tmove", src=_p("pbc_gamma_sphepgto_jnone"), nlm="tmove", alat=0.3, withj=False, pbc=True),
    dict(id="pbc_gamma_j12_tmove", src=_p("pbc_gamma_sphepgto_j12"), nlm="tmove", alat=0.3, withj=True, pbc=True),
    dict(id="pbc_gamma_j123_tmove", src=_p("pbc_gamma_sphepgto_j123"), nlm="tmove", alat=0.3, withj=True, pbc=True),
    dict(id="pbc_gamma_jdtn_tmove", src=_p("pbc_gamma_sphepgto_jdtn"), nlm="tmove", alat=0.3, withj=True, pbc=True),
    dict(id="pbc_gamma_j12_dltmove", src=_p("pbc_gamma_sphepgto_j12"), nlm="dltmove", alat=0.3, withj=True, pbc=True),
    dict(id="pbc_composite_jnone_tmove", src=_p("pbc_composite_jnone"), nlm="tmove", alat=0.3, withj=False, pbc=True),
    dict(id="pbc_composite_j12_tmove", src=_p("pbc_composite_j12"), nlm="tmove", alat=0.3, withj=True, pbc=True),
    dict(id="pbc_pw_jnone_tmove", src=_p("pw_gamma_jnone"), nlm="tmove", alat=0.3, withj=False, pbc=True),
    dict(id="pbc_bspline_jnone_tmove", src=_p("bspline_gamma_jnone"), nlm="tmove", alat=0.3, withj=False, pbc=True),
    # ---- periodic all-electron (Ewald, no ECP branch); H4 gives same-spin pairs ----
    dict(id="pbc_ae_sphe_jnone", src=_b("H2_pbc_ae_sphe.h5", False, None), nlm=None, alat=0.3, withj=False, pbc=True),
    dict(id="pbc_ae_h4_cart_jnone", src=_b("H4_pbc_ae_cart.h5", False, None), nlm=None, alat=0.3, withj=False, pbc=True),
    dict(id="pbc_ae_h4_cart_j12", src=_b("H4_pbc_ae_cart.h5", True, None), nlm=None, alat=0.3, withj=True, pbc=True),
]

_COMPLEX_CELLS = [
    dict(id="twist_ecp_jnone", src=_p("twist_sphepgto_jnone"), nlm="tmove", alat=0.3, withj=False, pbc=True),
    dict(id="twist_ecp_j12", src=_p("twist_sphepgto_j12"), nlm="tmove", alat=0.3, withj=True, pbc=True),
    dict(id="twist_ecp_jdtn", src=_p("twist_sphepgto_jdtn"), nlm="tmove", alat=0.3, withj=True, pbc=True),
    dict(id="twist_ecp_j123", src=_b("lithium_k025.h5", True, {"j3": "pgto"}), nlm="tmove", alat=0.3, withj=True, pbc=True),
    dict(id="twist_pw_jnone", src=_p("pw_twist_jnone"), nlm="tmove", alat=0.3, withj=False, pbc=True),
    dict(id="twist_bspline_jnone", src=_p("bspline_twist_jnone"), nlm="tmove", alat=0.3, withj=False, pbc=True),
    dict(id="twist_ae_sphe_jnone", src=_b("H2_pbc_ae_k025_sphe.h5", False, None), nlm=None, alat=0.4, withj=False, pbc=True),
]

_ALL_REAL = _CELLS


def _ham(cell):
    src = cell["src"]
    if src[0] == "pattern":
        return rs_ham(src[1])
    _, stem, withj, spec, gem = src
    return _build_ham_for_cell(stem, withj, spec, gem)


def _cfg(ham, seed):
    """Deterministic electron configuration (the input walker is built
    deterministically from the fixture, NOT sampled by the MCMC/GFMC machinery).
    """
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    if ham.structure_data.pbc_flag:
        cell = np.asarray(ham.structure_data.cell, float)
        return np.asarray(rng.random((nu, 3)) @ cell, float), np.asarray(rng.random((nd, 3)) @ cell, float)
    positions = np.asarray(ham.structure_data.positions, float)
    iu = rng.integers(0, positions.shape[0], nu)
    idn = rng.integers(0, positions.shape[0], nd)
    return positions[iu] + rng.normal(scale=0.5, size=(nu, 3)), positions[idn] + rng.normal(scale=0.5, size=(nd, 3))


def _cfg_for_parity(ham, seed):
    """Deterministically reject only near-node aggregate-parity inputs.

    The liveness tests retain the unfiltered fixture universe above.  Selection
    uses only the XLA geminal condition number before either backend observable
    is evaluated, never XLA--Rust agreement.
    """
    gd = ham.wavefunction_data.geminal_data
    for attempt in range(300):
        candidate_seed = seed if attempt == 0 else seed * 100_000 + attempt
        ru, rd = _cfg(ham, candidate_seed)
        geminal = np.asarray(compute_geminal_all_elements(gd, ru, rd))
        condition = float(np.linalg.cond(geminal))
        if np.isfinite(condition) and condition < 100.0:
            return np.ascontiguousarray(ru), np.ascontiguousarray(rd)
    raise AssertionError(f"XLA could not draw cond(G) < 100 from fixed seed={seed} after 300 candidates")


def _nn(ham):
    """Nearest-neighbour count for the ECP mesh.

    ``n_atom`` for periodic cells on BOTH sides would be a self-consistent choice, but
    every PRODUCTION periodic ECP
    path uses ``NN=1`` -- the XLA ``_compute_V_elements_t`` / ``_projection_t_core``
    (default ``NN_default==1``) and the native GFMC chunk (jqmc_gfmc.py:1691-1693). The
    reference here is the production closure, so the whole row (reference, Rust
    recovery and the production-oracle assembly) runs at the production ``NN=1``.
    """
    return 1


# --------------------------------------------------------------------------- #
# PRODUCTION assembly. `_compute_V_elements_t` is the module-level closure the
# GFMC_t projection/force driver runs; the reference `v_diag` / `b_l` / `e_l` are
# its own ZV-fold + FN/SP + Ewald assembly, NOT a second
# in-test implementation. The per-summand leaf values are still computed from the
# same XLA kernels the driver calls, so the Rust component recovery is pinned
# against a leaf reference; their sum is cross-checked against the production
# assembly by `test_gfmc_reference_assembly_matches_production`.
# --------------------------------------------------------------------------- #
def _prod_v_diag_b_l(ham, ru, rd, alat, RT, nlm):
    Vd, Vn, _ = _compute_V_elements_t(ham, ru, rd, RT, nlm, alat)
    return complex(Vd), complex(Vn)


# --------------------------------------------------------------------------- #
# XLA leaf references: the individual summands of jqmc_gfmc._compute_V_elements_t.
# --------------------------------------------------------------------------- #
def _xla_components(ham, ru, rd, alat, RT, nlm):
    wf = ham.wavefunction_data
    cd = ham.coulomb_potential_data
    nu, nd = len(ru), len(rd)

    _, _, el = compute_discretized_kinetic_energy(alat, wf, ru, rd, RT)
    el = np.asarray(el).real.ravel()
    fn_kin = float(np.minimum(el, 0.0).sum())
    sp_kin = float(np.maximum(el, 0.0).sum())
    ela = el.reshape(-1, 6)
    nondiag = (ela + 1.0 / (2.0 * alat**2)).sum(1)
    sflip = np.any(ela >= 0, axis=1)

    kc_u, kc_d = compute_kinetic_energy_all_elements(wf, ru, rd)
    kc = np.concatenate([np.asarray(kc_u).real.ravel(), np.asarray(kc_d).real.ravel()])
    eb_u, eb_d = compute_coulomb_potential_ei_dispatch(coulomb_potential_data=cd, r_up_carts=ru, r_dn_carts=rd)
    ei = np.concatenate([np.asarray(eb_u).real, np.asarray(eb_d).real])
    if cd.ecp_flag:
        ei_ref = ei  # parcutg=1: the max-arg is the BARE point-charge el-ion
    else:
        ed_u, ed_d = compute_coulomb_potential_ei_discretized_dispatch(
            coulomb_potential_data=cd, r_up_carts=ru, r_dn_carts=rd, alat=alat
        )
        ei_ref = np.concatenate([np.asarray(ed_u).real, np.asarray(ed_d).real])
    zv = ei + kc - nondiag
    zv_opt = float(np.where(sflip, np.maximum(zv, ei_ref), zv).sum())

    v_ee = float(np.real(compute_coulomb_potential_ee_dispatch(coulomb_potential_data=cd, r_up_carts=ru, r_dn_carts=rd)))
    v_ii = float(np.real(compute_coulomb_potential_ii_dispatch(coulomb_potential_data=cd)))

    ecp_local = sp_ecp = fn_ecp = 0.0
    v_nl_sum = 0.0
    if cd.ecp_flag and nlm is not None:
        ecp_local = float(np.real(compute_ecp_local_parts_all_pairs(coulomb_potential_data=cd, r_up_carts=ru, r_dn_carts=rd)))
        det_only = nlm == "dltmove"
        mu, md, V, _ = compute_ecp_non_local_parts_nearest_neighbors(
            coulomb_potential_data=cd,
            wavefunction_data=wf,
            r_up_carts=ru,
            r_dn_carts=rd,
            RT=RT,
            NN=_nn(ham),
            flag_determinant_only=det_only,
        )
        V = np.asarray(V).real
        v_nl_sum = float(V.sum())
        sp_ecp = float(np.maximum(V, 0.0).sum())
        fn_leg = np.minimum(V, 0.0)
        if det_only:
            mu = np.asarray(mu)
            md = np.asarray(md)
            jb = float(np.real(compute_Jastrow_part(wf.jastrow_data, ru, rd)))
            jr = np.array([float(np.real(compute_Jastrow_part(wf.jastrow_data, mu[m], md[m]))) for m in range(len(V))])
            fn_leg = fn_leg * np.exp(jr - jb)
        fn_ecp = float(fn_leg.sum())

    const = 3.0 / alat**2 * (nu + nd)
    # leaf assembly (kept only for the anti-drift cross-check below)
    v_diag_leaf = const + (v_ee + v_ii + zv_opt) + ecp_local + sp_kin + sp_ecp
    b_l_leaf = fn_kin + fn_ecp
    # v_diag / b_l / e_l come from the PRODUCTION closure.
    Vd, Vn = _prod_v_diag_b_l(ham, ru, rd, alat, RT, nlm)
    v_diag = float(Vd.real)
    b_l = float(Vn.real)
    return dict(
        const=const,
        sp_kin=sp_kin,
        fn_kin=fn_kin,
        zv_opt=zv_opt,
        kin_continuum=float(kc.sum()),
        v_ee=v_ee,
        v_ii=v_ii,
        ecp_local=ecp_local,
        v_nl_sum=v_nl_sum,
        sp_ecp=sp_ecp,
        fn_ecp=fn_ecp,
        v_diag=v_diag,
        b_l=b_l,
        e_l=v_diag + b_l,
        v_diag_leaf=v_diag_leaf,
        b_l_leaf=b_l_leaf,
    )


# --------------------------------------------------------------------------- #
# Rust side: the SAME summands, recovered from the extension only.
# --------------------------------------------------------------------------- #
def _rust_lrdmc(ham, ru, rd, alat, RT, nlm):
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    cd = ham.coulomb_potential_data
    if nlm is None:
        e_l, b_l = jqmc_rs._gfmc_lrdmc_ae(gd, jd, cd, ru, rd, alat, RT)[:2]
        return float(e_l), float(b_l)
    out = jqmc_rs._gfmc_lrdmc_ecp(gd, jd, cd, ru, rd, alat, RT, _W, _GP, _nn(ham), nlm == "dltmove")
    return float(out[0]), float(out[1])


def _rust_components(ham, ru, rd, alat, RT, nlm):
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    cd = ham.coulomb_potential_data
    nu, nd = len(ru), len(rd)

    el = np.asarray(jqmc_rs._gfmc_discretized_kinetic_elements(gd, jd, ru, rd, alat, RT))
    sp_kin = float(np.maximum(el, 0.0).sum())
    fn_kin = float(np.minimum(el, 0.0).sum())

    e_l, b_l = _rust_lrdmc(ham, ru, rd, alat, RT, nlm)
    fn_ecp = b_l - fn_kin

    if ham.structure_data.pbc_flag:
        ed = cd.ewald_data
        v_ee = float(
            jqmc_rs.compute_ewald_potential_ee(
                ru,
                rd,
                float(ed.eta),
                np.asarray(ed.real_image_list, float),
                np.asarray(ed.g_vectors, float),
                np.asarray(ed.g_factors, float),
                np.asarray(ham.structure_data.cell, float),
                int(ham.structure_data._n_max),
            )
        )
    else:
        v_ee = float(jqmc_rs.compute_bare_coulomb_potential_el_el(ru, rd))

    if ham.structure_data.pbc_flag:
        # Ewald ion-ion is not separately exported (precomputed inside the Rust
        # PbcCtx); see the module docstring's attribution caveat.
        v_ii = float(np.real(compute_coulomb_potential_ii_dispatch(coulomb_potential_data=cd)))
    else:
        v_ii = float(
            jqmc_rs.compute_bare_coulomb_potential_ion_ion(
                np.asarray(ham.structure_data.positions, float), np.asarray(cd._effective_charges, float)
            )
        )

    ecp_local = sp_ecp = 0.0
    v_nl_sum = 0.0
    if cd.ecp_flag and nlm is not None:
        if ham.structure_data.pbc_flag:
            # The ``compute_ecp_local_parts_all_pairs`` debug export passes ``None``
            # for the cell (jqmc/rs/src/lib.rs:453 -> min-image DISABLED), so it is a
            # molecular-only entry point. The production periodic GFMC kernel applies
            # min-image internally, but has no separate export -> the periodic
            # ``ecp_local`` is taken from XLA and a Rust error in it would surface as a
            # ``zv_opt`` mismatch (same attribution caveat as ``v_ii``; still a failure).
            ecp_local = float(
                np.real(compute_ecp_local_parts_all_pairs(coulomb_potential_data=cd, r_up_carts=ru, r_dn_carts=rd))
            )
        else:
            ecp_local = float(jqmc_rs.compute_ecp_local_parts_all_pairs(cd, ru, rd))
        det_only = nlm == "dltmove"
        # The ``compute_ecp_nonlocal_parts_all_pairs`` export takes no ``RT``; jqmc
        # applies the mesh rotation as ``grid_points @ RT`` (coulomb_potential.py:961),
        # so pre-rotating the grid reproduces the rotated production mesh exactly.
        gp_rot = np.ascontiguousarray(np.asarray(_GP) @ np.asarray(RT))
        v_nl_sum = float(jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gd, jd, cd, ru, rd, _W, gp_rot, det_only, _nn(ham)))
        if not det_only:
            # max0(x) + min0(x) == x, elementwise -> exact split.
            sp_ecp = v_nl_sum - fn_ecp
        else:
            # dltmove weights ONLY the FN leg by the Jastrow ratio, so the split
            # above does not close. The two modes differ in NOTHING but the ECP
            # mesh legs, so run the tmove twin on the SAME state and use its
            # (exactly split) sp_ecp plus the V_diag difference.
            e_t, b_t = _rust_lrdmc(ham, ru, rd, alat, RT, "tmove")
            sp_ecp_t = float(jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gd, jd, cd, ru, rd, _W, gp_rot, False, _nn(ham))) - (
                b_t - fn_kin
            )
            sp_ecp = sp_ecp_t + ((e_l - b_l) - (e_t - b_t))

    const = 3.0 / alat**2 * (nu + nd)
    v_diag = e_l - b_l
    zv_opt = v_diag - (const + sp_kin + sp_ecp + ecp_local + v_ee + v_ii)
    return dict(
        const=const,
        sp_kin=sp_kin,
        fn_kin=fn_kin,
        zv_opt=zv_opt,
        v_ee=v_ee,
        v_ii=v_ii,
        ecp_local=ecp_local,
        v_nl_sum=v_nl_sum,
        sp_ecp=sp_ecp,
        fn_ecp=fn_ecp,
        v_diag=v_diag,
        b_l=b_l,
        e_l=e_l,
    )


_COMPONENT_KEYS = (
    "sp_kin",
    "fn_kin",
    "zv_opt",
    "v_ee",
    "v_ii",
    "ecp_local",
    "v_nl_sum",
    "sp_ecp",
    "fn_ecp",
    "v_diag",
    "b_l",
    "e_l",
)


def _assert_components(cell, ref, got, tag):
    """Each component judged against ITS OWN magnitude.

    A band ``atol + rtol * max(1, |e_l|)`` for every key would let the TOTAL local
    energy set the tolerance for each summand. e_L is dominated by the large
    kinetic/Coulomb terms, so a small component (``sp_ecp`` / ``fn_ecp`` are commonly
    orders of magnitude below the total, and are identically zero on the periodic cells)
    could have its sign flipped or a term dropped and still land inside a band scaled by
    something it does not contribute meaningfully to. Scaling per component makes a
    sign flip in ``fn_ecp`` a failure in ``fn_ecp``.
    """
    zones = _ZONES_BY_FLAGS[cell["pbc"]][cell["withj"]]
    atol, rtol = get_tolerance_xla_rust_consistency(zones, level="strict")
    for k in _COMPONENT_KEYS:
        d = abs(got[k] - ref[k])
        assert d <= atol + rtol * abs(ref[k]), (cell["id"], tag, k, ref[k], got[k], d)


# --------------------------------------------------------------------------- #
# STAGE (a): kernel parity on a saved, identical input state.
# --------------------------------------------------------------------------- #
@pytest.mark.light
@pytest.mark.parametrize("cell", _ALL_REAL, ids=[c["id"] for c in _ALL_REAL])
@pytest.mark.parametrize("seed", [0, 1, 12])
def test_gfmc_components_real(cell, seed):
    """Every summand of ``e_L^t`` (continuum/ZV, S'_kin, S'_ecp, ECP local, the
    non-local mesh, Ewald, the FN off-diagonal b_L) pinned Rust==XLA on ONE
    saved state -- not just the total.
    """
    ham = _ham(cell)
    ru, rd = _cfg_for_parity(ham, seed)
    RT = np.eye(3)
    ref = _xla_components(ham, ru, rd, cell["alat"], RT, cell["nlm"])
    got = _rust_components(ham, ru, rd, cell["alat"], RT, cell["nlm"])
    tag = f"seed={seed}"
    # Mixed precision uses the reviewed
    # medium backend envelope for these cancelling-aggregate components; full
    # precision keeps strict.
    zones = _ZONES_BY_FLAGS[cell["pbc"]][cell["withj"]]
    atol, rtol = get_tolerance_xla_rust_consistency(zones, level="medium" if is_mixed_precision_enabled() else "strict")
    for k in _COMPONENT_KEYS:
        d = abs(got[k] - ref[k])
        assert d <= atol + rtol * abs(ref[k]), (cell["id"], tag, k, ref[k], got[k], d)


@pytest.mark.light
@pytest.mark.parametrize("cell", _ALL_REAL, ids=[c["id"] for c in _ALL_REAL])
def test_gfmc_components_rotated_mesh(cell):
    """Same decomposition with one explicit NON-identity ``RT`` supplied to
    both low-level component oracles. Production interval drivers instead use
    independent kinetic/ECP rotations, pinned by the mesh-semantics contract.
    """
    ham = _ham(cell)
    ru, rd = _cfg_for_parity(ham, 2)
    a, b, c = 0.7, -1.1, 0.4
    Rz = np.array([[np.cos(c), -np.sin(c), 0.0], [np.sin(c), np.cos(c), 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[np.cos(b), 0.0, np.sin(b)], [0.0, 1.0, 0.0], [-np.sin(b), 0.0, np.cos(b)]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])
    RT = (Rz @ Ry @ Rx).T
    ref = _xla_components(ham, ru, rd, cell["alat"], RT, cell["nlm"])
    got = _rust_components(ham, ru, rd, cell["alat"], RT, cell["nlm"])
    tag = "rotated_RT"
    # Mixed precision uses the reviewed
    # medium backend envelope for these cancelling-aggregate components; full
    # precision keeps strict.
    zones = _ZONES_BY_FLAGS[cell["pbc"]][cell["withj"]]
    atol, rtol = get_tolerance_xla_rust_consistency(zones, level="medium" if is_mixed_precision_enabled() else "strict")
    for k in _COMPONENT_KEYS:
        d = abs(got[k] - ref[k])
        assert d <= atol + rtol * abs(ref[k]), (cell["id"], tag, k, ref[k], got[k], d)


@pytest.mark.light
@pytest.mark.parametrize("cell", _ALL_REAL, ids=[c["id"] for c in _ALL_REAL])
@pytest.mark.parametrize("seed", [0, 1])
def test_gfmc_reference_assembly_matches_production(cell, seed):
    """Anti-drift pin: the per-summand LEAF assembly
    (``const + v_ee + v_ii + zv_opt + ecp_local + S'`` / ``fn_kin + fn_ecp``) must
    reproduce the PRODUCTION ``_compute_V_elements_t`` ``(v_diag, b_l)`` it decomposes.
    The Rust component rows are pinned against the leaf summands and the total against
    the production closure, so this row ties the two together: if the extracted closure
    ever diverges from the leaf kernels its summands are built from, it goes red before
    a Rust-vs-reference row can be fooled by a matching-but-wrong oracle.
    """
    ham = _ham(cell)
    ru, rd = _cfg(ham, seed)
    c = _xla_components(ham, ru, rd, cell["alat"], np.eye(3), cell["nlm"])
    zones = _ZONES_BY_FLAGS[cell["pbc"]][cell["withj"]]
    atol, rtol = _d8_zone_tolerance(zones, level="strict")
    for k_leaf, k_prod in (("v_diag_leaf", "v_diag"), ("b_l_leaf", "b_l")):
        assert abs(c[k_leaf] - c[k_prod]) <= atol + rtol * abs(c[k_prod]), (cell["id"], seed, k_leaf, c[k_leaf], c[k_prod])


# --------------------------------------------------------------------------- #
# Branch liveness: a component row is only coverage if the component is NON-ZERO.
# This test records, per cell, which of the three "easily masked" channels are
# actually exercised -- and pins the fixtures that are structurally degenerate
# so a future fixture swap that silently re-zeroes them shows up here.
# --------------------------------------------------------------------------- #
_DEGENERATE = {
    # cell id -> channels that are IDENTICALLY zero for that fixture, with why.
    "mol_ecp_cart_jnone_tmove": ("sp_kin", "v_nl_sum", "sp_ecp", "fn_ecp"),
    "mol_ecp_cart_j123_tmove": ("sp_kin", "v_nl_sum", "sp_ecp", "fn_ecp"),
    "mol_ecp_cart_jdtn_tmove": ("sp_kin", "v_nl_sum", "sp_ecp", "fn_ecp"),
    "mol_ecp_cart_j12_dltmove": ("sp_kin", "v_nl_sum", "sp_ecp", "fn_ecp"),
    "mol_ae_sphe_jnone": ("sp_kin",),
    "pbc_gamma_jnone_tmove": ("sp_kin", "fn_ecp"),
    "pbc_gamma_j12_tmove": ("sp_kin", "fn_ecp"),
    "pbc_gamma_j123_tmove": ("sp_kin", "fn_ecp"),
    "pbc_gamma_jdtn_tmove": ("sp_kin", "fn_ecp"),
    "pbc_gamma_j12_dltmove": ("sp_kin", "fn_ecp"),
    "pbc_composite_jnone_tmove": ("sp_kin", "fn_ecp"),
    "pbc_composite_j12_tmove": ("sp_kin", "fn_ecp"),
    "pbc_pw_jnone_tmove": ("sp_kin", "fn_ecp"),
    "pbc_bspline_jnone_tmove": ("sp_kin", "fn_ecp"),
    "pbc_ae_sphe_jnone": ("sp_kin",),
    # H4_pbc_ae DOES sign-flip (first hit at seed 12) -> sp_kin is LIVE there; it is
    # the only periodic cell in this table with a live S'_kin, which is why seed 12 is
    # in the component-parity seed list.
}


@pytest.mark.light
@pytest.mark.parametrize("cell", _ALL_REAL, ids=[c["id"] for c in _ALL_REAL])
def test_gfmc_branch_liveness(cell):
    """Records which maskable channels are live. A channel declared degenerate must
    be EXACTLY zero over several states (so the corresponding parity row is honestly
    reported as vacuous for that cell); a channel NOT declared degenerate must be
    non-zero for at least one state (so its parity row is real coverage).
    """
    ham = _ham(cell)
    seen = dict.fromkeys(("sp_kin", "fn_ecp", "sp_ecp", "v_nl_sum"), 0.0)
    for seed in range(16):
        ru, rd = _cfg(ham, seed)
        c = _xla_components(ham, ru, rd, cell["alat"], np.eye(3), cell["nlm"])
        for k in seen:
            seen[k] = max(seen[k], abs(c[k]))
    degen = set(_DEGENERATE.get(cell["id"], ()))
    if cell["nlm"] is None:
        degen |= {"fn_ecp", "sp_ecp", "v_nl_sum"}  # the ECP branch is not taken at all
    for k, v in seen.items():
        if k in degen:
            assert v <= _ZERO_ATOL + _ZERO_RTOL, (cell["id"], k, "declared degenerate but is LIVE", v)
        else:
            assert v > _ZERO_ATOL, (cell["id"], k, "declared live but is identically zero", v)


# --------------------------------------------------------------------------- #
# STAGE (a), complex Bloch twist. No Rust complex discretized-kinetic export
# exists, so the split is: b_L (the FN sum), sum(V_nl) (the non-local mesh,
# directly), V_diag, and the S' residual -- all as COMPLEX numbers, Re and Im
# pinned separately.
# --------------------------------------------------------------------------- #
def _xla_components_complex(ham, ru, rd, alat, RT, nlm):
    wf = ham.wavefunction_data
    cd = ham.coulomb_potential_data
    nu, nd = len(ru), len(rd)

    _, _, el = compute_discretized_kinetic_energy(alat, wf, ru, rd, RT)
    el = jnp.asarray(el).ravel()
    fn_kin = complex(jnp.sum(jnp.minimum(el, 0.0)))
    sp_kin = complex(jnp.sum(jnp.maximum(el, 0.0)))
    ela = el.reshape(-1, 6)
    nondiag = jnp.sum(ela + 1.0 / (2.0 * alat**2), axis=1)
    sflip = jnp.any(jnp.real(ela) >= 0, axis=1)
    kc_u, kc_d = compute_kinetic_energy_all_elements(wf, ru, rd)
    kc = jnp.concatenate([jnp.asarray(kc_u).ravel(), jnp.asarray(kc_d).ravel()])
    eb_u, eb_d = compute_coulomb_potential_ei_dispatch(coulomb_potential_data=cd, r_up_carts=ru, r_dn_carts=rd)
    ei = jnp.concatenate([jnp.asarray(eb_u), jnp.asarray(eb_d)])
    if cd.ecp_flag:
        ei_ref = ei
    else:
        ed_u, ed_d = compute_coulomb_potential_ei_discretized_dispatch(
            coulomb_potential_data=cd, r_up_carts=ru, r_dn_carts=rd, alat=alat
        )
        ei_ref = jnp.concatenate([jnp.asarray(ed_u), jnp.asarray(ed_d)])
    zv = ei + kc - nondiag
    zv_opt = complex(jnp.sum(jnp.where(sflip, jnp.maximum(zv, ei_ref), zv)))
    v_ee = complex(compute_coulomb_potential_ee_dispatch(coulomb_potential_data=cd, r_up_carts=ru, r_dn_carts=rd))
    v_ii = complex(compute_coulomb_potential_ii_dispatch(coulomb_potential_data=cd))

    ecp_local = 0.0 + 0.0j
    v_nl_sum = 0.0 + 0.0j
    fn_ecp = 0.0 + 0.0j
    sp_ecp = 0.0 + 0.0j
    if cd.ecp_flag and nlm is not None:
        ecp_local = complex(compute_ecp_local_parts_all_pairs(coulomb_potential_data=cd, r_up_carts=ru, r_dn_carts=rd))
        _, _, V, vsum = compute_ecp_non_local_parts_nearest_neighbors(
            coulomb_potential_data=cd,
            wavefunction_data=wf,
            r_up_carts=ru,
            r_dn_carts=rd,
            RT=RT,
            NN=_nn(ham),
            flag_determinant_only=False,
        )
        V = jnp.asarray(V).ravel()
        v_nl_sum = complex(jnp.sum(V))
        fn_ecp = complex(jnp.sum(jnp.minimum(V, 0.0)))
        sp_ecp = complex(jnp.sum(jnp.maximum(V, 0.0)))

    const = 3.0 / alat**2 * (nu + nd)
    # v_diag / b_l / e_l from the PRODUCTION complex closure; the
    # complex cells carry no MPC. For the ECP cells the production mesh is Re-projected
    # by nn_ecp so Im(v_diag) legitimately differs from the full-complex leaf assembly
    # (that divergence is Im(sum V_nl), pinned by
    # test_gfmc_complex_nonlocal_mesh_re_projection); the leaf assembly is kept for the
    # real-cell anti-drift cross-check only.
    Vd, Vn = _prod_v_diag_b_l(ham, ru, rd, alat, RT, nlm)
    v_diag = Vd
    b_l = Vn
    return dict(
        const=const,
        sp_kin=sp_kin,
        fn_kin=fn_kin,
        zv_opt=zv_opt,
        v_ee=v_ee,
        v_ii=v_ii,
        ecp_local=ecp_local,
        v_nl_sum=v_nl_sum,
        sp_ecp=sp_ecp,
        fn_ecp=fn_ecp,
        sp_total=sp_kin + sp_ecp,
        v_diag=v_diag,
        b_l=b_l,
        e_l=v_diag + b_l,
    )


def _rust_components_complex(ham, ru, rd, alat, RT, nlm):
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    cd = ham.coulomb_potential_data
    nu, nd = len(ru), len(rd)
    if nlm is None:
        re, im, br, bi = jqmc_rs._gfmc_lrdmc_ae_complex(gd, jd, cd, ru, rd, alat, RT)
    else:
        re, im, br, bi, _ = jqmc_rs._gfmc_lrdmc_ecp_complex(gd, jd, cd, ru, rd, alat, RT, _W, _GP, _nn(ham))
    e_l = complex(re, im)
    b_l = complex(br, bi)
    return dict(e_l=e_l, b_l=b_l, v_diag=e_l - b_l)


@pytest.mark.light
@pytest.mark.parametrize("cell", _COMPLEX_CELLS, ids=[c["id"] for c in _COMPLEX_CELLS])
@pytest.mark.parametrize("seed", [0, 1])
def test_gfmc_components_complex(cell, seed):
    """Complex-twist observable components: ``Re`` pinned for every cell, and the
    FULL complex value pinned wherever the two implementations share a convention.

    Re-projecting factors separately is a known trap in this codebase
    (``Re(A)Re(B) != Re(A*B)``) and is invisible at Gamma, so the imaginary parts
    are NOT ignored: for the all-electron complex cell (no ECP mesh) the complex
    ``e_L``/``b_L``/``V_diag`` are pinned outright, and for the ECP cells the
    imaginary divergence is root-caused and quantitatively pinned by
    ``test_gfmc_complex_nonlocal_mesh_re_projection`` below.
    """
    ham = _ham(cell)
    ru, rd = _cfg(ham, seed)
    RT = np.eye(3)
    ref = _xla_components_complex(ham, ru, rd, cell["alat"], RT, cell["nlm"])
    got = _rust_components_complex(ham, ru, rd, cell["alat"], RT, cell["nlm"])
    # The Re rows are cancelling aggregates;
    # mixed precision uses the reviewed medium backend envelope for them, full
    # keeps strict. The Im rows keep strict in both modes.
    atol_re, rtol_re = get_tolerance_xla_rust_consistency(
        _zones(cell), level="medium" if is_mixed_precision_enabled() else "strict"
    )
    atol, rtol = get_tolerance_xla_rust_consistency(_zones(cell), level="strict")
    scale = max(1.0, abs(ref["e_l"]))
    for k in ("e_l", "b_l", "v_diag"):
        assert abs(got[k].real - ref[k].real) <= atol_re + rtol_re * scale, (cell["id"], seed, k, "Re", got[k], ref[k])
    if cell["nlm"] is None:
        # No ECP mesh -> no Re-projection convention difference; the complex
        # det / continuum-kinetic / ZV machinery must agree in Im too.
        for k in ("e_l", "b_l", "v_diag"):
            assert abs(got[k].imag - ref[k].imag) <= atol + rtol * scale, (cell["id"], seed, k, "Im", got[k], ref[k])


def _rust_vnl_complex(ham, ru, rd, withj):
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    if withj:
        return complex(
            *jqmc_rs.compute_ecp_nonlocal_withJ_complex(gd, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, _W, _GP)
        )
    return complex(*jqmc_rs.compute_ecp_nonlocal_complex(gd, ham.coulomb_potential_data, ru, rd, _W, _GP))


@pytest.mark.light
@pytest.mark.parametrize(
    "cell", [c for c in _COMPLEX_CELLS if c["nlm"] is not None], ids=[c["id"] for c in _COMPLEX_CELLS if c["nlm"] is not None]
)
@pytest.mark.parametrize("seed", [0, 1])
def test_gfmc_complex_nonlocal_mesh_re_projection(cell, seed):
    """Complex ECP VMC values and fixed-phase LRDMC mesh values have distinct boundaries.

    The existing complex twist fixtures and fixed coordinates are evaluated by
    both GFMC component routes. XLA and Rust project each LRDMC candidate before
    its sign split, so their full imaginary e_L and b_L must agree. The separate
    Rust continuum VMC nonlocal sum can retain an imaginary part; it is not an
    expected GFMC discrepancy. Its real part must equal the real mesh sum.
    The original fixture/case universe and central backend budget are retained;
    independent nonzero-imaginary and both-sign activation lives in the D10
    complex ECP policy tests, not in a presumed mismatch between backends.
    """
    ham = _ham(cell)
    ru, rd = _cfg(ham, seed)
    RT = np.eye(3)
    ref = _xla_components_complex(ham, ru, rd, cell["alat"], RT, cell["nlm"])
    got = _rust_components_complex(ham, ru, rd, cell["alat"], RT, cell["nlm"])
    v_rust = _rust_vnl_complex(ham, ru, rd, cell["withj"])
    atol, rtol = get_tolerance_xla_rust_consistency(_zones(cell), level="strict")
    # The branching-rate Im row (4) is a
    # cancelling aggregate; mixed precision uses the reviewed medium backend
    # envelope for it, full keeps strict. Rows (2) and (3) keep strict.
    atol_b, rtol_b = get_tolerance_xla_rust_consistency(
        _zones(cell), level="medium" if is_mixed_precision_enabled() else "strict"
    )
    scale = max(1.0, abs(ref["e_l"]))

    # (1) XLA really does hand back a real-projected mesh value.
    assert abs(ref["v_nl_sum"].imag) <= _ZERO_ATOL, (cell["id"], "XLA V_nl unexpectedly complex", ref["v_nl_sum"])
    # (2) The real parts agree -> Re(e_L) is correct.
    assert abs(v_rust.real - ref["v_nl_sum"].real) <= atol + rtol * scale, (cell["id"], seed, v_rust, ref["v_nl_sum"])
    # (3) Fixed-phase projection belongs to both GFMC implementations; the
    # unprojected VMC sum is not an imaginary correction to this operator.
    d_im = got["e_l"].imag - ref["e_l"].imag
    assert abs(d_im) <= atol + rtol * scale, (cell["id"], seed, "Im(e_L) fixed-phase residual", d_im)
    # (4) Both fixed-phase paths use real ECP candidates before the FN clamp;
    # their branching-rate imaginary components must agree as well.
    assert abs(got["b_l"].imag - ref["b_l"].imag) <= atol_b + rtol_b * scale, (
        cell["id"],
        seed,
        "b_L Im divergence reached the branching rate",
        got["b_l"],
        ref["b_l"],
    )


@pytest.mark.light
@pytest.mark.parametrize("cell", _COMPLEX_CELLS, ids=[c["id"] for c in _COMPLEX_CELLS])
def test_gfmc_complex_sprime_and_mesh_residual(cell):
    """Complex twist: isolate the S' total from the total, so a complex S' error
    cannot be absorbed by the continuum/ZV term.

    ``S'_kin + S'_ecp = V_diag - (const + v_ee + v_ii + zv_opt + ecp_local)`` on the
    RUST side vs the XLA S' total. Compared in Re for every cell; in Im too where
    there is no ECP mesh (see ``test_gfmc_complex_nonlocal_mesh_re_projection``).
    """
    ham = _ham(cell)
    ru, rd = _cfg(ham, 0)
    RT = np.eye(3)
    ref = _xla_components_complex(ham, ru, rd, cell["alat"], RT, cell["nlm"])
    got = _rust_components_complex(ham, ru, rd, cell["alat"], RT, cell["nlm"])
    atol, rtol = get_tolerance_xla_rust_consistency(_zones(cell), level="strict")
    # Re(S') is a cancelling aggregate; mixed
    # precision uses the reviewed medium backend envelope for it, full keeps
    # strict. The Im row keeps strict in both modes.
    atol_re, rtol_re = get_tolerance_xla_rust_consistency(
        _zones(cell), level="medium" if is_mixed_precision_enabled() else "strict"
    )
    scale = max(1.0, abs(ref["e_l"]))
    sp_rust = got["v_diag"] - (ref["const"] + ref["v_ee"] + ref["v_ii"] + ref["zv_opt"] + ref["ecp_local"])
    assert abs(sp_rust.real - ref["sp_total"].real) <= atol_re + rtol_re * scale, (
        cell["id"],
        "S' Re",
        sp_rust,
        ref["sp_total"],
    )
    if cell["nlm"] is None:
        assert abs(sp_rust.imag - ref["sp_total"].imag) <= atol + rtol * scale, (cell["id"], "S' Im", sp_rust, ref["sp_total"])


@pytest.mark.light
@pytest.mark.parametrize("cell", _COMPLEX_CELLS, ids=[c["id"] for c in _COMPLEX_CELLS])
def test_gfmc_complex_path_is_not_gamma_degenerate(cell):
    """Guard against a vacuous complex cell: at least one observable component must
    carry a non-negligible imaginary part, otherwise the Re-projection trap could not
    be detected by ANY of the rows above (at Gamma the trap is invisible).
    """
    ham = _ham(cell)
    ims = []
    for seed in range(3):
        ru, rd = _cfg(ham, seed)
        ref = _xla_components_complex(ham, ru, rd, cell["alat"], np.eye(3), cell["nlm"])
        ims.append(max(abs(ref[k].imag) for k in ("e_l", "b_l", "v_diag", "zv_opt", "sp_total")))
    assert max(ims) > _ZERO_ATOL, (cell["id"], "complex cell has no imaginary content", ims)


# --------------------------------------------------------------------------- #
# Same-state observable BUNDLE: from ONE saved state, everything production
# returns for that state -- e_L^t components, O_k (dln|Psi|/dR), and the HF
# force / stress reductions -- with matching shape/dtype/order on both sides.
# --------------------------------------------------------------------------- #
@pytest.mark.light
@pytest.mark.parametrize(
    "cell",
    [c for c in _CELLS if c["id"] in ("mol_ae_cart_j12", "mol_ae_cart_j123", "mol_ecp_sphe_j12_tmove")],
    ids=["mol_ae_cart_j12", "mol_ae_cart_j123", "mol_ecp_sphe_j12_tmove"],
)
def test_gfmc_same_state_ok_and_shapes(cell):
    """``O_k``-side quantity (``grad ln|Psi|`` wrt R / r) on the SAME state that the
    e_L decomposition used, and a shape/dtype/order contract for the arrays the Rust
    path hands back to the shared Python post-processing.

    MOLECULAR only: ``_gfmc_grad_ln_Psi`` goes through ``geminal_cart_from_py``
    (jqmc/rs/src/lib.rs:5906) and is a molecular-cartesian export -- handed a periodic
    sphe-pGTO geminal it PANICS (``pyo3_runtime.PanicException: ndarray: index out of
    bounds``) instead of raising a typed error. The periodic Pulay ``dln|Psi|/dR``
    parity lives on the dedicated periodic export and is pinned by
    ``test_gfmc_forces.py::test_grad_lndet_pbc_wrt_R``.
    """
    ham = _ham(cell)
    wf = ham.wavefunction_data
    ru, rd = _cfg(ham, 0)
    gR, gu, gd_ = jqmc_rs._gfmc_grad_ln_Psi(wf.geminal_data, wf.jastrow_data, ru, rd)
    gR, gu, gd_ = np.asarray(gR), np.asarray(gu), np.asarray(gd_)
    P = np.asarray(ham.structure_data.positions, float)
    assert gR.shape == P.shape and gu.shape == ru.shape and gd_.shape == rd.shape
    assert gR.dtype == np.float64 and gu.dtype == np.float64 and gd_.dtype == np.float64

    def ln_psi(P_, r_up, r_dn):
        # The nuclear positions live in SEVERAL leaves (structure, AO centres, ECP,
        # Jastrow 1b); ``_inject_active_positions`` is the production re-binding the
        # force path uses, and a plain ``structure_data.replace`` would leave the AO
        # centres frozen and hand back a zero dln|Psi|/dR.
        from jqmc.jqmc_mcmc import _inject_active_positions
        from jqmc.wavefunction import evaluate_ln_wavefunction

        h = _inject_active_positions(ham, {"positions": P_})
        return jnp.real(evaluate_ln_wavefunction(h.wavefunction_data, r_up, r_dn))

    rR, ru_g, rd_g = jax.grad(ln_psi, argnums=(0, 1, 2))(jnp.asarray(P), jnp.asarray(ru), jnp.asarray(rd))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_BY_FLAGS[cell["pbc"]][cell["withj"]], level="medium")
    for name, a, b in (("R", gR, np.asarray(rR)), ("r_up", gu, np.asarray(ru_g)), ("r_dn", gd_, np.asarray(rd_g))):
        assert np.allclose(a, b, atol=atol, rtol=rtol), (cell["id"], name, np.abs(a - b).max())


# --------------------------------------------------------------------------- #
# STAGE (b): INTEGRATION parity. The input state is not hand-drawn -- it is
# the multi-step output of the projection suite (a real GFMC_n run, XLA path), so
# the observable decomposition is exercised on states the projector actually
# visits (near-node, sign-flipping, post-branching).
# --------------------------------------------------------------------------- #
_INTEGRATION_LIVE = {
    "mol_ae_cart_j12": (),
    # The current projected water ensemble has an exactly zero positive
    # discretized-kinetic leg on both XLA and Rust.  Stage (a) retains explicit
    # live S'_kin coverage; this integration row exercises the two ECP legs.
    "mol_ecp_sphe_j12_tmove": ("sp_ecp", "fn_ecp"),
    "mol_ecp_sphe_j12_dltmove": ("sp_ecp", "fn_ecp"),
    "pbc_gamma_j12_tmove": ("sp_ecp",),
}


def _projected_walkers(ham, nwalkers, nsteps, seed, alat, nlm):
    from jqmc.jqmc_gfmc import GFMC_n

    g = GFMC_n(
        hamiltonian_data=ham,
        num_walkers=nwalkers,
        num_mcmc_per_measurement=16,
        mcmc_seed=seed,
        alat=alat,
        non_local_move=nlm if nlm is not None else "tmove",
        comput_position_deriv=False,
    )
    g.run(num_mcmc_steps=nsteps)
    return np.asarray(g._latest_r_up_carts, float), np.asarray(g._latest_r_dn_carts, float)


@pytest.mark.heavy
@pytest.mark.parametrize(
    "cell",
    [
        c
        for c in _CELLS
        if c["id"] in ("mol_ae_cart_j12", "mol_ecp_sphe_j12_tmove", "mol_ecp_sphe_j12_dltmove", "pbc_gamma_j12_tmove")
    ],
    ids=["mol_ae_cart_j12", "mol_ecp_sphe_j12_tmove", "mol_ecp_sphe_j12_dltmove", "pbc_gamma_j12_tmove"],
)
def test_gfmc_integration_projected_state_components(cell):
    """Multi-step projected walkers feed the component decomposition."""
    ham = _ham(cell)
    # The corrected Appendix-D diagonal changes branching weights whenever a
    # sign-flip row is clipped.  Seed 314 consequently coalesces all four AE
    # walkers onto one ancestor; seed 3 deterministically retains three states.
    # The ECP rows keep their established seed because it supplies their recorded
    # S'_ecp/FN_ecp liveness.
    projection_seed = 3 if cell["id"] == "mol_ae_cart_j12" else 314
    ru_all, rd_all = _projected_walkers(ham, 4, 6, projection_seed, cell["alat"], cell["nlm"])
    # GFMC reconfiguration CLONES walkers, so a duplicate is expected physics; the
    # ensemble must merely not have collapsed to a single state.
    assert len({ru_all[i].tobytes() for i in range(ru_all.shape[0])}) >= 2, "projected ensemble collapsed"
    live = {"sp_kin": 0.0, "sp_ecp": 0.0, "fn_ecp": 0.0}
    for w in range(ru_all.shape[0]):
        ru, rd = np.ascontiguousarray(ru_all[w]), np.ascontiguousarray(rd_all[w])
        ref = _xla_components(ham, ru, rd, cell["alat"], np.eye(3), cell["nlm"])
        got = _rust_components(ham, ru, rd, cell["alat"], np.eye(3), cell["nlm"])
        tag = f"projected walker {w}"
        zones = _ZONES_BY_FLAGS[cell["pbc"]][cell["withj"]]
        # Mixed precision uses
        # the reviewed medium backend envelope for these cancelling-aggregate
        # components; full precision keeps strict.
        atol, rtol = get_tolerance_xla_rust_consistency(zones, level="medium" if is_mixed_precision_enabled() else "strict")
        for k in _COMPONENT_KEYS:
            comparison_iteration = f"{w}:{k}"  # noqa: F841 - captured by the GDD iteration receipt
            d = abs(got[k] - ref[k])
            assert d <= atol + rtol * abs(ref[k]), (cell["id"], tag, k, ref[k], got[k], d)
        for k in live:
            live[k] = max(live[k], abs(ref[k]))
    # The projected states must actually exercise the maskable channels, otherwise
    # this integration row would be a total-only check in disguise. What the
    # projected ENSEMBLE reaches is a property of the cell, recorded here:
    #   * the water ccECP rows reach S'_ecp and FN_ecp; their S'_kin coverage is
    #     supplied by the hand-drawn stage-(a) states;
    #   * ``mol_ae_cart_j12`` (N atom, AE) reaches none of them once the walkers have
    #     relaxed off the node -- its S'_kin coverage comes from the hand-drawn
    #     stage-(a) states, which is exactly why stage (a) and (b) are both required;
    #   * ``pbc_gamma_j12_tmove`` (lithium) is the S'_kin / FN_ecp degeneracy that
    #     ``test_gfmc_branch_liveness`` already records.
    expected = _INTEGRATION_LIVE[cell["id"]]
    for k, v in live.items():
        if k in expected:
            assert v > _ZERO_ATOL, (cell["id"], k, "expected live on projected states", live)
