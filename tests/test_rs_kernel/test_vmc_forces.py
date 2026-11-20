"""Layer-1 equivalence: VMC (MCMC) HF-force core, native Rust vs JAX oracle.

The MCMC HF force = dE_L/dR + Omega.dE_L/dr uses the VMC CONTINUOUS local-energy
gradient (``jqmc.derivatives.assemble.compute_grad_e_L`` wrt R / r), NOT the LRDMC
discretized ``_gfmc_grad_e_L_*`` (which adds the alat S' terms -- see
``gfmc_forces::vmc_diag_grads`` module doc). ``vmc_grad_e_L_ae`` /
``vmc_grad_e_L_ecp`` reuse the sweep SM ``geminal_inv``,
so this pins both the native==JAX equivalence and the geminal-inverse-reuse
plumbing (fresh vs with-inv agreement + near-singular no-NaN).
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

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import jqmc_rs  # noqa: E402

from jqmc._precision import get_tolerance_xla_rust_consistency, is_mixed_precision_enabled  # noqa: E402
from jqmc._setting import (  # noqa: E402
    ATOL_FLOOR_MEDIUM,
    RTOL_FLOOR_MEDIUM,
)
from jqmc.derivatives import assemble  # noqa: E402
from jqmc.jqmc_mcmc import _geminal_inv_batched, _NV_BASE_MESH, Nv_default  # noqa: E402
from tests.test_rs_kernel._rs_fixtures import build_twist_mo_j3_ham, rs_ham  # noqa: E402

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap")
_ATOL, _RTOL = get_tolerance_xla_rust_consistency(_ZONES, level="medium")

_MESH = _NV_BASE_MESH[Nv_default]
_W = np.asarray(_MESH.weights, float)
_GP = np.asarray(_MESH.grid_points, float)


def _cfg(ham, seed, scale=0.6):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[rng.integers(0, len(R), nu)] + rng.normal(scale=scale, size=(nu, 3))
    rd = R[rng.integers(0, len(R), nd)] + rng.normal(scale=scale, size=(nd, 3))
    return ru, rd


def _geminv(ham, ru, rd):
    _, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    # The native aggregate stores the running inverse in its fp64 accumulator
    # regardless of the AO forward-zone dtype.  Mirror that production boundary:
    # mixed XLA returns this helper value as float32, which PyO3 must not be asked
    # to reinterpret as an f64 matrix.
    return np.asarray(ginv[0], dtype=np.float64)


def _jax_ref(ham, ru, rd, ginv):
    RT = np.eye(3)
    dR = np.asarray(
        assemble.compute_grad_e_L(ham, jnp.asarray(ru), jnp.asarray(rd), RT, wrt="R", geminal_inv=jnp.asarray(ginv))
    )
    dru, drd = assemble.compute_grad_e_L(ham, jnp.asarray(ru), jnp.asarray(rd), RT, wrt="r", geminal_inv=jnp.asarray(ginv))
    return dR, np.asarray(dru), np.asarray(drd)


def _native(ham, ru, rd, ginv, ecp):
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cp = ham.coulomb_potential_data
    if ecp:
        return jqmc_rs.vmc_grad_e_L_ecp(gd, jd, cp, ru, rd, _W, _GP, ginv)
    return jqmc_rs.vmc_grad_e_L_ae(gd, jd, cp, ru, rd, ginv)


# --------------------------------------------------------------------------- #
# native == JAX de_L/d(R, r_up, r_dn) -- molecular AE (cart/sphe, SD/AGP, J-ladder)
# --------------------------------------------------------------------------- #
_AE_PATTERNS = [
    "mol_cart_sd_jnone",
    "mol_cart_sd_j12",
    "mol_cart_sd_j123",
    "mol_cart_sd_jdtn",
    "mol_sphe_sd_jnone",
    "mol_sphe_agp_jnone",
    "mol_cart_agp_jnone",
    # PRODUCTION-DEFAULT Jastrow form. ``forces.rs`` carries
    # its OWN ``is_exp`` J1 branches (the dJ1/dR term), separate from the
    # value/grad kernels, so these are genuinely new code rows -- and the J1 x J2
    # cross-product rows pin that the two channels are read independently.
    "mol_cart_sd_j12_pade",
    "mol_cart_sd_j12_p1e2",
    "mol_cart_sd_j12_e1p2",
    "mol_cart_sd_j123_pade",
]


@pytest.mark.parametrize("pattern", _AE_PATTERNS)
@pytest.mark.parametrize("seed", [0])
def test_vmc_grad_e_l_ae(pattern, seed):
    """Vmc grad e l ae."""
    ham = rs_ham(pattern)
    ru, rd = _cfg(ham, seed)
    ginv = _geminv(ham, ru, rd)
    dR_j, dru_j, drd_j = _jax_ref(ham, ru, rd, ginv)
    nR, nu, nd = _native(ham, ru, rd, ginv, ecp=False)
    np.testing.assert_allclose(np.asarray(nR), dR_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{pattern} dR")
    np.testing.assert_allclose(np.asarray(nu), dru_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{pattern} dr_up")
    np.testing.assert_allclose(np.asarray(nd), drd_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{pattern} dr_dn")


# molecular ECP CART + SPHE (local + non-local force). The sphe-pGTO ECP NON-LOCAL
# force (water ccECP ccpvqz, 3 close atoms) exercises two things the CART H2 fixture
# does not: (1) per-electron nearest-nucleus (NN=1) truncation -- the production e_L is
# nearest-neighbour, so an all-atoms mesh over-counts close neighbours (invisible for
# far-apart H2, ~1e-5 for water at jnone); (2) the FULL det*Jastrow ratio + ln|Psi|
# Pulay/drift in the mesh force (a det-only "jnone" mesh force is off by ~0.5 for water
# j12; H2's non-local force is ~0, so cart alone cannot see it). The VMC non-local
# force shares the GFMC/LRDMC ``ecp_nonlocal_force_mol`` kernel (NN=1, continuum,
# base O_R reuses ``geminal_inv``, mesh fresh).
_ECP_PATTERNS = [
    "mol_ecp_cart_jnone",
    "mol_ecp_cart_j12",
    "mol_ecp_cart_j123",
    "mol_ecp_cart_jdtn",
    "mol_ecp_sphe_jnone",
    "mol_ecp_sphe_j12",
    # PRODUCTION-DEFAULT Jastrow form: the ECP non-local mesh
    # force re-evaluates the FULL det*Jastrow ratio at every mesh point, so the
    # pade branch enters the quadrature as well as the drift.
    "mol_ecp_cart_j12_pade",
    "mol_ecp_sphe_j12_pade",
]


@pytest.mark.parametrize("pattern", _ECP_PATTERNS)
@pytest.mark.parametrize("seed", [0])
def test_vmc_grad_e_l_ecp(pattern, seed):
    """Vmc grad e l ECP."""
    ham = rs_ham(pattern)
    ru, rd = _cfg(ham, seed)
    ginv = _geminv(ham, ru, rd)
    dR_j, dru_j, drd_j = _jax_ref(ham, ru, rd, ginv)
    nR, nu, nd = _native(ham, ru, rd, ginv, ecp=True)
    np.testing.assert_allclose(np.asarray(nR), dR_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{pattern} dR")
    np.testing.assert_allclose(np.asarray(nu), dru_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{pattern} dr_up")
    np.testing.assert_allclose(np.asarray(nd), drd_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{pattern} dr_dn")


@pytest.mark.parametrize("stem", ["N2_ecp_ccpvtz_cart.h5", "H2_ecp_ccpvdz_cart.h5"])
@pytest.mark.parametrize("seed", [0])
def test_vmc_grad_e_l_ecp_fast_mesh_screen(stem, seed):
    """FAST-screened ECP mesh force agrees with the exact-mesh JAX derivative.

    The FAST MCMC chunk e_L evaluates the ECP non-local mesh AO contractions
    (determinant and J3) with the FAST value screen, drops ECP channels below
    the FAST radial cut, and contracts kinetic AO gradients / Laplacians on the
    FAST derivative screen. Its force (``fast_mesh_screen=True``) takes every
    one of these on the same keep sets. Committed Cartesian ECP SD fixtures with
    J1+J2 and a symmetric random Cartesian J3 (seed 7); electrons from
    ``_cfg`` (PCG64 seed 0) around the nuclei, unrotated default mesh. The
    reference is the exact (unscreened) XLA ``compute_grad_e_L``, an
    independent implementation: the dropped terms are below the FAST screen
    bounds, so the screened force must still agree at the existing medium
    XLA/Rust tolerance of the native-vs-XLA force parity tests. The
    kernel-level checks that the screened tables are those of the FAST e_L
    kernels (and their derivatives) are the Rust ``mesh_screen_tests``.
    """
    from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
    from tests.test_rs_kernel.test_gfmc_forces import _rand_j3

    ham = _build_ham_for_cell(stem, True, {"j3": "gto"}, "SD")
    ham = ham.replace(wavefunction_data=_rand_j3(ham.wavefunction_data))
    ru, rd = _cfg(ham, seed)
    ginv = _geminv(ham, ru, rd)
    dR_j, dru_j, drd_j = _jax_ref(ham, ru, rd, ginv)
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cp = ham.coulomb_potential_data
    nR, nu, nd = jqmc_rs.vmc_grad_e_L_ecp(gd, jd, cp, ru, rd, _W, _GP, ginv, fast_mesh_screen=True)
    np.testing.assert_allclose(np.asarray(nR), dR_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{stem} dR")
    np.testing.assert_allclose(np.asarray(nu), dru_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{stem} dr_up")
    np.testing.assert_allclose(np.asarray(nd), drd_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{stem} dr_dn")


# --------------------------------------------------------------------------- #
# Geminal-inverse reuse: the observable HF-force core NEVER does its own fresh
# LU -- it consumes the sweep-emitted SM inverse. At a
# near-node config where two same-spin electrons nearly coincide (geminal cond
# ~1e12), the reused-inverse force stays finite AND equals the JAX oracle (which
# threads the same geminal_inv) at strict. This is the stability contract behind
# the reuse: a fresh LU is the NaN risk (measured on the periodic near-nucleus
# regime); molecular fp64 keeps a high-condition LU finite too, so
# here we pin equivalence-to-JAX-at-high-condition + finiteness rather than a
# manufactured molecular NaN.
# --------------------------------------------------------------------------- #
# ``collapse``: same-spin separation. 1e-4 in full / 2e-3 in mixed gives a
# moderately conditioned near-node case where native and JAX agree while
# threading the same reused inverse.  Mixed does not manufacture bit agreement
# at the deeper fp32-conditioned point by changing the production reduction
# order; it uses a representable near-node contract instead.
# 1e-8 -> cond ~1e12: the reused-inverse force stays FINITE (the stability point;
# a fresh LU is the NaN risk), but native vs JAX legitimately diverge by fp64
# roundoff at that conditioning, so only finiteness is asserted there.
# The 1e-8 point is STABILITY evidence (finite, no fresh-LU blow-up), never an
# accuracy claim; the moderate point keeps the native == JAX comparison.
_NEAR_NODE_STABILITY_REASON = (
    "At the 1e-8 collapse (cond ~1e12) native and JAX legitimately diverge by roundoff; the case is "
    "stability evidence only and asserts that every native output of the reused inverse is finite"
)
_NEAR_NODE_STABILITY_CASE = pytest.param(
    1.0e-8, False, marks=pytest.mark.gdd_finiteness_evidence(mode="stability", reason=_NEAR_NODE_STABILITY_REASON)
)
_HF_NEAR_NODE_CASES = [(2.0e-3 if is_mixed_precision_enabled() else 1.0e-4, True), _NEAR_NODE_STABILITY_CASE]


@pytest.mark.parametrize("collapse,check_equal", _HF_NEAR_NODE_CASES)
def test_vmc_force_reused_inverse_near_node(collapse, check_equal):
    """Vmc force reused inverse near node."""
    ham = rs_ham("mol_cart_sd_j12")  # N_ae: n_up=5 -> >=2 same-spin electrons
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cp = ham.coulomb_potential_data
    assert int(gd.num_electron_up) >= 2
    ru, rd = _cfg(ham, seed=1)
    ru[1] = ru[0] + collapse  # near-coincident same-spin -> near-singular geminal
    ginv = _geminv(ham, ru, rd)
    assert np.all(np.isfinite(ginv)), "precondition: the reused SM inverse is finite"
    nR, nu, nd = jqmc_rs.vmc_grad_e_L_ae(gd, jd, cp, ru, rd, ginv)
    # finite (the reused inverse never triggers a fresh-LU blow-up) ...
    assert np.all(np.isfinite(np.asarray(nR)))
    assert np.all(np.isfinite(np.asarray(nu))) and np.all(np.isfinite(np.asarray(nd)))
    if check_equal:
        # ... and correct: equals the JAX oracle threading the same reused inverse.
        dR_j, dru_j, drd_j = _jax_ref(ham, ru, rd, ginv)
        np.testing.assert_allclose(np.asarray(nR), dR_j, atol=_ATOL, rtol=_RTOL)
        np.testing.assert_allclose(np.asarray(nu), dru_j, atol=_ATOL, rtol=_RTOL)
        np.testing.assert_allclose(np.asarray(nd), drd_j, atol=_ATOL, rtol=_RTOL)


# --------------------------------------------------------------------------- #
# VMC Pulay WF-gradient core: native ``vmc_grad_ln_psi`` == JAX ``_grad_ln_psi_rR``
# (= ``compute_grad_ln_Psi`` wrt R / r). ln|Psi| is LRDMC-independent, so the SAME
# ``gfmc_grad_ln_psi`` building block serves the VMC Pulay force -- NO S' discretization
# aggregate (verified bit-exact vs the JAX oracle, unlike the HF e_L force). The native
# core reuses the sweep SM ``geminal_inv`` for the determinant Pulay + drift (the Jastrow
# J1/J2/J3/DTN pieces are inverse-free). The Pulay force is a pure WF derivative with no
# ECP/mesh dependence, so AE and ECP share this one call; the ECP fixtures still pin that
# the ECP ham's geminal/jastrow flow the same path. The J3 rows (MO-wrapped production J3)
# pin that the AO-basis ``grad_j3_wrt_r``/``jastrow_grads`` kernels handle an MO-wrapped
# J3 with no ``as_raw_ao`` lowering (the HF ``j3_side`` path DID need it; the Pulay path
# does not).
# --------------------------------------------------------------------------- #
def _jax_ref_pulay(ham, ru, rd, ginv):
    from jqmc.derivatives import assemble as _assemble

    dR = np.asarray(
        _assemble.compute_grad_ln_Psi(ham, jnp.asarray(ru), jnp.asarray(rd), wrt="R", geminal_inv=jnp.asarray(ginv))
    )
    dru, drd = _assemble.compute_grad_ln_Psi(ham, jnp.asarray(ru), jnp.asarray(rd), wrt="r", geminal_inv=jnp.asarray(ginv))
    return dR, np.asarray(dru), np.asarray(drd)


@pytest.mark.parametrize("pattern", _AE_PATTERNS + _ECP_PATTERNS)
@pytest.mark.parametrize("seed", [0])
def test_vmc_grad_ln_psi(pattern, seed):
    """Vmc grad ln psi."""
    ham = rs_ham(pattern)
    ru, rd = _cfg(ham, seed)
    ginv = _geminv(ham, ru, rd)
    dR_j, dru_j, drd_j = _jax_ref_pulay(ham, ru, rd, ginv)
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    nR, nu, nd = jqmc_rs.vmc_grad_ln_psi(gd, jd, ru, rd, ginv)
    np.testing.assert_allclose(np.asarray(nR), dR_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{pattern} dln|Psi|/dR")
    np.testing.assert_allclose(np.asarray(nu), dru_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{pattern} dln|Psi|/dr_up")
    np.testing.assert_allclose(np.asarray(nd), drd_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{pattern} dln|Psi|/dr_dn")


# Geminal-inverse reuse for the Pulay force: same stability contract as the HF force --
# the observable core consumes the sweep-emitted SM inverse, never a fresh LU. At a
# near-node config the reused-inverse Pulay force stays FINITE (a fresh LU is the NaN
# risk) and, at moderate conditioning, equals the JAX oracle threading the same inverse.
@pytest.mark.parametrize("collapse,check_equal", [(1.0e-4, True), _NEAR_NODE_STABILITY_CASE])
def test_vmc_pulay_reused_inverse_near_node(collapse, check_equal):
    """Vmc pulay reused inverse near node."""
    ham = rs_ham("mol_cart_sd_j12")
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    assert int(gd.num_electron_up) >= 2
    ru, rd = _cfg(ham, seed=1)
    ru[1] = ru[0] + collapse
    ginv = _geminv(ham, ru, rd)
    assert np.all(np.isfinite(ginv)), "precondition: the reused SM inverse is finite"
    nR, nu, nd = jqmc_rs.vmc_grad_ln_psi(gd, jd, ru, rd, ginv)
    assert np.all(np.isfinite(np.asarray(nR)))
    assert np.all(np.isfinite(np.asarray(nu))) and np.all(np.isfinite(np.asarray(nd)))
    if check_equal:
        dR_j, dru_j, drd_j = _jax_ref_pulay(ham, ru, rd, ginv)
        np.testing.assert_allclose(np.asarray(nR), dR_j, atol=_ATOL, rtol=_RTOL)
        np.testing.assert_allclose(np.asarray(nu), dru_j, atol=_ATOL, rtol=_RTOL)
        np.testing.assert_allclose(np.asarray(nd), drd_j, atol=_ATOL, rtol=_RTOL)


# --------------------------------------------------------------------------- #
# MO-wrapped (production ``orb_data=mos_data``) J3: the AO-basis Pulay dR kernel
# ``grad_j3_wrt_r`` mixes MO-value coeffs (``compute``) with raw-AO gradients
# (``ao_grad``) indexed by MO index, so a ``num_mo != num_ao`` J3 silently corrupts
# ``dln|Psi|/dR`` (the drift path was already MO-correct). The committed ``rs_ham``
# J3 fixtures are raw-AO (cart_gto), so this row builds the production MO-wrapped J3
# explicitly (as the heavy MCMC driver does) and pins that the native VMC Pulay force
# lowers it to its AO-equivalent (``lower_j3_to_ao``) -- matching the JAX oracle. Also
# checks the HF force (``vmc_grad_e_L_ae``) on the same MO-wrapped J3.
# --------------------------------------------------------------------------- #
def _build_mo_j3_ham(stem):
    import os

    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import (
        Jastrow_data,
        Jastrow_one_body_data,
        Jastrow_three_body_data,
        Jastrow_two_body_data,
    )
    from jqmc.trexio_wrapper import read_trexio_file
    from jqmc.wavefunction import Wavefunction_data

    trexio_dir = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files")
    sd, _aos, mos, _mos_no, gem_mo, cp = read_trexio_file(trexio_file=os.path.join(trexio_dir, stem), store_tuple=True)
    n_atom = len(sd.atomic_numbers)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0, structure_data=sd, core_electrons=core_electrons_of(cp, n_atom), jastrow_1b_type="exp"
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp")
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=mos, random_init=True, random_scale=1.0e-3, seed=123)
    jas = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=j3)
    ham = Hamiltonian_data(
        structure_data=sd,
        coulomb_potential_data=cp,
        wavefunction_data=Wavefunction_data(jastrow_data=jas, geminal_data=gem_mo),
    )
    ham.sanity_check()
    return ham


@pytest.mark.parametrize("stem", ["H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
def test_vmc_pulay_and_hf_mo_wrapped_j3(stem):
    """Vmc pulay and HF MO wrapped j3."""
    ham = _build_mo_j3_ham(stem)
    # the production J3 is genuinely MO-wrapped (num_mo != num_ao) -- else this row is vacuous.
    j3o = ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data
    assert getattr(j3o, "mo_coefficients", None) is not None
    ru, rd = _cfg(ham, seed=0)
    ginv = _geminv(ham, ru, rd)
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cp = ham.coulomb_potential_data
    # Pulay: native vs JAX ``_grad_ln_psi_rR`` (dR is the corrupted axis for MO-J3).
    dR_j, dru_j, drd_j = _jax_ref_pulay(ham, ru, rd, ginv)
    nR, nu, nd = jqmc_rs.vmc_grad_ln_psi(gd, jd, ru, rd, ginv)
    np.testing.assert_allclose(np.asarray(nR), dR_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{stem} MO-J3 dln|Psi|/dR")
    np.testing.assert_allclose(np.asarray(nu), dru_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{stem} MO-J3 dln|Psi|/dr_up")
    np.testing.assert_allclose(np.asarray(nd), drd_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{stem} MO-J3 dln|Psi|/dr_dn")
    # HF force (vmc_diag_grads) on the same MO-wrapped J3.
    dRh_j, druh_j, drdh_j = _jax_ref(ham, ru, rd, ginv)
    nRh, nuh, ndh = jqmc_rs.vmc_grad_e_L_ae(gd, jd, cp, ru, rd, ginv)
    np.testing.assert_allclose(np.asarray(nRh), dRh_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{stem} MO-J3 de_L/dR")
    np.testing.assert_allclose(np.asarray(nuh), druh_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{stem} MO-J3 de_L/dr_up")
    np.testing.assert_allclose(np.asarray(ndh), drdh_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{stem} MO-J3 de_L/dr_dn")


# --------------------------------------------------------------------------- #
# RAW-AO SPHERICAL J3: the molecular J3 force/Pulay/stress kernels
# (``forces::grad_j3_wrt_r``, ``forces::j3_side``, ``stress::j3_grad_wrt_r``)
# evaluate CARTESIAN AO derivatives via ``ao_cart_deriv`` (empty ``poly_x/y/z`` on a
# spherical leaf -> panic ``index out of bounds``). The committed ``rs_ham`` sphe
# fixtures are jnone/agp -- a raw-AO ``sphe_gto`` J3 with a NONZERO ``j_matrix`` was
# never exercised (the cart H2 row and MO-wrapped rows above are cart-only). This row
# builds one and pins that the native VMC HF force (``vmc_grad_e_L_ae``) AND Pulay
# force (``vmc_grad_ln_psi``) lower the spherical J3 to its cartesian AO equivalent
# (``_aos_sphe_to_cart`` + ``lower_j3_to_ao``, ``j_matrix_cart = T^T M T``) -- the J3
# factor is basis-invariant, so both match the JAX oracle.
# --------------------------------------------------------------------------- #
def _build_raw_sphe_j3_ham(stem):
    import os

    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import (
        Jastrow_data,
        Jastrow_one_body_data,
        Jastrow_three_body_data,
        Jastrow_two_body_data,
    )
    from jqmc.trexio_wrapper import read_trexio_file
    from jqmc.wavefunction import Wavefunction_data

    trexio_dir = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files")
    sd, aos, _mos, _mos_no, gem_mo, cp = read_trexio_file(trexio_file=os.path.join(trexio_dir, stem), store_tuple=True)
    n_atom = len(sd.atomic_numbers)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0, structure_data=sd, core_electrons=core_electrons_of(cp, n_atom), jastrow_1b_type="exp"
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp")
    # RAW-AO J3 (orb_data = the spherical AO leaf), NONZERO random j_matrix.
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos, random_init=True, random_scale=1.0e-3, seed=123)
    jas = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=j3)
    ham = Hamiltonian_data(
        structure_data=sd,
        coulomb_potential_data=cp,
        wavefunction_data=Wavefunction_data(jastrow_data=jas, geminal_data=gem_mo),
    )
    ham.sanity_check()
    return ham


@pytest.mark.parametrize("stem", ["H2_ae_ccpvdz_sphe.h5"])
def test_vmc_hf_and_pulay_raw_sphe_j3(stem):
    """Vmc HF and pulay raw sphe j3."""
    ham = _build_raw_sphe_j3_ham(stem)
    # the J3 is a raw spherical AO leaf (num_ao AOs, no MO wrap) -- else this row is vacuous.
    j3o = ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data
    assert type(j3o).__name__ == "AOs_sphe_gto_data"
    ru, rd = _cfg(ham, seed=0)
    ginv = _geminv(ham, ru, rd)
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cp = ham.coulomb_potential_data
    # HF force de_L/d(R, r).
    dRh_j, druh_j, drdh_j = _jax_ref(ham, ru, rd, ginv)
    nRh, nuh, ndh = jqmc_rs.vmc_grad_e_L_ae(gd, jd, cp, ru, rd, ginv)
    np.testing.assert_allclose(np.asarray(nRh), dRh_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{stem} raw-sphe-J3 de_L/dR")
    np.testing.assert_allclose(np.asarray(nuh), druh_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{stem} raw-sphe-J3 de_L/dr_up")
    np.testing.assert_allclose(np.asarray(ndh), drdh_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{stem} raw-sphe-J3 de_L/dr_dn")
    # Pulay dln|Psi|/d(R, r).
    dR_j, dru_j, drd_j = _jax_ref_pulay(ham, ru, rd, ginv)
    nR, nu, nd = jqmc_rs.vmc_grad_ln_psi(gd, jd, ru, rd, ginv)
    np.testing.assert_allclose(np.asarray(nR), dR_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{stem} raw-sphe-J3 dln|Psi|/dR")
    np.testing.assert_allclose(np.asarray(nu), dru_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{stem} raw-sphe-J3 dln|Psi|/dr_up")
    np.testing.assert_allclose(np.asarray(nd), drd_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{stem} raw-sphe-J3 dln|Psi|/dr_dn")


# =========================================================================== #
# PERIODIC real-Gamma + COMPLEX-twist VMC HF + Pulay force.
# The MCMC HF force = dE_L/dR + Omega.dE_L/dr is the VMC CONTINUOUS periodic
# local-energy gradient (``vmc_grad_e_L_pbc`` / ``vmc_grad_e_L_pbc_complex`` --
# the ``gfmc_grad_e_l_ecp_pbc`` smooth base: periodic det+J kinetic + Ewald
# ee/ei/ii + ECP local/non-local continuum, WITHOUT the LRDMC S' discretization),
# and the Pulay force is the pure WF derivative (``vmc_grad_ln_psi_pbc`` /
# ``vmc_grad_ln_psi_pbc_complex``). Real reuses the sweep SM ``geminal_inv``; the
# complex path rebuilds the complex inverse fresh (matching the e_L path). Both
# match the JAX ``assemble.compute_grad_e_L`` / ``compute_grad_ln_Psi`` oracle at
# MEDIUM tol -- the SAME periodic-kinetic sphe-pGTO grad/lap fp-amplification floor
# the periodic e_L carries: two independent native cores (this
# smooth base and the LRDMC ``_gfmc_grad_e_L_ecp_pbc``) land at the same ~1e-6, so
# it is an fp floor, not a wiring error (Pulay, no kinetic-amplification, is ~1e-8).
# The complex-twist Re-projection (holomorphic Pulay/drift -> Re) is checked
# directly against the XLA/JAX analytical-autodiff route.
# =========================================================================== #
from tests.test_rs_kernel.test_gfmc_forces import _pbc_cfg, _rand_j3  # noqa: E402
from tests._ecp_core_electrons import core_electrons_of

# Periodic / twist force fp floor: the position-derivative of the periodic e_L
# AMPLIFIES the periodic-kinetic sphe-pGTO grad/lap far-from-cell-image fp floor the
# e_L already carries at MEDIUM tol. Measured across
# seeds 0-2 x {jnone, j12, j123, jdtn}: HF force <= ~1e-5 abs / ~1.6e-5 rel (worst on
# JNONE, whose bare det-kinetic grad^2/G has no Jastrow damping of the far-image
# noise); Pulay force is ~1e-8 (no kinetic amplification). ``2e-5`` covers the measured
# HF floor with margin -- still a strong bug guard (a wrong S' inclusion or a mis-wired
# term is O(0.1-1), e.g. the twist composite-PW gap is ~1.9e1). NOT the strict/medium
# named levels: this is the documented derivative-of-medium-e_L-floor treatment
# (constants + rationale in jqmc/_setting.py).
_ATOL_PBC, _RTOL_PBC = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
_ATOL_PBC = max(_ATOL_PBC, ATOL_FLOOR_MEDIUM)
_RTOL_PBC = max(_RTOL_PBC, RTOL_FLOOR_MEDIUM)

_PBC_REAL = ["pbc_gamma_sphepgto_jnone", "pbc_gamma_sphepgto_j12", "pbc_gamma_sphepgto_j123", "pbc_gamma_sphepgto_jdtn"]
_PBC_TWIST = ["twist_sphepgto_jnone", "twist_sphepgto_j12", "twist_sphepgto_jdtn"]

# FAR-ROAMED coverage (the gap that let two far-config force bugs ship):
# ``far=True`` scatters the electrons with scale 4.0 (several lattice vectors out of the home
# cell), the sampling that exposes (1) a J3 drift/value AO evaluated at the RAW
# coordinate (~1e-2 in the far j123 HF force, ~4e-2 in its Pulay dR) and (2) a RAW-Euclidean
# displacement in the DTN force kernels where jqmc min-images (~7e-1 HF / ~3e-1 Pulay on jdtn).
# NB a rigid integer-lattice shift of a near config (the ``test_grads_laplacian_far`` recipe)
# does NOT expose either error -- every kernel wraps/min-images the shift away identically --
# so the far rows here scatter instead, with seeds 3 and 6 = draws on which those errors are
# large (j123 HF 1.2e-2 / Pulay 3.8e-2 at seed 3; jdtn HF 7.3e-1 at seed 6). The kernels wrap
# (J3 home-cell wrap in ``compute_grads_and_laplacian_jastrow_three_body`` / ``jastrow_value``;
# DTN ``min_image_disp`` in ``forces_dtn``), so far configs sit at/below the same documented
# fp floor as near.
_FAR_SCALE = 4.0


def _pbc_geminv(ham, ru, rd, cplx):
    _, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    return np.asarray(ginv[0], dtype=(np.complex128 if cplx else np.float64))


@pytest.mark.parametrize("seed,far", [(0, False), (3, True), (6, True)], ids=["near", "far3", "far6"])
@pytest.mark.parametrize("pattern", _PBC_REAL)
def test_vmc_force_pbc_real(pattern, seed, far):
    """Vmc force PBC real."""
    ham = rs_ham(pattern)
    if "j123" in pattern:
        ham = ham.replace(wavefunction_data=_rand_j3(ham.wavefunction_data))
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg(ham, seed, scale=_FAR_SCALE) if far else _pbc_cfg(ham, seed)
    ginv = _pbc_geminv(ham, ru, rd, False)
    # HF force
    dR_j, dru_j, drd_j = _jax_ref(ham, ru, rd, ginv)
    nR, nu, nd = jqmc_rs.vmc_grad_e_L_pbc(gd, jd, cp, ru, rd, _W, _GP, ginv)
    np.testing.assert_allclose(np.asarray(nR), dR_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} HF dR")
    np.testing.assert_allclose(np.asarray(nu), dru_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} HF dr_up")
    np.testing.assert_allclose(np.asarray(nd), drd_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} HF dr_dn")
    # Pulay force
    pR_j, pru_j, prd_j = _jax_ref_pulay(ham, ru, rd, ginv)
    pR, pu, pd = jqmc_rs.vmc_grad_ln_psi_pbc(gd, jd, ru, rd, ginv)
    np.testing.assert_allclose(np.asarray(pR), pR_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} Pulay dR")
    np.testing.assert_allclose(np.asarray(pu), pru_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} Pulay dr_up")
    np.testing.assert_allclose(np.asarray(pd), prd_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} Pulay dr_dn")


@pytest.mark.parametrize("seed,far", [(0, False), (3, True), (6, True)], ids=["near", "far3", "far6"])
@pytest.mark.parametrize("pattern", _PBC_TWIST)
def test_vmc_force_pbc_complex(pattern, seed, far):
    """Vmc force PBC complex."""
    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg(ham, seed, scale=_FAR_SCALE) if far else _pbc_cfg(ham, seed)
    ginv = _pbc_geminv(ham, ru, rd, True)
    # HF force (real, Re-projected inside the complex kernels)
    dR_j, dru_j, drd_j = _jax_ref(ham, ru, rd, ginv)
    nR, nu, nd = jqmc_rs.vmc_grad_e_L_pbc_complex(gd, jd, cp, ru, rd, _W, _GP, None)
    np.testing.assert_allclose(np.asarray(nR), dR_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} HF dR")
    np.testing.assert_allclose(np.asarray(nu), dru_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} HF dr_up")
    np.testing.assert_allclose(np.asarray(nd), drd_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} HF dr_dn")
    # Pulay force (Re-projected holomorphic Pulay/drift)
    pR_j, pru_j, prd_j = _jax_ref_pulay(ham, ru, rd, ginv)
    pR, pu, pd = jqmc_rs.vmc_grad_ln_psi_pbc_complex(gd, jd, ru, rd)
    np.testing.assert_allclose(np.asarray(pR), pR_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} Pulay dR")
    np.testing.assert_allclose(np.asarray(pu), pru_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} Pulay dr_up")
    np.testing.assert_allclose(np.asarray(pd), prd_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} Pulay dr_dn")


@pytest.mark.parametrize("seed,far", [(0, False), (3, True)], ids=["near", "far3"])
def test_vmc_force_pbc_complex_mo_wrapped_j3(seed, far):
    """Genuine complex ``num_mo != num_ao`` J3: HF/Pulay force Rust == XLA.

    This is distinct from the generated identity-MO fixtures: the J3 basis is
    the committed complex ``(1, 26)`` occupied-MO projection, so a kernel must
    evaluate ``Re(C @ AO)`` at orbital rank rather than lower it through a real
    AO-space congruence or drop the MO layer.
    """
    ham = build_twist_mo_j3_ham()
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    j3_orb = jd.jastrow_three_body_data.orb_data
    mo = np.asarray(j3_orb.mo_coefficients)
    assert mo.shape[0] != mo.shape[1] and np.iscomplexobj(mo)

    ru, rd = _pbc_cfg(ham, seed, scale=_FAR_SCALE) if far else _pbc_cfg(ham, seed)
    ginv = _pbc_geminv(ham, ru, rd, True)

    hf_x = _jax_ref(ham, ru, rd, ginv)
    hf_r = jqmc_rs.vmc_grad_e_L_pbc_complex(gd, jd, cp, ru, rd, _W, _GP, ginv)
    for got, ref, tag in zip(hf_r, hf_x, ("dR", "dr_up", "dr_dn")):
        np.testing.assert_allclose(np.asarray(got), ref, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"complex MO-J3 HF {tag}")

    pp_x = _jax_ref_pulay(ham, ru, rd, ginv)
    pp_r = jqmc_rs.vmc_grad_ln_psi_pbc_complex(gd, jd, ru, rd, ginv)
    for got, ref, tag in zip(pp_r, pp_x, ("dR", "dr_up", "dr_dn")):
        np.testing.assert_allclose(np.asarray(got), ref, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"complex MO-J3 Pulay {tag}")


# --------------------------------------------------------------------------- #
# Real-Gamma PURE plane-wave VMC force. A pure-PW geminal
# has NO atom-centered pgto leaf (PW basis is R-independent -> dAO/dR = 0); the det
# nuclear/Pulay force is exactly zero and the electron-axis kinetic force is served
# via ``pw_det_egrad_side``. ``pgto_force_leaves`` must short-circuit on the
# ``plane_wave`` leaf; without it the force path reads ``_num_ao_kernel`` off the PW
# leaf and raises ``AttributeError``. No other force row exercises a pure-PW real
# leaf (the composite rows keep a pgto child; the twist-exotic rows are complex).
# --------------------------------------------------------------------------- #
def test_vmc_force_pbc_pure_pw():
    """Vmc force PBC pure pw."""
    ham = rs_ham("pw_gamma_jnone")
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg(ham, 0)
    ginv = _pbc_geminv(ham, ru, rd, False)
    dR_j, dru_j, drd_j = _jax_ref(ham, ru, rd, ginv)
    nR, nu, nd = jqmc_rs.vmc_grad_e_L_pbc(gd, jd, cp, ru, rd, _W, _GP, ginv)
    np.testing.assert_allclose(np.asarray(nR), dR_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="pure-PW HF dR")
    np.testing.assert_allclose(np.asarray(nu), dru_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="pure-PW HF dr_up")
    np.testing.assert_allclose(np.asarray(nd), drd_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="pure-PW HF dr_dn")
    pR, pu, pd = jqmc_rs.vmc_grad_ln_psi_pbc(gd, jd, ru, rd, ginv)
    pR_j, pru_j, prd_j = _jax_ref_pulay(ham, ru, rd, ginv)
    np.testing.assert_allclose(np.asarray(pR), pR_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="pure-PW Pulay dR")
    np.testing.assert_allclose(np.asarray(pu), pru_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="pure-PW Pulay dr_up")
    np.testing.assert_allclose(np.asarray(pd), prd_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="pure-PW Pulay dr_dn")


# --------------------------------------------------------------------------- #
# Composite pgto(+)pw ECP mesh force for an UNWRAPPED walker. The two
# configurations are the GFMC_n walker states of the
# ``test_gfmc_rust_vs_xla_end_to_end_force_stress[real-pgto-pw-n]`` row (seed 34356,
# fixed mesh, steps 22 and 32): the dn electron sits one lattice vector outside the home
# cell along a2 and within the ECP range of atom 1. The Rust composite mesh fast path took
# the moved electron's MO gradient at the RAW mesh point while the base fields and the
# moved MO value are home-wrapped, so the pgto block of that column fell outside the static
# image tiling and the atom-1 HF force was off by 1.6e-2 (step 22) and 1.4e-1 (step 32)
# while e_L stayed exact. A central finite difference of the production e_L with rebuilt
# displaced-nucleus Hamiltonians sided with XLA.
# Both backends must agree AND each must be invariant under translating the dn
# electron by a lattice vector.
# --------------------------------------------------------------------------- #
_COMPOSITE_GFMC_CONFIGS = {
    "gfmc_n_step22": (
        [[5.171481612234705, 3.151144777064383, 3.4986418066484863]],
        [[2.9913346531123626, 8.641776415108007, 3.206447506207641]],
    ),
    "gfmc_n_step32": (
        [[6.671481612234704, 4.651144777064382, 5.5986418066484855]],
        [[2.9913346531123626, 8.941776415108007, 3.8064475062076406]],
    ),
}


@pytest.mark.parametrize("config", sorted(_COMPOSITE_GFMC_CONFIGS))
def test_vmc_force_pbc_composite_pgto_pw_unwrapped_walker(config):
    """Composite pgto+pw ECP HF force: Rust == XLA and lattice-translation invariant."""
    ham = rs_ham("pbc_composite_jnone")
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru = np.asarray(_COMPOSITE_GFMC_CONFIGS[config][0], dtype=np.float64)
    rd = np.asarray(_COMPOSITE_GFMC_CONFIGS[config][1], dtype=np.float64)
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    # Precondition: the dn electron is outside the home cell (the unwrapped branch).
    frac_dn = rd[0] @ np.linalg.inv(cell)
    assert np.any((frac_dn < 0.0) | (frac_dn >= 1.0)), f"{config}: dn electron is inside the home cell"

    def both(ru_, rd_):
        ginv = _pbc_geminv(ham, ru_, rd_, False)
        xla = _jax_ref(ham, ru_, rd_, ginv)
        rust = tuple(np.asarray(x) for x in jqmc_rs.vmc_grad_e_L_pbc(gd, jd, cp, ru_, rd_, _W, _GP, ginv))
        for arr, tag in zip(xla + rust, ("xla dR", "xla dr_up", "xla dr_dn", "rust dR", "rust dr_up", "rust dr_dn")):
            assert np.all(np.isfinite(arr)), f"{config}: non-finite {tag}"
        return xla, rust

    tags = ("dR", "dr_up", "dr_dn")
    xla0, rust0 = both(ru, rd)
    for got, ref, tag in zip(rust0, xla0, tags):
        np.testing.assert_allclose(
            got, ref, atol=_ATOL_PBC, rtol=_RTOL_PBC, equal_nan=False, err_msg=f"{config} Rust vs XLA HF {tag}"
        )
    for shift in (-1, 1):
        rd_s = rd.copy()
        rd_s[0] = rd_s[0] + shift * cell[1]
        xla_s, rust_s = both(ru, rd_s)
        for backend, base, moved in (("xla", xla0, xla_s), ("rust", rust0, rust_s)):
            for got, ref, tag in zip(moved, base, tags):
                np.testing.assert_allclose(
                    got,
                    ref,
                    atol=_ATOL_PBC,
                    rtol=_RTOL_PBC,
                    equal_nan=False,
                    err_msg=f"{config} {backend} HF {tag} not invariant under dn += {shift}*a2",
                )


# --------------------------------------------------------------------------- #
# Periodic MULTI-ELECTRON cell + Jastrow-OWN-metric coverage.
# H2_pbc AE cart (2 electrons) with the dispatch-style J1+J2. jqmc keys EACH Jastrow
# term's metric on ITS OWN structure_data, and ``_build_ham``
# threads the loaded periodic structure into that J2, so its metric is the production
# Williamson/min-image periodic metric -- matching the geminal cell.
# (``Hamiltonian_data.sanity_check`` also REJECTS a periodic ham whose J2 carries a
# molecular/Euclidean structure, so a cell-less-J2 variant of this row is a
# construction bug by contract and cannot be built.) Two gaps this row pins:
# (1) every prior periodic force validation ran on lithium fixtures whose single
#     e-e pair sits ~5 Bohr apart (exp-damped J2) -- ALL e-e electron-axis force
#     channels were effectively vacuous; H2's close pair makes them live.
# (2) the J2/J1 kinetic-force blocks must follow the J2's OWN cell (forces_pbc::
#     jastrow_dist_derivs); a prior geminal-cell-unconditional metric measured a
#     3.5e-2 electron-axis kinetic error before that fix.
# --------------------------------------------------------------------------- #
def test_vmc_force_pbc_h2_multielectron_periodic_j2():
    """Vmc force PBC h2 multielectron periodic j2."""
    from tests.test_rs_kernel.test_rust_dispatch_equivalence import _build_ham

    ham = _build_ham("H2_pbc_ae_cart.h5")
    # Periodic ham whose J2 carries the periodic cell: its metric follows the
    # J2's own structure, which sanity_check requires to match the periodic lattice.
    assert ham.structure_data.pbc_flag
    assert ham.wavefunction_data.jastrow_data.jastrow_two_body_data.structure_data.pbc_flag
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg(ham, 0)
    ginv = _pbc_geminv(ham, ru, rd, False)
    dR_j, dru_j, drd_j = _jax_ref(ham, ru, rd, ginv)
    nR, nu, nd = jqmc_rs.vmc_grad_e_L_pbc(gd, jd, cp, ru, rd, _W, _GP, ginv)
    np.testing.assert_allclose(np.asarray(nR), dR_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="H2 periodic-J2 HF dR")
    np.testing.assert_allclose(np.asarray(nu), dru_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="H2 periodic-J2 HF dr_up")
    np.testing.assert_allclose(np.asarray(nd), drd_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="H2 periodic-J2 HF dr_dn")
    pR_j, pru_j, prd_j = _jax_ref_pulay(ham, ru, rd, ginv)
    pR, pu, pd = jqmc_rs.vmc_grad_ln_psi_pbc(gd, jd, ru, rd, ginv)
    np.testing.assert_allclose(np.asarray(pR), pR_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="H2 periodic-J2 Pulay dR")
    np.testing.assert_allclose(np.asarray(pu), pru_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="H2 periodic-J2 Pulay dr_up")
    np.testing.assert_allclose(np.asarray(pd), prd_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="H2 periodic-J2 Pulay dr_dn")


# =========================================================================== #
# (b) PERIODIC real-Gamma genuinely-MO-wrapped J3 (num_mo != num_ao) VMC HF + Pulay force
# Production periodic J3 (e.g. lithium 1<-26) wraps the geminal MOs, so
# ``j3_pbc_ctx_owned`` lowers it to the exact AO-equivalent (``lower_j3_to_ao``: raw-AO orbital
# + ``M_ao = C^T M_mo C``) -- else ``j3_pbc_side_w``'s ``jm[.., 0..num_ao]`` slice panicked on
# the 1x2 MO j_matrix. The committed ``rs_ham`` periodic J3 fixtures are raw-AO (pgto), so this
# builds the production MO-wrapped J3 explicitly (as the heavy MCMC driver does) and pins that
# the native HF + Pulay lower it correctly, near AND far-roamed (the far config that exposed the
# J3 raw-coordinate bug). ``lithium.h5`` is real-Gamma -> real ``mo_coefficients`` (lowerable).
# --------------------------------------------------------------------------- #
def _build_pbc_mo_j3_ham(stem):
    import os

    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import (
        Jastrow_data,
        Jastrow_one_body_data,
        Jastrow_three_body_data,
        Jastrow_two_body_data,
    )
    from jqmc.trexio_wrapper import read_trexio_file
    from jqmc.wavefunction import Wavefunction_data

    trexio_dir = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files")
    sd, _aos, mos, _mos_no, gem_mo, cp = read_trexio_file(trexio_file=os.path.join(trexio_dir, stem), store_tuple=True)
    n_atom = len(sd.atomic_numbers)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0, structure_data=sd, core_electrons=core_electrons_of(cp, n_atom), jastrow_1b_type="exp"
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=sd)
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=mos, random_init=True, random_scale=1.0e-3, seed=123)
    jas = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=j3)
    ham = Hamiltonian_data(
        structure_data=sd,
        coulomb_potential_data=cp,
        wavefunction_data=Wavefunction_data(jastrow_data=jas, geminal_data=gem_mo),
    )
    ham.sanity_check()
    return ham


@pytest.mark.parametrize("seed,far", [(0, False), (3, True), (6, True)], ids=["near", "far3", "far6"])
def test_vmc_force_pbc_mo_wrapped_j3(seed, far):
    """Vmc force PBC MO wrapped j3."""
    ham = _build_pbc_mo_j3_ham("lithium.h5")
    # the production J3 is genuinely MO-wrapped (num_mo != num_ao) -- else this row is vacuous.
    j3o = ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data
    j3mc = np.asarray(j3o.mo_coefficients)
    assert j3mc.ndim == 2 and j3mc.shape[0] != j3mc.shape[1]
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg(ham, seed, scale=_FAR_SCALE) if far else _pbc_cfg(ham, seed)
    ginv = _pbc_geminv(ham, ru, rd, False)
    # HF force (native lowers MO-J3 to AO inside vmc_grad_e_L_pbc)
    dR_j, dru_j, drd_j = _jax_ref(ham, ru, rd, ginv)
    nR, nu, nd = jqmc_rs.vmc_grad_e_L_pbc(gd, jd, cp, ru, rd, _W, _GP, ginv)
    np.testing.assert_allclose(np.asarray(nR), dR_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="MO-J3 HF dR")
    np.testing.assert_allclose(np.asarray(nu), dru_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="MO-J3 HF dr_up")
    np.testing.assert_allclose(np.asarray(nd), drd_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="MO-J3 HF dr_dn")
    # Pulay force
    pR_j, pru_j, prd_j = _jax_ref_pulay(ham, ru, rd, ginv)
    pR, pu, pd = jqmc_rs.vmc_grad_ln_psi_pbc(gd, jd, ru, rd, ginv)
    np.testing.assert_allclose(np.asarray(pR), pR_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="MO-J3 Pulay dR")
    np.testing.assert_allclose(np.asarray(pu), pru_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="MO-J3 Pulay dr_up")
    np.testing.assert_allclose(np.asarray(pd), prd_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="MO-J3 Pulay dr_dn")
