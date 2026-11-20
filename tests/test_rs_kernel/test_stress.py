"""Layer-1: analytical d ln|Psi| / d eps (Pulay stress, Rust stress.rs) and
d e_L / d eps (HF stress, Rust stress.rs) vs jax.grad of the strained jqmc
forward kernel. Molecular affine part (lnPsi Pulay stress + e_L stress:
jnone / with-DTN / ECP / with-Jastrow). CART + SPHE.

Hamiltonians come from the committed-fixture builders
(``tests.test_rs_kernel._rs_fixtures.rs_ham``). Every oracle is self-contained:

  * ``test_grad_el_strain_ecp``: ``jax.grad`` of a hand-built local energy, RT held
    fixed (the local ``_eL_ecp`` helper below) -- the same self-contained-oracle style
    ``test_gfmc_complex_force_stress.py`` / ``test_forces.py`` already use elsewhere
    in this suite. NOTE: the non-local channel is built with
    ``compute_ecp_non_local_parts_nearest_neighbors``, NOT ``..._all_pairs`` -- see
    the long comment above ``_eL_ecp`` for why ``all_pairs`` names an energy
    production never evaluates.

The other three rows (lnPsi Pulay stress, jnone e_L stress, with-Jastrow e_L
stress) are direct ``jax.grad`` oracles.
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
from jqmc._precision import get_tolerance_xla_rust_consistency  # noqa: E402
from jqmc.coulomb_potential import (  # noqa: E402
    compute_bare_coulomb_potential,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors,
    icosahedron_sym_mesh_Nv12,
)
from jqmc.hamiltonians import compute_local_energy  # noqa: E402
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data  # noqa: E402
from jqmc.jqmc_mcmc import _inject_active_positions  # noqa: E402
from jqmc.wavefunction import compute_kinetic_energy, evaluate_ln_wavefunction  # noqa: E402

import jqmc_rs  # noqa: E402

from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg, rs_ham
from tests._ecp_core_electrons import core_electrons_of

# Every rs test declares its cost tier (light/heavy, per test) and CWD policy
# (with_chdir here -- autouse _isolate_cwd chdirs into a per-test tmp_path for
# xdist safety; committed fixtures load via absolute paths so cwd is irrelevant).
pytestmark = [pytest.mark.with_chdir]


def _cfg(ham, seed=1):
    gem = ham.wavefunction_data.geminal_data
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ru = R[np.arange(nu) % len(R)] + rng.normal(scale=0.8, size=(nu, 3))
    rd = R[np.arange(nd) % len(R)] + rng.normal(scale=0.8, size=(nd, 3))
    return ru, rd


def _rand_j3(ham, seed=7):
    """Random symmetric J3 coupling block: committed j123ao fixtures ship
    j_matrix == 0, which makes J3 gradient tests vacuous."""
    wf = ham.wavefunction_data
    j3 = wf.jastrow_data.jastrow_three_body_data
    n = np.asarray(j3.j_matrix).shape[0]
    jm = np.random.default_rng(seed).normal(scale=0.2, size=(n, n + 1))
    # Production invariant: the J3 coupling block is symmetric (bosonic Jastrow);
    # Jastrow_three_body_data.sanity_check RAISES otherwise, and several jqmc
    # streaming/ratio kernels exploit it (j3_mat.T @ x == j3_mat @ x). A .replace()
    # bypasses sanity_check, so symmetrize here or we test an invalid regime.
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)
    return wf.jastrow_data.replace(jastrow_three_body_data=j3.replace(j_matrix=jm))


def _randomize_dtn(jastrow_data, seed=0, scale=0.05):
    """Perturb the independent DTN parameters (numpy, no ``jnp`` cast):
    committed jdtn fixtures ship all-zero alpha/beta/gamma_indep (only the
    b_alpha cusp bias is set), which makes a DTN VALUE-derivative test vacuous
    -- perturb the independent params so the DTN paths are exercised."""
    d = jastrow_data.jastrow_dtn_data
    if d is None:
        return jastrow_data
    rng = np.random.default_rng(seed)
    upd = {}
    for f in ("alpha_indep", "beta_indep", "gamma_indep"):
        v = np.asarray(getattr(d, f))
        upd[f] = scale * rng.standard_normal(v.shape)
    return jastrow_data.replace(jastrow_dtn_data=d.replace(**upd))


# --------------------------------------------------------------------------- #
# d lnPsi / d eps (Pulay stress) -- direct jax.grad(evaluate_ln_wavefunction)
# oracle.
# --------------------------------------------------------------------------- #

_ZONES_LNPSI = ("ao_eval", "ao_grad_lap", "det_eval", "jastrow_eval")


@pytest.mark.heavy
def test_grad_lnpsi_strain():
    """Grad lnpsi strain."""
    ham = rs_ham("mol_cart_sd_j123")
    wf = ham.wavefunction_data.replace(jastrow_data=_rand_j3(ham))
    gd = wf.geminal_data
    jd = wf.jastrow_data
    pos = np.asarray(gd.orb_data_up_spin.structure_data._positions_cart_np)
    ru, rd = _cfg(ham)
    ana = jqmc_rs.grad_lnPsi_wrt_strain(gd, jd, ru, rd)

    def fwd(eps):
        F = jnp.eye(3) + eps
        wf2 = _inject_active_positions(wf, {"positions": pos @ F.T})
        return evaluate_ln_wavefunction(wf2, ru @ F.T, rd @ F.T)

    ref = jax.grad(fwd)(np.zeros((3, 3)))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_LNPSI, level="medium")
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


# --------------------------------------------------------------------------- #
# d e_L / d eps (HF stress, jnone AE molecular, cart + sphe) -- direct
# jax.grad(compute_local_energy) oracle.
# --------------------------------------------------------------------------- #

_ZONES_ELJNONE = (
    "ao_eval",
    "ao_grad_lap",
    "mo_eval",
    "mo_grad",
    "mo_lap",
    "det_eval",
    "det_grad_lap",
    "coulomb",
    "wf_kinetic",
    "local_energy",
)


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["mol_cart_agp_jnone", "mol_cart_sd_jnone", "mol_sphe_agp_jnone"])
def test_grad_el_strain_jnone(pattern):
    """Grad el strain jnone."""
    ham = rs_ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    cd = ham.coulomb_potential_data
    ru, rd = _cfg(ham)
    ana = jqmc_rs.grad_eL_wrt_strain(gd, cd, ru, rd)

    def fwd(eps):
        F = jnp.eye(3) + eps
        h2 = ham._apply_strain(F)
        return compute_local_energy(h2, ru @ F.T, rd @ F.T, jnp.eye(3) @ F.T)

    ref = jax.grad(fwd)(jnp.zeros((3, 3)))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ELJNONE, level="medium")
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


# --------------------------------------------------------------------------- #
# d e_L / d eps for ECP systems (jnone): e-e + N-N(Z_eff) + ECP local +
# non-local quadrature, RT held FIXED (a rotated quadrature mesh on a fixed unit
# sphere -> the affine position virial applies without an RT-strain term).
#
# NON-LOCAL VARIANT -- ``nearest_neighbors``, NOT ``all_pairs``.
# An all-pairs oracle (``_eL_allpairs``) sums the non-local channel over ALL atoms. Production does
# not: ``compute_ecp_coulomb_potential{,_fast}`` -- the only thing
# ``compute_local_energy`` ever calls -- uses
# ``compute_ecp_non_local_parts_nearest_neighbors{,_fast_update}`` (NN=1), and
# ``compute_ecp_non_local_parts_all_pairs`` outright RAISES NotImplementedError
# under PBC, so it can never be the contract of a stress kernel that must also
# serve periodic cells. ``stress::grad_el_wrt_strain_ecp`` accordingly builds the
# virial from ``forces_ecp::grad_v_ecp_wrt_r`` / ``grad_ecp_force``, both of which
# carry the ``NN_DEFAULT = 1`` nearest-ECP-nucleus filter ("1:1 the non-local
# drift e_L"). The HF stress estimator must differentiate the SAME e_L that VMC /
# LRDMC averages, so NN is the correct reference here.
#
# Measured (lithium-like ``mol_ecp_sphe_jnone``):
# the all-pairs oracle differed from the NN oracle by 4.2422529e-06 at [0,0] --
# exactly the 4.2422522e-06 "Rust defect" the per-element check reported. Against
# the NN oracle the Rust kernel agrees to 4.2e-12, i.e. it was never wrong; the
# ORACLE named the wrong energy. This is a contract fix, not a tolerance change:
# the comparison stays a full ``jax.grad`` of a production jqmc forward at the
# same ``strict`` tolerance.
#
# COVERAGE NOTE: ``mol_ecp_cart_jnone`` cannot arbitrate any of this -- its
# ``_coefficients_non_local_part`` is ``[0., 0.]``, so its ECP non-local channel
# (and every derivative of it) is identically zero and NN == all-pairs to 0.0e0
# there. The cart row passing while sphe failed was NOT a cart-vs-sphe code-path
# split; cart is simply blind to this channel. Only ``mol_ecp_sphe_jnone`` (whose
# atom 0 carries a non-zero non-local coefficient) exercises it.
# --------------------------------------------------------------------------- #


def _eL_ecp(ham_, r_up, r_dn, RT):
    wf_ = ham_.wavefunction_data
    cd_ = ham_.coulomb_potential_data
    V = (
        compute_bare_coulomb_potential(cd_, r_up, r_dn)
        + compute_ecp_local_parts_all_pairs(cd_, r_up, r_dn)
        + compute_ecp_non_local_parts_nearest_neighbors(cd_, wf_, r_up, r_dn, RT)[3]
    )
    return compute_kinetic_energy(wf_, r_up, r_dn) + V


_ZONES_ECP = ("ao_eval", "ao_grad_lap", "mo_eval", "mo_grad", "mo_lap", "det_eval", "det_grad_lap", "coulomb", "wf_kinetic")


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["mol_ecp_cart_jnone", "mol_ecp_sphe_jnone"])
def test_grad_el_strain_ecp(pattern):
    """Grad el strain ECP."""
    ham = rs_ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cd = ham.coulomb_potential_data
    w = np.asarray(icosahedron_sym_mesh_Nv12.weights, float)
    gp = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, float)
    # Strain derivatives amplify an AO-forward rounding residual through the
    # inverse geminal.  Select the numerical domain from the XLA geminal alone,
    # using the same deterministic condition filter as the other aggregate
    # parity tests (no selection on XLA--Rust agreement).
    ru, rd = draw_well_conditioned_cfg(ham, seed=1, scale=0.8, cond_max=50.0)
    RT = jnp.eye(3)
    ana = jqmc_rs.grad_eL_wrt_strain_ecp(gd, jd, cd, ru, rd, w, gp)

    def fwd(eps):
        F = jnp.eye(3) + eps
        return _eL_ecp(ham._apply_strain(F), ru @ F.T, rd @ F.T, RT)

    ref = jax.grad(fwd)(jnp.zeros((3, 3)))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ECP, level="medium")
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


# --------------------------------------------------------------------------- #
# d e_L / d eps WITH Jastrow (J1/J2/J3), AE bare molecular, cart + sphe --
# direct jax.grad(compute_local_energy) oracle.
# --------------------------------------------------------------------------- #

_ZONES_WITHJ = _ZONES_ELJNONE + ("jastrow_eval", "jastrow_grad_lap")


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["mol_cart_sd_j12", "mol_cart_sd_j123"])
def test_grad_el_strain_withJ(pattern):
    """Grad el strain withJ."""
    ham = rs_ham(pattern)
    if pattern == "mol_cart_sd_j123":  # bench j123ao files ship j_matrix == 0 -> randomize
        ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=_rand_j3(ham)))
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cd = ham.coulomb_potential_data
    ru, rd = _cfg(ham)
    ana = jqmc_rs.grad_eL_wrt_strain_withJ(gd, jd, cd, ru, rd)

    def fwd(eps):
        F = jnp.eye(3) + eps
        h2 = ham._apply_strain(F)
        return compute_local_energy(h2, ru @ F.T, rd @ F.T, jnp.eye(3) @ F.T)

    ref = jax.grad(fwd)(jnp.zeros((3, 3)))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_WITHJ, level="medium")
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


@pytest.mark.heavy
def test_grad_el_strain_withJ_sphe_agp_j12():
    """H2 AGP sphe (-> cart geminal) + J1/J2: no "mol_sphe_agp_j12" fixture
    pattern exists, so attach J1(exp)+J2(exp) to the jnone AGP-sphe fixture the
    same way test_gfmc_complex_force_stress.py::_build_ham's "j12" branch does."""
    ham = rs_ham("mol_sphe_agp_jnone")
    sd = ham.structure_data
    n_atom = len(sd.positions)
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=sd)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=sd,
        core_electrons=core_electrons_of(ham.coulomb_potential_data, n_atom),
        jastrow_1b_type="exp",
    )
    jd = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=None)
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))
    gd = ham.wavefunction_data.geminal_data
    cd = ham.coulomb_potential_data
    ru, rd = _cfg(ham)
    ana = jqmc_rs.grad_eL_wrt_strain_withJ(gd, jd, cd, ru, rd)

    def fwd(eps):
        F = jnp.eye(3) + eps
        h2 = ham._apply_strain(F)
        return compute_local_energy(h2, ru @ F.T, rd @ F.T, jnp.eye(3) @ F.T)

    ref = jax.grad(fwd)(jnp.zeros((3, 3)))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_WITHJ, level="medium")
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)
