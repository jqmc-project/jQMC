"""Layer-1: analytical nuclear forces d(e_L)/dR (Rust forces.rs) vs jax.grad(jqmc
forward) wrt nucleus positions. Pulay (det/J1/J3), classical V, kinetic
force (jnone + withJ J1/J2/J3). CART + SPHE molecular.

Hamiltonians come from the committed-fixture builders (``tests.test_rs_kernel._rs_fixtures.rs_ham``)
and tolerances from the dedicated XLA--Rust consistency helper.

Two local pattern extensions (``mol_ecp_sphe_agp_jnone``, ``mol_ecp_sphe_jdtn``)
cover (AO family x geminal x Jastrow x ECP) combinations the committed
``_rs_fixtures._PATTERNS`` registry doesn't name yet; they are built directly via
``_build_ham_for_cell`` (the SAME shared builder ``rs_ham`` wraps, imported
read-only from ``_generated_matrix_base``), not invented from
scratch. A Rust==XLA equivalence is code-path sensitive, not system sensitive,
so any fixture flowing the same path stands in for the original bench system.
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

jax.config.update("jax_enable_x64", True)
from jqmc.jqmc_mcmc import _inject_active_positions  # noqa: E402
from jqmc.wavefunction import compute_kinetic_energy as CK  # noqa: E402

import jqmc_rs  # noqa: E402

from jqmc._precision import get_tolerance_xla_rust_consistency

from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import rs_ham

# Every rs test declares its cost tier (light/heavy, per test) and CWD policy
# (with_chdir here -- autouse _isolate_cwd chdirs into a per-test tmp_path for
# xdist safety; committed fixtures load via absolute paths so cwd is irrelevant).
pytestmark = pytest.mark.with_chdir

# name -> (stem, with_jastrow, spec, gem) for _build_ham_for_cell, mirroring
# _rs_fixtures._PATTERNS's shape but not promoted there (out of this task's
# edit scope). Verified buildable directly against the committed water ECP
# sphe stem: AGP dense-lift is well-conditioned on any molecular GTO leaf
# (``_agp_dense_constructible``, ao/bc only -- ECP-ness doesn't gate it), and
# the DTN spec only needs ``structure_data`` (AO-family independent).
_LOCAL_PATTERNS = {
    "mol_ecp_sphe_agp_jnone": ("water_ccecp_ccpvqz.h5", False, None, "AGP"),
    "mol_ecp_sphe_jdtn": ("water_ccecp_ccpvqz.h5", True, {"dtn": True}, "SD"),
}


def _ham(pattern):
    if pattern in _LOCAL_PATTERNS:
        return _build_ham_for_cell(*_LOCAL_PATTERNS[pattern])
    return rs_ham(pattern)


def _cfg(ham, seed=1):
    gem = ham.wavefunction_data.geminal_data
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ru = R[np.arange(nu) % len(R)] + rng.normal(scale=0.8, size=(nu, 3))
    rd = R[np.arange(nd) % len(R)] + rng.normal(scale=0.8, size=(nd, 3))
    return ru, rd


def _struct(od):
    """The structure_data reachable from a geminal orbital leaf: an MOs_data
    leaf nests it under ``aos_data``; a bare AO leaf (AGP) carries it directly.
    (MOs_data actually exposes a ``.structure_data`` property that already does
    this dispatch, so both branches agree -- kept explicit as a per-type check.)"""
    return od.aos_data.structure_data if type(od).__name__ == "MOs_data" else od.structure_data


def _pos(sd):
    return np.asarray(sd._positions_cart_np)


def _rand_j3(ham, seed=7):
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
    """Randomize the DTN parameters (numpy; callers here stay numpy):
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


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_agp_jnone", "mol_cart_sd_jnone"])
def test_grad_lndet_R(pattern):
    """Grad lndet R."""
    from jqmc.determinant import compute_ln_det_geminal_all_elements

    ham = _ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    # Keep the strict backend-parity SD point away from a determinant node.
    # seed=23 gives cond(G)=6.1e1; the default seed amplifies an
    # fp32 AO ULP into the nuclear derivative.
    ru, rd = _cfg(ham, seed=23 if pattern == "mol_cart_sd_jnone" else 1)
    ana = jqmc_rs.grad_lndet_wrt_R(gd, ru, rd)
    P = _pos(_struct(gd.orb_data_up_spin))
    ref = jax.grad(lambda P: compute_ln_det_geminal_all_elements(_inject_active_positions(gd, {"positions": P}), ru, rd))(P)
    atol, rtol = get_tolerance_xla_rust_consistency(("ao_eval", "det_eval"), level="medium")
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("seed", [1, 2])  # two data points on the same AE-cart-SD-jnone path
def test_grad_V_R(seed):
    """Grad V R."""
    from jqmc.coulomb_potential import compute_coulomb_potential

    ham = _ham("mol_cart_sd_jnone")
    cd = ham.coulomb_potential_data
    ru, rd = _cfg(ham, seed=seed)
    ana = jqmc_rs.grad_V_wrt_R(cd, ru, rd)
    P = _pos(cd.structure_data)
    ref = jax.grad(
        lambda P: compute_coulomb_potential(
            _inject_active_positions(cd, {"positions": P}), ru, rd, np.eye(3), wavefunction_data=ham.wavefunction_data
        )
    )(P)
    atol, rtol = get_tolerance_xla_rust_consistency(("coulomb",), level="medium")
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_agp_jnone", "mol_cart_sd_jnone"])
def test_grad_T_R_jnone(pattern):
    """Grad T R jnone."""
    ham = _ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    ru, rd = _cfg(ham, seed=23 if pattern == "mol_cart_sd_jnone" else 1)
    ana = jqmc_rs.grad_T_wrt_R(gd, jd, ru, rd)
    P = _pos(_struct(gd.orb_data_up_spin))
    ref = jax.grad(lambda P: CK(_inject_active_positions(ham.wavefunction_data, {"positions": P}), ru, rd))(P)
    atol, rtol = get_tolerance_xla_rust_consistency(
        ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic"), level="medium"
    )
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize(
    "pattern,withj3",
    [
        ("mol_cart_sd_j12", False),
        ("mol_cart_sd_j123", True),
    ],
)
def test_grad_T_R_withJ(pattern, withj3):
    """Grad T R withJ."""
    ham = _ham(pattern)
    wf = ham.wavefunction_data
    if withj3:
        wf = wf.replace(jastrow_data=_rand_j3(ham))
    gd = wf.geminal_data
    jd = wf.jastrow_data
    ru, rd = _cfg(ham)
    ana = jqmc_rs.grad_T_wrt_R(gd, jd, ru, rd)
    P = _pos(_struct(gd.orb_data_up_spin))
    ref = jax.grad(lambda P: CK(_inject_active_positions(wf, {"positions": P}), ru, rd))(P)
    atol, rtol = get_tolerance_xla_rust_consistency(
        ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic", "jastrow_grad_lap"),
        level="medium",
    )
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


# ---- sphe systems (sphe->cart conversion: to_cartesian / _aos_sphe_to_cart) ----


@pytest.mark.light
@pytest.mark.parametrize(
    "pattern",
    [
        "mol_ecp_sphe_jnone",  # SD sphe (MO)
        "mol_sphe_agp_jnone",  # AGP sphe (bare leaf)
    ],
)
def test_grad_lndet_R_sphe(pattern):
    """Grad lndet R sphe."""
    from jqmc.determinant import compute_ln_det_geminal_all_elements

    ham = _ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    # seed=24 gives cond(G)=6.0e1 for the spherical SD fixture.
    ru, rd = _cfg(ham, seed=24 if pattern == "mol_ecp_sphe_jnone" else 1)
    ana = jqmc_rs.grad_lndet_wrt_R(gd, ru, rd)
    P = _pos(_struct(gd.orb_data_up_spin))
    ref = jax.grad(lambda P: compute_ln_det_geminal_all_elements(_inject_active_positions(gd, {"positions": P}), ru, rd))(P)
    atol, rtol = get_tolerance_xla_rust_consistency(("ao_eval", "det_eval"), level="medium")
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize(
    "pattern",
    [
        "mol_ecp_sphe_jnone",
        "mol_sphe_agp_jnone",
        "mol_ecp_sphe_j12",  # withJ (J1+J2, no J3)
    ],
)
def test_grad_T_R_sphe(pattern):
    """Grad T R sphe."""
    ham = _ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    ru, rd = _cfg(ham, seed=24 if pattern.startswith("mol_ecp_sphe_") else 1)
    ana = jqmc_rs.grad_T_wrt_R(gd, jd, ru, rd)
    P = _pos(_struct(gd.orb_data_up_spin))
    ref = jax.grad(lambda P: CK(_inject_active_positions(ham.wavefunction_data, {"positions": P}), ru, rd))(P)
    atol, rtol = get_tolerance_xla_rust_consistency(
        ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic", "jastrow_grad_lap"),
        level="medium",
    )
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


# ---- ECP non-local nuclear force d(V_ecp)/dR (wf-ratio + explicit, jnone) ----


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1])  # two seeds on the single committed ECP-cart path
def test_grad_ecp_force_R(seed):
    """Grad ECP force R."""
    import jax.numpy as jnp

    from jqmc.coulomb_potential import compute_ecp_non_local_parts_all_pairs, icosahedron_sym_mesh_Nv12

    ham = _ham("mol_ecp_cart_jnone")
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    cd = ham.coulomb_potential_data
    w = np.array(icosahedron_sym_mesh_Nv12.weights, float)
    gp = np.array(icosahedron_sym_mesh_Nv12.grid_points, float)
    ru, rd = _cfg(ham, seed=seed)
    ana = jqmc_rs.grad_eL_ecp_force_wrt_R(gd, jd, cd, ru, rd, w, gp)
    struct = _struct(gd.orb_data_up_spin)
    pos = np.asarray(struct.positions)

    def full(P):
        wf2 = _inject_active_positions(wf, {"positions": P})
        cd2 = cd.replace(structure_data=cd.structure_data.replace(positions=P))
        return compute_ecp_non_local_parts_all_pairs(cd2, wf2, ru, rd, jnp.eye(3))[3]

    ref = jax.grad(full)(pos)
    atol, rtol = get_tolerance_xla_rust_consistency(
        ("coulomb", "ao_eval", "mo_eval", "det_eval", "jastrow_eval"), level="medium"
    )
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("seed", [9])
def test_grad_ecp_force_R_nn_filter(seed):
    """NN=1-vs-all-pairs regression for the ECP non-local force.

    ``grad_ecp_force`` sums the non-local projector / wf-ratio mesh only over each
    electron's NEAREST ECP nucleus (``NN_default = 1``), 1:1 the production
    nearest-neighbour non-local e_L. The sibling ``test_grad_ecp_force_R`` cannot see
    the filter: it uses H2 (the 2nd centre's non-local is ~0) AND an all-pairs JAX
    reference. Water ccECP (3 non-local centres) at a scattered config (seed 9) makes a
    non-NEAREST centre contribute: native (NN) matches ``jacrev(NN e_L)`` at ~1e-13 but
    diverges from all-pairs by ~7.6e-3. Reverting the ``forces_ecp`` NN filter
    (``collect_mesh`` NN_DEFAULT -> 0 + dropping the projector nearest-atom guard) makes
    native all-pairs and fails this row against the NN reference.
    """
    import jax.numpy as jnp

    from jqmc.coulomb_potential import (
        compute_ecp_non_local_parts_nearest_neighbors,
        icosahedron_sym_mesh_Nv12,
    )

    ham = _ham("mol_ecp_sphe_jnone")
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    cd = ham.coulomb_potential_data
    w = np.array(icosahedron_sym_mesh_Nv12.weights, float)
    gp = np.array(icosahedron_sym_mesh_Nv12.grid_points, float)
    # Scattered water config: an electron sits nearer a non-O centre so the non-nearest
    # ECP centre's projector is non-negligible (|NN - all_pairs| force ~7.6e-3), unlike
    # the default near-nucleus configs where NN == all-pairs.
    nup, ndn = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(cd.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[rng.integers(0, len(R), nup)] + rng.normal(scale=0.6, size=(nup, 3))
    rd = R[rng.integers(0, len(R), ndn)] + rng.normal(scale=0.6, size=(ndn, 3))
    ana = jqmc_rs.grad_eL_ecp_force_wrt_R(gd, jd, cd, ru, rd, w, gp)
    pos = np.asarray(cd.structure_data.positions)

    def full(P):
        wf2 = _inject_active_positions(wf, {"positions": P})
        cd2 = cd.replace(structure_data=cd.structure_data.replace(positions=P))
        return compute_ecp_non_local_parts_nearest_neighbors(cd2, wf2, jnp.array(ru), jnp.array(rd), jnp.eye(3))[3]

    ref = jax.grad(full)(pos)
    atol, rtol = get_tolerance_xla_rust_consistency(
        ("coulomb", "ao_eval", "mo_eval", "det_eval", "jastrow_eval"), level="medium"
    )
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1])
def test_grad_ecp_local_force_R(seed):
    """Grad ECP local force R."""
    from jqmc.coulomb_potential import compute_ecp_local_parts_all_pairs

    ham = _ham("mol_ecp_cart_jnone")
    cd = ham.coulomb_potential_data
    ru, rd = _cfg(ham, seed=seed)
    ana = jqmc_rs.grad_eL_ecp_local_force_wrt_R(cd, ru, rd)
    pos = np.asarray(cd.structure_data.positions)
    ref = jax.grad(
        lambda P: compute_ecp_local_parts_all_pairs(cd.replace(structure_data=cd.structure_data.replace(positions=P)), ru, rd)
    )(pos)
    atol, rtol = get_tolerance_xla_rust_consistency(("coulomb",), level="medium")
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


# ---- DTN Pulay force d(J_DTN)/dR (chi + f channels) ----


@pytest.mark.light
@pytest.mark.parametrize(
    "pattern",
    [
        "mol_cart_sd_jdtn",
        "mol_ecp_sphe_jdtn",  # local pattern extension: sphe + ECP + DTN
    ],
)
def test_grad_J_dtn_R(pattern):
    """Grad J DTN R."""
    from jqmc.jastrow_factor import compute_Jastrow_DTN

    ham = _ham(pattern)
    d = _randomize_dtn(ham.wavefunction_data.jastrow_data, seed=7).jastrow_dtn_data
    ru, rd = _cfg(ham)
    pos = np.asarray(d.structure_data.positions)
    ana = jqmc_rs.grad_J_dtn_wrt_R(d, ru, rd)
    ref = jax.grad(lambda P: compute_Jastrow_DTN(_inject_active_positions(d, {"positions": P}), ru, rd))(pos)
    atol, rtol = get_tolerance_xla_rust_consistency(("jastrow_eval", "jastrow_grad_lap"), level="medium")
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)
