"""Layer-1 equivalence: local + non-local ECP energy and the assembled
molecular local energy, Rust vs jqmc.

Fixtures are the committed hams. Each (AO family, ECP-presence, Jastrow) system
is the committed fixture pattern flowing the SAME code path (a Rust==XLA equivalence is
path-sensitive, not system-sensitive):
  - N2 ecp cart j12          -> mol_ecp_cart_j12   (H2 ecp cart j12)
  - water ecp sphe j12       -> mol_ecp_sphe_j12   (water ecp sphe j12)
  - Cl2 ecp cart jnone       -> mol_ecp_cart_jnone  (H2 ecp cart jnone)
  - H2 ae sphe jnone         -> mol_sphe_sd_jnone   (H2 ae sphe jnone)
For the J3 local-energy variant the committed fixtures ship no ECP+J3
pattern, so a Jastrow_three_body_data with a random j_matrix is attached directly.
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

import jqmc_rs

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.coulomb_potential import (
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_all_pairs,
    icosahedron_sym_mesh_Nv12,
    octahedron_sym_mesh_Nv6,
    octahedron_sym_mesh_Nv18,
    tetrahedron_sym_mesh_Nv4,
)
from jqmc.hamiltonians import compute_local_energy
from jqmc.jastrow_factor import Jastrow_three_body_data

from tests.test_rs_kernel._rs_fixtures import rs_ham

_ZONES_ECP = ("coulomb",)
_ZONES_ECP_NONLOCAL = ("coulomb", "det_ratio", "jastrow_ratio", "ao_eval", "jastrow_eval")
_ZONES_LOCAL_ENERGY = (
    "ao_eval",
    "ao_grad_lap",
    "det_eval",
    "det_grad_lap",
    "jastrow_eval",
    "jastrow_grad_lap",
    "coulomb",
    "wf_kinetic",
    "local_energy",
)

_ECP_PATTERNS = ["mol_ecp_cart_j12", "mol_ecp_sphe_j12", "mol_ecp_cart_jnone"]


def _electrons(ham, seed, scale=0.6):
    gem = ham.wavefunction_data.geminal_data
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    r_up = R[rng.integers(0, len(R), nu)] + rng.normal(scale=scale, size=(nu, 3))
    r_dn = R[rng.integers(0, len(R), nd)] + rng.normal(scale=scale, size=(nd, 3))
    return r_up.astype(np.float64), r_dn.astype(np.float64)


@pytest.mark.light
@pytest.mark.parametrize("pattern", _ECP_PATTERNS)
def test_ecp_local_parts_all_pairs(pattern):
    """ECP local parts all pairs."""
    ham = rs_ham(pattern)
    cp = ham.coulomb_potential_data
    r_up, r_dn = _electrons(ham, 3)
    vr = jqmc_rs.compute_ecp_local_parts_all_pairs(cp, r_up, r_dn)
    vj = float(compute_ecp_local_parts_all_pairs(cp, r_up, r_dn))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ECP, level="strict")
    assert np.isclose(vr, vj, rtol=rtol, atol=atol), (vr, vj)


@pytest.mark.light
@pytest.mark.parametrize("pattern", _ECP_PATTERNS)
def test_ecp_nonlocal_parts_all_pairs(pattern):
    """ECP nonlocal parts all pairs."""
    ham = rs_ham(pattern)
    cp = ham.coulomb_potential_data
    wf = ham.wavefunction_data
    gem = wf.geminal_data
    r_up, r_dn = _electrons(ham, 4, scale=0.5)
    RT = np.eye(3)
    *_, sumV = compute_ecp_non_local_parts_all_pairs(cp, wf, r_up, r_dn, RT, Nv=6, flag_determinant_only=False)
    w = np.array(octahedron_sym_mesh_Nv6.weights, float)
    gp = np.array(octahedron_sym_mesh_Nv6.grid_points, float) @ RT
    vr = jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gem, wf.jastrow_data, cp, r_up, r_dn, w, gp, False)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ECP_NONLOCAL, level="strict")
    assert np.isclose(vr, float(sumV), rtol=rtol, atol=atol), (vr, float(sumV))


# Non-default quadrature spheres (mesh comes from the
# caller, so the Rust kernel is mesh-agnostic -- verify one case per family).
@pytest.mark.light
@pytest.mark.parametrize("nv", [4, 12, 18])
@pytest.mark.parametrize("pattern", _ECP_PATTERNS[:2])
def test_ecp_nonlocal_parts_all_pairs_other_meshes(pattern, nv):
    """ECP nonlocal parts all pairs other meshes."""
    mesh = {4: tetrahedron_sym_mesh_Nv4, 12: icosahedron_sym_mesh_Nv12, 18: octahedron_sym_mesh_Nv18}[nv]
    ham = rs_ham(pattern)
    cp = ham.coulomb_potential_data
    wf = ham.wavefunction_data
    gem = wf.geminal_data
    r_up, r_dn = _electrons(ham, 4, scale=0.5)
    RT = np.eye(3)
    *_, sumV = compute_ecp_non_local_parts_all_pairs(cp, wf, r_up, r_dn, RT, Nv=nv, flag_determinant_only=False)
    w = np.array(mesh.weights, float)
    gp = np.array(mesh.grid_points, float) @ RT
    vr = jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gem, wf.jastrow_data, cp, r_up, r_dn, w, gp, False)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ECP_NONLOCAL, level="strict")
    assert np.isclose(vr, float(sumV), rtol=rtol, atol=atol), (vr, float(sumV))


# Full molecular local energy e_L = T + V (bare + ECP local + nonlocal NN=1).
_EL_PATTERNS = _ECP_PATTERNS + ["mol_sphe_sd_jnone"]


@pytest.mark.light
@pytest.mark.parametrize("pattern", _EL_PATTERNS)
def test_local_energy(pattern):
    """Local energy."""
    ham = rs_ham(pattern)
    r_up, r_dn = _electrons(ham, 5, scale=0.5)
    RT = np.eye(3)
    eL_jqmc = float(compute_local_energy(ham, r_up, r_dn, RT))
    w = np.array(icosahedron_sym_mesh_Nv12.weights, float)
    gp = np.array(icosahedron_sym_mesh_Nv12.grid_points, float) @ RT
    eL_rust = jqmc_rs.compute_local_energy(ham, r_up, r_dn, w, gp)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_LOCAL_ENERGY, level="strict")
    assert np.isclose(eL_rust, eL_jqmc, rtol=rtol, atol=atol), (eL_rust, eL_jqmc)


def _attach_random_j3(ham, seed=0, scale=0.15):
    """Attach an ACTIVE three-body Jastrow on the geminal's own AO leaf.

    The committed fixtures carry no j3 block for the ECP+sphe/cart patterns, so a
    random j_matrix is built from scratch on the same AO leaf the geminal already uses.
    """
    wf = ham.wavefunction_data
    orb = wf.geminal_data.orb_data_up_spin
    aos_data = getattr(orb, "aos_data", orb)  # unwrap MOs_data if MO-rep
    n = aos_data.num_ao
    jm = np.random.default_rng(seed).normal(scale=scale, size=(n, n + 1))
    # Production invariant: the J3 coupling block is symmetric (bosonic Jastrow);
    # Jastrow_three_body_data.sanity_check RAISES otherwise, and several jqmc
    # streaming/ratio kernels exploit it (j3_mat.T @ x == j3_mat @ x). A .replace()
    # bypasses sanity_check, so symmetrize here or we test an invalid regime.
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)
    j3d = Jastrow_three_body_data(orb_data=aos_data, j_matrix=jm)
    jd2 = wf.jastrow_data.replace(jastrow_three_body_data=j3d)
    return ham.replace(wavefunction_data=wf.replace(jastrow_data=jd2))


# Full e_L with three-body Jastrow (ECP + J1+J2+J3), on ECP cart/sphe patterns
# with a freshly-attached active J3 (see _attach_random_j3).
_J123ECP_PATTERNS = ["mol_ecp_cart_j12", "mol_ecp_sphe_j12"]


@pytest.mark.light
@pytest.mark.parametrize("pattern", _J123ECP_PATTERNS)
def test_local_energy_with_j3(pattern):
    """Local energy with j3."""
    ham = rs_ham(pattern)
    ham2 = _attach_random_j3(ham, seed=0)
    r_up, r_dn = _electrons(ham2, 5, scale=0.5)
    RT = np.eye(3)
    eL_jqmc = float(compute_local_energy(ham2, r_up, r_dn, RT))
    w = np.array(icosahedron_sym_mesh_Nv12.weights, float)
    gp = np.array(icosahedron_sym_mesh_Nv12.grid_points, float) @ RT
    eL_rust = jqmc_rs.compute_local_energy(ham2, r_up, r_dn, w, gp)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_LOCAL_ENERGY, level="strict")
    assert np.isclose(eL_rust, eL_jqmc, rtol=rtol, atol=atol), (eL_rust, eL_jqmc)


# =========================================================================== #
# ECP angular channel ``l >= 5``.
#
# THE GAP. The non-local ECP energy weights each quadrature point by
# ``(2l+1) P_l(cos theta)``. The two implementations of ``P_l`` are NOT the same
# algorithm:
#
#   * XLA  -- ``jqmc/coulomb_potential.py::_legendre_tablated``: a
#     ``jnp.select`` over CLOSED-FORM polynomials TABULATED for n = 0..6, NaN
#     outside that range;
#   * Rust -- ``jqmc/rs/src/coulomb_potential.rs::legendre``: the three-term
#     Bonnet RECURRENCE, closed form only for l = 0, 1.
#
# A tabulation typo, or a recurrence that drifts, can only show up on the
# channel that uses it. The MAXIMUM angular momentum over the ENTIRE committed
# fixture inventory is l = 3 (``CuBr_ecp_ccpvtz_cart.h5``; every other ECP
# fixture stops at l = 1 or 2), so the l = 5 and l = 6 arms of the tabulation --
# the two highest the codebase claims to support -- were compared against
# nothing at all.
#
# THE FIXTURE. Built programmatically from the committed ``water_ccecp_ccpvqz``
# ham: the oxygen's local channel is relabelled to l = 7 (a pure relabelling --
# the local radial function is untouched, so the LOCAL part of the potential is
# numerically identical to the committed fixture's) and non-local channels
# l = 1..6 are added, each reusing the oxygen's OWN committed l = 0 non-local
# radial (same exponent and power) with a distinct non-zero coefficient. This
# keeps the whole radial side inside committed data and puts the ANGULAR side --
# the only thing this row is about -- under test up to the highest supported l.
# ``max_ang_mom_plus_1`` must move with it: both implementations classify a term
# as non-local via ``ang_moms[i] != max_ang_mom_plus_1[atom]``, and the XLA
# reduction runs over ``arange(_global_max_ang_mom_plus_1)``, so the relabelling
# is what makes l = 5 and l = 6 reachable at all rather than silently dropped.
# =========================================================================== #
import os  # noqa: E402

from jqmc._setting import MCMC_MIN_STEPS  # noqa: E402
from jqmc.coulomb_potential import Coulomb_potential_data  # noqa: E402

from tests.test_rs_kernel._route_spy import assert_native_ran, install_ctx_spy  # noqa: E402

#: Local channel of the modified oxygen. ``_legendre_tablated`` covers n = 0..6,
#: so the non-local channels this exposes (l = 0..6) are exactly the full range
#: the XLA tabulation claims to support -- l = 7 itself is never evaluated as an
#: angular channel (it only names the local part).
_HIGH_L_LOCAL = 7
#: The non-local channels added on top of the fixture's own committed l = 0 one.
_HIGH_L_NEW_LS = (1, 2, 3, 4, 5, 6)
#: The channels this row exists for.
_HIGH_LS = (5, 6)
_HIGH_L_BASE_PATTERN = "mol_ecp_sphe_j12"


def _high_l_ham(zero_high_l: bool = False):
    """The committed water ccECP ham with non-local angular channels up to
    ``l = 6`` on the oxygen. ``zero_high_l`` returns the otherwise-identical ham
    with the ``l >= 5`` COEFFICIENTS set to zero -- the discriminator twin.
    """
    ham = rs_ham(_HIGH_L_BASE_PATTERN)
    cd = ham.coulomb_potential_data
    nucleus = [int(x) for x in cd.nucleus_index]
    ang = [int(x) for x in cd.ang_moms]
    exponents = [float(x) for x in cd.exponents]
    coefficients = [float(x) for x in cd.coefficients]
    powers = [int(x) for x in cd.powers]
    l_max_plus_1 = [int(x) for x in cd.max_ang_mom_plus_1]

    old_local_l = l_max_plus_1[0]
    for i in range(len(ang)):
        if nucleus[i] == 0 and ang[i] == old_local_l:
            ang[i] = _HIGH_L_LOCAL  # relabel the LOCAL channel; radial untouched
    l_max_plus_1[0] = _HIGH_L_LOCAL

    # The oxygen's own committed non-local radial (exponent + power), reused for
    # every synthetic channel so no new radial data enters the fixture.
    src = next(i for i in range(len(ang)) if nucleus[i] == 0 and ang[i] == 0)
    for l in _HIGH_L_NEW_LS:
        coeff = coefficients[src] / (l + 1.0)
        if zero_high_l and l in _HIGH_LS:
            coeff = 0.0
        nucleus.append(0)
        ang.append(l)
        exponents.append(exponents[src])
        coefficients.append(coeff)
        powers.append(powers[src])

    cd2 = Coulomb_potential_data.build(
        structure_data=cd.structure_data,
        ecp_flag=True,
        z_cores=tuple(cd.z_cores),
        max_ang_mom_plus_1=tuple(l_max_plus_1),
        num_ecps=len(ang),
        ang_moms=tuple(ang),
        nucleus_index=tuple(nucleus),
        exponents=tuple(exponents),
        coefficients=tuple(coefficients),
        powers=tuple(powers),
    )
    return ham.replace(coulomb_potential_data=cd2)


def _assert_high_l_ham_live(ham):
    """LIVENESS: the ``l >= 5`` channels exist ON THE NON-LOCAL SIDE and
    carry NON-ZERO coefficients.

    Both facts matter separately: a term whose ``ang_moms`` landed on the local
    index would never reach ``P_l`` at all, and a zero coefficient would make the
    channel arithmetically absent while still appearing in the table.
    """
    cd = ham.coulomb_potential_data
    ls = np.asarray(cd._ang_mom_non_local_part, dtype=int)
    coeffs = np.asarray(cd._coefficients_non_local_part, dtype=float)
    for l in _HIGH_LS:
        sel = ls == l
        assert sel.any(), (f"no NON-LOCAL ECP term with l = {l}", ls.tolist())
        assert np.any(np.abs(coeffs[sel]) > 0.0), (f"the l = {l} non-local coefficient is zero", coeffs[sel].tolist())
    assert int(cd._global_max_ang_mom_plus_1) > max(_HIGH_LS), (
        "the global l reduction stops below the high-l channels, so they are dropped before P_l",
        int(cd._global_max_ang_mom_plus_1),
    )


@pytest.mark.light
@pytest.mark.parametrize("seed", [3, 4, 5])
def test_ecp_nonlocal_angular_channel_l_ge_5(seed):
    """ROW 8 comparison: the non-local ECP energy with active ``l = 5`` and
    ``l = 6`` channels, Rust (Bonnet recurrence) vs XLA (tabulated polynomials).

    Use determinant-only ratios here so this angular-kernel contract is not
    polluted by the independent fp32 Jastrow-value reduction; the sibling
    local-energy and production-route rows retain the full Jastrow path.
    """
    ham = _high_l_ham()
    _assert_high_l_ham_live(ham)
    cp = ham.coulomb_potential_data
    wf = ham.wavefunction_data
    gem = wf.geminal_data
    r_up, r_dn = _electrons(ham, seed, scale=0.5)
    RT = np.eye(3)
    *_, sumV = compute_ecp_non_local_parts_all_pairs(cp, wf, r_up, r_dn, RT, Nv=6, flag_determinant_only=True)
    w = np.asarray(octahedron_sym_mesh_Nv6.weights, dtype=float)
    gp = np.asarray(octahedron_sym_mesh_Nv6.grid_points, dtype=float) @ RT
    vr = jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gem, wf.jastrow_data, cp, r_up, r_dn, w, gp, True)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ECP_NONLOCAL, level="strict")
    assert np.isclose(vr, float(sumV), rtol=rtol, atol=atol), (seed, vr, float(sumV))


@pytest.mark.light
@pytest.mark.parametrize("seed", [3, 4, 5])
def test_local_energy_with_angular_channel_l_ge_5(seed):
    """ROW 8 comparison, assembled: the full molecular ``e_L`` carrying the high-l
    non-local channels."""
    ham = _high_l_ham()
    _assert_high_l_ham_live(ham)
    r_up, r_dn = _electrons(ham, seed, scale=0.5)
    RT = np.eye(3)
    eL_jqmc = float(compute_local_energy(ham, r_up, r_dn, RT))
    w = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=float)
    gp = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=float) @ RT
    eL_rust = jqmc_rs.compute_local_energy(ham, r_up, r_dn, w, gp)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_LOCAL_ENERGY, level="strict")
    assert np.isclose(eL_rust, eL_jqmc, rtol=rtol, atol=atol), (seed, eL_rust, eL_jqmc)


@pytest.mark.light
def test_high_l_channels_change_the_rust_result():
    """LIVENESS DISCRIMINATOR. The two hams differ ONLY in the ``l >= 5``
    coefficients. If the high-l arms were unreachable -- dropped by the non-local
    classification, truncated by the ``arange`` reduction, or returning zero from
    a mis-indexed ``P_l`` -- both would give the SAME non-local energy and the
    rows above would be a re-test of l <= 4 while still passing. Required on BOTH
    sides, so neither implementation can be the one that silently drops them.
    """
    live = _high_l_ham()
    dead = _high_l_ham(zero_high_l=True)
    _assert_high_l_ham_live(live)
    wf = live.wavefunction_data
    r_up, r_dn = _electrons(live, 3, scale=0.5)
    RT = np.eye(3)
    w = np.asarray(octahedron_sym_mesh_Nv6.weights, dtype=float)
    gp = np.asarray(octahedron_sym_mesh_Nv6.grid_points, dtype=float) @ RT
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ECP_NONLOCAL)

    v_live = float(
        jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(
            wf.geminal_data, wf.jastrow_data, live.coulomb_potential_data, r_up, r_dn, w, gp, False
        )
    )
    v_dead = float(
        jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(
            wf.geminal_data, wf.jastrow_data, dead.coulomb_potential_data, r_up, r_dn, w, gp, False
        )
    )
    assert not np.isclose(v_live, v_dead, rtol=rtol, atol=atol), (
        "Rust gives the SAME non-local ECP energy with and without the l >= 5 coefficients -- "
        "the high-l angular channels never reach the Legendre evaluation, so the l >= 5 rows are vacuous",
        v_live,
        v_dead,
    )
    *_, s_live = compute_ecp_non_local_parts_all_pairs(
        live.coulomb_potential_data, wf, r_up, r_dn, RT, Nv=6, flag_determinant_only=False
    )
    *_, s_dead = compute_ecp_non_local_parts_all_pairs(
        dead.coulomb_potential_data, wf, r_up, r_dn, RT, Nv=6, flag_determinant_only=False
    )
    assert not np.isclose(float(s_live), float(s_dead), rtol=rtol, atol=atol), (
        "XLA gives the SAME non-local ECP energy with and without the l >= 5 coefficients",
        float(s_live),
        float(s_dead),
    )


@pytest.mark.heavy
def test_route_parity_angular_channel_l_ge_5(monkeypatch):
    """ROW 8 route parity: a PRODUCTION VMC run on the high-l Hamiltonian reaches
    the NATIVE fused energy chunk -- i.e. the ``l >= 5`` non-local channels are
    evaluated by the Rust ECP kernel on the route production actually takes, not
    only by the free function the rows above call."""
    from jqmc.jqmc_mcmc import MCMC

    ham = _high_l_ham()
    _assert_high_l_ham_live(ham)
    spy = install_ctx_spy(monkeypatch)
    os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    try:
        m = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=34356,
            epsilon_AS=1.0e-6,
            num_walkers=2,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )
        m.run(num_mcmc_steps=MCMC_MIN_STEPS)
    finally:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    ctx = "ecp/l_ge_5"
    assert_native_ran(spy, "run_energy_chunk_batch", context=ctx)
    assert bool(spy.build_arg("complex")) is False, f"{ctx}: build_obs_ctx asked for the complex family"
    assert not np.any(np.isnan(np.asarray(m.e_L))), f"{ctx}: the native high-l ECP run produced NaN e_L"


# --------------------------------------------------------------------------- #
# Mixed ECP / all-electron molecule: non-local NN projector against an
# independent numpy projection (issues #20 and #21). The system and the
# reference live in tests/_mixed_ecp_ae_reference.py; the XLA-only force
# detector is tests/test_jqmc_kernel/test_jqmc_derivatives.py::
# test_ecp_force_mixed_ecp_all_electron_matches_independent_derivative.
# --------------------------------------------------------------------------- #
from tests._mixed_ecp_ae_reference import (  # noqa: E402
    MIXED_AE_R_UP,
    mixed_ecp_ae_reference,
    mixed_ecp_ae_system,
)


@pytest.mark.light
@pytest.mark.parametrize(
    "ecp_atom,nn",
    [(0, 1), (0, 2), (1, 1)],
    ids=("trailing_ae_nn1", "trailing_ae_nn2", "leading_ae_nn1"),
)
@pytest.mark.parametrize("backend", ["xla", "rust"])
def test_ecp_nonlocal_mixed_ecp_all_electron_matches_independent_projection(backend, ecp_atom, nn):
    """Non-local ECP of a mixed ECP/all-electron molecule equals an independent projection.

    The NN nearest-neighbour projector ranks ECP nuclei only (an all-electron
    nucleus never hosts a projector and never displaces an ECP centre), and the
    per-atom ECP tables cover every atom, so an all-electron atom listed after
    the last ECP atom projects nothing instead of inheriting another atom's ECP.
    trailing_ae: ECP on nucleus 0, all-electron nucleus 1 listed last; leading_ae:
    all-electron nucleus 0, ECP on nucleus 1. In each layout one electron is
    nearer the all-electron nucleus. NN=2 exceeds the number of ECP nuclei, so the
    second slot reaches the all-electron nucleus, whose ECP table row must be empty.
    XLA runs the plain and the inverse-update (production LRDMC/e_L) evaluators;
    Rust runs its non-local kernel with the same NN. The reference is the ideal
    float64 physical value; each backend runs at its configured precision (in the
    mixed mode the XLA ao_eval zone and the native standalone kernel's captured AO
    evaluation are float32, the Coulomb/determinant accumulation float64) and is
    judged at the strict tolerance of the zones on its own path in that mode.
    """
    from jqmc._precision import get_tolerance_min
    from jqmc.coulomb_potential import (
        compute_ecp_non_local_parts_nearest_neighbors,
        compute_ecp_non_local_parts_nearest_neighbors_fast_update,
    )
    from jqmc.determinant import compute_geminal_all_elements

    cp, wf = mixed_ecp_ae_system(ecp_atom)
    gem = wf.geminal_data
    r_up = MIXED_AE_R_UP.copy()
    r_dn = np.empty((0, 3))
    RT = np.eye(3)
    if backend == "rust":
        w = np.array(octahedron_sym_mesh_Nv6.weights, float)
        gp = np.array(octahedron_sym_mesh_Nv6.grid_points, float) @ RT
        values = [float(jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gem, wf.jastrow_data, cp, r_up, r_dn, w, gp, False, nn))]
        reference = mixed_ecp_ae_reference(ecp_atom)
        atol, rtol = get_tolerance_min(("coulomb",), "strict")
    else:
        *_, sum_plain = compute_ecp_non_local_parts_nearest_neighbors(cp, wf, r_up, r_dn, RT, NN=nn, Nv=6)
        a_inv = np.linalg.inv(np.asarray(compute_geminal_all_elements(gem, r_up, r_dn), dtype=np.float64))
        *_, sum_fast = compute_ecp_non_local_parts_nearest_neighbors_fast_update(cp, wf, r_up, r_dn, RT, a_inv, NN=nn, Nv=6)
        values = [float(np.real(sum_plain)), float(np.real(sum_fast))]
        reference = mixed_ecp_ae_reference(ecp_atom)
        atol, rtol = get_tolerance_min(("coulomb", "ao_eval", "det_eval", "det_ratio"), "strict")
    for value in values:
        assert np.isfinite(value)
        np.testing.assert_allclose(value, reference, atol=atol, rtol=rtol)


# =========================================================================== #
# Periodic minimum image of the ECP kernels (issues 19, 22, 23).
#
# Every expected value comes from tests/_ecp_min_image_reference.py, which finds
# the nearest image of each nucleus by trying every lattice translation in a
# +-6 cell box, with no Wigner-Seitz reduction. The electrons sit several cells
# away from the home cell, so a kernel that drops the minimum image (or never
# receives the cell) uses a far image and misses the reference by O(0.1).
# =========================================================================== #
from tests._ecp_min_image_reference import (  # noqa: E402
    CELLS,
    ecp_local_reference_per_electron,
    far_electrons,
    local_ecp_on,
    min_image_distances,
    single_nucleus_ecp_on,
    synthetic_h2_ecp,
)

_PBC_CELLS = list(CELLS)


def _assert_raw_distance_misses(cp, r_all, expected, atol, rtol):
    """Non-vacuity: the energy from the raw (non-imaged) distance must miss the reference."""
    sd = cp.structure_data
    raw = np.linalg.norm(np.asarray(sd.positions)[None, :, :] - r_all[:, None, :], axis=-1)
    v_raw = 0.0
    for i_atom, l, c, a, p in zip(cp.nucleus_index, cp.ang_moms, cp.coefficients, cp.exponents, cp.powers):
        if l == cp.max_ang_mom_plus_1[i_atom]:
            v_raw += np.sum(c * raw[:, i_atom] ** (p - 2) * np.exp(-a * raw[:, i_atom] ** 2))
    assert not np.isclose(v_raw, expected, atol=atol, rtol=rtol), (v_raw, expected)


@pytest.mark.light
@pytest.mark.parametrize("cellname", _PBC_CELLS)
@pytest.mark.parametrize("backend", ["rust", "xla"])
def test_ecp_local_parts_all_pairs_pbc_brute_force_images(backend, cellname):
    """Periodic local ECP energy equals the brute-force minimum-image sum.

    Rust row: the exported ``jqmc_rs.compute_ecp_local_parts_all_pairs`` must read
    the cell from ``structure_data`` and take the minimum image, like every
    production Rust caller. XLA row: the public ``compute_ecp_local_parts_all_pairs``.
    """
    import jax.numpy as jnp

    from jqmc._precision import get_tolerance

    cp = synthetic_h2_ecp(cellname)
    r_all = far_electrons(cellname, 5)
    r_up, r_dn = r_all[:3], r_all[3:]
    expected = float(np.sum(ecp_local_reference_per_electron(cp, r_all)))
    atol, rtol = get_tolerance("coulomb", "strict")
    _assert_raw_distance_misses(cp, r_all, expected, atol, rtol)
    if backend == "rust":
        actual = float(jqmc_rs.compute_ecp_local_parts_all_pairs(cp, r_up, r_dn))
    else:
        actual = float(compute_ecp_local_parts_all_pairs(cp, jnp.asarray(r_up), jnp.asarray(r_dn)))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=f"{backend} {cellname}")


#: Lattice translations (in cell units) of the four H4 electrons placed at nucleus 0.
_NN_SHIFTS = np.array([[0, 0, 1], [1, 1, 0], [0, 0, 2], [1, 1, 1]], dtype=np.float64)


def _h4_wavefunction(cellname):
    """Periodic all-electron H4 wavefunction on an orthogonal or a skewed cell."""
    if cellname == "skew":
        from tests._gfmc_twin_fixtures import skewed_h4_ham

        ham = skewed_h4_ham(twist=False)
        return ham.structure_data, ham.wavefunction_data
    from jqmc.jastrow_factor import Jastrow_data
    from jqmc.trexio_wrapper import read_trexio_file
    from jqmc.wavefunction import Wavefunction_data

    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H4_pbc_ae_cart.h5")
    sd, *_, gem, _cp = read_trexio_file(path, store_tuple=True)
    jd = Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None)
    return sd, Wavefunction_data(jastrow_data=jd, geminal_data=gem)


#: Lattice translations (in cell units) of the four H4 electrons of the per-electron local test.
_LOCAL_ONE_SHIFTS = np.array([[2, -1, 1], [-2, 3, 0], [1, 1, -3], [3, -2, 2]], dtype=np.float64)


@pytest.mark.light
@pytest.mark.parametrize("cellname", ["cubic", "skew"])
def test_ecp_local_per_electron_pbc_brute_force_images(cellname):
    """Rust per-electron local ECP rows take the brute-force minimum image.

    ``compute_ecp_local_one`` feeds the GFMC stateful chain and the incremental
    projection-interval rows. The stateful LRDMC local energy
    (``jqmc_rs._gfmc_lrdmc_ecp`` with ``fast=True``) sums these rows for the
    local channel, and the local channel enters e_L additively. Two otherwise
    identical potentials, one with its local coefficients set to zero, so differ
    in e_L by exactly the local ECP energy, which must equal the brute-force
    minimum-image sum for electrons several cells away from the home cell.
    """
    from jqmc._precision import get_tolerance
    from jqmc.coulomb_potential import build_ewald_data

    sd, wf = _h4_wavefunction(cellname)

    def periodic_ecp(local_scale):
        cp = local_ecp_on(sd, local_scale=local_scale)
        return cp.replace(ewald_data=build_ewald_data(sd, cp))

    cp, cp_no_local = periodic_ecp(1.0), periodic_ecp(0.0)
    cell = np.asarray(sd.cell, dtype=np.float64)
    pos = np.asarray(sd.positions, dtype=np.float64)
    # Electron e sits within ~0.3 bohr of an image of nucleus e, several cells away.
    rng = np.random.default_rng(7)
    r_all = pos[np.arange(4) % len(pos)] + rng.normal(scale=0.3, size=(4, 3)) + _LOCAL_ONE_SHIFTS @ cell
    r_up, r_dn = r_all[:2], r_all[2:]
    expected = float(np.sum(ecp_local_reference_per_electron(cp, r_all)))
    atol, rtol = get_tolerance("coulomb", "strict")
    assert abs(expected) > atol, f"vacuous: local ECP energy {expected} is ~0"
    _assert_raw_distance_misses(cp, r_all, expected, atol, rtol)
    w = np.array(octahedron_sym_mesh_Nv6.weights, float)
    gp = np.array(octahedron_sym_mesh_Nv6.grid_points, float)
    gem, jd = wf.geminal_data, wf.jastrow_data
    alat = 0.3

    def e_l(potential):
        return float(jqmc_rs._gfmc_lrdmc_ecp(gem, jd, potential, r_up, r_dn, alat, np.eye(3), w, gp, 1, False, True)[0])

    actual = e_l(cp) - e_l(cp_no_local)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=cellname)


@pytest.mark.light
@pytest.mark.parametrize("cellname", ["cubic", "skew"])
@pytest.mark.parametrize("backend", ["rust", "xla"])
def test_ecp_nonlocal_nearest_nucleus_pbc_brute_force_images(backend, cellname):
    """The NN=1 non-local ECP filter picks the minimum-image nearest nucleus.

    Only nucleus 0 carries non-zero ECP terms, and every electron sits within
    0.25 bohr of an image of nucleus 0 but several cells away from the home
    cell. The brute-force image search confirms nucleus 0 is each electron's
    nearest nucleus, so the NN=1 energy must equal the all-pairs energy. A
    filter that ranks nuclei by the raw distance picks another (zero-term)
    nucleus for some electron and drops its contribution. All-pairs is
    ``nn=0`` on Rust and ``NN=natom`` on XLA.
    """
    import jax.numpy as jnp

    from jqmc._precision import get_tolerance_min

    sd, wf = _h4_wavefunction(cellname)
    cp = single_nucleus_ecp_on(sd, active=0)
    cell = np.asarray(sd.cell, dtype=np.float64)
    pos = np.asarray(sd.positions, dtype=np.float64)
    rng = np.random.default_rng(11)
    r_all = pos[0] + rng.normal(scale=0.25, size=(4, 3)) + _NN_SHIFTS @ cell
    r_up, r_dn = r_all[:2], r_all[2:]
    # Brute-force reference: nucleus 0 is the nearest image for every electron ...
    assert np.all(np.argmin(min_image_distances(pos, cell, r_all), axis=1) == 0)
    # ... while the raw distance ranks another nucleus first for some electron.
    raw = np.linalg.norm(pos[None, :, :] - r_all[:, None, :], axis=-1)
    assert np.any(np.argmin(raw, axis=1) != 0), "vacuous geometry: raw and minimum-image nearest agree"
    atol, rtol = get_tolerance_min(_ZONES_ECP_NONLOCAL, level="strict")
    RT = np.eye(3)
    if backend == "rust":
        w = np.array(octahedron_sym_mesh_Nv6.weights, float)
        gp = np.array(octahedron_sym_mesh_Nv6.grid_points, float)
        gem, jd = wf.geminal_data, wf.jastrow_data
        nn1 = float(jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gem, jd, cp, r_up, r_dn, w, gp, False, 1))
        all_pairs = float(jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gem, jd, cp, r_up, r_dn, w, gp, False, 0))
    else:
        from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors

        def nn_energy(nn):
            return float(
                compute_ecp_non_local_parts_nearest_neighbors(
                    cp, wf, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(RT), NN=nn, Nv=6
                )[-1]
            )

        # The XLA all-pairs entry rejects PBC; NN = natom keeps every nucleus.
        nn1, all_pairs = nn_energy(1), nn_energy(len(pos))
    assert abs(all_pairs) > atol, f"vacuous: all-pairs non-local energy {all_pairs} is ~0"
    np.testing.assert_allclose(nn1, all_pairs, atol=atol, rtol=rtol, err_msg=f"{backend} {cellname}")
