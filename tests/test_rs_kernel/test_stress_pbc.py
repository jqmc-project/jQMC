"""Layer-1: PBC d lnPsi / d eps (Pulay stress, Rust stress_pbc.rs) and PBC d T / d
eps / d e_L / d eps (Rust stress_pbc.rs) vs the analytic-derivative module
``jqmc.derivatives.pbc_stress`` (production code with the current strain semantics, per
``test_gfmc_complex_force_stress.py``'s established "``jqmc.derivatives`` as ground
truth" pattern) or a direct ``jax.grad(compute_local_energy)`` /
``jax.grad(compute_Jastrow_DTN)`` oracle.

Hamiltonians come from the committed-fixture patterns
(``tests.test_rs_kernel._rs_fixtures.rs_ham``). Oracles:

  * ``test_grad_lnpsi_pbc_strain``:
    ``jqmc.derivatives.pbc_stress.grad_lndet_wrt_strain(gd, r_up, r_dn)`` +
    (if J1/J2 present) ``grad_lnJastrow_wrt_strain(jd, r_up, r_dn)`` (default kwargs
    ``take_real=True, geminal_inv=None``).
  * ``test_grad_T_det_pbc_strain``:
    ``jqmc.derivatives.pbc_stress.grad_T_wrt_strain_orbital(gd, jastrow_data,
    r_up, r_dn)`` (the fixture's own jnone -- all-None-field -- jastrow_data;
    with a zero Jastrow grad/lap the det-side kinetic reverse-pass cotangents
    ``_geminal_reverse`` builds are those of the det-only kinetic stress).
  * ``test_grad_T_pbc_strain_withJ``: the combined orbital + Jastrow kinetic stress
    ``grad_T_wrt_strain_orbital(gd, jd, r_up, r_dn) + grad_T_wrt_strain_
    jastrow(jd, gd, r_up, r_dn)``, i.e. exactly
    ``jqmc.derivatives.pbc_stress.stress_eL_kinetic_pbc(hamiltonian_data, r_up,
    r_dn)``. Its J2 term follows `_apply_strain` and strains the J2 metric cell
    with the physical lattice.
    ``grad_T_wrt_strain_jastrow``'s signature is ``(jd, gd_geminal, r_up, r_dn,
    w=None, geminal_inv=None)`` -- ``jd`` first; ``stress_eL_kinetic_pbc`` already
    gets the argument order right internally, so calling it directly sidesteps any
    transposition risk.
  * ``test_grad_eL_pbc_ecp_strain``, ``test_grad_eL_pbc_dtn_strain_uchi``,
    ``test_grad_V_ecp_local_pbc_strain``, ``test_dtn_value_strain_uchif``:
    self-contained ``jax.grad`` oracles (plus, for the first two, a Rust-side
    sum-of-pieces cross-check using ONLY the public ``jqmc_rs`` strain kernels),
    with the local ``randomize_dtn`` helper.

Direct XLA--Rust comparisons use the dedicated consistency tolerance selected
only by pytest's full/mixed precision mode.

Current jqmc production semantics strain every periodic Jastrow cell and the electron
positions together. The Rust production wrappers use the same J2 value and kinetic
cell-strain derivatives and are compared here directly with
``jqmc.derivatives.pbc_stress`` and ``hamiltonian_data._apply_strain(F)``."""

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
from jqmc._setting import ATOL_RELATIVE_COMPARISON_SCALE_FLOOR  # noqa: E402
from jqmc.coulomb_potential import compute_ecp_local_parts_all_pairs, icosahedron_sym_mesh_Nv12  # noqa: E402
from jqmc.derivatives import _auto  # noqa: E402
from jqmc.derivatives import dtn as dtn_derivatives  # noqa: E402
from jqmc.derivatives import pbc_stress  # noqa: E402
from jqmc.hamiltonians import compute_local_energy  # noqa: E402

import jqmc_rs  # noqa: E402
from jqmc import derivatives  # noqa: E402

from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.with_chdir]


def _cfg(ham, seed=1):
    gem = ham.wavefunction_data.geminal_data
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ru = R[np.arange(nu) % len(R)] + rng.normal(scale=0.7, size=(nu, 3))
    rd = R[np.arange(nd) % len(R)] + rng.normal(scale=0.7, size=(nd, 3))
    return ru, rd


def _randomize_dtn(jastrow_data, seed=0, scale=0.05):
    """Perturb the independent DTN parameters (numpy, no jnp cast): committed jdtn fixtures ship
    all-zero alpha/beta/gamma_indep, making a DTN value-derivative test vacuous
    unless perturbed."""
    d = jastrow_data.jastrow_dtn_data
    if d is None:
        return jastrow_data
    rng = np.random.default_rng(seed)
    upd = {}
    for f in ("alpha_indep", "beta_indep", "gamma_indep"):
        v = np.asarray(getattr(d, f))
        upd[f] = scale * rng.standard_normal(v.shape)
    return jastrow_data.replace(jastrow_dtn_data=d.replace(**upd))


def _randomize_j3(jastrow_data, seed=0, scale=0.05):
    """Random J3 coupling with the block SYMMETRIZED (numpy, no jnp cast): j123ao fixtures ship
    j_matrix == 0, vacuous unless perturbed. The (nao, nao) coupling block must be
    bosonic-symmetric -- the production class asserts it (jastrow_factor.py
    sanity_check) and the native kernels exploit it -- so a raw asymmetric random
    matrix is out of the kernel contract; the one-body column stays free."""
    j3 = jastrow_data.jastrow_three_body_data
    if j3 is None:
        return jastrow_data
    rng = np.random.default_rng(seed)
    Jm = scale * rng.standard_normal(np.asarray(j3.j_matrix).shape)
    nao = Jm.shape[0]
    Jm[:, :nao] = 0.5 * (Jm[:, :nao] + Jm[:, :nao].T)
    return jastrow_data.replace(jastrow_three_body_data=j3.replace(j_matrix=Jm))


def _close(a, b, atol, rtol):
    a, b = np.asarray(a), np.asarray(b)
    d = np.abs(a - b).max()
    return d <= atol or d <= rtol * (np.abs(b).max() + ATOL_RELATIVE_COMPARISON_SCALE_FLOOR)


# --------------------------------------------------------------------------- #
# PBC Pulay stress: d lnPsi / d eps -- oracle = jqmc.derivatives.pbc_stress.
# grad_lndet_wrt_strain + grad_lnJastrow_wrt_strain.
# --------------------------------------------------------------------------- #

_ZONES_LNPSI = ("ao_eval", "ao_grad_lap", "det_eval", "jastrow_eval")


@pytest.mark.heavy
@pytest.mark.parametrize(
    "pattern",
    [
        "pbc_composite_jnone",
        "pbc_composite_j12",
        "pbc_gamma_sphepgto_jnone",
        "pbc_gamma_sphepgto_j12",
    ],
)
def test_grad_lnpsi_pbc_strain(pattern):
    """Grad lnpsi PBC strain."""
    ham = rs_ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    ru, rd = _cfg(ham)
    ana = jqmc_rs.grad_lnPsi_pbc_wrt_strain(gd, jd, ru, rd)
    ref = np.asarray(pbc_stress.grad_lndet_wrt_strain(gd, ru, rd))
    if jd.jastrow_one_body_data is not None or jd.jastrow_two_body_data is not None:
        ref = ref + np.asarray(pbc_stress.grad_lnJastrow_wrt_strain(jd, ru, rd))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_LNPSI, level="medium")
    assert _close(ana, ref, atol, rtol)


# --------------------------------------------------------------------------- #
# PBC kinetic stress: d T_det / d eps (jnone) -- oracle = jqmc.derivatives.pbc_stress.
# grad_T_wrt_strain_orbital (the det-only side; the row's own jnone jastrow_data
# contributes zero to the kinetic reverse-pass cotangents, so this is the
# det-only kinetic stress exactly).
# --------------------------------------------------------------------------- #

_ZONES_TDET = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic")


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["pbc_composite_jnone", "pbc_gamma_sphepgto_jnone"])
def test_grad_T_det_pbc_strain(pattern):
    """PBC kinetic stress d T_det/d eps (jnone): value+lap field-strain VJP + PW |kappa|^2 lap."""
    ham = rs_ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    ru, rd = _cfg(ham)
    ana = jqmc_rs.grad_T_det_pbc_wrt_strain(gd, ru, rd)
    ref = np.asarray(pbc_stress.grad_T_wrt_strain_orbital(gd, jd, ru, rd))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_TDET, level="medium")
    assert _close(ana, ref, atol, rtol)


# --------------------------------------------------------------------------- #
# Full periodic ECP e_L stress: self-contained jax.grad(compute_local_energy)
# oracle PLUS a Rust-side sum-of-pieces check built
# entirely from public jqmc_rs strain kernels -- no jqmc.derivatives needed.
# --------------------------------------------------------------------------- #

_ZONES_ECP_ELPBC = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb")


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["pbc_gamma_sphepgto_jnone", "pbc_composite_jnone"])
def test_grad_eL_pbc_ecp_strain(pattern):
    """Full periodic ECP local-energy stress d e_L/d eps = d T_det/d eps + d E_ewald/d
    eps + d V_ecp_local/d eps + d V_ecp_nonlocal/d eps vs jax.grad(compute_local_energy
    . strain), RT-fixed quadrature. The ECP non-local stress uses the per-mesh PBC
    Pulay stress + mesh-motion correction (derivation verified vs jax.grad)."""
    ham = rs_ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    cd = ham.coulomb_potential_data
    ed = cd.ewald_data
    sd = ham.structure_data
    w = np.asarray(icosahedron_sym_mesh_Nv12.weights, float)
    gp = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, float)
    eta = float(ed.eta)
    ri = np.asarray(ed.real_image_list, float)
    gv = np.asarray(ed.g_vectors, float)
    gf = np.asarray(ed.g_factors, float)
    cell = np.asarray(sd.cell, float)
    nmax = int(sd._n_max)
    z = np.asarray(cd._effective_charges, float)
    rion = np.asarray(ed.structure_data.positions, float)
    ru, rd = _cfg(ham)

    def fwd(eps):
        F = jnp.eye(3) + eps
        return compute_local_energy(ham._apply_strain(F), ru @ F.T, rd @ F.T, jnp.eye(3))

    full = np.asarray(jax.grad(fwd)(jnp.zeros((3, 3))))
    total = (
        np.asarray(jqmc_rs.grad_T_det_pbc_wrt_strain(gd, ru, rd))
        + np.asarray(jqmc_rs.grad_Eee_wrt_strain(ru, rd, eta, ri, gv, gf, cell, nmax))
        + np.asarray(jqmc_rs.grad_Eei_wrt_strain(ru, rd, eta, ri, gv, gf, z, rion, cell, nmax))
        + np.asarray(jqmc_rs.grad_Eii_wrt_strain(eta, ri, gv, gf, z, rion, cell))
        + np.asarray(jqmc_rs.grad_V_ecp_local_wrt_strain(cd, ru, rd))
        + np.asarray(jqmc_rs.grad_V_ecp_nonlocal_wrt_strain(gd, cd, ru, rd, w, gp))
    )
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ECP_ELPBC, level="medium")
    assert _close(total, full, atol, rtol)


@pytest.mark.heavy
def test_grad_eL_pbc_dtn_strain_uchi():
    """Full periodic DTN (u+chi) local-energy stress d e_L/d eps = det-kinetic + DTN-
    kinetic(u+chi) + Ewald + ECP-local + ECP-nonlocal(with DTN ratio) vs jax.grad(
    compute_local_energy . strain). f-channel OFF: this row covers the u and chi channels."""
    ham = rs_ham("pbc_gamma_sphepgto_jdtn")
    gd = ham.wavefunction_data.geminal_data
    jd0 = _randomize_dtn(ham.wavefunction_data.jastrow_data, seed=5)
    jd = jd0.replace(jastrow_dtn_data=jd0.jastrow_dtn_data.replace(use_f=False))
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))
    cd = ham.coulomb_potential_data
    ed = cd.ewald_data
    sd = ham.structure_data
    w = np.asarray(icosahedron_sym_mesh_Nv12.weights, float)
    gp = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, float)
    eta = float(ed.eta)
    ri = np.asarray(ed.real_image_list, float)
    gv = np.asarray(ed.g_vectors, float)
    gf = np.asarray(ed.g_factors, float)
    cell = np.asarray(sd.cell, float)
    nmax = int(sd._n_max)
    z = np.asarray(cd._effective_charges, float)
    rion = np.asarray(ed.structure_data.positions, float)
    ru, rd = _cfg(ham)

    def fwd(eps):
        F = jnp.eye(3) + eps
        return compute_local_energy(ham._apply_strain(F), ru @ F.T, rd @ F.T, jnp.eye(3))

    fixed_cutoff = np.asarray(jax.grad(fwd)(jnp.zeros((3, 3))))
    analytic = np.asarray(
        derivatives.compute_grad_e_L(
            ham,
            ru,
            rd,
            jnp.eye(3),
            {"opt_eps": True},
            analytic=True,
        )["eps"]
    )
    fixed_analytic = np.asarray(_auto.compute_grad_e_L(ham, ru, rd, jnp.eye(3), wrt="eps", mask=None))
    # ``analytic - fixed_analytic`` isolates the XLA comoving-cutoff policy
    # derivative.  Add it to the direct strained-forward oracle,
    # which independently pins every fixed-cutoff term in this component sum.
    full = fixed_cutoff + (analytic - fixed_analytic)
    total = (
        np.asarray(jqmc_rs.grad_T_pbc_wrt_strain(gd, jd, ru, rd))
        + np.asarray(jqmc_rs.grad_T_dtn_pbc_wrt_strain(gd, jd, ru, rd))
        + np.asarray(jqmc_rs.grad_Eee_wrt_strain(ru, rd, eta, ri, gv, gf, cell, nmax))
        + np.asarray(jqmc_rs.grad_Eei_wrt_strain(ru, rd, eta, ri, gv, gf, z, rion, cell, nmax))
        + np.asarray(jqmc_rs.grad_Eii_wrt_strain(eta, ri, gv, gf, z, rion, cell))
        + np.asarray(jqmc_rs.grad_V_ecp_local_wrt_strain(cd, ru, rd))
        + np.asarray(jqmc_rs.grad_V_ecp_nonlocal_wrt_strain(gd, cd, ru, rd, w, gp, jd))
    )
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ECP_ELPBC + ("jastrow_eval", "jastrow_grad_lap"), level="medium")
    assert _close(total, full, atol, rtol)


# --------------------------------------------------------------------------- #
# PBC ECP local-channel stress: fully self-contained jax.grad oracle (no
# external module).
# --------------------------------------------------------------------------- #

_ZONES_ECP_LOCAL = ("coulomb",)


@pytest.mark.light
def test_grad_V_ecp_local_pbc_strain():
    """PBC ECP local-channel stress d V_ecp_local/d eps = sum V_l'(r_eI).d(x)d/r_eI
    (min-image). Verified vs isolated jax.grad of compute_ecp_local_parts_all_pairs."""
    ham = rs_ham("pbc_gamma_sphepgto_jnone")
    cd = ham.coulomb_potential_data
    ru, rd = _cfg(ham)

    def floc(eps):
        F = jnp.eye(3) + eps
        return compute_ecp_local_parts_all_pairs(ham._apply_strain(F).coulomb_potential_data, ru @ F.T, rd @ F.T)

    ref = np.asarray(jax.grad(floc)(jnp.zeros((3, 3))))
    got = np.asarray(jqmc_rs.grad_V_ecp_local_wrt_strain(cd, ru, rd))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ECP_LOCAL, level="medium")
    assert _close(got, ref, atol, rtol)


# --------------------------------------------------------------------------- #
# PBC DTN VALUE strain (u+chi+f): fully self-contained jax.grad(compute_Jastrow_
# DTN) oracle.
# --------------------------------------------------------------------------- #

_ZONES_DTN_VALUE = ("jastrow_eval",)


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dtn_value_strain_uchif(seed):
    """PBC DTN VALUE strain for u+chi+f: Rust equals the XLA analytic route.

    The latter includes both fixed-cutoff coordinate geometry and the isotropic
    comoving-cutoff policy derivative.  Its independent finite-difference
    oracle lives in ``TestDTNCutoffPolicy``.
    """
    ham = rs_ham("pbc_gamma_sphepgto_jdtn")
    jd = _randomize_dtn(ham.wavefunction_data.jastrow_data, seed=5)
    d0 = jd.jastrow_dtn_data
    assert bool(d0.use_f)  # fixture has the f-channel active
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[:1] + rng.normal(scale=0.4, size=(1, 3))
    rd = R[1:2] + rng.normal(scale=0.4, size=(1, 3))

    ref = np.asarray(dtn_derivatives.grad_dtn_wrt_strain(d0, ru, rd))
    got = np.asarray(jqmc_rs._dtn_value_strain(jd, ru, rd))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_DTN_VALUE, level="medium")
    assert _close(got, ref, atol, rtol * max(1.0, np.abs(ref).max()))


# --------------------------------------------------------------------------- #
# PBC kinetic stress with Jastrow: d T/d eps -- oracle = jqmc.derivatives.pbc_stress.
# stress_eL_kinetic_pbc == grad_T_wrt_strain_orbital + grad_T_wrt_strain_jastrow
# (the sum of the orbital-side and Jastrow-side kinetic stress).
# --------------------------------------------------------------------------- #

_ZONES_TWITHJ = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "jastrow_eval", "jastrow_grad_lap")


@pytest.mark.heavy
@pytest.mark.parametrize(
    "pattern",
    [
        "pbc_gamma_sphepgto_j12",
        "pbc_composite_j12",
        "pbc_gamma_sphepgto_j123",
    ],
)
def test_grad_T_pbc_strain_withJ(pattern):
    """PBC kinetic stress d T/d eps WITH Jastrow: orbital value+lap+gou (Hessian)
    field-strain + PW lap+gou + J1/J2 smooth-distance kinetic strain + J3 basis
    field-strain (value+lap+gou)."""
    ham = rs_ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    if pattern == "pbc_gamma_sphepgto_j123":
        jd = _randomize_j3(jd, seed=7)
        ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))
    ru, rd = _cfg(ham)
    ana = jqmc_rs.grad_T_pbc_wrt_strain(gd, jd, ru, rd)
    ref = np.asarray(pbc_stress.stress_eL_kinetic_pbc(ham, ru, rd))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_TWITHJ, level="medium")
    assert _close(ana, ref, atol, rtol)


@pytest.mark.light
def test_grad_T_pbc_strain_withJ_composite_j3():
    """PBC kinetic STRESS dT/deps for a J3 orbital that is an MO-wrapped pgto(+)PW
    COMPOSITE (``_J3_KIND['composite'] == 'mo'``): the J3-force/stress-on-composite
    case. pGTO child -> value+lap+gou field-strain via the cart J3 leaf; PW
    child -> value STRAIN-INVARIANT + lap (``2 kappa_c kappa_d val``) + gou field via the
    reciprocal-metric response, mirroring the determinant composite stress. Oracle:
    ``pbc_stress.stress_eL_kinetic_pbc`` (the productionized analytic ground truth)."""
    from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell, _J3_KIND

    ham = _build_ham_for_cell("QE:Li_gamma_pgto+pw", True, {"j3": _J3_KIND["composite"]}, "SD")
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_TWITHJ, level="medium")
    for seed in range(2):
        ru, rd = _cfg(ham, seed=seed)
        ana = np.asarray(jqmc_rs.grad_T_pbc_wrt_strain(gd, jd, ru, rd))
        ref = np.asarray(pbc_stress.stress_eL_kinetic_pbc(ham, ru, rd))
        assert _close(ana, ref, atol, rtol), (seed, np.abs(ana - ref).max())
