"""Layer-1 equivalence: VMC (MCMC) periodic STRESS cores, native Rust vs JAX oracle.
de_L/deps (HF stress) + dln|Psi|/deps (Pulay stress).

The MCMC stress = the CONTINUOUS VMC strain derivative the driver's ``_grad_e_L_eps`` /
``_grad_ln_psi_eps`` emit (``jqmc.derivatives.assemble.compute_grad_*`` wrt="eps"),
NOT the LRDMC-discretized ``_gfmc_stress_ecp_pbc_*`` (which adds the alat S' terms). The
native aggregates ``vmc_grad_e_L_eps_pbc`` / ``vmc_grad_ln_psi_eps_pbc`` (+ complex) mirror
the ``gfmc_stress_ecp_pbc_{withj,complex}`` SMOOTH BASE (withJ periodic kinetic stress +
Ewald/MPC ee/ei/ii strain + ECP local + ECP non-local mesh stress with the quadrature
directions held fixed) WITHOUT S', reusing the sweep SM ``geminal_inv`` (real path).

J2 (e-e) convention: the two-body Jastrow cell strains with the physical lattice, so its
Williamson metric remains commensurate with the simulation cell. This mirrors the production
analytic ``pbc_stress.stress_{lnPsi,eL_kinetic}_pbc``.

Periodic / twist STRESS fp floor: like the periodic force, the strain derivative of the
periodic e_L amplifies the periodic-kinetic sphe-pGTO grad/lap far-from-cell-image fp floor
(``ATOL_FLOOR_MEDIUM`` = 2e-5; measured ~5e-7 real, ~7e-6 twist -- comfortably
inside). Composite(pGTO+PW) AE is bit-exact (no ECP mesh). Pure-B-spline and complex-twist
NON-pGTO leaves are covered by the native aggregate (real via ``bspline_kstress_side``,
complex via the PW metric response / ``bspline_kstress_side_complex``); their parity is pinned in
``test_rs_analytic_matrix_rRe`` / ``test_ungate_complex_stress_leaf``, not here.
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

from jqmc._precision import get_tolerance_xla_rust_consistency  # noqa: E402
from jqmc._setting import ATOL_FLOOR_MEDIUM, RTOL_FLOOR_MEDIUM  # noqa: E402
from jqmc.derivatives import assemble  # noqa: E402
from jqmc.jqmc_mcmc import _geminal_inv_batched, _NV_BASE_MESH, Nv_default  # noqa: E402
from tests.test_rs_kernel._rs_fixtures import build_twist_mo_j3_ham, rs_ham  # noqa: E402
from tests.test_rs_kernel.test_gfmc_forces import _pbc_cfg, _rand_j3  # noqa: E402  (numpy-leaf, symmetrized)

pytestmark = [pytest.mark.with_chdir]

_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap")
_ATOL, _RTOL = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
_ATOL_PBC, _RTOL_PBC = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
_ATOL_PBC = max(_ATOL_PBC, ATOL_FLOOR_MEDIUM)
_RTOL_PBC = max(_RTOL_PBC, RTOL_FLOOR_MEDIUM)

_MESH = _NV_BASE_MESH[Nv_default]
_W = np.asarray(_MESH.weights, float)
_GP = np.asarray(_MESH.grid_points, float)


def _pbc_cfg_far(ham, seed, scatter=False):
    """FAR-roamed configs, two physical classes:

    * ``scatter=False``: the ``test_periodic_invariants`` recipe r_i = R + n_i*T + N(0, 0.5)
      with n_i in {2, 5, 0} -- INTEGER-lattice-shifted out of the home cell. The n_i ORDER puts
      a NONZERO shift on electron 0: lithium has a single up electron, and a naive {0,2,5}
      order leaves it in the home cell, making the far row vacuous -- exactly how the e-i
      Ewald-strain min-image bug shipped past the first far rows.
    * ``scatter=True``: a wide Gaussian scatter (scale 4.0, the ``test_vmc_forces`` far recipe)
      -- NON-lattice far configs at arbitrary fractional offsets. This is the class a
      production MCMC walker actually reaches (measured frac ~1.7 within 40 steps) and the one
      that exposed the raw ``r_i - R_I - T_n`` e-i Ewald strain (no min-image; 4.0 Ha on a
      lithium walker)."""
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    T = np.asarray(ham.structure_data.cell, float)
    ns = np.array([2, 5, 0])
    rng = np.random.default_rng(seed)

    def mk(n):
        out = np.zeros((n, 3))
        for i in range(n):
            if scatter:
                out[i] = R[i % len(R)] + rng.normal(scale=4.0, size=3)
            else:
                out[i] = R[i % len(R)] + ns[i % 3] * T[i % 3] + rng.normal(scale=0.5, size=3)
        return out

    return mk(nu), mk(nd)


def _pbc_geminv(ham, ru, rd, cplx):
    _, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    return np.asarray(ginv[0], dtype=(np.complex128 if cplx else np.float64))


def _rand_dtn(wf, seed=5, scale=0.05):
    d = wf.jastrow_data.jastrow_dtn_data
    if d is None:
        return wf
    rng = np.random.default_rng(seed)
    upd = {
        f: np.asarray(scale * rng.standard_normal(np.asarray(getattr(d, f)).shape))
        for f in ("alpha_indep", "beta_indep", "gamma_indep")
    }
    return wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_dtn_data=d.replace(**upd)))


def _jax_stress(ham, ru, rd, RT, ginv):
    """Production analytic oracle == the driver's ``_grad_e_L_eps`` / ``_grad_ln_psi_eps``
    (analytic=True): de_L/deps (HF, mesh-involving RT) + dln|Psi|/deps (Pulay)."""
    hf = np.asarray(
        assemble.compute_grad_e_L(
            ham, jnp.asarray(ru), jnp.asarray(rd), jnp.asarray(RT), wrt="eps", geminal_inv=jnp.asarray(ginv)
        )
    )
    pp = np.asarray(
        assemble.compute_grad_ln_Psi(ham, jnp.asarray(ru), jnp.asarray(rd), wrt="eps", geminal_inv=jnp.asarray(ginv))
    )
    return hf, pp


# --------------------------------------------------------------------------- #
# PERIODIC real-Gamma (sphe-pGTO ECP) HF + Pulay stress, J-ladder, near + far.
# --------------------------------------------------------------------------- #
_PBC_REAL = ["pbc_gamma_sphepgto_jnone", "pbc_gamma_sphepgto_j12", "pbc_gamma_sphepgto_j123", "pbc_gamma_sphepgto_jdtn"]
_PBC_TWIST = ["twist_sphepgto_jnone", "twist_sphepgto_j12", "twist_sphepgto_jdtn"]


@pytest.mark.heavy
@pytest.mark.parametrize(
    "seed,far,scatter",
    [(0, False, False), (3, True, False), (6, True, False), (3, True, True)],
    ids=["near", "far3", "far6", "farscatter3"],
)
@pytest.mark.parametrize("pattern", _PBC_REAL)
def test_vmc_stress_pbc_real(pattern, seed, far, scatter):
    """Vmc stress PBC real."""
    ham = rs_ham(pattern)
    if "j123" in pattern:
        ham = ham.replace(wavefunction_data=_rand_j3(ham.wavefunction_data))
    if "jdtn" in pattern:
        ham = ham.replace(wavefunction_data=_rand_dtn(ham.wavefunction_data))
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg_far(ham, seed, scatter=scatter) if far else _pbc_cfg(ham, seed)
    ginv = _pbc_geminv(ham, ru, rd, False)
    rng = np.random.default_rng(seed + 11)
    RT = np.linalg.qr(rng.standard_normal((3, 3)))[0]  # random orthogonal ECP mesh rotation
    hf_j, pp_j = _jax_stress(ham, ru, rd, RT, ginv)
    hf_n = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, ru, rd, _W, _GP @ RT, ginv))
    pp_n = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc(gd, jd, ru, rd, ginv))
    np.testing.assert_allclose(hf_n, hf_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} HF stress")
    np.testing.assert_allclose(pp_n, pp_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} Pulay stress")


# The AE controls have no ECP mesh. The ECP J3+DTN cases require the slow mesh
# Jastrow path: a PW-containing J3 cannot use MeshJasFastPbc's cached pGTO path.
@pytest.mark.heavy
@pytest.mark.parametrize(
    "pattern",
    ["pbc_composite_jnone", "pbc_composite_j12", "QEPROJ:pw:Li_gamma:ECP", "QE:Li_gamma_pgto+pw"],
)
def test_vmc_stress_pbc_composite(pattern):
    """HF and Pulay stress include active DTN beside a non-cached PW/composite J3."""
    slow_mesh = pattern in ("QEPROJ:pw:Li_gamma:ECP", "QE:Li_gamma_pgto+pw")
    if slow_mesh:
        from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

        ham = _build_ham_for_cell(pattern, True, {"j3": "mo", "dtn": True}, "SD")
        jd = ham.wavefunction_data.jastrow_data
        dtn = jd.jastrow_dtn_data
        assert dtn is not None
        assert any(np.any(np.asarray(getattr(dtn, name))) for name in ("alpha_indep", "beta_indep", "gamma_indep"))
        assert np.any(np.asarray(jd.jastrow_three_body_data.j_matrix))
    else:
        ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    for seed in (0, 3) if slow_mesh else (0,):
        ru, rd = _pbc_cfg(ham, seed)
        ginv = _pbc_geminv(ham, ru, rd, False)
        RT = np.linalg.qr(np.random.default_rng(seed + 11).standard_normal((3, 3)))[0] if slow_mesh else np.eye(3)
        hf_j, pp_j = _jax_stress(ham, ru, rd, RT, ginv)
        hf_n = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, ru, rd, _W, _GP @ RT, ginv))
        pp_n = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc(gd, jd, ru, rd, ginv))
        np.testing.assert_allclose(hf_n, hf_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{pattern} seed={seed} HF stress")
        np.testing.assert_allclose(pp_n, pp_j, atol=_ATOL, rtol=_RTOL, err_msg=f"{pattern} seed={seed} Pulay stress")


@pytest.mark.heavy
@pytest.mark.parametrize(
    "stem,complex_path",
    [
        ("QEPROJ:pgto+bspline:Li_gamma:ECP", False),
        ("QEPROJ:pgto+bspline:Li_k025:ECP", True),
    ],
    ids=["gamma", "twist"],
)
def test_vmc_stress_pbc_pgto_bspline_composite(stem, complex_path):
    """The production pGTO+B-spline composite has no stress fallback or guard."""
    from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

    ham = _build_ham_for_cell(stem, False, None, "SD")
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg(ham, 0)
    ginv = _pbc_geminv(ham, ru, rd, complex_path)
    RT = np.eye(3)
    hf_j, pp_j = _jax_stress(ham, ru, rd, RT, ginv)
    if complex_path:
        hf_n = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc_complex(gd, jd, cp, ru, rd, _W, _GP @ RT, ginv))
        pp_n = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(gd, jd, ru, rd, ginv))
    else:
        hf_n = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, ru, rd, _W, _GP @ RT, ginv))
        pp_n = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc(gd, jd, ru, rd, ginv))
    np.testing.assert_allclose(hf_n, hf_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{stem} HF stress")
    np.testing.assert_allclose(pp_n, pp_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{stem} Pulay stress")


# --------------------------------------------------------------------------- #
# Pure-B-spline periodic real-Gamma (ECP) HF + Pulay stress.
# The det VALUE Pulay is affine (frac) invariant -> 0; the det KINETIC stress is the metric
# response served by ``bspline_kstress_side`` (wired into ``vmc_grad_e_L_eps_pbc`` sharing the
# g_inv-reused kinetic reverse cotangents); the ECP-nl mesh stress is basis-agnostic (empty
# pgto leaves + affine-invariant det ratio/Pulay). jnone + jdtn (exercises the bar_gou grad
# field + the DTN cell-strain). Near + integer-lattice-far + wide-scatter-far. Parity is
# effectively bit-exact (~1e-13, well inside FLOOR_MEDIUM) -- the B-spline t-space fields are
# strain-invariant so there is no periodic-kinetic sphe-pGTO fp floor here.
@pytest.mark.heavy
@pytest.mark.parametrize(
    "seed,far,scatter",
    [(0, False, False), (3, True, False), (3, True, True)],
    ids=["near", "far3", "farscatter3"],
)
@pytest.mark.parametrize("kind", ["jnone", "jdtn"])
def test_vmc_stress_pbc_bspline(kind, seed, far, scatter):
    """Vmc stress PBC bspline."""
    from tests.test_rs_kernel._rs_fixtures import _build_ham_for_cell

    if kind == "jnone":
        ham = rs_ham("bspline_gamma_jnone")
    else:
        ham = _build_ham_for_cell("QEPROJ:bspline:Li_gamma:ECP", True, {"dtn": True}, "SD")
        ham = ham.replace(wavefunction_data=_rand_dtn(ham.wavefunction_data))
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg_far(ham, seed, scatter=scatter) if far else _pbc_cfg(ham, seed)
    ginv = _pbc_geminv(ham, ru, rd, False)
    rng = np.random.default_rng(seed + 11)
    RT = np.linalg.qr(rng.standard_normal((3, 3)))[0]  # random orthogonal ECP mesh rotation
    hf_j, pp_j = _jax_stress(ham, ru, rd, RT, ginv)
    hf_n = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, ru, rd, _W, _GP @ RT, ginv))
    pp_n = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc(gd, jd, ru, rd, ginv))
    np.testing.assert_allclose(hf_n, hf_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"bspline {kind} HF stress")
    np.testing.assert_allclose(pp_n, pp_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"bspline {kind} Pulay stress")


# --------------------------------------------------------------------------- #
# Rank-1 mesh-STRESS fast path (``MeshStressFastPbc``) vs the full-recompute reference.
# The production ECP-nl mesh-stress loop rank-1 Sherman-Morrison-hops one electron off a
# shared (AS-regularized) base geminal inverse (``fast_mesh=True``, the default) instead of
# THREE fresh O(N^3) forwards per mesh candidate (``compute_geminal_all_elements`` det ratio
# + ``spp`` det Pulay + ``det_grad_field`` drift). ``fast_mesh=False`` forces the reference.
#
# Agreement is 1e-11-class in the WELL-CONDITIONED regime (B-spline; wide-scatter-far pGTO,
# whose moved AOs never approach the singular near-nucleus region). At NEAR-cell sphe-pGTO the
# two evaluate the moved-electron periodic AO via different image-tiling routes (batched fresh
# forward vs single-electron SM hop), so the spread is bounded by the SAME periodic-kinetic
# sphe-pGTO fp floor as the JAX-oracle test (``ATOL_FLOOR_MEDIUM``) -- BOTH paths sit ~1e-6
# from the JAX oracle there; neither is more correct. The fast path stays finite (no NaN) at
# near-node via the AS-regularized base inverse (the fresh-LU reference can be noisier there).
def _hf_fast_ref(gd, jd, cp, ru, rd, RT, ginv):
    fast = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, ru, rd, _W, _GP @ RT, ginv, True))
    ref = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, ru, rd, _W, _GP @ RT, ginv, False))
    return fast, ref


@pytest.mark.heavy
@pytest.mark.parametrize(
    "seed,far,scatter",
    [(0, False, False), (3, True, False), (3, True, True)],
    ids=["near", "far3", "farscatter3"],
)
@pytest.mark.parametrize("pattern", _PBC_REAL)
def test_vmc_stress_mesh_fast_vs_ref_pgto(pattern, seed, far, scatter):
    """Vmc stress mesh fast vs ref pgto."""
    ham = rs_ham(pattern)
    if "j123" in pattern:
        ham = ham.replace(wavefunction_data=_rand_j3(ham.wavefunction_data))
    if "jdtn" in pattern:
        ham = ham.replace(wavefunction_data=_rand_dtn(ham.wavefunction_data))
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg_far(ham, seed, scatter=scatter) if far else _pbc_cfg(ham, seed)
    ginv = _pbc_geminv(ham, ru, rd, False)
    rng = np.random.default_rng(seed + 11)
    RT = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    fast, ref = _hf_fast_ref(gd, jd, cp, ru, rd, RT, ginv)
    assert np.isfinite(fast).all(), f"{pattern} fast mesh-stress NaN"
    atol, rtol = _ATOL_PBC, _RTOL_PBC
    np.testing.assert_allclose(fast, ref, atol=atol, rtol=rtol, err_msg=f"{pattern} fast-vs-ref HF stress")


# Pure-B-spline: strain-invariant t-space fields (empty pgto leaves -> det Pulay 0, drift
# MO-level), well-conditioned everywhere -> rank-1-vs-reference is effectively bit-exact
# (~1e-15) at near + integer-lattice-far + wide-scatter-far. This is the B-spline mesh-stress
# path that the FORCE fast path cannot cover (no pgto-leaf force grads); the stress can.
@pytest.mark.heavy
@pytest.mark.parametrize(
    "seed,far,scatter",
    [(0, False, False), (3, True, False), (3, True, True)],
    ids=["near", "far3", "farscatter3"],
)
def test_vmc_stress_mesh_fast_vs_ref_bspline(seed, far, scatter):
    """Vmc stress mesh fast vs ref bspline."""
    ham = rs_ham("bspline_gamma_jnone")
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg_far(ham, seed, scatter=scatter) if far else _pbc_cfg(ham, seed)
    ginv = _pbc_geminv(ham, ru, rd, False)
    rng = np.random.default_rng(seed + 11)
    RT = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    fast, ref = _hf_fast_ref(gd, jd, cp, ru, rd, RT, ginv)
    assert np.isfinite(fast).all()
    np.testing.assert_allclose(fast, ref, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="bspline fast-vs-ref HF stress")


# Periodic real-Gamma genuinely-MO-wrapped J3 STRESS. Production periodic J3
# wraps the geminal MOs (num_mo != num_ao); ``j3_stress_ctx_owned`` must lower MO -> AO
# (``lower_j3_to_ao``, exactly as the force ctx ``j3_pbc_ctx_owned``), else
# ``pgto_j3_stress_pieces`` reads ``MOs_data.num_ao`` (AttributeError) on the lithium MCMC
# stress. The committed periodic J3 fixtures are raw-AO, so this builds the production
# MO-wrapped J3 explicitly (as the force MO-J3 row does) and pins the HF + Pulay stress ctx.
@pytest.mark.heavy
def test_vmc_stress_pbc_mo_wrapped_j3():
    """Vmc stress PBC MO wrapped j3."""
    from tests.test_rs_kernel.test_vmc_forces import _build_pbc_mo_j3_ham

    ham = _build_pbc_mo_j3_ham("lithium.h5")
    j3o = ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data
    j3mc = np.asarray(j3o.mo_coefficients)
    assert j3mc.ndim == 2 and j3mc.shape[0] != j3mc.shape[1]  # genuinely MO-wrapped
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg(ham, 0)
    ginv = _pbc_geminv(ham, ru, rd, False)
    rng = np.random.default_rng(11)
    RT = np.linalg.qr(rng.standard_normal((3, 3)))[0]  # random orthogonal ECP mesh rotation
    hf_j, pp_j = _jax_stress(ham, ru, rd, RT, ginv)
    hf_n = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, ru, rd, _W, _GP @ RT, ginv))
    pp_n = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc(gd, jd, ru, rd, ginv))
    np.testing.assert_allclose(hf_n, hf_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="MO-J3 HF stress")
    np.testing.assert_allclose(pp_n, pp_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="MO-J3 Pulay stress")


# --------------------------------------------------------------------------- #
# COMPLEX-twist (sphe-pGTO ECP) HF + Pulay stress (real, Re-projected), near + far.
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
@pytest.mark.parametrize(
    "seed,far,scatter",
    [(0, False, False), (3, True, False), (6, True, False), (3, True, True)],
    ids=["near", "far3", "far6", "farscatter3"],
)
@pytest.mark.parametrize("pattern", _PBC_TWIST)
def test_vmc_stress_pbc_complex(pattern, seed, far, scatter):
    """Vmc stress PBC complex."""
    ham = rs_ham(pattern)
    if "jdtn" in pattern:
        ham = ham.replace(wavefunction_data=_rand_dtn(ham.wavefunction_data))
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg_far(ham, seed, scatter=scatter) if far else _pbc_cfg(ham, seed)
    ginv = _pbc_geminv(ham, ru, rd, True)
    rng = np.random.default_rng(seed + 11)
    RT = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    hf_j, pp_j = _jax_stress(ham, ru, rd, RT, ginv)
    hf_n = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc_complex(gd, jd, cp, ru, rd, _W, _GP @ RT))
    pp_n = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(gd, jd, ru, rd))
    np.testing.assert_allclose(hf_n, hf_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} HF stress")
    np.testing.assert_allclose(pp_n, pp_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} Pulay stress")


@pytest.mark.heavy
@pytest.mark.parametrize(
    "seed,far,scatter",
    [(0, False, False), (3, True, False), (6, True, False), (3, True, True)],
    ids=["near", "far3-home", "far6-home", "farscatter3-home"],
)
def test_vmc_stress_pbc_complex_j123_home_cell(seed, far, scatter):
    """Complex-twist determinant with phase-free J3: Rust/XLA HF and Pulay stress agree.

    ``twist_sphepgto_j123`` uses lithium_k025.h5, ccECP, J1/J2, and phase-free
    spherical pGTO J3 with a nonzero symmetric coupling. Near configurations and
    genuinely unwrapped far lattice-image/scattered walkers exercise the same
    numerical comparison without replacing far positions by home-cell images.
    The far positions are not restricted to the home cell.
    """
    ham = rs_ham("twist_sphepgto_j123")
    ham = ham.replace(wavefunction_data=_rand_j3(ham.wavefunction_data))
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    assert jd.jastrow_three_body_data is not None and np.any(np.asarray(jd.jastrow_three_body_data.j_matrix) != 0.0)
    ru, rd = _pbc_cfg_far(ham, seed, scatter=scatter) if far else _pbc_cfg(ham, seed)
    if far:
        cell = np.asarray(ham.structure_data.cell, float)
        fractional_positions = np.concatenate((ru, rd)) @ np.linalg.inv(cell)
        assert np.any((fractional_positions < 0) | (fractional_positions >= 1))
    ginv = _pbc_geminv(ham, ru, rd, True)
    rng = np.random.default_rng(seed + 11)
    RT = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    hf_j, pp_j = _jax_stress(ham, ru, rd, RT, ginv)
    hf_n = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc_complex(gd, jd, cp, ru, rd, _W, _GP @ RT))
    pp_n = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(gd, jd, ru, rd))
    assert np.all(np.isfinite(hf_n)) and np.all(np.isfinite(hf_j)) and np.all(np.isfinite(pp_n)) and np.all(np.isfinite(pp_j))
    np.testing.assert_allclose(hf_n, hf_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="twist_sphepgto_j123 HF stress")
    np.testing.assert_allclose(pp_n, pp_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="twist_sphepgto_j123 Pulay stress")


@pytest.mark.heavy
@pytest.mark.parametrize(
    "seed,far,scatter",
    [(0, False, False), (3, True, True)],
    ids=["near", "farscatter3"],
)
def test_vmc_stress_pbc_complex_mo_wrapped_j3(seed, far, scatter):
    """Genuine complex ``num_mo != num_ao`` J3: HF/Pulay stress Rust == XLA."""
    ham = build_twist_mo_j3_ham()
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    j3_orb = jd.jastrow_three_body_data.orb_data
    mo = np.asarray(j3_orb.mo_coefficients)
    assert mo.shape[0] != mo.shape[1] and np.iscomplexobj(mo)

    ru, rd = _pbc_cfg_far(ham, seed, scatter=scatter) if far else _pbc_cfg(ham, seed)
    ginv = _pbc_geminv(ham, ru, rd, True)
    rng = np.random.default_rng(seed + 11)
    RT = np.linalg.qr(rng.standard_normal((3, 3)))[0]

    hf_x, pp_x = _jax_stress(ham, ru, rd, RT, ginv)
    hf_r = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc_complex(gd, jd, cp, ru, rd, _W, _GP @ RT, ginv))
    pp_r = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(gd, jd, ru, rd, ginv))
    np.testing.assert_allclose(hf_r, hf_x, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="complex MO-J3 HF stress")
    np.testing.assert_allclose(pp_r, pp_x, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="complex MO-J3 Pulay stress")


# --------------------------------------------------------------------------- #
# Periodic MULTI-ELECTRON J2-own-metric stress coverage: H2_pbc AE cart.
# jqmc keys the J2 stress metric on the J2's OWN structure_data;
# ``_build_ham`` threads the loaded periodic structure into that J2, so the metric is the
# production Williamson/min-image periodic metric (matching the geminal cell).
# ``Hamiltonian_data.sanity_check`` REJECTS a periodic ham whose J2 carries a
# molecular/Euclidean structure, so a cell-less-J2 variant of this row is a
# construction bug by contract. Pins the periodic J2 stress kernels following the J2's own
# cell (``jastrow_dist_derivs`` OWN-cell, never a panic).
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
def test_vmc_stress_pbc_h2_periodic_j2():
    """Vmc stress PBC h2 periodic j2."""
    from tests.test_rs_kernel.test_rust_dispatch_equivalence import _build_ham

    ham = _build_ham("H2_pbc_ae_cart.h5")
    assert ham.structure_data.pbc_flag
    assert ham.wavefunction_data.jastrow_data.jastrow_two_body_data.structure_data.pbc_flag
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg(ham, 0)
    ginv = _pbc_geminv(ham, ru, rd, False)
    RT = np.eye(3)
    hf_j, pp_j = _jax_stress(ham, ru, rd, RT, ginv)
    hf_n = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, ru, rd, _W, _GP @ RT, ginv))
    pp_n = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc(gd, jd, ru, rd, ginv))
    np.testing.assert_allclose(hf_n, hf_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="H2 periodic-J2 HF stress")
    np.testing.assert_allclose(pp_n, pp_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="H2 periodic-J2 Pulay stress")


# --------------------------------------------------------------------------- #
# geminal-inverse reuse: the native HF/Pulay stress consumes the sweep SM inverse
# (no fresh LU). At a near-node config the reused-inverse stress stays FINITE and
# equals the JAX oracle threading the same inverse.
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
@pytest.mark.parametrize(
    "collapse,check_equal",
    [
        (1.0e-4, True),
        pytest.param(
            1.0e-8,
            False,
            marks=pytest.mark.gdd_finiteness_evidence(
                mode="stability",
                reason=(
                    "At the 1e-8 collapse native and JAX legitimately diverge by roundoff; the case is "
                    "stability evidence only and asserts that the native HF and Pulay stress are finite"
                ),
            ),
        ),
    ],
)
def test_vmc_stress_reused_inverse_near_node(collapse, check_equal):
    """Vmc stress reused inverse near node."""
    ham = rs_ham("pbc_gamma_sphepgto_j12")
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg(ham, 1)
    if int(gd.num_electron_up) >= 2:
        ru[1] = ru[0] + collapse
    ginv = _pbc_geminv(ham, ru, rd, False)
    assert np.all(np.isfinite(ginv)), "precondition: the reused SM inverse is finite"
    RT = np.eye(3)
    hf_n = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, ru, rd, _W, _GP @ RT, ginv))
    pp_n = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc(gd, jd, ru, rd, ginv))
    assert np.all(np.isfinite(hf_n)) and np.all(np.isfinite(pp_n))
    if check_equal:
        hf_j, pp_j = _jax_stress(ham, ru, rd, RT, ginv)
        np.testing.assert_allclose(hf_n, hf_j, atol=_ATOL_PBC, rtol=_RTOL_PBC)
        np.testing.assert_allclose(pp_n, pp_j, atol=_ATOL_PBC, rtol=_RTOL_PBC)
