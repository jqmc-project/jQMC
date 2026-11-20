"""Numerical acceptance for the real/complex PyO3 signature ports.

Every argument added to a ``*_complex`` entry so it matches its real twin must be
GENUINELY CONSUMED, not accepted and ignored. Each test here supplies the newly threaded
argument and pins the result against an INDEPENDENT oracle:

* ``geminal_inv`` on ``vmc_grad_{e_L,ln_psi}_{,eps_}pbc_complex`` -- the complex-twist
  force / Pulay / stress cores reuse the MCMC sweep's running complex Sherman-Morrison
  inverse for every base-config determinant field (the real twins' behaviour). Supplying it
  must reproduce the fresh-rebuild result AND the XLA analytic reference.
* ``fast_mesh`` on ``vmc_grad_e_L_eps_pbc_complex`` -- each reflected ECP-mesh candidate is a
  single-electron hop, so its det ratio and its inverse follow from the base inverse by a
  rank-1 Sherman-Morrison update instead of a fresh O(N^3) rebuild. ``fast_mesh=False`` keeps
  the full recompute (the oracle); both must agree and both must match XLA.
* ``jastrow_data`` on ``grad_T_wrt_R_complex`` / ``grad_lnPsi_wrt_strain_complex`` /
  ``grad_V_ecp_nonlocal_wrt_strain_complex``. With a Jastrow
  the result must move to the WITH-JASTROW reference, not stay at the jnone value.

Convention pin: the complex derivative chains stay HOLOMORPHIC with ``Re``
taken LAST (``Re(A)*Re(B) != Re(A*B)``); the oracles below are the production XLA analytic
routes / ``jax.grad``, so a per-factor Re-projection would show up here.
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
from jqmc.derivatives import assemble  # noqa: E402
from jqmc.jqmc_mcmc import _geminal_inv_batched, _inject_active_positions, _NV_BASE_MESH, Nv_default  # noqa: E402
from jqmc.wavefunction import compute_kinetic_energy  # noqa: E402
from tests.test_rs_kernel._rs_fixtures import rs_ham  # noqa: E402
from tests.test_rs_kernel.test_gfmc_forces import _pbc_cfg  # noqa: E402

pytestmark = [pytest.mark.with_chdir]

_MESH = _NV_BASE_MESH[Nv_default]
_W = np.asarray(_MESH.weights, float)
_GP = np.asarray(_MESH.grid_points, float)

# Periodic/twist force + stress fp floor (the same floor `test_vmc_forces` /
# `test_vmc_stress` pin: the position/strain derivative of the periodic e_L amplifies the
# periodic-kinetic sphe-pGTO far-image floor).
_ATOL_PBC, _RTOL_PBC = get_tolerance_xla_rust_consistency(("ao_eval", "wf_kinetic", "local_energy", "swct"), level="medium")

_TWIST = ["twist_sphepgto_jnone", "twist_sphepgto_j12", "twist_sphepgto_jdtn"]


def _geminv_c(ham, ru, rd):
    """The batched complex geminal inverse the MCMC driver carries as its SM state."""
    _, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    return np.asarray(ginv[0], dtype=np.complex128)


def _xla_force(ham, ru, rd, ginv):
    RT = np.eye(3)
    dR = np.asarray(
        assemble.compute_grad_e_L(ham, jnp.asarray(ru), jnp.asarray(rd), RT, wrt="R", geminal_inv=jnp.asarray(ginv))
    )
    dru, drd = assemble.compute_grad_e_L(ham, jnp.asarray(ru), jnp.asarray(rd), RT, wrt="r", geminal_inv=jnp.asarray(ginv))
    return dR, np.asarray(dru), np.asarray(drd)


def _xla_pulay(ham, ru, rd, ginv):
    dR = np.asarray(assemble.compute_grad_ln_Psi(ham, jnp.asarray(ru), jnp.asarray(rd), wrt="R", geminal_inv=jnp.asarray(ginv)))
    dru, drd = assemble.compute_grad_ln_Psi(ham, jnp.asarray(ru), jnp.asarray(rd), wrt="r", geminal_inv=jnp.asarray(ginv))
    return dR, np.asarray(dru), np.asarray(drd)


def _xla_stress(ham, ru, rd, RT, ginv):
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
# `geminal_inv` on the complex FORCE + PULAY cores
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
@pytest.mark.parametrize("seed,far", [(0, False), (3, True)], ids=["near", "far3"])
@pytest.mark.parametrize("pattern", _TWIST)
def test_complex_force_pulay_consume_sm_geminal_inv(pattern, seed, far):
    """Supplying the sweep SM complex inverse == the fresh-rebuild result == XLA.

    Both halves matter: the fresh-vs-supplied equality proves the threaded inverse really
    drives the determinant fields (a silently-ignored argument would trivially pass that
    half, so the XLA comparison pins the VALUE too).
    """
    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg(ham, seed, scale=4.0) if far else _pbc_cfg(ham, seed)
    ginv = _geminv_c(ham, ru, rd)

    hf_fresh = [np.asarray(x) for x in jqmc_rs.vmc_grad_e_L_pbc_complex(gd, jd, cp, ru, rd, _W, _GP)]
    hf_sm = [np.asarray(x) for x in jqmc_rs.vmc_grad_e_L_pbc_complex(gd, jd, cp, ru, rd, _W, _GP, ginv)]
    hf_x = _xla_force(ham, ru, rd, ginv)
    for got_sm, got_fresh, ref, tag in zip(hf_sm, hf_fresh, hf_x, ("dR", "dr_up", "dr_dn")):
        np.testing.assert_allclose(got_sm, got_fresh, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} HF {tag} SM vs fresh")
        np.testing.assert_allclose(got_sm, ref, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} HF {tag} vs XLA")

    pp_fresh = [np.asarray(x) for x in jqmc_rs.vmc_grad_ln_psi_pbc_complex(gd, jd, ru, rd)]
    pp_sm = [np.asarray(x) for x in jqmc_rs.vmc_grad_ln_psi_pbc_complex(gd, jd, ru, rd, ginv)]
    pp_x = _xla_pulay(ham, ru, rd, ginv)
    for got_sm, got_fresh, ref, tag in zip(pp_sm, pp_fresh, pp_x, ("dR", "dr_up", "dr_dn")):
        np.testing.assert_allclose(
            got_sm, got_fresh, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} Pulay {tag} SM vs fresh"
        )
        np.testing.assert_allclose(got_sm, ref, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} Pulay {tag} vs XLA")


# --------------------------------------------------------------------------- #
# `geminal_inv` + `fast_mesh` on the complex STRESS cores
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
@pytest.mark.parametrize("pattern", _TWIST)
def test_complex_stress_consumes_geminal_inv_and_fast_mesh(pattern):
    """HF stress: {fresh, SM} x {fast_mesh, full recompute} all agree and all match XLA.

    ``fast_mesh=True`` replaces each reflected ECP-mesh candidate's fresh geminal build +
    determinant + two inversions with a rank-1 Sherman-Morrison det ratio and the exact
    rank-1 inverse update, so it is only reachable when the base inverse is supplied --
    which is precisely why the two arguments are pinned together here.
    """
    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg(ham, 0)
    ginv = _geminv_c(ham, ru, rd)
    rng = np.random.default_rng(11)
    RT = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    gp = _GP @ RT

    hf_x, pp_x = _xla_stress(ham, ru, rd, RT, ginv)
    hf_fresh = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc_complex(gd, jd, cp, ru, rd, _W, gp))
    hf_slow = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc_complex(gd, jd, cp, ru, rd, _W, gp, ginv, False))
    hf_fast = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc_complex(gd, jd, cp, ru, rd, _W, gp, ginv, True))
    np.testing.assert_allclose(hf_slow, hf_fresh, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} HF stress SM vs fresh")
    np.testing.assert_allclose(
        hf_fast, hf_slow, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} HF stress fast vs full mesh"
    )
    np.testing.assert_allclose(hf_fast, hf_x, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} HF stress vs XLA")

    pp_fresh = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(gd, jd, ru, rd))
    pp_sm = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(gd, jd, ru, rd, ginv))
    np.testing.assert_allclose(pp_sm, pp_fresh, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} Pulay stress SM vs fresh")
    np.testing.assert_allclose(pp_sm, pp_x, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{pattern} Pulay stress vs XLA")


# --------------------------------------------------------------------------- #
# `jastrow_data` on the complex cores
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
def test_grad_T_wrt_R_complex_is_with_jastrow_capable():
    """``grad_T_wrt_R_complex`` with a Jastrow == jax.grad of the WITH-JASTROW kinetic energy.

    The jnone value is asserted to be DIFFERENT, so a core that silently dropped the
    Jastrow could not pass.
    """
    ham = rs_ham("twist_sphepgto_j12")
    wf = ham.wavefunction_data
    gd, jd = wf.geminal_data, wf.jastrow_data
    jnone = wf.jastrow_data.replace(jastrow_two_body_data=None, jastrow_one_body_data=None, jastrow_three_body_data=None)
    P0 = jnp.asarray(np.asarray(ham.structure_data.positions, float))
    # MEDIUM level: in full precision medium resolves to
    # the same values as strict, so the fp64 judgment is unchanged; only the mixed
    # envelope widens. dT/dR components of O(1-3) inherit both backends' fp32
    # noise from kinetic internals of O(4-10); the measured mixed residual,
    # 3.1e-6 absolute, is ~1.5x the strict envelope and seed-marginal (it passes
    # or fails between runs of identical code), while the fp64
    # parity is at the 1e-13 floor.
    atol, rtol = get_tolerance_xla_rust_consistency(("wf_kinetic", "swct", "jastrow_grad_lap"), level="medium")
    for seed in (1, 2):
        seed_label = str(seed)
        ru, rd = _pbc_cfg(ham, seed)

        def Tf(P):
            return jnp.real(compute_kinetic_energy(_inject_active_positions(wf, {"positions": P}), ru, rd))

        ref = np.asarray(jax.grad(Tf)(P0))
        got = np.asarray(jqmc_rs.grad_T_wrt_R_complex(gd, jd, ru, rd))
        np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=f"withJ seed={seed}")
        got_jnone = np.asarray(jqmc_rs.grad_T_wrt_R_complex(gd, jnone, ru, rd))
        assert not np.allclose(got_jnone, ref, atol=atol, rtol=rtol), "jnone == withJ: the fixture has no live Jastrow"


@pytest.mark.heavy
def test_grad_lnPsi_wrt_strain_complex_is_with_jastrow_capable():
    """``grad_lnPsi_wrt_strain_complex`` with a Jastrow == the production analytic
    ``compute_grad_ln_Psi(wrt='eps')`` (det Pulay stress + every periodic Jastrow metric
    strained with the lattice). The jnone value must differ."""
    ham = rs_ham("twist_sphepgto_j12")
    wf = ham.wavefunction_data
    gd, jd = wf.geminal_data, wf.jastrow_data
    jnone = jd.replace(jastrow_two_body_data=None, jastrow_one_body_data=None, jastrow_three_body_data=None)
    for seed in (0, 1):
        seed_label = str(seed)
        ru, rd = _pbc_cfg(ham, seed)
        ginv = _geminv_c(ham, ru, rd)
        ref = np.asarray(
            assemble.compute_grad_ln_Psi(ham, jnp.asarray(ru), jnp.asarray(rd), wrt="eps", geminal_inv=jnp.asarray(ginv))
        )
        got = np.asarray(jqmc_rs.grad_lnPsi_wrt_strain_complex(gd, jd, ru, rd))
        np.testing.assert_allclose(got, ref, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"withJ seed={seed}")
        got_jnone = np.asarray(jqmc_rs.grad_lnPsi_wrt_strain_complex(gd, jnone, ru, rd))
        assert not np.allclose(got_jnone, ref, atol=_ATOL_PBC, rtol=_RTOL_PBC), "jnone == withJ: no live Jastrow strain"


@pytest.mark.heavy
def test_grad_V_ecp_nonlocal_wrt_strain_complex_is_with_jastrow_capable():
    """``grad_V_ecp_nonlocal_wrt_strain_complex(..., jastrow_data=J)`` drives the SAME
    with-Jastrow mesh core as the dedicated ``..._withJ_complex`` entry (which is itself
    pinned against the XLA strain oracle in ``test_complex_core``), and the jnone call
    (``jastrow_data`` omitted) stays at the old jnone value."""
    ham = rs_ham("twist_sphepgto_j12")
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    atol, rtol = get_tolerance_xla_rust_consistency(("coulomb", "swct", "jastrow_eval"), level="strict")
    for seed_label, seed in (("0", 0), ("1", 1)):
        ru, rd = _pbc_cfg(ham, seed)
        withj = np.asarray(jqmc_rs.grad_V_ecp_nonlocal_wrt_strain_complex(gd, cp, ru, rd, _W, _GP, jd))
        ref = np.asarray(jqmc_rs.grad_V_ecp_nonlocal_wrt_strain_withJ_complex(gd, jd, cp, ru, rd, _W, _GP))
        np.testing.assert_allclose(withj, ref, atol=atol, rtol=rtol, err_msg=f"withJ seed={seed_label}")
        jnone = np.asarray(jqmc_rs.grad_V_ecp_nonlocal_wrt_strain_complex(gd, cp, ru, rd, _W, _GP))
        assert not np.allclose(jnone, ref, atol=atol, rtol=rtol), "jnone == withJ: no live Jastrow in the mesh ratio"
