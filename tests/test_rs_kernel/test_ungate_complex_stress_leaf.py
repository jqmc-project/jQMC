"""Native stress on pure plane-wave / B-spline AO leaves.

Two sub-cases, both compared against the production XLA analytic stress
(``assemble.compute_grad_{e_L,ln_Psi}(wrt="eps")``):

  * COMPLEX-TWIST pure-PW / B-spline stress: the native complex stress
    cores ``vmc_grad_e_L_eps_pbc_complex`` / ``vmc_grad_ln_psi_eps_pbc_complex`` on a
    plane-wave / B-spline complex AO leaf, which has none of the pGTO-only fields
    (``_nucleus_index_jnp``). The tests call the cores WITHOUT a guard.

  * PURE-PW / B-spline GAMMA stress: the real native stress cores
    ``vmc_grad_e_L_eps_pbc`` / ``vmc_grad_ln_psi_eps_pbc`` on the same leaves at Gamma.

No xfail / skip: a raising native core fails the test.
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

import jqmc_rs

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import ATOL_FLOOR_MEDIUM, RTOL_FLOOR_MEDIUM
from jqmc.derivatives import assemble
from jqmc.jqmc_mcmc import _geminal_inv_batched, _NV_BASE_MESH, Nv_default

from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.with_chdir]

_W = np.asarray(_NV_BASE_MESH[Nv_default].weights, float)
_GP = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, float)
_ZONES = ("ao_eval", "mo_eval", "mo_grad", "det_grad_lap", "wf_kinetic", "coulomb", "ewald")


def _config(ham, seed=1, scale=0.6):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[np.arange(nu) % len(R)] + rng.normal(scale=scale, size=(nu, 3))
    rd = R[np.arange(nd) % len(R)] + rng.normal(scale=scale, size=(nd, 3))
    return np.ascontiguousarray(ru), np.ascontiguousarray(rd)


def _ginv(ham, ru, rd):
    _, g, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    return np.asarray(g[0])


def _tol():
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES)
    return max(atol, ATOL_FLOOR_MEDIUM), max(rtol, RTOL_FLOOR_MEDIUM)


def _xla_stress(ham, ru, rd, ginv):
    ruj, rdj, gj = jnp.asarray(ru), jnp.asarray(rd), jnp.asarray(ginv)
    hf = np.asarray(np.real(assemble.compute_grad_e_L(ham, ruj, rdj, np.eye(3), wrt="eps", geminal_inv=gj)), float)
    pp = np.asarray(np.real(assemble.compute_grad_ln_Psi(ham, ruj, rdj, wrt="eps", geminal_inv=gj)), float)
    return hf, pp


# --------------------------------------------------------------------------- #
# Complex-twist PW / B-spline stress.
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["pw_twist_jnone", "bspline_twist_jnone"])
def test_ungate_c6_complex_stress_leaf(pattern):
    """Complex-twist PW / B-spline stress: the native complex stress cores equal
    the XLA analytic stress on the non-pGTO complex AO leaf; a raise fails the test."""
    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _config(ham)
    ginv = _ginv(ham, ru, rd)
    n_hf = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc_complex(gd, jd, cp, ru, rd, _W, _GP), float)
    n_pp = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(gd, jd, ru, rd), float)
    x_hf, x_pp = _xla_stress(ham, ru, rd, ginv)
    atol, rtol = _tol()
    np.testing.assert_allclose(n_hf, x_hf, atol=atol, rtol=rtol, err_msg=f"{pattern} complex stress HF")
    np.testing.assert_allclose(n_pp, x_pp, atol=atol, rtol=rtol, err_msg=f"{pattern} complex stress PP")


# --------------------------------------------------------------------------- #
# Pure-PW / B-spline Gamma stress.
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["pw_gamma_jnone", "bspline_gamma_jnone"])
def test_ungate_c6_pure_pw_gamma_stress_green(pattern):
    """Real-Gamma pure-PW / B-spline stress: the native real stress cores run on these
    leaves and match the XLA analytic stress."""
    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _config(ham)
    ginv = _ginv(ham, ru, rd)
    n_hf = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, ru, rd, _W, _GP, ginv), float)
    n_pp = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc(gd, jd, ru, rd, ginv), float)
    x_hf, x_pp = _xla_stress(ham, ru, rd, ginv)
    atol, rtol = _tol()
    assert n_hf.shape == x_hf.shape == (3, 3)
    np.testing.assert_allclose(n_hf, x_hf, atol=atol, rtol=rtol, err_msg=f"{pattern} Gamma stress HF")
    np.testing.assert_allclose(n_pp, x_pp, atol=atol, rtol=rtol, err_msg=f"{pattern} Gamma stress PP")
