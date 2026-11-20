"""PW / B-spline / composite leaf param-derivs (O_k and dE_L/dc on ``mo_coefficients``).

This test calls the native leaf kernels (``grad_ln_det_wrt_mo_coeff`` for O_k,
``grad_eL_wrt_mo_coeff`` for dE_L/dc) on the pure plane-wave, pure B-spline and
pGTO(+)PW composite fixtures and compares against the ``jax.grad`` of the XLA
forward -- the autodiff reference (``derivatives/_auto``) for these leaves. A
regression in a leaf kernel fails the corresponding row.
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

import jqmc_rs

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import ATOL_FLOOR_MEDIUM, RTOL_FLOOR_MEDIUM
from jqmc.determinant import compute_ln_det_geminal_all_elements
from jqmc.wavefunction import Wavefunction_data, compute_kinetic_energy

from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.with_chdir]

_OK_ZONES = ("ao_eval", "det_eval")
_EL_ZONES = ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic")


def _tol(zones):
    # All fixtures here are periodic (PW / B-spline / composite): carry the pGTO/PW
    # far-image truncation floor (``ATOL_FLOOR_MEDIUM``), the same route-parity floor
    # ``test_vmc_forces`` / ``test_mcmc_observables`` use for periodic cells.
    atol, rtol = get_tolerance_xla_rust_consistency(zones)
    return max(atol, ATOL_FLOOR_MEDIUM), max(rtol, RTOL_FLOOR_MEDIUM)


def _config(ham, seed=1):
    gem = ham.wavefunction_data.geminal_data
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[np.arange(nu) % len(R)] + rng.normal(scale=1.0, size=(nu, 3))
    rd = R[np.arange(nd) % len(R)] + rng.normal(scale=1.0, size=(nd, 3))
    return np.ascontiguousarray(ru), np.ascontiguousarray(rd)


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["pw_gamma_jnone", "bspline_gamma_jnone", "pbc_composite_j12"])
def test_ungate_c4_okc_mo_leaf(pattern):
    """O_k = dln|det|/d(mo_coefficients) on the non-GTO leaf, native vs jax.grad of
    the XLA forward."""
    ham = rs_ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    ru, rd = _config(ham)
    gu, gdn = jqmc_rs.grad_ln_det_wrt_mo_coeff(gd, ru, rd)
    g = jax.grad(lambda G: compute_ln_det_geminal_all_elements(G, ru, rd))(gd)
    atol, rtol = _tol(_OK_ZONES)
    np.testing.assert_allclose(np.asarray(gu), np.asarray(g.orb_data_up_spin.mo_coefficients), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(gdn), np.asarray(g.orb_data_dn_spin.mo_coefficients), atol=atol, rtol=rtol)


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["pw_gamma_jnone", "bspline_gamma_jnone", "pbc_composite_j12"])
def test_ungate_c4_delc_mo_leaf(pattern):
    """dE_L/dc = dT/d(mo_coefficients) on the non-GTO leaf, native vs jax.grad of the
    XLA kinetic forward."""
    ham = rs_ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    ru, rd = _config(ham)
    au, ad = jqmc_rs.grad_eL_wrt_mo_coeff(gd, jd, ru, rd)
    g = jax.grad(lambda G: compute_kinetic_energy(Wavefunction_data(jastrow_data=jd, geminal_data=G), ru, rd))(gd)
    atol, rtol = _tol(_EL_ZONES)
    np.testing.assert_allclose(np.asarray(au), np.asarray(g.orb_data_up_spin.mo_coefficients), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(ad), np.asarray(g.orb_data_dn_spin.mo_coefficients), atol=atol, rtol=rtol)
