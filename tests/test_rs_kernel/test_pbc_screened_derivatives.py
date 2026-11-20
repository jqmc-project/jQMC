"""Periodic real-Gamma force / stress kernels on the screened FAST e_L (Cartesian pGTO, ECP).

The native FAST chunk e_L applies a runtime AO screen (``NATIVE_AO_SCREEN_EXPONENT`` /
``FAST_AO_SCREEN_ABS_BOUND``: shell ``pz_val`` for the orbital value, ``pz_deriv`` for the
kinetic gradient / Laplacian, shell + primitive ``pz_val`` on the ECP non-local mesh). The
periodic force and stress kernels (``vmc_grad_e_L_pbc``, ``vmc_grad_e_L_eps_pbc``) apply the
same screens to the AO derivative tables they contract, so they are the derivatives of that
screened energy. The committed sphe fixtures do not take the screen (it keys on the
Cartesian force leaves), hence the Cartesian lithium cell here.

References:

* ``jqmc_rs._e_L_fast_screened`` -- the screened FAST e_L built from scratch at a
  configuration (screened orbital values -> determinant state + inverse, FAST kinetic tables,
  ECP mesh cutoff). Central finite differences of it in the electron coordinates, the nuclear
  positions and the affine strain are an independent reference for the analytic kernels.
* The unscreened XLA autograd oracle, within the native-vs-XLA consistency tolerance the
  sphe-pGTO force / stress parity tests use.

The screen moves an AO by at most ``FAST_AO_SCREEN_ABS_BOUND``, far below what a finite
difference resolves, so these checks pin the kernels as derivatives of the screened energy
to finite-difference accuracy; they do not resolve the screened-out contributions themselves.

Mixed precision takes the same screens (the FAST e_L carries the screened fp32 ``ao_eval``
orbital values, the derivative tables stay fp64), but the fp32 zones put the native e_L below
the resolution of a finite difference; there only the XLA comparison applies.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs  # noqa: E402

from jqmc._precision import is_mixed_precision_enabled  # noqa: E402
from jqmc._setting import ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT  # noqa: E402
from tests.test_rs_kernel._rs_fixtures import _build_ham_for_cell  # noqa: E402
from tests.test_rs_kernel.test_gfmc_forces import _inject, _pbc_cfg  # noqa: E402
from tests.test_rs_kernel.test_vmc_forces import _ATOL_PBC, _RTOL_PBC, _jax_ref  # noqa: E402
from tests.test_rs_kernel.test_vmc_stress import _GP, _W, _jax_stress  # noqa: E402

pytestmark = [pytest.mark.with_chdir]

# Central-difference step (bohr / strain); the FD truncation error ~h^2 and the cancellation
# floor eps/h both sit well inside ATOL/RTOL_FD_VS_ANALYTIC_TIGHT.
_FD_STEP = 1.0e-4


def _cart_ham(with_jastrow: bool):
    return _build_ham_for_cell("lithium_cart.h5", with_jastrow, None, "SD")


def _host(ham):
    # strained / injected Hamiltonians carry jax leaves; the native binding takes numpy
    return jax.tree_util.tree_map(lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, ham)


def _e(ham, ru, rd):
    return jqmc_rs._e_L_fast_screened(_host(ham), ru, rd, _W, _GP)[0]


def _assert_fd(analytic, fd, label):
    np.testing.assert_allclose(
        np.asarray(analytic), np.asarray(fd), atol=ATOL_FD_VS_ANALYTIC_TIGHT, rtol=RTOL_FD_VS_ANALYTIC_TIGHT, err_msg=label
    )


@pytest.mark.heavy
@pytest.mark.parametrize("with_jastrow", [False, True], ids=["jnone", "j12"])
def test_pbc_force_matches_fd_of_screened_e_L(with_jastrow):
    """de_L/dr and de_L/dR of the native kernel vs central FD of the screened FAST e_L."""
    ham = _cart_ham(with_jastrow)
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg(ham, 0)
    _, ginv = jqmc_rs._e_L_fast_screened(ham, ru, rd, _W, _GP)
    dR, du, dd = (np.asarray(x) for x in jqmc_rs.vmc_grad_e_L_pbc(gd, jd, cp, ru, rd, _W, _GP, np.asarray(ginv)))
    if not is_mixed_precision_enabled():
        _check_force_fd(ham, ru, rd, dR, du, dd)
    # and the unscreened XLA autograd oracle, within the native-vs-XLA parity tolerance
    jR, ju, jdn = _jax_ref(ham, ru, rd, np.asarray(ginv))
    np.testing.assert_allclose(dR, jR, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="dR vs XLA")
    np.testing.assert_allclose(du, ju, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="dr_up vs XLA")
    np.testing.assert_allclose(dd, jdn, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="dr_dn vs XLA")


def _check_force_fd(ham, ru, rd, dR, du, dd):
    h = _FD_STEP
    for label, r, grad, is_up in (("r_up", ru, du, True), ("r_dn", rd, dd, False)):
        fd = np.zeros_like(r)
        for i in range(r.shape[0]):
            for c in range(3):
                rp, rm = r.copy(), r.copy()
                rp[i, c] += h
                rm[i, c] -= h
                if is_up:
                    fd[i, c] = (_e(ham, rp, rd) - _e(ham, rm, rd)) / (2 * h)
                else:
                    fd[i, c] = (_e(ham, ru, rp) - _e(ham, ru, rm)) / (2 * h)
        _assert_fd(grad, fd, label)
    P0 = np.asarray(ham.structure_data.positions, float)
    fd = np.zeros_like(P0)
    for a in range(P0.shape[0]):
        for c in range(3):
            Pp, Pm = P0.copy(), P0.copy()
            Pp[a, c] += h
            Pm[a, c] -= h
            fd[a, c] = (_e(_inject(ham, jnp.asarray(Pp)), ru, rd) - _e(_inject(ham, jnp.asarray(Pm)), ru, rd)) / (2 * h)
    _assert_fd(dR, fd, "R")


@pytest.mark.heavy
@pytest.mark.parametrize("with_jastrow", [False, True], ids=["jnone", "j12"])
def test_pbc_stress_matches_fd_of_screened_e_L(with_jastrow):
    """de_L/deps of the native kernel vs central FD of the affinely strained screened FAST e_L."""
    ham = _cart_ham(with_jastrow)
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _pbc_cfg(ham, 1)
    _, ginv = jqmc_rs._e_L_fast_screened(ham, ru, rd, _W, _GP)
    sig = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, ru, rd, _W, _GP, np.asarray(ginv), True))
    if not is_mixed_precision_enabled():
        h = _FD_STEP
        fd = np.zeros((3, 3))
        for c in range(3):
            for d in range(3):
                vals = []
                for sgn in (1.0, -1.0):
                    F = np.eye(3)
                    F[c, d] += sgn * h
                    hs = ham._apply_strain(jnp.asarray(F))
                    vals.append(_e(hs, ru @ F.T, rd @ F.T))
                fd[c, d] = (vals[0] - vals[1]) / (2 * h)
        _assert_fd(sig, fd, "stress")
    hf, _ = _jax_stress(ham, ru, rd, np.eye(3), np.asarray(ginv))
    np.testing.assert_allclose(sig, hf, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg="stress vs XLA")
