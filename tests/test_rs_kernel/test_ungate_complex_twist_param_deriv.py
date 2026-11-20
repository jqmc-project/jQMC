"""Native complex-twist O_k and dE_L/dc for the GEMINAL block.

The native complex geminal param-grad kernels -- ``grad_ln_det_mo_complex`` /
``grad_ln_det_lambda_complex`` (O_k), ``grad_T_mo_withJ_complex`` (dE_L/dc kinetic)
and the holomorphic complex ECP weight kernel ``grad_eL_ecp_wrt_mo_coeff_complex``
(dE_L/dc ECP) -- must reproduce the un-conjugated holomorphic O_k / dE_L convention
the driver needs, combined per leaf with the closed-shell singlet tie exactly as the
production reducer does. The XLA reference is the production assemble path
(``derivatives.assemble.compute_grad_{ln_Psi,e_L}(wrt="c")``), which this module
independently confirms equals the ``_auto`` autodiff O_k to ~1e-16 before trusting
it (``test_ungate_c1_reference_is_trustworthy``), so a failure here is a genuine
native gap and never a weakened oracle.

Native entry points under test:
  * O_k  mo (SD twist)   : ``jqmc_rs.grad_ln_det_mo_complex``
  * O_k  lambda (AGP)    : ``jqmc_rs.grad_ln_det_lambda_complex``
  * dE_L mo kinetic      : ``jqmc_rs.grad_T_mo_withJ_complex``
  * dE_L mo ECP weight   : ``jqmc_rs.grad_eL_ecp_wrt_mo_coeff_complex`` (the
                           Re-projected ``grad_V_ecp_nonlocal_wrt_mo_withJ_complex``
                           leaks Im and is not used here).
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

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jqmc_rs

from jqmc._precision import get_tolerance_min
from jqmc._precision import get_tolerance_xla_rust_consistency as _backend_consistency_tolerance

get_tolerance_xla_rust_consistency = partial(_backend_consistency_tolerance, level="medium")
from jqmc.derivatives import assemble
from jqmc.derivatives.bridge import _reduce_c_param_grads
from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default, _geminal_inv_batched
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.with_chdir]

_W = np.asarray(_NV_BASE_MESH[Nv_default].weights, float)
_GP = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, float)

_OK_ZONES = ("ao_eval", "mo_eval", "det_eval", "jastrow_eval")
_EL_ZONES = ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic", "jastrow_grad_lap", "coulomb")


def _config(ham, seed=1):
    gem = ham.wavefunction_data.geminal_data
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ru = R[np.arange(nu) % len(R)] + rng.normal(scale=1.0, size=(nu, 3))
    rd = R[np.arange(nd) % len(R)] + rng.normal(scale=1.0, size=(nd, 3))
    return ru, rd


def _ginv(ham, ru, rd):
    _, g, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    return np.asarray(g[0])


def _xla_c_block(ham, ru, rd, mask, quantity, block, ginv, use_auto=False):
    """The production XLA c-axis grad for ``block``, reduced with the SAME production
    reducer (``_reduce_c_param_grads``) the flag-off driver uses. ``use_auto`` swaps
    the analytic assemble path for the ``_auto`` autodiff twin (the trust check)."""
    src = _auto_mod() if use_auto else assemble
    if quantity == "ln_Psi":
        gw = src.compute_grad_ln_Psi(ham, jnp.asarray(ru), jnp.asarray(rd), wrt="c", mask=mask, geminal_inv=jnp.asarray(ginv))
    else:
        gw = src.compute_grad_e_L(
            ham, jnp.asarray(ru), jnp.asarray(rd), np.eye(3), wrt="c", mask=mask, geminal_inv=jnp.asarray(ginv)
        )
    batched = jax.tree_util.tree_map(lambda x: jnp.stack([x]), gw)
    flat = _reduce_c_param_grads(ham.wavefunction_data, batched, 1, mask)
    return np.asarray(flat[block])[0]


def _auto_mod():
    from jqmc.derivatives import _auto

    return _auto


def _tie(gd, up, dn):
    """Assemble the driver-facing ``mo_coefficients_up`` block from the per-leaf
    native up/dn c-axis grads, EXACTLY as the production reducer
    (``bridge._reduce_c_param_grads``) and the real-path driver
    (``jqmc_mcmc._rs_native_c_param_grads``): the closed-shell singlet tie
    ``C_dn = conj(C_up)`` makes the up-MO block one variational parameter, so its
    stored grad is ``up + conj(dn)``. Open shell (no tie): the up leaf directly. The
    native complex kernels stay per-leaf (shared with the complex force/stress cores,
    pinned by ``test_complex_core``); this test combines them the way the driver does."""
    up = np.asarray(up)
    dn = np.asarray(dn)
    return up + np.conj(dn) if bool(gd.time_reversal) else up


# --------------------------------------------------------------------------- #
# Oracle trust: the XLA-analytic assemble O_k == the _auto autodiff O_k on the
# complex-twist geminal. It certifies the reference before any native comparison
# uses it (never weaken an oracle).
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
def test_ungate_c1_reference_is_trustworthy():
    """Ungate c1 reference is trustworthy."""
    ham = rs_ham("twist_sphepgto_jnone")
    ru, rd = _config(ham)
    ginv = _ginv(ham, ru, rd)
    ana = _xla_c_block(ham, ru, rd, {"opt_geminal": True}, "ln_Psi", "mo_coefficients_up", ginv, use_auto=False)
    auto = _xla_c_block(ham, ru, rd, {"opt_geminal": True}, "ln_Psi", "mo_coefficients_up", ginv, use_auto=True)
    atol, rtol = get_tolerance_min(_OK_ZONES, level="strict")
    np.testing.assert_allclose(ana, auto, atol=atol, rtol=rtol, err_msg="assemble O_k != _auto O_k (oracle broken)")


# --------------------------------------------------------------------------- #
# Complex-twist O_k (dln|Psi|/dc) on the geminal.
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
def test_ungate_c1_okc_mo_complex():
    """SD complex twist: ``grad_ln_det_mo_complex`` must equal the XLA c-axis
    ``mo_coefficients_up`` block."""
    ham = rs_ham("twist_sphepgto_jnone")
    gd = ham.wavefunction_data.geminal_data
    ru, rd = _config(ham)
    ginv = _ginv(ham, ru, rd)
    gu, gdn = jqmc_rs.grad_ln_det_mo_complex(gd, ru, rd)
    native = _tie(gd, gu, gdn)
    ref = _xla_c_block(ham, ru, rd, {"opt_geminal": True}, "ln_Psi", "mo_coefficients_up", ginv)
    atol, rtol = _backend_consistency_tolerance(_OK_ZONES, level="medium")
    np.testing.assert_allclose(native, ref, atol=atol, rtol=rtol, err_msg="complex O_k mo not ported")


@pytest.mark.heavy
def test_ungate_c1_okc_lambda_complex():
    """AGP complex twist: ``grad_ln_det_lambda_complex`` must equal the XLA c-axis
    ``lambda_matrix`` block."""
    ham = _build_ham_for_cell("lithium_k025.h5", False, None, "AGP")
    gd = ham.wavefunction_data.geminal_data
    ru, rd = _config(ham)
    ginv = _ginv(ham, ru, rd)
    lam = jqmc_rs.grad_ln_det_lambda_complex(gd, ru, rd)
    ref = _xla_c_block(ham, ru, rd, {"opt_geminal": True}, "ln_Psi", "lambda_matrix", ginv)
    atol, rtol = _backend_consistency_tolerance(_OK_ZONES, level="medium")
    np.testing.assert_allclose(np.asarray(lam), ref, atol=atol, rtol=rtol, err_msg="complex O_k lambda not ported")


# --------------------------------------------------------------------------- #
# Complex-twist dE_L/dc on the geminal (kinetic; and ECP holomorphic weight).
# --------------------------------------------------------------------------- #
@pytest.mark.heavy
def test_ungate_c1_delc_mo_complex_kinetic_ae():
    """AE (non-ECP) complex twist: dE_L/dc = dT/dc has NO potential term, so
    ``grad_T_mo_withJ_complex`` alone must equal the XLA c-axis ``mo_coefficients_up``
    block; this isolates the complex kinetic reverse."""
    ham = _build_ham_for_cell("H2_pbc_ae_k025_sphe.h5", False, None, "SD")
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    ru, rd = _config(ham)
    ginv = _ginv(ham, ru, rd)
    ku, kdn = jqmc_rs.grad_T_mo_withJ_complex(gd, jd, ru, rd)
    native = _tie(gd, ku, kdn)
    ref = _xla_c_block(ham, ru, rd, {"opt_geminal": True}, "e_L", "mo_coefficients_up", ginv)
    atol, rtol = _backend_consistency_tolerance(_EL_ZONES, level="medium")
    np.testing.assert_allclose(native, ref, atol=atol, rtol=rtol, err_msg="complex dE_L/dc kinetic not ported")


@pytest.mark.heavy
def test_ungate_c1_delc_mo_complex_ecp():
    """ECP complex twist: dE_L/dc = dT/dc + dV_ecp_nl/dc. The native assembly is the
    kinetic ``grad_T_mo_withJ_complex`` + the DEDICATED holomorphic complex ECP weight
    kernel ``grad_eL_ecp_wrt_mo_coeff_complex``; their sum must equal
    the XLA c-axis ``mo_coefficients_up`` block. Both native entries fold in the
    closed-shell singlet tie (``up + conj(dn)``), matching the production reducer. The
    holomorphic weight chains the complex O_k un-projected and takes Re LAST, so the
    ``Im(wf_ratio).Im(O_k)`` cross term survives at the twist."""
    assert hasattr(jqmc_rs, "grad_eL_ecp_wrt_mo_coeff_complex"), "grad_eL_ecp_wrt_mo_coeff_complex missing from jqmc_rs"
    ham = rs_ham("twist_sphepgto_jnone")
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cp = ham.coulomb_potential_data
    ru, rd = _config(ham)
    ginv = _ginv(ham, ru, rd)
    ku, kdn = jqmc_rs.grad_T_mo_withJ_complex(gd, jd, ru, rd)
    vu, vdn = jqmc_rs.grad_eL_ecp_wrt_mo_coeff_complex(gd, jd, cp, ru, rd, _W, _GP)
    au = _tie(gd, ku, kdn) + _tie(gd, vu, vdn)
    ref = _xla_c_block(ham, ru, rd, {"opt_geminal": True}, "e_L", "mo_coefficients_up", ginv)
    atol, rtol = _backend_consistency_tolerance(_EL_ZONES, level="medium")
    np.testing.assert_allclose(au, ref, atol=atol, rtol=rtol, err_msg="complex ECP dE_L/dc holomorphic weight not ported")
