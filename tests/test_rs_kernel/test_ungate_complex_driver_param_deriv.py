"""The complex-twist param-deriv path through the PRODUCTION assembler.

The complex c-axis KERNELS (``grad_ln_det_mo_complex``,
``grad_T_mo_withJ_complex``, ``grad_eL_ecp_wrt_mo_coeff_complex``, the
``grad_eL_*_complex`` Jastrow/DTN family) are not reachable through ``ParamDerivArgs``,
which parses a REAL geminal (``build_obs_ctx(want_param_deriv=True)`` raises
``TypeError: 'ndarray' object cannot be converted to 'PyArray<T, D>'`` on a twist
geminal). ``_RsComplexPdCtx`` (same ``pd_*`` method surface, free complex kernels
behind it) wires them. These rows drive the PRODUCTION assembler ``_rs_native_c_param_grads``
through that adapter and compare every emitted block against the production XLA reducer
-- i.e. they pin the driver path a twist opt run actually takes, not the bare kernels
(which ``test_ungate_complex_twist_param_deriv.py`` pins).

Scope note: the twist ECP ``e_L`` / ``mo_coefficients_up`` row is deliberately NOT
duplicated here -- it is pinned in its own module (``c1_delc_mo_complex_ecp``), whose
residual is an fp-reassociation floor of the complex diffuse-AO image sum (one of 26
elements near 1.5e-7). The ``e_L`` rows below therefore use the AE twist cell (no ECP mesh).
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

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.derivatives import assemble
from jqmc.derivatives.bridge import _MASK_DEFAULTS, _reduce_c_param_grads
from jqmc.jqmc_mcmc import (
    _geminal_inv_batched,
    _NV_BASE_MESH,
    _PARAM_BLOCK_TO_MASK_KW,
    _rs_native_c_param_grads,
    _RsComplexPdCtx,
    Nv_default,
)

from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.with_chdir]

_W = np.asarray(_NV_BASE_MESH[Nv_default].weights, float)
_GP = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, float)
_ZONES = ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic", "jastrow_grad_lap", "coulomb")


def _tol():
    return get_tolerance_xla_rust_consistency(_ZONES, level="medium")


def _mask():
    mask = dict(_MASK_DEFAULTS)
    for kw in ("opt_geminal", "opt_J1_param", "opt_J2_param", "opt_J3_param"):
        mask[kw] = True
    return mask


def _batch(ham, nw=2, seed=3, scale=0.8):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[rng.integers(0, len(R), (nw, nu))] + rng.normal(scale=scale, size=(nw, nu, 3))
    rd = R[rng.integers(0, len(R), (nw, nd))] + rng.normal(scale=scale, size=(nw, nd, 3))
    return np.ascontiguousarray(ru), np.ascontiguousarray(rd)


def _native(ham, ru_b, rd_b, mask, quantity):
    """The production assembler driven through the complex adapter -- exactly what the
    ungated driver builds for a twist run (``_rs_pd_ctx``)."""
    wf = ham.wavefunction_data
    cp = ham.coulomb_potential_data
    nw = len(ru_b)
    ctx = _RsComplexPdCtx(wf.geminal_data, wf.jastrow_data, cp)
    return _rs_native_c_param_grads(
        ctx,
        ham,
        [np.ascontiguousarray(ru_b[w]) for w in range(nw)],
        [np.ascontiguousarray(rd_b[w]) for w in range(nw)],
        mask,
        nw,
        quantity=quantity,
        ecp_flag=bool(cp.ecp_flag),
        ecp_w=_W,
        ecp_gp=_GP,
        rts_b=np.stack([np.eye(3)] * nw),
    )


def _xla(ham, ru_b, rd_b, mask, quantity):
    wf = ham.wavefunction_data
    nw = len(ru_b)
    per = []
    for w in range(nw):
        ru, rd = jnp.asarray(ru_b[w]), jnp.asarray(rd_b[w])
        _, g, _, _ = _geminal_inv_batched(wf.geminal_data, ru[None], rd[None])
        ginv = jnp.asarray(g[0])
        if quantity == "ln_Psi":
            per.append(assemble.compute_grad_ln_Psi(ham, ru, rd, wrt="c", mask=mask, geminal_inv=ginv))
        else:
            per.append(assemble.compute_grad_e_L(ham, ru, rd, np.eye(3), wrt="c", mask=mask, geminal_inv=ginv))
    gb = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *per)
    flat = _reduce_c_param_grads(wf, gb, nw, mask)
    return {n: v for n, v in flat.items() if mask.get(_PARAM_BLOCK_TO_MASK_KW.get(n, ""), True)}


def _compare(ham, quantity):
    mask = _mask()
    ru_b, rd_b = _batch(ham)
    native = _native(ham, ru_b, rd_b, mask, quantity)
    ref = _xla(ham, ru_b, rd_b, mask, quantity)
    atol, rtol = _tol()
    assert set(ref) <= set(native), f"complex assembler missing blocks {sorted(set(ref) - set(native))}"
    for b in sorted(ref):
        np.testing.assert_allclose(
            np.real(np.asarray(native[b])), np.real(np.asarray(ref[b])), atol=atol, rtol=rtol, err_msg=f"{quantity} {b}"
        )


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["twist_sphepgto_jnone", "twist_sphepgto_j12"])
def test_complex_driver_okc_matches_xla(pattern):
    """O_k through the production assembler on an ECP twist cell (geminal + Jastrow
    blocks), vs the production XLA reducer."""
    _compare(rs_ham(pattern), "ln_Psi")


@pytest.mark.heavy
@pytest.mark.parametrize("with_jastrow", [False, True])
def test_complex_driver_delc_matches_xla(with_jastrow):
    """dE_L/dc through the production assembler on the AE twist cell (no ECP mesh):
    the kinetic reduction with and without a Jastrow (the with-Jastrow case exercises
    ``grad_T_mo_withJ_complex``'s ``-sum gJ.gD`` cross term)."""
    _compare(_build_ham_for_cell("H2_pbc_ae_k025_sphe.h5", with_jastrow, None, "SD"), "e_L")
