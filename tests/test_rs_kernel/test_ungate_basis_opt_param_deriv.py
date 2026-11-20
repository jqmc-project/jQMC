"""Route-parity net: basis-opt param-derivs
(``opt_lambda_basis_exp`` / ``opt_lambda_basis_coeff``) on the molecular cart-GTO row.

The radial kernels (``grad_ln_det_wrt_radial`` / ``grad_eL_wrt_radial``, pinned vs XLA
in ``test_param_grad.py`` / ``test_param_grad_el.py``) are wired into the driver assembler
``_rs_native_c_param_grads`` and ``NativeObsCtx.pd_{ok,el}_basis``. No capability gate
routes the native c-axis block to JAX when the basis flags are active (what is unported
strict-raises instead), so this driver assembler is the reachable production path on this
molecular fixture, not a bypassed call.

Periodic COMPOSITE (pGTO+PW/B-spline) basis-opt (the composite child layout, parsed
child-wise) has separate route-parity coverage in ``test_rs_analytic_matrix.py`` /
``test_route_parity_*``, so it is not re-litigated here. This file only pins the
molecular cart-GTO row.

This test drives the production assembler with the basis flags on and compares the
emitted block SET (``test_ungate_c3_basis_block_present``) and VALUES
(``test_ungate_c3_basis_block_values``) against the XLA reducer.
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
from jqmc._precision import precision_policy_snapshot

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.derivatives import assemble
from jqmc.derivatives.bridge import _MASK_DEFAULTS, _reduce_c_param_grads
from jqmc.jqmc_mcmc import (
    _geminal_inv_batched,
    _NV_BASE_MESH,
    _PARAM_BLOCK_TO_MASK_KW,
    _rs_native_c_param_grads,
    Nv_default,
)

from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.with_chdir]

_W = np.asarray(_NV_BASE_MESH[Nv_default].weights, float)
_GP = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, float)
_ZONES = ("ao_eval", "mo_eval", "det_eval", "jastrow_eval")
_BASIS_BLOCKS = {"lambda_basis_exp", "lambda_basis_coeff"}


def _batch(ham, nw=2, seed=0, scale=0.6):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = np.ascontiguousarray(R[rng.integers(0, len(R), (nw, nu))] + rng.normal(scale=scale, size=(nw, nu, 3)))
    rd = np.ascontiguousarray(R[rng.integers(0, len(R), (nw, nd))] + rng.normal(scale=scale, size=(nw, nd, 3)))
    return ru, rd


def _ginv1(ham, ru, rd):
    _, g, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    return np.asarray(g[0])


def _basis_mask():
    mask = dict(_MASK_DEFAULTS)
    mask["opt_lambda_basis_exp"] = True
    mask["opt_lambda_basis_coeff"] = True
    return mask


def _native(ham, ru_b, rd_b, mask, quantity):
    wf = ham.wavefunction_data
    cp = ham.coulomb_potential_data
    nw = len(ru_b)
    ctx = jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        cp,
        False,
        bool(ham.structure_data.pbc_flag),
        False,
        want_force=False,
        want_gfmc=False,
        want_param_deriv=True,
        precision_policy=precision_policy_snapshot(),
    )
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
        ginv = jnp.asarray(_ginv1(ham, np.asarray(ru), np.asarray(rd)))
        if quantity == "ln_Psi":
            per.append(assemble.compute_grad_ln_Psi(ham, ru, rd, wrt="c", mask=mask, geminal_inv=ginv))
        else:
            per.append(assemble.compute_grad_e_L(ham, ru, rd, np.eye(3), wrt="c", mask=mask, geminal_inv=ginv))
    gb = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *per)
    flat = _reduce_c_param_grads(wf, gb, nw, mask)
    return {n: v for n, v in flat.items() if mask.get(_PARAM_BLOCK_TO_MASK_KW.get(n, ""), True)}


@pytest.mark.light
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_ungate_c3_basis_block_present(quantity):
    """The native c-axis assembler must emit the ``lambda_basis_exp`` /
    ``lambda_basis_coeff`` blocks that XLA emits when the basis flags are active
    (see module docstring)."""
    ham = rs_ham("mol_cart_sd_j12")  # cart-GTO leaf -> basis-opt is meaningful
    ru_b, rd_b = _batch(ham)
    mask = _basis_mask()
    native = _native(ham, ru_b, rd_b, mask, quantity)
    ref = _xla(ham, ru_b, rd_b, mask, quantity)
    assert _BASIS_BLOCKS <= set(ref), f"XLA reference itself lacks basis blocks: {sorted(ref)}"
    missing = _BASIS_BLOCKS - set(native)
    assert not missing, f"native c-axis assembler ({quantity}) missing basis blocks {sorted(missing)}; emitted {sorted(native)}"


@pytest.mark.light
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_ungate_c3_basis_block_values(quantity):
    """The emitted native basis blocks must equal XLA. Kept separate from the
    block-set check so a VALUE regression is caught here rather than passing on
    presence alone."""
    ham = rs_ham("mol_cart_sd_j12")
    ru_b, rd_b = _batch(ham)
    mask = _basis_mask()
    native = _native(ham, ru_b, rd_b, mask, quantity)
    ref = _xla(ham, ru_b, rd_b, mask, quantity)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    for b in sorted(_BASIS_BLOCKS):
        assert b in native, f"native missing basis block {b} (block-set gap; see companion test)"
        np.testing.assert_allclose(np.real(native[b]), np.real(ref[b]), atol=atol, rtol=rtol, err_msg=f"{quantity} {b}")
