"""The geminal / J3 c-axis blocks at a FAR-ROAMED periodic walker.

jqmc's ``compute_AOs`` evaluates every periodic AO at the HOME-CELL representative; free
c-axis kernels that re-evaluate the AO stack from the RAW walker coordinate disagree with
it. Since dln_Psi/dc and dE_L/dc are invariant under shifting an electron by a whole
lattice vector, wrapping the INPUT (``_rs_wrap_home_cell``, applied inside
``_rs_native_c_param_grads``) is exact, so the native route serves periodic
``opt_geminal`` / ``opt_J3_param`` / DTN families without a capability gate.

These rows pin that:

  * far-roamed parity: every native c-axis block must equal the production XLA block for
    a walker displaced by whole lattice vectors (without the wrap the error is rel 1.0 --
    i.e. 100% wrong);
  * lattice-shift invariance of the NATIVE blocks themselves (XLA's are invariant to
    ~1e-15), which is the structural property the wrap restores and which fails loudly
    if a future kernel re-introduces raw-coordinate AO evaluation.

The DTN family is included: min-image Jastrow families are wrap-invariant by
construction (measured ~1e-14 far/near).
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
_ZONES = ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic", "jastrow_grad_lap", "coulomb")

# Periodic pGTO cells carrying the documented far-image truncation floor, the same
# route-parity floor ``test_ungate_leaf_param_deriv`` / ``test_mcmc_observables`` use.
_PATTERNS = ["pbc_gamma_sphepgto_j12", "pbc_gamma_sphepgto_j123", "pbc_gamma_sphepgto_jdtn"]


def _tol():
    return get_tolerance_xla_rust_consistency(_ZONES, level="medium")


def _mask():
    """Every family the old ``_rs_pd_config_ok`` excluded on a periodic run."""
    mask = dict(_MASK_DEFAULTS)
    for kw in ("opt_geminal", "opt_J3_param", "opt_JDTN_u_param", "opt_JDTN_chi_param", "opt_JDTN_f_param"):
        mask[kw] = True
    return mask


def _lattice(ham):
    sd = ham.structure_data
    return np.stack([np.asarray(sd.vec_a, float), np.asarray(sd.vec_b, float), np.asarray(sd.vec_c, float)])


def _batch(ham, nw=2, seed=0, scale=0.6, shift_cells=0):
    """A walker batch; ``shift_cells > 0`` displaces every electron by a random whole
    number of lattice vectors (the FAR-roamed regime)."""
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[rng.integers(0, len(R), (nw, nu))] + rng.normal(scale=scale, size=(nw, nu, 3))
    rd = R[rng.integers(0, len(R), (nw, nd))] + rng.normal(scale=scale, size=(nw, nd, 3))
    if shift_cells:
        A = _lattice(ham)
        ru = ru + rng.integers(-shift_cells, shift_cells + 1, (nw, nu, 3)) @ A
        rd = rd + rng.integers(-shift_cells, shift_cells + 1, (nw, nd, 3)) @ A
    return np.ascontiguousarray(ru), np.ascontiguousarray(rd)


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
        _, g, _, _ = _geminal_inv_batched(wf.geminal_data, ru[None], rd[None])
        ginv = jnp.asarray(g[0])
        if quantity == "ln_Psi":
            per.append(assemble.compute_grad_ln_Psi(ham, ru, rd, wrt="c", mask=mask, geminal_inv=ginv))
        else:
            per.append(assemble.compute_grad_e_L(ham, ru, rd, np.eye(3), wrt="c", mask=mask, geminal_inv=ginv))
    gb = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *per)
    flat = _reduce_c_param_grads(wf, gb, nw, mask)
    return {n: v for n, v in flat.items() if mask.get(_PARAM_BLOCK_TO_MASK_KW.get(n, ""), True)}


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", _PATTERNS)
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_far_roamed_periodic_c_axis_matches_xla(pattern, quantity):
    """Every native c-axis block equals the production XLA block for a walker displaced
    by whole lattice vectors (the FAR-roamed regime)."""
    ham = rs_ham(pattern)
    mask = _mask()
    ru_b, rd_b = _batch(ham, shift_cells=2)
    native = _native(ham, ru_b, rd_b, mask, quantity)
    ref = _xla(ham, ru_b, rd_b, mask, quantity)
    atol, rtol = _tol()
    assert set(ref) <= set(native), f"native c-axis assembler missing blocks {sorted(set(ref) - set(native))}"
    for b in sorted(ref):
        np.testing.assert_allclose(
            np.real(np.asarray(native[b])), np.real(np.asarray(ref[b])), atol=atol, rtol=rtol, err_msg=f"{quantity} {b}"
        )


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", _PATTERNS)
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_native_c_axis_is_lattice_shift_invariant(pattern, quantity):
    """Structural invariant: shifting every electron by whole lattice vectors must not
    change any native c-axis block (XLA's are invariant to ~1e-15). This is what the
    home-cell wrap restores; raw-coordinate AO evaluation fails it by ~100%."""
    ham = rs_ham(pattern)
    mask = _mask()
    ru_b, rd_b = _batch(ham)
    A = _lattice(ham)
    rng = np.random.default_rng(11)
    ru_s = np.ascontiguousarray(ru_b + rng.integers(-2, 3, ru_b.shape[:2] + (3,)) @ A)
    rd_s = np.ascontiguousarray(rd_b + rng.integers(-2, 3, rd_b.shape[:2] + (3,)) @ A)
    base = _native(ham, ru_b, rd_b, mask, quantity)
    shifted = _native(ham, ru_s, rd_s, mask, quantity)
    atol, rtol = _tol()
    for b in sorted(base):
        np.testing.assert_allclose(
            np.real(np.asarray(shifted[b])), np.real(np.asarray(base[b])), atol=atol, rtol=rtol, err_msg=f"{quantity} {b}"
        )
