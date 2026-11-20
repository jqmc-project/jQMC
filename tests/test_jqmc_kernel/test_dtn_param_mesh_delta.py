"""Weighted one-electron-move DTN parameter gradients over the ECP mesh (issue #566).

The ECP non-local dV_nl/dc of the local-energy parameter gradient needs, for the DTN
parameters, ``sum_g W_g [grad_dtn_wrt_param(config_g) - grad_dtn_wrt_param(base)]`` over the
one-electron-moved quadrature configurations. The dense O(N_e^2 N_atom) kernel for every
configuration makes dE_L/dc take 22 times e_L for Si8 on a GPU;
``dtn.dtn_param_mesh_weighted`` evaluates only the moved electron's pairs, legs and triples.

the weighted sum equals the dense per-config recomputation minus the base, with
   complex weights, for molecular and periodic H2 with every channel preset, and for an f pair
   across the periodic cell boundary.
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

from jqmc._precision import get_tolerance
from jqmc.derivatives import dtn
from tests.test_jqmc_kernel.test_jastrow_dtn import _build_dtn, _draw_near_nuclei, _h2_pbc_structure, _h2_structure

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


def _lattice_mesh(all_e: np.ndarray, alat: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    shifts = alat * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    pts = np.concatenate([all_e[m][None, :] + shifts for m in range(all_e.shape[0])])
    return pts, np.repeat(np.arange(all_e.shape[0]), shifts.shape[0])


def _dense_weighted(d, r_up, r_dn, pts, e_idx, weights) -> dict[str, jax.Array]:
    """``sum_c w_c [grad_dtn_wrt_param(config_c) - grad_dtn_wrt_param(base)]`` with the dense kernel."""
    r_up = jnp.asarray(r_up, jnp.float64)
    r_dn = jnp.asarray(r_dn, jnp.float64)
    n_up, n_dn = r_up.shape[0], r_dn.shape[0]
    base = dtn.grad_dtn_wrt_param(d, r_up, r_dn)

    def one(pt, ei):
        ru = jnp.where((jnp.arange(n_up) == ei)[:, None], pt[None, :], r_up)
        rd = jnp.where((jnp.arange(n_dn) == ei - n_up)[:, None], pt[None, :], r_dn)
        g = dtn.grad_dtn_wrt_param(d, ru, rd)
        return {k: g[k] - base[k] for k in g}

    per = jax.vmap(one)(jnp.asarray(pts, jnp.float64), jnp.asarray(e_idx, jnp.int32))
    w = jnp.asarray(weights)
    return {k: jnp.einsum("c,c...->...", w, v) for k, v in per.items()}


def _check(d, r_up, r_dn, pts, e_idx, weights):
    got = dtn.dtn_param_mesh_weighted(
        d, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(pts), jnp.asarray(e_idx), jnp.asarray(weights)
    )
    ref = _dense_weighted(d, r_up, r_dn, pts, e_idx, weights)
    assert set(got) == set(ref)
    atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
    for k in ref:
        np.testing.assert_allclose(np.asarray(got[k]), np.asarray(ref[k]), atol=atol, rtol=rtol, err_msg=k)
        assert np.max(np.abs(np.asarray(ref[k]))) > 100 * atol, f"{k} is vacuous"


@pytest.mark.parametrize("pbc", [False, True])
@pytest.mark.parametrize("preset", ["u_chi_f", "u_chi", "u_f"])
def test_param_mesh_weighted_matches_dense_per_config_oracle(pbc, preset):
    """The weighted rank-one DTN parameter gradients equal the dense per-config recomputation."""
    sd = _h2_pbc_structure() if pbc else _h2_structure()
    d = _build_dtn(sd, preset=preset, cutoff_smoothness=4, seed=37)
    r_up, r_dn = (np.asarray(x) for x in _draw_near_nuclei(sd, 3, 2, seed=41, spread=0.7))
    pts, e_idx = _lattice_mesh(np.concatenate([r_up, r_dn]))
    rng = np.random.default_rng(566)
    weights = rng.normal(size=len(pts)) + 1j * rng.normal(size=len(pts))
    _check(d, r_up, r_dn, pts, e_idx, weights)
    jax.clear_caches()


def test_param_mesh_weighted_matches_dense_across_the_cell_boundary():
    """An f pair on opposite faces of the periodic cell uses the dense kernel's minimum images."""
    sd = _h2_pbc_structure()
    d = _build_dtn(sd, preset="u_chi_f", cutoff_smoothness=4, seed=47)
    r_up = np.array([[9.9, 2.0, 2.0], [0.1, 2.0, 2.0]])
    r_dn = np.array([[2.5, 2.3, 2.1]])
    pts = np.array([[9.8, 2.0, 2.0], [0.2, 2.1, 2.0], [2.4, 2.2, 2.0]])
    e_idx = np.array([0, 1, 2])
    _check(d, r_up, r_dn, pts, e_idx, np.array([0.7 - 0.2j, -1.1 + 0.4j, 0.3 + 0.9j]))
