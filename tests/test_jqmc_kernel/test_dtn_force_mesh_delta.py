"""Rank-one DTN Pulay-force deltas over a one-electron-move mesh (issue #558).

``one_move_mesh.mesh_force`` needs, for every one-electron-moved ECP mesh configuration,
the DTN Pulay-force delta ``grad_dtn_wrt_R(config) - grad_dtn_wrt_R(base)``. The dense
``O(N_e^2 N_atom)`` kernel per configuration would dominate the GPU time of MCMC with atomic
forces. Only the chi leg and the f triples of the moved electron change, so ``dtn.dtn_force_mesh_delta`` evaluates those terms alone.

per configuration, the delta equals the dense oracle (``grad_dtn_wrt_R`` at the
   moved configuration minus the base) for molecular and periodic H2 with every channel
   preset, and for an f pair across the periodic cell boundary.
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


def _dense_force_deltas(d, r_up, r_dn, pts, e_idx) -> jax.Array:
    """``grad_dtn_wrt_R(config_c) - grad_dtn_wrt_R(base)`` per config, with the dense kernel."""
    r_up = jnp.asarray(r_up, jnp.float64)
    r_dn = jnp.asarray(r_dn, jnp.float64)
    n_up, n_dn = r_up.shape[0], r_dn.shape[0]
    base = dtn.grad_dtn_wrt_R(d, r_up, r_dn)

    def one(pt, ei):
        ru = jnp.where((jnp.arange(n_up) == ei)[:, None], pt[None, :], r_up)
        rd = jnp.where((jnp.arange(n_dn) == ei - n_up)[:, None], pt[None, :], r_dn)
        return dtn.grad_dtn_wrt_R(d, ru, rd) - base

    return jax.vmap(one)(jnp.asarray(pts, jnp.float64), jnp.asarray(e_idx, jnp.int32))


@pytest.mark.parametrize("pbc", [False, True])
@pytest.mark.parametrize("preset", ["u_chi_f", "u_chi", "u_f"])
def test_force_mesh_delta_matches_dense_per_config_oracle(pbc, preset):
    """Every per-config rank-one DTN force delta equals the dense recomputation minus the base."""
    sd = _h2_pbc_structure() if pbc else _h2_structure()
    d = _build_dtn(sd, preset=preset, cutoff_smoothness=4, seed=37)
    r_up, r_dn = (np.asarray(x) for x in _draw_near_nuclei(sd, 3, 2, seed=41, spread=0.7))
    pts, e_idx = _lattice_mesh(np.concatenate([r_up, r_dn]))

    got = np.asarray(dtn.dtn_force_mesh_delta(d, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(pts), jnp.asarray(e_idx)))
    ref = np.asarray(_dense_force_deltas(d, r_up, r_dn, pts, e_idx))

    atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
    assert got.shape == ref.shape == (len(pts), sd.positions.shape[0], 3)
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol)
    assert np.max(np.abs(ref)) > 100 * atol  # the nuclear channels are not vacuous
    jax.clear_caches()


def test_force_mesh_delta_matches_dense_across_the_cell_boundary():
    """A moved electron and its f partner on opposite faces of the cell use the dense kernel's minimum images."""
    sd = _h2_pbc_structure()
    d = _build_dtn(sd, preset="u_chi_f", cutoff_smoothness=4, seed=47)
    r_up = np.array([[9.9, 2.0, 2.0], [0.1, 2.0, 2.0]])
    r_dn = np.array([[2.5, 2.3, 2.1]])
    pts = np.array([[9.8, 2.0, 2.0], [0.2, 2.1, 2.0], [2.4, 2.2, 2.0]])
    e_idx = np.array([0, 1, 2])

    got = np.asarray(dtn.dtn_force_mesh_delta(d, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(pts), jnp.asarray(e_idx)))
    ref = np.asarray(_dense_force_deltas(d, r_up, r_dn, pts, e_idx))

    atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol)
    assert np.max(np.abs(ref)) > 100 * atol
