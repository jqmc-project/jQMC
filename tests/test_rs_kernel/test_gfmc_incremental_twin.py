"""Incremental-vs-full-recompute twin for the GFMC projection interval (Rust).

IMPL-GFMC-INCREMENTAL-TWIN-001. The production interval carries the diagonal potential
across hops -- the Ewald ee structure-factor state plus per-electron ei /
local-ECP rows, each advanced by a one-electron delta after the categorical
hop. The ``debug_full_recompute=True`` route disables the carried state and
recomputes every diagonal piece per sub-step (the debug oracle).

The twin contract has three legs, all on the SAME backend and the SAME pinned
threefry key:

1. **Route-invariant walk** (exact): ``b_L`` and the hop cdf never touch the
   carried potential, so the sub-step count, the final walker, and the final
   key are BYTE-identical between the two routes.
2. **Tight-absolute diagonal agreement**: the interval ``e_L`` / ``w_L`` /
   ``V_diag`` agree within ``ATOL_GFMC_INCREMENTAL_TWIN`` (the two routes are
   exact modulo fp reordering of the ee real-space sum and the local-ECP
   channel/electron-major fold).
3. **Liveness (anti-vacuity)**: the production route reports
   ``n_pot_updates == (number of applied hops)`` (> 0 on multi-sub-step
   intervals) while the twin reports 0 -- a test cannot pass by silently
   falling back to full recompute. This covers both PBC Ewald rows and the
   molecular one-electron bare-Coulomb delta route.
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

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from jqmc._setting import (
    ATOL_GFMC_INCREMENTAL_TWIN,
    RTOL_GFMC_INCREMENTAL_TWIN,
)

from tests.test_rs_kernel.test_gfmc_projection import _MESH_GP, _MESH_W, _cell_state, _ctx

pytestmark = [pytest.mark.with_chdir]

# PBC and molecular rows both exercise the carried state (liveness REQUIRED).
# ``h4_skew_ae`` is the periodic all-electron H4 cell on
# the strongly skewed triclinic basis of tests/_gfmc_twin_fixtures.py: the
# carried ee state goes through the general (non-orthogonal) nearest-image
# search and the discretized ei rows are carried. Its start is unwrapped
# (``far_start``: each electron translated by its own lattice vector).
_TWIN_CELLS = ["pbc_gamma", "pbc_gamma_far", "mol_ecp_tmove", "h4_skew_ae"]


def _twin_cell(cellname):
    """``(ham, r_up, r_dn, non_local_move)`` for a twin cell."""
    if cellname == "h4_skew_ae":
        from tests._gfmc_twin_fixtures import far_start, skewed_h4_ham

        ham = skewed_h4_ham(twist=False)
        r_up, r_dn = far_start(ham)
        return ham, r_up, r_dn, "tmove"
    ham, r_up, r_dn, nlm = _cell_state(cellname)
    return ham, r_up, r_dn, nlm


def _twin_pair_t(cellname, tau):
    ham, r_up, r_dn, nlm = _twin_cell(cellname)
    ctx = _ctx(ham)
    key = jax.random.PRNGKey(7)
    kw = dict(random_mesh=False, dltmove=(nlm == "dltmove"), nn=1)
    prod = ctx.run_gfmc_projection_t(r_up, r_dn, 0.30, tau, int(key[0]), int(key[1]), _MESH_W, _MESH_GP, **kw)
    twin = ctx.run_gfmc_projection_t(
        r_up,
        r_dn,
        0.30,
        tau,
        int(key[0]),
        int(key[1]),
        _MESH_W,
        _MESH_GP,
        debug_full_recompute=True,
        **kw,
    )
    return prod, twin


def _twin_pair_n(cellname, nsub):
    ham, r_up, r_dn, nlm = _twin_cell(cellname)
    ctx = _ctx(ham)
    key = jax.random.PRNGKey(7)
    kw = dict(random_mesh=False, dltmove=(nlm == "dltmove"), nn=1)
    args = (r_up, r_dn, 0.30, -1.0, nsub, int(key[0]), int(key[1]), np.zeros((0, 3, 3)), _MESH_W, _MESH_GP)
    prod = ctx.run_gfmc_projection_n(*args, **kw)
    twin = ctx.run_gfmc_projection_n(*args, debug_full_recompute=True, **kw)
    return prod, twin


def _assert_walk_identical(prod, twin):
    """Leg 1: the walk is route-invariant, byte for byte."""
    assert int(prod["n_substeps"]) == int(twin["n_substeps"])
    np.testing.assert_array_equal(np.asarray(prod["r_up"]), np.asarray(twin["r_up"]))
    np.testing.assert_array_equal(np.asarray(prod["r_dn"]), np.asarray(twin["r_dn"]))
    assert list(np.asarray(prod["final_key"], dtype=np.uint32)) == list(np.asarray(twin["final_key"], dtype=np.uint32))


def _assert_diag_close(prod, twin, keys):
    """Leg 2: diagonal interval outputs within the tight absolute twin pin."""
    for k in keys:
        np.testing.assert_allclose(
            float(np.real(prod[k])),
            float(np.real(twin[k])),
            atol=ATOL_GFMC_INCREMENTAL_TWIN,
            rtol=RTOL_GFMC_INCREMENTAL_TWIN,
            err_msg=f"incremental vs full-recompute twin: {k}",
        )


@pytest.mark.light
@pytest.mark.parametrize("cell", _TWIN_CELLS)
def test_gfmc_t_incremental_twin(cell):
    """GFMC_t escape-time interval: incremental == full-recompute twin + liveness."""
    prod, twin = _twin_pair_t(cell, tau=0.20)
    _assert_walk_identical(prod, twin)
    _assert_diag_close(prod, twin, ("e_L", "w_L"))
    nsub = int(prod["n_substeps"])
    # Applied hops = every sub-step but the tau-exhausted last one.
    assert int(prod["n_pot_updates"]) == nsub - 1
    assert nsub - 1 > 0, "vacuous interval: raise tau so the twin exercises hops"
    assert int(twin["n_pot_updates"]) == 0


@pytest.mark.light
@pytest.mark.parametrize("cell", _TWIN_CELLS)
def test_gfmc_n_incremental_twin(cell):
    """GFMC_n fixed-count interval: incremental == full-recompute twin + liveness."""
    nsub = 3
    prod, twin = _twin_pair_n(cell, nsub)
    _assert_walk_identical(prod, twin)
    _assert_diag_close(prod, twin, ("e_L", "w_L", "V_diag"))
    # GFMC_n hops unconditionally: one applied update per sub-step.
    assert int(prod["n_pot_updates"]) == nsub
    assert int(twin["n_pot_updates"]) == 0
