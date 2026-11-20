"""The J3 grad/lap cache must wrap periodic pGTO coordinates before the AO seed.

A periodic pGTO leaf's image sum is a static tiling enumerated relative to the
home cell. A raw coordinate several lattice vectors outside the cell can fall
beyond that tiling and miss its dominant images, collapsing the gradient,
Laplacian, and kinetic terms. Electrons must therefore be wrapped into the home
cell before AO evaluation.

The J3 derivative cache is gated on `st.j3_state.is_some()`, so the periodic
raw-AO pGTO J3 ratio state uses it. Its seed `init_j3_grad_lap_cache(orb, r_up, r_dn)`
must therefore receive the wrapped positions the advance path
(`update_j3_grad_lap_cache(..., wrapped_newpos)`) uses, not raw walker positions: the
seed's r_up/r_dn are wrapped with the SAME `crate::mcmc::wrap_rows` primitive
`mesh_fast_state`'s own constructor uses for every other AO-eval input. (For a
molecular case the walker-storage convention, raw positions, equals the AO-eval
convention trivially, since there is no cell to wrap into.)

This test is the regression guard: Rust vs XLA parity for the
periodic pGTO + J3 fixture, force AND stress enabled, over a full step budget
(so both the seed AND at least one advance are exercised).
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

import pytest

from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_gfmc_chain_modes import (
    _STEPS_STAT,
    _assert_deterministic_driver_parity,
    _run_gfmc,
)

pytestmark = [pytest.mark.with_chdir]


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["n", "t"])
def test_j3_grad_cache_wraps_periodic_coordinates(driver):
    """Rust and XLA must agree for periodic pGTO + J3, force AND stress, fast=True.

    Mesh rotation is disabled so both backends walk the same threefry stream
    and make the same projection decisions; any divergence then localises to
    the cached J3 derivative table, not the walk.
    """
    ham = rs_ham("pbc_gamma_sphepgto_j123")
    seed = 34356
    xla = _run_gfmc(driver, ham, seed, "jax", _STEPS_STAT, random_mesh=False, num_walkers=1, pos_deriv=True)
    rust = _run_gfmc(driver, ham, seed, "rust", _STEPS_STAT, random_mesh=False, num_walkers=1, pos_deriv=True)
    _assert_deterministic_driver_parity(rust, xla, driver, position_deriv=True, periodic=True)
