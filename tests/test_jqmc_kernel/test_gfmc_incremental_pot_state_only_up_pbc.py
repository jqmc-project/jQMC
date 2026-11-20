"""Per-move incremental-vs-full-recompute twin of the GFMC interval potential with no down electrons (PBC).

Issue #73. ``jqmc.jqmc_gfmc._gfmc_advance_interval_pot_state`` has a static
branch for an empty down-spin channel (``r_dn_carts.shape[0] == 0``) that picks
the moved electron's position and its global row index ``gi`` in the carried
Ewald ee state and the per-electron e-i rows. The two-spin twin
(``test_gfmc_incremental_pot_state_ae_pbc.py``, H4_pbc 2 up + 2 down) never
takes that branch, and the ``n_up = 1`` fixtures cannot tell a wrong row index
from the right one (every row index is 0).

This module replaces the geminal of ``H4_pbc_ae_cart.h5`` (periodic,
all-electron, no Jastrow) with a Slater determinant of ``n_up >= 2`` up
electrons and no down electrons, then drives the production carry through a
fixed schedule of single-up-electron moves (``alat`` hops, hops across a cell
face, lattice-translated unwrapped jumps, and placements inside the ``alat``
floor of a proton). After EVERY move the carried ee energy, bare e-i rows,
``alat``-floored e-i rows and the total diagonal Coulomb term are compared with
a from-scratch recompute at the same configuration (the dispatch kernels of the
full-recompute route; no carry), with ``ATOL_GFMC_INCREMENTAL_TWIN`` /
``RTOL_GFMC_INCREMENTAL_TWIN`` in both precision modes (the Ewald zone is fp64
in both). Non-vacuity: every up electron moves, at least half of the moves
are of an electron with index != 0, the carried liveness counter equals the
number of moves, and the schedule crosses cell faces and activates the floor.

A negative control rebuilds the production advance with the only-up row index
replaced by 0 (the surviving mutation reported in #73) and requires the twin
to fail.
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

import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import jqmc_gfmc
from jqmc._setting import ATOL_GFMC_INCREMENTAL_TWIN
from jqmc.determinant import SD_data
from jqmc.jqmc_gfmc import (
    _gfmc_advance_interval_pot_state,
    _gfmc_build_interval_pot_state,
    _gfmc_pot_active,
)
from jqmc.molecular_orbital import MOs_data
from tests.test_jqmc_kernel.test_gfmc_incremental_pot_state_ae_pbc import (
    _ALAT,
    _assert_state_matches_full,
    _build_ham,
    _carried,
    _full_recompute,
)

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_N_MOVES = 32
_SEED = 20261012


def _build_only_up_ham(n_up):
    """H4_pbc_ae_cart.h5 Coulomb data with an ``n_up``-up / 0-down Slater determinant (identity MO rows)."""
    ham = _build_ham()
    aos = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
    assert aos.num_ao >= n_up
    mos = MOs_data.build(aos_data=aos, mo_coefficients=jnp.eye(aos.num_ao)[:n_up])
    geminal = SD_data.build(num_electron_up=n_up, num_electron_dn=0, mos_up=mos, mos_dn=mos, time_reversal=False)
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=geminal))
    ham.sanity_check()
    return ham


def _only_up_move_schedule(ham, n_up):
    """Deterministic ``(index, new_position)`` up-electron moves; four kinds cycle over every electron."""
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    inv_cell = np.linalg.inv(cell)
    nuclei = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(_SEED)
    r_up = rng.random((n_up, 3)) @ cell
    pos = r_up.copy()
    moves = []
    for k in range(_N_MOVES):
        # (k // 4 + k) decorrelates the moved electron from the move kind (k % 4).
        idx = (k // 4 + k) % n_up
        old = pos[idx]
        kind = k % 4
        if kind == 0:
            step = np.zeros(3)
            step[int(rng.integers(0, 3))] = _ALAT * rng.choice([-1.0, 1.0])
            new = old + step
        elif kind == 1:
            axis = int(rng.integers(0, 3))
            frac = old @ inv_cell
            target = np.floor(frac[axis]) + (1.0 if frac[axis] % 1.0 >= 0.5 else 0.0)
            frac_new = frac.copy()
            frac_new[axis] = target + (0.02 if target > frac[axis] else -0.02)
            new = frac_new @ cell
        elif kind == 2:
            shift = rng.integers(-1, 2, size=3)
            shift[int(rng.integers(0, 3))] = rng.choice([-1, 1])
            new = old + shift @ cell + 0.1 * _ALAT * rng.standard_normal(3)
        else:
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            nucleus = nuclei[int(rng.integers(0, nuclei.shape[0]))]
            new = nucleus + int(rng.integers(-1, 2)) * cell[0] + (_ALAT / 3.0) * direction
        pos[idx] = new
        moves.append((idx, new.copy()))
    return r_up, moves, inv_cell


def _drive_only_up(ham, n_up, advance):
    """Advance the carry through the up-only schedule, comparing with the full recompute after every move."""
    r_up, moves, inv_cell = _only_up_move_schedule(ham, n_up)
    r_up = jnp.asarray(r_up)
    r_dn = jnp.zeros((0, 3), dtype=jnp.float64)
    state = _gfmc_build_interval_pot_state(ham, r_up, r_dn, _ALAT)
    _assert_state_matches_full(_carried(ham, state, n_up), _full_recompute(ham, r_up, r_dn), step=0)
    witness = {"moved": set(), "nonzero_index": 0, "face_crossings": 0, "floor_active": 0}
    for step, (idx, new) in enumerate(moves, start=1):
        old = r_up[idx]
        new_up = r_up.at[idx].set(jnp.asarray(new))
        state = advance(
            ham,
            state,
            r_up,
            r_dn,
            new_up,
            r_dn,
            jnp.asarray(True),
            jnp.asarray(idx),
            jnp.asarray(False),
            jnp.asarray(0),
            _ALAT,
        )
        r_up = new_up
        full = _full_recompute(ham, r_up, r_dn)
        _assert_state_matches_full(_carried(ham, state, n_up), full, step=step)
        witness["moved"].add(idx)
        witness["nonzero_index"] += int(idx != 0)
        if np.any(np.floor(np.asarray(old) @ inv_cell) != np.floor(np.asarray(new) @ inv_cell)):
            witness["face_crossings"] += 1
        if abs(full["ei_disc"][idx] - full["ei_bare"][idx]) > ATOL_GFMC_INCREMENTAL_TWIN:
            witness["floor_active"] += 1
    witness["moves"] = len(moves)
    witness["n_updates"] = int(round(float(state[6])))
    return witness


def _advance_with_only_up_row_zero():
    """Production advance rebuilt with the only-up branch's ``gi = up_index`` replaced by ``gi = 0`` (issue #73)."""
    source = inspect.getsource(_gfmc_advance_interval_pot_state)
    old = "        gi = up_index\n"
    assert source.count(old) == 1, "the only-up branch row index moved; update the negative control"
    tree = ast.parse(source.replace(old, "        gi = 0\n"))
    namespace = dict(vars(jqmc_gfmc))
    exec(compile(tree, "<only-up-row-index-mutation>", "exec"), namespace)
    return namespace["_gfmc_advance_interval_pot_state"]


@pytest.mark.light
@pytest.mark.parametrize("n_up", [2, 3])
def test_only_up_pbc_interval_pot_state_matches_full_recompute_after_every_move(n_up):
    """XLA carried interval potential == from-scratch recompute after every up move (n_dn = 0, n_up >= 2, AE PBC)."""
    ham = _build_only_up_ham(n_up)
    cp = ham.coulomb_potential_data
    assert cp.ewald_data is not None and not cp.ecp_flag, "fixture must be periodic all-electron"
    assert _gfmc_pot_active(ham), "incremental interval potential is not the active route"
    gd = ham.wavefunction_data.geminal_data
    assert (gd.num_electron_up, gd.num_electron_dn) == (n_up, 0)
    witness = _drive_only_up(ham, n_up, _gfmc_advance_interval_pot_state)
    assert witness["n_updates"] == witness["moves"] == _N_MOVES, witness
    assert witness["moved"] == set(range(n_up)), witness
    assert witness["nonzero_index"] >= _N_MOVES // 2, witness
    assert witness["face_crossings"] >= _N_MOVES // 4, witness
    assert witness["floor_active"] >= _N_MOVES // 8, witness


@pytest.mark.light
def test_only_up_pbc_interval_pot_state_twin_detects_a_wrong_moved_row_index():
    """Negative control: writing the moved up electron into row 0 (n_dn = 0 branch) fails the twin."""
    ham = _build_only_up_ham(3)
    with pytest.raises(AssertionError, match=r"incremental vs full-recompute interval potential: (ee|ei_bare|ei_disc|total)"):
        _drive_only_up(ham, 3, _advance_with_only_up_row_zero())
