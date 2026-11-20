"""Per-move incremental-vs-full-recompute twin of the GFMC interval potential (periodic all-electron).

The production XLA GFMC projection bodies carry the diagonal
potential across hops (``jqmc.jqmc_gfmc._gfmc_build_interval_pot_state`` /
``_gfmc_advance_interval_pot_state``, default ON; ``JQMC_GFMC_POT_FULL_RECOMPUTE=1``
selects the per-sub-step full recompute). For an all-electron periodic
Hamiltonian the carry has an ``ei_disc`` row channel (the LRDMC ``alat``-floored
electron-ion Ewald rows) that the ECP fixtures of the driver-level twins never
reach.

This module drives that state directly through a fixed schedule of
single-electron moves on ``H4_pbc_ae_cart.h5`` (2 up + 2 down electrons, no
ECP, periodic): both spins, ``alat``-sized hops along the axes, hops that
cross the cell faces, lattice-translated (unwrapped) jumps, and moves that put
an electron inside the ``alat`` floor of a proton. After EVERY move the carried
components are compared with a from-scratch recompute at the same
configuration using the dispatch kernels the full-recompute route calls:

* the ee Ewald energy composed from the carried ``(E_real_half, S_re, S_im)``
  vs ``compute_coulomb_potential_ee_dispatch``;
* the carried per-electron bare ei rows vs ``compute_coulomb_potential_ei_dispatch``;
* the carried per-electron discretized ei rows (``ei_disc``) vs
  ``compute_coulomb_potential_ei_discretized_dispatch``;
* the total diagonal Coulomb term ``V_ee + V_ii + sum(ei_disc)`` of both routes.

All four use ``ATOL_GFMC_INCREMENTAL_TWIN`` / ``RTOL_GFMC_INCREMENTAL_TWIN``
(CMP-GFMC-INCREMENTAL-TWIN-001), in both precision modes (the Ewald zone is fp64
in both). Non-vacuity is asserted by construction: the fixture is periodic
all-electron (so ``ei_disc`` is the live row channel), the carried liveness
counter equals the number of applied moves, both spins move, some moves cross a
cell face, and on some moves the moved electron's ``ei_disc`` row differs from
its bare row (the ``alat`` floor is active), so a row update that wrote the bare
value or skipped the row would be seen.

The Rust leg on the same fixture is
``tests/test_rs_kernel/test_gfmc_incremental_twin_ae_pbc.py``.
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

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._setting import ATOL_GFMC_INCREMENTAL_TWIN, RTOL_GFMC_INCREMENTAL_TWIN
from jqmc.coulomb_potential import (
    compute_coulomb_potential_ee_dispatch,
    compute_coulomb_potential_ei_discretized_dispatch,
    compute_coulomb_potential_ei_dispatch,
    compute_coulomb_potential_ii_dispatch,
    compute_ewald_ee_from_interval_state,
)
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.jqmc_gfmc import (
    _gfmc_advance_interval_pot_state,
    _gfmc_build_interval_pot_state,
    _gfmc_pot_active,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_FIXTURE = "H4_pbc_ae_cart.h5"
_ALAT = 0.30
_N_MOVES = 48
_SEED = 20260927


def _build_ham():
    """``H4_pbc_ae_cart.h5`` (periodic, all-electron) with no Jastrow; only the Coulomb data is used."""
    structure_data, _, _mos, _, geminal_mo_data, coulomb_potential_data = read_trexio_file(
        trexio_file=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "trexio_example_files",
            _FIXTURE,
        ),
        store_tuple=True,
    )
    wf = Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=geminal_mo_data)
    ham = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wf,
    )
    ham.sanity_check()
    return ham


def _move_schedule(ham):
    """Deterministic list of ``(is_up, index, new_position)`` single-electron moves and the start config.

    Cycles four move kinds so every kind hits both spins: an ``alat`` hop along
    a random axis (the production GFMC move), a hop that carries the electron
    across a cell face (fractional coordinate leaves [0, 1)), a lattice-vector
    translation plus a small hop (unwrapped walker), and a placement at distance
    ``alat / 3`` from a random proton (inside the discretization floor).
    """
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    inv_cell = np.linalg.inv(cell)
    nuclei = np.asarray(ham.structure_data.positions, dtype=np.float64)
    gd = ham.wavefunction_data.geminal_data
    n_up, n_dn = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(_SEED)
    r_up = rng.random((n_up, 3)) @ cell
    r_dn = rng.random((n_dn, 3)) @ cell
    pos = {True: r_up.copy(), False: r_dn.copy()}
    moves = []
    for k in range(_N_MOVES):
        is_up = (k // 4) % 2 == 0
        idx = int(rng.integers(0, n_up if is_up else n_dn))
        old = pos[is_up][idx]
        kind = k % 4
        if kind == 0:
            step = np.zeros(3)
            step[int(rng.integers(0, 3))] = _ALAT * rng.choice([-1.0, 1.0])
            new = old + step
        elif kind == 1:
            axis = int(rng.integers(0, 3))
            frac = old @ inv_cell
            # Land just past the nearest face along ``axis``.
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
        pos[is_up][idx] = new
        moves.append((is_up, idx, new.copy()))
    return r_up, r_dn, moves, inv_cell


def _full_recompute(ham, r_up, r_dn):
    """From-scratch components at one configuration: the dispatch calls of the full-recompute route."""
    cp = ham.coulomb_potential_data
    ee = compute_coulomb_potential_ee_dispatch(coulomb_potential_data=cp, r_up_carts=r_up, r_dn_carts=r_dn)
    ei_up, ei_dn = compute_coulomb_potential_ei_dispatch(coulomb_potential_data=cp, r_up_carts=r_up, r_dn_carts=r_dn)
    disc_up, disc_dn = compute_coulomb_potential_ei_discretized_dispatch(
        coulomb_potential_data=cp, r_up_carts=r_up, r_dn_carts=r_dn, alat=_ALAT
    )
    ii = compute_coulomb_potential_ii_dispatch(coulomb_potential_data=cp)
    ei_bare = np.concatenate([np.asarray(ei_up), np.asarray(ei_dn)])
    ei_disc = np.concatenate([np.asarray(disc_up), np.asarray(disc_dn)])
    total = float(ee) + float(ii) + float(np.sum(ei_disc))
    return {"ee": float(ee), "ei_bare": ei_bare, "ei_disc": ei_disc, "total": total}


def _carried(ham, state, n_e):
    """Components read from the carried interval state (the incremental route's inputs)."""
    cp = ham.coulomb_potential_data
    e_real_half, s_re, s_im, ei_bare, ei_disc, _ecp_local, _n = state
    ee = compute_ewald_ee_from_interval_state(e_real_half, s_re, s_im, n_e, cp.ewald_data)
    ii = compute_coulomb_potential_ii_dispatch(coulomb_potential_data=cp)
    ei_disc = np.asarray(ei_disc)
    total = float(ee) + float(ii) + float(np.sum(ei_disc))
    return {"ee": float(ee), "ei_bare": np.asarray(ei_bare), "ei_disc": ei_disc, "total": total}


def _assert_state_matches_full(carried, full, step):
    """One same-configuration comparison of every carried component against the full recompute."""
    for component in ("ee", "ei_bare", "ei_disc", "total"):
        np.testing.assert_allclose(
            carried[component],
            full[component],
            atol=ATOL_GFMC_INCREMENTAL_TWIN,
            rtol=RTOL_GFMC_INCREMENTAL_TWIN,
            err_msg=f"incremental vs full-recompute interval potential: {component} after move {step}",
        )


def _drive(ham, advance):
    """Advance the interval state through the move schedule, comparing after every move.

    Returns the non-vacuity witnesses: number of applied moves, the carried
    liveness counter, moves per spin, face crossings, and moves on which the
    moved electron's ``ei_disc`` row differs from its bare row.
    """
    r_up, r_dn, moves, inv_cell = _move_schedule(ham)
    r_up = jnp.asarray(r_up)
    r_dn = jnp.asarray(r_dn)
    n_up = r_up.shape[0]
    n_e = n_up + r_dn.shape[0]
    state = _gfmc_build_interval_pot_state(ham, r_up, r_dn, _ALAT)
    _assert_state_matches_full(_carried(ham, state, n_e), _full_recompute(ham, r_up, r_dn), step=0)
    witness = {"up": 0, "dn": 0, "face_crossings": 0, "floor_active": 0}
    for step, (is_up, idx, new) in enumerate(moves, start=1):
        new = jnp.asarray(new)
        if is_up:
            old = r_up[idx]
            new_up, new_dn = r_up.at[idx].set(new), r_dn
        else:
            old = r_dn[idx]
            new_up, new_dn = r_up, r_dn.at[idx].set(new)
        state = advance(
            ham,
            state,
            r_up,
            r_dn,
            new_up,
            new_dn,
            jnp.asarray(bool(is_up)),
            jnp.asarray(idx if is_up else 0),
            jnp.asarray(not is_up),
            jnp.asarray(0 if is_up else idx),
            _ALAT,
        )
        r_up, r_dn = new_up, new_dn
        full = _full_recompute(ham, r_up, r_dn)
        _assert_state_matches_full(_carried(ham, state, n_e), full, step=step)
        gi = idx if is_up else n_up + idx
        witness["up" if is_up else "dn"] += 1
        if np.any(np.floor(np.asarray(old) @ inv_cell) != np.floor(np.asarray(new) @ inv_cell)):
            witness["face_crossings"] += 1
        if abs(full["ei_disc"][gi] - full["ei_bare"][gi]) > ATOL_GFMC_INCREMENTAL_TWIN:
            witness["floor_active"] += 1
    witness["moves"] = len(moves)
    witness["n_updates"] = int(round(float(state[6])))
    return witness


def _skip_ei_disc_row(ham, state, r_up, r_dn, new_up, new_dn, has_up, up_i, has_dn, dn_i, alat):
    """Mutated advance for the negative control: the moved electron's ``ei_disc`` row is left stale."""
    new_state = _gfmc_advance_interval_pot_state(ham, state, r_up, r_dn, new_up, new_dn, has_up, up_i, has_dn, dn_i, alat)
    return new_state[:4] + (state[4],) + new_state[5:]


@pytest.mark.light
def test_ae_pbc_interval_pot_state_matches_full_recompute_after_every_move():
    """XLA carried interval potential == from-scratch recompute after every single-electron move (AE, PBC)."""
    ham = _build_ham()
    cp = ham.coulomb_potential_data
    assert cp.ewald_data is not None and not cp.ecp_flag, "fixture must be periodic all-electron"
    assert _gfmc_pot_active(ham), "incremental interval potential is not the active route"
    witness = _drive(ham, _gfmc_advance_interval_pot_state)
    assert witness["n_updates"] == witness["moves"] == _N_MOVES, witness
    assert witness["up"] > 0 and witness["dn"] > 0, witness
    assert witness["face_crossings"] >= _N_MOVES // 4, witness
    assert witness["floor_active"] >= _N_MOVES // 8, witness


@pytest.mark.light
def test_ae_pbc_interval_pot_state_twin_detects_a_stale_ei_disc_row():
    """Negative control: skipping the ``ei_disc`` row update of the moved electron fails the ei_disc comparison."""
    ham = _build_ham()
    with pytest.raises(AssertionError, match=r"incremental vs full-recompute interval potential: ei_disc after move 1\b"):
        _drive(ham, _skip_ei_disc_row)
