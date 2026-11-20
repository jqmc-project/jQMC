"""Per-sub-step incremental-vs-full-recompute twin of the Rust GFMC_n interval (periodic all-electron).

Rust leg. The native interval carries the diagonal potential
(``IntervalPotState``: Ewald ee structure factor plus per-electron bare and
discretized ei rows) across hops; ``debug_full_recompute=True`` recomputes it
per sub-step. The existing twin cells are all ECP, where the discretized-row
channel is absent. Here the fixture is ``H4_pbc_ae_cart.h5`` (periodic,
all-electron), and ``run_gfmc_projection_n`` is driven with ``nsub = 1 .. K``
from one key: each run follows the same walk one sub-step further, so the
``w_L`` / ``V_diag`` / ``e_L`` of run ``k`` are built from the carried state
through ``k`` hops, compared with the twin at the tight absolute twin pin. The
walk is byte-identical and the liveness counter equals ``nsub`` (zero on the twin).

The discretized rows enter the propagation ``V_diag`` only for an electron whose
kinetic elements flip sign and whose ``ei_disc`` row exceeds its ZV value. The
start configuration puts each same-spin pair on the two sides of a proton, 0.1
bohr away (inside the ``alat`` floor, across the exchange node), and the test
counts, with the XLA kernels at every returned configuration, the sub-steps on
which an electron that has MOVED (so its row is a carried incremental update, not
the build row) has its ``ei_disc`` row selected into ``V_diag`` (non-vacuity of
the channel on the Rust side). On this walk the moved electrons with a selected
row are outside the floor, where the discretized and bare rows coincide; the
floor-active row update is pinned on the XLA state by
``tests/test_jqmc_kernel/test_gfmc_incremental_pot_state_ae_pbc.py`` and, on the
Rust side, by ``test_rust_ae_pbc_gfmc_n_floor_active_carried_row_matches_full_recompute``
(start pairs at +-0.25 bohr, where a moved electron's selected row is inside
the floor).
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

import jax
import jqmc_rs
from jqmc._precision import precision_policy_snapshot
import numpy as np
import pytest

from jqmc._setting import ATOL_GFMC_INCREMENTAL_TWIN, RTOL_GFMC_INCREMENTAL_TWIN
from jqmc.coulomb_potential import (
    compute_coulomb_potential_ei_discretized_dispatch,
    compute_coulomb_potential_ei_dispatch,
)
from jqmc.wavefunction import compute_kinetic_energy_all_elements
from tests.test_jqmc_kernel.test_gfmc_incremental_pot_state_ae_pbc import _ALAT, _build_ham
from tests.test_rs_kernel.test_gfmc_projection import _MESH_GP, _MESH_W, _xla_step

pytestmark = [pytest.mark.with_chdir]

_N_SUB_MAX = 12


def _start_config(ham):
    """Same-spin pairs at +-0.1 bohr along z around protons 0 (up) and 2 (dn): inside the floor, across the node."""
    nuclei = np.asarray(ham.structure_data.positions, dtype=np.float64)
    ez = np.array([0.0, 0.0, 1.0])
    r_up = np.stack([nuclei[0] + 0.1 * ez, nuclei[0] - 0.1 * ez])
    r_dn = np.stack([nuclei[2] + 0.1 * ez, nuclei[2] - 0.1 * ez])
    return r_up, r_dn


def _disc_row_selected(ham, r_up, r_dn):
    """Per electron: the kinetic elements flip sign and the ``ei_disc`` row exceeds ZV (the row enters ``V_diag``)."""
    cp = ham.coulomb_potential_data
    step = _xla_step(ham, r_up, r_dn, _ALAT, np.eye(3), "tmove")
    bare_up, bare_dn = compute_coulomb_potential_ei_dispatch(coulomb_potential_data=cp, r_up_carts=r_up, r_dn_carts=r_dn)
    disc_up, disc_dn = compute_coulomb_potential_ei_discretized_dispatch(
        coulomb_potential_data=cp, r_up_carts=r_up, r_dn_carts=r_dn, alat=_ALAT
    )
    bare = np.concatenate([np.asarray(bare_up), np.asarray(bare_dn)])
    disc = np.concatenate([np.asarray(disc_up), np.asarray(disc_dn)])
    kc = compute_kinetic_energy_all_elements(ham.wavefunction_data, r_up, r_dn)
    kc = np.concatenate([np.asarray(kc[0]).ravel(), np.asarray(kc[1]).ravel()]).real
    zv = bare + kc - step["nondiag"]
    return step["sign_flip"] & (disc > zv)


@pytest.mark.light
def test_rust_ae_pbc_gfmc_n_interval_matches_full_recompute_per_substep():
    """Rust GFMC_n interval on the AE periodic fixture: carried w_L / V_diag / e_L == full recompute at every sub-step."""
    ham = _build_ham()
    wf = ham.wavefunction_data
    ctx = jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        False,
        True,
        False,
        want_force=False,
        want_gfmc=True,
        precision_policy=precision_policy_snapshot(),
    )
    r_up, r_dn = _start_config(ham)
    start = np.concatenate([r_up, r_dn])
    key = jax.random.PRNGKey(11)
    selected_moved = 0
    for nsub in range(1, _N_SUB_MAX + 1):
        args = (r_up, r_dn, _ALAT, -1.0, nsub, int(key[0]), int(key[1]), np.zeros((0, 3, 3)), _MESH_W, _MESH_GP)
        kw = dict(random_mesh=False, dltmove=False, nn=1)
        prod = ctx.run_gfmc_projection_n(*args, **kw)
        twin = ctx.run_gfmc_projection_n(*args, debug_full_recompute=True, **kw)
        np.testing.assert_array_equal(np.asarray(prod["r_up"]), np.asarray(twin["r_up"]))
        np.testing.assert_array_equal(np.asarray(prod["r_dn"]), np.asarray(twin["r_dn"]))
        assert int(prod["n_pot_updates"]) == nsub
        assert int(twin["n_pot_updates"]) == 0
        for k in ("w_L", "V_diag", "e_L"):
            comparison_label = f"{nsub}/{k}"
            np.testing.assert_allclose(
                float(np.real(prod[k])),
                float(np.real(twin[k])),
                atol=ATOL_GFMC_INCREMENTAL_TWIN,
                rtol=RTOL_GFMC_INCREMENTAL_TWIN,
                err_msg=f"Rust incremental vs full-recompute interval: {k} at sub-step {nsub} ({comparison_label})",
            )
        ru, rd = np.asarray(prod["r_up"]), np.asarray(prod["r_dn"])
        moved = np.any(np.concatenate([ru, rd]) != start, axis=1)
        selected_moved += int(np.count_nonzero(moved & _disc_row_selected(ham, ru, rd)))
    assert selected_moved > 0, "no sub-step selected the carried ei_disc row of a moved electron into V_diag"


#: Start offset and key of the floor-active walk: same-spin pairs at +-0.25 bohr
#: along z around protons 0 (up) and 2 (dn); with ``PRNGKey(0)`` a moved
#: electron lands inside the ``alat`` floor with a sign flip at sub-step 3, so
#: its carried ``ei_disc`` row differs from the bare row AND enters ``V_diag``.
_FLOOR_OFFSET = 0.25
_FLOOR_KEY = 0
_FLOOR_N_SUB = 4


def _floor_active_selected_moved(ham, start, r_up, r_dn):
    """Moved electrons whose selected ``ei_disc`` row differs from the bare row (the floor is active)."""
    cp = ham.coulomb_potential_data
    bare_up, bare_dn = compute_coulomb_potential_ei_dispatch(coulomb_potential_data=cp, r_up_carts=r_up, r_dn_carts=r_dn)
    disc_up, disc_dn = compute_coulomb_potential_ei_discretized_dispatch(
        coulomb_potential_data=cp, r_up_carts=r_up, r_dn_carts=r_dn, alat=_ALAT
    )
    bare = np.concatenate([np.asarray(bare_up), np.asarray(bare_dn)])
    disc = np.concatenate([np.asarray(disc_up), np.asarray(disc_dn)])
    moved = np.any(np.concatenate([r_up, r_dn]) != start, axis=1)
    floor_active = np.abs(disc - bare) > ATOL_GFMC_INCREMENTAL_TWIN
    return int(np.count_nonzero(moved & floor_active & _disc_row_selected(ham, r_up, r_dn)))


@pytest.mark.light
def test_rust_ae_pbc_gfmc_n_floor_active_carried_row_matches_full_recompute():
    """Rust GFMC_n interval where a moved electron's floor-active carried ``ei_disc`` row enters ``V_diag``.

    Same comparison as the test above (nsub = 1 .. ``_FLOOR_N_SUB`` from one
    key, walker byte-identical, w_L / V_diag / e_L at the twin pin, update
    counter == nsub, zero on the twin), from the ``_FLOOR_OFFSET`` start. In
    addition some sub-step must select, into ``V_diag``, the carried
    ``ei_disc`` row of a moved electron that sits inside the ``alat`` floor
    (discretized row != bare row), so a carried row written with the bare
    value or a scaled value changes ``V_diag`` there.
    """
    ham = _build_ham()
    wf = ham.wavefunction_data
    ctx = jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        False,
        True,
        False,
        want_force=False,
        want_gfmc=True,
        precision_policy=precision_policy_snapshot(),
    )
    nuclei = np.asarray(ham.structure_data.positions, dtype=np.float64)
    ez = np.array([0.0, 0.0, 1.0])
    r_up = np.stack([nuclei[0] + _FLOOR_OFFSET * ez, nuclei[0] - _FLOOR_OFFSET * ez])
    r_dn = np.stack([nuclei[2] + _FLOOR_OFFSET * ez, nuclei[2] - _FLOOR_OFFSET * ez])
    start = np.concatenate([r_up, r_dn])
    key = jax.random.PRNGKey(_FLOOR_KEY)
    floor_selected = 0
    for nsub in range(1, _FLOOR_N_SUB + 1):
        args = (r_up, r_dn, _ALAT, -1.0, nsub, int(key[0]), int(key[1]), np.zeros((0, 3, 3)), _MESH_W, _MESH_GP)
        kw = dict(random_mesh=False, dltmove=False, nn=1)
        prod = ctx.run_gfmc_projection_n(*args, **kw)
        twin = ctx.run_gfmc_projection_n(*args, debug_full_recompute=True, **kw)
        np.testing.assert_array_equal(np.asarray(prod["r_up"]), np.asarray(twin["r_up"]))
        np.testing.assert_array_equal(np.asarray(prod["r_dn"]), np.asarray(twin["r_dn"]))
        assert int(prod["n_pot_updates"]) == nsub
        assert int(twin["n_pot_updates"]) == 0
        for k in ("w_L", "V_diag", "e_L"):
            comparison_label = f"{nsub}/{k}"
            np.testing.assert_allclose(
                float(np.real(prod[k])),
                float(np.real(twin[k])),
                atol=ATOL_GFMC_INCREMENTAL_TWIN,
                rtol=RTOL_GFMC_INCREMENTAL_TWIN,
                err_msg=f"Rust incremental vs full-recompute interval (floor start): {k} ({comparison_label})",
            )
        ru, rd = np.asarray(prod["r_up"]), np.asarray(prod["r_dn"])
        floor_selected += _floor_active_selected_moved(ham, start, ru, rd)
    assert floor_selected > 0, "no sub-step selected a floor-active carried ei_disc row of a moved electron"
