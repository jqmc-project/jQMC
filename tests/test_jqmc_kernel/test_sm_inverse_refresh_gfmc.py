"""GFMC observable boundary after a projection that leaves a near-singular start (XLA).

GFMC_t: the near-singular H4 start. H4 periodic AE (J2 exp b=0.5), PCG64
seed 2025 coordinates with the first same-spin pair 1e-3 Bohr apart
(cond(G) ~ 3e8), three offset walkers, tau=0.03, alat=0.3, seed 34, fixed mesh,
force and stress on, SWCT off. GFMC_n: the same start, 16 projections per
measurement.

The test pins the single refresh rule: inside the interval the carried inverse is
screened after every hop and rebuilt from scratch when its probe residual exceeds
``SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD``. The near-singular start must make that
in-interval refresh fire, and the inverse handed to the boundary observables
(captured before any boundary refresh) must have a residual at most the
threshold for every walker, its geminal must equal the one rebuilt at the
boundary positions, and its fast Coulomb energy must equal that of the fresh
route (``tests/_sm_inverse_refresh_cases.py::assert_interval_handover``), each
precision mode judged within itself. GFMC_t also keeps the stored e_L and w_L
of the production run equal to the ``GFMC_T_OBS_FRESH_INV=1`` run. Force and
stress estimators are not compared at the strict pair, nor are the fast local energy, the
continuum kinetic energies per electron and the discretized kinetic mesh
elements of the handed-over inverse: all of them carry the cond(G) x residual
class (near-singular boundary, cond(G) = 5.8e5: mesh ratios up to 169x, kinetic 1.4x
the strict budget against an exact oracle on jax 0.7.2 with a residual the
size of a fresh factorization's, 0.8x on jax 0.10.2).
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

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import jqmc_gfmc
from jqmc._precision import get_tolerance_min
from tests import _sm_inverse_refresh_cases as cases
from tests.test_jqmc_kernel.test_mcmc_acceptance_reference import _input

pytestmark = [pytest.mark.without_chdir]

# The estimators the projection itself hands over (local energy and weight).
_STORED_FIELDS = ("_stored_e_L", "_stored_w_L")


@pytest.fixture
def handover(monkeypatch):
    """Record the inverse, positions and det caches the interval hands to its observables.

    Captured as passed INTO the boundary refresh, i.e. before any boundary refresh.
    """
    record = {}
    original = jqmc_gfmc._refresh_observable_geminal_inv

    def spy(geminal_data, r_up, r_dn, geminal_inv, det_state=None, threshold=None):
        record.update(up=np.asarray(r_up), dn=np.asarray(r_dn), inv=np.asarray(geminal_inv), det_state=det_state)
        return original(geminal_data, r_up, r_dn, geminal_inv, det_state, threshold)

    monkeypatch.setattr(jqmc_gfmc, "_refresh_observable_geminal_inv", spy)
    return record


def _near_singular_h4_walkers():
    """The near-singular H4 walkers (three offset copies)."""
    ham, state = _input("H4_pbc_ae_cart.h5")
    base_up = np.array(state[0])
    base_dn = np.array(state[1])
    base_up[1] = base_up[0] + np.array([0.001, 0.0003, -0.0004])
    up = np.stack([base_up + np.array([0.03 * i, -0.02 * i, 0.04 * i]) for i in range(3)])
    dn = np.stack([base_dn + np.array([-0.01 * i, 0.04 * i, 0.02 * i]) for i in range(3)])
    return ham, up, dn


def _run_gfmc_t(ham, up, dn):
    driver = jqmc_gfmc.GFMC_t(
        hamiltonian_data=ham,
        num_walkers=up.shape[0],
        num_gfmc_collect_steps=1,
        mcmc_seed=34,
        tau=0.03,
        alat=0.3,
        random_discretized_mesh=False,
        non_local_move="tmove",
        comput_position_deriv=True,
        comput_stress=True,
        use_swct=False,
    )
    driver._latest_r_up_carts = jnp.asarray(up)
    driver._latest_r_dn_carts = jnp.asarray(dn)
    driver.run(num_mcmc_steps=1)
    return driver


@pytest.mark.heavy
def test_gfmc_t_carried_observable_inverse_matches_fresh_inverse(monkeypatch, handover):
    """GFMC_t: the handed-over inverse obeys the single refresh rule; stored e_L/w_L equal the fresh route."""
    ham, up, dn = _near_singular_h4_walkers()
    monkeypatch.delenv("GFMC_T_OBS_FRESH_INV", raising=False)
    carried = _run_gfmc_t(ham, up, dn)
    assert "inv" in handover, "GFMC_t must hand its projection inverse through the boundary screen"
    assert getattr(carried, "_num_interval_inverse_refreshes", 0) > 0, "the near-singular start must trigger the refresh"
    cases.assert_interval_handover(ham, handover["up"], handover["dn"], handover["inv"], handover["det_state"])
    monkeypatch.setenv("GFMC_T_OBS_FRESH_INV", "1")
    fresh = _run_gfmc_t(ham, up, dn)

    atol, rtol = get_tolerance_min(cases.SAME_G_ZONES, "strict")
    # Same sampled configurations in both arms (the switch only affects observables).
    np.testing.assert_array_equal(np.asarray(carried._latest_r_up_carts), np.asarray(fresh._latest_r_up_carts))
    pairs = {}
    for field in _STORED_FIELDS:
        got = np.asarray(getattr(carried, field))
        ref = np.asarray(getattr(fresh, field))
        assert got.shape == ref.shape and got.size > 0, field
        pairs[field] = ([got], [ref])
    cases.assert_close_report(pairs, atol, rtol, "carried vs fresh observable inverse")


@pytest.mark.heavy
def test_gfmc_n_carried_observable_inverse_matches_same_g_inverse(handover):
    """GFMC_n: the handed-over inverse obeys the single refresh rule (residual, G generation, Coulomb)."""
    ham, up, dn = _near_singular_h4_walkers()
    driver = jqmc_gfmc.GFMC_n(
        hamiltonian_data=ham,
        num_walkers=up.shape[0],
        num_mcmc_per_measurement=16,
        num_gfmc_collect_steps=1,
        mcmc_seed=34,
        E_scf=-2.0,
        alat=0.3,
        random_discretized_mesh=False,
        non_local_move="tmove",
        comput_position_deriv=True,
        comput_stress=True,
        use_swct=False,
    )
    driver._latest_r_up_carts = jnp.asarray(up)
    driver._latest_r_dn_carts = jnp.asarray(dn)
    driver.run(num_mcmc_steps=1)
    assert "inv" in handover, "GFMC_n must hand its running inverse through the boundary screen"
    assert getattr(driver, "_num_interval_inverse_refreshes", 0) > 0, "the near-singular start must trigger the refresh"
    cases.assert_interval_handover(ham, handover["up"], handover["dn"], handover["inv"], handover["det_state"])
