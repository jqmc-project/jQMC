"""Native GFMC projection-inverse handoff after a near-singular start.

The native GFMC interval advances its own Sherman--Morrison inverse and hands
it to the Pulay force/stress fold-in. Under the single per-hop refresh rule, on
the near-singular H4 inputs (first same-spin pair 1e-3 Bohr apart) the production run must fire the
per-hop in-interval refresh (route-predicate counter 11), and at the interval
boundary the handoff screen must find the carried inverse's probe residual at
most ``SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD`` for every walker (every force
block takes the handoff, counter 64; none is screened out, counter 65). The
local energies and weights the projection hands over must equal those of the
fresh-factorization route (threshold 0 at the boundary) within the active
precision mode's central strict tolerances; both arms run the identical chain.
Pulay force/stress estimators are not compared at the strict pair: at
cond(G) ~ 1e5-1e6 they amplify the admitted residual.
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

import jqmc_rs
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import _setting, jqmc_gfmc
from jqmc._precision import get_tolerance_min
from tests import _sm_inverse_refresh_cases as cases
from tests.test_jqmc_kernel.test_mcmc_acceptance_reference import _input

pytestmark = [pytest.mark.without_chdir]

# The estimators the projection itself hands over (local energy and weight).
_FIELDS = ("_stored_e_L", "_stored_w_L")


def _near_singular_h4_walkers():
    ham, state = _input("H4_pbc_ae_cart.h5")
    base_up = np.array(state[0])
    base_dn = np.array(state[1])
    base_up[1] = base_up[0] + np.array([0.001, 0.0003, -0.0004])
    up = np.stack([base_up + np.array([0.03 * i, -0.02 * i, 0.04 * i]) for i in range(3)])
    dn = np.stack([base_dn + np.array([-0.01 * i, 0.04 * i, 0.02 * i]) for i in range(3)])
    return ham, up, dn


def _run(algorithm, ham, up, dn):
    common = dict(
        hamiltonian_data=ham,
        num_walkers=up.shape[0],
        num_gfmc_collect_steps=1,
        mcmc_seed=34,
        alat=0.3,
        random_discretized_mesh=False,
        non_local_move="tmove",
        comput_position_deriv=True,
        comput_stress=True,
        use_swct=False,
    )
    if algorithm == "t":
        driver = jqmc_gfmc.GFMC_t(tau=0.03, **common)
    else:
        driver = jqmc_gfmc.GFMC_n(num_mcmc_per_measurement=16, E_scf=-2.0, **common)
    driver._latest_r_up_carts = jnp.asarray(up)
    driver._latest_r_dn_carts = jnp.asarray(dn)
    driver.run(num_mcmc_steps=1)
    return {field: np.asarray(getattr(driver, field)).copy() for field in _FIELDS}, np.asarray(driver._latest_r_up_carts)


@pytest.mark.heavy
@pytest.mark.parametrize("algorithm", ["t", "n"])
def test_native_gfmc_projection_inverse_handoff_matches_fresh_route(algorithm, monkeypatch):
    """Native GFMC_t/GFMC_n: the handed-over inverse obeys the single refresh rule; outputs == fresh route."""
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    ham, up, dn = _near_singular_h4_walkers()
    jqmc_rs.gfmc_route_counters_reset()
    production, positions = _run(algorithm, ham, up, dn)
    predicate = jqmc_rs.route_predicate_counters()
    routes = jqmc_rs.gfmc_route_counters()
    assert predicate[11] > 0, (
        f"the near-singular start must trigger the in-interval refresh (carries/refreshes {predicate[10]}/{predicate[11]})"
    )
    assert routes[64] > 0, "the force fold-in must take the projection inverse handoff"
    assert routes[65] == 0, (
        f"{routes[65]} force block(s) screened the handed-over inverse out: its residual exceeded the refresh "
        "threshold at the interval boundary"
    )
    monkeypatch.setattr(_setting, "SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD", 0.0)
    fresh, fresh_positions = _run(algorithm, ham, up, dn)
    np.testing.assert_array_equal(positions, fresh_positions)

    atol, rtol = get_tolerance_min(cases.SAME_G_ZONES, "strict")
    pairs = {}
    for field in _FIELDS:
        got, ref = production[field], fresh[field]
        assert got.shape == ref.shape and got.size > 0, field
        pairs[field] = ([got], [ref])
    cases.assert_close_report(pairs, atol, rtol, "projection-inverse handoff vs fresh route")
