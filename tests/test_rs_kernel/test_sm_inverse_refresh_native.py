"""Carried Sherman--Morrison inverse after many MCMC updates (native).

Same near-singular H4 / water-ECP cases as the XLA twin
``tests/test_jqmc_kernel/test_sm_inverse_refresh.py``, driven through the two
native MCMC routes that carry the inverse:

* ``chunk``: the native energy chunk (sweep + e_L + force/stress threaded
  inside Rust for the whole run); the refresh runs per step inside the chunk.
* ``per_step``: the per-step native sweep (selected by the parameter-derivative
  observable); the driver refreshes the swept inverse before its consumers.

The final carried inverse handed to the consumers must agree with the seed's
SVD pseudo-inverse of the same carried geminal for every fast consumer within
the active precision mode's central strict tolerances.
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

import os

import jqmc_rs  # noqa: F401  (hard requirement of this suite)
import numpy as np
import pytest

from tests import _sm_inverse_refresh_cases as cases

pytestmark = [pytest.mark.without_chdir]

_MIN_SEED_CONDITION = 1.0e14


@pytest.mark.heavy
@pytest.mark.parametrize("route", ["chunk", "per_step"])
@pytest.mark.parametrize("case", ["h4_pbc_ae", "water_ecp"])
def test_native_mcmc_carried_inverse_matches_same_g_inverse_after_many_updates(case, route, monkeypatch):
    """Native driver routes: carried-inverse consumers equal same-G pseudo-inverse consumers."""
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    assert os.environ["JQMC_FLAG_RUST_ON_CPU"] == "1"
    ham, r_up, r_dn = cases.build_case(case)
    assert np.max(cases.seed_condition_numbers(ham, r_up, r_dn)) > _MIN_SEED_CONDITION
    steps = cases.CASES[case][2]
    observables = {
        "comput_position_deriv": True,
        "comput_stress": bool(ham.structure_data.pbc_flag),
    }
    if route == "chunk":
        capture = cases.capture_chunk_state()
    else:
        # The parameter-derivative observable keeps the per-step native sweep.
        observables["comput_log_WF_param_deriv"] = True
        capture = cases.capture_refreshed_state()
    with capture as record:
        driver = cases.run_driver(ham, r_up, r_dn, steps, **observables)
    assert driver.mcmc_counter == steps
    # Route witness: the chunk records its returned state; the per-step route
    # records the driver-side refresh call.
    assert ("chunk_refreshes" in record) == (route == "chunk")
    up = driver._MCMC__latest_r_up_carts
    dn = driver._MCMC__latest_r_dn_carts

    cases.assert_same_g_consumers(ham, up, dn, record["geminal"], record["geminal_inv"])
    cases.assert_g_generation(ham, up, dn, record["geminal"])
    assert driver.num_inverse_refreshes > 0, "the near-singular seed must trigger the refresh"
