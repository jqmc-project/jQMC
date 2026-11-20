"""Observe the direct DLA ECP consumer, independently of its route labels."""

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
import numpy as np
import pytest

from jqmc import jqmc_gfmc as implementation
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc.coulomb_potential import compute_ecp_non_local_parts_all_pairs
from tests.test_jqmc_kernel import test_integration_mesh_cache_routes as routes
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel.test_gfmc_chain_modes import _run_gfmc

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]

_CANDIDATES = {
    "j1": "j1_candidate_state",
    "j2": "j2_candidate_state",
    "j3": "j3_candidate_state",
    "dtn": "j_dtn_candidate_state",
}


@pytest.mark.parametrize("driver", ["n", "t"])
@pytest.mark.parametrize("pattern", ["mol_ecp_cart_j123", "mol_ecp_cart_jdtn"])
def test_dltmove_ecp_consumes_active_candidate_states(monkeypatch, driver, pattern):
    """Actual DLA calls must supply each active cache and report the same channels.

    The first state-building call can have no candidate mesh. Inspect all direct
    calls with any carried candidate state, without depending on their route tag.
    Observed calls and complete channel coverage are mandatory, so removing the
    tag or all states cannot turn a missing cache into a vacuous pass.
    """
    expected = {"j1", "j2", "j3" if pattern.endswith("j123") else "dtn"}
    observed = []
    original = implementation._compute_ratio_Jastrow_part_rank1_update

    def observe(*args, **kwargs):
        if any(kwargs.get(name) is not None for name in _CANDIDATES.values()):
            observed.append({channel for channel, name in _CANDIDATES.items() if kwargs.get(name) is not None})
        return original(*args, **kwargs)

    monkeypatch.setattr(implementation, "_compute_ratio_Jastrow_part_rank1_update", observe)
    routes._run_xla_gate(driver, pattern, non_local_move="dltmove")
    assert observed, "no direct DLT candidate consumer traced"
    assert all(expected <= supplied for supplied in observed), f"missing DLT candidate states: {observed}"
    assert routes._observed_channels("ecp", "projection") == expected | {"det"}, "incomplete DLT ECP route coverage"
    assert not routes._uncached_channels("ecp", "projection"), "uncached DLT ECP projection channel"


@pytest.mark.parametrize("driver", ["n", "t"])
def test_dltmove_ecp_gate_rejects_missing_j3(monkeypatch, driver):
    """A real candidate loss must fail the positive gate for either driver."""
    original = implementation._compute_ratio_Jastrow_part_rank1_update

    def lose_j3(*args, **kwargs):
        kwargs["j3_candidate_state"] = None
        return original(*args, **kwargs)

    monkeypatch.setattr(implementation, "_compute_ratio_Jastrow_part_rank1_update", lose_j3)
    with pytest.MonkeyPatch.context() as gate_patch:
        with pytest.raises(AssertionError, match="uncached DLT ECP projection channel"):
            test_dltmove_ecp_consumes_active_candidate_states(gate_patch, driver, "mol_ecp_cart_j123")
    jax.clear_caches()


@pytest.mark.parametrize("driver", ["n", "t"])
def test_dltmove_carried_j1_matches_recomputed_ratio(monkeypatch, driver):
    """Compare both J1 branches at identical real N2 ECP projection inputs.

    Nitrogen's nonzero ccECP projectors prevent the H-only route fixture from
    hiding a changed ECP contribution. The reference recomputes J1 from the
    electron and nucleus positions while retaining the other candidate states.
    """
    ham = _build_ham_for_cell("N2_ecp_ccpvtz_cart.h5", True, None, "SD")
    gd = ham.wavefunction_data.geminal_data
    rng = np.random.default_rng(3446)
    r_up = np.asarray(ham.structure_data.positions)[np.arange(gd.num_electron_up) % 2] + rng.normal(
        scale=0.5, size=(gd.num_electron_up, 3)
    )
    r_dn = np.asarray(ham.structure_data.positions)[np.arange(gd.num_electron_dn) % 2] + rng.normal(
        scale=0.5, size=(gd.num_electron_dn, 3)
    )
    nonlocal_sum = compute_ecp_non_local_parts_all_pairs(
        ham.coulomb_potential_data, ham.wavefunction_data, r_up, r_dn, np.eye(3)
    )[3]
    assert np.isfinite(nonlocal_sum) and nonlocal_sum != 0, "ECP reference must have a live projector"
    original = implementation._compute_ratio_Jastrow_part_rank1_update
    pairs = []

    def compare(*args, **kwargs):
        actual = original(*args, **kwargs)
        if kwargs.get("j1_candidate_state") is not None:
            reference = original(*args, **dict(kwargs, j1_candidate_state=None, _route_mesh=None))
            jax.debug.callback(lambda a, b: pairs.append((np.array(a), np.array(b))), actual, reference)
        return actual

    monkeypatch.setattr(implementation, "_compute_ratio_Jastrow_part_rank1_update", compare)
    jax.clear_caches()
    _run_gfmc(driver, ham, 3446, "jax", 3, nsub=2, tau=0.01, non_local_move="dltmove")
    jax.effects_barrier()
    assert pairs, "no carried J1 comparison executed"
    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"),
        "medium" if is_mixed_precision_enabled() else "strict",
    )
    for actual, reference in pairs:
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
