"""Real uncached-branch controls for periodic, twist and empty-spin XLA gates.

The positive cases extend the existing kinetic/ECP channel gates in
``test_integration_mesh_cache_routes``. These controls disconnect the supplied
J1 candidate state at the actual production Jastrow-ratio consumer: the normal
uncached formula executes, and production ``note_route`` records that branch.
No probe/counter is patched. The unchanged positive gate must reject the
resulting projection-site uncached channel or empty-spin coverage loss after
the driver completes.

Both algorithms, both meshes and the committed Gamma/twist J1+J2 fixtures are
covered in each precision mode. This is trace-time route evidence, not an
independent numerical oracle, thread-determinism claim or physical J3 test.
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
import pytest

import jqmc.coulomb_potential as coulomb
import jqmc.jqmc_gfmc as gfmc
import jqmc.wavefunction as wavefunction
from tests.test_jqmc_kernel import test_integration_mesh_cache_routes as routes
from tests.test_jqmc_kernel.test_integration_mesh_cache_routes import (
    test_xla_ecp_mesh_caches_every_jastrow_channel as _ecp_gate,
    test_xla_kinetic_mesh_caches_every_jastrow_channel as _kinetic_gate,
)

pytestmark = [pytest.mark.with_chdir]

_EXPECTED_CHANNELS = {"det", "j1", "j2"}


@pytest.mark.light
@pytest.mark.parametrize("driver", ["n", "t"])
def test_xla_projection_consumers_receive_j3_streaming_state(driver, monkeypatch):
    """Check the maintained J3 state independently of the candidate-table counters."""
    observed = {}
    for mesh, name, mesh_state_arg in (
        ("kinetic", "compute_discretized_kinetic_energy_from_streaming_state", "mesh_state"),
        ("ecp", "compute_ecp_non_local_parts_nearest_neighbors_fast_update", "candidate_mesh_state"),
    ):
        original = getattr(gfmc, name)
        states = observed[mesh] = []

        def record(*args, _original=original, _states=states, _mesh_state_arg=mesh_state_arg, **kwargs):
            # Substep zero can build the mesh without a maintained J3 state.
            # Observe the actual consumers once a carried mesh is supplied.
            if kwargs.get(_mesh_state_arg) is not None:
                _states.append(kwargs.get("j3_state") is not None)
            return _original(*args, **kwargs)

        monkeypatch.setattr(gfmc, name, record)

    routes._run_xla_gate(driver, "mol_ecp_cart_j123")
    for mesh, states in observed.items():
        assert states, f"{driver} never called the {mesh} consumer with a carried mesh"
        assert all(states), f"{driver} dropped the J3 streaming state at the {mesh} consumer"


@pytest.mark.light
@pytest.mark.parametrize("driver", ["n", "t"])
@pytest.mark.parametrize(
    "pattern",
    [
        pytest.param("pbc_gamma_sphepgto_j12", id="pbc-gamma-j1-j2"),
        pytest.param("twist_sphepgto_j12", id="twist-j1-j2"),
    ],
)
@pytest.mark.parametrize("mesh", ["kinetic", "ecp"])
def test_periodic_xla_cache_gate_rejects_real_uncached_j1(driver, pattern, mesh, monkeypatch):
    """Removing a real candidate input must fail the specific uncached-route verdict."""
    if mesh == "kinetic":
        module = wavefunction
        name = "compute_ratio_Jastrow_one_move_mesh"
    else:
        module = coulomb
        name = "_compute_ratio_Jastrow_part_split_spin"
    original = getattr(module, name)
    disconnected = []
    mesh_label = "ECP" if mesh == "ecp" else mesh

    def without_candidate(*args, **kwargs):
        if kwargs.get("j1_candidate_state") is not None:
            disconnected.append(True)
            kwargs["j1_candidate_state"] = None
        return original(*args, **kwargs)

    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(module, name, without_candidate)
            uncached_route = f"channels on the UNCACHED {mesh_label}-mesh route during projection"
            # Each mesh calls its own unchanged positive gate explicitly, so the
            # expected failure is bound to that gate's declared comparison.
            if mesh == "kinetic":
                with pytest.raises(AssertionError, match=uncached_route):
                    _kinetic_gate(driver, pattern, _EXPECTED_CHANNELS)
            else:
                with pytest.raises(AssertionError, match=uncached_route):
                    _ecp_gate(driver, pattern, _EXPECTED_CHANNELS)
            assert disconnected, "the production consumer never received the cache that this control disconnects"
            assert routes._observed_channels(mesh, "projection") == _EXPECTED_CHANNELS
            assert routes._uncached_channels(mesh, "projection") == {"j1"}
            other_mesh = "ecp" if mesh == "kinetic" else "kinetic"
            assert routes._observed_channels(other_mesh, "projection") == _EXPECTED_CHANNELS
            assert routes._uncached_channels(other_mesh, "projection") == set()
    finally:
        # Restoring the Python consumer alone does not evict its mutated graph.
        jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize("driver", ["n", "t"])
@pytest.mark.parametrize("mesh", ["kinetic", "ecp"])
def test_empty_spin_xla_cache_gate_rejects_real_jastrow_fallback(driver, mesh, monkeypatch):
    """A missing J1 cache must expose the real empty-spin uncached J1 fallback.

    An empty spin channel takes the same per-channel rank-one route as two
    spins (no whole ``compute_Jastrow_part`` recompute serves it), so disconnecting J1 records exactly J1 as uncached while
    the still-supplied J2 cache is consumed, as in the two-spin control above.
    J2 is physically zero for one electron; its candidate-state consumption is
    a structural route obligation, not evidence of a nonzero J2 energy.
    """
    if mesh == "kinetic":
        module = wavefunction
        name = "compute_ratio_Jastrow_one_move_mesh"
    else:
        module = coulomb
        name = "_compute_ratio_Jastrow_part_split_spin"
    mesh_label = "ECP" if mesh == "ecp" else mesh
    diagnostic = f"channels on the UNCACHED {mesh_label}-mesh route during projection"
    original = getattr(module, name)
    disconnected = []

    def without_candidate(*args, **kwargs):
        if kwargs.get("j1_candidate_state") is not None:
            disconnected.append(True)
            kwargs["j1_candidate_state"] = None
        return original(*args, **kwargs)

    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(module, name, without_candidate)
            with pytest.raises(AssertionError, match=diagnostic):
                if mesh == "kinetic":
                    _kinetic_gate(driver, "mol_ecp_single_spin_j12", _EXPECTED_CHANNELS)
                else:
                    _ecp_gate(driver, "mol_ecp_single_spin_j12", _EXPECTED_CHANNELS)
            assert disconnected, "the empty-spin projection never supplied a J1 candidate cache"
            assert routes._observed_channels(mesh, "projection") == _EXPECTED_CHANNELS
            assert routes._uncached_channels(mesh, "projection") == {"j1"}
            other_mesh = "ecp" if mesh == "kinetic" else "kinetic"
            assert routes._observed_channels(other_mesh, "projection") == _EXPECTED_CHANNELS
            assert routes._uncached_channels(other_mesh, "projection") == set()
    finally:
        jax.clear_caches()
