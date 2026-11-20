"""Focused regression tests for optimizer orchestration safety gates."""

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

from dataclasses import dataclass, field

import numpy as np
import pytest

from jqmc.jqmc_optimizer import (
    EnergyEstimate,
    OptimizationConfig,
    _collective_step_acceptance,
    run_optimize,
)

pytestmark = [pytest.mark.with_chdir]


class _FakeComm:
    def __init__(self, size: int):
        self._size = int(size)

    def Get_size(self) -> int:
        return self._size

    def Get_rank(self) -> int:
        return 0

    def Barrier(self) -> None:
        return None

    def allreduce(self, value, op=None):
        return value


@dataclass
class _ControlFlowSampler:
    energy: float = 0.0
    comm: _FakeComm = field(default_factory=lambda: _FakeComm(1))
    world_comm: _FakeComm = field(default_factory=lambda: _FakeComm(1))
    rank: int = 0
    size: int = 1
    optimization_step_index: int = 0
    hamiltonian_data: object = None
    advanced: int = 0
    configured: dict | None = None
    wavefunction_state_token: tuple[str, int] = ("control-test", 0)

    def derivative_capabilities(self):
        return True, True

    def configure_optimization(self, *, method, hyperparameters):
        self.configured = dict(hyperparameters)

    def select_parameters(self, flags):
        return None

    def sample(self, *, num_steps, max_time):
        return None

    def estimate_energy(self, *, num_warmup_steps, num_bin_blocks):
        return EnergyEstimate(
            mean=self.energy,
            standard_error=0.0,
            variance=0.0,
            variance_standard_error=0.0,
        )

    def advance_optimization_steps(self, count):
        self.advanced += int(count)
        self.optimization_step_index += int(count)

    def validate_wavefunction_candidate(self, candidate):
        # Control-only tests never accept a numerical candidate.
        raise AssertionError("Control-only sampler must not publish a candidate.")

    def advance_wavefunction_revision(self):
        lineage, revision = self.wavefunction_state_token
        self.wavefunction_state_token = (lineage, revision + 1)


def _disable_driver_side_effects(monkeypatch):
    monkeypatch.setattr("jqmc.jqmc_optimizer._check_mpi4py_jax_distribution_consistency", lambda: None)
    monkeypatch.setattr("jqmc.jqmc_optimizer._init_external_control_file", lambda *args, **kwargs: None)
    monkeypatch.setattr("jqmc.jqmc_optimizer._remove_external_control_file", lambda *args, **kwargs: None)
    monkeypatch.setattr("jqmc.jqmc_optimizer._log_memory_usage", lambda *args, **kwargs: None)


@pytest.mark.light
def test_device_optimizer_request_is_kept_and_the_removed_collectives_option_raises(monkeypatch):
    """A twist subcommunicator keeps the device optimizer; ``use_device_collectives`` is not an option.

    The device reduction route follows from the communicators, so a sampler
    subcommunicator neither loses the device optimizer nor publishes a
    collectives switch, and the removed keyword is rejected like any unknown
    one (``TypeError``), with no compatibility path.
    """
    _disable_driver_side_effects(monkeypatch)
    sampler = _ControlFlowSampler(comm=_FakeComm(1), world_comm=_FakeComm(2))
    run_optimize(sampler, OptimizationConfig(num_opt_steps=0, use_device_optimizer=True, twist_index=0))
    assert sampler.configured["use_device_optimizer"] is True
    assert "use_device_collectives" not in sampler.configured
    with pytest.raises(TypeError, match="use_device_collectives"):
        OptimizationConfig(num_opt_steps=0, use_device_collectives=False)


@pytest.mark.light
def test_twist_step_acceptance_is_world_atomic():
    """Twist step acceptance is world atomic."""
    assert _collective_step_acceptance(True, False, _FakeComm(1)) is True
    assert _collective_step_acceptance(False, False, _FakeComm(1)) is False

    class _RemoteRejectComm(_FakeComm):
        def allreduce(self, value, op=None):
            return 0

    assert _collective_step_acceptance(True, True, _RemoteRejectComm(2)) is False


@pytest.mark.light
def test_zero_step_optimization_is_a_noop(monkeypatch):
    """Zero step optimization is a noop."""
    _disable_driver_side_effects(monkeypatch)
    sampler = _ControlFlowSampler()

    run_optimize(
        sampler,
        OptimizationConfig(num_opt_steps=0, use_device_optimizer=False),
    )

    assert sampler.advanced == 0
    assert sampler.optimization_step_index == 0


@pytest.mark.light
@pytest.mark.parametrize(
    "key,value",
    [
        ("cg_tol", np.nan),
        ("cg_tol", np.inf),
        ("cg_tol", 0.0),
        ("cg_tol", -1.0e-5),
        ("cg_max_iter", 0),
        ("cg_max_iter", 2.5),
        ("epsilon", -1.0e-3),
        ("epsilon", np.nan),
        ("lm_subspace_dim", -2),
    ],
)
def test_invalid_sr_solver_option_is_rejected_before_configuration(monkeypatch, key, value):
    """An invalid cg_tol, cg_max_iter, epsilon or lm_subspace_dim raises ValueError while run_optimize parses its options."""
    _disable_driver_side_effects(monkeypatch)
    sampler = _ControlFlowSampler()
    with pytest.raises(ValueError, match=rf"optimizer_kwargs\['{key}'\]"):
        run_optimize(
            sampler,
            OptimizationConfig(num_opt_steps=1, use_device_optimizer=False, optimizer_kwargs={key: value}),
        )
    assert sampler.configured is None


@pytest.mark.light
def test_nonfinite_energy_does_not_advance_accepted_step_counter(monkeypatch):
    """Nonfinite energy does not advance accepted step counter."""
    _disable_driver_side_effects(monkeypatch)
    sampler = _ControlFlowSampler(energy=np.nan)

    run_optimize(
        sampler,
        OptimizationConfig(num_opt_steps=1, use_device_optimizer=False),
    )

    assert sampler.advanced == 0
    assert sampler.optimization_step_index == 0


@pytest.mark.light
@pytest.mark.parametrize(
    "local_ok,peer_ok,twists", [(True, False, False), (False, True, False), (True, True, False), (True, False, True)]
)
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Typed communicator protocol witness; no numerical model or electron positions."
)
def test_candidate_acceptance_rejects_on_every_sampler_rank(local_ok, peer_ok, twists):
    """A rank-local candidate error is reduced inside a sampler even without twist decomposition."""
    from mpi4py import MPI
    from jqmc.jqmc_optimizer import _collective_candidate_acceptance

    class SamplerComm:
        def __init__(self):
            self.calls = []

        def allreduce(self, value, op):
            self.calls.append((value, op))
            return min(value, int(peer_ok))

    sampler_comm = SamplerComm()
    result = _collective_candidate_acceptance(local_ok, sampler_comm, twists, _FakeComm(4 if twists else 2))
    assert result is (local_ok and peer_ok)
    assert sampler_comm.calls == [(int(local_ok), MPI.MIN)]
