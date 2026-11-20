"""SR/LM solver guards: unconverged CG is reported, bad inputs are rejected, twist aSR reports its scale."""

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

import logging
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from mpi4py import MPI

import jqmc.jqmc_optimizer as optimizer_module
from jqmc.jqmc_optimizer import (
    OptimizationCoordinateMap,
    compute_real_coordinate_optimization_target,
    compute_twist_decomposed_real_coordinate_optimization_target,
)

from .test_optimizer_lm_real import _device_batch, _host_stage_log_sources, _lm_case

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_TWIST_SCOPES = ("world", "sampler", "world")
_NONCONVERGED = "SR CG did not converge"


def _target(entry, batch, coordinate_map, **overrides):
    """Call one optimizer target entry on a one-rank communicator."""
    options = dict(
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=True,
        cg_max_iter=500,
        cg_tol=1.0e-12,
        use_lm=False,
        lm_subspace_dim=0,
        lm_condition_cutoff=1.0e-3,
    )
    options.update(overrides)
    if entry == "single-k":
        return compute_real_coordinate_optimization_target(
            batch,
            coordinate_map,
            comm=MPI.COMM_SELF,
            warm_start=None,
            energy_standard_error=0.1,
            **options,
        )
    return compute_twist_decomposed_real_coordinate_optimization_target(
        batch,
        coordinate_map,
        sampler_comm=MPI.COMM_SELF,
        world_comm=MPI.COMM_SELF,
        world_energy_standard_error=0.1,
        sampler_energy_standard_error=0.1,
        **options,
    )


@pytest.mark.parametrize("route", ["host", "device"])
@pytest.mark.parametrize("entry", ["single-k", "twist"])
def test_unconverged_sr_cg_warns_once_per_solve_and_keeps_the_step(caplog, entry, route):
    """A CG stopped by cg_max_iter warns once per unconverged solve (per twist scope) at default verbosity; the step stays.

    The twist case owns three twist-local coordinates (not solved in one CG
    step) and one shared coordinate (solved exactly in one step), so only the
    twist-local scope may warn.
    """
    blocks, batch, _O_q, _D_q = _lm_case("mixed", scopes=("sampler", "sampler", "world") if entry == "twist" else None)
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    if route == "device":
        batch = _host_stage_log_sources(blocks, _device_batch(blocks, batch))
    with caplog.at_level(logging.WARNING, logger="jqmc"):
        target = _target(entry, batch, coordinate_map, cg_max_iter=1)
    warnings = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    warnings = [message for message in warnings if _NONCONVERGED in message]
    assert len(warnings) == 1, warnings
    assert "iterations=1 (cg_max_iter=1)" in warnings[0] and "breakdown=False" in warnings[0]
    if entry == "twist":
        assert "twist-local scope" in warnings[0] and "1/1 sampler(s)" in warnings[0]
    assert not target.converged and not target.cg_breakdown
    assert np.all(np.isfinite(target.q)) and np.any(target.q != 0.0)


def _with_nan_sample(batch, kind):
    """Return the device batch with one non-finite log derivative or local energy."""
    if kind == "local-energy":
        return replace(batch, local_energies=jnp.asarray(batch.local_energies).at[3].set(jnp.nan))
    first = batch.log_derivative_sources[0]
    poisoned = replace(first, direct=jnp.asarray(first.direct).at[3].set(jnp.nan))
    return replace(batch, log_derivative_sources=(poisoned, *batch.log_derivative_sources[1:]))


@pytest.mark.parametrize("kind", ["log-derivative", "local-energy"])
@pytest.mark.parametrize("route", ["dense-single-k", "dense-planb", "streaming"])
def test_nonfinite_sr_sample_raises_on_every_device_route(route, kind):
    """A NaN SR sample raises the host route's ValueError on the device routes instead of being frozen or zeroed."""
    blocks, batch, _O_q, _D_q = _lm_case("mixed")
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    device_batch = _with_nan_sample(_device_batch(blocks, batch), kind)
    indices = np.arange(coordinate_map.coordinate_size, dtype=np.int64)
    solver = dict(epsilon=0.2, diagonal_floor=0.0, cg=True, cg_max_iter=500, cg_tol=1.0e-12)
    with pytest.raises(ValueError, match="wide SR sample arrays must be finite"):
        if route == "dense-single-k":
            # The dense single-k device route is the direct (cg=False) solve.
            optimizer_module._compute_device_sr_real_target_dense(
                device_batch, coordinate_map, epsilon=0.2, diagonal_floor=0.0, all_real=False
            )
        elif route == "dense-planb":
            optimizer_module._solve_planb_sr_scope_device_dense(
                device_batch, coordinate_map, indices, sync_scope="sampler", comm=MPI.COMM_SELF, all_real=False, **solver
            )
        else:
            optimizer_module._solve_device_sr_scope_streaming(
                device_batch,
                coordinate_map,
                indices,
                sync_scope=None,
                comm=MPI.COMM_SELF,
                device_collectives=True,
                all_real=False,
                **solver,
            )


@pytest.mark.parametrize("lm_subspace_dim", [0, 2])
def test_twist_asr_scale_is_the_smallest_magnitude_scope_gamma_with_its_sign(monkeypatch, lm_subspace_dim):
    """Twist aSR (and the aSR fallback of a rejected LM) report the scope gamma of smallest |gamma|, sign kept."""
    blocks, batch, _O_q, _D_q = _lm_case("mixed", scopes=_TWIST_SCOPES)
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    # Shared scope first, then the twist-local scope.
    gammas = iter((-0.6, 0.3))

    def scope_gamma(*coefficients, gamma_max, fallback_gamma):
        del coefficients, gamma_max, fallback_gamma
        return optimizer_module.ASRBoundedResult(gamma=next(gammas), predicted_energy=-1.0, used_fallback=False)

    def rejected_lm(pencil, *, condition_cutoff):
        del pencil, condition_cutoff
        raise optimizer_module._kernel.LMNoRealRootError("rejected for the test")

    monkeypatch.setattr(optimizer_module, "select_asr_gamma_bounded", scope_gamma)
    monkeypatch.setattr(optimizer_module, "solve_lm_real_pencil", rejected_lm)
    result = _target("twist", batch, coordinate_map, cg=False, use_lm=True, lm_subspace_dim=lm_subspace_dim)
    assert result.accepted_scale == 0.3


@pytest.mark.parametrize("entry", ["single-k", "twist"])
def test_lm_subspace_dim_below_minus_one_is_rejected_by_both_target_entries(entry):
    """lm_subspace_dim < -1 raises on the single-k and the twist entry instead of running full LM."""
    blocks, batch, _O_q, _D_q = _lm_case("mixed", scopes=_TWIST_SCOPES if entry == "twist" else None)
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    with pytest.raises(ValueError, match=r"optimizer_kwargs\['lm_subspace_dim'\]"):
        _target(entry, batch, coordinate_map, use_lm=True, lm_subspace_dim=-2)
