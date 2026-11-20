"""C12 typed, scaled optimizer-result and atomic update contracts."""

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

import numpy as np
import pytest

import jqmc.jqmc_optimizer as optimizer_module
from jqmc._setting import ATOL_BIT_EXACT, RTOL_PYTHON_FP64_LOOSE
from jqmc.determinant import Geminal_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data
from jqmc.jqmc_optimizer import (
    BlockStorageUpdate,
    OptimizationCoordinateMap,
    OptimizationDiagnostics,
    RealCoordinateOptimizationTarget,
    _log_optimization_diagnostics,
    finalize_optimization_step,
)
from jqmc.wavefunction import Wavefunction_data

from .optimizer_metadata_helpers import make_test_block


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _blocks():
    return (
        make_test_block(
            "j1_param",
            np.asarray(0.4, dtype=np.float64),
            field="real",
            derivative_policy="direct_real",
            sync_scope="world",
        ),
        make_test_block(
            "z",
            np.asarray([1.0 + 2.0j], dtype=np.complex128),
            field="complex",
            derivative_policy="holomorphic",
            sync_scope="sampler",
        ),
    )


def _target(coordinate_map):
    q = np.asarray([2.0, -3.0, 4.0])
    return RealCoordinateOptimizationTarget(
        q=q,
        sr_q=q,
        coordinate_force=np.asarray([1.0, 2.0, 3.0]),
        coordinate_force_std=np.asarray([0.1, 0.2, 0.3]),
        signal_to_noise=np.asarray([10.0, 10.0, 10.0]),
        sr_regime="wide",
        accepted_scale=1.0,
        predicted_energy=None,
        lm_reference_weight=None,
        warm_start_out=None,
        converged=True,
        num_iterations=7,
        residual_norm=1.0e-12,
        residual_tolerance=1.0e-8,
        metric_diagonal=np.asarray([4.0, 5.0, 6.0]),
        frozen_coordinate_groups=(),
        selected_coordinate_groups=coordinate_map.coordinate_groups,
    )


def test_finalize_step_scales_once_and_preserves_each_storage_dtype():
    """Finalize step scales once and preserves each storage dtype."""
    blocks = _blocks()
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    result = finalize_optimization_step(_target(coordinate_map), coordinate_map, learning_rate=0.25)

    assert np.array_equal(result.q, np.asarray([0.5, -0.75, 1.0]))
    assert [update.name for update in result.updates] == [block.name for block in blocks]
    assert all(update.state_token == coordinate_map.state_token for update in result.updates)
    assert result.updates[0].delta.dtype == np.asarray(blocks[0].values).dtype
    assert result.updates[1].delta.dtype == np.asarray(blocks[1].values).dtype
    np.testing.assert_array_equal(result.updates[0].delta, np.asarray(0.5))
    np.testing.assert_array_equal(result.updates[1].delta, [-0.75 + 1.0j])
    assert isinstance(result.diagnostics, OptimizationDiagnostics)
    assert result.diagnostics.num_iterations == 7
    np.testing.assert_array_equal(result.diagnostics.metric_diagonal, [4.0, 5.0, 6.0])


def test_optimizer_diagnostic_computation_is_logger_level_independent(monkeypatch):
    """Optimizer diagnostic computation is logger level independent."""
    blocks = _blocks()
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    target = _target(coordinate_map)
    result = finalize_optimization_step(target, coordinate_map, learning_rate=0.25)
    debug_calls = []

    def record_debug(message, *args, **kwargs):
        debug_calls.append((message, args, kwargs))

    monkeypatch.setattr(optimizer_module.logger, "debug", record_debug)
    optimizer_module.logger.setLevel(logging.INFO)
    _log_optimization_diagnostics(
        target,
        result,
        coordinate_map,
        use_lm=False,
        lm_subspace_dim=0,
        cg=True,
    )

    coordinate_labels = [args[0] for message, args, _kwargs in debug_calls if message.startswith("    %s: force=")]
    assert coordinate_labels == ["z[q1]", "z[q0]", "j1_param[q0]"]


def test_atomic_update_rejects_bad_publication_before_constructing_candidate():
    """Atomic update rejects bad publication before constructing candidate."""
    block = _blocks()[0]
    wf = Wavefunction_data(
        jastrow_data=Jastrow_data(
            jastrow_one_body_data=Jastrow_one_body_data(
                jastrow_1b_param=0.4,
                jastrow_1b_type="pade",
            )
        ),
        geminal_data=Geminal_data(lambda_matrix=np.empty((0, 0))),
    )
    good = BlockStorageUpdate(name=block.name, state_token=block.state_token, delta=np.asarray(0.1))

    with pytest.raises(ValueError, match="stale"):
        wf.apply_optimization_updates(
            blocks=(block,),
            updates=(replace(good, state_token=("test", 99)),),
            expected_state_token=block.state_token,
        )
    with pytest.raises(ValueError, match="non-finite"):
        wf.apply_optimization_updates(
            blocks=(block,),
            updates=(replace(good, delta=np.asarray(np.nan)),),
            expected_state_token=block.state_token,
        )

    candidate = wf.apply_optimization_updates(
        blocks=(block,),
        updates=(good,),
        expected_state_token=block.state_token,
    )
    np.testing.assert_allclose(
        candidate.jastrow_data.jastrow_one_body_data.jastrow_1b_param, 0.5, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT
    )
    np.testing.assert_allclose(
        wf.jastrow_data.jastrow_one_body_data.jastrow_1b_param, 0.4, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT
    )
