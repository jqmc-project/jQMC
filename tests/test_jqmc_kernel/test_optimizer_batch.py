"""C8 exact-source optimization batch and jackknife metadata contracts."""

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

import numpy as np
import pytest

from jqmc.jqmc_optimizer import (
    BlockDerivativeSource,
    EnergyEstimate,
    JackknifeBinMetadata,
    JackknifeBlockMoments,
    OptimizationBatch,
)

from .optimizer_metadata_helpers import TEST_STATE_TOKEN, make_test_block


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _blocks():
    return (
        make_test_block(
            "r",
            np.asarray(0.0),
            field="real",
            derivative_policy="direct_real",
            sync_scope="world",
        ),
        make_test_block(
            "z",
            np.zeros(1, dtype=np.complex128),
            field="complex",
            derivative_policy="holomorphic",
            sync_scope="sampler",
        ),
    )


def _source(block, values):
    if block.derivative_policy == "direct_real":
        return BlockDerivativeSource.build(block=block, direct=values)
    return BlockDerivativeSource.build(block=block, d_p=values)


def _moments(block, offset: float):
    shape = (3, *block.shape)
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + offset
    values = values + 1.0j * (values + 0.25)
    return JackknifeBlockMoments.build(
        block=block,
        source_sum=_source(block, values),
        conjugate_source_energy_sum=_source(block, 3.0 * values),
    )


def _batch():
    blocks = _blocks()
    raw = tuple(
        _source(block, np.ones((6, *block.shape), dtype=np.complex128) * (index + 1.0j))
        for index, block in enumerate(blocks, start=1)
    )
    token = "provider-7:12"
    metadata = JackknifeBinMetadata.build(
        blocks=blocks,
        batch_token=token,
        bin_sample_counts=np.asarray([2, 2, 2]),
        bin_weight_sums=np.asarray([1.5, 2.5, 3.5]),
        bin_weighted_energy_sums=np.asarray([1.0 + 0.2j, 2.0 - 0.1j, 3.0 + 0.4j]),
        log_source_moments=tuple(_moments(block, index) for index, block in enumerate(blocks)),
    )
    return blocks, OptimizationBatch.build(
        blocks=blocks,
        batch_token=token,
        state_token=TEST_STATE_TOKEN,
        weights=np.asarray([1.0, 2.0, 1.5, 0.5, 3.0, 2.5]),
        local_energies=np.asarray([1.0 + 0.1j, 2.0 - 0.2j, 3.0, 4.0 + 0.3j, 5.0, 6.0 - 0.1j]),
        log_derivative_sources=raw,
        local_energy_derivative_sources=raw,
        jackknife_bin_metadata=metadata,
    )


def test_energy_estimate_is_real_and_immutable():
    """Energy estimate is real and immutable."""
    estimate = EnergyEstimate(mean=-1.2, standard_error=0.1, variance=0.4, variance_standard_error=0.05)
    assert estimate.mean == -1.2
    with pytest.raises(ValueError, match="real scalar"):
        EnergyEstimate(mean=1.0 + 2.0j, standard_error=0.1, variance=0.4, variance_standard_error=0.05)


def test_batch_is_breaking_exact_source_schema_without_log_derivatives_alias():
    """Batch is breaking exact source schema without log derivatives alias."""
    blocks, batch = _batch()

    assert batch.state_token == blocks[0].state_token
    assert batch.jackknife_bin_metadata.batch_token == batch.batch_token
    assert not hasattr(batch, "log_derivatives")
    assert [source.name for source in batch.log_derivative_sources] == ["r", "z"]
    assert not batch.weights.flags.writeable
    assert not batch.local_energies.flags.writeable
    assert not batch.jackknife_bin_metadata.bin_sample_counts.flags.writeable


def test_batch_rejects_cross_batch_state_and_source_mismatches():
    """Batch rejects cross batch state and source mismatches."""
    blocks, batch = _batch()
    with pytest.raises(ValueError, match="batch_token"):
        OptimizationBatch.build(
            blocks=blocks,
            batch_token="other:1",
            state_token=batch.state_token,
            weights=batch.weights,
            local_energies=batch.local_energies,
            log_derivative_sources=batch.log_derivative_sources,
            local_energy_derivative_sources=batch.local_energy_derivative_sources,
            jackknife_bin_metadata=batch.jackknife_bin_metadata,
        )
    with pytest.raises(ValueError, match="state_token"):
        OptimizationBatch.build(
            blocks=blocks,
            batch_token=batch.batch_token,
            state_token=("stale", 2),
            weights=batch.weights,
            local_energies=batch.local_energies,
            log_derivative_sources=batch.log_derivative_sources,
            local_energy_derivative_sources=batch.local_energy_derivative_sources,
            jackknife_bin_metadata=batch.jackknife_bin_metadata,
        )
    with pytest.raises(ValueError, match="finite real"):
        OptimizationBatch.build(
            blocks=blocks,
            batch_token=batch.batch_token,
            state_token=batch.state_token,
            weights=np.asarray([1.0, np.inf, 1.0, 1.0, 1.0, 1.0]),
            local_energies=batch.local_energies,
            log_derivative_sources=batch.log_derivative_sources,
            local_energy_derivative_sources=batch.local_energy_derivative_sources,
            jackknife_bin_metadata=batch.jackknife_bin_metadata,
        )


def test_jackknife_metadata_rejects_empty_local_bins_and_moment_order():
    """Jackknife metadata rejects empty local bins and moment order."""
    blocks, batch = _batch()
    metadata = batch.jackknife_bin_metadata
    with pytest.raises(ValueError, match="at least one"):
        JackknifeBinMetadata.build(
            blocks=blocks,
            batch_token=batch.batch_token,
            bin_sample_counts=np.asarray([], dtype=np.int64),
            bin_weight_sums=np.asarray([]),
            bin_weighted_energy_sums=np.asarray([]),
            log_source_moments=metadata.log_source_moments,
        )
    with pytest.raises(ValueError, match="order/name"):
        JackknifeBinMetadata.build(
            blocks=blocks,
            batch_token=batch.batch_token,
            bin_sample_counts=metadata.bin_sample_counts,
            bin_weight_sums=metadata.bin_weight_sums,
            bin_weighted_energy_sums=metadata.bin_weighted_energy_sums,
            log_source_moments=tuple(reversed(metadata.log_source_moments)),
        )
