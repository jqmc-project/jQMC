"""Retained sample-row wrapper preserves values and actual local CPU placement."""

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

from jqmc import jqmc_optimizer_kernel as kernel

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize("input_placement", ["host", "named-sharded"])
def test_public_sample_rows_preserves_order_values_and_local_sharding(input_placement):
    """Sample-leading arrays retain exact values, tuple order and local device ownership.

    Deterministic integer vectors, real matrices and complex rank-three arrays
    use two rows per available local CPU device. Named-sharded inputs use a
    real JAX mesh, not an emulated MPI rank. This verifies only the public
    single-process route; it does not certify distributed process stitching.
    """
    devices = jax.local_devices(backend="cpu")
    assert jax.process_count() == 1
    rows = 2 * len(devices)
    references = (
        np.arange(rows, dtype=np.int32),
        np.arange(rows * 3, dtype=np.float32).reshape(rows, 3) / 8,
        (np.arange(rows * 4).reshape(rows, 2, 2) * (1 + 2j)).astype(np.complex64),
    )
    if input_placement == "named-sharded":
        mesh = jax.sharding.Mesh(np.asarray(devices), ("sample",))
        inputs = tuple(
            jax.device_put(
                array, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("sample", *([None] * (array.ndim - 1))))
            )
            for array in references
        )
    else:
        inputs = references
    outputs = kernel.shard_sample_rows(*inputs)
    assert isinstance(outputs, tuple) and len(outputs) == len(references)
    for actual, supplied, reference in zip(outputs, inputs, references, strict=True):
        assert isinstance(actual, jax.Array)
        assert actual.shape == reference.shape and actual.dtype == reference.dtype
        np.testing.assert_array_equal(actual, reference)
        assert actual.sharding.device_set <= set(devices)
        if input_placement == "named-sharded":
            assert actual.sharding.is_equivalent_to(supplied.sharding, actual.ndim)
        covered = np.zeros(rows, dtype=np.int32)
        for shard in actual.addressable_shards:
            np.testing.assert_array_equal(shard.data, reference[shard.index])
            covered[shard.index[0]] += 1
        np.testing.assert_array_equal(covered, np.ones(rows, dtype=np.int32))


def test_public_sample_rows_accepts_empty_argument_tuple():
    """No sample arrays yield exactly an empty tuple without creating an array."""
    assert kernel.shard_sample_rows() == ()
