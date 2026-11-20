"""Exact dyadic controls for periodic image-fold tails and logical widths."""

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
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import atomic_orbital
from jqmc._precision import get_dtype_jnp

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize("images", [32, 33, 179])
@pytest.mark.parametrize("width", [1, 5])
def test_ragged_fold_odd_images_and_logical_width(images, width):
    """Exact dyadic inputs pin odd tails, bucket permutation and both logical widths.

    All inputs and intermediate integer sums fit exactly in fp32 and fp64,
    so this checks the sums themselves; batch-width bit identity for
    arbitrary inputs is pinned separately below. Column-vmap folds each
    single-electron column of the same wider input.
    """
    dtype = get_dtype_jnp("ao_eval")
    rows = images * 2 + 3
    numerators = (np.arange(rows * width, dtype=np.int64).reshape(rows, width) % 101) - 50
    numerators[images * 2 - 1] += 104
    buckets = [{"m": images, "K": 2, "kernel_offset": 0}, {"m": 3, "K": 1, "kernel_offset": images * 2}]
    permutation = np.array([2, 0, 1], dtype=np.int32)
    homes = np.concatenate([np.tile(np.array([0, 1]), images), np.full(3, 2)])
    exact_integer_sums = np.array(
        [[sum(int(v) for v in numerators[homes == home, column]) for column in range(width)] for home in permutation]
    )
    expected = np.asarray(exact_integer_sums / 32, dtype=dtype)
    kernel = jnp.asarray(numerators, dtype=dtype) / 32

    def fold(values):
        return atomic_orbital._fold_image_axis_ragged(values, buckets, permutation)

    actual = fold(kernel)
    compiled = jax.jit(fold)(kernel)
    columnwise = jax.jit(jax.vmap(lambda column: fold(column[:, None])[:, 0], in_axes=1, out_axes=1))(kernel)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(compiled, expected)
    np.testing.assert_array_equal(columnwise, expected)
    assert actual.dtype == jnp.dtype(dtype)


@pytest.mark.parametrize("width", [2, 7, 64])
def test_ragged_fold_float32_single_column_bitwise_equals_full_width(width):
    """Folding one float32 electron column alone gives bitwise the full-width fold.

    In mixed precision a carried Sherman-Morrison inverse is updated from
    single-electron fp32 AO rows while a fresh inverse is formed from the
    full-width geminal, so the fp32 image fold must not round a column
    differently when the table is wider. The input is float32 in both
    precision modes: it is the dtype mixed ao_eval folds, and this pins that
    fold's contract (the fp64 fold keeps its separately pinned bit-exact
    sequential sum). The static buckets are those of a production periodic
    layout: odd and even multiplicities from 8 to 179 images, one to five
    home AOs per bucket, and a permuted AO order. PCG64(seed=20260929) draws
    non-dyadic values with magnitudes spread over several decades. Each
    column folded alone (eager, jit and column-vmap) must equal the matching
    column of the jitted full-width fold exactly. This is a batch-width
    invariance pin, not an accuracy claim.
    """
    dtype = jnp.float32
    layouts = [(8, 1), (19, 1), (27, 3), (32, 3), (33, 2), (88, 5), (93, 5), (160, 1), (179, 1)]
    buckets, offset = [], 0
    for images, homes in layouts:
        buckets.append({"m": images, "K": homes, "kernel_offset": offset})
        offset += images * homes
    rng = np.random.default_rng(20260929)
    permutation = rng.permutation(sum(homes for _, homes in layouts)).astype(np.int32)
    values = rng.normal(size=(offset, width)) * np.exp(rng.normal(scale=3.0, size=(offset, 1)))
    kernel = jnp.asarray(values, dtype=dtype)

    def fold(table):
        return atomic_orbital._fold_image_axis_ragged(table, buckets, permutation)

    full_width = np.asarray(jax.jit(fold)(kernel))
    single_jit = jax.jit(fold)
    eager_columns = np.stack([np.asarray(fold(kernel[:, [column]]))[:, 0] for column in range(width)], axis=1)
    jit_columns = np.stack([np.asarray(single_jit(kernel[:, [column]]))[:, 0] for column in range(width)], axis=1)
    vmap_columns = np.asarray(jax.jit(jax.vmap(lambda column: fold(column[:, None])[:, 0], in_axes=1, out_axes=1))(kernel))
    assert full_width.dtype == np.dtype(dtype)
    assert np.all(np.isfinite(full_width))
    np.testing.assert_array_equal(eager_columns, full_width)
    np.testing.assert_array_equal(jit_columns, full_width)
    np.testing.assert_array_equal(vmap_columns, full_width)
