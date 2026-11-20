"""C5 scalar real-coordinate expansion and real-adjoint contracts."""

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

from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_ZERO
from jqmc.jqmc_optimizer import OptimizationCoordinateMap, RealCoordinateMap

from .optimizer_metadata_helpers import make_test_block


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _real_block(name: str, values) -> object:
    return make_test_block(
        name,
        np.asarray(values, dtype=np.float64),
        field="real",
        derivative_policy="direct_real",
        sync_scope="sampler",
    )


def _complex_block(name: str, values) -> object:
    return make_test_block(
        name,
        np.asarray(values, dtype=np.complex128),
        field="complex",
        derivative_policy="holomorphic",
        sync_scope="sampler",
    )


@pytest.mark.parametrize(
    ("blocks", "q", "expected"),
    [
        ([_real_block("r", [0.0, 0.0])], [1.0, 2.0], [1.0, 2.0]),
        ([_complex_block("z", [0.0j, 0.0j])], [1.0, 2.0, 3.0, 4.0], [1.0 + 2.0j, 3.0 + 4.0j]),
        (
            [_real_block("left", [0.0]), _complex_block("middle", [0.0j]), _real_block("right", [0.0])],
            [10.0, 20.0, 30.0, 40.0],
            [10.0, 20.0 + 30.0j, 40.0],
        ),
    ],
)
def test_encode_exact_real_complex_and_mixed_layouts(blocks, q, expected):
    """Encode exact real complex and mixed layouts."""
    coordinate_map = RealCoordinateMap.from_blocks(blocks)
    np.testing.assert_array_equal(coordinate_map.encode(np.asarray(q)), np.asarray(expected))


def test_mixed_layout_metadata_uses_block_c_order_then_xy_pairs():
    """Mixed layout metadata uses block c order then xy pairs."""
    blocks = [
        _real_block("left", np.zeros((1, 2))),
        _complex_block("middle", np.zeros((2, 1), dtype=np.complex128)),
        _real_block("right", np.zeros(1)),
    ]
    coordinate_map = RealCoordinateMap.from_blocks(blocks)

    assert coordinate_map.block_names == ("left", "middle", "right")
    assert coordinate_map.block_fields == ("real", "complex", "real")
    assert coordinate_map.block_shapes == ((1, 2), (2, 1), (1,))
    assert coordinate_map.storage_size == 5
    assert coordinate_map.coordinate_size == 7
    assert tuple((part.start, part.stop) for part in coordinate_map.block_storage_slices) == ((0, 2), (2, 4), (4, 5))
    assert tuple((part.start, part.stop) for part in coordinate_map.block_coordinate_slices) == ((0, 2), (2, 6), (6, 7))
    assert coordinate_map.storage_to_coordinate == ((0,), (1,), (2, 3), (4, 5), (6,))
    assert coordinate_map.coordinate_groups == coordinate_map.storage_to_coordinate


def test_batched_actions_match_independently_assembled_dense_embedding_and_adjoint_identity():
    """Batched actions match independently assembled dense embedding and adjoint identity."""
    blocks = [
        _real_block("r0", np.zeros(2)),
        _complex_block("z", np.zeros(2, dtype=np.complex128)),
        _real_block("r1", np.zeros(1)),
    ]
    coordinate_map = RealCoordinateMap.from_blocks(blocks)

    # Independent test-side embedding: rows are storage scalars, columns are
    # real coordinates.  This deliberately does not inspect map internals.
    dense = np.zeros((5, 7), dtype=np.complex128)
    dense[0, 0] = 1.0
    dense[1, 1] = 1.0
    dense[2, 2] = 1.0
    dense[2, 3] = 1.0j
    dense[3, 4] = 1.0
    dense[3, 5] = 1.0j
    dense[4, 6] = 1.0

    rng = np.random.default_rng(20260726)
    q = rng.standard_normal((4, 3, 7))
    u = rng.standard_normal((4, 3, 5)) + 1.0j * rng.standard_normal((4, 3, 5))
    encoded = coordinate_map.encode(q)
    pulled_back = coordinate_map.real_adjoint(u)

    np.testing.assert_allclose(encoded, q @ dense.T, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    np.testing.assert_allclose(
        pulled_back,
        np.real(u @ dense.conj()),
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    lhs = np.real(np.sum(u.conj() * encoded, axis=-1))
    rhs = np.sum(pulled_back * q, axis=-1)
    np.testing.assert_allclose(lhs, rhs, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)


def test_real_adjoint_uses_hermitian_not_transpose_sign_for_imaginary_direction():
    """Real adjoint uses hermitian not transpose sign for imaginary direction."""
    coordinate_map = RealCoordinateMap.from_blocks([_complex_block("z", [0.0j])])
    storage_covector = np.asarray([1.0 + 2.0j])
    dense = np.asarray([[1.0, 1.0j]])

    np.testing.assert_array_equal(coordinate_map.real_adjoint(storage_covector), np.asarray([1.0, 2.0]))
    transpose_shortcut = np.real(storage_covector @ dense)
    np.testing.assert_array_equal(transpose_shortcut, np.asarray([1.0, -2.0]))


def test_actions_preserve_float32_complex64_field_precision():
    """Actions preserve float32 complex64 field precision."""
    real = make_test_block(
        "r32",
        np.zeros(1, dtype=np.float32),
        field="real",
        derivative_policy="direct_real",
        sync_scope="sampler",
    )
    complex_ = make_test_block(
        "z64",
        np.zeros(1, dtype=np.complex64),
        field="complex",
        derivative_policy="holomorphic",
        sync_scope="sampler",
    )
    coordinate_map = RealCoordinateMap.from_blocks([real, complex_])

    encoded = coordinate_map.encode(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    pulled_back = coordinate_map.real_adjoint(np.asarray([4.0 + 5.0j, 6.0 + 7.0j], dtype=np.complex64))

    assert coordinate_map.storage_dtype == np.dtype(np.complex64)
    assert encoded.dtype == np.complex64
    assert pulled_back.dtype == np.float32
    np.testing.assert_array_equal(encoded, np.asarray([1.0, 2.0 + 3.0j], dtype=np.complex64))
    np.testing.assert_array_equal(pulled_back, np.asarray([4.0, 6.0, 7.0], dtype=np.float32))


def test_scalar_map_rejects_invalid_blocks_and_action_shapes():
    """Scalar map rejects invalid blocks and action shapes."""
    with pytest.raises(ValueError, match="at least one"):
        RealCoordinateMap.from_blocks([])

    coordinate_map = RealCoordinateMap.from_blocks([_real_block("r", [0.0]), _complex_block("z", [0.0j])])
    with pytest.raises(ValueError, match="coordinate axis"):
        coordinate_map.encode(np.zeros(2))
    with pytest.raises(ValueError, match="real-valued"):
        coordinate_map.encode(np.zeros(3, dtype=np.complex128))
    with pytest.raises(ValueError, match="storage axis"):
        coordinate_map.real_adjoint(np.zeros(1, dtype=np.complex128))


def test_complete_coordinate_map_owns_the_validated_scalar_field_map():
    """Complete coordinate map owns the validated scalar field map."""
    blocks = [_real_block("r", [0.0]), _complex_block("z", [0.0j])]
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)

    assert coordinate_map.field_map.block_names == ("r", "z")
    assert coordinate_map.field_map.storage_size == 2
    assert coordinate_map.field_map.coordinate_size == 3
