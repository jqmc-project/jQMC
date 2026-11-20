"""C6 exact derivative-source schema and validation contracts."""

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
from jqmc._setting import ATOL_NONVACUITY_DISTINCTION, RTOL_NONVACUITY_DISTINCTION

import numpy as np
import pytest

from jqmc._setting import ATOL_PYTHON_FP64_AGGREGATOR_LOOSE, RTOL_PYTHON_FP64_AGGREGATOR_LOOSE
from jqmc.jqmc_optimizer import (
    BlockDerivativeSource,
    JackknifeBinMetadata,
    JackknifeBlockMoments,
    OptimizationBatch,
    validate_derivative_sources,
)

from .optimizer_metadata_helpers import TEST_STATE_TOKEN, make_test_block


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _block(name: str, shape: tuple[int, ...], *, field: str, policy: str):
    dtype = np.float64 if field == "real" else np.complex128
    return make_test_block(
        name,
        np.zeros(shape, dtype=dtype),
        field=field,
        derivative_policy=policy,
        sync_scope="sampler",
    )


def test_direct_real_preserves_the_complete_complex_sample_array():
    """Direct real preserves the complete complex sample array."""
    block = _block("j2_param", (2,), field="real", policy="direct_real")
    direct = np.asarray(
        [
            [[1.0 + 2.0j, 3.0 - 4.0j]],
            [[-5.0 + 6.0j, 7.0 + 8.0j]],
        ]
    )

    source = BlockDerivativeSource.build(block=block, direct=direct)

    assert source.name == block.name
    np.testing.assert_array_equal(source.direct, direct)
    assert source.d_p is None
    assert source.d_pbar is None
    assert not source.direct.flags.writeable


def test_holomorphic_source_keeps_A_and_structural_zero_separate():
    """Holomorphic source keeps A and structural zero separate."""
    block = _block("lambda_unpaired", (1, 2), field="complex", policy="holomorphic")
    A = np.asarray([[[[1.0 + 3.0j, 2.0 - 5.0j]]]])

    source = BlockDerivativeSource.build(block=block, d_p=A)

    np.testing.assert_array_equal(source.d_p, A)
    assert source.direct is None
    assert source.d_pbar is None


def test_wirtinger_pair_retains_A_and_B_without_conjugate_folding():
    """Wirtinger pair retains A and B without conjugate folding."""
    block = _block("mo_coefficients_up", (1, 2), field="complex", policy="wirtinger_pair")
    A = np.asarray([[[[1.0 + 2.0j, 3.0 + 4.0j]]]])
    B = np.asarray([[[[5.0 - 6.0j, 7.0 - 8.0j]]]])

    source = BlockDerivativeSource.build(block=block, d_p=A, d_pbar=B)

    np.testing.assert_array_equal(source.d_p, A)
    np.testing.assert_array_equal(source.d_pbar, B)
    assert not np.array_equal(source.d_p, A + np.conj(B))
    assert not source.d_p.flags.writeable
    assert not source.d_pbar.flags.writeable


@pytest.mark.parametrize(
    ("A", "B"),
    [
        (1.2 + 0.7j, -0.4 + 1.1j),  # logarithmic-derivative-shaped probe
        (-0.8 + 1.6j, 0.9 - 0.3j),  # local-energy-derivative-shaped probe
    ],
)
def test_two_component_source_matches_xy_directional_fd_and_conjugate_fold_does_not(A, B):
    """A tied source is ``A dp + B d(pbar)`` for both O and D.

    This independent scalar oracle deliberately chooses a complex ``B`` so
    ``A + conj(B)`` cannot accidentally represent both real directions.
    """
    p0 = 0.3 - 0.2j

    def observable(p):
        return A * p + B * np.conj(p)

    step = 1.0e-7
    fd_x = (observable(p0 + step) - observable(p0 - step)) / (2.0 * step)
    fd_y = (observable(p0 + 1.0j * step) - observable(p0 - 1.0j * step)) / (2.0 * step)

    np.testing.assert_allclose(
        fd_x,
        A + B,
        rtol=RTOL_PYTHON_FP64_AGGREGATOR_LOOSE,
        atol=ATOL_PYTHON_FP64_AGGREGATOR_LOOSE,
    )
    np.testing.assert_allclose(
        fd_y,
        1.0j * (A - B),
        rtol=RTOL_PYTHON_FP64_AGGREGATOR_LOOSE,
        atol=ATOL_PYTHON_FP64_AGGREGATOR_LOOSE,
    )
    assert not np.isclose(A + np.conj(B), fd_x, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION)
    assert not np.isclose(1.0j * (A + np.conj(B)), fd_y, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION)


@pytest.mark.parametrize(
    ("policy", "kwargs", "match"),
    [
        ("direct_real", {"d_p": np.ones((2, 1))}, "direct_real"),
        ("direct_real", {"direct": np.ones((2, 1)), "d_p": np.ones((2, 1))}, "direct_real"),
        ("holomorphic", {"direct": np.ones((2, 1))}, "holomorphic"),
        ("holomorphic", {"d_p": np.ones((2, 1)), "d_pbar": np.zeros((2, 1))}, "holomorphic"),
        ("wirtinger_pair", {"d_p": np.ones((2, 1))}, "wirtinger_pair"),
    ],
)
def test_build_rejects_component_sets_that_disagree_with_block_policy(policy, kwargs, match):
    """Build rejects component sets that disagree with block policy."""
    field = "real" if policy == "direct_real" else "complex"
    block = _block("p", (1,), field=field, policy=policy)
    with pytest.raises(ValueError, match=match):
        BlockDerivativeSource.build(block=block, **kwargs)


def test_build_rejects_wrong_trailing_block_shape_and_component_shape_mismatch():
    """Build rejects wrong trailing block shape and component shape mismatch."""
    direct_block = _block("r", (2, 2), field="real", policy="direct_real")
    with pytest.raises(ValueError, match="trailing shape"):
        BlockDerivativeSource.build(block=direct_block, direct=np.zeros((4, 2, 3)))

    paired_block = _block("z", (2,), field="complex", policy="wirtinger_pair")
    with pytest.raises(ValueError, match="same shape"):
        BlockDerivativeSource.build(
            block=paired_block,
            d_p=np.zeros((4, 2)),
            d_pbar=np.zeros((5, 2)),
        )


def test_publication_validation_rejects_missing_reordered_and_mixed_sample_shapes():
    """Publication validation rejects missing reordered and mixed sample shapes."""
    real = _block("real", (1,), field="real", policy="direct_real")
    complex_ = _block("complex", (2,), field="complex", policy="holomorphic")
    r_source = BlockDerivativeSource.build(block=real, direct=np.zeros((3, 4, 1)))
    z_source = BlockDerivativeSource.build(block=complex_, d_p=np.zeros((3, 4, 2)))

    published = validate_derivative_sources([real, complex_], [r_source, z_source])
    assert published == (r_source, z_source)

    with pytest.raises(ValueError, match="same length"):
        validate_derivative_sources([real, complex_], [r_source])
    with pytest.raises(ValueError, match="order/name"):
        validate_derivative_sources([real, complex_], [z_source, r_source])

    wrong_samples = BlockDerivativeSource.build(block=complex_, d_p=np.zeros((3, 5, 2)))
    with pytest.raises(ValueError, match="sample shape"):
        validate_derivative_sources([real, complex_], [r_source, wrong_samples])

    mutable = BlockDerivativeSource(name="real", direct=np.zeros((3, 4, 1)), d_p=None, d_pbar=None)
    with pytest.raises(ValueError, match="immutable"):
        validate_derivative_sources([real], [mutable])


def test_build_defensively_copies_inputs_without_changing_source_dtype():
    """Build defensively copies inputs without changing source dtype."""
    block = _block("r", (1,), field="real", policy="direct_real")
    values = np.asarray([[1.0], [2.0]], dtype=np.float32)
    source = BlockDerivativeSource.build(block=block, direct=values)
    values[0, 0] = 99.0

    assert source.direct.dtype == np.float32
    np.testing.assert_array_equal(source.direct, np.asarray([[1.0], [2.0]], dtype=np.float32))


def test_published_sources_stay_isolated_from_writable_buffers():
    """A built or batched source keeps its published values after every writable buffer it came from is overwritten."""
    block = make_test_block("r_param", np.zeros(3), field="real", derivative_policy="direct_real", sync_scope="sampler")
    rng = np.random.default_rng(840)
    expected = rng.normal(size=(8, 3))
    owner = expected.copy()
    readonly_view_of_writable = owner[2:].view()
    readonly_view_of_writable.setflags(write=False)
    strided = np.repeat(expected, 2, axis=0)
    strided.setflags(write=False)
    frozen = expected.copy()
    frozen.setflags(write=False)
    sources = {
        "writable": (BlockDerivativeSource.build(block=block, direct=owner), expected),
        "read-only view of writable memory": (
            BlockDerivativeSource.build(block=block, direct=readonly_view_of_writable),
            expected[2:],
        ),
        "non-contiguous read-only view": (BlockDerivativeSource.build(block=block, direct=strided[::2]), expected),
        "immutable stack": (BlockDerivativeSource.build(block=block, direct=frozen), expected),
    }
    weights = np.ones(8)
    metadata = JackknifeBinMetadata.build(
        blocks=(block,),
        batch_token="isolation:1",
        bin_sample_counts=np.full(2, 4, dtype=np.int64),
        bin_weight_sums=np.full(2, 4.0),
        bin_weighted_energy_sums=np.zeros(2),
        log_source_moments=(
            JackknifeBlockMoments.build(
                block=block,
                source_sum=BlockDerivativeSource.build(block=block, direct=owner.reshape(2, 4, 3).sum(axis=1)),
                conjugate_source_energy_sum=BlockDerivativeSource.build(block=block, direct=np.zeros((2, 3))),
            ),
        ),
    )
    batch = OptimizationBatch.build(
        blocks=(block,),
        batch_token="isolation:1",
        state_token=TEST_STATE_TOKEN,
        weights=weights,
        local_energies=np.zeros(8),
        log_derivative_sources=(sources["writable"][0],),
        local_energy_derivative_sources=(sources["immutable stack"][0],),
        jackknife_bin_metadata=metadata,
    )
    owner[...] = np.nan
    weights[...] = np.nan
    for label, (source, values) in sources.items():
        assert not source.direct.flags.writeable, f"{label} source must be published read-only"
        np.testing.assert_array_equal(source.direct, values, err_msg=f"{label} source changed after its buffer was written")
    np.testing.assert_array_equal(batch.log_derivative_sources[0].direct, expected)
    np.testing.assert_array_equal(batch.local_energy_derivative_sources[0].direct, expected)
    np.testing.assert_array_equal(batch.weights, np.ones(8))
    np.testing.assert_array_equal(
        batch.jackknife_bin_metadata.log_source_moments[0].source_sum.direct,
        expected.reshape(2, 4, 3).sum(axis=1),
    )
