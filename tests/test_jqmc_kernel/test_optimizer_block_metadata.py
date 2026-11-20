"""C4 contracts for optimizer-facing variational-parameter metadata."""

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

from jqmc._setting import (
    ATOL_PYTHON_FP64_LOOSE,
    RTOL_PYTHON_FP64_ZERO,
    get_s_stiefel_backward_error_limit,
)
from jqmc.jqmc_optimizer import OptimizationCoordinateMap
from jqmc.wavefunction import (
    LinearTieSpec,
    ParameterTangentSpec,
    VariationalParameterBlock,
    _build_s_stiefel_metric_factors,
    _hermitian_paired_linear_tie,
    _project_linear_tie_storage,
    _symmetric_paired_linear_tie,
    _symmetric_real_linear_tie,
)

TOKEN = ("metadata-test-lineage", 3)
pytestmark = pytest.mark.without_chdir


def _euclidean_block(
    name: str,
    values: np.ndarray,
    *,
    field: str,
    derivative_policy: str,
    sync_scope: str = "sampler",
    token: tuple[str, int] = TOKEN,
) -> VariationalParameterBlock:
    return VariationalParameterBlock.build(
        name=name,
        values=values,
        field=field,
        derivative_policy=derivative_policy,
        sync_scope=sync_scope,
        tangent_spec=ParameterTangentSpec.euclidean(state_token=token),
        state_token=token,
    )


@pytest.mark.light
def test_block_metadata_is_required_immutable_and_preserved_by_update():
    """Block metadata is required immutable and preserved by update."""
    source = np.array([1.0, 2.0], dtype=np.float64)
    block = _euclidean_block(
        "j2_param",
        source,
        field="real",
        derivative_policy="direct_real",
        sync_scope="world",
    )

    source[0] = 99.0
    np.testing.assert_array_equal(block.values, np.array([1.0, 2.0]))
    assert not np.asarray(block.values).flags.writeable
    with pytest.raises(ValueError):
        np.asarray(block.values)[0] = -1.0

    updated = block.apply_update(np.array([0.25, -0.5]), learning_rate=2.0)
    np.testing.assert_array_equal(updated.values, np.array([1.5, 1.0]))
    assert updated.field == "real"
    assert updated.derivative_policy == "direct_real"
    assert updated.sync_scope == "world"
    assert updated.tangent_spec is block.tangent_spec
    assert updated.state_token == TOKEN

    with pytest.raises(TypeError):
        VariationalParameterBlock.build(name="missing_metadata", values=np.ones(1))


@pytest.mark.light
@pytest.mark.parametrize(
    ("values", "field", "policy"),
    [
        (np.ones(2, dtype=np.complex128), "real", "direct_real"),
        (np.ones(2, dtype=np.float64), "complex", "holomorphic"),
        (np.ones(2, dtype=np.float64), "real", "holomorphic"),
        (np.ones(2, dtype=np.complex128), "complex", "direct_real"),
    ],
)
def test_block_rejects_inconsistent_field_policy_and_storage(values, field, policy):
    """Block rejects inconsistent field policy and storage."""
    with pytest.raises(ValueError):
        _euclidean_block("bad", values, field=field, derivative_policy=policy)


@pytest.mark.light
def test_coordinate_map_rejects_duplicate_mixed_and_mismatched_tokens():
    """Coordinate map rejects duplicate mixed and mismatched tokens."""
    first = _euclidean_block("a", np.ones(2), field="real", derivative_policy="direct_real")
    duplicate = _euclidean_block("a", np.ones(1), field="real", derivative_policy="direct_real")
    with pytest.raises(ValueError, match="duplicate"):
        OptimizationCoordinateMap.from_blocks([first, duplicate])

    other = _euclidean_block(
        "b",
        np.ones(1),
        field="real",
        derivative_policy="direct_real",
        token=("other-lineage", 0),
    )
    with pytest.raises(ValueError, match="state_token"):
        OptimizationCoordinateMap.from_blocks([first, other])

    with pytest.raises(ValueError, match="tangent_spec"):
        first.replace(
            tangent_spec=ParameterTangentSpec.euclidean(state_token=("stale-spec", 2)),
        )


@pytest.mark.light
def test_block_rejects_incomplete_structural_metadata():
    """Block rejects incomplete structural metadata."""
    with pytest.raises(ValueError, match="name"):
        _euclidean_block("", np.ones(1), field="real", derivative_policy="direct_real")
    with pytest.raises(ValueError, match="synchronization scope"):
        _euclidean_block(
            "bad_scope",
            np.ones(1),
            field="real",
            derivative_policy="direct_real",
            sync_scope="node",
        )
    with pytest.raises(ValueError, match="at least one"):
        _euclidean_block("empty", np.empty(0), field="real", derivative_policy="direct_real")
    with pytest.raises(ValueError, match="fully validated"):
        OptimizationCoordinateMap.from_blocks([object()])


@pytest.mark.light
def test_stiefel_metadata_matches_the_published_mo_storage_shape():
    """Stiefel metadata matches the published MO storage shape."""
    spec = ParameterTangentSpec.s_stiefel_rows(
        metric_sqrt=np.eye(3),
        metric_inv_sqrt=np.eye(3),
        state_token=TOKEN,
    )
    block = VariationalParameterBlock.build(
        name="mo_coefficients_up",
        values=np.ones((2, 3), dtype=np.float64),
        field="real",
        derivative_policy="direct_real",
        sync_scope="sampler",
        tangent_spec=spec,
        state_token=TOKEN,
    )
    assert block.shape == (2, 3)

    with pytest.raises(ValueError, match="column count"):
        block.replace(values=np.ones((2, 4)), shape=(2, 4), size=8)


@pytest.mark.light
def test_complex_linear_tie_requires_two_component_wirtinger_policy():
    """Complex linear tie requires two component wirtinger policy."""
    tie = _hermitian_paired_linear_tie((2, 2), n_paired=2)
    with pytest.raises(ValueError, match="two-component Wirtinger"):
        VariationalParameterBlock.build(
            name="lambda_matrix",
            values=np.eye(2, dtype=np.complex128),
            field="complex",
            derivative_policy="holomorphic",
            sync_scope="sampler",
            tangent_spec=ParameterTangentSpec.with_linear_tie(tie, state_token=TOKEN),
            state_token=TOKEN,
        )


@pytest.mark.light
def test_symmetric_real_tie_matches_independent_projection_for_complex_sources():
    """Symmetric real tie matches independent projection for complex sources."""
    rng = np.random.default_rng(51)
    n = 5
    shape = (n, n + 1)
    tie = _symmetric_real_linear_tie(shape)
    values = rng.standard_normal((7, *shape)) + 1j * rng.standard_normal((7, *shape))

    projected = _project_linear_tie_storage(values.reshape(7, -1), tie, field="real").reshape(values.shape)
    reference = values.copy()
    reference[:, :, :n] = 0.5 * (values[:, :, :n] + values[:, :, :n].transpose(0, 2, 1))

    np.testing.assert_allclose(
        projected,
        reference,
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )
    assert tie.output_size == n * (n + 1)
    assert tie.coordinate_size == n * (n + 1) // 2 + n
    dense = tie.to_dense()
    np.testing.assert_allclose(
        dense.T @ dense,
        np.eye(tie.coordinate_size),
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )


@pytest.mark.light
def test_real_gamma_lambda_tie_is_symmetric_without_imaginary_coordinates():
    """Real gamma lambda tie is symmetric without imaginary coordinates."""
    rng = np.random.default_rng(511)
    n_paired = 4
    n_unpaired = 2
    shape = (n_paired, n_paired + n_unpaired)
    tie = _symmetric_paired_linear_tie(shape, n_paired=n_paired)
    values = rng.standard_normal((5, *shape)) + 1j * rng.standard_normal((5, *shape))

    projected = _project_linear_tie_storage(values.reshape(5, -1), tie, field="real").reshape(values.shape)
    reference = values.copy()
    paired = values[:, :, :n_paired]
    reference[:, :, :n_paired] = 0.5 * (paired + paired.transpose(0, 2, 1))

    np.testing.assert_allclose(
        projected,
        reference,
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )
    assert not np.iscomplexobj(tie.coefficients)
    assert tie.coordinate_size == n_paired * (n_paired + 1) // 2 + n_paired * n_unpaired


@pytest.mark.light
def test_hermitian_tie_matches_owner_projection_and_real_metric():
    """Hermitian tie matches owner projection and real metric."""
    rng = np.random.default_rng(52)
    n_paired = 4
    n_unpaired = 2
    shape = (n_paired, n_paired + n_unpaired)
    tie = _hermitian_paired_linear_tie(shape, n_paired=n_paired)
    values = rng.standard_normal((6, *shape)) + 1j * rng.standard_normal((6, *shape))

    projected = _project_linear_tie_storage(values.reshape(6, -1), tie, field="complex").reshape(values.shape)
    reference = values.copy()
    paired = values[:, :, :n_paired]
    reference[:, :, :n_paired] = 0.5 * (paired + paired.conj().transpose(0, 2, 1))

    np.testing.assert_allclose(
        projected,
        reference,
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )
    assert tie.output_size == n_paired * (n_paired + n_unpaired)
    assert tie.coordinate_size == n_paired**2 + 2 * n_paired * n_unpaired
    dense = tie.to_dense()
    np.testing.assert_allclose(
        np.real(dense.conj().T @ dense),
        np.eye(tie.coordinate_size),
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )


@pytest.mark.light
@pytest.mark.parametrize("field", ["real", "complex"])
def test_tie_projection_coo_matches_dense_oracle(field):
    """COO gather + ``np.add.at`` projection vs the dense form.

    ``_project_linear_tie_storage`` does not densify the tie
    (``(output_size, coordinate_size)`` matrix -- 131 GiB at J3 n_orb = 432);
    ``to_dense()`` with two BLAS matmuls is retained purely as this
    oracle. A general non-orthogonal tie with duplicated coordinate indices
    exercises the segment sums; agreement is to fp round-off (the segment
    sum accumulates in COO-entry order, not the BLAS reduction order).
    """
    rng = np.random.default_rng(53)
    output_size, coordinate_size, nnz = 37, 11, 37
    coeff = rng.standard_normal(nnz)
    if field == "complex":
        coeff = coeff + 1j * rng.standard_normal(nnz)
    # Every coordinate must be used (build() validates coverage); the surplus
    # entries duplicate random coordinates so the segment sums see collisions.
    coordinate_indices = np.concatenate(
        [rng.permutation(coordinate_size), rng.integers(0, coordinate_size, size=nnz - coordinate_size)]
    )
    tie = LinearTieSpec.build(
        output_indices=np.arange(output_size),
        coordinate_indices=coordinate_indices,
        coefficients=coeff,
        output_size=output_size,
        coordinate_size=coordinate_size,
    )
    values = rng.standard_normal((6, output_size))
    if field == "complex":
        values = values + 1j * rng.standard_normal((6, output_size))

    projected = _project_linear_tie_storage(values, tie, field=field)

    dense = tie.to_dense()
    coordinates = np.real(values @ dense.conj()) if np.iscomplexobj(dense) else values @ dense
    reference = coordinates @ dense.T

    np.testing.assert_allclose(
        projected,
        reference,
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )


@pytest.mark.light
def test_tangent_spec_rejects_invalid_payload_and_mutable_input():
    """Tangent spec rejects invalid payload and mutable input."""
    tie = LinearTieSpec.build(
        output_indices=np.array([0, 1]),
        coordinate_indices=np.array([0, 0]),
        coefficients=np.array([2**-0.5, 2**-0.5]),
        output_size=2,
        coordinate_size=1,
    )
    spec = ParameterTangentSpec.with_linear_tie(tie, state_token=TOKEN)
    assert not spec.linear_tie.output_indices.flags.writeable
    assert not spec.linear_tie.coefficients.flags.writeable

    with pytest.raises(ValueError):
        ParameterTangentSpec.build(
            kind="euclidean",
            metric_sqrt=np.eye(2),
            metric_inv_sqrt=None,
            linear_tie=None,
            state_token=TOKEN,
        )
    with pytest.raises(ValueError):
        LinearTieSpec.build(
            output_indices=np.array([0, 2]),
            coordinate_indices=np.array([0, 0]),
            coefficients=np.ones(2),
            output_size=2,
            coordinate_size=1,
        )
    with pytest.raises(ValueError, match="mutual inverses"):
        ParameterTangentSpec.s_stiefel_rows(
            metric_sqrt=np.eye(2),
            metric_inv_sqrt=2.0 * np.eye(2),
            state_token=TOKEN,
        )


@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.light
def test_stiefel_metric_validation_uses_scaled_backward_error(complex_):
    """A well-conditioned algebraic identity must not depend on scalar storage width."""
    size = 66
    rng = np.random.default_rng(812 + int(complex_))
    raw = rng.standard_normal((size, size))
    if complex_:
        raw = raw + 1.0j * rng.standard_normal((size, size))
    eigenvectors, _ = np.linalg.qr(raw)
    eigenvalues = np.geomspace(1.0, 1.0e14, size)
    sqrt = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.conj().T
    inv_sqrt = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.conj().T

    spec = ParameterTangentSpec.s_stiefel_rows(
        metric_sqrt=sqrt,
        metric_inv_sqrt=inv_sqrt,
        state_token=TOKEN,
    )
    assert spec.metric_sqrt.dtype == sqrt.dtype

    broken = inv_sqrt.copy()
    broken[0, 0] += np.asarray(1.0e-3, dtype=broken.dtype)
    with pytest.raises(ValueError, match="mutual inverses"):
        ParameterTangentSpec.s_stiefel_rows(
            metric_sqrt=sqrt,
            metric_inv_sqrt=broken,
            state_token=TOKEN,
        )


@pytest.mark.light
def test_stiefel_backward_error_depends_on_real_precision_not_complex_storage_width():
    """Stiefel backward error depends on real precision not complex storage width."""
    for dimension in (1, 66, 512):
        assert get_s_stiefel_backward_error_limit(np.float32, dimension) == get_s_stiefel_backward_error_limit(
            np.complex64, dimension
        )
        assert get_s_stiefel_backward_error_limit(np.float64, dimension) == get_s_stiefel_backward_error_limit(
            np.complex128, dimension
        )


@pytest.mark.parametrize("scale", [1.0e-20, 1.0, 1.0e20])
@pytest.mark.light
def test_stiefel_metric_builder_rejects_significant_negative_overlap_at_every_scale(scale):
    """Stiefel metric builder rejects significant negative overlap at every scale."""
    overlap = scale * np.diag(np.array([2.0, 1.0, -0.1]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        _build_s_stiefel_metric_factors(overlap)


@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.light
def test_stiefel_metric_builder_accepts_roundoff_negative_overlap(complex_):
    """Stiefel metric builder accepts roundoff negative overlap."""
    dtype = np.complex128 if complex_ else np.float64
    eps = np.finfo(np.float64).eps
    overlap = np.diag(np.asarray([2.0, 1.0, -eps], dtype=dtype))
    sqrt, inv_sqrt = _build_s_stiefel_metric_factors(overlap)
    ParameterTangentSpec.s_stiefel_rows(
        metric_sqrt=sqrt,
        metric_inv_sqrt=inv_sqrt,
        state_token=TOKEN,
    )
