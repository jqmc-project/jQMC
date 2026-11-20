"""C7 complete real-coordinate and factor-tangent map contracts."""

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

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._setting import (
    ATOL_BIT_EXACT,
    ATOL_NONVACUITY_DISTINCTION,
    ATOL_PYTHON_FP64_LOOSE,
    RTOL_NONVACUITY_DISTINCTION,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_PYTHON_FP64_ZERO,
    atol_consistency,
    rtol_consistency,
)
from jqmc.jqmc_optimizer import BlockDerivativeSource, DeviceBlockDerivativeSource, OptimizationCoordinateMap
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import (
    LinearTieSpec,
    ParameterTangentSpec,
    VariationalParameterBlock,
    Wavefunction_data,
    _hermitian_paired_linear_tie,
    _symmetric_real_linear_tie,
)

from ._optimizer_transform_cases import case_blocks, device_sources, source_components
from .optimizer_metadata_helpers import TEST_STATE_TOKEN, make_test_block

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _metric_factors(rng: np.random.Generator, size: int, *, complex_: bool):
    raw = rng.standard_normal((size, size))
    if complex_:
        raw = raw + 1.0j * rng.standard_normal((size, size))
    metric = raw @ raw.conj().T + (size + 1.0) * np.eye(size)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    sqrt = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.conj().T
    inv_sqrt = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.conj().T
    return sqrt, inv_sqrt


def _orthonormal_rows(rng: np.random.Generator, rows: int, columns: int, *, complex_: bool):
    raw = rng.standard_normal((columns, rows))
    if complex_:
        raw = raw + 1.0j * rng.standard_normal((columns, rows))
    q, _ = np.linalg.qr(raw)
    return q.T


def _stiefel_block(name: str, rng: np.random.Generator, *, rows: int, columns: int, complex_: bool):
    sqrt, inv_sqrt = _metric_factors(rng, columns, complex_=complex_)
    whitened = _orthonormal_rows(rng, rows, columns, complex_=complex_)
    coefficients = whitened @ inv_sqrt.T
    if not complex_:
        coefficients = np.real(coefficients)
        sqrt = np.real(sqrt)
        inv_sqrt = np.real(inv_sqrt)
    spec = ParameterTangentSpec.s_stiefel_rows(
        metric_sqrt=sqrt,
        metric_inv_sqrt=inv_sqrt,
        state_token=TEST_STATE_TOKEN,
    )
    block = VariationalParameterBlock.build(
        name=name,
        values=coefficients,
        field="complex" if complex_ else "real",
        derivative_policy="wirtinger_pair" if complex_ else "direct_real",
        sync_scope="sampler",
        tangent_spec=spec,
        state_token=TEST_STATE_TOKEN,
    )
    return block, whitened


def _independent_stiefel_action(delta, coefficients, sqrt, inv_sqrt):
    whitened = coefficients @ sqrt.T
    z = delta @ sqrt.T
    gram = z @ whitened.conj().T
    sym = 0.5 * (gram + gram.conj().T)
    return (z - sym @ whitened) @ inv_sqrt.T


@pytest.mark.parametrize("complex_", [False, True])
def test_s_stiefel_actions_match_independent_dense_embedding_and_real_adjoint(complex_):
    """S stiefel actions match independent dense embedding and real adjoint."""
    rng = np.random.default_rng(701 + int(complex_))
    block, _ = _stiefel_block("mo", rng, rows=2, columns=4, complex_=complex_)
    coordinate_map = OptimizationCoordinateMap.from_blocks([block])

    q_identity = np.eye(coordinate_map.coordinate_size)
    encoded_basis = coordinate_map.encode(q_identity)
    ambient = coordinate_map.field_map.encode(q_identity)
    expected_basis = np.stack(
        [
            _independent_stiefel_action(
                direction.reshape(block.shape),
                np.asarray(block.values),
                block.tangent_spec.metric_sqrt,
                block.tangent_spec.metric_inv_sqrt,
            ).reshape(-1)
            for direction in ambient
        ]
    )
    np.testing.assert_allclose(
        encoded_basis,
        expected_basis,
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )

    covector = rng.standard_normal((3, block.size)) + 1.0j * rng.standard_normal((3, block.size))
    pulled_back = coordinate_map.real_adjoint(covector)
    dense = encoded_basis.T
    np.testing.assert_allclose(
        pulled_back,
        np.real(covector @ dense.conj()),
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )
    probe = rng.standard_normal((3, coordinate_map.coordinate_size))
    lhs = np.real(np.sum(covector.conj() * coordinate_map.encode(probe), axis=-1))
    rhs = np.sum(pulled_back * probe, axis=-1)
    np.testing.assert_allclose(lhs, rhs, rtol=RTOL_PYTHON_FP64_ZERO, atol=ATOL_PYTHON_FP64_LOOSE)


@pytest.mark.parametrize("complex_", [False, True])
def test_stiefel_projection_off_manifold_matches_independent_dense_constraint_projector(complex_):
    """The generalized row-Gram solve must not assume B B^H == I."""
    rng = np.random.default_rng(719 + int(complex_))
    rows, columns = 2, 4
    whitened = rng.standard_normal((rows, columns))
    ambient = rng.standard_normal((rows, columns))
    if complex_:
        whitened = whitened + 1.0j * rng.standard_normal((rows, columns))
        ambient = ambient + 1.0j * rng.standard_normal((rows, columns))

    size = rows * columns

    def unpack(vector):
        real = vector[:size].reshape(rows, columns)
        if not complex_:
            return real
        return real + 1.0j * vector[size:].reshape(rows, columns)

    def pack(matrix):
        if not complex_:
            return np.asarray(matrix).real.ravel()
        return np.concatenate((np.asarray(matrix).real.ravel(), np.asarray(matrix).imag.ravel()))

    def constraint(direction):
        value = direction @ whitened.conj().T + whitened @ direction.conj().T
        return pack(value)

    coordinate_size = size * (2 if complex_ else 1)
    constraint_matrix = np.column_stack(
        [constraint(unpack(np.eye(coordinate_size)[index])) for index in range(coordinate_size)]
    )
    ambient_vector = pack(ambient)
    expected_vector = ambient_vector - constraint_matrix.T @ np.linalg.pinv(constraint_matrix @ constraint_matrix.T) @ (
        constraint_matrix @ ambient_vector
    )
    expected = unpack(expected_vector)

    actual = OptimizationCoordinateMap._project_whitened_stiefel_tangent(whitened, ambient)
    np.testing.assert_allclose(actual, expected, rtol=RTOL_PYTHON_FP64_ZERO, atol=ATOL_PYTHON_FP64_LOOSE)
    np.testing.assert_allclose(
        constraint(actual),
        np.zeros_like(constraint(actual)),
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )


def test_linear_tie_is_composed_without_materializing_an_ambient_xy_unknown():
    """Linear tie is composed without materializing an ambient xy unknown."""
    tie = _hermitian_paired_linear_tie((2, 3), n_paired=2)
    block = VariationalParameterBlock.build(
        name="lambda_matrix",
        values=np.zeros((2, 3), dtype=np.complex128),
        field="complex",
        derivative_policy="wirtinger_pair",
        sync_scope="sampler",
        tangent_spec=ParameterTangentSpec.with_linear_tie(tie, state_token=TEST_STATE_TOKEN),
        state_token=TEST_STATE_TOKEN,
    )
    coordinate_map = OptimizationCoordinateMap.from_blocks([block])
    dense = tie.to_dense()
    q = np.arange(1, tie.coordinate_size + 1, dtype=np.float64)
    covector = np.arange(1, tie.output_size + 1) + 1.0j * np.arange(tie.output_size, 0, -1)

    assert coordinate_map.coordinate_size == tie.coordinate_size
    np.testing.assert_allclose(coordinate_map.encode(q), dense @ q, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)
    np.testing.assert_allclose(
        coordinate_map.real_adjoint(covector),
        np.real(covector @ dense.conj()),
        rtol=RTOL_PYTHON_FP64_LOOSE,
        atol=ATOL_BIT_EXACT,
    )

    rng = np.random.default_rng(708)
    A = rng.standard_normal((4, *block.shape)) + 1.0j * rng.standard_normal((4, *block.shape))
    B = rng.standard_normal((4, *block.shape)) + 1.0j * rng.standard_normal((4, *block.shape))
    source = BlockDerivativeSource.build(block=block, d_p=A, d_pbar=B)
    expected = A.reshape(4, -1) @ dense + B.reshape(4, -1) @ np.conj(dense)
    np.testing.assert_allclose(
        coordinate_map.transform_derivatives([source]),
        expected,
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )


def test_layout_fingerprint_is_structural_and_excludes_state_values_and_token():
    """Layout fingerprint is structural and excludes state values and token."""
    first = make_test_block(
        "r",
        np.asarray([1.0, 2.0]),
        field="real",
        derivative_policy="direct_real",
        sync_scope="world",
        state_token=("lineage-a", 0),
    )
    second = make_test_block(
        "r",
        np.asarray([7.0, 9.0]),
        field="real",
        derivative_policy="direct_real",
        sync_scope="world",
        state_token=("lineage-b", 11),
    )
    map_a = OptimizationCoordinateMap.from_blocks([first])
    map_b = OptimizationCoordinateMap.from_blocks([second])

    assert map_a.layout_fingerprint() == map_b.layout_fingerprint()
    assert map_a.state_token != map_b.state_token

    float32_block = make_test_block(
        "r",
        np.asarray([1.0, 2.0], dtype=np.float32),
        field="real",
        derivative_policy="direct_real",
        sync_scope="world",
        state_token=("lineage-c", 0),
    )
    assert map_a.layout_fingerprint() != OptimizationCoordinateMap.from_blocks([float32_block]).layout_fingerprint()


def test_exact_sources_transform_by_forward_directional_chain_rule_for_mixed_tangents():
    """Exact sources transform by forward directional chain rule for mixed tangents."""
    rng = np.random.default_rng(703)
    real = make_test_block(
        "real",
        np.asarray(0.0),
        field="real",
        derivative_policy="direct_real",
        sync_scope="world",
    )
    complex_ = make_test_block(
        "complex",
        np.zeros(1, dtype=np.complex128),
        field="complex",
        derivative_policy="holomorphic",
        sync_scope="sampler",
    )
    stiefel, _ = _stiefel_block("mo", rng, rows=2, columns=3, complex_=True)
    blocks = [real, complex_, stiefel]
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    sample_shape = (2, 3)
    direct = rng.standard_normal((*sample_shape, *real.shape)) + 1.0j * rng.standard_normal((*sample_shape, *real.shape))
    holomorphic = rng.standard_normal((*sample_shape, *complex_.shape)) + 1.0j * rng.standard_normal(
        (*sample_shape, *complex_.shape)
    )
    A = rng.standard_normal((*sample_shape, *stiefel.shape)) + 1.0j * rng.standard_normal((*sample_shape, *stiefel.shape))
    B = rng.standard_normal((*sample_shape, *stiefel.shape)) + 1.0j * rng.standard_normal((*sample_shape, *stiefel.shape))
    sources = (
        BlockDerivativeSource.build(block=real, direct=direct),
        BlockDerivativeSource.build(block=complex_, d_p=holomorphic),
        BlockDerivativeSource.build(block=stiefel, d_p=A, d_pbar=B),
    )

    transformed = coordinate_map.transform_derivatives(sources)
    basis = coordinate_map.encode(np.eye(coordinate_map.coordinate_size))
    expected_parts = []
    source_parts = ((direct, None), (holomorphic, None), (A, B))
    for index, ((primary, secondary), storage_slice, coordinate_slice) in enumerate(
        zip(
            source_parts,
            coordinate_map.block_storage_slices,
            coordinate_map.block_coordinate_slices,
            strict=True,
        )
    ):
        block_basis = basis[coordinate_slice, storage_slice].T
        flat_primary = primary.reshape(*sample_shape, -1)
        expected = flat_primary @ block_basis
        if index == 2:
            assert secondary is not None
            expected = expected + secondary.reshape(*sample_shape, -1) @ np.conj(block_basis)
        expected_parts.append(expected)
    expected = np.concatenate(expected_parts, axis=-1)
    np.testing.assert_allclose(
        transformed,
        expected,
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )
    device_sources = (
        DeviceBlockDerivativeSource.build(block=real, direct=direct),
        DeviceBlockDerivativeSource.build(block=complex_, d_p=holomorphic),
        DeviceBlockDerivativeSource.build(block=stiefel, d_p=A, d_pbar=B),
    )
    np.testing.assert_allclose(
        np.asarray(coordinate_map.transform_derivatives_device(device_sources)),
        transformed,
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )

    # A transpose/adjoint shortcut gives the wrong sign for the holomorphic
    # imaginary direction and is therefore an explicit counterexample.
    complex_slice = coordinate_map.block_coordinate_slices[1]
    np.testing.assert_allclose(
        transformed[..., complex_slice.start + 1], 1.0j * holomorphic[..., 0], rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT
    )
    assert not np.allclose(
        transformed[..., complex_slice.start + 1],
        -1.0j * holomorphic[..., 0],
        rtol=RTOL_NONVACUITY_DISTINCTION,
        atol=ATOL_NONVACUITY_DISTINCTION,
    )


@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.parametrize("time_reversal", [False, True])
def test_factor_tangent_obeys_independent_dense_rank_projector_oracle(time_reversal, complex_):
    """Factor tangents satisfy the rank constraint and the nonzero dense action."""
    rng = np.random.default_rng(704 + int(time_reversal) + 2 * int(complex_))
    up, B_up = _stiefel_block("mo_up", rng, rows=2, columns=4, complex_=complex_)
    if time_reversal:
        down = None
        B_down = np.conj(B_up)
        blocks = [up]
    else:
        down, B_down = _stiefel_block("mo_down", rng, rows=2, columns=4, complex_=complex_)
        blocks = [up, down]
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    q = rng.standard_normal(coordinate_map.coordinate_size)
    delta = coordinate_map.encode(q)
    delta_up = delta[coordinate_map.block_storage_slices[0]].reshape(up.shape)
    delta_B_up = delta_up @ up.tangent_spec.metric_sqrt.T
    if time_reversal:
        delta_B_down = np.conj(delta_B_up)
    else:
        assert down is not None
        delta_down = delta[coordinate_map.block_storage_slices[1]].reshape(down.shape)
        delta_B_down = delta_down @ down.tangent_spec.metric_sqrt.T

    sigma = np.diag(np.asarray([1.7, -0.8]))
    delta_geminal = delta_B_up.conj().T @ sigma @ B_down + B_up.conj().T @ sigma @ delta_B_down
    left_complement = np.eye(4) - B_up.conj().T @ B_up
    right_complement = np.eye(4) - B_down.conj().T @ B_down
    virtual_virtual = left_complement @ delta_geminal @ right_complement
    dense_projected = delta_geminal - virtual_virtual

    np.testing.assert_allclose(
        virtual_virtual,
        np.zeros_like(virtual_virtual),
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )
    np.testing.assert_allclose(
        delta_geminal,
        dense_projected,
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )

    # The homogeneous rank constraint cannot detect a common rescaling of
    # both factor directions. Anchor their amplitude using the same input q.
    ambient = coordinate_map.field_map.encode(q)
    expected_factors = []
    for block, storage_slice in zip(blocks, coordinate_map.block_storage_slices, strict=True):
        expected_delta = _independent_stiefel_action(
            ambient[storage_slice].reshape(block.shape),
            np.asarray(block.values),
            block.tangent_spec.metric_sqrt,
            block.tangent_spec.metric_inv_sqrt,
        )
        expected_factors.append(expected_delta @ block.tangent_spec.metric_sqrt.T)
    expected_up = expected_factors[0]
    expected_down = np.conj(expected_up) if time_reversal else expected_factors[1]
    expected_geminal = expected_up.conj().T @ sigma @ B_down + B_up.conj().T @ sigma @ expected_down
    np.testing.assert_allclose(delta_geminal, expected_geminal, rtol=RTOL_PYTHON_FP64_ZERO, atol=ATOL_PYTHON_FP64_LOOSE)


@pytest.mark.parametrize("trexio_file", ["H2_pbc_ae_cart.h5", "H2_pbc_ae_k025_cart.h5"])
def test_actual_gamma_and_twist_sd_publishers_build_matched_factor_actions(trexio_file):
    """Gamma/twist SD actions satisfy tangent, adjoint and dense amplitude references."""
    fixture = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file)
    geminal = read_trexio_file(fixture, store_tuple=True)[4]
    wavefunction = Wavefunction_data(jastrow_data=None, geminal_data=geminal)
    blocks = wavefunction.get_variational_blocks(
        state_token=TEST_STATE_TOKEN,
        opt_J1_param=False,
        opt_J2_param=False,
        opt_J3_param=False,
        opt_geminal=True,
    )
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    rng = np.random.default_rng(706)
    q = rng.standard_normal((2, coordinate_map.coordinate_size))
    covector = rng.standard_normal((2, coordinate_map.storage_size)) + 1.0j * rng.standard_normal(
        (2, coordinate_map.storage_size)
    )

    encoded = coordinate_map.encode(q)
    pulled_back = coordinate_map.real_adjoint(covector)
    lhs = np.real(np.sum(covector.conj() * encoded, axis=-1))
    rhs = np.sum(pulled_back * q, axis=-1)
    np.testing.assert_allclose(lhs, rhs, rtol=RTOL_PYTHON_FP64_ZERO, atol=ATOL_PYTHON_FP64_LOOSE)
    assert [block.name for block in blocks] == ["mo_coefficients_up"]
    assert blocks[0].tangent_spec.kind == "s_stiefel_rows"
    block = blocks[0]
    sqrt = block.tangent_spec.metric_sqrt
    assert sqrt is not None
    whitened = np.asarray(block.values) @ sqrt.T
    np.testing.assert_allclose(
        whitened @ whitened.conj().T,
        np.eye(whitened.shape[0]),
        rtol=rtol_consistency,
        atol=atol_consistency,
    )
    delta = encoded[0, coordinate_map.block_storage_slices[0]].reshape(block.shape)
    delta_whitened = delta @ sqrt.T
    tangent_constraint = delta_whitened @ whitened.conj().T + whitened @ delta_whitened.conj().T
    np.testing.assert_allclose(
        tangent_constraint,
        np.zeros_like(tangent_constraint),
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_LOOSE,
    )

    ambient = coordinate_map.field_map.encode(q)
    expected = np.stack(
        [
            _independent_stiefel_action(
                direction.reshape(block.shape), np.asarray(block.values), sqrt, block.tangent_spec.metric_inv_sqrt
            ).reshape(-1)
            for direction in ambient
        ]
    )
    np.testing.assert_allclose(encoded, expected, rtol=RTOL_PYTHON_FP64_ZERO, atol=ATOL_PYTHON_FP64_LOOSE)


def test_actual_open_shell_agpn_publishes_independent_spin_and_unpaired_coordinates():
    """Open-shell AGPn preserves Euclidean coordinates and each spin's dense tangent action."""
    from .optimizer_metadata_helpers import build_explicit_common_rank_agpn_from_sd

    fixture = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "N_ae_ccpvdz_cart.h5")
    sd = read_trexio_file(fixture, store_tuple=True)[4]
    agpn = build_explicit_common_rank_agpn_from_sd(sd)
    wavefunction = Wavefunction_data(jastrow_data=None, geminal_data=agpn)
    blocks = wavefunction.get_variational_blocks(
        state_token=TEST_STATE_TOKEN,
        opt_J1_param=False,
        opt_J2_param=False,
        opt_J3_param=False,
        opt_geminal=True,
    )
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    names = [block.name for block in blocks]

    assert names == ["mo_coefficients_up", "mo_coefficients_dn", "lambda_diag", "lambda_unpaired"]
    assert [block.tangent_spec.kind for block in blocks] == [
        "s_stiefel_rows",
        "s_stiefel_rows",
        "euclidean",
        "euclidean",
    ]
    assert blocks[2].field == "real"
    assert blocks[3].shape == np.asarray(agpn.lambda_unpaired).shape

    rng = np.random.default_rng(707)
    q = rng.standard_normal(coordinate_map.coordinate_size)
    delta = coordinate_map.encode(q)
    for comparison_index, (block, storage_slice) in enumerate(
        zip(blocks[:2], coordinate_map.block_storage_slices[:2], strict=True)
    ):
        comparison_iteration = str(comparison_index)  # noqa: F841 - captured by the GDD iteration receipt
        delta_c = delta[storage_slice].reshape(block.shape)
        sqrt = block.tangent_spec.metric_sqrt
        assert sqrt is not None
        B = np.asarray(block.values) @ sqrt.T
        delta_B = delta_c @ sqrt.T
        first_order_constraint = delta_B @ B.conj().T + B @ delta_B.conj().T
        np.testing.assert_allclose(
            first_order_constraint,
            np.zeros_like(first_order_constraint),
            rtol=RTOL_PYTHON_FP64_ZERO,
            atol=ATOL_PYTHON_FP64_LOOSE,
        )

    # The open-shell occupation block is intentionally Euclidean: C7 must not
    # apply the paired virtual-virtual projection to it.
    unpaired_index = names.index("lambda_unpaired")
    unpaired_q = coordinate_map.block_coordinate_slices[unpaired_index]
    unpaired_storage = coordinate_map.block_storage_slices[unpaired_index]
    np.testing.assert_allclose(delta[unpaired_storage], q[unpaired_q], rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_BIT_EXACT)

    ambient = coordinate_map.field_map.encode(q)
    for comparison_index, (block, storage_slice) in enumerate(
        zip(blocks[:2], coordinate_map.block_storage_slices[:2], strict=True)
    ):
        comparison_iteration = str(comparison_index)
        expected = _independent_stiefel_action(
            ambient[storage_slice].reshape(block.shape),
            np.asarray(block.values),
            block.tangent_spec.metric_sqrt,
            block.tangent_spec.metric_inv_sqrt,
        )
        np.testing.assert_allclose(
            delta[storage_slice].reshape(block.shape),
            expected,
            rtol=RTOL_PYTHON_FP64_ZERO,
            atol=ATOL_PYTHON_FP64_LOOSE,
        )


def _tie_test_map(tie, shape, *, complex_):
    block = VariationalParameterBlock.build(
        name="lambda_matrix" if complex_ else "j3_matrix",
        values=np.zeros(shape, dtype=np.complex128 if complex_ else np.float64),
        field="complex" if complex_ else "real",
        derivative_policy="wirtinger_pair" if complex_ else "direct_real",
        sync_scope="world",
        tangent_spec=ParameterTangentSpec.with_linear_tie(tie, state_token=TEST_STATE_TOKEN),
        state_token=TEST_STATE_TOKEN,
    )
    return block, OptimizationCoordinateMap.from_blocks([block])


@pytest.mark.parametrize("kind", ["symmetric", "hermitian", "general-f32", "general-f64"])
def test_device_tie_adjoint_and_directional_sources_match_dense_reference(kind):
    """Sparse real/complex ties accumulate all repeated coordinates for scalar, batched and empty samples.

    J3 symmetry and Hermitian paired/free AGP columns use their actual publishers.
    General COO has four contributions to one coordinate and multiple coordinates
    from one output, with dyadic coefficients to check fp32 dtype without a new
    tolerance. The independent dense matrix gives Re(v E*) and A E + B E*;
    neither reference invokes a production adjoint or coordinate transform.
    """
    complex_ = kind != "symmetric"
    dtype = np.complex64 if kind == "general-f32" else np.complex128
    if kind == "symmetric":
        shape = (4, 5)
        tie = _symmetric_real_linear_tie(shape)
    elif kind == "hermitian":
        shape = (3, 4)
        tie = _hermitian_paired_linear_tie(shape, n_paired=3)
    else:
        shape = (2, 3)
        tie = LinearTieSpec.build(
            output_indices=[0, 1, 2, 3, 4, 5, 0, 3],
            coordinate_indices=[2, 0, 2, 1, 2, 0, 0, 2],
            coefficients=np.asarray([1 + 2j, -2 + 1j, 0.5 - 1j, 2j, -1, 0.5, 1j, -0.5j], dtype=dtype),
            output_size=6,
            coordinate_size=3,
        )
    block, coordinate_map = _tie_test_map(tie, shape, complex_=complex_)
    dense = tie.to_dense()
    for sample_shape in ((), (2, 3), (0,)):
        count = int(np.prod(sample_shape)) * tie.output_size
        entries = np.arange(count, dtype=np.float64).reshape(*sample_shape, tie.output_size)
        complex_values = np.asarray((entries % 7 - 3) + 1j * (entries % 5 - 2), dtype=dtype)
        for values in (complex_values.real, complex_values):
            actual = np.asarray(coordinate_map._real_adjoint_block_device(0, jnp.asarray(values)))
            expected = np.real(values @ dense.conj())
            assert actual.shape == (*sample_shape, tie.coordinate_size)
            assert actual.dtype == np.result_type(values.real.dtype, tie.coefficients.real.dtype)
            np.testing.assert_allclose(actual, expected, rtol=RTOL_PYTHON_FP64_ZERO, atol=ATOL_PYTHON_FP64_LOOSE)
            if not sample_shape:
                continue  # Public derivative sources require at least one sample axis.
            primary = values.reshape(*sample_shape, *shape)
            secondary = (values.conj() * 0.5).reshape(*sample_shape, *shape)
            source = (
                DeviceBlockDerivativeSource.build(block=block, d_p=primary, d_pbar=secondary)
                if complex_
                else DeviceBlockDerivativeSource.build(block=block, direct=primary)
            )
            transformed = np.asarray(coordinate_map.transform_derivatives_device([source]))
            directional = values @ dense
            if complex_:
                directional = directional + secondary.reshape(*sample_shape, tie.output_size) @ dense.conj()
            np.testing.assert_allclose(transformed, directional, rtol=RTOL_PYTHON_FP64_ZERO, atol=ATOL_PYTHON_FP64_LOOSE)


def _jaxpr_primitive_names(value):
    """Include nested JAXprs so a jit/scan wrapper cannot hide per-entry work."""
    if hasattr(value, "jaxpr"):
        yield from _jaxpr_primitive_names(value.jaxpr)
    elif hasattr(value, "eqns"):
        for equation in value.eqns:
            yield equation.primitive.name
            yield from _jaxpr_primitive_names(equation.params)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _jaxpr_primitive_names(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _jaxpr_primitive_names(item)


def test_device_tie_adjoint_dispatch_structure_is_independent_of_nonzeros():
    """J3 ties with 20 and 930 COO entries use one scatter-add and bounded identical trace size.

    This is a machine-independent dispatch/copy detector, not a wall-time limit.
    Tracing the original Python loop exposes one whole-result scatter per entry;
    wrapping it in jit or scan does not satisfy the sparse vectorized contract.
    """
    counts = []
    for n_orb in (4, 30):
        tie = _symmetric_real_linear_tie((n_orb, n_orb + 1))
        _, coordinate_map = _tie_test_map(tie, (n_orb, n_orb + 1), complex_=False)
        covectors = jnp.ones((2, tie.output_size), dtype=jnp.complex128)
        traced = jax.make_jaxpr(lambda values: coordinate_map._real_adjoint_block_device(0, values))(covectors)
        names = list(_jaxpr_primitive_names(traced))
        assert names.count("scatter-add") == 1, "linear-tie adjoint must batch all COO contributions into one scatter-add"
        assert not {"scan", "while", "dot_general"}.intersection(names)
        counts.append(len(names))
    assert counts[0] == counts[1] <= 40, "linear-tie dispatch must stay bounded as nnz grows"


@pytest.mark.parametrize("contraction", ["columns", "right-basis"])
def test_real_wavefunction_device_transform_is_real_and_equals_complex_route_real_part(contraction):
    """A real wave function's device transform is real and equals Re of the complex route.

    J2 (Euclidean), J3 (symmetric linear tie) and SD MO (S-Stiefel rows) blocks with
    real sources take the device real route, both for the full columns and for the
    blockwise ``right=`` contraction used by reduced LM. The same sources cast to
    complex128 with a zero imaginary part force the complex route, which forms the
    imaginary covector branch and runs the S-Stiefel projection in complex
    arithmetic; the two agree within the coordinate-map pair. The route of a
    rank-local call is the publication's own (``all_real=None``) and equals the
    explicit entry decision ``all_real=True``.
    """
    blocks = case_blocks("j2-j3-sd-real-source")
    assert {block.tangent_spec.kind for block in blocks} == {"euclidean", "linear_tie", "s_stiefel_rows"}
    cmap = OptimizationCoordinateMap.from_blocks(blocks)
    components = source_components(blocks, (37,), complex_source=False, seed=8860)
    complex_components = [{name: values.astype(np.complex128) for name, values in parts.items()} for parts in components]
    right = None
    if contraction == "right-basis":
        right = np.random.default_rng(8861).standard_normal((cmap.coordinate_size, 3))
    real_route = cmap.transform_derivatives_device(device_sources(blocks, components), right=right)
    explicit_real_route = cmap.transform_derivatives_device(device_sources(blocks, components), right=right, all_real=True)
    complex_route = cmap.transform_derivatives_device(device_sources(blocks, complex_components), right=right)
    assert real_route.dtype == jnp.float64
    assert complex_route.dtype == jnp.complex128
    np.testing.assert_array_equal(np.asarray(explicit_real_route), np.asarray(real_route))
    np.testing.assert_allclose(
        np.asarray(real_route),
        np.real(np.asarray(complex_route)),
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
