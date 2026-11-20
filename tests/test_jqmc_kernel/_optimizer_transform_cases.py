"""Shared inputs for the host optimizer transform / jackknife Rust==XLA parity tests (#841).

Each case is a coordinate map plus seeded derivative sources, published both as host
``BlockDerivativeSource`` objects (consumed by the host functions, which dispatch to
``jqmc_rs`` on the CPU Rust route) and as identical ``DeviceBlockDerivativeSource``
objects (consumed by the JAX twins). Blocks come from the actual wave-function
publishers where one exists (J2 Euclidean, J3 symmetric linear tie with 1/sqrt(2)
couplings, SD and open-shell AGPn S-Stiefel MO rows) and from explicit specs for the
complex-only shapes (tied Hermitian lambda, complex Stiefel, holomorphic and Wirtinger
Euclidean blocks, a general COO tie with repeated coordinates).
"""

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

import numpy as np

from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data, Jastrow_two_body_data
from jqmc.jqmc_optimizer import (
    BlockDerivativeSource,
    DeviceBlockDerivativeSource,
    DeviceJackknifeBinMetadata,
    DeviceJackknifeBlockMoments,
    DeviceOptimizationBatch,
    JackknifeBinMetadata,
    JackknifeBlockMoments,
    OptimizationBatch,
    OptimizationCoordinateMap,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import (
    LinearTieSpec,
    ParameterTangentSpec,
    VariationalParameterBlock,
    Wavefunction_data,
    _hermitian_paired_linear_tie,
    _symmetric_real_linear_tie,
)

from .optimizer_metadata_helpers import TEST_STATE_TOKEN, build_explicit_common_rank_agpn_from_sd, make_test_block

_FIXTURES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")

#: Case ids of the transform/jackknife parity matrix.
TRANSFORM_CASES = (
    "j2-j3-sd-real-source",
    "j2-j3-sd-complex-source",
    "j3-frozen-j2-sd-active",
    "agpn-open-shell",
    "complex-tied-lambda-stiefel",
)


def _actual_blocks(trexio_file: str, *, opt_J3: bool, agpn: bool) -> list[VariationalParameterBlock]:
    _structure, aos, _mos_up, _mos_dn, geminal, _coulomb = read_trexio_file(
        trexio_file=os.path.join(_FIXTURES, trexio_file), store_tuple=True
    )
    if agpn:
        geminal = build_explicit_common_rank_agpn_from_sd(geminal)
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="pade")
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos, random_init=True, random_scale=1.0e-2, seed=17)
    jastrow = Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=j2, jastrow_three_body_data=j3)
    wavefunction = Wavefunction_data(geminal_data=geminal, jastrow_data=jastrow)
    return wavefunction.get_variational_blocks(
        state_token=TEST_STATE_TOKEN,
        opt_J1_param=False,
        opt_J2_param=True,
        opt_J3_param=opt_J3,
        opt_geminal=True,
    )


def _complex_stiefel_block(rng: np.random.Generator, *, rows: int, columns: int) -> VariationalParameterBlock:
    raw = rng.standard_normal((columns, columns)) + 1.0j * rng.standard_normal((columns, columns))
    metric = raw @ raw.conj().T + (columns + 1.0) * np.eye(columns)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    sqrt = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.conj().T
    inv_sqrt = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.conj().T
    seed = rng.standard_normal((columns, rows)) + 1.0j * rng.standard_normal((columns, rows))
    q, _ = np.linalg.qr(seed)
    coefficients = q.T @ inv_sqrt.T
    return VariationalParameterBlock.build(
        name="mo_coefficients_complex",
        values=coefficients,
        field="complex",
        derivative_policy="wirtinger_pair",
        sync_scope="sampler",
        tangent_spec=ParameterTangentSpec.s_stiefel_rows(
            metric_sqrt=sqrt, metric_inv_sqrt=inv_sqrt, state_token=TEST_STATE_TOKEN
        ),
        state_token=TEST_STATE_TOKEN,
    )


def _tie_block(name: str, tie: LinearTieSpec, shape: tuple[int, ...], *, complex_: bool) -> VariationalParameterBlock:
    return VariationalParameterBlock.build(
        name=name,
        values=np.zeros(shape, dtype=np.complex128 if complex_ else np.float64),
        field="complex" if complex_ else "real",
        derivative_policy="wirtinger_pair" if complex_ else "direct_real",
        sync_scope="world",
        tangent_spec=ParameterTangentSpec.with_linear_tie(tie, state_token=TEST_STATE_TOKEN),
        state_token=TEST_STATE_TOKEN,
    )


def _complex_blocks() -> list[VariationalParameterBlock]:
    rng = np.random.default_rng(841)
    general_tie = LinearTieSpec.build(
        output_indices=[0, 1, 2, 3, 4, 5, 0, 3],
        coordinate_indices=[2, 0, 2, 1, 2, 0, 0, 2],
        coefficients=np.asarray([1.0, -2.0, 0.5, 1.0 / np.sqrt(2.0), -1.0, 0.5, 3.0, -0.25]),
        output_size=6,
        coordinate_size=3,
    )
    return [
        make_test_block("real_scalar", np.asarray(0.3), field="real", derivative_policy="direct_real", sync_scope="world"),
        make_test_block(
            "holomorphic",
            np.zeros(3, dtype=np.complex128),
            field="complex",
            derivative_policy="holomorphic",
            sync_scope="sampler",
        ),
        make_test_block(
            "wirtinger",
            np.zeros((2, 2), dtype=np.complex128),
            field="complex",
            derivative_policy="wirtinger_pair",
            sync_scope="sampler",
        ),
        _tie_block("j3_general_tie", general_tie, (2, 3), complex_=False),
        _tie_block("j3_matrix", _symmetric_real_linear_tie((4, 5)), (4, 5), complex_=False),
        _tie_block("lambda_matrix", _hermitian_paired_linear_tie((3, 5), n_paired=3), (3, 5), complex_=True),
        _complex_stiefel_block(rng, rows=3, columns=5),
    ]


def case_blocks(case: str) -> list[VariationalParameterBlock]:
    """Variational blocks of one parity case."""
    if case in ("j2-j3-sd-real-source", "j2-j3-sd-complex-source"):
        return _actual_blocks("N2_ecp_ccpvtz_cart.h5", opt_J3=True, agpn=False)
    if case == "j3-frozen-j2-sd-active":
        return _actual_blocks("N2_ecp_ccpvtz_cart.h5", opt_J3=False, agpn=False)
    if case == "agpn-open-shell":
        return _actual_blocks("N_ae_ccpvdz_cart.h5", opt_J3=True, agpn=True)
    if case == "complex-tied-lambda-stiefel":
        return _complex_blocks()
    raise ValueError(f"unknown transform parity case {case!r}")


def source_components(
    blocks: list[VariationalParameterBlock],
    sample_shape: tuple[int, ...],
    *,
    complex_source: bool,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    """Seeded exact-source components for every block's derivative policy."""
    rng = np.random.default_rng(seed)
    components = []
    for block in blocks:
        shape = (*sample_shape, *block.shape)

        def draw(complex_: bool, shape=shape) -> np.ndarray:
            values = rng.standard_normal(shape)
            return values + 1.0j * rng.standard_normal(shape) if complex_ else values

        if block.derivative_policy == "direct_real":
            components.append({"direct": draw(complex_source)})
        elif block.derivative_policy == "holomorphic":
            components.append({"d_p": draw(True)})
        else:
            components.append({"d_p": draw(True), "d_pbar": draw(True)})
    return components


def host_sources(blocks, components) -> tuple[BlockDerivativeSource, ...]:
    return tuple(BlockDerivativeSource.build(block=block, **parts) for block, parts in zip(blocks, components, strict=True))


def device_sources(blocks, components) -> tuple[DeviceBlockDerivativeSource, ...]:
    return tuple(
        DeviceBlockDerivativeSource.build(block=block, **parts) for block, parts in zip(blocks, components, strict=True)
    )


def jackknife_batches(
    blocks: list[VariationalParameterBlock],
    *,
    bins: int,
    complex_source: bool,
    complex_energy: bool,
    seed: int,
    batch_token: str = "rust-jackknife-841",
) -> tuple[OptimizationBatch, DeviceOptimizationBatch]:
    """One-sample bins published as identical host and device optimization batches."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.5, 1.5, size=bins)
    energies = rng.standard_normal(bins)
    if complex_energy:
        energies = energies + 1.0j * rng.standard_normal(bins)
    components = source_components(blocks, (bins,), complex_source=complex_source, seed=seed + 1)

    def scaled(parts, factor, conjugate=False):
        result = {}
        for name, values in parts.items():
            expanded = factor.reshape(-1, *([1] * (values.ndim - 1)))
            result[name] = expanded * (np.conj(values) if conjugate else values)
        return result

    sums = [scaled(parts, weights) for parts in components]
    conjugate_energy_sums = [scaled(parts, weights * energies, conjugate=True) for parts in components]
    moments = tuple(
        JackknifeBlockMoments.build(
            block=block,
            source_sum=BlockDerivativeSource.build(block=block, **s),
            conjugate_source_energy_sum=BlockDerivativeSource.build(block=block, **c),
        )
        for block, s, c in zip(blocks, sums, conjugate_energy_sums, strict=True)
    )
    metadata = JackknifeBinMetadata.build(
        blocks=blocks,
        batch_token=batch_token,
        bin_sample_counts=np.ones(bins, dtype=np.int64),
        bin_weight_sums=weights,
        bin_weighted_energy_sums=weights * energies,
        log_source_moments=moments,
    )
    sources = host_sources(blocks, components)
    batch = OptimizationBatch.build(
        blocks=blocks,
        batch_token=batch_token,
        state_token=TEST_STATE_TOKEN,
        weights=weights,
        local_energies=energies,
        log_derivative_sources=sources,
        local_energy_derivative_sources=None,
        jackknife_bin_metadata=metadata,
    )
    device_moments = tuple(
        DeviceJackknifeBlockMoments(
            name=block.name,
            source_sum=DeviceBlockDerivativeSource.build(block=block, **s),
            conjugate_source_energy_sum=DeviceBlockDerivativeSource.build(block=block, **c),
        )
        for block, s, c in zip(blocks, sums, conjugate_energy_sums, strict=True)
    )
    device_batch = DeviceOptimizationBatch(
        batch_token=batch_token,
        state_token=TEST_STATE_TOKEN,
        weights=weights,
        local_energies=energies,
        log_derivative_sources=device_sources(blocks, components),
        local_energy_derivative_sources=None,
        jackknife_bin_metadata=DeviceJackknifeBinMetadata(
            batch_token=batch_token,
            bin_sample_counts=metadata.bin_sample_counts,
            bin_weight_sums=metadata.bin_weight_sums,
            bin_weighted_energy_sums=metadata.bin_weighted_energy_sums,
            log_source_moments=device_moments,
        ),
    )
    return batch, device_batch


def coordinate_map(blocks) -> OptimizationCoordinateMap:
    return OptimizationCoordinateMap.from_blocks(blocks)
