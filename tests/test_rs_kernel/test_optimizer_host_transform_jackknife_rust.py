"""Rust == XLA parity of the host optimizer coordinate transform and force jackknife (#841).

On the CPU Rust route (``JQMC_FLAG_RUST_ON_CPU=1``, JAX cpu backend) the host
functions ``OptimizationCoordinateMap.transform_derivatives`` and
``estimate_coordinate_force_jackknife`` run in ``jqmc_rs``. Each is compared with its
JAX twin (``transform_derivatives_device`` / ``estimate_coordinate_force_jackknife_device``)
on identical inputs: real and complex sources, J3 symmetric and tied Hermitian lambda
linear ties (1/sqrt(2) and +-i/sqrt(2) couplings), a general COO tie with repeated
coordinates, SD / open-shell AGPn / complex S-Stiefel MO blocks, holomorphic and
Wirtinger (d_p, d_pbar) Euclidean blocks, a frozen J3 family, several sample rows across
Rust row chunks, and a multi-axis sample shape. The optimizer arithmetic is float64 in
both precision modes, so the strict pair of float64 zones applies in both. A real wave
function's transform takes the real route (float64 result) on the Rust, the NumPy
and the device route, and equals the real part of the complex route.
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

import numpy as np
import pytest

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import ATOL_BIT_EXACT, ATOL_PYTHON_FP64_LOOSE, RTOL_BIT_EXACT, RTOL_PYTHON_FP64_ZERO
from jqmc.jqmc_optimizer import estimate_coordinate_force_jackknife, estimate_coordinate_force_jackknife_device

from tests.test_jqmc_kernel._optimizer_transform_cases import (
    TRANSFORM_CASES,
    case_blocks,
    coordinate_map,
    device_sources,
    host_sources,
    jackknife_batches,
    source_components,
)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

# Derivative sources (wf_eval) and local-energy moments (local_energy): float64 zones
# in full and mixed alike.
_ZONES = ("wf_eval", "local_energy")


@pytest.fixture
def rust_route(monkeypatch):
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")


# 37 rows span two Rust row chunks (32 + 5); 2x3 is a multi-axis sample shape.
_SAMPLE_SHAPES = {"37": (37,), "2x3": (2, 3)}

# Real wave functions (real fields, real tangent data, real sources): the host
# transform takes its real route and returns float64.
_REAL_WAVEFUNCTION_CASES = ("j2-j3-sd-real-source", "j3-frozen-j2-sd-active", "agpn-open-shell")


@pytest.mark.parametrize("rows", tuple(_SAMPLE_SHAPES))
@pytest.mark.parametrize("case", TRANSFORM_CASES)
def test_rust_host_transform_derivatives_matches_xla_twin(rust_route, case, rows):
    """Rust transform_derivatives equals transform_derivatives_device on identical sources."""
    sample_shape = _SAMPLE_SHAPES[rows]
    blocks = case_blocks(case)
    cmap = coordinate_map(blocks)
    components = source_components(
        blocks,
        sample_shape,
        complex_source=case == "j2-j3-sd-complex-source",
        seed=8410 + TRANSFORM_CASES.index(case),
    )
    actual = cmap.transform_derivatives(host_sources(blocks, components))
    reference = np.asarray(cmap.transform_derivatives_device(device_sources(blocks, components)))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
    assert actual.dtype == reference.dtype == (np.float64 if case in _REAL_WAVEFUNCTION_CASES else np.complex128)


@pytest.mark.parametrize("route", ["numpy", "rust"])
def test_real_wavefunction_host_transform_is_float64_and_equals_complex_route_real_part(monkeypatch, route):
    """A real wave function's host transform is float64 and equals Re of the complex route.

    J2 (Euclidean), J3 (symmetric linear tie) and SD MO (S-Stiefel rows) blocks with
    real sources take the real route. The same sources cast to complex128 with a zero
    imaginary part force the complex route. Rust runs the same real-linear adjoint on
    both routes, so they agree exactly; the NumPy complex route runs the S-Stiefel
    products in complex BLAS, so it agrees within the host coordinate-map pair.
    """
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1" if route == "rust" else "0")
    blocks = case_blocks("j2-j3-sd-real-source")
    assert {block.tangent_spec.kind for block in blocks} == {"euclidean", "linear_tie", "s_stiefel_rows"}
    cmap = coordinate_map(blocks)
    components = source_components(blocks, (37,), complex_source=False, seed=8390)
    real_route = cmap.transform_derivatives(host_sources(blocks, components))
    complex_components = [{name: values.astype(np.complex128) for name, values in parts.items()} for parts in components]
    complex_route = cmap.transform_derivatives(host_sources(blocks, complex_components))
    assert real_route.dtype == np.float64
    assert complex_route.dtype == np.complex128
    if route == "rust":
        np.testing.assert_allclose(real_route, np.real(complex_route), atol=ATOL_BIT_EXACT, rtol=RTOL_BIT_EXACT)
    else:
        np.testing.assert_allclose(real_route, np.real(complex_route), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)


@pytest.mark.parametrize("complex_energy", [False, True])
@pytest.mark.parametrize("case", TRANSFORM_CASES)
def test_rust_host_coordinate_force_jackknife_matches_xla_twin(rust_route, case, complex_energy):
    """Rust estimate_coordinate_force_jackknife equals the device twin (force and standard error)."""
    blocks = case_blocks(case)
    cmap = coordinate_map(blocks)
    batch, device_batch = jackknife_batches(
        blocks,
        bins=41,
        complex_source=case == "j2-j3-sd-complex-source",
        complex_energy=complex_energy,
        seed=8420 + TRANSFORM_CASES.index(case),
    )
    force, std = estimate_coordinate_force_jackknife(batch, cmap)
    reference_force, reference_std = estimate_coordinate_force_jackknife_device(device_batch, cmap)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    np.testing.assert_allclose(force, reference_force, atol=atol, rtol=rtol)
    np.testing.assert_allclose(std, reference_std, atol=atol, rtol=rtol)
