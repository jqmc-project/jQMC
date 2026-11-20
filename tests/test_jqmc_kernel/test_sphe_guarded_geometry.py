"""Spherical AO radial tangents and finite derivatives at nuclei."""

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

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block
from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL
from jqmc.atomic_orbital import (
    _apply_image_expansion,
    _sphe_primitive_prefactors,
    _compute_AOs_debug,
    _compute_AOs_grad_autodiff,
    _compute_AOs_laplacian_autodiff,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
)
from jqmc.derivatives.ao import sphe_basis_param_jvp
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file
from tests.conftest import CELL_FIXTURES, make_pbc_structure
from tests.test_jqmc_kernel.test_atomic_orbital import _make_aos_data

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _assert_orbital_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Require one homogeneous AO derivative block to meet its central budget."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


@pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_sphe.h5", "water_ccecp_ccpvqz.h5"])
@pytest.mark.parametrize("parameter_kind", ["exponents", "coefficients"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_standalone_sphe_value_radial_jvp_matches_analytic(trexio_file, parameter_kind):
    """Standalone spherical AO radial derivatives equal the analytic basis JVP.

    Molecular H2 all-electron cc-pVDZ and water ccECP cc-pVQZ TREXIO bases
    exercise tied radial parameters, multiple angular momenta and contraction
    depths. Seed 2025 supplies five AO evaluation points and nonzero directions
    for either exponents or coefficients, with the other block fixed. These
    are component points, not a many-body spin configuration; cond(G) does not
    apply. Full precision differentiates the actual standalone value route, not the
    fused VGL route, at the existing strict tolerance. Mixed compares the analytical
    JVP with its independently rebuilt full route at the orbital derivative fidelity budget.
    """

    def case_factory():
        fixture = Path(__file__).resolve().parents[1] / "trexio_example_files" / trexio_file
        _, aos, *_ = read_trexio_file(str(fixture), store_tuple=True)
        rng = np.random.default_rng(2025)
        points = jnp.asarray(rng.normal(size=(5, 3)))
        exponents = jnp.asarray(aos.radial.exponents)
        coefficients = jnp.asarray(aos.radial.coefficients)
        d_exponents = jnp.asarray(rng.normal(scale=0.1, size=exponents.shape))
        d_coefficients = jnp.asarray(rng.normal(scale=0.1, size=coefficients.shape))
        if parameter_kind == "exponents":
            d_coefficients = jnp.zeros_like(d_coefficients)
        else:
            d_exponents = jnp.zeros_like(d_exponents)

        return aos, points, exponents, coefficients, d_exponents, d_coefficients

    def analytical(case):
        aos, points, exponents, coefficients, d_exponents, d_coefficients = case
        return sphe_basis_param_jvp(aos, points, d_exponents, d_coefficients)[0]

    if is_mixed_precision_enabled():
        entry_liveness_atol, _ = get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")

        def checked_analytical(case):
            value = analytical(case)
            aos, points, *_ = case
            expected_shape = (int(aos.num_ao), points.shape[0])
            assert value.shape == expected_shape, (value.shape, expected_shape)
            assert np.all(np.isfinite(value)), value
            assert np.any(np.abs(value) > entry_liveness_atol), "the selected radial parameter block must be active"
            return value

        mixed, full = mixed_vs_full(case_factory, checked_analytical)
        blocks = mixed_fidelity_blocks(mixed, full)
        atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
        for mixed_block, full_block in blocks:
            _assert_orbital_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
        mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            _assert_orbital_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
        jax.clear_caches()
        return

    aos, points, exponents, coefficients, d_exponents, d_coefficients = case_factory()

    def standalone(e, c):
        radial = aos.radial.replace(exponents=e, coefficients=c)
        return compute_AOs(aos.replace(radial=radial), points)

    _, autodiff = jax.jvp(standalone, (exponents, coefficients), (d_exponents, d_coefficients))
    analytical = sphe_basis_param_jvp(aos, points, d_exponents, d_coefficients)[0]
    assert np.all(np.isfinite(analytical)) and np.all(np.isfinite(autodiff))
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")
    assert np.any(np.abs(analytical) > atol), "the selected radial parameter block must be active"
    np.testing.assert_allclose(analytical, autodiff, atol=atol, rtol=rtol, equal_nan=False)
    jax.clear_caches()


def _origin_component(boundary):
    """Build multi-contraction spherical shells and three explicit nucleus-relative points."""
    if boundary == "periodic":
        centre = np.asarray([[1.0, 1.0, 1.0]])
        structure = make_pbc_structure(
            CELL_FIXTURES["cubic_10bohr"],
            positions=centre,
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        molecular = _make_aos_data(structure, "sphe", l_max=2, uniform=False)
        aos = _apply_image_expansion(molecular, structure, tol=AO_IMAGE_TRUNCATION_ATOL)
    else:
        centre = np.zeros((1, 3))
        structure = Structure_data(
            positions=np.zeros((49, 3)),
            pbc_flag=False,
            atomic_numbers=(0,) * 49,
            element_symbols=("X",) * 49,
            atomic_labels=("X",) * 49,
        )
        aos = _make_aos_data(structure, "sphe", l_max=6, uniform=False)
    points = jnp.asarray(centre + np.asarray([[0.0, 0.0, 0.0], [1e-12, -2e-12, 3e-12], [0.31, -0.27, 0.49]]))
    return aos, points


@pytest.mark.parametrize("boundary", ["molecular", "periodic"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_sphe_multi_contraction_origin_values_match_debug(boundary):
    """Spherical AO values at and near nuclei match the independent debug evaluation.

    The synthetic rich AO builder supplies molecular l=0..6 (49 harmonics)
    at the origin or Gamma-periodic l=0..2 (9 harmonics) at (1,1,1) in a
    cubic 10-Bohr cell with production image expansion. Both include
    contraction depths K=3,2,1. Three explicit points have nucleus-relative
    displacements (0,0,0), (1,-2,3)*1e-12 and (0.31,-0.27,0.49) Bohr.
    No many-body spin state or cond(G) is involved. Values must be finite
    and agree with the debug route under the active precision's AO strict
    policy; no full-versus-mixed or bitwise comparison is made.
    """
    aos, points = _origin_component(boundary)
    actual = compute_AOs(aos, points)
    reference = _compute_AOs_debug(aos, np.asarray(points))
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
    jax.clear_caches()


@pytest.mark.parametrize("boundary", ["molecular", "periodic"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_sphe_multi_contraction_origin_derivatives_match_autodiff(boundary):
    """Spherical AO analytic gradients and Laplacians match finite value-route autodiff at nuclei.

    The synthetic rich AO builder supplies molecular l=0..6 (49 harmonics)
    at the origin or Gamma-periodic l=0..2 (9 harmonics) at (1,1,1) in a
    cubic 10-Bohr cell with production image expansion. Both include
    contraction depths K=3,2,1. Three explicit points have nucleus-relative
    displacements (0,0,0), (1,-2,3)*1e-12 and (0.31,-0.27,0.49) Bohr.
    No many-body spin state or cond(G) is involved. Analytic and autodiff
    derivatives must remain finite and agree at the existing strict tolerance
    in full precision. Mixed compares the analytical production derivatives with
    their independently rebuilt full counterparts at the orbital derivative fidelity budget.
    """

    def case_factory():
        return _origin_component(boundary)

    def analytical(case):
        aos, points = case
        return compute_AOs_grad(aos, points), compute_AOs_laplacian(aos, points)

    if is_mixed_precision_enabled():
        mixed, full = mixed_vs_full(case_factory, analytical)
        blocks = mixed_fidelity_blocks(mixed, full)
        atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
        for mixed_block, full_block in blocks:
            _assert_orbital_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
        mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            _assert_orbital_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
        jax.clear_caches()
        return

    aos, points = _origin_component(boundary)
    analytical_gradient = jnp.stack(compute_AOs_grad(aos, points))
    autodiff_gradient = jnp.stack(_compute_AOs_grad_autodiff(aos, points))
    analytical_laplacian = compute_AOs_laplacian(aos, points)
    autodiff_laplacian = _compute_AOs_laplacian_autodiff(aos, points)
    assert np.all(np.isfinite(analytical_gradient)) and np.all(np.isfinite(autodiff_gradient))
    assert np.all(np.isfinite(analytical_laplacian)) and np.all(np.isfinite(autodiff_laplacian))
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")
    np.testing.assert_allclose(analytical_gradient, autodiff_gradient, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(analytical_laplacian, autodiff_laplacian, atol=atol, rtol=rtol, equal_nan=False)
    jax.clear_caches()


def _independent_sphe_prefactors(exponents, coefficients, angular_momenta):
    """NumPy fp64 N_n * N_lm * c with exact integer factorials (no gammaln)."""
    values = []
    for exponent, coefficient, ell in zip(exponents, coefficients, angular_momenta, strict=True):
        ell = int(ell)
        radial = math.sqrt(
            2.0 ** (2 * ell + 3)
            * float(math.factorial(ell + 1))
            * (2.0 * float(exponent)) ** (ell + 1.5)
            / (float(math.factorial(2 * ell + 2)) * math.sqrt(math.pi))
        )
        values.append(radial * math.sqrt((2 * ell + 1) / (4.0 * math.pi)) * float(coefficient))
    return np.asarray(values, dtype=np.float64)


@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Per-primitive basis constants only; no electron coordinates."
)
def test_sphe_guarded_prefactors_are_the_rounded_exact_constants():
    """Mixed guarded spherical AO prefactors equal the float32-rounded exact constants.

    The molecular mixed spherical AO route multiplies each primitive radial
    Gaussian by N_n(l, Z) * N_lm(l) * c. Inputs: every primitive of the water
    ccECP cc-pVQZ TREXIO basis, plus a synthetic l = 0..6 set (five PCG64
    seed-20260930 exponents in [0.05, 60] and coefficients in [-1.5, 1.5] per l).
    With an explicit float32 output dtype in both precision modes, the
    production helper must return exactly the float32 rounding of the
    independent fp64 formula with integer factorials. Forming the constants in
    float32 (exp(gammaln) and the power) is up to 23 ulp off at l = 6 and fails
    this bit comparison. This does not claim that full-precision production
    evaluates AOs in float32, and no outputs of different modes are compared.
    """
    fixture = Path(__file__).resolve().parents[1] / "trexio_example_files" / "water_ccecp_ccpvqz.h5"
    _, aos, *_ = read_trexio_file(str(fixture), store_tuple=True)
    rng = np.random.default_rng(20260930)
    synthetic_l = np.repeat(np.arange(7), 5)
    exponents = np.concatenate([np.asarray(aos._exponents_jnp), rng.uniform(0.05, 60.0, synthetic_l.size)])
    coefficients = np.concatenate([np.asarray(aos._coefficients_jnp), rng.uniform(-1.5, 1.5, synthetic_l.size)])
    angular_momenta = np.concatenate([aos._angular_momentums_prim_np, synthetic_l])
    assert set(angular_momenta.tolist()) == set(range(7))

    actual = np.asarray(
        _sphe_primitive_prefactors(jnp.asarray(exponents), jnp.asarray(coefficients), jnp.asarray(angular_momenta), jnp.float32)
    )
    reference = _independent_sphe_prefactors(exponents, coefficients, angular_momenta).astype(np.float32)
    assert actual.dtype == reference.dtype == np.float32
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    np.testing.assert_array_equal(actual, reference)
    jax.clear_caches()
