"""Numerical coincidence conventions and nonzero near-image periodic J2 responses."""

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

from jqmc import jastrow_factor as jastrow
from jqmc._precision import get_dtype_jnp, get_dtype_np, get_tolerance_min
from jqmc._setting import EPS_safe_distance

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _cell(cell_kind="dyadic-triclinic"):
    """Integer lattice entries preserve the selected image offsets in both precision modes."""
    rows = ((5, 0, 0), (0, 5, 0), (0, 0, 5)) if cell_kind == "cubic" else ((5, 0, 0), (1, 6, 0), (2, 1, 7))
    return jnp.asarray(rows, dtype=jnp.float64)


def _pair_fields(displacement, cell, form):
    """Actual conventional J2 pair gradient and Laplacian with the existing b=1 and EPS policy."""
    dtype = get_dtype_jnp("jastrow_grad_lap")
    gradient, laplacian = jastrow._j2_pair_terms_pbc(
        form,
        jnp.asarray(1, dtype=dtype),
        jnp.asarray(EPS_safe_distance, dtype=dtype),
        displacement,
        cell,
        jnp.linalg.inv(cell),
    )
    return jnp.concatenate((gradient, laplacian[None]))


def _quartic_distance_square(direction, cell):
    """Independent degree-four term of the positive periodic distance square near the origin."""
    dtype = get_dtype_jnp("jastrow_grad_lap")
    inverse = jnp.linalg.inv(cell).astype(dtype)
    lattice = cell.astype(dtype)
    vector = direction.astype(dtype)
    fractional = vector @ inverse
    diagonal = jnp.sum(lattice * lattice, axis=1)
    pi = jnp.asarray(jnp.pi, dtype=dtype)
    return pi**2 * (jnp.sum(diagonal * fractional**4) - (4 / 3) * jnp.sum((vector @ lattice.T) * fractional**3))


def _leading_cell_response(direction, cell):
    """Coefficient of r in the J2 Laplacian's geometry-dependent near-cusp expansion."""
    quartic = _quartic_distance_square(direction, cell)
    lap_quartic = jnp.trace(jax.hessian(_quartic_distance_square, argnums=0)(direction, cell))
    return (lap_quartic - 8 * quartic) / 4


@pytest.mark.parametrize("form", ["exp", "pade"])
@pytest.mark.parametrize("cell_kind", ["cubic", "dyadic-triclinic"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_periodic_j2_masked_zero_outer_ad(form, cell_kind):
    """The excluded zero-displacement J2 pair has exact zero fields and zero outer AD.

    Use the cubic 5-Bohr or fixed integer triclinic cell, b=1 and an exactly
    zero displacement. This represents the numerical self-pair mask, not
    a differentiable physical cusp. No many-body spin assignment, file,
    RNG, determinant or cond(G) filter is involved. Actual pair gradient
    and Laplacian, their coordinate Jacobian, cell forward Jacobian and
    reverse cell gradient must all be exactly zero. Both precision modes
    retain the existing Jastrow gradient/Laplacian zone and EPS setting.
    Fields and forward Jacobians have that zone's dtype; the reverse cell
    gradient has the fixed fp64 cell-input dtype. Reference shapes and
    dtypes are specified independently of the computed outputs.
    """
    cell = _cell(cell_kind)
    r_cart = jnp.zeros(3, dtype=get_dtype_jnp("jastrow_grad_lap"))
    actual_fields = _pair_fields(r_cart, cell, form)
    actual_position_jacobian = jax.jacfwd(lambda position: _pair_fields(position, cell, form))(r_cart)
    actual_cell_jacobian = jax.jacfwd(lambda lattice: _pair_fields(r_cart, lattice, form))(cell)
    actual_cell_gradient = jax.grad(lambda lattice: jnp.sum(_pair_fields(r_cart, lattice, form)))(cell)
    field_dtype = get_dtype_np("jastrow_grad_lap")
    reference_fields = np.zeros(4, dtype=field_dtype)
    reference_position_jacobian = np.zeros((4, 3), dtype=field_dtype)
    reference_cell_jacobian = np.zeros((4, 3, 3), dtype=field_dtype)
    reference_cell_gradient = np.zeros((3, 3), dtype=cell.dtype)
    np.testing.assert_array_equal(actual_fields, reference_fields)
    np.testing.assert_array_equal(actual_position_jacobian, reference_position_jacobian)
    np.testing.assert_array_equal(actual_cell_jacobian, reference_cell_jacobian)
    np.testing.assert_array_equal(actual_cell_gradient, reference_cell_gradient)


@pytest.mark.parametrize("cell_kind", ["cubic", "dyadic-triclinic"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_periodic_distance_lattice_zero_ad_convention(cell_kind):
    """Exact periodic coincidences retain the declared zero-tangent and finite-clamp conventions.

    In the cubic or integer triclinic cell, test displacement zero and
    lattice vector combination (1,-1,1). Actual distance and its coordinate
    and co-moving cell AD are exactly zero by the numerical extension.
    The distance is NOT mathematically differentiable at either point.
    At the nonzero lattice image the J2 pair helper retains its existing
    EPS clamp instead of the Cartesian self-pair mask: require finite
    fields and outer coordinate/cell derivatives, not a zero Laplacian or
    equality of undefined physical derivatives. Both modes use their own
    arithmetic. There are no files, RNG, many-body spin state or determinant.
    """
    cell = _cell(cell_kind)
    for image_index, image in enumerate(((0, 0, 0), (1, -1, 1))):
        sample_label = str(image_index)
        image = jnp.asarray(image, dtype=jnp.float64)
        r_cart = image @ cell

        def distance(position, lattice):
            return jastrow._smooth_periodic_distance(position, lattice, jnp.linalg.inv(lattice))

        actual_distance = distance(r_cart, cell)
        actual_position_gradient = jax.grad(distance, argnums=0)(r_cart, cell)
        actual_cell_gradient = jax.grad(lambda lattice: distance(image @ lattice, lattice))(cell)
        reference_distance = np.asarray(0, dtype=r_cart.dtype)
        reference_position_gradient = np.zeros(3, dtype=r_cart.dtype)
        reference_cell_gradient = np.zeros((3, 3), dtype=cell.dtype)
        fields = _pair_fields(r_cart, cell, "pade")
        position_jacobian = jax.jacfwd(lambda position: _pair_fields(position, cell, "pade"))(r_cart)
        cell_jacobian = jax.jacfwd(lambda lattice: _pair_fields(image @ lattice, lattice, "pade"))(cell)
        cell_gradient = jax.grad(lambda lattice: jnp.sum(_pair_fields(image @ lattice, lattice, "pade")))(cell)
        actual_finite = np.isfinite(
            np.concatenate(
                (
                    np.asarray(fields).ravel(),
                    np.asarray(position_jacobian).ravel(),
                    np.asarray(cell_jacobian).ravel(),
                    np.asarray(cell_gradient).ravel(),
                )
            )
        )
        reference_finite = np.ones(61, dtype=bool)
        np.testing.assert_array_equal(actual_distance, reference_distance, err_msg=sample_label)
        np.testing.assert_array_equal(actual_position_gradient, reference_position_gradient, err_msg=sample_label)
        np.testing.assert_array_equal(actual_cell_gradient, reference_cell_gradient, err_msg=sample_label)
        np.testing.assert_array_equal(actual_finite, reference_finite, err_msg=sample_label)


@pytest.mark.parametrize("form", ["exp", "pade"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_periodic_j2_near_cusp_outer_response(form, axis):
    """Nonzero near-cusp J2 coordinate and cell responses match an independent local series.

    The fixed integer triclinic cell has rows (5,0,0), (1,6,0), (2,1,7).
    Use b=1, radius 1e-9 Bohr and each Cartesian axis, with exp/Pade in
    each precision mode. There is no lattice-image offset, so the input
    remains nonzero in either active zone. No file, RNG, determinant,
    condition-number selection or finite difference is used.

    Write d^2=r^2+A4+O(r^6). A4 is derived independently from the positive
    distance law; its homogeneous quartic derivatives give the nonzero
    transverse gradient of the J2 Laplacian and the nine cell responses
    divided by r. The radial gradient of Laplacian multiplied by r^2 tends
    to minus the unit direction. These normalized finite-radius quantities
    use existing strict Jastrow-zone tolerances. Differentiate the series
    at a unit direction, avoiding an ill-conditioned higher AD reference
    at a tiny radius. This is a local asymptotic check, complemented by the
    finite-distance law tests, and never compares different precision modes.
    """
    cell = _cell()
    dtype = get_dtype_jnp("jastrow_grad_lap")
    direction = jnp.eye(3, dtype=dtype)[axis]
    radius = jnp.asarray(1e-9, dtype=dtype)
    r_cart = radius * direction

    def laplacian(position, lattice):
        return _pair_fields(position, lattice, form)[3]

    actual_gradlap = jax.grad(laplacian, argnums=0)(r_cart, cell)
    actual_cell_response = jax.jacfwd(laplacian, argnums=1)(r_cart, cell) / radius
    quartic = _quartic_distance_square(direction, cell)
    quartic_gradient = jax.grad(_quartic_distance_square, argnums=0)(direction, cell)
    quartic_laplacian = jnp.trace(jax.hessian(_quartic_distance_square, argnums=0)(direction, cell))
    quartic_gradlap = jax.grad(lambda vector: jnp.trace(jax.hessian(_quartic_distance_square, argnums=0)(vector, cell)))(
        direction
    )
    reference_gradlap = (quartic_gradlap - quartic_laplacian * direction - 8 * quartic_gradient + 24 * quartic * direction) / 4
    tangent = [index for index in range(3) if index != axis]
    actual_transverse = np.asarray(actual_gradlap)[tangent]
    reference_transverse = np.asarray(reference_gradlap)[tangent]
    actual_scaled_radial = actual_gradlap * radius**2
    reference_scaled_radial = -direction
    reference_cell_response = jax.jacfwd(_leading_cell_response, argnums=1)(direction, cell)
    atol, rtol = get_tolerance_min(["jastrow_grad_lap"], "strict")
    np.testing.assert_allclose(actual_transverse, reference_transverse, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_scaled_radial, reference_scaled_radial, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_cell_response, reference_cell_response, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("form", ["exp", "pade"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_periodic_j2_near_image_response(form, axis):
    """Representable nonzero near-image J2 fields and outer responses are lattice periodic.

    Use the integer triclinic cell, b=1, exp/Pade, and displacement 2^-12
    Bohr along each Cartesian axis. Add lattice vector combination (1,-1,1).
    Both the unshifted and image inputs are exactly representable in both
    modes. This does not claim recovery of sub-ULP offsets rounded away
    by an existing zone cast. No files, RNG, determinant or cond(G) filter
    is involved. The compared quantities are actual pair fields, their
    coordinate Jacobian and their cell Jacobian with x(C)=n C+r*u, so the
    local displacement is held fixed under the cell response.

    Central strict Jastrow-zone tolerances apply in each mode. This is a
    translation identity, not an independent mathematical oracle: independent
    finite-distance and near-cusp-series evidence is required separately.
    No finite differences, stochastic criteria or cross-mode comparison occur.
    """
    cell = _cell()
    direction = jnp.eye(3, dtype=jnp.float64)[axis]
    reference_r_cart = 2.0**-12 * direction
    image = jnp.asarray((1, -1, 1), dtype=jnp.float64)
    r_cart = image @ cell + reference_r_cart
    actual_fields = _pair_fields(r_cart, cell, form)
    reference_fields = _pair_fields(reference_r_cart, cell, form)
    actual_position_jacobian = jax.jacfwd(lambda position: _pair_fields(position, cell, form))(r_cart)
    reference_position_jacobian = jax.jacfwd(lambda position: _pair_fields(position, cell, form))(reference_r_cart)
    actual_cell_jacobian = jax.jacfwd(lambda lattice: _pair_fields(image @ lattice + reference_r_cart, lattice, form))(cell)
    reference_cell_jacobian = jax.jacfwd(lambda lattice: _pair_fields(reference_r_cart, lattice, form))(cell)
    atol, rtol = get_tolerance_min(["jastrow_grad_lap"], "strict")
    np.testing.assert_allclose(actual_fields, reference_fields, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_position_jacobian, reference_position_jacobian, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_cell_jacobian, reference_cell_jacobian, atol=atol, rtol=rtol, equal_nan=False)
