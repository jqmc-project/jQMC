"""Independent higher-coordinate and affine responses of periodic Jastrow geometry."""

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

from jqmc._precision import get_tolerance_min
from jqmc.derivatives import force, kinetic, pbc_stress
from tests.test_jqmc_kernel.test_jastrow_skew_geometry import _cell, _data, _positive_distance

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_CELL_KINDS = ["cubic", "rotated-orthogonal", "monoclinic", "triclinic", "strong-skew"]
_FRACTIONS = ((0.17, 0.21, 0.33), (-0.23, 0.11, 0.27), (0.49, -0.37, 0.19))


def _reference_fields(displacement, cell):
    """Differentiate the independent test-only positive law at a nonzero displacement."""
    distance = _positive_distance(displacement, cell)
    gradient = jax.grad(_positive_distance, argnums=0)(displacement, cell)
    hessian = jax.hessian(_positive_distance, argnums=0)(displacement, cell)
    return distance, gradient, jnp.dot(gradient, gradient), jnp.trace(hessian)


@pytest.mark.parametrize("cell_kind", _CELL_KINDS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_periodic_force_kinematics_match_positive_law(cell_kind):
    """Kinetic and force geometry obey the full periodic law through its third coordinate derivatives.

    Five explicit cells are cubic, rotated orthogonal, monoclinic, triclinic
    and the fixed strong-skew matrix ((5,0,0),(4.5,2,0),(4,0.7,3)). Three
    fractional displacements are (0.17,0.21,0.33), (-0.23,0.11,0.27) and
    (0.49,-0.37,0.19). These are geometry evaluation points, not a many-body
    electron/spin state. No files, RNG, determinant or cond(G) filter is used.

    Each actual kinetic/force geometry owner is compared separately against
    AD of the test-only positive double-angle expression: distance, gradient,
    gradient norm squared, Laplacian, and additionally force Hessian and
    gradient of Laplacian. These analytical owners use fp64 geometry in BOTH
    precision modes; the oracle does likewise. The existing strict
    wf_kinetic policy is applied element by element. This is not a comparison
    between full and mixed production outputs, nor a complete force/energy
    estimator guarantee. No finite differences or statistical acceptance are
    involved, and no production geometry helper is called by the reference.
    """
    structure = _data("j1", "exp", cell_kind).structure_data
    cell = _cell(cell_kind)
    atol, rtol = get_tolerance_min(["wf_kinetic"], "strict")
    for sample_index, fraction in enumerate(_FRACTIONS):
        sample_label = str(sample_index)
        r_cart = jnp.asarray(fraction, dtype=jnp.float64) @ cell
        reference_distance, reference_gradient, reference_norm, reference_laplacian = _reference_fields(r_cart, cell)
        reference_hessian = jax.hessian(_positive_distance, argnums=0)(r_cart, cell)
        reference_gradlap = jax.grad(lambda position: _reference_fields(position, cell)[3])(r_cart)
        kinetic_distance, kinetic_gradient, kinetic_norm, kinetic_laplacian = kinetic._dist_and_derivs(structure, r_cart)
        force_distance, force_gradient, force_norm, force_hessian, force_laplacian, force_gradlap = (
            force._smooth_dist_kinematics(structure, r_cart)
        )
        np.testing.assert_allclose(
            kinetic_distance, reference_distance, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(
            kinetic_gradient, reference_gradient, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(kinetic_norm, reference_norm, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label)
        np.testing.assert_allclose(
            kinetic_laplacian, reference_laplacian, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(
            force_distance, reference_distance, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(
            force_gradient, reference_gradient, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(force_norm, reference_norm, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label)
        np.testing.assert_allclose(
            force_hessian, reference_hessian, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(
            force_laplacian, reference_laplacian, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(
            force_gradlap, reference_gradlap, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )


@pytest.mark.parametrize("cell_kind", _CELL_KINDS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_periodic_strain_kinematics_match_positive_law(cell_kind):
    """Both stress geometry owners keep all nine affine derivatives of the full periodic metric.

    Use the five explicit cells and three fixed nonzero fractional
    displacements of the coordinate-kinematics test. The synthetic geometry
    has no file, random generator, determinant, cond(G) selection, or physical
    many-body spin assignment. Apply x -> x F.T and C -> C F.T at F=identity.

    Compare actual distance, coordinate gradient, squared gradient norm and
    Laplacian, and each field's full nine-component strain derivative, to
    independently differentiated positive geometry. Also compare the separate
    Pulay distance/strain helper. Orthogonal-cell shear is included; equality
    of unstrained values alone is insufficient. Both modes retain the
    owners' fp64 geometry and the existing strict wf_kinetic tolerance; no
    cross-mode production comparison, FD, or statistical criterion is used.
    These geometry checks do not assert complete force/stress assembly.
    """
    cell = _cell(cell_kind)
    inv_cell = jnp.linalg.inv(cell)
    atol, rtol = get_tolerance_min(["wf_kinetic"], "strict")
    for sample_index, fraction in enumerate(_FRACTIONS):
        sample_label = str(sample_index)
        r_cart = jnp.asarray(fraction, dtype=jnp.float64) @ cell
        reference_distance, reference_gradient, reference_norm, reference_laplacian = _reference_fields(r_cart, cell)

        def reference_deformed(strain):
            return _reference_fields(r_cart @ strain.T, cell @ strain.T)

        reference_distance_strain, reference_gradient_strain, reference_norm_strain, reference_laplacian_strain = jax.jacfwd(
            reference_deformed
        )(jnp.eye(3, dtype=jnp.float64))
        (
            stress_distance,
            stress_gradient,
            stress_norm,
            stress_laplacian,
            stress_distance_strain,
            stress_gradient_strain,
            stress_norm_strain,
            stress_laplacian_strain,
        ) = pbc_stress._smooth_strain(r_cart, cell, inv_cell)
        pulay_distance, pulay_distance_strain = pbc_stress._dd_dstrain(r_cart, cell, inv_cell)
        np.testing.assert_allclose(
            stress_distance, reference_distance, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(
            stress_gradient, reference_gradient, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(stress_norm, reference_norm, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label)
        np.testing.assert_allclose(
            stress_laplacian, reference_laplacian, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(
            stress_distance_strain, reference_distance_strain, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(
            stress_gradient_strain, reference_gradient_strain, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(
            stress_norm_strain, reference_norm_strain, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(
            stress_laplacian_strain, reference_laplacian_strain, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(
            pulay_distance, reference_distance, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(
            pulay_distance_strain, reference_distance_strain, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
