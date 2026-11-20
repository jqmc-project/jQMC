"""Independent periodic-law and directional-cusp checks for conventional J1/J2."""

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
from jqmc.structure import Structure_data

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_CASES = [
    ("j1", "exp", "cubic", "up"),
    ("j1", "pade", "tetragonal", "down"),
    ("j2", "exp", "orthorhombic", "up-down"),
    ("j2", "pade", "rotated-orthogonal", "up-up"),
    ("j1", "exp", "monoclinic", "up"),
    ("j2", "pade", "monoclinic", "down-down"),
    ("j1", "pade", "triclinic", "down"),
    ("j2", "exp", "strong-skew", "up-down"),
]
_CASE_IDS = ["-".join(case) for case in _CASES]
_FRACTIONS = ((0.17, 0.21, 0.33), (-0.23, 0.11, 0.27), (0.49, -0.37, 0.19))
_CENTER = (0.2, 0.3, 0.4)


def _cell(cell_kind):
    """Seven explicit nonsingular cells; strong-skew is a bounded fixture, not a condition limit."""
    cells = {
        "cubic": ((5, 0, 0), (0, 5, 0), (0, 0, 5)),
        "tetragonal": ((5, 0, 0), (0, 5, 0), (0, 0, 7)),
        "orthorhombic": ((5, 0, 0), (0, 6, 0), (0, 0, 7)),
        "rotated-orthogonal": ((3, 4, 0), (-4.8, 3.6, 0), (0, 0, 7)),
        "monoclinic": ((5, 0, 0), (0, 6, 0), (2, 0, 7)),
        "triclinic": ((5, 0, 0), (1, 6, 0), (2, 0.8, 7)),
        "strong-skew": ((5, 0, 0), (4.5, 2, 0), (4, 0.7, 3)),
    }
    return jnp.asarray(cells[cell_kind], dtype=jnp.float64)


def _data(part, form, cell_kind):
    """Build synthetic periodic components with b=0.75 and one effective H nucleus."""
    cell = _cell(cell_kind)
    structure = Structure_data.build(
        positions=(_CENTER,),
        atomic_numbers=(1,),
        element_symbols=("H",),
        atomic_labels=("H1",),
        pbc_flag=True,
        vec_a=cell[0],
        vec_b=cell[1],
        vec_c=cell[2],
    )
    if part == "j1":
        return jastrow.Jastrow_one_body_data.build(
            jastrow_1b_param=0.75, jastrow_1b_type=form, structure_data=structure, core_electrons=(0,)
        )
    return jastrow.Jastrow_two_body_data.build(jastrow_2b_param=0.75, jastrow_2b_type=form, structure_data=structure)


def _positions(displacement, spin_sector, center):
    """One moved electron; J2 has exactly one partner at the translated origin."""
    moved = (center + displacement)[None, :]
    fixed = center[None, :]
    empty = jnp.empty((0, 3), dtype=moved.dtype)
    if spin_sector == "up":
        return moved, empty, 0
    if spin_sector == "down":
        return empty, moved, 1
    if spin_sector == "up-down":
        return moved, fixed, 0
    if spin_sector == "up-up":
        return jnp.concatenate((moved, fixed)), empty, 0
    return empty, jnp.concatenate((moved, fixed)), 1


def _positive_distance(displacement, cell):
    """Independent positive form: squared norm plus squared sine term, never a production helper."""
    pi = jnp.asarray(jnp.pi, dtype=cell.dtype)
    fractional = jnp.linalg.solve(cell.T, displacement)
    theta = pi * fractional
    vector = jnp.sin(2 * theta) @ cell
    height = 2 * jnp.sin(theta) ** 2
    squared = jnp.sum(vector * vector) + jnp.sum(jnp.sum(cell * cell, axis=1) * height * height)
    return jnp.sqrt(squared) / (2 * pi)


def _radial_value(radius, part, form):
    """Independent conventional radial forms, with the established spin-independent J2 factor."""
    b = jnp.asarray(0.75, dtype=radius.dtype)
    scale = jnp.asarray(2, dtype=radius.dtype) ** jnp.asarray(0.25, dtype=radius.dtype) if part == "j1" else 1
    coefficient = -1 if part == "j1" else 0.5
    if form == "pade":
        return coefficient * radius / (1 + b * scale * radius)
    return coefficient * (-jnp.expm1(-b * scale * radius)) / (b * scale)


def _reference(displacement, cell, part, form, zone):
    """Apply each route's existing geometry/zone boundary without comparing precision modes."""
    dtype = get_dtype_jnp(zone)
    if zone == "jastrow_grad_lap":
        radius = _positive_distance(displacement.astype(dtype), cell.astype(dtype))
    else:
        radius = _positive_distance(displacement, cell).astype(dtype)
    return _radial_value(radius, part, form)


@pytest.mark.parametrize("part,form,cell_kind,spin_sector", _CASES, ids=_CASE_IDS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_periodic_jastrow_directional_cusp(part, form, cell_kind, spin_sector):
    """Periodic J1/J2 retain Euclidean cusp-normalized slopes in four Cartesian directions.

    Eight selected component/form/cell/spin rows cover cubic, tetragonal,
    orthorhombic, rotated orthogonal, monoclinic, triclinic and strong-skew
    cells listed in _cell. J1 has one up/down electron and one translated H
    nucleus; J2 has two electrons in up-down/up-up/down-down sectors. Use
    b=0.75, radius 1e-7 Bohr, directions +x, -y, +z and (1,1,1)/sqrt(3).
    No files, RNG, orbitals, determinant or cond(G) selection is involved.

    Compare actual forward-AD and analytical radial slopes to independent
    same-mode exp/Pade radial envelopes. Periodic corrections at this small
    nonzero radius are below the existing strict component policy. J1's
    limiting slope is -1; all conventional J2 pairs retain 1/2, with no
    claim of the exact same-spin 1/4 cusp. No FD or cross-precision oracle
    is used. DTN and complete-wavefunction cusp are outside this test.
    """
    data = _data(part, form, cell_kind)
    value = jastrow.compute_Jastrow_one_body if part == "j1" else jastrow.compute_Jastrow_two_body
    derivatives = (
        jastrow.compute_grads_and_laplacian_Jastrow_one_body
        if part == "j1"
        else jastrow.compute_grads_and_laplacian_Jastrow_two_body
    )
    center = jnp.asarray(_CENTER, dtype=jnp.float64)
    directions = np.asarray(((1, 0, 0), (0, -1, 0), (0, 0, 1), (1, 1, 1)), dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    dtype = get_dtype_np("jastrow_grad_lap")
    radius = dtype(1e-7)
    scale = dtype(2) ** dtype(0.25) if part == "j1" else dtype(1)
    coefficient = dtype(-1) if part == "j1" else dtype(0.5)
    exponent = dtype(0.75) * scale * radius
    reference_slope = coefficient * (np.exp(-exponent) if form == "exp" else dtype(1) / (dtype(1) + exponent) ** 2)
    atol, rtol = get_tolerance_min(["jastrow_eval", "jastrow_grad_lap"], "strict")
    for sample_index, direction in enumerate(directions):
        sample_label = str(sample_index)
        r_up, r_dn, index = _positions(jnp.asarray(1e-7 * direction), spin_sector, center)
        automatic = jax.grad(value, argnums=(1, 2))(data, r_up, r_dn)
        analytical = derivatives(data, r_up, r_dn)
        autodiff_slope = np.dot(np.asarray(automatic[index])[0], direction)
        analytical_slope = np.dot(np.asarray(analytical[index])[0], direction)
        np.testing.assert_allclose(autodiff_slope, reference_slope, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label)
        np.testing.assert_allclose(
            analytical_slope, reference_slope, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )


@pytest.mark.parametrize("part,form,cell_kind,spin_sector", _CASES, ids=_CASE_IDS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_periodic_jastrow_positive_law_values_and_derivatives(part, form, cell_kind, spin_sector):
    """Periodic component values, gradients and Laplacians follow the positive full-Gram law.

    Use the same eight synthetic component/form/cell/spin rows as the
    directional-cusp test. Three explicit fractional displacements are
    (0.17,0.21,0.33), (-0.23,0.11,0.27), (0.49,-0.37,0.19); the common
    center is (0.2,0.3,0.4) Bohr. Translate the moved electron additionally
    by lattice vector combination (1,-1,2). There is no sampling, file
    fixture, determinant or condition-number screening.

    The independent positive expression uses sin(2*pi*s) and squared
    2*sin(pi*s)^2, with no shared production metric. Actual forward values
    and analytical gradient/Laplacian match this expression and its AD
    in their respective active zones. Full and mixed each use the existing
    strict element-wise policy. The same-spin J2 convention remains 1/2.
    This verifies selected fresh XLA component routes, not native/cache or
    whole-wavefunction coverage, and contains no finite differences.
    """
    data = _data(part, form, cell_kind)
    cell = _cell(cell_kind)
    center = jnp.asarray(_CENTER, dtype=jnp.float64)
    value = jastrow.compute_Jastrow_one_body if part == "j1" else jastrow.compute_Jastrow_two_body
    derivatives = (
        jastrow.compute_grads_and_laplacian_Jastrow_one_body
        if part == "j1"
        else jastrow.compute_grads_and_laplacian_Jastrow_two_body
    )
    atol, rtol = get_tolerance_min(["jastrow_eval", "jastrow_grad_lap"], "strict")
    for sample_index, fraction in enumerate(_FRACTIONS):
        sample_label = str(sample_index)
        displacement = jnp.asarray(fraction, dtype=jnp.float64) @ cell
        r_up, r_dn, index = _positions(displacement, spin_sector, center)
        actual_value = value(data, r_up, r_dn)
        reference_value = _reference(displacement, cell, part, form, "jastrow_eval")
        actual = derivatives(data, r_up, r_dn)
        analytical_gradient, analytical_laplacian = actual[index][0], actual[index + 2][0]

        def reference_position(position):
            return _reference(position, cell, part, form, "jastrow_grad_lap")

        reference_gradient = jax.grad(reference_position)(displacement)
        reference_laplacian = jnp.trace(jax.hessian(reference_position)(displacement))
        moved = displacement + jnp.asarray((1, -1, 2), dtype=jnp.float64) @ cell
        translated_r_up, translated_r_dn, _ = _positions(moved, spin_sector, center)
        actual_translated = value(data, translated_r_up, translated_r_dn)
        np.testing.assert_allclose(actual_value, reference_value, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label)
        np.testing.assert_allclose(
            analytical_gradient, reference_gradient, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(
            analytical_laplacian, reference_laplacian, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )
        np.testing.assert_allclose(
            actual_translated, reference_value, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label
        )


@pytest.mark.parametrize("part,form,cell_kind,spin_sector", _CASES, ids=_CASE_IDS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_periodic_jastrow_all_strain_components(part, form, cell_kind, spin_sector):
    """All nine affine-strain responses obey the periodic law, including orthogonal-cell shear.

    Use the same eight selected synthetic component/form/cell/spin rows and
    three finite displacements as the positive-law test. Apply x -> x F.T
    to both electron positions and the translated center, and C -> C F.T
    to the cell, at F=identity. Differentiate each of all nine independent
    entries, without a volume-only restriction or a symmetric-component
    subset. No files, RNG, determinant or cond(G) screening are involved.

    Public forward-AD must match AD of the independent positive full-Gram
    expression using the same active evaluation-zone casts and strict
    element-wise policy in each precision mode. The orthogonal rows are
    essential: equality of unstrained values with a diagonal-only metric
    does not permit dropping its nonzero shear response. No FD,
    mixed-versus-full comparison or native-route claim is made.
    """
    data = _data(part, form, cell_kind)
    cell = _cell(cell_kind)
    center = jnp.asarray(_CENTER, dtype=jnp.float64)
    value = jastrow.compute_Jastrow_one_body if part == "j1" else jastrow.compute_Jastrow_two_body
    atol, rtol = get_tolerance_min(["jastrow_eval", "jastrow_grad_lap"], "strict")
    for sample_index, fraction in enumerate(_FRACTIONS):
        sample_label = str(sample_index)
        r_cart = jnp.asarray(fraction, dtype=jnp.float64) @ cell

        def actual_deformed(strain):
            deformed_cell = cell @ strain.T
            deformed_center = center @ strain.T
            structure = data.structure_data.replace(
                vec_a=deformed_cell[0],
                vec_b=deformed_cell[1],
                vec_c=deformed_cell[2],
                positions=deformed_center[None, :],
            )
            r_up, r_dn, _ = _positions(r_cart @ strain.T, spin_sector, deformed_center)
            return value(data.replace(structure_data=structure), r_up, r_dn)

        def reference_deformed(strain):
            return _reference(r_cart @ strain.T, cell @ strain.T, part, form, "jastrow_eval")

        identity = jnp.eye(3, dtype=jnp.float64)
        actual_strain = jax.grad(actual_deformed)(identity)
        reference_strain = jax.grad(reference_deformed)(identity)
        np.testing.assert_allclose(actual_strain, reference_strain, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label)
