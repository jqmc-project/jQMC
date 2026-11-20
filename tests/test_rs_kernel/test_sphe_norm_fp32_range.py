"""Range of the mixed-precision spherical AO normalization, periodic values and basis sensitivities (issue #111).

Every spherical primitive carries the radial normalization

    N_n = sqrt(2^(2l+3) (l+1)! (2a)^(l+3/2) / ((2l+2)! sqrt(pi))),

a constant of the basis that does not depend on the electron coordinates. Assembled from
float32 pieces in mixed precision, ``(2a)^(l+3/2)`` (and the products around it) overflows
float32 and the result comes out ``inf`` although N_n and the quantity itself are
representable in float32, in three kernels:

- the XLA periodic spherical pGTO value (``AOs_sphe_gto_data._compute_AOs``, non-guarded
  branch; also used by the complex-Bloch leaf);
- the Rust f32 value kernel of periodic (and complex-Bloch) spherical pGTO leaves
  (``primitive_normalization_sphe_f32``), whose window was 10-20x narrower than XLA's;
- the mixed basis-parameter sensitivity ``d AO / d a`` and ``d AO / d c``
  (``derivatives/basis.py::_sphe_value_adjoint_fields``), for molecular and periodic leaves.

The molecular forward value already formed N_n in float64 and rounded once.

The cases are single m = 0 primitives (coefficient 1) at exponents just outside the old
mixed-precision windows, evaluated at the radial maximum r^2 = l / (2a) on the z axis and at
the same radius in a tilted direction, where the exact values are O(1e2) .. O(1e8). The
reference is the closed form

    phi(a, r) = N_r(a) |d|^l exp(-a |d|^2) P_l(cos theta) sqrt((2l+1) / (4 pi)),
    N_r(a)^2 = 2 (2a)^(l+3/2) / Gamma(l+3/2),

evaluated with ``math.lgamma`` and plain Python (no jqmc code), with ``d`` the displacement from
the atom. The periodic reference sums the lattice images n in [-1, 1]^3; at these exponents only
the home image is above the float64 range, so the complex-Bloch value equals the same real sum.
The exponent derivative reference is the central finite difference of the closed form in
``a`` (relative step ``_FD_REL_STEP``), and the coefficient derivative is the closed-form value
itself. Tolerances are the strict ao_eval tolerances of the active precision mode. In full
precision the kernels are unchanged and the same comparisons hold.
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

import itertools
from math import exp, lgamma, log, pi, sqrt

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import legendre

import jqmc_rs

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance
from jqmc.atomic_orbital import (
    AOs_sphe_gto_data,
    AOs_sphe_pgto_data,
    build_AOs_sphe_pgto_complex_data,
    compute_AOs,
)
from jqmc.derivatives.basis import _value_general_leaf
from jqmc.structure import Structure_data

#: Cubic cell edge (bohr) of the periodic leaves; the atom sits at the cell centre.
_CELL = 40.0
#: Fractional twist of the complex-Bloch leaf.
_TWIST = (0.25, 0.0, 0.0)
#: Relative step of the central finite difference in the exponent.
_FD_REL_STEP = 1.0e-6

#: case -> (l, exponent). Old mixed periodic value windows (upper bounds): XLA l=2 3.2e10,
#: 3 1.8e8, 4 3.2e6, 5 3.2e5, 6 5.6e4; Rust l=2 5.6e9, 3 1.8e7, 4 3.2e5, 5 3.2e4, 6 3.2e3.
#: The "_rust" cases sit between the two bounds (Rust inf, XLA finite), the others above
#: both. The old mixed basis-sensitivity bounds lie below the "_both" exponents too.
_CASES = {
    "l2_a1e10_rust": (2, 1.0e10),
    "l3_a1e8_rust": (3, 1.0e8),
    "l4_a1e6_rust": (4, 1.0e6),
    "l5_a1e5_rust": (5, 1.0e5),
    "l6_a1e4_rust": (6, 1.0e4),
    "l2_a1e11_both": (2, 1.0e11),
    "l3_a1e9_both": (3, 1.0e9),
    "l4_a1e7_both": (4, 1.0e7),
    "l5_a1e6_both": (5, 1.0e6),
    "l6_a1e5_both": (6, 1.0e5),
}


def _closed_form(l_ang, a, displacements):
    """Sum of N_r |d|^l exp(-a d^2) P_l(cos theta) sqrt((2l+1)/(4 pi)) over displacements (no jqmc code)."""
    ln_norm = 0.5 * (log(2.0) + (l_ang + 1.5) * log(2.0 * a) - lgamma(l_ang + 1.5))
    total = 0.0
    for d in displacements:
        r2 = float(d @ d)
        if ln_norm - a * r2 < -740.0:  # below the float64 range: contributes exactly nothing
            continue
        r = sqrt(r2)
        cos_theta = float(d[2]) / r
        total += (
            exp(ln_norm + l_ang * log(r) - a * r2)
            * legendre.legval(cos_theta, [0] * l_ang + [1])
            * sqrt((2 * l_ang + 1) / (4 * pi))
        )
    return total


def _points(l_ang, a, centre):
    """Radial maximum on the z axis and at the same radius in a tilted direction."""
    rr = sqrt(l_ang / (2.0 * a))
    tilt = np.array([0.3, 0.2, 1.0]) / np.linalg.norm([0.3, 0.2, 1.0])
    return centre[None, :] + rr * np.array([[0.0, 0.0, 1.0], tilt])


def _reference(leaf, l_ang, a, points, centre):
    if leaf == "molecular":
        return np.array([_closed_form(l_ang, a, [p - centre]) for p in points])
    images = [_CELL * np.array(n, dtype=np.float64) for n in itertools.product((-1, 0, 1), repeat=3)]
    return np.array([_closed_form(l_ang, a, [p - centre - t for t in images]) for p in points])


def _build(leaf, l_ang, a):
    """One m = 0 spherical primitive (coefficient 1) on a molecular, periodic or complex-Bloch leaf."""
    common = dict(
        nucleus_index=(0,),
        num_ao=1,
        num_ao_prim=1,
        orbital_indices=(0,),
        exponents=np.array([a], dtype=np.float64),
        coefficients=np.array([1.0], dtype=np.float64),
        angular_momentums=(l_ang,),
        magnetic_quantum_numbers=(0,),
    )
    if leaf == "molecular":
        structure = Structure_data(
            pbc_flag=False,
            positions=np.zeros((1, 3)),
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H",),
        )
        return AOs_sphe_gto_data.build(structure_data=structure, **common), np.zeros(3)
    centre = np.full(3, _CELL / 2.0)
    structure = Structure_data(
        pbc_flag=True,
        positions=centre[None, :],
        atomic_numbers=(1,),
        element_symbols=("H",),
        atomic_labels=("H",),
        vec_a=(_CELL, 0.0, 0.0),
        vec_b=(0.0, _CELL, 0.0),
        vec_c=(0.0, 0.0, _CELL),
    )
    if leaf == "periodic":
        return AOs_sphe_pgto_data.build(structure_data=structure, **common), centre
    return build_AOs_sphe_pgto_complex_data(structure, k_point=_TWIST, **common), centre


@pytest.mark.light
@pytest.mark.parametrize("leaf", ["periodic", "complex"])
@pytest.mark.parametrize("case", list(_CASES))
def test_periodic_spherical_ao_value_normalization_keeps_fp32_range(case, leaf):
    """XLA and Rust periodic / complex-Bloch spherical pGTO values stay finite and equal the closed form.

    Each case builds one m = 0 spherical primitive on a periodic (``AOs_sphe_pgto_data``) or
    complex-Bloch (``AOs_sphe_pgto_complex_data``) leaf and evaluates the XLA ``compute_AOs``
    and the Rust ``jqmc_rs.compute_AOs`` at the radial maximum (z axis and a tilted direction).
    Both must be finite and equal the lgamma closed-form lattice sum within the strict ao_eval
    tolerance of the active mode; the complex value must have a vanishing imaginary part (only
    the home image contributes). A radial normalization assembled from float32 pieces makes the
    mixed values ``inf``.
    """
    l_ang, a = _CASES[case]
    aos, centre = _build(leaf, l_ang, a)
    points = _points(l_ang, a, centre)
    expected = _reference(leaf, l_ang, a, points, centre)

    atol, rtol = get_tolerance("ao_eval", "strict")
    # The reference is O(1e2) .. O(1e8) here, far above atol: inf, 0 or a rescaled value cannot hide.
    assert np.all(np.isfinite(expected))
    assert np.min(np.abs(expected)) > atol
    for what, actual in (
        ("XLA compute_AOs", compute_AOs(aos, points)),
        ("Rust compute_AOs", jqmc_rs.compute_AOs(aos, points)),
    ):
        actual = np.asarray(actual)
        assert actual.shape == (1, points.shape[0]), what
        assert np.all(np.isfinite(actual)), what
        if leaf == "complex":
            np.testing.assert_allclose(actual.imag, 0.0, atol=atol, rtol=rtol, err_msg=what)
            actual = actual.real
        np.testing.assert_allclose(np.asarray(actual, dtype=np.float64)[0], expected, atol=atol, rtol=rtol, err_msg=what)


@pytest.mark.light
@pytest.mark.parametrize("leaf", ["molecular", "periodic", "complex"])
@pytest.mark.parametrize("case", list(_CASES))
def test_spherical_basis_parameter_sensitivity_keeps_fp32_range(case, leaf):
    """The spherical basis-parameter sensitivities d AO/d a and d AO/d c stay finite and match the closed form.

    The value-block basis sensitivity ``_value_general_leaf`` (the determinant and J3 O_k
    basis-exponent / coefficient leaf; in mixed it uses ``_sphe_value_adjoint_fields``) is
    evaluated with a unit AO cotangent for one m = 0 spherical primitive on a molecular,
    periodic or complex-Bloch leaf, at each point separately. ``d AO/d a`` must equal the
    central finite difference of the lgamma closed form in ``a`` and ``d AO/d c`` the closed
    form itself (coefficient 1), within the strict ao_eval tolerance of the active mode.
    An N_n assembled from float32 pieces makes both ``inf`` in mixed, although the forward AO
    value is finite.
    """
    l_ang, a = _CASES[case]
    aos, centre = _build(leaf, l_ang, a)
    points = _points(l_ang, a, centre)
    h = _FD_REL_STEP * a
    expected_da = (_reference(leaf, l_ang, a + h, points, centre) - _reference(leaf, l_ang, a - h, points, centre)) / (2.0 * h)
    expected_dc = _reference(leaf, l_ang, a, points, centre)

    atol, rtol = get_tolerance("ao_eval", "strict")
    # d AO/d a is O(1e-4) .. O(1e-1) and d AO/d c O(1e2) .. O(1e8) here, above atol.
    assert np.all(np.isfinite(expected_da)) and np.all(np.isfinite(expected_dc))
    assert np.min(np.abs(expected_da)) > atol
    actual_da, actual_dc = [], []
    for p in points:
        g_a, g_c = _value_general_leaf(aos, None, jnp.ones((1, 1)), jnp.asarray(p[None, :]))
        assert np.asarray(g_a).shape == (1,) and np.asarray(g_c).shape == (1,)
        actual_da.append(float(np.asarray(g_a)[0]))
        actual_dc.append(float(np.asarray(g_c)[0]))
    for what, actual, expected in (("d AO/d a", actual_da, expected_da), ("d AO/d c", actual_dc, expected_dc)):
        actual = np.asarray(actual, dtype=np.float64)
        assert np.all(np.isfinite(actual)), what
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=what)
