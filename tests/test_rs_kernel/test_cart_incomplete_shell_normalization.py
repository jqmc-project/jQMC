"""Cartesian GTO normalization for incomplete Cartesian shells, XLA and Rust (issue #110).

A Cartesian primitive is ``c N x^nx y^ny z^nz exp(-a r^2)`` with the closed-form
normalization

    N^-2 = prod_i Gamma(n_i + 1/2) / (2 a)^(n_i + 1/2),

equivalently ``N^2 = (2a/pi)^(3/2) (8a)^l prod_i n_i! / (2 n_i)!`` with the TOTAL
degree ``l = nx + ny + nz``. The XLA kernels evaluate ``(8a)^l`` with the static
unrolled power helper ``_int_pow_unrolled_cart``, which only handles exponents up to
its static bound. That bound must be the total degree: the largest single coordinate
power ``max(nx, ny, nz)`` of the AO set is smaller than ``l`` whenever a shell of the
top degree is incomplete (a lone d_xy has ``l = 2`` but a coordinate bound of 1), so
with that bound ``(8a)^l`` would be truncated to ``(8a)^1`` and every XLA AO field
scaled by ``(8a)^((1 - l)/2)``. Converter-built bases keep complete shells and cannot
show this.

This module builds AO sets with incomplete shells (a lone d_xy, a lone (2,1,0), a p
shell plus a lone d_xy, a lone (5,5,0) and a lone (41,41,0)) and a complete d shell
as a control, through the public ``AOs_cart_gto_data.build`` factory, and compares

* XLA ``compute_AOs``, ``compute_AOs_grad``, ``compute_AOs_laplacian`` and the fused
  ``compute_AOs_value_grad_lap``;
* Rust ``jqmc_rs.compute_AOs``, ``compute_AOs_grad`` and ``compute_AOs_laplacian``

with the closed form above, evaluated with ``math.lgamma`` and plain Python (no jqmc
code): value ``c N e^{-a r^2} P``, gradient ``c N e^{-a r^2} (d_i P - 2 a x_i P)`` and
Laplacian ``c N e^{-a r^2} (lap P - (4 a l + 6 a) P + 4 a^2 r^2 P)``. The points sit
at and around the radial maximum of the probed component, where the exact field is
O(1), so a constant scale error cannot hide under the absolute tolerance.

The standalone ``compute_AOs`` value of the lone (41,41,0) component is not compared
here: in mixed precision the ao_eval value kernel forms its normalization and its
polynomial x^41 y^41 in float32, which overflow/underflow for that degree (the separate
mixed-precision range issue #111). Its gradient, Laplacian and fused fields
(normalization and polynomial in float64, value rounded once to the ao_eval dtype) are
compared in both modes.
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

from math import exp, lgamma, log, sqrt

import numpy as np
import pytest

import jqmc_rs

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance
from jqmc.atomic_orbital import (
    AOs_cart_gto_data,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
    compute_AOs_value_grad_lap,
)
from jqmc.structure import Structure_data

_D_SHELL = [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]

#: case -> (components, exponent, coefficient, index of the probed AO, standalone value compared)
_CASES = {
    "lone_dxy": ([(1, 1, 0)], 0.25, 0.7, 0, True),
    "lone_210": ([(2, 1, 0)], 2.5, 1.0, 0, True),
    "p_shell_plus_lone_dxy": ([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)], 0.8, 1.0, 3, True),
    "lone_550": ([(5, 5, 0)], 1.0, 1.0, 0, True),
    "lone_41_41_0": ([(41, 41, 0)], 1.0, 1.0, 0, False),
    "complete_d_shell_control": (_D_SHELL, 0.25, 1.0, 1, True),
}


def _closed_form(n, a, c, r):
    """Value, gradient and Laplacian of c N x^nx y^ny z^nz exp(-a r^2) (no jqmc code)."""
    nx, ny, nz = n
    l_tot = nx + ny + nz
    ln_n = -0.5 * sum(lgamma(k + 0.5) - (k + 0.5) * log(2.0 * a) for k in n)
    x = [float(t) for t in r]
    r2 = sum(t * t for t in x)

    def mono(m):
        out = 1.0
        for xi, mi in zip(x, m):
            out *= xi**mi
        return out

    poly = mono(n)
    dpoly = []
    lap_poly = 0.0
    for i in range(3):
        m = list(n)
        if m[i] >= 1:
            k = m[i]
            m[i] -= 1
            dpoly.append(k * mono(m))
        else:
            dpoly.append(0.0)
        m = list(n)
        if m[i] >= 2:
            k = m[i]
            m[i] -= 2
            lap_poly += k * (k - 1) * mono(m)
    env = c * exp(ln_n - a * r2)
    val = env * poly
    grad = [env * (dpoly[i] - 2.0 * a * x[i] * poly) for i in range(3)]
    lap = env * (lap_poly - (4.0 * a * l_tot + 6.0 * a) * poly + 4.0 * a * a * r2 * poly)
    return val, grad, lap


def _points(n, a):
    """Radial maximum of |x^n e^{-a r^2}| along n, plus two nearby off-axis points."""
    l_tot = sum(n)
    rr = sqrt(max(l_tot, 1) / (2.0 * a))
    u = np.asarray(n, dtype=np.float64) + np.array([0.0, 0.0, 0.05])
    u /= np.linalg.norm(u)
    v = np.asarray(n, dtype=np.float64) + np.array([0.15, -0.1, 0.2]) * max(n)
    v /= np.linalg.norm(v)
    return np.stack([rr * u, 0.9 * rr * v, 1.1 * rr * u + np.array([0.0, 0.0, 0.1 * rr])])


def _build(comps, a, c):
    structure = Structure_data(
        pbc_flag=False,
        positions=np.zeros((1, 3)),
        atomic_numbers=(1,),
        element_symbols=("H",),
        atomic_labels=("H",),
    )
    k = len(comps)
    return AOs_cart_gto_data.build(
        structure_data=structure,
        nucleus_index=(0,) * k,
        num_ao=k,
        num_ao_prim=k,
        orbital_indices=tuple(range(k)),
        exponents=np.full(k, a, dtype=np.float64),
        coefficients=np.full(k, c, dtype=np.float64),
        angular_momentums=tuple(sum(n) for n in comps),
        polynominal_order_x=tuple(n[0] for n in comps),
        polynominal_order_y=tuple(n[1] for n in comps),
        polynominal_order_z=tuple(n[2] for n in comps),
    )


@pytest.mark.light
@pytest.mark.parametrize("case", list(_CASES))
def test_incomplete_cartesian_shell_fields_match_closed_form_normalization(case):
    """XLA and Rust Cartesian AO fields of incomplete shells equal the closed-form normalized Gaussian.

    Each case builds its AO set with ``AOs_cart_gto_data.build`` and evaluates it at the
    radial maximum of the probed component and two nearby points. The XLA value,
    gradient, Laplacian and fused value/gradient/Laplacian and the Rust value,
    gradient and Laplacian must equal the lgamma closed form within the strict
    ao_eval / ao_grad_lap tolerances of the active precision mode. A regression
    scales the XLA fields of every incomplete shell by (8a)^((1 - l)/2).
    """
    comps, a, c, probe, check_value = _CASES[case]
    aos = _build(comps, a, c)
    r = _points(comps[probe], a)

    ref_val = np.zeros((len(comps), r.shape[0]))
    ref_grad = np.zeros((3, len(comps), r.shape[0]))
    ref_lap = np.zeros((len(comps), r.shape[0]))
    for i, n in enumerate(comps):
        for e in range(r.shape[0]):
            v, g, lp = _closed_form(n, a, c, r[e])
            ref_val[i, e] = v
            ref_grad[:, i, e] = g
            ref_lap[i, e] = lp

    # The probed component is O(1) at its radial maximum: a constant normalization
    # factor cannot hide under the absolute tolerance.
    assert np.all(np.isfinite(ref_val)) and np.all(np.isfinite(ref_lap))
    assert abs(ref_val[probe, 0]) > 1.0e-2

    atol_v, rtol_v = get_tolerance("ao_eval", "strict")
    atol_g, rtol_g = get_tolerance("ao_grad_lap", "strict")

    def close(actual, expected, atol, rtol, what):
        actual = np.asarray(actual, dtype=np.float64)
        assert actual.shape == expected.shape, what
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=what)

    xla_vgl = compute_AOs_value_grad_lap(aos, r)
    fields = {
        "XLA fused value": (xla_vgl[0], ref_val, atol_v, rtol_v),
        "XLA grad": (np.stack([np.asarray(t) for t in compute_AOs_grad(aos, r)]), ref_grad, atol_g, rtol_g),
        "XLA laplacian": (compute_AOs_laplacian(aos, r), ref_lap, atol_g, rtol_g),
        "XLA fused grad": (np.stack([np.asarray(t) for t in xla_vgl[1:4]]), ref_grad, atol_g, rtol_g),
        "XLA fused laplacian": (xla_vgl[4], ref_lap, atol_g, rtol_g),
        "Rust grad": (np.stack([np.asarray(t) for t in jqmc_rs.compute_AOs_grad(aos, r)]), ref_grad, atol_g, rtol_g),
        "Rust laplacian": (jqmc_rs.compute_AOs_laplacian(aos, r), ref_lap, atol_g, rtol_g),
    }
    if check_value:
        fields["XLA value"] = (compute_AOs(aos, r), ref_val, atol_v, rtol_v)
        fields["Rust value"] = (jqmc_rs.compute_AOs(aos, r), ref_val, atol_v, rtol_v)

    for what, (actual, expected, atol, rtol) in fields.items():
        close(actual, expected, atol, rtol, what)
