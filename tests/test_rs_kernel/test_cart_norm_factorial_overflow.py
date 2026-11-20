"""Cartesian GTO normalization factorial ratio without int64 overflow, XLA and Rust (issue #148).

A Cartesian primitive is ``c N x^nx y^ny z^nz exp(-a r^2)`` with the closed-form
normalization

    N^-2 = prod_i Gamma(n_i + 1/2) / (2 a)^(n_i + 1/2),

equivalently ``N^2 = (2a/pi)^(3/2) (8a)^l prod_i n_i! / (2 n_i)!``. The XLA, NumPy
overlap and basis-derivative paths read ``prod_i n_i! / (2 n_i)!`` from
``AOs_cart_gto_data._normalization_factorial_ratio_prim_jnp``. Multiplying
``scipy.special.factorial(..., exact=True)`` arrays, which are int64 whenever every
argument of the array is <= 20, wraps the products ``(2nx)! (2ny)! (2nz)!`` and
``nx! ny! nz!`` around int64 for lone or incomplete high-degree components: lone
(7,6,0) comes out 2.93 times too large, lone (7,7,0), (5,5,5) and (8,7,0) are NaN,
and the result depends on the other components of the leaf (a (13,0,0) companion
promotes the x factors to exact Python integers and hides the overflow, a (0,0,13)
companion does not). The Rust kernels compute the ratio in f64.

This module builds such leaves through the public ``AOs_cart_gto_data.build`` factory
and compares

* XLA ``compute_AOs``, ``compute_AOs_grad``, ``compute_AOs_laplacian`` and the fused
  ``compute_AOs_value_grad_lap``;
* the basis-derivative evaluator ``jqmc.derivatives.ao_deriv`` (value, gradient,
  Laplacian);
* Rust ``jqmc_rs.compute_AOs``, ``compute_AOs_grad`` and ``compute_AOs_laplacian``

with the closed form above, evaluated with ``math.lgamma`` and plain Python (no jqmc
code). The diagonal of the analytic ``compute_overlap_matrix`` of a one-primitive AO
must equal ``c^2``, both exactly (closed form) and as a tensor Gauss-Hermite quadrature
of the squared Rust ``compute_AOs`` values (exact for these polynomial degrees).
Controls whose ratio was already correct ((6,6,0), (13,1,0), (7,6,0) with a (13,0,0)
companion) guard the leaf independence of the fixed ratio.
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

from jqmc._precision import get_tolerance, get_tolerance_min
from jqmc.atomic_orbital import (
    AOs_cart_gto_data,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
    compute_AOs_value_grad_lap,
    compute_overlap_matrix,
)
from jqmc.derivatives.ao_deriv import ao_deriv, ao_grad, ao_laplacian
from jqmc.structure import Structure_data

#: case -> (components, exponent, coefficient, index of the probed AO)
_CASES = {
    # (2nx)! (2ny)! (2nz)! wrapped around int64 (every power of the leaf <= 10).
    "lone_7_6_0": ([(7, 6, 0)], 1.0, 1.0, 0),
    "lone_7_7_0": ([(7, 7, 0)], 1.0, 1.0, 0),
    "lone_5_5_5": ([(5, 5, 5)], 1.0, 1.0, 0),
    "lone_8_7_0": ([(8, 7, 0)], 1.0, 1.0, 0),
    "lone_10_2_0": ([(10, 2, 0)], 0.7, 0.8, 0),
    "lone_10_1_1": ([(10, 1, 1)], 1.3, 1.0, 0),
    # nx! ny! nz! wrapped around int64 (every power of the leaf <= 20).
    "lone_20_3_0": ([(20, 3, 0)], 1.0, 1.0, 0),
    "lone_13_13_0": ([(13, 13, 0)], 1.0, 1.0, 0),
    # A companion with a large z power does not promote the x and y factors.
    "pair_7_6_0_with_0_0_13": ([(7, 6, 0), (0, 0, 13)], 1.0, 1.0, 0),
    # Controls: no int64 overflow even in an int64 product.
    "control_lone_6_6_0": ([(6, 6, 0)], 1.0, 1.0, 0),
    "control_lone_13_1_0": ([(13, 1, 0)], 1.0, 1.0, 0),
    "control_pair_7_6_0_with_13_0_0": ([(7, 6, 0), (13, 0, 0)], 1.0, 1.0, 0),
}


def _closed_form(n, a, c, r):
    """Value, gradient and Laplacian of c N x^nx y^ny z^nz exp(-a r^2) (no jqmc code)."""
    l_tot = sum(n)
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
    rr = sqrt(max(sum(n), 1) / (2.0 * a))
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


def _gauss_hermite_overlap_diag(values_fn, a, n_nodes):
    """Diagonal of int phi_mu^2 d^3r by tensor Gauss-Hermite quadrature of AO values.

    With nodes/weights for the weight exp(-t^2) and x = t / sqrt(2a),
    int f(x) exp(-2a x^2) dx = sum_k w_k f(x_k) / sqrt(2a); phi^2 = f exp(-2a r^2) with f a
    polynomial of per-axis degree 2 n_i, integrated exactly when 2 n_nodes - 1 >= 2 max n_i.
    """
    t, w = np.polynomial.hermite.hermgauss(n_nodes)
    x = t / sqrt(2.0 * a)
    gx, gy, gz = np.meshgrid(x, x, x, indexing="ij")
    wx, wy, wz = np.meshgrid(w, w, w, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    wts = (wx * wy * wz).ravel() / (2.0 * a) ** 1.5
    r2 = np.sum(pts * pts, axis=1)
    phi = np.asarray(values_fn(pts), dtype=np.float64)
    return np.sum(phi * phi * (np.exp(2.0 * a * r2) * wts)[None, :], axis=1)


@pytest.mark.light
@pytest.mark.parametrize("case", list(_CASES))
def test_cartesian_normalization_factorial_ratio_has_no_int64_overflow(case):
    """XLA, basis-derivative, Rust fields and the overlap diagonal of lone high-degree Cartesian components are exact.

    Each case builds its leaf with ``AOs_cart_gto_data.build``; the stored factorial ratio
    must equal ``prod_i n_i! / (2 n_i)!`` (lgamma), the XLA, ao_deriv and Rust value,
    gradient and Laplacian must equal the lgamma closed form within the strict ao_eval /
    ao_grad_lap tolerances, and the analytic overlap diagonal must equal ``c^2`` within the
    strict det_eval tolerance and agree with a Gauss-Hermite quadrature of the Rust values.
    int64 factorial products wrap around here, giving a wrong scale or NaN.
    """
    comps, a, c, probe = _CASES[case]
    aos = _build(comps, a, c)
    r = _points(comps[probe], a)

    atol_v, rtol_v = get_tolerance("ao_eval", "strict")
    atol_g, rtol_g = get_tolerance("ao_grad_lap", "strict")
    atol_s, rtol_s = get_tolerance_min(("det_eval",), "strict")

    def close(actual, expected, atol, rtol, what):
        actual = np.asarray(actual, dtype=np.float64)
        assert actual.shape == expected.shape, what
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=what)

    # The stored ratio itself, per component, against lgamma (fp64 storage).
    ref_ratio = np.array([exp(sum(lgamma(k + 1) - lgamma(2 * k + 1) for k in n)) for n in comps])
    close(aos._normalization_factorial_ratio_prim_jnp, ref_ratio, 0.0, rtol_s, "stored factorial ratio")

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
    assert abs(ref_val[probe, 0]) > 1.0e-1

    xla_vgl = compute_AOs_value_grad_lap(aos, r)
    fields = {
        "XLA value": (compute_AOs(aos, r), ref_val, atol_v, rtol_v),
        "XLA fused value": (xla_vgl[0], ref_val, atol_v, rtol_v),
        "XLA grad": (np.stack([np.asarray(t) for t in compute_AOs_grad(aos, r)]), ref_grad, atol_g, rtol_g),
        "XLA laplacian": (compute_AOs_laplacian(aos, r), ref_lap, atol_g, rtol_g),
        "XLA fused grad": (np.stack([np.asarray(t) for t in xla_vgl[1:4]]), ref_grad, atol_g, rtol_g),
        "XLA fused laplacian": (xla_vgl[4], ref_lap, atol_g, rtol_g),
        "ao_deriv value": (ao_deriv(aos, r, (0, 0, 0)), ref_val, atol_g, rtol_g),
        "ao_deriv grad": (ao_grad(aos, r), ref_grad, atol_g, rtol_g),
        "ao_deriv laplacian": (ao_laplacian(aos, r), ref_lap, atol_g, rtol_g),
        "Rust value": (jqmc_rs.compute_AOs(aos, r), ref_val, atol_v, rtol_v),
        "Rust grad": (np.stack([np.asarray(t) for t in jqmc_rs.compute_AOs_grad(aos, r)]), ref_grad, atol_g, rtol_g),
        "Rust laplacian": (jqmc_rs.compute_AOs_laplacian(aos, r), ref_lap, atol_g, rtol_g),
    }
    for what, (actual, expected, atol, rtol) in fields.items():
        close(actual, expected, atol, rtol, what)

    # Analytic overlap diagonal of one-primitive AOs: exactly c^2 ...
    s_diag = np.diag(np.asarray(compute_overlap_matrix(aos), dtype=np.float64))
    close(s_diag, np.full(len(comps), c * c), atol_s, rtol_s, "XLA overlap diagonal vs closed form")
    # ... and equal to the quadrature of the squared Rust AO values.
    max_power = max(max(n) for n in comps)
    n_nodes = max_power + 2  # 2 n_nodes - 1 >= 2 max_power + 3
    s_rust = _gauss_hermite_overlap_diag(lambda p: jqmc_rs.compute_AOs(aos, p), a, n_nodes)
    close(s_rust, np.full(len(comps), c * c), atol_v, rtol_v, "Rust quadrature overlap diagonal vs closed form")
    close(s_diag, s_rust, atol_v, rtol_v, "XLA overlap diagonal vs Rust quadrature")
