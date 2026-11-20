"""Range of the mixed-precision Cartesian AO value normalization, XLA and Rust (issue #111).

The standalone Cartesian AO value kernels (XLA ``AOs_cart_gto_data._compute_AOs`` and the
Rust ``compute_AOs`` f32 kernel used by the mixed MCMC value path) scale every primitive by

    N = sqrt((2a/pi)^(3/2) (8a)^l prod_i n_i! / (2 n_i)!),

a constant of the basis that does not depend on the electron coordinates. Assembled from
float32 pieces in mixed precision, ``(8a)^l`` overflows float32 for large exponents or
high l, and the factorial ratio underflows for high l, so the AO value comes out as Inf, 0
or NaN (or with a large error from subnormals) although N itself and the AO value are
representable in float32. N is therefore formed in float64 and rounded only once, as in the
fused value/grad/lap kernel.

This module evaluates complete Cartesian shells of degree 2..8 at exponents just outside the
old mixed-precision window (and l = 8 at a very diffuse exponent) and single high-degree axis
components (l = 12 at a = 1e-3, l = 13 at a = 1e2, l = 25..28 at a = 1), at the radial maximum
of |x^l exp(-a r^2)| and on the (1,1,1) diagonal at the same radius, where the exact values are
O(1e-4) .. O(1e3). The XLA and Rust standalone values must be finite and equal the closed form

    c N x^nx y^ny z^nz exp(-a r^2),  N^-2 = prod_i Gamma(n_i + 1/2) / (2a)^(n_i + 1/2),

evaluated with ``math.lgamma`` and plain Python (no jqmc code), within the strict ao_eval
tolerance of the active precision mode. In full precision the kernels are unchanged and the
same comparison holds.
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
from jqmc.atomic_orbital import AOs_cart_gto_data, compute_AOs
from jqmc.structure import Structure_data


def _shell(l_tot):
    return [(nx, ny, l_tot - nx - ny) for nx in range(l_tot, -1, -1) for ny in range(l_tot - nx, -1, -1)]


#: case -> (components, exponent). Complete shells at exponents just outside the old
#: mixed-precision window (old upper bounds: l=2 3.2e10, 3 1e8, 4 1.8e6, 5 1.8e5, 6 1.8e4,
#: 7 5.6e3, 8 1.8e3; old XLA lower bound for l=8 1.8e-4) and single axis components whose
#: degree exceeded the old smallest failing degree (a=1e-3: 11, a=1e2: 13, a=1: 25).
_CASES = {
    "shell_l2_a1e11": (_shell(2), 1.0e11),
    "shell_l3_a1e9": (_shell(3), 1.0e9),
    "shell_l4_a1e7": (_shell(4), 1.0e7),
    "shell_l5_a1e6": (_shell(5), 1.0e6),
    "shell_l6_a1e5": (_shell(6), 1.0e5),
    "shell_l7_a1e4": (_shell(7), 1.0e4),
    "shell_l8_a1e4": (_shell(8), 1.0e4),
    "shell_l8_a1e-4": (_shell(8), 1.0e-4),
    "axis_l12_a1e-3": ([(12, 0, 0)], 1.0e-3),
    "axis_l13_a1e2": ([(13, 0, 0)], 1.0e2),
    "axis_l25_a1": ([(25, 0, 0)], 1.0),
    "axis_l26_a1": ([(26, 0, 0)], 1.0),
    "axis_l27_a1": ([(27, 0, 0)], 1.0),
    "axis_l28_a1": ([(28, 0, 0)], 1.0),
}


def _closed_form_value(n, a, r):
    """c N x^nx y^ny z^nz exp(-a r^2) with c = 1 (no jqmc code)."""
    ln_n = -0.5 * sum(lgamma(k + 0.5) - (k + 0.5) * log(2.0 * a) for k in n)
    x = [float(t) for t in r]
    val = exp(ln_n - a * sum(t * t for t in x))
    for xi, k in zip(x, n):
        val *= xi**k
    return val


def _build(comps, a):
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
        coefficients=np.ones(k, dtype=np.float64),
        angular_momentums=tuple(sum(n) for n in comps),
        polynominal_order_x=tuple(n[0] for n in comps),
        polynominal_order_y=tuple(n[1] for n in comps),
        polynominal_order_z=tuple(n[2] for n in comps),
    )


@pytest.mark.light
@pytest.mark.parametrize("case", list(_CASES))
def test_cartesian_ao_value_normalization_keeps_fp32_range(case):
    """XLA and Rust standalone Cartesian AO values stay finite and equal the closed form at high l / extreme exponents.

    Each case builds a complete Cartesian shell or a single axis component with
    ``AOs_cart_gto_data.build`` and evaluates the XLA ``compute_AOs`` and the Rust
    ``jqmc_rs.compute_AOs`` at the radial maximum and on the (1,1,1) diagonal at the same
    radius. Both must be finite and equal the lgamma closed form within the strict ao_eval
    tolerance of the active mode. A normalization assembled from float32 pieces gives mixed-precision values that are
    Inf, 0, NaN or off by more than the tolerance.
    """
    comps, a = _CASES[case]
    l_tot = sum(comps[0])
    rr = sqrt(l_tot / (2.0 * a))
    r = np.array([[rr, 0.2 * rr, 0.1 * rr], [rr / sqrt(3.0)] * 3])
    expected = np.array([[_closed_form_value(n, a, p) for p in r] for n in comps])

    atol, rtol = get_tolerance("ao_eval", "strict")
    # The reference is O(1e-4) .. O(1e3) at these points for the probed components, well
    # above the absolute tolerance: a non-finite, zero or rescaled kernel value cannot hide.
    assert np.all(np.isfinite(expected))
    assert np.max(np.abs(expected)) > atol
    aos = _build(comps, a)
    for what, actual in (
        ("XLA compute_AOs", compute_AOs(aos, r)),
        ("Rust compute_AOs", jqmc_rs.compute_AOs(aos, r)),
    ):
        actual = np.asarray(actual, dtype=np.float64)
        assert actual.shape == expected.shape, what
        assert np.all(np.isfinite(actual)), what
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=what)
