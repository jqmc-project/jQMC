"""FAST-mode AO derivative screen stays absolute for accepted high angular momentum (issue #87, PR #97).

The screened FAST AO grad/lap kernels drop a shell's derivative rows for an
electron at ``pz_deriv * r^2 > LEPSBAS``; ``derivative_cut_exponent`` in
``jqmc/rs/src/atomic_orbital.rs`` picks ``pz_deriv`` as the first ``r^2`` at
which an analytic bound on every gradient component and Laplacian is below
``exp(-LEPSBAS)``. That first crossing is a valid cut for all larger ``r^2``
only where the bound decreases monotonically. The Gaussian-polynomial factor
``r^(l+2) exp(-zeta r^2)`` rises up to ``r^2 = (l+2) / (2 zeta)``, so for
``l >= 40`` the bound can dip below the threshold at the LEPSBAS starting point
``zeta_min r^2 = LEPSBAS = 20.72`` and rise above it farther out. Cartesian
construction and the native molecular FAST gates accept such shells.

Counterexample pinned here (from PR #97): one Cartesian AO at the
origin with powers ``(82, 0, 0)``, exponent 1 and contraction coefficient
``4.01499518920269e-9``. Its derivative bound at ``r^2 = 20.72`` is ``8.0e-10``,
below ``exp(-20.72) = 1.0e-9``, yet the exact Laplacian on the x axis at
``r^2 = 41`` is ``-1.93e-8``.

Reference: an independent closed form, not the XLA AO code. For
``phi = c N x^n exp(-zeta r^2)`` with ``N^2 = (2 zeta / pi)^(3/2) (4 zeta)^n /
(2n-1)!!`` (separable Gaussian integrals),
``d phi/dx = phi (n/x - 2 zeta x)``, ``d phi/dy = -2 zeta y phi``,
``d phi/dz = -2 zeta z phi`` and
``lap phi = phi (n(n-1)/x^2 - 2 zeta (2n+3) + 4 zeta^2 r^2)``. ``c`` is the
coefficient as stored in the built AO leaf (rounded in mixed mode), so the
reference describes the AO that both backends actually evaluate.

All three screened kernels (``batch``, ``single``, ``folded``) are reached
through the test-only binding ``jqmc_rs.compute_AOs_grad_lap_fast_cut`` and must
equal the closed form within the strict XLA-Rust consistency pair of the
``ao_grad_lap`` zone (fp64 in both precision modes). Points lie on the x axis
and on two nearby off-axis rays with ``r^2`` from 10 to 90, covering the
within-cut control region and the region beyond the LEPSBAS cut where the exact
derivatives exceed that envelope (non-vacuity is asserted).
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

import math

import numpy as np
import pytest

import jqmc_rs

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.atomic_orbital import AOs_cart_gto_data
from jqmc.structure import Structure_data

_ZONES_GRAD_LAP = ("ao_grad_lap",)
_ROUTES = ("batch", "single", "folded")
# Counterexample: Cartesian x^82 primitive, exponent 1.
_POWER = 82
_ZETA = 1.0
_COEF = 4.01499518920269e-9
# Normalization of that primitive as computed independently in PR #97.
_AUDIT_NORM = 2.8820304426801676e-49
# LEPSBAS derivative-screen starting point zeta_min * r^2 = LEPSBAS.
_FORMER_CUT_ZETA_R2 = 20.72
# r^2 samples: within the LEPSBAS cut (10, 15, 20) and beyond it, including the
# counterexample points 41 and 43.
_R2 = (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 41.0, 43.0, 47.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0)
# Small polar tilts away from the x axis (radians), toward +y and toward -z.
_TILTS = ((0.0, 0.0), (0.08, 0.0), (0.0, -0.05))


def _build_aos():
    structure = Structure_data(
        pbc_flag=False,
        positions=np.zeros((1, 3), dtype=np.float64),
        atomic_numbers=(18,),
        element_symbols=("Ar",),
        atomic_labels=("Ar",),
    )
    return AOs_cart_gto_data.build(
        structure_data=structure,
        nucleus_index=[0],
        num_ao=1,
        num_ao_prim=1,
        orbital_indices=[0],
        exponents=np.asarray([_ZETA], dtype=np.float64),
        coefficients=np.asarray([_COEF], dtype=np.float64),
        angular_momentums=[_POWER],
        polynominal_order_x=[_POWER],
        polynominal_order_y=[0],
        polynominal_order_z=[0],
    )


def _points():
    pts = []
    for r2 in _R2:
        r = math.sqrt(r2)
        for ty, tz in _TILTS:
            d = np.asarray([math.cos(ty) * math.cos(tz), math.sin(ty), math.sin(tz)])
            pts.append(r * d / np.linalg.norm(d))
    return np.ascontiguousarray(np.asarray(pts), dtype=np.float64)


def _closed_form(r, coef):
    """Independent value/gradient/Laplacian of c N x^n exp(-zeta r^2)."""
    n, z = _POWER, _ZETA
    log_double_factorial = math.lgamma(2 * n + 1) - n * math.log(2.0) - math.lgamma(n + 1)  # log (2n-1)!!
    norm = math.exp(0.5 * (1.5 * math.log(2.0 * z / math.pi) + n * math.log(4.0 * z) - log_double_factorial))
    x, y, zc = r[:, 0], r[:, 1], r[:, 2]
    r2 = np.einsum("ij,ij->i", r, r)
    phi = coef * norm * x**n * np.exp(-z * r2)
    gx = phi * (n / x - 2.0 * z * x)
    gy = -2.0 * z * y * phi
    gz = -2.0 * z * zc * phi
    lap = phi * (n * (n - 1) / x**2 - 2.0 * z * (2 * n + 3) + 4.0 * z * z * r2)
    return norm, (gx[None, :], gy[None, :], gz[None, :], lap[None, :])


@pytest.mark.light
def test_fast_cut_ao_derivative_screen_absolute_for_high_angular_momentum():
    """Screened FAST AO grad/lap of an accepted l=82 Cartesian AO equal its closed-form derivatives beyond the LEPSBAS cut."""
    aos = _build_aos()
    r = _points()
    coef = float(np.asarray(aos.coefficients, dtype=np.float64)[0])
    norm, exact = _closed_form(r, coef)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD_LAP, level="strict")
    # Cross-check of the normalization against the independent value.
    assert abs(norm / _AUDIT_NORM - 1.0) <= rtol, norm
    for name, ref in zip(("gx", "gy", "gz", "lap"), exact, strict=True):
        assert np.all(np.isfinite(ref)), f"closed-form {name} not finite"

    # Non-vacuity: beyond the LEPSBAS cut the exact derivatives exceed the
    # comparison envelope, so dropping them is rejected.
    beyond = _ZETA * np.einsum("ij,ij->i", r, r) > _FORMER_CUT_ZETA_R2
    excess = max(float(np.max(np.abs(ref[:, beyond]) / (atol + rtol * np.abs(ref[:, beyond])))) for ref in exact)
    assert excess > 1.0, f"derivatives beyond the LEPSBAS cut stay inside the envelope ({excess})"

    for route in _ROUTES:
        got = [np.asarray(x) for x in jqmc_rs.compute_AOs_grad_lap_fast_cut(aos, r, route)]
        for name, val, ref in zip(("gx", "gy", "gz", "lap"), got, exact, strict=True):
            label = f"l=82 cart {route}: {name} FAST screened vs closed form"
            assert val.shape == ref.shape, label
            assert np.all(np.isfinite(val)), f"{label}: non-finite"
            np.testing.assert_allclose(val, ref, atol=atol, rtol=rtol, err_msg=label)
