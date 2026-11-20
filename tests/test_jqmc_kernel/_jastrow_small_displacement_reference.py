"""Closed-form small-displacement references for the J1 / J2 Jastrow kernels.

Shared by the XLA node ``tests/test_jqmc_kernel/test_jastrow_small_displacement.py``
and the Rust node ``tests/test_rs_kernel/test_jastrow_small_displacement.py``.

Fixture: one hydrogen nucleus (Z_eff = 1) at ``(5, 5, 5)`` Bohr, either
molecular or in a cubic 10 Bohr cell. For J1 a single electron sits at
``(5 + h, 5, 5)``; for J2 one up electron sits at ``(5, 5, 5)`` and one down
electron at ``(5 + h, 5, 5)``. The large common coordinate and the tiny
separation ``h`` make any rounding of the absolute coordinates to the
evaluation precision *before* ``r - R`` is formed visible (fp32 cannot
represent ``5 + 2**-26``), and they expose subtraction cancellation in the
exp forms (``1 - exp(-x)`` for ``x ~ h``).

The references below are written directly from the documented functional
forms with NumPy ``expm1``; they share no code with ``jqmc`` beyond the
zone dtype lookup ``jqmc._precision.get_dtype_np``. Per the same-precision
reference rule (mixed-precision output is never judged against an fp64 reference) each
piece is evaluated in the dtype of the production zone that computes it:
the displacement ``h`` is supplied already reconstructed in fp64 (the
exact ``r - R`` / ``r_i - r_j`` difference, as production must form it),
cast to the zone dtype, and the distance, metric and radial closed forms
are then evaluated in that dtype -- ``jastrow_eval`` for the value,
``jastrow_grad_lap`` for the gradient and Laplacian. The final ``/ h`` and
``* h`` scalings are applied in fp64, as the tests do to the production
output. In full precision every zone is fp64 and the reference equals the
fp64 closed form bit for bit. Because only the *difference* is
cast, the mixed reference keeps the full separation: production that
rounds the absolute coordinates to fp32 before subtracting still collapses
or shifts the separation and fails the comparison.
Parameters are dyadic so that the mode's rounded parameters equal the
exact values (the fp32 rounding of ``(2 Z_eff) ** (1/4)`` in the mixed
lane is ~3e-8 relative, far below every strict window).

The distance used for a displacement ``h`` along x is the Cartesian
``h`` (molecular) or the Williamson smooth periodic distance
``d = (L / pi) sin(pi h / L)`` (cubic cell of side ``L``). For PBC the
metric derivatives along the displacement are ``d'(h) = cos(pi h / L)``
and ``nabla^2 d = -(pi / L) sin(pi h / L) + 2 / d`` (the two transverse
directions contribute ``1 / d`` each at zero transverse displacement).
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

from jqmc._precision import get_dtype_np

NUCLEUS = np.array([5.0, 5.0, 5.0], dtype=np.float64)
CUBIC_SIDE = 10.0
# 2**-16 and 2**-26 are the small-displacement acceptance points; 2**-38
# (still above EPS_safe_distance = 1e-12) additionally resolves the fp64
# ``1 - exp(-x)`` cancellation in the full lane.
SMALL_DISPLACEMENTS = (2.0**-16, 2.0**-26, 2.0**-38)
# Separations whose exponent y = a * c * r (J1) or a * r (J2) straddles the
# branch points of the Rust ``one_minus_exp_neg`` helpers (|y| = 1/4 for the
# f32 Taylor branch, y = ln 2 for the f64 ``exp_m1`` branch) for the
# parameters below.
SWITCH_SEPARATIONS = (0.125, 0.2, 0.3, 0.45, 0.6, 0.9, 1.2)
J1_PARAMS = (0.75, 1.5)
J2_PARAMS = (0.75, 1.5)


def _distance_and_metric_in(h: float, periodic: bool, dtype: type) -> tuple:
    """Return ``(d, d'(h), nabla^2 d)`` for the fp64 displacement ``h`` cast to ``dtype``, evaluated in ``dtype``."""
    h = dtype(h)
    one, two = dtype(1.0), dtype(2.0)
    if not periodic:
        return h, one, two / h
    k = dtype(np.pi / CUBIC_SIDE)
    d = np.sin(k * h) / k
    return d, np.cos(k * h), -k * np.sin(k * h) + two / d


def distance_and_metric(h: float, periodic: bool) -> tuple[float, float, float]:
    """Return ``(d, d'(h), nabla^2 d)`` for a displacement ``h`` along x (fp64)."""
    d, dp, lap_d = _distance_and_metric_in(h, periodic, np.float64)
    return float(d), float(dp), float(lap_d)


def j1_radial(j1b_type: str, a: float, d, dtype: type = np.float64) -> tuple:
    """Return ``(f, f', f'')`` of the Z_eff = 1 one-body term at distance ``d``, evaluated in ``dtype``."""
    two = dtype(2.0)
    z = dtype(1.0)
    a, d = dtype(a), dtype(d)
    c = (two * z) ** dtype(0.25)
    if j1b_type == "exp":
        big_a = (two * z) ** dtype(0.75)
        x = a * c * d
        f = -big_a / (two * a) * (-np.expm1(-x))
        fp = -big_a * c / two * np.exp(-x)
        fpp = big_a * a * c * c / two * np.exp(-x)
    elif j1b_type == "pade":
        den = dtype(1.0) + a * c * d
        f = -z * d / den
        fp = -z / (den * den)
        fpp = two * z * a * c / (den * den * den)
    else:
        raise ValueError(j1b_type)
    return f, fp, fpp


def j2_radial(j2b_type: str, a: float, d, dtype: type = np.float64) -> tuple:
    """Return ``(f, f', f'')`` of the two-body pair term at distance ``d``, evaluated in ``dtype``."""
    half, two = dtype(0.5), dtype(2.0)
    a, d = dtype(a), dtype(d)
    if j2b_type == "exp":
        f = (-np.expm1(-a * d)) / (two * a)
        fp = half * np.exp(-a * d)
        fpp = -half * a * np.exp(-a * d)
    elif j2b_type == "pade":
        den = dtype(1.0) + a * d
        f = d / (two * den)
        fp = half / (den * den)
        fpp = -a / (den * den * den)
    else:
        raise ValueError(j2b_type)
    return f, fp, fpp


def closed_form(kind: str, form: str, a: float, h: float, periodic: bool) -> dict[str, float]:
    """Zone-dtype closed-form ``J/h``, x-gradient of the moved electron and ``h * lap``.

    ``kind`` is ``"J1"`` (electron at nucleus + h x) or ``"J2"`` (down
    electron at up electron + h x). ``h`` is the separation reconstructed in
    fp64 from the absolute coordinates. The value is evaluated in the active
    ``jastrow_eval`` dtype and the gradient / Laplacian in the active
    ``jastrow_grad_lap`` dtype (fp32 in mixed, fp64 in full), each from
    ``h`` cast to that dtype; the ``/ h`` and ``* h`` scalings are fp64.
    The gradient/Laplacian refer to the electron displaced by ``+h`` (the
    J1 electron, the J2 down electron); for J2 the partner's gradient is
    the negative and its Laplacian equal.
    """
    radial = j1_radial if kind == "J1" else j2_radial
    value_dtype = get_dtype_np("jastrow_eval")
    d, _, _ = _distance_and_metric_in(h, periodic, value_dtype)
    f, _, _ = radial(form, a, d, value_dtype)
    deriv_dtype = get_dtype_np("jastrow_grad_lap")
    d, dp, lap_d = _distance_and_metric_in(h, periodic, deriv_dtype)
    _, fp, fpp = radial(form, a, d, deriv_dtype)
    return {
        "value_over_h": float(f) / h,
        "grad_x": float(fp * dp),
        "lap_times_h": float(fpp * dp * dp + fp * lap_d) * h,
    }
