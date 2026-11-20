"""Closed-form small-separation references for the J1 / J2 parameter derivatives.

Shared by ``tests/test_jqmc_kernel/test_jastrow_param_derivative_small_separation.py``
(XLA) and ``tests/test_rs_kernel/test_jastrow_param_derivative_small_separation.py``
(Rust). Same fixture as ``_jastrow_small_displacement_reference.py``:
one hydrogen nucleus (Z_eff = 1) or the up electron at the large common
coordinate ``(5, 5, 5)`` Bohr, the partner electron at ``+h`` along x,
``h in {2**-16, 2**-26, 2**-38}``, molecular or in a cubic 10 Bohr cell.

Exp-form parameter derivative. With ``x = a c d`` (J1, ``c = (2 Z_eff)**(1/4)``)
or ``x = a d`` (J2)::

    dJ1/da = (2 Z_eff)**(3/4) / (2 a**2) * q(x)
    dJ2/da = -q(x) / (2 a**2)
    q(x)   = 1 - (1 + x) exp(-x) = x**2 / 2 - x**3 / 3 + ...

``q`` is O(x**2) although both of its terms are O(x): even the fp64 NumPy
rewriting ``-expm1(-x) - x * exp(-x)`` keeps a relative error of order
``eps / x`` (the rounding of ``exp(-x)`` near 1 is multiplied by ``x`` and
compared with ``x**2 / 2``), i.e. up to ~3e-5 at ``x ~ 2**-38`` whenever ``x``
is not dyadic. The reference therefore evaluates the same ``expm1`` form in
60-digit decimal arithmetic (``decimal`` module), which is exact to fp64 and
shares no code with ``jqmc``.

Pade forms (no cancellation), ``x`` as above::

    dJ1/da = Z_eff c d**2 / (1 + x)**2
    dJ2/da = -d**2 / (2 (1 + x)**2)

The checked quotient is ``(dJ/da) / h**2`` (O(1) as h -> 0).

Off-grid anchor. ``OFFGRID_ANCHOR = (5.1, 5.2, 5.3)`` is not representable in
fp32. For a displacement ``h`` the fp64 walker coordinate is
``fl(5.1 + h)`` and the separation seen by any implementation that forms
``r - R`` in fp64 is ``fl(fl(5.1 + h) - 5.1)``, exact by Sterbenz;
:func:`offgrid_separation` returns it so the closed forms are evaluated at
the separation the inputs actually encode. Rounding the anchor (or the
electron) to fp32 BEFORE the difference shifts the separation by up to
``2.4e-7`` Bohr, which is ~1.6% of ``h = 2**-16``.
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

from decimal import Decimal, localcontext

import numpy as np

from tests.test_jqmc_kernel._jastrow_small_displacement_reference import distance_and_metric

OFFGRID_ANCHOR = np.array([5.1, 5.2, 5.3], dtype=np.float64)
_DECIMAL_DIGITS = 60


def offgrid_separation(h: float) -> float:
    """Return the exact fp64 separation ``fl(fl(x0 + h) - x0)`` along x at the off-grid anchor."""
    x0 = float(OFFGRID_ANCHOR[0])
    return float((x0 + h) - x0)


def _q_exact(x: float) -> float:
    """``q(x) = -expm1(-x) - x exp(-x)`` in 60-digit decimal arithmetic, rounded to fp64."""
    with localcontext() as ctx:
        ctx.prec = _DECIMAL_DIGITS
        xd = Decimal(float(x))
        e = (-xd).exp()
        expm1_neg = e - 1
        return float(-expm1_neg - xd * e)


def param_derivative_over_h2(kind: str, form: str, a: float, h: float, periodic: bool) -> float:
    """Closed-form ``(dJ/da) / h**2`` for the J1 (electron at nucleus + h x) or J2 (pair at separation h) fixture."""
    d, _, _ = distance_and_metric(h, periodic)
    if kind == "J1":
        z = 1.0
        c = (2.0 * z) ** 0.25
        x = a * c * d
        if form == "exp":
            deriv = (2.0 * z) ** 0.75 / (2.0 * a * a) * _q_exact(x)
        elif form == "pade":
            deriv = z * c * d * d / ((1.0 + x) * (1.0 + x))
        else:
            raise ValueError(form)
    elif kind == "J2":
        x = a * d
        if form == "exp":
            deriv = -_q_exact(x) / (2.0 * a * a)
        elif form == "pade":
            deriv = -d * d / (2.0 * (1.0 + x) * (1.0 + x))
        else:
            raise ValueError(form)
    else:
        raise ValueError(kind)
    return float(deriv / (h * h))


def j2_pair_value(form: str, a: float, d: float) -> float:
    """Two-body pair term ``u(d)``: exp ``-expm1(-a d) / (2 a)``, Pade ``d / (2 (1 + a d))`` (fp64)."""
    if form == "exp":
        return float(-np.expm1(-a * d) / (2.0 * a))
    if form == "pade":
        return float(d / (2.0 * (1.0 + a * d)))
    raise ValueError(form)


def diagonal_distance(h: float, periodic: bool) -> float:
    """Distance for the displacement ``(h, h, 0)``: ``sqrt(2) h`` or the cubic Williamson ``sqrt(2) (L/pi) sin(pi h/L)``."""
    d_axis, _, _ = distance_and_metric(h, periodic)
    return float(np.sqrt(2.0) * d_axis)


def j2_mesh_delta_over_h(form: str, a: float, h: float, periodic: bool) -> float:
    """One-move J2 value delta ``(u(d_new) - u(d_old)) / h`` for a move from ``+h x`` to ``+h (x + y)``."""
    d_old, _, _ = distance_and_metric(h, periodic)
    d_new = diagonal_distance(h, periodic)
    return float((j2_pair_value(form, a, d_new) - j2_pair_value(form, a, d_old)) / h)
