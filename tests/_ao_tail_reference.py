"""Pure Gaussian tail inputs and an independent closed-form reference, without Rust."""

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

from math import exp, lgamma, log, pi, sqrt

import numpy as np

from jqmc.atomic_orbital import AOs_cart_gto_data, AOs_sphe_gto_data
from jqmc.structure import Structure_data


def gaussian_tail_case(family):
    """Return the unchanged N-centered Cartesian or complete spherical tail witness."""
    if family not in {"cart_l60_a10", "sphe_l6_a1"}:
        raise ValueError(f"Unknown Gaussian tail witness: {family}")
    cart = family == "cart_l60_a10"
    ell, alpha = (60, 10.0) if cart else (6, 1.0)
    radii = np.asarray([sqrt(3.0), 4.7, 20.0, 100.0] if cart else [sqrt(3.0), 1.0e7, 1.0e12])
    structure = Structure_data.build(
        positions=np.zeros((1, 3)), pbc_flag=False, atomic_numbers=(7,), element_symbols=("N",), atomic_labels=("N1",)
    )
    count = 1 if cart else 2 * ell + 1
    kwargs = dict(
        structure_data=structure,
        nucleus_index=(0,) * count,
        num_ao=count,
        num_ao_prim=count,
        orbital_indices=tuple(range(count)),
        exponents=np.full(count, alpha, dtype=np.float64),
        coefficients=np.ones(count, dtype=np.float64),
        angular_momentums=(ell,) * count,
    )
    if cart:
        aos = AOs_cart_gto_data.build(**kwargs, polynominal_order_x=(ell,), polynominal_order_y=(0,), polynominal_order_z=(0,))
        log_normalization = -0.5 * (lgamma(ell + 0.5) + 2 * lgamma(0.5) - (ell + 1.5) * log(2 * alpha))
    else:
        aos = AOs_sphe_gto_data.build(**kwargs, magnetic_quantum_numbers=tuple(range(-ell, ell + 1)))
        log_normalization = 0.5 * (log(2.0) + (ell + 1.5) * log(2 * alpha) - lgamma(ell + 1.5))
        log_normalization += 0.5 * log((2 * ell + 1) / (4 * pi))
    points = np.zeros((len(radii), 3))
    points[:, 0 if cart else 2] = radii
    expected = np.asarray([[exp(log_normalization + ell * log(r) - alpha * r * r) for r in radii]])
    return aos, points, expected


def zero_coefficient_tangent_case(family):
    """Return a single-primitive l=1 Gaussian with an exactly zero coefficient.

    Its radial factor is exactly zero, and the coefficient tangent of the AO is
    the unit-coefficient normalized primitive, given in closed form.
    """
    if family not in {"cart_l1_a1", "sphe_l1_a1"}:
        raise ValueError(f"Unknown zero-coefficient tangent witness: {family}")
    cart = family == "cart_l1_a1"
    ell, alpha = 1, 1.0
    radii = np.asarray([0.5, 1.0, 1.7])
    structure = Structure_data.build(
        positions=np.zeros((1, 3)), pbc_flag=False, atomic_numbers=(7,), element_symbols=("N",), atomic_labels=("N1",)
    )
    count = 1 if cart else 2 * ell + 1
    kwargs = dict(
        structure_data=structure,
        nucleus_index=(0,) * count,
        num_ao=count,
        num_ao_prim=count,
        orbital_indices=tuple(range(count)),
        exponents=np.full(count, alpha, dtype=np.float64),
        coefficients=np.zeros(count, dtype=np.float64),
        angular_momentums=(ell,) * count,
    )
    if cart:
        aos = AOs_cart_gto_data.build(**kwargs, polynominal_order_x=(ell,), polynominal_order_y=(0,), polynominal_order_z=(0,))
        log_normalization = -0.5 * (lgamma(ell + 0.5) + 2 * lgamma(0.5) - (ell + 1.5) * log(2 * alpha))
    else:
        aos = AOs_sphe_gto_data.build(**kwargs, magnetic_quantum_numbers=tuple(range(-ell, ell + 1)))
        log_normalization = 0.5 * (log(2.0) + (ell + 1.5) * log(2 * alpha) - lgamma(ell + 1.5))
        log_normalization += 0.5 * log((2 * ell + 1) / (4 * pi))
    points = np.zeros((len(radii), 3))
    points[:, 0 if cart else 2] = radii
    expected = np.asarray([[exp(log_normalization + ell * log(r) - alpha * r * r) for r in radii]])
    return aos, points, expected
