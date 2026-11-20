"""Independent references for cross blocks of periodic AO composites.

For a Bloch pGTO AO ``chi_a(r) = sum_T exp(i k.T) g_a(r - T)`` (``g_a`` the real
molecular contracted Cartesian Gaussian) and any partner ``phi_b`` with
``phi_b(r + T) = exp(i k.T) phi_b(r)``, unfolding the lattice sum gives

    S_ab = int_cell conj(chi_a) phi_b dr = int_R3 g_a(r) phi_b(r) dr.

* plane-wave partner ``exp(i q.r)``, ``q = G + k``: the closed-form Fourier
  transform of the Cartesian Gaussian (Hermite functions per axis);
  ``cos(G.r)`` / ``sin(G.r)`` take its real / imaginary part.
* B-spline partner ``exp(i k.r) u_b(r)`` in an orthorhombic cell: both factors
  are products over the Cartesian axes, so ``S_ab`` is a sum over control points
  of products of one-dimensional integrals, each done with many Gauss-Legendre
  nodes on sub-intervals of every knot interval (the spline is a polynomial on
  each, the Gaussian is resolved by the subdivision).

Only the molecular primitive data (exponents, contraction coefficients, powers,
centres) and the spline control points are read from the leaves; the
normalisation, the Fourier transform, the cardinal B-spline (``scipy``'s
``BSpline.basis_element``) and the quadrature are written out here. Everything
is evaluated in float64 / complex128.
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

import numpy as np
import scipy.interpolate
import scipy.special

from jqmc import atomic_orbital as ao


def _cartesian_primitives(pgto):
    """Molecular primitives of a Cartesian pGTO leaf: AO index, exponent, weight, powers, centre."""
    molecular = ao._detile_ragged_to_molecular(pgto)
    if not isinstance(molecular, ao.AOs_cart_gto_data):
        raise TypeError("the cross-block reference needs a Cartesian pGTO leaf")
    exponents = np.asarray(molecular.exponents, dtype=np.float64)
    coefficients = np.asarray(molecular.coefficients, dtype=np.float64)
    owner = np.asarray(molecular.orbital_indices, dtype=np.int64)
    powers = np.stack(
        [np.asarray(getattr(molecular, f"polynominal_order_{axis}"), dtype=np.int64)[owner] for axis in "xyz"],
        axis=1,
    )
    factorial = scipy.special.factorial
    ratio = np.prod(factorial(powers), axis=1) / np.prod(factorial(2 * powers), axis=1)
    norm = np.sqrt((2.0 * exponents / np.pi) ** 1.5 * (8.0 * exponents) ** powers.sum(axis=1) * ratio)
    centers = np.asarray(molecular._atomic_center_carts_np, dtype=np.float64)[owner]
    return int(molecular.num_ao), owner, exponents, coefficients * norm, powers, centers


def _gaussian_moment_fourier(power, exponent, q):
    """``int x^n exp(-a x^2 + i q x) dx = i^n (2 sqrt a)^-n H_n(u) sqrt(pi / a) exp(-u^2)``, ``u = q / (2 sqrt a)``."""
    u = q / (2.0 * np.sqrt(exponent))
    return (
        (1j**power)
        * (2.0 * np.sqrt(exponent)) ** (-power)
        * scipy.special.eval_hermite(power, u)
        * np.sqrt(np.pi / exponent)
        * np.exp(-u * u)
    )


def pgto_fourier_transform(pgto, q_carts):
    """``F_a(q) = int g_a(r) exp(i q.r) dr`` over all space, shape ``(num_ao, n_q)``."""
    num_ao, owner, exponents, weights, powers, centers = _cartesian_primitives(pgto)
    q_carts = np.asarray(q_carts, dtype=np.float64)
    out = np.zeros((num_ao, q_carts.shape[0]), dtype=np.complex128)
    for p in range(exponents.size):
        value = weights[p] * np.exp(1j * (q_carts @ centers[p]))
        for axis in range(3):
            value = value * _gaussian_moment_fourier(int(powers[p, axis]), exponents[p], q_carts[:, axis])
        out[owner[p]] += value
    return out


def pgto_plane_wave_cross(pgto, plane_waves):
    """Closed-form ``S[a, b] = int_cell conj(chi_a) phi_b`` for a pGTO bra and a plane-wave ket."""
    cell = np.asarray(plane_waves.structure_data.cell, dtype=np.float64)
    g_vectors = np.asarray(plane_waves.g_vectors, dtype=np.float64)
    if isinstance(plane_waves, ao.AOs_plane_wave_complex_data):
        k_cart = np.asarray(plane_waves.k_frac_np, dtype=np.float64) @ (2.0 * np.pi * np.linalg.inv(cell).T)
        return pgto_fourier_transform(pgto, g_vectors + k_cart[None, :])
    fourier = pgto_fourier_transform(pgto, g_vectors)
    is_sin = np.asarray(plane_waves.phase_is_sin, dtype=np.float64) > 0.5
    return np.where(is_sin[None, :], fourier.imag, fourier.real)


def pgto_bspline_cross(pgto, spline, *, nodes_per_piece=24, pieces_per_knot=8, log_cutoff=60.0):
    """Axis-factorised ``S[a, b] = int_R3 g_a(r) exp(i k.r) u_b(r) dr`` (orthorhombic cell)."""
    cell = np.asarray(spline.structure_data.cell, dtype=np.float64)
    if not np.array_equal(cell, np.diag(np.diag(cell))):
        raise ValueError("the axis-factorised B-spline reference needs an orthorhombic cell")
    lengths = np.diag(cell)
    shape = tuple(int(n) for n in spline.spline_shape)
    order = int(spline.spline_order)
    controls = np.asarray(spline.spline_coefficients).reshape((spline.num_ao, *shape))
    k_cart = np.zeros(3)
    if isinstance(spline, ao.AOs_bspline_complex_data):
        if spline.conjugate_coefficients:
            controls = np.conj(controls)
        k_cart = np.asarray(spline.k_frac_np, dtype=np.float64) @ (2.0 * np.pi * np.linalg.inv(cell).T)
    cardinal = scipy.interpolate.BSpline.basis_element(
        np.arange(order + 2, dtype=np.float64) - 0.5 * (order + 1), extrapolate=False
    )
    support = 0.5 * (order + 1)
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(nodes_per_piece)
    num_ao, owner, exponents, weights, powers, centers = _cartesian_primitives(pgto)
    out = np.zeros((num_ao, spline.num_ao), dtype=np.complex128)
    for p in range(exponents.size):
        half_width = np.sqrt(log_cutoff / exponents[p]) + 2.0
        factors = []
        for axis in range(3):
            n_knots, spacing = shape[axis], lengths[axis] / shape[axis]
            first = int(np.floor((centers[p, axis] - half_width) / spacing))
            last = int(np.ceil((centers[p, axis] + half_width) / spacing))
            edges = np.linspace(first, last, (last - first) * pieces_per_knot + 1) * spacing
            middle, radius = 0.5 * (edges[1:] + edges[:-1]), 0.5 * (edges[1:] - edges[:-1])
            x = (middle[:, None] + radius[:, None] * gl_nodes[None, :]).ravel()
            w = (radius[:, None] * gl_weights[None, :]).ravel()
            d = x - centers[p, axis]
            smooth = w * d ** int(powers[p, axis]) * np.exp(-exponents[p] * d * d) * np.exp(1j * k_cart[axis] * x)
            t = x / spacing
            moments = np.zeros(n_knots, dtype=np.complex128)
            for j in range(int(np.floor(t.min() - support)) - 1, int(np.ceil(t.max() + support)) + 2):
                moments[j % n_knots] += np.sum(smooth * np.nan_to_num(cardinal(t - j), nan=0.0))
            factors.append(moments)
        out[owner[p]] += weights[p] * np.einsum("bijk,i,j,k->b", controls, *factors)
    return out
