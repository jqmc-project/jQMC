"""Independent finite active-set reference for periodic primitive image sets.

No production image, radius, Gram-metric or candidate-grid helper is imported.
The candidate bound uses the smallest singular value of the lattice. Each
box-distance problem checks all 27 faces/edges/vertices, solving its free
coordinates by Cartesian least squares rather than coordinate descent.
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

from itertools import product

import numpy as np


def primitive_radius(exponent, angular_l, tolerance):
    """Find the outer envelope crossing, not a fixed-point approximation."""
    if angular_l == 0:
        return float(np.sqrt(-np.log(tolerance) / exponent))
    lower = max(1.0, np.sqrt(angular_l / (2.0 * exponent)))

    def log_residual(radius):
        return angular_l * np.log(radius) - exponent * radius**2 - np.log(tolerance)

    if log_residual(lower) <= 0:
        raise ValueError("oracle fixture must have its envelope crossing outside the peak and unit radius")
    upper = 2.0 * lower
    while log_residual(upper) > 0:
        upper *= 2.0
    for _ in range(80):
        middle = (lower + upper) / 2.0
        if log_residual(middle) > 0:
            lower = middle
        else:
            upper = middle
    return upper


def box_distances(cell, lower, upper):
    """Minimum Cartesian distance from the origin to each fractional box."""
    cell = np.asarray(cell, dtype=np.float64)
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if lower.shape != upper.shape or lower.ndim != 2 or lower.shape[1] != 3 or np.any(lower > upper):
        raise ValueError("invalid fractional boxes")
    best_squared = np.full(len(lower), np.inf)
    for status in product((-1, 0, 1), repeat=3):
        free = [axis for axis, value in enumerate(status) if value == 0]
        points = np.zeros_like(lower)
        for axis, value in enumerate(status):
            if value:
                points[:, axis] = lower[:, axis] if value < 0 else upper[:, axis]
        if free:
            points[:, free] = np.linalg.lstsq(cell[free].T, -(points @ cell).T, rcond=None)[0].T
        feasible = np.all((points >= lower) & (points <= upper), axis=1)
        squared = np.sum((points @ cell) ** 2, axis=1)
        best_squared = np.minimum(best_squared, np.where(feasible, squared, np.inf))
    if not np.all(np.isfinite(best_squared)):
        raise AssertionError("active-set enumeration failed to visit a feasible box vertex")
    return np.sqrt(best_squared)


def image_indices(cell, exponent, angular_l, tolerance, fractional_atom=None):
    """Enumerate the complete integer image set with an independent outer bound."""
    cell = np.asarray(cell, dtype=np.float64)
    radius = primitive_radius(exponent, angular_l, tolerance)
    smallest = float(np.linalg.svd(cell, compute_uv=False)[-1])
    if not np.isfinite(smallest) or smallest <= 0:
        raise ValueError("oracle needs a nonsingular cell")
    # ||v @ cell|| >= sigma_min * ||v||. In either displacement box
    # |n_i| <= |v_i| + 1 + |s_i|, so no accepted image lies outside this cube.
    # The atom position s is the actual (unwrapped) one: its copies R + n @ cell
    # must reach the home cell [0, 1)^3 of the electrons.
    atom = np.zeros(3) if fractional_atom is None else np.asarray(fractional_atom, dtype=np.float64)
    bound = int(np.ceil(radius / smallest + 1.0 + float(np.max(np.abs(atom)))))
    integers = np.asarray(list(product(range(-bound, bound + 1), repeat=3)), dtype=np.int64)
    if fractional_atom is None:
        lower, upper = integers - 1.0, integers + 1.0
    else:
        lower, upper = -integers - atom, 1.0 - integers - atom
    return integers[box_distances(cell, lower, upper) <= radius]


def sorted_indices(values):
    """Canonicalize row ordering without discarding duplicates or changing dtype."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 3:
        raise AssertionError("image indices must have shape (N, 3)")
    return values[np.lexsort((values[:, 2], values[:, 1], values[:, 0]))]
