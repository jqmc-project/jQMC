"""Odd cardinal spline controls with independently known half-cell nodes."""

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

from jqmc.atomic_orbital import AOs_bspline_data
from jqmc.structure import Structure_data


def nodal_spline_case(cell_case):
    cell = np.asarray([[6.0, 0.0, 0.0], [1.0, 7.0, 0.0], [0.5, 1.5, 8.0]])
    structure = Structure_data.build(
        pbc_flag=True,
        positions=np.zeros((1, 3)),
        atomic_numbers=(2,),
        element_symbols=("He",),
        atomic_labels=("He1",),
        vec_a=cell[0],
        vec_b=cell[1],
        vec_c=cell[2],
    )
    shape = (32, 32, 32)
    frequency = 5
    controls = np.sin(2 * np.pi * frequency * np.indices(shape) / shape[0]).reshape(3, -1)
    aos = AOs_bspline_data.build(structure, controls, shape, 5)
    points = np.random.default_rng(460).uniform(0.1, 0.9, size=(3, 32, 3))
    for axis in range(3):
        points[axis, :, axis] = 0.5
    off_nodes = points.copy()
    for axis in range(3):
        off_nodes[axis, :, axis] += 1 / (4 * frequency)
    translation = {"home": np.zeros(3), "translated": np.asarray([2, -3, 1])}[cell_case]
    return aos, (points + translation) @ cell, (off_nodes + translation) @ cell


def mirror_complex_spline_case(leaf_kind, cell_case):
    """Odd complex controls about a half-integer node near the top of an N = 30 grid.

    Band ``a`` is ``sin(2 pi 7 (k_a - 28.5) / 30)`` along axis ``a`` times a smooth
    even profile on the other axes and the complex factor ``1 + 0.5 i``.  The
    symmetric cardinal kernel preserves the oddness, so the periodic part obeys
    ``u(c + d) = -u(c - d)`` exactly at ``t_a = 28.5 +- d`` for any offset ``d``.
    A grid of 30 is not a power of two and ``t_a`` lies close to ``N``, so the
    local offset ``t_a - floor(t_a)`` carries the fp32 rounding of ``t_a``
    unless it is formed before the cast.

    Returns ``(aos, plus, minus, k_cart, axes)``: the leaf, the Cartesian mirror
    points ``(3, 64, 3)`` (first index: odd axis), the Cartesian Bloch vector of
    each AO row and the odd axis of each AO row.  The tiled leaf stores the same
    three bands at both primitive k points of a 2 x 1 x 1 tiling.
    """
    from jqmc.atomic_orbital import AOs_bspline_complex_data, AOs_bspline_tiled_complex_data

    primitive = np.asarray([[6.0, 0.0, 0.0], [1.0, 7.0, 0.0], [0.5, 1.5, 8.0]])
    n_grid, center, frequency = 30, 28.5, 7
    knots = np.arange(n_grid)
    odd = np.sin(2 * np.pi * frequency * (knots - center) / n_grid)
    even = 1.0 + 0.3 * np.cos(2 * np.pi * knots / n_grid)
    bands = []
    for axis in range(3):
        profiles = [even, even, even]
        profiles[axis] = odd
        bands.append(np.einsum("i,j,k->ijk", *profiles) * (1.0 + 0.5j))
    controls = np.stack(bands).reshape(3, -1)
    shape = (n_grid, n_grid, n_grid)
    if leaf_kind == "complex":
        cell = primitive
        aos = AOs_bspline_complex_data.build(_structure(cell), controls, shape, spline_order=5, k_point=(0.11, 0.13, 0.09))
        k_cart = np.repeat((np.asarray(aos.k_frac_np) @ (2 * np.pi * np.linalg.inv(cell).T))[None, :], 3, axis=0)
        axes = (0, 1, 2)
    elif leaf_kind == "tiled":
        tiling = np.diag([2, 1, 1])
        cell = tiling @ primitive
        aos = AOs_bspline_tiled_complex_data.build(
            _structure(cell),
            [controls, controls],
            shape,
            tiling,
            [(0.11, 0.13, 0.09), (0.61, 0.13, 0.09)],
            spline_order=5,
            supercell_k_point=(0.22, 0.13, 0.09),
        )
        k_cart = np.asarray(aos.orbital_k_cart_np)
        axes = (0, 1, 2, 0, 1, 2)
    else:
        raise ValueError(leaf_kind)
    rng = np.random.default_rng(784)
    points = rng.uniform(0.05, 0.95, size=(3, 64, 3))
    offsets = rng.uniform(0.05, 0.45, size=(3, 64))
    plus, minus = points.copy(), points.copy()
    for axis in range(3):
        plus[axis, :, axis] = (center + offsets[axis]) / n_grid
        minus[axis, :, axis] = (center - offsets[axis]) / n_grid
    translation = {"home": np.zeros(3), "translated": np.asarray([2, -3, 1])}[cell_case]
    return aos, (plus + translation) @ primitive, (minus + translation) @ primitive, k_cart, axes


def _structure(cell):
    return Structure_data.build(
        pbc_flag=True,
        positions=np.zeros((1, 3)),
        atomic_numbers=(2,),
        element_symbols=("He",),
        atomic_labels=("He1",),
        vec_a=cell[0],
        vec_b=cell[1],
        vec_c=cell[2],
    )
