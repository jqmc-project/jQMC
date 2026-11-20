"""Cutoff-mode coverage under integer changes of a rectangular lattice basis."""

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
import pytest

from jqmc._setting import MPC_GRID_MIN_POINTS, MPC_GRID_POINTS_PER_MAX_FREQUENCY
from jqmc.mpc import resolve_grid_shape

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_GEOMETRIES = {
    "cubic": ((6, 6, 6), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
    "shear-1": ((6, 6, 6), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
    "shear-2": ((6, 6, 6), ((1, 0, 0), (2, 1, 0), (0, 0, 1))),
    "shear-4": ((6, 6, 6), ((1, 0, 0), (4, 1, 0), (0, 0, 1))),
    "shear-minus-4": ((6, 6, 6), ((1, 0, 0), (-4, 1, 0), (0, 0, 1))),
    "permuted-shear-4": ((6, 6, 6), ((0, 0, 1), (1, 0, 0), (4, 1, 0))),
    "orthorhombic": ((6, 8, 10), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
    "orthorhombic-shear-2": ((6, 8, 10), ((1, 0, 0), (2, 1, 0), (0, 0, 1))),
}

_ORTHOGONAL_CASES = {
    "cubic-50": ((6, 6, 6), 50.0, (64, 64, 64)),
    "orthorhombic-15": ((12, 9, 15), 15.0, (64, 64, 64)),
    "orthorhombic-30": ((12, 9, 15), 30.0, (64, 64, 76)),
    "orthorhombic-60": ((12, 9, 15), 60.0, (84, 64, 108)),
    "orthorhombic-240": ((12, 9, 15), 240.0, (168, 128, 212)),
}


def _integer_cutoff_modes(geometry):
    """Enumerate Cartesian modes, then transform their integer labels exactly.

    For ``cell = U @ diag(lengths)`` with integer unimodular ``U``, the
    Cartesian mode ``G = 2*pi*(p/lengths)`` has cell labels ``m = p @ U.T``.
    This does not use a reciprocal inverse or the resolver's norm bound.
    """
    lengths, transform = _GEOMETRIES[geometry]
    lengths = np.asarray(lengths)
    transform = np.asarray(transform, dtype=np.int64)
    a, b, c = transform
    determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
    assert abs(determinant) == 1, "The integer basis change must preserve the complete lattice."
    cutoff_ha = 50.0
    extent = 20
    # Any point outside this integer cube already exceeds the cutoff along
    # at least one Cartesian axis; no unenumerated mode can enter the sphere.
    assert np.all((2 * np.pi * (extent + 1) / lengths) ** 2 / 2 > cutoff_ha)
    axis = np.arange(-extent, extent + 1, dtype=np.int64)
    cartesian_indices = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    cartesian_g = 2 * np.pi * cartesian_indices / lengths
    retained = np.sum(cartesian_g**2, axis=1) / 2 <= cutoff_ha
    modes = cartesian_indices[retained] @ transform.T
    assert len(modes) > 1
    return transform @ np.diag(lengths), cutoff_ha, modes


@pytest.mark.parametrize("geometry", list(_GEOMETRIES))
def test_mpc_grid_covers_integer_sphere(geometry):
    """Every retained reciprocal mode fits the FFT box, including skew-axis cancellation.

    The independent reference enumerates a rectangular Cartesian reciprocal
    lattice and transforms its labels using an integer basis change. Both
    signed FFT edges are checked, not just an axis length or a norm estimate.
    """
    cell, cutoff_ha, modes = _integer_cutoff_modes(geometry)
    shape = np.asarray(resolve_grid_shape(cell, cutoff_ha))
    represented = np.all((modes >= -(shape // 2)) & (modes < shape // 2), axis=1)
    np.testing.assert_array_equal(represented, np.ones(len(modes), dtype=bool))


@pytest.mark.parametrize("geometry", list(_GEOMETRIES))
def test_mpc_grid_preserves_sampling_density(geometry):
    """Each FFT axis oversamples its actual integer cutoff modes at the existing density.

    Integer enumeration provides the per-axis maximum, independently of the
    grid resolver. Minimum size and even rounding remain required as well.
    """
    cell, cutoff_ha, modes = _integer_cutoff_modes(geometry)
    shape = np.asarray(resolve_grid_shape(cell, cutoff_ha))
    max_frequency = np.max(np.abs(modes), axis=0)
    required = np.maximum(MPC_GRID_MIN_POINTS, MPC_GRID_POINTS_PER_MAX_FREQUENCY * max_frequency)
    np.testing.assert_array_equal(shape >= required, np.ones(3, dtype=bool))
    np.testing.assert_array_equal(shape % 2, np.zeros(3, dtype=int))


@pytest.mark.parametrize("case", list(_ORTHOGONAL_CASES))
def test_mpc_grid_orthogonal_regression(case):
    """Orthogonal cells retain exact dimensions and singular cells remain rejected.

    These explicit dimensions cover the minimum-size floor and its crossover
    to cutoff-driven sizes without an energy cutoff on an integer boundary.
    """
    lengths, cutoff_ha, expected_shape = _ORTHOGONAL_CASES[case]
    np.testing.assert_array_equal(resolve_grid_shape(np.diag(lengths), cutoff_ha), expected_shape)
    singular_cell = np.diag(lengths)
    singular_cell[1] = singular_cell[0]
    with pytest.raises(np.linalg.LinAlgError):
        resolve_grid_shape(singular_cell, cutoff_ha)
