"""Physical derivative contract for the retained frozen B-spline tiling adapter."""

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc.atomic_orbital import (
    AOs_bspline_tiled_complex_data,
    compute_AOs,
    compute_AOs_value_grad_lap,
)
from jqmc.structure import tile_structure_data
from tests.test_jqmc_kernel.test_atomic_orbital import _build_bspline_leaf, _nongto_cubic_structure

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize("twist", ["gamma", "general"])
@pytest.mark.parametrize("conjugate", [False, True])
def test_frozen_tiled_bspline_physical_derivatives_match_autodiff(twist, conjugate):
    """Electron VGL and all nine fixed-Cartesian strain responses retain the phase."""
    primitive = _nongto_cubic_structure()
    matrix = np.diag([2, 1, 1])
    supercell = tile_structure_data(primitive, matrix)
    base = _build_bspline_leaf(primitive)
    coeffs = np.asarray(base.spline_coefficients, dtype=np.complex128)
    supertwist = np.zeros(3) if twist == "gamma" else np.array([0.125, -0.0625, 0.1875])
    k0 = np.linalg.solve(matrix, supertwist)
    leaf = AOs_bspline_tiled_complex_data.build(
        structure_data=supercell,
        spline_coefficients_by_k=(coeffs, (0.7 + 0.2j) * coeffs + 0.1j * np.roll(coeffs, 1, axis=1)),
        spline_shape=tuple(base.spline_shape),
        spline_order=base.spline_order,
        tiling_matrix=matrix,
        k_points=(k0, k0 + np.array([0.5, 0.0, 0.0])),
        supercell_k_point=supertwist,
    )
    if conjugate:
        leaf = leaf.with_negated_k()
    points = jnp.asarray([[0.31, 0.72, 1.13], [3.43, 1.27, 4.91], [-1.17, 6.32, -2.04]])
    fields = compute_AOs_value_grad_lap(leaf, points)

    def value_at_point(point):
        return compute_AOs(leaf, point[None, :])[:, 0]

    gradient = jax.vmap(jax.jacfwd(value_at_point))(points).transpose(1, 0, 2)
    hessian = jax.vmap(jax.jacfwd(jax.jacfwd(value_at_point)))(points)
    laplacian = jnp.trace(hessian, axis1=-2, axis2=-1).T
    reference = (*jnp.moveaxis(gradient, -1, 0), laplacian)
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")
    for actual, expected in zip(fields[1:], reference, strict=True):
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)

    def strained_value(eps):
        cell = supercell._cell_jnp @ (jnp.eye(3) + eps).T
        structure = supercell.replace(vec_a=cell[0], vec_b=cell[1], vec_c=cell[2])
        return compute_AOs(leaf.replace(structure_data=structure), points)

    # At fixed Cartesian r, a strain changes fractional coordinates by -eps r.
    # The full Bloch gradient therefore gives d psi / d eps_ij = -r_j d_i psi.
    analytical = -jnp.stack(fields[1:4], axis=-1)[..., :, None] * points[None, :, None, :]
    automatic = jax.jacfwd(strained_value)(jnp.zeros((3, 3)))
    np.testing.assert_allclose(analytical, automatic, atol=atol, rtol=rtol)
