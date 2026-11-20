"""Converter centre-translation laws, evaluated in the declared IO zone."""

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

from dataclasses import replace

import numpy as np
import pytest

from jqmc._setting import ATOL_PW_PROJECTION_FP64, RTOL_PW_PROJECTION_FP64
from jqmc.pw_pgto_projection import (
    CartPrimitive,
    _generated_pgto_overlap_condition,
    build_gto_design_matrix,
    eval_primitive_on_grid,
    grid_points,
    periodic_pgto_overlap_matrix,
)
from jqmc.qe_wrapper import QECell
from tests.test_jqmc_kernel.test_pw_pgto_projection import _toy_reference

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _offcell_case(cell_kind, angular):
    matrix = 6.0 * np.eye(3) if cell_kind == "cubic" else np.array([[6.0, 0.0, 0.0], [1.4, 5.5, 0.0], [-0.7, 0.8, 6.3]])
    cell = QECell(a=matrix)
    first = CartPrimitive(tuple(np.array([0.21, 0.31, 0.42]) @ matrix), (0, 0, 0), 0.7)
    second = CartPrimitive(tuple(np.array([0.35, 0.24, 0.37]) @ matrix), angular, 1.1)
    lattice_shift = np.array([9, -7, 4]) @ matrix
    moved = replace(second, center=tuple(np.asarray(second.center) + lattice_shift))
    return cell, first, second, moved, lattice_shift


@pytest.mark.parametrize("cell_kind", ["cubic", "skew"])
@pytest.mark.parametrize("angular", [(0, 0, 0), (1, 0, 0), (0, 1, 1)])
def test_converter_primitive_far_center_translation(cell_kind, angular):
    """A Gamma s, p, or d primitive is unchanged after its centre moves by many cells.

    Cubic and oblique cells use a fixed 4x5x6 evaluation grid. Moving only
    the centre by (9,-7,4) cells must retain every nonzero sampled value.
    Both precision modes exercise the converter's existing float64 IO zone.
    """
    cell, _, primitive, shifted, _ = _offcell_case(cell_kind, angular)
    r_carts = grid_points(cell, (4, 5, 6))
    reference = eval_primitive_on_grid(cell, primitive, r_carts)
    actual = eval_primitive_on_grid(cell, shifted, r_carts)
    assert np.max(np.abs(reference)) > ATOL_PW_PROJECTION_FP64
    np.testing.assert_allclose(actual, reference, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64)


@pytest.mark.parametrize("cell_kind", ["cubic", "skew"])
@pytest.mark.parametrize("angular", [(0, 0, 0), (1, 0, 0), (0, 1, 1)])
@pytest.mark.parametrize("fractional_k", [(0.0, 0.0, 0.0), (0.25, 0.125, -0.2)])
def test_converter_far_center_bloch_covariance(cell_kind, angular, fractional_k):
    """Grid values and the raw overlap obey the exact centre-translation Bloch phase.

    Moving one of two primitive centres by L multiplies its AO by exp(-ikL).
    The cross overlap must transform by conjugate/ordinary phases on its two
    indices, so the converter's raw overlap condition number is unchanged.
    No basis re-normalization, precision conversion, or phase removal is used.
    """
    cell, first, second, shifted, lattice_shift = _offcell_case(cell_kind, angular)
    r_carts = grid_points(cell, (4, 5, 6))
    k_cart = np.asarray(fractional_k) @ cell.b
    qe_reference = _toy_reference(cell, np.array([[0, 0, 0]]), (4, 5, 6), k_point=k_cart)
    phase = np.exp(-1j * (k_cart @ lattice_shift))
    phases = np.array([1.0, phase])
    original_overlap = periodic_pgto_overlap_matrix(qe_reference, (first, second))
    expected_overlap = phases.conj()[:, None] * original_overlap * phases[None, :]
    actual_overlap = periodic_pgto_overlap_matrix(qe_reference, (first, shifted))
    assert abs(expected_overlap[0, 1]) > ATOL_PW_PROJECTION_FP64
    np.testing.assert_allclose(actual_overlap, expected_overlap, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64)
    expected_grid = build_gto_design_matrix(cell, [second], r_carts, n_jobs=1, k_cart=k_cart) * phase
    actual_grid = build_gto_design_matrix(cell, [shifted], r_carts, n_jobs=1, k_cart=k_cart)
    assert np.max(np.abs(expected_grid)) > ATOL_PW_PROJECTION_FP64
    np.testing.assert_allclose(actual_grid, expected_grid, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64)
    original_condition = _generated_pgto_overlap_condition(qe_reference, (first, second))
    actual_condition = _generated_pgto_overlap_condition(qe_reference, (first, shifted))
    assert np.isfinite(original_condition)
    np.testing.assert_allclose(actual_condition, original_condition, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64)
    pw_millers = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    original_mixed_condition = _generated_pgto_overlap_condition(qe_reference, (first, second), pw_millers)
    actual_mixed_condition = _generated_pgto_overlap_condition(qe_reference, (first, shifted), pw_millers)
    assert np.isfinite(original_mixed_condition)
    np.testing.assert_allclose(
        actual_mixed_condition, original_mixed_condition, atol=ATOL_PW_PROJECTION_FP64, rtol=RTOL_PW_PROJECTION_FP64
    )
