"""Executable classifiers used by the wavefunction-fixture SSoT."""

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
from collections.abc import Sequence


CELL_METRIC_LENGTH_RTOL = 1.0e-8
CELL_METRIC_LENGTH_ATOL = 1.0e-10
CELL_METRIC_COSINE_ATOL = 1.0e-8


def _equal(left: float, right: float, *, rtol: float, atol: float) -> bool:
    return math.isclose(left, right, rel_tol=rtol, abs_tol=atol)


def _length_multiplicities(lengths: Sequence[float], *, rtol: float, atol: float) -> list[int]:
    groups: list[list[float]] = []
    for value in sorted(float(item) for item in lengths):
        for group in groups:
            if _equal(value, group[0], rtol=rtol, atol=atol):
                group.append(value)
                break
        else:
            groups.append([value])
    return sorted((len(group) for group in groups), reverse=True)


def classify_cell_metric(
    cell: Sequence[Sequence[float]],
    *,
    length_rtol: float = CELL_METRIC_LENGTH_RTOL,
    length_atol: float = CELL_METRIC_LENGTH_ATOL,
    cosine_atol: float = CELL_METRIC_COSINE_ATOL,
) -> str:
    """Classify the metric family of a periodic cell whose lattice vectors are rows."""
    try:
        matrix = [[float(value) for value in row] for row in cell]
    except (TypeError, ValueError) as exc:
        raise ValueError("cell must be a finite 3x3 matrix") from exc
    if (
        len(matrix) != 3
        or any(len(row) != 3 for row in matrix)
        or not all(math.isfinite(value) for row in matrix for value in row)
    ):
        raise ValueError("cell must be a finite 3x3 matrix")

    lengths = [math.sqrt(sum(value * value for value in row)) for row in matrix]
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if any(length <= length_atol for length in lengths) or abs(determinant) <= length_atol:
        raise ValueError("cell must contain three nonzero, linearly independent vectors")

    def cosine(left: int, right: int) -> float:
        return sum(a * b for a, b in zip(matrix[left], matrix[right], strict=True)) / (lengths[left] * lengths[right])

    cosines = [cosine(0, 1), cosine(0, 2), cosine(1, 2)]
    right = [abs(value) <= cosine_atol for value in cosines]
    right_count = sum(right)
    multiplicities = _length_multiplicities(lengths, rtol=length_rtol, atol=length_atol)

    if multiplicities == [3] and right_count == 3:
        return "cubic"
    if multiplicities == [2, 1] and right_count == 3:
        return "tetragonal"
    if right_count == 3:
        return "orthorhombic"
    non_right = [abs(value) for value, is_right in zip(cosines, right, strict=True) if not is_right]
    if (
        multiplicities == [2, 1]
        and right_count == 2
        and len(non_right) == 1
        and _equal(non_right[0], 0.5, rtol=0.0, atol=cosine_atol)
    ):
        return "hexagonal"
    if (
        multiplicities == [3]
        and right_count == 0
        and all(_equal(value, cosines[0], rtol=0.0, atol=cosine_atol) for value in cosines)
    ):
        return "rhombohedral"
    if right_count == 2:
        return "monoclinic"
    return "triclinic"
