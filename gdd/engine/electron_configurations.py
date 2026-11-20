"""Deterministic electron-configuration generators governed by the fixture SSoT."""

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

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from jqmc.determinant import compute_geminal_all_elements


@dataclass(frozen=True)
class ConditionTargetResult:
    """One accepted finite-condition target and the observations that selected it."""

    r_up: np.ndarray
    r_dn: np.ndarray
    cond_g: float
    sigma_max_g: float
    sigma_min_g: float
    machine_epsilon: float
    cond_times_epsilon: float
    matrix_dimension: int
    matrix_dtype: str
    control_parameter: float
    generation_attempts: int
    rejected_configuration_count: int


def generate_same_spin_condition_target(
    geminal_data: Any,
    initial_r_up: Any,
    initial_r_dn: Any,
    *,
    target_minimum_cond_g: float,
    target_maximum_cond_g: float,
    separation_grid: tuple[float, ...],
    direction: tuple[float, float, float] = (1.0, -0.5, 0.25),
    anchor_up_index: int = 0,
    moving_up_index: int = 1,
) -> ConditionTargetResult:
    """Move one up-spin electron toward another on a declared deterministic grid.

    Each candidate is evaluated with the production geminal operation.  The
    first candidate in the declared order whose finite spectral condition
    number lies strictly inside the requested interval is accepted.  This is a
    grid search, not seed selection: the initial configuration never changes,
    every rejected candidate is counted, and exhaustion fails loudly.
    """
    r_up = np.asarray(initial_r_up, dtype=np.float64)
    r_dn = np.asarray(initial_r_dn, dtype=np.float64)
    if r_up.ndim != 2 or r_dn.ndim != 2 or r_up.shape[-1] != 3 or r_dn.shape[-1] != 3:
        raise AssertionError("condition-target generator requires spin-resolved (n, 3) coordinates")
    if anchor_up_index == moving_up_index or min(anchor_up_index, moving_up_index) < 0:
        raise AssertionError("condition-target generator requires two distinct nonnegative up-spin indices")
    if max(anchor_up_index, moving_up_index) >= r_up.shape[0]:
        raise AssertionError("condition-target generator requires two available up-spin electrons")
    if (
        not np.isfinite(target_minimum_cond_g)
        or not np.isfinite(target_maximum_cond_g)
        or target_minimum_cond_g < 1.0
        or target_maximum_cond_g <= target_minimum_cond_g
    ):
        raise AssertionError("condition-target generator requires ordered finite condition bounds")
    if not separation_grid or any(not np.isfinite(delta) or delta <= 0.0 for delta in separation_grid):
        raise AssertionError("condition-target generator requires a nonempty positive finite separation grid")
    if any(next_delta >= delta for delta, next_delta in pairwise(separation_grid)):
        raise AssertionError("condition-target separation grid must be strictly decreasing")

    unit_direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(unit_direction))
    if not np.isfinite(norm) or norm == 0.0:
        raise AssertionError("condition-target direction must be finite and nonzero")
    unit_direction /= norm

    epsilon = float(np.finfo(np.float64).eps)
    last_cond = float("nan")
    for attempt, delta in enumerate(separation_grid, start=1):
        candidate_up = r_up.copy()
        candidate_up[moving_up_index] = candidate_up[anchor_up_index] + delta * unit_direction
        # This production call is the generation oracle.  Returning a reviewed
        # coordinate constant without exercising it cannot satisfy this route.
        matrix = np.asarray(compute_geminal_all_elements(geminal_data, candidate_up, r_dn))
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if not np.all(np.isfinite(singular_values)) or singular_values[-1] <= 0.0:
            last_cond = float("inf")
            continue
        sigma_max = float(singular_values[0])
        sigma_min = float(singular_values[-1])
        cond_g = sigma_max / sigma_min
        last_cond = cond_g
        if target_minimum_cond_g < cond_g < target_maximum_cond_g:
            return ConditionTargetResult(
                r_up=candidate_up,
                r_dn=r_dn.copy(),
                cond_g=cond_g,
                sigma_max_g=sigma_max,
                sigma_min_g=sigma_min,
                machine_epsilon=epsilon,
                cond_times_epsilon=cond_g * epsilon,
                matrix_dimension=int(matrix.shape[0]),
                matrix_dtype=str(matrix.dtype),
                control_parameter=float(delta),
                generation_attempts=attempt,
                rejected_configuration_count=attempt - 1,
            )
    raise AssertionError(
        "condition-target search exhausted "
        f"{len(separation_grid)} candidates without reaching "
        f"{target_minimum_cond_g} < cond_2(G) < {target_maximum_cond_g}; last={last_cond}"
    )
