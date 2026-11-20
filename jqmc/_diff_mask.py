"""Utilities for masking PyTree leaves by derivative type.

This module centralizes the logic for selectively stopping gradients on
parameter- or coordinate-like leaves so we can reuse the same dataclasses
for all derivative modes (params, positions, or none) without proliferating
specialized subclasses.
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

from dataclasses import dataclass, fields, is_dataclass
from typing import Any
import jax

# Field-name heuristics used when no explicit metadata is provided.  This is
# intentionally conservative and targets only the main variational and
# coordinate arrays we need to mask in practice.
_PARAM_FIELD_NAMES = {
    "lambda_matrix",
    # AGPn_data carries the natural-orbital occupation diagonal; it is a
    # variational parameter when ``opt_geminal=True`` and must be tagged
    # so the mask path can stop_gradient it when frozen.
    "lambda_diag",
    # AGPn_data open-shell unpaired block: variational on AGPn
    # (Euclidean step, no projection). Tag here so
    # the mask path can stop_gradient it when the geminal block is frozen.
    "lambda_unpaired",
    # AGPn_data / SD_data direct MO-opt path: ``mo_coefficients`` lives on
    # ``MOs_data`` inside ``orb_data_*_spin`` and is the actual variational
    # object for these subclasses. AGP_data with MOs_data treats it as
    # static storage; that case is handled by ``with_param_grad_mask``'s
    # AO-basis branch which explicitly freezes ``mo_coefficients`` for
    # AGP_data while leaving it differentiable for AGPn / SD.
    "mo_coefficients",
    "j_matrix",
    "jastrow_1b_param",
    "jastrow_2b_param",
    "jastrow_3b_param",
    "params",
    # DTN polynomial Jastrow (J1+J2+J3 family). Only the coefficient channels
    # are variational; the cutoff lengths L_u / L_chi / L_f are static
    # (pytree_node=False) and never appear as differentiable leaves.
    "alpha_indep",
    "beta_indep",
    "gamma_indep",
}
_COORD_FIELD_NAMES = {"positions"}


@dataclass(frozen=True)
class DiffMask:
    """Simple mask controlling which leaf types remain differentiable."""

    params: bool = True
    coords: bool = True

    def update(self, *, params: bool | None = None, coords: bool | None = None) -> DiffMask:
        """Return a new mask with any provided overrides applied."""
        return DiffMask(
            params=self.params if params is None else params,
            coords=self.coords if coords is None else coords,
        )


def _mask_sequence(
    seq: list[Any] | tuple[Any, ...],
    mask: DiffMask,
    memo: dict[int, Any],
) -> list[Any] | tuple[Any, ...]:
    """Apply ``apply_diff_mask`` elementwise to a list/tuple preserving type."""
    mapped = [_apply_diff_mask(v, mask, memo) for v in seq]
    return type(seq)(mapped) if isinstance(seq, tuple) else mapped


def apply_diff_mask(obj: Any, mask: DiffMask) -> Any:
    """Return a copy of ``obj`` with gradients stopped according to ``mask``.

    .. warning::

        This is a ``jax.lax.stop_gradient``-based mask. ``stop_gradient`` is
        a TRACE-TIME barrier -- calling this eagerly on a concrete pytree
        and then handing the result to ``jax.grad(..., argnums=0)`` is a
        silent no-op (every stop_gradient is wiped when the array becomes a
        grad tracer). Call this INSIDE the function differentiated by
        ``jax.grad``.

    The function recurses through dataclass fields (including Flax ``struct``
    dataclasses), honoring optional ``diff_tag`` metadata on fields when
    present. Otherwise, a small set of field-name heuristics is used to decide
    whether a leaf should be treated as a parameter ("param") or coordinate
    ("coord").  Lists/tuples are traversed elementwise; everything else is
    returned unchanged.

    Shared dataclass payloads remain shared within this traversal. The memo is
    local to this call, so no concrete array or tracer survives between traces.
    """
    return _apply_diff_mask(obj, mask, {})


def _apply_diff_mask(obj: Any, mask: DiffMask, memo: dict[int, Any]) -> Any:
    """Traverse an acyclic model graph without duplicating shared backends."""
    if id(obj) in memo:
        return memo[id(obj)]
    if is_dataclass(obj):
        updates = {}
        for f in fields(obj):
            value = getattr(obj, f.name)

            if is_dataclass(value) or isinstance(value, (list, tuple)):
                value = _apply_diff_mask(value, mask, memo)

            if f.metadata and "diff_tag" in f.metadata:
                diff_tag = f.metadata["diff_tag"]
            elif f.name in _PARAM_FIELD_NAMES:
                diff_tag = "param"
            elif f.name in _COORD_FIELD_NAMES:
                diff_tag = "coord"
            else:
                diff_tag = None

            if (diff_tag == "param" and not mask.params) or (diff_tag == "coord" and not mask.coords):
                if value is not None and not isinstance(value, (int, float, complex, bool)):
                    # Let failures on non-trivial leaves propagate. Silently
                    # passing one through would leave a frozen leaf differentiable.
                    value = jax.lax.stop_gradient(value)
            updates[f.name] = value

        replace_fn = getattr(obj, "replace", None)
        if callable(replace_fn):
            result = replace_fn(**updates)
        else:
            result = obj.__class__(**updates)
    elif isinstance(obj, (list, tuple)):
        result = _mask_sequence(obj, mask, memo)
    else:
        return obj
    memo[id(obj)] = result
    return result
