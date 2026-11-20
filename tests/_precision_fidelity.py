"""Shared execution support for mixed-versus-full numerical-fidelity tests."""

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

from collections.abc import Callable
from typing import TypeVar

import jax
import numpy as np

from jqmc._precision import configure, is_mixed_precision_enabled, precision_policy_snapshot

CaseT = TypeVar("CaseT")
ResultT = TypeVar("ResultT")


def mixed_fidelity_blocks(
    mixed: ResultT,
    full: ResultT,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return matching finite host-array leaves as homogeneous blocks."""
    mixed_leaves, mixed_tree = jax.tree_util.tree_flatten(mixed)
    full_leaves, full_tree = jax.tree_util.tree_flatten(full)
    if mixed_tree != full_tree:
        raise AssertionError("mixed and full results have different pytree structures")

    compared: list[tuple[np.ndarray, np.ndarray]] = []
    for block_index, (mixed_leaf, full_leaf) in enumerate(zip(mixed_leaves, full_leaves, strict=True)):
        mixed_block = np.asarray(mixed_leaf)
        full_block = np.asarray(full_leaf)
        if mixed_block.shape != full_block.shape:
            raise AssertionError(
                f"mixed-fidelity block {block_index} shape mismatch: {mixed_block.shape} != {full_block.shape}"
            )
        if mixed_block.size == 0:
            continue
        if not np.all(np.isfinite(mixed_block)) or not np.all(np.isfinite(full_block)):
            raise AssertionError(f"mixed-fidelity block {block_index} contains NaN or Inf")
        compared.append((mixed_block, full_block))
    if not compared:
        raise AssertionError("mixed-fidelity comparison has no nonempty blocks")
    return tuple(compared)


def out_of_budget_mixed_block(
    mixed_block: np.ndarray,
    full_block: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> np.ndarray:
    """Copy one mixed block and inject an error beyond its registered budget."""
    if mixed_block.size == 0 or full_block.size == 0:
        raise AssertionError("mixed-fidelity liveness check requires a nonempty block")
    mutated = mixed_block.copy()
    allowed = atol + rtol * float(np.max(np.abs(full_block)))
    mutated.flat[0] = full_block.flat[0] + 2.0 * allowed
    return mutated


def _configure_all_backends(
    mode: str,
    configure_native: Callable[[dict[str, str]], None] | None,
) -> None:
    """Configure JAX and, when supplied, the native extension."""
    configure(mode)
    if configure_native is not None:
        configure_native(precision_policy_snapshot())


def mixed_vs_full(
    case_factory: Callable[[], CaseT],
    evaluate: Callable[[CaseT], ResultT],
    *,
    configure_native: Callable[[dict[str, str]], None] | None = None,
    native_policy_snapshot: Callable[[], dict[str, str]] | None = None,
) -> tuple[ResultT, ResultT]:
    """Evaluate one route on independently rebuilt mixed and full cases.

    ``case_factory`` runs after each policy switch.  This is required because
    model leaves may be downcast while a case is built, and native contexts
    snapshot their precision policy at construction. A native test must pass
    its extension's ``configure_native_precision`` and policy getter functions
    explicitly; pure XLA tests do not import or initialize the native/MPI stack.
    Python and native entry policies are saved and restored independently
    because a caller may deliberately enter with different policies. Results
    are transferred to host arrays before the next switch so no lazy
    computation can execute under the wrong policy.

    Returns:
        ``(mixed_result, full_result)`` with each backend's original precision
        policy restored.

    Raises:
        ValueError: If only one native policy callback is supplied.
    """
    if (configure_native is None) != (native_policy_snapshot is None):
        raise ValueError("configure_native and native_policy_snapshot must be supplied together")
    original_mode = "mixed" if is_mixed_precision_enabled() else "full"
    original_native_policy = dict(native_policy_snapshot()) if native_policy_snapshot is not None else None

    def run(mode: str) -> ResultT:
        _configure_all_backends(mode, configure_native)
        result = evaluate(case_factory())
        return jax.tree_util.tree_map(
            lambda value: (
                np.asarray(jax.device_get(value)).copy() if isinstance(value, (jax.Array, np.ndarray, np.generic)) else value
            ),
            result,
        )

    try:
        full_result = run("full")
        mixed_result = run("mixed")
        return mixed_result, full_result
    finally:
        try:
            configure(original_mode)
        finally:
            if configure_native is not None:
                configure_native(original_native_policy)
