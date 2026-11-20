"""Wavefunction module.

Precision Zones:
    - ``wf_eval``: Psi and ln|Psi| evaluation.
    - ``wf_kinetic``: kinetic-energy evaluation (compute_kinetic_energy*).
    - ``wf_ratio``: wavefunction ratios Psi(R')/Psi(R).
    - Zone-boundary casts when combining results from the Jastrow and
      determinant zones.

See :mod:`jqmc._precision` for details.
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

# python modules
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from logging import getLogger
from typing import Literal, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import grad, jit, vmap
from jax import typing as jnpt

from ._diff_mask import DiffMask, apply_diff_mask
from ._precision import cast_to_zone, get_loosest_zone_dtype_np
from ._route_probe import note_route, route_site
from ._setting import (
    S_STIEFEL_METRIC_MIN_EIGENVALUE_RCOND,
    get_s_stiefel_backward_error_limit,
)
from .atomic_orbital import (
    AOs_base_data,
    AOs_cart_gto_data,
    AOs_GTO_base_data,
    AOs_sphe_gto_data,
    compute_overlap_matrix,
)
from .determinant import (
    Det_candidate_mesh_state,
    Det_ratio_streaming_state,
    Det_streaming_state,
    Geminal_data,
    _advance_det_candidate_mesh_state,
    _advance_grads_laplacian_ln_Det_streaming_state,
    _compute_ratio_determinant_from_winv_state,
    _compute_ratio_determinant_part_split_spin,
    _init_det_candidate_mesh_state,
    _init_det_winv_state,
    _init_grads_laplacian_ln_Det_streaming_state,
    compute_det_geminal_all_elements,
    compute_geminal_all_elements,
    compute_grads_and_laplacian_ln_Det,
    compute_grads_and_laplacian_ln_Det_fast,
    compute_ln_det_geminal_all_elements,
    compute_ln_det_geminal_all_elements_fast,
)
from .jastrow_factor import (
    J1_candidate_mesh_state,
    J2_candidate_mesh_state,
    J3_candidate_mesh_state,
    J_DTN_candidate_mesh_state,
    Jastrow_data,
    Jastrow_DTN_grad_lap_streaming_state,
    Jastrow_one_body_streaming_state,
    Jastrow_three_body_streaming_state,
    Jastrow_two_body_streaming_state,
    _advance_j3_candidate_mesh_state,
    _advance_j2_candidate_mesh_state,
    _advance_j_dtn_candidate_mesh_state,
    _advance_grads_laplacian_Jastrow_DTN_streaming_state,
    _advance_grads_laplacian_Jastrow_one_body_streaming_state,
    _advance_grads_laplacian_Jastrow_three_body_streaming_state,
    _advance_grads_laplacian_Jastrow_two_body_streaming_state,
    _init_grads_laplacian_Jastrow_DTN_streaming_state,
    _init_grads_laplacian_Jastrow_one_body_streaming_state,
    _init_grads_laplacian_Jastrow_three_body_streaming_state,
    _init_grads_laplacian_Jastrow_two_body_streaming_state,
    _init_j3_candidate_mesh_state,
    _init_j1_candidate_mesh_state,
    _init_j2_candidate_mesh_state,
    _init_j_dtn_candidate_mesh_state,
    _advance_j1_candidate_mesh_state,
    compute_grads_and_laplacian_Jastrow_DTN,
    compute_grads_and_laplacian_Jastrow_part,
    compute_ratio_Jastrow_one_move_mesh,
    compute_Jastrow_part,
    compute_Jastrow_part_ratio_zone,
)
from .molecular_orbital import MOs_data
from .orbital_tiling import TiledOrbitalSet_data

# set logger
logger = getLogger("jqmc").getChild(__name__)

_BlockT = TypeVar("_BlockT")

# JAX float64
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Shell-constraint helpers for AO basis optimization
# ---------------------------------------------------------------------------


def _get_aos_data(orb_data: AOs_base_data | MOs_data) -> AOs_base_data:
    """Extract the underlying AOs_*_data from orb_data (AO or MO)."""
    if isinstance(orb_data, (AOs_sphe_gto_data, AOs_cart_gto_data)):
        return orb_data
    if isinstance(orb_data, MOs_data):
        return orb_data.aos_data
    raise NotImplementedError(f"Unsupported orb_data type: {type(orb_data)}")


#: Orbital basis-payload pytree fields that are never coordinate-like. Used by
#: :func:`_freeze_orbital_basis` to stop-gradient the basis definition while
#: leaving ``structure_data`` (positions / cell) live.
_ORB_BASIS_PAYLOAD_FIELDS = (
    "mo_coefficients",  # MOs_data
    "radial",  # GTO leaves: nested AO_radial_shell_data (exponents + coefficients)
    "spline_coefficients",  # B-spline leaf
    "phase_is_sin",  # plane-wave leaf
)


def _freeze_orbital_basis_keep_mo_coefficients(
    orb: AOs_base_data | MOs_data | None,
) -> AOs_base_data | MOs_data | None:
    """Variant of :func:`_freeze_orbital_basis` that keeps ``mo_coefficients``
    differentiable.

    Used by :meth:`Wavefunction_data.with_param_grad_mask` for the
    ``AGPn_data`` / ``SD_data`` direct MO-opt path
    (``opt_geminal=True``): the MO coefficients ARE the variational
    object, so they must keep their grad tracer. The underlying AO basis
    (exponents / coefficients / spline grid / etc.) is still
    stop-gradient'd because it carries no variational parameter when the
    ``opt_lambda_basis_*`` flags are off.
    """
    if orb is None:
        return orb
    updates = {}
    if hasattr(orb, "aos_data"):
        updates["aos_data"] = _freeze_orbital_basis(orb.aos_data)
    if hasattr(orb, "children") and isinstance(orb.children, tuple):
        updates["children"] = tuple(_freeze_orbital_basis(c) for c in orb.children)
    # Stop the leaf-level basis payload arrays EXCEPT mo_coefficients.
    for fname in _ORB_BASIS_PAYLOAD_FIELDS:
        if fname == "mo_coefficients":
            continue
        if hasattr(orb, fname):
            current = getattr(orb, fname)
            if current is not None:
                updates[fname] = jax.lax.stop_gradient(current)
    if updates:
        return orb.replace(**updates)
    return orb


def _freeze_orbital_basis(
    orb: AOs_base_data | MOs_data | None,
) -> AOs_base_data | MOs_data | None:
    """Stop-gradient every non-variational basis-payload array in an orbital.

    .. warning::

        Like :py:meth:`Wavefunction_data.with_param_grad_mask`, this is a
        ``stop_gradient``-based mask and only has effect when called
        IN-TRACE inside the function differentiated by ``jax.grad``. Eager
        use on a concrete pytree followed by ``grad(argnums=0)`` is a
        silent no-op (the stop_gradient is wiped when the array becomes a
        tracer).

    Recurses through MO wrappers (``aos_data``) and composites (``children``),
    stopping the basis-defining arrays listed in
    :data:`_ORB_BASIS_PAYLOAD_FIELDS` (``mo_coefficients`` / ``exponents`` /
    ``coefficients`` / ``spline_coefficients`` / ``phase_is_sin``) while leaving
    ``structure_data`` (positions / cell) untouched -- so the same masked
    wavefunction remains valid for position- and cell-gradient paths.

    Used by :meth:`Wavefunction_data.with_param_grad_mask` when the orbital
    basis carries no variational parameter (the ``opt_*_basis_*`` flags are all
    off, including the default Jastrow-only optimization). It prevents
    ``grad(compute_local_energy, argnums=0)`` (the de_L/dc parameter gradient)
    from wasting compute -- or, for the large B-spline control-point tensor,
    OOM-ing -- differentiating the frozen orbital. ``mo_coefficients`` is never
    a variational block, so freezing it is always correct here; the per-array
    GTO basis-exponent / coefficient optimization (a non-frozen basis) is
    handled by the selective masking path instead.
    """
    if orb is None:
        return orb
    updates = {}
    if isinstance(orb, TiledOrbitalSet_data):
        return orb.replace(
            blocks=tuple(
                b.replace(
                    backend=_freeze_orbital_basis(b.backend),
                    coefficients=None if b.coefficients is None else jax.lax.stop_gradient(b.coefficients),
                )
                for b in orb.blocks
            )
        )
    # Recurse into the MO wrapper's AO data and composite children.
    if hasattr(orb, "aos_data"):
        updates["aos_data"] = _freeze_orbital_basis(orb.aos_data)
    if hasattr(orb, "children") and isinstance(orb.children, tuple):
        updates["children"] = tuple(_freeze_orbital_basis(c) for c in orb.children)
    # Stop the leaf-level basis payload arrays that this orbital carries.
    for fname in _ORB_BASIS_PAYLOAD_FIELDS:
        if hasattr(orb, fname):
            val = getattr(orb, fname)
            if val is not None:
                updates[fname] = jax.lax.stop_gradient(val)
    return orb.replace(**updates) if updates else orb


StateToken: TypeAlias = tuple[str, int]
ParameterField: TypeAlias = Literal["real", "complex"]
DerivativePolicy: TypeAlias = Literal["direct_real", "holomorphic", "wirtinger_pair"]
SynchronizationScope: TypeAlias = Literal["sampler", "world"]
TangentKind: TypeAlias = Literal["euclidean", "linear_tie", "s_stiefel_rows"]
_KineticElements: TypeAlias = tuple[jax.Array, jax.Array]


def _validate_state_token(state_token: StateToken) -> StateToken:
    if (
        not isinstance(state_token, tuple)
        or len(state_token) != 2
        or not isinstance(state_token[0], str)
        or not state_token[0]
        or not isinstance(state_token[1], int)
        or isinstance(state_token[1], bool)
        or state_token[1] < 0
    ):
        raise ValueError("state_token must be a non-empty lineage string and a non-negative integer revision.")
    return state_token


def _readonly_array(
    values: npt.ArrayLike,
    *,
    dtype: npt.DTypeLike | None = None,
) -> np.ndarray:
    arr = np.array(values, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


#: Precision Zones the S-Stiefel overlap's computation crosses. ``det_eval`` is
#: where ``compute_overlap_matrix`` stores the result; ``ao_eval`` is where the
#: AO primitives (exponents / coefficients) it is assembled from live, and it is
#: fp32 under mixed precision. The achievable accuracy is bounded by the weakest
#: of the two, which is what the metric guards must size their tolerance from.
_S_STIEFEL_OVERLAP_ZONES = ("ao_eval", "det_eval")


def _normalized_hermitian_backward_error(matrix: np.ndarray) -> float:
    scale = float(np.linalg.norm(matrix, ord=np.inf))
    residual = float(np.linalg.norm(matrix - matrix.conj().T, ord=np.inf))
    if scale == 0.0:
        return 0.0 if residual == 0.0 else float("inf")
    return residual / scale


def _normalized_inverse_backward_error(sqrt: np.ndarray, inv_sqrt: np.ndarray) -> float:
    identity = np.eye(sqrt.shape[0], dtype=np.result_type(sqrt.dtype, inv_sqrt.dtype))
    residual = sqrt @ inv_sqrt - identity
    scale = max(1.0, float(np.linalg.norm(np.abs(sqrt) @ np.abs(inv_sqrt), ord=np.inf)))
    return float(np.linalg.norm(residual, ord=np.inf)) / scale


def _validate_s_stiefel_metric_factors(sqrt: np.ndarray, inv_sqrt: np.ndarray) -> None:
    limit = get_s_stiefel_backward_error_limit(np.result_type(sqrt.dtype, inv_sqrt.dtype), sqrt.shape[0])
    if _normalized_hermitian_backward_error(sqrt) > limit:
        raise ValueError("s_stiefel_rows metric_sqrt must be Hermitian within backward error.")
    if _normalized_hermitian_backward_error(inv_sqrt) > limit:
        raise ValueError("s_stiefel_rows metric_inv_sqrt must be Hermitian within backward error.")
    if _normalized_inverse_backward_error(sqrt, inv_sqrt) > limit:
        raise ValueError("s_stiefel_rows metric square roots must be mutual inverses within backward error.")


def _build_s_stiefel_metric_factors(overlap: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Build a strict positive-definite S-Stiefel metric factor pair."""
    matrix = np.asarray(overlap)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
        raise ValueError("S-Stiefel overlap must be a finite square matrix.")
    # The bound must come from the accuracy the overlap can CARRY, not from the
    # dtype it is STORED in. ``compute_overlap_matrix`` stores S on ``det_eval``
    # (fp64 in both modes), but the AO primitives it is built from live on
    # ``ao_eval``, which is fp32 under mixed precision -- so S is only
    # ~fp32-accurate there. Keying the limit off ``matrix.dtype`` would demand
    # fp64-tight symmetry / positivity of fp32-accurate data and reject a
    # perfectly healthy overlap (lambda_min ~ -1e-10 vs a 6e-12 bound).
    effective_dtype = get_loosest_zone_dtype_np(_S_STIEFEL_OVERLAP_ZONES)
    limit = get_s_stiefel_backward_error_limit(effective_dtype, matrix.shape[0])
    if _normalized_hermitian_backward_error(matrix) > limit:
        raise ValueError("S-Stiefel overlap must be Hermitian within backward error.")
    matrix = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    maximum = float(np.max(eigenvalues, initial=0.0))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("S-Stiefel overlap must be positive semidefinite with nonzero rank.")
    negative_limit = limit * maximum
    minimum = float(np.min(eigenvalues, initial=0.0))
    if minimum < -negative_limit:
        raise ValueError(
            "S-Stiefel overlap must be positive semidefinite; "
            f"lambda_min={minimum:.3e}, backward-error bound={negative_limit:.3e}."
        )
    eigenvalue_floor = max(negative_limit, S_STIEFEL_METRIC_MIN_EIGENVALUE_RCOND * maximum)
    regularized = np.maximum(eigenvalues, eigenvalue_floor)
    sqrt = (eigenvectors * np.sqrt(regularized)) @ eigenvectors.conj().T
    inv_sqrt = (eigenvectors * (1.0 / np.sqrt(regularized))) @ eigenvectors.conj().T
    _validate_s_stiefel_metric_factors(sqrt, inv_sqrt)
    return sqrt, inv_sqrt


_S_STIEFEL_METRIC_CACHE: dict[int, tuple[weakref.ReferenceType, tuple[np.ndarray, np.ndarray]]] = {}


def _s_stiefel_metric_factors_for_aos(aos_data: AOs_base_data) -> tuple[np.ndarray, np.ndarray]:
    key = id(aos_data)
    cached = _S_STIEFEL_METRIC_CACHE.get(key)
    if cached is not None and cached[0]() is aos_data:
        return cached[1]
    factors = _build_s_stiefel_metric_factors(np.asarray(compute_overlap_matrix(aos_data)))

    def _drop(
        _reference: weakref.ReferenceType[object],
        *,
        cache_key: int = key,
    ) -> None:
        _S_STIEFEL_METRIC_CACHE.pop(cache_key, None)

    _S_STIEFEL_METRIC_CACHE[key] = (weakref.ref(aos_data, _drop), factors)
    return factors


@dataclass(frozen=True, eq=False)
class LinearTieSpec:
    """Immutable COO expansion from independent real coordinates to storage."""

    output_indices: np.ndarray
    coordinate_indices: np.ndarray
    coefficients: np.ndarray
    output_size: int
    coordinate_size: int

    @classmethod
    def build(
        cls,
        *,
        output_indices: npt.ArrayLike,
        coordinate_indices: npt.ArrayLike,
        coefficients: npt.ArrayLike,
        output_size: int,
        coordinate_size: int,
    ) -> "LinearTieSpec":
        out = np.asarray(output_indices)
        coord = np.asarray(coordinate_indices)
        coeff = np.asarray(coefficients)
        if out.ndim != 1 or coord.ndim != 1 or coeff.ndim != 1:
            raise ValueError("LinearTieSpec COO arrays must be one-dimensional.")
        if not (out.size == coord.size == coeff.size):
            raise ValueError("LinearTieSpec COO arrays must have identical lengths.")
        if not np.issubdtype(out.dtype, np.integer) or not np.issubdtype(coord.dtype, np.integer):
            raise ValueError("LinearTieSpec indices must have integer dtype.")
        if int(output_size) < 1 or int(coordinate_size) < 1:
            raise ValueError("LinearTieSpec sizes must be positive.")
        out_i = np.asarray(out, dtype=np.int64)
        coord_i = np.asarray(coord, dtype=np.int64)
        if np.any(out_i < 0) or np.any(out_i >= int(output_size)):
            raise ValueError("LinearTieSpec output index is outside output_size.")
        if np.any(coord_i < 0) or np.any(coord_i >= int(coordinate_size)):
            raise ValueError("LinearTieSpec coordinate index is outside coordinate_size.")
        if np.unique(out_i).size != int(output_size):
            raise ValueError("LinearTieSpec must publish every storage output exactly through its COO entries.")
        if np.unique(coord_i).size != int(coordinate_size):
            raise ValueError("LinearTieSpec coordinate indices must be contiguous and all used.")
        pairs = np.stack((out_i, coord_i), axis=1)
        if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
            raise ValueError("LinearTieSpec contains a duplicate output/coordinate entry.")
        if not np.all(np.isfinite(coeff)):
            raise ValueError("LinearTieSpec coefficients must be finite.")
        return cls(
            output_indices=_readonly_array(out_i, dtype=np.int64),
            coordinate_indices=_readonly_array(coord_i, dtype=np.int64),
            coefficients=_readonly_array(coeff),
            output_size=int(output_size),
            coordinate_size=int(coordinate_size),
        )

    def to_dense(self) -> np.ndarray:
        dtype = np.result_type(self.coefficients.dtype, np.float64)
        dense = np.zeros((self.output_size, self.coordinate_size), dtype=dtype)
        dense[self.output_indices, self.coordinate_indices] = self.coefficients
        return dense


@dataclass(frozen=True, eq=False)
class ParameterTangentSpec:
    """Data-only tangent policy published by the owning wave-function block."""

    kind: TangentKind
    metric_sqrt: np.ndarray | None
    metric_inv_sqrt: np.ndarray | None
    linear_tie: LinearTieSpec | None
    state_token: StateToken

    @classmethod
    def build(
        cls,
        *,
        kind: TangentKind,
        metric_sqrt: npt.ArrayLike | None,
        metric_inv_sqrt: npt.ArrayLike | None,
        linear_tie: LinearTieSpec | None,
        state_token: StateToken,
    ) -> "ParameterTangentSpec":
        token = _validate_state_token(state_token)
        if kind not in ("euclidean", "linear_tie", "s_stiefel_rows"):
            raise ValueError(f"Unknown tangent kind: {kind!r}.")
        if kind == "euclidean":
            if metric_sqrt is not None or metric_inv_sqrt is not None or linear_tie is not None:
                raise ValueError("euclidean tangent must not carry metric or linear-tie data.")
            return cls(kind=kind, metric_sqrt=None, metric_inv_sqrt=None, linear_tie=None, state_token=token)
        if kind == "linear_tie":
            if not isinstance(linear_tie, LinearTieSpec) or metric_sqrt is not None or metric_inv_sqrt is not None:
                raise ValueError("linear_tie tangent requires exactly one LinearTieSpec.")
            return cls(kind=kind, metric_sqrt=None, metric_inv_sqrt=None, linear_tie=linear_tie, state_token=token)
        if linear_tie is not None or metric_sqrt is None or metric_inv_sqrt is None:
            raise ValueError("s_stiefel_rows tangent requires both metric square roots and no linear tie.")
        sqrt = np.asarray(metric_sqrt)
        inv_sqrt = np.asarray(metric_inv_sqrt)
        if sqrt.ndim != 2 or sqrt.shape[0] != sqrt.shape[1] or inv_sqrt.shape != sqrt.shape:
            raise ValueError("s_stiefel_rows metric square roots must be same-shape square matrices.")
        if not np.all(np.isfinite(sqrt)) or not np.all(np.isfinite(inv_sqrt)):
            raise ValueError("s_stiefel_rows metric square roots must be finite.")
        _validate_s_stiefel_metric_factors(sqrt, inv_sqrt)
        return cls(
            kind=kind,
            metric_sqrt=_readonly_array(sqrt),
            metric_inv_sqrt=_readonly_array(inv_sqrt),
            linear_tie=None,
            state_token=token,
        )

    @classmethod
    def euclidean(cls, *, state_token: StateToken) -> "ParameterTangentSpec":
        return cls.build(
            kind="euclidean",
            metric_sqrt=None,
            metric_inv_sqrt=None,
            linear_tie=None,
            state_token=state_token,
        )

    @classmethod
    def with_linear_tie(cls, linear_tie: LinearTieSpec, *, state_token: StateToken) -> "ParameterTangentSpec":
        return cls.build(
            kind="linear_tie",
            metric_sqrt=None,
            metric_inv_sqrt=None,
            linear_tie=linear_tie,
            state_token=state_token,
        )

    @classmethod
    def s_stiefel_rows(
        cls,
        *,
        metric_sqrt: npt.ArrayLike,
        metric_inv_sqrt: npt.ArrayLike,
        state_token: StateToken,
    ) -> "ParameterTangentSpec":
        return cls.build(
            kind="s_stiefel_rows",
            metric_sqrt=metric_sqrt,
            metric_inv_sqrt=metric_inv_sqrt,
            linear_tie=None,
            state_token=state_token,
        )


def _symmetric_paired_linear_tie(shape: tuple[int, ...], *, n_paired: int) -> LinearTieSpec:
    """Return a real-linear tie for a symmetric paired block plus free columns."""
    if len(shape) != 2 or shape[0] != int(n_paired) or shape[1] < int(n_paired):
        raise ValueError(f"symmetric paired storage must have shape (n, n + u); got shape={shape}, n={n_paired}.")
    n = int(n_paired)
    n_cols = int(shape[1])
    output_indices: list[int] = []
    coordinate_indices: list[int] = []
    coefficients: list[float] = []
    q = 0
    for i in range(n):
        output_indices.append(i * n_cols + i)
        coordinate_indices.append(q)
        coefficients.append(1.0)
        q += 1
    offdiag_scale = 1.0 / np.sqrt(2.0)
    for i in range(n):
        for j in range(i + 1, n):
            output_indices.extend((i * n_cols + j, j * n_cols + i))
            coordinate_indices.extend((q, q))
            coefficients.extend((offdiag_scale, offdiag_scale))
            q += 1
    for i in range(n):
        for j in range(n, n_cols):
            output_indices.append(i * n_cols + j)
            coordinate_indices.append(q)
            coefficients.append(1.0)
            q += 1
    return LinearTieSpec.build(
        output_indices=np.asarray(output_indices),
        coordinate_indices=np.asarray(coordinate_indices),
        coefficients=np.asarray(coefficients),
        output_size=int(np.prod(shape)),
        coordinate_size=q,
    )


def _symmetric_real_linear_tie(shape: tuple[int, ...]) -> LinearTieSpec:
    if len(shape) != 2 or shape[1] != shape[0] + 1:
        raise ValueError(f"symmetric J3 storage must have shape (n, n + 1); got {shape}.")
    return _symmetric_paired_linear_tie(shape, n_paired=int(shape[0]))


def _hermitian_paired_linear_tie(shape: tuple[int, ...], *, n_paired: int) -> LinearTieSpec:
    if len(shape) != 2 or shape[0] != int(n_paired) or shape[1] < int(n_paired):
        raise ValueError(f"Hermitian paired storage must have shape (n, n + u); got shape={shape}, n={n_paired}.")
    n = int(n_paired)
    n_cols = int(shape[1])
    output_indices: list[int] = []
    coordinate_indices: list[int] = []
    coefficients: list[complex] = []
    q = 0
    for i in range(n):
        output_indices.append(i * n_cols + i)
        coordinate_indices.append(q)
        coefficients.append(1.0)
        q += 1
    offdiag_scale = 1.0 / np.sqrt(2.0)
    for i in range(n):
        for j in range(i + 1, n):
            ij = i * n_cols + j
            ji = j * n_cols + i
            output_indices.extend((ij, ji))
            coordinate_indices.extend((q, q))
            coefficients.extend((offdiag_scale, offdiag_scale))
            q += 1
            output_indices.extend((ij, ji))
            coordinate_indices.extend((q, q))
            coefficients.extend((1j * offdiag_scale, -1j * offdiag_scale))
            q += 1
    for i in range(n):
        for j in range(n, n_cols):
            output = i * n_cols + j
            output_indices.append(output)
            coordinate_indices.append(q)
            coefficients.append(1.0)
            q += 1
            output_indices.append(output)
            coordinate_indices.append(q)
            coefficients.append(1j)
            q += 1
    return LinearTieSpec.build(
        output_indices=np.asarray(output_indices),
        coordinate_indices=np.asarray(coordinate_indices),
        coefficients=np.asarray(coefficients),
        output_size=int(np.prod(shape)),
        coordinate_size=q,
    )


def _project_linear_tie_storage(
    values: npt.ArrayLike,
    tie: LinearTieSpec,
    *,
    field: ParameterField,
) -> np.ndarray:
    """Temporary original projection derived solely from immutable tie data.

    Works directly on the tie's COO index arrays (its primary representation --
    plain numpy gather + ``np.add.at`` segment sums, no sparse library). A
    dense form via ``to_dense()`` and two BLAS matmuls would build an
    ``(output_size, coordinate_size)`` matrix holding only ``nnz ~
    output_size`` non-zeros -- 131 GiB at J3 n_orb = 432 while ~0.001% dense.
    Peak is a few ``(batch, nnz)`` gathers, i.e. the same order as the
    input itself. ``to_dense()`` is retained as the small-system test oracle.

    Numerics: the segment-sum accumulates in COO-entry order instead of the
    BLAS reduction order, so results agree with the dense form to fp round-off
    (asserted against the ``to_dense`` oracle in ``tests/test_jqmc_kernel/
    test_optimizer_block_metadata.py::test_tie_projection_coo_matches_dense_oracle``),
    not bit-exactly.
    """
    arr = np.asarray(values)
    if arr.shape[-1] != tie.output_size:
        raise ValueError(f"linear-tie projection expected final axis {tie.output_size}; got {arr.shape}.")
    # Match ``to_dense()``'s dtype promotion so the arithmetic type is
    # unchanged relative to the dense form.
    coeff = np.asarray(tie.coefficients, dtype=np.result_type(tie.coefficients.dtype, np.float64))
    out_idx = tie.output_indices
    coord_idx = tie.coordinate_indices
    flat = arr.reshape(-1, tie.output_size)

    # coordinates[:, j] = sum over entries e with coord_e == j of
    #   flat[:, out_e] * coeff_e            (real field)
    #   Re(flat[:, out_e] * conj(coeff_e))  (complex field; Re commutes with the sum)
    if field == "real":
        if np.iscomplexobj(coeff):
            raise ValueError("real-field linear tie must have real coefficients.")
        weighted = flat[:, out_idx] * coeff[None, :]  # (batch, nnz)
    elif field == "complex":
        weighted = np.real(flat[:, out_idx] * np.conj(coeff)[None, :])  # (batch, nnz)
    else:
        raise ValueError(f"Unknown parameter field: {field!r}.")
    coordinates_t = np.zeros((tie.coordinate_size, flat.shape[0]), dtype=weighted.dtype)
    np.add.at(coordinates_t, coord_idx, weighted.T)

    # projected[:, i] = sum over entries e with out_e == i of
    #   coordinates[:, coord_e] * coeff_e   (coeff NOT conjugated, as in `@ dense.T`)
    back = coordinates_t[coord_idx, :] * coeff[:, None]  # (nnz, batch)
    projected_t = np.zeros((tie.output_size, flat.shape[0]), dtype=back.dtype)
    np.add.at(projected_t, out_idx, back)
    return projected_t.T.reshape(arr.shape)


@struct.dataclass
class VariationalParameterBlock:
    """A block of variational parameters (e.g., J1, J2, J3, lambda).

    Design overview
    ----------------
    * A *block* is the smallest unit that the optimizer (MCMC + SR) sees.
      Each block corresponds to a contiguous slice in the global
      variational parameter vector and carries enough metadata to
      reconstruct its original shape (name, values, shape, size).
    * This class is intentionally **structure-agnostic**: it does not
      know anything about Jastrow vs Geminal, matrix symmetry, or how a
      block maps to concrete fields in :class:`Jastrow_data` or
      :class:`Geminal_data`.
    * All physics- and structure-specific semantics are owned by the
      corresponding data classes via their ``get_variational_blocks`` and
      ``apply_block_update`` methods.

    The goal is that adding or modifying a variational parameter only
    requires changes on the wavefunction side (Jastrow/Geminal data),
    while the MCMC/SR driver remains completely agnostic and operates
    purely on a list of blocks.
    """

    name: str  #: Identifier for this block (for example ``"j1_param"`` or ``"lambda_matrix"``).
    values: jnpt.ArrayLike = struct.field(pytree_node=True)  #: Parameter payload (keeps PyTree structure if present).
    shape: tuple[int, ...] = struct.field(pytree_node=False)  #: Original shape of ``values`` for unflattening updates.
    size: int = struct.field(pytree_node=False)  #: Flattened size of ``values`` used when slicing the global vector.
    field: ParameterField = struct.field(pytree_node=False)
    derivative_policy: DerivativePolicy = struct.field(pytree_node=False)
    sync_scope: SynchronizationScope = struct.field(pytree_node=False)
    tangent_spec: ParameterTangentSpec = struct.field(pytree_node=False)
    state_token: StateToken = struct.field(pytree_node=False)

    def __post_init__(self) -> None:
        token = _validate_state_token(self.state_token)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("variational block name must be a non-empty string.")
        if not isinstance(self.tangent_spec, ParameterTangentSpec):
            raise ValueError("tangent_spec must be a ParameterTangentSpec.")
        if self.tangent_spec.state_token != token:
            raise ValueError("block state_token does not match tangent_spec.state_token.")
        if self.field not in ("real", "complex"):
            raise ValueError(f"Unknown parameter field: {self.field!r}.")
        if self.sync_scope not in ("sampler", "world"):
            raise ValueError(f"Unknown synchronization scope: {self.sync_scope!r}.")
        allowed_policy = {
            "real": ("direct_real",),
            "complex": ("holomorphic", "wirtinger_pair"),
        }
        if self.derivative_policy not in allowed_policy[self.field]:
            raise ValueError(f"derivative_policy={self.derivative_policy!r} is invalid for field={self.field!r}.")
        arr = np.asarray(self.values)
        if tuple(arr.shape) != tuple(self.shape) or int(arr.size) != int(self.size):
            raise ValueError("VariationalParameterBlock shape/size do not match values.")
        if int(self.size) < 1:
            raise ValueError("VariationalParameterBlock must contain at least one storage value.")
        storage_is_complex = np.issubdtype(arr.dtype, np.complexfloating)
        if (self.field == "complex") != storage_is_complex:
            raise ValueError(f"field={self.field!r} is inconsistent with storage dtype {arr.dtype}.")
        if self.tangent_spec.kind == "linear_tie":
            assert self.tangent_spec.linear_tie is not None
            if self.tangent_spec.linear_tie.output_size != int(self.size):
                raise ValueError("linear-tie output_size does not match block storage size.")
            if self.field == "real" and np.iscomplexobj(self.tangent_spec.linear_tie.coefficients):
                raise ValueError("a real-field linear tie must not introduce imaginary storage directions.")
            if self.field == "complex" and self.derivative_policy != "wirtinger_pair":
                raise ValueError("a complex linear-tie block requires two-component Wirtinger derivatives.")
        if self.tangent_spec.kind == "s_stiefel_rows":
            assert self.tangent_spec.metric_sqrt is not None
            if arr.ndim != 2 or self.tangent_spec.metric_sqrt.shape != (arr.shape[1], arr.shape[1]):
                raise ValueError(
                    "s_stiefel_rows requires a two-dimensional row-coefficient block "
                    "whose column count matches the overlap metric."
                )
        if isinstance(self.values, np.ndarray):
            object.__setattr__(self, "values", _readonly_array(self.values))

    def apply_update(self, delta_flat: npt.NDArray, learning_rate: float) -> "VariationalParameterBlock":
        r"""Return a new block with values updated by a generic additive rule.

        This method is intentionally *structure-agnostic* and only performs a
        simple additive update::

            X_new = X_old + learning_rate * delta

        Any parameter-specific constraints (e.g., symmetry of J3 or
        ``lambda_matrix``) must be enforced by the owner of the parameter
        (``jastrow_data``, ``geminal_data``, etc.) inside their
        ``apply_block_update`` implementations.

        Args:
            delta_flat: Flattened update vector with length equal to ``size``.
            learning_rate: Scaling factor for the update.
        """
        dX = delta_flat.reshape(self.shape)
        # Storage guard: real arrays cannot accept an imaginary update;
        # complex arrays retain both components. For a mixed parameter set the
        # solver must already have imposed the real-coordinate embedding.
        # This last-resort cast protects storage but is not a substitute for the
        # constrained mixed-manifold solve.
        _values_arr = np.array(self.values)
        if not np.issubdtype(_values_arr.dtype, np.complexfloating) and np.iscomplexobj(dX):
            dX = dX.real
        new_values = _values_arr + learning_rate * dX

        return self.replace(values=new_values, shape=new_values.shape, size=new_values.size)

    @classmethod
    def build(
        cls,
        *,
        name: str,
        values: jnpt.ArrayLike,
        field: ParameterField,
        derivative_policy: DerivativePolicy,
        sync_scope: SynchronizationScope,
        tangent_spec: ParameterTangentSpec,
        state_token: StateToken,
    ) -> "VariationalParameterBlock":
        """Build a :class:`VariationalParameterBlock`, deriving ``shape`` / ``size``.

        Blessed factory: infers ``shape`` and ``size`` from ``values`` while
        requiring the complete optimizer-facing metadata contract.
        """
        arr = np.asarray(values)
        return cls(
            name=name,
            values=values,
            shape=tuple(arr.shape),
            size=int(arr.size),
            field=field,
            derivative_policy=derivative_policy,
            sync_scope=sync_scope,
            tangent_spec=tangent_spec,
            state_token=state_token,
        )


@struct.dataclass
class Wavefunction_data:
    """Container for Jastrow and Geminal parts used to evaluate a wavefunction.

    The class owns only the data needed to construct the wavefunction. All
    computations are delegated to the functions in this module and the
    underlying Jastrow/Geminal helpers.

    Args:
        jastrow_data: Optional Jastrow parameters. If ``None``, the Jastrow part is omitted.
        geminal_data: Optional Geminal parameters. If ``None``, the determinant part is omitted.
    """

    jastrow_data: Jastrow_data = struct.field(
        pytree_node=True, default_factory=Jastrow_data
    )  #: Variational Jastrow parameters.
    geminal_data: Geminal_data = struct.field(
        pytree_node=True, default_factory=Geminal_data
    )  #: Variational Geminal/determinant parameters.

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        self.jastrow_data.sanity_check()
        self.geminal_data.sanity_check()

    @classmethod
    def build(
        cls,
        *,
        jastrow_data: Jastrow_data | None = None,
        geminal_data: Geminal_data | None = None,
    ) -> "Wavefunction_data":
        """Build a validated :class:`Wavefunction_data` from its components.

        Blessed factory: assembles the Jastrow and Geminal parts (each
        defaulting to an empty instance when ``None``), constructs the
        container, and runs :py:meth:`sanity_check`. Direct construction
        (``Wavefunction_data(...)``) remains supported.
        """
        data = cls(
            jastrow_data=Jastrow_data() if jastrow_data is None else jastrow_data,
            geminal_data=Geminal_data() if geminal_data is None else geminal_data,
        )
        data.sanity_check()
        return data

    def _get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        # Replace geminal_data.logger_info() with geminal_data.get_info() output.
        info_lines.extend(self.geminal_data._get_info())
        # Replace jastrow_data.logger_info() with jastrow_data.get_info() output.
        info_lines.extend(self.jastrow_data._get_info())
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    def apply_block_updates(
        self,
        blocks: list[VariationalParameterBlock],
        thetas: npt.NDArray,
        learning_rate: float,
    ) -> "Wavefunction_data":
        """Return a new :class:`Wavefunction_data` with variational blocks updated.

        ``blocks`` defines the ordering and shapes of the variational
        parameters. ``thetas`` is a flattened real-coordinate update in the
        same order. Each block decodes its slice according to its declared
        tangent action; the Jastrow and geminal owners then enforce their
        structural constraints.

        Args:
            blocks: Ordered variational-parameter blocks returned by
                :meth:`get_variational_blocks`.
            thetas: Flattened real-coordinate update. Its length must equal
                the sum of the block sizes.
            learning_rate: Scalar multiplier applied to every block update.

        Returns:
            A new :class:`Wavefunction_data`; the receiver is not modified.
        """
        jastrow_data = self.jastrow_data
        geminal_data = self.geminal_data

        pos = 0
        for block in blocks:
            start = pos
            end = pos + block.size
            pos = end
            delta_flat = thetas[start:end]
            if np.all(delta_flat == 0.0):
                continue

            updated_block = block.apply_update(delta_flat, learning_rate=learning_rate)

            # Delegate the mapping from block to internal parameters to
            # Jastrow_data and Geminal_data.
            if jastrow_data is not None:
                jastrow_data = jastrow_data.apply_block_update(updated_block)
            if geminal_data is not None:
                geminal_data = geminal_data.apply_block_update(updated_block)

        return Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)

    def apply_optimization_updates(
        self,
        *,
        blocks: tuple[VariationalParameterBlock, ...] | list[VariationalParameterBlock],
        updates: Sequence[VariationalParameterBlock],
        expected_state_token: StateToken,
    ) -> "Wavefunction_data":
        """Validate and atomically construct one optimizer-updated candidate.

        ``updates`` must be the complete ordered publication produced from the
        same immutable block snapshot.  All structural checks run before any
        component update is evaluated; ansatz owners then retract a temporary
        candidate, whose invariants are checked before it is returned.
        """
        token = _validate_state_token(expected_state_token)
        published = tuple(blocks)
        proposed = tuple(updates)
        if not published:
            raise ValueError("optimizer update requires at least one variational block.")
        if len(proposed) != len(published):
            raise ValueError("optimizer update is missing or has extra blocks.")

        block_names = [block.name for block in published]
        update_names = [getattr(update, "name", None) for update in proposed]
        if update_names != block_names:
            raise ValueError("optimizer update names/order do not match the published blocks.")
        if len(set(block_names)) != len(block_names):
            raise ValueError("optimizer update publication contains duplicate block names.")

        typed_deltas: list[np.ndarray] = []
        for block, update in zip(published, proposed, strict=True):
            if block.state_token != token or getattr(update, "state_token", None) != token:
                raise ValueError(f"stale optimizer update for block {block.name!r}.")
            delta = np.asarray(getattr(update, "delta", None))
            if tuple(delta.shape) != tuple(block.shape):
                raise ValueError(f"optimizer update for block {block.name!r} has the wrong shape.")
            if not np.issubdtype(delta.dtype, np.number) or not np.all(np.isfinite(delta)):
                raise ValueError(f"optimizer update for block {block.name!r} is non-finite or non-numeric.")
            block_dtype = np.asarray(block.values).dtype
            if delta.dtype != block_dtype:
                raise ValueError(f"optimizer update for block {block.name!r} has the wrong storage dtype.")
            if block.field == "real" and np.iscomplexobj(delta):
                raise ValueError(f"real optimizer block {block.name!r} received a complex update.")
            typed_deltas.append(delta)

        jastrow_data = self.jastrow_data
        geminal_data = self.geminal_data
        for block, delta in zip(published, typed_deltas, strict=True):
            if np.all(delta == 0):
                continue
            updated_block = block.apply_update(delta.reshape(-1), learning_rate=1.0)
            if jastrow_data is not None:
                jastrow_data = jastrow_data.apply_block_update(updated_block)
            if geminal_data is not None:
                geminal_data = geminal_data.apply_block_update(updated_block)

        candidate = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)
        candidate.sanity_check()
        return candidate

    # NOTE: this signature is the canonical LITERAL declaration
    # of the param-block default convention (parameter blocks True, "basis" blocks
    # False). A Python signature needs literal defaults, so it cannot itself derive
    # from the single source; instead the single source
    # ``jqmc.derivatives.bridge._MASK_DEFAULTS`` is DERIVED to mirror these defaults,
    # and drift between the two is caught by
    # ``tests/test_jqmc_kernel/test_derivatives_interface.py`` (the signature defaults
    # must equal ``_MASK_DEFAULTS``). Keep the two in lock-step.
    def with_param_grad_mask(
        self,
        *,
        opt_J1_param: bool = True,
        opt_J2_param: bool = True,
        opt_J3_param: bool = True,
        opt_geminal: bool = True,
        opt_J3_basis_exp: bool = False,
        opt_J3_basis_coeff: bool = False,
        opt_lambda_basis_exp: bool = False,
        opt_lambda_basis_coeff: bool = False,
        opt_JDTN_u_param: bool = True,
        opt_JDTN_chi_param: bool = True,
        opt_JDTN_f_param: bool = True,
    ) -> "Wavefunction_data":
        """Return a copy where disabled parameter blocks stop propagating gradients.

        .. warning::

            ``jax.lax.stop_gradient`` is a **trace-time** barrier. Calling this
            method *eagerly* on a concrete ``Wavefunction_data`` and then
            handing the result as the ``argnums=0`` primal of ``jax.grad``
            is a **silent no-op** -- every ``stop_gradient`` is wiped when the
            masked arrays become grad tracers. You MUST call this method
            *inside* the function differentiated by ``jax.grad``::

                def _loss(wf, *args):
                    masked = wf.with_param_grad_mask(**flags)
                    return f(masked, *args)
                jax.grad(_loss, argnums=0)(wf, *args)   # mask runs in-trace

            For the production param-grad path see
            ``jqmc.derivatives.bridge.batched_grad_e_L`` /
            ``batched_grad_ln_Psi`` (called with ``mask=`` + ``analytic=``);
            they apply this mask in-trace.

        Developer note
        --------------
        * The per-block flags (``opt_J1_param`` etc.) decide which high-level blocks are
            masked. Disabled blocks are wrapped with ``DiffMask(params=False, coords=True)``,
            meaning parameter gradients are stopped while coordinate gradients still flow.
        * Within each disabled block, ``apply_diff_mask`` uses field-name heuristics
            (see ``diff_mask._PARAM_FIELD_NAMES``) to tag parameter leaves such as
            ``lambda_matrix``, ``j_matrix``, ``jastrow_1b_param``, ``jastrow_2b_param``,
            ``jastrow_3b_param``, and ``params``. Those tagged leaves receive
            ``jax.lax.stop_gradient``, so their backpropagated gradients become zero.
        * AO basis flags (``opt_J3_basis_exp`` etc.) independently control whether
            ``exponents`` / ``coefficients`` gradients flow. These are handled as a
            separate masking step so they are fully independent of the high-level block
            flags.
        * Example: if ``opt_J1_param=False`` and others are True, only the J1 block is
            masked; its parameter leaves are stopped, while J2/J3/NN/lambda continue to
            propagate gradients normally.
        """
        if self.geminal_data is not None and isinstance(self.geminal_data.orb_data_up_spin, TiledOrbitalSet_data):
            if opt_geminal or opt_lambda_basis_exp or opt_lambda_basis_coeff:
                raise NotImplementedError(
                    "tiled orbital/geminal optimization is frozen; set opt_geminal and orbital-basis optimization flags to false."
                )
        mask_off = DiffMask(params=False, coords=True)

        def _maybe_mask(block: _BlockT | None, enabled: bool) -> tuple[_BlockT | None, bool]:
            if enabled or block is None:
                return block, False
            return apply_diff_mask(block, mask_off), True

        def _is_gto_leaf(orb: AOs_base_data | MOs_data) -> bool:
            """True iff ``orb`` (AO leaf or MOs_data) wraps a GTO basis.

            Basis-exponent / basis-coefficient masking is only meaningful for
            GTO-family AOs (cart / sphe, molecular or periodic). PW / Spline
            / composite leaves carry no primitive ``exponents`` /
            ``coefficients`` fields, so the masking step is a no-op for them
            and the geminal_data.ao_exponents_* accessors raise
            NotImplementedError -- skip those blocks instead.
            """
            if isinstance(orb, MOs_data):
                return isinstance(orb.aos_data, AOs_GTO_base_data)
            return isinstance(orb, AOs_GTO_base_data)

        jastrow_data = self.jastrow_data
        jastrow_updates = {}
        if jastrow_data is not None:
            j1_block, changed = _maybe_mask(jastrow_data.jastrow_one_body_data, opt_J1_param)
            if changed:
                jastrow_updates["jastrow_one_body_data"] = j1_block

            j2_block, changed = _maybe_mask(jastrow_data.jastrow_two_body_data, opt_J2_param)
            if changed:
                jastrow_updates["jastrow_two_body_data"] = j2_block

            j3_block, changed = _maybe_mask(jastrow_data.jastrow_three_body_data, opt_J3_param)
            if changed:
                jastrow_updates["jastrow_three_body_data"] = j3_block

            if jastrow_updates:
                jastrow_data = jastrow_data.replace(**jastrow_updates)

            # DTN polynomial Jastrow: per-channel surgical masking.
            # Unlike J1/J2/J3 above (whole-block apply_diff_mask), the
            # DTN block packs three independent channels (alpha/beta/gamma)
            # in one struct, so we stop_gradient only the disabled leaves
            # rather than masking the whole jastrow_dtn_data block. The
            # cutoff lengths L_u/L_chi/L_f are static (pytree_node=False)
            # and never optimized, so they need no masking here.
            dtn = jastrow_data.jastrow_dtn_data
            if dtn is not None:
                dtn_updates = {}
                if not opt_JDTN_u_param:
                    dtn_updates["alpha_indep"] = jax.lax.stop_gradient(dtn.alpha_indep)
                if not opt_JDTN_chi_param:
                    dtn_updates["beta_indep"] = jax.lax.stop_gradient(dtn.beta_indep)
                if not opt_JDTN_f_param:
                    dtn_updates["gamma_indep"] = jax.lax.stop_gradient(dtn.gamma_indep)
                if dtn_updates:
                    jastrow_data = jastrow_data.replace(jastrow_dtn_data=dtn.replace(**dtn_updates))

            # AO basis masking for J3: stop gradient on the orbital basis when
            # it is not being optimized. Same two-case structure as the geminal
            # basis above: fully frozen -> freeze the whole orbital basis
            # (GTO / PW / spline / composite / MO) so de_L/dc does not waste
            # compute on it; partial opt -> selective GTO exp/coeff masking.
            # Independent of the J3 j_matrix masking above.
            j3d = jastrow_data.jastrow_three_body_data
            if j3d is not None:
                j3_basis_frozen = (not opt_J3_basis_exp) and (not opt_J3_basis_coeff)
                if j3_basis_frozen:
                    new_orb = _freeze_orbital_basis(j3d.orb_data)
                    jastrow_data = jastrow_data.replace(jastrow_three_body_data=j3d.replace(orb_data=new_orb))
                elif _is_gto_leaf(j3d.orb_data):
                    # Selectively freeze the per-shell radial leaf's exponents
                    # and/or coefficients channel.
                    orb = j3d.orb_data
                    aos = orb.aos_data if isinstance(orb, MOs_data) else orb
                    new_exp = aos.radial.exponents if opt_J3_basis_exp else jax.lax.stop_gradient(aos.radial.exponents)
                    new_coef = aos.radial.coefficients if opt_J3_basis_coeff else jax.lax.stop_gradient(aos.radial.coefficients)
                    new_aos = aos.replace(radial=aos.radial.replace(exponents=new_exp, coefficients=new_coef))
                    if isinstance(orb, MOs_data):
                        # mo_coefficients are never a variational block, so freeze
                        # them even while the GTO exp/coeff are being optimized.
                        new_orb = orb.replace(aos_data=new_aos, mo_coefficients=jax.lax.stop_gradient(orb.mo_coefficients))
                    else:
                        new_orb = new_aos
                    jastrow_data = jastrow_data.replace(jastrow_three_body_data=j3d.replace(orb_data=new_orb))

        geminal_data = self.geminal_data
        geminal_updates = {}
        if geminal_data is not None:
            from .determinant import AGPn_data, SD_data

            geminal_masked, changed = _maybe_mask(geminal_data, opt_geminal)
            if changed:
                # AGP_data exposes ``lambda_matrix`` directly. AGPn / SD do
                # not -- they carry MO coefficients (in ``orb_data_*_spin``)
                # and (for AGPn) ``lambda_diag``. Use the masked geminal
                # whole-instance so the stop_gradient survives, instead of
                # extracting a single ``lambda_matrix`` field that does not
                # exist on these subclasses.
                if isinstance(geminal_data, (AGPn_data, SD_data)):
                    geminal_data = geminal_masked
                else:
                    geminal_updates["lambda_matrix"] = geminal_masked.lambda_matrix

            if geminal_updates:
                geminal_data = geminal_data.replace(**geminal_updates)

            # AO basis masking for Geminal: stop gradient on the orbital basis
            # when it is not being optimized. Two cases:
            #
            #  * basis fully frozen (neither exp nor coeff optimized -- the
            #    default, e.g. Jastrow-only optimization): the orbital carries
            #    NO variational parameter (mo_coefficients is never a block, and
            #    PW / spline / composite basis opt is not exposed for the
            #    geminal). Freeze every basis-payload array via
            #    ``_freeze_orbital_basis`` so de_L/dc does not waste compute --
            #    or OOM on B-spline control points -- differentiating it. Works
            #    for GTO / PW / spline / composite / MO uniformly.
            #  * partial basis opt (exactly one of exp / coeff optimized): only
            #    the GTO leaf supports per-array basis opt, so keep the existing
            #    selective masking (non-GTO leaves have no exp/coeff to split).
            geminal_basis_frozen = (not opt_lambda_basis_exp) and (not opt_lambda_basis_coeff)
            # For AGPn / SD direct MO opt with ``opt_geminal=True``,
            # ``mo_coefficients`` IS the variational object -- the default
            # ``_freeze_orbital_basis`` would stop_gradient it, killing the
            # parameter gradient. Use a tailored freeze that keeps
            # ``mo_coefficients`` differentiable while still freezing the
            # underlying AO basis arrays (exponents, coefficients, etc.).
            _agpn_sd_keep_mo = isinstance(geminal_data, (AGPn_data, SD_data)) and opt_geminal
            if geminal_basis_frozen:
                if _agpn_sd_keep_mo:
                    geminal_data = geminal_data.replace(
                        orb_data_up_spin=_freeze_orbital_basis_keep_mo_coefficients(geminal_data.orb_data_up_spin),
                        orb_data_dn_spin=_freeze_orbital_basis_keep_mo_coefficients(geminal_data.orb_data_dn_spin),
                    )
                else:
                    geminal_data = geminal_data.replace(
                        orb_data_up_spin=_freeze_orbital_basis(geminal_data.orb_data_up_spin),
                        orb_data_dn_spin=_freeze_orbital_basis(geminal_data.orb_data_dn_spin),
                    )
            elif _is_gto_leaf(geminal_data.orb_data_up_spin) and _is_gto_leaf(geminal_data.orb_data_dn_spin):
                # Selectively freeze the per-shell radial leaf's exponents and/or
                # coefficients channel independently of the direct-MO flag.
                # SD/AGPn can optimize MO coefficients and radial parameters
                # together; AGP's orbital coefficients remain fixed.
                def _freeze_selected_radial(
                    orb: AOs_GTO_base_data | MOs_data,
                ) -> AOs_GTO_base_data | MOs_data:
                    aos = orb.aos_data if isinstance(orb, MOs_data) else orb
                    new_exp = aos.radial.exponents if opt_lambda_basis_exp else jax.lax.stop_gradient(aos.radial.exponents)
                    new_coef = (
                        aos.radial.coefficients if opt_lambda_basis_coeff else jax.lax.stop_gradient(aos.radial.coefficients)
                    )
                    new_aos = aos.replace(radial=aos.radial.replace(exponents=new_exp, coefficients=new_coef))
                    if isinstance(orb, MOs_data):
                        new_mo = orb.mo_coefficients if _agpn_sd_keep_mo else jax.lax.stop_gradient(orb.mo_coefficients)
                        return orb.replace(aos_data=new_aos, mo_coefficients=new_mo)
                    return new_aos

                up_orb = _freeze_selected_radial(geminal_data.orb_data_up_spin)
                dn_orb = _freeze_selected_radial(geminal_data.orb_data_dn_spin)
                geminal_data = geminal_data.replace(
                    orb_data_up_spin=up_orb,
                    orb_data_dn_spin=dn_orb,
                )

        if jastrow_data is not self.jastrow_data or geminal_data is not self.geminal_data:
            return Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)

        return self

    def accumulate_position_grad(self, grad_wavefunction: "Wavefunction_data") -> jax.Array | float:
        """Aggregate position gradients from geminal and Jastrow parts."""
        grad = 0.0
        if self.geminal_data is not None and grad_wavefunction.geminal_data is not None:
            grad += self.geminal_data.accumulate_position_grad(grad_wavefunction.geminal_data)
        if self.jastrow_data is not None and grad_wavefunction.jastrow_data is not None:
            grad += self.jastrow_data.accumulate_position_grad(grad_wavefunction.jastrow_data)
        return grad

    def accumulate_strain_grad(self, grad_wavefunction: "Wavefunction_data") -> jax.Array:
        """Aggregate Cauchy-Born strain cotangents from geminal and Jastrow parts."""
        sigma_tilde = jnp.zeros((3, 3), dtype=jnp.float64)
        if self.geminal_data is not None and grad_wavefunction.geminal_data is not None:
            sigma_tilde += self.geminal_data.accumulate_strain_grad(grad_wavefunction.geminal_data)
        if self.jastrow_data is not None and grad_wavefunction.jastrow_data is not None:
            sigma_tilde += self.jastrow_data.accumulate_strain_grad(grad_wavefunction.jastrow_data)
        return sigma_tilde

    def _apply_strain(self, F: jax.Array) -> "Wavefunction_data":
        """Return a new ``Wavefunction_data`` with ``F`` applied across geminal and Jastrow."""
        new_geminal = self.geminal_data
        if new_geminal is not None:
            new_geminal = new_geminal._apply_strain(F)
        new_jastrow = self.jastrow_data
        if new_jastrow is not None:
            new_jastrow = new_jastrow._apply_strain(F)
        return self.replace(geminal_data=new_geminal, jastrow_data=new_jastrow)

    def collect_param_grads(self, grad_wavefunction: "Wavefunction_data") -> dict[str, object]:
        """Collect parameter gradients from Jastrow and Geminal into a flat dict."""
        grads: dict[str, object] = {}
        if self.jastrow_data is not None and grad_wavefunction.jastrow_data is not None:
            grads.update(self.jastrow_data.collect_param_grads(grad_wavefunction.jastrow_data))
        if self.geminal_data is not None and grad_wavefunction.geminal_data is not None:
            grads.update(self.geminal_data.collect_param_grads(grad_wavefunction.geminal_data))
        return grads

    def flatten_param_grads(self, param_grads: dict[str, object], num_walkers: int) -> dict[str, np.ndarray]:
        """Return parameter gradients as numpy arrays ready for storage."""
        flat: dict[str, np.ndarray] = {}
        for name, param_grad in param_grads.items():
            flat[name] = np.array(param_grad)
        return flat

    def get_variational_blocks(
        self,
        *,
        state_token: StateToken,
        opt_J1_param: bool = True,
        opt_J2_param: bool = True,
        opt_J3_param: bool = True,
        opt_geminal: bool = False,
        opt_J3_basis_exp: bool = False,
        opt_J3_basis_coeff: bool = False,
        opt_lambda_basis_exp: bool = False,
        opt_lambda_basis_coeff: bool = False,
        opt_JDTN_u_param: bool = True,
        opt_JDTN_chi_param: bool = True,
        opt_JDTN_f_param: bool = True,
    ) -> list[VariationalParameterBlock]:
        """Collect variational parameter blocks from Jastrow and Geminal parts.

        Each block corresponds to a contiguous group of variational parameters
        (e.g., J1, J2, J3 matrix, NN Jastrow, lambda matrix). This method only exposes the
        parameter arrays; the corresponding gradients are handled on the MCMC side.
        """
        if self.geminal_data is not None and isinstance(self.geminal_data.orb_data_up_spin, TiledOrbitalSet_data):
            if opt_geminal or opt_lambda_basis_exp or opt_lambda_basis_coeff:
                raise NotImplementedError(
                    "tiled orbital/geminal optimization is frozen; set opt_geminal and orbital-basis optimization flags to false."
                )
        token = _validate_state_token(state_token)
        blocks: list[VariationalParameterBlock] = []

        def _append_block(
            *,
            name: str,
            values: npt.ArrayLike,
            sync_scope: SynchronizationScope,
            tied: bool = False,
            tangent_spec: ParameterTangentSpec | None = None,
        ) -> None:
            if np.issubdtype(np.asarray(values).dtype, np.complexfloating):
                field: ParameterField = "complex"
                derivative_policy: DerivativePolicy = "wirtinger_pair" if tied else "holomorphic"
            else:
                field = "real"
                derivative_policy = "direct_real"
            spec = ParameterTangentSpec.euclidean(state_token=token) if tangent_spec is None else tangent_spec
            blocks.append(
                VariationalParameterBlock.build(
                    name=name,
                    values=values,
                    field=field,
                    derivative_policy=derivative_policy,
                    sync_scope=sync_scope,
                    tangent_spec=spec,
                    state_token=token,
                )
            )

        def _s_stiefel_spec(orb_data: MOs_data) -> ParameterTangentSpec:
            metric_sqrt, metric_inv_sqrt = _s_stiefel_metric_factors_for_aos(orb_data.aos_data)
            return ParameterTangentSpec.s_stiefel_rows(
                metric_sqrt=metric_sqrt,
                metric_inv_sqrt=metric_inv_sqrt,
                state_token=token,
            )

        # Jastrow part
        if self.jastrow_data is not None:
            if opt_J1_param and self.jastrow_data.jastrow_one_body_data is not None:
                j1 = self.jastrow_data.jastrow_one_body_data.jastrow_1b_param
                j1_arr = np.asarray(j1)
                _append_block(name="j1_param", values=j1_arr, sync_scope="world")

            if opt_J2_param and self.jastrow_data.jastrow_two_body_data is not None:
                j2 = self.jastrow_data.jastrow_two_body_data.jastrow_2b_param
                j2_arr = np.asarray(j2)
                _append_block(name="j2_param", values=j2_arr, sync_scope="world")

            if opt_J3_param and self.jastrow_data.jastrow_three_body_data is not None:
                j3 = self.jastrow_data.jastrow_three_body_data.j_matrix
                j3_arr = np.asarray(j3)
                _append_block(
                    name="j3_matrix",
                    values=j3_arr,
                    sync_scope="world",
                    tangent_spec=ParameterTangentSpec.with_linear_tie(
                        _symmetric_real_linear_tie(tuple(j3_arr.shape)),
                        state_token=token,
                    ),
                )

            # J3 AO basis blocks --- delegate to the AO leaf protocol.
            # The leaf returns blocks named ``j3_basis_exp``
            # / ``j3_basis_coeff`` directly (because we pass
            # ``block_name_prefix="j3_"``) with one degree of freedom per
            # shell primitive. PW / Spline leaves inherit the
            # no-op default and return an empty list, so this block
            # construction trivially skips them.
            if (opt_J3_basis_exp or opt_J3_basis_coeff) and self.jastrow_data.jastrow_three_body_data is not None:
                j3_data = self.jastrow_data.jastrow_three_body_data
                j3_aos = _get_aos_data(j3_data.orb_data)
                blocks.extend(
                    j3_aos.get_basis_variational_blocks(
                        state_token=token,
                        sync_scope="world",
                        opt_basis_exp=opt_J3_basis_exp,
                        opt_basis_coeff=opt_J3_basis_coeff,
                        block_name_prefix="j3_",
                    )
                )

            # DTN polynomial Jastrow variational blocks. Each active channel
            # contributes one independent-coefficient block:
            #
            #   opt_JDTN_u_param   -> alpha_indep  (e-e,   "u",   J2 analogue)
            #   opt_JDTN_chi_param -> beta_indep   (e-N,   "chi", J1 analogue)
            #   opt_JDTN_f_param   -> gamma_indep  (e-e-N, "f",   J3 analogue)
            #
            # The cutoff lengths L_u / L_chi / L_f are static (never
            # optimized) constants that also feed the pre-built constraint
            # inflation matrices M_alpha / M_beta / M_gamma, so no cutoff
            # variational block is ever added.
            if self.jastrow_data.jastrow_dtn_data is not None:
                dtn = self.jastrow_data.jastrow_dtn_data
                if opt_JDTN_u_param and dtn.use_u and dtn.alpha_indep.size > 0:
                    arr = np.asarray(dtn.alpha_indep)
                    _append_block(name="j3_poly_alpha", values=arr, sync_scope="world")
                if opt_JDTN_chi_param and dtn.use_chi and dtn.beta_indep.size > 0:
                    arr = np.asarray(dtn.beta_indep)
                    _append_block(name="j3_poly_beta", values=arr, sync_scope="world")
                if opt_JDTN_f_param and dtn.use_f and dtn.gamma_indep.size > 0:
                    arr = np.asarray(dtn.gamma_indep)
                    _append_block(name="j3_poly_gamma", values=arr, sync_scope="world")

        # Geminal part
        if opt_geminal and self.geminal_data is not None:
            # Direct MO opt: ``AGPn_data`` and ``SD_data`` expose
            # ``mo_coefficients_up`` (and ``lambda_diag`` for AGPn) instead
            # of the dense ``lambda_matrix``.
            from .determinant import AGPn_data, SD_data

            if isinstance(self.geminal_data, (AGPn_data, SD_data)):
                if self.geminal_data.orb_data_up_spin.identity or self.geminal_data.orb_data_dn_spin.identity:
                    raise NotImplementedError(
                        "opt_geminal=True is unavailable for implicit identity MOs. "
                        "The tiled B-spline representation is for frozen nodal surfaces; rebuild with dense MO coefficients to optimize them."
                    )
                mo_up = np.asarray(self.geminal_data.orb_data_up_spin.mo_coefficients)
                # A fully polarized SD has no parameters in its empty spin
                # channel. Keep zero-sized storage out of the optimizer while
                # retaining the shared nonempty VariationalParameterBlock guard.
                if mo_up.size > 0:
                    _append_block(
                        name="mo_coefficients_up",
                        values=mo_up,
                        sync_scope="sampler",
                        tied=bool(self.geminal_data.time_reversal),
                        tangent_spec=_s_stiefel_spec(self.geminal_data.orb_data_up_spin),
                    )
                # Open-shell / UHF (``time_reversal=False``): the dn
                # channel is a variationally-independent set of MOs
                # (the closed-shell singlet has ``C_dn = conj(C_up)``
                # derived in apply_block_update, so the dn block is
                # suppressed there).
                if not bool(self.geminal_data.time_reversal):
                    mo_dn = np.asarray(self.geminal_data.orb_data_dn_spin.mo_coefficients)
                    if mo_dn.size > 0:
                        _append_block(
                            name="mo_coefficients_dn",
                            values=mo_dn,
                            sync_scope="sampler",
                            tangent_spec=_s_stiefel_spec(self.geminal_data.orb_data_dn_spin),
                        )
                if isinstance(self.geminal_data, AGPn_data):
                    diag_arr = np.asarray(self.geminal_data.lambda_diag, dtype=np.float64)
                    _append_block(name="lambda_diag", values=diag_arr, sync_scope="sampler")
                    # Open-shell unpaired block. Variational on AGPn;
                    # absent from canonical SD_data, so we only
                    # expose it here. The block is empty (size 0) for
                    # closed-shell inputs; skip in that case so the
                    # optimiser does not see a zero-sized vector.
                    unp_arr = np.asarray(self.geminal_data.lambda_unpaired)
                    if unp_arr.size > 0:
                        _append_block(name="lambda_unpaired", values=unp_arr, sync_scope="sampler")
            else:
                # AGP_data: full ``lambda_matrix`` block (original path).
                lam = self.geminal_data.lambda_matrix
                lam_arr = np.asarray(lam)
                lambda_tied = bool(self.geminal_data.time_reversal)
                lambda_spec = ParameterTangentSpec.euclidean(state_token=token)
                if lambda_tied:
                    tie_builder = (
                        _hermitian_paired_linear_tie
                        if np.issubdtype(lam_arr.dtype, np.complexfloating)
                        else _symmetric_paired_linear_tie
                    )
                    lambda_spec = ParameterTangentSpec.with_linear_tie(
                        tie_builder(
                            tuple(lam_arr.shape),
                            n_paired=int(self.geminal_data.orb_num_dn),
                        ),
                        state_token=token,
                    )
                _append_block(
                    name="lambda_matrix",
                    values=lam_arr,
                    sync_scope="sampler",
                    tied=lambda_tied,
                    tangent_spec=lambda_spec,
                )

        # Geminal AO basis blocks (up + dn concatenated into single
        # blocks for the optimiser, but each side's payload is sourced
        # via the per-leaf protocol).
        #
        # The per-leaf
        # :meth:`AOs_base_data.get_basis_variational_blocks` is called
        # on the up and dn leaves separately --- each returns blocks
        # named ``basis_exp`` / ``basis_coeff`` (we pass no prefix
        # here, since the upper layer renames to
        # ``lambda_basis_*``). The optimiser still sees one
        # concatenated ``lambda_basis_exp`` (and / or
        # ``lambda_basis_coeff``) block per type --- the up + dn leaf
        # blocks concatenated --- so the block-name surface is unchanged. The Geminal_data side of
        # apply_block_update (in determinant.py) handles the
        # corresponding split + per-leaf delegation.
        if self.geminal_data is not None and (opt_lambda_basis_exp or opt_lambda_basis_coeff):
            up_aos = _get_aos_data(self.geminal_data.orb_data_up_spin)
            dn_aos = _get_aos_data(self.geminal_data.orb_data_dn_spin)

            def _collect_payload(
                aos: AOs_base_data,
                suffix: Literal["basis_exp", "basis_coeff"],
            ) -> np.ndarray:
                """Pull the value array for ``basis_{exp,coeff}`` from a leaf.

                A single GTO leaf publishes exactly one block named ``suffix``;
                a composite (pgto(+)PW) publishes per-child ``child{i}_{suffix}``
                blocks (only its GTO children -- PW / Spline children return
                nothing). We concatenate every block whose name ends with
                ``suffix`` in emission order, matching the child-ordered layout
                that ``Geminal_data.ao_{exponents,coefficients}_*`` and
                ``collect_param_grads`` use on the gradient side. Returns an
                empty ``float64`` array when the leaf owns no such block.
                """
                parts = [
                    np.asarray(blk.values, dtype=np.float64)
                    for blk in aos.get_basis_variational_blocks(
                        state_token=token,
                        sync_scope="sampler",
                        opt_basis_exp=(suffix == "basis_exp"),
                        opt_basis_coeff=(suffix == "basis_coeff"),
                    )
                    if blk.name.endswith(suffix)
                ]
                return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)

            # No symmetrise callback: each leaf's per-shell radial block is
            # already one DOF per shell-primitive, so the concatenated up + dn
            # block is unique by construction.
            if opt_lambda_basis_exp:
                lam_exp_arr = np.concatenate([_collect_payload(up_aos, "basis_exp"), _collect_payload(dn_aos, "basis_exp")])
                if lam_exp_arr.size > 0:
                    _append_block(name="lambda_basis_exp", values=lam_exp_arr, sync_scope="sampler")

            if opt_lambda_basis_coeff:
                lam_coeff_arr = np.concatenate(
                    [_collect_payload(up_aos, "basis_coeff"), _collect_payload(dn_aos, "basis_coeff")]
                )
                if lam_coeff_arr.size > 0:
                    _append_block(name="lambda_basis_coeff", values=lam_coeff_arr, sync_scope="sampler")

        return blocks


@jit
def evaluate_ln_wavefunction(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    r"""Evaluate the logarithm of ``|wavefunction|`` (:math:`\ln |\Psi|`).

    Add the Jastrow exponent to the stable log absolute determinant, without
    materializing an amplitude that may underflow or overflow under a harmless
    constant wavefunction normalization.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Scalar log-value of the wavefunction magnitude.
    """
    # NOTE: do not pre-cast r_*_carts. They are forwarded unchanged to
    # ``compute_Jastrow_part`` and ``compute_ln_det_geminal_all_elements`` (which
    # ultimately reach the AO kernels that reconstruct ``r - R`` in float64);
    # a wrapper-level downcast would defeat that precision guard. The scalar
    # arithmetic below uses ``Jastrow_part`` and ``ln_determinant_part`` which
    # are explicitly cast to ``wf_eval``.
    Jastrow_part = compute_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    ln_determinant_part = compute_ln_det_geminal_all_elements(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # Consumer-zone explicit cast: both terms cast to wf_eval before addition,
    # never relying on implicit fp32 + fp64 -> fp64 promotion.
    return cast_to_zone(Jastrow_part, "wf_eval") + cast_to_zone(ln_determinant_part, "wf_eval")


def evaluate_ln_wavefunction_fast(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inv: jax.Array,
) -> float:
    r"""Evaluate :math:`\ln|\Psi|` using pre-computed ``geminal_inv`` in gradients.

    Identical to :func:`evaluate_ln_wavefunction` in the forward direction.
    The backward pass (used when computing :math:`\partial\ln\Psi/\partial c`
    via JAX autodiff) replaces the fresh LU decomposition of the geminal matrix
    with ``geminal_inv`` -- the Sherman-Morrison running inverse -- so that
    near-singular configurations (``epsilon_AS > 0``) do not produce NaN
    gradients.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons ``(n_dn, 3)``.
        geminal_inv: Pre-computed inverse geminal matrix ``(N_up, N_up)`` from
            the Sherman-Morrison running update, valid at the supplied
            ``(r_up_carts, r_dn_carts)``.

    Returns:
        Scalar log-value of the wavefunction magnitude.

    Warning:
        ``geminal_inv`` **must** equal ``G(r_up_carts, r_dn_carts)^{-1}``
        exactly at the supplied electron positions.  Correctness is only
        guaranteed when the inverse is maintained via **single-electron
        (rank-1) Sherman-Morrison updates** starting from a freshly
        initialized LU inverse -- the pattern used in the MCMC loop.
        Passing an inverse from a different configuration silently produces
        incorrect parameter gradients (``O_matrix`` / SR).
    """
    # r_*_carts forwarded unchanged (see ``evaluate_ln_wavefunction`` for rationale).
    Jastrow_part = compute_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    ln_det = compute_ln_det_geminal_all_elements_fast(
        wavefunction_data.geminal_data,
        r_up_carts,
        r_dn_carts,
        geminal_inv,
    )

    # Consumer-zone explicit cast: both terms cast to wf_eval before addition.
    return cast_to_zone(Jastrow_part, "wf_eval") + cast_to_zone(ln_det, "wf_eval")


@jit
def evaluate_wavefunction(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float | complex:
    """Evaluate the wavefunction ``Psi`` at given electron coordinates.

    The method is for evaluate wavefunction (Psi) at ``(r_up_carts, r_dn_carts)`` and
    returns ``exp(Jastrow) * Determinant``. Inputs are coerced to the determinant
    zone dtype ``jax.Array`` to match other compute utilities.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Complex or real wavefunction value.
    """
    # r_*_carts forwarded unchanged (see ``evaluate_ln_wavefunction`` for rationale).
    Jastrow_part = compute_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    Determinant_part = compute_det_geminal_all_elements(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # Consumer-zone explicit cast: both factors cast to wf_eval before multiplication.
    return jnp.exp(cast_to_zone(Jastrow_part, "wf_eval")) * cast_to_zone(Determinant_part, "wf_eval")


def evaluate_jastrow(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    r"""Evaluate the Jastrow factor :math:`\exp(J)` at the given coordinates.

    The method is for evaluate the Jastrow part of the wavefunction (Psi) at
    ``(r_up_carts, r_dn_carts)``. The returned value already includes the
    exponential, i.e., ``exp(J)``.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Real Jastrow factor ``exp(J)``.
    """
    # r_*_carts forwarded unchanged (see ``evaluate_ln_wavefunction`` for rationale).
    Jastrow_part = compute_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    return jnp.exp(Jastrow_part)


def evaluate_determinant(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    """Evaluate the determinant (Geminal) part of the wavefunction.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Determinant value evaluated at the supplied coordinates.
    """
    # r_*_carts forwarded unchanged (see ``evaluate_ln_wavefunction`` for rationale).
    Determinant_part = compute_det_geminal_all_elements(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    return Determinant_part


@jit
def compute_kinetic_energy(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float | complex:
    """Compute kinetic energy using analytic gradients and Laplacians.

    The method is for computing kinetic energy of the given WF at
    ``(r_up_carts, r_dn_carts)`` and fully exploits the JAX library for the
    kinetic energy calculation. The Jastrow and determinant gradients and
    Laplacians are cast to the ``wf_kinetic`` zone dtype before the kinetic
    energy is assembled.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Kinetic energy evaluated for the supplied configuration.
    """
    grad_J_up, grad_J_dn, lap_J_up, lap_J_dn = compute_grads_and_laplacian_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn = compute_grads_and_laplacian_ln_Det(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # Dtype follows the geminal lambda AND the
    # actual grad/lap dtypes so the kinetic-energy assembly preserves Im on
    # complex Bloch. At twist k != 0 the MO coefficients can be complex
    # while lambda remains real (RHF singlet / closed shell); ``cast_to_zone``
    # always folds x.dtype into the promotion so complex grad_ln_D survives.
    # The algebraic ``v dot v`` form below is correct for complex too (NOT
    # |v|^2 -- the local-energy formula is the algebraic ratio T psi / psi).
    # Bit-identical at Gamma where everything is real.
    lambda_dtype = wavefunction_data.geminal_data._determinant_dtype_jnp

    # Explicitly cast jastrow_grad_lap and det_grad_lap zone values to wf_kinetic
    # zone dtype before assembling T_L; do not rely on JAX implicit promotion.
    grad_J_up = cast_to_zone(grad_J_up, "wf_kinetic", lambda_dtype)
    grad_J_dn = cast_to_zone(grad_J_dn, "wf_kinetic", lambda_dtype)
    lap_J_up = cast_to_zone(lap_J_up, "wf_kinetic", lambda_dtype)
    lap_J_dn = cast_to_zone(lap_J_dn, "wf_kinetic", lambda_dtype)
    grad_ln_D_up = cast_to_zone(grad_ln_D_up, "wf_kinetic", lambda_dtype)
    grad_ln_D_dn = cast_to_zone(grad_ln_D_dn, "wf_kinetic", lambda_dtype)
    lap_ln_D_up = cast_to_zone(lap_ln_D_up, "wf_kinetic", lambda_dtype)
    lap_ln_D_dn = cast_to_zone(lap_ln_D_dn, "wf_kinetic", lambda_dtype)

    # compute kinetic energy
    L = (
        1.0
        / 2.0
        * (
            -(jnp.sum(lap_J_up) + jnp.sum(lap_J_dn) + jnp.sum(lap_ln_D_up) + jnp.sum(lap_ln_D_dn))
            - (
                jnp.sum((grad_J_up + grad_ln_D_up) * (grad_J_up + grad_ln_D_up))
                + jnp.sum((grad_J_dn + grad_ln_D_dn) * (grad_J_dn + grad_ln_D_dn))
            )
        )
    )

    return L


@jit
def _compute_kinetic_energy_auto(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float | complex:
    """Sum the phase-aware autodiff kinetic reference over both spins."""
    kinetic_up, kinetic_dn = _compute_kinetic_energy_all_elements_auto(wavefunction_data, r_up_carts, r_dn_carts)
    return jnp.sum(kinetic_up) + jnp.sum(kinetic_dn)


def _compute_kinetic_energy_debug(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    *,
    diff_h: float,
) -> float | complex:
    """Sum the phase-preserving kinetic finite differences over both spins."""
    kinetic_up, kinetic_dn = _compute_kinetic_energy_all_elements_debug(
        wavefunction_data, r_up_carts, r_dn_carts, diff_h=diff_h
    )
    return np.sum(kinetic_up) + np.sum(kinetic_dn)


def _compute_kinetic_energy_all_elements_debug(
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    *,
    diff_h: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the local kinetic energy per electron, separated by spin.

    For each electron and Cartesian axis, use the fourth-order stencil
    ``(-Psi(R+2h) + 16 Psi(R+h) - 30 Psi(R) + 16 Psi(R-h) - Psi(R-2h)) / (12 h**2)``.
    Each element is ``-sum(axis_second_derivatives) / (2 Psi(R))``. Differencing
    the value itself retains the wavefunction's complex phase.

    Coordinates and the caller-selected step ``diff_h`` are in Bohr. Inputs
    are copied before displacement. Check convergence by reducing the step:
    the stencil has fourth-order truncation error, while rounding errors grow
    as the step shrinks. Use smooth configurations away from wavefunction nodes,
    and steps resolvable in the value evaluator's active precision.

    This test-only oracle uses production wavefunction values to check
    derivative assembly independently of the analytic and AD kernels.
    """
    if not np.isfinite(diff_h) or diff_h <= 0.0:
        raise ValueError("diff_h must be finite and positive.")
    r_up = np.asarray(r_up_carts, dtype=np.float64)
    r_dn = np.asarray(r_dn_carts, dtype=np.float64)
    psi_center = np.asarray(evaluate_wavefunction(wavefunction_data, r_up, r_dn)).item()
    if not np.isfinite(psi_center) or psi_center == 0.0:
        raise ValueError("The finite-difference kinetic oracle requires finite, nonzero Psi at the center.")

    kinetic_up = np.empty(len(r_up), dtype=type(psi_center))
    kinetic_dn = np.empty(len(r_dn), dtype=type(psi_center))
    for spin, kinetic in enumerate((kinetic_up, kinetic_dn)):
        for electron in range(len(kinetic)):
            laplacian = 0.0
            for axis in range(3):
                shifted_up = r_up.copy()
                shifted_dn = r_dn.copy()
                shifted = shifted_up if spin == 0 else shifted_dn
                center_coordinate = shifted[electron, axis]
                shifted[electron, axis] = center_coordinate + diff_h
                psi_plus = np.asarray(evaluate_wavefunction(wavefunction_data, shifted_up, shifted_dn)).item()
                shifted[electron, axis] = center_coordinate - diff_h
                psi_minus = np.asarray(evaluate_wavefunction(wavefunction_data, shifted_up, shifted_dn)).item()
                shifted[electron, axis] = center_coordinate + 2.0 * diff_h
                psi_plus2 = np.asarray(evaluate_wavefunction(wavefunction_data, shifted_up, shifted_dn)).item()
                shifted[electron, axis] = center_coordinate - 2.0 * diff_h
                psi_minus2 = np.asarray(evaluate_wavefunction(wavefunction_data, shifted_up, shifted_dn)).item()
                laplacian += (-psi_plus2 + 16.0 * psi_plus - 30.0 * psi_center + 16.0 * psi_minus - psi_minus2) / (
                    12.0 * diff_h**2
                )
            kinetic[electron] = -0.5 * laplacian / psi_center
    return kinetic_up, kinetic_dn


@jit
def _compute_kinetic_energy_all_elements_auto(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Differentiate a locally normalized Psi and return -laplacian(Psi) / (2 Psi) per electron."""
    n_up = r_up_carts.shape[0]
    positions = jnp.concatenate((cast_to_zone(r_up_carts, "wf_kinetic"), cast_to_zone(r_dn_carts, "wf_kinetic")))

    def log_polar(points: jax.Array) -> tuple[jax.Array, jax.Array]:
        r_up, r_dn = points[:n_up], points[n_up:]
        geminal = compute_geminal_all_elements(wavefunction_data.geminal_data, r_up, r_dn)
        phase, log_determinant = jnp.linalg.slogdet(cast_to_zone(geminal, "det_eval"))
        jastrow = compute_Jastrow_part(wavefunction_data.jastrow_data, r_up, r_dn)
        log_magnitude = cast_to_zone(jastrow, "wf_eval") + cast_to_zone(log_determinant, "wf_eval")
        return cast_to_zone(phase, "wf_eval"), log_magnitude

    # An electron-coordinate constant cancels in the normalized Laplacian.
    # Subtract it before exponentiation so the unscaled Psi need not be representable.
    _, center_log_magnitude = log_polar(positions)
    log_normalization = jax.lax.stop_gradient(center_log_magnitude)

    def psi(points: jax.Array) -> jax.Array:
        phase, log_magnitude = log_polar(points)
        return phase * jnp.exp(log_magnitude - log_normalization)

    hess = jax.jacfwd(jax.jacfwd(psi))(positions)
    kinetic = -0.5 * jnp.einsum("ijij->i", hess) / psi(positions)
    kinetic = cast_to_zone(kinetic, "wf_kinetic")
    return kinetic[:n_up], kinetic[n_up:]


@jit
def compute_kinetic_energy_all_elements(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    """Analytic-derivative kinetic energy per electron (matches auto output shape).

    Returns the per-electron kinetic energy using analytic gradients/Laplacians of
    both Jastrow and determinant parts. Shapes align with
    ``_compute_kinetic_energy_all_elements_auto``.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Tuple of two ``jax.Array`` objects containing per-electron kinetic energies
        for spin-up and spin-down electrons, respectively.
    """
    # --- Jastrow contributions (per-electron Laplacians) ---
    grad_J_up, grad_J_dn, lap_J_up, lap_J_dn = compute_grads_and_laplacian_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # --- Determinant contributions (per-electron Laplacians) ---
    grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn = compute_grads_and_laplacian_ln_Det(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # Promote against the geminal lambda dtype AND
    # the actual grad/lap dtypes so the kinetic-energy assembly preserves Im
    # on complex Bloch. At twist k != 0 the MO coefficients can be complex
    # while lambda remains real (RHF singlet / closed shell); ``cast_to_zone``
    # always folds x.dtype into the promotion so complex grad_ln_D survives.
    lambda_dtype = wavefunction_data.geminal_data._determinant_dtype_jnp

    # Cast jastrow_grad_lap / det_grad_lap zone values to wf_kinetic dtype.
    grad_J_up = cast_to_zone(grad_J_up, "wf_kinetic", lambda_dtype)
    grad_J_dn = cast_to_zone(grad_J_dn, "wf_kinetic", lambda_dtype)
    lap_J_up = cast_to_zone(lap_J_up, "wf_kinetic", lambda_dtype)
    lap_J_dn = cast_to_zone(lap_J_dn, "wf_kinetic", lambda_dtype)
    grad_ln_D_up = cast_to_zone(grad_ln_D_up, "wf_kinetic", lambda_dtype)
    grad_ln_D_dn = cast_to_zone(grad_ln_D_dn, "wf_kinetic", lambda_dtype)
    lap_ln_D_up = cast_to_zone(lap_ln_D_up, "wf_kinetic", lambda_dtype)
    lap_ln_D_dn = cast_to_zone(lap_ln_D_dn, "wf_kinetic", lambda_dtype)

    # --- Assemble kinetic energy per electron ---
    grad_ln_Psi_up = grad_J_up + grad_ln_D_up
    grad_ln_Psi_dn = grad_J_dn + grad_ln_D_dn

    lap_ln_Psi_up = lap_J_up + lap_ln_D_up
    lap_ln_Psi_dn = lap_J_dn + lap_ln_D_dn

    kinetic_energy_all_elements_up = -0.5 * (lap_ln_Psi_up + jnp.sum(grad_ln_Psi_up**2, axis=1))
    kinetic_energy_all_elements_dn = -0.5 * (lap_ln_Psi_dn + jnp.sum(grad_ln_Psi_dn**2, axis=1))

    return (kinetic_energy_all_elements_up, kinetic_energy_all_elements_dn)


@jit
def compute_kinetic_energy_all_elements_fast_update(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inverse: jax.Array,
) -> jax.Array:
    """Kinetic energy per electron using a precomputed geminal inverse.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons ``(n_dn, 3)``.
        geminal_inverse: Pre-computed inverse geminal matrix ``(N_up, N_up)``
            valid at the supplied ``(r_up_carts, r_dn_carts)``.

    Returns:
        Tuple of per-electron kinetic energies (up, down).

    Warning:
        ``geminal_inverse`` **must** equal ``G(r_up_carts, r_dn_carts)^{-1}``
        exactly at the supplied electron positions.  Correctness is only
        guaranteed when the inverse is maintained via **single-electron
        (rank-1) Sherman-Morrison updates** starting from a freshly
        initialized LU inverse -- the pattern used in the MCMC loop.
        Passing an inverse from a different configuration silently produces
        incorrect kinetic energy.
    """
    if geminal_inverse is None:
        raise ValueError("geminal_inverse must be provided for fast update")

    grad_J_up, grad_J_dn, lap_J_up, lap_J_dn = compute_grads_and_laplacian_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn = compute_grads_and_laplacian_ln_Det_fast(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        geminal_inverse=geminal_inverse,
    )

    # Promote against the geminal lambda dtype AND
    # the actual grad/lap dtypes so the kinetic-energy assembly preserves Im
    # on complex Bloch. At twist k != 0 the MO coefficients can be complex
    # while lambda remains real (RHF singlet / closed shell); ``cast_to_zone``
    # always folds x.dtype into the promotion so complex grad_ln_D survives.
    lambda_dtype = wavefunction_data.geminal_data._determinant_dtype_jnp

    # Cast jastrow_grad_lap / det_grad_lap zone values to wf_kinetic dtype.
    grad_J_up = cast_to_zone(grad_J_up, "wf_kinetic", lambda_dtype)
    grad_J_dn = cast_to_zone(grad_J_dn, "wf_kinetic", lambda_dtype)
    lap_J_up = cast_to_zone(lap_J_up, "wf_kinetic", lambda_dtype)
    lap_J_dn = cast_to_zone(lap_J_dn, "wf_kinetic", lambda_dtype)
    grad_ln_D_up = cast_to_zone(grad_ln_D_up, "wf_kinetic", lambda_dtype)
    grad_ln_D_dn = cast_to_zone(grad_ln_D_dn, "wf_kinetic", lambda_dtype)
    lap_ln_D_up = cast_to_zone(lap_ln_D_up, "wf_kinetic", lambda_dtype)
    lap_ln_D_dn = cast_to_zone(lap_ln_D_dn, "wf_kinetic", lambda_dtype)

    grad_ln_Psi_up = grad_J_up + grad_ln_D_up
    grad_ln_Psi_dn = grad_J_dn + grad_ln_D_dn

    lap_ln_Psi_up = lap_J_up + lap_ln_D_up
    lap_ln_Psi_dn = lap_J_dn + lap_ln_D_dn

    kinetic_energy_all_elements_up = -0.5 * (lap_ln_Psi_up + jnp.sum(grad_ln_Psi_up**2, axis=1))
    kinetic_energy_all_elements_dn = -0.5 * (lap_ln_Psi_dn + jnp.sum(grad_ln_Psi_dn**2, axis=1))

    return (kinetic_energy_all_elements_up, kinetic_energy_all_elements_dn)


def _compute_kinetic_energy_all_elements_fast_update_debug(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    """Debug helper that builds geminal inverse then calls the fast update path."""
    return compute_kinetic_energy_all_elements(
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )


# ---------------------------------------------------------------------------
# Per-electron kinetic-energy streaming state (used by GFMC projection)
# ---------------------------------------------------------------------------
#
# Maintains enough auxiliary information to advance the per-electron kinetic
# energies after a single-electron move without recomputing them from
# scratch.
#
# The state is freshly built at every branching boundary by
# ``_init_kinetic_energy_all_elements_streaming_state`` (lifetime matches
# the Sherman-Morrison ``A_old_inv``).


@struct.dataclass
class Kinetic_streaming_state:
    """Streaming state for per-electron kinetic-energy evaluation.

    Carry-fields at the current ``(r_up_carts, r_dn_carts)``:

    - ``j1_state`` / ``j2_state`` / ``j3_state``: per-Jastrow streaming
      sub-states (None when the corresponding component is absent).
    - ``det_state``: det streaming sub-state (always populated).
    - ``ke_up`` / ``ke_dn``: per-electron kinetic-energy values, pre-assembled
      so that :func:`_kinetic_energy_from_streaming_state` becomes a free
      struct-field read. Both init and advance compute these from the
      freshly produced grad/lap totals while those are still register-resident,
      avoiding a separate read kernel that would otherwise re-load all
      grad/lap totals from DRAM.
    """

    j1_state: Jastrow_one_body_streaming_state | None = struct.field(pytree_node=True, default=None)
    j2_state: Jastrow_two_body_streaming_state | None = struct.field(pytree_node=True, default=None)
    j3_state: Jastrow_three_body_streaming_state | None = struct.field(pytree_node=True, default=None)
    det_state: Det_streaming_state | None = struct.field(pytree_node=True, default=None)
    # DTN grad/lap sub-state makes the advance incremental like every other
    # channel instead of requiring a full recompute.
    dtn_grad_lap_state: "Jastrow_DTN_grad_lap_streaming_state | None" = struct.field(pytree_node=True, default=None)
    ke_up: jax.Array = struct.field(pytree_node=True, default=None)
    ke_dn: jax.Array = struct.field(pytree_node=True, default=None)


def _add_carried_dtn_grad_lap(
    grad_J_up: jax.Array,
    grad_J_dn: jax.Array,
    lap_J_up: jax.Array,
    lap_J_dn: jax.Array,
    dtn_state: "Jastrow_DTN_grad_lap_streaming_state",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Add the carried DTN leg to the J1/J2/J3 Jastrow totals.

    The carried DTN grad/lap lives in the ``jastrow_grad_lap_carry`` zone (fp64
    in both modes), so the sum is formed in that zone and not rounded back to
    ``jastrow_grad_lap``; the Rust ``MeshFastState`` adds its J1/J2 and DTN
    caches in f64 likewise. Without a DTN term the totals keep their zone.
    """
    return (
        cast_to_zone(grad_J_up, "jastrow_grad_lap_carry") + cast_to_zone(dtn_state.grad_up, "jastrow_grad_lap_carry"),
        cast_to_zone(grad_J_dn, "jastrow_grad_lap_carry") + cast_to_zone(dtn_state.grad_dn, "jastrow_grad_lap_carry"),
        cast_to_zone(lap_J_up, "jastrow_grad_lap_carry") + cast_to_zone(dtn_state.lap_up, "jastrow_grad_lap_carry"),
        cast_to_zone(lap_J_dn, "jastrow_grad_lap_carry") + cast_to_zone(dtn_state.lap_dn, "jastrow_grad_lap_carry"),
    )


def _kinetic_energy_from_grads_laps(
    grad_J_up: jax.Array,
    grad_J_dn: jax.Array,
    lap_J_up: jax.Array,
    lap_J_dn: jax.Array,
    grad_ln_D_up: jax.Array,
    grad_ln_D_dn: jax.Array,
    lap_ln_D_up: jax.Array,
    lap_ln_D_dn: jax.Array,
) -> _KineticElements:
    """Common assembly: ``-(1/2) * (nabla^2ln Psi + ||nablaln Psi||^2)`` per electron."""
    # ``cast_to_zone`` folds each input's dtype into the promotion so the
    # casts below preserve Im on complex Bloch, exactly as in
    # compute_kinetic_energy_all_elements.
    grad_J_up = cast_to_zone(grad_J_up, "wf_kinetic")
    grad_J_dn = cast_to_zone(grad_J_dn, "wf_kinetic")
    lap_J_up = cast_to_zone(lap_J_up, "wf_kinetic")
    lap_J_dn = cast_to_zone(lap_J_dn, "wf_kinetic")
    grad_ln_D_up = cast_to_zone(grad_ln_D_up, "wf_kinetic")
    grad_ln_D_dn = cast_to_zone(grad_ln_D_dn, "wf_kinetic")
    lap_ln_D_up = cast_to_zone(lap_ln_D_up, "wf_kinetic")
    lap_ln_D_dn = cast_to_zone(lap_ln_D_dn, "wf_kinetic")

    grad_ln_Psi_up = grad_J_up + grad_ln_D_up
    grad_ln_Psi_dn = grad_J_dn + grad_ln_D_dn
    lap_ln_Psi_up = lap_J_up + lap_ln_D_up
    lap_ln_Psi_dn = lap_J_dn + lap_ln_D_dn
    ke_up = -0.5 * (lap_ln_Psi_up + jnp.sum(grad_ln_Psi_up**2, axis=1))
    ke_dn = -0.5 * (lap_ln_Psi_dn + jnp.sum(grad_ln_Psi_dn**2, axis=1))
    return ke_up, ke_dn


def _init_kinetic_energy_all_elements_streaming_state(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inverse: jax.Array,
) -> Kinetic_streaming_state:
    """Build a fresh streaming state at the supplied ``(r_up, r_dn)``.

    J1, J2, J3, and det sub-states are all incrementally
    maintained. NN three-body falls back to ``compute_grads_and_laplacian_Jastrow_part``
    (the streaming dispatch in ``jqmc_gfmc.py`` already excludes the NN case).

    Note: ``geminal_inverse`` must be the inverse of ``G(r_up, r_dn)`` (the
    same invariant as :func:`compute_kinetic_energy_all_elements_fast_update`).
    """
    # Per-electron Jastrow grad/lap of the J1/J2/J3/NN legs -- the initial total
    # without DTN. Sub-states below are populated for the streaming path. The
    # DTN leg is added from its carried state below, as every advance does, so
    # the total at the seed and after each hop is the same quantity.
    jastrow_data = wavefunction_data.jastrow_data
    dtn_data = jastrow_data.jastrow_dtn_data
    grad_J_up, grad_J_dn, lap_J_up, lap_J_dn = compute_grads_and_laplacian_Jastrow_part(
        jastrow_data=jastrow_data.replace(jastrow_dtn_data=None),
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # Cast totals to the jastrow_grad_lap zone so init and advance store
    # ``grad_J_*`` / ``lap_J_*`` in the same dtype (required
    # for fori_loop carry-shape stability under mixed precision, where
    # ``advance`` reassembles the totals from streaming sub-states that
    # live in the jastrow_grad_lap zone).
    grad_J_up = cast_to_zone(grad_J_up, "jastrow_grad_lap")
    grad_J_dn = cast_to_zone(grad_J_dn, "jastrow_grad_lap")
    lap_J_up = cast_to_zone(lap_J_up, "jastrow_grad_lap")
    lap_J_dn = cast_to_zone(lap_J_dn, "jastrow_grad_lap")

    # The DTN grad/lap is carried incrementally: one full DTN evaluation per
    # state build, amortised over the ~220 substeps of a branching step.
    dtn_grad_lap_state = (
        _init_grads_laplacian_Jastrow_DTN_streaming_state(dtn_data, r_up_carts, r_dn_carts) if dtn_data is not None else None
    )
    if dtn_grad_lap_state is not None:
        grad_J_up, grad_J_dn, lap_J_up, lap_J_dn = _add_carried_dtn_grad_lap(
            grad_J_up, grad_J_dn, lap_J_up, lap_J_dn, dtn_grad_lap_state
        )

    # Determinant streaming state -- drives grad_ln_D_*/lap_ln_D_* fields.
    det_state = _init_grads_laplacian_ln_Det_streaming_state(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        geminal_inverse=geminal_inverse,
    )

    j1_data = jastrow_data.jastrow_one_body_data
    j2_data = jastrow_data.jastrow_two_body_data
    j3_data = jastrow_data.jastrow_three_body_data
    j1_state = (
        _init_grads_laplacian_Jastrow_one_body_streaming_state(j1_data, r_up_carts, r_dn_carts) if j1_data is not None else None
    )
    j2_state = (
        _init_grads_laplacian_Jastrow_two_body_streaming_state(j2_data, r_up_carts, r_dn_carts) if j2_data is not None else None
    )
    j3_state = (
        _init_grads_laplacian_Jastrow_three_body_streaming_state(j3_data, r_up_carts, r_dn_carts)
        if j3_data is not None
        else None
    )

    # Assemble per-electron kinetic energies once here so the streaming carry
    # only stores the small ke_up/ke_dn arrays. Subsequent reads via
    # _kinetic_energy_from_streaming_state become free struct-field accesses.
    ke_up, ke_dn = _kinetic_energy_from_grads_laps(
        grad_J_up,
        grad_J_dn,
        lap_J_up,
        lap_J_dn,
        det_state.grad_ln_D_up,
        det_state.grad_ln_D_dn,
        det_state.lap_ln_D_up,
        det_state.lap_ln_D_dn,
    )

    return Kinetic_streaming_state(
        j1_state=j1_state,
        j2_state=j2_state,
        j3_state=j3_state,
        det_state=det_state,
        ke_up=ke_up,
        ke_dn=ke_dn,
        dtn_grad_lap_state=dtn_grad_lap_state,
    )


def _kinetic_energy_from_streaming_state(state: Kinetic_streaming_state) -> _KineticElements:
    """Per-electron kinetic energies extracted from a streaming state.

    Returns the pre-assembled ``ke_up`` / ``ke_dn`` carried in the streaming
    state. Init and advance produce these in the same kernel that produced
    the grad/lap totals, so this read costs no DRAM traffic of its own.
    """
    return state.ke_up, state.ke_dn


def _advance_kinetic_energy_all_elements_streaming_state(
    wavefunction_data: Wavefunction_data,
    state: Kinetic_streaming_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
    A_new_inv: jax.Array,
) -> Kinetic_streaming_state:
    """Advance the streaming state after a single-electron move.

    J1, J2, J3, and det sub-states are all updated
    incrementally. NN three-body falls back to a fresh
    ``compute_grads_and_laplacian_Jastrow_part`` call (defensive -- the
    streaming dispatch in ``jqmc_gfmc.py`` excludes the NN case so this
    branch is unreachable in production).

    The returned state is consistent with ``(r_up_carts_new, r_dn_carts_new,
    A_new_inv)``; downstream consumers can read kinetic energies via
    :func:`_kinetic_energy_from_streaming_state`.
    """
    jastrow_data = wavefunction_data.jastrow_data

    # --- J1: incremental advance via streaming state ---------------------
    j1_data = jastrow_data.jastrow_one_body_data
    if j1_data is not None and state.j1_state is not None:
        new_j1_state = _advance_grads_laplacian_Jastrow_one_body_streaming_state(
            j1_data,
            state.j1_state,
            moved_spin_is_up,
            moved_index,
            r_up_carts_new,
            r_dn_carts_new,
        )
        grad_J1_up = new_j1_state.grad_J1_up
        grad_J1_dn = new_j1_state.grad_J1_dn
        lap_J1_up = new_j1_state.lap_J1_up
        lap_J1_dn = new_j1_state.lap_J1_dn
    else:
        new_j1_state = None
        grad_J1_up = jnp.zeros_like(state.det_state.grad_ln_D_up)
        grad_J1_dn = jnp.zeros_like(state.det_state.grad_ln_D_dn)
        lap_J1_up = jnp.zeros_like(state.det_state.lap_ln_D_up)
        lap_J1_dn = jnp.zeros_like(state.det_state.lap_ln_D_dn)

    # --- J2: incremental advance via streaming state ---------------------
    j2_data = jastrow_data.jastrow_two_body_data
    if j2_data is not None and state.j2_state is not None:
        new_j2_state = _advance_grads_laplacian_Jastrow_two_body_streaming_state(
            j2_data,
            state.j2_state,
            moved_spin_is_up,
            moved_index,
            r_up_carts_new,
            r_dn_carts_new,
        )
        grad_J2_up = new_j2_state.grad_J2_up
        grad_J2_dn = new_j2_state.grad_J2_dn
        lap_J2_up = new_j2_state.lap_J2_up
        lap_J2_dn = new_j2_state.lap_J2_dn
    else:
        new_j2_state = None
        grad_J2_up = jnp.zeros_like(state.det_state.grad_ln_D_up)
        grad_J2_dn = jnp.zeros_like(state.det_state.grad_ln_D_dn)
        lap_J2_up = jnp.zeros_like(state.det_state.lap_ln_D_up)
        lap_J2_dn = jnp.zeros_like(state.det_state.lap_ln_D_dn)

    # --- J3: incremental advance via streaming state ---------------------
    j3_data = jastrow_data.jastrow_three_body_data
    if j3_data is not None and state.j3_state is not None:
        new_j3_state = _advance_grads_laplacian_Jastrow_three_body_streaming_state(
            j3_data,
            state.j3_state,
            moved_spin_is_up,
            moved_index,
            r_up_carts_new,
            r_dn_carts_new,
        )
        grad_J3_up = new_j3_state.grad_J3_up
        grad_J3_dn = new_j3_state.grad_J3_dn
        lap_J3_up = new_j3_state.lap_J3_up
        lap_J3_dn = new_j3_state.lap_J3_dn
    else:
        new_j3_state = None
        grad_J3_up = jnp.zeros_like(state.det_state.grad_ln_D_up)
        grad_J3_dn = jnp.zeros_like(state.det_state.grad_ln_D_dn)
        lap_J3_up = jnp.zeros_like(state.det_state.lap_ln_D_up)
        lap_J3_dn = jnp.zeros_like(state.det_state.lap_ln_D_dn)

    # Reassemble Jastrow totals from the streamed sub-state contributions, in
    # the jastrow_grad_lap zone (init's storage zone of the J1/J2/J3 legs).
    grad_J_up = cast_to_zone(grad_J1_up + grad_J2_up + grad_J3_up, "jastrow_grad_lap")
    grad_J_dn = cast_to_zone(grad_J1_dn + grad_J2_dn + grad_J3_dn, "jastrow_grad_lap")
    lap_J_up = cast_to_zone(lap_J1_up + lap_J2_up + lap_J3_up, "jastrow_grad_lap")
    lap_J_dn = cast_to_zone(lap_J1_dn + lap_J2_dn + lap_J3_dn, "jastrow_grad_lap")

    # --- DTN: incremental advance, else fresh recompute ------------------
    # These totals must contain the DTN term, as the INIT does via
    # compute_grads_and_laplacian_Jastrow_part: without it a DTN wavefunction
    # on the streaming path loses the whole DTN kinetic contribution after the
    # FIRST accepted move (~0.9 Ha per-electron ke error on water).
    #
    # FAST PATH: when the grad/lap sub-state is
    # carried, DTN advances INCREMENTALLY like J1/J2/J3/det -- only the terms
    # containing the moved electron change, O(N_atom * N_e) instead of the
    # O(N_atom * N_e^2) full recompute, which costs ~220 evaluations per
    # branching step. The carried leg is in the jastrow_grad_lap_carry zone and
    # is added as in the init (``_add_carried_dtn_grad_lap``). The recompute
    # branch below remains for state-less callers and as the parity oracle.
    dtn_data = jastrow_data.jastrow_dtn_data
    new_dtn_grad_lap_state = state.dtn_grad_lap_state
    if dtn_data is not None:
        if state.dtn_grad_lap_state is not None:
            new_dtn_grad_lap_state = _advance_grads_laplacian_Jastrow_DTN_streaming_state(
                dtn_data,
                state.dtn_grad_lap_state,
                moved_spin_is_up,
                moved_index,
                r_up_carts_new,
                r_dn_carts_new,
            )
            grad_J_up, grad_J_dn, lap_J_up, lap_J_dn = _add_carried_dtn_grad_lap(
                grad_J_up, grad_J_dn, lap_J_up, lap_J_dn, new_dtn_grad_lap_state
            )
        else:
            grad_dtn_up, grad_dtn_dn, lap_dtn_up, lap_dtn_dn = compute_grads_and_laplacian_Jastrow_DTN(
                dtn_data, r_up_carts_new, r_dn_carts_new
            )
            grad_J_up = cast_to_zone(grad_J_up + grad_dtn_up, "jastrow_grad_lap")
            grad_J_dn = cast_to_zone(grad_J_dn + grad_dtn_dn, "jastrow_grad_lap")
            lap_J_up = cast_to_zone(lap_J_up + lap_dtn_up, "jastrow_grad_lap")
            lap_J_dn = cast_to_zone(lap_J_dn + lap_dtn_dn, "jastrow_grad_lap")

    # --- determinant: incremental advance via streaming state ------------
    new_det_state = _advance_grads_laplacian_ln_Det_streaming_state(
        geminal_data=wavefunction_data.geminal_data,
        state=state.det_state,
        moved_spin_is_up=moved_spin_is_up,
        moved_index=moved_index,
        r_up_carts_new=r_up_carts_new,
        r_dn_carts_new=r_dn_carts_new,
        A_new_inv=A_new_inv,
    )

    # Assemble per-electron kinetic energies in-kernel: the new grad/lap
    # totals are still register-resident here, so this avoids the dedicated
    # read kernel that would otherwise re-load all 8 arrays from DRAM.
    ke_up, ke_dn = _kinetic_energy_from_grads_laps(
        grad_J_up,
        grad_J_dn,
        lap_J_up,
        lap_J_dn,
        new_det_state.grad_ln_D_up,
        new_det_state.grad_ln_D_dn,
        new_det_state.lap_ln_D_up,
        new_det_state.lap_ln_D_dn,
    )

    return state.replace(
        j1_state=new_j1_state,
        j2_state=new_j2_state,
        j3_state=new_j3_state,
        det_state=new_det_state,
        ke_up=ke_up,
        ke_dn=ke_dn,
        dtn_grad_lap_state=new_dtn_grad_lap_state,
    )


def _compute_discretized_kinetic_energy_debug(
    alat: float, wavefunction_data: Wavefunction_data, r_up_carts: npt.NDArray, r_dn_carts: npt.NDArray
) -> list[tuple[npt.NDArray, npt.NDArray]]:
    r"""Build the LRDMC kinetic mesh and its wavefunction ratios (debug reference).

    Args:
        alat (float): Hamiltonian discretization (bohr).
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data.
        r_up_carts (npt.NDArray): up electron position (N_e,3).
        r_dn_carts (npt.NDArray): down electron position (N_e,3).

    Returns:
        list[tuple[npt.NDArray, npt.NDArray]], list[npt.NDArray]:
            return mesh for the LRDMC kinetic part, a list containing tuples containing (r_carts_up, r_carts_dn),
            and a list containing values of the \Psi(x')/\Psi(x) corresponding to the grid.
    """
    mesh_kinetic_part = []

    # PBC: the mesh point r_i +/- alat is passed to Psi UNWRAPPED, as in the
    # production compute_discretized_kinetic_energy. The home-cell wrap is
    # centralised in compute_AOs, which also restores the Bloch phase of the
    # lattice vector it removes; an explicit wrap here would move the point by T
    # before that guarantor and add a spurious exp(-i k.T) at a twist (the ECP
    # debug oracle had this defect). ``_shift_and_wrap`` returns a copy of
    # ``base`` with row ``idx`` shifted by ``delta`` along ``axis``.
    def _shift_and_wrap(
        base: npt.NDArray[np.floating],
        idx: int,
        axis: int,
        delta: float,
    ) -> npt.NDArray[np.floating]:
        out = base.copy()
        out[idx, axis] += delta
        return out

    # up electron
    for r_up_i in range(len(r_up_carts)):
        # x, plus
        mesh_kinetic_part.append((_shift_and_wrap(r_up_carts, r_up_i, 0, alat), r_dn_carts))
        # x, minus
        mesh_kinetic_part.append((_shift_and_wrap(r_up_carts, r_up_i, 0, -alat), r_dn_carts))
        # y, plus
        mesh_kinetic_part.append((_shift_and_wrap(r_up_carts, r_up_i, 1, alat), r_dn_carts))
        # y, minus
        mesh_kinetic_part.append((_shift_and_wrap(r_up_carts, r_up_i, 1, -alat), r_dn_carts))
        # z, plus
        mesh_kinetic_part.append((_shift_and_wrap(r_up_carts, r_up_i, 2, alat), r_dn_carts))
        # z, minus
        mesh_kinetic_part.append((_shift_and_wrap(r_up_carts, r_up_i, 2, -alat), r_dn_carts))

    # dn electron
    for r_dn_i in range(len(r_dn_carts)):
        # x, plus
        mesh_kinetic_part.append((r_up_carts, _shift_and_wrap(r_dn_carts, r_dn_i, 0, alat)))
        # x, minus
        mesh_kinetic_part.append((r_up_carts, _shift_and_wrap(r_dn_carts, r_dn_i, 0, -alat)))
        # y, plus
        mesh_kinetic_part.append((r_up_carts, _shift_and_wrap(r_dn_carts, r_dn_i, 1, alat)))
        # y, minus
        mesh_kinetic_part.append((r_up_carts, _shift_and_wrap(r_dn_carts, r_dn_i, 1, -alat)))
        # z, plus
        mesh_kinetic_part.append((r_up_carts, _shift_and_wrap(r_dn_carts, r_dn_i, 2, alat)))
        # z, minus
        mesh_kinetic_part.append((r_up_carts, _shift_and_wrap(r_dn_carts, r_dn_i, 2, -alat)))

    # Psi(R')/Psi(R) = exp(J(R') - J(R)) * det(R')/det(R). The Jastrow factor is a
    # ratio, so both whole values are evaluated in the ``jastrow_ratio`` zone that owns
    # it, not in ``jastrow_eval`` (fp32 in mixed), whose rounding of |J| a quotient of
    # two whole wavefunction values would keep (issue #781).
    jastrow_data = wavefunction_data.jastrow_data
    geminal_data = wavefunction_data.geminal_data
    J_ref = compute_Jastrow_part_ratio_zone(jastrow_data, r_up_carts, r_dn_carts)
    det_ref = compute_det_geminal_all_elements(geminal_data=geminal_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
    elements_kinetic_part = [
        float(
            -1.0
            / (2.0 * alat**2)
            * np.exp(np.float64(compute_Jastrow_part_ratio_zone(jastrow_data, r_up_carts_, r_dn_carts_)) - np.float64(J_ref))
            * compute_det_geminal_all_elements(geminal_data=geminal_data, r_up_carts=r_up_carts_, r_dn_carts=r_dn_carts_)
            / det_ref
        )
        for r_up_carts_, r_dn_carts_ in mesh_kinetic_part
    ]

    r_up_carts_combined = np.array([up for up, _ in mesh_kinetic_part])
    r_dn_carts_combined = np.array([dn for _, dn in mesh_kinetic_part])

    return r_up_carts_combined, r_dn_carts_combined, elements_kinetic_part


@jit
def compute_discretized_kinetic_energy(
    alat: float,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Compute discretized kinetic mesh points and energies for a given lattice spacing ``alat``.

    Function for computing discretized kinetic grid points and their energies with a
    given lattice space (alat). Ratios are computed as
    ``exp(J_xp - J_x) * det_xp / det_x``. Inputs are coerced to the
    ``wf_kinetic`` zone dtype before evaluation.

    Args:
        alat: Hamiltonian discretization (bohr).
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Up-electron positions with shape ``(n_up, 3)``.
        r_dn_carts: Down-electron positions with shape ``(n_dn, 3)``.
        RT: Rotation matrix (:math:`R^T`) with shape ``(3, 3)``.

    Returns:
        A tuple ``(r_up_carts_combined, r_dn_carts_combined, elements_kinetic_part)`` where the
        combined coordinate arrays have shapes ``(n_grid, n_up, 3)`` and ``(n_grid, n_dn, 3)``
        and ``elements_kinetic_part`` contains the kinetic prefactor-scaled ratios.
    """
    r_up = cast_to_zone(r_up_carts, "wf_kinetic")
    r_dn = cast_to_zone(r_dn_carts, "wf_kinetic")
    rt = cast_to_zone(RT, "wf_kinetic")
    # PBC note: each discretized hop moves one electron by +/- alat along a
    # rotated axis, so a mesh point can leave the home cell Omega. The wrap
    # that keeps the truncated AO image sum accurate is centralised in the
    # periodicity guarantor ``compute_AOs`` (jqmc/atomic_orbital.py), which
    # wraps every Psi-evaluation coordinate, so an explicit mesh wrap here
    # would be redundant. Walkers carried forward across cells are re-wrapped
    # at the sampler accept sites (jqmc_gfmc / jqmc_mcmc).
    # Define the shifts to apply (+/- alat in each coordinate direction)
    shifts = alat * jnp.array(
        [
            [1, 0, 0],  # x+
            [-1, 0, 0],  # x-
            [0, 1, 0],  # y+
            [0, -1, 0],  # y-
            [0, 0, 1],  # z+
            [0, 0, -1],  # z-
        ]
    )  # Shape: (6, 3)

    shifts = shifts @ rt  # Shape: (6, 3)

    # num shift
    num_shifts = shifts.shape[0]

    # Process up-spin electrons
    num_up_electrons = r_up.shape[0]
    num_up_configs = num_up_electrons * num_shifts

    # Create base positions repeated for each configuration
    base_positions_up = jnp.repeat(r_up[None, :, :], num_up_configs, axis=0)  # Shape: (num_up_configs, N_up, 3)

    # Initialize shifts_to_apply_up
    shifts_to_apply_up = jnp.zeros_like(base_positions_up)

    # Create indices for configurations
    config_indices_up = jnp.arange(num_up_configs)
    electron_indices_up = jnp.repeat(jnp.arange(num_up_electrons), num_shifts)
    shift_indices_up = jnp.tile(jnp.arange(num_shifts), num_up_electrons)

    # Apply shifts to the appropriate electron in each configuration
    shifts_to_apply_up = shifts_to_apply_up.at[config_indices_up, electron_indices_up, :].set(shifts[shift_indices_up])

    # Apply shifts to base positions
    r_up_carts_shifted = base_positions_up + shifts_to_apply_up  # Shape: (num_up_configs, N_up, 3)

    # PBC home-cell wrap is centralised in ``compute_AOs`` (see note above);
    # the shifted mesh positions flow into the wavefunction ratio through it.

    # Repeat down-spin electrons for up-spin configurations
    r_dn_carts_repeated_up = jnp.repeat(r_dn[None, :, :], num_up_configs, axis=0)  # Shape: (num_up_configs, N_dn, 3)

    # Process down-spin electrons
    num_dn_electrons = r_dn.shape[0]
    num_dn_configs = num_dn_electrons * num_shifts

    base_positions_dn = jnp.repeat(r_dn[None, :, :], num_dn_configs, axis=0)  # Shape: (num_dn_configs, N_dn, 3)
    shifts_to_apply_dn = jnp.zeros_like(base_positions_dn)

    config_indices_dn = jnp.arange(num_dn_configs)
    electron_indices_dn = jnp.repeat(jnp.arange(num_dn_electrons), num_shifts)
    shift_indices_dn = jnp.tile(jnp.arange(num_shifts), num_dn_electrons)

    # Apply shifts to the appropriate electron in each configuration
    shifts_to_apply_dn = shifts_to_apply_dn.at[config_indices_dn, electron_indices_dn, :].set(shifts[shift_indices_dn])

    r_dn_carts_shifted = base_positions_dn + shifts_to_apply_dn  # Shape: (num_dn_configs, N_dn, 3)

    # PBC home-cell wrap is centralised in ``compute_AOs`` (see note above).

    # Repeat up-spin electrons for down-spin configurations
    r_up_carts_repeated_dn = jnp.repeat(r_up[None, :, :], num_dn_configs, axis=0)  # Shape: (num_dn_configs, N_up, 3)

    # Combine configurations
    r_up_carts_combined = jnp.concatenate([r_up_carts_shifted, r_up_carts_repeated_dn], axis=0)  # Shape: (N_configs, N_up, 3)
    r_dn_carts_combined = jnp.concatenate([r_dn_carts_repeated_up, r_dn_carts_shifted], axis=0)  # Shape: (N_configs, N_dn, 3)

    # Evaluate the wavefunction at the original positions
    jastrow_x = compute_Jastrow_part_ratio_zone(wavefunction_data.jastrow_data, r_up, r_dn)
    # Evaluate the wavefunction at the shifted positions using vectorization
    jastrow_xp = vmap(compute_Jastrow_part_ratio_zone, in_axes=(None, 0, 0))(
        wavefunction_data.jastrow_data, r_up_carts_combined, r_dn_carts_combined
    )
    # Evaluate the wavefunction at the original positions
    det_x = compute_det_geminal_all_elements(wavefunction_data.geminal_data, r_up, r_dn)
    # Evaluate the wavefunction at the shifted positions using vectorization
    det_xp = vmap(compute_det_geminal_all_elements, in_axes=(None, 0, 0))(
        wavefunction_data.geminal_data, r_up_carts_combined, r_dn_carts_combined
    )
    # Explicitly cast both jastrow (jastrow_eval zone, possibly fp32) and det
    # values to the wf_ratio zone dtype so that exp() and the wf_ratio arithmetic
    # do not rely on JAX implicit fp32 x fp64 -> fp64 promotion.
    # Promote dtype against the geminal lambda dtype so
    # the complex Bloch ``det_*`` preserve Im through the cast (the discretized
    # kinetic ratio ``det_xp / det_x`` is complex at twist k != 0; the projector
    # / LRDMC e_L take the real part downstream). Bit-identical at Gamma where
    # lambda is real.
    # ``cast_to_zone`` folds each input's dtype into the promotion so the
    # cast preserves Im when lambda is real but AOs are complex (RHF singlet
    # at non-Gamma twist).
    lambda_dtype = wavefunction_data.geminal_data._determinant_dtype_jnp
    jastrow_x = cast_to_zone(jastrow_x, "wf_ratio", lambda_dtype, det_x.dtype, det_xp.dtype)
    jastrow_xp = cast_to_zone(jastrow_xp, "wf_ratio", lambda_dtype, det_x.dtype, det_xp.dtype)
    det_x = cast_to_zone(det_x, "wf_ratio", lambda_dtype, det_xp.dtype)
    det_xp = cast_to_zone(det_xp, "wf_ratio", lambda_dtype, det_x.dtype)
    wf_ratio = jnp.exp(jastrow_xp - jastrow_x) * det_xp / det_x

    # Compute the kinetic part elements
    elements_kinetic_part = -1.0 / (2.0 * alat**2) * wf_ratio

    # Return the combined configurations and the kinetic elements
    return r_up_carts_combined, r_dn_carts_combined, elements_kinetic_part


@struct.dataclass
class Discretized_kinetic_mesh_streaming_state:
    """Projection-scoped cache for the fixed six-point LRDMC lattice."""

    shifts: jax.Array
    candidate_up_positions: jax.Array
    candidate_dn_positions: jax.Array
    det_candidate_state: Det_candidate_mesh_state
    j1_candidate_state: J1_candidate_mesh_state | None = struct.field(pytree_node=True, default=None)
    j3_candidate_state: J3_candidate_mesh_state | None = struct.field(pytree_node=True, default=None)
    j2_candidate_state: J2_candidate_mesh_state | None = struct.field(pytree_node=True, default=None)
    j_dtn_candidate_state: J_DTN_candidate_mesh_state | None = struct.field(pytree_node=True, default=None)


@jit
def _init_discretized_kinetic_mesh_streaming_state(
    alat: float,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
) -> Discretized_kinetic_mesh_streaming_state:
    shifts = cast_to_zone(
        alat
        * jnp.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        )
        @ RT,
        "wf_kinetic",
    )
    r_up = cast_to_zone(r_up_carts, "wf_kinetic")
    r_dn = cast_to_zone(r_dn_carts, "wf_kinetic")
    candidate_up = r_up[:, None, :] + shifts[None, :, :]
    candidate_dn = r_dn[:, None, :] + shifts[None, :, :]
    det_candidates = _init_det_candidate_mesh_state(wavefunction_data.geminal_data, candidate_up, candidate_dn)
    j1_data = wavefunction_data.jastrow_data.jastrow_one_body_data
    j1_candidates = (
        _init_j1_candidate_mesh_state(j1_data, r_up_carts, r_dn_carts, candidate_up, candidate_dn)
        if j1_data is not None
        else None
    )
    j3_data = wavefunction_data.jastrow_data.jastrow_three_body_data
    j3_candidates = _init_j3_candidate_mesh_state(j3_data, candidate_up, candidate_dn) if j3_data is not None else None
    j2_data = wavefunction_data.jastrow_data.jastrow_two_body_data
    j2_candidates = (
        _init_j2_candidate_mesh_state(j2_data, r_up_carts, r_dn_carts, candidate_up, candidate_dn)
        if j2_data is not None
        else None
    )
    dtn_data = wavefunction_data.jastrow_data.jastrow_dtn_data
    j_dtn_candidates = (
        _init_j_dtn_candidate_mesh_state(dtn_data, r_up_carts, r_dn_carts, candidate_up, candidate_dn)
        if dtn_data is not None
        else None
    )
    return Discretized_kinetic_mesh_streaming_state(
        shifts=shifts,
        candidate_up_positions=candidate_up,
        candidate_dn_positions=candidate_dn,
        det_candidate_state=det_candidates,
        j1_candidate_state=j1_candidates,
        j3_candidate_state=j3_candidates,
        j2_candidate_state=j2_candidates,
        j_dtn_candidate_state=j_dtn_candidates,
    )


@jit
def _advance_discretized_kinetic_mesh_streaming_state(
    wavefunction_data: Wavefunction_data,
    state: Discretized_kinetic_mesh_streaming_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_old: jax.Array,
    r_dn_carts_old: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
) -> Discretized_kinetic_mesh_streaming_state:
    candidate_up = cast_to_zone(r_up_carts_new, "wf_kinetic")[:, None, :] + state.shifts[None, :, :]
    candidate_dn = cast_to_zone(r_dn_carts_new, "wf_kinetic")[:, None, :] + state.shifts[None, :, :]
    det_candidates = _advance_det_candidate_mesh_state(
        wavefunction_data.geminal_data,
        state.det_candidate_state,
        moved_spin_is_up,
        moved_index,
        candidate_up,
        candidate_dn,
    )
    j1_data = wavefunction_data.jastrow_data.jastrow_one_body_data
    j1_candidates = (
        _advance_j1_candidate_mesh_state(
            j1_data,
            state.j1_candidate_state,
            moved_spin_is_up,
            moved_index,
            r_up_carts_new,
            r_dn_carts_new,
            candidate_up,
            candidate_dn,
        )
        if j1_data is not None
        else None
    )
    j3_data = wavefunction_data.jastrow_data.jastrow_three_body_data
    j3_candidates = (
        _advance_j3_candidate_mesh_state(
            j3_data,
            state.j3_candidate_state,
            moved_spin_is_up,
            moved_index,
            candidate_up,
            candidate_dn,
        )
        if j3_data is not None
        else None
    )
    j2_data = wavefunction_data.jastrow_data.jastrow_two_body_data
    j2_candidates = (
        _advance_j2_candidate_mesh_state(
            j2_data,
            state.j2_candidate_state,
            moved_spin_is_up,
            moved_index,
            r_up_carts_old,
            r_dn_carts_old,
            r_up_carts_new,
            r_dn_carts_new,
            candidate_up,
            candidate_dn,
        )
        if j2_data is not None
        else None
    )
    dtn_data = wavefunction_data.jastrow_data.jastrow_dtn_data
    j_dtn_candidates = (
        _advance_j_dtn_candidate_mesh_state(
            dtn_data,
            state.j_dtn_candidate_state,
            moved_spin_is_up,
            moved_index,
            r_up_carts_old,
            r_dn_carts_old,
            r_up_carts_new,
            r_dn_carts_new,
            candidate_up,
            candidate_dn,
        )
        if dtn_data is not None
        else None
    )
    return state.replace(
        candidate_up_positions=candidate_up,
        candidate_dn_positions=candidate_dn,
        det_candidate_state=det_candidates,
        j1_candidate_state=j1_candidates,
        j3_candidate_state=j3_candidates,
        j2_candidate_state=j2_candidates,
        j_dtn_candidate_state=j_dtn_candidates,
    )


def _candidate_positions_to_configurations(r_carts: jax.Array, candidate_positions: jax.Array) -> jax.Array:
    n, k, _ = candidate_positions.shape
    delta = candidate_positions - r_carts[:, None, :]
    one_hot = jnp.eye(n, dtype=r_carts.dtype)
    return (r_carts[None, None, :, :] + delta[:, :, None, :] * one_hot[:, None, :, None]).reshape(n * k, n, 3)


@jit
def compute_discretized_kinetic_energy_from_streaming_state(
    alat: float,
    wavefunction_data: Wavefunction_data,
    A_old_inv: jax.Array,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    mesh_state: Discretized_kinetic_mesh_streaming_state,
    j3_state: Jastrow_three_body_streaming_state | None,
    det_ratio_state: Det_ratio_streaming_state | Det_streaming_state,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate the six-point lattice while reusing projection-scoped candidate AOs."""
    r_up = cast_to_zone(r_up_carts, "wf_kinetic")
    r_dn = cast_to_zone(r_dn_carts, "wf_kinetic")
    up_shifted = _candidate_positions_to_configurations(r_up, mesh_state.candidate_up_positions)
    dn_shifted = _candidate_positions_to_configurations(r_dn, mesh_state.candidate_dn_positions)
    up_other = jnp.broadcast_to(r_dn[None, :, :], (up_shifted.shape[0], r_dn.shape[0], 3))
    dn_other = jnp.broadcast_to(r_up[None, :, :], (dn_shifted.shape[0], r_up.shape[0], 3))
    combined_up = jnp.concatenate([up_shifted, dn_other], axis=0)
    combined_dn = jnp.concatenate([up_other, dn_shifted], axis=0)

    # This IS the per-sub-step projection kernel: every channel it evaluates --
    # the determinant and all four Jastrow legs -- reports under that site, so
    # a gate can demand "nothing uncached at site 'projection'" without also
    # condemning the substep-0 warm-up or the observable/force paths.
    with route_site("projection"):
        note_route("det", "kinetic", cached=True)
        # winv weights are DERIVED here from the running inverse -- never
        # chained (see ``Det_winv_state``): two O(M*N^2) GEMMs, then the
        # per-candidate one-dot ratio kernel.
        det_winv = _init_det_winv_state(wavefunction_data.geminal_data, A_old_inv, det_ratio_state)
        det_ratio_raw = _compute_ratio_determinant_from_winv_state(
            wavefunction_data.geminal_data,
            mesh_state.det_candidate_state,
            det_winv,
        )
        jastrow_data = wavefunction_data.jastrow_data
        jastrow_ratio_raw = compute_ratio_Jastrow_one_move_mesh(
            jastrow_data,
            r_up,
            r_dn,
            combined_up,
            combined_dn,
            mesh_state.shifts,
            j3_state=j3_state,
            j1_candidate_state=mesh_state.j1_candidate_state,
            j3_candidate_state=mesh_state.j3_candidate_state,
            j2_candidate_state=mesh_state.j2_candidate_state,
            j_dtn_candidate_state=mesh_state.j_dtn_candidate_state,
        )
    lambda_dtype = wavefunction_data.geminal_data._determinant_dtype_jnp
    # Preserve the production kernel's established cast order exactly.  In
    # mixed precision, adding the other ratio's dtype to the determinant cast
    # changes where rounding occurs even when the final dtype is identical.
    det_ratio = cast_to_zone(det_ratio_raw, "wf_ratio", lambda_dtype)
    jastrow_ratio = cast_to_zone(jastrow_ratio_raw, "wf_ratio", lambda_dtype, det_ratio_raw.dtype)
    elements = -1.0 / (2.0 * alat**2) * det_ratio * jastrow_ratio
    return combined_up, combined_dn, elements


@jit
def compute_discretized_kinetic_energy_fast_update(
    alat: float,
    wavefunction_data: Wavefunction_data,
    A_old_inv: jnp.ndarray,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    j3_state: "Jastrow_three_body_streaming_state | None" = None,
    det_ratio_state: Det_ratio_streaming_state | Det_streaming_state | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Fast-update version of discretized kinetic mesh and ratios.

    Function for computing discretized kinetic grid points and their energies with
    a given lattice space (alat). Uses precomputed ``A_old_inv`` to evaluate
    determinant ratios efficiently. Inputs are converted to the ``wf_kinetic``
    zone dtype before use.

    Args:
        alat: Hamiltonian discretization (bohr).
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        A_old_inv: Inverse of the geminal matrix evaluated at ``(r_up_carts, r_dn_carts)``.
        r_up_carts: Up-electron positions with shape ``(n_up, 3)``.
        r_dn_carts: Down-electron positions with shape ``(n_dn, 3)``.
        RT: Rotation matrix (:math:`R^T`) with shape ``(3, 3)``.
        j3_state: Optional cached J3 streaming auxiliaries consistent with
            ``(r_up_carts, r_dn_carts)``. Forwarded to the Jastrow ratio kernel
            so it can skip the per-call ``aos_*_old``/``W``/``U``/cross_vec
            recomputation. Use the value carried in the projection's
            ``Kinetic_streaming_state.j3_state``; pass ``None`` (default) for
            the one-shot path used by observation/MCMC code.
        det_ratio_state: Optional :class:`Det_ratio_streaming_state` (or a
            superset like :class:`Det_streaming_state`) consistent with
            ``(r_up_carts, r_dn_carts)``. Forwarded to
            ``_compute_ratio_determinant_part_split_spin`` so it can skip
            the bulk-side AO eval and the two ``lambda_paired @ ao_*``
            precontracts.

    The Jastrow ratio comes from ``compute_ratio_Jastrow_one_move_mesh``, which
    receives the ``(m, k)`` mesh structure and returns the WHOLE ratio. Which
    channels are fused over that structure is decided inside the Jastrow layer;
    this wavefunction-level caller remains independent of channel membership.

    Returns:
        Tuple ``(r_up_carts_combined, r_dn_carts_combined, elements_kinetic_part)`` with combined
        coordinate arrays of shapes ``(n_grid, n_up, 3)`` and ``(n_grid, n_dn, 3)``, and kinetic
        prefactor-scaled ratios ``elements_kinetic_part``.
    """
    r_up = cast_to_zone(r_up_carts, "wf_kinetic")
    r_dn = cast_to_zone(r_dn_carts, "wf_kinetic")
    # PBC note (see compute_discretized_kinetic_energy): the home-cell wrap is
    # centralised in the periodicity guarantor ``compute_AOs``, so no explicit
    # mesh wrap is needed here; identity for molecules.
    rt = cast_to_zone(RT, "wf_kinetic")
    # Define the shifts to apply (+/- alat in each coordinate direction)
    shifts = alat * jnp.array(
        [
            [1, 0, 0],  # x+
            [-1, 0, 0],  # x-
            [0, 1, 0],  # y+
            [0, -1, 0],  # y-
            [0, 0, 1],  # z+
            [0, 0, -1],  # z-
        ]
    )  # Shape: (6, 3)

    shifts = shifts @ rt  # Shape: (6, 3)

    # num shift
    num_shifts = shifts.shape[0]

    # Process up-spin electrons
    num_up_electrons = r_up.shape[0]
    num_up_configs = num_up_electrons * num_shifts

    # Build shifted configurations via outer-product broadcast (no scatter, GEMM-compatible).
    # delta_up[i, s, j, :] = shifts[s] if j==i else 0
    eye_up = jnp.eye(num_up_electrons)  # (N_up, N_up)
    delta_up = eye_up[:, None, :, None] * shifts[None, :, None, :]  # (N_up, 6, N_up, 3)
    r_up_carts_shifted = (r_up[None, None, :, :] + delta_up).reshape(
        num_up_configs, num_up_electrons, 3
    )  # (num_up_configs, N_up, 3)

    # PBC home-cell wrap is centralised in ``compute_AOs`` (see note above);
    # the shifted mesh positions flow into the wavefunction ratio through it.

    # Broadcast unchanged spin block (avoids repeat allocation)
    r_dn_carts_repeated_up = jnp.broadcast_to(r_dn[None, :, :], (num_up_configs, r_dn.shape[0], 3))  # (num_up_configs, N_dn, 3)

    # Process down-spin electrons
    num_dn_electrons = r_dn.shape[0]
    num_dn_configs = num_dn_electrons * num_shifts

    eye_dn = jnp.eye(num_dn_electrons)  # (N_dn, N_dn)
    delta_dn = eye_dn[:, None, :, None] * shifts[None, :, None, :]  # (N_dn, 6, N_dn, 3)
    r_dn_carts_shifted = (r_dn[None, None, :, :] + delta_dn).reshape(
        num_dn_configs, num_dn_electrons, 3
    )  # (num_dn_configs, N_dn, 3)

    # PBC home-cell wrap is centralised in ``compute_AOs`` (see note above).

    # Broadcast unchanged spin block (avoids repeat allocation)
    r_up_carts_repeated_dn = jnp.broadcast_to(r_up[None, :, :], (num_dn_configs, r_up.shape[0], 3))  # (num_dn_configs, N_up, 3)

    # Combine configurations
    r_up_carts_combined = jnp.concatenate([r_up_carts_shifted, r_up_carts_repeated_dn], axis=0)  # Shape: (N_configs, N_up, 3)
    r_dn_carts_combined = jnp.concatenate([r_dn_carts_repeated_up, r_dn_carts_shifted], axis=0)  # Shape: (N_configs, N_dn, 3)

    # Evaluate the ratios of wavefunctions between the shifted positions and the original position.
    # det_ratio (det_ratio zone) and jastrow_ratio (jastrow_ratio zone) are explicitly cast to
    # the wf_ratio zone dtype to avoid relying on JAX implicit promotion.
    # Promote dtype against the geminal lambda dtype AND
    # the actual det_ratio dtype so the complex Bloch det_ratio preserves Im
    # through the cast (lambda can be real for RHF singlet while det_ratio is
    # complex via the AO leaves). Mirrors the slow path in
    # compute_discretized_kinetic_energy. Bit-identical at Gamma.
    # This entry point takes NO mesh state, so every channel it evaluates is on
    # the uncached route by construction. It is the one the analytic
    # force/stress path calls (``derivatives/gfmc.py``), which is why a
    # force-enabled run gets no candidate caching at all.
    note_route("det", "kinetic", cached=False)
    _det_ratio_raw = _compute_ratio_determinant_part_split_spin(
        geminal_data=wavefunction_data.geminal_data,
        A_old_inv=A_old_inv,
        old_r_up_carts=r_up,
        old_r_dn_carts=r_dn,
        new_r_up_shifted=r_up_carts_shifted,
        new_r_dn_shifted=r_dn_carts_shifted,
        det_ratio_state=det_ratio_state,
    )
    lambda_dtype = wavefunction_data.geminal_data._determinant_dtype_jnp
    det_ratio = cast_to_zone(_det_ratio_raw, "wf_ratio", lambda_dtype)
    # This site declares the mesh STRUCTURE -- the (m, k) lattice built above,
    # whose up-then-dn shift-minor flattening the flattening-order gate pins --
    # and receives the WHOLE Jastrow ratio. Which channels exist and which of
    # them have a fused mesh kernel is decided inside the Jastrow layer; this
    # caller remains agnostic to the contents of ``jastrow_data``.
    _jastrow_ratio_raw = compute_ratio_Jastrow_one_move_mesh(
        jastrow_data=wavefunction_data.jastrow_data,
        old_r_up_carts=r_up,
        old_r_dn_carts=r_dn,
        new_r_up_carts_arr=r_up_carts_combined,
        new_r_dn_carts_arr=r_dn_carts_combined,
        shifts=shifts,
        j3_state=j3_state,
    )
    jastrow_ratio = cast_to_zone(
        _jastrow_ratio_raw,
        "wf_ratio",
        lambda_dtype,
        _det_ratio_raw.dtype,
    )
    wf_ratio = det_ratio * jastrow_ratio

    # Compute the kinetic part elements
    elements_kinetic_part = -1.0 / (2.0 * alat**2) * wf_ratio

    # Return the combined configurations and the kinetic elements
    return r_up_carts_combined, r_dn_carts_combined, elements_kinetic_part


# ---------------------------------------------------------------------------
# Nodal distance and f_epsilon regularization
# ---------------------------------------------------------------------------
# References:
#   Pathak & Wagner, AIP Advances 10, 085213 (2020).
#   DOI: 10.1063/5.0004008
# ---------------------------------------------------------------------------


def f_epsilon_PW(nodal_distance: jnpt.ArrayLike, epsilon_PW: float) -> jnpt.ArrayLike:
    r"""Evaluate the Pathak--Wagner regularization function :math:`f_\varepsilon`.

    .. math::

        f_\varepsilon(t) =
        \begin{cases}
            7|t|^6 - 15|t|^4 + 9|t|^2 & (|t| < 1) \\
            1                           & (|t| \geq 1)
        \end{cases}

    where :math:`t = |x| / \varepsilon` and :math:`|x|` is the nodal distance.

    The coefficients :math:`(7, -15, 9)` satisfy :math:`f(0) = 0`, :math:`f(1) = 1`,
    and continuity of the first two derivatives at :math:`|t| = 1`.

    Args:
        nodal_distance: Nodal distance(s), shape arbitrary (scalar or array).
        epsilon_PW: Regularization cutoff length :math:`\varepsilon`.

    Returns:
        Regularization factor with the same shape as *nodal_distance*.
    """
    t = jnp.abs(nodal_distance) / epsilon_PW
    t2 = t * t
    t4 = t2 * t2
    t6 = t4 * t2
    f_inner = 7.0 * t6 - 15.0 * t4 + 9.0 * t2
    return jnp.where(t < 1.0, f_inner, 1.0)


def compute_nodal_distance(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    r"""Compute the nodal distance using analytic derivatives.

    The nodal distance is defined as

    .. math::

        |x| = \frac{1}{|\nabla \ln |\Psi||},

    where :math:`\nabla` is the many-body gradient over all electron coordinates,
    computed analytically from the Jastrow and determinant parts.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Scalar nodal distance value.
    """
    # r_*_carts forwarded unchanged (see ``evaluate_ln_wavefunction`` for rationale).
    grad_J_up, grad_J_dn, _, _ = compute_grads_and_laplacian_Jastrow_part(
        jastrow_data=wavefunction_data.jastrow_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    grad_ln_D_up, grad_ln_D_dn, _, _ = compute_grads_and_laplacian_ln_Det(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # Cast jastrow_grad_lap / det_grad_lap zone values to wf_eval dtype.
    grad_J_up = cast_to_zone(grad_J_up, "wf_eval")
    grad_J_dn = cast_to_zone(grad_J_dn, "wf_eval")
    grad_ln_D_up = cast_to_zone(grad_ln_D_up, "wf_eval")
    grad_ln_D_dn = cast_to_zone(grad_ln_D_dn, "wf_eval")

    grad_ln_Psi_up = grad_J_up + grad_ln_D_up  # (n_up, 3)
    grad_ln_Psi_dn = grad_J_dn + grad_ln_D_dn  # (n_dn, 3)

    grad_norm_sq = jnp.sum(grad_ln_Psi_up**2) + jnp.sum(grad_ln_Psi_dn**2)
    return 1.0 / jnp.sqrt(grad_norm_sq)


def _compute_nodal_distance_debug(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    r"""Compute the nodal distance using autodiff of ``ln|Psi|`` (debug).

    Computes :math:`1 / |\nabla \ln|\Psi||` directly via
    ``jax.grad(evaluate_ln_wavefunction)``.  This is mathematically identical
    to the paper's :math:`|x| = |\Psi| / |\nabla \Psi|` formula but avoids the
    1/|grad Psi| amplification that arises near the node from a Psi-side
    autodiff path.  As a result the debug path matches the analytic path
    (which sums analytic :math:`\nabla J` and :math:`\nabla \ln |\det|`)
    to ordinary autodiff round-off.

    Args:
        wavefunction_data: Wavefunction parameters (Jastrow + Geminal).
        r_up_carts: Cartesian coordinates of up-spin electrons with shape ``(n_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons with shape ``(n_dn, 3)``.

    Returns:
        Scalar nodal distance value.
    """
    r_up = cast_to_zone(r_up_carts, "wf_eval")
    r_dn = cast_to_zone(r_dn_carts, "wf_eval")

    grad_ln_Psi_up = grad(evaluate_ln_wavefunction, argnums=1)(wavefunction_data, r_up, r_dn)  # (n_up, 3)
    grad_ln_Psi_dn = grad(evaluate_ln_wavefunction, argnums=2)(wavefunction_data, r_up, r_dn)  # (n_dn, 3)

    grad_norm_sq = jnp.sum(grad_ln_Psi_up**2) + jnp.sum(grad_ln_Psi_dn**2)

    return 1.0 / jnp.sqrt(grad_norm_sq)
