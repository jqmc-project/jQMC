r"""Primitive Bloch blocks supplying final physical-supercell orbitals.

This module defines the common block contract, independent of determinant and
source formats. Backends evaluate Bloch functions in a primitive cell; runtime,
native-kernel, and converter integrations share these contracts. Detailed
periodization and normalization exposition is work in progress.
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

from collections.abc import Sequence
from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct

from ._precision import cast_to_zone, get_complex_dtype_np
from ._setting import ATOL_TWIST_CHECKPOINT_KPOINT_MATCH
from .atomic_orbital import (
    AOs_base_data,
    AOs_composite_data,
    AOs_bspline_complex_data,
    AOs_cart_pgto_complex_data,
    AOs_plane_wave_complex_data,
    AOs_sphe_pgto_complex_data,
    _compute_AOs_debug,
    _structure_data_equal,
    check_ao_eval_value_range,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
    compute_AOs_value_grad_lap,
)
from .orbital_tiling_geometry import evaluate_physical_block
from .structure import (
    Structure_data,
    _tiling_determinant_adjugate,
    _tiling_quotient_residues,
    normalize_tiling_matrix,
    tile_structure_data,
)

logger = getLogger("jqmc").getChild(__name__)


def group_tiling_k_points(tiling, k_points, weights):
    """Partition a complete source mesh into supertwists, preserving source indices.

    Each sector must contain exactly one complete primitive reciprocal coset.
    Within-sector source weights must be equal; sector weights are their sums.
    This handles Gamma, shifted single sectors and weighted twist ensembles.
    """
    from ._setting import ATOL_TWIST_CHECKPOINT_KPOINT_MATCH, RTOL_BIT_EXACT

    matrix = normalize_tiling_matrix(tiling)
    if np.iscomplexobj(k_points) or np.iscomplexobj(weights):
        raise ValueError("tiling source k points and weights must be real")
    points = np.asarray(k_points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("tiling source k points must be a finite (n_k, 3) array")
    if weights.shape != (len(points),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("tiling source weights must be finite and strictly positive")
    sectors = []
    for index, k in enumerate(points):
        with np.errstate(over="ignore", invalid="ignore"):
            unfolded = matrix @ k
        if not np.isfinite(unfolded).all() or np.max(np.abs(unfolded)) >= 2**53:
            raise ValueError("tiling source k-coset integers must be resolvable in float64")
        twist = np.mod(unfolded, 1.0)
        twist[np.minimum(twist, 1 - twist) <= ATOL_TWIST_CHECKPOINT_KPOINT_MATCH] = 0
        for stored, indices in sectors:
            delta = twist - stored
            if np.all(np.abs(delta - np.rint(delta)) <= ATOL_TWIST_CHECKPOINT_KPOINT_MATCH):
                indices.append(index)
                break
        else:
            sectors.append((twist, [index]))
    result = []
    for twist, indices in sectors:
        order = match_tiling_k_points(matrix, points[indices], twist)
        ordered = tuple(indices[i] for i in order)
        relative_weights = weights[list(ordered)] / np.max(weights[list(ordered)])
        if not np.allclose(relative_weights, 1.0, atol=ATOL_TWIST_CHECKPOINT_KPOINT_MATCH, rtol=RTOL_BIT_EXACT):
            raise ValueError("tiling requires equal source weights within each complete k coset")
        with np.errstate(over="ignore"):
            sector_weight = float(weights[list(ordered)].sum())
        if not np.isfinite(sector_weight):
            raise ValueError("tiling sector weight sum must be finite")
        result.append((tuple(twist), ordered, sector_weight))
    if not result:
        raise ValueError("tiling requires a nonempty source k mesh")
    return result


def tiling_k_points(tiling: npt.ArrayLike, supertwist: npt.ArrayLike = (0.0, 0.0, 0.0)) -> np.ndarray:
    r"""Return canonical primitive fractional k representatives for one sector.

    The returned order is lexicographic in exact unshifted coset numerators.
    Detailed reciprocal-coset construction is work in progress.
    """
    matrix = normalize_tiling_matrix(tiling)
    twist = _validated_twist(supertwist)
    determinant, adjugate = _tiling_determinant_adjugate(matrix)
    shift = np.asarray(adjugate, dtype=np.float64) @ twist / determinant
    residues = np.asarray(_tiling_quotient_residues(matrix.T), dtype=np.float64) / abs(determinant)
    return (residues + shift) % 1.0


def _validated_twist(twist: npt.ArrayLike) -> np.ndarray:
    """Validate fractional reciprocal coordinates without choosing a new gauge."""
    raw = np.asarray(twist)
    if np.iscomplexobj(raw):
        raise ValueError("supertwist/k_point must be real fractional reciprocal coordinates.")
    point = np.asarray(raw, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("supertwist/k_point must contain three finite fractional reciprocal coordinates.")
    return point


def match_tiling_k_points(
    tiling: npt.ArrayLike, k_points: npt.ArrayLike, supertwist: npt.ArrayLike = (0.0, 0.0, 0.0)
) -> tuple[int, ...]:
    r"""Return source indices in canonical coset order, rejecting incomplete sets.

    A backend's original k gauge is preserved; only the blocks are reordered.
    Detailed matching and quotient-label construction is work in progress.
    """
    matrix = normalize_tiling_matrix(tiling)
    twist = _validated_twist(supertwist)
    determinant, adjugate = _tiling_determinant_adjugate(matrix)
    count = abs(determinant)
    raw_points = np.asarray(k_points)
    if np.iscomplexobj(raw_points):
        raise ValueError("primitive k points must be real.")
    points = np.asarray(raw_points, dtype=np.float64)
    if points.shape != (count, 3) or not np.all(np.isfinite(points)):
        raise ValueError(f"exactly |det(S)|={count} finite primitive k points are required.")
    by_coset = {}
    sign = 1 if determinant > 0 else -1
    for index, point in enumerate(points):
        with np.errstate(over="ignore", invalid="ignore"):
            unfolded = matrix @ point - twist
        if not np.all(np.isfinite(unfolded)) or np.max(np.abs(unfolded)) >= 2**53:
            raise ValueError(
                "k-coset matching requires fractional coordinates whose unfolded integers are resolvable in float64."
            )
        rounded = np.rint(unfolded)
        if np.max(np.abs(unfolded - rounded)) > ATOL_TWIST_CHECKPOINT_KPOINT_MATCH:
            raise ValueError(f"primitive k point {index} is incompatible: require S k = supertwist + integer.")
        integer = tuple(int(value) for value in rounded)
        key = tuple(sign * sum(a * m for a, m in zip(row, integer, strict=True)) % count for row in adjugate)
        if key in by_coset:
            raise ValueError(f"primitive k points {by_coset[key]} and {index} belong to the same reciprocal coset.")
        by_coset[key] = index
    return tuple(by_coset[key] for key in _tiling_quotient_residues(matrix.T))


@struct.dataclass
class OrbitalKBlock_data:
    """One primitive k/spin block and its local occupied-orbital operator.

    ``coefficients=None`` means the backend already supplies the selected final
    bands. Otherwise rows of C are occupied orbitals and columns are backend
    basis functions. Logical conjugation applies after contraction, sharing
    both backend and C storage. It changes the effective k to minus backend k.
    """

    backend: AOs_base_data
    coefficients: object = None
    source_band_indices: tuple[int, ...] = struct.field(pytree_node=False, default=())
    conjugate: bool = struct.field(pytree_node=False, default=False)

    @property
    def k_point(self) -> tuple[float, float, float]:
        sign = -1 if self.conjugate else 1
        backend = self.backend.children[0] if isinstance(self.backend, AOs_composite_data) else self.backend
        return tuple(sign * float(value) for value in backend.k_point)

    @property
    def num_orbitals(self) -> int:
        return len(self.source_band_indices)

    @classmethod
    def build(cls, *, backend, source_band_indices, coefficients=None, conjugate=False):
        """Validate local storage; do not rescale or pad it."""
        source_indices = tuple(source_band_indices)
        operator = None if coefficients is None else np.asarray(coefficients, dtype=np.complex128)
        data = cls(backend=backend, coefficients=operator, source_band_indices=source_indices, conjugate=conjugate)
        data.sanity_check()
        return data

    def sanity_check(self) -> None:
        # Exact types exclude the provisional spline subclass, whose tables
        # are already scaled and whose geometry is not a primitive backend.
        children = self.backend.children if type(self.backend) is AOs_composite_data else (self.backend,)
        allowed = (
            AOs_bspline_complex_data,
            AOs_plane_wave_complex_data,
            AOs_cart_pgto_complex_data,
            AOs_sphe_pgto_complex_data,
        )
        if any(type(child) not in allowed for child in children):
            raise ValueError("a tiled block requires primitive complex B-spline, PW or pGTO leaves (also at Gamma).")
        if type(self.backend) is AOs_composite_data:
            if (
                len(children) != 2
                or sum(type(c) is AOs_plane_wave_complex_data for c in children) != 1
                or sum(type(c) in (AOs_cart_pgto_complex_data, AOs_sphe_pgto_complex_data) for c in children) != 1
            ):
                raise ValueError("tiled composites require one complex pGTO and one complex PW child.")
            reference_k = _validated_twist(children[0].k_point)
            if any(
                np.max(np.abs(_validated_twist(c.k_point) - reference_k)) > ATOL_TWIST_CHECKPOINT_KPOINT_MATCH
                for c in children[1:]
            ):
                raise ValueError("tiled composite children must have the same primitive k representative.")
        for child in children:
            _validated_twist(child.k_point)
        if not isinstance(self.conjugate, (bool, np.bool_)):
            raise ValueError("logical conjugation must be boolean.")
        ids = self.source_band_indices
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0 for value in ids):
            raise ValueError("source_band_indices must be nonnegative integers.")
        if tuple(sorted(set(ids))) != ids:
            raise ValueError("source_band_indices must be unique and increasing; reorder coefficient rows with the bands.")
        if self.coefficients is None:
            if self.num_orbitals != self.backend.num_ao:
                raise ValueError("identity block requires one selected source band per backend row.")
        else:
            operator = np.asarray(self.coefficients)
            if operator.shape != (self.num_orbitals, self.backend.num_ao) or not np.all(np.isfinite(operator)):
                raise ValueError("local coefficients must be finite with shape (occupied bands, primitive basis size).")
        self.backend.sanity_check()
        check_ao_eval_value_range(self.backend)


@struct.dataclass
class TiledOrbitalSet_data:
    r"""Final orbitals for one spin in a physical supercell, with ragged k blocks.

    The stored primitive backend data is never expanded into a global basis.
    Input coefficients/bands are normalized in the primitive cell. Frozen
    coefficients are nonvariational.
    """

    structure_data: Structure_data
    blocks: tuple[OrbitalKBlock_data, ...]
    tiling_matrix: tuple[tuple[int, int, int], ...] = struct.field(pytree_node=False)
    supertwist: tuple[float, float, float] = struct.field(pytree_node=False, default=(0.0, 0.0, 0.0))
    spin: str = struct.field(pytree_node=False, default="up")
    normalization: str = struct.field(pytree_node=False, default="primitive-cell-to-supercell")

    @property
    def num_mo(self) -> int:
        return sum(block.num_orbitals for block in self.blocks)

    @property
    def _num_orb(self) -> int:
        return self.num_mo

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.complex128)

    @property
    def orbital_offsets(self) -> tuple[int, ...]:
        offsets = [0]
        for block in self.blocks:
            offsets.append(offsets[-1] + block.num_orbitals)
        return tuple(offsets)

    @property
    def primitive_structure(self) -> Structure_data:
        return self.blocks[0].backend.structure_data

    @classmethod
    def build(cls, *, blocks: Sequence[OrbitalKBlock_data], tiling_matrix, supertwist=(0.0, 0.0, 0.0), spin="up"):
        """Build the physical structure and put source blocks in canonical order."""
        matrix = normalize_tiling_matrix(tiling_matrix)
        blocks = tuple(blocks)
        for block in blocks:
            block.sanity_check()
        order = match_tiling_k_points(matrix, [block.k_point for block in blocks], supertwist)
        ordered = tuple(blocks[index] for index in order)
        data = cls(
            structure_data=tile_structure_data(ordered[0].backend.structure_data, matrix),
            blocks=ordered,
            tiling_matrix=tuple(tuple(int(value) for value in row) for row in matrix),
            supertwist=tuple(float(value) for value in supertwist),
            spin=spin,
        )
        data.sanity_check()
        return data

    def sanity_check(self) -> None:
        if self.normalization != "primitive-cell-to-supercell":
            raise ValueError("unsupported tiled orbital normalization convention.")
        if self.spin not in ("up", "down"):
            raise ValueError("a tiled orbital set describes exactly one spin channel: 'up' or 'down'.")
        for block in self.blocks:
            block.sanity_check()
        order = match_tiling_k_points(self.tiling_matrix, [block.k_point for block in self.blocks], self.supertwist)
        if order != tuple(range(len(self.blocks))):
            raise ValueError("tiled blocks must be stored in canonical reciprocal-coset order.")
        primitive = self.primitive_structure
        if any(not _structure_data_equal(block.backend.structure_data, primitive) for block in self.blocks):
            raise ValueError("every k block must describe the same primitive structure and lattice.")
        if not _structure_data_equal(self.structure_data, tile_structure_data(primitive, self.tiling_matrix)):
            raise ValueError("physical structure must be the declared tiling of the primitive structure.")

    def _get_info(self) -> list[str]:
        return [
            "**TiledOrbitalSet_data (frozen)",
            f"  spin={self.spin}, supertwist={tuple(float(v) for v in self.supertwist)}, tiling={self.tiling_matrix}",
            f"  primitive k blocks={len(self.blocks)}, occupied orbitals={self.num_mo}",
            f"  local basis sizes={tuple(block.backend.num_ao for block in self.blocks)}",
            f"  orbital offsets={self.orbital_offsets}, normalization={self.normalization}",
        ]

    def _logger_info(self) -> None:
        for line in self._get_info():
            logger.info(line)


def conjugate_tiled_orbitals(orbitals: TiledOrbitalSet_data, *, spin: str) -> TiledOrbitalSet_data:
    """Reverse the entire Bloch sector, sharing primitive tables and local C.

    The selected BCS convention applies this to the down-spin source provider,
    also for LSDA. That does not imply any relation between LSDA up/down C.
    Canonical row ordering is recomputed in the negative sector.
    """
    return TiledOrbitalSet_data.build(
        blocks=[block.replace(conjugate=not block.conjugate) for block in orbitals.blocks],
        tiling_matrix=orbitals.tiling_matrix,
        supertwist=tuple(-value for value in orbitals.supertwist),
        spin=spin,
    )


def _contract_block(block: OrbitalKBlock_data, values: jax.Array, zone: str) -> jax.Array:
    """Contract only a local C; freeze its parameters while retaining electron AD."""
    contracted = cast_to_zone(values, zone)
    if block.coefficients is not None:
        coefficients = cast_to_zone(jax.lax.stop_gradient(jnp.asarray(block.coefficients)), zone)
        contracted = coefficients @ contracted
    return jnp.conj(contracted) if block.conjugate else contracted


@jax.jit
def compute_tiled_orbitals(orbitals: TiledOrbitalSet_data, r_carts: jax.Array) -> jax.Array:
    """Return final orbital rows, with supercell normalization applied once."""
    values = [
        _contract_block(block, evaluate_physical_block(orbitals, block, r_carts, compute_AOs), "tiling_eval")
        for block in orbitals.blocks
    ]
    return jnp.concatenate(values, axis=0) / np.sqrt(len(orbitals.blocks))


@jax.jit
def compute_tiled_orbitals_grad(orbitals: TiledOrbitalSet_data, r_carts: jax.Array) -> tuple[jax.Array, ...]:
    """Return analytic Cartesian gradients of final orbitals."""
    gradients = [
        tuple(
            _contract_block(block, v, "tiling_grad")
            for v in evaluate_physical_block(orbitals, block, r_carts, compute_AOs_grad)
        )
        for block in orbitals.blocks
    ]
    return tuple(jnp.concatenate([g[axis] for g in gradients], axis=0) / np.sqrt(len(orbitals.blocks)) for axis in range(3))


@jax.jit
def compute_tiled_orbitals_laplacian(orbitals: TiledOrbitalSet_data, r_carts: jax.Array) -> jax.Array:
    """Return analytic Laplacians of final orbitals."""
    values = [
        _contract_block(block, evaluate_physical_block(orbitals, block, r_carts, compute_AOs_laplacian), "tiling_lap")
        for block in orbitals.blocks
    ]
    return jnp.concatenate(values, axis=0) / np.sqrt(len(orbitals.blocks))


@jax.jit
def compute_tiled_orbitals_value_grad_lap(orbitals: TiledOrbitalSet_data, r_carts: jax.Array) -> tuple[jax.Array, ...]:
    """Return fused (value, gx, gy, gz, lap), retaining each component's zone."""
    zones = ("tiling_eval", "tiling_grad", "tiling_grad", "tiling_grad", "tiling_lap")
    blocks = [
        tuple(
            _contract_block(block, value, zone)
            for value, zone in zip(
                evaluate_physical_block(orbitals, block, r_carts, compute_AOs_value_grad_lap), zones, strict=True
            )
        )
        for block in orbitals.blocks
    ]
    return tuple(jnp.concatenate([block[i] for block in blocks], axis=0) / np.sqrt(len(orbitals.blocks)) for i in range(5))


def _compute_tiled_orbitals_debug(orbitals: TiledOrbitalSet_data, r_carts: npt.ArrayLike) -> np.ndarray:
    """Independent NumPy backend evaluation followed by local NumPy contractions."""
    values = []
    dtype = get_complex_dtype_np("tiling_eval")
    for block in orbitals.blocks:
        local = np.asarray(_compute_AOs_debug(block.backend, np.asarray(r_carts)), dtype=dtype)
        if block.coefficients is not None:
            local = np.asarray(block.coefficients, dtype=dtype) @ local
        values.append(np.conj(local) if block.conjugate else local)
    return np.concatenate(values, axis=0) / np.sqrt(len(orbitals.blocks))


@jax.jit
def _compute_tiled_orbitals_value_grad_lap_autodiff(orbitals: TiledOrbitalSet_data, r_carts: jax.Array):
    """Differentiate final values by electron position as a VGL cross-check."""

    def single(position):
        return compute_tiled_orbitals(orbitals, position[None, :])[:, 0]

    gradient = jax.vmap(jax.jacfwd(single))(r_carts)
    hessian = jax.vmap(jax.jacfwd(jax.jacfwd(single)))(r_carts)
    return (
        compute_tiled_orbitals(orbitals, r_carts),
        *(gradient[:, :, axis].T for axis in range(3)),
        jnp.trace(hessian, axis1=-2, axis2=-1).T,
    )


def tiled_overlap_blocks(orbitals: TiledOrbitalSet_data) -> tuple[np.ndarray, ...]:
    r"""Return local occupied Gram blocks; distinct cosets are exactly orthogonal.

    These diagnostics are explicitly requested, never part of logging/loading.
    Detailed overlap normalization and contraction exposition is work in progress.
    """
    overlaps = []
    for block in orbitals.blocks:
        coefficients = block.coefficients
        if isinstance(block.backend, AOs_plane_wave_complex_data) and coefficients is not None:
            local = abs(np.linalg.det(block.backend.structure_data.cell)) * (coefficients.conj() @ coefficients.T)
        else:
            local = block.backend._compute_overlap_matrix()
            if coefficients is not None:
                local = coefficients.conj() @ local @ coefficients.T
        overlaps.append(np.conj(local) if block.conjugate else local)
    return tuple(overlaps)
