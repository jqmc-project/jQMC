"""Physical-supercell continuation of frozen primitive orbital blocks.

Primitive radial tables, band coefficients and fractional Bloch labels are
reference data. Only the provider's physical positions and lattice deform.
Precision Zones: coordinate maps use float64, AO kernels retain their zones.
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

from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from ._setting import ATOL_TILING_FRACTIONAL_BOUNDARY_SNAP
from .atomic_orbital import AOs_cart_pgto_complex_data, AOs_sphe_pgto_complex_data, AOs_composite_data
from .structure import _inv_3x3, _tiling_determinant_adjugate, _tiling_quotient_residues


class _CenterOffsetView:
    """Transient kernel geometry; the inherited structure still describes real atoms."""

    @property
    def _expanded_positions_cart_jnp(self):
        return super()._expanded_positions_cart_jnp + self.center_offsets

    @property
    def _expanded_positions_cart_np(self):
        return super()._expanded_positions_cart_np + np.asarray(self.center_offsets)


@struct.dataclass
class _Tiled_cart_pgto_complex_view(_CenterOffsetView, AOs_cart_pgto_complex_data):
    center_offsets: object = None


@struct.dataclass
class _Tiled_sphe_pgto_complex_view(_CenterOffsetView, AOs_sphe_pgto_complex_data):
    center_offsets: object = None


def physical_block_backend(orbitals, backend):
    """Derive the live primitive lattice from the physical cell; freeze source data."""
    if isinstance(backend, AOs_composite_data):
        return backend.replace(children=tuple(physical_block_backend(orbitals, child) for child in backend.children))
    source = jax.tree.map(jax.lax.stop_gradient, backend)
    reference = source.structure_data
    cell = jnp.asarray(np.linalg.inv(np.asarray(orbitals.tiling_matrix, dtype=float))) @ orbitals.structure_data._cell_jnp
    positions = reference._positions_cart_jnp @ _inv_3x3(reference._cell_jnp) @ cell
    return source.replace(structure_data=reference.replace(positions=positions, vec_a=cell[0], vec_b=cell[1], vec_c=cell[2]))


def physical_center_displacements(orbitals):
    """Independent SC displacements relative to the affinely deformed reference.

    Reproduce tile_structure_data's translation-major wrapping and snapping.
    No stored SC AO table or per-k copy of physical atoms is introduced.
    """
    reference = jax.tree.map(jax.lax.stop_gradient, orbitals.primitive_structure)
    matrix = np.asarray(orbitals.tiling_matrix, dtype=float)
    count = len(orbitals.blocks)
    translations = jnp.asarray(np.asarray(_tiling_quotient_residues(orbitals.tiling_matrix), dtype=float) / count)
    frac = reference._positions_cart_jnp @ _inv_3x3(reference._cell_jnp) @ jnp.asarray(np.linalg.inv(matrix))
    frac = frac[None, :, :] + translations[:, None, :]
    frac = frac - jnp.floor(frac)
    frac = jnp.where(jnp.abs(frac - 1) <= ATOL_TILING_FRACTIONAL_BOUNDARY_SNAP, 0, frac)
    anchors = frac.reshape(-1, 3) @ orbitals.structure_data._cell_jnp
    return orbitals.structure_data._positions_cart_jnp - anchors


def image_physical_atoms(orbitals, backend, position):
    """Map each primitive image to its physical atom, including electron wrapping.

    Binary lexicographic lookup avoids a kernel-by-supercell-size match table.
    All quotient arithmetic is integer; reciprocal and real-space quotients
    deliberately use different transposes.
    """
    matrix = orbitals.tiling_matrix
    determinant, adjugate = _tiling_determinant_adjugate(matrix)
    count = abs(determinant)
    keys = jnp.asarray(_tiling_quotient_residues(matrix), dtype=jnp.int64)
    n0 = jnp.floor(jnp.asarray(position, jnp.float64) @ _inv_3x3(backend.structure_data._cell_jnp)).astype(jnp.int64)
    image = jnp.asarray(backend._image_indices, dtype=jnp.int64)
    labels = jnp.mod((image + n0) @ jnp.asarray(adjugate, dtype=jnp.int64) * (1 if determinant > 0 else -1), count)
    low = jnp.zeros(len(backend._image_indices), dtype=jnp.int32)
    high = jnp.full_like(low, count)

    def bisect(_, state):
        lo, hi = state
        mid = (lo + hi) // 2
        key = keys[jnp.minimum(mid, count - 1)]
        less = (key[:, 0] < labels[:, 0]) | (
            (key[:, 0] == labels[:, 0])
            & ((key[:, 1] < labels[:, 1]) | ((key[:, 1] == labels[:, 1]) & (key[:, 2] < labels[:, 2])))
        )
        return jnp.where(less, mid + 1, lo), jnp.where(less, hi, mid)

    index, _ = jax.lax.fori_loop(0, count.bit_length(), bisect, (low, high))
    return index * backend.structure_data.natom + jnp.asarray(backend._ragged_kernel_atom, jnp.int32)


def physical_pgto_view(orbitals, backend, position, displacements):
    """Per-electron pGTO view sharing every radial/image table with its primitive leaf."""
    cls = _Tiled_cart_pgto_complex_view if isinstance(backend, AOs_cart_pgto_complex_data) else _Tiled_sphe_pgto_complex_view
    atoms = image_physical_atoms(orbitals, backend, position)
    return cls(**{field.name: getattr(backend, field.name) for field in fields(backend)}, center_offsets=displacements[atoms])


def evaluate_physical_block(orbitals, block, r_carts, evaluator):
    """Evaluate one primitive block with live physical geometry and unchanged rank."""
    if isinstance(block.backend, AOs_composite_data):
        values = [
            evaluate_physical_block(orbitals, block.replace(backend=child), r_carts, evaluator)
            for child in block.backend.children
        ]
        return jax.tree.map(lambda *parts: jnp.concatenate(parts, axis=0), *values)
    backend = physical_block_backend(orbitals, block.backend)
    if not isinstance(backend, (AOs_cart_pgto_complex_data, AOs_sphe_pgto_complex_data)):
        return evaluator(backend, r_carts)
    displacements = physical_center_displacements(orbitals)

    def one(position):
        view = physical_pgto_view(orbitals, backend, position, displacements)
        return jax.tree.map(lambda value: value[:, 0], evaluator(view, position[None, :]))

    return jax.tree.map(lambda value: value.T, jax.vmap(one)(jnp.asarray(r_carts)))
