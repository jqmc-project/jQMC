"""QE projections to primitive k blocks, independent of determinant packaging.

Coefficients retain the projection's primitive-cell normalization. This adapter
does not assemble a supercell coefficient matrix or change spin boundary sectors.
The determinant adapter explicitly conjugates the down-spin source provider.
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

import numpy as np

from .atomic_orbital import AOs_bspline_complex_data, compute_overlap_matrix
from .determinant import SD_data
from .orbital_tiling import OrbitalKBlock_data, TiledOrbitalSet_data
from .orbital_tiling_sources import _prepare_tiled_pgto_backend
from .pw_pgto_projection import ProjectionResult
from .qe_to_jqmc import (
    _build_pgto_leaf_and_mo,
    _build_pw_leaf_and_mo,
    _build_structure_data,
    _orthonormalize_mo_coefficients_band_order,
    _require_nonsingular_occupied_pgto_mos,
    check_binary_occupations,
)
from .qe_wrapper import QEReference


def build_tiled_wavefunction(
    references: Sequence[QEReference],
    projections: Sequence[ProjectionResult],
    *,
    kind: str,
    tiling_matrix,
    supertwist=(0.0, 0.0, 0.0),
    references_down: Sequence[QEReference] | None = None,
    projections_down: Sequence[ProjectionResult] | None = None,
    orthonormalize=True,
):
    """Assemble a frozen SD from positive-sector QE sources for both spins.

    Independent LSDA sources retain their own payloads. SD construction applies
    whole-orbital conjugation to the down source, yielding (+k_s, -k_s).
    The returned provider replaces the AO argument of Hamiltonian assembly;
    only the physical structure and SD are consumed by that assembly.
    """
    if (references_down is None) != (projections_down is None):
        raise ValueError("tiled LSDA requires both down-spin references and projections")
    options = dict(kind=kind, tiling_matrix=tiling_matrix, supertwist=supertwist, orthonormalize=orthonormalize)
    up = build_tiled_orbitals_from_projections(references, projections, **options)
    down = None
    if references_down is not None:
        down = build_tiled_orbitals_from_projections(references_down, projections_down, spin="down", **options)
    sd = SD_data.build_tiled(up_source=up, down_source=down)
    return up.structure_data, up, sd


def build_tiled_orbitals_from_projections(
    references: Sequence[QEReference],
    projections: Sequence[ProjectionResult],
    *,
    kind: str,
    tiling_matrix,
    supertwist=(0.0, 0.0, 0.0),
    spin="up",
    orthonormalize=True,
) -> TiledOrbitalSet_data:
    """Convert a complete single-spin source sector, retaining occupied band IDs.

    All three kinds use complex primitive leaves, including Gamma. PW basis
    sizes and spline grids may differ across k. pGTO orthonormalization acts on
    occupied rows only, preserving non-Aufbau occupied subspaces. Mixed pGTO+PW
    retains the joint projection coefficients without separate normalization. Fractional/nonfinite occupations and omitted
    occupied bands fail before building a provider.
    """
    if kind not in ("pw", "bspline", "pgto", "pgto+pw"):
        raise ValueError("tiled projections support pw, bspline, pgto and pgto+pw.")
    if len(references) != len(projections) or not references:
        raise ValueError("one projection per primitive k reference is required.")
    blocks = []
    for index, (reference, projection) in enumerate(zip(references, projections, strict=True)):
        structure, k_point = _build_structure_data(reference)
        k_point = (0.0, 0.0, 0.0) if k_point is None else k_point
        source_occupations = np.asarray(reference.bands.occupations)
        occupations = np.asarray(projection.occupations)
        ids = np.asarray(projection.band_indices)
        if source_occupations.ndim != 1 or not np.all(np.isfinite(source_occupations)):
            raise ValueError(f"k[{index}]: source occupations must be a finite vector.")
        if occupations.ndim != 1 or not np.all(np.isfinite(occupations)):
            raise ValueError(f"k[{index}]: projected occupations must be a finite vector.")
        check_binary_occupations(source_occupations, channel=f"{spin} k[{index}]")
        check_binary_occupations(occupations, channel=f"{spin} k[{index}]")
        if (
            ids.shape != occupations.shape
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0)
            or np.any(ids >= len(source_occupations))
            or len(set(ids.tolist())) != len(ids)
        ):
            raise ValueError(f"k[{index}]: invalid projected source-band indices.")
        if not np.array_equal(occupations > 0.5, source_occupations[ids] > 0.5):
            raise ValueError(f"k[{index}]: projection occupations disagree with the source bands.")
        selected = np.flatnonzero(occupations > 0.5)
        selected = selected[np.argsort(ids[selected])]
        occupied_ids = ids[selected]
        if not np.array_equal(occupied_ids, np.flatnonzero(source_occupations > 0.5)):
            raise ValueError(f"k[{index}]: projection omitted occupied source bands.")
        if kind == "bspline":
            table = np.asarray(projection.spline_coefficients)
            if table.ndim != 2 or table.shape[0] != len(ids) or not projection.spline_shape:
                raise ValueError(f"k[{index}]: invalid per-band spline projection.")
            backend = AOs_bspline_complex_data.build(
                structure_data=structure,
                k_point=k_point,
                spline_coefficients=table[selected],
                spline_shape=projection.spline_shape,
                spline_order=projection.spline_order,
            )
            local = None
        else:
            if kind == "pw":
                backend, coefficients = _build_pw_leaf_and_mo(reference, projection, structure, k_point)
            elif kind == "pgto+pw":
                from .qe_to_jqmc import _build_pgto_pw_composite

                backend, coefficients = _build_pgto_pw_composite(reference, projection, structure, k_point)
                backend = _prepare_tiled_pgto_backend(backend)
            else:
                backend, coefficients = _build_pgto_leaf_and_mo(projection, structure, reference.atom_positions, k_point)
                backend = _prepare_tiled_pgto_backend(backend)
            if coefficients.shape[0] != len(ids):
                raise ValueError(f"k[{index}]: coefficient rows disagree with projected band IDs.")
            local = np.ascontiguousarray(coefficients[selected])
            if kind == "pgto" and len(selected):
                # Issue #580: reject a zero / rank-deficient occupied block
                # (vanishing determinant) with or without orthonormalization.
                overlap = np.asarray(compute_overlap_matrix(backend))
                _require_nonsingular_occupied_pgto_mos(
                    local, overlap, label=f"tiled {spin} k[{index}] occupied", projection_result=projection
                )
                if orthonormalize:
                    local, _ = _orthonormalize_mo_coefficients_band_order(
                        local, overlap, label=f"tiled {spin} k[{index}] occupied"
                    )
        blocks.append(OrbitalKBlock_data.build(backend=backend, coefficients=local, source_band_indices=tuple(occupied_ids)))
    return TiledOrbitalSet_data.build(blocks=blocks, tiling_matrix=tiling_matrix, supertwist=supertwist, spin=spin)
