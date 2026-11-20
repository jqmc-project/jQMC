"""Primitive-block persistence, including shared spin payloads.

Hard links preserve *identity*, not numerical equality: independent LSDA data
is never deduplicated by value. The memo is local to one SD (or standalone
provider), so it cannot retain arrays beyond a save/load operation.
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

import dataclasses

import h5py
import numpy as np

from .orbital_tiling import OrbitalKBlock_data, TiledOrbitalSet_data


def save_tiled_orbitals(group, orbitals, memo):
    """Write metadata and local payloads, linking aliases within this save."""
    from .hamiltonians import _save_item

    orbitals.sanity_check()
    group.attrs["_class_name"] = type(orbitals).__name__
    group.attrs["_module_name"] = type(orbitals).__module__
    for field in dataclasses.fields(orbitals):
        if field.name != "blocks":
            _save_item(group, field.name, getattr(orbitals, field.name))
    blocks = group.create_group("blocks")
    for index, block in enumerate(orbitals.blocks):
        target = blocks.create_group(str(index))
        _save_item(target, "source_band_indices", block.source_band_indices)
        target.attrs["conjugate"] = block.conjugate
        target.attrs["coefficient_operator"] = "identity" if block.coefficients is None else "dense_local"
        for name in ("backend", "coefficients"):
            payload = getattr(block, name)
            if payload is None:
                continue  # local identity operator, never an identity dataset
            key = (name, id(payload))
            if key in memo:
                target[name] = memo[key]
            else:
                _save_item(target, name, payload)
                memo[key] = target[name]


def load_tiled_orbitals(group, memo):
    """Restore primitive backends once, before validating the outer supercell."""
    from .hamiltonians import (
        _align_orb_data_basis_storage,
        _coerce_nested_tuple,
        _expand_orb_data_for_pbc,
        _load_item,
    )

    # Validate the actual provider and normalization, without defaulting away a
    # missing physical convention or requiring an auxiliary schema marker.
    for name in ("normalization", "spin", "supertwist", "tiling_matrix", "structure_data", "blocks"):
        if name not in group and name not in group.attrs:
            raise ValueError(f"tiled orbital HDF5 is missing {name}")

    stored_blocks = group["blocks"]
    if set(stored_blocks) != {str(i) for i in range(len(stored_blocks))}:
        raise ValueError("tiled orbital HDF5 requires contiguous block indices")
    blocks = []
    for index in range(len(stored_blocks)):
        source = stored_blocks[str(index)]
        operator = source.attrs.get("coefficient_operator")
        if operator not in ("identity", "dense_local") or ("coefficients" in source) != (operator == "dense_local"):
            raise ValueError("tiled orbital HDF5 coefficient operator disagrees with its payload")
        payloads = {}
        for name in ("backend", "coefficients"):
            if name == "coefficients" and name not in source:
                payloads[name] = None
                continue
            node = source[name]
            key = (name, h5py.h5o.get_info(node.id).addr)
            if key not in memo:
                payload = _load_item(node)
                if name == "backend":
                    # A saved pGTO is molecular and carries its primitive cell.
                    # Using the Hamiltonian's physical cell here changes the
                    # Bloch sum and can multiply the image storage by |det S|.
                    payload = _expand_orb_data_for_pbc(payload, payload.structure_data)
                    payload = _align_orb_data_basis_storage(payload)
                memo[key] = payload
            payloads[name] = memo[key]
        conjugate = source.attrs["conjugate"]
        if not isinstance(conjugate, (bool, np.bool_)):
            raise ValueError("tiled orbital HDF5 conjugate must be boolean")
        blocks.append(
            OrbitalKBlock_data(
                **payloads,
                source_band_indices=tuple(source["source_band_indices"][()]),
                conjugate=bool(conjugate),
            )
        )
    orbitals = TiledOrbitalSet_data(
        structure_data=_load_item(group["structure_data"]),
        blocks=tuple(blocks),
        tiling_matrix=_coerce_nested_tuple(_load_item(group["tiling_matrix"])),
        supertwist=tuple(group["supertwist"][()]),
        spin=str(group.attrs["spin"]),
        normalization=str(group.attrs["normalization"]),
    )
    orbitals.sanity_check()
    return orbitals
