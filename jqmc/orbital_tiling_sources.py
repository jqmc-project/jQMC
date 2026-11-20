"""Physical-supercell source assembly for periodic TREXIO and QE twist sectors.

Precision Zones: this module owns no Precision Zone; source conversion uses
NumPy float64/complex128.
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

import numpy as np

from .atomic_orbital import (
    AOs_cart_gto_data,
    AOs_composite_data,
    AOs_pGTO_base_data,
    _apply_image_expansion_complex,
    _promote_pgto_to_complex,
)
from .hamiltonians import _detile_aos_data_for_save
from .orbital_tiling import OrbitalKBlock_data, TiledOrbitalSet_data, group_tiling_k_points
from .structure import tile_structure_data


def _prepare_tiled_pgto_backend(backend):
    """Converge primitive image tails for cross-cell force/stress comparisons.

    Preserve home radial/angular data and coefficients. An already tighter
    caller-selected cutoff is retained; HDF5 persists the resulting tolerance.
    """
    from ._setting import TILING_PGTO_IMAGE_TRUNCATION_ATOL

    if isinstance(backend, AOs_composite_data):
        return backend.replace(children=tuple(_prepare_tiled_pgto_backend(child) for child in backend.children))
    if not isinstance(backend, AOs_pGTO_base_data):
        return backend
    if backend._image_truncation_tol <= TILING_PGTO_IMAGE_TRUNCATION_ATOL:
        return backend
    return _apply_image_expansion_complex(
        _detile_aos_data_for_save(backend),
        backend.structure_data,
        k_point=backend.k_point,
        tol=TILING_PGTO_IMAGE_TRUNCATION_ATOL,
    )


def expand_supercell_gto_basis(primitive_aos, physical):
    """Replicate the ordinary home GTO basis on physical atoms, with Gamma image sums.

    This independent basis is for J3, not the Slater k-block provider. Radial
    contractions, spherical/cartesian ordering and nucleus identities survive.
    """
    from ._setting import AO_IMAGE_TRUNCATION_ATOL

    home = _detile_aos_data_for_save(primitive_aos) if isinstance(primitive_aos, AOs_pGTO_base_data) else primitive_aos
    primitive_atoms = home.structure_data.natom
    if physical.natom % primitive_atoms:
        raise ValueError("supercell atom count must be a multiple of the primitive basis structure")
    count = physical.natom // primitive_atoms
    common = dict(
        structure_data=physical,
        k_point=None,
        num_ao=home.num_ao * count,
        num_ao_prim=home.num_ao_prim * count,
        nucleus_index=tuple(a + t * primitive_atoms for t in range(count) for a in home.nucleus_index),
        orbital_indices=tuple(a + t * home.num_ao for t in range(count) for a in home.orbital_indices),
        exponents=np.tile(home.exponents, count),
        coefficients=np.tile(home.coefficients, count),
        angular_momentums=tuple(home.angular_momentums) * count,
    )
    if home.ao_coefficient_scales:
        radial = home.radial
        common.update(
            radial=radial.replace(
                exponents=np.tile(radial.exponents, count),
                coefficients=np.tile(radial.coefficients, count),
                shell_prim_to_shell=tuple(s + t * radial.num_shell for t in range(count) for s in radial.shell_prim_to_shell),
                nucleus_index=tuple(a + t * primitive_atoms for t in range(count) for a in radial.nucleus_index),
                angular_momentum=radial.angular_momentum * count,
            ),
            ao_prim_to_shell_prim=tuple(
                p + t * radial.num_shell_prim for t in range(count) for p in home.ao_prim_to_shell_prim
            ),
            ao_coefficient_scales=home.ao_coefficient_scales * count,
        )
    if isinstance(home, AOs_cart_gto_data):
        common.update(
            {
                name: tuple(getattr(home, name)) * count
                for name in ("polynominal_order_x", "polynominal_order_y", "polynominal_order_z")
            }
        )
    else:
        common["magnetic_quantum_numbers"] = tuple(home.magnetic_quantum_numbers) * count
    expanded = type(home)._from_full_basis(**common)
    return _apply_image_expansion_complex(expanded, physical, k_point=None, tol=AO_IMAGE_TRUNCATION_ATOL)


def select_tiling_sector(groups, supertwist=None):
    """Select one inferred sector, or an explicit fractional supertwist modulo G."""
    from ._setting import ATOL_TWIST_CHECKPOINT_KPOINT_MATCH

    if supertwist is None:
        if len(groups) != 1:
            raise ValueError("source contains multiple supertwists; specify supertwist or enable twist_average")
        return groups[0]
    target = np.asarray(supertwist, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("supertwist must contain three finite fractional coordinates")
    for group in groups:
        delta = np.asarray(group[0]) - target
        if np.all(np.abs(delta - np.rint(delta)) <= ATOL_TWIST_CHECKPOINT_KPOINT_MATCH):
            return group
    raise ValueError("the source has no complete k coset for the requested supertwist")


def trexio_tiling_sectors(path, matrix):
    """Plan supercell twists from TREXIO fractional k points and source weights."""
    from .trexio_wrapper import read_trexio_kpoint_list

    metadata = read_trexio_kpoint_list(str(path))
    return group_tiling_k_points(matrix, [k for k, _ in metadata], [w for _, w in metadata])


def _trexio_occupied_source_rows(path, index, source_sd):
    """Recover input MO row identities in the reader's occupation-selected order."""
    import trexio

    with trexio.File(str(path), mode="r", back_end=trexio.TREXIO_HDF5) as source:
        occupations = np.asarray(trexio.read_mo_occupation(source))
        spins = np.asarray(trexio.read_mo_spin(source))
        rows = np.arange(len(occupations))
        if trexio.has_mo_k_point(source):
            rows = rows[np.asarray(trexio.read_mo_k_point(source)) == index]
    up = [row for row in rows if spins[row] == 0 and np.rint(occupations[row]) > 0]
    if source_sd.time_reversal:
        up = sorted(up, key=lambda row: -float(occupations[row]))[: source_sd.num_electron_up]
        down = up[: source_sd.num_electron_dn]
    else:
        up = up[: source_sd.num_electron_up]
        down = [row for row in rows if spins[row] == 1 and np.rint(occupations[row]) > 0][: source_sd.num_electron_dn]
    if len(up) != source_sd.num_electron_up or len(down) != source_sd.num_electron_dn:
        raise ValueError("TREXIO occupied source rows disagree with the selected determinant")
    return up, down


def read_tiled_trexio_sector(path, matrix, sector, *, expand_j3_basis=True):
    """Read one complete GTO source coset and assemble a (+k_s,-k_s) frozen SD.

    The TREXIO reader already stores conjugated down MOs (restricted and
    independent alike); undo that convention so both positive-sector sources
    are the +k rows before SD.build_tiled applies whole-orbital conjugation
    once.
    """
    from .coulomb_potential import build_ewald_data
    from .determinant import SD_data
    from .trexio_wrapper import read_trexio_file

    twist, indices, _ = sector
    blocks_up, blocks_dn = [], []
    first = None
    restricted = True
    for index in indices:
        loaded = read_trexio_file(str(path), store_tuple=True, twist_ik=index)
        primitive, aos, up, dn, source_sd, cp = loaded
        if first is None:
            first = loaded
        if not isinstance(aos, AOs_pGTO_base_data):
            raise ValueError("TREXIO tiling requires periodic GTO orbitals")
        backend = up.aos_data
        if backend.k_point is None:
            backend = _promote_pgto_to_complex(backend, (0.0, 0.0, 0.0))
        backend = _prepare_tiled_pgto_backend(backend)
        up_ids, down_ids = _trexio_occupied_source_rows(path, index, source_sd)
        up_order, down_order = np.argsort(up_ids), np.argsort(down_ids)
        blocks_up.append(
            OrbitalKBlock_data.build(
                backend=backend,
                coefficients=np.asarray(up.mo_coefficients, dtype=complex)[up_order],
                source_band_indices=tuple(np.asarray(up_ids)[up_order]),
            )
        )
        # The TREXIO reader stores every down channel as conj(C) of its own
        # +k source rows (on the -k leaf); undo it to recover the +k source.
        down_coefficients = np.asarray(dn.mo_coefficients, dtype=complex).conj()
        if not source_sd.time_reversal:
            restricted = False
        if dn.num_mo != up.num_mo:
            restricted = False
        blocks_dn.append(
            OrbitalKBlock_data.build(
                backend=backend,
                coefficients=down_coefficients[down_order],
                source_band_indices=tuple(np.asarray(down_ids)[down_order]),
            )
        )
    up = TiledOrbitalSet_data.build(blocks=blocks_up, tiling_matrix=matrix, supertwist=twist)
    down = None if restricted else TiledOrbitalSet_data.build(blocks=blocks_dn, tiling_matrix=matrix, supertwist=twist)
    sd = SD_data.build_tiled(up_source=up, down_source=down)
    primitive, aos, _, _, _, cp = first
    physical = tile_structure_data(primitive, matrix)
    count = physical.natom // primitive.natom
    updates = {
        name: None if getattr(cp, name) is None else tuple(getattr(cp, name)) * count
        for name in ("z_cores", "max_ang_mom_plus_1", "ang_moms", "exponents", "coefficients", "powers")
    }
    cp = cp.replace(
        structure_data=physical,
        ewald_data=None,
        num_ecps=None if cp.num_ecps is None else cp.num_ecps * count,
        nucleus_index=None
        if cp.nucleus_index is None
        else tuple(a + t * primitive.natom for t in range(count) for a in cp.nucleus_index),
        **updates,
    )
    cp = cp.replace(ewald_data=build_ewald_data(physical, cp, n_electron=sd.num_electron_up + sd.num_electron_dn))
    cp.sanity_check()
    j3_basis = expand_supercell_gto_basis(aos, physical) if expand_j3_basis else None
    return physical, j3_basis, up, sd.orb_data_dn_spin, sd, cp


def convert_qe_tiled_twist_average(qe_params, jastrow_params, output_path, groups):
    """Stream single-sector conversions into shared-supercell twist storage."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from ._twist_average import TwistAverageWriter
    from .hamiltonians import Hamiltonian_data
    from .jqmc_qe import run_qe2jqmc

    writer = None
    try:
        with TemporaryDirectory(prefix="jqmc-tiled-twists-") as directory:
            for index, (twist, _, weight) in enumerate(groups):
                path = str(Path(directory) / "sector.h5")
                run_qe2jqmc(dict(qe_params, twist_average=False, supertwist=twist), jastrow_params, path)
                ham = Hamiltonian_data.load_from_hdf5(path)
                if writer is None:
                    writer = TwistAverageWriter(output_path, ham, len(groups))
                writer.write_tiled_twist(index, weight, ham.wavefunction_data.geminal_data)
            writer.close()
    except BaseException:
        if writer is not None:
            writer._f.close()
        raise
