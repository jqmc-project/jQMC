"""Independent explicit-supercell test builders; dense arrays are test-only."""

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


def explicit_supercell_pgto(orbitals):
    """Ordinary SC GTO/MO oracle, with translation phases evaluated from actual centers.

    Test-only dense C. Uses the source home basis metadata, without the tiled
    evaluator, quotient matcher or production supercell-J3 basis constructor.
    """
    from jqmc.atomic_orbital import AOs_cart_gto_data, AOs_sphe_gto_data, _apply_image_expansion_complex
    from jqmc.hamiltonians import _detile_aos_data_for_save
    from jqmc.molecular_orbital import MOs_data
    from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL

    home = _detile_aos_data_for_save(orbitals.blocks[0].backend)
    physical = orbitals.structure_data
    nprim = home.structure_data.natom
    count = physical.natom // nprim
    common = dict(
        structure_data=physical,
        num_ao=home.num_ao * count,
        num_ao_prim=home.num_ao_prim * count,
        nucleus_index=tuple(i + t * nprim for t in range(count) for i in home.nucleus_index),
        orbital_indices=tuple(i + t * home.num_ao for t in range(count) for i in home.orbital_indices),
        exponents=np.tile(home.exponents, count),
        coefficients=np.tile(home.coefficients, count),
        angular_momentums=tuple(home.angular_momentums) * count,
    )
    if isinstance(home, AOs_cart_gto_data):
        cls = AOs_cart_gto_data
        for name in ("polynominal_order_x", "polynominal_order_y", "polynominal_order_z"):
            common[name] = tuple(getattr(home, name)) * count
    else:
        cls = AOs_sphe_gto_data
        common["magnetic_quantum_numbers"] = tuple(home.magnetic_quantum_numbers) * count
    basis = _apply_image_expansion_complex(
        cls._from_full_basis(**common), physical, k_point=orbitals.supertwist, tol=AO_IMAGE_TRUNCATION_ATOL**2
    )
    shifts = np.asarray(physical.positions).reshape(count, nprim, 3) - np.asarray(home.structure_data.positions)[None, :, :]
    rows = []
    for block in orbitals.blocks:
        kcart = np.asarray(block.k_point) @ home.structure_data.recip_cell
        phases = np.exp(1j * (shifts @ kcart))[:, np.asarray(home.nucleus_index)]
        C = np.eye(home.num_ao) if block.coefficients is None else np.asarray(block.coefficients)
        if block.conjugate:
            C = C.conj()
        rows.append(np.concatenate([C * phase[None, :] for phase in phases], axis=1) / np.sqrt(count))
    return MOs_data.build(aos_data=basis, mo_coefficients=np.concatenate(rows))


def complete_pgto_source(kind, independent=False):
    """Complete s/p/d shells for full ordinary-supercell derivative assembly."""
    from tests.test_jqmc_kernel.test_orbital_tiling import _primitive
    from jqmc.atomic_orbital import build_AOs_sphe_pgto_complex_data
    from jqmc.orbital_tiling import TiledOrbitalSet_data, OrbitalKBlock_data
    from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL

    primitive = _primitive()
    matrix = np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    twist = np.array([0.23, -0.17, 0.31])
    rng = np.random.default_rng(71 if independent else 19)
    blocks = []
    for i in range(2):
        k = tuple(np.linalg.solve(matrix, twist + [i, 0, 0]))
        basis = build_AOs_sphe_pgto_complex_data(
            structure_data=primitive,
            k_point=k,
            nucleus_index=(0,) * 9,
            num_ao=9,
            num_ao_prim=9,
            orbital_indices=tuple(range(9)),
            exponents=np.repeat([0.8, 1.1, 1.3], [1, 3, 5]),
            coefficients=np.ones(9),
            angular_momentums=(0,) + (1,) * 3 + (2,) * 5,
            magnetic_quantum_numbers=(0, -1, 0, 1, -2, -1, 0, 1, 2),
            tol=AO_IMAGE_TRUNCATION_ATOL**2,
        )
        if kind == "pgto_cart":
            from jqmc.atomic_orbital import _aos_sphe_to_cart

            basis = _aos_sphe_to_cart(basis)[0]
        C = rng.normal(size=(1, basis.num_ao)) + 1j * rng.normal(size=(1, basis.num_ao))
        blocks.append(OrbitalKBlock_data.build(backend=basis, coefficients=C, source_band_indices=(2,)))
    return TiledOrbitalSet_data.build(blocks=blocks, tiling_matrix=matrix, supertwist=twist)


def explicit_supercell_pw(orbitals):
    """Test-only ordinary PW union and dense MO matrix in physical reciprocal units."""
    from jqmc.atomic_orbital import AOs_plane_wave_complex_data
    from jqmc.molecular_orbital import MOs_data

    structure = orbitals.structure_data
    reciprocal = np.asarray(structure.recip_cell)
    twist_cart = np.asarray(orbitals.supertwist) @ reciprocal
    waves, blocks = [], []
    for block in orbitals.blocks:
        source = block.backend
        vectors = np.asarray(source.g_vectors) + np.asarray(source.k_point) @ source.structure_data.recip_cell
        C = np.eye(source.num_ao, dtype=complex) if block.coefficients is None else np.asarray(block.coefficients)
        if block.conjugate:
            vectors, C = -vectors, C.conj()
        waves.append(vectors - twist_cart[None, :])
        blocks.append(C / np.sqrt(len(orbitals.blocks)))
    basis = AOs_plane_wave_complex_data.build(structure, np.concatenate(waves), k_point=orbitals.supertwist)
    coefficients = np.zeros((orbitals.num_mo, basis.num_ao), dtype=complex)
    row = column = 0
    for C in blocks:
        coefficients[row : row + len(C), column : column + C.shape[1]] = C
        row += len(C)
        column += C.shape[1]
    return MOs_data.build(aos_data=basis, mo_coefficients=coefficients)


def explicit_supercell_bspline(orbitals):
    """Replicate each control lattice exactly, keeping its supercell Bloch gauge."""
    from jqmc.atomic_orbital import AOs_bspline_complex_data, AOs_composite_data
    from jqmc.molecular_orbital import MOs_data

    repeats = np.diag(orbitals.tiling_matrix)
    assert np.array_equal(np.asarray(orbitals.tiling_matrix), np.diag(repeats))
    children, local = [], []
    for block in orbitals.blocks:
        source = block.backend
        cube = np.asarray(source.spline_coefficients).reshape(source.num_ao, *source.spline_shape)
        C = np.eye(source.num_ao, dtype=complex) if block.coefficients is None else np.asarray(block.coefficients)
        # Build the positive-sector child; a conjugated (down-spin) block is
        # its with_negated_k() partner, exactly as every production down leaf.
        child = AOs_bspline_complex_data.build(
            orbitals.structure_data,
            np.tile(cube, (1, *repeats)).reshape(source.num_ao, -1),
            tuple(np.asarray(source.spline_shape) * repeats),
            spline_order=source.spline_order,
            k_point=tuple(np.asarray(orbitals.tiling_matrix) @ np.asarray(source.k_point)),
        )
        if block.conjugate:
            child, C = child.with_negated_k(), C.conj()
        children.append(child)
        local.append(C / np.sqrt(len(orbitals.blocks)))
    basis = AOs_composite_data.build(children)
    coefficients = np.zeros((orbitals.num_mo, basis.num_ao), dtype=complex)
    row = column = 0
    for C in local:
        coefficients[row : row + len(C), column : column + C.shape[1]] = C
        row += len(C)
        column += C.shape[1]
    return MOs_data.build(aos_data=basis, mo_coefficients=coefficients)


def explicit_supercell_composite(orbitals):
    """Expand both child families independently, then join their dense SC operators."""
    from jqmc.atomic_orbital import AOs_plane_wave_complex_data
    from jqmc.molecular_orbital import MOs_data

    groups = {"pgto": [], "pw": []}
    for block in orbitals.blocks:
        offset = 0
        for child in block.backend.children:
            stop = offset + child.num_ao
            kind = "pw" if isinstance(child, AOs_plane_wave_complex_data) else "pgto"
            assert block.coefficients is not None
            groups[kind].append(block.replace(backend=child, coefficients=block.coefficients[:, offset:stop]))
            offset = stop
    pgto = explicit_supercell_pgto(orbitals.replace(blocks=tuple(groups["pgto"])))
    pw = explicit_supercell_pw(orbitals.replace(blocks=tuple(groups["pw"])))
    return MOs_data.build(
        aos_data=pgto.aos_data + pw.aos_data,
        mo_coefficients=np.concatenate([pgto.mo_coefficients, pw.mo_coefficients], axis=1),
    )


def add_pw_component(source, *, reverse=False):
    """Nonzero PW and pGTO contributions with ragged PW sizes and preserved source C."""
    from jqmc.atomic_orbital import AOs_plane_wave_complex_data

    blocks = []
    for i, block in enumerate(source.blocks):
        pgto = block.backend
        miller = np.array([(0, 0, 0), (1, 0, 0), (0, -1, 0)][: i + 2])
        pw = AOs_plane_wave_complex_data.build(
            pgto.structure_data, miller @ pgto.structure_data.recip_cell, k_point=pgto.k_point
        )
        C = np.full((block.num_orbitals, pw.num_ao), 0.17 + 0.23j) / np.sqrt(abs(np.linalg.det(pgto.structure_data.cell)))
        children = (pw, pgto) if reverse else (pgto, pw)
        coefficients = (C, block.coefficients) if reverse else (block.coefficients, C)
        blocks.append(block.replace(backend=children[0] + children[1], coefficients=np.concatenate(coefficients, axis=1)))
    result = source.replace(blocks=tuple(blocks))
    result.sanity_check()
    return result
