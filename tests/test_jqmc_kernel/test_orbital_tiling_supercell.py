"""Expanded supercell references that never evaluate a primitive tiled provider."""

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
import pytest

from jqmc._precision import get_tolerance_min
from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL
from jqmc.atomic_orbital import (
    AOs_bspline_complex_data,
    AOs_composite_data,
    build_AOs_cart_pgto_complex_data,
    build_AOs_sphe_pgto_complex_data,
)
from jqmc.determinant import SD_data, compute_det_geminal_all_elements
from jqmc.hamiltonians import compute_local_energy
from jqmc.molecular_orbital import MOs_data, compute_MOs_value_grad_lap
from jqmc.orbital_tiling import OrbitalKBlock_data, TiledOrbitalSet_data, compute_tiled_orbitals_value_grad_lap
from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction
from jqmc.wavefunction import compute_kinetic_energy
from tests.test_jqmc_kernel.test_orbital_tiling import _primitive

pytestmark = [pytest.mark.without_chdir, pytest.mark.heavy]


def _gto_basis(structure, k, spherical):
    """Place s, p and d functions on every physical atom, then sum supercell images."""
    atoms = structure.natom
    common = dict(
        structure_data=structure,
        k_point=tuple(k),
        nucleus_index=tuple(np.repeat(np.arange(atoms), 3)),
        num_ao=3 * atoms,
        num_ao_prim=3 * atoms,
        orbital_indices=tuple(range(3 * atoms)),
        exponents=np.tile([0.8, 1.1, 1.3], atoms),
        coefficients=np.ones(3 * atoms),
        angular_momentums=(0, 1, 2) * atoms,
        # The two image sums have different truncation shapes. Tighten their
        # construction error, rather than weakening the equality tolerance.
        tol=AO_IMAGE_TRUNCATION_ATOL**2,
    )
    if spherical:
        return build_AOs_sphe_pgto_complex_data(**common, magnetic_quantum_numbers=(0, 1, -2) * atoms)
    return build_AOs_cart_pgto_complex_data(
        **common,
        polynominal_order_x=(0, 1, 1) * atoms,
        polynominal_order_y=(0, 0, 1) * atoms,
        polynominal_order_z=(0, 0, 0) * atoms,
    )


def _source(backend, sector, independent_down=False):
    if backend.startswith("composite"):
        from tests.test_jqmc_kernel._orbital_tiling_reference import add_pw_component

        return add_pw_component(
            _source(backend.replace("composite", "pgto"), sector, independent_down), reverse=backend.endswith("cart")
        )
    primitive = _primitive()
    spline = backend.startswith("bspline")
    matrix = np.array([[2, 0 if spline else 1, 0], [0, 1, 0], [0, 0, 1]])
    twist = np.zeros(3) if sector == "gamma" else np.array([0.23, -0.17, 0.31])
    # Independently specify this two-coset mesh; do not ask the tiling matcher
    # to generate its own expected k set.
    points = [np.linalg.solve(matrix, twist + [i, 0, 0]) for i in range(2)]
    blocks = []
    rng = np.random.default_rng(71 if independent_down else 19)
    for k in points:
        if spline:
            order = int(backend[-1])
            shape = (6, 6, 6)
            table = rng.normal(size=(2, np.prod(shape))) + 1j * rng.normal(size=(2, np.prod(shape)))
            basis = AOs_bspline_complex_data.build(primitive, table, shape, spline_order=order, k_point=tuple(k))
        else:
            basis = _gto_basis(primitive, k, backend == "pgto_sphe")
        local = rng.normal(size=(1, basis.num_ao)) + 1j * rng.normal(size=(1, basis.num_ao))
        blocks.append(OrbitalKBlock_data.build(backend=basis, coefficients=local, source_band_indices=(2,)))
    return TiledOrbitalSet_data.build(blocks=blocks[::-1], tiling_matrix=matrix, supertwist=twist)


def _expanded_mos(orbitals, backend):
    """Build ordinary MOs on a fully expanded physical-supercell AO representation.

    No tiled evaluator, coset enumerator or primitive AO evaluation is used.
    B-spline grids are enlarged without fitting: each band keeps the reciprocal
    gauge S*k (equal to k_s modulo integers), hence exactly the same interpolant.
    The ordinary composite only groups fully expanded supercell spline leaves.
    pGTO uses one supercell AO set with coefficients C[k,mu]*exp(i*k*T)/sqrt(N).
    Dense matrices and replicated grids here are intentionally test-only.
    """
    if backend.startswith("composite"):
        from tests.test_jqmc_kernel._orbital_tiling_reference import explicit_supercell_composite

        return explicit_supercell_composite(orbitals)
    physical = orbitals.structure_data
    count = physical.natom // orbitals.primitive_structure.natom
    if backend.startswith("bspline"):
        repeats = tuple(np.diag(orbitals.tiling_matrix))
        children = []
        columns = sum(b.backend.num_ao for b in orbitals.blocks)
        coefficients = np.zeros((orbitals.num_mo, columns), dtype=np.complex128)
        column = 0
        for row, block in enumerate(orbitals.blocks):
            primitive = block.backend
            cube = np.asarray(primitive.spline_coefficients).reshape((primitive.num_ao, *primitive.spline_shape))
            expanded = np.tile(cube, (1, *repeats))
            local = block.coefficients
            shape = tuple(np.asarray(primitive.spline_shape) * repeats)
            # Positive-sector child; a conjugated (down-spin) block is its
            # with_negated_k() partner, exactly as every production down leaf.
            child = AOs_bspline_complex_data.build(
                physical,
                expanded.reshape(primitive.num_ao, -1),
                shape,
                spline_order=primitive.spline_order,
                k_point=tuple(np.asarray(orbitals.tiling_matrix) @ np.asarray(primitive.k_point)),
            )
            if block.conjugate:
                child, local = child.with_negated_k(), local.conj()
            children.append(child)
            coefficients[row : row + 1, column : column + primitive.num_ao] = local / np.sqrt(count)
            column += primitive.num_ao
        basis = AOs_composite_data.build(children)
        assert all(c.spline_shape == (12, 6, 6) for c in children)
    else:
        basis = _gto_basis(physical, orbitals.supertwist, backend == "pgto_sphe")
        translations = np.asarray(physical.positions) - np.asarray(orbitals.primitive_structure.positions)[0]
        coefficients = []
        for block in orbitals.blocks:
            k_cart = np.asarray(block.k_point) @ orbitals.primitive_structure.recip_cell
            phases = np.exp(1j * (translations @ k_cart))
            local = block.coefficients.conj() if block.conjugate else block.coefficients
            coefficients.append(np.concatenate([phase * local for phase in phases], axis=1) / np.sqrt(count))
        coefficients = np.concatenate(coefficients)
        assert basis.num_ao == count * 3
    return MOs_data.build(aos_data=basis, mo_coefficients=coefficients)


@pytest.mark.parametrize("backend", ["bspline3", "bspline5", "pgto_cart", "pgto_sphe", "composite_cart", "composite_sphe"])
@pytest.mark.parametrize("sector", ["gamma", "general"])
@pytest.mark.parametrize("restricted", [True, False])
def test_expanded_supercell_matches_tiled_vgl_determinant_and_energy(backend, sector, restricted):
    """Same finite orbital model and physical Hamiltonian agree at fixed electron positions."""
    source = _source(backend, sector)
    down = None if restricted else _source(backend, sector, independent_down=True)
    sd = SD_data.build_tiled(up_source=source, down_source=down)
    up_mos = _expanded_mos(sd.orb_data_up_spin, backend)
    dn_mos = _expanded_mos(sd.orb_data_dn_spin, backend)
    reference_sd = SD_data.build(num_electron_up=2, num_electron_dn=2, mos_up=up_mos, mos_dn=dn_mos, time_reversal=False)
    ham = build_hamiltonian_from_wavefunction(source.structure_data, source, sd, j2_param=None)
    reference_ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=reference_sd))
    centers = np.asarray(source.structure_data.positions)
    up = centers + np.array([[0.35, 0.41, 0.27], [0.61, -0.32, 0.28]])
    dn = centers + np.array([[-0.29, 0.37, 0.49], [0.44, 0.25, -0.36]])
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap", "mo_eval", "tiling_eval", "det_grad_lap"), "medium")
    for orbitals, mos, positions in ((sd.orb_data_up_spin, up_mos, up), (sd.orb_data_dn_spin, dn_mos, dn)):
        # Probe both interior points and points outside the physical cell.
        probes = np.concatenate(
            [positions, positions + source.structure_data.cell[0], positions - source.structure_data.cell[1]]
        )
        for tiled, explicit in zip(
            compute_tiled_orbitals_value_grad_lap(orbitals, probes), compute_MOs_value_grad_lap(mos, probes), strict=True
        ):
            assert np.isfinite(tiled).all() and np.isfinite(explicit).all()
            np.testing.assert_allclose(tiled, explicit, atol=atol, rtol=rtol)
    np.testing.assert_allclose(
        compute_det_geminal_all_elements(sd, up, dn),
        compute_det_geminal_all_elements(reference_sd, up, dn),
        atol=atol,
        rtol=rtol,
    )
    for evaluate in (
        lambda h: compute_kinetic_energy(h.wavefunction_data, up, dn),
        lambda h: compute_local_energy(h, up, dn, np.eye(3)),
    ):
        actual, expected = evaluate(ham), evaluate(reference_ham)
        assert np.isfinite(actual) and np.isfinite(expected)
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
