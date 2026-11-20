"""Independent physical-atom and strain continuation against explicit supercells."""

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL, ATOL_PYTHON_FP64_LOOSE, RTOL_BIT_EXACT
from jqmc.atomic_orbital import _apply_strain_to_aos, build_AOs_cart_pgto_complex_data, build_AOs_sphe_pgto_complex_data
from jqmc.molecular_orbital import MOs_data, compute_MOs, compute_MOs_value_grad_lap
from jqmc.orbital_tiling import (
    OrbitalKBlock_data,
    TiledOrbitalSet_data,
    compute_tiled_orbitals,
    compute_tiled_orbitals_value_grad_lap,
    conjugate_tiled_orbitals,
)
from jqmc.structure import Structure_data, _strain_structure_data
from tests.test_jqmc_kernel.test_orbital_tiling_supercell import _source, _expanded_mos

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]


@pytest.mark.parametrize("backend", ["pgto_cart", "pgto_sphe", "composite_cart", "composite_sphe"])
@pytest.mark.parametrize("conjugate", [False, True])
def test_independent_physical_atoms_and_strain_match_supercell(backend, conjugate):
    """Independently displaced atoms and all nine affine derivatives equal ordinary SC AOs."""
    source = _source(backend, "general")
    if conjugate:
        source = conjugate_tiled_orbitals(source, spin="down")
    reference = _expanded_mos(source, backend)
    positions = jnp.asarray(source.structure_data.positions) + jnp.array([[0.03, -0.02, 0.01], [-0.04, 0.01, 0.02]])
    points = jnp.array([[0.6, 0.7, 0.8], [4.2, -1.3, 1.1], [8.1, 7.2, -0.3]])

    def tiled(p, eps):
        structure = _strain_structure_data(source.structure_data.replace(positions=p), jnp.eye(3) + eps)
        return compute_tiled_orbitals(source.replace(structure_data=structure), points @ (jnp.eye(3) + eps).T)

    def with_positions(aos, p):
        if hasattr(aos, "children"):
            return aos.replace(children=tuple(with_positions(child, p) for child in aos.children))
        return aos.replace(structure_data=aos.structure_data.replace(positions=p))

    def explicit(p, eps):
        aos = with_positions(reference.aos_data, p)
        aos = _apply_strain_to_aos(aos, jnp.eye(3) + eps)
        return compute_MOs(reference.replace(aos_data=aos), points @ (jnp.eye(3) + eps).T)

    eps = jnp.array([[0.012, -0.008, 0.003], [0.004, -0.007, 0.002], [-0.006, 0.009, 0.005]])
    atol, rtol = get_tolerance_min(("tiling_eval", "tiling_grad", "tiling_lap", "ao_eval", "ao_grad_lap"), "medium")
    np.testing.assert_allclose(tiled(positions, eps), explicit(positions, eps), atol=atol, rtol=rtol)
    for arg in (0, 1):
        np.testing.assert_allclose(
            jax.jacfwd(tiled, argnums=arg)(positions, eps),
            jax.jacfwd(explicit, argnums=arg)(positions, eps),
            atol=atol,
            rtol=rtol,
        )
    moved = source.replace(structure_data=source.structure_data.replace(positions=positions))
    plain = reference.replace(aos_data=with_positions(reference.aos_data, positions))
    for actual, wanted in zip(
        compute_tiled_orbitals_value_grad_lap(moved, points), compute_MOs_value_grad_lap(plain, points), strict=True
    ):
        np.testing.assert_allclose(actual, wanted, atol=atol, rtol=rtol)
    phase = np.exp(2j * np.pi * source.supertwist[0])
    np.testing.assert_allclose(
        compute_tiled_orbitals(moved, points + moved.structure_data._cell_jnp[0]),
        phase * compute_tiled_orbitals(moved, points),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("backend", ["pw", "bspline3", "pgto_cart", "pgto_sphe", "composite_cart", "composite_sphe"])
@pytest.mark.parametrize("conjugate", [False, True])
def test_physical_vgl_analytic_cotangents_equal_autograd(backend, conjugate):
    """Analytic VGL bars match AD for every physical atom, strain and electron component."""
    from jqmc.derivatives.orbital_tiling import tiled_field_vjp
    from tests.test_jqmc_kernel.test_orbital_tiling_hdf5 import _hamiltonian

    source = (
        _hamiltonian("pw", True).wavefunction_data.geminal_data.orb_data_up_spin
        if backend == "pw"
        else _source(backend, "general")
    )
    if conjugate:
        source = conjugate_tiled_orbitals(source, spin="down")
    structure = source.structure_data.replace(
        positions=source.structure_data._positions_cart_jnp + jnp.array([[0.03, -0.02, 0.01], [-0.04, 0.01, 0.02]])
    )
    source = source.replace(structure_data=structure)
    points = jnp.array([[0.6, 0.7, 0.8], [4.2, -1.3, 1.1], [8.1, 7.2, -0.3]])
    rng = np.random.default_rng(321)
    bars = rng.normal(size=(5, source.num_mo, len(points))) + 1j * rng.normal(size=(5, source.num_mo, len(points)))

    def objective(orb, r):
        return jnp.real(jnp.sum(jnp.asarray(bars) * jnp.stack(compute_tiled_orbitals_value_grad_lap(orb, r))))

    ad = {
        "R": jax.grad(lambda p: objective(source.replace(structure_data=structure.replace(positions=p)), points))(
            structure._positions_cart_jnp
        ),
        "eps": jax.grad(
            lambda e: objective(
                source.replace(structure_data=_strain_structure_data(structure, jnp.eye(3) + e)), points @ (jnp.eye(3) + e).T
            )
        )(jnp.zeros((3, 3))),
        "r": jax.grad(lambda r: objective(source, r))(points),
    }
    atol, rtol = get_tolerance_min(("tiling_eval", "tiling_grad", "tiling_lap", "ao_eval", "ao_grad_lap"), "medium")
    for coordinate, expected in ad.items():
        actual = jnp.real(tiled_field_vjp(source, points, bars[0], bars[1:4], bars[4], wrt=coordinate))
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


_MULTI_ATOM_CELL = np.array([[3.1, 0.0, 0.0], [0.4, 3.5, 0.0], [-0.3, 0.5, 4.0]])
_MULTI_ATOM_SPECIES = {
    2: ("He", np.array([0.3, 0.4, 0.5]), (0.8, 1.1, 1.3)),
    4: ("Be", np.array([1.9, 1.2, 2.1]), (0.9, 1.4, 1.0)),
}


def _species_gto_basis(structure, k, spherical):
    """s, p and d pGTOs on every atom with species-dependent exponents."""
    atoms = structure.natom
    common = dict(
        structure_data=structure,
        k_point=tuple(k),
        nucleus_index=tuple(np.repeat(np.arange(atoms), 3)),
        num_ao=3 * atoms,
        num_ao_prim=3 * atoms,
        orbital_indices=tuple(range(3 * atoms)),
        exponents=np.concatenate([_MULTI_ATOM_SPECIES[z][2] for z in structure.atomic_numbers]),
        coefficients=np.ones(3 * atoms),
        angular_momentums=(0, 1, 2) * atoms,
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


def _multi_atom_source(matrix, spherical):
    """Two-species skew primitive; the k mesh is enumerated here, not by the tiling code."""
    numbers = tuple(_MULTI_ATOM_SPECIES)
    primitive = Structure_data.build(
        positions=np.array([_MULTI_ATOM_SPECIES[z][1] for z in numbers]),
        pbc_flag=True,
        vec_a=_MULTI_ATOM_CELL[0],
        vec_b=_MULTI_ATOM_CELL[1],
        vec_c=_MULTI_ATOM_CELL[2],
        atomic_numbers=numbers,
        element_symbols=tuple(_MULTI_ATOM_SPECIES[z][0] for z in numbers),
        atomic_labels=tuple(f"{_MULTI_ATOM_SPECIES[z][0]}1" for z in numbers),
    )
    matrix = np.asarray(matrix)
    count = round(abs(np.linalg.det(matrix)))
    twist = np.array([0.23, -0.17, 0.31])
    # S k = twist + m; k ~ k' (mod 1) iff adj(S) (m - m') = 0 (mod |det S|), in integers.
    adjugate = np.rint(np.linalg.det(matrix) * np.linalg.inv(matrix)).astype(int)
    points = {}
    for shift in np.ndindex(5, 5, 5):
        m = np.asarray(shift) - 2
        points.setdefault(tuple(adjugate @ m % count), np.linalg.solve(matrix, twist + m))
    assert len(points) == count
    rng = np.random.default_rng(670)
    blocks = []
    for k in points.values():
        basis = _species_gto_basis(primitive, k, spherical)
        local = rng.normal(size=(1, basis.num_ao)) + 1j * rng.normal(size=(1, basis.num_ao))
        blocks.append(OrbitalKBlock_data.build(backend=basis, coefficients=local, source_band_indices=(2,)))
    rng.shuffle(blocks)
    return TiledOrbitalSet_data.build(blocks=blocks, tiling_matrix=matrix, supertwist=twist)


def _hand_built_multi_atom_mos(orbitals, spherical):
    """Ordinary SC MOs with C[k, mu(atom j)] exp(2 pi i k.n_j)/sqrt(N).

    Each physical atom's species and integer primitive translation n_j are
    recovered from its own position, independent of the physical atom order.
    """
    physical = orbitals.structure_data
    count = len(orbitals.blocks)
    assert physical.natom == count * len(_MULTI_ATOM_SPECIES)
    order = list(_MULTI_ATOM_SPECIES)
    atoms, translations = [], []
    for z, position in zip(physical.atomic_numbers, np.asarray(physical.positions), strict=True):
        n = (position - _MULTI_ATOM_SPECIES[z][1]) @ np.linalg.inv(_MULTI_ATOM_CELL)
        np.testing.assert_allclose(n, np.rint(n), rtol=RTOL_BIT_EXACT, atol=ATOL_PYTHON_FP64_LOOSE)
        atoms.append(order.index(z))
        translations.append(np.rint(n).astype(int))
    matrix = np.asarray(orbitals.tiling_matrix, dtype=float)
    adjugate = np.rint(np.linalg.det(matrix) * np.linalg.inv(matrix)).astype(int)
    # Every (species, coset of Z^3 / Z^3 S) occurs exactly once.
    cosets = {(a, tuple(n @ adjugate % count)) for a, n in zip(atoms, translations, strict=True)}
    assert len(cosets) == physical.natom
    rows = []
    for block in orbitals.blocks:
        local = block.coefficients.conj() if block.conjugate else block.coefficients
        k = np.asarray(block.k_point)
        rows.append(
            np.concatenate(
                [np.exp(2j * np.pi * (n @ k)) * local[:, 3 * a : 3 * a + 3] for a, n in zip(atoms, translations, strict=True)],
                axis=1,
            )
            / np.sqrt(count)
        )
    basis = _species_gto_basis(physical, orbitals.supertwist, spherical)
    return MOs_data.build(aos_data=basis, mo_coefficients=np.concatenate(rows))


_MULTI_ATOM_TILINGS = {
    "det4": ((2, 1, 1), (0, 1, 1), (1, 0, 2)),
    "det_minus3": ((1, 2, 0), (0, 1, 0), (1, 0, -3)),
}


@pytest.mark.parametrize("tiling", list(_MULTI_ATOM_TILINGS))
@pytest.mark.parametrize("backend", ["pgto_cart", "pgto_sphe"])
@pytest.mark.parametrize("conjugate", [False, True])
def test_multi_atom_physical_atoms_match_hand_built_supercell(tiling, backend, conjugate):
    """Two-atom primitive, |det S| >= 3 and det S < 0: per-atom R-derivatives equal a hand-built SC.

    A one-atom N=2 tiling cannot distinguish the translation-major atom index
    or the sign of a negative-determinant quotient; this case does.
    """
    spherical = backend == "pgto_sphe"
    source = _multi_atom_source(_MULTI_ATOM_TILINGS[tiling], spherical)
    if conjugate:
        source = conjugate_tiled_orbitals(source, spin="down")
    reference = _hand_built_multi_atom_mos(source, spherical)
    natom = source.structure_data.natom
    displacement = jnp.asarray(np.random.default_rng(6700).uniform(-0.05, 0.05, size=(natom, 3)))
    positions = jnp.asarray(source.structure_data.positions) + displacement
    cell = np.asarray(source.structure_data.cell)
    points = jnp.asarray(np.array([[0.31, 0.27, 0.62], [0.74, 0.55, 0.18], [1.6, -0.7, 0.4], [-1.3, 2.2, -0.8]]) @ cell)

    def tiled(p):
        return compute_tiled_orbitals_value_grad_lap(
            source.replace(structure_data=source.structure_data.replace(positions=p)), points
        )

    def explicit(p):
        aos = reference.aos_data
        return compute_MOs_value_grad_lap(
            reference.replace(aos_data=aos.replace(structure_data=aos.structure_data.replace(positions=p))), points
        )

    atol, rtol = get_tolerance_min(("tiling_eval", "tiling_grad", "tiling_lap", "ao_eval", "ao_grad_lap"), "medium")
    for actual, wanted in zip(tiled(positions), explicit(positions), strict=True):
        np.testing.assert_allclose(actual, wanted, atol=atol, rtol=rtol)
    for actual, wanted in zip(jax.jacfwd(tiled)(positions), jax.jacfwd(explicit)(positions), strict=True):
        np.testing.assert_allclose(actual, wanted, atol=atol, rtol=rtol)
