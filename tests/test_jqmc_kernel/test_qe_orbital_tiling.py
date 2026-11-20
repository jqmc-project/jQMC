"""QE source bands enter the common model without a global coefficient matrix."""

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

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jqmc._precision import get_complex_dtype_np, get_dtype_np, get_tolerance_min
from jqmc._setting import ATOL_TWIST_CHECKPOINT_KPOINT_MATCH, RTOL_BIT_EXACT
from jqmc.atomic_orbital import _compute_AOs_debug
from jqmc.orbital_tiling import compute_tiled_orbitals, conjugate_tiled_orbitals, tiling_k_points
from jqmc.pw_pgto_projection import CartPrimitive, ProjectionResult, project_pw_to_pw
from jqmc.qe_orbital_tiling import build_tiled_orbitals_from_projections, build_tiled_wavefunction
from jqmc.qe_wrapper import QEBands, QECell, QEReference, QEWavefunction, load_qe_reference
from tests._test_data_cache import qe_example_files_root
from tests.test_jqmc_kernel.test_orbital_tiling import _primitive

pytestmark = [pytest.mark.without_chdir]


def _synthetic_source(kind):
    primitive = _primitive()
    cell = QECell(np.asarray(primitive.cell))
    references, projections = [], []
    for k in tiling_k_points((2, 1, 1), (0.26, 0.17, -0.09)):
        reference = QEReference(
            cell=cell,
            atom_labels=("He",),
            atom_elements=("He",),
            atom_positions=np.asarray(primitive.positions),
            ecutwfc=8.0,
            fft_grid=(4, 4, 4),
            bands=QEBands(np.arange(3.0), np.array([0.0, 0.0, 1.0]), 2.0, None),
            wfc=QEWavefunction(np.array([[0, 0, 0], [1, 0, 0]]), np.ones((3, 2), complex), False, k @ cell.b),
        )
        coefficients = np.array([[0.3 + 0.2j, -0.1 + 0.4j], [1.2, 0.8], [0.7, 0.2]])
        # Projection rows are deliberately not in source-band order. The
        # occupied band is source 2, at projection row 0, not a source prefix.
        projection = ProjectionResult(
            residuals=np.zeros(3),
            band_indices=np.array([2, 0, 1]),
            occupations=np.array([1.0, 0.0, 0.0]),
            coefficients=coefficients,
            primitives=tuple(CartPrimitive(tuple(primitive.positions[0]), (0, 0, 0), a) for a in (0.8, 1.2)),
            n_gto=2,
            n_pw=2,
            n_dropped=0,
            s_cond=1.0,
            s_cond_full=1.0,
            s_eig_min=1.0,
            pw_millers=reference.wfc.millers,
            pw_coefficients=coefficients,
            spline_coefficients=np.tile(coefficients[:, :1], (1, 64)),
            spline_shape=(4, 4, 4),
            spline_order=3,
        )
        references.append(reference)
        projections.append(projection)
    return references, projections


def _explicit_supercell_pw_sd(sd):
    """Deliberately dense supercell reference, confined to this small test."""
    from jqmc.atomic_orbital import AOs_plane_wave_complex_data
    from jqmc.determinant import SD_data
    from jqmc.molecular_orbital import MOs_data

    providers = []
    for orbitals in (sd.orb_data_up_spin, sd.orb_data_dn_spin):
        total_basis = sum(b.backend.num_ao for b in orbitals.blocks)
        coefficient_matrix = np.zeros((orbitals.num_mo, total_basis), dtype=np.complex128)
        millers = []
        column = 0
        for row, block in zip(orbitals.orbital_offsets, orbitals.blocks):
            sign = -1 if block.conjugate else 1
            primitive_g = np.asarray(block.backend._g_indices).reshape(-1, 3)
            unfolded = (sign * primitive_g + block.k_point) @ np.asarray(orbitals.tiling_matrix).T - orbitals.supertwist
            integer = np.rint(unfolded)
            np.testing.assert_allclose(unfolded, integer, atol=ATOL_TWIST_CHECKPOINT_KPOINT_MATCH, rtol=RTOL_BIT_EXACT)
            millers.extend(integer)
            c = np.conj(block.coefficients) if block.conjugate else block.coefficients
            coefficient_matrix[row : row + block.num_orbitals, column : column + block.backend.num_ao] = c / np.sqrt(
                len(orbitals.blocks)
            )
            column += block.backend.num_ao
        aos = AOs_plane_wave_complex_data.build(
            orbitals.structure_data, np.asarray(millers) @ orbitals.structure_data.recip_cell, k_point=orbitals.supertwist
        )
        providers.append(MOs_data(num_mo=orbitals.num_mo, aos_data=aos, mo_coefficients=coefficient_matrix))
    return SD_data.build(
        num_electron_up=sd.num_electron_up,
        num_electron_dn=sd.num_electron_dn,
        mos_up=providers[0],
        mos_dn=providers[1],
        time_reversal=False,
    )


@pytest.mark.parametrize("kind", ["pw", "bspline", "pgto"])
@pytest.mark.light
def test_projection_adapter_preserves_source_bands_and_explicit_spin_conjugation(kind):
    """Non-Aufbau source identities survive conversion and explicit BCS conjugation."""
    references, projections = _synthetic_source(kind)
    structure, orbitals, sd = build_tiled_wavefunction(
        references[::-1],
        projections[::-1],
        kind=kind,
        tiling_matrix=(2, 1, 1),
        supertwist=(0.26, 0.17, -0.09),
        orthonormalize=False,
    )
    assert orbitals.num_mo == 2
    assert structure is orbitals.structure_data
    assert sd.time_reversal and sd.num_electron_up == sd.num_electron_dn == 2
    assert all(block.source_band_indices == (2,) for block in orbitals.blocks)
    assert len(orbitals.structure_data.positions) == 2
    positions = np.array([[0.41, 0.73, 0.61], [1.21, 0.32, 0.85]])
    expected = []
    for block in orbitals.blocks:
        values = _compute_AOs_debug(block.backend, positions)
        expected.append(values if block.coefficients is None else block.coefficients @ values)
    atol, rtol = get_tolerance_min(("ao_eval", "tiling_eval"), "medium")
    np.testing.assert_allclose(
        compute_tiled_orbitals(orbitals, positions), np.concatenate(expected) / np.sqrt(2), atol=atol, rtol=rtol
    )
    down = conjugate_tiled_orbitals(orbitals, spin="down")
    assert down.supertwist == tuple(-x for x in orbitals.supertwist)
    order = [next(i for i, b in enumerate(down.blocks) if b.backend is up.backend) for up in orbitals.blocks]
    for block in down.blocks:
        assert any(block.backend is up.backend and block.coefficients is up.coefficients for up in orbitals.blocks)
    np.testing.assert_allclose(
        np.asarray(compute_tiled_orbitals(down, positions))[order],
        np.conj(compute_tiled_orbitals(orbitals, positions)),
        atol=atol,
        rtol=rtol,
    )
    # A separate LSDA down source must retain its own C rather than borrowing
    # the up source simply because their occupations are equal.
    down_source = orbitals.replace(
        blocks=tuple(
            b.replace(coefficients=(np.array([[0.6 + 0.8j]]) if b.coefficients is None else (0.6 + 0.8j) * b.coefficients))
            for b in orbitals.blocks
        ),
        spin="down",
    )
    independent_down = conjugate_tiled_orbitals(down_source, spin="down")
    np.testing.assert_allclose(
        compute_tiled_orbitals(independent_down, positions),
        (0.6 - 0.8j) * compute_tiled_orbitals(down, positions),
        atol=atol,
        rtol=rtol,
    )
    for bad in (
        replace(projections[0], occupations=np.array([np.nan, 0.0, 0.0])),
        replace(projections[0], occupations=np.array([0.5, 0.0, 0.0])),
        replace(projections[0], band_indices=np.array([0, 0, 1])),
        replace(projections[0], occupations=np.zeros(3)),
    ):
        with pytest.raises(ValueError):
            build_tiled_orbitals_from_projections(
                references, [bad, projections[1]], kind=kind, tiling_matrix=(2, 1, 1), supertwist=orbitals.supertwist
            )
    down_projections = [
        replace(
            p,
            coefficients=(0.6 + 0.8j) * p.coefficients,
            pw_coefficients=(0.6 + 0.8j) * p.pw_coefficients,
            spline_coefficients=(0.6 + 0.8j) * p.spline_coefficients,
        )
        for p in projections
    ]
    _, _, independent_sd = build_tiled_wavefunction(
        references,
        projections,
        kind=kind,
        tiling_matrix=(2, 1, 1),
        supertwist=orbitals.supertwist,
        references_down=references,
        projections_down=down_projections,
        orthonormalize=False,
    )
    assert not independent_sd.time_reversal
    np.testing.assert_allclose(
        compute_tiled_orbitals(independent_sd.orb_data_dn_spin, positions),
        (0.6 - 0.8j) * compute_tiled_orbitals(sd.orb_data_dn_spin, positions),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.heavy
def test_real_qe_multik_pw_matches_raw_source_supercell_sum(tmp_path):
    """Eight QE blocks feed a physical ECP Hamiltonian and survive HDF5 reload."""
    from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
    from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction
    from jqmc.upf_wrapper import parse_gaussinfo

    path = qe_example_files_root() / "Li_k_twist/out/li_twist.save"
    references = [load_qe_reference(str(path), ik=i) for i in range(1, 9)]
    projections = [project_pw_to_pw(ref) for ref in references]
    structure, orbitals, sd = build_tiled_wavefunction(
        references, projections, kind="pw", tiling_matrix=(2, 2, 2), supertwist=(0.5, 0.5, 0.5)
    )
    positions = np.random.default_rng(28).uniform(size=(8, 3)) @ orbitals.structure_data.cell
    expected = []
    for block in orbitals.blocks:
        reference = next(
            ref
            for ref in references
            if np.allclose(
                ref.wfc.k_point @ np.linalg.inv(ref.cell.b),
                block.k_point,
                atol=ATOL_TWIST_CHECKPOINT_KPOINT_MATCH,
                rtol=RTOL_BIT_EXACT,
            )
        )
        q = reference.wfc.g_vectors(reference.cell) + reference.wfc.k_point
        phase = np.exp(get_complex_dtype_np("ao_eval")(1j) * (q @ positions.T).astype(get_dtype_np("ao_eval")))
        expected.append(reference.wfc.coeffs[list(block.source_band_indices)] @ phase / np.sqrt(8 * reference.cell.volume))
    atol, rtol = get_tolerance_min(("ao_eval", "tiling_eval"), "medium")
    actual = np.asarray(compute_tiled_orbitals(orbitals, positions))
    assert actual.shape == (8, 8)
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, np.concatenate(expected), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.linalg.det(actual), np.linalg.det(np.concatenate(expected)), atol=atol, rtol=rtol)
    assert all(b.coefficients.shape == (1, b.backend.num_ao) for b in orbitals.blocks)
    ham = build_hamiltonian_from_wavefunction(
        structure, orbitals, sd, j2_param=None, ecp_by_element={"Li": parse_gaussinfo(str(path / "Li.ccECP.upf"))}
    )
    assert ham.structure_data.natom == 16
    assert sd.num_electron_up == sd.num_electron_dn == 8
    checkpoint = tmp_path / "li-tiled.h5"
    ham.save_to_hdf5(checkpoint)
    restored = Hamiltonian_data.load_from_hdf5(checkpoint)
    down_positions = np.random.default_rng(29).uniform(size=(8, 3)) @ structure.cell
    energy = compute_local_energy(ham, positions, down_positions, np.eye(3))
    assert np.isfinite(energy)
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap", "coulomb", "local_energy"), "medium")
    np.testing.assert_allclose(
        compute_local_energy(restored, positions, down_positions, np.eye(3)), energy, atol=atol, rtol=rtol
    )
    explicit = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=_explicit_supercell_pw_sd(sd)))
    np.testing.assert_allclose(
        compute_local_energy(explicit, positions, down_positions, np.eye(3)), energy, atol=atol, rtol=rtol
    )
