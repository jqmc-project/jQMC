"""Physical Hamiltonian persistence with primitive storage and spin aliasing."""

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

import h5py
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc._setting import get_tol_hdf5_roundtrip
from jqmc.determinant import SD_data, compute_det_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.orbital_tiling import TiledOrbitalSet_data, compute_tiled_orbitals_value_grad_lap
from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction
from tests.test_jqmc_kernel.test_orbital_tiling import _backend_set

pytestmark = [pytest.mark.without_chdir]


def _hamiltonian(backend, restricted):
    up = _backend_set(backend)
    if backend == "pw":
        # Two He atoms / four electrons; retain ragged local basis sizes.
        up = up.replace(blocks=tuple(b.replace(coefficients=b.coefficients[:1], source_band_indices=(0,)) for b in up.blocks))
    down = None
    if not restricted:
        # Equal populations do not imply shared LSDA payloads. Copy the basis
        # too, to test that equality never triggers serialization deduplication.
        down = up.replace(
            blocks=tuple(
                b.replace(
                    backend=b.backend.replace(),
                    coefficients=(np.array([[0.7 + 0.2j]]) if b.coefficients is None else (0.7 + 0.2j) * b.coefficients),
                )
                for b in up.blocks
            )
        )
    sd = SD_data.build_tiled(up_source=up, down_source=down)
    return build_hamiltonian_from_wavefunction(up.structure_data, up, sd, j2_param=None)


@pytest.mark.parametrize("backend", ["pw", "bspline", "pgto_cart", "pgto_sphe", "composite_cart", "composite_sphe"])
@pytest.mark.parametrize("restricted", [True, False])
@pytest.mark.heavy
def test_hamiltonian_roundtrip_retains_primitive_storage_spin_sharing_and_energy(backend, restricted, tmp_path):
    """Public HDF5 reload preserves VGL, SD and local energy in a physical supercell."""
    original = _hamiltonian(backend, restricted)
    path = tmp_path / "tiled.h5"
    original.save_to_hdf5(path)
    with h5py.File(path, "r") as file:
        sd_group = file["wavefunction_data/geminal_data"]
        assert "sd_representation" not in sd_group.attrs
        assert "pbc_schema_version" not in file.attrs
        for spin in ("up", "dn"):
            provider = sd_group[f"orb_data_{spin}_spin"]
            assert provider.attrs["_class_name"] == "TiledOrbitalSet_data"
            assert "schema_version" not in provider.attrs
            assert "orbital_representation" not in provider.attrs
            assert provider.attrs["normalization"] == "primitive-cell-to-supercell"
        up_groups = sd_group["orb_data_up_spin/blocks"]
        dn_groups = sd_group["orb_data_dn_spin/blocks"]
        up_addresses = {h5py.h5o.get_info(g["backend"].id).addr for g in up_groups.values()}
        dn_addresses = {h5py.h5o.get_info(g["backend"].id).addr for g in dn_groups.values()}
        assert bool(up_addresses & dn_addresses) is restricted
        if backend == "bspline":
            assert all("coefficients" not in block for block in up_groups.values())
        if backend.startswith("pgto"):
            assert all("_image_indices" not in block["backend"] for block in up_groups.values())

    restored = Hamiltonian_data.load_from_hdf5(path)
    restored.sanity_check()
    sd, saved_sd = original.wavefunction_data.geminal_data, restored.wavefunction_data.geminal_data
    assert saved_sd.time_reversal is restricted
    rng = np.random.default_rng(73)
    up = rng.uniform(size=(sd.num_electron_up, 3)) @ original.structure_data.cell
    down = rng.uniform(size=(sd.num_electron_dn, 3)) @ original.structure_data.cell
    atol, rtol = get_tol_hdf5_roundtrip("strict")
    for before, after in (
        (sd.orb_data_up_spin, saved_sd.orb_data_up_spin),
        (sd.orb_data_dn_spin, saved_sd.orb_data_dn_spin),
    ):
        assert before.tiling_matrix == after.tiling_matrix
        assert before.supertwist == after.supertwist
        for old_block, block in zip(before.blocks, after.blocks, strict=True):
            assert block.source_band_indices == old_block.source_band_indices
            assert type(block.backend) is type(old_block.backend)
            np.testing.assert_allclose(block.backend.structure_data.cell, before.primitive_structure.cell, atol=atol, rtol=rtol)
            if backend.startswith("pgto"):
                assert block.backend.num_ao_prim == old_block.backend.num_ao_prim
        for actual, expected in zip(
            compute_tiled_orbitals_value_grad_lap(after, up), compute_tiled_orbitals_value_grad_lap(before, up), strict=True
        ):
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    for block in saved_sd.orb_data_dn_spin.blocks:
        shared = any(
            block.backend is b.backend and block.coefficients is b.coefficients for b in saved_sd.orb_data_up_spin.blocks
        )
        assert shared is restricted
    np.testing.assert_allclose(
        compute_det_geminal_all_elements(saved_sd, up, down),
        compute_det_geminal_all_elements(sd, up, down),
        atol=atol,
        rtol=rtol,
    )
    energy = compute_local_energy(original, up, down, np.eye(3))
    assert np.isfinite(energy)
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap", "det_grad_lap", "coulomb", "local_energy"), "medium")
    np.testing.assert_allclose(compute_local_energy(restored, up, down, np.eye(3)), energy, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "corruption",
    ["provider_class", "missing_tiling", "normalization", "primitive_cell", "physical_cell", "spin", "conjugate", "operator"],
)
@pytest.mark.light
def test_tiled_hdf5_rejects_incompatible_or_corrupted_contract(corruption, tmp_path):
    """Providers, cells and spin sectors must agree before a restored SD can run."""
    path = tmp_path / "corrupted.h5"
    _hamiltonian("pw", True).save_to_hdf5(path)
    with h5py.File(path, "r+") as file:
        sd = file["wavefunction_data/geminal_data"]
        up = sd["orb_data_up_spin"]
        if corruption == "provider_class":
            up.attrs["_class_name"] = "MOs_data"
        elif corruption == "missing_tiling":
            del up["tiling_matrix"]
        elif corruption == "normalization":
            up.attrs["normalization"] = "already-supercell-normalized"
        elif corruption == "primitive_cell":
            up["blocks/0/backend/structure_data/vec_a"][...] *= 2
        elif corruption == "physical_cell":
            file["structure_data/vec_a"][...] *= 2
        elif corruption == "spin":
            up.attrs["spin"] = "down"
        elif corruption == "conjugate":
            up["blocks/0"].attrs["conjugate"] = 1
        else:
            del up["blocks/0/coefficients"]
    with pytest.raises(ValueError):
        Hamiltonian_data.load_from_hdf5(path)


def _scaled_structure(structure):
    """Return a copy of ``structure`` with cell and positions scaled by 1.05."""
    factor = 1.05
    return structure.replace(
        positions=factor * np.asarray(structure.positions),
        vec_a=factor * np.asarray(structure.vec_a),
        vec_b=factor * np.asarray(structure.vec_b),
        vec_c=factor * np.asarray(structure.vec_c),
    )


@pytest.mark.parametrize("leaf", ["control", "orbitals", "coulomb", "ewald", "hamiltonian"])
@pytest.mark.light
def test_tiled_hamiltonian_rejects_structure_mismatch(leaf):
    """Tiled orbitals, Coulomb, Ewald and the Hamiltonian must share one supercell.

    Each mismatched leaf is internally consistent, so only the Hamiltonian
    structure-consistency check can reject it.
    """
    hamiltonian = _hamiltonian("pw", True)
    coulomb = hamiltonian.coulomb_potential_data
    wavefunction = hamiltonian.wavefunction_data
    up = wavefunction.geminal_data.orb_data_up_spin
    if leaf == "control":
        hamiltonian.sanity_check()
        return
    if leaf == "orbitals":
        primitive = _scaled_structure(up.primitive_structure)
        up = TiledOrbitalSet_data.build(
            blocks=[block.replace(backend=block.backend.replace(structure_data=primitive)) for block in up.blocks],
            tiling_matrix=up.tiling_matrix,
            supertwist=up.supertwist,
        )
        mismatched = hamiltonian.replace(wavefunction_data=wavefunction.replace(geminal_data=SD_data.build_tiled(up_source=up)))
    elif leaf == "coulomb":
        mismatched = hamiltonian.replace(
            coulomb_potential_data=coulomb.replace(structure_data=_scaled_structure(coulomb.structure_data))
        )
    elif leaf == "ewald":
        ewald = coulomb.ewald_data.replace(structure_data=_scaled_structure(coulomb.ewald_data.structure_data))
        mismatched = hamiltonian.replace(coulomb_potential_data=coulomb.replace(ewald_data=ewald))
    else:
        mismatched = hamiltonian.replace(structure_data=_scaled_structure(hamiltonian.structure_data))
    with pytest.raises(ValueError, match="tiled orbitals, Coulomb and Hamiltonian must share the physical supercell structure"):
        mismatched.sanity_check()
