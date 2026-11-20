"""Tiled ensemble checkpoints retain ragged sectors and canonical supercell J3."""

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

from pathlib import Path

import h5py
import numpy as np
import pytest

from jqmc._setting import get_tol_hdf5_roundtrip
from jqmc._twist_average import TwistAverageWriter, load_twist
from jqmc._twist_checkpoint import save_shared_state, twist_checkpoint_path, write_twist_optimized_hamiltonian
from jqmc.determinant import SD_data
from jqmc.hamiltonians import _save_dataclass_to_hdf5
from jqmc.molecular_orbital import compute_MOs
from jqmc.orbital_tiling import TiledOrbitalSet_data, compute_tiled_orbitals, compute_tiled_orbitals_value_grad_lap
from jqmc.orbital_tiling_sources import read_tiled_trexio_sector, trexio_tiling_sectors
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import evaluate_ln_wavefunction
from tests.test_jqmc_kernel.test_orbital_tiling_hdf5 import _hamiltonian
from tests.test_jqmc_kernel.test_orbital_tiling_j3 import _positions, _supercell_j3

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]


def _shift_sector(ham):
    sd = ham.wavefunction_data.geminal_data
    delta = np.array([0.07, 0.11, 0.03])

    def shifted_backend(backend):
        if hasattr(backend, "children"):
            return backend.replace(children=tuple(shifted_backend(child) for child in backend.children))
        k = tuple(np.asarray(backend.k_point) + delta)
        kwargs = {"k_point": k}
        if hasattr(backend, "_image_phase_factors"):
            kwargs["_image_phase_factors"] = np.exp(2j * np.pi * (np.asarray(backend._image_indices) @ np.asarray(k)))
        return backend.replace(**kwargs)

    def shifted(orb):
        return TiledOrbitalSet_data.build(
            blocks=tuple(b.replace(backend=shifted_backend(b.backend)) for b in orb.blocks),
            tiling_matrix=orb.tiling_matrix,
            supertwist=tuple(np.asarray(orb.supertwist) + np.asarray(orb.tiling_matrix) @ delta),
        )

    up = shifted(sd.orb_data_up_spin)
    down = None
    if not sd.time_reversal:
        # Recover the positive source before SD applies the declared -k spin.
        from jqmc.orbital_tiling import conjugate_tiled_orbitals

        down = shifted(conjugate_tiled_orbitals(sd.orb_data_dn_spin, spin="up"))
    return ham.replace(
        wavefunction_data=ham.wavefunction_data.replace(geminal_data=SD_data.build_tiled(up_source=up, down_source=down))
    )


@pytest.mark.parametrize("backend", ["pw", "bspline", "pgto_cart", "pgto_sphe", "composite_cart", "composite_sphe"])
@pytest.mark.parametrize("restricted", [True, False])
def test_tiled_ensemble_checkpoint_preserves_sectors_and_shared_j3(tmp_path, backend, restricted):
    """Repeated multi-sector dumps preserve orbital fields, sharing, J3 and atomic output."""
    template = _hamiltonian(backend, restricted)
    expected = [template, _shift_sector(template)]
    source = str(tmp_path / "source.h5")
    with TwistAverageWriter(source, template, 2) as writer:
        for i, ham in enumerate(expected):
            writer.write_tiled_twist(i, [1.0, 3.0][i], ham.wavefunction_data.geminal_data)
    directory = str(tmp_path / "checkpoints")
    Path(directory).mkdir()
    j3 = _supercell_j3(template.structure_data)
    expected = [
        h.replace(
            wavefunction_data=h.wavefunction_data.replace(
                jastrow_data=h.wavefunction_data.jastrow_data.replace(jastrow_three_body_data=j3)
            )
        )
        for h in expected
    ]
    up, dn = _positions(template.structure_data)
    atol, rtol = get_tol_hdf5_roundtrip()
    for generation in range(2):
        for i, ham in enumerate(expected):
            with h5py.File(twist_checkpoint_path(directory, i), "w") as h5:
                _save_dataclass_to_hdf5(h5.create_group("hamiltonian_data"), ham)
        save_shared_state(directory, expected[0])
        output = str(tmp_path / f"dump_{generation}.h5")
        write_twist_optimized_hamiltonian(source, directory, output)
        restored = []
        for i, reference in enumerate(expected):
            actual = load_twist(output, i)
            restored.append(actual)
            np.testing.assert_array_equal(actual.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix, j3.j_matrix)
            sd = actual.wavefunction_data.geminal_data
            for field in ("orb_data_up_spin", "orb_data_dn_spin"):
                for observed, wanted in zip(
                    compute_tiled_orbitals_value_grad_lap(getattr(sd, field), up),
                    compute_tiled_orbitals_value_grad_lap(getattr(reference.wavefunction_data.geminal_data, field), up),
                    strict=True,
                ):
                    np.testing.assert_allclose(observed, wanted, atol=atol, rtol=rtol)
            for block in sd.orb_data_dn_spin.blocks:
                assert (
                    any(block.backend is b.backend and block.coefficients is b.coefficients for b in sd.orb_data_up_spin.blocks)
                    is restricted
                )
            np.testing.assert_allclose(
                evaluate_ln_wavefunction(actual.wavefunction_data, up, dn),
                evaluate_ln_wavefunction(reference.wavefunction_data, up, dn),
                atol=atol,
                rtol=rtol,
            )
        with h5py.File(output) as h5:
            shared = h5["shared/wavefunction_data/geminal_data"]
            assert shared.id == h5["twists/twist_000/geminal"].id
            for i in range(2):
                group = h5[f"twists/twist_{i:03d}"]
                assert "mo_coefficients_up" not in group and "lambda_matrix" not in group
                assert float(group["weight"][()]) == [1.0, 3.0][i]
        source, expected = output, restored
    # A checkpoint from the wrong sector must never replace an existing output.
    import shutil

    shutil.copyfile(twist_checkpoint_path(directory, 0), twist_checkpoint_path(directory, 1))
    sentinel = tmp_path / "existing-output.h5"
    sentinel.write_bytes(b"existing output")
    with pytest.raises(ValueError, match="sector"):
        write_twist_optimized_hamiltonian(source, directory, str(sentinel))
    assert sentinel.read_bytes() == b"existing output"
    assert not Path(str(sentinel) + ".tmp").exists()


@pytest.mark.parametrize("angular", ["cart", "sphe"])
def test_single_gamma_trexio_promotes_real_gto_without_changing_values(angular):
    """S=I is an identity boundary case, not nontrivial Gamma tiling evidence."""
    path = Path(__file__).parents[1] / "trexio_example_files" / f"H2_pbc_ae_{angular}.h5"
    matrix = (1, 1, 1)
    (sector,) = trexio_tiling_sectors(path, matrix)
    physical, _, up, _, _, _ = read_tiled_trexio_sector(path, matrix, sector, expand_j3_basis=False)
    _, _, original, _, _, _ = read_trexio_file(str(path), store_tuple=True)
    points = np.asarray(physical.positions) + [0.21, -0.13, 0.34]
    atol, rtol = get_tol_hdf5_roundtrip()
    np.testing.assert_allclose(compute_tiled_orbitals(up, points), compute_MOs(original, points), atol=atol, rtol=rtol)
