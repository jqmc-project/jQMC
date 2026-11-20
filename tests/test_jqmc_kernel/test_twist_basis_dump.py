"""Optimized per-twist AO bases must survive the combined VMC dump (#344)."""

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

from jqmc._setting import get_tol_hdf5_roundtrip
from jqmc._twist_average import load_twist, save_twist_average_input
from jqmc._twist_checkpoint import save_shared_state, twist_checkpoint_path, write_twist_optimized_hamiltonian
from jqmc.atomic_orbital import (
    AOs_composite_data,
    AOs_plane_wave_data,
    AOs_plane_wave_complex_data,
    _apply_image_expansion,
    _apply_image_expansion_complex,
)
from jqmc.coulomb_potential import Coulomb_potential_data
from jqmc.determinant import SD_data
from jqmc.hamiltonians import Hamiltonian_data, _save_dataclass_to_hdf5
from jqmc.jastrow_factor import Jastrow_data
from jqmc.molecular_orbital import MOs_data, compute_MOs_value_grad_lap
from jqmc.wavefunction import Wavefunction_data, evaluate_ln_wavefunction
from tests.conftest import make_pbc_structure
from tests.test_jqmc_kernel.test_atomic_orbital import _make_aos_data

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


def _hamiltonian(ao_kind, k_point, time_reversal, change=0):
    structure = make_pbc_structure(np.eye(3) * 6.0)

    def orbital(spin):
        base = _make_aos_data(structure, ao_kind.split("+")[0], l_max=0, uniform=False)
        # Change both parameters, differently for each twist and (UHF) spin.
        delta = change * (1 + spin * (not time_reversal))
        base = base.replace(
            radial=base.radial.replace(
                exponents=np.asarray(base.radial.exponents) * (1 + 0.2 * delta),
                coefficients=np.asarray(base.radial.coefficients) * np.array([1 + 0.3 * delta, 1 - 0.1 * delta, 1.0]),
            ),
        )
        # Both spin treatments pair the channels at opposite twists.
        kp = tuple(-x for x in k_point) if spin and k_point is not None else k_point
        aos = (
            _apply_image_expansion(base, structure)
            if kp is None
            else _apply_image_expansion_complex(base, structure, k_point=kp)
        )
        coeff = np.array([[1.0]]) if kp is None else np.array([[1.0 + (0.2j if not spin or not time_reversal else -0.2j)]])
        if ao_kind.endswith("+pw"):
            # Distinct PW ranks exercise complete AO overrides, including
            # cases whose optimized MO width differs from the shared template.
            indices = np.array([[j, 0, 0] for j in range(change + 1)])
            vectors = indices * (2 * np.pi / 6.0)
            if kp is None:
                pw = AOs_plane_wave_data.build(structure, vectors, np.zeros(len(indices)))
            else:
                pw = AOs_plane_wave_complex_data.build(structure, vectors, k_point=k_point)
                if spin:
                    pw = pw.with_negated_k()
            aos = AOs_composite_data.build((aos, pw))
            coeff = np.hstack((coeff, np.full((1, len(indices)), 0.1, dtype=coeff.dtype)))
        return MOs_data(num_mo=1, aos_data=aos, mo_coefficients=coeff)

    geminal = SD_data.build(
        num_electron_up=1,
        num_electron_dn=1,
        mos_up=orbital(0),
        mos_dn=orbital(1),
        time_reversal=time_reversal,
    )
    return Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=Coulomb_potential_data.build(structure_data=structure),
        wavefunction_data=Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=geminal),
    )


@pytest.mark.parametrize("ao_kind", ["cart", "sphe", "cart+pw"])
@pytest.mark.parametrize("non_gamma", [False, True])
@pytest.mark.parametrize("time_reversal", [False, True])
def test_optimized_basis_dump_preserves_orbitals(tmp_path, ao_kind, non_gamma, time_reversal):
    """Physical round-trip oracle with distinct twist/spin bases, repeated twice.

    The source has unchanged bases; checkpoints have changed primitive
    exponents AND contraction coefficients. Comparing MO values, derivatives
    and the wavefunction catches lost parameters even when MO matrices match.
    """
    k_points = [(0.13, 0.0, 0.0), (0.0, 0.21, 0.0)] if non_gamma else [None, None]
    template = _hamiltonian(ao_kind, k_points[0], time_reversal)
    geminal = template.wavefunction_data.geminal_data
    source = str(tmp_path / "source.h5")
    save_twist_average_input(
        source,
        template,
        k_points,
        [1.0, 3.0],
        [geminal.orb_data_up_spin.mo_coefficients] * 2,
        [geminal.orb_data_dn_spin.mo_coefficients] * 2,
        composite_child_per_twist=(
            [("pw_complex", np.array([[0, 0, 0]]))] * 2 if ao_kind.endswith("+pw") and non_gamma else None
        ),
    )
    checkpoint_dir = str(tmp_path / "checkpoints")
    (tmp_path / "checkpoints").mkdir()
    expected = [_hamiltonian(ao_kind, kp, time_reversal, i + 1) for i, kp in enumerate(k_points)]
    for i, ham in enumerate(expected):
        with h5py.File(twist_checkpoint_path(checkpoint_dir, i), "w") as h5:
            _save_dataclass_to_hdf5(h5.create_group("hamiltonian_data"), ham)
    save_shared_state(checkpoint_dir, expected[0])
    positions = np.array([[0.4, 0.3, 0.2], [1.1, 0.7, 0.5], [5.7, 0.2, 0.3]])
    atol, rtol = get_tol_hdf5_roundtrip()
    for generation in range(2):
        output = str(tmp_path / f"optimized_{generation}.h5")
        write_twist_optimized_hamiltonian(source, checkpoint_dir, output)
        for i, reference in enumerate(expected):
            restored = load_twist(output, i)
            for field in ("orb_data_up_spin", "orb_data_dn_spin"):
                before = getattr(reference.wavefunction_data.geminal_data, field)
                after = getattr(restored.wavefunction_data.geminal_data, field)
                for actual, wanted in zip(
                    compute_MOs_value_grad_lap(after, positions), compute_MOs_value_grad_lap(before, positions), strict=True
                ):
                    np.testing.assert_allclose(actual, wanted, atol=atol, rtol=rtol)
                # Re-saving must not multiply the periodic image expansion.
                before_gto = before.aos_data.children[0] if ao_kind.endswith("+pw") else before.aos_data
                after_gto = after.aos_data.children[0] if ao_kind.endswith("+pw") else after.aos_data
                assert after_gto.num_ao_prim == before_gto.num_ao_prim
                np.testing.assert_array_equal(after_gto.exponents, before_gto.exponents)
                np.testing.assert_array_equal(after_gto.coefficients, before_gto.coefficients)
            np.testing.assert_allclose(
                evaluate_ln_wavefunction(restored.wavefunction_data, positions[:1], positions[1:2]),
                evaluate_ln_wavefunction(reference.wavefunction_data, positions[:1], positions[1:2]),
                atol=atol,
                rtol=rtol,
            )
            with h5py.File(twist_checkpoint_path(checkpoint_dir, i), "w") as h5:
                _save_dataclass_to_hdf5(h5.create_group("hamiltonian_data"), restored)
        source = output


@pytest.mark.parametrize("spin", ["up", "dn"])
def test_incomplete_twist_basis_override_is_rejected(tmp_path, spin):
    """Never combine one optimized spin basis with the other input spin basis."""
    ham = _hamiltonian("cart", None, False)
    geminal = ham.wavefunction_data.geminal_data
    path = str(tmp_path / "incomplete.h5")
    save_twist_average_input(
        path,
        ham,
        [None],
        [1.0],
        [geminal.orb_data_up_spin.mo_coefficients],
        [geminal.orb_data_dn_spin.mo_coefficients],
    )
    with h5py.File(path, "r+") as h5:
        _save_dataclass_to_hdf5(
            h5["twists/twist_000"].create_group(f"aos_data_{spin}"),
            geminal.orb_data_up_spin.aos_data,
        )
    with pytest.raises(ValueError, match="must supply both aos_data_up and aos_data_dn"):
        load_twist(path, 0)
