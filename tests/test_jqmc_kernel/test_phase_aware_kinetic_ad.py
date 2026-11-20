"""The scalar, per-electron and local-energy AD references retain the Bloch phase."""

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
from jqmc.atomic_orbital import AOs_plane_wave_complex_data, AOs_plane_wave_data
from jqmc.coulomb_potential import Coulomb_potential_data, build_ewald_data, compute_coulomb_potential
from jqmc.determinant import SD_data
from jqmc.hamiltonians import Hamiltonian_data, _compute_local_energy_auto
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.wavefunction import Wavefunction_data, _compute_kinetic_energy_all_elements_auto, _compute_kinetic_energy_auto

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_KINETIC_ZONES = ("ao_eval", "mo_eval", "det_eval", "wf_eval", "wf_kinetic")


def _plane_wave_case(twist, miller_indices, coefficients):
    cell = np.diag([5.0, 6.0, 7.0])
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    structure = Structure_data.build(
        positions=np.zeros((1, 3)),
        atomic_numbers=(1,),
        element_symbols=("H",),
        atomic_labels=("H1",),
        pbc_flag=True,
        vec_a=cell[0],
        vec_b=cell[1],
        vec_c=cell[2],
    )
    g = miller_indices @ reciprocal
    if twist is None:
        ao = AOs_plane_wave_data.build(structure, g, np.zeros(len(g)))
    else:
        ao = AOs_plane_wave_complex_data.build(structure, g, k_point=twist)
    mo = MOs_data.build(aos_data=ao, mo_coefficients=coefficients[None, :])
    geminal = SD_data.build(num_electron_up=1, num_electron_dn=0, mos_up=mo, mos_dn=mo, time_reversal=False)
    coulomb = Coulomb_potential_data(structure_data=structure)
    coulomb = coulomb.replace(ewald_data=build_ewald_data(structure, coulomb, n_electron=1))
    ham = Hamiltonian_data.build(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(geminal_data=geminal),
    )
    return ham, reciprocal


@pytest.mark.parametrize("case", ["single-wave", "superposition"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_phase_aware_kinetic_ad_matches_fourier_reference(case):
    """Use a direct Fourier sum, with nonzero imaginary kinetic energy in the two-wave case."""
    twist = (0.25, -0.125, 0.2)
    if case == "single-wave":
        indices = np.array([[0, 0, 0]])
        coefficients = np.array([1.0 + 0j])
    else:
        indices = np.array([[0, 0, 0], [1, -1, 0]])
        coefficients = np.array([np.sqrt(3.0) / 2.0, 0.5j])
    ham, reciprocal = _plane_wave_case(twist, indices, coefficients)
    wf = ham.wavefunction_data
    ru = np.array([[0.31, 0.42, 0.53]])
    rd = np.empty((0, 3))
    rotation = np.eye(3)

    q = (indices + twist) @ reciprocal
    waves = coefficients * np.exp(1j * (q @ ru[0]))
    reference_kinetic = (np.sum(q**2, axis=1) @ waves) / (2.0 * np.sum(waves))
    potential = compute_coulomb_potential(ham.coulomb_potential_data, ru, rd, rotation, wavefunction_data=wf)
    reference = np.asarray([reference_kinetic, reference_kinetic, potential + reference_kinetic])
    if case == "superposition":
        assert np.abs(reference_kinetic.imag) > 0.0

    kinetic_up, kinetic_dn = _compute_kinetic_energy_all_elements_auto(wf, ru, rd)
    kinetic_total = _compute_kinetic_energy_auto(wf, ru, rd)
    local_energy = _compute_local_energy_auto(ham, ru, rd, rotation)
    assert kinetic_up.shape == (1,) and kinetic_dn.shape == (0,)
    assert kinetic_total.shape == local_energy.shape == ()
    actual = np.asarray([kinetic_up[0], kinetic_total, local_energy])
    atol, rtol = get_tolerance_min(_KINETIC_ZONES, "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
    assert kinetic_up.dtype == kinetic_dn.dtype == kinetic_total.dtype == local_energy.dtype == np.complex128


@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_phase_aware_kinetic_ad_preserves_real_gamma():
    """A constant real Gamma wave has exact zero kinetic energy and a real local energy."""
    ham, _ = _plane_wave_case(None, np.zeros((1, 3)), np.ones(1))
    wf = ham.wavefunction_data
    ru = np.array([[0.31, 0.42, 0.53]])
    rd = np.empty((0, 3))
    rotation = np.eye(3)
    kinetic_up, kinetic_dn = _compute_kinetic_energy_all_elements_auto(wf, ru, rd)
    kinetic_total = _compute_kinetic_energy_auto(wf, ru, rd)
    actual_local = _compute_local_energy_auto(ham, ru, rd, rotation)
    reference_local = compute_coulomb_potential(ham.coulomb_potential_data, ru, rd, rotation, wavefunction_data=wf)
    assert kinetic_up.shape == (1,) and kinetic_dn.shape == (0,)
    assert kinetic_total.shape == actual_local.shape == ()
    assert kinetic_up.dtype == kinetic_dn.dtype == kinetic_total.dtype == actual_local.dtype == np.float64
    actual_kinetic = np.asarray([kinetic_up[0], kinetic_total])
    reference_kinetic = np.zeros(2)
    np.testing.assert_array_equal(actual_kinetic, reference_kinetic)
    assert np.isfinite(reference_local)
    np.testing.assert_array_equal(actual_local, reference_local)
