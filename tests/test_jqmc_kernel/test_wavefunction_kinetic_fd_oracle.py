"""Calibrate the complex-Psi kinetic oracle against elementary Fourier states."""

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
import numpy as np
import pytest

from jqmc import wavefunction as wf_module
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT
from jqmc.atomic_orbital import AOs_plane_wave_complex_data
from jqmc.determinant import SD_data
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.wavefunction import (
    Wavefunction_data,
    _compute_kinetic_energy_all_elements_debug,
    _compute_kinetic_energy_debug,
    compute_kinetic_energy,
)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _fourier_state(case: str, translated: bool = False) -> tuple[Wavefunction_data, np.ndarray, np.ndarray, np.ndarray]:
    """One occupied MO per spin with an exact kinetic contribution per electron."""
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
    twist = (0.25, -0.125, 0.2)
    modes = np.array([[0, 0, 0]])
    coefficients = np.array([1.0 + 0.0j])
    if case in ("superposition", "paired-superposition"):
        modes = np.array([[0, 0, 0], [1, 0, 0]])
        coefficients = np.array([1.0, 0.3 + 0.2j])
    elif case == "real-standing-wave":
        twist = (0.0, 0.0, 0.0)
        modes = np.array([[1, 0, 0], [-1, 0, 0]])
        coefficients = np.array([0.7 + 0.0j, 0.7 + 0.0j])
    aos = AOs_plane_wave_complex_data.build(structure, modes @ reciprocal, k_point=twist)
    mos = MOs_data.build(aos_data=aos, mo_coefficients=coefficients[None, :])
    n_dn = int(case in ("paired-plane-wave", "paired-superposition"))
    # The down channel is the opposite-twist partner: conj(C) on the -k leaf.
    mos_dn = MOs_data.build(aos_data=aos.with_negated_k(), mo_coefficients=np.conj(coefficients)[None, :])
    geminal = SD_data.build(num_electron_up=1, num_electron_dn=n_dn, mos_up=mos, mos_dn=mos_dn, time_reversal=False)
    wavefunction = Wavefunction_data(geminal_data=geminal)
    wavefunction.sanity_check()
    r_up = np.array([[0.31, 0.42, 0.53]])
    r_dn = np.array([[0.71, -0.22, 0.19]]) if n_dn else np.empty((0, 3))
    if translated:
        r_up += cell[0] - cell[2]
        r_dn += 2.0 * cell[1]

    # Each spin determinant is 1x1, so Psi is a product of the occupied MOs.
    # For phi(r)=sum c_G exp(i q_G.r), -lap(phi)/(2 phi) is the sum below; the
    # down orbital is conj(phi), whose term is the conjugate.
    # No jqmc value or derivative evaluator supplies this reference.
    momenta = (modes + np.asarray(twist)) @ reciprocal
    squared_momenta = np.sum(momenta**2, axis=1)
    expected = []
    for spin, positions in (("up", r_up), ("dn", r_dn)):
        for position in positions:
            fourier_terms = coefficients * np.exp(1j * (momenta @ position))
            term = np.dot(squared_momenta, fourier_terms) / (2.0 * np.sum(fourier_terms))
            expected.append(np.conj(term) if spin == "dn" else term)
    return wavefunction, r_up, r_dn, np.asarray(expected)


@pytest.mark.parametrize("case", ["plane-wave", "paired-plane-wave", "superposition", "real-standing-wave"])
@pytest.mark.parametrize("translated", [False, True])
def test_kinetic_fd_oracle_matches_fourier_solution(case: str, translated: bool) -> None:
    """Retain phase, both spin sectors, input coordinates and decreasing Full step errors."""
    wavefunction, r_up, r_dn, expected_elements = _fourier_state(case, translated)
    scalar_expected = np.sum(expected_elements)
    original_up, original_dn = r_up.copy(), r_dn.copy()
    references = [_compute_kinetic_energy_debug(wavefunction, r_up, r_dn, diff_h=step) for step in (0.5, 0.25, 0.125)]
    assert np.all(np.isfinite(references))
    errors = np.abs(np.asarray(references) - scalar_expected)
    if not is_mixed_precision_enabled():
        # In Mixed, rounding of Psi can dominate before the smallest step.
        assert errors[2] < errors[1] < errors[0]
    scalar_reference = references[-1]
    reference, expected = scalar_reference, scalar_expected
    if case == "superposition" and not translated:
        # A separate paired state distinguishes the two spin arrays without
        # changing the scalar reference for any of the declared Fourier cases.
        paired_wf, paired_up, paired_dn, paired_expected = _fourier_state("paired-superposition")
        kinetic_up, kinetic_dn = _compute_kinetic_energy_all_elements_debug(paired_wf, paired_up, paired_dn, diff_h=0.125)
        assert kinetic_up.shape == kinetic_dn.shape == (1,)
        assert np.iscomplexobj(kinetic_up) and np.iscomplexobj(kinetic_dn)
        assert abs(paired_expected[0] - paired_expected[1]) > (
            ATOL_FD_VS_ANALYTIC_TIGHT + RTOL_FD_VS_ANALYTIC_TIGHT * max(np.abs(paired_expected))
        )
        reference = np.concatenate(([scalar_reference], kinetic_up, kinetic_dn))
        expected = np.concatenate(([scalar_expected], paired_expected))
    np.testing.assert_allclose(
        reference,
        expected,
        atol=ATOL_FD_VS_ANALYTIC_TIGHT,
        rtol=RTOL_FD_VS_ANALYTIC_TIGHT,
        err_msg="finite-difference oracle differs from Fourier kinetic energy",
    )
    reference, expected = scalar_reference, scalar_expected
    production = np.asarray(compute_kinetic_energy(wavefunction, r_up, r_dn)).item()
    np.testing.assert_allclose(
        production,
        reference,
        atol=ATOL_FD_VS_ANALYTIC_TIGHT,
        rtol=RTOL_FD_VS_ANALYTIC_TIGHT,
        err_msg="analytic kinetic energy differs from phase-preserving finite differences",
    )
    atol, rtol = get_tolerance_min(("ao_eval", "det_grad_lap", "wf_kinetic"), "strict")
    np.testing.assert_allclose(production, expected, atol=atol, rtol=rtol)
    np.testing.assert_array_equal(r_up, original_up)
    np.testing.assert_array_equal(r_dn, original_dn)
    if case == "superposition":
        assert abs(expected.imag) > ATOL_FD_VS_ANALYTIC_TIGHT + RTOL_FD_VS_ANALYTIC_TIGHT * abs(expected)
    jax.clear_caches()


def test_kinetic_fd_oracle_detects_discarded_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing phase from the actual value dependency must fail the analytic calibration."""
    wavefunction, r_up, r_dn, expected_elements = _fourier_state("plane-wave")
    expected = np.sum(expected_elements)
    original_value = wf_module.evaluate_wavefunction

    def magnitude_only(wavefunction_data: Wavefunction_data, up: np.ndarray, dn: np.ndarray) -> float:
        return float(np.abs(original_value(wavefunction_data, up, dn)))

    with monkeypatch.context() as patch:
        patch.setattr(wf_module, "evaluate_wavefunction", magnitude_only)
        magnitude_reference = _compute_kinetic_energy_debug(wavefunction, r_up, r_dn, diff_h=0.125)
    with pytest.raises(AssertionError, match="discarded phase"):
        np.testing.assert_allclose(
            magnitude_reference,
            expected,
            atol=ATOL_FD_VS_ANALYTIC_TIGHT,
            rtol=RTOL_FD_VS_ANALYTIC_TIGHT,
            err_msg="discarded phase must be detected",
        )
    jax.clear_caches()


@pytest.mark.parametrize("step", [0.0, -0.1, np.inf, np.nan])
def test_kinetic_fd_oracle_rejects_invalid_step(step: float) -> None:
    """The explicit finite-difference step must be finite and positive."""
    wavefunction, r_up, r_dn, _ = _fourier_state("plane-wave")
    with pytest.raises(ValueError, match="diff_h must be finite and positive"):
        _compute_kinetic_energy_debug(wavefunction, r_up, r_dn, diff_h=step)
