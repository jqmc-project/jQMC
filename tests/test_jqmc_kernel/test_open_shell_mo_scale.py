"""Open-shell SD logarithmic fields and energy ignore a uniform MO normalization."""

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
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import determinant as det
from jqmc._precision import get_tolerance, get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from jqmc._twist_average import load_twist
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import Jastrow_data
from jqmc.jqmc_qe import run_qe2jqmc
from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input
from jqmc.wavefunction import (
    evaluate_ln_wavefunction,
    evaluate_ln_wavefunction_fast,
)
from tests._test_data_cache import qe_example_files_root
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]
_SOURCES = (
    "tests/qe_example_files/O-lsda_gamma/out/o.save",
    "tests/qe_example_files/O-lsda_k025/out/o_k025.save",
)
_ROOT = Path(__file__).resolve().parents[2]
_DERIVATIVE_ZONES = ("ao_eval", "ao_grad_lap", "mo_grad", "mo_lap", "det_grad_lap")


def _oxygen_case(tmp_path, fixture_path, scale):
    """Rebuild the committed four-up/two-down pGTO cell under the active policy."""
    folder = Path(tempfile.mkdtemp(prefix="oxygen-rebuild-", dir=tmp_path))
    output = folder / "oxygen.h5"
    assert fixture_path in _SOURCES
    complex_twist = "k025" in fixture_path
    if complex_twist:
        source = qe_example_files_root() / Path(fixture_path).relative_to("tests/qe_example_files")
        convert_qe_to_twist_average_input(str(source), str(output), upf_paths={"O": str(source / "O.ccECP.upf")}, kind="pgto")
        ham = load_twist(str(output), 0)
    else:
        run_qe2jqmc(
            {
                "qe_save_dir": str(qe_example_files_root() / Path(fixture_path).relative_to("tests/qe_example_files")),
                "kind": "pgto",
                "upf_paths": [],
                "ecutwfc": None,
                "alpha_min": None,
            },
            {"j1_parameter": None, "j2_parameter": None},
            str(output),
        )
        ham = Hamiltonian_data.load_from_hdf5(str(output))
    wf, gem = ham.wavefunction_data, ham.wavefunction_data.geminal_data

    def phased(orb):
        coefficients = np.asarray(orb.mo_coefficients)
        assert np.iscomplexobj(coefficients) == complex_twist
        if complex_twist:
            coefficients = coefficients * np.exp(1j * np.arange(len(coefficients)))[:, None]
        return orb.replace(mo_coefficients=coefficients * scale)

    gem = gem.replace(orb_data_up_spin=phased(gem.orb_data_up_spin), orb_data_dn_spin=phased(gem.orb_data_dn_spin))
    ham = ham.replace(wavefunction_data=wf.replace(geminal_data=gem, jastrow_data=Jastrow_data()))
    ham.sanity_check()
    assert (gem.num_electron_up, gem.num_electron_dn) == (4, 2)
    rng = np.random.default_rng(13)
    centers = np.asarray(ham.structure_data.positions)
    up, dn = tuple(
        np.ascontiguousarray(centers[np.arange(n) % len(centers)] + rng.normal(scale=0.6, size=(n, 3)))
        for n in (gem.num_electron_up, gem.num_electron_dn)
    )
    matrix = np.asarray(det.compute_geminal_all_elements(gem, up, dn))
    logabs = np.linalg.slogdet(matrix)[1]
    assert np.isfinite(logabs) and np.all(np.isfinite(np.linalg.inv(matrix)))
    if scale < 1:
        assert logabs < np.log(np.nextafter(0.0, 1.0))
    elif scale > 1:
        assert logabs > np.log(np.finfo(np.float64).max)
    return ham, up, dn


def _fields(case):
    ham, up, dn = case
    return det.compute_grads_and_laplacian_ln_Det(ham.wavefunction_data.geminal_data, up, dn)


def _assert_fidelity(actual, reference, atol, rtol):
    """Bind the prescribed block norm to an executable source site."""
    assert np.abs(actual - reference).max() <= atol + rtol * np.abs(reference).max()


def _primal_ad_fields(case):
    """Differentiate real/imaginary unscaled determinant values at real coordinates."""
    ham, up, dn = case
    gem = ham.wavefunction_data.geminal_data

    def value(a, b):
        return jnp.linalg.det(det.compute_geminal_all_elements(gem, a, b))

    base = value(up, dn)
    gradients, laplacians = [], []
    for spin in (0, 1):
        gradient = jax.jacfwd(lambda a, b: jnp.real(value(a, b)), argnums=spin)(up, dn)
        gradient = gradient + 1j * jax.jacfwd(lambda a, b: jnp.imag(value(a, b)), argnums=spin)(up, dn)
        real_hessian = jax.jacfwd(jax.jacfwd(lambda a, b: jnp.real(value(a, b)), argnums=spin), argnums=spin)(up, dn)
        imag_hessian = jax.jacfwd(jax.jacfwd(lambda a, b: jnp.imag(value(a, b)), argnums=spin), argnums=spin)(up, dn)
        logarithmic_gradient = gradient / base
        logarithmic_laplacian = jnp.einsum("ijij->i", real_hessian + 1j * imag_hessian) / base
        logarithmic_laplacian -= jnp.sum(logarithmic_gradient**2, axis=1)
        gradients.append(logarithmic_gradient)
        laplacians.append(logarithmic_laplacian)
    return gradients[0], gradients[1], laplacians[0], laplacians[1]


@pytest.mark.parametrize("fixture_path", _SOURCES, ids=["oxygen-gamma", "oxygen-twist"])
@pytest.mark.parametrize("scale", [1.0e-85, 1.0e85], ids=["tiny", "huge"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_open_shell_local_energy_is_invariant_to_uniform_mo_scale(tmp_path, fixture_path, scale):
    """A constant SD normalization cannot change a finite local energy in either mode."""
    base, up, dn = _oxygen_case(tmp_path, fixture_path, 1.0)
    changed, scaled_up, scaled_dn = _oxygen_case(tmp_path, fixture_path, scale)
    np.testing.assert_array_equal(up, scaled_up)
    np.testing.assert_array_equal(dn, scaled_dn)
    gem = changed.wavefunction_data.geminal_data
    matrix = np.asarray(det.compute_geminal_all_elements(gem, up, dn))
    assert np.isfinite(np.linalg.slogdet(matrix)[1]) and np.all(np.isfinite(np.linalg.inv(matrix)))
    actual = complex(compute_local_energy(changed, up, dn, np.eye(3)))
    reference = complex(compute_local_energy(base, up, dn, np.eye(3)))
    assert np.isfinite(actual) and np.isfinite(reference) and abs(reference) > 0
    atol, rtol = get_tolerance("det_ratio", "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("fixture_path", _SOURCES, ids=["oxygen-gamma", "oxygen-twist"])
@pytest.mark.parametrize("scale", [1.0e-85, 1.0e85], ids=["tiny", "huge"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_open_shell_logdet_has_the_global_mo_scale_shift(tmp_path, fixture_path, scale):
    """Log amplitudes remain finite when the determinant amplitude is unrepresentable."""
    ham, up, dn = _oxygen_case(tmp_path, fixture_path, scale)
    gem = ham.wavefunction_data.geminal_data
    matrix = np.asarray(det.compute_geminal_all_elements(gem, up, dn))
    expected = np.linalg.slogdet(matrix)[1]
    inverse = jnp.asarray(np.linalg.inv(matrix))
    values = (
        det.compute_ln_det_geminal_all_elements(gem, up, dn),
        det.compute_ln_det_geminal_all_elements_fast(gem, up, dn, inverse),
        evaluate_ln_wavefunction(ham.wavefunction_data, up, dn),
        evaluate_ln_wavefunction_fast(ham.wavefunction_data, up, dn, inverse),
    )
    atol, rtol = get_tolerance("det_eval", "strict")
    for actual in values:
        assert np.isfinite(actual) and np.isfinite(expected)
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("fixture_path", _SOURCES, ids=["oxygen-gamma", "oxygen-twist"])
@pytest.mark.parametrize("scale", [1.0e-85, 1.0e85], ids=["tiny", "huge"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_scaled_open_shell_derivatives_have_full_oracles_or_mixed_fidelity(tmp_path, fixture_path, scale):
    """Full differentiates a stable independent primal; mixed checks production fidelity."""
    if is_mixed_precision_enabled():
        mixed, full = mixed_vs_full(lambda: _oxygen_case(tmp_path, fixture_path, scale), _fields)
        blocks = mixed_fidelity_blocks(mixed, full)
        atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
        for actual, reference in blocks:
            _assert_fidelity(actual, reference, atol, rtol)
        corrupted = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            _assert_fidelity(corrupted, blocks[0][1], atol, rtol)
        return
    case = _oxygen_case(tmp_path, fixture_path, scale)
    ham, up, dn = case
    actual_fields = _fields(case)
    # A constant normalization cancels from D'/D and D''/D-(D'/D)**2.
    # Differentiate the independent, unscaled determinant primal at real
    # coordinates: this includes the phase derivative at genuine complex twist.
    reference_fields = _primal_ad_fields(_oxygen_case(tmp_path, fixture_path, 1.0))
    atol, rtol = get_tolerance_min(_DERIVATIVE_ZONES, "strict")
    for actual, reference in zip(actual_fields, reference_fields, strict=True):
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
    if "k025" not in fixture_path:
        # The real-state scaled logdet custom VJP must also retain correct AD
        # fields. For complex states ln|D| omits phase derivatives, so the raw
        # complex-primal oracle above is the applicable comparison.
        scaled_log_ad = det._compute_grads_and_laplacian_ln_Det_auto(ham.wavefunction_data.geminal_data, up, dn)
        for actual, reference in zip(actual_fields, scaled_log_ad, strict=True):
            assert np.all(np.isfinite(reference))
            np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
