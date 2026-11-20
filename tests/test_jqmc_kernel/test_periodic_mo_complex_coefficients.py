"""Complex-coefficient periodic MO rows for the grad / Laplacian / fused nodes.

The periodic MO grad / Laplacian / fused nodes in ``test_molecular_orbital.py``
(``TestMOsPeriodic``, ``TestMOsFusedAPI``) use real MO coefficients only, so a
defect in the complex ``mo_coefficients @ ao_*`` contraction (for example a
conjugated or real-cast coefficient matrix) is invisible to them: the conjugate
of a real matrix is the matrix itself.  This module adds the complex rows.

Fixtures are the committed twist files ``lithium_k025.h5`` (spherical Bloch
pGTO, ECP) and ``H2_pbc_ae_k025_cart.h5`` (Cartesian Bloch pGTO, all electron).
The loaded complex MO matrix is re-mixed with fixed complex weights so that every
MO row carries a genuinely complex coefficient vector.  Electrons straddle the
home cell, so the pGTO wrap-restore phase is non-trivial.

Full retains the strict analytical/AD mathematical oracles below. Mixed instead
compares each analytical derivative block with the same production route rebuilt
in full using the central orbital-derivative budget. Same-precision fused/API
parity and complex non-vacuity controls are retained:

* analytic ``compute_MOs_grad`` equals forward-mode AD of the complex ``compute_MOs``
  value (Re and Im differentiated separately);
* analytic ``compute_MOs_laplacian`` equals the trace of the forward-mode AD
  Hessian of ``compute_MOs``;
* fused ``compute_MOs_value_grad_lap`` equals the standalone value, gradient and
  Laplacian APIs, and its gradient / Laplacian equal the autodiff oracle.

Non-vacuity: the imaginary part of every compared quantity exceeds the absolute
tolerance by a wide margin, and evaluating the autodiff oracle with conjugated
coefficients misses the comparison envelope, so a conjugated contraction cannot
pass.  The in-process negative control conjugates the analytic AO derivative
inputs of the MO layer and requires every comparison to fail.
"""

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

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jqmc.molecular_orbital as mo_module
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from tests._precision_fidelity import mixed_vs_full
from tests.test_jqmc_kernel._orbital_mixed_fidelity import assert_mixed_orbital_derivative_fidelity
from jqmc.atomic_orbital import is_complex_aos
from jqmc.molecular_orbital import (
    MOs_data,
    compute_MOs,
    compute_MOs_grad,
    compute_MOs_laplacian,
    compute_MOs_value_grad_lap,
)
from jqmc.trexio_wrapper import read_trexio_file

pytestmark = [pytest.mark.with_chdir]

_FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")
#: Committed complex twist fixtures (Bloch pGTO leaf + complex128 MO coefficients).
_COMPLEX_MO_FIXTURES = ("lithium_k025.h5", "H2_pbc_ae_k025_cart.h5")
#: Fixed complex re-mixing weights of the loaded MO rows (row 0 and two rolled copies).
_MIX_WEIGHTS = (1.0 + 0.25j, -0.4 + 0.8j, 0.6 - 0.35j)
#: Fractional electron positions in the home cell and its neighbours.
_ELECTRON_FRAC = np.array(
    [
        [0.13, 0.27, 0.41],
        [1.07, -0.19, 0.33],
        [-0.22, 0.58, 1.14],
        [0.61, 0.44, -0.83],
        [0.35, 1.72, 0.18],
    ],
    dtype=np.float64,
)
#: A compared quantity (Re and Im) must exceed its absolute tolerance by this factor.
_NON_VACUITY_FACTOR = 1.0e3

_GRAD_ZONES = ("ao_eval", "ao_grad_lap", "mo_grad")
_LAP_ZONES = ("ao_eval", "ao_grad_lap", "mo_lap")
_VALUE_ZONES = ("ao_eval", "mo_eval")


def _complex_mo_setup(trexio_file: str):
    """Return ``(mos_data, r_carts)`` with genuinely complex MO rows and straddling electrons."""
    _, _, mos_loaded, _, _, _ = read_trexio_file(
        trexio_file=os.path.join(_FIXTURE_DIR, trexio_file),
        store_tuple=True,
    )
    assert is_complex_aos(mos_loaded.aos_data), f"{trexio_file}: not a complex Bloch leaf"
    base = np.asarray(mos_loaded.mo_coefficients[0], dtype=np.complex128)
    coefficients = np.stack([np.roll(base, shift) * weight for shift, weight in zip((0, 1, 3), _MIX_WEIGHTS, strict=True)])
    assert np.max(np.abs(coefficients.imag)) > 0.0 and np.max(np.abs(coefficients.real)) > 0.0
    mos_data = MOs_data.build(aos_data=mos_loaded.aos_data, mo_coefficients=coefficients)
    mos_data.sanity_check()
    cell = np.asarray(mos_data.aos_data.structure_data.cell, dtype=np.float64)
    r_carts = _ELECTRON_FRAC @ cell
    n0 = np.floor(r_carts @ np.linalg.inv(cell))
    outside = np.any(n0 != 0, axis=1)
    assert outside.any() and (~outside).any(), "electrons must straddle the home cell"
    return mos_data, r_carts


def _assert_complex_non_vacuous(label: str, arr, atol: float) -> None:
    arr = np.asarray(arr)
    assert np.iscomplexobj(arr), f"{label}: complex MO row returned a real array"
    assert np.all(np.isfinite(arr)), f"{label}: non-finite"
    floor = _NON_VACUITY_FACTOR * atol
    assert np.max(np.abs(arr.real)) > floor, f"{label}: Re part vacuous"
    assert np.max(np.abs(arr.imag)) > floor, f"{label}: Im part vacuous"


def _checked_arrays(label: str, actual, reference):
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    assert actual.shape == reference.shape, label
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)), f"{label}: non-finite"
    return actual, reference


def _compare_grad(label: str, actual, reference) -> None:
    """Element-wise comparison of complex MO gradients (a comparison procedure; see ``_deviation``)."""
    atol, rtol = get_tolerance_min(_GRAD_ZONES, "strict")
    actual, reference = _checked_arrays(label, actual, reference)
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, err_msg=label)


def _compare_lap(label: str, actual, reference) -> None:
    """Element-wise comparison of complex MO Laplacians (a comparison procedure; see ``_deviation``)."""
    atol, rtol = get_tolerance_min(_LAP_ZONES, "strict")
    actual, reference = _checked_arrays(label, actual, reference)
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, err_msg=label)


def _compare_value(label: str, actual, reference) -> None:
    """Element-wise comparison of complex MO values (a comparison procedure; see ``_deviation``)."""
    atol, rtol = get_tolerance_min(_VALUE_ZONES, "strict")
    actual, reference = _checked_arrays(label, actual, reference)
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, err_msg=label)


def _deviation(actual, reference) -> float:
    """Max abs deviation, reported after the comparison of the same operands passed."""
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))


def _conjugated_oracle_misses(label: str, reference, conj_reference, zones) -> None:
    """The oracle evaluated with conjugated coefficients lies outside the comparison envelope."""
    atol, rtol = get_tolerance_min(zones, "strict")
    reference = np.asarray(reference)
    excess = np.abs(np.asarray(conj_reference) - reference) / (atol + rtol * np.abs(reference))
    assert np.max(excess) > _NON_VACUITY_FACTOR, f"{label}: conjugated coefficients indistinguishable"


def _ad_mo_grad_lap(mos_data: MOs_data, r_carts):
    """Forward-mode AD of the complex ``compute_MOs`` value: ``((gx, gy, gz), lap)``.

    ``jax.jacfwd`` on the real flattened coordinates differentiates the real and
    imaginary parts separately (the private ``_compute_MOs_*_autodiff`` oracles
    use ``jacrev`` and reject complex outputs).  The Laplacian of electron ``i``
    is the trace of its own ``3 x 3`` Hessian block.
    """
    n_el = r_carts.shape[0]

    def f(x_flat):
        return compute_MOs(mos_data, x_flat.reshape(n_el, 3))

    x0 = jnp.asarray(r_carts, dtype=jnp.float64).reshape(-1)
    jac = jax.jacfwd(f)(x0)  # (num_mo, N_e, 3 N_e)
    hess = jax.jacfwd(jax.jacfwd(f))(x0)  # (num_mo, N_e, 3 N_e, 3 N_e)
    idx = np.arange(n_el)
    grads = tuple(np.asarray(jac[:, idx, 3 * idx + d]) for d in range(3))
    lap = np.asarray(sum(hess[:, idx, 3 * idx + d, 3 * idx + d] for d in range(3)))
    return grads, lap


def _conjugated(mos_data: MOs_data) -> MOs_data:
    return MOs_data.build(aos_data=mos_data.aos_data, mo_coefficients=np.conj(np.asarray(mos_data.mo_coefficients)))


def _analytical_mo_outputs(case):
    mos_data, r_carts = case
    conjugated = _conjugated(mos_data)
    return {
        "value": compute_MOs(mos_data, r_carts),
        "grads": compute_MOs_grad(mos_data, r_carts),
        "lap": compute_MOs_laplacian(mos_data, r_carts),
        "fused": compute_MOs_value_grad_lap(mos_data, r_carts),
        "conjugated_grads": compute_MOs_grad(conjugated, r_carts),
        "conjugated_lap": compute_MOs_laplacian(conjugated, r_carts),
    }


def _mo_fidelity_pair(trexio_file, full_reference):
    if full_reference is None:
        return mixed_vs_full(lambda: _complex_mo_setup(trexio_file), _analytical_mo_outputs)
    # A negative control supplies the clean full host result captured before
    # installing the production fault; only the mixed route is rebuilt here.
    mixed = jax.tree_util.tree_map(np.asarray, _analytical_mo_outputs(_complex_mo_setup(trexio_file)))
    return mixed, full_reference


def _check_grad_fidelity(trexio_file, mixed, full):
    atol, _ = get_tolerance_min(_GRAD_ZONES, "strict")
    worst = 0.0
    for axis, actual, reference, conjugated in zip("xyz", mixed["grads"], full["grads"], full["conjugated_grads"], strict=True):
        label = f"{trexio_file} complex MO grad_{axis} mixed vs full"
        _assert_complex_non_vacuous(label, reference, atol)
        _conjugated_oracle_misses(label, reference, conjugated, _GRAD_ZONES)
        worst = max(worst, assert_mixed_orbital_derivative_fidelity(actual, reference, label))
    return worst


def _check_laplacian_fidelity(trexio_file, mixed, full):
    atol, _ = get_tolerance_min(_LAP_ZONES, "strict")
    label = f"{trexio_file} complex MO laplacian mixed vs full"
    _assert_complex_non_vacuous(label, full["lap"], atol)
    _conjugated_oracle_misses(label, full["lap"], full["conjugated_lap"], _LAP_ZONES)
    return assert_mixed_orbital_derivative_fidelity(mixed["lap"], full["lap"], label)


def _check_fused_fidelity(trexio_file, mixed, full):
    # All API-parity operands come from this same mixed evaluation.
    val, gx, gy, gz, lap = mixed["fused"]
    atol_value, _ = get_tolerance_min(_VALUE_ZONES, "strict")
    _assert_complex_non_vacuous(f"{trexio_file} complex MO value", mixed["value"], atol_value)
    _compare_value(f"{trexio_file} fused value vs compute_MOs", val, mixed["value"])
    worst = _deviation(val, mixed["value"])
    for axis, actual, standalone, reference in zip("xyz", (gx, gy, gz), mixed["grads"], full["fused"][1:4], strict=True):
        _compare_grad(f"{trexio_file} fused grad_{axis} vs compute_MOs_grad", actual, standalone)
        worst = max(
            worst, assert_mixed_orbital_derivative_fidelity(actual, reference, f"{trexio_file} fused grad_{axis} mixed vs full")
        )
    _compare_lap(f"{trexio_file} fused laplacian vs compute_MOs_laplacian", lap, mixed["lap"])
    return max(
        worst, assert_mixed_orbital_derivative_fidelity(lap, full["fused"][4], f"{trexio_file} fused laplacian mixed vs full")
    )


def _check_grad(trexio_file: str, *, full_reference=None) -> float:
    if is_mixed_precision_enabled():
        return _check_grad_fidelity(trexio_file, *_mo_fidelity_pair(trexio_file, full_reference))
    mos_data, r_carts = _complex_mo_setup(trexio_file)
    atol, _ = get_tolerance_min(_GRAD_ZONES, "strict")
    analytic = compute_MOs_grad(mos_data=mos_data, r_carts=r_carts)
    auto, _ = _ad_mo_grad_lap(mos_data, r_carts)
    conj_auto, _ = _ad_mo_grad_lap(_conjugated(mos_data), r_carts)
    worst = 0.0
    for axis, an, au, cau in zip("xyz", analytic, auto, conj_auto, strict=True):
        label = f"{trexio_file} complex MO grad_{axis} analytic vs autodiff"
        _assert_complex_non_vacuous(label, au, atol)
        _conjugated_oracle_misses(label, au, cau, _GRAD_ZONES)
        _compare_grad(label, an, au)
        worst = max(worst, _deviation(an, au))
    return worst


def _check_laplacian(trexio_file: str, *, full_reference=None) -> float:
    if is_mixed_precision_enabled():
        return _check_laplacian_fidelity(trexio_file, *_mo_fidelity_pair(trexio_file, full_reference))
    mos_data, r_carts = _complex_mo_setup(trexio_file)
    atol, _ = get_tolerance_min(_LAP_ZONES, "strict")
    analytic = compute_MOs_laplacian(mos_data=mos_data, r_carts=r_carts)
    _, auto = _ad_mo_grad_lap(mos_data, r_carts)
    _, conj_auto = _ad_mo_grad_lap(_conjugated(mos_data), r_carts)
    label = f"{trexio_file} complex MO laplacian analytic vs autodiff"
    _assert_complex_non_vacuous(label, auto, atol)
    _conjugated_oracle_misses(label, auto, conj_auto, _LAP_ZONES)
    _compare_lap(label, analytic, auto)
    return _deviation(analytic, auto)


def _check_fused(trexio_file: str, *, full_reference=None) -> float:
    if is_mixed_precision_enabled():
        return _check_fused_fidelity(trexio_file, *_mo_fidelity_pair(trexio_file, full_reference))
    mos_data, r_carts = _complex_mo_setup(trexio_file)
    val_f, gx_f, gy_f, gz_f, lap_f = compute_MOs_value_grad_lap(mos_data=mos_data, r_carts=r_carts)
    val_s = compute_MOs(mos_data=mos_data, r_carts=r_carts)
    grads_s = compute_MOs_grad(mos_data=mos_data, r_carts=r_carts)
    lap_s = compute_MOs_laplacian(mos_data=mos_data, r_carts=r_carts)
    grads_u, lap_u = _ad_mo_grad_lap(mos_data, r_carts)
    atol_val, _ = get_tolerance_min(_VALUE_ZONES, "strict")
    _assert_complex_non_vacuous(f"{trexio_file} complex MO value", val_s, atol_val)
    _compare_value(f"{trexio_file} fused value vs compute_MOs", val_f, val_s)
    worst = _deviation(val_f, val_s)
    for axis, g_f, g_s, g_u in zip("xyz", (gx_f, gy_f, gz_f), grads_s, grads_u, strict=True):
        _compare_grad(f"{trexio_file} fused grad_{axis} vs compute_MOs_grad", g_f, g_s)
        _compare_grad(f"{trexio_file} fused grad_{axis} vs autodiff", g_f, g_u)
        worst = max(worst, _deviation(g_f, g_s), _deviation(g_f, g_u))
    _compare_lap(f"{trexio_file} fused laplacian vs compute_MOs_laplacian", lap_f, lap_s)
    _compare_lap(f"{trexio_file} fused laplacian vs autodiff", lap_f, lap_u)
    worst = max(worst, _deviation(lap_f, lap_s), _deviation(lap_f, lap_u))
    return worst


@pytest.mark.light
@pytest.mark.parametrize("trexio_file", _COMPLEX_MO_FIXTURES)
def test_complex_periodic_MOs_grad_analytic_vs_auto(trexio_file: str):
    """Full retains complex AD; mixed analytical gradients retain independently rebuilt full fidelity."""
    worst = _check_grad(trexio_file)
    print(f"{trexio_file}: max gradient deviation = {worst:.3e}")
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize("trexio_file", _COMPLEX_MO_FIXTURES)
def test_complex_periodic_MOs_laplacian_analytic_vs_auto(trexio_file: str):
    """Full retains complex AD; mixed analytical Laplacians retain independently rebuilt full fidelity."""
    worst = _check_laplacian(trexio_file)
    print(f"{trexio_file}: max Laplacian deviation = {worst:.3e}")
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize("trexio_file", _COMPLEX_MO_FIXTURES)
def test_complex_periodic_MOs_fused_matches_standalone_and_autodiff(trexio_file: str):
    """Fused complex-coefficient MO value / grad / Laplacian equal the standalone APIs and autodiff."""
    worst = _check_fused(trexio_file)
    print(f"{trexio_file}: max |fused - reference| = {worst:.3e}")
    jax.clear_caches()


def _install_conjugated_ao_derivative_mutant(monkeypatch) -> None:
    """Conjugate the analytic AO derivative inputs that the MO layer contracts with complex coefficients."""
    original_grad = mo_module.compute_AOs_grad
    original_lap = mo_module.compute_AOs_laplacian
    original_fused = mo_module.compute_AOs_value_grad_lap

    def grad_mutant(aos_data, r_carts):
        return tuple(jnp.conj(g) for g in original_grad(aos_data, r_carts))

    def lap_mutant(aos_data, r_carts):
        return jnp.conj(original_lap(aos_data, r_carts))

    def fused_mutant(aos_data, r_carts):
        val, gx, gy, gz, lap = original_fused(aos_data=aos_data, r_carts=r_carts)
        return val, jnp.conj(gx), jnp.conj(gy), jnp.conj(gz), jnp.conj(lap)

    jax.clear_caches()
    monkeypatch.setattr(mo_module, "compute_AOs_grad", grad_mutant)
    monkeypatch.setattr(mo_module, "compute_AOs_laplacian", lap_mutant)
    monkeypatch.setattr(mo_module, "compute_AOs_value_grad_lap", fused_mutant)


@pytest.mark.light
def test_negative_control_conjugated_ao_derivatives_fail_complex_mo_rows(monkeypatch):
    """Conjugating the analytic AO derivatives consumed by the MO layer fails all three complex rows."""
    trexio_file = _COMPLEX_MO_FIXTURES[0]
    full_reference = None
    if is_mixed_precision_enabled():
        _, full_reference = mixed_vs_full(lambda: _complex_mo_setup(trexio_file), _analytical_mo_outputs)
    _install_conjugated_ao_derivative_mutant(monkeypatch)
    grad_comparison = "mixed vs full" if is_mixed_precision_enabled() else "analytic vs autodiff"
    fused_comparison = "mixed vs full" if is_mixed_precision_enabled() else "vs autodiff"
    with pytest.raises(AssertionError, match=f"complex MO grad_x {grad_comparison}"):
        _check_grad(trexio_file, full_reference=full_reference)
    with pytest.raises(AssertionError, match=f"complex MO laplacian {grad_comparison}"):
        _check_laplacian(trexio_file, full_reference=full_reference)
    with pytest.raises(AssertionError, match=f"fused grad_x {fused_comparison}"):
        _check_fused(trexio_file, full_reference=full_reference)
    monkeypatch.undo()
    jax.clear_caches()
