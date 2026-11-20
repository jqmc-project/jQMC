"""Independent same-mode orbital, geminal, J3 and AS value references."""

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

from jqmc import atomic_orbital, determinant, jastrow_factor, molecular_orbital
from jqmc._precision import get_complex_dtype_np, get_dtype_np, get_tolerance_min
from tests.canonical_wavefunction_fixtures import CANONICAL_PATTERN_IDS, build_canonical_hamiltonian

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_J3_PATTERNS = (
    "WF-PATTERN-001",
    "WF-PATTERN-002",
    "WF-PATTERN-003",
    "WF-PATTERN-004",
    "WF-PATTERN-005",
    "WF-PATTERN-006",
    "WF-PATTERN-007",
    "WF-PATTERN-008",
    "WF-PATTERN-009",
    "WF-PATTERN-010",
    "WF-PATTERN-011",
    "WF-PATTERN-016",
)


def _inputs(pattern):
    wf = build_canonical_hamiltonian(pattern).wavefunction_data
    geminal = wf.geminal_data
    rng = np.random.default_rng(864)
    r_up = rng.normal(scale=0.6, size=(geminal.num_electron_up, 3))
    r_dn = rng.normal(scale=0.6, size=(geminal.num_electron_dn, 3))
    return wf, r_up, r_dn


def _expected_ao_value_dtype(aos):
    """Apply the declared leaf policy without reading an evaluator's dtype."""
    if isinstance(aos, atomic_orbital.AOs_composite_data):
        return np.result_type(*(_expected_ao_value_dtype(child) for child in aos.children))
    bloch_gto = isinstance(aos, (atomic_orbital.AOs_cart_pgto_complex_data, atomic_orbital.AOs_sphe_pgto_complex_data))
    zone = "ao_bloch_eval" if bloch_gto else "ao_eval"
    return np.dtype(get_complex_dtype_np(zone) if atomic_orbital.is_complex_aos(aos) else get_dtype_np(zone))


def _compare_geminal(wf, r_up, r_dn):
    actual = np.asarray(determinant.compute_geminal_all_elements(wf.geminal_data, r_up, r_dn))
    reference = determinant._compute_geminal_all_elements_debug(wf.geminal_data, r_up, r_dn)
    atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "det_eval"), "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)


def _compare_as(wf, r_up, r_dn):
    actual = np.asarray(determinant.compute_AS_regularization_factor(wf.geminal_data, r_up, r_dn))
    reference = determinant._compute_AS_regularization_factor_debug(wf.geminal_data, r_up, r_dn)
    atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "det_eval"), "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)


def _compare_j3(wf, r_up, r_dn):
    j3 = wf.jastrow_data.jastrow_three_body_data
    assert j3 is not None
    actual = np.asarray(jastrow_factor.compute_Jastrow_three_body(j3, r_up, r_dn))
    reference = jastrow_factor._compute_Jastrow_three_body_debug(j3, r_up, r_dn)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval"), "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("pattern", CANONICAL_PATTERN_IDS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_geminal_and_as_use_independent_orbital_values(pattern):
    """Geminal and AS agree with NumPy orbital/contraction/inverse references on all 16 canonical inputs.

    Synthetic parameter-set wavefunctions, no TREXIO file, use their declared
    SD/AGPn/AGP, spin and cell signatures. One PCG64 seed-864 normal(scale=0.6)
    electron configuration is retained without cond(G) rejection. Full and
    mixed each use their own AO/MO/Det arithmetic and unchanged strict policy.
    The AS reference uses a direct NumPy inverse, not the production SVD.
    This tests value independence, not arbitrary near-singular stability.
    """
    args = _inputs(pattern)
    _compare_geminal(*args)
    _compare_as(*args)
    jax.clear_caches()


@pytest.mark.parametrize("pattern", _J3_PATTERNS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_j3_uses_independent_orbital_values(pattern):
    """AO/MO J3 equals independent NumPy orbital values and explicit pair sums on all 12 J3 canonical inputs.

    No files are read. Preserve each canonical wavefunction's orbital/spin/cell
    signature and seed-864 normal(scale=0.6) electron configuration without
    conditioning selection. NumPy AO arithmetic follows the active AO zone;
    J3 accumulation follows the Jastrow zone. Compare within each precision at
    unchanged strict AO/Jastrow tolerance, including complex Bloch inputs.
    """
    _compare_j3(*_inputs(pattern))
    jax.clear_caches()


@pytest.mark.parametrize("pattern", CANONICAL_PATTERN_IDS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_independent_ao_values_preserve_active_precision(pattern):
    """Both orbital channels match independent NumPy values with the active real/complex AO dtype.

    Use synthetic canonical 001--016 with seed-864 normal(scale=0.6) up-spin
    positions as evaluation points for both channels, including the unused
    down orbital on empty-down cases. No TREXIO, screening or cross-precision
    reference is used. Covers GTO, pGTO, PW, spline and composites, open/PBC,
    Gamma/non-Gamma and all declared cell families. Dtype equality is exact;
    values use the existing strict AO tolerance. Negative/outside-cell points
    retain the defining Bloch phase after periodic image reduction.
    Complex pGTO uses ao_bloch_eval; composite concatenation promotes the
    declared child dtypes. Both production and independent reference dtypes
    are checked, without inferring the expected dtype from either output.
    """
    wf, r_up, _ = _inputs(pattern)
    gd = wf.geminal_data
    for channel, orbitals in (("up", gd.orb_data_up_spin), ("down", gd.orb_data_dn_spin)):
        aos = orbitals.aos_data if isinstance(orbitals, molecular_orbital.MOs_data) else orbitals
        actual = np.asarray(atomic_orbital.compute_AOs(aos, r_up))
        reference = atomic_orbital._compute_AOs_debug(aos, r_up)
        dtype = _expected_ao_value_dtype(aos)
        assert actual.dtype == dtype
        actual_dtype = np.asarray([ord(reference.dtype.kind), reference.dtype.itemsize])
        expected_dtype = np.asarray([ord(np.dtype(dtype).kind), np.dtype(dtype).itemsize])
        np.testing.assert_array_equal(actual_dtype, expected_dtype, err_msg=channel)
        atol, rtol = get_tolerance_min(("ao_eval",), "strict")
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=channel)
    jax.clear_caches()


@pytest.mark.parametrize("target", ["geminal", "jastrow", "as"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_independent_composites_reject_production_operand_fault(monkeypatch, target):
    """A perturbed production orbital/geminal cannot move the independent composite reference with it.

    On canonical 007 (AGP/PW with MO J3), scale only the first orbital row or
    first geminal entry by 1.01 in the actual callable used by production.
    Baseline same-mode comparison must pass first, then the same comparison
    must fail, while the independent reference remains bit-identical. The
    nonuniform geminal fault avoids AS's exact uniform-scaling invariance.
    """
    args = _inputs("WF-PATTERN-007")
    compare = {"geminal": _compare_geminal, "jastrow": _compare_j3, "as": _compare_as}[target]
    compare(*args)
    wf, r_up, r_dn = args
    if target == "geminal":
        module, name = determinant, "compute_AOs"
        reference_function = lambda: determinant._compute_geminal_all_elements_debug(wf.geminal_data, r_up, r_dn)
    elif target == "jastrow":
        module, name = jastrow_factor, "compute_MOs"
        reference_function = lambda: jastrow_factor._compute_Jastrow_three_body_debug(
            wf.jastrow_data.jastrow_three_body_data, r_up, r_dn
        )
    else:
        module, name = determinant, "compute_geminal_all_elements"
        reference_function = lambda: determinant._compute_AS_regularization_factor_debug(wf.geminal_data, r_up, r_dn)
    reference_before = reference_function()
    original = getattr(module, name)

    def wrong(*args, **kwargs):
        values = original(*args, **kwargs)
        return values.at[0, 0].multiply(1.01) if target == "as" else values.at[0].multiply(1.01)

    monkeypatch.setattr(module, name, wrong)
    jax.clear_caches()
    with pytest.raises(AssertionError):
        compare(*args)
    np.testing.assert_array_equal(reference_before, reference_function())
    jax.clear_caches()
