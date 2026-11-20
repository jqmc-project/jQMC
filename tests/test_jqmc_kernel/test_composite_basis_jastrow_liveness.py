"""Composite geminal basis derivatives with active Jastrow factors remain live."""

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

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from tests.test_jqmc_kernel import test_jqmc_derivatives as fixtures

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


def _real_composite_j123():
    ham = fixtures._build_composite_pgtopw_ham(fixtures._LI_GAMMA_SAVE, fixtures._LI_GAMMA_UPF, "Li")
    wf = ham.wavefunction_data
    aos = wf.geminal_data.orb_data_up_spin.aos_data
    j1 = fixtures.Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=ham.structure_data,
        core_electrons=fixtures.core_electrons_of(ham.coulomb_potential_data, len(ham.structure_data.positions)),
        jastrow_1b_type="exp",
    )
    j3_orb = fixtures.MOs_data(num_mo=aos.num_ao, aos_data=aos, mo_coefficients=np.eye(aos.num_ao))
    j3 = fixtures._make_twist_j3(j3_orb)
    jd = wf.jastrow_data.replace(jastrow_one_body_data=j1, jastrow_three_body_data=j3)
    return ham.replace(wavefunction_data=wf.replace(jastrow_data=jd))


def _composite():
    ham = _real_composite_j123()
    fixtures._assert_pgtopw_composite(ham.wavefunction_data.geminal_data)
    jd = ham.wavefunction_data.jastrow_data
    assert jd.jastrow_one_body_data is not None
    assert jd.jastrow_two_body_data is not None
    assert jd.jastrow_three_body_data is not None
    return ham


def _assert_live_composite_blocks(analytic, reference, atol, rtol):
    budget_atol, budget_rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    assert (atol, rtol) == (budget_atol, budget_rtol), "Caller and direct original central budget must agree"
    for name in fixtures._PBC_BASIS_BLOCKS:
        actual, expected = analytic[name], reference[name]
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), name
        np.testing.assert_allclose(
            actual, expected, atol=budget_atol, rtol=budget_rtol, equal_nan=False, err_msg=f"composite {name}"
        )
        assert np.max(np.abs(actual)) > atol, f"inactive analytic composite {name}"
        assert np.max(np.abs(expected)) > atol, f"inactive autodiff composite {name}"


@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_real_composite_lambda_basis_is_live_and_matches_autodiff(quantity):
    """Composite geminal exponent/coefficient derivatives are nonzero and match same-mode autodiff.

    Li Gamma ccECP pGTO+PW carries J1-exp, J2-Pade and a seeded real MO-J3.
    It uses the existing seed-zero electron positions, no condition screening,
    explicit geminal radial-basis flags, and the existing strict crossed-zone
    tolerance. The local-energy blocks must change when Jastrow is removed.
    """
    if is_mixed_precision_enabled():
        _composite_jastrow_mixed_fidelity(quantity)
        return
    ham = _composite()
    r_up, r_dn = fixtures._draw(ham)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    analytic = fixtures._pbc_basis_blocks(ham, r_up, r_dn, quantity, analytic=True)
    reference = fixtures._pbc_basis_blocks(ham, r_up, r_dn, quantity, analytic=False)
    _assert_live_composite_blocks(analytic, reference, atol, rtol)
    if quantity == "e_L":
        bare = fixtures._pbc_basis_blocks(fixtures._without_jastrow(ham), r_up, r_dn, quantity, analytic=False)
        for name in fixtures._PBC_BASIS_BLOCKS:
            assert np.max(np.abs(reference[name] - bare[name])) > atol, f"inactive composite Jastrow coupling {name}"


def test_real_composite_basis_check_rejects_dropped_jastrow_kinetic_term(monkeypatch):
    """Dropping Jastrow's contribution to real-composite kinetic basis derivatives is rejected."""
    from jqmc.derivatives import basis

    ham = _real_composite_j123()
    r_up, r_dn = fixtures._draw(ham)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    reference = fixtures._pbc_basis_blocks(ham, r_up, r_dn, "e_L", analytic=False)
    original = basis.grad_T_wrt_basis

    def dropped(gd, jastrow_data, r_up, r_dn, geminal_inv=None, *, holomorphic=False):
        return original(gd, fixtures.Jastrow_data(), r_up, r_dn, geminal_inv=geminal_inv, holomorphic=holomorphic)

    jax.clear_caches()
    monkeypatch.setattr(basis, "grad_T_wrt_basis", dropped)
    try:
        mutant = fixtures._pbc_basis_blocks(ham, r_up, r_dn, "e_L", analytic=True)
    finally:
        monkeypatch.undo()
        jax.clear_caches()
    with pytest.raises(AssertionError, match="composite lambda_basis_"):
        _assert_live_composite_blocks(mutant, reference, atol, rtol)


def _composite_jastrow_assert_mixed_bound(mixed_block, full_block, atol, rtol, label):
    """Bind the existing central maximum-deviation budget to one field."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max(), label


def _composite_jastrow_mixed_fidelity(quantity):
    """Preserve old liveness and Jastrow coupling with fresh analytical models."""
    from jqmc.derivatives import bridge
    from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

    live_atol, _live_rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")

    def factory():
        bridge.cache_clear()
        fixtures.derivatives._jitted_assemble.cache_clear()
        fixtures._build_composite_pgtopw_ham.cache_clear()
        ham = _composite()
        r_up, r_dn = fixtures._draw(ham)
        return ham, r_up, r_dn

    def evaluate(case):
        ham, r_up, r_dn = case
        actual = fixtures._pbc_basis_blocks(ham, r_up, r_dn, quantity, analytic=True)
        assert set(actual) == set(fixtures._PBC_BASIS_BLOCKS)
        for name, value in actual.items():
            assert np.asarray(value).size and np.isfinite(value).all(), name
            assert np.max(np.abs(value)) > live_atol, f"inactive analytic composite {name}"
        if quantity == "e_L":
            bare = fixtures._pbc_basis_blocks(fixtures._without_jastrow(ham), r_up, r_dn, quantity, analytic=True)
            assert set(bare) == set(actual)
            for name in fixtures._PBC_BASIS_BLOCKS:
                assert np.asarray(actual[name]).shape == np.asarray(bare[name]).shape
                assert np.max(np.abs(actual[name] - bare[name])) > live_atol, f"inactive composite Jastrow coupling {name}"
        return {"source": actual, "up": r_up, "down": r_dn}

    try:
        mixed, full = mixed_vs_full(factory, evaluate)
    finally:
        # A Full callback must not leave its cached Hamiltonian for the next
        # untouched Mixed negative detector or another physical case.
        fixtures._build_composite_pgtopw_ham.cache_clear()
    assert np.array_equal(mixed["up"], full["up"]) and np.array_equal(mixed["down"], full["down"])
    assert set(mixed["source"]) == set(full["source"]) == set(fixtures._PBC_BASIS_BLOCKS)
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    compared = 0
    for name, reference in full["source"].items():
        actual = np.asarray(mixed["source"][name])
        reference = np.asarray(reference)
        assert actual.shape == reference.shape and actual.size
        assert np.iscomplexobj(actual) == np.iscomplexobj(reference)
        parts = ("real", "imag") if np.iscomplexobj(reference) else ("real",)
        for part in parts:
            for mixed_block, full_block in mixed_fidelity_blocks(getattr(actual, part), getattr(reference, part)):
                _composite_jastrow_assert_mixed_bound(
                    mixed_block, full_block, atol, rtol, f"composite {quantity}/{name}/{part}"
                )
                compared += 1
                faulty = out_of_budget_mixed_block(mixed_block, full_block, atol=atol, rtol=rtol)
                with pytest.raises(AssertionError):
                    _composite_jastrow_assert_mixed_bound(faulty, full_block, atol, rtol, "composite excess-error control")
    assert compared > 0
