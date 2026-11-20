"""Cartesian Bloch pGTO ccECP nonlocal basis derivatives retain complex weights."""

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
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors
from jqmc.derivatives import assemble, ecp_force, one_move_mesh
from tests._precision_fidelity import mixed_vs_full
from tests.test_jqmc_kernel._orbital_mixed_fidelity import assert_mixed_parameter_derivative_fidelity
from tests.test_jqmc_kernel import test_jqmc_derivatives as fixtures

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]
_FLAGS = {"opt_lambda_basis_exp": True, "opt_lambda_basis_coeff": True}
_BLOCKS = ("lambda_basis_exp", "lambda_basis_coeff")


def _cart_twist_case():
    ham = fixtures._build_ham("lithium_k025_cart.h5")
    assert type(ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data).__name__ == "AOs_cart_pgto_complex_data"
    r_up, r_dn = fixtures._draw(ham)
    return ham, r_up, r_dn


def _basis_blocks(ham, raw):
    real = jax.tree_util.tree_map(jnp.real, raw)
    blocks = fixtures._active_param_grads(ham.wavefunction_data, real, _FLAGS)
    assert set(blocks) == set(_BLOCKS)
    return {name: np.asarray(blocks[name]) for name in _BLOCKS}


def _nonlocal_autodiff(ham, r_up, r_dn):
    def forward(wavefunction):
        masked = wavefunction.with_param_grad_mask(**_FLAGS)
        return jnp.real(
            compute_ecp_non_local_parts_nearest_neighbors(ham.coulomb_potential_data, masked, r_up, r_dn, jnp.eye(3))[-1]
        )

    return _basis_blocks(ham, jax.grad(forward)(ham.wavefunction_data))


def _assert_live_ecp_basis(actual, reference, atol, rtol):
    budget_atol, budget_rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    assert (atol, rtol) == (budget_atol, budget_rtol), "Full caller and direct central budget must agree"
    for name in _BLOCKS:
        a, b = actual[name], reference[name]
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), name
        np.testing.assert_allclose(
            a, b, atol=budget_atol, rtol=budget_rtol, equal_nan=False, err_msg=f"cart twist nonlocal {name}"
        )
        assert np.max(np.abs(a)) > atol, f"inactive analytical nonlocal {name}"
        assert np.max(np.abs(b)) > atol, f"inactive autodiff nonlocal {name}"


def _cart_twist_analytical(case):
    ham, r_up, r_dn = case
    return _basis_blocks(ham, assemble._dV_ecp_dc(ham, r_up, r_dn, jnp.eye(3)))


def _assert_live_ecp_basis_fidelity(actual, reference):
    # The liveness floor is the medium tier; the numerical comparison uses the
    # central mixed/full production budget.
    atol, _ = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "medium")
    for name in _BLOCKS:
        assert np.max(np.abs(actual[name])) > atol, f"inactive analytical nonlocal {name}"
        assert np.max(np.abs(reference[name])) > atol, f"inactive full nonlocal {name}"
        assert_mixed_parameter_derivative_fidelity(actual[name], reference[name], f"cart twist nonlocal {name}")


def test_cart_twist_nonlocal_basis_matches_autodiff():
    """Live Cartesian twist ccECP basis derivatives match autodiff of the isolated nonlocal potential.

    The committed lithium_k025_cart.h5 fixture carries J1/J2 and uses the
    existing seed-zero electron positions and fixed identity ECP rotation.
    Both geminal exponent/coefficient blocks are compared independently of
    kinetic energy, so cancellation in the total energy cannot hide a defect.
    Full retains the strict same-mode AD oracle. Mixed uses independently
    rebuilt full production blocks and the central parameter-derivative fidelity
    budget, and keeps the mixed-medium liveness floor.
    """
    if is_mixed_precision_enabled():
        mixed, full = mixed_vs_full(_cart_twist_case, _cart_twist_analytical)
        _assert_live_ecp_basis_fidelity(mixed, full)
        return
    ham, r_up, r_dn = _cart_twist_case()
    level = "strict"
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), level)
    actual = _basis_blocks(ham, assemble._dV_ecp_dc(ham, r_up, r_dn, jnp.eye(3)))
    reference = _nonlocal_autodiff(ham, r_up, r_dn)
    _assert_live_ecp_basis(actual, reference, atol, rtol)


def test_cart_twist_nonlocal_basis_rejects_real_projected_weights(monkeypatch):
    """Discarding the mesh weight's imaginary part before basis contraction makes the Cartesian oracle fail.

    The unchanged finite/live comparison passes first. The mutation affects
    only the analytical fast-path weights, not the primal ECP or AD reference;
    a call witness requires that the injected defect actually ran.
    """
    ham, r_up, r_dn = _cart_twist_case()
    assert one_move_mesh.fast_supported_c(ham)
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), level)
    if is_mixed_precision_enabled():
        baseline, reference = mixed_vs_full(_cart_twist_case, _cart_twist_analytical)
        _assert_live_ecp_basis_fidelity(baseline, reference)
        # The full reference is a finite host result from the clean route.
        # Rebuild only the mixed case before the analytical fault is installed.
        ham, r_up, r_dn = _cart_twist_case()
    else:
        reference = _nonlocal_autodiff(ham, r_up, r_dn)
        baseline = _cart_twist_analytical((ham, r_up, r_dn))
        _assert_live_ecp_basis(baseline, reference, atol, rtol)
    original = ecp_force._ecp_nl_common
    calls = []

    def projected_weights(*args, **kwargs):
        common = original(*args, **kwargs)
        calls.append(True)
        return dict(common, Vgrid=jnp.real(common["Vgrid"]))

    jax.clear_caches()
    monkeypatch.setattr(ecp_force, "_ecp_nl_common", projected_weights)
    try:
        actual = _basis_blocks(ham, assemble._dV_ecp_dc(ham, r_up, r_dn, jnp.eye(3)))
    finally:
        monkeypatch.undo()
        jax.clear_caches()
    assert calls, "weight-projection fault did not engage"
    for name in _BLOCKS:
        assert np.all(np.isfinite(actual[name]))
        with pytest.raises(AssertionError, match=f"cart twist nonlocal {name}"):
            if is_mixed_precision_enabled():
                assert_mixed_parameter_derivative_fidelity(actual[name], reference[name], f"cart twist nonlocal {name}")
            else:
                np.testing.assert_allclose(
                    actual[name],
                    reference[name],
                    atol=atol,
                    rtol=rtol,
                    equal_nan=False,
                    err_msg=f"cart twist nonlocal {name}",
                )
