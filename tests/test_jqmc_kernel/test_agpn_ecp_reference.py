"""Same-precision primal reference for real open-shell AGPn ECP parameters."""

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

from jqmc import derivatives
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import ATOL_NONVACUITY_DISTINCTION, get_tolerance_mixed_fidelity
from jqmc.derivatives import bridge
from tests.test_rs_kernel.test_rs_analytic_matrix import _xla_c
from tests.test_rs_kernel.test_rs_complex_twist_param_parity import _open_shell_real_agpn_case

pytestmark = pytest.mark.with_chdir
_ENERGY_DERIVATIVE_ZONES = ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic", "jastrow_grad_lap")


@pytest.fixture(autouse=True)
def _clear_caches():
    yield
    bridge.cache_clear()
    derivatives._jitted_assemble.cache_clear()
    jax.clear_caches()


def _open_shell_real_el_autodiff(ham, r_up, r_dn):
    """Differentiate the unchanged forward model in the currently selected precision."""
    from jqmc.hamiltonians import compute_local_energy

    wf = ham.wavefunction_data
    gem = wf.geminal_data

    def energy(diag, unpaired):
        changed = ham.replace(
            wavefunction_data=wf.replace(geminal_data=gem.replace(lambda_diag=diag, lambda_unpaired=unpaired))
        )
        return jnp.real(compute_local_energy(changed, r_up, r_dn, jnp.eye(3)))

    diag, unpaired = jax.jit(jax.grad(energy, argnums=(0, 1)))(gem.lambda_diag, gem.lambda_unpaired)
    return {"paired": np.asarray(diag), "unpaired": np.asarray(unpaired)}


@pytest.mark.heavy
@pytest.mark.parametrize("block", ["paired", "unpaired"])
def test_open_shell_twist_agpn_real_el_analytic_matches_autodiff(tmp_path, block):
    """Real AGPn paired/unpaired e_L gradients equal same-precision primal AD.

    QE O-lsda_k025/out/o_k025.save and its O.ccECP.upf produce complex pGTO
    orbitals with four up/two down electrons (open shell). The explicit
    common-rank AGPn constructor retains the separate spin MO seeds. PCG64
    seed 431 selects real nonfactorized paired and unpaired parameters; seed
    41 selects one Gaussian configuration (scale 0.6 about the first nucleus).
    No cond(G) rejection is used. The ordinary 12-point ECP mesh has identity
    rotation. AD differentiates Re(e_L) for these real parameters in the same
    active precision; no fp64 oracle replaces a mixed forward calculation.
    Both leaves must be live. This is not a complex-parameter or ln_Psi claim.
    """
    ham, r_up, r_dn = _open_shell_real_agpn_case(tmp_path, np.random.default_rng(431), np.random.default_rng(41))
    oracle = _open_shell_real_el_autodiff(ham, r_up, r_dn)[block]
    grad = _xla_c(ham, r_up, r_dn, "Det", "e_L").geminal_data
    actual = np.asarray(grad.lambda_diag if block == "paired" else grad.lambda_unpaired)
    assert np.max(np.abs(oracle)) > ATOL_NONVACUITY_DISTINCTION
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(_ENERGY_DERIVATIVE_ZONES, level)
    np.testing.assert_allclose(actual, oracle, atol=atol, rtol=rtol, err_msg=f"AGPn real e_L AD/{block}")
    if is_mixed_precision_enabled():
        _twist_agpn_mixed_fidelity(tmp_path, block)


def _twist_agpn_assert_mixed_bound(mixed_block, full_block, atol, rtol, label):
    """Bind one unchanged central maximum-deviation comparison."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max(), label


def _twist_agpn_mixed_fidelity(tmp_path, block):
    """Preserve both live real-parameter leaves with independent analytical models."""
    from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

    def factory():
        bridge.cache_clear()
        derivatives._jitted_assemble.cache_clear()
        directory = tmp_path / ("mixed-source" if is_mixed_precision_enabled() else "full-source")
        directory.mkdir(exist_ok=True)
        return _open_shell_real_agpn_case(directory, np.random.default_rng(431), np.random.default_rng(41))

    def evaluate(case):
        ham, r_up, r_dn = case
        assert ham.structure_data.pbc_flag and ham.coulomb_potential_data.ecp_flag
        gem = ham.wavefunction_data.geminal_data
        grad = _xla_c(ham, r_up, r_dn, "Det", "e_L").geminal_data
        expected_shape = gem.lambda_diag.shape if block == "paired" else gem.lambda_unpaired.shape
        actual = np.asarray(grad.lambda_diag if block == "paired" else grad.lambda_unpaired)
        assert actual.shape == expected_shape and actual.size
        assert np.isfinite(actual).all()
        # The original AD-oracle non-vacuity control remains on each independent
        # production reference and actual leaf, with the identical named floor.
        assert np.max(np.abs(actual)) > ATOL_NONVACUITY_DISTINCTION
        return {"source": actual, "up": r_up, "down": r_dn}

    mixed, full = mixed_vs_full(factory, evaluate)
    assert np.array_equal(mixed["up"], full["up"]) and np.array_equal(mixed["down"], full["down"])
    assert np.iscomplexobj(mixed["source"]) == np.iscomplexobj(full["source"])
    parts = ("real", "imag") if np.iscomplexobj(full["source"]) else ("real",)
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    compared = 0
    for part in parts:
        left = getattr(mixed["source"], part)
        right = getattr(full["source"], part)
        for mixed_block, full_block in mixed_fidelity_blocks(left, right):
            _twist_agpn_assert_mixed_bound(mixed_block, full_block, atol, rtol, f"twist AGPn {block}/{part}")
            compared += 1
            faulty = out_of_budget_mixed_block(mixed_block, full_block, atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _twist_agpn_assert_mixed_bound(faulty, full_block, atol, rtol, "twist AGPn excess-error control")
    assert compared > 0
