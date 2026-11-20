"""Numerical evidence consuming the independently specified canonical fixtures."""

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

from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

from jqmc._precision import get_tolerance_min
from jqmc.determinant import compute_geminal_all_elements
from jqmc.wavefunction import evaluate_wavefunction
from gdd.engine.canonical_use import CanonicalKineticCapture
from gdd.engine.kinetic_activation import activation_errors
from gdd.engine.orbital_activation import orbital_activation_errors
from tests.canonical_wavefunction_fixtures import CANONICAL_PATTERN_IDS, build_canonical_hamiltonian, canonical_signature


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol):
    """Keep the homogeneous-block bound transparent to GDD runtime receipts."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max()


@jax.jit
def _kinetic_from_psi_hessian(wavefunction, r_up, r_dn):
    """Use the defining kinetic operator on complex Psi, not on log of its magnitude."""
    n_up = r_up.shape[0]
    positions = jnp.concatenate((r_up, r_dn))

    def psi(points):
        return evaluate_wavefunction(wavefunction, points[:n_up], points[n_up:])

    hessian = jax.jacfwd(jax.jacfwd(psi))(positions)
    return -0.5 * jnp.einsum("ijij->i", hessian) / psi(positions)


def _canonical_kinetic_case(pattern_id):
    """Rebuild the original canonical source and PCG64(864) positions."""
    ham = build_canonical_hamiltonian(pattern_id)
    geminal = ham.wavefunction_data.geminal_data
    rng = np.random.default_rng(864)
    r_up = rng.normal(scale=0.6, size=(geminal.num_electron_up, 3))
    r_dn = rng.normal(scale=0.6, size=(geminal.num_electron_dn, 3))
    return ham, r_up, r_dn


@pytest.mark.parametrize("pattern_id", CANONICAL_PATTERN_IDS)
def test_canonical_kinetic_matches_wavefunction_ad(pattern_id):
    """Each canonical wavefunction's per-electron kinetic equals phase-preserving AD.

    Input: a synthetic parameter-set Hamiltonian, not a TREXIO artifact, realizing
    the selected WF-PATTERN from the canonical SSoT. Electron positions are a single
    PCG64(seed=864) normal(scale=0.6) draw with the fixture's up/down counts. No
    conditioning screen is applied; cond(G) is measured and must be finite.
    The full complex Psi Hessian retains the original strict AD oracle. Mixed
    compares independently rebuilt analytical full/mixed per-electron outputs
    under the central spatial-derivative fidelity budget. Activation and orbital
    controls retain the original entry-policy pair in both callbacks, without
    component-wise cancellation hidden by a total energy sum.
    Removing each requested Jastrow/DTN channel individually must change this same
    per-electron kinetic outside that policy's element-wise comparison band.
    Each composite orbital leaf must also affect kinetic when its AO coupling block
    alone is zeroed: paired spins together, independent spins separately, and J3
    separately. The basis, retained parameters and electron positions stay unchanged.
    """
    if is_mixed_precision_enabled():
        # Activation/liveness is a same-policy contract, separate from the AD
        # oracle. Retain the original mixed entry pair in both callbacks.
        control_atol, control_rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "jastrow_grad_lap", "wf_kinetic"), "strict")

        def analytical(case):
            ham, r_up, r_dn = case
            wavefunction = ham.wavefunction_data
            geminal = wavefunction.geminal_data
            cond_g = float(np.linalg.cond(np.asarray(compute_geminal_all_elements(geminal, r_up, r_dn))))
            assert np.isfinite(cond_g)
            capture = CanonicalKineticCapture(ham, activation=True, orbital_activation=True)
            actual = np.concatenate(capture.evaluate(wavefunction, r_up, r_dn))
            assert actual.shape == (geminal.num_electron_up + geminal.num_electron_dn,)
            assert np.all(np.isfinite(actual))
            assert (
                activation_errors(capture.records[0], canonical_signature(pattern_id), atol=control_atol, rtol=control_rtol)
                == []
            )
            assert (
                orbital_activation_errors(
                    capture.records[0], canonical_signature(pattern_id), atol=control_atol, rtol=control_rtol
                )
                == []
            )
            return actual

        mixed, full = mixed_vs_full(lambda: _canonical_kinetic_case(pattern_id), analytical)
        blocks = mixed_fidelity_blocks(mixed, full)
        atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
        for mixed_block, full_block in blocks:
            _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol)
        corrupted = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            _assert_component_derivative_fidelity(corrupted, blocks[0][1], atol, rtol)
        jax.clear_caches()
        return

    ham, r_up, r_dn = _canonical_kinetic_case(pattern_id)
    wavefunction = ham.wavefunction_data
    geminal = wavefunction.geminal_data
    cond_g = float(np.linalg.cond(np.asarray(compute_geminal_all_elements(geminal, r_up, r_dn))))
    assert np.isfinite(cond_g)
    capture = CanonicalKineticCapture(ham, activation=True, orbital_activation=True)
    actual = np.concatenate(capture.evaluate(wavefunction, r_up, r_dn))
    reference = np.asarray(_kinetic_from_psi_hessian(wavefunction, r_up, r_dn))
    assert actual.shape == reference.shape == (geminal.num_electron_up + geminal.num_electron_dn,)
    assert np.all(np.isfinite(actual))
    assert np.all(np.isfinite(reference))
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "jastrow_grad_lap", "wf_kinetic"), "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
    assert activation_errors(capture.records[0], canonical_signature(pattern_id), atol=atol, rtol=rtol) == []
    assert orbital_activation_errors(capture.records[0], canonical_signature(pattern_id), atol=atol, rtol=rtol) == []
    jax.clear_caches()
