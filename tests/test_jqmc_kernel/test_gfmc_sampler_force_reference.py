"""Deterministic independent references for sampler-to-buffer force moments."""

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

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import jqmc_gfmc
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from tests._gfmc_sampler_force_oracle import ObserveBranching, force_moments, primal_force_reference, pw_factor
from tests.test_jqmc_kernel.test_gfmc_force_analytic import _build_ham
from tests.test_rs_kernel.conftest import assert_native_ran, assert_no_native_ran, native_ctx_spy
from tests.test_rs_kernel.test_gfmc_projection import _backend_env


pytestmark = [pytest.mark.with_chdir]


def _epsilon(regime):
    return {"disabled": 0.0, "outside": 0.05, "interior": 1.8}[regime]


def _run_one_interval(ham, algorithm, backend, use_swct, epsilon):
    nuclei = np.asarray(ham.structure_data.positions)
    r_up_carts = nuclei[0] + np.array([[0.2, 0.1, 0.15], [0.9, -0.3, 0.2], [-0.5, 0.4, -0.2]])
    r_dn_carts = nuclei[1] + np.array([[-0.3, 0.2, 0.1], [0.4, 0.7, -0.3], [0.1, -0.6, 0.5]])
    with _backend_env("fast" if backend == "rust" else "jax"):
        klass = jqmc_gfmc.GFMC_t if algorithm == "conventional" else jqmc_gfmc.GFMC_n
        specific = {"tau": 0.05} if algorithm == "conventional" else {"E_scf": -1.0, "num_mcmc_per_measurement": 1}
        driver = klass(
            hamiltonian_data=ham,
            num_walkers=3,
            num_gfmc_collect_steps=2,
            mcmc_seed=34,
            alat=0.30,
            random_discretized_mesh=False,
            non_local_move="tmove",
            comput_position_deriv=True,
            use_swct=use_swct,
            epsilon_PW=epsilon,
            **specific,
        )
        assert driver._mpi_size == 1, "this moment reference is a serial contract"
        driver._latest_r_up_carts = jnp.asarray(r_up_carts[:, None, :])
        driver._latest_r_dn_carts = jnp.asarray(r_dn_carts[:, None, :])
        observer = ObserveBranching(driver, algorithm)
        driver._branching_rng = observer
        with native_ctx_spy() as counts:
            driver.run(num_mcmc_steps=1)
    if backend == "rust":
        assert_native_ran(counts, "sampler force moment reference")
    else:
        assert_no_native_ran(counts, "sampler force moment reference")
    assert len(observer.rows) == 1, "exactly one pre-reconfiguration snapshot is required"
    return driver, observer.rows[0]


def _require_effect(wrong, reference, atol, rtol):
    assert np.any(np.abs(np.asarray(wrong) - reference) > atol + rtol * np.abs(reference)), (
        "force fixture lost its required numerical effect"
    )


@pytest.mark.heavy
@pytest.mark.parametrize("algorithm", ["conventional", "load-balanced"])
@pytest.mark.parametrize("backend", ["xla", "rust"])
@pytest.mark.parametrize("use_swct", [False, True])
@pytest.mark.parametrize("regime", ["disabled", "outside", "interior"])
def test_gfmc_sampler_force_moments_match_warped_primal_derivatives(algorithm, backend, use_swct, regime):
    """Actual GFMC force buffers equal independently differentiated fixed-state walker moments.

    Committed tests/trexio_example_files/H2_ae_ccpvdz_cart.h5: open-boundary AE
    Cartesian-GTO SD, no Jastrow, one up and one down electron (closed-shell).
    Three explicit asymmetric offset pairs start the walkers; seed 34 fixes the
    driver keys. One identity-mesh interval uses alat 0.30, tmove and either
    conventional tau 0.05 or load-balanced one substep at E_scf=-1. Coordinates,
    energies and projection weights are observed before reconfiguration without
    changing its random draw. No cond(G) screening or statistical acceptance band.

    Independent AD differentiates the continuum leaf energy and primal log|Psi|;
    with SWCT it differentiates an explicit coordinate map and its log Jacobian.
    No production force gradient, SWCT contraction or stored moment is a reference
    input. Explicit Pathak-Wagner epsilon 0, 0.05 or 1.8 exercises disabled,
    exterior and nontrivial interior factors, including values above one.
    Each original walker contributes its E_L*Pulay product before averaging.
    The actual weights (with load-balanced V_diag-E_scf division) are given inputs,
    not independent evidence for the projection probability law or its weights.

    Energy and stored HF/Pulay/E_L*Pulay are separate comparisons at the existing
    zone strict (full) / medium (mixed) policy within that mode. This guarantees
    these molecular sampler-to-buffer moments, not physical ground-state forces,
    stress, ECP/PBC/complex kernels, distributed reduction or convergence.
    """
    ham = _build_ham("H2_ae_ccpvdz_cart.h5")
    epsilon = _epsilon(regime)
    driver, snapshot = _run_one_interval(ham, algorithm, backend, use_swct, epsilon)
    r_up_carts, r_dn_carts = snapshot["r_up"], snapshot["r_dn"]
    assert r_up_carts.shape == r_dn_carts.shape == (3, 1, 3)
    assert all(np.all(np.isfinite(value)) for value in snapshot.values()), "snapshot inputs must be finite"
    assert np.all(snapshot["weights"] > 0) and np.all(snapshot["denominator"] > 0)
    energy, hf, pp, distance = (
        np.asarray(value) for value in primal_force_reference(ham, r_up_carts, r_dn_carts, use_swct=use_swct)
    )
    assert all(np.all(np.isfinite(value)) for value in (energy, hf, pp, distance)), "primal references must be finite"
    weights = snapshot["weights"] / snapshot["denominator"]
    fractions = weights / np.sum(weights)
    factor = pw_factor(distance, epsilon)
    expected_hf, expected_pp, expected_cross = force_moments(energy, hf, pp, fractions, factor)
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap", "det_eval", "wf_eval", "wf_kinetic", "coulomb", "swct"), level)
    _require_effect(np.zeros_like(expected_hf), expected_hf, atol, rtol)
    _require_effect(np.zeros_like(expected_pp), expected_pp, atol, rtol)
    _require_effect(
        force_moments(energy, hf, pp, np.full(3, 1 / 3), factor),
        np.array([expected_hf, expected_pp, expected_cross]),
        atol,
        rtol,
    )
    _require_effect(np.sum(fractions * energy) * expected_pp, expected_cross, atol, rtol)
    if use_swct:
        _, bare_hf, bare_pp, _ = (
            np.asarray(value) for value in primal_force_reference(ham, r_up_carts, r_dn_carts, use_swct=False)
        )
        _require_effect(
            force_moments(energy, bare_hf, bare_pp, fractions, factor),
            np.array([expected_hf, expected_pp, expected_cross]),
            atol,
            rtol,
        )
    if regime == "interior":
        assert np.all((distance > 0) & (distance < epsilon))
        assert np.any(factor < 1) and np.any(factor > 1)
        _require_effect(
            force_moments(energy, hf, pp, fractions, np.ones(3)),
            np.array([expected_hf, expected_pp, expected_cross]),
            atol,
            rtol,
        )
        _require_effect(
            force_moments(energy, hf, pp, fractions, np.minimum(factor, 1)),
            np.array([expected_hf, expected_pp, expected_cross]),
            atol,
            rtol,
        )
    elif regime == "outside":
        assert np.all(distance > epsilon)
        assert np.all(factor == 1)
    else:
        assert regime == "disabled" and epsilon == 0
        assert np.all(factor == 1)
    actual_energy = snapshot["energy"]
    actual_hf, actual_pp, actual_cross = (
        np.asarray(getattr(driver, name))[0, 0] for name in ("_stored_force_HF", "_stored_force_PP", "_stored_E_L_force_PP")
    )
    assert all(np.all(np.isfinite(value)) for value in (actual_hf, actual_pp, actual_cross)), "stored moments must be finite"
    np.testing.assert_allclose(actual_energy, energy, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_hf, expected_hf, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_pp, expected_pp, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_cross, expected_cross, atol=atol, rtol=rtol, equal_nan=False)
