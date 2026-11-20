"""Issues #131 / #134: the AS regularization factor in the XLA MCMC kernels is evaluated robustly.

* #131: ``compute_AS_regularization_factor_fast_update`` formed ``S * F`` in linear
  scale; a geminal row below ~1e-154 gave ``0 * inf`` and ``R_AS = 0``, after which
  every proposal from that state was rejected (``0 / 0`` at ``epsilon_AS = 0``,
  ``eps / 0`` at ``epsilon_AS > 0``) and ``w_L`` was NaN or 0.
* #134: the proposed geminal was formed as ``G + (G' - G)``, which is exactly 0 in
  the moved row / column once ``|G'| < eps |G|``; ``R_AS' = 0`` made the AS ratio
  ``inf`` and a move with exact acceptance below 1e-16 was accepted.

References (see ``tests/_as_factor_cases.py``) are independent of the production
ratio / inverse-update / AS / proposal helpers.
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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import jqmc_mcmc as mcmc
from jqmc._precision import get_tolerance_min
from jqmc.determinant import _init_det_ratio_streaming_state, compute_AS_regularization_factor_fast_update
from jqmc.jastrow_factor import init_Jastrow_ratio_state
from tests import _as_factor_cases as cases

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_AS_ZONES = ("det_eval", "det_ratio")


def _xla_move(ham, before, inverse, matrix, spin, normal, draw, epsilon):
    """One production XLA kernel move with injected electron / axis / normal / uniform draws."""
    up, dn = before
    geminal = ham.wavefunction_data.geminal_data
    two_spin = len(dn) > 0
    function = mcmc._update_electron_positions if two_spin else mcmc._update_electron_positions_only_up_electron
    indices = iter([0 if spin == "up" else len(up), 0, 0, 0] if two_spin else [0, 0])
    proxy = SimpleNamespace(**vars(jax))
    random_proxy = SimpleNamespace(**vars(jax.random))
    random_proxy.randint = lambda *args, **kwargs: jnp.asarray(next(indices), dtype=jnp.int64)
    random_proxy.normal = lambda *args, **kwargs: jnp.asarray(normal, dtype=jnp.float64)
    random_proxy.uniform = lambda *args, **kwargs: jnp.asarray(draw, dtype=jnp.float64)
    proxy.random = random_proxy
    with patch.object(mcmc, "jax", proxy):
        result = function.__wrapped__(
            jnp.asarray(up),
            jnp.asarray(dn),
            jax.random.PRNGKey(0),
            1,
            ham,
            cases.DT,
            epsilon,
            jnp.asarray(inverse),
            jnp.asarray(matrix),
            _init_det_ratio_streaming_state(geminal, up, dn),
            init_Jastrow_ratio_state(ham.wavefunction_data.jastrow_data, up, dn),
        )
        jax.block_until_ready(result)
    return int(result[0])


@pytest.mark.parametrize(
    "name", ["power-underflow", "power-cancel", "h2-up-12", "h2-up-17", "h2-down-12", "h2-down-17", "water-up-30"]
)
@pytest.mark.parametrize("epsilon", [0.0, 0.1])
def test_xla_as_factor_single_move_decisions_are_exact(name, epsilon):
    """XLA kernel decisions with an underflowing current row or a cancelling proposed row are exact.

    Witnesses (no Jastrow, Dt = 2): the x^12 one-electron model x = 1e-14 -> 0.3 (|G| ~ 4e-171, exact
    acceptance 1) and x = 1.0 -> 0.05 (exact acceptance ~4e-31); H2 cc-pVDZ electron 0 (up and
    down) moved +12 and +17 bohr (log10|R| ~ -8.6 / -16.3); water ccECP cc-pVQZ up electron 0 moved +30 bohr.
    From an exact numpy inverse (the closed-form 1x1 G for the x^12 model), the production kernel with
    injected draws 0.01 / 0.5 / 0.99 must accept exactly when the draw is below the exact AS-guided
    acceptance (rebuilt-G slogdet or closed form, numpy Kato transition density, row-rescaled log-domain
    R_AS reference; R_AS = 1 exactly for every 1x1 G), at epsilon_AS = 0 and 0.1.
    """
    ham, nuclei, charges, model, spin, before, normal = cases.case(name)
    matrix = cases.reference_geminal(ham, before, model)
    with np.errstate(over="ignore"):
        inverse = np.linalg.inv(matrix)
    after = cases.proposed(before, normal, spin, nuclei, charges)
    probability = cases.exact_acceptance(ham, before, after, spin, nuclei, charges, model, epsilon)
    if name == "power-underflow":
        assert np.max(np.abs(matrix)) < cases.UNDERFLOW_ROW_NORM, "precondition: underflowing current row"
    else:
        ratio = cases.ln_abs_psi(ham, after, model) - cases.ln_abs_psi(ham, before, model)
        assert ratio < np.log(cases.NATIVE_CANCELLATION_RATIO), "precondition: tiny determinant ratio"
    decisions = [_xla_move(ham, before, inverse, matrix, spin, normal, draw, epsilon) for draw in cases.XLA_DRAWS]
    expected = [int(draw < probability) for draw in cases.XLA_DRAWS]
    assert decisions == expected, (name, epsilon, probability, decisions)


@pytest.mark.parametrize("scale", [1.0e-160, 1.0e-170])
@pytest.mark.parametrize("spin", ["up", "down"])
def test_xla_as_factor_is_invariant_under_a_tiny_row_or_column(scale, spin):
    """``compute_AS_regularization_factor_fast_update`` is robust to a row / column below sqrt(tiny).

    Water ccECP cc-pVQZ, PCG64(2025) start. Row 0 (an up electron) or column 0 (a down electron) of G is
    scaled by c and the matching column / row of the exact numpy inverse by 1/c. Once c^2 is below eps
    relative to the other contributions, S ~ c^2 and F ~ c^-2 and R_AS = (S F)^(-3/8) no longer depends on
    c. For c in {1e-160, 1e-170} (squares underflow, inverse squares overflow) the production factor must
    equal the row-rescaled log-domain reference of the scaled G and the production factor at
    c = 1e-60 (still in the linear range), at the det_eval / det_ratio strict pair. For the x^12 model at
    x = 1e-14 (1x1 G ~ 4e-171) the factor must be 1.
    """
    ham, _, _, before = cases.water_model()
    matrix = cases.reference_geminal(ham, before, "water")
    inverse = np.linalg.inv(matrix)

    def scaled_pair(c):
        scaled, scaled_inverse = matrix.copy(), inverse.copy()
        if spin == "up":
            scaled[0, :] *= c
            scaled_inverse[:, 0] /= c
        else:
            scaled[:, 0] *= c
            scaled_inverse[0, :] /= c
        return scaled, scaled_inverse

    atol, rtol = get_tolerance_min(_AS_ZONES, "strict")
    scaled, scaled_inverse = scaled_pair(scale)
    reference = np.exp(cases.reference_ln_as_factor(scaled))
    linear = float(compute_AS_regularization_factor_fast_update(*map(jnp.asarray, scaled_pair(cases.LINEAR_RANGE_ROW_SCALE))))
    actual = float(compute_AS_regularization_factor_fast_update(jnp.asarray(scaled), jnp.asarray(scaled_inverse)))
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual, linear, atol=atol, rtol=rtol)
    power_ham, _, _ = cases.power_model()
    tiny = cases.reference_geminal(power_ham, cases.power_configuration(1.0e-14), "power")
    with np.errstate(over="ignore"):
        tiny_inverse = np.linalg.inv(tiny)
    one = float(compute_AS_regularization_factor_fast_update(jnp.asarray(tiny), jnp.asarray(tiny_inverse)))
    np.testing.assert_allclose(one, 1.0, atol=atol, rtol=rtol)


@pytest.mark.parametrize("rust", ["0", "1"])
def test_driver_walker_leaves_an_underflowing_state_at_zero_epsilon(monkeypatch, rust):
    """With epsilon_AS = 0 the MCMC driver moves a walker started at a tiny geminal and w_L == 1.

    One walker of the x^12 model started at x = 1e-14 (|G| ~ 4e-171 in full precision; exactly 0 under the
    float32 ao_eval of mixed precision), mcmc_seed 34, Dt 2, ten proposals per measurement, 20 measurements,
    XLA (JQMC_FLAG_RUST_ON_CPU=0) and native (=1) drivers. psi is its own guiding function, so the AS
    factor must not enter: the walker must accept at least one move and leave the start, and every stored
    w_L must be exactly 1. (The epsilon_AS > 0 decisions from such a state are covered by the single-move
    tests with an injected float64 state.)
    """
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", rust)
    ham, _, _ = cases.power_model()
    driver = mcmc.MCMC(
        hamiltonian_data=ham,
        mcmc_seed=34,
        num_walkers=1,
        num_mcmc_per_measurement=10,
        Dt=cases.DT,
        epsilon_AS=0.0,
        random_discretized_mesh=False,
    )
    start = np.array([[[1.0e-14, 0.3, 0.4]]])
    driver._MCMC__latest_r_up_carts = jnp.asarray(start)
    driver._MCMC__latest_r_dn_carts = jnp.asarray(np.zeros((1, 0, 3)))
    driver.run(num_mcmc_steps=20)
    assert driver._MCMC__accepted_moves > 0, "the walker is frozen"
    assert not np.array_equal(np.asarray(driver._MCMC__latest_r_up_carts), start), "the walker did not move"
    weights = np.asarray(driver.w_L)
    np.testing.assert_array_equal(weights, np.ones_like(weights))
