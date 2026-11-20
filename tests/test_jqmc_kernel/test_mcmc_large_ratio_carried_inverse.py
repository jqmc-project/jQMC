"""Issue #127: the XLA MCMC kernels after an accepted move with a huge determinant ratio.

The rank-1 update ``X - (X u)(v^T X) / R`` evaluates the moved electron's new
inverse column (row for a down move) as ``X e_k (1 - (R - 1) / R)``; for
``|R| > 1/eps`` every digit is lost, the next proposal's ratio
``1 + v^T X u`` is wrong and a move with small exact acceptance is accepted (or,
through a zero AS factor, a certain move is rejected). The references are
independent of every production ratio / inverse-update / AS / proposal helper:
rebuilt-G ``slogdet`` (water) or the closed-form orbital (power model), and a
dense ``numpy.linalg.inv`` of the carried geminal after that geminal has been
checked against the rebuilt one.
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
from jqmc.determinant import _init_det_ratio_streaming_state, _sherman_morrison_ratio_and_inverse
from jqmc.jastrow_factor import init_Jastrow_ratio_state
from tests import _sm_large_ratio_cases as cases

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_G_ZONES = ("ao_eval", "det_eval")
_INVERSE_ZONES = ("det_eval", "det_ratio")


def _xla_move(ham, state, spin, normal, draw):
    """One production XLA move along x of electron 0 of ``spin`` with fixed draws."""
    up, down, key, inverse, matrix, det_state, jastrow_state = state
    two_spin = len(down) > 0
    function = mcmc._update_electron_positions if two_spin else mcmc._update_electron_positions_only_up_electron
    indices = iter([0 if spin == "up" else len(up), 0, 0, 0] if two_spin else [0, 0])
    proxy = SimpleNamespace(**vars(jax))
    random_proxy = SimpleNamespace(**vars(jax.random))
    random_proxy.randint = lambda *args, **kwargs: jnp.asarray(next(indices), dtype=jnp.int64)
    random_proxy.normal = lambda *args, **kwargs: jnp.asarray(normal, dtype=jnp.float64)
    random_proxy.uniform = lambda *args, **kwargs: jnp.asarray(draw, dtype=jnp.float64)
    proxy.random = random_proxy
    with patch.object(mcmc, "jax", proxy):
        result = function.__wrapped__(up, down, key, 1, ham, cases.DT, 0.0, inverse, matrix, det_state, jastrow_state)
        jax.block_until_ready(result)
    accepted, _rejected, up, down, key, inverse, matrix, det_state, jastrow_state = result
    return int(accepted), (up, down, key, inverse, matrix, det_state, jastrow_state)


def _case(model, spin):
    if model == "water":
        return cases.water_case(spin)
    return cases.power_case()


@pytest.mark.parametrize(
    "model,spin",
    [("water", "up"), ("water", "down"), ("power", "up")],
)
def test_xla_large_ratio_move_keeps_exact_decisions_and_carried_inverse(model, spin):
    """Two actual XLA moves, the first with |R| > 1e15: exact decisions, carried G and G^-1 stay exact.

    water: committed water ccECP cc-pVQZ, no Jastrow, PCG64 seed 2025 start, electron 0 of the moved spin
    28 -> 12 -> 13 bohr on +x from nucleus 0 (two-spin kernel, up and down moves). power: one electron in
    x^12 exp(-r^2) on Z = 7, x 0.01 -> 0.3 -> 0.02 (up-only kernel). Uniform draws 0.955 / 0.2927.
    After each move: the decision equals the exact acceptance decision, the position equals the numpy
    proposal (or the old one), the carried G equals the rebuilt G (rows scaled by the reference row
    norm; ao_eval/det_eval strict), and the carried inverse equals the dense inverse of that carried G
    (columns scaled by the reference column norm; det_eval/det_ratio strict).
    """
    ham, nuclei, charges, configurations = _case(model, spin)
    geminal = ham.wavefunction_data.geminal_data
    jastrow = ham.wavefunction_data.jastrow_data
    up0, dn0 = configurations[0]
    matrix0 = cases.reference_geminal(ham, configurations[0], model)
    state = (
        jnp.asarray(up0),
        jnp.asarray(dn0),
        jax.random.PRNGKey(0),
        jnp.asarray(np.linalg.inv(matrix0)),
        jnp.asarray(matrix0),
        _init_det_ratio_streaming_state(geminal, up0, dn0),
        init_Jastrow_ratio_state(jastrow, up0, dn0),
    )
    g_atol, g_rtol = get_tolerance_min(_G_ZONES, "strict")
    i_atol, i_rtol = get_tolerance_min(_INVERSE_ZONES, "strict")
    held = configurations[0]
    for label, target, draw in (
        ("move-1", configurations[1], cases.XLA_DRAWS[0]),
        ("move-2", configurations[2], cases.XLA_DRAWS[1]),
    ):
        normal = cases.injected_normal(held, target, spin, nuclei, charges)
        proposal = cases.proposed(held, normal, spin, nuclei, charges)
        probability = cases.exact_acceptance(ham, held, proposal, spin, nuclei, charges, model)
        if label == "move-1":
            assert cases.log_abs_ratio(ham, held, proposal, model) > np.log(cases.LARGE_RATIO_WITNESS), "precondition"
        expected_accept = int(draw < probability)
        accepted, state = _xla_move(ham, state, spin, normal, draw)
        assert accepted == expected_accept, f"{label}: decision {accepted} vs exact {expected_accept} (p = {probability:.3e})"
        held = proposal if expected_accept else held
        actual_positions = np.concatenate([np.asarray(state[0]), np.asarray(state[1])])
        np.testing.assert_allclose(actual_positions, np.concatenate(held), atol=g_atol, rtol=g_rtol, err_msg=label)
        actual_matrix = np.asarray(state[4])
        reference_matrix = cases.reference_geminal(ham, held, model)
        actual_scaled, reference_scaled = cases.row_scaled(actual_matrix, reference_matrix)
        np.testing.assert_allclose(actual_scaled, reference_scaled, atol=g_atol, rtol=g_rtol, err_msg=label)
        actual_inverse = np.asarray(state[3])
        dense_inverse = np.linalg.inv(actual_matrix)
        actual_scaled, reference_scaled = cases.column_scaled(actual_inverse, dense_inverse)
        np.testing.assert_allclose(actual_scaled, reference_scaled, atol=i_atol, rtol=i_rtol, err_msg=label)
    # Witness that the class is exercised: move 1 is accepted, move 2 rejected.
    assert np.array_equal(np.concatenate(held), np.concatenate(configurations[1]))


@pytest.mark.parametrize("spin", ["up", "down"])
def test_shared_sherman_morrison_update_is_exact_for_large_ratio(spin):
    """The shared XLA rank-1 helper (MCMC, GFMC_n, GFMC_t) gives inv(G_new) for a 28 -> 12 bohr move.

    The production callers pass the moved electron; for |R| ~ 1e15.5 the returned inverse equals the dense
    inverse of the rebuilt G_new (columns scaled; det_eval/det_ratio strict) and the ratio equals the
    rebuilt-G slogdet ratio, for an up move (u = e_0) and a down move (v = e_0).
    """
    ham, _nuclei, _charges, configurations = cases.water_case(spin)
    matrix0 = cases.reference_geminal(ham, configurations[0], "water")
    matrix1 = cases.reference_geminal(ham, configurations[1], "water")
    identity = np.eye(matrix0.shape[0])
    if spin == "up":
        u, v = identity[:, :1], (matrix1[0] - matrix0[0])[:, None]
    else:
        u, v = (matrix1[:, 0] - matrix0[:, 0])[:, None], identity[:, :1]
    ratio, inverse = _sherman_morrison_ratio_and_inverse(
        jnp.asarray(np.linalg.inv(matrix0)), jnp.asarray(u), jnp.asarray(v), moved_index=0, moved_is_up=spin == "up"
    )
    i_atol, i_rtol = get_tolerance_min(_INVERSE_ZONES, "strict")
    log_ratio = np.linalg.slogdet(matrix1)[1] - np.linalg.slogdet(matrix0)[1]
    assert log_ratio > np.log(cases.LARGE_RATIO_WITNESS), "precondition: the move is in the large-ratio class"
    np.testing.assert_allclose(np.log(abs(float(ratio))), log_ratio, atol=i_atol, rtol=i_rtol)
    actual_scaled, reference_scaled = cases.column_scaled(np.asarray(inverse), np.linalg.inv(matrix1))
    np.testing.assert_allclose(actual_scaled, reference_scaled, atol=i_atol, rtol=i_rtol)
