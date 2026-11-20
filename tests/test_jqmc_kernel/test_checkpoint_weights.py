"""Exact, hand-evaluated historical weights for checkpoint and XLA consumers."""

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

from jqmc import _checkpoint
from jqmc import jqmc_gfmc

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize(
    "window,expected",
    [
        (0, ((1, 1), (1, 1), (1, 1), (1, 1), (1, 1))),
        (1, ((2, 3), (0.5, 0), (4, 2), (1, 5))),
        (2, ((1, 0), (2, 0), (4, 10))),
        (3, ((4, 0), (2, 0))),
        (4, ((4, 0),)),
    ],
)
def test_checkpoint_and_xla_weights_equal_hand_evaluated_history(window, expected):
    """Both weight accumulators exactly match five hand-evaluated historical windows.

    Two walkers have raw weights [(2,3), (1/2,0), (4,2), (1,5), (8,1/4)].
    The weight for measurement n multiplies the preceding k rows, excluding n.
    Hence the final raw row never enters any returned weight; k=0 gives ones.
    Non-unit, fractional and zero
    entries expose wrong window lengths, row offsets and walker-axis mixing.
    These binary-exact fp64 values are the sampler/checkpoint storage dtype
    in both global precision modes. No wavefunction, positions, random draw,
    numerical tolerance or statistical confidence interval is involved.
    """
    weights = np.array([[2.0, 3.0], [0.5, 0.0], [4.0, 2.0], [1.0, 5.0], [8.0, 0.25]], dtype=np.float64)
    reference = np.asarray(expected, dtype=np.float64).reshape(5 - window, 2)
    checkpoint_weights = _checkpoint.compute_accumulated_weights(weights, window)
    xla_weights = jqmc_gfmc._compute_G_L(jnp.asarray(weights), window)
    np.testing.assert_array_equal(checkpoint_weights, reference)
    np.testing.assert_array_equal(xla_weights, reference)


@pytest.mark.parametrize("window", [5])
def test_no_retained_measurement_returns_empty_weight_columns(window):
    """A five-row history and five-row window retain zero rows for both walkers.

    Check the shape, not a vacuous equality between empty numeric arrays.
    The explicit parameter-set weights require no wavefunction or positions.
    """
    weights = np.array([[2.0, 3.0], [0.5, 0.0], [4.0, 2.0], [1.0, 5.0], [8.0, 0.25]], dtype=np.float64)
    checkpoint_weights = _checkpoint.compute_accumulated_weights(weights, window)
    xla_weights = jqmc_gfmc._compute_G_L(jnp.asarray(weights), window)
    np.testing.assert_array_equal(checkpoint_weights.shape, (0, 2))
    np.testing.assert_array_equal(xla_weights.shape, (0, 2))
