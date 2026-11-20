"""XLA-only entry for CPU/GPU Gaussian zero-radial tail detection."""

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

from jqmc._precision import get_tolerance
from jqmc.atomic_orbital import check_ao_eval_value_range, compute_AOs
from tests._ao_tail_reference import gaussian_tail_case, zero_coefficient_tangent_case

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


@pytest.mark.parametrize("family", ["cart_l60_a10", "sphe_l6_a1"])
def test_xla_zero_radial_tail_matches_closed_form(family):
    """The active XLA device keeps an accepted Gaussian tail finite and negligible."""
    aos, points, expected = gaussian_tail_case(family)
    check_ao_eval_value_range(aos)
    actual = np.asarray(compute_AOs(aos, points))
    assert np.all(np.isfinite(actual))
    if family == "sphe_l6_a1":
        actual = actual[np.asarray(aos.magnetic_quantum_numbers) == 0]
    atol, rtol = get_tolerance("ao_eval", "strict")
    assert np.all(np.abs(expected[:, 1:]) < atol) and expected[0, 0] > atol
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("family", ["cart_l1_a1", "sphe_l1_a1"])
def test_xla_zero_coefficient_tangent_matches_closed_form(family):
    """A zero radial factor keeps the closed-form coefficient tangent of the AO value."""
    aos, points, expected = zero_coefficient_tangent_case(family)
    check_ao_eval_value_range(aos)
    coefficients = jnp.asarray(aos.radial.coefficients)

    def values(c):
        return compute_AOs(aos.replace(radial=aos.radial.replace(coefficients=c)), points)

    primal, tangent = jax.jvp(values, (coefficients,), (jnp.ones_like(coefficients),))
    primal, tangent = np.asarray(primal), np.asarray(tangent)
    if family == "sphe_l1_a1":
        primal = primal[np.asarray(aos.magnetic_quantum_numbers) == 0]
        tangent = tangent[np.asarray(aos.magnetic_quantum_numbers) == 0]
    assert np.all(primal == 0)
    atol, rtol = get_tolerance("ao_eval", "strict")
    assert np.all(expected > atol)
    np.testing.assert_allclose(tangent, expected, atol=atol, rtol=rtol, equal_nan=False)
