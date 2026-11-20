"""Exact numerical-policy resolution and stream batch boundary contracts."""

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

import math

import pytest

import jqmc._setting as setting


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize(
    "policy", ["lm_centered_overlap", "lm_active_diagonal", "lm_correlation_pivot", "asr_model", "force_standard_error"]
)
@pytest.mark.parametrize(
    "dtype,epsilon_exponent", [("float32", -23), ("float64", -52), ("complex64", -23), ("complex128", -52)]
)
@pytest.mark.parametrize("scale", [0.0, 0.25, 8.0])
def test_optimizer_roundoff_resolution_matches_binary_formula(policy, dtype, epsilon_exponent, scale):
    """Every registered policy resolves to its multiplier times real-component epsilon power and scale exactly.

    The multiplier and exponent are read from the production table, so the test
    checks the resolution without restating which values a policy uses. The
    reference epsilon comes from the IEEE
    binary32/binary64 exponents, not the finfo helper. Binary scales make an
    exact comparison appropriate. This checks the resolution, not which
    multiplier or exponent a policy should use.
    """
    multiplier, exponent = setting._OPTIMIZER_ROUNDOFF_POLICIES[policy]
    expected = math.ldexp(multiplier * scale, exponent * epsilon_exponent)
    actual = setting.get_optimizer_roundoff_tolerance(policy, dtype, scale=scale)
    assert type(actual) is float
    assert actual == expected
    assert setting.get_optimizer_roundoff_tolerance(policy, dtype) == math.ldexp(multiplier, exponent * epsilon_exponent)


def test_optimizer_roundoff_unknown_policy_is_rejected():
    """An unregistered optimizer policy cannot silently obtain a default threshold."""
    with pytest.raises(KeyError, match="Unknown optimizer round-off policy"):
        setting.get_optimizer_roundoff_tolerance("unregistered-policy", "float64")


@pytest.mark.parametrize(
    "dtype,normal_exponent", [("float32", -126), ("float64", -1022), ("complex64", -126), ("complex128", -1022)]
)
def test_optimizer_breakdown_uses_smallest_real_normal(dtype, normal_exponent):
    """CG breakdown uses the real component's smallest normal, not epsilon or its smallest subnormal."""
    actual = setting.get_optimizer_breakdown_threshold(dtype)
    assert type(actual) is float
    assert actual == math.ldexp(1.0, normal_exponent)


@pytest.mark.parametrize(
    "bytes_per_element,n_total,budget_bytes,min_batch,expected",
    [
        (8, 1000, 800, 16, 64),
        (8, 1000, 1024, 16, 128),
        (8, 1000, 8, 16, 16),
        (8, 7, 8, 16, 7),
        (8, 100, 800, 16, 100),
        (8, 100, 1600, 16, 100),
        (0, 1000, 100, 16, 64),
        (8, 0, 100, 16, 1),
        (8, -1, 100, 16, 1),
        (8, 100, 0, 16, 16),
    ],
)
def test_stream_batch_budget_boundaries(bytes_per_element, n_total, budget_bytes, min_batch, expected):
    """Batch width respects binary flooring, occupancy floor, total cap, zero-byte input and empty populations.

    The occupancy floor may exceed the nominal budget. This arithmetic
    selection contract does not assert a measured peak-memory upper bound.
    """
    assert setting.batch_size_for_budget(bytes_per_element, n_total, budget_bytes, min_batch) == expected
