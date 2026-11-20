"""Finite, equal-weight bin statistics without production jackknife helpers."""

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


def equal_bin_statistics(values, bins):
    """Return the mean and SE from pairwise bin-mean differences.

    Every walker/bin cell has equal weight. For M cell means y,
    SE^2 = sum(i<j)(y_i-y_j)^2 / (M^2*(M-1)). This is an identity
    on these finite inputs, not a sampling-distribution acceptance band.
    """
    steps, walkers = values.shape
    assert steps > 0 and walkers > 0 and bins > 0 and steps % bins == 0
    length = steps // bins
    means = [
        math.fsum(float(values[row, walker]) for row in range(block * length, (block + 1) * length)) / length
        for block in range(bins)
        for walker in range(walkers)
    ]
    count = len(means)
    assert count > 1
    mean = math.fsum(means) / count
    pair_squares = math.fsum((left - right) ** 2 for i, left in enumerate(means) for right in means[i + 1 :])
    return mean, math.sqrt(pair_squares / (count * count * (count - 1)))
