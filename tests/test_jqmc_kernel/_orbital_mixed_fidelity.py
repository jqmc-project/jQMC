"""Block comparisons for the orbital/elementary portion of the mixed AD migration."""

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

import numpy as np
import pytest

from jqmc._setting import get_tolerance_mixed_fidelity
from tests._precision_fidelity import mixed_fidelity_blocks, out_of_budget_mixed_block


def _assert_mixed_derivative_block(mixed_block, full_block, atol, rtol, label):
    """Compare one homogeneous logical derivative block with its registered budget."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max(), label


def assert_mixed_orbital_derivative_fidelity(mixed, full, label):
    """Require finite matching blocks and prove the same comparison rejects an excess error."""
    blocks = mixed_fidelity_blocks(mixed, full)
    atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
    worst = 0.0
    for mixed_block, full_block in blocks:
        _assert_mixed_derivative_block(mixed_block, full_block, atol, rtol, label)
        worst = max(worst, float(np.max(np.abs(mixed_block - full_block))))
    faulty = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
    with pytest.raises(AssertionError):
        _assert_mixed_derivative_block(faulty, blocks[0][1], atol, rtol, label)
    return worst


def assert_mixed_parameter_derivative_fidelity(mixed, full, label):
    """Require finite matching blocks and prove the same comparison rejects an excess error."""
    blocks = mixed_fidelity_blocks(mixed, full)
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    worst = 0.0
    for mixed_block, full_block in blocks:
        _assert_mixed_derivative_block(mixed_block, full_block, atol, rtol, label)
        worst = max(worst, float(np.max(np.abs(mixed_block - full_block))))
    faulty = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
    with pytest.raises(AssertionError):
        _assert_mixed_derivative_block(faulty, blocks[0][1], atol, rtol, label)
    return worst
