"""The generic projection-scoped J1/J2 cache must not freeze DTN rows.

``{J1: None, J2: pade, DTN: Some}`` is absent from the canonical fixture matrix
but is ordinary production input: the Jastrow legs are independent.  It pins
the important split-state contract after removal of the old molecular
Pade/AVX special route: the generic J12 table carries only J1/J2, while the DTN
table advances independently and the consumer combines the two current states.
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

import pytest

from tests.test_rs_kernel._rs_fixtures import build_pade_j2_dtn_no_j1_ham
from tests.test_rs_kernel.test_gfmc_chain_modes import (
    _STEPS_STAT,
    _assert_deterministic_driver_parity,
    _run_gfmc,
)

pytestmark = [pytest.mark.with_chdir]


@pytest.mark.light
@pytest.mark.parametrize("driver", ["n", "t"])
def test_j12_grad_cache_keeps_dtn_current_across_hops(driver):
    """Rust and XLA must agree for {no J1, pade J2, DTN} over a multi-hop run.

    The mesh rotation is disabled so both backends walk the same threefry
    stream and make the same projection decisions; any divergence is then the
    cached derivative tables, not the walk. A single sub-step would not prove
    state advance, so this runs the full step budget.
    """
    ham = build_pade_j2_dtn_no_j1_ham()
    seed = 34356
    xla = _run_gfmc(driver, ham, seed, "jax", _STEPS_STAT, random_mesh=False, num_walkers=1)
    rust = _run_gfmc(driver, ham, seed, "rust", _STEPS_STAT, random_mesh=False, num_walkers=1)
    _assert_deterministic_driver_parity(rust, xla, driver, periodic=False)
