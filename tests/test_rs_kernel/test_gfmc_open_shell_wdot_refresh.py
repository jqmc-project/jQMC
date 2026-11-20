"""Native GFMC on an open-shell Slater determinant survives down-spin hops.

After every accepted hop the native GFMC state rank-one advances the
determinant contraction rows (``advance_det_wdot_cache`` /
``advance_det_wdot_cache_c`` in ``jqmc/rs/src/gfmc.rs``) and re-evaluates the
affine bases of both spins.  The moved electron's orbital column has that
spin's MO count only, so in an open shell (``N_up != N_dn``) it may be read
only on the moved spin's own branch.  Closed-shell fixtures cannot tell the
difference.  Two open-shell fixtures cover the real and the complex refresh:
the all-electron Li atom (2 up, 1 down) and the periodic ccECP oxygen triplet
(4 up, 2 down, complex Bloch orbitals).
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

import numpy as np
import pytest

import jqmc_rs
from jqmc._setting import GFMC_MIN_BIN_BLOCKS, GFMC_MIN_WARMUP_STEPS
from tests.test_rs_kernel.test_gfmc_chain_modes import _STEPS_STAT, _build_ham, _run_gfmc

pytestmark = [pytest.mark.with_chdir]

_SEED = 34356
# gfmc_route_counters indices: complex native state intervals, winv rank-one advances.
_CPLX_STATE_INTERVALS = 14
_WINV_ADVANCES = 57


@pytest.mark.light
@pytest.mark.parametrize(
    ("stem", "spins", "complex_path"),
    [
        pytest.param("Li_ae_ccpvdz_cart.h5", (2, 1), False, id="real-li-ae"),
        pytest.param("oxygen_tiling_sectors_cart.h5", (4, 2), True, id="complex-o-pbc-ecp"),
    ],
)
def test_gfmc_open_shell_down_hops_complete(stem, spins, complex_path):
    """Native GFMC_n on an N_up != N_dn determinant runs to finite energies through the winv refresh."""
    ham = _build_ham(stem)
    geminal = ham.wavefunction_data.geminal_data
    assert (geminal.num_electron_up, geminal.num_electron_dn) == spins
    jqmc_rs.gfmc_route_counters_reset()
    gfmc = _run_gfmc("n", ham, _SEED, "rust", _STEPS_STAT, random_mesh=True, num_walkers=1)

    counters = jqmc_rs.gfmc_route_counters()
    assert counters[_WINV_ADVANCES] > 0, "no hop advanced the determinant contraction rows"
    assert (counters[_CPLX_STATE_INTERVALS] > 0) == complex_path, (
        f"complex native state intervals: {counters[_CPLX_STATE_INTERVALS]}"
    )
    for name in ("_stored_w_L", "_stored_e_L", "_stored_e_L2"):
        assert np.all(np.isfinite(np.asarray(getattr(gfmc, name)))), name
    assert np.all(np.isfinite(np.asarray(gfmc.get_E(GFMC_MIN_WARMUP_STEPS, GFMC_MIN_BIN_BLOCKS))))
