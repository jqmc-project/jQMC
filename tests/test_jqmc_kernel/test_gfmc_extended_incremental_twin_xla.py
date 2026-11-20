"""Same-backend production driver twins for existing composite and twist inputs."""

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

from jqmc._setting import ATOL_GFMC_INCREMENTAL_TWIN, RTOL_GFMC_INCREMENTAL_TWIN
from tests.test_jqmc_kernel.test_gfmc_incremental_twin_xla import _run
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]


@pytest.mark.parametrize("pattern", ["pbc_composite_jnone", "twist_sphepgto_j12"])
@pytest.mark.parametrize("driver", ["t", "n"])
def test_xla_composite_twist_incremental_twin(pattern, driver, monkeypatch):
    """Actual periodic composite and twist drivers preserve incremental/full twins.

    Both routes use the existing seed, two walkers and three stored steps.
    Walks and RNG agree exactly; complex stored energies and weights agree at
    the existing twin budget. Live carried updates are required only on the
    incremental route, independently of value parity.
    """
    import jqmc.jqmc_gfmc as gfmc_mod

    ham = rs_ham(pattern)
    reference = _run(driver, ham, full_recompute=True, monkeypatch=monkeypatch)
    witness_before = gfmc_mod._GFMC_POT_TRACE_WITNESS
    actual = _run(driver, ham, full_recompute=False, monkeypatch=monkeypatch)
    assert gfmc_mod._GFMC_POT_TRACE_WITNESS > witness_before
    assert np.all(np.asarray(actual._latest_gfmc_pot_updates) > 0)
    assert np.all(np.asarray(reference._latest_gfmc_pot_updates) == 0)
    for name in ("_latest_r_up_carts", "_latest_r_dn_carts", "_jax_PRNG_key_list"):
        np.testing.assert_array_equal(np.asarray(getattr(actual, name)), np.asarray(getattr(reference, name)))
    for name in ("_stored_w_L", "_stored_e_L", "_stored_e_L2"):
        actual_values = np.asarray(getattr(actual, name))
        reference_values = np.asarray(getattr(reference, name))
        assert actual_values.size > 0
        assert np.all(np.isfinite(actual_values)) and np.all(np.isfinite(reference_values))
        np.testing.assert_allclose(
            actual_values,
            reference_values,
            atol=ATOL_GFMC_INCREMENTAL_TWIN,
            rtol=RTOL_GFMC_INCREMENTAL_TWIN,
            equal_nan=False,
        )
