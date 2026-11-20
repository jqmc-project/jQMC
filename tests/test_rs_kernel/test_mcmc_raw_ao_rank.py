"""Public native MCMC preserves the complete rank of MO-less AGP orbitals.

The rank oracle is the explicitly constructed canonical basis (7 plane waves,
or 4 GTO plus 4 B-spline AOs). The independent Python/XLA driver supplies the
same-seed energy reference; no production/debug shared native evaluator is used.
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
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.jqmc_mcmc import MCMC
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian

pytestmark = pytest.mark.with_chdir

_ENERGY_ZONES = ("local_energy", "coulomb", "det_ratio", "ao_eval")


@pytest.mark.heavy
@pytest.mark.parametrize(
    "pattern_id,expected_rank",
    [
        pytest.param("WF-PATTERN-007", 7, id="raw_pw"),
        pytest.param("WF-PATTERN-013", 8, id="raw_pgto_bspline"),
    ],
)
def test_native_mcmc_raw_ao_agp_matches_xla(pattern_id, expected_rank, monkeypatch):
    """The public native sweep uses all AO columns and matches XLA in each precision."""
    ham = build_canonical_hamiltonian(pattern_id)
    geminal = ham.wavefunction_data.geminal_data
    for orbital in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
        assert not hasattr(orbital, "mo_coefficients")
        assert orbital.num_ao == expected_rank
    assert geminal.lambda_matrix.shape[0] == expected_rank

    def run():
        driver = MCMC(
            hamiltonian_data=ham,
            num_walkers=2,
            mcmc_seed=11,
            random_discretized_mesh=True,
            Dt=1.0,
            epsilon_AS=1e-6,
            num_mcmc_per_measurement=2,
        )
        driver.run(num_mcmc_steps=14)
        return np.asarray(driver.e_L).copy()

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    reference = run()
    assert reference.size > 2 and np.all(np.isfinite(reference))
    assert np.ptp(reference) > 0

    original_build = jqmc_rs.build_obs_ctx
    calls = []

    class CountedContext:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            value = getattr(self.inner, name)
            if name not in ("run_energy_chunk_batch", "mcmc_sweep_batch"):
                return value

            def counted(*args, **kwargs):
                calls.append(name)
                return value(*args, **kwargs)

            return counted

    def build(*args, **kwargs):
        return CountedContext(original_build(*args, **kwargs))

    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", build)
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    actual = run()
    assert calls, "the native MCMC production route was not executed"
    assert actual.shape == reference.shape and np.all(np.isfinite(actual))
    atol, rtol = get_tolerance_xla_rust_consistency(_ENERGY_ZONES, level="medium")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
