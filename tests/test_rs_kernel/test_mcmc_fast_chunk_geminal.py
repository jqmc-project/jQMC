"""Issue231: the fast native chain carries the geminal of its current orbitals.

The old seed detector covers the per-step optimization route. Energy-only fast
chunks enabled a different AO evaluator and retained a seed-versus-cache offset
under incremental row updates. Check the public driver and its actual returned
state, before an energy average or an AO-sized absolute tolerance can hide it.
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

import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import get_dtype_np, precision_policy_snapshot
from jqmc._setting import ATOL_REDUCTION_ORDER, RTOL_BIT_EXACT, RTOL_REDUCTION_ORDER
from jqmc.determinant import SD_data
from jqmc.jqmc_mcmc import MCMC
from tests.test_jqmc_kernel.test_mcmc_gaussian_energy_reference import _gaussian_hamiltonian
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


def _capture_fast_chunk(monkeypatch, ham, steps=503):
    original = jqmc_rs.build_obs_ctx
    captured = []

    class Context:
        def __init__(self, ctx):
            self.ctx = ctx

        def __getattr__(self, name):
            return getattr(self.ctx, name)

        def run_energy_chunk_batch(self, *args, **kwargs):
            assert kwargs["native_rng"] is True
            result = self.ctx.run_energy_chunk_batch(*args, **kwargs)
            captured.append({key: np.array(value, copy=True) for key, value in result.items()})
            return result

    def build(*args, precision_policy, **kwargs):
        assert precision_policy == precision_policy_snapshot()
        return Context(original(*args, precision_policy=precision_policy, **kwargs))

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", build)
    driver = MCMC(
        hamiltonian_data=ham,
        mcmc_seed=34456,
        num_walkers=1,
        num_mcmc_per_measurement=40,
        Dt=2.0,
        epsilon_AS=0.0,
    )
    driver.run(num_mcmc_steps=steps)
    assert captured, "the production fast native energy chunk did not execute"
    assert np.all(np.isfinite(np.asarray(driver.e_L)))
    return captured


@pytest.mark.parametrize("pattern", ["mol_ecp_cart_jnone", "mol_cart_sd_jnone", "mol_cart_agp_jnone", "mol_ecp_sphe_jnone"])
def test_fast_chunk_carried_geminal_matches_current_orbitals(pattern, monkeypatch):
    """Independent NumPy contractions of returned AO/MO caches define G, row by row.

    No production geminal, ratio, SM update or paired-cache helper supplies the
    reference. SD's unpaired columns and dense AGP's lambda are included. This
    is an fp64 same-evaluation algebra identity within each mode. The existing
    reduction-order tolerance applies; the generic zone budget would hide an
    fp32-sized offset. Scaling each row prevents large rows hiding a small row.
    """
    ham = rs_ham(pattern)
    geminal = ham.wavefunction_data.geminal_data
    nd = geminal.num_electron_dn
    captured = _capture_fast_chunk(monkeypatch, ham)
    assert len(captured) >= 2, "the same run must cross a native chunk boundary"
    for result in captured:
        assert np.all(result["accepted"] > 0)
        for up, dn, actual in zip(result["ao_up"], result["ao_dn"], result["geminal"], strict=True):
            if isinstance(geminal, SD_data):
                paired = up[:nd].T @ dn
                reference = np.concatenate((paired, up[nd:].T), axis=1)
            else:
                lam = np.asarray(geminal._lambda_matrix_jnp)
                norb_dn = dn.shape[0]
                paired = (up.T @ lam[:, :norb_dn]) @ dn
                reference = np.concatenate((paired, up.T @ lam[:, norb_dn:]), axis=1)
            assert reference.shape == actual.shape and paired.shape[1] == nd
            scale = np.max(np.abs(reference), axis=1, keepdims=True)
            assert np.all(scale > 0)
            # Each row has unit reference infinity norm after scaling. Its
            # central allclose budget is therefore atol + rtol. Compare the
            # row-relative error with zero, so tiny entries do not acquire a
            # separate entry-relative requirement inside that row.
            row_error = np.max(np.abs(actual - reference), axis=1) / scale[:, 0]
            np.testing.assert_allclose(
                row_error,
                np.zeros_like(row_error),
                atol=ATOL_REDUCTION_ORDER + RTOL_REDUCTION_ORDER,
                rtol=RTOL_BIT_EXACT,
            )


def test_fast_chunk_preserves_the_ao_value_precision(monkeypatch):
    """A one-orbital MO is its AO: mixed cached values must be representable in fp32.

    Normalized diffuse s GTO (O2 report's smallest exponent) on a two-electron
    helium model; coefficients are identity. This detects the fast evaluator
    bypassing the active AO zone even when a relaxed AO parity band would pass.
    """
    ham = _gaussian_hamiltonian(charge=2, exponent=0.083598)
    dtype = get_dtype_np("ao_eval")
    for result in _capture_fast_chunk(monkeypatch, ham, steps=11):
        for name in ("ao_up", "ao_dn"):
            values = result[name]
            np.testing.assert_array_equal(values, values.astype(dtype).astype(values.dtype))
