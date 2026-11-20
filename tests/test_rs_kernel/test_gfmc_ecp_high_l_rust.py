"""Deterministic Rust-vs-XLA GFMC_t / GFMC_n driver parity with a live l >= 1 ECP.

Cl2 ccECP (non-zero non-local s and p on both atoms), J1 + J2, one walker (the
multi-walker GFMC_t Rust/XLA divergence is a separate open defect), T-move and DLT-move,
random (production) mesh, energy rows and force rows (position derivatives + SWCT).
The native interval must run; the walk (threefry keys, final walkers) is byte-identical
and ``_stored_w_L`` / ``_stored_e_L`` / ``_stored_e_L2`` / HF and Pulay force buffers /
public energy and force agree through the shared
``test_gfmc_chain_modes._assert_deterministic_driver_parity``.  Stored forces must be
finite on both backends (the parity comparator treats matching NaN as equal).
Non-vacuity: the l >= 1 non-local contribution at the final walker exceeds the e_L
envelope. The production non-local ECP energy at the native driver's final walker must
equal the raw-ECP-array reference (nearest-nucleus convention, scipy Legendre), so a slip
in the shared non-local term selection fails this row.
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

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import GFMC_MIN_BIN_BLOCKS, GFMC_MIN_COLLECT_STEPS, GFMC_MIN_STEPS, GFMC_MIN_WARMUP_STEPS
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from tests._ecp_high_l import (
    assert_fixture_has_live_high_l_channel,
    assert_production_nonlocal_matches_raw_arrays,
    high_l_hamiltonian,
    high_l_nonlocal_contribution,
)
from tests.test_rs_kernel.test_gfmc_chain_modes import _PARITY_ZONES, _assert_deterministic_driver_parity, _install_ctx_spy


def _run(driver, ham, mode, non_local_move, pos_deriv):
    previous = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
    if mode == "jax":
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    else:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    try:
        common = dict(
            hamiltonian_data=ham,
            num_walkers=1,
            num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
            mcmc_seed=34356,
            alat=0.30,
            non_local_move=non_local_move,
            random_discretized_mesh=True,
            comput_position_deriv=pos_deriv,
            use_swct=pos_deriv,
        )
        g = GFMC_n(**common, num_mcmc_per_measurement=6, E_scf=-30.0) if driver == "n" else GFMC_t(**common, tau=0.05)
        g.run(num_mcmc_steps=GFMC_MIN_STEPS)
        return g
    finally:
        if previous is None:
            os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
        else:
            os.environ["JQMC_FLAG_RUST_ON_CPU"] = previous


@pytest.mark.heavy
@pytest.mark.parametrize("non_local_move", ["tmove", "dltmove"])
@pytest.mark.parametrize("driver", ["t", "n"])
@pytest.mark.parametrize("pos_deriv", [False, True], ids=["energy", "force"])
def test_gfmc_rust_vs_xla_high_l_ecp_values(driver, non_local_move, pos_deriv, monkeypatch):
    """Same seed, same walk: Rust == XLA GFMC values on the live p-channel Cl2 ccECP."""
    ham = high_l_hamiltonian()
    assert_fixture_has_live_high_l_channel(ham)
    xla = _run(driver, ham, "jax", non_local_move, pos_deriv)
    counts = _install_ctx_spy(monkeypatch)
    rust = _run(driver, ham, "rust", non_local_move, pos_deriv)
    assert counts[f"run_gfmc_projection_{driver}"] >= GFMC_MIN_STEPS, f"native interval did not run: {counts}"
    label = f"GFMC_{driver}/{non_local_move}"
    for run, side in ((rust, "Rust"), (xla, "XLA")):
        # The shared parity comparator uses assert_allclose's default equal_nan=True, so
        # every compared operand is required finite here first.
        names = ("_stored_w_L", "_stored_e_L", "_stored_e_L2")
        for name in names + (("_stored_force_HF", "_stored_force_PP") if pos_deriv else ()):
            value = np.asarray(getattr(run, name))
            assert value.size and np.all(np.isfinite(value)), f"{label} {side}: {name} empty or not finite"
        energy = np.asarray(run.get_E(GFMC_MIN_WARMUP_STEPS, GFMC_MIN_BIN_BLOCKS))
        assert np.all(np.isfinite(energy)), f"{label} {side}: public energy not finite"
    atol, rtol = get_tolerance_xla_rust_consistency(_PARITY_ZONES, level="medium")
    assert_production_nonlocal_matches_raw_arrays(ham, rust._latest_r_up_carts, rust._latest_r_dn_carts, atol, rtol, label)
    _assert_deterministic_driver_parity(rust, xla, driver, position_deriv=pos_deriv, random_mesh=True)
    delta = high_l_nonlocal_contribution(ham, rust._latest_r_up_carts, rust._latest_r_dn_carts)
    envelope = atol + rtol * np.abs(np.asarray(rust._stored_e_L)[-1])
    assert np.all(np.abs(delta) > envelope), f"{label}: l>=1 contribution {delta} within the e_L envelope {envelope}"
