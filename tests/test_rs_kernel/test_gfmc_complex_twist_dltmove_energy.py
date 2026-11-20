"""dltmove leg of ``test_gfmc_chain_modes.py::test_gfmc_complex_twist_rust_vs_jax_energy``.

The original node compares the native and XLA complex-twist GFMC_n / GFMC_t
production drivers for the complex J1/J2, DTN and J1/J2/J3 derivative caches,
but only with ``non_local_move="tmove"`` (its runner hard-codes the move).  This
module adds the same comparison with ``non_local_move="dltmove"`` -- the
determinant-locality approximation, where the non-local candidates carry the
rank-1 Jastrow ratio -- so each Jastrow family's candidate-ratio cache is
compared across backends under both non-local moves.  Same runner settings,
same seed, same route witnesses and the same
``_assert_deterministic_driver_parity`` oracle (full and mixed each at their own
backend-parity tier).  A power check confirms that on each fixture the XLA
dltmove run is not the tmove run.
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

from jqmc._setting import GFMC_MIN_BIN_BLOCKS, GFMC_MIN_COLLECT_STEPS, GFMC_MIN_WARMUP_STEPS
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_gfmc_chain_modes import (
    _STEPS_STAT,
    _assert_complex_driver_route_witnesses,
    _assert_deterministic_driver_parity,
    _build_ham,
    _install_ctx_spy,
)

pytestmark = [pytest.mark.with_chdir]


def _run_gfmc_move(driver, ham, seed, mode, non_local_move):
    """``test_gfmc_chain_modes._run_gfmc`` settings (fixed mesh, one walker) with a selectable move."""
    if mode == "jax":
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    else:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    try:
        common = dict(
            hamiltonian_data=ham,
            num_walkers=1,
            num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
            mcmc_seed=seed,
            alat=0.30,
            non_local_move=non_local_move,
            random_discretized_mesh=False,
            comput_position_deriv=False,
            comput_stress=False,
            use_swct=False,
        )
        if driver == "n":
            g = GFMC_n(**common, num_mcmc_per_measurement=6, E_scf=-1.0)
        else:
            g = GFMC_t(**common, tau=0.05)
        g.run(num_mcmc_steps=_STEPS_STAT)
        return g
    finally:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["n", "t"])
@pytest.mark.parametrize("jastrow_kind", ["j12", "dtn", "j123"])
def test_gfmc_complex_twist_rust_vs_jax_energy_dltmove(driver, jastrow_kind, monkeypatch):
    """Complex J1/J2, DTN and J1/J2/J3 caches preserve deterministic production parity under dltmove."""
    if jastrow_kind == "j12":
        ham = _build_ham("lithium_k025.h5")
    else:
        ham = rs_ham({"dtn": "twist_sphepgto_jdtn", "j123": "twist_sphepgto_j123"}[jastrow_kind])
    if jastrow_kind == "j123":
        assert ham.wavefunction_data.jastrow_data.jastrow_three_body_data is not None
    seed_a = 34356

    xla = _run_gfmc_move(driver, ham, seed_a, "jax", "dltmove")
    xla_tmove = _run_gfmc_move(driver, ham, seed_a, "jax", "tmove")
    # Power: on this fixture the dltmove walk is a different Hamiltonian from tmove.
    assert not np.array_equal(np.asarray(xla._stored_w_L), np.asarray(xla_tmove._stored_w_L)), (
        f"driver={driver} {jastrow_kind}: XLA dltmove and tmove stored identical weights -- the dltmove row is vacuous"
    )

    counts = _install_ctx_spy(monkeypatch)
    import jqmc_rs

    jqmc_rs.gfmc_route_counters_reset()
    rust = _run_gfmc_move(driver, ham, seed_a, "rust", "dltmove")

    key = f"run_gfmc_projection_{driver}"
    assert counts[key] >= _STEPS_STAT, f"native complex GFMC chunk did not engage in fast mode: {counts}"
    route_counts = list(jqmc_rs.gfmc_route_counters())
    _assert_complex_driver_route_witnesses(route_counts, jastrow_kind)
    _assert_deterministic_driver_parity(rust, xla, driver)
    args = (GFMC_MIN_WARMUP_STEPS, GFMC_MIN_BIN_BLOCKS)
    assert np.isfinite(rust.get_E(*args)[0]) and np.isfinite(xla.get_E(*args)[0])
