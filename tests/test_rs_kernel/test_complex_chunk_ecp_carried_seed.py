"""Complex-twist FAST chunk: the ECP non-local mesh state rides the sweep's carried base.

The native complex energy chunk (``run_energy_chunk(complex=True)``, FAST mode) builds
one ECP non-local mesh state per local energy. It seeds that state on the walker's
CARRIED determinant base -- the sweep's orbital tables and its running
Sherman-Morrison inverse, the inverse the kinetic term of the same e_L already
consumes -- instead of re-evaluating every orbital column, re-forming the geminal
and re-inverting it (``gfmc::mesh_fast_state_complex_from``,
IMPL-RS-CPLX-ECP-CARRIED-SEED-001). Both bases describe the same determinant at
the same positions; they differ only at the fp floor of the Sherman-Morrison carry.

The test-only hook ``JQMC_RS_ECP_COMPLEX_FRESH_SEED`` restores the fresh seed, so
the two routes run on identical walks: the sweep never reads the ECP state, so
walkers, keys and w_L are byte-identical, and every stored e_L agrees within the
carried-seed floor.
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

import jax

jax.config.update("jax_enable_x64", True)

import jqmc_rs  # noqa: E402

from jqmc._precision import precision_policy_snapshot  # noqa: E402
from jqmc._setting import (  # noqa: E402
    ATOL_CPLX_ECP_CARRIED_SEED,
    RTOL_CPLX_ECP_CARRIED_SEED,
    get_tolerance_xla_rust_consistency,
)
from tests.test_rs_kernel._rs_fixtures import rs_ham  # noqa: E402

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

# ``route_predicate_counters()`` (carried, fresh) ECP state seeds.
_SEED_INDICES = (20, 21)
_NW = 2
_STEPS = 7
_PATTERNS = (
    "twist_sphepgto_j12",
    "twist_sphepgto_j123",
    "twist_sphepgto_jdtn",
    "pgto_pw_twist_jnone",
    "bspline_twist_jnone",
)


def _run(ham, monkeypatch, fresh):
    from jqmc.jqmc_mcmc import MCMC

    monkeypatch.delenv("JQMC_FLAG_RUST_ON_CPU", raising=False)
    if fresh:
        monkeypatch.setenv("JQMC_RS_ECP_COMPLEX_FRESH_SEED", "1")
    else:
        monkeypatch.delenv("JQMC_RS_ECP_COMPLEX_FRESH_SEED", raising=False)
    mcmc = MCMC(
        hamiltonian_data=ham,
        Dt=2.0,
        mcmc_seed=34356,
        epsilon_AS=1.0e-6,
        num_walkers=_NW,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        random_discretized_mesh=True,
    )
    jqmc_rs.gfmc_route_counters_reset()
    mcmc.run(num_mcmc_steps=_STEPS)
    counters = list(jqmc_rs.route_predicate_counters())
    return mcmc, (counters[_SEED_INDICES[0]], counters[_SEED_INDICES[1]])


def _tolerance():
    """Carried-seed floor, raised to the active ao_eval zone envelope (mixed lane: fp32 AO values)."""
    atol, rtol = ATOL_CPLX_ECP_CARRIED_SEED, RTOL_CPLX_ECP_CARRIED_SEED
    if precision_policy_snapshot().get("ao_eval", "float64") != "float64":
        zone_atol, zone_rtol = get_tolerance_xla_rust_consistency("ao_eval")
        atol, rtol = max(atol, zone_atol), max(rtol, zone_rtol)
    return atol, rtol


@pytest.mark.parametrize("pattern", _PATTERNS)
def test_complex_chunk_ecp_carried_seed_matches_fresh(pattern, monkeypatch):
    """Carried-base ECP state == fresh-seeded state on the same walk, within the SM-carry floor.

    QE Li k=(1/4,1/4,1/4) ccECP fixtures (pGTO with J1/J2, J1/J2/J3, DTN; pGTO+PW and
    B-spline leaves without Jastrow). Two walkers, seven driver steps, seed 34356,
    random mesh, Rust backend on. The carried run seeds every chunk e_L from the
    sweep's base (counters (walkers x steps, 0)), the hooked run seeds every one fresh
    (counters (0, walkers x steps)). Walkers, keys and w_L are byte-identical; the
    stored complex e_L agree within ATOL/RTOL_CPLX_ECP_CARRIED_SEED.
    """
    ham = rs_ham(pattern)
    assert ham.coulomb_potential_data.ecp_flag, f"{pattern}: fixture must carry an ECP"
    carried, carried_counts = _run(ham, monkeypatch, fresh=False)
    fresh, fresh_counts = _run(ham, monkeypatch, fresh=True)
    seeds = _NW * _STEPS
    assert carried_counts == (seeds, 0), f"{pattern}: carried run seed counters {carried_counts}"
    assert fresh_counts == (0, seeds), f"{pattern}: fresh run seed counters {fresh_counts}"

    np.testing.assert_array_equal(carried._MCMC__latest_r_up_carts, fresh._MCMC__latest_r_up_carts)
    np.testing.assert_array_equal(carried._MCMC__latest_r_dn_carts, fresh._MCMC__latest_r_dn_carts)
    np.testing.assert_array_equal(carried._MCMC__jax_PRNG_key_list, fresh._MCMC__jax_PRNG_key_list)
    np.testing.assert_array_equal(np.asarray(carried.w_L), np.asarray(fresh.w_L))

    e_c, e_f = np.asarray(carried.e_L), np.asarray(fresh.e_L)
    assert e_c.shape == e_f.shape and np.all(np.isfinite(e_c)) and np.all(np.isfinite(e_f))
    atol, rtol = _tolerance()
    np.testing.assert_allclose(e_c, e_f, atol=atol, rtol=rtol, err_msg=f"{pattern}: carried vs fresh e_L")
