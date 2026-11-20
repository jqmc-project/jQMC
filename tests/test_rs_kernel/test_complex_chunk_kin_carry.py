"""Complex-twist FAST chunk: carried kinetic tables and carried determinant base.

The native complex energy chunk (``run_energy_chunk(complex=True)``, FAST mode) carries
two kinds of state across the steps of one chunk (IMPL-RS-CPLX-CHUNK-KIN-CARRY-001):

* the kinetic derivative tables (``gfmc::ComplexChunkKinCarry``: the complex GFMC
  e_L's det / J1-J2 / DTN / J3 derivative caches), seeded once per chunk, advanced by
  replaying every accepted sweep move in order and contracted per e_L, instead of the
  fresh per-e_L ``compute_kinetic_withj_complex_with_inv``;
* the determinant base (geminal, Sherman-Morrison inverse and ``DetRatioStateC``),
  built once per chunk and threaded through the sweeps, instead of rebuilt every step.

Two test-only hooks select the uncarried routes: ``JQMC_RS_KIN_COMPLEX_FRESH`` (fresh
kinetic every e_L) and ``JQMC_RS_DET_COMPLEX_FRESH_BASE`` (base rebuilt every step).

The A/B structure:

* kinetic carry alone: the sweep never reads the kinetic tables, so walkers, keys and
  w_L are byte-identical and every e_L agrees within ATOL/RTOL_CPLX_KIN_CARRY;
* determinant-base carry: the walk itself reads the carried inverse, but a Metropolis
  decision flips only if the carried and rebuilt acceptance ratios straddle the
  uniform draw, an fp-floor event. The keys never depend on acceptance; on these
  fixtures the positions stay byte-identical (asserted, so a flip would be reported
  rather than hidden), and w_L and e_L agree within ATOL/RTOL_CPLX_DET_BASE_CARRY;
* both carries against both hooks (the pre-carry route): the same walk, w_L within the
  base floor and e_L within the sum of the two floors.

The route-predicate counters 22-28 pin the routes taken in every run.
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
    ATOL_CPLX_DET_BASE_CARRY,
    ATOL_CPLX_KIN_CARRY,
    RTOL_CPLX_DET_BASE_CARRY,
    RTOL_CPLX_KIN_CARRY,
    get_tolerance_xla_rust_consistency,
)
from tests.test_rs_kernel._rs_fixtures import rs_ham  # noqa: E402

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

# ``route_predicate_counters()`` indices of the complex FAST chunk carries.
_KIN_STEPS, _KIN_DECLINES, _KIN_SEEDS, _KIN_ADVANCES, _KIN_RESEEDS = 22, 23, 24, 25, 26
_BASE_STEPS, _BASE_DECLINES = 27, 28
_NW = 2
_STEPS = 7
_PATTERNS = (
    "twist_sphepgto_j12",
    "twist_sphepgto_j123",
    "twist_sphepgto_jdtn",
    "pgto_pw_twist_jnone",
    "bspline_twist_jnone",
)


def _run(ham, monkeypatch, kin_fresh, base_fresh):
    from jqmc.jqmc_mcmc import MCMC

    monkeypatch.delenv("JQMC_FLAG_RUST_ON_CPU", raising=False)
    monkeypatch.delenv("JQMC_RS_ECP_COMPLEX_FRESH_SEED", raising=False)
    for name, on in (("JQMC_RS_KIN_COMPLEX_FRESH", kin_fresh), ("JQMC_RS_DET_COMPLEX_FRESH_BASE", base_fresh)):
        if on:
            monkeypatch.setenv(name, "1")
        else:
            monkeypatch.delenv(name, raising=False)
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
    return mcmc, counters


def _check_counters(pattern, mcmc, counters, kin_fresh, base_fresh):
    """Exact carry counts: one kinetic and one base decision per walker step, one seed per
    walker chunk (the driver runs the seven steps as one chunk), one advance per accepted move."""
    steps = _NW * _STEPS
    kin = (counters[_KIN_STEPS], counters[_KIN_DECLINES])
    base = (counters[_BASE_STEPS], counters[_BASE_DECLINES])
    assert kin == ((0, steps) if kin_fresh else (steps, 0)), f"{pattern}: kinetic carry (engaged, declined) {kin}"
    assert base == ((0, steps) if base_fresh else (steps, 0)), f"{pattern}: base carry (engaged, declined) {base}"
    seeds, advances, reseeds = counters[_KIN_SEEDS], counters[_KIN_ADVANCES], counters[_KIN_RESEEDS]
    if kin_fresh:
        assert (seeds, advances, reseeds) == (0, 0, 0), f"{pattern}: hooked run touched the tables {counters}"
    else:
        accepted = int(mcmc._MCMC__accepted_moves)
        assert accepted > 0, f"{pattern}: vacuous walk (no accepted move)"
        assert reseeds == 0, f"{pattern}: kinetic inverse probe re-seeded {reseeds} times"
        assert (seeds, advances) == (_NW, accepted), f"{pattern}: (seeds, advances) {(seeds, advances)}, accepted {accepted}"


def _zone_raise(atol, rtol, zones):
    """Raise a full-precision floor to the active zone envelope (mixed lane)."""
    snap = precision_policy_snapshot()
    if any(snap.get(z, "float64") != "float64" for z in zones):
        zone_atol, zone_rtol = get_tolerance_xla_rust_consistency(zones)
        atol, rtol = max(atol, zone_atol), max(rtol, zone_rtol)
    return atol, rtol


def _kin_tolerance():
    # Mixed lane: the fresh kinetic contracts against the sweep's ao_eval-zone value-chain
    # inverse while the tables carry their fused fp64 inverse, and the J1/J2 delta chain
    # accumulates outside the fp32 jastrow_grad_lap zone of the fresh rebuild.
    return _zone_raise(ATOL_CPLX_KIN_CARRY, RTOL_CPLX_KIN_CARRY, ("ao_eval", "jastrow_grad_lap"))


def _base_tolerance():
    # Mixed lane: the carried and rebuilt bases are ao_eval-zone value chains.
    return _zone_raise(ATOL_CPLX_DET_BASE_CARRY, RTOL_CPLX_DET_BASE_CARRY, ("ao_eval",))


def _same_walk(pattern, a, b):
    np.testing.assert_array_equal(a._MCMC__jax_PRNG_key_list, b._MCMC__jax_PRNG_key_list, err_msg=f"{pattern}: keys")
    np.testing.assert_array_equal(a._MCMC__latest_r_up_carts, b._MCMC__latest_r_up_carts, err_msg=f"{pattern}: r_up")
    np.testing.assert_array_equal(a._MCMC__latest_r_dn_carts, b._MCMC__latest_r_dn_carts, err_msg=f"{pattern}: r_dn")
    assert int(a._MCMC__accepted_moves) == int(b._MCMC__accepted_moves), f"{pattern}: accepted moves differ"


def _e_l(pattern, mcmc):
    e = np.asarray(mcmc.e_L)
    assert e.size == _NW * _STEPS and np.all(np.isfinite(e)), f"{pattern}: e_L not produced {e.shape}"
    return e


def _assert_close(pattern, what, a, b, atol, rtol):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=f"{pattern}: {what}")


@pytest.mark.parametrize("pattern", _PATTERNS)
def test_complex_chunk_kin_carry_matches_fresh(pattern, monkeypatch):
    """Carried kinetic tables == fresh per-e_L kinetic on the same walk, within the carry floor.

    QE Li k=(1/4,1/4,1/4) ccECP fixtures (pGTO with J1/J2, J1/J2/J3, DTN; pGTO+PW and
    B-spline leaves without Jastrow). Two walkers, seven driver steps, seed 34356, random
    mesh, Rust backend on, determinant base carried in both runs. Walkers, keys and w_L
    are byte-identical; the stored complex e_L agree within ATOL/RTOL_CPLX_KIN_CARRY.
    """
    ham = rs_ham(pattern)
    carried, c_counts = _run(ham, monkeypatch, kin_fresh=False, base_fresh=False)
    fresh, f_counts = _run(ham, monkeypatch, kin_fresh=True, base_fresh=False)
    _check_counters(pattern, carried, c_counts, kin_fresh=False, base_fresh=False)
    _check_counters(pattern, fresh, f_counts, kin_fresh=True, base_fresh=False)
    _same_walk(pattern, carried, fresh)
    np.testing.assert_array_equal(np.asarray(carried.w_L), np.asarray(fresh.w_L))
    atol, rtol = _kin_tolerance()
    _assert_close(pattern, "kin carry e_L", _e_l(pattern, carried), _e_l(pattern, fresh), atol, rtol)


@pytest.mark.parametrize("pattern", _PATTERNS)
def test_complex_chunk_det_base_carry_matches_rebuild(pattern, monkeypatch):
    """Carried determinant base == per-step rebuild on the same walk, within the SM-carry floor.

    Same fixtures and driver settings, kinetic tables carried in both runs. Keys are
    byte-identical by construction and the positions are asserted byte-identical (no
    fp-floor Metropolis flip on these walks); w_L and e_L agree within
    ATOL/RTOL_CPLX_DET_BASE_CARRY.
    """
    ham = rs_ham(pattern)
    carried, c_counts = _run(ham, monkeypatch, kin_fresh=False, base_fresh=False)
    rebuilt, r_counts = _run(ham, monkeypatch, kin_fresh=False, base_fresh=True)
    _check_counters(pattern, carried, c_counts, kin_fresh=False, base_fresh=False)
    _check_counters(pattern, rebuilt, r_counts, kin_fresh=False, base_fresh=True)
    _same_walk(pattern, carried, rebuilt)
    atol, rtol = _base_tolerance()
    _assert_close(pattern, "base carry w_L", np.asarray(carried.w_L), np.asarray(rebuilt.w_L), atol, rtol)
    _assert_close(pattern, "base carry e_L", _e_l(pattern, carried), _e_l(pattern, rebuilt), atol, rtol)


@pytest.mark.parametrize("pattern", _PATTERNS)
def test_complex_chunk_carries_match_pre_carry_route(pattern, monkeypatch):
    """Both carries vs both hooks (the pre-carry chunk route) on the same walk.

    w_L within the base-carry floor; e_L within the sum of the kinetic and base floors
    (the two differences add).
    """
    ham = rs_ham(pattern)
    carried, c_counts = _run(ham, monkeypatch, kin_fresh=False, base_fresh=False)
    old, o_counts = _run(ham, monkeypatch, kin_fresh=True, base_fresh=True)
    _check_counters(pattern, carried, c_counts, kin_fresh=False, base_fresh=False)
    _check_counters(pattern, old, o_counts, kin_fresh=True, base_fresh=True)
    _same_walk(pattern, carried, old)
    b_atol, b_rtol = _base_tolerance()
    k_atol, k_rtol = _kin_tolerance()
    _assert_close(pattern, "w_L", np.asarray(carried.w_L), np.asarray(old.w_L), b_atol, b_rtol)
    _assert_close(pattern, "e_L", _e_l(pattern, carried), _e_l(pattern, old), b_atol + k_atol, b_rtol + k_rtol)
