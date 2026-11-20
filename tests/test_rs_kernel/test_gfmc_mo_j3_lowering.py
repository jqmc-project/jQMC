"""A molecular MO-wrapped J3 streams through the native LRDMC mesh caches.

The production three-body Jastrow written by ``jqmc-tool trexio convert-to -j3 mo``
is MO-wrapped (``MOs_data`` over the molecular GTO basis, ``num_mo < num_ao``).
The native GFMC mesh state (``gfmc::mesh_fast_state``) streams J3 only for a
raw-AO orbital, so without the raw-AO lowering every kinetic and ECP mesh evaluation
of such a trial takes the full-value J3 slow fallback: every electron's J3 orbitals are
re-evaluated per mesh point and the persistent candidate tables are rebuilt on
every sub-step (route counter 12 > 0, counter 4 == 0). The result is correct
but ~40x slower (example01 water: 35-39 s instead of ~0.9 s per branching).
An all-zero ``j_matrix`` (the unoptimized trial) is skipped entirely, which hides
the slow route until the J3 is optimized.

The parser lowers a nonzero molecular MO-wrapped J3 to its exact raw-AO equivalent
(``M_ao = C^T M_mo C``, ``j1_ao = C^T j1_mo``) at the real-path GFMC parse
boundary. These tests pin:

* the route: zero J3 slow fallbacks and live candidate-table refreshes for a
  genuinely non-square MO-wrapped J3 (all-electron Cartesian and ECP spherical);
* the value: the production GFMC_n interval with that J3 (kinetic and ECP
  non-local meshes) equals the XLA reference machine sub-step by sub-step
  (the ``test_gfmc_projection`` interval contract), and the all-electron
  GFMC_n / GFMC_t production walk is deterministic across Rust and XLA (the
  shared ``_assert_deterministic_driver_parity`` oracle); full and mixed are
  each judged at their own backend-parity tier;
* the production witness: a J3 that still has no streaming state (periodic
  MO-wrapped composite) raises the always-on slow-fallback witness, and the
  driver logs exactly one warning per run.
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

import logging

import numpy as np
import pytest

from jqmc.jastrow_factor import Jastrow_three_body_data
from jqmc.molecular_orbital import MOs_data
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_gfmc_chain_modes import (
    _STEPS_STAT,
    _assert_deterministic_driver_parity,
    _install_ctx_spy,
    _run_gfmc,
)

pytestmark = [pytest.mark.with_chdir]

# Indices of ``jqmc_rs.gfmc_route_counters()`` (append-only list in gfmc.rs).
_J3_SLOW_FALLBACKS = 12
_CAND_BUILD_MOL = 3
_CAND_REFRESH_MOL = 4

_GATE_STEPS = 3
_GATE_WALKERS = 2

# Base fixtures whose geminal carries a non-square molecular ``MOs_data``.
_MO_J3_BASES = {
    "ae_cart": "mol_cart_sd_j12",  # N all-electron, Cartesian ccpvdz: 5 MOs over 15 AOs
    "ecp_sphe": "mol_ecp_sphe_j12",  # water ccECP, spherical ccpvqz: 4 MOs over 114 AOs
}


def _mo_j3_ham(kind: str):
    """``rs_ham`` base fixture plus a nonzero, non-square molecular MO-wrapped J3.

    The J3 orbital is the geminal's own occupied-MO leaf (production ``-j3 mo``
    shape: ``num_mo < num_ao``); the coupling is a deterministic symmetric draw.
    """
    ham = rs_ham(_MO_J3_BASES[kind])
    wf = ham.wavefunction_data
    orb = wf.geminal_data.orb_data_up_spin
    assert isinstance(orb, MOs_data), f"{kind}: base geminal orbital is not MOs_data"
    assert orb.num_mo < orb.aos_data.num_ao, f"{kind}: MO wrapper is square -- not the production shape"
    rng = np.random.default_rng(20261007)
    n = int(orb.num_mo)
    jm = 0.3 * rng.normal(size=(n, n + 1))
    jm[:, :n] = 0.5 * (jm[:, :n] + jm[:, :n].T)
    j3 = Jastrow_three_body_data(orb_data=orb, j_matrix=jm)
    ham = ham.replace(
        wavefunction_data=wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=j3)),
    )
    ham.sanity_check()
    assert ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data.mo_coefficients is not None
    assert np.any(np.asarray(ham.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix) != 0.0)
    return ham


@pytest.mark.light
@pytest.mark.parametrize("driver", ["n", "t"])
@pytest.mark.parametrize("kind", sorted(_MO_J3_BASES))
def test_rust_gfmc_mo_j3_takes_streaming_route(driver, kind):
    """A nonzero molecular MO-wrapped J3 never takes the J3 slow fallback.

    Without the raw-AO lowering, counter 12 counts every kinetic/ECP mesh evaluation
    and the molecular candidate tables are rebuilt per sub-step (counter 4 == 0).
    """
    import jqmc_rs

    ham = _mo_j3_ham(kind)
    jqmc_rs.gfmc_route_counters_reset()
    _run_gfmc(driver, ham, 34356, "rust", _GATE_STEPS, num_walkers=_GATE_WALKERS)
    counters = jqmc_rs.gfmc_route_counters()
    assert counters[_J3_SLOW_FALLBACKS] == 0, (
        f"{kind}/GFMC_{driver}: MO-wrapped J3 took the full-value J3 slow fallback {counters[_J3_SLOW_FALLBACKS]} times"
    )
    assert counters[_CAND_BUILD_MOL] > 0, f"{kind}/GFMC_{driver}: molecular candidate tables were never built"
    assert counters[_CAND_REFRESH_MOL] > 0, (
        f"{kind}/GFMC_{driver}: molecular candidate tables were never refreshed across hops "
        f"(builds={counters[_CAND_BUILD_MOL]})"
    )


@pytest.mark.light
@pytest.mark.parametrize("kind", sorted(_MO_J3_BASES))
def test_rust_gfmc_mo_j3_interval_matches_xla(kind):
    """The production GFMC_n interval with the lowered J3 equals the XLA machine.

    The ``test_gfmc_projection`` interval contract (pinned key, fixed mesh,
    tmove, prefixes of 1..3 sub-steps): w_L, e_L and V_diag within the
    established projection policy, walkers and the final Threefry key exact,
    the walk non-trivial. The Rust side streams the lowered raw-AO J3
    on the kinetic and (ECP fixture) non-local meshes -- counter 12 stays zero --
    while the XLA machine consumes the MO-wrapped J3 unchanged.
    """
    import jax
    import jqmc_rs

    from jqmc._setting import (
        ATOL_NONVACUITY_DISTINCTION,
        ATOL_PYTHON_FP64_LOOSE,
        RTOL_NONVACUITY_DISTINCTION,
        RTOL_PYTHON_FP64_ZERO,
    )
    from tests.test_rs_kernel.test_gfmc_projection import (
        _EL_ZONES_J,
        _MESH_GP,
        _MESH_W,
        _assert_close,
        _cfg,
        _ctx,
        _gfmc_projection_tolerance,
        _xla_interval_n,
    )

    ham = _mo_j3_ham(kind)
    r_up, r_dn = _cfg(ham, 1)
    atol, rtol = _gfmc_projection_tolerance(_EL_ZONES_J)
    ctx = _ctx(ham)
    key = jax.random.PRNGKey(7)
    jqmc_rs.gfmc_route_counters_reset()
    for nsub in (1, 2, 3):
        ref = _xla_interval_n(ham, r_up, r_dn, 0.30, -1.0, nsub, key, "tmove")
        got = ctx.run_gfmc_projection_n(
            r_up,
            r_dn,
            0.30,
            -1.0,
            nsub,
            int(key[0]),
            int(key[1]),
            np.zeros((0, 3, 3)),
            _MESH_W,
            _MESH_GP,
            random_mesh=False,
            dltmove=False,
            nn=1,
        )
        _assert_close(float(got["w_L"]), ref["w_L"], atol, rtol, f"w_L[{kind},nsub={nsub}]")
        _assert_close(float(got["e_L"]), ref["e_L"], atol, rtol, f"e_L[{kind},nsub={nsub}]")
        _assert_close(float(got["V_diag"]), ref["V_diag"], atol, rtol, f"V_diag[{kind},nsub={nsub}]")
        assert np.allclose(np.asarray(got["r_up"]), ref["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        assert np.allclose(np.asarray(got["r_dn"]), ref["r_dn"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        assert list(np.asarray(got["final_key"], dtype=np.uint32)) == list(np.asarray(ref["key"], dtype=np.uint32))
        assert not np.allclose(
            ref["r_up"], r_up, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION
        ) or not np.allclose(ref["r_dn"], r_dn, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION)
    counters = jqmc_rs.gfmc_route_counters()
    assert counters[_J3_SLOW_FALLBACKS] == 0, (kind, "interval took the J3 slow fallback", counters[_J3_SLOW_FALLBACKS])


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["n", "t"])
def test_rust_gfmc_mo_j3_matches_xla(driver, monkeypatch):
    """The lowered (raw-AO) native all-electron walk equals the XLA MO-wrapped walk.

    Same seed, fixed mesh, one walker: identical walk and Threefry keys, and
    per-step weights / local energies within the backend-parity tier of the
    active precision mode (strict in full precision).
    """
    ham = _mo_j3_ham("ae_cart")
    seed = 34356
    xla = _run_gfmc(driver, ham, seed, "jax", _STEPS_STAT, random_mesh=False, num_walkers=1)
    counts = _install_ctx_spy(monkeypatch)
    rust = _run_gfmc(driver, ham, seed, "rust", _STEPS_STAT, random_mesh=False, num_walkers=1)
    assert counts[f"run_gfmc_projection_{driver}"] >= _STEPS_STAT, f"native GFMC chunk did not engage: {counts}"
    _assert_deterministic_driver_parity(rust, xla, driver)


@pytest.mark.light
def test_rust_gfmc_j3_slow_fallback_warns_once(caplog):
    """A J3 without a streaming state raises the witness; the driver warns once per run.

    A lowered molecular MO-wrapped J3 must leave the always-on witness clear.
    The periodic MO-wrapped composite J3 (``pbc_composite_j123``) is outside the
    raw-AO lowering and still takes the slow fallback; the Rust witness must
    report it and ``_warn_native_j3_slow_fallback_once`` must log one warning
    across two driver runs.
    """
    import jqmc_rs

    import jqmc.jqmc_gfmc as gfmc_module

    jqmc_rs.gfmc_take_j3_slow_fallback_seen()  # clear any witness left by earlier tests
    _run_gfmc("n", _mo_j3_ham("ecp_sphe"), 34356, "rust", _GATE_STEPS, num_walkers=1)
    assert not jqmc_rs.gfmc_take_j3_slow_fallback_seen(), "lowered molecular MO-wrapped J3 raised the slow-fallback witness"

    ham = rs_ham("pbc_composite_j123")
    j3 = ham.wavefunction_data.jastrow_data.jastrow_three_body_data
    assert j3 is not None and j3.orb_data.mo_coefficients is not None
    gfmc_module._NATIVE_J3_SLOW_FALLBACK_WARNED = False
    jqmc_rs.gfmc_take_j3_slow_fallback_seen()
    with caplog.at_level(logging.WARNING, logger="jqmc"):
        _run_gfmc("n", ham, 34356, "rust", _GATE_STEPS, num_walkers=1)
        _run_gfmc("n", ham, 34357, "rust", _GATE_STEPS, num_walkers=1)
    warnings = [r for r in caplog.records if "three-body Jastrow took the full-value slow fallback" in r.getMessage()]
    assert len(warnings) == 1, f"expected exactly one slow-fallback warning per run, got {len(warnings)}"
    assert gfmc_module._NATIVE_J3_SLOW_FALLBACK_WARNED
