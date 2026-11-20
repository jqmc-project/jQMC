"""Layer 2: PRODUCTION ROUTE PARITY for the complex-twist GFMC interval.

This is the complex-twist ``dltmove`` cell, promoted from "call the PyO3 method directly" to "drive the
PRODUCTION ``GFMC_t`` / ``GFMC_n`` driver and prove, with a spy, which entry point ran
and what reached it".

The defect this exists to catch (and which shipped): the complex
``run_gfmc_projection_{n,t}_complex`` PyO3 methods had no ``dltmove`` parameter and
hard-coded ``dltmove: false``, so ``non_local_move="dltmove"`` silently executed the
t-move Hamiltonian on every complex-twist ECP GFMC run. Meanwhile
``test_gfmc_projection.py::test_complex_twist_chunk_honours_dltmove`` was green --
it called the FREE function ``jqmc_rs._gfmc_lrdmc_ecp_complex(..., dltmove)``, never the
production method. ``test_gfmc_complex_dltmove_route.py`` closes half
the gap by calling the production METHODS; this module closes the other half by going
through the DRIVER, which is where the argument is actually assembled from
``self._non_local_move``.

The four route-parity checks, per cell:

  1. **native ran** -- the spy saw ``run_gfmc_projection_{t,n}_complex``, once per walker
     per branching step. Zero calls would mean the eligibility gate silently sent the run
     to the per-step JAX path.
  2. **the discriminating argument arrived** -- ``dltmove`` is present in the live PyO3
     signature, was explicitly supplied by the driver, and carries the value the user
     asked for. The live signature and the recorded call are checked directly.
  3. **return structure** -- the interval returns the same block set, in the same order,
     with the same dtypes on both non-local moves (a complex ``e_L``, a real ``w_L``).
  4. **DLA separation + cross-backend values** -- ``dltmove`` genuinely separates from
     ``tmove`` on both backends, and for EACH non-local move the native and XLA runs
     (same seed, same walk, ``random_discretized_mesh=True``) agree per step and on the
     public energy under ``_assert_deterministic_driver_parity`` (the periodic
     fast-interval envelope of ``test_gfmc_chain_modes.py``). This is the only
     driver-level Rust/XLA value oracle for the complex twist path with the native
     Threefry/Haar rotation and with ``dltmove``.
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

from jqmc._setting import (
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_STEPS,
    GFMC_MIN_WARMUP_STEPS,
)
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t

from tests.test_rs_kernel._route_spy import (
    assert_arg_reached,
    assert_native_ran,
    install_ctx_spy,
)
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_gfmc_chain_modes import _assert_deterministic_driver_parity

# lithium_k025: periodic COMPLEX twist (k != Gamma) + ECP -> the only family where the
# non-local move choice reaches a complex kernel at all.
_PATTERN = "twist_sphepgto_j12"
_STEPS = 2 * GFMC_MIN_STEPS
_ALAT = 0.30
_TAU = 0.05
_NSUB = 3

_HAM_CACHE: dict[str, object] = {}


def _ham():
    if _PATTERN not in _HAM_CACHE:
        _HAM_CACHE[_PATTERN] = rs_ham(_PATTERN)
    return _HAM_CACHE[_PATTERN]


def _run(
    driver,
    ham,
    seed,
    mode,
    non_local_move,
    num_steps=_STEPS,
    comput_position_deriv=False,
    comput_stress=False,
    num_walkers=2,
    random_mesh=True,
):
    """One PRODUCTION GFMC run. ``mode`` in {"jax", "fast"}.

    Nothing here reaches into the driver: the backend is selected exactly as a user
    selects it (the env override the dispatch reads at run() head), and the non-local
    move is the public constructor argument.
    """
    if mode == "jax":
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    else:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    try:
        common = dict(
            hamiltonian_data=ham,
            num_walkers=num_walkers,
            num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
            mcmc_seed=seed,
            alat=_ALAT,
            non_local_move=non_local_move,
            random_discretized_mesh=random_mesh,
            comput_position_deriv=comput_position_deriv,
            comput_stress=comput_stress,
        )
        g = GFMC_n(**common, num_mcmc_per_measurement=_NSUB, E_scf=-1.0) if driver == "n" else GFMC_t(**common, tau=_TAU)
        g.run(num_mcmc_steps=num_steps)
        return g
    finally:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)


def _energy(g):
    E, E_err, _, _ = g.get_E(num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS, num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS)
    return float(E), float(E_err)


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["t", "n"])
@pytest.mark.parametrize("non_local_move", ["tmove", "dltmove"])
def test_route_parity_complex_gfmc_interval(driver, non_local_move, monkeypatch):
    """CHECKS 1-3 for ``run_gfmc_projection_{t,n}_complex`` driven by the PRODUCTION driver."""
    method = f"run_gfmc_projection_{driver}_complex_batch"
    ctx_tag = f"{driver}/{non_local_move}"
    ham = _ham()

    spy = install_ctx_spy(monkeypatch)
    g = _run(driver, ham, 34356, "fast", non_local_move)

    # CHECK 1: the native complex interval ran -- once per walker per branching step.
    assert_native_ran(spy, method, at_least=_STEPS, context=ctx_tag)
    # The ctx must have been built for the COMPLEX family; a real-family ctx reaching a
    # complex fixture would be a different (and equally invisible) wiring bug.
    assert bool(spy.build_arg("complex")) is True, f"{ctx_tag}: build_obs_ctx was not asked for the complex family"
    assert bool(spy.build_arg("pbc")) is True, f"{ctx_tag}: build_obs_ctx was not asked for the periodic family"
    assert bool(spy.build_arg("want_gfmc")) is True, f"{ctx_tag}: build_obs_ctx was not asked for the GFMC args"

    # CHECK 2: ``dltmove`` exists on the signature, was explicitly passed, and
    # carries what the user asked for -- not the Rust default.
    assert_arg_reached(spy, method, "dltmove", non_local_move == "dltmove", context=ctx_tag)

    # CHECK 3: return block set / order / dtype. The complex interval returns a mapping
    # with a COMPLEX e_L and a REAL w_L; a demoted (real) e_L would drop Im(T) silently.
    calls = spy.calls_to(method)
    blocks = tuple(k for k, _ in calls[0].ret[1])
    assert calls[0].ret[0] == "dict", f"{ctx_tag}: {method} did not return a mapping, got {calls[0].ret[0]}"
    assert set(blocks) >= {"e_L", "w_L"}, f"{ctx_tag}: {method} returned blocks {blocks}, missing e_L / w_L"
    for i, c in enumerate(calls):
        got = tuple(k for k, _ in c.ret[1])
        assert got == blocks, f"{ctx_tag}: {method} call {i} changed its block ORDER: {got} vs {blocks}"
    # Non-vacuity of the complex leg: a twist fixture whose stored e_L has an all-zero
    # imaginary part cannot tell a complex interval from a demoted real one.
    assert np.max(np.abs(np.imag(np.asarray(g._stored_e_L)))) > 0.0, (
        f"{ctx_tag}: stored complex e_L has an identically zero imaginary part -- the complex leg is vacuous"
    )
    assert np.iscomplexobj(np.asarray(g._stored_e_L)), (
        f"{ctx_tag}: the driver stored a REAL e_L from a complex-twist interval -- Im(T) was dropped on the way back"
    )
    assert not np.any(np.isnan(np.asarray(g._stored_e_L))), f"{ctx_tag}: stored e_L has NaN"
    assert not np.any(np.isnan(np.asarray(g._stored_w_L))), f"{ctx_tag}: stored w_L has NaN"


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["t", "n"])
@pytest.mark.parametrize("comput_stress", [False, True], ids=["force-only", "force-plus-stress"])
def test_route_parity_complex_gfmc_force_and_stress(driver, comput_stress, monkeypatch):
    """Force and independent stress selections reach the complex native route.

    This is the production-route companion to the free-kernel XLA comparisons in
    ``test_gfmc_complex_force_stress.py``.

    Both cases request position derivatives. The force-only case must carry
    ``want_stress=False``; the force-plus-stress case must explicitly request and
    carry ``want_stress=True``. A production ``NotImplementedError`` is honest,
    but it fails this test.
    """
    method = f"run_gfmc_projection_{driver}_complex_batch"
    context = f"{driver}/complex-{'force-plus-stress' if comput_stress else 'force-only'}"
    spy = install_ctx_spy(monkeypatch)
    _run(
        driver,
        _ham(),
        34356,
        "fast",
        "tmove",
        comput_position_deriv=True,
        comput_stress=comput_stress,
    )
    assert_native_ran(spy, method, at_least=_STEPS, context=context)
    assert_arg_reached(spy, method, "want_force", True, context=context)
    assert_arg_reached(spy, method, "want_stress", comput_stress, context=context)


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["t", "n"])
def test_route_parity_complex_gfmc_values_and_dla_separation(driver, monkeypatch):
    """CHECK 4 for both non-local moves, plus the POWER check and the value oracle.

    (a) on XLA, ``dltmove`` and ``tmove`` are DIFFERENT Hamiltonians on this fixture.
        Without this, (c) and (d) would be vacuous.
    (b) on the native route they are different too. Equality here is precisely the signature
        of a dropped flag (both runs executing t-move).
    (c) for each non-local move, native vs XLA with the same seed and the native
        random mesh: identical walk (Threefry keys, positions at the random-mesh
        coordinate pin) and per-step / public values at the periodic XLA-Rust envelope.
    (d) negative control: the same comparison on a +1e-3 shifted native e_L must fail.

    Both walkers participate in every interval. Conventional GFMC keeps the
    XLA batch-wide RNG advancement, including an early-finished walker.
    """
    ham = _ham()
    seed = 34356

    g_jax_tm = _run(driver, ham, seed, "jax", "tmove", num_walkers=2)
    g_jax_dl = _run(driver, ham, seed, "jax", "dltmove", num_walkers=2)

    spy_tm = install_ctx_spy(monkeypatch)
    g_rs_tm = _run(driver, ham, seed, "fast", "tmove", num_walkers=2)
    assert_native_ran(spy_tm, f"run_gfmc_projection_{driver}_complex_batch", at_least=_STEPS, context=f"{driver}/tmove")

    spy_dl = install_ctx_spy(monkeypatch)
    g_rs_dl = _run(driver, ham, seed, "fast", "dltmove", num_walkers=2)
    assert_native_ran(spy_dl, f"run_gfmc_projection_{driver}_complex_batch", at_least=_STEPS, context=f"{driver}/dltmove")

    E_jax_tm, E_jax_dl, E_rs_tm, E_rs_dl = (_energy(g) for g in (g_jax_tm, g_jax_dl, g_rs_tm, g_rs_dl))
    for tag, E in (("jax/tmove", E_jax_tm), ("jax/dltmove", E_jax_dl), ("rs/tmove", E_rs_tm), ("rs/dltmove", E_rs_dl)):
        assert np.isfinite(E[0]) and np.isfinite(E[1]), f"{tag}: non-finite energy {E}"

    # POWER: each backend distinguishes the two Hamiltonians here.
    assert E_jax_dl[0] != E_jax_tm[0], (
        f"driver={driver}: the XLA reference gave a bit-identical energy for tmove and dltmove on "
        f"{_PATTERN} -- this fixture cannot discriminate the two Hamiltonians, so the native check below "
        f"would be vacuous. Choose a fixture where the DLA off-diagonal is non-zero."
    )
    # (b) so does the native route.
    assert E_rs_dl[0] != E_rs_tm[0], (
        f"driver={driver}: the NATIVE route produced a bit-identical energy for tmove and dltmove while XLA "
        f"separates them -- the DLA flag is not reaching the complex kernel."
    )

    # (c) cross-backend value parity, per non-local move, on the random mesh.
    for g_rs, g_jax in ((g_rs_tm, g_jax_tm), (g_rs_dl, g_jax_dl)):
        assert np.max(np.abs(np.imag(np.asarray(g_jax._stored_e_L)))) > 0.0, (
            f"driver={driver}: XLA complex e_L has zero imaginary part -- the complex value oracle is vacuous"
        )
        _assert_deterministic_driver_parity(g_rs, g_jax, driver, periodic=True, random_mesh=True)

    # (d) negative control: a +1e-3 shift of the native complex e_L must turn (c) red.
    g_rs_tm._stored_e_L = np.asarray(g_rs_tm._stored_e_L) + 1.0e-3
    with pytest.raises(AssertionError, match=f"GFMC_{driver} _stored_e_L"):
        _assert_deterministic_driver_parity(g_rs_tm, g_jax_tm, driver, periodic=True, random_mesh=True)
