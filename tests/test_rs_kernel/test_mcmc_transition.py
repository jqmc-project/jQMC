"""Production parity for MCMC sample generation as a state machine.

Charter: compare the XLA and Rust MCMC *transition* -- single-electron proposal,
drift/Kato step size, Green's-function ratio ``T_ratio``, Metropolis accept/reject,
rank-1 (Sherman-Morrison) inverse update, the Jastrow running state and the walker
update -- sweep by sweep, on IDENTICAL injected randomness, comparing every
intermediate state and not only the final configuration.

Reference and injection path
----------------------------
The oracle here is the PRODUCTION XLA driver kernel
``jqmc.jqmc_mcmc._update_electron_positions`` itself -- the exact ``fori_loop``
``body_fun`` the JAX lane runs -- NOT a hand-written Python replica of it. Every
intermediate this module compares is therefore a genuine XLA-vs-Rust comparison;
there is no hand-written replica of the physics that could absorb a discrepancy
between the XLA and Rust implementations.

Randomness is pinned through the codebase's OWN injection path, the one production
uses in the fixed-input Rust entry-point tests:
``_draw_injected_normals`` replays the EXACT six-``jax.random.split``-per-proposal
threefry stream of ``body_fun`` and hands Rust only the 4th subkey's Gaussian. Rust
regenerates the other five draws (up/dn choice, up index, dn index, Cartesian axis,
Metropolis ``b``) from the same key words. Both engines therefore consume identical
randomness, and the advanced ``final_key`` is compared bit-for-bit -- top-level RNG
determinism is thus asserted, not assumed.

What is compared after every move / sweep
-----------------------------------------
``accepted`` (the accept/reject decision itself), ``r_up`` / ``r_dn`` (the walker
update, kept UNWRAPPED across the boundary), ``geminal`` (rank-1 row/col update),
``geminal_inv`` (the Sherman-Morrison running inverse), and the whole
``Det_ratio_streaming_state`` quartet ``ao_up`` / ``ao_dn`` / ``paired_dn`` /
``paired_up_lambda`` (the running AO + paired caches that the ratio kernels and the
downstream ``e_L`` consume). ``test_move_by_move_transition`` does this after EVERY
single move, so a transition that is wrong on move *k* cannot be masked by a later
move.

The Jastrow running state (J3 tables) is not exported across the PyO3 boundary by
``_mcmc_sweep``, so it is pinned INDIRECTLY but tightly, by two independent means:
(a) it drives ``J_ratio`` hence every accept/reject decision, which is compared
move-by-move against XLA's own ``jastrow_state``-fed ``body_fun``; and (b)
``test_rs_streaming_cache_no_drift`` pins the Rust-internal running state against a
from-scratch rebuild (an N-move sweep vs N chained 1-move sweeps that each
re-initialise all caches from positions). This is partial coverage of cache updates.

Floating-point tolerances come exclusively from the dedicated XLA--Rust
consistency policy selected by pytest's ``full`` / ``mixed`` precision mode.
Integer and decision channels remain bit-exact.

Two XLA-vs-Rust contracts are pinned at the bottom of the module as PLAIN tests
(no xfail -- repo policy: a known defect fails visibly): ``_single_move`` keeps the
J3/DTN Jastrow channels (case A: ``mcmc.rs::single_move`` builds the J3 + DTN ratio
state at the input config and folds ``j12 * j3r * jdtn`` into the Metropolis ratio
exactly as ``mcmc_sweep`` does; the ``_single_move`` PyO3 wrapper parses
``jastrow_three_body_data`` like ``_mcmc_sweep``), and a NaN Metropolis acceptance
ratio rejects (case B: ``mcmc.rs::accept_clamp``; the test injects a NaN-poisoned
running inverse deterministically).
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
from jax import numpy as jnp
from jax import random

import jqmc_rs

pytestmark = [pytest.mark.with_chdir]

from jqmc._precision import get_tolerance_min, get_tolerance_xla_rust_consistency, is_mixed_precision_enabled
from jqmc._setting import ATOL_BIT_EXACT, RTOL_BIT_EXACT
from jqmc.determinant import _init_det_ratio_streaming_state, compute_geminal_all_elements
from jqmc.jastrow_factor import (
    Jastrow_ratio_state,
    _init_jastrow_dtn_ratio_state,
    _init_jastrow_ratio_state,
)
from jqmc.jqmc_mcmc import _draw_injected_normals, _jit_vmap_update, _update_electron_positions

from tests.test_rs_kernel._rs_fixtures import _PATTERNS, rs_ham

# --------------------------------------------------------------------------- #
# Pattern table: every canonical committed-fixture leaf combination the Rust
# sweep supports (orbital family x geminal x Jastrow x AE/ECP x BC). Precision
# tolerance is mode-only; patterns do not select a second strict/medium axis.
# --------------------------------------------------------------------------- #
ALL_PATTERNS = tuple(_PATTERNS)
PERIODIC_PATTERNS = tuple(p for p in ALL_PATTERNS if p.startswith(("pbc_", "twist_", "pw_", "bspline_")))
# Patterns whose fixture carries more than one electron per spin channel -- the
# only ones on which "first vs last electron index" and a two-electron near-node
# are constructible. The committed 1up/1dn fixtures cannot express either cell.
MULTI_ELECTRON_PATTERNS = (
    "mol_cart_sd_j123",
    "mol_cart_sd_jdtn",
    "mol_ecp_sphe_j12",
    "mol_ecp_sphe_j12_pade",  # same water cell on the production-default radial form
)

# Zones on each compared channel's path.
_POS_ZONES = ("det_ratio",)  # walker positions: pure copy of the proposal arithmetic
_GEM_ZONES = ("det_ratio", "det_eval", "ao_eval")  # AO-built geminal + Sherman-Morrison running inverse
_AO_ZONES = ("det_ratio", "det_eval", "ao_eval")  # streaming AO / paired caches

# Deliberately huge relative corruption for the mutation check -- orders of
# magnitude above every named tolerance on every channel, so "the test goes red"
# is a statement about the assertions, not about a tolerance boundary.
_MUTATION_REL = 5.0e-2
_NEAR_NODE_GAPS = (1.0e-4,) if is_mixed_precision_enabled() else (1.0e-4, 1.0e-6)

_HAM_CACHE: dict[str, object] = {}


def _ham(pattern: str):
    """Build (and memoise) the committed-fixture ham for ``pattern``."""
    if pattern not in _HAM_CACHE:
        _HAM_CACHE[pattern] = rs_ham(pattern)
    return _HAM_CACHE[pattern]


def _charges(ham) -> np.ndarray:
    """Effective nuclear charges the Kato drift uses (ECP: Z - z_core)."""
    cp = ham.coulomb_potential_data
    Z = np.asarray(ham.structure_data.atomic_numbers, dtype=np.float64)
    if cp.ecp_flag:
        Z = Z - np.asarray(cp.z_cores, dtype=np.float64)
    return Z


def _cell_nmax(ham):
    """``(cell, n_max)`` for the Rust sweep; ``(None, 0)`` for molecular."""
    if bool(ham.structure_data.pbc_flag):
        return np.asarray(ham.structure_data.cell, dtype=np.float64), int(ham.structure_data._n_max)
    return None, 0


def _draw_cfg(ham, seed: int, scale: float = 0.7):
    """Draw an electron configuration seeded around the nuclei."""
    gem = ham.wavefunction_data.geminal_data
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    rng = np.random.default_rng(seed)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    r_up = R[rng.integers(0, len(R), nu)] + rng.normal(scale=scale, size=(nu, 3))
    r_dn = R[rng.integers(0, len(R), nd)] + rng.normal(scale=scale, size=(nd, 3))
    return np.ascontiguousarray(r_up, dtype=np.float64), np.ascontiguousarray(r_dn, dtype=np.float64)


def _init_running_state(ham, r_up, r_dn):
    """Build the full MCMC running state at ``(r_up, r_dn)``: geminal, its inverse,
    the det-ratio streaming quartet and the Jastrow (J3/DTN) running state -- exactly
    what the driver builds once at chain entry (jqmc_mcmc.py L1325-1360).
    """
    gem = ham.wavefunction_data.geminal_data
    jas = ham.wavefunction_data.jastrow_data
    G = np.asarray(compute_geminal_all_elements(gem, r_up, r_dn))
    G_inv = np.linalg.inv(G)
    det_state = _init_det_ratio_streaming_state(gem, jnp.asarray(r_up), jnp.asarray(r_dn))
    j3d = jas.jastrow_three_body_data
    j3_state = _init_jastrow_ratio_state(j3d, jnp.asarray(r_up), jnp.asarray(r_dn)) if j3d is not None else None
    dtnd = jas.jastrow_dtn_data
    dtn_state = _init_jastrow_dtn_ratio_state(dtnd, jnp.asarray(r_up), jnp.asarray(r_dn)) if dtnd is not None else None
    return G, G_inv, det_state, Jastrow_ratio_state(j3_state=j3_state, dtn_state=dtn_state)


def _xla_sweep(ham, r_up, r_dn, key, num_moves, Dt, eps, G, G_inv, det_state, jastrow_state):
    """One XLA sweep through the PRODUCTION driver kernel. Returns the full carry."""
    (accepted, _rejected, x_r_up, x_r_dn, x_key, x_G_inv, x_G, x_det, x_jas) = _update_electron_positions(
        jnp.asarray(r_up),
        jnp.asarray(r_dn),
        key,
        num_moves,
        ham,
        Dt,
        eps,
        jnp.asarray(G_inv),
        jnp.asarray(G),
        det_state,
        jastrow_state,
    )
    return dict(
        accepted=int(accepted),
        r_up=np.asarray(x_r_up),
        r_dn=np.asarray(x_r_dn),
        key=x_key,
        geminal=np.asarray(x_G),
        geminal_inv=np.asarray(x_G_inv),
        ao_up=np.asarray(x_det.ao_up),
        ao_dn=np.asarray(x_det.ao_dn),
        paired_dn=np.asarray(x_det.paired_dn),
        paired_up_lambda=np.asarray(x_det.paired_up_lambda),
        det_state=x_det,
        jastrow_state=x_jas,
    )


def _rs_sweep(ham, r_up, r_dn, key, num_moves, Dt, eps, G, G_inv, *, normals=None):
    """One Rust sweep on the SAME starting state and the SAME injected normals.

    ``normals`` defaults to the codebase's bit-exact injection path
    ``_draw_injected_normals`` (the deterministic test-input helper); it is only
    passed explicitly by the mutation check, which corrupts it on purpose.
    """
    gem = ham.wavefunction_data.geminal_data
    jas = ham.wavefunction_data.jastrow_data
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    Z = _charges(ham)
    cell, n_max = _cell_nmax(ham)
    if normals is None:
        normals = np.asarray(_draw_injected_normals(key, num_moves), dtype=np.float64)
    k = np.asarray(key, dtype=np.uint32)
    if np.iscomplexobj(G):
        res = jqmc_rs._mcmc_sweep_complex(
            gem, jas, R, Z, r_up, r_dn, G, G_inv, Dt, eps, int(k[0]), int(k[1]), normals, cell, n_max
        )
    else:
        res = jqmc_rs._mcmc_sweep(gem, jas, R, Z, r_up, r_dn, G, G_inv, Dt, eps, int(k[0]), int(k[1]), normals, cell, n_max)
    return res


def _assert_transition_parity(xla, rs, tag, *, complex_path):
    """Assert every compared state channel of one transition agrees.

    Channels split by precision budget: the accept COUNT and the advanced threefry
    key are exact integers (bit-exact, no tolerance); positions are a straight copy
    of the proposal arithmetic; the geminal / SM inverse carry the det zones; the
    streaming AO+paired quartet additionally carries the AO zone.
    """
    pos_atol, pos_rtol = get_tolerance_xla_rust_consistency(_POS_ZONES, level="medium")
    gem_atol, gem_rtol = get_tolerance_xla_rust_consistency(_GEM_ZONES, level="medium")
    ao_atol, ao_rtol = get_tolerance_xla_rust_consistency(_AO_ZONES, level="medium")

    # --- the accept/reject decision itself (exact) ---
    assert xla["accepted"] == int(rs["accepted"]), f"{tag}: accept count XLA={xla['accepted']} Rust={rs['accepted']}"

    # --- walker update ---
    assert np.allclose(rs["r_up"], xla["r_up"], atol=pos_atol, rtol=pos_rtol), (
        f"{tag}: r_up max|d|={np.abs(np.asarray(rs['r_up']) - xla['r_up']).max():.3e}"
    )
    assert np.allclose(rs["r_dn"], xla["r_dn"], atol=pos_atol, rtol=pos_rtol), (
        f"{tag}: r_dn max|d|={np.abs(np.asarray(rs['r_dn']) - xla['r_dn']).max():.3e}"
    )

    # --- rank-1 geminal + Sherman-Morrison running inverse ---
    # The complex-twist sweep does not emit the geminal matrix (the driver rebuilds
    # it, jqmc_mcmc.py L2178-2195), so only the inverse is available there.
    if not complex_path:
        assert np.allclose(rs["geminal"], xla["geminal"], atol=gem_atol, rtol=gem_rtol), (
            f"{tag}: geminal max|d|={np.abs(np.asarray(rs['geminal']) - xla['geminal']).max():.3e}"
        )
    assert np.allclose(rs["geminal_inv"], xla["geminal_inv"], atol=gem_atol, rtol=gem_rtol), (
        f"{tag}: geminal_inv max|d|={np.abs(np.asarray(rs['geminal_inv']) - xla['geminal_inv']).max():.3e}"
    )

    # --- det-ratio streaming (running AO + paired caches) ---
    for name in ("ao_up", "ao_dn", "paired_dn", "paired_up_lambda"):
        assert np.allclose(rs[name], xla[name], atol=ao_atol, rtol=ao_rtol), (
            f"{tag}: {name} max|d|={np.abs(np.asarray(rs[name]) - xla[name]).max():.3e}"
        )

    # --- advanced threefry key (top-level RNG determinism, exact) ---
    xk = np.asarray(xla["key"], dtype=np.uint32)
    rk = np.asarray(rs["final_key"], dtype=np.uint32)
    assert np.allclose(rk, xk, atol=ATOL_BIT_EXACT, rtol=RTOL_BIT_EXACT), f"{tag}: key drift XLA={xk} Rust={rk}"


def _run_and_compare(pattern, r_up, r_dn, *, seed, num_moves, Dt, eps=1.0e-6, tag="", corrupt=None):
    """Run one XLA sweep and one Rust sweep from the same state/randomness and
    assert full transition parity. ``corrupt`` (mutation check only) is a callable
    mutating the running state handed to RUST only.
    """
    ham = _ham(pattern)
    G, G_inv, det_state, jastrow_state = _init_running_state(ham, r_up, r_dn)
    complex_path = bool(np.iscomplexobj(G))
    key = random.PRNGKey(seed)

    xla = _xla_sweep(ham, r_up, r_dn, key, num_moves, Dt, eps, G, G_inv, det_state, jastrow_state)

    rs_G, rs_G_inv, rs_normals = G, G_inv, None
    if corrupt is not None:
        rs_G, rs_G_inv, rs_normals = corrupt(G.copy(), G_inv.copy(), key, num_moves)
    rs = _rs_sweep(ham, r_up, r_dn, key, num_moves, Dt, eps, rs_G, rs_G_inv, normals=rs_normals)

    _assert_transition_parity(xla, rs, tag or f"{pattern}/seed={seed}/Dt={Dt}", complex_path=complex_path)
    return xla, rs


@pytest.mark.light
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_real_mcmc_sweep_as_guide_tracks_accepted_state():
    """Compare the active real guide on a multi-electron geminal in one sweep."""
    pattern = "mol_cart_sd_j123"
    ham = _ham(pattern)
    r_up, r_dn = _draw_cfg(ham, 0)
    key = random.PRNGKey(100)
    num_moves, Dt, epsilon_as = 30, 1.2, 5.0
    G, G_inv, det_state, jastrow_state = _init_running_state(ham, r_up, r_dn)
    assert G.shape[0] > 1 and not np.iscomplexobj(G)
    singular_values = np.linalg.svd(G, compute_uv=False)
    assert np.all(singular_values > 0)
    min_norm = min(np.sum(G**2, axis=0).min(), np.sum(G**2, axis=1).min())
    initial_factor = (min_norm * np.sum(1 / singular_values**2)) ** (-3.0 / 8.0)
    assert 0 < initial_factor < epsilon_as

    xla = _xla_sweep(ham, r_up, r_dn, key, num_moves, Dt, epsilon_as, G, G_inv, det_state, jastrow_state)
    inactive = _xla_sweep(ham, r_up, r_dn, key, num_moves, Dt, 0.0, G, G_inv, det_state, jastrow_state)
    assert xla["accepted"] > 1
    assert xla["accepted"] != inactive["accepted"], "the AS guide must change the reference decisions"
    rust = _rs_sweep(ham, r_up, r_dn, key, num_moves, Dt, epsilon_as, G, G_inv)
    _assert_transition_parity(xla, rust, "active real AS sweep", complex_path=False)


# --------------------------------------------------------------------------- #
# 1. Multi-sweep parity over every supported orbital x Jastrow leaf combination.
# --------------------------------------------------------------------------- #
@pytest.mark.light
@pytest.mark.parametrize("pattern", ALL_PATTERNS)
@pytest.mark.parametrize("seed", [0, 5])
def test_multi_sweep_parity(pattern, seed):
    """Charter exit criterion: multi-sweep parity across all supported orbital /
    Jastrow leaf combinations. Three consecutive sweeps, each advancing BOTH
    engines from their own carried state (key, positions, geminal, SM inverse,
    streaming caches), with every channel compared after every sweep.
    """
    ham = _ham(pattern)
    r_up, r_dn = _draw_cfg(ham, seed)
    num_moves, Dt = 6, 1.2
    key = random.PRNGKey(seed)
    x_up, x_dn = r_up, r_dn
    s_up, s_dn = r_up.copy(), r_dn.copy()
    for i_sweep in range(3):
        G, G_inv, det_state, jastrow_state = _init_running_state(ham, x_up, x_dn)
        complex_path = bool(np.iscomplexobj(G))
        xla = _xla_sweep(ham, x_up, x_dn, key, num_moves, Dt, 1.0e-6, G, G_inv, det_state, jastrow_state)
        # Rust advances from ITS OWN carried positions (s_up/s_dn), so a divergence
        # accumulated in an earlier sweep cannot be laundered by re-seeding.
        sG, sG_inv, _, _ = _init_running_state(ham, s_up, s_dn)
        rs = _rs_sweep(ham, s_up, s_dn, key, num_moves, Dt, 1.0e-6, sG, sG_inv)
        _assert_transition_parity(xla, rs, f"{pattern}/sweep={i_sweep}", complex_path=complex_path)
        x_up, x_dn = xla["r_up"], xla["r_dn"]
        s_up, s_dn = np.ascontiguousarray(rs["r_up"]), np.ascontiguousarray(rs["r_dn"])
        key = xla["key"]


# --------------------------------------------------------------------------- #
# 2. Move-by-move transition: compare after EVERY single-electron move.
# --------------------------------------------------------------------------- #
@pytest.mark.light
@pytest.mark.parametrize("pattern", ALL_PATTERNS)
def test_move_by_move_transition(pattern):
    """The state-machine check proper: ten single-electron moves, compared after
    EACH one. A wrong proposal / drift / Green's ratio / accept decision / rank-1
    update on move k shows up at move k rather than being averaged away by the
    remaining moves (not only the final configuration).
    """
    ham = _ham(pattern)
    r_up, r_dn = _draw_cfg(ham, 2)
    Dt = 1.2
    key = random.PRNGKey(11)
    x_up, x_dn = r_up, r_dn
    s_up, s_dn = r_up.copy(), r_dn.copy()
    n_accept = 0
    for mv in range(10):
        G, G_inv, det_state, jastrow_state = _init_running_state(ham, x_up, x_dn)
        complex_path = bool(np.iscomplexobj(G))
        xla = _xla_sweep(ham, x_up, x_dn, key, 1, Dt, 1.0e-6, G, G_inv, det_state, jastrow_state)
        sG, sG_inv, _, _ = _init_running_state(ham, s_up, s_dn)
        rs = _rs_sweep(ham, s_up, s_dn, key, 1, Dt, 1.0e-6, sG, sG_inv)
        _assert_transition_parity(xla, rs, f"{pattern}/move={mv}", complex_path=complex_path)
        n_accept += xla["accepted"]
        x_up, x_dn = xla["r_up"], xla["r_dn"]
        s_up, s_dn = np.ascontiguousarray(rs["r_up"]), np.ascontiguousarray(rs["r_dn"])
        key = xla["key"]
    # Guard against a vacuous pass: a chain that accepted nothing would compare
    # ten identical no-op transitions.
    assert n_accept > 0, f"{pattern}: no move was accepted in 10 -- transition cells not exercised"


# --------------------------------------------------------------------------- #
# 3. Forced-accept and forced-reject cells.
# --------------------------------------------------------------------------- #
@pytest.mark.light
@pytest.mark.parametrize("pattern", ALL_PATTERNS)
def test_forced_accept(pattern):
    """FORCED-ACCEPT cell: with a vanishing drift step the proposal is
    infinitesimal, so ``T_ratio -> 1`` and ``R_ratio -> 1``, and the Metropolis
    test accepts unconditionally (``b < 1`` a.s.). Both engines must take the
    accept branch on EVERY move -- i.e. the accepted-branch rank-1 inverse update
    and streaming advance are the ones being compared, never the reject
    passthrough.
    """
    ham = _ham(pattern)
    r_up, r_dn = _draw_cfg(ham, 4)
    num_moves = 8
    # Drift step small enough that |r_new - r_old| is far below any WF length
    # scale; derived from the move formula (sigma = f_l * Dt), not tuned.
    Dt_tiny = 1.0e-6
    xla, rs = _run_and_compare(pattern, r_up, r_dn, seed=13, num_moves=num_moves, Dt=Dt_tiny, tag=f"{pattern}/forced-accept")
    assert xla["accepted"] == num_moves, f"{pattern}: forced-accept cell not reached (XLA accepted {xla['accepted']})"
    assert int(rs["accepted"]) == num_moves, f"{pattern}: Rust did not accept all (accepted {rs['accepted']})"


@pytest.mark.light
@pytest.mark.parametrize("pattern", MULTI_ELECTRON_PATTERNS)
def test_forced_reject(pattern):
    """FORCED-REJECT cell: with a large drift step every proposal lands far out in
    the wavefunction tail, ``R_ratio`` collapses and the Metropolis test rejects.
    Both engines must take the reject branch on EVERY move -- i.e. the rejected
    passthrough must leave positions, geminal, SM inverse and streaming caches
    untouched IDENTICALLY on both sides.

    Only the many-electron molecular rows can reach an all-reject sweep: the
    committed 1up/1dn periodic fixtures carry a delocalised Bloch orbital whose
    ratio stays O(1) at any displacement, so they accept ~always regardless of Dt
    (measured).

    Seeds are scanned rather than a single (Dt, seed) magic pair being hardcoded;
    the sweep asserts parity on EVERY seed and that at least one reached the cell.
    """
    ham = _ham(pattern)
    Dt_large = 8.0
    num_moves = 10
    reached = False
    for seed in range(6):
        r_up, r_dn = _draw_cfg(ham, seed)
        xla, rs = _run_and_compare(
            pattern, r_up, r_dn, seed=seed + 100, num_moves=num_moves, Dt=Dt_large, tag=f"{pattern}/forced-reject/s{seed}"
        )
        # A sweep whose running inverse blew up is the NaN-acceptance cell pinned
        # separately below, not a forced-reject cell; do not fold the two.
        if not np.all(np.isfinite(xla["geminal_inv"])):
            continue
        if xla["accepted"] == 0:
            reached = True
            assert int(rs["accepted"]) == 0, f"{pattern}: XLA rejected all, Rust accepted {rs['accepted']}"
            # Reject passthrough must be an exact no-op on the walker.
            assert np.allclose(xla["r_up"], r_up, atol=ATOL_BIT_EXACT, rtol=RTOL_BIT_EXACT)
            assert np.allclose(np.asarray(rs["r_up"]), r_up, atol=ATOL_BIT_EXACT, rtol=RTOL_BIT_EXACT)
    assert reached, f"{pattern}: no seed reached an all-reject sweep at Dt={Dt_large}"


# --------------------------------------------------------------------------- #
# 4. Spin-up / spin-down and first / last electron index cells.
# --------------------------------------------------------------------------- #
@pytest.mark.light
@pytest.mark.parametrize("pattern", MULTI_ELECTRON_PATTERNS)
def test_spin_and_edge_electron_index_cells(pattern):
    """Charter cells: spin-up AND spin-down, FIRST and LAST electron index.

    ``jqmc_rs._move_draws`` exposes the Rust electron-selection draws; the same
    selection is re-derived from ``jax.random`` (the XLA RNG primitives
    ``body_fun`` itself calls, in ``body_fun``'s split order) and the two are
    required to agree. Seeds are enumerated until all four cells (up/dn x
    first/last) are hit, and each hit seed's FULL one-move transition is compared
    XLA-vs-Rust, so the cell is not merely selected but verified.
    """
    ham = _ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    n_up, n_dn = int(gem.num_electron_up), int(gem.num_electron_dn)
    assert n_up > 1 and n_dn > 1, f"{pattern}: fixture cannot express a first/last index distinction"
    r_up, r_dn = _draw_cfg(ham, 3)
    wanted = {(True, 0), (True, n_up - 1), (False, 0), (False, n_dn - 1)}
    seen = set()
    for seed in range(400):
        key = random.PRNGKey(seed)
        # XLA-side selection, straight from body_fun's split order.
        k1, sk = random.split(key)
        rand_num = int(random.randint(sk, (), 0, n_up + n_dn))
        is_up = bool(rand_num < n_up)
        k1, sk = random.split(k1)
        up_index = int(random.randint(sk, (), 0, n_up))
        k1, sk = random.split(k1)
        dn_index = int(random.randint(sk, (), 0, n_dn))
        sel = up_index if is_up else dn_index
        # Rust-side selection from the same key words.
        # ``_move_draws`` -> (is_up, selected_index, axis, metropolis_b, key0, key1).
        kw = np.asarray(key, dtype=np.uint32)
        rs_is_up, rs_sel, _rs_axis, _rs_b, _k0, _k1 = jqmc_rs._move_draws(int(kw[0]), int(kw[1]), n_up, n_dn)
        assert bool(rs_is_up) == is_up, f"{pattern}/seed={seed}: is_up XLA={is_up} Rust={rs_is_up}"
        assert int(rs_sel) == sel, f"{pattern}/seed={seed}: index XLA={sel} Rust={rs_sel}"
        cell = (is_up, sel)
        if cell in wanted and cell not in seen:
            seen.add(cell)
            _run_and_compare(pattern, r_up, r_dn, seed=seed, num_moves=1, Dt=1.2, tag=f"{pattern}/spin={is_up}/idx={sel}")
        if seen == wanted:
            break
    assert seen == wanted, f"{pattern}: uncovered selection cells {sorted(wanted - seen)}"


# --------------------------------------------------------------------------- #
# 5. Near-node configurations.
# --------------------------------------------------------------------------- #
@pytest.mark.light
@pytest.mark.parametrize("pattern", MULTI_ELECTRON_PATTERNS)
@pytest.mark.parametrize("gap", _NEAR_NODE_GAPS)
def test_near_node_configuration(pattern, gap):
    """NEAR-NODE cell: two same-spin electrons driven nearly coincident make two
    geminal rows nearly identical, so ``det(G) -> 0`` and the Sherman-Morrison
    inverse sits on the edge of its conditioning. This is where a rank-1 update
    that is algebraically equivalent but differently ordered would part company,
    and where the AS regularisation (``epsilon_AS``) engages.

    Mixed precision stops at gap=1e-4: at gap=1e-6 the fp32 AO rounding changes
    the discrete Metropolis branch, which no numerical tolerance can reconcile.
    Full precision retains both depths.  The cell is asserted to actually BE
    near-node (conditioning check), so a
    fixture change that quietly de-degenerates it fails loudly instead of
    silently turning this into an ordinary configuration.
    """
    ham = _ham(pattern)
    r_up, r_dn = _draw_cfg(ham, 6)
    r_up = r_up.copy()
    r_up[1] = r_up[0] + gap
    G, _, _, _ = _init_running_state(ham, r_up, r_dn)
    cond = float(np.linalg.cond(G))
    baseline = float(np.linalg.cond(_init_running_state(ham, *_draw_cfg(ham, 6))[0]))
    assert cond > baseline, f"{pattern}: near-node cell not degenerate (cond={cond:.3e} vs baseline {baseline:.3e})"
    _run_and_compare(pattern, r_up, r_dn, seed=21, num_moves=6, Dt=1.2, tag=f"{pattern}/near-node/gap={gap}")


# --------------------------------------------------------------------------- #
# 6. Far / unwrapped PBC positions.
# --------------------------------------------------------------------------- #
@pytest.mark.light
@pytest.mark.parametrize("pattern", PERIODIC_PATTERNS)
@pytest.mark.parametrize("shift", [(3, -2, 1), (-7, 5, -4)])
def test_far_unwrapped_pbc_positions(pattern, shift):
    """FAR / UNWRAPPED PBC cell: walker positions are carried UNWRAPPED across the
    Python<->Rust boundary by design (jqmc_mcmc.py L2104: the complex Bloch orb
    wraps internally and the restore-phase factor reads the unwrapped position).
    Translating every electron by whole lattice vectors must therefore leave the
    entire transition -- proposal, min-image Kato drift, ratios, accept decision,
    rank-1 update and streaming caches -- in lock-step between the two engines,
    with positions still reported unwrapped.

    This is the cell that catches a home-cell wrap sneaking into either engine's
    nearest-atom / AO path, and far-image errors in either engine.
    """
    ham = _ham(pattern)
    r_up, r_dn = _draw_cfg(ham, 8)
    A = np.asarray(ham.structure_data.cell, dtype=np.float64)
    offset = np.asarray(shift, dtype=np.float64) @ A
    r_up_far = np.ascontiguousarray(r_up + offset)
    r_dn_far = np.ascontiguousarray(r_dn + offset)
    xla, rs = _run_and_compare(pattern, r_up_far, r_dn_far, seed=31, num_moves=6, Dt=1.2, tag=f"{pattern}/far-pbc/{shift}")
    # Positions must come back UNWRAPPED (still out near the shifted image), not
    # folded into the home cell by either side.
    pos_atol, pos_rtol = get_tolerance_xla_rust_consistency(_POS_ZONES, level="medium")
    assert np.allclose(np.asarray(rs["r_up"]) - r_up_far, xla["r_up"] - r_up_far, atol=pos_atol, rtol=pos_rtol)
    assert np.abs(np.asarray(rs["r_up"])).max() > np.abs(A).max(), (
        f"{pattern}: Rust wrapped the walker into the home cell (far-image convention broken)"
    )


# --------------------------------------------------------------------------- #
# 7. Multiple walkers.
# --------------------------------------------------------------------------- #
@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_sd_j123", "mol_ecp_sphe_j12", "pbc_gamma_sphepgto_j123", "twist_sphepgto_j12"])
def test_multiple_walkers(pattern):
    """MULTI-WALKER cell: the driver vmaps XLA over walkers but loops Rust over
    them in Python (jqmc_mcmc.py L2124-2160), each walker carrying its OWN
    threefry key. Walkers must stay independent and IN ORDER across that
    structural difference -- a shared/transposed key or a walker-order swap is
    exactly what this cell catches, and it is invisible to any single-walker
    test. Walkers are given distinct keys AND distinct configurations so an
    order swap cannot be self-cancelling.
    """
    ham = _ham(pattern)
    nw = 4
    cfgs = [_draw_cfg(ham, 40 + w) for w in range(nw)]
    keys = [random.PRNGKey(500 + w) for w in range(nw)]
    num_moves, Dt = 6, 1.2

    # XLA: one vmapped call over the walker batch (the production JAX lane).
    states = [_init_running_state(ham, ru, rd) for ru, rd in cfgs]
    complex_path = bool(np.iscomplexobj(states[0][0]))
    batch_up = jnp.asarray(np.stack([c[0] for c in cfgs]))
    batch_dn = jnp.asarray(np.stack([c[1] for c in cfgs]))
    batch_key = jnp.stack(keys)
    batch_G = jnp.asarray(np.stack([s[0] for s in states]))
    batch_G_inv = jnp.asarray(np.stack([s[1] for s in states]))
    batch_det = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[s[2] for s in states])
    batch_jas = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[s[3] for s in states])
    # The production vmapped driver entry point itself (jqmc_mcmc._jit_vmap_update).
    (b_acc, _b_rej, b_up, b_dn, b_key, b_G_inv, b_G, b_det, _b_jas) = _jit_vmap_update(
        batch_up, batch_dn, batch_key, num_moves, ham, Dt, 1.0e-6, batch_G_inv, batch_G, batch_det, batch_jas
    )

    for w in range(nw):
        xla = dict(
            accepted=int(b_acc[w]),
            r_up=np.asarray(b_up[w]),
            r_dn=np.asarray(b_dn[w]),
            key=b_key[w],
            geminal=np.asarray(b_G[w]),
            geminal_inv=np.asarray(b_G_inv[w]),
            ao_up=np.asarray(b_det.ao_up[w]),
            ao_dn=np.asarray(b_det.ao_dn[w]),
            paired_dn=np.asarray(b_det.paired_dn[w]),
            paired_up_lambda=np.asarray(b_det.paired_up_lambda[w]),
        )
        rs = _rs_sweep(ham, cfgs[w][0], cfgs[w][1], keys[w], num_moves, Dt, 1.0e-6, states[w][0], states[w][1])
        _assert_transition_parity(xla, rs, f"{pattern}/walker={w}", complex_path=complex_path)

    # Walkers really are distinct chains (otherwise the per-walker comparison
    # above would be four copies of one cell).
    assert len({int(b_acc[w]) for w in range(nw)}) > 1 or not np.allclose(
        np.asarray(b_up[0]), np.asarray(b_up[1]), atol=ATOL_BIT_EXACT, rtol=RTOL_BIT_EXACT
    ), f"{pattern}: walkers are not independent chains"


# --------------------------------------------------------------------------- #
# 8. Rust-internal running state: no drift vs a from-scratch rebuild.
# --------------------------------------------------------------------------- #
@pytest.mark.light
@pytest.mark.parametrize("pattern", ALL_PATTERNS)
def test_rs_streaming_cache_no_drift(pattern):
    """The Jastrow (J3) and det running states are advanced INSIDE the Rust sweep
    and are not all exported, so they are pinned here against a from-scratch
    rebuild: one N-move sweep must equal N chained 1-move sweeps, where each link
    re-initialises the geminal, its inverse and (inside Rust) every streaming
    cache from positions alone. Rank-1 drift, a stale J3 table or a cache that is
    advanced on a rejected move all break this identity while leaving the
    single-sweep comparison against XLA potentially intact.
    """
    ham = _ham(pattern)
    r_up, r_dn = _draw_cfg(ham, 9)
    num_moves, Dt = 8, 1.2
    key = random.PRNGKey(77)
    G, G_inv, _, _ = _init_running_state(ham, r_up, r_dn)
    one_shot = _rs_sweep(ham, r_up, r_dn, key, num_moves, Dt, 1.0e-6, G, G_inv)

    c_up, c_dn, c_key, acc = r_up.copy(), r_dn.copy(), key, 0
    for _ in range(num_moves):
        cG, cG_inv, _, _ = _init_running_state(ham, c_up, c_dn)
        step = _rs_sweep(ham, c_up, c_dn, c_key, 1, Dt, 1.0e-6, cG, cG_inv)
        acc += int(step["accepted"])
        c_up = np.ascontiguousarray(step["r_up"])
        c_dn = np.ascontiguousarray(step["r_dn"])
        c_key = jnp.asarray(np.asarray(step["final_key"], dtype=np.uint32))

    # Reference inverse rebuilt FROM SCRATCH at the final chained configuration --
    # never a rank-1-propagated one, so this really is streamed-vs-recomputed.
    _, final_G_inv, _, _ = _init_running_state(ham, c_up, c_dn)

    pos_atol, pos_rtol = get_tolerance_xla_rust_consistency(_POS_ZONES, level="medium")
    gem_atol, gem_rtol = get_tolerance_xla_rust_consistency(_GEM_ZONES, level="medium")
    assert int(one_shot["accepted"]) == acc, f"{pattern}: streamed accept {one_shot['accepted']} vs rebuilt {acc}"
    assert np.allclose(one_shot["r_up"], c_up, atol=pos_atol, rtol=pos_rtol)
    assert np.allclose(one_shot["r_dn"], c_dn, atol=pos_atol, rtol=pos_rtol)
    assert np.allclose(one_shot["geminal_inv"], final_G_inv, atol=gem_atol, rtol=gem_rtol), (
        f"{pattern}: streamed SM inverse drifted from the rebuilt inverse "
        f"max|d|={np.abs(np.asarray(one_shot['geminal_inv']) - final_G_inv).max():.3e}"
    )


# --------------------------------------------------------------------------- #
# 9. Mutation check.
# --------------------------------------------------------------------------- #
@pytest.mark.light
@pytest.mark.parametrize(
    "channel",
    ["geminal_inv", "geminal", "injected_normals"],
)
def test_mutation_check_corrupt_running_state_turns_red(channel):
    """CHARTER EXIT CRITERION: deliberately corrupting the cached / running state
    handed to Rust MUST turn these tests red.

    Each channel corrupts exactly one piece of the state machine on the RUST side
    only, leaving XLA on the pristine state, and asserts that
    ``_assert_transition_parity`` -- the very helper every test above uses --
    raises. This proves the comparisons are load-bearing rather than
    vacuously-passing shape checks:

      * ``geminal_inv``   -- the Sherman-Morrison running inverse (the central
        cached quantity of the transition).
      * ``geminal``       -- the rank-1-updated geminal matrix.
      * ``injected_normals`` -- the injected randomness itself; corrupting it must
        desynchronise the two chains, which pins that the injection path is
        actually consumed (a Rust side that re-derived its own normals would pass
        the other tests and fail to be caught by anything else here).

    The un-corrupted control at the end re-runs the SAME configuration and must
    pass, so a green result cannot come from a permanently-broken cell.
    """
    pattern = "mol_cart_sd_j123"
    ham = _ham(pattern)
    r_up, r_dn = _draw_cfg(ham, 5)
    kwargs = dict(seed=17, num_moves=6, Dt=1.2)

    def _corrupt(G, G_inv, key, num_moves):
        normals = None
        if channel == "geminal_inv":
            G_inv[0, 0] *= 1.0 + _MUTATION_REL
        elif channel == "geminal":
            G[0, 0] *= 1.0 + _MUTATION_REL
        elif channel == "injected_normals":
            # Perturb EVERY injected proposal, not just the first: corrupting only
            # move 0 is invisible whenever move 0 happens to be rejected.
            normals = np.array(_draw_injected_normals(key, num_moves), dtype=np.float64)
            normals *= 1.0 + _MUTATION_REL
        return G, G_inv, normals

    with pytest.raises(AssertionError):
        _run_and_compare(pattern, r_up, r_dn, corrupt=_corrupt, tag=f"{pattern}/mutated:{channel}", **kwargs)

    # Control: the identical cell with the state RESTORED must pass.
    _run_and_compare(pattern, r_up, r_dn, tag=f"{pattern}/mutation-control", **kwargs)


# --------------------------------------------------------------------------- #
# 10. XLA-vs-Rust discrepancies (pinned, NOT worked around).
# --------------------------------------------------------------------------- #
@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_sd_j123", "mol_cart_sd_jdtn"])
def test_single_move_includes_j3_and_dtn_jastrow(pattern):
    """``jqmc_rs._single_move``'s Metropolis ratio must match XLA's on a
    wavefunction carrying a J3 or DTN Jastrow leaf.

    The oracle is the production sweep restricted to one move -- ``_mcmc_sweep``
    with ``num_moves=1``, which this module has already pinned against XLA
    move-by-move for these very patterns. So this compares Rust against Rust
    against XLA: whatever ``_single_move`` returns must agree with the sweep that
    agrees with XLA. Minimal reproducer for case A.
    """
    ham = _ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    jas = ham.wavefunction_data.jastrow_data
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    Z = _charges(ham)
    Dt, eps = 1.2, 1.0e-6
    atol, rtol = get_tolerance_min(_GEM_ZONES, level="strict")

    # Scan configurations/keys: the dropped Jastrow channel perturbs r_ratio on
    # EVERY move, but only flips the accept/reject decision when the perturbed
    # ratio crosses the Metropolis ``b`` -- so a single hand-picked seed would be
    # a flaky reproducer. Every cell is asserted; the first divergence fails.
    for cfg_seed in range(12):
        r_up, r_dn = _draw_cfg(ham, cfg_seed)
        G, G_inv, _, _ = _init_running_state(ham, r_up, r_dn)
        for key_seed in range(12):
            configuration_key_label = f"{cfg_seed}/{key_seed}"
            key = random.PRNGKey(key_seed)
            kw = np.asarray(key, dtype=np.uint32)
            normals = np.asarray(_draw_injected_normals(key, 1), dtype=np.float64)
            single = jqmc_rs._single_move(
                gem, jas, R, Z, r_up, r_dn, G, G_inv, Dt, eps, int(kw[0]), int(kw[1]), float(normals[0])
            )
            sweep = jqmc_rs._mcmc_sweep(gem, jas, R, Z, r_up, r_dn, G, G_inv, Dt, eps, int(kw[0]), int(kw[1]), normals)
            tag = f"{pattern}/cfg={cfg_seed}/key={key_seed}"
            assert int(single["accepted"]) == int(sweep["accepted"]), (
                f"{tag}: _single_move accept={single['accepted']} vs XLA-verified sweep "
                f"accept={sweep['accepted']} (_single_move r_ratio={single['r_ratio']:.9e})"
            )
            assert np.allclose(np.asarray(single["r_up"]), np.asarray(sweep["r_up"]), atol=atol, rtol=rtol), (
                f"{tag}: _single_move r_up disagrees with the XLA-verified _mcmc_sweep"
            )
            assert np.allclose(np.asarray(single["geminal_inv"]), np.asarray(sweep["geminal_inv"]), atol=atol, rtol=rtol), (
                f"{tag}: _single_move geminal_inv disagrees with the XLA-verified _mcmc_sweep"
            )


@pytest.mark.light
def test_nan_acceptance_ratio_rejects_like_xla():
    """When the acceptance ratio degenerates to NaN, Rust must make the SAME
    accept/reject decision as XLA: REJECT (case B).

    XLA: ``jnp.min([1.0, NaN])`` propagates the NaN and ``b < NaN`` is False.
    A bare Rust ``f64::min`` is IEEE minNum: it absorbs the NaN
    (``NaN.min(1.0) == 1.0``) and accepts UNCONDITIONALLY, marching the walker
    off while XLA keeps it at its last valid configuration.

    The NaN cell is INJECTED deterministically: a running inverse poisoned with
    NaN makes every SM determinant ratio NaN, hence every acceptance ratio NaN,
    on both engines, for every move. A drift-based reproducer does not work:
    (a) its precondition (a degenerate XLA inverse) is not reached at Dt=25/n=10;
    (b) at Dt large enough to degenerate (60), the diverging decision is NOT the
    NaN cell but an AS-regularization fp divergence: at |Det_T_ratio| ~ 3e-14 the streaming
    inverse update amplifies fp noise by ~1/ratio ~ 3e13, and the AS factor
    computed FROM that amplified state (R_AS_p ~ 1e-17..1e-20) differs between
    engines by orders of magnitude, so eps/R_AS_p flips accept/reject. That
    cell is not bit-specifiable for ANY two implementations (same formula, ULP
    parity already pinned at cond ~3.4e6 by the multi-sweep rows); only the NaN
    contract below is.
    """
    pattern = "mol_cart_sd_jnone"  # Jastrow-independent: the same NaN algebra at j12/j123.
    ham = _ham(pattern)
    r_up, r_dn = _draw_cfg(ham, 11)
    num_moves = 10
    key = random.PRNGKey(3)
    G, G_inv, det_state, jastrow_state = _init_running_state(ham, r_up, r_dn)
    # Poison the running inverse: every SM ratio -> NaN -> acceptance ratio NaN.
    G_inv_poison = np.asarray(G_inv, dtype=np.float64).copy()
    G_inv_poison[0, 0] = np.nan
    xla = _xla_sweep(ham, r_up, r_dn, key, num_moves, 2.0, 1.0e-6, G, G_inv_poison, det_state, jastrow_state)
    rs = _rs_sweep(ham, r_up, r_dn, key, num_moves, 2.0, 1.0e-6, G, G_inv_poison)
    # Precondition: the poisoned cell is real -- XLA rejects EVERY move.
    assert int(xla["accepted"]) == 0, (
        f"precondition: XLA accepted {int(xla['accepted'])} of {num_moves} on a NaN-poisoned inverse"
    )
    assert int(rs["accepted"]) == 0, (
        f"NaN acceptance ratio must REJECT like XLA: Rust accepted {int(rs['accepted'])} of {num_moves}"
    )
    # Rejection is a no-op on the walker: positions bit-identical to the input.
    np.testing.assert_array_equal(np.asarray(rs["r_up"]), np.asarray(r_up))
    np.testing.assert_array_equal(np.asarray(rs["r_dn"]), np.asarray(r_dn))
