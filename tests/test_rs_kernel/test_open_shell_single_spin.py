"""Production parity for single-occupied-spin-channel
(``num_electron_dn == 0``) molecular AE/ECP systems, Rust vs jqmc.

Coverage note: the committed trexio fixtures ``H_ae_ccpvdz_cart.h5`` /
``H_ecp_ccpvdz_cart.h5`` (a bare hydrogen atom, ``n_up=1, n_dn=0`` -- one spin
channel entirely EMPTY) are exercised on the XLA side (``test_jqmc_mcmc.py``,
``test_determinant.py``) but were never wired into ``tests/test_rs_kernel``: every
existing rs-kernel fixture (``_rs_fixtures._PATTERNS``) has BOTH spin channels
populated (``N_ae_ccpvdz_cart.h5`` is the only open-shell row, but its down
channel still has 2 electrons). An empty-spin determinant is a structurally
distinct case (a (0, n_dn_orb) down-orbital MO block / an (0, 0) contribution to
the geminal matrix / a plain scalar down "ln-det" branch), not merely a smaller
system on the SAME code path, so it needed its own probe rather than a
same-path merge. This module builds that ham via the shared
``_build_ham_for_cell`` (single source of truth for committed-fixture ->
Hamiltonian_data, ``tests/test_jqmc_kernel/_generated_matrix_base.py``) and
checks the value-level chain any production e_L observable is assembled from:
ln-det grad/Laplacian, the assembled kinetic energy, and the aggregate
``compute_local_energy_fast`` (AE and ECP, jnone/J1+J2/J1+J2+J3).

State-transition and non-local ECP extension
--------------------------------------------
The value rows cover THREE value quantities on the two H atoms. This extension
adds the two axes not otherwise covered on any single-spin cell:

* **The MCMC state-transition** (``test_single_spin_mcmc_transition``). The bare
  ``n_dn == 0`` walker takes the driver's dedicated ONLY-UP branch
  (``jqmc_mcmc._update_electron_positions_only_up_electron`` /
  ``_jit_vmap_update_up``, selected by ``__only_up_electron`` -- the two-spin
  ``_update_electron_positions`` gathers ``r_dn_carts[idx]`` and cannot even
  trace at ``n_dn == 0``), while the production Rust replacement drives the SAME
  ``_mcmc_sweep``. That XLA-vs-Rust transition was untested for an empty spin
  channel. The oracle here is the production only-up kernel itself, on the
  codebase's own bit-exact injected-normals path (the only-up ``body_fun`` keeps
  the same three ``jax.random.split`` calls -- two are dummies -- so
  ``_draw_injected_normals`` replays the identical stream).

* **A GENUINE non-local ECP channel** (``Li_ecp_ccpvdz_cart.h5``, added this
  pass). ``H_ecp``'s ccECP has ``z_core = 0`` and a zero l=0 projector
  coefficient, so its non-local energy is IDENTICALLY zero -- every "single-spin
  x ECP" row above is vacuous on the non-local path. A neutral Li atom under
  ccECP keeps a single valence electron (2 core electrons removed ->
  ``n_up=1, n_dn=0``) AND a real l=0 projector, so it is the smallest committed
  fixture that is BOTH single-spin AND exercises ``V_nl != 0``
  (``test_single_spin_ecp_nonlocal_is_active`` pins that H's channel is exactly
  0 while Li's is ~5 Ha, so Li's ``compute_local_energy_fast`` parity is not a
  vacuous pass).

This is production-reachable: ``jqmc_mcmc.py``'s ``flag_rust_on_CPU`` dispatch
gates on CPU + full-precision only, with
NO leaf-type or electron-count check, so a spin-polarized single-electron run
under ``flag_rust_on_CPU=True`` unconditionally routes through these same Rust
kernels.
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

pytestmark = pytest.mark.with_chdir

import jax.numpy as jnp
from jax import random

from jqmc._precision import get_tolerance_min, get_tolerance_xla_rust_consistency, precision_policy_snapshot
from jqmc._setting import ATOL_BIT_EXACT, ATOL_REDUCTION_ORDER, RTOL_BIT_EXACT, RTOL_REDUCTION_ORDER
from jqmc.coulomb_potential import compute_ecp_non_local_parts_all_pairs, icosahedron_sym_mesh_Nv12
from jqmc.determinant import (
    SD_data,
    _init_det_ratio_streaming_state,
    compute_geminal_all_elements,
    compute_grads_and_laplacian_ln_Det,
    to_AGP,
)
from jqmc.hamiltonians import compute_local_energy_fast
from jqmc.jastrow_factor import Jastrow_ratio_state, _init_jastrow_ratio_state
from jqmc.jqmc_mcmc import _draw_injected_normals, _update_electron_positions_only_up_electron
from jqmc.molecular_orbital import compute_MOs
from jqmc.wavefunction import compute_kinetic_energy

from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

_ZONES_LNDET = ("ao_grad_lap", "det_grad_lap")
_ZONES_KINETIC = ("wf_kinetic", "ao_grad_lap", "det_grad_lap", "jastrow_grad_lap")
_ZONES_EL = ("wf_kinetic", "ao_eval", "det_eval", "coulomb", "jastrow_grad_lap")
_ZONES_ECP_NONLOCAL = ("coulomb", "det_ratio", "jastrow_ratio")
# Transition channels: positions are a copy of the proposal arithmetic; the
# geminal + Sherman-Morrison inverse carry the det zones (see test_mcmc_transition).
_ZONES_POS = ("det_ratio",)
_ZONES_GEM = ("det_ratio", "det_eval")
_MESH = icosahedron_sym_mesh_Nv12

# stem -> (with_jastrow, spec); jnone / J1+J2 / J1+J2+J3(cart-gto leaf), mirroring
# the _rs_fixtures molecular-cart naming convention (mol_cart_sd_{jnone,j12,j123}).
# ``Li_ecp`` rows add the only single-spin cell with a non-vacuous
# non-local ECP channel (H ccECP has z_core=0 -> V_nl == 0).
_CASES = [
    ("H_ae_ccpvdz_cart.h5", False, None, "ae_jnone"),
    ("H_ae_ccpvdz_cart.h5", True, None, "ae_j12"),
    ("H_ae_ccpvdz_cart.h5", True, {"j3": "cart_gto"}, "ae_j123"),
    ("H_ecp_ccpvdz_cart.h5", False, None, "ecp_jnone"),
    ("H_ecp_ccpvdz_cart.h5", True, None, "ecp_j12"),
    ("H_ecp_ccpvdz_cart.h5", True, {"j3": "cart_gto"}, "ecp_j123"),
    ("Li_ecp_ccpvdz_cart.h5", False, None, "li_ecp_jnone"),
    ("Li_ecp_ccpvdz_cart.h5", True, None, "li_ecp_j12"),
]


def _build(stem, with_jastrow, spec):
    ham = _build_ham_for_cell(stem, with_jastrow, spec, "SD")
    gd = ham.wavefunction_data.geminal_data
    assert int(gd.num_electron_up) == 1
    assert int(gd.num_electron_dn) == 0
    return ham


def _config(ham, seed):
    gd = ham.wavefunction_data.geminal_data
    rng = np.random.default_rng(seed)
    r_up = rng.normal(scale=0.6, size=(int(gd.num_electron_up), 3))
    r_dn = np.zeros((int(gd.num_electron_dn), 3), dtype=np.float64)
    return r_up, r_dn


@pytest.mark.light
@pytest.mark.parametrize("stem, with_jastrow, spec, tag", _CASES, ids=[c[3] for c in _CASES])
def test_grads_laplacian_ln_det_single_spin(stem, with_jastrow, spec, tag):
    """Grads laplacian ln det single spin."""
    ham = _build(stem, with_jastrow, spec)
    gd = ham.wavefunction_data.geminal_data
    r_up, r_dn = _config(ham, seed=0)
    gu, gdn, lu, ldn = jqmc_rs.compute_grads_and_laplacian_ln_det(gd, r_up, r_dn)
    Gu, Gdn, Lu, Ldn = (np.asarray(x, float) for x in compute_grads_and_laplacian_ln_Det(gd, r_up, r_dn))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_LNDET, level="strict")
    assert np.asarray(gdn).shape == (0, 3) == Gdn.shape
    assert np.asarray(ldn).shape == (0,) == Ldn.shape
    assert np.allclose(gu, Gu, atol=atol, rtol=rtol), np.max(np.abs(gu - Gu))
    assert np.allclose(lu, Lu, atol=atol, rtol=rtol), np.max(np.abs(lu - Lu))


@pytest.mark.light
@pytest.mark.parametrize("stem, with_jastrow, spec, tag", _CASES, ids=[c[3] for c in _CASES])
def test_kinetic_energy_single_spin(stem, with_jastrow, spec, tag):
    """Kinetic energy single spin."""
    ham = _build(stem, with_jastrow, spec)
    wf = ham.wavefunction_data
    r_up, r_dn = _config(ham, seed=1)
    k_rust = jqmc_rs.compute_kinetic_energy(wf.geminal_data, wf.jastrow_data, r_up, r_dn)
    k_jqmc = float(compute_kinetic_energy(wf, r_up, r_dn))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_KINETIC, level="strict")
    assert np.isclose(k_rust, k_jqmc, rtol=rtol, atol=atol), (k_rust, k_jqmc, abs(k_rust - k_jqmc))


@pytest.mark.light
@pytest.mark.parametrize("stem, with_jastrow, spec, tag", _CASES, ids=[c[3] for c in _CASES])
def test_local_energy_fast_single_spin(stem, with_jastrow, spec, tag):
    """Local energy fast single spin."""
    ham = _build(stem, with_jastrow, spec)
    gd = ham.wavefunction_data.geminal_data
    r_up, r_dn = _config(ham, seed=2)
    RT = np.eye(3)
    w = np.asarray(_MESH.weights, float)
    gp = np.asarray(_MESH.grid_points, float) @ RT
    g_inv = np.linalg.inv(np.asarray(compute_geminal_all_elements(gd, r_up, r_dn)))
    native = float(jqmc_rs.compute_local_energy_fast(ham, r_up, r_dn, w, gp, g_inv))
    jax_eL = float(compute_local_energy_fast(ham, r_up, r_dn, jnp.asarray(RT), jnp.asarray(g_inv), None))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="strict")
    assert np.isclose(native, jax_eL, rtol=rtol, atol=atol), (native, jax_eL, abs(native - jax_eL))


# --------------------------------------------------------------------------- #
# The MCMC state transition on an empty spin channel.
# --------------------------------------------------------------------------- #
_ONLY_UP_CASES = [c for c in _CASES if c[3] in ("ae_jnone", "ae_j12", "ecp_jnone", "li_ecp_jnone", "li_ecp_j12")]


def _charges(ham):
    """Effective nuclear charges the Kato drift uses (ECP: Z - z_core)."""
    cp = ham.coulomb_potential_data
    Z = np.asarray(ham.structure_data.atomic_numbers, dtype=np.float64)
    if cp.ecp_flag:
        Z = Z - np.asarray(cp.z_cores, dtype=np.float64)
    return Z


@pytest.mark.light
@pytest.mark.parametrize("stem, with_jastrow, spec, tag", _ONLY_UP_CASES, ids=[c[3] for c in _ONLY_UP_CASES])
def test_single_spin_mcmc_transition(stem, with_jastrow, spec, tag):
    """The ``n_dn == 0`` MCMC transition, XLA vs Rust, compared after every move.

    The XLA oracle is the driver's DEDICATED only-up kernel
    ``_update_electron_positions_only_up_electron`` (the ``__only_up_electron``
    branch of ``MCMC.run``); the two-spin ``_update_electron_positions`` cannot
    even trace here because its ``r_dn_carts[selected_electron_index]`` gathers on
    a size-0 axis. The Rust side is the production ``_mcmc_sweep`` (which the
    ``flag_rust_on_CPU`` replacement drives for both spin cases), fed the
    codebase's own bit-exact ``_draw_injected_normals`` stream. Ten single moves
    are compared move-by-move: the accept decision, the walker update, the rank-1
    geminal + Sherman-Morrison inverse, and the advanced threefry key (exact).
    """
    ham = _build(stem, with_jastrow, spec)
    gem = ham.wavefunction_data.geminal_data
    jas = ham.wavefunction_data.jastrow_data
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    Z = _charges(ham)
    Dt = 1.2
    key = random.PRNGKey(11)
    pos_atol, pos_rtol = get_tolerance_xla_rust_consistency(_ZONES_POS, level="strict")
    gem_atol, gem_rtol = get_tolerance_xla_rust_consistency(_ZONES_GEM, level="strict")

    x_up, x_dn = _config(ham, seed=2)
    x_up = np.ascontiguousarray(x_up)
    s_up, s_dn = x_up.copy(), x_dn.copy()
    n_accept = 0
    for mv in range(10):
        move_label = str(mv)
        G = np.asarray(compute_geminal_all_elements(gem, x_up, x_dn))
        G_inv = np.linalg.inv(G)
        det_state = _init_det_ratio_streaming_state(gem, jnp.asarray(x_up), jnp.asarray(x_dn))
        j3d = jas.jastrow_three_body_data
        j3_state = _init_jastrow_ratio_state(j3d, jnp.asarray(x_up), jnp.asarray(x_dn)) if j3d is not None else None
        jastrow_state = Jastrow_ratio_state(j3_state=j3_state)
        (acc, _rej, xr_up, xr_dn, x_key, xG_inv, xG, _xdet, _xjas) = _update_electron_positions_only_up_electron(
            jnp.asarray(x_up),
            jnp.asarray(x_dn),
            key,
            1,
            ham,
            Dt,
            1.0e-6,
            jnp.asarray(G_inv),
            jnp.asarray(G),
            det_state,
            jastrow_state,
        )

        sG = np.asarray(compute_geminal_all_elements(gem, s_up, s_dn))
        sG_inv = np.linalg.inv(sG)
        normals = np.asarray(_draw_injected_normals(key, 1), dtype=np.float64)
        kw = np.asarray(key, dtype=np.uint32)
        rs = jqmc_rs._mcmc_sweep(gem, jas, R, Z, s_up, s_dn, sG, sG_inv, Dt, 1.0e-6, int(kw[0]), int(kw[1]), normals, None, 0)

        tag_mv = f"{tag}/move={mv}"
        assert int(acc) == int(rs["accepted"]), f"{tag_mv}: accept XLA={int(acc)} Rust={int(rs['accepted'])}"
        assert np.allclose(np.asarray(rs["r_up"]), np.asarray(xr_up), atol=pos_atol, rtol=pos_rtol), (
            f"{tag_mv}: r_up max|d|={np.abs(np.asarray(rs['r_up']) - np.asarray(xr_up)).max():.3e}"
        )
        assert np.asarray(rs["r_dn"]).shape == (0, 3) == np.asarray(xr_dn).shape
        assert np.allclose(np.asarray(rs["geminal"]), np.asarray(xG), atol=gem_atol, rtol=gem_rtol), (
            f"{tag_mv}: geminal max|d|={np.abs(np.asarray(rs['geminal']) - np.asarray(xG)).max():.3e}"
        )
        assert np.allclose(np.asarray(rs["geminal_inv"]), np.asarray(xG_inv), atol=gem_atol, rtol=gem_rtol), (
            f"{tag_mv}: geminal_inv max|d|={np.abs(np.asarray(rs['geminal_inv']) - np.asarray(xG_inv)).max():.3e}"
        )
        assert np.array_equal(np.asarray(rs["final_key"], dtype=np.uint32), np.asarray(x_key, dtype=np.uint32)), (
            f"{tag_mv}: threefry key drift"
        )

        n_accept += int(acc)
        x_up, x_dn = np.ascontiguousarray(np.asarray(xr_up)), np.asarray(xr_dn)
        s_up, s_dn = np.ascontiguousarray(rs["r_up"]), np.ascontiguousarray(rs["r_dn"])
        key = x_key
    # Guard against a vacuous pass: a chain that accepted nothing would compare
    # ten identical no-op transitions.
    assert n_accept > 0, f"{tag}: no move accepted in 10 -- transition cells not exercised"
    # ``ATOL_BIT_EXACT`` is imported to document the key channel is an exact
    # comparison; np.array_equal above is the stricter statement of the same intent.
    _ = (ATOL_BIT_EXACT, RTOL_BIT_EXACT)


# --------------------------------------------------------------------------- #
# The non-local ECP channel on an empty spin channel.
# --------------------------------------------------------------------------- #
@pytest.mark.light
def test_single_spin_ecp_nonlocal_is_active():
    """``Li_ecp`` is the only single-spin fixture with ``V_nl != 0``.

    ``H_ecp``'s ccECP (z_core=0, zero l=0 projector) has an IDENTICALLY zero
    non-local energy, so every ``single-spin x ECP`` value row is vacuous on the
    non-local path. Li under ccECP keeps one valence electron and a real l=0
    projector. This pins that the Li fixture actually drives the non-local
    channel (so its ``compute_local_energy_fast`` parity is not a vacuous pass),
    and that H's is exactly the zero it is supposed to be.
    """
    w = np.asarray(_MESH.weights, float)
    gp = np.asarray(_MESH.grid_points, float)

    def _rust_nonlocal(stem):
        ham = _build_ham_for_cell(stem, False, None, "SD")
        wf = ham.wavefunction_data
        gd = wf.geminal_data
        cd = ham.coulomb_potential_data
        natom = np.asarray(ham.structure_data.positions).shape[0]
        vals = []
        for seed in range(3):
            r_up, r_dn = _config(_build_ham_for_cell(stem, False, None, "SD"), seed)
            vals.append(
                float(jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gd, wf.jastrow_data, cd, r_up, r_dn, w, gp, False, natom))
            )
        return np.asarray(vals)

    h_vals = _rust_nonlocal("H_ecp_ccpvdz_cart.h5")
    li_vals = _rust_nonlocal("Li_ecp_ccpvdz_cart.h5")
    assert np.allclose(h_vals, 0.0, atol=ATOL_BIT_EXACT, rtol=RTOL_BIT_EXACT), f"H ccECP non-local not zero: {h_vals}"
    assert np.abs(li_vals).max() > 1.0, f"Li ccECP non-local channel is vacuous ({li_vals}) -- fixture no longer exercises V_nl"
    # ``compute_ecp_non_local_parts_all_pairs`` is imported to document the XLA
    # counterpart; the value-level parity of this channel is carried end-to-end by
    # ``test_local_energy_fast_single_spin`` on the Li rows (compute_local_energy_fast
    # includes V_nl), because the XLA all-pairs kernel scatters on a size-0 dn axis
    # and cannot be called directly at n_dn == 0.
    _ = compute_ecp_non_local_parts_all_pairs


# --------------------------------------------------------------------------- #
# Issue #54: the native sweep on an empty down channel with n_up >= 2.
# --------------------------------------------------------------------------- #
_MULTI_UP_FIXTURE = "N_ae_ccpvdz_cart.h5"
_MULTI_UP_CASES = [(2, "sd"), (3, "sd"), (4, "sd"), (2, "agp"), (3, "agp"), (4, "agp")]
#: Zones crossed by the rebuilt Slater matrix, the carried rank-1 updates and their log-determinant.
_ZONES_MULTI_UP = ("ao_eval", "mo_eval", "det_eval", "det_ratio")
_MULTI_UP_SWEEPS = 4
_MULTI_UP_MOVES = 25
_MULTI_UP_DT = 1.2


def _multi_up_system(num_up, kind):
    """Committed AE N atom, no Jastrow, ``num_up`` up electrons on the first MOs, no down electron."""
    ham = _build_ham_for_cell(_MULTI_UP_FIXTURE, False, None, "SD")
    mos = ham.wavefunction_data.geminal_data.orb_data_up_spin
    sd = SD_data.build(num_electron_up=num_up, num_electron_dn=0, mos_up=mos, mos_dn=mos, time_reversal=False)
    gem = sd if kind == "sd" else to_AGP(sd)
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=gem))
    assert (int(gem.num_electron_up), int(gem.num_electron_dn)) == (num_up, 0)
    rng = np.random.default_rng(3)
    r_up = np.ascontiguousarray(rng.normal(scale=0.7, size=(num_up, 3)))
    return ham, sd, gem, r_up


def _native_sweep_seed(ham, r_up, r_dn):
    """``(G, G^-1)`` seeded exactly as the production MCMC driver seeds a real native sweep.

    The native sweep advances its carried G by rank-1 deltas against its own
    cached orbital columns, so the seed must come from those caches
    (``NativeObsCtx.sweep_seed_geminal_batch``, the call ``MCMC.run`` makes on the real
    native routes); a JAX-built seed differs from them at the ao_eval rounding level under
    the mixed fp32 zone and that difference stays in G as a fixed offset. The context is
    built with the active precision policy, as the driver builds it.
    """
    wf = ham.wavefunction_data
    ctx = jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        False,  # real
        False,  # open boundary
        False,  # want_stress
        want_force=False,
        precision_policy=precision_policy_snapshot(),
    )
    seed = ctx.sweep_seed_geminal_batch(np.ascontiguousarray(r_up[None]), np.ascontiguousarray(r_dn[None]))
    G = np.ascontiguousarray(np.asarray(seed, dtype=np.float64)[0])
    return G, np.ascontiguousarray(np.linalg.inv(G))


def _native_multi_up_chain(ham, gem, r_up, G, G_inv):
    """Yield ``(sweep, r_up, G, G_inv)`` after each of the native sweeps of the issue #54 chain."""
    jas = ham.wavefunction_data.jastrow_data
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    Z = _charges(ham)
    r_dn = np.zeros((0, 3), dtype=np.float64)
    key = random.PRNGKey(5)
    for sweep in range(_MULTI_UP_SWEEPS):
        normals = np.asarray(_draw_injected_normals(key, _MULTI_UP_MOVES), dtype=np.float64)
        kw = np.asarray(key, dtype=np.uint32)
        rs = jqmc_rs._mcmc_sweep(
            gem, jas, R, Z, r_up, r_dn, G, G_inv, _MULTI_UP_DT, 1.0e-6, int(kw[0]), int(kw[1]), normals, None, 0
        )
        assert np.asarray(rs["r_dn"]).shape == (0, 3)
        r_up = np.ascontiguousarray(rs["r_up"])
        G = np.ascontiguousarray(rs["geminal"], dtype=np.float64)
        G_inv = np.ascontiguousarray(rs["geminal_inv"], dtype=np.float64)
        key = jnp.asarray(np.asarray(rs["final_key"], dtype=np.uint32))
        yield sweep, r_up, G, G_inv, int(rs["accepted"])


@pytest.mark.light
@pytest.mark.parametrize("num_up, kind", _MULTI_UP_CASES, ids=[f"{k}_nup{n}" for n, k in _MULTI_UP_CASES])
def test_only_up_multi_electron_native_sweep_matches_rebuilt_slater(num_up, kind):
    """Carried G, G^-1 and ln|det| of ``jqmc_rs._mcmc_sweep`` at n_dn = 0, n_up >= 2 vs a Slater rebuild.

    Every other ``n_dn == 0`` row of this module has one up electron, so the moved index of
    the native rank-1 update is always 0 there. Here the committed all-electron N atom
    (cc-pVDZ, Cartesian, no Jastrow) carries ``num_up`` in {2, 3, 4} up electrons on its first
    MOs and no down electron, as an SD and as the same SD converted to AGP. The chain is seeded
    as the production driver seeds it (``_native_sweep_seed``). Four native sweeps of 25
    proposals each (Dt = 1.2, PRNGKey(5), injected normals from ``_draw_injected_normals``,
    start at 0.7 * standard normals with PCG64 seed 3) carry the walker, the threefry key, G
    and G^-1 from one sweep to the next. After each sweep:

    * the carried G must equal the Slater matrix rebuilt by the XLA ``compute_MOs(orb_up,
      r_up).T`` (independent AO/MO evaluator, value level);
    * the carried G^-1 must equal the numpy inverse of a fresh native geminal
      ``jqmc_rs.compute_geminal_all_elements`` at the returned positions,
      ``G_fresh @ G^-1`` the identity and ``slogdet(G^-1)`` minus ``slogdet(G_fresh)``.

    No reference uses a rank-1 update or a streaming cache. The inverse-weighted checks use the
    fresh native geminal because it is the same-policy evaluation of the sweep's own orbital
    columns: under the mixed fp32 ``ao_eval`` zone two different AO implementations differ at
    the ~6e-8 level, and ``G_ref @ G^-1`` multiplies that by the condition number (rank 4: 2.5e-6
    > 1e-6), which would measure AO rounding instead of the rank-1 update. Every up electron
    must be moved by an accepted proposal during the chain.
    """
    ham, sd, gem, r_up = _multi_up_system(num_up, kind)
    atol, rtol = get_tolerance_min(_ZONES_MULTI_UP, "strict")
    r_dn = np.zeros((0, 3), dtype=np.float64)
    G, G_inv = _native_sweep_seed(ham, r_up, r_dn)

    total_accepted = 0
    moved = np.zeros(num_up, dtype=bool)
    for sweep, new_up, G, G_inv, accepted in _native_multi_up_chain(ham, gem, r_up, G, G_inv):
        total_accepted += accepted
        moved |= np.any(new_up != r_up, axis=1)
        r_up = new_up

        sweep_label = str(sweep)
        label = f"{kind}/n_up={num_up}/sweep={sweep_label}"
        g_ref = np.asarray(compute_MOs(sd.orb_data_up_spin, jnp.asarray(r_up)), dtype=np.float64).T
        g_fresh = np.asarray(jqmc_rs.compute_geminal_all_elements(gem, r_up, r_dn), dtype=np.float64)
        assert g_ref.shape == g_fresh.shape == (num_up, num_up)
        sign_ref, logabs_ref = np.linalg.slogdet(g_fresh)
        sign_x, logabs_x = np.linalg.slogdet(G_inv)
        np.testing.assert_allclose(G, g_ref, atol=atol, rtol=rtol, err_msg=f"{label}: carried G")
        np.testing.assert_allclose(G_inv, np.linalg.inv(g_fresh), atol=atol, rtol=rtol, err_msg=f"{label}: carried G^-1")
        np.testing.assert_allclose(g_fresh @ G_inv, np.eye(num_up), atol=atol, rtol=rtol, err_msg=f"{label}: G_fresh G^-1")
        assert sign_x == sign_ref, f"{label}: sign det(G^-1)={sign_x} sign det(G_fresh)={sign_ref}"
        np.testing.assert_allclose(-logabs_x, logabs_ref, atol=atol, rtol=rtol, err_msg=f"{label}: ln|det|")

    assert total_accepted > 0, f"{kind}/n_up={num_up}: no move accepted"
    assert moved.all(), f"{kind}/n_up={num_up}: electrons never moved: {np.flatnonzero(~moved).tolist()}"


@pytest.mark.light
@pytest.mark.parametrize("num_up, kind", _MULTI_UP_CASES, ids=[f"{k}_nup{n}" for n, k in _MULTI_UP_CASES])
def test_only_up_native_sweep_carried_geminal_has_no_seed_offset(num_up, kind):
    """The multi-up native chain carries no seed-versus-cache offset in G (both precision modes).

    The native sweep advances G by rank-1 deltas against its own orbital caches. Under the
    mixed fp32 ``ao_eval`` zone a JAX-built seed differs from those caches (3.4e-8 here) and the
    difference stays in the carried G for the whole chain, so a carried-state test seeded from
    JAX would mix two evaluations. This pins the seed the carried-state test uses (``_native_sweep_seed``,
    the production driver's ``sweep_seed_geminal_batch``) to the sweep's own evaluation:

    * at the start positions the seed equals the fresh native geminal
      ``jqmc_rs.compute_geminal_all_elements`` (so that fresh geminal is the same-policy
      rebuild of the sweep caches the carried-state test may use as its reference);
    * after each of the four sweeps of the same chain (n_dn = 0, n_up in {2, 3, 4}, SD and
      AGP, committed AE N atom, no Jastrow) the carried G equals the fresh native geminal at
      the returned positions.

    Both are the same fp64 arithmetic on the same AO values (the carried G only replaces rows),
    so the comparison uses the same-arithmetic reduction-order pair
    (``ATOL_REDUCTION_ORDER``, ``RTOL_REDUCTION_ORDER``), not a zone budget: a zone budget
    (mixed strict atol 1e-6) would hide the 3.4e-8 offset. Every up electron must move.
    """
    ham, _sd, gem, r_up = _multi_up_system(num_up, kind)
    r_dn = np.zeros((0, 3), dtype=np.float64)
    G, G_inv = _native_sweep_seed(ham, r_up, r_dn)
    g_start = np.asarray(jqmc_rs.compute_geminal_all_elements(gem, r_up, r_dn), dtype=np.float64)
    np.testing.assert_allclose(
        G, g_start, atol=ATOL_REDUCTION_ORDER, rtol=RTOL_REDUCTION_ORDER, err_msg=f"{kind}/n_up={num_up}: seed G"
    )

    moved = np.zeros(num_up, dtype=bool)
    for sweep, new_up, G, G_inv, _accepted in _native_multi_up_chain(ham, gem, r_up, G, G_inv):
        moved |= np.any(new_up != r_up, axis=1)
        r_up = new_up
        sweep_label = str(sweep)
        g_fresh = np.asarray(jqmc_rs.compute_geminal_all_elements(gem, r_up, r_dn), dtype=np.float64)
        np.testing.assert_allclose(
            G,
            g_fresh,
            atol=ATOL_REDUCTION_ORDER,
            rtol=RTOL_REDUCTION_ORDER,
            err_msg=f"{kind}/n_up={num_up}/sweep={sweep_label}: carried G vs fresh native geminal",
        )
    assert moved.all(), f"{kind}/n_up={num_up}: electrons never moved: {np.flatnonzero(~moved).tolist()}"
