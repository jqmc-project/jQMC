"""Layer-2 equivalence: native single move vs jqmc body_fun (random
injection) on committed Hamiltonian fixtures.

Replicates the jqmc ``body_fun`` (jqmc_mcmc.py) in Python using jqmc's own
verified functions + the exact threefry RNG sequence + an INJECTED Gaussian
displacement, and compares the Rust ``_single_move`` outcome (accept,
Det_T_ratio, T_ratio, R_ratio, new positions, geminal_inv) tol-for-tol against
the jqmc reference computation.

Molecular J1/J2 path (J3 / PBC are follow-ups). A Rust==XLA equivalence is
code-path sensitive, not system sensitive, so each committed fixture stands in
for a larger system that flows the SAME code path:
  - H2 (AE, sphe-GTO, j12)  -> ``mol_cart_sd_j12`` (AE, cart-GTO,
    j12): the AO family (cart vs sphe) doesn't matter for this kernel's
    equivalence -- the rank-1 determinant/Jastrow update logic is
    AO-family-agnostic.
  - Cl2 (ECP, cart-GTO, j12) -> ``mol_ecp_cart_j12`` (exact
    AO-family + ECP match).
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

# Every rs test declares its cost tier (light/heavy, per test) and CWD policy
# (with_chdir here -- autouse _isolate_cwd chdirs into a per-test tmp_path for
# xdist safety; committed fixtures load via absolute paths so cwd is
# irrelevant).
pytestmark = pytest.mark.with_chdir

from jax import numpy as jnp
from jax import random

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import ATOL_BIT_EXACT, RTOL_BIT_EXACT
from jqmc.determinant import (
    _compute_u_dn_move_from_det_ratio_state,
    _compute_v_up_move_from_det_ratio_state,
    _init_det_ratio_streaming_state,
    compute_AS_regularization_factor_fast_update,
    compute_geminal_all_elements,
)
from jqmc.jastrow_factor import _compute_ratio_Jastrow_part_rank1_update

from tests.test_rs_kernel._rs_fixtures import rs_ham

# name -> canonical committed-fixture pattern (AE/ECP x cart-GTO x J1+J2).
SYSTEMS = {
    "H2": "mol_cart_sd_j12",
    "Cl2": "mol_ecp_cart_j12",
}

# Det_T_ratio / geminal_inv flow through the rank-1 determinant update.
_ZONES_DET = ("ao_eval", "det_ratio")
# R_ratio additionally folds in the Jastrow rank-1 ratio (WF_ratio = J_ratio *
# Det_T_ratio); T_ratio is the Kato-Green's-function move ratio consumed
# alongside Det_T_ratio in the same acceptance product, so it shares the same
# precision budget.
_ZONES_WF = ("det_ratio", "jastrow_ratio")


def _charges(ham):
    cp = ham.coulomb_potential_data
    Z = np.asarray(ham.structure_data.atomic_numbers, dtype=np.float64)
    if cp.ecp_flag:
        Z = Z - np.asarray(cp.z_cores, dtype=np.float64)
    return Z


def _reference_move(ham, r_up, r_dn, G, G_inv, dt, eps, seed, injected_normal):
    """Faithful Python replication of jqmc body_fun for one molecular move."""
    gem = ham.wavefunction_data.geminal_data
    jas = ham.wavefunction_data.jastrow_data
    pos = np.asarray(ham.structure_data.positions, dtype=np.float64)
    Z = _charges(ham)
    n_up, n_dn = len(r_up), len(r_dn)

    key = random.PRNGKey(seed)
    key, sk = random.split(key)
    rand_num = int(random.randint(sk, (), 0, n_up + n_dn))
    is_up = rand_num < n_up
    key, sk = random.split(key)
    up_i = int(random.randint(sk, (), 0, n_up))
    key, sk = random.split(key)
    dn_i = int(random.randint(sk, (), 0, n_dn))
    sel = up_i if is_up else dn_i
    old_r = (r_up if is_up else r_dn)[sel]

    def nearest(r):
        d2 = ((pos - r) ** 2).sum(axis=1)
        i = int(np.argmin(d2))
        return i, float(np.sqrt(d2[i]))

    ni, norm = nearest(old_r)
    z = Z[ni]
    f_l = 1.0 / z**2 * (1 + z**2 * norm) / (1 + norm)
    sigma = f_l * dt
    key, sk = random.split(key)  # normal subkey consumed
    g = injected_normal * sigma
    key, sk = random.split(key)
    axis = int(random.randint(sk, (), 0, 3))
    new_r = old_r.copy()
    new_r[axis] += g
    ni2, norm2 = nearest(new_r)
    z2 = Z[ni2]
    f_prime = 1.0 / z2**2 * (1 + z2**2 * norm2) / (1 + norm2)
    dist2 = float(((new_r - old_r) ** 2).sum())
    T_ratio = (f_l / f_prime) * np.exp(-dist2 * (1.0 / (2 * f_prime**2 * dt**2) - 1.0 / (2 * f_l**2 * dt**2)))

    st = _init_det_ratio_streaming_state(gem, r_up, r_dn)
    if is_up:
        rp = r_up.copy()
        rp[sel] = new_r
        v = np.asarray(_compute_v_up_move_from_det_ratio_state(gem, st, sel, rp))
        u = np.eye(n_up)[sel]
    else:
        rp = r_dn.copy()
        rp[sel] = new_r
        u = np.asarray(_compute_u_dn_move_from_det_ratio_state(gem, st, sel, rp))
        v = np.eye(n_up)[sel]
    Ainv_u = G_inv @ u
    Det_T_ratio = 1.0 + v @ Ainv_u
    G_inv_new = G_inv - np.outer(Ainv_u, v @ G_inv) / Det_T_ratio
    G_new = G.copy()
    if is_up:
        G_new[sel, :] += v
    else:
        G_new[:, sel] += u

    R_AS_p = float(compute_AS_regularization_factor_fast_update(G_new, G_inv_new))
    R_AS_o = float(compute_AS_regularization_factor_fast_update(G, G_inv))
    R_AS_p_eps = max(R_AS_p, eps)
    R_AS_o_eps = max(R_AS_o, eps)
    R_AS_ratio = (R_AS_p_eps / R_AS_p) / (R_AS_o_eps / R_AS_o)

    new_r_up = r_up.copy()
    new_r_dn = r_dn.copy()
    if is_up:
        new_r_up[sel] = new_r
    else:
        new_r_dn[sel] = new_r
    J_ratio = float(np.asarray(_compute_ratio_Jastrow_part_rank1_update(jas, r_up, r_dn, new_r_up[None], new_r_dn[None]))[0])
    WF_ratio = J_ratio * Det_T_ratio
    R_ratio = (R_AS_ratio * WF_ratio) ** 2
    key, sk = random.split(key)
    b = float(random.uniform(sk, (), dtype=jnp.float64, minval=0.0, maxval=1.0))
    acceptance = min(1.0, R_ratio * T_ratio)
    accepted = b < acceptance
    return dict(
        is_up=is_up,
        sel=sel,
        accepted=accepted,
        Det_T_ratio=Det_T_ratio,
        T_ratio=T_ratio,
        R_ratio=R_ratio,
        new_r=new_r,
        G_inv_new=(G_inv_new if accepted else G_inv),
        new_r_up=(new_r_up if accepted else r_up),
        new_r_dn=(new_r_dn if accepted else r_dn),
    )


@pytest.mark.light
@pytest.mark.parametrize("name", list(SYSTEMS))
@pytest.mark.parametrize("seed", [0, 1, 3, 7, 42])
@pytest.mark.parametrize("eps", [0.0, 0.03])
def test_single_move_matches_body(name, seed, eps):
    """Single move matches body."""
    ham = rs_ham(SYSTEMS[name])
    gem = ham.wavefunction_data.geminal_data
    jas = ham.wavefunction_data.jastrow_data
    rng = np.random.default_rng(seed + 99)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    n_up, n_dn = gem.num_electron_up, gem.num_electron_dn
    r_up = (R[rng.integers(0, len(R), n_up)] + rng.normal(scale=0.7, size=(n_up, 3))).astype(np.float64)
    r_dn = (R[rng.integers(0, len(R), n_dn)] + rng.normal(scale=0.7, size=(n_dn, 3))).astype(np.float64)
    G = np.asarray(compute_geminal_all_elements(gem, r_up, r_dn), dtype=np.float64)
    G_inv = np.asarray(jqmc_rs._matrix_inverse(G), dtype=np.float64)
    dt = 1.0
    injected = float(rng.normal())

    k = np.asarray(random.PRNGKey(seed), dtype=np.uint32)
    Z = _charges(ham)
    res = jqmc_rs._single_move(gem, jas, R, Z, r_up, r_dn, G, G_inv, dt, eps, int(k[0]), int(k[1]), injected)
    ref = _reference_move(ham, r_up, r_dn, G, G_inv, dt, eps, seed, injected)

    atol_det, rtol_det = get_tolerance_xla_rust_consistency(_ZONES_DET, level="medium")
    atol_wf, rtol_wf = get_tolerance_xla_rust_consistency(_ZONES_WF, level="medium")

    assert res["is_up"] == ref["is_up"]
    assert res["selected_index"] == ref["sel"]
    assert np.isclose(res["det_t_ratio"], ref["Det_T_ratio"], rtol=rtol_det, atol=atol_det)
    assert np.isclose(res["t_ratio"], ref["T_ratio"], rtol=rtol_wf, atol=atol_wf)
    assert np.isclose(res["r_ratio"], ref["R_ratio"], rtol=rtol_wf, atol=atol_wf)
    assert res["accepted"] == ref["accepted"], (res["r_ratio"], ref["R_ratio"], res["metropolis_b"])
    # Position updates are a bit-exact copy of the same (old_r + injected-normal
    # * sigma) arithmetic threaded through both Rust and the Python reference;
    # observed diff is exactly 0.0 across seeds/eps, so no arithmetic-derived
    # tolerance is warranted here.
    assert np.allclose(res["r_up"], ref["new_r_up"], rtol=RTOL_BIT_EXACT, atol=ATOL_BIT_EXACT)
    assert np.allclose(res["r_dn"], ref["new_r_dn"], rtol=RTOL_BIT_EXACT, atol=ATOL_BIT_EXACT)
    assert np.allclose(res["geminal_inv"], np.asarray(ref["G_inv_new"]), rtol=rtol_det, atol=atol_det)


# --------------------------------------------------------------------------- #
# Periodic cell (issue #155): the single move must evaluate its AOs at
# home-cell-wrapped coordinates, exactly as ``mcmc_sweep`` does.
# --------------------------------------------------------------------------- #
# WF-PATTERN-015 (canonical): periodic rhombohedral, bare Be (Z=4), real SD cart-pGTO, no
# Jastrow (the determinant alone decides the move). The two committed lithium.h5 Gamma rows
# (ccECP, sphe-pGTO) add the Jastrow channels of the single move under PBC: J1 + J2 + AO-J3
# (``pbc_gamma_sphepgto_j123``, whose J3 AOs need the same home-cell wrap) and J1 + J2 + DTN
# (``pbc_gamma_sphepgto_jdtn``).
_PBC_PATTERNS = ("WF-PATTERN-015", "pbc_gamma_sphepgto_j123", "pbc_gamma_sphepgto_jdtn")
# Walker translations, in lattice vectors: up electrons by +s*a0, dn electrons by -s*a1.
_PBC_LATTICE_SHIFTS = (0, 1, 3)
_PBC_MOVE_KEYS = (0, 1, 2, 3, 4, 5)
_PBC_DT = 1.0
# Brute-force minimum-image box: integer image offsets -3..3 along each lattice vector.
_PBC_IMAGE_OFFSETS = (-3, -2, -1, 0, 1, 2, 3)


def _pbc_case(pattern):
    """Committed periodic ham plus one fixed home-cell walker configuration."""
    from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian

    ham = build_canonical_hamiltonian(pattern) if pattern.startswith("WF-PATTERN-") else rs_ham(pattern)
    sd = ham.structure_data
    assert sd.pbc_flag
    gem = ham.wavefunction_data.geminal_data
    cell = np.asarray(sd.cell, dtype=np.float64)
    rng = np.random.default_rng(7)
    base_up = rng.random((gem.num_electron_up, 3)) @ cell
    base_dn = rng.random((gem.num_electron_dn, 3)) @ cell
    return ham, base_up, base_dn


def _pbc_shifted(ham, base_up, base_dn, shift):
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    return base_up + shift * cell[0], base_dn - shift * cell[1]


def _pbc_native_moves(case):
    """``_single_move`` (and the one-move ``_mcmc_sweep``) at every lattice shift and key."""
    ham, base_up, base_dn = case
    sd = ham.structure_data
    gem = ham.wavefunction_data.geminal_data
    jas = ham.wavefunction_data.jastrow_data
    R = np.ascontiguousarray(sd.positions, dtype=np.float64)
    Z = np.ascontiguousarray(_charges(ham), dtype=np.float64)
    cell = np.ascontiguousarray(sd.cell, dtype=np.float64)
    out = {name: [] for name in ("det_t_ratio", "r_ratio", "t_ratio", "accepted", "sweep_accepted", "is_up", "sel", "new_r")}
    for shift in _PBC_LATTICE_SHIFTS:
        r_up, r_dn = (np.ascontiguousarray(a) for a in _pbc_shifted(ham, base_up, base_dn, shift))
        G = np.ascontiguousarray(compute_geminal_all_elements(gem, r_up, r_dn), dtype=np.float64)
        G_inv = np.ascontiguousarray(np.linalg.inv(G))
        rows = {name: [] for name in out}
        for key_seed in _PBC_MOVE_KEYS:
            kw = np.asarray(random.PRNGKey(key_seed), dtype=np.uint32)
            injected = float(np.random.default_rng(100 + key_seed).normal())
            args = (gem, jas, R, Z, r_up, r_dn, G, G_inv, _PBC_DT, 0.0, int(kw[0]), int(kw[1]))
            single = jqmc_rs._single_move(*args, injected, cell=cell, n_max=int(sd._n_max))
            sweep = jqmc_rs._mcmc_sweep(*args, np.array([injected]), cell, int(sd._n_max))
            for name in ("det_t_ratio", "r_ratio", "t_ratio", "accepted", "is_up"):
                rows[name].append(single[name])
            rows["sel"].append(single["selected_index"])
            rows["new_r"].append(np.asarray(single["new_pos"], dtype=np.float64))
            rows["sweep_accepted"].append(bool(sweep["accepted"]))
        for name in out:
            out[name].append(rows[name])
    return {name: np.asarray(value) for name, value in out.items()}


def _kato_min_image(ham, r):
    """Kato step f_l(d) with d the brute-force minimum-image electron-nucleus distance."""
    import itertools

    sd = ham.structure_data
    cell = np.asarray(sd.cell, dtype=np.float64)
    R = np.asarray(sd.positions, dtype=np.float64)
    Z = _charges(ham)
    images = np.array(list(itertools.product(_PBC_IMAGE_OFFSETS, repeat=3)), dtype=np.float64) @ cell
    frac = (np.asarray(r, dtype=np.float64)[None, :] - R) @ np.linalg.inv(cell)
    d0 = (frac - np.round(frac)) @ cell
    dist = np.sqrt(((d0[:, None, :] + images[None]) ** 2).sum(-1)).min(axis=1)
    i = int(np.argmin(dist))
    return (1.0 + Z[i] ** 2 * dist[i]) / (Z[i] ** 2 * (1.0 + dist[i]))


_PBC_RATIO_KINDS = ("det_t_ratio", "r_ratio", "t_ratio")


@pytest.mark.light
@pytest.mark.parametrize("pattern", _PBC_PATTERNS)
def test_single_move_pbc_lattice_translated_walkers_match_from_scratch_ratios(pattern):
    """Periodic ``_single_move`` is lattice-translation invariant and exact.

    Full precision (correctness): with the up electrons shifted by 0, 1 and 3 lattice
    vectors (and the dn electrons by the opposite multiples of another vector),
    ``det_t_ratio`` must equal det G(new) / det G(old) with both geminals rebuilt from
    scratch by ``compute_geminal_all_elements``; ``r_ratio`` must equal
    ``(exp(J(new) - J(old)) * det ratio)**2`` from the from-scratch ``compute_Jastrow_part``
    (AS regularization off, eps = 0); ``t_ratio`` must equal the Kato ratio at a brute-force
    minimum-image distance; and the accept decision must be identical for every shift and
    equal to the one-move ``_mcmc_sweep`` decision. A native move that evaluates its
    determinant and AO-J3 AOs at the unwrapped coordinates gives det ratio 1.0 three cells out.

    Mixed precision: correctness belongs to full, so mixed is judged on the same lattice
    translation within its own precision (every shift against the mixed home-cell move,
    strict zone policy of its fp32 zones), and its accept decisions (single move and
    one-move sweep) must equal those of the move rebuilt independently in full precision
    at the home cell (``tests._precision_fidelity.mixed_vs_full``; exact).
    """
    from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
    from jqmc.jastrow_factor import compute_Jastrow_part

    from tests._precision_fidelity import mixed_vs_full

    if is_mixed_precision_enabled():
        mixed, full = mixed_vs_full(
            lambda: _pbc_case(pattern),
            _pbc_native_moves,
            configure_native=jqmc_rs.configure_native_precision,
            native_policy_snapshot=jqmc_rs.native_precision_support,
        )
        atol, rtol = get_tolerance_min(("ao_eval", "det_ratio", "jastrow_ratio"), "strict")
        for name in _PBC_RATIO_KINDS:
            for s, shift in enumerate(_PBC_LATTICE_SHIFTS):
                assert np.allclose(mixed[name][s], mixed[name][0], atol=atol, rtol=rtol), (
                    f"{pattern}: mixed {name} at lattice shift {shift} vs mixed home cell: {mixed[name][s]} vs {mixed[name][0]}"
                )
        for s, shift in enumerate(_PBC_LATTICE_SHIFTS):
            assert np.array_equal(mixed["accepted"][s], full["accepted"][0]), (pattern, shift)
            assert np.array_equal(mixed["sweep_accepted"][s], full["sweep_accepted"][0]), (pattern, shift)
        return

    case = _pbc_case(pattern)
    ham, base_up, base_dn = case
    gem = ham.wavefunction_data.geminal_data
    jas = ham.wavefunction_data.jastrow_data
    assert (jas.jastrow_three_body_data is not None) == (pattern == "pbc_gamma_sphepgto_j123")
    assert (jas.jastrow_dtn_data is not None) == (pattern == "pbc_gamma_sphepgto_jdtn")
    res = _pbc_native_moves(case)
    atol_det, rtol_det = get_tolerance_min(("ao_eval", "det_ratio"), "strict")
    atol_wf, rtol_wf = get_tolerance_min(("ao_eval", "det_ratio", "jastrow_ratio"), "strict")
    for s, shift in enumerate(_PBC_LATTICE_SHIFTS):
        r_up, r_dn = _pbc_shifted(ham, base_up, base_dn, shift)
        sign0, logdet0 = np.linalg.slogdet(np.asarray(compute_geminal_all_elements(gem, r_up, r_dn), dtype=np.float64))
        j0 = float(compute_Jastrow_part(jas, r_up, r_dn))
        for m, key_seed in enumerate(_PBC_MOVE_KEYS):
            tag = f"{pattern}/shift={shift}/key={key_seed}"
            sel, is_up, new_r = int(res["sel"][s, m]), bool(res["is_up"][s, m]), res["new_r"][s, m]
            pu, pd = r_up.copy(), r_dn.copy()
            old_r = (r_up if is_up else r_dn)[sel].copy()
            (pu if is_up else pd)[sel] = new_r
            sign1, logdet1 = np.linalg.slogdet(np.asarray(compute_geminal_all_elements(gem, pu, pd), dtype=np.float64))
            det_ref = sign0 * sign1 * np.exp(logdet1 - logdet0)
            j_ref = np.exp(float(compute_Jastrow_part(jas, pu, pd)) - j0)
            f_l, f_p = _kato_min_image(ham, old_r), _kato_min_image(ham, new_r)
            dist2 = float(((new_r - old_r) ** 2).sum())
            t_ref = (f_l / f_p) * np.exp(-dist2 * (1.0 / (2 * f_p**2 * _PBC_DT**2) - 1.0 / (2 * f_l**2 * _PBC_DT**2)))
            assert np.isclose(res["det_t_ratio"][s, m], det_ref, atol=atol_det, rtol=rtol_det), (
                f"{tag}: det ratio {res['det_t_ratio'][s, m]!r} vs from-scratch {det_ref!r}"
            )
            assert np.isclose(res["r_ratio"][s, m], (j_ref * det_ref) ** 2, atol=atol_wf, rtol=rtol_wf), (
                f"{tag}: r_ratio {res['r_ratio'][s, m]!r} vs from-scratch {(j_ref * det_ref) ** 2!r}"
            )
            assert np.isclose(res["t_ratio"][s, m], t_ref, atol=atol_wf, rtol=rtol_wf), tag
    for s, shift in enumerate(_PBC_LATTICE_SHIFTS):
        assert np.array_equal(res["accepted"][s], res["accepted"][0]), f"{pattern}: accept differs at shift {shift}"
        assert np.array_equal(res["accepted"][s], res["sweep_accepted"][s]), f"{pattern}: accept vs sweep at shift {shift}"
    # The fixed keys must move both spins. On the Be cell they also take both decisions (the
    # smooth ccECP lithium rows accept every one of these moves; their ratios carry the check).
    assert res["is_up"].any() and not res["is_up"].all()
    if pattern == "WF-PATTERN-015":
        assert res["accepted"].any() and not res["accepted"].all()
