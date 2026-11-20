"""Production route-predicate counters of the Rust MCMC chunk.

Four Rust route predicates choose between a fast and a same-value slower route; without
a counter an unintended decline is invisible to every value test: ``coulomb_potential::ecp_wdot_ok`` / ``ecp_j3_wdot_ok`` (electron-fixed w-dot
contraction of the ECP non-local mesh vs the batched orb-mesh path) and the real
MCMC chunk's ``det_carry_ok`` / ``j3_carry_ok`` (carry the streaming determinant /
J3 ratio state across steps vs rebuild it). ``jqmc_rs.route_predicate_counters()``
reports, per predicate, how many decisions engaged and how many declined; it shares
the enable flag and the reset window of ``gfmc_route_counters``.

Each case drives the production chunk entry ``run_energy_chunk_batch`` and pins the
EXACT counts: in the fast chunk (``native_rng``) every step makes one carry decision
per predicate, and every local energy makes one w-dot decision per spin. The
bit-exact chunk (``native_rng=False``, the per-step reference path) forces every
predicate to decline (and never applies the FAST ECP non-local radial cut, whose
kept / dropped pair counters 18 / 19 then stay zero; in the fast chunk they add up to
every (electron, non-local term) pair of every local energy); periodic pGTO images force
the J3 carry to decline, while the periodic MO-GTO determinant state is carried in both
precision modes (the FAST periodic sweep's screened fp64 or fp32 ``ao_eval`` column
kernel); a B-spline determinant leaf forces the determinant w-dot to decline.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jqmc_rs  # noqa: E402

from jqmc._precision import precision_policy_snapshot  # noqa: E402
from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default, _geminal_inv_batched  # noqa: E402
from tests.test_rs_kernel._rs_fixtures import rs_ham  # noqa: E402

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_MESH = _NV_BASE_MESH[Nv_default]
_W = np.asarray(_MESH.weights, float)
_GP = np.asarray(_MESH.grid_points, float)
_NW = 2
_STEPS = 3
_NUM_MOVES = 4

# ``route_predicate_counters()`` index of (engaged, declined) per predicate.
_PREDICATE_INDICES = {
    "ecp_wdot_ok": (0, 1),
    "ecp_j3_wdot_ok": (2, 3),
    "det_carry_ok": (4, 5),
    "j3_carry_ok": (6, 7),
}

# ``route_predicate_counters()`` indices of the FAST ECP non-local radial cut
# (``coulomb_potential::FAST_ECP_NONLOCAL_RADIAL_CUT``): (electron, non-local term)
# pairs whose quadrature ran / was skipped. Counted on the FAST route only.
_RADIAL_CUT_KEPT = 18
_RADIAL_CUT_DROPPED = 19
# ``route_predicate_counters()`` indices of the complex FAST chunk carries
# (IMPL-RS-CPLX-CHUNK-KIN-CARRY-001): kinetic tables (engaged, declined, seeds,
# advances, probe re-seeds) and determinant base (engaged, declined).
_CPLX_KIN_ON, _CPLX_KIN_OFF, _CPLX_KIN_SEEDS, _CPLX_KIN_ADVANCES, _CPLX_KIN_RESEEDS = 22, 23, 24, 25, 26
_CPLX_BASE_ON, _CPLX_BASE_OFF = 27, 28
_CPLX_CARRY_FIRST, _CPLX_CARRY_END = 22, 29

# case -> (pattern, native_rng, expected route per predicate). ``True`` = engaged
# on every decision, ``False`` = declined on every decision, ``None`` = no decision
# (no J3 term).
_CASES = {
    "engaged-molecular-j123": (
        "mol_ecp_cart_j123",
        True,
        {"ecp_wdot_ok": True, "ecp_j3_wdot_ok": True, "det_carry_ok": True, "j3_carry_ok": True},
    ),
    "fallback-bitexact-j123": (
        "mol_ecp_cart_j123",
        False,
        {"ecp_wdot_ok": False, "ecp_j3_wdot_ok": False, "det_carry_ok": False, "j3_carry_ok": False},
    ),
    "fallback-periodic-images-j123": (
        "pbc_gamma_sphepgto_j123",
        True,
        {"ecp_wdot_ok": True, "ecp_j3_wdot_ok": True, "det_carry_ok": True, "j3_carry_ok": False},
    ),
    "fallback-bspline-det-j12": (
        "bspline_gamma_j12",
        True,
        {"ecp_wdot_ok": False, "ecp_j3_wdot_ok": None, "det_carry_ok": True, "j3_carry_ok": None},
    ),
}


def _chunk_inputs(ham):
    gd = ham.wavefunction_data.geminal_data
    cp = ham.coulomb_potential_data
    positions = np.asarray(ham.structure_data.positions, float)
    charges = np.asarray(ham.structure_data.atomic_numbers, float) - np.asarray(cp.z_cores, float)
    if bool(ham.structure_data.pbc_flag):
        cell = np.asarray(ham.structure_data.cell, float)
        n_max = int(ham.structure_data._n_max)
    else:
        cell, n_max = None, 0
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(7)
    r_up = np.stack([positions[rng.integers(0, len(positions), nu)] + rng.normal(scale=0.6, size=(nu, 3)) for _ in range(_NW)])
    r_dn = np.stack([positions[rng.integers(0, len(positions), nd)] + rng.normal(scale=0.6, size=(nd, 3)) for _ in range(_NW)])
    g, gi, _, _ = _geminal_inv_batched(gd, jnp.asarray(r_up), jnp.asarray(r_dn))
    keys = np.asarray([[34356 + w, 778899 + 3 * w] for w in range(_NW)], dtype=np.uint32)
    return positions, charges, r_up, r_dn, np.asarray(g, float), np.asarray(gi, float), keys, cell, n_max


def _run_chunk(ham, native_rng):
    ctx = jqmc_rs.build_obs_ctx(
        ham.wavefunction_data.geminal_data,
        ham.wavefunction_data.jastrow_data,
        ham.coulomb_potential_data,
        False,
        bool(ham.structure_data.pbc_flag),
        False,
        precision_policy=precision_policy_snapshot(),
    )
    positions, charges, r_up, r_dn, g, gi, keys, cell, n_max = _chunk_inputs(ham)
    if native_rng:
        normals = np.zeros((_NW, 0, 0))
        grids = np.zeros((_NW, 0, 0, 0))
        kwargs = dict(native_rng=True, random_mesh=True, base_grid_points=_GP, n_steps_fast=_STEPS)
    else:
        rng = np.random.default_rng(11)
        normals = np.ascontiguousarray(rng.standard_normal((_NW, _STEPS, _NUM_MOVES)))
        grids = np.ascontiguousarray(np.broadcast_to(_GP, (_NW, _STEPS) + _GP.shape).copy())
        kwargs = {}
    jqmc_rs.gfmc_route_counters_reset()
    out = ctx.run_energy_chunk_batch(
        positions, charges, r_up, r_dn, g, gi, 2.0, 1.0e-6, keys, normals, grids, _W, _NUM_MOVES, cell, n_max, **kwargs
    )
    e_l = np.asarray(out["e_L"])
    assert e_l.shape == (_NW, _STEPS) and np.all(np.isfinite(e_l)), f"chunk e_L not produced: {e_l}"


def _expected_radial_cut_pairs(ham, steps):
    """(electron, non-local term) pairs of ``steps`` FAST local energies: every electron x n_t of its nearest ECP nucleus."""
    gd = ham.wavefunction_data.geminal_data
    nuc = np.asarray(ham.coulomb_potential_data._nucleus_index_non_local_part, int)
    if nuc.size == 0:
        return 0
    per_nucleus = np.bincount(nuc)[np.unique(nuc)]
    assert np.all(per_nucleus == per_nucleus[0]), (
        f"exact pair count needs one non-local term count for every ECP nucleus: {per_nucleus}"
    )
    return steps * (int(gd.num_electron_up) + int(gd.num_electron_dn)) * int(per_nucleus[0])


def _expected_counts(routes, steps=_NW * _STEPS):
    """Exact (engaged, declined) counts implied by the per-predicate route over ``steps`` walker-steps."""
    decisions = {"ecp_wdot_ok": 2 * steps, "ecp_j3_wdot_ok": 2 * steps, "det_carry_ok": steps, "j3_carry_ok": steps}
    expected = {}
    for name, route in routes.items():
        if route is None:
            expected[name] = (0, 0)
        elif route:
            expected[name] = (decisions[name], 0)
        else:
            expected[name] = (0, decisions[name])
    return expected


@pytest.mark.parametrize("case", sorted(_CASES))
def test_mcmc_chunk_route_predicate_counters(case):
    """The four Rust route predicates are observable: engaged and forced-fallback counts are exact."""
    pattern, native_rng, routes = _CASES[case]
    ham = rs_ham(pattern)
    has_j3 = ham.wavefunction_data.jastrow_data.jastrow_three_body_data is not None
    assert has_j3 == (routes["j3_carry_ok"] is not None), f"{case}: J3 presence does not match the case"
    _run_chunk(ham, native_rng)
    counters = list(jqmc_rs.route_predicate_counters())
    observed = {name: (counters[on], counters[off]) for name, (on, off) in _PREDICATE_INDICES.items()}
    assert observed == _expected_counts(routes), f"{case}: route-predicate counters {observed}"
    assert counters[16] == counters[17] == 0, f"{case}: complex ECP w-dot fired on a real run {counters}"
    assert counters[20] == counters[21] == 0, f"{case}: complex ECP seed fired on a real run {counters}"
    assert counters[_CPLX_CARRY_FIRST:_CPLX_CARRY_END] == [0] * (_CPLX_CARRY_END - _CPLX_CARRY_FIRST), (
        f"{case}: complex chunk carry counters fired on a real run {counters}"
    )
    # FAST ECP radial cut: every (electron, term) pair of every FAST local energy is
    # counted kept or dropped; the bit-exact chunk never cuts and counts nothing.
    cut_pairs = (counters[_RADIAL_CUT_KEPT], counters[_RADIAL_CUT_DROPPED])
    want_pairs = _expected_radial_cut_pairs(ham, _NW * _STEPS) if native_rng else 0
    # (The committed H2 ccECP carries a zero-coefficient non-local channel, so all of its
    # pairs are dropped; the per-pair drop rule is IMPL-RS-FAST-ECP-NONLOCAL-RADIAL-CUT-001.)
    assert sum(cut_pairs) == want_pairs, f"{case}: radial-cut (kept, dropped) {cut_pairs}, want {want_pairs} pairs"


# Production MCMC driver rows: pattern -> expected route per predicate.
_DRIVER_CASES = {
    "mol_ecp_cart_j123": {"ecp_wdot_ok": True, "ecp_j3_wdot_ok": True, "det_carry_ok": True, "j3_carry_ok": True},
    "pbc_gamma_sphepgto_j123": {
        "ecp_wdot_ok": True,
        "ecp_j3_wdot_ok": True,
        "det_carry_ok": True,
        "j3_carry_ok": False,
    },
}
_DRIVER_STEPS = 7


@pytest.mark.parametrize("pattern", sorted(_DRIVER_CASES))
def test_mcmc_driver_route_predicate_counters(pattern, monkeypatch):
    """The production energy-only MCMC driver takes the counted predicate routes on every step.

    Two walkers, seven steps, seed 34356, random mesh, Rust backend on. Every
    driver step is one native fast-chunk step, so the carry counters equal
    walkers x steps and the w-dot counters twice that (one decision per spin).
    Molecular ccECP H2 with J1/J2/J3 engages all four predicates; the periodic
    Li pGTO cell engages both w-dot tables, declines the J3 carry (image sums) and carries
    the determinant state in both precision modes (screened periodic sweep columns).
    """
    from jqmc.jqmc_mcmc import MCMC

    monkeypatch.delenv("JQMC_FLAG_RUST_ON_CPU", raising=False)
    ham = rs_ham(pattern)
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
    mcmc.run(num_mcmc_steps=_DRIVER_STEPS)
    counters = list(jqmc_rs.route_predicate_counters())
    steps = _NW * _DRIVER_STEPS
    expected = _expected_counts(_DRIVER_CASES[pattern], steps)
    observed = {name: (counters[on], counters[off]) for name, (on, off) in _PREDICATE_INDICES.items()}
    assert observed == expected, f"{pattern}: production MCMC route-predicate counters {observed}"
    assert counters[16] == counters[17] == 0, f"{pattern}: complex ECP w-dot fired on a real run {counters}"
    assert counters[20] == counters[21] == 0, f"{pattern}: complex ECP seed fired on a real run {counters}"
    assert counters[_CPLX_CARRY_FIRST:_CPLX_CARRY_END] == [0] * (_CPLX_CARRY_END - _CPLX_CARRY_FIRST), (
        f"{pattern}: complex chunk carry counters fired on a real run {counters}"
    )
    cut_pairs = (counters[_RADIAL_CUT_KEPT], counters[_RADIAL_CUT_DROPPED])
    want_pairs = _expected_radial_cut_pairs(ham, steps)
    assert sum(cut_pairs) == want_pairs, (
        f"{pattern}: production MCMC radial-cut (kept, dropped) {cut_pairs}, want {want_pairs} pairs"
    )


# ``route_predicate_counters()`` (engaged, declined) of the complex-twist FAST chunk's
# AO-space ECP mesh w-dot (``ComplexOrb::mesh_wdot_ok``).
_CPLX_WDOT_INDICES = (16, 17)
# (carried, fresh) seeds of the complex FAST ECP mesh state.
_CPLX_SEED_INDICES = (20, 21)
# Complex-twist production MCMC rows (QE Li k=(1/4,1/4,1/4) ccECP): pGTO, pGTO+PW
# composite and pure B-spline determinant leaves. All are covered by the w-dot in both
# precision modes (the Bloch pGTO block is in its fp64 ``ao_bloch_eval`` zone; under an
# fp32 ``ao_eval`` zone the plane-wave / B-spline child is contracted from its fp32 values).
_CPLX_DRIVER_PATTERNS = ("twist_sphepgto_j12", "pgto_pw_twist_jnone", "bspline_twist_jnone")


@pytest.mark.parametrize("pattern", _CPLX_DRIVER_PATTERNS)
def test_mcmc_driver_complex_ecp_wdot_route_counters(pattern, monkeypatch):
    """The complex-twist FAST chunk takes the counted AO-space ECP mesh w-dot on every e_L.

    Two walkers, seven steps, seed 34356, random mesh, Rust backend on. Every driver
    step is one native complex fast-chunk e_L with one w-dot decision per spin, so the
    engaged count is exactly 2 x walkers x steps in both precision modes and the declined
    count is zero; the real-path ECP w-dot counters stay silent.
    """
    from jqmc.jqmc_mcmc import MCMC

    monkeypatch.delenv("JQMC_FLAG_RUST_ON_CPU", raising=False)
    ham = rs_ham(pattern)
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
    mcmc.run(num_mcmc_steps=_DRIVER_STEPS)
    counters = list(jqmc_rs.route_predicate_counters())
    decisions = 2 * _NW * _DRIVER_STEPS
    on, off = _CPLX_WDOT_INDICES
    expected = (decisions, 0)
    assert (counters[on], counters[off]) == expected, f"{pattern}: complex ECP w-dot counters {counters}"
    assert counters[0] == counters[1] == 0, f"{pattern}: real ECP w-dot fired on a twist run {counters}"
    # Every chunk e_L seeds its ECP mesh state on the sweep's carried base
    # (IMPL-RS-CPLX-ECP-CARRIED-SEED-001), never fresh.
    seed = (counters[_CPLX_SEED_INDICES[0]], counters[_CPLX_SEED_INDICES[1]])
    assert seed == (_NW * _DRIVER_STEPS, 0), f"{pattern}: complex ECP seed (carried, fresh) {seed}"


# Complex FAST chunk carry rows: the five QE Li k=(1/4,1/4,1/4) ccECP twist fixtures
# (pGTO J1/J2, J1/J2/J3, DTN; pGTO+PW and B-spline without Jastrow).
_CPLX_CARRY_PATTERNS = (
    "twist_sphepgto_j12",
    "twist_sphepgto_j123",
    "twist_sphepgto_jdtn",
    "pgto_pw_twist_jnone",
    "bspline_twist_jnone",
)


def _run_complex_chunk(ham):
    ctx = jqmc_rs.build_obs_ctx(
        ham.wavefunction_data.geminal_data,
        ham.wavefunction_data.jastrow_data,
        ham.coulomb_potential_data,
        True,
        True,
        False,
        precision_policy=precision_policy_snapshot(),
    )
    positions, charges, r_up, r_dn, _, _, keys, cell, n_max = _chunk_inputs_positions(ham)
    placeholder = np.zeros((_NW, 1, 1))
    jqmc_rs.gfmc_route_counters_reset()
    out = ctx.run_energy_chunk_batch(
        positions,
        charges,
        r_up,
        r_dn,
        placeholder,
        placeholder,
        2.0,
        1.0e-6,
        keys,
        np.zeros((_NW, 0, 0)),
        np.zeros((_NW, 0, 0, 0)),
        _W,
        _NUM_MOVES,
        cell,
        n_max,
        native_rng=True,
        random_mesh=True,
        base_grid_points=_GP,
        n_steps_fast=_STEPS,
        complex=True,
    )
    e_l = np.asarray(out["e_L"])
    assert e_l.shape == (_NW, _STEPS) and np.all(np.isfinite(e_l)), f"complex chunk e_L not produced: {e_l}"
    return int(np.sum(np.asarray(out["accepted"])))


def _chunk_inputs_positions(ham):
    """``_chunk_inputs`` without the real geminal (the complex chunk builds its own)."""
    gd = ham.wavefunction_data.geminal_data
    cp = ham.coulomb_potential_data
    positions = np.asarray(ham.structure_data.positions, float)
    charges = np.asarray(ham.structure_data.atomic_numbers, float) - np.asarray(cp.z_cores, float)
    cell = np.asarray(ham.structure_data.cell, float)
    n_max = int(ham.structure_data._n_max)
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(7)
    r_up = np.stack([positions[rng.integers(0, len(positions), nu)] + rng.normal(scale=0.6, size=(nu, 3)) for _ in range(_NW)])
    r_dn = np.stack([positions[rng.integers(0, len(positions), nd)] + rng.normal(scale=0.6, size=(nd, 3)) for _ in range(_NW)])
    keys = np.asarray([[34356 + w, 778899 + 3 * w] for w in range(_NW)], dtype=np.uint32)
    return positions, charges, r_up, r_dn, None, None, keys, cell, n_max


@pytest.mark.parametrize("hooked", [False, True], ids=["carried", "hooked"])
@pytest.mark.parametrize("pattern", _CPLX_CARRY_PATTERNS)
def test_mcmc_chunk_complex_carry_route_counters(pattern, hooked, monkeypatch):
    """The complex FAST chunk's kinetic-table and determinant-base carries are counted exactly.

    ``run_energy_chunk_batch(complex=True)``, two walkers, three steps, four moves per
    step. Carried: every walker step engages both carries (walkers x steps each), the
    kinetic tables are seeded once per walker and advanced once per accepted move (the
    chunk's ``accepted`` total), and the probe of their own inverse never re-seeds on
    these walks. Hooked (``JQMC_RS_KIN_COMPLEX_FRESH`` + ``JQMC_RS_DET_COMPLEX_FRESH_BASE``):
    every decision declines and the tables are never touched. Real-path predicates and
    the real carries stay silent.
    """
    for name in ("JQMC_RS_KIN_COMPLEX_FRESH", "JQMC_RS_DET_COMPLEX_FRESH_BASE"):
        if hooked:
            monkeypatch.setenv(name, "1")
        else:
            monkeypatch.delenv(name, raising=False)
    ham = rs_ham(pattern)
    accepted = _run_complex_chunk(ham)
    counters = list(jqmc_rs.route_predicate_counters())
    steps = _NW * _STEPS
    kin = (counters[_CPLX_KIN_ON], counters[_CPLX_KIN_OFF])
    base = (counters[_CPLX_BASE_ON], counters[_CPLX_BASE_OFF])
    tables = (counters[_CPLX_KIN_SEEDS], counters[_CPLX_KIN_ADVANCES], counters[_CPLX_KIN_RESEEDS])
    if hooked:
        assert (kin, base, tables) == ((0, steps), (0, steps), (0, 0, 0)), f"{pattern}: hooked carry counters {counters}"
    else:
        assert accepted > 0, f"{pattern}: vacuous chunk (no accepted move)"
        assert (kin, base, tables) == ((steps, 0), (steps, 0), (_NW, accepted, 0)), (
            f"{pattern}: carry counters {counters}, accepted {accepted}"
        )
    real = [counters[i] for i in (*_PREDICATE_INDICES["det_carry_ok"], *_PREDICATE_INDICES["j3_carry_ok"])]
    assert real == [0, 0, 0, 0], f"{pattern}: real chunk carries fired on a twist run {counters}"
