"""``build_obs_ctx`` == per-call wrappers.

The native observable context (``NativeObsCtx``) owns the Hamiltonian extracted
ONCE per run() instead of re-parsing it per walker per step. The thin ``py_*``
wrappers delegate to the SAME ``*_Args::compute`` the ctx methods call, so
wrapper == ctx by construction; this pins that BYTE-for-BYTE
(bit-exact, zero tolerance) for a representative ham of every family the driver threads a ctx
through -- molecular (AE + ECP), periodic real-Gamma, and complex twist -- across
e_L, HF force, Pulay force, HF stress, and Pulay stress. Any future drift between
the extract-once path and the wrapper path fails here.
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
from jqmc._setting import ATOL_NONVACUITY_DISTINCTION, RTOL_NONVACUITY_DISTINCTION

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import jqmc_rs  # noqa: E402

from jqmc.jqmc_mcmc import MCMC, _geminal_inv_batched, _NV_BASE_MESH, Nv_default  # noqa: E402
from jqmc._setting import MCMC_MIN_STEPS  # noqa: E402
from jqmc._precision import precision_policy_snapshot  # noqa: E402
from tests.test_rs_kernel._rs_fixtures import rs_ham  # noqa: E402

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_MESH = _NV_BASE_MESH[Nv_default]
_W = np.asarray(_MESH.weights, float)
_GP = np.asarray(_MESH.grid_points, float)


def _cfg(ham, seed, scale=0.6):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[rng.integers(0, len(R), nu)] + rng.normal(scale=scale, size=(nu, 3))
    rd = R[rng.integers(0, len(R), nd)] + rng.normal(scale=scale, size=(nd, 3))
    return ru, rd


def _ginv(ham, ru, rd, cplx):
    _, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    return np.asarray(ginv[0], dtype=(np.complex128 if cplx else np.float64))


def _eq(a, b):
    """Byte-identical (bit-exact, zero tolerance): the ctx path must not perturb a single ulp."""
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _triple_eq(x, y):
    for a, b in zip(x, y):
        _eq(a, b)


def _ctx(ham, complex_, pbc, want_stress):
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cp = ham.coulomb_potential_data
    return (
        jqmc_rs.build_obs_ctx(
            gd,
            jd,
            cp,
            complex_,
            pbc,
            want_stress,
            precision_policy=precision_policy_snapshot(),
        ),
        gd,
        jd,
        cp,
    )


def _sweep_inputs(ham, seed, cplx=False):
    """Per-step sweep args (positions/charges/r_up/r_dn/geminal/geminal_inv/normals)."""
    gd = ham.wavefunction_data.geminal_data
    cp = ham.coulomb_potential_data
    positions = np.asarray(ham.structure_data.positions, float)
    if bool(cp.ecp_flag):
        charges = np.asarray(ham.structure_data.atomic_numbers, float) - np.asarray(cp.z_cores, float)
    else:
        charges = np.asarray(ham.structure_data.atomic_numbers, float)
    ru, rd = _cfg(ham, seed=seed)
    g, ginv, _, _ = _geminal_inv_batched(gd, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    dt = np.complex128 if cplx else np.float64
    rng = np.random.default_rng(seed + 100)
    normals = rng.standard_normal(4).astype(np.float64)
    return (
        positions,
        charges,
        ru.astype(np.float64),
        rd.astype(np.float64),
        np.asarray(g[0], dt),
        np.asarray(ginv[0], dt),
        normals,
    )


@pytest.mark.parametrize("pattern", ["mol_cart_sd_j12", "mol_ecp_sphe_j12"])
def test_ctx_sweep_equiv_molecular(pattern):
    """The ctx-taking ``mcmc_sweep`` (Geminal/Jastrow/J3 extracted ONCE) reproduces
    the free ``_mcmc_sweep`` (per-call parse) BYTE-for-BYTE -- same kernel, same owned
    data, only the extraction is hoisted. Every emitted array + the advanced key match."""
    ham = rs_ham(pattern)
    gd, jd = ham.wavefunction_data.geminal_data, ham.wavefunction_data.jastrow_data
    pos, chg, ru, rd, g, ginv, normals = _sweep_inputs(ham, seed=3)
    dt, eps, k0, k1 = 2.0, 1.0e-6, 34356, 778899
    ref = jqmc_rs._mcmc_sweep(gd, jd, pos, chg, ru, rd, g, ginv, dt, eps, k0, k1, normals, None, 0)
    ctx, _, _, _ = _ctx(ham, complex_=False, pbc=False, want_stress=False)
    got = ctx.mcmc_sweep(pos, chg, ru, rd, g, ginv, dt, eps, k0, k1, normals, None, 0)
    assert set(got) == set(ref)
    for k in ref:
        _eq(got[k], ref[k])


def test_ctx_sweep_equiv_twist():
    """Complex: ctx ``mcmc_sweep_complex`` == free ``_mcmc_sweep_complex``, byte-exact."""
    ham = rs_ham("twist_sphepgto_j12")
    gd, jd = ham.wavefunction_data.geminal_data, ham.wavefunction_data.jastrow_data
    cell = np.asarray(ham.structure_data.cell, float)
    n_max = int(ham.structure_data._n_max)
    pos, chg, ru, rd, g, ginv, normals = _sweep_inputs(ham, seed=3, cplx=True)
    dt, eps, k0, k1 = 2.0, 1.0e-6, 34356, 778899
    ref = jqmc_rs._mcmc_sweep_complex(gd, jd, pos, chg, ru, rd, g, ginv, dt, eps, k0, k1, normals, cell, n_max)
    ctx, _, _, _ = _ctx(ham, complex_=True, pbc=True, want_stress=False)
    got = ctx.mcmc_sweep_complex(pos, chg, ru, rd, g, ginv, dt, eps, k0, k1, normals, cell, n_max)
    assert set(got) == set(ref)
    for k in ref:
        _eq(got[k], ref[k])


def test_complex_bspline_dtn_ctx_shares_geminal_and_matches_free_kernels(tmp_path, monkeypatch):
    """Complex B-spline DTN must not reparse its grid-sized geminal per observable.

    The v3 QE Gamma representation is intentionally used here: it is a complex
    B-spline leaf at exactly zero k, matching the twist-average production path.
    Besides numerical parity with the established free wrappers, this pins the
    memory ownership contract (one parsed geminal) and proves that the Python
    complex-param adapter delegates DTN to the fixed-per-run context.  Replacing
    any delegated call with a free wrapper is therefore a blocking failure.
    """
    from jqmc.jqmc_mcmc import _RsComplexPdCtx
    from tests.test_rs_kernel._rs_fixtures import build_qe_zero_k_complex_dtn_ham

    ham = build_qe_zero_k_complex_dtn_ham("bspline", tmp_path / "zero_k_bspline_dtn.h5")
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cp = ham.coulomb_potential_data
    ru, rd = _cfg(ham, seed=19)

    ctx = jqmc_rs.build_obs_ctx(
        gd,
        jd,
        cp,
        True,
        True,
        False,
        False,
        False,
        True,
        precision_policy=precision_policy_snapshot(),
    )
    assert ctx._debug_complex_geminal_storage_count() == 1, (
        "complex observable/DTN contexts must share one parsed ComplexGeminal; "
        "a B-spline parse materializes grid-sized coefficient storage"
    )
    assert ctx._debug_complex_bspline_coefficient_storage_count() == 2, (
        "time-reversal up/down B-spline leaves must share exactly the real/imag grid-major coefficient allocations"
    )
    independent_ctx = jqmc_rs.build_obs_ctx(
        gd.replace(time_reversal=False),
        jd,
        cp,
        True,
        True,
        False,
        False,
        False,
        False,
        precision_policy=precision_policy_snapshot(),
    )
    assert independent_ctx._debug_complex_bspline_coefficient_storage_count() == 4, (
        "time_reversal=False must retain independent up/down real/imag coefficient storage"
    )

    ref_ok = jqmc_rs.grad_ln_Psi_dtn_params(jd.jastrow_dtn_data, ru, rd)
    ref_t = jqmc_rs.grad_eL_dtn_kinetic_params_complex(gd, jd, ru, rd, True)
    ref_ecp = jqmc_rs.grad_eL_ecp_dtn_params_complex(gd, jd, cp, ru, rd, _W, _GP, True)
    got_ok = ctx.pd_ok_dtn(ru, rd)
    got_t = ctx.pd_el_dtn(ru, rd)
    got_ecp = ctx.pd_el_ecp_dtn(ru, rd, _W, _GP)
    _triple_eq(got_ok, ref_ok)
    _triple_eq(got_t, ref_t)
    _triple_eq(got_ecp, ref_ecp)

    def _free_wrapper_is_a_regression(*_args, **_kwargs):
        raise AssertionError("complex DTN production reparsed the geminal through a free wrapper")

    monkeypatch.setattr(jqmc_rs, "grad_ln_Psi_dtn_params", _free_wrapper_is_a_regression)
    monkeypatch.setattr(jqmc_rs, "grad_eL_dtn_kinetic_params_complex", _free_wrapper_is_a_regression)
    monkeypatch.setattr(jqmc_rs, "grad_eL_ecp_dtn_params_complex", _free_wrapper_is_a_regression)
    adapter = _RsComplexPdCtx(gd, jd, cp, ctx)
    _triple_eq(adapter.pd_ok_dtn(ru, rd), got_ok)
    _triple_eq(adapter.pd_el_dtn(ru, rd), got_t)
    _triple_eq(adapter.pd_el_ecp_dtn(ru, rd, _W, _GP), got_ecp)


def _chunk_static(ham):
    """positions / charges / (cell, n_max) for the native chunk + sweep calls."""
    cp = ham.coulomb_potential_data
    positions = np.asarray(ham.structure_data.positions, float)
    if bool(cp.ecp_flag):
        charges = np.asarray(ham.structure_data.atomic_numbers, float) - np.asarray(cp.z_cores, float)
    else:
        charges = np.asarray(ham.structure_data.atomic_numbers, float)
    if bool(ham.structure_data.pbc_flag):
        cell = np.asarray(ham.structure_data.cell, float)
        n_max = int(ham.structure_data._n_max)
    else:
        cell, n_max = None, 0
    return positions, charges, cell, n_max


def _chunk_rng(n_steps, num_moves, n_grid, seed):
    """Deterministic per-step injected normals + RT-rotated ECP grids for a chunk.
    (In the driver these come from the JAX threefry scan; here any fixed buffer
    exercises the SAME Rust threading, which is what this test pins.)"""
    rng = np.random.default_rng(seed)
    normals = np.ascontiguousarray(rng.standard_normal((n_steps, num_moves)))
    RTs = np.stack([np.linalg.qr(rng.standard_normal((3, 3)))[0] for _ in range(n_steps)])
    grids = np.ascontiguousarray(np.matmul(_GP, RTs))  # (n_steps, n_grid, 3)
    return normals, grids


def _run_chunk(ctx, ham, ru, rd, g, gi, normals, grids, num_moves, k0=34356, k1=778899, dt=2.0, eps=1.0e-6):
    pos, chg, cell, n_max = _chunk_static(ham)
    return ctx.run_energy_chunk(pos, chg, ru, rd, g, gi, dt, eps, k0, k1, normals, grids, _W, num_moves, cell, n_max)


@pytest.mark.parametrize("pattern", ["mol_ecp_sphe_j12", "mol_cart_sd_j12", "pbc_gamma_sphepgto_j12"])
def test_run_energy_chunk_equiv_per_step(pattern):
    """``run_energy_chunk`` (N steps of sweep + e_L threaded in Rust) is
    BYTE-for-BYTE identical to the per-step ``ctx.mcmc_sweep`` + ``ctx.e_L`` loop it
    replaces -- same injected normals / RT-rotated meshes, same seed. Pins the
    per-step e_L, the per-step geminal / geminal_inv (which feed the driver's JAX
    w_L), and the threaded final walker state + advanced key + accepted count."""
    ham = rs_ham(pattern)
    pbc = bool(ham.structure_data.pbc_flag)
    ctx, _, _, _ = _ctx(ham, complex_=False, pbc=pbc, want_stress=False)
    pos, chg, cell, n_max = _chunk_static(ham)
    ru, rd = _cfg(ham, seed=7)
    ru, rd = ru.astype(float), rd.astype(float)
    g, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    g0, gi0 = np.asarray(g[0], float), np.asarray(ginv[0], float)
    N, num_moves = 6, 4
    normals, grids = _chunk_rng(N, num_moves, len(_GP), seed=11)

    res = _run_chunk(ctx, ham, ru, rd, g0, gi0, normals, grids, num_moves)

    # per-step reference: same kernels the chunk fuses, threaded in Python.
    cru, crd, cg, cgi = ru.copy(), rd.copy(), g0.copy(), gi0.copy()
    key = np.array([34356, 778899], np.uint32)
    eL_ref, g_ref, gi_ref, acc_ref = [], [], [], 0
    for i in range(N):
        sw = ctx.mcmc_sweep(
            pos, chg, cru, crd, cg, cgi, 2.0, 1.0e-6, int(key[0]), int(key[1]), np.ascontiguousarray(normals[i]), cell, n_max
        )
        cru, crd = np.asarray(sw["r_up"]), np.asarray(sw["r_dn"])
        cg, cgi = np.asarray(sw["geminal"]), np.asarray(sw["geminal_inv"])
        key = np.asarray(sw["final_key"], np.uint32)
        acc_ref += int(sw["accepted"])
        eL_ref.append(ctx.e_L(cru, crd, _W, np.ascontiguousarray(grids[i]), cgi))
        g_ref.append(cg.copy())
        gi_ref.append(cgi.copy())

    _eq(res["e_L"], np.array(eL_ref))
    _eq(res["geminal"], np.array(g_ref))
    _eq(res["geminal_inv"], np.array(gi_ref))
    _eq(res["r_up"], cru)
    _eq(res["r_dn"], crd)
    _eq(np.asarray(res["final_key"], np.uint32), key)
    assert int(res["accepted"]) == acc_ref


@pytest.mark.parametrize("pattern", ["mol_ecp_sphe_j12", "mol_cart_sd_j12", "pbc_gamma_sphepgto_j12", "bspline_gamma_jnone"])
def test_run_energy_chunk_forces_equiv_per_step(pattern):
    """The BITEXACT ``run_energy_chunk`` (want_force[/ want_stress]) is
    BYTE-for-BYTE identical to the per-step ``ctx.mcmc_sweep`` + ``ctx.e_L`` + ``ctx.grad_*``
    loop it replaces -- same injected normals / RT-rotated meshes, same seed. Pins the per-step
    HF force / Pulay force (dR, dr_up, dr_dn), the periodic HF / Pulay stress (3x3), and the
    per-step positions the driver's SWCT drift assembly consumes, as well as the threaded final
    walker state. The chunk folds these SAME ``*_Args::compute`` kernels the per-step ctx methods
    call into one loop, so the buffers must not perturb a single ulp."""
    ham = rs_ham(pattern)
    pbc = bool(ham.structure_data.pbc_flag)
    ecp = bool(ham.coulomb_potential_data.ecp_flag)
    ctx, _, _, _ = _ctx(ham, complex_=False, pbc=pbc, want_stress=pbc)
    pos, chg, cell, n_max = _chunk_static(ham)
    ru, rd = _cfg(ham, seed=7)
    ru, rd = ru.astype(float), rd.astype(float)
    g, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    g0, gi0 = np.asarray(g[0], float), np.asarray(ginv[0], float)
    N, num_moves = 6, 4
    normals, grids = _chunk_rng(N, num_moves, len(_GP), seed=11)

    res = ctx.run_energy_chunk(
        pos,
        chg,
        ru,
        rd,
        g0,
        gi0,
        2.0,
        1.0e-6,
        34356,
        778899,
        normals,
        grids,
        _W,
        num_moves,
        cell,
        n_max,
        native_rng=False,
        want_force=True,
        want_stress=pbc,
    )

    # Per-step reference: the SAME ctx kernels the chunk fuses, threaded in Python.
    cru, crd, cg, cgi = ru.copy(), rd.copy(), g0.copy(), gi0.copy()
    key = np.array([34356, 778899], np.uint32)
    hfR_ref, hfru_ref, hfrd_ref = [], [], []
    ppR_ref, ppru_ref, pprd_ref = [], [], []
    rus_ref, rds_ref, sHF_ref, sPP_ref = [], [], [], []
    for i in range(N):
        sw = ctx.mcmc_sweep(
            pos, chg, cru, crd, cg, cgi, 2.0, 1.0e-6, int(key[0]), int(key[1]), np.ascontiguousarray(normals[i]), cell, n_max
        )
        cru, crd = np.asarray(sw["r_up"]), np.asarray(sw["r_dn"])
        cg, cgi = np.asarray(sw["geminal"]), np.asarray(sw["geminal_inv"])
        key = np.asarray(sw["final_key"], np.uint32)
        gp_i = np.ascontiguousarray(grids[i])
        if pbc:
            hf = ctx.grad_e_L_pbc(cru, crd, _W, gp_i, cgi)
            pp = ctx.grad_ln_psi_pbc(cru, crd, cgi)
            sHF_ref.append(np.asarray(ctx.grad_e_L_eps_pbc(cru, crd, _W, gp_i, cgi)))
            sPP_ref.append(np.asarray(ctx.grad_ln_psi_eps_pbc(cru, crd, cgi)))
        elif ecp:
            hf = ctx.grad_e_L_ecp(cru, crd, _W, gp_i, cgi)
            pp = ctx.grad_ln_psi(cru, crd, cgi)
        else:
            hf = ctx.grad_e_L_ae(cru, crd, cgi)
            pp = ctx.grad_ln_psi(cru, crd, cgi)
        hfR_ref.append(np.asarray(hf[0]))
        hfru_ref.append(np.asarray(hf[1]))
        hfrd_ref.append(np.asarray(hf[2]))
        ppR_ref.append(np.asarray(pp[0]))
        ppru_ref.append(np.asarray(pp[1]))
        pprd_ref.append(np.asarray(pp[2]))
        rus_ref.append(cru.copy())
        rds_ref.append(crd.copy())

    _eq(res["hf_dR"], np.array(hfR_ref))
    _eq(res["hf_dr_up"], np.array(hfru_ref))
    _eq(res["hf_dr_dn"], np.array(hfrd_ref))
    _eq(res["pp_dR"], np.array(ppR_ref))
    _eq(res["pp_dr_up"], np.array(ppru_ref))
    _eq(res["pp_dr_dn"], np.array(pprd_ref))
    _eq(res["r_up_steps"], np.array(rus_ref))
    _eq(res["r_dn_steps"], np.array(rds_ref))
    _eq(res["r_up"], cru)
    _eq(res["r_dn"], crd)
    _eq(np.asarray(res["final_key"], np.uint32), key)
    if pbc:
        _eq(res["stress_HF"], np.array(sHF_ref))
        _eq(res["stress_PP"], np.array(sPP_ref))


def test_run_energy_chunk_forces_complex_gated_to_per_step():
    """Chunk gate (twist row): the observables chunk is REAL-only (it drives the real
    ``mcmc_sweep`` + real ``ClefArgs``). A complex-twist ham builds a complex ctx whose real
    ``sweep``/``clef`` args are absent, so ``run_energy_chunk`` raises the ctx-missing guard --
    the driver keeps complex-twist force/stress on the per-step dispatch (documented). The
    per-step complex force/stress kernels themselves are pinned by ``test_ctx_equiv_twist``."""
    ham = rs_ham("twist_sphepgto_j12")
    ctx, _, _, _ = _ctx(ham, complex_=True, pbc=True, want_stress=True)
    pos, chg, cell, n_max = _chunk_static(ham)
    ru, rd = _cfg(ham, seed=7)
    ru, rd = ru.astype(float), rd.astype(float)
    g, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    g0, gi0 = np.asarray(g[0], np.complex128), np.asarray(ginv[0], np.complex128)
    normals, grids = _chunk_rng(4, 4, len(_GP), seed=11)
    with pytest.raises(RuntimeError, match="sweep_real"):
        ctx.run_energy_chunk(
            pos,
            chg,
            ru,
            rd,
            np.real(g0),
            np.real(gi0),
            2.0,
            1.0e-6,
            1,
            2,
            normals,
            grids,
            _W,
            4,
            cell,
            n_max,
            want_force=True,
            want_stress=True,
        )


def test_run_energy_chunk_complex_energy_only_and_bitexact_gated():
    """Complex-twist chunk is FAST-mode + energy-only. ``complex=True`` with ``native_rng=False``
    (bitexact) or with ``want_force`` raises loudly (strict-raise) -- the driver keeps bitexact
    complex + complex force/stress on the per-step dispatch. The energy chunk itself is pinned by
    ``test_run_energy_chunk_complex_fast_equiv_per_step``."""
    ham = rs_ham("twist_sphepgto_j12")
    ctx, _, _, _ = _ctx(ham, complex_=True, pbc=True, want_stress=False)
    pos, chg, cell, n_max = _chunk_static(ham)
    ru, rd = _cfg(ham, seed=7)
    ru, rd = ru.astype(float), rd.astype(float)
    dummy = np.zeros((1, 1))
    empty_n, empty_g = np.zeros((0, 0)), np.zeros((0, 0, 0))
    # bitexact (native_rng defaults False) complex chunk -> raise.
    with pytest.raises(ValueError, match="fast mode only"):
        ctx.run_energy_chunk(
            pos, chg, ru, rd, dummy, dummy, 2.0, 1.0e-6, 1, 2, empty_n, empty_g, _W, 4, cell, n_max, complex=True
        )
    # fast complex chunk with a position-derivative request -> raise (energy-only).
    with pytest.raises(ValueError, match="energy-only"):
        ctx.run_energy_chunk(
            pos,
            chg,
            ru,
            rd,
            dummy,
            dummy,
            2.0,
            1.0e-6,
            1,
            2,
            empty_n,
            empty_g,
            _W,
            4,
            cell,
            n_max,
            native_rng=True,
            random_mesh=True,
            base_grid_points=_GP,
            n_steps_fast=3,
            complex=True,
            want_force=True,
        )


def test_run_energy_chunk_complex_fast_equiv_per_step(monkeypatch):
    """Complex-twist FAST chunk: an N-step ``run_energy_chunk`` (``complex=True``, ``native_rng``)
    vs N threaded SINGLE-step chunks. Fast mode is deterministic given the walker key + positions
    (native threefry normals + native RT rotation), so this pins the chunk's per-step complex e_L,
    the native complex AS-reg ``w_L``, and the final walker-state / key threading in the ONLY mode
    the complex chunk supports (bitexact complex stays on the per-step JAX-RNG dispatch,
    driver-gated). Reuses the SAME complex sweep + complex e_L kernels the per-step ctx methods
    drive (pinned bit-exact by ``test_ctx_equiv_twist``).

    With the test-only hooks ``JQMC_RS_KIN_COMPLEX_FRESH`` and ``JQMC_RS_DET_COMPLEX_FRESH_BASE``
    (fresh kinetic every e_L, base rebuilt every step: the pre-carry chunk) the two agree
    BYTE-for-BYTE. With the production carries (IMPL-RS-CPLX-CHUNK-KIN-CARRY-001) the N-step chunk
    threads its kinetic tables and determinant base across steps while every single-step chunk
    seeds them afresh, so the walk (keys, positions, acceptances) stays byte-identical and e_L /
    w_L agree within the carry floors (ATOL/RTOL_CPLX_KIN_CARRY + ATOL/RTOL_CPLX_DET_BASE_CARRY,
    raised to the ao_eval / jastrow_grad_lap zone envelopes in the mixed lane)."""
    from jqmc._setting import (
        ATOL_CPLX_DET_BASE_CARRY,
        ATOL_CPLX_KIN_CARRY,
        RTOL_CPLX_DET_BASE_CARRY,
        RTOL_CPLX_KIN_CARRY,
        get_tolerance_xla_rust_consistency,
    )

    ham = rs_ham("twist_sphepgto_j12")
    ctx, _, _, _ = _ctx(ham, complex_=True, pbc=True, want_stress=False)
    pos, chg, cell, n_max = _chunk_static(ham)
    ru, rd = _cfg(ham, seed=7)
    ru, rd = ru.astype(float), rd.astype(float)
    dummy = np.zeros((1, 1))
    empty_n, empty_g = np.zeros((0, 0)), np.zeros((0, 0, 0))
    N, num_moves = 6, 4
    k0, k1 = 34356, 778899

    def _chunk(r_up, r_dn, kk0, kk1, n):
        return ctx.run_energy_chunk(
            pos,
            chg,
            r_up,
            r_dn,
            dummy,
            dummy,
            2.0,
            1.0e-6,
            kk0,
            kk1,
            empty_n,
            empty_g,
            _W,
            num_moves,
            cell,
            n_max,
            native_rng=True,
            random_mesh=True,
            base_grid_points=_GP,
            n_steps_fast=n,
            complex=True,
        )

    def _both():
        full = _chunk(ru, rd, k0, k1, N)
        cru, crd = ru.copy(), rd.copy()
        key = np.array([k0, k1], np.uint32)
        eL_ref, wL_ref, acc_ref = [], [], 0
        for _ in range(N):
            r = _chunk(cru, crd, int(key[0]), int(key[1]), 1)
            cru, crd = np.asarray(r["r_up"]), np.asarray(r["r_dn"])
            key = np.asarray(r["final_key"], np.uint32)
            eL_ref.append(np.asarray(r["e_L"])[0])
            wL_ref.append(np.asarray(r["w_L"])[0])
            acc_ref += int(r["accepted"])
        _eq(np.asarray(full["r_up"]), cru)
        _eq(np.asarray(full["r_dn"]), crd)
        _eq(np.asarray(full["final_key"], np.uint32), key)
        assert int(full["accepted"]) == acc_ref
        return (np.asarray(full["e_L"]), np.array(eL_ref)), (np.asarray(full["w_L"]), np.array(wL_ref))

    # Pre-carry route: byte-for-byte.
    monkeypatch.setenv("JQMC_RS_KIN_COMPLEX_FRESH", "1")
    monkeypatch.setenv("JQMC_RS_DET_COMPLEX_FRESH_BASE", "1")
    (e_full, e_ref), (w_full, w_ref) = _both()
    _eq(e_full, e_ref)
    _eq(w_full, w_ref)

    # Production carries: same walk, e_L / w_L within the carry floors.
    monkeypatch.delenv("JQMC_RS_KIN_COMPLEX_FRESH")
    monkeypatch.delenv("JQMC_RS_DET_COMPLEX_FRESH_BASE")
    (e_full, e_ref), (w_full, w_ref) = _both()
    atol = ATOL_CPLX_KIN_CARRY + ATOL_CPLX_DET_BASE_CARRY
    rtol = RTOL_CPLX_KIN_CARRY + RTOL_CPLX_DET_BASE_CARRY
    snap = precision_policy_snapshot()
    zones = ("ao_eval", "jastrow_grad_lap")
    if any(snap.get(z, "float64") != "float64" for z in zones):
        zone_atol, zone_rtol = get_tolerance_xla_rust_consistency(zones)
        atol, rtol = max(atol, zone_atol), max(rtol, zone_rtol)
    np.testing.assert_allclose(e_full, e_ref, atol=atol, rtol=rtol)
    np.testing.assert_allclose(w_full, w_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("pattern", ["mol_cart_sd_j12", "pbc_gamma_sphepgto_j12"])
def test_run_energy_chunk_boundary_split(pattern):
    """Chunk-boundary semantics: 2 chunks of N/2 (threading the final walker
    state + geminal / geminal_inv + advanced key of the first into the second) equal
    1 chunk of N, BYTE-for-BYTE. Guards the driver's cross-chunk state hand-off."""
    ham = rs_ham(pattern)
    pbc = bool(ham.structure_data.pbc_flag)
    ctx, _, _, _ = _ctx(ham, complex_=False, pbc=pbc, want_stress=False)
    ru, rd = _cfg(ham, seed=3)
    ru, rd = ru.astype(float), rd.astype(float)
    g, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    g0, gi0 = np.asarray(g[0], float), np.asarray(ginv[0], float)
    N, num_moves = 8, 4
    normals, grids = _chunk_rng(N, num_moves, len(_GP), seed=5)
    k0, k1 = 12345, 67890

    full = _run_chunk(ctx, ham, ru, rd, g0, gi0, normals, grids, num_moves, k0=k0, k1=k1)

    h = N // 2
    c1 = _run_chunk(ctx, ham, ru, rd, g0, gi0, normals[:h], grids[:h], num_moves, k0=k0, k1=k1)
    fk = np.asarray(c1["final_key"], np.uint32)
    c2 = _run_chunk(
        ctx,
        ham,
        np.asarray(c1["r_up"]),
        np.asarray(c1["r_dn"]),
        np.asarray(c1["geminal"])[-1],
        np.asarray(c1["geminal_inv"])[-1],
        normals[h:],
        grids[h:],
        num_moves,
        k0=int(fk[0]),
        k1=int(fk[1]),
    )

    _eq(full["e_L"], np.concatenate([np.asarray(c1["e_L"]), np.asarray(c2["e_L"])]))
    _eq(full["geminal"], np.concatenate([np.asarray(c1["geminal"]), np.asarray(c2["geminal"])]))
    _eq(full["geminal_inv"], np.concatenate([np.asarray(c1["geminal_inv"]), np.asarray(c2["geminal_inv"])]))
    _eq(full["r_up"], c2["r_up"])
    _eq(full["r_dn"], c2["r_dn"])
    _eq(np.asarray(full["final_key"], np.uint32), np.asarray(c2["final_key"], np.uint32))
    assert int(full["accepted"]) == int(c1["accepted"]) + int(c2["accepted"])


def test_ctx_lazy_force_args_skipped_energy_only():
    """Laziness: with ``want_force=False`` the ctx builds the e_L + sweep args but NOT the
    force/Pulay args -- calling a force method then raises the ctx-missing guard (proving the
    ~4.8 s pgto_force_leaves extraction was skipped), while e_L + sweep still work."""
    ham = rs_ham("mol_cart_sd_j12")
    gd, jd, cp = ham.wavefunction_data.geminal_data, ham.wavefunction_data.jastrow_data, ham.coulomb_potential_data
    ctx = jqmc_rs.build_obs_ctx(
        gd,
        jd,
        cp,
        False,
        False,
        False,
        False,
        precision_policy=precision_policy_snapshot(),
    )  # want_stress=False, want_force=False
    pos, chg, ru, rd, g, ginv, normals = _sweep_inputs(ham, seed=5)
    # e_L + sweep are always built -> work.
    _ = ctx.e_L(ru, rd, _W, _GP, ginv)
    _ = ctx.mcmc_sweep(pos, chg, ru, rd, g, ginv, 2.0, 1.0e-6, 1, 2, normals, None, 0)
    # force args were NOT built -> the ctx-missing guard fires.
    with pytest.raises(RuntimeError, match="mol_force"):
        ctx.grad_e_L_ae(ru, rd, ginv)


def _perturb_params(ham):
    """Return a NEW ham object with the two-body Jastrow variational parameter shifted -- an
    optimizer-style param update (new pytree, new id()). The Jastrow enters the kinetic energy,
    so this genuinely changes e_L (unlike a global geminal scale, which leaves e_L invariant)."""
    leaves, treedef = jax.tree_util.tree_flatten_with_path(ham)
    new = []
    for path, leaf in leaves:
        ps = jax.tree_util.keystr(path)
        if ps.endswith(".jastrow_2b_param"):
            new.append(leaf + 0.2)
        else:
            new.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, new)


def test_ctx_cache_invalidation_on_param_update(monkeypatch):
    """Invalidation contract. The run-level ctx is cached ACROSS run() calls keyed on the
    IDENTITY of ``hamiltonian_data``: (i) a repeat run() on the same ham REUSES the ctx; (ii) an
    optimizer-style param update (new ham object) REBUILDS it -- a stale ctx would be silent
    wrongness. Also pins that the rebuilt ctx reflects the NEW params (e_L changes) and equals a
    fresh build from the updated ham."""
    monkeypatch.delenv("JQMC_FLAG_RUST_ON_CPU", raising=False)  # -> rust backend on (default)
    ham = rs_ham("mol_cart_sd_j12")
    mcmc = MCMC(
        hamiltonian_data=ham,
        Dt=2.0,
        mcmc_seed=34356,
        epsilon_AS=1.0e-6,
        num_walkers=2,
        comput_position_deriv=False,  # energy-only -> also exercises the lazy (no-force) build
        random_discretized_mesh=True,
    )
    mcmc.run(num_mcmc_steps=MCMC_MIN_STEPS)
    ctx1 = mcmc._MCMC__rs_run_ctx
    assert ctx1 is not None, "rust backend must have built + cached the run ctx"
    assert mcmc._MCMC__rs_run_ctx_ham is ham

    # (i) same ham object -> reuse the cached ctx (no rebuild).
    mcmc.run(num_mcmc_steps=MCMC_MIN_STEPS)
    assert mcmc._MCMC__rs_run_ctx is ctx1, "a repeat run() on the same ham must reuse the ctx"

    # (ii) param update (new ham object) -> ctx invalidated + rebuilt.
    ham2 = _perturb_params(ham)
    mcmc.hamiltonian_data = ham2
    mcmc.run(num_mcmc_steps=MCMC_MIN_STEPS)
    ctx2 = mcmc._MCMC__rs_run_ctx
    assert ctx2 is not ctx1, "a param update must REBUILD the ctx (no stale reuse)"
    assert mcmc._MCMC__rs_run_ctx_ham is ham2

    # The rebuilt ctx reflects the NEW params, and equals a fresh build from ham2. Evaluate e_L on
    # a fixed config with a shared ginv, so any difference is the ctx-owned geminal (the params).
    ru, rd = _cfg(ham, seed=7)
    ginv = _ginv(ham, ru, rd, cplx=False)
    eL1 = ctx1.e_L(ru, rd, _W, _GP, ginv)
    eL2 = ctx2.e_L(ru, rd, _W, _GP, ginv)
    assert not np.isclose(eL1, eL2, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION), (
        "the rebuilt ctx must reflect the updated params (e_L changed)"
    )
    fresh = jqmc_rs.build_obs_ctx(
        ham2.wavefunction_data.geminal_data,
        ham2.wavefunction_data.jastrow_data,
        ham2.coulomb_potential_data,
        False,
        False,
        False,
        False,
        precision_policy=precision_policy_snapshot(),
    )
    _eq(eL2, fresh.e_L(ru, rd, _W, _GP, ginv))


@pytest.mark.parametrize("pattern", ["mol_ecp_sphe_j12", "mol_cart_sd_j12"])
def test_ctx_equiv_molecular(pattern):
    """Ctx equiv molecular."""
    ham = rs_ham(pattern)
    ecp = bool(ham.coulomb_potential_data.ecp_flag)
    ru, rd = _cfg(ham, seed=1)
    ginv = _ginv(ham, ru, rd, cplx=False)
    ctx, gd, jd, cp = _ctx(ham, complex_=False, pbc=False, want_stress=False)

    _eq(jqmc_rs.compute_local_energy_fast(ham, ru, rd, _W, _GP, ginv), ctx.e_L(ru, rd, _W, _GP, ginv))
    if ecp:
        _triple_eq(jqmc_rs.vmc_grad_e_L_ecp(gd, jd, cp, ru, rd, _W, _GP, ginv), ctx.grad_e_L_ecp(ru, rd, _W, _GP, ginv))
    else:
        _triple_eq(jqmc_rs.vmc_grad_e_L_ae(gd, jd, cp, ru, rd, ginv), ctx.grad_e_L_ae(ru, rd, ginv))
    _triple_eq(jqmc_rs.vmc_grad_ln_psi(gd, jd, ru, rd, ginv), ctx.grad_ln_psi(ru, rd, ginv))


@pytest.mark.parametrize("pattern", ["pbc_gamma_sphepgto_j12", "bspline_gamma_jnone"])
def test_ctx_equiv_pbc_real(pattern):
    """Ctx equiv PBC real."""
    ham = rs_ham(pattern)
    ru, rd = _cfg(ham, seed=1)
    ginv = _ginv(ham, ru, rd, cplx=False)
    ctx, gd, jd, cp = _ctx(ham, complex_=False, pbc=True, want_stress=True)

    _eq(jqmc_rs.compute_local_energy_fast(ham, ru, rd, _W, _GP, ginv), ctx.e_L(ru, rd, _W, _GP, ginv))
    _triple_eq(
        jqmc_rs.vmc_grad_e_L_pbc(gd, jd, cp, ru, rd, _W, _GP, ginv),
        ctx.grad_e_L_pbc(ru, rd, _W, _GP, ginv),
    )
    _triple_eq(jqmc_rs.vmc_grad_ln_psi_pbc(gd, jd, ru, rd, ginv), ctx.grad_ln_psi_pbc(ru, rd, ginv))
    _eq(
        jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, ru, rd, _W, _GP, ginv),
        ctx.grad_e_L_eps_pbc(ru, rd, _W, _GP, ginv),
    )
    _eq(jqmc_rs.vmc_grad_ln_psi_eps_pbc(gd, jd, ru, rd, ginv), ctx.grad_ln_psi_eps_pbc(ru, rd, ginv))


@pytest.mark.parametrize("pattern", ["twist_sphepgto_j12"])
def test_ctx_equiv_twist(pattern):
    """Ctx equiv twist."""
    ham = rs_ham(pattern)
    ru, rd = _cfg(ham, seed=1)
    ginv = _ginv(ham, ru, rd, cplx=True)
    ctx, gd, jd, cp = _ctx(ham, complex_=True, pbc=True, want_stress=True)

    _eq(
        jqmc_rs.compute_local_energy_fast_complex(ham, ru, rd, _W, _GP, ginv),
        ctx.e_L_complex(ru, rd, _W, _GP, ginv),
    )
    _triple_eq(
        jqmc_rs.vmc_grad_e_L_pbc_complex(gd, jd, cp, ru, rd, _W, _GP, None),
        ctx.grad_e_L_pbc_complex(ru, rd, _W, _GP),
    )
    _triple_eq(
        jqmc_rs.vmc_grad_ln_psi_pbc_complex(gd, jd, ru, rd),
        ctx.grad_ln_psi_pbc_complex(ru, rd),
    )
    _eq(
        jqmc_rs.vmc_grad_e_L_eps_pbc_complex(gd, jd, cp, ru, rd, _W, _GP),
        ctx.grad_e_L_eps_pbc_complex(ru, rd, _W, _GP),
    )
    _eq(
        jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(gd, jd, ru, rd),
        ctx.grad_ln_psi_eps_pbc_complex(ru, rd),
    )
