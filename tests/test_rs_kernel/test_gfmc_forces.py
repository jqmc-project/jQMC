"""Layer-1 equivalence: GFMC (LRDMC) force observables, Rust vs jax.grad oracle.

Fixtures are committed-fixture patterns via
``tests.test_rs_kernel._rs_fixtures.rs_ham``.

Region (4) derivatives: the HF force differentiates the LRDMC
lattice-discretised diagonal local energy `_compute_local_energy_t` (jqmc_gfmc.py
-- a run-local closure, reconstructed here 1:1 from public jqmc components, the
same way test_gfmc.py builds its e_L reference), and the Pulay force
differentiates ln|Psi|. The oracle is jax.grad over (R via
_inject_active_positions, r_up, r_dn) -- exactly production's
`_jit_vmap_grad_e_L_t` / `_grad_ln_psi_rR` semantics (the variational-param
grad mask does not affect position axes). This oracle reconstruction is
self-contained (built directly from jqmc
primitives -- arguably a STRONGER oracle than delegating to
``jqmc.derivatives.gfmc``).

Every system is the closest
committed-fixture pattern from ``_rs_fixtures._PATTERNS``; multiple distinct
molecules that map to the SAME code path (e.g. H2/N closed vs open
shell SD) collapse onto a single committed pattern (a Rust==XLA equivalence is
path-sensitive, not system-sensitive). A few parametrize rows have no committed
equivalent (ECP+J3, composite+J3/DTN, molecular ECP+DTN) and are dropped or
skipped explicitly -- see the ``Coverage gap`` notes below.
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

from jqmc._precision import get_tolerance_xla_rust_consistency, is_mixed_precision_enabled  # noqa: E402
from jqmc._setting import ATOL_RS_GFMC_SPRIME_FAST_VS_FULL, RTOL_RS_GFMC_SPRIME_FAST_VS_FULL  # noqa: E402
from jqmc.derivatives import gfmc as xla_gfmc  # noqa: E402
from tests.test_rs_kernel._rs_fixtures import rs_ham  # noqa: E402

# Every rs test declares its cost tier (light/heavy) and CWD policy (with_chdir
# here -- autouse _isolate_cwd chdirs into a per-test tmp_path for xdist safety;
# committed fixtures load via absolute paths so cwd is irrelevant).
pytestmark = [pytest.mark.with_chdir]

ALAT = 0.30

# --------------------------------------------------------------------------- #
# Direct parity uses the dedicated XLA--Rust consistency tolerance. There is no
# molecular/periodic strict/medium selector; these tuples only declare the
# precision zones crossed by each kernel.
# --------------------------------------------------------------------------- #
_ZONES_MOL = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap")
_ZONES_PBC = _ZONES_MOL + ("ewald",)

_ATOL_MOL, _RTOL_MOL = get_tolerance_xla_rust_consistency(_ZONES_MOL, level="medium")
_ATOL_PBC, _RTOL_PBC = get_tolerance_xla_rust_consistency(_ZONES_PBC, level="medium")


def _withj_stress_fast_vs_full_tol():
    """Tolerance of the production periodic withJ GFMC stress against its ``debug=True`` oracle.

    Production evaluates the ECP mesh by rank-1 hops off the base geminal inverse, while the
    frozen oracle rebuilds every reflected configuration. In fp64 the two differ only by the
    rank-one/GEMM reduction order (the 1e-8 S' twin pair). In mixed precision the rank-1
    candidate ratio reads the fused AO values and the one-electron Jastrow ratio zone, while the
    oracle's fresh det ratio reads the standalone fp32 ``ao_eval`` zone and the full Jastrow
    value, so the pair differs at the fp32 zone envelope -- the same pair the VMC mesh-stress
    fast-vs-reference test uses.
    """
    if is_mixed_precision_enabled():
        return _ATOL_PBC, _RTOL_PBC
    return ATOL_RS_GFMC_SPRIME_FAST_VS_FULL, RTOL_RS_GFMC_SPRIME_FAST_VS_FULL


@pytest.fixture(autouse=True)
def _clear_jax_caches_between_force_cells():
    """Release compiled oracle programs after each distinct force/stress cell.

    This module runs more than 200 heterogeneous JAX derivative programs per
    precision mode.  Keeping them all alive across one full-then-mixed session
    eventually segfaults inside JAX's compiler on Python 3.12/JAX 0.7.2.  The
    production kernels and their operation order are unaffected.
    """
    yield
    jax.clear_caches()


def _cfg(ham, seed):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = gd.num_electron_up, gd.num_electron_dn
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[rng.integers(0, len(R), nu)] + rng.normal(scale=0.6, size=(nu, 3))
    rd = R[rng.integers(0, len(R), nd)] + rng.normal(scale=0.6, size=(nd, 3))
    return ru, rd


def _pbc_cfg(ham, seed, scale=0.4):
    """Electron-axis sample for a periodic (Ne1-1) fixture: up near atom 0, dn
    near atom 1."""
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[:nu] + rng.normal(scale=scale, size=(nu, 3))
    rd = R[nu : nu + nd] + rng.normal(scale=scale, size=(nd, 3))
    return ru, rd


def _rand_j3(wf, seed=7, scale=0.15):
    j3 = wf.jastrow_data.jastrow_three_body_data
    n = np.asarray(j3.j_matrix).shape[0]
    jm = np.random.default_rng(seed).normal(scale=scale, size=(n, n + 1))
    # Production invariant: the J3 coupling block is symmetric (bosonic Jastrow);
    # Jastrow_three_body_data.sanity_check RAISES otherwise, and several jqmc
    # streaming/ratio kernels exploit it (j3_mat.T @ x == j3_mat @ x). A .replace()
    # bypasses sanity_check, so symmetrize here or we test an invalid regime.
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)
    return wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=j3.replace(j_matrix=jm)))


def _inject(obj, positions):
    from jqmc.jqmc_mcmc import _inject_active_positions

    return _inject_active_positions(obj, {"positions": positions})


def _e_L_t(ham, ru, rd, alat, RT):
    """Continuum measurement energy for the all-electron ``novar=0`` route."""
    from jqmc.coulomb_potential import (
        compute_coulomb_potential_ee_dispatch as ee_d,
    )
    from jqmc.coulomb_potential import (
        compute_coulomb_potential_ei_dispatch as ei_d,
    )
    from jqmc.coulomb_potential import (
        compute_coulomb_potential_ii_dispatch as ii_d,
    )
    from jqmc.wavefunction import compute_kinetic_energy_all_elements

    wf = ham.wavefunction_data
    cd = ham.coulomb_potential_data
    kc_u, kc_dn = compute_kinetic_energy_all_elements(wf, ru, rd)
    elel = ee_d(coulomb_potential_data=cd, r_up_carts=ru, r_dn_carts=rd)
    iiv = ii_d(coulomb_potential_data=cd)
    ei_u, ei_dn = ei_d(coulomb_potential_data=cd, r_up_carts=ru, r_dn_carts=rd)
    return elel + iiv + jnp.sum(ei_u + kc_u) + jnp.sum(ei_dn + kc_dn)


def _e_L_t_ecp(ham, ru, rd, alat, RT, dltmove):
    """Continuum ECP measurement energy for the ``novar=0`` route."""
    from jqmc.coulomb_potential import (
        compute_coulomb_potential_ee_dispatch as ee_d,
    )
    from jqmc.coulomb_potential import (
        compute_coulomb_potential_ei_dispatch as ei_d,
    )
    from jqmc.coulomb_potential import (
        compute_coulomb_potential_ii_dispatch as ii_d,
    )
    from jqmc.coulomb_potential import (
        compute_ecp_local_parts_all_pairs,
    )
    from jqmc.coulomb_potential import (
        compute_ecp_non_local_parts_nearest_neighbors as nn_ecp,
    )
    from jqmc.jastrow_factor import compute_Jastrow_part
    from jqmc.wavefunction import compute_kinetic_energy_all_elements

    wf = ham.wavefunction_data
    cd = ham.coulomb_potential_data
    jd = wf.jastrow_data
    kc_u, kc_dn = compute_kinetic_energy_all_elements(wf, ru, rd)
    ei_u, ei_dn = ei_d(coulomb_potential_data=cd, r_up_carts=ru, r_dn_carts=rd)
    elel = ee_d(coulomb_potential_data=cd, r_up_carts=ru, r_dn_carts=rd)
    iiv = ii_d(coulomb_potential_data=cd)
    ecpl = compute_ecp_local_parts_all_pairs(coulomb_potential_data=cd, r_up_carts=ru, r_dn_carts=rd)
    if dltmove:
        mu, md, Vdet, _ = nn_ecp(
            coulomb_potential_data=cd,
            wavefunction_data=wf,
            r_up_carts=ru,
            r_dn_carts=rd,
            RT=RT,
            flag_determinant_only=True,
        )
        Jb = compute_Jastrow_part(jd, ru, rd)

        def _jr(m):
            return jnp.exp(compute_Jastrow_part(jd, mu[m], md[m]) - Jb)

        jr = jnp.array([_jr(m) for m in range(Vdet.shape[0])])
        nl = jnp.sum(jnp.maximum(Vdet, 0.0)) + jnp.sum(jnp.minimum(Vdet, 0.0) * jr)
    else:
        _, _, Vnl, _ = nn_ecp(
            coulomb_potential_data=cd,
            wavefunction_data=wf,
            r_up_carts=ru,
            r_dn_carts=rd,
            RT=RT,
        )
        nl = jnp.sum(jnp.maximum(Vnl, 0.0)) + jnp.sum(jnp.minimum(Vnl, 0.0))
    return elel + iiv + jnp.sum(ei_u + kc_u) + jnp.sum(ei_dn + kc_dn) + ecpl + nl


def _oracle_ecp_grads(ham, ru, rd, dltmove, alat=ALAT):
    P0 = jnp.asarray(ham.structure_data.positions, dtype=jnp.float64)
    RT = jnp.eye(3)

    def f(P, ru, rd):
        return jnp.real(_e_L_t_ecp(_inject(ham, P), ru, rd, alat, RT, dltmove))

    return jax.grad(f, argnums=(0, 1, 2))(P0, jnp.asarray(ru), jnp.asarray(rd))


def _oracle_ecp_stress(ham, ru, rd, dltmove=False, alat=ALAT):
    """de_L^t/deps via jax.jacrev of the affinely-strained molecular ECP e_L_t (NN mesh).

    The quadrature rotation stays the identity under the strain: the strained ECP
    quadrature points stay on the sphere of the strained electron-nucleus distance."""
    P0 = jnp.asarray(ham.structure_data.positions, dtype=jnp.float64)

    def eL(eps, ru, rd):
        F = jnp.eye(3) + eps
        hs = _inject(ham, P0 @ F.T)
        return jnp.real(_e_L_t_ecp(hs, ru @ F.T, rd @ F.T, alat, jnp.eye(3), dltmove))

    return np.asarray(jax.jacrev(eL, argnums=0)(jnp.zeros((3, 3)), jnp.asarray(ru), jnp.asarray(rd)))


def _oracle_ae_stress(ham, ru, rd, alat=ALAT):
    """de_L^t/deps via jax.jacrev of the affinely-strained molecular AE e_L_t
    (r->Fr, R->FR via inject, RT->F.T). Mirrors _compute_local_energy_t_strained."""
    P0 = jnp.asarray(ham.structure_data.positions, dtype=jnp.float64)

    def eL(eps, ru, rd):
        F = jnp.eye(3) + eps
        hs = _inject(ham, P0 @ F.T)
        return jnp.real(_e_L_t(hs, ru @ F.T, rd @ F.T, alat, F.T))

    return np.asarray(jax.jacrev(eL, argnums=0)(jnp.zeros((3, 3)), jnp.asarray(ru), jnp.asarray(rd)))


def _oracle_grads(ham, ru, rd, alat=ALAT):
    P0 = jnp.asarray(ham.structure_data.positions, dtype=jnp.float64)
    RT = jnp.eye(3)

    def f(P, ru, rd):
        return _e_L_t(_inject(ham, P), ru, rd, alat, RT)

    return jax.grad(f, argnums=(0, 1, 2))(P0, jnp.asarray(ru), jnp.asarray(rd))


def _oracle_lnpsi_grads(ham, ru, rd):
    from jqmc.wavefunction import evaluate_ln_wavefunction

    P0 = jnp.asarray(ham.structure_data.positions, dtype=jnp.float64)

    def f(P, ru, rd):
        return evaluate_ln_wavefunction(_inject(ham.wavefunction_data, P), ru, rd)

    return jax.grad(f, argnums=(0, 1, 2))(P0, jnp.asarray(ru), jnp.asarray(rd))


def _assert_close(got, ref, atol, rtol, msg):
    """Global-max comparison: ``max|got-ref| <= atol + rtol*max(1,|ref|.max)``.

    The tolerance scales by the *global* max magnitude of the
    reference, NOT a per-element rtol.
    Using ``np.testing.assert_allclose`` here would silently tighten the bound
    on the smaller components of a force/stress vector (the periodic pgto
    far-image cutoff, ~1e-6 absolute, lands on a component whose magnitude is
    below the vector max, so per-element rtol cannot absorb it). Replicating the
    global-max-scaled bound keeps each test's acceptance logic global-max scaled
    while sourcing (atol, rtol) from the precision-zone helper."""
    got = np.asarray(got)
    ref = np.asarray(ref)
    assert got.shape == ref.shape, (msg, got.shape, ref.shape)
    scale = max(1.0, float(np.abs(ref).max()))
    max_diff = float(np.abs(got - ref).max())
    assert max_diff <= atol + rtol * scale, (msg, max_diff, atol + rtol * scale)


# --------------------------------------------------------------------------- #
# Molecular AE: GFMC HF force + Pulay force + SWCT building block.
# --------------------------------------------------------------------------- #

AE_SYS = [
    "mol_cart_sd_jnone",  # closed/open-shell SD (stands in for the H2/N SD bench rows)
    "mol_cart_agp_jnone",  # AGP geminal
]


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 3, 18, 25])
@pytest.mark.parametrize("pattern", AE_SYS)
def test_gfmc_grad_e_l_ae_jnone(pattern, seed):
    """GFMC grad e l ae jnone."""
    ham = rs_ham(pattern)
    ru, rd = _cfg(ham, seed)
    ref_R, ref_u, ref_d = _oracle_grads(ham, ru, rd)
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    got_R, got_u, got_d = jqmc_rs._gfmc_grad_e_L_ae(gd, jd, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3))
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_R,
        ref_R,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "R"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_u,
        ref_u,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "r_up"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )
    _comparison_3_got, _comparison_3_ref, _comparison_3_atol, _comparison_3_rtol, _comparison_3_msg = (
        got_d,
        ref_d,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "r_dn"),
    )
    _comparison_3_got = np.asarray(_comparison_3_got)
    _comparison_3_ref = np.asarray(_comparison_3_ref)
    assert _comparison_3_got.shape == _comparison_3_ref.shape, (
        _comparison_3_msg,
        _comparison_3_got.shape,
        _comparison_3_ref.shape,
    )
    _comparison_3_scale = max(1.0, float(np.abs(_comparison_3_ref).max()))
    _comparison_3_max_diff = float(np.abs(_comparison_3_got - _comparison_3_ref).max())
    assert _comparison_3_max_diff <= _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale, (
        _comparison_3_msg,
        _comparison_3_max_diff,
        _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale,
    )


WITHJ_SYS = [
    ("mol_cart_sd_j12", False),
    ("mol_cart_sd_j123", True),  # AO-leaf J3 (stands in for the H2/N j123ao bench rows)
    # PRODUCTION-DEFAULT Jastrow form. ``gfmc_forces.rs`` is
    # the single largest concentration of ``is_exp`` J1 branches in the Rust
    # tree (19 sites), and every fixture reaching it pinned the NON-default
    # ``exp`` form until these rows.
    ("mol_cart_sd_j12_pade", False),
    ("mol_cart_sd_j12_p1e2", False),
    ("mol_cart_sd_j12_e1p2", False),
    ("mol_cart_sd_j123_pade", True),
]


@pytest.mark.light
@pytest.mark.parametrize("seed", [5, 14, 38])
@pytest.mark.parametrize("pattern,withj3", WITHJ_SYS)
def test_gfmc_grad_e_l_ae_withj(pattern, withj3, seed):
    """GFMC grad e l ae withj."""
    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    if withj3:
        wf = _rand_j3(wf)
        ham = ham.replace(wavefunction_data=wf)
    ru, rd = _cfg(ham, seed)
    ref_R, ref_u, ref_d = _oracle_grads(ham, ru, rd)
    got_R, got_u, got_d = jqmc_rs._gfmc_grad_e_L_ae(
        wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3)
    )
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_R,
        ref_R,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "R"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_u,
        ref_u,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "r_up"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )
    _comparison_3_got, _comparison_3_ref, _comparison_3_atol, _comparison_3_rtol, _comparison_3_msg = (
        got_d,
        ref_d,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "r_dn"),
    )
    _comparison_3_got = np.asarray(_comparison_3_got)
    _comparison_3_ref = np.asarray(_comparison_3_ref)
    assert _comparison_3_got.shape == _comparison_3_ref.shape, (
        _comparison_3_msg,
        _comparison_3_got.shape,
        _comparison_3_ref.shape,
    )
    _comparison_3_scale = max(1.0, float(np.abs(_comparison_3_ref).max()))
    _comparison_3_max_diff = float(np.abs(_comparison_3_got - _comparison_3_ref).max())
    assert _comparison_3_max_diff <= _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale, (
        _comparison_3_msg,
        _comparison_3_max_diff,
        _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale,
    )


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("pattern", ["mol_cart_sd_jnone", "mol_cart_sd_j12", "mol_cart_sd_j123"])
def test_gfmc_grad_ln_psi(pattern, seed):
    """GFMC grad ln psi."""
    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    if "j123" in pattern:
        wf = _rand_j3(wf)
        ham = ham.replace(wavefunction_data=wf)
    ru, rd = _cfg(ham, seed)
    ref_R, ref_u, ref_d = _oracle_lnpsi_grads(ham, ru, rd)
    got_R, got_u, got_d = jqmc_rs._gfmc_grad_ln_Psi(wf.geminal_data, wf.jastrow_data, ru, rd)
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_R,
        ref_R,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "R"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_u,
        ref_u,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "r_up"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )
    _comparison_3_got, _comparison_3_ref, _comparison_3_atol, _comparison_3_rtol, _comparison_3_msg = (
        got_d,
        ref_d,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "r_dn"),
    )
    _comparison_3_got = np.asarray(_comparison_3_got)
    _comparison_3_ref = np.asarray(_comparison_3_ref)
    assert _comparison_3_got.shape == _comparison_3_ref.shape, (
        _comparison_3_msg,
        _comparison_3_got.shape,
        _comparison_3_ref.shape,
    )
    _comparison_3_scale = max(1.0, float(np.abs(_comparison_3_ref).max()))
    _comparison_3_max_diff = float(np.abs(_comparison_3_got - _comparison_3_ref).max())
    assert _comparison_3_max_diff <= _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale, (
        _comparison_3_msg,
        _comparison_3_max_diff,
        _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale,
    )


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_swct_omega_domega(seed):
    """SWCT omega and Sum-grad-omega vs jqmc (swct.py). Force-assembly building block."""
    from jqmc.swct import evaluate_swct_domega, evaluate_swct_omega

    ham = rs_ham("mol_ecp_cart_jnone")
    ru, rd = _cfg(ham, seed)
    r = np.vstack([ru, rd])
    st = ham.structure_data
    ref_o = np.asarray(evaluate_swct_omega(st, r))
    ref_d = np.asarray(evaluate_swct_domega(st, r))
    got_o = jqmc_rs._swct_omega(st, r)
    got_d = jqmc_rs._swct_domega(st, r)
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_o,
        ref_o,
        _ATOL_MOL,
        _RTOL_MOL,
        (seed, "omega"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_d,
        ref_d,
        _ATOL_MOL,
        _RTOL_MOL,
        (seed, "domega"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )


# GFMC HF stress de_L^t/deps -- molecular AE jnone/j12/j123.
@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("pattern", ["mol_cart_sd_jnone", "mol_cart_sd_j12", "mol_cart_sd_j123"])
def test_gfmc_stress_ae(pattern, seed):
    """GFMC stress ae."""
    ham = rs_ham(pattern)
    if "j123" in pattern:
        ham = ham.replace(wavefunction_data=_rand_j3(ham.wavefunction_data))
    ru, rd = _cfg(ham, seed)
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    ref = _oracle_ae_stress(ham, ru, rd)
    got = np.asarray(jqmc_rs._gfmc_stress_ae(gd, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3)))
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got,
        ref,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )


# --------------------------------------------------------------------------- #
# Molecular ECP: GFMC HF stress + force (tmove/dltmove).
# --------------------------------------------------------------------------- #


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dltmove", [False, True])
@pytest.mark.parametrize("pattern", ["mol_ecp_cart_jnone"])
def test_gfmc_stress_ecp_jnone(pattern, seed, dltmove):
    """GFMC stress ECP jnone."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = rs_ham(pattern)
    gd = ham.wavefunction_data.geminal_data
    ru, rd = _cfg(ham, seed)
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    ref = _oracle_ecp_stress(ham, ru, rd, dltmove=dltmove)
    got = np.asarray(jqmc_rs._gfmc_stress_ecp_jnone(gd, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1, dltmove))
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got,
        ref,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, dltmove),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )


# --------------------------------------------------------------------------- #
# Periodic (Ewald, Gamma) building blocks: electron-axis kinetic + Pulay lndet.
# --------------------------------------------------------------------------- #


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("pattern", ["pbc_gamma_sphepgto_jnone", "bspline_gamma_jnone"])
def test_grad_T_pbc_det_wrt_r(pattern, seed):
    """Grad T PBC det wrt r."""
    # B-spline: the determinant NUCLEAR force is exactly zero (control points sit on a fixed
    # cell-fractional grid, dAO/dR=0), but the electron-axis kinetic force dT_det/dr is nonzero
    # (the electron drifts through the grid) -- served by the new `bspline_det_egrad_side`.
    from jqmc.wavefunction import compute_kinetic_energy_all_elements as CKall

    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    nu, nd = gd.num_electron_up, gd.num_electron_dn
    ru = R[rng.integers(0, len(R), nu)] + rng.normal(scale=0.4, size=(nu, 3))
    rd = R[rng.integers(0, len(R), nd)] + rng.normal(scale=0.4, size=(nd, 3))

    def T(ru, rd):
        u, d = CKall(wf, ru, rd)
        return jnp.sum(u) + jnp.sum(d)

    ref_u = np.asarray(jax.grad(T, argnums=0)(jnp.asarray(ru), jnp.asarray(rd)))
    ref_d = np.asarray(jax.grad(T, argnums=1)(jnp.asarray(ru), jnp.asarray(rd)))
    ref = np.vstack([ref_u, ref_d])
    got = np.asarray(jqmc_rs._grad_T_pbc_det_wrt_r(gd, ru, rd))
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got,
        ref,
        _ATOL_PBC,
        _RTOL_PBC,
        (pattern, seed),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("pattern", ["pbc_gamma_sphepgto_j12", "pbc_gamma_sphepgto_j123"])
def test_grad_T_pbc_withj_wrt_r(pattern, seed):
    """Periodic withJ (det+J1+J2[+J3]) electron-axis dT/dr vs jax.grad(CK) on lithium j12/j123."""
    from jqmc.wavefunction import compute_kinetic_energy_all_elements as CKall

    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    if "j123" in pattern:
        wf = _rand_j3(wf)
        ham = ham.replace(wavefunction_data=wf)
    gd = wf.geminal_data
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[:1] + rng.normal(scale=0.4, size=(1, 3))
    rd = R[1:2] + rng.normal(scale=0.4, size=(1, 3))

    def T(ru, rd):
        u, d = CKall(wf, ru, rd)
        return jnp.sum(u) + jnp.sum(d)

    ref_u = np.asarray(jax.grad(T, 0)(jnp.asarray(ru), jnp.asarray(rd)))
    ref_d = np.asarray(jax.grad(T, 1)(jnp.asarray(ru), jnp.asarray(rd)))
    ref = np.vstack([ref_u, ref_d])
    got = np.asarray(jqmc_rs._grad_T_pbc_withj_wrt_r(gd, wf.jastrow_data, ru, rd))
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got,
        ref,
        _ATOL_PBC,
        _RTOL_PBC,
        (pattern, seed),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grad_lndet_pbc_wrt_R(seed):
    """Periodic Pulay dln|detG|/dR vs jax.grad(compute_ln_det). Building block for
    the PBC GFMC Pulay force / mesh terms."""
    from jqmc.determinant import compute_ln_det_geminal_all_elements as lndet
    from jqmc.jqmc_mcmc import _inject_active_positions

    ham = rs_ham("pbc_gamma_sphepgto_jnone")
    gd = ham.wavefunction_data.geminal_data
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[:1] + rng.normal(scale=0.4, size=(1, 3))
    rd = R[1:2] + rng.normal(scale=0.4, size=(1, 3))
    P0 = jnp.asarray(R)
    ref = np.asarray(jax.grad(lambda P: lndet(_inject_active_positions(gd, {"positions": P}), ru, rd))(P0))
    got = np.asarray(jqmc_rs._grad_lnDet_pbc_wrt_R(gd, ru, rd))
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got,
        ref,
        _ATOL_PBC,
        _RTOL_PBC,
        seed,
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )


# --------------------------------------------------------------------------- #
# PBC (Ewald, Gamma) ECP GFMC HF force -- jnone / j12 / j123ao / jdtn.
# --------------------------------------------------------------------------- #


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dltmove", [False, True])
@pytest.mark.parametrize(
    "pattern",
    [
        "pbc_gamma_sphepgto_jnone",
        "pbc_gamma_sphepgto_j12",
        "pbc_gamma_sphepgto_j123",
        "pbc_gamma_sphepgto_jdtn",
    ],
)
def test_gfmc_grad_e_l_ecp_pbc(pattern, seed, dltmove):
    """GFMC grad e l ECP PBC."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = rs_ham(pattern)
    if "j123" in pattern:
        ham = ham.replace(wavefunction_data=_rand_j3(ham.wavefunction_data))
    # jdtn: the committed pattern already carries randomized (non-zero) DTN
    # alpha/beta/gamma_indep params (Jastrow_DTN_data.build(random_init=True)),
    # so no extra randomization step is needed here.
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    ru, rd = _pbc_cfg(ham, seed)
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    ref_R, ref_u, ref_d = _oracle_ecp_grads(ham, ru, rd, dltmove=dltmove)
    got_R, got_u, got_d = jqmc_rs._gfmc_grad_e_L_ecp_pbc(
        gd, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1, dltmove
    )
    if pattern.endswith(("jnone", "jdtn")):
        full_R, full_u, full_d = jqmc_rs._gfmc_grad_e_L_ecp_pbc(
            gd,
            wf.jastrow_data,
            ham.coulomb_potential_data,
            ru,
            rd,
            ALAT,
            np.eye(3),
            W,
            GP,
            1,
            dltmove,
            debug=True,
        )
        for label, fast, full in (
            ("R", got_R, full_R),
            ("r_up", got_u, full_u),
            ("r_dn", got_d, full_d),
        ):
            np.testing.assert_allclose(
                np.asarray(fast),
                np.asarray(full),
                atol=ATOL_RS_GFMC_SPRIME_FAST_VS_FULL,
                rtol=RTOL_RS_GFMC_SPRIME_FAST_VS_FULL,
                err_msg=f"Rust Gamma S' fast/full force mismatch ({pattern}, {seed}, {dltmove}, {label})",
            )
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_R,
        ref_R,
        _ATOL_PBC,
        _RTOL_PBC,
        (pattern, seed, dltmove, "R"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_u,
        ref_u,
        _ATOL_PBC,
        _RTOL_PBC,
        (pattern, seed, dltmove, "r_up"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )
    _comparison_3_got, _comparison_3_ref, _comparison_3_atol, _comparison_3_rtol, _comparison_3_msg = (
        got_d,
        ref_d,
        _ATOL_PBC,
        _RTOL_PBC,
        (pattern, seed, dltmove, "r_dn"),
    )
    _comparison_3_got = np.asarray(_comparison_3_got)
    _comparison_3_ref = np.asarray(_comparison_3_ref)
    assert _comparison_3_got.shape == _comparison_3_ref.shape, (
        _comparison_3_msg,
        _comparison_3_got.shape,
        _comparison_3_ref.shape,
    )
    _comparison_3_scale = max(1.0, float(np.abs(_comparison_3_ref).max()))
    _comparison_3_max_diff = float(np.abs(_comparison_3_got - _comparison_3_ref).max())
    assert _comparison_3_max_diff <= _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale, (
        _comparison_3_msg,
        _comparison_3_max_diff,
        _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale,
    )


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dltmove", [False, True])
@pytest.mark.parametrize("withj", [False, True])
def test_gfmc_grad_e_l_ecp_pbc_bspline(withj, seed, dltmove):
    """PBC (Ewald, Gamma) ECP GFMC HF force de_L/d(R,r_up,r_dn) for a B-spline geminal
    (QE-projection Li_gamma, ECP), jnone and J1/J2. The determinant Pulay (dAO/dR) is exactly
    zero on the fixed fractional grid, so the surviving orbital piece is the electron-axis
    det kinetic (bspline_det_egrad_side); the classical V (Ewald/ECP) and J1 Pulay/kinetic
    sides are basis-independent.

    J3 on a B-spline geminal is NOT covered: the committed builder cannot attach a pgto J3 to a
    B-spline leaf, and a B-spline-child J3 basis raises in XLA too (ao_deriv.py) -- strict-raise
    parity, a known documented gap, not a Rust regression.
    """
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
    from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
    from tests.test_rs_kernel._rs_fixtures import rs_ham as _rs

    if withj:
        ham = _build_ham_for_cell("QEPROJ:bspline:Li_gamma:ECP", True, None, "SD")  # J1/J2
    else:
        ham = _rs("bspline_gamma_jnone")
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    ru, rd = _pbc_cfg(ham, seed)
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    ref_R, ref_u, ref_d = _oracle_ecp_grads(ham, ru, rd, dltmove=dltmove)
    got_R, got_u, got_d = jqmc_rs._gfmc_grad_e_L_ecp_pbc(
        gd, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1, dltmove
    )
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_R,
        ref_R,
        _ATOL_PBC,
        _RTOL_PBC,
        (withj, seed, dltmove, "R"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_u,
        ref_u,
        _ATOL_PBC,
        _RTOL_PBC,
        (withj, seed, dltmove, "r_up"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )
    _comparison_3_got, _comparison_3_ref, _comparison_3_atol, _comparison_3_rtol, _comparison_3_msg = (
        got_d,
        ref_d,
        _ATOL_PBC,
        _RTOL_PBC,
        (withj, seed, dltmove, "r_dn"),
    )
    _comparison_3_got = np.asarray(_comparison_3_got)
    _comparison_3_ref = np.asarray(_comparison_3_ref)
    assert _comparison_3_got.shape == _comparison_3_ref.shape, (
        _comparison_3_msg,
        _comparison_3_got.shape,
        _comparison_3_ref.shape,
    )
    _comparison_3_scale = max(1.0, float(np.abs(_comparison_3_ref).max()))
    _comparison_3_max_diff = float(np.abs(_comparison_3_got - _comparison_3_ref).max())
    assert _comparison_3_max_diff <= _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale, (
        _comparison_3_msg,
        _comparison_3_max_diff,
        _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale,
    )


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("pattern", ["pbc_gamma_sphepgto_jnone"])
def test_gfmc_stress_ecp_pbc_jnone(pattern, seed):
    """GFMC stress ECP PBC jnone."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    ru, rd = _pbc_cfg(ham, seed)
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    ref = _oracle_ecp_pbc_stress(ham, ru, rd, dltmove=False)
    got = np.asarray(jqmc_rs._gfmc_stress_ecp_pbc_jnone(gd, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1))
    full = np.asarray(
        jqmc_rs._gfmc_stress_ecp_pbc_jnone(gd, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1, debug=True)
    )
    np.testing.assert_allclose(
        got,
        full,
        atol=ATOL_RS_GFMC_SPRIME_FAST_VS_FULL,
        rtol=RTOL_RS_GFMC_SPRIME_FAST_VS_FULL,
    )
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got,
        ref,
        _ATOL_PBC,
        _RTOL_PBC,
        (pattern, seed),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )


def _oracle_ecp_pbc_stress(ham, ru, rd, dltmove=False, alat=ALAT):
    """de_L^t/deps via jax.jacrev of the affinely-strained PBC (Ewald, Gamma) ECP e_L_t.
    `_apply_strain(F)` strains the whole Hamiltonian (cell + positions); r->Fr strains the
    electrons while the ECP quadrature rotation stays the identity (the strained
    quadrature points stay on the sphere of the strained electron-nucleus distance).
    The J2 cell strains with the physical lattice, matching `Jastrow_data._apply_strain`."""

    def eL(eps):
        F = jnp.eye(3) + eps
        hs = ham._apply_strain(F)
        return jnp.real(_e_L_t_ecp(hs, ru @ F.T, rd @ F.T, alat, jnp.eye(3), dltmove))

    return np.asarray(jax.jacrev(eL)(jnp.zeros((3, 3))))


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dltmove", [False, True])
@pytest.mark.parametrize("pattern", ["pbc_gamma_sphepgto_j12"])
def test_gfmc_stress_ecp_pbc_withj(pattern, seed, dltmove):
    """GFMC stress ECP PBC withj."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    ru, rd = _pbc_cfg(ham, seed)
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    ref = _oracle_ecp_pbc_stress(ham, ru, rd, dltmove=dltmove)
    got = np.asarray(
        jqmc_rs._gfmc_stress_ecp_pbc_withj(
            gd, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1, dltmove
        )
    )
    full = np.asarray(
        jqmc_rs._gfmc_stress_ecp_pbc_withj(
            gd,
            wf.jastrow_data,
            ham.coulomb_potential_data,
            ru,
            rd,
            ALAT,
            np.eye(3),
            W,
            GP,
            1,
            dltmove,
            debug=True,
        )
    )
    _fast_atol, _fast_rtol = _withj_stress_fast_vs_full_tol()
    np.testing.assert_allclose(got, full, atol=_fast_atol, rtol=_fast_rtol)
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got,
        ref,
        _ATOL_PBC,
        _RTOL_PBC,
        (pattern, seed, dltmove),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dltmove", [False, True])
def test_gfmc_stress_ecp_pbc_j3(seed, dltmove):
    """PBC (Ewald, Gamma) ECP GFMC HF stress -- J3 (randomized), lithium_pbc j123ao."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = rs_ham("pbc_gamma_sphepgto_j123")
    ham = ham.replace(wavefunction_data=_rand_j3(ham.wavefunction_data))
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    ru, rd = _pbc_cfg(ham, seed)
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    ref = _oracle_ecp_pbc_stress(ham, ru, rd, dltmove=dltmove)
    got = np.asarray(
        jqmc_rs._gfmc_stress_ecp_pbc_withj(
            gd, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1, dltmove
        )
    )
    full = np.asarray(
        jqmc_rs._gfmc_stress_ecp_pbc_withj(
            gd,
            wf.jastrow_data,
            ham.coulomb_potential_data,
            ru,
            rd,
            ALAT,
            np.eye(3),
            W,
            GP,
            1,
            dltmove,
            debug=True,
        )
    )
    _fast_atol, _fast_rtol = _withj_stress_fast_vs_full_tol()
    np.testing.assert_allclose(got, full, atol=_fast_atol, rtol=_fast_rtol)
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got,
        ref,
        _ATOL_PBC,
        _RTOL_PBC,
        (seed, dltmove),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dltmove", [False, True])
def test_gfmc_stress_ecp_pbc_dtn(seed, dltmove):
    """PBC (Ewald, Gamma) ECP GFMC HF stress -- DTN Jastrow (u+chi+f), lithium_pbc jdtn.
    The committed pattern already carries randomized DTN params (no extra
    randomization step needed)."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = rs_ham("pbc_gamma_sphepgto_jdtn")
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    ru, rd = _pbc_cfg(ham, seed)
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    # The finite-difference oracle above holds the absolute DTN
    # cutoffs fixed.  Periodic DTN has a comoving cutoff contract, so use
    # the independently implemented XLA production stress for cross-backend
    # parity; its policy field is itself pinned against finite differences in
    # TestDTNCutoffPolicy.
    ref = np.asarray(
        xla_gfmc.grad_e_L_t_wrt_strain(
            ham,
            ru,
            rd,
            np.eye(3),
            ALAT,
            non_local_move="dltmove" if dltmove else "tmove",
        )
    )
    got = np.asarray(
        jqmc_rs._gfmc_stress_ecp_pbc_withj(
            gd, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1, dltmove
        )
    )
    full = np.asarray(
        jqmc_rs._gfmc_stress_ecp_pbc_withj(
            gd,
            wf.jastrow_data,
            ham.coulomb_potential_data,
            ru,
            rd,
            ALAT,
            np.eye(3),
            W,
            GP,
            1,
            dltmove,
            debug=True,
        )
    )
    _fast_atol, _fast_rtol = _withj_stress_fast_vs_full_tol()
    np.testing.assert_allclose(got, full, atol=_fast_atol, rtol=_fast_rtol)
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got,
        ref,
        _ATOL_PBC,
        _RTOL_PBC,
        (seed, dltmove),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )


# --------------------------------------------------------------------------- #
# DTN Jastrow (u/chi/f) molecular GFMC HF force: AE + ECP(tmove).
# --------------------------------------------------------------------------- #


@pytest.mark.light
@pytest.mark.parametrize("seed", [10, 25, 46])
@pytest.mark.parametrize("pattern", ["mol_cart_sd_jdtn"])
def test_gfmc_grad_e_l_ae_dtn(pattern, seed):
    """GFMC grad e l ae DTN."""
    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    ru, rd = _cfg(ham, seed)
    ref_R, ref_u, ref_d = _oracle_grads(ham, ru, rd)
    got_R, got_u, got_d = jqmc_rs._gfmc_grad_e_L_ae(
        wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3)
    )
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_R,
        ref_R,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "R"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_u,
        ref_u,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "r_up"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )
    _comparison_3_got, _comparison_3_ref, _comparison_3_atol, _comparison_3_rtol, _comparison_3_msg = (
        got_d,
        ref_d,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "r_dn"),
    )
    _comparison_3_got = np.asarray(_comparison_3_got)
    _comparison_3_ref = np.asarray(_comparison_3_ref)
    assert _comparison_3_got.shape == _comparison_3_ref.shape, (
        _comparison_3_msg,
        _comparison_3_got.shape,
        _comparison_3_ref.shape,
    )
    _comparison_3_scale = max(1.0, float(np.abs(_comparison_3_ref).max()))
    _comparison_3_max_diff = float(np.abs(_comparison_3_got - _comparison_3_ref).max())
    assert _comparison_3_max_diff <= _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale, (
        _comparison_3_msg,
        _comparison_3_max_diff,
        _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale,
    )


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gfmc_grad_e_l_ecp_dtn_tmove(seed):
    """GFMC grad e l ECP DTN tmove."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = rs_ham("mol_ecp_cart_jdtn")
    wf = ham.wavefunction_data
    ru, rd = _cfg(ham, seed)
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    ref_R, ref_u, ref_d = _oracle_ecp_grads(ham, ru, rd, dltmove=False)
    got_R, got_u, got_d = jqmc_rs._gfmc_grad_e_L_ecp(
        wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1, False
    )
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_R,
        ref_R,
        _ATOL_MOL,
        _RTOL_MOL,
        (seed, "R"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_u,
        ref_u,
        _ATOL_MOL,
        _RTOL_MOL,
        (seed, "r_up"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )
    _comparison_3_got, _comparison_3_ref, _comparison_3_atol, _comparison_3_rtol, _comparison_3_msg = (
        got_d,
        ref_d,
        _ATOL_MOL,
        _RTOL_MOL,
        (seed, "r_dn"),
    )
    _comparison_3_got = np.asarray(_comparison_3_got)
    _comparison_3_ref = np.asarray(_comparison_3_ref)
    assert _comparison_3_got.shape == _comparison_3_ref.shape, (
        _comparison_3_msg,
        _comparison_3_got.shape,
        _comparison_3_ref.shape,
    )
    _comparison_3_scale = max(1.0, float(np.abs(_comparison_3_ref).max()))
    _comparison_3_max_diff = float(np.abs(_comparison_3_got - _comparison_3_ref).max())
    assert _comparison_3_max_diff <= _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale, (
        _comparison_3_msg,
        _comparison_3_max_diff,
        _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale,
    )


@pytest.mark.light
def test_gfmc_grad_ln_psi_dtn():
    """GFMC grad ln psi DTN."""
    ham = rs_ham("mol_cart_sd_jdtn")
    wf = ham.wavefunction_data
    ru, rd = _cfg(ham, 1)
    ref_R, ref_u, ref_d = _oracle_lnpsi_grads(ham, ru, rd)
    got_R, got_u, got_d = jqmc_rs._gfmc_grad_ln_Psi(wf.geminal_data, wf.jastrow_data, ru, rd)
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_R,
        ref_R,
        _ATOL_MOL,
        _RTOL_MOL,
        ("dtn", "R"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_u,
        ref_u,
        _ATOL_MOL,
        _RTOL_MOL,
        ("dtn", "r_up"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )
    _comparison_3_got, _comparison_3_ref, _comparison_3_atol, _comparison_3_rtol, _comparison_3_msg = (
        got_d,
        ref_d,
        _ATOL_MOL,
        _RTOL_MOL,
        ("dtn", "r_dn"),
    )
    _comparison_3_got = np.asarray(_comparison_3_got)
    _comparison_3_ref = np.asarray(_comparison_3_ref)
    assert _comparison_3_got.shape == _comparison_3_ref.shape, (
        _comparison_3_msg,
        _comparison_3_got.shape,
        _comparison_3_ref.shape,
    )
    _comparison_3_scale = max(1.0, float(np.abs(_comparison_3_ref).max()))
    _comparison_3_max_diff = float(np.abs(_comparison_3_got - _comparison_3_ref).max())
    assert _comparison_3_max_diff <= _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale, (
        _comparison_3_msg,
        _comparison_3_max_diff,
        _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale,
    )


# --------------------------------------------------------------------------- #
# ECP (Casula T-move / dltmove) molecular GFMC HF force.
# --------------------------------------------------------------------------- #

# Coverage gap: no Cl2 ECP+j123ao row -- no
# committed ECP+J3 fixture pattern exists (ECP+J1J2 is covered by
# mol_ecp_cart_j12, AE+J3 is covered separately by mol_cart_sd_j123).
ECP_SYS = ["mol_ecp_cart_jnone", "mol_ecp_cart_j12", "mol_ecp_cart_j12_pade"]  # _pade: production-default Jastrow form


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("pattern", ECP_SYS)
def test_gfmc_grad_e_l_ecp_tmove(pattern, seed):
    """GFMC grad e l ECP tmove."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    if "j123" in pattern:
        wf = _rand_j3(wf)
        ham = ham.replace(wavefunction_data=wf)
    ru, rd = _cfg(ham, seed)
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    ref_R, ref_u, ref_d = _oracle_ecp_grads(ham, ru, rd, dltmove=False)
    got_R, got_u, got_d = jqmc_rs._gfmc_grad_e_L_ecp(
        wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1, False
    )
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_R,
        ref_R,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "R"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_u,
        ref_u,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "r_up"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )
    _comparison_3_got, _comparison_3_ref, _comparison_3_atol, _comparison_3_rtol, _comparison_3_msg = (
        got_d,
        ref_d,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "r_dn"),
    )
    _comparison_3_got = np.asarray(_comparison_3_got)
    _comparison_3_ref = np.asarray(_comparison_3_ref)
    assert _comparison_3_got.shape == _comparison_3_ref.shape, (
        _comparison_3_msg,
        _comparison_3_got.shape,
        _comparison_3_ref.shape,
    )
    _comparison_3_scale = max(1.0, float(np.abs(_comparison_3_ref).max()))
    _comparison_3_max_diff = float(np.abs(_comparison_3_got - _comparison_3_ref).max())
    assert _comparison_3_max_diff <= _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale, (
        _comparison_3_msg,
        _comparison_3_max_diff,
        _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale,
    )


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("pattern", ["mol_ecp_cart_j12"])
def test_gfmc_grad_e_l_ecp_dltmove(pattern, seed):
    """(Coverage gap: no ECP+J3 row, for the same reason as ECP_SYS above:
    no committed ECP+J3 fixture pattern exists.)"""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    if "j123" in pattern:
        wf = _rand_j3(wf)
        ham = ham.replace(wavefunction_data=wf)
    ru, rd = _cfg(ham, seed)
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    ref_R, ref_u, ref_d = _oracle_ecp_grads(ham, ru, rd, dltmove=True)
    got_R, got_u, got_d = jqmc_rs._gfmc_grad_e_L_ecp(
        wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1, True
    )
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_R,
        ref_R,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "R"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_u,
        ref_u,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "r_up"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )
    _comparison_3_got, _comparison_3_ref, _comparison_3_atol, _comparison_3_rtol, _comparison_3_msg = (
        got_d,
        ref_d,
        _ATOL_MOL,
        _RTOL_MOL,
        (pattern, seed, "r_dn"),
    )
    _comparison_3_got = np.asarray(_comparison_3_got)
    _comparison_3_ref = np.asarray(_comparison_3_ref)
    assert _comparison_3_got.shape == _comparison_3_ref.shape, (
        _comparison_3_msg,
        _comparison_3_got.shape,
        _comparison_3_ref.shape,
    )
    _comparison_3_scale = max(1.0, float(np.abs(_comparison_3_ref).max()))
    _comparison_3_max_diff = float(np.abs(_comparison_3_got - _comparison_3_ref).max())
    assert _comparison_3_max_diff <= _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale, (
        _comparison_3_msg,
        _comparison_3_max_diff,
        _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale,
    )


# --------------------------------------------------------------------------- #
# Composite (pgto (+) pw) determinant GFMC stress/force.
# --------------------------------------------------------------------------- #


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1])
def test_gfmc_stress_ecp_pbc_composite_jnone(seed):
    """GFMC stress ECP PBC composite jnone."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = rs_ham("pbc_composite_jnone")
    gd = ham.wavefunction_data.geminal_data
    ru, rd = _pbc_cfg(ham, seed)
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    ref = _oracle_ecp_pbc_stress(ham, ru, rd, dltmove=False)
    got = np.asarray(jqmc_rs._gfmc_stress_ecp_pbc_jnone(gd, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1))
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got,
        ref,
        _ATOL_PBC,
        _RTOL_PBC,
        seed,
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("dltmove", [False, True])
@pytest.mark.parametrize("pattern", ["pbc_composite_j12"])
def test_gfmc_stress_ecp_pbc_composite_withj(pattern, seed, dltmove):
    """(Coverage gap: no composite+j123ao row; no
    committed composite+J3 fixture pattern exists.)"""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    ru, rd = _pbc_cfg(ham, seed)
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    ref = _oracle_ecp_pbc_stress(ham, ru, rd, dltmove=dltmove)
    got = np.asarray(
        jqmc_rs._gfmc_stress_ecp_pbc_withj(
            gd, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1, dltmove
        )
    )
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got,
        ref,
        _ATOL_PBC,
        _RTOL_PBC,
        (pattern, seed, dltmove),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )


@pytest.mark.heavy
@pytest.mark.parametrize("kind", ["gamma_pw", "gamma_bspline", "gamma_pgto_bspline"])
def test_gfmc_stress_ecp_pbc_qe_non_pgto_leaf_parity(kind):
    """Blocking XLA=Rust contract for the GFMC stress leaves used by the QE
    integration force run.

    This deliberately exercises both native real kernels: jnone catches the
    Laplacian metric response, while withJ also catches the determinant/Jastrow
    drift cross.  The pGTO+B-spline row verifies that the two leaf contributions
    are accumulated rather than selecting one child or falling back to JAX.
    """
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
    from tests.test_rs_kernel.test_rust_dispatch_equivalence import _build_ham_qe, _qe_gamma_ok

    assert _qe_gamma_ok, "committed Li_gamma QE fixture is required by this blocking parity contract"

    ham_withj = _build_ham_qe(kind)
    wf_withj = ham_withj.wavefunction_data
    ru, rd = _pbc_cfg(ham_withj, seed=0)
    weights = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    grid_points = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)

    jd_none = wf_withj.jastrow_data.replace(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=None,
        jastrow_dtn_data=None,
    )
    ham_jnone = ham_withj.replace(wavefunction_data=wf_withj.replace(jastrow_data=jd_none))
    ref_jnone = _oracle_ecp_pbc_stress(ham_jnone, ru, rd, dltmove=False)
    got_jnone = np.asarray(
        jqmc_rs._gfmc_stress_ecp_pbc_jnone(
            wf_withj.geminal_data,
            ham_jnone.coulomb_potential_data,
            ru,
            rd,
            ALAT,
            np.eye(3),
            weights,
            grid_points,
            1,
        )
    )
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_jnone,
        ref_jnone,
        _ATOL_PBC,
        _RTOL_PBC,
        (kind, "jnone"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )

    ref_withj = _oracle_ecp_pbc_stress(ham_withj, ru, rd, dltmove=False)
    got_withj = np.asarray(
        jqmc_rs._gfmc_stress_ecp_pbc_withj(
            wf_withj.geminal_data,
            wf_withj.jastrow_data,
            ham_withj.coulomb_potential_data,
            ru,
            rd,
            ALAT,
            np.eye(3),
            weights,
            grid_points,
            1,
            False,
        )
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_withj,
        ref_withj,
        _ATOL_PBC,
        _RTOL_PBC,
        (kind, "withj"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )


@pytest.mark.heavy
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("dltmove", [False, True])
@pytest.mark.parametrize("pattern", ["pbc_composite_jnone"])
def test_gfmc_grad_e_l_ecp_pbc_composite(pattern, seed, dltmove):
    """GFMC HF force on a composite (pgto (+) pw) determinant: the PW is R-independent
    (atom force = 0 Pulay) but its grad^2/grad depend on r -> PW electron kinetic force
    via pw_det_egrad_side.

    (Coverage gap: no composite+j123ao row; no
    committed composite+J3 fixture pattern exists.)"""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    ru, rd = _pbc_cfg(ham, seed)
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    ref_R, ref_u, ref_d = _oracle_ecp_grads(ham, ru, rd, dltmove=dltmove)
    got_R, got_u, got_d = jqmc_rs._gfmc_grad_e_L_ecp_pbc(
        gd, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1, dltmove
    )
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_R,
        ref_R,
        _ATOL_PBC,
        _RTOL_PBC,
        (pattern, seed, dltmove, "R"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_u,
        ref_u,
        _ATOL_PBC,
        _RTOL_PBC,
        (pattern, seed, dltmove, "r_up"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )
    _comparison_3_got, _comparison_3_ref, _comparison_3_atol, _comparison_3_rtol, _comparison_3_msg = (
        got_d,
        ref_d,
        _ATOL_PBC,
        _RTOL_PBC,
        (pattern, seed, dltmove, "r_dn"),
    )
    _comparison_3_got = np.asarray(_comparison_3_got)
    _comparison_3_ref = np.asarray(_comparison_3_ref)
    assert _comparison_3_got.shape == _comparison_3_ref.shape, (
        _comparison_3_msg,
        _comparison_3_got.shape,
        _comparison_3_ref.shape,
    )
    _comparison_3_scale = max(1.0, float(np.abs(_comparison_3_ref).max()))
    _comparison_3_max_diff = float(np.abs(_comparison_3_got - _comparison_3_ref).max())
    assert _comparison_3_max_diff <= _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale, (
        _comparison_3_msg,
        _comparison_3_max_diff,
        _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale,
    )


# =========================================================================== #
# the FORCE/STRESS halves of the
# single-spin and AGPn crosses.
#
# Every force/stress row above rides a fixture with BOTH spin channels populated
# and an SD (or, for one AE force row, AGP) geminal. The empty-spin-channel and
# AGPn-geminal families had no position-derivative parity row at all. Each row
# below carries three completion conditions: a liveness assertion
# (imported from ``test_gfmc.py``, single source of truth for what "single spin"
# / "AGPn geminal" means), a real XLA-vs-Rust comparison, and a production
# route-parity leg. The projection/value halves of the same two crosses are
# ``test_gfmc.py::test_gfmc_projection_single_spin`` /
# ``...::test_gfmc_projection_agp_agpn``.
# =========================================================================== #
import os  # noqa: E402

from jqmc._setting import GFMC_MIN_COLLECT_STEPS, GFMC_MIN_STEPS  # noqa: E402
from jqmc.determinant import AGPn_data  # noqa: E402
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t  # noqa: E402
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell  # noqa: E402
from tests.test_rs_kernel._route_spy import assert_arg_reached, assert_native_ran, install_ctx_spy  # noqa: E402
from tests.test_rs_kernel.test_gfmc import _assert_geminal_class, _assert_single_spin  # noqa: E402

_CROSS_STEPS = 2 * GFMC_MIN_STEPS


def _assert_empty_channel(got, ref, tag):
    """The down-spin derivative block of a single-spin cell is EMPTY on both
    sides. ``_assert_close`` reduces over the reference, which has no identity on
    a zero-size array, so the meaningful assertion for this channel is the shape
    agreement itself -- a Rust kernel that invented a phantom down electron (or
    dropped the axis entirely) fails here."""
    got = np.asarray(got)
    ref = np.asarray(ref)
    assert got.shape == ref.shape == (0, 3), (tag, got.shape, ref.shape)


def _run_gfmc_force(driver, ham, seed):
    """One PRODUCTION native GFMC run WITH position derivatives (the route that
    folds the force/stress kernels into the projection chunk)."""
    os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    try:
        common = dict(
            hamiltonian_data=ham,
            num_walkers=2,
            num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
            mcmc_seed=seed,
            alat=ALAT,
            non_local_move="tmove",
            random_discretized_mesh=True,
            comput_position_deriv=True,
            mpc=False,
        )
        g = GFMC_n(**common, num_mcmc_per_measurement=3, E_scf=-1.0) if driver == "n" else GFMC_t(**common, tau=0.05)
        g.run(num_mcmc_steps=_CROSS_STEPS)
        return g
    finally:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)


def _assert_force_route_native(spy, driver, context):
    """ROUTE PARITY: the driver reached the native projection chunk AND asked it
    for the force fold-in explicitly (``want_force`` supplied, not defaulted).
    ``want_stress`` is False here because these cells are molecular and no stress
    was requested -- asserted rather than skipped, so a future change of that rule
    shows up."""
    method = f"run_gfmc_projection_{driver}"
    assert_native_ran(spy, method, at_least=_CROSS_STEPS, context=context)
    assert bool(spy.build_arg("want_gfmc")) is True, f"{context}: build_obs_ctx was not asked for the GFMC args"
    assert bool(spy.build_arg("complex")) is False, f"{context}: build_obs_ctx asked for the complex family"
    assert_arg_reached(spy, method, "want_force", True, context=context)
    assert_arg_reached(spy, method, "want_stress", False, context=context)


# --------------------------------------------------------------------------- #
# Single-spin (n_dn == 0) x GFMC force / stress.
#
# ``H_ae_ccpvdz_cart.h5`` = the all-electron cell (``_gfmc_grad_e_L_ae`` /
# ``_gfmc_stress_ae``); ``Li_ecp_ccpvdz_cart.h5`` = the ECP cell
# (``_gfmc_grad_e_L_ecp`` / ``_gfmc_stress_ecp_jnone``). Li rather than H_ecp
# because H's ccECP projector coefficient is zero, so an H_ecp row would
# differentiate an identically-zero non-local channel.
# --------------------------------------------------------------------------- #
@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("withj", [False, True])
def test_gfmc_force_single_spin_ae(withj, seed):
    """Single-spin (AE force): de_L^t/d(R, r_up, r_dn) with an EMPTY down channel."""
    ham = _build_ham_for_cell("H_ae_ccpvdz_cart.h5", withj, None, "SD")
    _assert_single_spin(ham)
    wf = ham.wavefunction_data
    ru, rd = _cfg(ham, seed)
    assert rd.shape == (0, 3), rd.shape
    ref_R, ref_u, ref_d = _oracle_grads(ham, ru, rd)
    got_R, got_u, got_d = jqmc_rs._gfmc_grad_e_L_ae(
        wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3)
    )
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_R,
        ref_R,
        _ATOL_MOL,
        _RTOL_MOL,
        (withj, seed, "R"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_u,
        ref_u,
        _ATOL_MOL,
        _RTOL_MOL,
        (withj, seed, "r_up"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )
    _assert_empty_channel(got_d, ref_d, (withj, seed, "r_dn"))


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("withj", [False, True])
def test_gfmc_stress_single_spin_ae(withj, seed):
    """Single-spin (AE stress): de_L^t/deps with an EMPTY down channel."""
    ham = _build_ham_for_cell("H_ae_ccpvdz_cart.h5", withj, None, "SD")
    _assert_single_spin(ham)
    wf = ham.wavefunction_data
    ru, rd = _cfg(ham, seed)
    ref = _oracle_ae_stress(ham, ru, rd)
    got = np.asarray(
        jqmc_rs._gfmc_stress_ae(wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3))
    )
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got,
        ref,
        _ATOL_MOL,
        _RTOL_MOL,
        (withj, seed),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dltmove", [False, True])
def test_gfmc_force_single_spin_ecp(dltmove, seed):
    """Single-spin (ECP force): de_L^t/d(R, r_up, r_dn) on the single-spin ECP cell,
    both T-move and determinant-locality moves."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = _build_ham_for_cell("Li_ecp_ccpvdz_cart.h5", True, None, "SD")
    _assert_single_spin(ham)
    wf = ham.wavefunction_data
    ru, rd = _cfg(ham, seed)
    assert rd.shape == (0, 3), rd.shape
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    ref_R, ref_u, ref_d = _oracle_ecp_grads(ham, ru, rd, dltmove=dltmove)
    got_R, got_u, got_d = jqmc_rs._gfmc_grad_e_L_ecp(
        wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1, dltmove
    )
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_R,
        ref_R,
        _ATOL_MOL,
        _RTOL_MOL,
        (dltmove, seed, "R"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_u,
        ref_u,
        _ATOL_MOL,
        _RTOL_MOL,
        (dltmove, seed, "r_up"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )
    _assert_empty_channel(got_d, ref_d, (dltmove, seed, "r_dn"))


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dltmove", [False, True])
def test_gfmc_stress_single_spin_ecp(dltmove, seed):
    """Single-spin (ECP stress): de_L^t/deps on the single-spin ECP cell."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = _build_ham_for_cell("Li_ecp_ccpvdz_cart.h5", False, None, "SD")
    _assert_single_spin(ham)
    gd = ham.wavefunction_data.geminal_data
    ru, rd = _cfg(ham, seed)
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    ref = _oracle_ecp_stress(ham, ru, rd, dltmove=dltmove)
    got = np.asarray(jqmc_rs._gfmc_stress_ecp_jnone(gd, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1, dltmove))
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got,
        ref,
        _ATOL_MOL,
        _RTOL_MOL,
        (dltmove, seed),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["t", "n"])
@pytest.mark.parametrize("stem", ["H_ae_ccpvdz_cart.h5", "Li_ecp_ccpvdz_cart.h5"])
def test_route_parity_gfmc_force_single_spin(driver, stem, monkeypatch):
    """Single-spin route parity: a PRODUCTION position-derivative GFMC run on the
    single-spin Hamiltonian reaches the native chunk with ``want_force`` on."""
    ham = _build_ham_for_cell(stem, True, None, "SD")
    _assert_single_spin(ham)
    ctx = f"single-spin-force/{stem}/{driver}"
    spy = install_ctx_spy(monkeypatch)
    _run_gfmc_force(driver, ham, 34356)
    _assert_force_route_native(spy, driver, ctx)


# --------------------------------------------------------------------------- #
# AGPn geminal x GFMC force / stress.
# --------------------------------------------------------------------------- #
@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gfmc_force_agpn(seed):
    """AGPn (force): de_L^t/d(R, r_up, r_dn) on the AGPn pairing geminal."""
    ham = rs_ham("mol_cart_agpn_jnone")
    gd = _assert_geminal_class(ham, AGPn_data, "agpn")
    wf = ham.wavefunction_data
    ru, rd = _cfg(ham, seed)
    ref_R, ref_u, ref_d = _oracle_grads(ham, ru, rd)
    got_R, got_u, got_d = jqmc_rs._gfmc_grad_e_L_ae(gd, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3))
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_R,
        ref_R,
        _ATOL_MOL,
        _RTOL_MOL,
        (seed, "R"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_u,
        ref_u,
        _ATOL_MOL,
        _RTOL_MOL,
        (seed, "r_up"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )
    _comparison_3_got, _comparison_3_ref, _comparison_3_atol, _comparison_3_rtol, _comparison_3_msg = (
        got_d,
        ref_d,
        _ATOL_MOL,
        _RTOL_MOL,
        (seed, "r_dn"),
    )
    _comparison_3_got = np.asarray(_comparison_3_got)
    _comparison_3_ref = np.asarray(_comparison_3_ref)
    assert _comparison_3_got.shape == _comparison_3_ref.shape, (
        _comparison_3_msg,
        _comparison_3_got.shape,
        _comparison_3_ref.shape,
    )
    _comparison_3_scale = max(1.0, float(np.abs(_comparison_3_ref).max()))
    _comparison_3_max_diff = float(np.abs(_comparison_3_got - _comparison_3_ref).max())
    assert _comparison_3_max_diff <= _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale, (
        _comparison_3_msg,
        _comparison_3_max_diff,
        _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale,
    )


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gfmc_stress_agpn(seed):
    """AGPn (stress): de_L^t/deps on the AGPn pairing geminal."""
    ham = rs_ham("mol_cart_agpn_jnone")
    gd = _assert_geminal_class(ham, AGPn_data, "agpn")
    wf = ham.wavefunction_data
    ru, rd = _cfg(ham, seed)
    ref = _oracle_ae_stress(ham, ru, rd)
    got = np.asarray(jqmc_rs._gfmc_stress_ae(gd, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3)))
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got,
        ref,
        _ATOL_MOL,
        _RTOL_MOL,
        seed,
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gfmc_grad_ln_psi_agpn(seed):
    """AGPn (Pulay building block): dln|Psi|/d(R, r_up, r_dn) on AGPn -- the other
    half of the GFMC force assembly."""
    ham = rs_ham("mol_cart_agpn_jnone")
    gd = _assert_geminal_class(ham, AGPn_data, "agpn")
    wf = ham.wavefunction_data
    ru, rd = _cfg(ham, seed)
    ref_R, ref_u, ref_d = _oracle_lnpsi_grads(ham, ru, rd)
    got_R, got_u, got_d = jqmc_rs._gfmc_grad_ln_Psi(gd, wf.jastrow_data, ru, rd)
    _comparison_1_got, _comparison_1_ref, _comparison_1_atol, _comparison_1_rtol, _comparison_1_msg = (
        got_R,
        ref_R,
        _ATOL_MOL,
        _RTOL_MOL,
        (seed, "R"),
    )
    _comparison_1_got = np.asarray(_comparison_1_got)
    _comparison_1_ref = np.asarray(_comparison_1_ref)
    assert _comparison_1_got.shape == _comparison_1_ref.shape, (
        _comparison_1_msg,
        _comparison_1_got.shape,
        _comparison_1_ref.shape,
    )
    _comparison_1_scale = max(1.0, float(np.abs(_comparison_1_ref).max()))
    _comparison_1_max_diff = float(np.abs(_comparison_1_got - _comparison_1_ref).max())
    assert _comparison_1_max_diff <= _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale, (
        _comparison_1_msg,
        _comparison_1_max_diff,
        _comparison_1_atol + _comparison_1_rtol * _comparison_1_scale,
    )
    _comparison_2_got, _comparison_2_ref, _comparison_2_atol, _comparison_2_rtol, _comparison_2_msg = (
        got_u,
        ref_u,
        _ATOL_MOL,
        _RTOL_MOL,
        (seed, "r_up"),
    )
    _comparison_2_got = np.asarray(_comparison_2_got)
    _comparison_2_ref = np.asarray(_comparison_2_ref)
    assert _comparison_2_got.shape == _comparison_2_ref.shape, (
        _comparison_2_msg,
        _comparison_2_got.shape,
        _comparison_2_ref.shape,
    )
    _comparison_2_scale = max(1.0, float(np.abs(_comparison_2_ref).max()))
    _comparison_2_max_diff = float(np.abs(_comparison_2_got - _comparison_2_ref).max())
    assert _comparison_2_max_diff <= _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale, (
        _comparison_2_msg,
        _comparison_2_max_diff,
        _comparison_2_atol + _comparison_2_rtol * _comparison_2_scale,
    )
    _comparison_3_got, _comparison_3_ref, _comparison_3_atol, _comparison_3_rtol, _comparison_3_msg = (
        got_d,
        ref_d,
        _ATOL_MOL,
        _RTOL_MOL,
        (seed, "r_dn"),
    )
    _comparison_3_got = np.asarray(_comparison_3_got)
    _comparison_3_ref = np.asarray(_comparison_3_ref)
    assert _comparison_3_got.shape == _comparison_3_ref.shape, (
        _comparison_3_msg,
        _comparison_3_got.shape,
        _comparison_3_ref.shape,
    )
    _comparison_3_scale = max(1.0, float(np.abs(_comparison_3_ref).max()))
    _comparison_3_max_diff = float(np.abs(_comparison_3_got - _comparison_3_ref).max())
    assert _comparison_3_max_diff <= _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale, (
        _comparison_3_msg,
        _comparison_3_max_diff,
        _comparison_3_atol + _comparison_3_rtol * _comparison_3_scale,
    )


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["t", "n"])
def test_route_parity_gfmc_force_agpn(driver, monkeypatch):
    """AGPn route parity: a PRODUCTION position-derivative GFMC run on the AGPn
    Hamiltonian reaches the native chunk with ``want_force`` on."""
    ham = rs_ham("mol_cart_agpn_jnone")
    _assert_geminal_class(ham, AGPn_data, "agpn")
    ctx = f"agpn-force/{driver}"
    spy = install_ctx_spy(monkeypatch)
    _run_gfmc_force(driver, ham, 34356)
    _assert_force_route_native(spy, driver, ctx)
