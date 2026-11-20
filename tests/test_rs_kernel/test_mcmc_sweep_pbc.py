"""Layer-2 (PBC): native mcmc_sweep vs jqmc, periodic systems.

Same harness as ``test_mcmc_sweep.py`` but periodic: the proposal's nearest
nucleus + cusp distance use the Wigner-Seitz min image (jqmc
``_find_nearest_index_jnp`` / ``_get_min_dist_rel_R_cart_jnp``), and ``cell`` /
``n_max`` are passed to ``_mcmc_sweep``. The reference's det-ratio, geminal and
Jastrow-rank1 functions all wrap / min-image internally (``compute_AOs`` ->
``_wrap_into_cell_if_pbc``), so the Rust side must mirror that: it evaluates AOs
at home-cell-wrapped coordinates while carrying walker positions UNWRAPPED, and
its J1/J2 ratio + nearest atom use the translation-invariant periodic forms.
Validates the periodic streaming advance + the whole PBC move loop.

Fixtures: the committed ``lithium.h5`` periodic ECP sphe-pGTO stem via ``rs_ham``:
  - row "j12":    ``pbc_gamma_sphepgto_j12``  (J1+J2, exp).
  - row "j123ao": ``pbc_gamma_sphepgto_j123`` (J1+J2+J3 pgto leaf), the AO-J3
    streaming code path.
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

import jax  # noqa: E402
from jax import numpy as jnp  # noqa: E402
from jax import random  # noqa: E402

import jqmc_rs

pytestmark = pytest.mark.with_chdir

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
from jqmc.structure import (
    _find_nearest_index_jnp,
    _get_min_dist_rel_R_cart_jnp,
)

from tests.test_rs_kernel._rs_fixtures import rs_ham

# Periodic: the Rust AO eval carries a deliberate far-image cutoff (a perf
# truncation), so use the looser "medium" zone level (matching
# test_determinant.py's periodic row and _rs_fixtures's periodic convention).
_GEM_ZONES = ("det_ratio", "det_eval", "ao_eval")

SYSTEMS = {
    "lithium_pbc_j12": "pbc_gamma_sphepgto_j12",
    "lithium_pbc_j123ao": "pbc_gamma_sphepgto_j123",
}


def _charges(ham):
    cp = ham.coulomb_potential_data
    Z = np.asarray(ham.structure_data.atomic_numbers, dtype=np.float64)
    if cp.ecp_flag:
        Z = Z - np.asarray(cp.z_cores, dtype=np.float64)
    return Z


def _one_move(ham, r_up, r_dn, G, G_inv, dt, eps, key, injected_normal):
    gem = ham.wavefunction_data.geminal_data
    jas = ham.wavefunction_data.jastrow_data
    sd = ham.structure_data
    Z = _charges(ham)
    n_up, n_dn = len(r_up), len(r_dn)

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
        i = int(_find_nearest_index_jnp(sd, r))
        rel = np.asarray(_get_min_dist_rel_R_cart_jnp(sd, r, i), dtype=np.float64)
        return Z[i], float(np.linalg.norm(rel))

    z, norm = nearest(old_r)
    f_l = 1.0 / z**2 * (1 + z**2 * norm) / (1 + norm)
    key, sk = random.split(key)  # normal subkey consumed
    g = injected_normal * (f_l * dt)
    key, sk = random.split(key)
    axis = int(random.randint(sk, (), 0, 3))
    new_r = old_r.copy()
    new_r[axis] += g
    z2, norm2 = nearest(new_r)
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
    Det_T = 1.0 + v @ Ainv_u
    G_inv_new = G_inv - np.outer(Ainv_u, v @ G_inv) / Det_T
    G_new = G.copy()
    if is_up:
        G_new[sel, :] += v
    else:
        G_new[:, sel] += u

    R_AS_p = float(compute_AS_regularization_factor_fast_update(G_new, G_inv_new))
    R_AS_o = float(compute_AS_regularization_factor_fast_update(G, G_inv))
    R_AS_ratio = (max(R_AS_p, eps) / R_AS_p) / (max(R_AS_o, eps) / R_AS_o)

    nu, nd = r_up.copy(), r_dn.copy()
    if is_up:
        nu[sel] = new_r
    else:
        nd[sel] = new_r
    J_ratio = float(np.asarray(_compute_ratio_Jastrow_part_rank1_update(jas, r_up, r_dn, nu[None], nd[None]))[0])
    R_ratio = (R_AS_ratio * J_ratio * Det_T) ** 2
    key, sk = random.split(key)
    b = float(random.uniform(sk, (), dtype=jnp.float64, minval=0.0, maxval=1.0))
    accept = b < min(1.0, R_ratio * T_ratio)
    if accept:
        return nu, nd, G_new, G_inv_new, key, True
    return r_up, r_dn, G, G_inv, key, False


@pytest.mark.light
@pytest.mark.parametrize("name", list(SYSTEMS))
@pytest.mark.parametrize("seed", [0, 1, 5])
@pytest.mark.parametrize("num_moves", [1, 10, 40])
def test_mcmc_sweep_pbc_matches_reference(name, seed, num_moves):
    """MCMC sweep PBC matches reference."""
    ham = rs_ham(SYSTEMS[name])
    gem = ham.wavefunction_data.geminal_data
    jas = ham.wavefunction_data.jastrow_data
    sd = ham.structure_data
    cell = np.asarray(sd.cell, dtype=np.float64)
    n_max = int(sd._n_max)
    rng = np.random.default_rng(seed + 1234)
    n_up, n_dn = gem.num_electron_up, gem.num_electron_dn
    # Start inside the home cell; the sweep carries positions unwrapped thereafter.
    r_up0 = (rng.random((n_up, 3)) @ cell).astype(np.float64)
    r_dn0 = (rng.random((n_dn, 3)) @ cell).astype(np.float64)
    G0 = np.asarray(compute_geminal_all_elements(gem, r_up0, r_dn0), dtype=np.float64)
    G_inv0 = np.asarray(jqmc_rs._matrix_inverse(G0), dtype=np.float64)
    dt, eps = 1.0, 0.0
    injected = rng.normal(size=num_moves).astype(np.float64)
    Z = _charges(ham)

    # Rust sweep (periodic: pass cell + n_max)
    k = np.asarray(random.PRNGKey(seed), dtype=np.uint32)
    res = jqmc_rs._mcmc_sweep(
        gem,
        jas,
        np.asarray(sd.positions, np.float64),
        Z,
        r_up0,
        r_dn0,
        G0,
        G_inv0,
        dt,
        eps,
        int(k[0]),
        int(k[1]),
        injected,
        cell,
        n_max,
    )

    # The REFERENCE side gets its OWN inverse from numpy, not the Rust
    # ``_matrix_inverse`` the native sweep is seeded with. Feeding one Rust-computed
    # G^-1 to both sides made a bug in that kernel cancel out of the comparison -- both
    # trajectories would start from the same wrong matrix and still agree.
    # Reference: iterate _one_move (jqmc fns wrap / min-image internally)
    r_up, r_dn, G, G_inv = r_up0.copy(), r_dn0.copy(), G0.copy(), np.linalg.inv(G0)
    key = random.PRNGKey(seed)
    acc = 0
    for mv in range(num_moves):
        r_up, r_dn, G, G_inv, key, a = _one_move(ham, r_up, r_dn, G, G_inv, dt, eps, key, float(injected[mv]))
        acc += int(a)

    gem_atol, gem_rtol = get_tolerance_xla_rust_consistency(_GEM_ZONES, level="medium")
    assert res["accepted"] == acc, (res["accepted"], acc)
    # BIT-EXACT: an accepted position is a pure COPY of the same
    # ``old_r + injected_normal * sigma`` arithmetic on both sides -- no accumulation, no
    # reassociation -- which is why ``test_single_move.py`` already pins the identical
    # quantity at ATOL/RTOL_BIT_EXACT. Judging it at the medium zone level (1e-7/1e-5) here
    # meant a genuine position drift of up to 1e-7 Bohr per electron -- i.e. the two sides
    # silently walking apart -- read as agreement. Measured difference: exactly 0.0.
    np.testing.assert_allclose(
        res["r_up"], r_up, atol=ATOL_BIT_EXACT, rtol=RTOL_BIT_EXACT, err_msg=f"{name} seed={seed} r_up (pure copy)"
    )
    np.testing.assert_allclose(
        res["r_dn"], r_dn, atol=ATOL_BIT_EXACT, rtol=RTOL_BIT_EXACT, err_msg=f"{name} seed={seed} r_dn (pure copy)"
    )
    assert np.allclose(res["geminal_inv"], G_inv, atol=gem_atol, rtol=gem_rtol), np.abs(res["geminal_inv"] - G_inv).max()
    fk = np.asarray(res["final_key"], dtype=np.uint32)
    refk = np.asarray(key, dtype=np.uint32)
    assert (int(fk[0]), int(fk[1])) == (int(refk[0]), int(refk[1])), "key drift"
