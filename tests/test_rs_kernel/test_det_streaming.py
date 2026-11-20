"""Core equivalence: Sherman-Morrison det-ratio streaming, jqmc_rs vs jqmc.

Validates, on committed hams (molecular closed/open shell + periodic target):
  - init streaming state vs _init_det_ratio_streaming_state
  - v (up-move) / u (dn-move) vs _compute_v/u_*_from_det_ratio_state
  - advance state vs _advance_det_ratio_streaming_state
  - the rank-1 SM det ratio + inverse update vs direct full recompute

Fixtures are the committed hams:
  - water_cluster_4 ecp sphe jnone -> mol_ecp_sphe_jnone
  - Li_open ae cart jnone          -> mol_cart_sd_jnone (N open shell 5+2)
  - periodic ecp sphepgto jnone    -> pbc_gamma_sphepgto_jnone
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

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.determinant import (
    _advance_det_ratio_streaming_state,
    _compute_u_dn_move_from_det_ratio_state,
    _compute_v_up_move_from_det_ratio_state,
    _init_det_ratio_streaming_state,
    compute_geminal_all_elements,
)

from tests.test_rs_kernel._rs_fixtures import rs_ham

# name -> (canonical fixture pattern, is_periodic)
_SYSTEMS = {
    "mol_ecp_sphe": ("mol_ecp_sphe_jnone", False),
    "mol_open_cart": ("mol_cart_sd_jnone", False),
    "pbc_sphepgto": ("pbc_gamma_sphepgto_jnone", True),
}

_ZONES_STATE = ("ao_eval", "det_eval")
_ZONES_RATIO = ("det_ratio",)


def _electrons(ham, seed):
    rng = np.random.default_rng(seed)
    gem = ham.wavefunction_data.geminal_data
    sd = ham.structure_data
    n_up, n_dn = int(gem.num_electron_up), int(gem.num_electron_dn)
    if sd.pbc_flag:
        cell = np.asarray(sd._cell_np, dtype=np.float64)
        return (rng.random((n_up, 3)) @ cell), (rng.random((n_dn, 3)) @ cell), cell
    R = np.asarray(sd.positions, dtype=np.float64)
    r_up = R[rng.integers(0, len(R), n_up)] + rng.normal(scale=0.7, size=(n_up, 3))
    r_dn = R[rng.integers(0, len(R), n_dn)] + rng.normal(scale=0.7, size=(n_dn, 3))
    return r_up.astype(np.float64), r_dn.astype(np.float64), R


def _new_pos(ham, ref, seed):
    rng = np.random.default_rng(seed + 100)
    sd = ham.structure_data
    if sd.pbc_flag:
        return (rng.random(3) @ ref).astype(np.float64)
    return (ref[rng.integers(0, len(ref))] + rng.normal(scale=0.7, size=3)).astype(np.float64)


@pytest.mark.light
@pytest.mark.parametrize("name", list(_SYSTEMS))
def test_init_state_matches_jax(name):
    """Init state matches jax."""
    pattern, periodic = _SYSTEMS[name]
    ham = rs_ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    r_up, r_dn, _ = _electrons(ham, 0)
    st = _init_det_ratio_streaming_state(gem, r_up, r_dn)
    ao_up, ao_dn, paired_dn, paired_up_lambda = jqmc_rs._det_init_state(gem, r_up, r_dn)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_STATE, level="strict")
    assert np.allclose(ao_up, np.asarray(st.ao_up), rtol=rtol, atol=atol)
    assert np.allclose(ao_dn, np.asarray(st.ao_dn), rtol=rtol, atol=atol)
    assert np.allclose(paired_dn, np.asarray(st.paired_dn), rtol=rtol, atol=atol)
    assert np.allclose(paired_up_lambda, np.asarray(st.paired_up_lambda), rtol=rtol, atol=atol)


@pytest.mark.light
@pytest.mark.parametrize("name", list(_SYSTEMS))
def test_v_up_move_matches_jax(name):
    """V up move matches jax."""
    pattern, periodic = _SYSTEMS[name]
    ham = rs_ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    r_up, r_dn, ref = _electrons(ham, 1)
    i = 0
    new = _new_pos(ham, ref, 1)
    r_up_prop = r_up.copy()
    r_up_prop[i] = new
    st = _init_det_ratio_streaming_state(gem, r_up, r_dn)
    v_ref = np.asarray(_compute_v_up_move_from_det_ratio_state(gem, st, i, r_up_prop), dtype=np.float64)
    v_got = np.asarray(jqmc_rs._det_v_up_move(gem, r_up, r_dn, i, new), dtype=np.float64)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_STATE, level="strict")
    assert np.allclose(v_got, v_ref, rtol=rtol, atol=atol), np.abs(v_got - v_ref).max()


@pytest.mark.light
@pytest.mark.parametrize("name", list(_SYSTEMS))
def test_u_dn_move_matches_jax(name):
    """U dn move matches jax."""
    pattern, periodic = _SYSTEMS[name]
    ham = rs_ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    if gem.num_electron_dn == 0:
        pytest.skip("no dn electrons")
    r_up, r_dn, ref = _electrons(ham, 2)
    j = 0
    new = _new_pos(ham, ref, 2)
    r_dn_prop = r_dn.copy()
    r_dn_prop[j] = new
    st = _init_det_ratio_streaming_state(gem, r_up, r_dn)
    u_ref = np.asarray(_compute_u_dn_move_from_det_ratio_state(gem, st, j, r_dn_prop), dtype=np.float64)
    u_got = np.asarray(jqmc_rs._det_u_dn_move(gem, r_up, r_dn, j, new), dtype=np.float64)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_STATE, level="strict")
    assert np.allclose(u_got, u_ref, rtol=rtol, atol=atol), np.abs(u_got - u_ref).max()


@pytest.mark.light
@pytest.mark.parametrize("name", list(_SYSTEMS))
@pytest.mark.parametrize("is_up", [True, False])
def test_advance_state_matches_jax(name, is_up):
    """Advance state matches jax."""
    pattern, periodic = _SYSTEMS[name]
    ham = rs_ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    if not is_up and gem.num_electron_dn == 0:
        pytest.skip("no dn electrons")
    r_up, r_dn, ref = _electrons(ham, 3)
    idx = 0
    new = _new_pos(ham, ref, 3)
    r_up_new, r_dn_new = r_up.copy(), r_dn.copy()
    if is_up:
        r_up_new[idx] = new
    else:
        r_dn_new[idx] = new
    st = _init_det_ratio_streaming_state(gem, r_up, r_dn)
    st_adv = _advance_det_ratio_streaming_state(gem, st, np.bool_(is_up), idx, r_up_new, r_dn_new)
    ao_up, ao_dn, paired_dn, paired_up_lambda = jqmc_rs._det_advance_state(gem, r_up, r_dn, is_up, idx, new)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_STATE, level="strict")
    assert np.allclose(ao_up, np.asarray(st_adv.ao_up), rtol=rtol, atol=atol)
    assert np.allclose(ao_dn, np.asarray(st_adv.ao_dn), rtol=rtol, atol=atol)
    assert np.allclose(paired_dn, np.asarray(st_adv.paired_dn), rtol=rtol, atol=atol)
    assert np.allclose(paired_up_lambda, np.asarray(st_adv.paired_up_lambda), rtol=rtol, atol=atol)


@pytest.mark.light
@pytest.mark.parametrize("name", list(_SYSTEMS))
@pytest.mark.parametrize("is_up", [True, False])
def test_sm_ratio_and_inverse_vs_direct(name, is_up):
    """Sm ratio and inverse vs direct."""
    pattern, periodic = _SYSTEMS[name]
    ham = rs_ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    if not is_up and gem.num_electron_dn == 0:
        pytest.skip("no dn electrons")
    r_up, r_dn, ref = _electrons(ham, 4)
    idx = 0
    new = _new_pos(ham, ref, 4)

    G = np.asarray(compute_geminal_all_elements(gem, r_up, r_dn), dtype=np.float64)
    G_inv = np.asarray(jqmc_rs._matrix_inverse(G), dtype=np.float64)
    n_up = G.shape[0]

    if is_up:
        v = np.asarray(jqmc_rs._det_v_up_move(gem, r_up, r_dn, idx, new), dtype=np.float64)
        u = np.eye(n_up, dtype=np.float64)[idx]
        r_up_new = r_up.copy()
        r_up_new[idx] = new
        G_new = np.asarray(compute_geminal_all_elements(gem, r_up_new, r_dn), dtype=np.float64)
    else:
        u = np.asarray(jqmc_rs._det_u_dn_move(gem, r_up, r_dn, idx, new), dtype=np.float64)
        v = np.eye(n_up, dtype=np.float64)[idx]
        r_dn_new = r_dn.copy()
        r_dn_new[idx] = new
        G_new = np.asarray(compute_geminal_all_elements(gem, r_up, r_dn_new), dtype=np.float64)

    ratio, inv_new, gem_new = jqmc_rs._det_sm_update(G, G_inv, u, v, is_up, idx)

    # The SM ratio/inverse are for gem_new = G + u.v^T (the matrix SM builds from
    # the single-position AO path). gem_new matches the batch-built direct geminal
    # G_new only to ~1e-10 AO-eval rounding; for an ill-conditioned geminal (large
    # periodic systems) that rounding amplifies, so referencing G_new directly
    # is too strict there. Check the ratio against det(gem_new)/det(G) (matrix-
    # determinant lemma -- what SM actually computes) at the looser "medium" level,
    # regardless of the periodic axis, since the amplification is conditioning-
    # driven rather than a BC-specific truncation.
    ratio_direct = np.linalg.det(gem_new) / np.linalg.det(G)
    atol_r, rtol_r = get_tolerance_xla_rust_consistency(_ZONES_RATIO, level="strict")
    assert np.isclose(ratio, ratio_direct, rtol=rtol_r, atol=atol_r), (ratio, ratio_direct)
    # v/u is physically correct: gem_new equals the direct geminal to ~fp64 zone tol.
    atol_s, rtol_s = get_tolerance_xla_rust_consistency(_ZONES_STATE, level="strict")
    assert np.allclose(gem_new, G_new, rtol=rtol_s, atol=atol_s), np.abs(gem_new - G_new).max()
    # SM inverse correct via the condition-aware backward residual on its own
    # matrix gem_new (not element-wise vs inv(G_new), which the AO-rounding gap
    # amplifies for ill-conditioned systems).
    resid = np.abs(gem_new @ inv_new - np.eye(n_up)).max()
    atol_resid, _ = get_tolerance_xla_rust_consistency(_ZONES_RATIO, level="strict")
    assert resid < atol_resid, resid
