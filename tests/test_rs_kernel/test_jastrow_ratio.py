"""Equivalence: J1*J2 single-electron Jastrow ratio, jqmc_rs vs jqmc.

Validates `jastrow_1b2b_ratio` against `_compute_ratio_Jastrow_part_rank1_update`
(n_grid=1) on committed molecular j12 hamiltonians, AND on a periodic j12
system (Williamson smooth periodic distance for J2; min-image-agnostic J1).
The ground truth is the full Jastrow recompute exp(J(r')-J(r)) in both cases.

Committed Hamiltonian fixtures:
  - water_cluster_4 ecp sphe j12 -> mol_ecp_sphe_j12
  - Cl2 ecp cart j12             -> mol_ecp_cart_j12
  - H2 ae sphe j12               -> mol_cart_sd_j12 (ae cart j12; the AO
    family is immaterial to the ratio-vs-recompute equivalence)
  - lithium_pbc ecp sphepgto j12 -> pbc_gamma_sphepgto_j12
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

from jqmc._precision import get_tolerance_min, get_tolerance_xla_rust_consistency
from jqmc.jastrow_factor import (
    _compute_ratio_Jastrow_part_rank1_update,
    compute_Jastrow_part,
)

from tests.test_rs_kernel._rs_fixtures import rs_ham

# XLA=Rust parity compares the same rank-1 route with the dedicated strict
# tolerance.  The independent full recompute remains a numerical-consistency
# oracle at the ordinary zone tolerance: in fp32 its reduction tree is
# intentionally different and can differ by more than backend parity permits.
_ZONES = ("jastrow_eval", "jastrow_ratio")

_MOL_PATTERNS = {
    "mol_ecp_sphe_j12": "mol_ecp_sphe_j12",
    "mol_ecp_cart_j12": "mol_ecp_cart_j12",
    "mol_cart_sd_j12": "mol_cart_sd_j12",
}


def _electrons(ham, seed):
    rng = np.random.default_rng(seed)
    gem = ham.wavefunction_data.geminal_data
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    n_up, n_dn = int(gem.num_electron_up), int(gem.num_electron_dn)
    r_up = R[rng.integers(0, len(R), n_up)] + rng.normal(scale=0.7, size=(n_up, 3))
    r_dn = R[rng.integers(0, len(R), n_dn)] + rng.normal(scale=0.7, size=(n_dn, 3))
    return r_up.astype(np.float64), r_dn.astype(np.float64), R


@pytest.mark.light
@pytest.mark.parametrize("name", list(_MOL_PATTERNS))
@pytest.mark.parametrize("is_up", [True, False])
@pytest.mark.parametrize("seed", [0, 5])
def test_jastrow_1b2b_ratio_matches_jax(name, is_up, seed):
    """Jastrow 1b2b ratio matches jax."""
    ham = rs_ham(_MOL_PATTERNS[name])
    jdata = ham.wavefunction_data.jastrow_data
    r_up, r_dn, R = _electrons(ham, seed)
    rng = np.random.default_rng(seed + 50)
    idx = 0
    new = (R[rng.integers(0, len(R))] + rng.normal(scale=0.7, size=3)).astype(np.float64)

    new_r_up = r_up.copy()
    new_r_dn = r_dn.copy()
    if is_up:
        new_r_up[idx] = new
    else:
        new_r_dn[idx] = new

    # ground truth: full recompute
    truth = float(np.exp(compute_Jastrow_part(jdata, new_r_up, new_r_dn) - compute_Jastrow_part(jdata, r_up, r_dn)))
    rank1 = float(
        np.asarray(_compute_ratio_Jastrow_part_rank1_update(jdata, r_up, r_dn, new_r_up[None, ...], new_r_dn[None, ...]))[0]
    )
    got = jqmc_rs.jastrow_1b2b_ratio(jdata, r_up, r_dn, is_up, idx, new)
    # Rust must match the corresponding XLA rank-1 route strictly.
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    assert np.isclose(got, rank1, rtol=rtol, atol=atol), (got, rank1)
    # Full-recompute cancellation is a separate, ordinary fp32/fp64 check.
    atol_n, rtol_n = get_tolerance_min(_ZONES, level="strict")
    assert np.isclose(got, truth, rtol=rtol_n, atol=atol_n), (got, truth)


@pytest.mark.light
@pytest.mark.parametrize("is_up", [True, False])
@pytest.mark.parametrize("seed", [0, 5])
def test_jastrow_1b2b_ratio_pbc_matches_jax(is_up, seed):
    """Jastrow 1b2b ratio PBC matches jax."""
    ham = rs_ham("pbc_gamma_sphepgto_j12")
    jdata = ham.wavefunction_data.jastrow_data
    gem = ham.wavefunction_data.geminal_data
    n_up, n_dn = int(gem.num_electron_up), int(gem.num_electron_dn)
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    rng = np.random.default_rng(seed)
    # Unwrapped walkers (incl. outside the home cell), as the native sweep carries them.
    r_up = (rng.random((n_up, 3)) @ cell + rng.integers(-2, 3, (n_up, 3)) @ cell).astype(np.float64)
    r_dn = (rng.random((n_dn, 3)) @ cell).astype(np.float64)
    idx = 0
    new = ((r_up[idx] if is_up else r_dn[idx]) + rng.normal(scale=0.6, size=3)).astype(np.float64)

    new_r_up, new_r_dn = r_up.copy(), r_dn.copy()
    if is_up:
        new_r_up[idx] = new
    else:
        new_r_dn[idx] = new

    truth = float(np.exp(compute_Jastrow_part(jdata, new_r_up, new_r_dn) - compute_Jastrow_part(jdata, r_up, r_dn)))
    rank1 = float(
        np.asarray(_compute_ratio_Jastrow_part_rank1_update(jdata, r_up, r_dn, new_r_up[None, ...], new_r_dn[None, ...]))[0]
    )
    got = jqmc_rs.jastrow_1b2b_ratio(jdata, r_up, r_dn, is_up, idx, new)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    assert np.isclose(got, rank1, rtol=rtol, atol=atol), (got, rank1)
    atol_n, rtol_n = get_tolerance_min(_ZONES, level="strict")
    assert np.isclose(got, truth, rtol=rtol_n, atol=atol_n), (got, truth)
