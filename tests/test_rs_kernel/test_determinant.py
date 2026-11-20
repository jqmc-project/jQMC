"""Rust vs XLA: geminal matrix + ln|det| (fixtures are the committed hams).

Validates ``jqmc_rs.compute_geminal_all_elements`` and
``compute_ln_det_geminal_all_elements`` (hand-rolled matmul + LU, no BLAS) against
jqmc's XLA implementation on committed fixtures spanning: molecular closed/open
shell (the unpaired lambda block), molecular ECP sphe-GTO, and periodic sphe-pGTO
(the MO-basis periodic geminal). A Rust==XLA equivalence is code-path sensitive,
not system sensitive, so each committed fixture stands in for the larger bench
system that flowed the same path.
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
# xdist safety; committed fixtures load via absolute paths so cwd is irrelevant).
pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.determinant import compute_geminal_all_elements, compute_ln_det_geminal_all_elements

from tests.test_rs_kernel._rs_fixtures import draw_cfg, rs_ham

# Molecular is bit-exact; periodic carries the deliberate per-electron far-image AO
# cutoff (~1e-6, a perf truncation) that accumulates into ln|det| over N electrons,
# so periodic uses the looser ``medium`` zone tolerance.
_ZONES = ("ao_eval", "det_eval")

# name -> (canonical fixture pattern, is_periodic)
_SYSTEMS = {
    "mol_closed_ecp_sphe": ("mol_ecp_sphe_jnone", False),
    "mol_open_cart": ("mol_cart_sd_jnone", False),  # N (5+2) open shell -> unpaired block
    "pbc_sphepgto": ("pbc_gamma_sphepgto_jnone", True),
}


def _electrons(ham, seed):
    rng = np.random.default_rng(seed)
    gem = ham.wavefunction_data.geminal_data
    n_up, n_dn = int(gem.num_electron_up), int(gem.num_electron_dn)
    sd = ham.structure_data
    if sd.pbc_flag:
        cell = np.asarray(sd._cell_np, dtype=np.float64)
        return (rng.random((n_up, 3)) @ cell, rng.random((n_dn, 3)) @ cell)
    R = np.asarray(sd.positions, dtype=np.float64)
    r_up = R[rng.integers(0, len(R), size=n_up)] + rng.normal(scale=0.7, size=(n_up, 3))
    r_dn = R[rng.integers(0, len(R), size=n_dn)] + rng.normal(scale=0.7, size=(n_dn, 3))
    return r_up, r_dn


@pytest.mark.light
@pytest.mark.parametrize("name", list(_SYSTEMS))
@pytest.mark.parametrize("seed", [0, 7])
def test_geminal_matrix_matches_jax(name, seed):
    """Geminal matrix matches jax."""
    pattern, periodic = _SYSTEMS[name]
    ham = rs_ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    r_up, r_dn = _electrons(ham, seed)
    ref = np.asarray(compute_geminal_all_elements(gem, r_up, r_dn), dtype=np.float64)
    got = np.asarray(jqmc_rs.compute_geminal_all_elements(gem, r_up, r_dn))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert np.allclose(got, ref, rtol=rtol, atol=atol), np.abs(got - ref).max()


@pytest.mark.light
@pytest.mark.parametrize("name", list(_SYSTEMS))
@pytest.mark.parametrize("seed", [0, 7])
def test_ln_det_matches_jax(name, seed):
    """Ln det matches jax."""
    pattern, periodic = _SYSTEMS[name]
    ham = rs_ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    r_up, r_dn = _electrons(ham, seed)
    ref = float(compute_ln_det_geminal_all_elements(gem, r_up, r_dn))
    got = float(jqmc_rs.compute_ln_det_geminal_all_elements(gem, r_up, r_dn))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    assert np.isclose(got, ref, rtol=rtol, atol=atol), (got, ref)
