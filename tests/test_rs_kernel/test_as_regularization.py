"""Equivalence: Attaccalite-Sorella regularization factor, jqmc_rs vs jqmc.

R_AS = (S*F)^(-3/8), F = ||G_inv||^2, S = min(min row norm^2, min col norm^2).
Geminal-only (AO-independent at this layer); validated on committed hams.

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
    compute_AS_regularization_factor_fast_update,
    compute_geminal_all_elements,
)

from tests.test_rs_kernel._rs_fixtures import rs_ham

_ZONES = ("det_eval",)

# name -> (canonical fixture pattern, is_periodic)
_SYSTEMS = {
    "mol_ecp_sphe": ("mol_ecp_sphe_jnone", False),
    "mol_open_cart": ("mol_cart_sd_jnone", False),
    "pbc_sphepgto": ("pbc_gamma_sphepgto_jnone", True),
}


def _electrons(ham, seed):
    rng = np.random.default_rng(seed)
    gem = ham.wavefunction_data.geminal_data
    sd = ham.structure_data
    n_up, n_dn = int(gem.num_electron_up), int(gem.num_electron_dn)
    if sd.pbc_flag:
        cell = np.asarray(sd._cell_np, dtype=np.float64)
        return (rng.random((n_up, 3)) @ cell), (rng.random((n_dn, 3)) @ cell)
    R = np.asarray(sd.positions, dtype=np.float64)
    r_up = R[rng.integers(0, len(R), n_up)] + rng.normal(scale=0.7, size=(n_up, 3))
    r_dn = R[rng.integers(0, len(R), n_dn)] + rng.normal(scale=0.7, size=(n_dn, 3))
    return r_up.astype(np.float64), r_dn.astype(np.float64)


@pytest.mark.light
@pytest.mark.parametrize("name", list(_SYSTEMS))
@pytest.mark.parametrize("seed", [0, 7])
def test_as_factor_matches_jax(name, seed):
    """As factor matches jax."""
    pattern, periodic = _SYSTEMS[name]
    ham = rs_ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    r_up, r_dn = _electrons(ham, seed)
    G = np.asarray(compute_geminal_all_elements(gem, r_up, r_dn), dtype=np.float64)
    G_inv = np.asarray(jqmc_rs._matrix_inverse(G), dtype=np.float64)
    ref = float(compute_AS_regularization_factor_fast_update(G, G_inv))
    got = jqmc_rs.compute_AS_regularization_factor_fast_update(G, G_inv)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    assert np.isclose(got, ref, rtol=rtol, atol=atol), (got, ref)
