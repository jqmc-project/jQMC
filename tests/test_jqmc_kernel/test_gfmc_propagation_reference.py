"""Deterministic leaf-assembly anchors for the novar=0 routing guarantee."""

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

import jax
import numpy as np
import pytest

from jqmc import jqmc_gfmc
from jqmc._driver_helpers import derive_rank_rng_streams
from jqmc._precision import get_tolerance_min
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel import test_gfmc_observables as observables
from tests.test_rs_kernel import test_gfmc_projection as projection
from tests.test_rs_kernel.test_gfmc_projection import _cfg


pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


def _novar_state(ham, state):
    """Replay the routing fixture's initial configuration or fixed-key first hop."""
    r_up, r_dn = _cfg(ham, seed=0, scale=0.5)
    if state == "after-one-hop":
        _, mpi_seed = derive_rank_rng_streams(34, 0)
        key = jax.random.split(jax.random.PRNGKey(mpi_seed), 1)[0]
        interval = projection._xla_interval_n(ham, r_up, r_dn, 0.30, -1.0, 1, key, "tmove")
        r_up, r_dn = interval["r_up"], interval["r_dn"]
    else:
        assert state == "initial"
    return r_up, r_dn


@pytest.mark.parametrize("state", ["initial", "after-one-hop"])
def test_novar0_propagation_matches_independent_leaf_assembly(state):
    """The novar=0 propagation diagonal and off-diagonal equal independent leaf sums.

    The committed tests/trexio_example_files/N_ae_ccpvdz_cart.h5 supplies an open,
    all-electron nitrogen SD with Cartesian GTOs, no Jastrow and four up/three down
    electrons (open-shell). Positions use NumPy PCG64 seed 0, normal scale 0.5
    around the nucleus. Evaluate either that configuration or its first
    load-balanced hop with driver seed 34, rank 0 and one walker. The fixed
    identity mesh has alat=0.30, E_scf=-1 and tmove; no ensemble is sampled.
    The two configurations must have respectively four and three sign-flipping
    electrons. No cond(G) screening or upper bound is imposed.

    Compare the XLA closure's diagonal and FN off-diagonal against separately
    assembled Coulomb, continuum/discrete kinetic and sign-flip terms. Shared
    leaf kernels are not an independent physical Hamiltonian implementation.
    Only the helper's leaf sums, never its production-closure output fields,
    supply the reference. Together with the same guarantee's four backend/
    algorithm routing cases this anchors propagation energy before testing its
    use in weights. Each mode uses existing zone strict tolerances; no sigma
    band, comparison between precision modes or ground-state energy is used.
    """
    ham = _build_ham_for_cell("N_ae_ccpvdz_cart.h5", False, None, "SD")
    r_up, r_dn = _novar_state(ham, state)
    step = projection._xla_step(ham, r_up, r_dn, 0.30, np.eye(3), "tmove")
    expected_flips = {"initial": 4, "after-one-hop": 3}[state]
    assert np.count_nonzero(step["sign_flip"]) == expected_flips, "novar fixture lost its distinct sign-flip state"
    diagonal, off_diagonal, _ = jqmc_gfmc._compute_V_elements_t(ham, r_up, r_dn, np.eye(3), "tmove", 0.30)
    leaf = observables._xla_components(ham, r_up, r_dn, 0.30, np.eye(3), "tmove")
    expected_diagonal, expected_off_diagonal = leaf["v_diag_leaf"], leaf["b_l_leaf"]
    atol, rtol = get_tolerance_min(("ao_eval", "wf_kinetic", "coulomb", "local_energy", "det_ratio"), "strict")
    np.testing.assert_allclose(diagonal, expected_diagonal, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(off_diagonal, expected_off_diagonal, atol=atol, rtol=rtol, equal_nan=False)
