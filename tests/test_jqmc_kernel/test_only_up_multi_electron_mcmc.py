"""XLA only-up MCMC kernel with two or more up electrons and no down electron.

``jqmc_mcmc._update_electron_positions_only_up_electron`` is the transition kernel of the
``__only_up_electron`` branch of ``MCMC.run`` (any Hamiltonian with ``tot_dn == 0`` on the
XLA route). Its rank-1 Sherman-Morrison update adds the new row of the moved electron
``i`` to the carried geminal ``G`` and updates the carried inverse ``X = G^-1`` with
``u = e_i``. Every committed fixture that reached this kernel had a single up electron, so
``i`` was always 0 and replacing ``e_i`` by ``e_0`` survived every test (issue #54).

This module builds the committed all-electron N atom (cc-pVDZ, Cartesian) with ``n_dn = 0``
and ``n_up`` in {2, 3, 4}: a single determinant of the first ``n_up`` MOs, and the same
determinant converted to AGP. It runs the kernel for a chain of segments, carrying the
walker, the PRNG key, ``G``, ``X`` and the streaming determinant state between segments as
``MCMC.run`` does. After each segment it compares the carried ``G``, ``X`` and the
log-determinant of ``X`` against the Slater matrix rebuilt from scratch at the current
positions (``compute_MOs(...)(r_up).T``), its numpy inverse and its numpy ``slogdet``, and
checks that the carried ``X`` inverts the carried ``G``. The reference does not use the
Sherman-Morrison update or the streaming caches. The identity residual uses the carried ``G``:
the carried rows are evaluated in the kernel's XLA program and the reference in another, and
XLA:GPU rounds the same fp32 ``ao_eval`` value differently in the two, so
``G_ref X - I`` carries that rounding times ``|X|`` (7.8e-6 on a GPU, issue #546). The chain must
accept moves of every up electron, so an update of the wrong row is not hidden by a chain
that only ever moves electron 0. Full and mixed are judged separately with the active-mode
zone tolerances.
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

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jqmc import determinant as determinant
from jqmc._precision import get_tolerance_min
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_ratio_state
from jqmc.jqmc_mcmc import _update_electron_positions_only_up_electron
from jqmc.molecular_orbital import compute_MOs
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_FIXTURE = Path(__file__).resolve().parents[1] / "trexio_example_files" / "N_ae_ccpvdz_cart.h5"
_DT = 1.2
_EPSILON_AS = 1.0e-6
_NUM_SEGMENTS = 4
_MOVES_PER_SEGMENT = 25
#: Zones crossed by the rebuilt Slater matrix, the carried rank-1 updates and their log-determinant.
_ZONES = ("ao_eval", "mo_eval", "det_eval", "det_ratio")
_CASES = [(2, "sd"), (3, "sd"), (4, "sd"), (2, "agp"), (3, "agp"), (4, "agp")]


def _only_up_hamiltonian(num_up, kind):
    """N atom, no Jastrow, ``num_up`` up electrons on the first ``num_up`` MOs and no down electron."""
    structure_data, _, mos_data, _, _, coulomb_potential_data = read_trexio_file(str(_FIXTURE), store_tuple=True)
    sd = determinant.SD_data.build(
        num_electron_up=num_up, num_electron_dn=0, mos_up=mos_data, mos_dn=mos_data, time_reversal=False
    )
    geminal_data = sd if kind == "sd" else determinant.to_AGP(sd)
    assert (int(geminal_data.num_electron_up), int(geminal_data.num_electron_dn)) == (num_up, 0)
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=None
    )
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data),
    )
    return hamiltonian_data, sd.orb_data_up_spin


@pytest.mark.parametrize("num_up, kind", _CASES, ids=[f"{k}_nup{n}" for n, k in _CASES])
def test_only_up_kernel_multi_electron_matches_rebuilt_slater(num_up, kind):
    """Carried G, G^-1 and ln|det| of the XLA only-up kernel equal a from-scratch Slater rebuild.

    n_dn = 0, n_up in {2, 3, 4}, SD and AGP. Four segments of 25 proposals each, Dt = 1.2,
    PRNGKey(5), start at 0.7 * standard normals (PCG64 seed 3) around the N nucleus. After each
    segment: carried G vs compute_MOs(orb_up, r_up).T; carried X vs numpy inv of that matrix;
    carried G @ X vs the identity; slogdet(X) vs -slogdet(G_ref). Every up electron must have been
    moved by an accepted proposal by the end of the chain.
    """
    hamiltonian_data, orb_up = _only_up_hamiltonian(num_up, kind)
    geminal_data = hamiltonian_data.wavefunction_data.geminal_data
    atol, rtol = get_tolerance_min(_ZONES, "strict")

    rng = np.random.default_rng(3)
    r_up_init = rng.normal(scale=0.7, size=(num_up, 3))
    r_up = jnp.asarray(r_up_init)
    r_dn = jnp.zeros((0, 3), dtype=jnp.float64)
    geminal = jnp.asarray(determinant.compute_geminal_all_elements(geminal_data, r_up, r_dn))
    geminal_inv = jnp.asarray(np.linalg.inv(np.asarray(geminal)))
    det_state = determinant._init_det_ratio_streaming_state(geminal_data, r_up, r_dn)
    jastrow_state = Jastrow_ratio_state(j3_state=None)
    key = random.PRNGKey(5)

    total_accepted = 0
    moved = np.zeros(num_up, dtype=bool)
    for segment in range(_NUM_SEGMENTS):
        r_up_before = np.asarray(r_up)
        (accepted, _rejected, r_up, r_dn, key, geminal_inv, geminal, det_state, jastrow_state) = (
            _update_electron_positions_only_up_electron(
                r_up,
                r_dn,
                key,
                _MOVES_PER_SEGMENT,
                hamiltonian_data,
                _DT,
                _EPSILON_AS,
                geminal_inv,
                geminal,
                det_state,
                jastrow_state,
            )
        )
        total_accepted += int(accepted)
        moved |= np.any(np.asarray(r_up) != r_up_before, axis=1)
        assert np.asarray(r_dn).shape == (0, 3)

        segment_label = str(segment)
        label = f"{kind}/n_up={num_up}/segment={segment_label}"
        g_ref = np.asarray(compute_MOs(orb_up, r_up), dtype=np.float64).T
        assert g_ref.shape == (num_up, num_up)
        g_ref_inv = np.linalg.inv(g_ref)
        sign_ref, logabs_ref = np.linalg.slogdet(g_ref)
        g_carried = np.asarray(geminal, dtype=np.float64)
        x_carried = np.asarray(geminal_inv, dtype=np.float64)
        sign_x, logabs_x = np.linalg.slogdet(x_carried)

        np.testing.assert_allclose(g_carried, g_ref, atol=atol, rtol=rtol, err_msg=f"{label}: carried G")
        np.testing.assert_allclose(x_carried, g_ref_inv, atol=atol, rtol=rtol, err_msg=f"{label}: carried G^-1")
        np.testing.assert_allclose(g_carried @ x_carried, np.eye(num_up), atol=atol, rtol=rtol, err_msg=f"{label}: G X")
        assert sign_x == sign_ref, f"{label}: sign det(X)={sign_x} sign det(G_ref)={sign_ref}"
        np.testing.assert_allclose(-logabs_x, logabs_ref, atol=atol, rtol=rtol, err_msg=f"{label}: ln|det|")

    # Non-vacuity: the chain must have accepted moves of every up electron, including
    # electrons other than index 0.
    assert total_accepted > 0, f"{kind}/n_up={num_up}: no move accepted"
    assert moved.all(), f"{kind}/n_up={num_up}: electrons never moved: {np.flatnonzero(~moved).tolist()}"
