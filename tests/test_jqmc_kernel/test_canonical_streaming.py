"""Canonical fast/streaming kinetics with an actually carried SM inverse."""

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
import jax.numpy as jnp
import numpy as np
import pytest

from gdd.engine.canonical_chain import CanonicalStreamingCapture
from jqmc import determinant, wavefunction
from jqmc._precision import get_tolerance_min
from tests.canonical_wavefunction_fixtures import CANONICAL_PATTERN_IDS, build_canonical_hamiltonian


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@jax.jit
def _full_kinetic(wf, up, down):
    return wavefunction.compute_kinetic_energy_all_elements(wf, up, down)


def _geminal_from_cached_state(gem, state):
    """Assemble G from the consumed cache without re-evaluating orbitals.

    NumPy contracts the actual paired-down cache; the excess-up columns use
    the public geminal representation. Transpose is deliberately unconjugated.
    No production geminal-assembly or inverse-update helper is reused.
    """
    up = np.asarray(state.ao_up)
    paired_down = np.asarray(state.paired_dn)
    if isinstance(gem, determinant.SD_data):
        paired = up[: gem.num_occ_dn].T @ paired_down
        unpaired = up[gem.num_occ_dn : gem.num_occ_up].T
    else:
        paired = up.T @ paired_down
        unpaired_coefficients = np.asarray(gem.lambda_matrix)[:, gem.orb_num_dn :]
        unpaired = up.T @ unpaired_coefficients
    return np.concatenate((paired, unpaired), axis=1)


@pytest.mark.parametrize("pattern_id", CANONICAL_PATTERN_IDS)
def test_canonical_streaming_kinetic_carries_sm_inverse(pattern_id):
    """Fast and streaming kinetic follow two electron sweeps with a carried inverse.

    Input: the synthetic parameter-set Hamiltonian for each canonical
    WF-PATTERN cases, not a TREXIO artifact. PCG64(seed=864) draws the initial
    up/down positions with normal scale 0.6, then move vectors with scale 0.04.
    Two sweeps visit every up electron followed by every down electron; an
    empty down sector has no down moves. Every proposed move is taken, without
    rejection, conditioning screen, inverse refresh, or fallback.

    Starting with one LU inverse, production rank-1 Sherman-Morrison updates
    carry it into both the fast evaluation and the incremental kinetic state.
    At every hop the carried inverse equals an independent fresh inverse under
    the strict tier of the ao_eval and det_eval zones: the carried inverse is
    built from single-electron AO rows and the fresh one from the full-width
    geminal, and AO values are reproducible across batch widths only to the
    ao_eval zone (fp32 in mixed). In full
    precision this resolves to the unchanged det_eval strict pair. Both
    per-electron kinetic vectors equal the full recomputation under the
    existing fast/debug strict zone policy. cond(G)
    is measured and finite, without claiming ill-conditioned regime coverage.
    Full and mixed are checked separately. This is route parity, not a new AD
    or independent-physics oracle; the canonical Psi-Hessian test owns that.

    Two additional comparisons retain every hop in fixed-shape histories:
    cached G versus fresh G under ao_eval/det_eval strict, and the carried
    inverse versus a NumPy inverse of the same cached G under det_eval strict.
    These supplement, never replace, the existing whole-route comparisons.
    All reference arrays retain their input dtype. Neither the reference G
    nor its inverse is installed into the production state.
    """
    ham = build_canonical_hamiltonian(pattern_id)
    wf = ham.wavefunction_data
    gem = wf.geminal_data
    n_up, n_down = gem.num_electron_up, gem.num_electron_dn
    rng = np.random.default_rng(864)
    r_up = jnp.asarray(rng.normal(scale=0.6, size=(n_up, 3)))
    r_dn = jnp.asarray(rng.normal(scale=0.6, size=(n_down, 3)))
    capture = CanonicalStreamingCapture(ham)
    capture.initialize(wf, r_up, r_dn)
    moves = [(spin, index) for _ in range(2) for spin, count in ((True, n_up), (False, n_down)) for index in range(count)]
    observed_moves = []
    cached_g_history, fresh_g_history = [], []
    carried_inverse_history, cached_inverse_history = [], []
    inverse_atol, inverse_rtol = get_tolerance_min(("ao_eval", "det_eval"), "strict")
    same_g_atol, same_g_rtol = get_tolerance_min(("det_eval",), "strict")
    kinetic_atol, kinetic_rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_ratio", "wf_kinetic"), "strict"
    )

    for moved_up, index in moves:
        delta = jnp.asarray(rng.normal(scale=0.04, size=3))
        r_up_new, r_dn_new = (r_up.at[index].add(delta), r_dn) if moved_up else (r_up, r_dn.at[index].add(delta))
        rho, inverse = capture.advance(wf, moved_up, index, r_up_new, r_dn_new)
        assert np.isfinite(rho) and abs(rho) > 0
        r_up, r_dn = r_up_new, r_dn_new

        fresh_g = np.asarray(determinant.compute_geminal_all_elements(gem, r_up, r_dn))
        cond_g = float(np.linalg.cond(fresh_g))
        assert np.isfinite(cond_g)
        fresh_inverse = np.linalg.inv(fresh_g)
        np.testing.assert_allclose(
            inverse, fresh_inverse, atol=inverse_atol, rtol=inverse_rtol, equal_nan=False, err_msg="canonical carried inverse"
        )

        # The fresh inverse is only an oracle: never pass it to either consumer.
        fast, streaming = capture.evaluate(wf, r_up, r_dn, inverse)
        reference = np.concatenate(_full_kinetic(wf, r_up, r_dn))
        assert fast.shape == streaming.shape == reference.shape == (n_up + n_down,)
        assert np.all(np.isfinite(fast))
        assert np.all(np.isfinite(streaming))
        assert np.all(np.isfinite(reference))
        np.testing.assert_allclose(
            fast,
            reference,
            atol=kinetic_atol,
            rtol=kinetic_rtol,
            equal_nan=False,
            err_msg="canonical carried-inverse fast kinetic",
        )
        np.testing.assert_allclose(
            streaming, reference, atol=kinetic_atol, rtol=kinetic_rtol, equal_nan=False, err_msg="canonical streaming kinetic"
        )
        cached_g = _geminal_from_cached_state(gem, capture._det_state)
        cached_inverse = np.linalg.inv(cached_g)
        carried = np.asarray(inverse)
        assert cached_g.dtype == fresh_g.dtype == carried.dtype == cached_inverse.dtype
        cached_g_history.append(cached_g)
        fresh_g_history.append(fresh_g)
        carried_inverse_history.append(carried)
        cached_inverse_history.append(cached_inverse)
        observed_moves.append((moved_up, index))

    assert observed_moves == moves
    assert len(observed_moves) == 2 * (n_up + n_down)
    cached_g_history = np.stack(cached_g_history)
    fresh_g_history = np.stack(fresh_g_history)
    carried_inverse_history = np.stack(carried_inverse_history)
    cached_inverse_history = np.stack(cached_inverse_history)
    for history in (cached_g_history, fresh_g_history, carried_inverse_history, cached_inverse_history):
        assert history.shape == (len(moves), n_up, n_up)
        assert np.isfinite(history).all()
    np.testing.assert_allclose(
        cached_g_history,
        fresh_g_history,
        atol=inverse_atol,
        rtol=inverse_rtol,
        equal_nan=False,
        err_msg="canonical all-hop cached geminal",
    )
    np.testing.assert_allclose(
        carried_inverse_history,
        cached_inverse_history,
        atol=same_g_atol,
        rtol=same_g_rtol,
        equal_nan=False,
        err_msg="canonical all-hop same-geminal inverse",
    )
    jax.clear_caches()
