"""Adversarial parity gate for projection-scoped LRDMC mesh state.

The production projection carries candidate AO tables, determinant weights,
and Jastrow candidate caches across its hop interval.  This gate advances the
same states and Sherman--Morrison inverse helper used by production for 150
hops.  At every hop it compares the streamed mesh elements with a fresh
candidate evaluation that receives the same carried inverse and determinant
ratio state.  The comparison therefore isolates derived candidate-state drift;
fresh-inverse validity is governed separately in the determinant tests.

Each accepted hop uses the smallest finite determinant-ratio magnitude from a
different electron than the preceding hop.  This deliberately exercises the
``1 / rho`` amplification regime while continuing to visit distinct cached
rows and columns.  The gate requires both routes to remain finite and requires
their individual elements and total negative hopping rates to agree throughout
the chain.

After every Sherman--Morrison update the chain runs the production
per-hop refresh rule of the GFMC drivers (``jqmc_gfmc._interval_inverse_screen``),
which rebuilds the carried inverse from scratch when its probe residual
exceeds ``SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD``; the chain must refresh at
least once. On every hop the existing conditioning screen accepts, the streamed
elements are also compared with the independent from-scratch
``compute_discretized_kinetic_energy`` (no carried inverse, no carried state),
which sees a drifted carried inverse that the same-inverse oracle cannot. That
judgment is made in full precision only; with the fp32
source eps of mixed precision the existing screen accepts no hop of this chain.
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

from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import RTOL_LRDMC_STREAMING_MESH_CHAIN
from jqmc.determinant import (
    _advance_det_ratio_streaming_state,
    _compute_ratio_determinant_from_candidate_mesh_state,
    _compute_u_dn_move_from_det_ratio_state,
    _compute_v_up_move_from_det_ratio_state,
    _init_det_ratio_streaming_state,
    _sherman_morrison_ratio_and_inverse,
    compute_geminal_all_elements,
)
from jqmc.jqmc_gfmc import _interval_inverse_screen
from jqmc.jqmc_mcmc import _geminal_inv_single
from jqmc.wavefunction import (
    _advance_discretized_kinetic_mesh_streaming_state,
    _init_discretized_kinetic_mesh_streaming_state,
    compute_discretized_kinetic_energy,
    compute_discretized_kinetic_energy_fast_update,
    compute_discretized_kinetic_energy_from_streaming_state,
)
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

pytestmark = [pytest.mark.with_chdir]

_HOPS = 150
_ALAT = 0.30
_SEED = 918
# Consumer-vs-oracle agreement bound for the maximum-normalised worst hop.
_TOL = RTOL_LRDMC_STREAMING_MESH_CHAIN
# Exclude exactly-nodal candidates whose ratio is numerically zero; the picker
# wants small-but-finite rho (the amplification event), not a singular move.
_RHO_FLOOR = 1.0e-8


@pytest.mark.light
def test_streaming_mesh_elements_track_oracle_over_adversarial_chain():
    """Near-node mesh state stays finite and does not amplify hop rates.

    The oracle deliberately consumes the same carried inverse and determinant
    ratio state, so this is the near-node *derived-state* claim only.  Fresh-SVD
    inverse equality is asserted separately, and only in its well-conditioned
    domain.
    """
    ham = _build_ham_for_cell("silicon.h5", False, None, "SD")
    wf = ham.wavefunction_data
    gem = wf.geminal_data
    n_up, n_dn = gem.num_electron_up, gem.num_electron_dn

    rng = np.random.RandomState(_SEED)
    # Loose random positions around the atoms: the unequilibrated walker-init
    # regime in which the production event was observed.
    pos = np.asarray(ham.structure_data.positions, dtype=np.float64)
    r_up = jnp.asarray(pos[rng.randint(0, len(pos), n_up)] + rng.normal(scale=0.6, size=(n_up, 3)))
    r_dn = jnp.asarray(pos[rng.randint(0, len(pos), n_dn)] + rng.normal(scale=0.6, size=(n_dn, 3)))

    _, A_inv, _, _ = _geminal_inv_single(gem, None, r_up, r_dn)
    det_state = _init_det_ratio_streaming_state(gem, r_up, r_dn)
    RT = jnp.eye(3)
    mesh_state = _init_discretized_kinetic_mesh_streaming_state(_ALAT, wf, r_up, r_dn, RT)

    K = mesh_state.candidate_up_positions.shape[1]
    n_up_flat = n_up * K
    last_electron = -1
    worst = 0.0
    worst_hop = -1
    worst_rate_divergence = 0.0
    ill_conditioned_hops = 0
    # Independent-oracle judgment: streamed elements vs the from-scratch
    # recompute on every hop the existing conditioning screen accepts.
    worst_independent = 0.0
    worst_independent_hop = -1
    judged_independent = 0
    interval_refreshes = 0
    inverse_source_eps = np.finfo(np.float32 if is_mixed_precision_enabled() else np.float64).eps

    for hop in range(_HOPS):
        G_fresh = np.asarray(compute_geminal_all_elements(gem, r_up, r_dn))
        cond_times_eps = float(np.linalg.cond(G_fresh, p=2) * inverse_source_eps)
        assert np.isfinite(cond_times_eps), f"cond_2(G) * eps went non-finite at hop {hop}"
        if cond_times_eps >= _TOL:
            ill_conditioned_hops += 1

        # Production consumer vs the stateless debug oracle on identical input.
        _, _, elem_stream = compute_discretized_kinetic_energy_from_streaming_state(
            alat=_ALAT,
            wavefunction_data=wf,
            A_old_inv=A_inv,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            mesh_state=mesh_state,
            j3_state=None,
            det_ratio_state=det_state,
        )
        _, _, elem_oracle = compute_discretized_kinetic_energy_fast_update(
            _ALAT,
            wf,
            A_inv,
            r_up,
            r_dn,
            RT,
            det_ratio_state=det_state,
        )
        eo = np.asarray(elem_oracle)
        es = np.asarray(elem_stream)
        assert np.all(np.isfinite(es)), f"streamed elements went non-finite at hop {hop}"
        assert np.all(np.isfinite(eo)), f"oracle elements went non-finite at hop {hop}"
        div = float(np.max(np.abs(es - eo)) / max(np.max(np.abs(eo)), 1e-30))
        if div > worst:
            worst, worst_hop = div, hop
        if cond_times_eps < _TOL:
            # Independent oracle: the full recompute at this configuration uses no
            # carried inverse and no carried state, so a drifted carried A_inv
            # (a stale carried inverse) is visible here, unlike in the same-inverse
            # oracle above.
            _, _, elem_independent = compute_discretized_kinetic_energy(_ALAT, wf, r_up, r_dn, RT)
            ei = np.asarray(elem_independent).ravel()
            assert np.all(np.isfinite(ei)), f"independent oracle elements went non-finite at hop {hop}"
            judged_independent += 1
            div_independent = float(np.max(np.abs(es.ravel() - ei)) / max(np.max(np.abs(ei)), 1e-30))
            if div_independent > worst_independent:
                worst_independent, worst_independent_hop = div_independent, hop

        # The fixed-node off-diagonal rate is formed from the negative mesh
        # elements.  Pin the rate as well as individual elements: a stale state
        # can otherwise produce a finite but enormous total rate, collapse dtau,
        # and recreate the unbounded projection-counter failure.
        rate_stream = float(np.sum(np.abs(np.minimum(es, 0.0))))
        rate_oracle = float(np.sum(np.abs(np.minimum(eo, 0.0))))
        assert np.isfinite(rate_stream) and np.isfinite(rate_oracle)
        rate_divergence = abs(rate_stream - rate_oracle) / max(abs(rate_oracle), 1.0e-30)
        worst_rate_divergence = max(worst_rate_divergence, rate_divergence)

        # Adversarial accepted move: smallest-|rho| candidate of a different
        # electron than the previous hop (ORACLE ratios = ground truth).
        oracle_ratio = np.asarray(
            _compute_ratio_determinant_from_candidate_mesh_state(gem, A_inv, det_state, mesh_state.det_candidate_state)
        )
        absr = np.abs(oracle_ratio)
        flat_electron = np.arange(absr.size) // K
        ok = (absr > _RHO_FLOOR) & (flat_electron != last_electron)
        pick = int(np.where(ok)[0][np.argmin(absr[ok])])
        last_electron = int(flat_electron[pick])
        if pick < n_up_flat:
            moved_up, idx = True, pick // K
            r_up_new = r_up.at[idx].set(mesh_state.candidate_up_positions[idx, pick % K])
            r_dn_new = r_dn
        else:
            moved_up, idx = False, (pick - n_up_flat) // K
            r_up_new = r_up
            r_dn_new = r_dn.at[idx].set(mesh_state.candidate_dn_positions[idx, (pick - n_up_flat) % K])

        # Actual shared production SM update of A_inv.  MCMC, GFMC_n and
        # GFMC_t all consume this helper, so the chained regression cannot
        # silently diverge into a test-local copy of the formula.
        if moved_up:
            v = _compute_v_up_move_from_det_ratio_state(
                geminal_data=gem, state=det_state, moved_index=idx, r_up_carts_proposed=r_up_new
            )[:, None]
            u = jax.nn.one_hot(idx, n_up, dtype=A_inv.dtype)[:, None]
        else:
            u = _compute_u_dn_move_from_det_ratio_state(
                geminal_data=gem, state=det_state, moved_index=idx, r_dn_carts_proposed=r_dn_new
            )[:, None]
            v = jax.nn.one_hot(idx, n_up, dtype=A_inv.dtype)[:, None]
        _, A_inv_new = _sherman_morrison_ratio_and_inverse(A_inv, u, v)

        det_state_new = _advance_det_ratio_streaming_state(
            gem, det_state, jnp.bool_(moved_up), jnp.int64(idx), r_up_new, r_dn_new
        )
        # Production per-hop screen (the helper GFMC_t / GFMC_n run
        # after every hop): residual of the carried inverse against the post-hop
        # caches, rebuilt from scratch when it exceeds the refresh threshold.
        A_inv_new, _, refreshed = _interval_inverse_screen(gem, det_state_new, A_inv_new)
        interval_refreshes += int(refreshed)
        mesh_state = _advance_discretized_kinetic_mesh_streaming_state(
            wavefunction_data=wf,
            state=mesh_state,
            moved_spin_is_up=jnp.bool_(moved_up),
            moved_index=jnp.int64(idx),
            r_up_carts_old=r_up,
            r_dn_carts_old=r_dn,
            r_up_carts_new=r_up_new,
            r_dn_carts_new=r_dn_new,
        )
        r_up, r_dn, A_inv, det_state = r_up_new, r_dn_new, A_inv_new, det_state_new

    assert ill_conditioned_hops > 0, "adversarial chain never reached the declared near-node numerical regime"
    # Refresh witness: the adversarial chain must exercise the refresh.
    assert interval_refreshes > 0, "the per-hop screen never refreshed the carried inverse on the adversarial chain"
    # The adversarial chain is judged against the independent
    # oracle in full precision; in mixed the existing screen (fp32 source eps)
    # accepts no hop of this chain, so the judgment there is empty by the screen.
    if not is_mixed_precision_enabled():
        assert judged_independent > 0, "the conditioning screen accepted no hop for the independent oracle"
    assert worst_independent <= _TOL, (
        f"streamed mesh elements diverged from the independent full recompute at a well-conditioned hop: "
        f"{worst_independent:.3e} at hop {worst_independent_hop} over {judged_independent} judged hops "
        f"(bound {_TOL:.1e}) -- the carried interval inverse drifted from inverse(G)"
    )
    assert worst_rate_divergence <= _TOL, (
        f"streaming state amplified the oracle hopping rate by {worst_rate_divergence:.3e} (bound {_TOL:.1e})"
    )
    assert worst <= _TOL, (
        f"production streamed mesh elements diverged from the debug oracle: "
        f"max divergence {worst:.3e} at hop {worst_hop} (bound {_TOL:.1e}) -- "
        f"a projection-scoped mesh state (det winv / candidate caches) is no "
        f"longer consistent with the full recompute over a production-length "
        f"hop chain; this inflates hopping rates and explodes LRDMC hop "
        f"counts"
    )
