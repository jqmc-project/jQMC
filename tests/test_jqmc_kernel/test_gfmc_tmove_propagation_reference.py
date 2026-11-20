"""Per-hop same-input references for the tmove PROPAGATION path with live FN<0 elements.

The light Li ccECP rows of ``test_jqmc_gfmc.py`` exercise the T-move sign-flip
diagonal S' (``sum max(V_nl, 0)``), but on their visited configurations every
non-local element is non-negative, so the fixed-node off-diagonal channel
``min(V_nl, 0)`` is identically zero there: zeroing it in ``_projection_t_core``
(GFMC_t) or in ``GFMC_n``'s ``_body_step_core`` leaves those rows green.

These nodes run the PRODUCTION XLA drivers with ``non_local_move="tmove"`` on the
committed ``N2_ecp_ccpvtz_cart.h5`` (nitrogen ccECP, five up / five down
electrons, J1+J2), capture every real projection hop and recompute each hop's
candidates, transition CDF and diagonal from its captured configuration and
rotations with full-recompute leaves (``tests/_gfmc_n_same_input.leaf_reference``,
tmove).  Liveness is asserted in-test on the reference: the non-local fixed-node
elements ``min(V_nl, 0)`` are strictly negative at some captured hops, so the
off-diagonal T-move channel carries weight in the walk being compared.
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

import jax
import numpy as np
import pytest

from jqmc._setting import GFMC_MIN_COLLECT_STEPS
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from tests._gfmc_n_same_input import ObserveGFMCN, leaf_reference, ordered_hops, projection_comparisons
from tests.test_jqmc_kernel.test_gfmc_dltmove_propagation_reference import (
    _ALAT,
    _SEED,
    ObserveGFMCTHops,
    _ham,
    _num_kinetic_candidates,
    _tolerance,
)

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


def _assert_fn_offdiagonal_live(reference_candidates, n_kin, tag):
    """Liveness: the T-move fixed-node non-local elements ``min(V_nl, 0)`` are negative somewhere."""
    ecp = np.concatenate([np.asarray(row)[n_kin:] for row in reference_candidates])
    assert ecp.size > 0 and np.all(np.isfinite(ecp)), f"{tag}: no finite non-local candidates"
    n_negative = int(np.count_nonzero(ecp < 0.0))
    fn_weight = float(np.sum(np.abs(np.minimum(ecp, 0.0))))
    assert n_negative > 0 and fn_weight > 0.0, (
        f"{tag}: every non-local FN element min(V_nl, 0) is zero on the captured hops -- the T-move "
        "off-diagonal channel is vacuous on this row"
    )


def test_gfmc_n_tmove_propagation_matches_same_input_reference():
    """GFMC_n tmove: every hop's candidates, CDF, diagonal and weight at its own input, FN<0 live.

    Two walkers, seed 3446, forty intervals of five hops, rotating meshes,
    energy-only run.  Full and mixed each use their own strict/medium zone
    tolerance (the helper shared with the dltmove sibling).
    """
    ham = _ham()
    n_steps, n_hops, n_walkers = 40, 5, 2
    gfmc = GFMC_n(
        hamiltonian_data=ham,
        num_walkers=n_walkers,
        num_mcmc_per_measurement=n_hops,
        num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
        mcmc_seed=_SEED,
        E_scf=-19.0,
        alat=_ALAT,
        random_discretized_mesh=True,
        comput_position_deriv=False,
        non_local_move="tmove",
    )
    observer = ObserveGFMCN(gfmc)
    communicator = gfmc._mpi_comm
    with pytest.MonkeyPatch.context() as patch:
        observer.install(patch)
        patch.setattr(gfmc, "_mpi_comm", communicator)
        gfmc.run(num_mcmc_steps=n_steps)

    atol, rtol = _tolerance()
    projection = projection_comparisons(
        observer.intervals,
        expected_intervals=n_steps,
        expected_walkers=n_walkers,
        expected_hops=n_hops,
        expected_random_mesh=True,
        expected_non_local_move="tmove",
        expected_alat=_ALAT,
        expected_hamiltonian=observer.hamiltonian,
    )
    np.testing.assert_array_equal(projection["exact"][0], projection["exact"][1])
    for name in ("candidates", "cdf", "propagation", "weights", "inverse", "positions"):
        np.testing.assert_allclose(
            np.concatenate([np.ravel(v) for v in projection[name][0]]),
            np.concatenate([np.ravel(v) for v in projection[name][1]]),
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg=f"GFMC_n tmove {name}",
        )

    reference = []
    for interval in observer.intervals:
        for walker in ordered_hops(interval):
            for row, _, _ in walker:
                args = (interval["ham"], row["r_up_carts"], row["r_dn_carts"], row["RT"], row["ecp_RT"], _ALAT)
                reference.append(np.asarray(leaf_reference(*args, non_local_move="tmove")[2]))
    assert len(reference) == n_steps * n_walkers * n_hops
    _assert_fn_offdiagonal_live(reference, _num_kinetic_candidates(ham), "GFMC_n")


def test_gfmc_t_tmove_propagation_matches_same_input_reference():
    """GFMC_t tmove (one walker): every hop's candidates, CDF, off-diagonal sum and diagonal, FN<0 live.

    One walker, seed 3446, tau 0.05, twenty branching intervals, rotating
    meshes, energy only.  The lattice e_L that sets the escape time and the
    weight is ``diagonal + off-diagonal sum``; both summands are compared
    separately (the driver's diagonal is ``e_L - non_diagonal_sum_hamiltonian``)
    because on this walk e_L crosses zero at one hop (|e_L| ~ 7e-3 out of
    summands of order 10-50 Ha), where an absolute comparison of the sum would
    test fp32 cancellation rather than the assembly.  Full and mixed each use
    their own strict/medium zone tolerance.
    """
    ham = _ham()
    n_steps = 20
    gfmc = GFMC_t(
        hamiltonian_data=ham,
        num_walkers=1,
        num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
        mcmc_seed=_SEED,
        tau=0.05,
        alat=_ALAT,
        random_discretized_mesh=True,
        comput_position_deriv=False,
        non_local_move="tmove",
    )
    observer = ObserveGFMCTHops()
    with pytest.MonkeyPatch.context() as patch:
        observer.install(patch)
        gfmc.run(num_mcmc_steps=n_steps)
    jax.effects_barrier()
    hops = observer.hops
    assert len(hops) >= n_steps, f"GFMC_t observer captured {len(hops)} hops for {n_steps} intervals"

    atol, rtol = _tolerance()
    actual_c, ref_c, actual_cdf, ref_cdf, actual_e, ref_e, actual_nd, ref_nd = ([] for _ in range(8))
    actual_pos, ref_pos, exact = [], [], []
    for row in hops:
        args = (ham, row["r_up_carts"], row["r_dn_carts"], row["RT"], row["ecp_RT"], _ALAT)
        mesh_up, mesh_dn, off_diagonal, diagonal, _, probabilities, _ = (
            np.asarray(v) for v in leaf_reference(*args, non_local_move="tmove")
        )
        actual_c.append(np.real(row["p_list"]))
        ref_c.append(off_diagonal)
        actual_cdf.append(row["cdf"])
        ref_cdf.append(np.cumsum(probabilities))
        actual_nd.append(np.real(row["non_diagonal_sum_hamiltonian"]))
        ref_nd.append(off_diagonal.sum())
        actual_e.append(np.real(row["e_L"]) - np.real(row["non_diagonal_sum_hamiltonian"]))
        ref_e.append(diagonal)
        selected = int(row["k"])
        exact.extend(
            (
                selected == int(np.searchsorted(row["cdf"], row["random_value"])),
                0 <= selected < len(mesh_up),
                bool(np.all(np.isfinite(row["cdf"]))),
                bool(np.all(np.diff(row["cdf"]) >= 0)),
            )
        )
        actual_pos.extend((row["next_up"], row["next_dn"]))
        ref_pos.extend((mesh_up[selected], mesh_dn[selected]))

    assert all(exact), "GFMC_t categorical selection does not invert its own CDF"
    for name, a, r in (
        ("candidates", actual_c, ref_c),
        ("cdf", actual_cdf, ref_cdf),
        ("off-diagonal sum", actual_nd, ref_nd),
        ("lattice diagonal", actual_e, ref_e),
        ("selected move", actual_pos, ref_pos),
    ):
        np.testing.assert_allclose(
            np.concatenate([np.ravel(v) for v in a]),
            np.concatenate([np.ravel(v) for v in r]),
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg=f"GFMC_t tmove {name}",
        )
    _assert_fn_offdiagonal_live(ref_c, _num_kinetic_candidates(ham), "GFMC_t")
