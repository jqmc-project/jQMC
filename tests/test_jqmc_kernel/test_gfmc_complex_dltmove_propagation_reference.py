"""Per-hop same-input reference for the complex-twist GFMC_n dltmove PROPAGATION path.

``test_route_parity_complex_gfmc_values_and_dla_separation[n]`` compares the
native and XLA complex-twist GFMC_n drivers end to end.  On ``lithium_k025`` the
determinant-locality-approximation (DLA) factor only rescales small fixed-node
non-local elements, so a 5 percent error in the XLA GFMC_n body's DLA Jastrow
ratio moves the stored branching weights by ~3e-4 -- visible at the full-precision
backend-parity pin but inside the mixed-precision one.

This node checks the same XLA production path hop by hop instead: it runs the
PRODUCTION ``GFMC_n`` with ``non_local_move="dltmove"`` on the complex twist
fixture ``twist_sphepgto_j12`` (``lithium_k025.h5`` + J1/J2), captures every real
projection hop with ``tests/_gfmc_n_same_input.ObserveGFMCN`` and recomputes, at
each captured configuration and rotation, the candidate weights, the CDF, the
diagonal and the branching factor with full-recompute leaves
(``leaf_reference``: explicit ``exp(J(mesh) - J(r))`` DLA factor).  A wrong DLA
factor changes every live non-local candidate by the same relative amount, which
the per-candidate comparison sees at either precision's own zone tolerance.
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

import numpy as np
import pytest

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import GFMC_MIN_COLLECT_STEPS
from jqmc.jqmc_gfmc import GFMC_n
from tests._gfmc_n_same_input import ObserveGFMCN, leaf_reference, ordered_hops, projection_comparisons
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]

_PATTERN = "twist_sphepgto_j12"
_ALAT = 0.30
_SEED = 34356
_ZONES = ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic")


def _tolerance():
    return get_tolerance_min(_ZONES, "medium" if is_mixed_precision_enabled() else "strict")


def test_gfmc_n_complex_twist_dltmove_propagation_matches_same_input_reference():
    """Complex-twist GFMC_n dltmove: every hop's candidates, CDF, diagonal and weight at its own input.

    One walker, seed 34356, twenty intervals of three hops, rotating meshes,
    energy only.  Liveness: the non-local DLA candidates are strictly negative at
    some captured hop, and the dltmove reference diagonal differs from the tmove
    one (the fixture discriminates the two moves).  Full and mixed each use their
    own strict/medium zone tolerance.
    """
    ham = rs_ham(_PATTERN)
    n_steps, n_hops, n_walkers = 20, 3, 1
    gfmc = GFMC_n(
        hamiltonian_data=ham,
        num_walkers=n_walkers,
        num_mcmc_per_measurement=n_hops,
        num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
        mcmc_seed=_SEED,
        E_scf=-1.0,
        alat=_ALAT,
        random_discretized_mesh=True,
        comput_position_deriv=False,
        non_local_move="dltmove",
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
        expected_non_local_move="dltmove",
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
            err_msg=f"complex GFMC_n dltmove {name}",
        )

    geminal = ham.wavefunction_data.geminal_data
    n_kin = 6 * (int(geminal.num_electron_up) + int(geminal.num_electron_dn))
    dlt_ecp, dlt_diagonal, tmove_diagonal = [], [], []
    for interval in observer.intervals:
        for walker in ordered_hops(interval):
            for row, _, _ in walker:
                args = (interval["ham"], row["r_up_carts"], row["r_dn_carts"], row["RT"], row["ecp_RT"], _ALAT)
                dlt = leaf_reference(*args, non_local_move="dltmove")
                dlt_ecp.append(np.asarray(dlt[2])[n_kin:])
                dlt_diagonal.append(float(dlt[3]))
                tmove_diagonal.append(float(leaf_reference(*args, non_local_move="tmove")[3]))
    assert len(dlt_ecp) == n_steps * n_walkers * n_hops
    dlt_ecp = np.concatenate(dlt_ecp)
    assert dlt_ecp.size > 0 and np.all(np.isfinite(dlt_ecp)), "no finite non-local DLA candidates"
    # Liveness of the DLA-scaled channel: fixed-node non-local candidates
    # ``min(V det, 0) exp(dJ)`` are strictly negative at some captured hop.
    assert np.count_nonzero(dlt_ecp < 0.0) > 0, "every non-local DLA candidate is zero -- the DLA factor is vacuous here"
    # The Jastrow factor is positive, so the fixed-node off-diagonals of dltmove
    # and tmove are the same numbers; the moves differ in the sign-flip diagonal
    # ``sum max(V det, 0)`` versus ``sum max(V det exp(dJ), 0)`` (see the real
    # N2 sibling in test_gfmc_dltmove_propagation_reference.py).  On this Li
    # fixture the sign-flip part is small next to the ~67 Ha diagonal, so the
    # power check is on the absolute separation, at the comparison's own
    # absolute resolution ``atol``.
    separation = np.max(np.abs(np.asarray(dlt_diagonal) - np.asarray(tmove_diagonal)))
    assert separation > atol, (
        f"the dltmove and tmove diagonals coincide at every captured hop (max separation {separation:.3e} <= atol "
        f"{atol:.1e}) -- the fixture cannot discriminate the moves"
    )
