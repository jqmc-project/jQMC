"""Per-hop same-input references for the dltmove PROPAGATION path.

The independent composition oracle in ``test_gfmc_ecp_gaussian_reference.py``
covers ``_compute_V_elements_t`` only (the GFMC_t force/grad path).  The walk
itself is driven by the off-diagonal candidates and the diagonal assembled in
``_projection_t_core`` (GFMC_t) and ``GFMC_n``'s ``_body_step_core``. On
``H_ecp_ccpvdz_cart.h5``, whose only non-local coefficient is 0.0, dltmove and
tmove are bit-identical, so that fixture cannot test dltmove.

These nodes run the PRODUCTION XLA drivers with ``non_local_move="dltmove"`` on
the committed ``N2_ecp_ccpvtz_cart.h5`` (nitrogen ccECP, non-local l=0 channel
with coefficients 34.77692 / 15.2033, measured from the file) and a J1+J2
Jastrow, capture every real projection hop, and recompute at each captured
configuration -- without the production closure, its streaming caches or its
rank-1 kernels -- the candidate weights, the transition CDF and the diagonal
the driver used (``tests/_gfmc_n_same_input.leaf_reference``: full-recompute
discretized kinetic, full non-local ECP quadrature with the determinant-only
amplitude and the explicit ``exp(J(mesh) - J(r))`` factor).  The same captured
inputs are also evaluated with the tmove reference; the driver values must be
far from it (non-vacuity: this fixture discriminates the two Hamiltonians).
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

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import jqmc_gfmc as implementation
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import GFMC_MIN_COLLECT_STEPS
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from tests._gfmc_n_same_input import ObserveGFMCN, leaf_reference, ordered_hops, projection_comparisons
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]

_STEM = "N2_ecp_ccpvtz_cart.h5"
_ALAT = 0.30
_SEED = 3446
_ZONES = ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic")

_HAM_CACHE = {}


def _ham():
    if _STEM not in _HAM_CACHE:
        ham = _build_ham_for_cell(_STEM, True, None, "SD")
        ham.sanity_check()
        _HAM_CACHE[_STEM] = ham
    return _HAM_CACHE[_STEM]


def _tolerance():
    return get_tolerance_min(_ZONES, "medium" if is_mixed_precision_enabled() else "strict")


def _num_kinetic_candidates(ham):
    geminal = ham.wavefunction_data.geminal_data
    return 6 * (int(geminal.num_electron_up) + int(geminal.num_electron_dn))


def _assert_live_and_discriminating(actual_candidates, actual_diagonal, dlt_reference, tmove_reference, n_kin, atol, rtol, tag):
    """Liveness: the non-local legs are non-zero; power: dltmove is not tmove here.

    ``*_reference`` are ``(off_diagonal, diagonal)`` lists from ``leaf_reference``.
    The Jastrow factor is positive, so the fixed-node off-diagonals
    ``min(V det, 0) exp(dJ)`` (dltmove) and ``min(V det exp(dJ), 0)`` (tmove) are
    the same numbers; the two moves differ in the sign-flip diagonal
    ``sum max(V det, 0)`` versus ``sum max(V det exp(dJ), 0)``.  The separation
    is therefore asserted on the diagonal that drives the weights / escape time.
    """
    ecp_actual = np.concatenate([np.asarray(row)[n_kin:] for row in actual_candidates])
    assert ecp_actual.size > 0 and np.all(np.isfinite(ecp_actual)), f"{tag}: no finite non-local candidates"
    assert np.max(np.abs(ecp_actual)) > 0.0, f"{tag}: every non-local off-diagonal is zero -- dltmove is vacuous here"
    diag_actual = np.asarray(actual_diagonal, dtype=float)
    diag_dlt = np.asarray([d for _, d in dlt_reference], dtype=float)
    diag_tmove = np.asarray([d for _, d in tmove_reference], dtype=float)
    assert not np.allclose(diag_dlt, diag_tmove, atol=atol, rtol=rtol, equal_nan=False), (
        f"{tag}: the dltmove and tmove diagonals coincide at every captured hop -- the fixture cannot "
        "discriminate the two non-local moves"
    )
    assert not np.allclose(diag_actual, diag_tmove, atol=atol, rtol=rtol, equal_nan=False), (
        f"{tag}: the driver's dltmove diagonal equals the tmove reference -- dltmove fell back to tmove"
    )


def test_gfmc_n_dltmove_propagation_matches_same_input_reference():
    """GFMC_n dltmove: every hop's candidates, CDF, diagonal and weight at its own input.

    Two walkers, seed 3446, forty intervals of five hops, rotating meshes,
    energy-only run (no force collection).  The observer from
    ``tests/_gfmc_n_same_input.py`` records every real hop of the production
    ``_body_step_core``; ``projection_comparisons`` recomputes each hop from its
    captured configuration and rotations with full-recompute leaves.  Full and
    mixed each use their own strict/medium zone tolerance.
    """
    ham = _ham()
    n_steps, n_hops, n_walkers = 40, 5, 2
    settings = dict(
        hamiltonian_data=ham,
        num_walkers=n_walkers,
        num_mcmc_per_measurement=n_hops,
        num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
        mcmc_seed=_SEED,
        E_scf=-19.0,
        alat=_ALAT,
        random_discretized_mesh=True,
        comput_position_deriv=False,
        non_local_move="dltmove",
    )
    gfmc = GFMC_n(**settings)
    observer = ObserveGFMCN(gfmc)
    communicator = gfmc._mpi_comm
    with pytest.MonkeyPatch.context() as patch:
        observer.install(patch)
        # Energy-only run: the branching-row observer reads force locals that an
        # energy-only run never binds, so keep the real communicator.  Only the
        # hop/interval observations are used below.
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
            err_msg=f"GFMC_n dltmove {name}",
        )

    actual, actual_diagonal, dlt_ref, tmove_ref = [], [], [], []
    for interval in observer.intervals:
        for walker in ordered_hops(interval):
            for row, _, _ in walker:
                args = (interval["ham"], row["r_up_carts"], row["r_dn_carts"], row["RT"], row["ecp_RT"], _ALAT)
                actual.append(row["p_list"])
                actual_diagonal.append(np.real(row["diagonal_sum_hamiltonian"]))
                for store, move in ((dlt_ref, "dltmove"), (tmove_ref, "tmove")):
                    leaves = leaf_reference(*args, non_local_move=move)
                    store.append((np.asarray(leaves[2]), float(leaves[3])))
    assert len(actual) == n_steps * n_walkers * n_hops
    _assert_live_and_discriminating(
        actual, actual_diagonal, dlt_ref, tmove_ref, _num_kinetic_candidates(ham), atol, rtol, "GFMC_n"
    )


_T_FIELDS = (
    "r_up_carts",
    "r_dn_carts",
    "RT",
    "ecp_RT",
    "p_list",
    "e_L",
    "non_diagonal_sum_hamiltonian",
    "cdf",
    "random_value",
    "k",
    "next_up",
    "next_dn",
)


class ObserveGFMCTHops:
    """Record every real ``_projection_t_core`` hop of a production GFMC_t run.

    Wraps the actual ``jnp.searchsorted`` call of the categorical move (no AST
    copy of production); the captured values are the ones the driver used.
    """

    def __init__(self):
        self.hops = []

    def _receive(self, *values):
        self.hops.append({name: np.array(value, copy=True) for name, value in zip(_T_FIELDS, values, strict=True)})

    def install(self, monkeypatch):
        original = jnp.searchsorted
        observer = self
        core = implementation._projection_t_core.__code__

        def searchsorted(cdf, value, *args, **kwargs):
            selected = original(cdf, value, *args, **kwargs)
            frame = sys._getframe(1)
            if frame.f_code is core:
                local = frame.f_locals
                data = dict(
                    local,
                    cdf=cdf,
                    random_value=value,
                    k=selected,
                    next_up=local["non_diagonal_move_mesh_r_up_carts"][selected],
                    next_dn=local["non_diagonal_move_mesh_r_dn_carts"][selected],
                )
                jax.debug.callback(observer._receive, *(data[name] for name in _T_FIELDS))
            return selected

        monkeypatch.setattr(jnp, "searchsorted", searchsorted)


def test_gfmc_t_dltmove_propagation_matches_same_input_reference():
    """GFMC_t dltmove (one walker): every hop's candidates, CDF, diagonal and e_L at its own input.

    One walker (the multi-walker GFMC_t key contract is a separate item),
    seed 3446, tau 0.05, twenty branching intervals, rotating meshes, energy only.
    At every captured hop the off-diagonal candidates, the transition CDF, the
    off-diagonal sum and the lattice e_L (= diagonal + off-diagonal sum, which
    sets both the escape time and the weight) are recomputed from the captured
    configuration and rotations with full-recompute leaves, and the selected
    move is checked against the reference mesh.  Full and mixed each use their
    own strict/medium zone tolerance.
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
        non_local_move="dltmove",
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
    actual_pos, ref_pos, actual_diagonal, dlt_ref, tmove_ref, exact = [], [], [], [], [], []
    for row in hops:
        args = (ham, row["r_up_carts"], row["r_dn_carts"], row["RT"], row["ecp_RT"], _ALAT)
        mesh_up, mesh_dn, off_diagonal, diagonal, _, probabilities, _ = (
            np.asarray(v) for v in leaf_reference(*args, non_local_move="dltmove")
        )
        tmove = leaf_reference(*args, non_local_move="tmove")
        actual_c.append(np.real(row["p_list"]))
        ref_c.append(off_diagonal)
        actual_cdf.append(row["cdf"])
        ref_cdf.append(np.cumsum(probabilities))
        actual_nd.append(np.real(row["non_diagonal_sum_hamiltonian"]))
        ref_nd.append(off_diagonal.sum())
        actual_e.append(np.real(row["e_L"]))
        ref_e.append(diagonal + off_diagonal.sum())
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
        actual_diagonal.append(np.real(row["e_L"]) - np.real(row["non_diagonal_sum_hamiltonian"]))
        dlt_ref.append((off_diagonal, float(diagonal)))
        tmove_ref.append((np.asarray(tmove[2]), float(tmove[3])))

    assert all(exact), "GFMC_t categorical selection does not invert its own CDF"
    for name, a, r in (
        ("candidates", actual_c, ref_c),
        ("cdf", actual_cdf, ref_cdf),
        ("off-diagonal sum", actual_nd, ref_nd),
        ("lattice e_L", actual_e, ref_e),
        ("selected move", actual_pos, ref_pos),
    ):
        np.testing.assert_allclose(
            np.concatenate([np.ravel(v) for v in a]),
            np.concatenate([np.ravel(v) for v in r]),
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg=f"GFMC_t dltmove {name}",
        )
    _assert_live_and_discriminating(
        actual_c, actual_diagonal, dlt_ref, tmove_ref, _num_kinetic_candidates(ham), atol, rtol, "GFMC_t"
    )
