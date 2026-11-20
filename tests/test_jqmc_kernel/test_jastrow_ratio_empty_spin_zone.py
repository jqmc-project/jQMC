"""Issue #775: the shared one-move Jastrow ratio kernels keep an empty spin channel in jastrow_ratio.

The precision policy (``jqmc/_precision.py``) assigns the one-move Jastrow
ratio ``exp(J(R') - J(R))`` to ``jastrow_ratio`` (float64 in both modes);
whole Jastrow values belong to ``jastrow_eval`` (float32 in mixed). When one
spin channel was empty, ``_compute_ratio_Jastrow_part_rank1_update`` (the
tmove / dltmove / MCMC kernel) and ``_compute_ratio_Jastrow_part_split_spin``
(the ECP e_L kernel, which delegated to it) formed the ratio from two whole
``compute_Jastrow_part`` values unless every channel had a candidate cache,
so in mixed the log-ratio carried the float32 rounding of ``|J|`` itself.

Fixtures: the committed N2 ccECP Cartesian input (10 valence electrons) with
the reviewed ``_build_ham_ecp(..., "j12")`` exp J1/J2 Jastrow, and the reviewed
``_build_ham_ecp_dtn`` DTN channel combined with that J1 (``j1_dtn``). Neither
channel reads AO values, so the whole ratio path lies in ``jastrow_ratio`` and
the zone's strict tolerance is the policy budget in full and mixed alike. All
ten electrons are placed in one spin channel (``empty`` names the empty one).
The candidate configurations move one electron each (6 seeded displacements per
electron). The independent reference is the fp64 NumPy Jastrow debug evaluator
(``_compute_Jastrow_part_debug(..., zone="jastrow_ratio")``) applied to the same
configurations of an independently rebuilt full-precision case. The comparison
is on the log-ratio, the quantity the zone owns.

The rank-one kernel's uncached molecular J2 branch
reduces pair distances in Gram form (``|a|^2 + |b|^2 - 2 a.b``), so its
same-spin sums must exclude the moved electron's self pair, whose Gram
distance is a ``sqrt(eps64)``-sized artefact instead of zero (a 1.49e-8
log-ratio error in both modes, for two populated spins as well). The rank-one
J1/J2 test therefore also covers small moves (step scale 0.02) and a two-spin
configuration (``empty = "none"``: the first five electrons up, the rest down).
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

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_dtype_np, get_tolerance_min, is_mixed_precision_enabled
from jqmc.jastrow_factor import (
    Jastrow_data,
    _compute_Jastrow_part_debug,
    _compute_ratio_Jastrow_part_rank1_update,
    _compute_ratio_Jastrow_part_split_spin,
    _init_jastrow_dtn_ratio_state,
)
from tests._precision_fidelity import mixed_vs_full
from tests.test_jqmc_kernel.test_gfmc_force_analytic import _build_ham_ecp, _build_ham_ecp_dtn

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_STEM = "N2_ecp_ccpvtz_cart.h5"
_N_ELECTRONS = 10
_N_SHIFTS = 6
_SCALE = 0.9
_STEP = 0.4


def _jastrow(kind):
    """Reviewed N2 ccECP Jastrow: exp J1/J2 (``j12``) or exp J1 plus DTN (``j1_dtn``)."""
    j12 = _build_ham_ecp(_STEM, "j12").wavefunction_data.jastrow_data
    if kind == "j12":
        return j12
    dtn = _build_ham_ecp_dtn(_STEM).wavefunction_data.jastrow_data.jastrow_dtn_data
    return Jastrow_data(
        jastrow_one_body_data=j12.jastrow_one_body_data,
        jastrow_two_body_data=None,
        jastrow_three_body_data=None,
        jastrow_dtn_data=dtn,
    )


def _case(kind, empty, seed, step=_STEP):
    """Seeded configuration and its one-move candidates as up / dn blocks.

    ``empty`` names the empty spin channel (``"dn"`` / ``"up"``), or ``"none"``
    for two spins (the first ``_N_ELECTRONS // 2`` electrons up). Candidates move
    each electron in order ``_N_SHIFTS`` times by a normal step of scale ``step``;
    the up block holds the up-electron moves and the dn block the dn-electron moves.
    """
    jastrow_data = _jastrow(kind)
    positions = np.asarray(jastrow_data.jastrow_one_body_data.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    r = positions[rng.integers(0, len(positions), _N_ELECTRONS)] + rng.normal(scale=_SCALE, size=(_N_ELECTRONS, 3))
    n_up = {"dn": _N_ELECTRONS, "up": 0, "none": _N_ELECTRONS // 2}[empty]
    up_block, dn_block = [], []
    for i in range(_N_ELECTRONS):
        for _ in range(_N_SHIFTS):
            moved = r.copy()
            moved[i] += rng.normal(scale=step, size=3)
            (up_block if i < n_up else dn_block).append(moved[:n_up] if i < n_up else moved[n_up:])
    n_dn = _N_ELECTRONS - n_up
    up_block = np.asarray(up_block, float).reshape(n_up * _N_SHIFTS, n_up, 3)
    dn_block = np.asarray(dn_block, float).reshape(n_dn * _N_SHIFTS, n_dn, 3)
    return jastrow_data, r[:n_up], r[n_up:], up_block, dn_block


def _log_ratios(case):
    """Production one-move log-ratios; in full precision also the fp64 debug reference."""
    jastrow_data, r_up, r_dn, up_block, dn_block = case
    g_up, g_dn = up_block.shape[0], dn_block.shape[0]
    # Combined candidate configurations of the rank-one kernel: up block then dn block.
    mesh_up = np.concatenate([up_block, np.broadcast_to(r_up, (g_dn, *r_up.shape))])
    mesh_dn = np.concatenate([np.broadcast_to(r_dn, (g_up, *r_dn.shape)), dn_block])
    up, dn = jnp.asarray(r_up), jnp.asarray(r_dn)
    split = _compute_ratio_Jastrow_part_split_spin(jastrow_data, up, dn, jnp.asarray(up_block), jnp.asarray(dn_block))
    out = {"mesh_up": mesh_up, "mesh_dn": mesh_dn, "split_spin": split}
    out["rank1"] = _compute_ratio_Jastrow_part_rank1_update(jastrow_data, up, dn, jnp.asarray(mesh_up), jnp.asarray(mesh_dn))
    if jastrow_data.jastrow_dtn_data is not None:
        out["rank1_dtn_state"] = _compute_ratio_Jastrow_part_rank1_update(
            jastrow_data,
            up,
            dn,
            jnp.asarray(mesh_up),
            jnp.asarray(mesh_dn),
            dtn_state=_init_jastrow_dtn_ratio_state(jastrow_data.jastrow_dtn_data, up, dn),
        )
    if not is_mixed_precision_enabled():
        base = _compute_Jastrow_part_debug(jastrow_data, r_up, r_dn, zone="jastrow_ratio")
        moved = np.asarray(
            [
                _compute_Jastrow_part_debug(jastrow_data, u, d, zone="jastrow_ratio")
                for u, d in zip(mesh_up, mesh_dn, strict=True)
            ]
        )
        out["debug_log_ratio"] = moved - base
    return out


def _compared(kind, empty, step=_STEP):
    """Mixed (or full) result, full result, the fp64 debug log-ratio and the strict budget."""
    if is_mixed_precision_enabled():
        result, full = mixed_vs_full(lambda: _case(kind, empty, 0, step), _log_ratios)
    else:
        result = full = _log_ratios(_case(kind, empty, 0, step))
    np.testing.assert_array_equal(result["mesh_up"], full["mesh_up"])
    np.testing.assert_array_equal(result["mesh_dn"], full["mesh_dn"])
    reference = full["debug_log_ratio"]
    assert np.all(np.isfinite(reference)), "nonfinite debug Jastrow log-ratio"
    atol, rtol = get_tolerance_min(("jastrow_ratio",), "strict")
    assert np.max(np.abs(reference)) > atol, "inactive Jastrow ratio"
    return result, reference, atol, rtol


def _zone_ratio(result, name):
    """One kernel's ratio, required to carry the jastrow_ratio dtype and be finite."""
    ratio = np.asarray(result[name])
    assert ratio.dtype == get_dtype_np("jastrow_ratio"), f"{name} ratio left the jastrow_ratio zone: {ratio.dtype}"
    assert np.all(np.isfinite(ratio)), f"nonfinite {name} Jastrow ratio"
    return ratio


@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("empty", ["dn", "up"])
@pytest.mark.parametrize("kind", ["j12", "j1_dtn"])
def test_empty_spin_split_spin_jastrow_ratio_matches_full_precision_debug_reference(kind, empty):
    """Mixed and full empty-spin split-spin (ECP e_L) Jastrow ratios match the fp64 debug reference.

    N2_ecp_ccpvtz_cart.h5 with exp J1/J2 (``j12``) or exp J1 plus the reviewed
    DTN (``j1_dtn``); NumPy seed 0: ten electrons, each at a nucleus drawn by the
    stream plus a normal offset of scale 0.9, all in one spin channel (``empty``
    names the other); then, per electron in order, six candidates displaced by a
    normal step of scale 0.4. In mixed the actual values come from the mixed case
    and the reference from an independently rebuilt full case; the configurations
    must be identical. In full both come from the same full case (a consistency
    check). The ratio carries the ``jastrow_ratio`` dtype, and its log equals the
    debug log-ratio at the strict ``jastrow_ratio`` budget; values are finite and
    the ratio live.
    """
    result, reference, atol, rtol = _compared(kind, empty)
    split = _zone_ratio(result, "split_spin")
    np.testing.assert_allclose(np.log(split), reference, atol=atol, rtol=rtol, err_msg="split_spin")


@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("empty", ["dn", "up"])
def test_empty_spin_rank1_jastrow_ratio_matches_full_precision_debug_reference(empty):
    """Mixed and full empty-spin rank-one Jastrow ratios match the fp64 debug reference.

    Same ``j1_dtn`` case and candidates as the split-spin test. The rank-one
    kernel (tmove / dltmove / MCMC) is evaluated without and with the MCMC DTN
    ratio state built at the base configuration. Same reference, configuration
    identity, dtype, finiteness, liveness and strict ``jastrow_ratio`` budget.
    """
    result, reference, atol, rtol = _compared("j1_dtn", empty)
    stateless = _zone_ratio(result, "rank1")
    with_state = _zone_ratio(result, "rank1_dtn_state")
    np.testing.assert_allclose(np.log(stateless), reference, atol=atol, rtol=rtol, err_msg="rank1")
    np.testing.assert_allclose(np.log(with_state), reference, atol=atol, rtol=rtol, err_msg="rank1 dtn_state")


@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("step", [0.4, 0.02])
@pytest.mark.parametrize("empty", ["dn", "up", "none"])
def test_rank1_j12_jastrow_ratio_matches_full_precision_debug_reference(empty, step):
    """Mixed and full rank-one J1/J2 Jastrow ratios match the fp64 debug reference.

    The ``j12`` case of the split-spin test, with the candidate step scale 0.4
    or 0.02 (small moves, where a Gram-form self-pair residue in the uncached
    J2 branch would exceed the budget), for both empty-spin orientations
    and for two spins (``empty = "none"``: five up, five down, each spin's moves
    in its own block). The rank-one kernel (tmove / dltmove / MCMC, no candidate
    caches) and the split-spin kernel are checked against the same reference,
    with the same configuration identity, dtype, finiteness, liveness and strict
    ``jastrow_ratio`` budget.
    """
    result, reference, atol, rtol = _compared("j12", empty, step)
    rank1 = _zone_ratio(result, "rank1")
    split = _zone_ratio(result, "split_spin")
    np.testing.assert_allclose(np.log(rank1), reference, atol=atol, rtol=rtol, err_msg="rank1 j12")
    np.testing.assert_allclose(np.log(split), reference, atol=atol, rtol=rtol, err_msg="split_spin j12")
