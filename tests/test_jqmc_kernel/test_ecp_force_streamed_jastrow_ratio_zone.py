"""Issue #765: the XLA streamed ECP force/stress route keeps its mesh Jastrow ratio in jastrow_ratio.

The precision policy (``jqmc/_precision.py``) assigns the ECP mesh Jastrow ratio
``exp(J(R') - J(R))`` to ``jastrow_ratio`` (float64 in both modes); whole Jastrow
values belong to ``jastrow_eval`` (float32 in mixed). The XLA ECP force / stress
setup ``ecp_force._ecp_nl_common`` serves wavefunctions the rank-one mesh route
rejects (a complex Bloch J3 orbital leaf, tiled orbitals) with a streamed
full-recompute branch. That branch formed the tmove ratio ``c["ratio"]`` and the
dltmove reweight ``jr`` (``ecp_force._dltmove_delta_terms``) from two whole
``compute_Jastrow_part`` values, i.e. in ``jastrow_eval``. #720 had already moved
the rank-one route to the one-move log-ratio ``mesh_state["dJ"]``.

Fixtures: ``twist_sphepgto_j123`` (lithium_k025.h5, bcc Li ccECP,
k = (1/4, 1/4, 1/4), complex sphe-pGTO single determinant, exp J1/J2 and a complex
Bloch pGTO J3) and the same input with a DTN channel added by the reviewed
generated-matrix builder spec ``{"j3": "pgto", "dtn": True}``. Both are periodic
and complex-twist, and the complex Bloch J3 leaf routes both to the streamed
branch. At this twist every zone the ratio crosses is float64 in mixed:
``jastrow_ratio`` (J1/J2/DTN and the J3 contraction), ``ao_bloch_eval`` (the
complex Bloch J3 and determinant AO values), ``mo_eval``, ``det_eval`` and
``det_ratio``. The strict tolerance of those zones is therefore the policy
budget in full and mixed alike. The independent reference is an independently
rebuilt full-precision case of the same serialized input: the fp64 NumPy Jastrow
debug evaluator (``_compute_Jastrow_part_debug(..., zone="jastrow_ratio")``) at
the same ECP mesh configurations and, for the tmove ratio, the full-case
determinant ratio of whole geminal determinants at those configurations.

The comparison is on the Jastrow log-ratio ``log jr`` (and ``log`` of the
Jastrow factor of ``c["ratio"]``), the quantity the zone owns: a whole-value
difference carries the float32 rounding of ``|J|`` itself, while the log-ratio
is small, so an elementwise comparison of the exponentiated factor near one
would hide that rounding inside the relative budget.
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

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc.derivatives import ecp_force
from jqmc.determinant import compute_det_geminal_all_elements
from jqmc.jastrow_factor import _compute_Jastrow_part_debug
from tests._precision_fidelity import mixed_vs_full
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_STEM = "lithium_k025.h5"
_SPECS = {"j123": {"j3": "pgto"}, "j123_dtn": {"j3": "pgto", "dtn": True}}
_SCALE = 0.6
# Every zone the streamed ratio crosses at this twist; all are float64 in both modes.
_ZONES = ("jastrow_ratio", "ao_bloch_eval", "mo_eval", "det_eval", "det_ratio")


def _case(jastrow, seed):
    """Seeded twist case rebuilt from the serialized TREXIO input."""
    ham = _build_ham_for_cell(_STEM, True, _SPECS[jastrow])
    gd = ham.wavefunction_data.geminal_data
    positions = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    n_up, n_dn = int(gd.num_electron_up), int(gd.num_electron_dn)
    r_up = positions[rng.integers(0, len(positions), n_up)] + rng.normal(scale=_SCALE, size=(n_up, 3))
    r_dn = positions[rng.integers(0, len(positions), n_dn)] + rng.normal(scale=_SCALE, size=(n_dn, 3))
    return ham, jnp.asarray(r_up), jnp.asarray(r_dn)


def _streamed_ratios(case):
    """Streamed-route tmove ratio and dltmove Jastrow ratio; in full precision also their fp64 references."""
    ham, r_up, r_dn = case
    common, _, jastrow_ratio, *_ = ecp_force._dltmove_delta_terms(ham, r_up, r_dn, jnp.eye(3), None)
    mesh_up = np.asarray(common["RU"])
    mesh_dn = np.asarray(common["RD"])
    shape = np.shape(jastrow_ratio)
    out = {
        "fast": common["fast"],
        "mesh_up": mesh_up,
        "mesh_dn": mesh_dn,
        "tmove_ratio": np.asarray(common["ratio"]),
        "jastrow_ratio": np.asarray(jastrow_ratio),
    }
    if is_mixed_precision_enabled():
        # The references are taken from the full rebuild only (the NumPy debug
        # evaluator dominates the cost and is policy-independent fp64 anyway).
        return out
    wf = ham.wavefunction_data
    jastrow_data, geminal_data = wf.jastrow_data, wf.geminal_data
    base = _compute_Jastrow_part_debug(jastrow_data, np.asarray(r_up), np.asarray(r_dn), zone="jastrow_ratio")
    moved = np.asarray(
        [
            _compute_Jastrow_part_debug(jastrow_data, up, dn, zone="jastrow_ratio")
            for up, dn in zip(mesh_up, mesh_dn, strict=True)
        ]
    )
    det_base = np.asarray(compute_det_geminal_all_elements(geminal_data, r_up, r_dn))
    det_moved = np.asarray(
        [compute_det_geminal_all_elements(geminal_data, up, dn) for up, dn in zip(mesh_up, mesh_dn, strict=True)]
    )
    out["debug_log_ratio"] = (moved - base).reshape(shape)
    out["det_ratio"] = (det_moved / det_base).reshape(shape)
    return out


@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("jastrow", ["j123", "j123_dtn"])
def test_streamed_ecp_mesh_jastrow_ratio_matches_full_precision_debug_reference(jastrow, seed):
    """Mixed and full streamed-route mesh Jastrow ratios match the fp64 debug reference of a full rebuild.

    lithium_k025.h5 through ``_build_ham_for_cell`` with exp J1/J2 plus a complex
    Bloch pGTO J3 (``j123``) and the same plus DTN (``j123_dtn``); NumPy seed 0,
    up then down electrons each placed at a nucleus drawn by the stream plus a
    normal offset of scale 0.6; identity mesh rotation and the default Nv mesh.
    The streamed route must serve the case. In mixed the actual values come from
    the mixed case and the references from an independently rebuilt full case; the
    mesh configurations must be identical. In full both come from the same full
    case (a consistency check). The dltmove ``log jr`` must equal the debug
    log-ratio, and the tmove ratio divided by the reference determinant ratio must
    have the same log; both at the strict budget of the zones listed in
    ``_ZONES`` (all float64 in both modes). Values must be finite and the
    Jastrow ratio live.
    """
    if is_mixed_precision_enabled():
        result, full = mixed_vs_full(lambda: _case(jastrow, seed), _streamed_ratios)
    else:
        result = full = _streamed_ratios(_case(jastrow, seed))
    assert not result["fast"], "the streamed (non-rank-one) mesh route must serve this complex Bloch J3 case"
    np.testing.assert_array_equal(result["mesh_up"], full["mesh_up"])
    np.testing.assert_array_equal(result["mesh_dn"], full["mesh_dn"])
    reference = full["debug_log_ratio"]
    jastrow_ratio = result["jastrow_ratio"]
    tmove_jastrow = result["tmove_ratio"] / full["det_ratio"]
    for name, values in (
        ("dltmove jr", jastrow_ratio),
        ("tmove ratio", result["tmove_ratio"]),
        ("det ratio", full["det_ratio"]),
    ):
        assert np.all(np.isfinite(values)), f"nonfinite {name}"
    assert np.all(np.isfinite(reference)), "nonfinite debug Jastrow log-ratio"
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    assert np.max(np.abs(reference)) > atol, "inactive Jastrow ratio"
    np.testing.assert_allclose(np.log(jastrow_ratio), reference, atol=atol, rtol=rtol, err_msg="dltmove jr")
    np.testing.assert_allclose(np.log(tmove_jastrow), reference, atol=atol, rtol=rtol, err_msg="tmove ratio")
