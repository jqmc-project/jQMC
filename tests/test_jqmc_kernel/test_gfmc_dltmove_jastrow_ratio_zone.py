"""Issue #720: the dltmove ECP correction's Jastrow ratio stays in the fp64 jastrow_ratio zone.

The precision policy (``jqmc/_precision.py``) assigns the Jastrow ratio
``exp(J(R') - J(R))`` to ``jastrow_ratio``, which is float64 in both modes,
while whole Jastrow values belong to ``jastrow_eval`` (float32 in mixed). A
``dltmove`` ratio formed from two whole Jastrow values would therefore round
at fp32 in mixed precision.

The committed N2 ccECP Cartesian fixture carries exp J1/J2 only (the reviewed
``_build_ham_ecp(..., "j12")`` builder). Neither channel reads AO values, so
the Jastrow ratio's whole path lies in ``jastrow_ratio`` in both modes and the
zone's own strict tolerance is the policy budget. The independent reference is
the fp64 NumPy Jastrow debug evaluator (``_compute_Jastrow_part_debug``)
applied to an independently rebuilt full-precision case of the same serialized
input, at the same ECP mesh configurations.
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

from jqmc._precision import get_dtype_jnp, get_tolerance_min, is_mixed_precision_enabled
from jqmc.derivatives import ecp_force, one_move_mesh
from jqmc.jastrow_factor import _compute_Jastrow_part_debug
from tests._precision_fidelity import mixed_vs_full
from tests.test_jqmc_kernel.test_gfmc_force_analytic import _build_ham_ecp

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_STEM = "N2_ecp_ccpvtz_cart.h5"
_SCALE = 0.9


def _case(seed):
    """Seeded N2 ccECP exp J1/J2 case rebuilt from the serialized TREXIO input."""
    ham = _build_ham_ecp(_STEM, "j12")
    gd = ham.wavefunction_data.geminal_data
    rng = np.random.default_rng(seed)
    r_up = rng.normal(scale=_SCALE, size=(int(gd.num_electron_up), 3))
    r_dn = rng.normal(scale=_SCALE, size=(int(gd.num_electron_dn), 3))
    return ham, jnp.asarray(r_up), jnp.asarray(r_dn)


def _dltmove_jastrow_ratio(case):
    """Production dltmove Jastrow ratio and its fp64 debug whole-Jastrow reference."""
    ham, r_up, r_dn = case
    common, _, jastrow_ratio, *_ = ecp_force._dltmove_delta_terms(ham, r_up, r_dn, jnp.eye(3), None)
    mesh_up = np.asarray(common["RU"])
    mesh_dn = np.asarray(common["RD"])
    jastrow_data = ham.wavefunction_data.jastrow_data
    base = _compute_Jastrow_part_debug(jastrow_data, np.asarray(r_up), np.asarray(r_dn))
    moved = np.asarray([_compute_Jastrow_part_debug(jastrow_data, up, dn) for up, dn in zip(mesh_up, mesh_dn, strict=True)])
    return {
        "fast": common["fast"],
        "mesh_up": mesh_up,
        "mesh_dn": mesh_dn,
        "jastrow_ratio": np.asarray(jastrow_ratio),
        "debug_reference": np.exp(moved - base).reshape(np.shape(jastrow_ratio)),
    }


@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("seed", [0, 1])
def test_dltmove_jastrow_ratio_matches_full_precision_debug_reference(seed):
    """Mixed and full dltmove Jastrow ratios match the fp64 debug reference of a full rebuild.

    N2_ecp_ccpvtz_cart.h5 with ``_build_ham_ecp(..., "j12")`` (exp J1/J2, no
    AO-reading Jastrow), NumPy seeds 0/1, normal scale 0.9 (up then down),
    identity mesh rotation and the default Nv=12 mesh. In mixed the actual
    ratio comes from the mixed case and the reference from an independently
    rebuilt full case; the mesh configurations must be identical. In full both
    come from the same full case (a consistency check). The budget is the
    strict ``jastrow_ratio`` zone tolerance in both modes; the ratio must be
    live (not identically one) and finite.
    """
    if is_mixed_precision_enabled():
        result, full = mixed_vs_full(lambda: _case(seed), _dltmove_jastrow_ratio)
    else:
        result = full = _dltmove_jastrow_ratio(_case(seed))
    assert result["fast"], "the rank-one mesh route must serve this molecular J1/J2 case"
    np.testing.assert_array_equal(result["mesh_up"], full["mesh_up"])
    np.testing.assert_array_equal(result["mesh_dn"], full["mesh_dn"])
    actual = result["jastrow_ratio"]
    reference = full["debug_reference"]
    assert np.all(np.isfinite(actual)), "nonfinite dltmove Jastrow ratio"
    assert np.all(np.isfinite(reference)), "nonfinite debug Jastrow ratio"
    atol, rtol = get_tolerance_min(("jastrow_ratio",), "strict")
    assert np.max(np.abs(np.log(reference))) > atol, "inactive Jastrow ratio"
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_one_move_mesh_j1_log_ratio_values_use_the_jastrow_ratio_zone():
    """The rank-one mesh J1 values whose differences form the Jastrow log-ratio are jastrow_ratio values.

    ``one_move_mesh.j1_point_values`` feeds only ``J1(r') - J1(r_e)`` of the
    mesh state's ``dJ`` (the Jastrow factor of the tmove ratio and of the
    dltmove reweight). Same N2 j12 case, seed 0, at its own ECP mesh points:
    the values must carry the ``jastrow_ratio`` dtype (float64 in both modes),
    not the ``jastrow_eval`` dtype (float32 in mixed).
    """
    ham, r_up, r_dn = _case(0)
    common = ecp_force._ecp_nl_common(ham, r_up, r_dn, jnp.eye(3), None)
    j1d = ham.wavefunction_data.jastrow_data.jastrow_one_body_data
    values = one_move_mesh.j1_point_values(j1d, common["rprime"].reshape(-1, 3))
    assert values.shape == (common["rprime"].shape[0] * common["rprime"].shape[1],)
    assert bool(jnp.all(jnp.isfinite(values))), "nonfinite one-move J1 values"
    assert values.dtype == get_dtype_jnp("jastrow_ratio")
