"""Exact zero-amplitude ECP group skip on the real FAST GFMC mesh (IMPL-RS-GFMC-ECP-ZERO-GROUP-SKIP-001).

On the real FAST projection mesh (``gfmc::compute_ecp_nonlocal_mesh_fast``) an
electron whose every ECP quadrature amplitude ``ang`` is exactly 0.0 -- its
nearest ECP nucleus carries only zero-coefficient non-local channels (the ccECP
H placeholder), or the channel underflows -- contributes ``a_m = 0 * ratio *
jas = +-0`` to the FN sum, the SP sum and the hop cdf, i.e. nothing. Its
determinant / Jastrow ratios are not evaluated and its ECP candidate-table
rows (determinant MO table, J3 table and J3 log-ratio advance, J1/J2 log-ratio
advance) are neither built nor advanced; they are rebuilt when the electron
moves, the only event that can change its amplitudes before the next mesh
rotation rebuilds every table.

The test-only hook ``JQMC_RS_GFMC_ECP_KEEP_ZERO_GROUPS`` selects the
evaluation of every group. Asserted on GFMC_t and GFMC_n intervals (random mesh,
tmove and dltmove) of the water ccECP fixture (O groups live, H groups zero) and
the H2 ccECP fixture with J3 (every group zero): the skip and the hook give
BYTE-identical e_L, w_L, final walkers, sub-step counts and keys; the skip
counts zero groups (non-vacuity) and evaluates exactly the groups the hook
counts as live minus those (``jqmc_rs.gfmc_route_counters`` indices 68 live /
69 zero); the hook counts no zero group.
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

import jqmc_rs  # noqa: E402

from tests.test_rs_kernel._rs_fixtures import rs_ham  # noqa: E402
from tests.test_rs_kernel.test_gfmc_projection import _MESH_GP, _MESH_W, _cfg, _ctx  # noqa: E402

pytestmark = [pytest.mark.with_chdir]

_HOOK = "JQMC_RS_GFMC_ECP_KEEP_ZERO_GROUPS"
_PATTERNS = ["mol_ecp_sphe_j12", "mol_ecp_cart_j123"]


def _interval(ham, r_up, r_dn, mode, dltmove, keep, monkeypatch):
    if keep:
        monkeypatch.setenv(_HOOK, "1")
    else:
        monkeypatch.delenv(_HOOK, raising=False)
    ctx = _ctx(ham)
    key = jax.random.PRNGKey(41)
    kw = dict(random_mesh=True, dltmove=dltmove, nn=1)
    jqmc_rs.gfmc_route_counters_reset()
    if mode == "t":
        out = ctx.run_gfmc_projection_t(r_up, r_dn, 0.30, 0.20, int(key[0]), int(key[1]), _MESH_W, _MESH_GP, **kw)
    else:
        out = ctx.run_gfmc_projection_n(
            r_up, r_dn, 0.30, -1.0, 8, int(key[0]), int(key[1]), np.zeros((0, 3, 3)), _MESH_W, _MESH_GP, **kw
        )
    counters = jqmc_rs.gfmc_route_counters()
    return out, int(counters[68]), int(counters[69])


@pytest.mark.light
@pytest.mark.parametrize("pattern", _PATTERNS)
@pytest.mark.parametrize("mode", ["t", "n"])
@pytest.mark.parametrize("dltmove", [False, True])
def test_gfmc_ecp_zero_group_skip_is_bitwise(pattern, mode, dltmove, monkeypatch):
    """Skipping exactly-zero ECP groups changes no output bit and is route-observable."""
    ham = rs_ham(pattern)
    assert bool(ham.coulomb_potential_data.ecp_flag) and not bool(ham.structure_data.pbc_flag), pattern
    r_up, r_dn = _cfg(ham, 23)
    skip, live_s, zero_s = _interval(ham, r_up, r_dn, mode, dltmove, False, monkeypatch)
    keep, live_k, zero_k = _interval(ham, r_up, r_dn, mode, dltmove, True, monkeypatch)
    for k in ("e_L", "w_L"):
        np.testing.assert_array_equal(np.asarray(float(skip[k])), np.asarray(float(keep[k])), err_msg=f"{pattern}: {k}")
    np.testing.assert_array_equal(np.asarray(skip["r_up"]), np.asarray(keep["r_up"]))
    np.testing.assert_array_equal(np.asarray(skip["r_dn"]), np.asarray(keep["r_dn"]))
    assert int(skip["n_substeps"]) == int(keep["n_substeps"]), pattern
    assert list(np.asarray(skip["final_key"], dtype=np.uint32)) == list(np.asarray(keep["final_key"], dtype=np.uint32))
    assert zero_k == 0, (pattern, "the keep hook skipped groups", zero_k)
    assert zero_s > 0, (pattern, "vacuous: no exactly-zero ECP group", live_s, zero_s)
    assert live_s + zero_s == live_k, (pattern, live_s, zero_s, live_k)
