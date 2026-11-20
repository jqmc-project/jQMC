"""Exact far-row skip of the DTN candidate-table advance (IMPL-RS-DTN-TABPIP-FAR-ROW-SKIP-001).

After an accepted GFMC hop every unmoved row of a DTN candidate table (kinetic
and ECP, real and complex chains) changes only through its pair with the mover.
``dtn::advance_dtn_candidate_table_impl`` skips a row when the triangle
inequality on the minimum-image metric proves that each of its candidates is at
least ``L_u`` from both mover positions and at least ``L_f`` from every atom
the mover's f leg touches: every cutoff of that row's correction is then an
exact zero, so the skipped correction could only have turned a stored -0 into
+0. The test-only hook ``JQMC_RS_DTN_TABPIP_NO_FAR_SKIP`` selects the
evaluation of every row.

The fixtures rebuild the DTN leg with short cutoffs so far rows exist (the
production cutoffs of the small committed cells reach almost every pair). On
GFMC_t intervals of a molecular, a Gamma-point PBC and a complex-twist PBC DTN
fixture, the skip and the hook give BYTE-identical e_L, w_L, final walkers,
sub-step counts and keys; the skip counts skipped rows (non-vacuity,
``jqmc_rs.gfmc_route_counters`` index 72) and the hook counts none.
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

_HOOK = "JQMC_RS_DTN_TABPIP_NO_FAR_SKIP"
_FAR_ROWS = 72
# pattern -> (complex chain, short-cutoff DTN rebuild arguments)
_CASES = {
    "mol_cart_sd_jdtn": (False, dict(L_u=1.5, L_chi=2.0, L_f=1.0)),
    "pbc_gamma_sphepgto_jdtn": (False, dict(cutoff_ratio_u=0.35, cutoff_ratio_chi=0.5, cutoff_ratio_f=0.5)),
    "twist_sphepgto_jdtn": (True, dict(cutoff_ratio_u=0.35, cutoff_ratio_chi=0.5, cutoff_ratio_f=0.5)),
}


def _short_cutoff_ham(pattern, cutoffs):
    from jqmc.jastrow_factor import Jastrow_DTN_data

    ham = rs_ham(pattern)
    jd = ham.wavefunction_data.jastrow_data
    assert jd.jastrow_dtn_data is not None, pattern
    sd, cp = ham.structure_data, ham.coulomb_potential_data
    z = np.asarray(sd.atomic_numbers)
    zeff = np.asarray(cp._effective_charges)
    dtn = Jastrow_DTN_data.build(
        structure_data=sd,
        z_eff_per_species=tuple(float(np.mean(zeff[z == Z])) for Z in np.unique(z)),
        pseudized_per_species=tuple(bool(np.all(cp.pseudized_per_atom[z == Z])) for Z in np.unique(z)),
        cutoff_smoothness=4,
        n_order_u=2,
        n_order_chi=2,
        n_order_eN=2,
        n_order_ee=2,
        random_init=True,
        seed=17,
        **cutoffs,
    )
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd.replace(jastrow_dtn_data=dtn)))
    ham.sanity_check()
    return ham


def _interval(ham, complex_, r_up, r_dn, hook, monkeypatch):
    if hook:
        monkeypatch.setenv(_HOOK, "1")
    else:
        monkeypatch.delenv(_HOOK, raising=False)
    ctx = _ctx(ham, complex_=complex_)
    key = jax.random.PRNGKey(29)
    run = ctx.run_gfmc_projection_t_complex if complex_ else ctx.run_gfmc_projection_t
    jqmc_rs.gfmc_route_counters_reset()
    out = run(r_up, r_dn, 0.30, 0.40, int(key[0]), int(key[1]), _MESH_W, _MESH_GP, random_mesh=True, nn=1)
    return out, int(jqmc_rs.gfmc_route_counters()[_FAR_ROWS])


@pytest.mark.light
@pytest.mark.parametrize("pattern", sorted(_CASES))
def test_dtn_tabpip_far_row_skip_is_bitwise(pattern, monkeypatch):
    """Skipping provably-far DTN table rows changes no output bit and is route-observable."""
    complex_, cutoffs = _CASES[pattern]
    ham = _short_cutoff_ham(pattern, cutoffs)
    r_up, r_dn = _cfg(ham, 31)
    skip, far_s = _interval(ham, complex_, r_up, r_dn, False, monkeypatch)
    keep, far_k = _interval(ham, complex_, r_up, r_dn, True, monkeypatch)
    e_s, e_k = (complex(skip["e_L"]), complex(keep["e_L"])) if complex_ else (float(skip["e_L"]), float(keep["e_L"]))
    np.testing.assert_array_equal(np.asarray(e_s), np.asarray(e_k), err_msg=f"{pattern}: e_L")
    np.testing.assert_array_equal(np.asarray(float(skip["w_L"])), np.asarray(float(keep["w_L"])), err_msg=pattern)
    np.testing.assert_array_equal(np.asarray(skip["r_up"]), np.asarray(keep["r_up"]))
    np.testing.assert_array_equal(np.asarray(skip["r_dn"]), np.asarray(keep["r_dn"]))
    assert int(skip["n_substeps"]) == int(keep["n_substeps"]), pattern
    assert list(np.asarray(skip["final_key"], dtype=np.uint32)) == list(np.asarray(keep["final_key"], dtype=np.uint32))
    assert int(skip["n_substeps"]) > 1, (pattern, "vacuous: no accepted hop", int(skip["n_substeps"]))
    assert far_k == 0, (pattern, "the hook skipped rows", far_k)
    assert far_s > 0, (pattern, "vacuous: no far row skipped", far_s)
