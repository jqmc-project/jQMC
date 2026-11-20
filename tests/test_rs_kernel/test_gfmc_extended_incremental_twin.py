"""Additional same-backend incremental interval fixtures."""

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

from jqmc._setting import ATOL_GFMC_INCREMENTAL_TWIN, RTOL_GFMC_INCREMENTAL_TWIN
from tests.test_jqmc_kernel.test_gfmc_incremental_twin_xla import _build_h4_pbc_ae_ham
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_gfmc_projection import _MESH_GP, _MESH_W, _cfg, _ctx

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize("pattern", ["h4_ae", "pbc_composite_jnone", "twist_sphepgto_j12"])
@pytest.mark.parametrize("driver", ["t", "n"])
@pytest.mark.parametrize("random_mesh", [False, True])
def test_extended_incremental_twin(pattern, driver, random_mesh):
    """AE, composite and twist native intervals retain their same-backend twin.

    Seeded positions and the same RNG key feed both routes. Walks and RNG are
    exact, all complex energy components use the existing tight twin budget,
    and actual carried-potential updates are nonzero only in production.
    """
    ham = _build_h4_pbc_ae_ham() if pattern == "h4_ae" else rs_ham(pattern)
    r_up, r_dn = _cfg(ham, seed=2)
    complex_ = pattern == "twist_sphepgto_j12"
    context = _ctx(ham, complex_=complex_)
    method = getattr(context, f"run_gfmc_projection_{driver}{'_complex' if complex_ else ''}")
    key = np.asarray(jax.random.PRNGKey(7))
    if driver == "t":
        args = (r_up, r_dn, 0.30, 0.20, int(key[0]), int(key[1]), _MESH_W, _MESH_GP)
    else:
        args = (r_up, r_dn, 0.30, -1.0, 3, int(key[0]), int(key[1]), np.empty((0, 3, 3)), _MESH_W, _MESH_GP)
    production = method(*args, random_mesh=random_mesh, nn=1)
    reference = method(*args, random_mesh=random_mesh, nn=1, debug_full_recompute=True)
    assert production["n_substeps"] == reference["n_substeps"]
    for name in ("r_up", "r_dn", "final_key"):
        np.testing.assert_array_equal(production[name], reference[name])
    for name in ("e_L", "w_L"):
        assert np.all(np.isfinite(production[name])) and np.all(np.isfinite(reference[name]))
        np.testing.assert_allclose(
            production[name],
            reference[name],
            atol=ATOL_GFMC_INCREMENTAL_TWIN,
            rtol=RTOL_GFMC_INCREMENTAL_TWIN,
            equal_nan=False,
        )
    if driver == "n":
        assert np.all(np.isfinite(production["V_diag"])) and np.all(np.isfinite(reference["V_diag"]))
        np.testing.assert_allclose(
            production["V_diag"],
            reference["V_diag"],
            atol=ATOL_GFMC_INCREMENTAL_TWIN,
            rtol=RTOL_GFMC_INCREMENTAL_TWIN,
            equal_nan=False,
        )
    expected_updates = int(production["n_substeps"]) - (driver == "t")
    assert expected_updates > 0
    assert int(production["n_pot_updates"]) == expected_updates
    assert int(reference["n_pot_updates"]) == 0
