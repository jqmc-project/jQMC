"""Fixed Cl2 ccECP/J1+J2 mesh-endpoint position derivatives.

Preserve the original diagnostic configuration: Cl2_ecp_ccpvtz_cart.h5,
``_build_ham_ecp(..., "j12")``, NumPy seed 0, normal scale 0.9 (up then down),
up electron 0 at nucleus 0 plus 0.4 times normalized Nv=12 grid direction 0,
identity mesh rotation, and alat 0.8. No cond(G) screening or resampling.

The independent continuum-measurement primal ``_e_L_t_ecp`` is differentiated
in the SAME precision as the analytical GFMC position derivative, for both
tmove and dltmove and every nuclear/up/down component. The
registration uses the full-strict/mixed-medium GFMC policy. Missing electronic ECP derivatives and
the singular Legendre endpoint formula are the concrete fault controls.
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
from jqmc.derivatives import ecp_force, gfmc
from jqmc.jqmc_mcmc import _extract_active_positions_from_ham, _inject_active_positions
from tests.test_jqmc_kernel.test_gfmc_force_analytic import _build_ham_ecp, _e_L_t_ecp, _jit_grad_pos

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]


@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("nlm", ["tmove", "dltmove"])
def test_gfmc_force_matches_autograd_with_an_electron_on_a_mesh_direction(nlm):
    """Cl2 committed ccECP Cartesian-GTO SD + exp J1/J2, seed 0 normal scale 0.9.

    Up electron 0 is placed 0.4 from nucleus 0 along Nv=12 mesh direction 0;
    RT is identity, alat is 0.8, and cond(G) is neither screened nor bounded.
    Compare every nuclear/up/down derivative with same-precision primal AD:
    central strict in full and central medium in mixed, with finite operands.
    """
    ham = _build_ham_ecp("Cl2_ecp_ccpvtz_cart.h5", "j12")
    gd = ham.wavefunction_data.geminal_data
    seed = 0
    scale = 0.9
    mesh_radius = 0.4
    alat = 0.8
    rng = np.random.default_rng(seed)
    r_up = rng.normal(scale=scale, size=(int(gd.num_electron_up), 3))
    r_dn = rng.normal(scale=scale, size=(int(gd.num_electron_dn), 3))
    grid_direction = np.asarray(ecp_force._mesh(12).grid_points)[0]
    nucleus_zero = np.asarray(ham.structure_data.positions)[0]
    r_up[0] = nucleus_zero + mesh_radius * grid_direction / np.linalg.norm(grid_direction)
    r_up, r_dn = jnp.asarray(r_up), jnp.asarray(r_dn)
    RT = jnp.eye(3, dtype=jnp.float64)

    # Exact endpoint, with no epsilon displacement or conditioning selection.
    relative = r_up[0] - jnp.asarray(nucleus_zero)
    unit_electron = relative / jnp.linalg.norm(relative)
    direction = jnp.asarray(grid_direction) @ RT
    unit_grid = direction / jnp.linalg.norm(direction)
    assert float(jnp.dot(unit_electron, unit_grid)) == 1.0

    active0 = _extract_active_positions_from_ham(ham)

    def f_active(active, up, down):
        return _e_L_t_ecp(_inject_active_positions(ham, active), up, down, RT, alat, non_local_move=nlm)

    gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)
    dR, dr_up, dr_dn = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat, non_local_move=nlm)
    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap")
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(zones, level)
    for quantity, actual, reference in (
        ("nuclear", dR, gR_oracle),
        ("electron-up", dr_up, gru_oracle),
        ("electron-down", dr_dn, grd_oracle),
    ):
        actual, reference = np.asarray(actual), np.asarray(reference)
        assert np.all(np.isfinite(actual)), f"{nlm}: nonfinite analytical {quantity} derivative"
        assert np.all(np.isfinite(reference)), f"{nlm}: nonfinite same-precision AD {quantity} derivative"
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"{nlm}: {quantity}")
