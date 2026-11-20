"""The weighted periodic mesh Pulay-stress sum equals the dense per-configuration one (issue #622).

``one_move_mesh.mesh_stress_weighted`` contracts the configuration weights before any
strain field is evaluated; ``one_move_mesh.mesh_stress_wf`` evaluates the full periodic
Pulay stress ``stress_lnPsi_pbc`` at every one-electron configuration and then takes the
weighted sum.  For random (complex at a twist) weights both must agree to round-off,
with and without the DTN term.
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

import os
import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from jqmc._precision import get_tolerance_min
from jqmc.derivatives import one_move_mesh
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_DTN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._ecp_core_electrons import core_electrons_of
from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg

_TREXIO = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files")


def _build(stem, with_j1, with_j3, with_dtn):
    (sd, aos, _mu, _md, gem, cpd) = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=sd)
    j1 = None
    if with_j1:
        j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=sd,
            core_electrons=core_electrons_of(cpd, len(sd.positions)),
            jastrow_1b_type="exp",
        )
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos, random_init=True, seed=3) if with_j3 else None
    dtn = None
    if with_dtn:
        dtn = Jastrow_DTN_data.build(
            structure_data=sd,
            z_eff_per_species=(1.0,),  # Li ccECP: Z=3, two core electrons
            pseudized_per_species=(True,),
            use_u=True,
            use_chi=True,
            use_f=True,
            cutoff_smoothness=4,
            n_order_u=2,
            n_order_chi=2,
            n_order_eN=2,
            n_order_ee=2,
            L_u=None,
            L_chi=None,
            L_f=None,
            random_init=True,
            seed=7,
        )
    jd = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=j3, jastrow_dtn_data=dtn)
    wf = Wavefunction_data(jastrow_data=jd, geminal_data=gem)
    return Hamiltonian_data(structure_data=sd, coulomb_potential_data=cpd, wavefunction_data=wf)


def _mesh_and_weights(ham, seed, complex_w, nv=4):
    # Base configuration from the shared conditioned-case draw (cond(G) cap on the geminal value
    # alone): the dense reference rebuilds a per-config Sherman--Morrison inverse, whose fp32-zone
    # error in mixed mode grows with cond(G) independently of the weighted contraction under test.
    r_up, r_dn = draw_well_conditioned_cfg(ham, seed=seed)
    r_up, r_dn = jnp.asarray(r_up), jnp.asarray(r_dn)
    nu, nd = r_up.shape[0], r_dn.shape[0]
    rng = np.random.default_rng(seed)
    r_all = jnp.concatenate([r_up, r_dn], axis=0)
    rprime = r_all[:, None, :] + jnp.asarray(rng.normal(scale=0.3, size=(nu + nd, nv, 3)))
    W = rng.normal(size=(nu + nd, nv))
    if complex_w:
        W = W + 1j * rng.normal(size=(nu + nd, nv))
    return r_up, r_dn, rprime, jnp.asarray(W)


@pytest.mark.light
@pytest.mark.parametrize(
    "stem,with_j1,with_j3,with_dtn,complex_w,seed",
    [
        ("Li_pbc_ae_ccpvdz_cart.h5", True, True, False, False, 1),  # periodic real Gamma, J1 + J2 + J3
        ("Li_pbc_ae_ccpvdz_sphe.h5", False, True, False, False, 2),  # periodic real Gamma, spherical AOs, J2 + J3
        ("H2_pbc_ae_k025_cart.h5", True, False, False, True, 3),  # complex twist, complex weights
        ("lithium_k025.h5", False, False, True, True, 4),  # complex twist + ECP + DTN u/chi/f
    ],
)
def test_weighted_mesh_stress_matches_dense_per_config_stress(stem, with_j1, with_j3, with_dtn, complex_w, seed):
    """mesh_stress_weighted == mesh_stress_wf (both include_dtn modes) for random weights."""
    ham = _build(stem, with_j1, with_j3, with_dtn)
    r_up, r_dn, rprime, W = _mesh_and_weights(ham, seed, complex_w)
    _ratio, _oee, state = one_move_mesh.mesh_ratio_state(ham, r_up, r_dn, rprime)
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap", "det_eval", "jastrow_grad_lap"), "strict")
    for include_dtn in (True, False) if with_dtn else (True,):
        dense = np.asarray(
            one_move_mesh.mesh_stress_wf(ham, r_up, r_dn, rprime, W, state["base"], state["dm"], include_dtn=include_dtn)
        )
        weighted = np.asarray(
            one_move_mesh.mesh_stress_weighted(ham, r_up, r_dn, rprime, W, state["base"], state["dm"], include_dtn=include_dtn)
        )
        assert np.all(np.isfinite(dense))
        scale = max(1.0, float(np.max(np.abs(dense))))
        np.testing.assert_allclose(
            weighted, dense, atol=atol * scale, rtol=rtol, err_msg=f"weighted mesh stress, include_dtn={include_dtn}"
        )
