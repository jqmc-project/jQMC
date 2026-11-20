"""Weighted one-move mesh sums equal the weighted dense per-configuration fields (issue #596).

``one_move_mesh.mesh_drift_weighted`` / ``mesh_force_weighted`` contract the
configuration weights into the rank-one factors; ``mesh_ratio_and_drift`` /
``det_drift_mesh`` / ``mesh_force`` build every configuration's field.  For random
(complex at a twist) weights the two must agree to round-off, together with the
ratio and the moved electron's own drift ``Oee``.
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
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_three_body_data, Jastrow_two_body_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._ecp_core_electrons import core_electrons_of

_TREXIO = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files")


def _build(stem, with_j1, with_j3):
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
    jd = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=j3, jastrow_dtn_data=None)
    wf = Wavefunction_data(jastrow_data=jd, geminal_data=gem)
    return Hamiltonian_data(structure_data=sd, coulomb_potential_data=cpd, wavefunction_data=wf)


def _mesh_and_weights(ham, seed, complex_w, nv=6):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(R[rng.integers(0, len(R), nu)] + rng.normal(scale=0.35, size=(nu, 3)))
    r_dn = jnp.asarray(R[rng.integers(0, len(R), nd)] + rng.normal(scale=0.35, size=(nd, 3)))
    r_all = jnp.concatenate([r_up, r_dn], axis=0)
    rprime = r_all[:, None, :] + jnp.asarray(rng.normal(scale=0.3, size=(nu + nd, nv, 3)))
    W = rng.normal(size=(nu + nd, nv))
    if complex_w:
        W = W + 1j * rng.normal(size=(nu + nd, nv))
    return r_up, r_dn, rprime, jnp.asarray(W)


def _assert_close(actual, desired, atol, rtol, msg):
    desired = np.asarray(desired)
    scale = max(1.0, float(np.max(np.abs(desired))))
    np.testing.assert_allclose(np.asarray(actual), desired, atol=atol * scale, rtol=rtol, err_msg=msg)


@pytest.mark.light
@pytest.mark.parametrize(
    "stem,with_j1,with_j3,complex_w,seed",
    [
        ("H2_ecp_ccpvtz_cart.h5", True, True, False, 1),  # molecular ECP, J1 + J2 + J3
        ("N2_ecp_ccpvtz_cart.h5", False, True, False, 2),  # molecular ECP, J2 + J3
        ("Li_pbc_ae_ccpvdz_cart.h5", True, True, False, 4),  # periodic real Gamma
        ("H2_pbc_ae_k025_cart.h5", False, False, True, 5),  # complex twist, complex weights
    ],
)
def test_weighted_mesh_sums_match_dense_fields(stem, with_j1, with_j3, complex_w, seed):
    """Weighted drift (full and det-only) and Pulay force vs the dense per-config fields."""
    ham = _build(stem, with_j1, with_j3)
    r_up, r_dn, rprime, W = _mesh_and_weights(ham, seed, complex_w)
    ne, nv = rprime.shape[0], rprime.shape[1]

    ratio_d, O, base, dm = one_move_mesh.mesh_ratio_and_drift(ham, r_up, r_dn, rprime)
    O_det = one_move_mesh.det_drift_mesh(base, dm).reshape(ne, nv, ne, 3)
    OR = one_move_mesh.mesh_force(ham, r_up, r_dn, rprime, base, dm)

    ratio_w, Oee, state = one_move_mesh.mesh_ratio_state(ham, r_up, r_dn, rprime)
    drift_w = one_move_mesh.mesh_drift_weighted(ham, state, W)
    drift_det_w = one_move_mesh.mesh_drift_weighted(ham, state, W, det_only=True)
    force_w = one_move_mesh.mesh_force_weighted(ham, state, W)

    eidx = jnp.arange(ne)
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap", "det_eval", "jastrow_grad_lap"), "strict")
    _assert_close(ratio_w, ratio_d, atol, rtol, "ratio")
    _assert_close(Oee, O[eidx, :, eidx, :], atol, rtol, "moved-electron drift Oee")
    _assert_close(drift_w, jnp.einsum("ev,evmd->md", W, O), atol, rtol, "weighted drift")
    _assert_close(drift_det_w, jnp.einsum("ev,evmd->md", W, O_det), atol, rtol, "weighted det-only drift")
    _assert_close(force_w, jnp.einsum("ev,evad->ad", W, OR), atol, rtol, "weighted Pulay force")
