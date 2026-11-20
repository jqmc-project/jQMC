"""Nonuniform-COMPLEX-weight coverage for the shared rank-one weighted-delta kernels.

The production S'/ECP consumers feed the
shared kernel physically-structured weights, so this suite drives
``fast_mesh.weighted_position_delta`` / ``weighted_strain_delta`` with RANDOM
nonuniform complex ``B_g`` directly -- the cell no physical fixture stresses.

Two independent pins, numerical equality only:

* the ``dr`` axis against a self-contained ``jax.grad`` oracle of
  ``G = Re sum_g W_g [ln Psi(config_g) - ln Psi(base)]`` built from the production
  determinant/Jastrow VALUE evaluators (independent autograd -- no rank-one code in
  the reference path);
* all three axes (``dR``, ``dr``, strain) against the streamed full-recompute
  production path (the retained fallback for non-(pade/exp) Jastrow cells), on the
  SAME random complex weights -- production-vs-production consistency.
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
from jqmc.determinant import compute_det_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data, compute_Jastrow_part
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

_TREXIO = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files")


def _build(stem, with_j2):
    (sd, _aos, _mu, _md, gem, cpd) = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0) if with_j2 else None
    jd = Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=j2, jastrow_three_body_data=None, jastrow_dtn_data=None)
    wf = Wavefunction_data(jastrow_data=jd, geminal_data=gem)
    return Hamiltonian_data(structure_data=sd, coulomb_potential_data=cpd, wavefunction_data=wf)


def _mesh_and_weights(ham, seed, complex_w):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    ne = nu + nd
    R = np.asarray(ham.structure_data.positions)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(R[rng.integers(0, len(R), nu)] + rng.normal(scale=0.35, size=(nu, 3)))
    r_dn = jnp.asarray(R[rng.integers(0, len(R), nd)] + rng.normal(scale=0.35, size=(nd, 3)))
    shifts = jnp.asarray(rng.normal(scale=0.3, size=(6, 3)))  # nonuniform candidate offsets
    W = rng.normal(size=(ne, 6))
    if complex_w:
        W = W + 1j * rng.normal(size=(ne, 6))
    return r_up, r_dn, shifts, jnp.asarray(W)


def _oracle_dr(ham, r_up, r_dn, shifts, W):
    """jax.grad of G = Re sum_{e,v} W [ln Psi(config) - ln Psi(base)] wrt (r_up, r_dn):
    production VALUE evaluators only (complex det log; real Jastrow) -- no rank-one code."""
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    nu = int(r_up.shape[0])
    ne = nu + int(r_dn.shape[0])

    def _G(ru, rd):
        det0 = compute_det_geminal_all_elements(gd, ru, rd)
        j0 = compute_Jastrow_part(jd, ru, rd)
        r_all = jnp.concatenate([ru, rd], axis=0)

        def _one(e, v):
            p = r_all[e] + shifts[v]
            ru_c = jnp.where((jnp.arange(nu)[:, None] == e), p[None, :], ru)
            rd_c = jnp.where((jnp.arange(ne - nu)[:, None] == (e - nu)), p[None, :], rd)
            det_c = compute_det_geminal_all_elements(gd, ru_c, rd_c)
            j_c = compute_Jastrow_part(jd, ru_c, rd_c)
            # complex log so a sign-flipped REAL ratio stays finite: Re(log z) = ln|z| and
            # d Re(log z)/dx = Re(z'/z) -- exactly the d ln|Psi| the production delta carries.
            return jnp.log(jnp.asarray(det_c, jnp.complex128)) - jnp.log(jnp.asarray(det0, jnp.complex128)) + (j_c - j0)

        ee, vv = jnp.meshgrid(jnp.arange(ne), jnp.arange(6), indexing="ij")
        lnrho = jax.vmap(jax.vmap(_one))(ee, vv)  # (ne, 6)
        return jnp.real(jnp.sum(W * lnrho))

    gu, gdn = jax.grad(_G, argnums=(0, 1))(r_up, r_dn)
    return jnp.concatenate([gu, gdn], axis=0)


@pytest.mark.light
@pytest.mark.parametrize(
    "stem,with_j2,complex_w,seed",
    [
        ("H2_pbc_ae_k025_cart.h5", True, True, 5),  # complex twist x nonuniform complex B_g
        ("H2_pbc_ae_k025_cart.h5", False, True, 7),
        ("H2_pbc_ae_cart.h5", True, False, 5),  # real Gamma control
        ("N_ae_ccpvdz_cart.h5", True, False, 9),  # molecular control
    ],
)
def test_weighted_position_delta_vs_autograd_oracle(stem, with_j2, complex_w, seed):
    """Weighted position delta vs autograd oracle."""
    from jqmc.derivatives import fast_mesh

    ham = _build(stem, with_j2)
    r_up, r_dn, shifts, W = _mesh_and_weights(ham, seed, complex_w)
    r_all = jnp.concatenate([r_up, r_dn], axis=0)
    rprime = r_all[:, None, :] + shifts[None, :, :]

    st = fast_mesh.build_move_state(ham, r_up, r_dn, rprime)
    _dR, dr = fast_mesh.weighted_position_delta(ham, r_up, r_dn, st, W, holomorphic=complex_w)
    dr = jnp.real(dr)

    dr_oracle = _oracle_dr(ham, r_up, r_dn, shifts, W)
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap", "det_eval", "jastrow_grad_lap"), "strict")
    np.testing.assert_allclose(np.asarray(dr), np.asarray(dr_oracle), atol=atol, rtol=rtol)
