"""Full-channel sibling of the existing native DTN stress test."""

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
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_rs_kernel import test_stress_pbc as original

pytestmark = [pytest.mark.with_chdir, pytest.mark.heavy]


def test_grad_eL_pbc_dtn_strain_uchif():
    """Native periodic DTN stress includes a live f channel and matches the existing strained-energy oracle.

    The committed lithium Gamma pGTO ccECP pattern uses the existing coefficient
    seed5 and electron seed1, with u/chi/f all enabled. The full component sum
    matches same-mode strained-energy autodiff plus the existing comoving-cutoff
    correction. Switching only the DTN object's f flag off must change its
    standalone native kinetic-strain tensor beyond the existing parity band.
    This adds coverage; the original u/chi-only row remains unchanged.
    """
    ham = original.rs_ham("pbc_gamma_sphepgto_jdtn")
    gd = ham.wavefunction_data.geminal_data
    jd = original._randomize_dtn(ham.wavefunction_data.jastrow_data, seed=5)
    assert jd.jastrow_dtn_data.use_f
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))
    cd = ham.coulomb_potential_data
    ed = cd.ewald_data
    sd = ham.structure_data
    weights = np.asarray(original.icosahedron_sym_mesh_Nv12.weights, float)
    points = np.asarray(original.icosahedron_sym_mesh_Nv12.grid_points, float)
    eta = float(ed.eta)
    images = np.asarray(ed.real_image_list, float)
    gv = np.asarray(ed.g_vectors, float)
    gf = np.asarray(ed.g_factors, float)
    cell = np.asarray(sd.cell, float)
    nmax = int(sd._n_max)
    charges = np.asarray(cd._effective_charges, float)
    ions = np.asarray(ed.structure_data.positions, float)
    r_up, r_dn = original._cfg(ham)
    atol, rtol = original.get_tolerance_xla_rust_consistency(
        original._ZONES_ECP_ELPBC + ("jastrow_eval", "jastrow_grad_lap"), level="strict"
    )

    native_dtn = np.asarray(original.jqmc_rs.grad_T_dtn_pbc_wrt_strain(gd, jd, r_up, r_dn))
    no_f = jd.replace(jastrow_dtn_data=jd.jastrow_dtn_data.replace(use_f=False))
    native_uchi = np.asarray(original.jqmc_rs.grad_T_dtn_pbc_wrt_strain(gd, no_f, r_up, r_dn))
    assert np.all(np.isfinite(native_dtn)) and np.all(np.isfinite(native_uchi))
    assert not original._close(native_dtn, native_uchi, atol, rtol), "DTN f kinetic-strain channel is inactive"

    def forward(strain):
        transform = jnp.eye(3) + strain
        return original.compute_local_energy(ham._apply_strain(transform), r_up @ transform.T, r_dn @ transform.T, jnp.eye(3))

    fixed_forward = np.asarray(jax.grad(forward)(jnp.zeros((3, 3))))
    analytical = np.asarray(
        original.derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), {"opt_eps": True}, analytic=True)["eps"]
    )
    fixed_autodiff = np.asarray(original._auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="eps", mask=None))
    reference = fixed_forward + (analytical - fixed_autodiff)
    actual = (
        np.asarray(original.jqmc_rs.grad_T_pbc_wrt_strain(gd, jd, r_up, r_dn))
        + native_dtn
        + np.asarray(original.jqmc_rs.grad_Eee_wrt_strain(r_up, r_dn, eta, images, gv, gf, cell, nmax))
        + np.asarray(original.jqmc_rs.grad_Eei_wrt_strain(r_up, r_dn, eta, images, gv, gf, charges, ions, cell, nmax))
        + np.asarray(original.jqmc_rs.grad_Eii_wrt_strain(eta, images, gv, gf, charges, ions, cell))
        + np.asarray(original.jqmc_rs.grad_V_ecp_local_wrt_strain(cd, r_up, r_dn))
        + np.asarray(original.jqmc_rs.grad_V_ecp_nonlocal_wrt_strain(gd, cd, r_up, r_dn, weights, points, jd))
    )
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    assert original._close(actual, reference, atol, rtol), "full-channel native DTN energy-strain mismatch"
