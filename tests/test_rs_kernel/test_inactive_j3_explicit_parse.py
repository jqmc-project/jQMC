"""Free real native bindings with an INACTIVE (all-zero ``j_matrix``) J3 equal the J3-free call.

An all-zero ``j_matrix`` makes the three-body Jastrow factor identically zero, so every
quantity of the wave function equals the one of the same wave function without J3. The native
``JastrowData`` parser (``jqmc/rs/src/data.rs``) already drops such a J3. Five free real PyO3
wrappers in ``jqmc/rs/src/lib.rs`` re-read the J3 ``(orb_data, j_matrix)`` pair a second time,
unconditionally, and thread it to the kernels next to ``JastrowData``:

* ``compute_ecp_nonlocal_parts_all_pairs`` (non-local ECP energy);
* ``grad_T_bspline_kinetic_strain`` and ``grad_T_bspline_dcell`` (B-spline kinetic stress);
* ``_mcmc_sweep`` and ``_single_move`` (native Metropolis transition).

When the J3 orbital is an MO over a composite of a real pGTO and a COMPLEX plane-wave child,
the real ``OrbData`` parser drops the complex child, so the re-read orbital has fewer AO rows
than its MO coefficients and ``determinant::matmul`` panics (``ndarray: index out of bounds``,
issue #140). The real plane-wave child is the control that never panicked.

The references are independent of the J3 code: the same native call with
``jastrow_three_body_data=None``, and, for the non-local ECP energy with every ECP coefficient
set to zero, the analytic value ``0``. Equality is exact: once the inactive J3 is skipped the
two calls execute the same arithmetic.
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

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

import jqmc_rs
from jqmc.atomic_orbital import (
    AOs_cart_pgto_data,
    AOs_composite_data,
    AOs_plane_wave_complex_data,
    AOs_plane_wave_data,
)
from jqmc.determinant import SD_data, compute_geminal_all_elements
from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data
from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default
from jqmc.atomic_orbital import is_complex_aos
from jqmc.molecular_orbital import MOs_data
from tests.test_rs_kernel._rs_fixtures import draw_cfg, rs_ham

jax.config.update("jax_enable_x64", True)

# The test reads only committed fixtures through absolute paths.
pytestmark = pytest.mark.with_chdir

_WEIGHTS = np.asarray(_NV_BASE_MESH[Nv_default].weights, dtype=np.float64)
_GRID = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, dtype=np.float64)

#: J3 plane-wave child: complex with a nonzero G (two PWs), complex constant (k = 0, G = 0),
#: and the real constant PW control.
_CHILDREN = ["complex_pw_G", "complex_pw_0", "real_pw"]

_DCELL = np.array([[0.01, 0.002, 0.0], [0.0, -0.003, 0.004], [0.001, 0.0, 0.002]])


def _inactive_j3_jastrow(structure_data, child_kind):
    """Jastrow_data holding only a J3 with an all-zero j_matrix over MO(pGTO s (+) PW child)."""
    gauss = AOs_cart_pgto_data.build(
        structure_data=structure_data,
        nucleus_index=(0,),
        num_ao=1,
        num_ao_prim=1,
        orbital_indices=(0,),
        exponents=np.array([1.0]),
        coefficients=np.array([1.0]),
        angular_momentums=(0,),
        polynominal_order_x=(0,),
        polynominal_order_y=(0,),
        polynominal_order_z=(0,),
    )
    if child_kind == "complex_pw_G":
        rec = 2.0 * np.pi * np.linalg.inv(np.asarray(structure_data.cell)).T
        child = AOs_plane_wave_complex_data.build(structure_data, np.stack([np.zeros(3), rec[0]]), k_point=(0.0, 0.0, 0.0))
    elif child_kind == "complex_pw_0":
        child = AOs_plane_wave_complex_data.build(structure_data, np.zeros((1, 3)), k_point=(0.0, 0.0, 0.0))
    else:
        child = AOs_plane_wave_data.build(structure_data, np.zeros((1, 3)), np.zeros(1))
    n_ao = 1 + int(child.num_ao)
    basis = AOs_composite_data.build((gauss, child))
    dtype = np.complex128 if is_complex_aos(basis) else np.float64
    orb = MOs_data.build(aos_data=basis, mo_coefficients=np.eye(n_ao, dtype=dtype))
    j3 = Jastrow_three_body_data(orb_data=orb, j_matrix=np.zeros((n_ao, n_ao + 1)))
    j3.sanity_check()
    return Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=j3,
        jastrow_dtn_data=None,
    )


def _no_jastrow():
    return Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=None,
        jastrow_dtn_data=None,
    )


def _constant_pw_case():
    """Real Gamma determinant (constant G = 0 real PW orbital per spin) on the Li Gamma cell."""
    base = rs_ham("pbc_composite_jnone")
    st = base.structure_data
    const_pw = AOs_plane_wave_data.build(st, np.zeros((1, 3)), np.zeros(1))
    det_orb = MOs_data(aos_data=const_pw, num_mo=1, mo_coefficients=np.array([[1.0]]))
    gem = SD_data(num_electron_up=1, num_electron_dn=1, orb_data_up_spin=det_orb, orb_data_dn_spin=det_orb)
    c = np.asarray(st.positions[0], dtype=np.float64)
    r_up = (c + np.array([0.3, 0.2, 0.1]))[None, :]
    r_dn = (c + np.array([0.5, -0.2, 0.1]))[None, :]
    return base, gem, r_up, r_dn


@pytest.mark.light
@pytest.mark.parametrize("ecp", ["zero", "original"])
@pytest.mark.parametrize("child", _CHILDREN)
def test_free_ecp_nonlocal_inactive_complex_pw_j3_equals_j3_free(child, ecp):
    """compute_ecp_nonlocal_parts_all_pairs: inactive J3 == no J3; zero ECP coefficients give 0."""
    base, gem, r_up, r_dn = _constant_pw_case()
    cp = base.coulomb_potential_data
    if ecp == "zero":
        cp = dataclasses.replace(cp, coefficients=[0.0] * len(cp.coefficients))
    jas_j3 = _inactive_j3_jastrow(base.structure_data, child)

    def run(jas):
        return float(jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gem, jas, cp, r_up, r_dn, _WEIGHTS, _GRID, False))

    reference = run(_no_jastrow())
    value = run(jas_j3)
    assert np.isfinite(reference)
    if ecp == "zero":
        np.testing.assert_array_equal(reference, 0.0)
    else:
        assert reference != 0.0
    np.testing.assert_array_equal(value, reference)


@pytest.mark.light
@pytest.mark.parametrize("child", _CHILDREN)
def test_bspline_kinetic_stress_inactive_complex_pw_j3_equals_j3_free(child):
    """grad_T_bspline_kinetic_strain / grad_T_bspline_dcell: inactive J3 == no J3."""
    base = rs_ham("bspline_gamma_jnone")
    gem = base.wavefunction_data.geminal_data
    r_up, r_dn = draw_cfg(base, seed=0)
    r_up = np.asarray(r_up, dtype=np.float64)
    r_dn = np.asarray(r_dn, dtype=np.float64)
    jas_j3 = _inactive_j3_jastrow(base.structure_data, child)
    jas_0 = _no_jastrow()

    strain_ref = np.asarray(jqmc_rs.grad_T_bspline_kinetic_strain(gem, jas_0, r_up, r_dn))
    strain = np.asarray(jqmc_rs.grad_T_bspline_kinetic_strain(gem, jas_j3, r_up, r_dn))
    assert strain_ref.shape == (3, 3)
    assert np.all(np.isfinite(strain_ref)) and np.any(strain_ref != 0.0)
    np.testing.assert_array_equal(strain, strain_ref)

    dcell_ref = float(jqmc_rs.grad_T_bspline_dcell(gem, jas_0, r_up, r_dn, _DCELL))
    dcell = float(jqmc_rs.grad_T_bspline_dcell(gem, jas_j3, r_up, r_dn, _DCELL))
    assert np.isfinite(dcell_ref) and dcell_ref != 0.0
    np.testing.assert_array_equal(dcell, dcell_ref)


@pytest.mark.light
@pytest.mark.parametrize("child", _CHILDREN)
def test_native_mcmc_transition_inactive_complex_pw_j3_equals_j3_free(child):
    """_mcmc_sweep / _single_move: inactive J3 gives the same transition as no J3."""
    base, gem, r_up, r_dn = _constant_pw_case()
    st = base.structure_data
    cp = base.coulomb_potential_data
    jas_j3 = _inactive_j3_jastrow(st, child)
    jas_0 = _no_jastrow()
    G0 = np.asarray(compute_geminal_all_elements(gem, r_up, r_dn), dtype=np.float64)
    G_inv0 = np.asarray(jqmc_rs._matrix_inverse(G0), dtype=np.float64)
    R = np.asarray(st.positions, dtype=np.float64)
    Z = np.asarray(st.atomic_numbers, dtype=np.float64) - np.asarray(cp.z_cores, dtype=np.float64)
    cell = np.asarray(st.cell, dtype=np.float64)
    n_max = int(st._n_max)
    normals = np.random.default_rng(1).normal(size=8)

    def sweep(jas):
        return jqmc_rs._mcmc_sweep(gem, jas, R, Z, r_up, r_dn, G0, G_inv0, 1.2, 0.0, 1, 2, normals, cell, n_max)

    def single(jas):
        return jqmc_rs._single_move(gem, jas, R, Z, r_up, r_dn, G0, G_inv0, 1.2, 0.0, 1, 2, float(normals[0]), cell, n_max)

    for run in (sweep, single):
        reference = run(jas_0)
        value = run(jas_j3)
        assert sorted(value.keys()) == sorted(reference.keys())
        # The witnessed transition moved an electron (otherwise equality is vacuous).
        moved = np.concatenate([np.ravel(reference["r_up"]) - np.ravel(r_up), np.ravel(reference["r_dn"]) - np.ravel(r_dn)])
        assert np.any(moved != 0.0)
        for key in sorted(reference.keys()):
            np.testing.assert_array_equal(np.asarray(value[key]), np.asarray(reference[key]), err_msg=key)
