"""Native parity for the selected conventional periodic J1/J2 geometry law."""

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

import jqmc_rs
from jqmc import jastrow_factor as jastrow
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from jqmc.derivatives import gfmc
from jqmc.derivatives import pbc_stress
from jqmc.derivatives.force import grad_T_wrt_R_withJ
from tests._gfmc_twin_fixtures import far_start, skewed_h4_ham
from tests.test_jqmc_kernel.test_jastrow_skew_geometry import (
    _CASE_IDS,
    _CASES,
    _CENTER,
    _FRACTIONS,
    _cell,
    _data,
    _positions,
)

pytestmark = [pytest.mark.with_chdir]


def _skew_j12_ham(j1_form, j2_form, trexio_file="H4_pbc_ae_cart.h5"):
    """Reuse the skew H4 fixture with complementary conventional radial forms."""
    ham = skewed_h4_ham(twist={"H4_pbc_ae_cart.h5": False}[trexio_file])
    wf = ham.wavefunction_data
    jd = wf.jastrow_data
    j1 = jd.jastrow_one_body_data.replace(jastrow_1b_type=j1_form)
    j2 = jd.jastrow_two_body_data.replace(jastrow_2b_type=j2_form)
    jd = jd.replace(jastrow_one_body_data=j1, jastrow_two_body_data=j2)
    ham = ham.replace(wavefunction_data=wf.replace(jastrow_data=jd))
    ham.sanity_check()
    return ham


@pytest.mark.light
@pytest.mark.parametrize("part,form,cell_kind,spin_sector", _CASES, ids=_CASE_IDS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_native_periodic_jastrow_components_match_xla(part, form, cell_kind, spin_sector):
    """Native J1/J2 values, coordinate derivatives, parameter derivatives and ratios match XLA.

    Reuse the independent XLA-law suite's eight explicit component/form/cell/spin
    fixtures: cubic through strongly skewed cells, exp and Pade, both spin labels
    and same/opposite-spin pairs. Three fractional displacements and center
    (0.2,0.3,0.4) Bohr are fixed; there is no file, RNG or determinant screening.
    Move the selected electron by a fixed fractional offset plus (1,-1,2)
    lattice vectors. Compare the native ratio to the matching XLA rank-one
    route, not a differently rounded full recomputation.

    The reference is the corresponding XLA component, its parameter AD, and
    its analytic coordinate derivatives; the independent physical law is
    established separately in the XLA suite rather than duplicated here.
    Each active precision uses its existing strict backend-consistency policy.
    """
    data = _data(part, form, cell_kind)
    cell = _cell(cell_kind)
    center = jnp.asarray(_CENTER, dtype=jnp.float64)
    if part == "j1":
        jd = jastrow.Jastrow_data(jastrow_one_body_data=data)
        native_value = jqmc_rs.compute_Jastrow_one_body
        xla_value = jastrow.compute_Jastrow_one_body
        native_parameter = jqmc_rs.grad_ln_Psi_jastrow_1b_param
        parameter_name = "jastrow_1b_param"
    else:
        jd = jastrow.Jastrow_data(jastrow_two_body_data=data)
        native_value = jqmc_rs.compute_Jastrow_two_body
        xla_value = jastrow.compute_Jastrow_two_body
        native_parameter = jqmc_rs.grad_ln_Psi_jastrow_2b_param
        parameter_name = "jastrow_2b_param"
    value_atol, value_rtol = get_tolerance_xla_rust_consistency(("jastrow_eval",), level="strict")
    coordinate_atol, coordinate_rtol = get_tolerance_xla_rust_consistency(("jastrow_grad_lap",), level="strict")
    ratio_atol, ratio_rtol = get_tolerance_xla_rust_consistency(("jastrow_eval", "jastrow_ratio"), level="strict")
    for sample_index, fraction in enumerate(_FRACTIONS):
        sample_label = str(sample_index)
        displacement = jnp.asarray(fraction, dtype=jnp.float64) @ cell
        up, down, moved_spin = _positions(displacement, spin_sector, center)
        ru, rd = np.asarray(up), np.asarray(down)
        actual_value = native_value(data, ru, rd)
        reference_value = xla_value(data, up, down)
        np.testing.assert_allclose(
            actual_value, reference_value, atol=value_atol, rtol=value_rtol, equal_nan=False, err_msg=str(sample_index)
        )
        actual_derivatives = jqmc_rs.compute_grads_and_laplacian_jastrow_part(jd, ru, rd)
        reference_derivatives = jastrow.compute_grads_and_laplacian_Jastrow_part(jd, up, down)
        for actual_part, reference_part in zip(actual_derivatives, reference_derivatives, strict=True):
            assert actual_part.shape == reference_part.shape
        for quantity, indices in (("gradient", (0, 1)), ("laplacian", (2, 3))):
            # Preserve both spin partitions above, then compare every numerical
            # entry together. Empty-spin arrays are not numerical evidence.
            actual_derivative = np.concatenate([actual_derivatives[index] for index in indices], axis=0)
            reference_derivative = np.concatenate([reference_derivatives[index] for index in indices], axis=0)
            np.testing.assert_allclose(
                actual_derivative,
                reference_derivative,
                atol=coordinate_atol,
                rtol=coordinate_rtol,
                equal_nan=False,
                err_msg=quantity,
            )
        actual_parameter = native_parameter(data, ru, rd)
        parameter_gradient = jax.grad(lambda value: xla_value(value, up, down))(data)
        reference_parameter = getattr(parameter_gradient, parameter_name)
        np.testing.assert_allclose(
            actual_parameter,
            reference_parameter,
            atol=value_atol,
            rtol=value_rtol,
            equal_nan=False,
            err_msg=str(sample_index),
        )
        new_up, new_down = ru.copy(), rd.copy()
        moved = (new_up if moved_spin == 0 else new_down)[0] + np.asarray((1.13, -0.93, 2.0)) @ np.asarray(cell)
        (new_up if moved_spin == 0 else new_down)[0] = moved
        actual_ratio = jqmc_rs.jastrow_1b2b_ratio(jd, ru, rd, moved_spin == 0, 0, moved)
        reference_ratio = jastrow._compute_ratio_Jastrow_part_rank1_update(
            jd, up, down, jnp.asarray(new_up)[None, ...], jnp.asarray(new_down)[None, ...]
        )[0]
        np.testing.assert_allclose(
            actual_ratio, reference_ratio, atol=ratio_atol, rtol=ratio_rtol, equal_nan=False, err_msg=str(sample_index)
        )
    jax.clear_caches()


@pytest.mark.heavy
@pytest.mark.parametrize("j1_form,j2_form", [("exp", "pade"), ("pade", "exp")])
@pytest.mark.parametrize("trexio_file", ["H4_pbc_ae_cart.h5"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_native_skew_j12_force_stress_matches_xla(j1_form, j2_form, trexio_file):
    """Selected native log-wavefunction strain and kinetic force/stress match XLA.

    Reuse committed H4_pbc_ae_cart.h5 transformed by skewed_h4_ham onto
    its explicitly strongly skewed triclinic cell. Use exp/Pade and Pade/exp
    complementary J1/J2 pairs with unchanged parameters. far_start(seed=5)
    provides one unwrapped two-up/two-down configuration; there is no
    condition-number filtering or sampling acceptance window.

    Compare the complete log-wavefunction strain and kinetic nuclear/strain derivatives
    with the existing XLA analytical routes. All nine strain entries are
    asserted. These are selected derivative-route checks, not a claim that
    the complete local energy has an independent oracle in this test.
    Keep the existing medium propagated force/stress backend policies
    separately in each precision mode.
    """
    ham = _skew_j12_ham(j1_form, j2_form, trexio_file)
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    ru, rd = far_start(ham, seed=5)

    actual_log_strain = jqmc_rs.grad_lnPsi_pbc_wrt_strain(gd, jd, ru, rd)
    reference_log_strain = pbc_stress.grad_lndet_wrt_strain(gd, ru, rd) + pbc_stress.grad_lnJastrow_wrt_strain(jd, ru, rd)
    atol, rtol = get_tolerance_xla_rust_consistency(("ao_eval", "ao_grad_lap", "det_eval", "jastrow_eval"), level="medium")
    np.testing.assert_allclose(actual_log_strain, reference_log_strain, atol=atol, rtol=rtol, equal_nan=False)

    actual_kinetic_force = jqmc_rs.grad_T_pbc_wrt_R(gd, jd, ru, rd)
    reference_kinetic_force = grad_T_wrt_R_withJ(gd, jd, ru, rd)
    atol, rtol = get_tolerance_xla_rust_consistency(("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic"), level="medium")
    np.testing.assert_allclose(actual_kinetic_force, reference_kinetic_force, atol=atol, rtol=rtol, equal_nan=False)

    actual_kinetic_strain = jqmc_rs.grad_T_pbc_wrt_strain(gd, jd, ru, rd)
    reference_kinetic_strain = pbc_stress.stress_eL_kinetic_pbc(ham, ru, rd)
    atol, rtol = get_tolerance_xla_rust_consistency(
        ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "jastrow_eval", "jastrow_grad_lap"), level="medium"
    )
    np.testing.assert_allclose(actual_kinetic_strain, reference_kinetic_strain, atol=atol, rtol=rtol, equal_nan=False)
    jax.clear_caches()


@pytest.mark.heavy
@pytest.mark.parametrize("j1_form,j2_form", [("exp", "pade"), ("pade", "exp")])
@pytest.mark.parametrize("trexio_file", ["H4_pbc_ae_cart.h5"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_native_skew_j12_gfmc_measurement_derivatives_match_xla(j1_form, j2_form, trexio_file):
    """Native skew-cell GFMC measurement derivatives match the XLA analytic routes.

    Reuse the transformed H4_pbc_ae_cart.h5 strong-skew real-Gamma fixture,
    exp/Pade and Pade/exp J1/J2, and far_start(seed=5). The single unwrapped
    two-up/two-down configuration has no condition-number screening. Use
    alat=0.30, fixed identity rotation and the Nv12 quadrature with tmove;
    this all-electron fixture has no nonlocal ECP contribution.

    Assert nuclear and both electron derivatives and every strain component
    using the existing medium GFMC backend-consistency policy in each
    precision separately. These public native periodic measurement kernels
    consume the geometry contexts independently of the VMC derivative
    wrappers. XLA supplies the analytical reference; this test does not
    duplicate its independent physical oracle or run a statistical chain.
    """
    ham = _skew_j12_ham(j1_form, j2_form, trexio_file)
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    ru, rd = far_start(ham, seed=5)
    weights = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    points = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    args = (wf.geminal_data, wf.jastrow_data, cp, ru, rd, 0.30, np.eye(3), weights, points, 1, False)
    native_force = jqmc_rs._gfmc_grad_e_L_ecp_pbc(*args)
    native_stress = jqmc_rs._gfmc_stress_ecp_pbc_withj(*args)
    xla_force = gfmc.grad_e_L_t_wrt_positions(ham, jnp.asarray(ru), jnp.asarray(rd), jnp.eye(3), 0.30)
    xla_stress = gfmc.grad_e_L_t_wrt_strain(ham, jnp.asarray(ru), jnp.asarray(rd), jnp.eye(3), 0.30)
    atol, rtol = get_tolerance_xla_rust_consistency(
        ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb", "jastrow_eval", "jastrow_grad_lap"),
        level="medium",
    )
    for quantity, actual, reference in zip(
        ("nuclear", "electron-up", "electron-down", "strain"),
        (*native_force, native_stress),
        (*xla_force, xla_stress),
        strict=True,
    ):
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=quantity)
    jax.clear_caches()
