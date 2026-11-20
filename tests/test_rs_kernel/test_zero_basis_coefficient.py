"""Basis derivatives of a finite wavefunction at an exactly zero coefficient."""

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
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.derivatives.basis import grad_J3_wrt_basis, grad_ln_det_wrt_basis, grad_T_wrt_basis, grad_T_wrt_J3_basis
from jqmc.determinant import compute_ln_det_geminal_all_elements
from jqmc.jastrow_factor import Jastrow_three_body_data, compute_Jastrow_three_body
from jqmc.molecular_orbital import MOs_data
from jqmc.wavefunction import compute_kinetic_energy

from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_param_grad import _config

pytestmark = pytest.mark.with_chdir

_PATTERNS = ["mol_ecp_cart_j123", "mol_sphe_sd_jnone", "twist_sphepgto_j123"]


def _aos(orb):
    return orb.aos_data if isinstance(orb, MOs_data) else orb


def _with_radial(orb, radial):
    aos = _aos(orb).replace(radial=radial)
    return orb.replace(aos_data=aos) if isinstance(orb, MOs_data) else aos


def _active_config(ham):
    ru, rd = _config(ham)
    if not ham.structure_data.pbc_flag:
        # Keep the H2 walkers near the first shell so its zero-coefficient
        # sensitivity is resolved in both precision modes.
        center = np.asarray(ham.structure_data.positions)[0]
        return center + 0.25 * (ru - center), center + 0.25 * (rd - center)
    return ru, rd


def _check_derivatives(observable, radial, xla, rust):
    # AD of the forward observable with the same finite image list and precision.
    # Other live coefficients serve as nonzero controls.
    reference = jax.grad(observable)(radial)
    comparison_reference = np.broadcast_to(
        np.stack((reference.exponents, reference.coefficients)), (2, 2, radial.exponents.size)
    )
    comparison_actual = np.stack((xla, rust))
    atol, rtol = get_tolerance_xla_rust_consistency(
        ("ao_eval", "det_eval", "mo_grad", "mo_lap", "det_grad_lap", "jastrow_grad_lap", "wf_kinetic"),
        level="medium",
    )
    assert np.isfinite(observable(radial))
    assert abs(reference.coefficients[0]) > atol
    assert np.any(np.abs(reference.coefficients[1:]) > atol)
    assert reference.exponents[0] == 0
    np.testing.assert_allclose(comparison_actual, comparison_reference, atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("pattern", _PATTERNS)
@pytest.mark.parametrize("quantity", ["value", "kinetic"])
def test_zero_geminal_coefficient(pattern, quantity):
    """Geminal coefficient gradients at zero agree with the forward AD oracle."""
    ham = rs_ham(pattern)
    ru, rd = _active_config(ham)
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    orb = gd.orb_data_up_spin
    radial = _aos(orb).radial
    radial = radial.replace(coefficients=jnp.asarray(radial.coefficients).at[0].set(0))

    def geminal(p):
        return gd.replace(orb_data_up_spin=_with_radial(orb, p))

    actual_gd = geminal(radial)
    jd = wf.jastrow_data
    if quantity == "value":

        def observable(p):
            return compute_ln_det_geminal_all_elements(geminal(p), ru, rd)

        xla = grad_ln_det_wrt_basis(actual_gd, ru, rd)[0]
        native = jqmc_rs.grad_ln_det_wrt_radial_complex if pattern.startswith("twist") else jqmc_rs.grad_ln_det_wrt_radial
        rust = native(actual_gd, ru, rd)["orb_data_up_spin"]
    else:

        def observable(p):
            return jnp.real(compute_kinetic_energy(wf.replace(geminal_data=geminal(p)), ru, rd))

        xla = grad_T_wrt_basis(actual_gd, jd, ru, rd)[0]
        native = jqmc_rs.grad_eL_wrt_radial_complex if pattern.startswith("twist") else jqmc_rs.grad_eL_wrt_radial
        rust = native(actual_gd, ru, rd, jd)["orb_data_up_spin"]
    _check_derivatives(observable, radial, xla, rust)


@pytest.mark.light
@pytest.mark.parametrize("pattern", _PATTERNS)
@pytest.mark.parametrize("quantity", ["value", "kinetic"])
def test_zero_j3_coefficient(pattern, quantity):
    """Three-body Jastrow coefficient gradients at zero agree with forward AD."""
    ham = rs_ham(pattern)
    ru, rd = _active_config(ham)
    wf = ham.wavefunction_data
    jd = wf.jastrow_data
    j3 = jd.jastrow_three_body_data
    if j3 is None:
        # A molecular bare spherical AO leaf with nonzero one-electron and pair terms.
        orb = _aos(wf.geminal_data.orb_data_up_spin)
        size = orb.num_ao
        matrix = np.column_stack((0.3 * np.eye(size), np.full(size, 0.2)))
        j3 = Jastrow_three_body_data(orb_data=orb, j_matrix=matrix)
    orb = j3.orb_data
    radial = _aos(orb).radial
    radial = radial.replace(coefficients=jnp.asarray(radial.coefficients).at[0].set(0))

    def jastrow(p):
        return jd.replace(jastrow_three_body_data=j3.replace(orb_data=_with_radial(orb, p)))

    actual_jd = jastrow(radial)
    if quantity == "value":

        def observable(p):
            return compute_Jastrow_three_body(jastrow(p).jastrow_three_body_data, ru, rd)

        xla = grad_J3_wrt_basis(actual_jd, ru, rd)
        rust = jqmc_rs.grad_ln_Psi_jastrow_3b_radial_complex(actual_jd, ru, rd)
    else:

        def observable(p):
            return jnp.real(compute_kinetic_energy(wf.replace(jastrow_data=jastrow(p)), ru, rd))

        xla = grad_T_wrt_J3_basis(wf.geminal_data, actual_jd, ru, rd)
        rust = jqmc_rs.grad_eL_wrt_jastrow_3b_radial_complex(wf.geminal_data, actual_jd, ru, rd)
    _check_derivatives(observable, radial, xla, rust)
