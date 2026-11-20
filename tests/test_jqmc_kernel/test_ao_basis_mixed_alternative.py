"""Full AD and mixed full-route fidelity complement the independent basis FD anchors."""

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

from jqmc import derivatives
from jqmc._precision import get_tolerance, is_mixed_precision_enabled
from tests._precision_fidelity import mixed_vs_full
from tests.test_jqmc_kernel._orbital_mixed_fidelity import assert_mixed_parameter_derivative_fidelity
from jqmc.determinant import Geminal_data, compute_det_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.wavefunction import Wavefunction_data
from tests.test_jqmc_kernel.test_ao_basis_optimization import _load_trexio, _random_electron_coords

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


def _ao_geminal_case(trexio_file):
    structure, _, _, _, source_geminal, coulomb = _load_trexio(trexio_file)
    geminal = Geminal_data.convert_from_MOs_to_AOs(source_geminal)
    r_up, r_dn = _random_electron_coords(structure, coulomb, geminal, seed=42)
    wf = Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=geminal)
    ham = Hamiltonian_data(structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=wf)
    return ham, geminal, r_up, r_dn


def _ao_exponent_gradient(case):
    ham, geminal, r_up, r_dn = case
    gradient = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {"opt_lambda_basis_exp": True}, analytic=True)["c"]
    actual = np.asarray(gradient.geminal_data.ao_exponents_up)
    # Preserve the original input-model shape and nonempty gate in both policies.
    assert actual.shape == geminal.ao_exponents_up.shape and actual.size > 0
    return actual


@pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_ao_converted_geminal_exponent_gradient_matches_primal_ad(trexio_file):
    """AO-geminal up-spin exponent gradients equal same-precision primal log-det AD.

    The committed H2_ae_ccpvdz_cart.h5 is converted from its MO determinant to
    an AO Geminal_data, without adding Jastrow or perturbing its lambda matrix.
    One 1-up/1-down configuration uses the FD sentinel's RandomState seed 42
    nuclear-centred generator, without cond(G) rejection. Only up-spin radial
    exponents vary in the reference; down-spin exponents stay fixed. The public
    analytical ln-Psi basis derivative is compared to jax.grad of ln|det G|,
    in full precision using the existing strict det_eval band. Mixed instead
    compares this production gradient to the same route on independently
    rebuilt full inputs, using the central parameter-derivative fidelity budget.
    Nonempty, finite and live blocks complement the unchanged full-only FD sentinel.
    """
    if is_mixed_precision_enabled():
        mixed, full = mixed_vs_full(lambda: _ao_geminal_case(trexio_file), _ao_exponent_gradient)
        atol, rtol = get_tolerance("det_eval", "strict")
        assert np.any(np.abs(full) > atol + rtol * np.abs(full)), "zero gradient would pass the comparison"
        assert_mixed_parameter_derivative_fidelity(mixed, full, "AO geminal exponent mixed vs full")
        return

    ham, geminal, r_up, r_dn = _ao_geminal_case(trexio_file)
    actual = _ao_exponent_gradient((ham, geminal, r_up, r_dn))

    def logdet(exponents_up):
        changed = geminal.with_updated_ao_exponents(exponents_up, geminal.ao_exponents_dn)
        return jnp.log(jnp.abs(compute_det_geminal_all_elements(changed, r_up, r_dn)))

    reference = np.asarray(jax.grad(logdet)(geminal.ao_exponents_up))
    atol, rtol = get_tolerance("det_eval", "strict")
    assert actual.shape == reference.shape == geminal.ao_exponents_up.shape and reference.size > 0
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    assert np.any(np.abs(reference) > atol + rtol * np.abs(reference)), "zero gradient would pass the comparison"
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
