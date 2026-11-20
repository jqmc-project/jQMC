"""Native consumers must follow the physical periodic radial gather after updates."""

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

import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import get_tolerance_min
from tests.test_jqmc_kernel._periodic_j3_basis_cases import R_DN, R_UP, gaussian_lattice_reference, j3_leaf
from tests.test_jqmc_kernel.test_periodic_j3_basis_updates import update_case

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]


@pytest.mark.parametrize("angular", ["cart", "sphe"])
@pytest.mark.parametrize("wrapper", ["ao", "mo"])
@pytest.mark.gdd_electron_configuration(
    mode="required", position_source="explicit", reason="Unscreened fixed audit off-nucleus electron positions."
)
def test_native_periodic_j3_physical_radial_derivatives(angular, wrapper):
    """Accepted combined updates use one alpha/c DOF per physical shell in native code.

    The independent Gaussian lattice oracle and explicit J3 pair contraction
    provide the physical value and both radial derivatives. This tests the
    actual native radial scatter reductions, not only parity between two
    copies of an expanded-image parameter layout.
    """
    _, ham, _ = update_case(angular, wrapper, "combined")
    jd = ham.wavefunction_data.jastrow_data
    j3 = jd.jastrow_three_body_data
    leaf = j3_leaf(ham)
    assert len(leaf.radial.exponents) == 6
    reference, gradients = gaussian_lattice_reference(j3, angular)
    native_ao_up = np.asarray(jqmc_rs.compute_AOs_exact(leaf, R_UP))
    native_ao_dn = np.asarray(jqmc_rs.compute_AOs_exact(leaf, R_DN))
    if wrapper == "mo":
        native_ao_up = np.asarray(j3.orb_data.mo_coefficients) @ native_ao_up
        native_ao_dn = np.asarray(j3.orb_data.mo_coefficients) @ native_ao_dn
    actual = jqmc_rs.compute_Jastrow_three_body_value(
        np.asarray(j3.j_matrix, dtype=np.float64),
        np.asarray(native_ao_up, dtype=np.float64),
        np.asarray(native_ao_dn, dtype=np.float64),
    )
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval"), "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
    actual_gradients = jqmc_rs.grad_ln_Psi_jastrow_3b_radial_complex(jd, R_UP, R_DN)
    for actual_gradient, expected in zip(actual_gradients, gradients, strict=True):
        assert np.isfinite(actual_gradient).all() and np.any(np.asarray(actual_gradient) != 0)
        np.testing.assert_allclose(actual_gradient, expected, atol=atol, rtol=rtol)


@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="production-driver-generated",
    reason="Actual native MCMC seed718 and checkpoint continuation without reseeding.",
)
def test_native_audit_ordinary_supercell_j3_update_and_restart(tmp_path, monkeypatch):
    """Native MCMC produces physical J3 basis derivatives and rebuilds contexts after acceptance."""
    import jqmc.jqmc_mcmc as mcmc_module
    from tests.test_jqmc_kernel.test_periodic_j3_audit_optimization import (
        test_audit_ordinary_supercell_j3_update_and_restart as audit_case,
    )

    original = mcmc_module._rs_native_c_param_grads
    produced = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        produced.append(tuple(source.name for source in result))
        return result

    monkeypatch.setattr(mcmc_module, "_rs_native_c_param_grads", observe)
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    audit_case(tmp_path)
    assert any("j3_basis_exp" in record for record in produced), "The actual native derivative producer must execute."
