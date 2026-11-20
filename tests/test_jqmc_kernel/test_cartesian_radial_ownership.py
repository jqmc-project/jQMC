"""Physical shell identity for normalized molecular Cartesian TREXIO AOs."""

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

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE
from jqmc.atomic_orbital import _aos_cart_to_sphe, _aos_sphe_to_cart
from jqmc.derivatives.basis import grad_J3_wrt_basis
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests.test_jqmc_kernel.test_trexio_source_values import _basis_reference, _raw_fields

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

FIXTURE = Path(__file__).resolve().parents[1] / "trexio_example_files" / "Cl2_ecp_ccpvtz_cart.h5"
R_UP = np.array([[0.31, 0.43, 0.57], [1.12, -0.62, 0.91]])
R_DN = R_UP + np.array([0.13, 0.21, -0.17])


def _updated_shells(parameter):
    structure, aos, _, _, determinant, coulomb = read_trexio_file(str(FIXTURE), store_tuple=True)
    raw = _raw_fields(FIXTURE)
    # Compare every effective input coefficient with independently normalized
    # TREXIO source records before applying any optimizer update.
    stored, source = _basis_reference(aos, raw)
    np.testing.assert_allclose(stored, source, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)

    raw_shell = np.asarray(raw["prim_shell"])
    owner = np.asarray(aos.orbital_indices)
    ao_shell = np.asarray(raw["ao_shell"])
    raw_primitive = np.empty(len(owner), dtype=np.int64)
    for ao, shell in enumerate(ao_shell):
        raw_primitive[owner == ao] = np.flatnonzero(raw_shell == shell)
    # The exact TREXIO IDs, not equality of exponents or component-scaled
    # coefficients, determine the partition. Allow either ordering of DOFs.
    gather = np.asarray(aos.ao_prim_to_shell_prim)
    radial_of_source = np.empty(len(raw_shell), dtype=np.int64)
    radial_of_source[raw_primitive] = gather
    np.testing.assert_array_equal(gather, radial_of_source[raw_primitive])
    assert np.unique(radial_of_source).size == aos.radial.num_shell_prim == len(raw_shell)

    target_primitives = [
        int(np.flatnonzero(raw_shell == np.flatnonzero((raw["shell_l"] == l) & (raw["shell_atom"] == 0))[0])[0]) for l in (2, 3)
    ]
    blocks = aos.get_basis_variational_blocks(
        opt_basis_exp=parameter in ("exp", "combined"),
        opt_basis_coeff=parameter in ("coeff", "combined"),
        block_name_prefix="",
        state_token=("cartesian-physical-shell", 0),
        sync_scope="sampler",
    )
    exp_factors = np.ones(len(raw_shell))
    coeff_factors = np.ones(len(raw_shell))
    candidate = aos
    for block in blocks:
        values = np.array(block.values, copy=True)
        factors = exp_factors if block.name == "basis_exp" else coeff_factors
        factors[target_primitives] = 1 + 1 / 64
        values[radial_of_source[target_primitives]] *= factors[target_primitives]
        candidate = candidate.apply_basis_block_update(block.replace(values=values))
    np.testing.assert_array_equal(candidate.exponents, aos.exponents * exp_factors[raw_primitive])
    np.testing.assert_allclose(
        candidate.coefficients,
        aos.coefficients * coeff_factors[raw_primitive],
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_LOOSE,
    )
    # A Cartesian coefficient in the source multiplies an unnormalized
    # polynomial. The Gaussian normalization varies as alpha**(l/2+3/4)
    # when the physical exponent changes, with angular factors fixed.
    points = np.concatenate((R_UP, R_DN))
    displacement = points[None] - np.asarray(raw["positions"])[np.asarray(raw["shell_atom"])[ao_shell[owner]]][:, None]
    powers = np.column_stack((aos.polynominal_order_x, aos.polynominal_order_y, aos.polynominal_order_z))[owner]
    polynomial = np.prod(displacement ** powers[:, None], axis=-1)
    l = np.asarray(raw["shell_l"])[raw_shell[raw_primitive]]
    weights = source * coeff_factors[raw_primitive] * exp_factors[raw_primitive] ** (l / 2 + 0.75)
    exponent = np.asarray(raw["exponents"])[raw_primitive] * exp_factors[raw_primitive]
    primitives = weights[:, None] * polynomial * np.exp(-exponent[:, None] * np.sum(displacement**2, axis=-1))
    reference_values = np.zeros((aos.num_ao, len(points)))
    np.add.at(reference_values, owner, primitives)
    assert np.any(reference_values)

    determinant = determinant.replace(
        orb_data_up_spin=determinant.orb_data_up_spin.replace(aos_data=candidate),
        orb_data_dn_spin=determinant.orb_data_dn_spin.replace(aos_data=candidate),
    )
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(candidate)
    matrix = np.zeros(j3.j_matrix.shape)
    matrix[:, :-1] = np.eye(candidate.num_ao) / 16
    matrix[:, -1] = np.linspace(-0.05, 0.09, candidate.num_ao)
    jd = Jastrow_data(jastrow_three_body_data=j3.replace(j_matrix=matrix))
    ham = Hamiltonian_data(
        structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=Wavefunction_data(geminal_data=determinant)
    )
    return candidate, ham, jd, points, reference_values


def _j3_autodiff(aos, points, matrix):
    """Explicit electron-pair contraction, differentiated through the AO forward path."""
    rows, cols = np.triu_indices(len(points), 1)

    def evaluate(radial):
        values = aos.replace(radial=radial)._compute_AOs(points)
        pairs = values.T @ jnp.asarray(matrix[:, :-1]) @ values
        return jnp.sum(pairs[rows, cols]) + jnp.sum(jnp.asarray(matrix[:, -1]) @ values)

    gradient = jax.grad(evaluate)(aos.radial)
    return gradient.exponents, gradient.coefficients


@pytest.mark.parametrize("parameter", ["exp", "coeff", "combined"])
@pytest.mark.gdd_electron_configuration(
    mode="required", position_source="explicit", reason="Fixed off-nucleus molecular AO and J3 evaluation coordinates."
)
def test_cartesian_trexio_physical_shell_update(tmp_path, parameter):
    """Raw physical IDs, normalized AO values, and checkpoint/transform preserve one shell DOF."""
    aos, ham, jd, points, reference_values = _updated_shells(parameter)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval"), "strict")
    values = aos._compute_AOs(points)
    np.testing.assert_allclose(values, reference_values, atol=atol, rtol=rtol)
    actual_gradients = grad_J3_wrt_basis(jd, R_UP, R_DN)
    expected_gradients = _j3_autodiff(aos, points, jd.jastrow_three_body_data.j_matrix)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        assert np.any(expected)
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)

    path = tmp_path / "physical-shell.h5"
    ham.save_to_hdf5(path)
    restored = Hamiltonian_data.load_from_hdf5(path).wavefunction_data.geminal_data.orb_data_up_spin.aos_data
    assert restored.ao_coefficient_scales == aos.ao_coefficient_scales
    assert restored.ao_prim_to_shell_prim == aos.ao_prim_to_shell_prim
    np.testing.assert_array_equal(restored.radial.exponents, aos.radial.exponents, strict=True)
    np.testing.assert_array_equal(restored.radial.coefficients, aos.radial.coefficients, strict=True)
    np.testing.assert_array_equal(restored._compute_AOs(points), values)

    if parameter == "combined":
        spherical, _ = _aos_cart_to_sphe(restored)
        cartesian, embedding = _aos_sphe_to_cart(spherical)
        for converted in (spherical, cartesian):
            np.testing.assert_array_equal(converted.radial.exponents, aos.radial.exponents)
            np.testing.assert_array_equal(converted.radial.coefficients, aos.radial.coefficients)
            assert set(converted.ao_prim_to_shell_prim) == set(aos.ao_prim_to_shell_prim)
        np.testing.assert_allclose(
            embedding @ cartesian._compute_AOs(points), spherical._compute_AOs(points), atol=atol, rtol=rtol
        )
