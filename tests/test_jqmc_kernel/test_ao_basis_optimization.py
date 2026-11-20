"""Tests for AO basis optimization (exponents/coefficients as variational parameters)."""

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
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from tests.test_jqmc_kernel._gdd_fd_adapters import fd_case_parameters, record_fd_case

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance
from jqmc.determinant import Geminal_data, compute_det_geminal_all_elements
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_three_body_data,
    compute_Jastrow_three_body,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import (
    Wavefunction_data,
    evaluate_ln_wavefunction,
)
from tests.test_jqmc_kernel.optimizer_metadata_helpers import make_test_block

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


TREXIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")


def _load_trexio(name="H2_ae_ccpvdz_cart.h5"):
    """Load a TREXIO file and return all data."""
    return read_trexio_file(
        trexio_file=os.path.join(TREXIO_DIR, name),
        store_tuple=True,
    )


def _random_electron_coords(structure_data, coulomb_potential_data, geminal_data, seed=42):
    """Generate random electron positions near nuclei."""
    rng = np.random.RandomState(seed)
    num_up = geminal_data.num_electron_up
    num_dn = geminal_data.num_electron_dn

    if coulomb_potential_data.ecp_flag:
        charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
    else:
        charges = np.array(structure_data.atomic_numbers)

    coords = structure_data._positions_cart_np

    electrons = []
    for i in range(len(coords)):
        charge = charges[i]
        num_e = int(np.round(charge))
        for _ in range(num_e):
            dist = rng.uniform(0.3 / max(charge, 1), 1.2 / max(charge, 1))
            theta = rng.uniform(0, np.pi)
            phi = rng.uniform(0, 2 * np.pi)
            dx = dist * np.sin(theta) * np.cos(phi)
            dy = dist * np.sin(theta) * np.sin(phi)
            dz = dist * np.cos(theta)
            electrons.append(coords[i] + np.array([dx, dy, dz]))

    electrons = np.array(electrons)
    r_up = jnp.array(electrons[:num_up], dtype=jnp.float64)
    r_dn = jnp.array(electrons[num_up : num_up + num_dn], dtype=jnp.float64)
    return r_up, r_dn


# ============================================================
# Test 1: exponents/coefficients field type validation
# ============================================================


class TestAOBasisOptimizationMisc:
    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5", "H2_ae_ccpvdz_sphe.h5"])
    def test_exponents_coefficients_storage_is_numpy(self, trexio_file):
        """Storage fields exponents/coefficients are np.ndarray[float64] (jnp view via _*_jnp)."""
        structure_data, aos_data, *_ = _load_trexio(trexio_file)
        assert isinstance(aos_data.exponents, np.ndarray), f"exponents type: {type(aos_data.exponents)}"
        assert aos_data.exponents.dtype == np.float64, f"exponents dtype: {aos_data.exponents.dtype}"
        assert isinstance(aos_data.coefficients, np.ndarray), f"coefficients type: {type(aos_data.coefficients)}"
        assert aos_data.coefficients.dtype == np.float64, f"coefficients dtype: {aos_data.coefficients.dtype}"
        # The jnp accessor returns jax.Array
        assert isinstance(aos_data._exponents_jnp, jax.Array)
        assert isinstance(aos_data._coefficients_jnp, jax.Array)

    # ============================================================
    # Test 2: J3 ao_exponents/ao_coefficients property (AO path)
    # ============================================================

    @pytest.mark.light
    def test_j3_ao_properties_with_ao_data(self):
        """J3 ao_exponents/ao_coefficients should work with direct AO data."""
        _, aos_data, _, _, geminal_mo_data, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
        j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)
        npt.assert_array_equal(np.array(j3.ao_exponents), np.array(aos_data.radial.exponents))
        npt.assert_array_equal(np.array(j3.ao_coefficients), np.array(aos_data.radial.coefficients))

    # ============================================================
    # Test 3: J3 ao_exponents/ao_coefficients property (MO path)
    # ============================================================

    @pytest.mark.light
    def test_j3_ao_properties_with_mo_data(self):
        """J3 ao_exponents/ao_coefficients should work with MO data."""
        _, _, mos_data_up, _, _, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
        j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=mos_data_up)
        npt.assert_array_equal(np.array(j3.ao_exponents), np.array(mos_data_up.aos_data.radial.exponents))
        npt.assert_array_equal(np.array(j3.ao_coefficients), np.array(mos_data_up.aos_data.radial.coefficients))

    # ============================================================
    # Test 4: J3 with_updated_ao_exponents / with_updated_ao_coefficients
    # ============================================================

    @pytest.mark.light
    def test_j3_with_updated_ao_exponents(self):
        """with_updated_ao_exponents should produce a new J3 with updated exponents."""
        _, aos_data, _, _, _, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
        j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)

        new_exp = j3.ao_exponents * 1.1
        j3_new = j3.with_updated_ao_exponents(new_exp)
        npt.assert_allclose(
            np.array(j3_new.ao_exponents),
            np.array(new_exp),
            atol=get_tolerance("ao_eval", "strict")[0],
            rtol=get_tolerance("ao_eval", "strict")[1],
        )
        # Original should be unchanged
        npt.assert_allclose(
            np.array(j3.ao_exponents),
            np.array(aos_data.radial.exponents),
            atol=get_tolerance("ao_eval", "strict")[0],
            rtol=get_tolerance("ao_eval", "strict")[1],
        )

    # ============================================================
    # Test 5: Geminal ao_exponents_up/dn properties
    # ============================================================

    @pytest.mark.light
    def test_geminal_ao_properties(self):
        """Geminal ao_exponents_up/dn should return the correct exponents."""
        _, _, _, _, geminal_mo_data, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")

        # MO representation
        exp_up = geminal_mo_data.ao_exponents_up
        exp_dn = geminal_mo_data.ao_exponents_dn
        assert isinstance(exp_up, jax.Array)
        assert isinstance(exp_dn, jax.Array)

        # Convert to AO representation and check
        geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
        exp_up_ao = geminal_ao.ao_exponents_up
        exp_dn_ao = geminal_ao.ao_exponents_dn
        npt.assert_allclose(
            np.array(exp_up),
            np.array(exp_up_ao),
            atol=get_tolerance("ao_eval", "strict")[0],
            rtol=get_tolerance("ao_eval", "strict")[1],
        )
        npt.assert_allclose(
            np.array(exp_dn),
            np.array(exp_dn_ao),
            atol=get_tolerance("ao_eval", "strict")[0],
            rtol=get_tolerance("ao_eval", "strict")[1],
        )

    # ============================================================
    # Test 6: Geminal with_updated_ao_exponents
    # ============================================================

    @pytest.mark.light
    def test_geminal_with_updated_ao_exponents(self):
        """with_updated_ao_exponents should produce correct updates for both spins."""
        _, _, _, _, geminal_mo_data, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
        geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)

        new_exp_up = geminal_ao.ao_exponents_up * 0.9
        new_exp_dn = geminal_ao.ao_exponents_dn * 1.1
        geminal_new = geminal_ao.with_updated_ao_exponents(new_exp_up, new_exp_dn)

        npt.assert_allclose(
            np.array(geminal_new.ao_exponents_up),
            np.array(new_exp_up),
            atol=get_tolerance("ao_eval", "strict")[0],
            rtol=get_tolerance("ao_eval", "strict")[1],
        )
        npt.assert_allclose(
            np.array(geminal_new.ao_exponents_dn),
            np.array(new_exp_dn),
            atol=get_tolerance("ao_eval", "strict")[0],
            rtol=get_tolerance("ao_eval", "strict")[1],
        )
        # Lambda matrix should be unchanged
        npt.assert_array_equal(np.array(geminal_new.lambda_matrix), np.array(geminal_ao.lambda_matrix))

    # ============================================================
    # Test 7: Gradient of J3 with respect to AO exponents (finite difference)
    # ============================================================

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-DERIVAPI-CONSISTENCY-001",
    )
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5", "H2_ae_ccpvdz_sphe.h5"])
    def test_j3_exponent_gradient_finite_diff(self, trexio_file):
        """Verify that jax.grad of J3 w.r.t. exponents matches finite differences."""
        structure_data, aos_data, _, _, geminal_mo_data, coulomb_potential_data = _load_trexio(trexio_file)
        r_up, r_dn = _random_electron_coords(structure_data, coulomb_potential_data, geminal_mo_data)

        j3 = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=aos_data, random_init=True, random_scale=0.01, seed=42
        )

        def j3_value(exponents):
            j3_mod = j3.with_updated_ao_exponents(exponents)
            return compute_Jastrow_three_body(j3_mod, r_up, r_dn)

        # JAX gradient (w.r.t. the per-shell radial exponents -- the DOF)
        grad_fn = jax.grad(j3_value)
        grad_jax = grad_fn(jnp.asarray(aos_data.radial.exponents))

        # Finite difference gradient
        eps = 1e-5
        grad_fd = np.zeros_like(np.array(aos_data.radial.exponents))
        exp0 = np.array(aos_data.radial.exponents, dtype=np.float64)
        for i in range(len(exp0)):
            exp_plus = exp0.copy()
            exp_plus[i] += eps
            exp_minus = exp0.copy()
            exp_minus[i] -= eps
            f_plus = j3_value(jnp.array(exp_plus))
            f_minus = j3_value(jnp.array(exp_minus))
            grad_fd[i] = (float(f_plus) - float(f_minus)) / (2 * eps)

        atol, rtol = get_tolerance("jastrow_eval", "strict")
        npt.assert_allclose(np.array(grad_jax), grad_fd, atol=atol, rtol=rtol)
        record_fd_case(
            "fd-ad-j3-exponents",
            pytest_node_id="tests/test_jqmc_kernel/test_ao_basis_optimization.py::TestAOBasisOptimizationMisc::test_j3_exponent_gradient_finite_diff",
            fixture_parameters=fd_case_parameters("fd-ad-j3-exponents", trexio_file=trexio_file),
            source_ids=(("WF-SRC-H2-AE-CART-MOL",) if "cart" in trexio_file else ("WF-SRC-H2-AE-SPHE-MOL",)),
            transformations=("derivatives.fd.initialize-random-j3.v1",),
            coordinates=(r_up, r_dn),
            coordinate_generator="tests.test_jqmc_kernel.test_ao_basis_optimization._random_electron_coords",
            coordinate_seed=42,
            compared_pairs=((grad_jax, grad_fd),),
            atol=atol,
            rtol=rtol,
            zones=("jastrow_eval",),
            level="strict",
            stencil="central-first-order-2",
            step=eps,
            direction="parameter-basis",
        )

    # ============================================================
    # Test 8: Gradient of J3 with respect to AO coefficients (finite difference)
    # ============================================================

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-DERIVAPI-CONSISTENCY-001",
    )
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5", "H2_ae_ccpvdz_sphe.h5"])
    def test_j3_coefficient_gradient_finite_diff(self, trexio_file):
        """Differentiate J3 along each physical radial contraction coefficient."""
        structure_data, aos_data, _, _, geminal_mo_data, coulomb_potential_data = _load_trexio(trexio_file)
        r_up, r_dn = _random_electron_coords(structure_data, coulomb_potential_data, geminal_mo_data)

        j3 = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=aos_data, random_init=True, random_scale=0.01, seed=42
        )

        def j3_value(coefficients):
            j3_mod = j3.with_updated_ao_coefficients(coefficients)
            return compute_Jastrow_three_body(j3_mod, r_up, r_dn)

        grad_fn = jax.grad(j3_value)
        radial_coefficients = aos_data.radial.coefficients
        grad_jax = grad_fn(radial_coefficients)

        eps = 1e-5
        coeff0 = np.array(radial_coefficients, dtype=np.float64)
        grad_fd = np.zeros_like(coeff0)
        for i in range(len(coeff0)):
            c_plus = coeff0.copy()
            c_plus[i] += eps
            c_minus = coeff0.copy()
            c_minus[i] -= eps
            f_plus = j3_value(jnp.array(c_plus))
            f_minus = j3_value(jnp.array(c_minus))
            grad_fd[i] = (float(f_plus) - float(f_minus)) / (2 * eps)

        atol, rtol = get_tolerance("jastrow_eval", "strict")
        npt.assert_allclose(np.array(grad_jax), grad_fd, atol=atol, rtol=rtol)
        record_fd_case(
            "fd-ad-j3-coefficients",
            pytest_node_id="tests/test_jqmc_kernel/test_ao_basis_optimization.py::TestAOBasisOptimizationMisc::test_j3_coefficient_gradient_finite_diff",
            fixture_parameters=fd_case_parameters("fd-ad-j3-coefficients", trexio_file=trexio_file),
            source_ids=(("WF-SRC-H2-AE-CART-MOL",) if "cart" in trexio_file else ("WF-SRC-H2-AE-SPHE-MOL",)),
            transformations=("derivatives.fd.initialize-random-j3.v1",),
            coordinates=(r_up, r_dn),
            coordinate_generator="tests.test_jqmc_kernel.test_ao_basis_optimization._random_electron_coords",
            coordinate_seed=42,
            compared_pairs=((grad_jax, grad_fd),),
            atol=atol,
            rtol=rtol,
            zones=("jastrow_eval",),
            level="strict",
            stencil="central-first-order-2",
            step=eps,
            direction="parameter-basis",
        )

    # ============================================================
    # Test 9: Gradient of Geminal Det w.r.t. AO exponents (finite difference)
    # ============================================================

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-DERIVAPI-CONSISTENCY-001",
    )
    def test_geminal_exponent_gradient_finite_diff(self):
        """Verify that jax.grad of Geminal det w.r.t. exponents matches finite differences."""
        structure_data, _, _, _, geminal_mo_data, coulomb_potential_data = _load_trexio("H2_ae_ccpvdz_cart.h5")
        geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
        r_up, r_dn = _random_electron_coords(structure_data, coulomb_potential_data, geminal_ao)

        def det_value(exponents_up):
            g = geminal_ao.with_updated_ao_exponents(exponents_up, geminal_ao.ao_exponents_dn)
            return jnp.log(jnp.abs(compute_det_geminal_all_elements(g, r_up, r_dn)))

        grad_fn = jax.grad(det_value)
        grad_jax = grad_fn(geminal_ao.ao_exponents_up)

        eps = 1e-5
        exp0 = np.array(geminal_ao.ao_exponents_up, dtype=np.float64)
        grad_fd = np.zeros_like(exp0)
        for i in range(len(exp0)):
            e_plus = exp0.copy()
            e_plus[i] += eps
            e_minus = exp0.copy()
            e_minus[i] -= eps
            f_plus = det_value(jnp.array(e_plus))
            f_minus = det_value(jnp.array(e_minus))
            grad_fd[i] = (float(f_plus) - float(f_minus)) / (2 * eps)

        atol, rtol = get_tolerance("det_eval", "strict")
        npt.assert_allclose(np.array(grad_jax), grad_fd, atol=atol, rtol=rtol)
        record_fd_case(
            "fd-ad-geminal-exponents",
            pytest_node_id="tests/test_jqmc_kernel/test_ao_basis_optimization.py::TestAOBasisOptimizationMisc::test_geminal_exponent_gradient_finite_diff",
            fixture_parameters=fd_case_parameters("fd-ad-geminal-exponents"),
            source_ids=("WF-SRC-H2-AE-CART-MOL",),
            transformations=("derivatives.fd.convert-geminal-mo-to-ao.v1",),
            coordinates=(r_up, r_dn),
            coordinate_generator="tests.test_jqmc_kernel.test_ao_basis_optimization._random_electron_coords",
            coordinate_seed=42,
            compared_pairs=((grad_jax, grad_fd),),
            atol=atol,
            rtol=rtol,
            zones=("det_eval",),
            level="strict",
            stencil="central-first-order-2",
            step=eps,
            direction="parameter-basis",
        )

    # ============================================================
    # Test 10: get_variational_blocks with new basis flags
    # ============================================================

    @pytest.mark.light
    def test_get_variational_blocks_basis_flags(self):
        """get_variational_blocks should include/exclude basis blocks based on flags."""
        structure_data, aos_data, mos_data_up, mos_data_dn, geminal_mo_data, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
        geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)

        j3 = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=aos_data, random_init=True, random_scale=0.01, seed=42
        )
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=None,
            jastrow_three_body_data=j3,
        )
        wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_ao)

        # Default (all basis flags False)
        blocks = wf.get_variational_blocks(state_token=("test", 0), opt_J3_param=True, opt_geminal=True)
        block_names = [b.name for b in blocks]
        assert "j3_matrix" in block_names
        assert "lambda_matrix" in block_names
        assert "j3_basis_exp" not in block_names
        assert "j3_basis_coeff" not in block_names
        assert "lambda_basis_exp" not in block_names
        assert all(block.state_token == ("test", 0) for block in blocks)
        j3_block = next(block for block in blocks if block.name == "j3_matrix")
        lambda_block = next(block for block in blocks if block.name == "lambda_matrix")
        assert (j3_block.field, j3_block.derivative_policy, j3_block.sync_scope) == (
            "real",
            "direct_real",
            "world",
        )
        assert j3_block.tangent_spec.kind == "linear_tie"
        assert (lambda_block.field, lambda_block.derivative_policy, lambda_block.sync_scope) == (
            "real",
            "direct_real",
            "sampler",
        )
        assert lambda_block.tangent_spec.kind == "linear_tie"

        # With basis flags
        blocks = wf.get_variational_blocks(
            state_token=("test", 0),
            opt_J3_param=True,
            opt_geminal=True,
            opt_J3_basis_exp=True,
            opt_J3_basis_coeff=True,
            opt_lambda_basis_exp=True,
            opt_lambda_basis_coeff=True,
        )
        block_names = [b.name for b in blocks]
        assert "j3_basis_exp" in block_names
        assert "j3_basis_coeff" in block_names
        assert "lambda_basis_exp" in block_names
        assert "lambda_basis_coeff" in block_names

        # block.size is the per-shell primitive count (one
        # DOF per shell), and there is no shell-sharing symmetriser.
        j3_exp_block = next(b for b in blocks if b.name == "j3_basis_exp")
        assert j3_exp_block.size == aos_data.radial.num_shell_prim
        assert j3_exp_block.tangent_spec.kind == "euclidean"
        assert (j3_exp_block.field, j3_exp_block.derivative_policy, j3_exp_block.sync_scope) == (
            "real",
            "direct_real",
            "world",
        )
        npt.assert_allclose(
            np.asarray(j3_exp_block.values),
            np.asarray(aos_data.radial.exponents),
            atol=get_tolerance("ao_eval", "strict")[0],
            rtol=get_tolerance("ao_eval", "strict")[1],
        )

    # ============================================================
    # Test 11: apply_block_update for J3 basis blocks
    # ============================================================

    @pytest.mark.light
    def test_apply_block_update_j3_basis(self):
        """apply_block_update should handle j3_basis_exp and j3_basis_coeff blocks."""
        _, aos_data, _, _, _, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
        j3 = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=aos_data, random_init=True, random_scale=0.01, seed=42
        )
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=None,
            jastrow_three_body_data=j3,
        )

        new_exp = np.array(aos_data.radial.exponents) * 1.05
        block = make_test_block(
            name="j3_basis_exp",
            values=new_exp,
            field="real",
            derivative_policy="direct_real",
            sync_scope="world",
        )
        jastrow_new = jastrow_data.apply_block_update(block)
        npt.assert_allclose(
            np.array(jastrow_new.jastrow_three_body_data.ao_exponents),
            new_exp,
            atol=get_tolerance("ao_eval", "strict")[0],
            rtol=get_tolerance("ao_eval", "strict")[1],
        )

    # ============================================================
    # Test 12: apply_block_update for Geminal basis blocks
    # ============================================================

    @pytest.mark.light
    def test_apply_block_update_geminal_basis(self):
        """apply_block_update should handle lambda_basis_exp/coeff blocks (up+dn concatenated)."""
        _, _, _, _, geminal_mo_data, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
        geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)

        new_exp_up = np.array(geminal_ao.ao_exponents_up) * 0.95
        new_exp_dn = np.array(geminal_ao.ao_exponents_dn) * 1.05
        new_exp = np.concatenate([new_exp_up, new_exp_dn])
        block = make_test_block(
            name="lambda_basis_exp",
            values=new_exp,
            field="real",
            derivative_policy="direct_real",
            sync_scope="sampler",
        )
        geminal_new = geminal_ao.apply_block_update(block)
        npt.assert_allclose(
            np.array(geminal_new.ao_exponents_up),
            new_exp_up,
            atol=get_tolerance("ao_eval", "strict")[0],
            rtol=get_tolerance("ao_eval", "strict")[1],
        )
        npt.assert_allclose(
            np.array(geminal_new.ao_exponents_dn),
            new_exp_dn,
            atol=get_tolerance("ao_eval", "strict")[0],
            rtol=get_tolerance("ao_eval", "strict")[1],
        )

    # ============================================================
    # Test 15: Full wavefunction gradient through exponents (end-to-end)
    # ============================================================

    @pytest.mark.light
    def test_full_wavefunction_exponent_gradient(self):
        """End-to-end: grad of ln|Psi| w.r.t. J3 AO exponents is finite and nonzero."""
        structure_data, aos_data, _, _, geminal_mo_data, coulomb_potential_data = _load_trexio("H2_ae_ccpvdz_cart.h5")
        geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
        r_up, r_dn = _random_electron_coords(structure_data, coulomb_potential_data, geminal_ao)

        j3 = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=aos_data, random_init=True, random_scale=0.01, seed=42
        )
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=None,
            jastrow_three_body_data=j3,
        )
        wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_ao)

        def ln_psi(wf_data):
            return evaluate_ln_wavefunction(wf_data, r_up, r_dn)

        grad_fn = jax.grad(ln_psi)
        grad_wf = grad_fn(wf)

        # Check J3 exponent gradients
        grad_j3_exp = np.array(grad_wf.jastrow_data.jastrow_three_body_data.orb_data.exponents)
        assert np.all(np.isfinite(grad_j3_exp)), "J3 exponent gradients should be finite"
        assert np.any(grad_j3_exp != 0.0), "J3 exponent gradients should be nonzero"

        # Check Geminal exponent gradients
        grad_gem_exp = np.array(grad_wf.geminal_data.ao_exponents_up)
        assert np.all(np.isfinite(grad_gem_exp)), "Geminal exponent gradients should be finite"
        assert np.any(grad_gem_exp != 0.0), "Geminal exponent gradients should be nonzero"

    # ============================================================
    # Test 16: shell-constraint symmetrize in apply_block_update
    # ============================================================

    @pytest.mark.light
    def test_j3_basis_block_is_per_shell_no_symmetrise(self):
        """The J3 basis block exposes ONE DOF per shell-primitive
        with NO shell-sharing symmetriser. An update is stored on the per-shell
        radial leaf and the full per-AO exponents share it across each shell's
        m-components structurally -- replacing the old ShellPrimMap averaging.
        """
        _, aos_data, _, _, _, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
        j3 = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=aos_data, random_init=True, random_scale=0.01, seed=42
        )
        jastrow_data = Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=j3)
        wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=None)
        blocks = wf.get_variational_blocks(state_token=("test", 0), opt_J3_param=True, opt_J3_basis_exp=True)
        exp_block = next(b for b in blocks if b.name == "j3_basis_exp")

        # one DOF per shell-primitive, fewer than the full per-AO count, no symmetriser
        assert exp_block.size == aos_data.radial.num_shell_prim
        assert exp_block.size < int(aos_data.num_ao_prim)
        assert exp_block.tangent_spec.kind == "euclidean"

        # apply a per-shell update: the radial leaf stores it exactly, and the
        # full per-AO exponents broadcast it across each shell's m-components.
        new_vals = np.asarray(exp_block.values, dtype=np.float64) + 0.1
        block2 = exp_block.replace(values=new_vals, shape=new_vals.shape, size=int(new_vals.size))
        jnew = jastrow_data.apply_block_update(block2)
        atol, rtol = get_tolerance("ao_eval", "strict")
        npt.assert_allclose(np.asarray(jnew.jastrow_three_body_data.ao_exponents), new_vals, atol=atol, rtol=rtol)
        full = np.asarray(jnew.jastrow_three_body_data.orb_data.exponents, dtype=np.float64)
        m = np.asarray(jnew.jastrow_three_body_data.orb_data.ao_prim_to_shell_prim)
        npt.assert_allclose(full, new_vals[m], atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_geminal_basis_block_is_per_shell_no_symmetrise(self):
        """The geminal lambda basis block is per-shell (up + dn
        shell-primitive counts) with no symmetriser."""
        from jqmc.wavefunction import _get_aos_data

        _, _, _, _, geminal_mo_data, _ = _load_trexio("H2_ae_ccpvdz_cart.h5")
        geminal_ao = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
        wf = Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=geminal_ao)
        blocks = wf.get_variational_blocks(state_token=("test", 0), opt_lambda_basis_exp=True)
        exp_block = next(b for b in blocks if b.name == "lambda_basis_exp")

        up = _get_aos_data(geminal_ao.orb_data_up_spin)
        dn = _get_aos_data(geminal_ao.orb_data_dn_spin)
        assert exp_block.size == up.radial.num_shell_prim + dn.radial.num_shell_prim
        assert exp_block.tangent_spec.kind == "euclidean"

    # The lambda / basis conflict is enforced inside
    # :meth:`MCMC.run_optimize` directly (``fix_geminal_rank`` +
    # ``opt_lambda_basis_exp/coeff``) and covered by the integration tests
    # (``test_jqmc_mcmc.py::test_auto_convert_AGP_to_AGPn``).
