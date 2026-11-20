"""collections of unit tests."""

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
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

from jqmc._setting import ATOL_NONVACUITY_DISTINCTION, RTOL_NONVACUITY_DISTINCTION

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance, get_tolerance_min
from jqmc._setting import (
    ATOL_FD_VS_ANALYTIC_TIGHT,
    ATOL_PYTHON_FP64_AGGREGATOR_LOOSE,
    ATOL_PYTHON_FP64_LOOSE,
    ATOL_PYTHON_FP64_TIGHT,
    ATOL_SD_DIRECT_FACTOR_ORACLE,
    RTOL_FD_VS_ANALYTIC_TIGHT,
    RTOL_PYTHON_FP64_AGGREGATOR_LOOSE,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_SD_DIRECT_FACTOR_ORACLE,
)
from jqmc.atomic_orbital import (
    AOs_cart_pgto_complex_data,
    AOs_sphe_gto_data,
    AOs_sphe_pgto_complex_data,
    compute_AOs,
    compute_overlap_matrix,
)
from jqmc.determinant import (
    AGP_data,
    AGPn_data,
    Geminal_data,
    SD_data,
    _advance_det_candidate_mesh_state,
    _advance_det_ratio_streaming_state,
    _advance_grads_laplacian_ln_Det_streaming_state,
    _compute_AS_regularization_factor_debug,
    _compute_det_geminal_all_elements_debug,
    _compute_geminal_all_elements,
    _compute_geminal_all_elements_debug,
    _compute_grads_and_laplacian_ln_Det_auto,
    _compute_grads_and_laplacian_ln_Det_fast_debug,
    _compute_ratio_determinant_from_candidate_mesh_state,
    _compute_ratio_determinant_from_winv_state,
    _compute_ratio_determinant_part_debug,
    _compute_ratio_determinant_part_rank1_update,
    _compute_ratio_determinant_part_split_spin,
    _compute_u_dn_move_from_det_ratio_state,
    _compute_v_up_move_from_det_ratio_state,
    _init_det_candidate_mesh_state,
    _init_det_ratio_streaming_state,
    _init_det_winv_state,
    _init_grads_laplacian_ln_Det_streaming_state,
    _sherman_morrison_ratio_and_inverse,
    compute_AS_regularization_factor,
    compute_det_geminal_all_elements,
    compute_geminal_all_elements,
    compute_geminal_dn_one_column_elements,
    compute_geminal_up_one_row_elements,
    compute_grads_and_laplacian_ln_Det,
    compute_grads_and_laplacian_ln_Det_fast,
    compute_ln_det_geminal_all_elements,
    compute_ln_det_geminal_all_elements_fast,
    to_AGP,
    to_AGPn,
    to_SD,
)
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file
from tests.test_jqmc_kernel.optimizer_metadata_helpers import make_test_block

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


def _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol):
    """Keep the homogeneous-block bound transparent to GDD runtime receipts."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max()


class TestGeminalAOMOConversion:
    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
    def test_convert_from_MOs_to_AOs_closed_shell(self, trexio_file: str):
        """Test the consistency between AO and MO geminals."""
        fixture_rng = np.random.RandomState(20260923)
        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        # check geminal_data
        geminal_mo_data.sanity_check()

        num_electron_up = geminal_mo_data.num_electron_up
        num_electron_dn = geminal_mo_data.num_electron_dn

        # Initialization
        r_up_carts = []
        r_dn_carts = []

        total_electrons = 0

        if coulomb_potential_data.ecp_flag:
            charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
        else:
            charges = np.array(structure_data.atomic_numbers)

        coords = structure_data._positions_cart_np

        # Place electrons around each nucleus
        for i in range(len(coords)):
            charge = charges[i]
            num_electrons = int(np.round(charge))  # Number of electrons to place based on the charge

            # Retrieve the position coordinates
            x, y, z = coords[i]

            # Place electrons
            for _ in range(num_electrons):
                # Calculate distance range
                distance = fixture_rng.uniform(1.0 / charge, 2.0 / charge)
                theta = fixture_rng.uniform(0, np.pi)
                phi = fixture_rng.uniform(0, 2 * np.pi)

                # Convert spherical to Cartesian coordinates
                dx = distance * np.sin(theta) * np.cos(phi)
                dy = distance * np.sin(theta) * np.sin(phi)
                dz = distance * np.cos(theta)

                # Position of the electron
                electron_position = np.array([x + dx, y + dy, z + dz])

                # Assign spin
                if len(r_up_carts) < num_electron_up:
                    r_up_carts.append(electron_position)
                else:
                    r_dn_carts.append(electron_position)

            total_electrons += num_electrons

        # Handle surplus electrons
        remaining_up = num_electron_up - len(r_up_carts)
        remaining_dn = num_electron_dn - len(r_dn_carts)

        # Randomly place any remaining electrons
        for _ in range(remaining_up):
            r_up_carts.append(fixture_rng.choice(coords) + fixture_rng.normal(scale=0.1, size=3))
        for _ in range(remaining_dn):
            r_dn_carts.append(fixture_rng.choice(coords) + fixture_rng.normal(scale=0.1, size=3))

        r_up_carts = np.array(r_up_carts).reshape(-1, 3)
        r_dn_carts = np.array(r_dn_carts).reshape(-1, 3)

        atol_g, rtol_g = get_tolerance("det_eval", "strict")
        atol_d, rtol_d = get_tolerance("det_eval", "strict")
        # Debug-oracle vs production determinant: an ill-conditioned G amplifies
        # the fp32 ao_eval rounding of either route (N_ae: det -2.4e-4, G ~ 7.7),
        # so this pair is judged on the ao_eval/det_eval strict tier.
        # Full precision resolves to det_eval strict.
        atol_do, rtol_do = get_tolerance_min(("ao_eval", "det_eval"), "strict")

        geminal_mo_debug = _compute_geminal_all_elements_debug(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        geminal_mo_jax = _compute_geminal_all_elements(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        assert not np.any(np.isnan(np.asarray(geminal_mo_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(geminal_mo_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(geminal_mo_debug, geminal_mo_jax, atol=atol_g, rtol=rtol_g)

        geminal_mo = geminal_mo_jax

        """
        mo_lambda_matrix_paired, mo_lambda_matrix_unpaired = np.hsplit(geminal_mo_data.lambda_matrix, [geminal_mo_data.orb_num_dn])

        # generate matrices for the test
        ao_lambda_matrix_paired = np.dot(
            mos_data_up.mo_coefficients.T,
            np.dot(mo_lambda_matrix_paired, mos_data_dn.mo_coefficients),
        )
        ao_lambda_matrix_unpaired = np.dot(mos_data_up.mo_coefficients.T, mo_lambda_matrix_unpaired)
        ao_lambda_matrix = np.hstack([ao_lambda_matrix_paired, ao_lambda_matrix_unpaired])

        geminal_ao_data = Geminal_data(
            num_electron_up=num_electron_up,
            num_electron_dn=num_electron_dn,
            orb_data_up_spin=aos_data,
            orb_data_dn_spin=aos_data,
            lambda_matrix=ao_lambda_matrix,
        )
        """

        geminal_ao_data = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
        geminal_ao_data.sanity_check()

        geminal_ao_debug = _compute_geminal_all_elements_debug(
            geminal_data=geminal_ao_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        geminal_ao_jax = _compute_geminal_all_elements_debug(
            geminal_data=geminal_ao_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        assert not np.any(np.isnan(np.asarray(geminal_ao_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(geminal_ao_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(geminal_ao_debug, geminal_ao_jax, atol=atol_g, rtol=rtol_g)

        geminal_ao = geminal_ao_jax

        # check if geminals with AO and MO representations are consistent
        assert not np.any(np.isnan(np.asarray(geminal_ao))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(geminal_mo))), "NaN detected in second argument"
        np.testing.assert_allclose(geminal_ao, geminal_mo, atol=atol_g, rtol=rtol_g)

        det_geminal_mo_debug = _compute_det_geminal_all_elements_debug(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        det_geminal_mo_jax = compute_det_geminal_all_elements(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        assert not np.any(np.isnan(np.asarray(det_geminal_mo_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(det_geminal_mo_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(det_geminal_mo_debug, det_geminal_mo_jax, atol=atol_do, rtol=rtol_do)

        det_geminal_mo = det_geminal_mo_jax

        det_geminal_ao_debug = _compute_det_geminal_all_elements_debug(
            geminal_data=geminal_ao_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        det_geminal_ao_jax = compute_det_geminal_all_elements(
            geminal_data=geminal_ao_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        assert not np.any(np.isnan(np.asarray(det_geminal_ao_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(det_geminal_ao_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(det_geminal_ao_debug, det_geminal_ao_jax, atol=atol_do, rtol=rtol_do)
        det_geminal_ao = det_geminal_ao_jax

        assert not np.any(np.isnan(np.asarray(det_geminal_ao))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(det_geminal_mo))), "NaN detected in second argument"
        np.testing.assert_allclose(det_geminal_ao, det_geminal_mo, atol=atol_d, rtol=rtol_d)

        jax.clear_caches()


def _build_sphe_aos_l_le6(rng: np.random.Generator) -> AOs_sphe_gto_data:
    nucleus_index: list[int] = []
    orbital_indices: list[int] = []
    exponents: list[float] = []
    coefficients: list[float] = []
    angular_momentums: list[int] = []
    magnetic_quantum_numbers: list[int] = []

    ao_idx = 0
    for l in range(7):
        exp_l = rng.uniform(0.5, 2.0)
        coef_l = rng.uniform(0.7, 1.3)
        for m in range(-l, l + 1):
            nucleus_index.append(0)
            orbital_indices.append(ao_idx)
            exponents.append(exp_l)
            coefficients.append(coef_l)
            angular_momentums.append(l)
            magnetic_quantum_numbers.append(m)
            ao_idx += 1

    structure_data = Structure_data(
        pbc_flag=False,
        positions=np.zeros((1, 3), dtype=np.float64),
        atomic_numbers=(1,),
        element_symbols=("X",),
        atomic_labels=("X",),
    )

    return AOs_sphe_gto_data._from_full_basis(
        structure_data=structure_data,
        nucleus_index=tuple(nucleus_index),
        num_ao=len(angular_momentums),
        num_ao_prim=len(exponents),
        orbital_indices=tuple(orbital_indices),
        exponents=np.array(exponents, dtype=np.float64),
        coefficients=np.array(coefficients, dtype=np.float64),
        angular_momentums=tuple(angular_momentums),
        magnetic_quantum_numbers=tuple(magnetic_quantum_numbers),
    )


class TestGeminalAOBasisTransforms:
    @pytest.mark.light
    def test_geminal_sphe_to_cart_AOs_data(self):
        """Round-trip AOs l<=6: spherical->Cartesian keeps geminal values/grads."""
        # Comparison crosses ao_eval/det_eval (values) and ao_grad_lap/det_grad_lap (grads);
        # achievable agreement is bounded by the loosest zone on the path.
        atol_c, rtol_c = get_tolerance_min(("ao_eval", "det_eval", "ao_grad_lap", "det_grad_lap"), "strict")
        rng = np.random.default_rng(321)

        aos_sphe = _build_sphe_aos_l_le6(rng)
        num_electron_up = 4
        num_electron_dn = 4

        lambda_matrix = rng.uniform(-0.2, 0.2, size=(aos_sphe.num_ao, aos_sphe.num_ao))
        geminal_sph = Geminal_data(
            num_electron_up=num_electron_up,
            num_electron_dn=num_electron_dn,
            orb_data_up_spin=aos_sphe,
            orb_data_dn_spin=aos_sphe,
            lambda_matrix=lambda_matrix,
        )
        geminal_cart = geminal_sph.to_cartesian()

        r_up_carts = rng.uniform(-1.0, 1.0, size=(num_electron_up, 3))
        r_dn_carts = rng.uniform(-1.0, 1.0, size=(num_electron_dn, 3))

        G_sph = compute_geminal_all_elements(geminal_sph, r_up_carts, r_dn_carts)
        G_cart = compute_geminal_all_elements(geminal_cart, r_up_carts, r_dn_carts)
        assert not np.any(np.isnan(np.asarray(np.asarray(G_sph)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(G_cart)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(G_sph), np.asarray(G_cart), atol=atol_c, rtol=rtol_c)

        grads_sph = compute_grads_and_laplacian_ln_Det(geminal_sph, r_up_carts, r_dn_carts)
        grads_cart = compute_grads_and_laplacian_ln_Det(geminal_cart, r_up_carts, r_dn_carts)
        for channel, sph, cart in zip(("grad_up", "grad_dn", "lap_up", "lap_dn"), grads_sph, grads_cart, strict=True):
            assert not np.any(np.isnan(np.asarray(np.asarray(sph)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(cart)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(sph), np.asarray(cart), atol=atol_c, rtol=rtol_c)

        jax.clear_caches()

    @pytest.mark.light
    def test_geminal_cart_to_sphe_AOs_data(self):
        """Round-trip AOs l<=6: Cartesian->spherical keeps geminal values/grads."""
        # Comparison crosses ao_eval/det_eval (values) and ao_grad_lap/det_grad_lap (grads);
        # achievable agreement is bounded by the loosest zone on the path.
        atol_c, rtol_c = get_tolerance_min(("ao_eval", "det_eval", "ao_grad_lap", "det_grad_lap"), "strict")
        rng = np.random.default_rng(654)

        aos_sphe = _build_sphe_aos_l_le6(rng)
        num_electron_up = 5
        num_electron_dn = 5

        lambda_matrix = rng.uniform(-0.2, 0.2, size=(aos_sphe.num_ao, aos_sphe.num_ao))
        geminal_sph = Geminal_data(
            num_electron_up=num_electron_up,
            num_electron_dn=num_electron_dn,
            orb_data_up_spin=aos_sphe,
            orb_data_dn_spin=aos_sphe,
            lambda_matrix=lambda_matrix,
        )

        geminal_cart = geminal_sph.to_cartesian()
        geminal_cart_to_sph = geminal_cart.to_spherical()

        r_up_carts = rng.uniform(-1.0, 1.0, size=(num_electron_up, 3))
        r_dn_carts = rng.uniform(-1.0, 1.0, size=(num_electron_dn, 3))

        G_cart = compute_geminal_all_elements(geminal_cart, r_up_carts, r_dn_carts)
        G_sph = compute_geminal_all_elements(geminal_cart_to_sph, r_up_carts, r_dn_carts)
        assert not np.any(np.isnan(np.asarray(np.asarray(G_cart)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(G_sph)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(G_cart), np.asarray(G_sph), atol=atol_c, rtol=rtol_c)

        grads_cart = compute_grads_and_laplacian_ln_Det(geminal_cart, r_up_carts, r_dn_carts)
        grads_sph = compute_grads_and_laplacian_ln_Det(geminal_cart_to_sph, r_up_carts, r_dn_carts)
        for channel, cart, sph in zip(("grad_up", "grad_dn", "lap_up", "lap_dn"), grads_cart, grads_sph, strict=True):
            assert not np.any(np.isnan(np.asarray(np.asarray(cart)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(sph)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(cart), np.asarray(sph), atol=atol_c, rtol=rtol_c)

        jax.clear_caches()


class TestGeminalMOBasisTransforms:
    @pytest.mark.light
    def test_geminal_sphe_to_cart_MOs_data(self):
        """Round-trip MOs built on l<=6 AOs: spherical->Cartesian keeps geminal values/grads."""
        # Comparison crosses ao_eval/mo_eval/det_eval (values) and ao_grad_lap/mo_grad/mo_lap/det_grad_lap (grads);
        # achievable agreement is bounded by the loosest zone on the path.
        atol_c, rtol_c = get_tolerance_min(
            ("ao_eval", "mo_eval", "det_eval", "ao_grad_lap", "mo_grad", "mo_lap", "det_grad_lap"),
            "strict",
        )
        rng = np.random.default_rng(777)

        aos_sphe = _build_sphe_aos_l_le6(rng)
        num_mo = aos_sphe.num_ao
        mo_coefficients = rng.uniform(-0.5, 0.5, size=(num_mo, aos_sphe.num_ao))
        mos_sphe = MOs_data(num_mo=num_mo, aos_data=aos_sphe, mo_coefficients=mo_coefficients)
        mos_sphe.sanity_check()

        num_electron_up = 5
        num_electron_dn = 5
        lambda_matrix = rng.uniform(-0.2, 0.2, size=(mos_sphe.num_mo, mos_sphe.num_mo))
        geminal_sph = Geminal_data(
            num_electron_up=num_electron_up,
            num_electron_dn=num_electron_dn,
            orb_data_up_spin=mos_sphe,
            orb_data_dn_spin=mos_sphe,
            lambda_matrix=lambda_matrix,
        )
        geminal_cart = geminal_sph.to_cartesian()

        r_up_carts = rng.uniform(-1.0, 1.0, size=(num_electron_up, 3))
        r_dn_carts = rng.uniform(-1.0, 1.0, size=(num_electron_dn, 3))

        G_sph = compute_geminal_all_elements(geminal_sph, r_up_carts, r_dn_carts)
        G_cart = compute_geminal_all_elements(geminal_cart, r_up_carts, r_dn_carts)
        assert not np.any(np.isnan(np.asarray(np.asarray(G_sph)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(G_cart)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(G_sph), np.asarray(G_cart), atol=atol_c, rtol=rtol_c)

        grads_sph = compute_grads_and_laplacian_ln_Det(geminal_sph, r_up_carts, r_dn_carts)
        grads_cart = compute_grads_and_laplacian_ln_Det(geminal_cart, r_up_carts, r_dn_carts)
        for channel, sph, cart in zip(("grad_up", "grad_dn", "lap_up", "lap_dn"), grads_sph, grads_cart, strict=True):
            assert not np.any(np.isnan(np.asarray(np.asarray(sph)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(cart)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(sph), np.asarray(cart), atol=atol_c, rtol=rtol_c)

        jax.clear_caches()

    @pytest.mark.light
    def test_geminal_cart_to_sphe_MOs_data(self):
        """Round-trip MOs l<=6: Cartesian->spherical keeps geminal values/grads."""
        # Comparison crosses ao_eval/mo_eval/det_eval (values) and ao_grad_lap/mo_grad/mo_lap/det_grad_lap (grads);
        # achievable agreement is bounded by the loosest zone on the path.
        atol_c, rtol_c = get_tolerance_min(
            ("ao_eval", "mo_eval", "det_eval", "ao_grad_lap", "mo_grad", "mo_lap", "det_grad_lap"),
            "strict",
        )
        rng = np.random.default_rng(888)

        aos_sphe = _build_sphe_aos_l_le6(rng)
        num_mo = aos_sphe.num_ao
        mo_coefficients = rng.uniform(-0.5, 0.5, size=(num_mo, aos_sphe.num_ao))
        mos_sphe = MOs_data(num_mo=num_mo, aos_data=aos_sphe, mo_coefficients=mo_coefficients)
        mos_sphe.sanity_check()

        num_electron_up = 6
        num_electron_dn = 6
        lambda_matrix = rng.uniform(-0.2, 0.2, size=(mos_sphe.num_mo, mos_sphe.num_mo))
        geminal_sph = Geminal_data(
            num_electron_up=num_electron_up,
            num_electron_dn=num_electron_dn,
            orb_data_up_spin=mos_sphe,
            orb_data_dn_spin=mos_sphe,
            lambda_matrix=lambda_matrix,
        )

        geminal_cart = geminal_sph.to_cartesian()
        geminal_cart_to_sph = geminal_cart.to_spherical()

        r_up_carts = rng.uniform(-1.0, 1.0, size=(num_electron_up, 3))
        r_dn_carts = rng.uniform(-1.0, 1.0, size=(num_electron_dn, 3))

        G_cart = compute_geminal_all_elements(geminal_cart, r_up_carts, r_dn_carts)
        G_sph = compute_geminal_all_elements(geminal_cart_to_sph, r_up_carts, r_dn_carts)
        assert not np.any(np.isnan(np.asarray(np.asarray(G_cart)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(G_sph)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(G_cart), np.asarray(G_sph), atol=atol_c, rtol=rtol_c)

        grads_cart = compute_grads_and_laplacian_ln_Det(geminal_cart, r_up_carts, r_dn_carts)
        grads_sph = compute_grads_and_laplacian_ln_Det(geminal_cart_to_sph, r_up_carts, r_dn_carts)
        for channel, cart, sph in zip(("grad_up", "grad_dn", "lap_up", "lap_dn"), grads_cart, grads_sph, strict=True):
            assert not np.any(np.isnan(np.asarray(np.asarray(cart)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(sph)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(cart), np.asarray(sph), atol=atol_c, rtol=rtol_c)

        jax.clear_caches()


def _build_small_sphe_aos_for_conversion() -> AOs_sphe_gto_data:
    structure_data = Structure_data(
        pbc_flag=False,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        atomic_numbers=(1,),
        element_symbols=("X",),
        atomic_labels=("X",),
    )
    return AOs_sphe_gto_data._from_full_basis(
        structure_data=structure_data,
        nucleus_index=(0, 0, 0, 0),
        num_ao=4,
        num_ao_prim=4,
        orbital_indices=(0, 1, 2, 3),
        exponents=np.array([1.20, 1.00, 1.00, 1.00], dtype=np.float64),
        coefficients=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        angular_momentums=(0, 1, 1, 1),
        magnetic_quantum_numbers=(0, -1, 0, 1),
    )


class TestGeminalAOMOProjection:
    @pytest.mark.light
    def test_convert_from_AOs_to_MOs_full_projection_closed_shell(self):
        """AO->MO (all eigenvectors) followed by MO->AO recovers the AO lambda matrix."""
        atol_c, rtol_c = get_tolerance("det_eval", "strict")
        rng = np.random.default_rng(1234)
        aos_data = _build_small_sphe_aos_for_conversion()
        aos_data.sanity_check()

        num_electron_up = 2
        num_electron_dn = 2

        lambda_matrix_paired = rng.uniform(-0.25, 0.25, size=(aos_data.num_ao, aos_data.num_ao))
        lambda_matrix_paired = 0.5 * (lambda_matrix_paired + lambda_matrix_paired.T)
        lambda_matrix = np.hstack([lambda_matrix_paired, np.zeros((aos_data.num_ao, 0), dtype=np.float64)])

        geminal_ao = Geminal_data(
            num_electron_up=num_electron_up,
            num_electron_dn=num_electron_dn,
            orb_data_up_spin=aos_data,
            orb_data_dn_spin=aos_data,
            lambda_matrix=lambda_matrix,
        )
        geminal_mo = Geminal_data.convert_from_AOs_to_MOs(geminal_ao, num_eigenvectors="all")
        geminal_ao_back = Geminal_data.convert_from_MOs_to_AOs(geminal_mo)

        assert not np.any(np.isnan(np.asarray(np.asarray(geminal_ao_back.lambda_matrix)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(geminal_ao.lambda_matrix)))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(geminal_ao_back.lambda_matrix),
            np.asarray(geminal_ao.lambda_matrix),
            atol=atol_c,
            rtol=rtol_c,
        )

    @pytest.mark.light
    def test_convert_from_AOs_to_MOs_full_projection_open_shell(self):
        """AO->MO (all eigenvectors) round-trip recovers AO lambda matrix for open-shell case."""
        atol_c, rtol_c = get_tolerance("det_eval", "strict")
        rng = np.random.default_rng(1334)
        aos_data = _build_small_sphe_aos_for_conversion()
        aos_data.sanity_check()

        num_electron_up = 3
        num_electron_dn = 2
        num_unpaired = num_electron_up - num_electron_dn

        lambda_matrix_paired = rng.uniform(-0.25, 0.25, size=(aos_data.num_ao, aos_data.num_ao))
        lambda_matrix_paired = 0.5 * (lambda_matrix_paired + lambda_matrix_paired.T)
        lambda_matrix_unpaired = rng.uniform(-0.25, 0.25, size=(aos_data.num_ao, num_unpaired))
        lambda_matrix = np.hstack([lambda_matrix_paired, lambda_matrix_unpaired])

        geminal_ao = Geminal_data(
            num_electron_up=num_electron_up,
            num_electron_dn=num_electron_dn,
            orb_data_up_spin=aos_data,
            orb_data_dn_spin=aos_data,
            lambda_matrix=lambda_matrix,
        )
        geminal_mo = Geminal_data.convert_from_AOs_to_MOs(geminal_ao, num_eigenvectors="all")
        geminal_ao_back = Geminal_data.convert_from_MOs_to_AOs(geminal_mo)

        assert not np.any(np.isnan(np.asarray(np.asarray(geminal_ao_back.lambda_matrix)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(geminal_ao.lambda_matrix)))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(geminal_ao_back.lambda_matrix),
            np.asarray(geminal_ao.lambda_matrix),
            atol=atol_c,
            rtol=rtol_c,
        )

    @pytest.mark.light
    def test_convert_from_AOs_to_MOs_truncated_mode_closed_shell(self):
        """AO->MO integer mode accepts boundary values without enforcing eigenvalue scaling."""
        rng = np.random.default_rng(4321)
        aos_data = _build_small_sphe_aos_for_conversion()
        aos_data.sanity_check()

        num_electron_up = 2
        num_electron_dn = 2

        lambda_matrix_paired = rng.uniform(-0.20, 0.20, size=(aos_data.num_ao, aos_data.num_ao))
        lambda_matrix_paired = 0.5 * (lambda_matrix_paired + lambda_matrix_paired.T)
        lambda_matrix = np.hstack([lambda_matrix_paired, np.zeros((aos_data.num_ao, 0), dtype=np.float64)])

        geminal_ao = Geminal_data(
            num_electron_up=num_electron_up,
            num_electron_dn=num_electron_dn,
            orb_data_up_spin=aos_data,
            orb_data_dn_spin=aos_data,
            lambda_matrix=lambda_matrix,
        )

        geminal_mo = Geminal_data.convert_from_AOs_to_MOs(geminal_ao, num_eigenvectors=3)
        paired_block, _ = np.hsplit(np.asarray(geminal_mo.lambda_matrix), [geminal_mo.orb_num_dn])

        geminal_mo_small = Geminal_data.convert_from_AOs_to_MOs(geminal_ao, num_eigenvectors=2)
        assert geminal_mo_small.orb_num_up == 2
        assert geminal_mo_small.orb_num_dn == 2

        geminal_mo_full = Geminal_data.convert_from_AOs_to_MOs(geminal_ao, num_eigenvectors=4)
        assert geminal_mo_full.orb_num_up == aos_data.num_ao
        assert geminal_mo_full.orb_num_dn == aos_data.num_ao

        with pytest.raises(ValueError, match="exceeds the available rank"):
            Geminal_data.convert_from_AOs_to_MOs(geminal_ao, num_eigenvectors=5)

    @pytest.mark.light
    def test_convert_from_AOs_to_MOs_truncated_mode_open_shell(self):
        """Open-shell truncation keeps projected unpaired block without paired-eigenvalue scaling."""
        rng = np.random.default_rng(2468)
        aos_data = _build_sphe_aos_l_le6(rng)
        aos_data.sanity_check()

        num_electron_up = 3
        num_electron_dn = 2
        num_unpaired = num_electron_up - num_electron_dn

        lambda_matrix_paired = rng.uniform(-0.2, 0.2, size=(aos_data.num_ao, aos_data.num_ao))
        lambda_matrix_unpaired = rng.uniform(-0.2, 0.2, size=(aos_data.num_ao, num_unpaired))
        lambda_matrix = np.hstack([lambda_matrix_paired, lambda_matrix_unpaired])

        geminal_ao = Geminal_data(
            num_electron_up=num_electron_up,
            num_electron_dn=num_electron_dn,
            orb_data_up_spin=aos_data,
            orb_data_dn_spin=aos_data,
            lambda_matrix=lambda_matrix,
            time_reversal=False,
        )

        geminal_mo = Geminal_data.convert_from_AOs_to_MOs(geminal_ao, num_eigenvectors=5)
        paired_block, unpaired_block = np.hsplit(np.asarray(geminal_mo.lambda_matrix), [geminal_mo.orb_num_dn])

        assert unpaired_block.shape == (5, 1)
        # The retained subspace includes the unpaired direction. Its AO
        # reconstruction must therefore recover that column, not merely its
        # shape, even though the paired block was rank truncated.
        coefficients = np.asarray(geminal_mo.orb_data_up_spin.mo_coefficients)
        restored_unpaired = coefficients.T @ unpaired_block
        atol_c, rtol_c = get_tolerance("det_eval", "strict")
        np.testing.assert_allclose(restored_unpaired, lambda_matrix_unpaired, atol=atol_c, rtol=rtol_c)


class TestDeterminantFastUpdate:
    @pytest.mark.parametrize(
        "trexio_file",
        ["H_ae_ccpvdz_cart.h5", "Li_ae_ccpvdz_cart.h5", "H2_ecp_ccpvtz.h5"],
    )
    @pytest.mark.light
    def test_compute_grads_and_laplacian_ln_Det_fast_vs_fast_debug(self, trexio_file: str):
        """Guardrail: compute_grads_and_laplacian_ln_Det_fast == _fast_debug.

        Renamed from test_grads_and_laplacian_fast_update so the fast/_fast_debug
        pair follows the discoverable ..._fast_vs_fast_debug naming convention;
        the comparison itself is unchanged.
        """
        fixture_rng = np.random.RandomState(20260923)
        (
            _,
            aos_data,
            _,
            _,
            geminal_mo_data,
            _,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        geminal_mo_data.sanity_check()

        num_electron_up = geminal_mo_data.num_electron_up
        num_electron_dn = geminal_mo_data.num_electron_dn

        r_cart_min, r_cart_max = -2.0, 2.0
        r_up_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_electron_up, 3) + r_cart_min
        r_dn_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_electron_dn, 3) + r_cart_min

        # Build the direct occupied-SD matrix and its inverse.
        geminal = compute_geminal_all_elements(geminal_mo_data, r_up_carts, r_dn_carts)

        P, L, U = jsp_linalg.lu(geminal)
        n = geminal.shape[0]
        I = jnp.eye(n, dtype=geminal.dtype)
        Y = jsp_linalg.solve_triangular(L, jnp.dot(P.T, I), lower=True)
        geminal_inverse = jsp_linalg.solve_triangular(U, Y, lower=False)

        # Fast path (requires inverse)
        grad_up_fast, grad_dn_fast, lap_up_fast, lap_dn_fast = compute_grads_and_laplacian_ln_Det_fast(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            geminal_inverse=geminal_inverse,
        )

        # Debug helper
        grad_up_debug, grad_dn_debug, lap_up_debug, lap_dn_debug = _compute_grads_and_laplacian_ln_Det_fast_debug(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        # Debug helper above is autodiff through compute_ln_det -> bottlenecked by
        # ao_eval (fp32 in mixed mode); fast path is fp64 (ao_grad_lap), so the
        # achievable agreement is bounded by ao_eval, not det_grad_lap.
        atol, rtol = get_tolerance_min(["ao_eval", "det_grad_lap"], "strict")
        assert not np.any(np.isnan(np.asarray(grad_up_fast))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(grad_up_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(grad_up_fast, grad_up_debug, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(grad_dn_fast))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(grad_dn_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(grad_dn_fast, grad_dn_debug, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(lap_up_fast))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(lap_up_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(lap_up_fast, lap_up_debug, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(lap_dn_fast))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(lap_dn_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(lap_dn_fast, lap_dn_debug, atol=atol, rtol=rtol)


class TestASRegularization:
    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
    def test_comparing_AS_regularization(self, trexio_file: str):
        """Test the consistency between AS_regularization_debug and AS_regularization_jax."""
        fixture_rng = np.random.RandomState(20260923)
        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        # check geminal_data
        geminal_mo_data.sanity_check()

        num_electron_up = geminal_mo_data.num_electron_up
        num_electron_dn = geminal_mo_data.num_electron_dn

        # Initialization
        r_up_carts = []
        r_dn_carts = []

        total_electrons = 0

        if coulomb_potential_data.ecp_flag:
            charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
        else:
            charges = np.array(structure_data.atomic_numbers)

        coords = structure_data._positions_cart_np

        # Place electrons around each nucleus
        for i in range(len(coords)):
            charge = charges[i]
            num_electrons = int(np.round(charge))  # Number of electrons to place based on the charge

            # Retrieve the position coordinates
            x, y, z = coords[i]

            # Place electrons
            for _ in range(num_electrons):
                # Calculate distance range
                distance = fixture_rng.uniform(1.0 / charge, 2.0 / charge)
                theta = fixture_rng.uniform(0, np.pi)
                phi = fixture_rng.uniform(0, 2 * np.pi)

                # Convert spherical to Cartesian coordinates
                dx = distance * np.sin(theta) * np.cos(phi)
                dy = distance * np.sin(theta) * np.sin(phi)
                dz = distance * np.cos(theta)

                # Position of the electron
                electron_position = np.array([x + dx, y + dy, z + dz])

                # Assign spin
                if len(r_up_carts) < num_electron_up:
                    r_up_carts.append(electron_position)
                else:
                    r_dn_carts.append(electron_position)

            total_electrons += num_electrons

        # Handle surplus electrons
        remaining_up = num_electron_up - len(r_up_carts)
        remaining_dn = num_electron_dn - len(r_dn_carts)

        # Randomly place any remaining electrons
        for _ in range(remaining_up):
            r_up_carts.append(fixture_rng.choice(coords) + fixture_rng.normal(scale=0.1, size=3))
        for _ in range(remaining_dn):
            r_dn_carts.append(fixture_rng.choice(coords) + fixture_rng.normal(scale=0.1, size=3))

        r_up_carts = np.array(r_up_carts).reshape(-1, 3)
        r_dn_carts = np.array(r_dn_carts).reshape(-1, 3)

        R_AS_debug = _compute_AS_regularization_factor_debug(
            geminal_data=geminal_mo_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )

        R_AS_jax = compute_AS_regularization_factor(geminal_data=geminal_mo_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)

        # R_AS is a determinant-derived quantity of an AO-evaluated G: both routes
        # carry fp32 ao_eval rounding in mixed, amplified by the conditioning of G,
        # so it is judged on the ao_eval/det_eval strict tier.
        # Full precision resolves to det_eval strict.
        atol, rtol = get_tolerance_min(("ao_eval", "det_eval"), "strict")
        assert not np.any(np.isnan(np.asarray(R_AS_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(R_AS_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(R_AS_debug, R_AS_jax, atol=atol, rtol=rtol)

        jax.clear_caches()


class TestGeminalRowColumnUpdate:
    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
    def test_one_row_or_one_column_update(self, trexio_file: str):
        """Test the update of one row or one column in the geminal wave function."""
        fixture_rng = np.random.RandomState(20260923)
        """Test the consistency between AS_regularization_debug and AS_regularization_jax."""
        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        # check geminal_data
        geminal_mo_data.sanity_check()

        num_electron_up = geminal_mo_data.num_electron_up
        num_electron_dn = geminal_mo_data.num_electron_dn

        # Initialization
        r_up_carts = []
        r_dn_carts = []

        total_electrons = 0

        if coulomb_potential_data.ecp_flag:
            charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
        else:
            charges = np.array(structure_data.atomic_numbers)

        coords = structure_data._positions_cart_np

        # Place electrons around each nucleus
        for i in range(len(coords)):
            charge = charges[i]
            num_electrons = int(np.round(charge))  # Number of electrons to place based on the charge

            # Retrieve the position coordinates
            x, y, z = coords[i]

            # Place electrons
            for _ in range(num_electrons):
                # Calculate distance range
                distance = fixture_rng.uniform(1.0 / charge, 2.0 / charge)
                theta = fixture_rng.uniform(0, np.pi)
                phi = fixture_rng.uniform(0, 2 * np.pi)

                # Convert spherical to Cartesian coordinates
                dx = distance * np.sin(theta) * np.cos(phi)
                dy = distance * np.sin(theta) * np.sin(phi)
                dz = distance * np.cos(theta)

                # Position of the electron
                electron_position = np.array([x + dx, y + dy, z + dz])

                # Assign spin
                if len(r_up_carts) < num_electron_up:
                    r_up_carts.append(electron_position)
                else:
                    r_dn_carts.append(electron_position)

            total_electrons += num_electrons

        # Handle surplus electrons
        remaining_up = num_electron_up - len(r_up_carts)
        remaining_dn = num_electron_dn - len(r_dn_carts)

        # Randomly place any remaining electrons
        for _ in range(remaining_up):
            r_up_carts.append(fixture_rng.choice(coords) + fixture_rng.normal(scale=0.1, size=3))
        for _ in range(remaining_dn):
            r_dn_carts.append(fixture_rng.choice(coords) + fixture_rng.normal(scale=0.1, size=3))

        r_up_carts = np.array(r_up_carts).reshape(-1, 3)
        r_dn_carts = np.array(r_dn_carts).reshape(-1, 3)

        geminal_mo = compute_geminal_all_elements(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        # Pick test indices (any valid indices are fine)
        i_up = 0
        j_dn = 0

        # Compute the single "up row" against all dn electrons
        geminal_mo_up_one_row = compute_geminal_up_one_row_elements(
            geminal_data=geminal_mo_data,
            r_up_cart=np.reshape(r_up_carts[i_up], (1, 3)),  # enforce (1,3) for single point
            r_dn_carts=r_dn_carts,
        )

        # Compute the single "dn column" against all up electrons
        geminal_mo_dn_one_column = compute_geminal_dn_one_column_elements(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up_carts,
            r_dn_cart=np.reshape(r_dn_carts[j_dn], (1, 3)),  # enforce (1,3) for single point
        )

        # --- Numerical consistency asserts (no shape checks) ---
        atol, rtol = get_tolerance("det_eval", "strict")
        # up-one-row must equal the i-th row of the full geminal
        assert not np.any(np.isnan(np.asarray(np.asarray(geminal_mo_up_one_row).ravel()))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(geminal_mo[i_up, :])))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(geminal_mo_up_one_row).ravel(),
            np.asarray(geminal_mo[i_up, :]),
            atol=atol,
            rtol=rtol,
        )

        # dn-one-column must equal the j-th *paired* column of the full geminal
        assert not np.any(np.isnan(np.asarray(np.asarray(geminal_mo_dn_one_column).ravel()))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(geminal_mo[:, j_dn])))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(geminal_mo_dn_one_column).ravel(),
            np.asarray(geminal_mo[:, j_dn]),
            atol=atol,
            rtol=rtol,
        )


def _determinant_gradient_case(trexio_file):
    """Rebuild the original converted-AO derivative case and seeded positions."""
    (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
        store_tuple=True,
    )

    geminal_mo_data.sanity_check()

    num_electron_up = geminal_mo_data.num_electron_up
    num_electron_dn = geminal_mo_data.num_electron_dn

    # Seed RNG for determinism (CI must not depend on previous tests' draws).
    np.random.seed(42)

    # Initialization
    r_up_carts = []
    r_dn_carts = []

    total_electrons = 0

    if coulomb_potential_data.ecp_flag:
        charges = np.array(structure_data.atomic_numbers) - np.array(coulomb_potential_data.z_cores)
    else:
        charges = np.array(structure_data.atomic_numbers)

    coords = structure_data._positions_cart_np

    # Place electrons around each nucleus
    for i in range(len(coords)):
        charge = charges[i]
        num_electrons = int(np.round(charge))  # Number of electrons to place based on the charge

        # Retrieve the position coordinates
        x, y, z = coords[i]

        # Place electrons
        for _ in range(num_electrons):
            # Calculate distance range
            distance = np.random.uniform(0.5 / charge, 1.5 / charge)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            # Convert spherical to Cartesian coordinates
            dx = distance * np.sin(theta) * np.cos(phi)
            dy = distance * np.sin(theta) * np.sin(phi)
            dz = distance * np.cos(theta)

            # Position of the electron
            electron_position = np.array([x + dx, y + dy, z + dz])

            # Assign spin
            if len(r_up_carts) < num_electron_up:
                r_up_carts.append(electron_position)
            else:
                r_dn_carts.append(electron_position)

        total_electrons += num_electrons

    # Handle surplus electrons
    remaining_up = num_electron_up - len(r_up_carts)
    remaining_dn = num_electron_dn - len(r_dn_carts)

    # Randomly place any remaining electrons
    for _ in range(remaining_up):
        r_up_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))
    for _ in range(remaining_dn):
        r_dn_carts.append(np.random.choice(coords) + np.random.normal(scale=0.1, size=3))

    r_up_carts = np.array(r_up_carts).reshape(-1, 3)
    r_dn_carts = np.array(r_dn_carts).reshape(-1, 3)

    geminal_ao_data = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
    geminal_ao_data.sanity_check()

    return geminal_ao_data, r_up_carts, r_dn_carts


class TestDeterminantGradientVariants:
    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5", "H_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
    def test_analytic_and_auto_grads_and_laplacians_ln_Det(self, trexio_file: str):
        """Full retains strict analytical/AD fields; mixed checks fresh production fidelity."""
        if is_mixed_precision_enabled():

            def analytical(case):
                geminal, r_up, r_dn = case
                return compute_grads_and_laplacian_ln_Det(geminal, r_up, r_dn)

            mixed, full = mixed_vs_full(lambda: _determinant_gradient_case(trexio_file), analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
            for mixed_block, full_block in blocks:
                _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol)
            corrupted = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_component_derivative_fidelity(corrupted, blocks[0][1], atol, rtol)
            jax.clear_caches()
            return

        geminal_ao_data, r_up_carts, r_dn_carts = _determinant_gradient_case(trexio_file)

        grad_ln_D_up_analytic, grad_ln_D_dn_analytic, lap_ln_D_up_analytic, lap_ln_D_dn_analytic = (
            compute_grads_and_laplacian_ln_Det(
                geminal_data=geminal_ao_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )
        )

        grad_ln_D_up_auto, grad_ln_D_dn_auto, lap_ln_D_up_auto, lap_ln_D_dn_auto = _compute_grads_and_laplacian_ln_Det_auto(
            geminal_data=geminal_ao_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        # Auto path is autodiff through compute_ln_det -> bottlenecked by ao_eval
        # (fp32 in mixed mode); analytic path is fp64 (ao_grad_lap), so achievable
        # agreement is bounded by ao_eval, not det_grad_lap.
        atol, rtol = get_tolerance_min(["ao_eval", "det_grad_lap"], "strict")
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_ln_D_up_analytic)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_ln_D_up_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(grad_ln_D_up_analytic),
            np.asarray(grad_ln_D_up_auto),
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_ln_D_dn_analytic)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_ln_D_dn_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(grad_ln_D_dn_analytic),
            np.asarray(grad_ln_D_dn_auto),
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_ln_D_up_analytic)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_ln_D_up_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(lap_ln_D_up_analytic),
            np.asarray(lap_ln_D_up_auto),
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_ln_D_dn_analytic)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_ln_D_dn_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(lap_ln_D_dn_analytic),
            np.asarray(lap_ln_D_dn_auto),
            atol=atol,
            rtol=rtol,
        )

        jax.clear_caches()


def _prepare_ratio_determinant_inputs():
    (
        _structure_data,
        _aos_data,
        _mos_data_up,
        _mos_data_dn,
        geminal_mo_data,
        _coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "water_ccecp_ccpvqz.h5"),
        store_tuple=True,
    )

    geminal_mo_data.sanity_check()

    rng = np.random.default_rng(0)
    r_up_carts = rng.uniform(-1.0, 1.0, size=(geminal_mo_data.num_electron_up, 3))
    r_dn_carts = rng.uniform(-1.0, 1.0, size=(geminal_mo_data.num_electron_dn, 3))

    geminal_old = compute_geminal_all_elements(
        geminal_data=geminal_mo_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    P, L, U = jsp_linalg.lu(geminal_old)
    n = geminal_old.shape[0]
    I = jnp.eye(n, dtype=geminal_old.dtype)
    Y = jsp_linalg.solve_triangular(L, jnp.dot(P.T, I), lower=True)
    A_old_inv = jsp_linalg.solve_triangular(U, Y, lower=False)

    return geminal_mo_data, r_up_carts, r_dn_carts, A_old_inv


def _build_three_grid_moves_for_determinant(old_r_up_carts: np.ndarray, old_r_dn_carts: np.ndarray, pattern: str):
    new_r_up_carts_arr = np.repeat(old_r_up_carts[None, ...], 3, axis=0)
    new_r_dn_carts_arr = np.repeat(old_r_dn_carts[None, ...], 3, axis=0)

    up_alt_idx = min(1, old_r_up_carts.shape[0] - 1)
    dn_alt_idx = min(1, old_r_dn_carts.shape[0] - 1)

    if pattern == "all_moved":
        new_r_up_carts_arr[0, 0, 0] += 0.05
        new_r_dn_carts_arr[1, 0, 1] -= 0.07
        new_r_up_carts_arr[2, up_alt_idx, 2] += 0.09
    elif pattern == "none_moved":
        pass
    elif pattern == "mixed":
        new_r_up_carts_arr[0, 0, 0] += 0.05
        new_r_dn_carts_arr[2, dn_alt_idx, 1] += 0.04
    else:
        raise ValueError(f"Unknown pattern for ratio grid construction: {pattern}")

    return new_r_up_carts_arr, new_r_dn_carts_arr


class TestDeterminantRatio:
    @pytest.mark.light
    @pytest.mark.parametrize("pattern", ["all_moved", "none_moved", "mixed"])
    def test_ratio_determinant_rank1_update(self, pattern: str):
        """Rank-1 determinant ratio matches debug across three-grid scenarios."""
        geminal_data, old_r_up_carts, old_r_dn_carts, A_old_inv = _prepare_ratio_determinant_inputs()

        new_r_up_carts_arr, new_r_dn_carts_arr = _build_three_grid_moves_for_determinant(
            old_r_up_carts,
            old_r_dn_carts,
            pattern,
        )

        ratio_debug = _compute_ratio_determinant_part_debug(
            geminal_data=geminal_data,
            old_r_up_carts=old_r_up_carts,
            old_r_dn_carts=old_r_dn_carts,
            new_r_up_carts_arr=new_r_up_carts_arr,
            new_r_dn_carts_arr=new_r_dn_carts_arr,
        )

        ratio_rank1 = _compute_ratio_determinant_part_rank1_update(
            geminal_data=geminal_data,
            A_old_inv=A_old_inv,
            old_r_up_carts=old_r_up_carts,
            old_r_dn_carts=old_r_dn_carts,
            new_r_up_carts_arr=new_r_up_carts_arr,
            new_r_dn_carts_arr=new_r_dn_carts_arr,
        )

        atol, rtol = get_tolerance("det_eval", "strict")
        atol_c, rtol_c = atol, rtol
        assert not np.any(np.isnan(np.asarray(np.asarray(ratio_debug)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(ratio_rank1)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(ratio_debug), np.asarray(ratio_rank1), atol=atol, rtol=rtol)

        if pattern == "none_moved":
            assert not np.any(np.isnan(np.asarray(np.asarray(ratio_debug)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.ones_like(np.asarray(ratio_debug))))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(ratio_debug), np.ones_like(np.asarray(ratio_debug)), atol=atol_c, rtol=rtol_c)

        jax.clear_caches()


class TestDeterminantLogFastPath:
    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5", "H_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
    def test_compute_ln_det_geminal_all_elements_fast_forward(self, trexio_file):
        """Forward value of compute_ln_det_geminal_all_elements_fast must match the standard variant."""
        _, _, _, _, geminal_data, _ = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )
        rng = np.random.default_rng(0)
        n_up = geminal_data.num_electron_up
        n_dn = geminal_data.num_electron_dn

        atol, rtol = get_tolerance("det_eval", "strict")
        for sample in range(10):
            sample_label = str(sample)  # noqa: F841 - captured by the GDD iteration receipt
            r_up = jnp.array(rng.standard_normal((n_up, 3)) * 1.2, dtype=jnp.float64)
            r_dn = jnp.array(rng.standard_normal((n_dn, 3)) * 1.2, dtype=jnp.float64)
            G = compute_geminal_all_elements(geminal_data, r_up, r_dn)
            G_inv = jnp.linalg.inv(G)

            val_ref = float(compute_ln_det_geminal_all_elements(geminal_data, r_up, r_dn))
            val_fast = float(compute_ln_det_geminal_all_elements_fast(geminal_data, r_up, r_dn, G_inv))

            assert np.isfinite(val_ref), f"Reference value is not finite: {val_ref}"
            assert np.isfinite(val_fast), f"Fast value is not finite: {val_fast}"
            np.testing.assert_allclose(
                val_fast,
                val_ref,
                atol=atol,
                rtol=rtol,
                err_msg=f"Forward mismatch: fast={val_fast:.15f}, ref={val_ref:.15f}",
            )

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5", "H_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
    def test_compute_ln_det_geminal_all_elements_fast_backward(self, trexio_file):
        """Gradient of compute_ln_det_geminal_all_elements_fast w.r.t. geminal_data must match the standard variant."""
        _, _, _, _, geminal_data, _ = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )
        rng = np.random.default_rng(1)
        n_up = geminal_data.num_electron_up
        n_dn = geminal_data.num_electron_dn

        grad_ref_fn = jax.grad(compute_ln_det_geminal_all_elements, argnums=0)
        grad_fast_fn = jax.grad(compute_ln_det_geminal_all_elements_fast, argnums=0)

        # Backward-AD comparison: ref uses SVD-pseudoinverse custom VJP, fast uses
        # caller-supplied G_inv; both VJPs propagate through compute_geminal_all_elements
        # which crosses ao_eval/mo_eval/det_eval. Tolerance is bounded by the
        # loosest zone on the path.
        atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "det_eval"), "strict")
        for _ in range(10):
            r_up = jnp.array(rng.standard_normal((n_up, 3)) * 1.2, dtype=jnp.float64)
            r_dn = jnp.array(rng.standard_normal((n_dn, 3)) * 1.2, dtype=jnp.float64)
            G = compute_geminal_all_elements(geminal_data, r_up, r_dn)
            G_inv = jnp.linalg.inv(G)

            grad_ref = grad_ref_fn(geminal_data, r_up, r_dn)
            grad_fast = grad_fast_fn(geminal_data, r_up, r_dn, G_inv)

            ref_leaves, ref_tree = jax.tree_util.tree_flatten(grad_ref)
            fast_leaves, fast_tree = jax.tree_util.tree_flatten(grad_fast)
            assert fast_tree == ref_tree
            assert len(ref_leaves) > 0
            for a, b in zip(ref_leaves, fast_leaves, strict=True):
                np.testing.assert_allclose(
                    np.asarray(a),
                    np.asarray(b),
                    rtol=rtol,
                    atol=atol,
                    err_msg="Backward mismatch in compute_ln_det_geminal_all_elements_fast",
                )

        jax.clear_caches()


# ---------------------------------------------------------------------------
# L1: Unit tests for Geminal_data.symmetrize_lambda
# ---------------------------------------------------------------------------


def _make_geminal_with_lambda(lam, num_up=1, num_dn=1, time_reversal=True):
    """Build a minimal Geminal_data whose lambda_matrix is *lam*.

    Only the fields inspected by ``symmetrize_lambda`` are populated;
    orbital data are left as default (unused in that method).

    The ``time_reversal`` kwarg selects the paired (``True``) or
    unrestricted (``False``) path without building orbital fixtures.
    """
    return Geminal_data(
        num_electron_up=num_up,
        num_electron_dn=num_dn,
        lambda_matrix=np.asarray(lam, dtype=np.float64),
        time_reversal=time_reversal,
    )


class TestGeminalSymmetrization:
    @pytest.mark.light
    def test_symmetrize_lambda_square_symmetric(self):
        """L1-1: square symmetric lambda -> 0.5*(mat+mat.T)."""
        rng = np.random.RandomState(0)
        n = 5
        lam_sym = rng.randn(n, n)
        lam_sym = 0.5 * (lam_sym + lam_sym.T)  # make symmetric
        gd = _make_geminal_with_lambda(lam_sym)

        mat = rng.randn(n, n)  # intentionally non-symmetric input
        result = gd.symmetrize_lambda(mat)
        expected = 0.5 * (mat + mat.T)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(result, result.T)

    @pytest.mark.light
    def test_symmetrize_lambda_square_nonsymmetric(self):
        """L1-2: open-shell / UHF path (time_reversal=False) -> no symmetrization."""
        rng = np.random.RandomState(1)
        n = 5
        lam_nonsym = rng.randn(n, n)
        lam_nonsym[0, 1] = 100.0  # force asymmetry
        # time_reversal=False corresponds to the UHF / open-shell channel
        # independence regime; symmetrize_lambda must be a no-op.
        gd = _make_geminal_with_lambda(lam_nonsym, time_reversal=False)

        mat = rng.randn(n, n)
        result = gd.symmetrize_lambda(mat)
        np.testing.assert_array_equal(result, mat)

    @pytest.mark.light
    def test_symmetrize_lambda_rect_paired_symmetric(self):
        """L1-3: a rectangular unrestricted lambda stays free even with a symmetric paired block."""
        rng = np.random.RandomState(2)
        n_up = 3
        n_extra = 2
        n_cols = n_up + n_extra  # 5
        paired = rng.randn(n_up, n_up)
        paired = 0.5 * (paired + paired.T)
        unpaired = rng.randn(n_up, n_extra)
        lam_rect = np.column_stack([paired, unpaired])
        gd = _make_geminal_with_lambda(lam_rect, num_up=n_up + n_extra, num_dn=n_up, time_reversal=False)

        mat = rng.randn(n_up, n_cols)
        result = gd.symmetrize_lambda(mat)
        np.testing.assert_array_equal(result, mat)

    @pytest.mark.light
    def test_symmetrize_lambda_rect_paired_nonsymmetric(self):
        """L1-4: open-shell rectangular lambda (time_reversal=False) -> no symmetrization."""
        rng = np.random.RandomState(3)
        n_up = 3
        n_extra = 2
        paired = rng.randn(n_up, n_up)
        paired[0, 1] = 100.0  # force asymmetry
        unpaired = rng.randn(n_up, n_extra)
        lam_rect = np.column_stack([paired, unpaired])
        gd = _make_geminal_with_lambda(lam_rect, num_up=n_up + n_extra, num_dn=n_up, time_reversal=False)

        mat = rng.randn(n_up, n_up + n_extra)
        result = gd.symmetrize_lambda(mat)
        np.testing.assert_array_equal(result, mat)

    @pytest.mark.light
    def test_symmetrize_lambda_none(self):
        """L1-5: lambda_matrix=None -> no-op."""
        gd_none = Geminal_data(lambda_matrix=None)
        mat = np.random.RandomState(4).randn(3, 3)
        result = gd_none.symmetrize_lambda(mat)
        np.testing.assert_array_equal(result, mat)

    @pytest.mark.light
    def test_symmetrize_lambda_1d(self):
        """L1-6: 1-D lambda (ndim!=2) -> no-op."""
        gd = _make_geminal_with_lambda(np.array([1.0, 2.0, 3.0]))
        mat = np.random.RandomState(5).randn(3, 3)
        result = gd.symmetrize_lambda(mat)
        np.testing.assert_array_equal(result, mat)


# ---------------------------------------------------------------------------
# L2: apply_block_update symmetry preservation for Geminal_data
# ---------------------------------------------------------------------------


class TestGeminalBlockUpdateSymmetry:
    @pytest.mark.light
    def test_apply_block_update_square_symmetric_preserved(self):
        """L2-1: symmetric square lambda stays symmetric after non-symmetric delta."""
        rng = np.random.RandomState(10)
        n = 5
        lam_sym = rng.randn(n, n)
        lam_sym = 0.5 * (lam_sym + lam_sym.T)
        gd = _make_geminal_with_lambda(lam_sym)

        delta = rng.randn(n, n)  # intentionally non-symmetric
        lr = 0.1
        updated_values = lam_sym + lr * delta
        block = make_test_block(
            "lambda_matrix",
            updated_values,
            field="real",
            derivative_policy="direct_real",
            sync_scope="sampler",
        )
        new_gd = gd.apply_block_update(block)
        new_lam = np.asarray(new_gd.lambda_matrix)
        np.testing.assert_array_equal(new_lam, new_lam.T)

    @pytest.mark.light
    def test_apply_block_update_square_nonsymmetric_free(self):
        """L2-2: open-shell (time_reversal=False) lambda is not symmetrized after update."""
        rng = np.random.RandomState(11)
        n = 5
        lam_nonsym = rng.randn(n, n)
        lam_nonsym[0, 1] = 100.0
        gd = _make_geminal_with_lambda(lam_nonsym, time_reversal=False)

        updated_values = lam_nonsym + 0.1 * rng.randn(n, n)
        block = make_test_block(
            "lambda_matrix",
            updated_values,
            field="real",
            derivative_policy="direct_real",
            sync_scope="sampler",
        )
        new_gd = gd.apply_block_update(block)
        new_lam = np.asarray(new_gd.lambda_matrix)
        # Should be the raw updated values (not symmetrized)
        np.testing.assert_array_equal(new_lam, updated_values)

    @pytest.mark.light
    def test_apply_block_update_rect_open_shell_no_symmetrization(self):
        """L2-3: apply_block_update writes the raw rectangular unrestricted lambda."""
        rng = np.random.RandomState(12)
        n_up = 4
        n_extra = 2
        paired = rng.randn(n_up, n_up)
        paired = 0.5 * (paired + paired.T)
        unpaired = rng.randn(n_up, n_extra)
        lam_rect = np.column_stack([paired, unpaired])
        gd = _make_geminal_with_lambda(lam_rect, num_up=n_up + n_extra, num_dn=n_up, time_reversal=False)

        delta = rng.randn(n_up, n_up + n_extra)
        lr = 0.1
        updated_values = lam_rect + lr * delta
        block = make_test_block(
            "lambda_matrix",
            updated_values,
            field="real",
            derivative_policy="direct_real",
            sync_scope="sampler",
        )
        new_gd = gd.apply_block_update(block)
        new_lam = np.asarray(new_gd.lambda_matrix)
        np.testing.assert_array_equal(new_lam, updated_values)

    @pytest.mark.light
    @pytest.mark.parametrize("field", ["real", "complex"])
    def test_apply_block_update_rect_time_reversal_projects_only_paired_columns(self, field):
        """ROHF/kROHF projects the paired block and keeps the unpaired column exactly."""
        if field == "real":
            update = np.array([[4.0, 6.0, -8.0], [2.0, 10.0, 12.0]])
            expected = np.array([[4.0, 4.0, -8.0], [4.0, 10.0, 12.0]])
            derivative_policy = "direct_real"
        else:
            update = np.array([[4.0 + 2.0j, 6.0 + 6.0j, -8.0 - 4.0j], [2.0 - 2.0j, 10.0 - 8.0j, 12.0 + 16.0j]])
            expected = np.array([[4.0, 4.0 + 4.0j, -8.0 - 4.0j], [4.0 - 4.0j, 10.0, 12.0 + 16.0j]])
            derivative_policy = "holomorphic"
        gd = Geminal_data(
            num_electron_up=2,
            num_electron_dn=1,
            lambda_matrix=np.array([[1.0, 0.0, 3.0], [0.0, 2.0, -1.0]], dtype=update.dtype),
            time_reversal=True,
        )

        projected = gd.symmetrize_lambda(update)
        np.testing.assert_array_equal(projected, expected)
        block = make_test_block("lambda_matrix", update, field=field, derivative_policy=derivative_policy, sync_scope="sampler")
        stored = gd.apply_block_update(block).lambda_matrix
        np.testing.assert_array_equal(stored, expected)


class TestParameterGradientAssembly:
    @pytest.mark.light
    @pytest.mark.parametrize("num_walkers", [1, 2, 4])
    def test_collect_param_grads_walker_shapes(self, num_walkers):
        """collect_param_grads must concatenate up/dn along the parameter axis, not walkers."""
        rng = np.random.default_rng(42)

        # -- build two different AO bases for up / dn spins --
        def _make_aos(num_prim, seed):
            r = np.random.default_rng(seed)
            sd = Structure_data(
                pbc_flag=False,
                positions=np.zeros((1, 3)),
                atomic_numbers=(1,),
                element_symbols=("X",),
                atomic_labels=("X",),
            )
            return AOs_sphe_gto_data._from_full_basis(
                structure_data=sd,
                nucleus_index=tuple([0] * num_prim),
                num_ao=num_prim,
                num_ao_prim=num_prim,
                orbital_indices=tuple(range(num_prim)),
                exponents=jnp.array(r.uniform(0.5, 2.0, num_prim)),
                coefficients=jnp.array(r.uniform(0.5, 1.5, num_prim)),
                angular_momentums=tuple([0] * num_prim),
                magnetic_quantum_numbers=tuple([0] * num_prim),
            )

        num_prim_up, num_prim_dn = 5, 3
        aos_up = _make_aos(num_prim_up, seed=10)
        aos_dn = _make_aos(num_prim_dn, seed=20)

        geminal = Geminal_data(
            num_electron_up=2,
            num_electron_dn=2,
            orb_data_up_spin=aos_up,
            orb_data_dn_spin=aos_dn,
            lambda_matrix=rng.standard_normal((num_prim_up, num_prim_dn)),
        )

        # -- simulate vmap-batched gradient (walker dim prepended on pytree leaves) --
        grad_lambda = jnp.array(rng.standard_normal((num_walkers, num_prim_up, num_prim_dn)))
        grad_exp_up = jnp.array(rng.standard_normal((num_walkers, num_prim_up)))
        grad_exp_dn = jnp.array(rng.standard_normal((num_walkers, num_prim_dn)))
        grad_coeff_up = jnp.array(rng.standard_normal((num_walkers, num_prim_up)))
        grad_coeff_dn = jnp.array(rng.standard_normal((num_walkers, num_prim_dn)))

        grad_geminal = geminal.replace(
            lambda_matrix=grad_lambda,
            orb_data_up_spin=aos_up.replace(radial=aos_up.radial.replace(exponents=grad_exp_up, coefficients=grad_coeff_up)),
            orb_data_dn_spin=aos_dn.replace(radial=aos_dn.radial.replace(exponents=grad_exp_dn, coefficients=grad_coeff_dn)),
        )

        grads = geminal.collect_param_grads(grad_geminal)

        # -- shape checks --
        assert grads["lambda_matrix"].shape == (num_walkers, num_prim_up, num_prim_dn)
        assert grads["lambda_basis_exp"].shape == (num_walkers, num_prim_up + num_prim_dn)
        assert grads["lambda_basis_coeff"].shape == (num_walkers, num_prim_up + num_prim_dn)

        # -- value checks: up part then dn part along last axis --
        np.testing.assert_array_equal(grads["lambda_basis_exp"][:, :num_prim_up], grad_exp_up)
        np.testing.assert_array_equal(grads["lambda_basis_exp"][:, num_prim_up:], grad_exp_dn)
        np.testing.assert_array_equal(grads["lambda_basis_coeff"][:, :num_prim_up], grad_coeff_up)
        np.testing.assert_array_equal(grads["lambda_basis_coeff"][:, num_prim_up:], grad_coeff_dn)


# ---------------------------------------------------------------------------
# Det streaming state
# ---------------------------------------------------------------------------


def _build_geminal_inverse(geminal_data, r_up_carts, r_dn_carts):
    """Compute G(r_up, r_dn)^{-1} for a freshly-arrived configuration."""
    A = compute_geminal_all_elements(
        geminal_data=geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
    return jnp.linalg.inv(A)


# ---------------------------------------------------------------------------
# Det_ratio_streaming_state: slim ratio streaming state
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _compute_ratio_determinant_part_split_spin with det_ratio_state plumb
# ---------------------------------------------------------------------------


class TestDeterminantStreamingState:
    @pytest.mark.light
    def test_conditioned_sm_inverse_chain_matches_fresh_cached_state(self):
        """A 150-hop SM inverse agrees with fresh final-position G and fast derivatives.

        The reference matrix is assembled from the *same cached AO state* as
        the production helper used by MCMC, GFMC_n, and GFMC_t.  This isolates
        inverse-chain error from AO re-evaluation error without duplicating the
        Sherman--Morrison expression in the test.  Every accepted move is
        screened before the comparison so this strict statement is made only
        while ``cond_2(G)`` is within the declared well-conditioned domain.
        At the final positions, independently re-evaluated orbitals must yield
        the cached G and carried inverse.  The carried inverse itself, without
        refreshing it, must give the fresh eval-G inverse's fast derivatives
        under det_grad_lap strict.  Full-recomputation gradients and Laplacians
        must also agree in the same precision mode using the existing strict
        ao_eval / det_grad_lap policy for the two different orbital routes.
        """
        (_, _, _, _, geminal, _) = read_trexio_file(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "trexio_example_files",
                "water_ccecp_ccpvqz.h5",
            ),
            store_tuple=True,
        )
        assert isinstance(geminal, SD_data)
        assert geminal.num_electron_up == geminal.num_electron_dn

        rng = np.random.RandomState(20260826)
        n_up = geminal.num_electron_up
        n_dn = geminal.num_electron_dn
        max_cond_g = 1.0e3
        min_abs_rho = 0.20

        # Deterministic rejection screen for the initial production G.
        best_initial_cond = np.inf
        for _ in range(128):
            r_up = jnp.asarray(rng.normal(scale=1.2, size=(n_up, 3)))
            r_dn = jnp.asarray(rng.normal(scale=1.2, size=(n_dn, 3)))
            G = np.asarray(compute_geminal_all_elements(geminal, r_up, r_dn))
            cond_initial = float(np.linalg.cond(G, p=2))
            best_initial_cond = min(best_initial_cond, cond_initial)
            if np.isfinite(cond_initial) and cond_initial <= max_cond_g:
                break
        else:
            pytest.fail(
                "failed to generate a well-conditioned initial determinant in 128 attempts; "
                f"best cond_2(G)={best_initial_cond:.3e}"
            )

        det_state = _init_det_ratio_streaming_state(geminal, r_up, r_dn)

        def _cached_g(state):
            # Closed-shell SD: paired_dn is the occupied down-orbital table.
            return state.ao_up.T @ state.paired_dn

        def _fresh_svd_inverse(state):
            G_cached = np.asarray(_cached_g(state))
            U, singular_values, Vh = np.linalg.svd(G_cached, full_matrices=False)
            return (np.conj(Vh).T * (1.0 / singular_values)[None, :]) @ np.conj(U).T

        A_inv = jnp.asarray(_fresh_svd_inverse(det_state))
        worst_relative_error = 0.0
        observed_cond = []

        for hop in range(150):
            hop_label = str(hop)
            moved_up = hop % 2 == 0
            idx = (hop // 2) % (n_up if moved_up else n_dn)

            for _ in range(128):
                delta = jnp.asarray(rng.normal(scale=0.04, size=3))
                r_up_new = r_up.at[idx].add(delta) if moved_up else r_up
                r_dn_new = r_dn if moved_up else r_dn.at[idx].add(delta)

                if moved_up:
                    v = _compute_v_up_move_from_det_ratio_state(
                        geminal_data=geminal,
                        state=det_state,
                        moved_index=idx,
                        r_up_carts_proposed=r_up_new,
                    )[:, None]
                    u = jax.nn.one_hot(idx, n_up, dtype=A_inv.dtype)[:, None]
                else:
                    u = _compute_u_dn_move_from_det_ratio_state(
                        geminal_data=geminal,
                        state=det_state,
                        moved_index=idx,
                        r_dn_carts_proposed=r_dn_new,
                    )[:, None]
                    v = jax.nn.one_hot(idx, n_up, dtype=A_inv.dtype)[:, None]

                rho, A_inv_trial = _sherman_morrison_ratio_and_inverse(A_inv, u, v)
                trial_state = _advance_det_ratio_streaming_state(
                    geminal,
                    det_state,
                    jnp.bool_(moved_up),
                    jnp.int64(idx),
                    r_up_new,
                    r_dn_new,
                )
                cond_g = float(np.linalg.cond(np.asarray(_cached_g(trial_state)), p=2))
                if np.isfinite(cond_g) and cond_g <= max_cond_g and float(np.abs(rho)) >= min_abs_rho:
                    break
            else:
                pytest.fail(f"failed to keep hop {hop} in the declared conditioned domain")

            A_inv = A_inv_trial
            r_up, r_dn, det_state = r_up_new, r_dn_new, trial_state
            observed_cond.append(cond_g)

            A_fresh = _fresh_svd_inverse(det_state)
            scale = max(float(np.max(np.abs(A_fresh))), 1.0e-30)
            relative_error = float(np.max(np.abs(np.asarray(A_inv) - A_fresh)) / scale)
            worst_relative_error = max(worst_relative_error, relative_error)

            # The central strict determinant tolerance is the numerical verdict;
            # the explicit residual keeps the claim tied to inverse validity.
            atol, rtol = get_tolerance("det_eval", "strict")
            np.testing.assert_allclose(A_inv, A_fresh, atol=atol, rtol=rtol)
            residual = np.asarray(_cached_g(det_state)) @ np.asarray(A_inv) - np.eye(n_up)
            assert np.max(np.abs(residual)) <= max(atol, rtol * cond_g)

        assert len(observed_cond) == 150
        assert max(observed_cond) <= max_cond_g
        assert np.isfinite(worst_relative_error)

        # A mutually stale cache and inverse can agree above even when neither
        # belongs to the final positions.  Re-evaluate G without the AO cache.
        G_final = np.asarray(compute_geminal_all_elements(geminal, r_up, r_dn))
        atol, rtol = get_tolerance("det_eval", "strict")
        np.testing.assert_allclose(_cached_g(det_state), G_final, atol=atol, rtol=rtol, err_msg="fresh final-position geminal")
        cond_final = float(np.linalg.cond(G_final, p=2))
        assert np.isfinite(cond_final)
        assert cond_final <= max_cond_g
        np.testing.assert_allclose(
            A_inv, np.linalg.inv(G_final), atol=atol, rtol=rtol, err_msg="carried final-position inverse"
        )

        # Do not replace A_inv by the fresh reference: this is the inverse that
        # the streaming update chain actually passes to its fast consumer.
        fast_derivatives = compute_grads_and_laplacian_ln_Det_fast(
            geminal_data=geminal, r_up_carts=r_up, r_dn_carts=r_dn, geminal_inverse=A_inv
        )
        fresh_inverse_derivatives = compute_grads_and_laplacian_ln_Det_fast(
            geminal_data=geminal, r_up_carts=r_up, r_dn_carts=r_dn, geminal_inverse=jnp.asarray(np.linalg.inv(G_final))
        )
        fresh_derivatives = compute_grads_and_laplacian_ln_Det(geminal_data=geminal, r_up_carts=r_up, r_dn_carts=r_dn)
        assert len(fast_derivatives) == len(fresh_inverse_derivatives) == len(fresh_derivatives) == 4
        derivative_components = ("gradient-up", "gradient-down", "laplacian-up", "laplacian-down")
        atol, rtol = get_tolerance("det_grad_lap", "strict")
        for derivative_label, actual, reference in zip(
            derivative_components, fast_derivatives, fresh_inverse_derivatives, strict=True
        ):
            assert actual.shape == reference.shape
            np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, err_msg="carried-inverse same-zone derivatives")

        # Same policy as the existing fast/debug test: mixed eval-G uses
        # ao_eval values, whereas full recomputation uses fused ao_grad_lap
        # values.  Do not judge this cross-route comparison by the output zone
        # alone; the strict inverse-chain check above remains separate.
        atol, rtol = get_tolerance_min(["ao_eval", "det_grad_lap"], "strict")
        for derivative_label, actual, reference in zip(derivative_components, fast_derivatives, fresh_derivatives, strict=True):
            assert actual.shape == reference.shape
            np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, err_msg="carried-inverse fast derivatives")

    @pytest.mark.parametrize("det_kind", ["sd", "agp", "agpn"])
    @pytest.mark.light
    def test_det_winv_multi_hop_matches_full_debug_oracle(self, det_kind: str):
        """Production winv ratios equal the unchanged full contraction oracle.

        Alternating accepted up/down moves exercises both Sherman--Morrison
        update formulae.  The three determinant representations deliberately
        share the same starting wavefunction but dispatch through independent
        SD, dense-AGP, and diagonal-AGPn coefficient paths.
        """
        (_, _, _, _, sd, _) = read_trexio_file(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "trexio_example_files",
                "water_ccecp_ccpvqz.h5",
            ),
            store_tuple=True,
        )
        assert isinstance(sd, SD_data)
        geminal = {"sd": sd, "agp": to_AGP(sd), "agpn": to_AGPn(sd)}[det_kind]
        n_up = geminal.num_electron_up
        n_dn = geminal.num_electron_dn
        rng = np.random.RandomState(918)
        r_up = jnp.asarray(1.2 * rng.normal(size=(n_up, 3)))
        r_dn = jnp.asarray(1.2 * rng.normal(size=(n_dn, 3)))
        shifts = jnp.asarray(
            [[0.17, 0.0, 0.0], [-0.17, 0.0, 0.0], [0.0, 0.13, 0.0], [0.0, 0.0, -0.11]],
            dtype=jnp.float64,
        )

        def _candidate_positions(ru, rd):
            return ru[:, None, :] + shifts[None, :, :], rd[:, None, :] + shifts[None, :, :]

        A_inv = _build_geminal_inverse(geminal, r_up, r_dn)
        det_state = _init_det_ratio_streaming_state(geminal, r_up, r_dn)
        cand_up, cand_dn = _candidate_positions(r_up, r_dn)
        candidate_state = _init_det_candidate_mesh_state(geminal, cand_up, cand_dn)
        winv_state = _init_det_winv_state(geminal, A_inv, det_state)

        def _assert_oracle():
            full = _compute_ratio_determinant_from_candidate_mesh_state(geminal, A_inv, det_state, candidate_state)
            fast = _compute_ratio_determinant_from_winv_state(geminal, candidate_state, winv_state)
            np.testing.assert_allclose(
                fast,
                full,
                atol=ATOL_PYTHON_FP64_AGGREGATOR_LOOSE,
                rtol=RTOL_PYTHON_FP64_AGGREGATOR_LOOSE,
            )

        _assert_oracle()
        for hop in range(6):
            move_up = hop % 2 == 0 or n_dn == 0
            idx = hop % (n_up if move_up else n_dn)
            r_up_new = r_up
            r_dn_new = r_dn
            delta = jnp.asarray(rng.normal(scale=0.08, size=3))
            if move_up:
                r_up_new = r_up.at[idx].add(delta)
            else:
                r_dn_new = r_dn.at[idx].add(delta)
            det_state_new = _advance_det_ratio_streaming_state(
                geminal, det_state, jnp.bool_(move_up), jnp.int64(idx), r_up_new, r_dn_new
            )
            cand_up, cand_dn = _candidate_positions(r_up_new, r_dn_new)
            candidate_state = _advance_det_candidate_mesh_state(
                geminal,
                candidate_state,
                jnp.bool_(move_up),
                jnp.int64(idx),
                cand_up,
                cand_dn,
            )
            r_up, r_dn = r_up_new, r_dn_new
            A_inv = _build_geminal_inverse(geminal, r_up, r_dn)
            det_state = det_state_new
            # Derived-quantity contract: the production consumers rebuild the
            # winv weights from (A_inv, det_state) at every evaluation; no
            # advance chain exists (see Det_winv_state).
            winv_state = _init_det_winv_state(geminal, A_inv, det_state)
            _assert_oracle()

    @pytest.mark.parametrize(
        "trexio_file",
        ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"],
    )
    @pytest.mark.light
    def test_streaming_det_state_against_full(self, trexio_file: str):
        """Det streaming state, after K random single-electron moves, must
        reproduce ``compute_grads_and_laplacian_ln_Det_fast`` at the resulting
        configuration.
        """
        (
            _,
            _,
            _,
            _,
            geminal_mo_data,
            _,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        n_up = geminal_mo_data.num_electron_up
        n_dn = geminal_mo_data.num_electron_dn

        rng = np.random.RandomState(13)
        r_up = jnp.asarray(4.0 * rng.rand(n_up, 3) - 2.0)
        r_dn = jnp.asarray(4.0 * rng.rand(n_dn, 3) - 2.0)

        A_inv = _build_geminal_inverse(geminal_mo_data, r_up, r_dn)
        state = _init_grads_laplacian_ln_Det_streaming_state(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            geminal_inverse=A_inv,
        )

        K = 32
        for k in range(K):
            # alternate spin choices, but skip the spin if it has 0 electrons
            if n_dn == 0:
                spin_is_up = True
            elif n_up == 0:
                spin_is_up = False
            else:
                spin_is_up = bool(rng.randint(0, 2))

            if spin_is_up:
                idx = int(rng.randint(0, n_up))
                r_up = r_up.at[idx].set(jnp.asarray(rng.normal(size=(3,)) * 0.4 + np.asarray(r_up[idx])))
            else:
                idx = int(rng.randint(0, n_dn))
                r_dn = r_dn.at[idx].set(jnp.asarray(rng.normal(size=(3,)) * 0.4 + np.asarray(r_dn[idx])))

            A_new_inv = _build_geminal_inverse(geminal_mo_data, r_up, r_dn)
            state = _advance_grads_laplacian_ln_Det_streaming_state(
                geminal_data=geminal_mo_data,
                state=state,
                moved_spin_is_up=jnp.bool_(spin_is_up),
                moved_index=jnp.int32(idx),
                r_up_carts_new=r_up,
                r_dn_carts_new=r_dn,
                A_new_inv=A_new_inv,
            )

        # Reference: fresh fast call at the final configuration.
        A_inv_final = _build_geminal_inverse(geminal_mo_data, r_up, r_dn)
        grad_up_ref, grad_dn_ref, lap_up_ref, lap_dn_ref = compute_grads_and_laplacian_ln_Det_fast(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            geminal_inverse=A_inv_final,
        )

        atol, rtol = get_tolerance("det_grad_lap", "strict")
        np.testing.assert_allclose(state.grad_ln_D_up, grad_up_ref, atol=atol, rtol=rtol)
        np.testing.assert_allclose(state.grad_ln_D_dn, grad_dn_ref, atol=atol, rtol=rtol)
        np.testing.assert_allclose(state.lap_ln_D_up, lap_up_ref, atol=atol, rtol=rtol)
        np.testing.assert_allclose(state.lap_ln_D_dn, lap_dn_ref, atol=atol, rtol=rtol)

        # paired_up_lambda invariant (= ao_up.T @ lambda_paired).
        state_ratio_ref = _init_det_ratio_streaming_state(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
        )
        np.testing.assert_allclose(state.paired_up_lambda, state_ratio_ref.paired_up_lambda, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "trexio_file",
        ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"],
    )
    @pytest.mark.light
    def test_det_ratio_streaming_state_against_full(self, trexio_file: str):
        """The incremental AO state matches fresh AO/MO evaluation every hop.

        This is intentionally an AO-state gate, not an inverse-chain gate: the
        four cached tables are compared directly, before multiplication by a
        potentially ill-conditioned inverse.  Inverse-chain validity has its
        own conditioned-domain test above.

        Verifies the rank-1 advance invariants:
            ao_up           == compute_orb_api(orb_data_up_spin, r_up_carts)
            ao_dn           == compute_orb_api(orb_data_dn_spin, r_dn_carts)
            paired_dn       == lambda_paired @ ao_dn
            paired_up_lambda == ao_up.T @ lambda_paired
        """
        (
            _,
            _,
            _,
            _,
            geminal_mo_data,
            _,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        n_up = geminal_mo_data.num_electron_up
        n_dn = geminal_mo_data.num_electron_dn

        rng = np.random.RandomState(13)
        r_up = jnp.asarray(4.0 * rng.rand(n_up, 3) - 2.0)
        r_dn = jnp.asarray(4.0 * rng.rand(n_dn, 3) - 2.0)

        state = _init_det_ratio_streaming_state(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
        )

        atol, rtol = get_tolerance("det_grad_lap", "strict")
        K = 32
        for hop in range(K):
            # The receipt must retain every existing hop, not just the final one.
            hop_label = str(hop)
            # alternate spin choices, but skip the spin if it has 0 electrons
            if n_dn == 0:
                spin_is_up = True
            elif n_up == 0:
                spin_is_up = False
            else:
                spin_is_up = bool(rng.randint(0, 2))

            if spin_is_up:
                idx = int(rng.randint(0, n_up))
                r_up = r_up.at[idx].set(jnp.asarray(rng.normal(size=(3,)) * 0.4 + np.asarray(r_up[idx])))
            else:
                idx = int(rng.randint(0, n_dn))
                r_dn = r_dn.at[idx].set(jnp.asarray(rng.normal(size=(3,)) * 0.4 + np.asarray(r_dn[idx])))

            state = _advance_det_ratio_streaming_state(
                geminal_data=geminal_mo_data,
                state=state,
                moved_spin_is_up=jnp.bool_(spin_is_up),
                moved_index=jnp.int32(idx),
                r_up_carts_new=r_up,
                r_dn_carts_new=r_dn,
            )

            # Direct AO-state comparison at every accepted move.  Waiting until
            # only the final hop can hide a transient stale-column defect.
            state_ref = _init_det_ratio_streaming_state(
                geminal_data=geminal_mo_data,
                r_up_carts=r_up,
                r_dn_carts=r_dn,
            )
            np.testing.assert_allclose(state.ao_up, state_ref.ao_up, atol=atol, rtol=rtol)
            np.testing.assert_allclose(state.ao_dn, state_ref.ao_dn, atol=atol, rtol=rtol)
            np.testing.assert_allclose(state.paired_dn, state_ref.paired_dn, atol=atol, rtol=rtol)
            np.testing.assert_allclose(state.paired_up_lambda, state_ref.paired_up_lambda, atol=atol, rtol=rtol)

        # Reference: fresh init at the final configuration.
        state_ref = _init_det_ratio_streaming_state(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
        )

        np.testing.assert_allclose(state.ao_up, state_ref.ao_up, atol=atol, rtol=rtol)
        np.testing.assert_allclose(state.ao_dn, state_ref.ao_dn, atol=atol, rtol=rtol)
        np.testing.assert_allclose(state.paired_dn, state_ref.paired_dn, atol=atol, rtol=rtol)
        np.testing.assert_allclose(state.paired_up_lambda, state_ref.paired_up_lambda, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "trexio_file",
        ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"],
    )
    @pytest.mark.light
    def test_split_spin_with_det_ratio_state_against_fresh(self, trexio_file: str):
        """``_compute_ratio_determinant_part_split_spin`` must produce identical
        determinant ratios when called with a consistent ``det_ratio_state`` vs
        when called with ``det_ratio_state=None`` (reference path).

        Builds a small block-structured mesh (one shifted up-config + one
        shifted dn-config per direction) at a random reference configuration
        and checks both ratio paths agree at strict tolerance.
        """
        (
            _,
            _,
            _,
            _,
            geminal_mo_data,
            _,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        n_up = geminal_mo_data.num_electron_up
        n_dn = geminal_mo_data.num_electron_dn
        if n_up == 0 or n_dn == 0:
            pytest.skip("split-spin only meaningful when both spin sectors are non-empty")

        rng = np.random.RandomState(7)
        r_up = jnp.asarray(4.0 * rng.rand(n_up, 3) - 2.0)
        r_dn = jnp.asarray(4.0 * rng.rand(n_dn, 3) - 2.0)
        A_old_inv = _build_geminal_inverse(geminal_mo_data, r_up, r_dn)

        # Build a small block mesh: shift each up-electron by +0.1 in x;
        # likewise for dn.  (G_up = n_up, G_dn = n_dn.)
        new_r_up_shifted = []
        for k in range(n_up):
            shifted = np.array(r_up)
            shifted[k] = shifted[k] + np.array([0.1, 0.0, 0.0])
            new_r_up_shifted.append(shifted)
        new_r_up_shifted = jnp.asarray(np.stack(new_r_up_shifted, axis=0))  # (n_up, n_up, 3)

        new_r_dn_shifted = []
        for k in range(n_dn):
            shifted = np.array(r_dn)
            shifted[k] = shifted[k] + np.array([0.1, 0.0, 0.0])
            new_r_dn_shifted.append(shifted)
        new_r_dn_shifted = jnp.asarray(np.stack(new_r_dn_shifted, axis=0))  # (n_dn, n_dn, 3)

        # Reference: reference path.
        ratios_ref = _compute_ratio_determinant_part_split_spin(
            geminal_data=geminal_mo_data,
            A_old_inv=A_old_inv,
            old_r_up_carts=r_up,
            old_r_dn_carts=r_dn,
            new_r_up_shifted=new_r_up_shifted,
            new_r_dn_shifted=new_r_dn_shifted,
            det_ratio_state=None,
        )

        # Streaming path: build a fresh slim state at the reference config.
        state = _init_det_ratio_streaming_state(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
        )
        ratios_streaming = _compute_ratio_determinant_part_split_spin(
            geminal_data=geminal_mo_data,
            A_old_inv=A_old_inv,
            old_r_up_carts=r_up,
            old_r_dn_carts=r_dn,
            new_r_up_shifted=new_r_up_shifted,
            new_r_dn_shifted=new_r_dn_shifted,
            det_ratio_state=state,
        )

        atol, rtol = get_tolerance("det_ratio", "strict")
        np.testing.assert_allclose(ratios_streaming, ratios_ref, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Geminal class split tests
# ---------------------------------------------------------------------------


class TestGeminalDataTypes:
    @pytest.mark.light
    def test_geminal_data_alias_to_AGP_data(self):
        """``Geminal_data`` is an alias for ``AGP_data``."""
        assert Geminal_data is AGP_data


class TestAGPData:
    @pytest.mark.light
    def test_AGP_data_build_default_time_reversal_inferred(self):
        """``AGP_data.build`` infers ``time_reversal`` from electron counts."""
        n = 2
        aos_data = AOs_sphe_gto_data()  # default empty
        closed = AGP_data.build(
            num_electron_up=n,
            num_electron_dn=n,
            orb_data_up_spin=aos_data,
            orb_data_dn_spin=aos_data,
            lambda_matrix=np.zeros((0, 0), dtype=np.float64),
        )
        assert closed.time_reversal is True
        open_shell = AGP_data.build(
            num_electron_up=n + 1,
            num_electron_dn=n,
            orb_data_up_spin=aos_data,
            orb_data_dn_spin=aos_data,
            lambda_matrix=np.zeros((0, 1), dtype=np.float64),
        )
        assert open_shell.time_reversal is False

    @pytest.mark.light
    def test_AGP_data_build_open_shell_time_reversal_True_accepts_ROHF(self):
        """``time_reversal=True`` with ``N_up > N_dn`` is the ROHF / kROHF case
        (paired core has restricted TR-pair structure, unpaired alpha lives in
        ``lambda_unpaired`` columns only). The sanity check accepts it; the
        alpha-major (``N_up >= N_dn``) rule is verified separately at the
        sanity_check level.
        """
        aos_data = AOs_sphe_gto_data()
        # N_up > N_dn (open-shell, alpha-major) with time_reversal=True is the
        # ROHF / kROHF case -- accepted.
        rohf = AGP_data.build(
            num_electron_up=3,
            num_electron_dn=2,
            orb_data_up_spin=aos_data,
            orb_data_dn_spin=aos_data,
            lambda_matrix=np.zeros((0, 1), dtype=np.float64),
            time_reversal=True,
        )
        assert rohf.time_reversal is True
        assert int(rohf.num_electron_up) == 3
        assert int(rohf.num_electron_dn) == 2


def _build_minimal_mos_data(num_mo: int = 3, num_ao: int = 4) -> MOs_data:
    """Construct a minimal :class:`MOs_data` whose ``aos_data`` shape is
    consistent (``num_ao`` AOs, all l=0 primitives at the origin) so that
    ``MOs_data.sanity_check`` accepts it. Field values are placeholders --
    these tests exercise the new class APIs, not AO evaluation."""
    structure_data = Structure_data(
        pbc_flag=False,
        positions=np.zeros((1, 3)),
        atomic_numbers=(1,),
        element_symbols=("X",),
        atomic_labels=("X",),
    )
    aos = AOs_sphe_gto_data._from_full_basis(
        structure_data=structure_data,
        nucleus_index=(0,) * num_ao,
        num_ao=num_ao,
        num_ao_prim=num_ao,
        orbital_indices=tuple(range(num_ao)),
        exponents=np.ones(num_ao, dtype=np.float64),
        coefficients=np.ones(num_ao, dtype=np.float64),
        angular_momentums=(0,) * num_ao,
        magnetic_quantum_numbers=(0,) * num_ao,
    )
    coeffs = np.zeros((num_mo, num_ao), dtype=np.float64)
    return MOs_data(num_mo=num_mo, mo_coefficients=coeffs, aos_data=aos)


def _build_evaluable_mos_data(seed: int, num_mo: int = 4, num_ao: int = 4) -> MOs_data:
    """Build non-degenerate real MOs for determinant-level regression oracles."""
    if num_ao != 4 or num_mo > num_ao:
        raise ValueError("The deterministic determinant fixture requires num_ao=4 and num_mo<=4.")
    structure_data = Structure_data(
        pbc_flag=False,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.1, -0.2],
                [-0.3, 0.9, 0.2],
                [0.2, -0.4, 1.0],
            ],
            dtype=np.float64,
        ),
        atomic_numbers=(1, 1, 1, 1),
        element_symbols=("X", "X", "X", "X"),
        atomic_labels=("X0", "X1", "X2", "X3"),
    )
    aos = AOs_sphe_gto_data._from_full_basis(
        structure_data=structure_data,
        nucleus_index=tuple(range(num_ao)),
        num_ao=num_ao,
        num_ao_prim=num_ao,
        orbital_indices=tuple(range(num_ao)),
        exponents=np.array([0.7, 1.0, 1.3, 1.6], dtype=np.float64),
        coefficients=np.ones(num_ao, dtype=np.float64),
        angular_momentums=(0,) * num_ao,
        magnetic_quantum_numbers=(0,) * num_ao,
    )
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(num_mo, num_ao))
    coefficients[:, :num_mo] += 2.0 * np.eye(num_mo)
    mos = MOs_data(
        num_mo=num_mo,
        mo_coefficients=np.ascontiguousarray(coefficients),
        aos_data=aos,
    )
    mos.sanity_check()
    return mos


class TestAGPnData:
    @pytest.mark.light
    def test_AGPn_data_build_defaults_lambda_diag(self):
        """``AGPn_data.build`` initialises ``lambda_diag`` to ``[1]*N_occ + [0]*N_virt``."""
        mos = _build_minimal_mos_data(num_mo=5)
        agpn = AGPn_data.build(
            num_electron_up=2,
            num_electron_dn=2,
            mos_up=mos,
        )
        # N_occ = 2, num_mo = 5 -> [1, 1, 0, 0, 0]
        np.testing.assert_array_equal(np.asarray(agpn.lambda_diag), np.array([1.0, 1.0, 0.0, 0.0, 0.0]))
        assert agpn.num_mo == 5
        assert agpn.time_reversal is True

    @pytest.mark.light
    def test_AGPn_data_build_auto_derives_mos_dn_under_time_reversal(self):
        """When ``time_reversal=True`` (the default for closed-shell), ``mos_dn``
        is auto-derived as ``conj(mos_up)`` -- which is the identity on real
        coefficients."""
        mos = _build_minimal_mos_data(num_mo=3)
        mos = mos.replace(mo_coefficients=np.arange(12, dtype=np.float64).reshape(3, 4))
        agpn = AGPn_data.build(
            num_electron_up=2,
            num_electron_dn=2,
            mos_up=mos,
        )
        np.testing.assert_array_equal(
            np.asarray(agpn.orb_data_dn_spin.mo_coefficients),
            np.asarray(mos.mo_coefficients),
        )

    @pytest.mark.light
    def test_AGPn_data_build_open_shell_requires_explicit_mos_dn(self):
        """Open-shell (``time_reversal=False``) requires ``mos_dn`` explicitly."""
        mos = _build_minimal_mos_data(num_mo=4)
        with pytest.raises(ValueError, match="mos_dn is required"):
            AGPn_data.build(
                num_electron_up=3,
                num_electron_dn=2,
                mos_up=mos,
                time_reversal=False,
            )

    @pytest.mark.light
    def test_AGPn_data_sanity_check_rejects_complex_lambda_diag(self):
        """``lambda_diag`` is a physical invariant -- always real even on PBC."""
        mos = _build_minimal_mos_data(num_mo=2)
        bad_lambda = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)
        agpn = AGPn_data(
            num_electron_up=1,
            num_electron_dn=1,
            orb_data_up_spin=mos,
            orb_data_dn_spin=mos,
            lambda_diag=bad_lambda,
            time_reversal=True,
        )
        with pytest.raises(ValueError, match="natural-orbital occupations are real"):
            agpn.sanity_check()

    @pytest.mark.light
    def test_AGPn_data_lambda_matrix_property_materialises_diag(self):
        """The downstream-compat property returns ``diag(lambda_diag)``."""
        mos = _build_minimal_mos_data(num_mo=4)
        agpn = AGPn_data.build(
            num_electron_up=2,
            num_electron_dn=2,
            mos_up=mos,
        )
        np.testing.assert_array_equal(agpn.lambda_matrix, np.diag(np.asarray(agpn.lambda_diag)))


class TestSDData:
    @pytest.mark.light
    def test_complex_debug_determinant_preserves_imaginary_part(self):
        """The NumPy debug oracle must retain the non-Gamma determinant phase."""
        sd = read_trexio_file(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "trexio_example_files",
                "lithium_k025.h5",
            ),
            store_tuple=True,
        )[4]
        rng = np.random.default_rng(77)
        r_up = rng.normal(size=(sd.num_electron_up, 3))
        r_dn = rng.normal(size=(sd.num_electron_dn, 3))

        geminal = np.asarray(_compute_geminal_all_elements_debug(sd, r_up, r_dn))
        independent = np.linalg.det(geminal)
        assert abs(independent.imag) > ATOL_SD_DIRECT_FACTOR_ORACLE

        actual = _compute_det_geminal_all_elements_debug(sd, r_up, r_dn)
        assert np.iscomplexobj(actual)
        np.testing.assert_allclose(
            actual,
            independent,
            atol=ATOL_SD_DIRECT_FACTOR_ORACLE,
            rtol=RTOL_SD_DIRECT_FACTOR_ORACLE,
        )

    @pytest.mark.light
    @pytest.mark.parametrize(
        "trexio_file",
        ["H2_ae_ccpvdz_cart.h5", "lithium_k025.h5", "N_ae_ccpvdz_cart.h5"],
        ids=["real-closed", "complex-closed", "real-open"],
    )
    def test_SD_data_direct_determinant_matches_independent_spin_determinant_product(self, trexio_file):
        """Direct SD equals det(Phi_up^T) det(Phi_dn), independently of AGP."""
        from jqmc.molecular_orbital import compute_MOs

        sd = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )[4]
        assert isinstance(sd, SD_data)
        rng = np.random.default_rng(1201)
        r_up = rng.normal(scale=0.7, size=(sd.num_occ_up, 3))
        r_dn = rng.normal(scale=0.7, size=(sd.num_occ_dn, 3))
        phi_up = np.asarray(compute_MOs(sd.orb_data_up_spin, r_up))
        phi_dn = np.asarray(compute_MOs(sd.orb_data_dn_spin, r_dn))
        independent = np.linalg.det(phi_up.T) * np.linalg.det(phi_dn)
        actual = np.asarray(compute_det_geminal_all_elements(sd, r_up, r_dn))
        np.testing.assert_allclose(
            actual,
            independent,
            rtol=RTOL_SD_DIRECT_FACTOR_ORACLE,
            atol=ATOL_SD_DIRECT_FACTOR_ORACLE,
        )

    @pytest.mark.light
    @pytest.mark.parametrize(
        "trexio_file",
        ["H2_ae_ccpvdz_cart.h5", "lithium_k025.h5", "N_ae_ccpvdz_cart.h5"],
        ids=["real-closed", "complex-closed", "real-open"],
    )
    def test_SD_data_mo_parameter_directional_derivative_matches_independent_fd(self, trexio_file):
        """Occupied-MO O_k follows a scalar central difference in each spin block."""
        from jqmc.derivatives.assemble import grad_ln_det_wrt_mo_coeff

        sd = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )[4]
        # Test the two stored leaves independently. The production
        # time-reversal coordinate map folds them only after these raw source
        # derivatives have been formed.
        sd = sd.replace(time_reversal=False)
        rng = np.random.default_rng(1202)
        r_up = rng.normal(scale=0.6, size=(sd.num_occ_up, 3))
        r_dn = rng.normal(scale=0.6, size=(sd.num_occ_dn, 3))
        c_up = np.asarray(sd.orb_data_up_spin.mo_coefficients)
        c_dn = np.asarray(sd.orb_data_dn_spin.mo_coefficients)
        d_up = rng.normal(size=c_up.shape)
        d_dn = rng.normal(size=c_dn.shape)
        if np.iscomplexobj(c_up):
            d_up = d_up + 1.0j * rng.normal(size=c_up.shape)
        if np.iscomplexobj(c_dn):
            d_dn = d_dn + 1.0j * rng.normal(size=c_dn.shape)
        norm = np.sqrt(np.vdot(d_up, d_up).real + np.vdot(d_dn, d_dn).real)
        d_up = d_up / norm
        d_dn = d_dn / norm

        g_up, g_dn = grad_ln_det_wrt_mo_coeff(sd, r_up, r_dn)
        analytical = np.real(np.sum(np.asarray(g_up) * d_up) + np.sum(np.asarray(g_dn) * d_dn))

        def value(step):
            moved = sd.replace(
                orb_data_up_spin=sd.orb_data_up_spin.replace(mo_coefficients=c_up + step * d_up),
                orb_data_dn_spin=sd.orb_data_dn_spin.replace(mo_coefficients=c_dn + step * d_dn),
            )
            return float(compute_ln_det_geminal_all_elements(moved, r_up, r_dn))

        step = 2.0e-6
        finite_difference = (value(step) - value(-step)) / (2.0 * step)
        np.testing.assert_allclose(
            analytical,
            finite_difference,
            rtol=RTOL_FD_VS_ANALYTIC_TIGHT,
            atol=ATOL_FD_VS_ANALYTIC_TIGHT,
        )

    @pytest.mark.light
    def test_SD_data_canonical_form_keeps_only_occupied_rows_and_no_lambda(self):
        """SD owns occupied orbitals, not an AGP identity occupation matrix."""
        mos = _build_minimal_mos_data(num_mo=4)
        sd = SD_data.build(
            num_electron_up=2,
            num_electron_dn=2,
            mos_up=mos,
        )

        assert sd.num_occ_up == 2
        assert sd.num_occ_dn == 2
        assert sd.orb_data_up_spin.mo_coefficients.shape == (2, mos.aos_data.num_ao)
        assert sd.orb_data_dn_spin.mo_coefficients.shape == (2, mos.aos_data.num_ao)
        assert "lambda_matrix" not in sd.__dataclass_fields__
        assert "lambda_unpaired" not in sd.__dataclass_fields__
        assert not hasattr(sd, "lambda_matrix")
        assert not hasattr(sd, "_lambda_matrix_jnp")

    @pytest.mark.light
    def test_SD_data_time_reversal_rejects_inconsistent_down_coefficients(self):
        """A tied SD cannot carry an independently different down-spin leaf."""
        mos = _build_minimal_mos_data(num_mo=3)
        inconsistent_dn = mos.replace(mo_coefficients=np.ascontiguousarray(np.asarray(mos.mo_coefficients) + 0.125))

        with pytest.raises(ValueError, match="time_reversal=True requires"):
            SD_data.build(
                num_electron_up=2,
                num_electron_dn=2,
                mos_up=mos,
                mos_dn=inconsistent_dn,
                time_reversal=True,
            )

    @pytest.mark.light
    def test_SD_data_time_reversal_false_accepts_independent_down_coefficients(self):
        """An unrestricted SD keeps distinct occupied up/down variables."""
        mos_up = _build_minimal_mos_data(num_mo=3)
        coefficients_dn = np.ascontiguousarray(np.asarray(mos_up.mo_coefficients) * 0.75 + 0.125)
        mos_dn = mos_up.replace(mo_coefficients=coefficients_dn)

        sd = SD_data.build(
            num_electron_up=2,
            num_electron_dn=2,
            mos_up=mos_up,
            mos_dn=mos_dn,
            time_reversal=False,
        )

        assert sd.time_reversal is False
        np.testing.assert_array_equal(
            np.asarray(sd.orb_data_up_spin.mo_coefficients),
            np.asarray(mos_up.mo_coefficients)[:2],
        )
        np.testing.assert_array_equal(
            np.asarray(sd.orb_data_dn_spin.mo_coefficients),
            coefficients_dn[:2],
        )
        assert not np.allclose(
            np.asarray(sd.orb_data_dn_spin.mo_coefficients),
            np.conj(np.asarray(sd.orb_data_up_spin.mo_coefficients)),
            rtol=RTOL_NONVACUITY_DISTINCTION,
            atol=ATOL_NONVACUITY_DISTINCTION,
        )

        grad_sd = sd.replace(
            orb_data_up_spin=sd.orb_data_up_spin.replace(
                mo_coefficients=np.full_like(
                    sd.orb_data_up_spin.mo_coefficients,
                    2.0,
                )
            ),
            orb_data_dn_spin=sd.orb_data_dn_spin.replace(
                mo_coefficients=np.full_like(
                    sd.orb_data_dn_spin.mo_coefficients,
                    3.0,
                )
            ),
        )
        collected = sd.collect_param_grads(grad_sd)
        np.testing.assert_array_equal(
            np.asarray(collected["mo_coefficients_up"]),
            np.full_like(sd.orb_data_up_spin.mo_coefficients, 2.0),
        )
        np.testing.assert_array_equal(
            np.asarray(collected["mo_coefficients_dn"]),
            np.full_like(sd.orb_data_dn_spin.mo_coefficients, 3.0),
        )
        assert "lambda_matrix" not in collected

    @pytest.mark.light
    def test_SD_data_rejects_obsolete_lambda_symmetrization(self):
        """Canonical SD must fail loudly if reference common code asks for lambda."""
        sd = SD_data.build(
            num_electron_up=2,
            num_electron_dn=2,
            mos_up=_build_minimal_mos_data(num_mo=3),
        )
        with pytest.raises(TypeError, match="has no lambda matrix"):
            sd.symmetrize_lambda(np.eye(2))

    @pytest.mark.light
    def test_SD_data_open_shell_time_reversal_gradient_pads_down_rows(self):
        """ROHF ties only the paired down rows into the full up-spin block."""
        mos = read_trexio_file(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "trexio_example_files",
                "N_ae_ccpvdz_cart.h5",
            ),
            store_tuple=True,
        )[2]
        sd = SD_data.build(
            num_electron_up=3,
            num_electron_dn=2,
            mos_up=mos,
            time_reversal=True,
        )
        grad_up = np.full((3, mos.aos_data.num_ao), 2.0)
        grad_dn = np.full((2, mos.aos_data.num_ao), 3.0)
        grad_sd = sd.replace(
            orb_data_up_spin=sd.orb_data_up_spin.replace(
                mo_coefficients=grad_up,
            ),
            orb_data_dn_spin=sd.orb_data_dn_spin.replace(
                mo_coefficients=grad_dn,
            ),
        )

        collected = np.asarray(sd.collect_param_grads(grad_sd)["mo_coefficients_up"])
        expected = grad_up.copy()
        expected[: sd.num_occ_dn] += grad_dn
        np.testing.assert_array_equal(collected, expected)

        from jqmc.derivatives.bridge import _collect_c_param_derivative_sources
        from jqmc.wavefunction import Wavefunction_data

        num_walkers = 2
        batched_grad_sd = grad_sd.replace(
            orb_data_up_spin=grad_sd.orb_data_up_spin.replace(
                mo_coefficients=np.broadcast_to(
                    grad_up,
                    (num_walkers, *grad_up.shape),
                )
            ),
            orb_data_dn_spin=grad_sd.orb_data_dn_spin.replace(
                mo_coefficients=np.broadcast_to(
                    grad_dn,
                    (num_walkers, *grad_dn.shape),
                )
            ),
        )
        wf = Wavefunction_data(geminal_data=sd, jastrow_data=None)
        grad_wf = Wavefunction_data(
            geminal_data=batched_grad_sd,
            jastrow_data=None,
        )
        blocks = wf.get_variational_blocks(
            state_token=("test", 0),
            opt_J1_param=False,
            opt_J2_param=False,
            opt_J3_param=False,
            opt_geminal=True,
        )
        sources = _collect_c_param_derivative_sources(
            wf,
            grad_wf,
            blocks,
            num_walkers,
            {"opt_geminal": True},
        )
        assert len(sources) == 1
        np.testing.assert_array_equal(
            np.asarray(sources[0].direct),
            np.broadcast_to(expected, (num_walkers, *expected.shape)),
        )

    @pytest.mark.light
    def test_SD_data_cartesian_spherical_conversion_preserves_direct_sd(self):
        """AO representation conversion must not recreate a lambda-backed SD."""
        atol_c, rtol_c = get_tolerance_min(
            ("ao_eval", "mo_eval", "det_eval", "ao_grad_lap", "mo_grad", "mo_lap", "det_grad_lap"),
            "strict",
        )
        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            "H2_ae_ccpvdz_sphe.h5",
        )
        sd_spherical = read_trexio_file(
            trexio_file=fixture,
            store_tuple=True,
        )[4]
        rng = np.random.default_rng(1203)
        r_up = rng.normal(size=(sd_spherical.num_occ_up, 3))
        r_dn = rng.normal(size=(sd_spherical.num_occ_dn, 3))
        reference = compute_det_geminal_all_elements(sd_spherical, r_up, r_dn)

        sd_cartesian = sd_spherical.to_cartesian()
        sd_roundtrip = sd_cartesian.to_spherical()

        for converted in (sd_cartesian, sd_roundtrip):
            assert isinstance(converted, SD_data)
            assert not hasattr(converted, "lambda_matrix")
            assert converted.orb_data_up_spin.num_mo == converted.num_occ_up
            assert converted.orb_data_dn_spin.num_mo == converted.num_occ_dn
            converted.sanity_check()
            np.testing.assert_allclose(
                np.asarray(compute_det_geminal_all_elements(converted, r_up, r_dn)),
                np.asarray(reference),
                atol=atol_c,
                rtol=rtol_c,
            )

    @pytest.mark.light
    def test_SD_data_complex_bloch_conversion_preserves_direct_sd(self):
        """SD-level conversion retains both Bloch spin leaves and determinant."""
        atol_c, rtol_c = get_tolerance_min(
            ("ao_eval", "mo_eval", "det_eval", "ao_grad_lap", "mo_grad", "mo_lap", "det_grad_lap"),
            "strict",
        )
        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            "lithium_k025.h5",
        )
        sd_spherical = read_trexio_file(
            trexio_file=fixture,
            store_tuple=True,
        )[4]
        assert isinstance(sd_spherical, SD_data)
        assert isinstance(sd_spherical.orb_data_up_spin.aos_data, AOs_sphe_pgto_complex_data)
        k_up = sd_spherical.orb_data_up_spin.aos_data.k_point
        k_dn = sd_spherical.orb_data_dn_spin.aos_data.k_point
        assert k_up is not None
        assert k_dn is not None
        assert k_dn == tuple(-component for component in k_up)

        cell = np.asarray(sd_spherical.orb_data_up_spin.aos_data.structure_data.cell)
        r_up = np.array([[1.07, -0.19, 0.33]], dtype=np.float64) @ cell
        r_dn = np.array([[-0.22, 0.58, 1.14]], dtype=np.float64) @ cell
        reference = compute_det_geminal_all_elements(sd_spherical, r_up, r_dn)

        sd_cartesian = sd_spherical.to_cartesian()
        sd_roundtrip = sd_cartesian.to_spherical()
        assert isinstance(sd_cartesian.orb_data_up_spin.aos_data, AOs_cart_pgto_complex_data)
        assert isinstance(sd_cartesian.orb_data_dn_spin.aos_data, AOs_cart_pgto_complex_data)
        assert isinstance(sd_roundtrip.orb_data_up_spin.aos_data, AOs_sphe_pgto_complex_data)
        assert isinstance(sd_roundtrip.orb_data_dn_spin.aos_data, AOs_sphe_pgto_complex_data)

        for converted in (sd_cartesian, sd_roundtrip):
            assert converted.orb_data_up_spin.aos_data.k_point == k_up
            assert converted.orb_data_dn_spin.aos_data.k_point == k_dn
            assert converted.orb_data_up_spin.num_mo == converted.num_occ_up
            assert converted.orb_data_dn_spin.num_mo == converted.num_occ_dn
            assert not hasattr(converted, "lambda_matrix")
            converted.sanity_check()
            np.testing.assert_allclose(
                np.asarray(compute_det_geminal_all_elements(converted, r_up, r_dn)),
                np.asarray(reference),
                atol=atol_c,
                rtol=rtol_c,
            )

    @pytest.mark.light
    def test_SD_data_complex_bloch_conversion_time_reversal_false(self):
        """Independent spin leaves and occupied rows survive basis conversion."""
        atol_c, rtol_c = get_tolerance_min(
            ("ao_eval", "mo_eval", "det_eval", "ao_grad_lap", "mo_grad", "mo_lap", "det_grad_lap"),
            "strict",
        )
        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            "lithium_k025.h5",
        )
        source_mos = read_trexio_file(trexio_file=fixture, store_tuple=True)[2]
        aos_up = source_mos.aos_data
        aos_dn = aos_up.with_negated_k()
        base = np.asarray(source_mos.mo_coefficients[0], dtype=np.complex128)
        mos_up = MOs_data.build(
            aos_data=aos_up,
            mo_coefficients=np.stack(
                (
                    base * (1.0 + 0.2j),
                    np.roll(base, 2) * (-0.3 + 0.7j),
                )
            ),
        )
        mos_dn = MOs_data.build(
            aos_data=aos_dn,
            mo_coefficients=np.stack((np.roll(base.conj(), 1) * (0.8 - 0.4j),)),
        )
        sd_spherical = SD_data.build(
            num_electron_up=2,
            num_electron_dn=1,
            mos_up=mos_up,
            mos_dn=mos_dn,
            time_reversal=False,
        )
        assert sd_spherical.time_reversal is False

        cell = np.asarray(aos_up.structure_data.cell)
        r_up = np.array([[0.17, 0.31, 0.43], [1.11, -0.23, 0.37]], dtype=np.float64) @ cell
        r_dn = np.array([[-0.29, 0.61, 1.08]], dtype=np.float64) @ cell
        reference_det = np.asarray(compute_det_geminal_all_elements(sd_spherical, r_up, r_dn))
        reference_derivatives = tuple(np.asarray(x) for x in compute_grads_and_laplacian_ln_Det(sd_spherical, r_up, r_dn))

        sd_cartesian = sd_spherical.to_cartesian()
        sd_roundtrip = sd_cartesian.to_spherical()
        for converted in (sd_cartesian, sd_roundtrip):
            assert converted.time_reversal is False
            assert converted.orb_data_up_spin.num_mo == 2
            assert converted.orb_data_dn_spin.num_mo == 1
            assert converted.orb_data_up_spin.aos_data.k_point == aos_up.k_point
            assert converted.orb_data_dn_spin.aos_data.k_point == aos_dn.k_point
            converted.sanity_check()
            np.testing.assert_allclose(
                np.asarray(compute_det_geminal_all_elements(converted, r_up, r_dn)),
                reference_det,
                atol=atol_c,
                rtol=rtol_c,
            )
            actual_derivatives = compute_grads_and_laplacian_ln_Det(converted, r_up, r_dn)
            for actual, expected in zip(actual_derivatives, reference_derivatives, strict=True):
                np.testing.assert_allclose(
                    np.asarray(actual),
                    expected,
                    atol=atol_c,
                    rtol=rtol_c,
                )

    @pytest.mark.light
    def test_SD_data_direct_closed_shell_matches_debug_identity_geminal(self):
        """Direct real SD equals the debug identity-geminal diag(1,1,0,0) form."""
        from jqmc.determinant import AGP_data, compute_geminal_all_elements

        mos = _build_evaluable_mos_data(seed=920)
        sd = SD_data.build(num_electron_up=2, num_electron_dn=2, mos_up=mos)
        reference = AGP_data.build(
            num_electron_up=2,
            num_electron_dn=2,
            orb_data_up_spin=mos,
            orb_data_dn_spin=mos.replace(mo_coefficients=np.ascontiguousarray(np.conj(np.asarray(mos.mo_coefficients)))),
            lambda_matrix=np.diag([1.0, 1.0, 0.0, 0.0]),
            time_reversal=True,
        )
        rng = np.random.default_rng(921)
        r_up = rng.normal(size=(2, 3))
        r_dn = rng.normal(size=(2, 3))

        G_sd = compute_geminal_all_elements(sd, r_up, r_dn)
        G_reference = compute_geminal_all_elements(reference, r_up, r_dn)
        assert np.linalg.norm(np.asarray(G_sd)) > ATOL_SD_DIRECT_FACTOR_ORACLE
        assert abs(np.linalg.det(np.asarray(G_sd))) > ATOL_SD_DIRECT_FACTOR_ORACLE
        np.testing.assert_allclose(
            np.asarray(G_sd),
            np.asarray(G_reference),
            atol=ATOL_SD_DIRECT_FACTOR_ORACLE,
            rtol=RTOL_SD_DIRECT_FACTOR_ORACLE,
        )
        np.testing.assert_allclose(
            np.asarray(compute_det_geminal_all_elements(sd, r_up, r_dn)),
            np.asarray(compute_det_geminal_all_elements(reference, r_up, r_dn)),
            atol=ATOL_SD_DIRECT_FACTOR_ORACLE,
            rtol=RTOL_SD_DIRECT_FACTOR_ORACLE,
        )

        direct = compute_grads_and_laplacian_ln_Det(sd, r_up, r_dn)
        identity_selector = compute_grads_and_laplacian_ln_Det(reference, r_up, r_dn)
        for comparison_index, (actual, expected) in enumerate(zip(direct, identity_selector)):
            comparison_iteration = str(comparison_index)  # noqa: F841 - captured by the GDD iteration receipt
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                atol=ATOL_SD_DIRECT_FACTOR_ORACLE,
                rtol=RTOL_SD_DIRECT_FACTOR_ORACLE,
            )

        G_inv = jnp.linalg.inv(G_sd)
        direct_fast = compute_grads_and_laplacian_ln_Det_fast(sd, r_up, r_dn, G_inv)
        identity_fast = compute_grads_and_laplacian_ln_Det_fast(reference, r_up, r_dn, G_inv)
        for comparison_index, (actual, expected) in enumerate(zip(direct_fast, identity_fast)):
            comparison_iteration = str(comparison_index)  # noqa: F841 - captured by the GDD iteration receipt
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                atol=ATOL_SD_DIRECT_FACTOR_ORACLE,
                rtol=RTOL_SD_DIRECT_FACTOR_ORACLE,
            )

    @pytest.mark.light
    def test_SD_data_direct_complex_matches_identity_geminal(self):
        """The non-Gamma direct SD path equals a complex identity geminal."""
        _, _, _, _, sd, _ = read_trexio_file(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "trexio_example_files",
                "lithium_k025.h5",
            ),
            store_tuple=True,
        )
        assert isinstance(sd, SD_data)
        assert np.iscomplexobj(np.asarray(sd.orb_data_up_spin.mo_coefficients))
        n_occ = sd.num_occ_up
        reference = AGP_data.build(
            num_electron_up=n_occ,
            num_electron_dn=n_occ,
            orb_data_up_spin=sd.orb_data_up_spin,
            orb_data_dn_spin=sd.orb_data_dn_spin,
            lambda_matrix=np.eye(n_occ, dtype=np.complex128),
            time_reversal=True,
        )
        rng = np.random.default_rng(922)
        r_up = rng.normal(size=(n_occ, 3))
        r_dn = rng.normal(size=(n_occ, 3))

        G_sd = compute_geminal_all_elements(sd, r_up, r_dn)
        G_reference = compute_geminal_all_elements(reference, r_up, r_dn)
        assert np.linalg.norm(np.asarray(G_sd)) > ATOL_SD_DIRECT_FACTOR_ORACLE
        assert abs(np.linalg.det(np.asarray(G_sd))) > ATOL_SD_DIRECT_FACTOR_ORACLE
        np.testing.assert_allclose(
            np.asarray(G_sd),
            np.asarray(G_reference),
            atol=ATOL_SD_DIRECT_FACTOR_ORACLE,
            rtol=RTOL_SD_DIRECT_FACTOR_ORACLE,
        )
        np.testing.assert_allclose(
            np.asarray(compute_det_geminal_all_elements(sd, r_up, r_dn)),
            np.asarray(compute_det_geminal_all_elements(reference, r_up, r_dn)),
            atol=ATOL_SD_DIRECT_FACTOR_ORACLE,
            rtol=RTOL_SD_DIRECT_FACTOR_ORACLE,
        )
        direct = compute_grads_and_laplacian_ln_Det(sd, r_up, r_dn)
        identity_selector = compute_grads_and_laplacian_ln_Det(reference, r_up, r_dn)
        for comparison_index, (actual, expected) in enumerate(zip(direct, identity_selector)):
            comparison_iteration = str(comparison_index)  # noqa: F841 - captured by the GDD iteration receipt
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                atol=ATOL_SD_DIRECT_FACTOR_ORACLE,
                rtol=RTOL_SD_DIRECT_FACTOR_ORACLE,
            )

        G_inv = jnp.linalg.inv(G_sd)
        direct_fast = compute_grads_and_laplacian_ln_Det_fast(sd, r_up, r_dn, G_inv)
        identity_fast = compute_grads_and_laplacian_ln_Det_fast(reference, r_up, r_dn, G_inv)
        for comparison_index, (actual, expected) in enumerate(zip(direct_fast, identity_fast)):
            comparison_iteration = str(comparison_index)
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                atol=ATOL_SD_DIRECT_FACTOR_ORACLE,
                rtol=RTOL_SD_DIRECT_FACTOR_ORACLE,
            )

    @pytest.mark.light
    def test_SD_data_direct_open_shell_matches_debug_selector(self):
        """Excess occupied up rows replace the old lambda_unpaired selector."""
        mos = _build_evaluable_mos_data(seed=924)
        sd = SD_data.build(
            num_electron_up=3,
            num_electron_dn=2,
            mos_up=mos,
            mos_dn=mos,
            time_reversal=False,
        )
        reference_lambda = np.zeros((4, 5))
        reference_lambda[0, 0] = 1.0
        reference_lambda[1, 1] = 1.0
        reference_lambda[2, 4] = 1.0
        reference = AGP_data.build(
            num_electron_up=3,
            num_electron_dn=2,
            orb_data_up_spin=mos,
            orb_data_dn_spin=mos,
            lambda_matrix=reference_lambda,
            time_reversal=False,
        )
        rng = np.random.default_rng(923)
        r_up = rng.normal(size=(3, 3))
        r_dn = rng.normal(size=(2, 3))

        G_sd = compute_geminal_all_elements(sd, r_up, r_dn)
        G_reference = compute_geminal_all_elements(reference, r_up, r_dn)
        assert np.linalg.norm(np.asarray(G_sd)) > ATOL_SD_DIRECT_FACTOR_ORACLE
        assert abs(np.linalg.det(np.asarray(G_sd))) > ATOL_SD_DIRECT_FACTOR_ORACLE
        np.testing.assert_allclose(
            np.asarray(G_sd),
            np.asarray(G_reference),
            atol=ATOL_SD_DIRECT_FACTOR_ORACLE,
            rtol=RTOL_SD_DIRECT_FACTOR_ORACLE,
        )
        direct = compute_grads_and_laplacian_ln_Det(sd, r_up, r_dn)
        identity_selector = compute_grads_and_laplacian_ln_Det(reference, r_up, r_dn)
        for comparison_index, (actual, expected) in enumerate(zip(direct, identity_selector)):
            comparison_iteration = str(comparison_index)
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                atol=ATOL_SD_DIRECT_FACTOR_ORACLE,
                rtol=RTOL_SD_DIRECT_FACTOR_ORACLE,
            )
        G_inv = jnp.linalg.inv(G_sd)
        direct_fast = compute_grads_and_laplacian_ln_Det_fast(sd, r_up, r_dn, G_inv)
        identity_fast = compute_grads_and_laplacian_ln_Det_fast(reference, r_up, r_dn, G_inv)
        for comparison_index, (actual, expected) in enumerate(zip(direct_fast, identity_fast)):
            comparison_iteration = str(comparison_index)
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                atol=ATOL_SD_DIRECT_FACTOR_ORACLE,
                rtol=RTOL_SD_DIRECT_FACTOR_ORACLE,
            )

    @pytest.mark.light
    def test_SD_data_build_defaults_and_sanity(self):
        """``SD_data.build`` for closed-shell."""
        mos = _build_minimal_mos_data(num_mo=3)
        sd = SD_data.build(
            num_electron_up=2,
            num_electron_dn=2,
            mos_up=mos,
        )
        assert sd.time_reversal is True
        assert sd.num_mo == 2
        assert sd.orb_data_up_spin.mo_coefficients.shape[0] == 2
        assert sd.orb_data_dn_spin.mo_coefficients.shape[0] == 2
        sd.sanity_check()

    @pytest.mark.light
    def test_SD_data_build_fix_geminal_rank_warns_and_ignored(self, caplog):
        """``SD_data`` ignores ``fix_geminal_rank`` and logs a warning."""
        import logging

        mos = _build_minimal_mos_data(num_mo=3)
        with caplog.at_level(logging.WARNING):
            sd = SD_data.build(
                num_electron_up=2,
                num_electron_dn=2,
                mos_up=mos,
                fix_geminal_rank=7,
            )
        assert any(rec.levelno >= logging.WARNING for rec in caplog.records)
        assert sd.num_mo == 2  # construction succeeds with occupied rows only

    @pytest.mark.light
    def test_SD_data_to_AGPn_promotes_without_inventing_virtual_rows(self):
        """``SD_data.to_AGPn`` preserves the occupied rank exactly."""
        mos = _build_minimal_mos_data(num_mo=4)
        sd = SD_data.build(
            num_electron_up=2,
            num_electron_dn=2,
            mos_up=mos,
        )
        agpn = sd.to_AGPn()
        assert isinstance(agpn, AGPn_data)
        np.testing.assert_array_equal(
            np.asarray(agpn.lambda_diag),
            np.array([1.0, 1.0]),
        )
        assert agpn.num_mo == 2


class TestGeminalConversions:
    @pytest.mark.light
    def test_AGPn_data_to_AGP_materialises_diag_lambda(self):
        """``AGPn_data.to_AGP`` materialises a dense AGP with ``diag(lambda_diag)``."""
        mos = _build_minimal_mos_data(num_mo=3)
        agpn = AGPn_data.build(
            num_electron_up=2,
            num_electron_dn=2,
            mos_up=mos,
            lambda_diag=np.array([1.0, 0.7, 0.0], dtype=np.float64),
        )
        agp = agpn.to_AGP()
        assert isinstance(agp, AGP_data)
        np.testing.assert_array_equal(agp.lambda_matrix, np.diag([1.0, 0.7, 0.0]))
        assert agp.time_reversal is True

    @pytest.mark.light
    def test_to_AGP_free_function_dispatches_on_subtype(self):
        """``to_AGP`` free function delegates to per-class methods."""
        mos = _build_minimal_mos_data(num_mo=2)
        sd = SD_data.build(num_electron_up=1, num_electron_dn=1, mos_up=mos)
        agpn = AGPn_data.build(num_electron_up=1, num_electron_dn=1, mos_up=mos)

        sd_as_agp = to_AGP(sd)
        agpn_as_agp = to_AGP(agpn)
        assert isinstance(sd_as_agp, AGP_data)
        assert isinstance(agpn_as_agp, AGP_data)
        # Idempotent on AGP_data
        assert to_AGP(sd_as_agp) is sd_as_agp

    @pytest.mark.light
    def test_to_AGPn_truncate_rank(self):
        """``to_AGPn(geminal, n=k)`` truncates ``lambda_diag`` to top-k."""
        mos = _build_minimal_mos_data(num_mo=5)
        agpn = AGPn_data.build(
            num_electron_up=2,
            num_electron_dn=2,
            mos_up=mos,
            mos_dn=mos,
            lambda_diag=np.array([1.0, 1.0, 0.5, 0.1, 0.0], dtype=np.float64),
            time_reversal=False,
        )
        truncated = to_AGPn(agpn, n=3)
        np.testing.assert_array_equal(np.asarray(truncated.lambda_diag), np.array([1.0, 1.0, 0.5]))
        assert truncated.num_mo == 3

    @pytest.mark.light
    def test_to_AGPn_n_exceeds_num_mo_is_rejected(self):
        """``to_AGPn(geminal, n > num_mo)`` rejects an inexact rank cap."""
        mos = _build_minimal_mos_data(num_mo=2)
        agpn = AGPn_data.build(
            num_electron_up=1,
            num_electron_dn=1,
            mos_up=mos,
        )
        with pytest.raises(ValueError, match="exceeds the available rank"):
            to_AGPn(agpn, n=99)

    @pytest.mark.light
    def test_to_AGPn_from_AGP_closed_shell_trexio(self):
        """``to_AGPn(AGP_data, n)`` on a closed-shell TREXIO loader output:
        rank truncation + Convention B + S-orthonormality preserved.

        Covers the geminal class split path: AGP_data ->
        AGPn_data via :func:`Geminal_data.convert_from_AOs_to_MOs` when the
        geminal is AO-rep, or via in-place MO-basis SVD when it is already
        MO-rep.

        Note: the closed-shell TREXIO
        loader returns ``SD_data`` directly. To exercise the AGP -> AGPn
        truncation path, we promote the loaded geminal back to ``AGP_data``
        via the conversion API and then test the truncation.
        """
        from jqmc.atomic_orbital import compute_overlap_matrix
        from jqmc.determinant import SD_data
        from jqmc.molecular_orbital import MOs_data
        from jqmc.trexio_wrapper import read_trexio_file

        (_, _, _, _, geminal_data, _) = read_trexio_file(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "water_ccecp_ccpvqz.h5"
            ),
            store_tuple=False,
        )
        assert isinstance(geminal_data, SD_data)
        assert int(geminal_data.num_electron_up) == int(geminal_data.num_electron_dn)
        # Build an explicit MO-basis AGP only inside this
        # conversion test.  Canonical SD_data itself deliberately owns no
        # virtual rows or lambda selector.

        # Pad the occupied SD orbitals with 3 random S-orthonormal virtuals
        # (zero lambda
        # block) so the truncation path actually exercises ``n < num_mo``.
        n_occ = int(geminal_data.num_electron_dn)
        n_extra = 3
        C_up = np.asarray(geminal_data.orb_data_up_spin.mo_coefficients)
        n_ao = C_up.shape[1]
        aos = geminal_data.orb_data_up_spin.aos_data
        S = np.asarray(compute_overlap_matrix(aos))
        rng = np.random.default_rng(0)
        V = rng.standard_normal((n_extra, n_ao))
        for i in range(n_extra):
            for j in range(n_occ):
                V[i] -= (C_up[j].conj() @ S @ V[i]) * C_up[j]
        M = V.conj() @ S @ V.T
        L = np.linalg.cholesky(0.5 * (M + M.conj().T))
        V = np.linalg.solve(L, V)

        C_up_padded = np.vstack([C_up, V])
        C_dn_padded = np.conj(C_up_padded)
        n_new = n_occ + n_extra
        lam_padded = np.zeros((n_new, n_new))
        lam_padded[:n_occ, :n_occ] = np.eye(n_occ)

        mos_up = MOs_data(num_mo=n_new, aos_data=aos, mo_coefficients=C_up_padded)
        mos_dn = MOs_data(num_mo=n_new, aos_data=geminal_data.orb_data_dn_spin.aos_data, mo_coefficients=C_dn_padded)
        agp_padded = AGP_data(
            num_electron_up=int(geminal_data.num_electron_up),
            num_electron_dn=int(geminal_data.num_electron_dn),
            orb_data_up_spin=mos_up,
            orb_data_dn_spin=mos_dn,
            lambda_matrix=lam_padded,
            time_reversal=geminal_data.time_reversal,
        )

        # Truncate to N_occ -- expect single-Slater shape.
        agpn = to_AGPn(agp_padded, n=n_occ)
        assert isinstance(agpn, AGPn_data)
        assert agpn.num_mo == n_occ
        np.testing.assert_array_equal(np.asarray(agpn.lambda_diag), np.ones(n_occ, dtype=np.float64))

        # Convention B + S-orthonormality.  The achievable agreement here is
        # bounded by the weakest zone on the build path: AO primitive storage
        # (``ao_eval`` -- fp32 under mixed precision) for ``S`` and the SVD
        # rotation (``det_eval`` -- always fp64) for the rotated MOs.  Use
        # ``get_tolerance_min`` so the same atol is appropriate under both
        # ``--precision-mode=full`` (fp64-fp64 -> ~1e-8) and
        # ``--precision-mode=mixed`` (fp32-fp64 -> ~1e-5).
        C_new_up = np.asarray(agpn.orb_data_up_spin.mo_coefficients)
        C_new_dn = np.asarray(agpn.orb_data_dn_spin.mo_coefficients)
        np.testing.assert_allclose(C_new_dn, np.conj(C_new_up), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
        M_new = C_new_up.conj() @ S @ C_new_up.T
        atol_s, rtol_s = get_tolerance_min(("ao_eval", "det_eval"), "strict")
        np.testing.assert_allclose(M_new, np.eye(n_occ), atol=atol_s, rtol=rtol_s)

        # Selecting one member of a degenerate zero-occupation subspace has
        # no unique row order, so the exact-rank API rejects it.
        with pytest.raises(ValueError, match="ambiguous truncation"):
            to_AGPn(agp_padded, n=n_occ + 1)

    @pytest.mark.light
    def test_to_AGPn_from_open_shell_truncation_raises(self):
        """Open-shell ``SD -> AGPn`` needs explicit common-rank orbitals.

        Canonical SD has only occupied spin-specific rows and no unpaired
        lambda selector from which an AGPn common-rank representation could
        be reconstructed implicitly.
        """
        from jqmc.trexio_wrapper import read_trexio_file

        (_, _, _, _, geminal_data, _) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "N_ae_ccpvdz_cart.h5"),
            store_tuple=False,
        )
        assert int(geminal_data.num_electron_up) != int(geminal_data.num_electron_dn)
        with pytest.raises(ValueError, match="explicit common-rank natural orbitals"):
            to_AGPn(geminal_data, n=int(geminal_data.num_electron_dn))

    @pytest.mark.light
    def test_to_SD_rejects_non_SD_AGPn(self):
        """``to_SD(AGPn)`` requires ``lambda_diag == [1]*N_occ + [0]*N_virt``."""
        mos = _build_minimal_mos_data(num_mo=3)
        agpn = AGPn_data.build(
            num_electron_up=2,
            num_electron_dn=2,
            mos_up=mos,
            lambda_diag=np.array([1.0, 0.7, 0.0], dtype=np.float64),  # not pure SD
        )
        with pytest.raises(ValueError, match="non-trivial pair correlation"):
            to_SD(agpn)

    @pytest.mark.light
    def test_to_SD_accepts_pure_SD_AGPn(self):
        """``to_SD(AGPn)`` returns ``SD_data`` for a pure single-Slater AGPn."""
        mos = _build_minimal_mos_data(num_mo=3)
        agpn = AGPn_data.build(
            num_electron_up=2,
            num_electron_dn=2,
            mos_up=mos,
            lambda_diag=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        )
        sd = to_SD(agpn)
        assert isinstance(sd, SD_data)
        assert sd.num_electron_up == 2 and sd.num_electron_dn == 2


# ---------------------------------------------------------------------------
# SD / AGPn direct-MO opt path: duck-type interface + apply_block_update
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PBC paths: real (lithium.h5) and complex Bloch (lithium_k025.h5)
# ---------------------------------------------------------------------------


class TestAGPnDataIntegration:
    @pytest.mark.light
    def test_smetric_polar_retraction_complex_overlap_invariant(self, monkeypatch):
        """The complex row-oriented retraction must satisfy ``C* S C^T=I``.

        This synthetic case deliberately uses a complex Hermitian overlap;
        periodic fixtures whose overlap happens to be real cannot distinguish
        ``M^{-1/2}`` from its required complex conjugate.
        """
        import jqmc.determinant as determinant_module

        rng = np.random.default_rng(20260725)
        A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        S = A.conj().T @ A + 0.5 * np.eye(4)
        C = rng.standard_normal((2, 4)) + 1j * rng.standard_normal((2, 4))
        monkeypatch.setattr(determinant_module, "compute_overlap_matrix", lambda _aos_data: S)

        C_new = determinant_module._smetric_polar_retract(C, object())
        atol, rtol = get_tolerance("det_eval", "strict")

        np.testing.assert_allclose(
            C_new.conj() @ S @ C_new.T,
            np.eye(C.shape[0]),
            atol=atol,
            rtol=rtol,
        )

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvtz_cart.h5"])
    def test_AGPn_data_apply_block_update_polar_retracts_to_Stiefel(self, trexio_file):
        """``AGPn_data.apply_block_update("mo_coefficients_up")`` returns a
        geminal whose MO coefficients satisfy ``C^H S C = I`` (S-orthonormal)
        even when the input block carries a non-orthonormal random delta."""
        (_, _, _, _, geminal_mo_data, _) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        n_up = int(geminal_mo_data.num_electron_up)
        mos_up = geminal_mo_data.orb_data_up_spin
        agpn = AGPn_data.build(
            num_electron_up=n_up,
            num_electron_dn=int(geminal_mo_data.num_electron_dn),
            mos_up=mos_up,
            # default lambda_diag = [1]*N_occ + [0]*N_virt
        )

        rng = np.random.default_rng(0)
        C0 = np.asarray(mos_up.mo_coefficients)
        # Apply a small non-orthonormal perturbation -- this is what the SR / LM
        # driver would produce after ``block.apply_update``.
        new_C = C0 + 1e-3 * rng.standard_normal(C0.shape)
        block = make_test_block(
            name="mo_coefficients_up",
            values=new_C,
            field="real",
            derivative_policy="direct_real",
            sync_scope="sampler",
        )

        new_agpn = agpn.apply_block_update(block)
        C_new = np.asarray(new_agpn.orb_data_up_spin.mo_coefficients)
        S = np.asarray(compute_overlap_matrix(mos_up.aos_data))

        M = C_new.conj() @ S @ C_new.T
        atol, rtol = get_tolerance("det_eval", "strict")
        np.testing.assert_allclose(M, np.eye(C_new.shape[0]), atol=atol, rtol=rtol)
        # Convention B auto-derive: C_dn = conj(C_up) when time_reversal=True
        C_dn = np.asarray(new_agpn.orb_data_dn_spin.mo_coefficients)
        np.testing.assert_allclose(C_dn, np.conj(C_new), atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvtz_cart.h5"])
    def test_SD_data_apply_block_update_polar_retracts_to_Stiefel(self, trexio_file):
        """Mirror of the AGPn test for ``SD_data``."""
        (_, _, _, _, geminal_mo_data, _) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        sd = SD_data.build(
            num_electron_up=int(geminal_mo_data.num_electron_up),
            num_electron_dn=int(geminal_mo_data.num_electron_dn),
            mos_up=geminal_mo_data.orb_data_up_spin,
        )

        rng = np.random.default_rng(1)
        C0 = np.asarray(sd.orb_data_up_spin.mo_coefficients)
        new_C = C0 + 1e-3 * rng.standard_normal(C0.shape)
        block = make_test_block(
            name="mo_coefficients_up",
            values=new_C,
            field="real",
            derivative_policy="direct_real",
            sync_scope="sampler",
        )

        new_sd = sd.apply_block_update(block)
        C_new = np.asarray(new_sd.orb_data_up_spin.mo_coefficients)
        S = np.asarray(compute_overlap_matrix(sd.orb_data_up_spin.aos_data))

        M = C_new.conj() @ S @ C_new.T
        atol, rtol = get_tolerance("det_eval", "strict")
        np.testing.assert_allclose(M, np.eye(C_new.shape[0]), atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("case", ["rohf_up", "uhf_dn"])
    def test_SD_data_open_shell_apply_block_update_matches_numpy_retraction(self, case):
        """Open-shell SD block updates against a numpy Loewdin retraction.

        ``rohf_up``: N atom, 3 up / 2 dn, ``time_reversal=True``.  After an
        ``mo_coefficients_up`` step the dn rows must be the first ``n_dn``
        retracted up rows (``n_dn < n_up``, so a wrong row slice is visible).
        ``uhf_dn``: N atom UHF SD (``time_reversal=False``).  An
        ``mo_coefficients_dn`` step must retract only the dn rows.

        Reference: ``M = C S C^T``, ``C_ref = M^{-1/2} C`` by ``numpy.linalg.eigh``
        and the plain numpy Slater product
        ``slogdet(Phi_up C_up^T) + slogdet(Phi_dn C_dn^T)`` from AO values.
        """
        trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "N_ae_ccpvdz_cart.h5")
        loaded = read_trexio_file(trexio_file=trexio_path, store_tuple=True)
        if case == "rohf_up":
            sd = SD_data.build(num_electron_up=3, num_electron_dn=2, mos_up=loaded[2], time_reversal=True)
            name, aos = "mo_coefficients_up", sd.orb_data_up_spin.aos_data
            C0 = np.asarray(sd.orb_data_up_spin.mo_coefficients)
        else:
            sd = loaded[4]
            assert isinstance(sd, SD_data) and not sd.time_reversal
            assert 0 < sd.num_occ_dn < sd.num_occ_up
            name, aos = "mo_coefficients_dn", sd.orb_data_dn_spin.aos_data
            C0 = np.asarray(sd.orb_data_dn_spin.mo_coefficients)

        rng = np.random.default_rng(5)
        stepped = C0 + 5e-2 * rng.standard_normal(C0.shape)
        block = make_test_block(name, stepped, field="real", derivative_policy="direct_real", sync_scope="sampler")
        new_sd = sd.apply_block_update(block)

        S = np.asarray(compute_overlap_matrix(aos), dtype=np.float64)
        S = 0.5 * (S + S.T)
        w, V = np.linalg.eigh(stepped @ S @ stepped.T)
        retracted = (V * w**-0.5) @ V.T @ stepped

        C_up = np.asarray(new_sd.orb_data_up_spin.mo_coefficients)
        C_dn = np.asarray(new_sd.orb_data_dn_spin.mo_coefficients)
        atol, rtol = get_tolerance("det_eval", "strict")
        if case == "rohf_up":
            expected_up, expected_dn = retracted, retracted[: sd.num_occ_dn]
        else:
            expected_up, expected_dn = np.asarray(sd.orb_data_up_spin.mo_coefficients), retracted
            np.testing.assert_array_equal(C_up, expected_up)
        np.testing.assert_allclose(C_up, expected_up, atol=atol, rtol=rtol)
        np.testing.assert_allclose(C_dn, expected_dn, atol=atol, rtol=rtol)

        r_up = rng.normal(size=(sd.num_electron_up, 3))
        r_dn = rng.normal(size=(sd.num_electron_dn, 3))

        def _ao(r):
            return np.asarray(compute_AOs(aos, r), dtype=np.float64).T

        reference = np.linalg.slogdet(_ao(r_up) @ expected_up.T)[1] + np.linalg.slogdet(_ao(r_dn) @ expected_dn.T)[1]
        actual = float(compute_ln_det_geminal_all_elements(new_sd, r_up, r_dn))
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvtz_cart.h5"])
    def test_AGPn_data_eval_geminal_via_duck_type_interface(self, trexio_file):
        """The existing ``compute_geminal_all_elements`` kernel accepts an
        ``AGPn_data`` instance via its duck-type interface (``_lambda_matrix_jnp``,
        ``compute_orb_api``) and produces a Slater matrix of the right shape."""
        from jqmc.determinant import compute_geminal_all_elements

        (_, _, _, _, geminal_mo_data, _) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        agpn = AGPn_data.build(
            num_electron_up=int(geminal_mo_data.num_electron_up),
            num_electron_dn=int(geminal_mo_data.num_electron_dn),
            mos_up=geminal_mo_data.orb_data_up_spin,
        )
        n_e = int(geminal_mo_data.num_electron_up)
        rng = np.random.default_rng(2)
        r_up = rng.uniform(-1, 1, (n_e, 3))
        r_dn = rng.uniform(-1, 1, (n_e, 3))

        G = compute_geminal_all_elements(geminal_data=agpn, r_up_carts=r_up, r_dn_carts=r_dn)
        assert G.shape == (n_e, n_e)
        # Sanity: a closed-shell singlet with lambda_diag = [1]*N_occ + [0]*N_virt
        # should give the same Slater matrix as the source ``geminal_mo_data``.
        G_ref = compute_geminal_all_elements(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
        )
        atol, rtol = get_tolerance_min(("ao_eval", "det_eval"), "strict")
        np.testing.assert_allclose(np.asarray(G), np.asarray(G_ref), atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvtz_cart.h5"])
    def test_SD_data_eval_geminal_via_duck_type_interface(self, trexio_file):
        """Mirror for SD_data."""
        from jqmc.determinant import compute_geminal_all_elements

        (_, _, _, _, geminal_mo_data, _) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        sd = SD_data.build(
            num_electron_up=int(geminal_mo_data.num_electron_up),
            num_electron_dn=int(geminal_mo_data.num_electron_dn),
            mos_up=geminal_mo_data.orb_data_up_spin,
        )
        n_e = int(geminal_mo_data.num_electron_up)
        rng = np.random.default_rng(3)
        r_up = rng.uniform(-1, 1, (n_e, 3))
        r_dn = rng.uniform(-1, 1, (n_e, 3))

        G_sd = compute_geminal_all_elements(geminal_data=sd, r_up_carts=r_up, r_dn_carts=r_dn)
        G_ref = compute_geminal_all_elements(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
        )
        atol, rtol = get_tolerance_min(("ao_eval", "det_eval"), "strict")
        np.testing.assert_allclose(np.asarray(G_sd), np.asarray(G_ref), atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["lithium.h5"])
    def test_AGPn_SD_apply_block_update_pbc_real(self, trexio_file):
        """PBC Gamma real path: S-metric polar retraction works on a periodic
        pgto basis and time-reversal pairing is trivially preserved
        (``conj`` is the identity on real coefficients)."""
        (_, _, _, _, geminal_mo_data, _) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        mos_up = geminal_mo_data.orb_data_up_spin
        sd = SD_data.build(
            num_electron_up=int(geminal_mo_data.num_electron_up),
            num_electron_dn=int(geminal_mo_data.num_electron_dn),
            mos_up=mos_up,
        )
        agpn = AGPn_data.build(
            num_electron_up=int(geminal_mo_data.num_electron_up),
            num_electron_dn=int(geminal_mo_data.num_electron_dn),
            mos_up=mos_up,
        )

        rng = np.random.default_rng(11)
        C0 = np.asarray(mos_up.mo_coefficients)
        new_C = C0 + 1e-4 * rng.standard_normal(C0.shape)
        block = make_test_block(
            "mo_coefficients_up",
            new_C,
            field="real",
            derivative_policy="direct_real",
            sync_scope="sampler",
        )

        atol, rtol = get_tolerance("det_eval", "strict")
        for geminal in (sd, agpn):
            new_g = geminal.apply_block_update(block)
            C_new = np.asarray(new_g.orb_data_up_spin.mo_coefficients)
            S = np.asarray(compute_overlap_matrix(mos_up.aos_data))
            M = C_new.conj() @ S @ C_new.T
            np.testing.assert_allclose(M, np.eye(C_new.shape[0]), atol=atol, rtol=rtol)
            # Convention B on real path: C_dn == C_up (conj is identity).
            C_dn = np.asarray(new_g.orb_data_dn_spin.mo_coefficients)
            np.testing.assert_allclose(C_dn, C_new, atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["lithium.h5"])
    def test_AGPn_SD_eval_pbc_real(self, trexio_file):
        """Sanity: AGPn / SD via duck-type eval reproduce the AGP Slater matrix
        on the periodic real-Bloch path."""
        from jqmc.determinant import compute_geminal_all_elements

        (_, _, _, _, geminal_mo_data, _) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        mos_up = geminal_mo_data.orb_data_up_spin
        agpn = AGPn_data.build(
            num_electron_up=int(geminal_mo_data.num_electron_up),
            num_electron_dn=int(geminal_mo_data.num_electron_dn),
            mos_up=mos_up,
        )
        sd = SD_data.build(
            num_electron_up=int(geminal_mo_data.num_electron_up),
            num_electron_dn=int(geminal_mo_data.num_electron_dn),
            mos_up=mos_up,
        )

        n_e = int(geminal_mo_data.num_electron_up)
        rng = np.random.default_rng(12)
        r_up = rng.uniform(-1, 1, (n_e, 3))
        r_dn = rng.uniform(-1, 1, (n_e, 3))

        G_ref = compute_geminal_all_elements(geminal_data=geminal_mo_data, r_up_carts=r_up, r_dn_carts=r_dn)
        G_agpn = compute_geminal_all_elements(geminal_data=agpn, r_up_carts=r_up, r_dn_carts=r_dn)
        G_sd = compute_geminal_all_elements(geminal_data=sd, r_up_carts=r_up, r_dn_carts=r_dn)

        atol, rtol = get_tolerance_min(("ao_eval", "det_eval"), "strict")
        np.testing.assert_allclose(np.asarray(G_agpn), np.asarray(G_ref), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(G_sd), np.asarray(G_ref), atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["lithium_k025.h5"])
    def test_AGPn_SD_apply_block_update_pbc_complex(self, trexio_file):
        """PBC complex Bloch path: complex MO coefficients carry through
        ``apply_block_update`` (Hermitian polar retraction) and Convention B
        (``C_dn = conj(C_up)``) is non-trivial."""
        (_, _, _, _, geminal_mo_data, _) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        mos_up = geminal_mo_data.orb_data_up_spin
        assert np.iscomplexobj(np.asarray(mos_up.mo_coefficients)), (
            "Test precondition: lithium_k025.h5 must carry complex Bloch MOs."
        )
        sd = SD_data.build(
            num_electron_up=int(geminal_mo_data.num_electron_up),
            num_electron_dn=int(geminal_mo_data.num_electron_dn),
            mos_up=mos_up,
        )
        agpn = AGPn_data.build(
            num_electron_up=int(geminal_mo_data.num_electron_up),
            num_electron_dn=int(geminal_mo_data.num_electron_dn),
            mos_up=mos_up,
        )
        # lambda_diag must stay real even though MOs are complex.
        assert agpn.lambda_diag.dtype == np.float64

        rng = np.random.default_rng(21)
        C0 = np.asarray(mos_up.mo_coefficients)
        delta = 1e-4 * (rng.standard_normal(C0.shape) + 1j * rng.standard_normal(C0.shape))
        new_C = C0 + delta
        block = make_test_block(
            "mo_coefficients_up",
            new_C,
            field="complex",
            derivative_policy="wirtinger_pair",
            sync_scope="sampler",
        )

        atol, rtol = get_tolerance("det_eval", "strict")
        for geminal in (sd, agpn):
            new_g = geminal.apply_block_update(block)
            C_new = np.asarray(new_g.orb_data_up_spin.mo_coefficients)
            assert np.iscomplexobj(C_new), "Complex dtype must survive the retraction."
            S = np.asarray(compute_overlap_matrix(mos_up.aos_data))
            M = C_new.conj() @ S @ C_new.T
            # Hermitian projector (uses ``.conj().T``, not ``.T``).
            np.testing.assert_allclose(M, np.eye(C_new.shape[0]), atol=atol, rtol=rtol)
            # Convention B is non-trivial on the complex path: C_dn = conj(C_up).
            C_dn = np.asarray(new_g.orb_data_dn_spin.mo_coefficients)
            np.testing.assert_allclose(C_dn, np.conj(C_new), atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["lithium_k025.h5"])
    def test_AGPn_SD_eval_pbc_complex(self, trexio_file):
        """Complex Bloch eval parity: duck-type AGPn / SD reproduce the source
        AGP Slater matrix on the complex Bloch path."""
        from jqmc.determinant import compute_geminal_all_elements

        (_, _, _, _, geminal_mo_data, _) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        mos_up = geminal_mo_data.orb_data_up_spin
        agpn = AGPn_data.build(
            num_electron_up=int(geminal_mo_data.num_electron_up),
            num_electron_dn=int(geminal_mo_data.num_electron_dn),
            mos_up=mos_up,
        )
        sd = SD_data.build(
            num_electron_up=int(geminal_mo_data.num_electron_up),
            num_electron_dn=int(geminal_mo_data.num_electron_dn),
            mos_up=mos_up,
        )

        n_e = int(geminal_mo_data.num_electron_up)
        rng = np.random.default_rng(22)
        r_up = rng.uniform(-1, 1, (n_e, 3))
        r_dn = rng.uniform(-1, 1, (n_e, 3))

        G_ref = compute_geminal_all_elements(geminal_data=geminal_mo_data, r_up_carts=r_up, r_dn_carts=r_dn)
        G_agpn = compute_geminal_all_elements(geminal_data=agpn, r_up_carts=r_up, r_dn_carts=r_dn)
        G_sd = compute_geminal_all_elements(geminal_data=sd, r_up_carts=r_up, r_dn_carts=r_dn)

        atol, rtol = get_tolerance_min(("ao_eval", "det_eval"), "strict")
        np.testing.assert_allclose(np.asarray(G_agpn), np.asarray(G_ref), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(G_sd), np.asarray(G_ref), atol=atol, rtol=rtol)

        # ``lambda_diag`` is a real physical parameter, but its local
        # determinant contraction is complex at non-Gamma twist.  Exercise
        # the reverse-mode boundary directly: the session-wide
        # ComplexWarning guard makes an implicit complex-to-real cotangent
        # cast fail here, while the explicit physical Re projection remains
        # real and finite.
        assert jnp.iscomplexobj(agpn._lambda_matrix_jnp)

        def summed_real_geminal(lambda_diag):
            trial = agpn.replace(lambda_diag=lambda_diag)
            return jnp.real(jnp.sum(compute_geminal_all_elements(geminal_data=trial, r_up_carts=r_up, r_dn_carts=r_dn)))

        lambda_diag_grad = jax.grad(summed_real_geminal)(jnp.asarray(agpn.lambda_diag))
        assert not jnp.iscomplexobj(lambda_diag_grad)
        assert np.all(np.isfinite(np.asarray(lambda_diag_grad)))

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["lithium_k025.h5"])
    def test_density_TRS_invariant_pbc_complex(self, trexio_file):
        """Closed-shell singlet density satisfies the time-reversal invariant
        ``|Psi(r_up, r_dn)|^2 == |Psi(r_dn, r_up)|^2`` at every electron
        configuration.

        Mathematical form, for the BCS / TR-pair geminal at twist k:

          Psi(r_up, r_dn) = det[G(r_up_i, r_dn_j)]
                          = det[sum_k phi^k(r_up_i) * conj(phi^k)(r_dn_j)]

        Swapping ``r_up`` and ``r_dn`` (the spatial action of time reversal
        on a singlet, which flips spin labels and complex-conjugates) gives
        ``G_swap = G^H`` (Hermitian transpose) so
        ``Psi(r_dn, r_up) = conj(det(G)) = conj(Psi(r_up, r_dn))``, hence
        ``|Psi(r_dn, r_up)|^2 == |Psi(r_up, r_dn)|^2`` exactly.

        Regression target: a wavefunction in which the up-channel orbital is
        at ``+k`` but the dn-channel orbital is a non-physical mixed-phase
        state (the "conj(coefficients) on the same +k AO leaf" trap that the
        previous Convention-B fix attempt fell into) does NOT satisfy this
        invariant -- the density gains a spurious phase modulation under the
        swap. The geminal-eval consistency test
        ``test_AGPn_SD_eval_pbc_complex`` is happy with such a state because
        every code path uses the same broken convention; the swap test fails
        because the broken convention violates an absolute physical invariant.
        """
        from jqmc.determinant import compute_geminal_all_elements

        (_, _, _, _, geminal_data, _) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )
        if not bool(getattr(geminal_data, "time_reversal", True)):
            pytest.skip("test asserts closed-shell singlet invariant; geminal carries time_reversal=False")
        if int(geminal_data.num_electron_up) != int(geminal_data.num_electron_dn):
            pytest.skip("test asserts closed-shell singlet invariant; geminal has N_up != N_dn")

        n_e = int(geminal_data.num_electron_up)
        rng = np.random.default_rng(20260620)
        # Sample several independent configurations so a single accidentally-
        # symmetric draw cannot mask a broken invariant.
        n_samples = 5
        for sample_idx in range(n_samples):
            r_up = rng.uniform(-1.0, 1.0, (n_e, 3)).astype(np.float64)
            r_dn = rng.uniform(-1.0, 1.0, (n_e, 3)).astype(np.float64)
            G_orig = np.asarray(compute_geminal_all_elements(geminal_data=geminal_data, r_up_carts=r_up, r_dn_carts=r_dn))
            G_swap = np.asarray(compute_geminal_all_elements(geminal_data=geminal_data, r_up_carts=r_dn, r_dn_carts=r_up))
            psi_orig_sq = float(np.abs(np.linalg.det(G_orig)) ** 2)
            psi_swap_sq = float(np.abs(np.linalg.det(G_swap)) ** 2)
            atol, rtol = get_tolerance_min(("ao_eval", "det_eval"), "strict")
            np.testing.assert_allclose(
                psi_swap_sq,
                psi_orig_sq,
                atol=atol,
                rtol=rtol,
                err_msg=(
                    f"sample {sample_idx}: |Psi(r_up, r_dn)|^2 = {psi_orig_sq:.12e}, "
                    f"|Psi(r_dn, r_up)|^2 = {psi_swap_sq:.12e}; broken TRS singlet."
                ),
            )


class TestVariationalParameterBlock:
    @pytest.mark.light
    def test_real_gamma_agp_publishes_a_real_symmetric_lambda_tie(self):
        """A real TR AGP must not inherit imaginary Hermitian directions."""
        from jqmc.jastrow_factor import Jastrow_data
        from jqmc.wavefunction import Wavefunction_data, _project_linear_tie_storage

        mos = _build_minimal_mos_data(num_mo=3, num_ao=4)
        agp = AGP_data.build(
            num_electron_up=2,
            num_electron_dn=2,
            orb_data_up_spin=mos,
            orb_data_dn_spin=mos,
            lambda_matrix=np.eye(3, dtype=np.float64),
            time_reversal=True,
        )
        wf = Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=agp)
        blocks = wf.get_variational_blocks(
            state_token=("test", 0),
            opt_J1_param=False,
            opt_J2_param=False,
            opt_J3_param=False,
            opt_geminal=True,
        )
        block = next(block for block in blocks if block.name == "lambda_matrix")
        assert block.field == "real"
        assert block.derivative_policy == "direct_real"
        assert block.tangent_spec.kind == "linear_tie"
        tie = block.tangent_spec.linear_tie
        assert tie is not None
        assert not np.iscomplexobj(tie.coefficients)

        source = np.arange(block.size, dtype=np.float64)[None, :]
        projected = _project_linear_tie_storage(source, tie, field=block.field).reshape(block.shape)
        np.testing.assert_array_equal(projected, projected.T)

    @pytest.mark.light
    def test_get_variational_blocks_open_shell_exposes_mo_coefficients_dn(self):
        """For ``time_reversal=False`` open-shell SD, both
        ``mo_coefficients_up`` AND ``mo_coefficients_dn`` must appear in
        ``Wavefunction_data.get_variational_blocks``.

        Regression test for an issue where only ``mo_coefficients_up`` was
        exposed -- the dn channel is variationally independent under UHF
        (no time-reversal pairing) and was silently frozen by the opt loop.
        """
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        (_, _, _, _, geminal_data, _) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "N_ae_ccpvdz_cart.h5"),
            store_tuple=True,
        )
        assert isinstance(geminal_data, SD_data)
        assert geminal_data.time_reversal is False
        assert int(geminal_data.num_electron_up) != int(geminal_data.num_electron_dn)

        jastrow = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp"),
            jastrow_three_body_data=None,
        )
        wf = Wavefunction_data(geminal_data=geminal_data, jastrow_data=jastrow)
        blocks = wf.get_variational_blocks(
            state_token=("test", 0),
            opt_J1_param=False,
            opt_J2_param=False,
            opt_J3_param=False,
            opt_geminal=True,
        )
        names = [b.name for b in blocks]
        assert "mo_coefficients_up" in names
        assert "mo_coefficients_dn" in names, (
            f"Open-shell SD_data must expose mo_coefficients_dn as an independent variational block; got blocks={names}"
        )
        by_name = {block.name: block for block in blocks}
        assert by_name["mo_coefficients_up"].shape == (
            geminal_data.num_electron_up,
            geminal_data.orb_data_up_spin.aos_data.num_ao,
        )
        assert by_name["mo_coefficients_dn"].shape == (
            geminal_data.num_electron_dn,
            geminal_data.orb_data_dn_spin.aos_data.num_ao,
        )
        assert not ({"lambda_matrix", "lambda_diag", "lambda_unpaired"} & set(names))

        # SR step on the independent occupied dn channel: build a genuinely
        # complex delta that breaks S-orthonormality.
        rng = np.random.default_rng(7)
        C_dn_0 = np.asarray(geminal_data.orb_data_dn_spin.mo_coefficients)
        delta = 1e-2 * (rng.standard_normal(C_dn_0.shape) + 1j * rng.standard_normal(C_dn_0.shape))
        new_C_dn = C_dn_0 + delta
        block = make_test_block(
            name="mo_coefficients_dn",
            values=new_C_dn,
            field="complex",
            derivative_policy="holomorphic",
            sync_scope="sampler",
        )
        atol, rtol = get_tolerance("det_eval", "strict")
        for source in (geminal_data,):
            updated = source.apply_block_update(block)
            C_dn_post = np.asarray(updated.orb_data_dn_spin.mo_coefficients)

            S_dn = np.asarray(compute_overlap_matrix(updated.orb_data_dn_spin.aos_data))
            M_dn = C_dn_post.conj() @ S_dn @ C_dn_post.T
            np.testing.assert_allclose(M_dn, np.eye(C_dn_post.shape[0]), atol=atol, rtol=rtol)

            # The up channel must NOT have moved when only dn was applied.
            np.testing.assert_array_equal(
                np.asarray(updated.orb_data_up_spin.mo_coefficients),
                np.asarray(source.orb_data_up_spin.mo_coefficients),
            )

    @pytest.mark.light
    def test_get_variational_blocks_closed_shell_does_not_expose_mo_coefficients_dn(self):
        """``time_reversal=True`` (closed-shell singlet): the dn block is NOT
        exposed -- the dn channel is derived via Convention B
        ``C_dn = conj(C_up)`` inside the ``mo_coefficients_up`` apply.
        """
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        (_, _, _, _, geminal_data, _) = read_trexio_file(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ae_ccpvtz_cart.h5"
            ),
            store_tuple=True,
        )
        assert geminal_data.time_reversal is True
        jastrow = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp"),
            jastrow_three_body_data=None,
        )
        wf = Wavefunction_data(geminal_data=geminal_data, jastrow_data=jastrow)
        blocks = wf.get_variational_blocks(
            state_token=("test", 0),
            opt_J1_param=False,
            opt_J2_param=False,
            opt_J3_param=False,
            opt_geminal=True,
        )
        names = [b.name for b in blocks]
        assert "mo_coefficients_up" in names
        assert "mo_coefficients_dn" not in names, (
            f"Closed-shell singlet must NOT expose mo_coefficients_dn (Convention B derives it from C_up); got blocks={names}"
        )


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
