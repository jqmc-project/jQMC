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

from jqmc._setting import ATOL_ORBITAL_EMBEDDING_MATCH, RTOL_ORBITAL_EMBEDDING_MATCH
import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from tests.test_jqmc_kernel._gdd_fd_adapters import fd_case_parameters, record_fd_case

pytestmark = [pytest.mark.with_chdir]
# Add the project root directory to sys.path to allow executing this script directly
# This is necessary because relative imports (e.g. 'from ..jqmc') are not allowed
# when running a script directly (as __main__).
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# All AO leaves + the agnostic forward/grad/lap helpers come from
# jqmc.atomic_orbital. The MO-layer helpers (_compute_MOs_*) route through
# compute_MOs -> compute_AOs / _compute_AOs_debug, so they are leaf-agnostic
# and cover the periodic PW / B-spline / pGTO / composite leaves unchanged.
from jqmc.atomic_orbital import (
    _aos_cart_to_sphe,
    _aos_sphe_to_cart,
    _cart_to_spherical_matrix,
    _compute_overlap_matrix_cart_analytic,
    _spherical_to_cart_matrix,
)
from jqmc._precision import get_dtype_jnp, get_tolerance, get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL, get_tolerance_mixed_fidelity
from jqmc.atomic_orbital import (
    AOs_bspline_data,
    AOs_cart_gto_data,
    AOs_cart_pgto_data,
    AOs_cart_pgto_complex_data,
    AOs_plane_wave_data,
    AOs_sphe_gto_data,
    AOs_sphe_pgto_data,
    AOs_sphe_pgto_complex_data,
    _apply_image_expansion,
    _detile_ragged_to_molecular,
    compute_AOs,
    compute_overlap_matrix,
)
from jqmc.molecular_orbital import (
    MOs_data,
    _compute_MOs_debug,
    _compute_MOs_grad_autodiff,
    _compute_MOs_grad_debug,
    _compute_MOs_laplacian_autodiff,
    _compute_MOs_laplacian_debug,
    compute_MOs,
    compute_MOs_grad,
    compute_MOs_laplacian,
    compute_MOs_value_grad_lap,
)

# Spline prefiltering lives in the projection module (not atomic_orbital);
# only the periodic B-spline leaf builder needs it.
from jqmc.pw_pgto_projection import bspline_prefilter_periodic
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


def _assert_orbital_derivative_mixed_fidelity(
    mixed: np.ndarray,
    full: np.ndarray,
    atol: float,
    rtol: float,
) -> None:
    """Require one orbital-derivative block to satisfy its mixed error model."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


def _molecular_mos_derivative_case(
    exponents: tuple[float, ...],
    coefficients: tuple[float, ...],
) -> tuple[MOs_data, np.ndarray]:
    """Build the deterministic molecular MO derivative fixture."""
    fixture_rng = np.random.RandomState(20260923)
    num_el = 8
    num_mo = 4
    num_ao = 3
    r_carts = 4.0 * fixture_rng.rand(num_el, 3) - 2.0
    R_carts = 2.0 * fixture_rng.rand(num_ao, 3) - 1.0
    mo_coefficients = fixture_rng.rand(num_mo, num_ao)
    structure_data = Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([0] * num_ao),
        element_symbols=tuple(["X"] * num_ao),
        atomic_labels=tuple(["X"] * num_ao),
    )
    aos_data = AOs_sphe_gto_data._from_full_basis(
        structure_data=structure_data,
        nucleus_index=(0, 1, 2),
        num_ao=num_ao,
        num_ao_prim=4,
        orbital_indices=(0, 0, 1, 2),
        exponents=np.asarray(exponents, dtype=np.float64),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        angular_momentums=(1, 1, 1),
        magnetic_quantum_numbers=(0, 1, -1),
    )
    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)
    mos_data.sanity_check()
    return mos_data, r_carts


def _independent_native_cart_embedding(aos_cart: AOs_cart_gto_data, aos_sphe: AOs_sphe_gto_data) -> np.ndarray:
    """Build emitted-spherical -> native-Cartesian ``T`` without conversion helpers.

    TREXIO component normalizations make Cartesian radial vectors
    proportional rather than identical.  Match every emitted spherical shell
    to the native Cartesian shell and divide the analytic angular embedding by
    those independently fitted component scales.
    """
    cart_orbital = np.asarray(aos_cart.orbital_indices)
    cart_exponents = np.asarray(aos_cart.exponents)
    cart_coefficients = np.asarray(aos_cart.coefficients)
    cart_nucleus = np.asarray(aos_cart.nucleus_index)
    cart_angular = np.asarray(aos_cart.angular_momentums)
    px = np.asarray(aos_cart.polynominal_order_x)
    py = np.asarray(aos_cart.polynominal_order_y)
    pz = np.asarray(aos_cart.polynominal_order_z)
    sphe_orbital = np.asarray(aos_sphe.orbital_indices)
    sphe_exponents = np.asarray(aos_sphe.exponents)
    sphe_coefficients = np.asarray(aos_sphe.coefficients)
    sphe_nucleus = np.asarray(aos_sphe.nucleus_index)
    sphe_angular = np.asarray(aos_sphe.angular_momentums)
    embedding = np.zeros((aos_sphe.num_ao, aos_cart.num_ao), dtype=np.float64)

    sphe_start = 0
    while sphe_start < aos_sphe.num_ao:
        l = int(sphe_angular[sphe_start])
        n_sphe = 2 * l + 1
        sphe_indices = list(range(sphe_start, sphe_start + n_sphe))
        reference_primitives = np.flatnonzero(sphe_orbital == sphe_start)
        reference_exponents = sphe_exponents[reference_primitives]
        reference_coefficients = sphe_coefficients[reference_primitives]
        reference_norm_sq = float(reference_coefficients @ reference_coefficients)
        by_order: dict[tuple[int, int, int], tuple[int, float]] = {}

        for ao in range(aos_cart.num_ao):
            primitives = np.flatnonzero(cart_orbital == ao)
            if (
                cart_nucleus[ao] != sphe_nucleus[sphe_start]
                or cart_angular[ao] != l
                or len(primitives) != len(reference_primitives)
                or not np.allclose(
                    cart_exponents[primitives],
                    reference_exponents,
                    rtol=RTOL_ORBITAL_EMBEDDING_MATCH,
                    atol=ATOL_ORBITAL_EMBEDDING_MATCH,
                )
            ):
                continue
            scale = float(reference_coefficients @ cart_coefficients[primitives] / reference_norm_sq)
            if scale != 0.0 and np.allclose(
                cart_coefficients[primitives],
                scale * reference_coefficients,
                rtol=RTOL_ORBITAL_EMBEDDING_MATCH,
                atol=ATOL_ORBITAL_EMBEDDING_MATCH,
            ):
                by_order[(int(px[ao]), int(py[ao]), int(pz[ao]))] = (ao, scale)

        orders = [(nx, ny, l - nx - ny) for nx in range(l, -1, -1) for ny in range(l - nx, -1, -1)]
        assert set(by_order) == set(orders)
        cart_indices = [by_order[order][0] for order in orders]
        component_scales = np.asarray([by_order[order][1] for order in orders])
        embedding[np.ix_(sphe_indices, cart_indices)] = _spherical_to_cart_matrix(l) / component_scales[None, :]
        sphe_start += n_sphe

    return embedding


class TestMOsCore:
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "exponents": [50.0, 20.0, 10.0, 5.0],
                    "magnetic_quantum_numbers": [0, 0, -1],
                    "r_cart_min": -5.0,
                    "r_cart_max": 5.0,
                    "R_cart_min": -6.0,
                    "R_cart_max": 6.0,
                },
                id="config_A",
            ),
            pytest.param(
                {
                    "exponents": [10.0, 5.0, 1.0, 1.0],
                    "magnetic_quantum_numbers": [0, 1, -1],
                    "r_cart_min": -5.0,
                    "r_cart_max": 5.0,
                    "R_cart_min": -6.0,
                    "R_cart_max": 6.0,
                },
                id="config_B",
            ),
        ],
    )
    @pytest.mark.light
    def test_MOs_comparing_jax_and_debug_implemenetations(self, config):
        """Test the MO computation, comparing JAX and debug implementations."""
        fixture_rng = np.random.RandomState(20260923)
        num_el = 10
        num_mo = 5
        num_ao = 3
        num_ao_prim = 4
        orbital_indices = [0, 0, 1, 2]
        exponents = config["exponents"]
        coefficients = [1.0, 1.0, 1.0, 0.5]
        angular_momentums = [1, 1, 1]
        magnetic_quantum_numbers = config["magnetic_quantum_numbers"]

        orbital_indices = tuple(orbital_indices)
        exponents = np.array(exponents, dtype=np.float64)
        coefficients = np.array(coefficients, dtype=np.float64)
        angular_momentums = tuple(angular_momentums)
        magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

        num_r_cart_samples = num_el
        num_R_cart_samples = num_ao
        r_cart_min, r_cart_max = config["r_cart_min"], config["r_cart_max"]
        R_cart_min, R_cart_max = config["R_cart_min"], config["R_cart_max"]
        r_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_cart_samples, 3) + r_cart_min
        R_carts = (R_cart_max - R_cart_min) * fixture_rng.rand(num_R_cart_samples, 3) + R_cart_min

        mo_coefficients = fixture_rng.rand(num_mo, num_ao)

        structure_data = Structure_data(
            pbc_flag=False,
            positions=R_carts,
            atomic_numbers=tuple([0] * num_R_cart_samples),
            element_symbols=tuple(["X"] * num_R_cart_samples),
            atomic_labels=tuple(["X"] * num_R_cart_samples),
        )

        aos_data = AOs_sphe_gto_data._from_full_basis(
            structure_data=structure_data,
            nucleus_index=tuple(list(range(num_R_cart_samples))),
            num_ao=num_ao,
            num_ao_prim=num_ao_prim,
            orbital_indices=orbital_indices,
            exponents=exponents,
            coefficients=coefficients,
            angular_momentums=angular_momentums,
            magnetic_quantum_numbers=magnetic_quantum_numbers,
        )

        mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

        mos_data.sanity_check()

        mo_ans_all_jax = compute_MOs(mos_data=mos_data, r_carts=r_carts)

        mo_ans_all_debug = _compute_MOs_debug(mos_data=mos_data, r_carts=r_carts)

        # Path crosses ao_eval (fp32 in mixed) -> mo_eval (fp64); use the looser
        # of the two so the test reflects the achievable agreement.
        atol, rtol = get_tolerance_min(["ao_eval", "mo_eval"], "strict")
        assert not np.any(np.isnan(np.asarray(mo_ans_all_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mo_ans_all_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(mo_ans_all_debug, mo_ans_all_jax, atol=atol, rtol=rtol)

        jax.clear_caches()


class TestMOsGradLaplacian:
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "exponents": [50.0, 20.0, 10.0, 5.0],
                    "magnetic_quantum_numbers": [0, 0, -1],
                    "r_cart_min": -5.0,
                    "r_cart_max": 5.0,
                    "R_cart_min": 10.0,
                    "R_cart_max": 10.0,
                    "numerical_impl": "autodiff",
                },
                id="config_A",
            ),
            pytest.param(
                {
                    "exponents": [10.0, 5.0, 1.0, 1.0],
                    "magnetic_quantum_numbers": [0, 1, -1],
                    "r_cart_min": -1.0,
                    "r_cart_max": 1.0,
                    "R_cart_min": 3.0,
                    "R_cart_max": 3.0,
                    "numerical_impl": "debug",
                },
                id="config_B",
            ),
        ],
    )
    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-MO-DERIVATIVE-CONSISTENCY-001",
    )
    def test_MOs_comparing_auto_and_numerical_grads(self, config):
        """Test the MO gradient computation, comparing JAX and debug implementations.

        Scope of the cases: ``config_A`` puts all
        three nuclei at (10, 10, 10) and the electrons in [-5, 5]^3, so the smallest
        exponent (5) still gives exp(-1076) and every AO, and hence every compared
        gradient element, underflows to exactly zero in both arrays. ``config_A``
        therefore establishes only finite output on underflow, not derivative
        agreement. Nonzero MO derivative agreement is established by
        ``test_MOs_comparing_analytic_and_auto_grads`` (both precisions).
        """
        fixture_rng = np.random.RandomState(20260923)
        num_el = 10
        num_mo = 5
        num_ao = 3
        num_ao_prim = 4
        orbital_indices = [0, 0, 1, 2]
        exponents = config["exponents"]
        coefficients = [1.0, 1.0, 1.0, 0.5]
        angular_momentums = [1, 1, 1]
        magnetic_quantum_numbers = config["magnetic_quantum_numbers"]

        orbital_indices = tuple(orbital_indices)
        exponents = np.array(exponents, dtype=np.float64)
        coefficients = np.array(coefficients, dtype=np.float64)
        angular_momentums = tuple(angular_momentums)
        magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

        num_r_cart_samples = num_el
        num_R_cart_samples = num_ao
        r_cart_min, r_cart_max = config["r_cart_min"], config["r_cart_max"]
        R_cart_min, R_cart_max = config["R_cart_min"], config["R_cart_max"]
        r_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_cart_samples, 3) + r_cart_min
        R_carts = (R_cart_max - R_cart_min) * fixture_rng.rand(num_R_cart_samples, 3) + R_cart_min

        mo_coefficients = fixture_rng.rand(num_mo, num_ao)

        structure_data = Structure_data(
            pbc_flag=False,
            positions=R_carts,
            atomic_numbers=tuple([0] * num_R_cart_samples),
            element_symbols=tuple(["X"] * num_R_cart_samples),
            atomic_labels=tuple(["X"] * num_R_cart_samples),
        )

        aos_data = AOs_sphe_gto_data._from_full_basis(
            structure_data=structure_data,
            nucleus_index=tuple(list(range(num_R_cart_samples))),
            num_ao=num_ao,
            num_ao_prim=num_ao_prim,
            orbital_indices=orbital_indices,
            exponents=exponents,
            coefficients=coefficients,
            angular_momentums=angular_momentums,
            magnetic_quantum_numbers=magnetic_quantum_numbers,
        )

        mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)
        mos_data.sanity_check()

        mo_matrix_grad_x_auto, mo_matrix_grad_y_auto, mo_matrix_grad_z_auto = compute_MOs_grad(
            mos_data=mos_data, r_carts=r_carts
        )

        numerical_fn = _compute_MOs_grad_autodiff if config["numerical_impl"] == "autodiff" else _compute_MOs_grad_debug
        (
            mo_matrix_grad_x_numerical,
            mo_matrix_grad_y_numerical,
            mo_matrix_grad_z_numerical,
        ) = numerical_fn(mos_data=mos_data, r_carts=r_carts)

        atol, rtol = get_tolerance("mo_grad", "strict")
        assert not np.any(np.isnan(np.asarray(mo_matrix_grad_x_auto))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mo_matrix_grad_x_numerical))), "NaN detected in second argument"
        np.testing.assert_allclose(mo_matrix_grad_x_auto, mo_matrix_grad_x_numerical, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(mo_matrix_grad_y_auto))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mo_matrix_grad_y_numerical))), "NaN detected in second argument"
        np.testing.assert_allclose(mo_matrix_grad_y_auto, mo_matrix_grad_y_numerical, atol=atol, rtol=rtol)

        assert not np.any(np.isnan(np.asarray(mo_matrix_grad_z_auto))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mo_matrix_grad_z_numerical))), "NaN detected in second argument"
        np.testing.assert_allclose(mo_matrix_grad_z_auto, mo_matrix_grad_z_numerical, atol=atol, rtol=rtol)

        jax.clear_caches()

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "exponents": [50.0, 20.0, 10.0, 5.0],
                    "magnetic_quantum_numbers": [0, 0, -1],
                    "r_cart_min": -5.0,
                    "r_cart_max": 5.0,
                    "R_cart_min": 10.0,
                    "R_cart_max": 10.0,
                },
                id="config_A",
            ),
            pytest.param(
                {
                    "exponents": [10.0, 5.0, 1.0, 1.0],
                    "magnetic_quantum_numbers": [0, 1, -1],
                    "r_cart_min": -1.0,
                    "r_cart_max": 1.0,
                    "R_cart_min": 3.0,
                    "R_cart_max": 3.0,
                },
                id="config_B",
            ),
        ],
    )
    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-MO-DERIVATIVE-CONSISTENCY-001",
    )
    def test_MOs_comparing_auto_and_numerical_laplacians(self, config):
        """Test the MO Laplacian computation, comparing JAX and debug implementations.

        Scope of the cases: ``config_A`` puts all
        three nuclei at (10, 10, 10) and the electrons in [-5, 5]^3, so every AO, and
        hence every compared Laplacian element, underflows to exactly zero in both
        arrays. ``config_A`` therefore establishes only finite output on underflow,
        not derivative agreement. Nonzero MO Laplacian agreement is established by
        ``test_MOs_comparing_analytic_and_auto_laplacians`` (both precisions).
        """
        fixture_rng = np.random.RandomState(20260923)
        num_el = 10
        num_mo = 5
        num_ao = 3
        num_ao_prim = 4
        orbital_indices = [0, 0, 1, 2]
        exponents = config["exponents"]
        coefficients = [1.0, 1.0, 1.0, 0.5]
        angular_momentums = [1, 1, 1]
        magnetic_quantum_numbers = config["magnetic_quantum_numbers"]

        orbital_indices = tuple(orbital_indices)
        exponents = np.array(exponents, dtype=np.float64)
        coefficients = np.array(coefficients, dtype=np.float64)
        angular_momentums = tuple(angular_momentums)
        magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

        num_r_cart_samples = num_el
        num_R_cart_samples = num_ao
        r_cart_min, r_cart_max = config["r_cart_min"], config["r_cart_max"]
        R_cart_min, R_cart_max = config["R_cart_min"], config["R_cart_max"]
        r_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_cart_samples, 3) + r_cart_min
        R_carts = (R_cart_max - R_cart_min) * fixture_rng.rand(num_R_cart_samples, 3) + R_cart_min

        mo_coefficients = fixture_rng.rand(num_mo, num_ao)

        structure_data = Structure_data(
            pbc_flag=False,
            positions=R_carts,
            atomic_numbers=tuple([0] * num_R_cart_samples),
            element_symbols=tuple(["X"] * num_R_cart_samples),
            atomic_labels=tuple(["X"] * num_R_cart_samples),
        )

        aos_data = AOs_sphe_gto_data._from_full_basis(
            structure_data=structure_data,
            nucleus_index=tuple(list(range(num_R_cart_samples))),
            num_ao=num_ao,
            num_ao_prim=num_ao_prim,
            orbital_indices=orbital_indices,
            exponents=exponents,
            coefficients=coefficients,
            angular_momentums=angular_momentums,
            magnetic_quantum_numbers=magnetic_quantum_numbers,
        )

        mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)
        mos_data.sanity_check()

        mo_matrix_laplacian_numerical = _compute_MOs_laplacian_debug(mos_data=mos_data, r_carts=r_carts)

        mo_matrix_laplacian_auto = _compute_MOs_laplacian_autodiff(mos_data=mos_data, r_carts=r_carts)

        atol, rtol = get_tolerance("mo_lap", "medium")
        assert not np.any(np.isnan(np.asarray(mo_matrix_laplacian_auto))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mo_matrix_laplacian_numerical))), "NaN detected in second argument"
        np.testing.assert_allclose(
            mo_matrix_laplacian_auto,
            mo_matrix_laplacian_numerical,
            atol=atol,
            rtol=rtol,
        )

        jax.clear_caches()

    @pytest.mark.light
    def test_MOs_comparing_analytic_and_auto_grads(self):
        """Check full analytical/AD correctness and mixed/full MO-gradient fidelity."""

        def case_factory():
            return _molecular_mos_derivative_case(
                (20.0, 10.0, 5.0, 2.0),
                (1.0, 0.8, 1.1, 0.7),
            )

        def analytical(case):
            mos_data, r_carts = case
            return compute_MOs_grad(mos_data=mos_data, r_carts=r_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
            for mixed_block, full_block in blocks:
                _assert_orbital_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_orbital_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
            jax.clear_caches()
            return

        mos_data, r_carts = case_factory()
        grad_x_an, grad_y_an, grad_z_an = compute_MOs_grad(mos_data=mos_data, r_carts=r_carts)
        grad_x_auto, grad_y_auto, grad_z_auto = _compute_MOs_grad_autodiff(mos_data=mos_data, r_carts=r_carts)

        # Full precision retains the independent analytical-versus-AD oracle.
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap", "mo_grad"], "strict")
        assert not np.any(np.isnan(np.asarray(grad_x_an))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(grad_x_auto))), "NaN detected in second argument"
        np.testing.assert_allclose(grad_x_an, grad_x_auto, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(grad_y_an))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(grad_y_auto))), "NaN detected in second argument"
        np.testing.assert_allclose(grad_y_an, grad_y_auto, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(grad_z_an))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(grad_z_auto))), "NaN detected in second argument"
        np.testing.assert_allclose(grad_z_an, grad_z_auto, atol=atol, rtol=rtol)

        jax.clear_caches()

    @pytest.mark.light
    def test_MOs_comparing_analytic_and_auto_laplacians(self):
        """Check full analytical/AD correctness and mixed/full MO-Laplacian fidelity."""

        def case_factory():
            return _molecular_mos_derivative_case(
                (15.0, 8.0, 4.0, 2.5),
                (1.0, 0.9, 0.6, 0.7),
            )

        def analytical(case):
            mos_data, r_carts = case
            return compute_MOs_laplacian(mos_data=mos_data, r_carts=r_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
            for mixed_block, full_block in blocks:
                _assert_orbital_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_orbital_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
            jax.clear_caches()
            return

        mos_data, r_carts = case_factory()
        mo_lap_an = compute_MOs_laplacian(mos_data=mos_data, r_carts=r_carts)
        mo_lap_auto = _compute_MOs_laplacian_autodiff(mos_data=mos_data, r_carts=r_carts)

        # Full precision retains the independent analytical-versus-AD oracle.
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap", "mo_lap"], "strict")
        assert not np.any(np.isnan(np.asarray(mo_lap_an))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mo_lap_auto))), "NaN detected in second argument"
        np.testing.assert_allclose(mo_lap_an, mo_lap_auto, atol=atol, rtol=rtol)

        jax.clear_caches()


class TestMOsConversions:
    @pytest.mark.light
    @pytest.mark.parametrize(
        ("trexio_file", "conversion"),
        [
            ("H2_ae_ccpvdz_sphe.h5", "to_cartesian"),
            ("H2_ae_ccpvdz_cart.h5", "to_spherical"),
            ("lithium_k025.h5", "to_cartesian"),
            ("lithium_k025_cart.h5", "to_spherical"),
        ],
    )
    def test_implicit_identity_angular_conversion(self, trexio_file, conversion, monkeypatch):
        """Implicit identity has the same converted VGL as dense I, without expanding I."""
        fixture = Path(__file__).resolve().parents[1] / "trexio_example_files" / trexio_file
        _, _, source, _, _, _ = read_trexio_file(trexio_file=str(fixture), store_tuple=True)
        aos = source.aos_data
        implicit = MOs_data.build_identity(aos_data=aos)
        dense = MOs_data.build(aos_data=aos, mo_coefficients=np.eye(aos.num_ao, dtype=implicit.mo_coefficients.dtype))
        expected = getattr(dense, conversion)()

        def forbidden(self):
            pytest.fail("angular conversion materialized the implicit identity")

        with monkeypatch.context() as patch:
            patch.setattr(MOs_data, "_mo_coefficients_jnp", property(forbidden))
            actual = getattr(implicit, conversion)()
        assert not actual.identity
        assert actual.num_mo == aos.num_ao
        assert type(actual.aos_data) is type(expected.aos_data)
        assert actual.mo_coefficients.shape == (aos.num_ao, actual.aos_data.num_ao)
        assert actual.mo_coefficients.dtype == expected.mo_coefficients.dtype
        actual.sanity_check()
        np.testing.assert_array_equal(actual.mo_coefficients, expected.mo_coefficients)
        points = np.array([[0.13, 0.27, 0.41], [0.71, -0.19, 0.33], [-0.22, 0.58, 0.14]])
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap", "mo_eval", "mo_grad", "mo_lap"], "strict")
        for value, reference in zip(
            compute_MOs_value_grad_lap(actual, points), compute_MOs_value_grad_lap(expected, points), strict=True
        ):
            assert np.isfinite(value).all()
            np.testing.assert_allclose(value, reference, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        ("trexio_file", "is_complex"),
        [
            pytest.param("lithium.h5", False, id="Li-periodic-Gamma"),
            pytest.param("lithium_k025.h5", True, id="Li-periodic-nonGamma"),
        ],
    )
    @pytest.mark.light
    def test_periodic_lithium_spherical_cartesian_roundtrip_preserves_mos(
        self,
        trexio_file: str,
        is_complex: bool,
    ):
        """A d-shell periodic spherical function survives sphe -> cart -> sphe.

        This is the lossless direction.  Unlike the periodic H2
        fixtures, Li contains a d shell and therefore exercises a genuinely
        rectangular spherical/Cartesian angular transformation at both Gamma
        and a non-Gamma twist.
        """
        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            trexio_file,
        )
        _, _, mos_sphe, _, _, _ = read_trexio_file(trexio_file=fixture, store_tuple=True)
        expected_sphe_type = AOs_sphe_pgto_complex_data if is_complex else AOs_sphe_pgto_data
        expected_cart_type = AOs_cart_pgto_complex_data if is_complex else AOs_cart_pgto_data
        assert isinstance(mos_sphe.aos_data, expected_sphe_type)
        assert max(mos_sphe.aos_data.angular_momentums) == 2

        mos_cart = mos_sphe.to_cartesian()
        mos_roundtrip = mos_cart.to_spherical()
        assert isinstance(mos_cart.aos_data, expected_cart_type)
        assert isinstance(mos_roundtrip.aos_data, expected_sphe_type)
        assert mos_sphe.aos_data.num_ao == 26
        assert mos_cart.aos_data.num_ao == 28
        assert mos_roundtrip.aos_data.k_point == mos_sphe.aos_data.k_point

        cell = np.asarray(mos_sphe.aos_data.structure_data.cell)
        r_carts = (
            np.asarray(
                [[0.13, 0.27, 0.41], [0.71, -0.19, 0.33], [-0.22, 0.58, 0.14]],
                dtype=np.float64,
            )
            @ cell
        )
        representations = (mos_sphe, mos_cart, mos_roundtrip)
        values = [np.asarray(compute_MOs(mos_data=mos, r_carts=r_carts)) for mos in representations]
        gradients = [tuple(np.asarray(x) for x in compute_MOs_grad(mos_data=mos, r_carts=r_carts)) for mos in representations]
        laplacians = [np.asarray(compute_MOs_laplacian(mos_data=mos, r_carts=r_carts)) for mos in representations]
        atol_value, rtol_value = get_tolerance_min(["ao_eval", "mo_eval"], "strict")
        atol_deriv, rtol_deriv = get_tolerance_min(["ao_grad_lap", "mo_grad", "mo_lap"], "strict")

        for representation, actual in zip(("cart", "roundtrip"), values[1:], strict=True):
            np.testing.assert_allclose(actual, values[0], atol=atol_value, rtol=rtol_value)
        for representation, actual_gradients in zip(("cart", "roundtrip"), gradients[1:], strict=True):
            for axis, actual, expected in zip(("x", "y", "z"), actual_gradients, gradients[0], strict=True):
                comparison_label = f"{representation}-{axis}"
                np.testing.assert_allclose(actual, expected, atol=atol_deriv, rtol=rtol_deriv)
        for representation, actual in zip(("cart", "roundtrip"), laplacians[1:], strict=True):
            np.testing.assert_allclose(actual, laplacians[0], atol=atol_deriv, rtol=rtol_deriv)

    @pytest.mark.parametrize(
        "trexio_file",
        [
            pytest.param("lithium_cart.h5", id="Li-periodic-Gamma"),
            pytest.param("lithium_k025_cart.h5", id="Li-periodic-nonGamma"),
        ],
    )
    @pytest.mark.light
    def test_periodic_lithium_cartesian_to_spherical_uses_global_s_metric(self, trexio_file: str):
        r"""Cartesian-native Li must use the global periodic ``S(k)`` projection.

        The oracle is assembled independently from the complete periodic AO
        overlap and the exact spherical embedding.  It deliberately includes
        cross-shell and cross-centre overlap blocks, so a regression to an
        isolated-shell metric cannot satisfy the normal equation.

        For row-vector MO coefficients and possibly complex ``S(k)``,

        ``G = conj(T) @ S @ T.T`` and
        ``R = solve(G, conj(T) @ S).T``.

        The final operation is a transpose, not an adjoint.
        """
        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            trexio_file,
        )
        _, aos_cart, _, _, _, _ = read_trexio_file(trexio_file=fixture, store_tuple=True)
        assert isinstance(aos_cart, AOs_cart_pgto_data)
        assert max(aos_cart.angular_momentums) == 2

        aos_sphe, actual_reverse = _aos_cart_to_sphe(aos_cart)
        rebuilt_cart, _ = _aos_sphe_to_cart(aos_sphe)
        assert rebuilt_cart.num_ao == aos_cart.num_ao == 28
        assert aos_sphe.num_ao == 26
        assert aos_sphe.k_point == aos_cart.k_point

        overlap = np.asarray(compute_overlap_matrix(aos_cart))
        molecular_cart = _detile_ragged_to_molecular(aos_cart)
        molecular_sphe = _detile_ragged_to_molecular(aos_sphe)
        embedding = _independent_native_cart_embedding(molecular_cart, molecular_sphe)
        embedding = np.asarray(embedding, dtype=np.result_type(overlap.dtype, np.float64))
        gram = embedding.conj() @ overlap @ embedding.T
        expected_reverse = np.linalg.solve(gram, embedding.conj() @ overlap).T
        identity_cart = np.eye(aos_cart.num_ao, dtype=expected_reverse.dtype)
        projector = np.asarray(actual_reverse) @ embedding
        residual_orthogonality = embedding.conj() @ overlap @ (identity_cart - projector).T
        atol, rtol = get_tolerance("det_eval", "strict")

        np.testing.assert_allclose(actual_reverse, expected_reverse, atol=atol, rtol=rtol)
        np.testing.assert_allclose(residual_orthogonality, np.zeros_like(residual_orthogonality), atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        ("trexio_file", "highest_l"),
        [
            pytest.param("N2_ecp_ccpvtz_cart.h5", 3, id="f-shell"),
            pytest.param("CuBr_ecp_ccpvtz_cart.h5", 4, id="g-shell"),
        ],
    )
    @pytest.mark.parametrize("r_carts", [((0.27, 0.41, 0.63), (0.91, -0.38, 1.29), (-0.43, 0.67, -0.91))])
    @pytest.mark.light
    def test_cartesian_native_high_l_shells_convert_to_spherical(self, trexio_file: str, highest_l: int, r_carts):
        """Native f/g shells preserve the global-S projected coefficients and orbital values.

        TREXIO folds ``ao_normalization`` into each Cartesian AO's contracted
        coefficients.  Components such as ``xxx``, ``xxy``, and ``xyz`` then
        have proportional, rather than identical, radial coefficient vectors.
        They still belong to one physical shell and must be converted together.
        At three explicit AO evaluation points, compare the spherical result
        with an independently assembled global-S projection in the native
        Cartesian basis. General Cartesian MOs can contain discarded trace
        components: equality with the unprojected input is not required.
        No many-body electron configuration or cond(G) is needed.
        """
        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            trexio_file,
        )
        _, _, mos_cart, _, _, _ = read_trexio_file(trexio_file=fixture, store_tuple=True)
        assert isinstance(mos_cart.aos_data, AOs_cart_gto_data)
        assert max(mos_cart.aos_data.angular_momentums) == highest_l

        mos_sphe = mos_cart.to_spherical()

        assert isinstance(mos_sphe.aos_data, AOs_sphe_gto_data)
        assert max(mos_sphe.aos_data.angular_momentums) == highest_l
        assert mos_sphe.aos_data.num_ao < mos_cart.aos_data.num_ao

        embedding = _independent_native_cart_embedding(mos_cart.aos_data, mos_sphe.aos_data)
        overlap = np.asarray(_compute_overlap_matrix_cart_analytic(mos_cart.aos_data))
        metric_rhs = embedding.conj() @ overlap
        reverse = np.linalg.solve(metric_rhs @ embedding.T, metric_rhs).T
        expected_coefficients = np.asarray(mos_cart.mo_coefficients) @ reverse
        coefficient_atol, coefficient_rtol = get_tolerance("det_eval", "strict")
        np.testing.assert_allclose(
            mos_sphe.mo_coefficients, expected_coefficients, atol=coefficient_atol, rtol=coefficient_rtol, equal_nan=False
        )

        r_carts = np.asarray(r_carts)
        cart_values = np.asarray(compute_AOs(mos_cart.aos_data, r_carts))
        expected_values = (expected_coefficients @ embedding) @ cart_values
        actual_values = np.asarray(compute_MOs(mos_sphe, r_carts))
        value_atol, value_rtol = get_tolerance_min(("ao_eval", "mo_eval"), "strict")
        np.testing.assert_allclose(actual_values, expected_values, atol=value_atol, rtol=value_rtol, equal_nan=False)

        high_l = np.asarray(mos_sphe.aos_data.angular_momentums) == highest_l
        high_l_values = (expected_coefficients[:, high_l] @ embedding[high_l]) @ cart_values
        assert np.max(np.abs(high_l_values)) > value_atol + value_rtol * np.max(np.abs(expected_values))

    @pytest.mark.parametrize(
        "trexio_file",
        [
            pytest.param("N2_ecp_ccpvtz_cart.h5", id="f-shell"),
            pytest.param("CuBr_ecp_ccpvtz_cart.h5", id="g-shell"),
        ],
    )
    @pytest.mark.light
    def test_cart_to_spherical_projector_is_global_s_metric(self, trexio_file: str):
        """Independently verify the molecular global-S normal equation."""
        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            trexio_file,
        )
        _, aos_cart, _, _, _, _ = read_trexio_file(trexio_file=fixture, store_tuple=True)
        aos_sphe, cart_to_sphe = _aos_cart_to_sphe(aos_cart)
        overlap = np.asarray(_compute_overlap_matrix_cart_analytic(aos_cart), dtype=np.float64)
        embedding = _independent_native_cart_embedding(aos_cart, aos_sphe)
        metric_rhs = embedding.conj() @ overlap
        gram = metric_rhs @ embedding.T
        expected_reverse = np.linalg.solve(gram, metric_rhs).T
        projector = cart_to_sphe @ embedding
        identity_cart = np.eye(aos_cart.num_ao)
        identity_sphe = np.eye(aos_sphe.num_ao)
        atol, rtol = get_tolerance("det_eval", "strict")

        np.testing.assert_allclose(cart_to_sphe, expected_reverse, atol=atol, rtol=rtol)
        np.testing.assert_allclose(embedding @ cart_to_sphe, identity_sphe, atol=atol, rtol=rtol)
        np.testing.assert_allclose(projector @ projector, projector, atol=atol, rtol=rtol)
        residual_orthogonality = embedding.conj() @ overlap @ (identity_cart - projector).T
        np.testing.assert_allclose(residual_orthogonality, np.zeros_like(residual_orthogonality), atol=atol, rtol=rtol)

        # Mutation guard: the old Euclidean pseudoinverse is idempotent too,
        # but it does not make the discarded complete function S-orthogonal.
        euclidean_projector = np.linalg.pinv(embedding) @ embedding
        assert np.linalg.norm(embedding @ overlap @ (identity_cart - euclidean_projector).T) > 100 * atol

    @pytest.mark.light
    def test_cart_to_spherical_projection_is_invariant_to_cartesian_ao_normalization(self):
        """Changing Cartesian component conventions must not change the projected MO."""
        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            "N2_ecp_ccpvtz_cart.h5",
        )
        _, aos_cart, mos_cart, _, _, _ = read_trexio_file(trexio_file=fixture, store_tuple=True)
        component_scales = np.linspace(0.7, 1.3, aos_cart.num_ao, dtype=np.float64)
        orbital = np.asarray(aos_cart.orbital_indices, dtype=np.int64)
        aos_rescaled = AOs_cart_gto_data._from_full_basis(
            structure_data=aos_cart.structure_data,
            nucleus_index=aos_cart.nucleus_index,
            num_ao=aos_cart.num_ao,
            num_ao_prim=aos_cart.num_ao_prim,
            orbital_indices=aos_cart.orbital_indices,
            exponents=np.asarray(aos_cart.exponents),
            coefficients=np.asarray(aos_cart.coefficients) * component_scales[orbital],
            angular_momentums=aos_cart.angular_momentums,
            polynominal_order_x=aos_cart.polynominal_order_x,
            polynominal_order_y=aos_cart.polynominal_order_y,
            polynominal_order_z=aos_cart.polynominal_order_z,
        )
        mos_rescaled = MOs_data(
            num_mo=mos_cart.num_mo,
            aos_data=aos_rescaled,
            mo_coefficients=np.asarray(mos_cart.mo_coefficients) / component_scales[None, :],
        )

        projected = mos_cart.to_spherical()
        projected_rescaled = mos_rescaled.to_spherical()
        r_carts = np.asarray(
            [[0.13, -0.21, 0.34], [0.72, 0.18, -0.43], [-0.51, 0.62, 0.27]],
            dtype=np.float64,
        )
        values = np.asarray(compute_MOs(mos_data=projected, r_carts=r_carts))
        values_rescaled = np.asarray(compute_MOs(mos_data=projected_rescaled, r_carts=r_carts))
        gradients = tuple(np.asarray(x) for x in compute_MOs_grad(mos_data=projected, r_carts=r_carts))
        gradients_rescaled = tuple(np.asarray(x) for x in compute_MOs_grad(mos_data=projected_rescaled, r_carts=r_carts))
        laplacian = np.asarray(compute_MOs_laplacian(mos_data=projected, r_carts=r_carts))
        laplacian_rescaled = np.asarray(compute_MOs_laplacian(mos_data=projected_rescaled, r_carts=r_carts))
        atol_value, rtol_value = get_tolerance_min(["ao_eval", "mo_eval"], "strict")
        atol_deriv, rtol_deriv = get_tolerance_min(["ao_grad_lap", "mo_grad", "mo_lap"], "strict")
        np.testing.assert_allclose(values_rescaled, values, atol=atol_value, rtol=rtol_value)
        for actual, expected in zip(gradients_rescaled, gradients, strict=True):
            np.testing.assert_allclose(actual, expected, atol=atol_deriv, rtol=rtol_deriv)
        np.testing.assert_allclose(laplacian_rescaled, laplacian, atol=atol_deriv, rtol=rtol_deriv)

    @pytest.mark.light
    def test_complex_bloch_sphe_cart_roundtrip_preserves_twist_and_mos(self):
        """Complex multi-row MOs retain values, derivatives, and Bloch phase."""
        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            "lithium_k025.h5",
        )
        _, _, mos_sphe, _, _, _ = read_trexio_file(
            trexio_file=fixture,
            store_tuple=True,
        )
        assert isinstance(mos_sphe.aos_data, AOs_sphe_pgto_complex_data)
        base = np.asarray(mos_sphe.mo_coefficients[0], dtype=np.complex128)
        coefficients = np.stack(
            (
                base * (1.0 + 0.25j),
                np.roll(base, 1) * (-0.4 + 0.8j),
                np.roll(base, 3) * (0.6 - 0.35j),
            )
        )
        mos_sphe = MOs_data.build(aos_data=mos_sphe.aos_data, mo_coefficients=coefficients)
        k_point = mos_sphe.aos_data.k_point

        mos_cart = mos_sphe.to_cartesian()
        mos_roundtrip = mos_cart.to_spherical()
        assert isinstance(mos_cart.aos_data, AOs_cart_pgto_complex_data)
        assert isinstance(mos_roundtrip.aos_data, AOs_sphe_pgto_complex_data)
        assert mos_cart.aos_data.k_point == k_point
        assert mos_roundtrip.aos_data.k_point == k_point

        cell = np.asarray(mos_sphe.aos_data.structure_data.cell)
        r_carts = (
            np.array(
                [
                    [0.13, 0.27, 0.41],
                    [1.07, -0.19, 0.33],
                    [-0.22, 0.58, 1.14],
                ],
                dtype=np.float64,
            )
            @ cell
        )
        representations = (mos_sphe, mos_cart, mos_roundtrip)
        values = [np.asarray(compute_MOs(mos_data=mos, r_carts=r_carts)) for mos in representations]
        gradients = [tuple(np.asarray(x) for x in compute_MOs_grad(mos_data=mos, r_carts=r_carts)) for mos in representations]
        laplacians = [np.asarray(compute_MOs_laplacian(mos_data=mos, r_carts=r_carts)) for mos in representations]

        atol_value, rtol_value = get_tolerance_min(["ao_eval", "mo_eval"], "strict")
        atol_deriv, rtol_deriv = get_tolerance_min(["ao_grad_lap", "mo_grad", "mo_lap"], "strict")
        for representation, actual in zip(("cart", "roundtrip"), values[1:], strict=True):
            np.testing.assert_allclose(actual, values[0], atol=atol_value, rtol=rtol_value)
        for representation, actual_gradients in zip(("cart", "roundtrip"), gradients[1:], strict=True):
            for axis, actual, expected in zip(("x", "y", "z"), actual_gradients, gradients[0], strict=True):
                comparison_label = f"{representation}-{axis}"
                np.testing.assert_allclose(actual, expected, atol=atol_deriv, rtol=rtol_deriv)
        for representation, actual in zip(("cart", "roundtrip"), laplacians[1:], strict=True):
            np.testing.assert_allclose(actual, laplacians[0], atol=atol_deriv, rtol=rtol_deriv)

        # Independent Bloch oracle: translating every electron by the first
        # lattice vector multiplies value, gradient, and Laplacian by
        # exp(2*pi*i*k_0).  Do not call the leaf's private phase helper here.
        translated = r_carts + cell[0]
        expected_phase = np.exp(2j * np.pi * float(k_point[0]))
        for representation, mos, value, gradient, laplacian in zip(
            ("sphe", "cart", "roundtrip"), representations, values, gradients, laplacians, strict=True
        ):
            translated_value = np.asarray(compute_MOs(mos_data=mos, r_carts=translated))
            translated_gradient = tuple(np.asarray(x) for x in compute_MOs_grad(mos_data=mos, r_carts=translated))
            translated_laplacian = np.asarray(compute_MOs_laplacian(mos_data=mos, r_carts=translated))
            np.testing.assert_allclose(
                translated_value,
                expected_phase * value,
                atol=atol_value,
                rtol=rtol_value,
            )
            for axis, actual, expected in zip(("x", "y", "z"), translated_gradient, gradient, strict=True):
                comparison_label = f"{representation}-{axis}"
                np.testing.assert_allclose(
                    actual,
                    expected_phase * expected,
                    atol=atol_deriv,
                    rtol=rtol_deriv,
                )
            np.testing.assert_allclose(
                translated_laplacian,
                expected_phase * laplacian,
                atol=atol_deriv,
                rtol=rtol_deriv,
            )

    @pytest.mark.light
    def test_MOs_sphe_to_cart(self):
        """Ensure spherical -> Cartesian conversion preserves MO values up to l=6."""
        rng = np.random.default_rng(0)

        nucleus_index = []
        orbital_indices = []
        exponents = []
        coefficients = []
        angular_momentums = []
        magnetic_quantum_numbers = []

        ao_idx = 0
        for l in range(7):  # l = 0..6
            for m in range(-l, l + 1):
                nucleus_index.append(0)
                orbital_indices.append(ao_idx)
                exponents.append(float(l + 1))
                coefficients.append(1.0)
                angular_momentums.append(l)
                magnetic_quantum_numbers.append(m)
                ao_idx += 1

        num_ao = ao_idx
        num_ao_prim = len(orbital_indices)
        num_mo = 6
        num_el = 12

        r_carts = rng.standard_normal((num_el, 3))
        R_carts = np.array([[0.1, -0.2, 0.3]])

        mo_coefficients = rng.standard_normal((num_mo, num_ao))

        structure_data = Structure_data(
            pbc_flag=False,
            positions=R_carts,
            atomic_numbers=(1,),
            element_symbols=("X",),
            atomic_labels=("X",),
        )

        aos_data = AOs_sphe_gto_data._from_full_basis(
            structure_data=structure_data,
            nucleus_index=tuple(nucleus_index),
            num_ao=num_ao,
            num_ao_prim=num_ao_prim,
            orbital_indices=tuple(orbital_indices),
            exponents=np.array(exponents, dtype=np.float64),
            coefficients=np.array(coefficients, dtype=np.float64),
            angular_momentums=tuple(angular_momentums),
            magnetic_quantum_numbers=tuple(magnetic_quantum_numbers),
        )

        mos_sphe = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)
        mos_sphe.sanity_check()

        mos_cart = mos_sphe.to_cartesian()

        assert isinstance(mos_cart.aos_data, AOs_cart_gto_data)

        mo_sphe = compute_MOs(mos_data=mos_sphe, r_carts=r_carts)
        mo_cart = compute_MOs(mos_data=mos_cart, r_carts=r_carts)

        # Path crosses ao_eval (fp32 in mixed) -> mo_eval; use min for value cmp.
        atol, rtol = get_tolerance_min(["ao_eval", "mo_eval"], "strict")
        assert not np.any(np.isnan(np.asarray(mo_cart))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mo_sphe))), "NaN detected in second argument"
        np.testing.assert_allclose(mo_cart, mo_sphe, atol=atol, rtol=rtol)

        grad_sphe = compute_MOs_grad(mos_data=mos_sphe, r_carts=r_carts)
        grad_cart = compute_MOs_grad(mos_data=mos_cart, r_carts=r_carts)

        # grad path crosses ao_grad_lap (fp64) -> mo_grad.
        atol_g, rtol_g = get_tolerance_min(["ao_grad_lap", "mo_grad"], "strict")
        for g_cart, g_sphe in zip(grad_cart, grad_sphe, strict=True):
            assert not np.any(np.isnan(np.asarray(g_cart))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(g_sphe))), "NaN detected in second argument"
            np.testing.assert_allclose(g_cart, g_sphe, atol=atol_g, rtol=rtol_g)

        lap_sphe = compute_MOs_laplacian(mos_data=mos_sphe, r_carts=r_carts)
        lap_cart = compute_MOs_laplacian(mos_data=mos_cart, r_carts=r_carts)

        # lap path crosses ao_grad_lap (fp64) -> mo_lap (fp64).
        atol_l, rtol_l = get_tolerance_min(["ao_grad_lap", "mo_lap"], "strict")
        assert not np.any(np.isnan(np.asarray(lap_cart))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(lap_sphe))), "NaN detected in second argument"
        np.testing.assert_allclose(lap_cart, lap_sphe, atol=atol_l, rtol=rtol_l)

        jax.clear_caches()

    @pytest.mark.light
    def test_MOs_cart_to_sphe(self):
        """Ensure Cartesian -> spherical conversion preserves MO values up to l=6."""
        rng = np.random.default_rng(1)

        nucleus_index = []
        orbital_indices = []
        exponents = []
        coefficients = []
        angular_momentums = []
        polynominal_order_x = []
        polynominal_order_y = []
        polynominal_order_z = []

        shell_cart_indices: list[list[int]] = []
        ao_idx = 0
        for l in range(7):  # l = 0..6
            shell_indices: list[int] = []
            for nx, ny, nz in [(_nx, _ny, l - _nx - _ny) for _nx in range(l, -1, -1) for _ny in range(l - _nx, -1, -1)]:
                nucleus_index.append(0)
                orbital_indices.append(ao_idx)
                exponents.append(float(l + 1))
                coefficients.append(1.0)
                angular_momentums.append(l)
                polynominal_order_x.append(nx)
                polynominal_order_y.append(ny)
                polynominal_order_z.append(nz)
                shell_indices.append(ao_idx)
                ao_idx += 1
            shell_cart_indices.append(shell_indices)

        num_ao = ao_idx
        num_ao_prim = len(orbital_indices)
        num_mo = 6
        num_el = 12

        r_carts = rng.standard_normal((num_el, 3))
        R_carts = np.array([[0.1, -0.2, 0.3]])

        # Seed Cartesian coefficients from random spherical blocks to ensure pure-l content
        mo_coefficients = np.zeros((num_mo, num_ao))
        for l, cart_indices in enumerate(shell_cart_indices):
            transform = _cart_to_spherical_matrix(l)
            sph_block = rng.standard_normal((num_mo, 2 * l + 1))
            cart_block = sph_block @ transform.T
            mo_coefficients[:, cart_indices] = cart_block

        structure_data = Structure_data(
            pbc_flag=False,
            positions=R_carts,
            atomic_numbers=(1,),
            element_symbols=("X",),
            atomic_labels=("X",),
        )

        aos_data = AOs_cart_gto_data._from_full_basis(
            structure_data=structure_data,
            nucleus_index=tuple(nucleus_index),
            num_ao=num_ao,
            num_ao_prim=num_ao_prim,
            orbital_indices=tuple(orbital_indices),
            exponents=np.array(exponents, dtype=np.float64),
            coefficients=np.array(coefficients, dtype=np.float64),
            angular_momentums=tuple(angular_momentums),
            polynominal_order_x=tuple(polynominal_order_x),
            polynominal_order_y=tuple(polynominal_order_y),
            polynominal_order_z=tuple(polynominal_order_z),
        )

        mos_cart = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)
        mos_cart.sanity_check()

        mos_sphe = mos_cart.to_spherical()

        assert isinstance(mos_sphe.aos_data, AOs_sphe_gto_data)

        mo_cart = compute_MOs(mos_data=mos_cart, r_carts=r_carts)
        mo_sphe = compute_MOs(mos_data=mos_sphe, r_carts=r_carts)

        # Path crosses ao_eval (fp32 in mixed) -> mo_eval; use min for value cmp.
        atol, rtol = get_tolerance_min(["ao_eval", "mo_eval"], "strict")
        assert not np.any(np.isnan(np.asarray(mo_sphe))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mo_cart))), "NaN detected in second argument"
        np.testing.assert_allclose(mo_sphe, mo_cart, atol=atol, rtol=rtol)

        grad_cart = compute_MOs_grad(mos_data=mos_cart, r_carts=r_carts)
        grad_sphe = compute_MOs_grad(mos_data=mos_sphe, r_carts=r_carts)

        # grad path crosses ao_grad_lap (fp64) -> mo_grad.
        atol_g, rtol_g = get_tolerance_min(["ao_grad_lap", "mo_grad"], "strict")
        for g_cart, g_sphe in zip(grad_cart, grad_sphe, strict=True):
            assert not np.any(np.isnan(np.asarray(g_sphe))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(g_cart))), "NaN detected in second argument"
            np.testing.assert_allclose(g_sphe, g_cart, atol=atol_g, rtol=rtol_g)

        lap_cart = compute_MOs_laplacian(mos_data=mos_cart, r_carts=r_carts)
        lap_sphe = compute_MOs_laplacian(mos_data=mos_sphe, r_carts=r_carts)

        # lap path crosses ao_grad_lap (fp64) -> mo_lap (fp64).
        atol_l, rtol_l = get_tolerance_min(["ao_grad_lap", "mo_lap"], "strict")
        assert not np.any(np.isnan(np.asarray(lap_sphe))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(lap_cart))), "NaN detected in second argument"
        np.testing.assert_allclose(lap_sphe, lap_cart, atol=atol_l, rtol=rtol_l)

        jax.clear_caches()


class TestMOsFusedAPI:
    @pytest.mark.parametrize(
        "trexio_file",
        [
            "water_ccecp_ccpvqz.h5",  # spherical
            "H2_ae_ccpvdz_cart.h5",  # Cartesian
            "N_ae_ccpvdz_cart.h5",
        ],
    )
    @pytest.mark.light
    def test_MOs_value_grad_lap(self, trexio_file: str):
        """Fused ``compute_MOs_value_grad_lap`` matches the standalone APIs.

        All outputs (val/grad/lap) are bounded by ULP-level differences in
        the ``mo_eval`` / ``ao_eval`` zones: the fused MO function applies
        the same ``mo_coefficients @ ao_*`` matmul as the standalone kernels,
        but XLA may reassociate the upstream FMA chains across the different
        ``@jit`` boundaries.
        """
        parsed = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )
        _structure, _aos, mos_data_up, *_rest = parsed

        rng = np.random.default_rng(20260430)
        r_carts = (rng.standard_normal((6, 3)) * 1.5).astype(np.float64)

        val_f, gx_f, gy_f, gz_f, lap_f = compute_MOs_value_grad_lap(mos_data=mos_data_up, r_carts=r_carts)
        val_s = compute_MOs(mos_data=mos_data_up, r_carts=r_carts)
        gx_s, gy_s, gz_s = compute_MOs_grad(mos_data=mos_data_up, r_carts=r_carts)
        lap_s = compute_MOs_laplacian(mos_data=mos_data_up, r_carts=r_carts)

        for arr in (val_f, gx_f, gy_f, gz_f, lap_f, val_s, gx_s, gy_s, gz_s, lap_s):
            assert not np.any(np.isnan(np.asarray(arr)))

        # All outputs share the mo_eval/ao_eval zone tolerance; XLA may
        # reassociate FMA chains across @jit boundaries producing strictly
        # ULP-level differences.
        atol_val, rtol_val = get_tolerance_min(["ao_eval", "mo_eval"], "strict")
        np.testing.assert_allclose(np.asarray(gx_f), np.asarray(gx_s), atol=atol_val, rtol=rtol_val)
        np.testing.assert_allclose(np.asarray(gy_f), np.asarray(gy_s), atol=atol_val, rtol=rtol_val)
        np.testing.assert_allclose(np.asarray(gz_f), np.asarray(gz_s), atol=atol_val, rtol=rtol_val)
        np.testing.assert_allclose(np.asarray(lap_f), np.asarray(lap_s), atol=atol_val, rtol=rtol_val)
        np.testing.assert_allclose(np.asarray(val_f), np.asarray(val_s), atol=atol_val, rtol=rtol_val)

        jax.clear_caches()

    @pytest.mark.parametrize(
        "trexio_file",
        [
            "water_ccecp_ccpvqz.h5",
            "H2_ae_ccpvdz_cart.h5",
        ],
    )
    @pytest.mark.light
    def test_fused_MOs_dtypes_match_zones(self, trexio_file: str):
        """``compute_MOs_value_grad_lap`` outputs are pinned to their zones.

        val <-> ``mo_eval`` (fp32 mixed / fp64 full); gx/gy/gz <-> ``mo_grad``
        (fp64); lap <-> ``mo_lap`` (fp64).
        """
        parsed = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )
        _structure, _aos, mos_data_up, *_rest = parsed

        rng = np.random.default_rng(20260430)
        r_carts = (rng.standard_normal((4, 3)) * 1.5).astype(np.float64)
        val_f, gx_f, gy_f, gz_f, lap_f = compute_MOs_value_grad_lap(mos_data=mos_data_up, r_carts=r_carts)

        eval_dtype = get_dtype_jnp("mo_eval")
        grad_dtype = get_dtype_jnp("mo_grad")
        lap_dtype = get_dtype_jnp("mo_lap")
        assert val_f.dtype == eval_dtype, f"val.dtype = {val_f.dtype}, expected {eval_dtype}"
        assert gx_f.dtype == grad_dtype, f"gx.dtype = {gx_f.dtype}, expected {grad_dtype}"
        assert gy_f.dtype == grad_dtype, f"gy.dtype = {gy_f.dtype}, expected {grad_dtype}"
        assert gz_f.dtype == grad_dtype, f"gz.dtype = {gz_f.dtype}, expected {grad_dtype}"
        assert lap_f.dtype == lap_dtype, f"lap.dtype = {lap_f.dtype}, expected {lap_dtype}"

        jax.clear_caches()


# ---------------------------------------------------------------------------
# Periodic MO instances on top of PW / B-spline / pGTO / composite leaves.
#
# compute_MOs / compute_MOs_grad / compute_MOs_laplacian and their debug /
# autodiff siblings all route through compute_AOs / _compute_AOs_debug, so the
# MO layer is leaf-agnostic. Building tiny dummy periodic leaves here (mirroring
# the molecule-side dummy GTO instances) lets every MO cross-check run on the
# periodic PW / B-spline / pGTO / pGTO+PW / pGTO+B-spline bases without needing
# any external TREXIO / QE file.
# ---------------------------------------------------------------------------

_PMO_A = 6.0  # cubic cell edge (Bohr)
_PMO_BSPLINE_NS = (12, 12, 12)
_PMO_BSPLINE_ORDER = 5


def _pmo_cubic_structure(a: float = _PMO_A) -> Structure_data:
    """Single-atom cubic periodic cell shared by all periodic MO leaves."""
    cell = np.eye(3, dtype=np.float64) * a
    return Structure_data.build(
        pbc_flag=True,
        positions=np.zeros((1, 3), dtype=np.float64),
        vec_a=tuple(cell[0]),
        vec_b=tuple(cell[1]),
        vec_c=tuple(cell[2]),
        atomic_numbers=(1,),
        element_symbols=("H",),
        atomic_labels=("H1",),
    )


def _pmo_pw_leaf(structure: Structure_data):
    """Real cos/sin plane-wave leaf: constant + 5 (cos, sin) pairs = 11 AOs."""
    recip = structure.recip_cell
    millers = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -1, 1]],
        dtype=np.float64,
    )
    g_list: list[np.ndarray] = []
    is_sin: list[float] = []
    for m in millers:
        g = m @ recip
        if np.all(m == 0):
            g_list.append(g)
            is_sin.append(0.0)
        else:
            g_list.append(g)
            is_sin.append(0.0)
            g_list.append(g)
            is_sin.append(1.0)
    g_vectors = np.asarray(g_list, dtype=np.float64)
    phase_is_sin = np.asarray(is_sin, dtype=np.float64)
    return AOs_plane_wave_data.build(structure, g_vectors, phase_is_sin)


def _pmo_bspline_band_functions(frac):
    """Two smooth periodic bands of fractional coords, shape ``(n_band, ...)``."""
    x, y, z = frac[..., 0], frac[..., 1], frac[..., 2]
    tp = 2.0 * np.pi
    b0 = np.sin(tp * x) + 0.5 * np.cos(tp * y) + 0.25 * np.sin(tp * z)
    b1 = np.cos(tp * (x + y)) + 0.3 * np.sin(tp * z)
    return np.stack([b0, b1], axis=0)


def _pmo_bspline_leaf(structure: Structure_data, order: int = _PMO_BSPLINE_ORDER):
    """Interpolating B-spline leaf reproducing two smooth periodic bands."""
    ns = _PMO_BSPLINE_NS
    grids = [np.arange(n) / n for n in ns]
    fx, fy, fz = np.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
    frac = np.stack([fx, fy, fz], axis=-1)
    vals = _pmo_bspline_band_functions(frac)
    coeffs_cube = bspline_prefilter_periodic(vals.astype(np.complex128), order)
    coeffs_flat = coeffs_cube.reshape(coeffs_cube.shape[0], -1)
    return AOs_bspline_data.build(structure, coeffs_flat, ns, order)


def _pmo_pgto_leaf(structure: Structure_data):
    """Image-expanded periodic pGTO leaf: a single moderate-exponent s shell."""
    base = AOs_sphe_gto_data._from_full_basis(
        structure_data=structure,
        nucleus_index=(0,),
        num_ao=1,
        num_ao_prim=1,
        orbital_indices=(0,),
        exponents=np.array([0.8], dtype=np.float64),
        coefficients=np.array([1.0], dtype=np.float64),
        angular_momentums=(0,),
        magnetic_quantum_numbers=(0,),
    )
    return _apply_image_expansion(base, structure, AO_IMAGE_TRUNCATION_ATOL)


def _periodic_mos_setup(kind: str, seed: int):
    """Build ``(mos_data, r_carts)`` for one periodic leaf / composite kind."""
    structure = _pmo_cubic_structure()
    if kind == "pw":
        aos_data = _pmo_pw_leaf(structure)
    elif kind == "bspline":
        aos_data = _pmo_bspline_leaf(structure)
    elif kind == "pgto":
        aos_data = _pmo_pgto_leaf(structure)
    elif kind == "pgto+pw":
        aos_data = _pmo_pgto_leaf(structure) + _pmo_pw_leaf(structure)
    elif kind == "pgto+bspline":
        aos_data = _pmo_pgto_leaf(structure) + _pmo_bspline_leaf(structure)
    else:
        raise ValueError(f"unknown periodic MO kind: {kind}")
    aos_data.sanity_check()

    num_ao = aos_data.num_ao
    num_mo = min(3, num_ao)
    rng = np.random.default_rng(seed)
    mo_coefficients = rng.standard_normal((num_mo, num_ao)).astype(np.float64)
    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)
    mos_data.sanity_check()

    a = _PMO_A
    r_carts = rng.uniform(0.5, a - 0.5, size=(5, 3)).astype(np.float64)
    return mos_data, r_carts


_PERIODIC_MO_KINDS = ["pw", "bspline", "pgto", "pgto+pw", "pgto+bspline"]


class TestMOsPeriodic:
    @pytest.mark.light
    @pytest.mark.parametrize("kind", _PERIODIC_MO_KINDS)
    def test_MOs_periodic_forward_public_vs_debug(self, kind: str):
        """compute_MOs matches the numpy MO oracle and ``mo_coeff @ compute_AOs``."""
        mos_data, r_carts = _periodic_mos_setup(kind, seed=101)

        mo_public = np.asarray(compute_MOs(mos_data=mos_data, r_carts=r_carts))
        mo_debug = np.asarray(_compute_MOs_debug(mos_data=mos_data, r_carts=r_carts))
        mo_from_aos = np.asarray(np.dot(mos_data.mo_coefficients, compute_AOs(aos_data=mos_data.aos_data, r_carts=r_carts)))

        # The MO contraction is fp64, but its AO input is intentionally fp32 in
        # mixed mode.  The NumPy debug oracle stays fp64, so this comparison is
        # bounded by the weakest producer zone rather than by mo_eval alone.
        atol, rtol = get_tolerance_min(("ao_eval", "mo_eval"), "strict")
        assert not np.any(np.isnan(mo_public)), f"NaN in compute_MOs ({kind})"
        assert not np.any(np.isnan(mo_debug)), f"NaN in MO oracle ({kind})"
        np.testing.assert_allclose(mo_public, mo_debug, atol=atol, rtol=rtol)
        np.testing.assert_allclose(mo_public, mo_from_aos, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("kind", _PERIODIC_MO_KINDS)
    def test_MOs_periodic_grad_analytic_vs_auto(self, kind: str):
        """Check full analytical/AD correctness and mixed/full periodic-gradient fidelity."""

        def case_factory():
            return _periodic_mos_setup(kind, seed=202)

        def analytical(case):
            mos_data, r_carts = case
            return compute_MOs_grad(mos_data=mos_data, r_carts=r_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
            for mixed_block, full_block in blocks:
                _assert_orbital_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_orbital_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
            jax.clear_caches()
            return

        mos_data, r_carts = case_factory()
        gx_a, gy_a, gz_a = compute_MOs_grad(mos_data=mos_data, r_carts=r_carts)
        gx_u, gy_u, gz_u = _compute_MOs_grad_autodiff(mos_data=mos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "mo_grad"], "strict")
        for name, an, au in [("gx", gx_a, gx_u), ("gy", gy_a, gy_u), ("gz", gz_a, gz_u)]:
            assert not np.any(np.isnan(np.asarray(an))), f"NaN in analytic {name} ({kind})"
            assert not np.any(np.isnan(np.asarray(au))), f"NaN in autodiff {name} ({kind})"
            np.testing.assert_allclose(an, au, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-MO-DERIVATIVE-CONSISTENCY-001",
    )
    @pytest.mark.parametrize("kind", _PERIODIC_MO_KINDS)
    def test_MOs_periodic_grad_auto_vs_numerical(self, kind: str):
        """``_compute_MOs_grad_autodiff`` matches the central-FD MO oracle."""
        mos_data, r_carts = _periodic_mos_setup(kind, seed=303)
        gx_u, gy_u, gz_u = _compute_MOs_grad_autodiff(mos_data=mos_data, r_carts=r_carts)
        gx_n, gy_n, gz_n = _compute_MOs_grad_debug(mos_data=mos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "mo_grad"], "strict")
        for name, au, fd in [("gx", gx_u, gx_n), ("gy", gy_u, gy_n), ("gz", gz_u, gz_n)]:
            assert not np.any(np.isnan(np.asarray(au))), f"NaN in autodiff {name} ({kind})"
            assert not np.any(np.isnan(np.asarray(fd))), f"NaN in FD {name} ({kind})"
            np.testing.assert_allclose(au, fd, atol=atol, rtol=rtol)
        record_fd_case(
            "fd-ad-mo-grad-periodic",
            pytest_node_id="tests/test_jqmc_kernel/test_molecular_orbital.py::TestMOsPeriodic::test_MOs_periodic_grad_auto_vs_numerical",
            fixture_parameters=fd_case_parameters("fd-ad-mo-grad-periodic", kind=kind),
            coordinates=(r_carts, np.empty((0, 3))),
            coordinate_generator="tests.test_jqmc_kernel.test_molecular_orbital._periodic_mos_setup",
            coordinate_seed=303,
            compared_pairs=((gx_u, gx_n), (gy_u, gy_n), (gz_u, gz_n)),
            atol=atol,
            rtol=rtol,
            zones=("ao_eval", "mo_grad"),
            level="strict",
            stencil="central-first-order-2",
            step=1.0e-5,
            direction="cartesian-basis",
        )
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("kind", _PERIODIC_MO_KINDS)
    def test_MOs_periodic_laplacian_analytic_vs_auto(self, kind: str):
        """Check full analytical/AD correctness and mixed/full periodic-Laplacian fidelity."""

        def case_factory():
            return _periodic_mos_setup(kind, seed=404)

        def analytical(case):
            mos_data, r_carts = case
            return compute_MOs_laplacian(mos_data=mos_data, r_carts=r_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
            for mixed_block, full_block in blocks:
                _assert_orbital_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_orbital_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
            jax.clear_caches()
            return

        mos_data, r_carts = case_factory()
        lap_an = compute_MOs_laplacian(mos_data=mos_data, r_carts=r_carts)
        lap_auto = _compute_MOs_laplacian_autodiff(mos_data=mos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap", "mo_lap"], "strict")
        assert not np.any(np.isnan(np.asarray(lap_an))), f"NaN in analytic lap ({kind})"
        assert not np.any(np.isnan(np.asarray(lap_auto))), f"NaN in autodiff lap ({kind})"
        np.testing.assert_allclose(lap_an, lap_auto, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-MO-DERIVATIVE-CONSISTENCY-001",
    )
    @pytest.mark.parametrize("kind", _PERIODIC_MO_KINDS)
    def test_MOs_periodic_laplacian_auto_vs_numerical(self, kind: str):
        """``_compute_MOs_laplacian_autodiff`` matches the central-FD MO oracle."""
        mos_data, r_carts = _periodic_mos_setup(kind, seed=505)
        lap_auto = _compute_MOs_laplacian_autodiff(mos_data=mos_data, r_carts=r_carts)
        lap_fd = _compute_MOs_laplacian_debug(mos_data=mos_data, r_carts=r_carts)
        atol, rtol = get_tolerance_min(["ao_eval", "ao_grad_lap", "mo_lap"], "strict")
        assert not np.any(np.isnan(np.asarray(lap_auto))), f"NaN in autodiff lap ({kind})"
        assert not np.any(np.isnan(np.asarray(lap_fd))), f"NaN in FD lap ({kind})"
        np.testing.assert_allclose(lap_auto, lap_fd, atol=atol, rtol=rtol)
        record_fd_case(
            "fd-ad-mo-laplacian-periodic",
            pytest_node_id="tests/test_jqmc_kernel/test_molecular_orbital.py::TestMOsPeriodic::test_MOs_periodic_laplacian_auto_vs_numerical",
            fixture_parameters=fd_case_parameters("fd-ad-mo-laplacian-periodic", kind=kind),
            coordinates=(r_carts, np.empty((0, 3))),
            coordinate_generator="tests.test_jqmc_kernel.test_molecular_orbital._periodic_mos_setup",
            coordinate_seed=505,
            compared_pairs=((lap_auto, lap_fd),),
            atol=atol,
            rtol=rtol,
            zones=("ao_eval", "ao_grad_lap", "mo_lap"),
            level="strict",
            stencil="central-second-order-4",
            step=1.0e-3,
            direction="cartesian-basis",
        )
        jax.clear_caches()


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
