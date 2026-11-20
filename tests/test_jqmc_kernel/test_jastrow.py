"""Jastrow contracts: full AD correctness and mixed analytical/full fidelity.

Unrelated value, parity, finite-difference and invariant checks keep both modes.
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

import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_dtype_np, get_tolerance, get_tolerance_min
from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL
from jqmc.atomic_orbital import AOs_sphe_gto_data, _apply_image_expansion
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    _compute_grads_and_laplacian_Jastrow_one_body_auto,
    _compute_grads_and_laplacian_Jastrow_part_auto,
    _compute_grads_and_laplacian_Jastrow_three_body_auto,
    _compute_grads_and_laplacian_Jastrow_two_body_auto,
    _compute_Jastrow_one_body_molecular_debug,
    _compute_Jastrow_one_body_pbc_debug,
    _compute_Jastrow_three_body_debug,
    _compute_Jastrow_two_body_molecular_debug,
    _compute_Jastrow_two_body_pbc_debug,
    _compute_ratio_Jastrow_part_debug,
    _compute_ratio_Jastrow_part_rank1_update,
    _init_grads_laplacian_Jastrow_three_body_streaming_state,
    compute_grads_and_laplacian_Jastrow_one_body,
    compute_grads_and_laplacian_Jastrow_part,
    compute_grads_and_laplacian_Jastrow_three_body,
    compute_grads_and_laplacian_Jastrow_two_body,
    compute_Jastrow_one_body,
    compute_Jastrow_three_body,
    compute_Jastrow_two_body,
)
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file
from tests.test_jqmc_kernel.optimizer_metadata_helpers import make_test_block
from tests.conftest import CELL_FIXTURES, make_pbc_structure


from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block


def _assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Check a homogeneous electron-coordinate derivative block at its central budget."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


def _make_structure(R_carts, atomic_number=0):
    """Build a non-periodic Structure_data with dummy ``X`` atoms at *R_carts*."""
    n = len(R_carts)
    return Structure_data(
        pbc_flag=False,
        positions=R_carts,
        atomic_numbers=tuple([atomic_number] * n),
        element_symbols=tuple(["X"] * n),
        atomic_labels=tuple(["X"] * n),
    )


def _make_pbc_structure(R_carts, atomic_number=0):
    """Build a periodic Structure_data (cubic 10 Bohr) with dummy ``X`` atoms at *R_carts*."""
    n = len(R_carts)
    return make_pbc_structure(
        CELL_FIXTURES["cubic_10bohr"],
        positions=np.asarray(R_carts, dtype=np.float64),
        atomic_numbers=tuple([atomic_number] * n),
        element_symbols=tuple(["X"] * n),
        atomic_labels=tuple(["X"] * n),
    )


def _make_shell_aos(rng):
    """Build spherical AOs with one shell per ``l`` (l=0..6) on a single atom at origin."""
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

    structure_data = _make_structure(np.zeros((1, 3), dtype=np.float64), atomic_number=1)

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


# ---------------------------------------------------------------------------
# Same-precision J1 / J2 value references.
#
# The declared ``_compute_Jastrow_{one,two}_body_{molecular,pbc}_debug`` twins
# are fp64 by construction, while production evaluates J1 / J2 in the
# ``jastrow_eval`` zone (fp32 in mixed): the ``r - R`` / ``r_i - r_j``
# difference -- and, under PBC, the smooth periodic distance of that
# difference -- is formed in the caller's fp64 and cast to the zone, and the
# radial law and the accumulation then run in the zone dtype. The references
# below are literal transcriptions of those debug twins in which every step
# after the fp64 geometry is evaluated in ``dtype``. Production judgement uses
# ``dtype = get_dtype_np("jastrow_eval")``, so mixed is judged in its own
# precision (never against an fp64 reference). With ``dtype=np.float64``
# each transcription reproduces its debug twin bit for bit; the tests assert
# that, which keeps the declared oracles in use and shows that the full-mode
# reference is the fp64 debug reference.
# ---------------------------------------------------------------------------
def _periodic_distance_fp64(diff, cell, inv_cell):
    """fp64 smooth periodic distance of an fp64 difference (same operations as the debug twins)."""
    r_frac = np.asarray(diff, dtype=np.float64) @ inv_cell
    phase = np.pi * r_frac
    vector = np.sin(2 * phase) @ cell
    height = 2 * np.sin(phase) ** 2
    square = np.dot(vector, vector) + np.dot(np.sum(cell * cell, axis=1), height * height)
    return float(np.sqrt(square) / (2 * np.pi))


def _j1_zone_reference(jastrow_one_body_data, r_up_carts, r_dn_carts, dtype):
    """J1 with the distance from fp64 geometry and the radial law / sum in ``dtype``.

    Transcribes ``_compute_Jastrow_one_body_molecular_debug`` (open systems)
    or ``_compute_Jastrow_one_body_pbc_debug`` (``pbc_flag``), including their
    summation order.
    """
    sd = jastrow_one_body_data.structure_data
    positions = np.asarray(sd.positions, dtype=np.float64)
    effective_charges = np.asarray(
        np.array(sd.atomic_numbers, dtype=np.float64) - np.array(jastrow_one_body_data.core_electrons, dtype=np.float64),
        dtype=dtype,
    )
    a = dtype(jastrow_one_body_data.jastrow_1b_param)
    j1b_type = jastrow_one_body_data.jastrow_1b_type
    if j1b_type not in ("exp", "pade"):
        raise ValueError(f"Unknown jastrow_1b_type: {j1b_type}")

    if sd.pbc_flag:
        cell = np.asarray(sd.cell, dtype=np.float64)
        inv_cell = np.linalg.inv(cell)
        J1 = dtype(0.0)
        for r_cart in np.concatenate([np.asarray(r_up_carts), np.asarray(r_dn_carts)], axis=0):
            for R_cart, Z_eff in zip(positions, effective_charges, strict=True):
                r_eN = dtype(_periodic_distance_fp64(np.asarray(r_cart, dtype=np.float64) - R_cart, cell, inv_cell))
                coeff = (dtype(2.0) * Z_eff) ** dtype(1.0 / 4.0)
                prefactor = -((dtype(2.0) * Z_eff) ** dtype(3.0 / 4.0))
                if j1b_type == "exp":
                    J1 += prefactor / (dtype(2.0) * a) * (dtype(1.0) - np.exp(-a * coeff * r_eN))
                else:
                    J1 += prefactor * coeff * r_eN / (dtype(2.0) * (dtype(1.0) + a * coeff * r_eN))
        return J1

    def kernel(coeff, r_cart, R_cart):
        r_eN = np.linalg.norm(np.asarray(np.asarray(r_cart) - R_cart, dtype=dtype))
        if j1b_type == "exp":
            return dtype(1.0) / (dtype(2.0) * a) * (dtype(1.0) - np.exp(-a * coeff * r_eN))
        return coeff * r_eN / (dtype(2.0) * (dtype(1.0) + a * coeff * r_eN))

    totals = []
    for r_carts in (r_up_carts, r_dn_carts):
        total = dtype(0.0)
        for r_cart in r_carts:
            for R_cart, Z_eff in zip(positions, effective_charges, strict=True):
                coeff = (dtype(2.0) * Z_eff) ** dtype(1.0 / 4.0)
                total += -((dtype(2.0) * Z_eff) ** dtype(3.0 / 4.0)) * kernel(coeff, r_cart, R_cart)
        totals.append(total)
    return totals[0] + totals[1]


def _j2_zone_reference(jastrow_two_body_data, r_up_carts, r_dn_carts, dtype):
    """J2 with the distance from fp64 geometry and the pair law / sum in ``dtype``.

    Transcribes ``_compute_Jastrow_two_body_molecular_debug`` (open systems)
    or ``_compute_Jastrow_two_body_pbc_debug`` (``pbc_flag``), including their
    summation order.
    """
    import itertools

    sd = jastrow_two_body_data.structure_data
    a = dtype(jastrow_two_body_data.jastrow_2b_param)
    j2b_type = jastrow_two_body_data.jastrow_2b_type
    if j2b_type not in ("exp", "pade"):
        raise ValueError(f"Unknown jastrow_2b_type: {j2b_type}")

    if sd.pbc_flag:
        cell = np.asarray(sd.cell, dtype=np.float64)
        inv_cell = np.linalg.inv(cell)

        def u(r_i, r_j):
            r_ij = dtype(
                _periodic_distance_fp64(np.asarray(r_i, dtype=np.float64) - np.asarray(r_j, dtype=np.float64), cell, inv_cell)
            )
            if j2b_type == "exp":
                return dtype(1.0) / (dtype(2.0) * a) * (dtype(1.0) - np.exp(-a * r_ij))
            return r_ij / (dtype(2.0) * (dtype(1.0) + a * r_ij))

        J2 = dtype(0.0)
        for r_i, r_j in itertools.product(r_up_carts, r_dn_carts):
            J2 += u(r_i, r_j)
        for r_i, r_j in itertools.combinations(r_up_carts, 2):
            J2 += u(r_i, r_j)
        for r_i, r_j in itertools.combinations(r_dn_carts, 2):
            J2 += u(r_i, r_j)
        return J2

    def u(r_i, r_j):
        r_ij = np.linalg.norm(np.asarray(np.asarray(r_i) - np.asarray(r_j), dtype=dtype))
        if j2b_type == "exp":
            return dtype(1.0) / (dtype(2.0) * a) * (dtype(1.0) - np.exp(-a * r_ij))
        return r_ij / dtype(2.0) * (dtype(1.0) + a * r_ij) ** dtype(-1.0)

    pairs = (
        itertools.product(r_up_carts, r_dn_carts),
        itertools.combinations(r_up_carts, 2),
        itertools.combinations(r_dn_carts, 2),
    )
    anti, up_up, dn_dn = (np.sum(np.asarray([u(r_i, r_j) for r_i, r_j in group], dtype=dtype)) for group in pairs)
    return anti + up_up + dn_dn


class TestJastrowOneBody:
    @pytest.mark.light
    @pytest.mark.parametrize("j1b_type", ["exp", "pade"])
    def test_Jastrow_onebody_part_molecular(self, j1b_type):
        """Production molecular J1 equals its debug formula evaluated in the jastrow_eval zone dtype.

        The reference transcribes ``_compute_Jastrow_one_body_molecular_debug``
        with fp64 ``r - R`` cast to the zone and the radial law / sum in-zone, so
        mixed is judged in its own precision; its fp64 evaluation must reproduce
        the fp64 debug oracle bit for bit (the full-mode reference is unchanged).
        """
        fixture_rng = np.random.RandomState(20260923)
        atol, rtol = get_tolerance("jastrow_eval", "strict")
        num_r_up_cart_samples = 8
        num_r_dn_cart_samples = 4
        num_R_cart_samples = 6

        # generate matrices for the test
        r_cart_min, r_cart_max = -1.0, 1.0
        R_cart_min, R_cart_max = 0.0, 0.0
        r_up_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_up_cart_samples, 3) + r_cart_min
        r_dn_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_dn_cart_samples, 3) + r_cart_min
        R_carts = (R_cart_max - R_cart_min) * fixture_rng.rand(num_R_cart_samples, 3) + R_cart_min

        structure_data = _make_structure(R_carts, atomic_number=6)

        core_electrons = tuple([3] * num_R_cart_samples)

        jastrow_one_body_data = Jastrow_one_body_data(
            jastrow_1b_param=1.0, jastrow_1b_type=j1b_type, structure_data=structure_data, core_electrons=core_electrons
        )

        J1_debug = _compute_Jastrow_one_body_molecular_debug(
            jastrow_one_body_data=jastrow_one_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        # Judge production against the debug formula evaluated in the
        # jastrow_eval zone dtype (fp64 geometry, radial law and sum in-zone).
        J1_reference = _j1_zone_reference(jastrow_one_body_data, r_up_carts, r_dn_carts, get_dtype_np("jastrow_eval"))

        J1_jax = compute_Jastrow_one_body(
            jastrow_one_body_data=jastrow_one_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        assert np.all(np.isfinite(np.asarray(J1_reference))), "non-finite value in the zone-dtype reference"
        assert np.all(np.isfinite(np.asarray(J1_jax))), "non-finite value in the production J1"
        np.testing.assert_allclose(J1_reference, J1_jax, atol=atol, rtol=rtol, equal_nan=False)
        # The fp64 transcription is the declared fp64 debug oracle bit for bit.
        assert np.all(np.isfinite(np.asarray(J1_debug))), "non-finite value in the fp64 debug oracle"
        np.testing.assert_array_equal(_j1_zone_reference(jastrow_one_body_data, r_up_carts, r_dn_carts, np.float64), J1_debug)

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("j1b_type", ["exp", "pade"])
    def test_Jastrow_onebody_part_periodic(self, j1b_type):
        """PBC sibling of ``test_Jastrow_onebody_part_molecular``.

        Compares the production ``compute_Jastrow_one_body`` PBC branch on a
        cubic 10 Bohr cell with ``_compute_Jastrow_one_body_pbc_debug`` evaluated
        in the jastrow_eval zone dtype: the smooth periodic distance of the
        fp64 ``r - R`` is formed in fp64, as production forms it, and cast to the
        zone; the radial law and the sum run in-zone. The fp64 evaluation must
        reproduce the fp64 debug oracle bit for bit.
        """
        fixture_rng = np.random.RandomState(20260923)
        atol, rtol = get_tolerance("jastrow_eval", "strict")
        num_r_up_cart_samples = 8
        num_r_dn_cart_samples = 4
        num_R_cart_samples = 6

        # atoms placed inside the cell; electrons span the full cell.
        r_up_carts = 10.0 * fixture_rng.rand(num_r_up_cart_samples, 3)
        r_dn_carts = 10.0 * fixture_rng.rand(num_r_dn_cart_samples, 3)
        R_carts = 2.0 + 6.0 * fixture_rng.rand(num_R_cart_samples, 3)

        structure_data = _make_pbc_structure(R_carts, atomic_number=6)

        core_electrons = tuple([3] * num_R_cart_samples)

        jastrow_one_body_data = Jastrow_one_body_data(
            jastrow_1b_param=1.0, jastrow_1b_type=j1b_type, structure_data=structure_data, core_electrons=core_electrons
        )

        J1_debug = _compute_Jastrow_one_body_pbc_debug(
            jastrow_one_body_data=jastrow_one_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        # Judge production against the debug formula evaluated in the
        # jastrow_eval zone dtype (fp64 geometry, radial law and sum in-zone).
        J1_reference = _j1_zone_reference(jastrow_one_body_data, r_up_carts, r_dn_carts, get_dtype_np("jastrow_eval"))

        J1_jax = compute_Jastrow_one_body(
            jastrow_one_body_data=jastrow_one_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        assert np.all(np.isfinite(np.asarray(J1_reference))), "non-finite value in the zone-dtype reference"
        assert np.all(np.isfinite(np.asarray(J1_jax))), "non-finite value in the production J1"
        np.testing.assert_allclose(J1_reference, J1_jax, atol=atol, rtol=rtol, equal_nan=False)
        # The fp64 transcription is the declared fp64 debug oracle bit for bit.
        assert np.all(np.isfinite(np.asarray(J1_debug))), "non-finite value in the fp64 debug oracle"
        np.testing.assert_array_equal(_j1_zone_reference(jastrow_one_body_data, r_up_carts, r_dn_carts, np.float64), J1_debug)

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("j1b_type", ["exp", "pade"])
    def test_analytical_and_auto_grads_and_laplacian_Jastrow_onebody_part_molecular(self, j1b_type):
        """Analytic vs auto-diff gradients/laplacian for one-body Jastrow."""

        atol, rtol = get_tolerance_min(["jastrow_grad_lap"], "strict")

        def case_factory():
            fixture_rng = np.random.RandomState(20260923)
            num_r_up_cart_samples = 5
            num_r_dn_cart_samples = 4
            num_R_cart_samples = 4

            r_cart_min, r_cart_max = -2.0, 2.0
            R_cart_min, R_cart_max = 0.0, 0.0
            r_up_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_up_cart_samples, 3) + r_cart_min
            r_dn_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_dn_cart_samples, 3) + r_cart_min
            R_carts = (R_cart_max - R_cart_min) * fixture_rng.rand(num_R_cart_samples, 3) + R_cart_min

            structure_data = _make_structure(R_carts, atomic_number=6)

            core_electrons = tuple([3] * num_R_cart_samples)
            jastrow_one_body_data = Jastrow_one_body_data(
                jastrow_1b_param=1.0, jastrow_1b_type=j1b_type, structure_data=structure_data, core_electrons=core_electrons
            )
            return (jastrow_one_body_data, r_up_carts, r_dn_carts)

        def analytical(case):
            jastrow_one_body_data, r_up_carts, r_dn_carts = case
            return compute_grads_and_laplacian_Jastrow_one_body(jastrow_one_body_data, r_up_carts, r_dn_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            mixed_atol, mixed_rtol = get_tolerance_mixed_fidelity("parameter_derivative")
            for mixed_block, full_block in blocks:
                _assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, mixed_atol, mixed_rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=mixed_atol, rtol=mixed_rtol)
            with pytest.raises(AssertionError):
                _assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], mixed_atol, mixed_rtol)
            jax.clear_caches()
            return

        jastrow_one_body_data, r_up_carts, r_dn_carts = case_factory()
        grad_up_an, grad_dn_an, lap_up_an, lap_dn_an = compute_grads_and_laplacian_Jastrow_one_body(
            jastrow_one_body_data,
            r_up_carts,
            r_dn_carts,
        )

        grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_one_body_auto(
            jastrow_one_body_data,
            r_up_carts,
            r_dn_carts,
        )

        assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(grad_up_an), np.asarray(grad_up_auto), atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(grad_dn_an), np.asarray(grad_dn_auto), atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(lap_up_an), np.asarray(lap_up_auto), atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(lap_dn_an), np.asarray(lap_dn_auto), atol=atol, rtol=rtol)

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("j1b_type", ["exp", "pade"])
    def test_analytical_and_auto_grads_and_laplacian_Jastrow_onebody_part_periodic(self, j1b_type):
        """PBC sibling of ``..._onebody_part_molecular`` (analytical vs auto)."""

        atol, rtol = get_tolerance_min(["jastrow_grad_lap"], "strict")

        def case_factory():
            fixture_rng = np.random.RandomState(20260923)
            num_r_up_cart_samples = 5
            num_r_dn_cart_samples = 4
            num_R_cart_samples = 4

            r_up_carts = 10.0 * fixture_rng.rand(num_r_up_cart_samples, 3)
            r_dn_carts = 10.0 * fixture_rng.rand(num_r_dn_cart_samples, 3)
            R_carts = 2.0 + 6.0 * fixture_rng.rand(num_R_cart_samples, 3)

            structure_data = _make_pbc_structure(R_carts, atomic_number=6)

            core_electrons = tuple([3] * num_R_cart_samples)
            jastrow_one_body_data = Jastrow_one_body_data(
                jastrow_1b_param=1.0, jastrow_1b_type=j1b_type, structure_data=structure_data, core_electrons=core_electrons
            )
            return (jastrow_one_body_data, r_up_carts, r_dn_carts)

        def analytical(case):
            jastrow_one_body_data, r_up_carts, r_dn_carts = case
            return compute_grads_and_laplacian_Jastrow_one_body(jastrow_one_body_data, r_up_carts, r_dn_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            mixed_atol, mixed_rtol = get_tolerance_mixed_fidelity("parameter_derivative")
            for mixed_block, full_block in blocks:
                _assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, mixed_atol, mixed_rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=mixed_atol, rtol=mixed_rtol)
            with pytest.raises(AssertionError):
                _assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], mixed_atol, mixed_rtol)
            jax.clear_caches()
            return

        jastrow_one_body_data, r_up_carts, r_dn_carts = case_factory()
        grad_up_an, grad_dn_an, lap_up_an, lap_dn_an = compute_grads_and_laplacian_Jastrow_one_body(
            jastrow_one_body_data,
            r_up_carts,
            r_dn_carts,
        )

        grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_one_body_auto(
            jastrow_one_body_data,
            r_up_carts,
            r_dn_carts,
        )

        assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(grad_up_an), np.asarray(grad_up_auto), atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(grad_dn_an), np.asarray(grad_dn_auto), atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(lap_up_an), np.asarray(lap_up_auto), atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(lap_dn_an), np.asarray(lap_dn_auto), atol=atol, rtol=rtol)

        jax.clear_caches()


class TestJastrowTwoBody:
    @pytest.mark.light
    @pytest.mark.parametrize("j2b_type", ["pade", "exp"])
    def test_Jastrow_twobody_part_molecular(self, j2b_type):
        """Production molecular J2 equals its debug formula evaluated in the jastrow_eval zone dtype.

        The reference transcribes ``_compute_Jastrow_two_body_molecular_debug``
        with fp64 ``r_i - r_j`` cast to the zone and the pair law / sum in-zone;
        its fp64 evaluation must reproduce the fp64 debug oracle bit for bit.
        """
        fixture_rng = np.random.RandomState(20260923)
        atol, rtol = get_tolerance("jastrow_eval", "strict")
        num_r_up_cart_samples = 5
        num_r_dn_cart_samples = 2

        r_cart_min, r_cart_max = -3.0, 3.0

        r_up_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_up_cart_samples, 3) + r_cart_min
        r_dn_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_dn_cart_samples, 3) + r_cart_min

        jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type=j2b_type)
        J2_debug = _compute_Jastrow_two_body_molecular_debug(
            jastrow_two_body_data=jastrow_two_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )
        # Judge production against the debug formula evaluated in the
        # jastrow_eval zone dtype (fp64 geometry, pair law and sum in-zone).
        J2_reference = _j2_zone_reference(jastrow_two_body_data, r_up_carts, r_dn_carts, get_dtype_np("jastrow_eval"))

        J2_jax = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )

        assert np.all(np.isfinite(np.asarray(J2_reference))), "non-finite value in the zone-dtype reference"
        assert np.all(np.isfinite(np.asarray(J2_jax))), "non-finite value in the production J2"
        np.testing.assert_allclose(J2_reference, J2_jax, atol=atol, rtol=rtol, equal_nan=False)
        # The fp64 transcription is the declared fp64 debug oracle bit for bit.
        assert np.all(np.isfinite(np.asarray(J2_debug))), "non-finite value in the fp64 debug oracle"
        np.testing.assert_array_equal(_j2_zone_reference(jastrow_two_body_data, r_up_carts, r_dn_carts, np.float64), J2_debug)

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("j2b_type", ["pade", "exp"])
    def test_Jastrow_twobody_part_periodic(self, j2b_type):
        """PBC sibling of ``test_Jastrow_twobody_part_molecular``.

        The smooth periodic distance of the fp64 ``r_i - r_j`` is formed in
        fp64, as production forms it, and cast to the jastrow_eval zone; the pair
        law and the sum run in-zone. The fp64 evaluation must reproduce
        ``_compute_Jastrow_two_body_pbc_debug`` bit for bit.
        """
        fixture_rng = np.random.RandomState(20260923)
        atol, rtol = get_tolerance("jastrow_eval", "strict")
        num_r_up_cart_samples = 5
        num_r_dn_cart_samples = 2

        r_up_carts = 10.0 * fixture_rng.rand(num_r_up_cart_samples, 3)
        r_dn_carts = 10.0 * fixture_rng.rand(num_r_dn_cart_samples, 3)

        structure_data = _make_pbc_structure(np.zeros((1, 3)), atomic_number=1)
        jastrow_two_body_data = Jastrow_two_body_data(
            jastrow_2b_param=1.0, jastrow_2b_type=j2b_type, structure_data=structure_data
        )
        J2_debug = _compute_Jastrow_two_body_pbc_debug(
            jastrow_two_body_data=jastrow_two_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )
        # Judge production against the debug formula evaluated in the
        # jastrow_eval zone dtype (fp64 geometry, pair law and sum in-zone).
        J2_reference = _j2_zone_reference(jastrow_two_body_data, r_up_carts, r_dn_carts, get_dtype_np("jastrow_eval"))

        J2_jax = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )

        assert np.all(np.isfinite(np.asarray(J2_reference))), "non-finite value in the zone-dtype reference"
        assert np.all(np.isfinite(np.asarray(J2_jax))), "non-finite value in the production J2"
        np.testing.assert_allclose(J2_reference, J2_jax, atol=atol, rtol=rtol, equal_nan=False)
        # The fp64 transcription is the declared fp64 debug oracle bit for bit.
        assert np.all(np.isfinite(np.asarray(J2_debug))), "non-finite value in the fp64 debug oracle"
        np.testing.assert_array_equal(_j2_zone_reference(jastrow_two_body_data, r_up_carts, r_dn_carts, np.float64), J2_debug)

        jax.clear_caches()

    @pytest.mark.heavy
    @pytest.mark.parametrize("j2b_type", ["pade", "exp"])
    def test_analytical_and_auto_grads_and_laplacian_Jastrow_twobody_part_molecular(self, j2b_type):
        """Analytic vs auto-diff gradients/laplacian for two-body Jastrow."""

        atol, rtol = get_tolerance_min(["jastrow_grad_lap"], "strict")

        def case_factory():
            fixture_rng = np.random.RandomState(20260923)
            num_r_up_cart_samples = 5
            num_r_dn_cart_samples = 2

            r_cart_min, r_cart_max = -3.0, 3.0

            r_up_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_up_cart_samples, 3) + r_cart_min
            r_dn_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_dn_cart_samples, 3) + r_cart_min

            jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type=j2b_type)

            return (jastrow_two_body_data, r_up_carts, r_dn_carts)

        def analytical(case):
            jastrow_two_body_data, r_up_carts, r_dn_carts = case
            return compute_grads_and_laplacian_Jastrow_two_body(jastrow_two_body_data, r_up_carts, r_dn_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            mixed_atol, mixed_rtol = get_tolerance_mixed_fidelity("parameter_derivative")
            for mixed_block, full_block in blocks:
                _assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, mixed_atol, mixed_rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=mixed_atol, rtol=mixed_rtol)
            with pytest.raises(AssertionError):
                _assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], mixed_atol, mixed_rtol)
            jax.clear_caches()
            return

        jastrow_two_body_data, r_up_carts, r_dn_carts = case_factory()
        grad_J2_up_analytic, grad_J2_dn_analytic, lap_J2_up_analytic, lap_J2_dn_analytic = (
            compute_grads_and_laplacian_Jastrow_two_body(
                jastrow_two_body_data,
                r_up_carts,
                r_dn_carts,
            )
        )

        grad_J2_up_auto, grad_J2_dn_auto, lap_J2_up_auto, lap_J2_dn_auto = _compute_grads_and_laplacian_Jastrow_two_body_auto(
            jastrow_two_body_data,
            r_up_carts,
            r_dn_carts,
        )

        assert not np.any(np.isnan(np.asarray(np.asarray(grad_J2_up_analytic)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_J2_up_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(grad_J2_up_analytic),
            np.asarray(grad_J2_up_auto),
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_J2_dn_analytic)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_J2_dn_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(grad_J2_dn_analytic),
            np.asarray(grad_J2_dn_auto),
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_J2_up_analytic)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_J2_up_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(lap_J2_up_analytic),
            np.asarray(lap_J2_up_auto),
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_J2_dn_analytic)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_J2_dn_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(lap_J2_dn_analytic),
            np.asarray(lap_J2_dn_auto),
            atol=atol,
            rtol=rtol,
        )

        jax.clear_caches()

    @pytest.mark.heavy
    @pytest.mark.parametrize("j2b_type", ["pade", "exp"])
    def test_analytical_and_auto_grads_and_laplacian_Jastrow_twobody_part_periodic(self, j2b_type):
        """PBC sibling of ``..._twobody_part_molecular`` (analytical vs auto)."""

        atol, rtol = get_tolerance_min(["jastrow_grad_lap"], "strict")

        def case_factory():
            fixture_rng = np.random.RandomState(20260923)
            num_r_up_cart_samples = 5
            num_r_dn_cart_samples = 2

            r_up_carts = 10.0 * fixture_rng.rand(num_r_up_cart_samples, 3)
            r_dn_carts = 10.0 * fixture_rng.rand(num_r_dn_cart_samples, 3)

            structure_data = _make_pbc_structure(np.zeros((1, 3)), atomic_number=1)
            jastrow_two_body_data = Jastrow_two_body_data(
                jastrow_2b_param=1.0, jastrow_2b_type=j2b_type, structure_data=structure_data
            )

            return (jastrow_two_body_data, r_up_carts, r_dn_carts)

        def analytical(case):
            jastrow_two_body_data, r_up_carts, r_dn_carts = case
            return compute_grads_and_laplacian_Jastrow_two_body(jastrow_two_body_data, r_up_carts, r_dn_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            mixed_atol, mixed_rtol = get_tolerance_mixed_fidelity("parameter_derivative")
            for mixed_block, full_block in blocks:
                _assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, mixed_atol, mixed_rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=mixed_atol, rtol=mixed_rtol)
            with pytest.raises(AssertionError):
                _assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], mixed_atol, mixed_rtol)
            jax.clear_caches()
            return

        jastrow_two_body_data, r_up_carts, r_dn_carts = case_factory()
        grad_J2_up_analytic, grad_J2_dn_analytic, lap_J2_up_analytic, lap_J2_dn_analytic = (
            compute_grads_and_laplacian_Jastrow_two_body(
                jastrow_two_body_data,
                r_up_carts,
                r_dn_carts,
            )
        )

        grad_J2_up_auto, grad_J2_dn_auto, lap_J2_up_auto, lap_J2_dn_auto = _compute_grads_and_laplacian_Jastrow_two_body_auto(
            jastrow_two_body_data,
            r_up_carts,
            r_dn_carts,
        )

        assert not np.any(np.isnan(np.asarray(np.asarray(grad_J2_up_analytic)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_J2_up_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(grad_J2_up_analytic),
            np.asarray(grad_J2_up_auto),
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_J2_dn_analytic)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_J2_dn_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(grad_J2_dn_analytic),
            np.asarray(grad_J2_dn_auto),
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_J2_up_analytic)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_J2_up_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(lap_J2_up_analytic),
            np.asarray(lap_J2_up_auto),
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_J2_dn_analytic)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_J2_dn_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(
            np.asarray(lap_J2_dn_analytic),
            np.asarray(lap_J2_dn_auto),
            atol=atol,
            rtol=rtol,
        )

        jax.clear_caches()


class TestJastrowThreeBody:
    class TestAOBasis:
        @pytest.mark.light
        def test_Jastrow_threebody_part_with_AOs_data_molecular(self):
            """Test the three-body Jastrow factor, comparing the debug and JAX implementations, using AOs data."""
            fixture_rng = np.random.RandomState(20260923)
            atol, rtol = get_tolerance("jastrow_eval", "strict")
            num_r_up_cart_samples = 4
            num_r_dn_cart_samples = 2
            num_R_cart_samples = 6
            num_ao = 6
            num_ao_prim = 6
            orbital_indices = [0, 1, 2, 3, 4, 5]
            exponents = [1.2, 0.5, 0.1, 0.05, 0.05, 0.05]
            coefficients = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            angular_momentums = [0, 0, 0, 1, 1, 1]
            magnetic_quantum_numbers = [0, 0, 0, 0, +1, -1]

            orbital_indices = tuple(orbital_indices)
            exponents = np.array(exponents, dtype=np.float64)
            coefficients = np.array(coefficients, dtype=np.float64)
            angular_momentums = tuple(angular_momentums)
            magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

            # generate matrices for the test
            r_cart_min, r_cart_max = -1.0, 1.0
            R_cart_min, R_cart_max = 0.0, 0.0
            r_up_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_up_cart_samples, 3) + r_cart_min
            r_dn_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_dn_cart_samples, 3) + r_cart_min
            R_carts = (R_cart_max - R_cart_min) * fixture_rng.rand(num_R_cart_samples, 3) + R_cart_min

            structure_data = _make_structure(R_carts)

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

            j_matrix = fixture_rng.rand(aos_data.num_ao, aos_data.num_ao + 1)
            j_matrix[:, :-1] = 0.5 * (j_matrix[:, :-1] + j_matrix[:, :-1].T)  # symmetric J3 (bosonic)

            jastrow_three_body_data = Jastrow_three_body_data(orb_data=aos_data, j_matrix=j_matrix)

            J3_debug = _compute_Jastrow_three_body_debug(
                jastrow_three_body_data=jastrow_three_body_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )

            J3_jax = compute_Jastrow_three_body(
                jastrow_three_body_data=jastrow_three_body_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )

            assert not np.any(np.isnan(np.asarray(J3_debug))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(J3_jax))), "NaN detected in second argument"
            np.testing.assert_allclose(J3_debug, J3_jax, atol=atol, rtol=rtol)

            jax.clear_caches()

        @pytest.mark.light
        def test_Jastrow_threebody_part_with_AOs_data_periodic(self):
            """PBC sibling of ``..._with_AOs_data_molecular`` (periodic GTO image expansion)."""
            fixture_rng = np.random.RandomState(20260923)
            atol, rtol = get_tolerance("jastrow_eval", "strict")
            num_r_up_cart_samples = 4
            num_r_dn_cart_samples = 2
            num_R_cart_samples = 6
            num_ao = 6
            num_ao_prim = 6
            orbital_indices = tuple([0, 1, 2, 3, 4, 5])
            exponents = np.array([1.2, 0.8, 0.5, 1.0, 0.9, 0.7], dtype=np.float64)
            coefficients = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
            angular_momentums = tuple([0, 0, 0, 1, 1, 1])
            magnetic_quantum_numbers = tuple([0, 0, 0, 0, +1, -1])

            r_up_carts = 10.0 * fixture_rng.rand(num_r_up_cart_samples, 3)
            r_dn_carts = 10.0 * fixture_rng.rand(num_r_dn_cart_samples, 3)
            R_carts = 2.0 + 6.0 * fixture_rng.rand(num_R_cart_samples, 3)

            structure_data = _make_pbc_structure(R_carts, atomic_number=1)

            aos_pbc_mol = AOs_sphe_gto_data._from_full_basis(
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

            # ``j_matrix`` is sized by the molecular AO count; the public AO wrapper
            # collapses the image axis back to the molecular rank before contraction.
            j_matrix = fixture_rng.rand(aos_pbc_mol.num_ao, aos_pbc_mol.num_ao + 1)
            j_matrix[:, :-1] = 0.5 * (j_matrix[:, :-1] + j_matrix[:, :-1].T)  # symmetric J3 (bosonic)

            expanded_aos = _apply_image_expansion(aos_pbc_mol, structure_data, tol=AO_IMAGE_TRUNCATION_ATOL)

            jastrow_three_body_data = Jastrow_three_body_data(orb_data=expanded_aos, j_matrix=j_matrix)

            J3_debug = _compute_Jastrow_three_body_debug(
                jastrow_three_body_data=jastrow_three_body_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )

            J3_jax = compute_Jastrow_three_body(
                jastrow_three_body_data=jastrow_three_body_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )

            assert not np.any(np.isnan(np.asarray(J3_debug))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(J3_jax))), "NaN detected in second argument"
            np.testing.assert_allclose(J3_debug, J3_jax, atol=atol, rtol=rtol)

            jax.clear_caches()

        @pytest.mark.heavy
        def test_analytical_and_auto_grads_and_laplacian_Jastrow_threebody_part_with_AOs_data_molecular(self):
            """Analytic vs auto-diff gradients/laplacian for three-body Jastrow (AOs)."""

            # J3 grad/lap crosses two zones: jastrow_grad_lap (fp32 mixed) + ao_grad_lap (fp64).
            # Use the looser of the two -- under mixed precision, jastrow_grad_lap dominates.
            atol, rtol = get_tolerance_min(["jastrow_grad_lap", "ao_grad_lap"], "strict")

            def case_factory():
                fixture_rng = np.random.RandomState(20260923)
                num_r_up_cart_samples = 4
                num_r_dn_cart_samples = 2
                num_R_cart_samples = 5
                num_ao = 5
                num_ao_prim = 5
                orbital_indices = [0, 1, 2, 3, 4]
                exponents = [1.2, 0.6, 0.2, 0.1, 0.05]
                coefficients = [1.0, 0.9, 1.0, 1.0, 1.0]
                angular_momentums = [0, 0, 0, 1, 1]
                magnetic_quantum_numbers = [0, 0, 0, 0, 1]

                r_cart_min, r_cart_max = -1.5, 1.5
                R_cart_min, R_cart_max = -0.5, 0.5
                r_up_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_up_cart_samples, 3) + r_cart_min
                r_dn_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_dn_cart_samples, 3) + r_cart_min
                R_carts = (R_cart_max - R_cart_min) * fixture_rng.rand(num_R_cart_samples, 3) + R_cart_min

                structure_data = _make_structure(R_carts)

                aos_data = AOs_sphe_gto_data._from_full_basis(
                    structure_data=structure_data,
                    nucleus_index=tuple(list(range(num_R_cart_samples))),
                    num_ao=num_ao,
                    num_ao_prim=num_ao_prim,
                    orbital_indices=tuple(orbital_indices),
                    exponents=np.array(exponents, dtype=np.float64),
                    coefficients=np.array(coefficients, dtype=np.float64),
                    angular_momentums=tuple(angular_momentums),
                    magnetic_quantum_numbers=tuple(magnetic_quantum_numbers),
                )

                j_matrix = fixture_rng.rand(aos_data.num_ao, aos_data.num_ao + 1)
                j_matrix[:, :-1] = 0.5 * (j_matrix[:, :-1] + j_matrix[:, :-1].T)  # symmetric J3 (bosonic)
                jastrow_three_body_data = Jastrow_three_body_data(orb_data=aos_data, j_matrix=j_matrix)

                return (jastrow_three_body_data, r_up_carts, r_dn_carts)

            def analytical(case):
                jastrow_three_body_data, r_up_carts, r_dn_carts = case
                return compute_grads_and_laplacian_Jastrow_three_body(jastrow_three_body_data, r_up_carts, r_dn_carts)

            if is_mixed_precision_enabled():
                mixed, full = mixed_vs_full(case_factory, analytical)
                blocks = mixed_fidelity_blocks(mixed, full)
                mixed_atol, mixed_rtol = get_tolerance_mixed_fidelity("parameter_derivative")
                for mixed_block, full_block in blocks:
                    _assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, mixed_atol, mixed_rtol)
                mutated = out_of_budget_mixed_block(*blocks[0], atol=mixed_atol, rtol=mixed_rtol)
                with pytest.raises(AssertionError):
                    _assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], mixed_atol, mixed_rtol)
                jax.clear_caches()
                return

            jastrow_three_body_data, r_up_carts, r_dn_carts = case_factory()
            grad_up_an, grad_dn_an, lap_up_an, lap_dn_an = compute_grads_and_laplacian_Jastrow_three_body(
                jastrow_three_body_data,
                r_up_carts,
                r_dn_carts,
            )

            grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_three_body_auto(
                jastrow_three_body_data,
                r_up_carts,
                r_dn_carts,
            )

            assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(grad_up_an), np.asarray(grad_up_auto), atol=atol, rtol=rtol)
            assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(grad_dn_an), np.asarray(grad_dn_auto), atol=atol, rtol=rtol)
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(lap_up_an), np.asarray(lap_up_auto), atol=atol, rtol=rtol)
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(lap_dn_an), np.asarray(lap_dn_auto), atol=atol, rtol=rtol)

            jax.clear_caches()

        @pytest.mark.heavy
        def test_analytical_and_auto_grads_and_laplacian_Jastrow_threebody_part_with_AOs_data_periodic(self):
            """PBC sibling of ``..._threebody_part_with_AOs_data_molecular`` (analytical vs auto)."""

            atol, rtol = get_tolerance_min(["jastrow_grad_lap", "ao_grad_lap"], "strict")

            def case_factory():
                fixture_rng = np.random.RandomState(20260923)
                num_r_up_cart_samples = 4
                num_r_dn_cart_samples = 2
                num_R_cart_samples = 5
                num_ao = 5
                num_ao_prim = 5
                orbital_indices = tuple([0, 1, 2, 3, 4])
                exponents = np.array([1.2, 0.9, 0.6, 1.0, 0.8], dtype=np.float64)
                coefficients = np.array([1.0, 0.9, 1.0, 1.0, 1.0], dtype=np.float64)
                angular_momentums = tuple([0, 0, 0, 1, 1])
                magnetic_quantum_numbers = tuple([0, 0, 0, 0, 1])

                r_up_carts = 10.0 * fixture_rng.rand(num_r_up_cart_samples, 3)
                r_dn_carts = 10.0 * fixture_rng.rand(num_r_dn_cart_samples, 3)
                R_carts = 2.0 + 6.0 * fixture_rng.rand(num_R_cart_samples, 3)

                structure_data = _make_pbc_structure(R_carts, atomic_number=1)

                aos_pbc_mol = AOs_sphe_gto_data._from_full_basis(
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

                j_matrix = fixture_rng.rand(aos_pbc_mol.num_ao, aos_pbc_mol.num_ao + 1)
                j_matrix[:, :-1] = 0.5 * (j_matrix[:, :-1] + j_matrix[:, :-1].T)  # symmetric J3 (bosonic)

                expanded_aos = _apply_image_expansion(aos_pbc_mol, structure_data, tol=AO_IMAGE_TRUNCATION_ATOL)

                jastrow_three_body_data = Jastrow_three_body_data(orb_data=expanded_aos, j_matrix=j_matrix)

                return (jastrow_three_body_data, r_up_carts, r_dn_carts)

            def analytical(case):
                jastrow_three_body_data, r_up_carts, r_dn_carts = case
                return compute_grads_and_laplacian_Jastrow_three_body(jastrow_three_body_data, r_up_carts, r_dn_carts)

            if is_mixed_precision_enabled():
                mixed, full = mixed_vs_full(case_factory, analytical)
                blocks = mixed_fidelity_blocks(mixed, full)
                mixed_atol, mixed_rtol = get_tolerance_mixed_fidelity("parameter_derivative")
                for mixed_block, full_block in blocks:
                    _assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, mixed_atol, mixed_rtol)
                mutated = out_of_budget_mixed_block(*blocks[0], atol=mixed_atol, rtol=mixed_rtol)
                with pytest.raises(AssertionError):
                    _assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], mixed_atol, mixed_rtol)
                jax.clear_caches()
                return

            jastrow_three_body_data, r_up_carts, r_dn_carts = case_factory()
            grad_up_an, grad_dn_an, lap_up_an, lap_dn_an = compute_grads_and_laplacian_Jastrow_three_body(
                jastrow_three_body_data,
                r_up_carts,
                r_dn_carts,
            )

            grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_three_body_auto(
                jastrow_three_body_data,
                r_up_carts,
                r_dn_carts,
            )

            assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(grad_up_an), np.asarray(grad_up_auto), atol=atol, rtol=rtol)
            assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(grad_dn_an), np.asarray(grad_dn_auto), atol=atol, rtol=rtol)
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(lap_up_an), np.asarray(lap_up_auto), atol=atol, rtol=rtol)
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(lap_dn_an), np.asarray(lap_dn_auto), atol=atol, rtol=rtol)

            jax.clear_caches()

    class TestMOBasis:
        @pytest.mark.light
        def test_Jastrow_threebody_part_with_MOs_data_molecular(self):
            """Test the three-body Jastrow factor, comparing the debug and JAX implementations, using MOs data."""
            fixture_rng = np.random.RandomState(20260923)
            atol, rtol = get_tolerance("jastrow_eval", "strict")
            num_el = 10
            num_mo = 5
            num_ao = 3
            num_ao_prim = 4
            orbital_indices = [0, 0, 1, 2]
            exponents = [50.0, 20.0, 10.0, 5.0]
            coefficients = [1.0, 1.0, 1.0, 0.5]
            angular_momentums = [1, 1, 1]
            magnetic_quantum_numbers = [0, 0, -1]

            orbital_indices = tuple(orbital_indices)
            exponents = np.array(exponents, dtype=np.float64)
            coefficients = np.array(coefficients, dtype=np.float64)
            angular_momentums = tuple(angular_momentums)
            magnetic_quantum_numbers = tuple(magnetic_quantum_numbers)

            num_r_up_cart_samples = num_r_dn_cart_samples = num_el
            num_R_cart_samples = num_ao
            r_cart_min, r_cart_max = -5.0, 5.0
            R_cart_min, R_cart_max = 10.0, 10.0
            r_up_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_up_cart_samples, 3) + r_cart_min
            r_dn_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_dn_cart_samples, 3) + r_cart_min
            R_carts = (R_cart_max - R_cart_min) * fixture_rng.rand(num_R_cart_samples, 3) + R_cart_min

            mo_coefficients = fixture_rng.rand(num_mo, num_ao)

            structure_data = _make_structure(R_carts)

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

            j_matrix = fixture_rng.rand(mos_data.num_mo, mos_data.num_mo + 1)
            j_matrix[:, :-1] = 0.5 * (j_matrix[:, :-1] + j_matrix[:, :-1].T)  # symmetric J3 (bosonic)

            jastrow_three_body_data = Jastrow_three_body_data(orb_data=mos_data, j_matrix=j_matrix)

            J3_debug = _compute_Jastrow_three_body_debug(
                jastrow_three_body_data=jastrow_three_body_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )

            J3_jax = compute_Jastrow_three_body(
                jastrow_three_body_data=jastrow_three_body_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )

            assert not np.any(np.isnan(np.asarray(J3_debug))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(J3_jax))), "NaN detected in second argument"
            np.testing.assert_allclose(J3_debug, J3_jax, atol=atol, rtol=rtol)

            jax.clear_caches()

        @pytest.mark.light
        def test_Jastrow_threebody_part_with_MOs_data_periodic(self):
            """PBC sibling of ``..._with_MOs_data_molecular`` (periodic GTO image expansion)."""
            # Seed: the random atom geometry drives the per-shell RAGGED image
            # count; left unseeded this was flaky under pytest-xdist (a geometry
            # whose kernel-rank num_ao_prim is not a multiple of _n_image crashed
            # the prim->AO reduction -- now fixed). See the ``_ragged_nonuniform``
            # sibling for the explicit non-uniform regression.
            np.random.seed(0)
            atol, rtol = get_tolerance("jastrow_eval", "strict")
            num_el = 10
            num_mo = 5
            num_ao = 3
            num_ao_prim = 4
            orbital_indices = tuple([0, 0, 1, 2])
            exponents = np.array([1.5, 1.0, 0.8, 0.6], dtype=np.float64)
            coefficients = np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float64)
            angular_momentums = tuple([1, 1, 1])
            magnetic_quantum_numbers = tuple([0, 0, -1])

            num_r_up_cart_samples = num_r_dn_cart_samples = num_el
            num_R_cart_samples = num_ao
            r_up_carts = 10.0 * np.random.rand(num_r_up_cart_samples, 3)
            r_dn_carts = 10.0 * np.random.rand(num_r_dn_cart_samples, 3)
            R_carts = 2.0 + 6.0 * np.random.rand(num_R_cart_samples, 3)

            mo_coefficients = np.random.rand(num_mo, num_ao)

            structure_data = _make_pbc_structure(R_carts, atomic_number=1)

            aos_pbc_mol = AOs_sphe_gto_data._from_full_basis(
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

            expanded_aos = _apply_image_expansion(aos_pbc_mol, structure_data, tol=AO_IMAGE_TRUNCATION_ATOL)

            # ``mo_coefficients`` is sized by the molecular AO count (post-image view).
            mos_data = MOs_data(num_mo=num_mo, aos_data=expanded_aos, mo_coefficients=mo_coefficients)

            j_matrix = np.random.rand(mos_data.num_mo, mos_data.num_mo + 1)
            j_matrix[:, :-1] = 0.5 * (j_matrix[:, :-1] + j_matrix[:, :-1].T)  # symmetric J3 (bosonic)

            jastrow_three_body_data = Jastrow_three_body_data(orb_data=mos_data, j_matrix=j_matrix)

            J3_debug = _compute_Jastrow_three_body_debug(
                jastrow_three_body_data=jastrow_three_body_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )

            J3_jax = compute_Jastrow_three_body(
                jastrow_three_body_data=jastrow_three_body_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )

            assert not np.any(np.isnan(np.asarray(J3_debug))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(J3_jax))), "NaN detected in second argument"
            np.testing.assert_allclose(J3_debug, J3_jax, atol=atol, rtol=rtol)

            jax.clear_caches()

        @pytest.mark.light
        def test_Jastrow_threebody_part_with_MOs_data_periodic_ragged_nonuniform(self):
            """Regression: J3 (MO basis) over a periodic leaf whose per-shell RAGGED
            image expansion is NON-UNIFORM (kernel-rank ``num_ao_prim`` is NOT a
            multiple of ``_n_image``).

            A uniform ``values.reshape((_n_image, num_ao_prim // _n_image), ...)`` in
            the image-tiled fast path of ``_reduce_primitives_to_aos`` raises
            ``cannot reshape ...`` for such geometries. ``seed=39`` deterministically produces a non-uniform leaf
            (``_n_image=14``, kernel ``num_ao_prim=22``), pinning the gather fallback.
            """
            atol, rtol = get_tolerance("jastrow_eval", "strict")
            np.random.seed(39)
            num_el, num_mo, num_ao, num_ao_prim = 10, 5, 3, 4
            orbital_indices = (0, 0, 1, 2)
            exponents = np.array([1.5, 1.0, 0.8, 0.6], dtype=np.float64)
            coefficients = np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float64)
            angular_momentums = (1, 1, 1)
            magnetic_quantum_numbers = (0, 0, -1)

            r_up_carts = 10.0 * np.random.rand(num_el, 3)
            r_dn_carts = 10.0 * np.random.rand(num_el, 3)
            R_carts = 2.0 + 6.0 * np.random.rand(num_ao, 3)
            mo_coefficients = np.random.rand(num_mo, num_ao)

            structure_data = _make_pbc_structure(R_carts, atomic_number=1)
            aos_pbc_mol = AOs_sphe_gto_data._from_full_basis(
                structure_data=structure_data,
                nucleus_index=tuple(range(num_ao)),
                num_ao=num_ao,
                num_ao_prim=num_ao_prim,
                orbital_indices=orbital_indices,
                exponents=exponents,
                coefficients=coefficients,
                angular_momentums=angular_momentums,
                magnetic_quantum_numbers=magnetic_quantum_numbers,
            )
            expanded_aos = _apply_image_expansion(aos_pbc_mol, structure_data, tol=AO_IMAGE_TRUNCATION_ATOL)
            # Guard the premise: this geometry MUST be non-uniform, else the
            # regression would silently exercise only the uniform fast path.
            assert expanded_aos.num_ao_prim % expanded_aos._n_image != 0, (
                f"expected non-uniform ragged leaf; got num_ao_prim={expanded_aos.num_ao_prim}, "
                f"_n_image={expanded_aos._n_image}"
            )

            mos_data = MOs_data(num_mo=num_mo, aos_data=expanded_aos, mo_coefficients=mo_coefficients)
            j_matrix = np.random.rand(mos_data.num_mo, mos_data.num_mo + 1)
            j_matrix[:, :-1] = 0.5 * (j_matrix[:, :-1] + j_matrix[:, :-1].T)
            jastrow_three_body_data = Jastrow_three_body_data(orb_data=mos_data, j_matrix=j_matrix)

            J3_debug = _compute_Jastrow_three_body_debug(
                jastrow_three_body_data=jastrow_three_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
            )
            J3_jax = compute_Jastrow_three_body(
                jastrow_three_body_data=jastrow_three_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
            )
            assert not np.any(np.isnan(np.asarray(J3_jax))), "NaN in production J3"
            np.testing.assert_allclose(J3_debug, J3_jax, atol=atol, rtol=rtol)
            jax.clear_caches()

        @pytest.mark.heavy
        def test_analytical_and_auto_grads_and_laplacian_Jastrow_threebody_part_with_MOs_data_molecular(self):
            """Analytic vs auto-diff gradients/laplacian for three-body Jastrow (MOs)."""

            # J3-with-MOs crosses jastrow_grad_lap (fp32 mixed) + ao_grad_lap + mo_grad + mo_lap.
            # All non-jastrow zones are fp64; jastrow_grad_lap dominates as the loosest.
            atol, rtol = get_tolerance_min(["jastrow_grad_lap", "ao_grad_lap", "mo_grad", "mo_lap"], "strict")

            def case_factory():
                fixture_rng = np.random.RandomState(20260923)
                num_el = 8
                num_mo = 4
                num_ao = 3
                num_ao_prim = 4
                orbital_indices = [0, 0, 1, 2]
                exponents = [10.0, 6.0, 3.0, 1.5]
                coefficients = [1.0, 1.0, 0.8, 0.7]
                angular_momentums = [1, 1, 1]
                magnetic_quantum_numbers = [0, 1, -1]

                r_cart_min, r_cart_max = -2.0, 2.0
                R_cart_min, R_cart_max = -1.0, 1.0
                r_up_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_el, 3) + r_cart_min
                r_dn_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_el, 3) + r_cart_min
                R_carts = (R_cart_max - R_cart_min) * fixture_rng.rand(num_ao, 3) + R_cart_min

                mo_coefficients = fixture_rng.rand(num_mo, num_ao)

                structure_data = _make_structure(R_carts)

                aos_data = AOs_sphe_gto_data._from_full_basis(
                    structure_data=structure_data,
                    nucleus_index=tuple(list(range(num_ao))),
                    num_ao=num_ao,
                    num_ao_prim=num_ao_prim,
                    orbital_indices=tuple(orbital_indices),
                    exponents=np.array(exponents, dtype=np.float64),
                    coefficients=np.array(coefficients, dtype=np.float64),
                    angular_momentums=tuple(angular_momentums),
                    magnetic_quantum_numbers=tuple(magnetic_quantum_numbers),
                )

                mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)

                j_matrix = fixture_rng.rand(mos_data.num_mo, mos_data.num_mo + 1)
                j_matrix[:, :-1] = 0.5 * (j_matrix[:, :-1] + j_matrix[:, :-1].T)  # symmetric J3 (bosonic)
                jastrow_three_body_data = Jastrow_three_body_data(orb_data=mos_data, j_matrix=j_matrix)

                return (jastrow_three_body_data, r_up_carts, r_dn_carts)

            def analytical(case):
                jastrow_three_body_data, r_up_carts, r_dn_carts = case
                return compute_grads_and_laplacian_Jastrow_three_body(jastrow_three_body_data, r_up_carts, r_dn_carts)

            if is_mixed_precision_enabled():
                mixed, full = mixed_vs_full(case_factory, analytical)
                blocks = mixed_fidelity_blocks(mixed, full)
                mixed_atol, mixed_rtol = get_tolerance_mixed_fidelity("parameter_derivative")
                for mixed_block, full_block in blocks:
                    _assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, mixed_atol, mixed_rtol)
                mutated = out_of_budget_mixed_block(*blocks[0], atol=mixed_atol, rtol=mixed_rtol)
                with pytest.raises(AssertionError):
                    _assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], mixed_atol, mixed_rtol)
                jax.clear_caches()
                return

            jastrow_three_body_data, r_up_carts, r_dn_carts = case_factory()
            grad_up_an, grad_dn_an, lap_up_an, lap_dn_an = compute_grads_and_laplacian_Jastrow_three_body(
                jastrow_three_body_data,
                r_up_carts,
                r_dn_carts,
            )

            grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_three_body_auto(
                jastrow_three_body_data,
                r_up_carts,
                r_dn_carts,
            )

            assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(grad_up_an), np.asarray(grad_up_auto), atol=atol, rtol=rtol)
            assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(grad_dn_an), np.asarray(grad_dn_auto), atol=atol, rtol=rtol)
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(lap_up_an), np.asarray(lap_up_auto), atol=atol, rtol=rtol)
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(lap_dn_an), np.asarray(lap_dn_auto), atol=atol, rtol=rtol)

            jax.clear_caches()

        @pytest.mark.heavy
        def test_analytical_and_auto_grads_and_laplacian_Jastrow_threebody_part_with_MOs_data_periodic(self):
            """PBC sibling of ``..._threebody_part_with_MOs_data_molecular`` (analytical vs auto)."""

            atol, rtol = get_tolerance_min(["jastrow_grad_lap", "ao_grad_lap", "mo_grad", "mo_lap"], "strict")

            def case_factory():
                fixture_rng = np.random.RandomState(20260923)
                num_el = 8
                num_mo = 4
                num_ao = 3
                num_ao_prim = 4
                orbital_indices = tuple([0, 0, 1, 2])
                exponents = np.array([1.5, 1.0, 0.8, 0.6], dtype=np.float64)
                coefficients = np.array([1.0, 1.0, 0.8, 0.7], dtype=np.float64)
                angular_momentums = tuple([1, 1, 1])
                magnetic_quantum_numbers = tuple([0, 1, -1])

                r_up_carts = 10.0 * fixture_rng.rand(num_el, 3)
                r_dn_carts = 10.0 * fixture_rng.rand(num_el, 3)
                R_carts = 2.0 + 6.0 * fixture_rng.rand(num_ao, 3)

                mo_coefficients = fixture_rng.rand(num_mo, num_ao)

                structure_data = _make_pbc_structure(R_carts, atomic_number=1)

                aos_pbc_mol = AOs_sphe_gto_data._from_full_basis(
                    structure_data=structure_data,
                    nucleus_index=tuple(list(range(num_ao))),
                    num_ao=num_ao,
                    num_ao_prim=num_ao_prim,
                    orbital_indices=orbital_indices,
                    exponents=exponents,
                    coefficients=coefficients,
                    angular_momentums=angular_momentums,
                    magnetic_quantum_numbers=magnetic_quantum_numbers,
                )

                expanded_aos = _apply_image_expansion(aos_pbc_mol, structure_data, tol=AO_IMAGE_TRUNCATION_ATOL)

                mos_data = MOs_data(num_mo=num_mo, aos_data=expanded_aos, mo_coefficients=mo_coefficients)

                j_matrix = fixture_rng.rand(mos_data.num_mo, mos_data.num_mo + 1)
                j_matrix[:, :-1] = 0.5 * (j_matrix[:, :-1] + j_matrix[:, :-1].T)  # symmetric J3 (bosonic)
                jastrow_three_body_data = Jastrow_three_body_data(orb_data=mos_data, j_matrix=j_matrix)

                return (jastrow_three_body_data, r_up_carts, r_dn_carts)

            def analytical(case):
                jastrow_three_body_data, r_up_carts, r_dn_carts = case
                return compute_grads_and_laplacian_Jastrow_three_body(jastrow_three_body_data, r_up_carts, r_dn_carts)

            if is_mixed_precision_enabled():
                mixed, full = mixed_vs_full(case_factory, analytical)
                blocks = mixed_fidelity_blocks(mixed, full)
                mixed_atol, mixed_rtol = get_tolerance_mixed_fidelity("parameter_derivative")
                for mixed_block, full_block in blocks:
                    _assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, mixed_atol, mixed_rtol)
                mutated = out_of_budget_mixed_block(*blocks[0], atol=mixed_atol, rtol=mixed_rtol)
                with pytest.raises(AssertionError):
                    _assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], mixed_atol, mixed_rtol)
                jax.clear_caches()
                return

            jastrow_three_body_data, r_up_carts, r_dn_carts = case_factory()
            grad_up_an, grad_dn_an, lap_up_an, lap_dn_an = compute_grads_and_laplacian_Jastrow_three_body(
                jastrow_three_body_data,
                r_up_carts,
                r_dn_carts,
            )

            grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_three_body_auto(
                jastrow_three_body_data,
                r_up_carts,
                r_dn_carts,
            )

            assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(grad_up_an), np.asarray(grad_up_auto), atol=atol, rtol=rtol)
            assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(grad_dn_an), np.asarray(grad_dn_auto), atol=atol, rtol=rtol)
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(lap_up_an), np.asarray(lap_up_auto), atol=atol, rtol=rtol)
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_an)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_auto)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(lap_dn_an), np.asarray(lap_dn_auto), atol=atol, rtol=rtol)

            jax.clear_caches()

    class TestBasisConversion:
        @pytest.mark.light
        def test_periodic_gamma_j3_leaf_conversion_for_complex_twist_companion(self):
            """The real Gamma J3 leaf converts independently of the complex SD leaf."""
            fixture_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")
            gamma_aos = read_trexio_file(
                os.path.join(fixture_dir, "lithium.h5"),
                store_tuple=True,
            )[1]
            complex_sd = read_trexio_file(
                os.path.join(fixture_dir, "lithium_k025.h5"),
                store_tuple=True,
            )[4]
            assert max(gamma_aos.angular_momentums) == 2
            assert gamma_aos.k_point is None
            assert complex_sd.orb_data_up_spin.aos_data.k_point is not None
            np.testing.assert_array_equal(
                np.asarray(gamma_aos.structure_data.cell),
                np.asarray(complex_sd.orb_data_up_spin.aos_data.structure_data.cell),
            )

            rng = np.random.default_rng(1204)
            j_matrix = rng.normal(scale=0.15, size=(gamma_aos.num_ao, gamma_aos.num_ao + 1))
            j_matrix[:, :-1] = 0.5 * (j_matrix[:, :-1] + j_matrix[:, :-1].T)
            j3_spherical = Jastrow_three_body_data(orb_data=gamma_aos, j_matrix=j_matrix)
            j3_cartesian = j3_spherical.to_cartesian()
            j3_roundtrip = j3_cartesian.to_spherical()

            cell = np.asarray(gamma_aos.structure_data.cell)
            r_up = np.array([[0.13, 0.27, 0.41], [1.07, -0.19, 0.33]], dtype=np.float64) @ cell
            r_dn = np.array([[-0.22, 0.58, 1.14]], dtype=np.float64) @ cell
            reference_value = np.asarray(compute_Jastrow_three_body(j3_spherical, r_up, r_dn))
            reference_derivatives = tuple(
                np.asarray(x) for x in compute_grads_and_laplacian_Jastrow_three_body(j3_spherical, r_up, r_dn)
            )
            atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap", "jastrow_eval", "jastrow_grad_lap"), "strict")

            for converted in (j3_cartesian, j3_roundtrip):
                assert converted.orb_data.k_point is None
                np.testing.assert_allclose(
                    np.asarray(compute_Jastrow_three_body(converted, r_up, r_dn)),
                    reference_value,
                    atol=atol,
                    rtol=rtol,
                )
                actual_derivatives = compute_grads_and_laplacian_Jastrow_three_body(converted, r_up, r_dn)
                for actual, expected in zip(actual_derivatives, reference_derivatives, strict=True):
                    np.testing.assert_allclose(np.asarray(actual), expected, atol=atol, rtol=rtol)

        @pytest.mark.heavy
        def test_Jastrow_threebody_part_sphe_to_cart_AOs_data(self):
            """Round-trip AOs l<=6: spherical->Cartesian keeps J3 values/grads."""
            atol, rtol = get_tolerance("jastrow_eval", "strict")
            rng = np.random.default_rng(321)

            aos_sphe = _make_shell_aos(rng)

            j_matrix = rng.uniform(-0.3, 0.3, size=(aos_sphe.num_ao, aos_sphe.num_ao + 1))
            j3_sph = Jastrow_three_body_data(orb_data=aos_sphe, j_matrix=j_matrix)
            j3_cart = j3_sph.to_cartesian()

            r_up_carts = rng.uniform(-1.0, 1.0, size=(4, 3))
            r_dn_carts = rng.uniform(-1.0, 1.0, size=(3, 3))

            J_sph = compute_Jastrow_three_body(j3_sph, r_up_carts, r_dn_carts)
            J_cart = compute_Jastrow_three_body(j3_cart, r_up_carts, r_dn_carts)

            grads_sph = compute_grads_and_laplacian_Jastrow_three_body(j3_sph, r_up_carts, r_dn_carts)
            grads_cart = compute_grads_and_laplacian_Jastrow_three_body(j3_cart, r_up_carts, r_dn_carts)

            assert not np.any(np.isnan(np.asarray(np.asarray(J_sph)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(J_cart)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(J_sph), np.asarray(J_cart), atol=atol, rtol=rtol)
            for channel, sph, cart in zip(("grad_up", "grad_dn", "lap_up", "lap_dn"), grads_sph, grads_cart, strict=True):
                assert not np.any(np.isnan(np.asarray(np.asarray(sph)))), "NaN detected in first argument"
                assert not np.any(np.isnan(np.asarray(np.asarray(cart)))), "NaN detected in second argument"
                np.testing.assert_allclose(np.asarray(sph), np.asarray(cart), atol=atol, rtol=rtol)

            jax.clear_caches()

        @pytest.mark.heavy
        def test_Jastrow_threebody_part_cart_to_sphe_AOs_data(self):
            """Round-trip AOs l<=6: Cartesian->spherical keeps J3 values/grads."""
            atol, rtol = get_tolerance("jastrow_eval", "strict")
            rng = np.random.default_rng(654)

            aos_sphe = _make_shell_aos(rng)

            j_matrix = rng.uniform(-0.3, 0.3, size=(aos_sphe.num_ao, aos_sphe.num_ao + 1))
            j3_cart = Jastrow_three_body_data(orb_data=aos_sphe, j_matrix=j_matrix).to_cartesian()
            j3_cart_to_sph = j3_cart.to_spherical()

            r_up_carts = rng.uniform(-1.0, 1.0, size=(5, 3))
            r_dn_carts = rng.uniform(-1.0, 1.0, size=(4, 3))

            J_cart = compute_Jastrow_three_body(j3_cart, r_up_carts, r_dn_carts)
            J_sph = compute_Jastrow_three_body(j3_cart_to_sph, r_up_carts, r_dn_carts)

            grads_cart = compute_grads_and_laplacian_Jastrow_three_body(j3_cart, r_up_carts, r_dn_carts)
            grads_sph = compute_grads_and_laplacian_Jastrow_three_body(j3_cart_to_sph, r_up_carts, r_dn_carts)

            assert not np.any(np.isnan(np.asarray(np.asarray(J_cart)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(J_sph)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(J_cart), np.asarray(J_sph), atol=atol, rtol=rtol)
            for channel, cart, sph in zip(("grad_up", "grad_dn", "lap_up", "lap_dn"), grads_cart, grads_sph, strict=True):
                assert not np.any(np.isnan(np.asarray(np.asarray(cart)))), "NaN detected in first argument"
                assert not np.any(np.isnan(np.asarray(np.asarray(sph)))), "NaN detected in second argument"
                np.testing.assert_allclose(np.asarray(cart), np.asarray(sph), atol=atol, rtol=rtol)

            jax.clear_caches()

        @pytest.mark.heavy
        def test_Jastrow_threebody_part_sphe_to_cart_MOs_data(self):
            """Round-trip MOs built on l<=6 AOs: spherical->Cartesian keeps J3 values/grads."""
            atol, rtol = get_tolerance("jastrow_eval", "strict")
            rng = np.random.default_rng(777)

            aos_sphe = _make_shell_aos(rng)
            num_ao = aos_sphe.num_ao
            num_mo = max(8, num_ao // 2)

            mo_coefficients = rng.uniform(-0.5, 0.5, size=(num_mo, num_ao))
            mos_sphe = MOs_data(num_mo=num_mo, aos_data=aos_sphe, mo_coefficients=mo_coefficients)
            mos_sphe.sanity_check()

            j_matrix = rng.uniform(-0.2, 0.2, size=(mos_sphe.num_mo, mos_sphe.num_mo + 1))
            j3_sph = Jastrow_three_body_data(orb_data=mos_sphe, j_matrix=j_matrix)
            j3_cart = j3_sph.to_cartesian()

            r_up_carts = rng.uniform(-1.0, 1.0, size=(5, 3))
            r_dn_carts = rng.uniform(-1.0, 1.0, size=(4, 3))

            J_sph = compute_Jastrow_three_body(j3_sph, r_up_carts, r_dn_carts)
            J_cart = compute_Jastrow_three_body(j3_cart, r_up_carts, r_dn_carts)

            grads_sph = compute_grads_and_laplacian_Jastrow_three_body(j3_sph, r_up_carts, r_dn_carts)
            grads_cart = compute_grads_and_laplacian_Jastrow_three_body(j3_cart, r_up_carts, r_dn_carts)

            assert not np.any(np.isnan(np.asarray(np.asarray(J_sph)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(J_cart)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(J_sph), np.asarray(J_cart), atol=atol, rtol=rtol)
            for channel, sph, cart in zip(("grad_up", "grad_dn", "lap_up", "lap_dn"), grads_sph, grads_cart, strict=True):
                assert not np.any(np.isnan(np.asarray(np.asarray(sph)))), "NaN detected in first argument"
                assert not np.any(np.isnan(np.asarray(np.asarray(cart)))), "NaN detected in second argument"
                np.testing.assert_allclose(np.asarray(sph), np.asarray(cart), atol=atol, rtol=rtol)

            jax.clear_caches()

        @pytest.mark.heavy
        def test_Jastrow_threebody_part_cart_to_sphe_MOs_data(self):
            """Round-trip MOs l<=6: Cartesian->spherical keeps J3 values/grads."""
            atol, rtol = get_tolerance("jastrow_eval", "strict")
            rng = np.random.default_rng(888)

            aos_sphe = _make_shell_aos(rng)
            num_ao = aos_sphe.num_ao
            num_mo = max(8, num_ao // 2)

            mo_coefficients = rng.uniform(-0.5, 0.5, size=(num_mo, num_ao))
            mos_sphe = MOs_data(num_mo=num_mo, aos_data=aos_sphe, mo_coefficients=mo_coefficients)
            mos_sphe.sanity_check()

            j_matrix = rng.uniform(-0.2, 0.2, size=(mos_sphe.num_mo, mos_sphe.num_mo + 1))
            j3_cart = Jastrow_three_body_data(orb_data=mos_sphe, j_matrix=j_matrix).to_cartesian()
            j3_cart_to_sph = j3_cart.to_spherical()

            r_up_carts = rng.uniform(-1.0, 1.0, size=(6, 3))
            r_dn_carts = rng.uniform(-1.0, 1.0, size=(5, 3))

            J_cart = compute_Jastrow_three_body(j3_cart, r_up_carts, r_dn_carts)
            J_sph = compute_Jastrow_three_body(j3_cart_to_sph, r_up_carts, r_dn_carts)

            grads_cart = compute_grads_and_laplacian_Jastrow_three_body(j3_cart, r_up_carts, r_dn_carts)
            grads_sph = compute_grads_and_laplacian_Jastrow_three_body(j3_cart_to_sph, r_up_carts, r_dn_carts)

            assert not np.any(np.isnan(np.asarray(np.asarray(J_cart)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(J_sph)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(J_cart), np.asarray(J_sph), atol=atol, rtol=rtol)
            for channel, cart, sph in zip(("grad_up", "grad_dn", "lap_up", "lap_dn"), grads_cart, grads_sph, strict=True):
                assert not np.any(np.isnan(np.asarray(np.asarray(cart)))), "NaN detected in first argument"
                assert not np.any(np.isnan(np.asarray(np.asarray(sph)))), "NaN detected in second argument"
                np.testing.assert_allclose(np.asarray(cart), np.asarray(sph), atol=atol, rtol=rtol)

            jax.clear_caches()


# ---------------------------------------------------------------------------
# Jastrow combination matrix for "part" tests (J1+J2+J3).
# Each tuple is (j1b_type, j2b_type, include_nn). The builders ignore include_nn;
# its True row repeats a J1/J2 choice and does not exercise an NN component.
# ---------------------------------------------------------------------------
_JASTROW_COMBOS = [
    ("exp", "pade", False),  # default
    ("pade", "pade", False),  # J1 pade
    ("exp", "exp", False),  # J2 exp
    ("pade", "exp", False),  # both non-default
    ("exp", "pade", True),  # same components; retained unused parameter label
]


def _build_jastrow_data_for_part_tests(j1b_type: str = "exp", j2b_type: str = "pade", include_nn: bool = False):
    """Build molecular J1+J2+J3; include_nn is an unused parameter label."""
    fixture_rng = np.random.RandomState(20260923)
    num_r_up_cart_samples = 4
    num_r_dn_cart_samples = 3
    num_R_cart_samples = 4
    num_ao = 4
    num_ao_prim = 4

    r_cart_min, r_cart_max = -1.0, 1.0
    R_cart_min, R_cart_max = 0.0, 0.0
    r_up_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_up_cart_samples, 3) + r_cart_min
    r_dn_carts = (r_cart_max - r_cart_min) * fixture_rng.rand(num_r_dn_cart_samples, 3) + r_cart_min
    R_carts = (R_cart_max - R_cart_min) * fixture_rng.rand(num_R_cart_samples, 3) + R_cart_min

    structure_data = _make_structure(R_carts, atomic_number=6)

    core_electrons = tuple([3] * num_R_cart_samples)
    jastrow_one_body_data = Jastrow_one_body_data(
        jastrow_1b_param=1.0, jastrow_1b_type=j1b_type, structure_data=structure_data, core_electrons=core_electrons
    )

    jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type=j2b_type)

    orbital_indices = (0, 1, 2, 3)
    exponents = (1.2, 0.5, 0.1, 0.05)
    coefficients = (1.0, 1.0, 1.0, 1.0)
    angular_momentums = (0, 0, 1, 1)
    magnetic_quantum_numbers = (0, 0, 1, -1)

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

    j_matrix = fixture_rng.rand(aos_data.num_ao, aos_data.num_ao + 1)
    j_matrix[:, :-1] = 0.5 * (j_matrix[:, :-1] + j_matrix[:, :-1].T)  # symmetric J3 (bosonic)
    jastrow_three_body_data = Jastrow_three_body_data(orb_data=aos_data, j_matrix=j_matrix)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_one_body_data,
        jastrow_two_body_data=jastrow_two_body_data,
        jastrow_three_body_data=jastrow_three_body_data,
    )

    return jastrow_data, r_up_carts, r_dn_carts


def _build_jastrow_data_for_part_tests_periodic(j1b_type: str = "exp", j2b_type: str = "pade", include_nn: bool = False):
    """Build cubic 10-Bohr J1+J2+J3; include_nn is an unused parameter label."""
    fixture_rng = np.random.RandomState(20260923)
    num_r_up_cart_samples = 4
    num_r_dn_cart_samples = 3
    num_R_cart_samples = 4
    num_ao = 4
    num_ao_prim = 4

    r_up_carts = 10.0 * fixture_rng.rand(num_r_up_cart_samples, 3)
    r_dn_carts = 10.0 * fixture_rng.rand(num_r_dn_cart_samples, 3)
    R_carts = 2.0 + 6.0 * fixture_rng.rand(num_R_cart_samples, 3)

    structure_data = _make_pbc_structure(R_carts, atomic_number=6)

    core_electrons = tuple([3] * num_R_cart_samples)
    jastrow_one_body_data = Jastrow_one_body_data(
        jastrow_1b_param=1.0, jastrow_1b_type=j1b_type, structure_data=structure_data, core_electrons=core_electrons
    )

    jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type=j2b_type, structure_data=structure_data)

    orbital_indices = (0, 1, 2, 3)
    exponents = (1.2, 1.0, 0.8, 0.6)
    coefficients = (1.0, 1.0, 1.0, 1.0)
    angular_momentums = (0, 0, 1, 1)
    magnetic_quantum_numbers = (0, 0, 1, -1)

    aos_pbc_mol = AOs_sphe_gto_data._from_full_basis(
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

    j_matrix = fixture_rng.rand(aos_pbc_mol.num_ao, aos_pbc_mol.num_ao + 1)
    j_matrix[:, :-1] = 0.5 * (j_matrix[:, :-1] + j_matrix[:, :-1].T)  # symmetric J3 (bosonic)
    expanded_aos = _apply_image_expansion(aos_pbc_mol, structure_data, tol=AO_IMAGE_TRUNCATION_ATOL)
    jastrow_three_body_data = Jastrow_three_body_data(orb_data=expanded_aos, j_matrix=j_matrix)

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_one_body_data,
        jastrow_two_body_data=jastrow_two_body_data,
        jastrow_three_body_data=jastrow_three_body_data,
    )

    return jastrow_data, r_up_carts, r_dn_carts


class TestJastrowCombined:
    @pytest.mark.heavy
    @pytest.mark.parametrize("j1b_type,j2b_type,include_nn", _JASTROW_COMBOS)
    def test_analytical_and_auto_grads_and_laplacian_Jastrow_part_molecular(self, j1b_type, j2b_type, include_nn):
        """Compare molecular J1+J2+J3 gradients/Laplacians with autodiff.

        The five selectors cover four J1/J2 combinations. include_nn is unused
        by the builder: its True row is not neural-network coverage.
        """

        # Combined J1+J2+J3 grad/lap crosses jastrow_grad_lap (fp32 mixed) and the
        # AO/MO grad/lap zones via the J3 path. jastrow_grad_lap is the loosest under mixed.
        atol, rtol = get_tolerance_min(["jastrow_grad_lap", "ao_grad_lap", "mo_grad", "mo_lap"], "strict")

        def case_factory():
            jastrow_data, r_up_carts, r_dn_carts = _build_jastrow_data_for_part_tests(j1b_type, j2b_type, include_nn)

            return (jastrow_data, r_up_carts, r_dn_carts)

        def analytical(case):
            jastrow_data, r_up_carts, r_dn_carts = case
            return compute_grads_and_laplacian_Jastrow_part(jastrow_data, r_up_carts, r_dn_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            mixed_atol, mixed_rtol = get_tolerance_mixed_fidelity("parameter_derivative")
            for mixed_block, full_block in blocks:
                _assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, mixed_atol, mixed_rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=mixed_atol, rtol=mixed_rtol)
            with pytest.raises(AssertionError):
                _assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], mixed_atol, mixed_rtol)
            jax.clear_caches()
            return

        jastrow_data, r_up_carts, r_dn_carts = case_factory()
        grad_up_an, grad_dn_an, lap_up_an, lap_dn_an = compute_grads_and_laplacian_Jastrow_part(
            jastrow_data,
            r_up_carts,
            r_dn_carts,
        )

        grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_part_auto(
            jastrow_data,
            r_up_carts,
            r_dn_carts,
        )

        assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(grad_up_an), np.asarray(grad_up_auto), atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(grad_dn_an), np.asarray(grad_dn_auto), atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(lap_up_an), np.asarray(lap_up_auto), atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(lap_dn_an), np.asarray(lap_dn_auto), atol=atol, rtol=rtol)

        jax.clear_caches()

    @pytest.mark.heavy
    @pytest.mark.parametrize("j1b_type,j2b_type,include_nn", _JASTROW_COMBOS)
    def test_analytical_and_auto_grads_and_laplacian_Jastrow_part_periodic(self, j1b_type, j2b_type, include_nn):
        """Compare cubic 10-Bohr J1+J2+J3 gradients/Laplacians with autodiff.

        The five selectors cover four J1/J2 combinations. include_nn is unused
        by the builder: its True row is not neural-network coverage.
        """

        atol, rtol = get_tolerance_min(["jastrow_grad_lap", "ao_grad_lap", "mo_grad", "mo_lap"], "strict")

        def case_factory():
            jastrow_data, r_up_carts, r_dn_carts = _build_jastrow_data_for_part_tests_periodic(j1b_type, j2b_type, include_nn)

            return (jastrow_data, r_up_carts, r_dn_carts)

        def analytical(case):
            jastrow_data, r_up_carts, r_dn_carts = case
            return compute_grads_and_laplacian_Jastrow_part(jastrow_data, r_up_carts, r_dn_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            mixed_atol, mixed_rtol = get_tolerance_mixed_fidelity("parameter_derivative")
            for mixed_block, full_block in blocks:
                _assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, mixed_atol, mixed_rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=mixed_atol, rtol=mixed_rtol)
            with pytest.raises(AssertionError):
                _assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], mixed_atol, mixed_rtol)
            jax.clear_caches()
            return

        jastrow_data, r_up_carts, r_dn_carts = case_factory()
        grad_up_an, grad_dn_an, lap_up_an, lap_dn_an = compute_grads_and_laplacian_Jastrow_part(
            jastrow_data,
            r_up_carts,
            r_dn_carts,
        )

        grad_up_auto, grad_dn_auto, lap_up_auto, lap_dn_auto = _compute_grads_and_laplacian_Jastrow_part_auto(
            jastrow_data,
            r_up_carts,
            r_dn_carts,
        )

        assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_up_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(grad_up_an), np.asarray(grad_up_auto), atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(grad_dn_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(grad_dn_an), np.asarray(grad_dn_auto), atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_up_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(lap_up_an), np.asarray(lap_up_auto), atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_an)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(lap_dn_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(lap_dn_an), np.asarray(lap_dn_auto), atol=atol, rtol=rtol)

        jax.clear_caches()


# Hand-picked covering set replacing the 5x3x3x2 = 90-case full Cartesian
# product. The default Jastrow combo gets a full (pattern, n_grid,
# use_j3_state) sweep because the rank-1 update + J3 streaming-cache logic
# lives outside the J1/J2 function paths -- so the other four Jastrow
# combos only need representative configs to confirm those paths still
# integrate. ``mixed`` + ``n_grid=1`` is omitted because the test body's
# ``if n_grid >= 3`` branch does not fire there, making it a duplicate of
# ``all_moved`` + ``n_grid=1``. All single-value coverage and the
# (n_grid x use_j3_state), (pattern x use_j3_state), and valid
# (pattern x n_grid) pairs are preserved.
_RATIO_RANK1_PARAM_SETS = [
    # (j1b_type, j2b_type, include_nn, pattern, n_grid, use_j3_state)
    # --- default Jastrow combo: full (pattern, n_grid, use_j3_state) sweep ---
    ("exp", "pade", False, "all_moved", 1, False),
    ("exp", "pade", False, "all_moved", 1, True),
    ("exp", "pade", False, "all_moved", 3, False),
    ("exp", "pade", False, "all_moved", 3, True),
    ("exp", "pade", False, "all_moved", 6, False),
    ("exp", "pade", False, "all_moved", 6, True),
    ("exp", "pade", False, "none_moved", 1, False),
    ("exp", "pade", False, "none_moved", 6, True),
    ("exp", "pade", False, "mixed", 3, False),
    ("exp", "pade", False, "mixed", 3, True),
    ("exp", "pade", False, "mixed", 6, False),
    ("exp", "pade", False, "mixed", 6, True),
    # --- other Jastrow combos: representative configs ---
    ("pade", "pade", False, "all_moved", 3, True),
    ("pade", "pade", False, "none_moved", 1, False),
    ("exp", "exp", False, "all_moved", 6, False),
    ("exp", "exp", False, "mixed", 6, True),
    ("pade", "exp", False, "all_moved", 1, True),
    ("pade", "exp", False, "mixed", 3, False),
    ("exp", "pade", True, "all_moved", 1, False),
    ("exp", "pade", True, "none_moved", 3, True),
    ("exp", "pade", True, "mixed", 6, True),
]


class TestJastrowRank1FastUpdate:
    @pytest.mark.heavy
    @pytest.mark.parametrize(
        "j1b_type,j2b_type,include_nn,pattern,n_grid,use_j3_state",
        _RATIO_RANK1_PARAM_SETS,
    )
    def test_ratio_Jastrow_part_rank1_update_molecular(
        self, j1b_type, j2b_type, include_nn, pattern: str, n_grid: int, use_j3_state: bool
    ):
        """Compare ratio Jastrow part: debug vs rank-1 update implementation.

        Sweeps ``n_grid`` to cover the two production regimes:
          * ``n_grid = 1`` -- MCMC walker-update path. Exercises the J2 rewrite
            that replaced the (N-1)/N-wasted O(N^2) baseline with on-demand
            ``_batch_pairwise_sum`` calls.
          * ``n_grid = 3, 6`` -- representative ECP nonlocal mesh-sized paths.
            Confirms the rewrite is numerically equivalent when N_grid is large
            enough that the old baseline was fully amortized.

        ``use_j3_state`` toggles the J3 streaming-cache path inside the rank-1
        update (``j3_state=None`` -> recompute aos / j3_mat @ aos every call;
        ``j3_state=init(...)`` -> read from cache). Both must agree with the
        debug reference; this is the regression guard for the MCMC walker
        update integration of the cache.
        """
        # Both _compute_ratio_Jastrow_part_rank1_update and _compute_ratio_Jastrow_part_debug
        # operate in the jastrow_ratio zone (J(R')/J(R) log-ratio). Use that zone's tolerance
        # to honor the 1-zone-1-module principle.
        atol, rtol = get_tolerance("jastrow_ratio", "strict")
        np.random.seed(0)
        jastrow_data, old_r_up_carts, old_r_dn_carts = _build_jastrow_data_for_part_tests(j1b_type, j2b_type, include_nn)

        new_r_up_carts_arr = np.repeat(old_r_up_carts[None, ...], n_grid, axis=0)
        new_r_dn_carts_arr = np.repeat(old_r_dn_carts[None, ...], n_grid, axis=0)

        up_alt_idx = min(1, old_r_up_carts.shape[0] - 1)
        dn_alt_idx = min(1, old_r_dn_carts.shape[0] - 1)

        if pattern == "all_moved":
            # Every grid moves exactly one electron; rotate (spin, index, axis)
            # mod 3 to cover up_case/dn_case selection and alternate indices.
            for g in range(n_grid):
                r = g % 3
                if r == 0:
                    new_r_up_carts_arr[g, 0, 0] += 0.05
                elif r == 1:
                    new_r_dn_carts_arr[g, 0, 1] -= 0.07
                else:
                    new_r_up_carts_arr[g, up_alt_idx, 2] += 0.09
        elif pattern == "none_moved":
            pass
        elif pattern == "mixed":
            # Move on first grid; if n_grid >= 3 also move on the last grid (the
            # interior grids stay put -- their ratio must be exactly 1.0).
            new_r_up_carts_arr[0, 0, 0] += 0.05
            if n_grid >= 3:
                new_r_dn_carts_arr[n_grid - 1, dn_alt_idx, 1] += 0.04

        ratio_debug = _compute_ratio_Jastrow_part_debug(
            jastrow_data,
            old_r_up_carts=old_r_up_carts,
            old_r_dn_carts=old_r_dn_carts,
            new_r_up_carts_arr=new_r_up_carts_arr,
            new_r_dn_carts_arr=new_r_dn_carts_arr,
        )

        # Build the J3 streaming cache from the OLD configuration when requested.
        # ``_build_jastrow_data_for_part_tests`` always includes a J3 component,
        # so the cache is always constructible here.
        j3_state = None
        if use_j3_state:
            j3_state = _init_grads_laplacian_Jastrow_three_body_streaming_state(
                jastrow_data.jastrow_three_body_data,
                old_r_up_carts,
                old_r_dn_carts,
            )

        ratio_auto = _compute_ratio_Jastrow_part_rank1_update(
            jastrow_data,
            old_r_up_carts=old_r_up_carts,
            old_r_dn_carts=old_r_dn_carts,
            new_r_up_carts_arr=new_r_up_carts_arr,
            new_r_dn_carts_arr=new_r_dn_carts_arr,
            j3_state=j3_state,
        )

        assert ratio_auto.shape == (n_grid,)
        assert not np.any(np.isnan(np.asarray(np.asarray(ratio_debug)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(ratio_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(ratio_debug), np.asarray(ratio_auto), atol=atol, rtol=rtol)

        if pattern == "none_moved":
            assert not np.any(np.isnan(np.asarray(np.asarray(ratio_debug)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.ones_like(np.asarray(ratio_debug))))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(ratio_debug), np.ones_like(np.asarray(ratio_debug)), atol=atol, rtol=rtol)

        jax.clear_caches()

    @pytest.mark.heavy
    @pytest.mark.parametrize(
        "j1b_type,j2b_type,include_nn,pattern,n_grid,use_j3_state",
        _RATIO_RANK1_PARAM_SETS,
    )
    def test_ratio_Jastrow_part_rank1_update_periodic(
        self, j1b_type, j2b_type, include_nn, pattern: str, n_grid: int, use_j3_state: bool
    ):
        """PBC sibling of ``test_ratio_Jastrow_part_rank1_update_molecular``."""
        atol, rtol = get_tolerance("jastrow_ratio", "strict")
        np.random.seed(0)
        jastrow_data, old_r_up_carts, old_r_dn_carts = _build_jastrow_data_for_part_tests_periodic(
            j1b_type, j2b_type, include_nn
        )

        new_r_up_carts_arr = np.repeat(old_r_up_carts[None, ...], n_grid, axis=0)
        new_r_dn_carts_arr = np.repeat(old_r_dn_carts[None, ...], n_grid, axis=0)

        up_alt_idx = min(1, old_r_up_carts.shape[0] - 1)
        dn_alt_idx = min(1, old_r_dn_carts.shape[0] - 1)

        if pattern == "all_moved":
            for g in range(n_grid):
                r = g % 3
                if r == 0:
                    new_r_up_carts_arr[g, 0, 0] += 0.05
                elif r == 1:
                    new_r_dn_carts_arr[g, 0, 1] -= 0.07
                else:
                    new_r_up_carts_arr[g, up_alt_idx, 2] += 0.09
        elif pattern == "none_moved":
            pass
        elif pattern == "mixed":
            new_r_up_carts_arr[0, 0, 0] += 0.05
            if n_grid >= 3:
                new_r_dn_carts_arr[n_grid - 1, dn_alt_idx, 1] += 0.04

        ratio_debug = _compute_ratio_Jastrow_part_debug(
            jastrow_data,
            old_r_up_carts=old_r_up_carts,
            old_r_dn_carts=old_r_dn_carts,
            new_r_up_carts_arr=new_r_up_carts_arr,
            new_r_dn_carts_arr=new_r_dn_carts_arr,
        )

        j3_state = None
        if use_j3_state:
            j3_state = _init_grads_laplacian_Jastrow_three_body_streaming_state(
                jastrow_data.jastrow_three_body_data,
                old_r_up_carts,
                old_r_dn_carts,
            )

        ratio_auto = _compute_ratio_Jastrow_part_rank1_update(
            jastrow_data,
            old_r_up_carts=old_r_up_carts,
            old_r_dn_carts=old_r_dn_carts,
            new_r_up_carts_arr=new_r_up_carts_arr,
            new_r_dn_carts_arr=new_r_dn_carts_arr,
            j3_state=j3_state,
        )

        assert ratio_auto.shape == (n_grid,)
        assert not np.any(np.isnan(np.asarray(np.asarray(ratio_debug)))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(np.asarray(ratio_auto)))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(ratio_debug), np.asarray(ratio_auto), atol=atol, rtol=rtol)

        if pattern == "none_moved":
            assert not np.any(np.isnan(np.asarray(np.asarray(ratio_debug)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.ones_like(np.asarray(ratio_debug))))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(ratio_debug), np.ones_like(np.asarray(ratio_debug)), atol=atol, rtol=rtol)

        jax.clear_caches()


# ---------------------------------------------------------------------------
# L1: Unit tests for Jastrow_data.symmetrize_j3
# ---------------------------------------------------------------------------


def _make_jastrow_with_j3(j_matrix):
    """Build a minimal Jastrow_data whose j3 j_matrix is *j_matrix*."""
    j3_data = Jastrow_three_body_data(
        orb_data=AOs_sphe_gto_data(),
        j_matrix=np.asarray(j_matrix, dtype=np.float64),
    )
    return Jastrow_data(jastrow_three_body_data=j3_data)


class TestJ3MatrixUtilities:
    @pytest.mark.light
    def test_symmetrize_j3_symmetric_subblock(self):
        """L1-7: [:, :-1] symmetric -> 0.5*(A+A.T) on sub-block, last col unchanged."""
        rng = np.random.RandomState(10)
        n = 4
        sq_sym = rng.randn(n, n)
        sq_sym = 0.5 * (sq_sym + sq_sym.T)
        last_col = rng.randn(n)
        j_matrix = np.column_stack([sq_sym, last_col])
        jd = _make_jastrow_with_j3(j_matrix)

        mat = rng.randn(n, n + 1)  # non-symmetric input
        result = jd.symmetrize_j3(mat)
        np.testing.assert_array_equal(result[:, :-1], result[:, :-1].T)
        np.testing.assert_array_equal(result[:, -1], mat[:, -1])

    @pytest.mark.light
    def test_symmetrize_j3_nonsymmetric_subblock(self):
        """L1-8: the declared J3 tie does not depend on current values."""
        rng = np.random.RandomState(11)
        n = 4
        sq_nonsym = rng.randn(n, n)
        sq_nonsym[0, 1] = 100.0
        last_col = rng.randn(n)
        j_matrix = np.column_stack([sq_nonsym, last_col])
        jd = _make_jastrow_with_j3(j_matrix)

        mat = rng.randn(n, n + 1)
        result = jd.symmetrize_j3(mat)
        np.testing.assert_array_equal(result[:, :-1], result[:, :-1].T)
        np.testing.assert_array_equal(result[:, -1], mat[:, -1])

    @pytest.mark.light
    def test_symmetrize_j3_none(self):
        """L1-9: jastrow_three_body_data=None -> no-op."""
        jd = Jastrow_data()  # all components are None
        mat = np.random.RandomState(12).randn(3, 4)
        result = jd.symmetrize_j3(mat)
        np.testing.assert_array_equal(result, mat)

    @pytest.mark.light
    def test_symmetrize_j3_too_few_columns(self):
        """L1-10: an invalid stored J3 shape is a hard error."""
        j_matrix = np.array([[1.0], [2.0], [3.0]])  # (3, 1)
        jd = _make_jastrow_with_j3(j_matrix)
        mat = np.random.RandomState(13).randn(3, 1)
        with pytest.raises(ValueError, match="invalid stored J3 shape"):
            jd.symmetrize_j3(mat)

    @pytest.mark.light
    def test_apply_block_update_symmetric_j3_preserved(self):
        """L2-4: symmetric j3 stays symmetric after non-symmetric delta."""
        rng = np.random.RandomState(20)
        n = 4
        sq_sym = rng.randn(n, n)
        sq_sym = 0.5 * (sq_sym + sq_sym.T)
        last_col = rng.randn(n)
        j_matrix = np.column_stack([sq_sym, last_col])
        jd = _make_jastrow_with_j3(j_matrix)

        delta = rng.randn(n, n + 1)
        lr = 0.1
        updated_values = j_matrix + lr * delta
        block = make_test_block(
            "j3_matrix",
            updated_values,
            field="real",
            derivative_policy="direct_real",
            sync_scope="world",
        )
        new_jd = jd.apply_block_update(block)
        new_j3 = np.asarray(new_jd.jastrow_three_body_data.j_matrix)
        np.testing.assert_array_equal(new_j3[:, :-1], new_j3[:, :-1].T)

    @pytest.mark.light
    def test_apply_block_update_does_not_infer_j3_policy_from_values(self):
        """L2-5: J3 symmetry is structural even if legacy input is invalid."""
        rng = np.random.RandomState(21)
        n = 4
        sq_nonsym = rng.randn(n, n)
        sq_nonsym[0, 1] = 100.0
        last_col = rng.randn(n)
        j_matrix = np.column_stack([sq_nonsym, last_col])
        jd = _make_jastrow_with_j3(j_matrix)

        delta = rng.randn(n, n + 1)
        lr = 0.1
        updated_values = j_matrix + lr * delta
        block = make_test_block(
            "j3_matrix",
            updated_values,
            field="real",
            derivative_policy="direct_real",
            sync_scope="world",
        )
        new_jd = jd.apply_block_update(block)
        new_j3 = np.asarray(new_jd.jastrow_three_body_data.j_matrix)
        np.testing.assert_array_equal(new_j3[:, :-1], new_j3[:, :-1].T)
        # Jastrow parameter storage is float64 in both modes (issue #781): the update
        # keeps the exact fp64 block values, not a zone-rounded copy.
        expected_storage = np.asarray(updated_values[:, -1], dtype=np.float64)
        assert new_j3.dtype == expected_storage.dtype == np.float64
        np.testing.assert_array_equal(new_j3[:, -1], expected_storage)


# ---------------------------------------------------------------------------
# L2: apply_block_update symmetry preservation for Jastrow_data (j3)
# ---------------------------------------------------------------------------


class TestStreamingJastrowState:
    class TestStreamingJ1:
        @pytest.mark.light
        @pytest.mark.parametrize("j1b_type", ["exp", "pade"])
        @pytest.mark.parametrize("n_up,n_dn", [(5, 4), (1, 0), (3, 3)])
        def test_streaming_J1_state_against_full_molecular(self, j1b_type, n_up, n_dn):
            """After 32 seed-0 moves, both-spin J1 gradients/Laplacians match full recomputation.

            Synthetic five-nucleus input, exp/Pade parameter 1.0, and electron
            counts (5,4), (1,0), (3,3); no TREXIO or determinant screening.
            Compare the final cached arrays with the analytic full kernel in
            each precision using the existing jastrow_grad_lap strict policy.
            This is route parity, not an independent J1 formula oracle.
            J1 is per-electron independent (no electron-electron coupling), so the
            advance only re-evaluates one row of the cached arrays.
            """
            from jqmc.jastrow_factor import (
                _advance_grads_laplacian_Jastrow_one_body_streaming_state,
                _init_grads_laplacian_Jastrow_one_body_streaming_state,
            )

            rng = np.random.RandomState(0)
            num_R_cart_samples = 5
            R_carts = 4.0 * rng.rand(num_R_cart_samples, 3) - 2.0
            structure_data = _make_structure(R_carts, atomic_number=6)
            core_electrons = tuple([2] * num_R_cart_samples)

            jastrow_one_body_data = Jastrow_one_body_data(
                jastrow_1b_param=1.0,
                jastrow_1b_type=j1b_type,
                structure_data=structure_data,
                core_electrons=core_electrons,
            )

            r_up = (4.0 * rng.rand(n_up, 3) - 2.0) if n_up > 0 else np.zeros((0, 3))
            r_dn = (4.0 * rng.rand(n_dn, 3) - 2.0) if n_dn > 0 else np.zeros((0, 3))

            state = _init_grads_laplacian_Jastrow_one_body_streaming_state(
                jastrow_one_body_data, jax.numpy.asarray(r_up), jax.numpy.asarray(r_dn)
            )

            K = 32
            # J1 streaming exercises only the jastrow_grad_lap zone (electron-nucleus,
            # no AO/MO involvement). Tolerance must follow that zone, NOT wf_kinetic
            # (the latter is fp64-only and incorrectly tightens the bound under mixed
            # precision where jastrow_grad_lap = fp32).
            atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
            for _ in range(K):
                spin_choices = []
                if n_up > 0:
                    spin_choices.append(0)
                if n_dn > 0:
                    spin_choices.append(1)
                spin = spin_choices[rng.randint(0, len(spin_choices))]
                if spin == 0:
                    idx = rng.randint(0, n_up)
                    r_up = np.asarray(r_up).copy()
                    r_up[idx] = r_up[idx] + 0.1 * rng.randn(3)
                    moved_spin_is_up = True
                    moved_index = idx
                else:
                    idx = rng.randint(0, n_dn)
                    r_dn = np.asarray(r_dn).copy()
                    r_dn[idx] = r_dn[idx] + 0.1 * rng.randn(3)
                    moved_spin_is_up = False
                    moved_index = idx

                state = _advance_grads_laplacian_Jastrow_one_body_streaming_state(
                    jastrow_one_body_data,
                    state,
                    jax.numpy.asarray(moved_spin_is_up),
                    jax.numpy.asarray(moved_index, dtype=jax.numpy.int32),
                    jax.numpy.asarray(r_up),
                    jax.numpy.asarray(r_dn),
                )

            g_up_full, g_dn_full, l_up_full, l_dn_full = compute_grads_and_laplacian_Jastrow_one_body(
                jastrow_one_body_data, jax.numpy.asarray(r_up), jax.numpy.asarray(r_dn)
            )
            np.testing.assert_allclose(np.asarray(state.grad_J1_up), np.asarray(g_up_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.grad_J1_dn), np.asarray(g_dn_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J1_up), np.asarray(l_up_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J1_dn), np.asarray(l_dn_full), atol=atol, rtol=rtol)

        @pytest.mark.light
        @pytest.mark.parametrize("j1b_type", ["exp", "pade"])
        @pytest.mark.parametrize("n_up,n_dn", [(5, 4), (1, 0), (3, 3)])
        def test_streaming_J1_state_against_full_periodic(self, j1b_type, n_up, n_dn):
            """PBC sibling of ``test_streaming_J1_state_against_full_molecular``.

            J1 streaming both inits and advances through
            ``compute_grads_and_laplacian_Jastrow_one_body``, which dispatches on
            ``structure_data.pbc_flag`` to the Williamson smooth periodic distance, so
            the streaming path is PBC-transparent. Feeding a periodic ``structure_data``
            exercises that the final cached state agrees with a full recompute
            within the existing strict tolerance under PBC, not bit equality.
            """
            from jqmc.jastrow_factor import (
                _advance_grads_laplacian_Jastrow_one_body_streaming_state,
                _init_grads_laplacian_Jastrow_one_body_streaming_state,
            )

            rng = np.random.RandomState(0)
            num_R_cart_samples = 5
            R_carts = 2.0 + 6.0 * rng.rand(num_R_cart_samples, 3)
            structure_data = _make_pbc_structure(R_carts, atomic_number=6)
            core_electrons = tuple([2] * num_R_cart_samples)

            jastrow_one_body_data = Jastrow_one_body_data(
                jastrow_1b_param=1.0,
                jastrow_1b_type=j1b_type,
                structure_data=structure_data,
                core_electrons=core_electrons,
            )

            r_up = (10.0 * rng.rand(n_up, 3)) if n_up > 0 else np.zeros((0, 3))
            r_dn = (10.0 * rng.rand(n_dn, 3)) if n_dn > 0 else np.zeros((0, 3))

            state = _init_grads_laplacian_Jastrow_one_body_streaming_state(
                jastrow_one_body_data, jax.numpy.asarray(r_up), jax.numpy.asarray(r_dn)
            )

            K = 32
            atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
            for _ in range(K):
                spin_choices = []
                if n_up > 0:
                    spin_choices.append(0)
                if n_dn > 0:
                    spin_choices.append(1)
                spin = spin_choices[rng.randint(0, len(spin_choices))]
                if spin == 0:
                    idx = rng.randint(0, n_up)
                    r_up = np.asarray(r_up).copy()
                    r_up[idx] = r_up[idx] + 0.1 * rng.randn(3)
                    moved_spin_is_up = True
                    moved_index = idx
                else:
                    idx = rng.randint(0, n_dn)
                    r_dn = np.asarray(r_dn).copy()
                    r_dn[idx] = r_dn[idx] + 0.1 * rng.randn(3)
                    moved_spin_is_up = False
                    moved_index = idx

                state = _advance_grads_laplacian_Jastrow_one_body_streaming_state(
                    jastrow_one_body_data,
                    state,
                    jax.numpy.asarray(moved_spin_is_up),
                    jax.numpy.asarray(moved_index, dtype=jax.numpy.int32),
                    jax.numpy.asarray(r_up),
                    jax.numpy.asarray(r_dn),
                )

            g_up_full, g_dn_full, l_up_full, l_dn_full = compute_grads_and_laplacian_Jastrow_one_body(
                jastrow_one_body_data, jax.numpy.asarray(r_up), jax.numpy.asarray(r_dn)
            )
            np.testing.assert_allclose(np.asarray(state.grad_J1_up), np.asarray(g_up_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.grad_J1_dn), np.asarray(g_dn_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J1_up), np.asarray(l_up_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J1_dn), np.asarray(l_dn_full), atol=atol, rtol=rtol)

            jax.clear_caches()

    class TestStreamingJ2:
        @pytest.mark.light
        @pytest.mark.parametrize("j2b_type", ["pade", "exp"])
        @pytest.mark.parametrize("n_up,n_dn", [(5, 4), (1, 0), (3, 3)])
        def test_streaming_J2_state_against_full_molecular(self, j2b_type, n_up, n_dn):
            """After 32 seed-0 moves, cached J2 coordinates and both-spin derivatives match recomputation.

            Synthetic exp/Pade parameter 1.0 and electron counts (5,4), (1,0),
            (3,3); no TREXIO or determinant screening. Compare final gradients
            and Laplacians with the analytic full kernel using the existing
            jastrow_grad_lap strict policy in each precision. Route parity,
            not an independent J2 formula oracle or a per-hop value check.
            J2 is electron-pair coupled, so the advance updates the moved electron's
            same-spin row (i != k) and the cross-spin partners. Sign asymmetry between
            sigma=up and sigma=dn cross branches makes this the most error-prone of the
            streaming kernels -- exercise the available branches with K=32
            seed-0 randomly selected single-electron moves.
            """
            from jqmc.jastrow_factor import (
                _advance_grads_laplacian_Jastrow_two_body_streaming_state,
                _init_grads_laplacian_Jastrow_two_body_streaming_state,
            )

            rng = np.random.RandomState(0)
            jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type=j2b_type)

            r_up = (4.0 * rng.rand(n_up, 3) - 2.0) if n_up > 0 else np.zeros((0, 3))
            r_dn = (4.0 * rng.rand(n_dn, 3) - 2.0) if n_dn > 0 else np.zeros((0, 3))

            state = _init_grads_laplacian_Jastrow_two_body_streaming_state(
                jastrow_two_body_data, jax.numpy.asarray(r_up), jax.numpy.asarray(r_dn)
            )

            K = 32
            # J2 streaming exercises only the jastrow_grad_lap zone (electron-electron
            # pair coupling, no AO/MO involvement). Under mixed precision the pair
            # delta path additionally accumulates fp32 cancellation error over K steps;
            # the jastrow_grad_lap fp32 strict tolerance covers this.
            atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
            for _ in range(K):
                spin_choices = []
                if n_up > 0:
                    spin_choices.append(0)
                if n_dn > 0:
                    spin_choices.append(1)
                spin = spin_choices[rng.randint(0, len(spin_choices))]
                if spin == 0:
                    idx = rng.randint(0, n_up)
                    r_up = np.asarray(r_up).copy()
                    r_up[idx] = r_up[idx] + 0.1 * rng.randn(3)
                    moved_spin_is_up = True
                    moved_index = idx
                else:
                    idx = rng.randint(0, n_dn)
                    r_dn = np.asarray(r_dn).copy()
                    r_dn[idx] = r_dn[idx] + 0.1 * rng.randn(3)
                    moved_spin_is_up = False
                    moved_index = idx

                state = _advance_grads_laplacian_Jastrow_two_body_streaming_state(
                    jastrow_two_body_data,
                    state,
                    jax.numpy.asarray(moved_spin_is_up),
                    jax.numpy.asarray(moved_index, dtype=jax.numpy.int32),
                    jax.numpy.asarray(r_up),
                    jax.numpy.asarray(r_dn),
                )

            # Cached r_up/r_dn inside the streaming state must track the moves.
            np.testing.assert_allclose(np.asarray(state.r_up_carts), np.asarray(r_up), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.r_dn_carts), np.asarray(r_dn), atol=atol, rtol=rtol)

            g_up_full, g_dn_full, l_up_full, l_dn_full = compute_grads_and_laplacian_Jastrow_two_body(
                jastrow_two_body_data, jax.numpy.asarray(r_up), jax.numpy.asarray(r_dn)
            )
            np.testing.assert_allclose(np.asarray(state.grad_J2_up), np.asarray(g_up_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.grad_J2_dn), np.asarray(g_dn_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J2_up), np.asarray(l_up_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J2_dn), np.asarray(l_dn_full), atol=atol, rtol=rtol)

        @pytest.mark.light
        @pytest.mark.parametrize("j2b_type", ["pade", "exp"])
        @pytest.mark.parametrize("n_up,n_dn", [(5, 4), (1, 0), (3, 3)])
        def test_streaming_J2_state_against_full_periodic(self, j2b_type, n_up, n_dn):
            """PBC sibling of ``test_streaming_J2_state_against_full_molecular``.

            Under PBC the J2 fresh kernel (used by ``_init``) uses closed-form
            derivatives of the Williamson smooth periodic distance, while
            streaming ``_advance`` computes per-pair grad/lap deltas.
            The advance dispatches on ``structure_data.pbc_flag`` to the
            PBC-aware ``_j2_pair_terms_pbc`` so the final streaming state agrees
            within strict tolerance with a full recompute under PBC.
            A periodic ``structure_data`` is supplied to
            ``Jastrow_two_body_data`` to drive that dispatch.
            """
            from jqmc.jastrow_factor import (
                _advance_grads_laplacian_Jastrow_two_body_streaming_state,
                _init_grads_laplacian_Jastrow_two_body_streaming_state,
            )

            rng = np.random.RandomState(0)
            num_R_cart_samples = 4
            R_carts = 2.0 + 6.0 * rng.rand(num_R_cart_samples, 3)
            structure_data = _make_pbc_structure(R_carts, atomic_number=6)
            jastrow_two_body_data = Jastrow_two_body_data(
                jastrow_2b_param=1.0, jastrow_2b_type=j2b_type, structure_data=structure_data
            )

            r_up = (10.0 * rng.rand(n_up, 3)) if n_up > 0 else np.zeros((0, 3))
            r_dn = (10.0 * rng.rand(n_dn, 3)) if n_dn > 0 else np.zeros((0, 3))

            state = _init_grads_laplacian_Jastrow_two_body_streaming_state(
                jastrow_two_body_data, jax.numpy.asarray(r_up), jax.numpy.asarray(r_dn)
            )

            K = 32
            atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
            for _ in range(K):
                spin_choices = []
                if n_up > 0:
                    spin_choices.append(0)
                if n_dn > 0:
                    spin_choices.append(1)
                spin = spin_choices[rng.randint(0, len(spin_choices))]
                if spin == 0:
                    idx = rng.randint(0, n_up)
                    r_up = np.asarray(r_up).copy()
                    r_up[idx] = r_up[idx] + 0.1 * rng.randn(3)
                    moved_spin_is_up = True
                    moved_index = idx
                else:
                    idx = rng.randint(0, n_dn)
                    r_dn = np.asarray(r_dn).copy()
                    r_dn[idx] = r_dn[idx] + 0.1 * rng.randn(3)
                    moved_spin_is_up = False
                    moved_index = idx

                state = _advance_grads_laplacian_Jastrow_two_body_streaming_state(
                    jastrow_two_body_data,
                    state,
                    jax.numpy.asarray(moved_spin_is_up),
                    jax.numpy.asarray(moved_index, dtype=jax.numpy.int32),
                    jax.numpy.asarray(r_up),
                    jax.numpy.asarray(r_dn),
                )

            np.testing.assert_allclose(np.asarray(state.r_up_carts), np.asarray(r_up), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.r_dn_carts), np.asarray(r_dn), atol=atol, rtol=rtol)

            g_up_full, g_dn_full, l_up_full, l_dn_full = compute_grads_and_laplacian_Jastrow_two_body(
                jastrow_two_body_data, jax.numpy.asarray(r_up), jax.numpy.asarray(r_dn)
            )
            np.testing.assert_allclose(np.asarray(state.grad_J2_up), np.asarray(g_up_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.grad_J2_dn), np.asarray(g_dn_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J2_up), np.asarray(l_up_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J2_dn), np.asarray(l_dn_full), atol=atol, rtol=rtol)

            jax.clear_caches()

    class TestStreamingJ3:
        @pytest.mark.parametrize(
            "trexio_file",
            ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"],
        )
        @pytest.mark.light
        def test_streaming_J3_state_against_full_molecular(self, trexio_file):
            """K random single-electron moves advanced via the J3 streaming state must
            match a fresh init at the resulting configuration (and the existing analytic
            full computation) within strict tolerance. Both electron moves and
            the independently initialized J3 matrix use explicit seed 0.
            """
            import os

            from jqmc.jastrow_factor import (
                _advance_grads_laplacian_Jastrow_three_body_streaming_state,
                _init_grads_laplacian_Jastrow_three_body_streaming_state,
            )
            from jqmc.trexio_wrapper import read_trexio_file

            (_, aos_data, _, _, geminal_mo_data, _) = read_trexio_file(
                trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
                store_tuple=True,
            )

            rng = np.random.RandomState(0)
            jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
                orb_data=aos_data, random_init=True, random_scale=1.0e-3, seed=0
            )
            n_up = geminal_mo_data.num_electron_up
            n_dn = geminal_mo_data.num_electron_dn

            r_up = 4.0 * rng.rand(n_up, 3) - 2.0
            r_dn = 4.0 * rng.rand(n_dn, 3) - 2.0

            state = _init_grads_laplacian_Jastrow_three_body_streaming_state(jastrow_threebody_data, r_up, r_dn)

            K = 32
            # J3 streaming crosses two zones: jastrow_grad_lap (J3 grad/lap arithmetic)
            # and ao_grad_lap (AO grad/lap consumed inside J3). Use the looser of the
            # two -- under mixed precision, jastrow_grad_lap (fp32) dominates.
            atol, rtol = get_tolerance_min(["jastrow_grad_lap", "ao_grad_lap"], "strict")
            for _ in range(K):
                # pick a random single-electron move (alternating spins when available)
                spin_choices = []
                if n_up > 0:
                    spin_choices.append(0)
                if n_dn > 0:
                    spin_choices.append(1)
                spin = spin_choices[rng.randint(0, len(spin_choices))]
                if spin == 0:
                    idx = rng.randint(0, n_up)
                    r_up = np.asarray(r_up).copy()
                    r_up[idx] = r_up[idx] + 0.1 * rng.randn(3)
                    moved_spin_is_up = True
                    moved_index = idx
                else:
                    idx = rng.randint(0, n_dn)
                    r_dn = np.asarray(r_dn).copy()
                    r_dn[idx] = r_dn[idx] + 0.1 * rng.randn(3)
                    moved_spin_is_up = False
                    moved_index = idx

                state = _advance_grads_laplacian_Jastrow_three_body_streaming_state(
                    jastrow_threebody_data,
                    state,
                    jax.numpy.asarray(moved_spin_is_up),
                    jax.numpy.asarray(moved_index, dtype=jax.numpy.int32),
                    jax.numpy.asarray(r_up),
                    jax.numpy.asarray(r_dn),
                )

            fresh = _init_grads_laplacian_Jastrow_three_body_streaming_state(jastrow_threebody_data, r_up, r_dn)
            np.testing.assert_allclose(np.asarray(state.grad_J3_up), np.asarray(fresh.grad_J3_up), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.grad_J3_dn), np.asarray(fresh.grad_J3_dn), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J3_up), np.asarray(fresh.lap_J3_up), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J3_dn), np.asarray(fresh.lap_J3_dn), atol=atol, rtol=rtol)

            # cross-check against the existing analytic full computation
            g3u_full, g3d_full, l3u_full, l3d_full = compute_grads_and_laplacian_Jastrow_three_body(
                jastrow_threebody_data, jax.numpy.asarray(r_up), jax.numpy.asarray(r_dn)
            )
            np.testing.assert_allclose(np.asarray(state.grad_J3_up), np.asarray(g3u_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.grad_J3_dn), np.asarray(g3d_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J3_up), np.asarray(l3u_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J3_dn), np.asarray(l3d_full), atol=atol, rtol=rtol)

            jax.clear_caches()

        @pytest.mark.light
        @pytest.mark.parametrize("j3b_type", ["sphe"])
        def test_streaming_J3_state_against_full_periodic(self, j3b_type):
            """PBC sibling of ``test_streaming_J3_state_against_full_molecular``.

            J3 has no explicit electron-electron distance: the correlation enters only
            through AO products, so PBC support is carried entirely by periodic-expanded
            AOs. Building the J3 data on ``_make_pbc_structure`` + ``with_periodic_image_
            expansion`` (mirroring the validated 3b periodic part tests) makes the
            streaming advance PBC-aware without any kernel change; this test confirms the
            final cached state agrees within strict tolerance with a fresh init
            and the analytic full kernel under PBC, not bit equality.
            """
            from jqmc.jastrow_factor import (
                _advance_grads_laplacian_Jastrow_three_body_streaming_state,
                _init_grads_laplacian_Jastrow_three_body_streaming_state,
            )

            rng = np.random.RandomState(0)
            num_R_cart_samples = 4
            num_ao = 4
            num_ao_prim = 4
            R_carts = 2.0 + 6.0 * rng.rand(num_R_cart_samples, 3)
            structure_data = _make_pbc_structure(R_carts, atomic_number=6)

            orbital_indices = (0, 1, 2, 3)
            exponents = (1.2, 1.0, 0.8, 0.6)
            coefficients = (1.0, 1.0, 1.0, 1.0)
            angular_momentums = (0, 0, 1, 1)
            magnetic_quantum_numbers = (0, 0, 1, -1)

            aos_pbc_mol = AOs_sphe_gto_data._from_full_basis(
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
            expanded_aos = _apply_image_expansion(aos_pbc_mol, structure_data, tol=AO_IMAGE_TRUNCATION_ATOL)

            # ``j_matrix`` is sized by the molecular AO count; the public AO wrapper
            # collapses the image axis back to the molecular rank before contraction.
            rng_j = np.random.default_rng(0)
            j_matrix = rng_j.uniform(0.0, 1.0e-3, size=(aos_pbc_mol.num_ao, aos_pbc_mol.num_ao + 1))
            jastrow_threebody_data = Jastrow_three_body_data(orb_data=expanded_aos, j_matrix=j_matrix)

            n_up = 4
            n_dn = 3
            r_up = 10.0 * rng.rand(n_up, 3)
            r_dn = 10.0 * rng.rand(n_dn, 3)

            state = _init_grads_laplacian_Jastrow_three_body_streaming_state(jastrow_threebody_data, r_up, r_dn)

            K = 32
            atol, rtol = get_tolerance_min(["jastrow_grad_lap", "ao_grad_lap"], "strict")
            for _ in range(K):
                spin_choices = []
                if n_up > 0:
                    spin_choices.append(0)
                if n_dn > 0:
                    spin_choices.append(1)
                spin = spin_choices[rng.randint(0, len(spin_choices))]
                if spin == 0:
                    idx = rng.randint(0, n_up)
                    r_up = np.asarray(r_up).copy()
                    r_up[idx] = r_up[idx] + 0.1 * rng.randn(3)
                    moved_spin_is_up = True
                    moved_index = idx
                else:
                    idx = rng.randint(0, n_dn)
                    r_dn = np.asarray(r_dn).copy()
                    r_dn[idx] = r_dn[idx] + 0.1 * rng.randn(3)
                    moved_spin_is_up = False
                    moved_index = idx

                state = _advance_grads_laplacian_Jastrow_three_body_streaming_state(
                    jastrow_threebody_data,
                    state,
                    jax.numpy.asarray(moved_spin_is_up),
                    jax.numpy.asarray(moved_index, dtype=jax.numpy.int32),
                    jax.numpy.asarray(r_up),
                    jax.numpy.asarray(r_dn),
                )

            fresh = _init_grads_laplacian_Jastrow_three_body_streaming_state(jastrow_threebody_data, r_up, r_dn)
            np.testing.assert_allclose(np.asarray(state.grad_J3_up), np.asarray(fresh.grad_J3_up), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.grad_J3_dn), np.asarray(fresh.grad_J3_dn), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J3_up), np.asarray(fresh.lap_J3_up), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J3_dn), np.asarray(fresh.lap_J3_dn), atol=atol, rtol=rtol)

            g3u_full, g3d_full, l3u_full, l3d_full = compute_grads_and_laplacian_Jastrow_three_body(
                jastrow_threebody_data, jax.numpy.asarray(r_up), jax.numpy.asarray(r_dn)
            )
            np.testing.assert_allclose(np.asarray(state.grad_J3_up), np.asarray(g3u_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.grad_J3_dn), np.asarray(g3d_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J3_up), np.asarray(l3u_full), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(state.lap_J3_dn), np.asarray(l3d_full), atol=atol, rtol=rtol)

            jax.clear_caches()


class TestJastrowPBCAnalytic:
    """PBC J1/J2 grad/lap: closed-form smooth-periodic kernels vs the ``_auto`` oracles.

    The production dispatch under ``pbc_flag`` runs the closed-form chain rule on
    :func:`jqmc.jastrow_factor._smooth_periodic_distance_grad_lap` (a 1:1 port of the
    Rust ``smooth_dist``); the retained ``_auto`` implementations (``grad`` +
    ``hessian`` of the smooth-periodic value function) are the oracles here. Machine
    precision expected (same arithmetic, different evaluation order); asserted at the
    zone-strict tolerance. Triclinic cells exercise the off-diagonal ``inv_cell``
    terms of the metric derivatives that a cubic cell leaves untested.
    """

    @staticmethod
    def _pbc_structure(triclinic: bool, n_atoms: int = 4, box: float = 8.0):
        rng = np.random.default_rng(11)
        positions = rng.random((n_atoms, 3)) * box
        vec_b = (0.9, box, 0.0) if triclinic else (0.0, box, 0.0)
        vec_c = (0.3, 0.5, box) if triclinic else (0.0, 0.0, box)
        return Structure_data.build(
            positions=positions,
            pbc_flag=True,
            vec_a=(box, 0.0, 0.0),
            vec_b=vec_b,
            vec_c=vec_c,
            atomic_numbers=tuple([6] * n_atoms),
            element_symbols=tuple(["C"] * n_atoms),
            atomic_labels=tuple(["C"] * n_atoms),
        )

    @staticmethod
    def _draw_electrons(box: float = 8.0):
        rng = np.random.default_rng(23)
        return (
            jax.numpy.asarray(rng.random((7, 3)) * box),
            jax.numpy.asarray(rng.random((6, 3)) * box),
        )

    @pytest.mark.light
    @pytest.mark.parametrize("jb_type", ["exp", "pade"])
    @pytest.mark.parametrize("triclinic", [False, True], ids=["cubic", "triclinic"])
    def test_j1_pbc_analytic_vs_auto(self, jb_type, triclinic):
        """J1 PBC analytic vs auto."""

        atol, rtol = get_tolerance_min(["jastrow_grad_lap"], "strict")

        def case_factory():
            structure_data = self._pbc_structure(triclinic)
            r_up_carts, r_dn_carts = self._draw_electrons()
            j1 = Jastrow_one_body_data(
                jastrow_1b_param=0.8,
                jastrow_1b_type=jb_type,
                structure_data=structure_data,
                # Zero is the CORRECT value here, not the hardcoded placeholder that
                # tests/_ecp_core_electrons.py exists to replace: this structure is built
                # in-test with no Coulomb_potential_data at all, so there is no
                # pseudopotential and no core to remove. There is nothing to read it from.
                core_electrons=tuple([0.0] * 4),
            )
            return (j1, r_up_carts, r_dn_carts)

        def analytical(case):
            j1, r_up_carts, r_dn_carts = case
            return compute_grads_and_laplacian_Jastrow_one_body(j1, r_up_carts, r_dn_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            mixed_atol, mixed_rtol = get_tolerance_mixed_fidelity("parameter_derivative")
            for mixed_block, full_block in blocks:
                _assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, mixed_atol, mixed_rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=mixed_atol, rtol=mixed_rtol)
            with pytest.raises(AssertionError):
                _assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], mixed_atol, mixed_rtol)
            jax.clear_caches()
            return

        j1, r_up_carts, r_dn_carts = case_factory()
        analytic = compute_grads_and_laplacian_Jastrow_one_body(j1, r_up_carts, r_dn_carts)
        oracle = _compute_grads_and_laplacian_Jastrow_one_body_auto(j1, r_up_carts, r_dn_carts)
        for component, got, ref in zip(
            ("gradient-up", "gradient-down", "laplacian-up", "laplacian-down"), analytic, oracle, strict=True
        ):
            assert not np.any(np.isnan(np.asarray(got))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(ref))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("jb_type", ["exp", "pade"])
    @pytest.mark.parametrize("triclinic", [False, True], ids=["cubic", "triclinic"])
    def test_j2_pbc_analytic_vs_auto(self, jb_type, triclinic):
        """J2 PBC analytic vs auto."""

        atol, rtol = get_tolerance_min(["jastrow_grad_lap"], "strict")

        def case_factory():
            structure_data = self._pbc_structure(triclinic)
            r_up_carts, r_dn_carts = self._draw_electrons()
            j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
                jastrow_2b_param=0.9, jastrow_2b_type=jb_type, structure_data=structure_data
            )
            return (j2, r_up_carts, r_dn_carts)

        def analytical(case):
            j2, r_up_carts, r_dn_carts = case
            return compute_grads_and_laplacian_Jastrow_two_body(j2, r_up_carts, r_dn_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            mixed_atol, mixed_rtol = get_tolerance_mixed_fidelity("parameter_derivative")
            for mixed_block, full_block in blocks:
                _assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, mixed_atol, mixed_rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=mixed_atol, rtol=mixed_rtol)
            with pytest.raises(AssertionError):
                _assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], mixed_atol, mixed_rtol)
            jax.clear_caches()
            return

        j2, r_up_carts, r_dn_carts = case_factory()
        analytic = compute_grads_and_laplacian_Jastrow_two_body(j2, r_up_carts, r_dn_carts)
        oracle = _compute_grads_and_laplacian_Jastrow_two_body_auto(j2, r_up_carts, r_dn_carts)
        for component, got, ref in zip(
            ("gradient-up", "gradient-down", "laplacian-up", "laplacian-down"), analytic, oracle, strict=True
        ):
            assert not np.any(np.isnan(np.asarray(got))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(ref))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=atol, rtol=rtol)
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
