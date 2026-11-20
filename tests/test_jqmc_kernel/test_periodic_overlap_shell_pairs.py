"""Exact-term regression and deterministic work guard for periodic overlap."""

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

from unittest.mock import patch

import numpy as np
import pytest

from jqmc import atomic_orbital as ao
from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL, get_pgto_lattice_overlap_term_tolerance
from jqmc._precision import get_dtype_np, get_tolerance
from tests.conftest import make_pbc_structure

pytestmark = pytest.mark.without_chdir


def small_basis(kind, *, incomplete=False):
    """Build fixed interleaved off-cell s/p/d shells with independent contractions."""
    cell = np.array([[6.4, 0.0, 0.0], [0.8, 7.2, 0.0], [0.3, 0.6, 8.6]])
    positions = np.array([[1.2, -0.4, 2.1], [8.4, 3.8, -1.6]])
    structure = make_pbc_structure(
        cell, positions=positions, atomic_numbers=(1, 1), element_symbols=("H", "H"), atomic_labels=("H1", "H2")
    )
    rows = []
    for center in range(2):
        for ell in range(3):
            if kind == "cart":
                components = [(x, y, ell - x - y) for x in range(ell, -1, -1) for y in range(ell - x, -1, -1)]
                if incomplete and ell == 2:
                    components = components[::2]
            else:
                components = list(range(-ell, ell + 1))
            for component in components:
                rows.append((center, ell, component))
    # Interleaved centres/shells: contiguous AO order is not shell identity.
    rows = rows[::2] + rows[1::2]
    orbital_indices, exponents, coefficients = [], [], []
    for index, (center, ell, component) in enumerate(rows):
        for primitive, exponent in enumerate([0.23, 1.17]):
            orbital_indices.append(index)
            exponents.append(exponent)
            # Unequal Cartesian component contractions must not be mistaken
            # for a common prefactor when radial products are shared.
            coefficient = [0.8, -0.19][primitive]
            if kind == "cart":
                coefficient *= 1.0 + 0.03 * index
            coefficients.append(coefficient)
    common = dict(
        structure_data=structure,
        nucleus_index=tuple(row[0] for row in rows),
        num_ao=len(rows),
        num_ao_prim=len(exponents),
        orbital_indices=tuple(orbital_indices),
        exponents=np.array(exponents),
        coefficients=np.array(coefficients),
        angular_momentums=tuple(row[1] for row in rows),
    )
    if kind == "cart":
        return ao.AOs_cart_gto_data._from_full_basis(
            **common,
            polynominal_order_x=tuple(row[2][0] for row in rows),
            polynominal_order_y=tuple(row[2][1] for row in rows),
            polynominal_order_z=tuple(row[2][2] for row in rows),
        )
    return ao.AOs_sphe_gto_data._from_full_basis(**common, magnetic_quantum_numbers=tuple(row[2] for row in rows))


@pytest.mark.light
@pytest.mark.parametrize("kind", ["cart", "sphe"])
@pytest.mark.parametrize("twisted", [False, True])
def test_periodic_shell_pair_overlap_preserves_unbatched_terms(kind, twisted):
    """Public Cartesian/spherical Gamma/twist overlap preserves every original lattice term."""
    base = small_basis(kind)
    base.sanity_check()
    leaf = ao._apply_image_expansion(base, base.structure_data, tol=AO_IMAGE_TRUNCATION_ATOL)
    k_point = (0.21, -0.13, 0.17) if twisted else None
    if twisted:
        leaf = ao._promote_pgto_to_complex(leaf, k_point)
    if kind == "sphe":
        cart, transform = ao._aos_sphe_to_cart(base)
    else:
        cart, transform = base, np.eye(base.num_ao)
    tolerance = get_pgto_lattice_overlap_term_tolerance(get_dtype_np("det_eval"))
    expected = (
        transform
        @ ao._compute_overlap_matrix_periodic_cart_lattice_debug(cart, k_point=k_point, term_tolerance=tolerance)
        @ transform.T
    )
    actual = np.asarray(ao.compute_overlap_matrix(leaf))
    atol, rtol = get_tolerance("det_eval", "strict")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    np.testing.assert_array_equal(actual, actual.conj().T)
    assert actual.shape == (base.num_ao, base.num_ao)
    assert np.all(np.isfinite(actual))
    assert np.all(np.diag(actual).real > 0)


@pytest.mark.light
@pytest.mark.parametrize("incomplete", [False, True])
def test_periodic_shell_pair_overlap_reuses_cartesian_products(incomplete):
    """Complete and partial Cartesian shells reuse products beyond Hermitian-half savings."""
    base = small_basis("cart", incomplete=incomplete)
    tolerance = get_pgto_lattice_overlap_term_tolerance(get_dtype_np("det_eval"))
    original = ao._compute_overlap_1d_cart_vectorized
    with patch.object(ao, "_compute_overlap_1d_cart_vectorized", wraps=original) as old_integrals:
        expected = ao._compute_overlap_matrix_periodic_cart_lattice_debug(
            base, k_point=(0.21, -0.13, 0.17), term_tolerance=tolerance
        )
    with patch.object(ao, "_compute_overlap_1d_cart_vectorized", wraps=original) as new_integrals:
        actual = ao._compute_overlap_matrix_periodic_cart_lattice(base, k_point=(0.21, -0.13, 0.17), term_tolerance=tolerance)
    atol, rtol = get_tolerance("det_eval", "strict")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    # A deterministic work guard: batching saves beyond the Hermitian half.
    # This has no wall-clock dependency and changes no numerical tolerance.
    assert 2 * new_integrals.call_count < old_integrals.call_count
