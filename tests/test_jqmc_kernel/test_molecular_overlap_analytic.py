"""Vectorized molecular analytic AO overlap against an independent Gauss-Hermite reference."""

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

import numpy as np
import pytest

from jqmc import _setting
from jqmc import atomic_orbital as ao
from jqmc._precision import get_tolerance_min
from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL
from jqmc.atomic_orbital import (
    AOs_cart_gto_data,
    AOs_sphe_gto_data,
    _aos_sphe_to_cart,
    _apply_image_expansion,
    _compute_overlap_matrix_cart_analytic,
    compute_overlap_matrix,
)
from jqmc.trexio_wrapper import read_trexio_file
from tests.conftest import CELL_FIXTURES, make_pbc_structure

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_TREXIO_DIR = Path(__file__).resolve().parents[1] / "trexio_example_files"
# Cartesian fixtures up to f shells (cc-pVTZ), all-electron and ECP, one and two
# centres; water is spherical cc-pVQZ (up to g) evaluated through its Cartesian twin.
_FIXTURES = ["Li_ae_ccpvdz_cart.h5", "N2_ecp_ccpvtz_cart.h5", "Cl2_ecp_ccpvtz_cart.h5", "water_ccecp_ccpvqz.h5"]
_FAULTS = ["wrong-angular-factor", "dropped-ket-block"]
# 12-point Gauss-Hermite integrates polynomials of degree <= 23 exactly,
# which covers every (l_a + l_b) <= 8 used here.
_HERMITE_NODES, _HERMITE_WEIGHTS = np.polynomial.hermite.hermgauss(12)


def _cartesian_leaf(name):
    aos = read_trexio_file(str(_TREXIO_DIR / name), store_tuple=True)[1]
    return _aos_sphe_to_cart(aos)[0] if isinstance(aos, AOs_sphe_gto_data) else aos


def _image_expanded_leaf():
    """Multi-bucket image-expanded Cartesian leaf (diagnostic home-row x kernel-column form)."""
    structure = make_pbc_structure(
        CELL_FIXTURES["cubic_10bohr"],
        positions=np.array([[1.0, 1.2, 0.8]], dtype=np.float64),
        atomic_numbers=(3,),
        element_symbols=("Li",),
        atomic_labels=("Li1",),
    )
    base = AOs_cart_gto_data._from_full_basis(
        structure_data=structure,
        nucleus_index=(0, 0, 0, 0, 0),
        num_ao=5,
        num_ao_prim=5,
        orbital_indices=(0, 1, 2, 3, 4),
        exponents=np.array([0.15, 0.4, 0.4, 0.4, 3.0], dtype=np.float64),
        coefficients=np.ones(5),
        angular_momentums=(0, 1, 1, 1, 0),
        polynominal_order_x=(0, 1, 0, 0, 0),
        polynominal_order_y=(0, 0, 1, 0, 0),
        polynominal_order_z=(0, 0, 0, 1, 0),
    )
    return _apply_image_expansion(base, structure, tol=AO_IMAGE_TRUNCATION_ATOL)


def _gauss_hermite_overlap(leaf, bra_kernel_aos):
    """Independent reference: S[h, nu] = <chi_{bra[h]} | chi_nu> over all kernel AOs ``nu``.

    Each primitive pair is integrated per Cartesian axis by Gauss-Hermite
    quadrature of ``(x - A)^la (x - B)^lb`` against the product Gaussian
    ``exp(-p (x - X_P)^2)`` (exact for these polynomial degrees), times the
    Gaussian product prefactor; no binomial expansion and no moment formula is
    shared with production. Only the AO definition data (exponents,
    contraction coefficients, polynomial orders, centres and the per-primitive
    normalization of the AO evaluator) is read from the leaf.
    """
    centers = np.asarray(leaf._atomic_center_carts_np, dtype=np.float64)
    orbital = np.asarray(leaf.orbital_indices, dtype=np.int64)
    exponents = np.asarray(leaf.exponents, dtype=np.float64)
    coefficients = np.asarray(leaf.coefficients, dtype=np.float64)
    angular = np.asarray(leaf._angular_momentums_prim_np, dtype=np.float64)
    ratio = np.asarray(leaf._normalization_factorial_ratio_prim_jnp, dtype=np.float64)
    weight = coefficients * np.sqrt((2.0 * exponents / np.pi) ** 1.5 * (8.0 * exponents) ** angular * ratio)
    orders = np.stack(
        [np.asarray(o, dtype=np.int64) for o in (leaf.polynominal_order_x, leaf.polynominal_order_y, leaf.polynominal_order_z)],
        axis=1,
    )
    num_kernel = int(leaf._num_ao_kernel)
    prims_of = [np.nonzero(orbital == a)[0] for a in range(num_kernel)]
    reference = np.zeros((len(bra_kernel_aos), num_kernel), dtype=np.float64)
    for h, mu in enumerate(bra_kernel_aos):
        for nu in range(num_kernel):
            total = 0.0
            for p in prims_of[mu]:
                for q in prims_of[nu]:
                    a, b = exponents[p], exponents[q]
                    s = a + b
                    value = weight[p] * weight[q] * np.exp(-a * b / s * np.sum((centers[mu] - centers[nu]) ** 2))
                    for axis in range(3):
                        product_center = (a * centers[mu, axis] + b * centers[nu, axis]) / s
                        x = product_center + _HERMITE_NODES / np.sqrt(s)
                        integrand = (x - centers[mu, axis]) ** orders[mu, axis] * (x - centers[nu, axis]) ** orders[nu, axis]
                        value *= np.dot(_HERMITE_WEIGHTS, integrand) / np.sqrt(s)
                    total += value
            reference[h, nu] = total
    return reference


def _home_kernel_aos(leaf):
    """Kernel AO of each home AO: the image with zero lattice translation (identity for molecular leaves)."""
    if not getattr(leaf, "_ragged_buckets", ()):
        return list(range(int(leaf.num_ao)))
    images = np.asarray(leaf._image_indices, dtype=np.int64).reshape(-1, 3)
    home = np.asarray(leaf._ragged_kernel_to_home, dtype=np.int64)
    return [int(np.nonzero((home == h) & np.all(images == 0, axis=1))[0][0]) for h in range(int(leaf.num_ao))]


def _leaf(case):
    return _image_expanded_leaf() if case == "image-expanded" else _cartesian_leaf(case)


def _compare(actual, reference):
    atol, rtol = get_tolerance_min(("det_eval",), "strict")
    assert actual.shape == reference.shape
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("block_budget", ["default", "one-ao-per-block"])
@pytest.mark.parametrize("case", [*_FIXTURES, "image-expanded"])
def test_vectorized_analytic_overlap_matches_gauss_hermite_reference(case, block_budget, monkeypatch):
    """The vectorized molecular analytic overlap equals an independent quadrature reference.

    Molecular Cartesian TREXIO leaves up to f shells (and the Cartesian twin of
    spherical cc-pVQZ water, up to g), plus the image-expanded diagnostic form
    (home rows x kernel columns of a multi-bucket leaf), are compared with a
    per-primitive-pair Gauss-Hermite reference in the strict det_eval policy.
    The default block budget and a budget small enough to put one AO in every
    block (so every block boundary, segment reduction and multi-block
    assembly is exercised) must both agree. For molecular leaves the public
    ``compute_overlap_matrix`` must return the same matrix.
    """
    if block_budget == "one-ao-per-block":
        monkeypatch.setattr(_setting, "ANALYTIC_OVERLAP_PAIR_BUDGET_BYTES", 8 * _setting.ANALYTIC_OVERLAP_PAIR_WORK_ARRAYS)
    leaf = _leaf(case)
    reference = _gauss_hermite_overlap(leaf, _home_kernel_aos(leaf))
    actual = np.asarray(_compute_overlap_matrix_cart_analytic(leaf))
    _compare(actual, reference)
    if case != "image-expanded":
        _compare(np.asarray(compute_overlap_matrix(leaf)), reference)


@pytest.mark.parametrize("fault", _FAULTS)
@pytest.mark.parametrize("case", ["N2_ecp_ccpvtz_cart.h5", "image-expanded"])
def test_gauss_hermite_reference_rejects_vectorized_overlap_faults(case, fault, monkeypatch):
    """The reference comparison detects a wrong angular factor and a dropped ket block.

    ``wrong-angular-factor`` lowers the ket polynomial order by one in the 1D
    pair integrals (a p function treated as s, a d as p, ...).
    ``dropped-ket-block`` forces one AO per block and drops the ket block of
    the first home AO's own kernel AO, i.e. one kernel AO column including a
    diagonal element (for the image-expanded leaf, the zero-translation image
    of that shell). Both must fail the same strict det_eval comparison.
    """
    leaf = _leaf(case)
    reference = _gauss_hermite_overlap(leaf, _home_kernel_aos(leaf))
    if fault == "wrong-angular-factor":
        original = ao._compute_overlap_1d_cart_pairs

        def _faulty(alpha, beta, center_a, center_b, order_a, order_b):
            return original(alpha, beta, center_a, center_b, order_a, np.maximum(order_b - 1, 0))

        monkeypatch.setattr(ao, "_compute_overlap_1d_cart_pairs", _faulty)
    else:
        monkeypatch.setattr(_setting, "ANALYTIC_OVERLAP_PAIR_BUDGET_BYTES", 8 * _setting.ANALYTIC_OVERLAP_PAIR_WORK_ARRAYS)
        original_groups = ao._group_aos_by_primitive_budget
        calls = {"n": 0}

        dropped_column = _home_kernel_aos(leaf)[0]

        def _drop_one_ket_group(counts, target):
            groups = original_groups(counts, target)
            calls["n"] += 1
            # The first call groups the bra rows; every later call groups the ket columns.
            return groups if calls["n"] == 1 else [g for g in groups if not g[0] <= dropped_column < g[1]]

        monkeypatch.setattr(ao, "_group_aos_by_primitive_budget", _drop_one_ket_group)
    actual = np.asarray(_compute_overlap_matrix_cart_analytic(leaf))
    with pytest.raises(AssertionError):
        _compare(actual, reference)
