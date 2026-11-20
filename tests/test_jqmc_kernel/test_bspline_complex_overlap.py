"""Cell overlap of the complex (twisted) and real B-spline AO leaves.

The leaf's AOs are ``psi_a(r) = e^{i k . r} u_a(r)`` with one leaf-level twist
``k`` and a cell-periodic complex spline ``u_a``. The production overlap is the
exact Gram matrix of the periodic spline basis applied to the complex control
points (the Bloch phase cancels in ``conj(psi_a) psi_b``). The reference here is
built only from the leaf's own AO values: a tensor Gauss-Legendre rule with
``order + 1`` nodes on every knot interval of every fractional axis, which
integrates the piecewise-polynomial ``conj(u_a) u_b`` (degree ``2 * order`` per
axis) exactly, on a lattice-translated copy of the cell so that the wrap of
``u`` and the full-coordinate Bloch phase are both exercised.

Precision regime (reverse direction): the production overlap is an fp64
matrix in both precision modes, so the AO values the reference integrates are
evaluated with every precision zone at fp64 (``_fp64_reference_policy``; a no-op
in full mode, so the full-mode references are numerically unchanged). The mixed
policy's fp32 ``ao_eval`` values never enter the reference.
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

import contextlib
import logging

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import configure, get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import MO_ORTHONORMALITY_WARNING_THRESHOLD
from jqmc.atomic_orbital import (
    AOs_bspline_complex_data,
    AOs_bspline_data,
    compute_AOs,
    compute_overlap_matrix,
)
from jqmc.determinant import SD_data
from jqmc.hamiltonians import _compute_mo_orthonormality_metrics, _warn_mo_orthonormality
from jqmc.jastrow_factor import Jastrow_data
from jqmc.molecular_orbital import MOs_data
from jqmc.wavefunction import Wavefunction_data
from tests.conftest import CELL_FIXTURES, make_pbc_structure

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_TWISTS = {"gamma": (0.0, 0.0, 0.0), "general-k": (0.27, -0.1, 0.13)}
# Non-cubic grid so that a per-axis mix-up of N_x changes the result.
_SPLINE_SHAPE = (6, 7, 8)
# Integer lattice translation of the integration cell (fractional units).
_CELL_SHIFT = np.array([1.0, -2.0, 1.0])
_FAULTS = ["ket-conjugated", "no-conjugate", "dropped-bloch-phase-on-bra", "identity-placeholder"]
_WARNING_PREFIX = "MOs are not orthonormal"


def _structure():
    return make_pbc_structure(CELL_FIXTURES["triclinic_skewed"])


def _complex_leaf(order, twist, *, num_ao=3, seed=5):
    rng = np.random.default_rng(seed + order)
    coefficients = rng.normal(size=(num_ao, *_SPLINE_SHAPE)) + 1j * rng.normal(size=(num_ao, *_SPLINE_SHAPE))
    return AOs_bspline_complex_data.build(_structure(), coefficients, _SPLINE_SHAPE, order, k_point=_TWISTS[twist])


def _knot_gauss_legendre_cell(structure, order):
    """Cartesian nodes and weights of the exact per-knot-interval rule on the shifted cell."""
    nodes, weights = np.polynomial.legendre.leggauss(order + 1)
    axes, axis_weights = [], []
    for n_axis in _SPLINE_SHAPE:
        axes.append(((np.arange(n_axis)[:, None] + 0.5 * (nodes[None, :] + 1.0)) / n_axis).ravel())
        axis_weights.append(np.tile(0.5 * weights, n_axis) / n_axis)
    fa, fb, fc = np.meshgrid(*axes, indexing="ij")
    wa, wb, wc = np.meshgrid(*axis_weights, indexing="ij")
    cell = np.asarray(structure.cell, dtype=np.float64)
    r_frac = np.stack([fa.ravel(), fb.ravel(), fc.ravel()], axis=-1) + _CELL_SHIFT[None, :]
    return r_frac @ cell, (wa * wb * wc).ravel() * abs(np.linalg.det(cell))


@contextlib.contextmanager
def _fp64_reference_policy():
    """Evaluate the enclosed reference AO values with every precision zone at fp64.

    The overlap under test is fp64 in both modes, so the quadrature reference
    must not be built from the mixed policy's fp32 ``ao_eval`` values. In mixed
    mode the full policy is activated for the enclosed block and the mixed
    policy restored afterwards; in full mode this is a no-op.
    """
    was_mixed = is_mixed_precision_enabled()
    if was_mixed:
        configure("full")
    try:
        yield
    finally:
        if was_mixed:
            configure("mixed")


def _ao_values(leaf, r_carts, evaluator):
    """The leaf's AO values at ``r_carts``, evaluated with every zone at fp64."""
    with _fp64_reference_policy():
        if evaluator == "debug":
            values = np.asarray(leaf._compute_AOs_debug(r_carts))
        else:
            values = np.asarray(compute_AOs(leaf, jnp.asarray(r_carts)))
    assert np.dtype(values.dtype).itemsize == (16 if np.iscomplexobj(values) else 8)
    return values.astype(np.complex128)


def _gram(bra, ket, weights):
    """``sum_g w_g conj(bra_a(g)) ket_b(g)``."""
    return (bra.conj() * weights[None, :]) @ ket.T


def _bloch_phase(leaf, r_carts):
    k_cart = leaf.k_frac_np @ (2.0 * np.pi * np.linalg.inv(np.asarray(leaf.structure_data.cell, dtype=np.float64)).T)
    return np.exp(1j * (r_carts @ k_cart))


def _tolerance():
    # Production is exact in fp64 (det_eval); the reference integrates AO values
    # evaluated at fp64. The zone list is the declared comparison budget.
    return get_tolerance_min(("ao_eval", "det_eval"), "strict")


@pytest.mark.parametrize("twist", list(_TWISTS))
@pytest.mark.parametrize("order", [3, 5])
def test_complex_bspline_overlap_matches_exact_cell_quadrature(order, twist):
    """``compute_overlap_matrix`` equals the exact cell integral of ``conj(psi_a) psi_b``.

    Three random complex bands on a (6, 7, 8) cubic / quintic spline grid in the
    skewed triclinic cell, at Gamma and at k = (0.27, -0.1, 0.13). The reference
    integrates the production AO values (``compute_AOs``; for the cubic case also
    the plain-numpy ``_compute_AOs_debug`` oracle), evaluated with every zone at
    fp64, over the lattice-translated cell
    with the exact per-knot Gauss-Legendre rule. The matrix is complex Hermitian
    with a genuinely non-zero imaginary part.
    """
    leaf = _complex_leaf(order, twist)
    overlap = np.asarray(compute_overlap_matrix(leaf))
    assert overlap.dtype == np.complex128
    atol, rtol = _tolerance()
    assert np.all(np.isfinite(overlap))
    np.testing.assert_allclose(overlap, overlap.conj().T, atol=atol, rtol=rtol)
    assert np.max(np.abs(overlap.imag)) > atol

    r_carts, weights = _knot_gauss_legendre_cell(leaf.structure_data, order)
    evaluators = ["production", "debug"] if order == 3 else ["production"]
    for evaluator in evaluators:
        values = _ao_values(leaf, r_carts, evaluator)
        reference = _gram(values, values, weights)
        assert np.all(np.isfinite(reference))
        np.testing.assert_allclose(overlap, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fault", _FAULTS)
def test_cell_quadrature_rejects_overlap_convention_faults(fault):
    """Each wrong convention is rejected by the comparison of the reference test.

    At the general twist (cubic leaf): the conjugate on the ket (``S^T``), no
    conjugate at all (bilinear ``sum psi_a psi_b``), a Bloch phase dropped from
    the bra (``conj(u_a) psi_b``) and the old identity placeholder must each
    differ from the exact cell integral beyond the same tolerance.
    """
    leaf = _complex_leaf(3, "general-k")
    r_carts, weights = _knot_gauss_legendre_cell(leaf.structure_data, 3)
    values = _ao_values(leaf, r_carts, "production")
    reference = _gram(values, values, weights)
    overlap = np.asarray(compute_overlap_matrix(leaf))
    atol, rtol = _tolerance()
    np.testing.assert_allclose(overlap, reference, atol=atol, rtol=rtol)
    if fault == "ket-conjugated":
        candidate = overlap.T
    elif fault == "no-conjugate":
        candidate = (values * weights[None, :]) @ values.T
    elif fault == "dropped-bloch-phase-on-bra":
        candidate = _gram(values * np.conj(_bloch_phase(leaf, r_carts))[None, :], values, weights)
    else:
        candidate = np.eye(leaf.num_ao, dtype=np.complex128)
    assert np.all(np.isfinite(candidate))
    assert not np.allclose(candidate, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("order", [3, 5])
def test_complex_bspline_overlap_is_the_periodic_spline_gram(order):
    """The Bloch phase cancels: the overlap is the Gram matrix of the periodic spline basis.

    With the same real control points, the complex leaf at Gamma and at the
    general twist both return the exact cell Gram matrix of the real
    :class:`AOs_bspline_data` leaf's AO values. The time-reversal sibling
    (``with_negated_k``: negated twist, conjugated control points) of a complex
    leaf has the overlap ``conj(S)``.
    """
    structure = _structure()
    rng = np.random.default_rng(17 + order)
    real_coefficients = rng.normal(size=(3, *_SPLINE_SHAPE))
    real_leaf = AOs_bspline_data.build(structure, real_coefficients, _SPLINE_SHAPE, order)
    r_carts, weights = _knot_gauss_legendre_cell(structure, order)
    real_values = _ao_values(real_leaf, r_carts, "production").real
    real_gram = (real_values * weights[None, :]) @ real_values.T
    atol, rtol = _tolerance()
    for twist in _TWISTS.values():
        leaf = AOs_bspline_complex_data.build(structure, real_coefficients, _SPLINE_SHAPE, order, k_point=twist)
        overlap = np.asarray(compute_overlap_matrix(leaf))
        np.testing.assert_allclose(overlap, real_gram.astype(np.complex128), atol=atol, rtol=rtol)

    leaf = _complex_leaf(order, "general-k")
    overlap = np.asarray(compute_overlap_matrix(leaf))
    sibling = leaf.with_negated_k()
    np.testing.assert_allclose(np.asarray(compute_overlap_matrix(sibling)), overlap.conj(), atol=atol, rtol=rtol)


def _midpoint_gram_2n(leaf):
    """Reference real-leaf overlap: the 2 * spline_shape fractional midpoint rule."""
    cell = np.asarray(leaf.structure_data.cell, dtype=np.float64)
    axes = [(np.arange(2 * n) + 0.5) / (2 * n) for n in _SPLINE_SHAPE]
    fractional = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    values = _ao_values(leaf, fractional @ cell, "production").real
    return abs(np.linalg.det(cell)) / fractional.shape[0] * (values @ values.T)


@pytest.mark.parametrize("order", [3, 5])
def test_real_bspline_overlap_matches_exact_cell_quadrature(order):
    """The real (Gamma) B-spline leaf's overlap is the exact cell integral.

    Three random real bands on the (6, 7, 8) cubic / quintic grid in the skewed
    triclinic cell: ``compute_overlap_matrix`` (the exact spline Gram, real
    float64) equals the leaf's own AO values integrated over the lattice-translated
    cell with the exact per-knot Gauss-Legendre rule, within the strict
    ao_eval/det_eval policy. A ``2 * spline_shape`` midpoint rule
    differs from the same reference beyond that tolerance.
    """
    structure = _structure()
    rng = np.random.default_rng(23 + order)
    leaf = AOs_bspline_data.build(structure, rng.normal(size=(3, *_SPLINE_SHAPE)), _SPLINE_SHAPE, order)
    overlap = np.asarray(compute_overlap_matrix(leaf))
    assert overlap.dtype == np.float64
    atol, rtol = _tolerance()
    assert np.all(np.isfinite(overlap))
    np.testing.assert_allclose(overlap, overlap.T, atol=atol, rtol=rtol)
    r_carts, weights = _knot_gauss_legendre_cell(structure, order)
    evaluators = ["production", "debug"] if order == 3 else ["production"]
    for evaluator in evaluators:
        values = _ao_values(leaf, r_carts, evaluator).real
        reference = (values * weights[None, :]) @ values.T
        assert np.all(np.isfinite(reference))
        np.testing.assert_allclose(overlap, reference, atol=atol, rtol=rtol)
    old_midpoint = _midpoint_gram_2n(leaf)
    assert np.all(np.isfinite(old_midpoint))
    assert not np.allclose(old_midpoint, reference, atol=atol, rtol=rtol)


def _orthonormality_warnings(caplog):
    return [rec for rec in caplog.records if rec.levelno == logging.WARNING and rec.getMessage().startswith(_WARNING_PREFIX)]


def test_mo_check_measures_complex_bspline_mos_with_the_cell_overlap(caplog):
    """The MO orthonormality check measures standalone complex B-spline MOs.

    Three MOs made orthonormal with the inverse square root of the independent
    cell-quadrature Gram matrix (not the production overlap) build without a
    warning and with both metrics within the threshold in each spin channel.
    Scaling them by 5 gives one warning per channel with the normalization
    deviation 5^2 - 1, and the check never reports the leaf as skipped.
    """
    leaf = _complex_leaf(3, "general-k")
    r_carts, weights = _knot_gauss_legendre_cell(leaf.structure_data, 3)
    values = _ao_values(leaf, r_carts, "production")
    reference = _gram(values, values, weights)
    reference = 0.5 * (reference + reference.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(reference)
    inverse_sqrt = (eigenvectors / np.sqrt(eigenvalues)[None, :]) @ eigenvectors.conj().T
    # Rows of C are MOs: conj(C) S C^T = X S X = I for C = X^T, X = S^{-1/2}.
    coefficients = inverse_sqrt.T
    atol, rtol = _tolerance()
    for scale, expected_warnings in ((1.0, 0), (5.0, 2)):
        mos = MOs_data(num_mo=leaf.num_ao, aos_data=leaf, mo_coefficients=scale * coefficients)
        geminal = SD_data.build(num_electron_up=leaf.num_ao, num_electron_dn=leaf.num_ao, mos_up=mos)
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="jqmc"):
            _warn_mo_orthonormality(Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()))
        assert len(_orthonormality_warnings(caplog)) == expected_warnings
        assert not any("skipped" in rec.getMessage() for rec in caplog.records)
        for orb_data in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
            normalization, non_orthogonality = _compute_mo_orthonormality_metrics(
                orb_data, np.asarray(compute_overlap_matrix(orb_data.aos_data))
            )
            assert np.isfinite(normalization) and np.isfinite(non_orthogonality)
            assert non_orthogonality <= MO_ORTHONORMALITY_WARNING_THRESHOLD
            if scale == 1.0:
                assert normalization <= MO_ORTHONORMALITY_WARNING_THRESHOLD
            else:
                np.testing.assert_allclose(normalization, scale**2 - 1.0, atol=atol, rtol=rtol)
