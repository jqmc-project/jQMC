"""Exact block assembly of the periodic composite AO overlap.

``compute_overlap_matrix`` on a periodic ``AOs_composite_data`` takes every
diagonal block from that child's own exact / analytic overlap and every cross
block from a closed form (pGTO x plane wave: Gaussian Fourier transform; plane
wave x B-spline: exact spline Fourier coefficient; equal families: the
children's own exact kernels) or an exact reciprocal-space sum (pGTO x
B-spline), with the bra conjugated. No real-space quadrature is involved, so
the result is the same fp64 ``det_eval`` matrix in both precision modes.

Independent references (``tests/_composite_overlap_reference.py`` and this
file): the closed-form Fourier transform written out separately, an
axis-factorised one-dimensional integral of Gaussian x B-spline (orthorhombic
cells), and a fine cell quadrature of the children's own AO values (midpoint
for smooth partners, Gauss-Legendre on every knot interval for B-splines).
The cross blocks under test are fp64 in both precision modes, so (reverse
direction) the quadrature samples the AO values with every precision
zone at fp64 (``_fp64_reference_policy``; a no-op in full mode, so the
full-mode references are numerically unchanged), never the mixed policy's fp32
``ao_eval`` values.

Time-reversed cases also exercise lazily conjugated spline control tables.

Fixtures: the committed H2 periodic Cartesian pGTO leaves (cubic cell, Gamma
and k = (1/4, 1/4, 1/4)) with a plane-wave leaf (real cos/sin at Gamma,
complex at the twist) or a cubic B-spline leaf (real at Gamma, complex at the
twist) on a (6, 7, 8) knot grid; the QE Li ccECP pgto+pw / pgto+bspline
conversions (exponent up to 190) for the PSD and S-metric checks.
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
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import atomic_orbital as ao
from jqmc._precision import configure, get_tolerance, get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import get_s_stiefel_backward_error_limit
from jqmc.atomic_orbital import (
    AOs_bspline_complex_data,
    AOs_bspline_data,
    AOs_composite_data,
    AOs_plane_wave_complex_data,
    AOs_plane_wave_data,
    compute_AOs,
    compute_overlap_matrix,
)
from jqmc.trexio_wrapper import read_trexio_file
from tests._test_data_cache import qe_example_files_root
from tests._composite_overlap_reference import pgto_bspline_cross, pgto_plane_wave_cross

pytestmark = [pytest.mark.without_chdir]

_FIXTURES = {"gamma": "H2_pbc_ae_cart.h5", "twist": "H2_pbc_ae_k025_cart.h5"}
_MILLERS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, -1, 1]], dtype=np.float64)
_SPLINE_SHAPE = (6, 7, 8)
_SPLINE_ORDER = 3
_CASES = [("gamma", "plane-wave"), ("gamma", "bspline"), ("twist", "plane-wave"), ("twist", "bspline")]
_FAULTS = ["ket-conjugated-cross", "dropped-twist-phase", "dropped-diagonal-block", "old-quadrature-grid"]
_QE_ROOT = qe_example_files_root()
_QE_SAVES = {"gamma": "Li_gamma/out/li.save", "twist": "Li_k025/out/li_k025.save"}


def _pgto(twist):
    path = Path(__file__).resolve().parents[1] / "trexio_example_files" / _FIXTURES[twist]
    structure, pgto, *_ = read_trexio_file(str(path), store_tuple=True)
    assert structure.pbc_flag and (pgto.k_point is None) == (twist == "gamma")
    return structure, pgto


def _plane_waves(structure, pgto, millers=_MILLERS):
    reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(structure.cell, dtype=np.float64)).T
    g_vectors = millers @ reciprocal
    if pgto.k_point is not None:
        return AOs_plane_wave_complex_data.build(structure, g_vectors, k_point=pgto.k_point)
    paired = np.concatenate([g_vectors[:1], np.repeat(g_vectors[1:], 2, axis=0)])
    return AOs_plane_wave_data.build(structure, paired, np.array([0] + [0, 1] * (len(g_vectors) - 1)))


def _spline(structure, pgto, seed=11, shape=_SPLINE_SHAPE):
    rng = np.random.default_rng(seed)
    size = (2, *shape)
    if pgto.k_point is not None:
        controls = rng.normal(size=size) + 1j * rng.normal(size=size)
        return AOs_bspline_complex_data.build(structure, controls, shape, _SPLINE_ORDER, k_point=pgto.k_point)
    return AOs_bspline_data.build(structure, rng.normal(size=size), shape, _SPLINE_ORDER)


def _composite(twist, kind):
    if twist == "time-reversed":
        structure, pgto, partner, composite = _composite("twist", kind)
        reversed_composite = composite.with_negated_k()
        return structure, *reversed_composite.children, reversed_composite
    structure, pgto = _pgto(twist)
    partner = _plane_waves(structure, pgto) if kind == "plane-wave" else _spline(structure, pgto)
    return structure, pgto, partner, AOs_composite_data.build((pgto, partner))


def _midpoint_rule(n):
    return tuple(((np.arange(n, dtype=np.float64) + 0.5) / n, np.full(n, 1.0 / n)) for _ in range(3))


def _knot_gauss_legendre_rule(pieces, nodes, shape=_SPLINE_SHAPE):
    """``pieces`` equal sub-intervals per knot interval, ``nodes`` Gauss-Legendre nodes on each."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    rule = []
    for knots in shape:
        n = knots * pieces
        left = np.arange(n, dtype=np.float64) / n
        rule.append(((left[:, None] + 0.5 * (x[None, :] + 1.0) / n).ravel(), np.tile(0.5 * w / n, n)))
    return tuple(rule)


@contextlib.contextmanager
def _fp64_reference_policy():
    """Evaluate the enclosed reference AO samples with every precision zone at fp64.

    The overlap blocks under test are fp64 in both modes, so their quadrature
    reference must not be built from the mixed policy's fp32 ``ao_eval``
    values. In mixed mode the full policy is activated for the enclosed block
    and the mixed policy restored afterwards; in full mode this is a no-op.
    """
    was_mixed = is_mixed_precision_enabled()
    if was_mixed:
        configure("full")
    try:
        yield
    finally:
        if was_mixed:
            configure("mixed")


def _sampled_cross(bra, ket, rule, cell, *, conjugate_bra=True):
    """Fine cell quadrature ``sum_g w_g conj(bra_a(g)) ket_b(g)`` of fp64 AO samples (complex128 sum)."""
    (x0, w0), (x1, w1), (x2, w2) = rule
    fractional = np.stack(np.meshgrid(x0, x1, x2, indexing="ij"), axis=-1).reshape(-1, 3)
    weights = np.einsum("i,j,k->ijk", w0, w1, w2).ravel() * abs(np.linalg.det(cell))
    points = jnp.asarray(fractional @ cell)
    with _fp64_reference_policy():
        raw_a = np.asarray(compute_AOs(bra, points))
        raw_b = np.asarray(compute_AOs(ket, points))
    for raw in (raw_a, raw_b):
        assert np.dtype(raw.dtype).itemsize == (16 if np.iscomplexobj(raw) else 8)
    values_a = raw_a.astype(np.complex128)
    values_b = raw_b.astype(np.complex128)
    bra_values = values_a.conj() if conjugate_bra else values_a
    return (bra_values * weights[None, :]) @ values_b.T


def _fine_rule(kind):
    return _midpoint_rule(64) if kind == "plane-wave" else _knot_gauss_legendre_rule(2, 12)


def _analytic_reference(pgto, partner, kind):
    return pgto_plane_wave_cross(pgto, partner) if kind == "plane-wave" else pgto_bspline_cross(pgto, partner)


def _assert_positive_semidefinite(overlap):
    eigenvalues = np.linalg.eigvalsh(overlap)
    assert np.all(np.isfinite(eigenvalues)) and eigenvalues[-1] > 0.0
    limit = get_s_stiefel_backward_error_limit(overlap.dtype, overlap.shape[0])
    assert eigenvalues[0] >= -limit * eigenvalues[-1], f"overlap not PSD: lambda_min={eigenvalues[0]:.3e}"


@pytest.mark.light
@pytest.mark.parametrize(("twist", "kind"), _CASES + [("time-reversed", "bspline")])
def test_composite_overlap_diagonal_blocks_are_the_children_overlaps(twist, kind):
    """Diagonal blocks are the children's own overlaps; the matrix is Hermitian PSD.

    pGTO + plane-wave and pGTO + B-spline composites at Gamma (real) and at the
    twist (complex): each diagonal block equals ``compute_overlap_matrix`` of
    that child alone, the lower cross block is the conjugate transpose of the
    upper one (strict det_eval policy), and the smallest eigenvalue is above
    minus the S-Stiefel backward-error bound times the largest.
    """
    _, pgto, partner, composite = _composite(twist, kind)
    overlap = np.asarray(compute_overlap_matrix(composite))
    n = pgto.num_ao
    assert overlap.shape == (n + partner.num_ao, n + partner.num_ao)
    assert overlap.dtype == (np.complex128 if twist != "gamma" else np.float64)
    assert np.all(np.isfinite(overlap))
    atol, rtol = get_tolerance("det_eval", "strict")
    for block, child in ((overlap[:n, :n], pgto), (overlap[n:, n:], partner)):
        own = np.asarray(compute_overlap_matrix(child))
        assert np.all(np.isfinite(own)) and np.max(np.abs(own)) > atol
        np.testing.assert_allclose(block, own, atol=atol, rtol=rtol)
    np.testing.assert_allclose(overlap[n:, :n], overlap[:n, n:].conj().T, atol=atol, rtol=rtol)
    _assert_positive_semidefinite(overlap)
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize(("twist", "kind"), _CASES + [("time-reversed", "bspline")])
def test_composite_cross_block_matches_analytic_reference(twist, kind):
    """The cross block equals an independently written exact reference.

    Plane-wave partner: the closed-form Fourier transform of the Cartesian
    Gaussian (Hermite functions; cos / sin take the real / imaginary part at
    Gamma). B-spline partner: the axis-factorised one-dimensional integral of
    Gaussian x B-spline (``scipy`` cardinal B-spline, 24 Gauss-Legendre nodes on
    8 pieces of every knot interval). Both production and reference are fp64
    quantities independent of the AO precision (the overlap lives in the fp64
    det_eval zone in both modes); strict det_eval policy.
    """
    _, pgto, partner, composite = _composite(twist, kind)
    n = pgto.num_ao
    actual = np.asarray(compute_overlap_matrix(composite))[:n, n:]
    reference = _analytic_reference(pgto, partner, kind)
    atol, rtol = get_tolerance("det_eval", "strict")
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    assert np.max(np.abs(reference)) > 1.0e3 * atol
    if twist != "gamma":
        assert np.max(np.abs(reference.imag)) > 1.0e3 * atol
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize(("twist", "kind"), _CASES + [("time-reversed", "bspline")])
def test_composite_cross_block_matches_fine_cell_quadrature(twist, kind):
    """The cross block equals a fine cell quadrature of the children's own AO values.

    ``sum_g w_g conj(chi_pGTO(g)) chi_partner(g)`` on a 64^3 midpoint rule
    (plane-wave partner; spectrally accurate for the smooth periodic integrand
    at the H2 exponents <= 13) or 2 x 12 Gauss-Legendre nodes on every knot
    interval (B-spline partner; the spline is a polynomial on each interval),
    with the AO values evaluated at fp64 in both modes (the block under
    test is fp64); strict ao_eval/det_eval policy in units of the block's
    largest element.
    """
    structure, pgto, partner, composite = _composite(twist, kind)
    cell = np.asarray(structure.cell, dtype=np.float64)
    n = pgto.num_ao
    actual = np.asarray(compute_overlap_matrix(composite))[:n, n:]
    reference = _sampled_cross(pgto, partner, _fine_rule(kind), cell)
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval"), "strict")
    scale = float(np.max(np.abs(reference)))
    assert np.all(np.isfinite(reference)) and scale > 1.0e3 * atol
    # Compared in units of the block's largest element: the AO samples carry
    # their rounding (and the rule its quadrature error) relative to the
    # integrand scale, so entries that vanish by symmetry are reproduced only
    # to that level times the scale.
    np.testing.assert_allclose(actual / scale, reference / scale, atol=atol, rtol=rtol)
    jax.clear_caches()


def _old_quadrature_overlap(composite):
    """Debug-route algorithm: one midpoint quadrature of the whole matrix."""
    return np.asarray(ao._compute_overlap_matrix_composite_debug(composite))


@pytest.mark.light
@pytest.mark.parametrize("fault", _FAULTS)
def test_composite_overlap_rejects_convention_faults(fault):
    """Each convention fault is rejected by the comparisons above.

    Twisted pGTO + complex plane-wave composite: a ket-conjugated cross block
    and a cross block that drops the Bloch twist from the plane-wave momentum
    (``q = G`` instead of ``G + k``) fail the analytic reference; a dropped
    (zero) diagonal block and the previous production algorithm (one midpoint
    quadrature of the whole matrix on the old default grid) fail the comparison
    with the children's own overlaps.
    """
    _, pgto, partner, composite = _composite("twist", "plane-wave")
    n = pgto.num_ao
    overlap = np.asarray(compute_overlap_matrix(composite))
    atol, rtol = get_tolerance("det_eval", "strict")
    reference = _analytic_reference(pgto, partner, "plane-wave")
    np.testing.assert_allclose(overlap[:n, n:], reference, atol=atol, rtol=rtol)
    own = (np.asarray(compute_overlap_matrix(pgto)), np.asarray(compute_overlap_matrix(partner)))
    if fault == "ket-conjugated-cross":
        candidate = overlap[:n, n:].conj()
    elif fault == "dropped-twist-phase":
        candidate = ao._pgto_fourier_transform(pgto, np.asarray(partner.g_vectors))
    else:
        if fault == "dropped-diagonal-block":
            candidate = overlap.copy()
            candidate[n:, n:] = 0.0
        else:
            candidate = _old_quadrature_overlap(composite)
        assert np.all(np.isfinite(candidate))
        assert not (
            np.allclose(candidate[:n, :n], own[0], atol=atol, rtol=rtol)
            and np.allclose(candidate[n:, n:], own[1], atol=atol, rtol=rtol)
        )
        return
    assert np.all(np.isfinite(candidate))
    assert not np.allclose(candidate, reference, atol=atol, rtol=rtol)
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize("twist", ["gamma", "twist", "time-reversed"])
def test_composite_same_family_and_plane_wave_bspline_pairs_are_exact(twist):
    """Every other child pair has an exact cross block or raises.

    Duplicated children (pGTO, plane-wave, B-spline) give cross blocks equal to
    the child's own overlap (merged analytic lattice overlap, closed form,
    concatenated exact spline Gram; strict det_eval). The plane-wave x B-spline
    block (exact spline Fourier coefficients) equals the fine knot
    Gauss-Legendre quadrature of the AO values (strict ao_eval/det_eval).
    Children at different twists raise ``ValueError``; two B-spline children on
    different knot grids raise ``NotImplementedError``.
    """
    structure, pgto = _pgto("twist" if twist == "time-reversed" else twist)
    cell = np.asarray(structure.cell, dtype=np.float64)
    plane_waves, spline = _plane_waves(structure, pgto), _spline(structure, pgto)
    if twist == "time-reversed":
        pgto, plane_waves, spline = (child.with_negated_k() for child in (pgto, plane_waves, spline))
    atol, rtol = get_tolerance("det_eval", "strict")
    for child in (pgto, plane_waves, spline):
        block = np.asarray(ao._composite_cross_overlap_block(child, child))
        np.testing.assert_allclose(block, np.asarray(compute_overlap_matrix(child)), atol=atol, rtol=rtol)
    actual = np.asarray(ao._composite_cross_overlap_block(plane_waves, spline))
    reference = _sampled_cross(plane_waves, spline, _knot_gauss_legendre_rule(2, 12), cell)
    atol_x, rtol_x = get_tolerance_min(("ao_eval", "det_eval"), "strict")
    scale = float(np.max(np.abs(reference)))
    assert scale > 1.0e3 * atol_x
    np.testing.assert_allclose(actual / scale, reference / scale, atol=atol_x, rtol=rtol_x)
    np.testing.assert_allclose(
        np.asarray(ao._composite_cross_overlap_block(spline, plane_waves)), actual.conj().T, atol=atol, rtol=rtol
    )
    with pytest.raises(NotImplementedError, match="same knots"):
        ao._composite_cross_overlap_block(spline, _spline(structure, pgto, shape=(6, 7, 9)))
    other_twist = AOs_plane_wave_complex_data.build(structure, np.zeros((1, 3)), k_point=(0.125, 0.0, 0.0))
    with pytest.raises(ValueError, match="different twists"):
        ao._composite_cross_overlap_block(pgto, other_twist)
    jax.clear_caches()


def _qe_li_composite(twist, kind):
    from jqmc.pw_pgto_projection import project_pw_to_mixed, project_pw_to_pgto_spline
    from jqmc.qe_to_jqmc import build_wavefunction_from_projection
    from jqmc.qe_wrapper import load_qe_reference
    from jqmc.upf_wrapper import UPFExponentInit

    reference = load_qe_reference(str(_QE_ROOT / _QE_SAVES[twist]))
    g_max = float(np.sqrt(2.0 * reference.ecutwfc))
    strategy = UPFExponentInit({"Li": str(_QE_ROOT / "Li.ccECP.upf")}, g_target=g_max)
    shells = [strategy.shells_for_element(label) for label in reference.atom_labels]
    if kind == "plane-wave":
        projection = project_pw_to_mixed(reference, shells, g_cut=0.5 * g_max)
        name = "pgto+pw"
    else:
        projection = project_pw_to_pgto_spline(reference, shells, mesh_factor=0.5, order=5)
        name = "pgto+bspline"
    _, aos, geminal = build_wavefunction_from_projection(reference, projection, kind=name)
    return aos, geminal


@pytest.mark.heavy
@pytest.mark.parametrize(
    ("twist", "kind"), [("gamma", "plane-wave"), ("gamma", "bspline"), ("twist", "plane-wave"), ("twist", "bspline")]
)
def test_qe_li_composite_overlap_is_exact_psd_and_feeds_the_s_metric(twist, kind):
    """QE Li ccECP composites (pGTO exponent up to 190): exact, PSD, usable by the S-metric and the MO check.

    The cross block equals the analytic reference (closed-form Fourier
    transform / axis-factorised integral; strict det_eval), the assembled
    overlap is PSD within the S-Stiefel backward-error bound, the S-Stiefel
    metric factors build without error, and the MO orthonormality metrics are
    finite. The previous quadrature (old default grid) misses the analytic
    cross block by more than the same tolerance.
    """
    from jqmc.hamiltonians import _compute_mo_orthonormality_metrics
    from jqmc.wavefunction import _build_s_stiefel_metric_factors

    aos, geminal = _qe_li_composite(twist, kind)
    pgto, partner = aos.children
    n = pgto.num_ao
    overlap = np.asarray(compute_overlap_matrix(aos))
    reference = _analytic_reference(pgto, partner, kind)
    atol, rtol = get_tolerance("det_eval", "strict")
    assert np.max(np.abs(reference)) > 1.0e3 * atol
    np.testing.assert_allclose(overlap[:n, n:], reference, atol=atol, rtol=rtol)
    _assert_positive_semidefinite(overlap)
    sqrt_factor, inv_sqrt_factor = _build_s_stiefel_metric_factors(overlap)
    assert np.all(np.isfinite(sqrt_factor)) and np.all(np.isfinite(inv_sqrt_factor))
    normalization, non_orthogonality = _compute_mo_orthonormality_metrics(geminal.orb_data_up_spin, overlap)
    assert np.isfinite(normalization) and np.isfinite(non_orthogonality)
    old = _old_quadrature_overlap(aos)
    assert not np.allclose(old[:n, n:], reference, atol=atol, rtol=rtol)
    jax.clear_caches()
