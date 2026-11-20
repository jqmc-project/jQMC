"""Sampled complex overlap stability and live composite AO parameters."""

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

from jqmc import atomic_orbital as ao
from jqmc._precision import get_complex_dtype_np, get_dtype_jnp, get_dtype_np, get_tolerance_min
from jqmc.structure import _strain_structure_data
from jqmc.trexio_wrapper import read_trexio_file

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _float32_product_parts(left, right):
    """Dekker float32 product and rounding residual, without a wider oracle."""
    splitter = np.float32(4097)
    lc, rc = splitter * left, splitter * right
    lh, rh = lc - (lc - left), rc - (rc - right)
    ll, rl = left - lh, right - rh
    product = left * right
    residual = ((lh * rh - product) + lh * rl + ll * rh) + ll * rl
    return product, residual


def _float32_neumaier(values):
    """Independent sequential compensated addition in the sample dtype."""
    total = np.zeros(values.shape[:-1], dtype=np.float32)
    correction = np.zeros_like(total)
    for index in range(values.shape[-1]):
        value = values[..., index]
        updated = total + value
        correction += np.where(np.abs(total) >= np.abs(value), (total - updated) + value, (value - updated) + total)
        total = updated
    return total, correction


def _sampled_gram_reference(values):
    """Contract the same rounded samples independently; never reevaluate in full.

    The complex64 reference uses error-free product decompositions and
    Neumaier addition entirely in float32. The complex128/full reference
    retains the existing explicit-conjugation einsum. This avoids using an
    inaccurate float32 einsum as the arbiter of near-cancelling PW columns.
    """
    if values.dtype != np.complex64:
        return np.einsum("ig,jg->ij", values.conj(), values)
    ar, ai = values.real[:, None, :], values.imag[:, None, :]
    br, bi = values.real[None, :, :], values.imag[None, :, :]
    rr, rr_error = _float32_product_parts(ar, br)
    ii, ii_error = _float32_product_parts(ai, bi)
    ri, ri_error = _float32_product_parts(ar, bi)
    ir, ir_error = _float32_product_parts(ai, br)
    real, real_error = _float32_neumaier(np.concatenate((rr, ii), axis=-1))
    imag, imag_error = _float32_neumaier(np.concatenate((ri, -ir), axis=-1))
    real_low, real_low_error = _float32_neumaier(np.concatenate((rr_error, ii_error), axis=-1))
    imag_low, imag_low_error = _float32_neumaier(np.concatenate((ri_error, -ir_error), axis=-1))
    real = real + ((real_error + real_low) + real_low_error)
    imag = imag + ((imag_error + imag_low) + imag_low_error)
    return real.astype(np.complex64) + np.complex64(1j) * imag


def _component():
    fixture = Path(__file__).resolve().parents[1] / "trexio_example_files" / "H2_pbc_ae_k025_cart.h5"
    structure, pgto, *_ = read_trexio_file(str(fixture), store_tuple=True)
    pw = ao.AOs_plane_wave_complex_data(
        structure_data=structure, k_point=pgto.k_point, num_ao=3, _g_indices=(0, 0, 0, 1, 0, 0, 0, 1, 0)
    )
    return structure, pgto, pw, ao.AOs_composite_data(children=(pgto, pw))


def _midpoints(structure):
    axes = [(np.arange(6, dtype=np.float64) + 0.5) / 6] * 3
    mesh = np.meshgrid(*axes, indexing="ij")
    fractions = np.stack([axis.ravel() for axis in mesh], axis=-1)
    cell = np.asarray(structure.cell)
    return jnp.asarray(fractions @ cell), float(abs(np.linalg.det(cell))) / 216


@pytest.mark.parametrize("chunk_size", [1, 7, 31, 73])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_complex_composite_overlap_host_chunks_match_sampled_gram(monkeypatch, chunk_size):
    """Actual host chunk boundaries preserve the same sampled complex Gram.

    The numerical composite quadrature (``_compute_overlap_matrix_composite_debug``,
    the independent reference of the exact block-assembled production overlap)
    on the committed non-Gamma H2 pGTO plus three complex PW
    rows is evaluated on a
    deterministic 6x6x6 midpoint grid. Change only the documented memory-budget
    constant to force 1, 7, 31, or 73 columns per host chunk; a transparent call spy
    records actual chunk lengths while forwarding to the original evaluator.
    Compare every matrix element with independent same-sample contraction at
    existing AO/determinant strict tolerances; these component points have no
    many-electron spin state or cond(G) restriction.
    """
    structure, _, _, composite = _component()
    points, weight = _midpoints(structure)
    values = np.asarray(ao.compute_AOs(composite, points))
    expected = weight * _sampled_gram_reference(values)
    sample_dtype = np.result_type(get_dtype_np("ao_eval"), np.complex64)
    worst_rank = max(composite.num_ao, *(getattr(child, "_num_ao_kernel", child.num_ao) for child in composite.children))
    bytes_per_column = worst_rank * np.dtype(sample_dtype).itemsize
    repaired = sample_dtype == np.complex64 and jax.default_backend() == "cpu"
    if repaired:
        bytes_per_column += 2 * composite.num_ao * np.dtype(get_complex_dtype_np("det_eval")).itemsize
    monkeypatch.setattr(ao, "_COMPOSITE_OVERLAP_VALS_BUDGET_BYTES", bytes_per_column * chunk_size)
    target = "_compute_composite_overlap_AOs" if repaired else "compute_AOs"
    original_evaluator = getattr(ao, target)
    seen = []

    def observe(data, coordinates):
        # The evaluator is recursive during JIT tracing. Count only the
        # original host request, not leaf traces inside that same call.
        if data is composite:
            seen.append(int(coordinates.shape[0]))
        return original_evaluator(data, coordinates)

    monkeypatch.setattr(ao, target, observe)
    actual = ao._compute_overlap_matrix_composite_debug(composite, (6, 6, 6))
    expected_lengths = [min(chunk_size, 216 - start) for start in range(0, 216, chunk_size)]
    assert seen == expected_lengths
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval"), "strict")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
    jax.clear_caches()


@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_composite_overlap_evaluator_keeps_radial_parameters_dynamic():
    """Localized overlap AO compilation preserves nonzero radial coefficient JVP.

    On the committed non-Gamma H2 pGTO plus PW component, scaling all pGTO
    coefficients scales only its rows. The independent linearity identity
    supplies a nonzero derivative at three fixed Cartesian points. Reusing
    the same array-valued pytree with different coefficients must not capture
    old parameters or trigger a new trace per value change.
    """
    _, pgto, pw, composite = _component()
    points = jnp.asarray([[0.21, 0.41, 0.73], [1.83, 2.31, -0.42], [3.74, 0.12, 1.29]])
    coefficients = jnp.asarray(pgto.radial.coefficients)
    pgto = pgto.replace(radial=pgto.radial.replace(coefficients=coefficients))
    composite = composite.replace(children=(pgto, pw))
    evaluator = ao._compute_composite_overlap_AOs
    base = evaluator(composite, points)
    base.block_until_ready()
    initial_cache_size = evaluator._cache_size()
    modified = pgto.replace(radial=pgto.radial.replace(coefficients=coefficients * 1.125))
    changed = evaluator(composite.replace(children=(modified, pw)), points)
    changed.block_until_ready()
    assert evaluator._cache_size() == initial_cache_size

    def value(c):
        child = pgto.replace(radial=pgto.radial.replace(coefficients=c))
        return evaluator(composite.replace(children=(child, pw)), points)

    _, tangent = jax.jvp(value, (coefficients,), (coefficients,))
    expected = jnp.concatenate((base[: pgto.num_ao], jnp.zeros_like(base[pgto.num_ao :])), axis=0)
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")
    assert np.all(np.isfinite(tangent)) and np.all(np.isfinite(expected))
    assert np.max(np.abs(expected)) > atol
    np.testing.assert_allclose(tangent, expected, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(changed[: pgto.num_ao], base[: pgto.num_ao] * 1.125, atol=atol, rtol=rtol)
    np.testing.assert_allclose(changed[pgto.num_ao :], base[pgto.num_ao :], atol=atol, rtol=rtol)
    jax.clear_caches()


@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_overlap_pw_evaluator_cell_strain_matches_nonzero_analytic_identity():
    """Fixed-Cartesian PW cell strain equals minus i phi times (G+k) outer r.

    The cell changes while the three Cartesian evaluation points stay fixed,
    so the reciprocal-vector derivative is nonzero. Full and mixed each use
    their own AO arithmetic and the existing strict AO derivative policy.
    This probes a live cell input to the localized compiled evaluator, not
    a zero identity from co-straining both positions and cell.
    """
    structure, _, pw, _ = _component()
    points = jnp.asarray([[0.21, 0.41, 0.73], [1.83, 2.31, -0.42], [3.74, 0.12, 1.29]])
    evaluator = ao._compute_composite_overlap_AOs

    def value(strain):
        deformed = _strain_structure_data(structure, jnp.eye(3) + strain)
        return evaluator(pw.replace(structure_data=deformed), points)

    actual = jax.jacfwd(value)(jnp.zeros((3, 3)))
    phi = evaluator(pw, points)
    gk = pw._gk_vectors_jnp.astype(get_dtype_jnp("ao_eval"))
    position = points.astype(get_dtype_jnp("ao_eval"))
    expected = -jnp.asarray(1j, dtype=phi.dtype) * phi[:, :, None, None] * gk[:, None, :, None] * position[None, :, None, :]
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    assert np.max(np.abs(expected)) > atol
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
    jax.clear_caches()
