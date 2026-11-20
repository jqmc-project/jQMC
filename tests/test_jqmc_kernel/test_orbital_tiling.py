"""Common tiling contract: independent cosets, local contractions and Bloch VGL."""

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

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_complex_dtype_np, get_dtype_np, get_tolerance_min
from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_BIT_EXACT
from jqmc.atomic_orbital import (
    AOs_bspline_complex_data,
    AOs_plane_wave_complex_data,
    build_AOs_cart_pgto_complex_data,
    build_AOs_sphe_pgto_complex_data,
)
from jqmc.orbital_tiling import (
    OrbitalKBlock_data,
    TiledOrbitalSet_data,
    _compute_tiled_orbitals_debug,
    _compute_tiled_orbitals_value_grad_lap_autodiff,
    compute_tiled_orbitals,
    compute_tiled_orbitals_grad,
    compute_tiled_orbitals_laplacian,
    compute_tiled_orbitals_value_grad_lap,
    match_tiling_k_points,
    tiled_overlap_blocks,
    tiling_k_points,
)
from jqmc.structure import Structure_data, normalize_tiling_matrix, tiling_translation_fractions

pytestmark = [pytest.mark.without_chdir]


def _primitive():
    return Structure_data.build(
        positions=np.array([[0.3, 0.4, 0.5]]),
        pbc_flag=True,
        vec_a=(3.0, 0.0, 0.0),
        vec_b=(0.2, 3.5, 0.0),
        vec_c=(0.1, 0.3, 4.0),
        atomic_numbers=(2,),
        element_symbols=("He",),
        atomic_labels=("He1",),
    )


@pytest.mark.light
def test_exact_geometry_and_order_independent_shifted_cosets():
    """Integer quotient sizes and characters agree with independent lattice identities."""
    matrices = (
        ((2, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((1, 1, 0), (1, -1, 0), (0, 0, 1)),
        ((2, 1, 0), (0, 2, 1), (0, 0, 2)),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )
    for matrix, count in zip(matrices, (2, 2, 8, 1, 1), strict=True):
        for twist in ((0.0, 0.0, 0.0), (0.17, -0.23, 0.31)):
            k = tiling_k_points(matrix, twist)
            translations = tiling_translation_fractions(matrix)
            assert k.shape == translations.shape == (count, 3)
            assert len(set(map(tuple, k))) == count
            assert len(set(map(tuple, translations))) == count
            permutation = np.random.default_rng(8).permutation(count)
            recovered = match_tiling_k_points(matrix, k[permutation], twist)
            np.testing.assert_array_equal(k[permutation][list(recovered)], k)
            unfolded = k @ np.asarray(matrix).T - twist
            np.testing.assert_allclose(unfolded, np.rint(unfolded), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_BIT_EXACT)
            real_translations = translations @ matrix
            np.testing.assert_allclose(
                real_translations, np.rint(real_translations), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_BIT_EXACT
            )
            # Character orthogonality independently checks both quotients and
            # the transpose convention. A dense matrix is only a small oracle.
            characters = np.exp(2j * np.pi * real_translations @ (k - k[0]).T)
            np.testing.assert_allclose(
                characters.conj().T @ characters,
                count * np.eye(count),
                atol=ATOL_PYTHON_FP64_LOOSE,
                rtol=RTOL_BIT_EXACT,
            )
    # det is exactly +1, but floating det/inverse loses the unit difference.
    large = 10**9
    ill_conditioned = ((large, large - 1, 0), (large + 1, large, 0), (0, 0, 1))
    np.testing.assert_array_equal(normalize_tiling_matrix(ill_conditioned), ill_conditioned)
    np.testing.assert_array_equal(tiling_translation_fractions(ill_conditioned), np.zeros((1, 3)))
    huge = ((1, 2**53 + 1, 0), (0, 1, 0), (0, 0, -1))
    assert normalize_tiling_matrix(huge)[0, 1] == 2**53 + 1
    np.testing.assert_array_equal(tiling_translation_fractions(huge), np.zeros((1, 3)))


@pytest.mark.light
def test_common_model_rejects_invalid_geometry_and_blocks():
    """Malformed tilings, sectors, local matrices and schema versions fail explicitly."""
    for invalid in ((0, 1, 1), (1.5, 1, 1), (np.inf, 1, 1), (True, 1, 1), (2**63, 1, 1), ("1", 1, 1)):
        with pytest.raises(ValueError):
            normalize_tiling_matrix(invalid)
    for points in ([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)] * 2, [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0)]):
        with pytest.raises(ValueError):
            match_tiling_k_points((2, 1, 1), points)
    with pytest.raises(ValueError, match="resolvable"):
        match_tiling_k_points((1, 1, 1), [(2**54, 0.0, 0.0)])
    orbitals = _pw_set()
    block = orbitals.blocks[0]
    for bad in (
        block.replace(coefficients=np.zeros((1, 1))),
        block.replace(source_band_indices=(1, 1)),
        block.replace(source_band_indices=(-1,)),
        block.replace(coefficients=None),
    ):
        with pytest.raises(ValueError):
            bad.sanity_check()
    for bad in (
        orbitals.replace(blocks=()),
        orbitals.replace(normalization="supercell-scaled-prototype"),
        orbitals.replace(spin="unknown"),
        orbitals.replace(blocks=orbitals.blocks[::-1]),
        orbitals.replace(structure_data=_primitive()),
    ):
        with pytest.raises(ValueError):
            bad.sanity_check()


def _pw_set():
    primitive = _primitive()
    matrix = ((1, 1, 0), (1, -1, 0), (0, 0, 1))
    twist = (0.17, -0.23, 0.31)
    blocks = []
    rng = np.random.default_rng(12)
    for i, k in enumerate(tiling_k_points(matrix, twist)):
        indices = np.asarray([(0, 0, 0), (1, 0, 0), (0, -1, 0)][: i + 2])
        basis = AOs_plane_wave_complex_data.build(primitive, indices @ primitive.recip_cell, k_point=tuple(k))
        raw = rng.normal(size=(basis.num_ao, i + 1)) + 1j * rng.normal(size=(basis.num_ao, i + 1))
        q, _ = np.linalg.qr(raw)
        c = q.T / np.sqrt(abs(np.linalg.det(primitive.cell)))
        blocks.append(OrbitalKBlock_data.build(backend=basis, coefficients=c, source_band_indices=tuple(range(i + 1))))
    return TiledOrbitalSet_data.build(blocks=blocks[::-1], tiling_matrix=matrix, supertwist=twist)


def _explicit_supercell_pw(orbitals, positions):
    """Dense reference built from supercell reciprocal indices, only for tests."""
    count = len(orbitals.blocks)
    rows = []
    all_q = []
    offset = 0
    n_basis = sum(b.backend.num_ao for b in orbitals.blocks)
    for block in orbitals.blocks:
        indices = np.asarray(block.backend._g_indices).reshape(-1, 3)
        unfolded = (indices + block.k_point) @ np.asarray(orbitals.tiling_matrix).T - orbitals.supertwist
        super_g = np.rint(unfolded)
        np.testing.assert_allclose(unfolded, super_g, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_BIT_EXACT)
        all_q.extend((super_g + orbitals.supertwist) @ orbitals.structure_data.recip_cell)
        c = np.zeros((block.num_orbitals, n_basis), dtype=np.complex128)
        c[:, offset : offset + block.backend.num_ao] = block.coefficients / np.sqrt(count)
        rows.append(c)
        offset += block.backend.num_ao
    coefficients = np.concatenate(rows)
    wavevectors = np.asarray(all_q)
    results = []
    for i in range(5):
        zone = "ao_eval" if i == 0 else "ao_grad_lap"
        q = wavevectors.astype(get_dtype_np(zone))
        phase = np.exp(get_complex_dtype_np(zone)(1j) * (q @ positions.astype(get_dtype_np(zone)).T))
        if 1 <= i <= 3:
            phase *= 1j * q[:, i - 1, None]
        elif i == 4:
            phase *= -np.sum(q * q, axis=1)[:, None]
        results.append(coefficients @ phase)
    return results


@pytest.mark.light
def test_ragged_pw_matches_explicit_supercell_vgl_determinant_and_kinetic():
    """Nonidentity ragged blocks equal an independently assembled supercell PW matrix."""
    orbitals = _pw_set()
    positions = np.asarray([[0.31, 0.48, 0.61], [1.21, 0.74, 0.38], [2.43, 1.13, 0.91]])
    actual = compute_tiled_orbitals_value_grad_lap(orbitals, positions)
    reference = _explicit_supercell_pw(orbitals, positions)
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap", "tiling_eval"), "medium")
    for value, expected in zip(actual, reference, strict=True):
        np.testing.assert_allclose(value, expected, atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual[0], _compute_tiled_orbitals_debug(orbitals, positions), atol=atol, rtol=rtol)
    for value, expected in zip(
        actual, _compute_tiled_orbitals_value_grad_lap_autodiff(orbitals, jnp.asarray(positions)), strict=True
    ):
        np.testing.assert_allclose(value, expected, atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.linalg.det(actual[0]), np.linalg.det(reference[0]), atol=atol, rtol=rtol)
    kinetic = -0.5 * np.trace(np.linalg.solve(np.asarray(actual[0]), np.asarray(actual[4])))
    reference_kinetic = -0.5 * np.trace(np.linalg.solve(reference[0], reference[4]))
    np.testing.assert_allclose(kinetic, reference_kinetic, atol=atol, rtol=rtol)
    for local in tiled_overlap_blocks(orbitals):
        np.testing.assert_allclose(local, np.eye(len(local)), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_BIT_EXACT)
    assert orbitals.orbital_offsets == (0, 1, 3)
    assert tuple(b.coefficients.shape for b in orbitals.blocks) == ((1, 2), (2, 3))
    assert not hasattr(orbitals, "mo_coefficients")


def _backend_set(backend):
    if backend.startswith("composite"):
        from tests.test_jqmc_kernel._orbital_tiling_reference import add_pw_component, complete_pgto_source

        return add_pw_component(complete_pgto_source(backend.replace("composite", "pgto")), reverse=backend.endswith("cart"))
    if backend == "pw":
        return _pw_set()
    primitive = _primitive()
    matrix = ((2, 1, 0), (0, 1, 0), (0, 0, 1))
    twist = (0.2, 0.1, -0.3)
    blocks = []
    for k in tiling_k_points(matrix, twist):
        if backend == "bspline":
            coefficients = np.random.default_rng(9).normal(size=(1, 4**3)).astype(np.complex128) * (1 + 0.3j)
            basis = AOs_bspline_complex_data.build(primitive, coefficients, (4, 4, 4), spline_order=3, k_point=tuple(k))
            local = None
        else:
            common = dict(
                structure_data=primitive,
                nucleus_index=(0, 0),
                num_ao=2,
                num_ao_prim=2,
                orbital_indices=(0, 1),
                exponents=np.array([0.8, 1.2]),
                coefficients=np.ones(2),
                angular_momentums=(0, 0),
                k_point=tuple(k),
            )
            if backend == "pgto_cart":
                basis = build_AOs_cart_pgto_complex_data(
                    **common, polynominal_order_x=(0, 0), polynominal_order_y=(0, 0), polynominal_order_z=(0, 0)
                )
            else:
                basis = build_AOs_sphe_pgto_complex_data(**common, magnetic_quantum_numbers=(0, 0))
            local = np.array([[0.7 + 0.2j, -0.3 + 0.4j]])
        blocks.append(OrbitalKBlock_data.build(backend=basis, coefficients=local, source_band_indices=(2,)))
    return TiledOrbitalSet_data.build(blocks=blocks, tiling_matrix=matrix, supertwist=twist)


@pytest.mark.parametrize("backend", ["bspline", "pgto_cart", "pgto_sphe", "composite_cart", "composite_sphe"])
@pytest.mark.heavy
def test_backends_share_block_vgl_translation_and_conjugation_contract(backend):
    """Identity spline and nonidentity pGTO operators obey the same final-orbital API."""
    orbitals = _backend_set(backend)
    matrix, twist = orbitals.tiling_matrix, orbitals.supertwist
    positions = np.array([[0.51, 0.62, 0.73], [1.13, 0.41, 1.21]])
    actual = compute_tiled_orbitals_value_grad_lap(orbitals, positions)
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap", "tiling_eval"), "medium")
    np.testing.assert_allclose(actual[0], _compute_tiled_orbitals_debug(orbitals, positions), atol=atol, rtol=rtol)
    separate = (
        compute_tiled_orbitals(orbitals, positions),
        *compute_tiled_orbitals_grad(orbitals, positions),
        compute_tiled_orbitals_laplacian(orbitals, positions),
    )
    for fused, alone, auto in zip(
        actual, separate, _compute_tiled_orbitals_value_grad_lap_autodiff(orbitals, jnp.asarray(positions)), strict=True
    ):
        np.testing.assert_allclose(fused, alone, atol=atol, rtol=rtol)
        np.testing.assert_allclose(fused, auto, atol=atol, rtol=rtol)
    for i, translation in enumerate(orbitals.structure_data.cell):
        shifted = compute_tiled_orbitals_value_grad_lap(orbitals, positions + translation)
        for original, moved in zip(actual, shifted, strict=True):
            np.testing.assert_allclose(moved, np.exp(2j * np.pi * twist[i]) * original, atol=atol, rtol=rtol)
    siblings = [b.replace(conjugate=True) for b in orbitals.blocks]
    partner = TiledOrbitalSet_data.build(blocks=siblings, tiling_matrix=matrix, supertwist=-np.asarray(twist), spin="down")
    for b in partner.blocks:
        assert any(b.backend is original.backend and b.coefficients is original.coefficients for original in orbitals.blocks)
    # Match partners by band/backend identity, not their differently sorted -k cosets.
    order = [next(i for i, b in enumerate(partner.blocks) if b.backend is original.backend) for original in orbitals.blocks]
    for value, conjugated in zip(actual, compute_tiled_orbitals_value_grad_lap(partner, positions), strict=True):
        np.testing.assert_allclose(np.asarray(conjugated)[order], np.conj(value), atol=atol, rtol=rtol)


@pytest.mark.light
def test_empty_occupied_k_block_keeps_complete_sector():
    """A zero-row local operator preserves its k coset without adding orbital rows."""
    original = _pw_set()
    empty = original.blocks[0].replace(coefficients=np.empty((0, 2), dtype=np.complex128), source_band_indices=())
    orbitals = original.replace(blocks=(empty, original.blocks[1]))
    orbitals.sanity_check()
    positions = np.array([[0.3, 0.4, 0.7]])
    expected = compute_tiled_orbitals(original, positions)[1:]
    np.testing.assert_array_equal(compute_tiled_orbitals(orbitals, positions), expected)
    assert orbitals.orbital_offsets == (0, 0, 2)


@pytest.mark.parametrize("backend", ["pw", "bspline", "pgto_cart", "pgto_sphe", "composite_cart", "composite_sphe"])
@pytest.mark.parametrize("restricted", [True, False])
@pytest.mark.heavy
def test_tiled_sd_preserves_bcs_spin_sectors_and_kinetic(backend, restricted):
    """Final block orbitals feed SD, with opposite spin phases and independent LSDA rows."""
    from jqmc.determinant import (
        SD_data,
        _compute_geminal_all_elements_debug,
        compute_det_geminal_all_elements,
        compute_geminal_all_elements,
        compute_grads_and_laplacian_ln_Det,
        compute_ln_det_geminal_all_elements,
    )

    up_source = _backend_set(backend)
    down_source = None
    if not restricted:
        # Fewer occupied down rows, independent complex coefficients, and an
        # empty k block. No equality-of-populations or global-prefix heuristic.
        blocks = []
        for i, block in enumerate(up_source.blocks):
            local = np.eye(block.backend.num_ao, dtype=np.complex128) if block.coefficients is None else block.coefficients
            local = (0.7 + 0.2j) * local
            ids = block.source_band_indices
            if i == 0:
                local, ids = local[:0], ()
            blocks.append(block.replace(coefficients=local, source_band_indices=ids))
        down_source = up_source.replace(blocks=tuple(blocks), spin="down")
    sd = SD_data.build_tiled(up_source=up_source, down_source=down_source)
    assert sd.time_reversal is restricted
    assert sd.orb_data_dn_spin.supertwist == tuple(-x for x in sd.orb_data_up_spin.supertwist)
    rng = np.random.default_rng(55)
    up = rng.uniform(size=(sd.num_electron_up, 3)) @ up_source.structure_data.cell
    down = rng.uniform(size=(sd.num_electron_dn, 3)) @ up_source.structure_data.cell
    up_vgl = compute_tiled_orbitals_value_grad_lap(sd.orb_data_up_spin, up)
    down_vgl = compute_tiled_orbitals_value_grad_lap(sd.orb_data_dn_spin, down)
    up_matrix, down_matrix = np.asarray(up_vgl[0]), np.asarray(down_vgl[0])
    expected_det = np.linalg.det(up_matrix) * np.linalg.det(down_matrix)
    expected_log = np.linalg.slogdet(up_matrix)[1] + np.linalg.slogdet(down_matrix)[1]
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap", "det_eval", "det_grad_lap"), "medium")
    actual_det = compute_det_geminal_all_elements(sd, up, down)
    np.testing.assert_allclose(actual_det, expected_det, atol=atol, rtol=rtol)
    np.testing.assert_allclose(compute_ln_det_geminal_all_elements(sd, up, down), expected_log, atol=atol, rtol=rtol)
    np.testing.assert_allclose(
        compute_geminal_all_elements(sd, up, down), _compute_geminal_all_elements_debug(sd, up, down), atol=atol, rtol=rtol
    )
    gradient_up, gradient_down, lap_up, lap_down = compute_grads_and_laplacian_ln_Det(sd, up, down)
    kinetic = -0.5 * (
        np.sum(gradient_up * gradient_up) + np.sum(gradient_down * gradient_down) + np.sum(lap_up) + np.sum(lap_down)
    )
    reference_kinetic = -0.5 * (
        np.trace(np.linalg.solve(up_matrix, np.asarray(up_vgl[4])))
        + np.trace(np.linalg.solve(down_matrix, np.asarray(down_vgl[4])))
    )
    np.testing.assert_allclose(kinetic, reference_kinetic, atol=atol, rtol=rtol)
    for spin, sign in (("up", 1), ("down", -1)):
        moved_up, moved_down = up.copy(), down.copy()
        (moved_up if spin == "up" else moved_down)[0] += up_source.structure_data.cell[0]
        ratio = compute_det_geminal_all_elements(sd, moved_up, moved_down) / actual_det
        np.testing.assert_allclose(ratio, np.exp(sign * 2j * np.pi * up_source.supertwist[0]), atol=atol, rtol=rtol)


@pytest.mark.light
def test_tiled_sd_rejects_orbital_optimization_and_dense_conversion():
    """Both optimizer entry points reject frozen orbitals before constructing a metric."""
    from jqmc.determinant import SD_data
    from jqmc.wavefunction import Wavefunction_data

    sd = SD_data.build_tiled(up_source=_pw_set())
    wf = Wavefunction_data(geminal_data=sd, jastrow_data=None)
    for flags in (dict(opt_geminal=True), dict(opt_lambda_basis_exp=True), dict(opt_lambda_basis_coeff=True)):
        flags = dict(opt_geminal=False) | flags
        with pytest.raises(NotImplementedError, match="frozen"):
            wf.get_variational_blocks(state_token=("tiling", 0), **flags)
        with pytest.raises(NotImplementedError, match="frozen"):
            wf.with_param_grad_mask(**flags)
    assert wf.get_variational_blocks(state_token=("tiling", 0), opt_geminal=False) == []
    wf.with_param_grad_mask(opt_geminal=False)
    for convert in (sd.to_AGP, sd.to_AGPn, sd.natural_orbital_density_coefficients_jnp):
        with pytest.raises(NotImplementedError):
            convert()
    # Physical-coordinate derivatives are supported independently of frozen
    # variational parameters. Their numerical contracts live in the geometry,
    # full-energy derivative and real-source route comparison suites.


@pytest.mark.heavy
def test_tiled_composite_rejects_incompatible_children():
    """A k-local composite cannot mix phase representatives, cells or unsupported families."""
    orbitals = _backend_set("composite_sphe")
    block = orbitals.blocks[0]
    pgto, pw = block.backend.children
    for children, message in (
        ((pgto, pw.replace(k_point=tuple(np.asarray(pw.k_point) + [1, 0, 0]))), "same primitive k"),
        ((pgto, pw.replace(structure_data=pw.structure_data.replace(vec_a=(4.0, 0.0, 0.0)))), "structure"),
        ((pw, pw), "one complex pGTO"),
        ((pgto, pgto), "one complex pGTO"),
        ((pgto, pw, pw), "one complex pGTO"),
    ):
        composite = block.backend.replace(children=children)
        # Keep the local matrix shape valid so the intended child invariant
        # is the one that rejects this malformed provider.
        C = np.ones((block.num_orbitals, composite.num_ao), dtype=complex)
        with pytest.raises(ValueError, match=message):
            block.replace(backend=composite, coefficients=C).sanity_check()
