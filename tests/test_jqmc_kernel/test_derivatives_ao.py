"""Component tests for AO-type derivative kernels in ``jqmc.derivatives.ao``.

Each kernel is checked against ``jax.jvp`` of the production forward and against
finite difference, on small in-test AO leaves -- independent of the
``assemble.py`` wiring.
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

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block
from jqmc.atomic_orbital import AOs_bspline_data, AOs_plane_wave_data, AOs_sphe_gto_data
from jqmc.derivatives.ao import (
    bspline_value_grad_lap_dcell,
    pw_value_grad_lap_dcell,
    sphe_basis_param_jvp,
)
from jqmc.pw_pgto_projection import bspline_prefilter_periodic
from jqmc.structure import Structure_data


def _assert_orbital_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Require one homogeneous AO derivative block to meet its central budget."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


def _cubic_structure(a: float = 6.0) -> Structure_data:
    cell = np.eye(3, dtype=np.float64) * a
    positions = np.array([[0.0, 0.0, 0.0], [0.5 * a, 0.0, 0.0]], dtype=np.float64)
    return Structure_data.build(
        pbc_flag=True,
        positions=positions,
        vec_a=tuple(cell[0]),
        vec_b=tuple(cell[1]),
        vec_c=tuple(cell[2]),
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H1", "H2"),
    )


def _triclinic_structure(a: float = 6.0) -> Structure_data:
    """Non-orthorhombic (triclinic) cell -- off-diagonal lattice entries != 0."""
    cell = np.array([[a, 0.0, 0.0], [0.30 * a, a, 0.0], [0.20 * a, 0.15 * a, a]], dtype=np.float64)
    positions = np.array([[0.0, 0.0, 0.0], [0.4 * a, 0.1 * a, 0.0]], dtype=np.float64)
    return Structure_data.build(
        pbc_flag=True,
        positions=positions,
        vec_a=tuple(cell[0]),
        vec_b=tuple(cell[1]),
        vec_c=tuple(cell[2]),
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H1", "H2"),
    )


def _draw_in_cell_general(structure: Structure_data, n_e: int, seed: int) -> jax.Array:
    """Electrons strictly inside a general cell (frac in (0,1) so wrap is identity)."""
    rng = np.random.default_rng(seed)
    frac = 0.05 + 0.4 * rng.random((n_e, 3))
    cell = np.stack([structure.vec_a, structure.vec_b, structure.vec_c], axis=0)
    return jnp.asarray(frac @ cell)


def _pw_leaf(structure: Structure_data) -> AOs_plane_wave_data:
    recip = structure.recip_cell
    millers = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float64)
    g_list, is_sin = [], []
    for m in millers:
        g = m @ recip
        g_list.append(g)
        is_sin.append(0.0)
        if not np.all(m == 0):
            g_list.append(g)
            is_sin.append(1.0)
    return AOs_plane_wave_data.build(structure, np.asarray(g_list, dtype=np.float64), np.asarray(is_sin, dtype=np.float64))


def _draw_in_cell(a: float, n_e: int, seed: int) -> jax.Array:
    """Electrons strictly inside the home cell so wrap-into-cell is the identity."""
    rng = np.random.default_rng(seed)
    frac = 0.05 + 0.4 * rng.random((n_e, 3))
    return jnp.asarray(frac * a)


def _forward_vgl_of_cell(aos, r_carts):
    """f(vec_a, vec_b, vec_c) -> (val, gx, gy, gz, lap) for jvp/FD over the cell."""

    def f(va, vb, vc):
        sd = aos.structure_data.replace(vec_a=va, vec_b=vb, vec_c=vc)
        return aos.replace(structure_data=sd)._compute_AOs_value_grad_lap(r_carts)

    return f


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 7])
def test_pw_dcell_matches_jvp(seed):
    """Plane-wave AO: directional cell derivative of (value, grad, lap) == jax.jvp."""

    def case_factory():
        a = 6.0
        structure = _cubic_structure(a)
        aos = _pw_leaf(structure)
        r_carts = _draw_in_cell(a, 3, seed)

        rng = np.random.default_rng(seed + 100)
        dcell = jnp.asarray(rng.normal(scale=0.1, size=(3, 3)))

        return structure, aos, r_carts, dcell

    def analytical(case):
        structure, aos, r_carts, dcell = case
        return pw_value_grad_lap_dcell(aos, r_carts, dcell)

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

    structure, aos, r_carts, dcell = case_factory()
    analytic = pw_value_grad_lap_dcell(aos, r_carts, dcell)

    f = _forward_vgl_of_cell(aos, r_carts)
    va = jnp.asarray(structure.vec_a, dtype=jnp.float64)
    vb = jnp.asarray(structure.vec_b, dtype=jnp.float64)
    vc = jnp.asarray(structure.vec_c, dtype=jnp.float64)
    _, jvp_tangents = jax.jvp(f, (va, vb, vc), (dcell[0], dcell[1], dcell[2]))

    atol, rtol = get_tolerance_min(("ao_eval",), "strict")
    for name, da, dj in zip(("val", "gx", "gy", "gz", "lap"), analytic, jvp_tangents):
        np.testing.assert_allclose(np.asarray(da), np.asarray(dj), atol=atol, rtol=rtol, err_msg=name)
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.skip_mixed_precision(
    reason="FD comparison dominated by float32 round-off",
    alternative_guarantee_ref="MATH-AO-DERIVATIVE-CONSISTENCY-001",
)
def test_pw_dcell_matches_fd():
    """Plane-wave AO: directional cell derivative of the value matches central FD."""
    a = 6.0
    structure = _cubic_structure(a)
    aos = _pw_leaf(structure)
    r_carts = _draw_in_cell(a, 3, seed=3)

    rng = np.random.default_rng(42)
    dcell = jnp.asarray(rng.normal(scale=0.1, size=(3, 3)))
    dval_analytic = pw_value_grad_lap_dcell(aos, r_carts, dcell)[0]

    f = _forward_vgl_of_cell(aos, r_carts)
    va = jnp.asarray(structure.vec_a, dtype=jnp.float64)
    vb = jnp.asarray(structure.vec_b, dtype=jnp.float64)
    vc = jnp.asarray(structure.vec_c, dtype=jnp.float64)
    h = 1e-5
    vp = f(va + h * dcell[0], vb + h * dcell[1], vc + h * dcell[2])[0]
    vm = f(va - h * dcell[0], vb - h * dcell[1], vc - h * dcell[2])[0]
    dval_fd = (np.asarray(vp) - np.asarray(vm)) / (2 * h)

    atol, rtol = get_tolerance_min(("ao_eval",), "medium")
    np.testing.assert_allclose(np.asarray(dval_analytic), dval_fd, atol=atol, rtol=rtol)
    jax.clear_caches()


# ---------------------------------------------------------------------------
# B-spline AO cell derivative
# ---------------------------------------------------------------------------


def _bspline_bands(frac):
    x, y, z = frac[..., 0], frac[..., 1], frac[..., 2]
    tp = 2.0 * np.pi
    b0 = np.sin(tp * x) + 0.5 * np.cos(tp * y) + 0.25 * np.sin(tp * z)
    b1 = np.cos(tp * (x + y)) + 0.3 * np.sin(tp * z)
    return np.stack([b0, b1], axis=0)


def _bspline_leaf(structure: Structure_data, order: int = 5, ns=(10, 10, 10)) -> AOs_bspline_data:
    grids = [np.arange(n) / n for n in ns]
    fx, fy, fz = np.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
    frac = np.stack([fx, fy, fz], axis=-1)
    vals = _bspline_bands(frac)
    coeffs_cube = bspline_prefilter_periodic(vals.astype(np.complex128), order)
    coeffs_flat = coeffs_cube.reshape(coeffs_cube.shape[0], -1)
    return AOs_bspline_data.build(structure, coeffs_flat, ns, order)


@pytest.mark.light
@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("seed", [0, 7])
def test_bspline_dcell_matches_jvp(order, seed):
    """B-spline AO: directional cell derivative of (value, grad, lap) == jax.jvp."""

    def case_factory():
        a = 6.0
        structure = _cubic_structure(a)
        aos = _bspline_leaf(structure, order=order)
        r_carts = _draw_in_cell(a, 3, seed)

        rng = np.random.default_rng(seed + 100)
        dcell = jnp.asarray(rng.normal(scale=0.1, size=(3, 3)))

        return structure, aos, r_carts, dcell

    def analytical(case):
        structure, aos, r_carts, dcell = case
        return bspline_value_grad_lap_dcell(aos, r_carts, dcell)

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

    structure, aos, r_carts, dcell = case_factory()
    analytic = bspline_value_grad_lap_dcell(aos, r_carts, dcell)

    f = _forward_vgl_of_cell(aos, r_carts)
    va = jnp.asarray(structure.vec_a, dtype=jnp.float64)
    vb = jnp.asarray(structure.vec_b, dtype=jnp.float64)
    vc = jnp.asarray(structure.vec_c, dtype=jnp.float64)
    _, jvp_tangents = jax.jvp(f, (va, vb, vc), (dcell[0], dcell[1], dcell[2]))

    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")
    for name, da, dj in zip(("val", "gx", "gy", "gz", "lap"), analytic, jvp_tangents):
        np.testing.assert_allclose(np.asarray(da), np.asarray(dj), atol=atol, rtol=rtol, err_msg=name)
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.skip_mixed_precision(
    reason="FD comparison dominated by float32 round-off",
    alternative_guarantee_ref="MATH-AO-DERIVATIVE-CONSISTENCY-001",
)
def test_bspline_dcell_lap_matches_fd():
    """B-spline AO: directional cell derivative of the Laplacian matches central FD (needs d3w)."""
    a = 6.0
    structure = _cubic_structure(a)
    aos = _bspline_leaf(structure, order=5)
    r_carts = _draw_in_cell(a, 3, seed=3)

    rng = np.random.default_rng(42)
    dcell = jnp.asarray(rng.normal(scale=0.1, size=(3, 3)))
    dlap_analytic = bspline_value_grad_lap_dcell(aos, r_carts, dcell)[4]

    f = _forward_vgl_of_cell(aos, r_carts)
    va = jnp.asarray(structure.vec_a, dtype=jnp.float64)
    vb = jnp.asarray(structure.vec_b, dtype=jnp.float64)
    vc = jnp.asarray(structure.vec_c, dtype=jnp.float64)
    h = 1e-5
    lp = f(va + h * dcell[0], vb + h * dcell[1], vc + h * dcell[2])[4]
    lm = f(va - h * dcell[0], vb - h * dcell[1], vc - h * dcell[2])[4]
    dlap_fd = (np.asarray(lp) - np.asarray(lm)) / (2 * h)

    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap"), "medium")
    np.testing.assert_allclose(np.asarray(dlap_analytic), dlap_fd, atol=atol, rtol=rtol)
    jax.clear_caches()


@pytest.mark.light
def test_bspline_no_position_dependence():
    """B-spline AO does not depend on nuclear positions (dAO/dR = 0)."""
    a = 6.0
    structure = _cubic_structure(a)
    aos = _bspline_leaf(structure, order=5)
    r_carts = _draw_in_cell(a, 3, seed=1)

    def sum_val(positions):
        sd = aos.structure_data.replace(positions=positions)
        return jnp.sum(aos.replace(structure_data=sd)._compute_AOs(r_carts))

    pos = jnp.asarray(structure.positions, dtype=jnp.float64)
    dR = jax.grad(sum_val)(pos)
    np.testing.assert_array_equal(np.asarray(dR), np.zeros_like(np.asarray(pos)))
    jax.clear_caches()


# ---------------------------------------------------------------------------
# pgto / sphe radial basis-parameter derivative
# ---------------------------------------------------------------------------


def _sphe_leaf() -> AOs_sphe_gto_data:
    """Two atoms, contracted s + p shells (shared radial prims -> dedup)."""
    structure = Structure_data.build(
        pbc_flag=False,
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=np.float64),
        atomic_numbers=(3, 1),
        element_symbols=("Li", "H"),
        atomic_labels=("Li", "H"),
    )
    # AO prims: Li s (2-prim contraction), Li p_z (1 prim), H s (1 prim).
    return AOs_sphe_gto_data._from_full_basis(
        structure_data=structure,
        nucleus_index=(0, 0, 1),
        num_ao=3,
        num_ao_prim=4,
        orbital_indices=(0, 0, 1, 2),
        exponents=np.array([1.5, 0.45, 0.8, 0.9], dtype=np.float64),
        coefficients=np.array([0.6, 0.4, 1.0, 1.0], dtype=np.float64),
        angular_momentums=(0, 1, 0),
        magnetic_quantum_numbers=(0, 0, 0),
    )


def _forward_vgl_of_radial(aos, r_carts):
    def f(exps, coeffs):
        radial = aos.radial.replace(exponents=exps, coefficients=coeffs)
        return aos.replace(radial=radial)._compute_AOs_value_grad_lap(r_carts)

    return f


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 7])
def test_sphe_basis_param_matches_jvp(seed):
    """Spherical GTO: directional (exp, coeff) derivative of (value, grad, lap) == jax.jvp."""

    def case_factory():
        aos = _sphe_leaf()
        rng = np.random.default_rng(seed)
        r_carts = jnp.asarray(rng.normal(scale=0.8, size=(3, 3)))

        exps = jnp.asarray(np.asarray(aos.radial.exponents))
        coeffs = jnp.asarray(np.asarray(aos.radial.coefficients))
        d_exp = jnp.asarray(rng.normal(scale=0.1, size=exps.shape))
        d_coeff = jnp.asarray(rng.normal(scale=0.1, size=coeffs.shape))

        return aos, r_carts, exps, coeffs, d_exp, d_coeff

    def analytical(case):
        aos, r_carts, exps, coeffs, d_exp, d_coeff = case
        return sphe_basis_param_jvp(aos, r_carts, d_exp, d_coeff)

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

    aos, r_carts, exps, coeffs, d_exp, d_coeff = case_factory()
    analytic = sphe_basis_param_jvp(aos, r_carts, d_exp, d_coeff)

    f = _forward_vgl_of_radial(aos, r_carts)
    _, jvp_tangents = jax.jvp(f, (exps, coeffs), (d_exp, d_coeff))

    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")
    for name, da, dj in zip(("val", "gx", "gy", "gz", "lap"), analytic, jvp_tangents):
        np.testing.assert_allclose(np.asarray(da), np.asarray(dj), atol=atol, rtol=rtol, err_msg=name)
    jax.clear_caches()


@pytest.mark.light
def test_sphe_basis_param_jvp_at_zero_coefficient():
    """The value, gradient and Laplacian keep their coefficient tangent at zero."""
    aos = _sphe_leaf()
    r_carts = jnp.asarray([[0.2, 0.3, 0.4], [0.8, -0.1, 0.3]])
    exps = jnp.asarray(aos.radial.exponents)
    coeffs = jnp.asarray(aos.radial.coefficients).at[0].set(0)
    aos = aos.replace(radial=aos.radial.replace(coefficients=coeffs))
    d_exp = jnp.zeros_like(exps)
    d_coeff = jnp.zeros_like(coeffs).at[0].set(1)

    actual = sphe_basis_param_jvp(aos, r_carts, d_exp, d_coeff)
    forward = _forward_vgl_of_radial(aos, r_carts)
    _, reference = jax.jvp(forward, (exps, coeffs), (d_exp, d_coeff))
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")
    assert all(np.max(np.abs(component)) > atol for component in reference)
    np.testing.assert_allclose(np.stack(actual), np.stack(reference), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.skip_mixed_precision(
    reason="FD comparison dominated by float32 round-off",
    alternative_guarantee_ref="MATH-AO-DERIVATIVE-CONSISTENCY-001",
)
def test_sphe_basis_param_matches_fd():
    """Spherical GTO: directional (exp, coeff) derivative of value+lap matches central FD."""
    aos = _sphe_leaf()
    rng = np.random.default_rng(5)
    r_carts = jnp.asarray(rng.normal(scale=0.8, size=(3, 3)))
    exps = jnp.asarray(np.asarray(aos.radial.exponents))
    coeffs = jnp.asarray(np.asarray(aos.radial.coefficients))
    d_exp = jnp.asarray(rng.normal(scale=0.1, size=exps.shape))
    d_coeff = jnp.asarray(rng.normal(scale=0.1, size=coeffs.shape))

    dval_a, _, _, _, dlap_a = sphe_basis_param_jvp(aos, r_carts, d_exp, d_coeff)

    f = _forward_vgl_of_radial(aos, r_carts)
    h = 1e-6
    vp = f(exps + h * d_exp, coeffs + h * d_coeff)
    vm = f(exps - h * d_exp, coeffs - h * d_coeff)
    dval_fd = (np.asarray(vp[0]) - np.asarray(vm[0])) / (2 * h)
    dlap_fd = (np.asarray(vp[4]) - np.asarray(vm[4])) / (2 * h)

    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap"), "medium")
    np.testing.assert_allclose(np.asarray(dval_a), dval_fd, atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(dlap_a), dlap_fd, atol=atol, rtol=rtol)
    jax.clear_caches()


@pytest.mark.light
def test_pw_no_position_dependence():
    """Plane-wave AO does not depend on nuclear positions (dAO/dR = 0)."""
    a = 6.0
    structure = _cubic_structure(a)
    aos = _pw_leaf(structure)
    r_carts = _draw_in_cell(a, 3, seed=1)

    def sum_val(positions):
        sd = aos.structure_data.replace(positions=positions)
        return jnp.sum(aos.replace(structure_data=sd)._compute_AOs(r_carts))

    pos = jnp.asarray(structure.positions, dtype=jnp.float64)
    dR = jax.grad(sum_val)(pos)
    np.testing.assert_array_equal(np.asarray(dR), np.zeros_like(np.asarray(pos)))
    jax.clear_caches()


# ---------------------------------------------------------------------------
# non-orthorhombic (triclinic) cell verification
# ---------------------------------------------------------------------------
# The AO strain kernels are cell-general (they use inv(cell)); these confirm the
# directional cell derivatives stay correct when off-diagonal lattice entries
# are non-zero. The analytic-vs-auto parity of the assembled stress is checked
# at assembly; here we verify the plane-wave and B-spline AO-leaf cell derivatives.


@pytest.mark.light
def test_pw_dcell_matches_jvp_nonortho():
    """Pure-PW directional cell derivative == jax.jvp on a triclinic cell."""

    def case_factory():
        structure = _triclinic_structure(6.0)
        aos = _pw_leaf(structure)
        r_carts = _draw_in_cell_general(structure, 3, seed=2)

        rng = np.random.default_rng(202)
        dcell = jnp.asarray(rng.normal(scale=0.1, size=(3, 3)))
        return structure, aos, r_carts, dcell

    def analytical(case):
        structure, aos, r_carts, dcell = case
        return pw_value_grad_lap_dcell(aos, r_carts, dcell)

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

    structure, aos, r_carts, dcell = case_factory()
    analytic = pw_value_grad_lap_dcell(aos, r_carts, dcell)

    f = _forward_vgl_of_cell(aos, r_carts)
    va = jnp.asarray(structure.vec_a, dtype=jnp.float64)
    vb = jnp.asarray(structure.vec_b, dtype=jnp.float64)
    vc = jnp.asarray(structure.vec_c, dtype=jnp.float64)
    _, jvp_tangents = jax.jvp(f, (va, vb, vc), (dcell[0], dcell[1], dcell[2]))

    atol, rtol = get_tolerance_min(("ao_eval",), "strict")
    for name, da, dj in zip(("val", "gx", "gy", "gz", "lap"), analytic, jvp_tangents):
        np.testing.assert_allclose(np.asarray(da), np.asarray(dj), atol=atol, rtol=rtol, err_msg=name)
    jax.clear_caches()


@pytest.mark.light
def test_bspline_dcell_matches_jvp_nonortho():
    """B-spline directional cell derivative == jax.jvp on a triclinic cell."""

    def case_factory():
        structure = _triclinic_structure(6.0)
        aos = _bspline_leaf(structure, order=5)
        r_carts = _draw_in_cell_general(structure, 3, seed=4)

        rng = np.random.default_rng(303)
        dcell = jnp.asarray(rng.normal(scale=0.1, size=(3, 3)))
        return structure, aos, r_carts, dcell

    def analytical(case):
        structure, aos, r_carts, dcell = case
        return bspline_value_grad_lap_dcell(aos, r_carts, dcell)

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

    structure, aos, r_carts, dcell = case_factory()
    analytic = bspline_value_grad_lap_dcell(aos, r_carts, dcell)

    f = _forward_vgl_of_cell(aos, r_carts)
    va = jnp.asarray(structure.vec_a, dtype=jnp.float64)
    vb = jnp.asarray(structure.vec_b, dtype=jnp.float64)
    vc = jnp.asarray(structure.vec_c, dtype=jnp.float64)
    _, jvp_tangents = jax.jvp(f, (va, vb, vc), (dcell[0], dcell[1], dcell[2]))

    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")
    for name, da, dj in zip(("val", "gx", "gy", "gz", "lap"), analytic, jvp_tangents):
        np.testing.assert_allclose(np.asarray(da), np.asarray(dj), atol=atol, rtol=rtol, err_msg=name)
    jax.clear_caches()
