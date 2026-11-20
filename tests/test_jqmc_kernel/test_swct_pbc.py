"""Periodic-boundary guarantees of the SWCT weights.

The space-warp weights ``omega_{a i}`` decide which nucleus an electron follows
when a nucleus is displaced. In a periodic crystal a nuclear displacement moves
every image of that nucleus, so the warp must be a function on the torus:
``omega_{a i}`` and the SWCT force correction
``sum_i omega_{a i} f_i + 1/2 sum_i grad_{r_i} omega_{a i}`` must not change when
an electron (walker positions are kept UNWRAPPED by the drivers) or a nucleus is
moved by a lattice vector. The reference below is an independent brute-force
minimum-image implementation written in NumPy: it enumerates every lattice
translation in a fixed box and keeps the shortest ``r - R - T``; it does not use
the production parallelepiped round-then-search helper.
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

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_dtype_jnp, get_tolerance
from jqmc.structure import Structure_data
from jqmc.swct import (
    _evaluate_swct_domega_debug,
    _evaluate_swct_omega_debug,
    evaluate_swct_domega,
    evaluate_swct_omega,
    evaluate_swct_omega_dot,
    evaluate_swct_omega_dot_batched,
)

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

# Cells: an orthorhombic one and an oblique one (on the oblique cell the
# parallelepiped-rounded image is not always the nearest image).
_CELLS = {
    "orthorhombic": ((8.1, 0.0, 0.0), (0.0, 9.3, 0.0), (0.0, 0.0, 7.7)),
    "oblique": ((9.3, 0.0, 0.0), (2.1, 9.3, 0.0), (1.4, 1.9, 9.3)),
}
# Nuclei deliberately close to the cell faces, so that the nearest image of a
# nucleus for many electrons lies in a neighbouring cell.
_NUCLEI_FRAC = np.array(
    [
        [0.02, 0.49, 0.51],
        [0.97, 0.08, 0.93],
        [0.51, 0.96, 0.03],
        [0.26, 0.24, 0.74],
    ]
)
# Electrons: some just inside the opposite faces, the rest seeded-random.
_ELECTRON_FRAC_EDGE = np.array(
    [
        [0.985, 0.47, 0.52],
        [0.03, 0.11, 0.06],
        [0.49, 0.02, 0.97],
    ]
)
# One distinct lattice translation per electron (integer multiples of a, b, c),
# including far ones as produced by long unwrapped walks.
_ELECTRON_SHIFTS = np.array(
    [
        [1, 0, 0],
        [-1, 2, 0],
        [0, 0, -3],
        [7, -5, 4],
        [2, 1, -1],
        [0, -1, 1],
        [-4, 3, 6],
        [1, 1, 1],
        [0, 3, -2],
    ]
)
_NUCLEUS_SHIFTS = np.array([[-1, 0, 0], [1, 0, 1], [0, 1, -1], [2, -3, 1]])
_REFERENCE_IMAGE_RANGE = 3  # brute-force box half-width; covers both cells' Wigner-Seitz images


def _cell(name: str) -> np.ndarray:
    return np.asarray(_CELLS[name], dtype=np.float64)


def _structure(cell: np.ndarray, positions: np.ndarray) -> Structure_data:
    n_atoms = positions.shape[0]
    return Structure_data.build(
        positions=positions,
        pbc_flag=True,
        vec_a=tuple(cell[0]),
        vec_b=tuple(cell[1]),
        vec_c=tuple(cell[2]),
        atomic_numbers=tuple([6] * n_atoms),
        element_symbols=tuple(["C"] * n_atoms),
        atomic_labels=tuple(["C"] * n_atoms),
    )


def _fixture(cell_name: str):
    cell = _cell(cell_name)
    rng = np.random.default_rng(41)
    frac_e = np.vstack([_ELECTRON_FRAC_EDGE, rng.random((_ELECTRON_SHIFTS.shape[0] - _ELECTRON_FRAC_EDGE.shape[0], 3))])
    positions = _NUCLEI_FRAC @ cell
    r_carts = frac_e @ cell
    field = rng.standard_normal(r_carts.shape)
    return cell, positions, r_carts, field


def _reference_min_image(r_carts: np.ndarray, positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Brute-force nearest-image ``r_i - R_a - T`` with shape ``(N_e, N_a, 3)``."""
    span = range(-_REFERENCE_IMAGE_RANGE, _REFERENCE_IMAGE_RANGE + 1)
    translations = np.array([np.array(n, dtype=np.float64) @ cell for n in itertools.product(span, span, span)])
    out = np.empty((r_carts.shape[0], positions.shape[0], 3))
    for i, r in enumerate(r_carts):
        for a, R in enumerate(positions):
            candidates = r - R - translations
            out[i, a] = candidates[np.argmin(np.einsum("tk,tk->t", candidates, candidates))]
    return out


def _reference_swct(r_carts, positions, cell, field):
    """Independent NumPy fp64 omega, domega and SWCT force correction on minimum images."""
    diff = _reference_min_image(r_carts, positions, cell)  # (N_e, N_a, 3)
    d2 = np.einsum("iak,iak->ia", diff, diff)
    kappa = d2**-2
    s = kappa.sum(axis=1)
    omega = (kappa / s[:, None]).T  # (N_a, N_e)
    grad_kappa = -4.0 * diff * (kappa / d2)[..., None]
    grad_s = grad_kappa.sum(axis=1)
    grad_omega = (grad_kappa * s[:, None, None] - kappa[..., None] * grad_s[:, None, :]) / (s**2)[:, None, None]
    domega = grad_omega.sum(axis=0)  # (N_a, 3)
    correction = omega @ field + 0.5 * domega
    return omega, domega, grad_omega, correction


def _production(structure_data, r_carts, field):
    omega = np.asarray(evaluate_swct_omega(structure_data, jnp.asarray(r_carts)))
    domega = np.asarray(evaluate_swct_domega(structure_data, jnp.asarray(r_carts)))
    dot = np.asarray(evaluate_swct_omega_dot(structure_data, jnp.asarray(r_carts), jnp.asarray(field)))
    return omega, domega, dot + 0.5 * domega


def _assert_finite(*arrays):
    for x in arrays:
        assert np.all(np.isfinite(np.asarray(x))), "non-finite SWCT value"


@pytest.mark.light
@pytest.mark.parametrize("cell_name", sorted(_CELLS))
def test_swct_matches_independent_minimum_image_reference(cell_name):
    """Production and debug omega / domega / omega-dot equal the brute-force minimum-image reference."""
    atol, rtol = get_tolerance("swct", "strict")
    cell, positions, r_carts, field = _fixture(cell_name)
    structure_data = _structure(cell, positions)
    ref_omega, ref_domega, _, ref_corr = _reference_swct(r_carts, positions, cell, field)

    omega, domega, corr = _production(structure_data, r_carts, field)
    omega_dbg = _evaluate_swct_omega_debug(structure_data, r_carts)
    domega_dbg = _evaluate_swct_domega_debug(structure_data, r_carts)
    _assert_finite(omega, domega, corr, omega_dbg, domega_dbg)

    np.testing.assert_allclose(omega, ref_omega, atol=atol, rtol=rtol)
    np.testing.assert_allclose(domega, ref_domega, atol=atol, rtol=rtol)
    np.testing.assert_allclose(corr, ref_corr, atol=atol, rtol=rtol)
    np.testing.assert_allclose(omega_dbg, ref_omega, atol=atol, rtol=rtol)
    np.testing.assert_allclose(domega_dbg, ref_domega, atol=atol, rtol=rtol)
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize("cell_name", sorted(_CELLS))
def test_swct_partition_of_unity(cell_name):
    """sum_a omega_{a i} = 1 for every electron, and sum_a grad_{r_i} omega_{a i} = 0."""
    atol, rtol = get_tolerance("swct", "strict")
    cell, positions, r_carts, _ = _fixture(cell_name)
    structure_data = _structure(cell, positions)
    shifted = r_carts + _ELECTRON_SHIFTS @ cell  # unwrapped walkers far from the home cell

    for r in (r_carts, shifted):
        omega = np.asarray(evaluate_swct_omega(structure_data, jnp.asarray(r)))
        domega = np.asarray(evaluate_swct_domega(structure_data, jnp.asarray(r)))
        _assert_finite(omega, domega)
        np.testing.assert_allclose(omega.sum(axis=0), np.ones(r.shape[0]), atol=atol, rtol=rtol)
        np.testing.assert_allclose(domega.sum(axis=0), np.zeros(3), atol=atol, rtol=rtol)
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize("cell_name", sorted(_CELLS))
def test_swct_electron_lattice_translation_invariance(cell_name):
    """omega, domega and the SWCT force correction are unchanged when electrons move by lattice vectors.

    The field ``f_i`` (a gradient of a periodic quantity) takes the same value at
    ``r_i`` and ``r_i + L``. The batched streamed route used by the drivers is
    checked on the pair (home, translated) as well.
    """
    atol, rtol = get_tolerance("swct", "strict")
    cell, positions, r_carts, field = _fixture(cell_name)
    structure_data = _structure(cell, positions)
    shifted = r_carts + _ELECTRON_SHIFTS @ cell

    omega0, domega0, corr0 = _production(structure_data, r_carts, field)
    omega1, domega1, corr1 = _production(structure_data, shifted, field)
    _assert_finite(omega0, domega0, corr0, omega1, domega1, corr1)
    np.testing.assert_allclose(omega1, omega0, atol=atol, rtol=rtol)
    np.testing.assert_allclose(domega1, domega0, atol=atol, rtol=rtol)
    np.testing.assert_allclose(corr1, corr0, atol=atol, rtol=rtol)

    dtype_jnp = get_dtype_jnp("swct")
    r_b = jnp.asarray(np.stack([r_carts, shifted]), dtype=dtype_jnp)
    f_b = jnp.asarray(np.stack([field, field]))
    dot_b = np.asarray(evaluate_swct_omega_dot_batched(structure_data, r_b, f_b))
    domega_b = np.asarray(jax.vmap(evaluate_swct_domega, in_axes=(None, 0))(structure_data, r_b))
    corr_b = dot_b + 0.5 * domega_b
    _assert_finite(corr_b)
    np.testing.assert_allclose(corr_b[1], corr_b[0], atol=atol, rtol=rtol)

    omega_dbg0 = _evaluate_swct_omega_debug(structure_data, r_carts)
    omega_dbg1 = _evaluate_swct_omega_debug(structure_data, shifted)
    domega_dbg0 = _evaluate_swct_domega_debug(structure_data, r_carts)
    domega_dbg1 = _evaluate_swct_domega_debug(structure_data, shifted)
    _assert_finite(omega_dbg0, omega_dbg1, domega_dbg0, domega_dbg1)
    np.testing.assert_allclose(omega_dbg1, omega_dbg0, atol=atol, rtol=rtol)
    np.testing.assert_allclose(domega_dbg1, domega_dbg0, atol=atol, rtol=rtol)
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize("cell_name", sorted(_CELLS))
def test_swct_nucleus_image_invariance(cell_name):
    """omega, domega and the SWCT force correction do not depend on which image of a nucleus is stored."""
    atol, rtol = get_tolerance("swct", "strict")
    cell, positions, r_carts, field = _fixture(cell_name)
    home = _structure(cell, positions)
    moved = _structure(cell, positions + _NUCLEUS_SHIFTS @ cell)

    omega0, domega0, corr0 = _production(home, r_carts, field)
    omega1, domega1, corr1 = _production(moved, r_carts, field)
    _assert_finite(omega0, domega0, corr0, omega1, domega1, corr1)
    np.testing.assert_allclose(omega1, omega0, atol=atol, rtol=rtol)
    np.testing.assert_allclose(domega1, domega0, atol=atol, rtol=rtol)
    np.testing.assert_allclose(corr1, corr0, atol=atol, rtol=rtol)

    omega_dbg = _evaluate_swct_omega_debug(moved, r_carts)
    domega_dbg = _evaluate_swct_domega_debug(moved, r_carts)
    _assert_finite(omega_dbg, domega_dbg)
    np.testing.assert_allclose(omega_dbg, omega0, atol=atol, rtol=rtol)
    np.testing.assert_allclose(domega_dbg, domega0, atol=atol, rtol=rtol)
    jax.clear_caches()


@jax.jit
def _raw_omega_dot(R_carts, r_carts, field):
    """The molecular formula on raw ``r - R`` (no image handling), same operation order as production."""
    dtype_jnp = get_dtype_jnp("swct")
    diff = (r_carts[:, None, :] - R_carts[None, :, :]).astype(dtype_jnp)
    d2 = jnp.sum(diff * diff, axis=-1)
    kappa = 1.0 / (d2 * d2)
    omega_t = kappa / jnp.sum(kappa, axis=1, keepdims=True)
    return jnp.sum(omega_t[:, :, None] * jnp.asarray(field, dtype=dtype_jnp)[:, None, :], axis=0)


@jax.jit
def _raw_domega(R_carts, r_carts):
    dtype_jnp = get_dtype_jnp("swct")
    diff = (r_carts[:, None, :] - R_carts[None, :, :]).astype(dtype_jnp)
    d2 = jnp.sum(diff * diff, axis=-1)
    kappa = 1.0 / (d2 * d2)
    s_sum = jnp.sum(kappa, axis=1)
    grad_kappa = -4.0 * diff * (kappa / d2)[..., None]
    grad_s = jnp.sum(grad_kappa, axis=1)
    grad_omega = (grad_kappa * s_sum[:, None, None] - kappa[..., None] * grad_s[:, None, :]) / (s_sum * s_sum)[:, None, None]
    return jnp.sum(grad_omega, axis=0)


@jax.jit
def _raw_omega(R_carts, r_carts):
    dtype_jnp = get_dtype_jnp("swct")

    def one(R_cart, r_cart):
        diff_one = (r_cart - R_cart).astype(dtype_jnp)
        diff_all = (r_cart - R_carts).astype(dtype_jnp)
        kappa = 1.0 / jnp.linalg.norm(diff_one) ** 4
        kappa_sum = jnp.sum(1.0 / jnp.linalg.norm(diff_all, axis=1) ** 4)
        return kappa / kappa_sum

    return jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None))(R_carts, r_carts)


@pytest.mark.light
def test_swct_molecular_values_are_the_raw_formula_bit_for_bit():
    """Molecules keep the raw ``r - R`` weights exactly: the periodic image handling is PBC-only."""
    rng = np.random.default_rng(5)
    n_atoms, n_e = 4, 9
    positions = rng.uniform(-2.0, 2.0, (n_atoms, 3))
    structure_data = Structure_data.build(
        positions=positions,
        pbc_flag=False,
        atomic_numbers=tuple([6] * n_atoms),
        element_symbols=tuple(["C"] * n_atoms),
        atomic_labels=tuple(["C"] * n_atoms),
    )
    r_carts = jnp.asarray(rng.uniform(-5.0, 5.0, (n_e, 3)))
    field = jnp.asarray(rng.standard_normal((n_e, 3)))
    R = jnp.asarray(positions)

    omega = np.asarray(evaluate_swct_omega(structure_data, r_carts))
    domega = np.asarray(evaluate_swct_domega(structure_data, r_carts))
    dot = np.asarray(evaluate_swct_omega_dot(structure_data, r_carts, field))
    _assert_finite(omega, domega, dot)
    np.testing.assert_array_equal(omega, np.asarray(_raw_omega(R, r_carts)))
    np.testing.assert_array_equal(domega, np.asarray(_raw_domega(R, r_carts)))
    np.testing.assert_array_equal(dot, np.asarray(_raw_omega_dot(R, r_carts, field)))
    jax.clear_caches()
