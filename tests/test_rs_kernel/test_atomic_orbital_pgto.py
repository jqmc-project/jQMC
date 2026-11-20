"""Layer-1 equivalence test: jqmc_rs vs jqmc for *periodic* pGTO.

Image expansion + fold (Gamma / real path; the ice_pgto
benchmark system is `ice_ecp_gamma`, i.e. k=0). The kernel evaluates the
molecular GTO at the image-tiled rank `n_image * num_ao` (centers R+T_n) and
folds the image axis back to `num_ao`. Complex Bloch (k != 0) is a follow-up.

This file synthesizes its own periodic sphe/cart pGTO basis from scratch (no
Hamiltonian fixture needed), so no bench/ h5 load and no committed-fixture
builder are required here.
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

import numpy as np
import pytest

import jqmc_rs

# Every rs test declares its cost tier (light/heavy, per test) and CWD policy
# (with_chdir here -- autouse _isolate_cwd chdirs into a per-test tmp_path for
# xdist safety; this module builds its own synthetic bases so cwd is irrelevant).
pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL, ATOL_RELATIVE_COMPARISON_SCALE_FLOOR
from jqmc.atomic_orbital import (
    AOs_cart_pgto_data,
    AOs_sphe_gto_data,
    _apply_image_expansion,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
)
from jqmc.structure import Structure_data

# Backend parity uses the exact, unscreened Rust bindings. XLA fixes the image set
# at build time and evaluates every retained image at every electron position;
# the parity oracle must reproduce that convention. The screened bindings belong
# to the production MCMC sweep and are not an XLA-equivalence surface.
ATOL, RTOL = get_tolerance_xla_rust_consistency(("ao_eval",))


def _close(got, ref):
    d = np.abs(np.asarray(got) - np.asarray(ref)).max()
    return d <= ATOL or d <= RTOL * (np.abs(np.asarray(ref)).max() + ATOL_RELATIVE_COMPARISON_SCALE_FLOOR)


_L = 10.0  # cubic cell edge (Bohr)


def _pbc_structure(R_carts):
    n = len(R_carts)
    return Structure_data.build(
        pbc_flag=True,
        positions=np.asarray(R_carts, dtype=np.float64),
        vec_a=(_L, 0.0, 0.0),
        vec_b=(0.0, _L, 0.0),
        vec_c=(0.0, 0.0, _L),
        atomic_numbers=tuple([1] * n),
        element_symbols=tuple(["H"] * n),
        atomic_labels=tuple(["H"] * n),
    )


def _make_pgto(rng, lmax=2, n_atom=2):
    """Build a molecular sphe-GTO basis inside a cubic cell, then image-expand."""
    R = (2.0 + 6.0 * rng.random((n_atom, 3))).astype(np.float64)  # inside [0,10]
    structure = _pbc_structure(R)
    nucleus_index, orbital_indices, exponents, coefficients = [], [], [], []
    angular_momentums, magnetic = [], []
    ao_idx = 0
    for a in range(n_atom):
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                nucleus_index.append(a)
                angular_momentums.append(l)
                magnetic.append(m)
                for _ in range(2):
                    orbital_indices.append(ao_idx)
                    # moderately tight so the truncated image shell is small
                    exponents.append(0.8 + rng.random())
                    coefficients.append(0.3 + rng.random())
                ao_idx += 1
    base = AOs_sphe_gto_data._from_full_basis(
        exponents=np.array(exponents, dtype=np.float64),
        coefficients=np.array(coefficients, dtype=np.float64),
        structure_data=structure,
        nucleus_index=tuple(nucleus_index),
        num_ao=ao_idx,
        num_ao_prim=len(exponents),
        orbital_indices=tuple(orbital_indices),
        angular_momentums=tuple(angular_momentums),
        magnetic_quantum_numbers=tuple(magnetic),
    )
    pgto = _apply_image_expansion(base, structure, tol=AO_IMAGE_TRUNCATION_ATOL)
    return pgto


def _make_pgto_cart(rng, lmax=2, n_atom=2):
    """Build a molecular cart-GTO basis inside a cubic cell, then image-expand."""
    R = (2.0 + 6.0 * rng.random((n_atom, 3))).astype(np.float64)  # inside [0,10]
    structure = _pbc_structure(R)
    nucleus_index, orbital_indices, exponents, coefficients = [], [], [], []
    angular_momentums, poly_x, poly_y, poly_z = [], [], [], []
    ao_idx = 0
    for a in range(n_atom):
        for l in range(lmax + 1):
            for nx in range(l, -1, -1):
                for ny in range(l - nx, -1, -1):
                    nz = l - nx - ny
                    nucleus_index.append(a)
                    angular_momentums.append(l)
                    poly_x.append(nx)
                    poly_y.append(ny)
                    poly_z.append(nz)
                    for _ in range(2):
                        orbital_indices.append(ao_idx)
                        # moderately tight so the truncated image shell is small
                        exponents.append(0.8 + rng.random())
                        coefficients.append(0.3 + rng.random())
                    ao_idx += 1
    pgto = AOs_cart_pgto_data.build(
        structure_data=structure,
        nucleus_index=tuple(nucleus_index),
        num_ao=ao_idx,
        num_ao_prim=len(exponents),
        orbital_indices=tuple(orbital_indices),
        exponents=np.array(exponents, dtype=np.float64),
        coefficients=np.array(coefficients, dtype=np.float64),
        angular_momentums=tuple(angular_momentums),
        polynominal_order_x=tuple(poly_x),
        polynominal_order_y=tuple(poly_y),
        polynominal_order_z=tuple(poly_z),
        tol=AO_IMAGE_TRUNCATION_ATOL,
    )
    return pgto


def _electrons_in_cell(rng, n):
    # inside the cell so jqmc's wrap-into-cell is a no-op (periodic sum invariant)
    return (_L * rng.random((n, 3))).astype(np.float64)


@pytest.mark.light
@pytest.mark.parametrize("kind", ["sphe", "cart"])
@pytest.mark.parametrize("seed", [0, 5, 13])
@pytest.mark.parametrize("lmax", [1, 2])
def test_pgto_value_matches_jax(seed, lmax, kind):
    """Pgto value matches jax."""
    rng = np.random.default_rng(seed)
    make = _make_pgto if kind == "sphe" else _make_pgto_cart
    pgto = make(rng, lmax=lmax)
    # sanity: this really is a periodic (image-expanded) leaf
    assert pgto._n_image >= 1
    r_elec = _electrons_in_cell(rng, 6)
    ref = np.asarray(compute_AOs(pgto, r_elec), dtype=np.float64)
    got = jqmc_rs.compute_AOs_exact(pgto, r_elec)
    assert got.shape == (pgto.num_ao, 6), (got.shape, pgto.num_ao)
    assert _close(got, ref), np.abs(got - ref).max()


@pytest.mark.light
@pytest.mark.parametrize("kind", ["sphe", "cart"])
@pytest.mark.parametrize("seed", [0, 5, 13])
def test_pgto_grad_matches_jax(seed, kind):
    """Pgto grad matches jax."""
    rng = np.random.default_rng(seed)
    make = _make_pgto if kind == "sphe" else _make_pgto_cart
    pgto = make(rng, lmax=2)
    r_elec = _electrons_in_cell(rng, 5)
    gx, gy, gz = (np.asarray(g, dtype=np.float64) for g in compute_AOs_grad(pgto, r_elec))
    rx, ry, rz = jqmc_rs.compute_AOs_grad_exact(pgto, r_elec)
    for got, ref in [(rx, gx), (ry, gy), (rz, gz)]:
        assert _close(got, ref), np.abs(got - ref).max()


@pytest.mark.light
@pytest.mark.parametrize("kind", ["sphe", "cart"])
@pytest.mark.parametrize("seed", [0, 5, 13])
def test_pgto_laplacian_matches_jax(seed, kind):
    """Pgto laplacian matches jax."""
    rng = np.random.default_rng(seed)
    make = _make_pgto if kind == "sphe" else _make_pgto_cart
    pgto = make(rng, lmax=2)
    r_elec = _electrons_in_cell(rng, 5)
    ref = np.asarray(compute_AOs_laplacian(pgto, r_elec), dtype=np.float64)
    got = jqmc_rs.compute_AOs_laplacian(pgto, r_elec)
    assert _close(got, ref), np.abs(got - ref).max()


@pytest.mark.light
def test_pgto_value_in_non_reduced_basis_matches_bruteforce_image_sum():
    """Rust AO values of an image-expanded leaf on a non-reduced cell basis are complete.

    The Rust kernel consumes the Python-built image layout. The lattice diag(6, 7, 8)
    Bohr is written in the basis ``U @ diag(6, 7, 8)`` (``det U = 1``, strongly
    ill-conditioned Gram matrix), with one hydrogen and the s/p/d/f test shells of
    ``tests/test_jqmc_kernel/_noncubic_oracles.py``. ``jqmc_rs.compute_AOs_exact`` on
    the Gamma leaf must equal the brute-force lattice sum of the molecular AO kernel
    over the reduced cell (no production image list), within the ao_eval strict zone
    plus AO_IMAGE_TRUNCATION_ATOL.
    """
    from jqmc._precision import get_tolerance
    from tests.test_jqmc_kernel import _noncubic_oracles as ref
    from tests.test_jqmc_kernel.test_noncubic_cell_oracles import (
        NON_REDUCED_AO_ATOM,
        NON_REDUCED_AO_LATTICE,
        non_reduced_ao_electrons,
        non_reduced_ao_leaf,
    )

    r = non_reduced_ao_electrons()
    got = np.asarray(jqmc_rs.compute_AOs_exact(non_reduced_ao_leaf(), r))
    brute, n_images = ref.bruteforce_periodic_aos(NON_REDUCED_AO_LATTICE, NON_REDUCED_AO_ATOM, "cart", r, None)
    assert np.all(np.isfinite(got)) and np.all(np.isfinite(brute))
    assert n_images > 27, f"brute-force sum is not a lattice sum ({n_images} images)"
    ao_atol, ao_rtol = get_tolerance("ao_eval", "strict")
    np.testing.assert_allclose(got, brute, atol=AO_IMAGE_TRUNCATION_ATOL + ao_atol, rtol=ao_rtol)
