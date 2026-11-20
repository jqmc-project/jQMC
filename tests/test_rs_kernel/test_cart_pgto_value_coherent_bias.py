"""Coherent value-scale bias of the periodic Cartesian pGTO value routes (issue #718).

In mixed precision the standalone Cartesian AO value kernels (XLA ``compute_AOs`` through
``AOs_cart_gto_data._compute_AOs`` on a periodic leaf, and the Rust f32 value kernel behind
``jqmc_rs.compute_AOs`` / ``compute_AOs_exact``) evaluate the coordinate-dependent part
(r^2, exp) in float32. The electron-independent radial factor -- the primitive normalization N,
the contraction coefficient c and the exponent Z -- must stay at storage precision and enter the
value through ONE rounding of the primitive product N (c exp(-Z r^2)). Rounding N, c or Z to
float32 separately is carried coherently into every value of that primitive and every lattice
image (the spherical kernel was fixed the same way in issue #145).

Reference: an fp64 numpy lattice sum of textbook-normalized Cartesian s Gaussians that calls no
jqmc AO evaluation, normalization or image-enumeration code. For a Cartesian s function the
polynomial is 1 and the normalization from int |chi|^2 d^3r = 1 is (2 Z / pi)^(3/4), so

    chi(r) = sum_n sum_p c_p (2 Z_p / pi)^(3/4) exp(-Z_p |r - R_A - n . L|^2).

Only basis input data (exponents, coefficients, atom of each AO, positions, cell) and the leaf's
declared kept lattice-image set of each AO are read. The reference sums exactly that set; every
image of an independently enumerated box two layers wider that the leaf dropped is asserted to be
below the leaf's image truncation tolerance at every point, with the outermost layer entirely
dropped.

Statistic, for every s-type AO over the sample points i with weights w_i = v_ref,i^2:

    kappa = sum_i v_ref,i (v_i - v_ref,i) / sum_i v_ref,i^2.

kappa is a ratio estimator of the cell integral of v_ref (v - v_ref) over that of v_ref^2. Its
standard error is taken from the per-point terms (delta method),

    SE^2 = sum_i (v_ref,i e_i - kappa v_ref,i^2)^2 / (sum_i v_ref,i^2)^2,  e_i = v_i - v_ref,i,

which accounts for the concentration of the weights (effective sample size
N_eff = (sum w)^2 / sum w^2). The unweighted proxy relrms / sqrt(N) is not used: for an
all-electron 1s core it understates the standard error about tenfold (issue #718). The check
first asserts that the statistic is resolved (SE at most a fixed fraction of the bound and N_eff
above a floor), then that |kappa| is within the bound

    |kappa| <= AO_CART_PERIODIC_VALUE_COHERENT_BIAS_EPS_FRACTION * eps(ao_eval)
               + AO_VALUE_COHERENT_BIAS_FP64_EPS_MULTIPLE * eps(float64),

with SE <= AO_VALUE_COHERENT_BIAS_STANDARD_ERROR_FRACTION * bound and
N_eff >= AO_CART_PERIODIC_VALUE_COHERENT_BIAS_MIN_EFFECTIVE_POINTS. The constants and the
measurements justifying the 0.1 float32-eps bound are in ``jqmc/_setting.py``.
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

import os

import numpy as np
import pytest

import jqmc_rs

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_dtype_np
from jqmc._setting import (
    AO_CART_PERIODIC_VALUE_COHERENT_BIAS_EPS_FRACTION,
    AO_CART_PERIODIC_VALUE_COHERENT_BIAS_MIN_EFFECTIVE_POINTS,
    AO_VALUE_COHERENT_BIAS_FP64_EPS_MULTIPLE,
    AO_VALUE_COHERENT_BIAS_STANDARD_ERROR_FRACTION,
)
from jqmc.atomic_orbital import AOs_cart_pgto_data, compute_AOs
from jqmc.trexio_wrapper import read_trexio_file

#: Committed periodic Gamma Cartesian-pGTO fixture (bcc Li, ccECP, the Cartesian twin of
#: ``lithium.h5`` used by the spherical coherent-bias guarantees).
_TREXIO_FILE = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files", "lithium_cart.h5")
#: Number of fixed sample points (uniform in the cell), evaluated in batches of ``_BATCH``.
_N_POINTS = 32768
_BATCH = 4096
_SEED = 0

_ROUTES = {
    "xla_compute_AOs": lambda aos, r: compute_AOs(aos, r),
    "rust_compute_AOs_exact": lambda aos, r: jqmc_rs.compute_AOs_exact(aos, r),
    "rust_compute_AOs": lambda aos, r: jqmc_rs.compute_AOs(aos, r),
}


def _leaf():
    structure, aos, *_ = read_trexio_file(trexio_file=_TREXIO_FILE, store_tuple=True)
    assert type(aos) is AOs_cart_pgto_data and bool(structure.pbc_flag)
    return aos


def _points(aos, seed=_SEED, n_points=_N_POINTS):
    cell = np.asarray(aos.structure_data.cell, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.random((n_points, 3)) @ cell)


def _s_reference_model(aos):
    """Return ``(home_rows, terms)``: the s home AOs and, per AO, an fp64 image-term evaluator.

    No jqmc AO evaluation, normalization or image-enumeration helper is used. ``terms[row]``
    maps ``(points, image_indices)`` to ``(n_points, n_images)`` values
    ``sum_p c_p (2 Z_p / pi)^(3/4) exp(-Z_p |r - R_A - n . L|^2)``; ``kept[row]`` is the leaf's
    declared kept image set of that AO.
    """
    cell = np.asarray(aos.structure_data.cell, dtype=np.float64)
    positions = np.asarray(aos.structure_data.positions, dtype=np.float64)
    kernel_to_home = np.asarray(aos._ragged_kernel_to_home, dtype=np.int64)
    kernel_atom = np.asarray(aos._ragged_kernel_atom, dtype=np.int64)
    kernel_image = np.asarray(aos._image_indices, dtype=np.int64).reshape(-1, 3)
    kernel_l = np.asarray(aos.angular_momentums, dtype=np.int64)
    poly = np.stack(
        [np.asarray(getattr(aos, f"polynominal_order_{a}"), dtype=np.int64) for a in "xyz"],
        axis=1,
    )
    prim_to_kernel = np.asarray(aos.orbital_indices, dtype=np.int64)
    prim_exponents = np.asarray(aos.exponents, dtype=np.float64)
    prim_coefficients = np.asarray(aos.coefficients, dtype=np.float64)
    assert kernel_to_home.shape == kernel_atom.shape == kernel_l.shape == (kernel_image.shape[0],)
    assert poly.shape == (kernel_image.shape[0], 3)

    home_rows = np.unique(kernel_to_home[kernel_l == 0])
    models = []
    for home in home_rows:
        kernels = np.flatnonzero(kernel_to_home == home)
        images = kernel_image[kernels]
        # One atom, one primitive list per AO; s polynomial; each kept image once, T_0 included.
        assert np.unique(kernel_atom[kernels]).size == 1 and np.all(kernel_l[kernels] == 0)
        assert np.all(poly[kernels] == 0)
        assert np.unique(images, axis=0).shape[0] == images.shape[0]
        assert np.any(np.all(images == 0, axis=1))
        atom = int(kernel_atom[kernels[0]])
        exponents = prim_exponents[prim_to_kernel == kernels[0]]
        coefficients = prim_coefficients[prim_to_kernel == kernels[0]]
        assert exponents.size >= 1 and np.all(exponents > 0.0)
        for kernel in kernels[1:]:
            np.testing.assert_array_equal(prim_exponents[prim_to_kernel == kernel], exponents)
            np.testing.assert_array_equal(prim_coefficients[prim_to_kernel == kernel], coefficients)
        prefactors = coefficients * (2.0 * exponents / np.pi) ** 0.75

        def image_terms(points, image_indices, atom=atom, exponents=exponents, prefactors=prefactors):
            centers = positions[atom] + image_indices.astype(np.float64) @ cell
            diff = points[:, None, :] - centers[None, :, :]
            r2 = np.einsum("nix,nix->ni", diff, diff)
            return np.einsum("p,pni->ni", prefactors, np.exp(-exponents[:, None, None] * r2[None, :, :]))

        models.append((int(home), images, image_terms))
    return home_rows, models


def _independent_s_lattice_sums(aos, points):
    """fp64 textbook lattice sums ``(n_s_ao, n_points)`` over the leaf's kept images.

    Also enumerates, per AO, a box two layers wider than the kept set and asserts that every
    image the leaf dropped is below its truncation tolerance at every point and that the box's
    outermost layer is entirely dropped.
    """
    tol = float(aos._image_truncation_tol)
    assert tol > 0.0
    home_rows, models = _s_reference_model(aos)
    values = np.empty((home_rows.size, points.shape[0]), dtype=np.float64)
    for row, (home, images, image_terms) in enumerate(models):
        n_box = int(np.max(np.abs(images))) + 2
        grid = np.arange(-n_box, n_box + 1)
        box = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
        kept = {tuple(int(x) for x in image) for image in images}
        assert kept <= {tuple(int(x) for x in b) for b in box}
        dropped = np.asarray([image for image in box if tuple(int(x) for x in image) not in kept])
        outer = np.max(np.abs(box), axis=1) == n_box
        assert not any(tuple(int(x) for x in image) in kept for image in box[outer])
        for start in range(0, points.shape[0], _BATCH):
            chunk = points[start : start + _BATCH]
            # Smallest image terms first, so the fp64 reference is not order-limited.
            values[row, start : start + _BATCH] = np.sort(image_terms(chunk, images), axis=1).sum(axis=1)
            dropped_max = float(np.max(np.abs(image_terms(chunk, dropped))))
            assert dropped_max < tol, (home, dropped_max, tol)
    return home_rows, values


def _route_values(route, aos, points, home_rows):
    out = np.empty((home_rows.size, points.shape[0]), dtype=np.float64)
    for start in range(0, points.shape[0], _BATCH):
        chunk = np.ascontiguousarray(points[start : start + _BATCH])
        out[:, start : start + _BATCH] = np.asarray(_ROUTES[route](aos, chunk), dtype=np.float64)[home_rows]
    return out


def coherent_scale_statistics(actual, reference):
    """Return ``(kappa, standard_error, n_eff)`` per row (see module docstring)."""
    error = actual - reference
    weight = reference * reference
    norm2 = np.sum(weight, axis=1)
    assert np.all(norm2 > 0.0)
    kappa = np.sum(reference * error, axis=1) / norm2
    residual = reference * error - kappa[:, None] * weight
    standard_error = np.sqrt(np.sum(residual * residual, axis=1)) / norm2
    n_eff = norm2 * norm2 / np.sum(weight * weight, axis=1)
    return kappa, standard_error, n_eff


def coherent_bias_bounds(zone_eps):
    """Return ``(kappa_bound, standard_error_bound, min_effective_points)`` for a zone epsilon."""
    kappa_bound = (
        AO_CART_PERIODIC_VALUE_COHERENT_BIAS_EPS_FRACTION * zone_eps
        + AO_VALUE_COHERENT_BIAS_FP64_EPS_MULTIPLE * float(np.finfo(np.float64).eps)
    )
    return (
        kappa_bound,
        AO_VALUE_COHERENT_BIAS_STANDARD_ERROR_FRACTION * kappa_bound,
        AO_CART_PERIODIC_VALUE_COHERENT_BIAS_MIN_EFFECTIVE_POINTS,
    )


def measure(route, seed=_SEED, n_points=_N_POINTS):
    """Return ``(home_rows, kappa, standard_error, n_eff)`` of one route on the fixture."""
    aos = _leaf()
    points = _points(aos, seed=seed, n_points=n_points)
    home_rows, reference = _independent_s_lattice_sums(aos, points)
    actual = _route_values(route, aos, points, home_rows)
    assert actual.shape == reference.shape == (home_rows.size, n_points)
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    return (home_rows, *coherent_scale_statistics(actual, reference))


@pytest.mark.light
@pytest.mark.parametrize("route", list(_ROUTES))
def test_cart_pgto_value_has_no_coherent_scale_bias_vs_independent_lattice_sum(route):
    """Periodic Cartesian pGTO s-AO values carry no resolved coherent scale bias vs an fp64 lattice sum."""
    home_rows, kappa, standard_error, n_eff = measure(route)
    # Both s shells of the fixture on both atoms.
    assert home_rows.size >= 4
    zone_eps = float(np.finfo(get_dtype_np("ao_eval")).eps)
    bound, standard_error_bound, min_n_eff = coherent_bias_bounds(zone_eps)
    # Resolution guard: the verdict below must not be decided by sampling noise.
    assert np.all(n_eff >= min_n_eff), ("unresolved: effective sample size too small", route, n_eff, min_n_eff)
    assert np.all(standard_error <= standard_error_bound), (
        "unresolved: weighted standard error exceeds its fraction of the bound",
        route,
        standard_error,
        standard_error_bound,
    )
    assert np.all(np.abs(kappa) <= bound), (route, home_rows, kappa, bound)
