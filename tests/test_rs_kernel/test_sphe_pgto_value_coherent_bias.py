"""Coherent value-scale bias of the periodic spherical pGTO value routes (issue #145).

In mixed precision the standalone AO value kernels (XLA ``compute_AOs`` through the
non-guarded periodic branch of ``AOs_sphe_gto_data._compute_AOs``, and the Rust f32 value
kernel behind ``jqmc_rs.compute_AOs`` / ``compute_AOs_exact``) evaluate the coordinate
dependent part (r^2, exp) in float32. The electron-independent radial factor
N_n N_lm c and the exponent Z must stay at storage precision and enter the value through ONE
rounding of the primitive product; otherwise the float32 rounding of N_n, N_lm, c or of Z
inside exp(-Z r^2) is carried coherently into every value of that primitive.

The fast-update kinetic energy (``compute_local_energy_fast`` and its native twin) pairs the
inverse of the geminal built from these values with the float64 AO gradient / Laplacian
kernels, so such a coherent value bias shifts e_L systematically. Rounding the
normalization pieces to float32 separately gives the mixed periodic value a ~0.9
float32-eps coherent bias on the diffuse s shell of the periodic Gamma sphe-pGTO fixture,
and the native e_L then does not close with its components
(``test_mcmc_e_L_component_closure[pbc_gamma_sphepgto_j12_pade]``).

The reference route is the fused value of ``compute_AOs_value_grad_lap``, whose radial and
angular body is evaluated in the float64 ``ao_grad_lap`` zone and rounded once to ``ao_eval``
(the value the component kinetic energy uses). For each s-type AO (the solid harmonic S_00 is
exactly one in both backends, so the statistic isolates the radial factor) the coherent
relative scale

    kappa = sum_n v_ref (v - v_ref) / sum_n v_ref^2

over 4096 fixed points uniformly drawn in the cell must satisfy
|kappa| <= AO_VALUE_COHERENT_BIAS_EPS_FRACTION * eps(ao_eval)
+ AO_VALUE_COHERENT_BIAS_FP64_EPS_MULTIPLE * eps(float64); the second term admits coherent
float64 effects that are not zone rounding (association of the storage-precision constant factor,
the Rust polynomial fp64 exp): the whole budget in full precision, negligible in mixed. The random part of the
per-point error is resolved: its standard error relrms / sqrt(N) is asserted to be at most
AO_VALUE_COHERENT_BIAS_STANDARD_ERROR_FRACTION of the bound, so the verdict is not dominated by sampling noise.

The fused value shares the production normalization and lattice-image conventions with the
routes it checks, so the same statistic is also asserted against an implementation-independent
reference (``test_sphe_pgto_value_has_no_coherent_scale_bias_vs_independent_lattice_sum``): an
fp64 numpy lattice sum of textbook-normalized s Gaussians that calls no jqmc AO code. For l = 0
the normalized real spherical harmonic is the constant Y_00 = 1/sqrt(4 pi) and the radial
normalization from int_0^inf r^2 R(r)^2 dr = 1 is 2 (2 Z)^(3/4) pi^(-1/4), so the normalized
primitive is (2 Z / pi)^(3/4) exp(-Z r^2) and the periodic contracted AO is

    chi(r) = sum_n sum_p c_p (2 Z_p / pi)^(3/4) exp(-Z_p |r - R_A - n . L|^2).

Only basis input data (exponents, coefficients, atom of each AO, positions, cell) and the
declared kept lattice-image set of each AO are read from the leaf. The image set is the leaf's
declared truncation of the lattice sum (each dropped image below ``_image_truncation_tol`` in AO
value); the reference sums exactly that set so the coherent statistic is not dominated by the
(intended, sub-tolerance) truncation remainder, and the test independently enumerates a
surrounding image box and asserts that every image the leaf dropped is indeed below that
tolerance at every sample point, with the box's outermost layer entirely dropped.
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

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_dtype_np
from jqmc._setting import ao_value_coherent_bias_bounds
from jqmc.atomic_orbital import compute_AOs, compute_AOs_value_grad_lap
from tests.test_rs_kernel._rs_fixtures import rs_energy_ham

#: Committed periodic Gamma spherical-pGTO fixture of the native e_L closure test.
_PATTERN = "pbc_gamma_sphepgto_j12_pade"
#: Number of fixed sample points (uniform in the cell, seed 0).
_N_POINTS = 4096

_ROUTES = {
    "xla_compute_AOs": lambda aos, r: compute_AOs(aos, r),
    "rust_compute_AOs_exact": lambda aos, r: jqmc_rs.compute_AOs_exact(aos, r),
    "rust_compute_AOs": lambda aos, r: jqmc_rs.compute_AOs(aos, r),
}


def _leaf_and_points():
    ham = rs_energy_ham(_PATTERN)
    orb = ham.wavefunction_data.geminal_data.orb_data_up_spin
    aos = orb.aos_data if hasattr(orb, "aos_data") else orb
    assert type(aos).__name__ == "AOs_sphe_pgto_data" and bool(ham.structure_data.pbc_flag)
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    rng = np.random.default_rng(0)
    points = np.ascontiguousarray(rng.random((_N_POINTS, 3)) @ cell)
    return aos, points


def _assert_no_coherent_scale_bias(route):
    """Assert the coherent-scale statistic of one value route against the fused fp64-body value."""
    aos, points = _leaf_and_points()
    # Angular momentum of each molecular (home) AO; the leaf lists it at kernel rank.
    kernel_to_home = np.asarray(aos._ragged_kernel_to_home, dtype=np.int64)
    home_l = np.full(int(kernel_to_home.max()) + 1, -1, dtype=np.int64)
    home_l[kernel_to_home] = np.asarray(aos.angular_momentums, dtype=np.int64)
    s_rows = np.flatnonzero(home_l == 0)
    # Both s shells of the fixture (the diffuse and the compact exponent) on both atoms.
    assert s_rows.size >= 2
    reference = np.asarray(compute_AOs_value_grad_lap(aos, points)[0], dtype=np.float64)[s_rows]
    actual = np.asarray(_ROUTES[route](aos, points), dtype=np.float64)[s_rows]
    assert actual.shape == reference.shape == (s_rows.size, _N_POINTS)
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))

    error = actual - reference
    norm2 = np.sum(reference * reference, axis=1)
    assert np.all(norm2 > 0.0)
    kappa = np.sum(reference * error, axis=1) / norm2
    standard_error = np.sqrt(np.sum(error * error, axis=1) / norm2) / np.sqrt(_N_POINTS)
    bound, standard_error_bound = ao_value_coherent_bias_bounds(float(np.finfo(get_dtype_np("ao_eval")).eps))
    assert np.all(standard_error <= standard_error_bound), (route, standard_error, standard_error_bound)
    assert np.all(np.abs(kappa) <= bound), (route, kappa, bound)


@pytest.mark.light
def test_sphe_pgto_xla_value_has_no_coherent_scale_bias():
    """XLA ``compute_AOs`` periodic sphe-pGTO s-AO values carry no coherent scale bias vs the fused value."""
    _assert_no_coherent_scale_bias("xla_compute_AOs")


@pytest.mark.light
@pytest.mark.parametrize("route", ["rust_compute_AOs_exact", "rust_compute_AOs"])
def test_sphe_pgto_rust_value_has_no_coherent_scale_bias(route):
    """Rust periodic sphe-pGTO s-AO values carry no coherent scale bias vs the (XLA) fused value."""
    _assert_no_coherent_scale_bias(route)


#: Value routes judged against the independent lattice sum: the three standalone routes of the
#: route-consistency check above plus the fused value they were compared with.
_INDEPENDENT_ROUTES = {
    **_ROUTES,
    "xla_compute_AOs_value_grad_lap": lambda aos, r: compute_AOs_value_grad_lap(aos, r)[0],
}


def _independent_s_lattice_sums(aos, points):
    """Return ``(home_rows, values)`` of the fp64 textbook lattice sum for every s-type home AO.

    No jqmc AO evaluation, normalization or image-enumeration helper is used: the values are
    ``sum_n sum_p c_p (2 Z_p / pi)^(3/4) exp(-Z_p |r - R_A - n . L|^2)`` over the leaf's declared
    kept images ``n`` of that AO, summed in fp64 numpy. Every image of an independently enumerated
    box that the leaf dropped is asserted to be below the leaf's truncation tolerance at every point.
    """
    cell = np.asarray(aos.structure_data.cell, dtype=np.float64)
    positions = np.asarray(aos.structure_data.positions, dtype=np.float64)
    kernel_to_home = np.asarray(aos._ragged_kernel_to_home, dtype=np.int64)
    kernel_atom = np.asarray(aos._ragged_kernel_atom, dtype=np.int64)
    kernel_image = np.asarray(aos._image_indices, dtype=np.int64).reshape(-1, 3)
    kernel_l = np.asarray(aos.angular_momentums, dtype=np.int64)
    prim_to_kernel = np.asarray(aos.orbital_indices, dtype=np.int64)
    prim_exponents = np.asarray(aos.exponents, dtype=np.float64)
    prim_coefficients = np.asarray(aos.coefficients, dtype=np.float64)
    tol = float(aos._image_truncation_tol)
    assert kernel_to_home.shape == kernel_atom.shape == kernel_l.shape == (kernel_image.shape[0],)
    assert tol > 0.0

    home_rows = np.unique(kernel_to_home[kernel_l == 0])
    values = np.empty((home_rows.size, points.shape[0]), dtype=np.float64)
    for row, home in enumerate(home_rows):
        kernels = np.flatnonzero(kernel_to_home == home)
        images = kernel_image[kernels]
        # One atom and one primitive list per AO; each kept image appears once, T_0 included.
        assert np.unique(kernel_atom[kernels]).size == 1 and np.all(kernel_l[kernels] == 0)
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

        def image_terms(image_indices, atom=atom, exponents=exponents, prefactors=prefactors):
            centers = positions[atom] + image_indices.astype(np.float64) @ cell
            diff = points[:, None, :] - centers[None, :, :]
            r2 = np.einsum("nix,nix->ni", diff, diff)
            return np.einsum("p,pni->ni", prefactors, np.exp(-exponents[:, None, None] * r2[None, :, :]))

        # Sum each point's image terms smallest first, so the fp64 reference is not order-limited.
        values[row] = np.sort(image_terms(images), axis=1).sum(axis=1)

        # Independent enumeration: a box two layers wider than the kept set; every dropped image
        # is below the declared truncation tolerance, and the outermost layer is entirely dropped.
        n_box = int(np.max(np.abs(images))) + 2
        grid = np.arange(-n_box, n_box + 1)
        box = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
        kept = {tuple(int(x) for x in image) for image in images}
        assert all(image in {tuple(int(x) for x in b) for b in box} for image in kept)
        dropped = np.asarray([image for image in box if tuple(int(x) for x in image) not in kept])
        outer = np.max(np.abs(box), axis=1) == n_box
        assert not any(tuple(int(x) for x in image) in kept for image in box[outer])
        dropped_max = float(np.max(np.abs(image_terms(dropped))))
        assert dropped_max < tol, (int(home), dropped_max, tol)
    return home_rows, values


@pytest.mark.light
@pytest.mark.parametrize("route", list(_INDEPENDENT_ROUTES))
def test_sphe_pgto_value_has_no_coherent_scale_bias_vs_independent_lattice_sum(route):
    """Periodic sphe-pGTO s-AO values carry no coherent scale bias vs an independent fp64 lattice sum."""
    aos, points = _leaf_and_points()
    home_rows, reference = _independent_s_lattice_sums(aos, points)
    # Both s shells of the fixture (the diffuse and the compact exponent) on both atoms.
    assert home_rows.size >= 2
    actual = np.asarray(_INDEPENDENT_ROUTES[route](aos, points), dtype=np.float64)[home_rows]
    assert actual.shape == reference.shape == (home_rows.size, _N_POINTS)
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))

    error = actual - reference
    norm2 = np.sum(reference * reference, axis=1)
    assert np.all(norm2 > 0.0)
    kappa = np.sum(reference * error, axis=1) / norm2
    standard_error = np.sqrt(np.sum(error * error, axis=1) / norm2) / np.sqrt(_N_POINTS)
    bound, standard_error_bound = ao_value_coherent_bias_bounds(float(np.finfo(get_dtype_np("ao_eval")).eps))
    assert np.all(standard_error <= standard_error_bound), (route, standard_error, standard_error_bound)
    assert np.all(np.abs(kappa) <= bound), (route, kappa, bound)
