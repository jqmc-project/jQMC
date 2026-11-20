"""Periodic pGTO image truncation bounds the normalised AO value, XLA and Rust (issue #120).

The periodic AO is the lattice sum ``chi~_mu(r) = sum_n chi_mu(r - R_A - T_n)``.
Production truncates this sum once, at build time: ``_apply_image_expansion``
keeps, per shell, only the images that can reach the home cell, and documents
that every dropped image is below ``tol`` (``AO_IMAGE_TRUNCATION_ATOL``) at every
point of the home cell. The AO value of one primitive is
``c N(alpha, l) A(r) exp(-alpha r^2)`` with ``|A(r)| <= r^l``. The image radius used
to bound only ``r^l exp(-alpha r^2)``, without ``|c| N(alpha, l)``. For tight
d/f shells ``|c| N`` is 10 to 80, so a dropped image could exceed ``tol`` by that
factor (O ccECP cc-pVQZ d, alpha 4.008867: 1.2e-6; Cu ccECP cc-pVTZ f,
alpha 5.230236, c 1.3398: about 9e-6).

This module builds one tight shell with the public ``AOs_cart_pgto_data.build`` /
``AOs_sphe_pgto_data.build`` factories, with the atom at a general position of a
cubic cell, and checks two things:

* every image that the leaf leaves out of the shell's image set, evaluated with
  the open-boundary (molecular) AO kernel at the home-cell points closest to that
  image and at seeded points around them, is at most ``AO_IMAGE_TRUNCATION_ATOL``.
  Each dropped value is evaluated directly, not as a difference, so this check
  has the same meaning in full and mixed precision;
* XLA ``compute_AOs`` and Rust ``jqmc_rs.compute_AOs`` of the periodic leaf at the
  same points equal the brute-force lattice sum of the molecular AO kernel over
  every image inside a Gaussian cutoff that the test derives itself
  (``_noncubic_oracles``), within the ao_eval strict policy plus
  ``AO_IMAGE_TRUNCATION_ATOL``. The brute-force sum is reduced in the ao_eval zone
  dtype, so it is fp64 in full precision.

Neither check uses the production image radius, the per-atom image selector or
the ``_debug`` image-expansion oracle; the kept set is read from the built leaf.

A separate contracted-shell witness checks that the per-image
bound covers the whole contraction: two primitives of one d shell, each below
``AO_IMAGE_TRUNCATION_ATOL`` at a home-cell point, whose sum is above it. The
leaf must keep that image, and XLA, Rust and the ``_debug`` oracle must match the
analytic lattice sum of the contracted AO.
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

import itertools
import math

import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance
from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL
from jqmc.atomic_orbital import (
    AOs_cart_gto_data,
    AOs_cart_pgto_data,
    AOs_sphe_gto_data,
    AOs_sphe_pgto_data,
    _compute_AOs_debug,
    _with_periodic_image_expansion_debug,
    compute_AOs,
)
from jqmc.structure import Structure_data
from tests.test_jqmc_kernel._noncubic_oracles import _gaussian_cutoff, lattice_images_within

#: case -> (kind, l, exponent, coefficient, cubic edge in Bohr, fractional atom position, (Z, symbol)).
#: The shells are the tight polarisation primitives of the committed ccECP bases
#: (O cc-pVQZ d; Cu cc-pVTZ f with its contraction coefficient), the cells about
#: the solid density, the positions general (not high-symmetry).
_CASES = {
    "o_qz_d_sphe": ("sphe", 2, 4.008867, 1.0, 5.9, (0.6913, 0.1786, 0.3963), (8, "O")),
    "o_qz_d_cart": ("cart", 2, 4.008867, 1.0, 5.9, (0.6913, 0.1786, 0.3963), (8, "O")),
    "cu_tz_f_sphe": ("sphe", 3, 5.230236, 1.3398, 6.822, (0.5865, 0.8397, 0.7265), (29, "Cu")),
    "cu_tz_f_cart": ("cart", 3, 5.230236, 1.3398, 6.822, (0.5865, 0.8397, 0.7265), (29, "Cu")),
}

#: Seeded points around each closest-approach point, and their spread in Bohr.
_POINTS_PER_IMAGE = 8
_POINT_SPREAD = 0.15


def _shell_fields(kind: str, l: int, exponent: float, coefficient: float) -> dict:
    """One uncontracted shell (all angular components) as builder keyword arguments."""
    if kind == "sphe":
        components = list(range(-l, l + 1))
        angular = dict(magnetic_quantum_numbers=tuple(components))
    else:
        components = [(nx, ny, l - nx - ny) for nx in range(l, -1, -1) for ny in range(l - nx, -1, -1)]
        angular = dict(
            polynominal_order_x=tuple(p[0] for p in components),
            polynominal_order_y=tuple(p[1] for p in components),
            polynominal_order_z=tuple(p[2] for p in components),
        )
    n = len(components)
    return dict(
        nucleus_index=(0,) * n,
        num_ao=n,
        num_ao_prim=n,
        orbital_indices=tuple(range(n)),
        exponents=np.full(n, exponent, dtype=np.float64),
        coefficients=np.full(n, coefficient, dtype=np.float64),
        angular_momentums=(l,) * n,
        **angular,
    )


def _structure(position: np.ndarray, cell: np.ndarray | None, element: tuple[int, str]) -> Structure_data:
    kwargs = dict(
        pbc_flag=cell is not None,
        positions=position[None, :],
        atomic_numbers=(element[0],),
        element_symbols=(element[1],),
        atomic_labels=(element[1],),
    )
    if cell is not None:
        kwargs.update(vec_a=tuple(cell[0]), vec_b=tuple(cell[1]), vec_c=tuple(cell[2]))
    return Structure_data.build(**kwargs)


def _kept_images(leaf) -> list[set[tuple[int, int, int]]]:
    """Per home AO, the lattice images the built leaf keeps (read from its ragged layout)."""
    images = np.asarray(leaf._image_indices, dtype=np.int64).reshape(-1, 3)
    home = np.asarray(leaf._ragged_kernel_to_home, dtype=np.int64)
    kept: list[set[tuple[int, int, int]]] = [set() for _ in range(int(leaf.num_ao))]
    for image, ao in zip(images, home, strict=True):
        kept[int(ao)].add(tuple(int(v) for v in image))
    return kept


@pytest.mark.light
@pytest.mark.parametrize("case", list(_CASES))
def test_pgto_dropped_images_stay_below_truncation_tolerance_for_tight_normalised_shells(case):
    """Every dropped periodic image of a tight normalised d/f shell is below AO_IMAGE_TRUNCATION_ATOL."""
    kind, l, exponent, coefficient, edge, frac, element = _CASES[case]
    cell = edge * np.eye(3, dtype=np.float64)
    position = np.asarray(frac, dtype=np.float64) @ cell
    fields = _shell_fields(kind, l, exponent, coefficient)
    if kind == "sphe":
        leaf = AOs_sphe_pgto_data.build(structure_data=_structure(position, cell, element), **fields)
        molecular = AOs_sphe_gto_data._from_full_basis(structure_data=_structure(position, None, element), **fields)
    else:
        leaf = AOs_cart_pgto_data.build(structure_data=_structure(position, cell, element), **fields)
        molecular = AOs_cart_gto_data._from_full_basis(structure_data=_structure(position, None, element), **fields)
    kept = _kept_images(leaf)
    assert all(images == kept[0] for images in kept), "the angular components of one shell must share an image set"
    kept_set = kept[0]

    # Home-cell points closest to every image in a box beyond the shell's reach,
    # plus seeded points around them, all inside the home cell [0, edge)^3.
    radius = _gaussian_cutoff(exponent, l)
    half = int(np.ceil(radius / edge)) + 1
    upper = edge * (1.0 - np.finfo(np.float64).eps)
    rng = np.random.default_rng(20261015)
    dropped, dropped_points, points = [], [], []
    for n in itertools.product(range(-half, half + 1), repeat=3):
        closest = np.clip(position + np.asarray(n, dtype=np.float64) * edge, 0.0, upper)
        near = np.clip(closest + _POINT_SPREAD * rng.normal(size=(_POINTS_PER_IMAGE, 3)), 0.0, upper)
        block = np.vstack([closest[None, :], near])
        points.append(block)
        if tuple(n) not in kept_set:
            dropped.append(n)
            dropped_points.append(block)
    assert dropped, f"{case}: no image is dropped, so the truncation is not exercised"
    points = np.unique(np.vstack(points), axis=0)

    # (1) Each dropped image, evaluated on its own with the open-boundary kernel.
    worst = 0.0
    for n, block in zip(dropped, dropped_points, strict=True):
        shifted = block - np.asarray(n, dtype=np.float64) * edge
        values = np.asarray(compute_AOs(aos_data=molecular, r_carts=jnp.asarray(shifted)), dtype=np.float64)
        worst = max(worst, float(np.max(np.abs(values))))
    assert worst <= AO_IMAGE_TRUNCATION_ATOL, (
        f"{case}: a dropped periodic image contributes {worst:.3e} to an AO value in the home cell, "
        f"above AO_IMAGE_TRUNCATION_ATOL = {AO_IMAGE_TRUNCATION_ATOL:.1e}"
    )

    # (2) XLA and Rust periodic AO values against the brute-force lattice sum.
    images = lattice_images_within(cell, points - position[None, :], radius)
    shifts = images.astype(np.float64) @ cell
    image_points = (points[None, :, :] - shifts[:, None, :]).reshape(-1, 3)
    per_image = np.asarray(compute_AOs(aos_data=molecular, r_carts=jnp.asarray(image_points)))
    brute = per_image.reshape(per_image.shape[0], len(images), len(points)).sum(axis=1)
    ao_atol, ao_rtol = get_tolerance("ao_eval", "strict")
    atol = AO_IMAGE_TRUNCATION_ATOL + ao_atol
    xla = np.asarray(compute_AOs(aos_data=leaf, r_carts=jnp.asarray(points)))
    rust = np.asarray(jqmc_rs.compute_AOs(leaf, points))
    for label, got in (("XLA", xla), ("Rust", rust)):
        assert got.shape == brute.shape
        np.testing.assert_allclose(
            got, brute, atol=atol, rtol=ao_rtol, err_msg=f"{case}: {label} periodic AO values vs the brute-force image sum"
        )


#: Contracted-shell witness: one spherical d shell, two primitives
#: with unit coefficients and nearly equal exponents, so each primitive of image
#: (0, 0, 1) is below AO_IMAGE_TRUNCATION_ATOL at the electron while their sum is
#: above it. Atom and electron on the cell's z axis line x = y = 3, cubic edge 6.
_CONTRACTED_EXPONENTS = (4.008867, 4.012)
_CONTRACTED_COEFFICIENTS = (1.0, 1.0)
_CONTRACTED_EDGE = 6.0
_CONTRACTED_ATOM = (3.0, 3.0, 2.2877506501988543)
_CONTRACTED_ELECTRON_Z_GAP = 1.0e-12
_CONTRACTED_IMAGE = (0, 0, 1)


def _analytic_sphe_d0_contraction(
    displacement: np.ndarray,
    exponents: tuple[float, ...] = _CONTRACTED_EXPONENTS,
    coefficients: tuple[float, ...] = _CONTRACTED_COEFFICIENTS,
) -> float:
    """Contracted real-spherical (l=2, m=0) AO value from the analytic primitive normalisation.

    ``chi(r) = sum_p c_p N_r(alpha_p) r^2 exp(-alpha_p r^2) Y_20(r)`` with the radial
    norm ``N_r^2 = 2 (2 alpha)^(l + 3/2) / Gamma(l + 3/2)`` (from
    ``int_0^inf r^(2l+2) exp(-2 alpha r^2) dr = Gamma(l + 3/2) / (2 (2 alpha)^(l + 3/2))``)
    and ``r^2 Y_20 = sqrt(5 / (16 pi)) (3 z^2 - r^2)``. No jqmc code is used.
    """
    l = 2
    r2 = float(displacement @ displacement)
    angular = math.sqrt(5.0 / (16.0 * math.pi)) * (3.0 * float(displacement[2]) ** 2 - r2)
    value = 0.0
    for alpha, coefficient in zip(exponents, coefficients, strict=True):
        radial_norm = math.sqrt(2.0 * (2.0 * alpha) ** (l + 1.5) / math.gamma(l + 1.5))
        value += coefficient * radial_norm * angular * math.exp(-alpha * r2)
    return value


@pytest.mark.light
def test_pgto_contracted_shell_keeps_image_whose_primitive_sum_exceeds_truncation_tolerance():
    """A contracted shell keeps an image whose summed primitives exceed AO_IMAGE_TRUNCATION_ATOL.

    Each primitive of image (0, 0, 1) is below the tolerance at the electron but
    the contracted AO value is about 1.5 AO_IMAGE_TRUNCATION_ATOL, so the image may
    be dropped only if the per-primitive share ignores the number of primitives
    of the AO. The leaf must keep that image, and XLA ``compute_AOs`` and Rust
    ``jqmc_rs.compute_AOs`` of the leaf, and ``_compute_AOs_debug`` of the
    ``_with_periodic_image_expansion_debug`` oracle, must equal the analytic
    lattice sum of the contracted AO (images within two cells) within the ao_eval
    strict relative tolerance.
    """
    l = 2
    components = list(range(-l, l + 1))
    n_ao = len(components)
    n_prim = len(_CONTRACTED_EXPONENTS)
    fields = dict(
        nucleus_index=(0,) * n_ao,
        num_ao=n_ao,
        num_ao_prim=n_ao * n_prim,
        orbital_indices=tuple(ao for ao in range(n_ao) for _ in range(n_prim)),
        exponents=np.tile(np.asarray(_CONTRACTED_EXPONENTS, dtype=np.float64), n_ao),
        coefficients=np.tile(np.asarray(_CONTRACTED_COEFFICIENTS, dtype=np.float64), n_ao),
        angular_momentums=(l,) * n_ao,
        magnetic_quantum_numbers=tuple(components),
    )
    cell = _CONTRACTED_EDGE * np.eye(3, dtype=np.float64)
    position = np.asarray(_CONTRACTED_ATOM, dtype=np.float64)
    structure = _structure(position, cell, (8, "O"))
    leaf = AOs_sphe_pgto_data.build(structure_data=structure, **fields)
    # The helper-free _debug image-expansion oracle selects its images with its
    # own inlined contraction-aware bound; it is checked against the same reference.
    debug_aos = _with_periodic_image_expansion_debug(
        AOs_sphe_gto_data._from_full_basis(structure_data=structure, **fields), structure, tol=AO_IMAGE_TRUNCATION_ATOL
    )
    m0 = components.index(0)

    electron = np.array([[_CONTRACTED_ATOM[0], _CONTRACTED_ATOM[1], _CONTRACTED_EDGE - _CONTRACTED_ELECTRON_Z_GAP]])
    image_displacement = electron[0] - position - np.asarray(_CONTRACTED_IMAGE, dtype=np.float64) * _CONTRACTED_EDGE
    image_value = _analytic_sphe_d0_contraction(image_displacement)
    primitive_values = [
        _analytic_sphe_d0_contraction(image_displacement, (alpha,), (coefficient,))
        for alpha, coefficient in zip(_CONTRACTED_EXPONENTS, _CONTRACTED_COEFFICIENTS, strict=True)
    ]
    # Non-vacuity of the witness: every primitive alone is below the tolerance,
    # the contracted AO value of the image is above it.
    assert all(abs(value) < AO_IMAGE_TRUNCATION_ATOL for value in primitive_values), primitive_values
    assert abs(image_value) > AO_IMAGE_TRUNCATION_ATOL, image_value

    kept = _kept_images(leaf)
    assert _CONTRACTED_IMAGE in kept[m0], (
        f"image {_CONTRACTED_IMAGE} contributes {image_value:.6e} > AO_IMAGE_TRUNCATION_ATOL = "
        f"{AO_IMAGE_TRUNCATION_ATOL:.1e} to the contracted m=0 AO but the leaf drops it"
    )

    reference = sum(
        _analytic_sphe_d0_contraction(electron[0] - position - np.asarray(n, dtype=np.float64) * _CONTRACTED_EDGE)
        for n in itertools.product(range(-2, 3), repeat=3)
    )
    _, ao_rtol = get_tolerance("ao_eval", "strict")
    xla = np.asarray(compute_AOs(aos_data=leaf, r_carts=jnp.asarray(electron)), dtype=np.float64)
    rust = np.asarray(jqmc_rs.compute_AOs(leaf, electron), dtype=np.float64)
    # The oracle returns its image-major kernel rows; fold the image axis here.
    debug_rows = np.asarray(_compute_AOs_debug(aos_data=debug_aos, r_carts=electron), dtype=np.float64)
    debug = debug_rows.reshape(int(debug_aos._n_image), n_ao, -1).sum(axis=0)
    for label, got in (("XLA", xla), ("Rust", rust), ("_debug oracle", debug)):
        assert got.shape == (n_ao, 1)
        relative_error = abs(float(got[m0, 0]) - reference) / abs(reference)
        assert relative_error <= ao_rtol, (
            f"{label} contracted m=0 AO {float(got[m0, 0]):.6e} vs analytic lattice sum {reference:.6e}: "
            f"relative error {relative_error:.3e} > ao_eval strict rtol {ao_rtol:.1e}"
        )
