"""FAST-mode AO derivative screens drop only derivatives below 1e-12 in absolute AO units (issue #100).

The screened FAST AO gradient / Laplacian kernels (PR #97, issue #87) zero a
shell's derivative rows where ``pz_deriv * r^2 > LEPSBAS``. A ``pz_deriv`` that
only keeps every dropped gradient component and Laplacian below
``exp(-LEPSBAS) ~= 1e-9`` passes the strict ``ao_grad_lap`` comparison pair
(atol 1e-8), but on a spatially extended input the per-AO 1e-9 bound is amplified
in the FAST local energy (a linear (H2O)8 chain keeps a median per-walker e_L
error of ~2e-7 Ha from the derivative screen alone), so both FAST screens use the
bound ``FAST_AO_SCREEN_ABS_BOUND = 1e-12`` (issue #100; jqmc._setting; mirrors
``FAST_SCREEN_ABS_BOUND`` in jqmc/rs/src/atomic_orbital.rs).

This module reaches the three derivative-screen kernels (``batch``:
``compute_aos_grad_lap_fast_cut``; ``single``:
``compute_aos_single_value_grad_lap_fast_cut_into``; ``folded``:
``compute_aos_single_value_grad_lap_fast_cut_folded_into``) through the
test-only binding ``jqmc_rs.compute_AOs_grad_lap_fast_cut`` and checks them:

1. against the exact XLA ``compute_AOs_grad`` / ``compute_AOs_laplacian`` on the
   committed ccECP inputs ``water_ccecp_ccpvqz.h5`` (spherical) and
   ``CuBr_ecp_ccpvtz_cart.h5`` (Cartesian), at points around every nucleus with
   ``pz_min * r^2`` from 5 to 80 for every distinct shell exponent;
2. against the independent closed-form derivatives of one accepted Cartesian
   ``l = 82`` AO ``c N x^82 exp(-r^2)``, whose derivatives are below the bound at
   the LEPSBAS start ``r^2 = LEPSBAS`` and far above it farther out (the high-l
   monotone-start case of PR #97, rescaled to the 1e-12 bound).

Every derivative a kernel drops (an exact 0.0 where the reference is non-zero)
must have ``|reference| <= 1e-12``; every kept one must match the reference
within the strict XLA-Rust consistency pair of the ``ao_grad_lap`` zone (fp64 in
both precision modes). Non-vacuity: beyond the 1e-9 derivative cut,
reference derivatives exceed 1e-12, so a 1e-9 screen drops terms this
module rejects.
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

import math
import pathlib

import numpy as np
import pytest

import jqmc_rs

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import FAST_AO_SCREEN_ABS_BOUND
from jqmc.atomic_orbital import (
    AOs_cart_gto_data,
    AOs_sphe_gto_data,
    compute_AOs_grad,
    compute_AOs_laplacian,
)
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file

_ZONES_GRAD_LAP = ("ao_grad_lap",)
_ROUTES = ("batch", "single", "folded")
_NAMES = ("gx", "gy", "gz", "lap")
_TREXIO_DIR = pathlib.Path(__file__).resolve().parents[1] / "trexio_example_files"
_INPUTS = (
    ("water_ccecp_ccpvqz.h5", "sphe"),
    ("CuBr_ecp_ccpvtz_cart.h5", "cart"),
)
# Shell-relative start of the cut search, zeta_min * r^2 = LEPSBAS
# (LEPSBAS in jqmc/rs/src/atomic_orbital.rs).
_LEPSBAS = 20.72
_SCREEN_BOUND = FAST_AO_SCREEN_ABS_BOUND
# The 1e-9 bound of PR #97 (exp(-LEPSBAS)); used only for non-vacuity.
_FORMER_BOUND = math.exp(-_LEPSBAS)
_ZETA_R2 = np.concatenate([np.geomspace(5.0, 80.0, 9), (20.8, 22.0, 24.0, 26.0, 30.0, 35.0, 45.0, 50.0)])
_DIRECTIONS = 4
# Accepted high-l Cartesian AO (closed form), PR #97 shape: x^82 exp(-r^2).
_HIGH_L_POWER = 82
_HIGH_L_ZETA = 1.0
# PR #97's coefficient 4.01499518920269e-9 gives a derivative bound of
# 8.0e-10 at r^2 = LEPSBAS; scaled by 5e-4 it is ~4e-13 there (below 1e-12) while
# the exact Laplacian on the x axis at r^2 = 41 is ~1e-11 (above it).
_HIGH_L_COEF = 2.0e-12
_HIGH_L_R2 = (10.0, 15.0, 20.0, 20.72, 25.0, 30.0, 35.0, 41.0, 43.0, 47.0, 55.0, 60.0, 70.0, 80.0, 90.0, 110.0)
_HIGH_L_TILTS = ((0.0, 0.0), (0.08, 0.0), (0.0, -0.05))


def _kept_after_dropped_bound(label, got, ref):
    """Assert every dropped derivative (exact 0.0 where the reference is non-zero) is below the bound; return the kept mask."""
    assert got.shape == ref.shape, label
    assert np.all(np.isfinite(got)), f"{label}: non-finite"
    dropped = (got == 0.0) & (ref != 0.0)
    if np.any(dropped):
        worst = float(np.max(np.abs(ref[dropped])))
        assert worst <= _SCREEN_BOUND, f"{label}: a derivative screen dropped {worst:.3e} > {_SCREEN_BOUND:.1e}"
    return ~dropped


def _points_around_nuclei(aos):
    """Per nucleus and distinct shell exponent: radii sqrt(t / zeta), seeded random directions."""
    rng = np.random.default_rng(1001)
    positions = np.asarray(aos.structure_data.positions, dtype=np.float64)
    exps = np.asarray(aos.exponents, dtype=np.float64)
    orbital_indices = np.asarray(aos.orbital_indices)
    nucleus = np.asarray(aos.nucleus_index)
    pts = []
    for atom in range(positions.shape[0]):
        ao_on_atom = np.flatnonzero(nucleus == atom)
        if ao_on_atom.size == 0:
            continue
        pz_min = sorted({float(np.min(exps[orbital_indices == ao])) for ao in ao_on_atom})
        for zeta in pz_min:
            for t in _ZETA_R2:
                d = rng.normal(size=(_DIRECTIONS, 3))
                d /= np.linalg.norm(d, axis=1)[:, None]
                pts.append(positions[atom] + math.sqrt(t / zeta) * d)
    return np.ascontiguousarray(np.concatenate(pts), dtype=np.float64)


@pytest.mark.light
@pytest.mark.parametrize(("trexio_file", "kind"), _INPUTS)
def test_fast_derivative_screens_drop_only_terms_below_absolute_bound(trexio_file, kind):
    """Screened FAST AO grad/lap equal the exact XLA derivatives; dropped ones are below 1e-12."""
    (_, aos, *_rest) = read_trexio_file(trexio_file=str(_TREXIO_DIR / trexio_file), store_tuple=True)
    assert isinstance(aos, AOs_sphe_gto_data if kind == "sphe" else AOs_cart_gto_data)
    r = _points_around_nuclei(aos)
    exact = (
        *[np.asarray(g, dtype=np.float64) for g in compute_AOs_grad(aos, r)],
        np.asarray(compute_AOs_laplacian(aos, r), dtype=np.float64),
    )
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD_LAP, level="strict")
    for name, ref in zip(_NAMES, exact, strict=True):
        assert ref.shape == (aos.num_ao, r.shape[0]), name
        assert np.all(np.isfinite(ref)), f"{trexio_file}: exact {name} not finite"

    # Non-vacuity: somewhere the exact derivatives lie between the 1e-12 and the
    # 1e-9 bound, i.e. in the band only the 1e-12 screen must keep.
    between = max(int(np.count_nonzero((np.abs(ref) > _SCREEN_BOUND) & (np.abs(ref) < _FORMER_BOUND))) for ref in exact)
    assert between > 0, f"{trexio_file}: no derivative sample between the 1e-12 and 1e-9 bounds"

    for route in _ROUTES:
        got = [np.asarray(x) for x in jqmc_rs.compute_AOs_grad_lap_fast_cut(aos, r, route)]
        for name, val, ref in zip(_NAMES, got, exact, strict=True):
            label = f"{trexio_file} {route}: {name} FAST screened vs exact XLA"
            kept = _kept_after_dropped_bound(label, val, ref)
            np.testing.assert_allclose(val[kept], ref[kept], atol=atol, rtol=rtol, err_msg=label)


def _build_high_l_aos():
    structure = Structure_data(
        pbc_flag=False,
        positions=np.zeros((1, 3), dtype=np.float64),
        atomic_numbers=(18,),
        element_symbols=("Ar",),
        atomic_labels=("Ar",),
    )
    return AOs_cart_gto_data.build(
        structure_data=structure,
        nucleus_index=[0],
        num_ao=1,
        num_ao_prim=1,
        orbital_indices=[0],
        exponents=np.asarray([_HIGH_L_ZETA], dtype=np.float64),
        coefficients=np.asarray([_HIGH_L_COEF], dtype=np.float64),
        angular_momentums=[_HIGH_L_POWER],
        polynominal_order_x=[_HIGH_L_POWER],
        polynominal_order_y=[0],
        polynominal_order_z=[0],
    )


def _high_l_points():
    """Points on the x axis and two tilted rays; returns (points, nominal r^2 per point)."""
    pts, nominal = [], []
    for r2 in _HIGH_L_R2:
        r = math.sqrt(r2)
        for ty, tz in _HIGH_L_TILTS:
            d = np.asarray([math.cos(ty) * math.cos(tz), math.sin(ty), math.sin(tz)])
            pts.append(r * d / np.linalg.norm(d))
            nominal.append(r2)
    return np.ascontiguousarray(np.asarray(pts), dtype=np.float64), np.asarray(nominal)


def _high_l_closed_form(r, coef):
    """Independent derivatives of c N x^n exp(-zeta r^2), N^2 = (2 zeta / pi)^(3/2) (4 zeta)^n / (2n-1)!!."""
    n, z = _HIGH_L_POWER, _HIGH_L_ZETA
    log_double_factorial = math.lgamma(2 * n + 1) - n * math.log(2.0) - math.lgamma(n + 1)  # log (2n-1)!!
    log_norm = 0.5 * (1.5 * math.log(2.0 * z / math.pi) + n * math.log(4.0 * z) - log_double_factorial)
    x, y, zc = r[:, 0], r[:, 1], r[:, 2]
    r2 = np.einsum("ij,ij->i", r, r)
    # Assembled in logs so N (~1e-49) and x^82 (~1e54) never meet as separate factors.
    phi = coef * np.exp(log_norm + n * np.log(np.abs(x)) - z * r2) * np.sign(x) ** n
    gx = phi * (n / x - 2.0 * z * x)
    gy = -2.0 * z * y * phi
    gz = -2.0 * z * zc * phi
    lap = phi * (n * (n - 1) / x**2 - 2.0 * z * (2 * n + 3) + 4.0 * z * z * r2)
    return gx[None, :], gy[None, :], gz[None, :], lap[None, :]


@pytest.mark.light
def test_fast_derivative_screens_absolute_bound_for_high_angular_momentum():
    """Screened FAST AO grad/lap of an accepted l=82 Cartesian AO equal its closed form; dropped ones are below 1e-12."""
    aos = _build_high_l_aos()
    r, nominal_r2 = _high_l_points()
    coef = float(np.asarray(aos.coefficients, dtype=np.float64)[0])
    exact = _high_l_closed_form(r, coef)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD_LAP, level="strict")
    for name, ref in zip(_NAMES, exact, strict=True):
        assert np.all(np.isfinite(ref)), f"closed-form {name} not finite"

    # Non-vacuity: below the bound at the LEPSBAS start r^2 = LEPSBAS, above it
    # farther out.
    at_start = nominal_r2 == _LEPSBAS
    assert np.any(at_start)
    assert max(float(np.max(np.abs(ref[0, at_start]))) for ref in exact) < _SCREEN_BOUND
    beyond = _HIGH_L_ZETA * nominal_r2 > _LEPSBAS
    assert max(float(np.max(np.abs(ref[0, beyond]))) for ref in exact) > _SCREEN_BOUND * (1.0 + rtol)

    for route in _ROUTES:
        got = [np.asarray(x) for x in jqmc_rs.compute_AOs_grad_lap_fast_cut(aos, r, route)]
        for name, val, ref in zip(_NAMES, got, exact, strict=True):
            label = f"l=82 cart {route}: {name} FAST screened vs closed form"
            kept = _kept_after_dropped_bound(label, val, ref)
            np.testing.assert_allclose(val[kept], ref[kept], atol=atol, rtol=rtol, err_msg=label)
