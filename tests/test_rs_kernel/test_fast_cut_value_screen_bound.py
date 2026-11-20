"""FAST-mode AO value screens drop only AO values below 1e-12 in absolute AO units (issue #100).

The production FAST energy chunk evaluates AO VALUES through screened kernels
that skip a whole shell for a point far from the shell:

* ``mo_single``: ``compute_mo_single_cutoff_into`` (sphe) /
  ``compute_mo_single_cutoff_cart_into`` (cart), the per-move determinant
  orbital evaluation of the native MCMC sweep and of ``compute_mo_cutoff``;
* ``mesh_dot``: ``dot_aos_multi_cutoff_sphe`` / ``dot_aos_multi_cutoff_cart``,
  the ECP non-local mesh AO contraction (determinant and J3);
* ``dot_single``: ``dot_aos_single_cutoff_sphe`` (sphe only);
* ``candidate``: the GFMC molecular candidate AO batch.

A shell-relative screen that skips a shell where ``pz_min * r^2 > LEPSBAS``
(``LEPSBAS = 20.72``), i.e. once the shell's most diffuse primitive has decayed
by ``exp(-LEPSBAS) ~= 1e-9`` RELATIVE to the shell's peak factor
``|norm * coef|``, is not enough: the normalization of tight high-l shells is
large, so the dropped AO values reach ~1e-6 (Cu ccECP cc-pVTZ g: 1.4e-6;
O ccECP cc-pVQZ g: 7.3e-7; Br ccECP cc-pVTZ f: 1.2e-7), about 1e3 times
1e-9 (issue #100). The screens use a per-shell exponent chosen so that every
dropped AO value is below ``FAST_AO_SCREEN_ABS_BOUND = 1e-12`` (jqmc._setting;
mirrors ``FAST_SCREEN_ABS_BOUND`` in jqmc/rs/src/atomic_orbital.rs; issue
#100).

This module reaches the four kernels through the TEST-ONLY binding
``jqmc_rs.compute_AOs_value_fast_cut`` (identity MO coefficients / unit weight
vectors, so the output is the screened AO vector itself) and checks them. In
mixed precision the committed-input check also covers the per-move route of the
fp32 ``ao_eval`` zone (route ``single_f32``):

1. against the exact (unscreened) XLA ``compute_AOs`` on the committed ccECP
   inputs ``water_ccecp_ccpvqz.h5`` (spherical, O cc-pVQZ up to g) and
   ``CuBr_ecp_ccpvtz_cart.h5`` (Cartesian, Cu cc-pVTZ up to g, Br up to f), at
   points around every nucleus with ``pz_min * r^2`` from 5 to 80 for every
   distinct shell exponent, i.e. across the band where the shell-relative
   LEPSBAS cut switches shells off;
2. against an independent closed form for one accepted Cartesian ``l = 82`` AO
   (``x^82 exp(-r^2)``), whose value is below the bound at the LEPSBAS start
   ``r^2 = LEPSBAS`` and rises far above it farther out (the high-l
   monotone-start case of PR #97).

Assertions per kernel: every value the kernel keeps equals the reference within
the XLA-Rust consistency pair of the ``ao_eval`` zone (fp32 pair in mixed
precision, where XLA evaluates AOs in fp32); and every value it drops (an exact
0.0 where the reference is non-zero) has ``|reference| <= 1e-12``.
Non-vacuity: for every input, references at points beyond the LEPSBAS cut radius
exceed that bound, so a shell-relative LEPSBAS screen drops values this module rejects.
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

from jqmc._precision import get_dtype_np, get_tolerance_xla_rust_consistency
from jqmc._setting import FAST_AO_SCREEN_ABS_BOUND
from jqmc.atomic_orbital import AOs_cart_gto_data, AOs_sphe_gto_data, compute_AOs
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file

_ZONES_VALUE = ("ao_eval",)
_TREXIO_DIR = pathlib.Path(__file__).resolve().parents[1] / "trexio_example_files"
# LEPSBAS in jqmc/rs/src/atomic_orbital.rs: the shell-relative cut is
# pz_min * r^2 > LEPSBAS. The value screens drop only AO values below
# FAST_AO_SCREEN_ABS_BOUND.
_LEPSBAS = 20.72
_SCREEN_BOUND = FAST_AO_SCREEN_ABS_BOUND
# pz_min * r^2 band sampled around every nucleus for every distinct shell exponent.
# Extra samples just past the LEPSBAS cut zeta * r^2 = LEPSBAS, where the
# values a shell-relative screen drops are largest, and across the band where
# the 1e-12 cut of tight shells lies.
_ZETA_R2 = np.concatenate([np.geomspace(5.0, 80.0, 9), (20.8, 22.0, 24.0, 26.0, 30.0, 35.0, 45.0, 50.0)])
_DIRECTIONS = 4
_ROUTES = {
    "sphe": ("mo_single", "mesh_dot", "dot_single", "candidate"),
    "cart": ("mo_single", "mesh_dot", "candidate"),
}
_INPUTS = (
    ("water_ccecp_ccpvqz.h5", "sphe"),
    ("CuBr_ecp_ccpvtz_cart.h5", "cart"),
)
# Accepted high-l Cartesian AO (closed form): x^82 exp(-zeta r^2).
_HIGH_L_POWER = 82
_HIGH_L_ZETA = 1.0
# Chosen so the value at the LEPSBAS start r^2 = LEPSBAS is ~2.9e-13 (below the
# 1e-12 bound; for this single Cartesian AO the value bound is the value on the
# x axis) while the AO value peaks near r^2 = 41 at ~9e-10.
_HIGH_L_COEF = 1.0e-9
_HIGH_L_R2 = (10.0, 15.0, 20.0, 20.72, 25.0, 30.0, 35.0, 41.0, 47.0, 55.0, 60.0, 70.0, 80.0, 90.0, 110.0, 130.0)
_HIGH_L_TILTS = ((0.0, 0.0), (0.08, 0.0), (0.0, -0.05))


def _kept_after_dropped_bound(label, got, ref):
    """Assert every dropped value (exact 0.0 where the reference is non-zero) is below the bound; return the kept mask."""
    assert got.shape == ref.shape, label
    assert np.all(np.isfinite(got)), f"{label}: non-finite"
    dropped = (got == 0.0) & (ref != 0.0)
    if np.any(dropped):
        worst = float(np.max(np.abs(ref[dropped])))
        assert worst <= _SCREEN_BOUND, f"{label}: a value screen dropped |AO| = {worst:.3e} > {_SCREEN_BOUND:.1e}"
    return ~dropped


def _points_around_nuclei(aos):
    """Per nucleus and distinct shell exponent: radii sqrt(t / zeta), seeded random directions."""
    rng = np.random.default_rng(100)
    positions = np.asarray(aos.structure_data.positions, dtype=np.float64)
    exps = np.asarray(aos.exponents, dtype=np.float64)
    orbital_indices = np.asarray(aos.orbital_indices)
    nucleus = np.asarray(aos.nucleus_index)
    pts = []
    for atom in range(positions.shape[0]):
        ao_on_atom = np.flatnonzero(nucleus == atom)
        if ao_on_atom.size == 0:
            continue
        # Smallest exponent of every AO on this atom = the shell pz_min values.
        pz_min = sorted({float(np.min(exps[orbital_indices == ao])) for ao in ao_on_atom})
        for zeta in pz_min:
            for t in _ZETA_R2:
                d = rng.normal(size=(_DIRECTIONS, 3))
                d /= np.linalg.norm(d, axis=1)[:, None]
                pts.append(positions[atom] + math.sqrt(t / zeta) * d)
    return np.ascontiguousarray(np.concatenate(pts), dtype=np.float64)


def _former_cut_beyond(aos, r):
    """(num_ao, n) mask: point beyond the shell-relative cut pz_min * r^2 > LEPSBAS of that AO."""
    positions = np.asarray(aos.structure_data.positions, dtype=np.float64)
    exps = np.asarray(aos.exponents, dtype=np.float64)
    orbital_indices = np.asarray(aos.orbital_indices)
    nucleus = np.asarray(aos.nucleus_index)
    mask = np.zeros((aos.num_ao, r.shape[0]), dtype=bool)
    for ao in range(aos.num_ao):
        rel = r - positions[nucleus[ao]]
        r2 = np.einsum("ij,ij->i", rel, rel)
        mask[ao] = float(np.min(exps[orbital_indices == ao])) * r2 > _LEPSBAS
    return mask


@pytest.mark.light
@pytest.mark.parametrize(("trexio_file", "kind"), _INPUTS)
def test_fast_value_screens_drop_only_negligible_absolute_ao_values(trexio_file, kind):
    """Screened FAST AO values equal the exact XLA AO values; dropped ones are below 1e-12."""
    (_, aos, *_rest) = read_trexio_file(trexio_file=str(_TREXIO_DIR / trexio_file), store_tuple=True)
    assert isinstance(aos, AOs_sphe_gto_data if kind == "sphe" else AOs_cart_gto_data)
    r = _points_around_nuclei(aos)
    ref = np.asarray(compute_AOs(aos, r), dtype=np.float64)
    assert ref.shape == (aos.num_ao, r.shape[0])
    assert np.all(np.isfinite(ref))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_VALUE, level="strict")

    # Non-vacuity: beyond the shell-relative LEPSBAS cut, exact AO values exceed
    # the bound, so a shell-relative screen drops values the bound check rejects.
    beyond = _former_cut_beyond(aos, r)
    worst_beyond = float(np.max(np.abs(ref[beyond])))
    assert worst_beyond > _SCREEN_BOUND * (1.0 + rtol), (
        f"{trexio_file}: AO values beyond the LEPSBAS cut stay below the bound ({worst_beyond:.3e})"
    )

    # The per-move route of an fp32 ao_eval zone exists only in mixed precision
    # (its arithmetic is the zone-local fp32 one); compare it in that mode.
    routes = _ROUTES[kind] + (("single_f32",) if get_dtype_np("ao_eval") == np.float32 else ())
    for route in routes:
        label = f"{trexio_file} {route}: FAST screened AO values vs exact XLA"
        got = np.asarray(jqmc_rs.compute_AOs_value_fast_cut(aos, r, route))
        kept = _kept_after_dropped_bound(label, got, ref)
        np.testing.assert_allclose(got[kept], ref[kept], atol=atol, rtol=rtol, err_msg=label)


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
    """Independent value of c N x^n exp(-zeta r^2), N^2 = (2 zeta / pi)^(3/2) (4 zeta)^n / (2n-1)!!."""
    n, z = _HIGH_L_POWER, _HIGH_L_ZETA
    log_double_factorial = math.lgamma(2 * n + 1) - n * math.log(2.0) - math.lgamma(n + 1)  # log (2n-1)!!
    log_norm = 0.5 * (1.5 * math.log(2.0 * z / math.pi) + n * math.log(4.0 * z) - log_double_factorial)
    r2 = np.einsum("ij,ij->i", r, r)
    x = r[:, 0]
    # Assembled in logs so N (~1e-49) and x^82 (~1e54) never meet as separate factors.
    phi = coef * np.exp(log_norm + n * np.log(np.abs(x)) - z * r2) * np.sign(x) ** n
    return phi[None, :]


@pytest.mark.light
def test_fast_value_screens_absolute_for_high_angular_momentum():
    """Screened FAST AO values of an accepted l=82 Cartesian AO equal its closed form far beyond the LEPSBAS cut."""
    aos = _build_high_l_aos()
    r, nominal_r2 = _high_l_points()
    coef = float(np.asarray(aos.coefficients, dtype=np.float64)[0])
    ref = _high_l_closed_form(r, coef)
    assert np.all(np.isfinite(ref))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_VALUE, level="strict")

    # Non-vacuity: at the LEPSBAS start r^2 = LEPSBAS the AO is below the bound,
    # beyond it the AO exceeds the bound.
    at_start = nominal_r2 == _LEPSBAS
    assert np.any(at_start)
    assert np.all(np.abs(ref[0, at_start]) < _SCREEN_BOUND)
    beyond = _HIGH_L_ZETA * nominal_r2 > _LEPSBAS
    assert float(np.max(np.abs(ref[0, beyond]))) > _SCREEN_BOUND * (1.0 + rtol)

    for route in _ROUTES["cart"]:
        label = f"l=82 cart {route}: FAST screened AO values vs closed form"
        got = np.asarray(jqmc_rs.compute_AOs_value_fast_cut(aos, r, route))
        kept = _kept_after_dropped_bound(label, got, ref)
        np.testing.assert_allclose(got[kept], ref[kept], atol=atol, rtol=rtol, err_msg=label)
