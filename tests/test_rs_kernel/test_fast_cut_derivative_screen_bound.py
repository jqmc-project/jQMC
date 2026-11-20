"""FAST-mode AO derivative screen keeps every dropped derivative negligible in absolute AO units (issue #87).

The production FAST energy chunk (native MCMC / GFMC_t, the FAST kinetic and the
J3 grad/lap) evaluates AO gradients and Laplacians through screened kernels that
zero a shell's derivative rows for an electron far from the shell. A cut
``pz_min * r^2 > LEPSBAS`` (``LEPSBAS = 20.72``) is relative to the shell's own
peak factor ``|norm * coef|``. For an uncontracted tight primitive that factor
grows as ``zeta^(3/4 + l/2)``, so the dropped Laplacian at the cut radius grows as
``~zeta^1.75`` and biases the FAST local energy upward (+1e-5 Ha mean on Ar
unc-cc-pVDZ, +1e-4 on Kr; issue #87).

This module pins the three screened kernels behind the six screen sites, for
Cartesian and spherical shells:

* ``batch``: ``compute_aos_grad_lap_fast_cut`` (FAST chunk kinetic, J3);
* ``single``: ``compute_aos_single_value_grad_lap_fast_cut_into`` (GFMC
  derivative-cache column update);
* ``folded``: ``compute_aos_single_value_grad_lap_fast_cut_folded_into``
  (image-folded column update; identity fold on this molecular leaf),

through the TEST-ONLY binding ``jqmc_rs.compute_AOs_grad_lap_fast_cut`` against
the exact (unscreened) XLA ``compute_AOs_grad`` / ``compute_AOs_laplacian`` at the
same points: an independent implementation of the same AO derivatives. The basis
is built in-test: uncontracted s/p/d/f/g primitives with exponents up to 1e6 and
mixed-sign contracted s and p shells whose tight and diffuse primitives share one
shell. Points sit, for every exponent, on radii with ``zeta * r^2`` from 10 to 60,
so each tight shell is sampled across the band where a shell-relative screen
switches off its derivatives.

Tolerance: the strict XLA-Rust consistency pair of the ``ao_grad_lap`` zone. That
zone is fp64 in both precision modes, and the stored AO exponents and
coefficients are read by both backends from the same leaf, so full and mixed
share the fp64 pair. A dropped derivative passes only if it is below
``atol + rtol |exact|``; the screen keeps it below ``exp(-LEPSBAS) ~= 1e-9``.
Non-vacuity: for every tight shell, the exact derivatives at points beyond the
shell-relative cut radius exceed that envelope, so a shell-relative screen
would have to drop elements that this comparison rejects.
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

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.atomic_orbital import AOs_cart_gto_data, AOs_sphe_gto_data, compute_AOs_grad, compute_AOs_laplacian
from jqmc.structure import Structure_data

_ZONES_GRAD_LAP = ("ao_grad_lap",)
_ROUTES = ("batch", "single", "folded")
_CENTER = (0.3, -0.2, 0.1)
# (l, exponents, contraction coefficients). Uncontracted tight primitives of every
# l up to 4 plus mixed-sign contracted s and p shells (one tight and one diffuse
# primitive in the same shell, so the shell's pz_min is diffuse while its peak
# factor is set by the tight primitive).
_SHELLS = (
    (0, (1.0e6,), (1.0,)),
    (0, (1.0e5,), (1.0,)),
    (0, (1.0e4,), (1.0,)),
    (0, (1.0e3,), (1.0,)),
    (0, (1.0,), (1.0,)),
    (0, (5.0e5, 5.0e4, 2.0e3, 50.0), (0.3, -0.7, 1.2, -0.4)),
    (1, (1.0e5,), (1.0,)),
    (1, (1.0e3,), (1.0,)),
    (1, (1.0,), (1.0,)),
    (1, (2.0e4, 600.0), (1.0, -1.5)),
    (2, (1.0e4,), (1.0,)),
    (2, (100.0,), (1.0,)),
    (3, (1.0e3,), (1.0,)),
    (4, (300.0,), (1.0,)),
)
# zeta * r^2 band sampled for every exponent; it brackets the
# shell-relative derivative cut zeta_min * r^2 = 20.72 (LEPSBAS in
# jqmc/rs/src/atomic_orbital.rs).
_ZETA_R2 = np.geomspace(10.0, 60.0, 11)
_FORMER_CUT_ZETA_R2 = 20.72
_DIRECTIONS = 3
# Shells whose most diffuse primitive is at least this tight are the "tight
# shells" of the non-vacuity check.
_TIGHT_PZ_MIN = 1.0e3


def _cart_components(l):
    return [(nx, ny, l - nx - ny) for nx in range(l, -1, -1) for ny in range(l - nx, -1, -1)]


def _build_aos(kind):
    structure = Structure_data(
        pbc_flag=False,
        positions=np.asarray([_CENTER], dtype=np.float64),
        atomic_numbers=(18,),
        element_symbols=("Ar",),
        atomic_labels=("Ar",),
    )
    nucleus, ang, mag, px, py, pz = [], [], [], [], [], []
    orbital_indices, exponents, coefficients, ao_shell = [], [], [], []
    for s, (l, zetas, coefs) in enumerate(_SHELLS):
        comps = _cart_components(l) if kind == "cart" else list(range(-l, l + 1))
        for comp in comps:
            ao = len(ang)
            nucleus.append(0)
            ang.append(l)
            ao_shell.append(s)
            if kind == "cart":
                px.append(comp[0])
                py.append(comp[1])
                pz.append(comp[2])
            else:
                mag.append(comp)
            for zeta, coef in zip(zetas, coefs, strict=True):
                orbital_indices.append(ao)
                exponents.append(zeta)
                coefficients.append(coef)
    common = dict(
        structure_data=structure,
        nucleus_index=nucleus,
        num_ao=len(ang),
        num_ao_prim=len(exponents),
        orbital_indices=orbital_indices,
        exponents=np.asarray(exponents, dtype=np.float64),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        angular_momentums=ang,
    )
    if kind == "cart":
        aos = AOs_cart_gto_data.build(polynominal_order_x=px, polynominal_order_y=py, polynominal_order_z=pz, **common)
    else:
        aos = AOs_sphe_gto_data.build(magnetic_quantum_numbers=mag, **common)
    return aos, np.asarray(ao_shell)


def _points():
    """Per distinct exponent: radii sqrt(t / zeta) for t in _ZETA_R2, seeded random directions."""
    rng = np.random.default_rng(87)
    zetas = sorted({z for _, shell_z, _ in _SHELLS for z in shell_z})
    points = []
    for zeta in zetas:
        for t in _ZETA_R2:
            d = rng.normal(size=(_DIRECTIONS, 3))
            d /= np.linalg.norm(d, axis=1)[:, None]
            points.append(np.asarray(_CENTER) + np.sqrt(t / zeta) * d)
    return np.ascontiguousarray(np.concatenate(points), dtype=np.float64)


@pytest.mark.light
@pytest.mark.parametrize("kind", ["cart", "sphe"])
def test_fast_cut_ao_derivative_screen_drops_only_negligible_absolute_terms(kind):
    """Screened FAST AO grad/lap equal the exact XLA derivatives for tight uncontracted and contracted shells."""
    aos, ao_shell = _build_aos(kind)
    r = _points()
    exact_g = [np.asarray(g, dtype=np.float64) for g in compute_AOs_grad(aos, r)]
    exact_l = np.asarray(compute_AOs_laplacian(aos, r), dtype=np.float64)
    exact = (*exact_g, exact_l)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD_LAP, level="strict")
    for name, ref in zip(("gx", "gy", "gz", "lap"), exact, strict=True):
        assert ref.shape == (len(ao_shell), len(r)), name
        assert np.all(np.isfinite(ref)), f"{kind}: exact {name} not finite"

    # Non-vacuity: every tight shell has exact derivatives above the comparison
    # envelope at points beyond its shell-relative cut radius.
    rel = r - np.asarray(_CENTER)
    r2 = np.einsum("ij,ij->i", rel, rel)
    for s, (_, zetas, _) in enumerate(_SHELLS):
        pz_min = min(zetas)
        if pz_min < _TIGHT_PZ_MIN:
            continue
        beyond = pz_min * r2 > _FORMER_CUT_ZETA_R2
        rows = ao_shell == s
        assert np.any(beyond), f"{kind}: shell {s} has no sample beyond its shell-relative cut"
        excess = max(
            float(np.max(np.abs(ref[np.ix_(rows, beyond)]) / (atol + rtol * np.abs(ref[np.ix_(rows, beyond)]))))
            for ref in exact
        )
        assert excess > 1.0, f"{kind}: shell {s} derivatives beyond its shell-relative cut stay inside the envelope ({excess})"

    for route in _ROUTES:
        got = [np.asarray(x) for x in jqmc_rs.compute_AOs_grad_lap_fast_cut(aos, r, route)]
        for name, val, ref in zip(("gx", "gy", "gz", "lap"), got, exact, strict=True):
            label = f"{kind} {route}: {name} FAST screened vs exact XLA"
            assert val.shape == ref.shape, label
            assert np.all(np.isfinite(val)), f"{label}: non-finite"
            np.testing.assert_allclose(val, ref, atol=atol, rtol=rtol, err_msg=label)
