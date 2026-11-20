"""Basis-set listing of ``hamiltonian show-info``.

The listing printed by ``_get_info`` must show each contracted shell exactly
once, on its home atom, in the stored (TREXIO) AO order, with one coefficient
per primitive that is uniform over the shell's angular components:

* Cartesian components carry a per-component normalization factor in the
  stored coefficients (``x^2`` and ``xy`` of one d shell differ by sqrt(3));
  the listing must not split such a shell into one "shell" per distinct
  coefficient (two for d, three for f).
* Periodic leaves are evaluated on a bucket-major ragged image layout; the
  listing must not follow that layout, or the shell order changes from atom
  to atom.
* Two genuinely distinct but identical-looking shells on one atom must not be
  merged.

The listing is display-only: building it does not change AO values.
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

import re

import numpy as np
import pytest
import scipy.special

from jqmc.atomic_orbital import AOs_cart_gto_data, AOs_cart_pgto_data, AOs_sphe_gto_data, compute_AOs
from jqmc.structure import Structure_data

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_L_LABEL = {0: "s", 1: "p", 2: "d", 3: "f"}


def _cart_components(l):
    from itertools import combinations_with_replacement

    orders = ["".join(c) for c in combinations_with_replacement("xyz", l)]
    return [(o.count("x"), o.count("y"), o.count("z")) for o in orders]


def _component_factor(nx, ny, nz):
    fact = scipy.special.factorial
    return fact(nx) * fact(ny) * fact(nz) / (fact(2 * nx) * fact(2 * ny) * fact(2 * nz))


def _cart_kwargs(structure, shells):
    """Build Cartesian GTO kwargs the way the TREXIO loader stores them.

    ``shells`` is a list of ``(atom, l, exponents, coefficients)``. Every
    component stores ``coefficient / sqrt(f(nx, ny, nz))`` up to a factor
    common to the shell; here the common factor is chosen so that the
    ``x^l`` component stores exactly ``coefficient``.
    """
    kw = dict(nucleus_index=[], angular_momentums=[], orbital_indices=[], exponents=[], coefficients=[])
    px, py, pz = [], [], []
    ao = 0
    for atom, l, exps, coefs in shells:
        for nx, ny, nz in _cart_components(l):
            scale = np.sqrt(_component_factor(l, 0, 0) / _component_factor(nx, ny, nz))
            kw["nucleus_index"].append(atom)
            kw["angular_momentums"].append(l)
            px.append(nx)
            py.append(ny)
            pz.append(nz)
            for z, c in zip(exps, coefs, strict=True):
                kw["orbital_indices"].append(ao)
                kw["exponents"].append(z)
                kw["coefficients"].append(c * scale)
            ao += 1
    return dict(
        structure_data=structure,
        nucleus_index=tuple(kw["nucleus_index"]),
        num_ao=ao,
        num_ao_prim=len(kw["orbital_indices"]),
        orbital_indices=tuple(kw["orbital_indices"]),
        exponents=np.asarray(kw["exponents"], dtype=np.float64),
        coefficients=np.asarray(kw["coefficients"], dtype=np.float64),
        angular_momentums=tuple(kw["angular_momentums"]),
        polynominal_order_x=tuple(px),
        polynominal_order_y=tuple(py),
        polynominal_order_z=tuple(pz),
    )


def _sphe_kwargs(structure, shells):
    kw = dict(
        nucleus_index=[], angular_momentums=[], magnetic_quantum_numbers=[], orbital_indices=[], exponents=[], coefficients=[]
    )
    ao = 0
    for atom, l, exps, coefs in shells:
        for m in range(-l, l + 1):
            kw["nucleus_index"].append(atom)
            kw["angular_momentums"].append(l)
            kw["magnetic_quantum_numbers"].append(m)
            for z, c in zip(exps, coefs, strict=True):
                kw["orbital_indices"].append(ao)
                kw["exponents"].append(z)
                kw["coefficients"].append(c)
            ao += 1
    return dict(
        structure_data=structure,
        nucleus_index=tuple(kw["nucleus_index"]),
        num_ao=ao,
        num_ao_prim=len(kw["orbital_indices"]),
        orbital_indices=tuple(kw["orbital_indices"]),
        exponents=np.asarray(kw["exponents"], dtype=np.float64),
        coefficients=np.asarray(kw["coefficients"], dtype=np.float64),
        angular_momentums=tuple(kw["angular_momentums"]),
        magnetic_quantum_numbers=tuple(kw["magnetic_quantum_numbers"]),
    )


def _expected_listing(labels, shells):
    lines = []
    for atom, label in enumerate(labels):
        lines += ["  " + "-" * 36, f"  **basis set for atom index {atom + 1}: {label}**", "  " + "-" * 36]
        for shell_atom, l, exps, coefs in shells:
            if shell_atom != atom:
                continue
            lines.append(f"  {label} {_L_LABEL[l]}")
            lines += [f"    {z:.6f} {c:.7f}" for z, c in zip(exps, coefs, strict=True)]
    return lines


def _listing(info_lines):
    start = next(i for i, line in enumerate(info_lines) if "**basis set for atom index" in line) - 1
    return info_lines[start:]


# s (contracted), s, p, d, f, and a second d shell identical to the first one:
# it is a distinct shell and must be listed again, not merged.
_SHELLS_O = [
    (0, 0, (5.0, 1.2, 0.3), (0.2, 0.5, 0.4)),
    (0, 0, (0.25,), (1.0,)),
    (0, 1, (2.0, 0.4), (0.6, 0.5)),
    (0, 2, (1.1,), (1.0,)),
    (0, 3, (0.9,), (1.0,)),
    (0, 2, (1.1,), (1.0,)),
]
_SHELLS_H = [
    (1, 0, (3.0, 0.5), (0.3, 0.8)),
    (1, 1, (0.7,), (1.0,)),
]


def _molecule():
    return Structure_data(
        pbc_flag=False,
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]]),
        atomic_numbers=(8, 1),
        element_symbols=("O", "H"),
        atomic_labels=("O", "H"),
    )


def test_molecular_cartesian_listing_has_one_uniform_entry_per_shell():
    """s/p/d/f Cartesian shells print once each, in AO order, with the x^l coefficient."""
    shells = _SHELLS_O + _SHELLS_H
    aos = AOs_cart_gto_data._from_full_basis(**_cart_kwargs(_molecule(), shells))
    assert _listing(aos._get_info()) == _expected_listing(("O", "H"), shells)


def test_molecular_spherical_listing_has_one_entry_per_shell():
    """Real-spherical shells print once each, in AO order; identical adjacent shells are not merged."""
    shells = _SHELLS_O + _SHELLS_H
    aos = AOs_sphe_gto_data._from_full_basis(**_sphe_kwargs(_molecule(), shells))
    assert _listing(aos._get_info()) == _expected_listing(("O", "H"), shells)


def _periodic_structure():
    # Two atoms at inequivalent cell positions, so the per-shell lattice-image
    # sets (and hence the ragged bucket order) differ between the atoms.
    cell = np.diag([6.0, 6.5, 7.0])
    return Structure_data(
        pbc_flag=True,
        vec_a=tuple(cell[0]),
        vec_b=tuple(cell[1]),
        vec_c=tuple(cell[2]),
        positions=np.array([[0.05, 0.05, 0.05], [3.0, 3.25, 3.5]]),
        atomic_numbers=(8, 8),
        element_symbols=("O", "O"),
        atomic_labels=("O", "O"),
    )


_SHELLS_PBC = [
    (atom, l, exps, coefs)
    for atom in (0, 1)
    for l, exps, coefs in (
        (0, (8.0, 1.0, 0.2), (0.1, 0.5, 0.5)),
        (0, (0.15,), (1.0,)),
        (1, (3.0, 0.3), (0.4, 0.7)),
        (1, (0.12,), (1.0,)),
        (2, (1.5,), (1.0,)),
    )
]


def _periodic_leaf():
    return AOs_cart_pgto_data.build(**_cart_kwargs(_periodic_structure(), _SHELLS_PBC))


def test_periodic_listing_follows_home_ao_order_on_every_atom():
    """Image-tiled shells are listed once on their home atom, in the stored AO order."""
    aos = _periodic_leaf()
    # Non-vacuity: in the bucket-major kernel layout the first occurrence of
    # each home AO follows a different order on the two atoms, and neither is
    # the stored AO order.
    first_kernel = {}
    for kernel_idx, home in enumerate(aos._ragged_kernel_to_home):
        first_kernel.setdefault(int(home), kernel_idx)
    n = aos.num_ao // 2
    layout_orders = [[int(i) for i in np.argsort([first_kernel[h] for h in range(a * n, (a + 1) * n)])] for a in (0, 1)]
    assert layout_orders[0] != layout_orders[1]
    assert all(order != list(range(n)) for order in layout_orders)
    assert _listing(aos._get_info()) == _expected_listing(("O", "O"), _SHELLS_PBC)


def test_listing_is_display_only():
    """Building the listing leaves the AO values bit-identical, for molecular and periodic leaves."""
    rng = np.random.default_rng(20261007)
    r = rng.uniform(-1.0, 7.0, size=(7, 3))
    for build in (
        lambda: AOs_cart_gto_data._from_full_basis(**_cart_kwargs(_molecule(), _SHELLS_O + _SHELLS_H)),
        lambda: AOs_sphe_gto_data._from_full_basis(**_sphe_kwargs(_molecule(), _SHELLS_O + _SHELLS_H)),
        _periodic_leaf,
    ):
        aos = build()
        before = np.asarray(compute_AOs(aos, r))
        lines = aos._get_info()
        after = np.asarray(compute_AOs(aos, r))
        fresh = np.asarray(compute_AOs(build(), r))
        assert lines
        assert np.all(np.isfinite(before))
        np.testing.assert_array_equal(after, before)
        np.testing.assert_array_equal(fresh, before)
        assert not any(re.search(r"nan|inf", line) for line in _listing(lines))
