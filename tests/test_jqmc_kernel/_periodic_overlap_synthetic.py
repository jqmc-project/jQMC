"""Committed synthetic 19-atom /401-AO slab-size periodic overlap fixture.

This is a reproducible cc-pVTZ-size contracted s/p/d/f workload of 19 atoms in
a 10.914 x 10.914 x 47.243 Bohr slab cell, not a claim of a particular basis
set; contraction/image counts are reported by the benchmark.
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

import numpy as np
from jqmc.atomic_orbital import AOs_cart_gto_data
from tests.conftest import make_pbc_structure


def make_slab_size_cartesian_basis(*, exponent_scale=1.0):
    cell = np.diag([10.914, 10.914, 47.243])
    positions, symbols, numbers = [], [], []
    for species, number, offset in [("Li", 3, 0.25), ("H", 1, 0.0)]:
        for z in [2.0, 5.0]:
            for x in [offset, offset + 0.5]:
                for y in [offset, offset + 0.5]:
                    positions.append([x * cell[0, 0], y * cell[1, 1], z])
                    symbols.append(species)
                    numbers.append(number)
    positions += [[5.457, 5.457, 12.0], [6.90, 5.457, 13.10], [4.01, 5.457, 13.10]]
    symbols += ["O", "H", "H"]
    numbers += [8, 1, 1]
    structure = make_pbc_structure(
        cell,
        positions=np.array(positions),
        atomic_numbers=tuple(numbers),
        element_symbols=tuple(symbols),
        atomic_labels=tuple(f"{s}{i}" for i, s in enumerate(symbols)),
    )
    # Full Cartesian shells: Li2s3p1d1f=27, H3s2p1d=15,
    # O4s3p2d1f=35, hence 8*27+10*15+35=401 AOs.
    shells = {
        "Li": [
            (0, [6.5, 1.3, 0.30], [0.05, 0.30, 0.70]),
            (0, [0.06], [1.0]),
            (1, [1.5, 0.4], [0.25, 0.75]),
            (1, [0.12], [1.0]),
            (1, [0.04], [1.0]),
            (2, [0.2], [1.0]),
            (3, [0.16], [1.0]),
        ],
        "H": [
            (0, [3.4, 0.62, 0.17], [0.08, 0.35, 0.70]),
            (0, [0.08], [1.0]),
            (0, [0.025], [1.0]),
            (1, [0.75], [1.0]),
            (1, [0.17], [1.0]),
            (2, [0.5], [1.0]),
        ],
        "O": [
            (0, [120.0, 20.0, 4.0, 0.9], [0.02, 0.12, 0.40, 0.60]),
            (0, [0.4], [1.0]),
            (0, [0.15], [1.0]),
            (0, [0.04], [1.0]),
            (1, [7.0, 1.5, 0.35], [0.10, 0.40, 0.65]),
            (1, [0.15], [1.0]),
            (1, [0.05], [1.0]),
            (2, [1.2], [1.0]),
            (2, [0.3], [1.0]),
            (3, [0.7], [1.0]),
        ],
    }
    centers, angular, px, py, pz, orbital_indices, exponents, coefficients = [], [], [], [], [], [], [], []
    for center, symbol in enumerate(symbols):
        for ell, radial_exponents, radial_coefficients in shells[symbol]:
            for nx in range(ell, -1, -1):
                for ny in range(ell - nx, -1, -1):
                    index = len(centers)
                    centers.append(center)
                    angular.append(ell)
                    px.append(nx)
                    py.append(ny)
                    pz.append(ell - nx - ny)
                    orbital_indices.extend([index] * len(radial_exponents))
                    exponents.extend(radial_exponents)
                    coefficients.extend(radial_coefficients)
    assert len(centers) == 401 and len(positions) == 19
    return AOs_cart_gto_data._from_full_basis(
        structure_data=structure,
        nucleus_index=tuple(centers),
        num_ao=len(centers),
        num_ao_prim=len(exponents),
        orbital_indices=tuple(orbital_indices),
        exponents=np.array(exponents) * exponent_scale,
        coefficients=np.array(coefficients),
        angular_momentums=tuple(angular),
        polynominal_order_x=tuple(px),
        polynominal_order_y=tuple(py),
        polynominal_order_z=tuple(pz),
    )
