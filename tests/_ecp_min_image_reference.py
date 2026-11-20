"""Brute-force minimum-image reference for periodic ECP checks.

The production kernels (XLA ``_minimum_image_displacement`` and Rust
``ewald::min_image``) reduce a displacement to its Wigner-Seitz image. The
reference here does not reduce anything: it takes the shortest of the plain
lattice translations ``d + n @ cell`` over the box ``n in [-N, N]^3``. It
assumes nothing about where the electrons sit, so it also holds for
electrons several cells away from the home cell.
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

import numpy as np

from jqmc.coulomb_potential import Coulomb_potential_data
from jqmc.structure import Structure_data

#: Half-width of the brute-force image box. The test geometries put an
#: electron at most a few cells away along each axis, so +-6 cells covers its
#: nearest image on the cubic, fcc-like and oblique cells used here.
IMAGE_BOX = 6

#: Three cells: orthogonal, a skewed (fcc-like) basis and a strongly oblique basis.
CELLS = {
    "cubic": np.diag([6.0, 6.0, 6.0]),
    "skew": np.array([[5.0, 0.0, 0.0], [2.5, 4.330, 0.0], [2.5, 1.443, 4.082]]),
    "oblique": np.array([[6.0, 0.0, 0.0], [5.5, 1.5, 0.0], [0.3, 0.2, 5.0]]),
}


def _images(cell: np.ndarray) -> np.ndarray:
    n = np.array(list(itertools.product(range(-IMAGE_BOX, IMAGE_BOX + 1), repeat=3)), dtype=np.float64)
    return n @ np.asarray(cell, dtype=np.float64)


def min_image_distances(positions, cell, r_carts) -> np.ndarray:
    """``d[e, I] = min_n |R_I - r_e + n @ cell|`` by exhaustive image search."""
    images = _images(cell)
    pos = np.asarray(positions, dtype=np.float64)
    r = np.asarray(r_carts, dtype=np.float64)
    diff = pos[None, :, None, :] - r[:, None, None, :] + images[None, None, :, :]
    return np.sqrt(np.min(np.sum(diff**2, axis=-1), axis=-1))


def ecp_local_reference_per_electron(cp: Coulomb_potential_data, r_carts) -> np.ndarray:
    """Local-channel ECP energy of each electron from the brute-force image distance.

    The local channel is the term with ``ang_mom == max_ang_mom_plus_1`` of its
    nucleus; each term contributes ``c d^(p-2) exp(-a d^2)``.
    """
    sd = cp.structure_data
    d = min_image_distances(sd.positions, sd.cell, r_carts)
    out = np.zeros(d.shape[0])
    for i_atom, l, c, a, p in zip(cp.nucleus_index, cp.ang_moms, cp.coefficients, cp.exponents, cp.powers):
        if l == cp.max_ang_mom_plus_1[i_atom]:
            out += c * d[:, i_atom] ** (p - 2) * np.exp(-a * d[:, i_atom] ** 2)
    return out


def synthetic_h2_ecp(cellname: str) -> Coulomb_potential_data:
    """Two-nucleus periodic H2 with an s non-local and a p local ECP channel."""
    cell = CELLS[cellname]
    sd = Structure_data.build(
        pbc_flag=True,
        positions=np.array([[0.3, 0.2, 0.1], [2.0, 1.5, 1.0]]),
        vec_a=tuple(cell[0]),
        vec_b=tuple(cell[1]),
        vec_c=tuple(cell[2]),
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H1", "H2"),
    )
    cp = Coulomb_potential_data(
        structure_data=sd,
        ecp_flag=True,
        z_cores=(0.0, 0.0),
        max_ang_mom_plus_1=(1, 1),
        num_ecps=4,
        nucleus_index=(0, 0, 1, 1),
        ang_moms=(0, 1, 0, 1),
        coefficients=(1.0, 0.5, 0.8, 0.3),
        exponents=(0.7, 0.9, 0.6, 1.1),
        powers=(2, 1, 2, 1),
    )
    cp.sanity_check()
    return cp


def far_electrons(cellname: str, n: int, seed: int = 1) -> np.ndarray:
    """``n`` electrons spread over several cells (most of them outside the home cell)."""
    cell = CELLS[cellname]
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, (n, 3)) @ cell * 3.0 - 1.2 * cell.sum(0)


def local_ecp_on(structure_data: Structure_data, local_scale: float = 1.0) -> Coulomb_potential_data:
    """An ECP on every nucleus of ``structure_data``: an s non-local and a p local term.

    ``local_scale`` multiplies the local coefficients only, so ``local_scale=0``
    is the same potential without its local channel; the two differ in the
    local ECP energy alone.
    """
    natom = len(structure_data.atomic_numbers)
    nucleus_index = tuple(i for i in range(natom) for _ in range(2))
    ang_moms = tuple(l for _ in range(natom) for l in (0, 1))
    cp = Coulomb_potential_data(
        structure_data=structure_data,
        ecp_flag=True,
        z_cores=tuple(0.0 for _ in range(natom)),
        max_ang_mom_plus_1=tuple(1 for _ in range(natom)),
        num_ecps=2 * natom,
        nucleus_index=nucleus_index,
        ang_moms=ang_moms,
        coefficients=tuple(
            (0.4 + 0.1 * i) if l == 0 else local_scale * (0.5 + 0.2 * i) for i, l in zip(nucleus_index, ang_moms)
        ),
        exponents=tuple(0.6 if l == 0 else 0.9 for l in ang_moms),
        powers=tuple(2 if l == 0 else 1 for l in ang_moms),
    )
    cp.sanity_check()
    return cp


def single_nucleus_ecp_on(structure_data: Structure_data, active: int = 0) -> Coulomb_potential_data:
    """An ECP on every nucleus of ``structure_data`` whose terms vanish except on ``active``.

    Every nucleus carries an s non-local and a p local term, so the non-local
    nearest-nucleus filter chooses among all nuclei, while only ``active``
    contributes a non-zero energy.
    """
    natom = len(structure_data.atomic_numbers)
    nucleus_index, ang_moms, coefficients = [], [], []
    for i in range(natom):
        nucleus_index += [i, i]
        ang_moms += [0, 1]
        coefficients += [2.0, 0.5] if i == active else [0.0, 0.0]
    cp = Coulomb_potential_data(
        structure_data=structure_data,
        ecp_flag=True,
        z_cores=tuple(0.0 for _ in range(natom)),
        max_ang_mom_plus_1=tuple(1 for _ in range(natom)),
        num_ecps=2 * natom,
        nucleus_index=tuple(nucleus_index),
        ang_moms=tuple(ang_moms),
        coefficients=tuple(coefficients),
        exponents=tuple(0.6 if l == 0 else 0.9 for l in ang_moms),
        powers=tuple(2 if l == 0 else 1 for l in ang_moms),
    )
    cp.sanity_check()
    return cp
