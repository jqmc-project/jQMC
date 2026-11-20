"""Physical s/p-shell inputs and an independent Gaussian lattice-sum oracle."""

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

from itertools import product
from pathlib import Path

import numpy as np

from jqmc.atomic_orbital import AOs_cart_gto_data, AOs_sphe_gto_data, _apply_image_expansion
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data
from jqmc.molecular_orbital import MOs_data

AUDIT_FIXTURE = Path(__file__).resolve().parents[1] / "jqmc_hamiltonian_data" / "periodic_j3_basis_audit_initial.h5"
R_UP = np.asarray([[0.31, 0.43, 0.57], [1.12, -0.62, 0.91]])
R_DN = R_UP + np.asarray([0.13, 0.21, -0.17])


def physical_case(angular, wrapper):
    """Same actual supercell/PW determinant, with two small physical s/p shells."""
    ham = Hamiltonian_data.load_from_hdf5(str(AUDIT_FIXTURE))
    sd = ham.structure_data
    # Each of the first two sites owns a contracted s shell and one p shell.
    # The remaining physical sites still contribute to the determinant/ECP.
    orbital = []
    exponents = []
    coefficients = []
    for atom in range(2):
        for component in range(4):
            a = [0.4, 1.3] if component == 0 else [0.75]
            c = [0.7, -0.2] if component == 0 else [0.9]
            orbital.extend([atom * 4 + component] * len(a))
            exponents.extend(x * (1 + atom / 8) for x in a)
            coefficients.extend(x * (1 - atom / 8) for x in c)
    fields = dict(
        structure_data=sd,
        nucleus_index=(0, 0, 0, 0, 1, 1, 1, 1),
        num_ao=8,
        num_ao_prim=len(orbital),
        orbital_indices=tuple(orbital),
        exponents=np.asarray(exponents),
        coefficients=np.asarray(coefficients),
        angular_momentums=(0, 1, 1, 1, 0, 1, 1, 1),
    )
    if angular == "cart":
        home = AOs_cart_gto_data.build(
            **fields,
            polynominal_order_x=(0, 1, 0, 0, 0, 1, 0, 0),
            polynominal_order_y=(0, 0, 1, 0, 0, 0, 1, 0),
            polynominal_order_z=(0, 0, 0, 1, 0, 0, 0, 1),
        )
    else:
        home = AOs_sphe_gto_data.build(**fields, magnetic_quantum_numbers=(0, -1, 0, 1, 0, -1, 0, 1))
    leaf = _apply_image_expansion(home, sd)
    if wrapper == "mo":
        # A non-identity, non-orthogonal MO wrapper exercises its chain rule.
        matrix = np.eye(8) + np.arange(64).reshape(8, 8) / 256
        orb = MOs_data.build(aos_data=leaf, mo_coefficients=matrix)
    else:
        orb = leaf
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb)
    matrix = np.random.default_rng(41).normal(scale=0.02, size=j3.j_matrix.shape)
    matrix[:, :-1] = (matrix[:, :-1] + matrix[:, :-1].T) / 2
    j3 = j3.replace(j_matrix=matrix)
    jd = Jastrow_data(jastrow_three_body_data=j3)
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd)), home


def j3_leaf(ham):
    orb = ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data
    return orb.aos_data if isinstance(orb, MOs_data) else orb


def gaussian_lattice_reference(j3, angular):
    """Closed-form normalized s/p Gaussians, explicit images and electron pairs.

    No production AO kernel, image-list/radius helper, radial gather, autodiff,
    or J3 evaluator contributes to the reference. The fixed cube [-6,6]^3
    independently converges these alpha>=.28 shells in the actual Li cell.
    All physical parameters and off-nucleus positions are unscreened inputs.
    """
    leaf = j3.orb_data.aos_data if isinstance(j3.orb_data, MOs_data) else j3.orb_data
    alpha = np.asarray(leaf.radial.exponents)
    coeff = np.asarray(leaf.radial.coefficients)
    sd = leaf.structure_data
    images = np.asarray(list(product(range(-6, 7), repeat=3))) @ sd._cell_np
    points = np.concatenate([R_UP, R_DN])
    values = np.zeros((8, 4))
    d_exp = np.zeros((6, 8, 4))
    d_coeff = np.zeros_like(d_exp)
    for atom in range(2):
        dr = points[None] - (sd._positions_cart_np[atom] + images[:, None])
        r2 = np.sum(dr * dr, axis=-1)
        for component in range(4):
            ao = atom * 4 + component
            power = 0 if component == 0 else 1
            axis = component - 1 if angular == "cart" else (1, 2, 0)[component - 1]
            polynomial = np.ones_like(r2) if component == 0 else dr[..., axis]
            for primitive in [0, 1] if component == 0 else [2]:
                p = 3 * atom + primitive
                a, c = alpha[p], coeff[p]
                normalization = (2 * a / np.pi) ** 0.75 * (1 if power == 0 else 2 * np.sqrt(a))
                unit = normalization * polynomial * np.exp(-a * r2)
                values[ao] += c * np.sum(unit, axis=0)
                d_exp[p, ao] += c * np.sum(unit * ((0.75 + power / 2) / a - r2), axis=0)
                d_coeff[p, ao] += np.sum(unit, axis=0)
    if isinstance(j3.orb_data, MOs_data):
        C = np.asarray(j3.orb_data.mo_coefficients)
        values = C @ values
        d_exp = np.einsum("ij,pjk->pik", C, d_exp)
        d_coeff = np.einsum("ij,pjk->pik", C, d_coeff)
    matrix = np.asarray(j3.j_matrix)
    result = sum(matrix[:, -1] @ values[:, e] for e in range(4))
    derivatives = []
    for deriv in (d_exp, d_coeff):
        g = np.einsum("i,pij->p", matrix[:, -1], deriv)
        for i in range(4):
            for j in range(i + 1, 4):
                g += deriv[:, :, i] @ matrix[:, :-1] @ values[:, j]
                g += np.einsum("i,ij,pj->p", values[:, i], matrix[:, :-1], deriv[:, :, j])
        derivatives.append(g)
    for i in range(4):
        for j in range(i + 1, 4):
            result += values[:, i] @ matrix[:, :-1] @ values[:, j]
    return result, tuple(derivatives)
