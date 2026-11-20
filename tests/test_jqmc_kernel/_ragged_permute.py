"""Shared AO-permutation helpers for the non-shell-major (ragged) basis tests.

A permutation of the stored AO ordering yields a PHYSICALLY IDENTICAL basis: every
per-AO computation (values / grad / laplacian / analytic AO derivatives) must permute
its AO axis by exactly the same permutation and change no numbers. The helpers here
build such permutations -- including ones that INTERLEAVE same-``l`` shells so the
``2l+1`` (or Cartesian ``(l+1)(l+2)/2``) components of one shell become non-contiguous,
mirroring a QE-converted basis -- and apply them to any GTO / pGTO AO leaf
(molecular or periodic, spherical or Cartesian).

NOT collected as a test module (leading underscore, no ``test_`` prefix).
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

from collections import OrderedDict

import numpy as np

from jqmc.atomic_orbital import (
    AOs_cart_gto_data,
    AOs_cart_pgto_data,
    AOs_sphe_gto_data,
    AOs_sphe_pgto_data,
    _apply_image_expansion,
    _detile_ragged_to_molecular,
)


def _is_cart(mol) -> bool:
    return isinstance(mol, AOs_cart_gto_data)


def _shell_groups(mol) -> "OrderedDict[tuple, list[list[int]]]":
    """Group home AOs into shells by radial CONTENT, then bucket shells by ``(atom, l)``."""
    nuc = np.asarray(mol.nucleus_index)
    ang = np.asarray(mol.angular_momentums)
    orb = np.asarray(mol.orbital_indices)
    exps = np.asarray(mol.exponents)
    coef = np.asarray(mol.coefficients)
    prims = [np.nonzero(orb == a)[0] for a in range(mol.num_ao)]

    def sig(a):
        return (
            int(nuc[a]),
            int(ang[a]),
            tuple(np.round(exps[prims[a]], 8).tolist()),
            tuple(np.round(coef[prims[a]], 8).tolist()),
        )

    shells: "OrderedDict[tuple, list[int]]" = OrderedDict()
    for a in range(mol.num_ao):
        shells.setdefault(sig(a), []).append(a)
    byal: "OrderedDict[tuple, list[list[int]]]" = OrderedDict()
    for s, aos in shells.items():
        byal.setdefault((s[0], s[1]), []).append(aos)
    return byal


def interleave_perm(mol) -> list[int]:
    """new->old home-AO permutation that INTERLEAVES same-``(atom, l)`` shells component-major
    (``[c0_A, c0_B, c1_A, c1_B, ...]``), so each shell's components are non-contiguous."""
    perm: list[int] = []
    for _key, shlist in _shell_groups(mol).items():
        if len(shlist) == 1:
            perm.extend(shlist[0])
        else:
            width = len(shlist[0])
            for ci in range(width):
                for aos in shlist:
                    perm.append(aos[ci])
    return perm


def has_interleavable_shell(mol) -> bool:
    """True iff some ``(atom, l)`` has >1 shell (so interleaving breaks contiguity)."""
    return any(len(shlist) > 1 for shlist in _shell_groups(mol).values())


def random_perm(n: int, seed: int) -> list[int]:
    return list(np.random.default_rng(seed).permutation(n))


def _reorder_mol_gto(mol, perm: list[int]):
    """Reorder molecular GTO home AOs by ``perm`` (new[b] = old[perm[b]]); cart or sphe."""
    cart = _is_cart(mol)
    old_nuc = np.asarray(mol.nucleus_index)
    old_ang = np.asarray(mol.angular_momentums)
    old_orb = np.asarray(mol.orbital_indices)
    old_exp = np.asarray(mol.exponents)
    old_coef = np.asarray(mol.coefficients)
    if cart:
        opx, opy, opz = (
            np.asarray(mol.polynominal_order_x),
            np.asarray(mol.polynominal_order_y),
            np.asarray(mol.polynominal_order_z),
        )
    else:
        omag = np.asarray(mol.magnetic_quantum_numbers)
    prims_of = [np.nonzero(old_orb == a)[0] for a in range(mol.num_ao)]
    new_orb: list[int] = []
    new_exp: list[float] = []
    new_coef: list[float] = []
    new_nuc: list[int] = []
    new_ang: list[int] = []
    px: list[int] = []
    py: list[int] = []
    pz: list[int] = []
    mag: list[int] = []
    for b, a in enumerate(perm):
        pr = prims_of[a]
        new_orb += [b] * len(pr)
        new_exp += list(old_exp[pr])
        new_coef += list(old_coef[pr])
        new_nuc.append(int(old_nuc[a]))
        new_ang.append(int(old_ang[a]))
        if cart:
            px.append(int(opx[a]))
            py.append(int(opy[a]))
            pz.append(int(opz[a]))
        else:
            mag.append(int(omag[a]))
    kw = dict(
        structure_data=mol.structure_data,
        nucleus_index=tuple(new_nuc),
        num_ao=mol.num_ao,
        num_ao_prim=len(new_orb),
        orbital_indices=tuple(new_orb),
        exponents=np.asarray(new_exp, np.float64),
        coefficients=np.asarray(new_coef, np.float64),
        angular_momentums=tuple(new_ang),
    )
    if getattr(mol, "k_point", None) is not None:
        kw["k_point"] = mol.k_point
    if cart:
        return AOs_cart_gto_data._from_full_basis(
            **kw, polynominal_order_x=tuple(px), polynominal_order_y=tuple(py), polynominal_order_z=tuple(pz)
        )
    return AOs_sphe_gto_data._from_full_basis(**kw, magnetic_quantum_numbers=tuple(mag))


def molecular_view(aos):
    """Molecular (home) GTO base of any AO leaf: detile if ragged pGTO, else self."""
    if isinstance(aos, (AOs_cart_pgto_data, AOs_sphe_pgto_data)) and aos._ragged_buckets:
        return _detile_ragged_to_molecular(aos)
    return aos


def permute_aos(aos, perm: list[int]):
    """Return a physically identical AO leaf with home AOs reordered by ``perm``. For a periodic
    ragged pGTO leaf: detile -> permute molecular base -> re-expand (rebuilds the ragged layout)."""
    is_pgto = isinstance(aos, (AOs_cart_pgto_data, AOs_sphe_pgto_data)) and bool(aos._ragged_buckets)
    mol = _detile_ragged_to_molecular(aos) if is_pgto else aos
    new_mol = _reorder_mol_gto(mol, perm)
    if is_pgto:
        return _apply_image_expansion(new_mol, aos.structure_data, float(aos._image_truncation_tol))
    return new_mol
