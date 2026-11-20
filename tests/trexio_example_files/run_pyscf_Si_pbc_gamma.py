#!/usr/bin/env python
# coding: utf-8

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

# load python packages
import os, sys
import numpy as np

# load pyscf packages
from pyscf import gto, scf, mp, tools
from pyscf.pbc import gto as gto_pbc
from pyscf.pbc import dft as pbcdft
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.tools import super_cell
from pyscf.tools import trexio

# PBC
# Silicon, diamond structure (Fd-3m, origin choice 1), a = 5.381 Ang
# Source: COD entry 1526655 (Kitano et al., Phys. Rev. B 64, 045206 (2001))
# 8a Wyckoff (Si at 0,0,0) expanded by the CIF symmetry ops to 8 atoms / conventional cell.
a_lat = 5.381  # Angstrom
lattice_vectors = [
    [a_lat, 0.0, 0.0],
    [0.0, a_lat, 0.0],
    [0.0, 0.0, a_lat],
]
atom_cartesian = [
    ["Si", (0.00 * a_lat, 0.00 * a_lat, 0.00 * a_lat)],
    ["Si", (0.50 * a_lat, 0.50 * a_lat, 0.00 * a_lat)],
    ["Si", (0.50 * a_lat, 0.00 * a_lat, 0.50 * a_lat)],
    ["Si", (0.00 * a_lat, 0.50 * a_lat, 0.50 * a_lat)],
    ["Si", (0.25 * a_lat, 0.25 * a_lat, 0.25 * a_lat)],
    ["Si", (0.75 * a_lat, 0.75 * a_lat, 0.25 * a_lat)],
    ["Si", (0.75 * a_lat, 0.25 * a_lat, 0.75 * a_lat)],
    ["Si", (0.25 * a_lat, 0.75 * a_lat, 0.75 * a_lat)],
]
checkpoint_file = "silicon.chk"
trexio_file = "silicon.h5"
pyscf_output = "out_silicon_pyscf"
charge = 0
spin = 0
basis = {
    # basis set optimized for solids:
    # Correlation-Consistent Gaussian Basis Sets for Solids Made Simple, H.-Z. Ye and T. C. Berkelbach, J. Chem. Theory Comput., 18, 1595--1606 (2022). doi: 10.1021/acs.jctc.1c01245
    "Si": gto.basis.parse("""
    #BASIS SET: (5s,4p,2d,1f) -> [3s,3p,2d,1f] Si
    Si  S
    2.741335    4.650395e-02
    1.405628   -3.054975e-01
    0.273354    4.969654e-01
    Si  S
    0.123202    1.000000e+00
    Si  S
    0.055965    1.000000e+00
    Si  P
    1.616545   -3.536164e-02
    0.382436    3.057198e-01
    Si  P
    0.147904    1.000000e+00
    Si  P
    0.055817    1.000000e+00
    Si  D
    0.534081    1.000000e+00
    Si  D
    0.170283    1.000000e+00
    Si  F
    0.347183    1.000000e+00
""")
}
ecp = "ccecp"
twist_average = False
kpt = [0.00, 0.00, 0.00]
kpt_grid = [1, 1, 1]  # only consumed when twist_average = True (flip the flag above)
nx = 1
ny = 1
nz = 1

# construct cell
cell = gto_pbc.Cell()
cell.atom = atom_cartesian
cell.a = lattice_vectors
cell.unit = "Ang"

cell.verbose = 5
cell.output = pyscf_output
cell.charge = charge
cell.spin = spin
cell.symmetry = False
cell.precision = 1e-8

# basis set
cell.basis = basis
cell.exp_to_discard = 0.0

# define ecp
cell.ecp = ecp

cell.build(cart=False)

# make supercell
cell = super_cell(cell, [nx, ny, nz])


# dump POSCAR for structure verification
def dump_poscar(cell, filename):
    from pyscf.data.nist import BOHR

    lattice = cell.lattice_vectors() * BOHR  # Bohr -> Ang
    symbols = [cell.atom_symbol(i) for i in range(cell.natm)]
    coords = cell.atom_coords() * BOHR  # Bohr -> Ang
    species = []
    for s in symbols:
        if s not in species:
            species.append(s)
    counts = [symbols.count(s) for s in species]
    with open(filename, "w") as f:
        f.write("Written by jQMC test script (units: Angstrom)\n")
        f.write("1.0\n")
        for v in lattice:
            f.write(f"  {v[0]:18.10f}  {v[1]:18.10f}  {v[2]:18.10f}\n")
        f.write(" ".join(species) + "\n")
        f.write(" ".join(str(n) for n in counts) + "\n")
        f.write("Cartesian\n")
        for s_target in species:
            for s, c in zip(symbols, coords):
                if s == s_target:
                    f.write(f"  {c[0]:18.10f}  {c[1]:18.10f}  {c[2]:18.10f}\n")


poscar_filename = "silicon.vasp"
dump_poscar(cell, poscar_filename)
print(f"=== {poscar_filename} (POSCAR) ===")
with open(poscar_filename) as f:
    print(f.read())

# HF calculation
print("HF kernel=RHF")
if twist_average:
    print("twist_average=True")
    kpt_grid_m = cell.make_kpts(kpt_grid)
    mf = pbcscf.KRHF(cell, kpt_grid_m).density_fit()
else:
    print("twist_average=False")
    kpt_abs = cell.get_abs_kpts(scaled_kpts=[kpt])[0]
    mf = pbcscf.RHF(cell, kpt=kpt_abs).density_fit()

mf.chkfile = checkpoint_file
total_energy = mf.kernel()

# HF/DFT energy
print(f"Total HF/DFT energy = {total_energy}")
print("HF/DFT calculation is done.")
print("PySCF calculation is done.")
print(f"checkpoint file = {checkpoint_file}")
print(f"trexio file = {trexio_file}")

trexio.to_trexio(mf, trexio_file)
