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
# Lithium, body-centered cubic (Im-3m, space group 229), a = 3.51 Ang
# Reference: Wyckoff, Crystal Structures, Vol. 1 (Interscience, 1963).
# 2 atoms / conventional cubic cell at Wyckoff 2a sites (0,0,0) and (1/2,1/2,1/2).
a_lat = 3.51  # Angstrom
lattice_vectors = [
    [a_lat, 0.0, 0.0],
    [0.0, a_lat, 0.0],
    [0.0, 0.0, a_lat],
]
atom_cartesian = [
    ["Li", (0.00 * a_lat, 0.00 * a_lat, 0.00 * a_lat)],
    ["Li", (0.50 * a_lat, 0.50 * a_lat, 0.50 * a_lat)],
]
checkpoint_file = "lithium_tiling_sectors.chk"
trexio_file = "lithium_tiling_sectors.h5"
pyscf_output = "out_lithium_tiling_sectors_pyscf"
charge = 0
spin = 0
basis = {
    # basis set optimized for solids:
    # Correlation-Consistent Gaussian Basis Sets for Solids Made Simple, H.-Z. Ye and T. C. Berkelbach, J. Chem. Theory Comput., 18, 1595--1606 (2022). doi: 10.1021/acs.jctc.1c01245
    "Li": gto.basis.parse("""
    #BASIS SET: (2s,2p,1d) -> [2s,2p,1d] Liq1
    Li  S
    0.651887    1.000000e+00
    Li  S
    0.047742    1.000000e+00
    Li  P
    0.310459    1.000000e+00
    Li  P
    0.083036    1.000000e+00
    Li  D
    0.105925    1.000000e+00
""")
}
ecp = "ccecp"
twist_average = True
kpt = [0.0, 0.0, 0.0]
kpt_grid = [2, 2, 2]
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


poscar_filename = "lithium_tiling_sectors.vasp"
dump_poscar(cell, poscar_filename)
print(f"=== {poscar_filename} (POSCAR) ===")
with open(poscar_filename) as f:
    print(f.read())

# HF calculation
print("HF kernel=RHF")
if twist_average:
    print("twist_average=True")
    # Complete Gamma/general supercell sectors for S = diag(2,1,1).
    kpt_grid_m = cell.get_abs_kpts(np.array([[0, 0, 0], [0.5, 0, 0], [0.11, 0.13, 0.09], [0.61, 0.13, 0.09]]))
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

assert mf.converged, "KRHF source fixture must converge before export"
trexio.to_trexio(mf, trexio_file)

# QMC sector quadrature weights; SCF used the four-point uniform density.
import h5py

with h5py.File(trexio_file, "r+") as handle:
    handle["pbc/pbc_k_point_weight"][...] = np.array([0.1, 0.1, 0.4, 0.4])
