"""Generate the ice Ih single-k RHF trial wavefunction for jqmc-example06.

This script builds the ice Ih primitive cell (4 H2O, ccECP) with cartesian
ccecp-ccpvdz basis, runs a single-k-point RHF at the general twist
k = (0.25, 0.25, 0.25) in scaled coordinates (i.e. complex Bloch MOs, the
non-Gamma path), and dumps the result via ``pyscf.tools.trexio.to_trexio`` to::

    ice_ccecp_ccpvdz_cart.h5

This TREXIO file is then consumed by ``jqmc-tool trexio convert-to`` to
produce the ``hamiltonian_data.h5`` used by the subsequent VMC / MCMC /
LRDMC sub-steps in this example.
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
from pyscf.pbc import gto, scf
from pyscf.tools import trexio

filename = "ice_ccecp_ccpvdz_cart.h5"

basis_H = gto.basis.parse(
    """
H  S
    9.569723    2.630910e-02
    1.669254    1.560329e-01
    0.405502    4.888739e-01
H  S
    0.116087    1.000000e+00
H  P
    0.666515    1.000000e+00
"""
)

basis_O = gto.basis.parse(
    """
O  S
   19.617729    1.415845e-02
    7.154197   -1.740638e-01
    1.137108    3.984802e-01
    0.456668    5.352995e-01
    0.182222    1.954256e-01
O  S
    0.206853    1.000000e+00
O  P
   14.664866    3.867801e-02
    4.563435    1.586589e-01
    1.549011    3.591587e-01
    0.531230    4.522952e-01
    0.173419    2.457321e-01
O  P
    0.274779    1.000000e+00
O  D
    1.206147    1.000000e+00
"""
)

# ice Ih primitive cell, 4 H2O, proton-ordered (fractional coordinates)
lattice = np.array(
    [
        [3.839046, 2.216474, 0.000000],
        [-3.839047, 2.216474, 0.000000],
        [0.000000, 0.000000, 7.234567],
    ]
)
frac_atoms = [
    ("H", [0.544654, 0.090604, 0.016551]),
    ("H", [0.543538, 0.452633, 0.016125]),
    ("H", [0.094697, 0.545088, 0.981455]),
    ("H", [0.668198, 0.334083, 0.301320]),
    ("H", [0.334091, 0.665864, 0.801295]),
    ("H", [0.090862, 0.547297, 0.516122]),
    ("H", [0.453061, 0.549379, 0.516903]),
    ("H", [0.548617, 0.094916, 0.481821]),
    ("O", [0.662669, 0.331321, 0.061616]),
    ("O", [0.331314, 0.668602, 0.561601]),
    ("O", [0.335660, 0.664316, 0.936746]),
    ("O", [0.671322, 0.335643, 0.436771]),
]

cell = gto.Cell()
cell.a = lattice.tolist()
cell.atom = [(el, tuple(np.dot(f, lattice))) for el, f in frac_atoms]
cell.unit = "Angstrom"
cell.basis = {"H": basis_H, "O": basis_O}
cell.ecp = "ccecp"
cell.charge = 0
cell.spin = 0
cell.verbose = 5
cell.output = "ice.out"
cell.cart = True
cell.build()

# General twist: k = (0.25, 0.25, 0.25) in scaled (crystal) coordinates -> complex MOs
kpt_abs = cell.get_abs_kpts(scaled_kpts=[[0.25, 0.25, 0.25]])[0]

mf = scf.RHF(cell, kpt=kpt_abs).density_fit()  # GDF: avoids the huge FFT-grid in-core AO ERI
mf.max_cycle = 200
mf.kernel()

trexio.to_trexio(mf, filename)
