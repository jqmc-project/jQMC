"""Generate the all-electron general-contraction TREXIO fixtures (issue #112).

Dunning cc-pVDZ lists the same primitive Gaussian in more than one contracted
shell of one atom and angular momentum (general contraction). These fixtures
let the J3 ``ao*`` basis tests check that every distinct primitive is kept
exactly once, for Cartesian and spherical AOs, molecular and periodic:

    Ar_ae_ccpvdz_cart.h5        Ar atom, cc-pVDZ, Cartesian, RHF
    Ar_ae_ccpvdz_sphe.h5        Ar atom, cc-pVDZ, spherical, RHF
    Li_pbc_ae_ccpvdz_cart.h5    bcc Li (2 atoms, a = 3.49 A), cc-pVDZ, Cartesian, Gamma RHF (GDF)
    Li_pbc_ae_ccpvdz_sphe.h5    bcc Li (2 atoms, a = 3.49 A), cc-pVDZ, spherical, Gamma RHF (GDF)

Run with PySCF (generated with PySCF 2.14.0): ``python run_pyscf_general_contraction.py``.
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
from pyscf import gto, scf
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from pyscf.tools import trexio

for filename, cart in (("Ar_ae_ccpvdz_cart.h5", True), ("Ar_ae_ccpvdz_sphe.h5", False)):
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = """
Ar   0.00000000   0.00000000  0.00000000
"""
    mol.basis = "ccpvdz"
    mol.unit = "A"
    mol.ecp = None
    mol.charge = 0
    mol.spin = 0
    mol.symmetry = False
    mol.cart = cart
    mol.build()
    mf = scf.HF(mol)
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    trexio.to_trexio(mf, filename)

for filename, cart in (("Li_pbc_ae_ccpvdz_cart.h5", True), ("Li_pbc_ae_ccpvdz_sphe.h5", False)):
    cell = pbc_gto.Cell()
    cell.a = np.eye(3) * 3.49
    cell.atom = "Li 0 0 0; Li 1.745 1.745 1.745"
    cell.basis = "ccpvdz"
    cell.cart = cart
    cell.unit = "A"
    cell.verbose = 0
    cell.build()
    mf = pbc_scf.RHF(cell).density_fit(auxbasis="def2-universal-jkfit")
    mf.max_cycle = 50
    mf.kernel()
    assert mf.converged
    trexio.to_trexio(mf, filename)
