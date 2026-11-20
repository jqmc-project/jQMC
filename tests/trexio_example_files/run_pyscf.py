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

from pyscf import gto, scf
from pyscf.tools import trexio

#'''
# H_ae_ccpvdz_cart
filename = "H_ae_ccpvdz_cart.h5"
atoms = """
H    0.00000000   0.00000000  0.00000000
"""
basis = "ccpvdz"
ecp = None
cart = True
charge = 0
spin = 1
#'''

'''
# H2_ecp_ccpvdz_cart
filename = "H2_ecp_ccpvdz_cart.h5"
atoms = """
H    0.00000000   0.00000000  -0.35000000
H    0.00000000   0.00000000  +0.35000000
"""
basis = "ccecp-ccpvdz"
ecp = "ccecp"
cart = True
charge = 0
spin = 0
'''

'''
# H_ecp_ccpvdz_cart
filename = "H_ecp_ccpvdz_cart.h5"
atoms = """
H    0.00000000   0.00000000  0.00000000
"""
basis = "ccecp-ccpvdz"
ecp = "ccecp"
cart = True
charge = 0
spin = 1
'''

'''
# H2_ae_ccpvdz_cart
filename = "H2_ae_ccpvdz_cart.h5"
atoms = """
H    0.00000000   0.00000000  -0.35000000
H    0.00000000   0.00000000  +0.35000000
"""
basis = "ccpvdz"
ecp = None
cart = True
charge = 0
spin = 0
'''

'''
# Li_ae_ccpvdz_cart
filename = "Li_ae_ccpvdz_cart.h5"
atoms = """
Li    0.00000000   0.00000000  0.0000000
"""
basis = "ccpvdz"
ecp = None
cart = True
charge = 0
spin = 1
'''

'''
# Li_ecp_ccpvdz_cart -- the smallest committed fixture that is
# BOTH single-spin (n_up=1, n_dn=0) AND has a NON-vacuous non-local ECP channel.
# ccECP removes Li's 2 core electrons, leaving a single valence electron (doublet
# -> 1up/0dn), and unlike H's ccECP (z_core=0, zero l=0 projector) it keeps a real
# l=0 projector (V_nl ~ 5 Ha), so it exercises the non-local ECP path on an empty
# spin channel. Used by tests/test_rs_kernel/test_open_shell_single_spin.py.
filename = "Li_ecp_ccpvdz_cart.h5"
atoms = """
Li    0.00000000   0.00000000  0.0000000
"""
basis = "ccecp-ccpvdz"
ecp = "ccecp"
cart = True
charge = 0
spin = 1
'''

# N_ae_ccpvdz_cart
filename = "N_ae_ccpvdz_cart.h5"
atoms = """
N    0.00000000   0.00000000  0.0000000
"""
basis = "ccpvdz"
ecp = None
cart = True
charge = 0
spin = 3

# pyscf script below
mol = gto.Mole()
mol.verbose = 5
mol.atom = atoms
mol.basis = basis
mol.unit = "A"
mol.ecp = ecp
mol.charge = charge
mol.spin = spin
mol.symmetry = False
mol.cart = cart
mol.output = "Li.out"
mol.build()

mf = scf.HF(mol)
mf.max_cycle = 200
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)
