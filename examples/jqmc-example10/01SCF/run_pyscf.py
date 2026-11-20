"""Generate an unrestricted open-shell nitrogen TREXIO trial (ccECP)."""

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

mol = gto.Mole()
mol.verbose = 5
mol.atom = "N 0.0 0.0 0.0"
mol.basis = "ccecp-ccpvdz"
mol.ecp = "ccecp"
mol.unit = "angstrom"
mol.charge = 0
# Five valence electrons; N_up - N_down = 3: the atomic 4S ground-state spin sector.
mol.spin = 3
mol.symmetry = False
mol.cart = True
mol.output = "nitrogen_uhf.out"
mol.build()

mf = scf.UHF(mol)
mf.max_cycle = 200
mf.kernel()
if not mf.converged:
    raise RuntimeError("PySCF UHF did not converge")

trexio.to_trexio(mf, "nitrogen_uhf_ccecp_ccpvdz.h5")
