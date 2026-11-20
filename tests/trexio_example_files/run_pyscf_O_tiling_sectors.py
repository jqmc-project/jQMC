"""Independent-spin periodic oxygen, fixed per-k triplet populations."""

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
import h5py
from pyscf.pbc import gto, scf
from pyscf.tools import trexio

cell = gto.Cell(
    atom="O 5 5 5",
    a=np.eye(3) * 10,
    unit="Bohr",
    basis="ccpvdz",
    ecp="ccecp",
    spin=2,
    verbose=4,
    output="pyscf.log",
    precision=1e-8,
)
cell.build(cart=False)
kpts = cell.get_abs_kpts(np.array([[0, 0, 0], [0.5, 0, 0], [0.11, 0.13, 0.09], [0.61, 0.13, 0.09]]))
mf = scf.KUHF(cell, kpts).density_fit()


# Fixed integer per-k populations are part of this QMC source fixture.
def occupied(mo_energy=None, mo_coeff=None):
    e = mf.mo_energy if mo_energy is None else mo_energy
    return np.array(
        [[np.r_[np.ones(n), np.zeros(len(bands) - n)] for bands in spin] for n, spin in zip((4, 2), e, strict=True)]
    )


mf.get_occ = occupied
mf.chkfile = "oxygen_tiling_sectors.chk"
mf.conv_tol = 1e-9
mf.max_cycle = 150
energy = mf.kernel()
assert mf.converged, "KUHF source fixture must converge"
mf.mo_coeff = tuple(mf.mo_coeff)
mf.mo_occ = tuple(mf.mo_occ)
mf.mo_energy = tuple(mf.mo_energy)
trexio.to_trexio(mf, "oxygen_tiling_sectors_sphe.h5")
with h5py.File("oxygen_tiling_sectors_sphe.h5", "r+") as handle:
    handle["pbc/pbc_k_point_weight"][...] = [0.1, 0.1, 0.4, 0.4]
print("Converged KUHF", energy, flush=True)
