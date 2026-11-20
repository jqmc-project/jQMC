"""Export unchanged converged spherical PySCF k-point states as Cartesian TREXIO.

Usage: python export_tiling_cartesian.py state.chk output.h5
The source generator must check SCF convergence before producing the checkpoint.
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

import argparse

import h5py
import numpy as np
from pyscf.pbc.scf import KRHF, KUHF, chkfile
from pyscf.tools import trexio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    cell, state = chkfile.load_scf(args.checkpoint)
    transform = cell.cart2sph_coeff(normalized="sp")
    target = cell.copy()
    target.cart = True
    independent = np.asarray(state["mo_coeff"]).ndim == 4
    mf = (KUHF if independent else KRHF)(target, kpts=state["kpts"])
    if independent:
        mf.mo_coeff = tuple(np.asarray([transform @ C for C in spin]) for spin in state["mo_coeff"])
        mf.mo_occ = tuple(state["mo_occ"])
        mf.mo_energy = tuple(state["mo_energy"])
    else:
        mf.mo_coeff = np.asarray([transform @ C for C in state["mo_coeff"]])
        mf.mo_occ = state["mo_occ"]
        mf.mo_energy = state["mo_energy"]
    mf.e_tot = state["e_tot"]
    trexio.to_trexio(mf, args.output)
    with h5py.File(args.output, "r+") as handle:
        handle["pbc/pbc_k_point_weight"][...] = [0.1, 0.1, 0.4, 0.4]
    print("Exported unchanged spherical SCF energy", mf.e_tot)


if __name__ == "__main__":
    main()
