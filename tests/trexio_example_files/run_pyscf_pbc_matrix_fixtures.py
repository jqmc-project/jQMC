#!/usr/bin/env python
"""Generate the tiny periodic TREXIO fixtures that fill the pyscf-generable
``cart_pgto`` / ``cart_pgto_complex`` and AE-periodic ``sphe_pgto`` /
``sphe_pgto_complex`` route-fidelity matrix cells
(``tests/test_jqmc_kernel/_generated_matrix_base.py``).

Two physical systems, each exported at the Gamma point AND at a single non-Gamma
twist (kpt = [1/4, 1/4, 1/4], complex Bloch orbitals), and each written in both
cartesian and spherical angular form:

  * Li2 bcc, ccECP  -> lithium_cart.h5 / lithium_k025_cart.h5  (cart_pgto ECP)
  * H2 in a cubic box, all-electron cc-pVDZ -> H2_pbc_ae_{cart,sphe}[_k025].h5

Only the files actually referenced by the matrix ``_STEM_MAP`` are kept committed;
this script documents their provenance and regenerates them deterministically.

Requires a pyscf build carrying ``pyscf.tools.trexio`` (the TREXIO exporter).
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

from pyscf import gto
from pyscf.pbc import gto as gto_pbc
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.tools import super_cell
from pyscf.tools import trexio

A_LAT = 3.51  # Angstrom, Li bcc conventional cell
LI_BASIS = {
    "Li": gto.basis.parse(
        """
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
"""
    )
}


def build_cell(kind, cart):
    cell = gto_pbc.Cell()
    if kind == "li_ecp":
        cell.atom = [["Li", (0.0, 0.0, 0.0)], ["Li", (0.5 * A_LAT, 0.5 * A_LAT, 0.5 * A_LAT)]]
        cell.a = [[A_LAT, 0.0, 0.0], [0.0, A_LAT, 0.0], [0.0, 0.0, A_LAT]]
        cell.basis = LI_BASIS
        cell.ecp = "ccecp"
        cell.spin = 0
    elif kind == "h2_ae":
        b = 6.0  # cubic box, Angstrom
        cell.atom = [["H", (0.0, 0.0, 0.0)], ["H", (0.0, 0.0, 0.74)]]
        cell.a = [[b, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, b]]
        cell.basis = "cc-pVDZ"
        cell.spin = 0
    else:
        raise ValueError(kind)
    cell.unit = "Ang"
    cell.verbose = 3
    cell.charge = 0
    cell.symmetry = False
    cell.precision = 1e-8
    cell.exp_to_discard = 0.0
    cell.build(cart=cart)
    return super_cell(cell, [1, 1, 1])


def run(kind, cart, twist, outfile):
    cell = build_cell(kind, cart)
    if twist:
        kpt_abs = cell.get_abs_kpts(scaled_kpts=[[0.25, 0.25, 0.25]])[0]
        mf = pbcscf.RHF(cell, kpt=kpt_abs).density_fit()
    else:
        mf = pbcscf.RHF(cell).density_fit()
    e = mf.kernel()
    print(f"{outfile}: kind={kind} cart={cart} twist={twist} E={e}")
    trexio.to_trexio(mf, outfile)


if __name__ == "__main__":
    import sys

    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    jobs = [
        ("li_ecp", True, False, "lithium_cart.h5"),
        ("li_ecp", True, True, "lithium_k025_cart.h5"),
        ("h2_ae", True, False, "H2_pbc_ae_cart.h5"),
        ("h2_ae", False, False, "H2_pbc_ae_sphe.h5"),
        ("h2_ae", True, True, "H2_pbc_ae_k025_cart.h5"),
        ("h2_ae", False, True, "H2_pbc_ae_k025_sphe.h5"),
    ]
    for kind, cart, twist, name in jobs:
        run(kind, cart, twist, f"{outdir}/{name}")
