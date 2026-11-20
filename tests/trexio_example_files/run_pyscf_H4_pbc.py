#!/usr/bin/env python
"""Generate the H4-pbc TREXIO fixtures used by ``tests/test_rs_kernel/test_periodic_invariants.py``.

Two weakly-interacting H2 molecules (4 H atoms total) in a periodic cubic box, all-electron
cc-pVDZ, RHF (closed-shell, 2up + 2dn). Unlike every other committed periodic fixture (lithium
1up/1dn, H2_pbc 1up/1dn), this gives >=2 electrons PER SPIN -- the only committed source of a
natural PARALLEL-spin e-e channel (needed for the DTN f-channel / same-spin Jastrow force
coverage that no other committed periodic fixture exercises).

The two H2 units sit far apart (in-plane separation ~5.7 Angstrom, also far from every periodic
image) so RHF converges cleanly to (near) two independent H2 molecules -- deliberately NOT a
realistic solid, just a cheap, well-converged multi-electron-per-spin periodic source.

Exported at the Gamma point AND at a single non-Gamma twist (kpt = [1/4, 1/4, 1/4], complex
Bloch orbitals), cartesian AOs only (one angular form is enough; ``run_pyscf_pbc_matrix_fixtures.py``
already covers the cart/sphe axis on H2_pbc).

Requires a pyscf build carrying ``pyscf.tools.trexio`` (the TREXIO exporter; ships in the
``pyscf-forge`` extension, not pyscf core).
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

from pyscf.pbc import gto as gto_pbc
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.tools import super_cell
from pyscf.tools import trexio

B_ANG = 8.0  # cubic box edge, Angstrom
BOND_ANG = 0.74  # H-H bond length, Angstrom (matches H2_pbc_ae_*.h5)


def build_cell(cart):
    cell = gto_pbc.Cell()
    # unit A at (1,1,1)-(1,1,1+bond), unit B at (5,5,1)-(5,5,1+bond): in-plane separation
    # sqrt(4^2+4^2) ~= 5.66 Angstrom from unit A to unit B AND to every periodic image of B.
    cell.atom = [
        ["H", (1.0, 1.0, 1.0)],
        ["H", (1.0, 1.0, 1.0 + BOND_ANG)],
        ["H", (5.0, 5.0, 1.0)],
        ["H", (5.0, 5.0, 1.0 + BOND_ANG)],
    ]
    cell.a = [[B_ANG, 0.0, 0.0], [0.0, B_ANG, 0.0], [0.0, 0.0, B_ANG]]
    cell.basis = "cc-pVDZ"
    cell.spin = 0
    cell.unit = "Ang"
    cell.verbose = 3
    cell.charge = 0
    cell.symmetry = False
    cell.precision = 1e-8
    cell.exp_to_discard = 0.0
    cell.build(cart=cart)
    return super_cell(cell, [1, 1, 1])


def run(cart, twist, outfile):
    cell = build_cell(cart)
    if twist:
        kpt_abs = cell.get_abs_kpts(scaled_kpts=[[0.25, 0.25, 0.25]])[0]
        mf = pbcscf.RHF(cell, kpt=kpt_abs).density_fit()
    else:
        mf = pbcscf.RHF(cell).density_fit()
    e = mf.kernel()
    assert mf.converged, f"{outfile}: RHF did not converge (E={e})"
    print(f"{outfile}: cart={cart} twist={twist} E={e}")
    trexio.to_trexio(mf, outfile)


if __name__ == "__main__":
    import sys

    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    jobs = [
        (True, False, "H4_pbc_ae_cart.h5"),
        (True, True, "H4_pbc_ae_k025_cart.h5"),
    ]
    for cart, twist, name in jobs:
        run(cart, twist, f"{outdir}/{name}")
