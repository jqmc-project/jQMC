"""Hamiltonians for the GFMC incremental-vs-full-recompute twins (XLA and Rust).

``skewed_h4_ham`` puts the committed periodic all-electron H4 fixture onto the
strongly skewed triclinic basis of ``test_noncubic_cell_oracles.py``
(``h4_*_triclinic_lattice_strong_skew``: the triclinic lattice written in
``b1 = a1, b2 = a1 + a2, b3 = a1 + a2 + a3``), through the production TREXIO
loader, and adds the J1 (exp) + J2 (pade) Jastrow of the other twin fixtures.
The cell has off-diagonal metric entries, so the Ewald real-space kernels take
the general nearest-image search instead of the orthogonal fractional wrap,
and, being all-electron, the carried ``ei_disc`` row channel is live. With
``twist=True`` the k = (0.25, 0, 0) file is used (complex determinant).
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

from __future__ import annotations

import os
import tempfile

import numpy as np

from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._ecp_core_electrons import core_electrons_of
from tests.test_jqmc_kernel import _noncubic_oracles as noncubic

_TREXIO = os.path.join(os.path.dirname(__file__), "trexio_example_files")

#: The triclinic lattice and the strong unimodular skew of test_noncubic_cell_oracles.py.
TRICLINIC_LATTICE = np.array([[15.1178, 0.0, 0.0], [4.2, 13.9, 0.0], [3.1, 2.6, 13.4]], dtype=np.float64)
STRONG_SKEW = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.float64)
_NUCLEUS_MARGIN = 0.05


def _with_j1_j2(structure_data, geminal_data, coulomb_potential_data) -> Hamiltonian_data:
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
        jastrow_1b_type="exp",
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure_data
    )
    wf = Wavefunction_data(
        jastrow_data=Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=None),
        geminal_data=geminal_data,
    )
    ham = Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)
    ham.sanity_check()
    return ham


def skewed_h4_ham(twist: bool = False) -> Hamiltonian_data:
    """Periodic all-electron H4 (J1 + J2) on the strongly skewed triclinic cell basis."""
    stem = "H4_pbc_ae_k025_cart.h5" if twist else "H4_pbc_ae_cart.h5"
    src = os.path.join(_TREXIO, stem)
    shift = noncubic.choose_nucleus_shift(src, STRONG_SKEW, TRICLINIC_LATTICE, _NUCLEUS_MARGIN)
    with tempfile.TemporaryDirectory() as root:
        path = noncubic.transformed_trexio_copy(
            src, root, "h4_skew", STRONG_SKEW, lattice=TRICLINIC_LATTICE, nucleus_shift_frac=shift
        )
        structure_data, _aos, _mos_up, _mos_dn, geminal_data, coulomb_potential_data = read_trexio_file(path, store_tuple=True)
    ham = _with_j1_j2(structure_data, geminal_data, coulomb_potential_data)
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    metric = cell @ cell.T
    assert np.max(np.abs(metric[np.triu_indices(3, 1)])) > 1.0, "the rewritten basis is not skewed"
    assert bool(ham.structure_data.pbc_flag) and not bool(ham.coulomb_potential_data.ecp_flag)
    return ham


#: Distinct integer lattice translations per electron (cycled): the unwrapped
#: "far" start. Pairwise differences reach 8 cells along every axis, beyond the
#: +-6 (c) to +-9 (a, b) real-space image shells the skewed H4 Ewald tables
#: carry, so the carried ee update is only right through its nearest-image
#: search (without it the displaced pair falls outside the image set).
_FAR_SHIFTS = np.array([[0, 0, 0], [4, -4, 4], [-4, 4, -4], [4, 4, -4]], dtype=np.float64)


def far_start(ham, seed: int = 5):
    """Seeded in-cell fractional positions, each electron translated by its own lattice vector."""
    gd = ham.wavefunction_data.geminal_data
    n_up, n_dn = int(gd.num_electron_up), int(gd.num_electron_dn)
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    rng = np.random.default_rng(seed)
    frac = rng.random((n_up + n_dn, 3)) + _FAR_SHIFTS[np.arange(n_up + n_dn) % len(_FAR_SHIFTS)]
    r_all = frac @ cell
    return r_all[:n_up], r_all[n_up:]
