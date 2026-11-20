"""Triclinic Li ccECP J1+J2 Gamma Hamiltonians for the native periodic ECP parameter-derivative checks.

Every committed periodic TREXIO/QE fixture has a diagonal (cubic) cell. For a diagonal cell
``inv(cell) == inv(cell.T)``, so a transposed cell inverse in a native periodic kernel cannot
change any result there (issue #569: ``param_grad_ecp.rs::collect_mesh``). The Hamiltonians here
are rebuilt from the committed ``lithium.h5`` ccECP fixture through the public constructors by
``tests/test_jqmc_kernel/test_mcmc_sampler_stress_reference.py::_build_hamiltonian``, which maps
the cubic bcc cell and its nuclei by a fixed non-symmetric deformation (all three cell angles
differ from 90 degrees) and attaches active exp J1 and J2 factors:

* ``mild``: that deformation alone (the cell of the heavy strain finite-difference test);
* ``skew``: cell and nuclei additionally mapped by :data:`TRICLINIC_SKEW`.
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

from functools import cache

import numpy as np

#: Additional upper-triangular shear applied to the ``skew`` cell and nuclei.
TRICLINIC_SKEW = np.array([[1.0, 0.30, 0.15], [0.0, 1.0, 0.25], [0.0, 0.0, 1.0]])

#: Cell name -> deformation ``F`` passed to ``_build_hamiltonian``.
TRICLINIC_CELL_MAPS = {"mild": np.eye(3), "skew": TRICLINIC_SKEW}


@cache
def triclinic_li_ecp_ham(cell: str):
    """The triclinic Li ccECP J1+J2 Gamma Hamiltonian named ``cell`` (built once per process)."""
    from tests.test_jqmc_kernel.test_mcmc_sampler_stress_reference import _build_hamiltonian

    return _build_hamiltonian(TRICLINIC_CELL_MAPS[cell])


def assert_cell_inverse_not_symmetric(ham, atol: float, rtol: float) -> None:
    """Input validity: every cell angle differs from 90 degrees and ``inv(cell).T != inv(cell)``.

    The transposed inverse must differ from the inverse by more than the comparison band
    ``(atol, rtol)`` of the calling test, so a transposed cell inverse is observable there.
    """
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    gram = cell @ cell.T
    assert np.all(gram[np.triu_indices(3, 1)] != 0.0), f"cell is not triclinic: gram={gram}"
    inv_cell = np.linalg.inv(cell)
    assert not np.allclose(inv_cell.T, inv_cell, atol=atol, rtol=rtol), "inv(cell) is symmetric within the comparison band"
