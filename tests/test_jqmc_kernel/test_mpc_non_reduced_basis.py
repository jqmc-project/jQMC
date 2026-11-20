"""MPC on a cell given in a non-reduced basis (issue #53).

A ``wigner_seitz_radius`` that searches only ``|n_i| <= 2`` fails on a non-reduced
basis: for the physical box diag(5, 6, 7) Bohr written in the basis
``U @ diag(5, 6, 7)`` with ``U = [[1, 1, 0], [2, 3, 1], [0, 0, 1]]`` the shortest
lattice vector is ``a1 = 3 b1 - b2 + b3``, so the radius comes out too large and
the Hood split of ``build_mpc_data`` is matched outside the Wigner-Seitz cell.

Independent references (no production MPC helper):

* the inscribed Wigner-Seitz radius of the physical box, ``min(lengths) / 2``;
* the box-face quadrature V_LR of ``_mpc_independent_reference`` on the table nodes
  and off grid, for the same fixed plane-wave fixture and basis;
* the canonical MPC e-e energy assembled from that reference V_LR, its ``V_const``
  and an exact orthorhombic minimum image.

The table bounds are the existing fixture regression bounds of this plane-wave
fixture (``ATOL_MPC_TABLE_FIXTURE_REGRESSION`` / ``ATOL_MPC_OFFGRID_FIXTURE_REGRESSION``).
The Rust value kernel is covered by
``tests/test_rs_kernel/test_mpc_non_reduced_basis_rs.py``.
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

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_complex_dtype_np, get_dtype_np
from jqmc._setting import (
    ATOL_MPC_OFFGRID_FIXTURE_REGRESSION,
    ATOL_MPC_TABLE_FIXTURE_REGRESSION,
    ATOL_PYTHON_FP64_TIGHT,
    RTOL_PYTHON_FP64_ZERO,
)
from jqmc.atomic_orbital import AOs_plane_wave_complex_data
from jqmc.determinant import SD_data
from jqmc.molecular_orbital import MOs_data
from jqmc.mpc import build_mpc_data, compute_mpc_potential_ee, evaluate_v_lr, wigner_seitz_radius
from tests._opposite_spin_twist_cases import opposite_twist_partner
from tests.test_jqmc_kernel._mpc_independent_reference import (
    TABLE_FIXTURE_CUTOFF_HA,
    TABLE_FIXTURE_GRID,
    TABLE_FIXTURE_LENGTHS,
    TABLE_FIXTURE_MO_COEFFICIENTS,
    TABLE_FIXTURE_PW_MODES,
    TABLE_FIXTURE_TWIST,
    plane_wave_mpc_energy_reference,
    plane_wave_mpc_reference,
)
from tests.test_jqmc_kernel._noncubic_oracles import build_structure

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

#: Unimodular bases of the fixture box whose shortest vector needs an integer
#: coefficient of magnitude 3 or more in that basis (``|n_i| <= 2`` gives L = 3.0
#: and 6.5 Bohr instead of 2.5).
NON_REDUCED_TRANSFORMS = {
    "U-110-231-001": np.array([[1, 1, 0], [2, 3, 1], [0, 0, 1]]),
    "U-strongly-skewed": np.array([[-3, -3, -2], [-2, -1, -2], [1, -3, 3]]),
}
#: Table-node fractions (in units of 1/TABLE_FIXTURE_GRID along each basis vector of
#: the transformed cell), with whole-cell offsets so the periodic wrap of the lookup
#: is exercised. On a skewed 16^3 grid the off-node B-spline interpolation error is
#: set by the long basis vectors, not by the Wigner-Seitz radius, so lookups and
#: electrons sit on nodes, where the lookup reproduces the table.
LOOKUP_NODES = np.array([[3, 5, 7], [11, 2, 13], [8, 15, 0], [1, 9, 4]])
LOOKUP_CELL_OFFSETS = np.array([[0, 0, 0], [-1, 1, 0], [2, 0, -1], [0, -3, 1]])
#: Indices into LOOKUP_NODES used as the up and the down electron.
ELECTRON_NODE_UP = 0
ELECTRON_NODE_DN = 1


def build_fixture_mpc(transform: np.ndarray):
    """Production MPC tables of the plane-wave fixture in the basis ``transform @ diag(lengths)``."""
    cell = np.asarray(transform, dtype=np.float64) @ np.diag(TABLE_FIXTURE_LENGTHS)
    structure = build_structure(cell, np.array([[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]]))
    aos = AOs_plane_wave_complex_data.build(
        structure, 2.0 * np.pi * TABLE_FIXTURE_PW_MODES / TABLE_FIXTURE_LENGTHS, k_point=tuple(TABLE_FIXTURE_TWIST)
    )
    mos = MOs_data.build(aos_data=aos, mo_coefficients=TABLE_FIXTURE_MO_COEFFICIENTS[None, :])
    sd = SD_data.build(
        num_electron_up=1, num_electron_dn=1, mos_up=mos, mos_dn=opposite_twist_partner(mos), time_reversal=False
    )
    hamiltonian = SimpleNamespace(structure_data=structure, wavefunction_data=SimpleNamespace(geminal_data=sd))
    return cell, build_mpc_data(hamiltonian, cutoff_ha=TABLE_FIXTURE_CUTOFF_HA, grid=(TABLE_FIXTURE_GRID,) * 3)


def lookup_fractions() -> np.ndarray:
    """Fractions (transformed basis) of the lookup points: table nodes plus whole cells."""
    return LOOKUP_NODES / TABLE_FIXTURE_GRID + LOOKUP_CELL_OFFSETS


def electron_configuration(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian up / down electron on two of the lookup points of ``cell``."""
    points = lookup_fractions() @ cell
    return points[[ELECTRON_NODE_UP]], points[[ELECTRON_NODE_DN]]


@pytest.mark.parametrize("basis_name", list(NON_REDUCED_TRANSFORMS))
@pytest.mark.gdd_electron_configuration(
    mode="required", reason="Fixed fractional electron coordinates for the MPC e-e energy check."
)
def test_mpc_in_non_reduced_basis_matches_independent_references(basis_name):
    """L, the V_LR table, its lookup and V_MPC in a non-reduced basis match independent references."""
    transform = NON_REDUCED_TRANSFORMS[basis_name]
    cell, mpc = build_fixture_mpc(transform)

    # The basis really is non-reduced: no |n_i| <= 2 combination reaches the shortest vector.
    small = np.stack(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    small = small[np.any(small != 0, axis=1)]
    assert 0.5 * np.min(np.linalg.norm(small @ cell, axis=1)) > 0.5 * np.min(TABLE_FIXTURE_LENGTHS)

    np.testing.assert_allclose(
        wigner_seitz_radius(cell), 0.5 * np.min(TABLE_FIXTURE_LENGTHS), atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_ZERO
    )
    np.testing.assert_allclose(
        mpc.L, 0.5 * np.min(TABLE_FIXTURE_LENGTHS), atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_PYTHON_FP64_ZERO
    )

    assert np.array_equal(np.asarray(mpc.origin), np.zeros(3))
    reference_nodes, reference_lookup = plane_wave_mpc_reference(
        transform, get_dtype_np("ao_eval"), get_complex_dtype_np("ao_eval"), lookup_fractions()
    )
    table = np.asarray(mpc.v_lr_grid).reshape(-1)
    lookup = np.asarray(evaluate_v_lr(mpc, jnp.asarray(lookup_fractions() @ cell, dtype=jnp.float64)))
    assert np.all(np.isfinite(table)) and np.all(np.isfinite(lookup))
    assert np.ptp(reference_nodes) > 0.0
    np.testing.assert_allclose(table, reference_nodes, atol=ATOL_MPC_TABLE_FIXTURE_REGRESSION, rtol=RTOL_PYTHON_FP64_ZERO)
    np.testing.assert_allclose(lookup, reference_lookup, atol=ATOL_MPC_OFFGRID_FIXTURE_REGRESSION, rtol=RTOL_PYTHON_FP64_ZERO)

    r_up, r_dn = electron_configuration(cell)
    v_mpc = float(compute_mpc_potential_ee(mpc, jnp.asarray(r_up), jnp.asarray(r_dn)))
    v_reference = plane_wave_mpc_energy_reference(
        transform, get_dtype_np("ao_eval"), get_complex_dtype_np("ao_eval"), r_up, r_dn
    )
    np.testing.assert_allclose(v_mpc, v_reference, atol=ATOL_MPC_OFFGRID_FIXTURE_REGRESSION, rtol=RTOL_PYTHON_FP64_ZERO)
