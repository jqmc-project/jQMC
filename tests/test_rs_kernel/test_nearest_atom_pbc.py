"""Layer-1 equivalence: PBC Wigner-Seitz nearest-atom (MCMC proposal cusp).

`jqmc_rs._nearest_atom_periodic` mirrors the jqmc MCMC proposal's nearest-nucleus
selection + min-image distance used for the Kato cusp step size:
`_find_nearest_index_jnp` (nearest index by min-image) and
`_get_min_dist_rel_R_cart_jnp` (min-image distance to that atom). Both reduce
each per-atom displacement via `_minimum_image_displacement` (Wigner-Seitz,
exact for oblique cells). Checked on UNWRAPPED points (incl. far outside the
home cell), since the native sweep carries walker positions unwrapped
(matching jqmc).

Committed Hamiltonian fixtures:
  - lithium_pbc ecp sphepgto j12 -> pbc_gamma_sphepgto_j12
  - Li2_pbc ecp pgtopw jnone     -> pbc_composite_jnone (QE pgto+pw composite;
    the geometry axis under test here is cell + atom positions)
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

import numpy as np
import pytest

import jqmc_rs

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.structure import _find_nearest_index_jnp, _get_min_dist_rel_R_cart_jnp

from tests.test_rs_kernel._rs_fixtures import rs_ham

# Wigner-Seitz min-image reduction is exact (no truncation); "ewald" is the
# closest owning-module zone for a periodic-geometry primitive.
_ZONES = ("ewald",)

_PATTERNS = ["pbc_gamma_sphepgto_j12", "pbc_composite_jnone"]


@pytest.mark.light
@pytest.mark.parametrize("pattern", _PATTERNS)
def test_nearest_atom_periodic(pattern):
    """Nearest atom periodic."""
    ham = rs_ham(pattern)
    sd = ham.structure_data
    pos = np.asarray(sd.positions, dtype=np.float64)
    cell = np.asarray(sd.cell, dtype=np.float64)
    n_max = int(sd._n_max)
    rng = np.random.default_rng(11)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES)
    worst_d = 0.0
    for _ in range(200):
        r = (rng.random(3) @ cell) + (rng.integers(-3, 4, 3) @ cell)
        idx_r, d_r = jqmc_rs._nearest_atom_periodic(pos, list(r), cell, n_max)
        idx_j = int(_find_nearest_index_jnp(sd, r))
        rel = np.asarray(_get_min_dist_rel_R_cart_jnp(sd, r, idx_j), dtype=np.float64)
        d_j = float(np.linalg.norm(rel))
        assert idx_r == idx_j, (idx_r, idx_j, r)
        worst_d = max(worst_d, abs(d_r - d_j))
    assert worst_d < atol + rtol * max(1.0, worst_d), worst_d


@pytest.mark.light
def test_nearest_atom_periodic_in_non_reduced_basis_matches_orthorhombic_reference():
    """Rust min-image distance on a non-reduced cell basis is the exact shortest vector.

    The lattice diag(6, 7, 8) Bohr is written in the basis ``U @ diag(6, 7, 8)``
    (``det U = 1``) whose ``a + b + c`` diagonal is far shorter than the other body
    diagonals. ``jqmc_rs._nearest_atom_periodic`` (``ewald::min_image`` with the
    structure's ``_n_max``) must return, for one atom at the origin, the per-axis
    wrap length in the rectangular basis of the same lattice -- an exact reference
    that uses no production helper and no image box.
    """
    from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE
    from tests.conftest import make_pbc_structure
    from tests.test_jqmc_kernel.test_structure import (
        NON_REDUCED_MIC_LENGTHS,
        NON_REDUCED_MIC_TRANSFORM,
        non_reduced_mic_displacements,
        orthorhombic_minimum_image,
    )

    lengths = NON_REDUCED_MIC_LENGTHS
    cell = NON_REDUCED_MIC_TRANSFORM @ np.diag(lengths)
    structure = make_pbc_structure(cell)
    displacements = non_reduced_mic_displacements(cell)
    reference_norm = np.linalg.norm(orthorhombic_minimum_image(displacements, lengths), axis=1)
    origin = np.zeros((1, 3), dtype=np.float64)
    rust = np.array([jqmc_rs._nearest_atom_periodic(origin, list(r), cell, int(structure._n_max))[1] for r in displacements])
    np.testing.assert_allclose(rust, reference_norm, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
