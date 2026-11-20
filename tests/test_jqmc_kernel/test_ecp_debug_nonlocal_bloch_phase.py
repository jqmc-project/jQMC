"""The debug ECP non-local oracle agrees with production at a Bloch twist (issue #784 item D).

``coulomb_potential._compute_ecp_non_local_parts_nearest_neighbors_debug`` used
to wrap every quadrature point ``r_q = R_I + |r_iI| Omega_q`` into the home cell
before evaluating Psi. The home-cell wrap is centralised in ``compute_AOs``,
which also restores the Bloch phase ``exp(i k.T)`` of the lattice vector ``T``
it removes. The external wrap moved ``r_q`` by ``T`` before that guarantor, so
at a non-Gamma twist each wrapped mesh value carried a spurious ``exp(-i k.T)``.
That made every complex-twist ECP fixture disagree with every production route
(WF-PATTERN-008, 3 up / 0 down: Re sum 0.018 against 0.061). It was a bug, not
a convention. The convention that does remain, pinned here, is that the debug
oracle returns the holomorphic ratio sum. That sum equals
``compute_ecp_non_local_parts_nearest_neighbors_holomorphic``. The fixed-phase
production entry returns ``Re`` of each per-grid term.

The five canonical ECP-target patterns are used: WF-PATTERN-004 is a real Gamma
control, and 006/008/012/014 are complex twists covering pGTO, plane-wave and
pGTO+B-spline leaves, SD and AGPn, and polarized and unpolarized spins. Each
case uses one PCG64(seed=5) normal(scale=0.9) configuration, nearest neighbour
NN=1, Nv=12 and the identity rotation; at least one quadrature point lies
outside the home cell (liveness). Both routes run in the active precision mode
under the strict ao_eval/jastrow_eval/det_eval/coulomb budget, the zones both
ratio evaluations read.
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

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc.coulomb_potential import (
    _compute_ecp_non_local_parts_nearest_neighbors_debug,
    compute_ecp_non_local_parts_nearest_neighbors,
    compute_ecp_non_local_parts_nearest_neighbors_holomorphic,
)
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_PATTERNS = ["WF-PATTERN-004", "WF-PATTERN-006", "WF-PATTERN-008", "WF-PATTERN-012", "WF-PATTERN-014"]
_NN = 1
_NV = 12


def _case(pattern_id):
    ham = build_canonical_hamiltonian(pattern_id)
    geminal = ham.wavefunction_data.geminal_data
    rng = np.random.default_rng(5)
    r_up = rng.normal(scale=0.9, size=(int(geminal.num_electron_up), 3))
    r_dn = rng.normal(scale=0.9, size=(int(geminal.num_electron_dn), 3))
    return ham, r_up, r_dn


@pytest.mark.parametrize("pattern_id", _PATTERNS)
def test_debug_nonlocal_ecp_matches_production_routes(pattern_id):
    """The debug oracle equals the holomorphic production sum and, after Re, the fixed-phase grid terms."""
    ham, r_up, r_dn = _case(pattern_id)
    coulomb, wf = ham.coulomb_potential_data, ham.wavefunction_data
    RT = np.eye(3)
    debug_up, debug_dn, debug_grid, debug_sum = _compute_ecp_non_local_parts_nearest_neighbors_debug(
        coulomb, wf, r_up, r_dn, RT, NN=_NN, Nv=_NV
    )
    _, _, fixed_phase_grid, _ = compute_ecp_non_local_parts_nearest_neighbors(
        coulomb, wf, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(RT), NN=_NN, Nv=_NV
    )
    _, _, _, holomorphic_sum = compute_ecp_non_local_parts_nearest_neighbors_holomorphic(
        coulomb, wf, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(RT), NN=_NN, Nv=_NV
    )
    debug_grid = np.asarray(debug_grid)
    fixed_phase_grid = np.asarray(fixed_phase_grid)
    assert debug_grid.shape == fixed_phase_grid.shape
    assert np.all(np.isfinite(debug_grid)) and np.all(np.isfinite(fixed_phase_grid))

    # Liveness: some quadrature point leaves the home cell, where a wrap would act.
    cell = np.asarray(ham.structure_data.cell)
    frac = np.concatenate([np.asarray(debug_up).reshape(-1, 3), np.asarray(debug_dn).reshape(-1, 3)]) @ np.linalg.inv(cell)
    assert np.any((frac < 0.0) | (frac >= 1.0)), "no quadrature point outside the home cell"

    # Convention pin: the oracle is holomorphic; at a twist its sum keeps an imaginary part.
    if np.iscomplexobj(np.asarray(holomorphic_sum)):
        assert abs(np.imag(complex(debug_sum))) > 0.0
    else:
        assert np.isrealobj(debug_grid)

    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "coulomb"), "strict")
    debug_sum_value = np.asarray(complex(debug_sum))
    holomorphic_sum_value = np.asarray(complex(holomorphic_sum))
    np.testing.assert_allclose(debug_sum_value, holomorphic_sum_value, atol=atol, rtol=rtol)
    debug_real_grid = np.real(debug_grid)
    np.testing.assert_allclose(debug_real_grid, fixed_phase_grid, atol=atol, rtol=rtol)
