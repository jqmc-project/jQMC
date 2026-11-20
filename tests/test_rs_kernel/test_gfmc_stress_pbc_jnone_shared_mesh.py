"""Periodic LRDMC HF stress of a Jastrow-free wavefunction through the shared rank-1 ECP mesh.

The production periodic GFMC stress (``PbcStressHfArgs::compute_gfmc``) drives
``gfmc_stress_ecp_pbc_withj`` for every Jastrow content. With an empty Jastrow its Jastrow
terms vanish and its ECP mesh is the shared rank-1 ``ecp_nonlocal_stress_mesh_pbc``. These
tests pin that kernel on the Jastrow-free real-Gamma fixtures (pGTO and composite pGTO+PW)
against the XLA ``jax.jacrev`` oracle of the affinely strained ``e_L^t`` (T-move and DLT-move),
within the periodic native-vs-XLA tolerance of the other GFMC stress tests.
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

import jqmc_rs  # noqa: E402

from tests.test_rs_kernel._rs_fixtures import rs_ham  # noqa: E402
from tests.test_rs_kernel.test_gfmc_forces import _ATOL_PBC, _RTOL_PBC, ALAT, _oracle_ecp_pbc_stress, _pbc_cfg  # noqa: E402

pytestmark = [pytest.mark.with_chdir]


@pytest.mark.heavy
@pytest.mark.parametrize("dltmove", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("pattern", ["pbc_gamma_sphepgto_jnone", "pbc_composite_jnone"])
def test_gfmc_stress_ecp_pbc_jnone_shared_mesh(pattern, seed, dltmove):
    """``_gfmc_stress_ecp_pbc_withj`` with the empty Jastrow vs the XLA strained-e_L^t oracle."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    ru, rd = _pbc_cfg(ham, seed)
    W = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    ref = _oracle_ecp_pbc_stress(ham, ru, rd, dltmove=dltmove)
    got = np.asarray(
        jqmc_rs._gfmc_stress_ecp_pbc_withj(
            wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, ALAT, np.eye(3), W, GP, 1, dltmove
        )
    )
    assert got.shape == ref.shape and np.max(np.abs(ref)) > 0.0
    # Max-norm budget scaled by the largest stress entry, as the other periodic GFMC stress tests.
    scale = max(1.0, float(np.abs(ref).max()))
    max_diff = float(np.abs(got - ref).max())
    assert max_diff <= _ATOL_PBC + _RTOL_PBC * scale, (pattern, seed, dltmove, max_diff, got, ref)
