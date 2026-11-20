"""The J3 coupling-block SYMMETRY invariant that jqmc's streaming / ratio / ECP-mesh kernels
rely on -- pinned so it cannot silently rot.

``jqmc.derivatives.one_move_mesh.j3_mesh`` folds the same-spin ``sum_{i<j} a_i^T J a_j`` pair term
assuming ``J == J^T``, and several jqmc streaming / ratio kernels deliberately exploit the
symmetry (``j3_mat.T @ x == j3_mat @ x``, dropping a matvec and the ``U = j3_mat.T @ aos`` cached
tables). A non-symmetric J3 coupling block is therefore INVALID input, and
``Jastrow_three_body_data.sanity_check`` (jastrow_factor.py) is the guard that rejects it.

Constructing the dataclass directly, or injecting via ``.replace(j_matrix=...)``, BYPASSES
``sanity_check`` -- doing so and then observing an analytic-vs-``_auto`` mismatch diagnoses
nothing, because the forward and the fast path are then computing different objects. This test
pins the guard so the symmetry assumption the kernels exploit cannot silently break.

This is an INVARIANT pin, not an analytic-vs-``_auto`` gradient fidelity check -- the periodic-ECP
Jastrow ``dE_L/dc`` route fidelity (with a valid, symmetric J3) lives in the bridge-routed
``test_analytic_grad_matches_autograd_curated.py`` matrix (the ``Li_pbc_gamma_*`` fixtures), which exercises the
exact production bridge.
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
from jqmc._setting import ATOL_NONVACUITY_DISTINCTION, RTOL_NONVACUITY_DISTINCTION

import os

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from jqmc.jastrow_factor import Jastrow_three_body_data
from jqmc.trexio_wrapper import read_trexio_file

# Periodic-Gamma ECP sphe-pGTO committed fixture (README Table 3 row 3-14).
_TREXIO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")
_STEM = "lithium.h5"


@pytest.mark.light
def test_nonsymmetric_j3_rejected_by_sanity_check():
    """A non-symmetric J3 coupling block is INVALID input and must be rejected loudly.

    ``sanity_check`` is the only guard -- constructing the dataclass directly or injecting
    via ``.replace(j_matrix=...)`` bypasses it. This test pins the guard so the symmetry
    assumption the kernels exploit (``j3_mat.T @ x == j3_mat @ x``) cannot silently rot.
    """
    (_sd, aos_data, _mu, _md, _gd, _cp) = read_trexio_file(os.path.join(_TREXIO, _STEM), store_tuple=True)
    n = int(aos_data._num_orb)
    jm = np.random.default_rng(7).normal(size=(n, n + 1)) * 0.1  # deliberately NOT symmetrized
    assert not np.allclose(jm[:, :-1], jm[:, :-1].T, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION), (
        "probe must actually be non-symmetric"
    )
    j3 = Jastrow_three_body_data(orb_data=aos_data, j_matrix=jnp.asarray(jm))
    with pytest.raises(ValueError, match="symmetric"):
        j3.sanity_check()
