"""Native Rust non-shell-major periodic pGTO permutation invariance."""

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

from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default, _geminal_inv_batched
from tests.test_jqmc_kernel.test_derivatives_ragged_basis import _ATOL, _RTOL, _cfg_single, _make_pair

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_MESH = _NV_BASE_MESH[Nv_default]
_W = np.asarray(_MESH.weights, float)
_GP = np.asarray(_MESH.grid_points, float)


def _rust_ginv(ham, ru, rd):
    _, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, ru[None], rd[None])
    return np.asarray(ginv[0])


def _rust_axis(ham, ru, rd, ginv, axis):
    ru_n, rd_n = np.asarray(ru), np.asarray(rd)
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cp = ham.coulomb_potential_data
    gi = np.asarray(ginv)
    if axis == "eL":
        return np.asarray(jqmc_rs.compute_local_energy(ham, ru_n, rd_n, _W, _GP))
    if axis == "R":
        dR, du, dd = jqmc_rs.vmc_grad_e_L_pbc(gd, jd, cp, ru_n, rd_n, _W, _GP, gi)
        return np.concatenate([np.asarray(dR).ravel(), np.asarray(du).ravel(), np.asarray(dd).ravel()])
    if axis == "eps":
        return np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, ru_n, rd_n, _W, _GP @ np.eye(3), gi))
    raise ValueError(axis)


@pytest.mark.parametrize("spec_key", ["jnone", "j123"])
@pytest.mark.parametrize("axis", ["eL", "R", "eps"])
def test_rust_permuted_equals_reference(spec_key, axis):
    """Native e_L, force, and stress are invariant to physical AO permutation."""
    ref, perm = _make_pair(spec_key)
    ru, rd = _cfg_single(ref)
    a = _rust_axis(ref, ru, rd, _rust_ginv(ref, ru, rd), axis)
    b = _rust_axis(perm, ru, rd, _rust_ginv(perm, ru, rd), axis)
    np.testing.assert_allclose(b, a, atol=_ATOL, rtol=_RTOL, err_msg=f"{spec_key}/{axis} rust permuted != reference")
