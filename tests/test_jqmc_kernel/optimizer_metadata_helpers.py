"""Explicit test-only constructors for C4 variational-block metadata."""

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

from typing import Literal

import numpy as np

from jqmc.determinant import AGPn_data, SD_data, _smetric_polar_retract
from jqmc.molecular_orbital import MOs_data

from jqmc.wavefunction import ParameterTangentSpec, VariationalParameterBlock


TEST_STATE_TOKEN = ("test-lineage", 0)


def build_explicit_common_rank_agpn_from_sd(sd: SD_data) -> AGPn_data:
    """Construct a test AGPn by explicitly supplying its virtual MO rows.

    Canonical ``SD_data`` deliberately cannot perform this promotion for an
    open shell because it owns no down-spin virtual rows.  Tests that exercise
    AGPn-only coordinates must therefore create a common-rank orbital basis
    explicitly rather than relying on the removed persistent SD virtual tail.
    The occupied down rows are retained as seeds; excess up rows seed the
    additional down orbitals, and both channels are S-retracted before the
    AGPn is built.
    """
    n_common = max(int(sd.num_occ_up), int(sd.num_occ_dn))
    c_up = np.asarray(sd.orb_data_up_spin.mo_coefficients)
    c_dn_occ = np.asarray(sd.orb_data_dn_spin.mo_coefficients)
    if c_up.shape[0] < n_common:
        raise ValueError("test AGPn constructor requires the alpha-major SD convention")
    c_dn_seed = np.vstack([c_dn_occ, c_up[c_dn_occ.shape[0] : n_common]])
    c_up_common = _smetric_polar_retract(c_up[:n_common], sd.orb_data_up_spin.aos_data)
    c_dn_common = _smetric_polar_retract(c_dn_seed, sd.orb_data_dn_spin.aos_data)
    mos_up = MOs_data(
        num_mo=n_common,
        aos_data=sd.orb_data_up_spin.aos_data,
        mo_coefficients=np.ascontiguousarray(c_up_common),
    )
    mos_dn = MOs_data(
        num_mo=n_common,
        aos_data=sd.orb_data_dn_spin.aos_data,
        mo_coefficients=np.ascontiguousarray(c_dn_common),
    )
    return AGPn_data.build(
        num_electron_up=int(sd.num_electron_up),
        num_electron_dn=int(sd.num_electron_dn),
        mos_up=mos_up,
        mos_dn=mos_dn,
        time_reversal=bool(sd.time_reversal),
    )


def make_test_block(
    name,
    values,
    *,
    field: Literal["real", "complex"],
    derivative_policy: Literal["direct_real", "holomorphic", "wirtinger_pair"],
    sync_scope: Literal["sampler", "world"],
    state_token=TEST_STATE_TOKEN,
) -> VariationalParameterBlock:
    return VariationalParameterBlock.build(
        name=name,
        values=values,
        field=field,
        derivative_policy=derivative_policy,
        sync_scope=sync_scope,
        tangent_spec=ParameterTangentSpec.euclidean(state_token=state_token),
        state_token=state_token,
    )
