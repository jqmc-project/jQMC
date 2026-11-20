"""Independent nodal guard for the native standalone real-spline value path."""

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

import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import get_tolerance, precision_policy_snapshot
from jqmc.determinant import AGP_data
from tests._bspline_nodal_fixture import nodal_spline_case

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize("cell_case", ["home", "translated"])
def test_native_real_spline_half_cell_nodal_planes(cell_case):
    """An identity unpaired block exposes standalone AO values without inversion.

    The VGL export evaluates in the derivative zone and cannot exercise the
    fp32 standalone path. With zero down electrons and identity unpaired
    columns, this native geminal is exactly the transposed AO value table.
    """
    aos, nodes, off_nodes = nodal_spline_case(cell_case)
    gem = AGP_data.build(
        num_electron_up=3,
        num_electron_dn=0,
        orb_data_up_spin=aos,
        orb_data_dn_spin=aos,
        lambda_matrix=np.column_stack((np.zeros((3, 3)), np.eye(3))),
    )
    jqmc_rs.configure_native_precision(precision_policy_snapshot())
    atol, rtol = get_tolerance("ao_eval", "strict")

    def selected_values(points):
        matrices = np.asarray(
            [jqmc_rs.compute_geminal_all_elements(gem, points[:, sample, :], np.zeros((0, 3))) for sample in range(32)]
        )
        assert matrices.shape == (32, 3, 3)
        assert np.isfinite(matrices).all()
        return np.diagonal(matrices, axis1=1, axis2=2)

    node_values = selected_values(nodes)
    off_values = selected_values(off_nodes)
    assert np.min(np.abs(off_values)) > 100 * atol
    np.testing.assert_allclose(node_values, np.zeros_like(node_values), atol=atol, rtol=rtol, equal_nan=False)
