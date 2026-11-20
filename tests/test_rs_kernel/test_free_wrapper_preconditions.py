"""Unsupported free real parameter/stress calls raise ordinary Python exceptions."""

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

from jqmc._precision import precision_policy_snapshot
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from jqmc.determinant import to_AGP
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]

CASES = (
    ("grad_eL_wrt_lambda", "sd", "undefined for occupied-only SD"),
    ("grad_eL_ecp_wrt_lambda", "sd", "undefined for occupied-only SD"),
    ("grad_eL_ecp_wrt_lambda_diag", "sd", "undefined for occupied-only SD"),
    ("grad_eL_ecp_wrt_mo_coeff", "ao_agp", "up spin has no mo_coefficients"),
    ("grad_eL_wrt_jastrow_3b_jmatrix", "no_j3", "no jastrow_three_body_data"),
    ("grad_eL_ecp_wrt_jastrow_3b_jmatrix", "no_j3", "no jastrow_three_body_data"),
    ("grad_T_bspline_dcell", "non_bspline", "not a B-spline leaf"),
    ("grad_T_bspline_kinetic_strain", "non_bspline", "not a B-spline leaf"),
)


@pytest.mark.parametrize("wrapper,invalid_input,message", CASES, ids=[case[0] for case in CASES])
def test_free_real_wrapper_unsupported_input_is_value_error(wrapper, invalid_input, message):
    """Invalid representation/channel input stays catchable by except Exception."""
    ham = rs_ham("pbc_gamma_sphepgto_j12")
    wf = ham.wavefunction_data
    gem = to_AGP(wf.geminal_data) if invalid_input == "ao_agp" else wf.geminal_data
    rng = np.random.default_rng(7)
    cell = np.asarray(ham.structure_data.cell)
    up = np.ascontiguousarray(rng.uniform(0, 1, (gem.num_electron_up, 3)) @ cell)
    dn = np.ascontiguousarray(rng.uniform(0, 1, (gem.num_electron_dn, 3)) @ cell)
    args = [gem, wf.jastrow_data]
    if "ecp" in wrapper:
        args.append(ham.coulomb_potential_data)
    args.extend([up, dn])
    if "ecp" in wrapper:
        args.extend(
            [
                np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=float),
                np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=float),
            ]
        )
    if wrapper == "grad_T_bspline_dcell":
        args.append(np.eye(3))
    with jqmc_rs.NativeConstructionPrecision(precision_policy_snapshot()):
        with pytest.raises(ValueError, match=message) as error:
            getattr(jqmc_rs, wrapper)(*args)
    assert isinstance(error.value, Exception)
