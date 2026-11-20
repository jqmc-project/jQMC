"""Native AO and J3 pullback for shared Cartesian TREXIO shells."""

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
from jqmc._precision import get_tolerance_min
from tests.test_jqmc_kernel.test_cartesian_radial_ownership import R_DN, R_UP, _j3_autodiff, _updated_shells

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.gdd_electron_configuration(
    mode="required", position_source="explicit", reason="Fixed off-nucleus molecular AO and J3 evaluation coordinates."
)
def test_native_cartesian_trexio_physical_shell_pullback():
    """Native values and coefficient pullbacks include the fixed component normalization."""
    aos, _, jd, points, reference_values = _updated_shells("combined")
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval"), "strict")
    np.testing.assert_allclose(jqmc_rs.compute_AOs_exact(aos, points), reference_values, atol=atol, rtol=rtol)
    actual = jqmc_rs.grad_ln_Psi_jastrow_3b_radial_complex(jd, R_UP, R_DN)
    reference = _j3_autodiff(aos, points, jd.jastrow_three_body_data.j_matrix)
    for gradient, expected in zip(actual, reference, strict=True):
        assert np.any(expected)
        np.testing.assert_allclose(gradient, expected, atol=atol, rtol=rtol)
