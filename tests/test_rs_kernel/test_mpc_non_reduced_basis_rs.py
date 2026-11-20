"""Rust MPC value kernel on a cell given in a non-reduced basis (issue #53).

Rust counterpart of ``tests/test_jqmc_kernel/test_mpc_non_reduced_basis.py``. The Rust
kernel ``jqmc_rs._compute_mpc_potential_ee`` consumes the Python-built MPC tables, so
the radius bug of ``wigner_seitz_radius`` reached the Rust production path as well.
Here the Rust canonical MPC e-e energy of the plane-wave table fixture, written in
non-reduced bases, is compared with the independent energy reference
``plane_wave_mpc_energy_reference`` (box-face quadrature kernel, NumPy plane-wave
density, exact orthorhombic minimum image).
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
from jqmc._precision import get_complex_dtype_np, get_dtype_np
from jqmc._setting import ATOL_MPC_OFFGRID_FIXTURE_REGRESSION, RTOL_PYTHON_FP64_ZERO
from tests.test_jqmc_kernel._mpc_independent_reference import plane_wave_mpc_energy_reference
from tests.test_jqmc_kernel.test_mpc_non_reduced_basis import (
    NON_REDUCED_TRANSFORMS,
    build_fixture_mpc,
    electron_configuration,
)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize("basis_name", list(NON_REDUCED_TRANSFORMS))
@pytest.mark.gdd_electron_configuration(
    mode="required", reason="Electrons on fixed table nodes of the transformed cell for the MPC e-e energy check."
)
def test_rust_mpc_potential_in_non_reduced_basis_matches_independent_reference(basis_name):
    """Rust V_MPC on tables built in a non-reduced basis equals the independent energy reference."""
    transform = NON_REDUCED_TRANSFORMS[basis_name]
    cell, mpc = build_fixture_mpc(transform)

    r_up, r_dn = electron_configuration(cell)
    v_rust = float(jqmc_rs._compute_mpc_potential_ee(mpc, r_up, r_dn))
    v_reference = plane_wave_mpc_energy_reference(
        transform, get_dtype_np("ao_eval"), get_complex_dtype_np("ao_eval"), r_up, r_dn
    )
    assert np.isfinite(v_rust)
    np.testing.assert_allclose(v_rust, v_reference, atol=ATOL_MPC_OFFGRID_FIXTURE_REGRESSION, rtol=RTOL_PYTHON_FP64_ZERO)
