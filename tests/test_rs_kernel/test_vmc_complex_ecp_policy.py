"""Native complex VMC energy preserves the independently checked ECP imaginary part."""

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
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from jqmc.determinant import compute_geminal_all_elements
from jqmc.hamiltonians import compute_local_energy_fast
from tests.test_jqmc_kernel._complex_ecp_reference import complex_local_energy_reference, ecp_terms
from tests.test_jqmc_kernel.test_complex_ecp_policy import _case, _activated
from tests.test_jqmc_kernel.test_periodic_invariants import _COND_MAX, _ginv

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]
# Same existing zones as the shared XLA oracle. Keep this tuple local so the
# static tolerance resolver can verify the native comparator's exact budget.
_ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "mo_eval",
    "jastrow_eval",
    "jastrow_grad_lap",
    "det_eval",
    "wf_kinetic",
    "coulomb",
    "ewald",
    "local_energy",
)


def _check_condition(ham, up, down):
    """Expose the shared fixture's unchanged conditioning bound to this ledger."""
    G = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, up, down))
    cond_G = np.linalg.cond(G)
    assert cond_G < _COND_MAX


def _native(ham, up, down, inverse):
    weights = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
    points = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)
    return complex(
        *jqmc_rs.compute_local_energy_fast_complex(ham, up, down, weights, points, np.asarray(inverse, dtype=np.complex128))
    )


def test_native_vmc_complex_ecp_matches_shared_independent_absolute_reference():
    """Native Li twist DTN VMC agrees with the one shared independent complex-energy oracle.

    The exact same Li TREXIO, fixed seed7 image-offset positions, cond(G)<1e4,
    identity Nv12 nearest-ECP rule and primitive/Hessian/Ewald reference as the
    XLA policy guarantee are reused, not duplicated for Rust. Im(Vnl) and both
    candidate signs must be active. Compare real and imaginary components of
    the actual native aggregate consumed by MCMC, so either a projected native
    kernel or a consumer discarding vc.im fails. Both modes use the same-mode
    independent reference and unchanged central strict zone budget.
    """
    ham, up, down, RT = _case()
    _check_condition(ham, up, down)
    _activated(ecp_terms(ham, up, down, RT))
    expected = complex_local_energy_reference(ham, up, down, RT)
    actual = _native(ham, up, down, _ginv(ham, up, down))
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    for part in (np.real, np.imag):
        np.testing.assert_allclose(part(actual), part(expected), atol=atol, rtol=rtol, equal_nan=False)


def test_native_xla_vmc_complex_ecp_consumers_agree():
    """Native and XLA VMC consumers preserve the same independently activated complex ECP energy.

    Reuse the fixed Li twist DTN input and exact geminal inverse. This is backend
    parity, separate from the independent absolute-value guarantee; agreement
    alone is not treated as proof of the physical definition. Neither imaginary
    values nor near-zero components are dropped from the comparison.
    """
    ham, up, down, RT = _case()
    _check_condition(ham, up, down)
    _activated(ecp_terms(ham, up, down, RT))
    inverse = _ginv(ham, up, down)
    actual = _native(ham, up, down, inverse)
    expected = compute_local_energy_fast(ham, up, down, RT, inverse)
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
