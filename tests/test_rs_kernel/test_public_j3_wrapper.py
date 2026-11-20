"""The retained native J3 nuclear-gradient export has direct numerical evidence."""

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

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.derivatives import assemble
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_forces import _cfg, _rand_j3

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


@pytest.mark.parametrize("seed", [1, 3])
def test_native_public_j3_nuclear_gradient_matches_xla(seed):
    """Native grad_J3_wrt_R equals analytical XLA on a live molecular Cartesian J123 fixture.

    N_ae_ccpvdz_cart.h5 supplies the mol_cart_sd_j123 structure and basis.
    Symmetric J3 coefficients use seed 7; electron positions use the declared
    ion-centered configuration seed, without selecting on backend agreement.
    The same configured precision mode reaches both implementations.
    """
    ham = rs_ham("mol_cart_sd_j123")
    j3 = _rand_j3(ham, seed=7).jastrow_three_body_data
    up, dn = _cfg(ham, seed=seed)
    reference = np.asarray(assemble.grad_J3_wrt_R(j3, up, dn))
    actual = np.asarray(jqmc_rs.grad_J3_wrt_R(j3, up, dn))
    atol, rtol = get_tolerance_xla_rust_consistency(("ao_eval", "ao_grad_lap", "jastrow_eval"), "strict")
    assert reference.shape == np.asarray(ham.structure_data.positions).shape
    assert actual.shape == reference.shape
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    assert np.max(np.abs(reference)) > 100 * atol
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
