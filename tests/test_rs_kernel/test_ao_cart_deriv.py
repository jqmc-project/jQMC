"""Layer-1: autograd-free cart-GTO AO derivative kernel (Rust ao_cart_deriv) vs
jqmc compute_AOs* (value/grad/lap, machine-prec). Foundation for forces.

Fixtures are the committed hams:
  - H2 ae cart jnone -> mol_cart_agp_jnone (AGP geminal, but only its cart AO
    leaf is used here; the geminal representation is immaterial to a pure AO
    derivative kernel)
  - N ae cart jnone (open shell) -> mol_cart_sd_jnone

Precision regime (fp64 kernel, fp64 reference): ``jqmc_rs.ao_cart_deriv`` ignores the
native precision policy and is fp64 in both modes, so the ``compute_AOs*``
references are evaluated with every precision zone at fp64
(``_fp64_reference_policy``; a no-op in full mode, so the full-mode references are
numerically unchanged). No fp32 ``ao_eval`` output enters these comparisons.
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

import contextlib

import numpy as np
import pytest

import jqmc_rs

pytestmark = pytest.mark.with_chdir

from jqmc._precision import configure, get_tolerance_xla_rust_consistency, is_mixed_precision_enabled
from jqmc.atomic_orbital import compute_AOs, compute_AOs_grad, compute_AOs_laplacian

from tests.test_rs_kernel._rs_fixtures import rs_ham

_ZONES = ("ao_eval", "ao_grad_lap")

_PATTERNS = ["mol_cart_agp_jnone", "mol_cart_sd_jnone"]


def _aos_and_r(pattern, seed=1):
    ham = rs_ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    orb = gem.orb_data_up_spin
    aos = getattr(orb, "aos_data", orb)  # unwrap MOs_data if MO-rep
    R = np.asarray(ham.structure_data.positions, float)
    nu = int(gem.num_electron_up)
    rng = np.random.default_rng(seed)
    r = R[np.arange(nu) % len(R)] + rng.normal(scale=0.8, size=(nu, 3))
    return aos, r


@contextlib.contextmanager
def _fp64_reference_policy():
    """Evaluate the enclosed XLA reference with every precision zone at fp64.

    The Rust kernel under test is fp64 in both modes; its XLA reference must be
    in the same regime. In mixed mode the full Python policy is activated for the
    enclosed block and the mixed policy restored afterwards (the native Rust
    policy is not touched); in full mode this is a no-op.
    """
    was_mixed = is_mixed_precision_enabled()
    if was_mixed:
        configure("full")
    try:
        yield
    finally:
        if was_mixed:
            configure("mixed")


def _D(aos, r, a, b, c):
    return np.asarray(jqmc_rs.ao_cart_deriv(aos, r, a, b, c))


@pytest.mark.light
@pytest.mark.parametrize("pattern", _PATTERNS)
def test_value_grad_lap(pattern):
    """Value grad lap."""
    aos, r = _aos_and_r(pattern)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    # fp64 references for the fp64 Rust kernel.
    with _fp64_reference_policy():
        value_ref = np.asarray(compute_AOs(aos, r))
        grad_ref = np.stack([np.asarray(x) for x in compute_AOs_grad(aos, r)], 0)
        lap_ref = np.asarray(compute_AOs_laplacian(aos, r))
    assert value_ref.dtype == np.float64
    assert np.allclose(_D(aos, r, 0, 0, 0), value_ref, atol=atol, rtol=rtol)
    g = np.stack([_D(aos, r, 1, 0, 0), _D(aos, r, 0, 1, 0), _D(aos, r, 0, 0, 1)], 0)
    assert np.allclose(g, grad_ref, atol=atol, rtol=rtol)
    lap = _D(aos, r, 2, 0, 0) + _D(aos, r, 0, 2, 0) + _D(aos, r, 0, 0, 2)
    assert np.allclose(lap, lap_ref, atol=atol, rtol=rtol)
