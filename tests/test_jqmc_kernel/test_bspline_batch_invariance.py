"""CPU real-quintic contraction parity, not whole spline-evaluation parity.

Identical precomputed control blocks and weights must survive batching. The
fp32 CPU implementation promises a fixed reduction tree; fp64 deliberately
retains einsum and is checked under the existing strict AO tolerance instead.
Weight generation, coordinate transforms, and the complex leaf are not covered.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_dtype_jnp, get_tolerance
from jqmc.atomic_orbital import AOs_bspline_data

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _control_inputs(dtype):
    """Fixed one-band K=6 blocks and weights, cast to the requested input dtype."""
    rng = np.random.default_rng(2317)
    controls = jnp.asarray(rng.normal(size=(17, 1, 6, 6, 6)), dtype=dtype)
    weights = jnp.asarray(rng.uniform(size=(17, 3, 6)), dtype=dtype)
    return controls, weights


def test_real_quintic_control_contraction_batch_single():
    """Active-mode one-band K=6 contractions obey existing strict AO parity.

    The 17 control blocks and three weight vectors per block are fixed inputs,
    generated once and shared unchanged between independently compiled routes.
    One band is intentional: current CPU XLA can make the old einsum invariant
    for multiple bands while still changing its reduction for this shape.
    No full/mixed output comparison or higher-precision reference is involved.
    """
    assert jax.default_backend() == "cpu", "This regression requires JAX_PLATFORMS=cpu"
    dtype = get_dtype_jnp("ao_eval")
    controls, weights = _control_inputs(dtype)
    contract = AOs_bspline_data._contract_control_points
    bulk = jax.jit(jax.vmap(contract))(controls, weights[:, 0], weights[:, 1], weights[:, 2])
    single = jax.jit(contract)
    reference = np.stack([np.asarray(single(controls[i], weights[i, 0], weights[i, 1], weights[i, 2])) for i in range(17)])
    actual = np.asarray(bulk)
    assert actual.dtype == reference.dtype == np.dtype(dtype)
    assert actual.shape == reference.shape == (17, 1)
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    atol, rtol = get_tolerance("ao_eval", "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)


def test_fp32_real_quintic_control_contraction_batch_single_bits():
    """Explicit fp32 CPU kernel inputs give identical batch/single output bits.

    This unit test exercises the dtype-dispatched fp32 branch in both global
    precision modes. It does not claim that full-mode AO production uses fp32,
    and does not compare outputs from different modes or input dtypes.
    Geometry, weight generation, and whole `_eval_core` are outside its scope.
    """
    assert jax.default_backend() == "cpu", "This regression requires JAX_PLATFORMS=cpu"
    controls, weights = _control_inputs(jnp.float32)
    contract = AOs_bspline_data._contract_control_points
    bulk = jax.jit(jax.vmap(contract))(controls, weights[:, 0], weights[:, 1], weights[:, 2])
    single = jax.jit(contract)
    reference = np.stack([np.asarray(single(controls[i], weights[i, 0], weights[i, 1], weights[i, 2])) for i in range(17)])
    actual = np.asarray(bulk)
    assert controls.dtype == weights.dtype == actual.dtype == reference.dtype == np.dtype(np.float32)
    assert actual.shape == reference.shape == (17, 1)
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    actual_bits = actual.view(np.uint32)
    reference_bits = reference.view(np.uint32)
    np.testing.assert_array_equal(actual_bits, reference_bits)
