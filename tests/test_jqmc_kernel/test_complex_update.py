"""Complex-safe indexed updates (jqmc._complex_update) used by the streaming MCMC/LRDMC state updates.

Under ``vmap`` with a per-walker index, every helper is bit-identical to the native
``x.at[idx].set(v)`` / ``lax.dynamic_update_slice`` for complex128, complex64 and float64,
including signed zeros, infinities and NaN payload-free NaNs in the written values.
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
from jax import lax

from jqmc import _complex_update

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]

NW, N, K = 7, 5, 3
# Plain dtype names, so the collected IDs (e.g. ``col-complex128``) equal the GDD case binding.
DTYPES = ["complex128", "complex64", "float64"]


def _arr(rng, shape, dtype):
    a = rng.standard_normal(shape)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        a = a + 1j * rng.standard_normal(shape)
    return jnp.asarray(a, dtype=dtype)


def _with_special_values(v):
    """Put -0.0, +inf and nan into the update so component copies are checked exactly."""
    special = jnp.asarray([-0.0, jnp.inf, jnp.nan], dtype=jnp.float64)
    flat = v.reshape(v.shape[0], -1)
    if jnp.issubdtype(v.dtype, jnp.complexfloating):
        fill = lax.complex(special, special[::-1]).astype(v.dtype)
    else:
        fill = special.astype(v.dtype)
    return flat.at[:, : fill.shape[0]].set(fill).reshape(v.shape)


def _bits(x):
    x = np.asarray(x)
    return x.view(np.uint8)


CASES = {
    "col": (lambda m, i, v: m.at[:, i].set(v), lambda m, i, v: _complex_update.at_set(m, jnp.s_[:, i], v), (N, N), (N,)),
    "row": (lambda m, i, v: m.at[i, :].set(v), lambda m, i, v: _complex_update.at_set(m, jnp.s_[i, :], v), (N, N), (N,)),
    "plane": (
        lambda m, i, v: m.at[:, :, i].set(v),
        lambda m, i, v: _complex_update.at_set(m, jnp.s_[:, :, i], v),
        (3, N, N),
        (3, N),
    ),
    "block_dus": (
        lambda m, i, v: lax.dynamic_update_slice(m, v, (jnp.int32(0), i * K)),
        lambda m, i, v: _complex_update.dynamic_update_slice(m, v, (jnp.int32(0), i * K)),
        (N, N * K),
        (N, K),
    ),
}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", list(CASES))
def test_complex_update_is_bit_identical_to_native_under_vmap(case, dtype):
    """Each helper reproduces the native update bit for bit for a per-walker index."""
    native, helper, op_shape, val_shape = CASES[case]
    dtype = jnp.dtype(dtype)
    rng = np.random.default_rng(473)
    m = _arr(rng, (NW, *op_shape), dtype)
    v = _with_special_values(_arr(rng, (NW, *val_shape), dtype))
    idx = jnp.asarray(rng.integers(0, N, NW), dtype=jnp.int32)
    expected = jax.jit(jax.vmap(native))(m, idx, v)
    actual = jax.jit(jax.vmap(helper))(m, idx, v)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(_bits(actual), _bits(expected))
