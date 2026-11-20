"""Indexed updates of complex128 arrays that stay parallel on XLA:GPU.

XLA:GPU lowers an overwriting ``scatter`` on ``complex128`` data whose
indices differ across a ``vmap`` batch (``x.at[:, i].set(col)`` or
``lax.dynamic_update_slice`` with a per-walker index) to a sequential
``while`` loop over the batch, one tiny kernel launch per walker; the same
update on ``float64`` (and a complex ``.add``) stays a parallel scatter. In
the streaming MCMC/LRDMC state updates this loop runs for every moved
electron and dominated the GPU time of complex (twisted) wave functions.

The helpers below perform the same overwrite on the real and imaginary
``float64`` parts separately and recombine them with :func:`jax.lax.complex`.
Splitting, writing and recombining copies the IEEE components unchanged, so
the result is bit-identical to the native update. Arrays whose elements are
64 bits or narrower (real ``float64``, ``complex64``, integers) take the
native path unchanged.
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

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax


def _is_wide_complex(x: jax.Array) -> bool:
    """True for complex dtypes whose elements are wider than 64 bits (complex128)."""
    dtype = jnp.dtype(x.dtype)
    return jnp.issubdtype(dtype, jnp.complexfloating) and dtype.itemsize > 8


def at_set(x: jax.Array, index: Any, value: jax.Array) -> jax.Array:
    """Return ``x.at[index].set(value)``; complex128 is updated per real/imag part."""
    if not _is_wide_complex(x):
        return x.at[index].set(value)
    v = jnp.asarray(value).astype(x.dtype)
    return lax.complex(x.real.at[index].set(v.real), x.imag.at[index].set(v.imag))


def dynamic_update_slice(x: jax.Array, update: jax.Array, start_indices: Any) -> jax.Array:
    """Return ``lax.dynamic_update_slice(x, update, start_indices)``; complex128 per real/imag part."""
    if not _is_wide_complex(x):
        return lax.dynamic_update_slice(x, update, start_indices)
    u = jnp.asarray(update).astype(x.dtype)
    return lax.complex(
        lax.dynamic_update_slice(x.real, u.real, start_indices),
        lax.dynamic_update_slice(x.imag, u.imag, start_indices),
    )
