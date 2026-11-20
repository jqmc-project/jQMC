"""Specified Threefry inputs and independent inverse-Gaussian references."""

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
import scipy.special


def gaussian_inputs(key_words, count):
    """Generate the declared fp64 input stream without calling native RNG code."""
    assert jax.config.x64_enabled, "native Gaussian inputs are fp64 in both precision modes"
    key = jnp.asarray(key_words, dtype=jnp.uint32)
    subkeys = jax.random.split(key, count)
    uniforms = jax.vmap(
        lambda subkey: jax.random.uniform(subkey, (), dtype=jnp.float64, minval=np.nextafter(-1.0, 0.0), maxval=1.0)
    )(subkeys)
    return subkeys, np.asarray(uniforms)


def gaussian_inverse_reference(uniforms):
    """SciPy evaluates the inverse function, not native's piecewise polynomial."""
    return np.sqrt(2.0) * scipy.special.erfinv(uniforms)


def gaussian_xla_values(subkeys):
    """Evaluate the XLA transform for exactly the same fp64 subkey sequence."""
    return np.asarray(jax.vmap(lambda key: jax.random.normal(key, (), dtype=jnp.float64))(subkeys))
