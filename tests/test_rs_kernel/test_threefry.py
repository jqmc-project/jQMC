"""Bit-exact equivalence: jqmc_rs threefry RNG vs jax.random.

Validates the Rust Threefry-2x32 port and the split / randint / uniform
transforms against JAX's own PRNG (partitionable config) to the LAST BIT.
This underpins the bit-exact full-chain determinism option for mcmc_update;
the per-move correctness gate uses random *injection* (layer 2) and does not
depend on this, but matching exactly gives optional deterministic regression.

No Hamiltonian is needed anywhere in this file -- these tests only compare
Rust RNG primitives against ``jax.random`` directly.
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

# Every rs test declares its cost tier (light/heavy, per test) and CWD policy
# (with_chdir here -- autouse _isolate_cwd chdirs into a per-test tmp_path for
# xdist safety; this file touches no filesystem fixtures at all so cwd is
# irrelevant).
pytestmark = pytest.mark.with_chdir

from jax import random
from jax import numpy as jnp
import jax.extend.random as P  # stable public extension API (jax._src.prng moved/removed across versions)

KEYS = [0, 1, 7, 42, 123, 2024, 999983]


def _key_words(seed):
    k = np.asarray(random.PRNGKey(seed), dtype=np.uint32)
    return int(k[0]), int(k[1])


@pytest.mark.light
@pytest.mark.parametrize("seed", KEYS)
def test_threefry_core_matches_jax(seed):
    """Threefry core matches jax."""
    k0, k1 = _key_words(seed)
    for c0, c1 in [(0, 0), (0, 1), (5, 9), (0xDEADBEEF, 0x12345678), (2**32 - 1, 7)]:
        ref = np.asarray(
            P.threefry_2x32(
                jnp.array([k0, k1], dtype=jnp.uint32),
                jnp.array([c0, c1], dtype=jnp.uint32),
            ),
            dtype=np.uint32,
        )
        got = jqmc_rs._tf_threefry(k0, k1, c0, c1)
        assert (int(got[0]), int(got[1])) == (int(ref[0]), int(ref[1])), (got, ref)


@pytest.mark.light
@pytest.mark.parametrize("seed", KEYS)
@pytest.mark.parametrize("n", [1, 2, 3, 8, 17])
def test_split_matches_jax(seed, n):
    """Split matches jax."""
    k0, k1 = _key_words(seed)
    ref = np.asarray(random.split(random.PRNGKey(seed), n), dtype=np.uint32)
    got = np.asarray(jqmc_rs._tf_split(k0, k1, n), dtype=np.uint32)
    assert np.array_equal(got, ref), (got, ref)


@pytest.mark.light
@pytest.mark.parametrize("seed", KEYS)
@pytest.mark.parametrize("lo,hi", [(0, 3), (0, 96), (0, 48), (-5, 5), (0, 2), (10, 1000)])
def test_randint_i32_matches_jax(seed, lo, hi):
    """Randint i32 matches jax."""
    k0, k1 = _key_words(seed)
    n = 64
    ref = np.asarray(
        random.randint(random.PRNGKey(seed), (n,), lo, hi, dtype=jnp.int32),
        dtype=np.int32,
    )
    got = np.asarray(jqmc_rs._tf_randint_i32(k0, k1, n, lo, hi), dtype=np.int32)
    assert np.array_equal(got, ref), (got[:8], ref[:8])


@pytest.mark.light
@pytest.mark.parametrize("seed", KEYS)
@pytest.mark.parametrize("lo,hi", [(0, 3), (0, 96), (0, 48), (-5, 5), (0, 2), (10, 1000)])
def test_randint_i64_default_matches_jax(seed, lo, hi):
    # jqmc uses the DEFAULT int dtype (int64 under x64). This is the path the
    # MCMC move driver threads, so it must match bit-for-bit including for
    # split-derived keys (high word != 0).
    """Randint i64 default matches jax."""
    k0, k1 = _key_words(seed)
    n = 64
    ref = np.asarray(random.randint(random.PRNGKey(seed), (n,), lo, hi), dtype=np.int64)
    got = np.asarray(jqmc_rs._tf_randint_i64(k0, k1, n, lo, hi), dtype=np.int64)
    assert np.array_equal(got, ref), (got[:8], ref[:8])


@pytest.mark.light
@pytest.mark.parametrize("seed", KEYS)
@pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (-1.0, 1.0), (2.0, 5.0)])
def test_uniform_f64_matches_jax(seed, lo, hi):
    """Uniform f64 matches jax."""
    k0, k1 = _key_words(seed)
    n = 64
    ref = np.asarray(
        random.uniform(random.PRNGKey(seed), (n,), dtype=jnp.float64, minval=lo, maxval=hi),
        dtype=np.float64,
    )
    got = np.asarray(jqmc_rs._tf_uniform_f64(k0, k1, n, lo, hi), dtype=np.float64)
    # Bit-exact expected; array_equal on the raw bits (no tolerance needed).
    assert np.array_equal(got, ref), np.abs(got - ref).max()


# ---------------------------------------------------------------------------
# Native-RNG boundary (documented, not a bit-exact gate). The Gaussian proposal
# normal (``jax.random.normal = sqrt(2)*erf_inv(uniform)``) and the ECP-mesh RT
# rotation (``_generate_rotation_matrix``) CANNOT be reproduced natively bit-for-bit
# against XLA CPU: ``erf_inv`` routes through XLA's own ``log1p`` polynomial (differs
# from libm at ~12% of points by 1 ULP), and both the ``uniform`` scale and the
# rotation-matrix element expressions are fp-CONTRACTED into FMAs by XLA (differ from
# Rust's separate mul/add by ~0.5/0.25 ULP). All of this is backend/version dependent.
# A 1-ULP proposal perturbation flips a downstream Metropolis accept, so an
# "approximate" native normal/RT is NOT chain-identical. The native driver therefore
# INJECTS the JAX-generated normals + RTs per chunk (already the production dispatch
# pattern). The bit-exact native primitives below (threefry/split/randint/uniform)
# cover every DISCRETE chain decision; this note pins the boundary of what is native.
# See ``threefry.rs`` module docs for the measured evidence.
# ---------------------------------------------------------------------------
