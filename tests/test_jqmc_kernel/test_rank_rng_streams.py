"""Exact, fixed-key contracts for rank and twist RNG namespace derivation."""

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

from jqmc._driver_helpers import derive_rank_rng_streams
from jqmc._twist_average import _twist_sampler_seed


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize(
    "chain_seed,rank,expected_words,expected_jax_seed",
    [
        (
            19,
            0,
            (3679625778, 1889802809, 3627577030, 4254277910, 1144274852, 2815092471, 762279905, 3083669192),
            4101467731423769312,
        ),
        (
            19,
            1,
            (4250031973, 3879196237, 2306656852, 812125730, 2665171916, 508956924, 2437443248, 1023196390),
            9161557028074524615,
        ),
        (
            (19, 0),
            0,
            (720877394, 1843548269, 2314172011, 3811946637, 935770851, 893628652, 3251818950, 4006904508),
            423416729108126653,
        ),
        (
            (19, 1),
            0,
            (2460824369, 2974050826, 717057198, 3277912726, 3190563392, 250019416, 1311046103, 939521497),
            6076952566297391971,
        ),
        (
            (1099511627793, 3),
            2,
            (4161745771, 2921049047, 652907074, 4219742319, 2310194982, 2523794482, 2283163502, 3206201809),
            7970302719876562814,
        ),
    ],
)
def test_rank_rng_derivation_matches_fixed_vectors(chain_seed, rank, expected_words, expected_jax_seed):
    """The eight MT19937 seed words and 63-bit JAX seed equal fixed compatibility vectors.

    Expected vectors were obtained directly from SeedSequence applied to the
    declared key plus rank and nonzero terminator, not from the production
    helper. They are literals so changing the implementation cannot silently
    change both sides. This pins namespace/word-layout behavior, not the
    statistical quality or universal independence of random streams.
    """
    words, jax_seed = derive_rank_rng_streams(chain_seed, rank)
    assert words.shape == (8,)
    assert words.dtype == np.dtype(np.uint32)
    np.testing.assert_array_equal(words, np.asarray(expected_words, dtype=np.uint32))
    assert type(jax_seed) is int
    assert jax_seed == expected_jax_seed
    assert 0 <= jax_seed < 9223372036854775808


def test_rank_twist_rng_namespaces_do_not_alias_on_selected_keys():
    """Selected scalar/rank and twist/rank keys remain distinct, including trailing-zero extensions."""
    base_seed = 19
    keys = [(base_seed, rank) for rank in range(4)]
    keys.extend(((base_seed, twist), rank) for twist in range(4) for rank in range(4))
    streams = [derive_rank_rng_streams(seed, rank) for seed, rank in keys]
    numpy_streams = [tuple(int(word) for word in words) for words, _ in streams]
    jax_streams = [jax_seed for _, jax_seed in streams]
    assert len(set(numpy_streams)) == len(keys)
    assert len(set(jax_streams)) == len(keys)
    # Without a nonzero terminator these two distinct namespaces collide in
    # SeedSequence: appending zero entropy words alone does not separate keys.
    scalar_words, scalar_jax = derive_rank_rng_streams(base_seed, 1)
    twist_words, twist_jax = derive_rank_rng_streams((base_seed, 1), 0)
    assert not np.array_equal(scalar_words, twist_words)
    assert scalar_jax != twist_jax


@pytest.mark.parametrize("representation", ["tuple", "list", "array"])
@pytest.mark.parametrize("rank", [0, 3])
def test_rank_rng_checkpoint_seed_forms_and_single_twist_preserve_stream(representation, rank):
    """Restored seed representations and the single-twist wrapper preserve the scalar or tuple stream exactly."""
    scalar = 1099511627793
    wrapped = _twist_sampler_seed(scalar, 0, 1)
    assert wrapped == scalar
    expected_words, expected_jax = derive_rank_rng_streams(scalar, rank)
    actual_words, actual_jax = derive_rank_rng_streams(wrapped, rank)
    np.testing.assert_array_equal(actual_words, expected_words)
    assert actual_jax == expected_jax
    for key in ((scalar,), (scalar, 0), (scalar, 2)):
        restored = (
            key if representation == "tuple" else list(key) if representation == "list" else np.asarray(key, dtype=np.int64)
        )
        expected_words, expected_jax = derive_rank_rng_streams(key, rank)
        actual_words, actual_jax = derive_rank_rng_streams(restored, rank)
        np.testing.assert_array_equal(actual_words, expected_words)
        assert actual_jax == expected_jax
