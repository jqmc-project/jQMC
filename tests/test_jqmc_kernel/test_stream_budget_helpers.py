"""Regression tests for the measured stream-budget helpers of ``jqmc._setting``.

Pins the contracts of the single streaming policy (see the streaming header of
``jqmc._setting``):

* ``verified_stream_batch`` halves the selected batch until the COMPILED total
  fits the budget and keeps a fitting tentative;
* there is NO fallback width: a failed measurement or verification raises a
  ``RuntimeError`` naming the kernel (issue #483);
* ``measured_stream_batch`` returns ``n + 1`` when every element fits (plain vmap,
  no scan-of-one);
* ``measured_closure_stream_batch`` keys its memoisation by the converted kernel's
  jaxpr, so two closures with identical shapes but different math are measured
  separately;
* ``budget_map`` maps element ``i`` to ``f(x_i)``, chunked or not: for a kernel
  made of single correctly rounded IEEE operations (so no contraction whose XLA
  kernel may change with the batch shape) it equals the plain ``vmap`` bit for bit,
  including the padded tail and zero-size output leaves;
* ``derivatives.dtn._pair_chunks`` never pads the pair list beyond one chunk (the production
  kernels are checked in ``test_dtn_pair_chunk_width.py``).

All assertions are structural (routing decisions, exception contracts, exact
equality), so no tolerance constants are involved.
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

import jqmc._setting as setting_mod
from jqmc._setting import (
    STREAM_MIN_BATCH,
    budget_map,
    measured_bytes_per_config,
    measured_closure_stream_batch,
    measured_stream_batch,
    verified_stream_batch,
)

pytestmark = [pytest.mark.with_chdir]


def _toy_kernel(data, x):
    return jnp.sum(data["w"] * jnp.sum(x))


_TOY_DATA = {"w": np.ones(3)}
_TOY_SPECS = (jax.ShapeDtypeStruct((2, 3), np.float64),)


def _boom(fn, data, specs, batch, cache_identity=None):
    raise RuntimeError("memory_analysis unavailable")


@pytest.mark.light
def test_verified_stream_batch_halves_until_the_compiled_total_fits(monkeypatch):
    """Batch 32 with measured total 300 B against a 256 B budget -> 16."""
    totals = {32: 300, 16: 200}
    monkeypatch.setattr(
        setting_mod,
        "_measured_total_stream_bytes",
        lambda fn, data, specs, batch, cache_identity=None: totals[int(batch)],
    )
    batch = verified_stream_batch(_toy_kernel, _TOY_DATA, _TOY_SPECS, 32, 64, 256, STREAM_MIN_BATCH)
    assert batch == 16


@pytest.mark.light
def test_verified_stream_batch_keeps_a_fitting_tentative_and_degenerate(monkeypatch):
    """A fitting tentative is kept; a fitting monolithic keeps batch >= n."""
    monkeypatch.setattr(
        setting_mod,
        "_measured_total_stream_bytes",
        lambda fn, data, specs, batch, cache_identity=None: 200,
    )
    assert verified_stream_batch(_toy_kernel, _TOY_DATA, _TOY_SPECS, 32, 64, 256, STREAM_MIN_BATCH) == 32
    # degenerate contract: tentative >= n_total and the monolithic compile fits
    # -> the tentative is returned unchanged so the plain-vmap branch triggers.
    assert verified_stream_batch(_toy_kernel, _TOY_DATA, _TOY_SPECS, 128, 64, 256, STREAM_MIN_BATCH) == 128


@pytest.mark.light
def test_verified_stream_batch_raises_when_probing_fails(monkeypatch):
    """No fallback: a failing verification probe raises and names the kernel."""
    monkeypatch.setattr(setting_mod, "_measured_total_stream_bytes", _boom)
    with pytest.raises(RuntimeError, match="_toy_kernel"):
        verified_stream_batch(_toy_kernel, _TOY_DATA, _TOY_SPECS, 64, 128, 1 << 40, STREAM_MIN_BATCH)


@pytest.mark.light
def test_measured_bytes_per_config_raises_when_probing_fails(monkeypatch):
    """No fallback: a failing measurement raises (it does not return None and stream at
    STREAM_MIN_BATCH silently)."""
    monkeypatch.setattr(setting_mod, "_measured_total_stream_bytes", _boom)
    with pytest.raises(RuntimeError, match="_toy_kernel"):
        measured_bytes_per_config(_toy_kernel, _TOY_DATA, _TOY_SPECS)


@pytest.mark.light
def test_measured_stream_batch_degenerates_to_n_plus_one_when_everything_fits():
    """A budget that holds every element returns n + 1 (plain vmap, never a scan-of-one)."""
    n = 40
    assert measured_stream_batch(_toy_kernel, _TOY_DATA, _TOY_SPECS, n, 1 << 40) == n + 1


@pytest.mark.light
def test_measured_closure_stream_batch_keys_by_jaxpr_not_qualname(monkeypatch):
    """Two closures with the same shapes but different math are measured separately."""
    seen = []
    real = setting_mod._measured_total_stream_bytes

    def spy(fn, data, specs, batch, cache_identity=None):
        seen.append(cache_identity)
        return real(fn, data, specs, batch, cache_identity)

    monkeypatch.setattr(setting_mod, "_measured_total_stream_bytes", spy)
    a = jnp.ones((4, 4))
    x = jnp.ones(4)
    measured_closure_stream_batch(lambda v: a @ v, (x,), 8, 1 << 40)
    measured_closure_stream_batch(lambda v: jnp.tanh(a @ v), (x,), 8, 1 << 40)
    identities = {i for i in seen if i is not None}
    assert len(identities) == 2


@pytest.mark.light
@pytest.mark.parametrize("budget", [1 << 40, 1], ids=["one_vmap", "chunked"])
def test_budget_map_matches_vmap_bitwise_including_zero_size_leaves(budget):
    """``budget_map`` returns ``f(x_i)`` for every element, in order, with or without chunking
    (13 elements, so the chunked case pads its tail), and keeps a zero-size output leaf (the
    case ``jax.lax.map(batch_size=...)`` cannot collect).

    Every output element is one correctly rounded IEEE operation on the inputs (add, multiply,
    select), so its value cannot depend on how XLA vectorises the batch and the comparison is
    exact. A contraction (``a @ u``) is deliberately absent: XLA may pick a different dot
    kernel for a different batch shape, so chunked and unchunked contractions agree only to
    fp64 round-off (see the streaming header of ``jqmc._setting``); that is not a property of
    the chunking.
    """
    rng = np.random.default_rng(483)
    xs = (jnp.asarray(rng.standard_normal((13, 5))), jnp.asarray(rng.standard_normal((13, 3))))

    def f(u, w):
        return {
            "y": jnp.where(u > 0, u * w[0], u + w[1]),
            "w": w[::-1] * 2.0,
            "empty": jnp.zeros((0, 2)) + u[0],
        }

    expected = jax.vmap(f)(*xs)
    actual = budget_map(f, xs, budget)
    np.testing.assert_array_equal(np.asarray(actual["y"]), np.asarray(expected["y"]))
    np.testing.assert_array_equal(np.asarray(actual["w"]), np.asarray(expected["w"]))
    assert actual["empty"].shape == expected["empty"].shape == (13, 0, 2)


@pytest.mark.light
def test_dtn_pair_chunks_never_pad_beyond_one_chunk():
    """A pair list smaller than the budget width is one chunk of exactly n_pair pairs."""
    from jqmc.derivatives.dtn import _pair_chunks

    ia, ib = np.triu_indices(8, 1)  # 28 pairs
    pos = jnp.asarray(np.random.default_rng(1).standard_normal((8, 3)))

    def body(carry, xs):
        ra, rb, m = xs
        return carry + jnp.sum(jnp.linalg.norm(ra - rb, axis=-1) * m), None

    xs = _pair_chunks(body, jnp.zeros(()), lambda iaC, ibC, mask: (pos[iaC], pos[ibC], mask), ia, ib)
    assert xs[0].shape[:2] == (1, ia.shape[0])
    assert float(jnp.sum(xs[2])) == ia.shape[0]
