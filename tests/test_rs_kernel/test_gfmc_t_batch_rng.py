"""Conventional GFMC preserves the XLA batch-wide carried RNG convention."""

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

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc.jqmc_gfmc as gfmc
import jqmc_rs
from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_ZERO
from tests.test_rs_kernel._rs_fixtures import _PATTERNS, rs_ham
from tests.test_rs_kernel.test_gfmc_chain_modes import _assert_value_parity
from tests.test_rs_kernel.test_route_parity_gfmc_complex import _run

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]


def _record_intervals(monkeypatch):
    """Observe real driver calls without replacing any result or RNG key."""
    xla, native = [], []
    original_jit = gfmc.jit
    original_build = jqmc_rs.build_obs_ctx

    def recording_jit(fun=None, **options):
        if fun is None:
            return functools.partial(recording_jit, **options)
        compiled = original_jit(fun, **options)
        if fun.__name__ != "_run_projection_loop_streaming":
            return compiled

        @functools.wraps(compiled)
        def call(*args, **kwargs):
            result = compiled(*args, **kwargs)
            jax.block_until_ready(result)
            xla.append((args, result))
            return result

        call.lower = compiled.lower
        return call

    class RecordingContext:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            method = getattr(self.inner, name)
            if name not in ("run_gfmc_projection_t_batch", "run_gfmc_projection_t_complex_batch"):
                return method

            def call(*args, **kwargs):
                inputs = tuple(value.copy() if isinstance(value, np.ndarray) else value for value in args)
                result = method(*args, **kwargs)
                native.append((name, inputs, result, method, kwargs.copy()))
                return result

            return call

    monkeypatch.setattr(gfmc, "jit", recording_jit)
    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", lambda *args, **kwargs: RecordingContext(original_build(*args, **kwargs)))
    return xla, native


def _advance_keys(keys, splits):
    """Exact first-child Threefry continuation, independent of native padding."""
    keys = jnp.asarray(keys, dtype=jnp.uint32)
    for _ in range(splits):
        keys = jax.vmap(lambda key: jax.random.split(key)[0])(keys)
    return keys


def _assert_native_batch_boundaries(record):
    """Padding changes keys only; singleton, equal-count and zero-work keys retain their meanings."""
    _, inputs, batch, method, options = record
    singles = []
    for walker in range(2):
        one = list(inputs)
        for index in (0, 1, 4):
            one[index] = inputs[index][walker : walker + 1].copy()
        result = method(*one, **options)
        np.testing.assert_array_equal(result["final_key"], _advance_keys(one[4], 3 * int(result["n_substeps"][0])))
        singles.append(result)
    for name in ("r_up", "r_dn", "e_L", "w_L", "n_substeps"):
        np.testing.assert_array_equal(batch[name], np.concatenate([result[name] for result in singles], axis=0))

    equal = list(inputs)
    for index in (0, 1, 4):
        equal[index] = np.repeat(inputs[index][:1], 2, axis=0)
    equal_result = method(*equal, **options)
    np.testing.assert_array_equal(equal_result["n_substeps"], np.repeat(singles[0]["n_substeps"], 2))
    np.testing.assert_array_equal(equal_result["final_key"], np.repeat(singles[0]["final_key"], 2, axis=0))

    zero = list(inputs)
    zero[3] = 0.0
    zero_result = method(*zero, **options)
    np.testing.assert_array_equal(zero_result["final_key"], inputs[4])
    np.testing.assert_array_equal(zero_result["n_substeps"], np.zeros(2, dtype=int))


@pytest.mark.parametrize(
    ("pattern", "fixture_path", "random_mesh", "non_local_move"),
    [
        ("pbc_gamma_sphepgto_j12", "tests/trexio_example_files/lithium.h5", False, "tmove"),
        ("pbc_gamma_sphepgto_j12", "tests/trexio_example_files/lithium.h5", True, "dltmove"),
        ("twist_sphepgto_j12", "tests/trexio_example_files/lithium_k025.h5", False, "dltmove"),
        ("twist_sphepgto_j12", "tests/trexio_example_files/lithium_k025.h5", True, "tmove"),
    ],
    ids=["real-fixed-tmove", "real-random-dltmove", "complex-fixed-dltmove", "complex-random-tmove"],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_conventional_batch_rng_matches_xla(pattern, fixture_path, random_mesh, non_local_move, monkeypatch):
    """Native conventional GFMC matches XLA's batch-wide RNG and three-interval walk.

    Committed lithium Gamma/non-Gamma pGTO ECP fixtures use SD with J1/J2.
    Two electron configurations are generated by the unchanged production
    initializer at seed 34356, without cond(G) screening. With alat=0.30,
    tau=0.05 and fixed twelve-point ECP quadrature, at least one interval must
    have unequal walker termination counts. Each carried key advances three
    splits per batch iteration, including after that walker's termination.
    Fixed/random interval meshes and tmove/dltmove are paired across real and
    complex native kernels. The oracle observes the unchanged XLA driver in
    the same precision mode; no statistical energy acceptance is used.
    """
    assert Path(fixture_path).name == _PATTERNS[pattern][0]
    ham = rs_ham(pattern)
    xla_calls, native_calls = _record_intervals(monkeypatch)
    xla = _run(
        "t", ham, seed=34356, mode="jax", non_local_move=non_local_move, num_steps=3, num_walkers=2, random_mesh=random_mesh
    )
    native = _run(
        "t", ham, seed=34356, mode="fast", non_local_move=non_local_move, num_steps=3, num_walkers=2, random_mesh=random_mesh
    )
    assert len(xla_calls) == len(native_calls) == 3
    assert any(np.ptp(np.asarray(result[1])) > 0 for _, result in xla_calls), "fixture never finishes walkers separately"
    _assert_native_batch_boundaries(native_calls[0])
    for (xargs, reference), (method, nargs, actual, _, _) in zip(xla_calls, native_calls, strict=True):
        assert ("complex" in method) == pattern.startswith("twist")
        np.testing.assert_array_equal(nargs[4], xargs[6])
        np.testing.assert_array_equal(actual["n_substeps"], reference[1])
        expected_keys = _advance_keys(xargs[6], 3 * int(np.max(np.asarray(reference[1]))))
        np.testing.assert_array_equal(reference[10], expected_keys)
        np.testing.assert_array_equal(
            actual["final_key"], expected_keys, err_msg="early-finished walker lost XLA batch RNG advances"
        )
        for name, index in (("e_L", 0), ("w_L", 3)):
            _assert_value_parity(actual[name], reference[index], name, periodic=True)
        for name, index in (("r_up", 4), ("r_dn", 5)):
            if random_mesh:
                np.testing.assert_allclose(
                    actual[name], reference[index], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
                )
            else:
                np.testing.assert_array_equal(actual[name], reference[index])
    for name in ("_stored_e_L", "_stored_w_L"):
        _assert_value_parity(getattr(native, name), getattr(xla, name), name, periodic=True)
    np.testing.assert_array_equal(native._jax_PRNG_key_list, xla._jax_PRNG_key_list)
