"""Move-driver skeleton: bit-exact RNG threading vs jqmc body_fun.

Replicates the exact ``jax.random`` split/randint/uniform sequence of the jqmc
MCMC body (jqmc_mcmc.py body_fun) and checks the Rust ``move_draws`` produces
the SAME electron selection, axis, and post-move key bit-for-bit. The Gaussian
proposal subkey is consumed in both (normal value injected separately). Also
checks the Kato f_l and T_ratio geometry against a numpy transcription.

No Hamiltonian is needed anywhere in this file -- these are pure-RNG /
pure-numpy formula checks, independent of any wavefunction or AO family.
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

import jqmc_rs
import numpy as np
import pytest

# Every rs test declares its cost tier (light/heavy, per test) and CWD policy
# (with_chdir here -- autouse _isolate_cwd chdirs into a per-test tmp_path for
# xdist safety; this file touches no filesystem fixtures at all so cwd is
# irrelevant).
pytestmark = pytest.mark.with_chdir

from jax import numpy as jnp
from jax import random

from jqmc._setting import ATOL_PYTHON_FP64_TIGHT, RTOL_PYTHON_FP64_ZERO


def _jqmc_move_rng(seed, n_up, n_dn, b_dtype):
    """Replicate the jqmc body_fun RNG draw sequence exactly."""
    key = random.PRNGKey(seed)
    total = n_up + n_dn
    key, sk = random.split(key)
    rand_num = int(random.randint(sk, (), 0, total))
    is_up = rand_num < n_up
    key, sk = random.split(key)
    up_index = int(random.randint(sk, (), 0, n_up))
    key, sk = random.split(key)
    dn_index = int(random.randint(sk, (), 0, n_dn))
    selected = up_index if is_up else dn_index
    key, sk = random.split(key)  # normal subkey, consumed (value injected)
    key, sk = random.split(key)
    axis = int(random.randint(sk, (), 0, 3))
    key, sk = random.split(key)
    b = float(random.uniform(sk, (), dtype=b_dtype, minval=0.0, maxval=1.0))
    return is_up, selected, axis, b, np.asarray(key, dtype=np.uint32)


SEEDS = [0, 1, 42, 2024, 777]
SHAPES = [(48, 48), (16, 16), (7, 7), (2, 1), (1, 1)]


@pytest.mark.light
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_up,n_dn", SHAPES)
def test_move_draws_bit_exact(seed, n_up, n_dn):
    """Move draws bit exact."""
    k = np.asarray(random.PRNGKey(seed), dtype=np.uint32)
    is_up, sel, axis, b, fkey, fkey1 = jqmc_rs._move_draws(int(k[0]), int(k[1]), n_up, n_dn)
    # jqmc body uses fp64 uniform in x64 mode for the acceptance compare.
    ref_is_up, ref_sel, ref_axis, ref_b, ref_key = _jqmc_move_rng(seed, n_up, n_dn, jnp.float64)
    # Structural (integer / key) draws must be bit-exact.
    assert is_up == ref_is_up, (is_up, ref_is_up)
    assert sel == ref_sel, (sel, ref_sel)
    assert axis == ref_axis, (axis, ref_axis)
    assert (int(fkey), int(fkey1)) == (int(ref_key[0]), int(ref_key[1])), "post-move key mismatch"
    # Metropolis uniform b must match the fp64 draw exactly.
    assert b == ref_b, (b, ref_b)


@pytest.mark.light
def test_kato_f_l_and_t_ratio_numpy():
    """Kato f l and t ratio numpy."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        comparison_iteration = str(_)  # noqa: F841 - captured by the GDD iteration receipt
        z = float(rng.uniform(1.0, 8.0))
        r = float(rng.uniform(0.0, 3.0))
        f = 1.0 / z**2 * (1.0 + z**2 * r) / (1.0 + r)
        assert np.isclose(jqmc_rs._kato_f_l(z, r), f, rtol=RTOL_PYTHON_FP64_ZERO, atol=ATOL_PYTHON_FP64_TIGHT)
    for _ in range(20):
        comparison_iteration = str(_)
        fl = float(rng.uniform(0.1, 1.0))
        fp = float(rng.uniform(0.1, 1.0))
        d2 = float(rng.uniform(0.0, 1.0))
        dt = float(rng.uniform(0.5, 2.0))
        ref = (fl / fp) * np.exp(-d2 * (1.0 / (2 * fp**2 * dt**2) - 1.0 / (2 * fl**2 * dt**2)))
        assert np.isclose(jqmc_rs._t_ratio(fl, fp, d2, dt), ref, rtol=RTOL_PYTHON_FP64_ZERO, atol=ATOL_PYTHON_FP64_TIGHT)


@pytest.mark.light
def test_metropolis_accept():
    """Metropolis accept."""
    assert jqmc_rs._metropolis_accept(2.0, 1.0, 0.5) is True
    assert jqmc_rs._metropolis_accept(0.4, 1.0, 0.5) is False
    assert jqmc_rs._metropolis_accept(1.0, 1.0, 0.999999) is True
