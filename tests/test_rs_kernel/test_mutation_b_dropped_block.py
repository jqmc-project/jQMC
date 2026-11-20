"""Route-parity mutation (b), PERMANENT: a dropped return block must turn a route-parity row RED.

Mutations (a) and (c) -- an argument the driver stops passing, and a capability gate that
re-routes the run to JAX -- are edits to production source, so they are run as one-off
experiments and recorded in the commit message. Mutation (b) is different: "the PyO3
wrapper stopped emitting a block" cannot be produced by editing Python at all, it needs a
rebuilt extension. It CAN be simulated exactly at the seam, so it is kept here as a
standing test of the DETECTOR rather than a one-off.

Without this, ``_route_spy.assert_return_blocks`` / the block assertions in
``test_route_parity_observables`` are unfalsified code: they would look like coverage
while never having been shown to fire.
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

import os

import pytest

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

from jqmc._setting import MCMC_MIN_STEPS
from jqmc.jqmc_mcmc import MCMC

from tests.test_rs_kernel._route_spy import assert_return_blocks, install_ctx_spy
from tests.test_rs_kernel._rs_fixtures import rs_ham

_DROPPED = "geminal_inv"


def _run_with_dropped_block(monkeypatch, drop: str | None):
    """Run the production driver, optionally deleting ``drop`` from every
    ``run_energy_chunk`` return before the driver (and the spy) ever see it."""
    import jqmc_rs

    ham = rs_ham("mol_ecp_cart_j12")
    orig_build = jqmc_rs.build_obs_ctx

    class _Mutating:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            attr = getattr(self._inner, name)
            if name != "run_energy_chunk_batch" or drop is None or not callable(attr):
                return attr

            def _mutated(*a, **k):
                out = dict(attr(*a, **k))
                out.pop(drop, None)
                return out

            return _mutated

    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", lambda *a, **k: _Mutating(orig_build(*a, **k)))
    spy = install_ctx_spy(monkeypatch)  # wraps the mutating proxy -> records what the DRIVER sees

    os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    try:
        m = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=34356,
            epsilon_AS=1.0e-6,
            num_walkers=2,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )
        m.run(num_mcmc_steps=MCMC_MIN_STEPS)
    finally:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    return spy


def test_baseline_block_set_is_complete(monkeypatch):
    """Unmutated: the chunk emits the blocks the driver unpacks by name."""
    spy = _run_with_dropped_block(monkeypatch, drop=None)
    calls = spy.calls_to("run_energy_chunk_batch")
    assert calls, f"the native chunk did not run: {spy.summary()}"
    blocks = tuple(k for k, _ in calls[0].ret[1])
    assert _DROPPED in blocks, f"baseline chunk return lacks {_DROPPED!r}: {blocks}"
    # Pin the exact block tuple so an ADDED block is also a visible change.
    assert_return_blocks(spy, "run_energy_chunk_batch", blocks, context="baseline")


def test_dropping_a_return_block_is_detected(monkeypatch):
    """MUTATION (b): the block assertion must FAIL on a return that lost a block.

    The mutated descriptor is built from a REAL recorded call, so this is the genuine
    shape a Rust wrapper would produce, not a hand-written fixture. A pass here would
    mean the return-structure check is decorative."""
    spy = _run_with_dropped_block(monkeypatch, drop=None)
    call = spy.calls_to("run_energy_chunk_batch")[0]
    expected = tuple(k for k, _ in call.ret[1])
    assert _DROPPED in expected

    # Same recorded call, one block removed -- exactly what a wrapper that stopped
    # emitting ``geminal_inv`` would hand back.
    for c in spy.calls_to("run_energy_chunk_batch"):
        c.ret = ("dict", tuple((k, v) for k, v in c.ret[1] if k != _DROPPED))
    with pytest.raises(AssertionError, match="returned blocks"):
        assert_return_blocks(spy, "run_energy_chunk_batch", expected, context="mutation-b")


def test_production_itself_rejects_a_dropped_block(monkeypatch):
    """MUTATION (b), second half: the DRIVER must not silently absorb the loss either.

    Measured: with ``geminal_inv`` deleted from every ``run_energy_chunk`` return, the
    production chunk loop dies at ``jqmc_mcmc.py`` with ``KeyError: 'geminal_inv'``. That
    is the desirable behaviour -- it is recorded here so that a future refactor which
    starts tolerating a missing block (a ``.get(..., None)``, say) fails this test instead
    of quietly propagating a None inverse into the next step's Sherman-Morrison carry.
    """
    with pytest.raises(KeyError, match=_DROPPED):
        _run_with_dropped_block(monkeypatch, drop=_DROPPED)
