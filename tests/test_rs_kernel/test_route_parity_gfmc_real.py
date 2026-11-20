"""Layer 2: PRODUCTION ROUTE PARITY for the REAL GFMC interval.

``test_route_parity_gfmc_complex.py`` promoted the complex ``dltmove`` cell to a driver-driven,
spy-observed route-parity test -- but only for ``run_gfmc_projection_{n,t}_complex``,
which carries a SINGLE discriminating argument (``dltmove``; the complex chunk is
energy-only, so it has no ``want_force`` / ``want_stress``). The REAL
``run_gfmc_projection_{n,t}`` carries the remaining ``DISCRIMINATING_ARGS`` that the
free-function layer cannot see reach production:

  * ``dltmove``   -- the DLA non-local move (the complex cell's exact argument);
  * ``want_force`` / ``want_stress`` -- the force/stress fold-in switches; a chunk that
                     silently left them at their Rust default ``false`` would compute an
                     energy-only interval while the driver believed it folded the force in,
                     and the free-function force kernels would stay green throughout.

MPC is a PASSIVE, energy-only observable computed separately from the projection (it
never substitutes into e_L and the projection interval takes no MPC-related argument at
all), so it is out of scope for this module -- see
``tests/test_rs_kernel/test_mpc_observable.py``.

This module drives the PRODUCTION ``GFMC_t`` / ``GFMC_n`` and asserts, via the ``_route_spy``
seam, that each argument is present on the PyO3 signature, was explicitly SUPPLIED (not left
at the Rust default), and carries the value the run asked for -- the four route-parity checks,
on the real path, mirroring the complex rows.

  * ``test_route_parity_real_gfmc_force`` -- periodic ECP force run (lithium, real-Gamma,
    ``comput_position_deriv=True``), with ``comput_stress`` explicitly off and on:
    ``dltmove`` carries the requested move, ``want_force`` is True, and
    ``want_stress`` carries the independent public stress selection.

The values are not re-verified here (that is the parity suite's job -- and the complex
module's value/DLA-separation row); this module is strictly the ROUTE seam for the real
interval, so it runs at the same ``heavy`` tier as its complex sibling.
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

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from jqmc._setting import (
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_STEPS,
)
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t

from tests.test_rs_kernel._route_spy import (
    assert_arg_reached,
    assert_native_ran,
    install_ctx_spy,
)
from tests.test_rs_kernel._rs_fixtures import rs_ham

# lithium: periodic real-Gamma ECP with a sphe-pGTO determinant -- force is chunk-eligible
# (pGTO stress leaf).
_PATTERN = "pbc_gamma_sphepgto_j12"
_STEPS = 2 * GFMC_MIN_STEPS
_ALAT = 0.30
_TAU = 0.05
_NSUB = 3

_HAM_CACHE: dict[str, object] = {}


def _ham():
    if _PATTERN not in _HAM_CACHE:
        _HAM_CACHE[_PATTERN] = rs_ham(_PATTERN)
    return _HAM_CACHE[_PATTERN]


def _run(driver, ham, seed, non_local_move, *, comput_position_deriv, comput_stress, mpc, num_steps=_STEPS):
    """One PRODUCTION native (fast) GFMC run. The backend is selected as a user selects
    it (env override at run() head); no driver internals are touched."""
    os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)  # native path on (default True)
    try:
        common = dict(
            hamiltonian_data=ham,
            num_walkers=2,
            num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
            mcmc_seed=seed,
            alat=_ALAT,
            non_local_move=non_local_move,
            random_discretized_mesh=True,
            comput_position_deriv=comput_position_deriv,
            comput_stress=comput_stress,
            mpc=mpc,
        )
        g = GFMC_n(**common, num_mcmc_per_measurement=_NSUB, E_scf=-1.0) if driver == "n" else GFMC_t(**common, tau=_TAU)
        g.run(num_mcmc_steps=num_steps)
        return g
    finally:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["t", "n"])
@pytest.mark.parametrize("non_local_move", ["tmove", "dltmove"])
@pytest.mark.parametrize("comput_stress", [False, True], ids=["force-only", "force-plus-stress"])
def test_route_parity_real_gfmc_force(driver, non_local_move, comput_stress, monkeypatch):
    """The real interval carries move, force, and independent stress selections.

    Production fans the projection over all walkers through the
    BATCHED entry (one call per branching step), so the spy sees the ``_batch`` method;
    it carries the same discriminating args as the per-walker entry."""
    method = f"run_gfmc_projection_{driver}_batch"
    ctx_tag = f"{driver}/{non_local_move}/{'force-plus-stress' if comput_stress else 'force-only'}"
    ham = _ham()

    spy = install_ctx_spy(monkeypatch)
    _run(
        driver,
        ham,
        34356,
        non_local_move,
        comput_position_deriv=True,
        comput_stress=comput_stress,
        mpc=False,
    )

    # CHECK 1: the native REAL interval ran (a real-family, periodic, GFMC+force ctx).
    assert_native_ran(spy, method, at_least=_STEPS, context=ctx_tag)
    assert bool(spy.build_arg("complex")) is False, f"{ctx_tag}: build_obs_ctx asked for the complex family on a real fixture"
    assert bool(spy.build_arg("pbc")) is True, f"{ctx_tag}: build_obs_ctx was not asked for the periodic family"
    assert bool(spy.build_arg("want_gfmc")) is True, f"{ctx_tag}: build_obs_ctx was not asked for the GFMC args"

    # CHECK 2: discriminating args on the real path.
    assert_arg_reached(spy, method, "dltmove", non_local_move == "dltmove", context=ctx_tag)
    assert_arg_reached(spy, method, "want_force", True, context=ctx_tag)
    assert_arg_reached(spy, method, "want_stress", comput_stress, context=ctx_tag)
