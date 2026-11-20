"""Layer 2: PRODUCTION ROUTE PARITY for the energy chunk, e_L, force and stress.

Layer 1 pins the FORMULAS of ``run_energy_chunk`` / ``e_L`` / ``vmc_grad_*``. This module
pins that the production ``MCMC`` driver REACHES them, with the discriminating arguments
actually threaded through and the expected block structure coming back.

WHICH ROUTE A RUN TAKES IS NOT OBVIOUS, WHICH IS THE POINT
----------------------------------------------------------
``jqmc_mcmc.run()`` selects between a FUSED native chunk (``run_energy_chunk``: sweep +
e_L, optionally + force/stress, one call per walker per chunk) and a PER-STEP dispatch
(``mcmc_sweep`` + ``e_L`` + the ``grad_*`` cores individually). The selector
(``_rs_energy_chunk``) turns on chain mode, complexity, position derivatives AND param
derivatives. A test that just "runs MCMC and passes" therefore has no idea which of the
two families of ctx methods it exercised -- exactly the blindness this module is about.

So every row here DECLARES the methods it expects and the spy enforces it, in both
directions: the expected methods must fire, and (for the chunk rows) the per-step methods
must NOT, so a silent re-routing shows up as a failure rather than as continued green.

REACHING THE PER-STEP REAL CORES
--------------------------------
``run_energy_chunk`` swallows the real path whenever it can, so ``e_L`` / ``grad_e_L_ae``
/ ``grad_e_L_ecp`` / ``grad_ln_psi`` / the ``*_pbc`` cores are only reachable with a param
derivative requested (``_rs_energy_chunk`` requires ``not comput_*_param_deriv``). The
param-deriv rows below therefore switch it on purely as the MECHANISM that forces the
per-step dispatch; their param-deriv VALUES are asserted in
``test_route_parity_param_deriv.py``, not here, so the two modules fail independently.

DISCRIMINATING ARGUMENTS (manifest)
-----------------------------------
  * ``geminal_inv`` -- the sweep's running Sherman-Morrison inverse. Every real-path
    observable consumes it; a fresh LU rebuild instead is both slower and NaN-prone near
    singularity. Asserted non-None wherever the signature carries it.
  * ``want_force`` / ``want_stress`` -- independent chunk fold-in flags. Stress is
    periodic-only and must also be explicitly requested.

MPC is a PASSIVE, energy-only observable computed separately (never substituted into
e_L), so ``run_energy_chunk`` / ``e_L`` / the ``grad_*`` cores carry no MPC-related argument
at all -- ``mpc=True`` on the driver is exercised here only to confirm it does not perturb
the chunk route or the ``geminal_inv`` threading; the MPC value itself is covered by
``tests/test_rs_kernel/test_mpc_observable.py``.
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
    MCMC_MIN_STEPS,
)
from jqmc.jqmc_mcmc import MCMC

from tests.test_rs_kernel._route_spy import assert_arg_is_not_none, assert_arg_reached, assert_native_ran, install_ctx_spy
from tests.test_rs_kernel._rs_fixtures import rs_ham

_HAM_CACHE: dict[str, object] = {}


def _ham(pattern):
    if pattern not in _HAM_CACHE:
        _HAM_CACHE[pattern] = rs_ham(pattern)
    return _HAM_CACHE[pattern]


def _run(ham, rust_on, *, pos_deriv=False, stress=False, param_deriv=False, mpc=False, steps=MCMC_MIN_STEPS):
    if rust_on:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    else:
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    try:
        m = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=34356,
            epsilon_AS=1.0e-6,
            num_walkers=2,
            comput_position_deriv=pos_deriv,
            comput_stress=stress,
            comput_log_WF_param_deriv=param_deriv,
            comput_e_L_param_deriv=param_deriv,
            random_discretized_mesh=True,
            mpc=mpc,
        )
        m.run(num_mcmc_steps=steps)
        return m
    finally:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)


# --------------------------------------------------------------------------- #
# ROW TABLE. Each row declares the route it expects; the spy enforces both
# directions (expected methods fired, forbidden methods did not).
#
# ``args``   -> {method: {arg: expected}}, expected True/False = an explicit value,
#               "DATA" = must be non-None (not left at the None default).
# ``forbid`` -> methods that must NOT fire on this route.
# --------------------------------------------------------------------------- #
_CHUNK_PERSTEP = ("mcmc_sweep", "mcmc_sweep_complex", "e_L", "e_L_complex")
_CHUNK = ("run_energy_chunk", "run_energy_chunk_batch")

_ROWS = [
    dict(
        id="chunk_energy_mol_ecp",
        pattern="mol_ecp_cart_j12",
        expect=["run_energy_chunk_batch"],
        forbid=_CHUNK_PERSTEP,
        # This is a real route; the Rust call must receive the real-path default.
        args={"run_energy_chunk_batch": {"geminal_inv": "DATA", "want_force": False, "want_stress": False}},
        heavy=False,
    ),
    dict(
        id="chunk_force_mol_ecp",
        pattern="mol_ecp_cart_j12",
        pos_deriv=True,
        expect=["run_energy_chunk_batch"],
        forbid=_CHUNK_PERSTEP,
        # Stress is periodic-only (``_rs_chunk_stress = _rs_chunk_force and _rs_pbc``),
        # so a molecular force run must fold force in and stress OUT.
        args={"run_energy_chunk_batch": {"geminal_inv": "DATA", "want_force": True, "want_stress": False}},
        heavy=False,
    ),
    dict(
        id="chunk_force_only_pbc",
        pattern="pbc_gamma_sphepgto_j12",
        pos_deriv=True,
        expect=["run_energy_chunk_batch"],
        forbid=_CHUNK_PERSTEP,
        args={"run_energy_chunk_batch": {"geminal_inv": "DATA", "want_force": True, "want_stress": False}},
        heavy=True,
    ),
    dict(
        id="chunk_force_stress_pbc",
        pattern="pbc_gamma_sphepgto_j12",
        pos_deriv=True,
        stress=True,
        expect=["run_energy_chunk_batch"],
        forbid=_CHUNK_PERSTEP,
        args={"run_energy_chunk_batch": {"geminal_inv": "DATA", "want_force": True, "want_stress": True}},
        heavy=True,
    ),
    dict(
        id="chunk_mpc_pbc",
        pattern="pbc_gamma_sphepgto_j12",
        mpc=True,
        expect=["run_energy_chunk_batch"],
        forbid=_CHUNK_PERSTEP,
        # MPC is a PASSIVE, energy-only observable: enabling it must not perturb the
        # chunk route or the geminal_inv threading (it never reaches run_energy_chunk's
        # own e_L formula at all).
        args={"run_energy_chunk_batch": {"geminal_inv": "DATA"}},
        heavy=True,
    ),
    dict(
        id="chunk_complex_twist",
        pattern="twist_sphepgto_j12",
        mode="fast",
        expect=["run_energy_chunk_batch"],
        forbid=_CHUNK_PERSTEP,
        args={"run_energy_chunk_batch": {"geminal_inv": "DATA", "complex": True}},
        # Complex numerical parity is covered with a deterministic fixed-input
        # contract in the dedicated MCMC/GFMC suites; this row owns dispatch only.
        heavy=True,
    ),
    dict(
        id="perstep_mol_ae",
        pattern="mol_cart_sd_jnone",
        pos_deriv=True,
        param_deriv=True,
        expect=["mcmc_sweep", "e_L", "grad_e_L_ae", "grad_ln_psi"],
        forbid=_CHUNK,
        args={
            "mcmc_sweep": {"geminal_inv": "DATA"},
            "e_L": {"geminal_inv": "DATA"},
            "grad_e_L_ae": {"geminal_inv": "DATA"},
            "grad_ln_psi": {"geminal_inv": "DATA"},
        },
        heavy=True,
    ),
    dict(
        id="perstep_mol_ecp",
        pattern="mol_ecp_cart_j12",
        pos_deriv=True,
        param_deriv=True,
        expect=["mcmc_sweep", "e_L", "grad_e_L_ecp", "grad_ln_psi"],
        forbid=_CHUNK + ("grad_e_L_ae",),
        args={"e_L": {"geminal_inv": "DATA"}, "grad_e_L_ecp": {"geminal_inv": "DATA"}, "grad_ln_psi": {"geminal_inv": "DATA"}},
        heavy=True,
    ),
    dict(
        id="perstep_pbc_gamma",
        pattern="pbc_gamma_sphepgto_j12",
        pos_deriv=True,
        stress=True,
        param_deriv=True,
        expect=["mcmc_sweep", "e_L", "grad_e_L_pbc", "grad_ln_psi_pbc", "grad_e_L_eps_pbc", "grad_ln_psi_eps_pbc"],
        forbid=_CHUNK + ("grad_e_L_ae", "grad_e_L_ecp"),
        args={
            "e_L": {"geminal_inv": "DATA"},
            "grad_e_L_pbc": {"geminal_inv": "DATA"},
            "grad_ln_psi_pbc": {"geminal_inv": "DATA"},
            "grad_e_L_eps_pbc": {"geminal_inv": "DATA"},
            "grad_ln_psi_eps_pbc": {"geminal_inv": "DATA"},
        },
        heavy=True,
    ),
    dict(
        id="perstep_complex_twist",
        pattern="twist_sphepgto_j12",
        pos_deriv=True,
        stress=True,
        expect=[
            "mcmc_sweep_complex",
            "e_L_complex",
            "grad_e_L_pbc_complex",
            "grad_ln_psi_pbc_complex",
            "grad_e_L_eps_pbc_complex",
            "grad_ln_psi_eps_pbc_complex",
        ],
        forbid=_CHUNK + ("mcmc_sweep", "e_L"),
        # NOTE the asymmetry, which is a live manifest finding: the COMPLEX grad twins carry
        # no ``geminal_inv`` parameter at all (the manifest records it as missing in 6
        # pairs), so only ``e_L_complex`` can be asserted for it here. That is recorded,
        # not excused -- when the parameter is added, this row must gain the assertion.
        args={"e_L_complex": {"geminal_inv": "DATA"}},
        heavy=True,
    ),
]


@pytest.mark.parametrize(
    "row",
    [pytest.param(r, marks=pytest.mark.heavy if r["heavy"] else pytest.mark.light, id=r["id"]) for r in _ROWS],
)
def test_route_parity_observables(row, monkeypatch):
    """Route parity observables."""
    ham = _ham(row["pattern"])
    tag = row["id"]
    kw = dict(
        pos_deriv=row.get("pos_deriv", False),
        stress=row.get("stress", False),
        param_deriv=row.get("param_deriv", False),
        mpc=row.get("mpc", False),
    )

    spy = install_ctx_spy(monkeypatch)
    m_rust = _run(ham, rust_on=True, **kw)

    # CHECK 1: the declared route ran ...
    for meth in row["expect"]:
        assert_native_ran(spy, meth, context=tag)
    # ... and the other route did not, so a silent re-route is a failure, not a pass.
    for meth in row["forbid"]:
        assert spy.count(meth) == 0, (
            f"route parity [{tag}]: this run was expected on the {'chunk' if meth in _CHUNK_PERSTEP else 'per-step'} "
            f"route, but {meth}() fired {spy.count(meth)}x. Observed: {spy.summary()}"
        )

    # CHECK 2: the discriminating arguments arrived.
    for meth, args in row.get("args", {}).items():
        for arg, expected in args.items():
            if expected == "DATA":
                assert_arg_is_not_none(spy, meth, arg, context=tag)
            else:
                assert_arg_reached(spy, meth, arg, expected, context=tag)

    # CHECK 3: return structure -- shape/dtype/block names are captured per call by the
    # spy; assert the chunk's mapping carries the blocks the driver unpacks by name.
    if "run_energy_chunk_batch" in row["expect"]:
        c = spy.calls_to("run_energy_chunk_batch")[0]
        assert c.ret[0] == "dict", f"{tag}: run_energy_chunk_batch returned {c.ret[0]}, expected a mapping"
        blocks = {k for k, _ in c.ret[1]}
        assert {"e_L", "geminal", "geminal_inv"} <= blocks, (
            f"{tag}: run_energy_chunk_batch returned blocks {sorted(blocks)}; the driver unpacks at least "
            f"e_L / geminal / geminal_inv by name"
        )
