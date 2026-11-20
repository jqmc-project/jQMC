"""Repeated eager DTN derivative calls reuse their compiled XLA programs (issue #130).

The DTN derivative kernels in ``jqmc/derivatives/dtn.py`` reduce the electron-pair
axis with a ``lax.scan`` whose body is a closure built on every call. Called
eagerly (``assemble.compute_grad_e_L`` outside ``jax.jit``, as the test oracles and
reference helpers do), each call re-traced that closure and compiled and loaded a
NEW executable, which JAX's caches keep alive. A test that calls the kernel many
times therefore grew the process's memory-mapping count without bound until
``vm.max_map_count`` was reached and the worker aborted inside XLA (issue #130).

The test calls the eager kernels three times on the same periodic DTN Hamiltonian
and configuration -- all-electron H4 and ccECP lithium, whose nuclear force also
streams the per-config DTN Pulay force over the ECP one-move mesh in a
``lax.scan`` (``one_move_mesh.mesh_force``) -- and counts JAX's own compile events through
``jax.monitoring``: the first call may compile, the second and third must not
lower or compile anything, and every call must return the values of the first.
Lowering events are counted as well as backend compiles because a persistent
compilation cache hit skips the backend compile but not the lowering.

The values are compared with the same-precision reduction-order pair, not
bitwise: the reused program is the same, but on GPU the ECP nuclear force
(``ecp_force`` / ``one_move_mesh``) accumulates with XLA scatter-adds whose
atomics add in a run-dependent order. On a GPU the lithium ``grad_V_ecp_wrt_R``
took 3 to 9 distinct values in 10 calls, at most 8.9e-16 apart (#529).
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

import jax
import jax.monitoring
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jqmc._setting import ATOL_REDUCTION_ORDER, RTOL_REDUCTION_ORDER
from jqmc.derivatives import assemble, dtn
from jqmc.jqmc_mcmc import _geminal_inv_batched
from tests.test_jqmc_kernel._grad_fidelity_harness import _build_ham

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_COMPILE_EVENTS = (
    "/jax/core/compile/jaxpr_to_mlir_module_duration",
    "/jax/core/compile/backend_compile_duration",
)
_counts = {event: 0 for event in _COMPILE_EVENTS}


def _count_compile_event(event: str, duration_secs: float, **kwargs: object) -> None:
    if event in _counts:
        _counts[event] += 1


jax.monitoring.register_event_duration_secs_listener(_count_compile_event)

_N_CALLS = 3
_RT = np.eye(3)


def _config(ham, seed: int) -> tuple[np.ndarray, np.ndarray]:
    gd = ham.wavefunction_data.geminal_data
    n_up, n_dn = int(gd.num_electron_up), int(gd.num_electron_dn)
    positions = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    centers = positions[rng.integers(0, len(positions), size=n_up + n_dn)]
    r = centers + rng.normal(scale=0.7, size=centers.shape)
    return r[:n_up], r[n_up:]


def _eager_dtn_derivatives(ham, r_up, r_dn, ginv) -> list[np.ndarray]:
    wf = ham.wavefunction_data
    d = wf.jastrow_data.jastrow_dtn_data
    grad_R = assemble.compute_grad_e_L(ham, r_up, r_dn, _RT, wrt="R", geminal_inv=ginv)
    grad_r_up, grad_r_dn = assemble.compute_grad_e_L(ham, r_up, r_dn, _RT, wrt="r", geminal_inv=ginv)
    grad_T_r = dtn.grad_T_dtn_wrt_r(d, wf.jastrow_data, wf.geminal_data, r_up, r_dn, geminal_inv=ginv)
    return [np.asarray(x) for x in (grad_R, grad_r_up, grad_r_dn, grad_T_r)]


@pytest.mark.parametrize("trexio_filename", ["H4_pbc_ae_cart.h5", "lithium.h5"])
def test_repeated_eager_dtn_derivatives_compile_nothing_new(trexio_filename):
    """Calls 2 and 3 of the eager periodic DTN derivatives lower and compile nothing and repeat call 1."""
    ham = _build_ham(trexio_filename, True, {"dtn": True})
    d = ham.wavefunction_data.jastrow_data.jastrow_dtn_data
    assert ham.structure_data.pbc_flag
    assert d.use_u and d.use_chi and d.use_f  # the pair-chunk scans of every channel run
    r_up_np, r_dn_np = _config(ham, seed=130)
    assert len(r_up_np) + len(r_dn_np) >= 2
    r_up, r_dn = jnp.asarray(r_up_np), jnp.asarray(r_dn_np)
    _, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, r_up[None], r_dn[None])
    ginv = ginv[0]

    jax.clear_caches()  # call 1 must compile, which shows the listener sees these events
    new_events = []
    results = []
    for _ in range(_N_CALLS):
        before = dict(_counts)
        results.append(_eager_dtn_derivatives(ham, r_up, r_dn, ginv))
        new_events.append({event: _counts[event] - before[event] for event in _COMPILE_EVENTS})

    assert new_events[0]["/jax/core/compile/backend_compile_duration"] > 0, new_events
    for call, events in enumerate(new_events[1:], start=2):
        assert all(n == 0 for n in events.values()), f"eager call {call} lowered/compiled new programs: {new_events}"
    for call, result in enumerate(results[1:], start=2):
        for first, again in zip(results[0], result):
            np.testing.assert_allclose(
                again,
                first,
                atol=ATOL_REDUCTION_ORDER,
                rtol=RTOL_REDUCTION_ORDER,
                err_msg=f"eager call {call} changed a DTN derivative value",
            )
    assert all(np.all(np.isfinite(x)) for x in results[0])
