"""Shared cases: carried Sherman--Morrison inverse after many MCMC updates.

The walkers start with one same-spin pair ``delta`` Bohr apart, so the seed
geminal is near-singular (cond(G) ~ 1e14--1e15 in full mode; numerically
singular with the fp32 AO zone of mixed mode). The production MCMC driver then
runs many single-electron updates; the pair separates and G becomes well
conditioned, but a carried inverse that is never refreshed keeps the seed's
absolute error. The checks compare, at the final configuration and within the
active precision mode:

1. G generation: the carried geminal against the geminal rebuilt from the
   final positions (``ao_eval``/``det_eval`` strict: in mixed mode the carried
   rows and the rebuilt rows evaluate the fp32 AO zone through different
   routes, exactly as in the canonical fast-update check);
2. same-G consumers: every fast consumer of the carried inverse (fast local
   energy = kinetic + Coulomb fast incl. the ECP non-local fast ratio, fused
   HF force/drift, Pulay force/drift, HF and Pulay stress) evaluated with the
   carried inverse against the same consumer evaluated with the seed's SVD
   pseudo-inverse of the SAME carried G (``det_eval``/``wf_*``/``coulomb``
   strict).
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

import contextlib
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from jqmc import jqmc_mcmc as mcmc
from jqmc._precision import get_tolerance_min
from jqmc.derivatives import bridge
from jqmc.coulomb_potential import compute_coulomb_potential_fast
from jqmc.determinant import compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data, compute_kinetic_energy_all_elements_fast_update

TREXIO_DIR = Path(__file__).resolve().parent / "trexio_example_files"
# fixture, same-spin pair separations (Bohr) per walker, steps
CASES = {
    "h4_pbc_ae": ("H4_pbc_ae_cart.h5", (1.0e-14, 1.0e-13, 1.0e-14, 1.0e-13), 40),
    "water_ecp": ("water_ccecp_ccpvqz.h5", (1.0e-14, 1.0e-13), 40),
}
NUM_MCMC_PER_MEASUREMENT = 16
SAME_G_ZONES = ("det_eval", "wf_eval", "wf_kinetic", "coulomb")
G_GENERATION_ZONES = ("ao_eval", "det_eval")


def build_case(name: str):
    """Return ``(hamiltonian, r_up, r_dn)`` with near-coincident same-spin pairs."""
    filename, deltas, _ = CASES[name]
    structure, _, _, _, geminal, coulomb = read_trexio_file(str(TREXIO_DIR / filename), store_tuple=True)
    jastrow = Jastrow_data(
        jastrow_two_body_data=Jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp", structure_data=structure)
    )
    ham = Hamiltonian_data(
        structure_data=structure,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=jastrow),
        coulomb_potential_data=coulomb,
    )
    rng = np.random.default_rng(2025)
    centre = np.asarray(structure.positions).mean(axis=0)
    base_up = centre + rng.normal(size=(geminal.num_electron_up, 3)) * 0.5
    base_dn = centre + rng.normal(size=(geminal.num_electron_dn, 3)) * 0.5
    direction = np.array([1.0, 0.3, -0.4]) / np.linalg.norm([1.0, 0.3, -0.4])
    ups, dns = [], []
    for i, delta in enumerate(deltas):
        up = base_up + np.array([0.03 * i, -0.02 * i, 0.04 * i])
        up[1] = up[0] + delta * direction
        ups.append(up)
        dns.append(base_dn + np.array([-0.01 * i, 0.04 * i, 0.02 * i]))
    return ham, np.stack(ups), np.stack(dns)


def seed_condition_numbers(ham, r_up, r_dn) -> np.ndarray:
    """cond(G) at the near-singular start (precondition of the case)."""
    gd = ham.wavefunction_data.geminal_data
    return np.array([np.linalg.cond(np.asarray(compute_geminal_all_elements(gd, u, d))) for u, d in zip(r_up, r_dn)])


@contextlib.contextmanager
def capture_refreshed_state():
    """Record the (G, X) pair the driver hands to its consumers on the per-step route."""
    record: dict[str, np.ndarray] = {}
    original = mcmc._refresh_running_geminal_inv

    def spy(geminal, geminal_inv, threshold=None):
        refreshed, count = original(geminal, geminal_inv, threshold)
        record["geminal"] = np.asarray(geminal)
        record["geminal_inv"] = np.asarray(refreshed)
        return refreshed, count

    with patch.object(mcmc, "_refresh_running_geminal_inv", spy):
        yield record


class _ChunkCtxProxy:
    """Delegate to the native ctx; record the final (G, X) of every energy chunk."""

    def __init__(self, ctx, record):
        self._ctx = ctx
        self._record = record

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def run_energy_chunk_batch(self, *args, **kwargs):
        out = self._ctx.run_energy_chunk_batch(*args, **kwargs)
        self._record["geminal"] = np.asarray(out["geminal"])
        self._record["geminal_inv"] = np.asarray(out["geminal_inv"])
        self._record["chunk_refreshes"] = self._record.get("chunk_refreshes", 0) + int(np.sum(out["inverse_refreshes"]))
        self._record["chunk_e_L_last"] = np.asarray(out["e_L"])[:, -1]
        return out


@contextlib.contextmanager
def capture_chunk_state():
    """Record the (G, X) pair the native energy chunk returns after its last step."""
    import jqmc_rs

    record: dict[str, np.ndarray] = {}
    original = jqmc_rs.build_obs_ctx

    def build(*args, **kwargs):
        return _ChunkCtxProxy(original(*args, **kwargs), record)

    with patch.object(jqmc_rs, "build_obs_ctx", build):
        yield record


def run_driver(ham, r_up, r_dn, steps: int, **kwargs):
    """Run the production MCMC driver from the supplied walkers."""
    driver = mcmc.MCMC(
        hamiltonian_data=ham,
        mcmc_seed=34,
        num_walkers=r_up.shape[0],
        num_mcmc_per_measurement=NUM_MCMC_PER_MEASUREMENT,
        epsilon_AS=1.0e-3,
        random_discretized_mesh=False,
        use_swct=False,
        **kwargs,
    )
    driver._MCMC__latest_r_up_carts = jnp.asarray(r_up)
    driver._MCMC__latest_r_dn_carts = jnp.asarray(r_dn)
    driver.run(num_mcmc_steps=steps)
    return driver


_KINETIC_FAST = jax.jit(jax.vmap(compute_kinetic_energy_all_elements_fast_update, in_axes=(None, 0, 0, 0)))
# Coulomb fast incl. the ECP non-local fast ratio that consumes ``A_old_inv``.
_COULOMB_FAST = jax.jit(
    jax.vmap(
        lambda cp, ru, rd, rt, inv, wf, st: compute_coulomb_potential_fast(
            cp, ru, rd, rt, inv, wavefunction_data=wf, det_ratio_state=st
        ),
        in_axes=(None, 0, 0, 0, 0, None, 0),
    )
)


def _budget_ratio(actual, reference, atol, rtol) -> float:
    """Diagnostic only (failure messages): max |a - b| / (atol + rtol |b|)."""
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    return float(np.max(np.abs(actual - reference) / (atol + rtol * np.abs(reference))))


def assert_close_report(pairs: dict, atol: float, rtol: float, what: str) -> None:
    """``assert_allclose`` every (actual, reference) leaf pair; the message lists all budget ratios."""
    for name, (got, ref) in pairs.items():
        for a, b in zip(got, ref):
            assert np.all(np.isfinite(np.asarray(a))) and np.all(np.isfinite(np.asarray(b))), f"{what}: {name} not finite"
    ratios = {name: max(_budget_ratio(a, b, atol, rtol) for a, b in zip(got, ref)) for name, (got, ref) in pairs.items()}
    for name, (got, ref) in pairs.items():
        for a, b in zip(got, ref):
            np.testing.assert_allclose(
                np.asarray(a), np.asarray(b), atol=atol, rtol=rtol, err_msg=f"{what}: {name}; budget ratios {ratios}"
            )


def assert_g_generation(ham, r_up, r_dn, geminal) -> None:
    """The carried G equals G rebuilt from the final positions (ao_eval/det_eval strict)."""
    atol, rtol = get_tolerance_min(G_GENERATION_ZONES, "strict")
    gd = ham.wavefunction_data.geminal_data
    rebuilt = jax.vmap(compute_geminal_all_elements, in_axes=(None, 0, 0))(gd, jnp.asarray(r_up), jnp.asarray(r_dn))
    assert_close_report({"geminal": ([geminal], [rebuilt])}, atol, rtol, "carried G vs rebuilt G")


def assert_same_g_consumers(ham, r_up, r_dn, geminal, geminal_inv) -> None:
    """Every fast consumer: carried inverse vs the seed's pseudo-inverse of the same G (strict)."""
    atol, rtol = get_tolerance_min(SAME_G_ZONES, "strict")
    r_up = jnp.asarray(r_up)
    r_dn = jnp.asarray(r_dn)
    carried = jnp.asarray(geminal_inv)
    reference = mcmc._geminal_pinv_from_matrix_batched(jnp.asarray(geminal))
    nw = r_up.shape[0]
    rts = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (nw, 3, 3))
    state = mcmc._jit_vmap_init_det_ratio_state(ham.wavefunction_data.geminal_data, r_up, r_dn)
    consumers = {
        "e_L_fast": lambda inv: mcmc._jit_vmap_e_L_fast(ham, r_up, r_dn, rts, inv, state),
        "kinetic_fast": lambda inv: _KINETIC_FAST(ham.wavefunction_data, r_up, r_dn, inv),
        "coulomb_fast": lambda inv: _COULOMB_FAST(
            ham.coulomb_potential_data, r_up, r_dn, rts, inv, ham.wavefunction_data, state
        ),
        "grad_e_L_rR": lambda inv: bridge.batched_grad_e_L(wrt=("r", "R"), with_geminal_inv=True)(ham, r_up, r_dn, rts, inv),
        "grad_ln_Psi_rR": lambda inv: bridge.batched_grad_ln_Psi(wrt=("r", "R"), with_geminal_inv=True)(ham, r_up, r_dn, inv),
    }
    if ham.structure_data.pbc_flag:
        consumers["grad_e_L_eps"] = lambda inv: bridge.batched_grad_e_L(wrt="eps", with_geminal_inv=True)(
            ham, r_up, r_dn, rts, inv
        )
        consumers["grad_ln_Psi_eps"] = lambda inv: bridge.batched_grad_ln_Psi(wrt="eps", with_geminal_inv=True)(
            ham, r_up, r_dn, inv
        )
    pairs = {}
    for name, consumer in consumers.items():
        got = jax.tree_util.tree_leaves(consumer(carried))
        ref = jax.tree_util.tree_leaves(consumer(reference))
        assert len(got) == len(ref) and len(got) > 0, name
        pairs[name] = (got, ref)
    assert_close_report(pairs, atol, rtol, "carried vs same-G inverse consumers")


def assert_interval_handover(ham, r_up, r_dn, geminal_inv, det_state) -> None:
    """The inverse a GFMC projection interval hands over obeys the single refresh rule.

    Under the per-hop refresh rule the carried inverse is screened after every
    hop against the geminal its streaming caches imply, so at the interval
    boundary (before any boundary refresh) its probe residual against that
    geminal is at most ``SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD`` for every
    walker. That geminal equals the one rebuilt at the boundary positions (G
    generation, ``ao_eval``/``det_eval`` strict), and the fast Coulomb energy
    (incl. the ECP non-local det ratios when the fixture has an ECP) computed
    from the handed-over inverse equals that of the fresh route, the seed's
    pseudo-inverse of that geminal (``det_eval``/``wf_*``/``coulomb`` strict).
    On the all-electron near-singular H4 fixture there is no ECP, so the fast Coulomb route
    does not consume the inverse and the two agree exactly (0.0 on
    jax 0.7.2 and 0.10.2, full and mixed); the comparison pins the route.

    Not compared, like the raw inverse entries and the force/stress
    estimators: the fast local energy, the continuum kinetic energies per electron and the
    discretized kinetic mesh elements. They are ratios with one row/column of G
    replaced and carry the cond(G) x residual class: at the near-singular H4 boundary
    (cond(G) = 5.8e5) a carried inverse whose residual (7.7e-11, or 1.5e-11) is
    the size of a fresh factorization's of the same G puts the mesh ratios 169x
    (65x) and the kinetic elements 1.4x over the strict budget against an exact
    rational oracle, while the fresh pseudo-inverse stays within 0.005x; which
    rounding the chain takes depends on the jax version and XLA threading, so
    no residual threshold can hold that comparison.
    """
    from jqmc import _setting
    from jqmc.determinant import _geminal_from_det_ratio_state

    gd = ham.wavefunction_data.geminal_data
    carried = jnp.asarray(geminal_inv)
    residual = np.asarray(mcmc._det_state_probe_residual(gd, det_state, carried))
    assert np.all(np.isfinite(residual)), f"handed-over inverse residual not finite: {residual}"
    assert np.all(residual <= _setting.SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD), (
        f"handed-over inverse residual {residual} exceeds the refresh threshold "
        f"{_setting.SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD:.1e} at the interval boundary"
    )
    geminal = jax.vmap(_geminal_from_det_ratio_state, in_axes=(None, 0))(gd, det_state)
    assert_g_generation(ham, r_up, r_dn, geminal)
    fresh = mcmc._geminal_pinv_from_matrix_batched(geminal)
    r_up = jnp.asarray(r_up)
    r_dn = jnp.asarray(r_dn)
    nw = r_up.shape[0]
    rts = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (nw, 3, 3))
    state = mcmc._jit_vmap_init_det_ratio_state(gd, r_up, r_dn)

    def coulomb(inv):
        return _COULOMB_FAST(ham.coulomb_potential_data, r_up, r_dn, rts, inv, ham.wavefunction_data, state)

    pairs = {"coulomb_fast": ([coulomb(carried)], [coulomb(fresh)])}
    assert_close_report(pairs, *get_tolerance_min(SAME_G_ZONES, "strict"), "handed-over vs fresh-route inverse")
