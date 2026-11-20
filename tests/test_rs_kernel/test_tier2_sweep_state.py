"""Sweep-state contract: the native Rust sweep emits the per-walker state that jQMC's
`compute_local_energy_fast` (`_e_L_fast`) consumes -- `geminal_inv` + the streaming
`Det_ratio_streaming_state` (ao_up, ao_dn, paired_dn, paired_up_lambda).

Two checks:
  (a) BYTE-MATCH: the sweep-emitted state == jQMC `_init_det_ratio_streaming_state` rebuilt
      at the final (swept) electron config -- proves the streaming advance is exact.
  (b) FUNCTIONAL: `compute_local_energy_fast(ham, r, geminal_inv_rust, state_rust)` ==
      the full `compute_local_energy(ham, r)` -- proves the emitted pytree is consumable
      end-to-end by the JAX optimization path. (Light system only -- the fast path's ECP
      JAX compile is heavy.)

Rows (AO family x geminal x Jastrow x ECP x BC). Only bspline has a named
``_rs_fixtures`` pattern, so three rows build directly via ``_build_ham_for_cell``
(the same underlying builder ``_rs_fixtures.rs_ham`` wraps):
  - "H2" row: ``H2_ae_ccpvdz_sphe.h5`` + active J3 (AE sphe SD j123ao).
  - "Cl2" row: ``H2_ecp_ccpvdz_cart.h5`` + active J3 (ECP cart SD j123ao); the
    molecule is H2, the AO-family/ECP/Jastrow code path is the Cl2 one.
  - "AGPn" row: ``water_ccecp_ccpvqz.h5`` + gem="AGPn" (ECP, sphe-GTO, AGPn, no
    Jastrow).
  - "bspline" row: ``rs_ham("bspline_gamma_jnone")`` (periodic Gamma, B-spline AO,
    ECP, no Jastrow): the bspline-geminal-in-native-sweep path.
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
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jqmc._precision import get_tolerance_xla_rust_consistency as _d8_backend_tolerance

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.with_chdir

import jqmc_rs  # noqa: E402

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.determinant import (
    Det_ratio_streaming_state,
    _init_det_ratio_streaming_state,
    compute_geminal_all_elements,
)
from jqmc.hamiltonians import compute_local_energy, compute_local_energy_fast
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import rs_ham

# Molecular rows -> "strict"; the periodic bspline row -> "medium" (per-electron
# far-image AO cutoff noise, same rationale as test_determinant.py).
_STATE_ZONES = ("det_ratio", "ao_eval")
_EL_ZONES = ("local_energy", "coulomb", "det_ratio", "ao_eval")


def _ham_H2():
    return _build_ham_for_cell("H2_ae_ccpvdz_sphe.h5", True, {"j3": "sphe_gto"}, "SD")


def _ham_Cl2():
    return _build_ham_for_cell("H2_ecp_ccpvdz_cart.h5", True, {"j3": "cart_gto"}, "SD")


def _ham_AGPn():
    return _build_ham_for_cell("water_ccecp_ccpvqz.h5", False, None, "AGPn")


def _ham_bspline():
    return rs_ham("bspline_gamma_jnone")


SYSTEMS = {
    "H2": (_ham_H2, False),
    "Cl2": (_ham_Cl2, False),
    "AGPn": (_ham_AGPn, False),
    "bspline": (_ham_bspline, True),
}


def _charges(ham):
    cp = ham.coulomb_potential_data
    Z = np.asarray(ham.structure_data.atomic_numbers, dtype=np.float64)
    if cp.ecp_flag:
        Z = Z - np.asarray(cp.z_cores, dtype=np.float64)
    return Z


def _sweep(ham, seed, num_moves=30):
    gem = ham.wavefunction_data.geminal_data
    jas = ham.wavefunction_data.jastrow_data
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    Z = _charges(ham)
    sd = ham.structure_data
    pbc = bool(sd.pbc_flag)
    rng = np.random.default_rng(seed + 1234)
    nu, nd = gem.num_electron_up, gem.num_electron_dn
    if pbc:
        cell = np.asarray(sd.cell, dtype=np.float64)
        nmax = int(sd._n_max)
        ru0 = (rng.random((nu, 3)) @ cell).astype(np.float64)
        rd0 = (rng.random((nd, 3)) @ cell).astype(np.float64)
    else:
        cell, nmax = None, 0
        ru0 = (R[rng.integers(0, len(R), nu)] + rng.normal(scale=0.7, size=(nu, 3))).astype(np.float64)
        rd0 = (R[rng.integers(0, len(R), nd)] + rng.normal(scale=0.7, size=(nd, 3))).astype(np.float64)
    G0 = np.asarray(compute_geminal_all_elements(gem, ru0, rd0), dtype=np.float64)
    # Seed the native sweep from NUMPY's inverse, not Rust's own
    # ``_matrix_inverse``. Production hands the native sweep a JAX-computed G^-1, and a
    # test that generates the seed with the very kernel under test cannot see an error in
    # it -- the emitted streaming state would be consistent with the wrong seed.
    Ginv0 = np.asarray(np.linalg.inv(G0), dtype=np.float64)
    injected = rng.normal(size=num_moves).astype(np.float64)
    k = np.asarray(random.PRNGKey(seed), dtype=np.uint32)
    res = jqmc_rs._mcmc_sweep(gem, jas, R, Z, ru0, rd0, G0, Ginv0, 1.2, 0.0, int(k[0]), int(k[1]), injected, cell, nmax)
    return gem, res


@pytest.mark.light
@pytest.mark.parametrize("name", list(SYSTEMS))
@pytest.mark.parametrize("seed", [0, 1, 5])
def test_sweep_emits_det_ratio_state(name, seed):
    """The sweep-emitted Det_ratio_streaming_state byte-matches jQMC's own rebuilt-at-config."""
    ham_fn, periodic = SYSTEMS[name]
    gem, res = _sweep(ham_fn(), seed)
    st = _init_det_ratio_streaming_state(gem, res["r_up"], res["r_dn"])
    atol, rtol = get_tolerance_xla_rust_consistency(_STATE_ZONES, level="strict")
    for field in ("ao_up", "ao_dn", "paired_dn", "paired_up_lambda"):
        got = np.asarray(res[field])
        ref = np.asarray(getattr(st, field))
        assert got.shape == ref.shape, (name, seed, field, got.shape, ref.shape)
        # PER-ELEMENT: the old band was ``atol + rtol * max|ref|``, a
        # GLOBAL max norm -- one large AO row set the band for every element, so a small
        # ``paired_dn`` entry could be 100% wrong and still pass. assert_allclose judges
        # each element against its own magnitude and reports the worst offender's index.
        np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=f"{name} seed={seed} field={field}")


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 5])
def test_e_L_fast_consumes_rust_state(seed):
    """End-to-end: jQMC `compute_local_energy_fast` fed the Rust-emitted geminal_inv +
    det_ratio_state equals the full `compute_local_energy`. Light AE H2 only."""
    ham = _ham_H2()
    _, res = _sweep(ham, seed)
    st = Det_ratio_streaming_state(
        ao_up=jnp.asarray(res["ao_up"]),
        ao_dn=jnp.asarray(res["ao_dn"]),
        paired_dn=jnp.asarray(res["paired_dn"]),
        paired_up_lambda=jnp.asarray(res["paired_up_lambda"]),
    )
    ef = float(compute_local_energy_fast(ham, res["r_up"], res["r_dn"], jnp.eye(3), jnp.asarray(res["geminal_inv"]), st))
    es = float(compute_local_energy(ham, res["r_up"], res["r_dn"], jnp.eye(3)))
    atol, rtol = _d8_backend_tolerance(_EL_ZONES, level="strict")
    assert abs(ef - es) <= atol + rtol * max(1.0, abs(es)), (seed, abs(ef - es))


# --------------------------------------------------------------------------- #
# The production driver seeds the carried geminal it hands to the
# native sweep from the sweep's OWN orbital evaluation.
# --------------------------------------------------------------------------- #
class _SweepSeedSpy:
    """Forward every native-context call; record the first real sweep's inputs."""

    def __init__(self, ctx, sink):
        self._ctx = ctx
        self._sink = sink

    def mcmc_sweep_batch(self, positions, charges, r_up, r_dn, geminal, geminal_inv, *args, **kwargs):
        if "geminal" not in self._sink:
            self._sink.update(
                r_up=np.array(r_up, copy=True),
                r_dn=np.array(r_dn, copy=True),
                geminal=np.array(geminal, copy=True),
                geminal_inv=np.array(geminal_inv, copy=True),
            )
        return self._ctx.mcmc_sweep_batch(positions, charges, r_up, r_dn, geminal, geminal_inv, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def _native_det_ratio(gem, r_up_old, r_up_new, r_dn):
    """|det G(new) / det G(old)| from two fresh native geminals (same precision policy)."""
    sign_new, log_new = np.linalg.slogdet(np.asarray(jqmc_rs.compute_geminal_all_elements(gem, r_up_new, r_dn)))
    sign_old, log_old = np.linalg.slogdet(np.asarray(jqmc_rs.compute_geminal_all_elements(gem, r_up_old, r_dn)))
    return float(sign_new * sign_old * np.exp(log_new - log_old))


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1])
def test_driver_seed_keeps_far_electron_ratios_exact(monkeypatch, seed):
    """An electron that wanders far after the run seed keeps exact Metropolis ratios.

    All-electron open-shell UHF N with J1/J2 (``mol_cart_sd_j12``) on the per-step
    native route that VMC optimization with ``opt_geminal`` uses. The driver's seed
    ``(G, G^-1)`` is captured at its first native sweep; one up electron is then
    moved to 6.0 and to 6.3 bohr from the nucleus with the sweep's own rank-1
    kernels (far enough for its row to decay well below the near-nucleus scale,
    near enough that the fp64 rank-1 arithmetic itself stays exact to strict). The
    second determinant ratio must equal the exact ratio of two fresh native
    geminals. The rank-1 update keeps any seed-vs-cache difference as an absolute
    row offset; a JAX-built seed (~3e-7 of max|G| apart under the mixed fp32
    ``ao_eval`` zone) then dominates the decayed far row, the walker stops
    following ``|Psi|^2``, runs away, and its fp32 AO row finally underflows to an
    exactly singular geminal: non-finite ``mo_coefficients`` derivatives and the
    "SR global centering statistics" failure.
    """
    from jqmc._precision import get_tolerance_min, precision_policy_snapshot
    from jqmc.jqmc_mcmc import MCMC, _refresh_running_geminal_inv

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    sink = {}
    real_build = jqmc_rs.build_obs_ctx

    def _build(*args, precision_policy, **kwargs):
        # The native context is built with the explicit policy of
        # the active precision mode; the spy forwards it explicitly.
        assert precision_policy == precision_policy_snapshot(), precision_policy
        return _SweepSeedSpy(real_build(*args, precision_policy=precision_policy, **kwargs), sink)

    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", _build)
    ham = rs_ham("mol_cart_sd_j12")
    gem = ham.wavefunction_data.geminal_data
    driver = MCMC(
        hamiltonian_data=ham,
        mcmc_seed=seed,
        num_walkers=2,
        num_mcmc_per_measurement=1,
        Dt=0.5,
        epsilon_AS=0.0,
        comput_log_WF_param_deriv=True,
    )
    driver.run(num_mcmc_steps=1)
    assert "geminal" in sink, "the per-step native sweep route was not taken"

    nucleus = np.asarray(ham.structure_data.positions, dtype=np.float64)[0]
    atol, rtol = get_tolerance_min(("det_ratio",), "strict")
    for w in range(sink["r_up"].shape[0]):
        r_up = np.ascontiguousarray(sink["r_up"][w])
        r_dn = np.ascontiguousarray(sink["r_dn"][w])
        G0 = np.ascontiguousarray(sink["geminal"][w])
        X0 = np.ascontiguousarray(sink["geminal_inv"][w])
        k = 0
        direction = (r_up[k] - nucleus) / np.linalg.norm(r_up[k] - nucleus)
        far, farther = nucleus + 6.0 * direction, nucleus + 6.3 * direction
        one_hot = np.zeros(r_up.shape[0], dtype=np.float64)
        one_hot[k] = 1.0
        v_far = np.asarray(jqmc_rs._det_v_up_move(gem, r_up, r_dn, k, far), dtype=np.float64)
        _, X1, G1 = jqmc_rs._det_sm_update(G0, X0, one_hot, v_far, True, k)
        # The driver's boundary inverse refresh re-derives the inverse of the SAME carried
        # G (removing the rank-1 update's own conditioning error); it cannot repair G.
        X1, _ = _refresh_running_geminal_inv(jnp.asarray(G1)[None], jnp.asarray(X1)[None])
        X1 = np.asarray(X1[0])
        r_up_far = r_up.copy()
        r_up_far[k] = far
        v_farther = np.asarray(jqmc_rs._det_v_up_move(gem, r_up_far, r_dn, k, farther), dtype=np.float64)
        ratio, _, _ = jqmc_rs._det_sm_update(np.asarray(G1), np.asarray(X1), one_hot, v_farther, True, k)
        r_up_farther = r_up_far.copy()
        r_up_farther[k] = farther
        exact = _native_det_ratio(gem, r_up_far, r_up_farther, r_dn)
        assert np.isfinite(ratio) and np.isfinite(exact), (seed, w, ratio, exact)
        assert abs(ratio - exact) <= atol + rtol * abs(exact), (
            f"seed={seed} walker={w}: carried-seed ratio {ratio:.10e} vs exact {exact:.10e}"
        )


# --------------------------------------------------------------------------- #
# A walker that STARTS with far electrons gets an exact
# seed inverse (row/column-equilibrated pseudo-inverse) on the native route.
# --------------------------------------------------------------------------- #
_FAR_SEED_WALKERS = ((10.0, 10.0), (None, 14.0))  # (first up, first down) distance in bohr; None = near
_FAR_SEED_STEP = 0.3  # bohr: inward move of the first up / first down electron


@pytest.mark.light
def test_driver_far_seed_keeps_far_electron_ratios_exact(monkeypatch):
    """A walker seeded with far electrons gets exact Metropolis ratios on the native per-step route.

    All-electron open-shell UHF N with J1/J2
    (``mol_cart_sd_j12``), production MCMC driver on the per-step native route
    (``comput_log_WF_param_deriv=True``). Walker 0 starts with its first up and
    first down electron 10 bohr from the nucleus, walker 1 with only its first
    down electron 14 bohr out: both geminals are invertible but their singular
    values span more than ``1 / rcond_svd``. The seed ``(G, G^-1)`` handed to
    the first native sweep is captured; a 0.3 bohr inward move of the first up
    and of the first down electron through the sweep's own rank-1 kernels
    (``_det_v_up_move`` / ``_det_u_dn_move`` + ``_det_sm_update``) must give
    the ratio of two fresh native geminals within the strict det_ratio pair. A
    plain thresholded SVD seed drops the far direction (ratio 1.0 against 3.6),
    and such a walker can never move back.
    """
    from jqmc._precision import get_tolerance_min, precision_policy_snapshot
    from jqmc._setting import get_eps
    from jqmc.jqmc_mcmc import MCMC

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    sink = {}
    real_build = jqmc_rs.build_obs_ctx

    def _build(*args, precision_policy, **kwargs):
        assert precision_policy == precision_policy_snapshot(), precision_policy
        return _SweepSeedSpy(real_build(*args, precision_policy=precision_policy, **kwargs), sink)

    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", _build)
    ham = rs_ham("mol_cart_sd_j12")
    gem = ham.wavefunction_data.geminal_data
    nucleus = np.asarray(ham.structure_data.positions, dtype=np.float64)[0]
    rng = np.random.default_rng(3)
    near_up = nucleus + rng.normal(scale=0.8, size=(gem.num_electron_up, 3))
    near_dn = nucleus + rng.normal(scale=0.8, size=(gem.num_electron_dn, 3))
    directions = {"up": np.array([0.3, 0.5, 0.81]), "dn": np.array([-0.6, 0.2, 0.77])}
    directions = {spin: d / np.linalg.norm(d) for spin, d in directions.items()}
    r_up = np.stack([near_up.copy() for _ in _FAR_SEED_WALKERS])
    r_dn = np.stack([near_dn.copy() for _ in _FAR_SEED_WALKERS])
    for w, (up_distance, dn_distance) in enumerate(_FAR_SEED_WALKERS):
        if up_distance is not None:
            r_up[w, 0] = nucleus + up_distance * directions["up"]
        r_dn[w, 0] = nucleus + dn_distance * directions["dn"]

    driver = MCMC(
        hamiltonian_data=ham,
        mcmc_seed=0,
        num_walkers=len(_FAR_SEED_WALKERS),
        num_mcmc_per_measurement=1,
        Dt=0.5,
        epsilon_AS=0.0,
        comput_log_WF_param_deriv=True,
    )
    driver._MCMC__latest_r_up_carts = jnp.asarray(r_up)
    driver._MCMC__latest_r_dn_carts = jnp.asarray(r_dn)
    driver.run(num_mcmc_steps=1)
    assert "geminal" in sink, "the per-step native sweep route was not taken"
    # Preconditions: the seed was taken at the far configurations, and both
    # geminals are below the plain cutoff yet invertible.
    np.testing.assert_array_equal(sink["r_up"], r_up)
    np.testing.assert_array_equal(sink["r_dn"], r_dn)
    singular_values = np.linalg.svd(sink["geminal"], compute_uv=False)
    assert np.all(singular_values[:, -1] / singular_values[:, 0] < get_eps("rcond_svd", np.float64))
    assert np.all(np.isfinite(np.linalg.slogdet(sink["geminal"])[1]))

    atol, rtol = get_tolerance_min(("det_ratio",), "strict")
    for w in range(len(_FAR_SEED_WALKERS)):
        ru = np.ascontiguousarray(sink["r_up"][w])
        rd = np.ascontiguousarray(sink["r_dn"][w])
        G0 = np.ascontiguousarray(sink["geminal"][w])
        X0 = np.ascontiguousarray(sink["geminal_inv"][w])
        one_hot = np.zeros(G0.shape[0], dtype=np.float64)
        one_hot[0] = 1.0
        for spin in ("up", "dn"):
            old = ru[0] if spin == "up" else rd[0]
            new = old - _FAR_SEED_STEP * (old - nucleus) / np.linalg.norm(old - nucleus)
            ru_new, rd_new = ru.copy(), rd.copy()
            if spin == "up":
                ru_new[0] = new
                u, v = one_hot, np.asarray(jqmc_rs._det_v_up_move(gem, ru, rd, 0, new), dtype=np.float64)
            else:
                rd_new[0] = new
                u, v = np.asarray(jqmc_rs._det_u_dn_move(gem, ru, rd, 0, new), dtype=np.float64), one_hot
            ratio, _, _ = jqmc_rs._det_sm_update(G0, X0, u, v, spin == "up", 0)
            sign_new, log_new = np.linalg.slogdet(np.asarray(jqmc_rs.compute_geminal_all_elements(gem, ru_new, rd_new)))
            sign_old, log_old = np.linalg.slogdet(np.asarray(jqmc_rs.compute_geminal_all_elements(gem, ru, rd)))
            exact = float(sign_new * sign_old * np.exp(log_new - log_old))
            assert np.isfinite(ratio) and np.isfinite(exact), (w, spin, ratio, exact)
            assert abs(ratio - exact) <= atol + rtol * abs(exact), (
                f"walker={w} {spin} move: far-seed ratio {ratio:.10e} vs exact {exact:.10e}"
            )
