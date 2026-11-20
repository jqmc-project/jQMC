"""Carried Sherman--Morrison inverse after many MCMC updates (XLA).

Walkers start with a same-spin pair 1e-14--1e-13 Bohr apart (near-singular
seed). After 40 measurement steps x 16 single-electron updates the production
XLA driver's carried inverse must agree with the seed's SVD pseudo-inverse of
the same carried geminal for every fast consumer, within the active precision
mode's central strict tolerances. Without the boundary refresh the carried
inverse keeps the seed's absolute error and every consumer disagrees by
1e2--1e7 budget ratios in the inverse-drift measurements.
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

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import jqmc_mcmc as mcmc
from jqmc._precision import get_tolerance_min
from tests import _sm_inverse_refresh_cases as cases

pytestmark = [pytest.mark.without_chdir]

# A seed this ill-conditioned is what makes the drift observable (precondition).
_MIN_SEED_CONDITION = 1.0e14


@pytest.mark.heavy
@pytest.mark.parametrize("case", ["h4_pbc_ae", "water_ecp"])
def test_mcmc_carried_inverse_matches_same_g_inverse_after_many_updates(case):
    """XLA driver: carried-inverse consumers equal same-G pseudo-inverse consumers."""
    ham, r_up, r_dn = cases.build_case(case)
    assert np.max(cases.seed_condition_numbers(ham, r_up, r_dn)) > _MIN_SEED_CONDITION
    steps = cases.CASES[case][2]
    with cases.capture_refreshed_state() as record:
        driver = cases.run_driver(
            ham,
            r_up,
            r_dn,
            steps,
            comput_position_deriv=True,
            comput_stress=bool(ham.structure_data.pbc_flag),
        )
    up = driver._MCMC__latest_r_up_carts
    dn = driver._MCMC__latest_r_dn_carts
    assert driver.mcmc_counter == steps

    cases.assert_same_g_consumers(ham, up, dn, record["geminal"], record["geminal_inv"])

    # The stored local energy of the final step is the consumer output of the
    # carried inverse the driver handed over (ties the capture to the observable).
    atol, rtol = get_tolerance_min(cases.SAME_G_ZONES, "strict")
    rts = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (up.shape[0], 3, 3))
    state = mcmc._jit_vmap_init_det_ratio_state(ham.wavefunction_data.geminal_data, up, dn)
    reference = mcmc._jit_vmap_e_L_fast(
        ham, up, dn, rts, mcmc._geminal_pinv_from_matrix_batched(jnp.asarray(record["geminal"])), state
    )
    cases.assert_close_report({"stored e_L": ([driver.e_L[-1]], [reference])}, atol, rtol, "stored vs same-G e_L")

    cases.assert_g_generation(ham, up, dn, record["geminal"])
    assert driver.num_inverse_refreshes > 0, "the near-singular seed must trigger the refresh"


@pytest.mark.light
def test_refresh_helper_is_selective_and_exact():
    """Untriggered walkers are bit-identical; triggered ones equal pinv(G) bit-for-bit."""
    ham, r_up, r_dn = cases.build_case("h4_pbc_ae")
    gd = ham.wavefunction_data.geminal_data
    geminal, seed_inv, _, _ = mcmc._geminal_inv_batched(gd, jnp.asarray(r_up), jnp.asarray(r_dn))
    eye = jnp.eye(geminal.shape[-1], dtype=geminal.dtype)
    # walker 0/2: exact inverse of a well-conditioned matrix (never triggers);
    # walker 1/3: deliberately drifted inverse (residual 1e-6 >> threshold).
    well = geminal + 10.0 * eye
    well_inv = mcmc._geminal_pinv_from_matrix_batched(well)
    drifted = well_inv * (1.0 + 1.0e-6)
    G = jnp.stack([well[0], well[1], well[2], well[3]])
    X = jnp.stack([well_inv[0], drifted[1], well_inv[2], drifted[3]])
    indicator = np.asarray(mcmc._running_geminal_inv_probe_residual(G, X))
    threshold = mcmc._setting_module.SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD
    assert np.all(indicator[[0, 2]] <= threshold) and np.all(indicator[[1, 3]] > threshold)
    refreshed, count = mcmc._refresh_running_geminal_inv(G, X)
    assert count == 2
    assert np.isfinite(np.asarray(refreshed)).all() and np.isfinite(np.asarray(well_inv)).all()
    np.testing.assert_array_equal(np.asarray(refreshed)[[0, 2]], np.asarray(X)[[0, 2]])
    np.testing.assert_array_equal(np.asarray(refreshed)[[1, 3]], np.asarray(well_inv)[[1, 3]])
    disabled, none = mcmc._refresh_running_geminal_inv(G, X, threshold=float("inf"))
    assert none == 0
    np.testing.assert_array_equal(np.asarray(disabled), np.asarray(X))
    assert np.isfinite(np.asarray(seed_inv)).all()


# --------------------------------------------------------------------------- #
# The seed / refresh pseudo-inverse of an invertible but
# graded geminal keeps the far electron's direction.
# --------------------------------------------------------------------------- #
_FAR_DISTANCE = 10.0  # bohr: the far up and down electron of walker 1
_FAR_DN_ALONE = 14.0  # bohr: the lone far down electron of walker 2 (a column-graded G)
_FAR_STEP = 0.3  # bohr: inward move of the first up / first down electron whose ratio is judged
_POW2_GRADING = 80  # row scale 2**-80 (~8e-25, below rcond_svd): pure power-of-two grading


def _far_electron_walkers():
    """All-electron UHF N geminal and three walkers with their one-electron moves.

    Walker 0: regular configuration. Walker 1: the first up and the first down
    electron 10 bohr out (row- and column-graded G). Walker 2: only the first
    down electron, 14 bohr out (column-graded G). Returns the geminal data,
    ``r_up``, ``r_dn`` and the configurations with the first up / first down
    electron moved 0.3 bohr inward.
    """
    from jqmc.trexio_wrapper import read_trexio_file

    structure, _, _, _, gem, _ = read_trexio_file(str(cases.TREXIO_DIR / "N_ae_ccpvdz_cart.h5"), store_tuple=True)
    rng = np.random.default_rng(3)
    nucleus = np.asarray(structure.positions, dtype=np.float64)[0]
    r_up0 = nucleus + rng.normal(scale=0.8, size=(gem.num_electron_up, 3))
    r_dn0 = nucleus + rng.normal(scale=0.8, size=(gem.num_electron_dn, 3))
    up_dir = np.array([0.3, 0.5, 0.81]) / np.linalg.norm([0.3, 0.5, 0.81])
    dn_dir = np.array([-0.6, 0.2, 0.77]) / np.linalg.norm([-0.6, 0.2, 0.77])
    r_up, r_dn, r_up_moved, r_dn_moved = [], [], [], []
    for up_distance, dn_distance in ((None, None), (_FAR_DISTANCE, _FAR_DISTANCE), (None, _FAR_DN_ALONE)):
        up, dn = r_up0.copy(), r_dn0.copy()
        if up_distance is not None:
            up[0] = nucleus + up_distance * up_dir
        if dn_distance is not None:
            dn[0] = nucleus + dn_distance * dn_dir
        up_moved, dn_moved = up.copy(), dn.copy()
        up_moved[0] = up[0] - _FAR_STEP * (up[0] - nucleus) / np.linalg.norm(up[0] - nucleus)
        dn_moved[0] = dn[0] - _FAR_STEP * (dn[0] - nucleus) / np.linalg.norm(dn[0] - nucleus)
        r_up.append(up)
        r_dn.append(dn)
        r_up_moved.append(up_moved)
        r_dn_moved.append(dn_moved)
    return gem, np.stack(r_up), np.stack(r_dn), np.stack(r_up_moved), np.stack(r_dn_moved)


def _sm_ratio(geminal_inv, geminal_old, geminal_new, k, spin):
    """Production rank-1 determinant ratio of replacing row (up) or column (dn) ``k``."""
    from jqmc.determinant import _sherman_morrison_ratio_and_inverse

    one_hot = jnp.zeros((geminal_old.shape[0], 1), dtype=jnp.float64).at[k, 0].set(1.0)
    if spin == "up":
        u, v = one_hot, jnp.asarray(geminal_new[k] - geminal_old[k])[:, None]
    else:
        u, v = jnp.asarray(geminal_new[:, k] - geminal_old[:, k])[:, None], one_hot
    ratio, _ = _sherman_morrison_ratio_and_inverse(jnp.asarray(geminal_inv), u, v)
    return float(ratio)


def _exact_ratio(geminal_old, geminal_new):
    """det(new) / det(old) from two fresh LU log-determinants."""
    sign_new, log_new = np.linalg.slogdet(np.asarray(geminal_new))
    sign_old, log_old = np.linalg.slogdet(np.asarray(geminal_old))
    return float(sign_new * sign_old * np.exp(log_new - log_old))


@pytest.mark.light
def test_seed_pinv_keeps_far_electron_direction():
    """Seed and refresh pseudo-inverses give exact SM ratios for a far, invertible, graded G.

    All-electron UHF N (5 up / 2 dn). Walker 1 has its
    first up and first down electron 10 bohr from the nucleus, walker 2 only its
    first down electron 14 bohr out: both G are invertible but their singular
    values span more than ``1 / rcond_svd``, so a plain thresholded SVD drops a
    real direction and the far electron's ratio collapses (1.0 instead of 3.6).
    Every production site that seeds or re-anchors ``G^-1`` -- the MCMC/GFMC seed
    ``_geminal_inv_batched``, the MCMC inverse refresh and the GFMC observable inverse
    refresh (rebuilt-G and streaming det-state probes) -- must give, for a
    0.3 bohr inward move of the first up and of the first down electron, the
    ratio of two fresh LU determinants within the strict det_ratio pair.
    Walker 0 (regular configuration) is the control: the same ratio checks, the
    seed equal to the plain thresholded pseudo-inverse within the strict
    det_eval pair (bitwise once G is already equilibrated), and a power-of-two
    grading of one row changes the seed only by the inverse power of two in
    that column (bitwise). A truly singular G (an exactly zero AO row, two
    coincident up electrons) stays singular: finite, with the refresh probe
    residual above the refresh threshold.
    """
    from jqmc._setting import get_eps

    gem, r_up, r_dn, r_up_moved, r_dn_moved = _far_electron_walkers()
    geminal = np.asarray(mcmc._jit_vmap_geminal_all_elements(gem, jnp.asarray(r_up), jnp.asarray(r_dn)))
    moved = {
        "up": np.asarray(mcmc._jit_vmap_geminal_all_elements(gem, jnp.asarray(r_up_moved), jnp.asarray(r_dn))),
        "dn": np.asarray(mcmc._jit_vmap_geminal_all_elements(gem, jnp.asarray(r_up), jnp.asarray(r_dn_moved))),
    }
    rcond = get_eps("rcond_svd", jnp.float64)
    singular_values = np.linalg.svd(geminal, compute_uv=False)
    # Preconditions: the far walkers are below the plain cutoff, yet invertible.
    assert np.all(singular_values[1:, -1] / singular_values[1:, 0] < rcond)
    assert singular_values[0, -1] / singular_values[0, 0] > rcond
    assert np.all(np.isfinite(np.linalg.slogdet(geminal)[1]))

    threshold = mcmc._setting_module.SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD
    stale = jnp.zeros_like(jnp.asarray(geminal))  # probe residual 1 > threshold: always refreshed
    det_state = mcmc._jit_vmap_init_det_ratio_state(gem, jnp.asarray(r_up), jnp.asarray(r_dn))
    seed = mcmc._geminal_inv_batched(gem, jnp.asarray(r_up), jnp.asarray(r_dn))[1]
    mcmc_refresh, n_mcmc = mcmc._refresh_running_geminal_inv(jnp.asarray(geminal), stale)
    gfmc_refresh, n_gfmc = mcmc._refresh_observable_geminal_inv(gem, jnp.asarray(r_up), jnp.asarray(r_dn), stale)
    gfmc_state_refresh, n_state = mcmc._refresh_observable_geminal_inv(
        gem, jnp.asarray(r_up), jnp.asarray(r_dn), stale, det_state
    )
    assert (n_mcmc, n_gfmc, n_state) == (3, 3, 3)
    sites = {
        "seed _geminal_inv_batched": seed,
        "MCMC inverse refresh": mcmc_refresh,
        "GFMC inverse refresh (rebuilt G)": gfmc_refresh,
        "GFMC inverse refresh (det state)": gfmc_state_refresh,
    }
    atol, rtol = get_tolerance_min(("det_ratio",), "strict")
    for name, inverse in sites.items():
        inverse = np.asarray(inverse)
        assert np.isfinite(inverse).all(), name
        for w in range(3):
            for spin, moved_geminal in moved.items():
                ratio = _sm_ratio(inverse[w], geminal[w], moved_geminal[w], 0, spin)
                exact = _exact_ratio(geminal[w], moved_geminal[w])
                assert np.isfinite(ratio) and np.isfinite(exact), (name, w, spin, ratio, exact)
                assert abs(ratio - exact) <= atol + rtol * abs(exact), (
                    f"{name} walker {w} {spin} move: SM ratio {ratio:.10e} vs exact {exact:.10e}"
                )

    # Regular configuration: the equilibrated seed equals the plain thresholded
    # pseudo-inverse within the strict det_eval pair, and bitwise once G is
    # already equilibrated (every row/column infinity norm in [0.5, 1), all
    # scales exactly 1: the equilibration itself adds no rounding).
    def plain_pinv(g):
        U, s, Vt = jnp.linalg.svd(jnp.asarray(g), full_matrices=False)
        return np.asarray((Vt.T * jnp.where(s > rcond * s[0], 1.0 / s, 0.0)[None, :]) @ U.T)

    atol_g, rtol_g = get_tolerance_min(("det_eval",), "strict")
    np.testing.assert_allclose(np.asarray(seed)[0], plain_pinv(geminal[0]), atol=atol_g, rtol=rtol_g)
    equilibrated = geminal[0] * np.ldexp(1.0, -np.frexp(np.abs(geminal[0]).max(axis=1))[1])[:, None]
    equilibrated = equilibrated * np.ldexp(1.0, -np.frexp(np.abs(equilibrated).max(axis=0))[1])[None, :]
    np.testing.assert_array_equal(
        np.asarray(mcmc._geminal_pinv_from_matrix(jnp.asarray(equilibrated))), plain_pinv(equilibrated)
    )
    # Pure power-of-two grading of one row (the far-electron effect without
    # rounding) only rescales the matching column of the seed, bitwise.
    graded = geminal[0].copy()
    graded[0] *= 2.0**-_POW2_GRADING
    expected = np.asarray(mcmc._geminal_pinv_from_matrix(jnp.asarray(geminal[0]))).copy()
    expected[:, 0] *= 2.0**_POW2_GRADING
    np.testing.assert_array_equal(np.asarray(mcmc._geminal_pinv_from_matrix(jnp.asarray(graded))), expected)

    # A truly singular G is not hidden by the equilibration.
    zero_row = geminal[0].copy()
    zero_row[0] = 0.0
    coincident_up = r_up[:1].copy()
    coincident_up[0, 1] = coincident_up[0, 0]
    coincident = np.asarray(mcmc._jit_vmap_geminal_all_elements(gem, jnp.asarray(coincident_up), jnp.asarray(r_dn[:1])))[0]
    singular = jnp.asarray(np.stack([zero_row, coincident]))
    singular_inv = mcmc._geminal_pinv_from_matrix_batched(singular)
    assert np.isfinite(np.asarray(singular_inv)).all()
    residual = np.asarray(mcmc._running_geminal_inv_probe_residual(singular, singular_inv))
    assert np.all(residual > threshold), residual
