"""Driver-level production-vs-debug parity with a live non-local l >= 1 ECP channel.

Every other MCMC / GFMC driver row runs an ECP whose non-local part is s-only (Li,
water) or empty (H, H2 ccECP).  Here the production ``MCMC`` / ``GFMC_t`` / ``GFMC_n``
drivers and their ``_debug`` twins run on the committed Cl2 ccECP fixture, whose
non-local s AND p projectors are non-zero on both atoms, with J1 + J2.

Each row checks, in the precision mode of the run:

* the fixture precondition, from the raw ECP arrays;
* the walk (threefry keys, accepted moves / projection counters exactly, final walkers
  within the repo's fp64 walker-position pin);
* stored ``w_L`` / ``e_L`` / ``e_L2`` and the public energy and force estimators against
  the debug twin, at the same tolerance tier the existing driver rows use;
* non-vacuity: at every final walker the l >= 1 non-local contribution
  ``V_NL(full) - V_NL(l>=1 coefficients zeroed)``, from the eager all-pairs reference
  kernel, exceeds the e_L comparison envelope.  A driver that silently dropped the p
  channel therefore cannot pass the e_L comparison.

The MCMC row also runs the production driver on the l >= 1-ablated Hamiltonian (the
walk does not depend on the ECP) and requires that its ``e_L`` FAILS the comparison
with the full debug twin: a negative control for a production-only dropped channel.

The ``_debug`` twins share the non-local term selection (``_non_local_part_index``)
and, for MCMC, the Legendre table with production, so each driver row also carries a
reference built from the raw ECP arrays (``raw_array_nearest_nucleus_nonlocal``: terms
selected from ``ang_moms`` / ``max_ang_mom_plus_1``, scipy Legendre polynomials,
nearest-nucleus convention of the production kernels): the MCMC row compares the
stored production ``e_L`` of the last step with kinetic + bare Coulomb + local ECP +
that reference at each final walker and the rotation of its last key; the GFMC rows
compare the production non-local ECP energy at each final walker with it. A
production-only slip in the term selection or the Legendre table fails the row itself.
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
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from tests._finite_allclose import assert_finite_allclose
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import (
    ATOL_PYTHON_FP64_MEDIUM,
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_WARMUP_STEPS,
    MCMC_MIN_BIN_BLOCKS,
    MCMC_MIN_STEPS,
    MCMC_MIN_WARMUP_STEPS,
    RTOL_PYTHON_FP64_ZERO,
)
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t, _GFMC_n_debug, _GFMC_t_debug
from jqmc.jqmc_mcmc import MCMC, _MCMC_debug
from tests._ecp_high_l import (
    ablate_high_l,
    assert_fixture_has_live_high_l_channel,
    high_l_hamiltonian,
    assert_mcmc_stored_e_L_matches_raw_arrays,
    assert_production_nonlocal_matches_raw_arrays,
    high_l_nonlocal_contribution,
)

_ZONES = ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic")
_FORCE_ZONES = ("ao_eval", "ao_grad_lap", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic")
_GFMC_STEPS = GFMC_MIN_WARMUP_STEPS + GFMC_MIN_BIN_BLOCKS + GFMC_MIN_COLLECT_STEPS


def _assert_high_l_live(ham, r_up, r_dn, e_L_last, atol, rtol, label):
    delta = high_l_nonlocal_contribution(ham, r_up, r_dn)
    envelope = atol + rtol * np.abs(np.asarray(e_L_last, dtype=float))
    assert np.all(np.abs(delta) > envelope), (
        f"{label}: l>=1 non-local contribution {delta} does not exceed the e_L envelope {envelope}"
    )
    return delta


def _assert_walkers_close(debug_r, prod_r, label):
    assert_finite_allclose(
        np.asarray(debug_r), np.asarray(prod_r), atol=ATOL_PYTHON_FP64_MEDIUM, rtol=RTOL_PYTHON_FP64_ZERO, err_msg=label
    )


@pytest.mark.heavy
def test_mcmc_high_l_ecp_production_matches_debug():
    """MCMC production == ``_MCMC_debug`` on a live p-channel ECP; ablated production fails."""
    ham = high_l_hamiltonian()
    assert_fixture_has_live_high_l_channel(ham)
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    force_atol, force_rtol = get_tolerance_min(_FORCE_ZONES, "strict")
    kwargs = dict(
        Dt=2.0,
        mcmc_seed=34356,
        epsilon_AS=1.0e-6,
        num_walkers=2,
        comput_position_deriv=True,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        random_discretized_mesh=True,
    )
    debug = _MCMC_debug(hamiltonian_data=ham, **kwargs)
    debug.run(num_mcmc_steps=MCMC_MIN_STEPS)
    prod = MCMC(hamiltonian_data=ham, **kwargs)
    prod.run(num_mcmc_steps=MCMC_MIN_STEPS)

    np.testing.assert_array_equal(debug._MCMC_debug__jax_PRNG_key_list, prod._MCMC__jax_PRNG_key_list)
    np.testing.assert_array_equal(debug._MCMC_debug__accepted_moves, prod._MCMC__accepted_moves)
    _assert_walkers_close(debug._MCMC_debug__latest_r_up_carts, prod._MCMC__latest_r_up_carts, "MCMC r_up")
    _assert_walkers_close(debug._MCMC_debug__latest_r_dn_carts, prod._MCMC__latest_r_dn_carts, "MCMC r_dn")
    assert_mcmc_stored_e_L_matches_raw_arrays(ham, prod, atol, rtol, "MCMC")
    assert_finite_allclose(debug.w_L, prod.w_L, atol=atol, rtol=rtol, err_msg="MCMC high-l w_L")
    assert_finite_allclose(debug.e_L, prod.e_L, atol=atol, rtol=rtol, err_msg="MCMC high-l e_L")
    args = (MCMC_MIN_WARMUP_STEPS, MCMC_MIN_BIN_BLOCKS)
    _assert_finite("MCMC debug", e_L=debug.e_L, aF=debug.get_aF(*args))
    _assert_finite("MCMC production", e_L=prod.e_L, force_HF=prod.force_HF_stored, aF=prod.get_aF(*args))
    assert_finite_allclose(debug.get_E(*args), prod.get_E(*args), atol=atol, rtol=rtol, err_msg="MCMC high-l E")
    assert_finite_allclose(
        debug.get_aF(*args), prod.get_aF(*args), atol=force_atol, rtol=force_rtol, err_msg="MCMC high-l force"
    )
    _assert_high_l_live(
        ham,
        prod._MCMC__latest_r_up_carts,
        prod._MCMC__latest_r_dn_carts,
        np.asarray(prod.e_L)[-1],
        atol,
        rtol,
        "MCMC",
    )

    # Negative control: production with the p channel dropped, same walk, must fail.
    ablated = MCMC(hamiltonian_data=ablate_high_l(ham), **kwargs)
    ablated.run(num_mcmc_steps=MCMC_MIN_STEPS)
    np.testing.assert_array_equal(ablated._MCMC__jax_PRNG_key_list, prod._MCMC__jax_PRNG_key_list)
    with pytest.raises(AssertionError, match="MCMC high-l e_L"):
        assert_finite_allclose(debug.e_L, ablated.e_L, atol=atol, rtol=rtol, err_msg="MCMC high-l e_L")
    # ... and its stored e_L must miss the raw-array reference of the full Hamiltonian.
    with pytest.raises(AssertionError, match="MCMC ablated: stored e_L at final walker 0 misses the raw-array reference"):
        assert_mcmc_stored_e_L_matches_raw_arrays(ham, ablated, atol, rtol, "MCMC ablated")
    jax.clear_caches()


def _assert_finite(label, **arrays):
    for name, value in arrays.items():
        value = np.asarray(value)
        assert value.size and np.all(np.isfinite(value)), f"{label}: {name} is empty or not finite"


def _gfmc_parity(debug, prod, ham, label):
    # Same tiers as the existing GFMC_t / GFMC_n driver rows.
    level = "medium" if (label.startswith("GFMC_t") or is_mixed_precision_enabled()) else "strict"
    atol, rtol = get_tolerance_min(_ZONES, level)
    np.testing.assert_array_equal(debug._jax_PRNG_key_list, prod._jax_PRNG_key_list)
    _assert_walkers_close(debug._latest_r_up_carts, prod._latest_r_up_carts, f"{label} r_up")
    _assert_walkers_close(debug._latest_r_dn_carts, prod._latest_r_dn_carts, f"{label} r_dn")
    assert_production_nonlocal_matches_raw_arrays(ham, prod._latest_r_up_carts, prod._latest_r_dn_carts, atol, rtol, label)
    assert_finite_allclose(debug.w_L, prod.w_L, atol=atol, rtol=rtol, err_msg=f"{label} w_L")
    assert_finite_allclose(debug.e_L, prod.e_L, atol=atol, rtol=rtol, err_msg=f"{label} e_L")
    args = (GFMC_MIN_WARMUP_STEPS, GFMC_MIN_BIN_BLOCKS)
    for run, side in ((debug, "debug"), (prod, "production")):
        _assert_finite(
            f"{label} {side}",
            e_L=run.e_L,
            force_HF=run._stored_force_HF,
            force_PP=run._stored_force_PP,
            aF=run.get_aF(*args),
        )
    assert_finite_allclose(debug.get_E(*args), prod.get_E(*args), atol=atol, rtol=rtol, err_msg=f"{label} E")
    assert_finite_allclose(debug.get_aF(*args), prod.get_aF(*args), atol=atol, rtol=rtol, err_msg=f"{label} force")
    _assert_high_l_live(ham, prod._latest_r_up_carts, prod._latest_r_dn_carts, np.asarray(prod.e_L)[-1], atol, rtol, label)


@pytest.mark.heavy
@pytest.mark.parametrize("non_local_move", ["tmove", "dltmove"])
def test_gfmc_t_high_l_ecp_production_matches_debug(non_local_move):
    """GFMC_t production == ``_GFMC_t_debug`` on a live p-channel ECP (T-move / DLT-move)."""
    ham = high_l_hamiltonian()
    assert_fixture_has_live_high_l_channel(ham)
    kwargs = dict(
        hamiltonian_data=ham,
        num_walkers=2,
        num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
        mcmc_seed=3446,
        tau=0.03,
        alat=0.30,
        random_discretized_mesh=True,
        comput_position_deriv=True,
        non_local_move=non_local_move,
    )
    debug = _GFMC_t_debug(**kwargs)
    debug.run(num_mcmc_steps=_GFMC_STEPS)
    prod = GFMC_t(**kwargs)
    prod.run(num_mcmc_steps=_GFMC_STEPS)
    np.testing.assert_array_equal(debug.average_projection_counter, prod.average_projection_counter)
    _gfmc_parity(debug, prod, ham, f"GFMC_t/{non_local_move}")
    jax.clear_caches()


@pytest.mark.heavy
@pytest.mark.parametrize("non_local_move", ["tmove", "dltmove"])
def test_gfmc_n_high_l_ecp_production_matches_debug(non_local_move):
    """GFMC_n production == ``_GFMC_n_debug`` on a live p-channel ECP (T-move / DLT-move)."""
    ham = high_l_hamiltonian()
    assert_fixture_has_live_high_l_channel(ham)
    kwargs = dict(
        hamiltonian_data=ham,
        num_walkers=2,
        num_mcmc_per_measurement=10,
        num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
        mcmc_seed=3446,
        E_scf=-30.0,
        alat=0.30,
        random_discretized_mesh=True,
        comput_position_deriv=True,
        non_local_move=non_local_move,
    )
    debug = _GFMC_n_debug(**kwargs)
    debug.run(num_mcmc_steps=_GFMC_STEPS)
    prod = GFMC_n(**kwargs)
    prod.run(num_mcmc_steps=_GFMC_STEPS)
    _gfmc_parity(debug, prod, ham, f"GFMC_n/{non_local_move}")
    jax.clear_caches()


_KERNEL_SEEDS = (0, 1, 2)


def _production_nonlocal(ham, r_up, r_dn):
    """Per-configuration V_NL from the two production molecular kernels (identity mesh)."""
    import jax.numpy as jnp

    from jqmc.coulomb_potential import compute_ecp_non_local_parts_all_pairs, compute_ecp_non_local_parts_nearest_neighbors

    cpd, wf = ham.coulomb_potential_data, ham.wavefunction_data
    rt = jnp.eye(3)
    # NN = number of nuclei: the nearest-neighbour kernel then sums every atom, like the reference.
    n_atoms = len(np.asarray(ham.structure_data.positions))
    nn = compute_ecp_non_local_parts_nearest_neighbors(cpd, wf, jnp.asarray(r_up), jnp.asarray(r_dn), rt, NN=n_atoms)[3]
    ap = compute_ecp_non_local_parts_all_pairs(cpd, wf, jnp.asarray(r_up), jnp.asarray(r_dn), rt)[3]
    return float(np.real(np.asarray(nn))), float(np.real(np.asarray(ap)))


def _seeded_cl2_configuration(ham, seed):
    geminal = ham.wavefunction_data.geminal_data
    positions = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_up, n_dn = int(geminal.num_electron_up), int(geminal.num_electron_dn)
    r_up = positions[rng.integers(0, len(positions), n_up)] + rng.normal(scale=0.7, size=(n_up, 3))
    r_dn = positions[rng.integers(0, len(positions), n_dn)] + rng.normal(scale=0.7, size=(n_dn, 3))
    return r_up, r_dn


def _high_l_kernel_mismatches(ham):
    """Messages for every configuration where a production V_NL kernel misses the reference (empty = pass)."""
    from jqmc.coulomb_potential import _compute_ecp_non_local_parts_all_pairs_debug

    atol, rtol = get_tolerance_min(_ZONES, "strict")
    failures = []
    for seed in _KERNEL_SEEDS:
        r_up, r_dn = _seeded_cl2_configuration(ham, seed)
        ref = float(
            np.real(
                _compute_ecp_non_local_parts_all_pairs_debug(ham.coulomb_potential_data, ham.wavefunction_data, r_up, r_dn)[3]
            )
        )
        delta = high_l_nonlocal_contribution(ham, r_up[None], r_dn[None])[0]
        assert np.isfinite(ref) and abs(delta) > atol + rtol * abs(ref), (
            f"seed={seed}: l>=1 non-local contribution {delta} is inside the V_NL envelope of {ref}"
        )
        for name, value in zip(("nearest-neighbors", "all-pairs"), _production_nonlocal(ham, r_up, r_dn), strict=True):
            assert np.isfinite(value), (seed, name, value)
            if not np.isclose(value, ref, atol=atol, rtol=rtol, equal_nan=False):
                failures.append(f"seed={seed} {name}: production V_NL {value!r} != reference {ref!r}")
    return failures


@pytest.mark.light
def test_high_l_nonlocal_production_kernels_match_raw_array_reference():
    """Production molecular V_NL kernels equal the eager reference on the live Cl2 p channel.

    The drivers above compare production with ``_debug`` twins that share the
    non-local term selection and, for MCMC, the production Legendre table, so a
    production-only slip in either is invisible to them. Here, at three seeded
    configurations around the Cl nuclei (identity quadrature rotation), the sum of
    the non-local ECP terms from the two production molecular kernels
    (nearest-neighbour with NN equal to the number of nuclei, and all-pairs) must equal the eager all-pairs reference
    kernel, which selects the non-local terms from the raw ``ang_moms`` /
    ``max_ang_mom_plus_1`` arrays and uses scipy Legendre polynomials. The l >= 1
    contribution at each configuration must exceed the comparison envelope, so a
    dropped p channel cannot pass. Strict Coulomb-zone policy of the current mode.
    """
    ham = high_l_hamiltonian()
    assert_fixture_has_live_high_l_channel(ham)
    failures = _high_l_kernel_mismatches(ham)
    assert failures == [], "; ".join(failures)
    jax.clear_caches()
