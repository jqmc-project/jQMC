"""Issues #131 / #134: the native AS regularization factor is evaluated robustly.

* #131: ``compute_as_regularization_factor`` / ``as_recompute`` returned 0 for a
  geminal row below ~1e-154 (``S * F = 0 * inf``); at ``epsilon_AS > 0`` the
  native sweep then rejected every proposal (frozen walker).
* #131 (complex): ``compute_as_regularization_factor_complex`` likewise returned 0
  for a valid complex geminal below ~1e-154 (twist witness at the end of this module).
* #134: ``as_proposed`` updated the proposed cross column (up move) / row (dn
  move) norms as ``n2 + 2 G v + v^2``, which cancels to 0 or below when the moved
  electron dominates that column / row and ``|R| < ~1e-8``; ``R_AS' = 0`` made
  the AS ratio ``inf`` and moves with exact acceptance below 1e-16 were accepted.
* #131 / #134 (complex proposed state, He Bloch witnesses at the end of this module):
  the complex sweep's AS factor of a proposal must use the rebuilt moved line with
  the exact moved inverse line ``G^-1 e_k / R`` (the rank-one update cancels for a
  large ratio), and its complex quotients by ``R`` must stay finite when ``|R|^2``
  exceeds the float64 range.

References (see ``tests/_as_factor_cases.py``) are independent of the production
ratio / inverse-update / AS / proposal helpers.
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

from math import lgamma, log, log1p

import jax.numpy as jnp
import jqmc_rs
import numpy as np
import pytest
from jax import random as jax_random

from jqmc._precision import get_tolerance_min, precision_policy_snapshot
from jqmc.atomic_orbital import build_AOs_cart_pgto_complex_data
from jqmc.coulomb_potential import Coulomb_potential_data, Ewald_data
from jqmc.determinant import SD_data, compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.wavefunction import Wavefunction_data
from tests import _as_factor_cases as cases
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_AS_ZONES = ("det_eval", "det_ratio")
_KEY0 = 777
# One native threefry key per third of the unit interval, so that the uniform draw
# covers small, middle and large values (as the XLA draws 0.01 / 0.5 / 0.99).
_DRAW_BINS = ((0.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0))
# Upper bound of the second threefry key word searched by ``_keys``.
_KEY_SEARCH_LIMIT = 400000


def _keys(spin, n_up, n_dn):
    """Threefry keys whose move selects electron 0 of ``spin`` along x, one per draw bin.

    A ``while`` loop on purpose: the GDD case-binding extractor statically unrolls a
    ``for`` loop over a literal ``range`` once per value, which made binding
    generation (run at pytest start-up) take tens of minutes for this module.
    """
    found = {}
    counter = 0
    while counter < _KEY_SEARCH_LIMIT and len(found) < len(_DRAW_BINS):
        draws = jqmc_rs._move_draws(_KEY0, counter, n_up, n_dn)
        if draws[0] == (spin == "up") and draws[1] == 0 and draws[2] == 0:
            for low, high in _DRAW_BINS:
                if low <= draws[3] < high:
                    found.setdefault((low, high), ((_KEY0, counter), float(draws[3])))
        counter += 1
    if len(found) == len(_DRAW_BINS):
        return [found[b] for b in _DRAW_BINS]
    raise AssertionError("no threefry key selects the requested move")


def _ctx(ham):
    wf = ham.wavefunction_data
    return jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        False,
        False,
        False,
        want_force=False,
        want_gfmc=False,
        precision_policy=precision_policy_snapshot(),
    )


@pytest.mark.parametrize(
    "name", ["power-underflow", "power-cancel", "h2-up-12", "h2-up-17", "h2-down-12", "h2-down-17", "water-up-30"]
)
@pytest.mark.parametrize("epsilon", [0.0, 0.1])
def test_native_as_factor_single_move_decisions_are_exact(name, epsilon):
    """Native sweep and single-move decisions with an underflowing current row or cancelling proposed norms.

    Same witnesses as the XLA twin (x^12 model x = 1e-14 -> 0.3 and 1.0 -> 0.05; H2 cc-pVDZ electron 0 up
    and down moved +12 / +17 bohr; water ccECP cc-pVQZ up electron 0 moved +30 bohr; no Jastrow, Dt = 2),
    from an exact numpy inverse. For three threefry keys selecting that move with a uniform draw in each
    third of [0, 1), the production ``ObsCtx.mcmc_sweep`` (one move) and ``jqmc_rs._single_move`` must
    accept exactly when the draw is below the exact AS-guided acceptance (rebuilt-G slogdet or closed form,
    numpy Kato transition density, row-rescaled log-domain R_AS reference; R_AS = 1 exactly for every 1x1
    G), at epsilon_AS = 0 and 0.1.
    """
    ham, nuclei, charges, model, spin, before, normal = cases.case(name)
    geminal = ham.wavefunction_data.geminal_data
    matrix = cases.reference_geminal(ham, before, model)
    with np.errstate(over="ignore"):
        inverse = np.linalg.inv(matrix)
    after = cases.proposed(before, normal, spin, nuclei, charges)
    probability = cases.exact_acceptance(ham, before, after, spin, nuclei, charges, model, epsilon)
    ctx = _ctx(ham)
    observed, expected = [], []
    for (k0, k1), draw in _keys(spin, geminal.num_electron_up, geminal.num_electron_dn):
        expected.append(int(draw < probability))
        sweep = ctx.mcmc_sweep(
            nuclei, charges, before[0], before[1], matrix, inverse, cases.DT, epsilon, k0, k1, np.asarray([normal]), None, 0
        )
        single = jqmc_rs._single_move(
            geminal,
            ham.wavefunction_data.jastrow_data,
            nuclei,
            charges,
            before[0],
            before[1],
            matrix,
            inverse,
            cases.DT,
            epsilon,
            k0,
            k1,
            normal,
        )
        observed.append((int(sweep["accepted"]), int(single["accepted"])))
    assert observed == [(e, e) for e in expected], (name, epsilon, probability, observed)


@pytest.mark.parametrize("scale", [1.0e-160, 1.0e-170])
@pytest.mark.parametrize("spin", ["up", "down"])
def test_native_as_factor_is_invariant_under_a_tiny_row_or_column(scale, spin):
    """The native AS factor is robust to a geminal row / column below sqrt(tiny).

    Same construction as the XLA twin: water ccECP cc-pVQZ PCG64(2025) G with row 0 (up) or column 0 (down)
    scaled by c in {1e-160, 1e-170} and the exact inverse's matching column / row by 1/c. The native factor
    must equal the row-rescaled log-domain reference and the native factor at c = 1e-60 (linear range) at
    the det_eval / det_ratio strict pair; for the 1x1 x^12 G at x = 1e-14 it must be 1.
    """
    ham, _, _, before = cases.water_model()
    matrix = cases.reference_geminal(ham, before, "water")
    inverse = np.linalg.inv(matrix)

    def scaled_pair(c):
        scaled, scaled_inverse = matrix.copy(), inverse.copy()
        if spin == "up":
            scaled[0, :] *= c
            scaled_inverse[:, 0] /= c
        else:
            scaled[:, 0] *= c
            scaled_inverse[0, :] /= c
        return np.ascontiguousarray(scaled), np.ascontiguousarray(scaled_inverse)

    atol, rtol = get_tolerance_min(_AS_ZONES, "strict")
    scaled, scaled_inverse = scaled_pair(scale)
    actual = jqmc_rs.compute_AS_regularization_factor_fast_update(scaled, scaled_inverse)
    linear = jqmc_rs.compute_AS_regularization_factor_fast_update(*scaled_pair(cases.LINEAR_RANGE_ROW_SCALE))
    np.testing.assert_allclose(actual, np.exp(cases.reference_ln_as_factor(scaled)), atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual, linear, atol=atol, rtol=rtol)
    power_ham, _, _ = cases.power_model()
    tiny = cases.reference_geminal(power_ham, cases.power_configuration(1.0e-14), "power")
    with np.errstate(over="ignore"):
        tiny_inverse = np.linalg.inv(tiny)
    np.testing.assert_allclose(
        jqmc_rs.compute_AS_regularization_factor_fast_update(tiny, tiny_inverse), 1.0, atol=atol, rtol=rtol
    )


# Complex (twist) witness of #131 for ``compute_as_regularization_factor_complex``.
_COMPLEX_PATTERN = "twist_sphepgto_jnone"
_COMPLEX_MO_SCALE = 1.0e-85
_COMPLEX_DT = 1.2
_COMPLEX_NORMAL = 0.1
_COMPLEX_KEYS = ((777, 35), (0, 13), (29, 63))


def _complex_twist_witness():
    """Lithium twist one-pair geminal with both spin MO coefficient arrays scaled by 1e-85 (complex128 kept), PCG64(13) start."""
    ham = rs_ham(_COMPLEX_PATTERN)
    geminal = ham.wavefunction_data.geminal_data
    scaled = geminal.replace(
        orb_data_up_spin=geminal.orb_data_up_spin.replace(
            mo_coefficients=geminal.orb_data_up_spin.mo_coefficients * _COMPLEX_MO_SCALE
        ),
        orb_data_dn_spin=geminal.orb_data_dn_spin.replace(
            mo_coefficients=geminal.orb_data_dn_spin.mo_coefficients * _COMPLEX_MO_SCALE
        ),
    )
    structure = ham.structure_data
    nuclei = np.asarray(structure.positions, dtype=np.float64)
    charges = np.asarray(structure.atomic_numbers, dtype=np.float64)
    if ham.coulomb_potential_data.ecp_flag:
        charges = charges - np.asarray(ham.coulomb_potential_data.z_cores, dtype=np.float64)
    cell = np.asarray(structure.cell, dtype=np.float64)
    rng = np.random.default_rng(13)
    start = (rng.random((geminal.num_electron_up, 3)) @ cell, rng.random((geminal.num_electron_dn, 3)) @ cell)
    return ham, geminal, scaled, nuclei, charges, cell, int(structure._n_max), start


def _complex_reference_decision(geminal, nuclei, charges, start, key):
    """Independent Metropolis decision of one move: jax threefry draws, numpy Kato density, |det G'/det G|^2.

    Same draw order as ``test_complex_sweep._independent_acceptance_reference`` (choice, up index, down index,
    normal slot, axis, uniform); the determinant ratio from ``compute_geminal_all_elements`` at both positions.
    """
    up, dn = start
    rng_key = jnp.asarray(key, dtype=jnp.uint32)
    rng_key, sub = jax_random.split(rng_key)
    is_up = int(jax_random.randint(sub, (), 0, len(up) + len(dn))) < len(up)
    rng_key, sub = jax_random.split(rng_key)
    up_index = int(jax_random.randint(sub, (), 0, len(up)))
    rng_key, sub = jax_random.split(rng_key)
    dn_index = int(jax_random.randint(sub, (), 0, len(dn)))
    index = up_index if is_up else dn_index
    old = (up if is_up else dn)[index]
    rng_key, sub = jax_random.split(rng_key)
    rng_key, sub = jax_random.split(rng_key)
    axis = int(jax_random.randint(sub, (), 0, 3))
    width_old = cases.width(old, nuclei, charges) * _COMPLEX_DT
    new = old.copy()
    new[axis] += _COMPLEX_NORMAL * width_old
    width_new = cases.width(new, nuclei, charges) * _COMPLEX_DT
    transition = (width_old / width_new) * np.exp(
        -float(np.sum((new - old) ** 2)) / 2 * (width_new ** (-2) - width_old ** (-2))
    )
    after_up, after_dn = up.copy(), dn.copy()
    (after_up if is_up else after_dn)[index] = new

    def det(configuration):
        return complex(np.linalg.det(np.asarray(compute_geminal_all_elements(geminal, *configuration))))

    probability = min(1.0, abs(det((after_up, after_dn)) / det(start)) ** 2 * transition)
    rng_key, sub = jax_random.split(rng_key)
    draw = float(jax_random.uniform(sub, (), dtype=jnp.float64))
    return int(draw < probability)


@pytest.mark.parametrize("epsilon", [0.0, 0.1])
def test_native_complex_as_factor_tiny_twist_geminal_keeps_decisions(epsilon):
    """Native complex sweep decisions with a valid tiny complex twist geminal are those of the exact acceptance.

    Committed lithium ccECP twist cell (lithium_k025, no Jastrow, one pair, complex128 MOs) with both spin MO
    coefficient arrays scaled by 1e-85, PCG64(13) uniform start in the cell: G = c^2 G_1 ~ 3.8e-173 is nonzero
    with a finite inverse 1/G. For a nonzero complex scalar S = |G|^2 and F = 1/|G|^2, so R_AS = 1 exactly before and after every move, the AS
    guiding ratio is 1 and the decision at epsilon_AS = 0.1 is that at 0. For the threefry keys (777, 35), (0, 13),
    (29, 63), Dt = 1.2 and injected normal 0.1, the production ``ObsCtx.mcmc_sweep_complex`` and the per-step
    ``ObsCtx.mcmc_sweep_complex_batch`` (one walker per key) must accept exactly when the independent reference
    (jax threefry draws, numpy Kato transition density, |det G'/det G|^2 of the unscaled geminal -- the common MO
    scale cancels in the ratio) does.
    """
    ham, geminal, scaled, nuclei, charges, cell, n_max, start = _complex_twist_witness()
    unscaled_matrix = np.asarray(compute_geminal_all_elements(geminal, *start), dtype=np.complex128)
    matrix = np.ascontiguousarray(np.asarray(compute_geminal_all_elements(scaled, *start), dtype=np.complex128))
    # Nested validity of the witness: a nonzero 1x1 complex G below sqrt(tiny) with a finite exact inverse,
    # equal to the common scale squared times the unscaled G.
    assert matrix.shape == (1, 1) and np.iscomplexobj(scaled.orb_data_up_spin.mo_coefficients)
    assert 0.0 < abs(matrix[0, 0]) < cases.UNDERFLOW_ROW_NORM, matrix
    inverse = np.ascontiguousarray(1.0 / matrix)
    assert np.all(np.isfinite(inverse)), inverse
    atol, rtol = get_tolerance_min(_AS_ZONES, "strict")
    np.testing.assert_allclose(matrix / _COMPLEX_MO_SCALE**2, unscaled_matrix, atol=atol, rtol=rtol)

    expected = [_complex_reference_decision(geminal, nuclei, charges, start, key) for key in _COMPLEX_KEYS]
    ctx = jqmc_rs.build_obs_ctx(
        scaled,
        ham.wavefunction_data.jastrow_data,
        ham.coulomb_potential_data,
        True,
        True,
        False,
        want_force=False,
        want_gfmc=False,
        precision_policy=precision_policy_snapshot(),
    )
    normals = np.asarray([_COMPLEX_NORMAL])
    single = [
        int(
            ctx.mcmc_sweep_complex(
                nuclei, charges, start[0], start[1], matrix, inverse, _COMPLEX_DT, epsilon, k0, k1, normals, cell, n_max
            )["accepted"]
        )
        for k0, k1 in _COMPLEX_KEYS
    ]
    walkers = len(_COMPLEX_KEYS)
    batch = ctx.mcmc_sweep_complex_batch(
        nuclei,
        charges,
        np.ascontiguousarray(np.broadcast_to(start[0], (walkers, *start[0].shape))),
        np.ascontiguousarray(np.broadcast_to(start[1], (walkers, *start[1].shape))),
        np.ascontiguousarray(np.broadcast_to(matrix, (walkers, 1, 1))),
        np.ascontiguousarray(np.broadcast_to(inverse, (walkers, 1, 1))),
        _COMPLEX_DT,
        epsilon,
        np.asarray(_COMPLEX_KEYS, dtype=np.uint32),
        np.full((walkers, 1), _COMPLEX_NORMAL),
        cell,
        n_max,
    )
    batched = [int(value) for value in np.asarray(batch["accepted"]).ravel()]
    assert single == expected and batched == expected, (epsilon, expected, single, batched)


# Complex (Bloch) witnesses of the AS proposed state (PR #144): a neutral He
# atom in a 100-bohr cubic cell at a non-Gamma twist with one closed-form complex Cartesian Bloch orbital.
_BLOCH_CELL = 100.0
_BLOCH_TWIST = (0.25, 0.125, 0.0)
_BLOCH_CHARGE = 2
_BLOCH_NEAR = (0.4, 0.3, 0.4)
# A far electron (|R| ~ 1.2e18) moved back to the nucleus; acceptance 1/2.
_BLOCH_LINE_FAR = (6.6, 0.3, 0.4)
_BLOCH_LINE_DT = 0.7479083679405935
_BLOCH_LINE_NORMAL = -9.343232587286982
# An electron at x = 1e-14 of the x^12 orbital moved to x = 0.3 (|G| ~ 1e-178, |R| ~ 1e161); acceptance 1.
_BLOCH_EXTREME_POWER = 12
_BLOCH_EXTREME_START = (1.0e-14, 0.3, 0.4)
_BLOCH_EXTREME_TARGET_X = 0.3
_BLOCH_EXTREME_DT = 2.0
# Threefry keys whose single move selects the requested spin along x (checked against the jax reference draws).
_BLOCH_LINE_KEYS = {"up": ((777, 18), (777, 0), (777, 5)), "down": ((777, 2), (777, 3), (777, 15))}
_BLOCH_EXTREME_KEYS = {"up": ((777, 0), (777, 5), (777, 8)), "down": ((777, 2), (777, 3), (777, 15))}
# The squared ratio |R|^2 of a proposal above this bound is not representable in float64 (num_complex's
# a / R forms it unscaled).
_LN_FLOAT64_MAX = float(np.log(np.finfo(np.float64).max))


def _bloch_he_model(power):
    """Neutral He, one complex Bloch orbital phi = N_p x^p (1 + i x) exp(-r^2) per spin, no Jastrow.

    Cubic 100-bohr cell, He at the origin, twist (0.25, 0.125, 0); Cartesian Bloch AOs x^p exp(-r^2) and
    x^(p+1) exp(-r^2) with MO coefficients (1, i sqrt(2p+1)/2), so that phi = N_p x^p (1 + i x) exp(-r^2) in the home
    cell (N_p the normalization of x^p exp(-r^2)); the down spin carries the time-reversed partner conj(phi).
    """
    structure = Structure_data.build(
        positions=np.zeros((1, 3)),
        pbc_flag=True,
        vec_a=[_BLOCH_CELL, 0.0, 0.0],
        vec_b=[0.0, _BLOCH_CELL, 0.0],
        vec_c=[0.0, 0.0, _BLOCH_CELL],
        atomic_numbers=(_BLOCH_CHARGE,),
        element_symbols=("He",),
        atomic_labels=("He1",),
    )
    aos = build_AOs_cart_pgto_complex_data(
        structure,
        k_point=_BLOCH_TWIST,
        nucleus_index=(0, 0),
        num_ao=2,
        num_ao_prim=2,
        orbital_indices=(0, 1),
        exponents=np.ones(2),
        coefficients=np.ones(2),
        angular_momentums=(power, power + 1),
        polynominal_order_x=(power, power + 1),
        polynominal_order_y=(0, 0),
        polynominal_order_z=(0, 0),
    )
    orbital = MOs_data.build(aos_data=aos, mo_coefficients=np.array([[1.0, 0.5j * np.sqrt(2 * power + 1)]]))
    geminal = SD_data.build(num_electron_up=1, num_electron_dn=1, mos_up=orbital)
    bare = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
    coulomb = Coulomb_potential_data.build(
        structure_data=structure,
        ecp_flag=False,
        ewald_data=Ewald_data.build(structure_data=structure, coulomb_potential_data=bare, n_electron=2),
    )
    ham = Hamiltonian_data(
        structure_data=structure,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
        coulomb_potential_data=coulomb,
    )
    nuclei = np.zeros((1, 3))
    charges = np.array([float(_BLOCH_CHARGE)])
    return ham, geminal, nuclei, charges, np.asarray(structure.cell, dtype=np.float64), int(structure._n_max)


def _bloch_ln_norm(power):
    """Log normalization ln N_p of the unit-exponent Cartesian GTO x^p exp(-r^2): (2/pi)^(3/4) 2^p / sqrt((2p-1)!!)."""
    ln_double_factorial = lgamma(2 * power + 1) - power * log(2.0) - lgamma(power + 1)
    return 0.75 * log(2.0 / np.pi) + power * log(2.0) - 0.5 * ln_double_factorial


def _bloch_phi(power, position):
    """Closed-form phi = N_p x^p (1 + i x) exp(-r^2) at a home-cell position (numpy only)."""
    x = float(position[0])
    magnitude = np.exp(_bloch_ln_norm(power) + power * log(abs(x)) - float(position @ position))
    return complex(magnitude * np.sign(x) ** power * (1.0 + 1.0j * x))


def _bloch_ln_abs_phi(power, position):
    x = float(position[0])
    return _bloch_ln_norm(power) + power * log(abs(x)) - float(position @ position) + 0.5 * log1p(x * x)


def _bloch_state(power, up, dn):
    """Closed-form 1x1 G = phi(r_up) conj(phi(r_dn)) and its exact complex inverse."""
    matrix = np.array([[_bloch_phi(power, up[0]) * np.conj(_bloch_phi(power, dn[0]))]], dtype=np.complex128)
    return np.ascontiguousarray(matrix), np.ascontiguousarray(1.0 / matrix)


def _reference_draws(key, n_up, n_dn):
    """(is_up, index, axis, uniform) of one move from jax threefry in the native sweep order.

    Same order as ``_complex_reference_decision``: choice, up index, down index, normal slot (injected), axis,
    uniform.
    """
    rng_key = jnp.asarray(key, dtype=jnp.uint32)
    rng_key, sub = jax_random.split(rng_key)
    is_up = int(jax_random.randint(sub, (), 0, n_up + n_dn)) < n_up
    rng_key, sub = jax_random.split(rng_key)
    up_index = int(jax_random.randint(sub, (), 0, n_up))
    rng_key, sub = jax_random.split(rng_key)
    dn_index = int(jax_random.randint(sub, (), 0, n_dn))
    rng_key, sub = jax_random.split(rng_key)
    rng_key, sub = jax_random.split(rng_key)
    axis = int(jax_random.randint(sub, (), 0, 3))
    rng_key, sub = jax_random.split(rng_key)
    draw = float(jax_random.uniform(sub, (), dtype=jnp.float64))
    return is_up, (up_index if is_up else dn_index), axis, draw


def _bloch_reference(power, start, spin, dt, normal, nuclei, charges, keys):
    """Exact acceptance and per-key decisions of the x move of electron 0 of ``spin`` (numpy / jax only).

    ln acceptance = min(0, 2 (ln|phi(r')| - ln|phi(r)|) + ln T'/T) with the closed-form orbital and the Kato-width
    Gaussian transition density. R_AS = 1 exactly for every nonzero 1x1 complex G (S = |G|^2, F = 1/|G|^2), so the
    acceptance is the same at every epsilon_AS. Every key must select that move.
    """
    old = np.asarray((start[0] if spin == "up" else start[1])[0], dtype=np.float64)
    width_old = cases.width(old, nuclei, charges) * dt
    new = old.copy()
    new[0] += normal * width_old
    width_new = cases.width(new, nuclei, charges) * dt
    ln_transition = log(width_old / width_new) - float(np.sum((new - old) ** 2)) / 2 * (width_new ** (-2) - width_old ** (-2))
    ln_ratio = _bloch_ln_abs_phi(power, new) - _bloch_ln_abs_phi(power, old)
    probability = float(np.exp(min(0.0, 2.0 * ln_ratio + ln_transition)))
    decisions = []
    for key in keys:
        is_up, index, axis, draw = _reference_draws(key, 1, 1)
        assert (is_up, index, axis) == (spin == "up", 0, 0), (key, is_up, index, axis)
        decisions.append(int(draw < probability))
    return probability, ln_ratio, decisions


def _bloch_native_decisions(ham, geminal, nuclei, charges, cell, n_max, start, matrix, inverse, dt, epsilon, normal, keys):
    """Accept flags of the production ``ObsCtx.mcmc_sweep_complex`` (one move per key) and of the per-step batch."""
    ctx = jqmc_rs.build_obs_ctx(
        geminal,
        ham.wavefunction_data.jastrow_data,
        ham.coulomb_potential_data,
        True,
        True,
        False,
        want_force=False,
        want_gfmc=False,
        precision_policy=precision_policy_snapshot(),
    )
    up, dn = (np.ascontiguousarray(np.asarray(part, dtype=np.float64)) for part in start)
    normals = np.asarray([normal])
    single = [
        int(
            ctx.mcmc_sweep_complex(nuclei, charges, up, dn, matrix, inverse, dt, epsilon, k0, k1, normals, cell, n_max)[
                "accepted"
            ]
        )
        for k0, k1 in keys
    ]
    walkers = len(keys)
    batch = ctx.mcmc_sweep_complex_batch(
        nuclei,
        charges,
        np.ascontiguousarray(np.broadcast_to(up, (walkers, *up.shape))),
        np.ascontiguousarray(np.broadcast_to(dn, (walkers, *dn.shape))),
        np.ascontiguousarray(np.broadcast_to(matrix, (walkers, 1, 1))),
        np.ascontiguousarray(np.broadcast_to(inverse, (walkers, 1, 1))),
        dt,
        epsilon,
        np.asarray(keys, dtype=np.uint32),
        np.full((walkers, 1), normal),
        cell,
        n_max,
    )
    return single, [int(value) for value in np.asarray(batch["accepted"]).ravel()]


def _bloch_check_orbital(power, geminal):
    """Nested validity: the closed-form G equals the production geminal at a well-scaled home-cell configuration."""
    up, dn = np.array([[1.1, 0.3, 0.4]]), np.array([[0.9, 0.3, 0.4]])
    produced = np.asarray(compute_geminal_all_elements(geminal, up, dn), dtype=np.complex128)
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval"), "strict")
    np.testing.assert_allclose(produced / _bloch_state(power, up, dn)[0], 1.0, atol=atol, rtol=rtol)


def _bloch_start(spin, moved, other):
    moved, other = np.array([moved]), np.array([other])
    return (moved, other) if spin == "up" else (other, moved)


@pytest.mark.parametrize("spin", ["up", "down"])
@pytest.mark.parametrize("epsilon", [0.0, 0.1])
def test_native_complex_as_proposed_moved_line_keeps_decisions(spin, epsilon):
    """The native complex AS factor of a proposal uses the rebuilt moved line and the exact moved inverse line.

    Neutral He in a 100-bohr cubic cell at twist (0.25, 0.125, 0), no Jastrow, one complex Cartesian Bloch orbital
    phi = N_0 (1 + i x) exp(-r^2) per spin. Electron 0 of ``spin`` at (6.6, 0.3, 0.4) (the other at (0.4, 0.3, 0.4))
    is moved along x by Dt = 0.7479083679405935 and injected normal -9.343232587286982 to x = 0.3: |R| ~ 1.2e18,
    exact acceptance 1/2 (closed-form orbital, Kato-width Gaussian transition density). Rank-one inverse updates
    G^-1 - (G^-1 u)(v^T G^-1)/R cancel for such a ratio, so the AS factor of the proposal is that of an inverse
    without digits unless the moved inverse line is formed as G^-1 e_k / R. R_AS = 1 exactly for every nonzero 1x1
    complex G, hence the decision at epsilon_AS = 0.1 equals that at 0. For the threefry keys of ``_BLOCH_LINE_KEYS``
    (draws below and above 1/2), ``ObsCtx.mcmc_sweep_complex`` and ``ObsCtx.mcmc_sweep_complex_batch`` must accept
    exactly when the independent jax threefry draw is below the exact acceptance.
    """
    ham, geminal, nuclei, charges, cell, n_max = _bloch_he_model(0)
    _bloch_check_orbital(0, geminal)
    start = _bloch_start(spin, _BLOCH_LINE_FAR, _BLOCH_NEAR)
    matrix, inverse = _bloch_state(0, *start)
    keys = _BLOCH_LINE_KEYS[spin]
    probability, ln_ratio, expected = _bloch_reference(
        0, start, spin, _BLOCH_LINE_DT, _BLOCH_LINE_NORMAL, nuclei, charges, keys
    )
    # Non-vacuity: a large ratio (the cancelling regime of the rank-one inverse update), an acceptance strictly inside
    # (0, 1) and keys on both sides of it.
    assert ln_ratio > -log(np.finfo(np.float64).eps) and 0.0 < probability < 1.0 and 0 < sum(expected) < len(keys)
    single, batched = _bloch_native_decisions(
        ham, geminal, nuclei, charges, cell, n_max, start, matrix, inverse, _BLOCH_LINE_DT, epsilon, _BLOCH_LINE_NORMAL, keys
    )
    assert single == expected and batched == expected, (spin, epsilon, probability, expected, single, batched)


@pytest.mark.parametrize("spin", ["up", "down"])
@pytest.mark.parametrize("epsilon", [0.0, 0.1])
def test_native_complex_as_extreme_finite_proposal_keeps_decisions(spin, epsilon):
    """A finite complex proposal whose |R|^2 exceeds the float64 range keeps the exact AS decision.

    Same He Bloch cell with phi = N_12 x^12 (1 + i x) exp(-r^2): electron 0 of ``spin`` at (1e-14, 0.3, 0.4) (the
    other at (0.4, 0.3, 0.4)), so |G| ~ 1e-178 with the finite exact inverse 1/G (injected in float64 in both
    precision modes), moved along x to 0.3 (Dt = 2). |R| ~ 1e161 is finite but |R|^2 is not representable, so the
    unscaled complex quotients (G^-1 u)/R and (v^T G^-1)/R of the proposed inverse are NaN; the exact acceptance is 1
    (closed-form orbital, Kato-width Gaussian transition density, R_AS = 1 exactly for the 1x1 G at every epsilon_AS).
    ``ObsCtx.mcmc_sweep_complex`` and ``ObsCtx.mcmc_sweep_complex_batch`` must accept for every key of
    ``_BLOCH_EXTREME_KEYS``, at epsilon_AS = 0 and 0.1.
    """
    power = _BLOCH_EXTREME_POWER
    ham, geminal, nuclei, charges, cell, n_max = _bloch_he_model(power)
    _bloch_check_orbital(power, geminal)
    start = _bloch_start(spin, _BLOCH_EXTREME_START, _BLOCH_NEAR)
    matrix, inverse = _bloch_state(power, *start)
    # Nested validity: a nonzero G below sqrt(tiny) with a finite inverse, G * G^-1 = 1.
    atol, rtol = get_tolerance_min(_AS_ZONES, "strict")
    assert 0.0 < abs(matrix[0, 0]) < cases.UNDERFLOW_ROW_NORM and np.all(np.isfinite(inverse)), (matrix, inverse)
    np.testing.assert_allclose(matrix * inverse, 1.0, atol=atol, rtol=rtol)
    moved = np.asarray(_BLOCH_EXTREME_START)
    normal = (_BLOCH_EXTREME_TARGET_X - moved[0]) / (cases.width(moved, nuclei, charges) * _BLOCH_EXTREME_DT)
    keys = _BLOCH_EXTREME_KEYS[spin]
    probability, ln_ratio, expected = _bloch_reference(power, start, spin, _BLOCH_EXTREME_DT, normal, nuclei, charges, keys)
    # Non-vacuity: |R| finite, |R|^2 beyond float64, exact acceptance 1.
    assert 2.0 * ln_ratio > _LN_FLOAT64_MAX > ln_ratio and probability == 1.0, (ln_ratio, probability)
    single, batched = _bloch_native_decisions(
        ham, geminal, nuclei, charges, cell, n_max, start, matrix, inverse, _BLOCH_EXTREME_DT, epsilon, normal, keys
    )
    assert single == expected and batched == expected, (spin, epsilon, probability, expected, single, batched)
