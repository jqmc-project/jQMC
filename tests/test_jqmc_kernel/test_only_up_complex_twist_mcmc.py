"""XLA only-up MCMC kernel (``n_dn = 0``) at a complex Bloch twist against a complex determinant reference.

Issue #72. ``MCMC.run`` sends every ``n_dn = 0`` system on the XLA route to
``jqmc_mcmc._update_electron_positions_only_up_electron`` (through
``_jit_vmap_update_up``). For a complex (twisted) wavefunction that kernel
must keep the carried Sherman-Morrison inverse complex and accept with the
squared modulus of the complex determinant ratio. Casting the updated inverse
to a real dtype makes the ``lax.cond`` branches disagree (the kernel fails to
trace), and casting the determinant ratio to a real dtype makes the acceptance
use ``Re(ratio)^2`` instead of ``|ratio|^2``.

Fixture: the committed twist ``H4_pbc_ae_k025_cart.h5`` (k = 0.25, complex
Cartesian pGTO, two complex MOs), no Jastrow, with its geminal replaced by an
``n_up``-up / 0-down Slater determinant. For ``n_up > 2`` the extra up orbitals
are fixed seeded random complex combinations of the same AOs (PCG64 seed 11).

* ``test_only_up_xla_kernel_complex_twist_acceptance_matches_complex_determinant_reference``
  runs six actual kernel moves with fixed injected draws (moved electron, axis,
  normal, uniform) and carries every production cache between them. For each
  move the acceptance probability the kernel computes, its accept decision,
  the walker update, the carried geminal and the carried inverse are compared
  with an independent reference: ``numpy.linalg.det`` of a fresh
  ``compute_geminal_all_elements`` matrix at the old and proposed
  configuration (complex), an SVD form of the AS regularization, and a
  125-image nuclear-distance enumeration for the proposal widths. No fast
  ratio, inverse update, AS or proposal helper supplies the reference.
  Non-vacuity: the carried inverse is complex with a non-zero imaginary part,
  and on at least one move ``min(1, Re(ratio)^2 T)`` differs from
  ``min(1, |ratio|^2 T)`` by more than the comparison tolerance, so a real
  cast of the ratio would fail.
* ``test_only_up_xla_mcmc_complex_twist_chain_keeps_carried_state_consistent``
  runs the production ``_jit_vmap_update_up`` wrapper used by ``MCMC.run`` for
  four walkers and 40 random-number-driven steps and compares the carried
  complex geminal and inverse with a fresh rebuild at the final positions.

Both precision modes are required.
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

import itertools
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import jqmc_mcmc as mcmc
from jqmc._precision import get_tolerance_min
from jqmc.determinant import SD_data, _init_det_ratio_streaming_state, compute_geminal_all_elements
from jqmc.jastrow_factor import init_Jastrow_ratio_state
from jqmc.molecular_orbital import MOs_data
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_FIXTURE = "H4_pbc_ae_k025_cart.h5"
_DT = 1.2
_ZONES = ("det_eval", "wf_eval")
_CHAIN_ZONES = ("det_ratio", "det_eval")


def _build_ham(n_up):
    """Twist H4 (k = 0.25) with an ``n_up``-up / 0-down complex Slater determinant and no Jastrow."""
    ham = _build_ham_for_cell(_FIXTURE, False, None, "SD")
    gd0 = ham.wavefunction_data.geminal_data
    aos = gd0.orb_data_up_spin.aos_data
    c0 = np.asarray(gd0.orb_data_up_spin.mo_coefficients)
    assert np.iscomplexobj(c0) and np.abs(c0.imag).max() > 0, "fixture must be a complex twist"
    rng = np.random.default_rng(11)
    extra = rng.normal(size=(4, aos.num_ao)) + 1j * rng.normal(size=(4, aos.num_ao))
    coefficients = np.vstack([c0, 0.3 * extra])
    mos = MOs_data.build(aos_data=aos, mo_coefficients=jnp.asarray(coefficients[:n_up]))
    geminal = SD_data.build(num_electron_up=n_up, num_electron_dn=0, mos_up=mos, mos_dn=mos, time_reversal=False)
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=geminal))
    assert (ham.wavefunction_data.geminal_data.num_electron_up, ham.wavefunction_data.geminal_data.num_electron_dn) == (
        n_up,
        0,
    )
    return ham


def _initial_positions(ham, n_up, seed):
    cell = np.asarray(ham.structure_data._cell_jnp)
    rng = np.random.default_rng(seed)
    return rng.random((n_up, 3)) @ cell, np.zeros((0, 3))


def _kernel_state(ham, up, down, key):
    geminal = ham.wavefunction_data.geminal_data
    jastrow = ham.wavefunction_data.jastrow_data
    matrix = compute_geminal_all_elements(geminal, up, down)
    return (
        jnp.asarray(up),
        jnp.asarray(down),
        key,
        jnp.linalg.inv(matrix),
        matrix,
        _init_det_ratio_streaming_state(geminal, up, down),
        init_Jastrow_ratio_state(jastrow, up, down),
    )


def _reference(ham, up, index, axis, normal, epsilon):
    """Independent acceptance probability of one up move from fresh complex determinants.

    Returns ``(probability, real_cast_probability, proposed_up)``; the second
    value is what an acceptance built from ``Re(det ratio)`` would give and only
    serves the non-vacuity witness.
    """
    structure = ham.structure_data
    nuclei = np.asarray(structure.positions)
    charges = np.asarray(structure.atomic_numbers, dtype=np.float64)
    translations = np.asarray(list(itertools.product(range(-2, 3), repeat=3))) @ np.asarray(structure._cell_jnp)

    def width(position):
        distances = np.linalg.norm(position - nuclei[:, None, :] - translations[None, :, :], axis=-1)
        atom, _ = np.unravel_index(np.argmin(distances), distances.shape)
        radius = float(distances.min())
        charge = float(charges[atom])
        return (radius + charge ** (-2)) / (radius + 1)

    before = up[index]
    after = before.copy()
    after[axis] += normal * width(before) * _DT
    proposed_up = up.copy()
    proposed_up[index] = after
    old_sigma, new_sigma = width(before) * _DT, width(after) * _DT
    transition = (old_sigma / new_sigma) * np.exp(-np.sum((after - before) ** 2) / 2 * (new_sigma ** (-2) - old_sigma ** (-2)))
    geminal = ham.wavefunction_data.geminal_data
    down = np.zeros((0, 3))

    def det_and_as(u):
        matrix = np.asarray(compute_geminal_all_elements(geminal, u, down)).astype(np.complex128)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        smallest_norm = min(np.sum(np.abs(matrix) ** 2, axis=0).min(), np.sum(np.abs(matrix) ** 2, axis=1).min())
        regularizer = (smallest_norm * np.sum(singular_values ** (-2))) ** (-3 / 8)
        return complex(np.linalg.det(matrix)), max(regularizer, epsilon) / regularizer

    det_old, as_old = det_and_as(up)
    det_new, as_new = det_and_as(proposed_up)
    ratio = det_new / det_old
    weight = (as_new / as_old) ** 2 * transition
    assert np.isfinite(ratio) and np.isfinite(weight), "reference precondition: finite fresh-determinant ratio"
    return min(1.0, abs(ratio) ** 2 * weight), min(1.0, ratio.real**2 * weight), proposed_up


def _move(ham, state, index, axis, normal, draw, epsilon):
    """One actual only-up kernel move with fixed draws; observe the acceptance probability it computes."""
    up, down, key, inverse, matrix, det_state, jastrow_state = state
    indices = iter([index, axis])
    proxy = SimpleNamespace(**vars(jax))
    random_proxy = SimpleNamespace(**vars(jax.random))
    random_proxy.randint = lambda *args, **kwargs: jnp.asarray(next(indices), dtype=jnp.int64)
    random_proxy.normal = lambda *args, **kwargs: jnp.asarray(normal, dtype=jnp.float64)
    random_proxy.uniform = lambda *args, **kwargs: jnp.asarray(draw, dtype=jnp.float64)
    proxy.random = random_proxy
    numpy_proxy = SimpleNamespace(**vars(jnp))
    observations = []

    def observe_min(*args, **kwargs):
        value = jnp.min(*args, **kwargs)
        jax.debug.callback(lambda x: observations.append(float(x)), value)
        return value

    numpy_proxy.min = observe_min
    with patch.object(mcmc, "jax", proxy), patch.object(mcmc, "jnp", numpy_proxy):
        execute = mcmc._update_electron_positions_only_up_electron.__wrapped__
        result = execute(up, down, key, 1, ham, _DT, epsilon, inverse, matrix, det_state, jastrow_state)
        jax.block_until_ready(result)
    return result, observations


_SCHEDULE = (
    # (move index into the up block modulo n_up, axis, normal, uniform-flip)
    (1, 0, 0.9, False),
    (2, 1, -0.7, True),
    (0, 2, 1.1, False),
    (2, 0, -1.3, True),
    (1, 1, 0.8, False),
    (0, 2, -1.0, True),
)


@pytest.mark.light
@pytest.mark.parametrize("n_up", [1, 3])
@pytest.mark.parametrize("epsilon", [0.0, 1.0])
@pytest.mark.parametrize("draw", [0.2, 0.8])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_only_up_xla_kernel_complex_twist_acceptance_matches_complex_determinant_reference(n_up, epsilon, draw):
    """Six actual only-up XLA moves at a complex twist match a fresh complex-determinant acceptance reference."""
    ham = _build_ham(n_up)
    geminal = ham.wavefunction_data.geminal_data
    up, down = _initial_positions(ham, n_up, seed=2026)
    state = _kernel_state(ham, up, down, jax.random.PRNGKey(72))
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    real_cast_detectable = 0
    moved = set()
    for move, (slot, axis, normal, flip) in enumerate(_SCHEDULE):
        index = slot % n_up
        uniform = 1.0 - draw if flip else draw
        label = f"n_up={n_up} epsilon={epsilon} move={move}"
        up = np.asarray(state[0])
        expected_probability, real_cast_probability, proposed_up = _reference(ham, up, index, axis, normal, epsilon)
        result, probabilities = _move(ham, state, index, axis, normal, uniform, epsilon)
        assert len(probabilities) == 1, label
        np.testing.assert_allclose(probabilities[0], expected_probability, atol=atol, rtol=rtol, err_msg=label)
        if abs(real_cast_probability - expected_probability) > atol + rtol * expected_probability:
            real_cast_detectable += 1
        expected_decision = np.asarray([int(uniform < expected_probability), int(uniform >= expected_probability)])
        np.testing.assert_array_equal(np.asarray(result[:2]), expected_decision, err_msg=label)
        expected_up = proposed_up if expected_decision[0] else up
        np.testing.assert_allclose(np.asarray(result[2]), expected_up, atol=atol, rtol=rtol, err_msg=label)
        assert np.asarray(result[3]).shape == (0, 3), label
        moved.add(index)
        carried_matrix, carried_inverse = np.asarray(result[6]), np.asarray(result[5])
        fresh_matrix = np.asarray(compute_geminal_all_elements(geminal, result[2], result[3]))
        np.testing.assert_allclose(carried_matrix, fresh_matrix, atol=atol, rtol=rtol, err_msg=label)
        assert np.iscomplexobj(carried_inverse) and np.abs(carried_inverse.imag).max() > 0, label
        np.testing.assert_allclose(carried_inverse, np.linalg.inv(carried_matrix), atol=atol, rtol=rtol, err_msg=label)
        state = tuple(result[2:9])
    assert moved == set(range(n_up))
    assert real_cast_detectable >= 1, "no move separates |ratio|^2 from Re(ratio)^2 acceptance"


@pytest.mark.light
@pytest.mark.parametrize("n_up", [1, 3])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_only_up_xla_mcmc_complex_twist_chain_keeps_carried_state_consistent(n_up):
    """``_jit_vmap_update_up`` (MCMC.run's XLA n_dn = 0 route) keeps the complex G and G^-1 equal to a rebuild."""
    ham = _build_ham(n_up)
    geminal = ham.wavefunction_data.geminal_data
    num_walkers, num_steps = 4, 40
    states = [
        _kernel_state(ham, *_initial_positions(ham, n_up, seed=100 + w), jax.random.PRNGKey(w)) for w in range(num_walkers)
    ]
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *states)
    up, down, keys, inverse, matrix, det_state, jastrow_state = stacked
    result = mcmc._jit_vmap_update_up(up, down, keys, num_steps, ham, _DT, 0.0, inverse, matrix, det_state, jastrow_state)
    jax.block_until_ready(result)
    accepted, rejected = np.asarray(result[0]), np.asarray(result[1])
    np.testing.assert_array_equal(accepted + rejected, np.full(num_walkers, num_steps))
    assert accepted.sum() > 0 and rejected.sum() > 0, (accepted, rejected)
    atol, rtol = get_tolerance_min(_CHAIN_ZONES, "medium")
    for w in range(num_walkers):
        final_up, final_down = np.asarray(result[2][w]), np.asarray(result[3][w])
        assert final_down.shape == (0, 3)
        carried_matrix, carried_inverse = np.asarray(result[6][w]), np.asarray(result[5][w])
        fresh_matrix = np.asarray(compute_geminal_all_elements(geminal, final_up, final_down))
        assert np.iscomplexobj(carried_inverse) and np.abs(carried_inverse.imag).max() > 0, w
        np.testing.assert_allclose(carried_matrix, fresh_matrix, atol=atol, rtol=rtol, err_msg=f"walker {w}: G")
        np.testing.assert_allclose(
            carried_inverse, np.linalg.inv(fresh_matrix), atol=atol, rtol=rtol, err_msg=f"walker {w}: G^-1"
        )
