"""Same-input full-recompute references for the actual XLA Metropolis kernels."""

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

import ast
import inspect
import itertools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import jqmc_mcmc as mcmc
from jqmc._precision import get_dtype_np, get_tolerance_min
from jqmc.determinant import _init_det_ratio_streaming_state, compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data, init_Jastrow_ratio_state
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_CASES = [
    ("H2_ae_ccpvdz_cart.h5", "up", 0.4, False),
    ("H2_ae_ccpvdz_cart.h5", "down", 0.4, False),
    ("water_ccecp_ccpvqz.h5", "up", 0.4, False),
    ("water_ccecp_ccpvqz.h5", "down", 0.4, False),
    ("lithium_k025.h5", "up", 0.4, False),
    ("lithium_k025.h5", "down", 0.4, False),
    ("H_ae_ccpvdz_cart.h5", "up", -0.4, False),
    ("lithium_k025.h5", "up", 0.4, True),
    ("lithium_k025.h5", "down", 0.4, True),
]


def _input(filename, spin="up", cell_face=False):
    source = Path(__file__).resolve().parents[1] / "trexio_example_files" / filename
    structure, _, _, _, geminal, coulomb = read_trexio_file(str(source), store_tuple=True)
    jastrow = Jastrow_data(
        jastrow_two_body_data=Jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp", structure_data=structure)
    )
    ham = Hamiltonian_data(
        structure_data=structure,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=jastrow),
        coulomb_potential_data=coulomb,
    )
    rng = np.random.default_rng(2025)
    up = rng.normal(size=(geminal.num_electron_up, 3)) * 0.5
    down = rng.normal(size=(geminal.num_electron_dn, 3)) * 0.5
    if cell_face:
        cell = np.asarray(structure._cell_jnp)
        positions = up if spin == "up" else down
        fractional = positions[0] @ np.linalg.inv(cell)
        fractional[0] = -0.01
        positions[0] = fractional @ cell
    matrix = compute_geminal_all_elements(geminal, up, down)
    state = (
        jnp.asarray(up),
        jnp.asarray(down),
        jax.random.PRNGKey(2025),
        jnp.linalg.inv(matrix),
        matrix,
        _init_det_ratio_streaming_state(geminal, up, down),
        init_Jastrow_ratio_state(jastrow, up, down),
    )
    return ham, state


def _reference(ham, up, down, spin, normal, axis, epsilon):
    """Rebuild determinants/Jastrow, use SVD for AS, and enumerate nuclear images."""
    structure = ham.structure_data
    nuclei = np.asarray(structure.positions)
    charges = np.asarray(structure.atomic_numbers)
    if ham.coulomb_potential_data.ecp_flag:
        charges = charges - np.asarray(ham.coulomb_potential_data.z_cores)
    translations = np.zeros((1, 3))
    if structure.pbc_flag:
        translations = np.asarray(list(itertools.product(range(-2, 3), repeat=3))) @ np.asarray(structure._cell_jnp)

    def width(position):
        distances = np.linalg.norm(position - nuclei[:, None, :] - translations[None, :, :], axis=-1)
        atom, _ = np.unravel_index(np.argmin(distances), distances.shape)
        radius = float(distances.min())
        charge = float(charges[atom])
        return (radius + charge ** (-2)) / (radius + 1)

    before = (up if spin == "up" else down)[0]
    after = before.copy()
    after[axis] += normal * width(before) * 1.2
    proposed_up, proposed_down = up.copy(), down.copy()
    (proposed_up if spin == "up" else proposed_down)[0] = after
    old_sigma, new_sigma = width(before) * 1.2, width(after) * 1.2
    log_transition = np.log(old_sigma / new_sigma) - np.sum((after - before) ** 2) / 2 * (new_sigma ** (-2) - old_sigma ** (-2))
    geminal = ham.wavefunction_data.geminal_data
    jastrow = ham.wavefunction_data.jastrow_data

    def full_jastrow(u, d):
        # Compare within the active precision configuration, in the consumer's
        # jastrow_ratio zone. Subtracting two fp32 jastrow_eval totals is not
        # an oracle for a mixed-mode fp64 ratio operation.
        dtype = get_dtype_np("jastrow_ratio")
        positions = np.asarray(np.concatenate([u, d]), dtype=dtype)
        i, j = np.triu_indices(len(positions), k=1)
        displacements = positions[i] - positions[j]
        if structure.pbc_flag:
            cell = np.asarray(structure._cell_jnp, dtype=dtype)
            fractional = displacements @ np.linalg.inv(cell)
            displacements = np.sin(np.pi * fractional) * (np.linalg.norm(cell, axis=1) / np.pi)
        distances = np.linalg.norm(displacements, axis=1)
        parameter = np.asarray(jastrow.jastrow_two_body_data.jastrow_2b_param, dtype=dtype)
        return np.sum(-np.expm1(-parameter * distances) / (2 * parameter))

    log_probabilities = []
    for u, d in ((up, down), (proposed_up, proposed_down)):
        matrix = np.asarray(compute_geminal_all_elements(geminal, u, d))
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        smallest_norm = min(np.sum(np.abs(matrix) ** 2, axis=0).min(), np.sum(np.abs(matrix) ** 2, axis=1).min())
        regularizer = (smallest_norm * np.sum(singular_values ** (-2))) ** (-3 / 8)
        log_probabilities.append(
            2 * (np.linalg.slogdet(matrix)[1] + full_jastrow(u, d) + np.log(max(regularizer, epsilon) / regularizer))
        )
    assert np.all(np.isfinite(log_probabilities)), "reference precondition: finite full-recompute log weights"
    probability = np.exp(min(0.0, log_probabilities[1] - log_probabilities[0] + log_transition))
    return probability, proposed_up, proposed_down


def _move(ham, state, spin, normal, axis, epsilon, draw, *, observe, fault=None):
    """Supply fixed valid draws without changing the actual loop or its carried state."""
    up, down, key, inverse, matrix, det_state, jastrow_state = state
    function = mcmc._update_electron_positions if len(down) else mcmc._update_electron_positions_only_up_electron
    indices = iter([0 if spin == "up" else len(up), 0, 0, axis] if len(down) else [0, axis])
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

    if observe:
        numpy_proxy.min = observe_min
    with patch.object(mcmc, "jax", proxy), patch.object(mcmc, "jnp", numpy_proxy):
        execute = function.__wrapped__
        if fault is not None:
            replacements = {
                "transition": ("R_ratio * T_ratio", "R_ratio"),
                "jastrow": ("WF_ratio = J_ratio * Det_T_ratio", "WF_ratio = Det_T_ratio"),
                "regularizer": ("_ratio = R_AS_ratio * WF_ratio", "_ratio = WF_ratio"),
                "conjugation": ("_ratio * jnp.conj(_ratio)", "_ratio * _ratio"),
            }
            source = inspect.getsource(execute)
            old, new = replacements[fault]
            assert source.count(old) == 1
            tree = ast.parse(source.replace(old, new))
            tree.body[0].decorator_list = []
            namespace = dict(vars(mcmc))
            exec(compile(tree, "<fault-injected-MCMC-acceptance>", "exec"), namespace)
            execute = namespace[function.__name__]
        result = execute(up, down, key, 1, ham, 1.2, epsilon, inverse, matrix, det_state, jastrow_state)
        jax.block_until_ready(result)
    return result, observations


def _state_vector(result):
    return np.concatenate([np.asarray(value).ravel() for value in jax.tree.leaves(result)])


def _assert_probability(actual_probability, expected_probability, label):
    atol, rtol = get_tolerance_min(("det_eval", "wf_eval"), "strict")
    np.testing.assert_allclose(actual_probability, expected_probability, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("filename,spin,normal,cell_face", _CASES)
@pytest.mark.parametrize("epsilon", [0.0, 1.0])
@pytest.mark.parametrize("draw", [0.2, 0.8])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_xla_acceptance_and_carried_state_match_full_recompute(filename, spin, normal, cell_face, epsilon, draw):
    """Four actual XLA moves match independent same-input probabilities and endpoint matrices.

    Committed H2 AE, water ECP and non-Gamma Li ECP each cover up/down moves;
    H AE covers the separate empty-down kernel. J2 exp has b=0.5. PCG64 seed
    2025 creates one initial normal*0.5 configuration; there is no cond(G)
    filtering, reselection or trajectory pairing. Carry all production caches
    through four fixed proposals (axes 0,1,2,0; signs +,-,+,+) and alternating
    uniform draws 0.2/0.8. H uses an outward first move. AS epsilon is 0 or 1.
    Two additional Li inputs fix the moved electron's first fractional
    coordinate to -0.01 and require the first accepted move to cross that face.
    Rebuild full G and independently sum every J2 exp electron pair in the
    active jastrow_ratio zone (fp64 in both configurations, not fp32 eval).
    Use SVD for AS and an explicit
    125-image nuclear-distance reference in the committed periodic cell.
    No fast ratio, inverse-update, AS or proposal helper supplies the reference.
    Acceptance probabilities, positions and rebuilt G use same-mode central
    det_eval/wf_eval strict tolerances. Separately compare the carried inverse
    with a dense inverse of its identical saved G, only after checking that G.
    Decisions and the entire observer-on/off state are exact. Both precision
    modes are required. This tests fixed-input Metropolis composition, not
    random-number distributions or statistical ground-state convergence.
    """
    ham, state = _input(filename, spin, cell_face)
    atol, rtol = get_tolerance_min(("det_eval", "wf_eval"), "strict")
    for label, axis, sign, uniform in (
        ("move-0", 0, 1, draw),
        ("move-1", 1, -1, 1 - draw),
        ("move-2", 2, 1, draw),
        ("move-3", 0, 1, 1 - draw),
    ):
        up, down = np.asarray(state[0]), np.asarray(state[1])
        expected_probability, proposed_up, proposed_down = _reference(ham, up, down, spin, sign * normal, axis, epsilon)
        observed, probabilities = _move(ham, state, spin, sign * normal, axis, epsilon, uniform, observe=True)
        ordinary, _ = _move(ham, state, spin, sign * normal, axis, epsilon, uniform, observe=False)
        observed_state, ordinary_state = _state_vector(observed), _state_vector(ordinary)
        np.testing.assert_array_equal(observed_state, ordinary_state)
        assert len(probabilities) == 1, label
        actual_probability = probabilities[0]
        np.testing.assert_allclose(actual_probability, expected_probability, atol=atol, rtol=rtol, equal_nan=False)
        expected_decision = np.asarray([int(uniform < expected_probability), int(uniform >= expected_probability)])
        actual_decision = np.asarray(ordinary[:2])
        np.testing.assert_array_equal(actual_decision, expected_decision)
        expected_positions = np.concatenate([proposed_up, proposed_down] if expected_decision[0] else [up, down])
        actual_positions = np.concatenate([np.asarray(ordinary[2]), np.asarray(ordinary[3])])
        np.testing.assert_allclose(actual_positions, expected_positions, atol=atol, rtol=rtol, equal_nan=False)
        if cell_face and label == "move-0":
            inverse_cell = np.linalg.inv(np.asarray(ham.structure_data._cell_jnp))
            before = (up if spin == "up" else down)[0]
            after = np.asarray(ordinary[2] if spin == "up" else ordinary[3])[0]
            assert np.floor((before @ inverse_cell)[0]) != np.floor((after @ inverse_cell)[0])
        actual_matrix = np.asarray(ordinary[6])
        expected_matrix = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, ordinary[2], ordinary[3]))
        np.testing.assert_allclose(actual_matrix, expected_matrix, atol=atol, rtol=rtol, equal_nan=False)
        actual_inverse, expected_inverse = np.asarray(ordinary[5]), np.linalg.inv(actual_matrix)
        np.testing.assert_allclose(actual_inverse, expected_inverse, atol=atol, rtol=rtol, equal_nan=False)
        state = (ordinary[2], ordinary[3], ordinary[4], ordinary[5], ordinary[6], ordinary[7], ordinary[8])


@pytest.mark.parametrize(
    "filename,epsilon,fault",
    [
        ("water_ccecp_ccpvqz.h5", 0.0, "transition"),
        ("water_ccecp_ccpvqz.h5", 0.0, "jastrow"),
        ("water_ccecp_ccpvqz.h5", 1.0, "regularizer"),
        ("lithium_k025.h5", 0.0, "conjugation"),
    ],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_acceptance_reference_rejects_actual_formula_faults(filename, epsilon, fault):
    """Dropping T, Jastrow, AS or complex conjugation fails the independent probability oracle."""
    ham, state = _input(filename)
    expected, _, _ = _reference(ham, np.asarray(state[0]), np.asarray(state[1]), "up", 0.4, 0, epsilon)
    _, baseline = _move(ham, state, "up", 0.4, 0, epsilon, 0.2, observe=True)
    assert len(baseline) == 1
    _assert_probability(baseline[0], expected, "baseline")
    _, changed = _move(ham, state, "up", 0.4, 0, epsilon, 0.2, observe=True, fault=fault)
    assert len(changed) == 1
    with pytest.raises(AssertionError):
        _assert_probability(changed[0], expected, "fault")
