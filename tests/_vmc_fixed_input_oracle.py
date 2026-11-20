"""Deterministic, same-input references for the legacy molecular VMC composition.

Each sweep is anchored to its observed input, not to a separately evolving
trajectory. Input threading, random draws and measurement placement are
checked separately. No distributional or ground-state convergence is tested.
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

import numpy as np
from jax import random

from jqmc.determinant import compute_geminal_all_elements
from jqmc.hamiltonians import compute_local_energy
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from tests._fixed_sample_statistics import equal_bin_statistics
from tests.test_rs_kernel.test_mcmc_sweep import _one_move


def state_vector(state):
    """Flatten the supplied state quantities without dropping a spin channel."""
    return np.concatenate([np.asarray(value).ravel() for value in state])


def recomputed_matrix_state(ham, up, down, observed_matrix):
    """Independently rebuild G and invert the exact matrix the native inverse belongs to.

    These are two separate same-input checks: coordinates -> G and G -> inverse.
    The independent G comparison must also pass; using the observed matrix for
    the second check must never make a consistently wrong G its own reference.
    """
    matrix = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, up, down), dtype=np.float64)
    return matrix, np.linalg.inv(np.asarray(observed_matrix, dtype=np.float64))


def vmc_reference(ham, *, seed, dt, eps, warmup, measurements, sweeps, evaluations):
    """Verify input wiring and independently evaluate all recorded native calls."""
    gem, jas = ham.wavefunction_data.geminal_data, ham.wavefunction_data.jastrow_data
    nuclei = np.asarray(ham.structure_data.positions, dtype=np.float64)
    charges = np.asarray(ham.structure_data.atomic_numbers, dtype=np.float64) - np.asarray(
        ham.coulomb_potential_data.z_cores, dtype=np.float64
    )
    n_up, n_down = gem.num_electron_up, gem.num_electron_dn
    moves = n_up + n_down
    rng = np.random.default_rng(seed)
    up = nuclei[rng.integers(0, len(nuclei), n_up)] + rng.normal(scale=0.5, size=(n_up, 3))
    down = nuclei[rng.integers(0, len(nuclei), n_down)] + rng.normal(scale=0.5, size=(n_down, 3))
    matrix = np.asarray(compute_geminal_all_elements(gem, up, down), dtype=np.float64)
    expected_state = (up, down, matrix, np.linalg.inv(matrix))
    expected_key = np.asarray(random.PRNGKey(seed), dtype=np.uint32)
    wiring = [len(sweeps) == warmup + measurements, len(evaluations) == measurements]
    states, inverses, decisions = [], [], []
    for inputs, output in sweeps:
        wiring.extend([inputs[0] is gem, inputs[1] is jas, len(inputs) == 13])
        wiring.extend([np.array_equal(inputs[2], nuclei), np.array_equal(inputs[3], charges)])
        before = inputs[4:8]
        wiring.extend(np.array_equal(actual, expected) for actual, expected in zip(before, expected_state))
        supplied_key = np.asarray(inputs[10:12], dtype=np.uint32)
        wiring.extend(
            [
                float(inputs[8]) == dt,
                float(inputs[9]) == eps,
                np.array_equal(supplied_key, expected_key),
                np.array_equal(inputs[12], rng.normal(size=moves)),
            ]
        )
        up, down, matrix, inverse = [np.array(value, copy=True) for value in before]
        key = supplied_key.copy()
        accepted = 0
        for normal in inputs[12]:
            up, down, matrix, inverse, key, decision = _one_move(ham, up, down, matrix, inverse, dt, eps, key, float(normal))
            accepted += int(decision)
        # Coordinate decisions are referenced from the pre-sweep input.
        # G is rebuilt at the actual endpoint, independently of the saved G.
        # The inverse is fully recomputed for the SAME saved G it belongs to:
        # mixed AO roundoff need not have the same forward inverse error.
        rebuilt, rebuilt_inverse = recomputed_matrix_state(ham, output["r_up"], output["r_dn"], output["geminal"])
        states.append(state_vector((up, down, rebuilt)))
        inverses.append(rebuilt_inverse)
        decisions.append((accepted, int(key[0]), int(key[1])))
        # Actual handoff is a separate exact requirement. These observations
        # never supply a reference G/inverse answer or the coordinate decision.
        expected_state = tuple(output[name] for name in ("r_up", "r_dn", "geminal", "geminal_inv"))
        expected_key = output["final_key"]

    weights = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
    grid = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)
    energies = []
    for index, (block, actual_ham, up_eval, down_eval, actual_weights, actual_grid) in enumerate(evaluations):
        expected = sweeps[warmup + index][1]
        wiring.extend(
            [
                block == warmup + index,
                actual_ham is ham,
                np.array_equal(up_eval, expected["r_up"]),
                np.array_equal(down_eval, expected["r_dn"]),
                np.array_equal(actual_weights, weights),
                np.array_equal(actual_grid, grid),
            ]
        )
        energies.append(float(compute_local_energy(ham, up_eval, down_eval, np.eye(3))))
    energies = np.asarray(energies)
    mean, standard_error = equal_bin_statistics(energies[:, None], len(energies))
    return {
        "wiring": np.asarray(wiring, dtype=bool),
        "states": np.asarray(states),
        "inverses": np.asarray(inverses),
        "decisions": np.asarray(decisions, dtype=np.uint64),
        "energies": energies,
        "mean": mean,
        "stderr": standard_error,
        "accept_rate": sum(row[0] for row in decisions) / ((warmup + measurements) * moves),
        "final_positions": np.concatenate([up.ravel(), down.ravel()]),
    }
