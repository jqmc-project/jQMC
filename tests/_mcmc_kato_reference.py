"""Independent periodic Kato widths and from-scratch Metropolis replay."""

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

import itertools

import jax.numpy as jnp
from jax import random
import numpy as np

from jqmc._setting import ATOL_PBC_TRANSLATION_INVARIANT, RTOL_BIT_EXACT
from jqmc.determinant import _init_det_ratio_streaming_state, compute_geminal_all_elements
from jqmc.jastrow_factor import compute_Jastrow_part, init_Jastrow_ratio_state
from jqmc.jqmc_mcmc import _draw_injected_normals
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian

PATTERNS = ("WF-PATTERN-003", "WF-PATTERN-015", "WF-PATTERN-016")
NUM_MOVES = 50
DT = 1.0


def nearest_charge_distance(positions, charges, r, cell=None):
    """Search explicit lattice images after a fractional round, independently of jqmc MIC."""
    displacement = np.asarray(r, dtype=np.float64)[None] - np.asarray(positions, dtype=np.float64)
    if cell is None:
        squared = np.sum(displacement**2, axis=1)
    else:
        cell = np.asarray(cell, dtype=np.float64)
        fractional = displacement @ np.linalg.inv(cell)
        centered = (fractional - np.round(fractional)) @ cell
        images = np.asarray(list(itertools.product(range(-3, 4), repeat=3)), dtype=np.float64) @ cell
        squared = np.min(np.sum((centered[:, None] + images[None]) ** 2, axis=-1), axis=1)
    i = int(np.argmin(squared))
    return float(charges[i]), float(np.sqrt(squared[i]))


def kato_case(pattern, lattice_shift):
    """Unmodified canonical bare-Z>1 cells, including polarized and complex AGP cases."""
    ham = build_canonical_hamiltonian(pattern)
    wf, sd = ham.wavefunction_data, ham.structure_data
    charges = np.asarray(sd.atomic_numbers, dtype=np.float64)
    if ham.coulomb_potential_data.ecp_flag:
        charges -= np.asarray(ham.coulomb_potential_data.z_cores)
    assert np.all(charges > 1.0), "Z_eff=1 makes the Kato width constant"
    seed = 2 if lattice_shift else 0
    rng = np.random.default_rng(seed + 1234)
    nu, nd = int(wf.geminal_data.num_electron_up), int(wf.geminal_data.num_electron_dn)
    positions = np.asarray(sd.positions, dtype=np.float64)
    up = positions[np.arange(nu) % len(positions)] + rng.normal(scale=0.6, size=(nu, 3))
    dn = positions[np.arange(nd) % len(positions)] + rng.normal(scale=0.6, size=(nd, 3))
    cell = np.asarray(sd.cell, dtype=np.float64)
    if lattice_shift:
        up += rng.integers(-lattice_shift, lattice_shift + 1, size=(nu, 3)) @ cell
        dn += rng.integers(-lattice_shift, lattice_shift + 1, size=(nd, 3)) @ cell
        z, periodic_distance = nearest_charge_distance(positions, charges, up[0], cell)
        _, raw_distance = nearest_charge_distance(positions, charges, up[0])
        assert periodic_distance < raw_distance and z > 1.0
    key = random.PRNGKey(seed)
    normals = np.asarray(_draw_injected_normals(key, NUM_MOVES), dtype=np.float64)
    return ham, charges, np.ascontiguousarray(up), np.ascontiguousarray(dn), key, normals


def replay(ham, charges, up, dn, key, normals):
    """Full determinants/Jastrows and the analytic asymmetric Gaussian transition density."""
    gem, jas = ham.wavefunction_data.geminal_data, ham.wavefunction_data.jastrow_data
    positions = np.asarray(ham.structure_data.positions, dtype=np.float64)
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    up, dn = up.copy(), dn.copy()

    def det(a, b):
        return complex(np.linalg.det(np.asarray(compute_geminal_all_elements(gem, a, b))))

    def log_jastrow(a, b):
        return float(np.asarray(compute_Jastrow_part(jas, a, b)))

    old_det, old_j = det(up, dn), log_jastrow(up, dn)
    assert np.isfinite(old_det) and old_det != 0.0
    accepted = 0
    for normal in normals:
        key, subkey = random.split(key)
        is_up = int(random.randint(subkey, (), 0, len(up) + len(dn))) < len(up)
        key, subkey = random.split(key)
        ui = int(random.randint(subkey, (), 0, len(up)))
        key, subkey = random.split(key)
        di = int(random.randint(subkey, (), 0, len(dn)))
        i = ui if is_up else di
        old = (up if is_up else dn)[i]
        z, distance = nearest_charge_distance(positions, charges, old, cell)
        width = (1 + z**2 * distance) / (z**2 * (1 + distance))
        key, _ = random.split(key)
        key, subkey = random.split(key)
        axis = int(random.randint(subkey, (), 0, 3))
        proposed = old.copy()
        proposed[axis] += float(normal) * width * DT
        z_new, distance_new = nearest_charge_distance(positions, charges, proposed, cell)
        reverse_width = (1 + z_new**2 * distance_new) / (z_new**2 * (1 + distance_new))
        displacement_squared = np.sum((proposed - old) ** 2)
        transition = (
            width
            / reverse_width
            * np.exp(-displacement_squared * (1 / (2 * reverse_width**2 * DT**2) - 1 / (2 * width**2 * DT**2)))
        )
        new_up, new_dn = up.copy(), dn.copy()
        (new_up if is_up else new_dn)[i] = proposed
        new_det, new_j = det(new_up, new_dn), log_jastrow(new_up, new_dn)
        probability = min(1.0, abs(np.exp(new_j - old_j) * new_det / old_det) ** 2 * transition)
        key, subkey = random.split(key)
        draw = float(random.uniform(subkey, (), dtype=jnp.float64))
        if draw < probability:
            up, dn, old_det, old_j = new_up, new_dn, new_det, new_j
            accepted += 1
    assert 0 < accepted < len(normals), "both acceptance outcomes must be live"
    return {"accepted": accepted, "r_up": up, "r_dn": dn, "key": np.asarray(key)}


def xla_sweep(ham, up, dn, key):
    from jqmc.jqmc_mcmc import _update_electron_positions, _update_electron_positions_only_up_electron

    gem, jas = ham.wavefunction_data.geminal_data, ham.wavefunction_data.jastrow_data
    matrix = np.asarray(compute_geminal_all_elements(gem, up, dn))
    state = _init_det_ratio_streaming_state(gem, jnp.asarray(up), jnp.asarray(dn))
    jstate = init_Jastrow_ratio_state(jas, jnp.asarray(up), jnp.asarray(dn))
    kernel = _update_electron_positions if len(dn) else _update_electron_positions_only_up_electron
    result = kernel(up, dn, key, NUM_MOVES, ham, DT, 0.0, np.linalg.inv(matrix), matrix, state, jstate)
    return {"accepted": int(result[0]), "r_up": np.asarray(result[2]), "r_dn": np.asarray(result[3]), "key": result[4]}


def assert_kato_replay(actual, expected):
    """Exact decisions/key; f64 coordinate budget for independent image arithmetic."""
    assert int(actual["accepted"]) == expected["accepted"]
    np.testing.assert_array_equal(np.asarray(actual["key"]), expected["key"])
    for spin in ("r_up", "r_dn"):
        np.testing.assert_allclose(
            np.asarray(actual[spin]), expected[spin], atol=ATOL_PBC_TRANSLATION_INVARIANT, rtol=RTOL_BIT_EXACT
        )
