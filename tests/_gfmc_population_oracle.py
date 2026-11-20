"""Fixed-input reference for legacy AE and ECP GFMC runners.

This is independent of their propagation/aggregation assembly, not of the
underlying XLA Hamiltonian primitives or the declared legacy random stream.
It does not estimate ground-state energies or test statistical convergence.
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

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np

from jqmc.coulomb_potential import (
    compute_coulomb_potential_ee_dispatch,
    compute_coulomb_potential_ei_discretized_dispatch,
    compute_coulomb_potential_ei_dispatch,
    compute_coulomb_potential_ii_dispatch,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors,
)
from jqmc.wavefunction import compute_discretized_kinetic_energy, compute_kinetic_energy_all_elements


@partial(jax.jit, static_argnames=("ecp_neighbors",))
def _leaf_step(ham, up, down, rotation=None, *, ecp_neighbors=1):
    """Assemble leaf rates/energies on the specified mesh, without a GFMC closure.

    The ECP zero-variance lower bound uses bare ei, not AE's discretized ei.
    Add the local and nonlocal energies and concatenate the kinetic/ECP moves.
    Only real parts enter this declared fixed-phase propagation reference.
    """
    wf, cd, alat = ham.wavefunction_data, ham.coulomb_potential_data, 0.3
    rotation = jnp.eye(3) if rotation is None else rotation
    mesh_up, mesh_dn, elements = compute_discretized_kinetic_energy(alat, wf, up, down, rotation)
    elements = jnp.real(elements).reshape(-1, 6)
    kinetic = jnp.concatenate(compute_kinetic_energy_all_elements(wf, up, down)).real
    bare = jnp.concatenate(compute_coulomb_potential_ei_dispatch(cd, up, down)).real
    clipped = (
        bare if cd.ecp_flag else jnp.concatenate(compute_coulomb_potential_ei_discretized_dispatch(cd, up, down, alat)).real
    )
    pair = compute_coulomb_potential_ee_dispatch(cd, up, down).real + compute_coulomb_potential_ii_dispatch(cd).real
    difference = kinetic + bare - jnp.sum(elements + 1 / (2 * alat**2), axis=1)
    folded = jnp.where(jnp.any(elements >= 0, axis=1), jnp.maximum(difference, clipped), difference)
    propagation = 3 * (up.shape[0] + down.shape[0]) / alat**2 + pair + jnp.sum(folded) + jnp.sum(elements)
    measurement = pair + jnp.sum(kinetic + bare)
    off_diagonal = jnp.minimum(elements, 0).ravel()
    if cd.ecp_flag:
        local = compute_ecp_local_parts_all_pairs(cd, up, down).real
        extra_up, extra_dn, nonlocal_values, _ = compute_ecp_non_local_parts_nearest_neighbors(
            cd, wf, up, down, rotation, NN=ecp_neighbors
        )
        nonlocal_values = jnp.real(nonlocal_values)
        potential = local + jnp.sum(nonlocal_values)
        propagation, measurement = propagation + potential, measurement + potential
        mesh_up, mesh_dn = jnp.concatenate([mesh_up, extra_up]), jnp.concatenate([mesh_dn, extra_dn])
        off_diagonal = jnp.concatenate([off_diagonal, jnp.minimum(nonlocal_values, 0).ravel()])
    return mesh_up, mesh_dn, off_diagonal, propagation, measurement


class _Tape:
    """Specified SplitMix64/Box-Muller inputs, not an RNG-quality oracle."""

    def __init__(self, seed):
        self.state = (seed * 0x2545F4914F6CDD1D + 1) & ((1 << 64) - 1)
        self.draws = 0

    def uniform(self):
        mask = (1 << 64) - 1
        self.state = (self.state + 0x9E3779B97F4A7C15) & mask
        value = ((self.state ^ (self.state >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        self.draws += 1
        return (value >> 11) / (1 << 53)

    def normal(self):
        first, second = self.uniform(), self.uniform()
        if not 0 < first < 1:
            raise ValueError("the selected reference stream must avoid the Box-Muller boundary")
        return math.sqrt(-2 * math.log(first)) * math.cos(math.tau * second)

    def rotation(self):
        """Compose three axis rotations from the specified legacy uniform draws."""
        alpha = math.tau * self.uniform()
        beta = math.asin(2 * self.uniform() - 1)
        gamma = math.tau * self.uniform()
        ca, sa = math.cos(alpha), math.sin(alpha)
        cb, sb = math.cos(beta), math.sin(beta)
        cg, sg = math.cos(gamma), math.sin(gamma)
        rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
        return (rz @ ry @ rx).T


def population_reference(
    ham, up, down, population, *, seed=9, warmup=50, measurements=4000, random_mesh=False, ecp_neighbors=1
):
    """Replay every prescribed interval; defaults preserve the population fixture.

    Integrate the propagation energy in log-weight space; invert cumulative
    rate/weight masses for hops and comb reconfiguration. Measurement energy
    comes from continuum primitives. No native output is a reference input.
    The legacy tau=.1, alat=.3, one/four-walker scope is supported here.
    An ECP input adds its declared nearest-neighbor mesh, with NN explicit.
    For rotated meshes hold the first rotation throughout an interval while
    consuming each substep's historical rotation draws. The matrix is an
    independent axis-matrix product, not the production closed form.
    A replay is a deterministic algorithm check, not a
    statistical comparison between independent one/four-walker estimates.
    """
    tape = _Tape(seed)
    count = 4 if population else 1
    walkers = []
    for index in range(count):
        spins = [up.copy(), down.copy()]
        if index:
            for points in spins:
                points += 0.3 * np.array([tape.normal() for _ in range(points.size)]).reshape(points.shape)
        walkers.append(spins)
    energies, weights, hops, duplicates, uneven = [], [], 0, 0, 0
    rotations = 0
    for tick in range(warmup + measurements):
        current_e, current_w = [], []
        for spins in walkers:
            remaining, exponent = 0.1, 0.0
            rotation = None
            while remaining > 0:
                if random_mesh:
                    sampled = tape.rotation()
                    rotations += 1
                    if rotation is None:
                        rotation = sampled
                mesh_up, mesh_dn, fn, eprop, emeas = (
                    np.asarray(value) for value in _leaf_step(ham, *spins, rotation, ecp_neighbors=ecp_neighbors)
                )
                rate = -float(fn.sum())
                if not math.isfinite(rate) or rate <= 0:
                    raise ValueError("the selected reference path must have finite positive hopping mass")
                waiting = -math.log1p(-tape.uniform()) / rate
                delta = min(remaining, waiting)
                exponent -= delta * eprop
                remaining -= delta
                if remaining > 0:
                    cumulative = np.cumsum(-fn)
                    selected = int(np.searchsorted(cumulative, tape.uniform() * cumulative[-1]))
                    spins[:] = [mesh_up[selected].copy(), mesh_dn[selected].copy()]
                    hops += 1
            current_e.append(float(emeas))
            current_w.append(math.exp(exponent))
        current_e, current_w = np.array(current_e), np.array(current_w)
        if tick >= warmup:
            energies.append(float(np.dot(current_w, current_e) / np.sum(current_w)))
            weights.append(float(np.mean(current_w)))
        if population:
            cumulative = np.cumsum(current_w)
            selected = np.searchsorted(cumulative, (np.arange(count) + tape.uniform()) * cumulative[-1] / count)
            duplicates += int(len(set(selected)) < count)
            uneven += int(np.max(current_w) > np.min(current_w))
            walkers = [[points.copy() for points in walkers[index]] for index in selected]
    return (
        np.array(energies),
        np.array(weights),
        dict(hops=hops, duplicates=duplicates, uneven=uneven, draws=tape.draws, rotations=rotations),
    )


def _history_estimate(energies, weights, collect):
    """Reduce the specified delayed-weight history without a production helper."""
    products = np.prod(np.lib.stride_tricks.sliding_window_view(np.asarray(weights), collect)[:-1], axis=1)
    return math.fsum(float(w * e) for w, e in zip(products, energies[collect:])) / math.fsum(products)


def fixed_step_reference(
    ham, up, down, *, seed=4, warmup=80, measurements=6000, hops_per_interval=16, collect=5, shift=-1.1, ecp_neighbors=1
):
    """Replay the fixed-step projector and its deterministic running shift.

    The rate divided by the shifted diagonal multiplies the branching weight.
    Accumulate it in log space. Measure at the final hopped configuration and
    update the shift every tenth interval from all stored finite inputs. This
    is a finite-history algorithm check, not equality of independent GFMC_n
    and GFMC_t ground-state energy estimates. Defaults are the original case.
    """
    tape = _Tape(seed)
    spins = [up.copy(), down.copy()]
    energies, weights = [], []
    for tick in range(warmup + measurements):
        exponent = 0.0
        for _ in range(hops_per_interval):
            mesh_up, mesh_dn, fn, propagation, _ = (
                np.asarray(value) for value in _leaf_step(ham, *spins, ecp_neighbors=ecp_neighbors)
            )
            rate = -float(fn.sum())
            factor = rate / (float(propagation) + rate - shift)
            if not math.isfinite(factor) or factor <= 0:
                raise ValueError("the selected fixed-step path must have finite positive branching factors")
            exponent += math.log(factor)
            cumulative = np.cumsum(-fn)
            selected = int(np.searchsorted(cumulative, tape.uniform() * cumulative[-1]))
            spins[:] = [mesh_up[selected].copy(), mesh_dn[selected].copy()]
        energies.append(float(_leaf_step(ham, *spins, ecp_neighbors=ecp_neighbors)[-1]))
        weights.append(math.exp(exponent))
        if len(energies) > collect and tick % 10 == 0:
            shift = _history_estimate(energies, weights, collect)
    return np.array(energies[warmup:]), np.array(weights[warmup:]), dict(draws=tape.draws, hops=tape.draws)
