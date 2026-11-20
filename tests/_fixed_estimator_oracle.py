"""Independent derivatives of a finite, normalized, weighted ensemble.

No production estimator, binning, window-product or statistics helper is used.
The model is an accessor reference, not a physical wavefunction or an oracle for
how propagation, SWCT or regularization generates the stored samples.
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

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class FixedSamples:
    """Ten time slices, each containing three unequally weighted walkers."""

    fractions: np.ndarray
    energy: np.ndarray
    bare_weights: np.ndarray
    force_hf: np.ndarray
    force_score: np.ndarray
    stress_hf: np.ndarray
    stress_score: np.ndarray


def fixed_samples(complex_values, dtype):
    """Explicit nonseparable values with real log-amplitude/local-energy gradients."""
    time = np.arange(10, dtype=dtype)[:, None]
    walker = np.arange(3, dtype=dtype)[None, :]
    base = 1.0 + 0.12 * time + 0.19 * walker
    fractions = base / base.sum(axis=1, keepdims=True)
    energy = -1.1 + 0.13 * time + 0.21 * np.sin(0.5 * time + walker)
    if complex_values:
        energy = energy + 0.17j * np.cos(0.31 * time + 0.7 * walker)
    derivatives = []
    for dimensions in (6, 9):
        axis = np.arange(dimensions, dtype=dtype)[None, None, :]
        phase = 0.37 * (time[:, :, None] + 1) * (axis + 1) + 0.43 * walker[:, :, None]
        derivatives.extend(
            (
                0.15 * np.sin(phase) + 0.03 * axis + 0.01 * time[:, :, None],
                0.22 * np.cos(0.61 * phase) - 0.04 * walker[:, :, None],
            )
        )
    bare_weights = 0.8 + 0.06 * time[:, 0] + 0.1 * np.cos(0.3 * time[:, 0])
    return FixedSamples(fractions, energy, bare_weights, *derivatives)


def leave_block_out_samples(samples, collect, warmup, bins):
    """Select whole time blocks and multiply preceding bare weights explicitly."""
    retained = list(range(collect + warmup, len(samples.bare_weights)))
    if collect < 0 or warmup < 0 or not 2 <= bins <= len(retained):
        raise ValueError("fixed-ensemble reference needs at least two nonempty blocks")
    quotient, remainder = divmod(len(retained), bins)
    offset = 0
    for block in range(bins):
        size = quotient + (block < remainder)
        keep = retained[:offset] + retained[offset + size :]
        offset += size
        weights = []
        for index in keep:
            product = 1.0
            for previous in range(index - collect, index):
                product *= samples.bare_weights[previous]
            weights.append(product)
        yield keep, np.asarray(weights)[:, None] * samples.fractions[keep]


def fixed_mpc_corrections(dtype):
    """Explicit real per-walker MPC-minus-Ewald electron-electron differences."""
    time = np.arange(10, dtype=dtype)[:, None]
    walker = np.arange(3, dtype=dtype)[None, :]
    return 0.04 + 0.011 * time - 0.017 * np.cos(0.8 * time + 1.3 * walker)


def weighted_mean_reference(samples, values, collect, warmup, bins):
    """Mean and jackknife error of the real part of a normalized weighted per-walker mean.

    Each replicate is sum(G_t f_tw v_tw) / sum(G_t f_tw) over one leave-one-block-out
    ensemble of ``leave_block_out_samples``, i.e. over individual walkers and explicit
    products of preceding bare weights, before any per-slice or per-block aggregation.
    """
    replicates = np.asarray(
        [
            np.real(np.sum(weights * values[keep]) / np.sum(weights))
            for keep, weights in leave_block_out_samples(samples, collect, warmup, bins)
        ]
    )
    mean = np.mean(replicates)
    error = np.sqrt((bins - 1) / bins * np.sum((replicates - mean) ** 2))
    return mean, error


def normalized_energy(direction, values, hf, score, weights):
    """Differentiate this finite quotient itself, not a copied covariance formula."""
    perturbed_weights = jnp.asarray(weights) * jnp.exp(2 * jnp.einsum("twd,d->tw", jnp.asarray(score), direction))
    perturbed_values = jnp.asarray(values) + jnp.einsum("twd,d->tw", jnp.asarray(hf), direction)
    return jnp.real(jnp.sum(perturbed_weights * perturbed_values) / jnp.sum(perturbed_weights))


def estimator_reference(samples, quantity, collect, warmup, bins, volume):
    """Mean and jackknife error of exact fixed-sample AD replicates.

    Stress differentiates a symmetric strain; pressure separately differentiates
    a scalar isotropic strain, preserving correlations between diagonal entries.
    The returned error is a deterministic number, never a pass/fail sigma band.
    """
    if quantity not in {"force", "stress", "pressure"}:
        raise ValueError("unknown fixed-ensemble quantity")
    prefix = "force" if quantity == "force" else "stress"
    hf = getattr(samples, prefix + "_hf")
    score = getattr(samples, prefix + "_score")
    replicates = []
    for keep, weights in leave_block_out_samples(samples, collect, warmup, bins):

        def objective(parameter):
            if quantity == "force":
                direction = parameter.reshape(-1)
            elif quantity == "stress":
                direction = ((parameter + parameter.T) / 2).reshape(-1)
            else:
                direction = (parameter * jnp.eye(3, dtype=hf.dtype)).reshape(-1)
            return normalized_energy(direction, samples.energy[keep], hf[keep], score[keep], weights)

        shape = (2, 3) if quantity == "force" else (3, 3) if quantity == "stress" else ()
        derivative = np.asarray(jax.grad(objective)(jnp.zeros(shape, dtype=hf.dtype)))
        replicates.append(
            -derivative if quantity == "force" else derivative / volume * (-1 / 3 if quantity == "pressure" else 1)
        )
    replicates = np.asarray(replicates)
    mean = np.mean(replicates, axis=0)
    error = np.sqrt((bins - 1) / bins * np.sum((replicates - mean) ** 2, axis=0))
    return mean, error


def energy_variance_reference(samples, collect, warmup, bins):
    """Weighted delete-one-block jackknife of the energy and the energy variance.

    Each replicate is a normalized finite average over the unaggregated walkers of
    the retained time slices, weighted by the explicit product of the preceding
    bare weights and the walker fraction. The energy replicate is Re<e_L>; the
    variance replicate is <|e_L|^2> - |<e_L>|^2 with the complex replicate mean
    (doc/notes/error_analysis.md, "Complex observables"). Returns
    (energy mean, energy error, variance mean, variance error).
    """
    energies, variances = [], []
    for keep, weights in leave_block_out_samples(samples, collect, warmup, bins):
        values = samples.energy[keep]
        mean = np.sum(weights * values) / np.sum(weights)
        second_moment = np.sum(weights * np.abs(values) ** 2) / np.sum(weights)
        energies.append(np.real(mean))
        variances.append(second_moment - np.abs(mean) ** 2)
    result = []
    for replicates in (np.asarray(energies), np.asarray(variances)):
        mean = np.mean(replicates)
        result.extend((mean, np.sqrt((bins - 1) / bins * np.sum((replicates - mean) ** 2))))
    return tuple(result)
