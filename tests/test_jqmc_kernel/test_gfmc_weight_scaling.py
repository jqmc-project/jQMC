"""GFMC estimators on collect windows whose accumulated weights exceed the float64 range (issue #816).

Each stored step carries the walker-reduced moments of a molecule with a total
energy near -100 Ha and a branching weight ``exp(-tau e_proj)`` with tau = 0.1
(about e^10 per step), as GFMC_t stores them without a reference-energy shift.
A collect window of 100 or 200 steps then multiplies factors whose product is
about e^1000 or e^2000, beyond float64 (about e^709). The estimators must still
equal an independent log-space reference: the window log-weights are summed
with ``math.fsum`` and the largest retained one is subtracted before
exponentiation, which leaves every ratio of the definition unchanged.
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

import math
from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI

from jqmc import jqmc_gfmc
from jqmc._precision import get_tolerance_min
from jqmc.structure import Structure_data

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

TAU = 0.1
WARMUP = 3
RETAINED = 37
BINS = 4
NUM_ATOMS = 2
CELL = np.asarray([[4.0, 0.0, 0.0], [0.7, 5.0, 0.0], [0.2, 0.4, 6.0]])
CELL_VOLUME = 120.0  # 4 * 5 * 6 for the lower-triangular cell above


def overflow_history(collect, seed=816):
    """Per-step walker-reduced moments of a ~-100 Ha system for ``collect + WARMUP + RETAINED`` steps."""
    rng = np.random.default_rng(seed + collect)
    rows = collect + WARMUP + RETAINED
    energy = -100.0 + 0.5 * rng.standard_normal(rows)
    projected_energy = energy + 0.3 * rng.standard_normal(rows)
    history = SimpleNamespace(
        weights=np.exp(-TAU * projected_energy),
        energy=energy,
        energy2=energy**2 + 0.2 + 0.05 * rng.standard_normal(rows),
    )
    for quantity, shape in (("force", (rows, NUM_ATOMS, 3)), ("stress", (rows, 3, 3))):
        score = rng.standard_normal(shape)
        setattr(history, quantity + "_hf", rng.standard_normal(shape))
        setattr(history, quantity + "_pp", score)
        setattr(
            history, quantity + "_epp", energy.reshape((rows,) + (1,) * (len(shape) - 1)) * score + rng.standard_normal(shape)
        )
    return history


def log_space_weights(weights, collect, warmup):
    """Retained window weights ``exp(log G - max log G)`` with ``log G`` an fsum of the window log-weights."""
    log_weights = np.log(weights)
    log_products = np.asarray([math.fsum(log_weights[j : j + collect]) for j in range(len(weights) - collect)])[warmup:]
    return np.exp(log_products - np.max(log_products)), log_products


def unscaled_product_weights(weights, collect, warmup):
    """Retained window weights as the explicit unscaled products of the preceding ``collect`` weights."""
    return np.asarray([np.prod(weights[j : j + collect]) for j in range(len(weights) - collect)])[warmup:]


def jackknife_reference(history, window_weights, collect, warmup, bins):
    """Delete-one-time-block jackknife of energy, variance, force and stress from the definitions.

    Block edges follow the documented equal split (the first ``n % bins`` blocks
    hold one more step). For each deleted block the weighted means over the
    other steps give E = <e>, Var = <e^2> - <e>^2, F = -<F_HF> - 2(<e F_PP> - <e><F_PP>)
    and sigma = sym(<S_HF> + 2(<e S_PP> - <e><S_PP>)) / volume. The reported value
    is the replica mean and the error sqrt(B - 1) times the replica standard deviation.
    """
    retained = slice(collect + warmup, None)
    moments = {name: np.asarray(getattr(history, name))[retained] for name in vars(history) if name != "weights"}
    n = window_weights.shape[0]
    sizes = [n // bins + (1 if b < n % bins else 0) for b in range(bins)]
    edges = np.concatenate([[0], np.cumsum(sizes)])
    replicas = {"energy": [], "variance": [], "force": [], "stress": []}
    for deleted in range(bins):
        kept = np.ones(n, dtype=bool)
        kept[edges[deleted] : edges[deleted + 1]] = False
        g = window_weights[kept]

        def mean(values, g=g, kept=kept):
            values = values[kept]
            return np.tensordot(g, values, axes=(0, 0)) / np.sum(g)

        e_mean = mean(moments["energy"])
        replicas["energy"].append(e_mean)
        replicas["variance"].append(mean(moments["energy2"]) - e_mean**2)
        replicas["force"].append(
            -mean(moments["force_hf"]) - 2.0 * (mean(moments["force_epp"]) - e_mean * mean(moments["force_pp"]))
        )
        stress = mean(moments["stress_hf"]) + 2.0 * (mean(moments["stress_epp"]) - e_mean * mean(moments["stress_pp"]))
        replicas["stress"].append(0.5 * (stress + stress.T) / CELL_VOLUME)
    reference = {}
    for name, values in replicas.items():
        values = np.asarray(values)
        reference[name] = (np.mean(values, axis=0), np.sqrt(bins - 1) * np.std(values, axis=0))
    return reference


def stored_driver(driver_kind, implementation, history, collect, comm=MPI.COMM_SELF):
    """Inject the stored per-step moments; no sampling, wavefunction or electron coordinates."""
    name = {"conventional": "GFMC_t", "load-balanced": "GFMC_n"}[driver_kind]
    driver_class = getattr(jqmc_gfmc, "_" + name + "_debug" if implementation == "debug" else name)
    driver = driver_class.__new__(driver_class)
    driver._mpi_comm, driver._mpi_rank, driver._mpi_size = comm, comm.Get_rank(), comm.Get_size()
    rows = history.energy.shape[0]
    driver._mcmc_counter = rows
    driver._num_gfmc_collect_steps = collect
    driver._comput_stress = True
    structure = Structure_data.build(
        positions=np.asarray([[0.0, 0.0, 0.0], [1.2, 0.1, 0.2]]),
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H1", "H2"),
        pbc_flag=True,
        vec_a=CELL[0],
        vec_b=CELL[1],
        vec_c=CELL[2],
    )
    driver._hamiltonian_data = SimpleNamespace(structure_data=structure)
    driver._stored_w_L = history.weights[:, None]
    driver._stored_e_L = history.energy[:, None]
    driver._stored_e_L2 = history.energy2[:, None]
    for quantity in ("force", "stress"):
        for suffix, stored in (("hf", "_HF"), ("pp", "_PP")):
            values = getattr(history, quantity + "_" + suffix)
            setattr(driver, "_stored_" + quantity + stored, values[:, None])
        setattr(driver, "_stored_E_L_" + quantity + "_PP", getattr(history, quantity + "_epp")[:, None])
    return driver


def _driver_estimates(driver):
    energy_mean, energy_error, variance_mean, variance_error = driver.get_E(WARMUP, BINS)
    force_mean, force_error = driver.get_aF(WARMUP, BINS)
    stress_mean, stress_error = driver.get_aT(WARMUP, BINS)
    return {
        "energy": (energy_mean, energy_error),
        "variance": (variance_mean, variance_error),
        "force": (force_mean, force_error),
        "stress": (stress_mean, stress_error),
    }


@pytest.mark.parametrize("driver_kind", ["conventional", "load-balanced"])
@pytest.mark.parametrize("implementation", ["production", "debug"])
@pytest.mark.parametrize("collect", [100, 200])
def test_gfmc_estimators_match_log_space_reference_beyond_float64_products(driver_kind, implementation, collect):
    """get_E, get_aF and get_aT equal a log-space reference when the window products exceed float64.

    Parameter-set stored moments (no TREXIO, wavefunction or sampling) of a ~-100 Ha
    system with tau = 0.1 and collect windows of 100 and 200 steps, whose
    unscaled products are about e^1000 and e^2000; the input check below states
    that they exceed the float64 range. Conventional/load-balanced accessors,
    production and debug twins, COMM_SELF, three warmup windows and four unequal
    blocks of 37 retained windows. The independent reference sums the window
    log-weights with math.fsum, subtracts their maximum before exponentiating and
    evaluates the jackknife definitions directly. Every mean and error uses the
    local_energy strict policy of the current precision mode (float64 storage in
    both modes). No statistical acceptance band.
    """
    history = overflow_history(collect)
    window_weights, log_products = log_space_weights(history.weights, collect, WARMUP)
    assert np.max(log_products) > np.log(np.finfo(np.float64).max)
    expected = jackknife_reference(history, window_weights, collect, WARMUP, BINS)
    actual = _driver_estimates(stored_driver(driver_kind, implementation, history, collect))
    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    for quantity in ("energy", "variance", "force", "stress"):
        for actual_value, expected_value in zip(actual[quantity], expected[quantity], strict=True):
            np.testing.assert_allclose(actual_value, expected_value, atol=atol, rtol=rtol, equal_nan=False, err_msg=quantity)


@pytest.mark.parametrize("driver_kind", ["conventional", "load-balanced"])
@pytest.mark.parametrize("implementation", ["production", "debug"])
@pytest.mark.parametrize("collect", [2, 5])
def test_gfmc_estimators_equal_unscaled_products_for_short_windows(driver_kind, implementation, collect):
    """For representable products the estimates equal those of the unscaled window products.

    Same ~-100 Ha stored moments with collect windows of 2 and 5 steps, whose
    products (about e^20 and e^50) are ordinary float64 numbers. The reference
    weights are the explicit np.prod of the preceding raw weights, i.e. the
    direct-product definition, evaluated in the same direct jackknife as above. Every
    mean and error uses the local_energy strict policy of the current mode.
    """
    history = overflow_history(collect)
    window_weights = unscaled_product_weights(history.weights, collect, WARMUP)
    assert np.all(np.isfinite(window_weights))
    expected = jackknife_reference(history, window_weights, collect, WARMUP, BINS)
    actual = _driver_estimates(stored_driver(driver_kind, implementation, history, collect))
    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    for quantity in ("energy", "variance", "force", "stress"):
        for actual_value, expected_value in zip(actual[quantity], expected[quantity], strict=True):
            np.testing.assert_allclose(actual_value, expected_value, atol=atol, rtol=rtol, equal_nan=False, err_msg=quantity)
