"""Deterministic independent references for stored GFMC estimator aggregation."""

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

from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI

from jqmc import jqmc_gfmc
from jqmc._precision import get_dtype_np, get_tolerance_min
from jqmc.structure import Structure_data
from tests._fixed_estimator_oracle import (
    energy_variance_reference,
    estimator_reference,
    fixed_mpc_corrections,
    fixed_samples,
    weighted_mean_reference,
)


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _history_settings(history):
    return {"uniform": (0, 0, 4), "history-warmup": (2, 1, 3)}[history]


def _stored_driver(driver_kind, samples, history, volume_scale, *, debug=False):
    """Inject reviewed moments; no sampling, wavefunction or electronic coordinates."""
    names = {"conventional": "GFMC_t", "load-balanced": "GFMC_n"}
    name = names[driver_kind]
    driver_class = getattr(jqmc_gfmc, "_" + name + "_debug" if debug else name)
    driver = driver_class.__new__(driver_class)
    driver._mpi_comm, driver._mpi_rank, driver._mpi_size = MPI.COMM_SELF, 0, 1
    driver._mcmc_counter = len(samples.energy)
    driver._num_gfmc_collect_steps = _history_settings(history)[0]
    driver._comput_stress = True
    cell = np.asarray([[4.0, 0.0, 0.0], [0.7, 5.0, 0.0], [0.2, 0.4, 6.0]]) * volume_scale
    structure = Structure_data.build(
        positions=np.asarray([[0.0, 0.0, 0.0], [1.2, 0.1, 0.2]]),
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H1", "H2"),
        pbc_flag=True,
        vec_a=cell[0],
        vec_b=cell[1],
        vec_c=cell[2],
    )
    driver._hamiltonian_data = SimpleNamespace(structure_data=structure)
    fractions, energy = samples.fractions, samples.energy
    driver._stored_w_L = samples.bare_weights[:, None]
    driver._stored_e_L = np.sum(fractions * energy, axis=1)[:, None]
    for quantity, shape in (("force", (10, 1, 2, 3)), ("stress", (10, 1, 3, 3))):
        hf = getattr(samples, quantity + "_hf")
        score = getattr(samples, quantity + "_score")
        setattr(driver, "_stored_" + quantity + "_HF", np.sum(fractions[:, :, None] * hf, axis=1).reshape(shape))
        setattr(driver, "_stored_" + quantity + "_PP", np.sum(fractions[:, :, None] * score, axis=1).reshape(shape))
        cross_moment = np.sum((fractions * energy)[:, :, None] * score, axis=1)
        setattr(driver, "_stored_E_L_" + quantity + "_PP", cross_moment.reshape(shape))
    return driver


@pytest.mark.parametrize("driver_kind", ["conventional", "load-balanced"])
@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("history", ["uniform", "history-warmup"])
def test_gfmc_force_matches_fixed_ensemble_energy_derivative(driver_kind, complex_values, history):
    """GFMC force mean and jackknife error equal negative AD of a fixed normalized energy.

    Parameter-set input, no TREXIO, wavefunction, spin state, electron positions or
    cond(G): ten time slices with three unequal walker fractions each. Explicit
    real/complex energies, real local-energy/log-amplitude gradients and nonzero
    within-slice covariance are injected as stored moments into the conventional
    or load-balanced accessor (COMM_SELF, no propagation). History 0 with warmup 0
    uses four unequal blocks; history 2 with warmup 1 uses three unequal blocks.
    Independent AD differentiates the normalized mean of linear local energies
    and exponential weights on each leave-one-block-out ensemble, before any
    stored-moment aggregation. Mean and error use the central local_energy strict
    policy within each mode; storage is float64/complex128 in both modes, not a
    comparison between precision modes. No stochastic confidence criterion, SWCT,
    Pathak-Wagner, distributed reduction or sampler-to-buffer correctness is claimed.
    """
    samples = fixed_samples(complex_values, get_dtype_np("local_energy"))
    collect, warmup, bins = _history_settings(history)
    driver = _stored_driver(driver_kind, samples, history, 1.0)
    actual_mean, actual_error = driver.get_aF(warmup, bins)
    expected_mean, expected_error = estimator_reference(samples, "force", collect, warmup, bins, 120.0)
    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    np.testing.assert_allclose(actual_mean, expected_mean, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_error, expected_error, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("driver_kind", ["conventional", "load-balanced"])
@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("history", ["uniform", "history-warmup"])
@pytest.mark.parametrize("volume_scale", [1.0, 2.0])
def test_gfmc_stress_pressure_match_fixed_ensemble_strain_derivatives(driver_kind, complex_values, history, volume_scale):
    """GFMC stress and pressure means and jackknife errors equal independent fixed-ensemble strain derivatives.

    Parameter-set stored buffers, no TREXIO, wavefunction, spin state, electronic
    configuration or cond(G). Ten time slices each contain three unequal walkers
    with explicit real/complex energies and real nine-component gradients. Actual
    conventional/load-balanced accessors run on COMM_SELF, without sampling.
    Independent AD differentiates a normalized energy under symmetric strain;
    pressure separately differentiates scalar isotropic strain, retaining diagonal
    covariance. A triangular cell has known volume 120 or 960 bohr^3 (length scale
    1 or 2), independently of production's determinant calculation. History/warmup
    0/0 with four blocks and 2/1 with three blocks exercise nonuniform weights and
    unequal block lengths. All four means/errors use the existing local_energy
    strict policy in the current precision mode (float64/complex128 storage in
    both), never a sigma acceptance band. This guarantees postprocessing only,
    not physical energy derivatives, propagation, SWCT or distributed reduction.
    """
    samples = fixed_samples(complex_values, get_dtype_np("local_energy"))
    collect, warmup, bins = _history_settings(history)
    driver = _stored_driver(driver_kind, samples, history, volume_scale)
    stress_mean, stress_error = driver.get_aT(warmup, bins)
    pressure_mean, pressure_error = driver.get_P_hydrostatic(warmup, bins)
    volume = 120.0 * volume_scale**3
    expected_stress, expected_stress_error = estimator_reference(samples, "stress", collect, warmup, bins, volume)
    expected_pressure, expected_pressure_error = estimator_reference(samples, "pressure", collect, warmup, bins, volume)
    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    np.testing.assert_allclose(stress_mean, expected_stress, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(stress_error, expected_stress_error, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(pressure_mean, expected_pressure, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(pressure_error, expected_pressure_error, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("driver_kind", ["conventional", "load-balanced"])
@pytest.mark.parametrize("history", ["uniform", "history-warmup"])
def test_gfmc_energy_and_mpc_match_fixed_ensemble_weighted_jackknife(driver_kind, history):
    """GFMC energy, MPC energy and MPC correction equal an independent weighted jackknife.

    Parameter-set stored buffers, no TREXIO, wavefunction, spin state, electronic
    configuration or cond(G): ten time slices with three unequal walker fractions,
    explicit real local energies and real per-walker MPC-minus-Ewald differences,
    injected as per-slice moments into the conventional or load-balanced accessor
    (COMM_SELF, no propagation). History/warmup 0/0 with four blocks (ten retained
    slices) and 2/1 with three blocks (seven retained slices) exercise unequal
    blocks, nonuniform two-step weight products and warmup. The independent numpy
    reference forms each leave-one-block-out ensemble over individual walkers with
    explicit products of the preceding bare weights and returns the replicate mean
    and the delete-one-block jackknife error sqrt((B-1)/B sum (r_b - r)^2) of
    Re<e_L>, Re<e_L + dV_MPC> and <dV_MPC>, without any production binning,
    window-product or statistics helper. Mean and error use the central
    local_energy strict policy within each mode; storage is float64 in both modes.
    The energy variance and complex local energies are not asserted here, and no
    sampling, distributed reduction or statistical confidence criterion is claimed.
    """
    samples = fixed_samples(False, get_dtype_np("local_energy"))
    mpc_corrections = fixed_mpc_corrections(get_dtype_np("local_energy"))
    collect, warmup, bins = _history_settings(history)
    driver = _stored_driver(driver_kind, samples, history, 1.0)
    driver._stored_e_L2 = np.sum(samples.fractions * np.abs(samples.energy) ** 2, axis=1)[:, None]
    driver._stored_mpc_dV = np.sum(samples.fractions * mpc_corrections, axis=1)[:, None]
    driver._mpc = True
    energy_mean, energy_error, _, _ = driver.get_E(warmup, bins)
    mpc_energy_mean, mpc_energy_error = driver.get_E_mpc(warmup, bins)
    correction_mean, correction_error = driver.get_dE_mpc(warmup, bins)
    expected_energy, expected_energy_error = weighted_mean_reference(samples, samples.energy, collect, warmup, bins)
    expected_mpc_energy, expected_mpc_energy_error = weighted_mean_reference(
        samples, samples.energy + mpc_corrections, collect, warmup, bins
    )
    expected_correction, expected_correction_error = weighted_mean_reference(samples, mpc_corrections, collect, warmup, bins)
    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    np.testing.assert_allclose(energy_mean, expected_energy, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(energy_error, expected_energy_error, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(mpc_energy_mean, expected_mpc_energy, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(mpc_energy_error, expected_mpc_energy_error, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(correction_mean, expected_correction, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(correction_error, expected_correction_error, atol=atol, rtol=rtol, equal_nan=False)


def _with_energy_second_moment(driver, samples):
    """Store the walker-fraction average of ``|e_L|^2`` per slice, as production stores ``e_L2``."""
    driver._stored_e_L2 = np.sum(samples.fractions * np.abs(samples.energy) ** 2, axis=1)[:, None]
    return driver


@pytest.mark.parametrize("driver_kind", ["conventional", "load-balanced"])
@pytest.mark.parametrize("implementation", ["production", "debug"])
@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("history", ["uniform", "history-warmup"])
def test_gfmc_energy_variance_matches_weighted_jackknife_modulus_reference(
    driver_kind, implementation, complex_values, history
):
    """GFMC get_E energy and variance equal an independent weighted delete-one-block jackknife.

    Parameter-set stored buffers, no TREXIO, wavefunction, electronic configuration
    or sampling: ten time slices of three unequal walkers with explicit real or
    complex energies (complex imaginary parts are not small, so a wrong complex
    convention is visible). Conventional and load-balanced accessors, production
    and debug twins, run on COMM_SELF. The reference averages the unaggregated
    walkers of each leave-one-block-out ensemble with explicit products of the
    preceding bare weights; it reports Re<e_L> and the variance
    <|e_L|^2> - |<e_L>|^2 of the complex replicate mean (error_analysis.md,
    "Complex observables"), not (Re<e_L>)^2. History/warmup 0/0 with four
    blocks and 2/1 with three blocks give unequal blocks that do not divide the
    retained slices. All four outputs use the local_energy strict policy of the
    current mode (float64/complex128 storage in both); no statistical band.
    """
    samples = fixed_samples(complex_values, get_dtype_np("local_energy"))
    collect, warmup, bins = _history_settings(history)
    driver = _stored_driver(driver_kind, samples, history, 1.0, debug=implementation == "debug")
    driver = _with_energy_second_moment(driver, samples)
    energy, energy_error, variance, variance_error = driver.get_E(warmup, bins)
    expected_energy, expected_energy_error, expected_variance, expected_variance_error = energy_variance_reference(
        samples, collect, warmup, bins
    )
    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    np.testing.assert_allclose(energy, expected_energy, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(energy_error, expected_energy_error, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(variance, expected_variance, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(variance_error, expected_variance_error, atol=atol, rtol=rtol, equal_nan=False)
