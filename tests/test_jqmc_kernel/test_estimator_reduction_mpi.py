"""Fixed-ensemble references for actual MPI force/stress reductions."""

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

from dataclasses import replace
import inspect
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mpi4py import MPI

from jqmc._precision import get_dtype_np, get_tolerance_min
from jqmc.jqmc_mcmc import MCMC
from tests._fixed_estimator_oracle import energy_variance_reference, estimator_reference, fixed_samples, normalized_energy
from tests._mpi_collective_check import collective_check
from tests.test_jqmc_kernel.test_mcmc_energy_rational_reference import _rational_reference
from tests.test_jqmc_kernel.test_gfmc_estimator_reference import _history_settings, _stored_driver, _with_energy_second_moment
from tests.test_jqmc_kernel.test_gfmc_weight_scaling import BINS as OVERFLOW_BINS
from tests.test_jqmc_kernel.test_gfmc_weight_scaling import WARMUP as OVERFLOW_WARMUP
from tests.test_jqmc_kernel.test_gfmc_weight_scaling import jackknife_reference, log_space_weights, overflow_history
from tests.test_jqmc_kernel.test_gfmc_weight_scaling import stored_driver as overflow_stored_driver

pytestmark = [pytest.mark.light, pytest.mark.without_chdir, pytest.mark.mpi_only]


def _rank_samples(rank, complex_values):
    samples = fixed_samples(complex_values, get_dtype_np("local_energy"))
    return replace(
        samples,
        energy=samples.energy + rank * (0.7 + 0.2 * samples.energy**2),
        bare_weights=samples.bare_weights * (1 + 0.13 * rank),
        force_hf=samples.force_hf * (1 + rank * 0.23),
        force_score=samples.force_score + rank * 0.09,
        stress_hf=samples.stress_hf * (1 - rank * 0.31),
        stress_score=samples.stress_score + rank * 0.07,
    )


def _mcmc_driver(samples, comm):
    driver = MCMC.__new__(MCMC)
    driver._mpi_comm, driver._mpi_rank, driver._mpi_size = comm, comm.Get_rank(), comm.Get_size()
    driver._MCMC__mcmc_counter, driver._MCMC__num_walkers = samples.energy.shape
    driver._MCMC__comput_stress = True
    driver._MCMC__hamiltonian_data = _stored_driver("conventional", samples, "uniform", 1.0)._hamiltonian_data
    driver._MCMC__stored_w_L = samples.fractions * samples.bare_weights[:, None]
    driver._MCMC__stored_e_L = samples.energy
    for quantity, shape in (("force", (10, 3, 2, 3)), ("stress", (10, 3, 3, 3))):
        hf = getattr(samples, quantity + "_hf")
        score = getattr(samples, quantity + "_score")
        setattr(driver, "_MCMC__stored_" + quantity + "_HF", hf.reshape(shape))
        setattr(driver, "_MCMC__stored_" + quantity + "_PP", score.reshape(shape))
        setattr(driver, "_MCMC__stored_E_L_" + quantity + "_PP", (samples.energy[..., None] * score).reshape(shape))
    return driver


def _distributed_reference(samples_by_rank, quantity, warmup, bins):
    """Delete one rank/walker/time block from raw ensembles, then differentiate."""
    prefix = "force" if quantity == "force" else "stress"
    energy = np.stack([s.energy[warmup:] for s in samples_by_rank])
    hf = np.stack([getattr(s, prefix + "_hf")[warmup:] for s in samples_by_rank])
    score = np.stack([getattr(s, prefix + "_score")[warmup:] for s in samples_by_rank])
    weights = np.stack([(s.fractions * s.bare_weights[:, None])[warmup:] for s in samples_by_rank])
    dimensions = hf.shape[-1]
    values = []
    blocks = np.array_split(np.arange(energy.shape[1]), bins)
    for rank in range(len(samples_by_rank)):
        for block in blocks:
            for walker in range(energy.shape[2]):
                keep = np.ones(energy.shape, dtype=bool)
                keep[rank, block, walker] = False

                def objective(parameter):
                    if quantity == "force":
                        direction = parameter.reshape(-1)
                    elif quantity == "stress":
                        direction = ((parameter + parameter.T) / 2).reshape(-1)
                    else:
                        direction = (parameter * jnp.eye(3, dtype=hf.dtype)).reshape(-1)
                    return normalized_energy(
                        direction,
                        energy[keep][None, :],
                        hf[keep].reshape(1, -1, dimensions),
                        score[keep].reshape(1, -1, dimensions),
                        weights[keep][None, :],
                    )

                shape = (2, 3) if quantity == "force" else (3, 3) if quantity == "stress" else ()
                derivative = np.asarray(jax.grad(objective)(jnp.zeros(shape, dtype=hf.dtype)))
                values.append(
                    -derivative if quantity == "force" else derivative / 120 * (-1 / 3 if quantity == "pressure" else 1)
                )
    values = np.asarray(values)
    mean = values.mean(axis=0)
    error = np.sqrt((len(values) - 1) / len(values) * np.sum((values - mean) ** 2, axis=0))
    return np.asarray([mean, error])


def _estimate(driver, quantity, warmup, bins):
    return np.asarray(
        getattr(driver, {"force": "get_aF", "stress": "get_aT", "pressure": "get_P_hydrostatic"}[quantity])(warmup, bins)
    )


class _DropNonRootContributions:
    """Keep real collectives synchronized while dropping rank one's summed data."""

    def __init__(self, comm):
        self.comm = comm

    def __getattr__(self, name):
        return getattr(self.comm, name)

    def allreduce(self, value, op=MPI.SUM):
        local = value if self.comm.Get_rank() == 0 else np.zeros_like(value).item()
        return self.comm.allreduce(local, op=op)

    def Allreduce(self, send, receive, op=MPI.SUM):
        local = send if self.comm.Get_rank() == 0 else [np.zeros_like(send[0]), *send[1:]]
        self.comm.Allreduce(local, receive, op=op)


def _assert_faulty_estimate(actual, expected):
    """Apply the unchanged numerical verdict inside an expected-failure invocation."""
    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("quantity", ["force", "stress", "pressure"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_mcmc_mpi_estimators_match_raw_ensemble_derivatives(quantity, complex_values):
    """Two-rank force/stress/pressure means and errors match AD of a fixed raw ensemble.

    Parameter-set input, no wavefunction, TREXIO, spin or electronic positions.
    Each rank has ten slices and three unequal walkers, with deliberately different
    rank energies, weights and derivatives. Drop one warmup slice and use three
    local time bins. The independent reference deletes one rank/walker/time bin
    at a time and differentiates the normalized raw ensemble; it uses neither
    production moments nor collective/statistics routines. Actual accessors run
    on COMM_WORLD with two ranks. Mean and jackknife error are deterministic
    numbers checked at same-mode local_energy strict tolerances, never sigma bands.
    A second actual accessor run uses COMM_SELF and all six walkers; its bins are
    identical after repartition. This is aggregation, not sampler/force physics.
    """
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    samples = [_rank_samples(rank, complex_values) for rank in range(2)]
    driver = _mcmc_driver(samples[comm.Get_rank()], comm)
    actual = _estimate(driver, quantity, 1, 3)
    expected = _distributed_reference(samples, quantity, 1, 3)
    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    with collective_check(comm):
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
    serial = _mcmc_driver(samples[0], MPI.COMM_SELF)
    serial._MCMC__num_walkers = 6
    other = _mcmc_driver(samples[1], MPI.COMM_SELF)
    for name in ("w_L", "e_L", "force_HF", "force_PP", "E_L_force_PP", "stress_HF", "stress_PP", "E_L_stress_PP"):
        attr = "_MCMC__stored_" + name
        setattr(serial, attr, np.concatenate([getattr(serial, attr), getattr(other, attr)], axis=1))
    serial_actual = _estimate(serial, quantity, 1, 3)
    with collective_check(comm):
        np.testing.assert_allclose(serial_actual, expected, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("quantity", ["force", "stress", "pressure"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_mcmc_mpi_oracle_rejects_missing_nonroot_contribution(quantity, complex_values):
    """After a passing two-rank baseline, dropping non-root sums fails the same AD oracle."""
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    test_mcmc_mpi_estimators_match_raw_ensemble_derivatives(quantity, complex_values)
    samples = [_rank_samples(rank, complex_values) for rank in range(2)]
    driver = _mcmc_driver(samples[comm.Get_rank()], _DropNonRootContributions(comm))
    actual = _estimate(driver, quantity, 1, 3)
    expected = _distributed_reference(samples, quantity, 1, 3)
    with collective_check(comm), pytest.raises(AssertionError):
        _assert_faulty_estimate(actual, expected)


@pytest.mark.parametrize("driver_kind", ["conventional", "load-balanced"])
@pytest.mark.parametrize("history", ["uniform", "history-warmup"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_gfmc_mpi_estimators_match_fixed_ensemble_derivatives(driver_kind, history, complex_values):
    """GFMC distributed force and root-owned stress/pressure equal independent raw-ensemble AD.

    Ten fixed time slices of three unequal walkers are globally reduced moments,
    as production GFMC stores on rank zero; non-root buffers are poison NaNs.
    No sampling, wavefunction, positions, TREXIO or statistical acceptance band.
    History/warmup 0/0 and 2/1 exercise four/three unequal jackknife blocks, so
    force scatters unequal bins across two ranks. All three accessor mean/error
    pairs match independent AD and actual serial accessors on the same moments.
    Stress and pressure intentionally remain root-owned; non-root must return
    the documented exact zeros. Full/mixed use their own strict local_energy
    policy. This does not assert rank-invariant random trajectories.
    """
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    samples = fixed_samples(complex_values, get_dtype_np("local_energy"))
    collect, warmup, bins = _history_settings(history)
    driver = _stored_driver(driver_kind, samples, history, 1.0)
    driver._mpi_comm, driver._mpi_rank, driver._mpi_size = comm, comm.Get_rank(), comm.Get_size()
    if comm.Get_rank() != 0:
        for name in ("w_L", "e_L", "force_HF", "force_PP", "E_L_force_PP", "stress_HF", "stress_PP", "E_L_stress_PP"):
            attr = "_stored_" + name
            setattr(driver, attr, np.full_like(getattr(driver, attr), np.nan))
    serial = _stored_driver(driver_kind, samples, history, 1.0)
    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    for quantity in ("force", "stress", "pressure"):
        actual = _estimate(driver, quantity, warmup, bins)
        expected = np.asarray(estimator_reference(samples, quantity, collect, warmup, bins, 120.0))
        serial_actual = _estimate(serial, quantity, warmup, bins)
        with collective_check(comm):
            np.testing.assert_allclose(serial_actual, expected, atol=atol, rtol=rtol, equal_nan=False)
            if quantity != "force" and comm.Get_rank() != 0:
                np.testing.assert_array_equal(actual, np.zeros_like(expected))
        if quantity != "force":
            actual = comm.bcast(actual if comm.Get_rank() == 0 else None, root=0)
        with collective_check(comm):
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("driver_kind", ["conventional", "load-balanced"])
@pytest.mark.parametrize("history", ["uniform", "history-warmup"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_gfmc_mpi_force_oracle_rejects_missing_nonroot_bin(driver_kind, history, complex_values):
    """Dropping one rank's scattered GFMC force bins fails the independent fixed-ensemble oracle."""
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    test_gfmc_mpi_estimators_match_fixed_ensemble_derivatives(driver_kind, history, complex_values)
    samples = fixed_samples(complex_values, get_dtype_np("local_energy"))
    collect, warmup, bins = _history_settings(history)
    driver = _stored_driver(driver_kind, samples, history, 1.0)
    driver._mpi_comm = _DropNonRootContributions(comm)
    driver._mpi_rank, driver._mpi_size = comm.Get_rank(), comm.Get_size()
    actual = _estimate(driver, "force", warmup, bins)
    expected = np.asarray(estimator_reference(samples, "force", collect, warmup, bins, 120.0))
    with collective_check(comm), pytest.raises(AssertionError):
        _assert_faulty_estimate(actual, expected)


@pytest.mark.parametrize("driver_kind", ["conventional", "load-balanced"])
@pytest.mark.parametrize("history", ["uniform", "history-warmup"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_gfmc_mpi_energy_variance_matches_weighted_jackknife_modulus_reference(driver_kind, history, complex_values):
    """Two-rank GFMC get_E energy and variance equal an independent weighted jackknife.

    The same ten fixed time slices of three unequal walkers are globally reduced
    moments on rank zero, as production GFMC stores them; non-root buffers are
    poison NaNs. Four/three blocks on two ranks take the scattered-bin branch.
    The reference is the walker-level weighted delete-one-block jackknife with
    explicit bare-weight products, reporting Re<e_L> and
    <|e_L|^2> - |<e_L>|^2 of the complex replicate mean. All four outputs on
    every rank use the same-mode local_energy strict policy; no statistical band.
    """
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    samples = fixed_samples(complex_values, get_dtype_np("local_energy"))
    collect, warmup, bins = _history_settings(history)
    driver = _with_energy_second_moment(_stored_driver(driver_kind, samples, history, 1.0), samples)
    driver._mpi_comm, driver._mpi_rank, driver._mpi_size = comm, comm.Get_rank(), comm.Get_size()
    if comm.Get_rank() != 0:
        for name in ("w_L", "e_L", "e_L2"):
            attr = "_stored_" + name
            setattr(driver, attr, np.full_like(getattr(driver, attr), np.nan))
    actual = driver.get_E(warmup, bins)
    expected = energy_variance_reference(samples, collect, warmup, bins)
    with collective_check(comm):
        _assert_energy_quartet(actual, expected)


@pytest.mark.parametrize("driver_kind", ["conventional", "load-balanced"])
@pytest.mark.parametrize("collect", [100, 200])
def test_gfmc_mpi_energy_force_match_log_space_reference_beyond_float64_products(driver_kind, collect):
    """Two-rank GFMC get_E and get_aF equal a log-space reference when window products exceed float64.

    The ~-100 Ha stored moments of test_gfmc_weight_scaling (tau = 0.1, collect
    100/200, unscaled window products about e^1000/e^2000) are globally reduced
    moments on rank zero, as production GFMC stores them; non-root buffers are
    poison NaNs. Four blocks on two ranks take the scattered-bin branch of get_E
    and get_aF. The reference sums window log-weights with math.fsum, subtracts
    their maximum and evaluates the jackknife definitions directly. Energy,
    variance and force means/errors on every rank use the same-mode local_energy
    strict policy; no statistical band.
    """
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    history = overflow_history(collect)
    window_weights, log_products = log_space_weights(history.weights, collect, OVERFLOW_WARMUP)
    assert np.max(log_products) > np.log(np.finfo(np.float64).max)
    expected = jackknife_reference(history, window_weights, collect, OVERFLOW_WARMUP, OVERFLOW_BINS)
    driver = overflow_stored_driver(driver_kind, "production", history, collect, comm)
    if comm.Get_rank() != 0:
        for name in ("w_L", "e_L", "e_L2", "force_HF", "force_PP", "E_L_force_PP"):
            attr = "_stored_" + name
            setattr(driver, attr, np.full_like(getattr(driver, attr), np.nan))
    energy_mean, energy_error, variance_mean, variance_error = driver.get_E(OVERFLOW_WARMUP, OVERFLOW_BINS)
    force_mean, force_error = driver.get_aF(OVERFLOW_WARMUP, OVERFLOW_BINS)
    actual = {
        "energy": (energy_mean, energy_error),
        "variance": (variance_mean, variance_error),
        "force": (force_mean, force_error),
    }
    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    with collective_check(comm):
        for quantity, values in actual.items():
            for actual_value, expected_value in zip(values, expected[quantity], strict=True):
                np.testing.assert_allclose(
                    actual_value, expected_value, atol=atol, rtol=rtol, equal_nan=False, err_msg=quantity
                )


# --------------------------------------------------------------------------- #
# get_E across rank counts
# --------------------------------------------------------------------------- #

_ENERGY_RANK_WALKERS = 2


def _global_integer_energy_samples(complex_values):
    """Integer raw samples for 2 ranks x 2 walkers, no sampling, wavefunction or positions.

    Columns 0-1 belong to rank 0 and columns 2-3 to rank 1, so the two ranks
    hold different weights and energies.
    """
    walkers = 2 * _ENERGY_RANK_WALKERS
    time = np.arange(10)[:, None]
    walker = np.arange(walkers)[None, :]
    weights = 1 + (time + 2 * walker) % 5 + 3 * (walker >= _ENERGY_RANK_WALKERS)
    real = 2 * time - 3 * walker + (time * walker) % 4 + 7 * (walker >= _ENERGY_RANK_WALKERS)
    imag = (time - 2 * walker) % 3 if complex_values else np.zeros_like(real)
    return weights, real, imag


def _energy_driver(weights, real, imag, complex_values, comm):
    driver = MCMC.__new__(MCMC)
    driver._mpi_comm, driver._mpi_rank, driver._mpi_size = comm, comm.Get_rank(), comm.Get_size()
    driver._MCMC__mcmc_counter, driver._MCMC__num_walkers = weights.shape
    dtype = get_dtype_np("local_energy")
    driver._MCMC__stored_w_L = weights.astype(dtype)
    driver._MCMC__stored_e_L = real.astype(dtype) + (1j * imag if complex_values else 0)
    driver._MCMC__stored_e_L2 = (real**2 + imag**2).astype(dtype)
    return driver


def _assert_energy_quartet(actual, expected):
    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    for quantity, actual_value, expected_value in zip(
        ("energy-mean", "energy-error", "variance-mean", "variance-error"), actual, expected, strict=True
    ):
        np.testing.assert_allclose(actual_value, expected_value, atol=atol, rtol=rtol, equal_nan=False, err_msg=quantity)


def _local_deletion_get_E():
    """``MCMC.get_E`` whose jackknife deletes a bin from the LOCAL sums instead of the global ones."""
    source = textwrap.dedent(inspect.getsource(MCMC.get_E))
    replacements = (
        (
            "(w_L_e_L_binned_global_sum - w_L_e_L_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])",
            "(w_L_e_L_binned_local_sum - w_L_e_L_binned_local[m]) / (w_L_binned_local_sum - w_L_binned_local[m])",
        ),
        (
            "(w_L_e_L2_binned_global_sum - w_L_e_L2_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])",
            "(w_L_e_L2_binned_local_sum - w_L_e_L2_binned_local[m]) / (w_L_binned_local_sum - w_L_binned_local[m])",
        ),
    )
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    namespace = dict(MCMC.get_E.__globals__)
    exec(compile(source, "<local-deletion-MCMC.get_E>", "exec"), namespace)
    return namespace["get_E"]


@pytest.mark.parametrize("warmup,bins", [(0, 4), (1, 3)])
@pytest.mark.parametrize("complex_values", [False, True])
def test_mcmc_mpi_energy_matches_exact_rational_reference_and_serial(complex_values, warmup, bins):
    """Two-rank get_E equals the exact-rational reference and a one-rank run on the same samples.

    Parameter-set input: ten time slices, two walkers per rank, integer weights
    and real or complex energies that differ between the ranks. No TREXIO,
    wavefunction, positions or sampling. Warmup/bins 0/4 and 1/3 give unequal
    and equal time blocks. The reference is the exact-rational delete-one
    (time block, walker) jackknife of test_mcmc_energy_rational_reference on
    the concatenated 4-walker set, so it deletes each rank's cell from the
    GLOBAL sums. The four outputs (energy mean and error, variance mean and
    error) of get_E on COMM_WORLD with two ranks, and of get_E on COMM_SELF
    holding all four walkers, must each equal the reference at the same-mode
    local_energy strict tolerance.
    """
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    weights, real, imag = _global_integer_energy_samples(complex_values)
    expected = _rational_reference(weights, real, imag, warmup, bins)
    rank_columns = slice(comm.Get_rank() * _ENERGY_RANK_WALKERS, (comm.Get_rank() + 1) * _ENERGY_RANK_WALKERS)
    distributed = _energy_driver(weights[:, rank_columns], real[:, rank_columns], imag[:, rank_columns], complex_values, comm)
    with collective_check(comm):
        _assert_energy_quartet(distributed.get_E(warmup, bins), expected)
    serial = _energy_driver(weights, real, imag, complex_values, MPI.COMM_SELF)
    with collective_check(comm):
        _assert_energy_quartet(serial.get_E(warmup, bins), expected)


@pytest.mark.parametrize("complex_values", [False, True])
def test_mcmc_mpi_energy_oracle_rejects_local_jackknife_deletion(monkeypatch, complex_values):
    """Deleting a bin from the rank-local sums is invisible on one rank and fails on two.

    After the passing baseline, get_E is rebuilt from its own source with the
    global sums in the two jackknife numerators/denominators replaced by the
    rank-local sums (production files are not edited). On COMM_SELF with all
    walkers local == global, so the faulty accessor still matches the exact
    reference; on two ranks the same reference must fail at unchanged
    tolerance.
    """
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    test_mcmc_mpi_energy_matches_exact_rational_reference_and_serial(complex_values, 1, 3)
    weights, real, imag = _global_integer_energy_samples(complex_values)
    expected = _rational_reference(weights, real, imag, 1, 3)
    monkeypatch.setattr(MCMC, "get_E", _local_deletion_get_E())
    serial = _energy_driver(weights, real, imag, complex_values, MPI.COMM_SELF)
    with collective_check(comm):
        _assert_energy_quartet(serial.get_E(1, 3), expected)
    rank_columns = slice(comm.Get_rank() * _ENERGY_RANK_WALKERS, (comm.Get_rank() + 1) * _ENERGY_RANK_WALKERS)
    distributed = _energy_driver(weights[:, rank_columns], real[:, rank_columns], imag[:, rank_columns], complex_values, comm)
    actual = distributed.get_E(1, 3)
    with collective_check(comm), pytest.raises(AssertionError, match="energy-mean|energy-error|variance"):
        _assert_energy_quartet(actual, expected)
