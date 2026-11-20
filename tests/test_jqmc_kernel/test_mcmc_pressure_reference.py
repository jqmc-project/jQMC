"""Deterministic pressure aggregation from fixed and production MCMC moments."""

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

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI
from uncertainties import ufloat

from jqmc import jqmc_cli
from jqmc._precision import get_tolerance_min
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.jqmc_mcmc import MCMC
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.without_chdir]


def _run_silicon(trexio_file):
    """Keep the existing stress fixture settings explicit for provenance extraction."""
    source = Path(__file__).resolve().parents[1] / "trexio_example_files" / trexio_file
    structure, _, _, _, geminal, coulomb = read_trexio_file(str(source), store_tuple=True)
    hamiltonian = Hamiltonian_data.build(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
    )
    driver = MCMC(
        hamiltonian_data=hamiltonian,
        Dt=2.0,
        mcmc_seed=34356,
        num_walkers=2,
        comput_position_deriv=True,
        comput_stress=True,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        epsilon_AS=1.0e-2,
    )
    driver.run(num_mcmc_steps=5)
    return driver


def _fixed_driver(correlation):
    """Populate real accessors with explicit correlated moments, without sampling."""
    driver = MCMC.__new__(MCMC)
    driver._mpi_comm, driver._mpi_rank, driver._mpi_size = MPI.COMM_SELF, 0, 1
    steps, walkers = 8, 2
    driver._MCMC__mcmc_counter, driver._MCMC__num_walkers = steps, walkers
    driver._MCMC__comput_stress = True
    structure = Structure_data.build(
        positions=np.asarray([[0.0, 0.0, 0.0]]),
        atomic_numbers=(1,),
        element_symbols=("H",),
        atomic_labels=("H1",),
        pbc_flag=True,
        vec_a=[4.0, 0.0, 0.0],
        vec_b=[0.7, 5.0, 0.0],
        vec_c=[0.2, 0.4, 6.0],
    )
    driver._MCMC__hamiltonian_data = SimpleNamespace(structure_data=structure)
    x = np.arange(steps * walkers, dtype=float).reshape(steps, walkers) / 8.0
    driver._MCMC__stored_w_L = 1.0 + x / 4.0
    driver._MCMC__stored_e_L = -2.0 + x**2
    diagonal = np.asarray([1.0, 2.0, 4.0 if correlation == "positive" else -2.5])
    driver._MCMC__stored_stress_HF = (1.0 + x[..., None, None]) * np.diag(diagonal)
    driver._MCMC__stored_stress_PP = x[..., None, None] * np.diag(diagonal / 8.0)
    driver._MCMC__stored_E_L_stress_PP = driver.e_L[..., None, None] * driver._MCMC__stored_stress_PP
    return driver


def _direct_deleted_sample_reference(driver, warmup, bins, volume):
    """Recompute scalar pressure after deleting raw samples; use pairwise variance."""
    weights, energy = driver.w_L[warmup:], driver.e_L[warmup:]
    hf, pp, epp = [
        np.trace(np.asarray(getattr(driver, "_MCMC__stored_" + name))[warmup:], axis1=2, axis2=3)
        for name in ("stress_HF", "stress_PP", "E_L_stress_PP")
    ]
    blocks = np.array_split(np.arange(len(weights)), bins)
    pressures = []
    for block in blocks:
        for walker in range(driver.num_walkers):
            keep = np.ones(weights.shape, dtype=bool)
            keep[block, walker] = False
            w = weights[keep]
            average = lambda values: np.dot(w, values[keep]) / np.sum(w)  # noqa: E731
            pressure = -(average(hf) + 2 * (average(epp) - average(energy) * average(pp))) / (3 * volume)
            pressures.append(pressure)
    values = np.asarray(pressures)
    count = len(values)
    variance = (count - 1) * np.sum((values[:, None] - values[None, :]) ** 2) / (2 * count**2)
    return np.asarray([values.mean(), np.sqrt(variance)])


def _assert_pressure(driver, warmup, bins, volume):
    expected = _direct_deleted_sample_reference(driver, warmup, bins, volume)
    actual = np.asarray(driver.get_P_hydrostatic(warmup, bins))
    assert actual.shape == expected.shape == (2,)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    assert expected[1] > 0
    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return actual


@pytest.mark.light
@pytest.mark.parametrize("correlation", ["positive", "negative"])
def test_pressure_matches_correlated_stored_moments(correlation):
    """MCMC pressure mean/error preserve positive and negative diagonal covariance.

    Explicit parameter-set moments, eight time slices, two walkers, nonuniform
    weights, no wavefunction, TREXIO, electronic positions or cond(G). A triangular
    cell has volume 120 bohr^3. Discard one sample and use three unequal time bins.
    Call the real accessor on COMM_SELF; independently delete each bin/walker
    from raw arrays and evaluate scalar pressure with pairwise-difference variance.
    Both outputs use the central local_energy strict policy within each mode.
    The two correlations exercise quadrature over- and under-estimation; neither
    sampling correctness nor statistical confidence calibration is claimed.
    """
    driver = _fixed_driver(correlation)
    _assert_pressure(driver, 1, 3, 120.0)


@pytest.mark.light
@pytest.mark.parametrize("correlation", ["positive", "negative"])
def test_pressure_reference_rejects_wrong_error(correlation, monkeypatch):
    """The real accessor reference rejects divided and diagonal-quadrature error bars.

    Establish the normal fixed-moment baseline, then corrupt only the hydrostatic
    result of get_aT while retaining its real stress replicates and mean. Each
    fault must fail the same strict numerical comparison, not a source-text check.
    """
    driver = _fixed_driver(correlation)
    _assert_pressure(driver, 1, 3, 120.0)
    original = MCMC.get_aT
    for fault in ("divide-error", "quadrature"):

        def wrong_error(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if not kwargs.get("_return_hydrostatic"):
                return result
            sigma, sigma_error, mean, error = result
            replacement = error / 3 if fault == "divide-error" else np.linalg.norm(np.diag(sigma_error)) / 3
            return sigma, sigma_error, mean, replacement

        with monkeypatch.context() as patch:
            patch.setattr(MCMC, "get_aT", wrong_error)
            with pytest.raises(AssertionError, match="Not equal to tolerance"):
                _assert_pressure(driver, 1, 3, 120.0)


@pytest.mark.heavy
@pytest.mark.parametrize("trexio_file", ["silicon.h5"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_pressure_matches_actual_silicon_moments(trexio_file):
    """Real Silicon MCMC pressure equals direct deletion of its saved stress moments.

    Use committed silicon.h5 and the existing stress fixture unchanged: no Jastrow,
    periodic MO, two walkers, seed 34356, Dt=2, epsilon_AS=0.01, five measurements,
    position/stress derivatives enabled. No cond(G) filtering. Discard one sample
    and use two time bins. Compute volume via an independent scalar triple product.
    Recompute the scalar pressure and pairwise jackknife error from actual stored
    HF/PP/E_L_PP moments, then compare both real accessor outputs within the current
    precision's central strict policy. No comparison across precision modes, no
    sigma threshold or physical-stress derivative/convergence claim.
    """
    driver = _run_silicon(trexio_file)
    cell = np.asarray(driver.hamiltonian_data.structure_data._cell_jnp)
    volume = abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
    _assert_pressure(driver, 1, 2, volume)


@pytest.mark.light
@pytest.mark.parametrize("correlation", ["positive"])
def test_pressure_formatter_uses_joint_error_or_refuses(monkeypatch, correlation):
    """Actual CLI formatting preserves the joint pressure error and refuses its absence.

    Use the positive-correlated explicit moment driver, no electron configuration.
    Capture the logger only: formatting executes with the accessor's covariance-aware
    result, whose expected value is independently recomputed from deleted samples.
    Never synthesize an error from the three diagonal error bars.
    """
    driver = _fixed_driver(correlation)
    pressure = _assert_pressure(driver, 1, 3, 120.0)
    sigma, error = driver.get_aT(1, 3)
    messages = []
    monkeypatch.setattr(jqmc_cli.logger, "info", messages.append)
    jqmc_cli._log_stress_section(sigma, error, -np.diag(sigma), np.diag(error), tuple(pressure))
    rows = [line.split() for line in messages if line.strip().startswith("P_hydro ")]
    assert rows == [["P_hydro", f"{ufloat(*pressure):+2uS}"]]
    with pytest.raises(ValueError, match="no error bar"):
        jqmc_cli._log_stress_section(sigma, error, -np.diag(sigma), np.diag(error), None)
