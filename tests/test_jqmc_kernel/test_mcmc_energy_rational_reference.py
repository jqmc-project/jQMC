"""Exact finite-sample references for energy, variance and their jackknife errors."""

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

from fractions import Fraction
import inspect
from math import sqrt
import textwrap

import numpy as np
import pytest
from mpi4py import MPI

from jqmc._precision import get_dtype_np, get_tolerance_min
from jqmc.jqmc_mcmc import MCMC

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _fixed_samples(walkers, complex_values):
    """Integer raw samples, no random generation, wavefunction or electronic positions."""
    time = np.arange(10)[:, None]
    walker = np.arange(walkers)[None, :]
    weights = 1 + (time + 2 * walker) % 5
    real = 2 * time - 3 * walker + (time * walker) % 4
    imag = (time - 2 * walker) % 3 if complex_values else np.zeros_like(real)
    driver = MCMC.__new__(MCMC)
    driver._mpi_comm, driver._mpi_rank, driver._mpi_size = MPI.COMM_SELF, 0, 1
    driver._MCMC__mcmc_counter = 10
    driver._MCMC__num_walkers = walkers
    dtype = get_dtype_np("local_energy")
    driver._MCMC__stored_w_L = weights.astype(dtype)
    driver._MCMC__stored_e_L = real.astype(dtype) + (1j * imag if complex_values else 0)
    driver._MCMC__stored_e_L2 = (real**2 + imag**2).astype(dtype)
    return driver, weights, real, imag


def _rational_reference(weights, real, imag, warmup, bins):
    """Delete raw blocks and use exact pair differences, not production moments."""
    quotient, remainder = divmod(len(weights) - warmup, bins)
    slices = []
    offset = warmup
    for block in range(bins):
        size = quotient + (block < remainder)
        slices.append(range(offset, offset + size))
        offset += size
    energies, variances = [], []
    for omitted in slices:
        for walker in range(weights.shape[1]):
            kept = [
                (int(weights[t, w]), int(real[t, w]), int(imag[t, w]))
                for t in range(warmup, len(weights))
                for w in range(weights.shape[1])
                if not (t in omitted and w == walker)
            ]
            mass = sum(w for w, _, _ in kept)
            energies.append(Fraction(sum(w * x for w, x, _ in kept), mass))
            pair_sum = sum(
                w * v * ((x - u) ** 2 + (y - z) ** 2) for i, (w, x, y) in enumerate(kept) for v, u, z in kept[i + 1 :]
            )
            variances.append(Fraction(pair_sum, mass**2))

    def summarize(replicates):
        count = len(replicates)
        mean = sum(replicates, Fraction(0)) / count
        pair_sum = sum((x - y) ** 2 for i, x in enumerate(replicates) for y in replicates[i + 1 :])
        return float(mean), sqrt(float(Fraction(count - 1, count**2) * pair_sum))

    return (*summarize(energies), *summarize(variances))


def _assert_energy(driver, weights, real, imag, warmup, bins):
    actual = driver.get_E(warmup, bins)
    expected = _rational_reference(weights, real, imag, warmup, bins)
    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    for quantity, actual_value, expected_value in zip(
        ("energy-mean", "energy-error", "variance-mean", "variance-error"), actual, expected, strict=True
    ):
        np.testing.assert_allclose(actual_value, expected_value, atol=atol, rtol=rtol, equal_nan=False, err_msg=quantity)


@pytest.mark.parametrize("walkers", [1, 2])
@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("warmup,bins", [(0, 4), (1, 3)])
def test_energy_variance_and_errors_match_exact_raw_sample_reference(walkers, complex_values, warmup, bins):
    """All four MCMC energy outputs equal an exact-rational delete-block reference.

    Explicit parameter-set input has ten time slices and one or two walkers.
    Integer weights and real/complex energies are unequal in time and walker;
    warmup/block choices 0/4 and 1/3 exercise unequal and equal block lengths.
    No TREXIO, wavefunction, spins, positions, cond(G) screening or sampling.
    Independently delete each raw time-block/walker cell, calculate the weighted
    complex variance from pairwise distances with Fraction arithmetic, then
    calculate jackknife errors from exact pairwise replicate differences.
    Only conversion and the final square root are floating point. Each scalar
    uses the same-mode local_energy strict policy, not the energy's magnitude
    as the error-bar scale. Full and mixed are separate tests. This guarantees
    finite-sample arithmetic, not asymptotic calibration or decorrelation.
    """
    driver, weights, real, imag = _fixed_samples(walkers, complex_values)
    _assert_energy(driver, weights, real, imag, warmup, bins)


@pytest.mark.parametrize("walkers", [1, 2])
@pytest.mark.parametrize("fault", ["energy-error-factor", "variance-error-factor", "complex-variance"])
def test_rational_energy_reference_rejects_production_formula_faults(monkeypatch, walkers, fault):
    """A passing complex baseline becomes red for each injected production formula defect.

    Use the actual get_E source in an isolated function namespace, never edit
    production files. Remove the replicate factor from either reported error,
    or discard the imaginary mean when computing the complex variance. Each
    defect must fail the same four-scalar rational reference at unchanged tol.
    """
    driver, weights, real, imag = _fixed_samples(walkers, True)
    _assert_energy(driver, weights, real, imag, 1, 3)
    replacements = {
        "energy-error-factor": ("np.sqrt((M_total - 1) * E_var)", "np.sqrt(E_var)"),
        "variance-error-factor": ("np.sqrt((M_total - 1) * Var_var)", "np.sqrt(Var_var)"),
        "complex-variance": ("np.abs(E_jackknife_binned_local) ** 2", "np.real(E_jackknife_binned_local) ** 2"),
    }
    source = textwrap.dedent(inspect.getsource(MCMC.get_E))
    old, new = replacements[fault]
    assert source.count(old) == 1
    namespace = dict(MCMC.get_E.__globals__)
    exec(compile(source.replace(old, new), "<fault-injected-MCMC.get_E>", "exec"), namespace)
    monkeypatch.setattr(MCMC, "get_E", namespace["get_E"])
    with pytest.raises(AssertionError):
        _assert_energy(driver, weights, real, imag, 1, 3)
