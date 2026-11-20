"""Accepted mixed AO bases keep negligible zero-radial tails finite."""

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
import pytest
import jqmc_rs

from jqmc._precision import get_tolerance, is_mixed_precision_enabled, precision_policy_snapshot
from jqmc.atomic_orbital import AOEvalRangeError, check_ao_eval_value_range, compute_AOs
from jqmc.coulomb_potential import Coulomb_potential_data
from jqmc.jqmc_mcmc import MCMC
from tests._ao_tail_reference import gaussian_tail_case
from tests.test_rs_kernel.test_ao_eval_range_precondition import _x82_hamiltonian

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


@pytest.mark.parametrize("family", ["cart_l60_a10", "sphe_l6_a1"])
@pytest.mark.parametrize("backend", ["XLA", "Rust"])
def test_accepted_ao_zero_radial_tail_matches_closed_form(family, backend):
    """An accepted basis returns finite negligible values after its radial factor vanishes."""
    cart = family == "cart_l60_a10"
    aos, points, expected = gaussian_tail_case(family)
    check_ao_eval_value_range(aos)
    atol, rtol = get_tolerance("ao_eval", "strict")
    assert np.all(np.abs(expected[:, 1:]) < atol)
    assert expected[0, 0] > atol
    with jqmc_rs.NativeConstructionPrecision(precision_policy_snapshot()):
        actual = np.asarray(compute_AOs(aos, points) if backend == "XLA" else jqmc_rs.compute_AOs(aos, points))
    assert np.all(np.isfinite(actual)), backend
    if not cart:
        # The shared builder returns the complete l=6 shell. Every component
        # must be finite; the independent z-axis formula below is for m=0.
        actual = actual[np.asarray(aos.magnetic_quantum_numbers) == 0]
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=backend)


@pytest.mark.parametrize("backend", ["XLA", "Rust"])
def test_original_x82_driver_is_correct_in_full_or_rejected_in_mixed(backend, monkeypatch):
    """The original N x82/seed34 driver stays covered after its full residual recovered."""
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1" if backend == "Rust" else "0")
    ham = _x82_hamiltonian()
    structure = ham.structure_data.replace(atomic_numbers=(7,), element_symbols=("N",), atomic_labels=("N1",))
    wf, gem = ham.wavefunction_data, ham.wavefunction_data.geminal_data
    aos = gem.orb_data_up_spin.aos_data.replace(structure_data=structure)
    gem = gem.replace(
        orb_data_up_spin=gem.orb_data_up_spin.replace(aos_data=aos),
        orb_data_dn_spin=gem.orb_data_dn_spin.replace(aos_data=aos),
    )
    ham = ham.replace(
        structure_data=structure,
        coulomb_potential_data=Coulomb_potential_data(structure_data=structure),
        wavefunction_data=wf.replace(geminal_data=gem),
    )
    kwargs = dict(
        hamiltonian_data=ham,
        num_walkers=4,
        mcmc_seed=34,
        num_mcmc_per_measurement=2,
        Dt=0.3,
        epsilon_AS=0.0,
        random_discretized_mesh=False,
    )
    if is_mixed_precision_enabled():
        with pytest.raises(AOEvalRangeError, match="full precision"):
            MCMC(**kwargs)
        return
    driver = MCMC(**kwargs)
    expected = []
    for _ in range(3):
        driver.run(num_mcmc_steps=1)
        positions = np.asarray(driver._MCMC__latest_r_up_carts)[:, 0]
        radius = np.linalg.norm(positions, axis=1)
        r2 = np.sum(positions**2, axis=1)
        # For psi = N x^l exp(-a r^2), lap(psi)/psi is
        # l(l-1)/x^2 - 2a(2l+3) + 4a^2 r^2; here l=82,a=1,Z=7.
        expected.append(-0.5 * (82 * 81 / positions[:, 0] ** 2 - 2 * (2 * 82 + 3) + 4 * r2) - 7 / radius)
    actual = np.asarray(driver.e_L)
    assert actual.shape == (3, 4) and np.all(np.isfinite(actual))
    assert (getattr(driver, "_MCMC__rs_run_ctx", None) is not None) == (backend == "Rust")
    atol, rtol = get_tolerance("local_energy", "strict")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
