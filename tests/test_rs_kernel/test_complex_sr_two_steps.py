"""Complex native SR continuation at the real consumed optimization batches."""

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

from tests.test_rs_kernel import test_driver_loops_rust as driver

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_complex_native_sr_two_steps_equal_xla():
    """Two native complex-Li SR updates match same-mode XLA batches and applied parameters.

    Li k=0.25 ccECP with exp J1/J2 uses four walkers, seed 20260927,
    25 measurements per step, five bins, delta 0.05 and epsilon 1e-3.
    Only the proposal-normal seam supplies identical random inputs; sampling,
    native parameter-gradient assembly and both optimizer updates run normally.
    Each consumed weight, complex energy, log-derivative coordinate and solved
    update is checked with the existing backend medium policy. The final walkers
    and keys are exact. J1/J2 log derivatives remain physically real even though
    the wavefunction and energy are complex; preserving their real/imaginary
    payload slots is checked, without claiming an imaginary Jastrow derivative.
    """
    ham = driver._OPT_CASES["li_twist_exp_j12"]()
    rust = driver._optimize_one_step(ham, rust_on=True, num_opt_steps=2)
    xla = driver._optimize_one_step(ham, rust_on=False, num_opt_steps=2)
    driver.assert_native_ran(rust["counts"], "native two-step SR")
    driver.assert_no_native_ran(xla["counts"], "XLA two-step SR")
    assert rust["native_param_grads"] >= 2 * driver._OPT_STEPS
    assert xla["native_param_grads"] == 0
    assert len(rust["steps"]) == len(xla["steps"]) == 2
    assert rust["seen"]["sweep"] >= 2 * driver._OPT_STEPS
    for index, (actual, reference) in enumerate(zip(rust["steps"], xla["steps"], strict=True)):
        for field in ("weights", "local_energies"):
            driver._parity(getattr(actual["batch"], field), getattr(reference["batch"], field), f"step {index} {field}")
        actual_o = driver._host_coordinates(actual["coordinate_map"], actual["batch"], actual["batch"].log_derivative_sources)
        reference_o = driver._host_coordinates(
            reference["coordinate_map"], reference["batch"], reference["batch"].log_derivative_sources
        )
        driver._parity(actual_o, reference_o, f"step {index} log derivatives", zones=driver._DERIV_ZONES)
        driver._parity(actual["target"].q, reference["target"].q, f"step {index} solved SR update", zones=driver._DERIV_ZONES)
        driver._parity(actual["step"].q, reference["step"].q, f"step {index} applied update", zones=driver._DERIV_ZONES)
        assert np.any(np.asarray(reference["step"].q) != 0), f"step {index} update is vacuous"
    driver._assert_optimize_parity(rust, xla, "two-step complex SR")
