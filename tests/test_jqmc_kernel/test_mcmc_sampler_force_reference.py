"""Same-mode independent references for actual MCMC force measurement buffers."""

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

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc.jqmc_mcmc import MCMC
from tests._gfmc_sampler_force_oracle import primal_force_reference
from tests.test_jqmc_kernel.test_gfmc_force_analytic import _build_ham
from tests.test_rs_kernel.conftest import assert_native_ran, assert_no_native_ran, native_ctx_spy
from tests.test_rs_kernel.test_gfmc_projection import _backend_env


pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]


@pytest.mark.parametrize("backend", ["xla", "rust"])
@pytest.mark.parametrize("use_swct", [False, True])
def test_mcmc_sampler_force_buffers_match_primal_derivatives(backend, use_swct):
    """Every measured MCMC force component equals AD of the same sampled primal input.

    Committed H2_ae_ccpvdz_cart.h5: molecular AE Cartesian-GTO SD with one
    up/down electron, no Jastrow. Three walkers start from seed 34 and take
    two moves per measurement at Dt=0.7, AS=0, with identity quadrature mesh.
    Two successive one-measurement production runs retain their actual sampled
    positions without conditioning rejection. Each walker energy, HF, Pulay,
    and energy-times-Pulay buffer is checked separately against primal energy
    and log-amplitude derivatives. SWCT on/off are both required; the warped
    reference differentiates the explicit map and Jacobian, never production
    force outputs. XLA and Rust each compare within the active precision mode.
    Use the same strict/full and medium/mixed zone policy as the GFMC sampler
    force reference. No statistical acceptance or SWCT/no-SWCT mean equality
    is claimed; ensemble/accessor reduction has a separate fixed-data contract.
    """
    ham = _build_ham("H2_ae_ccpvdz_cart.h5")
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap", "det_eval", "wf_eval", "wf_kinetic", "coulomb", "swct"), level)
    with _backend_env("fast" if backend == "rust" else "jax"):
        driver = MCMC(
            hamiltonian_data=ham,
            num_walkers=3,
            mcmc_seed=34,
            num_mcmc_per_measurement=2,
            Dt=0.7,
            epsilon_AS=0.0,
            random_discretized_mesh=False,
            comput_position_deriv=True,
            use_swct=use_swct,
        )
        with native_ctx_spy() as counts:
            for step in range(2):
                driver.run(num_mcmc_steps=1)
                r_up_carts = np.asarray(driver._MCMC__latest_r_up_carts)
                r_dn_carts = np.asarray(driver._MCMC__latest_r_dn_carts)
                energy, hf, pp, _ = (
                    np.asarray(value) for value in primal_force_reference(ham, r_up_carts, r_dn_carts, use_swct=use_swct)
                )
                assert r_up_carts.shape == r_dn_carts.shape == (3, 1, 3)
                for field, reference in (
                    ("e_L", energy),
                    ("force_HF", hf),
                    ("force_PP", pp),
                    ("E_L_force_PP", energy[:, None, None] * pp),
                ):
                    actual = np.asarray(getattr(driver, "_MCMC__stored_" + field))[step]
                    label = f"{step}/{field}"
                    assert actual.shape == reference.shape
                    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
                    assert np.any(np.abs(reference) > atol + rtol * np.abs(reference)), label
                    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=label)
    if backend == "rust":
        assert_native_ran(counts, "MCMC sampler force reference")
    else:
        assert_no_native_ran(counts, "MCMC sampler force reference")
