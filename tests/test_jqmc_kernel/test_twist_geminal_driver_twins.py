"""Production drivers vs their debug twins at k != Gamma for AGP, AGPn and an untied SD.

Before this module every MCMC / GFMC driver cell at a twist used a time-reversal SD
geminal; complex AGP (dense complex lambda), AGPn (real occupations in complex
arithmetic) and ``time_reversal=False`` (independent complex spin channels) were
exercised only by kernel-level tests.  Each node runs the production XLA driver and
its independent eager debug twin from the same seed on the same Hamiltonian.  The
walk is compared first (PRNG keys and acceptance counters exactly, walker positions
at the established fp64 accumulator pin), then the stored complex local energies,
weights and, for MCMC, the stored parameter derivatives.

Fixture: ``H4_pbc_ae_k025_cart.h5`` all-electron + J1/J2 (``_twist_geminal_cases``);
no ECP, no J3 at the twist, GFMC_t with one walker.
Every MCMC row, AGPn included, compares the stored parameter derivatives (the AGPn debug
twin collects its direct natural-orbital blocks).
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

from __future__ import annotations

import os

import jax
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc._setting import ATOL_PYTHON_FP64_MEDIUM, RTOL_PYTHON_FP64_ZERO
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t, _GFMC_n_debug, _GFMC_t_debug
from jqmc.jqmc_mcmc import MCMC, _MCMC_debug
from tests.test_jqmc_kernel._twist_geminal_cases import build_case

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_DRIVER_CASES = ("agp", "agpn", "sd_untied")
_ZONES = ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic")
_SEED = 3446
_STEPS = 4


def _assert_values(debug_value, production_value, label):
    debug_value = np.asarray(debug_value)
    production_value = np.asarray(production_value)
    assert debug_value.shape == production_value.shape, label
    assert np.all(np.isfinite(debug_value)) and np.all(np.isfinite(production_value)), label
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    np.testing.assert_allclose(debug_value, production_value, atol=atol, rtol=rtol, equal_nan=False, err_msg=label)


def _assert_walk(debug_r_up, debug_r_dn, production_r_up, production_r_dn):
    np.testing.assert_allclose(debug_r_up, production_r_up, atol=ATOL_PYTHON_FP64_MEDIUM, rtol=RTOL_PYTHON_FP64_ZERO)
    np.testing.assert_allclose(debug_r_dn, production_r_dn, atol=ATOL_PYTHON_FP64_MEDIUM, rtol=RTOL_PYTHON_FP64_ZERO)


def _assert_complex_e_l(e_l, label):
    e_l = np.asarray(e_l)
    assert np.iscomplexobj(e_l), f"{label}: stored e_L lost its complex dtype ({e_l.dtype})"
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    assert np.max(np.abs(e_l.imag) - (atol + rtol * np.abs(e_l.real))) > 0.0, f"{label}: Im e_L inside the band"


@pytest.mark.light
@pytest.mark.parametrize("case", _DRIVER_CASES)
def test_twist_geminal_mcmc_matches_debug_twin(case):
    """Production XLA MCMC equals _MCMC_debug at k != Gamma for AGP, AGPn and an untied SD.

    H4_pbc_ae_k025_cart.h5 with J1+J2; 2 walkers, seed 3446, 4 measurements of 4 moves, Dt=0.5,
    epsilon_AS=0. PRNG keys and accepted/rejected counters must be identical and final walkers
    equal at ATOL_PYTHON_FP64_MEDIUM; then stored complex e_L (with measurable Im), e_L2 and w_L,
    and every d ln Psi / dc block stored by production, agree strictly per precision mode.
    """
    hamiltonian = build_case(case)
    kwargs = dict(
        hamiltonian_data=hamiltonian,
        num_walkers=2,
        num_mcmc_per_measurement=4,
        mcmc_seed=_SEED,
        Dt=0.5,
        epsilon_AS=0.0,
        comput_log_WF_param_deriv=True,
    )
    assert os.environ.get("JQMC_FLAG_RUST_ON_CPU") == "0", "this twin is the XLA production route"
    debug = _MCMC_debug(**kwargs)
    production = MCMC(**kwargs)
    debug.run(num_mcmc_steps=_STEPS)
    production.run(num_mcmc_steps=_STEPS)

    np.testing.assert_array_equal(debug._MCMC_debug__jax_PRNG_key_list, production._MCMC__jax_PRNG_key_list)
    np.testing.assert_array_equal(debug._MCMC_debug__accepted_moves, production._MCMC__accepted_moves)
    np.testing.assert_array_equal(debug._MCMC_debug__rejected_moves, production._MCMC__rejected_moves)
    _assert_walk(
        debug._MCMC_debug__latest_r_up_carts,
        debug._MCMC_debug__latest_r_dn_carts,
        production._MCMC__latest_r_up_carts,
        production._MCMC__latest_r_dn_carts,
    )
    for name in ("e_L", "e_L2", "w_L"):
        _assert_values(getattr(debug, name), getattr(production, name), f"{case}: MCMC stored {name}")
    _assert_complex_e_l(production.e_L, f"{case}: MCMC")
    debug_grads = debug.dln_Psi_dc
    production_grads = production.dln_Psi_dc
    # The debug twin also stores the geminal AO-basis blocks that production only computes
    # when basis optimization is requested; every production block must be matched.
    assert set(production_grads) <= set(debug_grads), (sorted(debug_grads), sorted(production_grads))
    geminal_keys = set(production_grads) - {"j1_param", "j2_param"}
    assert geminal_keys, f"{case}: no geminal derivative stored"
    if case == "agpn":
        assert {"mo_coefficients_up", "lambda_diag"} <= geminal_keys, sorted(geminal_keys)
    for name in sorted(production_grads):
        _assert_values(debug_grads[name], production_grads[name], f"{case}: MCMC dln_Psi_dc[{name}]")
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize("driver", ["n", "t"])
@pytest.mark.parametrize("case", _DRIVER_CASES)
def test_twist_geminal_gfmc_matches_debug_twin(case, driver):
    """Production XLA GFMC_n / GFMC_t equal their debug twins at k != Gamma for AGP, AGPn and an untied SD.

    H4_pbc_ae_k025_cart.h5 with J1+J2; seed 3446, alat=0.30, random discretized mesh, 4 branching
    steps. GFMC_n: 2 walkers, 4 projections per measurement, E_scf=-2. GFMC_t: 1 walker,
    tau=0.05. PRNG keys must be identical and final walkers equal at ATOL_PYTHON_FP64_MEDIUM; the
    public stored e_L (complex, measurable Im), e_L2 and w_L agree strictly per precision mode.
    """
    hamiltonian = build_case(case)
    common = dict(
        hamiltonian_data=hamiltonian,
        num_walkers=2 if driver == "n" else 1,
        num_gfmc_collect_steps=1,
        mcmc_seed=_SEED,
        alat=0.30,
        random_discretized_mesh=True,
        comput_position_deriv=False,
    )
    assert os.environ.get("JQMC_FLAG_RUST_ON_CPU") == "0", "this twin is the XLA production route"
    if driver == "n":
        debug = _GFMC_n_debug(**common, num_mcmc_per_measurement=4, E_scf=-2.0)
        production = GFMC_n(**common, num_mcmc_per_measurement=4, E_scf=-2.0)
    else:
        debug = _GFMC_t_debug(**common, tau=0.05)
        production = GFMC_t(**common, tau=0.05)
    debug.run(num_mcmc_steps=_STEPS)
    production.run(num_mcmc_steps=_STEPS)

    np.testing.assert_array_equal(debug._jax_PRNG_key_list, production._jax_PRNG_key_list)
    _assert_walk(
        debug._latest_r_up_carts,
        debug._latest_r_dn_carts,
        production._latest_r_up_carts,
        production._latest_r_dn_carts,
    )
    for name in ("e_L", "e_L2", "w_L"):
        _assert_values(getattr(debug, name), getattr(production, name), f"{case}: GFMC_{driver} stored {name}")
    _assert_complex_e_l(production.e_L, f"{case}: GFMC_{driver}")
    jax.clear_caches()
