"""Per-twist MPC estimates of the twist drivers against standalone samplers.

``run_twist_average_from_hamiltonians`` and ``run_twist_average_gfmc`` sample
each twist with an ordinary ``MCMC`` / ``GFMC_t`` chain and publish that chain's
``get_E``, ``get_dE_mpc`` and ``get_E_mpc`` (means and jackknife errors) as the
per-twist inputs of the MPC Brillouin-zone aggregate.  The reference here is a
standalone sampler of the same class, built with the documented per-twist seed
(``_twist_sampler_seed``; the bare seed at ``N_twist = 1``) and the same sampler
keyword arguments, whose three estimators are evaluated with the warmup and
bin-block counts requested from the driver.

The warmup and bin-block counts are deliberately different from every getter
default (``get_E`` / ``get_E_mpc``: 50 / 10, MCMC ``get_dE_mpc``: 0 / 5,
GFMC ``get_dE_mpc``: 50 / 10) and from each other, so a driver that drops,
replaces or swaps either argument of any of the three per-twist estimator
calls publishes different numbers.  The test also checks that the fixture is
sensitive to a dropped warmup and to swapped arguments.
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

from pathlib import Path

import numpy as np
import pytest

from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_ZERO
from jqmc._twist_average import _twist_sampler_seed, run_twist_average_from_hamiltonians, run_twist_average_gfmc
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.jqmc_gfmc import GFMC_t
from jqmc.jqmc_mcmc import MCMC
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.with_chdir]

_LI_TWIST = Path(__file__).resolve().parent.parent / "trexio_example_files" / "lithium_twist_222.h5"

_SEED = 12345
_NUM_WALKERS = 2
_NUM_STEPS = 24
# Different from every getter default and from each other (see module docstring).
_NUM_WARMUP = 8
_NUM_BINS = 4
_MPC = {"mpc": True, "mpc_cutoff": 4.0}
_ROUTES = {
    "mcmc": (MCMC, {"num_mcmc_per_measurement": 1, **_MPC}),
    "gfmc_tau": (GFMC_t, {"tau": 0.1, "alat": 0.5, **_MPC}),
}


def _li_twist0_hamiltonian():
    structure, _, _, _, geminal, coulomb = read_trexio_file(str(_LI_TWIST), store_tuple=True, twist_ik=0)
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, structure_data=structure)
    ham = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data(jastrow_two_body_data=j2)),
    )
    ham.sanity_check()
    return ham


def _estimates(sampler, num_mcmc_warmup_steps, num_mcmc_bin_blocks):
    """(E, sigma_E, dE_mpc, sigma_dE_mpc, E_mpc, sigma_E_mpc) of one chain."""
    args = dict(num_mcmc_warmup_steps=num_mcmc_warmup_steps, num_mcmc_bin_blocks=num_mcmc_bin_blocks)
    E, E_std, _, _ = sampler.get_E(**args)
    dE, dE_std = sampler.get_dE_mpc(**args)
    E_mpc, E_mpc_std = sampler.get_E_mpc(**args)
    return np.array([E, E_std, dE, dE_std, E_mpc, E_mpc_std], dtype=np.float64)


@pytest.mark.heavy
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_driver_per_twist_mpc_estimates_match_standalone_sampler(route):
    """Driver per-twist E, dE_mpc and E_mpc (with errors) equal a standalone chain's."""
    sampler_cls, sampler_kwargs = _ROUTES[route]
    ham = _li_twist0_hamiltonian()

    # ---- standalone side: the documented per-twist chain ----
    ref = sampler_cls(
        hamiltonian_data=ham,
        mcmc_seed=_twist_sampler_seed(_SEED, 0, 1),
        num_walkers=_NUM_WALKERS,
        comput_position_deriv=False,
        comput_stress=False,
        **sampler_kwargs,
    )
    ref.run(num_mcmc_steps=_NUM_STEPS)
    expected = _estimates(ref, _NUM_WARMUP, _NUM_BINS)

    # The fixture distinguishes the requested warmup and bin blocks from a
    # dropped warmup and from swapped arguments in every MPC estimator.
    for warmup, bins in ((0, _NUM_BINS), (_NUM_BINS, _NUM_WARMUP)):
        perturbed = _estimates(ref, warmup, bins)
        assert np.all(np.abs(perturbed[2:] - expected[2:]) > ATOL_PYTHON_FP64_LOOSE), (warmup, bins, perturbed, expected)

    # ---- driver side: same seed, sampler kwargs, steps, warmup and bins ----
    common = dict(
        hamiltonian_list=[ham],
        weights=[1.0],
        mcmc_seed=_SEED,
        num_walkers=_NUM_WALKERS,
        num_mcmc_steps=_NUM_STEPS,
        num_mcmc_warmup_steps=_NUM_WARMUP,
        num_mcmc_bin_blocks=_NUM_BINS,
    )
    if route == "mcmc":
        result = run_twist_average_from_hamiltonians(**common, mcmc_kwargs=dict(sampler_kwargs))
    else:
        result = run_twist_average_gfmc(**common, gfmc_kind="tau", gfmc_kwargs=dict(sampler_kwargs))

    assert result.mpc is not None
    actual = np.array(
        [
            result.per_twist_E[0],
            result.per_twist_E_std[0],
            result.mpc.per_twist_correction[0],
            result.mpc.per_twist_correction_std[0],
            result.mpc.per_twist_energy[0],
            result.mpc.per_twist_energy_std[0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(actual, expected, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
