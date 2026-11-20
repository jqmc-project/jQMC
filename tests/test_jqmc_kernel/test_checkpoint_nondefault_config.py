"""Constructor-valued checkpoint restoration and non-default continuation (#364)."""

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

from jqmc._checkpoint import merge_rank_checkpoints
from jqmc._precision import get_tolerance_min
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.jqmc_mcmc import MCMC
from tests.test_jqmc_kernel.test_integration_checkpoint_mcmc import _build_hamiltonian

pytestmark = [pytest.mark.with_chdir]

_DRIVER_TYPES = {"MCMC": MCMC, "GFMC_t": GFMC_t, "GFMC_n": GFMC_n}


def _nondefault_constructor_settings(driver_kind):
    """Independent explicit constructor inputs, not values recovered from disk."""
    common = dict(
        mcmc_seed=24680,
        num_walkers=2,
        comput_position_deriv=True,
        comput_stress=True,
        random_discretized_mesh=False,
        mpc_cutoff=17.0,
    )
    if driver_kind == "MCMC":
        return dict(
            common,
            Dt=1.1,
            epsilon_AS=0.03,
            num_mcmc_per_measurement=3,
            use_swct=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=True,
        )
    shared = dict(common, alat=0.25, num_gfmc_collect_steps=7, non_local_move="dltmove", epsilon_PW=0.002, use_swct=True)
    return dict(shared, tau=0.07) if driver_kind == "GFMC_t" else dict(shared, E_scf=-1.3, num_mcmc_per_measurement=3)


def _saved_driver(driver, ham, root):
    """Use the real rank writer and merge, including the embedded Hamiltonian."""
    root.mkdir(exist_ok=True)
    rank_file = root / "rank0.h5"
    driver.save_to_hdf5(str(rank_file))
    merged = root / "merged.h5"
    merge_rank_checkpoints(
        output_path=str(merged),
        mpi_size=1,
        driver_type=type(driver).__name__,
        hamiltonian_data=ham,
        tmp_pattern=str(root / "rank{rank}.h5"),
    )
    return type(driver).load_from_hdf5(str(merged), rank=0)


def _config_attribute(driver, name):
    prefix = "_MCMC__" if isinstance(driver, MCMC) else "_"
    return getattr(driver, prefix + name)


@pytest.mark.heavy
@pytest.mark.parametrize("case", ["mcmc_dt", "mcmc_swct", "gfmc_t_dltmove", "gfmc_n_dltmove"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_nondefault_checkpoint_continuation_matches_uninterrupted(case, tmp_path):
    """Three+three real sampler measurements preserve settings, states, RNG and histories.

    N2 ccECP has a live non-local term; J1+J2 makes dltmove differ from tmove.
    Constructor inputs and the uninterrupted path are independent of both the
    saved config dictionary and the loader. No walker/error screening is used.
    Each mode retains the established strict same-mode restart budget.
    """
    ham = _build_hamiltonian("N2_ecp_ccpvtz_cart.h5", "1b+2b")
    common = dict(hamiltonian_data=ham, mcmc_seed=24680, num_walkers=2, random_discretized_mesh=False)
    if case.startswith("mcmc"):
        cls = MCMC
        settings = dict(
            common,
            Dt=1.1,
            epsilon_AS=0.03,
            num_mcmc_per_measurement=3,
            comput_position_deriv=case == "mcmc_swct",
            use_swct=False,
        )
        prefix = "_MCMC__"
    else:
        cls = GFMC_t if case == "gfmc_t_dltmove" else GFMC_n
        settings = dict(common, alat=0.25, non_local_move="dltmove", num_gfmc_collect_steps=5)
        settings.update(tau=0.07) if cls is GFMC_t else settings.update(E_scf=-1.3, num_mcmc_per_measurement=2)
        prefix = "_"
    uninterrupted, first = cls(**settings), cls(**settings)
    uninterrupted.run(num_mcmc_steps=6)
    first.run(num_mcmc_steps=3)
    resumed = _saved_driver(first, ham, tmp_path / "checkpoint")
    for key, expected in settings.items():
        if key != "hamiltonian_data":
            assert _config_attribute(resumed, key) == expected, (case, key)
    resumed.run(num_mcmc_steps=3)
    raw_counter = "_MCMC__mcmc_counter" if cls is MCMC else "_mcmc_counter"
    assert getattr(resumed, raw_counter) == getattr(uninterrupted, raw_counter) == 6
    expected_public_counter = 6 if cls is MCMC else 6 - settings["num_gfmc_collect_steps"]
    assert resumed.mcmc_counter == uninterrupted.mcmc_counter == expected_public_counter
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy"), "strict")
    fields = ["stored_e_L", "stored_e_L2", "stored_w_L"] if cls is MCMC else ["stored_e_L", "stored_w_L"]
    if case == "mcmc_swct":
        fields += ["stored_force_HF", "stored_force_PP", "stored_E_L_force_PP"]
    for name in fields:
        actual, expected = np.asarray(getattr(resumed, prefix + name)), np.asarray(getattr(uninterrupted, prefix + name))
        assert actual.size and expected.size and np.isfinite(actual).all() and np.isfinite(expected).all()
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=f"{case}: {name}")
    for name in ("latest_r_up_carts", "latest_r_dn_carts", "jax_PRNG_key_list"):
        np.testing.assert_array_equal(getattr(resumed, prefix + name), getattr(uninterrupted, prefix + name))
    if cls is not MCMC:
        actual_rng, expected_rng = resumed._branching_rng.get_state(), uninterrupted._branching_rng.get_state()
        assert actual_rng[0] == expected_rng[0] and actual_rng[2:] == expected_rng[2:]
        np.testing.assert_array_equal(actual_rng[1], expected_rng[1])
