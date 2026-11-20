"""Native tiled SD, independent supercell J3 optimization and checkpoint continuation."""

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
from jqmc._setting import get_tol_hdf5_roundtrip
from jqmc.jqmc_mcmc import MCMC
from tests.test_jqmc_kernel.test_orbital_tiling_hdf5 import _hamiltonian
from tests.test_jqmc_kernel.test_orbital_tiling_j3 import _supercell_j3
from tests.test_rs_kernel.conftest import native_ctx_spy

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]


@pytest.mark.parametrize("backend", ["pw", "bspline", "pgto_cart", "pgto_sphe", "composite_cart", "composite_sphe"])
@pytest.mark.parametrize("restricted", [True, False])
def test_native_tiled_j3_optimization_force_stress_restart(tmp_path, monkeypatch, backend, restricted):
    """Native J3-only update and resumed RNG/energy/force/stress agree with uninterrupted continuation."""
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    ham = _hamiltonian(backend, restricted)
    ham = ham.replace(
        wavefunction_data=ham.wavefunction_data.replace(
            jastrow_data=ham.wavefunction_data.jastrow_data.replace(jastrow_three_body_data=_supercell_j3(ham.structure_data))
        )
    )
    before = np.asarray(ham.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix)
    driver = MCMC(
        hamiltonian_data=ham,
        num_walkers=2,
        mcmc_seed=812,
        num_mcmc_per_measurement=1,
        comput_position_deriv=True,
        comput_stress=True,
        comput_log_WF_param_deriv=True,
    )
    with native_ctx_spy() as calls:
        driver.run_optimize(
            num_mcmc_steps=4,
            num_opt_steps=1,
            num_mcmc_bin_blocks=2,
            dump_wavefunction=False,
            opt_J1_param=False,
            opt_J2_param=False,
            opt_J3_param=True,
            opt_geminal=False,
            optimizer_kwargs={"method": "sr", "delta": 1e-3, "epsilon": 1e-3},
        )
        driver.run(num_mcmc_steps=2)
    after = np.asarray(driver.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix)
    assert np.isfinite(after).all() and np.any(after != before)
    for spin in ("up", "dn"):
        old = getattr(ham.wavefunction_data.geminal_data, "orb_data_" + spin + "_spin")
        new = getattr(driver.hamiltonian_data.wavefunction_data.geminal_data, "orb_data_" + spin + "_spin")
        for a, b in zip(old.blocks, new.blocks, strict=True):
            if a.coefficients is None:
                assert b.coefficients is None
            else:
                np.testing.assert_array_equal(a.coefficients, b.coefficients)
    for method in (
        "mcmc_sweep_complex_batch",
        "grad_e_L_pbc_complex_batch",
        "grad_ln_psi_pbc_complex_batch",
        "grad_e_L_eps_pbc_complex_batch",
        "grad_ln_psi_eps_pbc_complex_batch",
    ):
        assert calls[method] > 0, dict(calls)
    pattern = str(tmp_path / "rank-{rank}.h5")
    driver.save_to_hdf5(pattern.format(rank=0))
    merged = str(tmp_path / "restart.h5")
    merge_rank_checkpoints(
        merged, mpi_size=1, driver_type="MCMC", hamiltonian_data=driver.hamiltonian_data, tmp_pattern=pattern
    )
    restored = MCMC.load_from_hdf5(merged)
    with native_ctx_spy() as resumed_calls:
        driver.run(num_mcmc_steps=2)
        restored.run(num_mcmc_steps=2)
    assert resumed_calls["mcmc_sweep_complex_batch"] > 0
    atol, rtol = get_tol_hdf5_roundtrip("strict")
    for name in (
        "latest_r_up_carts",
        "latest_r_dn_carts",
        "stored_e_L",
        "stored_w_L",
        "stored_force_HF",
        "stored_force_PP",
        "stored_stress_HF",
        "stored_stress_PP",
    ):
        actual = np.asarray(getattr(restored, "_MCMC__" + name))
        assert actual.size and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, getattr(driver, "_MCMC__" + name), atol=atol, rtol=rtol)
    np.testing.assert_array_equal(restored._MCMC__jax_PRNG_key_list, driver._MCMC__jax_PRNG_key_list)
