"""XLA sampling and checkpoint continuation for frozen primitive-block SDs."""

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
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.jqmc_mcmc import MCMC
from tests.test_jqmc_kernel.test_orbital_tiling_hdf5 import _hamiltonian

pytestmark = [pytest.mark.with_chdir, pytest.mark.heavy]


@pytest.mark.parametrize(
    ("driver_name", "backend"),
    [
        ("MCMC", "pw"),
        ("MCMC", "bspline"),
        ("MCMC", "pgto_cart"),
        ("MCMC", "pgto_sphe"),
        ("MCMC", "composite_cart"),
        ("MCMC", "composite_sphe"),
        ("GFMC_n", "pw"),
        ("GFMC_t", "pw"),
    ],
)
def test_tiled_xla_sampling_checkpoint_continuation(driver_name, backend, tmp_path, monkeypatch):
    """Complex energy, primitive aliases and RNG survive a real single-rank restart."""
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    monkeypatch.chdir(tmp_path)
    ham = _hamiltonian(backend, True)
    if backend.startswith("composite"):
        from tests.test_jqmc_kernel.test_orbital_tiling_j3 import _supercell_j3

        ham = ham.replace(
            wavefunction_data=ham.wavefunction_data.replace(
                jastrow_data=ham.wavefunction_data.jastrow_data.replace(
                    jastrow_three_body_data=_supercell_j3(ham.structure_data)
                )
            )
        )
    cls = {"MCMC": MCMC, "GFMC_n": GFMC_n, "GFMC_t": GFMC_t}[driver_name]
    options = {"num_mcmc_per_measurement": 1} if driver_name != "GFMC_t" else {"tau": 0.01}
    if driver_name != "MCMC":
        options["num_gfmc_collect_steps"] = 0
    if backend.startswith("composite"):
        options.update(comput_position_deriv=True, comput_stress=True)
    driver = cls(hamiltonian_data=ham, num_walkers=2, mcmc_seed=812, **options)
    assert driver._wf_is_complex
    driver.run(num_mcmc_steps=2)
    pattern = str(tmp_path / "rank-{rank}.h5")
    driver.save_to_hdf5(pattern.format(rank=0))
    merged = str(tmp_path / "restart.h5")
    merge_rank_checkpoints(merged, mpi_size=1, driver_type=driver_name, hamiltonian_data=ham, tmp_pattern=pattern)
    restored = cls.load_from_hdf5(merged)
    assert restored._wf_is_complex
    sd = restored.hamiltonian_data.wavefunction_data.geminal_data
    for block in sd.orb_data_dn_spin.blocks:
        assert any(block.backend is up.backend and block.coefficients is up.coefficients for up in sd.orb_data_up_spin.blocks)
    prefix = "_MCMC__" if driver_name == "MCMC" else "_"
    for name in ("latest_r_up_carts", "latest_r_dn_carts", "jax_PRNG_key_list", "stored_e_L", "stored_e_L2", "stored_w_L"):
        before = np.asarray(getattr(driver, prefix + name))
        after = np.asarray(getattr(restored, prefix + name))
        assert before.size and np.isfinite(before).all()
        np.testing.assert_array_equal(after, before)
        if name == "stored_e_L":
            assert before.dtype == after.dtype == np.complex128
    # Compare continuation of the saved streams, not a separately seeded run.
    driver.run(num_mcmc_steps=2)
    restored.run(num_mcmc_steps=2)
    atol, rtol = get_tol_hdf5_roundtrip("strict")
    for name in ("latest_r_up_carts", "latest_r_dn_carts", "stored_e_L", "stored_w_L"):
        np.testing.assert_allclose(getattr(restored, prefix + name), getattr(driver, prefix + name), atol=atol, rtol=rtol)
    np.testing.assert_array_equal(
        getattr(restored, prefix + "jax_PRNG_key_list"), getattr(driver, prefix + "jax_PRNG_key_list")
    )
    if backend.startswith("composite"):
        for name in ("stored_force_HF", "stored_force_PP", "stored_stress_HF", "stored_stress_PP"):
            actual = np.asarray(getattr(restored, prefix + name))
            assert actual.size and np.isfinite(actual).all()
            np.testing.assert_allclose(actual, getattr(driver, prefix + name), atol=atol, rtol=rtol)
    assert driver.mcmc_counter == restored.mcmc_counter == 4
