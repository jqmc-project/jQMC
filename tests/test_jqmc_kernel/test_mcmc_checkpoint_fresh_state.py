"""MCMC saves validate fresh wavefunctions while preserving valid small/tail states."""

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

from jqmc.determinant import compute_geminal_all_elements
from jqmc.jqmc_mcmc import MCMC
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian
from tests.test_jqmc_kernel.test_integration_checkpoint_mcmc import _build_hamiltonian

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


@pytest.mark.parametrize("rust_enabled", [False, True], ids=["xla", "rust"])
@pytest.mark.parametrize("invalid_state", ["zero-determinant", "nonfinite-coordinate"])
def test_checkpoint_save_rejects_fresh_invalid_wavefunction_before_writing(tmp_path, monkeypatch, rust_enabled, invalid_state):
    """Finite old energies do not permit a fresh zero/nonfinite determinant checkpoint."""
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1" if rust_enabled else "0")
    ham = _build_hamiltonian("H2_ae_ccpvdz_cart.h5", "none")
    driver = MCMC(hamiltonian_data=ham, num_walkers=1, mcmc_seed=12345)
    down = np.asarray(driver._MCMC__latest_r_dn_carts).copy()
    down[0, 0] = [-115.0, 40.0, 15.0] if invalid_state == "zero-determinant" else [np.nan, 0.0, 0.0]
    driver._MCMC__latest_r_dn_carts = down
    driver._MCMC__stored_e_L = np.array([[-1.0]])
    driver._MCMC__stored_e_L2 = np.array([[1.0]])
    driver._MCMC__stored_w_L = np.array([[1.0]])
    target = tmp_path / "rank.h5"
    field = "ln_abs_det" if invalid_state == "zero-determinant" else "positions_finite"
    with pytest.raises(FloatingPointError, match=rf"(?s)MCMC.*checkpoint save.*walker=0.*field={field}"):
        driver.save_to_hdf5(str(target))
    assert not target.exists()


@pytest.mark.parametrize("rust_enabled", [False, True], ids=["xla", "rust"])
@pytest.mark.parametrize("valid_state", ["tiny-determinant", "periodic-unwrapped", "complex-periodic-unwrapped", "near-node"])
def test_checkpoint_save_preserves_valid_finite_tail_and_small_determinants(tmp_path, monkeypatch, rust_enabled, valid_state):
    """Validation uses finite log determinants, with no radius or small-value cutoff."""
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1" if rust_enabled else "0")
    if valid_state == "tiny-determinant":
        ham = _build_hamiltonian("Cl2_ecp_ccpvtz_cart.h5", "none")
        gem = ham.wavefunction_data.geminal_data
        gem = gem.replace(
            orb_data_up_spin=gem.orb_data_up_spin.replace(
                mo_coefficients=np.asarray(gem.orb_data_up_spin.mo_coefficients) * 1e-25
            ),
            orb_data_dn_spin=gem.orb_data_dn_spin.replace(
                mo_coefficients=np.asarray(gem.orb_data_dn_spin.mo_coefficients) * 1e-25
            ),
        )
        ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=gem))
    else:
        ham = build_canonical_hamiltonian("WF-PATTERN-016" if valid_state == "complex-periodic-unwrapped" else "WF-PATTERN-015")
    driver = MCMC(hamiltonian_data=ham, num_walkers=1, mcmc_seed=12345)
    up = np.asarray(driver._MCMC__latest_r_up_carts).copy()
    down = np.asarray(driver._MCMC__latest_r_dn_carts).copy()
    if valid_state.endswith("periodic-unwrapped"):
        translation = np.array([90.0, -60.0, 40.0]) @ np.asarray(ham.structure_data.cell)
        up += translation
        down -= translation
    elif valid_state == "near-node":
        up[0, 1] = up[0, 0] + np.array([1e-5, 0.0, 0.0])
    driver._MCMC__latest_r_up_carts = up
    driver._MCMC__latest_r_dn_carts = down
    matrix = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, up[0], down[0]))
    sign, logabsdet = np.linalg.slogdet(matrix)
    assert np.all(np.isfinite(matrix)) and sign != 0 and np.isfinite(logabsdet)
    if valid_state == "tiny-determinant":
        assert np.linalg.det(matrix) == 0, "Completed determinant underflows but finite slogdet remains meaningful"
    target = tmp_path / "rank.h5"
    driver.save_to_hdf5(str(target))
    assert target.exists()
    np.testing.assert_array_equal(np.asarray(driver._MCMC__latest_r_up_carts), up)
    np.testing.assert_array_equal(np.asarray(driver._MCMC__latest_r_dn_carts), down)
