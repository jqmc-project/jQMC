"""Exact ordinary-supercell J3 optimizer witness from issue #405.

The ordinary PW determinant and the independent ao-small J3 are serialized in
an attributed input fixture. The scientific oracle is physical parameter
identity, positive finite Gaussian exponents, and exact within-mode persistence
and continuation; no saved optimizer output supplies an expected value.
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

from pathlib import Path

import numpy as np
import pytest

from jqmc._checkpoint import merge_rank_checkpoints
from jqmc.atomic_orbital import compute_AOs
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import compute_Jastrow_three_body
from jqmc.jqmc_mcmc import MCMC

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]

_FIXTURE = Path(__file__).resolve().parents[1] / "jqmc_hamiltonian_data" / "periodic_j3_basis_audit_initial.h5"
_R_UP = np.asarray([[0.31, 0.43, 0.57], [1.12, -0.62, 0.91]])
_R_DN = _R_UP + np.asarray([0.13, 0.21, -0.17])


def _j3(hamiltonian):
    return hamiltonian.wavefunction_data.jastrow_data.jastrow_three_body_data


def _assert_same_basis(left, right):
    """Parameter order and image expansion are continuation state as well as values."""
    for name in ("exponents", "coefficients"):
        np.testing.assert_array_equal(getattr(left.radial, name), getattr(right.radial, name), strict=True)
    for name in (
        "ao_prim_to_shell_prim",
        "_image_indices",
        "_ragged_kernel_to_home",
        "_ragged_kernel_atom",
        "_ragged_buckets",
        "_ragged_fold_perm",
    ):
        assert getattr(left, name) == getattr(right, name), name
    for name in ("shell_prim_to_shell", "nucleus_index", "angular_momentum"):
        assert getattr(left.radial, name) == getattr(right.radial, name), name
    np.testing.assert_array_equal(compute_AOs(left, _R_UP), compute_AOs(right, _R_UP), strict=True)
    np.testing.assert_array_equal(compute_AOs(left, _R_DN), compute_AOs(right, _R_DN), strict=True)


def _continuation_state(driver):
    return {
        name: np.asarray(getattr(driver, f"_MCMC__{name}"))
        for name in ("latest_r_up_carts", "latest_r_dn_carts", "stored_e_L", "stored_w_L", "jax_PRNG_key_list")
    }


@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="seeded-random",
    reason=(
        "Unscreened actual MCMC seed718, two walkers and four samplingsteps; "
        "independent fixed off-nucleus J3 probes use ru=[[.31,.43,.57],[1.12,-.62,.91]], rd=ru+[.13,.21,-.17]."
    ),
)
def test_audit_ordinary_supercell_j3_update_and_restart(tmp_path):
    """An accepted real SR step preserves physical J3 basis state through save/restart.

    The fixture's 24 Cartesian ao-small orbitals belong to four physical Li
    sites, with three uncontracted s shells and one p shell per atom. Thus it
    has 16 physical shell primitives; lattice images/angular components are
    shared uses, not independent parameters. The exact nonzero J3 matrix is
    default_rng(41).normal(scale=.02), symmetrized only in its square block.
    The original seed718 trajectory accepted 129 nonpositive image exponents
    and a NaN J3, then saving/reloading selected zero-image representatives and
    silently changed the state. This test exercises the public optimizer and
    real HDF5/checkpoint consumers, rather than merely a checked setter.
    """
    initial = Hamiltonian_data.load_from_hdf5(str(_FIXTURE))
    initial_j3 = _j3(initial)
    matrix = np.random.default_rng(41).normal(scale=0.02, size=initial_j3.j_matrix.shape)
    matrix[:, :-1] = (matrix[:, :-1] + matrix[:, :-1].T) / 2
    np.testing.assert_array_equal(initial_j3.j_matrix, matrix)
    before = float(compute_Jastrow_three_body(initial_j3, _R_UP, _R_DN))
    assert np.isfinite(before) and before != 0
    driver = MCMC(
        hamiltonian_data=initial,
        num_walkers=2,
        mcmc_seed=718,
        num_mcmc_per_measurement=1,
        comput_log_WF_param_deriv=True,
    )
    driver.run_optimize(
        num_mcmc_steps=4,
        num_opt_steps=1,
        num_mcmc_bin_blocks=2,
        dump_wavefunction=False,
        opt_J3_basis_exp=True,
        opt_J1_param=False,
        opt_J2_param=False,
        opt_J3_param=False,
        opt_geminal=False,
        optimizer_kwargs={"method": "sr", "delta": 1e-3, "epsilon": 1e-3},
    )
    assert driver.i_opt == 1, "The original trajectory must accept a finite physical update."
    updated = driver.hamiltonian_data
    updated.sanity_check()
    current_j3 = _j3(updated)
    exponents = np.asarray(current_j3.orb_data.radial.exponents)
    assert np.isfinite(exponents).all() and (exponents > 0).all()
    after = float(compute_Jastrow_three_body(current_j3, _R_UP, _R_DN))
    assert np.isfinite(after) and after != before
    assert len(exponents) == 16
    np.testing.assert_array_equal(current_j3.j_matrix, initial_j3.j_matrix)
    assert np.any(exponents != np.asarray(initial_j3.orb_data.radial.exponents))
    path = Path(tmp_path) / "optimized.h5"
    updated.save_to_hdf5(str(path))
    reloaded = Hamiltonian_data.load_from_hdf5(str(path))
    reloaded.sanity_check()
    loaded_j3 = _j3(reloaded)
    _assert_same_basis(current_j3.orb_data, loaded_j3.orb_data)
    np.testing.assert_array_equal(current_j3.j_matrix, loaded_j3.j_matrix, strict=True)
    np.testing.assert_array_equal(
        compute_Jastrow_three_body(current_j3, _R_UP, _R_DN),
        compute_Jastrow_three_body(loaded_j3, _R_UP, _R_DN),
        strict=True,
    )

    # Compare continuation from one physical accepted state on the two actual
    # production routes. Keep saved RNG/walkers rather than reseeding a clone.
    rank_pattern = str(Path(tmp_path) / "rank{rank}.h5")
    driver.save_to_hdf5(rank_pattern.format(rank=0))
    checkpoint = str(Path(tmp_path) / "restart.h5")
    merge_rank_checkpoints(
        checkpoint,
        mpi_size=1,
        driver_type="MCMC",
        hamiltonian_data=updated,
        tmp_pattern=rank_pattern,
    )
    restored = MCMC.load_from_hdf5(checkpoint)
    assert restored.i_opt == driver.i_opt
    _assert_same_basis(_j3(driver.hamiltonian_data).orb_data, _j3(restored.hamiltonian_data).orb_data)
    for sampler in (driver, restored):
        sampler.run(num_mcmc_steps=2)
    expected = _continuation_state(driver)
    actual = _continuation_state(restored)
    for name in expected:
        assert np.isfinite(actual[name]).all(), name
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name, strict=True)
