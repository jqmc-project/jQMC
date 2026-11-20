"""Two-species DTN production-driver parity (issue #367).

Every other DTN driver fixture has a single chemical species, so ``species_index``
is constant and each per-species lookup in the native DTN kernels (chi
coefficients ``beta[sp]``, cutoffs ``L_chi[sp]`` / ``L_f[sp]``, the f-channel
gamma block ``sp``) selects the same row whatever index it uses. Water ccECP has
two species (H and O), a live non-local ECP on O and H neighbours inside the DTN
cutoffs, so a native species-index error changes the ECP mesh ratio and the
kinetic and candidate tables that feed ``e_L``. The DTN leg is built with the
production ``Jastrow_DTN_data.build`` defaults (orders 8/8/4/4, smoothness 4),
not the order-2 recipe of the single-species rows.

The deterministic driver oracles of ``test_chain_modes`` (MCMC, injected XLA
normals and meshes) and ``test_gfmc_chain_modes`` (GFMC_n / GFMC_t, native
Threefry/Haar random mesh) are reused unchanged, with their within-mode
Rust/XLA tolerance policies.
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

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_ZERO
from jqmc.jastrow_factor import Jastrow_DTN_data, _dtn_inflate_beta, _dtn_inflate_gamma
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel.test_chain_modes import _assert_value_parity as _assert_mcmc_value_parity
from tests.test_rs_kernel.test_chain_modes import _inject_xla_mcmc_inputs_into_rust, _run_mode
from tests.test_rs_kernel.test_gfmc_chain_modes import (
    _STEPS_STAT,
    _assert_deterministic_driver_parity,
    _install_ctx_spy,
    _run_gfmc,
)

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]

_SEED = 34356


def _water_two_species_dtn_ham():
    """Water ccECP SD + J1/J2 with an all-channel DTN at the production defaults."""
    ham = _build_ham_for_cell("water_ccecp_ccpvqz.h5", True, {"dtn": True}, "SD")
    structure = ham.structure_data
    ecp = ham.coulomb_potential_data
    atomic_numbers = np.asarray(structure.atomic_numbers)
    species = np.unique(atomic_numbers)
    charges = np.asarray(ecp._effective_charges)
    dtn = Jastrow_DTN_data.build(
        structure_data=structure,
        z_eff_per_species=tuple(float(np.mean(charges[atomic_numbers == z])) for z in species),
        pseudized_per_species=tuple(bool(np.all(ecp.pseudized_per_atom[atomic_numbers == z])) for z in species),
        random_init=True,
        seed=13,
    )
    jastrow = ham.wavefunction_data.jastrow_data.replace(jastrow_dtn_data=dtn)
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jastrow))
    ham.sanity_check()

    # The fixture must keep the properties that make a species-index error visible.
    assert not structure.pbc_flag and ecp.ecp_flag
    assert dtn.use_u and dtn.use_chi and dtn.use_f
    defaults = inspect.signature(Jastrow_DTN_data.build).parameters
    for name in ("n_order_u", "n_order_chi", "n_order_eN", "n_order_ee", "cutoff_smoothness"):
        assert getattr(dtn, name) == defaults[name].default, name
    assert sorted(set(dtn.species_index)) == [0, 1] and len(dtn.species_index) == len(atomic_numbers)
    beta = np.asarray(_dtn_inflate_beta(dtn, jnp.float64))
    gamma = np.asarray(_dtn_inflate_gamma(dtn, jnp.float64))
    assert beta.shape[0] == gamma.shape[0] == 2
    assert np.max(np.abs(beta[0] - beta[1])) > 0.0, "the two species must carry different chi coefficients"
    assert np.max(np.abs(gamma[0] - gamma[1])) > 0.0, "the two species must carry different f coefficients"
    return ham


def test_two_species_dtn_mcmc_driver_matches_xla():
    """Production MCMC on two-species DTN reproduces XLA on identical proposals.

    Four walkers, seed 34356, injected XLA normals and rotated meshes, as in
    ``test_chain_modes.test_rust_vs_jax_energy``. Acceptance is compared
    exactly. The returned walkers on this eight-electron water cell differ by
    a few final ulps between backends, with or without DTN (4.4e-16 bohr with
    this DTN, 1.3e-15 bohr for the same cell with J1/J2 only), so they are
    compared at the absolute fp64 accumulator pin with zero relative tolerance,
    as the GFMC random-mesh rows are. Stored weights, local energies and the
    public energy use the reviewed MCMC driver policy of each precision mode.
    """
    ham = _water_two_species_dtn_ham()
    xla = _run_mode(ham, _SEED, "jax")
    with _inject_xla_mcmc_inputs_into_rust():
        rust = _run_mode(ham, _SEED, "rust")
    assert rust["accepted"] == xla["accepted"]
    np.testing.assert_allclose(rust["r_up"], xla["r_up"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    np.testing.assert_allclose(rust["r_dn"], xla["r_dn"], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    compared = [
        _assert_mcmc_value_parity(rust["w_L"], xla["w_L"], "MCMC per-step w_L"),
        _assert_mcmc_value_parity(rust["e_L"], xla["e_L"], "MCMC per-step e_L"),
        _assert_mcmc_value_parity(rust["E"], xla["E"], "MCMC public energy estimator"),
    ]
    assert compared == ["MCMC per-step w_L", "MCMC per-step e_L", "MCMC public energy estimator"]


@pytest.mark.parametrize("driver", ["n", "t"])
def test_two_species_dtn_gfmc_driver_matches_xla(driver, monkeypatch):
    """Production GFMC_n / GFMC_t on two-species DTN reproduce XLA.

    One walker, seed 34356, ``GFMC_MIN_STEPS`` branching steps with the native
    Threefry/Haar random mesh, so the native projection interval evaluates the
    DTN candidate tables and ECP mesh ratios. The shared deterministic driver
    oracle compares stored weights, energies and squared energies, the public
    energy, final walkers and keys under its existing within-mode policy.
    """
    ham = _water_two_species_dtn_ham()
    xla = _run_gfmc(driver, ham, _SEED, "jax", _STEPS_STAT, random_mesh=True, num_walkers=1)
    counts = _install_ctx_spy(monkeypatch)
    rust = _run_gfmc(driver, ham, _SEED, "rust", _STEPS_STAT, random_mesh=True, num_walkers=1)
    assert counts[f"run_gfmc_projection_{driver}"] >= _STEPS_STAT, f"native GFMC_{driver} chunk did not engage: {counts}"
    _assert_deterministic_driver_parity(rust, xla, driver, periodic=False, random_mesh=True)
