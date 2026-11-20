"""Deterministic production checks at an explicitly measured ill-conditioned G."""

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

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_dtype_np, is_mixed_precision_enabled
from jqmc.hamiltonians import compute_local_energy
from jqmc.wavefunction import (
    _compute_kinetic_energy_auto,
    compute_kinetic_energy,
    evaluate_ln_wavefunction,
    evaluate_wavefunction,
)
from tests.test_jqmc_kernel.test_analytic_coverage_evidence import _condition_target_configuration


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize("filename,seed", [("N_ae_ccpvdz_cart.h5", 71)])
def test_ill_conditioned_g_production_local_energy_is_finite_and_condition_accurate(filename: str, seed: int):
    """Production local energy survives a precision-scaled ill-conditioned G.

    This is the numerical-behaviour companion to the generator-only contract.
    It exercises public wavefunction, analytical kinetic-energy and local-energy
    entries on the same measured condition-target configuration. Full targets
    ``1e11 < cond_2(G) < 1e12``; mixed targets ``1e4 < cond_2(G) < 1e5`` so the
    active fp32 AO error budget remains below one. NaN/Inf, an exact-node
    collapse, or analytical/autodiff disagreement beyond first-order
    ``cond(G) * eps_active`` amplification is RED.
    """
    lower, upper = (1.0e4, 1.0e5) if is_mixed_precision_enabled() else (1.0e11, 1.0e12)
    ham, target = _condition_target_configuration(lower, upper, filename=filename, seed=seed)
    r_up = jnp.asarray(target.r_up)
    r_dn = jnp.asarray(target.r_dn)

    assert lower < target.cond_g < upper
    assert target.sigma_min_g > 0.0
    active_epsilon = float(np.finfo(get_dtype_np("ao_eval")).eps)
    active_condition_budget = target.cond_g * active_epsilon
    assert 0.0 < active_condition_budget < 1.0

    psi = np.asarray(evaluate_wavefunction(ham.wavefunction_data, r_up, r_dn))
    ln_abs_psi = np.asarray(evaluate_ln_wavefunction(ham.wavefunction_data, r_up, r_dn))
    kinetic = np.asarray(compute_kinetic_energy(ham.wavefunction_data, r_up, r_dn))
    kinetic_auto = np.asarray(_compute_kinetic_energy_auto(ham.wavefunction_data, r_up, r_dn))
    local_energy = np.asarray(compute_local_energy(ham, r_up, r_dn, jnp.eye(3)))

    for name, value in (
        ("Psi", psi),
        ("ln|Psi|", ln_abs_psi),
        ("analytical kinetic energy", kinetic),
        ("autodiff kinetic energy", kinetic_auto),
        ("local energy", local_energy),
    ):
        assert np.all(np.isfinite(value)), f"{name} is non-finite at cond_2(G)={target.cond_g:.6e}: {value}"
    assert np.abs(psi) > 0.0, "the condition-target configuration collapsed to an exact node"

    relative_error = float(np.abs(kinetic - kinetic_auto) / max(np.abs(kinetic), np.abs(kinetic_auto)))
    assert relative_error <= active_condition_budget, (
        f"kinetic error {relative_error:.6e} exceeds cond_2(G)*eps_active={active_condition_budget:.6e} "
        f"at cond_2(G)={target.cond_g:.6e}"
    )
