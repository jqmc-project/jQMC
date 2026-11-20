"""``_MCMC_debug`` publishes the production AGPn parameter-derivative blocks.

Collecting the per-walker parameter gradients through
``Wavefunction_data.collect_param_grads`` does not work for ``AGPn_data``: it reads
the dense ``lambda_matrix`` compatibility property, which calls ``np.diag`` /
``np.hstack`` on the walker-stacked gradient and raises ``ValueError`` for every AGPn
geminal. Production MCMC collects AGPn sources through
``derivatives.bridge._collect_c_param_derivative_sources`` and stores the
``mo_coefficients_up`` / ``lambda_diag`` (/ ``lambda_unpaired``) blocks.

Each node runs production XLA MCMC and its eager debug twin from the same seed on the
same AGPn Hamiltonian with ``comput_log_WF_param_deriv=True`` and
``comput_e_L_param_deriv=True``.  The walk is compared first (PRNG keys and acceptance
counters exactly, walkers at the fp64 accumulator pin); then every block production
stores in ``dln_Psi_dc`` and ``de_L_dc`` must be present in the debug twin with the
same shape and agree strictly per precision mode, all values finite.  The only extra
debug blocks allowed are the geminal AO-basis blocks production computes only when
basis optimization is requested.
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
from jqmc.determinant import AGPn_data
from jqmc.jqmc_mcmc import MCMC, _MCMC_debug
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "jastrow_eval",
    "jastrow_grad_lap",
    "det_eval",
    "det_grad_lap",
    "coulomb",
    "ewald",
    "wf_eval",
    "wf_kinetic",
    "local_energy",
)
_SEED = 3446
_STEPS = 3
_BASIS_ONLY_DEBUG_BLOCKS = {"lambda_basis_exp", "lambda_basis_coeff"}


def _build(case: str):
    if case == "mol_cart_agpn_jnone":
        return rs_ham(case)
    if case == "pbc_gamma_sphepgto_j12_agpn":
        # lithium.h5 (real Gamma, ECP) with the reviewed J1+J2 recipe, AGPn geminal.
        return _build_ham_for_cell("lithium.h5", True, None, "AGPn")
    raise ValueError(case)


def _assert_block(debug_value, production_value, label):
    debug_value = np.asarray(debug_value)
    production_value = np.asarray(production_value)
    assert debug_value.shape == production_value.shape, (label, debug_value.shape, production_value.shape)
    assert np.all(np.isfinite(debug_value)), label
    assert np.all(np.isfinite(production_value)), label
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    np.testing.assert_allclose(debug_value, production_value, atol=atol, rtol=rtol, equal_nan=False, err_msg=label)


def _assert_blocks(debug_grads, production_grads, label, nonzero):
    assert set(production_grads) <= set(debug_grads), (label, sorted(debug_grads), sorted(production_grads))
    assert set(debug_grads) - set(production_grads) <= _BASIS_ONLY_DEBUG_BLOCKS, (label, sorted(debug_grads))
    for required in ("mo_coefficients_up", "lambda_diag"):
        assert required in production_grads, (label, required, sorted(production_grads))
    for name in nonzero:
        assert np.max(np.abs(np.asarray(production_grads[name]))) > 0.0, (label, name)
    for name in sorted(production_grads):
        _assert_block(debug_grads[name], production_grads[name], f"{label}[{name}]")


@pytest.mark.light
@pytest.mark.parametrize("case", ["mol_cart_agpn_jnone", "pbc_gamma_sphepgto_j12_agpn"])
def test_mcmc_debug_agpn_parameter_derivatives_match_production(case):
    """_MCMC_debug stores the same AGPn d ln Psi / dc and de_L / dc blocks as production XLA MCMC.

    Fixtures: molecular all-electron H2 cc-pVDZ Cartesian AGPn without Jastrow
    (``mol_cart_agpn_jnone``) and periodic real-Gamma lithium ECP sphe-pGTO AGPn with J1+J2.
    2 walkers, seed 3446, 3 measurements of 4 moves, Dt=0.5, epsilon_AS=0.  PRNG keys and
    acceptance counters identical, final walkers at ATOL_PYTHON_FP64_MEDIUM, then every
    production block equal per precision mode (mo_coefficients_up and lambda_diag required; both
    non-zero in d ln Psi / dc, mo_coefficients_up non-zero in de_L / dc).
    """
    hamiltonian = _build(case)
    assert isinstance(hamiltonian.wavefunction_data.geminal_data, AGPn_data)
    kwargs = dict(
        hamiltonian_data=hamiltonian,
        num_walkers=2,
        num_mcmc_per_measurement=4,
        mcmc_seed=_SEED,
        Dt=0.5,
        epsilon_AS=0.0,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=True,
    )
    assert os.environ.get("JQMC_FLAG_RUST_ON_CPU") == "0", "this twin is the XLA production route"
    debug = _MCMC_debug(**kwargs)
    production = MCMC(**kwargs)
    debug.run(num_mcmc_steps=_STEPS)
    production.run(num_mcmc_steps=_STEPS)

    np.testing.assert_array_equal(debug._MCMC_debug__jax_PRNG_key_list, production._MCMC__jax_PRNG_key_list)
    np.testing.assert_array_equal(debug._MCMC_debug__accepted_moves, production._MCMC__accepted_moves)
    np.testing.assert_array_equal(debug._MCMC_debug__rejected_moves, production._MCMC__rejected_moves)
    np.testing.assert_allclose(
        debug._MCMC_debug__latest_r_up_carts,
        production._MCMC__latest_r_up_carts,
        atol=ATOL_PYTHON_FP64_MEDIUM,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    np.testing.assert_allclose(
        debug._MCMC_debug__latest_r_dn_carts,
        production._MCMC__latest_r_dn_carts,
        atol=ATOL_PYTHON_FP64_MEDIUM,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    _assert_block(debug.e_L, production.e_L, f"{case}: e_L")
    _assert_blocks(debug.dln_Psi_dc, production.dln_Psi_dc, f"{case}: dln_Psi_dc", ("mo_coefficients_up", "lambda_diag"))
    # A rank-one H2 AGPn has Psi proportional to its single occupation, so de_L / d lambda_diag is
    # identically zero there; only the MO block is required to be non-zero for de_L / dc.
    _assert_blocks(debug.de_L_dc, production.de_L_dc, f"{case}: de_L_dc", ("mo_coefficients_up",))
    jax.clear_caches()
