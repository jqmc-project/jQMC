"""Empty spin channels carry no MO parameters; the remaining SD parameters work.

WF-PATTERN-008 is a validated three-up, zero-down complex plane-wave SD.
Full checks every real and imaginary occupied-MO coefficient against the
primitive ln|Psi| at the driver's final measured coordinates, with direct raw
coefficient replacement and a centered step-doubling check. No update/retraction
path participates in this oracle. Mixed checks production parameter fidelity
at those same frozen coordinates, while block-map and driver controls run in
both modes. The spin-swapped SD tests publication under public validation;
its sampler numerical domain is a separate question.
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

from contextlib import contextmanager
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT, get_tolerance_mixed_fidelity
from jqmc.derivatives.bridge import _reduce_c_param_grads, batched_grad_ln_Psi
from jqmc.determinant import AGPn_data, SD_data
from jqmc.jqmc_mcmc import MCMC, _RsComplexPdCtx, _leaf_to_fp64, _rs_native_c_param_grads
from jqmc.wavefunction import ParameterTangentSpec, VariationalParameterBlock, evaluate_ln_wavefunction
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian
from tests.test_rs_kernel.conftest import assert_native_ran, assert_no_native_ran, native_ctx_spy


pytestmark = [pytest.mark.with_chdir]
_RAW_COEFFICIENT_STEP = 1.0e-5
_MASK = {
    "opt_J1_param": True,
    "opt_J2_param": True,
    "opt_J3_param": True,
    "opt_geminal": True,
    "opt_J3_basis_exp": False,
    "opt_J3_basis_coeff": False,
    "opt_lambda_basis_exp": False,
    "opt_lambda_basis_coeff": False,
    "opt_JDTN_u_param": True,
    "opt_JDTN_chi_param": True,
    "opt_JDTN_f_param": True,
}


def _assert_parameter_fidelity(mixed_block, full_block, atol, rtol):
    """Expose the single homogeneous-block norm to source and runtime receipts."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max()


def _polarized_hamiltonian(active_spin="up"):
    ham = build_canonical_hamiltonian("WF-PATTERN-008")
    geminal = ham.wavefunction_data.geminal_data
    assert isinstance(geminal, SD_data)
    assert not geminal.time_reversal
    assert (geminal.num_electron_up, geminal.num_electron_dn) == (3, 0)
    if active_spin == "down":
        geminal = geminal.replace(
            num_electron_up=0,
            num_electron_dn=3,
            orb_data_up_spin=geminal.orb_data_dn_spin,
            orb_data_dn_spin=geminal.orb_data_up_spin,
        )
        ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=geminal))
    else:
        assert active_spin == "up"
    ham.sanity_check()
    return ham


@contextmanager
def _backend(rust_on):
    previous = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
    os.environ["JQMC_FLAG_RUST_ON_CPU"] = "1" if rust_on else "0"
    jax.clear_caches()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
        else:
            os.environ["JQMC_FLAG_RUST_ON_CPU"] = previous
        jax.clear_caches()


def _sampler(ham, *, energy_derivatives=False):
    return MCMC(
        hamiltonian_data=ham,
        num_walkers=2,
        mcmc_seed=11,
        random_discretized_mesh=True,
        Dt=1.0,
        epsilon_AS=1.0e-6,
        num_mcmc_per_measurement=2,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=energy_derivatives,
    )


def _assert_backend(counts, rust_on):
    if rust_on:
        assert_native_ran(counts, "polarized SD parameter workflow")
    else:
        assert_no_native_ran(counts, "polarized SD parameter workflow")


def _raw_mo_finite_differences(wavefunction, r_up, r_dn, step):
    """Differentiate raw real/imaginary coefficients, without any retraction."""
    geminal = wavefunction.geminal_data
    orbital = geminal.orb_data_up_spin
    coefficients = np.asarray(orbital.mo_coefficients)
    assert np.iscomplexobj(coefficients)
    fd = np.empty((r_up.shape[0], *coefficients.shape, 2), dtype=np.float64)
    primal = jax.jit(evaluate_ln_wavefunction)
    for index in np.ndindex(coefficients.shape):
        for component, direction in enumerate((1.0, 1.0j)):
            values = []
            for sign in (1.0, -1.0):
                changed = coefficients.copy()
                changed[index] += sign * step * direction
                perturbed = wavefunction.replace(
                    geminal_data=geminal.replace(orb_data_up_spin=orbital.replace(mo_coefficients=changed))
                )
                values.append(np.asarray([primal(perturbed, up, dn) for up, dn in zip(r_up, r_dn, strict=True)]))
            fd[(slice(None), *index, component)] = (values[0] - values[1]) / (2.0 * step)
    assert np.all(np.isfinite(fd))
    return fd


def _production_parameter_gradient(case, rust_on):
    """Rebuild the same backend's production c-axis map at frozen coordinates."""
    ham, r_up, r_dn = case
    if rust_on:
        ham = jax.tree_util.tree_map(_leaf_to_fp64, ham)
        wf = ham.wavefunction_data
        ctx = _RsComplexPdCtx(wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data)
        flat = _rs_native_c_param_grads(
            ctx, ham, r_up, r_dn, {"opt_geminal": True}, r_up.shape[0], quantity="ln_Psi", complex_twist=True
        )
        return np.asarray(flat["mo_coefficients_up"])
    call = batched_grad_ln_Psi(wrt="c", mask=_MASK)
    raw = call(ham, jnp.asarray(r_up), jnp.asarray(r_dn))
    flat = _reduce_c_param_grads(ham.wavefunction_data, raw, r_up.shape[0], _MASK)
    return np.asarray(flat["mo_coefficients_up"])


@pytest.mark.light
@pytest.mark.parametrize("active_spin", ["up", "down"])
def test_polarized_sd_publishes_only_nonempty_mo_channel(active_spin):
    """Both publicly validated spin orderings publish only their active MO rows."""
    ham = _polarized_hamiltonian(active_spin)
    geminal = ham.wavefunction_data.geminal_data
    inactive = "down" if active_spin == "up" else "up"
    inactive_orbital = geminal.orb_data_dn_spin if inactive == "down" else geminal.orb_data_up_spin
    assert np.asarray(inactive_orbital.mo_coefficients).shape == (0, 4)
    blocks = ham.wavefunction_data.get_variational_blocks(state_token=("polarized-blocks", 0), opt_geminal=True)
    names = {block.name for block in blocks}
    assert f"mo_coefficients_{active_spin if active_spin == 'up' else 'dn'}" in names
    assert f"mo_coefficients_{'dn' if inactive == 'down' else 'up'}" not in names
    assert all(block.size > 0 and np.asarray(block.values).size == block.size for block in blocks)


@pytest.mark.light
@pytest.mark.parametrize("pattern_id", ["WF-PATTERN-002", "WF-PATTERN-006"])
def test_nonempty_independent_mo_channels_remain_published(pattern_id):
    """Nonempty SD/AGPn spin channels and the AGPn empty-unpaired rule are retained."""
    ham = build_canonical_hamiltonian(pattern_id)
    ham.sanity_check()
    geminal = ham.wavefunction_data.geminal_data
    assert not geminal.time_reversal
    blocks = ham.wavefunction_data.get_variational_blocks(state_token=("nonempty-blocks", 0), opt_geminal=True)
    by_name = {block.name: block for block in blocks}
    assert by_name["mo_coefficients_up"].shape == np.asarray(geminal.orb_data_up_spin.mo_coefficients).shape
    assert by_name["mo_coefficients_dn"].shape == np.asarray(geminal.orb_data_dn_spin.mo_coefficients).shape
    assert all(block.size > 0 for block in blocks)
    if isinstance(geminal, AGPn_data):
        assert np.asarray(geminal.lambda_unpaired).size == 0
        assert "lambda_unpaired" not in by_name


@pytest.mark.light
def test_empty_storage_variational_block_is_still_rejected():
    """The shared nonempty-block invariant remains strict after channel omission."""
    token = ("empty-block-guard", 0)
    with pytest.raises(ValueError, match="must contain at least one storage value"):
        VariationalParameterBlock(
            name="empty",
            values=np.empty((0, 4), dtype=np.complex128),
            shape=(0, 4),
            size=0,
            field="complex",
            derivative_policy="holomorphic",
            sync_scope="sampler",
            tangent_spec=ParameterTangentSpec.euclidean(state_token=token),
            state_token=token,
        )


@pytest.mark.heavy
@pytest.mark.parametrize("rust_on", [False, True])
def test_polarized_sd_mcmc_parameters_match_raw_full_fd_or_mixed_fidelity(rust_on):
    """MCMC publishes correct raw MO derivatives without an inactive empty block."""
    with _backend(rust_on):
        ham = _polarized_hamiltonian()
        sampler = _sampler(ham)
        with native_ctx_spy() as counts:
            sampler.run(num_mcmc_steps=14)
        _assert_backend(counts, rust_on)
        blocks = ham.wavefunction_data.get_variational_blocks(state_token=sampler.wavefunction_state_token, **_MASK)
        by_name = {block.name: block for block in blocks}
        assert "mo_coefficients_dn" not in by_name
        assert set(sampler.dln_Psi_dc) == set(by_name)
        assert all(np.all(np.isfinite(values)) for values in sampler.dln_Psi_dc.values())
        mo = by_name["mo_coefficients_up"]
        matrix = sampler.get_dln_WF([mo], num_mcmc_warmup_steps=0)
        assert matrix.shape == (sampler.mcmc_counter, 2, mo.size)
        stored = np.asarray(matrix[-1]).reshape(2, *mo.shape)
        np.testing.assert_array_equal(stored, sampler.dln_Psi_dc[mo.name][-1])
        # The last run step is a measurement (14 is divisible by 2). Read its
        # exact coordinates rather than comparing different stochastic chains.
        fixed_up = np.asarray(sampler._MCMC__latest_r_up_carts).copy()
        fixed_dn = np.asarray(sampler._MCMC__latest_r_dn_carts).copy()
        assert fixed_up.shape == (2, 3, 3) and fixed_dn.shape == (2, 0, 3)
        assert np.all(np.isfinite(stored)) and stored.size == 24
        if is_mixed_precision_enabled():

            def factory():
                return _polarized_hamiltonian(), fixed_up.copy(), fixed_dn.copy()

            import jqmc_rs

            mixed_production, full = mixed_vs_full(
                factory,
                lambda case: _production_parameter_gradient(case, rust_on),
                configure_native=jqmc_rs.configure_native_precision,
                native_policy_snapshot=jqmc_rs.native_precision_support,
            )
            # Stored gradients and direct point reevaluation use the SAME
            # backend's production MO map in both arithmetic policies.
            blocks = mixed_fidelity_blocks((stored, mixed_production), (full, full))
            atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
            for derivative_source, (mixed_block, full_block) in zip(("stored-driver", "mixed-production"), blocks, strict=True):
                _assert_parameter_fidelity(mixed_block, full_block, atol, rtol)
            corrupted = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_parameter_fidelity(corrupted, blocks[0][1], atol, rtol)
        else:
            primitive_fd = _raw_mo_finite_differences(ham.wavefunction_data, fixed_up, fixed_dn, _RAW_COEFFICIENT_STEP)
            doubled_fd = _raw_mo_finite_differences(ham.wavefunction_data, fixed_up, fixed_dn, 2.0 * _RAW_COEFFICIENT_STEP)
            production_components = np.stack((np.real(stored), -np.imag(stored)), axis=-1)
            np.testing.assert_allclose(
                production_components, primitive_fd, atol=ATOL_FD_VS_ANALYTIC_TIGHT, rtol=RTOL_FD_VS_ANALYTIC_TIGHT
            )
            np.testing.assert_allclose(primitive_fd, doubled_fd, atol=ATOL_FD_VS_ANALYTIC_TIGHT, rtol=RTOL_FD_VS_ANALYTIC_TIGHT)


@pytest.mark.heavy
@pytest.mark.parametrize("rust_on", [False, True])
def test_polarized_sd_geminal_optimization_completes_one_step(rust_on):
    """Geminal-enabled optimization actually completes one step on each backend."""
    with _backend(rust_on):
        sampler = _sampler(_polarized_hamiltonian(), energy_derivatives=True)
        with native_ctx_spy() as counts:
            sampler.run_optimize(
                num_mcmc_steps=20,
                num_opt_steps=1,
                num_mcmc_warmup_steps=2,
                num_mcmc_bin_blocks=2,
                opt_geminal=True,
                dump_wavefunction=False,
            )
        _assert_backend(counts, rust_on)
        assert sampler.optimization_step_index == 1
        blocks = sampler.hamiltonian_data.wavefunction_data.get_variational_blocks(
            state_token=sampler.wavefunction_state_token, opt_geminal=True
        )
        assert "mo_coefficients_dn" not in {block.name for block in blocks}
        assert all(block.size > 0 and np.all(np.isfinite(block.values)) for block in blocks)
