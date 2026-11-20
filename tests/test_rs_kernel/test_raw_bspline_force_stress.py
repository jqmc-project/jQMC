"""Public force/stress channels preserve raw composite AO cotangent rank.

The independent reference is the same-policy Python/XLA production driver.
Raw composites are canonical WF-PATTERN-013; child reversal permutes the AGP
coefficients along with the AO order, preserving the physical wavefunction.
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

import jax.numpy as jnp
import jqmc_rs
import numpy as np
import pytest

from jqmc import jqmc_mcmc as mcmc_module
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.derivatives.bridge import batched_grad_e_L, batched_grad_ln_Psi
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.jqmc_mcmc import MCMC
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]
_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "jastrow_eval", "ewald", "local_energy")
_FORCE_FIELDS = ("force_HF", "force_PP", "E_L_force_PP")
_STRESS_FIELDS = ("stress_HF", "stress_PP", "E_L_stress_PP")


def _hamiltonian(order):
    ham = build_canonical_hamiltonian("WF-PATTERN-013")
    wf = ham.wavefunction_data
    geminal = wf.geminal_data
    up = geminal.orb_data_up_spin
    down = geminal.orb_data_dn_spin
    assert len(up.children) == len(down.children) == 2
    assert up.num_ao == down.num_ao == 8
    assert not hasattr(up, "mo_coefficients") and not hasattr(down, "mo_coefficients")
    gto, spline = up.children
    assert gto.num_ao == spline.num_ao == 4
    matrix = np.asarray(geminal.lambda_matrix)
    if order == "bspline-first":
        permutation = np.asarray([4, 5, 6, 7, 0, 1, 2, 3])
        up = up.replace(children=(up.children[1], up.children[0]))
        down = down.replace(children=(down.children[1], down.children[0]))
        matrix = np.column_stack((matrix[:, :8][np.ix_(permutation, permutation)], matrix[:, 8:][permutation]))
    elif order == "pure-bspline":
        up, down = up.children[1], down.children[1]
        matrix = np.column_stack((matrix[4:8, 4:8], matrix[4:8, 8:]))
    else:
        assert order == "pgto-first"
    geminal = geminal.replace(orb_data_up_spin=up, orb_data_dn_spin=down, lambda_matrix=matrix)
    ham = ham.replace(wavefunction_data=wf.replace(geminal_data=geminal))
    ham.sanity_check()
    assert up.num_ao == down.num_ao == (4 if order == "pure-bspline" else 8)
    assert geminal.lambda_matrix.shape[0] == up.num_ao
    return ham


def _run(ham, driver_kind, *, force, stress):
    common = dict(hamiltonian_data=ham, comput_position_deriv=force, comput_stress=stress, random_discretized_mesh=True)
    if driver_kind == "MCMC":
        driver = MCMC(**common, num_walkers=2, mcmc_seed=11, Dt=1.0, epsilon_AS=1e-6, num_mcmc_per_measurement=2)
        driver.run(num_mcmc_steps=14)
    elif driver_kind == "GFMC_t":
        driver = GFMC_t(**common, num_walkers=4, mcmc_seed=34467, num_gfmc_collect_steps=1, tau=0.1, alat=0.3)
        driver.run(num_mcmc_steps=6)
    else:
        assert driver_kind == "GFMC_n"
        driver = GFMC_n(
            **common, num_walkers=4, mcmc_seed=34467, num_gfmc_collect_steps=1, num_mcmc_per_measurement=2, alat=0.3
        )
        driver.run(num_mcmc_steps=6)
    outputs = {"e_L": np.asarray(driver.e_L).copy(), "w_L": np.asarray(driver.w_L).copy()}
    fields = (*(_FORCE_FIELDS if force else ()), *(_STRESS_FIELDS if stress else ()))
    for field in fields:
        attribute = field + "_stored" if driver_kind == "MCMC" else field
        outputs[field] = np.asarray(getattr(driver, attribute)).copy()
    return outputs


def _assert_fixed_mcmc_derivatives(ham, chunks, *, force):
    """Actual fused public outputs match XLA at all inputs and its final inverse."""
    assert not ham.coulomb_potential_data.ecp_flag  # WF-PATTERN-013 is the AE fixture.

    def observed(name):
        block = np.concatenate([chunk[name] for chunk in chunks], axis=1)
        return block.swapaxes(0, 1).reshape(-1, *block.shape[2:])

    r_up = np.asarray(observed("up"), dtype=np.float64)
    r_dn = np.asarray(observed("down"), dtype=np.float64)
    assert len(r_up) == len(r_dn) == 28
    fresh_inverse = mcmc_module._geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(r_up), jnp.asarray(r_dn))[
        1
    ]
    final_inverse = np.asarray(chunks[-1]["final_inverse"], dtype=np.float64)
    assert final_inverse.shape[0] == 2
    native_eps = observed("stress_HF")
    native_ln_eps = observed("stress_PP")
    if force:
        native = (
            observed("hf_dR"),
            observed("hf_dr_up"),
            observed("hf_dr_dn"),
            native_eps,
            observed("pp_dR"),
            observed("pp_dr_up"),
            observed("pp_dr_dn"),
            native_ln_eps,
        )
    else:
        native = (native_eps, native_ln_eps)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    for selection, inverse in ((slice(None), fresh_inverse), (slice(-2, None), final_inverse)):
        up, down = jnp.asarray(r_up[selection]), jnp.asarray(r_dn[selection])
        rotations = jnp.asarray(np.broadcast_to(np.eye(3), (len(up), 3, 3)))
        if force:
            production = batched_grad_e_L(wrt=("r", "R", "eps"), with_geminal_inv=True, analytic=True)
            ln_production = batched_grad_ln_Psi(wrt=("r", "R", "eps"), with_geminal_inv=True, analytic=True)
            xla_up, xla_down, xla_R, xla_eps = production(ham, up, down, rotations, jnp.asarray(inverse))
            xla_ln_R, xla_ln_up, xla_ln_down, xla_ln_eps = ln_production(ham, up, down, jnp.asarray(inverse))
            reference = (xla_R, xla_up, xla_down, xla_eps, xla_ln_R, xla_ln_up, xla_ln_down, xla_ln_eps)
        else:
            production = batched_grad_e_L(wrt="eps", with_geminal_inv=True, analytic=True)
            ln_production = batched_grad_ln_Psi(wrt="eps", with_geminal_inv=True, analytic=True)
            xla_eps = production(ham, up, down, rotations, jnp.asarray(inverse))
            xla_ln_eps = ln_production(ham, up, down, jnp.asarray(inverse))
            reference = (xla_eps, xla_ln_eps)
        for block, reference_block in zip(native, reference, strict=True):
            actual_block = block[selection]
            assert actual_block.shape == reference_block.shape and actual_block.size
            assert np.all(np.isfinite(actual_block)) and np.all(np.isfinite(reference_block))
            np.testing.assert_allclose(actual_block, reference_block, atol=atol, rtol=rtol)


def _assert_mcmc_fixed_input_parity(ham, *, force, stress, monkeypatch):
    """Public MCMC completes; force/stress kernels agree at all 28 observed inputs."""
    assert stress
    original = jqmc_rs.build_obs_ctx
    contexts, chunks = [], []

    class CapturedContext:
        def __init__(self, inner):
            self.inner = inner
            contexts.append(inner)

        def __getattr__(self, name):
            method = getattr(self.inner, name)
            if name != "run_energy_chunk_batch":
                return method

            def captured(*args, **kwargs):
                # Passive coordinate snapshots expose the stress-only inputs.
                # This flag only allocates/copies positions in the real chunk.
                observed_kwargs = dict(kwargs)
                observed_kwargs["want_steps"] = True
                result = method(*args, **observed_kwargs)
                fields = ("hf_dR", "hf_dr_up", "hf_dr_dn", "pp_dR", "pp_dr_up", "pp_dr_dn") if force else ()
                row = {name: np.asarray(result[name]).copy() for name in (*fields, "stress_HF", "stress_PP")}
                row.update(
                    {
                        "up": np.asarray(result["r_up_steps"]).copy(),
                        "down": np.asarray(result["r_dn_steps"]).copy(),
                        "final_inverse": np.asarray(result["geminal_inv"]).copy(),
                    }
                )
                chunks.append(row)
                return result

            return captured

    def build(*args, **kwargs):
        return CapturedContext(original(*args, **kwargs))

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    baseline = _run(ham, "MCMC", force=force, stress=stress)
    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", build)
    actual = _run(ham, "MCMC", force=force, stress=stress)
    assert set(actual) == set(baseline)
    for field in actual:
        assert np.array_equal(actual[field], baseline[field]), f"passive coordinate recording changed {field}"
    assert contexts and chunks, "native MCMC observable route did not run"
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    reference = _run(ham, "MCMC", force=force, stress=stress)
    assert reference["e_L"].size > 2 and np.ptp(reference["e_L"]) > 0
    assert set(actual) == set(reference)
    for field in reference:
        assert actual[field].shape == reference[field].shape and actual[field].size
        assert np.all(np.isfinite(actual[field])) and np.all(np.isfinite(reference[field]))
    if force:
        assert np.max(np.abs(reference["force_HF"])) > 0
    assert np.max(np.abs(reference["stress_HF"])) > 0
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    for field in ("e_L", "w_L"):
        np.testing.assert_allclose(actual[field], reference[field], atol=atol, rtol=rtol)
    _assert_fixed_mcmc_derivatives(ham, chunks, force=force)


def _assert_driver_parity(ham, driver_kind, *, force, stress, monkeypatch):
    if driver_kind == "MCMC":
        _assert_mcmc_fixed_input_parity(ham, force=force, stress=stress, monkeypatch=monkeypatch)
        return
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    reference = _run(ham, driver_kind, force=force, stress=stress)
    assert reference["e_L"].size > 2 and np.ptp(reference["e_L"]) > 0
    assert all(block.size and np.all(np.isfinite(block)) for block in reference.values())
    if force:
        assert np.max(np.abs(reference["force_HF"])) > 0
    if stress:
        assert np.max(np.abs(reference["stress_HF"])) > 0

    original = jqmc_rs.build_obs_ctx
    calls = []
    method = {
        "MCMC": "run_energy_chunk_batch",
        "GFMC_t": "run_gfmc_projection_t_batch",
        "GFMC_n": "run_gfmc_projection_n_batch",
    }[driver_kind]

    class CountedContext:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            value = getattr(self.inner, name)
            if name != method:
                return value

            def counted(*args, **kwargs):
                calls.append(name)
                return value(*args, **kwargs)

            return counted

    def build(*args, **kwargs):
        return CountedContext(original(*args, **kwargs))

    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", build)
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    actual = _run(ham, driver_kind, force=force, stress=stress)
    assert calls, f"native {driver_kind} observable route did not run"
    assert set(actual) == set(reference)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    for field in reference:
        assert actual[field].shape == reference[field].shape and actual[field].size
        assert np.all(np.isfinite(actual[field]))
        np.testing.assert_allclose(actual[field], reference[field], atol=atol, rtol=rtol, err_msg=f"{driver_kind} {field}")


@pytest.mark.parametrize("order", ["pgto-first", "bspline-first"])
@pytest.mark.parametrize("driver_kind", ["MCMC", "GFMC_t", "GFMC_n"])
def test_native_raw_composite_force_stress_matches_xla(order, driver_kind, monkeypatch):
    """Both raw AO child orders complete all public force/stress channels with XLA parity."""
    _assert_driver_parity(_hamiltonian(order), driver_kind, force=True, stress=True, monkeypatch=monkeypatch)


@pytest.mark.parametrize("order", ["pgto-first", "bspline-first"])
def test_native_raw_composite_stress_only_matches_xla(order, monkeypatch):
    """Stress without a force request still selects only the B-spline cotangent rows."""
    _assert_driver_parity(_hamiltonian(order), "MCMC", force=False, stress=True, monkeypatch=monkeypatch)


def test_native_pure_bspline_force_stress_matches_xla(monkeypatch):
    """Pure-spline None passthrough completes a driver and agrees at identical inputs.

    Two independently propagated trajectories can round to different coordinates.
    Compare the original step-eight walker at identical coordinates and a fresh
    production inverse; retain the original seed, public run and central budget.
    """
    ham = _hamiltonian("pure-bspline")
    original = jqmc_rs.build_obs_ctx
    contexts, chunks = [], []

    class CapturedContext:
        def __init__(self, inner):
            self.inner = inner
            contexts.append(inner)

        def __getattr__(self, name):
            method = getattr(self.inner, name)
            if name != "run_energy_chunk_batch":
                return method

            def captured(*args, **kwargs):
                result = method(*args, **kwargs)
                chunks.append(
                    {
                        "up": np.asarray(result["r_up_steps"]).copy(),
                        "down": np.asarray(result["r_dn_steps"]).copy(),
                        "weights": np.asarray(args[11]).copy(),
                        "grid": np.asarray(kwargs["base_grid_points"]).copy(),
                    }
                )
                return result

            return captured

    def build(*args, **kwargs):
        return CapturedContext(original(*args, **kwargs))

    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", build)
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    outputs = _run(ham, "MCMC", force=True, stress=True)
    assert contexts and chunks
    assert outputs["e_L"].size > 2 and np.ptp(outputs["e_L"]) > 0
    assert all(block.size and np.all(np.isfinite(block)) for block in outputs.values())
    assert np.max(np.abs(outputs["force_HF"])) > 0
    assert np.max(np.abs(outputs["stress_HF"])) > 0
    up_steps = np.concatenate([chunk["up"] for chunk in chunks], axis=1)
    down_steps = np.concatenate([chunk["down"] for chunk in chunks], axis=1)
    assert up_steps.shape[:2] == down_steps.shape[:2] == (2, 14)
    r_up = np.asarray(up_steps[1, 8][None], dtype=np.float64)
    r_dn = np.asarray(down_steps[1, 8][None], dtype=np.float64)
    context = contexts[-1]
    geminal = context.sweep_seed_geminal_batch(
        r_up, r_dn, np.asarray(ham.structure_data.cell, dtype=np.float64), int(ham.structure_data._n_max)
    )
    inverse = np.asarray(mcmc_module._geminal_pinv_from_matrix_batched(jnp.asarray(geminal)), dtype=np.float64)
    weights = chunks[0]["weights"]
    grid = chunks[0]["grid"][None]
    native_R, native_up, native_down = context.grad_e_L_pbc_batch(r_up, r_dn, weights, grid, inverse)
    native_eps = context.grad_e_L_eps_pbc_batch(r_up, r_dn, weights, grid, inverse)
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    production = batched_grad_e_L(wrt=("r", "R", "eps"), with_geminal_inv=True, analytic=mcmc_module._ANALYTIC_GRAD_ENABLED)
    xla_up, xla_down, xla_R, xla_eps = production(
        ham, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(np.eye(3)[None]), jnp.asarray(inverse)
    )
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    for actual_block, reference_block in zip(
        (native_R, native_up, native_down, native_eps), (xla_R, xla_up, xla_down, xla_eps), strict=True
    ):
        assert actual_block.shape == reference_block.shape and actual_block.size
        assert np.all(np.isfinite(actual_block)) and np.all(np.isfinite(reference_block))
        np.testing.assert_allclose(actual_block, reference_block, atol=atol, rtol=rtol)
