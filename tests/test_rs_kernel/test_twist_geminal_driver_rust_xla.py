"""Rust vs XLA production drivers at k != Gamma for AGP, AGPn and SD geminals.

The only driver-level Rust/XLA cells at a twist used a time-reversal SD on the ECP
lithium_k025 fixture.  This module drives the production MCMC, GFMC_n and GFMC_t on
the all-electron ``H4_pbc_ae_k025_cart.h5`` twist with J1/J2 and the geminal cases of
``tests/test_jqmc_kernel/_twist_geminal_cases.py``.  Each row proves that the native
complex route ran (ctx spy), asserts the walk identical first and then compares the
stored values.

* GFMC_n / GFMC_t (one walker for GFMC_t): the native complex projection
  interval reproduces the threefry key chain, so keys and every discrete decision are
  identical; walker coordinates carry the established native-vs-XLA trig ULP pin of
  the random mesh.
* MCMC with parameter derivatives uses the per-step native complex sweep.  Its
  proposal normals are drawn by the native route itself; the test injects the XLA
  normals for the same keys (the sweep reproduces every other draw from the keys), so
  the walk becomes byte-identical and the stored e_L, w_L and optimizer derivative
  sources are compared, including the complex AGPn occupation derivative even
  though occupations themselves are real optimization parameters.
* Energy-only MCMC at a twist runs the native complex chunk, which refuses injected
  inputs; that explicit refusal is the observed contract (pytest.raises), so this
  chain has no deterministic Rust/XLA twin.
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

import collections
import contextlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_ZERO
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.jqmc_mcmc import MCMC, _jit_vmap_draw_injected_normals
from tests.test_jqmc_kernel._twist_geminal_cases import build_case
from tests.test_rs_kernel.test_gfmc_chain_modes import _assert_value_parity

jax.config.update("jax_enable_x64", True)

_SEED = 3446
_GFMC_STEPS = 6
_MCMC_STEPS = 4


@contextlib.contextmanager
def _backend(rust: bool):
    previous = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
    os.environ["JQMC_FLAG_RUST_ON_CPU"] = "1" if rust else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
        else:
            os.environ["JQMC_FLAG_RUST_ON_CPU"] = previous


@contextlib.contextmanager
def _native_ctx(counts, *, inject_mcmc_normals=False, force_chunk_injection=False, observed_sweeps=None):
    """Count native ctx calls; optionally replace only the per-step sweep's proposal normals.

    ``inject_mcmc_normals`` swaps the normals argument of ``mcmc_sweep_complex_batch`` for the
    XLA normals of the same per-walker keys. ``force_chunk_injection`` asks the energy chunk for
    injected inputs, which the complex chunk must refuse.
    """
    import jqmc_rs

    original_build = jqmc_rs.build_obs_ctx

    class _Ctx:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            attr = getattr(self._inner, name)
            if not callable(attr):
                return attr

            def _call(*args, **kwargs):
                counts[name] += 1
                if inject_mcmc_normals and name == "mcmc_sweep_complex_batch":
                    keys = jnp.asarray(args[8], dtype=jnp.uint32)
                    num_moves = int(np.asarray(args[9]).shape[1])
                    args = list(args)
                    args[9] = np.ascontiguousarray(
                        np.asarray(_jit_vmap_draw_injected_normals(keys, num_moves), dtype=np.float64)
                    )
                if force_chunk_injection and name == "run_energy_chunk_batch":
                    kwargs["test_injected_inputs"] = True
                if observed_sweeps is not None and name == "mcmc_sweep_complex_batch":
                    matrices = np.array(args[4])
                    epsilon_as = args[7]
                    result = attr(*args, **kwargs)
                    observed_sweeps.append(
                        {"geminal": matrices, "epsilon_as": epsilon_as, "accepted": np.array(result["accepted"])}
                    )
                    return result
                return attr(*args, **kwargs)

            return _call

    jqmc_rs.build_obs_ctx = lambda *a, **k: _Ctx(original_build(*a, **k))
    try:
        yield
    finally:
        jqmc_rs.build_obs_ctx = original_build


def _run_gfmc(driver, hamiltonian, rust, counts):
    common = dict(
        hamiltonian_data=hamiltonian,
        num_walkers=2 if driver == "n" else 1,
        num_gfmc_collect_steps=1,
        mcmc_seed=_SEED,
        alat=0.30,
        random_discretized_mesh=True,
        comput_position_deriv=False,
    )
    with _backend(rust), _native_ctx(counts):
        if driver == "n":
            run = GFMC_n(**common, num_mcmc_per_measurement=4, E_scf=-2.0)
        else:
            run = GFMC_t(**common, tau=0.05)
        run.run(num_mcmc_steps=_GFMC_STEPS)
    return run


@pytest.mark.light
@pytest.mark.parametrize("driver", ["n", "t"])
@pytest.mark.parametrize("case", ["agp", "agpn", "sd_untied"])
def test_twist_geminal_gfmc_rust_matches_xla(case, driver):
    """Production GFMC_n / GFMC_t native complex interval equals XLA at k != Gamma for AGP, AGPn, untied SD.

    H4_pbc_ae_k025_cart.h5 with J1+J2; seed 3446, alat=0.30, random discretized mesh, 6 branching
    steps; GFMC_n 2 walkers with 4 projections per measurement and E_scf=-2, GFMC_t 1 walker with
    tau=0.05. The Rust run must call the native complex projection interval every step and the XLA
    run no native method. PRNG keys must be identical and final walkers equal at the random-mesh
    ULP pin (ATOL_PYTHON_FP64_LOOSE, zero rtol); the stored complex e_L (measurable Im), e_L2 and
    w_L agree within the periodic fast-interval Rust/XLA envelope of test_gfmc_chain_modes.
    """
    hamiltonian = build_case(case)
    xla_counts, rust_counts = collections.Counter(), collections.Counter()
    xla = _run_gfmc(driver, hamiltonian, False, xla_counts)
    rust = _run_gfmc(driver, hamiltonian, True, rust_counts)
    assert sum(xla_counts.values()) == 0, dict(xla_counts)
    native = f"run_gfmc_projection_{driver}_complex_batch"
    assert rust_counts[native] == _GFMC_STEPS, dict(rust_counts)

    np.testing.assert_array_equal(rust._jax_PRNG_key_list, xla._jax_PRNG_key_list)
    np.testing.assert_allclose(
        rust._latest_r_up_carts, xla._latest_r_up_carts, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
    )
    np.testing.assert_allclose(
        rust._latest_r_dn_carts, xla._latest_r_dn_carts, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
    )
    for name in ("_stored_w_L", "_stored_e_L", "_stored_e_L2"):
        rust_value, xla_value = np.asarray(getattr(rust, name)), np.asarray(getattr(xla, name))
        assert rust_value.shape == xla_value.shape and np.all(np.isfinite(rust_value)) and np.all(np.isfinite(xla_value))
        _assert_value_parity(rust_value, xla_value, f"{case}: GFMC_{driver} {name}", periodic=True)
    e_l = np.asarray(xla._stored_e_L)
    assert np.iscomplexobj(e_l) and np.max(np.abs(e_l.imag)) > 0.0
    jax.clear_caches()


def _run_mcmc(hamiltonian, rust, counts, *, param_deriv=True, force_chunk_injection=False):
    with _backend(rust), _native_ctx(counts, inject_mcmc_normals=rust, force_chunk_injection=force_chunk_injection):
        run = MCMC(
            hamiltonian_data=hamiltonian,
            num_walkers=2,
            num_mcmc_per_measurement=4,
            mcmc_seed=_SEED,
            Dt=0.5,
            epsilon_AS=0.0,
            comput_log_WF_param_deriv=param_deriv,
        )
        run.run(num_mcmc_steps=_MCMC_STEPS)
    return run


@pytest.mark.light
@pytest.mark.parametrize("case", ["agp", "agpn", "sd_untied", "sd_tr"])
def test_twist_geminal_mcmc_rust_matches_xla(case):
    """Production MCMC native complex per-step sweep equals XLA at k != Gamma (AGP, AGPn, untied and tied SD).

    H4_pbc_ae_k025_cart.h5 with J1+J2; 2 walkers, seed 3446, 4 measurements of 4 moves, Dt=0.5,
    epsilon_AS=0, comput_log_WF_param_deriv=True (per-step route). The XLA proposal normals of the
    same keys are injected into the native sweep. The Rust run must call the native complex sweep
    and native complex e_L every measurement and the XLA run no native method. Acceptance counts,
    PRNG keys and final walkers must be byte-identical; stored complex e_L and w_L, and for every
    case each optimizer derivative source component (direct, d_p, d_pbar), agree within
    the Rust/XLA consistency envelope, including AGPn occupations.
    """
    hamiltonian = build_case(case)
    xla_counts, rust_counts = collections.Counter(), collections.Counter()
    xla = _run_mcmc(hamiltonian, False, xla_counts)
    rust = _run_mcmc(hamiltonian, True, rust_counts)
    assert sum(xla_counts.values()) == 0, dict(xla_counts)
    assert rust_counts["mcmc_sweep_complex_batch"] == _MCMC_STEPS, dict(rust_counts)
    assert rust_counts["e_L_complex_batch"] == _MCMC_STEPS, dict(rust_counts)

    assert int(rust._MCMC__accepted_moves) == int(xla._MCMC__accepted_moves)
    np.testing.assert_array_equal(rust._MCMC__jax_PRNG_key_list, xla._MCMC__jax_PRNG_key_list)
    np.testing.assert_array_equal(rust._MCMC__latest_r_up_carts, xla._MCMC__latest_r_up_carts)
    np.testing.assert_array_equal(rust._MCMC__latest_r_dn_carts, xla._MCMC__latest_r_dn_carts)
    for name in ("w_L", "e_L", "e_L2"):
        rust_value, xla_value = np.asarray(getattr(rust, name)), np.asarray(getattr(xla, name))
        assert rust_value.shape == xla_value.shape and np.all(np.isfinite(rust_value)) and np.all(np.isfinite(xla_value))
        _assert_value_parity(rust_value, xla_value, f"{case}: MCMC stored {name}")
    assert np.iscomplexobj(xla.e_L) and np.max(np.abs(np.asarray(xla.e_L).imag)) > 0.0
    blocks = hamiltonian.wavefunction_data.get_variational_blocks(state_token=("twist-rust-xla", 0), opt_geminal=True)
    rust_sources = rust._optimization_derivative_sources(blocks=blocks)
    xla_sources = xla._optimization_derivative_sources(blocks=blocks)
    compared = 0
    for rust_source, xla_source in zip(rust_sources, xla_sources, strict=True):
        assert rust_source.name == xla_source.name
        for component in ("direct", "d_p", "d_pbar"):
            rust_value, xla_value = getattr(rust_source, component), getattr(xla_source, component)
            assert (rust_value is None) == (xla_value is None), f"{case}: {rust_source.name}.{component} presence differs"
            if xla_value is None:
                continue
            rust_value, xla_value = np.asarray(rust_value), np.asarray(xla_value)
            assert rust_value.dtype == xla_value.dtype, f"{case}: {rust_source.name}.{component} dtype"
            assert np.all(np.isfinite(rust_value)) and np.all(np.isfinite(xla_value))
            _assert_value_parity(rust_value, xla_value, f"{case}: MCMC source {rust_source.name}.{component}")
            compared += 1
    assert compared >= 3, f"{case}: only {compared} source components compared"
    jax.clear_caches()


@pytest.mark.heavy
def test_twist_geminal_mcmc_as_guide_tracks_accepted_state():
    """The complex native sweep refreshes the AS guide after accepted moves.

    The API-default epsilon_AS=0.1 is active at a same-spin near-node start
    on the committed H4 twist (a genuine 2x2 geminal). Both real drivers get
    identical initial positions and proposal normals. The XLA path recomputes
    the current AS factor, independently of the native carried-state update.
    """
    from jqmc._precision import get_tolerance_min
    from jqmc.determinant import compute_AS_regularization_factor
    from tests.test_jqmc_kernel._twist_geminal_cases import reference_geminal

    hamiltonian = build_case("sd_tr")
    geminal_data = hamiltonian.wavefunction_data.geminal_data
    epsilon_as, num_steps = 0.1, 8
    runs, route_counts, observed_sweeps = [], [], []

    def reference_factor(matrix):
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        assert np.all(singular_values > 0.0)
        minimum_norm = min(np.min(np.sum(abs(matrix) ** 2, axis=0)), np.min(np.sum(abs(matrix) ** 2, axis=1)))
        return (minimum_norm * np.sum(1.0 / singular_values**2)) ** (-3.0 / 8.0)

    for rust in (False, True):
        counts = collections.Counter()
        with _backend(rust), _native_ctx(counts, inject_mcmc_normals=rust, observed_sweeps=observed_sweeps if rust else None):
            run = MCMC(
                hamiltonian_data=hamiltonian,
                num_walkers=4,
                num_mcmc_per_measurement=6,
                mcmc_seed=_SEED,
                Dt=2.0,
                epsilon_AS=epsilon_as,
                comput_log_WF_param_deriv=True,
            )
            r_up = np.array(run._MCMC__latest_r_up_carts)
            r_up[:, 1, :] = r_up[:, 0, :] + 0.05
            run._MCMC__latest_r_up_carts = jnp.asarray(r_up)
            r_dn = np.asarray(run._MCMC__latest_r_dn_carts)
            initial_factors = np.asarray(
                jax.vmap(compute_AS_regularization_factor, in_axes=(None, 0, 0))(geminal_data, r_up, r_dn)
            )
            # Independently assemble G and use its singular values for ||G^-1||_F.
            # The initial anchor is a defining matrix identity, not the native
            # carried inverse or its AS helper.
            reference_factors = []
            for up, dn in zip(r_up, r_dn, strict=True):
                matrix = np.asarray(reference_geminal(hamiltonian.wavefunction_data, up, dn))
                assert matrix.shape == (2, 2) and np.iscomplexobj(matrix)
                reference_factors.append(reference_factor(matrix))
            atol, rtol = get_tolerance_min(("det_eval", "det_ratio"), "strict")
            np.testing.assert_allclose(initial_factors, reference_factors, atol=atol, rtol=rtol)
            assert np.all(np.isfinite(initial_factors)) and np.all(initial_factors > 0.0)
            assert np.any(initial_factors < epsilon_as), "the AS guide must engage at the supplied initial state"
            run.run(num_mcmc_steps=num_steps)
        runs.append(run)
        route_counts.append(counts)

    xla, rust = runs
    xla_counts, rust_counts = route_counts
    assert sum(xla_counts.values()) == 0, dict(xla_counts)
    assert rust_counts["mcmc_sweep_complex_batch"] == num_steps, dict(rust_counts)
    assert rust_counts["e_L_complex_batch"] == num_steps, dict(rust_counts)
    assert len(observed_sweeps) == num_steps
    first_sweep = observed_sweeps[0]
    assert first_sweep["epsilon_as"] == epsilon_as and np.any(first_sweep["accepted"] > 0)
    first_factors = np.array([reference_factor(matrix) for matrix in first_sweep["geminal"]])
    np.testing.assert_allclose(first_factors, initial_factors, atol=atol, rtol=rtol)
    assert np.any(first_factors < epsilon_as), "the actual first native sweep must receive an active AS guide"
    assert int(xla._MCMC__accepted_moves) > 1
    assert int(rust._MCMC__accepted_moves) == int(xla._MCMC__accepted_moves)
    np.testing.assert_array_equal(rust._MCMC__jax_PRNG_key_list, xla._MCMC__jax_PRNG_key_list)
    np.testing.assert_array_equal(rust._MCMC__latest_r_up_carts, xla._MCMC__latest_r_up_carts)
    np.testing.assert_array_equal(rust._MCMC__latest_r_dn_carts, xla._MCMC__latest_r_dn_carts)
    for name in ("w_L", "e_L", "e_L2"):
        rust_value, xla_value = np.asarray(getattr(rust, name)), np.asarray(getattr(xla, name))
        assert rust_value.shape == xla_value.shape and rust_value.size
        assert np.isfinite(rust_value).all() and np.isfinite(xla_value).all()
        _assert_value_parity(rust_value, xla_value, f"active AS guide: {name}")
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize("case", ["agp", "agpn", "sd_untied"])
def test_twist_geminal_mcmc_energy_chunk_refuses_injected_inputs(case):
    """Energy-only MCMC at k != Gamma runs the native complex chunk, which refuses injected inputs.

    H4_pbc_ae_k025_cart.h5 with J1+J2; production MCMC with comput_log_WF_param_deriv=False on the
    Rust route delegates to run_energy_chunk_batch(complex=True). Asking that chunk for test-injected
    inputs must raise ValueError("... complex-twist chunk supports fast mode only ..."): the native
    energy-only complex chain uses native normals and meshes, so it has no deterministic XLA twin.
    """
    hamiltonian = build_case(case)
    counts = collections.Counter()
    with pytest.raises(ValueError, match="complex-twist chunk supports fast mode only"):
        _run_mcmc(hamiltonian, True, counts, param_deriv=False, force_chunk_injection=True)
    assert counts["run_energy_chunk_batch"] == 1, dict(counts)
    jax.clear_caches()
