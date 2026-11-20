"""MCMC production-driver Rust-vs-XLA deterministic end-to-end parity.

Production always generates normals, mesh rotations and ``w_L`` natively. The Rust
and XLA chains normally differ. For parity tests, a test-only Rust entry-point
seam injects the exact XLA proposal normals and ECP meshes while preserving all
production native arithmetic and output assembly.

This module holds two contracts:
  1. ``test_native_normal_sanity`` -- pointwise native/XLA Gaussian values on specified keys;
     ``test_native_normal_inverse_transform_reference`` anchors native values independently
     to SciPy's inverse function. Neither uses sample moments or a statistical error band.
  2. deterministic Rust-vs-XLA parity for the complete MCMC driver, including
     stored chains and public energy/force/stress estimators.
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

import contextlib
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special

pytestmark = [pytest.mark.with_chdir]

from jqmc._setting import (
    ATOL_PYTHON_FP64_LOOSE,
    MCMC_MIN_BIN_BLOCKS,
    MCMC_MIN_STEPS,
    MCMC_MIN_WARMUP_STEPS,
    RTOL_BIT_EXACT,
    atol_consistency,
    rtol_consistency,
)
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_three_body_data, Jastrow_two_body_data
from jqmc.molecular_orbital import MOs_data
from jqmc.jqmc_mcmc import MCMC
from jqmc.jqmc_mcmc import _draw_injected_normals_with_key, _generate_rotation_matrix
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests.test_rs_kernel._rs_fixtures import _build_ham_for_cell, rs_ham
from tests.test_rs_kernel.conftest import assert_native_ran, assert_no_native_ran, native_ctx_spy
from tests._ecp_core_electrons import core_electrons_of
from tests._native_gaussian_reference import gaussian_inputs, gaussian_inverse_reference, gaussian_xla_values

_TREXIO = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files")

# Deterministic parity needs enough samples to exercise binning, not to estimate
# an independent-chain statistical error.
_STEPS = MCMC_MIN_STEPS
_WARMUP = MCMC_MIN_WARMUP_STEPS
_NW = 4
# Use the same project-wide minimum trajectory length as the energy E2E rows.
# Longer trajectories add condition-number-dependent Sherman--Morrison drift;
# that numerical-stability regime is covered separately by fixed-input kernel
# and conditioned-walker tests, rather than by widening this driver contract.
_OBS_STEPS = MCMC_MIN_STEPS
_OBS_WARMUP = MCMC_MIN_BIN_BLOCKS * 3


def _draw_chunk_inputs_one(key, n_steps, num_moves, random_mesh):
    def _step(current_key, _):
        advanced_key, normals = _draw_injected_normals_with_key(current_key, num_moves)
        rotation = _generate_rotation_matrix(advanced_key) if random_mesh else jnp.eye(3, dtype=jnp.float64)
        return advanced_key, (normals, rotation)

    return jax.lax.scan(_step, key, None, length=n_steps)


_draw_chunk_inputs = jax.jit(
    jax.vmap(_draw_chunk_inputs_one, in_axes=(0, None, None, None)),
    static_argnums=(1, 2, 3),
)


@contextlib.contextmanager
def _inject_xla_mcmc_inputs_into_rust():
    """Replace only the Rust chunk's RNG/mesh source during a parity run.

    ``test_injected_inputs=True`` keeps the production-native arithmetic path
    (streaming state, AO cutoff, local energy and native ``w_L``), but consumes
    XLA-generated normals and rotated meshes. No production setting or alternate
    driver route is introduced.
    """
    import jqmc_rs

    original_build = jqmc_rs.build_obs_ctx

    class _InjectedCtx:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            attr = getattr(self._inner, name)
            if name != "run_energy_chunk_batch" or not callable(attr):
                return attr

            def _run(*args, **kwargs):
                assert kwargs.get("native_rng") is True
                assert not kwargs.get("complex", False), "deterministic MCMC fixture must use the real chunk"
                keys = jnp.asarray(args[8], dtype=jnp.uint32)
                n_steps = int(kwargs["n_steps_fast"])
                num_moves = int(args[12])
                _, (normals, rotations) = _draw_chunk_inputs(
                    keys,
                    n_steps,
                    num_moves,
                    bool(kwargs.get("random_mesh", False)),
                )
                base_grid = np.asarray(kwargs["base_grid_points"], dtype=np.float64)
                call_args = list(args)
                call_args[9] = np.ascontiguousarray(np.asarray(normals, dtype=np.float64))
                call_args[10] = np.ascontiguousarray(np.matmul(base_grid, np.asarray(rotations, dtype=np.float64)))
                kwargs["test_injected_inputs"] = True
                return attr(*call_args, **kwargs)

            return _run

    def _build(*args, **kwargs):
        return _InjectedCtx(original_build(*args, **kwargs))

    jqmc_rs.build_obs_ctx = _build
    try:
        yield
    finally:
        jqmc_rs.build_obs_ctx = original_build


def _build_ham(stem):
    structure_data, aos_data, mos_data, mos_data_no, geminal_mo_data, coulomb_potential_data = read_trexio_file(
        trexio_file=os.path.join(_TREXIO, stem), store_tuple=True
    )
    n_atom = len(structure_data.atomic_numbers)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=core_electrons_of(coulomb_potential_data, n_atom),
        jastrow_1b_type="exp",
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure_data
    )
    jastrow_data = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=None)
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    ham = Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)
    ham.sanity_check()
    return ham


def _build_molecular_mo_j3_ham(name):
    """A non-identity rectangular MO J3 over the committed molecular GTO leaf.

    Keep this energy-only fixture local: adding it must not implicitly expand
    the shared force/parameter matrices. Water has a live non-local ECP, while
    the non-square six-MO wrapper takes the native real chunk's MO fast-cut
    gradient/Laplacian projection instead of the raw-AO or periodic routes.
    """
    ham = rs_ham(name)
    wf = ham.wavefunction_data
    aos = wf.geminal_data.orb_data_up_spin.aos_data
    n_mo = 6
    assert not ham.structure_data.pbc_flag
    assert ham.coulomb_potential_data.ecp_flag
    assert n_mo < aos.num_ao
    rng = np.random.default_rng(20261104)
    coefficients = rng.normal(size=(n_mo, aos.num_ao)) / np.sqrt(aos.num_ao)
    orb = MOs_data(num_mo=n_mo, aos_data=aos, mo_coefficients=coefficients)
    j_matrix = 0.3 * rng.normal(size=(n_mo, n_mo + 1))
    j_matrix[:, :n_mo] = (j_matrix[:, :n_mo] + j_matrix[:, :n_mo].T) / 2
    j3 = Jastrow_three_body_data(orb_data=orb, j_matrix=j_matrix)
    ham = ham.replace(wavefunction_data=wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=j3)))
    ham.sanity_check()
    assert np.any(np.asarray(j3.j_matrix) != 0)
    assert np.count_nonzero(coefficients) > n_mo, "the MO projection must not be an identity wrapper"
    return ham


def _run_mode(ham, seed, mode, *, epsilon_as=1.0e-6):
    """One energy-only production MCMC run using ``mode`` in {"jax", "rust"}.

    The run is wrapped in ``native_ctx_spy`` and the BACKEND IS ASSERTED: "pop the env var and
    trust that the default is True" would leave nothing verifying that native code ran. Should
    the default flip, or a capability gate silently route this family back to XLA, the
    Rust and XLA runs would both be the same XLA run and parity would pass
    vacuously. The Rust run must show native calls and the XLA run none.
    """
    _prev_flag = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
    if mode == "jax":
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    else:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)  # -> rust backend on (default True)
    try:
        mcmc = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=seed,
            epsilon_AS=epsilon_as,
            num_walkers=_NW,
            comput_position_deriv=False,  # energy-only -> the native chunk path
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )
        with native_ctx_spy() as counts:
            mcmc.run(num_mcmc_steps=_STEPS)
        if mode == "jax":
            assert_no_native_ran(counts, f"XLA chain (seed={seed})")
        else:
            assert_native_ran(counts, f"Rust chain (seed={seed})")
        E = mcmc.get_E(num_mcmc_warmup_steps=_WARMUP, num_mcmc_bin_blocks=MCMC_MIN_BIN_BLOCKS)
        return {
            "E": (float(E[0]), float(E[1])),
            "e_L": mcmc.e_L,
            "w_L": mcmc.w_L,
            "accepted": int(mcmc._MCMC__accepted_moves),
            "native_chunk_calls": counts["run_energy_chunk_batch"],
            "r_up": np.asarray(mcmc._MCMC__latest_r_up_carts, dtype=np.float64),
            "r_dn": np.asarray(mcmc._MCMC__latest_r_dn_carts, dtype=np.float64),
        }
    finally:
        # Restore the PREVIOUS values: popping both unconditionally would make a caller that
        # had deliberately set either env var lose it after the first ``_run_mode`` and
        # silently continue on the default backend.
        if _prev_flag is None:
            os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
        else:
            os.environ["JQMC_FLAG_RUST_ON_CPU"] = _prev_flag


def _assert_value_parity(rust_value, xla_value, label, *, level="medium"):
    """Compare deterministic end-to-end values at the propagated-state envelope.

    ``medium`` is identical to ``strict`` in full precision.  In mixed precision
    it covers arithmetic-order drift accumulated in the running Sherman--Morrison
    state; fixed-input kernel parity remains strict in the kernel test modules.
    """
    # All existing callers deliberately use this reviewed driver envelope.
    # Keep the argument checked rather than silently discarding an override.
    assert level == "medium", "MCMC driver parity requires its declared medium policy"
    atol, rtol = get_tolerance_xla_rust_consistency(
        ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic", "swct"),
        level="medium",
    )
    np.testing.assert_allclose(rust_value, xla_value, atol=atol, rtol=rtol, err_msg=label)
    return label


def _run_position_observables(ham, rust_on, *, comput_position_deriv=True, comput_stress=True):
    previous = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
    if rust_on:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    else:
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    try:
        driver = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=34356,
            epsilon_AS=1.0e-6,
            num_walkers=_NW,
            comput_position_deriv=comput_position_deriv,
            comput_stress=comput_stress,
            use_swct=True,
            random_discretized_mesh=True,
        )
        with native_ctx_spy() as counts:
            driver.run(num_mcmc_steps=_OBS_STEPS)
        if rust_on:
            assert_native_ran(counts, "MCMC position-observable end-to-end parity")
        else:
            assert_no_native_ran(counts, "MCMC XLA reference")
        return driver
    finally:
        if previous is None:
            os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
        else:
            os.environ["JQMC_FLAG_RUST_ON_CPU"] = previous


@pytest.mark.light
@pytest.mark.parametrize("key_words,count", [([20260715, 9911], 1_000_000), ([1234567, 890123], 1_000_000)])
def test_native_normal_sanity(key_words, count):
    """Native split keys are exact and every Gaussian value agrees with XLA on the same key.

    The original Python and Rust-unit key pairs each specify one million inputs.
    Both full/mixed runs use fp64 normals, as the sampler does; no wavefunction,
    TREXIO artifact, electron positions or cond(G) applies. Compare each indexed
    value, not moments, KS statistics, or independent-chain sampling error.
    Native's approximate erfinv is not required to be bit-identical to XLA;
    use the unchanged central atol_consistency/rtol_consistency numerical pair.
    The independent SciPy inverse-function test anchors the approximation's
    mathematical meaning separately from this backend comparison.
    """
    import jqmc_rs

    subkeys, _ = gaussian_inputs(key_words, count)
    expected_keys = np.asarray(subkeys, dtype=np.uint32)
    actual_keys = np.asarray(jqmc_rs._tf_split(*key_words, count), dtype=np.uint32)
    actual = np.asarray(jqmc_rs.native_normals(*key_words, count))
    reference = gaussian_xla_values(subkeys)
    assert actual.shape == reference.shape == (count,)
    assert actual.dtype == reference.dtype == np.float64
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    np.testing.assert_array_equal(actual_keys, expected_keys)
    np.testing.assert_allclose(actual, reference, atol=atol_consistency, rtol=rtol_consistency)


@pytest.mark.light
@pytest.mark.parametrize("key_words,count", [([20260715, 9911], 1_000_000), ([1234567, 890123], 1_000_000)])
def test_native_normal_inverse_transform_reference(key_words, count):
    """Every native Gaussian output equals the independent inverse function of its specified uniform.

    Two explicit key pairs each define one million fp64 uniforms through JAX
    Threefry split/uniform; native output is not a reference input. SciPy's
    erfinv gives sqrt(2)*erfinv(u), and SciPy erf independently checks the
    inverse relation on the actual native value. XLA values must also agree
    with the inverse function and with native values, using the same bound.
    Both polynomial regions and both signs are exercised. Central atol_consistency/rtol_consistency apply
    element by element, unchanged in full/mixed; no sigma or sample-frequency
    tolerance is used. These scalar-function inputs use no wavefunction,
    committed file, electron positions or conditioning filter. This verifies
    the specified transform, not a proof of the PRNG's statistical quality.
    """
    import jqmc_rs

    subkeys, uniforms = gaussian_inputs(key_words, count)
    actual = np.asarray(jqmc_rs.native_normals(*key_words, count))
    reference = gaussian_inverse_reference(uniforms)
    xla_values = gaussian_xla_values(subkeys)
    recovered = scipy.special.erf(actual / np.sqrt(2.0))
    assert actual.shape == reference.shape == uniforms.shape == (count,)
    assert actual.dtype == reference.dtype == uniforms.dtype == np.float64
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    assert np.all((uniforms > -1) & (uniforms < 1))
    assert np.any(uniforms < 0) and np.any(uniforms > 0)
    central = (1 - uniforms) * (1 + uniforms) > np.exp(-5)
    assert np.any(central) and np.any(~central), "both native polynomial regions must be exercised"
    np.testing.assert_allclose(actual, reference, atol=atol_consistency, rtol=rtol_consistency)
    np.testing.assert_allclose(recovered, uniforms, atol=atol_consistency, rtol=rtol_consistency)
    np.testing.assert_allclose(xla_values, reference, atol=atol_consistency, rtol=rtol_consistency)
    np.testing.assert_allclose(actual, xla_values, atol=atol_consistency, rtol=rtol_consistency)


@pytest.mark.light
@pytest.mark.parametrize("key_words,count,sign", [([0, 25070191], 1, -1), ([0, 73650073], 1, 1)])
def test_native_normal_tail_reference(key_words, count, sign):
    """Fixed positive and negative tail keys agree with independent inverse-function values.

    JAX constructs the specified fp64 uniforms independently of native output.
    Both inputs must activate the far-tail region w >= 16, with the declared
    sign; the negative key is an input on which tail extrapolation fails.
    Direct SciPy inverse values, not the saturated forward-erf round trip,
    anchor native and XLA values using the unchanged central consistency pair.
    Both full/mixed use the sampler's same fp64 inputs; no wavefunction,
    TREXIO artifact, electron configuration or conditioning filter applies.
    """
    import jqmc_rs

    subkeys, uniforms = gaussian_inputs(key_words, count)
    actual = np.asarray(jqmc_rs.native_normals(*key_words, count))
    reference = gaussian_inverse_reference(uniforms)
    xla_values = gaussian_xla_values(subkeys)
    expected_keys = np.asarray(subkeys, dtype=np.uint32)
    actual_keys = np.asarray(jqmc_rs._tf_split(*key_words, count), dtype=np.uint32)
    input_checks = np.asarray(
        [
            uniforms.shape == actual.shape == reference.shape == xla_values.shape == (count,),
            uniforms.dtype == actual.dtype == reference.dtype == xla_values.dtype == np.float64,
            np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)) and np.all(np.isfinite(xla_values)),
            np.all((uniforms > -1) & (uniforms < 1)),
            np.all(np.sign(uniforms) == sign),
            np.all(-np.log((1 - uniforms) * (1 + uniforms)) >= 16),
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(input_checks, np.ones_like(input_checks))
    np.testing.assert_array_equal(actual_keys, expected_keys)
    np.testing.assert_allclose(actual, reference, atol=atol_consistency, rtol=rtol_consistency)
    np.testing.assert_allclose(xla_values, reference, atol=atol_consistency, rtol=rtol_consistency)
    np.testing.assert_allclose(actual, xla_values, atol=atol_consistency, rtol=rtol_consistency)


@pytest.mark.parametrize(
    "kind,name",
    [
        pytest.param("trexio", "H2_ecp_ccpvdz_cart.h5", marks=pytest.mark.light, id="H2_mol"),
        # Dense-J3 molecular ECP row: exercises the symmetric J3 ratio/streaming kernels
        # and the production chunk's carried J3 state. Light H2 fixture.
        pytest.param("rs", "mol_ecp_cart_j123", marks=pytest.mark.light, id="H2_ecp_j123"),
        pytest.param("trexio", "lithium.h5", marks=pytest.mark.heavy, id="Li_pbc"),
        # Dense MO-AGP geminal and the three single-channel DTN masks.
        pytest.param("rs", "mol_ecp_cart_agpmo_j12", marks=pytest.mark.light, id="H2_ecp_agpmo_j12"),
        pytest.param("rs", "mol_ecp_cart_jdtn_u", marks=pytest.mark.light, id="H2_ecp_jdtn_u"),
        pytest.param("rs", "mol_ecp_cart_jdtn_chi", marks=pytest.mark.light, id="H2_ecp_jdtn_chi"),
        pytest.param("rs", "mol_ecp_cart_jdtn_f", marks=pytest.mark.light, id="H2_ecp_jdtn_f"),
        # QE plane-wave / B-spline Gamma leaves with J1+J2.
        pytest.param("rs", "pw_gamma_j12", marks=pytest.mark.heavy, id="Li_pw_j12"),
        pytest.param("rs", "bspline_gamma_j12", marks=pytest.mark.heavy, id="Li_bspline_j12"),
    ],
)
def test_rust_vs_jax_energy(kind, name):
    """Production MCMC is deterministic across Rust and XLA on fixed inputs."""
    ham = rs_ham(name) if kind == "rs" else _build_ham(name)
    seed = 34356
    run_xla = _run_mode(ham, seed, "jax")
    with _inject_xla_mcmc_inputs_into_rust():
        run_rust = _run_mode(ham, seed, "rust")
    assert run_rust["accepted"] == run_xla["accepted"]
    np.testing.assert_array_equal(run_rust["r_up"], run_xla["r_up"])
    np.testing.assert_array_equal(run_rust["r_dn"], run_xla["r_dn"])
    compared = []
    compared.append(_assert_value_parity(run_rust["w_L"], run_xla["w_L"], "MCMC per-step w_L"))
    compared.append(_assert_value_parity(run_rust["e_L"], run_xla["e_L"], "MCMC per-step e_L"))
    compared.append(_assert_value_parity(run_rust["E"], run_xla["E"], "MCMC public energy estimator"))
    assert compared == ["MCMC per-step w_L", "MCMC per-step e_L", "MCMC public energy estimator"]


@pytest.mark.heavy
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_real_mcmc_chunk_as_guide_tracks_accepted_state():
    """Use the API-default guide in the real water ECP production energy chunk."""
    ham = rs_ham("mol_ecp_sphe_j12")
    seed, epsilon_as = 34356, 0.1
    run_xla = _run_mode(ham, seed, "jax", epsilon_as=epsilon_as)
    with _inject_xla_mcmc_inputs_into_rust():
        run_rust = _run_mode(ham, seed, "rust", epsilon_as=epsilon_as)

    assert run_rust["native_chunk_calls"] > 0
    assert run_xla["accepted"] > 1
    assert np.any(np.asarray(run_xla["w_L"]) < 1), "the AS guide must be active"
    assert run_rust["accepted"] == run_xla["accepted"], (
        f"active AS accept count: XLA={run_xla['accepted']} Rust={run_rust['accepted']}"
    )
    for spin in ("r_up", "r_dn"):
        np.testing.assert_allclose(
            run_rust[spin], run_xla[spin], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_BIT_EXACT, err_msg=spin
        )
    for name in ("w_L", "e_L", "E"):
        _assert_value_parity(run_rust[name], run_xla[name], f"active AS chunk: {name}")


@pytest.mark.heavy
def test_molecular_mo_j3_chunk_energy_matches_xla_at_returned_walkers(monkeypatch):
    """The real native fast-cut MO-J3 energy equals an independent endpoint recomputation.

    Fixed ECP quadrature isolates the J3 gradient/Laplacian projection from the
    native random-mesh generator. Full precision reaches the f64 MO fast-cut
    branch; mixed independently checks its supported f32 path. No equality of
    independently propagated trajectories or statistical error bars is assumed.
    """
    _assert_chunk_endpoint_energy_matches_xla(
        _build_molecular_mo_j3_ham("mol_ecp_sphe_j12"),
        monkeypatch,
        "molecular rectangular MO-J3 native chunk endpoint energy",
    )


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["mol_ecp_cart_j123", "mol_cart_sd_j123"])
def test_molecular_raw_ao_j3_chunk_energy_matches_xla_at_returned_walkers(pattern, monkeypatch):
    """The native FAST raw-AO-J3 energy equals an independent endpoint recomputation.

    A raw-AO molecular J3 takes the chunk-carried J3 derivative table in both
    precision modes (in mixed it is contracted in fp64 and the summed Jastrow
    gradient / Laplacian is rounded to the fp32 jastrow_grad_lap zone). The XLA
    reference evaluates the fresh local energy at the walkers the native chunk
    returns, so the carried table is checked against an independent rebuild.
    """
    _assert_chunk_endpoint_energy_matches_xla(
        rs_ham(pattern), monkeypatch, f"{pattern}: raw-AO J3 native chunk endpoint energy"
    )


@pytest.mark.heavy
@pytest.mark.parametrize("cell", ["lithium.h5", "lithium_cart.h5"], ids=["sphe", "cart"])
def test_periodic_pgto_chunk_energy_matches_xla_at_returned_walkers(cell, monkeypatch):
    """The native FAST periodic MO-pGTO ECP energy equals an independent endpoint recomputation.

    A real-Gamma periodic MO-pGTO geminal takes the FAST route of the native chunk in both
    precision modes: the sweep's determinant state is carried across the chunk's steps and
    handed to the e_L, whose kinetic contracts the chunk-carried fp64 derivative tables
    against the carried orbital values and whose ECP non-local reuses that state. Under an
    fp32 ao_eval zone the carried values are the zone-local fp32 sweep values. The XLA
    reference rebuilds the local energy from scratch at the walkers the native chunk
    returns, so the carried state is checked against an independent evaluation.
    """
    _assert_chunk_endpoint_energy_matches_xla(
        _build_ham_for_cell(cell, True, None, "SD"), monkeypatch, f"{cell}: periodic pGTO native chunk endpoint energy"
    )


def _assert_chunk_endpoint_energy_matches_xla(ham, monkeypatch, label):
    """Run the production native chunk twice; compare its last e_L with XLA at the returned walkers."""
    monkeypatch.delenv("JQMC_FLAG_RUST_ON_CPU", raising=False)
    driver = MCMC(
        hamiltonian_data=ham,
        Dt=2.0,
        mcmc_seed=34356,
        epsilon_AS=1.0e-6,
        num_walkers=2,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        random_discretized_mesh=False,
    )
    with native_ctx_spy() as counts:
        for _ in range(2):
            previous_calls = counts["run_energy_chunk_batch"]
            driver.run(num_mcmc_steps=3)
            assert counts["run_energy_chunk_batch"] > previous_calls, "the real native production chunk must execute"
            r_up = np.asarray(driver._MCMC__latest_r_up_carts)
            r_dn = np.asarray(driver._MCMC__latest_r_dn_carts)
            actual = np.asarray(driver.e_L)[-1]
            with monkeypatch.context() as reference_context:
                reference_context.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
                reference = np.asarray(
                    [
                        compute_local_energy(ham, jnp.asarray(up), jnp.asarray(dn), jnp.eye(3))
                        for up, dn in zip(r_up, r_dn, strict=True)
                    ]
                )
            assert actual.shape == reference.shape == (2,)
            assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
            if np.iscomplexobj(actual) != np.iscomplexobj(reference):
                # A complex-twist driver may store Re(e_L) only; compare the stored part.
                actual, reference = np.real(actual), np.real(reference)
            _assert_value_parity(actual, reference, label)


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["twist_sphepgto_j12", "pgto_pw_twist_jnone", "bspline_twist_jnone"])
def test_complex_twist_chunk_energy_matches_xla_at_returned_walkers(pattern, monkeypatch):
    """The native complex-twist FAST chunk energy equals an independent endpoint recomputation.

    QE Li k=(1/4,1/4,1/4) ccECP pGTO (J1+J2), pGTO+PW and pure B-spline determinant leaves.
    The complex FAST chunk contracts the ECP non-local mesh determinant ratio in AO space
    (``ComplexOrb::mesh_wdot_into``) in both precision modes: the Bloch pGTO block in its
    fp64 ``ao_bloch_eval`` zone and, under an fp32 ``ao_eval`` zone, the plane-wave /
    B-spline child from its zone-local fp32 values. The XLA reference rebuilds the local
    energy from scratch at the walkers the native chunk returns.
    """
    _assert_chunk_endpoint_energy_matches_xla(
        rs_ham(pattern), monkeypatch, f"{pattern}: complex-twist native chunk endpoint energy"
    )


@pytest.mark.heavy
@pytest.mark.parametrize(
    "comput_position_deriv,comput_stress",
    [
        pytest.param(True, True, id="force-stress"),
        pytest.param(False, True, id="stress-only"),
    ],
)
def test_mcmc_rust_vs_xla_end_to_end_force_stress(comput_position_deriv, comput_stress):
    """Production MCMC ON/OFF parity through sampling, storage and requested estimators.

    The periodic ECP fixture makes energy, HF/Pulay force and HF/Pulay stress all
    non-vacuous. XLA proposal normals and meshes are injected into the otherwise
    unchanged native production chunk, making every stored step comparable.
    """
    ham = _build_ham("lithium.h5")
    xla = _run_position_observables(
        ham,
        rust_on=False,
        comput_position_deriv=comput_position_deriv,
        comput_stress=comput_stress,
    )
    with _inject_xla_mcmc_inputs_into_rust():
        rust = _run_position_observables(
            ham,
            rust_on=True,
            comput_position_deriv=comput_position_deriv,
            comput_stress=comput_stress,
        )

    assert rust._MCMC__accepted_moves == xla._MCMC__accepted_moves
    np.testing.assert_array_equal(rust._MCMC__latest_r_up_carts, xla._MCMC__latest_r_up_carts)
    np.testing.assert_array_equal(rust._MCMC__latest_r_dn_carts, xla._MCMC__latest_r_dn_carts)
    compared = []
    for name in ("w_L", "e_L", "e_L2"):
        compared.append(_assert_value_parity(getattr(rust, name), getattr(xla, name), f"MCMC stored {name}"))
    observable_names = ["stress_HF_stored", "stress_PP_stored"]
    if comput_position_deriv:
        observable_names.extend(["force_HF_stored", "force_PP_stored"])
    for name in observable_names:
        compared.append(_assert_value_parity(getattr(rust, name), getattr(xla, name), f"MCMC stored {name}", level="medium"))
    args = (_OBS_WARMUP, MCMC_MIN_BIN_BLOCKS)
    compared.append(_assert_value_parity(rust.get_E(*args), xla.get_E(*args), "MCMC public energy"))
    if comput_position_deriv:
        compared.append(_assert_value_parity(rust.get_aF(*args), xla.get_aF(*args), "MCMC public force", level="medium"))
    else:
        for run in (rust, xla):
            assert run.force_HF_stored.size == 0
            assert run.force_PP_stored.size == 0
            assert run.E_L_force_PP_stored.size == 0
    compared.append(_assert_value_parity(rust.get_aT(*args), xla.get_aT(*args), "MCMC public stress", level="medium"))
    expected_comparisons = [
        "MCMC stored w_L",
        "MCMC stored e_L",
        "MCMC stored e_L2",
        "MCMC stored stress_HF_stored",
        "MCMC stored stress_PP_stored",
    ]
    if comput_position_deriv:
        expected_comparisons.extend(["MCMC stored force_HF_stored", "MCMC stored force_PP_stored"])
    expected_comparisons.append("MCMC public energy")
    if comput_position_deriv:
        expected_comparisons.append("MCMC public force")
    expected_comparisons.append("MCMC public stress")
    assert compared == expected_comparisons
