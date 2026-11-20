"""Native CPU selection, lossless input transport and live zone-policy wiring."""

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import jqmc_rs

from jqmc import jqmc_gfmc, jqmc_mcmc
from jqmc._precision import is_mixed_precision_enabled, precision_policy_snapshot
from jqmc._setting import MCMC_MIN_STEPS
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize(
    "env_flag,default_flag,backend,expected",
    [
        (None, True, "cpu", True),
        (None, False, "cpu", False),
        ("1", False, "cpu", True),
        ("0", True, "cpu", False),
        (" False ", True, "cpu", False),
        ("NO", True, "cpu", False),
        ("off", True, "cpu", False),
        ("", True, "cpu", False),
        ("1", True, "gpu", False),
    ],
)
def test_native_cpu_predicate_does_not_exclude_mixed(monkeypatch, env_flag, default_flag, backend, expected):
    """Both precision modes use the explicit flag/default and CPU-only truth table.

    Literal parameters cover default on/off, overriding each default, every
    disabled spelling and a non-CPU backend. No Hamiltonian or electron
    coordinates are needed. This shared GFMC predicate is distinct from
    extension import success and per-observable eligibility.
    """
    if env_flag is None:
        monkeypatch.delenv("JQMC_FLAG_RUST_ON_CPU", raising=False)
    else:
        monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", env_flag)
    monkeypatch.setattr(jqmc_gfmc, "flag_rust_on_CPU", default_flag)
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    assert jqmc_gfmc._rust_cpu_backend_active() is expected


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128", "int64", "bool"])
def test_native_leaf_transport_preserves_values_and_backend(backend, dtype):
    """Widen six literal array kinds without losing values, imaginary parts or backend.

    Non-dyadic inputs exercise already-rounded binary32 values. An int64 above
    2**53 detects accidental float conversion. Integers and booleans retain
    object identity; complex inputs retain their nonzero imaginary components.
    This exact transport test has no wavefunction or electron configuration
    and does not compare arithmetic performed at different precisions.
    """
    if dtype.startswith("complex"):
        values = [1.1 + 0.7j, -2.3 - 0.9j]
    elif dtype == "int64":
        values = [2**53 + 1, -7]
    elif dtype == "bool":
        values = [True, False]
    else:
        values = [1.1, -2.3]
    original = np.asarray(values, dtype=dtype)
    source = original if backend == "numpy" else jnp.asarray(original)
    actual = jqmc_mcmc._leaf_to_fp64(source)
    expected_dtype = "complex128" if dtype.startswith("complex") else "float64" if dtype.startswith("float") else dtype
    reference = original.astype(expected_dtype)
    assert str(actual.dtype) == expected_dtype
    assert isinstance(actual, np.ndarray) if backend == "numpy" else isinstance(actual, jax.Array)
    np.testing.assert_array_equal(np.asarray(actual), reference)
    np.testing.assert_array_equal(np.asarray(source), original)
    if dtype in ("int64", "bool"):
        assert actual is source


@pytest.mark.parametrize("driver", ["MCMC", "GFMC_n", "GFMC_t"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_native_context_receives_current_policy_and_unchanged_input(monkeypatch, driver):
    """Real native contexts receive the active policy and the same rounded Hamiltonian.

    Committed molecular H2 AE Cartesian/MO has one up/one down electron and
    a Pade J2 parameter of 1.1, built in the current precision mode. Use seed
    12345 and two walkers without cond(G) filtering. MCMC runs its minimum
    measurement count; each GFMC driver resolves its real native interval
    context. A thin wrapper records the inputs then calls the real Rust
    builder. Exact leaf values and the complete live zone policy are checked.
    Empty ECP leaves retain shape and dtype, without counting empty value
    comparisons as numerical evidence.
    This proves dispatch and transport, not an independent energy oracle.
    """
    path = Path(__file__).resolve().parents[1] / "trexio_example_files" / "H2_ae_ccpvdz_cart.h5"
    structure, _, _, _, geminal, coulomb = read_trexio_file(str(path), store_tuple=True)
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.1, jastrow_2b_type="pade", structure_data=structure
    )
    jastrow = Jastrow_data(jastrow_two_body_data=j2)
    ham = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=jastrow),
    )
    expected_policy = precision_policy_snapshot()
    assert expected_policy["ao_eval"] == ("float32" if is_mixed_precision_enabled() else "float64")
    observed = []
    real_builder = jqmc_rs.build_obs_ctx

    def capture(*args, **kwargs):
        observed.append((args[:3], dict(kwargs["precision_policy"])))
        return real_builder(*args, **kwargs)

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", capture)
    common = dict(hamiltonian_data=ham, num_walkers=2, mcmc_seed=12345, comput_position_deriv=False)
    if driver == "MCMC":
        sampler = jqmc_mcmc.MCMC(**common)
        sampler.run(num_mcmc_steps=MCMC_MIN_STEPS)
        assert np.isfinite(sampler.e_L).all()
    else:
        if driver == "GFMC_n":
            sampler = jqmc_gfmc.GFMC_n(**common, alat=0.3, num_mcmc_per_measurement=1, E_scf=-1.0)
        else:
            sampler = jqmc_gfmc.GFMC_t(**common, alat=0.3, tau=0.05)
        assert sampler._resolve_native_gfmc_chunk(driver[-1]) is not None
    assert len(observed) == 1
    actual_parts, actual_policy = observed[0]
    assert actual_policy == expected_policy
    for original, actual in zip((geminal, jastrow, coulomb), actual_parts, strict=True):
        original_leaves, original_tree = jax.tree_util.tree_flatten(original)
        actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
        assert actual_tree == original_tree
        for source, packed in zip(original_leaves, actual_leaves, strict=True):
            source_array = np.asarray(source)
            target_dtype = (
                np.float64
                if np.issubdtype(source_array.dtype, np.floating)
                else np.complex128
                if np.issubdtype(source_array.dtype, np.complexfloating)
                else source_array.dtype
            )
            reference = source_array.astype(target_dtype)
            assert np.asarray(packed).shape == reference.shape
            if source_array.size:
                np.testing.assert_array_equal(np.asarray(packed), reference)
            if isinstance(source, (np.ndarray, jax.Array)):
                if np.issubdtype(source.dtype, np.floating):
                    assert packed.dtype == np.float64
                elif np.issubdtype(source.dtype, np.complexfloating):
                    assert packed.dtype == np.complex128


@pytest.mark.parametrize("driver", ["MCMC", "GFMC_n", "GFMC_t"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_native_policy_verdict_rejects_wrong_driver_policy(monkeypatch, driver):
    """After a passing real-context baseline, detect a different AO arithmetic policy.

    Flip only the ao_eval zone supplied by the driver; both values are valid
    native schema entries, so an unrelated schema exception cannot pass this
    test. The actual context is still constructed and the exact policy verdict
    must fail. No trajectory average or statistical acceptance is involved.
    """
    with monkeypatch.context() as baseline:
        test_native_context_receives_current_policy_and_unchanged_input(baseline, driver)
    policy = precision_policy_snapshot()
    changed = dict(policy)
    changed["ao_eval"] = "float64" if policy["ao_eval"] == "float32" else "float32"
    module = jqmc_mcmc if driver == "MCMC" else jqmc_gfmc
    monkeypatch.setattr(module, "precision_policy_snapshot", lambda: dict(changed))
    with pytest.raises(AssertionError, match="ao_eval"):
        test_native_context_receives_current_policy_and_unchanged_input(monkeypatch, driver)


@pytest.mark.parametrize("env_flag", ["1"])
def test_native_predicate_verdict_rejects_forced_off_cpu(monkeypatch, env_flag):
    """A passing enabled-CPU baseline rejects a predicate that incorrectly disables it.

    This is a nonnumerical truth-table check in both precision modes, separate
    from the exact numerical transport checks. No Hamiltonian is needed.
    """
    test_native_cpu_predicate_does_not_exclude_mixed(monkeypatch, env_flag, True, "cpu", True)
    monkeypatch.setattr(jqmc_gfmc, "_rust_cpu_backend_active", lambda: False)
    with pytest.raises(AssertionError):
        test_native_cpu_predicate_does_not_exclude_mixed(monkeypatch, env_flag, True, "cpu", True)


@pytest.mark.parametrize("fault", ["value-change", "integer-cast"])
def test_native_transport_verdict_rejects_corrupted_leaf(monkeypatch, fault):
    """A passing numerical baseline rejects a changed value or integer-to-float cast.

    Change one widened value or cast a large integer to float64. Each literal
    violation must fail its corresponding existing verdict without changing
    the central precision policy or tolerance.
    """
    dtype = "float32" if fault == "value-change" else "int64"
    test_native_leaf_transport_preserves_values_and_backend("numpy", dtype)
    real_pack = jqmc_mcmc._leaf_to_fp64

    def corrupt(source):
        packed = real_pack(source)
        if fault == "value-change":
            packed[0] += 1
            return packed
        return packed.astype(np.float64)

    monkeypatch.setattr(jqmc_mcmc, "_leaf_to_fp64", corrupt)
    with pytest.raises(AssertionError):
        test_native_leaf_transport_preserves_values_and_backend("numpy", dtype)
