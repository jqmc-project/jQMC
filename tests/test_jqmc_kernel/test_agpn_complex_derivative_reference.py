"""Independent same-precision references for complex AGPn optimizer sources."""

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import derivatives
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from jqmc.atomic_orbital import compute_AOs
from jqmc.derivatives import bridge
from jqmc.hamiltonians import compute_local_energy
from tests.test_jqmc_kernel._twist_geminal_cases import build_case
from tests.test_rs_kernel.test_rs_analytic_matrix import _BLOCK_FLAGS
from tests.test_rs_kernel.test_rs_complex_twist_param_parity import _open_shell_real_agpn_case

pytestmark = pytest.mark.with_chdir
_SOURCE_ZONES = ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic", "jastrow_grad_lap")


@pytest.fixture(autouse=True)
def _clear_caches():
    yield
    bridge.cache_clear()
    derivatives._jitted_assemble.cache_clear()
    jax.clear_caches()


def _complex_agpn_case(tmp_path, case, fixture_path):
    """Reuse the independently reviewed closed/open-shell nonfactorized inputs."""
    if fixture_path == "tests/trexio_example_files/H4_pbc_ae_k025_cart.h5":
        assert case == "closed-real"
        ham = build_case("agpn")
        rng = np.random.default_rng(2026)
        return ham, jnp.asarray(rng.normal(scale=1.5, size=(2, 3))), jnp.asarray(rng.normal(scale=1.5, size=(2, 3)))
    if fixture_path != "tests/qe_example_files/O-lsda_k025/out/o_k025.save":
        raise ValueError(f"Unknown AGPn fixture path: {fixture_path}")
    assert case in {"open-real", "open-complex"}
    ham, r_up, r_dn = _open_shell_real_agpn_case(tmp_path, np.random.default_rng(431), np.random.default_rng(41))
    if case == "open-complex":
        wf = ham.wavefunction_data
        gem = wf.geminal_data
        imaginary = 0.2 * np.random.default_rng(432).normal(size=gem.lambda_unpaired.shape)
        ham = ham.replace(
            wavefunction_data=wf.replace(geminal_data=gem.replace(lambda_unpaired=gem.lambda_unpaired + 1j * imaginary))
        )
    return ham, r_up, r_dn


def _agpn_parameter_coordinates(gem):
    coordinates = {"lambda_diag": jnp.real(gem.lambda_diag)}
    if gem.lambda_unpaired.size:
        coordinates["unpaired_real"] = jnp.real(gem.lambda_unpaired)
        if jnp.iscomplexobj(gem.lambda_unpaired):
            coordinates["unpaired_imag"] = jnp.imag(gem.lambda_unpaired)
    return coordinates


def _agpn_complex_forward_reference(ham, r_up, r_dn, quantity):
    """Differentiate complex outputs along real coordinates, not an analytic source."""
    wf = ham.wavefunction_data
    gem = wf.geminal_data
    # The log reference bypasses production MO, geminal and determinant kernels.
    # The e_L reference differentiates the primal energy, not its analytic
    # parameter gradient. AO/spatial-kinetic/ECP evaluators remain shared.
    up = jnp.asarray(gem.orb_data_up_spin.mo_coefficients) @ compute_AOs(gem.orb_data_up_spin.aos_data, r_up)
    dn = jnp.asarray(gem.orb_data_dn_spin.mo_coefficients) @ compute_AOs(gem.orb_data_dn_spin.aos_data, r_dn)

    def output(coordinates):
        unpaired = coordinates.get("unpaired_real", gem.lambda_unpaired)
        if "unpaired_imag" in coordinates:
            unpaired = unpaired + 1j * coordinates["unpaired_imag"]
        diag = coordinates["lambda_diag"]
        if quantity == "ln_Psi":
            paired_columns = up.T @ (diag[:, None] * dn)
            # Promote a real unpaired coordinate to the complex MO dtype before the
            # matmul. The derivative is unchanged (the convert transpose keeps the
            # real part of the cotangent), but JAX 0.7.x transposes a mixed
            # complex x real dot_general with a complex->real cast that emits
            # ComplexWarning, which the test configuration turns into an error.
            unpaired = unpaired.astype(jnp.result_type(up, unpaired))
            matrix = jnp.concatenate((paired_columns, up.T @ unpaired), axis=1) if unpaired.size else paired_columns
            value = jnp.log(jnp.linalg.det(matrix))
        else:
            changed = ham.replace(
                wavefunction_data=wf.replace(geminal_data=gem.replace(lambda_diag=diag, lambda_unpaired=unpaired))
            )
            value = compute_local_energy(changed, r_up, r_dn, jnp.eye(3))
        return jnp.stack((jnp.real(value), jnp.imag(value)))

    gradients = jax.jit(jax.jacrev(output))(_agpn_parameter_coordinates(gem))
    return {name: np.asarray(value[0] + 1j * value[1]) for name, value in gradients.items()}


@pytest.mark.heavy
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize(
    "case,fixture_path",
    [
        ("closed-real", "tests/trexio_example_files/H4_pbc_ae_k025_cart.h5"),
        ("open-real", "tests/qe_example_files/O-lsda_k025/out/o_k025.save"),
        ("open-complex", "tests/qe_example_files/O-lsda_k025/out/o_k025.save"),
    ],
    ids=["closed-real", "open-real", "open-complex"],
)
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_agpn_complex_optimizer_sources_match_real_coordinate_reference(tmp_path, case, fixture_path, quantity):
    """AGPn optimizer sources retain the complex derivative along every active real coordinate.

    Closed shell: committed H4_pbc_ae_k025_cart.h5, J1/J2, four complex MOs
    for two pairs, real occupations (1, 1/2, 1/4, 1/8), one PCG64(2026)
    Gaussian(scale=1.5) configuration. Open shell: O-lsda_k025/out/o_k025.save
    with O.ccECP.upf, four up/two down electrons, common-rank MOs, real
    parameter seed 431, electron seed 41 (scale 0.6); the complex-unpaired
    case adds imaginary PCG64(432) Gaussian(scale=0.2) entries. No cond(G)
    screening is used. The ECP mesh is the ordinary 12-point identity rotation.
    The log reference assembles MO values and paired/unpaired geminal columns
    directly from AO values; the energy reference differentiates the unchanged
    primal local-energy function. Both differentiate Re and Im along real
    parameter coordinates in the selected precision. Complex unpaired leaves
    additionally test the imaginary coordinate direction, whose derivative is
    i times the holomorphic source. Real occupations stay real parameters.
    The public analytic source_mode=True path is compared, not the intentionally
    projected ordinary derivative API. Full uses central strict, mixed medium.
    The same comparison must reject projecting away the source imaginary part.
    """
    ham, r_up, r_dn = _complex_agpn_case(tmp_path, case, fixture_path)
    reference = _agpn_complex_forward_reference(ham, r_up, r_dn, quantity)
    flags = _BLOCK_FLAGS["Det"]
    if quantity == "ln_Psi":
        source = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True, source_mode=True)["c"]
    else:
        source = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=True, source_mode=True)["c"]
    geminal_source = source.geminal_data
    actual = {
        "lambda_diag": np.asarray(geminal_source.lambda_diag),
        "unpaired_real": np.asarray(geminal_source.lambda_unpaired),
    }
    if case == "closed-real":
        assert actual["unpaired_real"].size == 0
    if case == "open-complex":
        actual["unpaired_imag"] = 1j * actual["unpaired_real"]
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(_SOURCE_ZONES, level)
    for name, expected in reference.items():
        assert np.iscomplexobj(actual[name])
        assert np.all(np.isfinite(actual[name])) and np.all(np.isfinite(expected))
        np.testing.assert_allclose(
            actual[name], expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"{case}/{quantity}/{name}"
        )
        assert np.any(np.abs(np.real(actual[name]) - expected) > atol + rtol * np.abs(expected))

    if is_mixed_precision_enabled():
        _complex_agpn_mixed_fidelity(tmp_path, case, fixture_path, quantity)


def _complex_agpn_assert_mixed_bound(mixed_block, full_block, atol, rtol, label):
    """Bind one unchanged central real-component comparison."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max(), label


def _complex_agpn_mixed_fidelity(tmp_path, case, fixture_path, quantity):
    """Keep complex-source capability and compare fresh analytical policy models."""
    from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block
    from tests.test_jqmc_kernel._generated_matrix_base import _qe_proj_pieces
    from tests.test_jqmc_kernel._grad_fidelity_harness import _qe_composite_pieces

    # Keep the original entry-Mixed projection-rejection band in both callbacks.
    projection_atol, projection_rtol = get_tolerance_min(_SOURCE_ZONES, "medium")

    def factory():
        bridge.cache_clear()
        derivatives._jitted_assemble.cache_clear()
        _qe_proj_pieces.cache_clear()
        _qe_composite_pieces.cache_clear()
        directory = tmp_path / ("mixed-source" if is_mixed_precision_enabled() else "full-source")
        directory.mkdir(exist_ok=True)
        return _complex_agpn_case(directory, case, fixture_path)

    def evaluate(model):
        ham, r_up, r_dn = model
        flags = _BLOCK_FLAGS["Det"]
        raw = (
            derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True, source_mode=True)["c"]
            if quantity == "ln_Psi"
            else derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=True, source_mode=True)["c"]
        )
        grad = raw.geminal_data
        actual = {"lambda_diag": np.asarray(grad.lambda_diag), "unpaired_real": np.asarray(grad.lambda_unpaired)}
        if case == "closed-real":
            assert actual["unpaired_real"].size == 0
        if case == "open-complex":
            actual["unpaired_imag"] = 1j * actual["unpaired_real"]
        coordinates = _agpn_parameter_coordinates(ham.wavefunction_data.geminal_data)
        blocks = {name: actual[name] for name in coordinates}
        assert blocks
        for name, value in blocks.items():
            assert value.shape == coordinates[name].shape and value.size
            assert np.iscomplexobj(value) and np.isfinite(value).all()
            # Retain the old capability: throwing away the source imaginary
            # component must fail beyond the same entry-policy band. The Full
            # independent mathematical reference remains in the original body.
            assert np.any(np.abs(np.real(value) - value) > projection_atol + projection_rtol * np.abs(value))
        return {"source": blocks, "up": r_up, "down": r_dn}

    try:
        mixed, full = mixed_vs_full(factory, evaluate)
    finally:
        # Restore an empty precision-blind model cache for later tests in the
        # entry policy; the final callback otherwise left Full model pieces.
        _qe_proj_pieces.cache_clear()
        _qe_composite_pieces.cache_clear()
    assert np.array_equal(mixed["up"], full["up"]) and np.array_equal(mixed["down"], full["down"])
    assert set(mixed["source"]) == set(full["source"])
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    compared = 0
    for name, reference in full["source"].items():
        assert mixed["source"][name].shape == reference.shape
        for part in ("real", "imag"):
            for mixed_block, full_block in mixed_fidelity_blocks(
                getattr(mixed["source"][name], part), getattr(reference, part)
            ):
                _complex_agpn_assert_mixed_bound(mixed_block, full_block, atol, rtol, f"{case}/{quantity}/{name}/{part}")
                compared += 1
                faulty = out_of_budget_mixed_block(mixed_block, full_block, atol=atol, rtol=rtol)
                with pytest.raises(AssertionError):
                    _complex_agpn_assert_mixed_bound(faulty, full_block, atol, rtol, "complex AGPn excess-error control")
    assert compared > 0
