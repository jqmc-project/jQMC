"""Complex AGPn lambda sources consumed by the native VMC optimizer."""

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import derivatives
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.derivatives import bridge
from jqmc.jqmc_mcmc import _RsComplexPdCtx, _rs_native_c_param_grads, _rs_pd_complex_gap
from tests.test_jqmc_kernel.test_agpn_complex_derivative_reference import _complex_agpn_case
from tests.test_rs_kernel.test_rs_analytic_matrix import _one_move_mesh

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]
_MASK = {"opt_geminal": True}
_SOURCE_ZONES_BY_QUANTITY = {
    "ln_Psi": ("ao_eval", "det_eval"),
    "e_L": ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic", "jastrow_grad_lap"),
}


@pytest.fixture(autouse=True)
def _clear_caches():
    yield
    bridge.cache_clear()
    derivatives._jitted_assemble.cache_clear()
    jax.clear_caches()


def _sources(ham, up, dn, quantity, *, native, diagnostic=False):
    wf = ham.wavefunction_data
    blocks = tuple(
        block
        for block in wf.get_variational_blocks(state_token=("agpn-complex-source", 0), **_MASK)
        if block.name in {"lambda_diag", "lambda_unpaired", "mo_coefficients_up", "mo_coefficients_dn"}
    )
    if native:
        ctx = _RsComplexPdCtx(wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data)
        weights, grid = _one_move_mesh()
        return _rs_native_c_param_grads(
            ctx,
            ham,
            [up],
            [dn],
            _MASK,
            1,
            quantity=quantity,
            ecp_flag=bool(ham.coulomb_potential_data.ecp_flag),
            ecp_w=weights,
            ecp_gp=grid,
            rts_b=[np.eye(3)],
            complex_twist=True,
            source_blocks=None if diagnostic else blocks,
        )
    provider = bridge._batched_c_source(quantity, mask=_MASK, analytic=True)
    args = (ham, jnp.asarray(up)[None], jnp.asarray(dn)[None])
    grad = provider(*args) if quantity == "ln_Psi" else provider(*args, jnp.eye(3)[None])
    return bridge._collect_c_param_derivative_sources(wf, grad, blocks, 1, _MASK)


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
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_native_agpn_complex_lambda_sources_match_xla(tmp_path, case, fixture_path, quantity):
    """Native optimizer sources preserve complex AGPn paired and unpaired derivatives.

    Closed-shell H4 all-electron twist uses four natural orbitals, real occupations,
    J1/J2 and PCG64 seed 2026. Open-shell O-lsda_k025 ccECP uses four up/two down,
    real occupations and either real or genuinely complex unpaired coefficients,
    parameter seeds 431/432 and electron seed 41. One fixed configuration per case,
    no additional rejection screen; twelve-point ECP quadrature and identity rotation.
    The native production assembler is compared with the XLA optimizer source provider
    in the same precision, including imaginary parts and source metadata. A real
    projection must fail the identical numerical comparison band for each lambda leaf.
    The separate legacy diagnostic dictionary intentionally projects real-parameter
    leaves only; complex unpaired leaves and optimizer sources retain their phase.
    """
    ham, up, dn = _complex_agpn_case(tmp_path, case, fixture_path)
    up, dn = np.asarray(up), np.asarray(dn)
    expected = {source.name: source for source in _sources(ham, up, dn, quantity, native=False)}
    actual = {source.name: source for source in _sources(ham, up, dn, quantity, native=True)}
    diagnostic = _sources(ham, up, dn, quantity, native=True, diagnostic=True)
    assert set(actual) == set(expected)
    required = {"lambda_diag"} if case == "closed-real" else {"lambda_diag", "lambda_unpaired"}
    assert {name for name in expected if name.startswith("lambda_")} == required
    atol, rtol = get_tolerance_xla_rust_consistency(_SOURCE_ZONES_BY_QUANTITY[quantity], "medium")
    for name in sorted(required):
        field = "direct" if expected[name].direct is not None else "d_p"
        for component in ("direct", "d_p", "d_pbar"):
            assert (getattr(actual[name], component) is None) == (getattr(expected[name], component) is None), name
        reference = np.asarray(getattr(expected[name], field))
        value = np.asarray(getattr(actual[name], field))
        assert value.shape == reference.shape, name
        assert np.iscomplexobj(reference) and np.iscomplexobj(value), name
        assert np.all(np.isfinite(value)) and np.all(np.isfinite(reference)), name
        np.testing.assert_allclose(value, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=name)
        assert np.any(np.abs(reference.imag) > atol + rtol * np.abs(reference)), name
        diagnostic_reference = (
            reference if np.iscomplexobj(getattr(ham.wavefunction_data.geminal_data, name)) else reference.real
        )
        np.testing.assert_allclose(
            diagnostic[name], diagnostic_reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"diagnostic/{name}"
        )
    assert (
        _rs_pd_complex_gap(ham, _MASK, ecp_flag=bool(ham.coulomb_potential_data.ecp_flag), want_e_L=quantity == "e_L") is None
    )
