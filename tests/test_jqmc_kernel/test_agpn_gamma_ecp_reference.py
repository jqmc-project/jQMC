"""Same-precision local-energy references for ECP-sensitive real AGPn leaves."""

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
from jqmc._precision import get_tolerance_min, get_tolerance_xla_rust_consistency, is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from jqmc.derivatives import bridge
from jqmc.hamiltonians import compute_local_energy
from jqmc.wavefunction import compute_kinetic_energy
from tests.test_jqmc_kernel.optimizer_metadata_helpers import build_explicit_common_rank_agpn_from_sd
from tests.test_rs_kernel.test_periodic_lsda import _build_gamma, _config

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]
_ZONES = ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic", "jastrow_grad_lap", "coulomb")
_MASK = {"opt_geminal": True}


@pytest.fixture(autouse=True)
def _clear_caches():
    yield
    bridge.cache_clear()
    derivatives._jitted_assemble.cache_clear()
    jax.clear_caches()


def _gamma_agpn_case(tmp_path, fixture_path):
    """Select one committed Gamma input and make both occupation leaves active."""
    if fixture_path != "tests/qe_example_files/O-lsda_gamma/out/o.save":
        raise ValueError(f"Unknown real AGPn ECP fixture: {fixture_path}")
    ham = _build_gamma("pgto", tmp_path)
    wf = ham.wavefunction_data
    gem = build_explicit_common_rank_agpn_from_sd(wf.geminal_data)
    rng = np.random.default_rng(431)
    gem = gem.replace(
        lambda_diag=jnp.asarray(0.3 + rng.random(gem.lambda_diag.shape)),
        lambda_unpaired=jnp.asarray(np.real(gem.lambda_unpaired) + 0.2 * rng.normal(size=gem.lambda_unpaired.shape)),
    )
    ham = ham.replace(wavefunction_data=wf.replace(geminal_data=gem))
    assert (gem.num_electron_up, gem.num_electron_dn) == (4, 2)
    assert gem.lambda_diag.shape == (4,) and gem.lambda_unpaired.shape == (4, 2)
    assert not np.iscomplexobj(gem.lambda_diag) and not np.iscomplexobj(gem.lambda_unpaired)
    assert bool(ham.structure_data.pbc_flag) and bool(ham.coulomb_potential_data.ecp_flag)
    assert wf.jastrow_data is None or all(
        getattr(wf.jastrow_data, name) is None
        for name in ("jastrow_one_body_data", "jastrow_two_body_data", "jastrow_three_body_data", "jastrow_dtn_data")
    )
    r_up, r_dn = _config(ham, np.random.default_rng(41))
    return ham, jnp.asarray(r_up), jnp.asarray(r_dn)


def _gamma_agpn_primal_derivatives(ham, r_up, r_dn):
    """Differentiate total and kinetic forward energies, not analytical gradients."""
    wf = ham.wavefunction_data
    gem = wf.geminal_data
    parameters = {"lambda_diag": gem.lambda_diag, "lambda_unpaired": gem.lambda_unpaired}

    def total(parameters):
        changed = ham.replace(wavefunction_data=wf.replace(geminal_data=gem.replace(**parameters)))
        return compute_local_energy(changed, r_up, r_dn, jnp.eye(3))

    def kinetic(parameters):
        return compute_kinetic_energy(wf.replace(geminal_data=gem.replace(**parameters)), r_up, r_dn)

    return jax.jit(jax.grad(total))(parameters), jax.jit(jax.grad(kinetic))(parameters)


def _gamma_agpn_lambda_blocks(wf):
    """Publish the two real optimizer leaves with their production metadata."""
    blocks = tuple(
        block
        for block in wf.get_variational_blocks(state_token=("gamma-agpn-ecp", 0), **_MASK)
        if block.name in {"lambda_diag", "lambda_unpaired"}
    )
    assert {block.name for block in blocks} == {"lambda_diag", "lambda_unpaired"}
    assert all(block.derivative_policy == "direct_real" for block in blocks)
    return blocks


def _gamma_agpn_xla_sources(ham, r_up, r_dn, blocks):
    """Use the same analytical source provider and collector as the XLA driver."""
    grad = bridge._batched_c_source("e_L", mask=_MASK, analytic=True)(ham, r_up[None], r_dn[None], jnp.eye(3)[None])
    return bridge._collect_c_param_derivative_sources(ham.wavefunction_data, grad, blocks, 1, _MASK)


@pytest.mark.parametrize("fixture_path", ["tests/qe_example_files/O-lsda_gamma/out/o.save"], ids=["oxygen-gamma"])
@pytest.mark.parametrize("leaf", ["lambda_diag", "lambda_unpaired"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_agpn_gamma_ecp_lambda_reference(tmp_path, fixture_path, leaf):
    """Real Gamma AGPn total-energy lambda derivatives equal same-precision primal AD.

    Committed O-lsda_gamma/out/o.save with O.ccECP.upf is converted to real pGTO,
    no Jastrow, four up/two down electrons, and common-rank natural orbitals.
    PCG64 seed 431 gives nonfactorized real occupations and unpaired coefficients;
    electron seed 41 gives one Gaussian(scale=0.6) configuration around the first
    nucleus, without cond(G) rejection. The ECP rule is twelve points and identity
    rotation. Each nonempty lambda leaf is tested separately in full and mixed.
    The reference differentiates primal total and kinetic energies independently
    of the analytical parameter-gradient code, while sharing forward evaluators.
    Total versus kinetic AD must differ outside both existing comparison bands.
    """
    ham, r_up, r_dn = _gamma_agpn_case(tmp_path, fixture_path)
    total_ad, kinetic_ad = _gamma_agpn_primal_derivatives(ham, r_up, r_dn)
    blocks = _gamma_agpn_lambda_blocks(ham.wavefunction_data)
    sources = {source.name: source for source in _gamma_agpn_xla_sources(ham, r_up, r_dn, blocks)}
    assert set(sources) == {"lambda_diag", "lambda_unpaired"}
    actual = np.asarray(sources[leaf].direct)[0]
    reference = np.asarray(total_ad[leaf])
    kinetic = np.asarray(kinetic_ad[leaf])
    assert actual.shape == reference.shape == kinetic.shape and reference.size > 0
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)) and np.all(np.isfinite(kinetic))
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(_ZONES, level)
    native_atol, native_rtol = get_tolerance_xla_rust_consistency(_ZONES, "medium")
    assert np.any(np.abs(reference - kinetic) > atol + rtol * np.abs(reference))
    assert np.any(np.abs(reference - kinetic) > native_atol + native_rtol * np.abs(reference))
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=leaf)
    if is_mixed_precision_enabled():
        _gamma_agpn_mixed_fidelity(tmp_path, fixture_path, leaf)


def _gamma_agpn_assert_mixed_bound(mixed_block, full_block, atol, rtol, label):
    """Apply the central fidelity budget to one real occupation or unpaired leaf."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max(), label


def _gamma_agpn_mixed_fidelity(tmp_path, fixture_path, leaf):
    """Retain ECP sensitivity and compare independent analytical source models."""
    from jqmc.derivatives import assemble, kinetic
    from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

    floors = (
        get_tolerance_min(_ZONES, "medium"),
        get_tolerance_xla_rust_consistency(_ZONES, "medium"),
    )

    def factory():
        bridge.cache_clear()
        derivatives._jitted_assemble.cache_clear()
        directory = tmp_path / ("mixed-source" if is_mixed_precision_enabled() else "full-source")
        directory.mkdir(exist_ok=True)
        return _gamma_agpn_case(directory, fixture_path)

    def evaluate(case):
        ham, r_up, r_down = case
        wf = ham.wavefunction_data
        blocks = _gamma_agpn_lambda_blocks(wf)
        sources = {source.name: source for source in _gamma_agpn_xla_sources(ham, r_up, r_down, blocks)}
        assert set(sources) == {"lambda_diag", "lambda_unpaired"}
        actual = np.asarray(sources[leaf].direct)[0]
        gd = wf.geminal_data
        # This is the same kinetic provider and structured-leaf projector used
        # in the production local-energy assembly, not a changed Hamiltonian.
        raw_kinetic = kinetic.grad_T_wrt_lambda_withJ(gd, wf.jastrow_data, r_up, r_down)
        kinetic_gd = assemble._set_geminal_lambda_grad(
            jax.tree_util.tree_map(jnp.zeros_like, gd), gd, raw_kinetic, preserve_complex_diag=True
        )
        kinetic_source = np.asarray(getattr(kinetic_gd, leaf))
        assert actual.shape == kinetic_source.shape == getattr(gd, leaf).shape and actual.size > 0
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(kinetic_source))
        # Preserve both original ECP non-degeneracy floors at the entry-Mixed
        # values even while evaluating the independently rebuilt Full model.
        for floor_atol, floor_rtol in floors:
            assert np.any(np.abs(actual - kinetic_source) > floor_atol + floor_rtol * np.abs(actual))
        return {"source": actual, "up": r_up, "down": r_down}

    mixed, full = mixed_vs_full(factory, evaluate)
    assert np.array_equal(mixed["up"], full["up"]) and np.array_equal(mixed["down"], full["down"])
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    checked = []
    for mixed_block, full_block in mixed_fidelity_blocks(mixed["source"], full["source"]):
        _gamma_agpn_assert_mixed_bound(mixed_block, full_block, atol, rtol, f"Gamma AGPn {leaf}")
        checked.append((mixed_block, full_block))
    assert checked, "Gamma AGPn source comparison would be empty"
    faulty = out_of_budget_mixed_block(*checked[0], atol=atol, rtol=rtol)
    with pytest.raises(AssertionError):
        _gamma_agpn_assert_mixed_bound(faulty, checked[0][1], atol, rtol, "Gamma AGPn excess-error control")
