"""Real native AGPn total-energy optimizer sources must include the ECP term."""

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
import numpy as np
import pytest

import jqmc_rs
from jqmc import derivatives
from jqmc._precision import get_tolerance_xla_rust_consistency, precision_policy_snapshot
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from jqmc.derivatives import bridge
from jqmc.jqmc_mcmc import _leaf_to_fp64, _rs_native_c_param_grads
from tests.test_jqmc_kernel.test_agpn_gamma_ecp_reference import (
    _gamma_agpn_case,
    _gamma_agpn_lambda_blocks,
    _gamma_agpn_primal_derivatives,
    _gamma_agpn_xla_sources,
)

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]
_ZONES = ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic", "jastrow_grad_lap", "coulomb")
_MASK = {"opt_geminal": True}


@pytest.fixture(autouse=True)
def _clear_caches():
    yield
    bridge.cache_clear()
    derivatives._jitted_assemble.cache_clear()
    jax.clear_caches()


@pytest.mark.parametrize("fixture_path", ["tests/qe_example_files/O-lsda_gamma/out/o.save"], ids=["oxygen-gamma"])
@pytest.mark.parametrize("leaf", ["lambda_diag", "lambda_unpaired"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_native_agpn_gamma_ecp_lambda_source(tmp_path, fixture_path, leaf):
    """Native real AGPn optimizer lambda sources include nonlocal ECP derivatives.

    The independently anchored O-lsda_gamma pGTO ccECP fixture has four up/two
    down electrons, no Jastrow, common-rank AGPn, real parameter seed 431, and
    electron seed 41: one Gaussian(scale=0.6) configuration, no cond(G) screen.
    Use the fixed twelve-point ECP rule and identity rotation. Production input
    transport widens already-rounded arrays, retaining the selected native zone
    policy. Compare the unmodified production assembler's published direct source
    with the XLA analytical source, including membership, component presence,
    shape and finite values. The ECP effect must exceed the same comparison band.
    No free-kernel correction is added to the actual native result.
    """
    ham, r_up, r_dn = _gamma_agpn_case(tmp_path, fixture_path)
    blocks = _gamma_agpn_lambda_blocks(ham.wavefunction_data)
    expected = {source.name: source for source in _gamma_agpn_xla_sources(ham, r_up, r_dn, blocks)}
    total_ad, kinetic_ad = _gamma_agpn_primal_derivatives(ham, r_up, r_dn)
    native_ham = jax.tree_util.tree_map(_leaf_to_fp64, ham)
    native_wf = native_ham.wavefunction_data
    policy = precision_policy_snapshot()
    ctx = jqmc_rs.build_obs_ctx(
        native_wf.geminal_data,
        native_wf.jastrow_data,
        native_ham.coulomb_potential_data,
        False,
        True,
        False,
        want_force=False,
        want_gfmc=False,
        want_param_deriv=True,
        precision_policy=policy,
    )
    assert ctx.precision_policy() == policy
    sources = _rs_native_c_param_grads(
        ctx,
        ham,
        [np.asarray(r_up)],
        [np.asarray(r_dn)],
        _MASK,
        1,
        quantity="e_L",
        ecp_flag=True,
        ecp_w=np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64),
        ecp_gp=np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64),
        rts_b=[np.eye(3)],
        source_blocks=blocks,
    )
    published = {source.name: source for source in sources}
    assert set(published) == set(expected) == {"lambda_diag", "lambda_unpaired"}
    for component in ("direct", "d_p", "d_pbar"):
        assert (getattr(published[leaf], component) is None) == (getattr(expected[leaf], component) is None)
    assert published[leaf].direct is not None and expected[leaf].direct is not None
    actual = np.asarray(published[leaf].direct)[0]
    reference = np.asarray(expected[leaf].direct)[0]
    kinetic = np.asarray(kinetic_ad[leaf])
    total = np.asarray(total_ad[leaf])
    assert actual.shape == reference.shape == kinetic.shape == total.shape and reference.size > 0
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    assert np.all(np.isfinite(kinetic)) and np.all(np.isfinite(total))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, "medium")
    assert np.any(np.abs(total - kinetic) > atol + rtol * np.abs(reference))
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=leaf)
