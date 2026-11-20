"""Deterministic Rust-vs-XLA MCMC driver parity with a live non-local l >= 1 ECP.

The committed Cl2 ccECP has non-zero non-local s AND p projectors on both atoms; every
other Rust MCMC driver row uses an s-only (Li, water) or empty (H, H2) non-local part.
The molecular real force run fuses HF/Pulay force into the native chunk, so the XLA
proposal normals and meshes are injected through the ``run_energy_chunk_batch`` test seam
(``test_mcmc_exotic_leaf_value_parity._inject_xla_inputs``).  The walk must be identical
(accepted moves and keys exactly; final walkers differ by at most one fp64 ulp here, so the
MCMC production-vs-debug walker pin ``ATOL_PYTHON_FP64_MEDIUM`` applies),
then stored ``w_L`` / ``e_L`` / ``e_L2`` / HF+Pulay force and the public energy / force
estimators agree at ``get_tolerance_xla_rust_consistency`` (medium), each precision mode
within itself.  Non-vacuity: the l >= 1 non-local contribution at every final walker
exceeds the e_L envelope.  Negative control: a native e_L scaled by 1 + 10 rtol fails.
The stored native e_L of the last step must also equal kinetic + bare Coulomb + local
ECP + the raw-ECP-array non-local reference (terms selected from the raw ``ang_moms``,
scipy Legendre) at each final walker, so a slip in the non-local term selection shared
by both backends fails this row.
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

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from tests._finite_allclose import assert_finite_allclose
from jqmc._precision import get_tolerance_squared_observable, get_tolerance_xla_rust_consistency
from jqmc._setting import MCMC_MIN_BIN_BLOCKS, MCMC_MIN_STEPS
from jqmc.jqmc_mcmc import MCMC
from tests._ecp_high_l import (
    assert_fixture_has_live_high_l_channel,
    assert_mcmc_stored_e_L_matches_raw_arrays,
    high_l_hamiltonian,
    high_l_nonlocal_contribution,
)
from tests.test_rs_kernel.conftest import assert_native_ran, assert_no_native_ran, native_ctx_spy
from tests.test_rs_kernel.test_mcmc_exotic_leaf_value_parity import _ZONES, _assert_same_walk, _assert_value, _inject_xla_inputs

_NW = 2
_WARMUP = MCMC_MIN_BIN_BLOCKS * 3
_STORED = ("w_L", "e_L", "e_L2", "force_HF_stored", "force_PP_stored")


def _run(ham, rust_on, e_L_scale=None):
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
            comput_position_deriv=True,
            use_swct=True,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )
        if rust_on:
            with _inject_xla_inputs(e_L_scale=e_L_scale) as seen, native_ctx_spy() as counts:
                driver.run(num_mcmc_steps=MCMC_MIN_STEPS)
            assert_native_ran(counts, "Cl2 Rust MCMC")
            assert seen["chunk"] >= 1, f"molecular real force run must use the injected native chunk: {seen}"
        else:
            with native_ctx_spy() as counts:
                driver.run(num_mcmc_steps=MCMC_MIN_STEPS)
            assert_no_native_ran(counts, "Cl2 XLA MCMC")
        return driver
    finally:
        if previous is None:
            os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
        else:
            os.environ["JQMC_FLAG_RUST_ON_CPU"] = previous


@pytest.mark.heavy
def test_mcmc_rust_vs_xla_high_l_ecp_values():
    """Same seed, same walk: Rust == XLA MCMC values on the live p-channel Cl2 ccECP."""
    ham = high_l_hamiltonian()
    assert_fixture_has_live_high_l_channel(ham)
    xla = _run(ham, rust_on=False)
    rust = _run(ham, rust_on=True)
    _assert_same_walk(rust, xla, "Cl2_high_l", exact_positions=False)
    # Stored native e_L against the raw-ECP-array reference (independent of both backends' term selection).
    assert_mcmc_stored_e_L_matches_raw_arrays(
        ham, rust, *get_tolerance_xla_rust_consistency(_ZONES, level="medium"), "Cl2_high_l Rust MCMC"
    )
    for name in _STORED:
        r, x = np.asarray(getattr(rust, name)), np.asarray(getattr(xla, name))
        assert r.size and np.all(np.isfinite(r)) and np.all(np.isfinite(x)), f"Cl2_high_l: {name} empty or not finite"
        if name == "e_L2":
            # |e_L|^2 doubles the relative error of e_L: the repo's squared-observable image.
            atol, rtol = get_tolerance_squared_observable(*get_tolerance_xla_rust_consistency(_ZONES, level="medium"))
            assert_finite_allclose(r, x, atol=atol, rtol=rtol, err_msg="Cl2_high_l: MCMC stored e_L2")
        else:
            _assert_value(r, x, f"Cl2_high_l: MCMC stored {name}")
    args = (_WARMUP, MCMC_MIN_BIN_BLOCKS)
    _assert_value(rust.get_E(*args), xla.get_E(*args), "Cl2_high_l: MCMC public energy")
    _assert_value(rust.get_aF(*args), xla.get_aF(*args), "Cl2_high_l: MCMC public force")
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    delta = high_l_nonlocal_contribution(ham, rust._MCMC__latest_r_up_carts, rust._MCMC__latest_r_dn_carts)
    envelope = atol + rtol * np.abs(np.asarray(rust.e_L)[-1])
    assert np.all(np.abs(delta) > envelope), f"l>=1 non-local contribution {delta} within the e_L envelope {envelope}"


@pytest.mark.heavy
def test_mcmc_rust_vs_xla_high_l_ecp_rejects_scaled_native_e_L():
    """Negative control: native e_L * (1 + 10 rtol) on the same walk fails the e_L comparison."""
    ham = high_l_hamiltonian()
    _, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    xla = _run(ham, rust_on=False)
    rust = _run(ham, rust_on=True, e_L_scale=1.0 + 10.0 * rtol)
    _assert_same_walk(rust, xla, "Cl2_high_l", exact_positions=False)
    with pytest.raises(AssertionError, match="Cl2_high_l: MCMC stored e_L"):
        _assert_value(rust.e_L, xla.e_L, "Cl2_high_l: MCMC stored e_L")
    _assert_value(rust.w_L, xla.w_L, "Cl2_high_l: MCMC stored w_L")
