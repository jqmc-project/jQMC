"""Native molecular ECP GFMC refreshes a moved down-spin electron with the down-spin orbitals.

When a down-spin electron hops, the native ECP-mesh fast state of the GFMC
projection chunk (``combined_molecular_det_refresh`` in ``jqmc/rs/src/gfmc.rs``)
re-evaluates the six kinetic-lattice rows and the ECP quadrature rows of that
electron through the occupied down-spin MO coefficients. Closed-shell restricted
fixtures carry identical up- and down-spin orbital data, so reading the wrong
spin block there is an exact no-op (issue #575). This row uses ccECP H2 with a
broken-symmetry unrestricted Slater determinant: the up-spin electron occupies
the fixture's bonding MO and the down-spin electron an AO-normalized MO
polarized towards the first nucleus. Every down-spin refresh therefore reads
coefficients that differ from the up-spin ones while N_up == N_dn (a wrong spin
block cannot hide behind a shape error).

Reference: the production XLA drivers ``GFMC_n`` / ``GFMC_t`` with
``JQMC_FLAG_RUST_ON_CPU=0`` at the same seed and random mesh; the XLA chain
shares no code with the native candidate-table refresh.
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

import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import (
    ATOL_PYTHON_FP64_LOOSE,
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_WARMUP_STEPS,
    RTOL_PYTHON_FP64_ZERO,
)
from jqmc.atomic_orbital import compute_overlap_matrix
from jqmc.determinant import SD_data
from tests.test_rs_kernel.test_gfmc_chain_modes import (
    _STEPS_STAT,
    _build_ham,
    _install_ctx_spy,
    _run_gfmc,
)

pytestmark = [pytest.mark.with_chdir]

_STEM = "H2_ecp_ccpvdz_cart.h5"
_SEED = 34356
# Down-spin orbital: the bonding MO with the AO coefficients of the second
# nucleus scaled by this factor, then normalized in the AO metric.
_DN_SECOND_CENTRE_SCALE = 0.4
# Zones crossed by a deterministic end-to-end GFMC value comparison (the same
# set as ``test_gfmc_chain_modes._PARITY_ZONES``), judged at the strict tier.
_PARITY_ZONES = ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic", "swct")
# gfmc_route_counters index 4: molecular-GTO candidate-table refreshes.
_CAND_REFRESH_MOL = 4


def _unrestricted_h2_ecp_ham():
    """ccECP H2 (one up, one down) with an independent, polarized down-spin orbital."""
    ham = _build_ham(_STEM)
    geminal = ham.wavefunction_data.geminal_data
    assert isinstance(geminal, SD_data) and (geminal.num_electron_up, geminal.num_electron_dn) == (1, 1)
    aos = geminal.orb_data_dn_spin.aos_data
    c_up = np.asarray(geminal.orb_data_up_spin.mo_coefficients)
    centre = np.asarray(aos.nucleus_index)
    assert set(centre.tolist()) == {0, 1}
    c_dn = np.where(centre == 1, _DN_SECOND_CENTRE_SCALE, 1.0)[None, :] * c_up
    c_dn = c_dn / np.sqrt(float((c_dn @ np.asarray(compute_overlap_matrix(aos)) @ c_dn.T)[0, 0]))
    sd = SD_data(
        num_electron_up=1,
        num_electron_dn=1,
        orb_data_up_spin=geminal.orb_data_up_spin,
        orb_data_dn_spin=geminal.orb_data_dn_spin.replace(num_mo=1, mo_coefficients=c_dn),
        time_reversal=False,
    )
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=sd))
    ham.sanity_check()
    assert ham.coulomb_potential_data.ecp_flag
    assert np.any(c_dn != c_up), "down-spin orbital must differ from the up-spin orbital"
    return ham


def _assert_strict_driver_parity(rust, xla, driver):
    """Per-step and public GFMC values at the strict XLA-Rust tier; walk identity."""
    atol, rtol = get_tolerance_xla_rust_consistency(_PARITY_ZONES, level="strict")
    for name in ("_stored_w_L", "_stored_e_L", "_stored_e_L2"):
        native_values = np.asarray(getattr(rust, name))
        xla_values = np.asarray(getattr(xla, name))
        np.testing.assert_allclose(
            native_values,
            xla_values,
            atol=atol,
            rtol=rtol,
            err_msg=f"GFMC_{driver} {name}: native vs XLA (unrestricted ccECP H2)",
        )
    args = (GFMC_MIN_WARMUP_STEPS, GFMC_MIN_BIN_BLOCKS)
    native_energy = np.asarray(rust.get_E(*args))
    xla_energy = np.asarray(xla.get_E(*args))
    np.testing.assert_allclose(
        native_energy,
        xla_energy,
        atol=atol,
        rtol=rtol,
        err_msg=f"GFMC_{driver} public energy: native vs XLA (unrestricted ccECP H2)",
    )
    # Threefry keys and discrete decisions are identical; the random-mesh
    # rotated displacement may differ by final ulps (native libm vs XLA sin/cos).
    for name in ("_latest_r_up_carts", "_latest_r_dn_carts"):
        native_positions = np.asarray(getattr(rust, name))
        xla_positions = np.asarray(getattr(xla, name))
        np.testing.assert_allclose(
            native_positions,
            xla_positions,
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_ZERO,
            err_msg=f"GFMC_{driver} {name}",
        )
    np.testing.assert_array_equal(rust._jax_PRNG_key_list, xla._jax_PRNG_key_list)


@pytest.mark.light
@pytest.mark.parametrize("driver", ["n", "t"])
def test_gfmc_unrestricted_ecp_down_spin_refresh_matches_xla(driver, monkeypatch):
    """Native GFMC_n/t on an unrestricted ccECP SD reproduce the XLA chain step by step."""
    ham = _unrestricted_h2_ecp_ham()
    xla = _run_gfmc(driver, ham, _SEED, "jax", _STEPS_STAT, random_mesh=True, num_walkers=1)
    counts = _install_ctx_spy(monkeypatch)
    jqmc_rs.gfmc_route_counters_reset()
    rust = _run_gfmc(driver, ham, _SEED, "rust", _STEPS_STAT, random_mesh=True, num_walkers=1)

    key = f"run_gfmc_projection_{driver}"
    assert counts[key] >= _STEPS_STAT, f"native GFMC chunk did not engage: {counts}"
    refreshes = int(jqmc_rs.gfmc_route_counters()[_CAND_REFRESH_MOL])
    assert refreshes > 0, "no molecular candidate-table refresh ran, so no moved electron was re-evaluated"
    _assert_strict_driver_parity(rust, xla, driver)
