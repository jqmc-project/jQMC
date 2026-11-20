"""Multi-electron complex reference ratios survive underflow of the raw determinant."""

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

import numpy as np
import pytest
import jqmc_rs

from jqmc._precision import get_tolerance, get_tolerance_xla_rust_consistency, precision_policy_snapshot
from jqmc.hamiltonians import compute_local_energy
from jqmc.jastrow_factor import Jastrow_data
from tests.test_rs_kernel.test_periodic_lsda import _build_twist_pgto
from tests.test_rs_kernel.test_complex_tiny_geminal_native import (
    _SCALE,
    _ZONES,
    _assert_finite,
    _base_ham,
    _free_route,
    _geminal,
    _mesh,
    _scaled,
    _start,
)

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]

CASES = (
    ("h4_k025_ae", "lrdmc_kinetic_reference"),
    ("h4_k025_ae", "lrdmc_kinetic_fast"),
    ("oxygen_k025_ccecp", "lrdmc_ecp_reference"),
    ("oxygen_k025_ccecp", "lrdmc_ecp_fast"),
    ("oxygen_k025_ccecp", "free_ecp"),
    ("oxygen_k025_ccecp", "e_L_reference_ecp"),
    ("oxygen_k025_ccecp", "ecp_force"),
    ("oxygen_k025_ccecp", "ecp_strain"),
    ("oxygen_k025_ccecp", "gfmc_hf_force_stress"),
)


def _ham(witness, tmp_path):
    if witness == "h4_k025_ae":
        ham = _base_ham(witness)
        ham.sanity_check()
        return ham
    ham = _build_twist_pgto(tmp_path)
    wf = ham.wavefunction_data
    gem = wf.geminal_data

    def phased(orb):
        coefficients = np.asarray(orb.mo_coefficients, dtype=np.complex128)
        phase = np.exp(1j * np.arange(len(coefficients)))[:, None]
        return orb.replace(mo_coefficients=coefficients * phase)

    gem = gem.replace(orb_data_up_spin=phased(gem.orb_data_up_spin), orb_data_dn_spin=phased(gem.orb_data_dn_spin))
    changed = ham.replace(wavefunction_data=wf.replace(geminal_data=gem, jastrow_data=Jastrow_data()))
    changed.sanity_check()
    assert np.iscomplexobj(np.asarray(gem.orb_data_up_spin.mo_coefficients))
    return changed


def _positions(ham, witness):
    if witness == "h4_k025_ae":
        return _start(ham)
    rng = np.random.default_rng(13)
    g = ham.wavefunction_data.geminal_data
    centers = np.asarray(ham.structure_data.positions, dtype=float)
    return tuple(
        np.ascontiguousarray(centers[np.arange(n) % len(centers)] + rng.normal(scale=0.6, size=(n, 3)))
        for n in (g.num_electron_up, g.num_electron_dn)
    )


def _evaluate(ham, up, dn, route, monkeypatch):
    gd, cd = ham.wavefunction_data.geminal_data, ham.coulomb_potential_data
    w, gp = _mesh()
    with jqmc_rs.NativeConstructionPrecision(precision_policy_snapshot()):
        if route == "free_ecp":
            return {"values": np.asarray(jqmc_rs.compute_ecp_nonlocal_complex(gd, cd, up, dn, w, gp))}
        if route == "e_L_reference_ecp":
            with monkeypatch.context() as scoped:
                scoped.setenv("JQMC_RS_ECP_COMPLEX_REFERENCE", "1")
                inverse = np.ascontiguousarray(np.linalg.inv(_geminal(ham, up, dn)))
                return {"values": np.asarray(jqmc_rs.compute_local_energy_fast_complex(ham, up, dn, w, gp, inverse))}
        return _free_route(ham, up, dn, route)


@pytest.mark.parametrize("witness,route", CASES, ids=[f"{witness}-{route}" for witness, route in CASES])
def test_multielectron_complex_reference_routes_keep_determinant_ratios_representable(witness, route, monkeypatch, tmp_path):
    """Reference and control routes preserve same-policy MO scale invariance for n>=2."""
    base = _ham(witness, tmp_path)
    up, dn = _positions(base, witness)
    scaled = _scaled(base, _SCALE)
    scaled.sanity_check()
    matrix = _geminal(scaled, up, dn)
    assert matrix.shape[0] >= 2
    phase, logabs = np.linalg.slogdet(matrix)
    assert np.isfinite(logabs) and phase != 0
    assert logabs < np.log(np.nextafter(0.0, 1.0)), "the determinant itself must be unrepresentable"
    assert np.all(np.isfinite(np.linalg.inv(matrix)))
    expected = _evaluate(base, up, dn, route, monkeypatch)
    actual = _evaluate(scaled, up, dn, route, monkeypatch)
    _assert_finite(expected, "unscaled")
    _assert_finite(actual, "scaled")
    assert np.any(np.abs(expected["values"]) > 0)
    atol, rtol = get_tolerance("det_ratio", "strict")
    np.testing.assert_allclose(actual["values"], expected["values"], atol=atol, rtol=rtol, equal_nan=False)
    if route == "e_L_reference_ecp":
        # Scale invariance supplies a well-conditioned independent reference.
        # The scaled open-shell XLA inverse has its own range failure; it is
        # not a valid oracle for this native determinant-ratio repair.
        xla = complex(compute_local_energy(base, up, dn, np.eye(3)))
        native = complex(*actual["values"])
        xatol, xrtol = get_tolerance_xla_rust_consistency(_ZONES)
        np.testing.assert_allclose(native, xla, atol=xatol, rtol=xrtol, equal_nan=False)
