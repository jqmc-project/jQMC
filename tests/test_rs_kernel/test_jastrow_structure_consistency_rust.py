"""Rust: Jastrow component periodicity must match the Hamiltonian structure (issue #147).

The native ``JastrowOneBodyData`` / ``JastrowTwoBodyData`` / DTN parsers
dispatch on the component's *own* ``structure_data.pbc_flag``. Every periodic
Hamiltonian accepted by ``Hamiltonian_data.build`` must therefore give a
lattice-periodic native Jastrow value and gradient/Laplacian that are
unchanged by save/load, and a Rust-chain MCMC run must restart from its
checkpoint with the same wavefunction (or be rejected up front).

Independent references: lattice-translation invariance of a Gamma-point
periodic Jastrow, J(r_i + a_k) = J(r_i), and identity across save/load.
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

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance_min
from jqmc._setting import get_tol_hdf5_roundtrip
from jqmc.hamiltonians import Hamiltonian_data
from tests.test_jqmc_kernel.test_jastrow_structure_consistency import (
    _PERIODIC_ACCEPTED,
    _PERIODIC_CANDIDATES,
    _build,
    _cell,
    _configurations,
    _periodic_hamiltonian_unchecked,
    mcmc_restart_ln_psi,
)

_ZONES = ("jastrow_eval", "jastrow_grad_lap")


def _native_jastrow(jastrow_data, r_up, r_dn) -> np.ndarray:
    """Native J value (J1 + J2 + DTN) followed by the flattened J gradients and Laplacians."""
    value = 0.0
    if jastrow_data.jastrow_one_body_data is not None:
        value += float(jqmc_rs.compute_Jastrow_one_body(jastrow_data.jastrow_one_body_data, r_up, r_dn))
    if jastrow_data.jastrow_two_body_data is not None:
        value += float(jqmc_rs.compute_Jastrow_two_body(jastrow_data.jastrow_two_body_data, r_up, r_dn))
    if jastrow_data.jastrow_dtn_data is not None:
        value += float(jqmc_rs.compute_dtn_value(jastrow_data.jastrow_dtn_data, r_up, r_dn))
    grads = jqmc_rs.compute_grads_and_laplacian_jastrow_part(jastrow_data, r_up, r_dn)
    return np.concatenate([[value]] + [np.asarray(x, dtype=np.float64).ravel() for x in grads])


@pytest.mark.light
@pytest.mark.parametrize("case", _PERIODIC_CANDIDATES)
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="seeded-random",
    reason="PCG64 seed 7, four uniform fractional positions in the H2_pbc_ae_cart cell for each spin.",
)
def test_rust_accepted_periodic_jastrow_is_lattice_periodic_and_reload_invariant(tmp_path, case):
    """Native Jastrow of every accepted periodic Hamiltonian is lattice periodic and survives save/load.

    Same candidates as the XLA test on H2_pbc_ae_cart.h5. Hamiltonian_data.build
    either rejects the input with ValueError, or the native J value
    (compute_Jastrow_one_body + compute_Jastrow_two_body + compute_dtn_value) and
    compute_grads_and_laplacian_jastrow_part are invariant under every lattice
    vector applied to either electron, in memory and after
    save_to_hdf5/load_from_hdf5, and equal across the reload. Consistent
    candidates must be accepted. An accepted default-structure J2 would change the
    native J gradient by 4.9e-2 under translation and 8.5e-2 on reload.
    """
    ham = _periodic_hamiltonian_unchecked(case)
    try:
        built = _build(ham)
    except ValueError:
        assert case not in _PERIODIC_ACCEPTED, f"{case}: a consistent periodic Hamiltonian was rejected"
        return
    path = str(tmp_path / "ham.h5")
    built.save_to_hdf5(path)
    reloaded = Hamiltonian_data.load_from_hdf5(path)

    atol, rtol = get_tolerance_min(_ZONES, "strict")
    rt_atol, rt_rtol = get_tol_hdf5_roundtrip("strict")
    cell = _cell(built.structure_data)
    jas_mem = built.wavefunction_data.jastrow_data
    jas_rel = reloaded.wavefunction_data.jastrow_data
    for r_up, r_dn in _configurations(built.structure_data):
        base = _native_jastrow(jas_mem, r_up, r_dn)
        assert np.all(np.isfinite(base)), case
        np.testing.assert_allclose(
            _native_jastrow(jas_rel, r_up, r_dn), base, atol=rt_atol, rtol=rt_rtol, err_msg=f"{case}: reload"
        )
        for k in range(3):
            for label, (up, dn) in (("up", (r_up + cell[k], r_dn)), ("dn", (r_up, r_dn + cell[k]))):
                for name, jas in (("memory", jas_mem), ("reloaded", jas_rel)):
                    np.testing.assert_allclose(
                        _native_jastrow(jas, up, dn),
                        _native_jastrow(jas, r_up, r_dn),
                        atol=atol,
                        rtol=rtol,
                        err_msg=f"{case}: {name} native J not invariant under a_{k} on the {label} electron",
                    )


@pytest.mark.heavy
@pytest.mark.parametrize("case", ["j2_pade_periodic", "j2_pade_default_structure", "j1_molecular_structure"])
def test_rust_mcmc_checkpoint_restart_keeps_periodic_jastrow_or_rejects(tmp_path, monkeypatch, case):
    """A Rust-chain MCMC run on a periodic Hamiltonian restarts with the same ln Psi, or is rejected up front.

    As the XLA test, with JQMC_FLAG_RUST_ON_CPU=1 so the MCMC chain runs on the
    native backend. An accepted default-structure J2 would change the restarted
    ln Psi by up to 1.42 per walker.
    """
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    result = mcmc_restart_ln_psi(case, tmp_path)
    if result is None:
        assert case not in _PERIODIC_ACCEPTED, f"{case}: a consistent periodic Hamiltonian was rejected"
        return
    before, after = result
    assert np.all(np.isfinite(before)), case
    atol, rtol = get_tol_hdf5_roundtrip("strict")
    np.testing.assert_allclose(after, before, atol=atol, rtol=rtol, err_msg=f"{case}: restarted ln Psi")
