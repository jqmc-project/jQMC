"""The native molecular ECP chunk uses independent down-spin orbitals."""

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

import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import get_tolerance_min
from jqmc._setting import Nv_default
from jqmc.coulomb_potential import (
    _compute_ecp_non_local_parts_nearest_neighbors_debug,
    compute_ecp_non_local_parts_nearest_neighbors_fast_update,
)
from jqmc.hamiltonians import compute_local_energy
from jqmc.jqmc_mcmc import _geminal_inv_batched
from tests.test_rs_kernel.test_chain_modes import _assert_value_parity
from tests.test_rs_kernel.test_ecp_j3_wdot_mo_parity import (
    _NW,
    _STEPS,
    _rotation,
    _run_native_chunk,
    _water_mo_j3_ham,
)


@pytest.mark.with_chdir
@pytest.mark.heavy
@pytest.mark.parametrize("n_down", [4, 3], ids=["4up-4dn", "4up-3dn"])
def test_unrestricted_ecp_wdot_uses_down_spin_orbitals(n_down, monkeypatch):
    """The fast chunk agrees with XLA at each returned walker on the same rotated mesh."""
    ham = _water_mo_j3_ham(n_mo=40)
    wavefunction = ham.wavefunction_data
    gd = wavefunction.geminal_data
    down = gd.orb_data_dn_spin
    c_down = np.asarray(down.mo_coefficients[:n_down])
    rng = np.random.default_rng(5)
    independent_coeffs = c_down + 0.15 * np.max(np.abs(c_down)) * rng.standard_normal(c_down.shape)
    gd = gd.replace(
        num_electron_dn=n_down,
        orb_data_dn_spin=down.replace(num_mo=n_down, mo_coefficients=independent_coeffs),
        time_reversal=False,
    )
    wavefunction = wavefunction.replace(geminal_data=gd)
    ham = ham.replace(wavefunction_data=wavefunction)
    ham.sanity_check()
    assert gd.num_electron_up == 4 and gd.num_electron_dn == n_down and not gd.time_reversal
    assert np.any(independent_coeffs != np.asarray(gd.orb_data_up_spin.mo_coefficients[:n_down]))

    rng = np.random.default_rng(20261104)
    rotations = [_rotation(rng) for _ in range(_STEPS)]
    # The shared chunk harness supports eight proposals with either spin count.
    normals = rng.standard_normal((_NW, _STEPS, 8))
    monkeypatch.delenv("JQMC_FLAG_RUST_ON_CPU", raising=False)
    e_native, r_up, r_down = _run_native_chunk(ham, rotations, normals)
    # The determinant w-dot decision is made once per spin for every energy.
    assert tuple(jqmc_rs.route_predicate_counters()[:2]) == (2 * _NW * _STEPS, 0)

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    e_xla = np.array(
        [
            [
                compute_local_energy(ham, jnp.asarray(r_up[w, k]), jnp.asarray(r_down[w, k]), jnp.asarray(rotations[k]))
                for k in range(_STEPS)
            ]
            for w in range(_NW)
        ]
    )
    assert np.all(np.isfinite(e_xla))

    # Anchor the XLA fast-update ECP reference to full determinant evaluation
    # at the same quadrature points for one of the returned configurations.
    ru, rd, rotation = r_up[0, 0], r_down[0, 0], rotations[0]
    _, inverse, _, _ = _geminal_inv_batched(gd, jnp.asarray(ru[None]), jnp.asarray(rd[None]))
    potential = ham.coulomb_potential_data
    v_fast = compute_ecp_non_local_parts_nearest_neighbors_fast_update(
        potential, wavefunction, jnp.asarray(ru), jnp.asarray(rd), jnp.asarray(rotation), inverse[0], NN=1, Nv=Nv_default
    )[3]
    v_full_det = _compute_ecp_non_local_parts_nearest_neighbors_debug(
        potential, wavefunction, ru, rd, rotation, NN=1, Nv=Nv_default
    )[3]
    atol, rtol = get_tolerance_min(["coulomb", "jastrow_eval", "jastrow_ratio", "det_eval", "det_ratio"], "strict")
    assert np.isfinite(v_full_det) and abs(v_full_det) > atol
    np.testing.assert_allclose(v_fast, v_full_det, atol=atol, rtol=rtol)
    _assert_value_parity(e_native, e_xla, f"unrestricted down-spin ECP w-dot for 4 up and {n_down} down")
