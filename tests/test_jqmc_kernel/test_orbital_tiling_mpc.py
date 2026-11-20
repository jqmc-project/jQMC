"""MPC consumes occupied tiled values without materializing global coefficients."""

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

from jqmc._precision import get_tolerance_min
from jqmc.determinant import SD_data
from jqmc.mpc import build_mpc_data, compute_mpc_potential_ee
from tests.test_jqmc_kernel._orbital_tiling_reference import explicit_supercell_pgto, explicit_supercell_pw
from tests.test_jqmc_kernel.test_mpc import _assert_close_mpc_data
from tests.test_jqmc_kernel.test_orbital_tiling_hdf5 import _hamiltonian
from tests.test_jqmc_kernel.test_orbital_tiling_j3 import _positions
from tests.test_jqmc_kernel.test_orbital_tiling_supercell import _expanded_mos
from tests.test_jqmc_kernel.test_orbital_tiling_supercell import _source

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]


@pytest.mark.parametrize("backend", ["pw", "bspline", "pgto_cart", "pgto_sphe", "composite_cart", "composite_sphe"])
@pytest.mark.parametrize("restricted", [True, False])
def test_tiled_mpc_matches_explicit_supercell_without_global_coefficients(backend, restricted, monkeypatch):
    """MPC density, tables and off-grid energy agree for both spin conventions."""
    ham = _hamiltonian(backend, restricted)
    if backend == "bspline":
        source = _source("bspline3", "general")
        down = None if restricted else _source("bspline3", "general", independent_down=True)
        sd = SD_data.build_tiled(up_source=source, down_source=down)
        from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction

        ham = build_hamiltonian_from_wavefunction(source.structure_data, source, sd, j2_param=None)
    sd = ham.wavefunction_data.geminal_data
    from tests.test_jqmc_kernel._orbital_tiling_reference import explicit_supercell_composite

    reference = {
        "composite_cart": explicit_supercell_composite,
        "composite_sphe": explicit_supercell_composite,
        "pw": explicit_supercell_pw,
        "bspline": lambda orb: _expanded_mos(orb, "bspline3"),
        "pgto_cart": explicit_supercell_pgto,
        "pgto_sphe": explicit_supercell_pgto,
    }[backend]
    explicit_sd = SD_data.build(
        num_electron_up=sd.num_electron_up,
        num_electron_dn=sd.num_electron_dn,
        mos_up=reference(sd.orb_data_up_spin),
        mos_dn=reference(sd.orb_data_dn_spin),
        time_reversal=False,
    )
    explicit = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=explicit_sd))
    expected = build_mpc_data(explicit, cutoff_ha=2.0, grid=(8, 8, 8))

    def forbidden(*args, **kwargs):
        raise AssertionError("MPC requested a global natural-orbital coefficient matrix")

    monkeypatch.setattr(SD_data, "natural_orbital_density_coefficients_jnp", forbidden)
    actual = build_mpc_data(ham, cutoff_ha=2.0, grid=(8, 8, 8))
    atol, rtol = get_tolerance_min(("tiling_eval", "ao_eval", "coulomb"), "medium")
    _assert_close_mpc_data(actual, expected, rtol, atol)
    up, down = map(jnp.asarray, _positions(ham.structure_data))
    np.testing.assert_allclose(
        compute_mpc_potential_ee(actual, up, down),
        compute_mpc_potential_ee(expected, up, down),
        atol=atol,
        rtol=rtol,
    )
