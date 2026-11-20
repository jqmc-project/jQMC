"""Full physical energy derivatives of frozen tiled SDs with finite supercell Jastrow."""

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
from jqmc.derivatives import assemble, _auto
from jqmc.determinant import SD_data
from jqmc.hamiltonians import compute_local_energy
from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction
from tests.test_jqmc_kernel.test_orbital_tiling_j3 import _supercell_j3, _positions
from tests.test_jqmc_kernel.test_orbital_tiling_supercell import _source, _expanded_mos

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]


@pytest.mark.parametrize("backend", ["bspline3", "pgto_cart", "pgto_sphe"])
@pytest.mark.parametrize("restricted", [True, False])
def test_tiled_full_energy_force_stress_with_nonzero_jastrow(backend, restricted):
    """Every raw force/stress component matches explicit SC and AD with finite J1/J2/J3."""
    from tests.test_jqmc_kernel._orbital_tiling_reference import complete_pgto_source, explicit_supercell_pgto

    pgto = backend.startswith("pgto")
    source = complete_pgto_source(backend) if pgto else _source(backend, "general")
    down = (
        None
        if restricted
        else (complete_pgto_source(backend, independent=True) if pgto else _source(backend, "general", independent_down=True))
    )
    reference = explicit_supercell_pgto if pgto else lambda orb: _expanded_mos(orb, backend)
    sd = SD_data.build_tiled(up_source=source, down_source=down)
    ham = build_hamiltonian_from_wavefunction(
        source.structure_data,
        source,
        sd,
        j1_param=0.5,
        j2_param=0.7,
        jastrow_three_body_data=_supercell_j3(source.structure_data),
    )
    explicit_sd = SD_data.build(
        num_electron_up=2,
        num_electron_dn=2,
        mos_up=reference(sd.orb_data_up_spin),
        mos_dn=reference(sd.orb_data_dn_spin),
        time_reversal=False,
    )
    explicit = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=explicit_sd))
    up, dn = map(jnp.asarray, _positions(ham.structure_data))
    atol, rtol = get_tolerance_min(
        (
            "tiling_eval",
            "tiling_grad",
            "tiling_lap",
            "ao_eval",
            "ao_grad_lap",
            "det_grad_lap",
            "jastrow_grad_lap",
            "local_energy",
        ),
        "medium",
    )
    np.testing.assert_allclose(
        compute_local_energy(ham, up, dn, jnp.eye(3)), compute_local_energy(explicit, up, dn, jnp.eye(3)), atol=atol, rtol=rtol
    )
    for coordinate in ("R", "eps"):
        for name in ("compute_grad_ln_Psi", "compute_grad_e_L"):
            args = (up, dn) if name == "compute_grad_ln_Psi" else (up, dn, jnp.eye(3))
            actual = getattr(assemble, name)(ham, *args, wrt=coordinate)
            wanted = getattr(assemble, name)(explicit, *args, wrt=coordinate)
            auto = getattr(_auto, name)(ham, *args, wrt=coordinate)
            np.testing.assert_allclose(actual, wanted, atol=atol, rtol=rtol)
            np.testing.assert_allclose(actual, auto, atol=atol, rtol=rtol)
