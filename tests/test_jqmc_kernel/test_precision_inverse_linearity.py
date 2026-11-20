"""An exact projector identity detects lost fp64 inverse bits within each mode."""

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

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import coulomb_potential, determinant
from jqmc._precision import get_tolerance_min
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]


@pytest.fixture(scope="module")
def copper_bromide_ecp_wavefunction():
    """Committed heavy ECP input with both Jastrow channels, unchanged setup."""
    filename = Path(__file__).resolve().parents[1] / "trexio_example_files" / "CuBr_ecp_ccpvtz_cart.h5"
    structure, _, _, _, geminal, coulomb = read_trexio_file(trexio_file=str(filename), store_tuple=True)
    j1 = Jastrow_one_body_data(
        jastrow_1b_param=1.0,
        jastrow_1b_type="exp",
        structure_data=structure,
        core_electrons=tuple(coulomb.z_cores),
    )
    j2 = Jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp")
    wf = Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2))
    return structure, coulomb, wf


@pytest.mark.parametrize("seed", list(range(100, 108)))
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_ecp_projector_preserves_running_inverse_low_bits(seed, copper_bromide_ecp_wavefunction):
    """The nonlocal projector retains the contribution of fp64 inverse low bits.

    With mesh, orbitals and Jastrow ratios fixed, each rank-one determinant
    ratio is linear in the supplied inverse: F(A)-F(H)=F(A-H), F(0)=0.
    H is the fp32-rounded part, widened back to fp64. The independently
    prescribed algebraic identity must retain the nonzero low-part response.
    It catches a hidden fp32 cast even when every output still has dtype f64.

    A is the actual geminal inverse at each of eight seed100..107 CuBr/J1/J2
    configurations. No condition or walker screening is used.
    H and A-H are arithmetic diagnostic inputs, not claims of new consistent
    walker states. Six fixed quadrature directions and NN=1 use the real
    fast-update producer; there is no finite difference, cross-mode budget,
    absolute physical ECP oracle or assertion about every fp64 instruction.
    """
    structure, coulomb, wf = copper_bromide_ecp_wavefunction
    geminal = wf.geminal_data
    rng = np.random.default_rng(seed)
    centres = np.asarray(structure.positions)

    def coordinates(count):
        return jnp.asarray(centres[rng.integers(0, len(centres), count)] + rng.standard_normal((count, 3)) * 0.8)

    r_up, r_dn = coordinates(geminal.num_electron_up), coordinates(geminal.num_electron_dn)
    inverse = jnp.linalg.inv(determinant.compute_geminal_all_elements(geminal, r_up, r_dn))
    assert inverse.dtype == jnp.float64
    high = inverse.astype(jnp.float32).astype(jnp.float64)
    low = inverse - high
    assert np.any(np.asarray(low) != 0)
    np.testing.assert_array_equal(np.asarray(high + low), np.asarray(inverse))

    def project(supplied_inverse):
        result = coulomb_potential.compute_ecp_non_local_parts_nearest_neighbors_fast_update(
            coulomb, wf, r_up, r_dn, jnp.eye(3), supplied_inverse, NN=1, Nv=6
        )
        assert result[3].dtype == jnp.float64
        return np.asarray(result[3])

    original_projection = project(inverse)
    high_projection = project(high)
    low_projection = project(low)
    zero_projection = project(jnp.zeros_like(inverse))
    assert np.isfinite([original_projection, high_projection, low_projection, zero_projection]).all()
    np.testing.assert_array_equal(zero_projection, np.zeros_like(low_projection))
    atol, rtol = get_tolerance_min(("det_ratio", "coulomb"), "strict")
    assert abs(low_projection) > atol, "the carried inverse low bits must reach this nonlocal projection"
    actual = original_projection - high_projection
    np.testing.assert_allclose(actual, low_projection, atol=atol, rtol=rtol)
