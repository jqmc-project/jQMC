"""Deterministic whole-mesh ECP route comparison with the committed water/J123 input."""

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

import numpy as np
import pytest
from jqmc import coulomb_potential as coulomb
from jqmc._precision import get_tolerance
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests.test_jqmc_kernel.test_coulomb_potential import _build_full_jastrow_data

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize("family, neighbors", [("all_pairs", None), ("nearest_neighbors", 1), ("nearest_neighbors", 3)])
@pytest.mark.parametrize("quadrature", [6, 18])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_water_j123_ecp_whole_mesh_matches_scalar_assembly(family, neighbors, quadrature):
    """Water/J123 ECP mesh coordinates, every contribution and total match scalar assembly.

    Use committed water_ccecp_ccpvqz.h5, its 4-up/4-down determinant and
    nonzero J1/J2/J3 built with PCG64 seed 31. Electron positions use the
    original PCG64 seed 0, uniform in [-1,1]^3, one configuration with no
    rejection or cond(G) screening. Test all-pairs and nearest 1/3 nuclei,
    quadrature sizes 6/18, identity rotation and the complete Jastrow ratio.

    Production batches candidates while the retained independent ECP
    assembler evaluates them individually; the Jastrow leaf is shared,
    so this is a route/ordering guarantee, not an independent J1 formula.
    Pair mesh rows by their full up/down coordinates with a bijection,
    then compare both coordinate arrays, each weighted ECP contribution
    and the total. No mesh entries may be dropped. Both modes retain the
    original element-wise Coulomb strict policy. No statistical threshold,
    finite difference, precision substitution or new skip is used.
    """
    source = Path(__file__).resolve().parents[1] / "trexio_example_files" / "water_ccecp_ccpvqz.h5"
    structure, _, _, _, geminal, potential = read_trexio_file(str(source), store_tuple=True)
    jastrow = _build_full_jastrow_data(structure, geminal, potential, np.random.default_rng(31))
    wavefunction = Wavefunction_data(geminal_data=geminal, jastrow_data=jastrow)
    rng = np.random.default_rng(0)
    r_up_carts = rng.uniform(-1.0, 1.0, (geminal.num_electron_up, 3))
    r_dn_carts = rng.uniform(-1.0, 1.0, (geminal.num_electron_dn, 3))
    if family == "all_pairs":
        actual = coulomb.compute_ecp_non_local_parts_all_pairs(
            potential, wavefunction, r_up_carts, r_dn_carts, Nv=quadrature, RT=np.eye(3), flag_determinant_only=False
        )
        reference = coulomb._compute_ecp_non_local_parts_all_pairs_debug(
            potential, wavefunction, r_up_carts, r_dn_carts, Nv=quadrature, RT=np.eye(3), flag_determinant_only=False
        )
    else:
        actual = coulomb.compute_ecp_non_local_parts_nearest_neighbors(
            potential,
            wavefunction,
            r_up_carts,
            r_dn_carts,
            NN=neighbors,
            Nv=quadrature,
            RT=np.eye(3),
            flag_determinant_only=False,
        )
        reference = coulomb._compute_ecp_non_local_parts_nearest_neighbors_debug(
            potential,
            wavefunction,
            r_up_carts,
            r_dn_carts,
            NN=neighbors,
            Nv=quadrature,
            RT=np.eye(3),
            flag_determinant_only=False,
        )
    expected_rows = (
        (geminal.num_electron_up + geminal.num_electron_dn) * quadrature * (structure.natom if neighbors is None else neighbors)
    )
    assert np.shape(actual[2]) == np.shape(reference[2]) == (expected_rows,)
    actual_coordinates = np.concatenate([np.asarray(x).reshape(len(actual[2]), -1) for x in actual[:2]], axis=1)
    reference_coordinates = np.concatenate([np.asarray(x).reshape(len(reference[2]), -1) for x in reference[:2]], axis=1)
    distances = np.max(np.abs(actual_coordinates[:, None, :] - reference_coordinates[None, :, :]), axis=2)
    permutation = np.argmin(distances, axis=1)
    assert len(permutation) == len(reference[2])
    assert len(set(permutation)) == len(permutation)
    actual_up, actual_down, actual_contributions, actual_total = map(np.asarray, actual)
    reference_up, reference_down, reference_contributions = (np.asarray(x)[permutation] for x in reference[:3])
    reference_total = np.asarray(reference[3])
    atol, rtol = get_tolerance("coulomb", "strict")
    np.testing.assert_allclose(actual_up, reference_up, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_down, reference_down, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_contributions, reference_contributions, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_total, reference_total, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("fault", ["one-contribution", "duplicate-row"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_water_mesh_verdict_rejects_incomplete_or_corrupt_rows(monkeypatch, fault):
    """A passing full mesh rejects a changed contribution or a duplicated row, even with the total unchanged."""
    test_water_j123_ecp_whole_mesh_matches_scalar_assembly("all_pairs", None, 6)
    original = coulomb.compute_ecp_non_local_parts_all_pairs

    def corrupted(*args, **kwargs):
        result = [np.array(value, copy=True) for value in original(*args, **kwargs)]
        if fault == "one-contribution":
            result[2][0] += 1.0
        else:
            result[0][0] = result[0][1]
            result[1][0] = result[1][1]
        return tuple(result)

    monkeypatch.setattr(coulomb, "compute_ecp_non_local_parts_all_pairs", corrupted)
    with pytest.raises(AssertionError):
        test_water_j123_ecp_whole_mesh_matches_scalar_assembly("all_pairs", None, 6)
