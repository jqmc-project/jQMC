"""Empty-down-spin determinant ratios on a fixed sequence of electron moves."""

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

from jqmc import determinant as determinant
from jqmc._precision import get_tolerance_min
from jqmc.molecular_orbital import compute_MOs
from jqmc.trexio_wrapper import read_trexio_file

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize("trexio_file", ["H_ae_ccpvdz_cart.h5", "H_ecp_ccpvdz_cart.h5"])
@pytest.mark.parametrize("det_kind", ["sd", "agp", "agpn"])
def test_single_spin_ratios_follow_explicit_moves(trexio_file, det_kind):
    """Four empty-down-spin ratio routes equal the one-electron MO quotient at every hop.

    Load the committed H AE/ECP Cartesian-GTO orbital and represent its single
    up electron as SD, dense AGP, or explicitly built rank-one AGPn. The paired
    AGPn occupation is zero; its unpaired coefficient is one. No implicit
    open-shell SD-to-AGPn conversion or virtual-orbital inference is used.

    Start at (0.31, 0.42, 0.53) Bohr. At each of seven configurations compare
    all three candidate moves (+0.13 x, -0.11 y, +0.07 z). Accept those moves
    cyclically for six hops, carrying the Sherman-Morrison inverse and both
    orbital caches without a rebuild. The reference evaluates the original
    occupied MO afresh, not a determinant/update kernel. G is a nonzero 1x1
    matrix, so cond(G)=1; there is no random draw or rejection screen.
    This tests supported fallback values as well as cache consumers, not the
    absence of a full-compute fallback or the accuracy of the shared MO API.
    """
    fixture = Path(__file__).resolve().parents[1] / "trexio_example_files" / trexio_file
    _, _, _, _, sd, _ = read_trexio_file(str(fixture), store_tuple=True)
    assert (sd.num_electron_up, sd.num_electron_dn) == (1, 0)
    if det_kind == "sd":
        geminal = sd
    elif det_kind == "agp":
        geminal = determinant.to_AGP(sd)
    else:
        geminal = determinant.AGPn_data.build(
            num_electron_up=1,
            num_electron_dn=0,
            mos_up=sd.orb_data_up_spin,
            mos_dn=sd.orb_data_up_spin,
            lambda_diag=np.zeros(1),
            lambda_unpaired=np.ones((1, 1)),
            time_reversal=True,
        )
    geminal.sanity_check()
    r_up_carts = jnp.asarray([[0.31, 0.42, 0.53]])
    r_dn_carts = jnp.empty((0, 3))
    shifts = jnp.asarray([[0.13, 0.0, 0.0], [0.0, -0.11, 0.0], [0.0, 0.0, 0.07]])
    candidate_up = r_up_carts[:, None, :] + shifts[None, :, :]
    candidate_dn = jnp.empty((0, 3, 3))
    ratio_state = determinant._init_det_ratio_streaming_state(geminal, r_up_carts, r_dn_carts)
    candidate_state = determinant._init_det_candidate_mesh_state(geminal, candidate_up, candidate_dn)
    inverse = jnp.linalg.inv(determinant.compute_geminal_all_elements(geminal, r_up_carts, r_dn_carts))
    atol, rtol = get_tolerance_min(("det_ratio",), "strict")
    for hop in range(7):
        hop_label = str(hop)
        occupied_value = np.asarray(compute_MOs(sd.orb_data_up_spin, r_up_carts))[0, 0]
        assert np.isfinite(occupied_value) and occupied_value != 0
        cond_g = float(np.linalg.cond(np.asarray([[occupied_value]])))
        assert cond_g == 1
        reference = np.asarray(compute_MOs(sd.orb_data_up_spin, candidate_up.reshape(-1, 3)))[0] / occupied_value
        assert reference.shape == (3,)
        winv_state = determinant._init_det_winv_state(geminal, inverse, ratio_state)
        rank1_ratios = determinant._compute_ratio_determinant_part_rank1_update(
            geminal, inverse, r_up_carts, r_dn_carts, candidate_up[0, :, None, :], jnp.empty((3, 0, 3))
        )
        split_ratios = determinant._compute_ratio_determinant_part_split_spin(
            geminal,
            inverse,
            r_up_carts,
            r_dn_carts,
            candidate_up[0, :, None, :],
            jnp.empty((0, 0, 3)),
            det_ratio_state=ratio_state,
        )
        cached_ratios = determinant._compute_ratio_determinant_from_candidate_mesh_state(
            geminal, inverse, ratio_state, candidate_state
        )
        winv_ratios = determinant._compute_ratio_determinant_from_winv_state(geminal, candidate_state, winv_state)
        np.testing.assert_allclose(rank1_ratios, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=hop_label)
        np.testing.assert_allclose(split_ratios, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=hop_label)
        np.testing.assert_allclose(cached_ratios, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=hop_label)
        np.testing.assert_allclose(winv_ratios, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=hop_label)
        if hop == 6:
            break
        next_up = candidate_up[:, hop % 3, :]
        v = determinant._compute_v_up_move_from_det_ratio_state(geminal, ratio_state, 0, next_up)[:, None]
        _, inverse = determinant._sherman_morrison_ratio_and_inverse(inverse, jnp.ones((1, 1)), v)
        ratio_state = determinant._advance_det_ratio_streaming_state(
            geminal, ratio_state, jnp.bool_(True), jnp.int64(0), next_up, r_dn_carts
        )
        r_up_carts = next_up
        candidate_up = r_up_carts[:, None, :] + shifts[None, :, :]
        candidate_state = determinant._advance_det_candidate_mesh_state(
            geminal, candidate_state, jnp.bool_(True), jnp.int64(0), candidate_up, candidate_dn
        )
