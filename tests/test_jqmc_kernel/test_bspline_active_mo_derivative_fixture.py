"""Virtual-band B-spline inputs supplement occupied-space derivative fixtures."""

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

import functools
import gc
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc.derivatives import bridge
from jqmc.determinant import compute_geminal_all_elements
from jqmc.pw_pgto_projection import project_pw_to_spline
from jqmc.qe_to_jqmc import build_wavefunction_from_projection
from jqmc.qe_wrapper import load_qe_reference
from tests._test_data_cache import qe_example_files_root
from tests.test_jqmc_kernel._generated_matrix_base import _QE_PROJ, _apply_gem, _build_ham_for_cell, _resolve
from tests.test_jqmc_kernel._grad_fidelity_harness import _batched_geminal_inv, _draw_batch, _flags_for_axis

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]


@functools.lru_cache(maxsize=4)
def _all_band_bspline_geminal(fixture_path):
    """Project all committed bands while retaining occupied MO rows only."""
    source = qe_example_files_root() / Path(fixture_path).relative_to("tests/qe_example_files")
    reference = load_qe_reference(str(source), ik=1)
    projection = project_pw_to_spline(reference, mesh_factor=1.0, order=5, occ_only=False)
    structure, _, geminal = build_wavefunction_from_projection(reference, projection, kind="bspline")
    return structure, geminal


@pytest.mark.parametrize(
    "cell,fixture_path",
    [
        (("bspline", "DTN", "AE", "pbc_gamma", "AGPn"), "tests/qe_example_files/H-ae_gamma/out/h_gamma.save"),
        (("bspline", "DTN", "AE", "pbc_gamma", "SD"), "tests/qe_example_files/H-ae_gamma/out/h_gamma.save"),
        (("bspline", "J1", "ECP", "pbc_gamma", "SD"), "tests/qe_example_files/Li_gamma/out/li.save"),
        (("bspline", "J2", "AE", "pbc_gamma", "SD"), "tests/qe_example_files/H-ae_gamma/out/h_gamma.save"),
        (("bspline", "J3poly", "AE", "pbc_gamma", "SD"), "tests/qe_example_files/H-ae_gamma/out/h_gamma.save"),
        (("bspline", "none", "AE", "pbc_gamma", "SD"), "tests/qe_example_files/H-ae_gamma/out/h_gamma.save"),
        (("bspline_complex", "DTN", "AE", "twist", "SD"), "tests/qe_example_files/H-ae_k025/out/h_k025.save"),
        (("bspline_complex", "DTN", "ECP", "twist", "SD"), "tests/qe_example_files/Li_k025/out/li_k025.save"),
        (("bspline_complex", "J1", "AE", "twist", "SD"), "tests/qe_example_files/H-ae_k025/out/h_k025.save"),
        (("bspline_complex", "J1", "ECP", "twist", "AGPn"), "tests/qe_example_files/Li_k025/out/li_k025.save"),
        (("bspline_complex", "J1", "ECP", "twist", "SD"), "tests/qe_example_files/Li_k025/out/li_k025.save"),
        (("bspline_complex", "J2", "AE", "twist", "SD"), "tests/qe_example_files/H-ae_k025/out/h_k025.save"),
        (("bspline_complex", "J2", "ECP", "twist", "SD"), "tests/qe_example_files/Li_k025/out/li_k025.save"),
        (("bspline_complex", "J3poly", "AE", "twist", "SD"), "tests/qe_example_files/H-ae_k025/out/h_k025.save"),
        (("bspline_complex", "J3poly", "ECP", "twist", "SD"), "tests/qe_example_files/Li_k025/out/li_k025.save"),
        (("bspline_complex", "none", "AE", "twist", "SD"), "tests/qe_example_files/H-ae_k025/out/h_k025.save"),
        (("bspline_complex", "none", "ECP", "twist", "SD"), "tests/qe_example_files/Li_k025/out/li_k025.save"),
    ],
    ids=[
        "bspline__DTN__AE__pbc_gamma__AGPn-h_gamma.save",
        "bspline__DTN__AE__pbc_gamma__SD-h_gamma.save",
        "bspline__J1__ECP__pbc_gamma__SD-li.save",
        "bspline__J2__AE__pbc_gamma__SD-h_gamma.save",
        "bspline__J3poly__AE__pbc_gamma__SD-h_gamma.save",
        "bspline__none__AE__pbc_gamma__SD-h_gamma.save",
        "bspline_complex__DTN__AE__twist__SD-h_k025.save",
        "bspline_complex__DTN__ECP__twist__SD-li_k025.save",
        "bspline_complex__J1__AE__twist__SD-h_k025.save",
        "bspline_complex__J1__ECP__twist__AGPn-li_k025.save",
        "bspline_complex__J1__ECP__twist__SD-li_k025.save",
        "bspline_complex__J2__AE__twist__SD-h_k025.save",
        "bspline_complex__J2__ECP__twist__SD-li_k025.save",
        "bspline_complex__J3poly__AE__twist__SD-h_k025.save",
        "bspline_complex__J3poly__ECP__twist__SD-li_k025.save",
        "bspline_complex__none__AE__twist__SD-h_k025.save",
        "bspline_complex__none__ECP__twist__SD-li_k025.save",
    ],
)
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="seeded-random",
    reason="Original PCG64 seed 0, two normal(scale=0.7) configurations; no cond(G) screen or resampling.",
)
def test_bspline_virtual_mo_derivatives_match_autodiff(cell, fixture_path, record_property):
    """Virtual-basis MO energy derivatives agree with autodiff and reject zero.

    Committed H Gamma/twist and Li Gamma/twist QE saves supply all-band quintic
    B-splines at mesh_factor=1: H has four occupied MO rows and ten basis bands,
    Li has one occupied row and eight bands per spin. The selected cell retains
    its original Jastrow, Coulomb potential, SD/AGPn representation and spin/twist
    convention. Original occupied-only matrix fixtures are not changed.

    PCG64(0) supplies the original two normal(scale=0.7 Bohr) electron
    configurations: H has four up/four down, Li one up/one down. cond(G) is
    recorded without screening or resampling. Both bridge paths retain the
    original c-axis mask and running inverse. The production parameter reducer
    supplies mo_coefficients_up, including any time-reversal spin tie; only
    columns beyond the occupied count are compared. Full uses central strict,
    mixed central medium, in the original six derivative zones. Every walker
    must reject a zero replacement at that same budget. No sampling uncertainty
    or full-versus-mixed comparison enters this deterministic test.
    """
    try:
        resolved = _resolve(cell)
        assert resolved is not None, f"missing committed matrix source for {cell}"
        source = Path(__file__).resolve().parents[2] / fixture_path
        assert Path(_QE_PROJ[resolved[0]][1]).resolve() == source.resolve()
        ham = _build_ham_for_cell(*resolved)
        old_geminal = ham.wavefunction_data.geminal_data
        structure, all_band_sd = _all_band_bspline_geminal(fixture_path)
        np.testing.assert_array_equal(structure.positions, ham.structure_data.positions)
        np.testing.assert_array_equal(structure.cell, ham.structure_data.cell)
        assert all_band_sd.num_electron_up == old_geminal.num_electron_up
        assert all_band_sd.num_electron_dn == old_geminal.num_electron_dn
        assert all_band_sd.time_reversal == old_geminal.time_reversal
        assert bool(ham.coulomb_potential_data.ecp_flag) == (cell[2] == "ECP")
        ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=all_band_sd))
        ham = _apply_gem(ham, cell[4])
        geminal = ham.wavefunction_data.geminal_data
        n_occ = int(geminal.num_electron_up)
        expected_shape = (4, 10) if cell[2] == "AE" else (1, 8)
        assert geminal.orb_data_up_spin.mo_coefficients.shape == expected_shape
        assert geminal.orb_data_dn_spin.mo_coefficients.shape == expected_shape
        r_up, r_down = _draw_batch(ham, nw=2, seed=0)
        cond_G = [
            float(np.linalg.cond(np.asarray(compute_geminal_all_elements(geminal, up, down))))
            for up, down in zip(r_up, r_down, strict=True)
        ]
        record_property("cond_G_unscreened", cond_G)
        inverse = _batched_geminal_inv(ham, r_up, r_down)
        rotation = jnp.broadcast_to(jnp.eye(3), (2, 3, 3))
        mask = _flags_for_axis("c")
        reduced = []
        for analytic in (True, False):
            evaluate = bridge.batched_grad_e_L(wrt="c", mask=mask, with_geminal_inv=True, analytic=analytic)
            raw = evaluate(ham, r_up, r_down, rotation, inverse)
            blocks = bridge._reduce_c_param_grads(ham.wavefunction_data, raw, 2, mask)
            reduced.append(np.asarray(blocks["mo_coefficients_up"])[..., n_occ:])
        analytical, autodiff = reduced
        assert analytical.shape == autodiff.shape == (2, n_occ, expected_shape[1] - n_occ)
        assert analytical.size > 0
        assert np.all(np.isfinite(analytical)) and np.all(np.isfinite(autodiff))
        level = "medium" if is_mixed_precision_enabled() else "strict"
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic"), level
        )
        for actual, reference in zip(analytical, autodiff, strict=True):
            np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
            assert not np.allclose(np.zeros_like(actual), reference, atol=atol, rtol=rtol, equal_nan=False), (
                "Zero virtual-basis MO derivatives would satisfy the numerical comparison"
            )
    finally:
        bridge.cache_clear()
        jax.clear_caches()
        gc.collect()
