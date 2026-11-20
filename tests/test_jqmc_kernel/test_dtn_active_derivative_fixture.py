"""Active DTN f-channel inputs supplement the existing derivative matrix."""

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

import gc
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from jqmc.derivatives import bridge
from jqmc.determinant import compute_geminal_all_elements
from tests._precision_fidelity import mixed_vs_full
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell, _qe_proj_pieces, _resolve
from tests.test_jqmc_kernel._grad_fidelity_harness import _batched_geminal_inv, _flags_for_axis

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]

_SOURCE_STEMS = {
    "bspline": "QEPROJ:bspline:H_gamma:AE",
    "bspline_complex": "QEPROJ:bspline:H_k025:AE",
    "composite": "QEPROJ:pgto+pw:H_k025:AE",
    "plane_wave": "QEPROJ:pw:H_gamma:AE",
    "plane_wave_complex": "QEPROJ:pw:H_k025:AE",
}


def _active_electron_configurations(ham, seed=42):
    """Draw two fixed nuclear-centred configurations, without rejection sampling.

    PCG64 initially samples four nuclei without replacement for each spin, then
    overwrites the first down-spin assignment with the first up-spin nucleus.
    That overwrite can duplicate a down-spin home nucleus; independent Gaussian
    offsets of width 0.35 Bohr still give distinct electron positions.
    Candidate indices 0 and 9 are frozen: they were the first
    two satisfying cond(G) < 700 across all eight cells in both modes in a
    conditioning-only inspection of 64 candidates, before evaluating derivatives.
    Runtime never retries or selects based on conditioning or derivative results.
    """
    ions = np.asarray(ham.structure_data.positions)
    rng = np.random.Generator(np.random.PCG64(seed))
    up, down = [], []
    for candidate_index in range(10):
        iu = rng.choice(len(ions), size=4, replace=False)
        idn = rng.choice(len(ions), size=4, replace=False)
        idn[0] = iu[0]
        candidate_up = ions[iu] + rng.normal(scale=0.35, size=(4, 3))
        candidate_down = ions[idn] + rng.normal(scale=0.35, size=(4, 3))
        if candidate_index in (0, 9):
            up.append(candidate_up)
            down.append(candidate_down)
    return jnp.asarray(np.stack(up)), jnp.asarray(np.stack(down))


def _assert_active_geometry_and_conditioning(ham, r_up, r_down):
    """Independently observe an in-cutoff electron pair and cond_2(G) < 700."""
    gem = ham.wavefunction_data.geminal_data
    assert (gem.num_electron_up, gem.num_electron_dn) == (4, 4)
    assert len(ham.structure_data.positions) == 8
    dtn = ham.wavefunction_data.jastrow_data.jastrow_dtn_data
    assert dtn.gamma_indep.size > 0
    cell = np.asarray(ham.structure_data.cell)
    ions = np.asarray(ham.structure_data.positions)
    shifts = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)))
    for up, down in zip(r_up, r_down, strict=True):
        electrons = np.concatenate([up, down])
        fractional = (electrons[:, None, :] - ions[None, :, :]) @ np.linalg.inv(cell)
        fractional -= np.rint(fractional)
        distances = np.linalg.norm((fractional[..., None, :] + shifts) @ cell, axis=-1).min(axis=-1)
        inside = distances < np.asarray(dtn.L_f)
        assert np.any(np.sum(inside, axis=0) >= 2), "DTN f-channel has no same-nucleus pair inside L_f"
        geminal_matrix = np.asarray(compute_geminal_all_elements(gem, up, down))
        cond_G = float(np.linalg.cond(geminal_matrix))
        assert np.isfinite(cond_G) and cond_G < 700, cond_G


def _assert_active_gamma_fidelity(mixed_gamma, full_gamma, atol, rtol):
    assert np.max(np.abs(mixed_gamma - full_gamma)) <= atol + rtol * np.max(np.abs(full_gamma))


def _compare_active_gamma_mixed(cell, quantity):
    """Compare each walker with a fresh Full build; retain the old zero control.

    The AD result is used only for the original non-vacuity rejection, under
    the entry Mixed medium band. It is not the Mixed numerical oracle.
    """
    control_atol, control_rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic"), "medium"
    )

    def build():
        # Projection pieces contain policy-dependent leaves. No cache from the
        # other policy may supply this independently reconstructed source.
        bridge.cache_clear()
        jax.clear_caches()
        _qe_proj_pieces.cache_clear()
        resolved = _resolve(cell)
        assert resolved is not None, f"missing committed matrix source for {cell}"
        assert resolved[0] == _SOURCE_STEMS[cell[0]]
        ham = _build_ham_for_cell(*resolved)
        r_up, r_down = _active_electron_configurations(ham, seed=42)
        _assert_active_geometry_and_conditioning(ham, r_up, r_down)
        inverse = _batched_geminal_inv(ham, r_up, r_down)
        assert inverse.shape == (2, 4, 4)
        assert np.all(np.isfinite(np.asarray(inverse)))
        return ham, r_up, r_down, inverse

    def evaluate(case):
        ham, r_up, r_down, inverse = case
        mask = _flags_for_axis("c")
        reduced = []
        for analytic in (True, False):
            if quantity == "ln_Psi":
                evaluate_bridge = bridge.batched_grad_ln_Psi(wrt="c", mask=mask, with_geminal_inv=True, analytic=analytic)
                raw = evaluate_bridge(ham, r_up, r_down, inverse)
            else:
                rotation = jnp.broadcast_to(jnp.eye(3), (2, 3, 3))
                evaluate_bridge = bridge.batched_grad_e_L(wrt="c", mask=mask, with_geminal_inv=True, analytic=analytic)
                raw = evaluate_bridge(ham, r_up, r_down, rotation, inverse)
            blocks = bridge._reduce_c_param_grads(ham.wavefunction_data, raw, 2, mask)
            assert "j3_poly_gamma" in blocks
            reduced.append(np.asarray(blocks["j3_poly_gamma"]))
        analytical, autodiff = reduced
        assert analytical.shape == autodiff.shape and analytical.shape[0] == 2
        assert analytical.size > 0
        assert np.all(np.isfinite(analytical)) and np.all(np.isfinite(autodiff))
        for actual, reference in zip(analytical, autodiff, strict=True):
            assert not np.allclose(np.zeros_like(actual), reference, atol=control_atol, rtol=control_rtol, equal_nan=False), (
                "A zero gamma derivative would satisfy the numerical comparison"
            )
        return {
            "gamma": analytical,
            "r_up": r_up,
            "r_down": r_down,
            "ions": ham.structure_data.positions,
            "cell": ham.structure_data.cell,
        }

    try:
        mixed, full = mixed_vs_full(build, evaluate)
        assert set(mixed) == set(full) == {"gamma", "r_up", "r_down", "ions", "cell"}
        for field in ("r_up", "r_down", "ions", "cell"):
            np.testing.assert_array_equal(mixed[field], full[field])
        mixed_gamma, full_gamma = mixed["gamma"], full["gamma"]
        assert mixed_gamma.shape == full_gamma.shape and mixed_gamma.shape[0] == 2
        assert mixed_gamma.size > 0
        assert np.all(np.isfinite(mixed_gamma)) and np.all(np.isfinite(full_gamma))
        fidelity_atol, fidelity_rtol = get_tolerance_mixed_fidelity("parameter_derivative")
        for walker in ("candidate-0", "candidate-9"):
            walker_index = 0 if walker == "candidate-0" else 1
            _assert_active_gamma_fidelity(mixed_gamma[walker_index], full_gamma[walker_index], fidelity_atol, fidelity_rtol)
    finally:
        bridge.cache_clear()
        jax.clear_caches()
        _qe_proj_pieces.cache_clear()
        gc.collect()


@pytest.mark.parametrize(
    "cell",
    [
        ("bspline", "DTN", "AE", "pbc_gamma", "SD"),
        ("bspline", "DTN", "AE", "pbc_gamma", "AGPn"),
        ("bspline_complex", "DTN", "AE", "twist", "SD"),
        ("composite", "DTN", "AE", "twist", "SD"),
        ("composite", "DTN", "AE", "twist", "AGPn"),
        ("plane_wave", "DTN", "AE", "pbc_gamma", "SD"),
        ("plane_wave", "DTN", "AE", "pbc_gamma", "AGPn"),
        ("plane_wave_complex", "DTN", "AE", "twist", "AGPn"),
    ],
    ids=lambda cell: "__".join(cell),
)
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="seeded-random",
    reason="PCG64 seed 42, fixed candidate indices 0 and 9; each cond_2(G) < 700; no runtime rejection sampling.",
)
def test_active_dtn_gamma_matches_autodiff(cell, quantity):
    """Full gamma derivatives agree with AD; Mixed agrees with independent Full.

    The unchanged H_gamma/H_k025 QE matrix builders provide eight-atom AE
    systems with four up and four down electrons, SD or AGPn determinants,
    PW/B-spline/composite orbitals and real-Gamma or complex-twist boundaries.
    DTN coefficients, cutoffs and the original matrix fixtures are unchanged.
    PCG64(42) supplies fixed candidate indices 0 and 9 with nuclear-centred
    width 0.35 Bohr, without retries. These were the first two common passing
    configurations in a 64-candidate conditioning-only inspection across all
    eight cells and both modes, before any derivative comparison. Each must
    have cond_2(G) < 700 and a same-nucleus
    electron pair strictly inside L_f, checked independently with 27 images.

    Both production bridge calls retain the original full c-axis mask and
    running geminal inverse; only the reduced j3_poly_gamma block is claimed
    here. Full keeps the strict mathematical AD comparison. Mixed compares
    each walker with an independently rebuilt Full analytical block using the
    central parameter-derivative budget. Both policy builds retain the original
    zero-block rejection using the entry Mixed six-zone medium band and AD
    reference, as a non-vacuity control rather than a numerical oracle.
    This is a deterministic derivative test, not a sampled energy estimate.
    """
    if is_mixed_precision_enabled():
        _compare_active_gamma_mixed(cell, quantity)
        return
    try:
        resolved = _resolve(cell)
        assert resolved is not None, f"missing committed matrix source for {cell}"
        assert resolved[0] == _SOURCE_STEMS[cell[0]]
        ham = _build_ham_for_cell(*resolved)
        r_up, r_down = _active_electron_configurations(ham, seed=42)
        _assert_active_geometry_and_conditioning(ham, r_up, r_down)
        inverse = _batched_geminal_inv(ham, r_up, r_down)
        mask = _flags_for_axis("c")
        reduced = []
        for analytic in (True, False):
            if quantity == "ln_Psi":
                evaluate = bridge.batched_grad_ln_Psi(wrt="c", mask=mask, with_geminal_inv=True, analytic=analytic)
                raw = evaluate(ham, r_up, r_down, inverse)
            else:
                rotation = jnp.broadcast_to(jnp.eye(3), (2, 3, 3))
                evaluate = bridge.batched_grad_e_L(wrt="c", mask=mask, with_geminal_inv=True, analytic=analytic)
                raw = evaluate(ham, r_up, r_down, rotation, inverse)
            blocks = bridge._reduce_c_param_grads(ham.wavefunction_data, raw, 2, mask)
            reduced.append(np.asarray(blocks["j3_poly_gamma"]))
        analytical, autodiff = reduced
        assert analytical.shape == autodiff.shape and analytical.shape[0] == 2
        assert analytical.size > 0
        assert np.all(np.isfinite(analytical)) and np.all(np.isfinite(autodiff))
        level = "medium" if is_mixed_precision_enabled() else "strict"
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic"), level
        )
        for actual, reference in zip(analytical, autodiff, strict=True):
            np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
            assert not np.allclose(np.zeros_like(actual), reference, atol=atol, rtol=rtol, equal_nan=False), (
                "A zero gamma derivative would satisfy the numerical comparison"
            )
    finally:
        bridge.cache_clear()
        jax.clear_caches()
        gc.collect()
