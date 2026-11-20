"""T3/T4 -- get_aH + solve_linear_method vs a dense generalized-eig reference,
and the aSR scalars vs the pinned quadratic forms + FD directional anchor.

Shared fixtures / dense references / the theory write-up live in
``complex_opt_ground_truth_harness.py``; the pinned convention is
``doc/notes/wavefunction_optimization.md``.
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
from numpy.testing import assert_allclose

from jqmc._setting import (
    ATOL_BIT_EXACT,
    ATOL_LM_EIGVAL_VS_DENSE,
    ATOL_LM_STEP_VS_DENSE_SCALE,
    ATOL_QUADFORM_VS_DENSE,
    RTOL_LM_EIGVAL_VS_DENSE,
    RTOL_LM_STEP_VS_DENSE,
    RTOL_PYTHON_FP64_AGGREGATOR_INVARIANT,
    RTOL_QUADFORM_VS_DENSE,
)
from jqmc.jqmc_optimizer_kernel import solve_linear_method

from tests.test_jqmc_kernel.complex_opt_ground_truth_harness import (
    _CASES,
    get_case,
    _FD_RTOL,
    _lm_reference,
    _lm_subset,
    _rayleigh_quotient,
)

pytestmark = [pytest.mark.with_chdir]


@pytest.fixture(scope="module", params=_CASES)
def case(request):
    return get_case(request.param)


class TestT3LinearMethod:
    @pytest.mark.heavy
    def test_lm_step_matches_dense_generalized_eig(self, case):
        """(c_vec, E_lm) from get_aH + solve_linear_method == the dense reference.
        At the twist a Re-projected f or a swapped H_bar f-block orientation
        fails it; at Gamma neither is visible.
        """
        subset = _lm_subset(case)
        delta_ref, E_ref, (H_bar, S_bar) = _lm_reference(case, subset)

        m = case.injected_mcmc()
        H_0, f_vec, S_mat, K_mat, B_mat, _bra_row = m.get_aH(
            blocks=case.blocks,
            num_mcmc_warmup_steps=0,
            chosen_param_index=np.asarray(subset, dtype=np.intp),
            return_matrices=True,
        )
        c_vec, E_lm, v0_sq = solve_linear_method(H_0, f_vec, S_mat, K_mat, B_mat, epsilon=1e-12)

        assert_allclose(
            E_lm,
            E_ref,
            rtol=RTOL_LM_EIGVAL_VS_DENSE,
            atol=ATOL_LM_EIGVAL_VS_DENSE,
            err_msg=f"LM eigenvalue != dense reference ({case.kind})",
        )
        assert_allclose(
            np.asarray(c_vec, dtype=np.complex128),
            delta_ref,
            rtol=RTOL_LM_STEP_VS_DENSE,
            atol=ATOL_LM_STEP_VS_DENSE_SCALE * np.max(np.abs(delta_ref)),
            err_msg=f"LM update direction != dense reference ({case.kind})",
        )
        # Variational meaning: the returned step attains the returned eigenvalue
        # on the pinned-estimator Rayleigh quotient.
        assert_allclose(
            _rayleigh_quotient(H_bar, S_bar, np.asarray(c_vec, dtype=np.complex128)),
            E_lm,
            rtol=RTOL_LM_EIGVAL_VS_DENSE,
            atol=ATOL_LM_EIGVAL_VS_DENSE,
            err_msg=f"returned LM step does not attain E_lm on the Rayleigh quotient ({case.kind})",
        )


class TestT4ASRScalars:
    def _direction(self, case):
        rng = np.random.default_rng(5)
        if case.kind == "twist":
            g = rng.standard_normal(case.K) + 1j * rng.standard_normal(case.K)
        else:
            g = rng.standard_normal(case.K).astype(np.float64)
        return g / np.linalg.norm(g)

    @pytest.mark.heavy
    def test_directional_slope_fd_anchor(self, case):
        """FD anchor of the raw Wirtinger chain rule along a direction g
        supported on the holomorphic (untied complex MO) blocks:
        dE/dgamma == sum_k g_k (<dO_k de> + <dE_k>) + sum_k conj(g_k) <conj(dO_k) de>
        EXACTLY on the fixed samples. Grounds both the covariance force and the
        stored de_L/dc buffer semantics; independent of optimizer code.
        """
        g = self._direction(case)
        s0, _ = case.offsets["mo_coefficients_up"]
        g_c = np.zeros(case.K, dtype=np.complex128)
        g_c[s0:] = g[s0:]  # zero out the real (j2) component: e_L holomorphic support only
        W = case.w.sum()
        slope_fd = case.fd_dE_along(g_c)
        cov_hol = np.sum(g_c * (case.dO.T @ (case.w * case.de))) / W
        cov_anti = np.sum(np.conj(g_c) * (case.dO.conj().T @ (case.w * case.de))) / W
        eterm = np.sum(g_c * (case.w @ case.dE)) / W
        assert_allclose(
            slope_fd,
            cov_hol + cov_anti + eterm,
            rtol=_FD_RTOL,
            atol=_FD_RTOL * max(1.0, abs(slope_fd)),
            err_msg=f"directional Wirtinger chain rule violated ({case.kind})",
        )

    @pytest.mark.heavy
    def test_asr_scalars_match_pinned_references(self, case):
        """get_aH aSR scalars vs the doc formulas:
        H_1 = -1/2 Re(vdot(g, f)) + 1/2 Re(vdot(g, <de_L/dc>))
        S_2 = <w |g.dO|^2>, H_2 = Re(g^H (K+B) g)  [a guard on the conjugation-correct scalars].
        """
        g = self._direction(case)
        W = case.w.sum()
        m = case.injected_mcmc()
        H_0, H_1, H_2, S_2 = m.get_aH(blocks=case.blocks, g=g, num_mcmc_warmup_steps=0, return_matrices=False)

        gdO = case.dO @ g
        ddE = case.dE - (case.w @ case.dE)[None, :] / W
        gdE = ddE @ g
        dE_bar = case.w @ case.dE / W
        H_1_ref = -0.5 * float(np.real(np.vdot(g, case.f_ref))) + 0.5 * float(np.real(np.vdot(g, dE_bar)))
        S_2_ref = float(np.sum(case.w * np.abs(gdO) ** 2) / W)
        H_2_ref = float(
            np.real(np.sum(case.w * case.e * np.abs(gdO) ** 2) / W) + np.real(np.sum(case.w * np.conj(gdE) * gdO) / W)
        )

        assert_allclose(
            H_0, case.H_0, atol=ATOL_BIT_EXACT, rtol=RTOL_PYTHON_FP64_AGGREGATOR_INVARIANT, err_msg=f"aSR H_0 ({case.kind})"
        )
        assert_allclose(
            S_2, S_2_ref, atol=ATOL_BIT_EXACT, rtol=RTOL_PYTHON_FP64_AGGREGATOR_INVARIANT, err_msg=f"aSR S_2 ({case.kind})"
        )
        assert_allclose(H_2, H_2_ref, atol=ATOL_BIT_EXACT, rtol=RTOL_QUADFORM_VS_DENSE, err_msg=f"aSR H_2 ({case.kind})")
        assert_allclose(
            H_1,
            H_1_ref,
            rtol=RTOL_QUADFORM_VS_DENSE,
            atol=ATOL_QUADFORM_VS_DENSE,
            err_msg=f"aSR H_1 must retain the non-symmetric <de_L/dc> bra-row term ({case.kind})",
        )


# ---------------------------------------------------------------------------
# T5 -- exact gauge (global phase) constraints at the twist
# ---------------------------------------------------------------------------
