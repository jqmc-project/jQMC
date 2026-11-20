"""T1 -- get_gF vs the Wirtinger finite difference of the reweighted energy
(the complex generalized-force oracle).

Shared fixtures / dense references / the theory write-up live in
``complex_opt_ground_truth_harness.py``; the pinned convention is
``doc/notes/derivatives.md`` and ``doc/notes/complex_gradients.md``.
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

from tests.test_jqmc_kernel.complex_opt_ground_truth_harness import (
    _CASES,
    get_case,
    _FD_RTOL,
    _N_JACKKNIFE_RTOL,
    _N_SAMPLES,
    _probe_indices,
)

pytestmark = [pytest.mark.with_chdir]


@pytest.fixture(scope="module", params=_CASES)
def case(request):
    return get_case(request.param)


class TestT1ForceWirtingerFD:
    @pytest.mark.heavy
    def test_covariance_force_matches_fd_exactly(self, case):
        """EXACT finite-sample identity for a COMPLEX (holomorphic) parameter:

            f = -2<w conj(dO) de> == -(dE/dx + i dE/dy).

        The <de_L/dc> contribution enters dE/dx and dE/dy with opposite
        Wirtinger weights and cancels EXACTLY in the combination (de_L/dc* = 0),
        so no correction term is needed. For a REAL-dtype parameter (the Gamma
        control -- there is no Im displacement) the same calculus gives the
        corrected x-direction identity  dE/dx = 2<w dO de> + <w de_L/dc>,
        verified with the stored de_L/dc buffer. Anchors the pinned formula to
        elementary calculus; independent of any optimizer code.
        """
        W = case.w.sum()
        atol = _FD_RTOL * np.max(np.abs(case.f_ref))
        for k in _probe_indices(case):
            if np.iscomplexobj(case.O[:, k : k + 1]) and case.kind == "twist":
                f_fd = case.fd_wirtinger_force(k)
                assert_allclose(
                    case.f_ref[k],
                    f_fd,
                    rtol=_FD_RTOL,
                    atol=atol,
                    err_msg=f"covariance force != Wirtinger FD at flat index {k} ({case.kind})",
                )
            else:
                unit = np.zeros(case.K, dtype=np.complex128)
                unit[k] = 1.0
                dE_dx = case.fd_dE_along(unit)
                eterm = np.real(np.sum(case.w * case.dE[:, k]) / W)
                assert_allclose(
                    np.real(-case.f_ref[k]) + eterm,
                    np.real(dE_dx),
                    rtol=_FD_RTOL,
                    atol=atol,
                    err_msg=f"real-dtype corrected FD identity violated at flat index {k} ({case.kind})",
                )

    @pytest.mark.heavy
    def test_real_param_exact_decomposition(self, case):
        """For the REAL j2 parameter: dE/dx = 2<w dO de> + <w de_L/dx> exactly
        in its physical real direction.  The derivative source itself remains
        complex; only this real scalar energy identity takes an outer Re.  The
        covariance part is the optimizer's force and the <de_L/dx> term is the
        finite-sample piece Hermiticity kills in expectation.
        """
        s, _ = case.offsets["j2_param"]
        W = case.w.sum()
        dE_dx = case.fd_dE_along(np.eye(case.K, dtype=np.complex128)[s])
        cov = 2.0 * np.sum(case.w * case.dO[:, s] * case.de) / W
        eterm = np.sum(case.w * case.dE[:, s]) / W
        assert_allclose(
            np.real(dE_dx),
            np.real(cov) + np.real(eterm),
            rtol=_FD_RTOL,
            atol=_FD_RTOL * max(1.0, abs(cov)),
            err_msg=f"real-parameter exact decomposition violated ({case.kind})",
        )

    @pytest.mark.heavy
    def test_get_gF_returns_the_complex_force(self, case):
        """get_gF must return the COMPLEX f (Re AND Im) of the pinned formula
        (within the deterministic jackknife O(1/N) deviation). A get_gF that
        Re-projects f fails at the twist.
        """
        m = case.injected_mcmc()
        f_impl, _f_std = m.get_gF(
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=_N_SAMPLES,
            blocks=case.blocks,
        )
        atol = _N_JACKKNIFE_RTOL * np.max(np.abs(case.f_ref))
        assert_allclose(
            np.real(f_impl),
            np.real(case.f_ref),
            rtol=_N_JACKKNIFE_RTOL,
            atol=atol,
            err_msg=f"get_gF Re(f) mismatch ({case.kind})",
        )
        assert_allclose(
            np.imag(np.asarray(f_impl, dtype=np.complex128)),
            np.imag(case.f_ref),
            rtol=_N_JACKKNIFE_RTOL,
            atol=atol,
            err_msg=f"get_gF must carry Im(f) -- the imaginary DOF of complex parameters ({case.kind})",
        )
