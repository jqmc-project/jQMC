"""T5 -- exact gauge (global phase) constraints of the complex force at a twist.

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

from jqmc._precision import get_tolerance_min
from jqmc._setting import ATOL_GAUGE_NULLSPACE_EXACT, ATOL_GAUGE_REFERENCE_SCALE_FLOOR

from tests.test_jqmc_kernel.complex_opt_ground_truth_harness import (
    _CASES,
    get_case,
    _N_JACKKNIFE_RTOL,
    _N_SAMPLES,
)

pytestmark = [pytest.mark.with_chdir]

_GAUGE_ROTATION_ZONES = (
    "ao_eval",
    "mo_eval",
    "det_eval",
    "wf_eval",
    "jastrow_eval",
    "ao_grad_lap",
    "mo_grad",
    "mo_lap",
    "det_grad_lap",
    "jastrow_grad_lap",
    "wf_kinetic",
    "coulomb",
    "ewald",
    "local_energy",
)


# TWIST ONLY, by construction rather than by skip: the gauge phase constraint
# is nontrivial only for complex parameters, so the gamma member of _CASES can
# never exercise either test in this file. A parametrized case that skips in
# EVERY session is a vacuous pin that the module receipt closure refuses to
# close, so the dead case is deselected at collection instead of skipped at
# runtime.
@pytest.fixture(scope="module", params=[c for c in _CASES if c == "twist"])
def case(request):
    return get_case(request.param)


class TestT5GaugeConstraints:
    @pytest.mark.heavy
    def test_gauge_rotation_exact_invariance_and_null_force(self, case):
        """C_up -> e^{i phi} C_up is an EXACT invariance of every w_i and e_L,i
        (|Psi|^2 and H Psi/Psi are phase-blind), so on the fixed samples:
        (a) E is exactly invariant under the finite rotation;
        (b) the gauge tangent g = i*C_up lies in the exact null space of the
            centered dO, hence vdot(g, f) == 0 to machine precision;
        (c) genericity: away from the gauge direction Im(f) is O(Re(f)) --
            the perturbed fixture must NOT have accidentally small Im(f),
            otherwise the RED assertions of T1/T2 would be vacuous.
        """
        s, e = case.offsets["mo_coefficients_up"]
        c_up = case.blocks[1].values.ravel().astype(np.complex128)
        assert case.blocks[1].name == "mo_coefficients_up"

        # (a) exact finite rotation: delta = (e^{ih} - 1) c
        h = 0.05
        thetas = np.zeros(case.K, dtype=np.complex128)
        thetas[s:e] = (np.exp(1j * h) - 1.0) * c_up
        E_rot = case.energy_at(thetas)
        E_0 = complex(np.sum(case.w * case.e) / case.w.sum())
        # The invariant is exact, but the two independently evaluated and
        # reduced energies inherit the weakest arithmetic zone on the path.
        # In mixed mode AO/Jastrow forward values are fp32; keep this test at
        # the standard strict floor for the active policy instead of applying
        # an fp64-only cancellation bound or skipping the invariant entirely.
        rotation_atol, _ = get_tolerance_min(_GAUGE_ROTATION_ZONES, "strict")
        assert abs(E_rot - E_0) < rotation_atol * max(1.0, abs(E_0)), "E not exactly gauge-invariant on fixed samples"

        # (b) gauge direction is in the null space of the centered dO
        g_gauge = 1j * c_up
        gauge_obs = case.dO[:, s:e] @ g_gauge  # == i*N_up - <i*N_up> == 0 per sample
        assert np.max(np.abs(gauge_obs)) < ATOL_GAUGE_NULLSPACE_EXACT, "gauge tangent not in the null space of centered dO"
        proj = np.vdot(g_gauge, case.f_ref[s:e])
        assert abs(proj) < ATOL_GAUGE_NULLSPACE_EXACT * np.linalg.norm(g_gauge) * max(
            np.max(np.abs(case.f_ref)), ATOL_GAUGE_REFERENCE_SCALE_FLOOR
        ), "complex force has a spurious component along the exact gauge direction"

        # (c) genericity of the fixture: Im(f) is a real signal here
        f_mo = case.f_ref[s:e]
        assert np.max(np.abs(f_mo.imag)) > 0.05 * np.max(np.abs(f_mo.real)), (
            "fixture not generic: Im(f) accidentally small, RED assertions would be vacuous"
        )

    @pytest.mark.heavy
    def test_get_gF_respects_gauge_and_carries_im(self, case):
        """The implementation's f must be complex AND respect the exact gauge
        null direction. A get_gF that returns a real vector, whose projection on
        i*C_up does not vanish, fails.
        """
        s, e = case.offsets["mo_coefficients_up"]
        c_up = case.blocks[1].values.ravel().astype(np.complex128)
        m = case.injected_mcmc()
        f_impl, _ = m.get_gF(num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=_N_SAMPLES, blocks=case.blocks)
        f_impl = np.asarray(f_impl, dtype=np.complex128)
        # Im must be carried at all (a Re-projecting get_gF gives exactly zero) ...
        assert np.max(np.abs(f_impl[s:e].imag)) > 0.2 * np.max(np.abs(case.f_ref[s:e].imag)), (
            "get_gF dropped Im(f) -- imaginary DOF of complex parameters frozen"
        )
        # ... and the gauge projection must vanish (loose bound: jackknife bias)
        proj = np.vdot(1j * c_up, f_impl[s:e])
        bound = _N_JACKKNIFE_RTOL * np.linalg.norm(c_up) * np.max(np.abs(f_impl[s:e]))
        assert abs(proj) < max(bound, ATOL_GAUGE_NULLSPACE_EXACT), "implementation force leaks into the exact gauge direction"
