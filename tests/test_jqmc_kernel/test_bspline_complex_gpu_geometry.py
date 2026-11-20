"""Complex B-spline stencil positions off the CPU backend stay exact at large translations (issue #543).

``AOs_bspline_complex_data._eval_u_core`` selects the stencil and its weights from the fractional
coordinates of each electron. In mixed precision (fp32 AO zone) it applies the fp64 wrap on every
backend. Casting the Cartesian positions to fp32 first gives the stencil position of an electron at
``|r|`` an error of about ``6e-8 |r|``; MCMC walkers are not wrapped, so ``|r|`` grows along a run, and
on a GPU such a cast puts the Bloch condition ``chi(r + n.cell) = exp(2 pi i k.n) chi(r)`` off by 5e-3 at
``n = (10001, -7003, 4005)`` (``test_bloch_phase_hermiticity`` bspline cases, mixed).

This check runs the non-CPU code path on any backend: while the leaf is evaluated, ``jax.default_backend``
reports ``"gpu"``, which is the only input of the route selection. It then applies the far-translation
Bloch law of ``test_bloch_phase_hermiticity`` to the quintic complex B-spline leaf on the cubic and the
monoclinic cell, under the ao_eval strict zone policy of the active mode (meaningful in mixed; in full the
fp32 route is not taken).
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

import jax
import numpy as np
import pytest

from tests.test_jqmc_kernel.test_bloch_phase_hermiticity import _LEAF_FAR_SHIFTS, _single_orbital_bloch_violations

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize("cell_name", ["cubic_10bohr", "monoclinic"])
def test_gpu_route_complex_bspline_keeps_the_bloch_law_far_from_the_home_cell(monkeypatch, cell_name):
    """The off-CPU route of the complex B-spline leaf obeys the Bloch law at far lattice translations."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    jax.clear_caches()  # trace the leaf afresh under the patched backend
    try:
        failures = _single_orbital_bloch_violations("bspline", cell_name, np.asarray(_LEAF_FAR_SHIFTS))
    finally:
        jax.clear_caches()
    assert failures == [], "; ".join(failures)
