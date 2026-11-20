"""The public accelerated-SR gamma entry agrees with independent polynomial roots."""

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

from jqmc import jqmc_optimizer_kernel as kernel
from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE

jax.config.update("jax_enable_x64", True)
pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize("H1", [-0.4, 0.4])
@pytest.mark.parametrize("H2", [0.7, -1.3])
def test_public_asr_gamma_matches_independent_polynomial_roots(H1, H2):
    """The public aSR entry returns the smaller-magnitude real stationary root.

    Independent NumPy polynomial roots check both signs of the linear-energy
    coefficient and both signs of H2-H0*S2, without calling the shared selector.
    The residual independently checks the derivative numerator itself.
    """
    H0, S2 = -0.8, 0.6
    coefficients = np.array([-H1 * S2, H2 - H0 * S2, H1])
    roots = np.roots(coefficients)
    assert np.all(np.isreal(roots)) and np.prod(roots) < 0
    reference = roots[np.argmin(np.abs(roots))]
    actual = kernel.compute_asr_gamma(H0, H1, H2, S2)
    assert np.isfinite(actual) and actual != 0
    np.testing.assert_allclose(actual, reference, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
    np.testing.assert_allclose(np.polyval(coefficients, actual), 0, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
