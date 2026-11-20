"""Single-rank unit tests for ``jqmc.jqmc_mcmc.allreduce_sum``.

The complex-aware ``allreduce_sum`` relies on viewing a ``complex128``
buffer as a length-``2N`` ``float64`` array and reducing through
``MPI.DOUBLE``. On a single-rank communicator the reduction is a no-op
SUM with one contributor, so the output must equal the input
bit-for-bit (this is what these tests pin); on multi-rank communicators
the same byte-view trick combined with the SUM linearity gives the
full estimator-correct complex reduction without depending on
the MPI build's complex-type support.

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

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
from mpi4py import MPI

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.jqmc_mcmc import allreduce_sum

_COMM_SELF = MPI.COMM_SELF


class TestAllreduceSum:
    @pytest.mark.light
    @pytest.mark.parametrize("shape", [(8,), (3, 5), (2, 4, 6)])
    def test_allreduce_sum_real_single_rank(self, shape):
        """Single-rank SUM == identity for ``float64`` arrays."""
        rng = np.random.default_rng(0)
        local = rng.standard_normal(shape).astype(np.float64)
        out = allreduce_sum(_COMM_SELF, local)
        assert out.shape == local.shape
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, local)
        # And the input is not mutated.
        rng2 = np.random.default_rng(0)
        np.testing.assert_array_equal(local, rng2.standard_normal(shape).astype(np.float64))

    @pytest.mark.light
    @pytest.mark.parametrize("shape", [(8,), (3, 5), (2, 4, 6)])
    def test_allreduce_sum_complex_single_rank(self, shape):
        """Single-rank SUM == identity for ``complex128`` arrays.

        The bit-exact byte-view trick (``view(float64)`` of a complex buffer)
        means ``Im`` is reduced just like ``Re`` -- so the single-rank no-op
        SUM must preserve both parts exactly.
        """
        rng = np.random.default_rng(1)
        re = rng.standard_normal(shape)
        im = rng.standard_normal(shape)
        local = (re + 1j * im).astype(np.complex128)
        out = allreduce_sum(_COMM_SELF, local)
        assert out.shape == local.shape
        assert out.dtype == np.complex128
        np.testing.assert_array_equal(out, local)

    @pytest.mark.light
    def test_allreduce_sum_complex_preserves_im_part(self):
        """Specifically pin that ``Im`` is NOT silently truncated -- this is
        the failure mode the helper exists to prevent against an accidental
        ``MPI.DOUBLE`` reduction of a complex buffer."""
        local = np.array([1.0 + 2.0j, 3.0 - 4.0j, 0.0 + 1e-100j], dtype=np.complex128)
        out = allreduce_sum(_COMM_SELF, local)
        np.testing.assert_array_equal(out, local)
        # Explicitly check Im parts to call out the value vs zero.
        np.testing.assert_array_equal(out.imag, np.array([2.0, -4.0, 1e-100]))

    @pytest.mark.light
    def test_allreduce_sum_rejects_complex64(self):
        """fp32-complex is forbidden by the complex precision policy."""
        local = np.array([1.0 + 1.0j], dtype=np.complex64)
        with pytest.raises(NotImplementedError, match="complex64 buffers are forbidden"):
            allreduce_sum(_COMM_SELF, local)

    @pytest.mark.light
    def test_allreduce_sum_rejects_unsupported_dtype(self):
        """Allreduce sum rejects unsupported dtype."""
        local = np.array([1, 2, 3], dtype=np.int64)
        with pytest.raises(TypeError, match="unsupported dtype"):
            allreduce_sum(_COMM_SELF, local)
