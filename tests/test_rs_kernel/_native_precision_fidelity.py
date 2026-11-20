"""Fresh native derivative fixtures and the central homogeneous-block predicate."""

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

import tempfile
from pathlib import Path

import numpy as np

from jqmc._precision import is_mixed_precision_enabled
from jqmc.hamiltonians import Hamiltonian_data


def full_source_hamiltonian_factory(build):
    """Freeze full physical input once, then load fresh model leaves in each policy.

    The matrix builder caches projected model objects without a precision key.
    Clearing that cache before the full source build prevents a prior mixed
    construction from supplying downcast source data. The source is serialized;
    neither policy receives the builder's cached model or the other policy's model.
    """
    directory = tempfile.TemporaryDirectory(prefix="jqmc_native_fidelity_")
    source = None

    def factory():
        nonlocal source
        if source is None:
            assert not is_mixed_precision_enabled(), "physical source must be built in full first"
            from tests.test_jqmc_kernel._generated_matrix_base import _qe_proj_pieces

            _qe_proj_pieces.cache_clear()
            source = Path(directory.name) / "full_physical_source.h5"
            build().save_to_hdf5(source)
        return Hamiltonian_data.load_from_hdf5(source)

    return factory


def assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Apply the unchanged parameter-derivative budget to one homogeneous block."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()
