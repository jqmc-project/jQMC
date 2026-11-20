"""AO_radial_shell_data round-trip + dedup guardrails.

The radial/shell level (`AO_radial_shell_data`) stores the contracted-Gaussian
radial basis once per shell; the standalone `expand_radial_to_ao_prims`
reconstructs the per-AO-primitive `(exponents, coefficients)` the kernel
consumes. These tests pin that the expansion reproduces the current per-AO
layout byte-for-byte (so swapping the AO leaf onto the nested radial level
keeps AO output identical) and that the dedup actually shrinks storage for
systems with p/d/.. shells.
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

import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.atomic_orbital import AO_radial_shell_data, expand_radial_to_ao_prims
from jqmc.trexio_wrapper import read_trexio_file


def _load_aos(trexio_file: str):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file)
    if not os.path.exists(path):
        pytest.skip(f"missing fixture {trexio_file}")
    out = read_trexio_file(trexio_file=path, store_tuple=True)
    return out[1]  # aos_data


class TestAORadialShell:
    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["N_ae_ccpvdz_cart.h5", "water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5"])
    def test_expand_round_trips_byte_identical(self, trexio_file: str):
        """expand(from_aos_data(aos)) reproduces the per-AO exponents/coefficients exactly."""
        aos = _load_aos(trexio_file)
        radial, ao_prim_to_shell_prim = AO_radial_shell_data.from_aos_data(aos)
        exp_full, coef_full = expand_radial_to_ao_prims(radial, ao_prim_to_shell_prim)
        np.testing.assert_array_equal(np.asarray(exp_full), np.asarray(aos.exponents, dtype=np.float64))
        np.testing.assert_array_equal(np.asarray(coef_full), np.asarray(aos.coefficients, dtype=np.float64))

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["N_ae_ccpvdz_cart.h5", "water_ccecp_ccpvqz.h5"])
    def test_dedup_reduces_storage_and_maps_are_consistent(self, trexio_file: str):
        """Systems with p/d/.. shells dedup: fewer shell-prims than AO-prims; maps consistent."""
        aos = _load_aos(trexio_file)
        radial, ao_prim_to_shell_prim = AO_radial_shell_data.from_aos_data(aos)
        # dedup actually happened (these bases have l >= 1 shells)
        assert radial.num_shell_prim < int(aos.num_ao_prim)
        # map shapes / ranges
        assert len(ao_prim_to_shell_prim) == int(aos.num_ao_prim)
        assert len(radial.shell_prim_to_shell) == radial.num_shell_prim
        assert radial.num_shell == len(radial.nucleus_index) == len(radial.angular_momentum)
        assert int(ao_prim_to_shell_prim.min()) >= 0
        assert int(ao_prim_to_shell_prim.max()) < radial.num_shell_prim
        # every shell-prim's parent shell index is in range
        spts = np.asarray(radial.shell_prim_to_shell)
        assert int(spts.min()) >= 0
        assert int(spts.max()) < radial.num_shell
