"""Unit tests for jqmc.trexio_wrapper element-vs-label handling.

TREXIO stores a nucleus' element identity (``nucleus_charge``, plus the
per-nucleus ``ecp_z_core`` when pseudised) separately from its free-form
``nucleus_label``. jQMC therefore reconstructs the atomic number from the
charge and keeps the raw label as ``atomic_labels`` -- so a QE-origin file
that tags magnetically inequivalent sites ('O' vs 'O1') round-trips without
ever parsing the label string for its element.
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

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
from jqmc.trexio_wrapper import read_trexio_file

# water_ccecp: O (ccECP, z_core=2) + 2 H. nucleus_charge stores the *valence*
# charge (O -> 6.0), so the full Z = 6 + z_core(2) = 8 exercises the
# charge+z_core reconstruction, not a trivial round(charge).
_WATER = Path(__file__).resolve().parents[1] / "trexio_example_files" / "water_ccecp_ccpvqz.h5"
requires_water = pytest.mark.skipif(not _WATER.is_file(), reason="water_ccecp_ccpvqz.h5 fixture not present")


class TestTrexioElementVsLabel:
    @pytest.mark.light
    @requires_water
    def test_untagged_fixture_element_equals_label(self):
        """Baseline: an untagged file yields element_symbols == atomic_labels."""
        sd = read_trexio_file(str(_WATER), store_tuple=True)[0]
        assert tuple(sd.atomic_labels) == ("O", "H", "H")
        assert tuple(sd.element_symbols) == ("O", "H", "H")
        assert tuple(sd.atomic_numbers) == (8, 1, 1)

    @pytest.mark.light
    @requires_water
    def test_tagged_label_splits_from_element(self, tmp_path):
        """An AFM-style 'O1' tag must keep element='O' (from charge+z_core)
        while preserving the raw 'O1' as the atomic_label.
        """
        tagged = tmp_path / "water_tagged.h5"
        shutil.copy(_WATER, tagged)
        with h5py.File(tagged, "r+") as h:
            ds = h["nucleus/nucleus_label"]
            vals = [v for v in ds[()]]
            vals[0] = b"O1"  # relabel the oxygen site
            ds[...] = np.array(vals, dtype=ds.dtype)

        sd = read_trexio_file(str(tagged), store_tuple=True)[0]
        # Raw label preserved verbatim ...
        assert tuple(sd.atomic_labels) == ("O1", "H", "H")
        # ... but the element is recovered from the charge (6.0) + ecp_z_core (2),
        # never by parsing the 'O1' string (which would be unmappable).
        assert tuple(sd.element_symbols) == ("O", "H", "H")
        assert tuple(sd.atomic_numbers) == (8, 1, 1)
