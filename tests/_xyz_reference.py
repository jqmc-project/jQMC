"""Independent geometry reference for the XYZ export tests."""

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

from pathlib import Path

import h5py
import numpy as np

from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_BIT_EXACT


def assert_xyz_matches_trexio(xyz_file: Path, trexio_file: Path) -> None:
    """Check atom count, label order and every Cartesian component in angstroms."""
    with h5py.File(trexio_file, "r") as fixture:
        nuclei = fixture["nucleus"]
        expected_labels = nuclei["nucleus_label"].asstr()[:].tolist()
        # CODATA 2018 Bohr radius in angstroms (doi:10.1063/5.0064853).
        expected_coords = nuclei["nucleus_coord"][:] * 0.529177210903

    lines = xyz_file.read_text().splitlines()
    records = [line.split() for line in lines[2:]]
    assert int(lines[0]) == len(records) == len(expected_labels)
    assert [record[0] for record in records] == expected_labels
    coords = np.asarray([record[1:] for record in records], dtype=np.float64)
    np.testing.assert_allclose(coords, expected_coords, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_BIT_EXACT)
