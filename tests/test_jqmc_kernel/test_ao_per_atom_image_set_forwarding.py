"""Atomic-orbital-owned forwarding entry for the per-atom image-set checker.

The checker ``test_structure.py::TestStructureImageList::test_per_atom_image_indices_match_independent_active_set``
is owned by the structure ledger. One pytest node may carry only one module contract,
so the AO guarantee MATH-AO-OFF-CELL-NUCLEUS-BRUTE-FORCE-001 is bound to this entry
instead. It forwards to the SAME checker implementation with the same cell and
fractional-atom cases, ids and markers; it is not an independent oracle.
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

import pytest

from tests.conftest import CELL_FIXTURES
from tests.test_jqmc_kernel.test_structure import TestStructureImageList as _StructureImageList

pytestmark = [pytest.mark.with_chdir]


@pytest.mark.light
@pytest.mark.parametrize("cell_name", list(CELL_FIXTURES))
@pytest.mark.parametrize(
    "fractional_atom",
    [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (0.0, 0.5, 0.5),
        (0.15, 0.4, 0.75),
        (-0.1, 0.4, 0.75),
        (0.15, 1.1, 0.75),
        (-1.0e-12, 0.5, 1.0),
        (-1.1, 0.4, 2.15),
    ],
)
def test_ao_per_atom_image_indices_match_independent_active_set(cell_name, fractional_atom):
    """Forwarding entry: the same per-atom image-set checker, bound to the AO off-cell nucleus guarantee.

    Forwards to test_structure.py::TestStructureImageList::test_per_atom_image_indices_match_independent_active_set
    (the same implementation and the same cell / fractional-atom cases). It is not an
    independent oracle: the production per-atom integer image rows must equal the
    independent active-set enumeration exactly, as in the original node.
    """
    _StructureImageList.test_per_atom_image_indices_match_independent_active_set(
        _StructureImageList(), cell_name, fractional_atom
    )


# The forwarding entry must keep the forwarded checker's exact ordered cases and markers.
if [(mark.name, mark.args) for mark in test_ao_per_atom_image_indices_match_independent_active_set.pytestmark] != [
    (mark.name, mark.args) for mark in _StructureImageList.test_per_atom_image_indices_match_independent_active_set.pytestmark
]:
    raise RuntimeError("forwarding entry cases drifted from the forwarded checker")
