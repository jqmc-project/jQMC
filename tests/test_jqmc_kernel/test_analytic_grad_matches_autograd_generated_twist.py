"""Generated route-fidelity matrix -- TWIST k!=0 (``bc == twist``) split.

The machine-generated covering-array matrix (see ``_generated_matrix_base``) partitioned by
boundary condition for CI parallelism: this file hosts the HEAVY twist cells (complex-Bloch pGTO /
plane-wave / B-spline / composite at k!=0; dtype-doubled traces). The molecular (``bc == mol``) and
periodic-Gamma (``bc == pbc_gamma``) subsets live in the sibling split files; the UNION of the three
is byte-identical to the pre-split monolith.
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

pytestmark = [pytest.mark.with_chdir]

from tests.test_jqmc_kernel._generated_matrix_base import (
    _AXES,
    _QUANTITIES,
    _cell_id,
    _clear_bridge_caches,  # noqa: F401  (autouse fixture, re-exported for collection)
    _generate_matrix,
    make_inventory_report,
    run_matrix_cell,
)

# TWIST subset: cell[3] is ALWAYS the bc field (legacy 4-tuple or 5-tuple), so filter on it.
_MATRIX = [(c, h) for c, h in _generate_matrix() if c[3] == "twist"]
assert _MATRIX, "twist (bc == twist) subset of the generated matrix must be non-empty"

_CELL_PARAMS = [
    pytest.param(cell, marks=(pytest.mark.heavy if heavy else pytest.mark.light), id=_cell_id(cell)) for cell, heavy in _MATRIX
]

_matrix_report = make_inventory_report(_MATRIX, "twist")  # session-scoped autouse inventory


@pytest.mark.parametrize("axis", _AXES)
@pytest.mark.parametrize("quantity", _QUANTITIES)
@pytest.mark.parametrize("cell", _CELL_PARAMS)
def test_matrix_cell(cell, quantity, axis):
    receipt = run_matrix_cell(cell, quantity, axis)
    assert receipt.status == "passed"
