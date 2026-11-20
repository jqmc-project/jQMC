"""Coulomb-owned forwarding entry for the non-cubic negative-control checker.

The negative-control checker
``test_noncubic_cell_oracles.py::test_noncubic_oracles_reject_metric_and_truncation_faults``
is owned by the atomic-orbital ledger. One pytest node may carry only one module
contract, so the Coulomb guarantee MATH-COUL-EWALD-SKEWED-BASIS-REFERENCE-001 is
bound to this entry instead. It forwards to the SAME checker implementation with the
same seven fault cases, ids, markers and fixtures; it is not an independent oracle.
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

from tests.test_jqmc_kernel.test_noncubic_cell_oracles import basis_pairs as basis_pairs  # noqa: F401 (fixture)
from tests.test_jqmc_kernel.test_noncubic_cell_oracles import (
    test_noncubic_oracles_reject_metric_and_truncation_faults as _check_original_fault,
)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize(
    "fault",
    [
        "halve-ao-image-radius",
        "transpose-ao-image-metric",
        pytest.param(
            "halve-overlap-image-radius",
            marks=pytest.mark.gdd_electron_configuration(
                mode="not-required",
                reason=(
                    "The overlap fault branch evaluates the analytic overlap only; the one coordinate-named "
                    "symbol this case reaches is the nucleus array 'coords' read while the basis_pairs fixture builds "
                    "its Hamiltonians, not an electron or evaluation coordinate"
                ),
            ),
        ),
        "orthogonal-only-minimum-image",
        "transpose-home-cell-wrap",
        "drop-ecp-minimum-image",
        "orthogonal-only-g-index-box",
    ],
)
def test_coulomb_noncubic_oracles_reject_metric_and_truncation_faults(fault, monkeypatch, basis_pairs):
    """Forwarding entry: the same non-cubic fault checker, bound to the Coulomb Ewald guarantee.

    Forwards to test_noncubic_cell_oracles.py::test_noncubic_oracles_reject_metric_and_truncation_faults
    (the same implementation, the same seven fault cases and the same fixtures). It is
    not an independent oracle: the orthogonal-only-g-index-box row must fail the
    strongly skewed Ewald reference with its own message, as in the original node.
    """
    _check_original_fault(fault, monkeypatch, basis_pairs)


# The forwarding entry must keep the forwarded checker's exact ordered cases and markers.
if [(mark.name, mark.args) for mark in test_coulomb_noncubic_oracles_reject_metric_and_truncation_faults.pytestmark] != [
    (mark.name, mark.args) for mark in _check_original_fault.pytestmark
]:
    raise RuntimeError("forwarding entry cases drifted from the forwarded checker")
