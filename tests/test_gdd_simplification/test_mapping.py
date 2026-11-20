"""Focused framework examples for the simplified relation and coverage rules.

These tests exercise pure helpers, without jQMC/JAX imports or runtime tracing.
Repository collection and CI integration are separate migration steps.
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

import pytest
import yaml

from tests._gdd_ssot_layout import write_ssot_text

from gdd.engine.coverage import missing_requirements
from gdd.engine.mapping import authored_mapping, mapping_errors


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


NODE_A = "tests/test_component.py::test_value"


@pytest.mark.parametrize("ledger", ["direct", "unbound-matrix"])
def test_empty_guarantee_fails(tmp_path, ledger):
    """A guarantee cannot claim backing from an empty evidence list.

    The ``unbound-matrix`` case reads a matrix-shaped guarantee without a
    binding through the authored-ledger adapter: it must stay an uncovered
    guarantee instead of disappearing from the relation.
    """
    if ledger == "direct":
        relation = {"VALUE": []}
    else:
        directory = tmp_path / "gdd/ssot/guarantees"
        directory.mkdir(parents=True)
        document = {"guarantees": [{"id": "VALUE", "required_evidence": {"operation": "unbound"}}]}
        write_ssot_text(directory / "component.yaml", yaml.safe_dump(document))
        relation = authored_mapping(tmp_path)
    assert mapping_errors(relation, []) == ["VALUE: no test evidence registered"]


def test_deleted_test_fails_for_every_affected_guarantee():
    """A deleted shared test is diagnosed at each remaining registration."""
    assert mapping_errors({"VALUE": [NODE_A], "RELATION": [NODE_A]}, []) == [
        f"RELATION: registered test not collected: {NODE_A}",
        f"VALUE: registered test not collected: {NODE_A}",
    ]


def test_different_cases_cannot_manufacture_one_interaction():
    """B-spline with twist and Gaussian with DTN do not cover B-spline/DTN/twist."""
    required = [{"basis": "bspline", "jastrow": "dtn", "twist": True}]
    actual = [
        {"basis": "bspline", "jastrow": "none", "twist": True},
        {"basis": "gto", "jastrow": "dtn", "twist": True},
    ]
    assert missing_requirements(required, actual) == required
