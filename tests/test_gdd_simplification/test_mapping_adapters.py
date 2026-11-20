"""The core mapping reads authored declarations, not generated evidence."""

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

import json

import pytest
import yaml

from tests._gdd_ssot_layout import write_ssot_text

from gdd.engine.mapping import (
    authored_mapping,
    collected_json_nodes,
    collection_manifest_nodes,
    mapping_errors,
    native_test_nodes,
)


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _ledger(root, name="component", **document):
    directory = root / "gdd/ssot/guarantees"
    directory.mkdir(parents=True, exist_ok=True)
    write_ssot_text((directory / f"{name}.yaml"), yaml.safe_dump(document))


@pytest.mark.parametrize("shape", ["matrix-and-module", "native-and-python", "duplicate-id"])
def test_matrix_and_ordinary_guarantees_are_both_included(tmp_path, shape):
    """No authored guarantee or evidence shape is silently dropped from the relation.

    The matrix dictionary is kept beside module_evidence, supplemental native
    evidence is kept beside the Python nodes, and a duplicated guarantee id
    fails instead of one copy overwriting the other.
    """
    node = "tests/test_component.py::test_value"
    if shape == "matrix-and-module":
        _ledger(
            tmp_path,
            "derivatives",
            guarantees=[{"id": "MATRIX", "required_evidence": {"nodes": [{"pytest_node": node}]}}],
            module_evidence=[{"id": "MODULE", "required_evidence": [{"nodes": [{"pytest_node": node}]}]}],
        )
        assert authored_mapping(tmp_path) == {"MATRIX": {node}, "MODULE": {node}}
    elif shape == "native-and-python":
        native = "jqmc_rs::determinant::tests::inverse"
        _ledger(
            tmp_path,
            guarantees=[{"id": "G", "native_evidence": [native], "required_evidence": [{"nodes": [{"pytest_node": node}]}]}],
        )
        assert authored_mapping(tmp_path) == {"G": {native, node}}
    else:
        for name in ("one", "two"):
            _ledger(tmp_path, name, guarantees=[{"id": "G", "required_evidence": []}])
        with pytest.raises(ValueError, match="duplicate guarantee"):
            authored_mapping(tmp_path)


@pytest.mark.parametrize("route", ["fd-fixture", "pytest-symbols"])
def test_fd_guarantee_uses_only_its_named_exact_fixture_nodes(tmp_path, route):
    """Unrelated nodes cannot fill a guarantee's registration.

    An FD guarantee takes only its own fixture set's nodes; a pytest-symbol
    route declaration binds the named functions, not every test of the file.
    """
    node = "tests/test_component.py::test_value"
    if route == "fd-fixture":
        _ledger(
            tmp_path,
            fixture_sets=[
                {"id": "FD", "reviewed_evidence": {"nodes": [{"pytest_node": node}]}},
                {"id": "OTHER", "reviewed_evidence": {"nodes": [{"pytest_node": "tests/test_other.py::test_other"}]}},
            ],
            guarantees=[{"id": "G", "scope": {"fixture_set": "FD"}, "required_evidence": {}}],
        )
    else:
        _ledger(
            tmp_path,
            guarantees=[
                {"id": "G", "required_evidence": {"pytest": {"path": "tests/test_component.py", "symbols": ["test_value"]}}}
            ],
        )
    assert authored_mapping(tmp_path) == {"G": {node}}


def test_native_tests_are_discovered_without_an_inventory_pin(tmp_path):
    """Adding a native test requires a guarantee edge, not a count/digest repin."""
    source = tmp_path / "jqmc/rs/src/determinant.rs"
    source.parent.mkdir(parents=True)
    write_ssot_text(source, "mod tests { #[test] fn inverse() {} }\n// #[test] fn fake() {}\n")
    nodes = native_test_nodes(tmp_path)
    assert nodes == {"jqmc_rs::determinant::tests::inverse"}
    assert mapping_errors({}, nodes) == ["jqmc_rs::determinant::tests::inverse: no guarantee registered"]


@pytest.mark.parametrize("collection", ["empty-report", "failed-collection"])
def test_static_definition_without_actual_collection_fails(tmp_path, collection):
    """A disabled or uncollectable test cannot back a guarantee at collection time.

    A failed pytest collection that still lists the node before selection is
    rejected instead of proving mapping closure.
    """
    node = "tests/test_component.py::test_value"
    if collection == "empty-report":
        report = tmp_path / "collection.json"
        write_ssot_text(report, "[]")
        assert mapping_errors({"G": [node]}, collected_json_nodes([report]))
    else:
        payload = {"schema": "gdd-pytest-execution.v1", "kind": "collection", "exitstatus": 2, "before_selection": [node]}
        write_ssot_text((tmp_path / "gdd-collection-full.json"), json.dumps(payload))
        with pytest.raises(ValueError):
            collection_manifest_nodes(tmp_path)
