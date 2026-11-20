"""The module-receipt contract inventory honors many-to-many test registration (issue #85)."""

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

import copy
from dataclasses import replace

import pytest

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_SHARED_NODE = "tests/test_scratch_shared_evidence.py::test_shared"
_GUARANTEES = ("IMPL-SCRATCH-SHARED-A-001", "IMPL-SCRATCH-SHARED-B-001")


def _clear_caches(module_receipts, receipt_case_checks) -> None:
    module_receipts.load_node_contracts.cache_clear()
    receipt_case_checks.expected_obligations.cache_clear()
    receipt_case_checks.node_obligations.cache_clear()


def _document(node: dict, guarantee_ids: tuple[str, ...], verdict_method: str) -> dict:
    return {
        "guarantees": [
            {"id": guarantee_id, "required_evidence": [{"verdict_method": verdict_method, "nodes": [copy.deepcopy(node)]}]}
            for guarantee_id in guarantee_ids
        ]
    }


def test_every_registered_guarantee_node_edge_has_its_own_module_contract(monkeypatch, tmp_path):
    """Each (guarantee, pytest node) registration yields one module contract and its obligations.

    gdd/README.md allows a test to support several guarantees. The live
    inventory must load, and its (guarantee, node) edges must equal the edges
    read directly from the authored YAML. In an isolated two-guarantee ledger,
    one node shared by both guarantees gets one contract and one obligation
    set per guarantee, while the same node registered
    twice under one guarantee is still rejected.
    """
    from gdd.engine import module_receipts, receipt_case_checks
    from scripts import check_gdd_module_contracts as gate

    # Live repository: the loader succeeds and drops or duplicates no edge.
    assert gate.check() == []
    live = module_receipts.load_node_contracts()
    edges = gate.authored_edges()
    assert edges and len(edges) == len(set(edges))
    assert {(c.guarantee_id, c.pytest_node) for owners in live.values() for c in owners} == set(edges)

    # The comparison itself rejects a dropped edge (non-vacuity of the gate).
    dropped_node, dropped_owners = next(iter(live.items()))
    pruned = {**live, dropped_node: dropped_owners[1:]}
    obligation_edges = {(c.guarantee_id, c.pytest_node) for owners in live.values() for c in owners}
    assert any("has no module contract" in error for error in gate.inventory_errors(pruned, edges, obligation_edges))

    template = next(
        c
        for owners in live.values()
        for c in owners
        if len(owners) == 1
        and len(c.cases) == 1
        and not c.cases[0].get("parameter_id")
        and c.executable_checks is None
        and c.canonical_fixture is None
        and c.condition_check_iterations is None
        and "full" in c.case_binding["execution"]["precision_modes"]
        and [p["world_size"] for p in c.case_binding["execution"]["profiles"]] == [1]
    )
    binding = copy.deepcopy(template.case_binding)
    node = {"pytest_node": _SHARED_NODE, "case_binding": binding}
    verdict_method = binding["comparison"]["verdict_method"]

    (tmp_path / "scratch.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module_receipts, "GUARANTEE_DIR", tmp_path)
    shared_document = _document(node, _GUARANTEES, verdict_method)
    monkeypatch.setattr(module_receipts, "load_ledger", lambda path: copy.deepcopy(shared_document))
    _clear_caches(module_receipts, receipt_case_checks)
    try:
        contracts = module_receipts.load_node_contracts()
        assert set(contracts) == {_SHARED_NODE}
        assert sorted(c.guarantee_id for c in contracts[_SHARED_NODE]) == sorted(_GUARANTEES)

        obligations = receipt_case_checks.expected_obligations()
        per_guarantee = {
            guarantee_id: {key for key in obligations if key.guarantee_id == guarantee_id} for guarantee_id in _GUARANTEES
        }
        assert all(per_guarantee.values())
        strip = [{replace(key, guarantee_id="") for key in keys} for keys in per_guarantee.values()]
        assert strip[0] == strip[1]
        assert set(receipt_case_checks.node_obligations(_SHARED_NODE)) == set(obligations)
        shared_edges = [(guarantee_id, _SHARED_NODE) for guarantee_id in _GUARANTEES]
        obligation_edges = {(key.guarantee_id, key.pytest_node) for key in obligations}
        assert gate.inventory_errors(contracts, shared_edges, obligation_edges) == []

        # The same node twice under ONE guarantee is a malformed duplicate.
        duplicated = _document(node, _GUARANTEES[:1], verdict_method)
        duplicated["guarantees"][0]["required_evidence"][0]["nodes"].append(copy.deepcopy(node))
        monkeypatch.setattr(module_receipts, "load_ledger", lambda path: copy.deepcopy(duplicated))
        _clear_caches(module_receipts, receipt_case_checks)
        with pytest.raises(ValueError, match="registered more than once under IMPL-SCRATCH-SHARED-A-001"):
            module_receipts.load_node_contracts()
    finally:
        _clear_caches(module_receipts, receipt_case_checks)
