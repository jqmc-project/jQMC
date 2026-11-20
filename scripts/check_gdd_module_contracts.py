#!/usr/bin/env python3
"""Check that the module-receipt contract inventory loads from the live SSoT.

The test/guarantee relation is many-to-many (gdd/README.md). The module
receipt layer (``gdd.engine.module_receipts.load_node_contracts``) therefore
holds one contract for every registered (guarantee, pytest node) edge, and a
node shared by several guarantees has one contract per guarantee. This gate
loads that inventory and the completion obligations built from it, and
compares the loaded edges with the edges read directly from the authored
YAML. A registration that the loader rejects, drops or duplicates then fails
``scripts/gdd_sync.py --check`` (pre-commit and the CI preflight)
instead of every receipt self-test at test time (issue #85).
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

from collections.abc import Iterable, Mapping
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


Edge = tuple[str, str]


def authored_edges(root: Path = ROOT) -> list[Edge]:
    """Every (guarantee, pytest node) registration, read from the authored YAML only."""
    edges: list[Edge] = []
    from gdd.engine.guarantee_store import load_module

    for source in sorted((root / "gdd/ssot/guarantees").glob("*.yaml")):
        document = load_module(source)
        section = "module_evidence" if source.name == "derivatives.yaml" else "guarantees"
        for guarantee in document.get(section) or ():
            evidence = guarantee.get("required_evidence")
            if not isinstance(evidence, list):
                continue
            for group in evidence:
                for node in group.get("nodes") or ():
                    edges.append((str(guarantee["id"]), str(node["pytest_node"])))
    return edges


def inventory_errors(contracts: Mapping[str, Iterable], edges: Iterable[Edge], obligation_edges: Iterable[Edge]) -> list[str]:
    """Compare loaded contracts and obligations with the authored edges.

    ``contracts`` maps each pytest node to its contracts (one per guarantee).
    ``obligation_edges`` are the (guarantee, node) pairs of the expected
    completion obligations; each loaded contract must own at least one.
    """
    errors: list[str] = []
    authored = list(edges)
    seen: set[Edge] = set()
    for edge in authored:
        if edge in seen:
            errors.append(f"{edge[0]}: pytest node registered twice in this guarantee: {edge[1]}")
        seen.add(edge)
    loaded: list[Edge] = []
    for node, owners in contracts.items():
        for contract in owners:
            if contract.pytest_node != node:
                errors.append(f"{node}: contract is filed under another node: {contract.pytest_node}")
            loaded.append((contract.guarantee_id, contract.pytest_node))
    loaded_set = set(loaded)
    if len(loaded) != len(loaded_set):
        errors.append("module contract inventory holds the same (guarantee, node) contract more than once")
    for guarantee_id, node in sorted(seen - loaded_set):
        errors.append(f"{guarantee_id}: registered node has no module contract: {node}")
    for guarantee_id, node in sorted(loaded_set - seen):
        errors.append(f"{guarantee_id}: module contract for an unregistered node: {node}")
    for guarantee_id, node in sorted(loaded_set - set(obligation_edges)):
        errors.append(f"{guarantee_id}: module contract has no completion obligation: {node}")
    return errors


def check(root: Path = ROOT) -> list[str]:
    from gdd.engine.guarantee_store import layout_errors
    from gdd.engine.module_receipts import load_node_contracts
    from gdd.engine.receipt_case_checks import expected_obligations

    # Issue 205: monolithic or split module ledgers; orphans, duplicates and misnamed files fail first.
    layout = layout_errors(root)
    if layout:
        return [f"guarantee layout: {error}" for error in layout]
    try:
        contracts = load_node_contracts()
        obligations = expected_obligations()
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return [f"module contract inventory does not load: {exc}"]
    obligation_edges = {(key.guarantee_id, key.pytest_node) for key in obligations}
    return inventory_errors(contracts, authored_edges(root), obligation_edges)


def main() -> int:
    errors = check()
    if errors:
        print("GDD module contract inventory check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("GDD module contract inventory loads: every registered (guarantee, node) edge has one module contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
