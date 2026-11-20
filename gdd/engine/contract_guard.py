"""Semantic monotonicity checks for GDD contract changes.

Routine synchronization may add evidence and new requirements. It may not
delete, replace, narrow, or relax an already-reviewed contract. Intentional
changes are handled as a separate reviewed decision, never as a side effect of
regenerating derived files.
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

import copy
import json
from pathlib import Path
from typing import Any

from gdd.engine.ledger import authored_document, evidence_section
from gdd.engine.canonical_contract import KINETIC_OPERATION, ORBITAL_ACTIVATION


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _ordinary_guarantees(document: dict[str, Any], source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for guarantee in document.get(evidence_section(document, source), ()):
        if isinstance(guarantee.get("required_evidence"), list):
            result[guarantee["id"]] = guarantee
    return result


def _guarantee_contract(guarantee: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(guarantee)
    value.pop("required_evidence", None)
    value.pop("claims", None)
    # These fields duplicated contracts that are derived directly from each
    # evidence node.  They were removed from the authored SSoT in issue 223.
    value.pop("fixture_usage", None)
    value.pop("contract_authority", None)
    value.pop("cutover_requirement", None)
    scope = value.get("scope")
    if isinstance(scope, dict):
        scope.pop("expected_obligations", None)
        if not scope:
            value.pop("scope", None)
    return value


def _evidence_groups(guarantee: dict[str, Any]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for group in guarantee.get("required_evidence", ()):
        metadata = {key: value for key, value in group.items() if key != "nodes"}
        identity = _canonical(metadata)
        groups.setdefault(identity, set()).update(node["pytest_node"] for node in group.get("nodes", ()))
    return groups


def _node_contracts(guarantee: dict[str, Any], identity: str) -> dict[str, list[dict[str, Any]]]:
    """Keep node-owned requirements, not just the existence of its pytest id."""
    result: dict[str, list[dict[str, Any]]] = {}
    for group in guarantee.get("required_evidence", ()):
        if _canonical({key: value for key, value in group.items() if key != "nodes"}) != identity:
            continue
        for node in group.get("nodes", ()):
            result.setdefault(node["pytest_node"], []).append(
                {key: value for key, value in node.items() if key != "pytest_node"}
            )
    return result


def _node_contract_preserved(required: dict[str, Any], candidate: dict[str, Any]) -> bool:
    for key, value in required.items():
        if key not in candidate:
            return False
        # A new independent check is additive. Changing an existing check's
        # parameters is not: adding a zone to a resolver can loosen its tol.
        if key == "executable_checks" and isinstance(value, dict) and isinstance(candidate[key], dict):
            if any(
                name not in candidate[key] or _canonical(candidate[key][name]) != _canonical(check)
                for name, check in value.items()
            ):
                return False
        elif key == "executable_checks" and isinstance(value, list) and isinstance(candidate[key], list):
            if not {_canonical(check) for check in value} <= {_canonical(check) for check in candidate[key]}:
                return False
        elif (
            key == "canonical_fixture"
            and isinstance(value, dict)
            and isinstance(candidate[key], dict)
            and "orbital_activation" not in value
            and candidate[key].get("orbital_activation") == ORBITAL_ACTIVATION
            and value.get("operation") == KINETIC_OPERATION
        ):
            # This additional validator is conjunctive: it changes neither the
            # consumption input nor the original comparison/activation policy.
            # No other nested additions or changes are approved by this rule.
            retained = {name: item for name, item in candidate[key].items() if name != "orbital_activation"}
            if _canonical(retained) != _canonical(value):
                return False
        elif _canonical(candidate[key]) != _canonical(value):
            return False
    return True


def ledger_monotonicity_errors(
    base_document: dict[str, Any],
    current_document: dict[str, Any],
    source: Path,
) -> list[str]:
    """Return protected semantic changes between two versions of one ledger."""
    base = authored_document(base_document, source)
    current = authored_document(current_document, source)
    errors: list[str] = []
    base_guarantees = _ordinary_guarantees(base, source)
    current_guarantees = _ordinary_guarantees(current, source)
    for guarantee_id, before in base_guarantees.items():
        after = current_guarantees.get(guarantee_id)
        if after is None:
            errors.append(f"{source.name}: removed guarantee {guarantee_id}")
            continue
        before_contract = _guarantee_contract(before)
        after_contract = _guarantee_contract(after)
        if before_contract != after_contract:
            errors.append(f"{source.name}: changed protected contract of {guarantee_id}")
        before_groups = _evidence_groups(before)
        after_groups = _evidence_groups(after)
        for identity, before_nodes in before_groups.items():
            after_nodes = after_groups.get(identity)
            if after_nodes is None:
                errors.append(f"{source.name}: removed or changed evidence group in {guarantee_id}: {identity}")
                continue
            removed = sorted(before_nodes - after_nodes)
            if removed:
                errors.append(f"{source.name}: removed evidence from {guarantee_id}: {removed}")
            before_contracts = _node_contracts(before, identity)
            after_contracts = _node_contracts(after, identity)
            for node in sorted(before_nodes & after_nodes):
                # The authored projection has removed counts and snapshots.
                # Everything remaining (e.g. executable_checks, fixture scope,
                # precision and policy choices) is a reviewed requirement.
                for required in before_contracts[node]:
                    if not any(_node_contract_preserved(required, candidate) for candidate in after_contracts[node]):
                        errors.append(f"{source.name}: changed protected node contract in {guarantee_id}: {node}")

    # Specialized derivative guarantees and every other top-level contract are
    # protected exactly. Ordinary guarantees permit evidence additions and the
    # separately validated provenance growth above, not scientific rule changes.
    section = evidence_section(base, source)
    before_other = copy.deepcopy(base)
    after_other = copy.deepcopy(current)
    for document in (before_other, after_other):
        document.pop("ledger_extensions", None)
        document.pop("known_cutover_gaps", None)
        document.pop("known_migration_gaps", None)
        metadata = document.get("document")
        if isinstance(metadata, dict):
            metadata.pop("authority", None)
        enforcement_map = document.get("enforcement_map")
        if isinstance(enforcement_map, dict):
            enforcement_map.pop("ledger_extensions", None)
            enforcement_map.pop("known_cutover_gaps", None)
            enforcement_map.pop("known_migration_gaps", None)
            if source.name == "derivatives.yaml":
                enforcement_map.pop("symbol_contract", None)
    for document in (before_other, after_other):
        for row_section in ("guarantees", "module_evidence"):
            for row in document.get(row_section, ()):
                row.pop("fixture_usage", None)
                row.pop("contract_authority", None)
                row.pop("cutover_requirement", None)
        specialized_rows = [row for row in document.get(section, ()) if not isinstance(row.get("required_evidence"), list)]
        document[section] = specialized_rows
    if before_other != after_other:
        errors.append(f"{source.name}: changed specialized or top-level protected contract")
    return errors


def monotonic_structure_errors(base: Any, current: Any, path: str) -> list[str]:
    """Reject removal/replacement while permitting additions to registries."""
    if isinstance(base, dict):
        if not isinstance(current, dict):
            return [f"{path}: mapping was replaced"]
        errors: list[str] = []
        for key, value in base.items():
            if key not in current:
                errors.append(f"{path}: removed key {key}")
            else:
                errors.extend(monotonic_structure_errors(value, current[key], f"{path}.{key}"))
        return errors
    if isinstance(base, list):
        if not isinstance(current, list):
            return [f"{path}: list was replaced"]
        current_values = {_canonical(value) for value in current}
        removed = [value for value in base if _canonical(value) not in current_values]
        return [f"{path}: removed or changed list entries {removed}"] if removed else []
    return [] if base == current else [f"{path}: changed {base!r} to {current!r}"]


def _count(value: Any) -> int | None:
    return value if type(value) is int else None


def _shrunk(path: str, label: str, before: Any, after: Any) -> list[str]:
    old, new = _count(before), _count(after)
    if old is None:
        return []
    if new is None or new < old:
        return [f"{path}: {label} decreased from {old} to {after!r}; explicit review required"]
    return []


def _superset_errors(path: str, label: str, before: Any, after: Any) -> list[str]:
    """Every base list entry must survive; additional entries are free."""
    if not isinstance(before, list):
        return []
    if not isinstance(after, list):
        return [f"{path}: {label} was removed or replaced; explicit review required"]
    remaining = {_canonical(value) for value in after}
    lost = [value for value in before if _canonical(value) not in remaining]
    return [f"{path}: {label} lost trusted entries {lost}; explicit review required"] if lost else []


def derived_obligation_errors(before: dict[str, Any], after: dict[str, Any], path: str) -> list[str]:
    """Reject per-node narrowing of a regenerated binding relative to the trusted base.

    Precision modes are compared separately (skip_protection.precision_removal_errors).
    This rule covers the other derived requirements a test-source edit plus
    re-materialization could shrink: guarantees, nodes, case counts, backends
    and execution profiles. Additions (a new guarantee, node, backend or
    profile) are always allowed. The only accepted backend substitution is
    python -> xla: the materializer labels a node python only when it reaches
    no JAX/XLA call, so gaining one moves the receipt to the stricter XLA
    identity. The reverse and any rust loss are rejected. Observation kinds
    are not compared here (their classifier is still evolving).
    """
    if not isinstance(before, dict) or not isinstance(before.get("guarantees"), dict):
        raise ValueError(f"{path}: trusted derived binding has no guarantee mapping")
    if not isinstance(after, dict) or not isinstance(after.get("guarantees"), dict):
        return [f"{path}: derived guarantee mapping was removed or replaced; explicit review required"]
    errors: list[str] = []
    for guarantee, payload in before["guarantees"].items():
        current = after["guarantees"].get(guarantee)
        where = f"{path}::{guarantee}"
        if not isinstance(current, dict):
            errors.append(f"{where}: removed trusted derived guarantee; explicit review required")
            continue
        errors.extend(
            _shrunk(where, "expected_obligations", payload.get("expected_obligations"), current.get("expected_obligations"))
        )
        current_nodes = current.get("nodes") if isinstance(current.get("nodes"), dict) else {}
        for node, entry in (payload.get("nodes") or {}).items():
            node_where = f"{where}::{node}"
            after_entry = current_nodes.get(node)
            if not isinstance(after_entry, dict):
                errors.append(f"{node_where}: removed trusted derived node; explicit review required")
                continue
            errors.extend(_shrunk(node_where, "expected_cases", entry.get("expected_cases"), after_entry.get("expected_cases")))
            binding = entry.get("case_binding") if isinstance(entry.get("case_binding"), dict) else {}
            after_binding = after_entry.get("case_binding") if isinstance(after_entry.get("case_binding"), dict) else {}
            execution = binding.get("execution") if isinstance(binding.get("execution"), dict) else {}
            after_execution = after_binding.get("execution") if isinstance(after_binding.get("execution"), dict) else {}
            if "backends" in execution:
                required = execution["backends"]
                realized = after_execution.get("backends")
                if isinstance(required, list) and isinstance(realized, list) and "python" in required and "xla" in realized:
                    required = [backend for backend in required if backend != "python"]
                errors.extend(_superset_errors(node_where, "execution.backends", required, realized))
            if "profiles" in execution:
                errors.extend(
                    _superset_errors(node_where, "execution.profiles", execution["profiles"], after_execution.get("profiles"))
                )
            fixture = binding.get("fixture") if isinstance(binding.get("fixture"), dict) else {}
            after_fixture = after_binding.get("fixture") if isinstance(after_binding.get("fixture"), dict) else {}
            errors.extend(
                _shrunk(
                    node_where, "fixture.expected_cases", fixture.get("expected_cases"), after_fixture.get("expected_cases")
                )
            )
    return errors


def universe_errors(before: dict[str, Any], after: Any, path: str) -> list[str]:
    """The reviewed universe may grow; a row, guarantee count or node count may not shrink.

    The per-row sha256 legitimately changes when evidence is added, so it is
    not compared here; per-node expected_cases are protected through the
    derived bindings (derived_obligation_errors).
    """
    if not isinstance(after, dict):
        return [f"{path}: universe document was removed or replaced"]
    errors = monotonic_structure_errors(before.get("document"), after.get("document"), f"{path}.document")
    errors.extend(
        monotonic_structure_errors(before.get("enforcement_map"), after.get("enforcement_map"), f"{path}.enforcement_map")
    )
    after_rows = {row.get("ledger"): row for row in after.get("universe") or () if isinstance(row, dict)}
    for row in before.get("universe") or ():
        ledger = row.get("ledger")
        current = after_rows.get(ledger)
        if current is None:
            errors.append(f"{path}: removed universe row for {ledger}; explicit review required")
            continue
        for key in ("guarantees", "nodes"):
            errors.extend(_shrunk(f"{path}::{ledger}", key, row.get(key), current.get(key)))
    return errors


def test_definition_inclusion_errors(before_source: str, after_source: str | None, path: str) -> list[str]:
    """Require every trusted framework-test definition to remain present."""
    from gdd.engine.mapping import qualified_test_source_definitions

    if after_source is None:
        return [f"{path}: removed trusted framework test file"]
    try:
        before = qualified_test_source_definitions(before_source, path)
        after = qualified_test_source_definitions(after_source, path)
    except (SyntaxError, ValueError) as exc:
        return [f"{path}: cannot inventory trusted framework tests: {exc}"]
    missing = sorted(before - after)
    return [f"{path}: removed trusted framework test definitions {missing}"] if missing else []
