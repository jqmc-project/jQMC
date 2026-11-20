"""Canonical real-consumer map: the ten registration fields per consumed pattern.

``gdd/ssot/fixtures/wavefunctions.yaml``
declares, under ``evidence_required_from_each_subscribing_feature``, the ten
registration fields every canonical subscription must carry.  This module
builds one row per (subscribing node, consumed canonical pattern) and fills
each field **only** from

* the node's own ``canonical_fixture`` requirement,
* the node's own ``required_parameter_cases`` / ``required_precision_modes``,
* the dedicated manifests: the canonical suite (``required_canonical_cases``),
  the node's own ``executable_checks`` and ``gdd/ssot/verdicts.yaml``.

The realized side (the materialized case binding in ``gdd/derived``) is used
only to detect drift between what the node-level manifest demands and what the
node registers and consumes.
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

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from gdd.engine.canonical_contract import (
    KINETIC_OPERATION,
    canonical_comparison_precision_modes,
    canonical_comparisons,
    canonical_precision_coverage_errors,
    required_canonical_cases,
)
from gdd.engine.ledger import iter_registered_nodes, load_ledger

ROOT = Path(__file__).resolve().parents[2]
_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

#: The single input-binding kind a canonical pattern can have: every pattern is
#: built by the canonical provider from its reviewed signature, never read from a
#: committed artifact (``gdd/ssot/fixtures/input_bindings.yaml`` kind vocabulary).
CANONICAL_INPUT_BINDING_KIND = "parameter-set"


def _load(path: Path) -> dict[str, Any]:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_LOADER) or {}


def registration_fields(root: Path = ROOT) -> list[str]:
    document = _load(root / "gdd/ssot/fixtures/wavefunctions.yaml")
    return list(document["evidence_required_from_each_subscribing_feature"]["registration_fields"])


def _policies(root: Path) -> set[str]:
    document = _load(root / "gdd/ssot/verdicts.yaml")
    return {row["id"] for row in document.get("comparison_policies", ())}


def node_consumer_rows(
    guarantee_id: str,
    node: dict[str, Any],
    suite: dict[str, Any],
    policies: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Ten-field rows for one subscribing node and its drift errors."""
    requirement = node["canonical_fixture"]
    name = node["pytest_node"]
    errors: list[str] = []
    if requirement.get("suite") != suite.get("id"):
        errors.append(f"{name}: canonical-suite-id {requirement.get('suite')!r} is not the manifest suite {suite.get('id')!r}")
    selector = requirement.get("selector")
    manifest_patterns = [case["id"] for case in required_canonical_cases(requirement, suite)]
    declared_cases = node.get("required_parameter_cases")
    if declared_cases is not None:
        declared = [case.get(selector) for case in declared_cases if isinstance(case, dict)]
        if sorted(map(str, declared)) != sorted(manifest_patterns) or len(declared) != len(declared_cases):
            errors.append(
                f"{name}: required_parameter_cases {sorted(map(str, declared))} drift from the manifest-required "
                f"canonical patterns {sorted(manifest_patterns)}"
            )
    binding = node.get("case_binding") or {}
    realized = [case.get("selector", {}).get(selector) for case in (binding.get("fixture") or {}).get("concrete_cases", ())]
    if sorted(map(str, realized)) != sorted(manifest_patterns) or len(realized) != len(manifest_patterns):
        errors.append(
            f"{name}: consumed canonical patterns {sorted(map(str, realized))} drift from the manifest-required "
            f"patterns {sorted(manifest_patterns)}"
        )
    required_modes = node.get("required_precision_modes")
    realized_modes = list((binding.get("execution") or {}).get("precision_modes") or ())
    precision = sorted(required_modes if required_modes is not None else realized_modes)
    partitioned_kinetic = (
        requirement.get("operation") == KINETIC_OPERATION
        and precision in (["full"], ["mixed"])
        and canonical_comparison_precision_modes(requirement, node.get("executable_checks")) == set(precision)
    )
    if precision != ["full", "mixed"] and not partitioned_kinetic:
        errors.append(f"{name}: canonical consumption must be required in both full and mixed, got {precision}")
    if partitioned_kinetic and sorted(realized_modes) != ["full", "mixed"]:
        errors.append(f"{name}: partitioned kinetic consumption must retain full and mixed physical execution")
    elif not partitioned_kinetic and required_modes is not None and sorted(realized_modes) != sorted(required_modes):
        errors.append(f"{name}: realized precision modes {sorted(realized_modes)} drift from required {sorted(required_modes)}")
    realized_input = (binding.get("input") or {}).get("kind")
    if realized_input != CANONICAL_INPUT_BINDING_KIND or (binding.get("input") or {}).get("committed_artifacts"):
        errors.append(
            f"{name}: realized input binding {realized_input!r} is not the canonical provider's "
            f"{CANONICAL_INPUT_BINDING_KIND!r}"
        )
    electron = requirement.get("electron_configuration") or {}
    realized_electron = binding.get("electron_configuration") or {}
    if realized_electron.get("position_source") != "seeded-random" or electron.get("seed") not in (
        realized_electron.get("seeds") or ()
    ):
        errors.append(
            f"{name}: realized electron configuration (source={realized_electron.get('position_source')!r}, "
            f"seeds={realized_electron.get('seeds')}) does not consume the manifest seed {electron.get('seed')!r}"
        )
    checks = node.get("executable_checks") or {}
    check_names = list(canonical_comparisons(requirement).values())
    observed_sites = {
        observation.get("tolerance_provenance", {}).get("comparisons", [{}])[0].get("site")
        for observation in (binding.get("comparison") or {}).get("typed_observations", ())
        if observation.get("kind") == "tolerance-comparison"
    }
    observed_sites |= {
        comparison.get("site")
        for observation in (binding.get("comparison") or {}).get("typed_observations", ())
        for comparison in (observation.get("tolerance_provenance") or {}).get("comparisons", ())
    }
    policies_by_precision: dict[str, dict[str, str]] = {}
    for check_name in check_names:
        check = checks.get(check_name)
        if not isinstance(check, dict):
            errors.append(f"{name}: executable-check-id {check_name!r} is not registered on the node")
            continue
        if check.get("site") not in observed_sites:
            errors.append(f"{name}: executable check {check_name!r} site {check.get('site')!r} is not a realized comparison")
        by_mode = {}
        for mode in precision:
            policy = ((check.get("precision") or {}).get(mode) or {}).get("policy")
            if policy not in policies:
                errors.append(f"{name}: {check_name} {mode} policy {policy!r} is not a comparison policy of verdicts.yaml")
            by_mode[mode] = policy
        policies_by_precision[check_name] = by_mode
    rows = [
        {
            "guarantee-id": guarantee_id,
            "canonical-suite-id": requirement.get("suite"),
            "canonical-pattern-id": pattern,
            "pytest-node-id": name,
            "input-binding-kind": CANONICAL_INPUT_BINDING_KIND,
            "input-binding-id": f"{requirement.get('suite')}:{pattern}",
            "electron-configuration-binding": dict(sorted(electron.items())),
            "required-precision": precision,
            "executable-check-ids": check_names,
            "comparison-policy-ids-by-precision": policies_by_precision,
        }
        for pattern in manifest_patterns
    ]
    return rows, errors


def consumer_map(
    root: Path = ROOT, documents: dict[str, dict[str, Any]] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """All canonical consumer rows and drift errors, reading verified ledgers."""
    fields = registration_fields(root)
    suite = _load(root / "gdd/ssot/fixtures/wavefunctions.yaml")["canonical_suite"]
    policies = _policies(root)
    if documents is None:
        documents = {path.name: load_ledger(path, root=root) for path in sorted((root / "gdd/ssot/guarantees").glob("*.yaml"))}
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    subscriptions = []
    for ledger_name, document in sorted(documents.items()):
        source = root / "gdd/ssot/guarantees" / ledger_name
        for guarantee, _group, node in iter_registered_nodes(document, source):
            if node.get("canonical_fixture") is None:
                continue
            subscriptions.append(node)
            node_rows, node_errors = node_consumer_rows(guarantee["id"], node, suite, policies)
            errors.extend(node_errors)
            for row in node_rows:
                if list(row) != fields:
                    errors.append(f"{node['pytest_node']}: consumer row fields {list(row)} differ from the manifest's {fields}")
            rows.extend(node_rows)
    if not rows:
        errors.append("no canonical consumer is registered")
    errors.extend(canonical_precision_coverage_errors(subscriptions, suite=suite))
    return rows, errors


def consumer_map_digest(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(sorted(rows, key=lambda row: (row["pytest-node-id"], row["canonical-pattern-id"])), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
