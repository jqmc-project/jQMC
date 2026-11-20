"""Fail-closed loading for hand-authored GDD ledgers and derived bindings.

The SSoT owns guarantee meaning and evidence-node registration.  Large case
bindings, claims, and counts are deterministic products of those registrations
and the pytest sources; they live under :mod:`gdd.derived`.  Consumers must use
this module so a missing, stale, duplicate, or extra generated row cannot be
silently accepted.
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

import contextlib
import copy
import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

from gdd.engine.guarantee_store import assemble_texts, is_module_ledger, module_sha256, module_texts


ROOT = Path(__file__).resolve().parents[2]
GUARANTEE_DIR = ROOT / "gdd/ssot/guarantees"
DERIVED_DIR = ROOT / "gdd/derived"
DERIVED_FORMAT = "gdd-derived-bindings.v1"
AUTHORED_NODE_FIELDS = frozenset(
    {
        "pytest_node",
        "executable_checks",
        "condition_check_iterations",
        "canonical_fixture",
        "required_parameter_cases",
        "required_precision_modes",
        "required_execution_profiles",
    }
)
GENERATOR_PATH = Path("scripts/materialize_gdd_case_bindings.py")
GENERATOR_DEPENDENCIES = (
    Path("gdd/engine/evidence_roles.py"),
    Path("gdd/ssot/evidence_roles.yaml"),
    Path("gdd/engine/case_binding_inventory.py"),
    Path("gdd/engine/comparison_paths.py"),
    Path("gdd/engine/inequality_receipts.py"),
    Path("gdd/engine/precision_exclusions.py"),
    Path("gdd/engine/tolerance_provenance.py"),
    Path("scripts/check_gdd_module_ledgers.py"),
    Path("gdd/ssot/fixtures/electron_configurations.yaml"),
    Path("gdd/ssot/fixtures/input_bindings.yaml"),
    Path("gdd/ssot/fixtures/sources.yaml"),
    Path("gdd/ssot/fixtures/wavefunctions.yaml"),
    Path("gdd/ssot/verdicts.yaml"),
)


def canonical_yaml_sha256(path: Path) -> str:
    """Hash the exact reviewed source bytes used to generate a binding file.

    A guarantee module ledger is its module file plus one file per guarantee
    (issue 205); the digest covers every one of those files.
    """
    if is_module_ledger(path):
        return module_sha256(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ledger_path(name_or_path: str | Path, *, root: Path = ROOT) -> Path:
    path = Path(name_or_path)
    if not path.is_absolute():
        if len(path.parts) == 1:
            path = root / "gdd/ssot/guarantees" / path
        else:
            path = root / path
    return path


def derived_path(source: Path, *, root: Path = ROOT) -> Path:
    return root / "gdd/derived" / f"{source.stem}.bindings.yaml"


def _source_texts(path: Path) -> tuple[tuple[str, str], ...]:
    if is_module_ledger(path):
        return module_texts(path)
    return ((path.as_posix(), path.read_text(encoding="utf-8")),)


def _parse_texts(path: Path, texts: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    if is_module_ledger(path):
        value = assemble_texts(texts)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: YAML document must be a mapping")
        return value
    return _parse_mapping(path, texts[0][1])


def _load_mapping(path: Path) -> dict[str, Any]:
    return _parse_texts(path, _source_texts(path))


def _parse_mapping(path: Path, text: str) -> dict[str, Any]:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    value = yaml.load(text, Loader=loader)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: YAML document must be a mapping")
    return value


# Run-scoped parse cache.  Inside ``ledger_cache()`` the parsed
# SSoT and derived mappings are reused per (path, exact file text); a changed
# file is a miss and is parsed again.  A shared mapping is only read by
# ``load_ledger`` (every returned structure is a fresh deep copy), and parse
# failures are never cached, so verdicts and errors are those of a fresh load.
_MAPPING_CACHE: dict[Path, tuple[tuple[tuple[str, str], ...], dict[str, Any]]] | None = None


@contextlib.contextmanager
def ledger_cache() -> Iterator[None]:
    """Reuse ledger parses for the duration of one checker run (nested scopes share it)."""
    global _MAPPING_CACHE
    if _MAPPING_CACHE is not None:
        yield
        return
    _MAPPING_CACHE = {}
    try:
        yield
    finally:
        _MAPPING_CACHE = None


def _read_only_mapping(path: Path) -> tuple[dict[str, Any], bool]:
    """``_load_mapping(path)``, and whether the result is shared with later calls."""
    if _MAPPING_CACHE is None:
        return _load_mapping(path), False
    text = _source_texts(path)
    key = path.resolve()
    cached = _MAPPING_CACHE.get(key)
    if cached is not None and cached[0] == text:
        return cached[1], True
    value = _parse_texts(path, text)
    _MAPPING_CACHE[key] = (text, value)
    return value, True


def evidence_section(document: dict[str, Any], source: Path) -> str:
    return "module_evidence" if source.name == "derivatives.yaml" else "guarantees"


def iter_registered_nodes(
    document: dict[str, Any], source: Path
) -> Iterable[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Yield ordinary ``(guarantee, group, node)`` registrations.

    Matrix-shaped derivative guarantees deliberately have no ordinary list and
    remain under their specialized manifest checker.
    """
    for guarantee in document.get(evidence_section(document, source), ()):
        evidence = guarantee.get("required_evidence")
        if not isinstance(evidence, list):
            continue
        for group in evidence:
            for node in group.get("nodes", ()):
                yield guarantee, group, node


def authored_document(document: dict[str, Any], source: Path) -> dict[str, Any]:
    """Return the hand-owned projection, removing every derived field."""
    cleaned = copy.deepcopy(document)
    section = evidence_section(cleaned, source)
    for guarantee in cleaned.get(section, ()):
        if not isinstance(guarantee.get("required_evidence"), list):
            continue
        guarantee.pop("claims", None)
        scope = guarantee.get("scope")
        if isinstance(scope, dict):
            scope.pop("expected_obligations", None)
            if not scope:
                guarantee.pop("scope", None)
        for group in guarantee.get("required_evidence", ()):
            for node in group.get("nodes", ()):
                node.pop("expected_cases", None)
                node.pop("case_binding", None)
    return cleaned


def _expected_index(document: dict[str, Any], source: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    guarantees: dict[str, dict[str, Any]] = {}
    nodes: set[str] = set()
    for guarantee in document.get(evidence_section(document, source), ()):
        evidence = guarantee.get("required_evidence")
        if not isinstance(evidence, list):
            continue
        guarantee_id = guarantee.get("id")
        if not isinstance(guarantee_id, str) or guarantee_id in guarantees:
            raise ValueError(f"{source}: ordinary guarantee ids must be present and unique")
        guarantees[guarantee_id] = guarantee
        for group in evidence:
            for node in group.get("nodes", ()):
                if not isinstance(node, dict):
                    raise ValueError(f"{source}: evidence-node registration must be a mapping")
                unknown = set(node) - AUTHORED_NODE_FIELDS
                if unknown:
                    raise ValueError(f"{source}: unknown authored evidence-node fields: {sorted(map(str, unknown))}")
                pytest_node = node.get("pytest_node")
                if not isinstance(pytest_node, str) or not pytest_node:
                    raise ValueError(f"{source}: every evidence node must name pytest_node")
                # A test may support several guarantees. Generated payloads
                # are keyed by (guarantee, node), not by exclusive ownership.
                nodes.add(pytest_node)
    return guarantees, nodes


def load_ledger(
    name_or_path: str | Path,
    *,
    root: Path = ROOT,
    require_derived: bool = True,
) -> dict[str, Any]:
    """Load one ledger and inject only verified generated fields.

    ``require_derived=False`` exists solely for migration tools.  Runtime and
    checker consumers use the default and therefore fail closed.
    """
    source = ledger_path(name_or_path, root=root)
    ssot, ssot_shared = _read_only_mapping(source)
    generated_path = derived_path(source, root=root)
    if not generated_path.is_file():
        if require_derived:
            raise ValueError(f"{generated_path}: required derived binding file is missing")
        return copy.deepcopy(ssot) if ssot_shared else ssot
    derived, _derived_shared = _read_only_mapping(generated_path)
    header = derived.get("document")
    expected_source = source.relative_to(root).as_posix()
    if not isinstance(header, dict) or header.get("format") != DERIVED_FORMAT:
        raise ValueError(f"{generated_path}: invalid generated format")
    if header.get("source") != expected_source:
        raise ValueError(f"{generated_path}: source path does not match {expected_source}")
    digest = canonical_yaml_sha256(source)
    if header.get("source_sha256") != digest:
        raise ValueError(f"{generated_path}: stale source_sha256; run scripts/gdd_sync.py")
    generator = root / GENERATOR_PATH
    if header.get("generator_sha256") != hashlib.sha256(generator.read_bytes()).hexdigest():
        raise ValueError(f"{generated_path}: stale generator_sha256; run scripts/gdd_sync.py")
    dependency_sha256 = header.get("dependency_sha256")
    expected_dependencies = {
        path.as_posix(): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in GENERATOR_DEPENDENCIES
    }
    if dependency_sha256 != expected_dependencies:
        raise ValueError(f"{generated_path}: stale dependency_sha256; run scripts/gdd_sync.py")

    expected_guarantees, expected_nodes = _expected_index(ssot, source)
    generated_guarantees = derived.get("guarantees")
    if not isinstance(generated_guarantees, dict):
        raise ValueError(f"{generated_path}: guarantees must be a mapping")
    payload = json.dumps(generated_guarantees, sort_keys=True, separators=(",", ":")).encode()
    if header.get("payload_sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError(f"{generated_path}: payload_sha256 mismatch; generated file was edited")
    if set(generated_guarantees) != set(expected_guarantees):
        missing = sorted(set(expected_guarantees) - set(generated_guarantees))
        extra = sorted(set(generated_guarantees) - set(expected_guarantees))
        raise ValueError(f"{generated_path}: guarantee set mismatch; missing={missing}, extra={extra}")

    realized_nodes: set[str] = set()
    merged = copy.deepcopy(ssot)
    merged_by_id = {
        row["id"]: row
        for row in merged.get(evidence_section(merged, source), ())
        if isinstance(row.get("required_evidence"), list)
    }
    for guarantee_id, generated in generated_guarantees.items():
        if not isinstance(generated, dict):
            raise ValueError(f"{generated_path}: {guarantee_id} binding must be a mapping")
        guarantee = merged_by_id[guarantee_id]
        guarantee["claims"] = copy.deepcopy(generated.get("claims", []))
        guarantee.setdefault("scope", {})["expected_obligations"] = generated.get("expected_obligations")
        by_node = generated.get("nodes")
        if not isinstance(by_node, dict):
            raise ValueError(f"{generated_path}: {guarantee_id}.nodes must be a mapping")
        derived_total = sum(payload.get("expected_cases", 0) for payload in by_node.values() if isinstance(payload, dict))
        if generated.get("expected_obligations") != derived_total:
            raise ValueError(f"{generated_path}: obligation total mismatch for {guarantee_id}")
        claims = generated.get("claims")
        if not isinstance(claims, list) or {claim.get("evidence_node") for claim in claims} != set(by_node):
            raise ValueError(f"{generated_path}: claim/node set mismatch for {guarantee_id}")
        for group in guarantee["required_evidence"]:
            for node in group.get("nodes", ()):
                pytest_node = node["pytest_node"]
                payload = by_node.get(pytest_node)
                if not isinstance(payload, dict) or set(payload) != {"expected_cases", "case_binding"}:
                    raise ValueError(f"{generated_path}: malformed or missing node {pytest_node}")
                count = payload["expected_cases"]
                binding = payload["case_binding"]
                if not isinstance(count, int) or count < 1 or not isinstance(binding, dict):
                    raise ValueError(f"{generated_path}: invalid derived payload for {pytest_node}")
                if binding.get("fixture", {}).get("expected_cases") != count:
                    raise ValueError(f"{generated_path}: count/binding mismatch for {pytest_node}")
                node["expected_cases"] = count
                node["case_binding"] = copy.deepcopy(binding)
                realized_nodes.add(pytest_node)
        if set(by_node) != {node["pytest_node"] for group in guarantee["required_evidence"] for node in group.get("nodes", ())}:
            raise ValueError(f"{generated_path}: extra or missing nodes for {guarantee_id}")
    if realized_nodes != expected_nodes:
        raise ValueError(f"{generated_path}: global evidence-node set mismatch")
    return merged
