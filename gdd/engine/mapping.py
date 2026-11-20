"""The many-to-many test/guarantee relation, independent of generated bindings.

Collection adapters supply in-scope, unparameterized test identifiers. Ledger
adapters supply every guarantee and its explicitly registered identifiers,
including guarantees with no evidence. Parameter/fixture coverage and execution
outcomes are separate checks: this relation alone cannot establish either.
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
import ast
import json
from pathlib import Path
import re

from gdd.engine.guarantee_store import load_module


def _identifiers(values: Iterable[str], *, context: str) -> set[str]:
    """Reject malformed input instead of accidentally iterating a node string."""
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{context}: expected an iterable of test identifiers")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{context}: test identifiers must be nonempty strings")
        result.add(value)
    return result


def mapping_errors(guarantees: Mapping[str, Iterable[str]], collected_nodes: Iterable[str]) -> list[str]:
    """Return deterministic diagnostics for missing edges in either direction.

    Test identifiers are opaque: callers normalize collection and registration
    through the same adapter, not by stripping parameter suffixes here. Several
    guarantees may reference the same test. Repeated collection/registration
    entries do not create an additional obligation or a bookkeeping failure.

    The caller owns scope selection; no exemptions, count pins, approval tables,
    or derived files are consulted. Empty input on both sides is valid for an
    empty scope; the repository adapter must supply the complete intended scope.
    """
    collected = _identifiers(collected_nodes, context="collection")
    registered: set[str] = set()
    errors = []
    if any(not isinstance(key, str) or not key.strip() for key in guarantees):
        raise ValueError("guarantee identifiers must be nonempty strings")
    for guarantee_id in sorted(guarantees):
        nodes = _identifiers(guarantees[guarantee_id], context=guarantee_id)
        if not nodes:
            errors.append(f"{guarantee_id}: no test evidence registered")
        for node in sorted(nodes - collected):
            errors.append(f"{guarantee_id}: registered test not collected: {node}")
        registered.update(nodes)
    for node in sorted(collected - registered):
        errors.append(f"{node}: no guarantee registered")
    return errors


def definition_node(node: str) -> str:
    """Normalize actual pytest parameter items to their registered definition."""
    if not isinstance(node, str) or "::" not in node:
        raise ValueError(f"not a pytest function node: {node!r}")
    return node.split("[", 1)[0]


def qualified_test_source_definitions(source: str, path: str) -> set[str]:
    """Inventory module tests and every class-owned test source definition.

    This source inventory is intentionally broader than pytest's collection
    approximation below: a plain mixin can define a test that a collected
    subclass inherits. Actual collection remains a separate runtime contract.
    """
    result: set[str] = set()

    def walk(body, parents):
        for statement in body:
            if isinstance(statement, ast.ClassDef):
                walk(statement.body, [*parents, statement.name])
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name.startswith("test_"):
                node = "::".join([path, *parents, statement.name])
                if node in result:
                    raise ValueError(f"duplicate test definition: {node}")
                result.add(node)

    walk(ast.parse(source, filename=path).body, [])
    return result


def qualified_definitions(source: str, path: str) -> set[str]:
    """Fast source-side discovery; actual pytest collection is checked separately."""
    tree = ast.parse(source, filename=path)
    result: set[str] = set()

    unittest_modules: set[str] = set()
    testcase_names: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            unittest_modules.update(alias.asname or alias.name for alias in statement.names if alias.name == "unittest")
        elif isinstance(statement, ast.ImportFrom) and statement.module == "unittest":
            testcase_names.update(alias.asname or alias.name for alias in statement.names if alias.name == "TestCase")

    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    testcase_classes: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in classes:
            is_testcase = any(
                (isinstance(base, ast.Name) and base.id in testcase_names | testcase_classes)
                or (
                    isinstance(base, ast.Attribute)
                    and base.attr == "TestCase"
                    and isinstance(base.value, ast.Name)
                    and base.value.id in unittest_modules
                )
                for base in node.bases
            )
            if is_testcase and node.name not in testcase_classes:
                testcase_classes.add(node.name)
                changed = True

    def walk(body, parents):
        for statement in body:
            if isinstance(statement, ast.ClassDef) and (
                statement.name.startswith("Test") or statement.name in testcase_classes
            ):
                walk(statement.body, [*parents, statement.name])
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name.startswith("test_"):
                node = "::".join([path, *parents, statement.name])
                if node in result:
                    raise ValueError(f"duplicate test definition: {node}")
                result.add(node)

    walk(tree.body, [])
    return result


def in_scope(node: str, registered: Iterable[str] = ()) -> bool:
    """Select jQMC tests plus explicit evidence, without recursive framework ownership.

    Infrastructure self-tests are exercised independently; adding a test of
    GDD itself does not require another GDD registration. Explicit scientific
    evidence remains in scope even when hosted in an infrastructure file.
    """
    path = definition_node(node).split("::", 1)[0]
    if definition_node(node) in registered:
        return True
    return path.startswith("tests/") and not (
        path.startswith(("tests/test_jqmc_workflow/", "tests/test_gdd_simplification/", "tests/test_gdd/"))
        or Path(path).name.startswith("test_gdd_")
    )


def static_collected_nodes(root: Path, *, registered: Iterable[str] = ()) -> set[str]:
    """Find test definitions without importing JAX, executing tests, or reading pins."""
    registered = set(registered)
    nodes: set[str] = set()
    for source in sorted((root / "tests").rglob("test_*.py")):
        relative = source.relative_to(root).as_posix()
        nodes.update(
            node for node in qualified_definitions(source.read_text(encoding="utf-8"), relative) if in_scope(node, registered)
        )
    return nodes


def authored_mapping(root: Path) -> dict[str, set[str]]:
    """Load explicit many-to-many edges from SSoT, never from derived bindings.

    Both ordinary evidence groups and derivative matrix dictionaries may carry
    ``nodes``. The latter still retain their existing family/fixture/operation
    contract; this adapter only adds a function-level edge, never replaces the
    matrix's finer-grained completion check. FD sentinel registrations are read
    from the exact fixture set named by that guarantee.
    """
    mapping: dict[str, set[str]] = {}
    for path in sorted((root / "gdd/ssot/guarantees").glob("*.yaml")):
        document = load_module(path)
        fixtures = {fixture["id"]: fixture for fixture in document.get("fixture_sets", [])}
        for section in ("guarantees", "module_evidence"):
            guarantees = document.get(section, [])
            if not isinstance(guarantees, list):
                raise ValueError(f"{path}: {section} must be a list")
            for guarantee in guarantees:
                identifier = guarantee.get("id")
                if not isinstance(identifier, str) or not identifier.strip():
                    raise ValueError(f"{path}: guarantee has no identifier")
                if identifier in mapping:
                    raise ValueError(f"duplicate guarantee identifier: {identifier}")
                evidence = guarantee.get("required_evidence", [])
                groups = [evidence] if isinstance(evidence, dict) else evidence
                if not isinstance(groups, list) or any(not isinstance(group, dict) for group in groups):
                    raise ValueError(f"{identifier}: malformed required_evidence")
                rows = [row for group in groups for row in group.get("nodes", [])]
                fixture_id = guarantee.get("scope", {}).get("fixture_set")
                if fixture_id in fixtures:
                    rows += fixtures[fixture_id].get("reviewed_evidence", {}).get("nodes", [])
                for group in groups:
                    direct = group.get("pytest")
                    if direct is not None:
                        rows += [{"pytest_node": f"{direct['path']}::{symbol}"} for symbol in direct["symbols"]]
                if any(not isinstance(row, dict) or not isinstance(row.get("pytest_node"), str) for row in rows):
                    raise ValueError(f"{identifier}: every evidence row needs pytest_node")
                mapping[identifier] = {definition_node(row["pytest_node"]) for row in rows}
                native = guarantee.get("native_evidence", [])
                if not isinstance(native, list) or any(
                    not isinstance(node, str) or not node.startswith("jqmc_rs::") for node in native
                ):
                    raise ValueError(f"{identifier}: native_evidence must name crate-qualified Rust tests")
                mapping[identifier].update(native)
    if not mapping:
        raise ValueError("no authored guarantees found")
    return mapping


def native_test_nodes(root: Path) -> set[str]:
    """Discover native unit tests with their enclosing Rust module identities.

    This uses the comment/string-masking parser, not a pinned inventory. It discovers source definitions; a Cargo build/test is
    still needed to prove those tests compile and pass.
    """
    from gdd.engine.rust_sources import RUST_ROOT, _attributes_before, _brace_pairs, functions, rust_files

    nodes: set[str] = set()
    for relative, original, masked in rust_files(root):
        pairs = _brace_pairs(masked)
        modules = [
            (match.start(), match.end() - 1, pairs[match.end() - 1], match.group(1))
            for match in re.finditer(r"\bmod\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", masked)
            if match.end() - 1 in pairs
        ]
        path_parts = list(Path(relative).relative_to(RUST_ROOT).with_suffix("").parts)
        if path_parts[-1] in ("lib", "mod"):
            path_parts.pop()
        for function in functions(masked):
            if "test" not in _attributes_before(original, masked, function.start):
                continue
            parents = [name for _start, begin, end, name in modules if begin < function.start < end]
            identity = "::".join(["jqmc_rs", *path_parts, *parents, function.name])
            if identity in nodes:
                raise ValueError(f"ambiguous native test identity: {identity}")
            nodes.add(identity)
    return nodes


def cargo_list_nodes(path: Path) -> set[str]:
    """Read ``cargo test -- --list`` output, independently of source discovery."""
    nodes = {
        "jqmc_rs::" + line.removesuffix(": test")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.endswith(": test")
    }
    if not nodes:
        raise ValueError(f"{path}: Cargo list contains no native tests")
    return nodes


def collected_json_nodes(paths: Iterable[Path], *, registered: Iterable[str] = ()) -> set[str]:
    """Read actual pytest collection reports, unioning full and mixed lanes."""
    registered = set(registered)
    nodes: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or any(not isinstance(node, str) for node in payload):
            raise ValueError(f"{path}: expected a JSON array of pytest node IDs")
        nodes.update(definition_node(node) for node in payload if in_scope(node, registered))
    return nodes


def collection_manifest_nodes(directory: Path, *, registered: Iterable[str] = ()) -> set[str]:
    """Read real pytest collection manifests before precision/lane deselection.

    Execution completion checks each report's run identity and selected cases;
    this adapter checks the function relation, including statically skipped
    definitions. A failed or absent collection is never an empty success.
    """
    registered = set(registered)
    paths = sorted(directory.rglob("gdd-collection-*.json"))
    if not paths:
        raise ValueError(f"{directory}: no actual collection manifests found")
    nodes: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != "gdd-pytest-execution.v1":
            raise ValueError(f"{path}: invalid collection manifest schema")
        if payload.get("kind") != "collection" or payload.get("exitstatus") != 0:
            raise ValueError(f"{path}: collection did not finish successfully")
        before = payload.get("before_selection")
        if not isinstance(before, list) or any(not isinstance(node, str) for node in before):
            raise ValueError(f"{path}: invalid before_selection nodes")
        nodes.update(definition_node(node) for node in before if in_scope(node, registered))
    return nodes


def pytest_addoption(parser):
    """Optional pytest plugin: emit actual collection, without executing tests."""
    parser.addoption("--gdd-collection-output", default=None, help="Write collected pytest node IDs as JSON")


def pytest_collection_finish(session):
    destination = session.config.getoption("--gdd-collection-output", default=None)
    if destination:
        Path(destination).write_text(json.dumps(sorted(item.nodeid for item in session.items)) + "\n", encoding="utf-8")
