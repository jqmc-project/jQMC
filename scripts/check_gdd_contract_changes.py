#!/usr/bin/env python3
"""Reject unreviewed weakening relative to a trusted Git baseline.

Trust model: both the baseline DATA and the
deciding POLICY come from the trusted base ref (origin/main in CI):

* When the base's copy of this script declares TRUSTED_POLICY_PROTOCOL, the
  launcher extracts the base's ``scripts/`` and ``gdd/engine/`` into a
  temporary directory and runs THAT checker (``python -I``, no candidate
  path on sys.path) against the candidate tree. The candidate cannot add a
  permitted migration or relax a rule for its own review; any difference in
  GUARD_FILES is itself reported as a guard change requiring separate review.
* Bootstrap is limited to the single protocol-1 to protocol-2 transition.
  Every later mismatch runs the trusted base policy; incrementing the
  candidate protocol cannot permit another candidate-policy migration.
* Locally without origin/main the base falls back to HEAD; the result is
  printed as UNTRUSTED.

The launcher itself is candidate code; making CI run the trusted copy even
if the candidate edits this file or its invocation needs a repository-level
required check.
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

import argparse
import difflib
import io
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import yaml


POLICY_ROOT = Path(__file__).resolve().parents[1]
ROOT = POLICY_ROOT
sys.path.insert(0, str(POLICY_ROOT))

from gdd.engine import contract_guard as _contract_guard_module
from gdd.engine import hook_coverage as _hook_coverage_module
from gdd.engine import skip_protection as _skip_protection_module
from gdd.engine import rust_references as _rust_references_module
from gdd.engine import tolerance_manifest as _tolerance_manifest_module
from gdd.engine.rust_references import export_status, reference_sets, referenced_names
from gdd.engine.contract_guard import (
    derived_obligation_errors,
    ledger_monotonicity_errors,
    monotonic_structure_errors,
    test_definition_inclusion_errors,
)
from gdd.engine.tolerance_manifest import settings_contract_errors
from gdd.engine.skip_protection import (
    precision_removal_errors,
    skip_change_errors,
)
from gdd.engine.guarantee_store import (
    GuaranteeLayoutError,
    assemble_document,
    load_module,
    module_directory_relative,
)
from gdd.engine.hook_coverage import path_filter_relaxation_errors


# A base copy declaring this protocol is executed as the deciding policy.
TRUSTED_POLICY_PROTOCOL = 2
# Reviewed bases with this capability need their complete snapshot, not ancestry.
TRUSTED_BASE_SNAPSHOTS = True
_PROTOCOL_PATTERN = re.compile(r"^TRUSTED_POLICY_PROTOCOL = (\d+)$", re.MULTILINE)
# Trees extracted from the base for the trusted run (the whole import closure).
POLICY_TREES = ("gdd/engine", "scripts")
# Files whose content decides this guard's verdict. A candidate
# difference is never silently used: the base copy runs and the difference is
# reported as a guard change that needs its own review before promotion.
GUARD_FILES = (
    "scripts/check_gdd_contract_changes.py",
    "gdd/engine/contract_guard.py",
    "gdd/engine/skip_protection.py",
    "gdd/engine/hook_coverage.py",
    "gdd/engine/rust_references.py",
    "gdd/engine/tolerance_manifest.py",
)
DECISIVE_MODULES = (
    _contract_guard_module,
    _skip_protection_module,
    _hook_coverage_module,
    _rust_references_module,
    _tolerance_manifest_module,
)
PRE_COMMIT_CONFIG = ".pre-commit-config.yaml"
RUST_INVENTORY = "gdd/ssot/inventories/rust_inventory.yaml"
# Standing rule: a rust_inventory export status may change without separate
# review only when a test reference that the PR itself adds causes it. Every
# other inventory change stays protected.
RUST_TEST_REFERENCE_TRANSITIONS = frozenset({("production-only", "tested"), ("unused", "test-only")})

# A protected or weakening change stays red; it is reviewed from the reported
# changes and accepted only by merging the PR.

LEDGER_PREFIX = "gdd/ssot/guarantees/"
PROTECTED_REGISTRIES = (
    "gdd/ssot/evidence_roles.yaml",
    "gdd/ssot/fixtures/electron_configurations.yaml",
    "gdd/ssot/fixtures/input_bindings.yaml",
    "gdd/ssot/fixtures/sources.yaml",
    "gdd/ssot/fixtures/wavefunctions.yaml",
    "gdd/ssot/verdicts.yaml",
    "gdd/ssot/symbol_ownership.yaml",
)
TEST_REGISTRY = "gdd/ssot/test_registry.yaml"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)


def require_base_snapshot(base_ref: str) -> str:
    """Pin a complete baseline tree; shallow ancestry does not affect comparisons."""
    resolved = _git("rev-parse", "--verify", f"{base_ref}^{{commit}}")
    if resolved.returncode:
        raise ValueError(f"trusted base ref does not resolve: {base_ref}")
    commit = resolved.stdout.strip()
    listed = _git("ls-tree", "-r", "--full-tree", commit)
    if listed.returncode:
        raise ValueError(f"cannot read trusted base tree {commit}: {listed.stderr.strip()}")
    # Recursive traversal reads every subtree. Gitlinks refer to other repositories
    # and are not baseline files; all file blobs, including symlinks, must exist.
    blobs = sorted({row.split()[2] for row in listed.stdout.splitlines() if row.split()[1] == "blob"})
    if blobs:
        checked = subprocess.run(
            ["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
            cwd=ROOT,
            input="\n".join(blobs) + "\n",
            text=True,
            capture_output=True,
            check=False,
        )
        # A missing object is reported on stdout, often with exit status zero.
        if checked.returncode or checked.stdout.splitlines() != [f"{oid} blob" for oid in blobs]:
            raise ValueError(f"cannot read complete trusted base snapshot {commit}: missing or unreadable file blobs")
    return commit


def _default_base() -> str:
    configured = os.environ.get("GDD_TRUSTED_BASE_REF")
    if configured:
        return configured
    probe = _git("rev-parse", "--verify", "origin/main^{commit}")
    if probe.returncode == 0:
        return "origin/main"
    if os.environ.get("CI"):
        raise ValueError("origin/main is unavailable in CI; fetch the trusted base or set GDD_TRUSTED_BASE_REF explicitly")
    return "HEAD"


def _yaml_at(ref: str, relative: str) -> dict[str, Any] | None:
    shown = _git("show", f"{ref}:{relative}")
    if shown.returncode != 0:
        return None
    value = yaml.load(shown.stdout, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    if not isinstance(value, dict):
        raise ValueError(f"{ref}:{relative} must be a YAML mapping")
    return value


def _ledger_at(ref: str, relative: str, files: list[str]) -> dict[str, Any] | None:
    """The assembled module ledger at ``ref`` (monolithic or split), read through ``_yaml_at``."""
    module = _yaml_at(ref, relative)
    if module is None:
        return None
    # Before issue 223, split markers carried an ordering snapshot.  It has no
    # effect on guarantee identity or meaning, and current files derive order
    # from the guarantee IDs themselves.
    for section in ("guarantees", "module_evidence"):
        marker = module.get(section)
        if isinstance(marker, dict):
            marker.pop("legacy_order", None)
    prefix = module_directory_relative(relative) + "/"
    split_files = [path for path in files if path.startswith(prefix)]

    def load_row(path: str) -> Any:
        row = _yaml_at(ref, path)
        if row is None:
            raise GuaranteeLayoutError(f"cannot read {ref}:{path}")
        return row

    return assemble_document(relative, module, load_row, split_files)


def fully_registered_test_files(root: Path) -> frozenset[str]:
    """Test files whose every collected test function is registered guarantee evidence.

    Moving such a file out of ``exempt_files`` subjects it to the orphan gate and
    to evidence-removal protection, so it is a tightening, not a new escape hatch
    (issue 180). Files without test functions never qualify.
    """
    from scripts.check_gdd_orphan_tests import qualified_test_functions

    # Complete static identities (file::Class::function); a pytest parameter id
    # ``[...]`` names a case of the same definition. Class names are kept so that
    # TestA::test_same never stands in for TestB::test_same.
    registered: set[str] = set()

    def harvest(value: Any) -> None:
        if isinstance(value, dict):
            node = value.get("pytest_node")
            if isinstance(node, str):
                registered.add(node.split("[", 1)[0])
            for child in value.values():
                harvest(child)
        elif isinstance(value, list):
            for child in value:
                harvest(child)

    # Issue 205: monolithic or split module ledgers, through the shared loader.
    for path in sorted((root / "gdd/ssot/guarantees").glob("*.yaml")):
        harvest(load_module(path))
    governed = set()
    for relative in sorted({node.split("::", 1)[0] for node in registered}):
        path = root / relative
        if not path.is_file():
            continue
        try:
            names = qualified_test_functions(path.read_text(encoding="utf-8"), relative)
        except (OSError, SyntaxError, ValueError):
            continue
        if names and all(name in registered for name in names):
            governed.add(relative)
    return frozenset(governed)


def _test_registry_errors(
    base: dict[str, Any],
    current: dict[str, Any],
    *,
    governed_files: frozenset[str] = frozenset(),
) -> list[str]:
    """Protect escape hatches while allowing reviewed derived root-pin growth.

    ``governed_files`` (issue 180) are files whose every test function is
    registered evidence in the candidate; only those may leave ``exempt_files``.
    """
    errors: list[str] = []
    for key in ("document", "enforcement_map"):
        if base.get(key) != current.get(key):
            errors.append(f"{TEST_REGISTRY}: changed protected {key}")

    before_roots = {row.get("root"): row for row in base.get("exempt_roots", ())}
    after_roots = {row.get("root"): row for row in current.get("exempt_roots", ())}
    added_roots = set(after_roots) - set(before_roots)
    if set(before_roots) - set(after_roots) or added_roots:
        errors.append(f"{TEST_REGISTRY}: exempt-root set changed")
    for root, row in after_roots.items():
        if not str(row.get("reason", "")).strip():
            errors.append(f"{TEST_REGISTRY}: exempt-root reason is blank for {root}")

    before_files = {row.get("path"): row.get("reason") for row in base.get("exempt_files", ())}
    after_files = {row.get("path"): row.get("reason") for row in current.get("exempt_files", ())}
    for path in before_files:
        if path not in after_files and (path in governed_files or not (ROOT / path).is_file()):
            continue  # Registered evidence is governed; deleting a framework-only file creates no orphan.
        if path not in after_files:
            errors.append(f"{TEST_REGISTRY}: removed exempt file {path}")
        elif not str(after_files[path]).strip():
            errors.append(f"{TEST_REGISTRY}: exempt-file reason is blank for {path}")
    for path in set(after_files) - set(before_files):
        errors.append(f"{TEST_REGISTRY}: added unreviewed exempt file {path}")
    before_pending = {
        (row.get("path"), name) for row in base.get("pending_unregistered", ()) for name in row.get("functions", ())
    }
    after_pending = {
        (row.get("path"), name) for row in current.get("pending_unregistered", ()) for name in row.get("functions", ())
    }
    additions = sorted(after_pending - before_pending)
    if additions:
        errors.append(f"{TEST_REGISTRY}: added pending/unregistered tests {additions}")
    return errors


def _semantic_registry(document: dict[str, Any], relative: str) -> dict[str, Any]:
    """Drop retired snapshots while retaining the registries' live facts."""
    import copy

    value = copy.deepcopy(document)
    metadata = value.get("document")
    if isinstance(metadata, dict):
        metadata.pop("authority", None)
    value.pop("known_cutover_gaps", None)
    value.pop("known_migration_gaps", None)
    enforcement = value.get("enforcement_map")
    if isinstance(enforcement, dict):
        enforcement.pop("known_cutover_gaps", None)
        enforcement.pop("known_migration_gaps", None)
    if relative.endswith(("fixtures/wavefunctions.yaml", "fixtures/electron_configurations.yaml")):
        value.pop("guarantee_profiles", None)
        if isinstance(enforcement, dict):
            enforcement.pop("guarantee_profiles", None)
    elif relative.endswith("fixtures/input_bindings.yaml"):
        value.pop("guarantee_profiles", None)
        if isinstance(enforcement, dict):
            enforcement.pop("guarantee_profiles", None)
            enforcement.pop("source_policy", None)
            enforcement.pop("runtime_receipt", None)
    elif relative.endswith("fixtures/sources.yaml"):
        value.pop("source_sets", None)
        if isinstance(enforcement, dict):
            enforcement.pop("source_sets", None)
    elif relative.endswith("verdicts.yaml"):
        registry = value.get("tolerance_registry")
        if isinstance(registry, dict):
            for key in ("decimal_value_manifest", "ulp_value_manifest", "value_manifest", "table_manifests"):
                registry.pop(key, None)
    elif relative.endswith("symbol_ownership.yaml"):
        ownership = value.get("ownership")
        if isinstance(ownership, dict):
            for entry in ownership.values():
                if isinstance(entry, dict):
                    entry.pop("public_symbols", None)
                    entry.pop("private_symbols", None)
    return value


GOVERNED_LEDGER_SCHEMA = "gdd/ssot/schemas/guarantee-ledger-governed.v1.yaml"


def _global_enforcement() -> dict[str, Any]:
    """The candidate's global section -> checker table of the governed guarantee ledgers.

    The schema check requires each of its checkers to run in the GDD preflight.
    """
    schema = yaml.safe_load((ROOT / GOVERNED_LEDGER_SCHEMA).read_text(encoding="utf-8"))
    table = schema.get("global_enforcement", {})
    return table if isinstance(table, dict) else {}


def _global_enforcement_errors(base_ref: str, global_rows: dict[str, Any]) -> list[str]:
    """The table is protected like the per-ledger rows it replaces.

    Rows moved from the ledgers into the table compare clean through
    ``_without_global_enforcement``; once the base has the table, any change of it is reported.
    """
    base = _yaml_at(base_ref, GOVERNED_LEDGER_SCHEMA) or {}
    if "global_enforcement" in base and base["global_enforcement"] != global_rows:
        return [f"{GOVERNED_LEDGER_SCHEMA}: changed protected global_enforcement"]
    return []


def _without_global_enforcement(document: dict[str, Any], global_rows: dict[str, Any]) -> dict[str, Any]:
    """A ledger without the enforcement_map rows that only repeat a global checker."""
    import copy

    value = copy.deepcopy(document)
    enforcement = value.get("enforcement_map")
    if isinstance(enforcement, dict):
        for section, checker in global_rows.items():
            if enforcement.get(section) == checker:
                del enforcement[section]
        if not enforcement:
            del value["enforcement_map"]
    return value


def _semantic_validation_errors(base_ref: str) -> tuple[list[str], dict[str, int]]:
    """Compare only live guarantees and semantic registries with the trusted base."""
    if _git("rev-parse", "--verify", f"{base_ref}^{{commit}}").returncode != 0:
        return [f"trusted base ref does not resolve: {base_ref}"], {}
    tree = _git("ls-tree", "-r", "--name-only", base_ref, "gdd/ssot/guarantees")
    if tree.returncode != 0:
        return [tree.stderr.strip() or f"cannot inspect {base_ref}"], {}
    base_files = tree.stdout.splitlines()
    base_ledgers = {line for line in base_files if line.endswith(".yaml") and line.count("/") == 3}
    errors: list[str] = []
    added_nodes = 0
    added_guarantees = 0
    global_rows = _global_enforcement()
    errors.extend(_global_enforcement_errors(base_ref, global_rows))

    for relative in sorted(base_ledgers):
        current_path = ROOT / relative
        if not current_path.is_file():
            errors.append(f"removed ledger {relative}")
            continue
        try:
            base = _ledger_at(base_ref, relative, base_files)
            current = load_module(current_path)
        except GuaranteeLayoutError as exc:
            errors.append(f"cannot compare {relative}: {exc}")
            continue
        if base is None:
            errors.append(f"cannot compare {relative}")
            continue
        base, current = (_without_global_enforcement(document, global_rows) for document in (base, current))
        errors.extend(ledger_monotonicity_errors(base, current, current_path))

        derived_relative = f"gdd/derived/{current_path.stem}.bindings.yaml"
        before_binding = _yaml_at(base_ref, derived_relative)
        generated = ROOT / derived_relative
        after_binding = None
        if generated.is_file():
            try:
                after_binding = yaml.safe_load(generated.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                errors.append(f"cannot read provenance obligations in {derived_relative}: {exc}")
        if before_binding is not None:
            if after_binding is None:
                errors.append(f"removed trusted precision inventory {derived_relative}")
            else:
                try:
                    errors.extend(precision_removal_errors(before_binding, after_binding, derived_relative))
                    errors.extend(derived_obligation_errors(before_binding, after_binding, derived_relative))
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"cannot compare precision obligations in {derived_relative}: {exc}")

        section = "module_evidence" if current_path.name == "derivatives.yaml" else "guarantees"
        base_rows = {row["id"]: row for row in base.get(section, ()) if isinstance(row.get("required_evidence"), list)}
        current_rows = {row["id"]: row for row in current.get(section, ()) if isinstance(row.get("required_evidence"), list)}
        added_guarantees += len(set(current_rows) - set(base_rows))
        for guarantee_id in set(base_rows) & set(current_rows):
            before = {
                node["pytest_node"] for group in base_rows[guarantee_id]["required_evidence"] for node in group.get("nodes", ())
            }
            after = {
                node["pytest_node"]
                for group in current_rows[guarantee_id]["required_evidence"]
                for node in group.get("nodes", ())
            }
            added_nodes += len(after - before)

    for relative in PROTECTED_REGISTRIES:
        base = _yaml_at(base_ref, relative)
        if base is None:
            continue
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"removed protected registry {relative}")
            continue
        current = yaml.safe_load(path.read_text(encoding="utf-8"))
        errors.extend(
            monotonic_structure_errors(
                _semantic_registry(base, relative),
                _semantic_registry(current, relative),
                relative,
            )
        )

    base_registry = _yaml_at(base_ref, TEST_REGISTRY)
    if base_registry is not None:
        registry_path = ROOT / TEST_REGISTRY
        if not registry_path.is_file():
            errors.append(f"removed protected registry {TEST_REGISTRY}")
        else:
            errors.extend(
                _test_registry_errors(
                    base_registry,
                    yaml.safe_load(registry_path.read_text(encoding="utf-8")),
                    governed_files=fully_registered_test_files(ROOT),
                )
            )
    changed = _git("diff", "--name-only", base_ref, "--", "tests")
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", "tests")
    if changed.returncode or untracked.returncode:
        errors.append("cannot inventory test-source changes for skip protection")
    else:
        paths = set(changed.stdout.splitlines()) | set(untracked.stdout.splitlines())
        for relative in sorted(path for path in paths if path.endswith(".py")):
            path = ROOT / relative
            if not path.is_file():
                continue
            try:
                errors.extend(_test_source_skip_errors(base_ref, relative, path.read_text()))
            except (OSError, SyntaxError, ValueError, RecursionError) as exc:
                errors.append(f"cannot check skip capabilities in {relative}: {exc}")
    return errors, {"added_guarantees": added_guarantees, "added_nodes": added_nodes}


# Public name retained for the launcher and focused self-tests.
validation_errors = _semantic_validation_errors


def _test_source_skip_errors(base_ref: str, relative: str, current: str) -> list[str]:
    """Distinguish a proven absent baseline file from an unreadable existing one."""
    before = _git("show", f"{base_ref}:{relative}")
    baseline_exists = before.returncode == 0
    if not baseline_exists:
        listed = _git("ls-tree", "--name-only", base_ref, "--", relative)
        if listed.returncode or listed.stdout.strip():
            return [f"cannot establish absent baseline test source for skip protection: {relative}"]
    return skip_change_errors(before.stdout if baseline_exists else "", current, relative, baseline_exists=baseline_exists)


def _blob_at(ref: str, relative: str) -> str | None:
    shown = _git("show", f"{ref}:{relative}")
    return shown.stdout if shown.returncode == 0 else None


def _current_text(relative: str) -> str | None:
    path = ROOT / relative
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _gdd_hooks(document: Any) -> dict[str, Any]:
    hooks: dict[str, Any] = {}
    for repo in (document or {}).get("repos") or ():
        for hook in repo.get("hooks") or ():
            if isinstance(hook, dict) and str(hook.get("id", "")).startswith("gdd-"):
                hooks[hook["id"]] = hook
    return hooks


def guard_change_errors(base_ref: str, *, trusted_policy: bool) -> list[str]:
    """Candidate differences in the deciding guard files (never silently used)."""
    applied = (
        "the trusted-base policy was applied, not the candidate's" if trusted_policy else "BOOTSTRAP: the candidate's copy ran"
    )
    errors = []
    for relative in GUARD_FILES:
        before = _blob_at(base_ref, relative)
        after = _current_text(relative)
        if before != after:
            state = "is new relative to" if before is None else "was removed from" if after is None else "differs from"
            errors.append(f"guard change requires separate review: {relative} {state} trusted base {base_ref}; {applied}")
    return errors


def trusted_source_errors(base_ref: str) -> list[str]:
    """Protect enforcement wiring, gate files and jqmc/_setting.py against the base.

    Rules: every base ``gdd-*`` pre-commit hook stays identical (new hooks are
    free), except for one path-filter rule: a base hook with
    ``always_run: true`` may become ``always_run: false`` plus a ``files``
    regex that selects every protected GDD input prefix with no
    ``exclude``/``types`` narrowing (``gdd.engine.hook_coverage``). Every other field (id, name,
    entry, language, pass_filenames, stages, args, dependencies) must stay
    identical and removing a hook stays rejected. Separately, the deciding
    policy tree declares exact dependency-only additions in
    ``hook_dependencies.yaml``; those preserve every other hook field after
    independently checking any historical path-filter transition. The path-filter rule is a rewrite of the
    rule, not an exception or migration; no base
    data is normalized. No base file under scripts/ or gdd/engine/ is deleted or renamed
    away. The central settings (jqmc/_setting.py, jqmc/_precision.py) are reviewed files
    (gdd.engine.tolerance_manifest, issue 227): their data bindings are evaluated
    statically and may only tighten (a loosening, change or removal of a reviewed value
    is one reported line), and any change to their code is one reported
    ``settings code changed: <shape id>`` line that needs separate review.

    GDD reads no CI workflow; the static GDD gates stay protected as commit
    hooks (the ``gdd-*`` rule above).
    """
    errors: list[str] = []
    base_hooks_text = _blob_at(base_ref, PRE_COMMIT_CONFIG)
    if base_hooks_text is not None:
        current_hooks_text = _current_text(PRE_COMMIT_CONFIG)
        before_hooks = _gdd_hooks(yaml.safe_load(base_hooks_text))
        after_hooks = _gdd_hooks(yaml.safe_load(current_hooks_text or ""))
        for hook_id, hook in sorted(before_hooks.items()):
            candidate = after_hooks.get(hook_id)
            reviewed_hook = hook
            if isinstance(candidate, dict):
                # First check the existing path-filter rule with the
                # trusted dependencies restored. This never grants a new path
                # rule or hides entry/args/exclude/type changes. Then the exact
                # dependency-only migration sees otherwise identical hooks.
                historical_candidate = dict(candidate)
                if "additional_dependencies" in hook:
                    historical_candidate["additional_dependencies"] = hook["additional_dependencies"]
                else:
                    historical_candidate.pop("additional_dependencies", None)
                if not path_filter_relaxation_errors(hook, historical_candidate, hook_id):
                    reviewed_hook = historical_candidate
            errors.extend(
                f"{PRE_COMMIT_CONFIG}: {error}; explicit review required"
                for error in path_filter_relaxation_errors(reviewed_hook, candidate, hook_id)
            )
    errors.extend(settings_contract_errors(base_ref, ROOT, git_cwd=ROOT))
    # Protocol 1 carried a one-time, reviewed retirement of the old GDD
    # framework corpus. Once protocol 2 is the trusted base, definition-set
    # inclusion protects every retained framework detector without restoring a
    # count, digest, order snapshot, or blocking additive tests.
    if (base_policy_protocol(base_ref) or 0) >= 2:
        registry = _yaml_at(base_ref, TEST_REGISTRY)
        if not isinstance(registry, dict):
            errors.append(f"cannot read trusted framework registry {TEST_REGISTRY}")
        else:
            errors.extend(_trusted_framework_definition_errors(base_ref, registry))
    errors.extend(reviewed_inventory_change_errors(base_ref))
    return errors


def _trusted_framework_definition_errors(base_ref: str, registry: dict[str, Any]) -> list[str]:
    """Protect retained GDD-framework test definitions by set inclusion."""
    paths = {
        row["path"]
        for row in registry.get("exempt_files", ())
        if isinstance(row, dict) and isinstance(row.get("path"), str) and Path(row["path"]).name.startswith("test_gdd_")
    }
    for row in registry.get("exempt_roots", ()):
        root = row.get("root") if isinstance(row, dict) else None
        if not isinstance(root, str) or not root.startswith("tests/test_gdd"):
            continue
        listed = _git("ls-tree", "-r", "--name-only", base_ref, root)
        if listed.returncode:
            raise ValueError(f"cannot inspect trusted framework root {root}: {listed.stderr.strip()}")
        paths.update(path for path in listed.stdout.splitlines() if path.endswith(".py"))

    errors: list[str] = []
    for relative in sorted(paths):
        before = _blob_at(base_ref, relative)
        if before is None:
            errors.append(f"cannot read trusted framework test file {relative}")
            continue
        errors.extend(test_definition_inclusion_errors(before, _current_text(relative), relative))
    return errors


def reviewed_inventory_change_errors(base_ref: str) -> list[str]:
    """Protect the small set of live, semantic inventories."""
    protected = (
        "gdd/ssot/inventories/derivative_exclusions.yaml",
        "gdd/ssot/inventories/medium_tolerance.yaml",
        "gdd/ssot/inventories/route_counters.yaml",
        RUST_INVENTORY,
    )
    errors = []
    for relative in protected:
        before = _blob_at(base_ref, relative)
        current = _current_text(relative)
        if before is None or before == current:
            continue
        if relative == RUST_INVENTORY and current is not None:
            errors.extend(rust_inventory_status_change_errors(base_ref, before, current))
        else:
            errors.append(f"{relative}: reviewed inventory or schema changed; explicit review required")
    return errors


def _base_test_references(base_ref: str) -> set[str]:
    """Qualified Rust-export references in the trusted base's tests/ tree."""
    archive = subprocess.run(
        ["git", "archive", "--format=tar", base_ref, "--", "tests"], cwd=ROOT, capture_output=True, check=False
    )
    if archive.returncode:
        raise ValueError(f"cannot extract trusted tests at {base_ref}: {archive.stderr.decode(errors='replace').strip()}")
    with tempfile.TemporaryDirectory(prefix="gdd-base-tests-") as temporary:
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
            bundle.extractall(temporary, filter="data")
        return referenced_names(sorted(Path(temporary).rglob("*.py")))


def rust_inventory_status_change_errors(base_ref: str, before_text: str, current_text: str) -> list[str]:
    """Apply the standing Rust-inventory rule to a changed rust_inventory.yaml.

    Accepted without separate review: only python_exports status changes
    production-only -> tested or unused -> test-only, where the candidate's
    recomputed status equals the new value, the export's test reference is
    absent from the trusted base's tests/ and present in the candidate's, and
    with the base tests the candidate production code still gives the old
    status (the PR's own new test reference causes the change). Anything else,
    including any other field, row or status change, needs separate review.
    """
    review = "explicit review required"
    try:
        before = yaml.safe_load(before_text)
        after = yaml.safe_load(current_text)
    except yaml.YAMLError as exc:
        return [f"cannot parse {RUST_INVENTORY} for the standing Rust-inventory rule: {exc}"]
    if not isinstance(before, dict) or not isinstance(after, dict):
        return [f"{RUST_INVENTORY}: reviewed inventory is not a mapping; {review}"]
    other = sorted(key for key in set(before) | set(after) if key != "python_exports" and before.get(key) != after.get(key))
    if other:
        return [
            f"{RUST_INVENTORY}: changed reviewed fields {other} (only export statuses explained by new PR tests are accepted without review); {review}"
        ]
    before_rows = before.get("python_exports")
    after_rows = after.get("python_exports")
    if (
        not isinstance(before_rows, list)
        or not isinstance(after_rows, list)
        or len(before_rows) != len(after_rows)
        or any(
            not isinstance(old, dict)
            or not isinstance(new, dict)
            or set(old) != {"name", "status"}
            or set(new) != set(old)
            or old["name"] != new["name"]
            for old, new in zip(before_rows, after_rows)
        )
    ):
        return [f"{RUST_INVENTORY}: python_exports rows were added, removed, renamed, reordered or reshaped; {review}"]
    changed = [(old["name"], old["status"], new["status"]) for old, new in zip(before_rows, after_rows) if old != new]
    if not changed:
        return [f"{RUST_INVENTORY}: text changed without an export status change; {review}"]
    production, tests = reference_sets(ROOT)
    base_tests = _base_test_references(base_ref)
    errors = []
    for name, old, new in changed:
        short = name.rsplit(".", 1)[-1]
        if (old, new) not in RUST_TEST_REFERENCE_TRANSITIONS:
            reason = "this transition is not one that a new test reference explains"
        elif export_status(name, production, tests) != new:
            reason = f"the candidate's recomputed status is {export_status(name, production, tests)!r}"
        elif short in base_tests:
            reason = "the trusted base's tests already reference it"
        elif export_status(name, production, base_tests) != old:
            reason = "the candidate's production references, not its new tests, change the status"
        else:
            continue
        errors.append(
            f"{RUST_INVENTORY}: export {name} status {old!r} -> {new!r} is not explained by test references "
            f"this PR adds ({reason}); {review}"
        )
    return errors


def base_policy_protocol(base_ref: str) -> int | None:
    """Return the base checker's trusted-policy protocol, or None before it existed."""
    source = _blob_at(base_ref, "scripts/check_gdd_contract_changes.py")
    match = _PROTOCOL_PATTERN.search(source or "")
    return int(match.group(1)) if match else None


def run_trusted_base_policy(base_ref: str) -> int:
    """Execute the base's own checker on the candidate tree, isolated from candidate code."""
    declared = _git("ls-tree", "-r", "--name-only", base_ref, "--", *POLICY_TREES)
    if declared.returncode:
        print(
            f"GDD protected-contract change check failed against {base_ref}:\n"
            f"  - cannot inspect the trusted-base policy data: {declared.stderr.strip()}",
            file=sys.stderr,
        )
        return 1
    # Historical policies can predate any of the declarative migrations.
    # Archive only policy data actually present at the trusted ref, never a
    # candidate substitute. A policy needing missing data fails in its loader.
    # The two executable trees remain mandatory even when inspection is empty.
    declared_paths = set(declared.stdout.splitlines())
    policy_trees = tuple(path for path in POLICY_TREES if path in {"gdd/engine", "scripts"} or path in declared_paths)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", base_ref, "--", *policy_trees], cwd=ROOT, capture_output=True, check=False
    )
    if archive.returncode:
        print(
            f"GDD protected-contract change check failed against {base_ref}:\n"
            f"  - cannot extract the trusted-base policy: {archive.stderr.decode(errors='replace').strip()}",
            file=sys.stderr,
        )
        return 1
    with tempfile.TemporaryDirectory(prefix="gdd-trusted-policy-") as temporary:
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
            bundle.extractall(temporary, filter="data")
        script = Path(temporary) / "scripts/check_gdd_contract_changes.py"
        environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
        command = [
            sys.executable,
            "-I",
            str(script),
            "--trusted-policy-run",
            "--candidate-root",
            str(ROOT),
            "--base-ref",
            base_ref,
        ]
        completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    errors = [line[4:] for line in completed.stderr.splitlines() if line.startswith("  - ")]
    if completed.returncode and not errors:
        errors = [f"the trusted-base policy exited with status {completed.returncode} without listing changes"]
    if completed.returncode:
        report_protected_changes(base_ref, errors)
    return completed.returncode


_PATH_TOKEN = re.compile(r"(?<![\w/.-])(\.?[\w-]+(?:[./][\w-]+)*\.(?:ya?ml|py|json))(?![\w/-])")
_DIFF_LINES_PER_FILE = 200
REVIEW_GUIDANCE = (
    "Protected or weakening change. This check stays red: review the changes above; they are accepted "
    "only by merging the pull request."
)


def changed_protected_paths(base_ref: str, errors: list[str]) -> list[str]:
    """Repository paths named by the reported changes (ledger names resolve to gdd/ssot/guarantees/)."""
    paths: list[str] = []
    for error in errors:
        for token in _PATH_TOKEN.findall(error):
            for candidate in (token, f"{LEDGER_PREFIX}{token}"):
                if candidate in paths:
                    break
                if _blob_at(base_ref, candidate) is not None or (ROOT / candidate).is_file():
                    paths.append(candidate)
                    if candidate.startswith(LEDGER_PREFIX) and candidate.count("/") == 3:
                        # Issue 205: a ledger's guarantees live in its split files.
                        paths.extend(path for path in _changed_split_files(base_ref, candidate) if path not in paths)
                    break
    return paths


def _changed_split_files(base_ref: str, ledger: str) -> list[str]:
    directory = module_directory_relative(ledger)
    changed = _git("diff", "--name-only", base_ref, "--", directory)
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", directory)
    if changed.returncode or untracked.returncode:
        return []
    return sorted(set(changed.stdout.splitlines()) | set(untracked.stdout.splitlines()))


def _readable_diff(base_ref: str, relative: str) -> list[str]:
    before = _blob_at(base_ref, relative)
    after = _current_text(relative)
    lines = list(
        difflib.unified_diff(
            (before or "").splitlines(),
            (after or "").splitlines(),
            fromfile=f"{base_ref}:{relative}" if before is not None else "/dev/null",
            tofile=f"candidate:{relative}" if after is not None else "/dev/null",
            lineterm="",
        )
    )
    if len(lines) > _DIFF_LINES_PER_FILE:
        lines = lines[:_DIFF_LINES_PER_FILE] + [f"... diff truncated after {_DIFF_LINES_PER_FILE} lines"]
    return lines


def report_protected_changes(base_ref: str, errors: list[str]) -> None:
    """Show what changed and how on stderr (the CI job log)."""
    if not errors:
        return
    print(f"\nHow the protected inputs differ from the trusted base {base_ref}:", file=sys.stderr)
    for relative in changed_protected_paths(base_ref, errors):
        try:
            lines = _readable_diff(base_ref, relative)
        except (OSError, UnicodeDecodeError) as exc:
            lines = [f"cannot render diff: {exc}"]
        print(f"--- {relative}", file=sys.stderr)
        for line in lines or ["(no textual difference; see the reported change)"]:
            print(f"    {line}", file=sys.stderr)
    print("\n" + REVIEW_GUIDANCE, file=sys.stderr)


def _assert_policy_isolated() -> None:
    for module in DECISIVE_MODULES:
        location = Path(module.__file__).resolve()
        if not location.is_relative_to(POLICY_ROOT) or location.is_relative_to(ROOT):
            raise ValueError(f"trusted-policy run imported {module.__name__} from {location}, not the extracted base")


def _candidate_policy_bootstrap_allowed(base_protocol: int | None, candidate_protocol: int) -> bool:
    """Only the protocol-1 to protocol-2 transition may self-bootstrap."""
    return base_protocol == 1 and candidate_protocol == 2


def main() -> int:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ref", default=None)
    parser.add_argument("--candidate-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--trusted-policy-run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.candidate_root:
        ROOT = Path(args.candidate_root).resolve()
    warnings: list[str] = []
    policy = "candidate"
    try:
        if args.base_ref is not None and not args.base_ref.strip():
            raise ValueError("trusted base ref is empty; pass the PR base commit explicitly")
        base_ref = args.base_ref or _default_base()
        untrusted_base = args.base_ref is None and base_ref == "HEAD"
        base_ref = require_base_snapshot(base_ref)
        if untrusted_base and not args.trusted_policy_run:
            print(
                "WARNING: UNTRUSTED result: no trusted base (origin/main or GDD_TRUSTED_BASE_REF) is available; "
                "comparing with HEAD, which the candidate controls.",
                file=sys.stderr,
            )
        if args.trusted_policy_run:
            _assert_policy_isolated()
            policy = "trusted-base"
        else:
            protocol = base_policy_protocol(base_ref)
            if not _candidate_policy_bootstrap_allowed(protocol, TRUSTED_POLICY_PROTOCOL):
                if protocol is None:
                    raise ValueError(
                        "trusted base has no policy protocol and no candidate-policy bootstrap is permitted; "
                        "use a trusted base with protocol 2"
                    )
                if protocol > TRUSTED_POLICY_PROTOCOL:
                    print(f"note: trusted base uses newer policy protocol {protocol}; running it as-is.", file=sys.stderr)
                return run_trusted_base_policy(base_ref)
            warnings.append(
                f"BOOTSTRAP: trusted base {base_ref} uses policy protocol 1 and candidate protocol 2; this is the "
                "single permitted candidate-policy migration. Guard changes listed below need separate review."
            )
        guard_changes = guard_change_errors(base_ref, trusted_policy=args.trusted_policy_run)
        errors, summary = validation_errors(base_ref)
        errors.extend(trusted_source_errors(base_ref))
        if args.trusted_policy_run:
            errors = guard_changes + errors
        else:
            warnings.extend(guard_changes)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        base_ref = args.base_ref or os.environ.get("GDD_TRUSTED_BASE_REF") or "origin/main"
        errors, summary = [str(exc)], {}
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if errors:
        print(f"GDD protected-contract change check failed against {base_ref} ({policy} policy):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            "Intentional contract changes require a separate review; gdd_sync never accepts them.",
            file=sys.stderr,
        )
        if not args.trusted_policy_run:
            report_protected_changes(base_ref, errors)
        return 1
    print(
        f"GDD protected-contract change check passed against {base_ref} ({policy} policy; "
        f"{summary.get('added_guarantees', 0)} guarantees and {summary.get('added_nodes', 0)} evidence nodes added)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
