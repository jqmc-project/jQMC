#!/usr/bin/env python3
"""Static CI gate validating every GDD SSoT document against its versioned schema.

Every YAML document under ``gdd/ssot/`` (except ``schemas/`` itself) must match
exactly one schema file from ``gdd/ssot/schemas/``. The checker also enforces the
repository-wide contract-authoring rules:

- a mapping key written in lowercase word style uses snake_case or kebab-case,
  never both inside one key, and stays short enough to be a token, not a sentence;
- scalar string values containing whitespace (prose) are only allowed under
  designated prose keys, so the machine-readable subset of a contract stays
  mechanically separable from its documentation;
- every ``lifecycle`` value uses the single reviewed vocabulary.
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

import ast
import fnmatch
import re
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SSOT_ROOT = ROOT / "gdd/ssot"
SCHEMA_ROOT = SSOT_ROOT / "schemas"
PREFLIGHT_PATH = ROOT / "scripts/run_gdd_preflight.py"

ALLOWED_LIFECYCLES = {"placeholder", "pre-cutover", "cutover"}
WORD_KEY = re.compile(r"^[a-z0-9_-]+$")
MAX_KEY_LENGTH = 48
MAX_KEY_SEGMENTS = 6

# Keys whose scalar string values are prose (may contain whitespace). A key also
# qualifies when it ends with "rule" or "condition". Everything else must hold
# whitespace-free tokens, references, or structured data.
PROSE_KEYS = {
    "applies_when",
    "command",
    "condition",
    "construction",
    "description",
    "formula",
    "input",
    "name",
    "notes",
    "numpy-equivalent",
    "plain_language",
    "purpose",
    "reason",
    "relationship",
    "requirement",
    "statement",
    "terminology",
    "title",
}
PROSE_KEY_SUFFIXES = ("rule", "condition")
# Materialized typed observations copy Python source expressions verbatim.
# Whitespace in these fields is machine syntax, not authored YAML prose.
MACHINE_EXPRESSION_KEYS = {
    "actual",
    "atol",
    "call",
    "expression",
    "parameter_id",
    "reference",
    "rtol",
    "values_source",
}


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject shadowed contract fields before YAML can discard an obligation."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
                seen.add(key)
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while reading a contract mapping", node.start_mark, "unhashable key", key_node.start_mark
                ) from exc
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while reading a contract mapping", node.start_mark, f"duplicate key {key!r}", key_node.start_mark
                )
        return super().construct_mapping(node, deep=deep)


def _load_yaml(path: Path, errors: list[str]) -> dict[str, Any] | None:
    from gdd.engine.guarantee_store import is_module_ledger, load_module

    try:
        if is_module_ledger(path):
            value = load_module(path, loader=_UniqueKeyLoader)
        else:
            value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        errors.append(f"{path.relative_to(ROOT)}: cannot load: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path.relative_to(ROOT)}: document must be a YAML mapping")
        return None
    return value


def _load_schemas(errors: list[str]) -> list[dict[str, Any]]:
    schemas = []
    ids: set[str] = set()
    for path in sorted(SCHEMA_ROOT.glob("*.yaml")):
        schema = _load_yaml(path, errors)
        if schema is None:
            continue
        label = path.relative_to(ROOT).as_posix()
        if schema.get("schema_format") != 1:
            errors.append(f"{label}: schema_format must be 1")
            continue
        schema_id = schema.get("schema_id")
        if not schema_id or schema_id in ids:
            errors.append(f"{label}: schema_id must be present and unique")
            continue
        ids.add(schema_id)
        match = schema.get("match", {})
        if not isinstance(match, dict) or not ({"path", "path_glob"} & set(match)):
            errors.append(f"{label}: match must declare path or path_glob")
            continue
        schema["_label"] = label
        errors.extend(_global_enforcement_errors(schema))
        schemas.append(schema)
    return schemas


def _retained_checkers() -> set[str]:
    """``RETAINED_CHECKERS`` of the GDD preflight, read from its source without importing it."""
    tree = ast.parse(PREFLIGHT_PATH.read_text(encoding="utf-8"), filename=str(PREFLIGHT_PATH))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RETAINED_CHECKERS" for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    return set()


def _global_enforcement_errors(schema: dict[str, Any]) -> list[str]:
    """A global enforcer is an existing checker script that the GDD preflight runs."""
    table = schema.get("global_enforcement", {})
    label = schema["_label"]
    if not isinstance(table, dict):
        return [f"{label}: global_enforcement must map sections to checker scripts"]
    retained = _retained_checkers()
    errors = []
    for section, enforced_by in table.items():
        name = Path(enforced_by).stem if isinstance(enforced_by, str) else None
        if not (
            isinstance(enforced_by, str)
            and enforced_by == f"scripts/{name}.py"
            and (ROOT / enforced_by).is_file()
            and name in retained
        ):
            errors.append(
                f"{label}: global_enforcement.{section} must be an existing scripts/<checker>.py listed in "
                f"RETAINED_CHECKERS of scripts/run_gdd_preflight.py: {enforced_by!r}"
            )
    return errors


def _select_schema(
    relative: str, document: dict[str, Any], schemas: list[dict[str, Any]], errors: list[str]
) -> dict[str, Any] | None:
    exact = [schema for schema in schemas if schema["match"].get("path") == relative]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        errors.append(f"{relative}: matches multiple exact-path schemas")
        return None
    lifecycle = (
        document.get("document", {}).get("lifecycle")
        if isinstance(document.get("document"), dict)
        else document.get("lifecycle")
    )
    globbed = [
        schema
        for schema in schemas
        if "path_glob" in schema["match"]
        and fnmatch.fnmatch(relative, schema["match"]["path_glob"])
        and schema["match"].get("lifecycle", lifecycle) == lifecycle
    ]
    if len(globbed) == 1:
        return globbed[0]
    if not globbed:
        errors.append(f"{relative}: no schema in gdd/ssot/schemas/ applies to this document")
    else:
        errors.append(f"{relative}: matches multiple schemas: {[schema['_label'] for schema in globbed]}")
    return None


def _enforcer_error(enforced_by: Any) -> str | None:
    """Why an enforced_by value is invalid, or None when it names a real enforcer."""
    if enforced_by == "not-yet-enforced":
        return None
    if isinstance(enforced_by, str) and enforced_by.startswith("gdd.engine."):
        module = enforced_by.split(".")[2]
        if not (ROOT / "gdd/engine" / f"{module}.py").is_file():
            return f"names a missing engine module: {enforced_by}"
        return None
    if isinstance(enforced_by, str) and (ROOT / enforced_by).is_file():
        return None
    return f"must be an existing script path, a gdd.engine symbol, or not-yet-enforced: {enforced_by!r}"


def _segment_count(key: str) -> int:
    return len([segment for segment in re.split(r"[-_]", key) if segment])


def _is_prose_key(key: str, extra: set[str]) -> bool:
    return key in PROSE_KEYS or key in extra or key.endswith(PROSE_KEY_SUFFIXES)


def _is_materialized_case_selector_value(path: str) -> bool:
    """Whether path is an opaque pytest parameter value copied from test AST.

    ``selector`` keys are the test's own parameter names, so their scalar values
    are allowed to be arbitrary literals rather than contract prose authored in
    the SSoT.  The surrounding path restriction keeps this exception local to
    the generated, per-case fixture inventory.
    """
    return ".case_binding.fixture.concrete_cases[" in path and "].selector." in path


def _is_required_parameter_value(path: str) -> bool:
    """Literal required pytest inputs may contain whitespace, just like collected inputs.

    Restrict this exception to the evidence-node parameter tuples. Their names,
    JSON types and exact equality with collected cases are checked separately
    by parameter_case_errors; no guarantee prose or other metadata is exempt.
    """
    return bool(
        re.match(
            r"^(?:guarantees|module_evidence)\[\d+\]\.required_evidence\[\d+\]"
            r"\.nodes\[\d+\]\.required_parameter_cases\[\d+\]\.",
            path,
        )
    )


def _is_machine_identifier_key(path: str) -> bool:
    """Executable-check mapping keys are stable machine IDs, not prose."""
    return ".executable_checks." in path


def _walk(value: Any, path: str, relative: str, extra_prose: set[str], errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and WORD_KEY.fullmatch(key):
                machine_identifier = _is_machine_identifier_key(child_path)
                if "_" in key and "-" in key and not machine_identifier:
                    errors.append(f"{relative}: key mixes snake_case and kebab-case: {child_path}")
                if len(key) > MAX_KEY_LENGTH and not machine_identifier:
                    errors.append(f"{relative}: key exceeds {MAX_KEY_LENGTH} characters: {child_path}")
                if _segment_count(key) > MAX_KEY_SEGMENTS and not machine_identifier:
                    errors.append(f"{relative}: key reads as a sentence ({_segment_count(key)} segments): {child_path}")
                if (
                    isinstance(child, str)
                    and re.search(r"\s", child)
                    and not _is_prose_key(key, extra_prose)
                    and key not in MACHINE_EXPRESSION_KEYS
                    and not _is_materialized_case_selector_value(child_path)
                    and not _is_required_parameter_value(child_path)
                ):
                    errors.append(
                        f"{relative}: prose value under non-prose key {child_path!r}; move it to a designated prose key such as notes"
                    )
            if isinstance(key, str) and key == "lifecycle" and child not in ALLOWED_LIFECYCLES:
                errors.append(f"{relative}: lifecycle {child!r} is not in {sorted(ALLOWED_LIFECYCLES)} at {child_path}")
            _walk(child, child_path, relative, extra_prose, errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, dict) and "rule" in child:
                enforced_by = child.get("enforced_by")
                if enforced_by is None:
                    errors.append(f"{relative}: declared rule has no enforced_by at {path}[{index}]")
                else:
                    problem = _enforcer_error(enforced_by)
                    if problem:
                        errors.append(f"{relative}: rule enforced_by at {path}[{index}] {problem}")
            _walk(child, f"{path}[{index}]", relative, extra_prose, errors)


def _check_document(path: Path, document: dict[str, Any], schema: dict[str, Any], errors: list[str]) -> None:
    relative = path.relative_to(ROOT).as_posix()
    if document.get("schema_version") != schema.get("document_schema_version"):
        errors.append(
            f"{relative}: schema_version {document.get('schema_version')!r} does not match "
            f"{schema['_label']} ({schema.get('document_schema_version')!r})"
        )
    present = set(document)
    required = set(schema.get("required_top_level", []))
    optional = set(schema.get("optional_top_level", []))
    selected_extensions = document.get("ledger_extensions", [])
    if not isinstance(selected_extensions, list) or any(not isinstance(item, str) for item in selected_extensions):
        errors.append(f"{relative}: ledger_extensions must be a list of extension IDs")
        selected_extensions = []
    extension_registry = schema.get("extensions", {})
    unknown_extensions = sorted(set(selected_extensions) - set(extension_registry))
    if unknown_extensions:
        errors.append(f"{relative}: unknown ledger extensions {unknown_extensions}")
    if not selected_extensions:
        required.update(schema.get("default_required_top_level", []))
    document_id = document.get("document", {}).get("id") if isinstance(document.get("document"), dict) else None
    for extension_id in selected_extensions:
        extension = extension_registry.get(extension_id, {})
        allowed_ids = extension.get("document_ids", [])
        if document_id not in allowed_ids:
            errors.append(f"{relative}: extension {extension_id} is not allowed for document {document_id!r}")
        required.update(extension.get("required_top_level", []))
        optional.update(extension.get("optional_top_level", []))
    missing = sorted(required - present)
    unknown = sorted(present - required - optional)
    if missing:
        errors.append(f"{relative}: missing required top-level fields {missing}")
    if unknown:
        errors.append(f"{relative}: top-level fields not declared by {schema['_label']}: {unknown}")

    block = schema.get("document_block")
    if block is not None:
        header = document.get("document")
        if not isinstance(header, dict):
            errors.append(f"{relative}: document block is required")
        else:
            required_fields = set(block.get("required_fields", []))
            if not selected_extensions:
                required_fields.update(block.get("default_required_fields", []))
            for extension_id in selected_extensions:
                required_fields.update(extension_registry.get(extension_id, {}).get("document_required_fields", []))
            missing_fields = sorted(required_fields - set(header))
            if missing_fields:
                errors.append(f"{relative}: document block is missing fields {missing_fields}")
            if not isinstance(header.get("version"), int):
                errors.append(f"{relative}: document version must be an integer")

    if schema.get("enforcement_map_required"):
        # A section is covered by the schema's global enforcer, which runs over
        # every document of this schema, or by a check specific to the document.
        global_enforcement = schema.get("global_enforcement", {})
        enforcement_map = document.get("enforcement_map", {} if "enforcement_map" in optional else None)
        if not isinstance(enforcement_map, dict):
            errors.append(f"{relative}: enforcement_map is required by {schema['_label']}")
        else:
            sections = set(document) - {"schema_version", "document", "enforcement_map"}
            missing_sections = sorted(sections - set(enforcement_map) - set(global_enforcement))
            unknown_sections = sorted(set(enforcement_map) - sections)
            repeated = sorted(
                section for section, enforced_by in enforcement_map.items() if global_enforcement.get(section) == enforced_by
            )
            if missing_sections:
                errors.append(f"{relative}: enforcement_map does not cover sections {missing_sections}")
            if unknown_sections:
                errors.append(f"{relative}: enforcement_map names unknown sections {unknown_sections}")
            if repeated:
                errors.append(f"{relative}: enforcement_map repeats the global enforcer of {schema['_label']} for {repeated}")
            for section, enforced_by in enforcement_map.items():
                if enforced_by == "per-rule":
                    body = document.get(section)
                    if not (
                        isinstance(body, list)
                        and body
                        and all(isinstance(item, dict) and "rule" in item and "enforced_by" in item for item in body)
                    ):
                        errors.append(
                            f"{relative}: enforcement_map.{section} is per-rule but the section is not a "
                            "list of rule items that each carry enforced_by"
                        )
                    continue
                problem = _enforcer_error(enforced_by)
                if problem:
                    errors.append(f"{relative}: enforcement_map.{section} {problem}")

    if schema.get("placeholder_invariants"):
        header = document.get("document", {})
        symbol_contract = document.get("symbol_contract", {})
        if header.get("lifecycle") != "placeholder" or symbol_contract.get("status") != "placeholder":
            errors.append(f"{relative}: placeholder ledger must declare placeholder lifecycle and symbol status together")
        if document.get("guarantees") != []:
            errors.append(f"{relative}: placeholder ledger must declare an empty guarantee list")

    extra_prose = set(schema.get("prose_keys_extra", []))
    _walk(document, "", relative, extra_prose, errors)


def check() -> list[str]:
    errors: list[str] = []
    schemas = _load_schemas(errors)
    if not schemas:
        errors.append("no schemas found under gdd/ssot/schemas/")
        return errors
    from gdd.engine.guarantee_store import GUARANTEES_RELATIVE, is_split_guarantee_file, layout_errors

    if (ROOT / GUARANTEES_RELATIVE).is_dir():
        errors.extend(layout_errors(ROOT))
    documents = [
        path
        for path in sorted(SSOT_ROOT.rglob("*.yaml"))
        if SCHEMA_ROOT not in path.parents and not is_split_guarantee_file(path)
    ]
    if not documents:
        errors.append("no SSoT documents found under gdd/ssot/")
    for path in documents:
        document = _load_yaml(path, errors)
        if document is None:
            continue
        relative = path.relative_to(ROOT).as_posix()
        schema = _select_schema(relative, document, schemas, errors)
        if schema is None:
            continue
        _check_document(path, document, schema, errors)
    return errors


def main() -> int:
    errors = check()
    if errors:
        print("GDD SSoT schema check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("GDD SSoT schema check passed (every document matches one versioned schema).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
