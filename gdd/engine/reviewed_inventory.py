"""Load reviewed inventories without embedding their contract contents in Python.

The inventory owns values and vocabularies; its versioned YAML schema owns
structural constraints. Missing or malformed inputs fail closed. Nothing is
cached, so callers auditing a scratch repository read that repository's files.
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

import re
from pathlib import Path
from typing import Any

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys, including keys introduced by YAML merges."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
                seen.add(key)
            except TypeError as exc:
                raise ValueError("reviewed inventory contains an unhashable mapping key") from exc
            if duplicate:
                raise ValueError(f"reviewed inventory contains duplicate key {key!r}")
        return super().construct_mapping(node, deep=deep)


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"cannot load reviewed inventory input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def _validate(value: Any, rule: dict[str, Any], document: dict[str, Any], path: str) -> None:
    if not isinstance(rule, dict):
        raise ValueError(f"{path}: malformed runtime schema")
    kind = rule.get("type")
    supported = {
        "mapping": {"properties", "required", "additional_properties", "disjoint_sections"},
        "sequence": {"items", "min_items", "unique_items", "unique_by"},
        "string": {"min_length", "enum_from"},
        "integer": {"minimum"},
    }
    if kind not in supported or set(rule) - (supported[kind] | {"type", "const"}):
        raise ValueError(f"{path}: unsupported runtime schema constraint")
    matches = {
        "mapping": isinstance(value, dict),
        "sequence": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": type(value) is int,
    }
    if not matches[kind]:
        raise ValueError(f"{path}: expected {kind}")
    if "const" in rule and (type(value) is not type(rule["const"]) or value != rule["const"]):
        raise ValueError(f"{path}: value does not match schema constant")
    if kind == "mapping":
        properties = rule["properties"]
        required = rule["required"]
        if not isinstance(properties, dict) or not isinstance(required, list) or set(required) - set(properties):
            raise ValueError(f"{path}: malformed mapping schema")
        missing = set(required) - set(value)
        unknown = set(value) - set(properties)
        if missing or (unknown and rule["additional_properties"] is not True):
            raise ValueError(f"{path}: missing fields {sorted(missing)}; unknown fields {sorted(unknown, key=str)}")
        for key, child in value.items():
            if key in properties:
                _validate(child, properties[key], document, f"{path}.{key}")
        for constraint in rule.get("disjoint_sections", []):
            for field in constraint["fields"]:
                seen = set()
                for section in constraint["sections"]:
                    for row in value[section]:
                        if row[field] in seen:
                            raise ValueError(f"{path}: repeated {field} across {constraint['sections']}")
                        seen.add(row[field])
    elif kind == "sequence":
        if len(value) < rule["min_items"]:
            raise ValueError(f"{path}: requires at least {rule['min_items']} entries")
        for index, row in enumerate(value):
            _validate(row, rule["items"], document, f"{path}[{index}]")
        if rule.get("unique_items") and len(value) != len(set(value)):
            raise ValueError(f"{path}: duplicate entries")
        for field in rule.get("unique_by", []):
            keys = [row[field] for row in value]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{path}: duplicate {field}")
    elif kind == "string":
        if len(value.strip()) < rule["min_length"]:
            raise ValueError(f"{path}: empty string")
        if "enum_from" in rule:
            choices: Any = document
            for key in rule["enum_from"].split("."):
                choices = choices[key]
            if not isinstance(choices, list) or not choices or value not in choices:
                raise ValueError(f"{path}: value {value!r} is not in {rule['enum_from']}")
    elif kind == "integer" and "minimum" in rule and value < rule["minimum"]:
        raise ValueError(f"{path}: value below schema minimum")


def load_reviewed_inventory(name: str, root: Path | None = None) -> dict[str, Any]:
    """Read and validate one inventory, including its authoritative vocabulary.

    ``root`` is the candidate repository root, not the current working directory.
    The default locates the repository containing this loader. Consumers auditing
    a scratch tree should pass its root explicitly rather than use module exports.
    """
    if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
        raise ValueError("inventory name must be a lowercase filename token")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    relative = f"gdd/ssot/inventories/{name}.yaml"
    schema = _read_mapping(root / "gdd/ssot/schemas" / f"reviewed-{name}.v1.yaml")
    document = _read_mapping(root / relative)
    try:
        if type(schema["schema_format"]) is not int or schema["schema_format"] != 1:
            raise ValueError("unsupported schema format")
        if schema["match"] != {"path": relative}:
            raise ValueError("schema path does not match inventory")
        if document["schema_version"] != schema["document_schema_version"]:
            raise ValueError("document/schema version mismatch")
        present = set(document)
        required = set(schema["required_top_level"])
        optional = set(schema["optional_top_level"])
        if required - present or present - required - optional:
            raise ValueError("inventory top-level fields do not match schema")
        _validate(document, schema["runtime_schema"], document, relative)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid reviewed inventory {relative}: {exc}") from exc
    return document
