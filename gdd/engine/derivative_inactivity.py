"""Authored non-vacuity declarations shared by derivative tests and audit views.

Declarations do not waive comparisons. The harness still fails undeclared inactive
blocks and declarations whose reference has become live. The SSoT itself declares
the scope; adding or removing a declaration needs no duplicate inventory pin.
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
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RELATIVE_PATH = Path("gdd/ssot/fixtures/derivative_inactivity.yaml")
DISPOSITIONS = {"structurally-zero", "fixture-inactive", "zero-unexplained", "low-relative-sensitivity"}


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys instead of silently accepting the last one."""


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"derivative inactivity: invalid or duplicate YAML key {key!r}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _keys(value, expected, context):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"derivative inactivity: {context} requires exactly {sorted(expected)}")


def _text(value, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"derivative inactivity: {context} must be a nonempty string")
    return value


def load_derivative_inactivity(path: Path | None = None) -> dict[tuple[str, str, str, str], tuple[str, str]]:
    """Load authored declarations, rejecting malformed or ambiguous scope."""
    source = ROOT / RELATIVE_PATH if path is None else Path(path)
    try:
        document = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"derivative inactivity: invalid YAML: {exc}") from exc
    _keys(document, {"schema_version", "document", "enforcement_map", "reasons", "declarations"}, "document")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("derivative inactivity: schema_version must be 1")
    metadata = document["document"]
    _keys(metadata, {"id", "version", "lifecycle", "authority", "title", "purpose"}, "metadata")
    if metadata["id"] != "jqmc.fixtures.derivative-inactivity" or type(metadata["version"]) is not int:
        raise ValueError("derivative inactivity: invalid document identity/version")
    if metadata["version"] != 1:
        raise ValueError("derivative inactivity: document version must be 1")
    for name in ("lifecycle", "authority", "title", "purpose"):
        _text(metadata[name], name)
    enforcement = document["enforcement_map"]
    _keys(enforcement, {"reasons", "declarations"}, "enforcement_map")
    if any(value != "scripts/check_gdd_derivatives_contract.py" for value in enforcement.values()):
        raise ValueError("derivative inactivity: declarations and reasons require the derivative contract gate")
    reasons = {}
    if not isinstance(document["reasons"], list) or not document["reasons"]:
        raise ValueError("derivative inactivity: reasons must be a nonempty list")
    for row in document["reasons"]:
        _keys(row, {"id", "description"}, "reason")
        reason_id = _text(row["id"], "reason id")
        if reason_id in reasons:
            raise ValueError(f"derivative inactivity: duplicate reason {reason_id}")
        reasons[reason_id] = _text(row["description"], "reason description")
    if not isinstance(document["declarations"], list):
        raise ValueError("derivative inactivity: declarations must be a list")
    result = {}
    used_reasons = set()
    for row in document["declarations"]:
        _keys(row, {"cell_id", "quantity", "axis", "block", "disposition", "reason"}, "declaration")
        for name, value in row.items():
            _text(value, name)
        if row["quantity"] not in {"ln_Psi", "e_L"} or row["axis"] not in {"c", "r", "R", "eps"}:
            raise ValueError("derivative inactivity: invalid quantity or axis")
        if row["disposition"] not in DISPOSITIONS:
            raise ValueError("derivative inactivity: invalid disposition")
        if row["reason"] not in reasons:
            raise ValueError(f"derivative inactivity: unknown reason {row['reason']}")
        selector = tuple(row[name] for name in ("cell_id", "quantity", "axis", "block"))
        if selector in result:
            raise ValueError(f"derivative inactivity: duplicate selector {selector}")
        result[selector] = (row["disposition"], reasons[row["reason"]])
        used_reasons.add(row["reason"])
    if used_reasons != set(reasons):
        raise ValueError("derivative inactivity: unused reason")
    return result


def inactivity_contract_errors(root: Path = ROOT) -> list[str]:
    """Check the SSoT and the harness import/assignment without importing JAX."""
    try:
        load_derivative_inactivity(root / RELATIVE_PATH)
        tree = ast.parse((root / "tests/test_jqmc_kernel/_grad_fidelity_harness.py").read_text(encoding="utf-8"))
        imports = [
            alias
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "gdd.engine.derivative_inactivity"
            for alias in node.names
            if alias.name == "load_derivative_inactivity" and alias.asname is None
        ]
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            and any(
                isinstance(target, ast.Name) and target.id == "STRUCTURALLY_ZERO_BLOCKS"
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
        ]
        if len(imports) != 1 or len(assignments) != 1:
            raise ValueError("derivative inactivity: harness must import and assign the SSoT loader exactly once")
        call = assignments[0].value
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id != "load_derivative_inactivity"
            or call.args
            or call.keywords
        ):
            raise ValueError("derivative inactivity: harness declaration inventory must come from the default SSoT")
    except (OSError, SyntaxError, ValueError) as exc:
        return [str(exc)]
    return []
