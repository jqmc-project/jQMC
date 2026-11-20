"""Authored comparison requirements, independent of regenerated test descriptions.

The adapters cover NumPy assert_allclose/allclose/isclose with registered zone and named-pair policies,
and assert_array_equal with the registered exact-values policy (not raw bits).
Hand-written ``assert`` inequalities in a reviewed form (gdd.engine.inequality_receipts)
use the ``python.inequality`` comparator kind with the same allclose tolerance policies;
their receipt is the operand/tolerance values captured on entry and the replayed form.
A zone policy may declare ``case_zones`` (one explicit zone list per concrete
parameter case, never a union) instead of ``zones`` when the source selects
its zone list from a test parameter through a reviewed static table.
Absence of a requirement is NOT certification of a source-described tolerance.
Unsupported requirements fail closed; expanding the adapters is separate work.
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
import copy
from functools import cache
import json
import keyword
import math
from pathlib import Path
from typing import Any

import yaml

from gdd.engine import exact_tree_requirements, inequality_receipts, spacing_requirements, zone_zero_requirements
from gdd.engine.evidence_roles import comparison_method, load_roles, role_errors


ROOT = Path(__file__).resolve().parents[2]
_EXACT_POLICY = "CMP-EXACT-VALUES-001"
_DECIMAL_POLICY = "CMP-TURBORVB-DECIMAL-001"
_CENTRAL_PAIR_POLICY = "CMP-SOURCE-CENTRAL-PAIR-001"
_MIXED_FIDELITY_POLICY = "CMP-MIXED-FULL-FIDELITY-001"
_DECIMAL_COMPARATORS = {"numpy.testing.assert_almost_equal", "numpy.testing.assert_array_almost_equal"}
_DECIMAL_SYMBOLS = {
    "jqmc._setting.DECIMAL_TURBORVB_VALUES",
    "jqmc._setting.DECIMAL_TURBORVB_ECP_POTENTIAL",
    "jqmc._setting.DECIMAL_COULOMB_ROUTE_PARITY",
}
_POLICIES = {
    "CMP-ZONE-ALLCLOSE-STRICT-001": ("_TOLERANCE", "strict", "jqmc._precision.get_tolerance_min"),
    "CMP-ZONE-ALLCLOSE-MEDIUM-001": ("_TOLERANCE", "medium", "jqmc._precision.get_tolerance_min"),
    "CMP-XLA-RUST-STRICT-001": (
        "_XLA_RUST_CONSISTENCY_TOLERANCE",
        "strict",
        "jqmc._precision.get_tolerance_xla_rust_consistency",
    ),
    "CMP-XLA-RUST-MEDIUM-001": (
        "_XLA_RUST_CONSISTENCY_TOLERANCE",
        "medium",
        "jqmc._precision.get_tolerance_xla_rust_consistency",
    ),
    "CMP-XLA-RUST-INDEPENDENT-FP32-001": (
        "_XLA_RUST_CONSISTENCY_TOLERANCE",
        "independent_fp32",
        "jqmc._precision.get_tolerance_xla_rust_consistency",
    ),
}
_SYMBOL_POLICIES = {
    "CMP-PW-PROJECTION-FP64-001": "TOL-PW-PROJECTION-FP64-001",
    "CMP-LM-HOST-FP64-001": "TOL-LM-HOST-FP64-001",
    "CMP-TREXIO-STORAGE-FP64-001": "TOL-TREXIO-STORAGE-FP64-001",
    "CMP-FIXED-SAMPLE-FP64-001": "TOL-FIXED-SAMPLE-FP64-001",
    "CMP-NATIVE-GAUSSIAN-F64-001": "TOL-NATIVE-GAUSSIAN-F64-001",
    "CMP-DERIV-AUTODIFF-ANALYTICAL-FULL-001": "TOL-DERIV-AUTODIFF-ANALYTICAL-FULL-001",
    "CMP-FD-AUTODIFF-FULL-001": "TOL-FD-AUTODIFF-FULL-001",
    "CMP-INTEGRATION-MIXED-001": "TOL-INTEGRATION-MIXED-001",
    "CMP-GFMC-INCREMENTAL-TWIN-001": "TOL-GFMC-INCREMENTAL-TWIN-001",
    "CMP-DTN-EN-CUSP-CONSTRAINT-001": "TOL-DTN-EN-CUSP-CONSTRAINT-001",
}


@cache
def _definitions(path: Path) -> dict[str, ast.expr]:
    result = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            result[node.target.id] = node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node.value
    return result


def _static_value(value: ast.expr, definitions: dict[str, ast.expr], seen: frozenset[str] = frozenset()) -> Any:
    if isinstance(value, ast.Name):
        if value.id in seen or value.id not in definitions:
            raise ValueError(f"unresolved central definition: {value.id}")
        return _static_value(definitions[value.id], definitions, seen | {value.id})
    if isinstance(value, ast.Dict):
        result = {}
        for key, child in zip(value.keys, value.values, strict=True):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str) or key.value in result:
                raise ValueError("central definition has an invalid/duplicate mapping key")
            result[key.value] = _static_value(child, definitions, seen)
        return result
    if isinstance(value, (ast.Tuple, ast.List)):
        return [_static_value(child, definitions, seen) for child in value.elts]
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.UAdd, ast.USub)):
        operand = _static_value(value.operand, definitions, seen)
        if type(operand) not in (float, int):
            raise ValueError("nonnumeric central unary expression")
        return operand if isinstance(value.op, ast.UAdd) else -operand
    if isinstance(value, ast.Constant) and type(value.value) in (str, float, int):
        return value.value
    raise ValueError("central definition is not a supported static value")


def _case_json(value: Any) -> str:
    """Canonical text of a JSON parameter value; bool, int and float stay distinct."""

    def check(item):
        if type(item) in (str, bool, int, type(None)):
            return
        if type(item) is float and math.isfinite(item):
            return
        if isinstance(item, list):
            for child in item:
                check(child)
            return
        if isinstance(item, dict) and all(type(name) is str for name in item):
            for child in item.values():
                check(child)
            return
        raise ValueError("per-case selectors require JSON parameter values")

    check(value)
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _validate_case_zones(entries: Any) -> None:
    """Each entry is one explicit parameter selector with its own zone list."""
    if not isinstance(entries, list) or not entries:
        raise ValueError("case_zones requires a nonempty explicit case list")
    names = None
    identities = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"selector", "zones"}:
            raise ValueError("a per-case zone entry requires exactly selector and zones")
        selector, zones = entry["selector"], entry["zones"]
        if (
            not isinstance(selector, dict)
            or not selector
            or any(type(name) is not str or not name.isidentifier() or keyword.iskeyword(name) for name in selector)
        ):
            raise ValueError("per-case zone selectors require named test parameters")
        identity = _case_json(selector)
        if identity in identities or (names is not None and set(selector) != names):
            raise ValueError("per-case zone selectors are duplicated or have different parameter names")
        identities.add(identity)
        names = set(selector)
        if (
            not isinstance(zones, list)
            or not zones
            or any(not isinstance(zone, str) for zone in zones)
            or len(zones) != len(set(zones))
        ):
            raise ValueError("comparison zones must be nonempty, distinct names")


def _case_selected(entry_selector: dict[str, Any], selector: Any) -> bool:
    return isinstance(selector, dict) and all(
        name in selector and _case_json(selector[name]) == _case_json(value) for name, value in entry_selector.items()
    )


def case_zones_for(entries: Any, selector: Any) -> list[str]:
    """Return the zone list declared for exactly this concrete case; never merge entries."""
    _validate_case_zones(entries)
    if not isinstance(selector, dict) or not selector:
        raise ValueError("per-case zones require a concrete parameter selector")
    matches = [entry for entry in entries if _case_selected(entry["selector"], selector)]
    if len(matches) != 1:
        raise ValueError("the parameter case is not covered by exactly one per-case zone entry")
    return list(matches[0]["zones"])


def resolve_requirement(
    disposition: dict[str, Any],
    precision: str,
    *,
    root: Path = ROOT,
    selector: dict[str, Any] | None = None,
    component_index: int | None = None,
) -> dict[str, float]:
    """Read central definitions without calling production or its mutable resolver.

    A per-case zone disposition resolves only for a concrete ``selector``.
    """
    if not isinstance(disposition, dict) or "policy" not in disposition:
        raise ValueError("a comparison requires an explicit registered policy")
    policy = disposition["policy"]
    if (
        not isinstance(policy, str)
        or policy
        not in (
            *_POLICIES,
            *_SYMBOL_POLICIES,
            _MIXED_FIDELITY_POLICY,
            _EXACT_POLICY,
            _DECIMAL_POLICY,
            _CENTRAL_PAIR_POLICY,
            exact_tree_requirements.POLICY,
            spacing_requirements.POLICY,
            zone_zero_requirements.POLICY,
        )
        or precision not in {"full", "mixed"}
    ):
        raise ValueError("unsupported comparison policy or precision")
    registry = _verdict_registry(root)
    definition = next((row for row in registry["comparison_policies"] if row["id"] == policy), None)
    if definition is None or precision not in definition["allowed-precision-modes"]:
        raise ValueError("comparison policy is not registered for this precision")
    if policy == _MIXED_FIDELITY_POLICY:
        if (
            set(disposition) != {"policy", "quantity_class"}
            or definition["method"] != "allclose"
            or definition["tolerance-binding"]
            != {"registry-id": "TOL-MIXED-FIDELITY-001", "case-required-arguments": ["quantity_class"]}
        ):
            raise ValueError("mixed fidelity requires one registered quantity class")
        quantity_class = disposition["quantity_class"]
        if type(quantity_class) is not str:
            raise ValueError("mixed-fidelity quantity class must be a string")
        resolver = next(
            (row for row in registry["tolerance_registry"]["resolvers"] if row["id"] == "TOL-MIXED-FIDELITY-001"),
            None,
        )
        if resolver != {
            "id": "TOL-MIXED-FIDELITY-001",
            "resolver": "jqmc._setting.get_tolerance_mixed_fidelity",
            "required-arguments": ["quantity_class"],
            "mode-sensitive": False,
        }:
            raise ValueError("mixed-fidelity resolver differs from its registry")
        settings = _definitions(root / "jqmc/_setting.py")
        table = _static_value(settings["_MIXED_FIDELITY_TOLERANCE"], settings)
        if quantity_class not in table:
            raise ValueError("unknown mixed-fidelity quantity class")
        pair = table[quantity_class]
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in pair)
        ):
            raise ValueError("mixed-fidelity tolerance must be a finite nonnegative pair")
        return {"atol": float(pair[0]), "rtol": float(pair[1])}
    if policy == zone_zero_requirements.POLICY:
        zone_zero_requirements.binding(definition)
        rows = zone_zero_requirements.components(disposition)
        if component_index is None:
            if len(rows) != 1:
                raise ValueError("repeated zone/zero comparisons require a concrete component index")
            component_index = 0
        if type(component_index) is not int or component_index not in range(len(rows)):
            raise ValueError("zone/zero component is outside its declared inventory")
        result = resolve_requirement(
            {"policy": "CMP-ZONE-ALLCLOSE-STRICT-001", "zones": rows[component_index]["zones"]}, precision, root=root
        )
        zero = _central_symbol_reader(root)(zone_zero_requirements.ZERO_SYMBOL)
        if type(zero) not in (float, int) or zero != 0:
            raise ValueError("zone/zero relative tolerance must remain the central exact zero")
        return {"atol": result["atol"], "rtol": float(zero)}
    if policy == spacing_requirements.POLICY:
        return spacing_requirements.resolve(disposition, definition, selector, _central_symbol_reader(root))
    if policy == _DECIMAL_POLICY:
        if (
            set(disposition) != {"policy", "symbol"}
            or disposition["symbol"] not in _DECIMAL_SYMBOLS
            or definition["method"] != "decimal-absolute"
            or definition["tolerance-binding"] != {"registry-id": "TOL-TURBORVB-DECIMAL-001"}
        ):
            raise ValueError("decimal comparison must retain its registered central digit count")
        row = next(
            (
                row
                for row in registry["tolerance_registry"].get("decimal_symbols", ())
                if row.get("id") == "TOL-TURBORVB-DECIMAL-001"
            ),
            None,
        )
        if row is None or set(row.get("symbols", ())) != _DECIMAL_SYMBOLS:
            raise ValueError("decimal symbol registry differs from the reviewed adapter")
        settings = _definitions(root / "jqmc/_setting.py")
        digits = _static_value(settings[disposition["symbol"].removeprefix("jqmc._setting.")], settings)
        if type(digits) is not int or not 0 <= digits <= 308:
            raise ValueError("central decimal count must be a supported nonnegative integer")
        return {"decimal": digits}
    if policy == exact_tree_requirements.POLICY:
        if (
            set(disposition) != {"policy"}
            or definition["method"] != "source-array-equality"
            or definition["tolerance-binding"] != {"registry-id": "TOL-BIT-EXACT-001"}
        ):
            raise ValueError("source array equality must retain its registered exact semantics")
        return {}
    if policy == _EXACT_POLICY:
        if (
            set(disposition) != {"policy"}
            or definition["method"] != "exact-values"
            or definition["tolerance-binding"] != {"registry-id": "TOL-BIT-EXACT-001"}
        ):
            raise ValueError("exact-value policy cannot carry tolerances or change its registered semantics")
        return {}
    if definition["method"] != "allclose":
        raise ValueError("comparison policy is not registered as allclose")
    if policy in _SYMBOL_POLICIES or policy == _CENTRAL_PAIR_POLICY:
        fields = {"policy", "pair"} if policy == _CENTRAL_PAIR_POLICY else {"policy"}
        if set(disposition) != fields:
            raise ValueError("a named-symbol comparison cannot carry resolver arguments")
        settings = _definitions(root / "jqmc/_setting.py")
        result = {}
        for component, symbol in named_tolerance_symbols(policy, root=root, pair_id=disposition.get("pair")).items():
            value = _static_value(settings[symbol.removeprefix("jqmc._setting.")], settings)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("central tolerance must be a finite, nonnegative numeric scalar")
            result[component] = float(value)
        return result
    if set(disposition) == {"policy", "case_zones"}:
        disposition = {"policy": policy, "zones": case_zones_for(disposition["case_zones"], selector)}
    if set(disposition) != {"policy", "zones"}:
        raise ValueError("a zone comparison requires exactly policy and zones (or policy and case_zones)")
    zones = disposition["zones"]
    if (
        not isinstance(zones, list)
        or not zones
        or any(not isinstance(zone, str) for zone in zones)
        or len(zones) != len(set(zones))
    ):
        raise ValueError("comparison zones must be nonempty, distinct names")
    table_name, level, resolver = _POLICIES[policy]
    binding = definition["tolerance-binding"]
    declared_resolver = next(
        (row for row in registry["tolerance_registry"]["resolvers"] if row["id"] == binding["registry-id"]), None
    )
    if declared_resolver is None or declared_resolver["resolver"] != resolver or binding["fixed-arguments"] != {"level": level}:
        raise ValueError("comparison policy adapter differs from its registry")
    settings = _definitions(root / "jqmc/_setting.py")
    layout = _definitions(root / "jqmc/_precision.py")
    table = _static_value(settings[table_name], settings)
    dtypes = _static_value(layout[f"_{precision.upper()}_PRECISION"], layout)
    pairs = [table[level][dtypes[zone]] for zone in zones]
    if any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs):
        raise ValueError("central tolerance row must contain atol and rtol")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for pair in pairs for value in pair):
        raise ValueError("central tolerance must be finite and nonnegative")
    return {"atol": float(max(pair[0] for pair in pairs)), "rtol": float(max(pair[1] for pair in pairs))}


def named_tolerance_symbols(policy: str, *, root: Path = ROOT, pair_id: str | None = None) -> dict[str, str]:
    """Return only a supported policy's registered pair, never a guessed pair of names."""
    if policy not in _SYMBOL_POLICIES and policy != _CENTRAL_PAIR_POLICY:
        raise ValueError("not a supported named-symbol policy")
    registry = _verdict_registry(root)
    definition = next(row for row in registry["comparison_policies"] if row["id"] == policy)
    if policy == _CENTRAL_PAIR_POLICY:
        if definition.get("tolerance-binding") != {"registry-kind": "symbol-pairs", "selector": "pair"}:
            raise ValueError("central-pair policy must select a declared symbol pair")
        scopes = _central_pair_scopes(definition)
        if not isinstance(pair_id, str) or pair_id not in {row["pair"] for row in scopes}:
            raise ValueError("central pair is not registered for a reviewed comparison")
        registry_id = pair_id
    else:
        if pair_id is not None:
            raise ValueError("fixed named-symbol policy cannot select another pair")
        registry_id = _SYMBOL_POLICIES[policy]
        if definition["tolerance-binding"] != {"registry-id": registry_id}:
            raise ValueError("named-symbol policy adapter differs from its registry")
    pairs = [row for row in registry["tolerance_registry"]["symbol_pairs"] if row["id"] == registry_id]
    if len(pairs) != 1:
        raise ValueError("named-symbol policy requires one unique registered pair")
    pair = pairs[0]
    result = {name: pair[f"{name}-symbol"] for name in ("atol", "rtol")}
    if any(not isinstance(name, str) or not name.startswith("jqmc._setting.") for name in result.values()):
        raise ValueError("named tolerance must refer to the central setting module")
    return result


def _central_pair_scopes(definition: dict[str, Any]) -> list[dict[str, Any]]:
    """Read finite consumer tuples from SSoT, never infer permission from source."""
    rows = definition.get("allowed-comparison-bindings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("central-pair policy requires explicit finite comparison bindings")
    fields = {"site", "pytest_node", "verdict_method", "comparator", "precision", "pair"}
    seen = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != fields
            or any(not isinstance(row[key], str) or not row[key] for key in fields)
            or row["precision"] not in {"full", "mixed"}
            or row["comparator"] not in {"numpy.testing.assert_allclose", "numpy.allclose", "numpy.isclose"}
            or "::" not in row["site"]
            or "#comparison-" not in row["site"]
            or "::" not in row["pytest_node"]
        ):
            raise ValueError("central-pair comparison binding is malformed")
        identity = tuple(row[key] for key in ("site", "pytest_node", "verdict_method", "precision"))
        if identity in seen:
            raise ValueError("central-pair comparison binding is duplicate or ambiguous")
        seen.add(identity)
    return rows


@cache
def _verdict_registry(root: Path) -> dict[str, Any]:
    return yaml.safe_load((root / "gdd/ssot/verdicts.yaml").read_text(encoding="utf-8"))


def _central_symbol_reader(root):
    settings = _definitions(root / "jqmc/_setting.py")

    def read(name):
        if not isinstance(name, str) or not name.startswith("jqmc._setting."):
            raise ValueError("comparison symbol must belong to the central settings")
        return _static_value(settings[name.removeprefix("jqmc._setting.")], settings)

    return read


def _description_value(value: dict[str, Any], *, precision: str | None = None) -> Any:
    if value.get("kind") == "literal":
        return ast.literal_eval(value["value"])
    if value.get("kind") in {"list", "tuple"}:
        return [_description_value(child, precision=precision) for child in value["items"]]
    if value.get("kind") == "expression" and value.get("operation") == "BinOp":
        # This evaluates only source-resolved string sequences, not Python
        # objects, arbitrary addition or tolerance arithmetic. Preserve the
        # original container kind so list + tuple cannot silently be accepted.
        def sequence(description):
            kind = description.get("kind")
            if kind in {"list", "tuple"}:
                items = _description_value(description, precision=precision)
                if all(type(item) is str for item in items):
                    return kind, items
            if kind == "expression" and description.get("operation") == "BinOp":
                fields = description.get("fields", {})
                if set(fields) == {"left", "op", "right"} and fields["op"] == {
                    "kind": "expression",
                    "operation": "Add",
                    "fields": {},
                }:
                    left_kind, left = sequence(fields["left"])
                    right_kind, right = sequence(fields["right"])
                    if left_kind == right_kind:
                        return left_kind, left + right
            raise ValueError("tolerance sequence concatenation requires same-kind static string sequences")

        return sequence(value)[1]
    if value.get("kind") == "expression" and value.get("operation") == "IfExp":
        # Resolve only the reviewed central mode predicate, using the authored
        # lane rather than importing/calling a potentially patched function.
        # Both arms must be literal names; arbitrary expressions stay closed.
        fields = value.get("fields", {})
        predicate = {
            "kind": "call",
            "function": {"kind": "central-reference", "name": "jqmc._precision.is_mixed_precision_enabled"},
            "args": [],
            "keywords": [],
        }
        if (
            precision not in {"full", "mixed"}
            or set(fields) != {"test", "body", "orelse"}
            or fields["test"] != predicate
            or any(fields[arm].get("kind") != "literal" for arm in ("body", "orelse"))
        ):
            raise ValueError("unsupported precision-dependent tolerance selector")
        choices = {arm: _description_value(fields[arm]) for arm in ("body", "orelse")}
        if any(type(choice) is not str for choice in choices.values()):
            raise ValueError("precision-dependent tolerance levels must be literal strings")
        return choices["body" if precision == "mixed" else "orelse"]
    raise ValueError("tolerance resolver arguments are not statically resolved")


def _case_table_entry(value: Any, *, precision: str, selector: dict[str, Any], scope: str) -> Any:
    """Return the description a subscript selects from a source dict display, else None.

    The display may be subscripted directly or through nested table subscripts
    (``TABLE[a][b]``). Keys must be distinct literals; the key is evaluated for the
    case and must match one literal of the same type. Other values are not read.
    """
    if not (isinstance(value, dict) and value.get("kind") == "expression" and value.get("operation") == "Subscript"):
        return None
    fields = value.get("fields", {})
    if not isinstance(fields, dict) or set(fields) != {"value", "slice"}:
        raise ValueError("unsupported per-case subscript")
    container = fields["value"]
    if not (isinstance(container, dict) and container.get("kind") == "expression" and container.get("operation") == "Dict"):
        container = _case_table_entry(container, precision=precision, selector=selector, scope=scope)
        if not (isinstance(container, dict) and container.get("kind") == "expression" and container.get("operation") == "Dict"):
            return None
    table = container.get("fields", {})
    keys, values = table.get("keys"), table.get("values")
    if (
        not isinstance(table, dict)
        or set(table) != {"keys", "values"}
        or not isinstance(keys, list)
        or not isinstance(values, list)
        or len(keys) != len(values)
    ):
        raise ValueError("unsupported per-case table")
    literals = []
    for item in keys:
        # ``None`` is a ``**mapping`` expansion; only literal keys are reviewed.
        if not isinstance(item, dict) or item.get("kind") != "literal":
            raise ValueError("per-case table keys must be literals")
        literal = ast.literal_eval(item["value"])
        if type(literal) not in (str, int, bool):
            raise ValueError("per-case table keys must be string, integer or boolean literals")
        literals.append(literal)
    # Python merges equal keys (e.g. True and 1); such a table is ambiguous.
    if len({(literal, type(literal) is str) for literal in literals}) != len(literals):
        raise ValueError("per-case table has duplicate or aliased keys")
    key = _case_description_value(fields["slice"], precision=precision, selector=selector, scope=scope)
    if type(key) not in (str, int, bool):
        raise ValueError("per-case table keys must be string, integer or boolean values")
    selected = [index for index, literal in enumerate(literals) if type(literal) is type(key) and literal == key]
    if len(selected) != 1:
        raise ValueError("per-case table has no entry for this parameter case")
    return values[selected[0]]


def _case_description_value(value: Any, *, precision: str, selector: dict[str, Any], scope: str) -> Any:
    """Evaluate a resolver argument for ONE concrete parameter case, failing closed.

    Beyond ``_description_value`` this reads only (a) a parameter of the owning
    test function, taken from the case selector, and (b) a subscript of a
    source dict display with distinct literal keys (the selected value is then
    evaluated alone) or of a selector-provided mapping by a string key. No
    call, comparison, branch or arithmetic on parameters is evaluated.
    """
    if not isinstance(value, dict):
        raise ValueError("tolerance resolver arguments are not statically resolved")
    if value.get("kind") == "parameter":
        name = value.get("name")
        if value.get("scope") != scope or not isinstance(name, str) or name not in selector:
            raise ValueError("per-case zone selection reads a value that is not a parameter of this test case")
        _case_json(selector[name])
        return copy.deepcopy(selector[name])
    if value.get("kind") == "expression" and value.get("operation") == "Subscript":
        context = {"precision": precision, "selector": selector, "scope": scope}
        entry = _case_table_entry(value, **context)
        if entry is not None:
            return _case_description_value(entry, **context)
        fields = value.get("fields", {})
        if not isinstance(fields, dict) or set(fields) != {"value", "slice"}:
            raise ValueError("unsupported per-case subscript")
        key = _case_description_value(fields["slice"], **context)
        mapping = _case_description_value(fields["value"], **context)
        if not isinstance(mapping, dict) or type(key) is not str or key not in mapping:
            raise ValueError("per-case subscript does not select a parameter mapping entry")
        return copy.deepcopy(mapping[key])
    return _description_value(value, precision=precision)


def _resolver_arguments(
    call,
    precision,
    single_zone,
    *,
    selector=None,
    scope=None,
    parameter_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if selector is None:
        args = [_description_value(arg, precision=precision) for arg in call["args"]]
        kwargs = {item["name"]: _description_value(item["value"], precision=precision) for item in call["keywords"]}
    else:
        args = [_case_description_value(arg, precision=precision, selector=selector, scope=scope) for arg in call["args"]]
        kwargs = {
            item["name"]: _case_description_value(item["value"], precision=precision, selector=selector, scope=scope)
            for item in call["keywords"]
        }
    if len(kwargs) != len(call["keywords"]):
        raise ValueError("duplicate tolerance resolver keywords")
    parameter_names = parameter_names or (("zone", "level") if single_zone else ("zones", "level"))
    if (
        len(args) > len(parameter_names)
        or set(kwargs) - set(parameter_names)
        or any(key in kwargs for key in parameter_names[: len(args)])
    ):
        raise ValueError("unsupported/duplicate tolerance resolver arguments")
    actual = dict(zip(parameter_names, args)) | kwargs
    if single_zone:
        if type(actual.get("zone")) is not str:
            raise ValueError("single-zone resolver requires one explicit zone name")
        actual = {"zones": [actual.pop("zone")], **actual}
    return actual


def _test_scope(pytest_node: str | None) -> str:
    if not isinstance(pytest_node, str) or "::" not in pytest_node or "[" in pytest_node:
        raise ValueError("per-case zones require the owning pytest node")
    return ".".join(pytest_node.split("::")[1:])


def _case_universe(entries, cases, selector) -> list[dict[str, Any]]:
    """Static: every collected case is covered exactly once and no entry is extra."""
    _validate_case_zones(entries)
    if cases is not None:
        if not isinstance(cases, (list, tuple)) or not cases:
            raise ValueError("per-case zones require the node's collected parameter cases")
        selectors = [case.get("selector") if isinstance(case, dict) else None for case in cases]
        for current in selectors:
            if sum(_case_selected(entry["selector"], current) for entry in entries) != 1:
                raise ValueError("a collected parameter case is not covered by exactly one per-case zone entry")
        if any(not any(_case_selected(entry["selector"], current) for current in selectors) for entry in entries):
            raise ValueError("a per-case zone entry matches no collected parameter case")
        return selectors
    if selector is not None:
        case_zones_for(entries, selector)
        return [selector]
    raise ValueError("per-case zones require the node's collected parameter cases")


def case_zone_coverage_errors(checks: Any, cases) -> list[str]:
    """Contract-load check: every per-case zone list covers the collected cases exactly once."""
    errors = []
    for identity, check in (checks or {}).items() if isinstance(checks, dict) else ():
        if isinstance(check, dict) and "tree_cases" in check:
            try:
                exact_tree_requirements.validate_tree_cases(check["tree_cases"], cases=cases)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{identity}: invalid tree case coverage: {exc}")
        precision = check.get("precision") if isinstance(check, dict) else None
        for mode, disposition in precision.items() if isinstance(precision, dict) else ():
            if isinstance(disposition, dict) and disposition.get("policy") == spacing_requirements.POLICY:
                errors.extend(spacing_requirements.coverage_errors(disposition, [case["selector"] for case in cases]))
            if isinstance(disposition, dict) and "case_zones" in disposition:
                try:
                    _case_universe(disposition["case_zones"], cases, None)
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"{identity}: {mode}: {exc}")
    return errors


def _validate_empty_cases(cases):
    """An empty spin is an explicit case/shape obligation, never a generic waiver."""
    if not isinstance(cases, list) or not cases:
        raise ValueError("empty_cases requires a nonempty explicit case list")
    selectors = set()
    names = None
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"selector", "shape"}:
            raise ValueError("an empty case requires exactly selector and shape")
        selector, shape = case["selector"], case["shape"]
        if (
            not isinstance(selector, dict)
            or not selector
            or any(type(key) is not str or not key.isidentifier() or keyword.iskeyword(key) for key in selector)
            or any(type(value) not in (str, int) or (type(value) is str and not value) for value in selector.values())
        ):
            raise ValueError("empty-case selectors require named string or integer parameters")
        identity = tuple(sorted(selector.items()))
        if identity in selectors or (names is not None and set(selector) != names):
            raise ValueError("empty-case selectors are duplicated or have different parameter names")
        selectors.add(identity)
        names = set(selector)
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(size) is not int or size < 0 for size in shape)
            or 0 not in shape
        ):
            raise ValueError("an empty case requires an exact zero-size shape")


def declared_empty_shape(check, selector):
    """Resolve only an explicitly declared empty case; do not infer it from data."""
    for case in check.get("empty_cases", ()):
        if isinstance(selector, dict) and all(
            type(value) in (str, int) and type(selector.get(name)) is type(value) and selector[name] == value
            for name, value in case["selector"].items()
        ):
            return case["shape"]
    return None


def _imported_tolerance(value, *, root, seen=frozenset()):
    """Resolve a reviewed test-module constant alias without importing its code.

    The lexical source index rejects rebinding, unresolved branches and cycles.
    Only repository test modules are followed; runtime invocation values must
    still equal the independently resolved policy in the receipt validator.
    """
    if value.get("kind") == "unresolved":
        raise ValueError("unresolved imported tolerance alias")
    if value.get("kind") != "imported-reference":
        return value
    name = value.get("name", "")
    if not name.startswith("tests.") or name in seen:
        raise ValueError("unsupported or cyclic imported tolerance alias")
    module, _, symbol = name.rpartition(".")
    path = root / (module.replace(".", "/") + ".py")
    from gdd.engine.tolerance_provenance import ToleranceSourceIndex

    tree = ast.parse(path.read_text(encoding="utf-8"))
    index = ToleranceSourceIndex(tree, path.relative_to(root).as_posix())
    writes = index.writes.get(id(tree), {}).get(symbol, [])
    if len(writes) != 1:
        raise ValueError("imported tolerance must have one source definition")
    resolved = index.expression(ast.Name(id=symbol, ctx=ast.Load()), scope=tree, before=ast.Pass())
    return _imported_tolerance(resolved, root=root, seen=seen | {name})


def requirement_errors(
    checks: Any,
    observations,
    modes,
    *,
    root: Path = ROOT,
    method: str | None = None,
    pytest_node: str | None = None,
    cases=None,
    selector: dict[str, Any] | None = None,
) -> list[str]:
    """Match authored site/policy requirements to source, never derive approval from it.

    ``cases`` (the node's collected concrete cases) is required to validate a
    ``case_zones`` requirement statically; a runtime caller may pass only the
    invocation's ``selector``, which then must be covered by exactly one entry.
    """
    declaration = load_roles(root).get(pytest_node)
    if declaration is not None:
        errors = role_errors(
            declaration,
            observations,
            checks,
            {row["id"]: row for row in _verdict_registry(root)["verdict_methods"]},
        )
        if errors:
            return errors
    if checks is None:
        return []  # Unmigrated, not approved; callers/renderers must preserve this distinction.
    if not isinstance(checks, dict) or not checks:
        return ["executable_checks must be a nonempty mapping"]
    sites = {}
    for observation in observations:
        for comparison in observation.get("tolerance_provenance", {}).get("comparisons", ()):
            sites.setdefault(comparison["site"], []).append((comparison, observation))
    errors = []
    used = set()
    for identity, check in checks.items():
        try:
            check_method = comparison_method(pytest_node, identity, method, root=root)
            fields = {"site", "comparator", "precision", "actual", "reference"}
            if (
                not isinstance(identity, str)
                or not isinstance(check, dict)
                or not fields <= set(check) <= fields | {"iterations", "empty_cases", "tree_cases", "component_loop"}
            ):
                raise ValueError(
                    "comparison check requires site, comparator, precision, both operands and known optional fields"
                )
            if "empty_cases" in check:
                _validate_empty_cases(check["empty_cases"])
            if "tree_cases" in check:
                exact_tree_requirements.validate_tree_cases(check["tree_cases"], cases=cases, selector=selector)
                if "empty_cases" in check or "iterations" in check:
                    raise ValueError("tree paths cannot be replaced by empty-case or iteration exceptions")
            if "iterations" in check:
                iterations = check["iterations"]
                if not isinstance(iterations, dict) or set(iterations) != {"variable", "required_values"}:
                    raise ValueError("iterations require exactly variable and required_values")
                variable, values = iterations["variable"], iterations["required_values"]
                if (
                    not isinstance(variable, str)
                    or not variable.isidentifier()
                    or keyword.iskeyword(variable)
                    or not isinstance(values, list)
                    or not values
                    or any(not isinstance(value, str) or not value for value in values)
                    or len(set(values)) != len(values)
                ):
                    raise ValueError("iterations require a local variable and nonempty distinct string values")
            site = check["site"]
            if not isinstance(site, str) or site in used or len(sites.get(site, ())) != 1:
                raise ValueError("comparison site is missing, duplicate or ambiguous")
            used.add(site)
            source, observation = sites[site][0]
            for operand in ("actual", "reference"):
                description = check[operand]
                if (
                    not isinstance(description, dict)
                    or set(description) != {"role", "expression"}
                    or not isinstance(description["role"], str)
                    or not description["role"].strip()
                    or description["expression"] != observation.get(operand)
                ):
                    raise ValueError("comparison operand/role differs from requirement")
            if check["comparator"] == inequality_receipts.COMPARATOR:
                # A hand-written inequality is certified only in a recognized
                # source form; its bound terms are the described arguments.
                if (
                    source["comparator"] != inequality_receipts.SOURCE_COMPARATOR
                    or source.get("form") not in inequality_receipts.FORMS
                    or set(source.get("arguments", ())) != {"atol", "rtol"}
                ):
                    raise ValueError("unsupported or substituted comparator")
                if "empty_cases" in check:
                    raise ValueError("inequality receipts require nonempty operands")
            elif check["comparator"] not in {
                "numpy.testing.assert_allclose",
                "numpy.allclose",
                "numpy.isclose",
                "numpy.testing.assert_array_equal",
                *_DECIMAL_COMPARATORS,
            } or source["comparator"] != {
                "kind": "imported-reference",
                "name": check["comparator"],
            }:
                raise ValueError("unsupported or substituted comparator")
            if not isinstance(check["precision"], dict) or set(check["precision"]) != set(modes):
                raise ValueError("comparison precision requirements do not match the node's required modes")
            for precision, disposition in check["precision"].items():
                spacing = isinstance(disposition, dict) and disposition.get("policy") == spacing_requirements.POLICY
                tree_exact = isinstance(disposition, dict) and disposition.get("policy") == exact_tree_requirements.POLICY
                zone_zero = isinstance(disposition, dict) and disposition.get("policy") == zone_zero_requirements.POLICY
                if "component_loop" in check and not zone_zero:
                    raise ValueError("component_loop is reserved for the reviewed zone/zero policy")
                if tree_exact != ("tree_cases" in check):
                    raise ValueError("source array equality requires its finite tree inventory and no other policy")
                per_case = isinstance(disposition, dict) and "case_zones" in disposition
                if zone_zero:
                    if any(name in check for name in ("iterations", "empty_cases", "tree_cases")):
                        raise ValueError("zone/zero component inventory cannot use alternate iteration or empty exceptions")
                    definition = next(
                        row
                        for row in _verdict_registry(root)["comparison_policies"]
                        if row["id"] == zone_zero_requirements.POLICY
                    )
                    zone_zero_requirements.validate_scope(
                        definition, check, disposition, pytest_node, check_method, precision, cases, selector
                    )
                    for row in zone_zero_requirements.components(disposition):
                        resolve_requirement(disposition, precision, root=root, component_index=row["index"])
                elif spacing:
                    if "empty_cases" in check:
                        raise ValueError("spacing comparisons require nonempty operands")
                    if cases is not None:
                        universe = [case["selector"] for case in cases]
                        coverage = spacing_requirements.coverage_errors(disposition, universe)
                        if coverage:
                            raise ValueError("; ".join(coverage))
                    elif selector is not None:
                        universe = [selector]
                    else:
                        raise ValueError("spacing comparisons require the concrete case universe")
                    for current in universe:
                        resolve_requirement(disposition, precision, root=root, selector=current)
                elif per_case:
                    if "empty_cases" in check:
                        raise ValueError("per-case zones require nonempty comparison operands")
                    universe = _case_universe(disposition["case_zones"], cases, selector)
                    for entry in disposition["case_zones"]:
                        resolve_requirement(disposition, precision, root=root, selector=entry["selector"])
                else:
                    resolve_requirement(disposition, precision, root=root)
                if disposition["policy"] == _CENTRAL_PAIR_POLICY:
                    definition = next(
                        row for row in _verdict_registry(root)["comparison_policies"] if row["id"] == _CENTRAL_PAIR_POLICY
                    )
                    required_scope = {
                        "site": site,
                        "pytest_node": pytest_node,
                        "verdict_method": check_method,
                        "comparator": check["comparator"],
                        "precision": precision,
                        "pair": disposition["pair"],
                    }
                    if required_scope not in _central_pair_scopes(definition):
                        raise ValueError("central pair is outside its reviewed comparison binding")
                if disposition["policy"] in {
                    "CMP-PW-PROJECTION-FP64-001",
                    "CMP-DTN-EN-CUSP-CONSTRAINT-001",
                    "CMP-TREXIO-STORAGE-FP64-001",
                    "CMP-LM-HOST-FP64-001",
                    _DECIMAL_POLICY,
                    exact_tree_requirements.POLICY,
                    spacing_requirements.POLICY,
                }:
                    label = {
                        "CMP-PW-PROJECTION-FP64-001": "PW projection residual",
                        "CMP-DTN-EN-CUSP-CONSTRAINT-001": "DTN cusp",
                        "CMP-TREXIO-STORAGE-FP64-001": "TREXIO storage",
                        "CMP-LM-HOST-FP64-001": "host LM",
                        _DECIMAL_POLICY: "historical TurboRVB decimal",
                        exact_tree_requirements.POLICY: "source array equality",
                        spacing_requirements.POLICY: "source scaled spacing",
                    }[disposition["policy"]]
                    policy = next(
                        row for row in _verdict_registry(root)["comparison_policies"] if row["id"] == disposition["policy"]
                    )
                    allowed = policy.get("allowed-comparison-sites")
                    if (
                        not isinstance(allowed, list)
                        or not allowed
                        or any(not isinstance(item, str) for item in allowed)
                        or len(set(allowed)) != len(allowed)
                        or site not in allowed
                    ):
                        raise ValueError(f"{label} policy is restricted to its reviewed comparison sites")
                    source_path, qualified = site.split("::", 1)
                    owner = source_path + "::" + qualified.split("#", 1)[0].replace(".", "::")
                    consumers = (
                        policy.get("allowed-evidence-nodes")
                        if disposition["policy"]
                        in {
                            "CMP-LM-HOST-FP64-001",
                            "CMP-PW-PROJECTION-FP64-001",
                            _DECIMAL_POLICY,
                            exact_tree_requirements.POLICY,
                            spacing_requirements.POLICY,
                        }
                        else [owner]
                    )
                    if (
                        not isinstance(consumers, list)
                        or not consumers
                        or any(not isinstance(item, str) or "::" not in item for item in consumers)
                        or len(set(consumers)) != len(consumers)
                        or (pytest_node is not None and pytest_node not in consumers)
                    ):
                        raise ValueError(f"{label} policy is restricted to its reviewed evidence node")
                if check_method is not None:
                    verdict = next(row for row in _verdict_registry(root)["verdict_methods"] if row["id"] == check_method)
                    if "allowed-comparison-sites" in verdict and site not in verdict["allowed-comparison-sites"]:
                        raise ValueError("comparison is outside the verdict method's reviewed sites")
                    if "allowed-precision-modes" in verdict:
                        allowed_modes = verdict["allowed-precision-modes"]
                        if (
                            not isinstance(allowed_modes, list)
                            or not allowed_modes
                            or any(item not in {"full", "mixed"} for item in allowed_modes)
                            or len(set(allowed_modes)) != len(allowed_modes)
                            or precision not in allowed_modes
                        ):
                            raise ValueError("comparison is outside the verdict method's reviewed precision modes")
                    if disposition["policy"] not in verdict["comparison_policies"]:
                        raise ValueError("comparison policy is not allowed for this verdict method")
                decimal = disposition["policy"] == _DECIMAL_POLICY
                if decimal != (check["comparator"] in _DECIMAL_COMPARATORS):
                    raise ValueError("decimal semantics require the matching decimal comparator")
                if decimal:
                    if (
                        source.get("arguments") != {"decimal": {"kind": "central-reference", "name": disposition["symbol"]}}
                        or source.get("forwarded_keywords")
                        or source.get("unresolved_signature")
                        or "empty_cases" in check
                    ):
                        raise ValueError("decimal comparison source must use the declared central symbol and nonempty operands")
                    continue
                exact = disposition["policy"] in {_EXACT_POLICY, exact_tree_requirements.POLICY}
                if exact != (check["comparator"] == "numpy.testing.assert_array_equal"):
                    raise ValueError("comparison policy requires a different comparator")
                if exact:
                    if source.get("arguments") != {} or source.get("forwarded_keywords") or source.get("unresolved_signature"):
                        raise ValueError("exact comparison cannot have tolerances or unresolved forwarded arguments")
                    if "empty_cases" in check:
                        raise ValueError("exact-value adapter requires nonempty operands")
                    if tree_exact:
                        source_text = (root / site.split("::", 1)[0]).read_text(encoding="utf-8")
                        signature_errors = exact_tree_requirements.source_signature_errors(source_text, source)
                        if signature_errors:
                            raise ValueError("; ".join(signature_errors))
                    continue
                if zone_zero:
                    source_text = (root / site.split("::", 1)[0]).read_text(encoding="utf-8")
                    zone_zero_requirements.validate_source(source, check, disposition, source_text)
                    continue
                if spacing:
                    if (
                        check["comparator"] != "numpy.testing.assert_allclose"
                        or source.get("forwarded_keywords")
                        or source.get("unresolved_signature")
                    ):
                        raise ValueError("spacing requires the explicit authored allclose call")
                    policy = next(
                        row
                        for row in _verdict_registry(root)["comparison_policies"]
                        if row["id"] == spacing_requirements.POLICY
                    )
                    for current in universe:
                        spacing_requirements.validate_source(
                            source["arguments"], disposition, policy, current, _test_scope(pytest_node)
                        )
                    continue
                if disposition["policy"] in _SYMBOL_POLICIES or disposition["policy"] == _CENTRAL_PAIR_POLICY:
                    symbols = named_tolerance_symbols(disposition["policy"], root=root, pair_id=disposition.get("pair"))
                    for component, symbol in symbols.items():
                        if source["arguments"][component] != {"kind": "central-reference", "name": symbol}:
                            raise ValueError("comparison tolerance symbol differs from requirement")
                    continue
                if disposition["policy"] == _MIXED_FIDELITY_POLICY:
                    for component, index in (("atol", 0), ("rtol", 1)):
                        value = _imported_tolerance(source["arguments"][component], root=root)
                        if value.get("kind") != "item" or value.get("index") != index:
                            raise ValueError("mixed-fidelity tolerance is not the declared resolver component")
                        call = value["value"]
                        if call.get("kind") != "central-resolver-call" or call.get("function") != {
                            "kind": "central-reference",
                            "name": "jqmc._setting.get_tolerance_mixed_fidelity",
                        }:
                            raise ValueError("mixed-fidelity tolerance resolver differs from requirement")
                        actual = _resolver_arguments(
                            call,
                            precision,
                            False,
                            parameter_names=("quantity_class",),
                        )
                        if actual != {"quantity_class": disposition["quantity_class"]}:
                            raise ValueError("mixed-fidelity quantity class differs from requirement")
                    continue
                table_name, level, resolver = _POLICIES[disposition["policy"]]
                for component, index in (("atol", 0), ("rtol", 1)):
                    value = _imported_tolerance(source["arguments"][component], root=root)
                    if value.get("kind") != "item" or type(value.get("index")) is not int or value["index"] != index:
                        raise ValueError("comparison tolerance is not the declared resolver component")
                    call = value["value"]
                    # A single-zone lookup is exactly the one-element zone
                    # aggregation of the same central table, not a new budget.
                    # Do not apply this equivalence to the distinct Rust table.
                    single_zone = table_name == "_TOLERANCE" and call.get("function") == {
                        "kind": "central-reference",
                        "name": "jqmc._precision.get_tolerance",
                    }
                    if call.get("kind") != "central-resolver-call" or (
                        not single_zone and call["function"] != {"kind": "central-reference", "name": resolver}
                    ):
                        raise ValueError("comparison tolerance resolver differs from requirement")
                    if not per_case:
                        actual = _resolver_arguments(call, precision, single_zone)
                        if actual != {"zones": disposition["zones"], "level": level}:
                            raise ValueError("comparison zones/level differ from requirement")
                        continue
                    # Evaluate the source's zone argument separately for every
                    # covered case and require that case's own declared list.
                    scope = _test_scope(pytest_node)
                    for current in universe:
                        actual = _resolver_arguments(call, precision, single_zone, selector=current, scope=scope)
                        expected = case_zones_for(disposition["case_zones"], current)
                        if actual != {"zones": expected, "level": level}:
                            raise ValueError("comparison zones/level differ from the per-case requirement")
        except (KeyError, TypeError, ValueError, StopIteration, SyntaxError, OSError) as exc:
            errors.append(f"{identity}: invalid executable comparison requirement: {exc}")
    return errors


@cache
def _source_repeats_comparison(source_text: str, line: int, column: int) -> bool:
    """Allow repetition only inside the comparison's own lexical loop scope.

    Calling a helper repeatedly creates separate invocation receipts; a loop
    outside that helper must not authorize duplicate attempts within one call.
    Unknown source coordinates cannot establish a repetition exception.
    """
    tree = ast.parse(source_text)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    matches = [
        node for node in ast.walk(tree) if (getattr(node, "lineno", None), getattr(node, "col_offset", None)) == (line, column)
    ]
    if not matches:
        raise ValueError("comparison location is absent from source")
    repeated = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    for node in matches:
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                break
            if isinstance(node, repeated):
                return True
    return False


def _comparison_source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _repeated_attempts_allowed(spec, site: str, root: Path) -> bool:
    source = next(row for row in spec["tolerance_provenance"]["comparisons"] if row["site"] == site)
    location = source.get("location", {})
    try:
        line, column = location["line"], location["column"]
        if type(line) is not int or type(column) is not int:
            raise ValueError("comparison location is not an integer coordinate")
        return _source_repeats_comparison(_comparison_source_text(root / site.split("::", 1)[0]), line, column)
    except (KeyError, TypeError, OSError, SyntaxError, UnicodeError) as exc:
        raise ValueError("comparison source cannot establish repeat cardinality") from exc


def runtime_requirement_errors(
    checks,
    observations,
    invocations,
    precision: str,
    *,
    root: Path = ROOT,
    selector=None,
    pytest_node: str | None = None,
    method: str | None = None,
) -> list[str]:
    if checks is None and pytest_node in load_roles(root):
        return ["declared evidence roles require their executable checks; runtime fallback is forbidden"]
    if checks is None:
        return []
    if not isinstance(checks, dict) or not checks:
        return ["executable_checks must be a nonempty mapping"]
    first = next(iter(checks.values()))
    modes = tuple(first.get("precision", ())) if isinstance(first, dict) else ()
    # A per-case zone requirement is re-validated for THIS invocation's case.
    errors = requirement_errors(
        checks,
        observations,
        modes,
        root=root,
        method=method,
        pytest_node=pytest_node,
        selector=selector if isinstance(selector, dict) and selector else None,
    )
    if precision not in modes:
        errors.append("runtime precision has no executable comparison requirement")
    if errors:
        return errors
    site_specs = {
        row["site"]: observation
        for observation in observations
        for row in observation.get("tolerance_provenance", {}).get("comparisons", ())
    }
    required = {check["site"]: check for check in checks.values()}

    def executed_ids(record):
        result = set(record.get("executed", ()))
        for branch in record.get("branches", ()):
            result.update(executed_ids(branch))
        return result

    if not isinstance(invocations, list) or not invocations:
        return ["comparison invocations are missing for executable requirements"]
    # Expected-error procedures may contain a passing primary and fail
    # elsewhere. Such an exception cannot certify that numerical primary.
    observation_by_id = {row["id"]: row for row in observations}
    primary_functions = {}
    procedure_calls = {}
    for observation in observations:
        function = observation.get("required_in_invocation") or observation.get("conditional_execution", {}).get("function")
        if "helper_call" in observation:
            procedure_calls.setdefault(function, set()).add(observation["helper_call"]["callee"])
    for site in required:
        spec = site_specs[site]
        function = spec.get("required_in_invocation") or spec.get("conditional_execution", {}).get("function")
        primary_functions.setdefault(function, set()).add(spec["id"])

    def procedure_primaries(function):
        pending, seen, primaries = [function], set(), set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            primaries.update(primary_functions.get(current, ()))
            pending.extend(procedure_calls.get(current, ()))
        return primaries

    def linked_calls(record):
        yield from record.get("calls", ())
        for branch in record.get("branches", ()):
            yield from linked_calls(branch)

    def failed_primary(index, primaries):
        # Child receipts complete before their caller. Follow only the actual
        # failing call and its reciprocal link; unrelated failures do not count.
        while type(index) is int and 0 <= index < len(invocations):
            record = invocations[index]
            if not isinstance(record, dict) or not isinstance(record.get("exception"), dict):
                return False
            failed = record["exception"].get("failed")
            if failed in primaries:
                return True
            call = next((row for row in linked_calls(record) if row.get("call") == failed), {})
            child = call.get("invocation")
            helper = observation_by_id.get(failed, {}).get("helper_call", {})
            if (
                type(child) is not int
                or not 0 <= child < index
                or not isinstance(invocations[child], dict)
                or invocations[child].get("scope") != helper.get("callee")
                or invocations[child].get("caller") != {"call": failed, "invocation": index}
            ):
                return False
            index = child
        return False

    accepted_empty = False
    nonempty_numerical = False
    for record_index, record in enumerate(invocations):
        if not isinstance(record, dict):
            errors.append("invalid comparison invocation for tolerance requirement")
            continue
        parent = record.get("caller")
        parent_call = parent.get("call") if isinstance(parent, dict) else None
        caller = observation_by_id.get(parent_call, {}).get("helper_call", {})
        if caller.get("expected_error") is not None:
            primaries = procedure_primaries(caller["callee"])
            if primaries and not failed_primary(record_index, primaries):
                errors.append("expected-error invocation did not fail at an authored primary comparison")
        executed = executed_ids(record)
        attempts = record.get("tolerance_calls", [])
        if not isinstance(attempts, list):
            errors.append("invalid tolerance call receipts")
            continue
        # Negative evidence: the comparison an exceptional invocation failed
        # at, or a comparator whose pytest.raises context accepted its
        # rejection. Its last attempt must have rejected with the authored
        # comparator and tolerance; it is never positive numerical evidence.
        exception = record.get("exception")
        negative_ids = {exception.get("failed")} if isinstance(exception, dict) else set()
        negative_ids |= {
            observation["id"]
            for observation in observations
            if observation.get("expected_rejection") is not None and observation["expected_rejection"] in executed
        }
        for site, check in required.items():
            spec = site_specs[site]
            selected = [row for row in attempts if isinstance(row, dict) and row.get("site") == site]
            iterations = check.get("iterations")
            seen_values = set()
            zone_zero = check["precision"][precision]["policy"] == zone_zero_requirements.POLICY
            seen_components = []
            if spec["id"] in executed and not selected:
                errors.append(f"{site}: executed comparison has no actual tolerance call")
            if len(selected) > 1:
                try:
                    if not _repeated_attempts_allowed(spec, site, root):
                        errors.append(f"{site}: non-loop comparison must have exactly one attempt per invocation")
                except ValueError as exc:
                    errors.append(f"{site}: cannot validate repeated comparison attempts: {exc}")
            for attempt in selected:
                calls = attempt.get("calls")
                negative = attempt is selected[-1] and spec["id"] in negative_ids
                if attempt.get("observation_id") != spec["id"] or (spec["id"] not in executed and not negative):
                    errors.append(f"{site}: tolerance call belongs to a different/unfinished comparison")
                if not isinstance(calls, list) or len(calls) != 1:
                    errors.append(f"{site}: comparison statement must call the actual comparator exactly once")
                    continue
                call = calls[0]
                if not isinstance(call, dict):
                    errors.append(f"{site}: invalid actual comparison call")
                    continue
                if negative:
                    # A Python assert has no comparator return frame: the
                    # captured inequality remains "evaluated". Its actual
                    # AssertionError identifies this failed site, and replay
                    # must independently show that the finite bound failed.
                    rejected = call.get("outcome") == "failed"
                    if check["comparator"] == inequality_receipts.COMPARATOR:
                        rejected = (
                            call.get("outcome") == inequality_receipts.EVALUATED
                            and call.get("operand_check") == "value-mismatch"
                        )
                    if call.get("comparator") != check["comparator"] or not rejected:
                        errors.append(f"{site}: negative comparison did not reject through the authored comparator")
                    elif check["precision"][precision]["policy"] not in {
                        _EXACT_POLICY,
                        exact_tree_requirements.POLICY,
                        spacing_requirements.POLICY,
                        zone_zero_requirements.POLICY,
                    }:
                        try:
                            expected = resolve_requirement(
                                check["precision"][precision], precision, root=root, selector=selector
                            )
                            actual = call["arguments"]
                            if any(
                                type(actual[name]) not in (int, float)
                                or not math.isfinite(actual[name])
                                or actual[name] != value
                                for name, value in expected.items()
                            ):
                                errors.append(f"{site}: actual tolerance arguments differ from the authored requirement")
                        except (KeyError, TypeError, ValueError):
                            errors.append(f"{site}: invalid actual comparison arguments")
                    continue
                if iterations is not None:
                    observed = call.get("iteration")
                    if (
                        not isinstance(observed, dict)
                        or set(observed) != {"variable", "value"}
                        or observed["variable"] != iterations["variable"]
                        or not isinstance(observed["value"], str)
                        or observed["value"] not in iterations["required_values"]
                    ):
                        errors.append(f"{site}: invalid or undeclared comparison iteration value")
                    else:
                        seen_values.add(observed["value"])
                expected_outcome = (
                    inequality_receipts.EVALUATED if check["comparator"] == inequality_receipts.COMPARATOR else "normal"
                )
                if call.get("outcome") != expected_outcome or call.get("comparator") != check["comparator"]:
                    errors.append(f"{site}: actual comparator did not complete normally")
                try:
                    component_index = None
                    if zone_zero:
                        component_index = zone_zero_requirements.captured_index(
                            call.get("zone_zero_component"), check, check["precision"][precision]
                        )
                        seen_components.append(component_index)
                    expected = resolve_requirement(
                        check["precision"][precision],
                        precision,
                        root=root,
                        selector=selector,
                        component_index=component_index,
                    )
                    actual = call["arguments"]
                    if check["precision"][precision]["policy"] == spacing_requirements.POLICY:
                        policy = next(
                            row
                            for row in _verdict_registry(root)["comparison_policies"]
                            if row["id"] == spacing_requirements.POLICY
                        )
                        errors.extend(
                            spacing_requirements.runtime_errors(
                                call.get("spacing_basis"),
                                check["precision"][precision],
                                policy,
                                selector,
                                _central_symbol_reader(root),
                            )
                        )
                    if check["precision"][precision]["policy"] == exact_tree_requirements.POLICY:
                        errors.extend(exact_tree_requirements.tree_call_errors(call, check["tree_cases"], selector=selector))
                        if call.get("operand_check") == "passed":
                            nonempty_numerical = True
                        continue
                    if check["precision"][precision]["policy"] == _EXACT_POLICY:
                        if actual != {}:
                            errors.append(f"{site}: exact comparison must not report tolerance arguments")
                        shapes, dtypes = call.get("operand_shapes"), call.get("operand_dtypes")
                        if (
                            not isinstance(shapes, dict)
                            or set(shapes) != {"actual", "reference"}
                            or any(
                                not isinstance(dims, list) or any(type(size) is not int or size < 0 for size in dims)
                                for dims in shapes.values()
                            )
                            or shapes["actual"] != shapes["reference"]
                            or math.prod(shapes["actual"]) == 0
                        ):
                            errors.append(f"{site}: exact comparison requires equal nonempty operand shapes")
                        if (
                            not isinstance(dtypes, dict)
                            or set(dtypes) != {"actual", "reference"}
                            or any(not isinstance(dtype, str) or not dtype for dtype in dtypes.values())
                            or dtypes["actual"] != dtypes["reference"]
                        ):
                            errors.append(f"{site}: exact comparison requires identical operand dtypes")
                        if call.get("operand_check") != "passed":
                            errors.append(f"{site}: comparison operands fail the registered exact-value semantics")
                        else:
                            nonempty_numerical = True
                        continue
                    if any(
                        type(actual[name]) not in (int, float) or not math.isfinite(actual[name]) or actual[name] != value
                        for name, value in expected.items()
                    ):
                        errors.append(f"{site}: actual tolerance arguments differ from the authored requirement")
                    shape = declared_empty_shape(check, selector)
                    operand_status = call.get("operand_check")
                    if "empty_cases" in check:
                        if not isinstance(selector, dict) or any(
                            name not in selector for case in check["empty_cases"] for name in case["selector"]
                        ):
                            errors.append(f"{site}: empty-case comparison requires a concrete parameter selector")
                        recorded = call.get("operand_shapes")
                        if (
                            not isinstance(recorded, dict)
                            or set(recorded) != {"actual", "reference"}
                            or any(
                                not isinstance(dims, list) or any(type(size) is not int or size < 0 for size in dims)
                                for dims in recorded.values()
                            )
                            or recorded["actual"] != recorded["reference"]
                        ):
                            errors.append(f"{site}: missing or inconsistent actual operand shapes")
                        elif shape is not None:
                            if recorded["actual"] != shape or operand_status != "empty":
                                errors.append(f"{site}: declared empty case has different operand shape or status")
                            else:
                                accepted_empty = True
                        elif math.prod(recorded["actual"]) == 0:
                            errors.append(f"{site}: empty operands are not declared for this case")
                    if operand_status != ("empty" if shape is not None else "passed"):
                        errors.append(f"{site}: comparison operands fail the registered numerical semantics")
                    if operand_status == "passed" and call.get("outcome") == "normal":
                        nonempty_numerical = True
                except (KeyError, TypeError, ValueError):
                    errors.append(f"{site}: invalid actual comparison arguments")
            # An exceptional invocation (negative evidence) stops at its
            # failing iteration; completeness binds normal invocations only.
            if (
                iterations is not None
                and spec["id"] in executed
                and not isinstance(exception, dict)
                and seen_values != set(iterations["required_values"])
            ):
                errors.append(f"{site}: required comparison iteration values are missing in this invocation")
            if zone_zero and seen_components != list(range(len(check["precision"][precision]["components"]))):
                errors.append(f"{site}: component comparisons must occur exactly once in their declared order")
        if any(not isinstance(row, dict) or row.get("site") not in required for row in attempts):
            errors.append("tolerance receipt contains an undeclared comparison site")
    for check in required.values():
        if "tree_cases" in check:
            errors.extend(exact_tree_requirements.tree_coverage_errors(check, invocations, selector=selector))
    if accepted_empty and not nonempty_numerical:
        errors.append("empty comparisons cannot replace a nonempty numerical verdict in the same test")
    return errors
