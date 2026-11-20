#!/usr/bin/env python3
"""Every authored numerical comparison is a live comparison at its bound site.

A guarantee node's ``executable_checks`` name the comparisons its verdict rests
on: the source site, the comparator and both operands. The derived case binding
records what the test source really asserts (``typed_observations``). Each
authored check must match exactly one recorded comparison site with the same
comparator and the same operands, and that comparison must be a positive
verdict, not a negative control expected to fail.

So a test edit that turns a bound comparison into a finiteness, shape or bare
bound assertion, deletes it, swaps its comparator or its operands, turns this
check red (issue 829: before it, such an edit kept every wired gate green).
A node with a numerical-agreement verdict method and no authored checks must
still record at least one agreement comparison: an equality or closeness
comparison of a computed operand with a reference, a bound whose two sides are
both non-constant (or that bounds a difference of two computed values), or an
asserted closeness helper called with two computed operands. A finiteness,
shape or one-sided constant bound (``x >= 1``) is not an agreement comparison.
The agreement methods are read from ``gdd/ssot/verdicts.yaml``: every method
that binds comparison policies to an oracle other than a reviewed expectation.
The derived bindings are refreshed from source before this check runs
(``gdd_sync --check`` and the preflight both rebuild them first).
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
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdd.engine.ledger import iter_registered_nodes, load_ledger  # noqa: E402

GUARANTEE_DIR = Path("gdd/ssot/guarantees")
VERDICTS = Path("gdd/ssot/verdicts.yaml")
COMPARISON_KINDS = frozenset({"tolerance-comparison", "exact-comparison"})
BOUND_KINDS = frozenset({"numerical-bound", "numerical-distinction"})


def agreement_methods(root: Path = ROOT) -> frozenset[str]:
    """Verdict methods whose oracle is an independent comparison target.

    A method that binds comparison policies claims a numerical comparison; one
    whose oracle is a reviewed expectation (``reviewed-*``) is a hand-written
    contract, identity or regression target instead, and is not an agreement
    with an independently computed reference.
    """
    document = yaml.safe_load((root / VERDICTS).read_text(encoding="utf-8"))
    return frozenset(
        method["id"]
        for method in document["verdict_methods"]
        if method.get("comparison_policies") and not str(method.get("oracle", "")).startswith("reviewed-")
    )


def _parse(expression: Any) -> ast.expr | None:
    if not isinstance(expression, str):
        return None
    try:
        return ast.parse(expression, mode="eval").body
    except SyntaxError:
        return None


def _constant(node: ast.expr | None) -> bool:
    while isinstance(node, ast.UnaryOp):
        node = node.operand
    return node is None or isinstance(node, ast.Constant)


def _difference(node: ast.expr) -> bool:
    return any(
        isinstance(part, ast.BinOp) and isinstance(part.op, ast.Sub) and not _constant(part.left) and not _constant(part.right)
        for part in ast.walk(node)
    )


def _two_sided_bound(node: ast.expr | None) -> bool:
    """A comparison relating two computed quantities, not a one-sided constant bound."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        node = node.operand
    if isinstance(node, ast.Call):
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        if name in {"all", "any"} and len(node.args) == 1 and not node.keywords:
            return _two_sided_bound(node.args[0])
        # np.isclose(a, b) and similar: two computed arguments.
        return name in {"isclose", "allclose"} and sum(not _constant(arg) for arg in node.args) >= 2
    if isinstance(node, ast.BoolOp):
        return all(_two_sided_bound(value) for value in node.values)
    if not isinstance(node, ast.Compare) or any(isinstance(op, (ast.Is, ast.IsNot, ast.In, ast.NotIn)) for op in node.ops):
        return False
    sides = [node.left, *node.comparators]
    return any(_difference(side) for side in sides) or not any(_constant(side) for side in sides)


def _asserted_helper_calls(observations: list[dict[str, Any]]) -> bool:
    """An asserted call of a helper whose truthy return is a closeness verdict, on two computed operands."""
    truthy = {
        row.get("required_in_invocation")
        for row in observations
        if row.get("truthy_return") and row.get("role") == "verdict" and _two_sided_bound(_parse(row.get("expression")))
    }
    asserted = {(row["source_location"]["line"], row["expression"]) for row in observations if "#assert-" in str(row.get("id"))}
    for row in observations:
        callee = (row.get("helper_call") or {}).get("callee")
        call = _parse(row.get("expression"))
        if (
            callee in truthy
            and (row["source_location"]["line"], row["expression"]) in asserted
            and isinstance(call, ast.Call)
            and sum(not _constant(arg) for arg in call.args) >= 2
        ):
            return True
    return False


def has_agreement_comparison(observations: Any) -> bool:
    rows = [row for row in observations if isinstance(row, dict)]
    for row in rows:
        kind = row.get("kind")
        positive = row.get("role") == "verdict" or row.get("expected_rejection")
        if kind in COMPARISON_KINDS and positive and row.get("reference") and not _constant(_parse(row.get("actual"))):
            return True
        if kind in BOUND_KINDS and _two_sided_bound(_parse(row.get("expression"))):
            return True
    return _asserted_helper_calls(rows)


def _same_expression(left: Any, right: Any) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    try:
        return ast.dump(ast.parse(left, mode="eval")) == ast.dump(ast.parse(right, mode="eval"))
    except SyntaxError:
        return False


def node_errors(
    pytest_node: str,
    checks: Any,
    observations: Any,
    verdict_method: str | None = None,
    methods: frozenset[str] = frozenset(),
) -> list[str]:
    """Match one node's authored checks to the comparisons its source records."""
    if checks is None:
        if verdict_method in methods and not has_agreement_comparison(observations):
            return [
                f"{pytest_node}: {verdict_method} verdict has no agreement comparison in the test source; "
                "a finiteness, shape or one-sided constant bound is not a numerical verdict"
            ]
        return []
    if not isinstance(checks, dict) or not checks:
        return [f"{pytest_node}: executable_checks must be a nonempty mapping"]
    sites: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for observation in observations:
        for comparison in (observation.get("tolerance_provenance") or {}).get("comparisons", ()):
            sites.setdefault(comparison.get("site"), []).append((comparison, observation))
    errors = []
    for identity, check in sorted(checks.items()):
        label = f"{pytest_node}: bound comparison {identity!r}"
        if not isinstance(check, dict) or not isinstance(check.get("site"), str):
            errors.append(f"{label} has no site")
            continue
        found = sites.get(check["site"], [])
        if len(found) != 1:
            errors.append(
                f"{label}: site {check['site']} is {'missing' if not found else 'ambiguous'} in the test source; "
                "a numerical verdict needs its comparison, not a finiteness or shape assertion"
            )
            continue
        comparison, observation = found[0]
        if (comparison.get("comparator") or {}).get("name") != check.get("comparator"):
            errors.append(f"{label}: the source comparator is not the declared {check.get('comparator')!r}")
        if observation.get("role") != "verdict" or observation.get("expected_rejection"):
            errors.append(f"{label}: the bound site is not a positive verdict comparison")
        for operand in ("actual", "reference"):
            declared = (check.get(operand) or {}).get("expression") if isinstance(check.get(operand), dict) else None
            if not _same_expression(declared, observation.get(operand)):
                errors.append(f"{label}: the {operand} operand in the source differs from the declared {declared!r}")
    return errors


def validation_errors(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    methods = agreement_methods(root)
    for path in sorted((root / GUARANTEE_DIR).glob("*.yaml")):
        document = load_ledger(path, root=root)
        for _guarantee, group, node in iter_registered_nodes(document, path):
            observations = node["case_binding"]["comparison"]["typed_observations"]
            errors.extend(
                node_errors(
                    node["pytest_node"], node.get("executable_checks"), observations, group.get("verdict_method"), methods
                )
            )
    return sorted(set(errors))


def main() -> int:
    try:
        errors = validation_errors()
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors = [str(exc)]
    if errors:
        print("GDD bound-comparison contract failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("GDD bound-comparison contract passed (every authored comparison is live at its bound site).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
