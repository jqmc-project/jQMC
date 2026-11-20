#!/usr/bin/env python3
"""Require the derivative matrix exclusions to be exactly the reviewed exclusion table.

``gdd/ssot/inventories/derivative_exclusions.yaml`` holds one row
per excluded cell pattern (reason, reviewer note, owning predicate, matched-cell count)
and one row per Jastrow realization alias. The generator reads that table. This gate
recomputes, over the full declared axis product of ``gdd/engine/derivative_matrix.py``:

* the cells the engineering predicates exclude (``_predicate_valid``) versus the cells
  the table excludes, in both directions;
* the cells the generator actually leaves out of ``_candidates()`` versus the table;
* per row: known axes and tokens, a named predicate that exists, a nonempty reason and
  reviewer note, at least one matched cell, the declared matched-cell count, and no
  cell matched by two rows;
* the Jastrow tokens whose builder realization (``_resolve``) is identical on every
  admitted cell that has a builder versus the declared realization aliases, in both
  directions.
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

import itertools
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdd.engine import derivative_exclusions as table  # noqa: E402
from gdd.engine import derivative_matrix as matrix  # noqa: E402


def axis_values() -> dict[str, list[str]]:
    return {
        "ao": matrix._ao_families(),
        "jas": matrix._jastrow_types(),
        "ecp": list(matrix._ECPS),
        "bc": list(matrix._BCS),
        "gem": list(matrix._GEMS),
    }


def axis_product() -> list[tuple[str, str, str, str, str]]:
    values = axis_values()
    return sorted(itertools.product(*(values[axis] for axis in table.AXES)))


def _rule_errors(rules: Iterable[table.ExcludedCellRule], cells: list[tuple]) -> list[str]:
    errors: list[str] = []
    values = axis_values()
    seen: set[str] = set()
    for rule in rules:
        if rule.rule_id in seen:
            errors.append(f"duplicate reviewed exclusion row {rule.rule_id}")
        seen.add(rule.rule_id)
        for axis, tokens in rule.pattern:
            if axis not in values:
                errors.append(f"reviewed exclusion row {rule.rule_id} names unknown axis {axis!r}")
                continue
            unknown = sorted(set(tokens) - set(values[axis]))
            if unknown:
                errors.append(f"reviewed exclusion row {rule.rule_id} names unknown {axis} token(s) {unknown}")
        if not callable(getattr(matrix, rule.predicate, None)):
            errors.append(f"reviewed exclusion row {rule.rule_id} names no derivative_matrix predicate {rule.predicate!r}")
        if not rule.reason.strip() or not rule.reviewer_note.strip():
            errors.append(f"reviewed exclusion row {rule.rule_id} has no reason or reviewer note")
        matched = sum(rule.matches(cell) for cell in cells)
        if matched == 0:
            errors.append(f"stale reviewed exclusion row {rule.rule_id}: it matches no cell of the declared axis product")
        elif matched != rule.expected_cells:
            errors.append(
                f"reviewed exclusion row {rule.rule_id} matches {matched} cells, expected_cells={rule.expected_cells}"
            )
    return errors


def _alias_errors(
    aliases: Iterable[table.RealizationAlias],
    cells: list[tuple],
    resolver: Callable[[tuple], object],
) -> list[str]:
    """Jastrow tokens that realize identically on every admitted cell must be exactly the aliases."""
    errors: list[str] = []
    groups: dict[tuple, dict[str, object]] = {}
    for ao, jas, ecp, bc, gem in cells:
        groups.setdefault((ao, ecp, bc, gem), {})[jas] = resolver((ao, jas, ecp, bc, gem))
    collapsing: set[frozenset[str]] = set()
    for first, second in itertools.combinations(matrix._jastrow_types(), 2):
        shared = [
            group
            for group in groups.values()
            if first in group and second in group and (group[first] is not None or group[second] is not None)
        ]
        if shared and all(group[first] == group[second] for group in shared):
            collapsing.add(frozenset((first, second)))
    declared: set[frozenset[str]] = set()
    for alias in aliases:
        if alias.axis != "jas":
            errors.append(f"realization alias {alias.alias_id} names axis {alias.axis!r}; only jas aliases are reviewed")
            continue
        if not alias.reason.strip() or not alias.reviewer_note.strip() or not alias.realization.strip():
            errors.append(f"realization alias {alias.alias_id} has no realization, reason or reviewer note")
        for pair in itertools.combinations(sorted(alias.tokens), 2):
            declared.add(frozenset(pair))
            if frozenset(pair) not in collapsing:
                errors.append(
                    f"stale realization alias {alias.alias_id}: Jastrow tokens {sorted(pair)} do not resolve to the "
                    "same builder realization"
                )
    for pair in sorted(collapsing - declared, key=sorted):
        errors.append(
            f"Jastrow tokens {sorted(pair)} resolve to the same builder realization on every cell but no reviewed "
            "realization alias declares it"
        )
    return errors


def validation_errors(
    rules: tuple[table.ExcludedCellRule, ...] | None = None,
    aliases: tuple[table.RealizationAlias, ...] | None = None,
    predicate_valid: Callable[[tuple], bool] | None = None,
    candidates: Callable[[], list[tuple]] | None = None,
    resolver: Callable[[tuple], object] | None = None,
) -> list[str]:
    rules = table.EXCLUDED_CELL_RULES if rules is None else rules
    aliases = table.REALIZATION_ALIASES if aliases is None else aliases
    predicate_valid = matrix._predicate_valid if predicate_valid is None else predicate_valid
    candidates = matrix._candidates if candidates is None else candidates
    resolver = (lambda cell: matrix._resolve(cell, require_sources=False)) if resolver is None else resolver

    cells = axis_product()
    errors = _rule_errors(rules, cells)
    for cell in cells:
        matched = table.matching_rules(cell, rules)
        if len(matched) > 1:
            errors.append(f"cell {cell} is matched by more than one reviewed exclusion row: {matched}")
        admitted = predicate_valid(cell)
        if not admitted and not matched:
            errors.append(
                f"cell {cell} is excluded by the derivative_matrix predicates but no reviewed exclusion row covers it"
            )
        if admitted and matched:
            errors.append(
                f"reviewed exclusion row(s) {matched} exclude cell {cell} that the derivative_matrix predicates admit"
            )
    generated = set(candidates())
    for cell in sorted(set(cells) - generated):
        if not table.matching_rules(cell, rules):
            errors.append(f"the generator excludes cell {cell} that no reviewed exclusion row covers")
    for cell in sorted(generated):
        if table.matching_rules(cell, rules):
            errors.append(f"the generator admits cell {cell} that a reviewed exclusion row excludes")
    unknown = sorted(generated - set(cells))
    if unknown:
        errors.append(f"the generator admits cells outside the declared axis product: {unknown[:5]}")
    errors.extend(_alias_errors(aliases, sorted(generated), resolver))
    return errors


def main() -> int:
    try:
        errors = validation_errors()
    except (ImportError, OSError, ValueError, TypeError) as exc:
        errors = [str(exc)]
    if errors:
        print("GDD derivative exclusion table check failed:", file=sys.stderr)
        for error in errors[:50]:
            print(f"  - {error}", file=sys.stderr)
        if len(errors) > 50:
            print(f"  ... {len(errors) - 50} more", file=sys.stderr)
        return 1
    excluded = sum(rule.expected_cells for rule in table.EXCLUDED_CELL_RULES)
    print(
        "GDD derivative exclusion table check passed "
        f"({len(axis_product())} declared cells, {excluded} excluded by {len(table.EXCLUDED_CELL_RULES)} reviewed rows, "
        f"{len(matrix._candidates())} admitted, {len(table.REALIZATION_ALIASES)} realization alias)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
