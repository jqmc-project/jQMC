"""Reviewed exclusion table of the derivative fixture matrix (GDD engine, requirement side).

The cells that ``derivatives.continuous_fixture_matrix.v1`` never enumerates are
decided by Python predicates in ``gdd/engine/derivative_matrix.py`` and justified by
a reviewed YAML data table of those exclusions, which this module loads. The generator reads it
(``derivative_matrix._valid`` admits a cell only when no row matches), and
``scripts/check_gdd_derivative_exclusions.py`` fails when

* the predicates in ``derivative_matrix`` exclude a cell no row covers,
* a row excludes a cell the predicates admit, matches no cell, overlaps another
  row, or matches a different number of cells than it declares,
* the generator's excluded set differs from the table, or
* two Jastrow tokens resolve to the same builder realization without a reviewed
  realization alias here (or an alias here no longer collapses).

The declarations live in ``gdd/ssot/inventories/derivative_exclusions.yaml``;
this module only validates and converts them to the existing typed interface.

This module imports nothing from jqmc, JAX, pytest or the tests.
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

from dataclasses import dataclass
from pathlib import Path

from gdd.engine.reviewed_inventory import load_reviewed_inventory


@dataclass(frozen=True)
class ExcludedCellRule:
    """One reviewed exclusion: every cell whose axis values lie in ``pattern`` is excluded.

    ``pattern`` maps an axis name to the tuple of tokens it matches; an axis that is
    absent matches every token. ``expected_cells`` counts the matched cells of the
    full declared axis product, so widening or narrowing a row is a visible edit.
    ``predicate`` names the ``derivative_matrix`` function whose docstring holds the
    engineering argument.
    """

    rule_id: str
    pattern: tuple[tuple[str, tuple[str, ...]], ...]
    expected_cells: int
    predicate: str
    reason: str
    reviewer_note: str

    def matches(self, cell: tuple[str, str, str, str, str]) -> bool:
        values = dict(zip(AXES, cell))
        return all(values[axis] in tokens for axis, tokens in self.pattern)


@dataclass(frozen=True)
class RealizationAlias:
    """Distinct universe tokens of one axis that the committed builders realize identically."""

    alias_id: str
    axis: str
    tokens: tuple[str, ...]
    realization: str
    reason: str
    reviewer_note: str


def load_exclusion_table(root: Path | None = None):
    """Load typed declarations from the reviewed YAML, including scratch roots.

    The existing gate still independently checks predicates, counts, overlaps
    and actual builder aliases. Schema validation does not replace that gate.
    """
    document = load_reviewed_inventory("derivative_exclusions", root)
    rules = tuple(
        ExcludedCellRule(
            **{
                **row,
                "pattern": tuple((part["axis"], tuple(part["tokens"])) for part in row["pattern"]),
            }
        )
        for row in document["excluded_cell_rules"]
    )
    aliases = tuple(RealizationAlias(**{**row, "tokens": tuple(row["tokens"])}) for row in document["realization_aliases"])
    return tuple(document["axes"]), rules, aliases


AXES, EXCLUDED_CELL_RULES, REALIZATION_ALIASES = load_exclusion_table()


def matching_rules(cell: tuple, rules: tuple[ExcludedCellRule, ...] = EXCLUDED_CELL_RULES) -> list[str]:
    """Rule IDs of every reviewed exclusion that matches the 5-tuple ``cell``."""
    return [rule.rule_id for rule in rules if rule.matches(cell)]


def is_excluded(cell: tuple, rules: tuple[ExcludedCellRule, ...] = EXCLUDED_CELL_RULES) -> bool:
    """Whether a reviewed exclusion row removes ``cell`` from the matrix universe."""
    return any(rule.matches(cell) for rule in rules)
