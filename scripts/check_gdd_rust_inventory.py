#!/usr/bin/env python3
"""Pin Rust tests, Python-visible exports and route predicates to reviewed inventories.

* every ``#[test]`` unit under ``jqmc/rs/src`` is listed with a
  reviewed kind; adding or deleting one without review fails.
* every Python-visible PyO3 export (registered
  ``#[pyfunction]``, ``#[pymethods]`` method, ``#[pyclass]``) is listed with its
  status recomputed from QUALIFIED references, so a dead export cannot hide
  behind a same-named Python function and a status change needs review.
* every route-like Rust predicate has an exact reviewed
  classification. Capability/performance routes retain their counter witness;
  numerically safer equivalent arithmetic is recorded separately.

Rayon / GIL-release sites are pinned by the separate
``scripts/check_rust_parallel_sites.py``.
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

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdd.engine import rust_inventory  # noqa: E402
from gdd.engine.route_counter_evidence import (  # noqa: E402
    COUNTER_READ_EVIDENCE,
    PREDICATE_COUNTER_READ_EVIDENCE,
)
from gdd.engine.rust_predicate_reviews import (  # noqa: E402
    RUST_NUMERIC_SAFETY_PREDICATES,
    RUST_ROUTE_PREDICATES,
    PredicateKey,
    RouteReview,
)
from gdd.engine.rust_references import export_status, reference_sets  # noqa: E402
from gdd.engine.rust_sources import (  # noqa: E402
    python_exports,
    route_predicates,
    rust_counter_symbols,
    rust_predicate_counter_symbols,
    rust_tests,
)

TEST_KINDS = rust_inventory.TEST_KINDS
EXPORT_STATUSES = rust_inventory.EXPORT_STATUSES


def test_inventory_errors(root: Path, reviewed: dict[str, str], test_kinds: frozenset[str] = TEST_KINDS) -> list[str]:
    errors = []
    found = [f"{path}::{name}" for path, name, _line in rust_tests(root)]
    duplicates = sorted(name for name, count in collections.Counter(found).items() if count > 1)
    for name in duplicates:
        errors.append(f"Rust #[test] unit {name} is defined more than once")
    for name in sorted(set(found) - set(reviewed)):
        errors.append(f"Rust #[test] unit {name} is not in the reviewed Rust inventory")
    for name in sorted(set(reviewed) - set(found)):
        errors.append(f"Rust #[test] unit {name} is in the reviewed inventory but no longer exists")
    for name, kind in sorted(reviewed.items()):
        if kind not in test_kinds:
            errors.append(f"Rust #[test] unit {name} has unreviewed kind {kind!r}")
    return errors


def export_inventory_errors(
    root: Path, reviewed: dict[str, str], export_statuses: frozenset[str] = EXPORT_STATUSES
) -> list[str]:
    errors = []
    found = [name for name, _path, _line in python_exports(root)]
    for name in sorted(name for name, count in collections.Counter(found).items() if count > 1):
        errors.append(f"Python-visible Rust export {name} is defined more than once")
    for name in sorted(set(found) - set(reviewed)):
        errors.append(f"Python-visible Rust export {name} is not in the reviewed Rust inventory")
    for name in sorted(set(reviewed) - set(found)):
        errors.append(f"Python-visible Rust export {name} is in the reviewed inventory but no longer exists")
    production, tests = reference_sets(root)
    for name in sorted(set(found) & set(reviewed)):
        status = reviewed[name]
        if status not in export_statuses:
            errors.append(f"Python-visible Rust export {name} has unreviewed status {status!r}")
            continue
        actual = export_status(name, production, tests)
        if actual != status:
            errors.append(f"Python-visible Rust export {name} is reviewed as {status!r} but is now {actual!r}")
    return errors


def predicate_inventory_errors(
    root: Path,
    route_reviews: dict[PredicateKey, RouteReview] = RUST_ROUTE_PREDICATES,
    numeric_reviews: dict[PredicateKey, str] = RUST_NUMERIC_SAFETY_PREDICATES,
) -> list[str]:
    """Reject unreviewed, stale, overlapping or unwitnessed Rust predicates."""
    errors = []
    found_rows = route_predicates(root)
    found = [(path, owner, name) for path, owner, name, _line in found_rows]
    lines = {(path, owner, name): line for path, owner, name, line in found_rows}
    for key, count in sorted(collections.Counter(found).items()):
        if count > 1:
            errors.append(f"Rust predicate {key} is detected more than once")

    route_keys = set(route_reviews)
    numeric_keys = set(numeric_reviews)
    overlap = route_keys & numeric_keys
    for key in sorted(overlap):
        errors.append(f"Rust predicate {key} has both route and numerical-safety reviews")

    reviewed = route_keys | numeric_keys
    for key in sorted(set(found) - reviewed):
        errors.append(f"Rust predicate {key} at line {lines[key]} has no exact reviewed classification")
    for key in sorted(reviewed - set(found)):
        errors.append(f"reviewed Rust predicate {key} no longer exists")

    route_symbols = rust_counter_symbols(root) if route_reviews else []
    predicate_symbols = rust_predicate_counter_symbols(root) if route_reviews else []
    for key, (witness, reason) in sorted(route_reviews.items()):
        if not reason.strip():
            errors.append(f"Rust route predicate {key} has no reason")
        if witness is None:
            if not reason.startswith("performance:"):
                errors.append(f"Rust route predicate {key} has no counter witness or performance classification")
            continue
        if isinstance(witness, tuple):
            family, index = witness
            if family != "predicate" or not 0 <= index < len(predicate_symbols):
                errors.append(f"Rust route predicate {key} has invalid witness {witness}")
            elif not predicate_symbols[index].endswith("_DECLINES"):
                errors.append(f"Rust route predicate {key} witness {witness} is not a decline counter")
            elif index not in PREDICATE_COUNTER_READ_EVIDENCE:
                errors.append(f"Rust route predicate {key} witness {witness} has no registered test read")
            continue
        if not 0 <= witness < len(route_symbols):
            errors.append(f"Rust route predicate {key} counter {witness} is outside the route-counter array")
        elif witness not in COUNTER_READ_EVIDENCE:
            errors.append(f"Rust route predicate {key} counter {witness} has no registered test read")

    for key, reason in sorted(numeric_reviews.items()):
        if not reason.startswith("numerical-safety:"):
            errors.append(f"Rust numerical-safety predicate {key} lacks a numerical-safety reason")
    return errors


def validation_errors(root: Path = ROOT) -> list[str]:
    reviewed = rust_inventory.load_rust_inventory(root)
    return (
        test_inventory_errors(root, reviewed["RUST_TESTS"], reviewed["TEST_KINDS"])
        + export_inventory_errors(root, reviewed["PYTHON_EXPORTS"], reviewed["EXPORT_STATUSES"])
        + predicate_inventory_errors(root)
    )


def main() -> int:
    try:
        errors = validation_errors()
    except (OSError, SyntaxError, ValueError) as exc:
        errors = [str(exc)]
    if errors:
        print("GDD Rust inventory contract failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    statuses = collections.Counter(rust_inventory.PYTHON_EXPORTS.values())
    print(
        "GDD Rust inventory contract passed "
        f"({len(rust_inventory.RUST_TESTS)} #[test] units; {len(rust_inventory.PYTHON_EXPORTS)} exports "
        f"{dict(sorted(statuses.items()))}; "
        f"{len(RUST_ROUTE_PREDICATES)} route predicates; "
        f"{len(RUST_NUMERIC_SAFETY_PREDICATES)} numerical-safety predicates)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
