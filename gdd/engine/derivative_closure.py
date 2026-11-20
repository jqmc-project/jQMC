"""Derivative receipt validation and the exact derivative universes, shared by every consumer.

Issue 789: ``validate_record`` is the receipt check of the in-test self-check
(``gdd.engine.derivative_registry``); ``lane_universes`` and its sha256 pin are
what ``collection_closure_errors`` closes the registered keys of the collected
cases against.

The universes are computed from the authored contract (no pinned counts) and
their digests are pinned: XLA full and mixed against ``REQUIRED_CASES`` by
precision, the reviewed FD ids, the ten MIXED production routes, and the Rust
fragments of every required case (Rust R03/R09 derived from their leaves).
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

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FD_GUARANTEE_ID = "NUM-DERIV-FD-AUTODIFF-001"
MIXED_GUARANTEE_ID = "NUM-DERIV-MIXED-001"
XLA_GUARANTEE_PREFIX = "MATH-DERIV-"
# Evidence lane -> precision of its XLA/Rust universe.
XLA_LANES = {"xla-full": "full", "mixed": "mixed"}
RUST_LANES = {"rust-full": "full", "rust-mixed": "mixed"}
DERIVATIVE_LANES = (*XLA_LANES, *RUST_LANES)


def required_fd_ids(contract: dict | None = None) -> set[str]:
    """Expand authored FD selectors; no hand-maintained count or digest."""
    if contract is None:
        from gdd.engine.guarantee_store import load_module

        contract = load_module(ROOT / "gdd/ssot/guarantees/derivatives.yaml")
    fixture = next(item for item in contract["fixture_sets"] if item["id"] == "DERIV-FIX-FD-AUTODIFF-001")
    result = set()
    for node in fixture["reviewed_evidence"]["nodes"]:
        keys = node["id_dimensions"]
        dimensions = node["fixture"]["dimensions"]
        if not keys or keys != sorted(set(keys)):
            raise ValueError(f"{node['id']}: FD identity dimensions must be nonempty, unique and sorted")
        values = [dimensions[key] if isinstance(dimensions[key], list) else [dimensions[key]] for key in keys]
        if any(not values_for_key for values_for_key in values):
            raise ValueError(f"{node['id']}: empty FD selector dimension")
        for combination in product(*values):
            identifier = f"FD::{node['id']}::" + ",".join(
                f"{key}={value}" for key, value in zip(keys, combination, strict=True)
            )
            if identifier in result:
                raise ValueError(f"duplicate FD requirement {identifier}")
            result.add(identifier)
    if not result:
        raise ValueError("FD requirement universe is empty")
    return result


def required_rust_fragments(cases=None) -> set[str]:
    """Expand every upstream requirement into its physical Rust components.

    Parameter blocks are required by the declared fixture features, not by
    whichever block registrations happened to be present in a Rust test.
    """
    from gdd.engine import derivative_requirements as requirements
    from gdd.engine.derivative_matrix import _cell_id, _split_cell

    cases = requirements.REQUIRED_CASES if cases is None else cases
    cells = {_cell_id(cell): _split_cell(cell) for cell, _flags in requirements._MATRIX}
    result = set()
    for case in cases:
        parts = case.id.split("|")
        if case.category == "matrix":
            axis = parts[-1].split("=", 1)[1]
            blocks = [axis]
            if axis == "c":
                ao, jastrow, *_ = cells[parts[2]]
                radial = "gto" in ao or ao == "composite"
                blocks = ["Det"] + (["Basis"] if radial else [])
                if jastrow in {"J1", "J2", "J3poly", "DTN"}:
                    blocks += ["J1", "J2"]
                if jastrow == "J3poly":
                    blocks += ["J3"] + (["J3Basis"] if radial else [])
                if jastrow == "DTN":
                    blocks += ["DTN"]
        elif case.category == "leaf":
            blocks = [case.source_id]
        elif case.category == "regime":
            if case.family in {"ln_Psi", "e_L"} and case.source_id in {"R03", "R09"}:
                continue  # Closed by the explicit leaf conjunction below.
            blocks = ["regime"]
        elif case.category == "validity":
            blocks = ["validity" if case.family in {"e_L_t", "e_L_n"} else parts[-1]]
        elif case.category == "output":
            blocks = ["output"]
        elif case.category in {"gfmc-direct", "gfmc-identity"}:
            blocks = [parts[3]]
        elif case.category == "identity" and case.source_id == "B20":
            blocks = ["c-r-R"]
        else:
            raise ValueError(f"unclassified Rust requirement: {case.id}")
        result.update(f"{case.id}::{block}" for block in blocks)
    return result


def mixed_route_ids() -> set[str]:
    return {f"MIXED::continuous::{quantity}::{route}" for quantity in ("ln_Psi", "e_L") for route in ("c", "r+R", "eps")} | {
        f"MIXED::gfmc::{family}::{route}" for family in ("t", "n") for route in ("positions", "strain")
    }


# Issue 789: the exact per-lane derivative universes are pinned. A contract edit
# that changes a universe must update this pin in the same reviewed change.
PINNED_UNIVERSE_SHA256 = {
    "xla-full": "eb8f7e989fc3447d54ef6388921d1874721453296d2e113f0934f87100fda3d1",
    "mixed": "e5190a58a18c0b7c31056fe3358ee565a996d08ffa27570677cec90e247b4b44",
    "rust-full": "0bdcd6d14b8d02bf3fbd071fd4ad8d2e9414429860bf7e5689f08fc1026317b5",
    "rust-mixed": "0bdcd6d14b8d02bf3fbd071fd4ad8d2e9414429860bf7e5689f08fc1026317b5",
}


def lane_universes(cases=None) -> dict[str, set[str]]:
    """The exact receipt keys each lane must close, from the authored contract.

    XLA lanes: every required case of the precision (R03/R09 have their own
    receipts) plus the FD ids (full) or the MIXED routes (mixed). Rust lanes: the
    fragments of every required case, with ln_Psi/e_L R03/R09 derived from leaves.
    """
    if cases is None:
        from gdd.engine.derivative_requirements import REQUIRED_CASES as cases
    universes = {}
    for lane, precision in XLA_LANES.items():
        keys = {case.id for case in cases if precision in case.required_precision}
        universes[lane] = keys | (required_fd_ids() if lane == "xla-full" else mixed_route_ids())
    for lane, precision in RUST_LANES.items():
        universes[lane] = required_rust_fragments([case for case in cases if precision in case.required_precision])
    return universes


def universe_pin_errors(universes: dict[str, set[str]], pins: dict[str, str] | None = None) -> list[str]:
    pins = PINNED_UNIVERSE_SHA256 if pins is None else pins
    errors = []
    for lane in sorted(set(universes) | set(pins)):
        actual = digest(universes.get(lane, set()))
        if actual != pins.get(lane):
            errors.append(
                f"derivative universe drift: {lane} sha256 {actual} ({len(universes.get(lane, ()))} keys) "
                f"is not the pinned {pins.get(lane)}; review the contract change and update PINNED_UNIVERSE_SHA256"
            )
    return errors


def digest(values: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def set_delta(actual: set[str], expected: set[str]) -> str:
    """Render a bounded but actionable exact-set delta for CI logs."""
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return (
        f"missing_count={len(missing)}, missing_first_20={missing[:20]}, extra_count={len(extra)}, extra_first_20={extra[:20]}"
    )


def validate_record(record: dict) -> None:
    """Payload, kind, FD and matrix-route validation of one checksum-verified receipt."""
    from gdd.engine.evidence import CompositeCaseReceipt
    from gdd.engine.fd_comparison_contract import GUARANTEE as FD_GUARANTEE, validate_fd_receipt
    from gdd.engine.fixture_observations import validate_matrix_fixture_route
    from gdd.engine.receipt_records import decode_case_receipt

    if not isinstance(record, dict) or record.get("schema_version") != 2 or record.get("status") != "passed":
        raise AssertionError("invalid derivative receipt status/schema")
    receipt = decode_case_receipt(record.get("evidence"))
    for field in ("guarantee_id", "obligation_id", "pytest_node_id", "status"):
        if record.get(field) != getattr(receipt, field):
            raise AssertionError(f"derivative receipt payload {field} mismatch")
    kind = record.get("record_kind")
    if kind not in {"case", "composite", "rust-fragment"}:
        raise AssertionError("invalid derivative receipt kind")
    if (kind == "composite") != isinstance(receipt, CompositeCaseReceipt):
        raise AssertionError("derivative receipt kind disagrees with payload type")
    if receipt.guarantee_id == FD_GUARANTEE:
        if record.get("lane") != "xla-full" or kind != "case":
            raise AssertionError("FD comparison contract: receipts require the xla-full case lane")
        validate_fd_receipt(receipt)
    if record.get("lane") in {"xla-full", "mixed"} and "|matrix|" in receipt.obligation_id:
        if isinstance(receipt, CompositeCaseReceipt):
            raise AssertionError("XLA matrix requires a direct case receipt")
        validate_matrix_fixture_route(receipt)


def load_checked_records(paths, validate=None) -> list[dict]:
    """Read receipts, verify each sha256 over its canonical payload, then validate it."""
    validate = validate_record if validate is None else validate
    records = []
    for path in paths:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise AssertionError(f"derivative receipt must be a JSON object: {path}")
        checksum = record.pop("sha256", None)
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        if checksum != hashlib.sha256(encoded.encode()).hexdigest():
            raise AssertionError(f"receipt checksum mismatch: {path}")
        validate(record)
        records.append(record)
    return records


def derive_r03_r09(ids: set[str], cases=None) -> None:
    """Close the ln_Psi/e_L R03 and R09 regimes from their leaf conjunctions."""
    if cases is None:
        from gdd.engine.derivative_requirements import REQUIRED_CASES as cases
    required_ids = {case.id for case in cases}
    for family in ("ln_Psi", "e_L"):
        leaves = {case.id for case in cases if case.family == family and case.category == "leaf"}
        if f"{family}|regime|R03" in required_ids and leaves and leaves <= ids:
            ids.add(f"{family}|regime|R03")
        if f"{family}|regime|R09" in required_ids and {f"{family}|leaf|P01", f"{family}|leaf|P03"} <= ids:
            ids.add(f"{family}|regime|R09")


def classify(record: dict) -> tuple[str, str] | None:
    """The (lane, key) a receipt closes, with the aggregator's exact predicates.

    XLA: lane xla-full or mixed and a MATH-DERIV-* guarantee (any record kind);
    FD: lane xla-full and the FD guarantee; MIXED: lane mixed and the MIXED
    guarantee; Rust: lane rust-full or rust-mixed and a rust-fragment record,
    keyed ``upstream::component``. Any other receipt is validated but closes
    nothing, exactly as in the aggregator.
    """
    lane, guarantee = record.get("lane"), str(record.get("guarantee_id"))
    if lane in XLA_LANES and guarantee.startswith(XLA_GUARANTEE_PREFIX):
        return lane, str(record["obligation_id"])
    if lane == "xla-full" and guarantee == FD_GUARANTEE_ID:
        return lane, str(record["obligation_id"])
    if lane == "mixed" and guarantee == MIXED_GUARANTEE_ID:
        return lane, str(record["obligation_id"])
    if lane in RUST_LANES and record.get("record_kind") == "rust-fragment":
        return lane, f"{record['upstream_obligation_id']}::{record['component_id']}"
    return None


def universe_errors(keys_by_lane: dict[str, set[str]], cases=None) -> list[str]:
    """Do these emitted keys close the aggregator's exact universes? (no counts pinned)"""
    if cases is None:
        from gdd.engine.derivative_requirements import REQUIRED_CASES as cases
    errors = []
    fd_expected, mixed_expected = required_fd_ids(), mixed_route_ids()
    xla_by_precision = {}
    for lane, precision in XLA_LANES.items():
        keys = set(keys_by_lane.get(lane, ()))
        fd = {key for key in keys if key.startswith("FD::")}
        mixed = {key for key in keys if key.startswith("MIXED::")}
        xla = keys - fd - mixed
        derive_r03_r09(xla, cases)
        expected = {case.id for case in cases if precision in case.required_precision}
        if xla != expected:
            errors.append(f"XLA {precision} universe mismatch: {set_delta(xla, expected)}")
        xla_by_precision[precision] = xla
        if lane == "xla-full" and fd != fd_expected:
            errors.append(f"FD universe mismatch: {set_delta(fd, fd_expected)}")
        if lane == "mixed" and mixed != mixed_expected:
            errors.append(f"mixed route universe mismatch: {set_delta(mixed, mixed_expected)}")
        if lane == "xla-full" and mixed or lane == "mixed" and fd:
            errors.append(f"{lane} carries keys of another lane's universe")
    for lane, precision in RUST_LANES.items():
        fragments = set(keys_by_lane.get(lane, ()))
        expected_cases = [case for case in cases if precision in case.required_precision]
        expected_fragments = required_rust_fragments(expected_cases)
        if fragments != expected_fragments:
            errors.append(f"Rust {precision} fragment universe mismatch: {set_delta(fragments, expected_fragments)}")
        obligations = {key.split("::", 1)[0] for key in fragments}
        derive_r03_r09(obligations, cases)
        expected = {case.id for case in expected_cases}
        if obligations != expected or obligations != xla_by_precision[precision]:
            errors.append(f"Rust {precision} obligation universe mismatch: {set_delta(obligations, expected)}")
    unknown = sorted(set(keys_by_lane) - set(DERIVATIVE_LANES))
    if unknown:
        errors.append(f"derivative keys in lanes without a universe: {unknown}")
    return errors


def _case_required(row: dict) -> bool:
    """A ledger-required collected case, or one with registered derivative keys that the collection did not exclude."""
    if row.get("derivative_keys") and not row.get("excluded"):
        return True
    return row.get("required", not row.get("excluded"))


def collection_closure_errors(documents, *, universes=None, pins=None) -> list[str]:
    """Close the registered derivative keys of the collected cases against the pinned exact universes.

    ``documents`` are the collection records of ``gdd.engine.execution_records``
    (every precision and execution profile; the ranks of one MPI collection must
    agree). Per lane: the union of the rows' ``derivative_keys`` equals the
    universe (whose sha256 is pinned), every key belongs to exactly one collected
    node, and every such node is a required case.
    """
    from collections import defaultdict

    errors: list[str] = []
    authored = universes is None
    universes = lane_universes() if authored else universes
    errors.extend(universe_pin_errors(universes, pins))
    rows_by_profile: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for document in documents:
        execution = document["execution"]
        rows = rows_by_profile[(document["precision"], execution["name"], execution["world_size"])]
        for row in document["selected"]:
            previous = rows.setdefault(row["nodeid"], row)
            if previous is not row and previous != row:
                errors.append(f"MPI ranks disagree on collected row {row['nodeid']}")
    owners: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for rows in rows_by_profile.values():
        for nodeid, row in rows.items():
            planned = row.get("derivative_keys")
            if not planned:
                continue
            lane = planned["lane"]
            if not _case_required(row):
                errors.append(
                    f"derivative-emitting case {nodeid} ({lane}) is not a required collected case "
                    f"(excluded: {row.get('excluded')!r})"
                )
            if counts[lane].setdefault(nodeid, len(planned["keys"])) != len(planned["keys"]):
                errors.append(f"collections disagree on the derivative keys of {nodeid} ({lane})")
            for key in planned["keys"]:
                owners[lane][key].add(nodeid)
    for lane, keys in sorted(owners.items()):
        for key, nodes in sorted(keys.items()):
            if len(nodes) > 1:
                errors.append(f"derivative {lane} {key} belongs to {len(nodes)} collected nodes {sorted(nodes)[:4]}")
    for lane in sorted(set(universes) | set(owners)):
        emitted, expected = set(owners.get(lane, ())), universes.get(lane, set())
        if lane not in universes:
            errors.append(f"derivative keys in a lane without a universe: {lane}")
        elif emitted != expected:
            errors.append(f"{lane} derivative universe is not closed by the collected cases: {set_delta(emitted, expected)}")
    if authored:
        # Rust R03/R09 are derived from their leaf fragments; the Rust obligations
        # must then equal the XLA ones of the same precision.
        errors.extend(universe_errors({lane: set(keys) for lane, keys in owners.items()}))
    return errors
