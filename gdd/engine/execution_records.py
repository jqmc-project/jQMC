"""Small pytest execution records, independent of derived fixture descriptions.

Collection manifests come from a separate preflight collection, not from the
workers whose completeness they check.  A passed call is insufficient: setup,
call and teardown must all pass on every declared MPI rank.  No Python frame
tracing or interpretation of scientific assertions is performed here.
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
import importlib
import json
import os
import uuid
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest


SCHEMA = "gdd-pytest-execution.v1"
PHASES = {"setup", "call", "teardown"}
# Cost tier of a collected item. ``tests/conftest.py`` enforces exactly one of
# the two markers, so every CI ``-m`` expression is one of these predicates.
TIERS = ("light", "heavy")


def item_tier(item) -> str | None:
    """The item's cost tier, or None for a folder whose conftest declares none."""
    if item.get_closest_marker("heavy"):
        return "heavy"
    if item.get_closest_marker("light"):
        return "light"
    return None


def execution_profile(environ: dict[str, str] | None = None, *, collect: bool = False) -> dict[str, Any]:
    """Describe the actual MPI world, or an explicitly requested collect-only world."""
    env = os.environ if environ is None else environ
    size = int(env.get("OMPI_COMM_WORLD_SIZE") or env.get("PMI_SIZE") or env.get("MV2_COMM_WORLD_SIZE") or "1")
    rank = int(env.get("OMPI_COMM_WORLD_RANK") or env.get("PMI_RANK") or env.get("MV2_COMM_WORLD_RANK") or "0")
    if collect and "GDD_EXECUTION_WORLD_SIZE" in env:
        size = int(env["GDD_EXECUTION_WORLD_SIZE"])
        rank = 0
    if size < 1 or not 0 <= rank < size:
        raise ValueError(f"invalid MPI execution profile: size={size}, rank={rank}")
    return {"name": "mpi" if size > 1 else "serial", "world_size": size, "rank": rank}


def _case_key(nodeid: str, document: dict[str, Any], rank: int) -> tuple[str, str, str, int, int]:
    profile = document["execution"]
    return nodeid, document["precision"], profile["name"], profile["world_size"], rank


def declared_case_required(node, case, execution, *, precision, profile) -> bool:
    """Preserve declared per-case lanes, rather than inventing Cartesian coverage."""
    modes = case.get("precision_modes", execution["precision_modes"])
    # Precision-independent checks have one canonical full-lane obligation.
    supported_modes = {"full"} if modes == ["not-applicable"] else set(modes)
    if "required_parameter_cases" not in node or case.get("selector") in node["required_parameter_cases"]:
        supported_modes.update(node.get("required_precision_modes", ()))
    profiles = {(value["name"], value["world_size"]) for value in execution["profiles"]}
    profiles.update((value["name"], value["world_size"]) for value in node.get("required_execution_profiles", ()))
    return precision in supported_modes and (profile["name"], profile["world_size"]) in profiles


def verified_source_selector(item, selector):
    """Retain source symbols only after checking actual callable identity.

    The legacy canonical serializer describes every plain function by the same
    type. That is insufficient to distinguish two parameter factory functions.
    ``case_for`` still checks all ordinary values; source-bound callable leaves
    additionally must be the exact object supplied by pytest's callspec.
    """
    namespace = dict(vars(item.module))

    def contains_expression(value):
        if isinstance(value, dict):
            return set(value) == {"source_expression"} or any(contains_expression(child) for child in value.values())
        if isinstance(value, list):
            return any(contains_expression(child) for child in value)
        return False

    def check(declared, actual):
        if not contains_expression(declared):
            return
        if isinstance(declared, dict) and set(declared) == {"source_expression"}:
            expression = declared["source_expression"]
            # Match the existing resolver's aliases without importing numerical
            # production modules when a plain local factory symbol is enough.
            names = {node.id for node in ast.walk(ast.parse(expression, mode="eval")) if isinstance(node, ast.Name)}
            for module_name in names & {"jqmc", "numpy", "os"}:
                if module_name not in namespace:
                    namespace[module_name] = importlib.import_module(module_name)
            expected = eval(expression, {"__builtins__": {}}, namespace)
            if callable(expected) and expected is not actual:
                raise AssertionError(f"{item.nodeid}: callable parameter differs from {declared['source_expression']}")
        elif isinstance(declared, dict):
            for key, child in declared.items():
                if contains_expression(child):
                    check(child, actual[key])
        elif isinstance(declared, list):
            for child, value in zip(declared, actual, strict=True):
                check(child, value)

    callspec = getattr(item, "callspec", None)
    check(selector, callspec.params if callspec is not None else {})
    return selector


def collected_contract_requirements(items, *, root, precision, profile, selectors=None) -> dict[str, bool]:
    """Match collection once, without runtime tracing or a single-owner restriction.

    Only preflight uses the generated descriptions. The execution workers need
    neither a binding parser nor a scientific observation recorder. Specialized
    derivative matrix obligations retain their own explicit receipt completion.
    """
    from gdd.engine.guarantee_store import load_module
    from gdd.engine.ledger import iter_registered_nodes, load_ledger
    from gdd.engine.module_receipts import EvidenceNodeContract, runtime_selector, split_nodeid

    items_by_node = defaultdict(list)
    required = {item.nodeid: False for item in items}
    for item in items:
        items_by_node[split_nodeid(item.nodeid)[0]].append(item)
    for source in sorted((Path(root) / "gdd/ssot/guarantees").glob("*.yaml")):
        # Inspect only the small authored relation to choose relevant ledgers;
        # load_ledger verifies the generated payload before using any binding.
        authored = load_module(source)
        if not any(node["pytest_node"] in items_by_node for _, _, node in iter_registered_nodes(authored, source)):
            continue
        document = load_ledger(source, root=Path(root))
        for guarantee, _, node in iter_registered_nodes(document, source):
            matching = items_by_node.get(node["pytest_node"], ())
            if not matching:
                continue
            binding = node["case_binding"]
            contract = EvidenceNodeContract(guarantee["id"], source.name, node["pytest_node"], (), binding)
            for item in matching:
                _, parameter_id = split_nodeid(item.nodeid)
                case = contract.case_for(parameter_id, runtime_selector(item), vars(item.module))
                selector = verified_source_selector(item, case["selector"])
                if selectors is not None:
                    if item.nodeid in selectors and selectors[item.nodeid] != selector:
                        raise AssertionError(f"{item.nodeid}: shared guarantees disagree on its source selector")
                    selectors[item.nodeid] = selector
                required[item.nodeid] |= declared_case_required(
                    node, case, binding["execution"], precision=precision, profile=profile
                )
    return required


def _validate_document(document: dict[str, Any], *, kind: str, commit: str, run_id: str) -> None:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA or document.get("kind") != kind:
        raise ValueError(f"expected {SCHEMA} {kind} document")
    if document.get("commit") != commit or document.get("run_id") != run_id:
        raise ValueError("execution record belongs to another commit or CI run")
    if document.get("precision") not in {"full", "mixed"}:
        raise ValueError("execution record must declare full or mixed precision")
    profile = document.get("execution", {})
    size, rank = profile.get("world_size"), profile.get("rank")
    if type(size) is not int or size < 1 or type(rank) is not int or not 0 <= rank < size:
        raise ValueError("execution record has an invalid MPI size or rank")
    if profile.get("name") != ("mpi" if size > 1 else "serial"):
        raise ValueError("execution profile name disagrees with world size")
    if type(document.get("exitstatus")) is not int or document["exitstatus"] != 0:
        raise ValueError(f"{kind} session did not finish successfully: {document.get('exitstatus')!r}")
    if not isinstance(document.get("selected"), list):
        raise ValueError("execution record has no selected collection")
    seen = set()
    for row in document["selected"]:
        if not isinstance(row, dict) or not isinstance(row.get("nodeid"), str) or not row["nodeid"].strip():
            raise ValueError("selected collection has an invalid node id")
        if row["nodeid"] in seen:
            raise ValueError(f"duplicate node in one collection: {row['nodeid']}")
        seen.add(row["nodeid"])
        if row.get("excluded") is not None and (not isinstance(row["excluded"], str) or not row["excluded"].strip()):
            raise ValueError("collection exclusion must carry its pytest skip reason")
        if "required" in row and type(row["required"]) is not bool:
            raise ValueError("case required flag must be boolean")
        if row.get("tier") not in {None, *TIERS}:
            raise ValueError("case tier must be light, heavy or absent")


def completion_errors(
    expected: Iterable[dict[str, Any]],
    records: Iterable[dict[str, Any]],
    *,
    commit: str,
    run_id: str,
) -> list[str]:
    """Compare independently collected obligations with actual complete outcomes.

    Existing collection-time skips remain explicit exclusions, not successes.
    A new runtime skip, failure, missing phase, missing shard or missing MPI
    rank cannot close a non-excluded case. Repeated workers do not add coverage.
    """
    errors: list[str] = []
    required: set[tuple[str, str, str, int, int]] = set()
    for document in expected:
        _validate_document(document, kind="collection", commit=commit, run_id=run_id)
        for row in document["selected"]:
            keys = {_case_key(row["nodeid"], document, rank) for rank in range(document["execution"]["world_size"])}
            if row.get("required", not row.get("excluded")):
                required.update(keys)
    # A case required by one collection is not waived by a skip in another.
    if not required:
        errors.append("no required cases in the independent collection manifests")
    observations: dict[tuple[str, str, str, int, int], list[dict[str, set[str]]]] = defaultdict(list)
    for document in records:
        _validate_document(document, kind="execution", commit=commit, run_id=run_id)
        selected = {row["nodeid"] for row in document["selected"]}
        phases: dict[tuple[str, str, str, int, int], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        if not isinstance(document.get("reports"), list):
            raise ValueError("execution record has no phase reports")
        for row in document["reports"]:
            if not isinstance(row, dict) or row.get("nodeid") not in selected:
                raise ValueError("phase report is not part of its session collection")
            if row.get("when") not in PHASES or row.get("outcome") not in {"passed", "failed", "skipped", "xpassed"}:
                raise ValueError("invalid phase report")
            key = _case_key(row["nodeid"], document, document["execution"]["rank"])
            phases[key][row["when"]].add(row["outcome"])
        for key, outcome in phases.items():
            observations[key].append(outcome)
    for key in sorted(required):
        label = f"{key[0]} [{key[1]}, {key[2]}:{key[3]}, rank={key[4]}]"
        runs = observations.get(key, [])
        if not runs:
            errors.append(f"{label}: no execution evidence (case, shard or rank missing)")
            continue
        if any(any(values != {"passed"} for values in phases.values()) for phases in runs):
            errors.append(f"{label}: skipped, failed or expected-failure outcome is not passing evidence")
        elif not any(set(phases) == PHASES and all(values == {"passed"} for values in phases.values()) for phases in runs):
            errors.append(f"{label}: no single execution passed setup, call and teardown")
    return errors


def read_records(directory: str | Path, kind: str) -> list[dict[str, Any]]:
    """Read only the distinct new schema files; legacy derivative receipts coexist."""
    paths = sorted(Path(directory).rglob(f"gdd-{kind}-*.json"))
    if not paths:
        raise ValueError(f"no GDD {kind} records found in {directory}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


class ExecutionRecorder:
    """Pytest plugin collecting native outcomes without inspecting test internals."""

    def __init__(self, config):
        self.config = config
        self.directory = Path(os.environ["GDD_RECEIPT_DIR"]).resolve()
        self.collect = config.getoption("collectonly", False)
        requested_mode = os.environ.get("GDD_EXECUTION_MODE", "run")
        if requested_mode not in {"run", "collect"} or (requested_mode == "collect" and not self.collect):
            raise pytest.UsageError("GDD_EXECUTION_MODE=collect requires pytest --collect-only")
        if self.collect and os.environ.get("JQMC_TEST_SHARD_COUNT", "1") != "1":
            raise pytest.UsageError("independent GDD preflight collection must not be sharded")
        self.profile = execution_profile(collect=self.collect)
        self.before_selection: list[str] = []
        self.selected: list[dict[str, Any]] = []
        self.reports: list[dict[str, str]] = []

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, config, items):
        self.before_selection = [item.nodeid for item in items]
        yield
        from _pytest.skipping import evaluate_skip_marks

        requirements = None
        selectors = {}
        if self.collect and os.environ.get("GDD_EXECUTION_CONTRACT_SCOPE") == "1":
            from gdd.engine.derivative_registry import derivative_keys

            requirements = collected_contract_requirements(
                items,
                root=Path(__file__).resolve().parents[2],
                precision=config.getoption("precision_mode", "full"),
                profile=self.profile,
                selectors=selectors,
            )
        for item in items:
            skip = evaluate_skip_marks(item)
            row = {"nodeid": item.nodeid, "excluded": skip.reason if skip else None, "tier": item_tier(item)}
            if requirements is not None:
                from gdd.engine.module_receipts import runtime_selector, split_nodeid

                row["required"] = requirements[item.nodeid]
                row["definition"] = split_nodeid(item.nodeid)[0]
                row["selector"] = runtime_selector(item)
                if item.nodeid in selectors:
                    row["contract_selector"] = selectors[item.nodeid]
                # Issue 789: the registered derivative receipts of this case, from
                # its definition and real callspec values (stage i closes them).
                keys = derivative_keys(
                    row["definition"],
                    dict(getattr(getattr(item, "callspec", None), "params", {})),
                    config.getoption("precision_mode", "full"),
                )
                if keys is not None and keys["keys"]:
                    row["derivative_keys"] = keys
            self.selected.append(row)

    def pytest_runtest_logreport(self, report):
        outcome = "xpassed" if report.outcome == "passed" and hasattr(report, "wasxfail") else report.outcome
        self.reports.append({"nodeid": report.nodeid, "when": report.when, "outcome": outcome})

    def pytest_sessionfinish(self, session, exitstatus):
        # xdist controllers receive forwarded reports but do not collect items.
        # Workers each persist their own selected set and actual phase outcomes.
        if not hasattr(self.config, "workerinput") and self.config.getoption("numprocesses", 0) and not self.collect:
            return
        # Some matrix files have no cases of a cost tier. Their CI wrapper
        # independently verifies that the file is nonempty. An empty lane
        # contributes no evidence; the preflight manifest still detects any
        # required case missing from the union of the actual nonempty lanes.
        if int(exitstatus) == 5 and not self.selected and not self.reports:
            return
        kind = "collection" if self.collect else "execution"
        document = {
            "schema": SCHEMA,
            "kind": kind,
            "commit": os.environ.get("GDD_COMMIT_SHA", "local"),
            "run_id": os.environ.get("GDD_RUN_ID", "local"),
            # The workflow attempt (issue 789): the in-job verifier binds every
            # record to the attempt that is running it.
            "attempt": os.environ.get("GDD_RUN_ATTEMPT", "local"),
            "build_id": os.environ.get("GDD_BUILD_ID", "local"),
            "lane": os.environ.get("GDD_EVIDENCE_LANE", "local"),
            "job": os.environ.get("GDD_JOB_ID", "local"),
            "shard": os.environ.get("GDD_SHARD_ID", "local"),
            "precision": self.config.getoption("precision_mode", "full"),
            "execution": self.profile,
            "exitstatus": int(exitstatus),
            "before_selection": self.before_selection,
            "selected": self.selected,
            "reports": self.reports,
        }
        if not self.collect:
            # Issue 789: cases whose emitted derivative receipts equalled their
            # registered keys at teardown ({nodeid: {"lane", "keys"}}).
            from gdd.engine.derivative_registry import SELFCHECK_PLUGIN

            selfcheck = self.config.pluginmanager.get_plugin(SELFCHECK_PLUGIN)
            document["derivative_selfcheck"] = dict(selfcheck.attestations) if selfcheck is not None else {}
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"gdd-{kind}-{uuid.uuid4().hex}.json"
        path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def register(config) -> None:
    """Enable recording only for authoritative CI lanes or an explicit local run."""
    if os.environ.get("GDD_RECEIPT_DIR") and os.environ.get("GDD_RECEIPT_AUTHORITY", "1") == "1":
        config.pluginmanager.register(ExecutionRecorder(config), "gdd-execution-records")
