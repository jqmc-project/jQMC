#!/usr/bin/env python3
"""Generate and verify derived GDD case bindings.

Only node registration and guarantee semantics remain in the hand-authored
SSoT. Counts, claims and large AST-derived snapshots are emitted under
``gdd/derived`` and accepted only when source and generator digests match.
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

import argparse
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
GUARANTEE_DIR = ROOT / "gdd/ssot/guarantees"
sys.path.insert(0, str(ROOT))

from gdd.engine.case_binding_inventory import derivation_cache, derive_case_binding
from gdd.engine.guarantee_store import assemble_texts, is_module_ledger, module_texts
from gdd.engine.ledger import (
    DERIVED_FORMAT,
    GENERATOR_DEPENDENCIES,
    _expected_index,
    authored_document,
    canonical_yaml_sha256,
    derived_path,
    evidence_section,
    load_ledger,
)
from scripts.check_gdd_module_ledgers import _actual_cases


GENERATED_HEADER = "# GENERATED -- do not edit; run python scripts/gdd_sync.py\n"


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _represent_string(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_NoAliasDumper.add_representer(str, _represent_string)


def _dump(value: dict[str, Any], *, generated: bool = False) -> str:
    body = yaml.dump(
        value,
        Dumper=_NoAliasDumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
        width=120,
    )
    return (GENERATED_HEADER if generated else "") + body


# Parsed SSoT documents keyed by path and exact file text (for a guarantee
# module: the texts of its module file and every split guarantee file).  A hit
# returns a fresh deep copy, so callers may still mutate what they receive;
# any edit to a file text is a miss and is parsed again by the same loader.
_LOAD_MEMO: dict[Path, tuple[Any, dict[str, Any]]] = {}


def _source_texts(path: Path) -> Any:
    return module_texts(path) if is_module_ledger(path) else path.read_text(encoding="utf-8")


def _parse_source(texts: Any) -> Any:
    return assemble_texts(texts) if isinstance(texts, tuple) else yaml.safe_load(texts)


def _load(path: Path) -> dict[str, Any]:
    text = _source_texts(path)
    memo = _LOAD_MEMO.get(path)
    if memo is not None and memo[0] == text:
        return copy.deepcopy(memo[1])
    value = _parse_source(text)
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must be a mapping")
    _LOAD_MEMO[path] = (text, copy.deepcopy(value))
    return value


def _derive(
    source: Path,
    document: dict[str, Any],
    source_sha256: str,
) -> tuple[dict[str, Any], list[str]]:
    generated_guarantees: dict[str, Any] = {}
    idiom_warnings: list[str] = []
    section = evidence_section(document, source)
    for guarantee in document.get(section, ()):
        if not isinstance(guarantee.get("required_evidence"), list):
            continue
        # Scope is inferred from the evidence node's reachable production calls.
        # ``full-wavefunction`` is only a neutral fallback for ingestion-only
        # tests; component and non-wavefunction routes are narrowed by
        # ``derive_case_binding`` itself.
        scope = "full-wavefunction"
        claims: list[dict[str, Any]] = []
        nodes: dict[str, Any] = {}
        total = 0
        for group in guarantee["required_evidence"]:
            verdict_method = group["verdict_method"]
            for node in group.get("nodes", ()):
                pytest_node = node["pytest_node"]
                if pytest_node in nodes:
                    raise ValueError(f"{source.name}: duplicate pytest node {pytest_node}")
                prederived = _PREDERIVED.get((pytest_node, verdict_method, scope))
                if prederived is not None:
                    # Exactly the successful outcome of the two calls below,
                    # computed by a worker; failures are always recomputed here.
                    expected_cases, binding = prederived
                else:
                    count_errors: list[str] = []
                    expected_cases = _actual_cases(pytest_node, count_errors)
                    if count_errors or expected_cases is None:
                        detail = "; ".join(count_errors) or "case count could not be classified"
                        raise ValueError(f"{pytest_node}: {detail}")
                    binding = derive_case_binding(
                        pytest_node=pytest_node,
                        expected_cases=expected_cases,
                        verdict_method=verdict_method,
                        wavefunction_scope=scope,
                    )
                if binding["test_summary"].get("source") != "test-docstring":
                    raise ValueError(f"{pytest_node}: every governed pytest function must provide a human-readable docstring")
                unregistered = [
                    row["path"]
                    for row in binding["sources"].get("artifacts", ())
                    if str(row.get("source_id", "")).startswith("UNREGISTERED::")
                ]
                if unregistered:
                    raise ValueError(
                        f"{pytest_node}: committed fixtures are absent from gdd/ssot/fixtures/sources.yaml: {unregistered}"
                    )
                observations = binding["comparison"]["typed_observations"]
                if not observations:
                    idiom_warnings.append(f"{pytest_node}: zero typed observations")
                    raise ValueError(f"{pytest_node}: blocking evidence has no source-addressed contractual observation")
                exclusion = binding["execution"].get("precision_exclusion")
                if exclusion and exclusion.get("kind") == "finite-difference-or-numerical-oracle-not-applicable":
                    references = exclusion.get("alternative_guarantee_refs", ())
                    if len(references) != 1 or not references[0]:
                        raise ValueError(f"{pytest_node}: FD/numerical mixed N/A requires one alternative_guarantee_ref")
                if exclusion and exclusion.get("kind") == "precision-independent-contract":
                    if exclusion.get("mode") != "precision" or binding["execution"]["precision_modes"] != ["not-applicable"]:
                        raise ValueError(f"{pytest_node}: malformed precision-independent N/A contract")
                unsupported = list(binding["execution"].get("unsupported_mixed_skips", ()))
                unsupported.extend(
                    reason
                    for case in binding["fixture"]["concrete_cases"]
                    for reason in case.get("unsupported_mixed_skips", ())
                )
                if unsupported:
                    raise ValueError(f"{pytest_node}: non-FD mixed skip is forbidden: {unsupported}")
                claims.append(
                    {
                        "id": binding["claim_refs"][0],
                        "statement": binding["test_summary"]["statement"],
                        "scope": binding["wavefunction"]["scope"],
                        "verdict_method": verdict_method,
                        "required_observation_kinds": sorted({row["kind"] for row in observations}),
                        "observation_roles": sorted({row["role"] for row in observations}),
                        "evidence_node": pytest_node,
                    }
                )
                from gdd.engine.evidence_roles import load_roles

                declaration = load_roles().get(pytest_node)
                if declaration:
                    claims[-1]["evidence_roles"] = declaration["roles"]
                nodes[pytest_node] = {"expected_cases": expected_cases, "case_binding": binding}
                total += expected_cases
        generated_guarantees[guarantee["id"]] = {
            "expected_obligations": total,
            "claims": claims,
            "nodes": nodes,
        }
    payload_sha256 = hashlib.sha256(
        json.dumps(generated_guarantees, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    generated = {
        "schema_version": 1,
        "document": {
            "format": DERIVED_FORMAT,
            "source": source.relative_to(ROOT).as_posix(),
            "source_sha256": source_sha256,
            "generator": "scripts/materialize_gdd_case_bindings.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependency_sha256": {
                path.as_posix(): hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in GENERATOR_DEPENDENCIES
            },
            "payload_sha256": payload_sha256,
        },
        "guarantees": generated_guarantees,
    }
    return generated, idiom_warnings


# Parallel precomputation.  ``_check`` stays the sequential
# reference algorithm.  With ``jobs > 1`` its expensive pure steps are computed
# ahead of it by worker processes and *replayed* at the exact point where the
# sequential loop would have computed them:
#
# * ``_PREDERIVED``: successful ``(_actual_cases, derive_case_binding)`` pairs
#   per (pytest node, verdict method, scope).  A worker failure is dropped, so
#   the sequential calls reproduce the original exception in place.
# * ``_REPLAYED``: the outcome of ``_derive`` per (ledger, source digest),
#   either its return value or the exception object it raised.
# * ``_RENDERED``: the YAML text of exactly those replayed ``generated``
#   objects (matched by object identity).
# * ``_LEDGER_OUTCOMES``: read-only ``load_ledger`` outcomes in check mode,
#   where nothing on disk changes before the sequential call would run.
#
# Nothing here changes what is derived or checked, the order of the errors or
# their messages; generated files are still written only by ``_check``.
_PREDERIVED: dict[tuple[str, str, str], tuple[int, dict[str, Any]]] = {}
_REPLAYED: dict[tuple[Path, str], tuple[str, Any]] = {}
_RENDERED: dict[int, tuple[dict[str, Any], str]] = {}
_LEDGER_OUTCOMES: dict[Path, tuple[str, str | None]] = {}
_LEDGER_ERRORS = (OSError, ValueError, KeyError, yaml.YAMLError)


def default_jobs() -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return max(1, min(16, available))


def _worker_init() -> None:
    # One derivation memo scope for the worker's lifetime; a worker process
    # serves a single materializer run.
    derivation_cache().__enter__()


def _derive_group(tasks: list[tuple[str, str, str]]) -> list[tuple[tuple[str, str, str], Any]]:
    results = []
    for task in tasks:
        pytest_node, verdict_method, scope = task
        outcome = None
        try:
            count_errors: list[str] = []
            expected_cases = _actual_cases(pytest_node, count_errors)
            if not count_errors and expected_cases is not None:
                binding = derive_case_binding(
                    pytest_node=pytest_node,
                    expected_cases=expected_cases,
                    verdict_method=verdict_method,
                    wavefunction_scope=scope,
                )
                outcome = (expected_cases, binding)
        except Exception:  # recomputed sequentially, where it raises in place
            outcome = None
        results.append((task, outcome))
    return results


def _parse_yaml(text: Any) -> Any:
    try:
        return _parse_source(text)
    except Exception:  # the sequential ``_load`` reproduces the error
        return None


def _render(generated: dict[str, Any]) -> str | None:
    try:
        return _dump(generated, generated=True)
    except Exception:  # the sequential loop renders (and fails) in place
        return None


def _ledger_outcome(source: Path) -> tuple[str, str | None] | None:
    try:
        load_ledger(source)
    except _LEDGER_ERRORS as exc:
        return ("error", str(exc))
    except Exception:  # an unexpected failure is re-raised sequentially
        return None
    return ("ok", None)


def _selected_sources(selected: set[str] | None) -> list[Path]:
    return [
        source
        for source in sorted(GUARANTEE_DIR.glob("*.yaml"))
        if not (selected and source.stem not in selected and source.name not in selected)
    ]


def _plan_nodes(source: Path, document: dict[str, Any]) -> list[tuple[str, str, str]]:
    """The (node, verdict method, scope) calls ``_derive`` will make, best effort."""
    tasks: list[tuple[str, str, str]] = []
    try:
        section = evidence_section(document, source)
        for guarantee in document.get(section, ()):
            if not isinstance(guarantee.get("required_evidence"), list):
                continue
            scope = "full-wavefunction"
            for group in guarantee["required_evidence"]:
                verdict_method = group["verdict_method"]
                for node in group.get("nodes", ()):
                    tasks.append((node["pytest_node"], verdict_method, scope))
    except Exception:  # planning only prefetches; ``_derive`` decides
        pass
    return tasks


def _precompute(update: bool, selected: set[str] | None, executor: Executor) -> None:
    sources = _selected_sources(selected)
    ledger_futures = {}
    if not update:
        ledger_futures = {source: executor.submit(_ledger_outcome, source) for source in sources}

    texts: dict[Path, Any] = {}
    for path in sorted(GUARANTEE_DIR.glob("*.yaml")):
        try:
            texts[path] = _source_texts(path)
        except OSError:
            continue
    for path, value in zip(list(texts), executor.map(_parse_yaml, texts.values()), strict=True):
        if isinstance(value, dict):
            _LOAD_MEMO[path] = (texts[path], value)

    # Replicate the sequential pre-checks so that only ledgers which reach
    # ``_derive`` are derived, with the same arguments and in the same order.
    candidates = []
    for source in sources:
        try:
            original = _load(source)
            if original.get("document", {}).get("lifecycle") == "placeholder":
                continue
            cleaned = authored_document(original, source)
            if cleaned != original:
                continue
            _expected_index(cleaned, source)
            digest = canonical_yaml_sha256(source)
        except Exception:
            continue
        candidates.append((source, cleaned, digest))

    planned = {source: _plan_nodes(source, cleaned) for source, cleaned, _digest in candidates}
    owners: dict[tuple[str, str, str], list[Path]] = {}
    groups: dict[str, list[tuple[str, str, str]]] = {}
    for source, _cleaned, _digest in candidates:
        for task in planned[source]:
            if task not in owners:
                owners[task] = []
                groups.setdefault(task[0].split("::", 1)[0], []).append(task)
            if source not in owners[task]:
                owners[task].append(source)
    # Work items: one test file each, so a worker's parsed module and memo
    # tables stay warm for all of that file's nodes.  The largest ledgers are
    # served first (their YAML rendering then overlaps the remaining
    # derivation), and within a ledger the largest files first.
    ledger_rank = {
        source: rank
        for rank, (source, _cleaned, _digest) in enumerate(
            sorted(candidates, key=lambda row: (-len(set(planned[row[0]])), row[0]))
        )
    }
    ordered = sorted(
        groups.values(),
        key=lambda tasks: (min(ledger_rank[source] for task in tasks for source in owners[task]), -len(tasks), tasks[0][0]),
    )
    pending = {source: set(planned[source]) for source, _cleaned, _digest in candidates}
    arguments = {source: (cleaned, digest) for source, cleaned, digest in candidates}
    renders = []

    def replay(source: Path) -> None:
        cleaned, digest = arguments[source]
        try:
            outcome: tuple[str, Any] = ("return", _derive(source, cleaned, digest))
        except Exception as exc:
            outcome = ("raise", exc)
        _REPLAYED[(source, digest)] = outcome
        if outcome[0] == "return":
            generated = outcome[1][0]
            renders.append((generated, executor.submit(_render, generated)))

    for source, _cleaned, _digest in candidates:
        if not pending[source]:
            replay(source)
    for future in as_completed([executor.submit(_derive_group, tasks) for tasks in ordered]):
        for task, outcome in future.result():
            if outcome is not None:
                _PREDERIVED[task] = outcome
            for source in owners[task]:
                pending[source].discard(task)
                if not pending[source]:
                    replay(source)
    for generated, future in renders:
        rendered = future.result()
        if rendered is not None:
            _RENDERED[id(generated)] = (generated, rendered)
    for source, future in ledger_futures.items():
        outcome = future.result()
        if outcome is not None:
            _LEDGER_OUTCOMES[source] = outcome


def _clear_precomputed() -> None:
    _PREDERIVED.clear()
    _REPLAYED.clear()
    _RENDERED.clear()
    _LEDGER_OUTCOMES.clear()


def _replayed_derive(source: Path, document: dict[str, Any], source_sha256: str) -> tuple[dict[str, Any], list[str]]:
    outcome = _REPLAYED.pop((source, source_sha256), None)
    if outcome is None:
        return _derive(source, document, source_sha256)
    kind, value = outcome
    if kind == "raise":
        raise value
    return value


def _rendered(generated: dict[str, Any]) -> str:
    entry = _RENDERED.get(id(generated))
    if entry is not None and entry[0] is generated:
        return entry[1]
    return _dump(generated, generated=True)


def _checked_ledger(source: Path, errors: list[str]) -> None:
    outcome = _LEDGER_OUTCOMES.pop(source, None)
    if outcome is not None:
        if outcome[0] == "error":
            errors.append(outcome[1])
        return
    try:
        load_ledger(source)
    except _LEDGER_ERRORS as exc:
        errors.append(str(exc))


def _verify_written(sources: list[Path], errors: list[str], executor: Executor | None) -> None:
    """``load_ledger`` every written ledger, in order (workers read the new files)."""
    outcomes: list[tuple[str, str | None] | None] = [None] * len(sources)
    if executor is not None:
        try:
            outcomes = list(executor.map(_ledger_outcome, sources))
        except Exception:
            outcomes = [None] * len(sources)
    for source, outcome in zip(sources, outcomes, strict=True):
        if outcome is not None:
            if outcome[0] == "error":
                errors.append(outcome[1])
            continue
        try:
            load_ledger(source)
        except _LEDGER_ERRORS as exc:
            errors.append(str(exc))


def check(update: bool, selected: set[str] | None = None, jobs: int = 1) -> tuple[list[str], int, list[str]]:
    # One memo scope for the whole run: no source file or helper changes while
    # the selected ledgers are derived (generated files are written last).
    with derivation_cache():
        _clear_precomputed()
        try:
            if jobs <= 1:
                return _check(update, selected)
            # ``fork`` (Linux) reuses the already imported engine; elsewhere the
            # platform default start method re-imports this module.
            context = multiprocessing.get_context("fork" if sys.platform.startswith("linux") else None)
            try:
                executor = ProcessPoolExecutor(max_workers=jobs, mp_context=context, initializer=_worker_init)
            except (OSError, ValueError, NotImplementedError):
                return _check(update, selected)
            with executor:
                try:
                    _precompute(update, selected, executor)
                except Exception:  # e.g. a broken pool: run the plain sequential path
                    _clear_precomputed()
                    executor = None
                return _check(update, selected, executor)
        finally:
            _clear_precomputed()


def _check(
    update: bool, selected: set[str] | None = None, executor: Executor | None = None
) -> tuple[list[str], int, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    count = 0
    planned_writes: list[tuple[Path, str, Path, str]] = []
    for source in sorted(GUARANTEE_DIR.glob("*.yaml")):
        if selected and source.stem not in selected and source.name not in selected:
            continue
        original = _load(source)
        if original.get("document", {}).get("lifecycle") == "placeholder":
            continue
        cleaned = authored_document(original, source)
        if cleaned != original:
            errors.append(f"{source.relative_to(ROOT)}: contains derived fields; run scripts/gdd_sync.py")
            continue
        try:
            # Reject inert/misspelled requirements before deriving or writing
            # anything, not only when the finished pair is loaded afterward.
            _expected_index(cleaned, source)
            source_digest = canonical_yaml_sha256(source)
            generated, local_warnings = _replayed_derive(source, cleaned, source_digest)
        except (OSError, SyntaxError, ValueError, KeyError, yaml.YAMLError) as exc:
            errors.append(f"{source.relative_to(ROOT)}: {exc}")
            continue
        warnings.extend(local_warnings)
        count += sum(len(row["nodes"]) for row in generated["guarantees"].values())
        target = derived_path(source)
        rendered = _rendered(generated)
        if update:
            planned_writes.append((source, source_digest, target, rendered))
        elif not target.is_file() or target.read_text(encoding="utf-8") != rendered:
            errors.append(f"{target.relative_to(ROOT)}: stale or missing generated bindings")
            continue
        if not update:
            _checked_ledger(source, errors)
    expected_files = {
        f"{path.stem}.bindings.yaml"
        for path in GUARANTEE_DIR.glob("*.yaml")
        if _load(path).get("document", {}).get("lifecycle") != "placeholder"
    }
    derived_dir = ROOT / "gdd/derived"
    if derived_dir.is_dir():
        extra = sorted(path.name for path in derived_dir.glob("*.bindings.yaml") if path.name not in expected_files)
        if not update:
            errors.extend(f"gdd/derived/{name}: generated binding has no source ledger" for name in extra)
    else:
        extra = []
    if update and not errors:
        # Derive every selected ledger successfully before touching either side
        # of the SSoT/generated pair. A bad final ledger cannot leave earlier
        # ledgers half-refreshed.
        for source, source_text, target, rendered in planned_writes:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")
        # Obsolete cache files have no authority and are recoverable by
        # regeneration. Removing a ledger must not require a cache re-pin.
        for name in extra:
            (derived_dir / name).unlink()
        _verify_written([source for source, _source_text, _target, _rendered in planned_writes], errors, executor)
    return errors, count, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--ledger", action="append", default=[])
    parser.add_argument("--report-idioms", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=default_jobs(),
        help="worker processes for the pure derivation steps (1 = in-process); the results are identical",
    )
    args = parser.parse_args()
    try:
        errors, count, warnings = check(args.update, set(args.ledger) or None, jobs=args.jobs)
    except (OSError, SyntaxError, ValueError, KeyError, yaml.YAMLError) as exc:
        errors, count, warnings = [str(exc)], 0, []
    if args.report_idioms:
        print("GDD extractor idiom report:")
        if warnings:
            for warning in warnings:
                print(f"  WARN {warning}")
        else:
            print("  no unclassified registered constructs")
    if errors:
        print("GDD case-binding inventory check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    action = "materialized" if args.update else "verified"
    print(f"GDD case-binding inventory {action} ({count} pytest nodes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
