#!/usr/bin/env python3
"""Check authored guarantees, required coverage and actual test collection.

The checks run are an explicit list: a check_gdd_* script that is not in it
does not run.
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
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GDD_CONTRACT_TESTS = ("tests/test_gdd_simplification",)
# The small replacement framework suite imports no numerical production code.
# Scientific tests and the legacy unit suites run in their own test jobs.
STATIC_PREFLIGHT_PLUGIN = "scripts.gdd_static_preflight"
GDD_CONTRACT_PYTEST_ARGS = (
    "-q",
    "-o",
    "addopts=",
    "--confcutdir=tests/test_gdd_simplification",
)

RETAINED_CHECKERS = (
    "check_gdd_test_mapping",
    "check_gdd_required_cases",
    "check_gdd_wavefunction_fixtures_contract",
    "check_gdd_guarantee_fixtures",
    "check_gdd_derivatives_contract",
    # Derivative registry 1:1 with the tests' recorder call sites (issue 789).
    "check_gdd_derivative_registry",
    "check_gdd_module_contracts",
    # Guarantee class, ID prefix and verdict-method legality on every ledger.
    "check_gdd_module_ledgers",
    # Rust tests, exports and reviewed predicate classifications, checked on
    # every preflight so deterministic drift cannot reach main (issues 177, 181).
    "check_gdd_rust_inventory",
    # Every test function registered, exempt by a pinned declaration, or reviewed pending
    # (issue 180).
    "check_gdd_orphan_tests",
    # Issue 829: the gate that runs the checkers of these three guarded
    # integrity properties.
    # A medium XLA-Rust tolerance only at a reviewed site.
    "check_gdd_medium_tolerance_sites",
    # Every production module and symbol owned by a guarantee ledger.
    "check_gdd_symbol_ownership",
    # Every authored numerical comparison live at its bound source site.
    "check_gdd_bound_comparisons",
    # Every SSoT document matches its versioned schema and names real enforcers.
    "check_gdd_ssot_schema",
    # The external pymatgen / TurboRVB reference literals are still present.
    "check_gdd_external_references",
    # The derivative matrix exclusions equal the reviewed exclusion table.
    "check_gdd_derivative_exclusions",
)


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def seal_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(certificate)
    sealed.pop("sha256", None)
    sealed["sha256"] = hashlib.sha256(_canonical_json(sealed).encode()).hexdigest()
    return sealed


def verify_sealed_certificate(certificate: dict[str, Any]) -> None:
    payload = dict(certificate)
    checksum = payload.pop("sha256", None)
    expected = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    if checksum != expected:
        raise ValueError("certificate checksum mismatch")


def contract_snapshot_sha256(root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    # Markdown is display-only, never part of machine-readable contract identity.
    contract_roots = (root / "gdd/ssot",)
    paths = sorted(
        path
        for contract_root in contract_roots
        for path in contract_root.rglob("*")
        if path.is_file() and path.suffix.lower() != ".md"
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def preflight_commands(
    root: Path = ROOT,
    python: str = sys.executable,
    *,
    include_pytest: bool = True,
    include_materializer: bool = True,
) -> list[tuple[str, list[str]]]:
    # The tolerance-source rule is an existing project contract, not only a
    # lint convention. Its checker lacks the check_gdd_ prefix, so include it
    # explicitly before the expensive derived-binding freshness check.
    commands = [("check_no_inline_tolerances", [python, "scripts/check_no_inline_tolerances.py"])]
    if include_materializer:
        commands.append(
            (
                "materialize_gdd_case_bindings",
                [python, "-m", "gdd.engine.derived_cache"],
            )
        )
    commands.extend((name, [python, f"scripts/{name}.py"]) for name in RETAINED_CHECKERS)
    if include_pytest:
        for precision in ("full", "mixed"):
            commands.append(
                (
                    f"gdd_collection_{precision}",
                    [
                        python,
                        "-m",
                        "pytest",
                        "--collect-only",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "-o",
                        "addopts=",
                        f"--precision-mode={precision}",
                        "tests",
                    ],
                )
            )
        for world_size, files in sorted(mpi_collection_targets(root, python).items()):
            for precision in ("full", "mixed"):
                commands.append(
                    (
                        f"gdd_collection_mpi{world_size}_{precision}",
                        [
                            "mpirun",
                            "-np",
                            str(world_size),
                            python,
                            "-m",
                            "pytest",
                            "--collect-only",
                            "-q",
                            "-p",
                            "no:cacheprovider",
                            "-o",
                            "addopts=",
                            f"--precision-mode={precision}",
                            *files,
                        ],
                    )
                )
        commands.append(
            (
                "gdd_actual_test_mapping",
                [
                    python,
                    "scripts/check_gdd_test_mapping.py",
                    "--collection-dir",
                    "gdd-expected-cases",
                    *(["--native-list", os.environ["GDD_NATIVE_TEST_LIST"]] if os.environ.get("GDD_NATIVE_TEST_LIST") else []),
                ],
            )
        )
        commands.append(
            (
                "gdd_actual_required_cases",
                [python, "scripts/check_gdd_required_cases.py", "--collection-dir", "gdd-expected-cases"],
            )
        )
        commands.append(
            (
                "gdd_contract_pytest",
                [python, "-m", "pytest", *GDD_CONTRACT_PYTEST_ARGS, *GDD_CONTRACT_TESTS],
            )
        )
    return commands


def mpi_collection_targets(root: Path = ROOT, python: str = sys.executable) -> dict[int, list[str]]:
    """Test files with a registered case whose execution profile needs an MPI world, by world size.

    Read from the case bindings the required-case check validates (authored
    ``required_execution_profiles`` and the profiles the binding derives from the
    test source), so every MPI tuple it requires is collected in its own world.
    """
    from gdd.engine.derived_cache import prepare_derived
    from scripts.check_gdd_required_cases import authored_contracts

    prepare_derived(root, python)
    targets: dict[int, set[str]] = {}
    for _label, node, binding in authored_contracts(root):
        for profile in binding.get("execution", {}).get("profiles", ()):
            size = int(profile.get("world_size", 1))
            if size > 1:
                targets.setdefault(size, set()).add(node["pytest_node"].split("::", 1)[0])
    return {size: sorted(files) for size, files in targets.items()}


def gdd_meta_test_paths(root: Path = ROOT) -> list[str]:
    """Return every current GDD framework/meta-test source in stable order."""
    paths = sorted((root / "tests/test_jqmc_kernel").glob("test_gdd*.py"))
    paths.extend(sorted((root / "tests/test_gdd_simplification").rglob("test*.py")))
    if not paths:
        raise ValueError("no GDD meta-test files found")
    return [path.relative_to(root).as_posix() for path in paths]


def gdd_meta_test_command(
    root: Path,
    python: str,
    *,
    precision: str,
    junit: Path,
) -> list[str]:
    """Build the unnarrowed PR structure-check meta-test command for one precision."""
    if precision not in {"full", "mixed"}:
        raise ValueError(f"unsupported precision for GDD meta-tests: {precision}")
    return [
        python,
        "-m",
        "pytest",
        "-q",
        "-n",
        "4",
        "--max-worker-restart=0",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        f"--precision-mode={precision}",
        f"--junitxml={junit}",
        *gdd_meta_test_paths(root),
    ]


def pr_collection_commands(
    root: Path = ROOT,
    python: str = sys.executable,
) -> list[tuple[str, list[str]]]:
    """Reuse the Heavy preflight's materialization and actual-collection subset."""
    return [
        (check_id, command)
        for check_id, command in preflight_commands(root, python)
        if check_id == "materialize_gdd_case_bindings"
        or check_id.startswith("gdd_collection_")
        or check_id in {"gdd_actual_test_mapping", "gdd_actual_required_cases"}
    ]


def preflight_check_environment(check_id: str) -> dict[str, str]:
    """Pin actual binding validation without inherited pytest selection or xdist."""
    if check_id.startswith("gdd_collection_"):
        return {
            "GDD_EXECUTION_MODE": "collect",
            "GDD_EXECUTION_CONTRACT_SCOPE": "1",
            "GDD_RECEIPT_DIR": "gdd-expected-cases",
            "GDD_RECEIPT_AUTHORITY": "1",
            "PYTEST_ADDOPTS": "",
        }
    if check_id == "gdd_contract_pytest":
        return {"PYTEST_ADDOPTS": ""}
    return {}


def _ledger_lifecycles(root: Path = ROOT) -> dict[str, list[str]]:
    grouped = {"placeholder": [], "pre-cutover": [], "cutover": []}
    from gdd.engine.guarantee_store import load_module

    for path in sorted((root / "gdd/ssot/guarantees").glob("*.yaml")):
        document = load_module(path)
        lifecycle = document.get("document", {}).get("lifecycle")
        if lifecycle not in grouped:
            raise ValueError(f"{path.relative_to(root)} has invalid lifecycle {lifecycle!r}")
        grouped[lifecycle].append(path.relative_to(root).as_posix())
    return grouped


def _repository_lifecycle(root: Path = ROOT) -> str:
    contract = yaml.safe_load((root / "gdd/ssot/symbol_ownership.yaml").read_text(encoding="utf-8"))
    lifecycle = contract.get("lifecycle")
    if lifecycle not in {"pre-cutover", "cutover"}:
        raise ValueError(f"invalid repository lifecycle {lifecycle!r}")
    return lifecycle


def _run_checks(
    *,
    root: Path,
    commit_sha: str,
    commands: list[tuple[str, list[str]]],
) -> list[dict[str, Any]]:
    """Run an ordered GDD command set without inherited pytest shard filters."""
    results = []
    child_env = {key: value for key, value in os.environ.items() if not key.startswith("JQMC_TEST_SHARD_")}
    child_env["GDD_COMMIT_SHA"] = commit_sha
    child_env["PYTHONPATH"] = str(root)
    for check_id, command in commands:
        check_env = preflight_check_environment(check_id)
        started = time.monotonic()
        print(f"GDD preflight START {check_id}", flush=True)
        completed = subprocess.run(
            command,
            cwd=root,
            env={**child_env, **check_env},
            check=False,
            text=True,
            # Inherit the parent's streams: a job killed at its wall-clock cap
            # must not lose every diagnostic since this child started.
            capture_output=False,
        )
        if completed.returncode != 0:
            print(f"GDD preflight FAILED {check_id} after {time.monotonic() - started:.1f}s", flush=True)
            raise RuntimeError(f"GDD preflight check failed: {check_id}")
        print(f"GDD preflight PASSED {check_id} in {time.monotonic() - started:.1f}s", flush=True)
        results.append(
            {
                "id": check_id,
                "command": command[1:],
                "environment": check_env,
                "status": "passed",
            }
        )
    return results


def run_preflight(
    *,
    root: Path = ROOT,
    python: str = sys.executable,
    commit_sha: str,
    include_pytest: bool = True,
    include_materializer: bool = True,
) -> dict[str, Any]:
    results = _run_checks(
        root=root,
        commit_sha=commit_sha,
        commands=preflight_commands(
            root,
            python,
            include_pytest=include_pytest,
            include_materializer=include_materializer,
        ),
    )

    return seal_certificate(
        {
            "schema_version": 1,
            "certificate_kind": "gdd-preflight",
            "status": "complete",
            "commit_sha": commit_sha,
            "repository_lifecycle": _repository_lifecycle(root),
            "ledger_lifecycles": _ledger_lifecycles(root),
            "contract_snapshot_sha256": contract_snapshot_sha256(root),
            "checks": results,
        }
    )


def run_pr_collection(
    *,
    root: Path = ROOT,
    python: str = sys.executable,
    commit_sha: str,
) -> dict[str, Any]:
    """Collect both precisions and validate their realized GDD IDs for the PR structure check."""
    results = _run_checks(
        root=root,
        commit_sha=commit_sha,
        commands=pr_collection_commands(root, python),
    )
    return seal_certificate(
        {
            "schema_version": 1,
            "certificate_kind": "gdd-pr-collection",
            "status": "complete",
            "commit_sha": commit_sha,
            "contract_snapshot_sha256": contract_snapshot_sha256(root),
            "checks": results,
        }
    )


def run_pr_meta_tests(
    *,
    root: Path = ROOT,
    python: str = sys.executable,
    precision: str,
    junit: Path,
    shard_index: int = 0,
    shard_count: int = 1,
) -> int:
    """Execute one deterministic meta-test shard and reject vacuous XML."""
    if shard_count < 1:
        raise ValueError("GDD meta-test shard count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"GDD meta-test shard index {shard_index} is outside [0, {shard_count})")
    completed = subprocess.run(
        gdd_meta_test_command(root, python, precision=precision, junit=junit),
        cwd=root,
        env={
            **os.environ,
            "GDD_RECEIPT_AUTHORITY": "0",
            "JQMC_TEST_SHARD_INDEX": str(shard_index),
            "JQMC_TEST_SHARD_COUNT": str(shard_count),
            "PYTEST_ADDOPTS": "",
        },
        check=False,
        text=True,
        capture_output=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"GDD meta-tests failed for {precision} shard {shard_index}/{shard_count}")
    cases = list(ET.parse(junit).iter("testcase"))
    executed = [case for case in cases if case.find("skipped") is None]
    if not executed:
        raise RuntimeError(f"GDD meta-tests executed no non-skipped tests for {precision} shard {shard_index}/{shard_count}")
    if any(case.find("failure") is not None or case.find("error") is not None for case in executed):
        raise RuntimeError(f"GDD meta-test JUnit records failures for {precision} shard {shard_index}/{shard_count}")
    print(
        f"GDD meta-tests PASSED {precision} shard {shard_index}/{shard_count}: "
        f"{len(executed)} executed, {len(cases) - len(executed)} skipped"
    )
    return len(executed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gates-only", action="store_true", help="run static gates without contract pytest")
    mode.add_argument("--pr-collection", action="store_true", help="run only actual PR collection validation")
    mode.add_argument("--pr-meta", choices=("full", "mixed"), help="run the complete PR GDD meta-test corpus")
    parser.add_argument(
        "--skip-materializer",
        action="store_true",
        help="internal: materialization was just completed in the same gdd_sync transaction",
    )
    args = parser.parse_args()
    try:
        if args.pr_meta:
            if args.junit is None:
                raise ValueError("--pr-meta requires --junit")
            run_pr_meta_tests(
                precision=args.pr_meta,
                junit=args.junit,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
            return 0
        if args.certificate is None:
            raise ValueError("preflight and collection modes require --certificate")
        if args.pr_collection:
            certificate = run_pr_collection(commit_sha=args.commit_sha)
        else:
            certificate = run_preflight(
                commit_sha=args.commit_sha,
                include_pytest=not args.gates_only,
                include_materializer=not args.skip_materializer,
            )
    except (ET.ParseError, OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        print(f"GDD preflight failed: {exc}", file=sys.stderr)
        return 1
    encoded = json.dumps(certificate, sort_keys=True, indent=2) + "\n"
    args.certificate.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
