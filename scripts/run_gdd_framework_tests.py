#!/usr/bin/env python3
"""Execute the isolated GDD framework folder in one precision, with outcome records.

The folder runs without the numerical root conftest. Every selected test must
run: a skipped or xfailed test, or an empty selection, fails the run.
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
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FRAMEWORK_FOLDER = "tests/test_gdd_simplification"
PLUGIN = "scripts.run_gdd_framework_tests"


def pytest_addoption(parser):
    """The isolated folder does not import the numerical root conftest."""
    parser.addoption("--precision-mode", choices=("full", "mixed"), default="full")


def pytest_configure(config):
    from gdd.engine.execution_records import register

    register(config)


def framework_command(python: str, precision: str, junit: Path) -> list[str]:
    """Select every current framework test, without a marker or file inventory."""
    if precision not in {"full", "mixed"}:
        raise ValueError("framework precision must be full or mixed")
    return [
        python,
        "-m",
        "pytest",
        "-p",
        PLUGIN,
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        f"--confcutdir={FRAMEWORK_FOLDER}",
        f"--precision-mode={precision}",
        f"--junitxml={junit}",
        "-q",
        FRAMEWORK_FOLDER,
    ]


def execution_environment(
    environ: dict[str, str], *, commit: str, run_id: str, precision: str, directory: Path
) -> dict[str, str]:
    """Pin identity and total selection without disguising an actual MPI rank."""
    from gdd.engine.execution_records import execution_profile

    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("framework execution requires the exact 40-character source SHA")
    # The workflow run id without the attempt: a same-source "re-run failed
    # jobs" recovery reuses records of earlier attempts (issue 340).
    if not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ValueError("framework execution requires the actual workflow run ID")
    if precision not in {"full", "mixed"}:
        raise ValueError("framework precision must be full or mixed")
    if execution_profile(environ) != {"name": "serial", "world_size": 1, "rank": 0}:
        raise ValueError("framework execution requires an actual serial process")
    if environ.get("JQMC_TEST_SHARD_COUNT", "1") != "1" or environ.get("JQMC_TEST_SHARD_INDEX", "0") != "0":
        raise ValueError("framework execution must not inherit a narrowed shard")
    env = dict(environ)
    # The command owns its entire selection. Never inherit -k/-m, collect-only,
    # xdist options or an implicit recorder plugin from another lane.
    env.pop("PYTEST_PLUGINS", None)
    env.update(
        {
            "PYTEST_ADDOPTS": "",
            "PYTHONPATH": str(ROOT),
            "GDD_EXECUTION_MODE": "run",
            "GDD_RECEIPT_AUTHORITY": "1",
            "GDD_RECEIPT_DIR": str(directory.resolve()),
            "GDD_COMMIT_SHA": commit,
            "GDD_RUN_ID": run_id,
            "GDD_BUILD_ID": f"gdd-framework-{commit}",
            "GDD_EVIDENCE_LANE": f"framework-{precision}",
            "GDD_JOB_ID": "gdd-framework-execution",
            "GDD_SHARD_ID": precision,
        }
    )
    return env


def argument_parser() -> argparse.ArgumentParser:
    """The command-line interface of the framework runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("full", "mixed"), required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = argument_parser().parse_args(argv)
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        if head != args.commit:
            raise ValueError(f"framework source HEAD {head} does not match requested {args.commit}")
        env = execution_environment(
            dict(os.environ), commit=args.commit, run_id=args.run_id, precision=args.precision, directory=args.receipt_dir
        )
        if args.receipt_dir.exists() and any(args.receipt_dir.iterdir()):
            raise ValueError("framework receipt directory must be empty for this source/run/attempt/mode")
        # The preflight artifact is restored first. Its input/output stamp is
        # validated against this checkout; absent/stale bindings are rebuilt by
        # the existing source-only generator before any authoritative execution.
        from gdd.engine.derived_cache import prepare_derived

        prepare_derived(ROOT)
        args.junit.parent.mkdir(parents=True, exist_ok=True)
        returncode = subprocess.run(
            framework_command(sys.executable, args.precision, args.junit), cwd=ROOT, env=env, check=False
        ).returncode
        if returncode:
            return returncode
        cases = list(ET.parse(args.junit).iter("testcase"))
        not_run = [f"{case.get('classname')}::{case.get('name')}" for case in cases if case.find("skipped") is not None]
        if not cases or not_run:
            raise ValueError(f"every framework test must run; {len(cases)} collected, not run: {not_run[:20]}")
        return 0
    except (ET.ParseError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"GDD framework execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
