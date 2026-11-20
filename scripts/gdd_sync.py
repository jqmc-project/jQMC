#!/usr/bin/env python3
"""Refresh derived YAML bindings and validate authored contracts without display outputs."""

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
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gdd.engine.derived_cache import prepare_derived


def _run(label: str, command: list[str], rows: list[tuple[str, str]]) -> None:
    start = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, check=False)
    rows.append((label, f"{time.monotonic() - start:.1f}s"))
    if completed.returncode:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="prepare derived YAML bindings and validate authored contracts")
    mode.add_argument("--update", action="store_true", help="refresh and validate derived YAML bindings (the default)")
    # Shared helper edits always require global freshness, even when a caller
    # supplies --ledger.
    parser.add_argument("--ledger", action="append", default=[], help="compatibility option; all bindings stay current")
    parser.parse_args()
    rows: list[tuple[str, str]] = []
    try:
        started = time.monotonic()
        rebuilt = prepare_derived(ROOT, sys.executable)
        rows.append(("bindings rebuilt" if rebuilt else "bindings already current", f"{time.monotonic() - started:.1f}s"))
        with tempfile.TemporaryDirectory(prefix="gdd-sync-") as temporary:
            _run(
                "preflight-gates",
                [
                    sys.executable,
                    "scripts/run_gdd_preflight.py",
                    "--commit-sha",
                    "working-tree",
                    "--certificate",
                    str(Path(temporary) / "certificate.json"),
                    "--gates-only",
                    "--skip-materializer",
                ],
                rows,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"GDD sync failed: {exc}", file=sys.stderr)
        return 1
    print("step | elapsed\n--- | ---")
    for label, duration in rows:
        print(f"{label} | {duration}")
    print("GDD sync complete. Test execution and completion checking remain separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
