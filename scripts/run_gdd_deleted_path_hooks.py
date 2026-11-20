#!/usr/bin/env python3
"""Run path-filtered GDD hooks that a pure deletion would otherwise skip.

The static GDD
gate hooks run only when a relevant path changes (``always_run: false`` plus a
``files`` regex, ``gdd.engine.hook_coverage``). pre-commit selects hook inputs
from staged files with ``--diff-filter=ACMRTUXB``, so a commit whose only
change to a hook's inputs is a deletion (or a rename away from them) would skip
that hook. This always-run hook closes that gap: for every local, path-filtered,
``pass_filenames: false`` hook of the pre-commit stage whose filter selects a
staged deletion (renames counted as delete + add) but no other staged path, it
runs the hook's entry itself and fails if that command fails. With no staged
deletion it returns at once.
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

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdd.engine.hook_coverage import DELETED_PATH_HOOK_ID, selects  # noqa: E402

CONFIG = ".pre-commit-config.yaml"


def _staged(*filters: str) -> list[str]:
    command = ["git", "diff", "--staged", "--name-only", "--no-ext-diff", "-z", *filters]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    return [path for path in result.stdout.split("\0") if path]


def hooks_to_run(config: dict[str, Any], deleted: list[str], present: list[str]) -> list[dict[str, Any]]:
    """Path-filtered local hooks selected only by a staged deletion."""
    default_stages = config.get("default_stages", ["pre-commit"])
    selected = []
    for repo in config.get("repos") or ():
        if not isinstance(repo, dict) or repo.get("repo") != "local":
            continue
        for hook in repo.get("hooks") or ():
            if (
                not isinstance(hook, dict)
                or hook.get("id") == DELETED_PATH_HOOK_ID
                or hook.get("always_run") is True
                or hook.get("pass_filenames") is not False
                or not isinstance(hook.get("files"), str)
                or "pre-commit" not in hook.get("stages", default_stages)
            ):
                continue
            if any(selects(hook, path) for path in deleted) and not any(selects(hook, path) for path in present):
                selected.append(hook)
    return selected


def main() -> int:
    deleted = _staged("--no-renames", "--diff-filter=D")
    if not deleted:
        return 0
    # The same selection pre-commit itself uses for hook inputs.
    present = _staged("--diff-filter=ACMRTUXB")
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    failed = []
    for hook in hooks_to_run(config, deleted, present):
        entry = shlex.split(hook["entry"])
        if entry[0] in {"python", "python3"}:
            entry[0] = sys.executable
        print(f"{DELETED_PATH_HOOK_ID}: staged deletion selects {hook['id']}; running {hook['entry']}", flush=True)
        if subprocess.run(entry, cwd=ROOT, check=False).returncode:
            failed.append(hook["id"])
    if failed:
        print(f"{DELETED_PATH_HOOK_ID}: failed hooks: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
