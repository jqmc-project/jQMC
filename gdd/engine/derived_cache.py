"""Rebuildable local bindings cache, never an approval or coverage inventory.

The stamp is only a performance optimization. Missing or changed inputs or
outputs cause regeneration; a failed generator is always an error. Consumers
still validate the generated documents through ``load_ledger``.
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
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


INPUT_ROOTS = ("gdd/ssot", "gdd/engine", "tests", "scripts", "jqmc")
INPUT_FILES = ("pyproject.toml", "pytest.ini", "conftest.py")
IGNORED_PARTS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "target", "build", "dist"})
IGNORED_SUFFIXES = frozenset({".pyc", ".so", ".pyd", ".dylib"})


def _digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _inputs(root: Path) -> str:
    # Include fixture bytes and uncommitted source edits, not merely Git HEAD.
    paths = []
    for relative in INPUT_ROOTS:
        for directory, children, files in os.walk(root / relative):
            # Prune builds before descending: a Rust target directory can be GB.
            children[:] = [name for name in children if name not in IGNORED_PARTS]
            for name in files:
                path = Path(directory) / name
                if path.suffix in IGNORED_SUFFIXES or name == "_version.py":
                    continue
                if path.suffix.lower() == ".md":
                    continue
                paths.append(path)
    paths.extend(root / relative for relative in INPUT_FILES if (root / relative).is_file())
    return _digest(root, paths)


def _outputs(root: Path) -> str:
    return _digest(root, list((root / "gdd/derived").glob("*.bindings.yaml")))


def prepare_derived(root: Path, python: str | None = None) -> bool:
    """Prepare all bindings before a consumer runs; return whether rebuilt.

    Only ``gdd/derived`` is written. This is safe in a read-only *tracked-file*
    check and after a fresh checkout. A filesystem lock prevents concurrent CI
    tools in the same checkout from racing the generator. The existing
    materializer derives all rows before writing and verifies its outputs.
    """
    root = Path(root).resolve()
    cache = root / "gdd/derived"
    cache.mkdir(parents=True, exist_ok=True)
    stamp = cache / ".cache-state.json"
    with (cache / ".cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inputs = _inputs(root)
        try:
            previous = json.loads(stamp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
        # Cache identity must survive different venv paths on CI runners.
        state = {"version": 2, "inputs": inputs, "outputs": _outputs(root), "python": list(sys.version_info[:2])}
        if previous == state:
            return False
        stamp.unlink(missing_ok=True)
        result = subprocess.run(
            [python or sys.executable, "scripts/materialize_gdd_case_bindings.py", "--update"],
            cwd=root,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"derived binding generation failed with exit code {result.returncode}")
        if _inputs(root) != inputs:
            raise RuntimeError("GDD inputs changed during generation; rerun after source edits finish")
        outputs = list(cache.glob("*.bindings.yaml"))
        if not outputs:
            raise RuntimeError("derived binding generation produced no binding files")
        state["outputs"] = _digest(root, outputs)
        temporary = cache / ".cache-state.json.tmp"
        temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(stamp)
        return True


def main() -> int:
    """Prepare the cache for direct checker/renderer invocations in CI."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        rebuilt = prepare_derived(Path(__file__).resolve().parents[2])
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"GDD cache preparation failed: {exc}", file=sys.stderr)
        return 1
    print("GDD bindings rebuilt." if rebuilt else "GDD bindings already current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
