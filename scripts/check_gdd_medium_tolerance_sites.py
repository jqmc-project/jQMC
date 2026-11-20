#!/usr/bin/env python3
"""Reject unreviewed test call sites of the ``medium`` XLA-Rust tolerance tier.

``jqmc/_setting.py`` forbids the ``medium`` envelope for ordinary leaf-kernel
parity; this check enforces which tests select it. Every
call of ``get_tolerance_xla_rust_consistency`` under ``tests/`` whose level can
be ``"medium"`` (an unresolved level counts) must be an exact entry of the
reviewed inventory ``gdd/ssot/inventories/medium_tolerance.yaml`` with a
reviewed class. Entries of a migration class (``leaf-kernel``,
``wrong-family``) are frozen current uses: they may shrink, not grow.
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

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdd.engine import medium_tolerance_allowlist as allowlist  # noqa: E402
from gdd.engine.medium_tolerance_sites import medium_sites  # noqa: E402


def validation_errors(root: Path = ROOT, reviewed: dict[str, tuple[str, str]] | None = None) -> list[str]:
    inventory = allowlist.load_medium_tolerance_allowlist(root)
    reviewed = inventory["MEDIUM_SITES"] if reviewed is None else reviewed
    errors: list[str] = []
    found = {f"{path}::{qualname}": lines for (path, qualname), lines in medium_sites(root).items()}
    for key in sorted(set(found) - set(reviewed)):
        errors.append(
            f"medium XLA-Rust tolerance used at an unreviewed site {key} (lines {found[key]}); "
            "leaf-kernel parity must use the strict tier"
        )
    for key in sorted(set(reviewed) - set(found)):
        errors.append(f"stale medium-tolerance allow-list entry {key}: no medium-capable call remains there")
    vocabulary = inventory["ALLOWED_CLASSES"] | inventory["MIGRATION_CLASSES"]
    for key, (klass, reason) in sorted(reviewed.items()):
        if klass not in vocabulary:
            errors.append(f"medium-tolerance allow-list entry {key} has unreviewed class {klass!r}")
        if not str(reason).strip():
            errors.append(f"medium-tolerance allow-list entry {key} has no reason")
    return errors


def main() -> int:
    try:
        inventory = allowlist.load_medium_tolerance_allowlist(ROOT)
        errors = validation_errors()
    except (OSError, SyntaxError, ValueError) as exc:
        errors = [str(exc)]
    if errors:
        print("GDD medium-tolerance site contract failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    reviewed = inventory["MEDIUM_SITES"]
    migrate = sum(klass in inventory["MIGRATION_CLASSES"] for klass, _ in reviewed.values())
    print(
        "GDD medium-tolerance site contract passed "
        f"({len(reviewed)} reviewed sites; {migrate} frozen leaf/wrong-family uses pending migration)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
