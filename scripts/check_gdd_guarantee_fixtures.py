#!/usr/bin/env python3
"""Validate direct fixture bindings derived from current guarantee evidence.

Guarantees name their evidence nodes; the binding materializer derives concrete
inputs, sources, electron configurations, and wavefunction scope from those
nodes. This gate therefore validates the direct consumer map instead of a
second, hand-maintained union of fixture profiles on every guarantee.
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


def canonical_consumer_errors(root: Path = ROOT) -> list[str]:
    """Return errors in the direct evidence-to-fixture consumer map."""
    from gdd.engine.canonical_consumers import consumer_map

    _rows, errors = consumer_map(root)
    return [f"canonical consumer map: {error}" for error in errors]


def check() -> tuple[list[str], int]:
    from gdd.engine.canonical_consumers import consumer_map

    rows, errors = consumer_map(ROOT)
    return [f"canonical consumer map: {error}" for error in errors], len(rows)


def main() -> int:
    errors, count = check()
    if errors:
        print("GDD direct fixture-binding check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"GDD direct fixture-binding check passed ({count} consumed evidence patterns).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
