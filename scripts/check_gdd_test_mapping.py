#!/usr/bin/env python3
"""Check authored test/guarantee edges without counts, pins, or derived files.

The default is a fast source check. For actual collection closure, collect each
CI precision with ``-p gdd.engine.mapping --gdd-collection-output=PATH`` and pass
the resulting reports through repeatable ``--collection-json PATH`` arguments.
Function mapping does not replace required parameter-case/precision completion.
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
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gdd.engine.mapping import (
    authored_mapping,
    collected_json_nodes,
    collection_manifest_nodes,
    cargo_list_nodes,
    mapping_errors,
    native_test_nodes,
    static_collected_nodes,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--collection-json", type=Path, action="append", default=[])
    parser.add_argument("--collection-dir", type=Path, help="Directory containing gdd-collection-*.json execution manifests")
    parser.add_argument("--native-list", type=Path, help="Optional cargo test -- --list output (otherwise source discovery)")
    args = parser.parse_args(argv)
    try:
        relation = authored_mapping(args.root)
        registered = set().union(*relation.values())
        if args.collection_dir:
            collected = collection_manifest_nodes(args.collection_dir, registered=registered)
        elif args.collection_json:
            collected = collected_json_nodes(args.collection_json, registered=registered)
        else:
            collected = static_collected_nodes(args.root, registered=registered)
        collected.update(cargo_list_nodes(args.native_list) if args.native_list else native_test_nodes(args.root))
        errors = mapping_errors(relation, collected)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"GDD mapping failed: {exc}", file=sys.stderr)
        return 1
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"GDD mapping: {len(relation)} guarantees, {len(collected)} test definitions; both directions complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
