"""Qualified Python references to Rust exports (not bare-name matches).

A bare-name census would credit a dead export such as
``jqmc_rs.grad_J3_wrt_R`` whenever a pure-Python function of the same name is
called. A reference here is an attribute access ``<expr>.<name>`` or a string
constant equal to ``<name>`` (``getattr(jqmc_rs, "<name>")``), including the
same-module concatenation ``"<stem>" + "<suffix>"`` form (``getattr(ctx, name +
"_batch")`` with ``"<stem>"`` a string constant of that module); a plain call of
a same-named Python function is NOT a reference. For methods
(``Class.method``) the method name is matched.
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
from pathlib import Path


def referenced_names(paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        strings: set[str] = set()
        suffixes: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.isidentifier():
                strings.add(node.value)
            elif (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Add)
                and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, str)
                and node.right.value.isidentifier()
            ):
                suffixes.add(node.right.value)
        names |= strings
        names |= {stem + suffix for stem in strings for suffix in suffixes}
    return names


def reference_sets(root: Path) -> tuple[set[str], set[str]]:
    """``(production names, test names)`` referenced from ``jqmc/`` and ``tests/``."""
    production = sorted(path for path in (root / "jqmc").rglob("*.py"))
    tests = sorted(path for path in (root / "tests").rglob("*.py"))
    return referenced_names(production), referenced_names(tests)


def export_status(export: str, production: set[str], tests: set[str]) -> str:
    short = export.rsplit(".", 1)[-1]
    in_production = short in production
    in_tests = short in tests
    if in_tests and in_production:
        return "tested"
    if in_tests:
        return "test-only"
    if in_production:
        return "production-only"
    return "unused"
