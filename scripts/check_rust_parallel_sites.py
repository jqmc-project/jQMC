#!/usr/bin/env python3
"""Inventory the reviewed Rust parallel boundary, without importing native code.

This is a source-token guard, not a Rust compiler or a purity proof. It scans
all src/**/*.rs, ignores comments/literals and directly #[cfg(test)]-gated
items, and conservatively scans other cfg branches and local macro bodies.
External/procedural macro expansion, generated/out-of-tree sources, Cargo
dependency renames, arbitrary third-party parallel APIs and state shared by
walker closures are outside this guard. Those need separate review/runtime
evidence. Every par_* identifier is conservatively considered a parallel site,
even through UFCS or a function value; unrelated new names require review.
The existing global-pool configuration body's tokens are pinned to prevent its
approved builder path becoming a local-pool work-dispatch bypass. This is a
structural pin: comments and string/character literal contents are not pinned.
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
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("jqmc/rs/src")
WALKER_BODY = "py.allow_threads(|| (0..nw).into_par_iter().map(f).collect())"
# Reviewed global-pool initialization only. Do not derive this from the source
# being checked: building/returning a local pool would enable new dispatch sites.
CONFIGURE_BODY = r"""
    let requested = n.or_else(|| {
        std::env::var("JQMC_RS_NUM_THREADS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
    });
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(k) = requested {
        builder = builder.num_threads(k);
    }
    match builder.build_global() {
        Ok(()) => Ok(()),
        Err(e) => {
            let live = rayon::current_num_threads();
            match requested {
                None => Ok(()),
                Some(k) if k == live => Ok(()),
                Some(k) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "jqmc_rs.configure_threads({k}): the rayon global pool is already \
                     initialized with {live} threads and cannot be resized. \
                     build_global runs once per process -- call configure_threads BEFORE \
                     the first parallel kernel. rayon: {e}"
                ))),
            }
        }
    }
"""
NATIVE_CALLS = {
    "py_native_vmc": "run",
    "py_native_vmc_complex": "run_complex",
}
# These are configuration/introspection, NOT additional work-dispatch sites.
RAYON_PATHS = {
    ("walker_batch.rs", "<module>"): "rayon::prelude::*;",
    ("lib.rs", "py_configure_threads"): "rayon::ThreadPoolBuilder::new(",
    ("lib.rs", "py_current_num_threads"): "rayon::current_num_threads(",
}
EXPECTED = Counter(
    {
        ("walker_batch.rs", "<module>", "rayon::prelude"): 1,
        ("walker_batch.rs", "for_each_walker", "into_par_iter"): 1,
        ("walker_batch.rs", "for_each_walker", "allow_threads"): 1,
        ("lib.rs", "py_configure_threads", "rayon::ThreadPoolBuilder"): 1,
        ("lib.rs", "py_configure_threads", "rayon::current_num_threads"): 1,
        ("lib.rs", "py_current_num_threads", "rayon::current_num_threads"): 1,
        ("native_driver.rs", "py_native_vmc", "allow_threads"): 1,
        ("native_driver.rs", "py_native_vmc_complex", "allow_threads"): 1,
    }
)
TOKEN = re.compile(r"r#[A-Za-z_][A-Za-z_0-9]*|[A-Za-z_][A-Za-z_0-9]*|::|[^\s]")
CHAR = re.compile(r"'(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|.)|[^\\'\n])'")
RAW = re.compile(r'(?:br|cr|r)(#*)"')


def _tokens(source: str) -> tuple[list[str], list[int]]:
    """Mask Rust nested comments/strings/chars; retain offsets for diagnostics."""
    masked = list(source)
    i = 0
    while i < len(source):
        start = i
        if source.startswith("//", i):
            end = source.find("\n", i)
            i = len(source) if end < 0 else end
        elif source.startswith("/*", i):
            depth = 1
            i += 2
            while i < len(source) and depth:
                if source.startswith("/*", i):
                    depth += 1
                    i += 2
                elif source.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth:
                raise ValueError("unterminated block comment")
        elif raw := RAW.match(source, i):
            terminator = '"' + raw[1]
            end = source.find(terminator, raw.end())
            if end < 0:
                raise ValueError("unterminated raw string")
            i = end + len(terminator)
        elif source[i] == '"':
            i += 1
            while i < len(source) and source[i] != '"':
                i += 2 if source[i] == "\\" else 1
            if i >= len(source):
                raise ValueError("unterminated string")
            i += 1
        elif char := CHAR.match(source, i):
            i = char.end()
        else:
            i += 1
            continue
        masked[start:i] = ["\n" if c == "\n" else " " for c in source[start:i]]
    matches = list(TOKEN.finditer("".join(masked)))
    return [m[0].removeprefix("r#") for m in matches], [m.start() for m in matches]


def _pairs(tokens: list[str]) -> dict[int, int]:
    stack = []
    pairs = {}
    opening = {"(": ")", "[": "]", "{": "}"}
    for i, token in enumerate(tokens):
        if token in opening:
            stack.append(i)
        elif token in opening.values():
            if not stack or opening[tokens[stack[-1]]] != token:
                raise ValueError("unbalanced source delimiters")
            pairs[stack.pop()] = i
    if stack:
        raise ValueError("unclosed source delimiters")
    return pairs


def _item_end(tokens: list[str], pairs: dict[int, int], start: int) -> int:
    i = start
    while i < len(tokens):
        if tokens[i] == "{":
            return pairs[i] + 1
        if tokens[i] == ";":
            return i + 1
        i = pairs[i] + 1 if tokens[i] in {"(", "["} else i + 1
    raise ValueError("cannot locate item boundary")


def _production_tokens(source: str) -> tuple[list[str], list[int]]:
    tokens, offsets = _tokens(source)
    pairs = _pairs(tokens)
    excluded = set()
    cfg_test = ["#", "[", "cfg", "(", "test", ")", "]"]
    for i in range(len(tokens)):
        if tokens[i : i + len(cfg_test)] == cfg_test:
            excluded.update(range(i, _item_end(tokens, pairs, i + len(cfg_test))))
    return (
        [t for i, t in enumerate(tokens) if i not in excluded],
        [p for i, p in enumerate(offsets) if i not in excluded],
    )


def _functions(tokens: list[str], pairs: dict[int, int]) -> list[tuple[str, int, int]]:
    functions = []
    for i in range(len(tokens) - 1):
        if tokens[i] != "fn" or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", tokens[i + 1]):
            continue
        end = _item_end(tokens, pairs, i + 2)
        # A body ends in }, unlike trait declarations ending in ;.
        if tokens[end - 1] == "}":
            begin = next(k for k in pairs if pairs[k] == end - 1)
            functions.append((tokens[i + 1], begin, end - 1))
    return functions


def validation_errors(root: Path = ROOT) -> list[str]:
    errors = []
    found = Counter()
    files = sorted((root / SOURCE).rglob("*.rs"))
    if not files:
        return [f"RAYON-INVENTORY: no Rust source files under {SOURCE}"]
    for path in files:
        relative = path.relative_to(root / SOURCE).as_posix()
        source = path.read_text(encoding="utf-8")
        try:
            tokens, offsets = _production_tokens(source)
            pairs = _pairs(tokens)
            functions = _functions(tokens, pairs)
        except ValueError as exc:
            errors.append(f"RAYON-SOURCE-SYNTAX: {relative}: {exc}")
            continue
        for name, begin, end in functions:
            if relative == "walker_batch.rs" and name == "for_each_walker":
                if tokens[begin + 1 : end] != _tokens(WALKER_BODY)[0]:
                    errors.append("RAYON-WALKER-BODY: walker_batch.rs::for_each_walker ordered map/collect body changed")
            if relative == "lib.rs" and name == "py_configure_threads":
                if tokens[begin + 1 : end] != _tokens(CONFIGURE_BODY)[0]:
                    errors.append("RAYON-CONFIGURE-BODY: lib.rs::py_configure_threads global-pool configuration changed")
        for i, token in enumerate(tokens):
            if not token.startswith("par_") and token not in {"into_par_iter", "rayon", "rayon_core", "allow_threads"}:
                continue
            owners = [(name, start) for name, start, end in functions if start < i < end]
            function = max(owners, key=lambda entry: entry[1])[0] if owners else "<module>"
            location = (relative, function)
            line = source.count("\n", 0, offsets[i]) + 1
            diagnostic = f"{relative}:{line}::{function}"
            if token == "rayon":
                allowed = RAYON_PATHS.get(location)
                if location == ("lib.rs", "py_configure_threads") and tokens[i + 2 : i + 3] == ["current_num_threads"]:
                    allowed = "rayon::current_num_threads("
                expected = _tokens(allowed)[0] if allowed else []
                if not expected or tokens[i : i + len(expected)] != expected:
                    errors.append(f"RAYON-SITE: {diagnostic}: unreviewed rayon path/import/alias")
                else:
                    found[(*location, "rayon::" + tokens[i + 2])] += 1
            elif token == "into_par_iter" and location == ("walker_batch.rs", "for_each_walker"):
                found[(*location, token)] += 1
            elif token == "allow_threads":
                if location == ("walker_batch.rs", "for_each_walker"):
                    found[(*location, token)] += 1
                elif relative == "native_driver.rs" and function in NATIVE_CALLS:
                    # Pin the existing serial release closure, not a whole driver.
                    call = NATIVE_CALLS[function]
                    expected = _tokens(
                        f"allow_threads(|| {{ {call}(&terms, seed, dt, epsilon, moves_per_block, n_warmup, n_blocks,) }})"
                    )[0]
                    if tokens[i : i + len(expected)] != expected:
                        errors.append(f"GIL-RELEASE-BODY: {diagnostic}: expected direct serial {call} closure")
                    found[(*location, token)] += 1
                else:
                    errors.append(f"GIL-RELEASE-SITE: {diagnostic}: unreviewed allow_threads site")
            else:
                errors.append(f"RAYON-SITE: {diagnostic}: unreviewed parallel entry {token}")
    if found != EXPECTED:
        errors.append(f"RAYON-INVENTORY: missing={dict(EXPECTED - found)}; extra={dict(found - EXPECTED)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    errors = validation_errors(args.root)
    if errors:
        print("\n".join(errors))
        return 1
    print("Rust parallel source inventory passed (one ordered walker site; three GIL-release sites).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
