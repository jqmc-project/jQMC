"""Narrow, dependency-free parsing of the jqmc Rust sources for static GDD gates.

This is not a Rust parser. It masks comments and string/char literals (so a
token inside a doc comment or a format string is never matched), then finds
``fn`` items and their brace spans, ``#[test]`` units, PyO3 exports and
route-selecting predicate shapes. Every finder returns source line numbers against
the ORIGINAL text so diagnostics point at real lines. The counter readers return
the symbols of the append-only route-counter arrays in ``gfmc.rs`` in index order.
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

import functools
import re
from dataclasses import dataclass
from pathlib import Path

RUST_ROOT = Path("jqmc/rs/src")
COUNTER_CALL = "gfmc_route_counters"
PREDICATE_COUNTER_CALL = "route_predicate_counters"


_RAW = re.compile(r'r(#*)"')
_CHAR = re.compile(r"'(\\.[^']*|[^'\\\n])'")


def mask(text: str) -> str:
    """Replace comments and string/char literal contents by spaces (newlines kept)."""
    out = list(text)
    i, n = 0, len(text)

    def blank(start: int, end: int) -> None:
        for k in range(start, end):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            end = text.find("\n", i)
            end = n if end < 0 else end
            blank(i, end)
            i = end
        elif text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif c == "r" and _RAW.match(text, i) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            hashes = _RAW.match(text, i).group(1)
            start = i + 2 + len(hashes)
            end = text.find('"' + hashes, start)
            end = n if end < 0 else end + 1 + len(hashes)
            blank(start, end - 1 - len(hashes))
            i = end
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            blank(i + 1, j)
            i = j + 1
        elif c == "'":
            literal = _CHAR.match(text, i)
            if literal:
                blank(i + 1, literal.end() - 1)
                i = literal.end()
            else:
                i += 1  # lifetime
        else:
            i += 1
    return "".join(out)


@dataclass(frozen=True)
class RustFn:
    name: str
    start: int  # offset of ``fn``
    body_start: int  # offset of ``{``
    body_end: int  # offset after the matching ``}``
    line: int


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _brace_pairs(masked: str) -> dict[int, int]:
    pairs: dict[int, int] = {}
    stack: list[int] = []
    for index, ch in enumerate(masked):
        if ch == "{":
            stack.append(index)
        elif ch == "}" and stack:
            pairs[stack.pop()] = index
    return pairs


def functions(masked: str) -> list[RustFn]:
    pairs = _brace_pairs(masked)
    found = []
    for match in re.finditer(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)", masked):
        j = match.end()
        depth = 0
        while j < len(masked):
            ch = masked[j]
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif ch == ";" and depth <= 0:
                j = -1
                break
            elif ch == "{" and depth <= 0:
                break
            j += 1
        if j < 0 or j >= len(masked) or j not in pairs:
            continue
        found.append(RustFn(match.group(1), match.start(), j, pairs[j] + 1, _line(masked, match.start())))
    return found


def enclosing_function(fns: list[RustFn], offset: int) -> str:
    best = None
    for fn in fns:
        if fn.body_start <= offset < fn.body_end and (best is None or fn.body_start > best.body_start):
            best = fn
    return best.name if best else "<module>"


@functools.lru_cache(maxsize=4)
def _rust_files_cached(root: str, stamp: tuple) -> tuple:
    rows = []
    for path in sorted((Path(root) / RUST_ROOT).rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        rows.append((path.relative_to(root).as_posix(), text, mask(text)))
    return tuple(rows)


def rust_files(root: Path) -> list[tuple[str, str, str]]:
    """``(relative path, original text, masked text)`` for every ``.rs`` file."""
    paths = sorted((root / RUST_ROOT).rglob("*.rs"))
    stamp = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    return list(_rust_files_cached(str(root), stamp))


def _skip_space_back(masked: str, end: int) -> int:
    while end > 0 and masked[end - 1].isspace():
        end -= 1
    return end


def _attributes_before(text: str, masked: str, offset: int) -> list[str]:
    """Outer attributes directly preceding the item at ``offset`` (comments skipped).

    Spans are located on the masked text (so doc comments between attributes
    are ignored) and read back from the original text (so string arguments
    such as ``pyo3(name = "...")`` survive).
    """
    attrs = []
    window = max(0, offset - 4000)
    end = _skip_space_back(masked, offset)
    while True:
        word = re.search(r"(?:\bpub\s*(?:\([^()]*\))?|\bunsafe|\basync|\bconst)$", masked[max(window, end - 40) : end])
        if not word:
            break
        end = _skip_space_back(masked, end - len(word.group(0)))
    while end > window and masked[end - 1] == "]":
        start = masked.rfind("#[", window, end)
        if start < 0:
            break
        attrs.insert(0, text[start + 2 : end - 1].strip())
        end = _skip_space_back(masked, start)
    return attrs


def rust_tests(root: Path) -> list[tuple[str, str, int]]:
    """``(path, fn name, line)`` of every ``#[test]`` unit."""
    rows = []
    for relative, _text, masked in rust_files(root):
        for fn in functions(masked):
            if any(attr == "test" for attr in _attributes_before(_text, masked, fn.start)):
                rows.append((relative, fn.name, fn.line))
    return rows


def _pyo3_name(attrs: list[str]) -> str | None:
    for attr in attrs:
        match = re.match(r"(?:pyo3::)?(?:pyo3|pyfunction|pyclass)\s*\((.*)\)\s*$", attr, re.S)
        if match:
            name = re.search(r"\bname\s*=\s*\"([^\"]*)\"", match.group(1))
            if name:
                return name.group(1)
    return None


def _has_attribute(attrs: list[str], name: str) -> bool:
    """Attribute arguments do not change whether an item is a PyO3 export."""
    return any(re.match(rf"(?:pyo3::)?{name}(?:\s*\(|\s*$)", attr) for attr in attrs)


def python_exports(root: Path) -> list[tuple[str, str, int]]:
    """``(python name, path, line)`` of every Python-visible Rust export.

    Module functions are the ``#[pyfunction]`` items registered with
    ``wrap_pyfunction!`` in the ``#[pymodule]``; their Python name is the
    ``#[pyo3(name = ...)]`` value or the Rust name. Methods of every
    ``#[pymethods]`` block are reported as ``Class.method``; classes as
    their ``#[pyclass]`` name. The name masking is done on the original text
    because ``pyo3(name = "...")`` lives in a string.
    """
    rows = []
    registered: set[str] = set()
    for _relative, text, masked in rust_files(root):
        for match in re.finditer(r"wrap_pyfunction!\s*\(\s*([A-Za-z_][A-Za-z0-9_:]*)", masked):
            registered.add(match.group(1).split("::")[-1])
    for relative, text, masked in rust_files(root):
        fns = functions(masked)
        for fn in fns:
            attrs = _attributes_before(text, masked, fn.start)
            if _has_attribute(attrs, "pyfunction") and fn.name in registered:
                rows.append((_pyo3_name(attrs) or fn.name, relative, fn.line))
        class_names: dict[str, str] = {}
        for match in re.finditer(r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)", masked):
            attrs = _attributes_before(text, masked, match.start())
            if _has_attribute(attrs, "pyclass"):
                name = _pyo3_name(attrs) or match.group(1)
                class_names[match.group(1)] = name
                rows.append((name, relative, _line(text, match.start())))
        for match in re.finditer(r"#\[(?:pyo3::)?pymethods\]\s*impl\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", masked):
            cls = match.group(1)
            start = match.end() - 1
            level, k = 0, start
            while k < len(masked):
                if masked[k] == "{":
                    level += 1
                elif masked[k] == "}":
                    level -= 1
                    if level == 0:
                        break
                k += 1
            for fn in fns:
                if fn.start < start or fn.start > k or enclosing_function(fns, fn.start) != "<module>":
                    continue
                attrs = _attributes_before(text, masked, fn.start)
                name = _pyo3_name(attrs) or fn.name
                if name.startswith("__") and name.endswith("__"):
                    continue
                rows.append((f"{class_names.get(cls, cls)}.{name}", relative, fn.line))
    return sorted(rows)


_PREDICATE_FN = re.compile(r"\bfn\s+([a-z_][a-z0-9_]*)\s*(?:<[^>{]*>)?\s*\([^{;]*?\)\s*->\s*(bool|CandidateBasisClass)\b")
_PREDICATE_FN_NAME = re.compile(r"(_ok|_ok_cell)$|^fast_")
_PREDICATE_LET = re.compile(r"\blet\s+(?:mut\s+)?([a-z_][a-z0-9_]*(?:_ok|_fast))\s*(?::\s*bool\s*)?=")
_BOOLEAN_SHAPE = re.compile(r"&&|\|\||is_some_and|map_or|!=|==|(?<![A-Za-z0-9_])!\s*[A-Za-z_(]|is_none\(\)|is_some\(\)")


def masked_route_predicates(relative: str, masked: str) -> list[tuple[str, str, str, int]]:
    """Route-selecting predicate shapes in one masked Rust source.

    The Rust analogue of the Python capability-gate shape
    (tests/test_rs_kernel/test_no_capability_gates.py): (a) a ``fn`` returning
    ``bool`` whose name ends in ``_ok`` / ``_ok_cell`` or starts with ``fast_``,
    or any ``fn`` returning the ``CandidateBasisClass`` capability class;
    (b) a ``let`` binding named ``*_ok`` / ``*_fast`` whose initializer is a
    boolean expression (``&&``, ``||``, ``!``, comparisons, ``is_some_and``,
    ``map_or``, ``is_none()``/``is_some()``). Rows are
    ``(path, owner, name, line)``; ``owner`` is ``"fn"`` for (a) and the
    enclosing function for (b).
    """
    rows = []
    fns = functions(masked)
    for match in _PREDICATE_FN.finditer(masked):
        name, returns = match.group(1), match.group(2)
        if returns == "CandidateBasisClass" or _PREDICATE_FN_NAME.search(name):
            rows.append((relative, "fn", name, _line(masked, match.start())))
    for match in _PREDICATE_LET.finditer(masked):
        end = masked.find(";", match.end())
        initializer = masked[match.end() : end if end >= 0 else len(masked)]
        if _BOOLEAN_SHAPE.search(initializer):
            owner = enclosing_function(fns, match.start())
            rows.append((relative, owner, match.group(1), _line(masked, match.start())))
    return rows


def route_predicates(root: Path) -> list[tuple[str, str, str, int]]:
    rows = []
    for relative, _text, masked in rust_files(root):
        rows.extend(masked_route_predicates(relative, masked))
    return rows


def _rust_counter_array_symbols(root: Path, function: str, prefix: str) -> list[str]:
    path = root / RUST_ROOT / "gfmc.rs"
    tree = path.read_text(encoding="utf-8")
    # Rust is intentionally parsed narrowly: the append-only public array is
    # the production authority and every element has ``NAME.load(...)`` form.
    signature = re.search(rf"pub fn {function}\(\) -> \[u64;\s*(\d+)\]\s*\{{", tree)
    if signature is None:
        raise ValueError(f"Rust {function} append-only authority was not found")
    declared_count = int(signature.group(1))
    start = signature.start()
    array_start = tree.find("[", tree.find("{", start))
    array_end = tree.find("\n    ]", array_start)
    if array_start < 0 or array_end < 0:
        raise ValueError(f"Rust {function} return array could not be parsed")
    names = re.findall(rf"\b({prefix}[A-Z0-9_]+)\.load\(Ordering::Relaxed\)", tree[array_start:array_end])
    if len(names) != declared_count:
        raise ValueError(
            f"Rust {function} declared length does not match its append-only array: "
            f"declared={declared_count}, parsed={len(names)}"
        )
    return names


def rust_counter_symbols(root: Path) -> list[str]:
    return _rust_counter_array_symbols(root, COUNTER_CALL, "GFMC_")


def rust_predicate_counter_symbols(root: Path) -> list[str]:
    """Symbols of ``route_predicate_counters()``, in index order."""
    return _rust_counter_array_symbols(root, PREDICATE_COUNTER_CALL, "RS_")
