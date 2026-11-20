"""Bounded source check for numerical budgets in Rust test code.

This is a lexer and a small assertion/alias analysis, not a Rust compiler.
Strings, nested comments and raw strings cannot create assertions. Only test
items and their helpers are inspected; production algorithm coefficients and
test inputs are outside the budget rule. Exact equality and zero bounds stay
allowed. Like the Python checker, arbitrary imported helper behavior is outside
the provenance proof.
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

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    value: str
    line: int


_NUMBER = re.compile(
    r"(?:0[xX][\da-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|\d[\d_]*(?:\.(?!\.)[\d_]*)?(?:[eE][+-]?[\d_]+)?)(?:_?(?:f32|f64|u\d+|i\d+|usize|isize))?"
)
_IDENT = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*")
_TOL_WORD = re.compile(r"(?:^|_)(?:tol|atol|rtol|eps|epsilon|threshold|tolerance)(?:_|$)", re.I)
_ERROR_WORD = re.compile(r"(?:^|_)(?:err|error|resid|residual|diff|delta|deviation|worst|mismatch|maxabs)(?:_|$)", re.I)
_RAW = re.compile(r'(?:br|cr|r)(#*)"')
_CHAR = re.compile(r"(?:b)?'(?:\\(?:u\{[\da-fA-F_]+\}|x[\da-fA-F]{2}|.)|[^'\\\n])'")


def tokens(source: str) -> list[Token]:
    """Tokenize relevant Rust syntax while retaining physical line numbers."""
    result = []
    pos = 0
    line = 1
    while pos < len(source):
        start = pos
        if source[pos].isspace():
            pos += 1
        elif source.startswith("//", pos):
            end = source.find("\n", pos)
            pos = len(source) if end < 0 else end
        elif source.startswith("/*", pos):
            level = 1
            pos += 2
            while pos < len(source) and level:
                if source.startswith("/*", pos):
                    level += 1
                    pos += 2
                elif source.startswith("*/", pos):
                    level -= 1
                    pos += 2
                else:
                    pos += 1
            if level:
                raise ValueError(f"unterminated Rust block comment at line {line}")
        elif raw := _RAW.match(source, pos):
            terminator = '"' + raw[1]
            end = source.find(terminator, raw.end())
            if end < 0:
                raise ValueError(f"unterminated Rust raw string at line {line}")
            pos = end + len(terminator)
            result.append(Token("STRING", line))
        elif source[pos] == '"' or source.startswith(('b"', 'c"'), pos):
            pos += 1 if source[pos] == '"' else 2
            while pos < len(source) and source[pos] != '"':
                pos += 2 if source[pos] == "\\" else 1
            if pos >= len(source):
                raise ValueError(f"unterminated Rust string at line {line}")
            pos += 1
            result.append(Token("STRING", line))
        elif char := _CHAR.match(source, pos):
            pos = char.end()
            result.append(Token("CHAR", line))
        elif number := _NUMBER.match(source, pos):
            pos = number.end()
            result.append(Token(number[0], line))
        elif ident := _IDENT.match(source, pos):
            pos = ident.end()
            result.append(Token(ident[0], line))
        else:
            item = next(
                (op for op in ("<=", ">=", "==", "!=", "&&", "||", "::", "->", "=>", "..") if source.startswith(op, pos)),
                source[pos],
            )
            pos += len(item)
            result.append(Token(item, line))
        line += source[start:pos].count("\n")
    return result


def _pairs(items: list[Token]) -> dict[int, int]:
    stack = []
    result = {}
    for i, item in enumerate(items):
        if item.value in ("(", "[", "{"):
            stack.append(i)
        elif item.value in (")", "]", "}"):
            if not stack or items[stack[-1]].value != {")": "(", "]": "[", "}": "{"}[item.value]:
                raise ValueError(f"unbalanced Rust group at line {item.line}")
            opening = stack.pop()
            result[opening] = i
            result[i] = opening
    if stack:
        raise ValueError(f"unclosed Rust group at line {items[stack[-1]].line}")
    return result


def _split(items: list[Token], separator: str) -> list[list[Token]]:
    out = []
    start = 0
    depth = 0
    for i, item in enumerate(items):
        if item.value in ("(", "[", "{"):
            depth += 1
        elif item.value in (")", "]", "}"):
            depth -= 1
        elif not depth and item.value == separator:
            out.append(items[start:i])
            start = i + 1
    out.append(items[start:])
    return out


def _numeric(item: Token) -> float | None:
    text = re.sub(r"_?(?:f32|f64|u\d+|i\d+|usize|isize)$", "", item.value).replace("_", "")
    if not _NUMBER.fullmatch(item.value):
        return None
    try:
        return float(int(text, 0)) if text.lower().startswith(("0x", "0o", "0b")) else float(text)
    except ValueError:
        return None


def _test_ranges(items: list[Token], pairs: dict[int, int]) -> list[tuple[int, int]]:
    ranges = []
    for i in range(len(items) - 2):
        if [x.value for x in items[i : i + 2]] != ["#", "["]:
            continue
        end = pairs[i + 1]
        attribute = [x.value for x in items[i + 2 : end]]
        if not (attribute == ["test"] or (attribute[:2] == ["cfg", "("] and "test" in attribute)):
            continue
        j = end + 1
        # Other attributes may intervene between #[cfg(test)] and the item.
        while j < len(items) - 1 and items[j].value == "#" and items[j + 1].value == "[":
            j = pairs[j + 1] + 1
        while j < len(items) and items[j].value not in ("{", ";"):
            j += 1
        if j < len(items) and items[j].value == "{":
            ranges.append((j, pairs[j]))
    return ranges


def _bindings(items: list[Token]) -> dict[str, list[Token]]:
    bindings = {}
    pairs = {}
    stack = []
    for position, token in enumerate(items):
        if token.value in ("(", "[", "{"):
            stack.append(position)
        elif token.value in (")", "]", "}") and stack:
            opening = stack.pop()
            pairs[position] = opening
    for i, item in enumerate(items[:-1]):
        if item.value != "=":
            continue
        j = i + 1
        depth = 0
        while j < len(items):
            if items[j].value in ("(", "[", "{"):
                depth += 1
            elif items[j].value in (")", "]", "}"):
                depth -= 1
            elif items[j].value == ";" and depth == 0:
                break
            j += 1
        value = items[i + 1 : j]
        if i and items[i - 1].value == ")":
            opening = pairs[i - 1]
            if opening and items[opening - 1].value == "let":
                names = _split(items[opening + 1 : i - 1], ",")
                values = _split(value[1:-1], ",") if value and value[0].value == "(" and value[-1].value == ")" else []
                if len(names) == len(values):
                    for name, element in zip(names, values):
                        identifiers = [token.value for token in name if token.value not in ("mut", "ref")]
                        if len(identifiers) == 1 and _IDENT.fullmatch(identifiers[0]):
                            bindings[identifiers[0]] = element
            continue
        start = i - 1
        while start >= 0 and items[start].value not in (";", "{", "}"):
            start -= 1
        left = items[start + 1 : i]
        if left and left[0].value in ("let", "const", "static"):
            left = left[1:]
            while left and left[0].value in ("mut", "ref"):
                left = left[1:]
            left = _split(left, ":")[0]
        if len(left) == 2 and _IDENT.fullmatch(left[0].value) and left[1].value in ("+", "-", "*", "/", "%"):
            # The lexer keeps `+=` etc. as an operator and `=`. Preserve the
            # old budget's provenance when applying an augmented assignment.
            previous = bindings.get(left[0].value, [left[0]])
            bindings[left[0].value] = previous + [left[1]] + value
        elif len(left) == 1 and _IDENT.fullmatch(left[0].value):
            bindings[left[0].value] = value
    return bindings


def _from_central(items: list[Token], bindings: dict[str, list[Token]], seen: frozenset[str] = frozenset()) -> bool:
    for item in items:
        if item.value in ("setting_consts", "central_setting_f64"):
            return True
        if (
            item.value in bindings
            and item.value not in seen
            and _from_central(bindings[item.value], bindings, seen | {item.value})
        ):
            return True
    return False


def _central_imports(items: list[Token]) -> dict[str, list[Token]]:
    """Keep ordinary `use ...setting_consts::{NAME as ALIAS}` provenance."""
    imports = {}
    for i, item in enumerate(items):
        if item.value != "use":
            continue
        end = i + 1
        while end < len(items) and items[end].value != ";":
            end += 1
        declaration = items[i + 1 : end]
        central = next((j for j, token in enumerate(declaration) if token.value == "setting_consts"), None)
        if central is not None:
            for token in declaration[central + 1 :]:
                if _IDENT.fullmatch(token.value) and token.value not in ("as", "self"):
                    imports[token.value] = [Token("central_setting_f64", token.line)]
    return imports


def _residual(items: list[Token]) -> bool:
    return any(item.value in ("abs", "norm", "norm_sqr") or _ERROR_WORD.search(item.value) for item in items)


def _local_budget(
    items: list[Token],
    bindings: dict[str, list[Token]],
    helpers: dict[str, list[Token]],
    seen: frozenset[str] = frozenset(),
    allow_scale: bool = False,
) -> bool:
    allow_scale = allow_scale or _from_central(items, bindings)
    # The common reference scale `1.max(x.abs()).max(y.abs())` contains a
    # unit floor, not a local error budget. Other numeric factors are retained.
    numeric = [value for item in items if (value := _numeric(item)) is not None and value != 0]
    if (
        allow_scale
        and numeric
        and all(value == 1 for value in numeric)
        and any(item.value == "abs" for item in items)
        and any(item.value == "max" for item in items)
        and not any(_TOL_WORD.search(item.value) for item in items)
    ):
        return False
    magnitudes = set()
    if allow_scale:
        # In `atol + rtol * reference.abs()` the reference is data, even when
        # its initializer contains literals. Follow budget aliases, not that
        # independent reference construction. A bare `local_bound.abs()` has
        # no central budget and still resolves to its literal definition.
        for method, token in enumerate(items):
            if token.value not in ("abs", "norm", "norm_sqr") or method < 2 or items[method - 1].value != ".":
                continue
            if [item.value for item in items[method + 1 : method + 3]] != ["(", ")"]:
                continue
            start = method - 2
            if items[start].value in ("]", ")"):
                closing = items[start].value
                opening = {"[": "]", "(": ")"}
                depth = 1
                j = start - 1
                while j >= 0 and depth:
                    if items[j].value == closing:
                        depth += 1
                    elif opening.get(items[j].value) == closing:
                        depth -= 1
                    j -= 1
                start = j + 1
                if start > 0 and _IDENT.fullmatch(items[start - 1].value):
                    start -= 1
            receiver = items[start : method - 1]
            if (
                receiver
                and not any(_TOL_WORD.search(item.value) for item in receiver)
                and any(_IDENT.fullmatch(item.value) for item in receiver)
            ):
                magnitudes.update(range(start, method + 3))
    i = 0
    while i < len(items):
        item = items[i]
        if i in magnitudes:
            i += 1
            continue
        # Reference magnitudes keep their conventional max(1) scale floor.
        if (
            allow_scale
            and item.value == "max"
            and i
            and items[i - 1].value == "."
            and i + 3 < len(items)
            and items[i + 1].value == "("
            and _numeric(items[i + 2]) == 1
            and items[i + 3].value == ")"
            and any(token.value == "abs" for token in items[:i])
        ):
            i += 4
            continue
        value = _numeric(item)
        if (
            allow_scale
            and value == 1
            and i + 1 < len(items)
            and items[i + 1].value == "+"
            and any(token.value == "abs" for token in items[i + 2 :])
        ):
            i += 1
            continue
        if value is not None and value != 0:
            return True
        if item.value not in seen:
            if item.value in bindings and _local_budget(
                bindings[item.value], bindings, helpers, seen | {item.value}, allow_scale
            ):
                return True
            if item.value in helpers and _local_budget(
                helpers[item.value], bindings, helpers, seen | {item.value}, allow_scale
            ):
                return True
        i += 1
    return False


def scan_rust(source: str) -> list[tuple[int, str]]:
    """Return line/reason for local test budgets, preserving exact comparisons."""
    if not re.search(r"#\s*\[\s*(?:cfg\s*\([^)]*\btest\b|test\s*\])", source):
        return []
    items = tokens(source)
    pairs = _pairs(items)
    scopes = _test_ranges(items, pairs)
    findings = set()
    # Same-file constants may live outside the cfg(test) module. Inspect their
    # provenance only when they reach a test assertion's numerical budget.
    constant_items = []
    for i, item in enumerate(items):
        if item.value in ("const", "static"):
            j = i
            while j < len(items) and items[j].value not in (";", "{"):
                j += 1
            if j < len(items) and items[j].value == ";":
                constant_items.extend(items[i : j + 1])
    file_constants = _central_imports(items) | _bindings(constant_items)
    # A module range includes its ordinary helper functions, not just #[test]s.
    for begin, end in scopes:
        scope = items[begin + 1 : end]
        local_pairs = _pairs(scope)
        functions = []
        budget_arguments = {}
        return_helpers = {}
        formal_budgets = {}
        for i, item in enumerate(scope[:-2]):
            if item.value != "fn" or scope[i + 2].value != "(":
                continue
            name = scope[i + 1].value
            args_end = local_pairs[i + 2]
            args = _split(scope[i + 3 : args_end], ",")
            budget_arguments[name] = [n for n, arg in enumerate(args) if arg and _TOL_WORD.search(arg[0].value)]
            body = args_end + 1
            while body < len(scope) and scope[body].value not in ("{", ";"):
                body += 1
            if body < len(scope) and scope[body].value == "{":
                body_end = local_pairs[body]
                functions.append((body, body_end))
                # Calls of these local helpers are checked below, so their
                # formal budget parameters may multiply reference scales.
                formal_budgets[body] = {
                    args[n][0].value: [Token("central_setting_f64", item.line)] for n in budget_arguments[name]
                }
                numeric_return = [token.value for token in scope[args_end + 1 : body]] in (["->", "f64"], ["->", "f32"])
                if _TOL_WORD.search(name) or numeric_return or (not any(args) and len(scope[body + 1 : body_end]) <= 3):
                    return_helpers[name] = scope[body + 1 : body_end]
        module_bindings = file_constants | _bindings(
            [item for i, item in enumerate(scope) if not any(a < i < b for a, b in functions)]
        )
        for i, item in enumerate(scope[:-2]):
            assertion = item.value == "assert" and scope[i + 1].value == "!" and scope[i + 2].value in ("(", "{", "[")
            helper_call = scope[i + 1].value == "(" and item.value in budget_arguments and (not i or scope[i - 1].value != "fn")
            if not assertion and not helper_call:
                continue
            # Function-local aliases override module names; other functions'
            # inputs do not leak into this assertion's budget provenance.
            function = min(((a, b) for a, b in functions if a < i < b), key=lambda pair: pair[1] - pair[0], default=None)
            bindings = (
                module_bindings | formal_budgets[function[0]] | _bindings(scope[function[0] + 1 : i])
                if function
                else module_bindings
            )
            if assertion:
                closing = local_pairs[i + 2]
                condition = _split(scope[i + 3 : closing], ",")[0]
                for part in _split(condition, "&&"):
                    part_pairs = _pairs(part)
                    for j, operator in enumerate(part):
                        if operator.value not in ("<", "<=", ">", ">="):
                            continue
                        left, right = part[:j], part[j + 1 :]
                        negations = sum(
                            a < j < b and a > 0 and part[a - 1].value == "!" for a, b in part_pairs.items() if a < b
                        )
                        upper = (operator.value in ("<", "<=")) != bool(negations % 2)
                        residual, budget = (left, right) if upper else (right, left)
                        # The small-literal fallback catches an unnamed bound;
                        # named residuals additionally reject large relaxations.
                        small = any((value := _numeric(token)) is not None and 0 < abs(value) <= 1e-3 for token in budget)
                        if (
                            _residual(residual) or small or any(_TOL_WORD.search(token.value) for token in budget)
                        ) and _local_budget(budget, bindings, return_helpers):
                            findings.add((operator.line, "Rust asserted numerical budget must come from jqmc/_setting.py"))
            if helper_call:
                args = _split(scope[i + 2 : local_pairs[i + 1]], ",")
                for n in budget_arguments[item.value]:
                    if n < len(args) and _local_budget(args[n], bindings, return_helpers):
                        findings.add(
                            (args[n][0].line, f"Rust helper {item.value} tolerance argument must come from jqmc/_setting.py")
                        )
    return sorted(findings)
