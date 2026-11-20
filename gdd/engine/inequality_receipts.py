"""Hand-written numerical inequalities as receipt-bearing comparison sites.

A hand-written ``assert`` inequality is not a comparator call, so the call
tracer in ``tolerance_runtime`` cannot observe its operands or tolerance. This
module recognizes a small, reviewed set of source forms and replays exactly
that form on the operand and tolerance values captured from the test frame when
the comparison expression is entered. No test-source ``eval``, object method or
production callable is executed; the replay uses NumPy on captured arrays.

Recognized forms (``<=`` or ``<``; tolerances are local names; operands are
names, exact builtin-dict lookups with str/int keys, or numeric real/imag parts):

- ``abs-deviation``:      ``abs(A - B) <= atol + rtol * abs(B)``
- ``max-abs-deviation``:  ``np.abs(A - B).max() <= atol + rtol * np.abs(B).max()``
  (``np.max(np.abs(...))`` / ``np.amax`` spellings are the same form)
- ``scaled-*``: the same bound with ``max(1., magnitude(B))`` or the full
  complex reference magnitude when the deviation uses its real/imag part.

Explicit builtin float conversion of magnitudes is replayed. The source index
may substitute an immediately preceding straight-line deviation/scale alias;
the receipt additionally requires its captured value to equal the replay.
Other reference scales, slices, custom mappings/properties and nonfinite values
are rejected. Direct helper-call substitution retains its names-only grammar.

``np.abs``/``np.absolute`` and builtin ``abs`` are accepted for the magnitude.
Every other spelling is left unrecognized: it keeps its ordinary
``numerical-bound`` observation and cannot be registered as an executable
comparison (fail closed). This is not a claim that unrecognized inequalities
are wrong, only that they are not yet receipt-certified.
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
import copy
from functools import lru_cache
import hashlib
import math
from pathlib import Path
from types import CodeType
from typing import Any
from collections.abc import Callable


COMPARATOR = "python.inequality"
SOURCE_COMPARATOR = {"kind": "python-inequality", "name": COMPARATOR}
SITE_CATEGORY = "inequality"
FORMS = ("abs-deviation", "max-abs-deviation", "scaled-abs-deviation", "scaled-max-abs-deviation")
_OPERATORS = {ast.Lt: "<", ast.LtE: "<="}
_ABS = {"numpy.abs", "numpy.absolute", "builtins.abs"}
_MAX = {"numpy.max", "numpy.amax"}
EVALUATED = "evaluated"


@lru_cache(maxsize=128)
def helper_code_candidates(filename: str, name: str, line: int, syntax_sha256: str) -> tuple[CodeType, ...]:
    """Compile, never execute, the reviewed helper, including pytest rewriting.

    File/name/line matching alone would accept a substituted function whose
    metadata imitates the original. Actual bytecode/constants must also match
    source with the exact recorded AST identity. Only pytest's ordinary
    assertion rewriting is an additional accepted compilation mode.
    """
    from gdd.engine.tolerance_provenance import _syntax

    try:
        source = Path(filename).read_bytes()
        tree = ast.parse(source, filename=filename)
        helpers = [
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name and node.lineno == line
        ]
        if len(helpers) != 1 or hashlib.sha256(_syntax(helpers[0]).encode()).hexdigest() != syntax_sha256:
            return ()
        trees = [tree]
        try:
            from _pytest.assertion.rewrite import rewrite_asserts

            rewritten = copy.deepcopy(tree)
            rewrite_asserts(rewritten, source, module_path=filename)
            trees.append(rewritten)
        except ImportError:
            pass
        candidates = []
        for variant in trees:
            compiled = compile(variant, filename, "exec", dont_inherit=True)
            candidates.extend(
                code
                for code in compiled.co_consts
                if isinstance(code, CodeType) and code.co_name == name and code.co_firstlineno == line
            )
        return tuple(candidates)
    except (OSError, SyntaxError, ValueError, TypeError):
        return ()


def _same(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _call(node: ast.AST, names: set[str], resolve: Callable[[ast.AST], str | None]) -> ast.expr | None:
    """Return the single positional argument of a call to one of ``names``."""
    if (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and not isinstance(node.args[0], ast.Starred)
        and resolve(node.func) in names
    ):
        return node.args[0]
    return None


def _magnitude(node: ast.AST, resolve) -> tuple[bool, ast.expr] | None:
    """``abs(X)`` -> (False, X); ``np.abs(X).max()`` or ``np.max(np.abs(X))`` -> (True, X)."""
    converted = _call(node, {"builtins.float"}, resolve)
    if converted is not None:
        return _magnitude(converted, resolve)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "max"
        and not node.args
        and not node.keywords
    ):
        inner = _call(node.func.value, _ABS, resolve)
        return None if inner is None else (True, inner)
    reduced = _call(node, _MAX, resolve)
    if reduced is not None:
        inner = _call(reduced, _ABS, resolve)
        return None if inner is None else (True, inner)
    inner = _call(node, _ABS, resolve)
    return None if inner is None else (False, inner)


def _operand(node: ast.AST) -> bool:
    """Restricted capture grammar; subscripts are exact builtin dictionaries at runtime."""
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return node.attr in {"real", "imag"} and _operand(node.value)
    return (
        isinstance(node, ast.Subscript)
        and _operand(node.value)
        and (
            isinstance(node.slice, ast.Name) or (isinstance(node.slice, ast.Constant) and type(node.slice.value) in (str, int))
        )
    )


def recognize(test: ast.AST, resolve: Callable[[ast.AST], str | None]) -> dict[str, Any] | None:
    """Recognize one reviewed inequality form, or return None (never a guess)."""
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or type(test.ops[0]) not in _OPERATORS:
        return None
    deviation = _magnitude(test.left, resolve)
    if deviation is None or not isinstance(deviation[1], ast.BinOp) or not isinstance(deviation[1].op, ast.Sub):
        return None
    actual, reference = deviation[1].left, deviation[1].right
    bound = test.comparators[0]
    if not (
        isinstance(bound, ast.BinOp)
        and isinstance(bound.op, ast.Add)
        and isinstance(bound.left, ast.Name)
        and isinstance(bound.right, ast.BinOp)
        and isinstance(bound.right.op, ast.Mult)
        and isinstance(bound.right.left, ast.Name)
    ):
        return None
    scale_node = bound.right.right
    floor = False
    if (
        isinstance(scale_node, ast.Call)
        and resolve(scale_node.func) == "builtins.max"
        and len(scale_node.args) == 2
        and not scale_node.keywords
        and isinstance(scale_node.args[0], ast.Constant)
        and type(scale_node.args[0].value) in (int, float)
        and scale_node.args[0].value == 1
    ):
        floor, scale_node = True, scale_node.args[1]
    scale = _magnitude(scale_node, resolve)
    if scale is None or scale[0] != deviation[0] or not _operand(scale[1]):
        return None
    same_reference = _same(scale[1], reference)
    component_reference = isinstance(reference, ast.Attribute) and _same(scale[1], reference.value)
    if not same_reference and not component_reference:
        return None
    atol, rtol = bound.left, bound.right.left
    if not _operand(actual) or not _operand(reference):
        return None
    names = [ast.unparse(node) for node in (actual, reference, atol, rtol)]
    if len(set(names)) != 4:
        return None
    result = {
        "form": "max-abs-deviation" if deviation[0] else "abs-deviation",
        "operator": _OPERATORS[type(test.ops[0])],
        "actual": actual,
        "reference": reference,
        "atol": atol,
        "rtol": rtol,
    }
    if _call(test.left, {"builtins.float"}, resolve) is not None:
        result["deviation_float"] = True
    if _call(scale_node, {"builtins.float"}, resolve) is not None:
        result["scale_float"] = True
    if floor or not same_reference:
        result["form"] = "scaled-" + result["form"]
        result["scale_reference"] = scale[1]
        result["scale_floor"] = floor
    return result


def _numeric(value: Any) -> bool:
    """Reject user conversion/property hooks before NumPy or attribute access."""
    import numpy as np

    if type(value) in (int, float, complex, bool, np.ndarray) or type(value) in set(np.sctypeDict.values()):
        return True
    if type(value).__module__ in {"jaxlib._jax", "jaxlib.xla_extension"}:
        import sys

        module = sys.modules.get(type(value).__module__)
        return module is not None and type(value) is getattr(module, "ArrayImpl", None)
    return False


def _read(node: ast.AST, local: dict[str, Any], global_: dict[str, Any]) -> Any:
    """Read the small capture grammar without eval, custom lookup or properties."""
    if isinstance(node, ast.Name):
        if node.id in local:
            return local[node.id]
        if node.id in global_:
            return global_[node.id]
        raise ValueError(f"unobserved name: {node.id}")
    if isinstance(node, ast.Subscript):
        mapping = _read(node.value, local, global_)
        key = node.slice.value if isinstance(node.slice, ast.Constant) else _read(node.slice, local, global_)
        # Reject custom key hashing/equality too, even in an otherwise plain dict.
        if type(mapping) is not dict or type(key) not in (str, int):
            raise ValueError("only builtin dict and str/int keys are capturable")
        if any(type(existing) not in (str, int) for existing in mapping):
            raise ValueError("dictionary has custom keys")
        return mapping[key]
    if isinstance(node, ast.Attribute) and node.attr in {"real", "imag"}:
        value = _read(node.value, local, global_)
        if not _numeric(value):
            raise ValueError("numeric component requires a trusted numeric value")
        # Access only NumPy's own component descriptors, never a user property.
        import numpy as np

        array = np.asarray(value)
        return array.real if node.attr == "real" else array.imag
    raise ValueError("unsupported operand expression")


def _indexed_address(node: ast.AST, local: dict[str, Any], global_: dict[str, Any]) -> tuple | None:
    """Identify an already validated dict slot, catching two key aliases of itself."""
    if isinstance(node, ast.Subscript):
        key = node.slice.value if isinstance(node.slice, ast.Constant) else _read(node.slice, local, global_)
        return (id(_read(node.value, local, global_)), key)
    if isinstance(node, ast.Attribute):
        base = _indexed_address(node.value, local, global_)
        return None if base is None else (*base, node.attr)
    return None


def _scalar(value: Any) -> Any:
    import numpy as np

    if _numeric(value) and isinstance(value, (np.floating, np.integer)) and not isinstance(value, np.bool_):
        value = value.item()
    if type(value) in (int, float) and math.isfinite(value):
        return value
    return {"invalid_type": type(value).__name__}


def operand_check(
    form: str,
    operator: str,
    actual: Any,
    reference: Any,
    atol: Any,
    rtol: Any,
    *,
    scale_reference: Any = None,
    scale_floor: bool = False,
    deviation_float: bool = False,
    scale_float: bool = False,
    aliases: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay the recognized source form on captured values, without scalar broadcasting."""
    # Imported lazily: static inventory/rendering of this module must not
    # import numerical libraries.
    import numpy as np

    result: dict[str, Any] = {}
    try:
        if not _numeric(actual) or not _numeric(reference):
            result["operand_check"] = "unsupported-operands"
            return result
        # The same array object on both sides compares a value with itself.
        # (Identity of cached Python scalars carries no such meaning.)
        if actual is reference and isinstance(actual, np.ndarray):
            result["operand_check"] = "aliased-operands"
            return result
        a, b = np.asarray(actual), np.asarray(reference)
        result["operand_shapes"] = {"actual": list(a.shape), "reference": list(b.shape)}
        result["operand_dtypes"] = {"actual": str(a.dtype), "reference": str(b.dtype)}
        if (
            form not in FORMS
            or operator not in _OPERATORS.values()
            or a.shape != b.shape
            or a.dtype.kind not in "biufc"
            or b.dtype.kind not in "biufc"
            or (form in {"abs-deviation", "scaled-abs-deviation"} and a.ndim != 0)
        ):
            result["operand_check"] = "invalid-shape-or-dtype"
            return result
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in (atol, rtol)):
            result["operand_check"] = "invalid-tolerance"
            return result
        if not a.size:
            result["operand_check"] = "empty"
            return result
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            result["operand_check"] = "nan"
            return result
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            result["operand_check"] = "nonfinite"
            return result
        deviation = np.abs(a - b)
        if form.startswith("scaled-"):
            if not _numeric(scale_reference):
                result["operand_check"] = "unsupported-operands"
                return result
            scaling = np.asarray(scale_reference)
            if scaling.shape != b.shape or scaling.dtype.kind not in "biufc" or not np.all(np.isfinite(scaling)):
                result["operand_check"] = "invalid-scale"
                return result
        else:
            scaling = b
        scale = np.abs(scaling)
        if form in {"max-abs-deviation", "scaled-max-abs-deviation"}:
            deviation, scale = deviation.max(), scale.max()
        if deviation_float:
            deviation = float(deviation)
        if scale_float:
            scale = float(scale)
        if scale_floor:
            scale = max(1.0, scale)
        for role, replayed in (("deviation", deviation), ("scale", scale)):
            if aliases is not None and role in aliases:
                observed = _scalar(aliases[role])
                if isinstance(observed, dict) or observed != replayed:
                    result["operand_check"] = "stale-expression-alias"
                    return result
        bound = atol + rtol * scale
        holds = deviation <= bound if operator == "<=" else deviation < bound
        result["evaluated"] = {"deviation": float(deviation), "bound": float(bound)}
        result["operand_check"] = "passed" if bool(holds) else "value-mismatch"
    except (TypeError, ValueError, OverflowError):
        result["operand_check"] = "unsupported-operands"
    return result


def capture(spec: dict[str, Any], local: dict[str, Any], global_: dict[str, Any]) -> dict[str, Any]:
    """Build the receipt call for one entered inequality from the actual frame values."""
    symbols = spec.get("symbols", {})
    call: dict[str, Any] = {"comparator": COMPARATOR, "arguments": {}, "outcome": EVALUATED}
    values = {}
    addresses = {}
    roles = ["actual", "reference", "atol", "rtol"]
    if spec.get("form", "").startswith("scaled-"):
        roles.append("scale_reference")
    for role in roles:
        name = symbols.get(role)
        if not isinstance(name, str):
            call["capture_error"] = f"inequality site has no {role} symbol"
            call["operand_check"] = "unsupported-operands"
            return call
        try:
            node = ast.parse(name, mode="eval").body
            if not _operand(node) or (role in {"atol", "rtol"} and not isinstance(node, ast.Name)):
                raise ValueError("unsupported capture grammar")
            values[role] = _read(node, local, global_)
            if role in {"actual", "reference"}:
                addresses[role] = _indexed_address(node, local, global_)
        except (KeyError, ValueError, SyntaxError, TypeError) as exc:
            call["capture_error"] = f"inequality {role} was not observed: {name}: {exc}"
            call["operand_check"] = "unsupported-operands"
            return call
    if addresses["actual"] is not None and addresses["actual"] == addresses["reference"]:
        call["operand_check"] = "aliased-operands"
        return call
    call["arguments"] = {name: _scalar(values[name]) for name in ("atol", "rtol")}
    aliases = {}
    for role, name in spec.get("expression_aliases", {}).items():
        if role not in {"deviation", "scale"} or not isinstance(name, str) or not name.isidentifier():
            call["operand_check"] = "unsupported-operands"
            return call
        try:
            aliases[role] = _read(ast.Name(id=name), local, global_)
        except ValueError:
            call["operand_check"] = "unsupported-operands"
            return call
    call.update(
        operand_check(
            spec.get("form"),
            spec.get("operator"),
            values["actual"],
            values["reference"],
            call["arguments"]["atol"],
            call["arguments"]["rtol"],
            scale_reference=values.get("scale_reference"),
            scale_floor=spec.get("scale_floor", False),
            deviation_float=spec.get("deviation_float", False),
            scale_float=spec.get("scale_float", False),
            aliases=aliases,
        )
    )
    return call
