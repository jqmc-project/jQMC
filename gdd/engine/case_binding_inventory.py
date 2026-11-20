"""Source-derived, reviewable fixture bindings for governed pytest nodes.

The generated bindings are materialized below ``gdd/derived``.  This module is
the deterministic authoring/checking implementation: it reads test syntax but
never imports pytest, jqmc, JAX, or a compiled extension.
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
from collections.abc import Iterator
import contextlib
import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache, cache
from itertools import product
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "gdd/ssot/fixtures/sources.yaml"
sys.path.insert(0, str(ROOT / "scripts"))

from check_gdd_module_ledgers import _import_bindings, _literal, _module_constants, _ParamValue
from gdd.engine.comparison_paths import (
    EXECUTION_SCOPE_KEYS,
    UNTRACED_COMPARISON_SCOPES,
    assertion_guard,
    conditional_execution,
    function_scope,
    nonrecovering_try,
)
from gdd.engine.precision_exclusions import (
    ADVERSARIAL_CHAIN_FULL_ONLY_DECISION,
    ADVERSARIAL_CHAIN_FULL_ONLY_KIND,
    ADVERSARIAL_CHAIN_FULL_ONLY_REASON,
    EXTERNAL_FP64_FULL_ONLY_DECISION,
    EXTERNAL_FP64_FULL_ONLY_KIND,
    EXTERNAL_FP64_FULL_ONLY_REASON,
    adversarial_chain_full_only_case,
    external_fp64_full_only,
)
from gdd.engine.pinned_fixtures import PIN_MANIFEST, QE_EXAMPLE_ROOT_HELPER, fixture_digest, fixture_exists
from gdd.engine.tolerance_provenance import ToleranceSourceIndex

_COORDINATE_WORDS = ("electron", "walker", "coords", "r_up", "r_dn", "r_cart", "r_elec")
_RANDOM_WORDS = ("random", "rng", "uniform", "normal", "permutation", "condition_target")
_CONSTRUCTION_WORDS = (
    "build",
    "make",
    "create",
    "setup",
    "load",
    "read",
    "convert",
    "generate",
    "fixture",
    "hamiltonian",
    "wavefunction",
    "jastrow",
    "geminal",
    "orbital",
)
_EXPLICIT_CONSTRUCTION_SYMBOLS = ("_sample_lrdmc_rotation_matrix",)
_PHYSICS_SYMBOL_WORDS = (
    "AOs_",
    "MOs_",
    "SD_data",
    "AGP",
    "Geminal_data",
    "Jastrow",
    "Wavefunction",
    "Hamiltonian",
    "Coulomb",
    "ECP",
    "MCMC",
    "GFMC",
)
_WAVEFUNCTION_KINETIC_APIS = (
    "compute_kinetic_energy_all_elements",
    "compute_kinetic_energy_all_elements_fast_update",
    "_init_kinetic_energy_all_elements_streaming_state",
    "_advance_kinetic_energy_all_elements_streaming_state",
)
_PHYSICS_API_WORDS = (
    *_WAVEFUNCTION_KINETIC_APIS,
    "compute_aos",
    "compute_mos",
    "compute_jastrow",
    "compute_geminal_all_elements",
    "compute_det_geminal",
    "evaluate_wavefunction",
    "evaluate_ln_wavefunction",
    "compute_local_energy",
    "read_trexio",
    "rs_ham",
    "_build_ham",
    "_make_ham",
    "load_twist",
    "run_twist_average",
    "_cli",
)
_ASSERTION_WORDS = ("assert", "allclose", "raises", "warns", "match", "equal")
_CONDITIONING_PATTERN = re.compile(r"(?:^|[^a-z0-9])(?:cond(?:ition(?:ed|ing)?)?|sigma_min|machine_epsilon)(?:[^a-z0-9]|$)")
_PATH_SUFFIXES = (".h5", ".xml", ".json", ".yaml", ".yml", ".toml", ".xyz", ".xsf", ".dat", ".txt", ".upf")
_SUMMARY_ACRONYMS = {
    "ad": "autodiff",
    "ao": "AO",
    "aos": "AOs",
    "cli": "CLI",
    "dtn": "DTN",
    "ecp": "ECP",
    "fd": "finite difference",
    "gfmc": "GFMC",
    "hdf5": "HDF5",
    "hf": "HF",
    "jvp": "JVP",
    "lm": "LM",
    "mcmc": "MCMC",
    "mo": "MO",
    "mos": "MOs",
    "mpc": "MPC",
    "pbc": "PBC",
    "pp": "PP",
    "qe": "QE",
    "rng": "RNG",
    "sr": "SR",
    "xla": "XLA",
}


# Derivation-scoped memo tables.  Every table is keyed on the AST
# node objects themselves (identity hash, strong reference) or on exact,
# type-tagged frozen values, and is alive only inside ``derivation_cache()``.
# Outside such a scope nothing is memoized, so a caller that monkeypatches a
# helper between two derivations always observes the patched behaviour.
_DERIVATION_CACHE: dict[str, dict[Any, Any]] | None = None


@contextlib.contextmanager
def derivation_cache() -> Iterator[None]:
    """Memoize pure AST/source analyses for the duration of one derivation run.

    The repository files, the parsed module trees and every helper consulted
    are treated as immutable while the scope is open; the scope is therefore
    opened by a single derivation or by one materializer run, never across a
    caller that edits sources or patches helpers in between.  Nested scopes
    share the outermost table.
    """
    global _DERIVATION_CACHE
    if _DERIVATION_CACHE is not None:
        yield
        return
    _DERIVATION_CACHE = {}
    try:
        yield
    finally:
        _DERIVATION_CACHE = None


def _memo_table(name: str) -> dict[Any, Any] | None:
    if _DERIVATION_CACHE is None:
        return None
    return _DERIVATION_CACHE.setdefault(name, {})


class _Unfreezable(Exception):
    """A value without an exact hashable spelling; the call is not memoized."""


def _frozen(value: Any) -> Any:
    """Exact, type-tagged hashable spelling of a selector-like value.

    ``1``, ``1.0`` and ``True`` (and ``0.0``/``-0.0``) compare equal in Python
    but must never share a memo entry, so scalars are keyed by type and repr.
    """
    if value is None or isinstance(value, (str, int, float, complex, bool)):
        return (type(value), repr(value))
    if isinstance(value, dict):
        return (dict, tuple((_frozen(key), _frozen(item)) for key, item in value.items()))
    if type(value) in (list, tuple):
        return (type(value), tuple(_frozen(item) for item in value))
    raise _Unfreezable(type(value).__qualname__)


def _module_node(node: ast.AST) -> bool:
    """Nodes owned by a cached ``_ModuleIndex`` tree (long-lived, never mutated)."""
    return hasattr(node, "_gdd_source_path")


def _walk(node: ast.AST):
    """``ast.walk(node)`` (same order), memoized as a tuple for module-tree nodes."""
    table = _memo_table("walk")
    if table is None or not _module_node(node):
        return ast.walk(node)
    nodes = table.get(node)
    if nodes is None:
        nodes = tuple(ast.walk(node))
        table[node] = nodes
    return nodes


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return {name for child in node.elts for name in _target_names(child)}
    return set()


def _string_values(node: ast.AST) -> frozenset[str]:
    table = _memo_table("string_values") if _module_node(node) else None
    if table is not None and node in table:
        return table[node]
    values = frozenset(_string_values_uncached(node))
    if table is not None:
        table[node] = values
    return values


def _string_values_uncached(node: ast.AST) -> set[str]:
    docstrings: set[int] = set()
    for owner in ast.walk(node):
        if not isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) or not owner.body:
            continue
        first = owner.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            docstrings.add(id(first.value))
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and id(child) not in docstrings and isinstance(child.value, str) and child.value
    }


class _StableAstNormalizer(ast.NodeTransformer):
    """Remove interpreter-owned AST details that have no source semantics."""

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.JoinedStr:  # noqa: N802 - AST visitor API
        self.generic_visit(node)
        # CPython 3.12.0--3.12.3 append an empty string to some nested f-string
        # format specifications.  Later CPython releases and 3.11 do not.  It
        # is a parser implementation detail and changes neither the expression
        # nor its formatting result.
        node.values = [value for value in node.values if not (isinstance(value, ast.Constant) and value.value == "")]
        return node


def _normalized_ast(node: ast.AST) -> ast.AST:
    return _StableAstNormalizer().visit(copy.deepcopy(node))


def _requires_normalization(node: ast.AST) -> bool:
    """Whether ``_StableAstNormalizer`` would change anything below ``node``."""
    return any(
        isinstance(child, ast.JoinedStr)
        and any(isinstance(value, ast.Constant) and value.value == "" for value in child.values)
        for child in ast.walk(node)
    )


def _normalized_ast_view(node: ast.AST) -> ast.AST:
    """Read-only normalized view: the node itself when normalization is a no-op.

    ``_StableAstNormalizer`` only drops empty f-string constants, so a subtree
    without one is structurally identical to its normalized deep copy.  Callers
    must not mutate the returned tree (unparsing and ``ast.dump`` do not).
    """
    return _normalized_ast(node) if _requires_normalization(node) else node


class _StableUnparser(ast._Unparser):  # type: ignore[attr-defined]
    """Use the Python 3.11 f-string spelling on every supported interpreter.

    Python 3.12's PEP 701 unparser may reuse a quote inside an f-string
    expression, while Python 3.11 necessarily chooses another outer quote.
    Both spellings are equivalent on 3.12, but only the latter is portable and
    stable.  This is the Python 3.11 ``JoinedStr`` algorithm, kept deliberately
    small so generated evidence does not depend on the CI patch release.
    """

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:  # noqa: N802 - AST visitor API
        self.write("f")
        # ``_avoid_backslashes`` was an implementation detail of
        # ``ast._Unparser`` through Python 3.12 and was removed in 3.13.
        # The compatibility branch is needed only when an older interpreter
        # explicitly enables it; absence therefore has the same meaning as
        # ``False``.
        if getattr(self, "_avoid_backslashes", False):
            with self.buffered() as buffer:
                self._write_fstring_inner(node)
            self._write_str_avoiding_backslashes("".join(buffer))
            return

        fstring_parts: list[tuple[str, bool]] = []
        for value in node.values:
            with self.buffered() as buffer:
                self._write_fstring_inner(value)
            fstring_parts.append(("".join(buffer), isinstance(value, ast.Constant)))

        new_fstring_parts: list[str] = []
        quote_types = list(ast._ALL_QUOTES)  # type: ignore[attr-defined]
        fallback_to_repr = False
        for value, is_constant in fstring_parts:
            value, new_quote_types = self._str_literal_helper(
                value,
                quote_types=quote_types,
                escape_special_whitespace=is_constant,
            )
            new_fstring_parts.append(value)
            if set(new_quote_types).isdisjoint(quote_types):
                fallback_to_repr = True
                break
            quote_types = new_quote_types

        if fallback_to_repr:
            quote_types = ["'''"]
            new_fstring_parts.clear()
            for value, _is_constant in fstring_parts:
                value = repr('"' + value)
                expected_prefix = "'\""
                if not value.startswith(expected_prefix):
                    raise ValueError(f"cannot render stable f-string part: {value!r}")
                new_fstring_parts.append(value[len(expected_prefix) : -1])

        value = "".join(new_fstring_parts)
        quote_type = quote_types[0]
        self.write(f"{quote_type}{value}{quote_type}")

    def _write_fstring_inner(self, node: ast.AST, is_format_spec: bool = False) -> None:
        # ``is_format_spec`` is accepted for Python 3.12's caller but ignored
        # intentionally: Python 3.11 used this same rendering for both paths.
        del is_format_spec
        if isinstance(node, ast.JoinedStr):
            for value in node.values:
                self._write_fstring_inner(value)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            self.write(node.value.replace("{", "{{").replace("}", "}}"))
        elif isinstance(node, ast.FormattedValue):
            self.visit_FormattedValue(node)
        else:
            raise ValueError(f"unexpected node inside JoinedStr: {node!r}")


def _source_expression(node: ast.AST) -> str:
    table = _memo_table("source_expression") if _module_node(node) else None
    if table is not None and node in table:
        return table[node]
    rendered = _StableUnparser().visit(_normalized_ast_view(node)).replace("\n", " ")
    if table is not None:
        table[node] = rendered
    return rendered


def _is_parametrize(decorator: ast.AST) -> bool:
    return isinstance(decorator, ast.Call) and _dotted_name(decorator.func).endswith("parametrize")


def _parametrize_names(value: ast.AST) -> list[str]:
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return [part.strip() for part in value.value.split(",")]
    if isinstance(value, (ast.List, ast.Tuple)):
        return [item.value for item in value.elts if isinstance(item, ast.Constant) and isinstance(item.value, str)]
    return [_source_expression(value)]


@dataclass(frozen=True)
class _SourceRecord:
    source_id: str
    path: str
    format: str


@lru_cache(maxsize=1)
def _source_records() -> tuple[_SourceRecord, ...]:
    document = yaml.safe_load(SOURCE_PATH.read_text(encoding="utf-8"))
    roots = document["roots"]
    return tuple(
        _SourceRecord(
            source_id=source_id,
            path=(Path(roots[row["root"]]["path"]) / row["path"]).as_posix(),
            format=row["format"],
        )
        for source_id, row in document["sources"].items()
    )


class _ModuleIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(self.tree):
            node._gdd_source_path = path.relative_to(ROOT)
        relative = path.relative_to(ROOT).with_suffix("")
        package = ".".join(relative.parts[:-1])
        imports = _import_bindings(self.tree, package=package)
        self.precision_constants: dict[str, dict[str, Any]] = {}
        for precision, mixed in (("full", False), ("mixed", True)):
            seed = dict(imports)
            # Test collection is allowed to choose different parameter values
            # in full and mixed precision.  Materialize both collections so the
            # SSoT records the union with a precision owner for every case.
            seed["is_mixed_precision_enabled"] = lambda value=mixed: value
            self.precision_constants[precision] = _module_constants(self.tree, seed=seed)
        self.constants = self.precision_constants["full"]
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.assignments: dict[str, ast.Assign | ast.AnnAssign] = {}
        self.imported_test_functions: dict[str, tuple[Path, str]] = {}
        # ``from tests import _cases as cases`` binds a repository test MODULE;
        # ``cases.build_case(...)`` must be followed like an imported helper.
        self.imported_test_modules: dict[str, Path] = {}
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                self.classes[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for name in _target_names(target):
                        self.assignments[name] = node
            elif isinstance(node, ast.ImportFrom):
                module_parts = package.split(".") if package else []
                if node.level:
                    module_parts = module_parts[: len(module_parts) - (node.level - 1)]
                else:
                    module_parts = []
                if node.module:
                    module_parts.extend(node.module.split("."))
                module_name = ".".join(module_parts)
                if module_name == "tests" or module_name.startswith("tests."):
                    for alias in node.names:
                        module_path = ROOT / (f"{module_name}.{alias.name}".replace(".", "/") + ".py")
                        if alias.name != "*" and module_path.is_file():
                            self.imported_test_modules[alias.asname or alias.name] = module_path
                imported_path = ROOT / (module_name.replace(".", "/") + ".py")
                if not module_name.startswith("tests.") or not imported_path.is_file():
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    self.imported_test_functions[alias.asname or alias.name] = (imported_path, alias.name)
        self.call_aliases = {
            name: _dotted_name(statement.value)
            for name, statement in self.assignments.items()
            if isinstance(statement.value, (ast.Name, ast.Attribute))
        }
        for statement in self.tree.body:
            if isinstance(statement, ast.ImportFrom) and statement.module in {"numpy", "numpy.testing", "pytest"}:
                for alias in statement.names:
                    if alias.name != "*":
                        self.call_aliases[alias.asname or alias.name] = f"{statement.module}.{alias.name}"

    def test(self, parts: list[str]) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, list[ast.ClassDef]]:
        if len(parts) == 1:
            return self.functions[parts[0]], []
        body: list[ast.stmt] = self.tree.body
        class_nodes: list[ast.ClassDef] = []
        for class_name in parts[:-1]:
            matches = [node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name]
            if len(matches) != 1:
                raise ValueError(f"cannot resolve {'::'.join(parts)} in {self.path.relative_to(ROOT)}")
            class_nodes.append(matches[0])
            body = matches[0].body
        matches = [
            node for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == parts[-1]
        ]
        if len(matches) != 1:
            raise ValueError(f"cannot resolve {'::'.join(parts)} in {self.path.relative_to(ROOT)}")
        return matches[0], class_nodes

    def reachable(
        self,
        test: ast.FunctionDef | ast.AsyncFunctionDef,
        class_nodes: list[ast.ClassDef],
    ) -> list[ast.AST]:
        table = _memo_table("reachable")
        key = (self, test, tuple(class_nodes))
        if table is not None and key in table:
            return list(table[key])
        result = self._reachable_uncached(test, class_nodes)
        if table is not None:
            table[key] = tuple(result)
        return result

    def _reachable_uncached(
        self,
        test: ast.FunctionDef | ast.AsyncFunctionDef,
        class_nodes: list[ast.ClassDef],
    ) -> list[ast.AST]:
        class_functions = {
            node.name: node
            for class_node in class_nodes
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        class_assignments: dict[str, ast.Assign | ast.AnnAssign] = {}
        for class_node in class_nodes:
            for node in class_node.body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for name in _target_names(target):
                        class_assignments[name] = node
        queue: list[ast.AST] = [test]
        if class_nodes:
            queue.extend(node for name, node in class_functions.items() if name in {"setup_class", "setup_method", "setup"})
        seen: set[int] = set()
        result: list[ast.AST] = []
        while queue:
            node = queue.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            result.append(node)
            names = {child.id for child in _walk(node) if isinstance(child, ast.Name)}
            names.update(child.attr for child in _walk(node) if isinstance(child, ast.Attribute))
            calls = {_dotted_name(child.func) for child in _walk(node) if isinstance(child, ast.Call)}
            referenced = names | {name.rsplit(".", 1)[-1] for name in calls}
            for name in sorted(referenced):
                if name in class_functions:
                    queue.append(class_functions[name])
                elif name in class_assignments:
                    queue.append(class_assignments[name])
                elif name in self.functions:
                    queue.append(self.functions[name])
                elif name in self.assignments:
                    queue.append(self.assignments[name])
        return result

    def assertion_dependencies(self, nodes: list[ast.AST], *, include_callbacks: bool = True) -> list[ast.AST]:
        """Follow repository-local test helpers without importing test modules.

        Keep this graph separate from fixture inference: expanding a comparison
        helper must not accidentally change the fixture universe. Cycles and
        repeated imports have one source-owned entry.
        """
        result = list(nodes)
        seen = {id(node) for node in nodes}
        queue = [(self, node) for node in nodes]
        lexical_indices = {}
        while queue:
            index, owner = queue.pop()
            if index.path not in lexical_indices:
                parents, _, bindings = _comparison_lexical_index(index.tree, index.path.relative_to(ROOT))
                lexical_indices[index.path] = (parents, bindings)
            parents, contextual_bindings = lexical_indices[index.path]
            for child in _walk(owner):
                if not isinstance(child, ast.Call):
                    continue
                contexts = []
                cursor = child
                while (cursor := parents.get(id(cursor))) is not None:
                    if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                        contexts.append(cursor)
                bindings = {}
                for context in reversed(contexts):
                    bindings.update(contextual_bindings.get(id(context), {}))
                # A comparison procedure passed as a callback must also be
                # discovered, so unsupported lazy calls cannot disappear.
                references = [child.func]
                if include_callbacks:
                    references.extend([*child.args, *(keyword.value for keyword in child.keywords)])
                for reference in references:
                    name = _expand_comparison_alias(_dotted_name(reference), bindings)
                    parts = name.split(".")
                    path, symbol = ROOT / ("/".join(parts[:-1]) + ".py"), parts[-1]
                    class_nodes: list[ast.ClassDef] = []
                    if len(parts) > 2 and not path.is_file():
                        # A forwarding entry may call a test method as Class.method
                        # (a forwarding entry): follow that method with its class context.
                        path = ROOT / ("/".join(parts[:-2]) + ".py")
                        if path.is_relative_to(ROOT / "tests") and path.is_file():
                            owner_class = _module_index(path).classes.get(parts[-2])
                            class_nodes = [owner_class] if owner_class is not None else []
                        if not class_nodes:
                            continue
                    if not path.is_relative_to(ROOT / "tests") or not path.is_file():
                        continue
                    imported = _module_index(path)
                    if class_nodes:
                        methods = [
                            node
                            for node in class_nodes[0].body
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol
                        ]
                        function = methods[0] if len(methods) == 1 else None
                    else:
                        function = imported.functions.get(symbol)
                    if function is None:
                        continue
                    for dependency in imported.reachable(function, class_nodes):
                        if id(dependency) not in seen:
                            seen.add(id(dependency))
                            result.append(dependency)
                            queue.append((imported, dependency))
        return result

    def parameter_cases(
        self,
        test: ast.FunctionDef | ast.AsyncFunctionDef,
        class_nodes: list[ast.ClassDef],
        pytest_fixtures: list[str],
        expected_cases: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        definitions: list[dict[str, Any]] = []
        parameter_axes: list[list[dict[str, Any]]] = []
        fixture_axes: list[list[dict[str, Any]]] = []
        decorators = [decorator for class_node in class_nodes for decorator in class_node.decorator_list]
        decorators.extend(test.decorator_list)
        # Pytest applies stacked decorators bottom-up.  Collection IDs and case
        # order therefore follow the reverse of source decorator order.
        for decorator in reversed(decorators):
            if not _is_parametrize(decorator) or len(decorator.args) < 2:
                continue
            names = _parametrize_names(decorator.args[0])
            source = _source_expression(decorator.args[1])
            definitions.append({"names": names, "values_source": source})
            ids_node = next((keyword.value for keyword in decorator.keywords if keyword.arg == "ids"), None)
            explicit_ids: list[Any] = []
            explicit_id_callable: Any | None = None
            if ids_node is not None:
                try:
                    id_provider = _literal(ids_node, self.constants)
                    if callable(id_provider):
                        explicit_id_callable = id_provider
                    else:
                        explicit_ids = list(id_provider)
                except Exception:  # Unsupported runtime IDs retain the deterministic fallback.
                    explicit_ids = []

            def resolve_explicit_id(value: Any, value_index: int) -> Any | None:
                if explicit_id_callable is not None:
                    try:
                        return explicit_id_callable(value)
                    except Exception:
                        return None
                return explicit_ids[value_index] if value_index < len(explicit_ids) else None

            rows_by_precision: dict[str, list[dict[str, Any]]] = {}
            for precision, constants in self.precision_constants.items():
                try:
                    values = list(_literal(decorator.args[1], constants))
                except Exception:
                    values = []
                precision_rows: list[dict[str, Any]] = []
                for value_index, value in enumerate(values):
                    # Imported-symbol shims synthesize arbitrary attributes;
                    # hasattr/getattr would invent pytest.param metadata and
                    # even override an explicitly authored collection ID.
                    is_pytest_param = isinstance(value, _ParamValue)
                    parameter_id = value.gdd_parameter_id if is_pytest_param else None
                    parameter_marks = list(value.gdd_marks) if is_pytest_param else []
                    if len(names) == 1:
                        if is_pytest_param and isinstance(value, tuple) and len(value) == 1:
                            value = value[0]
                        row = {names[0]: _yaml_value(value)}
                        selected_explicit_id = resolve_explicit_id(value, value_index)
                        row["__pytest_id_part__"] = (
                            parameter_id
                            if parameter_id is not None
                            else selected_explicit_id
                            if selected_explicit_id is not None
                            else _default_parameter_id(value, names[0], value_index)
                        )
                        if parameter_marks:
                            row["__pytest_marks__"] = parameter_marks
                        precision_rows.append(row)
                    elif isinstance(value, (tuple, list)) and len(value) >= len(names):
                        row = {name: _yaml_value(item) for name, item in zip(names, value, strict=False)}
                        explicit_row_id = resolve_explicit_id(value, value_index)
                        row["__pytest_id_part__"] = (
                            parameter_id
                            if parameter_id is not None
                            else explicit_row_id
                            if explicit_row_id is not None
                            else "-".join(
                                _default_parameter_id(item, name, value_index) for name, item in zip(names, value, strict=False)
                            )
                        )
                        if parameter_marks:
                            row["__pytest_marks__"] = parameter_marks
                        precision_rows.append(row)
                rows_by_precision[precision] = precision_rows

            if rows_by_precision["full"] == rows_by_precision["mixed"]:
                rows = rows_by_precision["full"]
            else:
                rows = []
                for precision in ("full", "mixed"):
                    for candidate in rows_by_precision[precision]:
                        match = next(
                            (
                                row
                                for row in rows
                                if {key: value for key, value in row.items() if key != "__precision_modes__"} == candidate
                            ),
                            None,
                        )
                        if match is None:
                            rows.append({**candidate, "__precision_modes__": [precision]})
                        else:
                            match["__precision_modes__"].append(precision)
            if rows:
                parameter_axes.append(rows)

        class_functions = {
            node.name: node
            for class_node in class_nodes
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for fixture_name in pytest_fixtures:
            fixture = class_functions.get(fixture_name, self.functions.get(fixture_name))
            if fixture is None:
                continue
            for decorator in fixture.decorator_list:
                if not isinstance(decorator, ast.Call) or not _dotted_name(decorator.func).endswith("fixture"):
                    continue
                params = next((kw.value for kw in decorator.keywords if kw.arg == "params"), None)
                if params is None:
                    continue
                source = _source_expression(params)
                definitions.append({"names": [fixture_name], "values_source": source})
                try:
                    values = list(_literal(params, self.constants))
                except Exception:
                    values = []
                if values:
                    ids_node = next((kw.value for kw in decorator.keywords if kw.arg == "ids"), None)
                    try:
                        id_provider = _literal(ids_node, self.constants) if ids_node is not None else None
                        explicit_ids = [] if id_provider is None or callable(id_provider) else list(id_provider)
                        if explicit_ids and len(explicit_ids) != len(values):
                            raise ValueError("ids length does not match params")
                        rows = []
                        for value_index, parameter in enumerate(values):
                            is_pytest_param = isinstance(parameter, _ParamValue)
                            parameter_id = parameter.gdd_parameter_id if is_pytest_param else None
                            value = parameter[0] if is_pytest_param else parameter
                            if parameter_id is None:
                                if callable(id_provider):
                                    parameter_id = id_provider(value)
                                elif explicit_ids:
                                    parameter_id = explicit_ids[value_index]
                            if parameter_id is not None and not isinstance(parameter_id, (str, int, float, complex, bool)):
                                raise ValueError("unsupported ID value")
                            row = {
                                fixture_name: _yaml_value(value),
                                "__pytest_id_part__": _default_parameter_id(
                                    value if parameter_id is None else parameter_id, fixture_name, value_index
                                ),
                            }
                            parameter_marks = list(parameter.gdd_marks) if is_pytest_param else []
                            if parameter_marks:
                                row["__pytest_marks__"] = parameter_marks
                            rows.append(row)
                    except Exception as exc:
                        raise ValueError(f"fixture {fixture_name}: cannot resolve exact pytest IDs: {exc}") from exc
                    fixture_axes.append(rows)

        # Pytest renders IDs from parametrized fixtures before IDs contributed
        # by ``@pytest.mark.parametrize`` on the test function.  Keeping the
        # two axis classes separate is essential for an exact collection-ID
        # contract (for example ``gamma-tall_cg``, not ``tall_cg-gamma``).
        axes = [*fixture_axes, *parameter_axes]
        selectors = [dict()]
        if axes:
            selectors = []
            for combination in product(*axes):
                constrained_modes = [set(row["__precision_modes__"]) for row in combination if "__precision_modes__" in row]
                precision_modes = set.intersection(*constrained_modes) if constrained_modes else set()
                if constrained_modes and not precision_modes:
                    continue
                selector = {
                    key: value
                    for row in combination
                    for key, value in row.items()
                    if key not in {"__pytest_id_part__", "__pytest_marks__", "__precision_modes__"}
                }
                id_parts = [row["__pytest_id_part__"] for row in combination if "__pytest_id_part__" in row]
                marks = [mark for row in combination for mark in row.get("__pytest_marks__", ())]
                if id_parts:
                    selector["__pytest_id_parts__"] = id_parts
                if marks:
                    selector["__pytest_marks__"] = marks
                if constrained_modes:
                    selector["__precision_modes__"] = [mode for mode in ("full", "mixed") if mode in precision_modes]
                selectors.append(selector)
        precision_counts = {
            mode: sum(mode in selector.get("__precision_modes__", ("full", "mixed")) for selector in selectors)
            for mode in ("full", "mixed")
        }
        precision_union_is_valid = any("__precision_modes__" in selector for selector in selectors) and all(
            count == expected_cases for count in precision_counts.values()
        )
        if len(selectors) != expected_cases and not precision_union_is_valid:
            if selectors and expected_cases % len(selectors) == 0:
                multiplier = expected_cases // len(selectors)
                selectors = [
                    {**selector, "fixture_case": fixture_case}
                    for selector in selectors
                    for fixture_case in range(1, multiplier + 1)
                ]
            else:
                selectors = [
                    {"collection_case": index, "selector_source": definitions or ["pytest collection"]}
                    for index in range(1, expected_cases + 1)
                ]
        concrete_cases = []
        for index, selector in enumerate(selectors, start=1):
            parameter_id_parts = selector.pop("__pytest_id_parts__", None)
            parameter_marks = selector.pop("__pytest_marks__", None)
            precision_modes = selector.pop("__precision_modes__", None)
            row: dict[str, Any] = {"case": index, "selector": selector}
            if parameter_id_parts is not None:
                row["parameter_id"] = "-".join(map(str, parameter_id_parts))
            if parameter_marks:
                row["markers"] = parameter_marks
            if precision_modes is not None:
                row["precision_modes"] = precision_modes
            concrete_cases.append(row)
        return definitions, concrete_cases


@cache
def _module_index(path: Path) -> _ModuleIndex:
    return _ModuleIndex(path)


def _imported_case_sources(
    index: _ModuleIndex,
    nodes: list[ast.AST],
    selector: dict[str, Any],
    *,
    seen: set[tuple[Path, str]] | None = None,
) -> list[dict[str, str]]:
    """Resolve committed sources used through repository-test helper imports.

    Test suites deliberately share expensive, reviewed Hamiltonian builders.
    Following only functions defined in the evidence module makes those
    runtime inputs invisible to the static ledger.  Follow direct ``from
    tests... import helper`` calls without importing or executing test code;
    case-local branch selection remains delegated to ``_case_strings``.
    """
    visited = set() if seen is None else seen
    selected: list[dict[str, str]] = []
    imported_functions = dict(index.imported_test_functions)
    package = ".".join(index.path.relative_to(ROOT).with_suffix("").parts[:-1])
    for owner in nodes:
        for import_node in _walk(owner):
            if not isinstance(import_node, ast.ImportFrom):
                continue
            module_parts = package.split(".") if package else []
            if import_node.level:
                module_parts = module_parts[: len(module_parts) - (import_node.level - 1)]
            else:
                module_parts = []
            if import_node.module:
                module_parts.extend(import_node.module.split("."))
            module_name = ".".join(module_parts)
            imported_path = ROOT / (module_name.replace(".", "/") + ".py")
            if not module_name.startswith("tests.") or not imported_path.is_file():
                continue
            for alias in import_node.names:
                if alias.name != "*":
                    imported_functions[alias.asname or alias.name] = (imported_path, alias.name)
    # Only call sites this concrete case can execute, each with the static
    # scope at the site (selector, bound locals, local-helper arguments).
    # Walking every call of every reachable node would attach the sources of
    # unselected dispatch entries and untaken branches to this case.
    calls = [
        (owner_name, child, scope, _dotted_name(child.func).rsplit(".", 1)[-1])
        for child, scope, owner_name in _case_reachable_calls(nodes, selector, index.constants)
    ]
    module_aliases = dict(getattr(index, "imported_test_modules", {}))
    for owner_name, call, scope, local_name in calls:
        dotted = _dotted_name(call.func)
        prefix = dotted.rsplit(".", 1)[0] if "." in dotted else None
        if prefix in module_aliases:
            imported_path, imported_name = module_aliases[prefix], local_name
        elif local_name in imported_functions:
            imported_path, imported_name = imported_functions[local_name]
        else:
            continue
        imported_index = _module_index(imported_path)
        helper = imported_index.functions.get(imported_name)
        if helper is None:
            continue
        reachable = imported_index.reachable(helper, [])
        referenced_classes = {
            child.id
            for owner_node in reachable
            for child in _walk(owner_node)
            if isinstance(child, ast.Name) and child.id in imported_index.classes
        }
        for class_name in sorted(referenced_classes):
            for node in imported_index.reachable(imported_index.classes[class_name], []):
                if node not in reachable:
                    reachable.append(node)
        nested_selector = dict(selector)
        formals = [*helper.args.posonlyargs, *helper.args.args]
        for formal, argument in zip(formals, call.args, strict=False):
            try:
                if _dotted_name(argument) == "request.param" and owner_name is not None:
                    nested_selector[formal.arg] = selector[owner_name]
                else:
                    nested_selector[formal.arg] = _yaml_value(_literal(argument, scope))
            except (KeyError, ValueError, TypeError, NameError, SyntaxError):
                continue
        for keyword in call.keywords:
            if keyword.arg is None:
                continue
            try:
                nested_selector[keyword.arg] = _yaml_value(_literal(keyword.value, scope))
            except (ValueError, TypeError, NameError, SyntaxError):
                continue
        # The same helper reached with different concrete arguments selects
        # different sources (``rs_ham("pbc_gamma...")`` vs ``rs_ham("twist...")``).
        try:
            key = (imported_path, imported_name, _frozen(nested_selector))
        except _Unfreezable:
            key = (imported_path, imported_name, repr(sorted(nested_selector.items(), key=lambda item: item[0])))
        if key in visited:
            continue
        visited.add(key)
        rows = _case_sources(nested_selector, imported_index.constants, reachable)
        if not rows:
            case_strings = _case_strings(reachable, nested_selector, imported_index.constants)
            rows = _case_sources(
                {**nested_selector, "reachable_strings": sorted(case_strings)},
                imported_index.constants,
                reachable,
            )
        for row in rows:
            if row not in selected:
                selected.append(row)
        if not rows:
            for row in _imported_case_sources(imported_index, reachable, nested_selector, seen=visited):
                if row not in selected:
                    selected.append(row)
    return sorted(selected, key=lambda row: row["source_id"])


def _yaml_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if isinstance(value, (tuple, list)):
        return [_yaml_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_yaml_value(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, dict):
        return {str(key): _yaml_value(item) for key, item in value.items()}
    if isinstance(getattr(value, "gdd_expression", None), str):
        return {"source_expression": value.gdd_expression}
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _default_parameter_id(value: Any, name: str, index: int) -> str:
    if isinstance(value, str):
        # Match pytest's default ``ascii_escaped`` ID rendering.  In
        # particular, a literal backslash in a string-valued parameter is
        # doubled in the collected node ID.
        return value.encode("unicode_escape").decode("ascii")
    if value is None or isinstance(value, (int, float, complex, bool)):
        return str(value)
    expression = getattr(value, "gdd_expression", None)
    if isinstance(expression, str):
        constant = _float_constant_expression(expression)
        if constant is not None:
            # pytest renders a float parameter as ``str(value)`` whatever
            # source spelled it, e.g. ``-np.inf`` collects as ``-inf`` (#200).
            return str(constant)
        if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", expression):
            return expression.rsplit(".", 1)[-1]
        return f"{name}{index}"
    return f"{name}{index}"


# Float-valued module constants whose pytest ID is ``str(value)``.  Kept as a
# static table: this module never imports numpy (see the module docstring).
_FLOAT_MODULE_CONSTANTS = MappingProxyType(
    {
        "math.e": math.e,
        "math.inf": math.inf,
        "math.nan": math.nan,
        "math.pi": math.pi,
        "math.tau": math.tau,
        "numpy.e": math.e,
        "numpy.euler_gamma": 0.5772156649015329,
        "numpy.inf": math.inf,
        "numpy.nan": math.nan,
        "numpy.pi": math.pi,
    }
)


def _float_constant_expression(expression: str) -> float | None:
    """Realize ``[+-]<module constant>`` (any number of unary signs) as a float."""
    match = re.fullmatch(r"\s*((?:[+-]\s*)*)([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*", expression)
    if match is None or match.group(2) not in _FLOAT_MODULE_CONSTANTS:
        return None
    value = _FLOAT_MODULE_CONSTANTS[match.group(2)]
    return -value if match.group(1).count("-") % 2 else value


def _explicit_fixture_path(value: str) -> str | None:
    """Resolve inert Path/join suffixes without executing test code.

    A basename is not enough for UPF: the example tree contains both standalone
    files and same-named copies inside QE save directories.
    """
    table = _memo_table("explicit_fixture_path")
    if table is not None and value in table:
        return table[value]
    resolved = _explicit_fixture_path_uncached(value)
    if table is not None:
        table[value] = resolved
    return resolved


def _explicit_fixture_path_uncached(value: str) -> str | None:

    def parts(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return list(Path(node.value).parts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return parts(node.left) + parts(node.right)
        if isinstance(node, ast.Call) and _dotted_name(node.func) in {"Path", "pathlib.Path", "os.path.join"}:
            return [part for arg in node.args for part in parts(arg)]
        if isinstance(node, ast.Call) and _dotted_name(node.func).rsplit(".", 1)[-1] == QE_EXAMPLE_ROOT_HELPER:
            # The test session's mirror of the committed QE example tree.
            return ["qe_example_files"]
        if isinstance(node, ast.Call) and _dotted_name(node.func) == "os.fspath" and len(node.args) == 1:
            return parts(node.args[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"resolve", "absolute"}:
            return parts(node.func.value)
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            return parts(node.value)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
            return []
        if isinstance(node, ast.Name):
            return []
        raise ValueError("not an inert path expression")

    candidates = [list(Path(value).parts)]
    try:
        candidates.append(parts(ast.parse(value, mode="eval").body))
    except (SyntaxError, ValueError):
        pass
    for components in candidates:
        for marker in ("trexio_example_files", "qe_example_files"):
            if marker in components:
                tail = components[components.index(marker) :]
                if all(part not in {"..", "."} for part in tail):
                    return str(Path("tests", *tail))
    return None


def _registered_sources(strings: set[str]) -> list[dict[str, str]]:
    table = _memo_table("registered_sources")
    key = frozenset(strings) if table is not None else None
    if key is not None and key in table:
        return [dict(row) for row in table[key]]
    selected = _registered_sources_uncached(strings)
    if key is not None:
        table[key] = tuple(dict(row) for row in selected)
    return selected


def _source_needles(path: str) -> tuple[str, ...] | None:
    """Committed-root-relative components an inert path expression must name."""
    parts = Path(path).parts
    markers = {"trexio_example_files", "qe_example_files"}
    marker_index = next((index for index, part in enumerate(parts) if part in markers), None)
    if marker_index is None:
        return None
    return parts[marker_index:]


def _registered_sources_uncached(strings: set[str]) -> list[dict[str, str]]:
    def expression_names_source(value: str, needles: tuple[str, ...] | None) -> bool:
        """Recognize inert ``Path``/``os.path.join`` provenance expressions.

        The static collector intentionally does not execute test-module path
        constructors.  Their opaque expression still contains the committed
        fixture root and every relative path component, which is sufficient to
        identify one registry row without importing the test or consulting the
        filesystem.
        """
        if needles is None:
            return False
        normalized = value.replace("\\", "/")
        offset = 0
        for needle in needles:
            position = normalized.find(needle, offset)
            if position < 0:
                return False
            offset = position + len(needle)
        return True

    # Hoisted per-string and per-record path spellings; each predicate below
    # is the same pure function of (value, record) as before.
    values = list(strings)
    basenames = {value: Path(value).name for value in values}
    explicit_paths: dict[str, str | None] = {}

    def explicit_path(value: str) -> str | None:
        if value not in explicit_paths:
            explicit_paths[value] = _explicit_fixture_path(value)
        return explicit_paths[value]

    selected: list[dict[str, str]] = []
    for record in _source_records():
        basename = Path(record.path).name
        if record.format == "upf":
            if any(value == record.path or explicit_path(value) == record.path for value in values):
                selected.append({"source_id": record.source_id, "path": record.path, "format": record.format})
            continue
        needles = _source_needles(record.path)
        if any(
            value == record.path or basenames[value] == basename or expression_names_source(value, needles) for value in values
        ):
            selected.append({"source_id": record.source_id, "path": record.path, "format": record.format})
    return selected


@cache
def _rs_fixture_patterns() -> dict[str, Any]:
    fixture_path = ROOT / "tests/test_rs_kernel/_rs_fixtures.py"
    fixture_tree = ast.parse(fixture_path.read_text(encoding="utf-8"), filename=str(fixture_path))
    constants = _module_constants(fixture_tree, seed=_import_bindings(fixture_tree))
    patterns = constants.get("_PATTERNS", {})
    return patterns if isinstance(patterns, dict) else {}


@cache
def _generated_source_by_stem() -> dict[str, dict[str, str]]:
    """Committed source record behind each generated-matrix builder stem."""
    path = ROOT / "tests/test_jqmc_kernel/_generated_matrix_base.py"
    index = _module_index(path)
    mapping = index.constants.get("_GDD_SOURCE_ID_BY_STEM", {})
    records = {record.source_id: record for record in _source_records()}
    resolved: dict[str, dict[str, str]] = {}
    if not isinstance(mapping, dict):
        return resolved
    for stem, source_id in mapping.items():
        if not isinstance(stem, str) or not isinstance(source_id, str) or source_id not in records:
            continue
        record = records[source_id]
        resolved[stem] = {"source_id": record.source_id, "path": record.path, "format": record.format}
    return resolved


def _case_sources(
    selector: dict[str, Any],
    module_constants: dict[str, Any] | None = None,
    nodes: list[ast.AST] | None = None,
) -> list[dict[str, str]]:
    table = _memo_table("case_sources")
    key = None
    if table is not None:
        try:
            key = (_frozen(selector), id(module_constants), None if nodes is None else tuple(nodes))
        except _Unfreezable:
            key = None
    if key is not None and key in table and table[key][0] is module_constants:
        return [dict(row) for row in table[key][1]]
    selected = _case_sources_uncached(selector, module_constants, nodes)
    if key is not None:
        table[key] = (module_constants, tuple(dict(row) for row in selected))
    return selected


def _case_sources_uncached(
    selector: dict[str, Any],
    module_constants: dict[str, Any] | None = None,
    nodes: list[ast.AST] | None = None,
) -> list[dict[str, str]]:
    strings: set[str] = set()
    patterns: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str):
            strings.add(value)
        elif isinstance(value, dict):
            pattern = value.get("pattern")
            if isinstance(pattern, str):
                patterns.add(pattern)
            source = value.get("src")
            if isinstance(source, list) and len(source) >= 2 and source[0] == "pattern" and isinstance(source[1], str):
                patterns.add(source[1])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(selector)
    # Some suites pass a named fixture pattern through a larger pytest axis so
    # that the production driver can consume both ordinary file names and
    # reviewed Rust fixture builders.  The prefix is a discriminator, not part
    # of the committed source identity.
    for value in tuple(strings):
        if value.startswith("RSPATTERN:"):
            patterns.add(value.removeprefix("RSPATTERN:"))

    # Resolve selector-indexed constant tables such as
    # ``_LEAVES[leaf_id]``.  Walking the table assignment as undifferentiated
    # syntax attaches every sibling source to every case; only the selected
    # value is reachable for this concrete pytest item.
    namespace = {**(module_constants or {}), **selector}
    for owner in nodes or ():
        for child in _walk(owner):
            if not isinstance(child, ast.Subscript):
                continue
            index_names = {name.id for name in ast.walk(child.slice) if isinstance(name, ast.Name)}
            if not index_names.intersection(selector):
                continue
            try:
                selected_value = _literal(child, namespace)
            except Exception:
                selected_value = None
            if selected_value is None:
                # The static constant evaluator leaves an entry unresolved when it
                # is built from a runtime root such as ``Path(__file__).parents[1]``
                # (``_SAVES[fixture] = (_QE / "Li_gamma" / "out" / "li.save", ...)``).
                # The selected entry's own literals still name exactly one source.
                strings.update(_selected_table_entry_strings(child, nodes or (), namespace))
                continue
            visit(_yaml_value(selected_value))
    pattern = selector.get("pattern")
    cell = selector.get("cell")
    cells = (module_constants or {}).get("_CELLS", {})
    references_cells = any(
        isinstance(child, ast.Name) and child.id == "_CELLS" for owner in nodes or () for child in _walk(owner)
    )
    if not isinstance(pattern, str) and isinstance(cell, str) and isinstance(cells, dict) and references_cells:
        cell_definition = cells.get(cell)
        if isinstance(cell_definition, dict):
            pattern = cell_definition.get("pattern")
    if isinstance(pattern, str):
        patterns.add(pattern)
    selected: list[dict[str, str]] = []
    fixture_patterns = _rs_fixture_patterns()
    generated_sources = _generated_source_by_stem()
    # A selected local builder may pass the reviewed pattern name as an
    # ordinary string (``rs_ham("bspline_gamma_jnone")``), without exposing a
    # selector field named ``pattern``.  Once branch resolution has made that
    # string case-local, it has the same authority as an explicit pattern
    # selector and must resolve to its committed input artifact.
    patterns.update(value for value in strings if value in fixture_patterns)
    # A selector may carry the generated-matrix stem directly instead of an
    # ``_rs_fixtures`` pattern name.  The stem is still only a builder
    # discriminator; the committed QE/TREXIO artifact behind it is the source
    # that belongs in the auditable case binding.
    for value in strings:
        if value in generated_sources and generated_sources[value] not in selected:
            selected.append(generated_sources[value])
    for selected_pattern in patterns:
        if isinstance(fixture_patterns, dict) and selected_pattern in fixture_patterns:
            stem = fixture_patterns[selected_pattern][0]
            if isinstance(stem, str):
                if stem in generated_sources:
                    if generated_sources[stem] not in selected:
                        selected.append(generated_sources[stem])
                else:
                    strings.add(stem)
    for row in _registered_sources(strings):
        if row not in selected:
            selected.append(row)
    return sorted(selected, key=lambda row: row["source_id"])


@cache
def _test_module_assignments(path: Path) -> dict[str, ast.expr]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments: dict[str, ast.expr] = {}
    for statement in tree.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)) and statement.value is not None:
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                for name in _target_names(target):
                    assignments[name] = statement.value
    return assignments


def _imported_test_constant_strings(expression: Any, depth: int = 0) -> set[str]:
    """Literals behind an imported repository-test constant the evaluator left opaque.

    ``from tests.x import _QE_SAVE`` binds an inert token named
    ``tests.x._QE_SAVE`` when ``_QE_SAVE`` is built from a runtime root
    (``Path(__file__)...``). Its assignment's own literals (and those of the
    module-level names it is built from) still name the committed source.
    """
    if not isinstance(expression, str) or depth > 3 or not re.fullmatch(r"tests(?:\.\w+)+", expression):
        return set()
    module, _, name = expression.rpartition(".")
    path = ROOT / (module.replace(".", "/") + ".py")
    if not path.is_file():
        return set()
    value = _test_module_assignments(path).get(name)
    if value is None:
        return set()
    strings = set(_string_values(value))
    for child in ast.walk(value):
        if isinstance(child, ast.Name) and child.id != name:
            strings.update(_imported_test_constant_strings(f"{module}.{child.id}", depth + 1))
    return strings


def _selected_table_entry_strings(subscript: ast.Subscript, nodes: Any, namespace: dict[str, Any]) -> set[str]:
    """String literals of the one dict-literal entry a selector-indexed subscript selects.

    Only ``TABLE[key]`` on a reachable module-level ``TABLE = {...}`` literal with a
    constant key is resolved; every other form contributes nothing (never the union
    of all entries).
    """
    if not isinstance(subscript.value, ast.Name):
        return set()
    try:
        key = _literal(subscript.slice, namespace)
    except Exception:
        return set()
    for owner in nodes:
        if not isinstance(owner, (ast.Assign, ast.AnnAssign)) or not isinstance(owner.value, ast.Dict):
            continue
        targets = owner.targets if isinstance(owner, ast.Assign) else [owner.target]
        if subscript.value.id not in {name for target in targets for name in _target_names(target)}:
            continue
        for entry_key, entry_value in zip(owner.value.keys, owner.value.values, strict=True):
            if isinstance(entry_key, ast.Constant) and entry_key.value == key and isinstance(key, (str, int)):
                return set(_string_values(entry_value))
    return set()


def _selected_table_entry(subscript: ast.Subscript, nodes: Any, namespace: dict[str, Any]) -> ast.expr | None:
    """The value AST of the one dict-literal entry ``TABLE[key]`` selects (``None`` when unresolved)."""
    if not isinstance(subscript.value, ast.Name):
        return None
    try:
        key = _literal(subscript.slice, namespace)
    except Exception:
        return None
    if not isinstance(key, (str, int)):
        return None
    for owner in nodes:
        if not isinstance(owner, (ast.Assign, ast.AnnAssign)) or not isinstance(owner.value, ast.Dict):
            continue
        targets = owner.targets if isinstance(owner, ast.Assign) else [owner.target]
        if subscript.value.id not in {name for target in targets for name in _target_names(target)}:
            continue
        for entry_key, entry_value in zip(owner.value.keys, owner.value.values, strict=True):
            if isinstance(entry_key, ast.Constant) and entry_key.value == key:
                return entry_value
    return None


def _table_entries(subscript: ast.Subscript, nodes: Any) -> list[ast.expr]:
    """Every value AST of the reachable module-level dict literal ``TABLE`` that ``TABLE[...]`` subscripts."""
    if not isinstance(subscript.value, ast.Name):
        return []
    for owner in nodes:
        if not isinstance(owner, (ast.Assign, ast.AnnAssign)) or not isinstance(owner.value, ast.Dict):
            continue
        targets = owner.targets if isinstance(owner, ast.Assign) else [owner.target]
        if subscript.value.id in {name for target in targets for name in _target_names(target)}:
            return list(owner.value.values)
    return []


@dataclass(frozen=True, eq=False)
class _SelectedLambda:
    """A local bound by the case to a ``lambda`` of the dispatch-table row it selected.

    ``builder, flag = _CASES[case]`` (row ``(lambda d: build("x.h5"), True)``)
    binds ``builder`` to the row's lambda for this concrete case only. The
    lambda runs where the case calls the local (``builder(tmp)``); defining
    the table row does not run it, and the unselected rows' lambdas never run.
    """

    node: ast.Lambda


def _selected_row_bindings(statement: ast.stmt, nodes: Any, scope: dict[str, Any]) -> dict[str, Any]:
    """Names an executed ``name = TABLE[key]`` / ``a, b = TABLE[key]`` binds from the selected row.

    A row ``lambda`` binds a ``_SelectedLambda``; when the row is unpacked, a row
    element that is itself plain data (``"mol_ecp_sphe_j123"``, ``True``) binds its
    value for this concrete case even when a sibling element is a call the static
    evaluator cannot run (``(pattern, _ecp_call("kernel"), False, True)``), so the
    unpacked discriminator resolves the branches and table lookups that follow.
    Elements that are neither stay unbound, exactly as before.
    """
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return {}
    if not isinstance(statement.value, ast.Subscript):
        return {}
    entry = _selected_table_entry(statement.value, nodes, scope)
    if entry is None:
        return {}
    target = statement.targets[0]
    if isinstance(target, ast.Name):
        return {target.id: _SelectedLambda(entry)} if isinstance(entry, ast.Lambda) else {}
    if not (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(entry, (ast.Tuple, ast.List))
        and len(target.elts) == len(entry.elts)
        and not any(isinstance(item, ast.Starred) for item in (*target.elts, *entry.elts))
    ):
        return {}
    bindings: dict[str, Any] = {}
    for name, value in zip(target.elts, entry.elts, strict=True):
        if not isinstance(name, ast.Name):
            continue
        if isinstance(value, ast.Lambda):
            bindings[name.id] = _SelectedLambda(value)
            continue
        # Only a self-contained literal element (no name, no call) is bound: it
        # means the same thing at the table's definition and at this use.
        try:
            literal = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            continue
        if _plain_value(literal):
            bindings[name.id] = literal
    return bindings


def _alias_constant_expression(node: ast.AST, scope: dict[str, Any]) -> str | None:
    """``tests.x.NAME`` for ``alias.NAME`` where ``alias`` is an inert repository test-module token."""
    if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
        return None
    expression = getattr(scope.get(node.value.id), "gdd_expression", None)
    if isinstance(expression, str) and re.fullmatch(r"tests(?:\.\w+)+", expression):
        return f"{expression}.{node.attr}"
    return None


def _imported_constant_is_dict(expression: str) -> bool:
    module, _, name = expression.rpartition(".")
    path = ROOT / (module.replace(".", "/") + ".py")
    return path.is_file() and isinstance(_test_module_assignments(path).get(name), ast.Dict)


def _imported_table_entry_strings(subscript: ast.Subscript, scope: dict[str, Any]) -> set[str] | None:
    """Literals of the one entry ``alias.TABLE[key]`` selects in an imported test module's dict literal."""
    expression = _alias_constant_expression(subscript.value, scope)
    if expression is None:
        return None
    module, _, name = expression.rpartition(".")
    path = ROOT / (module.replace(".", "/") + ".py")
    if not path.is_file():
        return None
    table = _test_module_assignments(path).get(name)
    if not isinstance(table, ast.Dict):
        return None
    try:
        key = _literal(subscript.slice, scope)
    except Exception:
        return None
    for entry_key, entry_value in zip(table.keys, table.values, strict=True):
        if isinstance(entry_key, ast.Constant) and entry_key.value == key and isinstance(key, (str, int)):
            return set(_string_values(entry_value))
    return None


def _fixture_returned_function(function: ast.AST) -> ast.FunctionDef | None:
    """The nested function a ``@pytest.fixture`` factory returns as its value (``return get``)."""
    if not isinstance(function, ast.FunctionDef) or not function.body:
        return None
    if not any(
        _dotted_name(decorator.func if isinstance(decorator, ast.Call) else decorator).rsplit(".", 1)[-1] == "fixture"
        for decorator in function.decorator_list
    ):
        return None
    last = function.body[-1]
    if not isinstance(last, ast.Return) or not isinstance(last.value, ast.Name):
        return None
    nested = [
        statement for statement in function.body if isinstance(statement, ast.FunctionDef) and statement.name == last.value.id
    ]
    return nested[0] if len(nested) == 1 else None


def _inert_selector_value(value: Any) -> bool:
    """A selector entry that is only the YAML form of an inert runtime token."""
    return isinstance(value, dict) and set(value) == {"source_expression"}


def _calls_outside_lambdas(node: ast.AST) -> list[ast.Call]:
    """Call nodes of an expression that are not inside a ``lambda`` body."""
    found: list[ast.Call] = []
    pending = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, ast.Lambda):
            continue
        if isinstance(current, ast.Call):
            found.append(current)
        pending.extend(ast.iter_child_nodes(current))
    return found


def _inline_inert_path_roots(node: ast.AST, scope: dict[str, Any], nodes: Any = ()) -> str:
    """Source of a path expression with path-root names replaced by their defining expressions.

    ``_QE_ROOT / "Li.ccECP.upf"`` names its committed root only through the
    module constant ``_QE_ROOT = Path(__file__).resolve().parents[1] / "qe_example_files"``,
    which the constant evaluator leaves unresolved (a runtime ``Path`` root).
    A name is replaced by its inert expression or, failing that, by the value
    of its single reachable module-level assignment.
    """
    assignments: dict[str, list[ast.expr]] = {}
    for owner in nodes:
        if isinstance(owner, ast.Assign) and owner.value is not None:
            for target in owner.targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(owner.value)

    class _Inline(ast.NodeTransformer):
        def __init__(self) -> None:
            self.depth = 0

        def visit_Name(self, name: ast.Name) -> ast.AST:
            expression = getattr(scope.get(name.id), "gdd_expression", None)
            replacement: ast.AST | None = None
            if self.depth >= 4:
                return name
            if isinstance(expression, str):
                try:
                    replacement = ast.parse(expression, mode="eval").body
                except SyntaxError:
                    replacement = None
                if isinstance(replacement, (ast.Name, ast.Attribute)):
                    # An import token (``Path``, ``os.path``) is already its own spelling.
                    replacement = None
            elif name.id not in scope and len(assignments.get(name.id, ())) == 1:
                replacement = copy.deepcopy(assignments[name.id][0])
            if replacement is None:
                return name
            self.depth += 1
            try:
                return self.visit(replacement)
            finally:
                self.depth -= 1

    return ast.unparse(_Inline().visit(copy.deepcopy(node)))


def _names_upf_literal(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Constant) and isinstance(child.value, str) and child.value.lower().endswith(".upf")
        for child in ast.walk(node)
    )


def _repo_artifacts(strings: set[str], test_path: Path) -> list[str]:
    artifacts: set[str] = set()
    fixture_roots = (ROOT / "tests/trexio_example_files", ROOT / "tests/qe_example_files")
    strings = strings | {
        path for value in strings if (path := _explicit_fixture_path(value)) is not None and path.lower().endswith(".upf")
    }
    for value in strings:
        if not value.lower().endswith(_PATH_SUFFIXES):
            continue
        candidates = [ROOT / value, test_path.parent / value]
        candidates.extend(root / value for root in fixture_roots)
        basename_matches = [
            path for root in fixture_roots for path in root.rglob(Path(value).name) if path.is_file() and path != PIN_MANIFEST
        ]
        if value.lower().endswith(".upf") and len(Path(value).parts) == 1 and len(basename_matches) != 1:
            continue
        if len(basename_matches) == 1:
            candidates.append(basename_matches[0])
        for candidate in candidates:
            if fixture_exists(candidate):
                if candidate.suffix.lower() == ".upf":
                    parent = next(
                        (
                            record.path
                            for record in _source_records()
                            if record.format == "qe-save" and candidate.is_relative_to(ROOT / record.path)
                        ),
                        None,
                    )
                    if parent is not None:
                        artifacts.add(parent)
                        break
                artifacts.add(candidate.relative_to(ROOT).as_posix())
                break
    return sorted(artifacts)


# Node classes with a dedicated branch in ``_case_strings``' visitor.
_CASE_STRING_BRANCH_TYPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Assign,
    ast.AnnAssign,
    ast.Subscript,
    ast.If,
    ast.IfExp,
    ast.For,
    ast.AsyncFor,
    ast.Call,
    ast.Attribute,
    ast.BinOp,
    ast.Name,
    ast.Constant,
)


# Literal ``for`` loops are unrolled per value so that branches, table lookups
# and helper arguments that read the loop variable resolve per iteration.  A
# loop longer than this whose body reads its variable is visited once with the
# variable unbound instead (issue #153: ``for counter in range(400000)`` in a
# test helper made cold binding preparation exceed the CI job timeout).
_MAX_UNROLLED_LOOP_ITERATIONS = 1024
# Loop values that contribute no string when a store of the loop variable is
# visited (``collect_strings`` reads strings only).
_INERT_LOOP_VALUE_TYPES = (int, float, bool, type(None))


def _plain_value(value: Any, depth: int = 0) -> bool:
    """Whether a statically evaluated value is plain data or a bare function-name token.

    An inert token is plain only when it spells one identifier (``_fx_agp_mo_dtn``
    from a ``(builder, required)`` table row): calling it is resolved against the
    module's own functions, never evaluated.
    """
    if depth > 4:
        return False
    if value is None or type(value) in (str, int, float, complex, bool):
        return True
    expression = getattr(value, "gdd_expression", None)
    if isinstance(expression, str):
        return depth > 0 and expression.isidentifier()
    if isinstance(value, (tuple, list, set, frozenset)):
        return len(value) <= 64 and all(_plain_value(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return len(value) <= 64 and all(
            _plain_value(key, depth + 1) and _plain_value(item, depth + 1) for key, item in value.items()
        )
    return False


def _pure_plain_expression(node: ast.AST, scope: dict[str, Any]) -> bool:
    """No call in ``node`` except methods of names bound to plain data (``case.split("-")``)."""
    for child in ast.walk(node):
        if isinstance(child, (ast.Lambda, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)):
            return False
        if isinstance(child, ast.Call):
            func = child.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in scope
                and _plain_value(scope[func.value.id])
            ):
                return False
    return True


def _stored_names(statement: ast.AST) -> set[str]:
    """Every name a statement stores (any depth, nested scopes included: conservative)."""
    names = {
        child.id for child in ast.walk(statement) if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del))
    }
    names.update(
        child.name for child in ast.walk(statement) if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    return names


def _plain_assignment_bindings(statement: ast.stmt, scope: dict[str, Any]) -> dict[str, Any]:
    """Bindings of one executed ``name = value`` / ``a, b = value`` whose value is plain data."""
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return {}
    target = statement.targets[0]
    if isinstance(target, ast.Name):
        targets: list[str] | None = [target.id]
    elif isinstance(target, (ast.Tuple, ast.List)) and all(isinstance(item, ast.Name) for item in target.elts):
        targets = [item.id for item in target.elts]
    else:
        return {}
    value_names = {child.id for child in ast.walk(statement.value) if isinstance(child, ast.Name)}
    if any(isinstance(getattr(scope.get(name), "gdd_expression", None), str) for name in value_names):
        return {}
    try:
        value = _literal(statement.value, scope)
    except Exception:
        return {}
    if not _plain_value(value):
        return {}
    if isinstance(target, ast.Name):
        return {target.id: value}
    if not isinstance(value, (tuple, list)) or len(value) != len(targets):
        return {}
    return dict(zip(targets, value, strict=True))


@cache
def _case_string_generic_only(node_type: type) -> bool:
    """True when no ``isinstance`` branch of the visitor can match ``node_type``."""
    return not issubclass(node_type, _CASE_STRING_BRANCH_TYPES)


def _child_nodes(node: ast.AST) -> tuple[ast.AST, ...]:
    """``ast.iter_child_nodes`` as a tuple, memoized for module-tree nodes."""
    table = _memo_table("child_nodes")
    if table is None:
        return tuple(ast.iter_child_nodes(node))
    children = table.get(node)
    if children is None:
        children = tuple(ast.iter_child_nodes(node))
        if _module_node(node):
            table[node] = children
    return children


def _case_strings(nodes: list[ast.AST], selector: dict[str, Any], constants: dict[str, Any]) -> set[str]:
    """String literals reachable for one simple selector-controlled branch."""
    table = _memo_table("case_strings")
    key = None
    if table is not None:
        try:
            key = (tuple(nodes), id(constants), _frozen(selector))
        except _Unfreezable:
            key = None
    if key is not None and key in table and table[key][0] is constants:
        return set(table[key][1])
    strings = _case_strings_uncached(nodes, selector, constants)
    if key is not None:
        table[key] = (constants, frozenset(strings))
    return strings


def _case_reachable_calls(
    nodes: list[ast.AST], selector: dict[str, Any], constants: dict[str, Any]
) -> list[tuple[ast.Call, dict[str, Any], str | None]]:
    """Calls on the selector-resolved path of one concrete case, with their static scope.

    The same branch, dispatch-table, local-helper and single-assignment
    resolution as ``_case_strings`` decides which call sites one concrete
    case can execute.  Each call is returned with the static scope in force
    at that site and the name of the function containing it.  Module-level
    statements execute at import, so they are included, but a lambda stored
    in a table is not executed by being defined: it is followed only where a
    case selects its entry.
    """
    calls: list[tuple[ast.Call, dict[str, Any], str | None]] = []
    _case_strings_uncached(nodes, selector, constants, calls_out=calls)
    return calls


def _case_strings_uncached(
    nodes: list[ast.AST],
    selector: dict[str, Any],
    constants: dict[str, Any],
    *,
    calls_out: list[tuple[ast.Call, dict[str, Any], str | None]] | None = None,
) -> set[str]:
    strings: set[str] = set()
    namespace = {**constants, **selector}

    # A parametrized pytest fixture commonly aliases ``request.param`` to a
    # local discriminator (``kind = request.param``).  Bind that local to the
    # concrete fixture selector so branches inside the fixture/builder are
    # resolved case by case rather than unioned across every parameter.
    for owner in nodes:
        if not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)) or owner.name not in selector:
            continue
        fixture_value = selector[owner.name]
        for child in _walk(owner):
            if not isinstance(child, (ast.Assign, ast.AnnAssign)) or child.value is None:
                continue
            if _dotted_name(child.value) != "request.param":
                continue
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            for target in targets:
                for name in _target_names(target):
                    namespace[name] = fixture_value

    # Selector-indexed builder tables (``SYSTEMS[name]``,
    # ``_KIND_BUILDERS[kind]``) make all helper names statically reachable.
    # Identify the selected helper and suppress the unselected siblings while
    # retaining the table assignment itself as provenance.
    selector_values: set[str] = set()

    def collect_selector_values(value: Any) -> None:
        if isinstance(value, str):
            selector_values.add(value)
        elif isinstance(value, dict):
            for child in value.values():
                collect_selector_values(child)
        elif isinstance(value, list):
            for child in value:
                collect_selector_values(child)

    collect_selector_values(selector)
    dispatch_helpers: set[str] = set()
    selected_helpers: set[str] = set()

    def helper_expressions(candidate: Any) -> set[str]:
        expression = getattr(candidate, "gdd_expression", None)
        if isinstance(expression, str):
            return {expression}
        if isinstance(candidate, dict):
            return {item for child in candidate.values() for item in helper_expressions(child)}
        if isinstance(candidate, (tuple, list, set, frozenset)):
            return {item for child in candidate for item in helper_expressions(child)}
        return set()

    for value in constants.values():
        if not isinstance(value, dict):
            continue
        mapped_helpers = {key: helper_expressions(candidate) for key, candidate in value.items() if isinstance(key, str)}
        dispatch_helpers.update(item for items in mapped_helpers.values() for item in items)
        selected_helpers.update(item for key in selector_values for item in mapped_helpers.get(key, ()))

    selector_indexed_tables: set[str] = set()
    for owner in nodes:
        for child in _walk(owner):
            if not isinstance(child, ast.Subscript) or not isinstance(child.value, ast.Name):
                continue
            index_names = {name.id for name in ast.walk(child.slice) if isinstance(name, ast.Name)}
            if not index_names.intersection(selector):
                continue
            try:
                _literal(child, namespace)
            except Exception:
                continue
            selector_indexed_tables.add(child.value.id)

    def collect_strings(value: Any) -> None:
        if isinstance(value, str):
            strings.add(value)
        elif isinstance(value, dict):
            for key, child in value.items():
                collect_strings(key)
                collect_strings(child)
        elif isinstance(value, (tuple, list, set, frozenset)):
            for child in value:
                collect_strings(child)

    functions = {node.name: node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    # Module-level path constants the evaluator leaves unresolved whose single
    # assignment spells a UPF file (``_UPF = _FIX / "Li.ccECP.upf"``).
    assignment_counts: Counter[str] = Counter(
        target.id
        for owner in nodes
        if isinstance(owner, ast.Assign) and owner.value is not None
        for target in owner.targets
        if isinstance(target, ast.Name)
    )
    upf_path_constants = {
        target.id
        for owner in nodes
        if isinstance(owner, ast.Assign) and owner.value is not None and _names_upf_literal(owner.value)
        for target in owner.targets
        if isinstance(target, ast.Name) and assignment_counts[target.id] == 1
    }
    called_helpers = {
        _dotted_name(child.func).rsplit(".", 1)[-1] for owner in nodes for child in _walk(owner) if isinstance(child, ast.Call)
    }

    called_entries: set[str] = set()
    # Names some node the visitor can reach reads (issue #153): a loop whose
    # variable is not among them cannot change between iterations.
    loaded_names = {
        child.id for owner in nodes for child in _walk(owner) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }
    # Depth of long literal loops visited once with their variable unbound
    # (issue #153).  Within one, an unresolved dispatch-table key may be any
    # key the per-value unrolling would have selected.
    symbolic_loop_depth = 0
    # Table lambdas whose body such a visit is following, so that an entry
    # subscripting its own table (``_T = {0: lambda: _T[i]()}``) is not
    # re-entered without bound.
    symbolic_entries: set[ast.Lambda] = set()

    def visit(node: ast.AST, scope: dict[str, Any], stack: tuple[str, ...] = ()) -> None:
        nonlocal symbolic_loop_depth
        if _case_string_generic_only(type(node)):
            # No branch below applies to this node class: same as falling
            # through every check to the final child loop.
            for child in _child_nodes(node):
                visit(child, scope, stack)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A dispatch-table helper runs for this case only when the case
            # selected it (selector key) or called the entry it selected; a
            # root visit (empty stack) never counts as such a call.
            if (
                node.name in dispatch_helpers
                and node.name not in selected_helpers
                and not (stack and node.name in called_entries)
            ):
                return
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            positional = [*node.args.posonlyargs, *node.args.args]
            positional_defaults = zip(positional[-len(node.args.defaults) :], node.args.defaults, strict=False)
            for argument, default in positional_defaults:
                if argument.arg not in scope:
                    visit(default, scope, stack)
            for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=False):
                if default is not None and argument.arg not in scope:
                    visit(default, scope, stack)
            visit_block(body, scope, (*stack, node.name), set())
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {name for target in targets for name in _target_names(target)}
            selected_tables = names.intersection(selector_indexed_tables)
            if selected_tables:
                for name in selected_tables:
                    table = constants.get(name)
                    if not isinstance(table, dict):
                        continue
                    for key in selector_values:
                        if key in table:
                            collect_strings(_yaml_value(table[key]))
                return
        if isinstance(node, ast.Subscript):
            imported_entry = _imported_table_entry_strings(node, scope)
            if imported_entry is not None:
                # ``module_alias.TABLE[key]``: only the selected entry of the
                # imported test table, never the whole table.
                strings.update(imported_entry)
                visit(node.slice, scope, stack)
                return
            try:
                selected_value = _literal(node, scope)
            except Exception:
                if symbolic_loop_depth:
                    # Inside a long loop visited with its variable unbound the
                    # key is unknown: every entry of a dict-literal table may be
                    # the one a pass selects, so each entry contributes as a
                    # selected one would (its literals; a lambda's body runs).
                    # A table bound in scope is read whole below via its name.
                    for entry in _table_entries(node, nodes):
                        if not isinstance(entry, ast.Lambda):
                            strings.update(_string_values(entry))
                        for child in ast.walk(entry):
                            if isinstance(child, ast.Lambda) and child not in symbolic_entries:
                                symbolic_entries.add(child)
                                try:
                                    visit(child.body, scope, stack)
                                finally:
                                    symbolic_entries.discard(child)
            else:
                if selected_value is None:
                    # Unresolved table entry (runtime path root): read the selected
                    # entry's own literals, never the whole table.
                    strings.update(_selected_table_entry_strings(node, nodes, scope))
                else:
                    collect_strings(_yaml_value(selected_value))
                # A selected ``lambda`` entry of a dict-literal dispatch table
                # (``_SYSTEMS[system]()``) runs only for this case: follow its body.
                entry = _selected_table_entry(node, nodes, scope)
                if isinstance(entry, ast.Lambda):
                    if not symbolic_loop_depth:
                        visit(entry.body, scope, stack)
                    elif entry not in symbolic_entries:
                        # Inside a symbolic long loop a resolved key may name a
                        # lambda whose body this traversal is already following
                        # (``1: lambda n: ... TABLE[1](n - 1)``): the same
                        # active-entry guard applies.
                        symbolic_entries.add(entry)
                        try:
                            visit(entry.body, scope, stack)
                        finally:
                            symbolic_entries.discard(entry)
                return
        if isinstance(node, ast.IfExp):
            try:
                condition_names = {name.id for name in ast.walk(node.test) if isinstance(name, ast.Name)}
                if any(isinstance(getattr(scope.get(name), "gdd_expression", None), str) for name in condition_names):
                    raise ValueError("condition depends on an inert runtime expression")
                chosen = [node.body if bool(_literal(node.test, scope)) else node.orelse]
            except Exception:  # unresolved runtime condition: retain both arms
                chosen = [node.test, node.body, node.orelse]
            for child in chosen:
                visit(child, scope, stack)
            return
        if isinstance(node, ast.Attribute):
            # ``module_alias.CONSTANT`` of a repository test module bound as an
            # inert import token: read that constant's own literals.
            # A dict table referenced as a whole (``monkeypatch.setitem(alias._PATTERNS, ...)``)
            # is a catalog, not a use of every entry; its entries count only
            # where a case subscripts them (see the ``ast.Subscript`` branch).
            constant = _alias_constant_expression(node, scope)
            if constant is not None and not _imported_constant_is_dict(constant):
                strings.update(_imported_test_constant_strings(constant))
        if isinstance(node, (ast.BinOp, ast.Call)) and _names_upf_literal(node):
            # A complete standalone UPF path expression (``_QE_ROOT / "Li.ccECP.upf"``):
            # its basename alone cannot distinguish the standalone file from
            # same-named copies inside QE save directories.
            path = _explicit_fixture_path(_inline_inert_path_roots(node, scope, nodes))
            if path is not None and path.lower().endswith(".upf"):
                strings.add(path)
        if isinstance(node, ast.If):
            try:
                condition_names = {name.id for name in ast.walk(node.test) if isinstance(name, ast.Name)}
                if any(isinstance(getattr(scope.get(name), "gdd_expression", None), str) for name in condition_names):
                    raise ValueError("condition depends on an inert runtime expression")
                branch = node.body if bool(_literal(node.test, scope)) else node.orelse
            except Exception:  # unresolved runtime condition: retain both branches
                branch = [*node.body, *node.orelse]
            for child in branch:
                visit(child, scope, stack)
            return
        if isinstance(node, (ast.For, ast.AsyncFor)):
            try:
                values = list(_literal(node.iter, scope))
            except Exception:
                values = []
            target_names = _target_names(node.target)
            if values and len(target_names) == 1:
                target_name = next(iter(target_names))
                if target_name not in loaded_names and all(type(value) in _INERT_LOOP_VALUE_TYPES for value in values):
                    # No visited node reads the loop variable, and a scalar
                    # value adds no string where a store visits it: every
                    # iteration repeats the same visit.  Only a dispatch entry
                    # first called in one pass can change the next pass, so
                    # repeat until a pass adds none (at most ``len(values)``).
                    for value in values:
                        nested = dict(scope)
                        nested[target_name] = value
                        entries_before = len(called_entries)
                        for child in node.body:
                            visit(child, nested, stack)
                        if len(called_entries) == entries_before:
                            break
                elif len(values) <= _MAX_UNROLLED_LOOP_ITERATIONS:
                    for value in values:
                        nested = dict(scope)
                        nested[target_name] = value
                        for child in node.body:
                            visit(child, nested, stack)
                else:
                    # A long literal loop whose body reads its variable: visit
                    # the iterable and the body once with the variable
                    # unbound, as for a runtime iterable.  Every branch or
                    # argument that depended on the value is then unresolved
                    # and retained: a helper formal it feeds is unbound, and a
                    # table subscript it keys visits every entry (a superset
                    # of the unrolled union).
                    visit(node.iter, scope, stack)
                    nested = {name: value for name, value in scope.items() if name != target_name}
                    symbolic_loop_depth += 1
                    try:
                        for child in node.body:
                            visit(child, nested, stack)
                    finally:
                        symbolic_loop_depth -= 1
                for child in node.orelse:
                    visit(child, scope, stack)
                return
        if isinstance(node, ast.Call):
            # Follow local helpers with concrete arguments.  This resolves
            # patterns such as ``_ham("mol_cart_sd_jdtn")`` inside an otherwise
            # unparameterized test, rather than unioning every branch of
            # ``_ham(pattern)`` into that one case.
            helper_name = _dotted_name(node.func).rsplit(".", 1)[-1]
            helper = functions.get(helper_name)
            selected_lambda = scope.get(node.func.id) if isinstance(node.func, ast.Name) else None
            if helper is None and isinstance(selected_lambda, _SelectedLambda):
                # The case calls the lambda of the table row it selected: its body
                # runs for this case, with its parameters bound to the call arguments.
                lambda_scope = dict(scope)
                parameters = [*selected_lambda.node.args.posonlyargs, *selected_lambda.node.args.args]
                for parameter in parameters:
                    lambda_scope.pop(parameter.arg, None)
                for parameter, argument in zip(parameters, node.args, strict=False):
                    try:
                        lambda_scope[parameter.arg] = _literal(argument, scope)
                    except Exception:
                        pass
                visit(selected_lambda.node.body, lambda_scope, stack)
            if helper is None and isinstance(node.func, ast.Name):
                # A local bound to a module function (``builder, _ = _FIXTURES[name]``; ``builder()``).
                target = getattr(scope.get(node.func.id), "gdd_expression", None)
                if isinstance(target, str) and target in functions:
                    helper, helper_name = functions[target], target
                    # The case selected this table entry by its key: it runs.
                    called_entries.add(target)
            # Calling a factory fixture's value (``basis_pairs(case)``) runs the
            # function the fixture returns, not the fixture body itself.
            factory = _fixture_returned_function(helper) if helper is not None else None
            if factory is not None:
                helper, helper_name = factory, f"{helper_name}.{factory.name}"
            if helper is not None and helper_name not in stack:
                nested = dict(scope)
                formals = [
                    *helper.args.posonlyargs,
                    *helper.args.args,
                ]
                if formals and formals[0].arg in {"self", "cls"} and isinstance(node.func, ast.Attribute):
                    formals = formals[1:]
                for formal, argument in zip(formals, node.args, strict=False):
                    try:
                        nested[formal.arg] = _literal(argument, scope)
                    except Exception:
                        # An unresolved argument leaves its formal unbound: the
                        # caller's own binding of that name is a different
                        # variable.
                        nested.pop(formal.arg, None)
                for keyword in node.keywords:
                    if keyword.arg is None:
                        continue
                    try:
                        nested[keyword.arg] = _literal(keyword.value, scope)
                    except Exception:
                        nested.pop(keyword.arg, None)
                visit_function(helper, nested, stack)
            if calls_out is not None:
                calls_out.append((node, scope, stack[-1] if stack else None))
            # The callee expression runs too when it is not a plain dotted name:
            # ``rs_ham("x").save_to_hdf5(path)``, ``_SYSTEMS[system]()``.
            callee = (node.func,) if any(isinstance(child, (ast.Call, ast.Subscript)) for child in ast.walk(node.func)) else ()
            for child in (*callee, *node.args, *(keyword.value for keyword in node.keywords)):
                visit(child, scope, stack)
            return
        if isinstance(node, ast.Name) and node.id not in scope and node.id in upf_path_constants:
            path = _explicit_fixture_path(_inline_inert_path_roots(node, scope, nodes))
            if path is not None and path.lower().endswith(".upf"):
                strings.add(path)
            return
        if isinstance(node, ast.Name) and node.id in scope:
            collect_strings(_yaml_value(scope[node.id]))
            expression = getattr(scope[node.id], "gdd_expression", None)
            strings.update(_imported_test_constant_strings(expression))
            if isinstance(expression, str) and ".upf" in expression.lower():
                path = _explicit_fixture_path(expression)
                if path is not None and path.lower().endswith(".upf"):
                    strings.add(path)
            return
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value:
            strings.add(node.value)
        for child in _child_nodes(node):
            visit(child, scope, stack)

    def resolved_condition(test: ast.expr, scope: dict[str, Any]) -> bool | None:
        condition_names = {name.id for name in ast.walk(test) if isinstance(name, ast.Name)}
        if any(isinstance(getattr(scope.get(name), "gdd_expression", None), str) for name in condition_names):
            return None
        try:
            return bool(_literal(test, scope))
        except Exception:
            return None

    def visit_block(
        statements: list[ast.stmt], scope: dict[str, Any], stack: tuple[str, ...], bound_names: set[str]
    ) -> tuple[dict[str, Any], bool]:
        """Visit statements in execution order along the case-resolved path.

        A plain-data assignment executed on this path (``family, method, kind =
        case.split("-")``, ``patterns = {...}``, ``case = "WF-PATTERN-016"`` in
        a selected branch) binds its names for the statements that follow, so
        later branches and table lookups resolve per case.  A store of such a
        name on a path that is not resolved (loop, ``with``, ``try``, an
        unresolved branch) unbinds it again.  A resolved branch that ends in
        ``return``/``raise`` ends the block: the statements after it do not run
        for this case.  Returns the scope after the block and whether the block
        terminated.
        """
        local = scope
        for child in statements:
            if isinstance(child, ast.If):
                taken = resolved_condition(child.test, local)
                if taken is not None:
                    local, terminated = visit_block(child.body if taken else child.orelse, local, stack, bound_names)
                    if terminated:
                        return local, True
                    continue
            if isinstance(child, (ast.Return, ast.Raise)):
                visit(child, local, stack)
                return local, True
            bound = _plain_assignment_bindings(child, local)
            if not bound or not _pure_plain_expression(child.value, local):
                visit(child, local, stack)
            if not bound:
                # ``builder, flag = _CASES[case]``: a row lambda (and every plain
                # literal element of an unpacked row) is bound for the statements
                # that follow (see ``_SelectedLambda`` / ``_selected_row_bindings``).
                bound = _selected_row_bindings(child, nodes, local)
            stale = _stored_names(child).intersection(bound_names) - set(bound)
            if stale:
                local = {name: value for name, value in local.items() if name not in stale}
                bound_names.difference_update(stale)
            # A bound literal (``patterns = {...}``) contributes its strings
            # where it is used (``patterns[fixture]`` selects one entry), not as
            # the union of every entry at its definition.
            if bound:
                local = {**local, **bound}
                bound_names.update(bound)
        return local, False

    # Memoized function visits.  Besides the fixed per-call context (nodes,
    # constants, selector-derived helper/table sets), ``visit(function, scope,
    # stack)`` reads only ``stack`` and the scope entries named by an
    # identifier (``ast.Name``/``ast.arg``) somewhere in ``function`` or in a
    # local helper it can call: the visitor never mutates a scope (it copies
    # one before binding) and ``_literal`` evaluates only names spelled in the
    # expression.  Equal keys therefore add exactly the same strings.
    visit_memo: dict[Any, tuple[frozenset[str], list[Any]]] | None = None
    relevant_names: dict[ast.AST, tuple[str, ...]] = {}
    # Call collection needs every visit to run (a memo hit would skip the
    # recorded call sites), so it bypasses the string memo.
    memo_tables = _memo_table("case_string_visits") if calls_out is None else None
    if memo_tables is not None:
        context = (
            tuple(nodes),
            id(constants),
            frozenset(selector_values),
            frozenset(dispatch_helpers),
            frozenset(selected_helpers),
            frozenset(selector_indexed_tables),
        )
        entry = memo_tables.get(context)
        if entry is None or entry[0] is not constants:
            entry = (constants, {})
            memo_tables[context] = entry
        _constants, visit_memo = entry
        # The identifier closure depends on ``functions`` (that is, on
        # ``nodes``) only, so it is shared by every case of the same nodes.
        relevant_names = _memo_table("case_string_names").setdefault(tuple(nodes), {})

    def function_names(function: ast.AST) -> tuple[str, ...]:
        names = relevant_names.get(function)
        if names is None:
            found: set[str] = set()
            pending = [function]
            seen: set[int] = set()
            while pending:
                current = pending.pop()
                if id(current) in seen:
                    continue
                seen.add(id(current))
                for child in _walk(current):
                    if isinstance(child, ast.Name):
                        found.add(child.id)
                    elif isinstance(child, ast.arg):
                        found.add(child.arg)
                    elif isinstance(child, ast.Call):
                        callee = functions.get(_dotted_name(child.func).rsplit(".", 1)[-1])
                        if callee is not None:
                            pending.append(callee)
            names = tuple(sorted(found))
            relevant_names[function] = names
        return names

    def visit_function(function: ast.AST, scope: dict[str, Any], stack: tuple[str, ...]) -> None:
        nonlocal strings
        if visit_memo is None:
            visit(function, scope, stack)
            return
        retained: list[Any] = []
        values = []
        for name in function_names(function):
            if name not in scope:
                values.append(None)
                continue
            value = scope[name]
            if value is None or type(value) in (str, int, float, complex, bool):
                values.append((type(value), repr(value)))
            else:
                # Identity of a value retained by the memo entry, so the id
                # cannot be reused by another object while the entry exists.
                values.append(id(value))
                retained.append(value)
        # A visit inside a long symbolic loop also adds every entry of an
        # unresolved table subscript, so it never shares a memo entry with one
        # outside such a loop.
        key = (function, stack, tuple(values), bool(symbolic_loop_depth))
        cached = visit_memo.get(key)
        if cached is not None:
            strings.update(cached[0])
            return
        outer = strings
        strings = set()
        try:
            visit(function, scope, stack)
            added = strings
        finally:
            strings = outer
        strings.update(added)
        visit_memo[key] = (frozenset(added), retained)

    roots = [
        node
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (
            node.name not in called_helpers
            # A fixture selected by value (indirect parametrization) runs as a
            # root; an inert token forwarded under the fixture's name (a
            # forwarding test passing its ``basis_pairs`` fixture on) does not.
            or (node.name in selector and not _inert_selector_value(selector[node.name]))
        )
    ]
    for node in roots:
        visit_function(node, namespace, ())
    if calls_out is not None:
        # Module-level statements run at import for every case.  Their table
        # literals are skipped: a stored lambda runs only where a case selects
        # it, and a selector-indexed table contributes only its selected entry.
        for owner in nodes:
            if isinstance(owner, (ast.Assign, ast.AnnAssign)) and owner.value is not None:
                calls_out.extend((child, namespace, None) for child in _calls_outside_lambdas(owner.value))
    return strings


def _calls(nodes: list[ast.AST]) -> list[str]:
    return sorted(
        {
            name
            for node in nodes
            for child in _walk(node)
            if isinstance(child, ast.Call)
            for name in [_dotted_name(child.func)]
            if name
        }
    )


def _construction_calls(calls: list[str]) -> list[str]:
    selected = [
        name
        for name in calls
        if any(word.lower() in name.lower() for word in _CONSTRUCTION_WORDS)
        or any(word in name for word in _PHYSICS_SYMBOL_WORDS)
        or any(name.endswith(symbol) for symbol in _EXPLICIT_CONSTRUCTION_SYMBOLS)
    ]
    return selected or [name for name in calls if not any(word in name.lower() for word in _ASSERTION_WORDS)][:8]


def _physics_symbols(calls: list[str]) -> list[str]:
    return [
        name
        for name in calls
        if any(word in name for word in _PHYSICS_SYMBOL_WORDS) or any(word in name.lower() for word in _PHYSICS_API_WORDS)
    ]


# A reachable call through which a test itself materializes a jqmc physics
# object: a physics ``*_data`` constructor (or its ``.build``), a TREXIO load,
# a reviewed Hamiltonian builder, or a production QMC driver. A test reaching
# one cannot declare ``gdd_physics_object(mode="not-required")``.
_PHYSICS_OBJECT_DATA_PREFIXES = (
    "Hamiltonian",
    "Wavefunction",
    "Jastrow",
    "Coulomb",
    "Structure",
    "MOs",
    "AOs",
    "AGP",
    "SD",
    "Geminal",
    "Ewald",
    "MPC",
)
_PHYSICS_OBJECT_BUILDERS = frozenset(
    {
        "read_trexio_file",
        "rs_ham",
        "_build_ham",
        "_build_ham_for_cell",
        "_make_ham",
        "build_hamiltonian_from_wavefunction",
        "load_twist",
        "MCMC",
        "GFMC_t",
        "GFMC_n",
        "run_qe2jqmc",
        "convert_qe_to_twist_average_input",
    }
)


def _physics_object_constructors(calls: list[str]) -> list[str]:
    """Reachable calls that construct a jqmc physics object (see above)."""
    found = []
    for name in calls:
        parts = name.split(".")
        if any(part in _PHYSICS_OBJECT_BUILDERS for part in parts) or any(
            part.endswith("_data") and part.startswith(_PHYSICS_OBJECT_DATA_PREFIXES) for part in parts
        ):
            found.append(name)
    return sorted(set(found))


def _runtime_physics_object_contract(
    scope: str,
    physics_symbols: list[str],
    observations: list[dict[str, Any]],
) -> tuple[str, str]:
    """Return the machine disposition and its stable reason code."""
    if scope == "not-applicable":
        return "not-required", "no-wavefunction-or-component"
    decisions = [row for row in observations if row.get("role") != "non-vacuity"]
    if decisions and all(row.get("kind") == "expected-error" for row in decisions):
        return "not-required", "expected-error-before-materialization"
    if not physics_symbols:
        return "not-required", "raw-numerical-component-fixture"
    return "required", "exercised-runtime-route"


def _inferred_wavefunction_scope(
    calls: list[str],
    sources: list[dict[str, str]],
    declared_scope: str,
    *,
    production_driver_route: bool = False,
) -> str:
    lowered = " ".join(calls).lower()
    leaves = {name.rsplit(".", 1)[-1].lower() for name in calls}
    # TREXIO ingestion returns many components at once. A recognized consumer
    # narrows the scope to the component actually exercised. With ingestion
    # alone retain the complete loaded-object scope and its fixture obligation;
    # that fallback is not evidence of a full-wavefunction physics evaluation.
    trexio_ingestion = "read_trexio_file" in leaves
    full_route_calls = {
        "rs_ham",
        "_run_gfmc",
        "_build_ham_for_cell",
        "_build_ham",
        "_make_ham",
        "mcmc",
        "gfmc_n",
        "gfmc_t",
        "load_twist",
        "build_hamiltonian_from_wavefunction",
        "run_qe2jqmc",
        "convert_qe_to_twist_average_input",
    }
    if production_driver_route or leaves.intersection(full_route_calls | set(_WAVEFUNCTION_KINETIC_APIS)):
        return "full-wavefunction"
    if any(
        token in lowered
        for token in (
            "wavefunction_data",
            "hamiltonian_data",
            "evaluate_wavefunction",
            "evaluate_ln_wavefunction",
            "compute_local_energy",
        )
    ):
        return "full-wavefunction"
    if declared_scope == "not-applicable" and not trexio_ingestion:
        return declared_scope
    # Generic validation/replacement of a loaded tuple is not a component
    # evaluation. Keep the complete ingestion-object scope for those tests.
    component_calls = [name for name in calls if name.rsplit(".", 1)[-1] not in {"sanity_check", "replace"}]
    component_lowered = " ".join(component_calls).lower() if trexio_ingestion else lowered
    if any(token in component_lowered for token in ("jastrow", "compute_j1", "compute_j2", "compute_j3")):
        return "jastrow-component"
    if any(
        token in component_lowered for token in ("geminal", "determinant", "compute_det", "agp_data", "agpn_data", "sd_data")
    ):
        return "determinant-component"
    if any(token in component_lowered for token in ("compute_mos", "mos_data", "molecular_orbital")):
        return "molecular-orbital-component"
    if any(token in component_lowered for token in ("compute_aos", "aos_", "atomic_orbital")):
        return "atomic-orbital-component"
    if trexio_ingestion:
        return "full-wavefunction"
    return "not-applicable"


def _is_coordinate_metadata(value: str) -> bool:
    """Reject nuclear centres, derivatives and counts even in coordinate-like names."""
    leaf = value.rsplit(".", 1)[-1].split("=", 1)[0].lstrip("_")
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    normalized_leaf = re.sub(r"[^a-z0-9]+", "_", leaf.lower()).strip("_")
    if normalized_leaf.startswith(("grad_", "gradient_", "interaction_", "interactions_", "mesh_")):
        return True
    if re.search(r"(?:^|_)r_(?:up|dn|cart|carts|elec)(?:_carts?)?_(?:min|max)$", normalized_leaf):
        return True
    if re.search(r"(?:^|_)(?:n|num|number|total)(?:_of)?_(?:electrons?|walkers?)(?:_|$)", normalized):
        return True
    if "core_electron" in normalized or "position_deriv" in normalized:
        return True
    tokens = set(normalized.split("_"))
    if tokens.intersection({"atom", "atomic", "nuclear", "nucleus", "ion", "ionic"}) and not tokens.intersection(
        {"electron", "electrons", "walker", "walkers"}
    ):
        return True
    if (
        ".structure_data.positions" in value
        or ".structure.positions" in value
        or re.search(r"(?:^|\.)(?:structure|struct|sd|aos_data|orb_data)[^=]*\.?_?positions_cart", value.lower())
    ):
        return True
    # Uppercase R is the repository convention for nuclear centres; lowercase
    # r is an electron or AO-evaluation point.  Uppercase constants that say
    # R_UP/R_DN explicitly remain electron coordinates.
    if leaf.startswith(("R_cart", "R_carts")):
        return True
    return False


def _is_coordinate_symbol(value: str) -> bool:
    """Recognize electron/evaluation coordinates, never nuclear/count metadata."""
    if _is_coordinate_metadata(value):
        return False
    leaf = value.rsplit(".", 1)[-1].split("=", 1)[0].lstrip("_")
    normalized_leaf = re.sub(r"[^a-z0-9]+", "_", leaf.lower()).strip("_")
    if leaf[:1].islower() and re.fullmatch(r"r|ru\d*|rd\d*", leaf.lower()):
        return True
    if normalized_leaf == "r_pbc":
        return True
    if any(
        re.search(rf"(?:^|_){re.escape(word)}s?(?:_|$)", normalized_leaf)
        for word in ("r_up", "r_dn", "r_cart", "r_elec", "coords")
    ):
        return True
    if re.search(r"(?:^|_)walker(?:s)?(?:_|$)", normalized_leaf):
        walker_tokens = set(normalized_leaf.split("_"))
        return bool(walker_tokens.intersection({"batch", "state", "projected", "coordinates", "positions"}))
    tokens = set(normalized_leaf.split("_"))
    return bool(
        tokens.intersection({"electron", "electrons"})
        and tokens.intersection({"position", "positions", "configuration", "configurations", "coords"})
    )


def _is_production_driver_route(calls: list[str], nodes: list[ast.AST]) -> bool:
    """Identify tests whose call actually constructs/runs a walker driver.

    ``_cli()`` is a production driver route in the end-to-end CLI tests, but
    not when the driver class is replaced by a sentinel that only audits text
    formatting.  Keeping that distinction here prevents both a missing walker
    fixture contract and a fictitious one on the counter/formatting tests.
    """
    leaves = {name.rsplit(".", 1)[-1].lower() for name in calls}
    if leaves.intersection({"mcmc", "gfmc_n", "gfmc_t"}):
        return True
    if "_cli" not in leaves:
        return False
    # Logging-only entrypoint tests stop at the header hook, before distributed
    # initialization or input construction. Recognize a scoped replacement with
    # an unconditionally raising local function, not a marker/name exemption.
    raising = set()
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            if len(body) == 1 and isinstance(body[0], ast.Raise) and body[0].exc is not None:
                raising.add(node.name)
    cli_calls = {
        child
        for node in nodes
        for child in _walk(node)
        if isinstance(child, ast.Call) and _dotted_name(child.func).rsplit(".", 1)[-1] == "_cli"
    }
    for node in nodes:
        for call in _walk(node):
            if (
                isinstance(call, ast.Call)
                and _dotted_name(call.func).rsplit(".", 1)[-1] in {"setattr", "delattr"}
                and any(isinstance(arg, ast.Constant) and arg.value == "_print_header" for arg in call.args)
                and not (len(call.args) == 3 and isinstance(call.args[2], ast.Name) and call.args[2].id in raising)
            ):
                return True  # A replacement may undo the early-stop hook.
    stopped = set()
    for node in nodes:
        for block in _walk(node):
            if not isinstance(block, ast.With) or len(block.items) != 1:
                continue
            item = block.items[0]
            if (
                not isinstance(item.context_expr, ast.Call)
                or _dotted_name(item.context_expr.func) != "monkeypatch.context"
                or not isinstance(item.optional_vars, ast.Name)
            ):
                continue
            setters = {}
            for statement in block.body:
                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                    call = statement.value
                    if (
                        _dotted_name(call.func) == f"{item.optional_vars.id}.setattr"
                        and len(call.args) == 3
                        and isinstance(call.args[1], ast.Constant)
                        and call.args[1].value == "_print_header"
                        and isinstance(call.args[2], ast.Name)
                        and call.args[2].id in raising
                    ):
                        setters[_dotted_name(call.args[0])] = call
                for call in _walk(statement):
                    if call in cli_calls and isinstance(call.func, ast.Attribute):
                        target = _dotted_name(call.func.value)
                        if target in setters and call.lineno > setters[target].lineno:
                            stopped.add(call)
    if cli_calls and stopped == cli_calls:
        return False
    source = "\n".join(_source_expression(node) for node in nodes)
    return not re.search(r"monkeypatch\.setattr\([^\n]+(?:MCMC|VMC|LRDMC)[^\n]+Sentinel", source)


def _coordinate_symbols(nodes: list[ast.AST]) -> list[str]:
    # Reuse the same source walk for the separate classification passes.
    flattened = [child for node in nodes for child in _walk(node)]
    # These module APIs take their evaluation coordinates as argument 1 or
    # r_carts=. Do not recognize arbitrary `points`/`positions` identifiers:
    # the consuming API, not their spelling or a TREXIO loader, establishes
    # that they are AO/MO evaluation points. Keep this signature list exact;
    # similarly named constructors and bound methods have different inputs.
    evaluation_apis = {
        "compute_AOs",
        "compute_AOs_grad",
        "compute_AOs_laplacian",
        "compute_AOs_value_grad_lap",
        "compute_MOs",
        "compute_MOs_grad",
        "compute_MOs_laplacian",
        "compute_MOs_value_grad_lap",
        "sphe_basis_param_jvp",
        "pw_value_grad_lap_dcell",
        "bspline_value_grad_lap_dcell",
        # Jitted overlap-sample AO evaluator (aos_data, r_carts).
        "_compute_composite_overlap_AOs",
    }
    # ``evaluator = ao._compute_composite_overlap_AOs`` binds a local name to
    # one of these exact APIs; calls through that name have the same slots.
    evaluation_aliases = {
        target.id
        for child in flattened
        if isinstance(child, ast.Assign) and isinstance(child.value, (ast.Name, ast.Attribute))
        for target in child.targets
        if isinstance(target, ast.Name) and _dotted_name(child.value).rsplit(".", 1)[-1] in evaluation_apis
    }
    coordinate_consumers = (
        "compute_aos",
        "compute_mos",
        "evaluate_swct",
        "ao_cart_deriv",
        "compute_aos_bspline",
        "compute_aos_vgl",
        "pgto_",
        "bspline_value_grad_lap",
    )
    bare_r_is_coordinate = any(
        isinstance(child, ast.Call)
        and any(token in _dotted_name(child.func).lower() for token in coordinate_consumers)
        and any(
            isinstance(descendant, ast.Name) and descendant.id == "r"
            for argument in (*child.args, *(keyword.value for keyword in child.keywords))
            for descendant in ast.walk(argument)
        )
        for child in flattened
    )
    called_names = {_dotted_name(child.func).rsplit(".", 1)[-1] for child in flattened if isinstance(child, ast.Call)}
    values = {
        value
        for child in flattened
        for value in [
            _dotted_name(child) if isinstance(child, ast.Attribute) else child.id if isinstance(child, ast.Name) else ""
        ]
        if value
        and not value.lower().lstrip("_").startswith(("num_", "number_"))
        and _is_coordinate_symbol(value)
        and value.rsplit(".", 1)[-1] not in called_names
        and (value.rsplit(".", 1)[-1].lstrip("_") != "r" or bare_r_is_coordinate)
    }
    values.update(
        f"{child.arg}={_source_expression(child.value)}"
        for child in flattened
        if isinstance(child, ast.keyword)
        and child.arg
        and _is_coordinate_symbol(child.arg)
        and not (isinstance(child.value, ast.Constant) and isinstance(child.value.value, (int, float, bool)))
        and not (isinstance(child.value, (ast.Name, ast.Attribute)) and _is_coordinate_metadata(_dotted_name(child.value)))
    )
    for child in flattened:
        if not isinstance(child, ast.Call):
            continue
        api = _dotted_name(child.func).rsplit(".", 1)[-1]
        if api in evaluation_apis or (isinstance(child.func, ast.Name) and api in evaluation_aliases):
            coordinate_slots = ((1, "r_carts"),)
        elif api == "compute_geminal_all_elements":
            coordinate_slots = ((1, "r_up_carts"), (2, "r_dn_carts"))
        else:
            continue
        for position, keyword in coordinate_slots:
            argument = next((item.value for item in child.keywords if item.arg == keyword), None)
            if argument is None and len(child.args) > position:
                argument = child.args[position]
            if isinstance(argument, (ast.Name, ast.Attribute)):
                symbol = _dotted_name(argument)
                if not _is_coordinate_metadata(symbol):
                    values.add(symbol)
    return sorted(values)


_UNRESOLVED_BRANCH = object()
_BRANCH_SELECTOR_TYPES = (str, int, float, bool, type(None))


def _branch_condition(test: ast.expr, selector: dict[str, Any], *, bool_names: bool = False) -> Any:
    """Statically decide an ``if`` test from one parametrize case.

    Only ``param ==/!=/in/not in literal`` (``is``/``is not`` against
    None/True/False), ``not``, ``and`` and ``or`` over such terms are decided.
    With ``bool_names`` (finiteness stability verification only) a bare
    parameter whose case value is a ``bool`` is decided too.
    Anything else is ``_UNRESOLVED_BRANCH``: the caller keeps every branch.
    """
    if bool_names and isinstance(test, ast.Name) and type(selector.get(test.id)) is bool:
        return selector[test.id]
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        value = _branch_condition(test.operand, selector, bool_names=bool_names)
        return value if value is _UNRESOLVED_BRANCH else not value
    if isinstance(test, ast.BoolOp):
        values = [_branch_condition(value, selector, bool_names=bool_names) for value in test.values]
        if any(value is _UNRESOLVED_BRANCH for value in values):
            return _UNRESOLVED_BRANCH
        return all(values) if isinstance(test.op, ast.And) else any(values)
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return _UNRESOLVED_BRANCH
    left, operator, right = test.left, test.ops[0], test.comparators[0]
    if not isinstance(left, ast.Name) and isinstance(right, ast.Name) and isinstance(operator, (ast.Eq, ast.NotEq)):
        left, right = right, left
    if not isinstance(left, ast.Name) or left.id not in selector:
        return _UNRESOLVED_BRANCH
    value = selector[left.id]
    if type(value) not in _BRANCH_SELECTOR_TYPES:
        return _UNRESOLVED_BRANCH
    try:
        literal = ast.literal_eval(right)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _UNRESOLVED_BRANCH
    if isinstance(operator, (ast.Eq, ast.NotEq)):
        if type(literal) not in _BRANCH_SELECTOR_TYPES:
            return _UNRESOLVED_BRANCH
        return (value == literal) is isinstance(operator, ast.Eq)
    if isinstance(operator, (ast.In, ast.NotIn)):
        if not isinstance(literal, (tuple, list, set, frozenset)) or any(
            type(item) not in _BRANCH_SELECTOR_TYPES for item in literal
        ):
            return _UNRESOLVED_BRANCH
        return (value in literal) is isinstance(operator, ast.In)
    if isinstance(operator, (ast.Is, ast.IsNot)) and literal in (None, True, False) and type(literal) in (bool, type(None)):
        return (value is literal) is isinstance(operator, ast.Is)
    return _UNRESOLVED_BRANCH


def _branch_pruned_test(
    test: ast.FunctionDef | ast.AsyncFunctionDef, selector: dict[str, Any], *, bool_names: bool = False
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return ``test`` without the ``if`` branches this case never takes.

    The result is a shallow copy whose kept statements are the original AST
    nodes, so identity-keyed lexical lookups (imports, aliases) still apply.

    Conservative by construction: ``None`` (no pruning) when nothing was
    decided, and a parameter that the body rebinds (assignment, loop/with/
    except target, walrus, del, import, global/nonlocal) or that is not an
    argument of ``test`` is never used to decide a branch. Nested functions,
    lambdas and classes are kept whole; an undecidable ``if`` keeps every branch.
    """
    arguments = {argument.arg for argument in (*test.args.posonlyargs, *test.args.args, *test.args.kwonlyargs)}
    rebound: set[str] = set()
    for child in _walk(test):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            rebound.add(child.id)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            rebound.update(child.names)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            rebound.update((alias.asname or alias.name).split(".", 1)[0] for alias in child.names)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            rebound.add(child.name)
    usable = {
        name: value
        for name, value in selector.items()
        if name in arguments and name not in rebound and type(value) in _BRANCH_SELECTOR_TYPES
    }
    if not usable:
        return None
    pruned = False

    def prune_block(statements: list[ast.stmt]) -> list[ast.stmt]:
        kept: list[ast.stmt] = []
        for statement in statements:
            kept.extend(prune_statement(statement))
        return kept

    def prune_statement(statement: ast.stmt) -> list[ast.stmt]:
        # Kept statements are the ORIGINAL nodes (or shallow copies of their
        # containers): identity-keyed lexical indices still resolve them.
        nonlocal pruned
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return [statement]
        if isinstance(statement, ast.If):
            decided = _branch_condition(statement.test, usable, bool_names=bool_names)
            if decided is not _UNRESOLVED_BRANCH:
                pruned = True
                return prune_block(statement.body if decided else statement.orelse)
        replaced: dict[str, Any] = {}
        for field in ("body", "orelse", "finalbody"):
            block = getattr(statement, field, None)
            if isinstance(block, list) and block and all(isinstance(item, ast.stmt) for item in block):
                new_block = prune_block(block)
                if len(new_block) != len(block) or any(x is not y for x, y in zip(new_block, block)):
                    replaced[field] = new_block or [ast.Pass()]
        handlers = getattr(statement, "handlers", None)
        if isinstance(handlers, list) and handlers:
            new_handlers = []
            for handler in handlers:
                new_body = prune_block(handler.body)
                if len(new_body) != len(handler.body) or any(x is not y for x, y in zip(new_body, handler.body)):
                    handler = copy.copy(handler)
                    handler.body = new_body or [ast.Pass()]
                new_handlers.append(handler)
            if any(x is not y for x, y in zip(new_handlers, handlers)):
                replaced["handlers"] = new_handlers
        if not replaced:
            return [statement]
        shallow = copy.copy(statement)
        for field, value in replaced.items():
            setattr(shallow, field, value)
        return [shallow]

    body = prune_block(test.body)
    if not pruned:
        return None
    shallow_test = copy.copy(test)
    shallow_test.body = body or [ast.Pass()]
    return shallow_test


def _case_coordinates_required(
    index: _ModuleIndex,
    test: ast.FunctionDef | ast.AsyncFunctionDef,
    class_nodes: list[ast.ClassDef],
    fixture_reachable: list[ast.AST],
    selector: dict[str, Any],
    *,
    fixture_names: frozenset[str] | None = None,
    _forward_depth: int = 0,
) -> bool | None:
    """Whether one parametrize case reaches an electron/evaluation coordinate.

    The case graph is the node graph with ``test`` replaced by its
    branch-pruned copy. ``None``: no branch of ``test`` is decided by this
    case, so the node obligation stands unchanged.

    By default every fixture graph is part of the case graph (a fixture runs
    for every case). With ``fixture_names`` the BRANCH-OWN graph is used
    instead: fixture graphs are left out, and a fixture argument referenced in
    the body is not followed to a same-named module function. Only a verified
    per-case declaration uses that narrower graph.

    A test whose whole body forwards its arguments to an imported test function
    (a forwarding entry) is judged on that function's branch, with
    the forwarded fixture values' graphs included (or, branch-own, excluded).
    """
    pruned = _branch_pruned_test(test, selector)
    if pruned is None:
        forward = None if _forward_depth >= 2 or class_nodes else _forwarded_case(index, test, selector)
        if forward is None:
            return None
        target_index, target, target_selector, fixture_pairs = forward
        if fixture_names is None:
            target_fixtures = list(fixture_reachable)
            target_fixtures.extend(_fixture_closure(index, [argument for argument, _ in fixture_pairs]))
            target_fixtures.extend(_fixture_closure(target_index, [parameter for _, parameter in fixture_pairs]))
            target_fixture_names = None
        else:
            target_fixtures = []
            target_fixture_names = frozenset(parameter for _, parameter in fixture_pairs)
        return _case_coordinates_required(
            target_index,
            target,
            [],
            target_fixtures,
            target_selector,
            fixture_names=target_fixture_names,
            _forward_depth=_forward_depth + 1,
        )
    if fixture_names is None:
        case_reachable = index.reachable(pruned, class_nodes)
        seen = {id(node) for node in case_reachable}
        for node in fixture_reachable:
            if id(node) not in seen:
                seen.add(id(node))
                case_reachable.append(node)
    else:
        # Expand the pruned body by hand, one level, so a fixture ARGUMENT is
        # never followed to its module-level fixture function; every other
        # referenced helper contributes its whole ordinary reachable graph.
        class_members: dict[str, ast.AST] = {}
        for class_node in class_nodes:
            for member in class_node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_members.setdefault(member.name, member)
                elif isinstance(member, (ast.Assign, ast.AnnAssign)):
                    for target in member.targets if isinstance(member, ast.Assign) else [member.target]:
                        for target_name in _target_names(target):
                            class_members.setdefault(target_name, member)
        names = {child.id for child in ast.walk(pruned) if isinstance(child, ast.Name)}
        names.update(child.attr for child in ast.walk(pruned) if isinstance(child, ast.Attribute))
        names.update(_dotted_name(child.func).rsplit(".", 1)[-1] for child in ast.walk(pruned) if isinstance(child, ast.Call))
        case_reachable = [pruned]
        seen = {id(pruned)}
        for name in sorted(names - fixture_names):
            target = class_members.get(name) or index.functions.get(name) or index.assignments.get(name)
            if target is None:
                continue
            for node in index.reachable(target, class_nodes):
                if id(node) not in seen:
                    seen.add(id(node))
                    case_reachable.append(node)
    dependencies = index.assertion_dependencies(case_reachable, include_callbacks=False)
    return bool(_coordinate_symbols(dependencies)) or _is_production_driver_route(_calls(dependencies), dependencies)


def _forwarded_case(
    index: _ModuleIndex, test: ast.FunctionDef | ast.AsyncFunctionDef, selector: dict[str, Any]
) -> tuple[_ModuleIndex, ast.FunctionDef | ast.AsyncFunctionDef, dict[str, Any], list[tuple[str, str]]] | None:
    """Resolve ``def test_x(a, b): imported_check(a, b)`` to the forwarded function.

    Only a body that is exactly one call of an imported module-level test
    function with the test's own arguments passed positionally qualifies.
    Returns the target index and function, the selector renamed to the
    target's parameters, and (argument, parameter) pairs for the other
    (fixture) arguments; ``None`` for anything else.
    """
    body = list(test.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Expr) or not isinstance(body[0].value, ast.Call):
        return None
    call = body[0].value
    arguments = {argument.arg for argument in (*test.args.posonlyargs, *test.args.args, *test.args.kwonlyargs)}
    if (
        not isinstance(call.func, ast.Name)
        or call.func.id not in index.imported_test_functions
        or call.keywords
        or not all(isinstance(argument, ast.Name) and argument.id in arguments for argument in call.args)
    ):
        return None
    path, symbol = index.imported_test_functions[call.func.id]
    target_index = _module_index(path)
    target = target_index.functions.get(symbol)
    if target is None:
        return None
    parameters = [argument.arg for argument in (*target.args.posonlyargs, *target.args.args)]
    if len(parameters) != len(call.args) or target.args.vararg is not None or target.args.kwonlyargs:
        return None
    target_selector: dict[str, Any] = {}
    fixture_pairs: list[tuple[str, str]] = []
    for argument, parameter in zip(call.args, parameters):
        if argument.id in selector:
            target_selector[parameter] = selector[argument.id]
        else:
            fixture_pairs.append((argument.id, parameter))
    return (target_index, target, target_selector, fixture_pairs) if target_selector else None


def _fixture_closure(index: _ModuleIndex, names: list[str]) -> list[ast.AST]:
    """Graphs of the named fixture functions (local or imported test functions) and their fixtures."""
    result: list[ast.AST] = []
    seen_nodes: set[int] = set()
    queue = [(index, name) for name in names]
    seen_names: set[tuple[Path, str]] = set()
    while queue:
        owner, name = queue.pop()
        if (owner.path, name) in seen_names:
            continue
        seen_names.add((owner.path, name))
        function, home = owner.functions.get(name), owner
        if function is None and name in owner.imported_test_functions:
            path, symbol = owner.imported_test_functions[name]
            home = _module_index(path)
            function = home.functions.get(symbol)
        if function is None:
            continue
        for node in home.reachable(function, []):
            if id(node) not in seen_nodes:
                seen_nodes.add(id(node))
                result.append(node)
        queue.extend(
            (home, argument.arg)
            for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
            if argument.arg not in {"self", "cls", "request"}
        )
    return result


def _case_electron_declarations(case: dict[str, Any]) -> list[dict[str, str]]:
    """``pytest.param(..., marks=pytest.mark.gdd_electron_configuration(...))`` on one case."""
    declarations: list[dict[str, str]] = []
    for mark in case.get("markers", ()):
        if "gdd_electron_configuration" not in str(mark):
            continue
        try:
            expression = ast.parse(str(mark), mode="eval").body
        except SyntaxError as exc:
            raise ValueError(f"unreadable per-case gdd_electron_configuration mark {mark!r}") from exc
        declarations.extend(_marker_contracts([expression], "gdd_electron_configuration"))
    return declarations


# Per-case finiteness-only evidence. ``smoke`` is admitted only under
# the smoke verdict method; ``stability`` (finite and bounded at a reviewed
# ill-conditioned point, never accuracy evidence) only on a statically decided
# branch that performs no numerical comparison.
_FINITENESS_EVIDENCE_MARK = "gdd_finiteness_evidence"
_FINITENESS_EVIDENCE_MODES = {"smoke": "VM-NUM-SMOKE-001", "stability": None}
_NUMERICAL_OBSERVATION_KINDS = frozenset({"tolerance-comparison", "exact-comparison", "numerical-bound"})


def _case_finiteness_declarations(case: dict[str, Any]) -> list[dict[str, str]]:
    """``pytest.param(..., marks=pytest.mark.gdd_finiteness_evidence(mode=..., reason=...))`` on one case."""
    declarations: list[dict[str, str]] = []
    for mark in case.get("markers", ()):
        if _FINITENESS_EVIDENCE_MARK not in str(mark):
            continue
        try:
            expression = ast.parse(str(mark), mode="eval").body
        except SyntaxError as exc:
            raise ValueError(f"unreadable per-case {_FINITENESS_EVIDENCE_MARK} mark {mark!r}") from exc
        declarations.extend(_marker_contracts([expression], _FINITENESS_EVIDENCE_MARK))
    return declarations


def _case_finiteness_evidence(
    pytest_node: str,
    test: ast.FunctionDef | ast.AsyncFunctionDef,
    test_path: Path,
    verdict_method: str,
    observations: list[dict[str, Any]],
    concrete_cases: list[dict[str, Any]],
    direct_parameter_names: set[str],
) -> None:
    """Verify and record per-case finiteness-only declarations.

    A declaration is accepted only when (i) it is a single per-case mark with a
    known mode and a non-blank reason, (ii) ``smoke`` sits on the smoke verdict
    method and ``stability`` does not, (iii) the case asserts, with
    unconditional test-body finiteness assertions, every output the node's
    numerical comparisons would judge (every non-underscore target of the
    test-body assignments that bind a comparison operand), (iv) for
    ``stability`` the case's statically decided branch reaches none of the
    node's numerical comparisons, all of which lie in the test body, and
    (v) at least one case of the node keeps its numerical obligation.
    Anything else raises ``ValueError``.
    """
    relative = str(test_path.relative_to(ROOT))
    first, last = test.lineno, getattr(test, "end_lineno", test.lineno)

    def in_test(observation: dict[str, Any]) -> bool:
        location = observation.get("source_location", {})
        return location.get("path") == relative and first <= int(location.get("line", -1)) <= last

    # The comparisons the completion gate requires (its ``needed_kinds`` rule:
    # tolerance comparisons when the node declares any), and the helper calls
    # that may carry them; bare preconditions are not the skipped verdict.
    kinds = {observation.get("kind") for observation in observations}
    needed_kinds = {"tolerance-comparison"} if "tolerance-comparison" in kinds else kinds & _NUMERICAL_OBSERVATION_KINDS
    numerical = [
        observation
        for observation in observations
        if observation.get("kind") in needed_kinds or observation.get("kind") == "comparison-helper-call"
    ]
    operand_names: set[str] = set()
    for observation in numerical:
        if not in_test(observation):
            continue
        try:
            parsed = ast.parse(str(observation.get("expression", "")), mode="eval")
        except SyntaxError:
            continue
        operand_names.update(child.id for child in ast.walk(parsed) if isinstance(child, ast.Name))
    outputs: list[str] = []
    for statement in test.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        names = [
            child.id
            for target in targets
            for child in ast.walk(target)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
        ]
        if operand_names.intersection(names):
            outputs.extend(name for name in names if not name.startswith("_") and name not in outputs)
    finiteness_by_line = {
        int(observation["source_location"]["line"]): observation["id"]
        for observation in observations
        if observation.get("kind") == "finiteness" and in_test(observation)
    }
    covered: dict[str, set[str]] = {}
    for statement in test.body:
        if isinstance(statement, ast.Assert) and statement.lineno in finiteness_by_line:
            names = {child.id for child in ast.walk(statement.test) if isinstance(child, ast.Name)}
            for name in names:
                covered.setdefault(name, set()).add(finiteness_by_line[statement.lineno])

    declared = 0
    for case in concrete_cases:
        declarations = _case_finiteness_declarations(case)
        if not declarations:
            continue
        declared += 1
        where = f"{pytest_node}[{case.get('parameter_id', case['case'])}]"
        if len(declarations) > 1:
            raise ValueError(f"{where}: more than one per-case {_FINITENESS_EVIDENCE_MARK} mark is declared")
        declaration = declarations[0]
        mode, reason = declaration.get("mode"), declaration.get("reason")
        if mode not in _FINITENESS_EVIDENCE_MODES:
            raise ValueError(f"{where}: invalid {_FINITENESS_EVIDENCE_MARK} mode {mode!r}")
        if not reason or not reason.strip():
            raise ValueError(f"{where}: a finiteness-only declaration needs a reason")
        if mode == "smoke" and verdict_method != _FINITENESS_EVIDENCE_MODES["smoke"]:
            raise ValueError(f"{where}: finiteness-only smoke is admitted only under VM-NUM-SMOKE-001, not {verdict_method}")
        if mode == "stability" and verdict_method == _FINITENESS_EVIDENCE_MODES["smoke"]:
            raise ValueError(f"{where}: a smoke-method node declares finiteness-only smoke, not stability")
        if not outputs:
            raise ValueError(f"{where}: no test-body output of the node's numerical comparisons is identifiable")
        uncovered = [name for name in outputs if not covered.get(name)]
        if uncovered:
            raise ValueError(f"{where}: the case does not assert finiteness of every declared output {uncovered}")
        if mode == "stability":
            if any(not in_test(observation) for observation in numerical):
                raise ValueError(f"{where}: a numerical comparison outside the test body cannot be verified unselected")
            selector = {name: value for name, value in case.get("selector", {}).items() if name in direct_parameter_names}
            pruned = _branch_pruned_test(test, selector, bool_names=True)
            kept = {child.lineno for child in ast.walk(pruned or test) if hasattr(child, "lineno")}
            reached = sorted(
                observation["id"]
                for observation in numerical
                if int(observation["source_location"]["line"]) in kept
                and (pruned is not None or "conditional_execution" not in observation)
            )
            if reached:
                raise ValueError(f"{where}: stability evidence, but the case's branch reaches numerical comparisons {reached}")
            if pruned is None:
                raise ValueError(f"{where}: stability evidence needs a branch statically decided by the case")
        case["finiteness_evidence"] = {
            "mode": mode,
            "reason": reason,
            "outputs": outputs,
            "finiteness_observations": sorted(set().union(*(covered[name] for name in outputs))),
            "declared_by": f"pytest.param marks=pytest.mark.{_FINITENESS_EVIDENCE_MARK}",
            "verified_by": "gdd.engine.case_binding_inventory._case_finiteness_evidence",
            "decision": "per-case-finiteness-evidence",
        }
    if declared and declared == len(concrete_cases):
        raise ValueError(f"{pytest_node}: every case is finiteness-only; the node keeps no numerical claim")


def _seed_literals(nodes: list[ast.AST], *, source_index: ToleranceSourceIndex | None = None) -> list[Any]:
    seeds: list[Any] = []
    for node in nodes:
        for child in _walk(node):
            if isinstance(child, ast.keyword) and child.arg and "seed" in child.arg.lower():
                try:
                    value = ast.literal_eval(child.value)
                except (ValueError, TypeError, SyntaxError):
                    value = _source_expression(child.value)
                if value not in seeds:
                    seeds.append(value)
            elif isinstance(child, ast.Call) and child.args:
                call_name = _dotted_name(child.func).lower()
                # A method named .seed is not necessarily NumPy's global RNG.
                # Reuse lexical import/alias resolution and reject shadowed or
                # ambiguous bindings. This describes source, not executed state.
                numpy_global_seed = source_index is not None and source_index.expression(child.func) == {
                    "kind": "imported-reference",
                    "name": "numpy.random.seed",
                }
                if call_name.endswith(("default_rng", "randomstate")) or numpy_global_seed:
                    try:
                        value = ast.literal_eval(child.args[0])
                    except (ValueError, TypeError, SyntaxError):
                        value = _source_expression(child.args[0])
                    if value not in seeds:
                        seeds.append(value)
    return sorted(seeds, key=lambda value: repr(value))


def _evaluation_conditioning_contract(
    tree: ast.AST,
    nodes: list[ast.AST],
    assertions: list[str],
    parameter_names: set[str],
    pytest_node: str,
) -> dict[str, Any] | None:
    """Derive a declared evaluation-kappa screen or return ``None``.

    Only an assertion over a name containing ``evaluation_kappa`` declares the
    metric, and its cap must be a local name assigned from the central
    ``jqmc._precision.get_evaluation_kappa_max(zone, level, degree)`` call. A
    metric assertion without such a cap is a derivation error (fail closed),
    never a silently unconstrained case. The cond(G) contract is not consulted
    or changed here.
    """
    if not any("evaluation_kappa" in assertion for assertion in assertions):
        return None
    from gdd.engine import evaluation_conditioning as kappa

    metric_assertions = []
    for assertion in sorted(set(assertions)):
        if not re.search(r"evaluation_kappa", assertion):
            continue
        try:
            metric, _comparison, cap_symbol = kappa._comparison_parts(assertion)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"{pytest_node}: unsupported evaluation-kappa assertion {assertion!r}: {exc}") from exc
        if not kappa.is_evaluation_kappa_name(metric):
            raise ValueError(f"{pytest_node}: evaluation-kappa assertion {assertion!r} must constrain the metric")
        metric_assertions.append((assertion, metric, cap_symbol))
    if not metric_assertions:
        return None
    imported = any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == kappa.CAP_MODULE
        and any(alias.name == kappa.CAP_FUNCTION and alias.asname is None for alias in statement.names)
        for statement in _walk(tree)
    )
    caps: dict[str, list[ast.Call]] = {}
    for node in nodes:
        for child in _walk(node):
            if not isinstance(child, ast.Assign) or len(child.targets) != 1 or not isinstance(child.targets[0], ast.Name):
                continue
            if isinstance(child.value, ast.Call) and _dotted_name(child.value.func) == kappa.CAP_FUNCTION:
                caps.setdefault(child.targets[0].id, []).append(child.value)
    cap_symbols = {cap_symbol for _assertion, _metric, cap_symbol in metric_assertions}
    if len(cap_symbols) != 1:
        raise ValueError(f"{pytest_node}: evaluation-kappa assertions must share one central cap")
    cap_symbol = cap_symbols.pop()
    calls = caps.get(cap_symbol, [])
    if not imported or len(calls) != 1:
        raise ValueError(
            f"{pytest_node}: evaluation-kappa cap {cap_symbol!r} must be assigned once from the imported central "
            f"{kappa.CAP_RESOLVER}"
        )
    call = calls[0]
    if call.keywords or len(call.args) != 3:
        raise ValueError(f"{pytest_node}: the central evaluation-kappa cap takes exactly (zone, level, degree)")
    zone, level, degree = call.args
    if not (isinstance(zone, ast.Constant) and isinstance(zone.value, str)):
        raise ValueError(f"{pytest_node}: evaluation-kappa cap zone must be a literal zone name")
    if not (isinstance(level, ast.Constant) and isinstance(level.value, str)):
        raise ValueError(f"{pytest_node}: evaluation-kappa cap level must be a literal level")
    if isinstance(degree, ast.Name) and degree.id in parameter_names:
        degree_spec: dict[str, Any] = {"selector": degree.id}
    elif isinstance(degree, ast.Constant) and type(degree.value) is int and degree.value >= 0:
        degree_spec = {"literal": degree.value}
    else:
        raise ValueError(f"{pytest_node}: evaluation-kappa cap degree must be a test parameter or an integer literal")
    contract = {
        "subject": kappa.SUBJECT,
        "definition": kappa.DEFINITION,
        "operator": kappa.OPERATOR,
        "metric_symbols": sorted({metric for _assertion, metric, _cap in metric_assertions}),
        "cap": {
            "symbol": cap_symbol,
            "resolver": kappa.CAP_RESOLVER,
            "zone": zone.value,
            "level": level.value,
            "degree": degree_spec,
        },
        "constraints": [assertion for assertion, _metric, _cap in metric_assertions],
        "realized_observation": dict(kappa.REALIZED_OBSERVATION),
    }
    errors = kappa.contract_errors(contract)
    if errors:
        raise ValueError(f"{pytest_node}: {'; '.join(errors)}")
    return contract


def _geminal_conditioning_assertions(nodes: list[ast.AST], assertions: list[str]) -> list[str]:
    """Select only constraints whose condition number belongs to geminal ``G``.

    Generic basis/projector conditions (for example the pGTO+PW Schur
    complement) are useful tests, but they are not the electron-configuration
    ``cond(G)`` contract.  Follow the small, explicit data flow used by the
    tests instead of relabelling every variable containing ``cond``.
    """
    condition_symbols: set[str] = set()

    def explicit_geminal_metric(name: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return any(
            token in normalized for token in ("cond_g", "condition_g", "sigma_min_g", "sigma_max_g", "cond_times_epsilon")
        )

    for node in nodes:
        for child in _walk(node):
            if isinstance(child, (ast.Name, ast.Attribute)):
                name = _dotted_name(child)
                if explicit_geminal_metric(name):
                    condition_symbols.add(name.rsplit(".", 1)[-1])
            if not isinstance(child, (ast.Assign, ast.AnnAssign)) or child.value is None:
                continue
            value_source = _source_expression(child.value)
            call_names = {_dotted_name(call.func) for call in ast.walk(child.value) if isinstance(call, ast.Call)}
            geminal_condition = explicit_geminal_metric(value_source) or any(
                name.endswith("linalg.cond")
                and any(token in value_source for token in ("geminal", "compute_geminal", "compute_G", "(G)", " G"))
                for name in call_names
            )
            if not geminal_condition:
                continue
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            condition_symbols.update(name for target in targets for name in _target_names(target))

    # Propagate a known per-draw cond(G) into simple collection accumulators
    # such as ``observed_cond.append(cond_g)``.
    for node in nodes:
        for child in _walk(node):
            if not isinstance(child, ast.Call) or not child.args:
                continue
            call_name = _dotted_name(child.func)
            if not call_name.endswith(".append"):
                continue
            argument_names = {_dotted_name(item) for item in ast.walk(child.args[0]) if isinstance(item, ast.Name)}
            if argument_names.intersection(condition_symbols):
                condition_symbols.add(call_name.rsplit(".append", 1)[0].rsplit(".", 1)[-1])

    selected: list[str] = []
    for assertion in assertions:
        lowered = assertion.lower()
        names = set(re.findall(r"\b[A-Za-z_]\w*\b", assertion))
        if (
            names.intersection(condition_symbols)
            or any(token in lowered for token in ("cond_g", "condition_g", "sigma_min_g", "sigma_max_g", "cond_times_epsilon"))
            or ("linalg.cond" in lowered and "geminal" in lowered)
        ):
            selected.append(assertion)
    return selected


def _test_summary(test: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    """Copy the test docstring, with a declared name-derived fallback."""
    docstring = ast.get_docstring(test, clean=True)
    if docstring:
        return {"statement": docstring, "source": "test-docstring"}

    words = [_SUMMARY_ACRONYMS.get(word, word) for word in test.name.removeprefix("test_").split("_")]
    statement = " ".join(words)
    if statement:
        statement = statement[0].upper() + statement[1:] + "."
    return {"statement": statement, "source": "pytest-function-name-fallback"}


def _claim_id(test_path: Path, parts: list[str]) -> str:
    """Stable, readable claim slug owned by one concrete pytest node."""
    tokens = [test_path.stem.removeprefix("test_"), *parts[:-1], parts[-1].removeprefix("test_")]
    slug = "-".join(re.sub(r"[^a-z0-9]+", "-", token.lower()).strip("-") for token in tokens)
    return slug.strip("-")


def _assertion_kind(expression: str, call_name: str = "", context: str = "") -> tuple[str, str]:
    """Classify the decision, never diagnostic strings or coincidental names.

    A conjunction proves each conjunct. A disjunction/conditional does not
    prove that its numerical comparison ran; its leaves must be tracked
    separately. Unknown predicates remain visible but are not numerical oracles.
    """
    try:
        root = ast.parse(expression, mode="eval").body
        if isinstance(root, ast.Call) and call_name and call_name != _dotted_name(root.func):
            # Alias resolution must not erase a bound receiver such as
            # ``np.isfinite(values).all()``: its dotted leaf is just "all",
            # but that name alone no longer contains the checked predicate.
            root.func = ast.parse(call_name, mode="eval").body
    except SyntaxError:
        return "unclassified-predicate", "non-vacuity"

    def structural(node: ast.AST) -> bool:
        # np.eye(actual.shape[0]) is a numerical reference, not a shape.
        if isinstance(node, ast.Attribute):
            return node.attr in {"shape", "dtype", "ndim", "size"}
        if isinstance(node, ast.Subscript):
            return structural(node.value)
        if isinstance(node, ast.UnaryOp):
            return structural(node.operand)
        if isinstance(node, (ast.BinOp, ast.Tuple, ast.List)):
            operands = [node.left, node.right] if isinstance(node, ast.BinOp) else node.elts
            # Arithmetic/packing of dimensions is still metadata. Do not
            # taint real numerical values merely normalized by a dimension.
            return any(structural(value) for value in operands) and all(
                structural(value) or isinstance(value, ast.Constant) for value in operands
            )
        if isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            leaf = name.rsplit(".", 1)[-1]
            if name in {"len", "type", "isinstance"}:
                return True
            if leaf in {"shape", "size", "ndim", "dtype"}:
                return True
            # A value-preserving container/cast cannot turn shape metadata
            # into a numerical oracle. Constructors such as eye(shape[0])
            # deliberately remain values, not metadata.
            reducers = {"sum", "prod", "min", "max", "amin", "amax"}
            if leaf in {"asarray", "array", "asanyarray", "tuple", "list", "int", "float"} | reducers:
                if node.args and structural(node.args[0]):
                    return True
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"tolist", "astype", "copy"} | reducers:
                return structural(node.func.value)
        return False

    def classify(node: ast.AST) -> tuple[str, str]:
        if isinstance(node, ast.IfExp):
            return "compound-disjunction", "non-vacuity"
        if isinstance(node, ast.BoolOp):
            kinds = [classify(value) for value in node.values]
            if isinstance(node.op, ast.Or):
                numerical = {"tolerance-comparison", "exact-comparison", "numerical-bound"}
                if all(kind in numerical and role == "verdict" for kind, role in kinds):
                    return "numerical-bound", "verdict"
                return "compound-disjunction", "non-vacuity"
            for kind in ("tolerance-comparison", "exact-comparison", "numerical-bound"):
                if (kind, "verdict") in kinds:
                    return kind, "verdict"
            return kinds[0]
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                kind, role = classify(node.operand)
                if kind in {"tolerance-comparison", "exact-comparison", "numerical-bound"}:
                    # Non-vacuity controls often require two values to differ.
                    # That is useful, but cannot certify numerical agreement.
                    return "numerical-distinction", "non-vacuity"
                return kind, role
            return classify(node.operand)
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if any(isinstance(op, ast.NotEq) for op in node.ops):
                # Distinguishing a corrupted output is useful negative
                # evidence, never proof that an output equals its oracle.
                return "numerical-distinction", "non-vacuity"
            if any(isinstance(value, ast.Constant) and value.value is False for value in operands) and any(
                classify(value)[0] == "tolerance-comparison" for value in operands
            ):
                return "numerical-distinction", "non-vacuity"
            if any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
                if any(isinstance(value, ast.Constant) and value.value is None for value in operands):
                    return "existence", "non-vacuity"
                return "structural-identity", "verdict"
            if any(isinstance(value, ast.Call) and _dotted_name(value.func) == "len" for value in operands):
                return "cardinality-comparison", "verdict"
            if any(structural(value) for value in operands):
                return "shape-or-dtype", "non-vacuity"
            if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops) or any(
                isinstance(value, ast.Constant) and isinstance(value.value, (str, bool)) for value in operands
            ):
                return "structural-identity", "verdict"
            if any(classify(value)[0] == "tolerance-comparison" for value in operands):
                return "tolerance-comparison", "verdict"
            if all(isinstance(op, ast.Eq) for op in node.ops):
                return "exact-comparison", "verdict"
            return "numerical-bound", "verdict"
        if isinstance(node, ast.Call):
            leaf = _dotted_name(node.func).rsplit(".", 1)[-1].lower()
            if leaf in {"raises", "warns"}:
                return "expected-error", "verdict"
            if leaf in {"hasattr", "isinstance", "issubclass"}:
                return "existence", "non-vacuity"
            if leaf in {
                "allclose",
                "isclose",
                "approx",
                "assert_allclose",
                "assert_almost_equal",
                "assert_array_almost_equal",
                "assert_array_max_ulp",
            }:
                if any(structural(value) for value in node.args[:2]):
                    return "shape-or-dtype", "non-vacuity"
                return "tolerance-comparison", "verdict"
            if leaf in {"array_equal", "assert_equal", "assert_array_equal"}:
                if any(structural(value) for value in node.args[:2]):
                    return "shape-or-dtype", "non-vacuity"
                return "exact-comparison", "verdict"
            if leaf in {"isfinite", "isnan", "isinf"} or leaf.startswith("_assert_finite"):
                return "finiteness", "non-vacuity" if "precondition" in context.lower() else "safety"
            if leaf in {"all", "any", "bool"}:
                value = node.args[0] if node.args else node.func.value if isinstance(node.func, ast.Attribute) else None
                if value is not None:
                    if leaf == "any" and classify(value)[0] in {"tolerance-comparison", "exact-comparison", "numerical-bound"}:
                        # A single matching array entry does not establish
                        # agreement of the declared output with its reference.
                        return "partial-comparison", "non-vacuity"
                    return classify(value)
            if leaf.startswith(("assert", "_assert")):
                return "assertion-helper", "non-vacuity"
        return "unclassified-predicate", "non-vacuity"

    return classify(root)


def _tolerance_arguments(call: ast.Call, call_name: str) -> dict[str, str]:
    explicit = {
        keyword.arg: _source_expression(keyword.value)
        for keyword in call.keywords
        if keyword.arg in {"atol", "rtol", "rel", "abs", "decimal", "maxulp"}
    }
    lowered = call_name.lower()
    if lowered.endswith("assert_array_max_ulp"):
        if "maxulp" in explicit:
            return {"maxulp": explicit["maxulp"]}
        return {"maxulp": _source_expression(call.args[2])} if len(call.args) > 2 else {}
    if explicit:
        return explicit
    forwarded = [keyword for keyword in call.keywords if keyword.arg is None]
    if forwarded:
        return {"forwarded_keywords": _source_expression(forwarded[0].value)}
    if lowered.endswith(("assert_almost_equal", "assert_array_almost_equal")):
        if len(call.args) > 2:
            return {"decimal": _source_expression(call.args[2])}
        return {"resolver": "numpy-almost-equal-default", "decimal": "6" if "array_almost" in lowered else "7"}
    if lowered.endswith("testing.assert_allclose"):
        return {"resolver": "numpy.testing.assert_allclose-default", "rtol": "1e-7", "atol": "0"}
    if lowered.endswith(("allclose", "isclose")):
        return {"resolver": "numpy-allclose-default", "rtol": "1e-5", "atol": "1e-8"}
    if lowered.endswith("approx"):
        return {"resolver": "pytest.approx-default", "rel": "1e-6", "abs": "1e-12"}
    return {}


def _is_assertion_call(call_name: str) -> bool:
    """Whether a call is itself a verdict operation, not an error value.

    Matched as a substring, ``assert`` would make ``AssertionError(...)`` on an
    unreachable failure branch a mandatory verdict site.  An exception constructor describes failure, whereas ``assert_allclose`` or
    ``pytest.raises`` performs the test's decision.
    """
    leaf = call_name.rsplit(".", 1)[-1].lower()
    if leaf.endswith(("error", "exception")):
        return False
    return leaf.lstrip("_").startswith("assert") or leaf in {"raises", "warns"}


def _expand_comparison_alias(name: str, bindings: dict[str, str]) -> str:
    """Resolve simple aliases once per binding; object self-rebinding is not a cycle to expand."""
    visited = set()
    while True:
        head, separator, tail = name.partition(".")
        if head in visited or head not in bindings:
            return name
        visited.add(head)
        name = bindings[head] + (separator + tail if separator else "")


@lru_cache(maxsize=256)
def _comparison_lexical_index(tree: ast.AST, source: Path):
    """Index names by lexical scope without borrowing sibling-local imports."""
    parents = {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    definitions = {}
    imported_names = {}
    for statement in ast.walk(tree):
        context = statement
        while (context := parents.get(id(context))) is not None:
            if isinstance(context, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                break
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.setdefault((id(context), statement.name), []).append(statement)
        bindings = imported_names.setdefault(id(context), {})
        if isinstance(statement, ast.ImportFrom):
            package = list(source.with_suffix("").parts[:-1])
            prefix = package[: len(package) - statement.level + 1] if statement.level else []
            module = ".".join([*prefix, *(statement.module or "").split(".")]).strip(".")
            for alias in statement.names:
                bindings[alias.asname or alias.name] = f"{module}.{alias.name}"
        elif isinstance(statement, ast.Import):
            for alias in statement.names:
                bindings[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)) and isinstance(statement.value, (ast.Name, ast.Attribute)):
            for target in statement.targets if isinstance(statement, ast.Assign) else [statement.target]:
                if isinstance(target, ast.Name):
                    bindings.setdefault(target.id, _dotted_name(statement.value))
    return parents, definitions, imported_names


def _tree_positions(tree: ast.AST):
    """Read-only ``(type, line, column) -> node`` index of one module tree."""
    table = _memo_table("tree_positions") if _module_node(tree) else None
    if table is not None and tree in table:
        return table[tree]
    positions = MappingProxyType(
        {(type(node), getattr(node, "lineno", None), getattr(node, "col_offset", None)): node for node in ast.walk(tree)}
    )
    if table is not None:
        table[tree] = positions
    return positions


def _comparison_function_graph(nodes: list[ast.AST], path: Path):
    """Resolve local/imported comparison procedures without importing their code."""
    functions = {}
    parents = {}
    trees = {}
    source_positions = {}
    definitions = {}
    imported_names = {}
    for owner in nodes:
        source = getattr(owner, "_gdd_source_path", path)
        if source not in trees:
            trees[source] = (
                _module_index(ROOT / source).tree
                if (ROOT / source).is_file()
                else ast.Module(
                    body=[node for node in nodes if getattr(node, "_gdd_source_path", path) == source], type_ignores=[]
                )
            )
            source_parents, source_definitions, source_imports = _comparison_lexical_index(trees[source], source)
            parents.update(source_parents)
            definitions.update(source_definitions)
            imported_names.update(source_imports)
            source_positions[source] = _tree_positions(trees[source])
        # Scratch inventories may have parsed the same file independently of
        # the cached module index. Preserve lexical ancestry by source position
        # while retaining the actual nodes used by assertion extraction.
        matching = source_positions[source].get(
            (type(owner), getattr(owner, "lineno", None), getattr(owner, "col_offset", None))
        )
        if matching is not None and id(matching) in parents:
            parents[id(owner)] = parents[id(matching)]
        parents.update({id(child): parent for parent in _walk(owner) for child in ast.iter_child_nodes(parent)})
        for function in _walk(owner):
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[function_scope(function, source.as_posix())] = (source, function)

    def enclosing(node, kinds):
        while (node := parents.get(id(node))) is not None:
            if isinstance(node, kinds):
                return node
        return None

    module_prefixes: dict[Path, str] = {}

    def resolve(caller, call):
        source, function = functions[caller]
        name = _dotted_name(call.func)
        if name.startswith(("self.", "cls.")) and name.count(".") == 1:
            owner = enclosing(function, ast.ClassDef)
            matches = [
                child
                for child in getattr(owner, "body", ())
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == name.split(".")[1]
            ]
            return function_scope(matches[0], source.as_posix()) if len(matches) == 1 else None
        if "." not in name:
            context = function
            while context is not None:
                matches = definitions.get((id(context), name), ())
                if len(matches) == 1:
                    return function_scope(matches[0], source.as_posix())
                context = enclosing(context, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module))

        # Imports in enclosing functions are lexical; imports in a sibling
        # function must never decide the target of this call.
        contexts = [function]
        context = function
        while (context := enclosing(context, (ast.FunctionDef, ast.AsyncFunctionDef))) is not None:
            contexts.append(context)
        contexts.append(trees[source])
        bindings = {}
        for context in reversed(contexts):
            context = source_positions[source].get(
                (type(context), getattr(context, "lineno", None), getattr(context, "col_offset", None)), context
            )
            bindings.update(imported_names.get(id(context), {}))
        name = _expand_comparison_alias(name, bindings)
        for identity, (candidate_source, candidate) in functions.items():
            if candidate_source not in module_prefixes:
                module_prefixes[candidate_source] = ".".join(candidate_source.with_suffix("").parts)
            qualified = module_prefixes[candidate_source] + "." + candidate.name
            if name == qualified or (
                candidate_source == source
                and name == candidate.name
                and enclosing(candidate, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) is None
            ):
                return identity
        return None

    return functions, parents, enclosing, resolve


@lru_cache(maxsize=256)
def _tolerance_source_index(tree: ast.Module, path: Path) -> ToleranceSourceIndex:
    return ToleranceSourceIndex(tree, path.as_posix())


def _assertion_contract(nodes: list[ast.AST], path: Path, test: ast.AST | None = None) -> list[dict[str, Any]]:
    """Typed, source-addressed contractual observations reachable from a test."""
    observations: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    predicate_helpers = {
        _dotted_name(call.func)
        for parent in nodes
        for assertion in _walk(parent)
        if isinstance(assertion, ast.Assert)
        for call in ast.walk(assertion.test)
        if isinstance(call, ast.Call)
    }
    functions, call_parents, enclosing, resolve_helper = _comparison_function_graph(nodes, path)
    predicate_functions = set()
    for caller, (_, function) in functions.items():
        for child in _walk(function):
            if isinstance(child, ast.Call) and enclosing(child, ast.Assert) is not None:
                predicate_functions.add(resolve_helper(caller, child))
    unconditional: set[int] = set()
    invocation_required: dict[int, set[int]] = {}
    tolerance_indexes = {}

    def source_index(source: Path) -> ToleranceSourceIndex:
        if source not in tolerance_indexes:
            tree = (
                _module_index(ROOT / source).tree
                if (ROOT / source).is_file()
                else ast.Module(
                    body=[owner for owner in nodes if getattr(owner, "_gdd_source_path", path) == source], type_ignores=[]
                )
            )
            tolerance_indexes[source] = _tolerance_source_index(tree, source)
        return tolerance_indexes[source]

    def tolerance_source(node: ast.AST, source: Path) -> dict[str, Any]:
        return source_index(source).observation(node)

    def visit_required(node: ast.AST, destination: set[int], *, helper_call: bool = False) -> None:
        if isinstance(node, ast.Try):
            if nonrecovering_try(node):
                for statement in (*node.body, *node.orelse, *node.finalbody):
                    visit_required(statement, destination, helper_call=helper_call)
            elif helper_call:
                # Every attempted comparison procedure must return normally;
                # a handler swallowing its error cannot certify that call.
                for statement in node.body:
                    visit_required(statement, destination, helper_call=True)
            return
        if (guarded := assertion_guard(node)) is not None:
            # ``if not compare(...): raise AssertionError`` evaluates its
            # comparator whenever the statement runs, like an assertion.
            destination.add(id(node))
            destination.add(id(node.test))
            visit_required(guarded, destination, helper_call=helper_call)
            return
        if isinstance(
            node, (ast.If, ast.IfExp, ast.TryStar, ast.FunctionDef, ast.AsyncFunctionDef, *UNTRACED_COMPARISON_SCOPES)
        ):
            return
        destination.add(id(node))
        for child in ast.iter_child_nodes(node):
            visit_required(child, destination, helper_call=helper_call)

    if isinstance(test, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for statement in test.body:
            visit_required(statement, unconditional)

    def resolve_call(node: ast.AST, call_name: str) -> str:
        source_path = getattr(node, "_gdd_source_path", path)
        aliases = _module_index(ROOT / source_path).call_aliases if (ROOT / source_path).is_file() else {}
        visited: set[str] = set()
        while call_name in aliases and call_name not in visited:
            visited.add(call_name)
            call_name = aliases[call_name]
        return call_name

    def raises_contract(with_node: ast.AST, *, dynamic_match: bool = False) -> dict[str, Any] | None:
        """A single-item ``pytest.raises(<builtin>[, match=<str>])`` suite with one statement.

        ``dynamic_match`` admits a computed ``match`` (recorded as None): the
        raises context is completed only by pytest's own accepting exit, which
        applied the actual pattern to the actual exception; the closure then
        cannot re-check the message and keeps every reachable failure site.
        """
        import builtins

        if not isinstance(with_node, ast.With) or len(with_node.items) != 1 or len(with_node.body) != 1:
            return None
        context = with_node.items[0].context_expr
        if not isinstance(context, ast.Call) or resolve_call(context, _dotted_name(context.func)) != "pytest.raises":
            return None
        exception = context.args[0] if len(context.args) == 1 else None
        exception_class = getattr(builtins, exception.id, None) if isinstance(exception, ast.Name) else None
        keywords = {keyword.arg: keyword.value for keyword in context.keywords}
        match = keywords.get("match")
        if (
            not isinstance(exception_class, type)
            or not issubclass(exception_class, BaseException)
            or not set(keywords) <= {"match"}
            or match is not None
            and not (isinstance(match, ast.Constant) and isinstance(match.value, str))
            and not dynamic_match
        ):
            return None
        return {
            "context": context,
            "exception": exception.id,
            "class": exception_class,
            "match": match.value if isinstance(match, ast.Constant) else None,
            "end_lineno": with_node.end_lineno,
        }

    rejections: dict[str, dict[str, Any]] = {}
    observation_nodes: dict[str, ast.AST] = {}
    guard_messages: dict[str, ast.AST] = {}

    def rejection_context(node: ast.AST) -> dict[str, Any] | None:
        """The pytest.raises(AssertionError) suite whose only statement is this comparison."""
        statement = node if isinstance(node, ast.Assert) else parents.get(id(node))
        if isinstance(statement, ast.Expr) and statement.value is not node:
            return None
        if not isinstance(statement, (ast.Expr, ast.Assert)):
            return None
        with_node = parents.get(id(statement))
        contract = raises_contract(with_node, dynamic_match=True)
        if contract is None or with_node.body[0] is not statement or not issubclass(AssertionError, contract["class"]):
            return None
        return contract

    def add(
        node: ast.AST,
        expression: str,
        call_name: str = "",
        actual: str | None = None,
        reference: str | None = None,
        tolerance: dict[str, str] | None = None,
        context: str = "",
    ) -> None:
        source_path = getattr(node, "_gdd_source_path", path)
        key = (source_path.as_posix(), node.lineno, node.col_offset, expression)
        if key in seen:
            return
        seen.add(key)
        resolved_name = resolve_call(node, call_name)
        # Resolve imported comparator aliases inside reducers too, without
        # rewriting the source expression or its operand provenance.
        classification_tree = ast.parse(expression, mode="eval")
        for call in ast.walk(classification_tree):
            if isinstance(call, ast.Call):
                name = _dotted_name(call.func)
                resolved = resolve_call(node, name)
                if resolved and resolved != name:
                    call.func = ast.parse(resolved, mode="eval").body
        kind, role = _assertion_kind(ast.unparse(classification_tree), resolved_name, context)
        if kind == "exact-comparison":
            expression_node = ast.parse(expression, mode="eval").body
            if isinstance(expression_node, ast.Compare) and len(expression_node.ops) == 1:
                actual = _source_expression(expression_node.left)
                reference = _source_expression(expression_node.comparators[0])
        if kind == "tolerance-comparison":
            # ``actual == pytest.approx(reference)`` and compound assertions
            # contain their comparator below the assertion's root node.
            expression_node = ast.parse(expression, mode="eval").body
            for comparator in ast.walk(expression_node):
                if not isinstance(comparator, ast.Call):
                    continue
                name = resolve_call(node, _dotted_name(comparator.func))
                if name.rsplit(".", 1)[-1] not in {"approx", "allclose", "isclose", "assert_allclose"}:
                    continue
                if name.endswith("approx") and isinstance(expression_node, ast.Compare):
                    actual = _source_expression(expression_node.left)
                    reference = _source_expression(comparator.args[0]) if comparator.args else None
                elif len(comparator.args) >= 2:
                    actual, reference = (_source_expression(argument) for argument in comparator.args[:2])
                tolerance = _tolerance_arguments(comparator, name)
                break
        observation: dict[str, Any] = {
            **source_index(source_path).observation_address(node),
            "end_lineno": node.end_lineno,
            "role": role,
            "kind": kind,
            "expression": expression,
        }
        if isinstance(node, ast.Return):
            observation["truthy_return"] = True
        if resolved_name.endswith("assert_array_max_ulp") and not tolerance:
            observation["unsupported_execution"] = "declare the ULP comparison limit explicitly"
        if statement_lines[node.lineno] > 1:
            observation["unsupported_execution"] = "put each comparison statement on its own source line"
        ancestor = parents.get(id(node))
        while ancestor is not None and ancestor is not owner:
            if (
                isinstance(ancestor, (ast.If, ast.For, ast.While))
                and ancestor.lineno == node.lineno
                and assertion_guard(ancestor) is not node
            ):
                observation["unsupported_execution"] = "put the conditional comparison on its own source line"
            if isinstance(ancestor, (ast.BoolOp, ast.IfExp)):
                observation["unsupported_execution"] = "short-circuited comparator call needs an explicit assertion"
            ancestor = parents.get(id(ancestor))
        if id(node) in unconditional and kind in {"tolerance-comparison", "exact-comparison", "numerical-bound"}:
            observation["required_execution"] = True
        containing_function = parents.get(id(node))
        while containing_function is not None and not isinstance(containing_function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            containing_function = parents.get(id(containing_function))
        if containing_function is not None:
            identity = id(containing_function)
            if identity not in invocation_required:
                required_nodes: set[int] = set()
                if not any(isinstance(child, (ast.Yield, ast.YieldFrom, ast.Await)) for child in _walk(containing_function)):
                    for statement in containing_function.body:
                        visit_required(statement, required_nodes)
                invocation_required[identity] = required_nodes
            if id(node) in invocation_required[identity] and kind in {
                "tolerance-comparison",
                "exact-comparison",
                "numerical-bound",
            }:
                observation["required_in_invocation"] = function_scope(containing_function, source_path.as_posix())
        if kind in {"tolerance-comparison", "exact-comparison", "numerical-bound"}:
            if source_index(source_path).unreachable_comparison(node):
                observation["unsupported_execution"] = "statically unreachable numerical comparison in a constant if arm"
            conditional = conditional_execution(node, parents, source_path.as_posix(), test)
            if conditional is not None:
                observation["conditional_execution"] = conditional
            if not observation.get("required_in_invocation") and conditional is None:
                observation.setdefault(
                    "unsupported_execution",
                    "numerical comparison has no supported normal-call or selected-branch scope; "
                    "use an explicit synchronous comparison outside recovering handlers/lambdas",
                )
            elif (rejection := rejection_context(node)) is not None:
                # The comparator is the whole pytest.raises(AssertionError)
                # suite: it must reject. Its execution obligation moves to the
                # raises context, which only the accepting exit completes.
                context_id = source_index(source_path).observation_address(rejection["context"])["id"]
                moved = {key: observation.pop(key) for key in EXECUTION_SCOPE_KEYS if key in observation}
                observation["expected_rejection"] = context_id
                rejections[context_id] = {"scope": moved, "end_lineno": rejection["end_lineno"]}
        inequality = (
            source_index(source_path).inequality_observation(node)
            if kind == "numerical-bound" and isinstance(node, ast.Assert)
            else None
        )
        if inequality is not None:
            # A hand-written bound in a reviewed form names its two operands
            # and tolerance symbols; it stays a numerical-bound observation.
            symbols = inequality["comparisons"][0]["symbols"]
            actual, reference = symbols["actual"], symbols["reference"]
        if actual is not None:
            observation["actual"] = actual
        if reference is not None:
            observation["reference"] = reference
        if (
            kind in {"tolerance-comparison", "exact-comparison"}
            and actual
            and reference
            and actual != reference
            and source_index(source_path).comparison_aliases(node, actual, reference)
        ):
            observation["unsupported_execution"] = "numerical comparison uses aliases of the same source value"
        if tolerance:
            observation["tolerance_arguments"] = tolerance
        if kind == "tolerance-comparison":
            observation["tolerance_provenance"] = tolerance_source(node, source_path)
        elif kind == "exact-comparison":
            provenance = tolerance_source(node, source_path)
            if provenance["status"] == "source-described":
                observation["tolerance_provenance"] = provenance
        elif inequality is not None:
            observation["tolerance_provenance"] = inequality
        observation_nodes[observation["id"]] = node
        observations.append(observation)

    for owner in nodes:
        parents = {id(child): parent for parent in _walk(owner) for child in ast.iter_child_nodes(parent)}
        statement_lines = Counter(child.lineno for child in _walk(owner) if isinstance(child, ast.stmt))
        for child in _walk(owner):
            if (guarded := assertion_guard(child)) is not None:
                call_name = _dotted_name(guarded.func)
                raised = child.body[0].exc
                message = raised.args[0] if isinstance(raised, ast.Call) and raised.args else None
                before = len(observations)
                add(
                    guarded,
                    _source_expression(guarded),
                    call_name,
                    _source_expression(guarded.args[0]) if guarded.args else None,
                    _source_expression(guarded.args[1]) if len(guarded.args) > 1 else None,
                    _tolerance_arguments(guarded, call_name),
                    _source_expression(message) if message is not None else "",
                )
                if len(observations) > before:
                    # The statement completes only past its raise suite.
                    observations[-1]["end_lineno"] = child.end_lineno
                    if message is not None:
                        guard_messages[observations[-1]["id"]] = message
            elif isinstance(child, ast.Assert):
                expression = _source_expression(child.test)
                call = child.test if isinstance(child.test, ast.Call) else None
                call_name = _dotted_name(call.func) if call is not None else ""
                actual = _source_expression(call.args[0]) if call is not None and call.args else None
                reference = _source_expression(call.args[1]) if call is not None and len(call.args) > 1 else None
                tolerance = _tolerance_arguments(call, call_name) if call is not None else None
                context = _source_expression(child.msg) if child.msg is not None else ""
                add(child, expression, call_name, actual, reference, tolerance, context)
            elif isinstance(child, ast.Return) and child.value is not None:
                # Boolean comparator helpers (assert _close(...)) often return
                # a bound instead of raising. The returned decision has its
                # own source site; calling the wrapper is not enough evidence.
                returning_function = enclosing(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                returning_scope = (
                    function_scope(returning_function, getattr(child, "_gdd_source_path", path).as_posix())
                    if returning_function is not None
                    else None
                )
                if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    owner.name in predicate_helpers or returning_scope in predicate_functions
                ):
                    expression = _source_expression(child.value)
                    if _assertion_kind(expression)[0] in {"tolerance-comparison", "exact-comparison", "numerical-bound"}:
                        add(child, expression)
            elif isinstance(child, ast.Call):
                call_name = resolve_call(child, _dotted_name(child.func))
                if not _is_assertion_call(call_name):
                    continue
                ancestor = parents.get(id(child))
                conditional_assertion = False
                while ancestor is not None:
                    if isinstance(ancestor, ast.Assert):
                        conditional_assertion = True
                        break
                    ancestor = parents.get(id(ancestor))
                if conditional_assertion:
                    # The containing assertion is the decision site. Nested
                    # predicates are not independently executed assertions,
                    # especially when boolean operators short-circuit them.
                    continue
                expression = _source_expression(child)
                actual = _source_expression(child.args[0]) if child.args else None
                reference = _source_expression(child.args[1]) if len(child.args) > 1 else None
                tolerance = _tolerance_arguments(child, call_name)
                add(child, expression, call_name, actual, reference, tolerance)
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                for item in child.items:
                    context = item.context_expr
                    if isinstance(context, ast.Call) and _dotted_name(context.func).endswith("raises"):
                        add(context, _source_expression(context), _dotted_name(context.func))

    # Some contracts intentionally assert that a validator/reader/driver call
    # completes without raising.  Make that oracle explicit instead of
    # allowing an assertion-free pytest pass to masquerade as typed evidence.
    # Only direct statements in the test body are eligible; setup/build calls
    # hidden in helpers are not silently promoted to verdicts.
    if not observations and nodes:
        test = nodes[0]
        if isinstance(test, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for statement in test.body:
                call: ast.Call | None = None
                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                    call = statement.value
                elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    value = statement.value
                    if isinstance(value, ast.Call):
                        call = value
                if call is None:
                    continue
                call_name = _dotted_name(call.func)
                if any(token in call_name.lower() for token in ("clear_cache", "cache_clear")):
                    continue
                expression = _source_expression(call)
                key = (call.lineno, call.col_offset, expression)
                if key in seen:
                    continue
                seen.add(key)
                observations.append(
                    {
                        **source_index(path).observation_address(call),
                        "role": "verdict",
                        "kind": "successful-return",
                        "expression": expression,
                    }
                )
    for context_id, rejection in rejections.items():
        context_observation = next((item for item in observations if item["id"] == context_id), None)
        scope = rejection["scope"]
        function = scope.get("required_in_invocation") or scope.get("conditional_execution", {}).get("function")
        if context_observation is None or function is None:
            continue
        for key in EXECUTION_SCOPE_KEYS:
            context_observation.pop(key, None)
        context_observation.update(copy.deepcopy(scope))
        context_observation["expected_error_context"] = {"function": function, "end_lineno": rejection["end_lineno"]}
    numerical_functions = {
        row.get("required_in_invocation") or row.get("conditional_execution", {}).get("function")
        for row in observations
        if row["kind"] in {"tolerance-comparison", "exact-comparison", "numerical-bound"}
    }
    calls = []
    expected_calls = {}
    escaped_helpers = []
    predicates = set()
    for caller, (source, function) in functions.items():
        for call in _walk(function):
            if not isinstance(call, ast.Call) or enclosing(call, (ast.FunctionDef, ast.AsyncFunctionDef)) is not function:
                continue
            for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
                if isinstance(argument, (ast.Name, ast.Attribute)):
                    escaped = resolve_helper(caller, ast.Call(func=argument, args=[], keywords=[]))
                    if escaped in functions:
                        escaped_helpers.append((source, argument, escaped))
            target = resolve_helper(caller, call)
            if target not in functions:
                continue
            assertion = enclosing(call, ast.Assert)
            if assertion is not None:
                predicates.add(target)
            cursor = call
            expected_exception = False
            while (cursor := call_parents.get(id(cursor))) is not None and cursor is not function:
                if isinstance(cursor, ast.With) and any(
                    isinstance(item.context_expr, ast.Call) and _dotted_name(item.context_expr.func).endswith("raises")
                    for item in cursor.items
                ):
                    expected_exception = True
                    # A single-item pytest.raises suite whose only statement
                    # is this call has bounded exceptional-call semantics,
                    # wherever the suite sits (loops, branches, other with
                    # suites). More general recovering regions stay fail-closed.
                    contract = raises_contract(cursor, dynamic_match=True)
                    if contract is not None and isinstance(cursor.body[0], ast.Expr) and cursor.body[0].value is call:
                        expected_calls[id(call)] = {
                            "observation": source_index(source).observation_address(contract["context"])["id"],
                            "exception": contract["exception"],
                            "match": contract["match"],
                            "end_lineno": contract["end_lineno"],
                        }
            if not expected_exception or id(call) in expected_calls:
                calls.append((caller, source, function, call, target))
    procedures = {
        identity
        for identity, (_, function) in functions.items()
        if identity in predicates
        or not any(
            isinstance(child, ast.Return)
            and child.value is not None
            and not (isinstance(child.value, ast.Constant) and child.value.value is None)
            and enclosing(child, (ast.FunctionDef, ast.AsyncFunctionDef)) is function
            for child in _walk(function)
        )
    }
    relevant = numerical_functions & procedures
    while True:
        expanded = relevant | {caller for caller, _, _, _, target in calls if target in relevant and caller in procedures}
        if expanded == relevant:
            break
        relevant = expanded

    def message_fragments(node: ast.AST, identity: str) -> list[str]:
        messages = [node.msg] if isinstance(node, ast.Assert) else []
        if identity in guard_messages:
            messages.append(guard_messages[identity])
        if isinstance(node, ast.Call):
            messages.extend(keyword.value for keyword in node.keywords if keyword.arg == "err_msg")
        fragments = []
        for message in messages:
            parts = message.values if isinstance(message, ast.JoinedStr) else [message]
            fragments.extend(part.value for part in parts if isinstance(part, ast.Constant) and isinstance(part.value, str))
        return fragments

    def supported_expected_helper(target: str, expected: dict[str, Any]) -> bool:
        """Admit an expected-error call whose comparison-procedure closure declares numerical comparisons.

        The failure may occur in the helper itself or in a comparison
        procedure it calls normally, in any branch or loop iteration. The
        receipts bind the actual failing site, the prefix completed before it
        and every linked call the exception propagated through. A literal
        ``match`` narrows the declared failure sites to comparisons whose
        source message contains it, when any does.
        """
        reachable: set[str] = set()
        stack = [target]
        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)
            stack.extend(
                callee
                for caller, _, _, call, callee in calls
                if caller == current and callee in relevant and id(call) not in expected_calls
            )
        numerical = [
            row
            for row in observations
            if row["kind"] in {"tolerance-comparison", "exact-comparison", "numerical-bound"}
            and (row.get("required_in_invocation") or row.get("conditional_execution", {}).get("function")) in reachable
        ]
        if not numerical:
            return False
        failure_sites = sorted(row["id"] for row in numerical)
        match = expected["match"]
        if isinstance(match, str) and match and not any(character in match for character in r".^$*+?{}[]\|()"):
            matching = sorted(
                row["id"]
                for row in numerical
                if any(
                    match in fragment for fragment in message_fragments(observation_nodes.get(row["id"], ast.Pass()), row["id"])
                )
            )
            failure_sites = matching or failure_sites
        expected["failure_sites"] = failure_sites
        return True

    calls = [
        row
        for row in calls
        if id(row[3]) not in expected_calls or supported_expected_helper(row[4], expected_calls[id(row[3])])
    ]
    # Unsupported expected-error calls were excluded before relevance discovery
    # historically; removing them must not leave a synthetic helper dependency.
    relevant = numerical_functions & procedures
    while True:
        expanded = relevant | {caller for caller, _, _, _, target in calls if target in relevant and caller in procedures}
        if expanded == relevant:
            break
        relevant = expanded
    for caller, source, function, call, target in calls:
        if target not in relevant and id(call) not in expected_calls:
            continue
        address = source_index(source).observation_address(call)
        identity = address["id"] + "-helper-call"
        row = {
            **address,
            "id": identity,
            "end_lineno": call.end_lineno,
            "role": "route-engagement",
            "kind": "comparison-helper-call",
            "expression": _source_expression(call),
            "helper_call": {"callee": target, "position": [call.lineno, call.end_lineno, call.col_offset, call.end_col_offset]},
        }
        assertion = enclosing(call, ast.Assert)
        if (
            assertion is not None
            and isinstance(assertion.test, ast.UnaryOp)
            and isinstance(assertion.test.op, ast.Not)
            and assertion.test.operand is call
        ):
            # ``assert not predicate(...)``: the call expects a false return,
            # so the predicate's returned bound is not positive evidence.
            row["helper_call"]["negated"] = True
        if not isinstance(call_parents.get(id(call)), ast.Expr) and assertion is None:
            row["unsupported_execution"] = "comparison procedure must be called in an explicit statement or assertion"
        conditional = conditional_execution(call, call_parents, source.as_posix(), test, allow_try_body=True)
        required = set()
        for statement in function.body:
            visit_required(statement, required, helper_call=True)
        if id(call) in required:
            row["required_in_invocation"] = caller
            if function is test:
                row["required_execution"] = True
        elif conditional is not None:
            row["conditional_execution"] = conditional
        else:
            row["unsupported_execution"] = "comparison helper call has no supported normal-call or selected-branch scope"
        if (expected := expected_calls.get(id(call))) and ("required_in_invocation" in row or "conditional_execution" in row):
            # The raises context shares the call's execution scope; only its
            # accepting exit (the actual exception) completes it.
            row["helper_call"]["expected_error"] = expected
            context_observation = next(item for item in observations if item["id"] == expected["observation"])
            for key in EXECUTION_SCOPE_KEYS:
                context_observation.pop(key, None)
                if key in row:
                    context_observation[key] = copy.deepcopy(row[key])
            context_observation["expected_error_context"] = {"function": caller, "end_lineno": expected["end_lineno"]}
        cursor = call
        while (cursor := call_parents.get(id(cursor))) is not None and cursor is not function:
            if isinstance(cursor, (ast.BoolOp, ast.IfExp, *UNTRACED_COMPARISON_SCOPES)) or (
                isinstance(cursor, (ast.If, ast.For, ast.While)) and cursor.lineno == call.lineno
            ):
                row["unsupported_execution"] = "put the comparison helper call in an explicit synchronous statement"
        observations.append(row)
        described = source_index(source).helper_inequality_observation(call)
        if described is not None:
            numerical = {key: value for key, value in row.items() if key != "helper_call"}
            numerical.update(
                id=identity + "-value",
                kind="numerical-bound",
                role="verdict",
                actual=described["actual"],
                reference=described["reference"],
                tolerance_provenance=described["provenance"],
            )
            if described["actual"] == described["reference"] or source_index(source).comparison_aliases(
                call, described["actual"], described["reference"]
            ):
                numerical["unsupported_execution"] = "numerical comparison uses aliases of the same source value"
            observations.append(numerical)
    for source, reference, target in escaped_helpers:
        if target in relevant:
            observations.append(
                {
                    **source_index(source).observation_address(reference),
                    "id": source_index(source).observation_address(reference)["id"] + "-helper-reference",
                    "end_lineno": reference.end_lineno,
                    "role": "route-engagement",
                    "kind": "comparison-helper-reference",
                    "expression": _source_expression(reference),
                    "unsupported_execution": "call the comparison procedure explicitly instead of passing an untraced callback",
                }
            )
    return sorted(observations, key=lambda row: (row["id"], row["expression"]))


def _has_marker(nodes: list[ast.AST], marker: str) -> bool:
    return any(
        marker in _dotted_name(decorator.func if isinstance(decorator, ast.Call) else decorator)
        for node in nodes
        for decorator in getattr(node, "decorator_list", ())
    )


def _module_has_marker(tree: ast.Module, marker: str) -> bool:
    for statement in tree.body:
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets, value = [statement.target], statement.value
        if value is None or not any("pytestmark" in _target_names(target) for target in targets):
            continue
        if any(
            marker in _dotted_name(child.func if isinstance(child, ast.Call) else child)
            for child in ast.walk(value)
            if isinstance(child, (ast.Call, ast.Attribute, ast.Name))
        ):
            return True
    return False


def _marker_reasons(nodes: list[ast.AST], marker: str) -> list[str]:
    reasons: list[str] = []
    for node in nodes:
        candidates = getattr(node, "decorator_list", None)
        children = candidates if candidates is not None else _walk(node)
        for child in children:
            if not isinstance(child, ast.Call) or marker not in _dotted_name(child.func):
                continue
            reason = next((keyword.value for keyword in child.keywords if keyword.arg == "reason"), None)
            reasons.append(_source_expression(reason) if reason is not None else "reason-not-declared")
    return reasons


def _marker_contracts(nodes: list[ast.AST], marker: str) -> list[dict[str, str]]:
    """Return the review fields carried by a pytest marker.

    ``reason`` remains human prose.  ``alternative_guarantee_ref`` is a stable
    SSoT identifier used only by finite-difference/numerical-oracle precision
    exclusions; it is deliberately not inferred from a neighbouring test.
    """
    contracts: list[dict[str, str]] = []
    for node in nodes:
        candidates = getattr(node, "decorator_list", None)
        children = candidates if candidates is not None else _walk(node)
        for child in children:
            if not isinstance(child, ast.Call) or marker not in _dotted_name(child.func):
                continue
            keywords = {keyword.arg: keyword.value for keyword in child.keywords if keyword.arg is not None}
            row: dict[str, str] = {}
            for field in ("reason", "alternative_guarantee_ref", "mode", "position_source"):
                value = keywords.get(field)
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    row[field] = value.value
                elif value is not None:
                    row[field] = _source_expression(value)
            contracts.append(row)
    return contracts


def _module_marker_reasons(tree: ast.Module, marker: str) -> list[str]:
    values: list[ast.AST] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if any("pytestmark" in _target_names(target) for target in statement.targets):
                values.append(statement.value)
        elif isinstance(statement, ast.AnnAssign) and "pytestmark" in _target_names(statement.target):
            if statement.value is not None:
                values.append(statement.value)
    return _marker_reasons(values, marker)


def _numerical_oracle_precision_exclusion(contract: dict[str, str]) -> bool:
    reason = contract.get("reason", "")
    lowered = reason.lower()
    if "finite difference" in lowered or "finite-difference" in lowered or "fd " in lowered or "fd /" in lowered:
        return True
    # An AD cross-check can itself be precision-limited even though the
    # production analytical route is supported in mixed precision.  It is a
    # typed numerical-oracle exclusion only when the test names a blocking
    # mixed-precision alternative explicitly; prose alone cannot create it.
    return bool(contract.get("alternative_guarantee_ref")) and any(
        token in lowered for token in ("autodiff", "autograd", "cross-check")
    )


def _numerical_oracle_marker(mark: str) -> bool:
    """Classify a source-rendered per-parameter mixed-skip marker."""
    lowered = mark.lower()
    if "finite difference" in lowered or "finite-difference" in lowered or "fd " in lowered or "fd /" in lowered:
        return True
    return "alternative_guarantee_ref" in lowered and any(token in lowered for token in ("autodiff", "autograd", "cross-check"))


def _marker_alternative_ref(mark: str) -> str | None:
    match = re.search(r"alternative_guarantee_ref\s*=\s*(['\"])([^'\"]+)\1", mark)
    return match.group(2) if match is not None else None


def _selector_backends(selector: dict[str, Any]) -> list[str]:
    """Resolve an explicitly parameterized backend to its receipt identity."""
    selected: set[str] = set()
    for key in ("backend", "mode"):
        value = selector.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.lower()
        if normalized in {"xla", "jax"}:
            selected.add("xla")
        elif normalized in {"rust", "fast", "native"}:
            selected.add("rust")
        elif normalized == "python":
            selected.add("python")
    return [backend for backend in ("xla", "rust", "python") if backend in selected]


def _execution_contract(
    *,
    pytest_node: str,
    test: ast.FunctionDef | ast.AsyncFunctionDef,
    class_nodes: list[ast.ClassDef],
    reachable: list[ast.AST],
    calls: list[str],
    concrete_cases: list[dict[str, Any]],
    realized_scope: str,
    module_tree: ast.Module,
) -> dict[str, Any]:
    path = pytest_node.split("::", 1)[0]
    source = "\n".join(_source_expression(node) for node in reachable)
    decorators: list[ast.AST] = [*class_nodes, test]
    mixed_skip_contracts = _marker_contracts(decorators, "skip_mixed_precision")
    mixed_skip_reasons = [row.get("reason", "reason-not-declared") for row in mixed_skip_contracts]
    # Full-precision-only rule for external fp64 references: exactly the seven
    # TurboRVB external-fp64-literal nodes, with exactly the reviewed marker, are
    # full-only. Every other non-FD mixed skip stays unsupported (forbidden).
    is_external_fp64_full_only = external_fp64_full_only(pytest_node, mixed_skip_contracts)
    if path.startswith("tests/test_jqmc_workflow/"):
        precision_modes = ["not-applicable"]
        precision_reason = "workflow/serialization contract does not execute a precision-dependent numerical kernel"
    elif (
        path.endswith("test_mixed_precision.py")
        or ("if not is_mixed_precision_enabled()" in source and "pytest.skip" in source)
        or _marker_contracts(decorators, "skip_full_precision")
        or _module_marker_reasons(module_tree, "skip_full_precision")
    ):
        precision_modes = ["mixed"]
        precision_reason = None
    elif is_external_fp64_full_only:
        precision_modes = ["full"]
        precision_reason = EXTERNAL_FP64_FULL_ONLY_REASON
    elif mixed_skip_contracts and all(_numerical_oracle_precision_exclusion(row) for row in mixed_skip_contracts):
        precision_modes = ["full"]
        precision_reason = "; ".join(mixed_skip_reasons)
    else:
        precision_modes = ["full", "mixed"]
        precision_reason = None

    declared_backends = {backend for case in concrete_cases for backend in _selector_backends(case.get("selector", {}))}

    rust_subject = path.startswith("tests/test_rs_kernel/") or any(name.startswith("jqmc_rs.") for name in calls)
    xla_numeric_subject = any(
        token in name.lower()
        for name in calls
        for token in (
            "compute_aos",
            "compute_mos",
            "compute_ewald",
            "compute_jastrow",
            "compute_geminal",
            "compute_det",
            "evaluate_wavefunction",
            "evaluate_ln_wavefunction",
            "compute_local_energy",
        )
    )
    if declared_backends:
        backends = [backend for backend in ("xla", "rust", "python") if backend in declared_backends]
    elif rust_subject:
        backends = ["rust"]
    elif (
        realized_scope == "not-applicable"
        and not xla_numeric_subject
        and not any(name.startswith(("jax.", "jnp.")) for name in calls)
    ):
        backends = ["python"]
    else:
        backends = ["xla"]

    # Launcher identity belongs to the governed test itself.  Looking through
    # every reachable helper overclassified ordinary serial optimizer tests
    # whenever a shared helper also contained an MPI branch.  Likewise a
    # ``requires_mpirun`` availability marker on a test that *spawns* an MPI
    # subprocess does not mean pytest itself runs under MPI.
    test_source = _source_expression(test)
    # A launch requirement may be owned by a pytest fixture injected directly
    # into the governed test.  Inspect only those direct fixture definitions:
    # following every reachable helper would overclassify serial tests that
    # happen to share a helper containing an unrelated MPI branch.
    argument_names = {
        argument.arg
        for argument in (*test.args.posonlyargs, *test.args.args, *test.args.kwonlyargs)
        if argument.arg not in {"self", "cls", "request"}
    }
    direct_fixture_sources = []
    for candidate in module_tree.body:
        if not isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)) or candidate.name not in argument_names:
            continue
        if not any(
            _dotted_name(decorator.func if isinstance(decorator, ast.Call) else decorator).endswith("pytest.fixture")
            for decorator in candidate.decorator_list
        ):
            continue
        direct_fixture_sources.append(_source_expression(candidate))
    launcher_source = "\n".join((test_source, *direct_fixture_sources))
    test_name = test.name.lower()
    file_stem = Path(path).stem.lower()
    launch_size_read = (
        bool(re.search(r"(?:Get_size|process_count|get_mpi_size)\s*\(", launcher_source))
        or bool(re.search(r"\b(?:mpi_size|world_size)\s*(?:!=|==|<|>)", launcher_source))
        or bool(re.search(r"\b\w+\.size\s*(?:%|!=|==|<|>)", launcher_source))
    )
    explicitly_serial_world = bool(
        re.search(r"Get_size\s*\(\s*\)\s*!=\s*1", launcher_source) and "pytest.skip" in launcher_source
    )
    mpi_named = file_stem.endswith("_mpi") or "_mpi" in file_stem or "multirank" in test_name

    # MPI-world eligibility markers, read like the precision markers
    # (tests/conftest.py deselects a marked test outside its world):
    # ``skip_serial`` is an MPI launch, ``skip_mpi`` a single process.
    def _declares(marker: str) -> bool:
        return bool(_marker_contracts(decorators, marker) or _module_marker_reasons(module_tree, marker))

    declared_mpi = _declares("skip_serial")
    explicitly_serial_world = explicitly_serial_world or _declares("skip_mpi")
    needs_mpi = (
        mpi_named or declared_mpi or (launch_size_read and "pytest.skip" in launcher_source and not explicitly_serial_world)
    )
    profiles = [{"name": "mpi", "world_size": 2}] if needs_mpi else [{"name": "serial", "world_size": 1}]
    execution: dict[str, Any] = {
        "precision_modes": precision_modes,
        "backends": backends,
        "profiles": profiles,
    }
    if precision_reason is not None:
        if precision_modes == ["not-applicable"]:
            execution["precision_exclusion"] = {
                "mode": "precision",
                "disposition": "not-applicable",
                "kind": "precision-independent-contract",
                "reason": precision_reason,
            }
        elif is_external_fp64_full_only:
            execution["precision_exclusion"] = {
                "mode": "mixed",
                "disposition": "not-applicable",
                "kind": EXTERNAL_FP64_FULL_ONLY_KIND,
                "reason": precision_reason,
                "decision": EXTERNAL_FP64_FULL_ONLY_DECISION,
                "alternative_guarantee_refs": [],
            }
        else:
            alternative_refs = sorted(
                {row["alternative_guarantee_ref"] for row in mixed_skip_contracts if row.get("alternative_guarantee_ref")}
            )
            execution["precision_exclusion"] = {
                "mode": "mixed",
                "disposition": "not-applicable",
                "kind": "finite-difference-or-numerical-oracle-not-applicable",
                "reason": precision_reason,
                "alternative_guarantee_refs": alternative_refs,
            }
    non_fd_skips = [
        row.get("reason", "reason-not-declared")
        for row in mixed_skip_contracts
        if not is_external_fp64_full_only and not _numerical_oracle_precision_exclusion(row)
    ]
    if non_fd_skips:
        execution["unsupported_mixed_skips"] = non_fd_skips
    return execution


@cache
def _sha256(path: Path) -> str:
    # Committed bytes, or the pinned sha256 of a manifest-pinned pseudopotential.
    return fixture_digest(path)


def _stable_ast_dump(node: ast.AST) -> str:
    """Return an AST dump stable across the supported Python versions.

    Python 3.12 added an empty ``type_params`` field to function and class
    nodes, and early 3.12 patch releases emitted empty f-string constants that
    later releases omit.  Both are interpreter-owned defaults.  Non-empty type
    parameters cannot occur in source parsed by the oldest supported
    interpreter, so only the added empty field is removed after structural
    normalization.
    """
    normalized = _normalized_ast_view(node)
    try:
        # Python 3.13 changed ``ast.dump`` to hide empty optional fields by
        # default.  Request the pre-3.13 complete structural spelling so the
        # same source has one snapshot on every supported interpreter.
        dumped = ast.dump(normalized, include_attributes=False, show_empty=True)
    except TypeError:
        dumped = ast.dump(normalized, include_attributes=False)
    return dumped.replace(", type_params=[]", "")


def _reachable_ast_dump(node: ast.AST) -> str:
    table = _memo_table("stable_ast_dump") if _module_node(node) else None
    if table is not None and node in table:
        return table[node]
    dumped = _stable_ast_dump(node)
    if table is not None:
        table[node] = dumped
    return dumped


def _source_snapshot(test_path: Path, reachable: list[ast.AST]) -> str:
    payload = {
        "path": test_path.relative_to(ROOT).as_posix(),
        "file_sha256": _sha256(test_path),
        "reachable_ast": [_reachable_ast_dump(node) for node in reachable],
        "helper_sources": {
            path.as_posix(): _sha256(ROOT / path)
            for path in sorted({getattr(node, "_gdd_source_path", test_path.relative_to(ROOT)) for node in reachable})
            if path != test_path.relative_to(ROOT)
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _assertions(nodes: list[ast.AST]) -> tuple[list[str], list[str]]:
    assertions: set[str] = set()
    tolerances: set[str] = set()
    for node in nodes:
        for child in _walk(node):
            if isinstance(child, ast.Assert):
                assertions.add(_source_expression(child.test))
            elif isinstance(child, ast.Call):
                name = _dotted_name(child.func)
                if _is_assertion_call(name):
                    assertions.add(name)
            if isinstance(child, (ast.Name, ast.Attribute)):
                name = _dotted_name(child)
                if any(token in name.upper() for token in ("ATOL", "RTOL", "MAXULP", "TOLERANCE")):
                    tolerances.add(name)
    return sorted(assertions), sorted(tolerances)


def derive_case_binding(
    pytest_node: str,
    expected_cases: int,
    verdict_method: str,
    wavefunction_scope: str,
) -> dict[str, Any]:
    """Return the deterministic binding snapshot for one registered pytest node."""
    with derivation_cache():
        return _derive_case_binding(pytest_node, expected_cases, verdict_method, wavefunction_scope)


def _derive_case_binding(
    pytest_node: str,
    expected_cases: int,
    verdict_method: str,
    wavefunction_scope: str,
) -> dict[str, Any]:
    parts = pytest_node.split("::")
    test_path = ROOT / parts[0]
    index = _module_index(test_path)
    test, class_nodes = index.test(parts[1:])
    reachable = index.reachable(test, class_nodes)

    parameter_names: set[str] = set()
    decorators = [decorator for class_node in class_nodes for decorator in class_node.decorator_list]
    decorators.extend(test.decorator_list)
    no_committed_artifact_contracts = _marker_contracts([*class_nodes, test], "gdd_no_committed_artifact")
    no_committed_artifact_reasons = [row.get("reason", "reason-not-declared") for row in no_committed_artifact_contracts]
    electron_overrides = _marker_contracts([*class_nodes, test], "gdd_electron_configuration")
    if len(electron_overrides) > 1:
        raise ValueError(f"{pytest_node}: more than one gdd_electron_configuration marker is declared")
    electron_override = electron_overrides[0] if electron_overrides else {}
    physics_overrides = _marker_contracts([*class_nodes, test], "gdd_physics_object")
    if len(physics_overrides) > 1:
        raise ValueError(f"{pytest_node}: more than one gdd_physics_object marker is declared")
    physics_override = physics_overrides[0] if physics_overrides else {}
    # Directly parametrized arguments carry their selector value into the body;
    # ``indirect`` ones pass through a fixture and never decide a branch.
    direct_parameter_names: set[str] = set()
    for decorator in decorators:
        if not _is_parametrize(decorator) or len(decorator.args) < 2:
            continue
        names = _parametrize_names(decorator.args[0])
        parameter_names.update(names)
        if not any(keyword.arg == "indirect" for keyword in decorator.keywords):
            direct_parameter_names.update(names)

    arguments = [*test.args.posonlyargs, *test.args.args, *test.args.kwonlyargs]
    pytest_fixtures = sorted(
        argument.arg for argument in arguments if argument.arg != "self" and argument.arg not in parameter_names
    )
    local_fixtures = dict(index.functions)
    local_fixtures.update(
        {
            node.name: node
            for class_node in class_nodes
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    )
    fixture_queue = list(pytest_fixtures)
    fixture_seen: set[str] = set()
    # Fixture-owned nodes, kept for the per-case branch-pruned graph.
    fixture_reachable: list[ast.AST] = []
    while fixture_queue:
        fixture_name = fixture_queue.pop()
        if fixture_name in fixture_seen:
            continue
        fixture_seen.add(fixture_name)
        fixture = local_fixtures.get(fixture_name)
        if fixture is None:
            continue
        for node in index.reachable(fixture, class_nodes if fixture_name not in index.functions else []):
            fixture_reachable.append(node)
            if node not in reachable:
                reachable.append(node)
        fixture_arguments = [*fixture.args.posonlyargs, *fixture.args.args, *fixture.args.kwonlyargs]
        fixture_queue.extend(
            argument.arg
            for argument in fixture_arguments
            if argument.arg not in {"self", "cls", "request"} and argument.arg in local_fixtures
        )
    parameterization, concrete_cases = index.parameter_cases(test, class_nodes, pytest_fixtures, expected_cases)
    strings = {value for node in reachable for value in _string_values(node)}
    # Preserve complete inert UPF path expressions: individual AST string
    # literals lose the distinction between standalone and QE-save copies.
    strings.update(
        path
        for owner in reachable
        for child in _walk(owner)
        if isinstance(child, (ast.BinOp, ast.Call))
        and (path := _explicit_fixture_path(_source_expression(child))) is not None
        and path.lower().endswith(".upf")
    )
    sources = _registered_sources(strings)
    resolved_case_sources: dict[int, list[dict[str, str]]] = {}
    # Call-argument literals depend only on ``reachable``, which is fixed for
    # the case loop; collect them once instead of once per concrete case.
    call_argument_strings: set[str] | None = None
    for case in concrete_cases:
        selected = _case_sources(case["selector"], index.constants, reachable)
        case_strings = _case_strings(reachable, case["selector"], index.constants)
        if call_argument_strings is None:
            call_argument_strings = set()
            for owner in reachable:
                roots = owner.body if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)) else [owner]
                for root in roots:
                    for child in _walk(root):
                        if not isinstance(child, ast.Call):
                            continue
                        for argument in (*child.args, *(keyword.value for keyword in child.keywords)):
                            call_argument_strings.update(_string_values(argument))
        # Complete standalone UPF path expressions resolved on this case's path
        # (``_QE_ROOT / "Li.ccECP.upf"``) are explicit sources even when they
        # are not themselves a call-argument literal.
        case_upf_paths = {value for value in case_strings if value.startswith("tests/") and value.lower().endswith(".upf")}
        inferred = (
            _registered_sources(case_strings.intersection(call_argument_strings) | case_upf_paths)
            if selected
            else _case_sources(
                {**case["selector"], "reachable_strings": sorted(case_strings)},
                index.constants,
                reachable,
            )
        )
        if not selected and not inferred:
            inferred.extend(_imported_case_sources(index, reachable, case["selector"]))
        for row in inferred:
            if row not in selected:
                selected.append(row)
        selected.sort(key=lambda row: row["source_id"])
        resolved_case_sources[case["case"]] = selected
    case_source_rows = [row for selected in resolved_case_sources.values() for row in selected]
    for row in case_source_rows:
        if row not in sources:
            sources.append(row)
    sources.sort(key=lambda row: row["source_id"])
    if no_committed_artifact_reasons:
        sources = []
        resolved_case_sources = {case["case"]: [] for case in concrete_cases}
    artifacts = (
        []
        if no_committed_artifact_reasons
        else sorted(set(_repo_artifacts(strings, test_path)) | {row["path"] for row in sources})
    )
    registered_paths = {row["path"] for row in sources}
    sources.extend(
        {
            "source_id": f"UNREGISTERED::{path}",
            "path": path,
            "format": Path(path).suffix.lstrip(".") or "directory",
        }
        for path in artifacts
        if path not in registered_paths
    )
    sources.sort(key=lambda row: row["source_id"])
    source_by_path = {row["path"]: row for row in sources}
    case_artifacts: dict[int, list[dict[str, str]]] = {}
    for case in concrete_cases:
        # A path-valued pytest parameter is the strongest case-local authority.
        # Module-level parameter tables contain the paths for *all* rows, so walking
        # every reachable literal would otherwise attach the entire table to each
        # concrete case in the human ledger.
        selected = resolved_case_sources[case["case"]]
        if not selected:
            case_strings = _case_strings(reachable, case["selector"], index.constants)
            paths = _repo_artifacts(case_strings, test_path)
            selected = [source_by_path[path] for path in paths if path in source_by_path]
        case_artifacts[case["case"]] = selected
    fallback_sources = sources if len(sources) == 1 and not any(case_artifacts.values()) else []
    # Imported test-driver calls need the same coordinate/physics contract as
    # their directly written bodies. Keep artifact/case inference above local:
    # helper parameter tables must not broaden the caller fixture universe.
    # A merely passed callback is not evidence that its driver was invoked.
    physics_dependencies = index.assertion_dependencies(reachable, include_callbacks=False)
    calls = _calls(physics_dependencies)
    construction = _construction_calls(calls)
    physics = _physics_symbols(calls)
    coordinates = _coordinate_symbols(physics_dependencies)
    generators = [
        name
        for name in calls
        if any(word in name.lower() for word in _RANDOM_WORDS) and name.rsplit(".", 1)[-1] not in {"get_state", "set_state"}
    ]
    assertions, tolerances = _assertions(reachable)
    assertion_dependencies = index.assertion_dependencies(reachable)
    assertion_contract = _assertion_contract(assertion_dependencies, test_path.relative_to(ROOT), test)
    driver_coordinates = _is_production_driver_route(calls, physics_dependencies) or (
        electron_override.get("mode") == "required"
        and electron_override.get("position_source") == "production-driver-generated"
    )

    if sources:
        sources = [{**row, "sha256": _sha256(ROOT / row["path"])} for row in sources]
        source_binding: dict[str, Any] = {
            "kind": "committed-artifact",
            "artifacts": sources,
            "cases": [
                {
                    "case": case["case"],
                    "artifacts": [
                        {**row, "sha256": _sha256(ROOT / row["path"])}
                        for row in (case_artifacts[case["case"]] or fallback_sources)
                    ],
                }
                for case in concrete_cases
            ],
        }
        input_kind = "committed-artifact"
    else:
        reason = (
            no_committed_artifact_reasons[0] if no_committed_artifact_reasons else "no committed source artifact is reachable"
        )
        source_binding = {"kind": "not-applicable", "reason": reason}
        input_kind = "parameter-set"

    realized_scope = _inferred_wavefunction_scope(
        calls,
        sources,
        wavefunction_scope,
        production_driver_route=driver_coordinates,
    )
    runtime_object, runtime_object_basis = _runtime_physics_object_contract(
        realized_scope,
        physics,
        assertion_contract,
    )
    # An explicit per-test physics-object disposition, mirroring
    # gdd_electron_configuration. "not-required" needs a reason and is rejected
    # when the test itself reaches a physics-object constructor or a production
    # driver route; "required" can only strengthen the derived disposition.
    declared_physics_mode = physics_override.get("mode")
    if declared_physics_mode not in {None, "required", "not-required"}:
        raise ValueError(f"{pytest_node}: invalid GDD physics-object mode {declared_physics_mode!r}")
    physics_declaration: dict[str, str] = {}
    if declared_physics_mode == "not-required":
        reason = physics_override.get("reason")
        if not reason or not reason.strip():
            raise ValueError(f"{pytest_node}: not-required physics object needs a reason")
        constructors = _physics_object_constructors(calls)
        if constructors or driver_coordinates:
            raise ValueError(
                f"{pytest_node}: not-required physics object conflicts with reachable physics-object construction "
                f"{constructors or ['production-driver-route']}"
            )
        runtime_object, runtime_object_basis = "not-required", "declared-no-physics-object"
        physics_declaration = {
            "runtime_object_reason": reason,
            "runtime_object_declared_by": "pytest.mark.gdd_physics_object",
        }
    elif declared_physics_mode == "required":
        runtime_object, runtime_object_basis = "required", "declared-physics-object"
        physics_declaration = {"runtime_object_declared_by": "pytest.mark.gdd_physics_object"}
    if realized_scope == "not-applicable":
        wavefunction: dict[str, Any] = {
            "scope": "not-applicable",
            "runtime_object": runtime_object,
            "reason": "test does not construct a wavefunction",
            "runtime_object_basis": runtime_object_basis,
            **physics_declaration,
        }
    else:
        wavefunction = {
            "scope": realized_scope,
            "runtime_object": runtime_object,
            "runtime_object_basis": runtime_object_basis,
            **physics_declaration,
            "construction": list(construction),
            "physics_symbols": physics or ["explicit-raw-numerical-component-fixture"],
        }

    declared_electron_mode = electron_override.get("mode")
    if declared_electron_mode not in {None, "required", "not-required"}:
        raise ValueError(f"{pytest_node}: invalid GDD electron-configuration mode {declared_electron_mode!r}")
    if declared_electron_mode == "not-required":
        reason = electron_override.get("reason")
        if not reason:
            raise ValueError(f"{pytest_node}: not-required electron configuration needs a reason")
        electron = {
            "mode": "not-required",
            "reason": reason,
            "declared_by": "pytest.mark.gdd_electron_configuration",
        }
    elif coordinates or driver_coordinates or declared_electron_mode == "required":
        generic_conditioning_assertions = [
            assertion for assertion in assertions if _CONDITIONING_PATTERN.search(assertion.lower())
        ]
        conditioning_assertions = _geminal_conditioning_assertions(physics_dependencies, generic_conditioning_assertions)
        declared_position_source = electron_override.get("position_source")
        electron: dict[str, Any] = {
            "mode": "required",
            "position_source": (
                declared_position_source
                or ("seeded-random" if generators else "production-driver-generated" if driver_coordinates else "explicit")
            ),
            "coordinate_symbols": coordinates or (["production-driver-walker-state"] if driver_coordinates else []),
            "generators": (
                ["explicit-test-source"]
                if declared_position_source == "explicit"
                else generators or (["production-driver-initialization"] if driver_coordinates else ["explicit-test-source"])
            ),
            "seeds": (
                ["not-applicable"]
                if declared_position_source == "explicit"
                else _seed_literals(
                    physics_dependencies, source_index=_tolerance_source_index(index.tree, test_path.relative_to(ROOT))
                )
                or ["no-seed-random-generator-observed"]
            ),
            "cond_G": (
                {
                    "subject": "geminal-matrix-G",
                    "operator": "asserted-runtime-constraint",
                    "constraints": conditioning_assertions,
                    "realized_observation": {
                        "value": "runtime-receipt-required",
                        "dimension": "runtime-receipt-required",
                        "dtype": "runtime-receipt-required",
                        "accepted_draw": "runtime-receipt-required-if-sampled",
                    },
                }
                if conditioning_assertions
                else "not-limited"
            ),
        }
        if conditioning_assertions:
            electron["conditioning_assertions"] = conditioning_assertions
        evaluation_conditioning = _evaluation_conditioning_contract(
            index.tree, physics_dependencies, assertions, parameter_names, pytest_node
        )
        if evaluation_conditioning is not None:
            electron["evaluation_conditioning"] = evaluation_conditioning
        if electron_override:
            electron["declared_by"] = "pytest.mark.gdd_electron_configuration"
    else:
        electron = {"mode": "not-required", "reason": "no electron/evaluation-coordinate symbol is reachable"}
    # Per-parametrize-case electron disposition. A derived node
    # obligation is narrowed for a case only when the case's branch-pruned
    # graph (fixtures included) reaches no coordinate evaluation. A per-case
    # ``gdd_electron_configuration(mode="not-required", reason=...)`` mark is
    # accepted only on a statically decided branch whose own graph (fixtures
    # excluded) reaches none; on an undecided branch it is recorded as not
    # applied (obligation kept); any other per-case declaration is rejected.
    for case in concrete_cases:
        declarations = _case_electron_declarations(case)
        selector = {name: value for name, value in case.get("selector", {}).items() if name in direct_parameter_names}
        if declarations:
            where = f"{pytest_node}[{case.get('parameter_id', case['case'])}]"
            if len(declarations) > 1:
                raise ValueError(f"{where}: more than one per-case gdd_electron_configuration mark is declared")
            declaration = declarations[0]
            if declaration.get("mode") != "not-required" or not declaration.get("reason"):
                raise ValueError(f"{where}: a per-case electron declaration must be mode='not-required' with a reason")
            if electron["mode"] != "required" or declared_electron_mode is not None:
                raise ValueError(f"{where}: a per-case electron declaration needs a derived node-level 'required' obligation")
            branch_own = _case_coordinates_required(
                index, test, class_nodes, fixture_reachable, selector, fixture_names=frozenset(pytest_fixtures)
            )
            if branch_own is None:
                # Not verifiable here (e.g. a forwarding entry that
                # mirrors the forwarded checker's cases): the obligation stays.
                case["electron_declaration_not_applied"] = {
                    "declared_reason": declaration["reason"],
                    "why": "no branch of this test is statically decided by the case; the node obligation is kept",
                    "decision": "per-case-electron-disposition",
                }
                continue
            if branch_own:
                raise ValueError(f"{where}: per-case 'not-required' declared, but the case's own branch reaches coordinates")
            case["electron_configuration"] = {
                "mode": "not-required",
                "reason": declaration["reason"],
                "declared_by": "pytest.param marks=pytest.mark.gdd_electron_configuration",
                "verified_by": "gdd.engine.case_binding_inventory._branch_pruned_test (branch-own graph)",
                "decision": "per-case-electron-disposition",
            }
        elif (
            electron["mode"] == "required"
            and declared_electron_mode is None
            and _case_coordinates_required(index, test, class_nodes, fixture_reachable, selector) is False
        ):
            case["electron_configuration"] = {
                "mode": "not-required",
                "reason": "branch-pruned static derivation: no electron/evaluation-coordinate symbol is reachable "
                "on this parametrize case's branch",
                "derived_by": "gdd.engine.case_binding_inventory._branch_pruned_test",
                "decision": "per-case-electron-disposition",
            }

    # Per-case finiteness-only evidence (smoke / stability); per case only.
    if _marker_contracts([*class_nodes, test], _FINITENESS_EVIDENCE_MARK):
        raise ValueError(f"{pytest_node}: {_FINITENESS_EVIDENCE_MARK} is a per-case (pytest.param) declaration only")
    _case_finiteness_evidence(
        pytest_node, test, test_path, verdict_method, assertion_contract, concrete_cases, direct_parameter_names
    )

    claim_id = _claim_id(test_path, parts[1:])
    execution = _execution_contract(
        pytest_node=pytest_node,
        test=test,
        class_nodes=class_nodes,
        reachable=reachable,
        calls=calls,
        concrete_cases=concrete_cases,
        realized_scope=realized_scope,
        module_tree=index.tree,
    )
    for case in concrete_cases:
        case_backends = _selector_backends(case.get("selector", {}))
        if case_backends:
            case["backends"] = case_backends
        mixed_marks = [mark for mark in case.get("markers", ()) if "skip_mixed_precision" in mark]
        if not mixed_marks:
            continue
        # Full-precision-only rule for the adversarial LRDMC chains: exactly the
        # policy=adversarial case of the three Rust LRDMC long-chain nodes, with
        # exactly the reviewed marker, is full-only. Nothing else qualifies.
        if adversarial_chain_full_only_case(pytest_node, case.get("parameter_id"), case.get("selector"), mixed_marks):
            case["precision_modes"] = ["full"]
            case["precision_exclusion"] = {
                "mode": "mixed",
                "disposition": "not-applicable",
                "kind": ADVERSARIAL_CHAIN_FULL_ONLY_KIND,
                "reason": ADVERSARIAL_CHAIN_FULL_ONLY_REASON,
                "decision": ADVERSARIAL_CHAIN_FULL_ONLY_DECISION,
                "alternative_guarantee_refs": [],
            }
            continue
        if all(_numerical_oracle_marker(mark) for mark in mixed_marks):
            alternative_refs = sorted({ref for mark in mixed_marks if (ref := _marker_alternative_ref(mark)) is not None})
            case["precision_modes"] = ["full"]
            case["precision_exclusion"] = {
                "mode": "mixed",
                "disposition": "not-applicable",
                "kind": "finite-difference-or-numerical-oracle-not-applicable",
                "reason": "; ".join(mixed_marks),
                "alternative_guarantee_refs": alternative_refs,
            }
        else:
            case["precision_modes"] = list(execution["precision_modes"])
            case["unsupported_mixed_skips"] = [mark for mark in mixed_marks if not _numerical_oracle_marker(mark)]
    return {
        "test_summary": _test_summary(test),
        "claim_refs": [claim_id],
        "execution": execution,
        "fixture": {
            "pytest_fixtures": pytest_fixtures,
            "parameterization": parameterization,
            "concrete_cases": concrete_cases,
            "expected_cases": expected_cases,
            "source_location": f"{parts[0]}:{test.lineno}",
            "source_snapshot": _source_snapshot(test_path, assertion_dependencies),
        },
        "input": {
            "kind": input_kind,
            "construction": construction,
            "committed_artifacts": artifacts,
        },
        "sources": source_binding,
        "wavefunction": wavefunction,
        "electron_configuration": electron,
        "comparison": {
            "verdict_method": verdict_method,
            "assertions": assertions or ["bare-test-completion-or-exception-contract"],
            "tolerance_symbols": tolerances or ["not-declared"],
            "typed_observations": assertion_contract,
        },
    }
