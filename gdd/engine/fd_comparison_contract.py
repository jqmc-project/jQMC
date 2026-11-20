"""Finite, case-owned FD comparison contracts.

The default FD verdict remains unchanged for other guarantees. Only the reviewed
52-case fixture set can use the exact per-case bindings below. Loading contracts
and resolving central settings does not import scientific production modules.
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
import itertools
import json
from pathlib import Path

import yaml

from gdd.engine.comparison_requirements import resolve_requirement
from gdd.engine.guarantee_store import load_module

ROOT = Path(__file__).resolve().parents[2]
GUARANTEE = "NUM-DERIV-FD-AUTODIFF-001"
FIXTURE = "DERIV-FIX-FD-AUTODIFF-001"
METHOD = "VM-NUM-FD-ANCHOR-001"
CASE_AUTHORITY = {
    "guarantee": GUARANTEE,
    "fixture_set": FIXTURE,
    "node_field": "comparison_contract",
    "enforcement": "gdd.engine.fd_comparison_contract",
}
ZONE_POLICIES = {
    "CMP-ZONE-ALLCLOSE-STRICT-001": "strict",
    "CMP-ZONE-ALLCLOSE-MEDIUM-001": "medium",
}
FIXED_POLICY = "CMP-FD-AUTODIFF-FULL-001"
# Rule: a VM-NUM-FD-ANCHOR-001 comparison
# may use CENTRAL_PAIR_POLICY only when both operands are computed by pure NumPy
# in fp64, with no jqmc production numerics in either operand. It is checked
# structurally on the test source of every admitted site, never by a digest.
CENTRAL_PAIR_POLICY = "CMP-SOURCE-CENTRAL-PAIR-001"
PURE_NUMPY_COMPARATOR = "numpy.testing.assert_allclose"
# The FD anchor method itself: field set and semantic fields, not a fingerprint.
METHOD_FIELDS = frozenset(
    {"id", "classes", "oracle", "comparison_policies", "case_comparison_contract", "observation_policies", "meaning", "limits"}
)
METHOD_SEMANTICS = {
    "classes": ["numerical"],
    "oracle": "independent-finite-difference",
    "observation_policies": ["OBS-FD-ROUTE-001"],
}
# NumPy names that leave fp64 (or the floating point domain) are never pure fp64.
_NON_FP64_NAMES = frozenset(
    {
        "float16",
        "float32",
        "half",
        "single",
        "complex64",
        "csingle",
        "longdouble",
        "float128",
        "complex256",
        "clongdouble",
        "astype",
        "dtype",
    }
)
_FP64_BUILTINS = frozenset({"abs", "complex", "float"})
_PASS_THROUGH_ATTRIBUTES = frozenset({"real", "imag", "T"})
FD_OBSERVATIONS = (
    "perturbation-is-nonzero",
    "plus-and-minus-points-are-distinct",
    "autodiff-route-is-live",
    "finite-difference-result-is-finite",
    "stencil-declared",
)


def _yaml(path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def _require(condition, message):
    if not condition:
        raise AssertionError("FD comparison contract: " + message)


def _identity(evidence_id, parameters):
    return f"FD::{evidence_id}::" + ",".join(f"{name}={value}" for name, value in sorted(parameters.items()))


def _disposition(comparison):
    result = {"policy": comparison["policy"]}
    if comparison["policy"] in ZONE_POLICIES:
        result["zones"] = comparison["zones"]
    return result


def _module_bindings(tree):
    """Top-level imports (alias -> qualified name), assignments and definition names."""
    imports, assigned, definitions = {}, {}, set()
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.add(statement.name)
        elif isinstance(statement, ast.Import):
            for alias in statement.names:
                imports[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                imports[alias.asname or alias.name] = f"{'.' * statement.level}{statement.module or ''}.{alias.name}"
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        assigned.setdefault(name.id, []).append(statement.value)
    return imports, assigned, definitions


def _own_scope_nodes(function):
    """Every node of one function body, without descending into nested scopes."""
    pending = list(function.body)
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        pending.extend(ast.iter_child_nodes(node))


def _local_bindings(function):
    """Names bound in one function body: assignments and nested function definitions."""
    assigned, nested = {}, {}
    for node in _own_scope_nodes(function):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _require(
                isinstance(node, ast.FunctionDef) and node in function.body and node.name not in nested,
                "FD pure-numpy operand: nested definition is not a unique top-level helper",
            )
            nested[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                _require(
                    isinstance(target, ast.Name) and not isinstance(node, ast.AugAssign) and node.value is not None,
                    "FD pure-numpy operand: destructured, augmented or annotation-only local writes are unsupported",
                )
                assigned.setdefault(target.id, []).append(node.value)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Global, ast.Nonlocal)):
            raise AssertionError(
                "FD comparison contract: FD pure-numpy operand: loop, context, exception or scope-rebinding writes are unsupported"
            )
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            raise AssertionError("FD comparison contract: FD pure-numpy operand: comprehension scopes are unsupported")
    for name in nested:
        _require(name not in assigned, "FD pure-numpy operand: helper name is rebound")
    return assigned, nested


def _numeric_parametrize_names(function, imports):
    """Test parameters whose every pytest.mark.parametrize value is a real/complex number literal."""
    names = set()
    for decorator in function.decorator_list:
        if not (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "parametrize"
            and isinstance(decorator.func.value, ast.Attribute)
            and decorator.func.value.attr == "mark"
            and isinstance(decorator.func.value.value, ast.Name)
            and imports.get(decorator.func.value.value.id) == "pytest"
            and len(decorator.args) == 2
        ):
            continue
        try:
            argnames = ast.literal_eval(decorator.args[0])
            argvalues = ast.literal_eval(decorator.args[1])
        except ValueError:
            continue
        argnames = [argnames] if isinstance(argnames, str) else list(argnames)
        if len(argnames) == 1 and isinstance(argnames[0], str) and "," in argnames[0]:
            argnames = [name.strip() for name in argnames[0].split(",")]
        rows = [value if len(argnames) > 1 else (value,) for value in argvalues]

        def numeric(value):
            return type(value) in (int, float, complex)

        if rows and all(isinstance(row, (tuple, list)) and len(row) == len(argnames) for row in rows):
            for index, name in enumerate(argnames):
                if all(numeric(row[index]) for row in rows):
                    names.add(name)
    return names


class _PureNumpyOperand:
    """Fail-closed lexical proof that an operand is built only from NumPy fp64 arithmetic.

    Admitted leaves are real/complex number literals, numeric pytest parameters,
    NumPy attributes (``import numpy``), and the fp64 builtins ``abs``/``complex``/
    ``float``. Local names and nested pure helpers are followed through every
    write. Any import other than NumPy (in particular every ``jqmc`` symbol),
    fixture argument, non-fp64 NumPy dtype, ``astype``/``dtype`` or unsupported
    syntax rejects the operand.
    """

    def __init__(self, tree, function):
        self.imports, self.module_assigned, self.module_definitions = _module_bindings(tree)
        self.function = function
        self.assigned, self.nested = _local_bindings(function)
        self.parameters = {argument.arg for argument in function.args.args}
        self.numeric_parameters = _numeric_parametrize_names(function, self.imports)
        self.active = set()

    def _numpy_attribute(self, node, local):
        chain = []
        while isinstance(node, ast.Attribute):
            chain.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name) or any(
            node.id in names for names in (local, self.assigned, self.parameters, self.module_assigned, self.module_definitions)
        ):
            return None
        qualified = self.imports.get(node.id)
        if qualified is None or not (qualified == "numpy" or qualified.startswith("numpy.")):
            return None
        if any(name in _NON_FP64_NAMES for name in chain):
            raise AssertionError("FD comparison contract: FD pure-numpy operand: non-fp64 NumPy name")
        return ".".join([qualified, *reversed(chain)])

    def check(self, node, local=frozenset()):
        if isinstance(node, ast.Constant):
            _require(type(node.value) in (int, float, complex), "FD pure-numpy operand: non-numeric literal")
        elif isinstance(node, ast.Name):
            self._name(node.id, local)
        elif isinstance(node, (ast.BinOp,)):
            self.check(node.left, local)
            self.check(node.right, local)
        elif isinstance(node, ast.UnaryOp):
            self.check(node.operand, local)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                self.check(element, local)
        elif isinstance(node, ast.Subscript):
            self.check(node.value, local)
            self.check(node.slice, local)
        elif isinstance(node, ast.Attribute):
            if node.attr in _NON_FP64_NAMES:
                raise AssertionError("FD comparison contract: FD pure-numpy operand: non-fp64 attribute")
            if self._numpy_attribute(node, local) is None:
                _require(node.attr in _PASS_THROUGH_ATTRIBUTES, "FD pure-numpy operand: non-NumPy attribute")
                self.check(node.value, local)
        elif isinstance(node, ast.Call):
            _require(
                not any(keyword.arg in (None, "dtype") for keyword in node.keywords)
                and not any(isinstance(argument, ast.Starred) for argument in node.args),
                "FD pure-numpy operand: dtype or forwarded call arguments",
            )
            for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
                self.check(argument, local)
            function = node.func
            if isinstance(function, ast.Attribute) and self._numpy_attribute(function, local) is not None:
                return
            _require(isinstance(function, ast.Name), "FD pure-numpy operand: unsupported callee")
            if function.id in self.nested and function.id not in local:
                self._helper(self.nested[function.id])
                return
            _require(
                function.id in _FP64_BUILTINS
                and function.id not in self.imports
                and function.id not in self.assigned
                and function.id not in self.module_assigned
                and function.id not in self.module_definitions
                and function.id not in self.parameters
                and function.id not in local,
                "FD pure-numpy operand: callee is not NumPy or a pure local helper",
            )
        else:
            raise AssertionError(f"FD comparison contract: FD pure-numpy operand: unsupported syntax {type(node).__name__}")

    def _name(self, name, local):
        if name in local:
            return
        if name in self.assigned:
            if name in self.active:
                raise AssertionError("FD comparison contract: FD pure-numpy operand: recursive local definition")
            self.active.add(name)
            try:
                for value in self.assigned[name]:
                    self.check(value, local)
            finally:
                self.active.discard(name)
            return
        if name in self.parameters:
            _require(name in self.numeric_parameters, "FD pure-numpy operand: non-numeric or fixture test parameter")
            return
        if name in self.imports:
            raise AssertionError(
                f"FD comparison contract: FD pure-numpy operand: imported value {self.imports[name]} is not NumPy arithmetic"
            )
        if name in self.module_assigned and name not in self.active:
            self.active.add(name)
            try:
                for value in self.module_assigned[name]:
                    self.check(value, local)
            finally:
                self.active.discard(name)
            return
        raise AssertionError(f"FD comparison contract: FD pure-numpy operand: unbound name {name}")

    def _helper(self, helper):
        _require(
            not helper.decorator_list
            and not helper.args.vararg
            and not helper.args.kwarg
            and not helper.args.kwonlyargs
            and not helper.args.defaults
            and not helper.args.posonlyargs,
            "FD pure-numpy operand: helper signature is not a plain positional function",
        )
        local = frozenset(argument.arg for argument in helper.args.args)
        body = helper.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        _require(
            len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None,
            "FD pure-numpy operand: helper must be a single return expression",
        )
        self.check(body[0].value, local)


def _require_pure_numpy_site(root, site, pytest_node):
    """Prove both operands of one admitted comparison site are pure NumPy fp64."""
    from gdd.engine.tolerance_provenance import ToleranceSourceIndex

    path, *names = pytest_node.split("::")
    _require(len(names) == 1 and site.startswith(f"{pytest_node}#comparison-"), "FD central-pair site/node mismatch")
    tree = ast.parse((Path(root) / path).read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == names[0]]
    _require(len(functions) == 1, "FD central-pair test function is absent or duplicated")
    rows = [row for row in ToleranceSourceIndex(tree, path).comparisons() if row["site"] == site]
    _require(len(rows) == 1, "FD central-pair comparison site is absent")
    _require(rows[0]["comparator"].get("name") == PURE_NUMPY_COMPARATOR, "FD central-pair comparator is not assert_allclose")
    location = rows[0]["location"]
    calls = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Call) and (node.lineno, node.col_offset) == (location["line"], location["column"])
    ]
    _require(len(calls) == 1, "FD central-pair comparison is outside its test function")
    call = calls[0]
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    operands = list(call.args[:2]) + [keywords[name] for name in ("actual", "desired") if name in keywords]
    _require(len(operands) == 2, "FD central-pair comparison does not name exactly two operands")
    checker = _PureNumpyOperand(tree, functions[0])
    for operand in operands:
        checker.check(operand)


def _require_method_permissions(method, registry, root):
    """The FD anchor method permissions follow one rule, re-derived for every admission.

    ``[FIXED_POLICY]`` is the protected default. ``[FIXED_POLICY, CENTRAL_PAIR_POLICY]``
    is accepted only when every central-pair admission of this method is a
    pure-NumPy fp64 comparison: both operands are proved structurally from the
    test source, and because such operands are precision-lane invariant a
    mixed-lane admission must be the identical twin of a full-lane admission.
    No other policy can be added. The protected 52-case inner whitelist is not
    widened by this rule.
    """
    _require(set(method) == METHOD_FIELDS, "FD method fields changed")
    for field, value in METHOD_SEMANTICS.items():
        _require(method[field] == value, f"FD method {field} changed")
    permissions = method.get("comparison_policies")
    _require(permissions in ([FIXED_POLICY], [FIXED_POLICY, CENTRAL_PAIR_POLICY]), "default FD method permissions changed")
    if permissions == [FIXED_POLICY]:
        return
    policies = [row for row in registry["comparison_policies"] if row.get("id") == CENTRAL_PAIR_POLICY]
    _require(len(policies) == 1, "central-pair policy is missing or duplicated")
    policy = policies[0]
    _require(
        policy.get("method") == "allclose"
        and policy.get("tolerance-binding") == {"registry-kind": "symbol-pairs", "selector": "pair"},
        "central-pair policy semantics changed",
    )
    rows = [row for row in policy.get("allowed-comparison-bindings", ()) if row.get("verdict_method") == METHOD]
    _require(bool(rows), "FD method permits the central-pair policy without an admitted comparison")
    fields = {"site", "pytest_node", "verdict_method", "comparator", "precision", "pair"}
    lanes = {}
    for row in rows:
        _require(isinstance(row, dict) and set(row) == fields, "malformed FD central-pair admission")
        _require(row["comparator"] == PURE_NUMPY_COMPARATOR, "FD central-pair admission comparator is not assert_allclose")
        _require(row["precision"] in {"full", "mixed"}, "FD central-pair admission has an unknown lane")
        identity = (row["site"], row["pytest_node"])
        lane = lanes.setdefault(identity, {})
        _require(row["precision"] not in lane, "duplicate FD central-pair admission")
        lane[row["precision"]] = {key: value for key, value in row.items() if key != "precision"}
    for (site, pytest_node), lane in lanes.items():
        _require("full" in lane, "FD central-pair mixed-lane admission has no full-lane pure-fp64 twin")
        _require(
            lane.get("mixed", lane["full"]) == lane["full"],
            "FD central-pair mixed-lane admission differs from its lane-invariant full-lane twin",
        )
        try:
            _require_pure_numpy_site(root, site, pytest_node)
        except (OSError, SyntaxError, ValueError, KeyError) as exc:
            raise AssertionError(f"FD comparison contract: {site}: cannot prove pure-numpy operands: {exc}") from exc


def load_fd_contract(*, root=ROOT):
    """Expand the authored fixture selectors into the required exact FD universe.

    Intentionally uncached: a changed contract must not reuse a prior approval.
    The shared central-definition resolver has its normal process-local cache.
    """
    root = Path(root)
    document = load_module(root / "gdd/ssot/guarantees/derivatives.yaml")
    registry = _yaml(root / "gdd/ssot/verdicts.yaml")
    methods = [row for row in registry["verdict_methods"] if row["id"] == METHOD]
    _require(
        len(methods) == 1 and methods[0].get("case_comparison_contract") == CASE_AUTHORITY, "missing scoped method authority"
    )
    _require_method_permissions(methods[0], registry, root)
    guarantees = [row for row in document["guarantees"] if row["id"] == GUARANTEE]
    _require(len(guarantees) == 1 and guarantees[0]["verdict_methods"] == [METHOD], "wrong FD guarantee method")
    comparison = guarantees[0]["required_evidence"]["comparison"]
    _require(comparison.get("migration_state") == "case-bindings-enforced", "per-case enforcement not enabled")
    fixtures = [row for row in document["fixture_sets"] if row["id"] == FIXTURE]
    _require(len(fixtures) == 1, "FD fixture set is missing or duplicated")
    reviewed = fixtures[0]["reviewed_evidence"]
    cases = {}
    nodes = set()
    groups = set()
    for node in reviewed["nodes"]:
        _require(node["id"] not in groups and node["pytest_node"] not in nodes, "duplicate FD node or evidence ID")
        groups.add(node["id"])
        nodes.add(node["pytest_node"])
        _require(node["precision"] == "full", "FD cases must retain full precision")
        keys = node["id_dimensions"]
        _require(bool(keys) and keys == sorted(set(keys)), "case dimensions must be nonempty, sorted and unique")
        dimensions = node["fixture"]["dimensions"]
        _require(set(keys) <= set(dimensions), "unknown case dimension")
        values = [dimensions[key] if isinstance(dimensions[key], list) else [dimensions[key]] for key in keys]
        contracts = node.get("comparison_contract")
        _require(isinstance(contracts, list) and bool(contracts), f"{node['id']}: missing comparison bindings")
        for contract in contracts:
            _require(
                isinstance(contract, dict)
                and set(contract) == {"policy", "zones", "level", "tolerance_source", "compared_array_count"},
                f"{node['id']}: malformed comparison binding",
            )
            policy = contract["policy"]
            _require(policy in {*ZONE_POLICIES, FIXED_POLICY}, "unsupported FD comparison policy")
            _require(
                type(contract["compared_array_count"]) is int and contract["compared_array_count"] > 0,
                "empty comparison binding",
            )
            if policy in ZONE_POLICIES:
                _require(contract["level"] == ZONE_POLICIES[policy], "policy/tier mismatch")
                _require(contract["tolerance_source"] == "TOL-ZONE-MIN-001", "wrong zone tolerance source")
            else:
                _require(contract["level"] is None and contract["zones"] == [], "fixed FD policy has resolver arguments")
                _require(contract["tolerance_source"] == "TOL-FD-AUTODIFF-FULL-001", "wrong fixed tolerance source")
            try:
                resolve_requirement(_disposition(contract), "full", root=root)
            except (KeyError, TypeError, ValueError) as exc:
                raise AssertionError(f"FD comparison contract: {node['id']}: invalid central binding: {exc}") from exc
        combinations = list(itertools.product(*values))
        _require(bool(combinations), "FD node has no required selector combinations")
        for values in combinations:
            parameters = dict(zip(keys, values, strict=True))
            identity = _identity(node["id"], parameters)
            _require(identity not in cases, "duplicate concrete FD case")
            cases[identity] = {
                "node": node["pytest_node"],
                "evidence_id": node["id"],
                "parameters": parameters,
                "comparisons": contracts,
            }
    _require(bool(cases), "FD requirement universe must not be empty")
    return cases


def _literal_provenance(value):
    if isinstance(value, list):
        return {"kind": "list", "items": [_literal_provenance(item) for item in value]}
    return {"kind": "literal", "type": type(value).__name__, "value": repr(value)}


def _central_source_matches(description, expected, index):
    if expected["policy"] == FIXED_POLICY:
        name = ("ATOL", "RTOL")[index] + "_FD_VS_ANALYTIC_TIGHT"
        return description == {"kind": "central-reference", "name": f"jqmc._setting.{name}"}
    for resolver, zones in (("get_tolerance_min", expected["zones"]), ("get_tolerance", None)):
        if zones is None:
            if len(expected["zones"]) != 1:
                continue
            zones = expected["zones"][0]
        call = {
            "kind": "central-resolver-call",
            "function": {"kind": "central-reference", "name": f"jqmc._precision.{resolver}"},
            "args": [_literal_provenance(zones), _literal_provenance(expected["level"])],
            "keywords": [],
        }
        if description == {"kind": "item", "value": call, "index": index}:
            return True
    return False


def fd_source_contract_errors(*, root=ROOT):
    """Check the reviewed adapter calls without importing pytest, JAX or jqmc.

    Unsupported source forms fail closed. The runtime check remains necessary:
    source provenance alone cannot attest the actual invocation's tolerances.
    """
    from gdd.engine.tolerance_provenance import ToleranceSourceIndex

    root = Path(root)
    errors = []
    try:
        cases = load_fd_contract(root=root)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return [f"FD comparison contract: {exc}"]
    nodes = {case["node"]: case for case in cases.values()}
    for node_id, case in nodes.items():
        try:
            path, *names = node_id.split("::")
            tree = ast.parse((root / path).read_text(encoding="utf-8"))
            scope = tree
            for name in names:
                matches = [
                    child for child in scope.body if isinstance(child, (ast.FunctionDef, ast.ClassDef)) and child.name == name
                ]
                _require(len(matches) == 1, "source node is absent or duplicated")
                scope = matches[0]
            index = ToleranceSourceIndex(tree, path)
            calls = [
                call
                for call in ast.walk(scope)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "record_fd_case"
            ]
            _require(len(calls) == 1, "expected exactly one reviewed FD adapter call")
            call = calls[0]
            _require(
                index.expression(call.func)
                == {"kind": "imported-reference", "name": "tests.test_jqmc_kernel._gdd_fd_adapters.record_fd_case"},
                "FD adapter import is shadowed or unrecognized",
            )
            _require(
                len(call.args) == 1 and ast.literal_eval(call.args[0]) == case["evidence_id"], "wrong FD evidence ID in source"
            )
            kwargs = {keyword.arg: keyword.value for keyword in call.keywords}
            _require(None not in kwargs and len(kwargs) == len(call.keywords), "ambiguous FD keyword arguments")
            _require(ast.literal_eval(kwargs["pytest_node_id"]) == node_id, "wrong pytest node in source")
            primary = case["comparisons"][0]
            policy = (
                ast.literal_eval(kwargs["comparison_policy_id"])
                if "comparison_policy_id" in kwargs
                else f"CMP-ZONE-ALLCLOSE-{ast.literal_eval(kwargs['level']).upper()}-001"
            )
            _require(policy == primary["policy"], "source comparison policy differs")
            comparisons = [(kwargs["compared_pairs"], kwargs["atol"], kwargs["rtol"], kwargs["zones"], kwargs["level"])]
            if "additional_comparisons" in kwargs:
                extra = kwargs["additional_comparisons"]
                _require(isinstance(extra, (ast.Tuple, ast.List)), "unresolved additional comparisons")
                for row in extra.elts:
                    _require(isinstance(row, (ast.Tuple, ast.List)) and len(row.elts) == 5, "malformed additional comparison")
                    comparisons.append(tuple(row.elts))
            _require(len(comparisons) == len(case["comparisons"]), "source comparison count differs")
            for (pairs, atol, rtol, zones, level), expected in zip(comparisons, case["comparisons"], strict=True):
                _require(
                    isinstance(pairs, (ast.Tuple, ast.List)) and len(pairs.elts) == expected["compared_array_count"],
                    "source array-pair count differs",
                )
                _require(
                    all(isinstance(pair, (ast.Tuple, ast.List)) and len(pair.elts) == 2 for pair in pairs.elts),
                    "malformed array pair",
                )
                _require(
                    list(ast.literal_eval(zones)) == expected["zones"] and ast.literal_eval(level) == expected["level"],
                    "source zones/tier differ",
                )
                for part, value in enumerate((atol, rtol)):
                    _require(
                        _central_source_matches(index.expression(value), expected, part),
                        "source tolerance does not match the declared central binding",
                    )
        except (AssertionError, KeyError, TypeError, ValueError, OSError, SyntaxError) as exc:
            errors.append(f"{node_id}: FD comparison contract: {exc}")
    return errors


def validate_fd_call(evidence_id, pytest_node_id, fixture_parameters, precision_mode, comparisons, *, root=ROOT):
    """Reject wrong cases, methods, zones, tiers or actual numerical budgets."""
    _require(precision_mode == "full", "FD comparison executed outside full precision")
    cases = load_fd_contract(root=root)
    identity = _identity(evidence_id, fixture_parameters)
    _require(identity in cases, "unregistered FD case")
    case = cases[identity]
    _require(case["node"] == pytest_node_id, "wrong pytest node for FD case")
    _require(
        json.dumps(fixture_parameters, sort_keys=True) == json.dumps(case["parameters"], sort_keys=True),
        "FD selector values or types differ",
    )
    _require(len(comparisons) == len(case["comparisons"]), "missing or extra FD comparison")
    for actual, expected in zip(comparisons, case["comparisons"], strict=True):
        _require(type(actual["compared_array_count"]) is int, "invalid compared-array count")
        for key in ("policy", "zones", "level", "compared_array_count"):
            value = list(actual[key]) if key == "zones" else actual[key]
            _require(value == expected[key], f"FD comparison {key} differs from the case contract")
        budget = resolve_requirement(_disposition(expected), "full", root=Path(root))
        for name in ("atol", "rtol"):
            _require(
                type(actual[name]) in (int, float) and actual[name] == budget[name],
                f"FD actual {name} differs from its central binding",
            )
    return case


def validate_fd_receipt(receipt, *, root=ROOT):
    """Replay finite FD comparison requirements after transport, not merely IDs.

    Zone/tier arguments are checked on the live adapter call. The transported
    receipt stores policy, resolved budgets and counts; replay checks precisely
    those fields. The caller separately requires the xla-full transport lane.
    """
    from gdd.engine.evidence import ComparisonReceipt, ObservationReceipt

    _require(receipt.guarantee_id == GUARANTEE, "wrong FD guarantee")
    cases = load_fd_contract(root=root)
    _require(receipt.obligation_id in cases, "unregistered FD receipt")
    case = cases[receipt.obligation_id]
    _require(receipt.pytest_node_id == case["node"], "wrong pytest node in FD receipt")
    checks = (receipt.comparison, *receipt.additional_checks)
    expected_checks = case["comparisons"]
    _require(len(checks) == len(expected_checks) + 1, "missing or extra FD receipt comparison/witness")
    for index, expected in enumerate(expected_checks):
        actual = checks[index]
        _require(type(actual) is ComparisonReceipt, "FD receipt comparison has wrong type")
        actual.validate()
        suffix = "" if index == 0 else f"::additional-{index - 1}"
        _require(actual.binding_id == f"COMPARISON::{receipt.obligation_id}{suffix}", "wrong FD comparison identity/order")
        _require(actual.policy_id == expected["policy"], "wrong policy in FD receipt")
        _require(
            type(actual.compared_array_count) is int and actual.compared_array_count == expected["compared_array_count"],
            "wrong FD compared-array count",
        )
        budget = resolve_requirement(_disposition(expected), "full", root=Path(root))
        _require(
            actual.resolved_atol == budget["atol"] and actual.resolved_rtol == budget["rtol"],
            "wrong central tolerance in FD receipt",
        )
    witness = checks[-1]
    _require(type(witness) is ObservationReceipt and witness.policy_id == "OBS-FD-ROUTE-001", "missing FD route witness")
    _require(witness.binding_id == f"FD-WITNESS::{receipt.obligation_id}", "wrong FD witness identity")
    witness.validate()
    _require(witness.completed_observations == tuple(sorted(FD_OBSERVATIONS)), "FD route witness lost required observations")
