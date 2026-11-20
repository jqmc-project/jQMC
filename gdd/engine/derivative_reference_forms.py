"""Partition existing GFMC obligations by the reference actually declared at their source.

This reporting classification does not replace primary proofs, change case IDs,
remove obligations or permit a new oracle. Runtime certification is separate.
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
from collections import Counter


def _literal_selector(expression):
    """Only explicit equality guards are supported; unknown conditions are not guessed."""
    if isinstance(expression, ast.BoolOp) and isinstance(expression.op, ast.And):
        result = {}
        for value in expression.values:
            selector = _literal_selector(value)
            if selector is None or result.keys() & selector.keys():
                return None
            result.update(selector)
        return result
    if isinstance(expression, ast.Compare) and len(expression.ops) == 1 and isinstance(expression.ops[0], ast.Eq):
        left, right = expression.left, expression.comparators[0]
        if isinstance(left, ast.Constant):
            left, right = right, left
        if isinstance(left, ast.Name) and isinstance(right, ast.Constant):
            return {left.id: right.value}
    return None


# Each conditioned reference names the derivative target it replaces and the reference form it
# uses.  A spec without these keys is the original strain component split (ECP + DTN dltmove).
_ORACLE_BY_TARGET = {"strain": "_jit_strain_oracle", "positions": "_jit_grad_pos"}
_CONDITIONED_FORMS = ("conditioned-component-ad", "rank-one-ecp-ratio-ad", "rank-one-ecp-ratio-fused-base-ad")
# Source keywords each rank-one form must pass to the energy it differentiates.
_RANK_ONE_KEYWORDS = {
    "rank-one-ecp-ratio-ad": ("rank_one_ratios",),
    "rank-one-ecp-ratio-fused-base-ad": ("rank_one_ratios", "fused_base_inverse"),
}


def _conditioned_target(spec):
    target = spec.get("target", "strain")
    if target not in _ORACLE_BY_TARGET:
        raise ValueError(f"conditioned reference has an unknown target: {target!r}")
    return target


def _conditioned_form(spec):
    form = spec.get("reference-form", "conditioned-component-ad")
    if form not in _CONDITIONED_FORMS:
        raise ValueError(f"conditioned reference has an unknown reference form: {form!r}")
    return form


def measured_reference_forms(gfmc, *, source_tree=None, payload=None):
    from gdd.engine import derivative_requirements as requirements

    tree = requirements._gfmc_source_tree() if source_tree is None else source_tree
    rows = requirements.gfmc_matrix_payload() if payload is None else payload
    affine = {
        function.name
        for function in tree.body
        if isinstance(function, ast.FunctionDef)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_jit_molecular_affine_stress_oracle"
            for node in ast.walk(function)
        )
    }
    prefix = "tests.test_jqmc_kernel.test_gfmc_force_analytic."
    symbols = gfmc["molecular-all-electron-stress-oracle"]["pytest-symbols"]
    if not isinstance(symbols, list) or len(set(symbols)) != len(symbols):
        raise ValueError("affine reference symbols must be a unique list")
    if any(not isinstance(name, str) or not name.startswith(prefix) for name in symbols):
        raise ValueError("affine reference symbols must name the GFMC evidence module")
    if {name.removeprefix(prefix) for name in symbols} != affine:
        raise ValueError("affine reference symbols differ from the source calls")
    conditioned = gfmc["conditioned-direct-ad-cases"]
    functions = {function.name: function for function in tree.body if isinstance(function, ast.FunctionDef)}
    for spec in conditioned.values():
        name = spec["pytest-symbol"]
        selector = spec["selector"]
        if not isinstance(name, str) or not name.startswith(prefix) or not isinstance(selector, dict) or not selector:
            raise ValueError("conditioned reference requires an explicit source and nonempty selector")
        target = _conditioned_target(spec)
        reference_form = _conditioned_form(spec)
        function = functions[name.removeprefix(prefix)]
        branches = [
            node for node in ast.walk(function) if isinstance(node, ast.If) and _literal_selector(node.test) == selector
        ]
        if len(branches) != 1 or not any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == _ORACLE_BY_TARGET[target]
            for statement in branches[0].body
            for node in ast.walk(statement)
        ):
            raise ValueError("conditioned reference selector does not match its source AD branch")
        for keyword in _RANK_ONE_KEYWORDS.get(reference_form, ()):
            if not any(
                isinstance(node, ast.keyword) and node.arg == keyword and getattr(node.value, "value", None) is True
                for statement in branches[0].body
                for node in ast.walk(statement)
            ):
                raise ValueError(f"{reference_form} reference branch does not pass {keyword}=True to its energy")
    counts, hits, affine_hits = Counter(), Counter(), set()
    for row in rows:
        matches = [
            name
            for name, spec in conditioned.items()
            if spec["pytest-symbol"] == prefix + row["function"]
            and row["target"] == _conditioned_target(spec)
            and spec["selector"]
            and all(row["parameters"].get(k) == v for k, v in spec["selector"].items())
        ]
        if len(matches) > 1:
            raise ValueError("one GFMC reference case matches multiple conditioned rules")
        if row["proof"] == "identity":
            if matches or row["function"] in affine:
                raise ValueError("algorithm identity overlaps an AD reference rule")
            form = "algorithm-identity"
        elif row["proof"] != "direct_ad":
            raise ValueError("unknown GFMC primary proof")
        elif row["function"] in affine:
            if row["target"] != "strain" or matches:
                raise ValueError("affine reference must be a distinct strain case")
            affine_hits.add(row["function"])
            form = "position-ad-affine-chain-rule"
        elif matches:
            hits.update(matches)
            form = _conditioned_form(conditioned[matches[0]])
        else:
            form = "target-coordinate-ad"
        counts[row["family"], row["target"], form] += 1
    if set(hits) != set(conditioned) or affine_hits != affine:
        raise ValueError("a declared special reference route has no obligation")
    return [
        dict(family=family, target=target, reference_form=form, cases=cases)
        for (family, target, form), cases in sorted(counts.items())
    ]


def reference_form_errors(gfmc, *, source_tree=None, payload=None):
    try:
        measured = measured_reference_forms(gfmc, source_tree=source_tree, payload=payload)
        if sum(row["cases"] for row in measured) != gfmc["expected-concrete-obligations"]:
            raise ValueError("reference form partition differs from the required total")
        categories = Counter()
        for row in measured:
            category = "gfmc-identity" if row["reference_form"] == "algorithm-identity" else "gfmc-direct"
            categories[category] += row["cases"]
        if dict(categories) != gfmc["by_category"]:
            raise ValueError("reference form partition differs from the required primary proof categories")
    except (KeyError, TypeError, ValueError, OSError, SyntaxError) as exc:
        return [f"GFMC reference-form contract: {exc}"]
    return []
