"""Evaluation-condition-number (kappa) screening contracts.

The cond(G) electron-configuration contract stays exactly as it was; this
module adds a second, explicitly named conditioning subject:

``polynomial-evaluation-kappa``
    For a quantity evaluated as a sum of terms ``S(x) = sum_i t_i(x)`` the
    evaluation condition number is ``kappa(x) = sum_i |t_i(x)| / |S(x)|``.
    A test may accept only draws with ``kappa <= kappa_max`` where the cap
    comes from the central resolver ``jqmc._precision.get_evaluation_kappa_max``
    (``rtol / ((4 l + 4) eps)``, see ``jqmc._setting.evaluation_kappa_max``).

The derived binding carries a separate ``evaluation_conditioning`` mapping; a
kappa can never be declared through, observed as, or completed by a cond(G)
field. Every validator here fails closed: unknown keys, a missing cap, a cap
without an upper comparator, a non-central resolver, or a subject mismatch are
errors, not defaults.
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
import math
import re
from pathlib import Path
from typing import Any, Mapping

# Static gates and the case-binding derivation import this module in
# numerical-import-free processes, so numpy and the replay language are
# imported lazily, only by the runtime/completion functions that need them.

ROOT = Path(__file__).resolve().parents[2]

SUBJECT = "polynomial-evaluation-kappa"
DEFINITION = "sum-of-absolute-evaluated-terms-over-absolute-value"
OPERATOR = "asserted-runtime-constraint"
CAP_RESOLVER = "jqmc._precision.get_evaluation_kappa_max"
CAP_FUNCTION = CAP_RESOLVER.rsplit(".", 1)[-1]
CAP_MODULE = CAP_RESOLVER.rsplit(".", 1)[0]
LEVELS = ("strict", "medium")
REALIZED_OBSERVATION = {
    "value": "runtime-receipt-required",
    "cap": "runtime-receipt-required",
    "accepted_draw": "runtime-receipt-required",
}
CONTRACT_KEYS = frozenset({"subject", "definition", "operator", "metric_symbols", "cap", "constraints", "realized_observation"})
CAP_KEYS = frozenset({"symbol", "resolver", "zone", "level", "degree"})
_METRIC = re.compile(r"(?:^|_)evaluation_kappa(?:$|_)")


def is_evaluation_kappa_name(name: str) -> bool:
    """A declared kappa metric is named ``evaluation_kappa`` or ``*_evaluation_kappa_*``."""
    return _METRIC.search(name.lstrip("_").lower()) is not None


def _comparison_parts(expression: str) -> tuple[str, str, str]:
    """Return ``(metric, comparator, cap)`` of ``metric < cap`` / ``metric <= cap``."""
    tree = ast.parse(expression, mode="eval").body
    if not isinstance(tree, ast.Compare) or len(tree.ops) != 1 or len(tree.comparators) != 1:
        raise ValueError(
            "an evaluation-kappa constraint must be one explicit comparison against its cap "
            "(the cap requires an explicit upper comparator)"
        )
    operation = tree.ops[0]
    if not isinstance(operation, (ast.Lt, ast.LtE)):
        raise ValueError("an evaluation-kappa cap requires an explicit upper comparator (< or <=)")
    if not isinstance(tree.left, ast.Name) or not isinstance(tree.comparators[0], ast.Name):
        raise ValueError("an evaluation-kappa constraint compares one metric name with one cap name")
    return tree.left.id, "less-than" if isinstance(operation, ast.Lt) else "less-than-or-equal", tree.comparators[0].id


def contract_errors(contract: Any) -> list[str]:
    """Validate a derived ``evaluation_conditioning`` mapping; never infer a default."""
    errors: list[str] = []
    if not isinstance(contract, dict):
        return ["evaluation conditioning must be an explicit mapping"]
    if set(contract) != CONTRACT_KEYS:
        return [f"evaluation conditioning keys must be exactly {sorted(CONTRACT_KEYS)}"]
    if contract["subject"] != SUBJECT:
        errors.append(f"evaluation conditioning subject must be {SUBJECT!r}, not {contract['subject']!r}")
    if contract["definition"] != DEFINITION:
        errors.append("evaluation conditioning definition differs from the reviewed kappa definition")
    if contract["operator"] != OPERATOR:
        errors.append("evaluation conditioning must be an asserted runtime constraint")
    if contract["realized_observation"] != REALIZED_OBSERVATION:
        errors.append("evaluation conditioning must require runtime kappa, cap, and accepted-draw observations")
    metrics = contract["metric_symbols"]
    if (
        not isinstance(metrics, list)
        or not metrics
        or len(set(metrics)) != len(metrics)
        or any(not isinstance(name, str) or not name.isidentifier() or not is_evaluation_kappa_name(name) for name in metrics)
    ):
        errors.append("evaluation conditioning requires distinct declared evaluation_kappa metric names")
        metrics = []
    cap = contract["cap"]
    cap_symbol = None
    if not isinstance(cap, dict) or set(cap) != CAP_KEYS:
        errors.append(f"evaluation conditioning requires a central cap with exactly {sorted(CAP_KEYS)}")
    else:
        cap_symbol = cap["symbol"]
        if not isinstance(cap_symbol, str) or not cap_symbol.isidentifier() or is_evaluation_kappa_name(cap_symbol):
            errors.append("evaluation-kappa cap symbol must be a distinct identifier")
        if cap["resolver"] != CAP_RESOLVER:
            errors.append(f"evaluation-kappa cap must come from {CAP_RESOLVER}")
        if not isinstance(cap["zone"], str) or not cap["zone"]:
            errors.append("evaluation-kappa cap requires its precision zone")
        if cap["level"] not in LEVELS:
            errors.append("evaluation-kappa cap requires a strict or medium level")
        degree = cap["degree"]
        valid_degree = isinstance(degree, dict) and (
            (set(degree) == {"selector"} and isinstance(degree["selector"], str) and degree["selector"].isidentifier())
            or (set(degree) == {"literal"} and type(degree["literal"]) is int and degree["literal"] >= 0)
        )
        if not valid_degree:
            errors.append("evaluation-kappa cap degree must be a parameter selector or a nonnegative integer literal")
    constraints = contract["constraints"]
    if not isinstance(constraints, list) or not constraints or len(set(constraints)) != len(constraints):
        errors.append("evaluation conditioning requires concrete constraint assertions")
        constraints = []
    constrained_metrics = set()
    for expression in constraints:
        try:
            if not isinstance(expression, str):
                raise ValueError("constraint must be source text")
            metric, _comparison, bound = _comparison_parts(expression)
            if metric not in metrics:
                raise ValueError(f"constraint metric {metric!r} is not a declared evaluation_kappa metric")
            if bound != cap_symbol:
                raise ValueError("constraint must compare against the declared central cap symbol")
            constrained_metrics.add(metric)
        except (ValueError, SyntaxError) as exc:
            errors.append(f"unsupported evaluation-kappa constraint {expression!r}: {exc}")
    if metrics and constrained_metrics != set(metrics):
        errors.append("every declared evaluation_kappa metric needs a cap constraint")
    return errors


def binding_subject_errors(electron: Any) -> list[str]:
    """Reject a kappa declared through the cond(G) contract or a subject mixed into it."""
    if not isinstance(electron, dict):
        return []
    from gdd.engine.condition_checks import condition_symbols

    errors = []
    condition = electron.get("cond_G")
    if isinstance(condition, dict):
        if condition.get("subject") != "geminal-matrix-G":
            errors.append("cond(G) contract must keep the geminal-matrix-G subject")
        for expression in condition.get("constraints", ()):
            try:
                symbols = condition_symbols(expression)
            except (ValueError, SyntaxError):
                continue
            if any(is_evaluation_kappa_name(symbol.split(".")[-1]) for symbol in symbols):
                errors.append("evaluation kappa cannot be declared through the cond(G) contract")
    for expression in electron.get("conditioning_assertions", ()) or ():
        try:
            symbols = condition_symbols(expression)
        except (ValueError, SyntaxError):
            continue
        if any(is_evaluation_kappa_name(symbol.split(".")[-1]) for symbol in symbols):
            errors.append("evaluation kappa cannot be listed among cond(G) conditioning assertions")
    if "evaluation_conditioning" in electron:
        if electron.get("mode") != "required":
            errors.append("evaluation conditioning requires realized evaluation positions")
        errors.extend(contract_errors(electron["evaluation_conditioning"]))
    return errors


def _central_cap_inputs(zone: str, level: str, precision: str, root: Path) -> tuple[float, float, int, int]:
    import numpy as np

    from gdd.engine.comparison_requirements import _definitions, _static_value

    if precision not in {"full", "mixed"}:
        raise ValueError("evaluation-kappa cap requires full or mixed precision")
    settings = _definitions(root / "jqmc/_setting.py")
    layout = _definitions(root / "jqmc/_precision.py")
    dtypes = _static_value(layout[f"_{precision.upper()}_PRECISION"], layout)
    if zone not in dtypes:
        raise ValueError(f"unknown precision zone for evaluation-kappa cap: {zone}")
    table = _static_value(settings["_TOLERANCE"], settings)
    pair = table[level][dtypes[zone]]
    rtol = pair[1]
    slope = _static_value(settings["EVALUATION_KAPPA_ROUNDING_SLOPE"], settings)
    offset = _static_value(settings["EVALUATION_KAPPA_ROUNDING_OFFSET"], settings)
    if type(slope) is not int or type(offset) is not int or slope < 1 or offset < 1:
        raise ValueError("central evaluation-kappa rounding coefficients must be positive integers")
    if type(rtol) not in (int, float) or not math.isfinite(rtol) or rtol <= 0:
        raise ValueError("central comparison rtol must be finite and positive")
    eps = float(np.finfo(np.dtype(dtypes[zone])).eps)
    return float(rtol), eps, slope, offset


def central_cap(cap: Mapping[str, Any], precision: str, selector: Mapping[str, Any] | None, *, root: Path = ROOT) -> float:
    """Recompute the cap from central definitions (static read, no production import)."""
    degree_spec = cap["degree"]
    if "selector" in degree_spec:
        if not isinstance(selector, Mapping) or degree_spec["selector"] not in selector:
            raise ValueError("evaluation-kappa cap degree selector is absent from the concrete case")
        degree = selector[degree_spec["selector"]]
    else:
        degree = degree_spec["literal"]
    if type(degree) is not int or degree < 0:
        raise ValueError("evaluation-kappa cap degree must resolve to a nonnegative integer")
    rtol, eps, slope, offset = _central_cap_inputs(cap["zone"], cap["level"], precision, root)
    # Same expression and operand order as jqmc._setting.evaluation_kappa_max.
    return float(rtol) / ((slope * int(degree) + offset) * float(eps))


def receipt_errors(
    contract: Any,
    rows: list[Mapping[str, Any]],
    *,
    precision: str,
    selector: Mapping[str, Any] | None,
    observations: list[Mapping[str, Any]],
    root: Path = ROOT,
) -> list[str]:
    """Join runtime kappa receipts to their derived contract.

    Each declared constraint needs an executed assertion site whose captured
    operands replay true, a captured cap equal to the central value for this
    precision and case, and at least one recorded kappa observation.
    """
    from gdd.engine.condition_checks import condition_input_errors

    realized = [row.get("realized_fixture", {}).get("evaluation_conditioning") for row in rows]
    if not isinstance(contract, dict):
        if any(value is not None for value in realized):
            return ["evaluation-kappa receipt has no evaluation-conditioning contract"]
        return []
    errors = [f"evaluation conditioning: {error}" for error in contract_errors(contract)]
    if errors:
        return errors
    try:
        expected_cap = central_cap(contract["cap"], precision, selector, root=root)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"evaluation conditioning: cannot resolve the central cap: {exc}"]
    constraints = set(contract["constraints"])
    sites = {row["id"]: row["expression"] for row in observations if row.get("expression") in constraints}
    if {sites[key] for key in sites} != constraints:
        errors.append("evaluation-kappa constraints lack typed assertion sites")
    cap_symbol = contract["cap"]["symbol"]
    witnessed = set()
    observed_kappa = False
    for row, receipt in zip(rows, realized, strict=True):
        if receipt is None:
            continue
        if not isinstance(receipt, dict) or set(receipt) != {"kappa", "checks"}:
            errors.append("malformed evaluation-kappa receipt")
            continue
        if isinstance(receipt["kappa"], dict) and receipt["kappa"]:
            observed_kappa = True
        executed = row.get("typed_verdict", {}).get("executed_observation_ids", ())
        checks = receipt["checks"]
        if not isinstance(checks, list):
            errors.append("evaluation-kappa checks must be an explicit list")
            continue
        for check in checks:
            if not isinstance(check, dict) or set(check) != {"observation_id", "expression", "inputs"}:
                errors.append(
                    "evaluation-kappa check lacks its captured inputs: "
                    f"{check.get('capture_error') if isinstance(check, dict) else check!r}"
                )
                continue
            observation_id, expression = check["observation_id"], check["expression"]
            if sites.get(observation_id) != expression or observation_id not in executed:
                errors.append("evaluation-kappa check lacks its matching executed assertion site")
                continue
            failures = condition_input_errors(expression, check["inputs"])
            cap_input = check["inputs"].get(cap_symbol) if isinstance(check["inputs"], dict) else None
            if (
                not isinstance(cap_input, dict)
                or cap_input.get("shape") != []
                or type(cap_input.get("real")) not in (int, float)
                or float(cap_input["real"]) != expected_cap
            ):
                failures.append(f"captured cap differs from the central value {expected_cap!r}")
            errors.extend(f"{observation_id}: {failure}" for failure in failures)
            if not failures:
                witnessed.add(expression)
    if not observed_kappa:
        errors.append("evaluation-kappa contract has no runtime kappa observation")
    missing = constraints - witnessed
    if missing:
        errors.append(f"evaluation-kappa assertions lack valid runtime inputs: {sorted(missing)}")
    return errors
