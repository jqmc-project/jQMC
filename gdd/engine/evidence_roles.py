"""Reviewed, exhaustive evidence roles inside a single owned pytest node.

Roles classify assertions; they do not turn a source description or a passing
assertion into tolerance-policy certification. Unsupported policies are explicitly
listed as source-only. Undeclared nodes retain their existing node-wide method.
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

from functools import cache
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
ROLE_PATH = "gdd/ssot/evidence_roles.yaml"
OBSERVATION_FIELDS = ("kind", "expression", "actual", "reference", "tolerance_arguments")
COMPARISON_KINDS = {"exact-comparison", "tolerance-comparison", "numerical-bound"}
TOLERANCE_METHODS = {
    "VM-NUM-COMPLEX-ECP-FD-001",
    "VM-MATH-COMPLEX-ECP-CONSUMPTION-001",
    "VM-MATH-LM-GENERALIZED-EIGEN-001",
    "VM-IMPL-TREXIO-STORAGE-001",
    "VM-MATH-FIXED-SAMPLE-001",
    "VM-MATH-NATIVE-GAUSSIAN-001",
    "VM-MATH-AD-ORACLE-001",
    "VM-NUM-MIXED-FIDELITY-001",
    "VM-NUM-FD-ANCHOR-001",
    "VM-NUM-FD-SOURCE-ZONE-FULL-001",
}
COMPARISON_METHODS = {
    "VM-NUM-REVIEWED-REGRESSION-001",
    "VM-MATH-REFERENCE-001",
    "VM-MATH-REVIEWED-IDENTITY-001",
    "VM-IMPL-BACKEND-PARITY-001",
    "VM-IMPL-ROUTE-PARITY-001",
    "VM-IMPL-GFMC-SAME-INPUT-001",
    "VM-PHY-INVARIANT-001",
    "VM-PHY-REFERENCE-001",
    "VM-PHY-CONVERGENCE-001",
    "VM-MATH-SOURCE-BUDGET-IDENTITY-001",
    "VM-MATH-SOURCE-ARRAY-IDENTITY-001",
    "VM-MATH-SOURCE-SPACING-IDENTITY-001",
}


def observation_contract(observation):
    return {key: observation[key] for key in OBSERVATION_FIELDS if key in observation}


@cache
def load_roles(root: Path = ROOT):
    path = root / ROLE_PATH
    if not path.is_file() and root.resolve() != ROOT.resolve():
        return {}  # Minimal isolated contract fixtures have no role registry.
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("nodes"), dict):
        raise ValueError("invalid evidence-role registry")
    return data["nodes"]


def role_errors(declaration, observations, checks, methods):
    """Reject omissions, duplicate assignments, source drift and unbound checks."""
    errors = []
    if not isinstance(declaration, dict) or set(declaration) != {"guarantee", "registration_method", "observations", "roles"}:
        return ["invalid evidence-role node declaration"]
    source = {row["id"]: observation_contract(row) for row in observations}
    if declaration["observations"] != source or len(source) != len(observations):
        errors.append("evidence-role observation universe or source semantics changed")
    roles = declaration["roles"]
    if not isinstance(roles, dict) or not roles:
        return errors + ["evidence roles must be a nonempty mapping"]
    if not checks:
        # Mirrors the runtime rule: a declared finite node never falls back to
        # the unmigrated (check-free) acceptance path.
        errors.append("declared evidence roles require their executable checks; static fallback is forbidden")
    checks = checks or {}
    assigned, bound, source_only = [], [], []
    sites = {
        comparison["site"]: row["id"]
        for row in observations
        for comparison in row.get("tolerance_provenance", {}).get("comparisons", ())
    }
    for name, role in roles.items():
        if (
            not isinstance(name, str)
            or not isinstance(role, dict)
            or set(role) != {"method", "meaning", "observations", "checks", "source_only"}
        ):
            errors.append("invalid or unknown evidence-role fields")
            continue
        if role["method"] not in methods or not isinstance(role["meaning"], str) or not role["meaning"].strip():
            errors.append(f"{name}: unknown method or empty role meaning")
        if not isinstance(role["observations"], list) or not role["observations"]:
            errors.append(f"{name}: role requires explicit observations")
            continue
        assigned.extend(role["observations"])
        kinds = {source[identity]["kind"] for identity in role["observations"] if identity in source}
        if role["method"] in TOLERANCE_METHODS and "tolerance-comparison" not in kinds:
            errors.append(f"{name}: this role's method requires its own direct tolerance comparison")
        if role["method"] in COMPARISON_METHODS and not kinds.intersection(COMPARISON_KINDS):
            errors.append(f"{name}: this role's method requires its own direct comparison")
        if not isinstance(role["checks"], list) or not isinstance(role["source_only"], dict):
            errors.append(f"{name}: invalid check or source-only declaration")
            continue
        if role["method"] in TOLERANCE_METHODS | COMPARISON_METHODS and not role["checks"]:
            # A comparison role whose bound has no recognized site would otherwise
            # pass with neither a check nor a source-only limitation.
            errors.append(f"{name}: this role's comparison method requires at least one policy-bound executable check")
        bound.extend(role["checks"])
        source_only.extend(role["source_only"])
        for check_id in role["checks"]:
            check = checks.get(check_id)
            if not check or sites.get(check["site"]) not in role["observations"]:
                errors.append(f"{name}: comparison check missing or assigned to another role")
        method = methods.get(role["method"], {})
        allowed = method.get("allowed-comparison-sites")
        if allowed is not None and any(
            site not in allowed for site, observation in sites.items() if observation in role["observations"]
        ):
            errors.append(f"{name}: comparison is outside the verdict method's reviewed sites")
        for site, reason in role["source_only"].items():
            if sites.get(site) not in role["observations"] or not isinstance(reason, str) or not reason.strip():
                errors.append(f"{name}: source-only comparison needs its own observation and an explicit limitation")
    if len(assigned) != len(set(assigned)) or set(assigned) != set(source):
        errors.append("every observation must belong to exactly one evidence role; no fallback is allowed")
    if len(bound) != len(set(bound)) or set(bound) != set(checks):
        errors.append("every executable check must belong to exactly one evidence role")
    bound_sites = [check["site"] for check in checks.values()]
    if len(source_only) != len(set(source_only)) or set(source_only) & set(bound_sites):
        errors.append("source-only and policy-bound comparisons must be disjoint and unique")
    if set(source_only) | set(bound_sites) != set(sites):
        errors.append("every comparison needs a policy-bound check or an explicit source-only limitation")
    return errors


def comparison_method(pytest_node, check_id, default, *, root=ROOT):
    declaration = load_roles(root).get(pytest_node)
    if declaration is None:
        return default
    matches = [role["method"] for role in declaration["roles"].values() if check_id in role["checks"]]
    if len(matches) != 1:
        raise ValueError("declared evidence node has no unique comparison role; node-wide fallback is forbidden")
    return matches[0]
