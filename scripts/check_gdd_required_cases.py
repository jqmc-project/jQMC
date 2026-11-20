#!/usr/bin/env python3
"""Check explicit parameter, precision, execution-profile and canonical requirements.

This compares independently authored required tuples with current source-derived
cases and, when supplied, real pytest collection.
No counts, snapshot digests, approval tables, or whole-test tracing are involved.
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
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gdd.engine.canonical_contract import canonical_requirement_errors
from gdd.engine.derivative_closure import collection_closure_errors
from gdd.engine.guarantee_store import load_module
from gdd.engine.ledger import iter_registered_nodes, load_ledger
from gdd.engine.parameter_requirements import parameter_case_errors


def selector_key(selector):
    """Keep tuple values and their types together, not unions of separate axes."""
    if not isinstance(selector, dict):
        raise ValueError("collection selector must be a mapping")
    return json.dumps(selector, sort_keys=True, allow_nan=False, separators=(",", ":"))


def authored_contracts(root):
    """Read each guarantee/node independently; shared evidence is not ownership."""
    for source in sorted((root / "gdd/ssot/guarantees").glob("*.yaml")):
        document = load_ledger(source, root=root)
        for guarantee, _group, node in iter_registered_nodes(document, source):
            yield f"{guarantee['id']} / {node['pytest_node']}", node, node["case_binding"]


def required_node_errors(node, binding, *, canonical_suite=None):
    """Validate only authored minimum conditions, not redundant case-count pins."""
    cases = binding.get("fixture", {}).get("concrete_cases", [])
    execution = binding.get("execution", {})
    errors = parameter_case_errors(node, cases, execution=execution)
    errors.extend(
        canonical_requirement_errors(
            node.get("canonical_fixture"),
            cases,
            node.get("executable_checks"),
            execution=execution,
            suite=canonical_suite,
        )
    )
    return errors


def gfmc_parameter_contracts(root):
    """Bind independent GFMC minimum inputs to today's actual source expansion."""
    from gdd.engine.derivative_requirements import _GFMC_FUNCTION_SPECS, _gfmc_parameter_rows

    document = load_module(root / "gdd/ssot/guarantees/derivatives.yaml")
    groups = document["domain"]["gfmc_required_parameter_sets"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("GFMC independently required parameter sets must be nonempty")
    specs = {spec.function_name: spec for spec in _GFMC_FUNCTION_SPECS}
    for group in groups:
        path, separator, function = group["pytest_node"].partition("::")
        if not separator or function not in specs:
            raise ValueError(f"unbound GFMC requirement: {group.get('pytest_node')}")
        spec = specs[function]
        if (
            group["family"] != spec.family
            or not set(group["targets"]) <= set(spec.targets)
            or group["proof"] != spec.proof.value
        ):
            raise ValueError(f"{function}: GFMC family, derivative target or proof no longer matches its requirement")
        node = {
            "pytest_node": group["pytest_node"],
            "required_parameter_cases": group["parameters"],
            "required_precision_modes": group["precision"],
        }
        binding = {
            "fixture": {
                "concrete_cases": [{"selector": parameters} for parameters in _gfmc_parameter_rows(function, root / path)]
            },
            "execution": {"precision_modes": ["full", "mixed"], "profiles": [{"name": "serial", "world_size": 1}]},
        }
        yield f"GFMC {group['family']} {group['targets']} / {group['pytest_node']}", node, binding


def collection_requirement_errors(contracts, documents):
    """Require every explicit tuple in each required precision/profile collection.

    Only the declared tuples are required; no cross-product of fixture axes is
    created. The existing contract's launch profiles apply to those tuples.
    An excluded collection row cannot satisfy a required case, even if another
    row with different parameters did run successfully.
    """
    observed = set()
    for document in documents:
        if (
            document.get("schema") != "gdd-pytest-execution.v1"
            or document.get("kind") != "collection"
            or document.get("exitstatus") != 0
        ):
            raise ValueError("required cases need successful pytest collection manifests")
        profile = document["execution"]
        for row in document["selected"]:
            if row.get("excluded") or "selector" not in row:
                continue
            node = row["nodeid"].split("[", 1)[0]
            selector = row.get("contract_selector", row["selector"])
            observed.add((node, selector_key(selector), document["precision"], profile["name"], profile["world_size"]))
    errors = []
    for label, node, binding in contracts:
        if not (
            {"required_parameter_cases", "required_precision_modes", "required_execution_profiles", "canonical_fixture"}
            & node.keys()
        ):
            continue
        cases = binding["fixture"]["concrete_cases"]
        execution = binding["execution"]
        selectors = node.get("required_parameter_cases", [case["selector"] for case in cases])
        by_selector = {selector_key(case["selector"]): case for case in cases}
        profiles = node.get("required_execution_profiles", execution["profiles"])
        for selector in selectors:
            key = selector_key(selector)
            case = by_selector.get(key)
            if case is None:
                errors.append(f"{label}: required tuple absent from source cases: {key}")
                continue
            modes = node.get("required_precision_modes", case.get("precision_modes", execution["precision_modes"]))
            modes = ["full"] if modes == ["not-applicable"] else modes
            for precision in modes:
                for profile in profiles:
                    expected = (node["pytest_node"], key, precision, profile["name"], profile["world_size"])
                    if expected not in observed:
                        errors.append(
                            f"{label}: required tuple {key} not collected without exclusion in {precision}/{profile['name']}:{profile['world_size']}"
                        )
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--collection-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        contracts = list(authored_contracts(args.root)) + list(gfmc_parameter_contracts(args.root))
        fixture_path = args.root / "gdd/ssot/fixtures/wavefunctions.yaml"
        suite = yaml.load(fixture_path.read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))["canonical_suite"]
        errors = [
            f"{label}: {error}"
            for label, node, binding in contracts
            for error in required_node_errors(node, binding, canonical_suite=suite)
        ]
        if args.collection_dir:
            paths = sorted(args.collection_dir.rglob("gdd-collection-*.json"))
            if not paths:
                raise ValueError("no actual collection manifests found")
            documents = [json.loads(path.read_text()) for path in paths]
            errors.extend(collection_requirement_errors(contracts, documents))
            # Every derivative obligation is registered by exactly one collected required case.
            errors.extend(collection_closure_errors(documents))
    except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as exc:
        errors = [str(exc)]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("GDD explicit required cases: parameter tuples, precisions, launch profiles and canonical fixtures satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
