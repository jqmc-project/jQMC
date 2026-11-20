"""Pure-Python canonical subscriptions for isolated static gates and MD views.

No NumPy, JAX or production import is needed to read authored requirements and
compare them with the source-derived parameter inventory.
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

import json
import math
from pathlib import Path

import yaml

from gdd.engine.ledger import iter_registered_nodes


ROOT = Path(__file__).resolve().parents[2]
KINETIC_OPERATION = "jqmc.wavefunction.compute_kinetic_energy_all_elements"
KINETIC_ACTIVATION = "kinetic-jastrow-component-ablation"
ORBITAL_ACTIVATION = "kinetic-composite-coefficient-ablation"
STREAMING_OPERATION = "jqmc.wavefunction.canonical-sm-streaming-chain"
ECP_OPERATION = "jqmc.coulomb_potential.compute_ecp_non_local_parts_nearest_neighbors"


def required_canonical_cases(requirement, suite):
    """Select an operation's physical domain, never a test-selected passing subset.

    The ECP subscription consumes every core-suite row whose nuclear target is
    ECP-zero. AE rows remain mandatory for kinetic/streaming; they cannot witness
    nonlocal ECP. This does not claim an exhaustive universe of ECP wavefunctions.
    """
    if requirement["operation"] == ECP_OPERATION:
        return [case for case in suite["cases"] if case["signature"]["jastrow"]["dtn_en_cusp_target"] == "ecp-zero"]
    return suite["cases"]


def canonical_comparisons(requirement):
    """Map consumed quantities to independently authored comparator names."""
    if requirement["operation"] == STREAMING_OPERATION:
        return {
            "inverse": requirement["chain"]["inverse_comparison"],
            "fast": requirement["chain"]["fast_comparison"],
            "streaming": requirement["comparison"],
        }
    return {"output": requirement["comparison"]}


def required_jastrow_components(signature):
    """Read the required components from a canonical signature, never observations."""
    jastrow = signature["jastrow"]
    components = [
        name for name, field in (("J1", "j1_form"), ("J2", "j2_form"), ("J3", "j3_representation")) if jastrow[field] != "off"
    ]
    if jastrow["dtn_channel_mask"] != "off":
        components.extend("DTN-" + channel for channel in jastrow["dtn_channel_mask"].split("_"))
    return components


def _fixture_document():
    return yaml.load(
        (ROOT / "gdd/ssot/fixtures/wavefunctions.yaml").read_text(encoding="utf-8"),
        Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader),
    )


def canonical_suite():
    """Read authored requirements, never infer them from collected tests."""
    return _fixture_document()["canonical_suite"]


def required_composite_components(signature):
    """Name ordered leaves and spin consumers from the independent fixture contract."""
    definitions = _fixture_document()["orbital_signatures"]
    determinant, jastrow = signature["determinant"], signature["jastrow"]
    paired = signature["electrons"]["spin_restriction"] == "spin-restricted"
    roles = {"determinant.paired" if paired else "determinant.up": determinant["orbitals"]["up"]}
    if not paired and determinant["orbitals"]["down"] != "not-applicable":
        roles["determinant.down"] = determinant["orbitals"]["down"]
    if jastrow["j3_representation"] != "off":
        roles["J3"] = jastrow["j3_orbital_signature"]
    return {
        f"{role}/{index}:{leaf}": {"role": role, "index": index, "leaf": leaf}
        for role, orbital in roles.items()
        for index, leaf in enumerate(definitions[orbital].get("children", ()))
    }


def canonical_requirement_errors(requirement, cases, checks, suite=None, execution=None):
    """Require the independent pattern universe and a bound numerical comparison."""
    if requirement is None:
        return []
    suite = canonical_suite() if suite is None else suite
    fields = {
        "suite",
        "selector",
        "operation",
        "comparison",
        "electron_configuration",
    }
    if not isinstance(requirement, dict) or not fields <= set(requirement) <= fields | {
        "activation",
        "orbital_activation",
        "chain",
    }:
        return ["canonical fixture requirement fields are invalid"]
    errors = []
    if "activation" in requirement and requirement["activation"] != KINETIC_ACTIVATION:
        errors.append("unsupported canonical activation requirement")
    if "orbital_activation" in requirement and requirement["orbital_activation"] != ORBITAL_ACTIVATION:
        errors.append("unsupported canonical orbital activation requirement")
    if requirement["suite"] != suite["id"] or requirement["operation"] not in {
        KINETIC_OPERATION,
        STREAMING_OPERATION,
        ECP_OPERATION,
    }:
        errors.append("canonical suite or consumption operation is unknown")
    if requirement["operation"] == ECP_OPERATION and any(
        name in requirement for name in ("activation", "orbital_activation", "chain")
    ):
        errors.append("canonical ECP requires its own nonlocal coefficient ablation, not kinetic activation")
    streaming = requirement["operation"] == STREAMING_OPERATION
    if streaming:
        chain = requirement.get("chain")
        if not isinstance(chain, dict) or set(chain) != {
            "sweeps",
            "order",
            "displacement_algorithm",
            "displacement_scale",
            "inverse_comparison",
            "fast_comparison",
        }:
            return [*errors, "canonical chain requirement fields are invalid"]
        if (
            type(chain["sweeps"]) is not int
            or chain["sweeps"] < 1
            or chain["order"] != "up-then-down-by-index"
            or chain["displacement_algorithm"] != "numpy.random.PCG64.normal-continuation"
            or type(chain["displacement_scale"]) not in (int, float)
            or not math.isfinite(chain["displacement_scale"])
            or chain["displacement_scale"] <= 0
            or "activation" in requirement
            or "orbital_activation" in requirement
        ):
            errors.append("unsupported canonical chain policy")
    elif "chain" in requirement:
        errors.append("canonical chain requires the streaming consumption operation")
    selector = requirement["selector"]
    if not isinstance(selector, str) or not selector.isidentifier():
        return [*errors, "canonical pattern selector is invalid"]
    patterns = [case.get("selector", {}).get(selector) for case in cases]
    wanted = {case["id"] for case in required_canonical_cases(requirement, suite)}
    if not wanted:
        errors.append("canonical operation has no independently required patterns")
    if any(not isinstance(pattern, str) for pattern in patterns) or set(patterns) != wanted:
        errors.append("canonical registered pattern set differs from the required suite")
    if len(patterns) != len(wanted):
        errors.append("canonical patterns must each have one concrete parameter case")
    for case in cases:
        modes = case.get("precision_modes", (execution or {}).get("precision_modes", ("full", "mixed")))
        if set(modes) != {"full", "mixed"}:
            errors.append("canonical parameter cases must retain both full and mixed obligations")
    names = list(canonical_comparisons(requirement).values())
    if any(not isinstance(name, str) for name in names):
        errors.append("canonical comparison names must be strings")
    else:
        sites = []
        for name in names:
            check = (checks or {}).get(name)
            scope = _canonical_comparison_scope(requirement, check)
            if not scope:
                errors.append("canonical use requires a bound numerical comparison with an authorized precision scope")
            else:
                sites.append(check.get("site"))
        if streaming and (len(set(names)) != 3 or len(set(sites)) != 3):
            errors.append("canonical chain requires three distinct comparison sites")
    electron = requirement["electron_configuration"]
    if not isinstance(electron, dict) or set(electron) != {"algorithm", "seed", "scale", "configurations", "cond_G"}:
        errors.append("canonical electron configuration fields are invalid")
    elif (
        electron["algorithm"] != "numpy.random.PCG64.normal"
        or type(electron["seed"]) is not int
        or electron["seed"] < 0
        or type(electron["scale"]) not in (int, float)
        or not math.isfinite(electron["scale"])
        or electron["scale"] <= 0
        or type(electron["configurations"]) is not int
        or electron["configurations"] != 1
        or electron["cond_G"] != "finite-not-limited"
    ):
        errors.append("unsupported canonical electron configuration policy")
    return errors


def _canonical_comparison_scope(requirement, check):
    """Keep legacy contracts or the reviewed kinetic oracle partition."""
    if not isinstance(check, dict):
        return set()
    precision = check.get("precision", {})
    if not isinstance(precision, dict) or any(not isinstance(row, dict) for row in precision.values()):
        return set()
    scope = set(precision)
    comparator = check.get("comparator")
    if scope == {"full", "mixed"} and comparator == "numpy.testing.assert_allclose":
        return scope
    if requirement.get("operation") != KINETIC_OPERATION:
        return set()
    if scope == {"full"} and comparator == "numpy.testing.assert_allclose":
        return scope if precision["full"].get("policy") == "CMP-ZONE-ALLCLOSE-STRICT-001" else set()
    if scope == {"mixed"} and comparator == "python.inequality":
        budget = precision["mixed"]
        if budget.get("policy") == "CMP-MIXED-FULL-FIDELITY-001" and budget.get("quantity_class") == "orbital_derivative":
            return scope
    return set()


def canonical_comparison_precision_modes(requirement, checks):
    """Read one authorized receipt scope without changing physical execution."""
    try:
        scopes = [
            _canonical_comparison_scope(requirement, (checks or {}).get(name))
            for name in canonical_comparisons(requirement).values()
        ]
    except (KeyError, TypeError, AttributeError):
        return set()
    if not scopes or not scopes[0] or any(scope != scopes[0] for scope in scopes[1:]):
        return set()
    return scopes[0]


def canonical_precision_coverage_errors(nodes, suite=None):
    """Require both numerical oracles for every independently required pattern.

    Pair registrations only when their test node and entire authored physical
    requirement agree. A full-only kinetic registration cannot stand alone or
    borrow a mixed oracle from a different seed, activation or operation.
    """
    suite = canonical_suite() if suite is None else suite
    grouped = {}
    for node in nodes:
        requirement = node.get("canonical_fixture")
        if (
            not isinstance(requirement, dict)
            or not {"operation", "comparison", "selector"} <= set(requirement)
            or not isinstance(requirement["selector"], str)
            or not isinstance(requirement["comparison"], str)
        ):
            continue
        key = (node["pytest_node"], json.dumps(requirement, sort_keys=True))
        patterns = grouped.setdefault(key, {})
        scope = canonical_comparison_precision_modes(requirement, node.get("executable_checks"))
        for case in node["case_binding"]["fixture"]["concrete_cases"]:
            pattern = case.get("selector", {}).get(requirement["selector"])
            patterns.setdefault(pattern, set()).update(scope)
    errors = []
    for (pytest_node, requirement_text), patterns in grouped.items():
        requirement = json.loads(requirement_text)
        for case in required_canonical_cases(requirement, suite):
            missing = {"full", "mixed"} - patterns.get(case["id"], set())
            if missing:
                errors.append(
                    f"{pytest_node}: canonical pattern {case['id']} lacks paired numerical coverage for {sorted(missing)}"
                )
    return errors


def ledger_subscription_errors(document, suite, source):
    """Inspect node subscriptions without importing the runtime receipt engine."""
    errors = []
    subscriptions = []
    # Derivative matrix evidence is owned by its manifest gate. Its ordinary
    # module_evidence nodes still participate through the shared registration API.
    for _guarantee, _group, node in iter_registered_nodes(document, source):
        if node.get("canonical_fixture") is None:
            continue
        subscriptions.append(node)
        binding = node["case_binding"]
        failures = canonical_requirement_errors(
            node["canonical_fixture"],
            binding["fixture"]["concrete_cases"],
            node.get("executable_checks"),
            suite=suite,
            execution=binding.get("execution"),
        )
        errors.extend(f"{node['pytest_node']}: {failure}" for failure in failures)
    errors.extend(canonical_precision_coverage_errors(subscriptions, suite=suite))
    return errors
