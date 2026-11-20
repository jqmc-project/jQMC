"""Canonical paired-oracle closure preserves all independently authored patterns."""

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

import copy

import pytest

from gdd.engine import canonical_contract as engine

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]
SUITE = engine.canonical_suite()
PATTERN_IDS = tuple(case["id"] for case in SUITE["cases"])
REQUIREMENT = {
    "suite": SUITE["id"],
    "selector": "pattern_id",
    "operation": engine.KINETIC_OPERATION,
    "comparison": "per-electron-kinetic",
    "activation": engine.KINETIC_ACTIVATION,
    "orbital_activation": engine.ORBITAL_ACTIVATION,
    "electron_configuration": {
        "algorithm": "numpy.random.PCG64.normal",
        "seed": 864,
        "scale": 0.6,
        "configurations": 1,
        "cond_G": "finite-not-limited",
    },
}


def subscription(scope):
    if scope == "mixed":
        comparator = "python.inequality"
        precision = {"mixed": {"policy": "CMP-MIXED-FULL-FIDELITY-001", "quantity_class": "orbital_derivative"}}
    else:
        comparator = "numpy.testing.assert_allclose"
        modes = ["full", "mixed"] if scope == "legacy" else ["full"]
        precision = {mode: {"policy": "CMP-ZONE-ALLCLOSE-STRICT-001", "zones": ["wf_kinetic"]} for mode in modes}
    return {
        "pytest_node": "tests/test_canonical.py::test_kinetic",
        "canonical_fixture": copy.deepcopy(REQUIREMENT),
        "executable_checks": {"per-electron-kinetic": {"site": "#output", "comparator": comparator, "precision": precision}},
        "case_binding": {
            "fixture": {"concrete_cases": [{"selector": {"pattern_id": case["id"]}} for case in SUITE["cases"]]},
            "execution": {"precision_modes": ["full", "mixed"]},
        },
    }


def local_errors(node):
    return engine.canonical_requirement_errors(
        node["canonical_fixture"],
        node["case_binding"]["fixture"]["concrete_cases"],
        node["executable_checks"],
        suite=SUITE,
        execution=node["case_binding"]["execution"],
    )


def all_errors(nodes):
    return [error for node in nodes for error in local_errors(node)] + engine.canonical_precision_coverage_errors(
        nodes, suite=SUITE
    )


@pytest.mark.parametrize("pairing", ["paired", "full", "mixed", "other-seed"])
def test_unpaired_registration_is_rejected_for_every_pattern(pairing):
    """Removing either oracle, or pairing it with another seed, leaves a gap for every canonical pattern."""
    if pairing in {"full", "mixed"}:
        nodes = [subscription(pairing)]
    else:
        nodes = [subscription("full"), subscription("mixed")]
        if pairing == "other-seed":
            nodes[1]["canonical_fixture"]["electron_configuration"]["seed"] += 1
    unpaired = sum("lacks paired numerical coverage" in error for error in all_errors(nodes))
    assert unpaired == {"paired": 0, "full": 1, "mixed": 1, "other-seed": 2}[pairing] * len(PATTERN_IDS)


def test_missing_one_mixed_pattern_is_rejected_by_both_pattern_and_coverage_checks():
    """A missing mixed selector fails the pattern universe and paired closure."""
    pattern_id = PATTERN_IDS[-1]
    nodes = [subscription("full"), subscription("mixed")]
    cases = nodes[1]["case_binding"]["fixture"]["concrete_cases"]
    cases[:] = [case for case in cases if case["selector"]["pattern_id"] != pattern_id]
    errors = all_errors(nodes)
    assert any("pattern set differs" in error for error in errors)
    assert any(f"{pattern_id} lacks paired" in error for error in errors)


@pytest.mark.parametrize(
    "scope,field,value",
    [
        ("full", "policy", "CMP-ZONE-ALLCLOSE-MEDIUM-001"),
        ("mixed", "policy", "CMP-ZONE-ALLCLOSE-STRICT-001"),
        ("mixed", "quantity_class", "parameter_derivative"),
        ("mixed", "comparator", "numpy.testing.assert_allclose"),
        ("full", "operation", "jqmc.coulomb_potential.compute_ecp_non_local_parts_nearest_neighbors"),
    ],
)
def test_partition_cannot_change_approved_oracle_budget(scope, field, value):
    """Partial registrations cannot substitute an unapproved policy, quantity or comparator, or leave the kinetic oracle."""
    node = subscription(scope)
    if field == "comparator":
        node["executable_checks"]["per-electron-kinetic"]["comparator"] = value
    elif field == "operation":
        node["canonical_fixture"]["operation"] = value
        node["canonical_fixture"].pop("activation")
        node["canonical_fixture"].pop("orbital_activation")
    else:
        node["executable_checks"]["per-electron-kinetic"]["precision"][scope][field] = value
    assert any("authorized precision scope" in error for error in local_errors(node))


def test_physical_test_execution_still_requires_both_modes():
    """Partitioned receipts do not authorize removing mixed physical execution."""
    node = subscription("full")
    node["case_binding"]["execution"]["precision_modes"] = ["full"]
    assert any("retain both full and mixed" in error for error in local_errors(node))


@pytest.mark.parametrize(
    "key,value,fragment",
    [
        ("activation", "disabled", "activation"),
        ("orbital_activation", "disabled", "orbital activation"),
        ("cond_G", "less-than-1e8", "electron configuration"),
        ("configurations", 2, "electron configuration"),
    ],
)
def test_activation_and_unscreened_electron_contracts_remain_mandatory(key, value, fragment):
    """Activation and finite-not-limited electron requirements survive the partition."""
    node = subscription("legacy")
    target = (
        node["canonical_fixture"]
        if key in {"activation", "orbital_activation"}
        else node["canonical_fixture"]["electron_configuration"]
    )
    target[key] = value
    assert any(fragment in error for error in local_errors(node))


def test_streaming_three_distinct_sites_remain_mandatory():
    """Streaming keeps independently bound inverse, fast and streaming comparisons."""
    node = subscription("legacy")
    requirement = node["canonical_fixture"]
    requirement.update(
        operation=engine.STREAMING_OPERATION,
        chain={
            "sweeps": 1,
            "order": "up-then-down-by-index",
            "displacement_algorithm": "numpy.random.PCG64.normal-continuation",
            "displacement_scale": 0.01,
            "inverse_comparison": "inverse",
            "fast_comparison": "fast",
        },
    )
    requirement.pop("activation")
    requirement.pop("orbital_activation")
    check = node["executable_checks"]["per-electron-kinetic"]
    node["executable_checks"].update(inverse=copy.deepcopy(check), fast=copy.deepcopy(check))
    assert any("three distinct comparison sites" in error for error in local_errors(node))
    node["executable_checks"]["inverse"]["site"] = "#inverse"
    node["executable_checks"]["fast"]["site"] = "#fast"
    assert all_errors([node]) == []


@pytest.mark.parametrize("paired", [False, True])
def test_runtime_loader_rejects_missing_sibling_and_emits_only_its_own_precision(monkeypatch, tmp_path, paired):
    """Runtime collection rejects an absent sibling and scopes each retained receipt."""
    from gdd.engine import module_receipts

    nodes = [subscription("full")]
    if paired:
        nodes.append(subscription("mixed"))
    document = {
        "guarantees": [
            {"id": f"IMPL-SCRATCH-CANONICAL-{index:03d}", "required_evidence": [{"nodes": [node]}]}
            for index, node in enumerate(nodes)
        ]
    }
    (tmp_path / "wavefunction.yaml").write_text("{}\n")
    monkeypatch.setattr(module_receipts, "GUARANTEE_DIR", tmp_path)
    monkeypatch.setattr(module_receipts, "load_ledger", lambda path: copy.deepcopy(document))
    module_receipts.load_node_contracts.cache_clear()
    try:
        if not paired:
            with pytest.raises(ValueError, match="lacks paired numerical coverage"):
                module_receipts.load_node_contracts()
        else:
            contracts = module_receipts.load_node_contracts()[nodes[0]["pytest_node"]]
            assert len(contracts) == 2
            assert [module_receipts._receipt_precision_modes(contract, contract.cases[0]) for contract in contracts] == [
                ("full",),
                ("mixed",),
            ]
    finally:
        module_receipts.load_node_contracts.cache_clear()
