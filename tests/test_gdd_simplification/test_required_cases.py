"""Explicit authored fixture interactions survive the lightweight execution model."""

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

from scripts.check_gdd_required_cases import collection_requirement_errors, required_node_errors


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _contract():
    node = {
        "pytest_node": "tests/test_value.py::test_value",
        "required_parameter_cases": [{"basis": "bspline", "jastrow": "dtn", "twist": True}],
        "required_precision_modes": ["full", "mixed"],
        "required_execution_profiles": [{"name": "serial", "world_size": 1}],
    }
    binding = {
        "fixture": {"concrete_cases": [{"selector": dict(node["required_parameter_cases"][0])}]},
        "execution": {"precision_modes": ["full", "mixed"], "profiles": [{"name": "serial", "world_size": 1}]},
    }
    return "G / test_value", node, binding


def _documents(contract):
    _, node, _ = contract
    return [
        {
            "schema": "gdd-pytest-execution.v1",
            "kind": "collection",
            "exitstatus": 0,
            "precision": precision,
            "execution": {"name": "serial", "world_size": 1, "rank": 0},
            "selected": [
                {
                    "nodeid": node["pytest_node"] + "[case]",
                    "selector": dict(node["required_parameter_cases"][0]),
                    "excluded": None,
                }
            ],
        }
        for precision in ("full", "mixed")
    ]


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("tuple", "missing required parameter tuples"),
        ("profile", "execution scope"),
        ("precision", "missing required precision"),
    ],
)
def test_deleted_required_source_tuple_fails(mutation, message):
    """Regenerating the cache after narrowing the source does not bless a smaller case set.

    The source loses a required tuple, keeps only a serial case where an MPI
    launch is required, or drops a still-required mixed precision.
    """
    _, node, binding = _contract()
    if mutation == "tuple":
        binding["fixture"]["concrete_cases"][0]["selector"]["jastrow"] = "none"
    elif mutation == "profile":
        node["required_execution_profiles"] = [{"name": "mpi", "world_size": 2}]
    else:
        binding["execution"]["precision_modes"] = ["full"]
    assert any(message in error for error in required_node_errors(node, binding))


@pytest.mark.parametrize("mutation", ["none", "omitted", "mixed-lane", "excluded", "split-interaction", "shared-node"])
def test_actual_collection_cannot_omit_required_tuple(mutation):
    """A source-declared tuple must exist, unexcluded, in the actual collection of every lane.

    ``none`` is the control. A full collection cannot stand in for the mixed
    lane, a static skip stays an exclusion, two partial cases do not cover one
    required interaction, and a node shared by two guarantees keeps both
    requirements.
    """
    contract = _contract()
    contracts = [contract]
    documents = _documents(contract)
    if mutation == "omitted":
        documents[0]["selected"] = []
    elif mutation == "mixed-lane":
        documents = documents[:1]
    elif mutation == "excluded":
        documents[1]["selected"][0]["excluded"] = "not available"
    elif mutation == "split-interaction":
        for document in documents:
            first = document["selected"][0]
            second = copy.deepcopy(first)
            first["selector"]["jastrow"] = "none"
            second["selector"]["basis"] = "gto"
            document["selected"].append(second)
    elif mutation == "shared-node":
        second = copy.deepcopy(contract)
        second[1]["required_execution_profiles"] = [{"name": "mpi", "world_size": 2}]
        contracts.append(second)
    assert bool(collection_requirement_errors(contracts, documents)) == (mutation != "none")
