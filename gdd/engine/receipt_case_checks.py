"""The module-contract obligation inventory (one per registered guarantee/node edge).

``expected_obligations`` expands every module contract loaded by
``gdd.engine.module_receipts.load_node_contracts`` into its concrete
(case, precision, backend, execution profile) obligations, and
``node_obligations`` restricts that inventory to one pytest node.
``scripts/check_gdd_module_contracts.py`` compares both with the authored
YAML edges in ``scripts/gdd_sync.py --check`` and the CI preflight.

Required-case completion uses ``gdd.engine.execution_records``.
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

from dataclasses import dataclass
from functools import cache
from typing import Any

from gdd.engine.module_receipts import (
    EvidenceNodeContract,
    _receipt_precision_modes,
    canonical_digest,
    expected_artifacts,
    load_node_contracts,
)


@dataclass(frozen=True, order=True)
class ObligationKey:
    guarantee_id: str
    pytest_node: str
    parameter_id: str
    selector_digest: str
    precision: str
    backend: str
    profile: str
    world_size: int


@dataclass(frozen=True)
class ExpectedObligation:
    key: ObligationKey
    ledger: str
    claim_refs: tuple[str, ...]
    selector: dict[str, Any]
    case: int | str
    binding_digest: str
    source_snapshot: str
    artifacts: tuple[tuple[str, str], ...]
    electron_mode: str
    cond_g_contract: Any
    wavefunction_scope: str
    runtime_object: str
    observations: tuple[dict[str, Any], ...]
    runtime_parameter_id: str | None
    executable_checks: dict[str, Any] | None = None
    canonical_fixture: dict[str, Any] | None = None
    verdict_method: str | None = None
    condition_check_iterations: dict[str, Any] | None = None
    # A declared evaluation-kappa screen (absent for every cond(G) case).
    evaluation_conditioning_contract: Any = None
    # A verified per-case finiteness-only (smoke / stability) disposition.
    finiteness_evidence: dict[str, Any] | None = None


def _expected_for_contract(contract: EvidenceNodeContract) -> list[ExpectedObligation]:
    binding = contract.case_binding
    execution = binding["execution"]
    electron = binding["electron_configuration"]
    wavefunction = binding["wavefunction"]
    observations = tuple(binding["comparison"].get("typed_observations", ()))
    obligations: list[ExpectedObligation] = []
    for case in contract.cases:
        selector = case.get("selector", {})
        runtime_parameter_id = case.get("parameter_id")
        parameter_id = "not-parametrized" if runtime_parameter_id is None else runtime_parameter_id
        for precision in _receipt_precision_modes(contract, case):
            for backend in case.get("backends", execution["backends"]):
                for profile in execution["profiles"]:
                    key = ObligationKey(
                        guarantee_id=contract.guarantee_id,
                        pytest_node=contract.pytest_node,
                        parameter_id=parameter_id,
                        selector_digest=canonical_digest(selector),
                        precision=precision,
                        backend=backend,
                        profile=profile["name"],
                        world_size=int(profile["world_size"]),
                    )
                    obligations.append(
                        ExpectedObligation(
                            key=key,
                            ledger=contract.ledger,
                            claim_refs=contract.claim_refs,
                            selector=selector,
                            case=case["case"],
                            binding_digest=contract.binding_digest,
                            source_snapshot=binding["fixture"]["source_snapshot"],
                            artifacts=tuple(sorted(expected_artifacts(contract, case).items())),
                            # A statically derived/verified per-case disposition narrows the node mode.
                            electron_mode=case.get("electron_configuration", electron)["mode"],
                            cond_g_contract=electron.get("cond_G", "not-applicable"),
                            wavefunction_scope=str(wavefunction.get("scope", "not-applicable")),
                            runtime_object=str(wavefunction.get("runtime_object", "required")),
                            observations=observations,
                            runtime_parameter_id=runtime_parameter_id,
                            executable_checks=contract.executable_checks,
                            canonical_fixture=contract.canonical_fixture,
                            verdict_method=binding["comparison"]["verdict_method"],
                            condition_check_iterations=contract.condition_check_iterations,
                            evaluation_conditioning_contract=electron.get("evaluation_conditioning"),
                            finiteness_evidence=case.get("finiteness_evidence"),
                        )
                    )
    return obligations


@cache
def expected_obligations() -> dict[ObligationKey, ExpectedObligation]:
    expected: dict[ObligationKey, ExpectedObligation] = {}
    for contract in (contract for owners in load_node_contracts().values() for contract in owners):
        for obligation in _expected_for_contract(contract):
            if obligation.key in expected:
                raise ValueError(f"duplicate concrete GDD obligation: {obligation.key}")
            expected[obligation.key] = obligation
    return expected


@cache
def node_obligations(pytest_node: str) -> dict[ObligationKey, ExpectedObligation]:
    """The obligation universe restricted to one node, built by the same constructor.

    A receipt key names its guarantee and node.  A node shared by several
    guarantees has one contract per guarantee, so this is the union of their
    obligations: the completion universe ``expected_obligations()`` restricted
    to ``pytest_node`` without materializing all other nodes in a pytest worker.
    """
    return {
        obligation.key: obligation
        for contract in load_node_contracts().get(pytest_node, ())
        for obligation in _expected_for_contract(contract)
    }
