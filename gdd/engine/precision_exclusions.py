"""Shared static/runtime validation of precision-exclusion dependency edges.

This validates a recorded exclusion, not its scientific justification. In
particular, a reason containing 'FD' is not evidence that a test computes FD.
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

from typing import Any


# Full-precision-only rule for external fp64 references: the TurboRVB literals
# are fp64 values produced by an independent external fp64 program, and mixed
# output must never be compared against an independent fp64 program. Exactly
# these seven literal-regression nodes run in full precision only. This is a
# reviewed constant, never a generic external-reference exemption: any other
# node, guarantee, reason, parameter case or an added alternative is rejected.
EXTERNAL_FP64_FULL_ONLY_KIND = "external-fp64-reference-full-only"
EXTERNAL_FP64_FULL_ONLY_DECISION = "turborvb-external-fp64-literals-full-only"
EXTERNAL_FP64_FULL_ONLY_REASON = (
    "TurboRVB external fp64 literals: mixed output is never compared against an independent fp64 program; "
    "run in full precision only"
)
EXTERNAL_FP64_FULL_ONLY_GUARANTEE = "NUM-HAM-TURBORVB-LITERAL-REGRESSION-001"
EXTERNAL_FP64_FULL_ONLY_NODES = frozenset(
    {
        "tests/test_jqmc_kernel/test_integration_turborvb_AE.py::TestTurborvbAE::test_comparison_with_TurboRVB_wo_Jastrow_AE",
        "tests/test_jqmc_kernel/test_integration_turborvb_AE.py::TestTurborvbAE::test_comparison_with_TurboRVB_w_2b_1b3b_Jastrow_AE",
        "tests/test_jqmc_kernel/test_integration_turborvb_ECP.py::TestTurborvbECP::test_comparison_with_TurboRVB_wo_Jastrow_w_ecp",
        "tests/test_jqmc_kernel/test_integration_turborvb_ECP.py::TestTurborvbECP::test_comparison_with_TurboRVB_w_2b_Jastrow_w_ecp",
        "tests/test_jqmc_kernel/test_integration_turborvb_ECP.py::TestTurborvbECP::test_comparison_with_TurboRVB_w_2b_3b_Jastrow_w_ecp",
        "tests/test_jqmc_kernel/test_integration_turborvb_ECP.py::TestTurborvbECP::test_comparison_with_TurboRVB_w_2b_1b3b_Jastrow_w_ecp",
        "tests/test_jqmc_kernel/test_integration_turborvb_ECP.py::TestTurborvbECP::test_full_comparison_with_TurboRVB_w_2b_1b3b_Jastrow_w_ecp",
    }
)


def external_fp64_full_only(pytest_node: str, contracts: list[dict[str, str]]) -> bool:
    """True only for one of the seven listed nodes carrying exactly the reviewed marker."""
    return pytest_node in EXTERNAL_FP64_FULL_ONLY_NODES and contracts == [{"reason": EXTERNAL_FP64_FULL_ONLY_REASON}]


# Full-precision-only rule for the adversarial LRDMC interval chains: the
# adversarial cond(G) 1e7-1e10 chains are judged in full precision only,
# because the independent mixed oracle carries fp32 error above the mixed
# tolerance there; the precision-independent per-hop refresh logic keeps its
# mixed coverage through the categorical chains of the same nodes. Exactly the
# policy=adversarial parameter case of the three Rust long-chain nodes, with
# exactly this marker and reason, is full-only; the categorical cases, any other
# node, case, reason or an added alternative are rejected.
ADVERSARIAL_CHAIN_FULL_ONLY_KIND = "adversarial-chain-mixed-oracle-full-only"
ADVERSARIAL_CHAIN_FULL_ONLY_DECISION = "adversarial-chain-full-only"
ADVERSARIAL_CHAIN_FULL_ONLY_REASON = (
    "Adversarial cond(G) 1e7-1e10 chains: the mixed oracle carries fp32 error above the mixed tolerance; "
    "the parameter-free refresh logic is precision-independent and is judged in full; mixed coverage via "
    "the categorical chains"
)
ADVERSARIAL_CHAIN_FULL_ONLY_GUARANTEES = frozenset(
    {
        "IMPL-RS-GFMC-INTERVAL-STATE-LONG-CHAIN-001",
        "IMPL-RS-GFMC-DTN-TABLES-LONG-CHAIN-001",
        "IMPL-RS-GFMC-INTERVAL-STATE-XLA-ANCHOR-001",
    }
)
ADVERSARIAL_CHAIN_FULL_ONLY_NODES = frozenset(
    {
        "tests/test_rs_kernel/test_lrdmc_mesh_chain_rust.py::test_rust_lrdmc_chain_inverse_winv_and_mesh_match_fresh",
        "tests/test_rs_kernel/test_lrdmc_mesh_chain_rust.py::test_rust_lrdmc_chain_dtn_tables_match_fresh",
        "tests/test_rs_kernel/test_lrdmc_mesh_chain_rust.py::test_rust_lrdmc_chain_matches_xla_anchors",
    }
)
ADVERSARIAL_CHAIN_FULL_ONLY_PARAMETER_IDS = frozenset(
    f"{case}-adversarial"
    for case in ("h8_bspline_real_jdtn", "h8_bspline_twist_jdtn", "o_bspline_twist_jdtn", "si_pgto_real_jdtn")
)
ADVERSARIAL_CHAIN_FULL_ONLY_MARKER = f"pytest.mark.skip_mixed_precision(reason={ADVERSARIAL_CHAIN_FULL_ONLY_REASON!r})"


def adversarial_chain_full_only_case(pytest_node: str, parameter_id: Any, selector: Any, mixed_marks: list[str]) -> bool:
    """True only for a listed adversarial case carrying exactly the reviewed marker."""
    return (
        pytest_node in ADVERSARIAL_CHAIN_FULL_ONLY_NODES
        and parameter_id in ADVERSARIAL_CHAIN_FULL_ONLY_PARAMETER_IDS
        and isinstance(selector, dict)
        and selector.get("policy") == "adversarial"
        and mixed_marks == [ADVERSARIAL_CHAIN_FULL_ONLY_MARKER]
    )


def declares_mixed_precision(value: Any) -> bool:
    """Only a precision field can declare a derivative mixed obligation."""
    if isinstance(value, dict):
        return any(
            (key == "precision" and isinstance(child, list) and "mixed" in child) or declares_mixed_precision(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(declares_mixed_precision(child) for child in value)
    return False


def binding_exclusions(guarantee_id: str, pytest_node: str, execution: dict, cases: Any) -> list[dict]:
    """Retain both whole-node and per-parameter exclusions for the same validator."""
    rows = []
    owners = [(execution, {})]
    owners.extend((case, {"parameter_case": case.get("case"), "parameter_id": case.get("parameter_id")}) for case in cases)
    for owner, identity in owners:
        if "precision_exclusion" not in owner:
            continue
        exclusion = owner["precision_exclusion"]
        if not isinstance(exclusion, dict):
            raise ValueError(f"{guarantee_id}::{pytest_node}: precision exclusion must be a mapping")
        references = exclusion.get("alternative_guarantee_refs", ())
        if not isinstance(references, (list, tuple)) or not all(isinstance(ref, str) for ref in references):
            raise ValueError(f"{guarantee_id}::{pytest_node}: alternative guarantee references must be a sequence of IDs")
        rows.append(
            {
                "guarantee_id": guarantee_id,
                "pytest_node": pytest_node,
                **identity,
                "kind": exclusion.get("kind"),
                "mode": exclusion.get("mode"),
                "disposition": exclusion.get("disposition"),
                "reason": exclusion.get("reason"),
                "alternative_guarantee_refs": tuple(references),
                **({"decision": exclusion["decision"]} if "decision" in exclusion else {}),
            }
        )
    return rows


def exclusion_errors(rows: Any, catalog: dict, mixed_guarantees: set[str]) -> tuple[list[str], set[str]]:
    """Reject missing, nonblocking, or non-mixed alternatives before/after pytest."""
    errors = []
    external_alternatives = set()
    for row in rows:
        label = f"{row['guarantee_id']}::{row['pytest_node']}"
        if "parameter_case" in row:
            label += f"[case={row['parameter_case']}, id={row.get('parameter_id')!r}]"
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            errors.append(f"{label}: precision N/A has no reason")
        if row.get("disposition") != "not-applicable":
            errors.append(f"{label}: precision exclusion is not typed not-applicable")
        if row.get("kind") == "precision-independent-contract":
            if row.get("mode") != "precision" or row.get("alternative_guarantee_refs"):
                errors.append(f"{label}: malformed precision-independent N/A")
            continue
        if row.get("kind") == EXTERNAL_FP64_FULL_ONLY_KIND:
            if (
                row.get("guarantee_id") != EXTERNAL_FP64_FULL_ONLY_GUARANTEE
                or row.get("pytest_node") not in EXTERNAL_FP64_FULL_ONLY_NODES
                or "parameter_case" in row
                or row.get("mode") != "mixed"
                or row.get("reason") != EXTERNAL_FP64_FULL_ONLY_REASON
                or row.get("decision") != EXTERNAL_FP64_FULL_ONLY_DECISION
                or row.get("alternative_guarantee_refs")
            ):
                errors.append(f"{label}: {EXTERNAL_FP64_FULL_ONLY_KIND} is admitted only for the seven TurboRVB literal nodes")
            continue
        if row.get("kind") == ADVERSARIAL_CHAIN_FULL_ONLY_KIND:
            if (
                row.get("guarantee_id") not in ADVERSARIAL_CHAIN_FULL_ONLY_GUARANTEES
                or row.get("pytest_node") not in ADVERSARIAL_CHAIN_FULL_ONLY_NODES
                or "parameter_case" not in row
                or row.get("parameter_id") not in ADVERSARIAL_CHAIN_FULL_ONLY_PARAMETER_IDS
                or row.get("mode") != "mixed"
                or row.get("reason") != ADVERSARIAL_CHAIN_FULL_ONLY_REASON
                or row.get("decision") != ADVERSARIAL_CHAIN_FULL_ONLY_DECISION
                or row.get("alternative_guarantee_refs")
            ):
                errors.append(f"{label}: {ADVERSARIAL_CHAIN_FULL_ONLY_KIND} is admitted only for the adversarial chain cases")
            continue
        if row.get("kind") != "finite-difference-or-numerical-oracle-not-applicable":
            errors.append(f"{label}: unknown precision-exclusion kind {row.get('kind')!r}")
            continue
        if row.get("mode") != "mixed":
            errors.append(f"{label}: numerical-oracle exclusion must target mixed precision")
        references = row.get("alternative_guarantee_refs", ())
        if len(references) != 1:
            errors.append(f"{label}: FD/numerical mixed N/A requires exactly one alternative guarantee")
            continue
        target = references[0]
        target_contract = catalog.get(target)
        if target_contract is None:
            errors.append(f"{label}: alternative guarantee {target!r} does not exist")
            continue
        if not target_contract["blocking"]:
            errors.append(f"{label}: alternative guarantee {target!r} is not blocking")
        if target in mixed_guarantees:
            continue
        if target_contract["ledger"] == "derivatives.yaml" and target_contract["declares_mixed"]:
            external_alternatives.add(target)
            continue
        errors.append(f"{label}: alternative guarantee {target!r} has no blocking mixed obligation")
    return errors, external_alternatives


def document_exclusion_errors(documents: Any) -> list[str]:
    """Inspect materialized cases without importing or collecting scientific tests."""
    catalog = {}
    mixed_guarantees = set()
    rows = []
    for path, document in documents:
        for section in ("guarantees", "module_evidence"):
            for guarantee in document.get(section, ()):
                identity = guarantee["id"]
                if identity in catalog:
                    raise ValueError(f"duplicate guarantee identity: {identity}")
                evidence = guarantee.get("required_evidence", ())
                catalog[identity] = {
                    "ledger": path.name,
                    "blocking": True,
                    "declares_mixed": declares_mixed_precision(evidence),
                }
                if not isinstance(evidence, list):
                    continue
                for group in evidence:
                    for node in group.get("nodes", ()):
                        binding = node.get("case_binding", {})
                        execution = binding.get("execution", {})
                        cases = binding.get("fixture", {}).get("concrete_cases", ())
                        rows.extend(binding_exclusions(identity, node["pytest_node"], execution, cases))
                        if any("mixed" in case.get("precision_modes", execution.get("precision_modes", ())) for case in cases):
                            mixed_guarantees.add(identity)
    return exclusion_errors(rows, catalog, mixed_guarantees)[0]
