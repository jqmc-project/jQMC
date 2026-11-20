#!/usr/bin/env python3
"""Render the human-first derivative guarantee ledger from its YAML SSoT."""

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
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONTRACT_PATH = ROOT / "gdd/ssot/guarantees/derivatives.yaml"
OUTPUT_PATH = ROOT / "gdd/ssot/guarantees/README.md"
GENERATED_OUTPUT_PATH = ROOT / "gdd/derived/derivatives-requirements.md"


def continuous_matrix_scope() -> tuple[int, list[list[str]]]:
    """Describe the measured selected scope, never imply all admissible cells were tested."""
    from gdd.engine import derivative_matrix, derivative_requirements

    selected = {tuple(cell) for cell, _ in derivative_requirements._MATRIX}
    admissible = set(derivative_matrix._candidates())
    buildable = {cell for cell in admissible if derivative_matrix._has_declared_builder(cell)}
    return len(selected), [
        [
            "Continuous matrix selection",
            f"{len(selected)} selected cells from {len(admissible)} admissible cells; "
            f"{len(buildable)} have declared builders and {len(buildable - selected)} of those are not selected.",
        ],
        [
            "Required interactions",
            "All admissible pairs across AO family, Jastrow, potential, boundary and determinant; "
            "reviewed twist/Jastrow/potential triples; the periodic-Gamma ECP/Jastrow block for real Cartesian "
            "and spherical pGTO. Each selected cell retains both quantities and all four derivative axes.",
        ],
        [
            "Coverage limit",
            "This is a reviewed pairwise covering array with the selective interactions above, not the full "
            "Cartesian product or every admissible higher-order interaction. Unselected cells are not "
            "certified by another cell sharing individual axis labels. Targeted and numerical-regime "
            "obligations remain additional requirements.",
        ],
    ]


def reference_form_table(gfmc: dict) -> list[str]:
    """Display reference distinctions without treating every AD-based row as strain AD."""
    from gdd.engine.derivative_reference_forms import measured_reference_forms, reference_form_errors

    if errors := reference_form_errors(gfmc):
        raise ValueError("; ".join(errors))
    labels = {
        "target-coordinate-ad": "AD of the target coordinates",
        "position-ad-affine-chain-rule": "Position AD followed by the affine chain rule",
        "conditioned-component-ad": "Separate AD of tmove and the dltmove-minus-tmove correction",
        "rank-one-ecp-ratio-ad": "AD of the energy with determinant-lemma non-local ECP ratios",
        "rank-one-ecp-ratio-fused-base-ad": (
            "AD of the energy with determinant-lemma non-local ECP ratios off the fused-table base inverse"
        ),
        "algorithm-identity": "Reviewed equality of conventional and load-balanced derivatives",
    }
    algorithms = {"e_L_t": "Conventional algorithm", "e_L_n": "Load-balanced algorithm"}
    targets = {"positions": "Electron and nuclear positions", "strain": "Strain / affine response"}
    partition = measured_reference_forms(gfmc)
    total = sum(row["cases"] for row in partition)
    affine = sum(row["cases"] for row in partition if row["reference_form"] == "position-ad-affine-chain-rule")
    conditioned = sum(row["cases"] for row in partition if row["reference_form"] == "conditioned-component-ad")
    rank_one = sum(row["cases"] for row in partition if row["reference_form"] == "rank-one-ecp-ratio-ad")
    rank_one_fused = sum(row["cases"] for row in partition if row["reference_form"] == "rank-one-ecp-ratio-fused-base-ad")
    lines = [
        "## GFMC reference methods",
        "",
        f"This partition covers the broad matrix only: {total} required cases in **each** precision lane. "
        "The `gfmc-direct` registration category means AD-based evidence; it does not "
        "mean that every strain reference differentiates a single final tensor graph. "
        "This partition does not change the case IDs, required precision modes or tolerances.",
        "",
        "| Algorithm | Derivative | Reference actually used | Cases per precision |",
        "| --- | --- | --- | ---: |",
    ]
    for row in partition:
        lines.append(
            f"| {algorithms[row['family']]} | {targets[row['target']]} | {labels[row['reference_form']]} | {row['cases']} |"
        )
    lines.extend(
        [
            "",
            f"The {affine} affine-chain-rule cases concern molecular all-electron inputs. "
            f"The {conditioned} conditioned-component cases concern molecular ECP + DTN with dltmove; "
            "they compare both independently differentiated terms, not the cancelled final tensor directly. "
            f"The {rank_one} rank-one ECP-ratio cases concern molecular N2 ECP without a Jastrow factor; "
            "their AD reference evaluates the same energy with the mesh ratios formed from the base inverse, "
            "so the derivative is not a cancellation of separately recomputed mesh determinants. "
            f"The {rank_one_fused} fused-base rank-one cases concern molecular Cl2 ECP with the dtn and j123ao_dtn Jastrows; "
            "their reference uses the same rank-one ratio energy with the base inverse taken from the fused "
            "orbital table that the kinetic term inverts, and the Jastrow ratio from the jastrow_ratio zone. "
            "The static contract measures this partition from the registered parameter cases and source calls, "
            "and rejects changed counts, missing routes or overlapping selectors. "
            "These are reference-method counts, not a claim that runtime evidence has passed.",
            "",
        ]
    )
    return lines


def live_obligation_counts(contract: dict) -> dict[str, int]:
    """Measure displayed coverage from requirements, never archival count pins."""
    from collections import Counter

    from gdd.engine.derivative_closure import mixed_route_ids, required_fd_ids
    from gdd.engine.derivative_requirements import REQUIRED_CASES

    families = Counter(case.family for case in REQUIRED_CASES)
    return {
        "NUM-DERIV-FD-AUTODIFF-001": len(required_fd_ids(contract)),
        "MATH-DERIV-LOG-WAVEFUNCTION-001": families["ln_Psi"],
        "MATH-DERIV-LOCAL-ENERGY-001": families["e_L"],
        "MATH-DERIV-CONVENTIONAL-ALGORITHM-001": families["e_L_t"],
        "MATH-DERIV-LOAD-BALANCED-ALGORITHM-001": families["e_L_n"],
        "NUM-DERIV-MIXED-001": len(mixed_route_ids()),
        "IMPL-DERIV-RUST-XLA-001": len(REQUIRED_CASES),
        "OPS-DERIV-CLOSURE-001": len(REQUIRED_CASES),
    }


def render(contract: dict) -> str:
    guarantees = {item["id"]: item for item in contract["guarantees"]}
    integration = contract["existing_test_integration"]
    counts = live_obligation_counts(contract)
    rows = (
        (
            "NUM-DERIV-FD-AUTODIFF-001",
            f"{counts['NUM-DERIV-FD-AUTODIFF-001']} reviewed sentinel cases",
            integration["fd_autodiff_sentinels"]["evidence_collection"],
        ),
        (
            "MATH-DERIV-LOG-WAVEFUNCTION-001",
            f"{counts['MATH-DERIV-LOG-WAVEFUNCTION-001']} obligations",
            "connected",
        ),
        (
            "MATH-DERIV-LOCAL-ENERGY-001",
            f"{counts['MATH-DERIV-LOCAL-ENERGY-001']} obligations",
            "connected",
        ),
        (
            "MATH-DERIV-CONVENTIONAL-ALGORITHM-001",
            f"{counts['MATH-DERIV-CONVENTIONAL-ALGORITHM-001']} obligations",
            "connected",
        ),
        (
            "MATH-DERIV-LOAD-BALANCED-ALGORITHM-001",
            f"{counts['MATH-DERIV-LOAD-BALANCED-ALGORITHM-001']} obligations",
            "connected",
        ),
        (
            "NUM-DERIV-MIXED-001",
            f"{counts['NUM-DERIV-MIXED-001']} production routes",
            integration["mixed_precision_routes"]["evidence_collection"],
        ),
        (
            "IMPL-DERIV-RUST-XLA-001",
            (f"{counts['IMPL-DERIV-RUST-XLA-001']} required Rust obligations; no N/A obligations"),
            integration["rust_xla_parity"]["evidence_collection"],
        ),
        (
            "OPS-DERIV-CLOSURE-001",
            "complete XLA, FD, mixed-precision, and Rust receipt universes",
            integration["cross_worker_completion"]["evidence_collection"],
        ),
    )

    lines = [
        "# Derivative guarantee ledger",
        "",
        "<!-- Generated by scripts/render_gdd_derivatives_ledger.py. Do not edit by hand. -->",
        "",
        "This is the human-first view of `derivatives.yaml`, which remains the machine-readable",
        "source of truth. Obligation families use the production quantity tokens ln_Psi,",
        "e_L, e_L_t, and e_L_n; guarantee names stay descriptive.",
        "",
        "The ledger separates CI evidence collection from contract authority. Actual pass/fail",
        "certification belongs to each CI run and is not stored as a timeless repository status.",
        "",
        "## Guarantees",
        "",
        "| Guarantee | What must be true | Required coverage | CI evidence collection |",
        "| --- | --- | ---: | --- |",
    ]
    for guarantee_id, coverage, evidence_state in rows:
        guarantee = guarantees[guarantee_id]
        plain = " ".join(guarantee["plain_language"].split())
        evidence_display = {"connected": "Connected"}[evidence_state]
        lines.append(f"| {guarantee['name']} | {plain} | {coverage} | {evidence_display} |")

    lines.extend(
        [
            "",
            "## Per-run certification",
            "",
            "The repository ledger does not claim that an unspecified or latest CI run passed.",
            "Each CI run independently produces `gdd-derivative-completion-certificate` with",
            "result `complete`, or fails without a completion certificate. Missing or modified",
            "receipts make that run fail certification.",
            "",
            "## Derivative coordinates",
            "",
            "| Contract term | Meaning |",
            "| --- | --- |",
            "| Wavefunction parameters | Determinant, orbital/basis, Jastrow, and DTN parameters |",
            "| Electron positions | Derivatives with respect to spin-up and spin-down electron coordinates |",
            "| Ion positions | Force-related derivatives with respect to nuclear coordinates |",
            "| Molecular affine response | Simultaneous affine deformation of molecular electron and nuclear coordinates; an implementation capability, not a stress observable |",
            "| Cell strain | Periodic-cell deformation derivatives used for physical stress |",
            "",
            "The conventional and load-balanced GFMC/LRDMC algorithms are separate production",
            "consumers. Their local energies may share an implementation or a reviewed identity,",
            "but each algorithm retains its own guarantee and route evidence.",
            "",
            "## What this ledger does not claim",
            "",
            "- Mixed-precision smoke evidence is not a numerical correctness oracle.",
            "- Rust/XLA agreement alone is not proof of physical correctness.",
            "- Finite differences are independent sentinels, not the full wavefunction cross product.",
            "- Fast-update equivalence and the absence of a full-compute fallback require a separate guarantee.",
            "- Branching-weight derivatives are outside the current derivative contract.",
            "",
        ]
    )
    lines.extend(["## Continuous-matrix coverage scope", "", "| Scope | Meaning |", "| --- | --- |"])
    _, scope_rows = continuous_matrix_scope()
    lines.extend(f"| {name} | {meaning} |" for name, meaning in scope_rows)
    lines.append("")
    lines.extend(reference_form_table(integration["gfmc_broad_matrix"]))
    return "\n".join(lines).rstrip("\n") + "\n"


def navigation() -> str:
    """Keep the generated local entry independent of names and coverage counts."""
    return (
        "# Derivative guarantee ledger\n\n"
        "<!-- Generated by scripts/render_gdd_derivatives_ledger.py. Do not edit by hand. -->\n\n"
        "Read the local generated [per-guarantee audit dossiers](../../ledger/guarantees/derivatives/) "
        "and [authored requirements](derivatives.yaml).\n\n"
        "Run `python scripts/gdd_sync.py` for the complete current derivative requirements "
        "view in `gdd/derived/derivatives-requirements.md`. CI regenerates that full view, including "
        "the guarantee table, coverage scope and reference methods. It is not an execution certificate.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compatibility alias: refresh ignored display outputs")
    parser.parse_args()
    from gdd.engine.guarantee_store import load_module

    contract = load_module(CONTRACT_PATH)
    GENERATED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_OUTPUT_PATH.write_text(render(contract), encoding="utf-8")
    expected = navigation()
    OUTPUT_PATH.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
