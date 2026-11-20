"""Fixture-coverage page renderer for the analytic-derivative matrix.

Reads the pairwise-selected cell list produced by ``_generated_matrix_base.py``
(``_MATRIX``, ``_buildable``, ``_STEM_MAP``, ...) and renders, as ``generate()``, a
per-boundary-condition coverage map of which analytic-derivative "kinds" (Det / basis /
Jastrow param blocks on the c-axis; ln_Psi vs e_L on the geometry axes) are asserted for
which committed fixture, together with the numerical-verification ledgers. It does NOT
re-derive coverage; it only reads the matrix module's own introspected structures and
renders them. ``scripts/render_gdd_ssot_ledgers.py`` embeds its numerical-verification
sections in the derivative guarantee dossiers.

Deterministic: rendering an unchanged repository state gives byte-identical text.
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

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jqmc.derivatives._capability import NO_BASIS_PARAM_AO
from tests.test_jqmc_kernel import _generated_matrix_base as _mtx
from tests.test_jqmc_kernel.analytic_coverage_registration import (
    APPROVED_GFMC_MATRIX_SHA256,
    APPROVED_MATRIX_SHA256,
    NUMERICAL_CASES,
    REQUIRED_CASES,
    family_completion_issues,
    gfmc_matrix_snapshot_matches,
    matrix_snapshot_matches,
    registration_diff,
    requirements_by_id,
)

# The matrix generation moved to the ``_generated_matrix_base`` shared module (the monolithic
# test file was split per boundary condition); the full generated cell list is rebuilt here.
_MTX_MATRIX = _mtx._generate_matrix()

_BC_ORDER = ("mol", "pbc_gamma", "twist", "k_average")
_BC_LABEL = {
    "mol": "Molecule (mol)",
    "pbc_gamma": "Crystal / k=Gamma (pbc_gamma)",
    "twist": "Crystal / twist k!=0 (twist)",
    "k_average": "Crystal / k-average (k_average)",
}
# AO-token display labels. The generic "composite" leaf is decomposed to the ONE composition
# actually used, pGTO(+)PW (other composite patterns exist in the code but are not exercised by
# the matrix). If another composite pattern is added, extend this map (or derive the child
# families from the fixture's AOs_composite_data.children).
_AO_LABEL = {
    "composite": "composite (pGTO+PW)",
}
# Boundary conditions that are NOT yet a grad-matrix axis: rendered as a PENDING note instead of
# tables (no _MATRIX cells exist for them). k-average is covered at the integration level.
_PENDING_BC_NOTE = {
    "k_average": (
        "PENDING: k-average (twist-averaged) derivatives are not yet a grad-matrix axis. The "
        "matrix covers single-twist (k!=0) per-k-point derivatives; a k-average derivative is the "
        "mean of the per-twist derivatives (each already covered under `twist`). Twist-averaging is "
        "currently exercised at the integration level (test_integration_qe_twist_average, "
        "test_integration_gfmc_twist_average), not per-cell in the grad matrix."
    ),
}
# c-axis param-owner block -> (matrix jastrow token or None, column label)
_JAS_BLOCKS = (("J1", "J1"), ("J2", "J2"), ("J3poly", "J3"), ("DTN", "DTN"))

# Machine-readable family states shared by the generated ledger and its CI completion gate.
# ln_Psi/e_L are derived, never manually flipped: the manifest snapshot, one-to-one
# registrations, builder availability, proof kind, required precision lanes, and
# leaf-liveness declarations must all be complete. Runtime assertion success is
# supplied separately by the full and mixed pytest lanes and their exact receipt
# completion certificate. Both lanes carry blocking analytical correctness;
# mixed smoke evidence remains supplemental rather than a correctness substitute.
FAMILY_COMPLETION_ISSUES = {family: family_completion_issues(family) for family in ("ln_Psi", "e_L", "e_L_t", "e_L_n")}
FAMILY_STATES = {
    family: "finished" if not FAMILY_COMPLETION_ISSUES[family] else "partial" for family in FAMILY_COMPLETION_ISSUES
}
CI_REQUIRED_FINISHED_FAMILIES = ("ln_Psi", "e_L", "e_L_t", "e_L_n")

# Type registries are DECLARED in gdd.engine.derivative_inventory and verified against
# production introspection by the GDD structural gate. A new concrete class fails that
# gate until the declared registry (and this generated page) is reviewed.

# Production-first verification inventory.  These rows describe behavior-changing branches in the
# implementation, independently of which fixtures/tests happen to exist.  They are the reviewed
# source from which the shared continuous and GFMC_t/n numerical universes will be generated.
PRODUCTION_ENTRY_INVENTORY = (
    (
        "ln_Psi",
        "VMC/MCMC + GFMC/LRDMC",
        "continuous `ln_Psi`",
        "jqmc.derivatives",
        "compute_grad_ln_Psi",
        "jqmc.derivatives.assemble",
        "compute_grad_ln_Psi",
        "VMC: `c,r,R,eps`; GFMC: `r,R`, plus `eps` for periodic stress",
    ),
    (
        "e_L",
        "VMC/MCMC + GFMC/LRDMC",
        "continuous `e_L`",
        "jqmc.derivatives",
        "compute_grad_e_L",
        "jqmc.derivatives.assemble",
        "compute_grad_e_L",
        "VMC: `c,r,R,eps`; GFMC component: `r,R,eps` (driver `eps`: PBC stress)",
    ),
    (
        "e_L_t",
        "GFMC/LRDMC",
        "`e_L_t`",
        "jqmc.derivatives.gfmc",
        "grad_e_L_t_wrt_positions",
        "jqmc.derivatives.gfmc",
        "grad_e_L_wrt_positions",
        "`r,R`",
    ),
    (
        "e_L_t",
        "GFMC/LRDMC",
        "`e_L_t`",
        "jqmc.derivatives.gfmc",
        "grad_e_L_t_wrt_strain",
        "jqmc.derivatives.gfmc",
        "grad_e_L_wrt_strain",
        "`eps` (driver consumes it for PBC stress)",
    ),
    (
        "e_L_n",
        "GFMC/LRDMC",
        "`e_L_n`",
        "jqmc.derivatives.gfmc",
        "grad_e_L_n_wrt_positions",
        "jqmc.derivatives.gfmc",
        "grad_e_L_wrt_positions",
        "`r,R`",
    ),
    (
        "e_L_n",
        "GFMC/LRDMC",
        "`e_L_n`",
        "jqmc.derivatives.gfmc",
        "grad_e_L_n_wrt_strain",
        "jqmc.derivatives.gfmc",
        "grad_e_L_wrt_strain",
        "`eps` (driver consumes it for PBC stress)",
    ),
)

# The reviewed B/C/P/R inventory tables were promoted into the GDD engine
# (requirement side); they are re-exported here for the ledger renderer and gates.
from gdd.engine.derivative_inventory import (  # noqa: F401  (re-exported inventories)
    PRODUCTION_AO_LEAF_CLASSES,
    PRODUCTION_DTN_CHANNEL_FLAGS,
    PRODUCTION_GEMINAL_CLASSES,
    PRODUCTION_JASTROW_CHANNEL_FIELDS,
    PARAMETER_LEAF_INVENTORY,
    PRODUCTION_BRANCH_INVENTORY,
    PRODUCTION_NUMERICAL_REGIMES,
    PRODUCTION_VALIDITY_CONSTRAINTS,
)


def _master_family_table() -> list[str]:
    """Calculation-family inventory, cross-cutting axes, and supporting-kernel registries."""
    observable_rows = [
        [
            "ln_Psi",
            "VMC/MCMC + GFMC/LRDMC",
            "continuous `ln_Psi` (observable / shared foundation)",
            "VMC: `c,r,R,eps`; GFMC: `r,R`, plus `eps` for periodic stress",
            "shared continuous tables below",
            FAMILY_STATES["ln_Psi"],
        ],
        [
            "e_L",
            "VMC/MCMC + GFMC/LRDMC",
            "continuous `e_L` (observable / GFMC component)",
            "VMC: `c,r,R,eps`; GFMC component: `r,R,eps` (driver `eps`: PBC stress)",
            "shared continuous tables below",
            FAMILY_STATES["e_L"],
        ],
        [
            "e_L_t",
            "GFMC/LRDMC",
            "`e_L_t` estimator",
            "`r,R`; `eps` (driver: PBC stress)",
            "GFMC expanded ledger below",
            FAMILY_STATES["e_L_t"],
        ],
        [
            "e_L_n",
            "GFMC/LRDMC",
            "`e_L_n` estimator",
            "`r,R`; `eps` (driver: PBC stress)",
            "GFMC expanded ledger below",
            FAMILY_STATES["e_L_n"],
        ],
    ]
    lines = [
        "## Verification-target inventory",
        "",
        "The top-level units are differentiated quantities/estimators plus their calculation",
        "consumers. ln_Psi/e_L are shared continuous foundations, not VMC-only quantities. Physical",
        "choices such as `dltmove` are axes inside those family ledgers, not peer families. `fast`",
        "is an implementation-route guardrail, not an analytical-derivative coverage axis.",
        "Kernel/component checks are tracked separately because they support several observables.",
        "Independent AST discovery scans the production boundary and the GDD static gate requires",
        "every observed derivative entry or adapter to have an exact SSoT disposition.",
        "",
        "A family row becomes complete only when its own ledger declares the valid universe,",
        "accounts for every valid cell by direct AD comparison or documented identity, checks output",
        "liveness, and rejects unsupported cells explicitly. Targeted directional FD remains a",
        "sentinel for high-risk seams rather than a cross-product axis.",
        "Here `finished` means that this static obligation and registration inventory is complete; it",
        "does not claim that the current run passed. Runtime authority is the exact full/mixed receipt",
        "completion certificate, and no certificate is issued when any listed pytest case is red.",
        "",
        "### Verification families and calculation consumers",
        "",
        _md_row(["ID", "Calculation consumers", "Differentiated quantity / role", "Consumed axes", "Detailed ledger", "State"]),
        _md_row(["---"] * 6),
    ]
    lines.extend(_md_row(row) for row in observable_rows)
    lines += [
        "",
        "### Production entry points in the current inventory",
        "",
        "These SSoT-declared public entries are checked against independently discovered production references.",
        "",
        _md_row(["ID", "Consumers", "Quantity", "Public API", "Analytical implementation", "Consumed axes"]),
        _md_row(["---"] * 6),
    ]
    lines.extend(
        _md_row(
            [
                family,
                consumers,
                quantity,
                f"`{public_module}.{public_symbol}`",
                f"`{impl_module}.{impl_symbol}`",
                axes,
            ]
        )
        for (
            family,
            consumers,
            quantity,
            public_module,
            public_symbol,
            impl_module,
            impl_symbol,
            axes,
        ) in PRODUCTION_ENTRY_INVENTORY
    )
    lines += [
        "",
        "### Production branch inventory for ln_Psi/e_L/e_L_t/e_L_n",
        "",
        "This is the production-first checklist from which valid numerical cells must be derived.",
        "A branch row is not itself evidence of coverage. Numerical obligations and runtime receipts",
        "must account for the reviewed values independently of this human-readable checklist.",
        "",
        _md_row(["ID", "Applies to", "Branch class", "Values / regimes", "Production basis"]),
        _md_row(["---"] * 5),
    ]
    lines.extend(_md_row(list(row)) for row in PRODUCTION_BRANCH_INVENTORY)
    lines += [
        "",
        "### Validity and N/A constraints",
        "",
        "These constraints classify combinations; they must never be implemented as silent filters.",
        "Every excluded cell needs one of these reason codes or a newly reviewed constraint.",
        "",
        _md_row(["ID", "Constraint", "Rule"]),
        _md_row(["---"] * 3),
    ]
    lines.extend(_md_row(list(row)) for row in PRODUCTION_VALIDITY_CONSTRAINTS)
    lines += [
        "",
        "### ln_Psi/e_L c-axis output-leaf contract",
        "",
        "A combined pytree equality is insufficient: every conditionally expected stored leaf must",
        "be present and demonstrably live. This table is the leaf-level guard against an unwritten",
        "`lambda_diag`-class defect.",
        "",
        _md_row(["ID", "Input flag", "Condition / owner", "Expected published leaf or block", "Special contract"]),
        _md_row(["---"] * 5),
    ]
    lines.extend(_md_row(list(row)) for row in PARAMETER_LEAF_INVENTORY)
    lines += [
        "",
        "### Value-dependent numerical regimes",
        "",
        "These regimes are not datatype branches, but ordinary fixtures can miss them even when all",
        "discrete labels are present.",
        "",
        _md_row(["ID", "Applies to", "Required regime", "Why it is distinct"]),
        _md_row(["---"] * 4),
    ]
    lines.extend(_md_row(list(row)) for row in PRODUCTION_NUMERICAL_REGIMES)
    lines += [
        "",
        "### Deliberate scope boundaries",
        "",
        "- `fast`/running-state is a standard-equivalence and no-fresh-SVD implementation contract,",
        "  not another analytical-versus-autograd universe.",
        "- k-average is a reduction over covered per-twist derivatives and requires a separate",
        "  aggregation contract; it is not a single-Hamiltonian ln_Psi/e_L branch.",
        "- MPC is `physical=no` on current main and is therefore an N/A contract, not a e_L term.",
        "- GFMC_t versus GFMC_n branching-weight differentiation is outside e_L_t/e_L_n: their",
        "  differentiated local energy is shared; only the driver projection weights differ.",
    ]
    lines.append("")
    return lines


def _requirement_registration_ledger() -> list[str]:
    """Render the reviewed requirement-vs-pytest registration comparison."""
    diff = registration_diff()
    lookup = requirements_by_id()
    families = ("ln_Psi", "e_L", "e_L_t", "e_L_n")
    required_family_counts = {
        family: sum(requirement.family == family for requirement in REQUIRED_CASES) for family in families
    }
    missing_family_counts = {
        family: sum(lookup[requirement_id].family == family for requirement_id in diff.missing) for family in families
    }
    lines = [
        "## Analytical-derivative requirement-registration ledger",
        "",
        "`REQUIRED_CASES` is the reviewed obligation set; `NUMERICAL_CASES` is discovered",
        "independently from the actual broad-matrix and targeted-evidence pytest parametrizations.",
        "A family state is derived from snapshot, one-to-one registration, builder, proof, required",
        "precision-lane, and liveness-contract checks. Separate full and mixed CI lanes supply the",
        "runtime analytical-correctness receipts; no state is manually flipped. Mixed shape/finite",
        "smoke evidence is supplemental and cannot replace the mixed analytical-oracle receipts.",
        "",
        f"- Reviewed ln_Psi/e_L matrix snapshot: `{APPROVED_MATRIX_SHA256}` ({'matches' if matrix_snapshot_matches() else 'MISMATCH'})",
        f"- Reviewed e_L_t/e_L_n matrix snapshot: `{APPROVED_GFMC_MATRIX_SHA256}` "
        f"({'matches' if gfmc_matrix_snapshot_matches() else 'MISMATCH'})",
        f"- Required obligations: {len(REQUIRED_CASES)} ("
        + ", ".join(f"{family}={required_family_counts[family]}" for family in families)
        + ")",
        f"- Registered pytest cases: {len(NUMERICAL_CASES)}",
        f"- Missing registrations: {len(diff.missing)} ("
        + ", ".join(f"{family}={missing_family_counts[family]}" for family in families)
        + ")",
        f"- Extra / duplicate / unavailable registrations: "
        f"{len(diff.extra)} / {len(diff.duplicates)} / {len(diff.unavailable)}",
        f"- Precision / proof / liveness mismatches: "
        f"{len(diff.precision_mismatch)} / {len(diff.proof_mismatch)} / {len(diff.missing_liveness)}",
        "",
        "### Missing requirements",
        "",
        _md_row(["Requirement ID", "Category", "Expected", "Primary proof", "Required lane", "Leaf/liveness contract"]),
        _md_row(["---"] * 6),
    ]
    for requirement_id in diff.missing:
        requirement = lookup[requirement_id]
        leaves = ", ".join(f"`{leaf}`" for leaf in requirement.expected_leaves) or "-"
        if requirement.liveness_required:
            leaves += "; live"
        lines.append(
            _md_row(
                [
                    f"`{requirement.id}`",
                    requirement.category,
                    requirement.expected_outcome.value,
                    requirement.proof.value if requirement.proof is not None else "-",
                    ", ".join(requirement.required_precision),
                    leaves,
                ]
            )
        )
    if not diff.missing:
        lines.append(_md_row(["(none)", "-", "-", "-", "-", "-"]))
    lines.append("")
    return lines


def _gfmc_numerical_ledger(family: str) -> list[str]:
    """Render every reviewed e_L_t/e_L_n obligation and its actual pytest registration."""
    lookup = {case.requirement_id: case for case in NUMERICAL_CASES if case.requirement_id.startswith(f"{family}|")}
    requirements = sorted(
        (requirement for requirement in REQUIRED_CASES if requirement.family == family),
        key=lambda requirement: (requirement.category, requirement.description, requirement.id),
    )
    title = "GFMC_t / LRDMC `e_L_t`" if family == "e_L_t" else "GFMC_n / LRDMC `e_L_n`"
    proof_note = (
        "e_L_t rows use the standalone driver-equivalent `e_L_t` forward as the JAX AD oracle."
        if family == "e_L_t"
        else "e_L_n contains independent AD rows plus explicit B31 n==t identity rows; the identity is limited to the differentiated `use_fast_update=False` local energy."
    )
    lines = [
        f"### {family}: {title}",
        "",
        proof_note,
        "Each expanded parametrization/output target is a separate reviewed obligation; the table is not a function-name-only checklist.",
        f"`{family}-N` is an obligation-row label, not a fixture identifier. `Base fixture(s)` names",
        "the committed input behind the row; setup variants remain visible in `Reviewed case`.",
        "",
        _md_row(
            [
                "Obligation",
                "Category",
                "Source",
                "Proof",
                "Required lane",
                "Base fixture(s)",
                "Reviewed case",
                "Registered pytest evidence",
            ]
        ),
        _md_row(["---"] * 8),
    ]
    for index, requirement in enumerate(requirements, 1):
        case = lookup.get(requirement.id)
        nodeid = case.pytest_evidence if case is not None else "MISSING"
        escaped_nodeid = nodeid.replace("|", "\\|")
        lines.append(
            _md_row(
                [
                    f"{family}-{index}",
                    requirement.category,
                    requirement.source_id,
                    requirement.proof.value if requirement.proof is not None else requirement.expected_outcome.value,
                    ", ".join(requirement.required_precision),
                    _gfmc_base_fixtures(requirement.id, nodeid),
                    requirement.description.replace("|", "\\|"),
                    f"`{escaped_nodeid}`" if case is not None else "**MISSING**",
                ]
            )
        )
    lines += [
        "",
        f"State: **{FAMILY_STATES[family]}**; obligations={len(requirements)}; "
        f"completion issues={len(FAMILY_COMPLETION_ISSUES[family])}.",
        "",
    ]
    return lines


_GFMC_FIXED_FIXTURES = {
    "test_gfmc_t_force_analytic_vs_autograd_twist": "lithium_k025.h5",
    "test_gfmc_t_stress_analytic_vs_autograd_twist": "lithium_k025.h5",
    "test_gfmc_t_force_analytic_vs_autograd_twist_dtn": "lithium_k025.h5",
    "test_gfmc_t_stress_analytic_vs_autograd_twist_dtn": "lithium_k025.h5",
    "test_gfmc_t_force_analytic_vs_autograd_twist_ae": "QE:H-ae_k025",
    "test_gfmc_t_stress_analytic_vs_autograd_twist_ae": "QE:H-ae_k025",
    "test_gfmc_t_force_stress_analytic_vs_autograd_dtn_pbc": "lithium.h5",
    "test_gfmc_n_force_analytic_vs_autograd_twist_dtn": "lithium_k025.h5",
    "test_gfmc_n_stress_analytic_vs_autograd_twist_dtn": "lithium_k025.h5",
    "test_gfmc_n_equals_t_force_and_stress": "Li_ae_ccpvdz_cart.h5",
    "test_gfmc_n_equals_t_twist": "lithium_k025.h5",
    "test_gfmc_n_force_stress_dtn_equals_t": "Li_ae_ccpvdz_cart.h5",
}

_GFMC_TARGETED_FIXTURES = {
    "output|B24": "H2_ae_ccpvdz_cart.h5",
    "error|C10|unknown-non-local-move": "H2_ae_ccpvdz_cart.h5; H2_ecp_ccpvdz_cart.h5",
    "regime|R01": "H2_ae_ccpvdz_cart.h5",
    "regime|R02": "N_ae_ccpvdz_cart.h5",
    "regime|R04": "lithium_k025.h5",
    "regime|R05": "H2_ae_ccpvdz_cart.h5; N_ae_ccpvdz_cart.h5",
    "regime|R06": "N2_ecp_ccpvtz_cart.h5",
    "regime|R07": "N_ae_ccpvdz_cart.h5",
    "regime|R08": "N_ae_ccpvdz_cart.h5",
    "regime|R09": "H2_ae_ccpvdz_cart.h5; N_ae_ccpvdz_cart.h5",
    "regime|R10": "Li_ae_ccpvdz_cart.h5",
}


def _gfmc_base_fixtures(requirement_id: str, nodeid: str) -> str:
    """Render fixture provenance without confusing it with an obligation label."""
    stem_match = re.search(r"stem=(['\"])(?P<stem>.+?)\1", nodeid)
    if stem_match is not None:
        return stem_match.group("stem")

    function_match = re.search(r"::(?P<function>[^\[\s]+)", nodeid)
    if function_match is not None and function_match.group("function") in _GFMC_FIXED_FIXTURES:
        return _GFMC_FIXED_FIXTURES[function_match.group("function")]

    targeted_key = requirement_id.split("|", 1)[1]
    if targeted_key in _GFMC_TARGETED_FIXTURES:
        return _GFMC_TARGETED_FIXTURES[targeted_key]
    if nodeid == "MISSING":
        return "MISSING"
    raise ValueError(f"no GFMC base-fixture provenance for {requirement_id}: {nodeid}")


def _no_basis_ao(ao: str) -> bool:
    """True iff the AO token has no radial basis parameters (plane-wave / B-spline)."""
    if f"AOs_{ao}_data" in NO_BASIS_PARAM_AO:
        return True
    return "plane_wave" in ao or "bspline" in ao


def _ao_label(ao: str) -> str:
    """Display label for an AO token (decomposes ``composite`` to its concrete pairing)."""
    return _AO_LABEL.get(ao, ao)


# Geminal-representation display labels. ``AGP`` means the dense-AO form;
# ``AGP_MO`` is an explicit ``AGP_data`` fixture whose orbital leaves are ``MOs_data``.
_GEM_LABEL = {"AGP": "AGP (dense-AO)", "AGP_MO": "AGP (MO)"}


def _gem_label(gem: str) -> str:
    return _GEM_LABEL.get(gem, gem)


def _fixtures_for_bc(bc: str) -> list[tuple[str, str, str]]:
    """Sorted (ao, ecp, gem) fixture rows that appear in ``_MATRIX`` for this boundary condition.
    The geminal representation is a matrix axis (gem in {SD, AGPn, AGP, AGP_MO}), so it is a row sub-axis.
    """
    seen = set()
    for cell, _heavy in _MTX_MATRIX:
        ao, _jas, ecp, cbc, gem = _mtx._split_cell(cell)
        if cbc == bc:
            seen.add((ao, ecp, gem))
    return sorted(seen)


def _fixture_stem(ao: str, ecp: str, bc: str, fallback_cell: tuple) -> str:
    stem = _mtx._STEM_MAP.get((ao, bc, ecp))
    return stem if stem is not None else _mtx._cell_id(fallback_cell)


def _row_kinds(ao: str, ecp: str, gem: str, bc: str):
    """``(cells, built, jas_present)`` for one (ao, ecp, gem, bc) fixture row."""
    cells = []
    for c, _h in _MTX_MATRIX:
        sao, _sj, secp, sbc, sgem = _mtx._split_cell(c)
        if sao == ao and secp == ecp and sgem == gem and sbc == bc:
            cells.append(c)
    built = [c for c in cells if _mtx._buildable(c)]
    jas_present = {_mtx._split_cell(c)[1] for c in built}
    return cells, built, jas_present


def _md_row(cols: list[str]) -> str:
    return "| " + " | ".join(cols) + " |"


def _parameter_table(bc: str, fixtures: list[tuple[str, str, str]], no_builder_notes: dict[str, str]) -> list[str]:
    header = ["AO", "Gem", "ECP", "Det/lnPsi", "Det/eL", "Basis/lnPsi", "Basis/eL"]
    for _tok, label in _JAS_BLOCKS:
        header += [f"{label}/lnPsi", f"{label}/eL"]
    header.append("Base fixture")
    lines = [_md_row(header), _md_row(["---"] * len(header))]
    for ao, ecp, gem in fixtures:
        cells, built, jas_present = _row_kinds(ao, ecp, gem, bc)
        if not built:
            note_id = f"nb-{ao}-{ecp}-{gem}-{bc}"
            no_builder_notes[note_id] = _mtx._no_builder_reason(cells[0])
            fixture_col = f"no-builder[^{note_id}]"
            row = [_ao_label(ao), _gem_label(gem), ecp] + ["-"] * (len(header) - 4) + [fixture_col]
            lines.append(_md_row(row))
            continue
        basis_val = "n/a" if _no_basis_ao(ao) else "Y"
        row = [_ao_label(ao), _gem_label(gem), ecp, "Y", "Y", basis_val, basis_val]
        for tok, _label in _JAS_BLOCKS:
            val = "Y" if tok in jas_present else "-"
            row += [val, val]
        row.append(_fixture_stem(ao, ecp, bc, cells[0]))
        lines.append(_md_row(row))
    return lines


def _geometry_table(bc: str, fixtures: list[tuple[str, str, str]], no_builder_notes: dict[str, str]) -> list[str]:
    header = [
        "AO",
        "Gem",
        "ECP",
        "r/lnPsi",
        "r/eL",
        "R/lnPsi",
        "R/eL",
        "eps/lnPsi",
        "eps/eL",
        "Base fixture",
    ]
    lines = [_md_row(header), _md_row(["---"] * len(header))]
    for ao, ecp, gem in fixtures:
        cells, built, _jas_present = _row_kinds(ao, ecp, gem, bc)
        if not built:
            note_id = f"nb-{ao}-{ecp}-{gem}-{bc}"
            no_builder_notes[note_id] = _mtx._no_builder_reason(cells[0])
            fixture_col = f"no-builder[^{note_id}]"
            row = [_ao_label(ao), _gem_label(gem), ecp] + ["-"] * (len(header) - 4) + [fixture_col]
            lines.append(_md_row(row))
            continue
        row = [
            _ao_label(ao),
            _gem_label(gem),
            ecp,
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            _fixture_stem(ao, ecp, bc, cells[0]),
        ]
        lines.append(_md_row(row))
    return lines


def _section(bc: str) -> list[str]:
    fixtures = _fixtures_for_bc(bc)
    if not fixtures:
        note = _PENDING_BC_NOTE.get(bc, "PENDING: not yet a grad-matrix axis (no committed fixtures).")
        return [f"### {_BC_LABEL[bc]}", "", f"_{note}_", ""]
    no_builder_notes: dict[str, str] = {}
    lines = [f"### {_BC_LABEL[bc]}", ""]
    lines.append("#### c-axis (parameter gradients)")
    lines.append("")
    lines += _parameter_table(bc, fixtures, no_builder_notes)
    lines.append("")
    lines.append("#### geometry (r / R / eps)")
    lines.append("")
    lines += _geometry_table(bc, fixtures, no_builder_notes)
    lines.append("")
    if no_builder_notes:
        for note_id in sorted(no_builder_notes):
            lines.append(f"[^{note_id}]: {no_builder_notes[note_id]}")
        lines.append("")
    return lines


def _fixture_debt() -> list[str]:
    lines = ["### Fixture debt", "", "Cells in `_MATRIX` with no committed builder:", ""]
    no_builder_cells = sorted(c for c, _h in _MTX_MATRIX if not _mtx._buildable(c))
    if not no_builder_cells:
        lines.append("- (none -- every generated cell has a committed builder)")
    for cell in no_builder_cells:
        lines.append(f"- `{_mtx._cell_id(cell)}` -- {_mtx._no_builder_reason(cell)}")
    lines.append("")
    return lines


def _legend() -> list[str]:
    return [
        "## Legend",
        "",
        "- `Base fixture` -- committed HDF5 or QE source used to construct the row. A shared-continuous",
        "  row groups matrix cells with the same boundary/AO/ECP/geminal base but may exercise several",
        "  Jastrow variants; it is therefore provenance, not a unique numerical-case identifier.",
        "- `Gem` -- geminal representation (a matrix axis): `SD` (single determinant), `AGPn` (diagonal",
        "  natural-orbital form), `AGP (dense-AO)` (a full dense pairing matrix in the non-orthogonal AO",
        "  basis), and `AGP (MO)` (a dense pairing matrix over explicit MO leaves). The stored matrix",
        "  tokens are `AGP` and `AGP_MO`, respectively. Current dense-AO",
        "  cells are admitted only where the fixture overlap is well-conditioned; that test policy",
        "  is not a production-validity restriction.",
        "- `Y` -- required assertion: this kind is exercised by the pairwise matrix (a buildable",
        "  cell compares analytical output with its independent oracle). This map is derived",
        "  STATICALLY from the matrix parametrization, not from a test run, so `Y` does not claim",
        "  that the current workspace or commit passed. Runtime authority is the exact full/mixed",
        "  receipt completion certificate.",
        "- `n/a` -- structurally absent (e.g. a plane-wave / B-spline AO family has no radial",
        "  basis parameters, so there is no `Basis` gradient to assert).",
        "- `-` -- gap: NOT asserted here (unverified). Either this (fixture x Jastrow) pair was not",
        "  picked by the pairwise covering array, or (in a `no-builder` row) no committed fixture",
        "  exists for this (ao, ecp, bc) at all. This does NOT mean the analytic derivative is",
        "  unimplemented -- the kernels are shared across fixtures, so a `-` block is generally",
        "  implemented and exercised on OTHER fixtures; it is simply not verified on THIS one.",
        "  (An actually-unimplemented analytic path does not show as `-`: with the capability gate",
        "  removed it RAISES, surfacing as a failing test / red CI, not a gap glyph.) Treat `-` as",
        "  'unverified', not 'known-good' -- an untested combination can still hide a real gap.",
        "",
        "**Joint-verification caveat (c-axis tables):** the c-axis param-block columns",
        "(`Det`, `Basis`, `J1`, `J2`, `J3`, `DTN`) are verified JOINTLY, not in isolation --",
        "the matrix asserts ONE combined c-axis gradient per cell, under a mask that turns on",
        "exactly the row's active blocks (`Det` is always active; `Basis` is active whenever the",
        "AO family carries radial parameters; a Jastrow block is active iff that channel is",
        'attached for the cell). A `Y` therefore means "this block is active in the required',
        'combined c-axis assertion for this fixture", not "this block was isolated and verified',
        'alone". The geometry axes and the ln_Psi / e_L quantity split ARE separate',
        "test cells (one `test_matrix_cell` invocation per (quantity, axis) pair).",
        "",
        "Generated by `tests/test_jqmc_kernel/gen_coverage_readme.py`; regenerate after",
        "fixture, requirement, or pytest-registration changes. Exact reviewed universes are",
        "identified by the ln_Psi/e_L and e_L_t/e_L_n snapshot hashes below.",
        "",
    ]


def generate() -> str:
    lines: list[str] = [
        "# Analytic-derivative fixture coverage",
        "",
        "This is a machine-generated coverage map for the generated fixture matrix",
        "(`tests/test_jqmc_kernel/_generated_matrix_base.py`). It shows, at a glance, which",
        'analytic-derivative "kinds" -- c-axis parameter blocks (Det / Basis / J1 / J2 / J3 / DTN)',
        "and geometry axes (r / R / eps) split by quantity (`ln_Psi` vs `e_L`) -- are covered by",
        "which committed fixture, grouped by boundary condition. It is a fixture-debt tracker for",
        'the ongoing "all grads analytic" effort, not a substitute for reading the matrix test',
        "itself.",
        "",
    ]
    lines += _legend()
    lines += _master_family_table()
    lines += _requirement_registration_ledger()
    lines += [
        "## Numerical-verification ledgers",
        "",
    ]
    lines += _gfmc_numerical_ledger("e_L_t")
    lines += _gfmc_numerical_ledger("e_L_n")
    lines += [
        "### Shared continuous foundation (`ln_Psi` and `e_L`)",
        "",
        "This is the first family-level ledger. VMC/MCMC consumes both quantities directly;",
        "GFMC/LRDMC reuses their `r/R/eps` derivatives at the base walker and shifted LRDMC",
        "configurations inside drift, Pulay, mesh S-prime, and stress reductions. The ledger records",
        "direct analytical-versus-JAX-autograd fixture cells for the shared continuous kernels; it",
        "does not by itself cover the assembled GFMC_t/n reductions.",
        "",
    ]
    for bc in _BC_ORDER:
        lines += _section(bc)
    lines += _fixture_debt()
    return "\n".join(lines).rstrip() + "\n"
