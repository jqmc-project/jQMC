"""CI gate for analytical-derivative family-ledger completion."""

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

import re
from pathlib import Path
from unittest.mock import patch

import pytest

import jqmc.derivatives as derivatives_api
from gdd.engine.production_entry_discovery import discover_external_derivative_references
from tests.test_jqmc_kernel import test_mixed_precision as _mixed_precision
from tests.test_jqmc_kernel import test_analytic_coverage_evidence as _targeted_evidence
from tests.test_jqmc_kernel.analytic_coverage_registration import (
    REQUIRED_CASES,
    family_completion_issues,
    gfmc_broad_matrix_covers_both_precisions,
    gfmc_function_inventory_diff,
    generated_matrix_binding_issues,
    matrix_modules_cover_both_precisions,
    registration_diff,
    requirements_by_id,
)
from tests.test_jqmc_kernel.gen_coverage_readme import (
    CI_REQUIRED_FINISHED_FAMILIES,
    FAMILY_STATES,
    PARAMETER_LEAF_INVENTORY,
    PRODUCTION_AO_LEAF_CLASSES,
    PRODUCTION_BRANCH_INVENTORY,
    PRODUCTION_DTN_CHANNEL_FLAGS,
    PRODUCTION_ENTRY_INVENTORY,
    PRODUCTION_GEMINAL_CLASSES,
    PRODUCTION_JASTROW_CHANNEL_FIELDS,
    PRODUCTION_NUMERICAL_REGIMES,
    PRODUCTION_VALIDITY_CONSTRAINTS,
)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir, pytest.mark.design_pin]

_ROOT = Path(__file__).resolve().parents[2]


def test_production_verification_inventory_is_well_formed():
    branch_ids = tuple(row[0] for row in PRODUCTION_BRANCH_INVENTORY)
    constraint_ids = tuple(row[0] for row in PRODUCTION_VALIDITY_CONSTRAINTS)
    leaf_ids = tuple(row[0] for row in PARAMETER_LEAF_INVENTORY)
    regime_ids = tuple(row[0] for row in PRODUCTION_NUMERICAL_REGIMES)
    assert branch_ids == tuple(f"B{i:02d}" for i in range(1, 32))
    assert constraint_ids == tuple(f"C{i:02d}" for i in range(1, 13))
    assert leaf_ids == tuple(f"P{i:02d}" for i in range(1, 16))
    assert regime_ids == tuple(f"R{i:02d}" for i in range(1, 11))

    known_families = set(FAMILY_STATES)
    assert {row[0] for row in PRODUCTION_ENTRY_INVENTORY} == known_families
    for inventory in (PRODUCTION_BRANCH_INVENTORY, PRODUCTION_NUMERICAL_REGIMES):
        for row in inventory:
            assert set(row[1].split(",")) <= known_families

    inventoried_flags = {row[1].strip("`") for row in PARAMETER_LEAF_INVENTORY}
    assert inventoried_flags == set(derivatives_api._PARAM_FLAG_KEYS)

    branches = {row[0]: row for row in PRODUCTION_BRANCH_INVENTORY}
    assert set(re.findall(r"`([^`]+)`", branches["B04"][3])) == set(PRODUCTION_AO_LEAF_CLASSES)
    assert set(re.findall(r"`([^`]+)`", branches["B07"][3])) == set(PRODUCTION_GEMINAL_CLASSES)
    assert set(re.findall(r"`([^`]+)`", branches["B11"][3])) == set(PRODUCTION_JASTROW_CHANNEL_FIELDS)
    assert set(re.findall(r"`([^`]+)`", branches["B15"][3])) == set(PRODUCTION_DTN_CHANNEL_FLAGS)

    manifest_leaf_ids = {requirement.source_id for requirement in REQUIRED_CASES if requirement.category == "leaf"}
    manifest_regime_families = {
        (requirement.source_id, requirement.family) for requirement in REQUIRED_CASES if requirement.category == "regime"
    }
    assert manifest_leaf_ids == set(leaf_ids)
    assert manifest_regime_families == {
        (row[0], family) for row in PRODUCTION_NUMERICAL_REGIMES for family in row[1].split(",")
    }
    known_source_ids = set(branch_ids) | set(constraint_ids) | set(leaf_ids) | set(regime_ids) | {"fixture-matrix"}
    assert {requirement.source_id for requirement in REQUIRED_CASES} <= known_source_ids


def test_production_entry_inventory_matches_discovery():
    expected_public = {
        f"{public_module}.{public_symbol}"
        for (
            _family,
            _consumers,
            _quantity,
            public_module,
            public_symbol,
            _impl_module,
            _impl_symbol,
            _axes,
        ) in PRODUCTION_ENTRY_INVENTORY
    }
    expected_implementation = {
        f"{impl_module}.{impl_symbol}"
        for (
            _family,
            _consumers,
            _quantity,
            _public_module,
            _public_symbol,
            impl_module,
            impl_symbol,
            _axes,
        ) in PRODUCTION_ENTRY_INVENTORY
    }
    observed_boundary = set(discover_external_derivative_references(_ROOT))
    assert expected_public <= observed_boundary
    for target in expected_implementation:
        module, _, symbol = target.rpartition(".")
        implementation = __import__(module, fromlist=[symbol])
        assert callable(getattr(implementation, symbol))


def test_generated_matrix_has_all_required_fixture_bindings():
    """Live required matrix cases retain their concrete input and comparison bindings."""
    assert generated_matrix_binding_issues() == ()
    assert matrix_modules_cover_both_precisions(), (
        "ln_Psi/e_L analytical-vs-autograd matrices must execute in both full and mixed precision"
    )


def test_gfmc_matrix_retains_independently_required_parameter_sets():
    """Live GFMC parameters include every independently authored scientific tuple."""
    from scripts.check_gdd_required_cases import gfmc_parameter_contracts, required_node_errors

    for _, node, binding in gfmc_parameter_contracts(_ROOT):
        assert required_node_errors(node, binding) == []
    missing, extra = gfmc_function_inventory_diff()
    assert not missing and not extra, f"GFMC evidence function inventory changed: missing={missing}, extra={extra}"
    assert gfmc_broad_matrix_covers_both_precisions(), (
        "e_L_t/e_L_n analytical-vs-autograd evidence must execute in both full and mixed precision"
    )


def test_derivative_correctness_oracles_never_switch_precision_lanes():
    """Analytical and autodiff evidence must execute under one precision policy."""
    governed_sources = (
        "tests/test_jqmc_kernel/_gdd_adapters.py",
        "tests/test_jqmc_kernel/_generated_matrix_base.py",
        "tests/test_jqmc_kernel/_grad_fidelity_harness.py",
        "tests/test_jqmc_kernel/test_analytic_coverage_evidence.py",
        "tests/test_jqmc_kernel/test_gfmc_force_analytic.py",
    )
    for relative_path in governed_sources:
        source = (_ROOT / relative_path).read_text(encoding="utf-8")
        assert "same_input_fp64_oracle" not in source, relative_path
        assert "same-input-fp64-primal-autodiff" not in source, relative_path


def test_required_case_ids_are_unique():
    assert len(requirements_by_id()) == len(REQUIRED_CASES)


@pytest.mark.gdd_synthetic_derivative_receipts(reason="replays the R03 composite with emission patched out")
def test_all_required_cases_are_registered_in_pytest():
    diff = registration_diff()
    assert diff.complete, (
        "Analytical-derivative coverage registration is incomplete. Missing requirements must receive explicit "
        "pytest evidence; extras/duplicates/unavailable cases must be resolved. "
        f"missing={diff.missing}, extra={diff.extra}, duplicates={diff.duplicates}, unavailable={diff.unavailable}, "
        f"precision={diff.precision_mismatch}, proof={diff.proof_mismatch}, liveness={diff.missing_liveness}"
    )

    class _ComponentReceipt:
        status = "passed"

        def __init__(self, obligation_id):
            self.obligation_id = obligation_id

        def validate(self):
            return None

    leaf_calls = []
    emitted = []

    def validate_leaf_without_emitting(family, scenario, *, emit_completion):
        leaf_calls.append((family, scenario, emit_completion))
        return tuple(
            _ComponentReceipt(f"{family}|leaf|{source_id}")
            for source_id in _targeted_evidence._LEAF_SOURCE_IDS_BY_SCENARIO[scenario]
        )

    with (
        patch.object(_targeted_evidence, "_run_parameter_leaf_evidence", validate_leaf_without_emitting),
        patch.object(
            _targeted_evidence,
            "emit_completion_record",
            lambda receipt, **kwargs: emitted.append((receipt, kwargs)),
        ),
    ):
        _targeted_evidence.test_numerical_regime_evidence("ln_Psi|regime|R03")

    assert len(leaf_calls) == len(_targeted_evidence._LEAF_SCENARIOS)
    assert all(emit_completion is False for _family, _scenario, emit_completion in leaf_calls)
    assert len(emitted) == 1
    composite, options = emitted[0]
    assert options == {"record_kind": "composite"}
    assert composite.obligation_id == "ln_Psi|regime|R03"
    assert len(composite.component_receipts) == 15


def test_required_analytical_derivative_families_are_finished():
    derived_issues = {family_id: family_completion_issues(family_id) for family_id in CI_REQUIRED_FINISHED_FAMILIES}
    assert not {family: issues for family, issues in derived_issues.items() if issues}, derived_issues
    unfinished = {
        family_id: FAMILY_STATES.get(family_id, "missing")
        for family_id in CI_REQUIRED_FINISHED_FAMILIES
        if FAMILY_STATES.get(family_id) != "finished"
    }
    assert not unfinished, (
        "Analytical-derivative coverage gate is still open. Complete each family's mechanical "
        f"ledger checks before changing its state to 'finished': {unfinished}"
    )


def test_mixed_precision_keeps_production_smoke_in_addition_to_autograd_oracles():
    contract = _mixed_precision.MIXED_ANALYTICAL_SMOKE_CONTRACT
    assert contract == {
        "continuous": {"quantities": ("ln_Psi", "e_L"), "routes": ("c", "r+R", "eps")},
        "gfmc": {"families": ("t", "n"), "routes": ("positions", "strain")},
    }

    def parameter_values(function, name):
        marks = [mark for mark in function.pytestmark if mark.name == "parametrize" and mark.args[0] == name]
        assert len(marks) == 1
        return tuple(marks[0].args[1])

    continuous = _mixed_precision.test_mixed_production_continuous_analytical_smoke
    gfmc = _mixed_precision.test_mixed_production_gfmc_analytical_smoke
    assert {mark.name for mark in continuous.pytestmark} >= {"light"}
    assert parameter_values(continuous, "quantity") == contract["continuous"]["quantities"]
    assert parameter_values(continuous, "route") == contract["continuous"]["routes"]
    assert {mark.name for mark in gfmc.pytestmark} >= {"heavy"}
    assert parameter_values(gfmc, "family") == contract["gfmc"]["families"]
    assert parameter_values(gfmc, "route") == contract["gfmc"]["routes"]
