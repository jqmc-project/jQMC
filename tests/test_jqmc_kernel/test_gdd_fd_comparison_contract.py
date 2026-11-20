"""Case-scoped finite-difference contracts reject unreviewed comparison changes."""

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

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from tests._gdd_ssot_layout import write_ssot_text

from gdd.engine.guarantee_store import is_module_ledger, load_module, load_ssot_document

from gdd.engine.fd_comparison_contract import load_fd_contract, validate_fd_call, validate_fd_receipt
from gdd.engine.fd_comparison_contract import ROOT as _ROOT
from gdd.engine.derivative_closure import required_fd_ids

pytestmark = [pytest.mark.light, pytest.mark.without_chdir, pytest.mark.design_pin]


def _resolved_comparisons(row):
    """Read unchanged full-precision budgets independently from the central table."""
    from jqmc import _setting

    result = []
    for declaration in row["comparisons"]:
        if declaration["policy"] == "CMP-FD-AUTODIFF-FULL-001":
            budget = (_setting.ATOL_FD_VS_ANALYTIC_TIGHT, _setting.RTOL_FD_VS_ANALYTIC_TIGHT)
        else:
            budget = _setting._TOLERANCE[declaration["level"]]["float64"]
        result.append(
            {
                "policy": declaration["policy"],
                "zones": tuple(declaration["zones"]),
                "level": declaration["level"],
                "atol": budget[0],
                "rtol": budget[1],
                "compared_array_count": declaration["compared_array_count"],
            }
        )
    return result


def _call(row, **overrides):
    kwargs = {
        "evidence_id": row["evidence_id"],
        "pytest_node_id": row["node"],
        "fixture_parameters": deepcopy(row["parameters"]),
        "precision_mode": "full",
        "comparisons": _resolved_comparisons(row),
    }
    kwargs.update(overrides)
    return validate_fd_call(**kwargs)


def _row(evidence_id="fd-ad-ao-grad-molecular"):
    return next(row for row in load_fd_contract().values() if row["evidence_id"] == evidence_id)


def test_fd_case_contract_preserves_reviewed_universe_and_budgets(tmp_path):
    """Every authored FD case retains its exact ordered comparison and tolerance declarations."""
    root, path = _shadow_file(tmp_path, "gdd/ssot/verdicts.yaml")
    verdicts = load_ssot_document(path)
    method = next(row for row in verdicts["verdict_methods"] if row["id"] == "VM-NUM-FD-ANCHOR-001")
    assert method["comparison_policies"] == ["CMP-FD-AUTODIFF-FULL-001", "CMP-SOURCE-CENTRAL-PAIR-001"]
    method["comparison_policies"] = ["CMP-FD-AUTODIFF-FULL-001"]
    write_ssot_text(path, yaml.safe_dump(verdicts, sort_keys=False))
    manifest = load_fd_contract(root=root)
    assert manifest == load_fd_contract()
    assert set(manifest) == required_fd_ids()
    contract = load_module(root / "gdd/ssot/guarantees/derivatives.yaml")
    fixture = next(row for row in contract["fixture_sets"] if row["id"] == "DERIV-FIX-FD-AUTODIFF-001")
    declarations = {node["id"]: node for node in fixture["reviewed_evidence"]["nodes"]}
    assert {row["evidence_id"] for row in manifest.values()} == set(declarations)
    assert {row["node"] for row in manifest.values()} == {node["pytest_node"] for node in declarations.values()}
    for row in manifest.values():
        assert row["comparisons"] == declarations[row["evidence_id"]]["comparison_contract"]


@pytest.mark.parametrize("mutation", ["evidence", "node", "missing-parameter", "extra-parameter", "parameter", "precision"])
def test_fd_case_contract_rejects_unregistered_call_identity(mutation):
    """Valid numerical budgets do not authorize another case, node or precision lane."""
    row = _row()
    changes = {}
    if mutation == "evidence":
        changes["evidence_id"] = "fd-ad-unreviewed"
    elif mutation == "node":
        changes["pytest_node_id"] = row["node"] + "_other"
    elif mutation == "precision":
        changes["precision_mode"] = "mixed"
    else:
        parameters = deepcopy(row["parameters"])
        if mutation == "missing-parameter":
            parameters.pop("ao_kind")
        elif mutation == "extra-parameter":
            parameters["unreviewed"] = True
        else:
            parameters["ao_kind"] = "unreviewed-orbital"
        changes["fixture_parameters"] = parameters
    with pytest.raises((AssertionError, ValueError)):
        _call(row, **changes)


@pytest.mark.parametrize(
    "mutation", ["policy", "zones", "level", "atol", "rtol", "zero-arrays", "extra-arrays", "bool-arrays", "dtn-dropped-check"]
)
def test_fd_case_contract_rejects_changed_comparison(mutation):
    """Identical values cannot hide changed policy, resolver arguments, array counts or a dropped check.

    ``dtn-dropped-check``: the DTN gradient and Laplacian keep both ordered comparisons;
    omitting the medium Laplacian comparison is rejected.
    """
    row = _row("fd-ad-dtn-grad-laplacian-molecular" if mutation == "dtn-dropped-check" else "fd-ad-ao-grad-molecular")
    comparisons = _resolved_comparisons(row)
    comparison = comparisons[0]
    if mutation == "dtn-dropped-check":
        assert len(comparisons) == 2
        comparisons.pop()
    elif mutation == "policy":
        comparison["policy"] = "CMP-ZONE-ALLCLOSE-MEDIUM-001"
    elif mutation == "zones":
        comparison["zones"] = ("det_eval",)
    elif mutation == "level":
        comparison["level"] = "medium"
    elif mutation in {"atol", "rtol"}:
        comparison[mutation] = float(np.nextafter(comparison[mutation], np.inf))
    elif mutation == "zero-arrays":
        comparison["compared_array_count"] = 0
    elif mutation == "extra-arrays":
        comparison["compared_array_count"] += 1
    else:
        comparison["compared_array_count"] = True
    with pytest.raises((AssertionError, ValueError)):
        _call(row, comparisons=comparisons)


def _receipt(obligation_id, row):
    """Make typed metadata evidence without substituting for the scientific FD tests."""
    from gdd.engine.evidence import CaseReceipt, ComparisonReceipt, ObservationReceipt
    from tests.test_jqmc_kernel._gdd_adapters import parameter_input, seeded_single_electron_receipt

    comparisons = []
    for index, comparison in enumerate(_resolved_comparisons(row)):
        suffix = "" if index == 0 else f"::additional-{index - 1}"
        comparisons.append(
            ComparisonReceipt(
                binding_id=f"COMPARISON::{obligation_id}{suffix}",
                policy_id=comparison["policy"],
                resolved_atol=comparison["atol"],
                resolved_rtol=comparison["rtol"],
                max_absolute_error=0.0,
                max_relative_error=0.0,
                failing_element_count=0,
                compared_array_count=comparison["compared_array_count"],
                compared_element_count=comparison["compared_array_count"],
            )
        )
    witness = ObservationReceipt(
        binding_id=f"FD-WITNESS::{obligation_id}",
        policy_id="OBS-FD-ROUTE-001",
        completed_observations=tuple(
            sorted(
                (
                    "perturbation-is-nonzero",
                    "plus-and-minus-points-are-distinct",
                    "autodiff-route-is-live",
                    "finite-difference-result-is-finite",
                    "stencil-declared",
                )
            )
        ),
        status="passed",
    )
    return CaseReceipt(
        guarantee_id="NUM-DERIV-FD-AUTODIFF-001",
        obligation_id=obligation_id,
        pytest_node_id=row["node"],
        input_binding=parameter_input(
            obligation_id,
            parameter_set_id=f"FD-PARAMS::{row['evidence_id']}",
            builder_operation=row["node"],
            declaration=row["parameters"],
        ),
        electron_configuration=seeded_single_electron_receipt(
            obligation_id,
            np.zeros((1, 3)),
            np.empty((0, 3)),
            generator_operation="fd-contract-unit-input",
            seed=1,
            generator_arguments=tuple(sorted(row["parameters"].items())),
            generation_pattern="direct",
            geminal_conditioning_mode="not-applicable",
            conditioning_reason="contract metadata unit input has no geminal matrix",
        ),
        comparison=comparisons[0],
        additional_checks=tuple(comparisons[1:]) + (witness,),
        status="passed",
    )


@pytest.mark.parametrize(
    "mutation",
    ["node", "atol", "zero-elements", "missing-extra", "missing-witness"],
)
def test_fd_case_contract_rejects_forged_receipt(mutation):
    """A valid receipt envelope cannot authorize altered numerical evidence metadata.

    One case per property: another node's identity, a loosened budget, a vacuous
    comparison, a dropped additional check and a missing route witness.
    """
    obligation_id, row = next(
        (key, row) for key, row in load_fd_contract().items() if row["evidence_id"] == "fd-ad-dtn-grad-laplacian-molecular"
    )
    receipt = _receipt(obligation_id, row)
    validate_fd_receipt(receipt)
    if mutation == "node":
        receipt = replace(receipt, pytest_node_id=row["node"] + "_other")
    elif mutation == "atol":
        atol = float(np.nextafter(receipt.comparison.resolved_atol, np.inf))
        receipt = replace(receipt, comparison=replace(receipt.comparison, resolved_atol=atol))
    elif mutation == "zero-elements":
        receipt = replace(receipt, comparison=replace(receipt.comparison, compared_element_count=0))
    elif mutation == "missing-extra":
        receipt = replace(receipt, additional_checks=receipt.additional_checks[1:])
    else:
        receipt = replace(receipt, additional_checks=receipt.additional_checks[:-1])
    with pytest.raises((AssertionError, ValueError)):
        validate_fd_receipt(receipt)


def _shadow_file(tmp_path, relative):
    """Isolate one candidate file while reading unchanged dependencies from the repository."""
    root = tmp_path / "candidate"
    root.mkdir()
    source = _ROOT
    target = root
    parts = Path(relative).parts
    for part in parts[:-1]:
        for child in source.iterdir():
            if child.name != part:
                (target / child.name).symlink_to(child, target_is_directory=child.is_dir())
        source = source / part
        target = target / part
        target.mkdir()
    for child in source.iterdir():
        if child.name != parts[-1] and child.name != _split_directory(relative):
            (target / child.name).symlink_to(child, target_is_directory=child.is_dir())
    path = target / parts[-1]
    _copy_candidate(source / parts[-1], path)
    return root, path


def _split_directory(relative):
    """A guarantee module's split directory is copied with its candidate, never linked (issue 205)."""
    path = Path(relative)
    return path.stem if is_module_ledger(_ROOT / path) else None


def _copy_candidate(source, path):
    shutil.copyfile(source, path)
    if is_module_ledger(source):
        shutil.copytree(source.parent / source.stem, path.parent / path.stem)


def _shadow_files(tmp_path, *relatives):
    """Isolate several candidate files under one root; every other path is a symlink."""
    root = tmp_path / "candidate"
    root.mkdir()
    for child in _ROOT.iterdir():
        (root / child.name).symlink_to(child, target_is_directory=child.is_dir())
    paths = []
    for relative in relatives:
        parts = Path(relative).parts
        source, target = _ROOT, root
        for part in parts[:-1]:
            source, target = source / part, target / part
            if target.is_symlink():
                target.unlink()
                target.mkdir()
                for child in source.iterdir():
                    (target / child.name).symlink_to(child, target_is_directory=child.is_dir())
        path = target / parts[-1]
        path.unlink()
        split = _split_directory(relative)
        if split is not None and (target / split).is_symlink():
            (target / split).unlink()
        _copy_candidate(source / parts[-1], path)
        paths.append(path)
    return root, paths


_FD_PAIR_SOURCE = "tests/test_jqmc_kernel/test_optimizer_derivative_sources.py"
_FD_PAIR_NODE = _FD_PAIR_SOURCE + "::test_two_component_source_matches_xy_directional_fd_and_conjugate_fold_does_not"


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "jqmc-call-operand",
        "jqmc-constant-operand",
        "float32-operand",
        "complex64-operand",
        "module-helper-operand",
        "fixture-operand",
        "unsupported-syntax",
    ],
)
def test_fd_central_pair_rule_rejects_non_numpy_operands(tmp_path, mutation):
    """The admitted FD-anchor central-pair site stays admitted only while both operands are pure NumPy fp64.

    Each mutation edits only the test source; the verdict SSoT keeps the four
    existing admissions, so a rejection can only come from the structural rule.
    """
    root, (path,) = _shadow_files(tmp_path, _FD_PAIR_SOURCE)
    source = path.read_text()
    edits = {
        "jqmc-call-operand": ("        A + B,\n", "        validate_derivative_sources(A + B),\n"),
        "jqmc-constant-operand": ("    step = 1.0e-7\n", "    step = ATOL_PYTHON_FP64_AGGREGATOR_LOOSE\n"),
        "float32-operand": ("    step = 1.0e-7\n", "    step = np.float32(1.0e-7)\n"),
        "complex64-operand": ("        A + B,\n", "        np.asarray(A + B, np.complex64),\n"),
        "module-helper-operand": (
            "        return A * p + B * np.conj(p)\n",
            "        return A * p + B * np.conj(_block(p, (1,), field=1, policy=1))\n",
        ),
        "fixture-operand": (
            "def test_two_component_source_matches_xy_directional_fd_and_conjugate_fold_does_not(A, B):",
            "def test_two_component_source_matches_xy_directional_fd_and_conjugate_fold_does_not(A, B, tmp_path):",
        ),
        "unsupported-syntax": ("    step = 1.0e-7\n", "    step = [1.0e-7 for _ in range(1)][0]\n"),
    }
    if mutation == "fixture-operand":
        old, new = edits[mutation]
        assert source.count(old) == 1
        source = source.replace(old, new).replace("    p0 = 0.3 - 0.2j\n", "    p0 = 0.3 - 0.2j + 0.0 * tmp_path\n")
    elif mutation != "none":
        old, new = edits[mutation]
        assert source.count(old) == 1, mutation
        source = source.replace(old, new)
    write_ssot_text(path, source)
    if mutation == "none":
        assert load_fd_contract(root=root) == load_fd_contract()
        return
    with pytest.raises(AssertionError, match="pure-numpy"):
        load_fd_contract(root=root)


@pytest.mark.parametrize(
    "mutation",
    [
        "drop-binding",
        "precision",
        "zero-count",
        "tier",
        "unscoped-policy",
        "different-policy",
        "mixed-lane-divergent-pair",
    ],
)
def test_fd_case_contract_rejects_static_declaration_drift(tmp_path, mutation):
    """Weakened FD declarations stay rejected, in the derivative ledger and in the method authority.

    Ledger: a dropped comparison contract, another precision, a vacuous array count or
    a looser tier. Verdicts: an unscoped extra policy, a substituted (looser) policy, or a
    mixed-lane central-pair admission that diverges from its full-lane twin.
    """
    root, (ledger, verdicts) = _shadow_files(tmp_path, "gdd/ssot/guarantees/derivatives.yaml", "gdd/ssot/verdicts.yaml")
    assert load_fd_contract(root=root) == load_fd_contract()
    if mutation in {"unscoped-policy", "different-policy", "mixed-lane-divergent-pair"}:
        registry = load_ssot_document(verdicts)
        method = next(row for row in registry["verdict_methods"] if row["id"] == "VM-NUM-FD-ANCHOR-001")
        policy = next(row for row in registry["comparison_policies"] if row["id"] == "CMP-SOURCE-CENTRAL-PAIR-001")
        if mutation == "unscoped-policy":
            method["comparison_policies"].append("CMP-ZONE-ALLCLOSE-STRICT-001")
        elif mutation == "different-policy":
            method["comparison_policies"] = ["CMP-FD-AUTODIFF-FULL-001", "CMP-XLA-RUST-MEDIUM-001"]
        else:
            mixed = next(
                row
                for row in policy["allowed-comparison-bindings"]
                if row["verdict_method"] == method["id"] and row["precision"] == "mixed"
            )
            mixed["pair"] = "TOL-D8-FLOOR-MEDIUM-R-FLOOR-MEDIUM-001"
        write_ssot_text(verdicts, yaml.safe_dump(registry, sort_keys=False))
    else:
        document = load_ssot_document(ledger)
        fixture = next(row for row in document["fixture_sets"] if row["id"] == "DERIV-FIX-FD-AUTODIFF-001")
        node = fixture["reviewed_evidence"]["nodes"][0]
        comparison = node["comparison_contract"][0]
        if mutation == "drop-binding":
            del node["comparison_contract"]
        elif mutation == "precision":
            node["precision"] = "mixed"
        elif mutation == "zero-count":
            comparison["compared_array_count"] = 0
        else:
            comparison["level"] = "medium"
        write_ssot_text(ledger, yaml.safe_dump(document, sort_keys=False))
    with pytest.raises((AssertionError, ValueError)):
        load_fd_contract(root=root)


@pytest.mark.parametrize("mutation", ["zones", "level", "arrays", "atol", "missing-call", "identity", "extra-comparison"])
def test_fd_case_contract_rejects_source_drift(tmp_path, mutation):
    """Editing a live FD adapter call cannot leave the authored comparison contract green."""
    from gdd.engine.fd_comparison_contract import fd_source_contract_errors

    root, path = _shadow_file(tmp_path, "tests/test_jqmc_kernel/test_atomic_orbital.py")
    assert not fd_source_contract_errors(root=root)
    source = path.read_text()
    start = source.index('        record_fd_case(\n            "fd-ad-ao-grad-molecular",')
    end = source.index("        jax.clear_caches()", start)
    call = source[start:end]
    if mutation == "zones":
        mutated = call.replace('zones=("ao_eval", "ao_grad_lap"),', 'zones=("det_eval",),')
    elif mutation == "level":
        mutated = call.replace('level="strict",', 'level="medium",')
    elif mutation == "arrays":
        mutated = call.replace(
            "compared_pairs=((gx_u, gx_n), (gy_u, gy_n), (gz_u, gz_n)),", "compared_pairs=((gx_u, gx_n), (gy_u, gy_n)),"
        )
    elif mutation == "atol":
        mutated = call.replace("atol=atol,", "atol=rtol,")
    elif mutation == "missing-call":
        mutated = call.replace("record_fd_case(", "unused_fd_call(")
    elif mutation == "identity":
        mutated = call.replace('"fd-ad-ao-grad-molecular",', '"fd-ad-unreviewed",')
    else:
        mutated = call.replace(
            '            level="strict",',
            '            level="strict",\n            additional_comparisons=((((gx_u, gx_n),), atol, rtol, ("ao_eval", "ao_grad_lap"), "strict"),),',
        )
    assert mutated != call
    write_ssot_text(path, source[:start] + mutated + source[end:])
    assert fd_source_contract_errors(root=root)


@pytest.mark.gdd_synthetic_derivative_receipts(reason="drives the FD adapter with synthetic arrays and emission patched out")
@pytest.mark.parametrize(
    "family,mutation",
    [
        ("ordinary", "valid"),
        ("dtn", "valid"),
        ("ordinary", "empty-primary"),
        ("dtn", "empty-primary"),
        ("dtn", "empty-additional"),
        ("ordinary", "mixed"),
        ("dtn", "mixed"),
    ],
)
def test_fd_adapter_rejects_empty_arrays_and_actual_mixed_execution(monkeypatch, family, mutation):
    """The live adapter observes nonempty operands and actual execution precision before emission."""
    from gdd.engine.evidence import CaseReceipt
    from jqmc._precision import configure, is_mixed_precision_enabled
    from tests.test_jqmc_kernel import _gdd_fd_adapters as adapter

    row = _row("fd-ad-dtn-grad-laplacian-molecular" if family == "dtn" else "fd-ad-ao-grad-molecular")
    comparisons = _resolved_comparisons(row)
    pairs = [tuple((np.zeros(1), np.zeros(1)) for _ in range(comparison["compared_array_count"])) for comparison in comparisons]
    if mutation in {"empty-primary", "empty-additional"}:
        index = 1 if mutation == "empty-additional" else 0
        # Retain a nonempty sibling: aggregate element count alone would miss this hole.
        assert len(pairs[index]) > 1
        pairs[index] = ((np.empty(0), np.empty(0)), *pairs[index][1:])
    captured = []

    def construct_receipt(**kwargs):
        assert kwargs.pop("emit_completion") is False
        kwargs["comparison"] = kwargs.pop("primary_check")
        receipt = CaseReceipt(status="passed", **kwargs)
        receipt.validate()
        return receipt

    monkeypatch.setattr(adapter, "case_receipt", construct_receipt)
    monkeypatch.setattr(adapter, "emit_completion_record", captured.append)
    kwargs = {
        "pytest_node_id": row["node"],
        "fixture_parameters": row["parameters"],
        "coordinates": (np.zeros((1, 3)), np.empty((0, 3))),
        "coordinate_generator": "fd-contract-unit-input",
        "coordinate_seed": 1,
        "compared_pairs": pairs[0],
        "atol": comparisons[0]["atol"],
        "rtol": comparisons[0]["rtol"],
        "zones": comparisons[0]["zones"],
        "level": comparisons[0]["level"],
        "stencil": "central-first-order-2",
        "step": 1.0,
        "direction": "cartesian-basis",
        "additional_comparisons": tuple(
            (pair, comparison["atol"], comparison["rtol"], comparison["zones"], comparison["level"])
            for pair, comparison in zip(pairs[1:], comparisons[1:], strict=True)
        ),
    }
    original_mode = "mixed" if is_mixed_precision_enabled() else "full"
    try:
        configure("mixed" if mutation == "mixed" else "full")
        if mutation == "valid":
            adapter.record_fd_case(row["evidence_id"], **kwargs)
            assert len(captured) == 1
            validate_fd_receipt(captured[0])
        else:
            message = "empty array" if mutation.startswith("empty") else "outside full precision"
            with pytest.raises(AssertionError, match=message):
                adapter.record_fd_case(row["evidence_id"], **kwargs)
            assert captured == []
    finally:
        configure(original_mode)
