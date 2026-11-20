"""Actual synthetic comparator traces distinguish rejection from unrelated errors."""

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
from pathlib import Path
import shutil

import pytest

from gdd.engine import case_binding_inventory, module_receipts
from gdd.engine.comparison_requirements import requirement_errors, runtime_requirement_errors
from gdd.engine.runtime_trace import trace_events

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.light, pytest.mark.without_chdir]
METHOD = "VM-NUM-MIXED-FIDELITY-001"
RELATIVE = "tests/test_rejection_fixture.py"
SOURCE = """import numpy as np
import pytest
from jqmc._setting import get_tolerance_mixed_fidelity

def _assert_blocks(mutated, reference):
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    for name in ("first", "second"):
        if not np.all(np.isfinite(mutated[name])) or not np.all(np.isfinite(reference[name])):
            raise AssertionError("finite blocks required")
        assert np.abs(mutated[name] - reference[name]).max() <= atol + rtol * np.abs(reference[name]).max()
    # ELSEWHERE

def _outer(mutated, reference):
    _assert_blocks(mutated, reference)

def test_case():
    reference = {name: np.asarray([1.0, -0.5]) for name in ("first", "second")}
    mutated = {name: block.copy() for name, block in reference.items()}
    # PERTURBATION
    with pytest.raises(AssertionError):
        _assert_blocks(mutated, reference)
"""


def _clear_fixture_caches():
    for function in [
        case_binding_inventory._module_index,
        case_binding_inventory._source_records,
        case_binding_inventory._rs_fixture_patterns,
        case_binding_inventory._generated_source_by_stem,
        module_receipts.committed_artifact_universe,
        module_receipts._artifact_registry_index,
    ]:
        function.cache_clear()


@pytest.fixture
def runtime_root(tmp_path, monkeypatch):
    """An isolated source fixture has no physical artifacts or scientific registration."""
    for directory in ["tests/test_rs_kernel", "tests/test_jqmc_kernel", "gdd/ssot/fixtures", "gdd/ssot/guarantees", "jqmc"]:
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests/test_rs_kernel/_rs_fixtures.py").write_text("_PATTERNS = {}\n")
    (tmp_path / "tests/test_jqmc_kernel/_generated_matrix_base.py").write_text("_GDD_SOURCE_ID_BY_STEM = {}\n")
    (tmp_path / "gdd/ssot/fixtures/sources.yaml").write_text("roots: {}\nsources: {}\n")
    for relative in ["gdd/ssot/verdicts.yaml", "jqmc/_setting.py"]:
        shutil.copyfile(ROOT / relative, tmp_path / relative)
    monkeypatch.setattr(case_binding_inventory, "ROOT", tmp_path)
    monkeypatch.setattr(case_binding_inventory, "SOURCE_PATH", tmp_path / "gdd/ssot/fixtures/sources.yaml")
    monkeypatch.setattr(module_receipts, "ROOT", tmp_path)
    monkeypatch.setattr(module_receipts, "GUARANTEE_DIR", tmp_path / "gdd/ssot/guarantees")
    _clear_fixture_caches()
    yield tmp_path
    _clear_fixture_caches()


def _observe(root, *, failure="first", nested=False):
    source = SOURCE
    category = "inequality"
    comparator = "python.inequality"
    if failure == "first":
        source = source.replace("# PERTURBATION", 'mutated["first"] += 1.0')
    elif failure == "nonfinite":
        source = source.replace("# PERTURBATION", 'mutated["first"][0] = np.nan')
    elif failure == "elsewhere":
        source = source.replace("# ELSEWHERE", 'raise AssertionError("elsewhere")')
    else:
        assert failure == "other-assertion"
        source = source.replace("# ELSEWHERE", 'assert False, "elsewhere"')
    if nested:
        source = source.replace("        _assert_blocks(mutated, reference)", "        _outer(mutated, reference)")
    path = root / RELATIVE
    path.write_text(source)
    case_binding_inventory._module_index.cache_clear()
    node = RELATIVE + "::test_case"
    binding = case_binding_inventory.derive_case_binding(node, 1, METHOD, "full-wavefunction")
    specs = binding["comparison"]["typed_observations"]
    checks = {
        "primary": {
            "site": RELATIVE + f"::_assert_blocks#{category}-1",
            "comparator": comparator,
            "actual": {"role": "mutated-production-block", "expression": "mutated[name]"},
            "reference": {"role": "independent-reference-block", "expression": "reference[name]"},
            "precision": {"mixed": {"policy": "CMP-MIXED-FULL-FIDELITY-001", "quantity_class": "parameter_derivative"}},
        }
    }
    assert not requirement_errors(
        checks, specs, ["mixed"], root=root, method=METHOD, pytest_node=node, cases=binding["fixture"]["concrete_cases"]
    )
    namespace = {"__name__": "rejection_fixture"}
    exec(compile(source, str(path), "exec"), namespace)
    observation = module_receipts.RuntimeObservation(
        expected_assertion_ids={row["id"] for row in specs},
        expected_artifacts={},
        assertion_specs=tuple(specs),
        executable_checks=checks,
    )
    with trace_events(observation.tracer, observation.needs_call_events):
        namespace["test_case"]()
    observation.finalize()
    return checks, specs, observation.comparison_invocations


def _errors(root, checks, specs, records):
    return runtime_requirement_errors(
        checks, specs, records, "mixed", root=root, selector={}, pytest_node=RELATIVE + "::test_case", method=METHOD
    )


@pytest.mark.parametrize(
    "fault",
    ["none", "passing-fabrication", "different-failed-site", "budget-substitution", "missing-failure", "tampered-caller"],
)
@pytest.mark.gdd_no_committed_artifact(
    reason="derives and traces synthetic NumPy modules in tmp_path; repository YAML is policy metadata, not a physical input"
)
def test_fabricated_negative_receipts_cannot_certify_rejection(runtime_root, fault):
    """Changing failure identity, the bound, its actual violation or the failing call chain invalidates the receipt.

    The control binds a real AssertionError to the exact failing comparison;
    the nested case is valid through its reciprocal caller link until that
    link is tampered with.
    """
    checks, specs, records = _observe(runtime_root, nested=fault == "tampered-caller")
    records = deepcopy(records)
    if fault == "none":
        assert any(record.get("exception", {}).get("failed") for record in records)
        assert not _errors(runtime_root, checks, specs, records)
        return
    if fault == "tampered-caller":
        assert not _errors(runtime_root, checks, specs, records)
        record = next(row for row in records if row.get("caller") and row.get("tolerance_calls"))
        record["caller"]["invocation"] = len(records) + 1
        assert _errors(runtime_root, checks, specs, records)
        return
    record = next(row for row in records if row.get("exception"))
    call = record["tolerance_calls"][0]["calls"][0]
    if fault == "passing-fabrication":
        call["operand_check"] = "passed"
    elif fault == "different-failed-site":
        record["exception"]["failed"] = "different-source-site"
    elif fault == "budget-substitution":
        call["arguments"]["atol"] *= 2
    else:
        record.pop("exception")
    assert _errors(runtime_root, checks, specs, records)


@pytest.mark.parametrize("failure", ["elsewhere", "other-assertion", "nonfinite"])
@pytest.mark.gdd_no_committed_artifact(
    reason="derives and traces synthetic NumPy modules in tmp_path; repository YAML is policy metadata, not a physical input"
)
def test_an_unrelated_exception_is_not_a_numerical_primary_rejection(runtime_root, failure):
    """Passing norms or finite-precondition errors cannot replace a failed primary."""
    checks, specs, records = _observe(runtime_root, failure=failure)
    errors = _errors(runtime_root, checks, specs, records)
    assert any("did not fail at an authored primary" in error for error in errors), errors
