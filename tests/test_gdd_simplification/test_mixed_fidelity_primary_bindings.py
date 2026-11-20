"""Mixed-fidelity authoring must bind the actual node, not a lexical lookalike."""

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

from pathlib import Path
import shutil

import pytest

from gdd.engine import case_binding_inventory
from scripts import check_gdd_module_ledgers
from tests._mixed_fidelity_primary_binding import (
    authored_full_math_on_migrated_nodes,
    authored_mixed_fidelity_nodes,
    primary_binding_errors,
)

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.light, pytest.mark.without_chdir]
SOURCE = """import numpy as np
from jqmc._setting import get_tolerance_mixed_fidelity

def _assert_bound(mixed, full, atol, rtol):
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()

class TestFidelity:
    def test_case(self):
        mixed = np.asarray([1.0])
        full = np.asarray([1.0])
        atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
        _assert_bound(mixed, full, atol, rtol)
"""
NODE = "tests/test_primary_fixture.py::TestFidelity::test_case"
SITE = "tests/test_primary_fixture.py::TestFidelity.test_case#helper-inequality-1"


def _authored_node():
    return {
        "pytest_node": NODE,
        "required_precision_modes": ["mixed"],
        "executable_checks": {
            "primary": {
                "site": SITE,
                "comparator": "python.inequality",
                "actual": {"role": "mixed-production-block", "expression": "mixed"},
                "reference": {"role": "independent-full-production-block", "expression": "full"},
                "precision": {"mixed": {"policy": "CMP-MIXED-FULL-FIDELITY-001", "quantity_class": "parameter_derivative"}},
            }
        },
    }


@pytest.fixture
def source_root(tmp_path, monkeypatch):
    """Each mutation gets fresh real source and real engine derivation, without science."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_primary_fixture.py").write_text(SOURCE)
    for name in [
        "gdd/ssot/verdicts.yaml",
        "jqmc/_setting.py",
        "jqmc/_precision.py",
        "tests/test_rs_kernel/_rs_fixtures.py",
        "tests/test_jqmc_kernel/_generated_matrix_base.py",
    ]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (tmp_path / "gdd/ssot/fixtures").symlink_to(ROOT / "gdd/ssot/fixtures", target_is_directory=True)
    monkeypatch.setattr(case_binding_inventory, "ROOT", tmp_path)
    monkeypatch.setattr(check_gdd_module_ledgers, "ROOT", tmp_path)
    # Both APIs memoize parsed modules; regeneration must inspect fresh source.
    case_binding_inventory._module_index.cache_clear()
    check_gdd_module_ledgers._EVAL_CONTEXT_CACHE.clear()
    yield tmp_path
    case_binding_inventory._module_index.cache_clear()
    check_gdd_module_ledgers._EVAL_CONTEXT_CACHE.clear()


def test_all_authored_mixed_fidelity_primaries_bind_actual_nodes():
    """Every current numerical fidelity node owns a source-matched execution primary."""
    registrations = list(authored_mixed_fidelity_nodes(ROOT))
    assert registrations, "No mixed-fidelity authoring was inspected"
    errors = [f"{guarantee}: {error}" for guarantee, node in registrations for error in primary_binding_errors(ROOT, node)]
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("replacement", [None, {}])
@pytest.mark.gdd_no_committed_artifact(
    reason="derives synthetic test modules in tmp_path; copied YAML is GDD policy metadata, not a physical input"
)
def test_missing_authored_primary_is_not_legacy_acceptance(source_root, replacement):
    """Regenerating descriptions cannot turn missing numerical authoring into certification."""
    node = _authored_node()
    node["executable_checks"] = replacement
    assert primary_binding_errors(source_root, node) == [f"{NODE}: missing authored mixed-fidelity primary"]


@pytest.mark.parametrize("fault", ["none", "quantity", "comparator", "operand", "mode"])
@pytest.mark.gdd_no_committed_artifact(
    reason="derives synthetic test modules in tmp_path; copied YAML is GDD policy metadata, not a physical input"
)
def test_regenerated_binding_rejects_policy_and_operand_substitution(source_root, fault):
    """Actual source, comparator and central policy remain independently required."""
    node = _authored_node()
    check = node["executable_checks"]["primary"]
    if fault == "quantity":
        check["precision"]["mixed"]["quantity_class"] = "orbital_derivative"
    elif fault == "comparator":
        check["comparator"] = "numpy.testing.assert_allclose"
    elif fault == "operand":
        check["actual"]["expression"] = "full"
    elif fault == "mode":
        node["required_precision_modes"] = ["full", "mixed"]
    errors = primary_binding_errors(source_root, node)
    if fault == "none":
        assert not errors, errors
        return
    expected = {
        "quantity": "quantity class differs",
        "comparator": "substituted comparator",
        "operand": "operand/role differs",
        "mode": "precision requirements do not match",
    }[fault]
    assert any(expected in error for error in errors), errors


FULL_SOURCE = (
    SOURCE.split("class TestFidelity:")[0]
    + """from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled

def out_of_budget_mixed_block(mixed, full, *, atol, rtol):
    return mixed + atol + rtol * np.abs(full).max()

class TestFidelity:
    def test_case(self):
        mixed = np.asarray([1.0])
        full = np.asarray([1.0])
        if is_mixed_precision_enabled():
            atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
            _assert_bound(mixed, full, atol, rtol)
            # MIXED_PERTURBATION
            return
        full_analytical = np.asarray([1.0])
        full_ad = np.asarray([1.0])
        atol, rtol = get_tolerance_min(("ao_eval",), "strict")
        np.testing.assert_allclose(full_analytical, full_ad, atol=atol, rtol=rtol)
"""
)


def _full_authored_node():
    return {
        "pytest_node": NODE,
        "required_precision_modes": ["full"],
        "executable_checks": {
            "retained-full-ad": {
                "site": "tests/test_primary_fixture.py::TestFidelity.test_case#comparison-1",
                "comparator": "numpy.testing.assert_allclose",
                "actual": {"role": "full-analytical-output", "expression": "full_analytical"},
                "reference": {"role": "independent-full-ad-oracle", "expression": "full_ad"},
                "precision": {"full": {"policy": "CMP-ZONE-ALLCLOSE-STRICT-001", "zones": ["ao_eval"]}},
            }
        },
    }


def test_retained_authored_full_math_bindings_on_migrated_nodes():
    """Adding mixed detection cannot silently rebind the retained full mathematical proof."""
    errors = [
        f"{guarantee}: {error}"
        for guarantee, node in authored_full_math_on_migrated_nodes(ROOT)
        for error in primary_binding_errors(ROOT, node, method="VM-MATH-AD-ORACLE-001")
    ]
    assert not errors, "\n".join(errors)


@pytest.mark.gdd_no_committed_artifact(
    reason="derives a synthetic perturbation-site mutant in tmp_path, not scientific AD values"
)
def test_mixed_perturbation_call_cannot_shift_the_retained_full_binding(source_root):
    """A mixed-only tolerance-bearing helper may shift FULL ordinals while NUM stays valid."""
    source = FULL_SOURCE.replace(
        "# MIXED_PERTURBATION", "mutated = out_of_budget_mixed_block(mixed, full, atol=atol, rtol=rtol)"
    )
    (source_root / "tests/test_primary_fixture.py").write_text(source)
    assert not primary_binding_errors(source_root, _authored_node())
    errors = primary_binding_errors(source_root, _full_authored_node(), method="VM-MATH-AD-ORACLE-001")
    assert any("lexical operand expressions differ" in error for error in errors), errors
    fixed = _full_authored_node()
    fixed["executable_checks"]["retained-full-ad"]["site"] = fixed["executable_checks"]["retained-full-ad"]["site"].replace(
        "#comparison-1", "#comparison-2"
    )
    assert not primary_binding_errors(source_root, fixed, method="VM-MATH-AD-ORACLE-001")


@pytest.mark.parametrize(
    "call_form",
    ["direct", "unreachable", "escaped", "swallowed-error", "partial-case", "wrong-source", "helper-only"],
)
@pytest.mark.gdd_no_committed_artifact(
    reason="derives synthetic imported assertion-call contracts and literal case forwarding, not physical calculations"
)
def test_imported_diagnostic_return_primary_requires_actual_case_calls(source_root, call_form):
    """An imported primary certifies only when the node actually calls it for every case.

    An unreachable, escaped or swallowed call, a call missing for one case, a
    same-named helper from another source, or a helper that exists only
    lexically cannot stand in for the executed comparison.
    """
    helper_source = """import numpy as np
from jqmc._setting import get_tolerance_mixed_fidelity

def _assert_bound(mixed, full, atol, rtol):
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()

def assert_fidelity(mixed, full):
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    _assert_bound(mixed, full, atol, rtol)
    return float(np.abs(mixed - full).max())
"""
    (source_root / "tests/helper_primary_fixture.py").write_text(helper_source)
    module_source = """import numpy as np
import pytest
from jqmc._setting import get_tolerance_mixed_fidelity
from tests.helper_primary_fixture import assert_fidelity

def _assert_bound(mixed, full, atol, rtol):
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()

def _route(case, mixed, full):
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    _assert_bound(mixed, full, atol, rtol)
    if case == "first":
        assert_fidelity(mixed, full)
    elif case == "second":
        assert_fidelity(mixed, full)

def _forward(callback, mixed, full):
    callback(mixed, full)

class TestFidelity:
    @pytest.mark.parametrize("case", ["first", "second"])
    def test_case(self, case):
        mixed = np.asarray([1.0])
        full = np.asarray([1.0])
        # CALL
"""
    forms = {
        "direct": "assert_fidelity(mixed, full)",
        "unreachable": "if False:\n            assert_fidelity(mixed, full)",
        "escaped": "_forward(assert_fidelity, mixed, full)",
        "wrong-source": "assert_fidelity(mixed, full)",
        "partial-case": "_route(case, mixed, full)",
        "swallowed-error": "try:\n            assert_fidelity(mixed, full)\n        except AssertionError:\n            pass",
        "helper-only": "assert True",
    }
    module_source = module_source.replace("# CALL", forms[call_form])
    if call_form == "partial-case":
        module_source = module_source.replace('    elif case == "second":\n        assert_fidelity(mixed, full)\n', "")
    if call_form == "wrong-source":
        (source_root / "tests/decoy_primary_fixture.py").write_text(helper_source)
        module_source = module_source.replace("from tests.helper_primary_fixture", "from tests.decoy_primary_fixture")
    (source_root / "tests/test_primary_fixture.py").write_text(module_source)
    node = _authored_node()
    node["executable_checks"]["primary"]["site"] = "tests/helper_primary_fixture.py::assert_fidelity#helper-inequality-1"
    errors = primary_binding_errors(source_root, node)
    if call_form == "direct":
        assert not errors, errors
    else:
        assert errors, f"Unsupported {call_form} cannot certify an imported primary"
