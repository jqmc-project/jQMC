"""Focused RED/GREEN sentinels for the contracts that must not weaken."""

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

import copy
import subprocess
from pathlib import Path

import pytest

from gdd.engine.contract_guard import ledger_monotonicity_errors
from gdd.engine.skip_protection import skip_change_errors


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

LEDGER = Path("gdd/ssot/guarantees/toy.yaml")


def _ledger() -> dict:
    return {
        "schema_version": 1,
        "document": {"id": "jqmc.toy", "version": 1, "lifecycle": "cutover", "title": "Toy"},
        "enforcement_map": {"guarantees": "scripts/check_gdd_test_mapping.py"},
        "symbol_contract": {
            "status": "governed",
            "default_guarantee_refs": ["MATH-TOY-001"],
            "overrides": {},
            "critical_private_symbols": [],
        },
        "guarantees": [
            {
                "id": "MATH-TOY-001",
                "kind": "mathematical",
                "plain_language": "The production value equals the independent oracle.",
                "scope": {"quantities": ["value"]},
                "required_evidence": [
                    {
                        "verdict_method": "VM-ALLCLOSE-001",
                        "nodes": [
                            {
                                "pytest_node": "tests/test_toy.py::test_value",
                                "required_precision_modes": ["full", "mixed"],
                                "required_parameter_cases": [{"case": "regular"}, {"case": "edge"}],
                                "executable_checks": {"value": {"comparison_policy": "CMP-STRICT-001", "actual": "value"}},
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _errors(candidate: dict) -> list[str]:
    return ledger_monotonicity_errors(_ledger(), candidate, LEDGER)


@pytest.mark.parametrize(
    "mutation",
    [
        "delete-guarantee",
        "narrow-precision",
        "narrow-parameters",
        "rewrite-plain-language",
        "append-plain-language-qualification",
    ],
)
def test_guarantee_weakening_is_red(mutation: str):
    candidate = copy.deepcopy(_ledger())
    if mutation == "delete-guarantee":
        candidate["guarantees"].clear()
    elif mutation == "narrow-precision":
        candidate["guarantees"][0]["required_evidence"][0]["nodes"][0]["required_precision_modes"] = ["full"]
    elif mutation == "narrow-parameters":
        candidate["guarantees"][0]["required_evidence"][0]["nodes"][0]["required_parameter_cases"] = [{"case": "regular"}]
    elif mutation == "rewrite-plain-language":
        candidate["guarantees"][0]["plain_language"] = "The production route executes."
    else:
        candidate["guarantees"][0]["plain_language"] += " Except for difficult inputs."

    assert _errors(candidate)


# --- Issue 227: static settings manifest (gdd/engine/tolerance_manifest.py) ---
#
# The two settings modules are reviewed files. Their DATA bindings are evaluated
# statically and may only tighten; everything else is CODE and any change to it
# is one `settings code changed: <shape id>` line.

REPO = Path(__file__).resolve().parents[2]
SETTING = "jqmc/_setting.py"
PRECISION = "jqmc/_precision.py"
TOY_SETTINGS = (
    "ATOL_TOY = 1.0e-8\n"
    "RTOL_TOY = 1.0e-6\n"
    "DECIMAL_TOY = 6\n"
    'BACKEND_TOY = "cpu"\n'
    '_TOLERANCE = {"toy": {"strict": (1.0e-8, 1.0e-6)}}\n\n\n'
    'def get_tolerance(zone, level="strict"):\n'
    "    return _TOLERANCE[zone][level]\n"
)
TOY_LOOSENED_ATOL = "jqmc/_setting.py: tolerance loosened: ATOL_TOY 1e-08 -> 1e-06"
TOY_LOOSENED_TABLE = "jqmc/_setting.py: tolerance loosened: _TOLERANCE[toy][strict][0] 1e-08 -> "
UNEVALUABLE = "cannot evaluate the settings manifest: "


def _toy_settings_repo(tmp_path: Path, settings: str = TOY_SETTINGS) -> Path:
    """A git repository whose trusted base is ``settings``; the working tree is the candidate."""
    root = tmp_path / "repo"
    (root / "jqmc").mkdir(parents=True)
    (root / "jqmc/_setting.py").write_text(settings, encoding="utf-8")
    git = ["git", "-c", "user.name=gdd", "-c", "user.email=gdd@example.invalid", "-c", "commit.gpgsign=false"]
    for command in (["init", "-q", "-b", "trusted"], ["add", "-A"], ["commit", "-q", "-m", "trusted base"]):
        subprocess.run([*git, *command], cwd=root, check=True, capture_output=True)
    return root


def _toy_gate_errors(root: Path) -> list[str]:
    from gdd.engine.tolerance_manifest import settings_contract_errors

    return settings_contract_errors("trusted", root, git_cwd=root)


def _candidate(root: Path, settings: str) -> list[str]:
    (root / "jqmc/_setting.py").write_text(settings, encoding="utf-8")
    return _toy_gate_errors(root)


def _is_code_line(error: str) -> bool:
    import re

    return re.fullmatch(r"jqmc/_(setting|precision)\.py: settings code changed: [0-9a-f]{12}", error) is not None


def test_manifest_comparator_directions_and_removals():
    """A tolerance may only shrink, a digit count or floor only grow, every other setting stays equal; nothing disappears."""
    from gdd.engine.tolerance_manifest import manifest_errors

    def entry(kind, value):
        return {"kind": kind, "value": value}

    base = {
        "m::ATOL": entry("tolerance", 1e-8),
        "m::DECIMAL": entry("places", 6),
        "m::FLOOR": entry("floor", 4),
        "m::NAME": entry("setting", "cpu"),
        "m::NAN": entry("tolerance", float("nan")),
        "m::GONE": entry("tolerance", 1.0),
    }
    same = {key: dict(value) for key, value in base.items()}
    assert manifest_errors(base, same) == []
    tightened = {**same, "m::ATOL": entry("tolerance", 1e-9), "m::DECIMAL": entry("places", 7), "m::FLOOR": entry("floor", 5)}
    assert manifest_errors(base, tightened) == []
    assert manifest_errors(base, {**same, "m::NEW": entry("tolerance", 1.0)}) == []
    loosened = {**same, "m::ATOL": entry("tolerance", 1e-7), "m::DECIMAL": entry("places", 5), "m::FLOOR": entry("floor", 3)}
    assert manifest_errors(base, loosened) == [
        "m: tolerance loosened: ATOL 1e-08 -> 1e-07",
        "m: decimal places reduced: DECIMAL 6 -> 5",
        "m: floor lowered: FLOOR 4 -> 3",
    ]
    assert manifest_errors(base, {**same, "m::NAME": entry("setting", "gpu")}) == [
        "m: reviewed setting changed: NAME 'cpu' -> 'gpu'"
    ]
    assert manifest_errors(base, {**same, "m::NAN": entry("tolerance", 1.0)}) == ["m: tolerance loosened: NAN nan -> 1.0"]
    assert manifest_errors(base, {**same, "m::ATOL": entry("setting", "1e-8")}) == [
        "m: reviewed setting changed: ATOL 1e-08 -> '1e-8'"
    ]
    without = {key: value for key, value in same.items() if key != "m::GONE"}
    assert manifest_errors(base, without) == ["m: tolerance removed: GONE 1.0"]


@pytest.mark.parametrize(
    "mutation",
    [
        "direct",
        "table",
        "alias",
        "augmented",
        "delete",
    ],
)
def test_effective_tolerance_loosening_is_red(tmp_path, mutation):
    """Every data route to a looser value is reported by value, whatever syntax binds it."""
    root = _toy_settings_repo(tmp_path)
    settings = TOY_SETTINGS
    if mutation == "direct":
        settings = settings.replace("ATOL_TOY = 1.0e-8", "ATOL_TOY = 1.0e-6")
    elif mutation == "table":
        settings += '_TOLERANCE["toy"]["strict"] = (1.0, 1.0)\n'
    elif mutation == "alias":
        settings += '_alias = _TOLERANCE["toy"]\n_alias["strict"] = (1.0, 1.0)\n'
    elif mutation == "augmented":
        settings += "ATOL_TOY *= 100\n"
    else:
        settings += "del ATOL_TOY\n"
    errors = _candidate(root, settings)
    assert errors, mutation
    assert all(not _is_code_line(error) for error in errors), errors
    if mutation in ("direct", "augmented"):
        assert any(error.startswith("jqmc/_setting.py: tolerance loosened: ATOL_TOY 1e-08 -> ") for error in errors), errors
    elif mutation == "delete":
        assert errors == ["jqmc/_setting.py: tolerance removed: ATOL_TOY 1e-08"]
    else:
        assert TOY_LOOSENED_TABLE + "1.0" in errors, errors


CODE_EDITS = {
    "helper": TOY_SETTINGS + "\n\ndef unrelated(value):\n    return value\n",
    "import": "import math\n\n" + TOY_SETTINGS,
    "lambda-rebind": TOY_SETTINGS + '\nget_tolerance = lambda zone, level="strict": (1.0, 1.0)\n',
    "if-suite": TOY_SETTINGS + '\nif True:\n    _TOLERANCE["toy"]["strict"] = (1.0, 1.0)\n',
    "update-call": TOY_SETTINGS + '\n_TOLERANCE["toy"].update(strict=(1.0, 1.0))\n',
    "resolver-body": TOY_SETTINGS.replace("return _TOLERANCE[zone][level]", "return (1.0, 1.0)"),
    "default": TOY_SETTINGS.replace('level="strict"', 'level="loose"'),
    "positional-only": TOY_SETTINGS.replace('level="strict")', 'level="strict", /)'),
    "removed-resolver": TOY_SETTINGS[: TOY_SETTINGS.index("def get_tolerance")],
    "global": TOY_SETTINGS + "\n\ndef _late():\n    global get_tolerance\n    get_tolerance = None\n\n\n_late()\n",
    "walrus-rebind": TOY_SETTINGS + '\nif (get_tolerance := lambda zone, level="strict": (1.0, 1.0)):\n    pass\n',
    "decorator": "import functools\n\n"
    + TOY_SETTINGS.replace("def get_tolerance", "@functools.lru_cache(maxsize=None)\ndef get_tolerance"),
    "call-assignment": TOY_SETTINGS + "\n\ndef _scaled(value):\n    return value + 1.0e-6\n\n\nATOL_TOY = _scaled(ATOL_TOY)\n",
}


@pytest.mark.parametrize("edit", sorted(CODE_EDITS))
def test_code_changes_are_one_reported_line(tmp_path, edit):
    """Any change to the code of a settings module is exactly one reported line naming the new code shape."""
    root = _toy_settings_repo(tmp_path)
    errors = _candidate(root, CODE_EDITS[edit])
    assert len(errors) == 1 and _is_code_line(errors[0]), errors


def test_code_shape_id_names_exactly_the_approved_code(tmp_path):
    """The id is stable under data, docstring, annotation and comment edits and changes with every code change."""
    from gdd.engine.tolerance_manifest import code_shape, code_shape_id, settings_code_change_errors

    helper = TOY_SETTINGS + "\n\ndef unrelated(value):\n    return value\n"
    (line,) = settings_code_change_errors(TOY_SETTINGS, helper, SETTING)
    for free in (
        helper.replace("ATOL_TOY = 1.0e-8", "ATOL_TOY = 1.0e-9"),
        '"""doc"""\n' + helper,
        helper.replace("def unrelated(value):", "def unrelated(value: float) -> float:"),
        helper + "# comment\n",
        helper + "ATOL_NEW = 1.0\n",
    ):
        assert settings_code_change_errors(TOY_SETTINGS, free, SETTING) == [line], free
    again = helper.replace("return value", "return value * 2")
    (other,) = settings_code_change_errors(TOY_SETTINGS, again, SETTING)
    assert other != line
    assert code_shape_id(code_shape(helper)) == line.rsplit(" ", 1)[1]
    assert settings_code_change_errors(helper, helper, SETTING) == []
    # The line names that shape only: the gate against the same base reports the new id.
    root = _toy_settings_repo(tmp_path)
    assert _candidate(root, helper) == [line]
    assert _candidate(root, again) == [other]


def test_unreadable_candidate_fails_closed(tmp_path):
    root = _toy_settings_repo(tmp_path)
    errors = _candidate(root, TOY_SETTINGS + "\ndef broken(:\n")
    assert len(errors) == 1 and errors[0].startswith(UNEVALUABLE + "cannot parse settings module jqmc/_setting.py"), errors
    (root / "jqmc/_setting.py").unlink()
    errors = _toy_gate_errors(root)
    assert errors == [UNEVALUABLE + "jqmc/_setting.py was deleted from the candidate"]
    assert _toy_gate_errors(_toy_settings_repo(tmp_path / "clean")) == []


def test_trusted_source_check_compares_settings(monkeypatch):
    """trusted_source_errors runs the settings comparison against the base and reports its lines."""
    import scripts.check_gdd_contract_changes as checker

    calls: list[tuple] = []

    def fake(base_ref, candidate_root, *, git_cwd):
        calls.append((base_ref, candidate_root, git_cwd))
        return [TOY_LOOSENED_ATOL]

    monkeypatch.setattr(checker, "settings_contract_errors", fake)
    monkeypatch.setattr(checker, "_blob_at", lambda _ref, _relative: None)
    monkeypatch.setattr(checker, "reviewed_inventory_change_errors", lambda _ref: [])
    errors = checker.trusted_source_errors("base")
    assert calls == [("base", checker.ROOT, checker.ROOT)]
    assert TOY_LOOSENED_ATOL in errors
    assert "gdd/engine/tolerance_manifest.py" in checker.GUARD_FILES
    assert "gdd/engine/tolerance_session.py" not in checker.GUARD_FILES
    assert any(module.__name__ == "gdd.engine.tolerance_manifest" for module in checker.DECISIVE_MODULES)


def test_value_kinds_follow_the_name():
    from gdd.engine.tolerance_manifest import SETTING_KINDS, value_kind

    assert value_kind("ATOL_BIT_EXACT") == "tolerance"
    assert value_kind("MAXULP_MPC_SENTINEL_HISTORY") == "tolerance"
    assert value_kind("EVALUATION_KAPPA_MAX") == "tolerance"
    assert value_kind("DECIMAL_TURBORVB_VALUES") == "places"
    assert value_kind("SIGNIFICANT_DIGITS") == "places"
    assert value_kind("RS_NONVACUITY_TOL") == "setting"
    assert value_kind("DISTINCTNESS_RTOL") == "setting"
    assert value_kind("BATCH_SIZE") == "setting"
    assert SETTING_KINDS["AO_VALUE_COHERENT_BIAS_EPS_FRACTION"] == "tolerance"
    assert SETTING_KINDS["EVALUATION_KAPPA_ROUNDING_SLOPE"] == "floor"


# --- the real modules ------------------------------------------------------------------


def _real(relative: str) -> str:
    return (REPO / relative).read_text(encoding="utf-8")


def test_real_settings_modules_are_fully_static():
    """Every data binding of the real modules is evaluated statically (no reference texts) and every tolerance name is covered."""
    import ast

    from gdd.engine.tolerance_manifest import Ref, data_bindings, is_data_binding, settings_manifest, value_kind

    for relative in (SETTING, PRECISION):
        source = _real(relative)
        values = data_bindings(source)
        references = {name: value for name, value in values.items() if isinstance(value, Ref)}
        assert references == {}, references
        manifest = settings_manifest(source, relative)
        assert len(manifest) > (200 if relative == SETTING else 20)
        for statement in ast.parse(source).body:
            if is_data_binding(statement) and isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("__"):
                        assert any(key.startswith(f"{relative}::{target.id}") for key in manifest), target.id
    manifest = settings_manifest(_real(SETTING), SETTING)
    tolerances = [key for key, entry in manifest.items() if entry["kind"] == "tolerance"]
    assert len(tolerances) > 100
    assert manifest["jqmc/_setting.py::_TOLERANCE[strict][float64][0]"]["kind"] == "tolerance"
    assert manifest["jqmc/_setting.py::AO_VALUE_COHERENT_BIAS_EPS_FRACTION"]["kind"] == "tolerance"
    assert manifest["jqmc/_setting.py::EVALUATION_KAPPA_ROUNDING_SLOPE"]["kind"] == "floor"
    assert manifest["jqmc/_setting.py::DECIMAL_TURBORVB_VALUES"]["kind"] == "places"
    assert value_kind("get_tol_hdf5_roundtrip") == "tolerance"  # resolvers are code, reviewed through the shape


def test_real_settings_gate_is_clean_against_itself(tmp_path):
    """The complete gate function on a repository whose base is the current real modules reports nothing."""
    from gdd.engine.tolerance_manifest import settings_contract_errors

    root = tmp_path / "repo"
    for relative in (SETTING, PRECISION):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(_real(relative), encoding="utf-8")
    git = ["git", "-c", "user.name=gdd", "-c", "user.email=gdd@example.invalid", "-c", "commit.gpgsign=false"]
    for command in (["init", "-q", "-b", "trusted"], ["add", "-A"], ["commit", "-q", "-m", "trusted base"]):
        subprocess.run([*git, *command], cwd=root, check=True, capture_output=True)
    assert settings_contract_errors("trusted", root, git_cwd=root) == []
    (root / SETTING).write_text(_real(SETTING) + '\n_TOLERANCE["strict"]["float64"] = (1e-3, 1e-2)\n', encoding="utf-8")
    errors = settings_contract_errors("trusted", root, git_cwd=root)
    assert errors == [
        "jqmc/_setting.py: tolerance loosened: _TOLERANCE[strict][float64][0] 1e-08 -> 0.001",
        "jqmc/_setting.py: tolerance loosened: _TOLERANCE[strict][float64][1] 1e-06 -> 0.01",
    ]
    (root / SETTING).write_text(_real(SETTING), encoding="utf-8")
    real_precision = _real(PRECISION)
    assert real_precision.count('"ao_eval": "float64",') == 1
    (root / PRECISION).write_text(real_precision.replace('"ao_eval": "float64",', '"ao_eval": "float32",'), encoding="utf-8")
    assert settings_contract_errors("trusted", root, git_cwd=root) == [
        "jqmc/_precision.py: reviewed setting changed: _FULL_PRECISION[ao_eval] 'float64' -> 'float32'"
    ]


def test_retained_framework_definition_removal_is_red_but_addition_is_green():
    from gdd.engine.contract_guard import test_definition_inclusion_errors

    before = "def test_retained():\n    pass\n"
    removed = "def helper():\n    pass\n"
    added = before + "\ndef test_added():\n    pass\n"
    unittest_before = """\
import unittest

class EvidenceCase(unittest.TestCase):
    def test_current_boundary(self):
        self.assertEqual(2 + 2, 4)

def test_module_boundary():
    pass
"""
    unittest_removed = unittest_before.replace(
        "    def test_current_boundary(self):\n        self.assertEqual(2 + 2, 4)\n",
        "    pass\n",
    )
    mixin_before = """\
from unittest import TestCase as Base

class SharedChecks:
    def test_current_boundary(self):
        self.assertEqual(2 + 2, 4)

class EvidenceCase(SharedChecks, Base):
    pass

def test_other_retained():
    assert 3 * 3 == 9
"""
    mixin_removed = mixin_before.replace(
        "    def test_current_boundary(self):\n        self.assertEqual(2 + 2, 4)\n",
        "    pass\n",
    )
    assert test_definition_inclusion_errors(before, removed, "tests/test_gdd_framework.py")
    assert test_definition_inclusion_errors(before, added, "tests/test_gdd_framework.py") == []
    assert test_definition_inclusion_errors(unittest_before, unittest_removed, "tests/test_gdd_framework.py")
    assert test_definition_inclusion_errors(mixin_before, mixin_removed, "tests/test_gdd_framework.py")


def test_only_the_approved_policy_one_to_two_bootstrap_is_allowed():
    from scripts.check_gdd_contract_changes import _candidate_policy_bootstrap_allowed

    assert _candidate_policy_bootstrap_allowed(1, 2)
    assert not _candidate_policy_bootstrap_allowed(2, 3)
    assert not _candidate_policy_bootstrap_allowed(None, 2)


def test_new_skip_capability_is_red():
    before = "def test_value():\n    assert True\n"
    after = "def test_value():\n    pytest.skip('not today')\n"
    assert skip_change_errors(before, after, "tests/test_toy.py", baseline_exists=True)
