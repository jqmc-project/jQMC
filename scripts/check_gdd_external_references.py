#!/usr/bin/env python3
"""Reject silent removal of the external (pymatgen / TurboRVB) reference literals.

The pymatgen Ewald references in
``tests/test_jqmc_kernel/test_coulomb_potential.py`` are module constants with
a ``None -> pytest.skip`` switch, and the TurboRVB literals in
``tests/test_jqmc_kernel/test_integration_turborvb_{AE,ECP}.py`` are recorded
in ``tests/trexio_example_files/turborvb_reference_provenance.json``. Setting a
pymatgen constant to ``None`` (or deleting it) or deleting / editing a
TurboRVB literal would remove the project's only cross-code reference
evidence, and only a manually dispatched full pytest run would notice. This
static check fails in the static GDD preflight instead:

* every reviewed ``PYMATGEN_E_*`` constant exists, is assigned exactly once at
  module level, and is a finite numeric literal (never ``None``), and no
  unreviewed ``PYMATGEN_E_*`` constant appears;
* every case of the TurboRVB provenance record names an existing test
  function, and every recorded reference literal (``historical``) is still
  assigned in that function with exactly the recorded value.
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

import ast
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PYMATGEN_FILE = "tests/test_jqmc_kernel/test_coulomb_potential.py"
REVIEWED_PYMATGEN_CONSTANTS = (
    "PYMATGEN_E_II_NACL_SC_HA",
    "PYMATGEN_E_II_MGO_B1_HA",
    "PYMATGEN_E_II_ZNO_WURTZITE_HA",
    "PYMATGEN_E_II_CUO_TENORITE_HA",
    "PYMATGEN_E_EI_H2_BOX_HA",
    "PYMATGEN_E_EE_H2_BOX_HA",
    "PYMATGEN_E_EI_H2_MONO_HA",
    "PYMATGEN_E_EE_H2_MONO_HA",
    "PYMATGEN_E_II_NACL_SLAB_VAC_HA",
    "PYMATGEN_E_II_WATER_LIH_FAR_HA",
    "PYMATGEN_E_II_WATER_LIH_ADS_HA",
    "PYMATGEN_E_TOTAL_WATER_LIH_FAR_HA",
    "PYMATGEN_E_TOTAL_WATER_LIH_ADS_HA",
    "PYMATGEN_E_EI_H4_SLAB_HA",
    "PYMATGEN_E_EE_H4_SLAB_HA",
    "PYMATGEN_E_EI_H4_SLAB_VACUUM_ELECTRON_HA",
    "PYMATGEN_E_EE_H4_SLAB_VACUUM_ELECTRON_HA",
    "PYMATGEN_E_TOTAL_H4_SLAB_VACUUM_ELECTRON_HA",
)
TURBORVB_PROVENANCE = "tests/trexio_example_files/turborvb_reference_provenance.json"
TURBORVB_MODULES = (
    "tests/test_jqmc_kernel/test_integration_turborvb_AE.py",
    "tests/test_jqmc_kernel/test_integration_turborvb_ECP.py",
)


def _numeric_literal(node: ast.AST) -> float | None:
    """The value of a (signed) int/float literal, else ``None``."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _numeric_literal(node.operand)
        return None if inner is None else (-inner if isinstance(node.op, ast.USub) else inner)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    return None


def pymatgen_errors(source: str, reviewed: tuple[str, ...] = REVIEWED_PYMATGEN_CONSTANTS) -> list[str]:
    tree = ast.parse(source)
    found: dict[str, list[ast.AST]] = {}
    for statement in tree.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            targets, value = [statement.target], statement.value
        elif isinstance(statement, ast.Assign):
            targets, value = [t for t in statement.targets if isinstance(t, ast.Name)], statement.value
        else:
            continue
        for target in targets:
            if target.id.startswith("PYMATGEN_E_"):
                found.setdefault(target.id, []).append(value)
    errors = []
    for name in sorted(set(found) - set(reviewed)):
        errors.append(f"unreviewed pymatgen reference constant {name} in {PYMATGEN_FILE}")
    for name in reviewed:
        values = found.get(name, [])
        if not values:
            errors.append(f"pymatgen reference constant {name} is missing from {PYMATGEN_FILE}")
            continue
        if len(values) != 1:
            errors.append(f"pymatgen reference constant {name} is assigned {len(values)} times in {PYMATGEN_FILE}")
        number = None if values[0] is None else _numeric_literal(values[0])
        if number is None or not math.isfinite(number):
            errors.append(f"pymatgen reference constant {name} must be a finite numeric literal (None would skip its test)")
    return errors


def _function_literals(function: ast.FunctionDef) -> dict:
    """Evaluate only the fixed ``*_turborvb`` / ``*_turbo`` reference assignments of a test body."""
    namespace: dict = {"np": np}
    for statement in function.body:
        if not isinstance(statement, ast.Assign):
            continue
        target = statement.targets[0]
        base = target.value if isinstance(target, ast.Subscript) else target
        if not isinstance(base, ast.Name):
            continue
        if base.id.startswith(("old_r_", "new_r_", "R_AS_turborvb_")) or base.id.endswith(("_turborvb", "_turbo")):
            exec(compile(ast.Module(body=[statement], type_ignores=[]), "<committed-reference-literal>", "exec"), namespace)
    return namespace


def _recorded_literal(names: dict, key: str):
    """The source value the provenance record's quantity ``key`` pins, or ``None`` if a name is missing."""
    try:
        if key == "kinetic":
            return names["kinc_ref_turborvb"]
        if key == "potential":
            return names["vpot_ref_turborvb"] + names.get("vpotoff_ref_turborvb", 0)
        if key == "wavefunction_ratio":
            return names["WF_ratio_ref_turborvb"]
        if key == "reweight":
            return names["reweight_turborvb"]
        if key in ("fa", "fb", "T_ratio", "final_ratio"):
            return names[f"{key}_turbo"]
        side, what = key.split("_", 1)
        mapping = {
            "geminal": f"geminal_{side}_turborvb",
            "inverse": f"geminal_{side}_inv_turborvb",
            "S": f"S_{side}_turborvb",
            "F": f"F_{side}_turborvb",
            "R_AS": f"R_AS_turborvb_{side}",
        }
        return names[mapping[what]]
    except (KeyError, ValueError):
        return None


def turborvb_errors(report: dict, module_sources: dict[str, str]) -> list[str]:
    functions = {}
    for path, source in module_sources.items():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                functions[node.name] = node
    errors = []
    cases = report.get("cases", {})
    if not cases:
        errors.append(f"{TURBORVB_PROVENANCE} records no TurboRVB cases")
    for name, case in sorted(cases.items()):
        if name not in functions:
            errors.append(f"TurboRVB reference case {name} has no test function in {', '.join(module_sources)}")
            continue
        names = _function_literals(functions[name])
        for key, row in sorted(case.get("quantities", {}).items()):
            literal = _recorded_literal(names, key)
            if literal is None:
                errors.append(f"TurboRVB reference literal for {name}/{key} is missing from its test")
                continue
            recorded = np.asarray(row["historical"], dtype=np.float64)
            value = np.asarray(literal, dtype=np.float64)
            if value.shape != recorded.shape or not np.array_equal(value, recorded):
                errors.append(f"TurboRVB reference literal for {name}/{key} differs from the provenance record")
    return errors


def validation_errors(root: Path = ROOT, overrides: dict[str, str] | None = None) -> list[str]:
    """All violations; ``overrides`` maps a repository path to replacement text (self-tests)."""
    overrides = overrides or {}

    def read(path: str) -> str:
        return overrides[path] if path in overrides else (root / path).read_text(encoding="utf-8")

    errors = pymatgen_errors(read(PYMATGEN_FILE))
    report = json.loads(read(TURBORVB_PROVENANCE))
    errors.extend(turborvb_errors(report, {path: read(path) for path in TURBORVB_MODULES}))
    return errors


def main() -> int:
    try:
        errors = validation_errors()
    except (OSError, SyntaxError, ValueError, KeyError) as exc:
        errors = [str(exc)]
    if errors:
        print("GDD external reference literal contract failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(
        "GDD external reference literal contract passed "
        f"({len(REVIEWED_PYMATGEN_CONSTANTS)} pymatgen constants; TurboRVB provenance literals present)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
