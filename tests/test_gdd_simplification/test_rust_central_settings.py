"""Native constants and test budgets obey the shared source contract without Rust/JAX bootstrap."""

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
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check_no_inline_tolerances.py"
GENERATOR = ROOT / "jqmc/rs/generate_setting_consts.py"
pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _in_test(body: str) -> str:
    return "#[cfg(test)] mod tests { #[test] fn regression() { " + body + " } }\n"


@pytest.mark.parametrize(
    "source",
    [
        _in_test("assert!((got - expected).abs() <= 1e-12);"),
        _in_test("let banana = 1e-12; let bound = banana; assert!((got - expected).abs() < bound);"),
        _in_test("fn check(a: f64, b: f64, tol: f64) { assert!((a-b).abs() <= tol); } check(got, expected, 1e-12);"),
        _in_test("let bound = crate::setting_consts::ATOL_PYTHON_FP64_LOOSE * 2.0; assert!(err.abs() <= bound);"),
        _in_test("assert!(err.abs() < central.max(1e-6));"),
        "const TOL: f64 = 1e-12; #[cfg(test)] mod tests { use super::*; #[test] fn f() { assert!(err.abs() <= TOL); } }",
    ],
    ids=["direct", "alias", "helper-argument", "scaled-central", "literal-floor", "parent-constant"],
)
def test_rust_inline_budget_cli_rejects_literals_and_aliases(tmp_path, source):
    """The same actual CLI called by the hook rejects direct, indirect and relocated budgets."""
    path = tmp_path / "fixture.rs"
    path.write_text(source)
    result = subprocess.run([sys.executable, str(CHECKER), str(path)], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "must come from jqmc/_setting.py" in result.stderr


def test_native_setting_generator_preserves_literal_binary64_bits(tmp_path):
    """The emitted constants retain authored decimal bits and need no scientific imports."""
    source = (ROOT / "jqmc/_setting.py").read_text()
    fixture = tmp_path / "_setting.py"
    fixture.write_text(source + '\nraise RuntimeError("source must never be imported")\n')
    output = tmp_path / "settings.rs"
    result = subprocess.run(
        [sys.executable, str(GENERATOR), str(fixture), str(output)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    actual = {
        name: int(bits, 16)
        for name, bits in re.findall(r"pub const (\w+): f64 = f64::from_bits\(0x([\da-f]+)\);", output.read_text())
    }
    tree = ast.parse(source)
    literals = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert len(actual) >= 10
    for name, bits in actual.items():
        expected = float(ast.literal_eval(literals[name]))
        assert struct.pack(">Q", bits) == struct.pack(">d", expected), name
    # An infinite budget would be a vacuous bound: the build helper fails before writing Rust.
    assignment = "EPS_safe_distance = 1.0e-12"
    fixture.write_text(
        source.replace(assignment, "EPS_safe_distance = 1e999") + '\nraise RuntimeError("source must never be imported")\n'
    )
    output.unlink()
    result = subprocess.run(
        [sys.executable, str(GENERATOR), str(fixture), str(output)], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0 and "EPS_safe_distance" in result.stderr, result.stderr
    assert not output.exists() and "source must never be imported" not in result.stderr
