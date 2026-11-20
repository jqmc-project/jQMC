"""``jqmc`` / ``jqmc-tool`` answer --version and --help without MPI, JAX or jqmc_rs (issue #185).

Importing ``mpi4py.MPI`` initializes MPI, which aborts outside ``srun`` on some MPI
stacks. Each invocation runs the light console entry in a fresh interpreter. The
check records which heavy modules were imported, both normally and with an import
hook that makes ``mpi4py``, ``jax`` and ``jqmc_rs`` unimportable, as on a broken
or absent MPI installation.
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

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from jqmc import _build_info, _cli_entry

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

ROOT = Path(__file__).resolve().parents[2]
HEAVY = ("mpi4py", "mpi4py.MPI", "jax", "jqmc_rs", "jqmc.jqmc_cli", "jqmc.jqmc_tool")

#: case -> (console script, arguments)
CASES = {
    "version": ("jqmc", ["--version"]),
    "V": ("jqmc", ["-V"]),
    "help": ("jqmc", ["--help"]),
    "h": ("jqmc", ["-h"]),
    "no-arguments": ("jqmc", []),
    "two-inputs": ("jqmc", ["a.toml", "b.toml"]),
    "missing-input": ("jqmc", ["missing-input.toml"]),
    "tool-version": ("jqmc-tool", ["--version"]),
}

_RUN = """
import json, sys
report, block, script, *args = sys.argv[1:]
if block == "block":
    class _Unimportable:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in {"mpi4py", "jax", "jqmc_rs"}:
                raise ImportError(f"{name} is unavailable here")
    sys.meta_path.insert(0, _Unimportable())
sys.argv = [script, *args]
from jqmc import _cli_entry
code = None
try:
    (_cli_entry.tool_main if script == "jqmc-tool" else _cli_entry.main)()
except SystemExit as stop:
    code = stop.code
heavy = [name for name in %r if name in sys.modules]
with open(report, "w") as handle:
    json.dump({"code": code, "heavy": heavy}, handle)
sys.exit(code)
""" % (HEAVY,)


def _invoke(tmp_path: Path, case: str, block: bool) -> tuple[subprocess.CompletedProcess, dict]:
    script, args = CASES[case]
    report = tmp_path / "report.json"
    environment = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    completed = subprocess.run(
        [sys.executable, "-c", _RUN, str(report), "block" if block else "plain", script, *args],
        cwd=str(tmp_path),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert report.is_file(), completed.stderr
    return completed, json.loads(report.read_text())


def _check_outcome(case: str, completed: subprocess.CompletedProcess) -> None:
    """Output and exit status of each invocation, unchanged from the pre-#185 CLI."""
    version = "".join(f"{line}\n" for line in _build_info.build_info.__wrapped__().banner_lines())
    if case in ("version", "V", "tool-version"):
        assert (completed.returncode, completed.stdout) == (0, version), completed.stderr
    elif case in ("help", "h", "no-arguments"):
        assert completed.returncode == (2 if case == "no-arguments" else 0), completed.stderr
        assert completed.stdout == _cli_entry.CLI_USAGE
    else:
        message = {
            "two-inputs": "jqmc: more than one input TOML file is not acceptable.",
            "missing-input": "jqmc: input file 'missing-input.toml' does not exist.",
        }[case]
        assert completed.returncode == 1
        assert completed.stdout == ""
        assert completed.stderr == _cli_entry.CLI_USAGE + message + "\n"


@pytest.mark.parametrize("case", list(CASES))
def test_light_invocations_import_no_mpi_jax_or_rust(tmp_path, case):
    """Version, help, no-argument and argument-error invocations finish without importing MPI, JAX or jqmc_rs."""
    completed, report = _invoke(tmp_path, case, block=False)
    assert report["heavy"] == []
    _check_outcome(case, completed)


@pytest.mark.parametrize("case", list(CASES))
def test_light_invocations_succeed_when_mpi_jax_and_rust_cannot_be_imported(tmp_path, case):
    """The same invocations give the same output when importing mpi4py, jax or jqmc_rs raises."""
    completed, report = _invoke(tmp_path, case, block=True)
    assert report["heavy"] == []
    _check_outcome(case, completed)


@pytest.mark.gdd_no_committed_artifact(
    reason="reads the repository build configuration as text; no scientific fixture artifact is materialized"
)
def test_console_scripts_point_at_the_light_entry():
    """``jqmc`` and ``jqmc-tool`` are installed as the light entries, and the CLI shares their usage text."""
    from jqmc import jqmc_cli

    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["jqmc"] == "jqmc._cli_entry:main"
    assert scripts["jqmc-tool"] == "jqmc._cli_entry:tool_main"
    assert jqmc_cli._CLI_USAGE is _cli_entry.CLI_USAGE


def test_job_and_tool_arguments_delegate_to_the_full_clis(monkeypatch, tmp_path):
    """An existing input file goes to ``jqmc_cli.main``; any other jqmc-tool arguments go to ``jqmc_tool._cli_main``."""
    from jqmc import jqmc_cli, jqmc_tool

    calls = []
    monkeypatch.setattr(jqmc_cli, "main", lambda: calls.append("jqmc") or "job")
    monkeypatch.setattr(jqmc_tool, "_cli_main", lambda: calls.append("jqmc-tool") or "tool")
    toml_file = tmp_path / "input.toml"
    toml_file.write_text("[control]\n")
    monkeypatch.setattr(sys, "argv", ["jqmc", str(toml_file)])
    assert _cli_entry.main() == "job"
    monkeypatch.setattr(sys, "argv", ["jqmc-tool", "trexio", "show-info", "x.h5"])
    assert _cli_entry.tool_main() == "tool"
    assert calls == ["jqmc", "jqmc-tool"]


def test_delegated_clis_receive_the_original_arguments(monkeypatch, tmp_path):
    """The full CLIs see the user's argv unchanged when the light entry hands over.

    The argv is recorded inside each delegated function, so an entry that dropped or rewrote the
    arguments before delegating (the real CLI would then print usage and exit 2 without reading the
    input) fails here.
    """
    from jqmc import jqmc_cli, jqmc_tool

    received = []
    monkeypatch.setattr(jqmc_cli, "main", lambda: received.append(("jqmc", list(sys.argv))) or "job")
    monkeypatch.setattr(jqmc_tool, "_cli_main", lambda: received.append(("jqmc-tool", list(sys.argv))) or "tool")
    toml_file = tmp_path / "input.toml"
    toml_file.write_text("[control]\n")
    job_argv = ["jqmc", str(toml_file)]
    tool_argv = ["jqmc-tool", "trexio", "show-info", "x.h5"]
    monkeypatch.setattr(sys, "argv", list(job_argv))
    assert _cli_entry.main() == "job"
    monkeypatch.setattr(sys, "argv", list(tool_argv))
    assert _cli_entry.tool_main() == "tool"
    assert received == [("jqmc", job_argv), ("jqmc-tool", tool_argv)]
