"""The version lines reach the user through every production route (PR #184).

``test_build_info.py`` checks the resolver itself. Here each actual caller runs:
fresh Python processes construct or restart the MCMC / GFMC drivers with default or
configured logging; the jqmc run header, the workflow-launcher header and a jqmc-tool
subcommand report the lines; a merged restart checkpoint records the commit; the
in-process ``jqmc --version`` boundary leaves MPI usable; and the build-requirement
floor keeps the full commit through an sdist -> wheel build.
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

import os
import re
import subprocess
import sys
import tomllib
from logging import INFO
from pathlib import Path

import pytest
from click.testing import CliRunner
from mpi4py import MPI

from jqmc import _build_info
from jqmc._checkpoint import load_checkpoint_meta, merge_rank_checkpoints
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

ROOT = Path(__file__).resolve().parents[2]
DRIVERS = ["MCMC", "GFMC_t", "GFMC_n"]
#: GFMC_n has no default projection count (#690); these tests use 360 at the
#: class default alat = 0.1.
GFMC_N_PROJECTIONS = 360
TREXIO_FIXTURE = ROOT / "tests" / "trexio_example_files" / "H_ae_ccpvdz_cart.h5"

#: setuptools_scm releases that keep the full commit in a wheel built from the
#: sdist without .git start at 10.1.1; 9.2.2 and 10.0.x rewrite it to None
#: (measured with this repository's version_file_template).
SETUPTOOLS_SCM_SDIST_FULL_COMMIT_FLOOR = (10, 1, 1)


@pytest.fixture(autouse=True)
def _fresh_banner_state(monkeypatch):
    """Each test starts with an unreported, unresolved process state."""
    resolver = _build_info.build_info
    resolver.cache_clear()
    monkeypatch.setattr(_build_info, "_banner_logged", False)
    yield
    resolver.cache_clear()


def _expected_lines() -> list[str]:
    """The lines this checkout reports, resolved independently of any cached value."""
    return _build_info.build_info.__wrapped__().banner_lines()


def _version_lines(text: str) -> list[str]:
    """The ``version = `` / ``git commit = `` lines of an output, in order."""
    return [line for line in text.splitlines() if re.match(r"(version|git commit) = ", line)]


def _write_checkpoint(driver: str, directory: Path) -> Path:
    """Construct ``driver`` on a canonical Hamiltonian, save its rank state and merge a restart file."""
    import jqmc

    hamiltonian = build_canonical_hamiltonian("WF-PATTERN-001")
    extra = {"num_mcmc_per_measurement": GFMC_N_PROJECTIONS} if driver == "GFMC_n" else {}
    instance = getattr(jqmc, driver)(hamiltonian_data=hamiltonian, num_walkers=1, **extra)
    instance.save_to_hdf5(str(directory / "._restart_rank0.h5"))
    merged = directory / f"{driver}_restart.h5"
    merge_rank_checkpoints(
        str(merged),
        mpi_size=1,
        driver_type=driver,
        hamiltonian_data=hamiltonian,
        tmp_pattern=str(directory / "._restart_rank{rank}.h5"),
    )
    return merged


_FRESH_PROCESS = """
import logging, sys
from jqmc._precision import configure
configure(sys.argv[1])
import jqmc
driver, route = sys.argv[2], sys.argv[3]
cls = getattr(jqmc, driver)
if route == "construct-warning-handler":
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    route = "construct"
if route == "construct":
    from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian
    hamiltonian = build_canonical_hamiltonian("WF-PATTERN-001")
    extra = {"num_mcmc_per_measurement": 360} if driver == "GFMC_n" else {}
    cls(hamiltonian_data=hamiltonian, num_walkers=1, **extra)
    cls(hamiltonian_data=hamiltonian, num_walkers=1, **extra)
else:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    cls.load_from_hdf5(sys.argv[4], rank=0)
    cls.load_from_hdf5(sys.argv[4], rank=0)
"""


def _fresh_python(pytestconfig, *args: str) -> subprocess.CompletedProcess:
    """Run ``_FRESH_PROCESS`` in a new interpreter importing this checkout's jqmc."""
    environment = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    mode = pytestconfig.getoption("--precision-mode")
    return subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS, mode, *args],
        cwd=str(ROOT),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("driver", DRIVERS)
def test_python_api_driver_construction_reports_the_version_lines_once(pytestconfig, driver):
    """A fresh process with default logging (WARNING, no handler) constructing a driver twice prints the lines once."""
    completed = _fresh_python(pytestconfig, driver, "construct")
    assert completed.returncode == 0, completed.stderr
    assert _version_lines(completed.stderr) == _expected_lines()
    assert _version_lines(completed.stdout) == []


@pytest.mark.parametrize("driver", DRIVERS)
def test_python_api_driver_with_an_info_logger_and_warning_handler_reports_the_lines_once(pytestconfig, driver):
    """Root logger at INFO with only a WARNING console handler drops INFO records; the lines still appear once.

    An enabled logger plus an existing handler is not a handler that accepts INFO.
    """
    completed = _fresh_python(pytestconfig, driver, "construct-warning-handler")
    assert completed.returncode == 0, completed.stderr
    assert _version_lines(completed.stderr) == _expected_lines()
    assert _version_lines(completed.stdout) == []


@pytest.mark.parametrize("driver", DRIVERS)
def test_python_api_restart_reports_the_version_lines_once(pytestconfig, tmp_path, driver):
    """A fresh process that only restarts a driver from a checkpoint, with INFO logging, reports the lines once."""
    checkpoint = _write_checkpoint(driver, tmp_path)
    completed = _fresh_python(pytestconfig, driver, "restart", str(checkpoint))
    assert completed.returncode == 0, completed.stderr
    assert _version_lines(completed.stderr) == _expected_lines()


def test_jqmc_run_header_reports_the_version_lines(caplog):
    """The jqmc run banner logs the two lines after the ASCII header."""
    from jqmc import jqmc_cli

    caplog.set_level(INFO, logger="jqmc")
    jqmc_cli._print_header()
    assert _version_lines("\n".join(caplog.messages)) == _expected_lines()


def test_workflow_launcher_header_reports_the_version_lines(caplog):
    """The jqmc_workflow launcher banner logs the two lines."""
    from jqmc_workflow import launcher

    caplog.set_level(INFO)
    launcher._print_header()
    assert _version_lines("\n".join(caplog.messages)) == _expected_lines()


def test_jqmc_tool_subcommand_reports_the_version_lines_on_stderr():
    """A jqmc-tool subcommand prints the two lines on stderr and leaves its stdout free of them."""
    from jqmc import jqmc_tool

    result = CliRunner().invoke(jqmc_tool._cli, ["trexio", "show-info", str(TREXIO_FIXTURE)])
    assert result.exit_code == 0, result.output
    assert _version_lines(result.stderr) == _expected_lines()
    assert _version_lines(result.stdout) == []
    assert result.stdout.strip()


def test_merged_checkpoint_records_the_code_version(tmp_path):
    """A merged MCMC restart checkpoint stores jqmc_version and jqmc_git_commit of the writing code."""
    meta = load_checkpoint_meta(str(_write_checkpoint("MCMC", tmp_path)))
    version_line, commit_line = _expected_lines()
    assert f"version = {meta['jqmc_version']}" == version_line
    assert f"git commit = {meta['jqmc_git_commit']}" == commit_line


def test_jqmc_version_option_in_process_keeps_mpi_running(monkeypatch, capsys):
    """``jqmc --version`` through the in-process ``_cli`` boundary prints the lines and does not finalize MPI."""
    from jqmc import jqmc_cli

    monkeypatch.setattr(sys, "argv", ["jqmc", "--version"])
    with pytest.raises(SystemExit) as stop:
        jqmc_cli._cli()
    assert stop.value.code == 0
    assert capsys.readouterr().out.splitlines() == _expected_lines()
    assert MPI.Is_initialized()
    assert not MPI.Is_finalized()
    assert MPI.COMM_WORLD.allreduce(1) == MPI.COMM_WORLD.Get_size()


@pytest.mark.gdd_no_committed_artifact(
    reason="reads the repository build configuration as text; no scientific fixture artifact is materialized"
)
def test_build_requirement_floor_keeps_the_full_commit_through_sdist():
    """The setuptools_scm build requirement cannot select a release that drops the full commit in sdist -> wheel."""
    requires = tomllib.loads((ROOT / "pyproject.toml").read_text())["build-system"]["requires"]
    floors = [
        tuple(int(part) for part in match[1].split("."))
        for requirement in requires
        if (match := re.fullmatch(r"\s*setuptools_scm(?:\[[^]]*\])?\s*>=\s*([0-9][0-9.]*)\s*", requirement))
    ]
    assert len(floors) == 1, requires
    assert floors[0] >= SETUPTOOLS_SCM_SDIST_FULL_COMMIT_FLOOR
