"""The version and git commit every jQMC run reports (issue #183).

The resolver in ``jqmc/_build_info.py`` is exercised with the package directory, the
git subprocess and ``jqmc._version`` replaced, so each source of the commit is checked
independently of the checkout or installation running the tests.
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
import subprocess
import sys
import types
from logging import INFO, Handler, getLogger

import h5py
import pytest
from click.testing import CliRunner

from jqmc import _build_info
from jqmc.hamiltonians import Hamiltonian_data
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

LIVE_HEAD = "0123456789abcdef0123456789abcdef01234567"
BUILT_HEAD = "fedcba9876543210fedcba9876543210fedcba98"
BANNER = re.compile(r"version = \S.*\ngit commit = (?:[0-9a-f]{40}|unknown)\n")


@pytest.fixture(autouse=True)
def _fresh_resolution(monkeypatch):
    """Every test resolves from scratch and leaves no cached result behind."""
    resolver = _build_info.build_info  # the cached function, even if a test replaces the attribute
    resolver.cache_clear()
    monkeypatch.setattr(_build_info, "_banner_logged", False)
    yield
    resolver.cache_clear()


def _fake_git(monkeypatch, responses, calls=None):
    """Replace ``subprocess.run`` in the resolver; ``responses`` maps a git subcommand to stdout or an exception."""

    def run(command, **kwargs):
        assert command[:3] == ["git", "-C", str(_build_info._PACKAGE_DIR)]
        assert kwargs["timeout"] == _build_info.GIT_TIMEOUT_SECONDS
        key = " ".join(command[3:])
        if calls is not None:
            calls.append(key)
        outcome = responses.get(key, subprocess.CalledProcessError(128, command))
        if isinstance(outcome, BaseException):
            if isinstance(outcome, subprocess.CalledProcessError):
                return subprocess.CompletedProcess(command, outcome.returncode, "", "fatal")
            raise outcome
        return subprocess.CompletedProcess(command, 0, outcome + "\n", "")

    monkeypatch.setattr(_build_info.subprocess, "run", run)


def _checkout(monkeypatch, tmp_path):
    """A package directory inside a git work tree."""
    root = tmp_path / "checkout"
    (root / ".git").mkdir(parents=True)
    (root / "jqmc").mkdir()
    monkeypatch.setattr(_build_info, "_PACKAGE_DIR", root / "jqmc")
    return root


def _built_version_module(monkeypatch, **attributes):
    """Install a stand-in ``jqmc._version`` carrying ``attributes``."""
    module = types.ModuleType("jqmc._version")
    module.__version__ = "1.2.3"
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, "jqmc._version", module)
    import jqmc

    monkeypatch.setattr(jqmc, "_version", module, raising=False)


LIVE_GIT = {
    "ls-files --error-unmatch __init__.py": "__init__.py",
    "rev-parse --show-toplevel": "/checkout",
    "rev-parse HEAD": LIVE_HEAD,
}


def test_live_git_checkout_reports_its_current_head_and_live_version(monkeypatch, tmp_path):
    """A tracked source checkout reports ``git rev-parse HEAD`` and the version computed now."""
    _checkout(monkeypatch, tmp_path)
    _fake_git(monkeypatch, LIVE_GIT)
    _built_version_module(monkeypatch, __git_commit_full__="g" + BUILT_HEAD)
    monkeypatch.setattr(_build_info, "_live_version", lambda toplevel: "1.2.4")
    info = _build_info.build_info()
    assert info == _build_info.BuildInfo(version="1.2.4", git_commit=LIVE_HEAD)
    assert info.banner_lines() == ["version = 1.2.4", f"git commit = {LIVE_HEAD}"]


def test_live_git_without_setuptools_scm_marks_the_metadata_version_stale(monkeypatch, tmp_path):
    """Without a live version the install-time one is reported, marked as possibly stale."""
    _checkout(monkeypatch, tmp_path)
    _fake_git(monkeypatch, LIVE_GIT)
    monkeypatch.setattr(_build_info, "_live_version", lambda toplevel: None)
    monkeypatch.setattr(_build_info, "_metadata_version", lambda: "1.2.3")
    info = _build_info.build_info()
    assert info.version == "1.2.3 (metadata, may be stale)"
    assert info.git_commit == LIVE_HEAD


def _git_failure(name):
    """The git response that produces each fallback case."""
    return {
        "untracked": {"ls-files --error-unmatch __init__.py": subprocess.CalledProcessError(1, "git")},
        "no-git-binary": {"ls-files --error-unmatch __init__.py": FileNotFoundError("git")},
        "timeout": {"rev-parse HEAD": subprocess.TimeoutExpired("git", _build_info.GIT_TIMEOUT_SECONDS)},
        "malformed-head": {"rev-parse HEAD": "not-a-hash"},
    }[name]


@pytest.mark.parametrize("failure", ["untracked", "no-git-binary", "timeout", "malformed-head"])
def test_git_failures_fall_back_to_the_build_time_commit(monkeypatch, tmp_path, failure):
    """An untracked package, a missing git, a timeout or a bad HEAD use the commit written at build time."""
    _checkout(monkeypatch, tmp_path)
    _fake_git(monkeypatch, {**LIVE_GIT, **_git_failure(failure)})
    _built_version_module(monkeypatch, __git_commit_full__="g" + BUILT_HEAD)
    monkeypatch.setattr(_build_info, "_metadata_version", lambda: "1.2.3")
    assert _build_info.build_info() == _build_info.BuildInfo(version="1.2.3", git_commit=BUILT_HEAD)


def test_installed_distribution_without_git_reports_the_build_time_commit(monkeypatch, tmp_path):
    """A package that no git work tree tracks never reads a HEAD; the full hash from ``_version.py`` is reported.

    The temporary directory may itself lie inside a repository (as a virtual environment can):
    the tracking check then fails and no ``rev-parse HEAD`` is consulted.
    """
    package = tmp_path / "site-packages" / "jqmc"
    package.mkdir(parents=True)
    monkeypatch.setattr(_build_info, "_PACKAGE_DIR", package)
    calls = []
    _fake_git(monkeypatch, {}, calls)
    _built_version_module(monkeypatch, __git_commit_full__="g" + BUILT_HEAD)
    monkeypatch.setattr(_build_info, "_metadata_version", lambda: "1.2.3")
    assert _build_info.build_info() == _build_info.BuildInfo(version="1.2.3", git_commit=BUILT_HEAD)
    assert not any(call.startswith("rev-parse") for call in calls), calls


_INCOMPLETE_VERSION_FILES = {
    "abbreviated-only": {"__commit_id__": "gfedcba987"},
    "no-node": {"__git_commit_full__": None},
    "short-node": {"__git_commit_full__": "gfedcba9"},
    "no-version-file": None,
}


@pytest.mark.parametrize("version_file", ["abbreviated-only", "no-node", "short-node", "no-version-file"])
def test_missing_full_commit_is_reported_as_unknown(monkeypatch, tmp_path, version_file):
    """No live git and no full build-time hash give ``git commit = unknown``, never a short or absent line."""
    package = tmp_path / "jqmc"
    package.mkdir()
    monkeypatch.setattr(_build_info, "_PACKAGE_DIR", package)
    _fake_git(monkeypatch, {})
    attributes = _INCOMPLETE_VERSION_FILES[version_file]
    if attributes is None:
        monkeypatch.setitem(sys.modules, "jqmc._version", None)
    else:
        _built_version_module(monkeypatch, **attributes)
    monkeypatch.setattr(_build_info, "_metadata_version", lambda: None)
    info = _build_info.build_info()
    assert info == _build_info.BuildInfo(version="unknown", git_commit="unknown")
    assert info.banner_lines() == ["version = unknown", "git commit = unknown"]


def test_version_options_and_banner_print_the_fixed_lines(monkeypatch):
    """``jqmc-tool --version`` and the once-only banner print exactly the two lines.

    ``jqmc --version`` is checked through the in-process ``_cli`` boundary in
    ``test_build_info_routes.py``: ``jqmc_cli.main`` finalizes MPI.
    """
    from jqmc import jqmc_tool

    fixed = _build_info.BuildInfo(version="1.2.3", git_commit=LIVE_HEAD)
    monkeypatch.setattr(_build_info, "build_info", lambda: fixed)
    monkeypatch.setattr(jqmc_tool, "build_info", lambda: fixed)
    expected = f"version = 1.2.3\ngit commit = {LIVE_HEAD}\n"

    result = CliRunner().invoke(jqmc_tool._cli, ["--version"])
    assert result.exit_code == 0
    assert result.output == expected
    assert BANNER.fullmatch(result.output)

    records = []

    class _Collect(Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = getLogger("jqmc.test_build_info")
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "level", INFO)
    monkeypatch.setattr(logger, "handlers", [_Collect(level=INFO)])
    _build_info.log_build_info_once(logger)
    _build_info.log_build_info_once(logger)
    assert records == expected.splitlines()


def test_hamiltonian_files_record_the_code_version(monkeypatch, tmp_path):
    """``hamiltonian_data.h5`` stores ``jqmc_version`` and ``jqmc_git_commit`` of the writing code."""
    fixed = _build_info.BuildInfo(version="1.2.3", git_commit=LIVE_HEAD)
    monkeypatch.setattr(_build_info, "build_info", lambda: fixed)
    path = tmp_path / "hamiltonian_data.h5"
    build_canonical_hamiltonian("WF-PATTERN-001").save_to_hdf5(str(path))
    with h5py.File(path, "r") as handle:
        assert handle.attrs["jqmc_version"] == "1.2.3"
        assert handle.attrs["jqmc_git_commit"] == LIVE_HEAD
    Hamiltonian_data.load_from_hdf5(str(path))
