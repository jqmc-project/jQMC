"""Version and git commit of the running jQMC code (issue #183).

Every run reports where its code comes from in two fixed-format lines, which
downstream tools parse:

    version = 1.2.3
    git commit = 0123456789abcdef0123456789abcdef01234567

Resolution order, evaluated once per process:

1. Live git. When this package directory is tracked by an enclosing git work
   tree (a clone with ``pip install -e .``), ``git rev-parse HEAD`` gives the
   commit, so it follows ``git pull`` / ``git checkout`` without reinstalling.
   The version is computed live with ``setuptools_scm`` when it is importable;
   otherwise the install-time metadata is reported as ``(metadata, may be stale)``.
2. Build time. Otherwise (a wheel, an sdist install, PyPI) the full hash that
   the ``version_file_template`` in ``pyproject.toml`` wrote into
   ``jqmc/_version.py`` when the distribution was built.
3. ``unknown``.

Nothing here raises: a missing git binary, a timeout or a missing
``_version.py`` falls through to the next source.
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

import functools
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: Seconds allowed for each git subprocess call.
GIT_TIMEOUT_SECONDS = 5.0

_PACKAGE_DIR = Path(__file__).resolve().parent
_FULL_HASH = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class BuildInfo:
    """Where the running code comes from."""

    version: str
    git_commit: str

    def banner_lines(self) -> list[str]:
        """The fixed-format lines every run prints."""
        return [f"version = {self.version}", f"git commit = {self.git_commit}"]


def _git(args: list[str]) -> str | None:
    """Stripped stdout of ``git -C <package dir> <args>``, or None on any failure."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(_PACKAGE_DIR), *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _full_hash(value: object) -> str | None:
    """A 40-digit lowercase hash from ``value``, dropping setuptools_scm's ``g`` prefix."""
    if not isinstance(value, str):
        return None
    candidate = value[1:] if value.startswith("g") else value
    return candidate if _FULL_HASH.fullmatch(candidate) else None


def _live_git_commit() -> tuple[str, Path] | None:
    """HEAD and work-tree root when this package's source is tracked by git."""
    if not any((parent / ".git").exists() for parent in (_PACKAGE_DIR, *_PACKAGE_DIR.parents)):
        return None
    # Tracked, not merely located inside some repository: a virtual environment
    # placed in a project directory would otherwise report that project's HEAD.
    if _git(["ls-files", "--error-unmatch", "__init__.py"]) is None:
        return None
    toplevel = _git(["rev-parse", "--show-toplevel"])
    commit = _full_hash(_git(["rev-parse", "HEAD"]))
    if toplevel is None or commit is None:
        return None
    return commit, Path(toplevel)


def _metadata_version() -> str | None:
    """The installed distribution's version, or the build-time one."""
    try:
        from importlib.metadata import version

        return version("jqmc")
    except Exception:
        pass
    try:
        from ._version import __version__

        return str(__version__)
    except Exception:
        return None


def _live_version(toplevel: Path) -> str | None:
    """The version setuptools_scm computes for the checkout now."""
    try:
        from setuptools_scm import get_version

        return str(get_version(root=str(toplevel), local_scheme="no-local-version"))
    except Exception:
        return None


def _build_time_commit() -> str | None:
    """The full hash written into ``_version.py`` when the distribution was built."""
    try:
        from . import _version
    except Exception:
        return None
    return _full_hash(getattr(_version, "__git_commit_full__", None))


@functools.lru_cache(maxsize=1)
def build_info() -> BuildInfo:
    """Resolve the version and git commit once per process."""
    live = _live_git_commit()
    if live is not None:
        commit, toplevel = live
        version = _live_version(toplevel)
        if version is None:
            metadata = _metadata_version()
            version = f"{metadata} (metadata, may be stale)" if metadata else "unknown"
        return BuildInfo(version=version, git_commit=commit)
    return BuildInfo(version=_metadata_version() or "unknown", git_commit=_build_time_commit() or "unknown")


_banner_logged = False


def _is_root_rank() -> bool:
    """Rank 0 of an already initialized MPI run, or True when mpi4py is not loaded (never initializes MPI)."""
    mpi = sys.modules.get("mpi4py.MPI")
    if mpi is None or not mpi.Is_initialized() or mpi.Is_finalized():
        return True
    return mpi.COMM_WORLD.Get_rank() == 0


def _emits_info(logger: logging.Logger) -> bool:
    """Whether an INFO record from ``logger`` reaches a handler that accepts INFO.

    Follows the propagation chain the way ``Logger.callHandlers`` does. A logger
    enabled for INFO whose only handlers are set to WARNING drops the record, so
    it does not count (PR #184).
    """
    if not logger.isEnabledFor(logging.INFO):
        return False
    current: logging.Logger | None = logger
    while current is not None:
        if any(handler.level <= logging.INFO for handler in current.handlers):
            return True
        if not current.propagate:
            return False
        current = current.parent
    return False


def log_build_info_once(logger: logging.Logger) -> None:
    """Report the version lines the first time in this process.

    Through ``logger`` when an INFO record reaches a handler that accepts it
    (the jqmc CLI configures one); otherwise, as for a Python API process with
    default logging (level WARNING, no handler) or with only WARNING handlers,
    directly on stderr of rank 0, so the lines are never silently dropped
    (PR #184).
    """
    global _banner_logged
    if _banner_logged:
        return
    lines = build_info().banner_lines()
    if _emits_info(logger):
        for line in lines:
            logger.info(line)
    elif _is_root_rank():
        sys.stderr.write("".join(f"{line}\n" for line in lines))
        sys.stderr.flush()
    _banner_logged = True
