"""Fixture files that the repository pins by source and hash instead of committing.

``tests/qe_example_files/pseudopotentials.toml`` lists the third-party
pseudopotentials of the QE example tree: source URL, sha256 of the upstream
bytes and every path in the tree where QE reads one. The test session links each
of those paths, in its mirror of the tree, to the hash-checked file in the
host-local cache (``tests/_test_data_cache.py``).

GDD treats a pinned path as a file of the committed tree whose content identity
is its pinned sha256; it never downloads one. Runtime observation spells a path
in the session mirror as the checkout path it mirrors.
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

import hashlib
import os
from functools import cache
from pathlib import Path

from tests._test_data_cache import MANIFEST, MIRROR_ENV, QE_EXAMPLE_FILES, ROOT, manifest_locations, qe_example_files_root

# Name of the test helper that returns the session mirror of tests/qe_example_files.
QE_EXAMPLE_ROOT_HELPER = qe_example_files_root.__name__
# The manifest describes fixtures; it is not one.
PIN_MANIFEST = MANIFEST


@cache
def pinned_fixture_paths() -> dict[str, str]:
    """Pinned sha256 by repository-relative path (``tests/qe_example_files/...``)."""
    return {path: entry.sha256 for path, entry in manifest_locations().items()}


def _repository_relative(path: str | os.PathLike[str]) -> tuple[Path, str | None]:
    absolute = Path(os.path.normpath(Path(path) if Path(path).is_absolute() else ROOT / path))
    return absolute, absolute.relative_to(ROOT).as_posix() if absolute.is_relative_to(ROOT) else None


def fixture_exists(path: str | os.PathLike[str]) -> bool:
    """Whether ``path`` (absolute or repository-relative) exists in the checkout or is pinned."""
    absolute, relative = _repository_relative(path)
    return absolute.exists() or relative in pinned_fixture_paths()


def fixture_digest(path: str | os.PathLike[str]) -> str:
    """Content identity of a fixture file or directory of the committed tree.

    A checkout file contributes its bytes; a pinned path contributes its pinned
    sha256. A directory is digested over its relative paths in sorted order.
    """
    absolute, relative = _repository_relative(path)
    pinned = pinned_fixture_paths()
    digest = hashlib.sha256()
    if absolute.is_dir():
        members: dict[str, Path | str] = {
            child.relative_to(absolute).as_posix(): child for child in absolute.rglob("*") if child.is_file()
        }
        prefix = f"{relative}/" if relative is not None else None
        for location, sha256 in pinned.items():
            if prefix is not None and location.startswith(prefix):
                member = location.removeprefix(prefix)
                if member in members:
                    raise ValueError(f"{location} is both committed and pinned by {PIN_MANIFEST.name}")
                members[member] = sha256
        for member in sorted(members):
            content = members[member]
            digest.update(member.encode())
            digest.update(b"\0")
            digest.update(content.read_bytes() if isinstance(content, Path) else f"pinned-sha256:{content}".encode())
            digest.update(b"\0")
        return f"sha256:{digest.hexdigest()}"
    if relative in pinned:
        if absolute.exists():
            raise ValueError(f"{relative} is both committed and pinned by {PIN_MANIFEST.name}")
        return f"sha256:{pinned[relative]}"
    digest.update(absolute.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def checkout_path(path: str) -> str:
    """``path`` with the running test session's mirror root replaced by ``tests/qe_example_files``."""
    mirror = os.environ.get(MIRROR_ENV)
    if not mirror or not os.path.isabs(path):
        return path
    mirror = os.path.normpath(mirror)
    if os.path.commonpath((mirror, path)) != mirror:
        return path
    return os.path.normpath(os.path.join(QE_EXAMPLE_FILES, os.path.relpath(path, mirror)))
