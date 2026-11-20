"""Host-local cache of third-party test inputs that git does not carry.

``tests/qe_example_files/pseudopotentials.toml`` pins every pseudopotential the
QE example tree reads: its source URL, the sha256 of the upstream bytes and the
paths in the tree where QE expects it. This module resolves those entries into a
host-local cache and builds the per-session mirror of the example tree that the
tests read through :func:`qe_example_files_root`.

Cache
    ``$JQMC_TEST_DATA_CACHE`` (a leading ``~`` is expanded), default
    ``~/.cache/jqmc/test-data``; never inside the checkout. A file lives at
    ``<cache>/<sha256>/<name>``. An exclusive ``flock`` on ``<cache>/.lock``
    serializes the xdist workers, MPI ranks and concurrent jobs of one host. A
    missing file is downloaded into a temporary file in its cache directory,
    checked against the pinned sha256 and published with ``os.replace``. A
    cached file whose sha256 differs is an error, not a silent re-download.

Every failure raises :class:`TestDataCacheError`, which fails the test session;
nothing is skipped and there is no second source.

``python -m tests._test_data_cache`` pre-populates the cache.
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

import contextlib
import fcntl
import hashlib
import http.client
import os
import shutil
import ssl
import tempfile
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QE_EXAMPLE_FILES = ROOT / "tests" / "qe_example_files"
MANIFEST = QE_EXAMPLE_FILES / "pseudopotentials.toml"
CACHE_ENV = "JQMC_TEST_DATA_CACHE"
DEFAULT_CACHE = "~/.cache/jqmc/test-data"
# The mirror root of the running test session; inherited by its subprocesses.
MIRROR_ENV = "JQMC_QE_EXAMPLE_FILES_MIRROR"
DOWNLOAD_TIMEOUT_S = 60


class TestDataCacheError(RuntimeError):
    """A pinned test input cannot be provided; the test session must fail."""

    __test__ = False  # not a pytest test class


@dataclass(frozen=True)
class ManifestEntry:
    """One pinned file: where it comes from and where the example tree expects it."""

    name: str
    url: str
    sha256: str
    locations: tuple[str, ...]


def manifest_entries(manifest: Path = MANIFEST) -> tuple[ManifestEntry, ...]:
    """The pinned entries of ``manifest``, validated."""
    document = tomllib.loads(manifest.read_text(encoding="utf-8"))
    entries = []
    for row in document["pseudopotential"]:
        entry = ManifestEntry(name=row["name"], url=row["url"], sha256=row["sha256"].lower(), locations=tuple(row["locations"]))
        if len(entry.sha256) != 64 or any(char not in "0123456789abcdef" for char in entry.sha256):
            raise TestDataCacheError(f"{manifest}: {entry.name}: sha256 must be 64 hex digits")
        for location in entry.locations:
            if Path(location).name != entry.name or Path(location).is_absolute() or ".." in Path(location).parts:
                raise TestDataCacheError(f"{manifest}: {entry.name}: invalid location {location!r}")
        entries.append(entry)
    return tuple(entries)


def manifest_locations(manifest: Path = MANIFEST) -> dict[str, ManifestEntry]:
    """Repository-relative path (``tests/qe_example_files/...``) of every pinned location."""
    base = manifest.parent.relative_to(ROOT)
    return {(base / location).as_posix(): entry for entry in manifest_entries(manifest) for location in entry.locations}


def cache_dir() -> Path:
    """The host-local cache directory (not created)."""
    # Resolve before comparing: ROOT is resolved, so a relative sibling
    # (``../cache``) must not be read as inside it, and a symlink alias that
    # points into the checkout must be.
    directory = Path(os.environ.get(CACHE_ENV) or DEFAULT_CACHE).expanduser().resolve()
    if directory == ROOT or ROOT in directory.parents:
        raise TestDataCacheError(f"{CACHE_ENV}={directory} is inside the checkout {ROOT}; choose a host-local directory")
    return directory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(entry: ManifestEntry, target: Path) -> None:
    """Fetch ``entry.url`` into ``target`` through a verified temporary file in its directory."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{entry.name}.", suffix=".part", dir=target.parent)
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as output:
            # urlopen follows redirects and honors the proxy environment variables.
            with urllib.request.urlopen(entry.url, timeout=DOWNLOAD_TIMEOUT_S) as response:
                for block in iter(lambda: response.read(1 << 16), b""):
                    digest.update(block)
                    output.write(block)
        if digest.hexdigest() != entry.sha256:
            raise TestDataCacheError(
                f"download of {entry.url} has sha256 {digest.hexdigest()}, the manifest {MANIFEST} pins {entry.sha256}"
            )
        os.chmod(temporary, 0o444)
        os.replace(temporary, target)
    except (OSError, http.client.HTTPException) as error:  # URLError, HTTPError and timeouts are OSError
        hint = (
            " Python's OpenSSL does not trust the certificate chain (an intercepting proxy?): "
            "point SSL_CERT_FILE at the host's CA bundle."
            if isinstance(getattr(error, "reason", error), ssl.SSLCertVerificationError)
            else ""
        )
        raise TestDataCacheError(
            f"cannot download {entry.url} into {target.parent}: {error}.{hint} "
            f"The test session needs network access once per host to fill {CACHE_ENV}, "
            "or run `python -m tests._test_data_cache` where the network is available."
        ) from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ensure_cached(entries: tuple[ManifestEntry, ...] | None = None) -> dict[str, Path]:
    """Cached path of every entry (by name), downloading the missing ones under the host lock."""
    entries = manifest_entries() if entries is None else entries
    directory = cache_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        lock = (directory / ".lock").open("a")
    except OSError as error:
        raise TestDataCacheError(f"cannot use the test-data cache {directory} ({CACHE_ENV}): {error}") from error
    cached: dict[str, Path] = {}
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for entry in entries:
            target = directory / entry.sha256 / entry.name
            if target.exists():
                actual = _sha256(target)
                if actual != entry.sha256:
                    raise TestDataCacheError(
                        f"cached {target} has sha256 {actual}, the manifest pins {entry.sha256}: "
                        "the cache is corrupted; remove the file to download it again"
                    )
            else:
                target.parent.mkdir(exist_ok=True)
                _download(entry, target)
            cached[entry.name] = target
    return cached


def build_qe_example_mirror(destination: Path) -> Path:
    """Build the mirror of ``tests/qe_example_files`` at ``destination`` (an empty directory).

    Directories are recreated; every checkout file becomes a symbolic link to the
    checkout, and every manifest location a symbolic link to the cached file.
    """
    entries = manifest_entries()
    cached = ensure_cached(entries)
    for directory, children, files in os.walk(QE_EXAMPLE_FILES):
        relative = Path(directory).relative_to(QE_EXAMPLE_FILES)
        children.sort()
        for child in children:
            (destination / relative / child).mkdir()
        for name in sorted(files):
            os.symlink(Path(directory) / name, destination / relative / name)
    for entry in entries:
        for location in entry.locations:
            link = destination / location
            if os.path.lexists(link):
                raise TestDataCacheError(
                    f"tests/qe_example_files/{location} is in the checkout and pinned by {MANIFEST.name}; "
                    "third-party pseudopotentials are not kept in git"
                )
            if not link.parent.is_dir():
                raise TestDataCacheError(f"{MANIFEST.name}: {location}: no directory {link.parent.relative_to(destination)}")
            os.symlink(cached[entry.name], link)
    return destination


def create_session_mirror() -> Path:
    """Build this process's mirror in a fresh temporary directory and publish it to :data:`MIRROR_ENV`."""
    destination = Path(tempfile.mkdtemp(prefix="jqmc-qe-example-files-"))
    try:
        build_qe_example_mirror(destination)
    except BaseException:
        shutil.rmtree(destination)
        raise
    os.environ[MIRROR_ENV] = str(destination)
    return destination


def remove_session_mirror(destination: Path) -> None:
    """Remove a mirror built by :func:`create_session_mirror` (links only; targets are kept)."""
    shutil.rmtree(destination)
    if os.environ.get(MIRROR_ENV) == str(destination):
        del os.environ[MIRROR_ENV]


@contextlib.contextmanager
def qe_example_mirror():
    """The QE example mirror for code that imports test helpers outside pytest.

    Inside a pytest session the session's mirror is reused. Otherwise a mirror is
    built for the duration of the block (downloading into the cache if needed)
    and removed afterwards.
    """
    if os.environ.get(MIRROR_ENV):
        yield Path(os.environ[MIRROR_ENV])
        return
    mirror = create_session_mirror()
    try:
        yield mirror
    finally:
        remove_session_mirror(mirror)


def qe_example_files_root() -> Path:
    """Root of the running test session's mirror of ``tests/qe_example_files``."""
    mirror = os.environ.get(MIRROR_ENV)
    if not mirror:
        raise TestDataCacheError(
            f"{MIRROR_ENV} is not set: tests/conftest.py builds the QE example mirror in pytest_configure; "
            "read tests/qe_example_files only inside a pytest session"
        )
    return Path(mirror)


def main() -> int:
    for name, path in ensure_cached().items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
