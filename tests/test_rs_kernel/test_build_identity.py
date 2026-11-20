"""Guard against running Rust parity tests with a stale native extension.

A stale ``jqmc_rs.abi3.so`` -- one older than the
Rust source it is meant to embody -- silently passes the entire parity suite:
``import jqmc_rs`` succeeds, ``conftest.py``'s "hard error if it cannot import"
sentinel is satisfied, and nothing anywhere checks that the loaded binary was
compiled from the CURRENT source. A developer whose incremental
build silently no-op'd can therefore run every equivalence test against a
different revision of the kernels than the one in the tree.

``build.rs`` folds a dependency-free FNV-1a 64-bit digest over the crate's
Rust source and Cargo/build/generator/central-setting inputs at BUILD time and bakes it into
the extension as ``jqmc_rs.__source_fnv64__``. This test recomputes the SAME
digest from the working tree and asserts they match, so a stale binary fails
loudly here instead of being tested silently everywhere else.

The digest scheme is duplicated (not shared) between ``build.rs`` and this file
ON PURPOSE: the whole point is an INDEPENDENT recomputation on the two sides of
the compile boundary. If the two implementations drift, this test goes red --
which is the correct signal that the guard itself needs attention.
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

import pathlib
import ast
import struct

import pytest

import jqmc_rs

# ``without_chdir``: every path here is absolute (crate root resolved from
# ``__file__``), so the test manages its own CWD and must opt out of the autouse
# ``_isolate_cwd`` chdir the rest of the suite uses.
pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

# FNV-1a 64-bit (matches build.rs). Not a crypto hash -- a change detector.
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_U64 = (1 << 64) - 1

# tests/test_rs_kernel/ -> repo root -> jqmc/rs (the crate root build.rs runs in).
_CRATE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "jqmc" / "rs"


def _fnv_update(h: int, data: bytes) -> int:
    for b in data:
        h = ((h ^ b) * _FNV_PRIME) & _U64
    return h


def _source_files() -> list[tuple[str, pathlib.Path]]:
    """Same file set build.rs hashes: Rust sources and all build/setting inputs,
    keyed by the crate-root-relative POSIX path, sorted by that path.
    """
    files = [(p.relative_to(_CRATE_ROOT).as_posix(), p) for p in (_CRATE_ROOT / "src").rglob("*.rs")]
    for relative in ("Cargo.toml", "build.rs", "generate_setting_consts.py", "../_setting.py"):
        files.append((relative, _CRATE_ROOT / relative))
    files.sort(key=lambda t: t[0])
    return files


def _source_fnv64() -> str:
    digest = _FNV_OFFSET
    for rel, path in _source_files():
        digest = _fnv_update(digest, rel.encode())
        digest = _fnv_update(digest, b"\0")
        digest = _fnv_update(digest, path.read_bytes())
        digest = _fnv_update(digest, b"\0")
    return f"{digest:016x}"


def test_source_tree_is_present():
    """Precondition: the guard is meaningful only when the source tree is on disk
    (a test run from the source checkout always has it). A missing tree
    would make ``_source_fnv64`` hash an empty file set and match nothing -- fail
    loudly rather than pass vacuously.
    """
    assert _CRATE_ROOT.is_dir(), f"crate root not found at {_CRATE_ROOT}"
    files = _source_files()
    assert len(files) > 10, f"suspiciously few Rust source files found ({len(files)}) under {_CRATE_ROOT}"
    assert any(rel == "src/lib.rs" for rel, _ in files), "src/lib.rs missing from the hashed file set"


def test_imported_extension_matches_current_source():
    """The imported ``jqmc_rs`` was built from the CURRENT Rust source.

    If this fails, ``jqmc_rs.abi3.so`` is stale: rebuild the extension (a clean,
    isolated wheel build -- ``pip install -e . --no-build-isolation`` alone can
    silently keep an old ``.so`` if ``setuptools-rust`` is absent) before trusting any parity result in this suite.
    """
    embedded = getattr(jqmc_rs, "__source_fnv64__", None)
    assert embedded is not None, (
        "jqmc_rs has no __source_fnv64__ attribute -- the imported extension predates the "
        "build-identity guard (build.rs). Rebuild the extension from current source."
    )
    recomputed = _source_fnv64()
    assert embedded == recomputed, (
        f"STALE jqmc_rs.abi3.so: embedded source digest {embedded} != current source digest "
        f"{recomputed}. The loaded extension was built from a DIFFERENT revision of jqmc/rs than "
        "the working tree, so every parity test in this suite is running against the wrong binary. "
        "Rebuild the extension (clean isolated wheel build) before trusting any result."
    )
    # Independently compare the compiled bit patterns with the runtime source.
    # The generator is not imported: no shared conversion or default can make
    # both sides agree on a stale/missing selected name.
    from jqmc import _setting

    generator = ast.parse((_CRATE_ROOT / "generate_setting_consts.py").read_text())
    names = next(
        ast.literal_eval(node.value)
        for node in generator.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "NAMES" for target in node.targets)
    )
    compiled = dict(jqmc_rs.__setting_consts__)
    assert len(jqmc_rs.__setting_consts__) == len(compiled) == len(names)
    assert set(compiled) == set(names)
    for name in names:
        assert struct.pack(">d", compiled[name]) == struct.pack(">d", getattr(_setting, name)), name
    import numpy as np

    for dtype in (np.float64, np.float32):
        info = np.finfo(dtype)
        assert _setting.as_linear_min_sq_norm(dtype) == float(info.tiny / info.eps)
        assert _setting.as_incremental_norm_cancellation_threshold(dtype) == float(info.eps) ** 0.5
        assert _setting.sm_large_ratio_threshold(dtype) == float(info.eps) ** -0.5


@pytest.mark.parametrize("changed", ["src/lib.rs", "build.rs", "generate_setting_consts.py", "../_setting.py"])
def test_digest_is_sensitive_to_source_change(changed):
    """Mutation guard: the digest MUST change if any hashed byte changes.

    Recompute over the real file set, then over the same set with one file's
    bytes perturbed, and assert the two digests differ -- so a real source edit
    could not leave the digest (and hence this guard) unmoved.
    """
    files = _source_files()
    baseline = _source_fnv64()

    # Recompute with a single byte appended to lib.rs's content (in memory only).
    digest = _FNV_OFFSET
    for rel, path in files:
        digest = _fnv_update(digest, rel.encode())
        digest = _fnv_update(digest, b"\0")
        data = path.read_bytes()
        if rel == changed:
            data = data + b"// mutation\n"
        digest = _fnv_update(digest, data)
        digest = _fnv_update(digest, b"\0")
    mutated = f"{digest:016x}"
    assert mutated != baseline, "digest did not change under a source mutation -- the guard is vacuous"
