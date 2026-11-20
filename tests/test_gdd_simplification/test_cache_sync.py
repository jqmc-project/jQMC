"""Derived bindings are never stale or hand-edited evidence, and contract digests follow the YAML."""

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

from types import SimpleNamespace
from pathlib import Path

import pytest

from tests._gdd_ssot_layout import write_ssot_text

from gdd.engine import derived_cache

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.fixture
def generation(tmp_path, monkeypatch):
    """A tiny generator records calls and emits one disposable binding."""
    source = tmp_path / "gdd/ssot/guarantees/example.yaml"
    source.parent.mkdir(parents=True)
    write_ssot_text(source, "guarantees: []\n")
    calls = []

    def generate(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == tmp_path
        cache = tmp_path / "gdd/derived/example.bindings.yaml"
        write_ssot_text(cache, "guarantees: {}\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(derived_cache.subprocess, "run", generate)
    return source, calls


@pytest.mark.parametrize(
    "relative",
    ["tests/test_new.py", "gdd/ssot/guarantees/example.yaml", "tests/fixture.h5", "gdd/derived/example.bindings.yaml"],
)
def test_source_or_fixture_change_rebuilds_cache(tmp_path, generation, relative):
    """Uncommitted edits, fixture byte changes and a hand-edited binding invalidate the cache."""
    derived_cache.prepare_derived(tmp_path)
    changed = tmp_path / relative
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_bytes(b"changed input")
    assert derived_cache.prepare_derived(tmp_path)
    assert len(generation[1]) == 2


def test_failed_generator_does_not_create_success_stamp(tmp_path, generation, monkeypatch):
    """Malformed source or generator failure cannot fall back to stale bindings."""
    monkeypatch.setattr(derived_cache.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=2))
    with pytest.raises(RuntimeError, match="exit code 2"):
        derived_cache.prepare_derived(tmp_path)
    assert not (tmp_path / "gdd/derived/.cache-state.json").exists()


@pytest.mark.gdd_no_committed_artifact(
    reason="The YAML contract and display Markdown are synthetic files created only in tmp_path"
)
def test_contract_snapshots_ignore_markdown_but_change_with_yaml(tmp_path, monkeypatch):
    """Both real contract digests ignore absent/edited display pages but remain sensitive to YAML bytes."""
    from gdd.engine import module_receipts
    from scripts.run_gdd_preflight import contract_snapshot_sha256

    source = tmp_path / "gdd/ssot/guarantees/example.yaml"
    write_ssot_text(source, "guarantees: []\n")
    monkeypatch.setattr(module_receipts, "ROOT", tmp_path)

    def snapshots():
        module_receipts.contract_snapshot.cache_clear()
        try:
            return contract_snapshot_sha256(tmp_path), module_receipts.contract_snapshot()
        finally:
            module_receipts.contract_snapshot.cache_clear()

    baseline = snapshots()
    assert baseline[0] == baseline[1]
    pages = [
        tmp_path / relative
        for relative in (
            "gdd/ledger/README.md",
            "gdd/derived/human-ledger/guarantees/example.md",
            "gdd/ssot/README.md",
            "gdd/ssot/guarantees/README.md",
        )
    ]
    read_bytes = Path.read_bytes

    def reject_display_read(path):
        assert not (path.suffix.lower() == ".md" and path.is_relative_to(tmp_path)), (
            "GDD attempted to read display-only Markdown",
            str(path),
        )
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_display_read)
    for content in ("display page\n", "edited display page\n"):
        for page in pages:
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text(content)
        assert snapshots() == baseline
    for page in pages:
        page.unlink()
    assert snapshots() == baseline
    source.write_text("guarantees: []\nsource_change: true\n")
    changed = snapshots()
    assert changed[0] == changed[1]
    assert all(after != before for before, after in zip(baseline, changed, strict=True))
