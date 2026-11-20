"""One file per guarantee (issue 205): the loader fails closed instead of silently dropping a guarantee."""

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

from pathlib import Path

import pytest
import yaml

from gdd.engine import guarantee_store as store

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

MONOLITHIC = """\
schema_version: 1
document:
  id: toy
  lifecycle: cutover
guarantees:
- id: TOY-B-001
  kind: implementation
  required_evidence:
  - verdict_method: VM-X
    nodes:
    # a reviewed comment stays with its guarantee
    - pytest_node: tests/test_toy.py::test_b
- id: TOY-A-001
  kind: mathematical
  plain_language: |
    two
    lines
"""


def _module(root: Path, name: str = "toy", text: str = MONOLITHIC) -> Path:
    path = root / "gdd/ssot/guarantees" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def _dump(value) -> str:
    return yaml.dump(value, Dumper=_NoAliasDumper, sort_keys=False)


def _split(root: Path, name: str = "toy", text: str = MONOLITHIC) -> Path:
    """Write ``text`` in the one-file-per-guarantee layout (independent of the loader)."""
    path = _module(root, name, text)
    document = yaml.safe_load(text)
    for section in store.SPLIT_SECTIONS:
        rows = document.get(section)
        if not isinstance(rows, list):
            continue
        for row in rows:
            target = store.guarantee_path(path, section, row["id"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_dump(row), encoding="utf-8")
        document[section] = {"layout": store.SPLIT_LAYOUT}
    path.write_text(_dump(document), encoding="utf-8")
    return path


def _guarantee(guarantee_id: str) -> str:
    return f"id: {guarantee_id}\nkind: implementation\n"


def test_split_reassembles_rows_in_stable_id_order(tmp_path):
    """The loader preserves content while deriving a stable order from guarantee ids."""
    original = yaml.safe_load(MONOLITHIC)
    path = _split(tmp_path)
    assembled = store.load_module(path)
    expected = dict(original)
    expected["guarantees"] = sorted(original["guarantees"], key=lambda row: row["id"])
    assert assembled == expected
    assert list(assembled) == list(original)
    assert [row["id"] for row in assembled["guarantees"]] == ["TOY-A-001", "TOY-B-001"]
    marker = yaml.safe_load(path.read_text())["guarantees"]
    assert marker == {"layout": store.SPLIT_LAYOUT}
    assert store.layout_errors(tmp_path) == []


def test_duplicate_guarantee_id_across_files_fails(tmp_path):
    """The same id in two modules, or in two sections of one module, is rejected."""
    _split(tmp_path)
    other = _split(tmp_path, "other", MONOLITHIC.replace("TOY-B-001", "OTHER-B-001"))
    errors = store.layout_errors(tmp_path)
    assert any("duplicate guarantee id TOY-A-001" in error for error in errors)

    text = other.read_text() + "module_evidence:\n  layout: one-file-per-guarantee\n"
    other.write_text(text)
    (other.parent / "other/module_evidence").mkdir()
    (other.parent / "other/module_evidence/OTHER-B-001.yaml").write_text(_guarantee("OTHER-B-001"))
    with pytest.raises(store.GuaranteeLayoutError, match="duplicate guarantee id OTHER-B-001"):
        store.load_module(other)


@pytest.mark.parametrize(
    "defect, expected",
    [
        ("orphan-directory", "orphan guarantee directory"),
        ("undeclared-section", "undeclared section"),
        ("unknown-marker-key", "unknown split-marker keys"),
        ("file-name-mismatch", "file name does not match"),
        ("monolithic", "section guarantees is a monolithic inline list; every guarantee must be its own file"),
        ("stray-file", "unexpected files"),
    ],
)
def test_orphan_module_sections_and_unknown_marker_keys_fail(tmp_path, defect, expected):
    """A guarantee the loader would not read as its own file fails the layout check instead of vanishing."""
    path = _split(tmp_path)
    if defect == "orphan-directory":
        orphan = tmp_path / "gdd/ssot/guarantees/ghost"
        orphan.mkdir()
        (orphan / "GHOST-001.yaml").write_text(_guarantee("GHOST-001"))
    elif defect == "undeclared-section":
        section = path.parent / "toy/module_evidence"
        section.mkdir()
        (section / "TOY-E-001.yaml").write_text(_guarantee("TOY-E-001"))
    elif defect == "unknown-marker-key":
        document = yaml.safe_load(path.read_text())
        document["guarantees"]["snapshot"] = ["TOY-A-001"]
        path.write_text(_dump(document))
    elif defect == "file-name-mismatch":
        (path.parent / "toy/TOY-C-001.yaml").write_text(_guarantee("TOY-D-001"))
    elif defect == "monolithic":
        _module(tmp_path)
    else:
        (path.parent / "toy/notes.txt").write_text("x")
    assert any(expected in error for error in store.layout_errors(tmp_path)), store.layout_errors(tmp_path)


def test_fully_registered_test_files_reads_the_split_layout(tmp_path):
    """The exempt-file reader sees the registrations of split guarantee files."""
    from scripts.check_gdd_contract_changes import fully_registered_test_files

    _split(tmp_path)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_toy.py").write_text("def test_b():\n    pass\n")
    (tests / "test_other.py").write_text("def test_b():\n    pass\n\n\ndef test_unregistered():\n    pass\n")
    assert fully_registered_test_files(tmp_path) == frozenset({"tests/test_toy.py"})
