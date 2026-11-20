"""Test-only writer for the one-file-per-guarantee SSoT layout (issue 205).

The loader (``gdd.engine.guarantee_store``) reads only the split layout, so
synthetic ledgers in tests are written through this helper instead of as one
monolithic YAML file.
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

import shutil
from pathlib import Path
from typing import Any

import yaml

from gdd.engine.guarantee_store import SPLIT_LAYOUT, SPLIT_SECTIONS, guarantee_path, is_module_ledger


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def dump(value: Any) -> str:
    return yaml.dump(value, Dumper=_NoAliasDumper, sort_keys=False, default_flow_style=False, width=120)


def write_module(path: Path, document: dict[str, Any] | str) -> Path:
    """Write a module ledger document (mapping or YAML text) in the split layout."""
    path = Path(path)
    if isinstance(document, str):
        document = yaml.safe_load(document)
    document = dict(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = path.parent / path.stem
    if directory.is_symlink():
        directory.unlink()
    elif directory.exists():
        shutil.rmtree(directory)
    for section in SPLIT_SECTIONS:
        rows = document.get(section)
        if not isinstance(rows, list) or not rows:
            continue
        for row in rows:
            target = guarantee_path(path, section, row["id"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(dump(row), encoding="utf-8")
        document[section] = {"layout": SPLIT_LAYOUT}
    path.write_text(dump(document), encoding="utf-8")
    return path


def write_ssot_text(path: Path, text: str, *args: Any, **kwargs: Any) -> int:
    """``path.write_text`` for tests; a guarantee module ledger is written in the split layout.

    Text that does not parse to a module mapping with a guarantee list
    (malformed fixtures) is written unchanged, so loader rejections stay
    observable.
    """
    path = Path(path)
    if is_module_ledger(path):
        try:
            document = yaml.safe_load(text)
        except yaml.YAMLError:
            document = None
        if isinstance(document, dict) and any(isinstance(document.get(key), list) for key in SPLIT_SECTIONS):
            write_module(path, document)
            return len(text)
    return path.write_text(text, *args, **kwargs)
