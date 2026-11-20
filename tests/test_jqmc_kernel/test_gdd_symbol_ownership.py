"""The wired symbol-ownership gate on the real tree (``gdd_sync --check`` runs it)."""

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
import shutil

import pytest

from scripts.check_gdd_symbol_ownership import _load_contract, validation_errors


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]
ROOT = Path(__file__).resolve().parents[2]


def test_symbol_ownership_rejects_an_unowned_production_module(tmp_path):
    """Every real production module is owned; one new public module without an owner turns the gate red."""
    contract = _load_contract()
    assert validation_errors(contract) == []
    # The real production sources and ledgers, plus one new module that no
    # ledger owns.
    shutil.copytree(ROOT / "jqmc", tmp_path / "jqmc", ignore=shutil.ignore_patterns("target", "__pycache__", "*.so"))
    (tmp_path / "gdd").symlink_to(ROOT / "gdd", target_is_directory=True)
    (tmp_path / "jqmc/probe_unowned.py").write_text("def unowned_public_entry():\n    return 1\n", encoding="utf-8")
    assert validation_errors(contract, tmp_path) == ["unowned production modules: ['jqmc.probe_unowned']"]
