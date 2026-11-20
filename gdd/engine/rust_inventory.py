"""Compatibility views of the reviewed Rust inventories in YAML SSoT.

The YAML owns test kinds, export statuses and their complete reviewed mappings.
Cargo units remain inventory-only; this adapter does not make them GDD evidence.
Use load_rust_inventory(root) when auditing a scratch/candidate repository.
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

from pathlib import Path
from typing import Any

from gdd.engine.reviewed_inventory import load_reviewed_inventory


def load_rust_inventory(root: Path | None = None) -> dict[str, Any]:
    """Return the legacy mappings and vocabulary from the specified repository."""
    data = load_reviewed_inventory("rust_inventory", root)
    return {
        "RUST_TESTS": {row["id"]: row["kind"] for row in data["rust_tests"]},
        "PYTHON_EXPORTS": {row["name"]: row["status"] for row in data["python_exports"]},
        "TEST_KINDS": frozenset(data["vocabularies"]["test_kinds"]),
        "EXPORT_STATUSES": frozenset(data["vocabularies"]["export_statuses"]),
    }


_inventory = load_rust_inventory()
RUST_TESTS: dict[str, str] = _inventory["RUST_TESTS"]
PYTHON_EXPORTS: dict[str, str] = _inventory["PYTHON_EXPORTS"]
TEST_KINDS: frozenset[str] = _inventory["TEST_KINDS"]
EXPORT_STATUSES: frozenset[str] = _inventory["EXPORT_STATUSES"]
