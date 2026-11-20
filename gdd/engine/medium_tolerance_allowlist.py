"""Compatibility views of reviewed medium-tolerance declarations in YAML SSoT.

The detector requires exact equality with the reviewed medium-capable sites.
Migration entries retain existing uses awaiting a separately reviewed policy
change; this adapter neither adds permission nor changes numerical tolerances.
Use load_medium_tolerance_allowlist(root) when auditing a candidate repository.
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


def load_medium_tolerance_allowlist(root: Path | None = None) -> dict[str, Any]:
    """Read the candidate inventory and preserve the legacy tuple-valued views."""
    data = load_reviewed_inventory("medium_tolerance", root)
    allowed = frozenset(data["vocabularies"]["allowed_classes"])
    migration = frozenset(data["vocabularies"]["migration_classes"])
    if allowed & migration:
        raise ValueError("medium-tolerance allowed and migration classes must be disjoint")
    return {
        "ALLOWED_CLASSES": allowed,
        "MIGRATION_CLASSES": migration,
        "MEDIUM_SITES": {row["site"]: (row["class"], row["reason"]) for row in data["allowed_sites"] + data["migration_sites"]},
    }


_inventory = load_medium_tolerance_allowlist()
ALLOWED_CLASSES: frozenset[str] = _inventory["ALLOWED_CLASSES"]
MIGRATION_CLASSES: frozenset[str] = _inventory["MIGRATION_CLASSES"]
MEDIUM_SITES: dict[str, tuple[str, str]] = _inventory["MEDIUM_SITES"]
