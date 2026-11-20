"""Compatibility views of reviewed route-counter evidence in YAML SSoT.

Each reviewed row maps a counter index to its test reader or to an explicit
unread disposition. This adapter does not change engagement semantics.
Use load_route_counter_evidence(root) to audit a scratch/candidate repository.
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


def load_route_counter_evidence(root: Path | None = None) -> dict[str, Any]:
    """Return the legacy integer-keyed tuple mappings and unread vocabulary."""
    data = load_reviewed_inventory("route_counters", root)

    def reads(rows):
        return {row["index"]: (row["symbol"], row["pytest_node"]) for row in rows}

    def unread(rows):
        return {row["index"]: (row["symbol"], row["status"], row["reason"]) for row in rows}

    return {
        "COUNTER_READ_EVIDENCE": reads(data["counter_read_evidence"]),
        "COUNTER_UNREAD": unread(data["counter_unread"]),
        "PREDICATE_COUNTER_READ_EVIDENCE": reads(data["predicate_counter_read_evidence"]),
        "PREDICATE_COUNTER_UNREAD": unread(data["predicate_counter_unread"]),
        "COUNTER_UNREAD_STATUSES": frozenset(data["vocabularies"]["unread_statuses"]),
    }


_inventory = load_route_counter_evidence()
COUNTER_READ_EVIDENCE: dict[int, tuple[str, str]] = _inventory["COUNTER_READ_EVIDENCE"]
COUNTER_UNREAD: dict[int, tuple[str, str, str]] = _inventory["COUNTER_UNREAD"]
PREDICATE_COUNTER_READ_EVIDENCE: dict[int, tuple[str, str]] = _inventory["PREDICATE_COUNTER_READ_EVIDENCE"]
PREDICATE_COUNTER_UNREAD: dict[int, tuple[str, str, str]] = _inventory["PREDICATE_COUNTER_UNREAD"]
COUNTER_UNREAD_STATUSES: frozenset[str] = _inventory["COUNTER_UNREAD_STATUSES"]
