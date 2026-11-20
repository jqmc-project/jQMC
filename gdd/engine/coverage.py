"""Exact subset coverage for selected GDD interactions.

A requirement is satisfied only when every fact in its tuple is present in one
realized case. Facts observed in separate cases are never unioned. Precision,
backend, execution world, conditioning region, and activation facts are normal
keys, so one lane cannot close another lane's requirement.
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
import json
from collections.abc import Iterable
from typing import Any


def canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def case_covers(requirement: dict[str, Any], realized: dict[str, Any]) -> bool:
    """Whether one and the same realized case contains the required tuple."""
    return all(key in realized and realized[key] == value for key, value in requirement.items())


def missing_requirements(
    requirements: Iterable[dict[str, Any]],
    realized_cases: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    realized = list(realized_cases)
    return [requirement for requirement in requirements if not any(case_covers(requirement, case) for case in realized)]


def distinct_sample_count(realized_cases: Iterable[dict[str, Any]], sample_keys: Iterable[str]) -> int:
    """Count semantic samples, independent of test names and seed labels."""
    keys = tuple(sample_keys)
    return len({canonical_fingerprint({key: case.get(key) for key in keys}) for case in realized_cases})
