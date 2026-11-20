"""Executable helpers for jQMC Guarantee-Driven Development contracts."""

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

from typing import Any

__all__ = [
    "CaseReceipt",
    "CompositeCaseReceipt",
    "ComparisonBinding",
    "ElectronConfigurationBinding",
    "ExactComparisonReceipt",
    "InputBinding",
    "ObservationBinding",
    "ObservationReceipt",
    "ProductionRouteCapture",
    "RouteCaptureReceipt",
    "canonical_snapshot",
    "emit_completion_record",
    "observe_allclose",
    "observe_electron_configurations",
    "observe_electron_configuration_sequence",
    "observe_exact",
    "observe_no_electron_configuration",
    "observe_non_numerical",
]


def __getattr__(name: str) -> Any:
    """Load NumPy-backed runtime evidence only when an evidence symbol is used."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import evidence

    value = getattr(evidence, name)
    globals()[name] = value
    return value
