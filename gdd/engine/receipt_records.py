"""Typed JSON transport for validated derivative evidence, without dynamic imports.

Only the receipt classes below can be reconstructed. Payload tags cannot select
an arbitrary callable. All fields are retained and checked again by the join.
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

from dataclasses import fields
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from gdd.engine.evidence import (
    CaseReceipt,
    ComparisonReceipt,
    CompositeCaseReceipt,
    ElectronConfigurationReceipt,
    ExactComparisonReceipt,
    InputBinding,
    ObservationReceipt,
    RouteCaptureReceipt,
    RouteFixtureReceipt,
)
from gdd.engine.fixture_observations import (
    ComponentBoundaryReceipt,
    DTNCuspReceipt,
    HamiltonianFixtureReceipt,
    NuclearPotentialReceipt,
)


_RECEIPT_TYPES = (
    CaseReceipt,
    CompositeCaseReceipt,
    InputBinding,
    ElectronConfigurationReceipt,
    ComparisonReceipt,
    ExactComparisonReceipt,
    ObservationReceipt,
    RouteCaptureReceipt,
    RouteFixtureReceipt,
    HamiltonianFixtureReceipt,
    ComponentBoundaryReceipt,
    NuclearPotentialReceipt,
    DTNCuspReceipt,
)
_FIELD_TYPES = {kind: get_type_hints(kind) for kind in _RECEIPT_TYPES}


def _encode(value: Any) -> Any:
    if type(value) in _RECEIPT_TYPES:
        return {
            "type": type(value).__name__,
            "fields": {field.name: _encode(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError(f"unsupported derivative receipt value type: {type(value).__name__}")


def _decode(value: Any, expected: Any) -> Any:
    origin = get_origin(expected)
    arguments = get_args(expected)
    if origin in (UnionType, Union):
        for candidate in arguments:
            try:
                return _decode(value, candidate)
            except ValueError:
                pass
        raise ValueError("derivative receipt field has an incompatible union member")
    if origin is tuple:
        if not isinstance(value, list):
            raise ValueError("derivative receipt tuple must be encoded as a list")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_decode(item, arguments[0]) for item in value)
        if len(value) != len(arguments):
            raise ValueError("derivative receipt tuple has the wrong length")
        return tuple(_decode(item, kind) for item, kind in zip(value, arguments, strict=True))
    if expected in _RECEIPT_TYPES:
        if not isinstance(value, dict) or set(value) != {"type", "fields"} or value["type"] != expected.__name__:
            raise ValueError("derivative receipt type tag or structure is invalid")
        values = value["fields"]
        names = {field.name for field in fields(expected)}
        if not isinstance(values, dict) or set(values) != names:
            raise ValueError(f"{expected.__name__} receipt field set differs from its schema")
        return expected(**{name: _decode(values[name], _FIELD_TYPES[expected][name]) for name in names})
    if expected is float and type(value) in (int, float):
        # Preserve the original JSON number spelling: electron sequence tokens
        # include repr() of conditioning observations, so 1 and 1.0 differ.
        return value
    if expected in (str, bool, int, type(None)) and type(value) is expected:
        return value
    raise ValueError(f"invalid derivative receipt field for {expected}")


def encode_fixture_receipt(receipt: HamiltonianFixtureReceipt) -> dict[str, Any]:
    """Transport concrete fixture facts without accepting arbitrary dataclasses."""
    if type(receipt) is not HamiltonianFixtureReceipt:
        raise ValueError("fixture evidence must be a Hamiltonian fixture receipt")
    receipt.validate()
    return _encode(receipt)


def decode_fixture_receipt(payload: Any) -> HamiltonianFixtureReceipt:
    """Replay the structural invariants after JSON transport."""
    receipt = _decode(payload, HamiltonianFixtureReceipt)
    receipt.validate()
    return receipt


def encode_case_receipt(receipt: CaseReceipt | CompositeCaseReceipt) -> dict[str, Any]:
    """Preserve the whole validated case; fail on unsupported or lossy values."""
    if type(receipt) not in (CaseReceipt, CompositeCaseReceipt):
        raise ValueError("derivative evidence must be a case or composite receipt")
    receipt.validate()
    encoded = _encode(receipt)
    decode_case_receipt(encoded)
    return encoded


def decode_case_receipt(payload: Any) -> CaseReceipt | CompositeCaseReceipt:
    """Reconstruct only known receipt types and replay all existing invariants."""
    try:
        receipt = _decode(payload, CaseReceipt | CompositeCaseReceipt)
        receipt.validate()
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid derivative evidence payload: {exc}") from exc
    return receipt
