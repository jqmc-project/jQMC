"""Measured component ablations on the same consumed kinetic input.

This certifies a detectable contribution to this operation, not a component's
independent correctness or the activation of nonlocal ECP. The original kinetic
versus AD comparison is unchanged. Completion recomputes every difference from
the transported vectors using the authored comparison's central policy.
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

import json
import re

import numpy as np

from gdd.engine.canonical_contract import KINETIC_ACTIVATION, required_jastrow_components
from gdd.engine.fixture_observations import input_object_snapshot


_JASTROW_FIELDS = {"J1": "jastrow_one_body_data", "J2": "jastrow_two_body_data", "J3": "jastrow_three_body_data"}
_VECTOR_DTYPES = {"float32", "float64", "complex64", "complex128"}


def encode_vector(value):
    """Keep the exact real/complex dtype and values, including signed zeros."""
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.name not in _VECTOR_DTYPES or not value.dtype.isnative:
        raise ValueError("kinetic activation requires a native floating vector")
    if not np.all(np.isfinite(value)):
        raise ValueError("kinetic activation vector is nonfinite")
    return {
        "dtype": value.dtype.name,
        "real": value.real.tolist(),
        "imag": value.imag.tolist() if np.iscomplexobj(value) else None,
    }


def _decode_vector(payload, size):
    if not isinstance(payload, dict) or set(payload) != {"dtype", "real", "imag"}:
        raise ValueError("activation vector fields are invalid")
    if not isinstance(payload["dtype"], str) or payload["dtype"] not in _VECTOR_DTYPES:
        raise ValueError("activation vector dtype is invalid")
    dtype = np.dtype(payload["dtype"])
    parts = ("real", "imag") if dtype.kind == "c" else ("real",)
    if dtype.kind != "c" and payload["imag"] is not None:
        raise ValueError("real activation vector contains imaginary values")
    for part in parts:
        values = payload[part]
        if not isinstance(values, list) or len(values) != size or any(type(x) not in (float, int) for x in values):
            raise ValueError("activation vector values or shape are invalid")
    result = np.empty(size, dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        result.real = payload["real"]
        if dtype.kind == "c":
            result.imag = payload["imag"]
    if not np.all(np.isfinite(result)):
        raise ValueError("activation vector is nonfinite")
    return result


def _input_parts(wavefunction):
    """Fingerprint retained parameters separately from the channel-enable flags."""
    jastrow = wavefunction.jastrow_data
    parts = {"geminal": input_object_snapshot(wavefunction.geminal_data)}
    for name, field in _JASTROW_FIELDS.items():
        component = getattr(jastrow, field)
        parts[name] = input_object_snapshot(component) if component is not None else None
    dtn = jastrow.jastrow_dtn_data
    parts["DTN"] = (
        {
            "parameters": input_object_snapshot(dtn.replace(use_u=False, use_chi=False, use_f=False)),
            **{channel: bool(getattr(dtn, "use_" + channel)) for channel in ("u", "chi", "f")},
        }
        if dtn is not None
        else None
    )
    return parts


def capture_activation(wavefunction, positions, baseline, fixture_snapshot, evaluate):
    """Evaluate each actual input ablation; do not infer activation from labels."""
    jastrow = wavefunction.jastrow_data
    inputs = _input_parts(wavefunction)
    ablations = []
    variants = [(name, jastrow.replace(**{field: None})) for name, field in _JASTROW_FIELDS.items() if inputs[name]]
    if inputs["DTN"] is not None:
        for channel in ("u", "chi", "f"):
            if inputs["DTN"][channel]:
                without = jastrow.jastrow_dtn_data.replace(**{"use_" + channel: False})
                variants.append(("DTN-" + channel, jastrow.replace(jastrow_dtn_data=without)))
    for name, without in variants:
        ablated = wavefunction.replace(jastrow_data=without)
        coordinates = input_object_snapshot(positions)
        retained = _input_parts(ablated)
        output = np.concatenate(evaluate(ablated, *positions))
        ablations.append({"component": name, "coordinates": coordinates, "inputs": retained, "output": encode_vector(output)})
    return {
        "method": KINETIC_ACTIVATION,
        "fixture_snapshot": fixture_snapshot,
        "inputs": inputs,
        "baseline": encode_vector(baseline),
        "ablations": ablations,
    }


def _validate_inputs(inputs, wanted):
    def is_snapshot(value):
        return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None

    if not isinstance(inputs, dict) or set(inputs) != {"geminal", "J1", "J2", "J3", "DTN"}:
        raise ValueError("activation input-part fields are invalid")
    if not is_snapshot(inputs["geminal"]):
        raise ValueError("activation geminal snapshot is invalid")
    for name in _JASTROW_FIELDS:
        if (name in wanted and not is_snapshot(inputs[name])) or (name not in wanted and inputs[name] is not None):
            raise ValueError("activation Jastrow input parts differ from required components")
    dtn = inputs["DTN"]
    if not any(name.startswith("DTN-") for name in wanted):
        if dtn is not None:
            raise ValueError("activation contains an unrequested DTN input")
    elif (
        not isinstance(dtn, dict)
        or set(dtn) != {"parameters", "u", "chi", "f"}
        or not is_snapshot(dtn["parameters"])
        or any(type(dtn[channel]) is not bool or dtn[channel] != ("DTN-" + channel in wanted) for channel in ("u", "chi", "f"))
    ):
        raise ValueError("activation DTN input parts differ from required channels")


def activation_errors(record, signature, *, atol, rtol):
    """Recompute nonzero effects; labels, booleans and recorded margins are not proof."""
    try:
        if any(type(x) not in (int, float) or not np.isfinite(x) or x < 0 for x in (atol, rtol)):
            raise ValueError("activation comparison band is invalid")
        activation = record.get("activation")
        if not isinstance(activation, dict) or set(activation) != {
            "method",
            "fixture_snapshot",
            "inputs",
            "baseline",
            "ablations",
        }:
            raise ValueError("activation record is absent or malformed")
        if activation["method"] != KINETIC_ACTIVATION:
            raise ValueError("activation method is unknown")
        fixture = record["fixture"]["fields"]
        if activation["fixture_snapshot"] != fixture["content_snapshot"]:
            raise ValueError("activation belongs to a different consumed fixture")
        size = sum(fixture["electron_counts"])
        baseline = _decode_vector(activation["baseline"], size)
        if input_object_snapshot(baseline) != record["output"]:
            raise ValueError("activation baseline is not the consumed comparison output")
        wanted = required_jastrow_components(signature)
        inputs = activation["inputs"]
        _validate_inputs(inputs, wanted)
        ablations = activation["ablations"]
        if not isinstance(ablations, list) or len(ablations) != len(wanted):
            raise ValueError("activation ablations do not cover every required component exactly once")
        seen = set()
        for ablation in ablations:
            if not isinstance(ablation, dict) or set(ablation) != {"component", "coordinates", "inputs", "output"}:
                raise ValueError("activation ablation fields are invalid")
            name = ablation["component"]
            if not isinstance(name, str) or name not in wanted or name in seen:
                raise ValueError("activation has an extra, duplicate or unknown component")
            seen.add(name)
            if ablation["coordinates"] != record["coordinates"]:
                raise ValueError("activation ablation uses different electron positions")
            expected = dict(inputs)
            if name in _JASTROW_FIELDS:
                expected[name] = None
            else:
                expected["DTN"] = dict(inputs["DTN"], **{name.removeprefix("DTN-"): False})
            # Python equality aliases False/0 and True/1. Transported flags must
            # keep their exact types, not merely compare equal after coercion.
            if json.dumps(ablation["inputs"], sort_keys=True, allow_nan=False) != json.dumps(
                expected, sort_keys=True, allow_nan=False
            ):
                raise ValueError("activation ablation changed inputs other than its declared component")
            without = _decode_vector(ablation["output"], size)
            if without.dtype != baseline.dtype:
                raise ValueError("activation ablation output dtype differs from the consumed output")
            with np.errstate(over="ignore", invalid="ignore"):
                difference = np.abs(baseline - without)
                band = atol + rtol * np.abs(baseline)
            if not np.all(np.isfinite(difference)) or not np.all(np.isfinite(band)):
                raise ValueError("activation difference or band is nonfinite")
            if not np.any(difference > band):
                raise ValueError(f"activation {name} has no effect outside the required comparison band")
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        return [f"canonical activation: {exc}"]
    return []
