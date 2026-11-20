"""Composite leaf contributions to the same consumed per-electron kinetic.

Zero only the leaf's coupling block, retaining the basis, precision and all other
parameters. Paired spin orbitals are changed together; independent spins are not.
Transport verification replays the coefficient mask, not a supplied success flag.
Production imports are confined to capture; completion needs only NumPy.
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

import re

import numpy as np

from gdd.engine.canonical_contract import ORBITAL_ACTIVATION, required_composite_components
from gdd.engine.fixture_observations import AO_SIGNATURES, COMPOSITE_SIGNATURES, input_object_snapshot
from gdd.engine.kinetic_activation import encode_vector, _decode_vector


def _parameters(wavefunction):
    """Separate mutable couplings from a fingerprint of every retained field."""
    from jqmc import atomic_orbital
    from jqmc.molecular_orbital import MOs_data

    layouts, matrices = {}, {}
    types = {getattr(atomic_orbital, f"AOs_{key}_data"): value for key, value in AO_SIGNATURES.items()}

    def orbital(value, role):
        mo = type(value) is MOs_data
        ao = value.aos_data if mo else value
        composite = type(ao) is atomic_orbital.AOs_composite_data
        children = ao.children if composite else (ao,)
        leaves = [{"signature": types[type(child)], "size": int(child.num_ao)} for child in children]
        if composite and tuple(leaf["signature"] for leaf in leaves) not in COMPOSITE_SIGNATURES:
            raise ValueError("unregistered composite leaf order")
        layouts[role] = {"representation": "MO" if mo else "AO", "leaves": leaves}
        if mo:
            matrices[role] = np.asarray(value.mo_coefficients)
        return value

    geminal = wavefunction.geminal_data
    for side in ("up", "down"):
        orbital(getattr(geminal, "orb_data_" + ("dn" if side == "down" else side) + "_spin"), "determinant." + side)
    if layouts["determinant.up"]["representation"] == "AO":
        matrices["determinant.lambda"] = np.asarray(geminal.lambda_matrix)
    j3 = wavefunction.jastrow_data.jastrow_three_body_data
    if j3 is not None:
        orbital(j3.orb_data, "J3")
        if layouts["J3"]["representation"] == "AO":
            matrices["J3.matrix"] = np.asarray(j3.j_matrix)
    fixed = _replace_parameters(wavefunction, {key: np.zeros_like(value) for key, value in matrices.items()})
    return {"fixed": input_object_snapshot(fixed), "layouts": layouts, "matrices": matrices}


def _replace_parameters(wavefunction, matrices):
    geminal = wavefunction.geminal_data
    replacements = {}
    for side in ("up", "down"):
        key = "determinant." + side
        field = "orb_data_" + ("dn" if side == "down" else side) + "_spin"
        if key in matrices:
            replacements[field] = getattr(geminal, field).replace(mo_coefficients=matrices[key])
    if "determinant.lambda" in matrices:
        replacements["lambda_matrix"] = matrices["determinant.lambda"]
    jastrow = wavefunction.jastrow_data
    j3 = jastrow.jastrow_three_body_data
    if "J3" in matrices:
        j3 = j3.replace(orb_data=j3.orb_data.replace(mo_coefficients=matrices["J3"]))
    if "J3.matrix" in matrices:
        j3 = j3.replace(j_matrix=matrices["J3.matrix"])
    return wavefunction.replace(
        geminal_data=geminal.replace(**replacements),
        jastrow_data=jastrow.replace(jastrow_three_body_data=j3),
    )


def _observed_components(parameters, paired, has_down):
    result = {}
    for role, layout in parameters["layouts"].items():
        if role == "determinant.down" and (paired or not has_down):
            continue
        if len(layout["leaves"]) < 2:
            continue
        role = "determinant.paired" if role == "determinant.up" and paired else role
        for index, leaf in enumerate(layout["leaves"]):
            result[f"{role}/{index}:{leaf['signature']}"] = {"role": role, "index": index, "leaf": leaf["signature"]}
    return result


def _mask(parameters, component):
    """Zero AO columns or AO-coupling rows/columns, never unpaired down columns."""
    matrices = {key: value.copy() for key, value in parameters["matrices"].items()}
    roles = ("determinant.up", "determinant.down") if component["role"] == "determinant.paired" else (component["role"],)
    for role in roles:
        layout = parameters["layouts"][role]
        index = component["index"]
        if layout["leaves"][index]["signature"] != component["leaf"]:
            raise ValueError("orbital activation leaf differs between paired consumers")
        lo = sum(leaf["size"] for leaf in layout["leaves"][:index])
        hi = lo + layout["leaves"][index]["size"]
        if layout["representation"] == "MO":
            matrices[role][:, lo:hi] = 0
        elif role == "J3":
            matrices["J3.matrix"][lo:hi, :] = 0
            matrices["J3.matrix"][:, lo:hi] = 0
        elif role == "determinant.up":
            matrices["determinant.lambda"][lo:hi, :] = 0
        else:
            matrices["determinant.lambda"][:, lo:hi] = 0
    return dict(parameters, matrices=matrices)


def _encode_parameters(parameters):
    return dict(
        parameters,
        matrices={
            key: {"shape": list(value.shape), "values": encode_vector(value.ravel())}
            for key, value in parameters["matrices"].items()
        },
    )


def capture_orbital_activation(wavefunction, positions, baseline, fixture_snapshot, evaluate):
    """Bank normally returned ablations and the actual changed/retained parameters."""
    parameters = _parameters(wavefunction)
    geminal = wavefunction.geminal_data
    components = _observed_components(parameters, bool(geminal.time_reversal), bool(geminal.num_electron_dn))
    ablations = []
    for name, component in components.items():
        masked = _mask(parameters, component)
        without = _replace_parameters(wavefunction, masked["matrices"])
        coordinates = input_object_snapshot(positions)
        observed = _parameters(without)
        output = np.concatenate(evaluate(without, *positions))
        ablations.append(
            {
                "component": name,
                "coordinates": coordinates,
                "inputs": _encode_parameters(observed),
                "output": encode_vector(output),
            }
        )
    return {
        "method": ORBITAL_ACTIVATION,
        "fixture_snapshot": fixture_snapshot,
        "inputs": _encode_parameters(parameters),
        "baseline": encode_vector(baseline),
        "ablations": ablations,
    }


def _decode_parameters(payload, signature, counts):
    if not isinstance(payload, dict) or set(payload) != {"fixed", "layouts", "matrices"}:
        raise ValueError("orbital input fields are invalid")
    if not isinstance(payload["fixed"], str) or re.fullmatch(r"sha256:[0-9a-f]{64}", payload["fixed"]) is None:
        raise ValueError("orbital retained-input snapshot is invalid")
    det, jas = signature["determinant"], signature["jastrow"]
    orbitals = {
        "determinant.up": (det["representation"], det["orbitals"]["up"]),
        "determinant.down": (det["representation"], det["orbitals"]["down"]),
    }
    if jas["j3_representation"] != "off":
        orbitals["J3"] = (jas["j3_representation"], jas["j3_orbital_signature"])
    layouts = payload["layouts"]
    if not isinstance(layouts, dict) or set(layouts) != set(orbitals):
        raise ValueError("orbital layouts do not cover the consumed roles")
    sizes = {}
    for role, (representation, name) in orbitals.items():
        layout = layouts[role]
        if (
            not isinstance(layout, dict)
            or set(layout) != {"representation", "leaves"}
            or layout["representation"] != representation
        ):
            raise ValueError("orbital representation differs from the request")
        leaves = layout["leaves"]
        if not isinstance(leaves, list) or not leaves:
            raise ValueError("orbital leaf inventory is empty")
        for leaf in leaves:
            if (
                not isinstance(leaf, dict)
                or set(leaf) != {"signature", "size"}
                or not isinstance(leaf["signature"], str)
                or leaf["signature"] not in AO_SIGNATURES.values()
                or type(leaf["size"]) is not int
                or leaf["size"] < 1
            ):
                raise ValueError("orbital leaf fields are invalid")
        names = tuple(leaf["signature"] for leaf in leaves)
        observed = names[0] if len(names) == 1 else COMPOSITE_SIGNATURES.get(names)
        if observed is None or (name != "not-applicable" and observed != name):
            raise ValueError("orbital leaf order differs from the required signature")
        sizes[role] = sum(leaf["size"] for leaf in leaves)
    if (
        signature["electrons"]["spin_restriction"] == "spin-restricted"
        and layouts["determinant.up"] != layouts["determinant.down"]
    ):
        raise ValueError("paired orbital layouts differ")
    wanted = {role for role, layout in layouts.items() if layout["representation"] == "MO"}
    if det["representation"] == "AO":
        wanted.add("determinant.lambda")
    if jas["j3_representation"] == "AO":
        wanted.add("J3.matrix")
    if not isinstance(payload["matrices"], dict) or set(payload["matrices"]) != wanted:
        raise ValueError("orbital coupling inventory differs from the request")
    matrices = {}
    for key, matrix in payload["matrices"].items():
        if not isinstance(matrix, dict) or set(matrix) != {"shape", "values"}:
            raise ValueError("orbital coupling fields are invalid")
        shape = matrix["shape"]
        if not isinstance(shape, list) or len(shape) != 2 or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError("orbital coupling shape is invalid")
        if key == "determinant.lambda":
            expected = [sizes["determinant.up"], sizes["determinant.down"] + counts[0] - counts[1]]
        elif key == "J3.matrix":
            expected = [sizes["J3"], sizes["J3"] + 1]
        else:
            expected = [shape[0], sizes[key]]
            if shape[0] < 1 and not (key == "determinant.down" and counts[1] == 0):
                raise ValueError("orbital coupling has no rows")
        if shape != expected:
            raise ValueError("orbital coupling shape differs from the leaf layout")
        matrices[key] = _decode_vector(matrix["values"], shape[0] * shape[1]).reshape(shape)
    return dict(payload, matrices=matrices)


def orbital_activation_errors(record, signature, *, atol, rtol):
    """Require every declared composite leaf, an exact mask and a detectable effect."""
    try:
        if any(type(x) not in (int, float) or not np.isfinite(x) or x < 0 for x in (atol, rtol)):
            raise ValueError("orbital activation comparison band is invalid")
        activation = record.get("orbital_activation")
        if not isinstance(activation, dict) or set(activation) != {
            "method",
            "fixture_snapshot",
            "inputs",
            "baseline",
            "ablations",
        }:
            raise ValueError("orbital activation record is absent or malformed")
        if activation["method"] != ORBITAL_ACTIVATION:
            raise ValueError("orbital activation method is unknown")
        fixture = record["fixture"]["fields"]
        if activation["fixture_snapshot"] != fixture["content_snapshot"]:
            raise ValueError("orbital activation belongs to a different consumed fixture")
        counts = fixture["electron_counts"]
        baseline = _decode_vector(activation["baseline"], sum(counts))
        if input_object_snapshot(baseline) != record["output"]:
            raise ValueError("orbital activation baseline is not the consumed comparison output")
        inputs = _decode_parameters(activation["inputs"], signature, counts)
        wanted = required_composite_components(signature)
        ablations = activation["ablations"]
        if not isinstance(ablations, list) or len(ablations) != len(wanted):
            raise ValueError("orbital ablations do not cover every required leaf exactly once")
        seen = set()
        for row in ablations:
            if not isinstance(row, dict) or set(row) != {"component", "coordinates", "inputs", "output"}:
                raise ValueError("orbital ablation fields are invalid")
            name = row["component"]
            if not isinstance(name, str) or name not in wanted or name in seen:
                raise ValueError("orbital ablation has an extra, duplicate or unknown leaf")
            seen.add(name)
            if row["coordinates"] != record["coordinates"]:
                raise ValueError("orbital ablation uses different electron positions")
            observed = _decode_parameters(row["inputs"], signature, counts)
            expected = _mask(inputs, wanted[name])
            if observed["fixed"] != expected["fixed"] or observed["layouts"] != expected["layouts"]:
                raise ValueError("orbital ablation changed retained inputs or the basis layout")
            for key, value in expected["matrices"].items():
                if input_object_snapshot(observed["matrices"][key]) != input_object_snapshot(value):
                    raise ValueError("orbital ablation did not apply exactly the required coefficient mask")
            without = _decode_vector(row["output"], sum(counts))
            if without.dtype != baseline.dtype:
                raise ValueError("orbital ablation changed output precision")
            with np.errstate(over="ignore", invalid="ignore"):
                difference = np.abs(baseline - without)
                band = atol + rtol * np.abs(baseline)
            if not np.all(np.isfinite(difference)) or not np.all(np.isfinite(band)) or not np.any(difference > band):
                raise ValueError(f"orbital activation {name} has no finite effect outside the required comparison band")
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, AssertionError) as exc:
        return [f"canonical orbital activation: {exc}"]
    return []
