"""Bind a canonical ECP evaluation and nonlocal-only ablation to its comparator.

The adapter is intentionally specific: nearest neighbour, Nv=18, identity
rotation. It does not certify other meshes, holomorphic projection or high-l
coverage. The transported values, not a success flag, establish activation.
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

from gdd.engine.canonical_contract import ECP_OPERATION
from gdd.engine.canonical_use import stored_partner_matches
from gdd.engine.fixture_observations import input_object_snapshot, observe_hamiltonian
from gdd.engine.kinetic_activation import _decode_vector, encode_vector
from gdd.engine.receipt_records import decode_fixture_receipt, encode_fixture_receipt


def _inputs(coulomb, wavefunction, positions, rotation):
    """Separate coefficient values from every retained input, including local terms."""
    return {
        "wavefunction": input_object_snapshot(wavefunction),
        "coordinates": input_object_snapshot(positions),
        "rotation": input_object_snapshot(rotation),
        "fixed_coulomb": input_object_snapshot(coulomb.replace(coefficients=(0.0,) * coulomb.num_ecps)),
        "coefficients": encode_vector(np.asarray(coulomb.coefficients, dtype=np.float64)),
    }


class CanonicalECPCapture:
    """Bank evidence only after both normally returned calls on their exact inputs."""

    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian
        self.records = []

    def evaluate(self, coulomb, wavefunction, r_up, r_dn):
        from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors
        from jqmc.determinant import compute_geminal_all_elements

        if coulomb is not self.hamiltonian.coulomb_potential_data or wavefunction is not self.hamiltonian.wavefunction_data:
            raise AssertionError("captured Hamiltonian is not the consumed ECP input")
        fixture = observe_hamiltonian(self.hamiltonian)
        if not fixture.ecp:
            raise AssertionError("canonical ECP requires a realized pseudopotential")
        positions = (r_up, r_dn)
        for points, count in zip(positions, fixture.electron_counts, strict=True):
            if np.asarray(points).shape != (count, 3):
                raise AssertionError("canonical ECP coordinate shape differs from electron counts")
        geminal = wavefunction.geminal_data
        g = np.asarray(compute_geminal_all_elements(geminal, r_up, r_dn))
        cond_g = float(np.linalg.cond(g.astype(np.complex128 if np.iscomplexobj(g) else np.float64)))
        if not np.isfinite(cond_g):
            raise AssertionError("canonical ECP cond(G) must be finite")
        rotation = np.eye(3)
        before = _inputs(coulomb, wavefunction, positions, rotation)
        result = compute_ecp_non_local_parts_nearest_neighbors(coulomb, wavefunction, r_up, r_dn, rotation, NN=1, Nv=18)
        baseline = np.asarray(result[-2])
        indices = [
            i
            for i, (nucleus, ell) in enumerate(zip(coulomb.nucleus_index, coulomb.ang_moms, strict=True))
            if ell < coulomb.max_ang_mom_plus_1[nucleus]
        ]
        coefficients = list(coulomb.coefficients)
        for index in indices:
            coefficients[index] = 0.0
        without = coulomb.replace(coefficients=tuple(coefficients))
        after = _inputs(without, wavefunction, positions, rotation)
        ablated = compute_ecp_non_local_parts_nearest_neighbors(without, wavefunction, r_up, r_dn, rotation, NN=1, Nv=18)
        record = {
            "operation": ECP_OPERATION,
            "fixture": encode_fixture_receipt(fixture),
            "pairing": {
                "time_reversal": bool(geminal.time_reversal),
                "stored_partner": stored_partner_matches(geminal.orb_data_up_spin, geminal.orb_data_dn_spin),
            },
            "coordinates": before["coordinates"],
            "cond_G": cond_g,
            "output": input_object_snapshot(baseline),
            "ecp": {
                "method": "nonlocal-coefficient-zeroing",
                "fixture_snapshot": fixture.content_snapshot,
                "NN": 1,
                "Nv": 18,
                "nucleus_index": list(coulomb.nucleus_index),
                "angular_momenta": list(coulomb.ang_moms),
                "local_channels": list(coulomb.max_ang_mom_plus_1),
                "before": before,
                "after": after,
                "baseline": encode_vector(baseline),
                "ablated": encode_vector(np.asarray(ablated[-2])),
            },
        }
        self.records.append(record)
        return result


def ecp_activation_errors(record, *, atol, rtol):
    """Recompute nonlocal masks and numerical activation after receipt transport."""
    try:
        fixture = decode_fixture_receipt(record["fixture"])
        data = record["ecp"]
        expected = {
            "method",
            "fixture_snapshot",
            "NN",
            "Nv",
            "nucleus_index",
            "angular_momenta",
            "local_channels",
            "before",
            "after",
            "baseline",
            "ablated",
        }
        if not isinstance(data, dict) or set(data) != expected or data["method"] != "nonlocal-coefficient-zeroing":
            raise ValueError("nonlocal activation fields or method are invalid")
        if not fixture.ecp or data["fixture_snapshot"] != fixture.content_snapshot:
            raise ValueError("nonlocal activation is not bound to the consumed ECP fixture")
        if type(data["NN"]) is not int or data["NN"] != 1 or type(data["Nv"]) is not int or data["Nv"] != 18:
            raise ValueError("nonlocal activation changed the required quadrature")
        nuclei, ells, local = (data[name] for name in ("nucleus_index", "angular_momenta", "local_channels"))
        if any(
            not isinstance(values, list) or any(type(x) is not int or x < 0 for x in values) for values in (nuclei, ells, local)
        ):
            raise ValueError("nonlocal term indices are invalid")
        if len(local) != len(fixture.nuclei) or len(nuclei) != len(ells) or not nuclei:
            raise ValueError("nonlocal term inventory has the wrong shape")
        if any(n >= len(local) or ell > local[n] for n, ell in zip(nuclei, ells, strict=True)):
            raise ValueError("nonlocal term inventory is outside the nuclear channel domain")
        if any(nuclei.count(i) != atom.ecp_terms or local[i] != atom.ecp_channels for i, atom in enumerate(fixture.nuclei)):
            raise ValueError("nonlocal terms disagree with the consumed nuclear inventory")
        mask = np.asarray([ell < local[n] for n, ell in zip(nuclei, ells, strict=True)])
        before, after = data["before"], data["after"]
        fields = {"wavefunction", "coordinates", "rotation", "fixed_coulomb", "coefficients"}
        for inputs in (before, after):
            if not isinstance(inputs, dict) or set(inputs) != fields:
                raise ValueError("nonlocal input fields are invalid")
            for name in fields - {"coefficients"}:
                if not isinstance(inputs[name], str) or re.fullmatch(r"sha256:[0-9a-f]{64}", inputs[name]) is None:
                    raise ValueError("nonlocal retained input snapshot is invalid")
                if inputs[name] != before[name]:
                    raise ValueError("nonlocal ablation changed a retained input")
            if inputs["coordinates"] != record["coordinates"] or inputs["rotation"] != input_object_snapshot(np.eye(3)):
                raise ValueError("nonlocal ablation changed the consumed coordinates or fixed rotation")
        original = _decode_vector(before["coefficients"], len(nuclei))
        removed = _decode_vector(after["coefficients"], len(nuclei))
        if original.dtype != np.dtype("float64") or removed.dtype != original.dtype:
            raise ValueError("nonlocal coefficient transport dtype is invalid")
        if not np.any(original[mask] != 0):
            raise ValueError("consumed ECP has no nonzero nonlocal coefficient")
        expected_coefficients = original.copy()
        expected_coefficients[mask] = 0
        if not np.array_equal(removed, expected_coefficients):
            raise ValueError("nonlocal ablation did not zero only the nonlocal coefficients")
        size = sum(fixture.electron_counts) * data["NN"] * data["Nv"]
        baseline, ablated = (_decode_vector(data[name], size) for name in ("baseline", "ablated"))
        if ablated.dtype != baseline.dtype or baseline.dtype.kind != "f":
            raise ValueError("nonlocal forward output dtype is invalid")
        if input_object_snapshot(baseline) != record["output"]:
            raise ValueError("nonlocal baseline was not the consumed comparison output")
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.abs(baseline - ablated)
            band = atol + rtol * np.abs(baseline)
        if not np.all(np.isfinite(difference)) or not np.all(np.isfinite(band)) or not np.any(difference > band):
            raise ValueError("nonlocal ECP has no measured effect outside the comparison band")
        if np.any(ablated != 0):
            raise ValueError("zero nonlocal coefficients did not produce zero nonlocal output")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as exc:
        return [f"canonical ECP activation: {exc}"]
    return []
