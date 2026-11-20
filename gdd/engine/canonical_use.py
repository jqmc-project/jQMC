"""Canonical requirements versus inputs consumed by a numerical comparison.

The first adapter is per-electron kinetic energy. Construction is not evidence:
only a normally returned call is banked, and its output must reach the comparator.
An optional measured ablation binds Jastrow activation to that same input/output;
kinetic evidence does not certify nonlocal ECP evaluation.
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

from dataclasses import fields, is_dataclass
import re

import numpy as np

from gdd.engine.canonical_contract import ECP_OPERATION, KINETIC_OPERATION, STREAMING_OPERATION, canonical_suite
from gdd.engine.comparison_requirements import resolve_requirement
from gdd.engine.fixture_observations import input_object_snapshot, observe_hamiltonian
from gdd.engine.kinetic_activation import activation_errors, capture_activation
from gdd.engine.receipt_records import decode_fixture_receipt, encode_fixture_receipt


def stored_partner_matches(up, down):
    """Compare stored time-reversed data, not flags or general orbital equivalence."""
    from jqmc.atomic_orbital import AOs_plane_wave_complex_data
    from jqmc.molecular_orbital import MOs_data

    if type(up) is not type(down):
        return False
    if type(up) is MOs_data:
        return (
            down.num_mo <= up.num_mo
            and np.array_equal(down.mo_coefficients, np.conj(np.asarray(up.mo_coefficients)[: down.num_mo]))
            and stored_partner_matches(up.aos_data, down.aos_data)
        )
    if is_dataclass(up):
        for field in fields(up):
            left, right = getattr(up, field.name), getattr(down, field.name)
            if (
                field.name == "k_point" or (field.name == "_g_indices" and type(up) is AOs_plane_wave_complex_data)
            ) and left is not None:
                matches = np.array_equal(np.asarray(right), -np.asarray(left))
            elif field.name in {"_image_phase_factors", "spline_coefficients"}:
                matches = np.array_equal(np.asarray(right), np.conj(np.asarray(left)))
            else:
                matches = stored_partner_matches(left, right)
            if not matches:
                return False
        return True
    if isinstance(up, (tuple, list)):
        return len(up) == len(down) and all(stored_partner_matches(a, b) for a, b in zip(up, down, strict=True))
    if up is None or isinstance(up, (str, bool, int, float)):
        return up == down
    return bool(np.array_equal(np.asarray(up), np.asarray(down)))


def observed_signature(fixture, pairing):
    """Normalize observed facts for exact, same-case matching against a pattern."""
    fixture.validate()
    if (
        not isinstance(pairing, dict)
        or set(pairing) != {"time_reversal", "stored_partner"}
        or any(type(x) is not bool for x in pairing.values())
    ):
        raise ValueError("canonical stored-pairing observation is invalid")
    n_up, n_dn = fixture.electron_counts
    if not n_dn:
        restriction = "not-applicable"
    elif pairing["time_reversal"] and pairing["stored_partner"]:
        restriction = "spin-restricted"
    elif not pairing["time_reversal"] and not pairing["stored_partner"]:
        restriction = "spin-unrestricted"
    else:
        raise ValueError("canonical spin flag and stored orbital data disagree")
    nonzero = [any(x != 0 for x in k) for k in fixture.k_points]
    if any(nonzero) and not all(nonzero):
        raise ValueError("canonical determinant leaves mix Gamma and non-Gamma")
    targets = set(fixture.dtn_cusp.physical_targets(fixture.nuclei)) if fixture.dtn_cusp else {"not-applicable"}
    if len(targets) != 1:
        raise ValueError("mixed DTN nuclear targets cannot match a homogeneous canonical pattern")
    return {
        "determinant": {
            "kind": fixture.determinant_kind,
            "representation": fixture.representation,
            "orbitals": {
                "up": fixture.orbital_signatures[0],
                "down": fixture.orbital_signatures[1] if n_dn else "not-applicable",
            },
        },
        "jastrow": {
            "j1_form": fixture.j1_form,
            "j2_form": fixture.j2_form,
            "j3_representation": fixture.j3_representation,
            "j3_orbital_signature": fixture.j3_orbital_signature,
            "dtn_channel_mask": "_".join(fixture.dtn_channels) or "off",
            "dtn_en_cusp_target": targets.pop(),
        },
        "electrons": {"shell_structure": "closed-shell" if n_up == n_dn else "open-shell", "spin_restriction": restriction},
        "periodicity": "periodic" if fixture.periodic else "open",
        "bloch_k_regime": ("non-gamma" if any(nonzero) else "gamma") if fixture.periodic else "not-applicable",
        "cell_metric_family": fixture.cell_metric,
    }


class CanonicalKineticCapture:
    """Observe the exact wavefunction/coordinates passed to the kinetic operation."""

    def __init__(self, hamiltonian, *, activation=False, orbital_activation=False):
        self.hamiltonian = hamiltonian
        self.activation = activation
        self.orbital_activation = orbital_activation
        self.records = []

    def evaluate(self, wavefunction, r_up, r_dn):
        from jqmc.determinant import compute_geminal_all_elements
        from jqmc.wavefunction import compute_kinetic_energy_all_elements

        if wavefunction is not self.hamiltonian.wavefunction_data:
            raise AssertionError("captured Hamiltonian is not the consumed wavefunction")
        fixture = observe_hamiltonian(self.hamiltonian)
        geminal = wavefunction.geminal_data
        for points, count in zip((r_up, r_dn), fixture.electron_counts, strict=True):
            if np.asarray(points).shape != (count, 3):
                raise AssertionError("canonical coordinate shape differs from consumed electron counts")
        coordinates = input_object_snapshot((r_up, r_dn))
        g = np.asarray(compute_geminal_all_elements(geminal, r_up, r_dn))
        g = g.astype(np.complex128 if np.iscomplexobj(g) else np.float64)
        cond_g = float(np.linalg.cond(g))
        if not np.isfinite(cond_g):
            raise AssertionError("canonical cond(G) must be finite")
        pairing = {
            "time_reversal": bool(geminal.time_reversal),
            "stored_partner": stored_partner_matches(geminal.orb_data_up_spin, geminal.orb_data_dn_spin),
        }
        result = compute_kinetic_energy_all_elements(wavefunction, r_up, r_dn)
        output = np.concatenate(result)
        # Reading the output synchronizes device errors before granting evidence.
        record = {
            "operation": KINETIC_OPERATION,
            "fixture": encode_fixture_receipt(fixture),
            "pairing": pairing,
            "coordinates": coordinates,
            "cond_G": cond_g,
            "output": input_object_snapshot(output),
        }
        if self.activation:
            record["activation"] = capture_activation(
                wavefunction, (r_up, r_dn), output, fixture.content_snapshot, compute_kinetic_energy_all_elements
            )
        if self.orbital_activation:
            from gdd.engine.orbital_activation import capture_orbital_activation

            record["orbital_activation"] = capture_orbital_activation(
                wavefunction, (r_up, r_dn), output, fixture.content_snapshot, compute_kinetic_energy_all_elements
            )
        self.records.append(record)
        return result


def canonical_use_errors(requirement, selector, records, checks, invocations, suite=None, *, precision=None):
    """Revalidate transported input facts and the actual comparator operand."""
    if requirement is None:
        return []
    if requirement.get("operation") == STREAMING_OPERATION:
        from gdd.engine.canonical_chain import canonical_chain_errors

        return canonical_chain_errors(requirement, selector, records, checks, invocations, suite)
    suite = canonical_suite() if suite is None else suite
    try:
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError("canonical use requires exactly one normally returned consumption record")
        record = records[0]
        fields = {
            "operation",
            "fixture",
            "pairing",
            "coordinates",
            "cond_G",
            "output",
        }
        if "activation" in requirement:
            fields.add("activation")
        if "orbital_activation" in requirement:
            fields.add("orbital_activation")
        if requirement["operation"] == ECP_OPERATION:
            fields.add("ecp")
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError("canonical consumption record fields are invalid")
        if record["operation"] != requirement["operation"]:
            raise ValueError("canonical consumption operation mismatch")
        if not isinstance(record["output"], str) or re.fullmatch(r"sha256:[0-9a-f]{64}", record["output"]) is None:
            raise ValueError("canonical output snapshot is invalid")
        fixture = decode_fixture_receipt(record["fixture"])
        patterns = {row["id"]: row["signature"] for row in suite["cases"]}
        pattern = selector[requirement["selector"]]
        if observed_signature(fixture, record["pairing"]) != patterns[pattern]:
            raise ValueError("consumed canonical signature differs from the required pattern")
        electron = requirement["electron_configuration"]
        rng = np.random.Generator(np.random.PCG64(electron["seed"]))
        positions = tuple(rng.normal(scale=electron["scale"], size=(n, 3)) for n in fixture.electron_counts)
        if record["coordinates"] != input_object_snapshot(positions):
            raise ValueError("consumed canonical coordinates differ from the required configuration")
        if type(record["cond_G"]) not in (int, float) or not np.isfinite(record["cond_G"]) or record["cond_G"] < 1:
            raise ValueError("consumed canonical cond(G) is invalid")
        site = checks[requirement["comparison"]]["site"]
        calls = [
            call
            for invocation in (invocations or ())
            for attempt in invocation.get("tolerance_calls", ())
            if attempt.get("site") == site
            for call in attempt.get("calls", ())
            if call.get("outcome") == "normal"
        ]
        if len(calls) != 1 or calls[0].get("operand_snapshots", {}).get("actual") != record["output"]:
            raise ValueError("canonical consumed output did not reach its required numerical comparison")
        snapshots = calls[0]["operand_snapshots"]
        if set(snapshots) != {"actual", "reference"} or any(
            not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None for value in snapshots.values()
        ):
            raise ValueError("canonical comparator operand snapshots are invalid")
        errors = []
        if "activation" in requirement or "orbital_activation" in requirement or requirement["operation"] == ECP_OPERATION:
            tolerance = resolve_requirement(checks[requirement["comparison"]]["precision"][precision], precision)
        if requirement["operation"] == ECP_OPERATION:
            from gdd.engine.canonical_ecp import ecp_activation_errors

            if patterns[pattern]["jastrow"]["dtn_en_cusp_target"] != "ecp-zero":
                raise ValueError("canonical ECP consumed an AE-only pattern")
            errors.extend(ecp_activation_errors(record, **tolerance))
        if "activation" in requirement:
            if record["activation"].get("method") != requirement["activation"]:
                raise ValueError("canonical activation method differs from the requirement")
            errors.extend(activation_errors(record, patterns[pattern], **tolerance))
        if "orbital_activation" in requirement:
            from gdd.engine.orbital_activation import orbital_activation_errors

            if record["orbital_activation"].get("method") != requirement["orbital_activation"]:
                raise ValueError("canonical orbital activation method differs from the requirement")
            errors.extend(orbital_activation_errors(record, patterns[pattern], **tolerance))
        return errors
    except (ValueError, AssertionError, TypeError, KeyError, AttributeError) as exc:
        return [f"canonical fixture use: {exc}"]
    return []
