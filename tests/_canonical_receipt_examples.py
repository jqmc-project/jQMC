"""Artificial transport records for aggregator self-tests, never science evidence.

The fixture structure is real; the output and condition number are deliberately
invented. These records belong only to synthetic-build/synthetic-run envelopes.
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

from functools import cache
import json

import numpy as np

from gdd.engine.canonical_use import KINETIC_OPERATION, stored_partner_matches
from gdd.engine.canonical_contract import ECP_OPERATION, STREAMING_OPERATION, canonical_comparisons
from gdd.engine.fixture_observations import input_object_snapshot, observe_hamiltonian
from gdd.engine.kinetic_activation import capture_activation
from gdd.engine.receipt_records import encode_fixture_receipt
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian


@cache
def _example_json(pattern_id, seed, scale, activation, orbital_activation):
    ham = build_canonical_hamiltonian(pattern_id)
    fixture = observe_hamiltonian(ham)
    geminal = ham.wavefunction_data.geminal_data
    rng = np.random.Generator(np.random.PCG64(seed))
    points = tuple(rng.normal(scale=scale, size=(n, 3)) for n in fixture.electron_counts)
    baseline = np.ones(sum(fixture.electron_counts))
    record = {
        "operation": KINETIC_OPERATION,
        "fixture": encode_fixture_receipt(fixture),
        "pairing": {
            "time_reversal": bool(geminal.time_reversal),
            "stored_partner": stored_partner_matches(geminal.orb_data_up_spin, geminal.orb_data_dn_spin),
        },
        "coordinates": input_object_snapshot(points),
        "cond_G": 1.0,
        "output": input_object_snapshot(baseline),
    }
    if activation:
        # Deliberately invented ablation outputs, for synthetic envelopes only.
        record["activation"] = capture_activation(
            ham.wavefunction_data,
            points,
            baseline,
            fixture.content_snapshot,
            lambda wf, up, down: (np.zeros(len(up)), np.zeros(len(down))),
        )
    if orbital_activation:
        from gdd.engine.orbital_activation import capture_orbital_activation

        record["orbital_activation"] = capture_orbital_activation(
            ham.wavefunction_data,
            points,
            baseline,
            fixture.content_snapshot,
            lambda wf, up, down: (np.zeros(len(up)), np.zeros(len(down))),
        )
    return json.dumps(record)


def synthetic_canonical_record(requirement, selector):
    """Return an independent mutable copy for a scratch certificate corpus."""
    electron = requirement["electron_configuration"]
    record = json.loads(
        _example_json(
            selector[requirement["selector"]],
            electron["seed"],
            electron["scale"],
            requirement.get("activation"),
            requirement.get("orbital_activation"),
        )
    )
    if requirement["operation"] == ECP_OPERATION:
        from gdd.engine.canonical_ecp import _inputs
        from gdd.engine.kinetic_activation import encode_vector

        ham = build_canonical_hamiltonian(selector[requirement["selector"]])
        coulomb, wf = ham.coulomb_potential_data, ham.wavefunction_data
        geminal = wf.geminal_data
        rng = np.random.Generator(np.random.PCG64(electron["seed"]))
        points = tuple(
            rng.normal(scale=electron["scale"], size=(n, 3)) for n in (geminal.num_electron_up, geminal.num_electron_dn)
        )
        coefficients = tuple(
            0.0 if ell < coulomb.max_ang_mom_plus_1[n] else value
            for n, ell, value in zip(coulomb.nucleus_index, coulomb.ang_moms, coulomb.coefficients, strict=True)
        )
        baseline = np.ones(sum(len(x) for x in points) * 18)
        record["operation"] = ECP_OPERATION
        record["output"] = input_object_snapshot(baseline)
        record["ecp"] = {
            "method": "nonlocal-coefficient-zeroing",
            "fixture_snapshot": observe_hamiltonian(ham).content_snapshot,
            "NN": 1,
            "Nv": 18,
            "nucleus_index": list(coulomb.nucleus_index),
            "angular_momenta": list(coulomb.ang_moms),
            "local_channels": list(coulomb.max_ang_mom_plus_1),
            "before": _inputs(coulomb, wf, points, np.eye(3)),
            "after": _inputs(coulomb.replace(coefficients=coefficients), wf, points, np.eye(3)),
            "baseline": encode_vector(baseline),
            "ablated": encode_vector(np.zeros_like(baseline)),
        }
        return record
    if requirement["operation"] != STREAMING_OPERATION:
        return record
    from gdd.engine.receipt_records import decode_fixture_receipt

    counts = decode_fixture_receipt(record["fixture"]).electron_counts
    rng = np.random.Generator(np.random.PCG64(electron["seed"]))
    points = [rng.normal(scale=electron["scale"], size=(n, 3)) for n in counts]

    def artificial_state(ordinal):
        return {key: input_object_snapshot((key, ordinal)) for key in ("inverse", "det_state", "kinetic_state")}

    previous = artificial_state(-1)
    initial = {"coordinates": record["coordinates"], "cond_G": 1.0, "state": previous}
    steps = []
    chain = requirement["chain"]
    for _ in range(chain["sweeps"]):
        for spin, count in zip((True, False), counts, strict=True):
            for index in range(count):
                ordinal = len(steps)
                points[0 if spin else 1][index] += rng.normal(scale=chain["displacement_scale"], size=3)
                after = artificial_state(ordinal)
                steps.append(
                    {
                        "ordinal": ordinal,
                        "moved_up": spin,
                        "index": index,
                        "coordinates": input_object_snapshot(tuple(points)),
                        "cond_G": 1.0,
                        "before": previous,
                        "after": after,
                        "consumed_inverse": after["inverse"],
                        "outputs": {
                            "inverse": after["inverse"],
                            "fast": input_object_snapshot(("fast", ordinal)),
                            "streaming": input_object_snapshot(("streaming", ordinal)),
                        },
                    }
                )
                previous = after
    return {
        "operation": STREAMING_OPERATION,
        "fixture": record["fixture"],
        "pairing": record["pairing"],
        "initial": initial,
        "steps": steps,
    }


def synthetic_comparator_outputs(requirement, checks, check, record):
    """Invented per-hop operands for scratch rows, never live receipt evidence."""
    for quantity, name in canonical_comparisons(requirement).items():
        if check is checks[name]:
            return (
                [
                    {
                        "snapshot": step["outputs"][quantity],
                        "chain_step": {
                            "ordinal": step["ordinal"],
                            "coordinates": step["coordinates"],
                            "initial_inverse": record["initial"]["state"]["inverse"],
                        },
                    }
                    for step in record["steps"]
                ]
                if "chain" in requirement
                else [{"snapshot": record["output"]}]
            )
    return [None]
