"""Observed SM state continuity and per-hop comparisons, not counter labels.

Production imports occur only when capturing a real calculation. Completion
replays the authored move plan with NumPy and validates transported facts.
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

from functools import cache
import re

import numpy as np

from gdd.engine.canonical_contract import STREAMING_OPERATION, canonical_comparisons, canonical_suite
from gdd.engine.canonical_use import observed_signature, stored_partner_matches
from gdd.engine.fixture_observations import input_object_snapshot, observe_hamiltonian
from gdd.engine.receipt_records import decode_fixture_receipt, encode_fixture_receipt


@cache
def _compiled_kinetic(name):
    import jax
    from jqmc import wavefunction

    return jax.jit(getattr(wavefunction, name))


def _init_kinetic(*args):
    return _compiled_kinetic("_init_kinetic_energy_all_elements_streaming_state")(*args)


def _advance_kinetic(*args):
    return _compiled_kinetic("_advance_kinetic_energy_all_elements_streaming_state")(*args)


def _fast_kinetic(*args):
    return _compiled_kinetic("compute_kinetic_energy_all_elements_fast_update")(*args)


def _state_snapshots(inverse, det_state, kinetic_state):
    return {
        name: input_object_snapshot(value)
        for name, value in (("inverse", inverse), ("det_state", det_state), ("kinetic_state", kinetic_state))
    }


def _condition(geminal, r_up, r_dn):
    from jqmc.determinant import compute_geminal_all_elements

    value = float(np.linalg.cond(np.asarray(compute_geminal_all_elements(geminal, r_up, r_dn))))
    if not np.isfinite(value):
        raise AssertionError("canonical chain cond(G) must be finite")
    return value


class CanonicalStreamingCapture:
    """Own the carried inverse/cache and bank only actual consumer returns."""

    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian
        self.records = []
        self._pending = None
        self._initialized = False

    def _check_input(self, wavefunction, r_up, r_dn):
        if wavefunction is not self.hamiltonian.wavefunction_data:
            raise AssertionError("canonical chain Hamiltonian is not the consumed wavefunction")
        if self._initialized and input_object_snapshot(wavefunction) != self._wavefunction_snapshot:
            raise AssertionError("canonical chain wavefunction changed during consumption")
        gem = wavefunction.geminal_data
        if (np.shape(r_up), np.shape(r_dn)) != ((gem.num_electron_up, 3), (gem.num_electron_dn, 3)):
            raise AssertionError("canonical chain coordinates differ from electron counts")

    def initialize(self, wavefunction, r_up, r_dn):
        import jax.numpy as jnp
        from jqmc import determinant

        self._check_input(wavefunction, r_up, r_dn)
        if self._initialized:
            raise AssertionError("canonical chain cannot refresh its initial inverse")
        gem = wavefunction.geminal_data
        inverse = jnp.linalg.inv(determinant.compute_geminal_all_elements(gem, r_up, r_dn))
        det_state = determinant._init_det_ratio_streaming_state(gem, r_up, r_dn)
        kinetic_state = _init_kinetic(wavefunction, r_up, r_dn, inverse)
        initial = {
            "coordinates": input_object_snapshot((r_up, r_dn)),
            "cond_G": _condition(gem, r_up, r_dn),
            "state": _state_snapshots(inverse, det_state, kinetic_state),
        }
        self._record = {
            "operation": STREAMING_OPERATION,
            "fixture": encode_fixture_receipt(observe_hamiltonian(self.hamiltonian)),
            "pairing": {
                "time_reversal": bool(gem.time_reversal),
                "stored_partner": stored_partner_matches(gem.orb_data_up_spin, gem.orb_data_dn_spin),
            },
            "initial": initial,
            "steps": [],
        }
        self._inverse, self._det_state, self._kinetic_state = inverse, det_state, kinetic_state
        self._wavefunction_snapshot = input_object_snapshot(wavefunction)
        self._initialized = True

    def advance(self, wavefunction, moved_up, index, r_up, r_dn):
        import jax
        import jax.numpy as jnp
        from jqmc import determinant

        self._check_input(wavefunction, r_up, r_dn)
        if not self._initialized or self._pending is not None:
            raise AssertionError("canonical chain advance requires a completed previous consumption")
        self.records = []
        gem = wavefunction.geminal_data
        count = gem.num_electron_up if moved_up else gem.num_electron_dn
        if type(moved_up) is not bool or type(index) is not int or not 0 <= index < count:
            raise AssertionError("canonical chain move is invalid")
        before = _state_snapshots(self._inverse, self._det_state, self._kinetic_state)
        if moved_up:
            u = jax.nn.one_hot(index, gem.num_electron_up, dtype=self._inverse.dtype)[:, None]
            v = determinant._compute_v_up_move_from_det_ratio_state(gem, self._det_state, index, r_up)[:, None]
        else:
            u = determinant._compute_u_dn_move_from_det_ratio_state(gem, self._det_state, index, r_dn)[:, None]
            v = jax.nn.one_hot(index, gem.num_electron_up, dtype=self._inverse.dtype)[:, None]
        rho, inverse = determinant._sherman_morrison_ratio_and_inverse(self._inverse, u, v)
        if not np.isfinite(rho) or abs(rho) == 0:
            raise AssertionError("canonical chain update ratio must be finite and nonzero")
        det_state = determinant._advance_det_ratio_streaming_state(
            gem, self._det_state, jnp.bool_(moved_up), jnp.int64(index), r_up, r_dn
        )
        kinetic_state = _advance_kinetic(
            wavefunction, self._kinetic_state, jnp.bool_(moved_up), jnp.int64(index), r_up, r_dn, inverse
        )
        self._pending = {
            "ordinal": len(self._record["steps"]),
            "moved_up": moved_up,
            "index": index,
            "coordinates": input_object_snapshot((r_up, r_dn)),
            "cond_G": _condition(gem, r_up, r_dn),
            "before": before,
            "after": _state_snapshots(inverse, det_state, kinetic_state),
        }
        self._inverse, self._det_state, self._kinetic_state = inverse, det_state, kinetic_state
        return rho, inverse

    def evaluate(self, wavefunction, r_up, r_dn, inverse):
        from jqmc.wavefunction import _kinetic_energy_from_streaming_state

        self._check_input(wavefunction, r_up, r_dn)
        if self._pending is None:
            raise AssertionError("canonical chain has no pending move to consume")
        if inverse is not self._inverse:
            raise AssertionError("canonical consumer must use the returned carried inverse")
        if input_object_snapshot((r_up, r_dn)) != self._pending["coordinates"]:
            raise AssertionError("canonical consumer coordinates differ from the advanced state")
        fast = np.concatenate(_fast_kinetic(wavefunction, r_up, r_dn, inverse))
        streaming = np.concatenate(_kinetic_energy_from_streaming_state(self._kinetic_state))
        step = dict(
            self._pending,
            consumed_inverse=input_object_snapshot(inverse),
            outputs={
                "inverse": input_object_snapshot(inverse),
                "fast": input_object_snapshot(fast),
                "streaming": input_object_snapshot(streaming),
            },
        )
        self._record["steps"].append(step)
        self._pending = None
        self.records = [self._record]
        return fast, streaming

    def comparison_context(self, quantity):
        """Read the currently consumed hop at comparator entry, not a test label."""
        if quantity == "inverse":
            step = self._pending
        else:
            step = self._record["steps"][-1] if self.records and self._pending is None else None
        if step is None:
            raise ValueError("canonical comparison has no current consumed hop")
        return {
            "ordinal": step["ordinal"],
            "coordinates": step["coordinates"],
            "initial_inverse": self._record["initial"]["state"]["inverse"],
        }


def _snapshot(value):
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _state_valid(value):
    return (
        isinstance(value, dict)
        and set(value) == {"inverse", "det_state", "kinetic_state"}
        and all(_snapshot(item) for item in value.values())
    )


def _finite_condition(value):
    return type(value) in (int, float) and np.isfinite(value) and value >= 1


def canonical_chain_errors(requirement, selector, records, checks, invocations, suite=None):
    """Require every authored move and all three consumed operands at each hop."""
    try:
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError("chain requires exactly one consumption record")
        record = records[0]
        if not isinstance(record, dict) or set(record) != {"operation", "fixture", "pairing", "initial", "steps"}:
            raise ValueError("chain record fields are invalid")
        if record["operation"] != STREAMING_OPERATION or record["operation"] != requirement["operation"]:
            raise ValueError("chain operation mismatch")
        fixture = decode_fixture_receipt(record["fixture"])
        suite = canonical_suite() if suite is None else suite
        patterns = {row["id"]: row["signature"] for row in suite["cases"]}
        if observed_signature(fixture, record["pairing"]) != patterns[selector[requirement["selector"]]]:
            raise ValueError("chain consumed signature differs from the required pattern")
        electron, chain = requirement["electron_configuration"], requirement["chain"]
        rng = np.random.Generator(np.random.PCG64(electron["seed"]))
        points = [rng.normal(scale=electron["scale"], size=(n, 3)) for n in fixture.electron_counts]
        initial = record["initial"]
        if not isinstance(initial, dict) or set(initial) != {"coordinates", "cond_G", "state"}:
            raise ValueError("chain initial record fields are invalid")
        if initial["coordinates"] != input_object_snapshot(tuple(points)):
            raise ValueError("chain initial coordinates differ from the requirement")
        if not _finite_condition(initial["cond_G"]) or not _state_valid(initial["state"]):
            raise ValueError("chain initial state or condition is invalid")
        moves = [
            (spin, index)
            for _ in range(chain["sweeps"])
            for spin, count in zip((True, False), fixture.electron_counts, strict=True)
            for index in range(count)
        ]
        steps = record["steps"]
        if not isinstance(steps, list) or len(steps) != len(moves):
            raise ValueError("chain required move count differs from consumption records")
        previous = initial["state"]
        for ordinal, ((spin, index), step) in enumerate(zip(moves, steps, strict=True)):
            if not isinstance(step, dict) or set(step) != {
                "ordinal",
                "moved_up",
                "index",
                "coordinates",
                "cond_G",
                "before",
                "after",
                "consumed_inverse",
                "outputs",
            }:
                raise ValueError("chain step fields are invalid")
            if (
                type(step["ordinal"]) is not int
                or step["ordinal"] != ordinal
                or type(step["moved_up"]) is not bool
                or step["moved_up"] != spin
                or type(step["index"]) is not int
                or step["index"] != index
            ):
                raise ValueError("chain move order differs from the requirement")
            points[0 if spin else 1][index] += rng.normal(scale=chain["displacement_scale"], size=3)
            if step["coordinates"] != input_object_snapshot(tuple(points)):
                raise ValueError("chain consumed coordinates differ from the required move")
            if not _finite_condition(step["cond_G"]):
                raise ValueError("chain consumed condition is invalid")
            if not _state_valid(step["before"]) or not _state_valid(step["after"]) or step["before"] != previous:
                raise ValueError("chain carried state continuity is broken")
            outputs = step["outputs"]
            if (
                not isinstance(outputs, dict)
                or set(outputs) != {"inverse", "fast", "streaming"}
                or not all(_snapshot(value) for value in outputs.values())
            ):
                raise ValueError("chain output snapshots are invalid")
            if outputs["inverse"] != step["after"]["inverse"] or step["consumed_inverse"] != outputs["inverse"]:
                raise ValueError("chain consumer did not use the carried inverse")
            previous = step["after"]
        for quantity, name in canonical_comparisons(requirement).items():
            site = checks[name]["site"]
            calls = [
                call
                for invocation in (invocations or ())
                for attempt in invocation.get("tolerance_calls", ())
                if attempt.get("site") == site
                for call in attempt.get("calls", ())
            ]
            if len(calls) != len(moves):
                raise ValueError(f"chain {quantity} comparison count differs from required moves")
            for step, call in zip(steps, calls, strict=True):
                context = call.get("canonical_chain_step")
                if (
                    not isinstance(context, dict)
                    or type(context.get("ordinal")) is not int
                    or context
                    != {
                        "ordinal": step["ordinal"],
                        "coordinates": step["coordinates"],
                        "initial_inverse": initial["state"]["inverse"],
                    }
                ):
                    raise ValueError(f"chain {quantity} comparison belongs to a different consumed hop")
                snapshots = call.get("operand_snapshots", {})
                if (
                    call.get("outcome") != "normal"
                    or call.get("comparator") != "numpy.testing.assert_allclose"
                    or call.get("operand_check") != "passed"
                    or not isinstance(snapshots, dict)
                    or set(snapshots) != {"actual", "reference"}
                    or not all(_snapshot(v) for v in snapshots.values())
                    or snapshots["actual"] != step["outputs"][quantity]
                ):
                    raise ValueError(f"chain {quantity} consumed output did not reach its per-hop comparison")
    except (ValueError, AssertionError, TypeError, KeyError, AttributeError) as exc:
        return [f"canonical fixture use: {exc}"]
    return []
