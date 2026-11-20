"""Small runtime receipt types shared by GDD-connected pytest evidence."""

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
import os
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar

import numpy as np

from gdd.engine.fixture_observations import HamiltonianFixtureReceipt, observe_hamiltonian


def canonical_snapshot(value: Any) -> str:
    """Hash a JSON-compatible declaration using one canonical representation."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


_EMISSION_COUNTER = 0
# Set by ``gdd.engine.derivative_registry.DerivativeSelfCheck`` for the duration of
# one pytest case: every emitted receipt payload, written or not (issue 789).
_SELFCHECK_SINK = None


def emit_completion_record(
    receipt: CaseReceipt | CompositeCaseReceipt,
    *,
    record_kind: str = "case",
    upstream_obligation_id: str | None = None,
    component_id: str | None = None,
) -> None:
    """Hand the receipt to the in-test self-check and append it to this worker's CI shard."""
    from gdd.engine.receipt_records import encode_case_receipt

    record = {
        "schema_version": 2,
        "record_kind": record_kind,
        "guarantee_id": receipt.guarantee_id,
        "obligation_id": receipt.obligation_id,
        "upstream_obligation_id": upstream_obligation_id,
        "component_id": component_id,
        "pytest_node_id": receipt.pytest_node_id,
        "status": receipt.status,
        "evidence": encode_case_receipt(receipt),
    }
    if _SELFCHECK_SINK is not None:
        _SELFCHECK_SINK(dict(record))
    directory = os.environ.get("GDD_RECEIPT_DIR")
    if not directory or os.environ.get("GDD_RECEIPT_AUTHORITY", "1") != "1":
        return
    lane = os.environ.get("GDD_EVIDENCE_LANE")
    if not lane:
        raise AssertionError("GDD_RECEIPT_DIR requires an explicit GDD_EVIDENCE_LANE")
    payload = {
        **record,
        "lane": lane,
        "worker_id": os.environ.get("PYTEST_XDIST_WORKER", "controller"),
        "run_envelope": {
            "commit_sha": os.environ.get("GDD_COMMIT_SHA", "local"),
            "contract_snapshot": os.environ.get("GDD_CONTRACT_SNAPSHOT") or _contract_snapshot(),
            "build_id": os.environ.get("GDD_BUILD_ID", "local"),
            "run_id": os.environ.get("GDD_RUN_ID", "local"),
            "run_attempt": os.environ.get("GDD_RUN_ATTEMPT", "local"),
            "job_id": os.environ.get("GDD_JOB_ID", "local"),
            "shard_id": os.environ.get("GDD_SHARD_ID", "local"),
            "worker_id": os.environ.get("PYTEST_XDIST_WORKER", "controller"),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    record = {**payload, "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()}
    os.makedirs(directory, exist_ok=True)
    global _EMISSION_COUNTER
    _EMISSION_COUNTER += 1
    path = os.path.join(directory, f"receipts-{os.getpid()}-{_EMISSION_COUNTER}.json")
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True)
        stream.write("\n")


def _contract_snapshot() -> str:
    """Resolve the same SSoT digest used by ordinary module receipts."""
    from gdd.engine.module_receipts import contract_snapshot

    return contract_snapshot()


@dataclass(frozen=True)
class InputBinding:
    """Resolved origin and transformation chain for one concrete test input."""

    binding_id: str
    kind: str
    builder_operation: str
    transformations: tuple[str, ...]
    source_id: str | None = None
    source_ids: tuple[str, ...] = ()
    artifact_format: str | None = None
    parameter_set_id: str | None = None
    parameter_source: str | None = None
    content_snapshot: str | None = None

    def validate(self) -> None:
        for name, value in (("binding_id", self.binding_id), ("builder_operation", self.builder_operation)):
            if not value:
                raise AssertionError(f"input binding has no {name}")
        if self.kind == "committed-artifact":
            resolved_sources = self.source_ids or ((self.source_id,) if self.source_id else ())
            if not resolved_sources or len(set(resolved_sources)) != len(resolved_sources) or not self.artifact_format:
                raise AssertionError("committed-artifact input requires unique source IDs and format")
            if self.source_id is not None and self.source_ids:
                raise AssertionError("committed-artifact input must use source_id or source_ids, not both")
            if any(value is not None for value in (self.parameter_set_id, self.parameter_source, self.content_snapshot)):
                raise AssertionError("committed-artifact input cannot carry parameter-set fields")
        elif self.kind == "parameter-set":
            if not self.parameter_set_id or not self.parameter_source or not self.content_snapshot:
                raise AssertionError("parameter-set input requires ID, source, and content snapshot")
            if self.source_id is not None or self.source_ids or self.artifact_format is not None:
                raise AssertionError("parameter-set input cannot carry artifact fields")
        else:
            raise AssertionError(f"unsupported input kind: {self.kind}")


@dataclass(frozen=True)
class ElectronConfigurationBinding:
    """Requested electron-position sequence and geminal ``cond_2(G)`` policy."""

    binding_id: str
    mode: str
    generation_pattern: str | None = None
    position_source_kind: str | None = None
    generator_operation: str | None = None
    generator_arguments: tuple[tuple[str, Any], ...] = ()
    random_algorithm: str | None = None
    seed: int | None = None
    configuration_count: int = 0
    position_set_id: str | None = None
    coordinates_source: str | None = None
    content_snapshot: str | None = None
    geminal_conditioning_mode: str | None = None
    conditioning_reason: str | None = None
    minimum_cond_g: float | None = None
    maximum_cond_g: float | None = None
    lower_bound_comparison: str | None = None
    upper_bound_comparison: str | None = None
    minimum_singular_value: float | None = None
    conditioning_violation_action: str | None = None
    maximum_generation_attempts: int | None = None
    production_route_capture: str | None = None
    # The screened conditioning subject.
    # ``geminal-matrix-G`` is the default and keeps every cond(G) rule below
    # unchanged; ``polynomial-evaluation-kappa`` screens the evaluation
    # condition number kappa(x) = sum|t_i| / |sum t_i| of a polynomial-type
    # quantity against a central cap and never uses the cond(G) fields.
    conditioning_subject: str = "geminal-matrix-G"
    maximum_evaluation_kappa: float | None = None
    evaluation_kappa_comparison: str | None = None
    evaluation_kappa_cap_resolver: str | None = None

    CONDITIONING_SUBJECTS: ClassVar[tuple[str, ...]] = ("geminal-matrix-G", "polynomial-evaluation-kappa")
    EVALUATION_KAPPA_RESOLVERS: ClassVar[tuple[str, ...]] = ("jqmc._precision.get_evaluation_kappa_max",)

    def validate(self) -> None:
        if not self.binding_id:
            raise AssertionError("electron-configuration binding has no ID")
        if self.conditioning_subject not in self.CONDITIONING_SUBJECTS:
            raise AssertionError(f"unknown conditioning subject: {self.conditioning_subject!r}")
        if self.conditioning_subject == "polynomial-evaluation-kappa":
            self._validate_evaluation_kappa()
            return
        if any(
            value is not None
            for value in (
                self.maximum_evaluation_kappa,
                self.evaluation_kappa_comparison,
                self.evaluation_kappa_cap_resolver,
            )
        ):
            raise AssertionError("evaluation-kappa fields require the polynomial-evaluation-kappa conditioning subject")
        if self.production_route_capture is not None:
            if not self.production_route_capture:
                raise AssertionError("declared production route capture has no route name")
            if self.mode != "required":
                raise AssertionError("route capture requires realized electron positions")
        position_fields = (
            self.generation_pattern,
            self.position_source_kind,
            self.generator_operation,
            self.random_algorithm,
            self.seed,
            self.position_set_id,
            self.coordinates_source,
            self.content_snapshot,
            self.geminal_conditioning_mode,
        )
        if self.mode == "not-required":
            if self.configuration_count != 0 or self.generator_arguments or any(value is not None for value in position_fields):
                raise AssertionError("not-required electron binding cannot carry position fields")
            return
        if self.mode != "required":
            raise AssertionError(f"unknown electron-configuration mode: {self.mode}")
        if self.configuration_count < 1:
            raise AssertionError("electron configuration count must be positive")
        if self.generation_pattern not in {
            "direct",
            "rejection-screen",
            "condition-target",
            "rank-deficient-target",
        }:
            raise AssertionError(f"unknown or missing electron generation pattern: {self.generation_pattern}")
        if self.generation_pattern == "rank-deficient-target":
            raise AssertionError(
                "rank-deficient-target is reserved until numerical-rank observations and production disposition exist"
            )
        if self.position_source_kind == "seeded-random":
            if not self.generator_operation or not self.random_algorithm or self.seed is None or self.seed < 0:
                raise AssertionError("seeded electron positions require operation, algorithm, and nonnegative seed")
            if any(value is not None for value in (self.position_set_id, self.coordinates_source, self.content_snapshot)):
                raise AssertionError("seeded electron positions cannot carry explicit-position fields")
        elif self.position_source_kind == "explicit":
            if not self.position_set_id or not self.coordinates_source or not self.content_snapshot:
                raise AssertionError("explicit electron positions require ID, source, and content snapshot")
            if self.generator_operation is not None or self.random_algorithm is not None or self.seed is not None:
                raise AssertionError("explicit electron positions cannot carry random-generator fields")
            if self.generator_arguments:
                raise AssertionError("explicit electron positions cannot carry generator arguments")
        else:
            raise AssertionError(f"unknown electron-position source: {self.position_source_kind}")
        thresholds = (
            self.minimum_cond_g,
            self.maximum_cond_g,
            self.lower_bound_comparison,
            self.upper_bound_comparison,
            self.minimum_singular_value,
            self.conditioning_violation_action,
            self.maximum_generation_attempts,
        )
        if self.geminal_conditioning_mode == "not-applicable":
            if not self.conditioning_reason:
                raise AssertionError("not-applicable cond(G) requires a reason")
            if any(value is not None for value in thresholds):
                raise AssertionError("not-applicable cond(G) cannot carry thresholds")
        elif self.geminal_conditioning_mode == "not-limited":
            if self.conditioning_reason is not None or any(value is not None for value in thresholds):
                raise AssertionError("not-limited cond(G) cannot carry a hidden threshold")
        elif self.geminal_conditioning_mode == "upper-bounded":
            self._validate_constrained_common()
            if self.minimum_cond_g is not None or self.lower_bound_comparison is not None:
                raise AssertionError("upper-bounded cond(G) cannot carry a lower bound")
            if self.maximum_cond_g is None or not np.isfinite(self.maximum_cond_g) or self.maximum_cond_g < 1.0:
                raise AssertionError("upper-bounded cond(G) requires a finite maximum >= 1")
            if self.upper_bound_comparison not in {"less-than", "less-than-or-equal"}:
                raise AssertionError("upper-bounded cond(G) requires an explicit upper comparator")
            if self.minimum_singular_value is not None:
                raise AssertionError("upper-bounded cond(G) cannot carry a target-range sigma_min guard")
        elif self.geminal_conditioning_mode == "target-range":
            self._validate_constrained_common()
            if (
                self.minimum_cond_g is None
                or self.maximum_cond_g is None
                or not np.isfinite(self.minimum_cond_g)
                or not np.isfinite(self.maximum_cond_g)
                or self.minimum_cond_g < 1.0
                or self.maximum_cond_g <= self.minimum_cond_g
            ):
                raise AssertionError("target-range cond(G) requires ordered finite bounds >= 1")
            if self.lower_bound_comparison not in {"greater-than", "greater-than-or-equal"}:
                raise AssertionError("target-range cond(G) requires an explicit lower comparator")
            if self.upper_bound_comparison not in {"less-than", "less-than-or-equal"}:
                raise AssertionError("target-range cond(G) requires an explicit upper comparator")
            if self.minimum_singular_value is not None and (
                not np.isfinite(self.minimum_singular_value) or self.minimum_singular_value <= 0.0
            ):
                raise AssertionError("sigma_min(G) guard must be finite and positive")
        else:
            raise AssertionError(f"unknown geminal conditioning mode: {self.geminal_conditioning_mode}")
        self._validate_generation_pattern()

    def _validate_evaluation_kappa(self) -> None:
        """Validate an evaluation-kappa screen; it can never borrow a cond(G) field."""
        if self.mode != "required":
            raise AssertionError("an evaluation-kappa screen requires realized evaluation positions")
        if self.production_route_capture is not None:
            raise AssertionError("an evaluation-kappa screen cannot claim a production-G route capture")
        if self.configuration_count < 1:
            raise AssertionError("electron configuration count must be positive")
        if self.position_source_kind != "seeded-random":
            raise AssertionError("an evaluation-kappa screen requires seeded-random positions")
        if not self.generator_operation or not self.random_algorithm or self.seed is None or self.seed < 0:
            raise AssertionError("seeded electron positions require operation, algorithm, and nonnegative seed")
        if any(value is not None for value in (self.position_set_id, self.coordinates_source, self.content_snapshot)):
            raise AssertionError("seeded electron positions cannot carry explicit-position fields")
        if any(
            value is not None
            for value in (
                self.minimum_cond_g,
                self.maximum_cond_g,
                self.lower_bound_comparison,
                self.upper_bound_comparison,
                self.minimum_singular_value,
            )
        ):
            raise AssertionError("evaluation kappa cannot be declared through cond(G) fields")
        if self.geminal_conditioning_mode != "not-applicable" or not self.conditioning_reason:
            raise AssertionError("an evaluation-kappa screen declares cond(G) not-applicable with a reason")
        cap = self.maximum_evaluation_kappa
        if cap is None or isinstance(cap, (bool, np.bool_)) or not isinstance(cap, Real) or not np.isfinite(cap) or cap < 1.0:
            raise AssertionError("an evaluation-kappa screen requires a finite cap >= 1")
        if self.evaluation_kappa_comparison not in {"less-than", "less-than-or-equal"}:
            raise AssertionError("an evaluation-kappa cap requires an explicit upper comparator")
        if self.evaluation_kappa_cap_resolver not in self.EVALUATION_KAPPA_RESOLVERS:
            raise AssertionError("an evaluation-kappa cap must come from the central resolver")
        if self.generation_pattern not in {"direct", "rejection-screen"}:
            raise AssertionError(f"unsupported evaluation-kappa generation pattern: {self.generation_pattern}")
        if self.conditioning_violation_action not in {"fail-case", "deterministic-resample"}:
            raise AssertionError("an evaluation-kappa screen requires an explicit violation action")
        if self.generation_pattern == "rejection-screen":
            if self.conditioning_violation_action != "deterministic-resample":
                raise AssertionError("rejection-screen requires deterministic-resample")
            if self.maximum_generation_attempts is None or self.maximum_generation_attempts < 1:
                raise AssertionError("rejection-screen requires a maximum attempt count")
        else:
            if self.conditioning_violation_action == "deterministic-resample":
                raise AssertionError("direct generation cannot claim deterministic rejection resampling")
            if self.maximum_generation_attempts is not None:
                raise AssertionError("fail-case evaluation kappa cannot carry a resampling attempt limit")
        argument_names = [name for name, _value in self.generator_arguments]
        if len(argument_names) != len(set(argument_names)):
            raise AssertionError("electron generator argument names must be unique")

    def _validate_generation_pattern(self) -> None:
        constrained = self.geminal_conditioning_mode in {"upper-bounded", "target-range"}
        argument_names = [name for name, _value in self.generator_arguments]
        if len(argument_names) != len(set(argument_names)):
            raise AssertionError("electron generator argument names must be unique")
        if self.generation_pattern == "direct":
            if self.conditioning_violation_action == "deterministic-resample":
                raise AssertionError("direct generation cannot claim deterministic rejection resampling")
        elif self.generation_pattern == "rejection-screen":
            if self.position_source_kind != "seeded-random":
                raise AssertionError("rejection-screen requires seeded-random positions")
            if not constrained:
                raise AssertionError("rejection-screen requires constrained cond(G)")
            if self.conditioning_violation_action != "deterministic-resample":
                raise AssertionError("rejection-screen requires deterministic-resample")
            if self.maximum_generation_attempts is None:
                raise AssertionError("rejection-screen requires a maximum attempt count")
        elif self.generation_pattern == "condition-target":
            if self.geminal_conditioning_mode != "target-range":
                raise AssertionError("condition-target requires target-range cond(G)")
            if self.conditioning_violation_action != "fail-case":
                raise AssertionError("condition-target must fail loudly when its search misses the target")
            if self.maximum_generation_attempts is None or self.maximum_generation_attempts < 1:
                raise AssertionError("condition-target requires a maximum search attempt count")
            if self.minimum_singular_value is None:
                raise AssertionError("condition-target requires a positive sigma_min(G) scale guard")
            required_arguments = {"control_parameter", "search_method"}
            if not required_arguments <= set(argument_names):
                raise AssertionError("condition-target requires control_parameter and search_method generator arguments")

    def _validate_constrained_common(self) -> None:
        if self.conditioning_reason is not None:
            raise AssertionError("a constrained cond(G) binding cannot carry an N/A reason")
        if self.conditioning_violation_action not in {"fail-case", "deterministic-resample"}:
            raise AssertionError("constrained cond(G) requires an explicit violation action")
        if self.conditioning_violation_action == "deterministic-resample":
            if self.maximum_generation_attempts is None or self.maximum_generation_attempts < 1:
                raise AssertionError("deterministic cond(G) resampling requires a positive attempt limit")
        elif self.maximum_generation_attempts is not None and self.generation_pattern != "condition-target":
            raise AssertionError("fail-case cond(G) cannot carry a resampling attempt limit")


@dataclass(frozen=True)
class ElectronConfigurationReceipt:
    binding_id: str
    mode: str
    generation_pattern: str | None
    generator_operation: str | None
    canonical_generator_arguments: tuple[tuple[str, str], ...]
    requested_configuration_count: int
    realized_configuration_count: int
    ordered_coordinate_snapshots: tuple[str, ...]
    electron_counts_per_configuration: tuple[tuple[int, int], ...]
    finite_coordinate_check: bool
    geminal_conditioning_mode: str | None
    configured_minimum_cond_g: float | None
    configured_maximum_cond_g: float | None
    lower_bound_comparison: str | None
    upper_bound_comparison: str | None
    configured_minimum_singular_value: float | None
    cond_g_per_configuration: tuple[float, ...]
    sigma_min_g_per_configuration: tuple[float, ...]
    sigma_max_g_per_configuration: tuple[float, ...]
    machine_epsilon_per_configuration: tuple[float, ...]
    cond_times_epsilon_per_configuration: tuple[float, ...]
    generation_attempts: int
    rejected_configuration_count: int
    configured_maximum_generation_attempts: int | None
    declared_production_route: str | None = None
    conditioning_subject: str = "geminal-matrix-G"
    configured_maximum_evaluation_kappa: float | None = None
    evaluation_kappa_comparison: str | None = None
    evaluation_kappa_per_configuration: tuple[float, ...] = ()

    def sequence_token(self) -> str:
        """Deterministic digest of this finalized receipt.

        A comparison receipt stamped with this token proves it was created only
        after the realized configurations and their screening were finalized.
        """
        payload = {
            "binding_id": self.binding_id,
            "mode": self.mode,
            "generation_pattern": self.generation_pattern,
            "generator_operation": self.generator_operation,
            "generator_arguments": list(self.canonical_generator_arguments),
            "snapshots": list(self.ordered_coordinate_snapshots),
            "cond_g": [repr(value) for value in self.cond_g_per_configuration],
            "sigma_min_g": [repr(value) for value in self.sigma_min_g_per_configuration],
            "sigma_max_g": [repr(value) for value in self.sigma_max_g_per_configuration],
            "machine_epsilon": [repr(value) for value in self.machine_epsilon_per_configuration],
            "cond_times_epsilon": [repr(value) for value in self.cond_times_epsilon_per_configuration],
            "attempts": self.generation_attempts,
            "rejected": self.rejected_configuration_count,
            "maximum_attempts": self.configured_maximum_generation_attempts,
            "route": self.declared_production_route,
        }
        if self.conditioning_subject != "geminal-matrix-G":
            # cond(G) receipts keep their historical payload; only the new
            # subject adds its own observations to the finalized digest.
            payload["conditioning_subject"] = self.conditioning_subject
            payload["evaluation_kappa"] = [repr(value) for value in self.evaluation_kappa_per_configuration]
            payload["maximum_evaluation_kappa"] = repr(self.configured_maximum_evaluation_kappa)
            payload["evaluation_kappa_comparison"] = self.evaluation_kappa_comparison
        return canonical_snapshot(payload)

    def validate(self) -> None:
        if self.conditioning_subject not in ElectronConfigurationBinding.CONDITIONING_SUBJECTS:
            raise AssertionError(f"unknown conditioning subject: {self.conditioning_subject!r}")
        if self.conditioning_subject == "polynomial-evaluation-kappa":
            _validate_evaluation_kappa_receipt(self)
        elif (
            self.configured_maximum_evaluation_kappa is not None
            or self.evaluation_kappa_comparison is not None
            or self.evaluation_kappa_per_configuration
        ):
            raise AssertionError("evaluation-kappa observations require the polynomial-evaluation-kappa subject")
        if self.mode == "not-required":
            if (
                any(
                    (
                        self.requested_configuration_count,
                        self.realized_configuration_count,
                        len(self.ordered_coordinate_snapshots),
                        len(self.electron_counts_per_configuration),
                        self.generation_attempts,
                        self.rejected_configuration_count,
                        self.configured_maximum_generation_attempts,
                    )
                )
                or self.geminal_conditioning_mode is not None
                or self.generation_pattern is not None
                or self.generator_operation is not None
                or self.canonical_generator_arguments
            ):
                raise AssertionError("not-required electron receipt contains realized positions")
            return
        if self.mode != "required":
            raise AssertionError(f"unknown electron receipt mode: {self.mode}")
        if self.realized_configuration_count != self.requested_configuration_count:
            raise AssertionError("requested and realized electron-configuration counts differ")
        if self.generation_attempts < 1 or self.rejected_configuration_count < 0:
            raise AssertionError("electron generation attempt counts are invalid")
        if len(self.ordered_coordinate_snapshots) != self.realized_configuration_count:
            raise AssertionError("one coordinate snapshot is required per configuration")
        if len(set(self.ordered_coordinate_snapshots)) != len(self.ordered_coordinate_snapshots):
            raise AssertionError("generated electron configurations unexpectedly duplicate")
        if not self.finite_coordinate_check:
            raise AssertionError("electron coordinates contain NaN or infinity")
        if self.generation_pattern not in {"direct", "rejection-screen", "condition-target"}:
            raise AssertionError(f"electron receipt has an unsupported generation pattern: {self.generation_pattern}")
        argument_names = [name for name, _value in self.canonical_generator_arguments]
        if len(argument_names) != len(set(argument_names)):
            raise AssertionError("electron receipt has duplicate canonical generator arguments")
        if self.generation_pattern in {"rejection-screen", "condition-target"} and not self.generator_operation:
            raise AssertionError("screened or targeted electron receipt has no generator operation")
        if self.generation_pattern == "condition-target" and not {
            "control_parameter",
            "search_method",
        } <= set(argument_names):
            raise AssertionError("condition-target receipt is missing its canonical control/search arguments")
        if self.generation_pattern in {"rejection-screen", "condition-target"}:
            if self.configured_maximum_generation_attempts is None:
                raise AssertionError("screened or targeted electron receipt has no maximum attempt count")
        elif self.configured_maximum_generation_attempts is not None:
            raise AssertionError("direct electron receipt cannot carry a maximum attempt count")
        if self.generation_attempts != self.realized_configuration_count + self.rejected_configuration_count:
            raise AssertionError("electron attempt count must equal realized plus rejected configurations")
        if (
            self.configured_maximum_generation_attempts is not None
            and self.generation_attempts > self.configured_maximum_generation_attempts
        ):
            raise AssertionError("electron generation exceeded its configured maximum attempt count")
        if self.geminal_conditioning_mode in {"upper-bounded", "target-range"}:
            if len(self.cond_g_per_configuration) != self.realized_configuration_count:
                raise AssertionError("constrained cond(G) requires one observation per configuration")
            for observed in self.cond_g_per_configuration:
                if not np.isfinite(observed):
                    raise AssertionError("constrained cond(G) observation is not finite")
                _validate_observed_bound(observed, self.configured_minimum_cond_g, self.lower_bound_comparison)
                _validate_observed_bound(observed, self.configured_maximum_cond_g, self.upper_bound_comparison)
        elif self.cond_g_per_configuration:
            raise AssertionError("unconstrained cond(G) receipt cannot claim conditioning observations")
        if self.configured_minimum_singular_value is not None:
            if len(self.sigma_min_g_per_configuration) != self.realized_configuration_count:
                raise AssertionError("sigma_min(G) guard requires one observation per configuration")
            if any(
                not np.isfinite(value) or value < self.configured_minimum_singular_value
                for value in self.sigma_min_g_per_configuration
            ):
                raise AssertionError("observed sigma_min(G) violates its configured minimum")
        elif self.sigma_min_g_per_configuration:
            raise AssertionError("sigma_min(G) observations require a configured guard")
        extended_observations = (
            self.sigma_max_g_per_configuration,
            self.machine_epsilon_per_configuration,
            self.cond_times_epsilon_per_configuration,
        )
        if self.generation_pattern == "condition-target":
            if any(len(values) != self.realized_configuration_count for values in extended_observations):
                raise AssertionError("condition-target requires sigma_max, machine epsilon, and cond*epsilon observations")
            for cond_g, sigma_min, sigma_max, epsilon, cond_epsilon in zip(
                self.cond_g_per_configuration,
                self.sigma_min_g_per_configuration,
                self.sigma_max_g_per_configuration,
                self.machine_epsilon_per_configuration,
                self.cond_times_epsilon_per_configuration,
                strict=True,
            ):
                if not all(np.isfinite(value) and value > 0.0 for value in (sigma_min, sigma_max, epsilon, cond_epsilon)):
                    raise AssertionError("condition-target observations must be finite and positive")
                if not np.isclose(cond_g, sigma_max / sigma_min, rtol=1.0e-12, atol=0.0):
                    raise AssertionError("condition-target cond(G) disagrees with sigma_max/sigma_min")
                if not np.isclose(cond_epsilon, cond_g * epsilon, rtol=1.0e-12, atol=0.0):
                    raise AssertionError("condition-target cond(G)*epsilon observation is inconsistent")
        elif any(extended_observations):
            raise AssertionError("extended conditioning observations require condition-target generation")


def _validate_evaluation_kappa_receipt(receipt: ElectronConfigurationReceipt) -> None:
    """Check one evaluation-kappa receipt without touching any cond(G) rule."""
    if receipt.mode != "required":
        raise AssertionError("an evaluation-kappa receipt requires realized evaluation positions")
    if receipt.geminal_conditioning_mode != "not-applicable" or any(
        (
            receipt.configured_minimum_cond_g is not None,
            receipt.configured_maximum_cond_g is not None,
            receipt.lower_bound_comparison is not None,
            receipt.upper_bound_comparison is not None,
            receipt.configured_minimum_singular_value is not None,
            bool(receipt.cond_g_per_configuration),
            bool(receipt.sigma_min_g_per_configuration),
            bool(receipt.sigma_max_g_per_configuration),
            bool(receipt.machine_epsilon_per_configuration),
            bool(receipt.cond_times_epsilon_per_configuration),
        )
    ):
        raise AssertionError("evaluation kappa cannot be recorded through cond(G) fields")
    cap = receipt.configured_maximum_evaluation_kappa
    if cap is None or isinstance(cap, (bool, np.bool_)) or not isinstance(cap, Real) or not np.isfinite(cap) or cap < 1.0:
        raise AssertionError("an evaluation-kappa receipt requires its finite configured cap")
    if receipt.evaluation_kappa_comparison not in {"less-than", "less-than-or-equal"}:
        raise AssertionError("an evaluation-kappa receipt requires its explicit upper comparator")
    if len(receipt.evaluation_kappa_per_configuration) != receipt.realized_configuration_count:
        raise AssertionError("an evaluation-kappa screen requires one observation per accepted configuration")
    for observed in receipt.evaluation_kappa_per_configuration:
        if isinstance(observed, (bool, np.bool_)) or not isinstance(observed, Real) or not np.isfinite(observed):
            raise AssertionError("evaluation-kappa observation is not finite")
        if observed < 1.0:
            raise AssertionError("evaluation kappa is sum|t|/|sum t| and cannot be below one")
        passed = observed < cap if receipt.evaluation_kappa_comparison == "less-than" else observed <= cap
        if not passed:
            raise AssertionError(f"observed evaluation kappa={observed} violates {receipt.evaluation_kappa_comparison} {cap}")


def _validate_observed_bound(observed: float, threshold: float | None, comparison: str | None) -> None:
    if threshold is None and comparison is None:
        return
    if threshold is None:
        raise AssertionError(f"cond(G) comparator {comparison} has no threshold")
    if comparison == "greater-than":
        passed = observed > threshold
    elif comparison == "greater-than-or-equal":
        passed = observed >= threshold
    elif comparison == "less-than":
        passed = observed < threshold
    elif comparison == "less-than-or-equal":
        passed = observed <= threshold
    else:
        raise AssertionError(f"unknown cond(G) comparator: {comparison}")
    if not passed:
        raise AssertionError(f"observed cond(G)={observed} violates {comparison} {threshold}")


def _validate_tolerance(value: float) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise AssertionError("comparison tolerance must be a finite nonnegative real number")


def _validate_comparison_counts(arrays: int, elements: int) -> None:
    if type(arrays) is not int or type(elements) is not int or arrays < 1 or elements < 1:
        raise AssertionError("comparison receipt is vacuous or has invalid counts")


@dataclass(frozen=True)
class ComparisonBinding:
    _ZONE_POLICIES: ClassVar[dict[str, str]] = {
        "CMP-ZONE-ALLCLOSE-STRICT-001": "strict",
        "CMP-ZONE-ALLCLOSE-MEDIUM-001": "medium",
        "CMP-XLA-RUST-STRICT-001": "strict",
        "CMP-XLA-RUST-MEDIUM-001": "medium",
        "CMP-XLA-RUST-INDEPENDENT-FP32-001": "independent_fp32",
    }
    _FIXED_POLICIES: ClassVar[dict[str, str]] = {
        "CMP-DERIV-AUTODIFF-ANALYTICAL-FULL-001": "full",
        "CMP-FD-AUTODIFF-FULL-001": "full",
        "CMP-INTEGRATION-MIXED-001": "mixed",
        "CMP-MIXED-FULL-FIDELITY-001": "mixed",
    }
    # These central pairs apply in either execution mode. The DTN pair is
    # only for stored fp64 coefficient constraints, not a Jastrow error budget.
    _FIXED_ANY_MODE_POLICIES: ClassVar[set[str]] = {
        "CMP-PW-PROJECTION-FP64-001",
        "CMP-LM-HOST-FP64-001",
        "CMP-TREXIO-STORAGE-FP64-001",
        "CMP-FIXED-SAMPLE-FP64-001",
        "CMP-GFMC-INCREMENTAL-TWIN-001",
        "CMP-NATIVE-GAUSSIAN-F64-001",
        "CMP-DTN-EN-CUSP-CONSTRAINT-001",
    }
    _EXACT_POLICIES: ClassVar[set[str]] = {"CMP-RAW-BITS-001", "CMP-EXACT-VALUES-001"}
    # Module adapters carry decimal digits or finite central-pair consumer
    # identities that this engine's simpler allclose/exact bindings do not.
    _MODULE_ONLY_POLICIES: ClassVar[set[str]] = {
        "CMP-TURBORVB-DECIMAL-001",
        "CMP-SOURCE-CENTRAL-PAIR-001",
        "CMP-SOURCE-ARRAY-EQUALITY-001",
        "CMP-SOURCE-SCALED-SPACING-001",
        "CMP-ZONE-ATOL-ZERO-RTOL-001",
    }
    _ALLCLOSE_POLICIES: ClassVar[set[str]] = set(_ZONE_POLICIES) | set(_FIXED_POLICIES) | _FIXED_ANY_MODE_POLICIES

    binding_id: str
    policy_id: str
    precision_mode: str
    actual_role: str
    reference_role: str
    zones: tuple[str, ...] = ()
    level: str | None = None

    def validate(self) -> None:
        allowed_modes = {"full", "mixed"}
        if self.precision_mode not in allowed_modes:
            raise AssertionError(f"unknown comparison precision: {self.precision_mode}")
        if not self.actual_role or not self.reference_role:
            raise AssertionError("comparison operand roles are required")
        if self.policy_id in self._ZONE_POLICIES:
            if not self.zones or self.level != self._ZONE_POLICIES[self.policy_id]:
                raise AssertionError("zone comparison requires matching nonempty zones and level")
        elif self.policy_id in self._FIXED_POLICIES:
            if self.precision_mode != self._FIXED_POLICIES[self.policy_id] or self.zones or self.level is not None:
                raise AssertionError("fixed comparison policy has incompatible precision or resolver arguments")
        elif self.policy_id in self._FIXED_ANY_MODE_POLICIES:
            if self.zones or self.level is not None:
                raise AssertionError("fixed comparison policy has incompatible resolver arguments")
        elif self.policy_id in self._EXACT_POLICIES:
            if self.zones or self.level is not None:
                raise AssertionError("exact comparison cannot carry tolerance resolver arguments")
        else:
            raise AssertionError(f"unknown comparison policy: {self.policy_id}")


@dataclass(frozen=True)
class ComparisonReceipt:
    binding_id: str
    policy_id: str
    resolved_atol: float
    resolved_rtol: float
    max_absolute_error: float
    max_relative_error: float
    failing_element_count: int
    compared_array_count: int
    compared_element_count: int
    electron_sequence_token: str | None = None

    def validate(self) -> None:
        if not self.binding_id or self.policy_id not in ComparisonBinding._ALLCLOSE_POLICIES:
            raise AssertionError("comparison receipt has no binding or has an incompatible policy")
        _validate_tolerance(self.resolved_atol)
        _validate_tolerance(self.resolved_rtol)
        _validate_comparison_counts(self.compared_array_count, self.compared_element_count)
        if not np.isfinite(self.max_absolute_error) or self.max_absolute_error < 0:
            raise AssertionError("comparison absolute error must be finite and nonnegative")
        # Nonzero error at a zero reference has infinite relative error but can
        # still satisfy the absolute tolerance. NaN and negative errors cannot.
        if np.isnan(self.max_relative_error) or self.max_relative_error < 0:
            raise AssertionError("comparison relative error must be nonnegative and not NaN")
        if self.failing_element_count != 0:
            raise AssertionError(f"comparison receipt has {self.failing_element_count} failing elements")


@dataclass(frozen=True)
class ExactComparisonReceipt:
    binding_id: str
    policy_id: str
    differing_element_count: int
    compared_array_count: int
    compared_element_count: int
    electron_sequence_token: str | None = None

    def validate(self) -> None:
        if not self.binding_id or self.policy_id not in ComparisonBinding._EXACT_POLICIES:
            raise AssertionError("exact comparison receipt has no binding or has an incompatible policy")
        _validate_comparison_counts(self.compared_array_count, self.compared_element_count)
        if self.differing_element_count != 0:
            raise AssertionError(f"exact comparison has {self.differing_element_count} differing elements")


@dataclass(frozen=True)
class ObservationBinding:
    binding_id: str
    policy_id: str
    precision_mode: str
    subject_role: str
    required_observations: tuple[str, ...]
    applicability_rule_id: str | None = None

    def validate(self) -> None:
        allowed_policies = {
            "OBS-MIXED-ROUTE-SMOKE-001",
            "OBS-ROUTE-LIVENESS-001",
            "OBS-EXPECTED-ERROR-001",
            "OBS-REVIEWED-N-A-001",
            "OBS-GFMC-ROUTE-001",
            "OBS-COMPOSITE-CLOSURE-001",
            "OBS-NUMERICAL-REGIME-001",
            "OBS-FD-ROUTE-001",
        }
        if self.policy_id not in allowed_policies:
            raise AssertionError(f"unknown observation policy: {self.policy_id}")
        if self.precision_mode not in {"full", "mixed"} or not self.subject_role:
            raise AssertionError("observation precision and subject role are required")
        if not self.required_observations or len(set(self.required_observations)) != len(self.required_observations):
            raise AssertionError("observation binding requires unique observations")
        if self.policy_id == "OBS-REVIEWED-N-A-001" and not self.applicability_rule_id:
            raise AssertionError("reviewed N/A observation requires an applicability rule")
        if self.policy_id != "OBS-REVIEWED-N-A-001" and self.applicability_rule_id is not None:
            raise AssertionError("only reviewed N/A may carry an applicability rule")


@dataclass(frozen=True)
class ObservationReceipt:
    binding_id: str
    policy_id: str
    completed_observations: tuple[str, ...]
    status: str
    electron_sequence_token: str | None = None

    def validate(self) -> None:
        if self.status != "passed":
            raise AssertionError(f"observation receipt status is {self.status!r}")
        if not self.completed_observations:
            raise AssertionError("observation receipt is vacuous")


@dataclass(frozen=True)
class CaseReceipt:
    guarantee_id: str
    obligation_id: str
    pytest_node_id: str
    input_binding: InputBinding
    electron_configuration: ElectronConfigurationReceipt
    comparison: ComparisonReceipt | ExactComparisonReceipt | ObservationReceipt
    status: str
    additional_checks: tuple[ComparisonReceipt | ExactComparisonReceipt | ObservationReceipt, ...] = ()
    route_capture: RouteCaptureReceipt | None = None

    def validate(self) -> None:
        if not self.guarantee_id or not self.obligation_id or not self.pytest_node_id:
            raise AssertionError("case receipt identifiers are incomplete")
        self.input_binding.validate()
        self.electron_configuration.validate()
        self.comparison.validate()
        for check in self.additional_checks:
            check.validate()
        if self.status != "passed":
            raise AssertionError(f"case receipt status is {self.status!r}")
        declared_route = self.electron_configuration.declared_production_route
        if self.route_capture is not None:
            self.route_capture.validate()
        if declared_route is not None:
            if self.route_capture is None:
                raise AssertionError("declared production route has no route-capture receipt")
            if self.route_capture.route_name != declared_route:
                raise AssertionError(
                    f"route capture {self.route_capture.route_name!r} does not match "
                    f"the declared production route {declared_route!r}"
                )
            unreached = set(self.electron_configuration.ordered_coordinate_snapshots) - set(
                self.route_capture.coordinate_snapshots
            )
            if unreached:
                raise AssertionError(f"{len(unreached)} declared electron configuration(s) never reached the production route")
            expected_token = self.electron_configuration.sequence_token()
            for check in (self.comparison, *self.additional_checks):
                if getattr(check, "electron_sequence_token", None) != expected_token:
                    raise AssertionError(
                        "comparison receipt is not bound to the finalized electron receipt (screening-order proof missing)"
                    )
        else:
            expected_token = self.electron_configuration.sequence_token()
            for check in (self.comparison, *self.additional_checks):
                token = getattr(check, "electron_sequence_token", None)
                if token is not None and token != expected_token:
                    raise AssertionError("comparison receipt is bound to a different electron receipt")


@dataclass(frozen=True)
class CompositeCaseReceipt:
    guarantee_id: str
    obligation_id: str
    pytest_node_id: str
    component_receipts: tuple[CaseReceipt, ...]
    completion_check: ObservationReceipt
    status: str

    def validate(self) -> None:
        if not self.guarantee_id or not self.obligation_id or not self.pytest_node_id:
            raise AssertionError("composite receipt identifiers are incomplete")
        if not self.component_receipts:
            raise AssertionError("composite receipt has no component evidence")
        component_ids = [receipt.obligation_id for receipt in self.component_receipts]
        if len(component_ids) != len(set(component_ids)):
            raise AssertionError("composite receipt contains duplicate component obligations")
        for receipt in self.component_receipts:
            receipt.validate()
        self.completion_check.validate()
        if self.status != "passed":
            raise AssertionError(f"composite receipt status is {self.status!r}")


@dataclass(frozen=True)
class RouteFixtureReceipt:
    """An input and coordinates observed together in one normally returned call."""

    fixture: HamiltonianFixtureReceipt
    coordinate_snapshots: tuple[str, ...]

    def validate(self) -> None:
        self.fixture.validate()
        if not self.coordinate_snapshots or len(set(self.coordinate_snapshots)) != len(self.coordinate_snapshots):
            raise AssertionError("fixture route requires unique realized coordinate snapshots")


@dataclass(frozen=True)
class RouteCaptureReceipt:
    """Coordinates a wrapped production route actually received, by snapshot."""

    route_name: str
    call_count: int
    coordinate_snapshots: tuple[str, ...]
    fixture_calls: tuple[RouteFixtureReceipt, ...] = ()

    def validate(self) -> None:
        if not self.route_name:
            raise AssertionError("route capture has no route name")
        if self.call_count < 1:
            raise AssertionError("route capture observed no production calls")
        if not self.coordinate_snapshots or len(set(self.coordinate_snapshots)) != len(self.coordinate_snapshots):
            raise AssertionError("route capture requires unique realized coordinate snapshots")
        if self.fixture_calls:
            if len(self.fixture_calls) != self.call_count:
                raise AssertionError("fixture observations do not cover every route call")
            for call in self.fixture_calls:
                call.validate()
            if set(self.coordinate_snapshots) != {s for call in self.fixture_calls for s in call.coordinate_snapshots}:
                raise AssertionError("fixture observations differ from route coordinates")


class ProductionRouteCapture:
    """Wrap a production callable and hash the electron coordinates it receives.

    The hash happens inside the call boundary, so whatever batch the production
    route computed on is what the receipt records; a test cannot generate one
    batch, feed another, and still assemble a valid ``CaseReceipt``. Only a
    normally returned call contributes coverage; caught exceptions do not.
    """

    def __init__(self, route_name: str, extract_coordinates, *, extract_hamiltonian=None) -> None:
        self._route_name = route_name
        self._extract = extract_coordinates
        self._call_count = 0
        self._snapshots: list[str] = []
        self._extract_hamiltonian = extract_hamiltonian
        self._fixture_calls: list[RouteFixtureReceipt] = []

    def wrap(self, function):
        def wrapped(*args, **kwargs):
            snapshots = []
            fixture = observe_hamiltonian(self._extract_hamiltonian(args, kwargs)) if self._extract_hamiltonian else None
            for r_up, r_dn in self._extract(args, kwargs):
                up = np.asarray(r_up)
                down = np.asarray(r_dn)
                if up.ndim == 2:
                    up = up[np.newaxis, ...]
                if down.ndim == 2:
                    down = down[np.newaxis, ...]
                if up.ndim != 3 or down.ndim != 3 or up.shape[0] != down.shape[0] or up.shape[-1] != 3 or down.shape[-1] != 3:
                    raise AssertionError("route received invalid electron batch shapes")
                if fixture is not None and (up.shape[1], down.shape[1]) != fixture.electron_counts:
                    raise AssertionError("route coordinates disagree with Hamiltonian electron counts")
                for index in range(up.shape[0]):
                    snapshot = _coordinate_snapshot(up[index], down[index])
                    if snapshot not in snapshots:
                        snapshots.append(snapshot)
            result = function(*args, **kwargs)
            self._call_count += 1
            if fixture is not None:
                self._fixture_calls.append(RouteFixtureReceipt(fixture, tuple(snapshots)))
            for snapshot in snapshots:
                if snapshot not in self._snapshots:
                    self._snapshots.append(snapshot)
            return result

        return wrapped

    def receipt(self) -> RouteCaptureReceipt:
        receipt = RouteCaptureReceipt(
            route_name=self._route_name,
            call_count=self._call_count,
            coordinate_snapshots=tuple(self._snapshots),
            fixture_calls=tuple(self._fixture_calls),
        )
        receipt.validate()
        return receipt


def _coordinate_snapshot(r_up: np.ndarray, r_dn: np.ndarray) -> str:
    digest = hashlib.sha256()
    for label, value in ((b"up", r_up), (b"down", r_dn)):
        array = np.ascontiguousarray(value)
        digest.update(label)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def observe_electron_configurations(
    binding: ElectronConfigurationBinding,
    r_up_batch: Any,
    r_dn_batch: Any,
    *,
    cond_g_per_configuration: tuple[float, ...] = (),
    sigma_min_g_per_configuration: tuple[float, ...] = (),
    sigma_max_g_per_configuration: tuple[float, ...] = (),
    machine_epsilon_per_configuration: tuple[float, ...] = (),
    cond_times_epsilon_per_configuration: tuple[float, ...] = (),
    generation_attempts: int | None = None,
    rejected_configuration_count: int = 0,
) -> ElectronConfigurationReceipt:
    """Observe and validate a walker-leading batch against its declared binding."""
    binding.validate()
    up = np.asarray(r_up_batch)
    down = np.asarray(r_dn_batch)
    if up.ndim == 2:
        up = up[np.newaxis, ...]
    if down.ndim == 2:
        down = down[np.newaxis, ...]
    if up.ndim != 3 or down.ndim != 3 or up.shape[0] != down.shape[0] or up.shape[-1] != 3 or down.shape[-1] != 3:
        raise AssertionError(f"invalid electron batch shapes: up={up.shape}, down={down.shape}")
    return observe_electron_configuration_sequence(
        binding,
        tuple((up[index], down[index]) for index in range(up.shape[0])),
        cond_g_per_configuration=cond_g_per_configuration,
        sigma_min_g_per_configuration=sigma_min_g_per_configuration,
        sigma_max_g_per_configuration=sigma_max_g_per_configuration,
        machine_epsilon_per_configuration=machine_epsilon_per_configuration,
        cond_times_epsilon_per_configuration=cond_times_epsilon_per_configuration,
        generation_attempts=generation_attempts,
        rejected_configuration_count=rejected_configuration_count,
    )


def observe_electron_configuration_sequence(
    binding: ElectronConfigurationBinding,
    configurations: Iterable[tuple[Any, Any]],
    *,
    cond_g_per_configuration: tuple[float, ...] = (),
    sigma_min_g_per_configuration: tuple[float, ...] = (),
    sigma_max_g_per_configuration: tuple[float, ...] = (),
    machine_epsilon_per_configuration: tuple[float, ...] = (),
    cond_times_epsilon_per_configuration: tuple[float, ...] = (),
    generation_attempts: int | None = None,
    rejected_configuration_count: int = 0,
    evaluation_kappa_per_configuration: tuple[float, ...] = (),
) -> ElectronConfigurationReceipt:
    """Observe configurations that may have different spin-resolved electron counts."""
    binding.validate()
    arrays = tuple((np.asarray(r_up), np.asarray(r_dn)) for r_up, r_dn in configurations)
    if any(r_up.ndim != 2 or r_dn.ndim != 2 or r_up.shape[-1] != 3 or r_dn.shape[-1] != 3 for r_up, r_dn in arrays):
        raise AssertionError("electron configuration sequence contains an invalid coordinate array")
    snapshots = tuple(_coordinate_snapshot(r_up, r_dn) for r_up, r_dn in arrays)
    counts = tuple((r_up.shape[0], r_dn.shape[0]) for r_up, r_dn in arrays)
    resolved_generation_attempts = len(arrays) if generation_attempts is None else generation_attempts
    receipt = ElectronConfigurationReceipt(
        binding_id=binding.binding_id,
        mode=binding.mode,
        generation_pattern=binding.generation_pattern,
        generator_operation=binding.generator_operation,
        canonical_generator_arguments=tuple(
            (name, canonical_snapshot(value)) for name, value in sorted(binding.generator_arguments)
        ),
        requested_configuration_count=binding.configuration_count,
        realized_configuration_count=len(arrays),
        ordered_coordinate_snapshots=snapshots,
        electron_counts_per_configuration=counts,
        finite_coordinate_check=all(np.all(np.isfinite(r_up)) and np.all(np.isfinite(r_dn)) for r_up, r_dn in arrays),
        geminal_conditioning_mode=binding.geminal_conditioning_mode,
        configured_minimum_cond_g=binding.minimum_cond_g,
        configured_maximum_cond_g=binding.maximum_cond_g,
        lower_bound_comparison=binding.lower_bound_comparison,
        upper_bound_comparison=binding.upper_bound_comparison,
        configured_minimum_singular_value=binding.minimum_singular_value,
        cond_g_per_configuration=cond_g_per_configuration,
        sigma_min_g_per_configuration=sigma_min_g_per_configuration,
        sigma_max_g_per_configuration=sigma_max_g_per_configuration,
        machine_epsilon_per_configuration=machine_epsilon_per_configuration,
        cond_times_epsilon_per_configuration=cond_times_epsilon_per_configuration,
        generation_attempts=resolved_generation_attempts,
        rejected_configuration_count=rejected_configuration_count,
        configured_maximum_generation_attempts=binding.maximum_generation_attempts,
        declared_production_route=binding.production_route_capture,
        conditioning_subject=binding.conditioning_subject,
        configured_maximum_evaluation_kappa=binding.maximum_evaluation_kappa,
        evaluation_kappa_comparison=binding.evaluation_kappa_comparison,
        evaluation_kappa_per_configuration=tuple(evaluation_kappa_per_configuration),
    )
    receipt.validate()
    return receipt


def observe_no_electron_configuration(binding: ElectronConfigurationBinding) -> ElectronConfigurationReceipt:
    """Create the explicit receipt for an operation that consumes no electron positions."""
    binding.validate()
    receipt = ElectronConfigurationReceipt(
        binding_id=binding.binding_id,
        mode=binding.mode,
        generation_pattern=None,
        generator_operation=None,
        canonical_generator_arguments=(),
        requested_configuration_count=0,
        realized_configuration_count=0,
        ordered_coordinate_snapshots=(),
        electron_counts_per_configuration=(),
        finite_coordinate_check=True,
        geminal_conditioning_mode=None,
        configured_minimum_cond_g=None,
        configured_maximum_cond_g=None,
        lower_bound_comparison=None,
        upper_bound_comparison=None,
        configured_minimum_singular_value=None,
        cond_g_per_configuration=(),
        sigma_min_g_per_configuration=(),
        sigma_max_g_per_configuration=(),
        machine_epsilon_per_configuration=(),
        cond_times_epsilon_per_configuration=(),
        generation_attempts=0,
        rejected_configuration_count=0,
        configured_maximum_generation_attempts=None,
    )
    receipt.validate()
    return receipt


def observe_allclose(
    binding: ComparisonBinding,
    compared_arrays: Iterable[tuple[Any, Any]],
    *,
    atol: float,
    rtol: float,
    electron_receipt: ElectronConfigurationReceipt | None = None,
) -> ComparisonReceipt:
    """Create the numerical receipt for arrays already compared by pytest."""
    binding.validate()
    if binding.policy_id not in ComparisonBinding._ALLCLOSE_POLICIES:
        raise AssertionError(f"policy {binding.policy_id} is not a tolerance comparison")
    _validate_tolerance(atol)
    _validate_tolerance(rtol)
    max_absolute = 0.0
    max_relative = 0.0
    failures = 0
    count = 0
    elements = 0
    for actual_value, reference_value in compared_arrays:
        actual = np.asarray(actual_value)
        reference = np.asarray(reference_value)
        if actual.shape != reference.shape:
            raise AssertionError(f"comparison receipt shape mismatch: {actual.shape} != {reference.shape}")
        difference = np.abs(actual - reference)
        allowed = atol + rtol * np.abs(reference)
        failures += int(np.count_nonzero(~np.isfinite(difference) | (difference > allowed)))
        if difference.size:
            max_absolute = max(max_absolute, float(np.max(difference)))
            denominator = np.abs(reference)
            relative = np.divide(
                difference,
                denominator,
                out=np.full_like(difference, np.inf, dtype=np.float64),
                where=denominator != 0,
            )
            relative[(denominator == 0) & (difference == 0)] = 0.0
            max_relative = max(max_relative, float(np.max(relative)))
        count += 1
        elements += actual.size
    receipt = ComparisonReceipt(
        binding_id=binding.binding_id,
        policy_id=binding.policy_id,
        resolved_atol=float(atol),
        resolved_rtol=float(rtol),
        max_absolute_error=max_absolute,
        max_relative_error=max_relative,
        failing_element_count=failures,
        compared_array_count=count,
        compared_element_count=elements,
        electron_sequence_token=None if electron_receipt is None else electron_receipt.sequence_token(),
    )
    receipt.validate()
    return receipt


def observe_exact(
    binding: ComparisonBinding,
    compared_arrays: Iterable[tuple[Any, Any]],
    *,
    electron_receipt: ElectronConfigurationReceipt | None = None,
) -> ExactComparisonReceipt:
    """Evaluate an exact-value or raw-bit policy and emit its receipt."""
    binding.validate()
    if binding.policy_id not in ComparisonBinding._EXACT_POLICIES:
        raise AssertionError(f"policy {binding.policy_id} is not exact")
    differing = 0
    count = 0
    elements = 0
    for actual_value, reference_value in compared_arrays:
        actual = np.asarray(actual_value)
        reference = np.asarray(reference_value)
        if actual.shape != reference.shape or actual.dtype != reference.dtype:
            raise AssertionError("exact comparison requires identical shape and dtype")
        if binding.policy_id == "CMP-RAW-BITS-001":
            differing += int(np.count_nonzero(actual.view(np.uint8) != reference.view(np.uint8)))
        else:
            if np.any(np.isnan(actual)) or np.any(np.isnan(reference)):
                raise AssertionError("exact-value comparison forbids NaN")
            differing += int(np.count_nonzero(actual != reference))
        count += 1
        elements += actual.size
    receipt = ExactComparisonReceipt(
        binding_id=binding.binding_id,
        policy_id=binding.policy_id,
        differing_element_count=differing,
        compared_array_count=count,
        compared_element_count=elements,
        electron_sequence_token=None if electron_receipt is None else electron_receipt.sequence_token(),
    )
    receipt.validate()
    return receipt


def observe_non_numerical(
    binding: ObservationBinding,
    observations: dict[str, bool],
    *,
    electron_receipt: ElectronConfigurationReceipt | None = None,
) -> ObservationReceipt:
    """Evaluate liveness, smoke, expected-error, or reviewed-N/A observations."""
    binding.validate()
    missing = set(binding.required_observations) - set(observations)
    extra = set(observations) - set(binding.required_observations)
    failed = sorted(name for name, passed in observations.items() if not passed)
    if missing or extra or failed:
        raise AssertionError(f"observation mismatch: missing={sorted(missing)}, extra={sorted(extra)}, failed={failed}")
    receipt = ObservationReceipt(
        binding_id=binding.binding_id,
        policy_id=binding.policy_id,
        completed_observations=tuple(sorted(observations)),
        status="passed",
        electron_sequence_token=None if electron_receipt is None else electron_receipt.sequence_token(),
    )
    receipt.validate()
    return receipt
