"""GDD runtime fragments emitted by Rust/XLA analytical parity tests."""

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

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from gdd.engine.evidence import (
    CaseReceipt,
    ComparisonBinding,
    ElectronConfigurationBinding,
    observe_allclose,
    observe_electron_configurations,
    ObservationBinding,
    observe_non_numerical,
    observe_exact,
    canonical_snapshot,
    emit_completion_record,
)
from tests.test_jqmc_kernel._generated_matrix_base import matrix_case_binding
from tests.test_jqmc_kernel._gdd_adapters import current_precision_mode


@dataclass(frozen=True)
class RustCaseFragment:
    """One validated Rust fragment belonging to an upstream XLA obligation."""

    upstream_obligation_id: str
    component_id: str
    receipt: CaseReceipt

    def validate(self) -> None:
        if not self.upstream_obligation_id or not self.component_id:
            raise AssertionError("Rust fragment identifiers are incomplete")
        self.receipt.validate()


RUNTIME_FRAGMENTS: list[RustCaseFragment] = []


def mirror_upstream_receipt(receipt: CaseReceipt, *, component_id: str, pytest_node_id: str) -> RustCaseFragment:
    """Reclassify an upstream receipt whose analytical route was replaced by Rust."""
    comparison = receipt.comparison
    if hasattr(comparison, "resolved_atol"):
        comparison = replace(
            comparison,
            binding_id=f"RUST-{comparison.binding_id}",
            policy_id="CMP-XLA-RUST-MEDIUM-001",
        )
    mirrored = replace(
        receipt,
        guarantee_id="IMPL-DERIV-RUST-XLA-001",
        obligation_id=f"{receipt.obligation_id}::rust-component={component_id}",
        pytest_node_id=pytest_node_id,
        comparison=comparison,
    )
    fragment = RustCaseFragment(receipt.obligation_id, component_id, mirrored)
    fragment.validate()
    emit_completion_record(
        mirrored,
        record_kind="rust-fragment",
        upstream_obligation_id=receipt.obligation_id,
        component_id=component_id,
    )
    RUNTIME_FRAGMENTS.append(fragment)
    return fragment


def record_rust_bound_case(
    *,
    upstream_obligation_id: str,
    component_id: str,
    pytest_node_id: str,
    input_binding: Any,
    r_up: Any,
    r_dn: Any,
    generator_operation: str,
    seed: int | None,
    compared_pairs: tuple[tuple[Any, Any], ...] = (),
    zones: tuple[str, ...] = (),
    level: str = "medium",
    atol: float | None = None,
    rtol: float | None = None,
    observations: tuple[str, ...] = (),
    exact_pairs: tuple[tuple[Any, Any], ...] = (),
) -> RustCaseFragment:
    """Validate a targeted Rust case with a pre-resolved upstream input binding."""
    if seed is None:
        position_payload = {"r_up": np.asarray(r_up).tolist(), "r_dn": np.asarray(r_dn).tolist()}
        electron_binding = ElectronConfigurationBinding(
            binding_id=f"RUST-ELECTRONS::{upstream_obligation_id}::{component_id}",
            mode="required",
            generation_pattern="direct",
            position_source_kind="explicit",
            position_set_id=f"RUST-POSITIONS::{upstream_obligation_id}::{component_id}",
            coordinates_source=generator_operation,
            content_snapshot=canonical_snapshot(position_payload),
            configuration_count=1,
            geminal_conditioning_mode="not-limited",
        )
    else:
        electron_binding = ElectronConfigurationBinding(
            binding_id=f"RUST-ELECTRONS::{upstream_obligation_id}::{component_id}",
            mode="required",
            generation_pattern="direct",
            position_source_kind="seeded-random",
            generator_operation=generator_operation,
            generator_arguments=(),
            random_algorithm="numpy.random.PCG64",
            seed=seed,
            configuration_count=1,
            geminal_conditioning_mode="not-limited",
        )
    electrons = observe_electron_configurations(
        electron_binding,
        r_up,
        r_dn,
    )
    if compared_pairs:
        if atol is None or rtol is None:
            raise AssertionError("Rust numerical fragment has no resolved tolerance")
        check = observe_allclose(
            ComparisonBinding(
                binding_id=f"RUST-COMPARISON::{upstream_obligation_id}::{component_id}",
                policy_id=f"CMP-XLA-RUST-{level.upper()}-001",
                precision_mode=current_precision_mode(),
                actual_role="rust-analytical-derivative",
                reference_role="xla-reference-derivative",
                zones=zones,
                level=level,
            ),
            compared_pairs,
            atol=atol,
            rtol=rtol,
        )
    elif exact_pairs:
        check = observe_exact(
            ComparisonBinding(
                binding_id=f"RUST-EXACT::{upstream_obligation_id}::{component_id}",
                policy_id="CMP-EXACT-VALUES-001",
                precision_mode=current_precision_mode(),
                actual_role="rust-analytical-derivative",
                reference_role="rust-control-derivative",
            ),
            exact_pairs,
        )
    else:
        check = observe_non_numerical(
            ObservationBinding(
                binding_id=f"RUST-OBSERVATION::{upstream_obligation_id}::{component_id}",
                policy_id="OBS-ROUTE-LIVENESS-001",
                precision_mode=current_precision_mode(),
                subject_role="rust-analytical-disposition",
                required_observations=observations,
            ),
            dict.fromkeys(observations, True),
        )
    receipt = CaseReceipt(
        guarantee_id="IMPL-DERIV-RUST-XLA-001",
        obligation_id=f"{upstream_obligation_id}::rust-component={component_id}",
        pytest_node_id=pytest_node_id,
        input_binding=input_binding,
        electron_configuration=electrons,
        comparison=check,
        status="passed",
    )
    fragment = RustCaseFragment(upstream_obligation_id, component_id, receipt)
    fragment.validate()
    emit_completion_record(
        receipt,
        record_kind="rust-fragment",
        upstream_obligation_id=upstream_obligation_id,
        component_id=component_id,
    )
    RUNTIME_FRAGMENTS.append(fragment)
    return fragment


def record_rust_matrix_fragment(
    *,
    cell: tuple[str, ...],
    quantity: str,
    axis: str,
    component_id: str,
    pytest_node_id: str,
    ham: Any,
    r_up: Any,
    r_dn: Any,
    compared_pairs: tuple[tuple[Any, Any], ...],
    zones: tuple[str, ...],
    level: str,
    atol: float,
    rtol: float,
    generation_attempts: int,
) -> RustCaseFragment:
    """Validate one generated-matrix Rust/XLA fragment using its actual values."""
    binding = matrix_case_binding(cell, quantity, axis)
    gd = ham.wavefunction_data.geminal_data
    from jqmc.determinant import compute_geminal_all_elements

    cond_g = float(np.linalg.cond(np.asarray(compute_geminal_all_elements(gd, r_up, r_dn))))
    electron_binding = ElectronConfigurationBinding(
        binding_id=f"RUST-ELECTRONS::{binding.obligation_id}::{component_id}",
        mode="required",
        generation_pattern="rejection-screen",
        position_source_kind="seeded-random",
        generator_operation="tests.test_rs_kernel._rs_fixtures.draw_well_conditioned_cfg",
        generator_arguments=(("scale", 1.0),),
        random_algorithm="numpy.random.PCG64",
        seed=1,
        configuration_count=1,
        geminal_conditioning_mode="upper-bounded",
        maximum_cond_g=700.0,
        upper_bound_comparison="less-than",
        conditioning_violation_action="deterministic-resample",
        maximum_generation_attempts=300,
    )
    electrons = observe_electron_configurations(
        electron_binding,
        r_up,
        r_dn,
        cond_g_per_configuration=(cond_g,),
        generation_attempts=generation_attempts,
        rejected_configuration_count=generation_attempts - 1,
    )
    comparison = observe_allclose(
        ComparisonBinding(
            binding_id=f"RUST-COMPARISON::{binding.obligation_id}::{component_id}",
            policy_id=f"CMP-XLA-RUST-{level.upper()}-001",
            precision_mode=current_precision_mode(),
            actual_role="rust-analytical-derivative",
            reference_role="xla-analytical-derivative",
            zones=zones,
            level=level,
        ),
        compared_pairs,
        atol=atol,
        rtol=rtol,
    )
    receipt = CaseReceipt(
        guarantee_id="IMPL-DERIV-RUST-XLA-001",
        obligation_id=f"{binding.obligation_id}::rust-component={component_id}",
        pytest_node_id=pytest_node_id,
        input_binding=binding.input_binding,
        electron_configuration=electrons,
        comparison=comparison,
        status="passed",
    )
    fragment = RustCaseFragment(binding.obligation_id, component_id, receipt)
    fragment.validate()
    emit_completion_record(
        receipt,
        record_kind="rust-fragment",
        upstream_obligation_id=binding.obligation_id,
        component_id=component_id,
    )
    RUNTIME_FRAGMENTS.append(fragment)
    return fragment
