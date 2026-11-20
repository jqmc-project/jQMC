"""Thin adapters that bind existing derivative pytest evidence to GDD receipts."""

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

from gdd.engine.evidence import (
    CaseReceipt,
    ElectronConfigurationBinding,
    InputBinding,
    canonical_snapshot,
    emit_completion_record,
    observe_electron_configurations,
)
from jqmc._precision import is_mixed_precision_enabled


_GUARANTEE_ID_BY_MANIFEST_FAMILY = {
    "ln_Psi": "MATH-DERIV-LOG-WAVEFUNCTION-001",
    "e_L": "MATH-DERIV-LOCAL-ENERGY-001",
    "e_L_t": "MATH-DERIV-CONVENTIONAL-ALGORITHM-001",
    "e_L_n": "MATH-DERIV-LOAD-BALANCED-ALGORITHM-001",
}


def current_precision_mode() -> str:
    """Return the precision lane realized by the current pytest case."""
    return "mixed" if is_mixed_precision_enabled() else "full"


def guarantee_id_for_manifest_family(manifest_family: str) -> str:
    """Translate a legacy coverage-manifest code to a human-readable GDD ID."""
    try:
        return _GUARANTEE_ID_BY_MANIFEST_FAMILY[manifest_family]
    except KeyError as exc:
        raise AssertionError(f"unknown derivative manifest family: {manifest_family}") from exc


def committed_input(
    obligation_id: str,
    *,
    source_ids: tuple[str, ...],
    builder_operation: str,
    transformations: tuple[str, ...] = (),
    artifact_format: str = "trexio",
) -> InputBinding:
    binding = InputBinding(
        binding_id=f"INPUT::{obligation_id}",
        kind="committed-artifact",
        builder_operation=builder_operation,
        transformations=transformations,
        source_ids=source_ids,
        artifact_format=artifact_format,
    )
    binding.validate()
    return binding


def parameter_input(
    obligation_id: str,
    *,
    parameter_set_id: str,
    builder_operation: str,
    declaration: Any,
    transformations: tuple[str, ...] = (),
) -> InputBinding:
    binding = InputBinding(
        binding_id=f"INPUT::{obligation_id}",
        kind="parameter-set",
        builder_operation=builder_operation,
        transformations=transformations,
        parameter_set_id=parameter_set_id,
        parameter_source="registered-python-operation",
        content_snapshot=canonical_snapshot(declaration),
    )
    binding.validate()
    return binding


def seeded_single_electron_receipt(
    obligation_id: str,
    r_up: Any,
    r_dn: Any,
    *,
    generator_operation: str,
    seed: int,
    generator_arguments: tuple[tuple[str, Any], ...],
    generation_pattern: str,
    geminal_conditioning_mode: str = "not-limited",
    conditioning_reason: str | None = None,
    minimum_cond_g: float | None = None,
    maximum_cond_g: float | None = None,
    lower_bound_comparison: str | None = None,
    upper_bound_comparison: str | None = None,
    minimum_singular_value: float | None = None,
    conditioning_violation_action: str | None = None,
    maximum_generation_attempts: int | None = None,
    cond_g_per_configuration: tuple[float, ...] = (),
    sigma_min_g_per_configuration: tuple[float, ...] = (),
    generation_attempts: int = 1,
    rejected_configuration_count: int = 0,
):
    binding = ElectronConfigurationBinding(
        binding_id=f"ELECTRONS::{obligation_id}",
        mode="required",
        generation_pattern=generation_pattern,
        position_source_kind="seeded-random",
        generator_operation=generator_operation,
        generator_arguments=generator_arguments,
        random_algorithm="numpy.random.PCG64",
        seed=seed,
        configuration_count=1,
        geminal_conditioning_mode=geminal_conditioning_mode,
        conditioning_reason=conditioning_reason,
        minimum_cond_g=minimum_cond_g,
        maximum_cond_g=maximum_cond_g,
        lower_bound_comparison=lower_bound_comparison,
        upper_bound_comparison=upper_bound_comparison,
        minimum_singular_value=minimum_singular_value,
        conditioning_violation_action=conditioning_violation_action,
        maximum_generation_attempts=maximum_generation_attempts,
    )
    return observe_electron_configurations(
        binding,
        r_up,
        r_dn,
        cond_g_per_configuration=cond_g_per_configuration,
        sigma_min_g_per_configuration=sigma_min_g_per_configuration,
        generation_attempts=generation_attempts,
        rejected_configuration_count=rejected_configuration_count,
    )


def case_receipt(
    *,
    guarantee_id: str,
    obligation_id: str,
    pytest_node_id: str,
    input_binding: InputBinding,
    electron_configuration,
    primary_check,
    additional_checks: tuple = (),
    emit_completion: bool = True,
) -> CaseReceipt:
    receipt = CaseReceipt(
        guarantee_id=guarantee_id,
        obligation_id=obligation_id,
        pytest_node_id=pytest_node_id,
        input_binding=input_binding,
        electron_configuration=electron_configuration,
        comparison=primary_check,
        status="passed",
        additional_checks=additional_checks,
    )
    receipt.validate()
    if emit_completion:
        emit_completion_record(receipt)
    return receipt
