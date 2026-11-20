"""Runtime GDD receipts for the reviewed finite-difference/autodiff sentinels."""

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
from typing import Any

import numpy as np

from gdd.engine.evidence import (
    ComparisonBinding,
    ObservationBinding,
    emit_completion_record,
    observe_allclose,
    observe_non_numerical,
)
from gdd.engine.fd_comparison_contract import _identity, load_fd_contract, validate_fd_call, validate_fd_receipt
from jqmc._precision import is_mixed_precision_enabled
from tests.test_jqmc_kernel._gdd_adapters import (
    case_receipt,
    committed_input,
    parameter_input,
    seeded_single_electron_receipt,
)


# Reviewed FD evidence id -> the case's selector values, computed from the test's
# own pytest callspec values (the keyword names are the callspec parameter names).
# The test call sites and the derivative registry both go through
# ``fd_case_parameters``, so a selector exists in exactly one place.
_FD_CASE_PARAMETERS = {
    "fd-ad-ao-grad-molecular": lambda ao_kind: {"ao_kind": ao_kind, "boundary_condition": "molecular", "seed": 2025},
    "fd-ad-ao-grad-periodic": lambda ao_kind: {"ao_kind": ao_kind, "boundary_condition": "periodic", "seed": 2025},
    "fd-ad-ao-laplacian-molecular": lambda ao_kind: {"ao_kind": ao_kind, "boundary_condition": "molecular", "seed": 2025},
    "fd-ad-ao-laplacian-periodic": lambda ao_kind: {"ao_kind": ao_kind, "boundary_condition": "periodic", "seed": 2025},
    "fd-ad-mo-grad-periodic": lambda kind: {"mo_kind": kind, "boundary_condition": "periodic", "seed": 303},
    "fd-ad-mo-laplacian-periodic": lambda kind: {"mo_kind": kind, "boundary_condition": "periodic", "seed": 505},
    "fd-ad-dtn-grad-laplacian-molecular": lambda C, preset: {
        "preset": preset,
        "cutoff_smoothness": C,
        "boundary_condition": "molecular",
    },
    "fd-ad-dtn-grad-laplacian-periodic": lambda C, preset: {
        "preset": preset,
        "cutoff_smoothness": C,
        "boundary_condition": "periodic",
    },
    "fd-ad-dtn-ion-position": lambda pbc: {"boundary_condition": ("periodic" if pbc else "molecular"), "preset": "u_chi_f"},
    "fd-ad-j3-exponents": lambda trexio_file: {"trexio_file": trexio_file, "jastrow_seed": 42},
    "fd-ad-j3-coefficients": lambda trexio_file: {"trexio_file": trexio_file, "jastrow_seed": 42},
    "fd-ad-geminal-exponents": lambda: {"trexio_file": "H2_ae_ccpvdz_cart.h5"},
    "fd-ad-mcmc-stress": lambda: {"system": "silicon", "walker_seed": 2026},
    "fd-ad-qe-periodic-stress": lambda kind: {"ao_kind": kind, "walker_seed": 2026},
    "fd-ad-qe-periodic-force": lambda kind: {"ao_kind": kind, "walker_seed": 2026},
}


def fd_case_parameters(evidence_id: str, **callspec: Any) -> dict[str, Any]:
    """The FD selector values of one case, from its pytest callspec values."""
    return _FD_CASE_PARAMETERS[evidence_id](**callspec)


def fd_obligation_id(evidence_id: str, fixture_parameters: dict[str, Any]) -> str:
    """The FD obligation id, with the comparison contract's own identity function."""
    return _identity(evidence_id, fixture_parameters)


@cache
def _evidence_id_by_node() -> dict[str, str]:
    by_node: dict[str, str] = {}
    for case in load_fd_contract().values():
        if by_node.setdefault(case["node"], case["evidence_id"]) != case["evidence_id"]:
            raise ValueError(f"{case['node']}: two FD evidence ids")
    return by_node


def fd_keys(definition: str, params: dict[str, Any], precision: str) -> tuple[list[str], list[str]]:
    """Derivative-registry key function: the one FD obligation this case emits."""
    evidence_id = _evidence_id_by_node()[definition]
    return [fd_obligation_id(evidence_id, fd_case_parameters(evidence_id, **params))], []


def record_fd_case(
    evidence_id: str,
    *,
    pytest_node_id: str,
    fixture_parameters: dict[str, Any],
    coordinates: tuple[Any, Any],
    coordinate_generator: str,
    coordinate_seed: int,
    compared_pairs: tuple[tuple[Any, Any], ...],
    atol: float,
    rtol: float,
    zones: tuple[str, ...],
    level: str | None,
    stencil: str,
    step: float,
    direction: str,
    source_ids: tuple[str, ...] = (),
    artifact_format: str = "trexio",
    transformations: tuple[str, ...] = (),
    comparison_policy_id: str | None = None,
    additional_comparisons: tuple[tuple[tuple[tuple[Any, Any], ...], float, float, tuple[str, ...], str], ...] = (),
) -> None:
    """Validate one concrete FD/AD case using values already asserted by pytest."""
    policy_id = comparison_policy_id or f"CMP-ZONE-ALLCLOSE-{level.upper()}-001"
    comparisons = [(compared_pairs, atol, rtol, zones, level, policy_id)]
    comparisons.extend(
        (pairs, extra_atol, extra_rtol, extra_zones, extra_level, f"CMP-ZONE-ALLCLOSE-{extra_level.upper()}-001")
        for pairs, extra_atol, extra_rtol, extra_zones, extra_level in additional_comparisons
    )
    for pairs, *_ in comparisons:
        if any(np.asarray(value).size == 0 for pair in pairs for value in pair):
            raise AssertionError("FD comparison contract: empty array in a required comparison")
    validate_fd_call(
        evidence_id,
        pytest_node_id,
        fixture_parameters,
        "mixed" if is_mixed_precision_enabled() else "full",
        [
            {
                "policy": policy,
                "zones": list(pair_zones),
                "level": pair_level,
                "atol": pair_atol,
                "rtol": pair_rtol,
                "compared_array_count": len(pairs),
            }
            for pairs, pair_atol, pair_rtol, pair_zones, pair_level, policy in comparisons
        ],
    )
    obligation_id = fd_obligation_id(evidence_id, fixture_parameters)
    input_binding = (
        committed_input(
            obligation_id,
            source_ids=source_ids,
            builder_operation=pytest_node_id,
            transformations=transformations,
            artifact_format=artifact_format,
        )
        if source_ids
        else parameter_input(
            obligation_id,
            parameter_set_id=f"FD-PARAMS::{evidence_id}",
            builder_operation=pytest_node_id,
            declaration=fixture_parameters,
            transformations=transformations,
        )
    )
    r_up, r_dn = coordinates
    electrons = seeded_single_electron_receipt(
        obligation_id,
        r_up,
        r_dn,
        generator_operation=coordinate_generator,
        seed=coordinate_seed,
        generator_arguments=tuple(sorted(fixture_parameters.items())),
        generation_pattern="direct",
        geminal_conditioning_mode=("not-limited" if source_ids else "not-applicable"),
        conditioning_reason=(None if source_ids else "component fixture has no geminal matrix G"),
    )
    comparison = observe_allclose(
        ComparisonBinding(
            binding_id=f"COMPARISON::{obligation_id}",
            policy_id=policy_id,
            precision_mode="full",
            actual_role="xla-autodiff-derivative",
            reference_role="independent-finite-difference-derivative",
            zones=(() if comparison_policy_id else zones),
            level=(None if comparison_policy_id else level),
        ),
        compared_pairs,
        atol=atol,
        rtol=rtol,
    )
    comparison_receipts = []
    all_pairs = list(compared_pairs)
    for index, (pairs, extra_atol, extra_rtol, extra_zones, extra_level) in enumerate(additional_comparisons):
        comparison_receipts.append(
            observe_allclose(
                ComparisonBinding(
                    binding_id=f"COMPARISON::{obligation_id}::additional-{index}",
                    policy_id=f"CMP-ZONE-ALLCLOSE-{extra_level.upper()}-001",
                    precision_mode="full",
                    actual_role="xla-autodiff-derivative",
                    reference_role="independent-finite-difference-derivative",
                    zones=extra_zones,
                    level=extra_level,
                ),
                pairs,
                atol=extra_atol,
                rtol=extra_rtol,
            )
        )
        all_pairs.extend(pairs)
    arrays = [np.asarray(value) for pair in all_pairs for value in pair]
    route = observe_non_numerical(
        ObservationBinding(
            binding_id=f"FD-WITNESS::{obligation_id}",
            policy_id="OBS-FD-ROUTE-001",
            precision_mode="full",
            subject_role="independent-finite-difference-stencil",
            required_observations=(
                "perturbation-is-nonzero",
                "plus-and-minus-points-are-distinct",
                "autodiff-route-is-live",
                "finite-difference-result-is-finite",
                "stencil-declared",
            ),
        ),
        {
            "perturbation-is-nonzero": bool(np.isfinite(step) and step > 0.0),
            "plus-and-minus-points-are-distinct": bool(step > 0.0 and direction),
            "autodiff-route-is-live": bool(all_pairs),
            "finite-difference-result-is-finite": bool(arrays and all(np.all(np.isfinite(a)) for a in arrays)),
            "stencil-declared": bool(stencil),
        },
    )
    receipt = case_receipt(
        guarantee_id="NUM-DERIV-FD-AUTODIFF-001",
        obligation_id=obligation_id,
        pytest_node_id=pytest_node_id,
        input_binding=input_binding,
        electron_configuration=electrons,
        primary_check=comparison,
        additional_checks=tuple(comparison_receipts) + (route,),
        emit_completion=False,
    )
    validate_fd_receipt(receipt)
    emit_completion_record(receipt)
