"""Runtime completion gate for the derivative GDD contract."""

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

import json
import re
from dataclasses import replace
from functools import cache
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests._gdd_ssot_layout import write_ssot_text

from gdd.engine.guarantee_store import load_module, load_ssot_document

from gdd.engine.evidence import (
    ComparisonBinding,
    ElectronConfigurationBinding,
    InputBinding,
    observe_electron_configurations,
    observe_exact,
)
from gdd.engine.ledger import load_ledger
from gdd.engine.production_entry_discovery import (
    discover_direct_autodiff_calls,
    discover_external_derivative_references,
)
from gdd.engine.derivative_closure import validate_record
from tests.test_jqmc_kernel.analytic_coverage_registration import (
    REQUIRED_CASES,
    fd_binding_issues,
    generated_matrix_binding_issues,
    gfmc_matrix_binding_issues,
    mixed_binding_issues,
    regime_binding_issues,
    registration_diff,
    requirements_by_id,
    targeted_binding_issues,
)
from tests.test_rs_kernel.analytic_coverage_manifest import (
    required_rust_case_ids,
    rust_required_registration_diff,
    rust_gdd_binding_issues,
)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir, pytest.mark.design_pin]

_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT = load_ledger("derivatives.yaml")


@pytest.mark.parametrize(
    "mutation", ["missing-declarations", "invalid-row-key", "invalid-disposition", "unknown-reason", "duplicate-key"]
)
def test_derivative_inactivity_rejects_mutated_declarations(tmp_path, mutation):
    """A permissive inactivity loader would suppress the nonvacuity screens; malformed input fails.

    One case per route to a permissive mapping: dropped declarations, an unknown
    row key (``allow_live``), an unknown disposition (``ignore``), an unreviewed
    reason, and a duplicate YAML key that would overwrite a declaration.
    """
    from gdd.engine.derivative_inactivity import load_derivative_inactivity

    source_path = _ROOT / "gdd/ssot/fixtures/derivative_inactivity.yaml"
    candidate = tmp_path / "derivative_inactivity.yaml"
    if mutation == "duplicate-key":
        # Repeat the first block key with its identical value: even a duplicate
        # that does not alter the parsed fingerprint must fail strict loading.
        pattern = re.compile(r"^(\s+)block: ([^\n]+)$", re.MULTILINE)
        source, count = pattern.subn(r"\1block: \2\n\1block: \2", source_path.read_text(encoding="utf-8"), count=1)
        assert count == 1
        write_ssot_text(candidate, source, encoding="utf-8")
    else:
        document = load_ssot_document(source_path)
        row = document["declarations"][0]
        if mutation == "missing-declarations":
            document.pop("declarations")
        elif mutation == "invalid-row-key":
            row["allow_live"] = True
        elif mutation == "invalid-disposition":
            row["disposition"] = "ignore"
        else:
            row["reason"] = "unreviewed_reason"
        write_ssot_text(candidate, yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_derivative_inactivity(candidate)


@pytest.mark.parametrize("mutation", ["removed-row", "disconnected-loader", "duplicate-assignment", "missing-import"])
def test_derivative_inactivity_static_gate_rejects_disconnected_contract(tmp_path, mutation):
    """Authored scope edits remain possible while disconnected harness wiring fails."""
    from gdd.engine.derivative_inactivity import inactivity_contract_errors

    declaration_path = tmp_path / "gdd/ssot/fixtures/derivative_inactivity.yaml"
    harness_path = tmp_path / "tests/test_jqmc_kernel/_grad_fidelity_harness.py"
    for target in (declaration_path, harness_path):
        target.parent.mkdir(parents=True, exist_ok=True)
        write_ssot_text(target, (_ROOT / target.relative_to(tmp_path)).read_text(encoding="utf-8"), encoding="utf-8")
    assert inactivity_contract_errors(tmp_path) == []
    if mutation == "removed-row":
        document = load_ssot_document(declaration_path)
        document["declarations"].pop(0)
        write_ssot_text(declaration_path, yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    else:
        source = harness_path.read_text(encoding="utf-8")
        if mutation == "disconnected-loader":
            assert "= load_derivative_inactivity()" in source
            source = source.replace("= load_derivative_inactivity()", "= {}", 1)
        elif mutation == "duplicate-assignment":
            source += "\nSTRUCTURALLY_ZERO_BLOCKS = {}\n"
        else:
            statement = "from gdd.engine.derivative_inactivity import load_derivative_inactivity"
            assert statement in source
            source = source.replace(statement, "", 1)
        write_ssot_text(harness_path, source, encoding="utf-8")
    if mutation == "removed-row":
        assert inactivity_contract_errors(tmp_path) == []
    else:
        assert inactivity_contract_errors(tmp_path), f"static gate accepted {mutation}"


@pytest.mark.parametrize("case", ["undeclared-inactive", "stale-live", "declared-inactive", "undeclared-live"])
def test_derivative_inactivity_runtime_screen_preserves_rejection(case):
    """The actual screen rejects undeclared zeros and stale declarations after migration."""
    from gdd.engine.derivative_inactivity import load_derivative_inactivity
    from jqmc._precision import get_tolerance_min
    from tests.test_jqmc_kernel._grad_fidelity_harness import _assert_nonvacuous

    declarations = load_derivative_inactivity()
    selector = ("bspline__DTN__AE__pbc_gamma__SD", "e_L", "c", "j3_poly_gamma")
    assert selector in declarations
    cell_id, quantity, axis, block = selector
    if case.startswith("undeclared"):
        cell_id = "synthetic-undeclared-nonvacuity-probe"
        assert (cell_id, quantity, axis, block) not in declarations
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval"), "strict")
    reference = np.ones(2) if case in {"stale-live", "undeclared-live"} else np.zeros(2)
    if case in {"undeclared-inactive", "stale-live"}:
        expected = "vacuous" if case == "undeclared-inactive" else "stale"
        with pytest.raises(AssertionError, match=expected):
            _assert_nonvacuous(cell_id, quantity, axis, [(block, reference)], atol, rtol)
    else:
        _assert_nonvacuous(cell_id, quantity, axis, [(block, reference)], atol, rtol)


@cache
def _transport_case():
    """A small real route/position/comparison receipt for transport negative controls."""
    from gdd.engine.evidence import CaseReceipt, ProductionRouteCapture

    electrons = observe_electron_configurations(
        ElectronConfigurationBinding(
            binding_id="transport-positions",
            mode="required",
            generation_pattern="direct",
            position_source_kind="seeded-random",
            generator_operation="unit-generator",
            random_algorithm="numpy.random.PCG64",
            seed=1,
            configuration_count=1,
            geminal_conditioning_mode="not-limited",
            production_route_capture="unit-route",
        ),
        np.ones((1, 3)),
        np.empty((0, 3)),
    )
    capture = ProductionRouteCapture("unit-route", lambda args, kwargs: [(args[0], args[1])])
    values = capture.wrap(lambda up, down: up + 1)(np.ones((1, 3)), np.empty((0, 3)))
    compared = observe_exact(
        ComparisonBinding("transport-compare", "CMP-EXACT-VALUES-001", "full", "actual", "reference"),
        [(values, np.full((1, 3), 2.0))],
        electron_receipt=electrons,
    )
    return CaseReceipt(
        guarantee_id="transport-guarantee",
        obligation_id="transport-obligation",
        pytest_node_id="unit::transport",
        input_binding=InputBinding(
            binding_id="transport-input",
            kind="parameter-set",
            builder_operation="unit-builder",
            transformations=(),
            parameter_set_id="unit-parameters",
            parameter_source="unit-source",
            content_snapshot="sha256:unit",
        ),
        electron_configuration=electrons,
        comparison=compared,
        additional_checks=(compared,),
        route_capture=capture.receipt(),
        status="passed",
    )


@pytest.mark.parametrize(
    "mutation", ["none", "missing-field", "unknown-type", "wrong-scalar", "no-route", "wrong-positions", "zero-values"]
)
def test_derivative_receipt_transport_revalidates_payload(mutation):
    """A valid outer JSON/checksum cannot substitute for valid inner numerical evidence.

    ``none`` is the positive control: the JSON round-trip restores the receipt exactly.
    """
    from gdd.engine.receipt_records import decode_case_receipt, encode_case_receipt

    original = _transport_case()
    payload = encode_case_receipt(original)
    if mutation == "none":
        assert decode_case_receipt(json.loads(json.dumps(payload))) == original
        return
    values = payload["fields"]
    if mutation == "missing-field":
        del values["electron_configuration"]
    elif mutation == "unknown-type":
        values["comparison"]["type"] = "arbitrary-import.Callable"
    elif mutation == "wrong-scalar":
        values["comparison"]["fields"]["compared_element_count"] = True
    elif mutation == "no-route":
        values["route_capture"] = None
    elif mutation == "wrong-positions":
        values["electron_configuration"]["fields"]["ordered_coordinate_snapshots"] = ["uncomputed-input"]
    else:
        values["comparison"]["fields"]["compared_element_count"] = 0
    with pytest.raises(ValueError, match="invalid derivative evidence payload"):
        decode_case_receipt(payload)


@pytest.mark.parametrize("backend", ["xla", "rust"])
def test_gdd_xla_autodiff_analytical_universe_is_live_and_complete(backend):
    """Every live requirement has registered blocking evidence without a count pin.

    ``rust``: every XLA analytical obligation also keeps one blocking Rust
    implementation and parity registration; a reduced Rust subset is rejected.
    """
    if backend == "rust":
        rust_required = required_rust_case_ids()
        assert _CONTRACT["baseline"]["rust_manifest"]["explicit_not_applicable"] == 0
        assert len(rust_required) == len(set(rust_required))
        assert set(rust_required) == {requirement.id for requirement in REQUIRED_CASES}
        diff = rust_required_registration_diff()
        assert diff.complete, (
            "GDD forbids a reduced Rust fixture subset; "
            f"missing={diff.missing}, extra={diff.extra}, duplicate={diff.duplicate}, "
            f"function_mismatch={diff.function_mismatch}"
        )
        return
    declared_families = {row["family"] for row in _CONTRACT["guarantees"] if row.get("family")}
    assert {case.family for case in REQUIRED_CASES} == declared_families
    assert len(requirements_by_id()) == len(REQUIRED_CASES)

    diff = registration_diff()
    assert diff.complete, (
        "GDD requires every XLA analytical obligation to retain blocking "
        "autodiff/identity pytest evidence; "
        f"missing={diff.missing}, extra={diff.extra}, duplicates={diff.duplicates}, "
        f"unavailable={diff.unavailable}, precision={diff.precision_mismatch}, "
        f"proof={diff.proof_mismatch}, liveness={diff.missing_liveness}"
    )


_BINDING_ISSUES = {
    "matrix": generated_matrix_binding_issues,
    "targeted": targeted_binding_issues,
    "gfmc": gfmc_matrix_binding_issues,
    "regime": regime_binding_issues,
    "fd": fd_binding_issues,
    "mixed": mixed_binding_issues,
    "rust": rust_gdd_binding_issues,
}


@pytest.mark.parametrize("category", list(_BINDING_ISSUES))
def test_gdd_generated_matrix_cases_have_executable_bindings(category):
    """Every required derivative case of each category runs an executable binding (Rust: a runtime fragment)."""
    if category == "matrix":
        assert any(requirement.category == "matrix" for requirement in REQUIRED_CASES)
    issues = _BINDING_ISSUES[category]()
    assert not issues, f"{category} derivative GDD binding gaps: {issues}"


def test_gdd_mixed_correctness_oracle_is_shared_and_same_precision():
    """Every mixed analytical matrix must use the same-precision oracle policy."""
    authority = _CONTRACT["common_contract"]["comparison_policy"]["mixed_precision"]
    analytical = authority["analytical_correctness"]
    generated = _CONTRACT["existing_test_integration"]["generated_continuous_matrix"]["comparison"]["mixed"]
    gfmc = _CONTRACT["existing_test_integration"]["gfmc_broad_matrix"]["mixed-correctness-oracle"]
    molecular_stress = _CONTRACT["existing_test_integration"]["gfmc_broad_matrix"]["molecular-all-electron-stress-oracle"]

    for field in (
        "reference",
        "reference-operation",
        "policy-id",
        "reference-precision",
    ):
        assert generated[field] == analytical[field]
        assert gfmc[field] == analytical[field]
    assert analytical["reference"] == "xla-autodiff-same-precision"
    assert analytical["reference-precision"] == "same-as-production"
    assert authority["supplemental_smoke"] == {
        "observation-policy": "OBS-MIXED-ROUTE-SMOKE-001",
        "numerical-correctness-oracle": False,
    }
    assert molecular_stress["reference-operation"] == ("derivatives.reference.same-precision-position-ad-affine-virial.v1")
    assert molecular_stress["comparison-policy"] == {
        "full": "CMP-ZONE-ALLCLOSE-STRICT-001",
        "mixed": "CMP-ZONE-ALLCLOSE-MEDIUM-001",
    }
    assert molecular_stress["tolerance-scale"] == "elementwise-reference-component"


def test_gdd_production_entry_discovery_observes_new_boundary_and_direct_ad(tmp_path):
    derivative_dir = tmp_path / "jqmc/derivatives"
    derivative_dir.mkdir(parents=True)
    write_ssot_text((tmp_path / "jqmc/__init__.py"), "", encoding="utf-8")
    write_ssot_text((derivative_dir / "__init__.py"), "", encoding="utf-8")
    write_ssot_text((derivative_dir / "future.py"), "def new_force():\n    return 0\n", encoding="utf-8")
    consumer = tmp_path / "jqmc/jqmc_future.py"
    write_ssot_text(
        consumer,
        "import jax\n"
        "from .derivatives.future import new_force\n\n"
        "def energy(x):\n"
        "    return x * x\n\n"
        "def run(x):\n"
        "    new_force()\n"
        "    return jax.grad(energy)(x)\n",
        encoding="utf-8",
    )

    observed = discover_external_derivative_references(tmp_path)
    assert "jqmc.derivatives.future.new_force" in observed
    direct = discover_direct_autodiff_calls(tmp_path, ("jqmc/jqmc_future.py",))
    assert [item.id for item in direct] == ["jqmc/jqmc_future.py::run::jax.grad::energy"]


def _completion_payload(**fields) -> dict:
    from gdd.engine.module_receipts import contract_snapshot
    from gdd.engine.receipt_records import encode_case_receipt

    payload = {
        "schema_version": 2,
        "record_kind": "case",
        "lane": "xla-full",
        "guarantee_id": "MATH-DERIV-LOG-WAVEFUNCTION-001",
        "obligation_id": "placeholder",
        "upstream_obligation_id": None,
        "component_id": None,
        "pytest_node_id": "synthetic-contract-check",
        "status": "passed",
        "worker_id": "synthetic",
        "run_envelope": {
            "commit_sha": "local",
            "contract_snapshot": contract_snapshot(),
            "build_id": "synthetic-build",
            "run_id": "synthetic-run",
            "job_id": "synthetic-job",
            "shard_id": "synthetic-shard",
            "worker_id": "synthetic",
        },
        **fields,
    }
    receipt = replace(
        _transport_case(),
        guarantee_id=payload["guarantee_id"],
        obligation_id=payload["obligation_id"],
        pytest_node_id=payload["pytest_node_id"],
        status=payload["status"],
    )
    if receipt.guarantee_id == "NUM-DERIV-FD-AUTODIFF-001":
        # Synthetic transport evidence must carry the same per-case metadata
        # as real FD receipts; this is not a replacement scientific reference.
        from gdd.engine.fd_comparison_contract import load_fd_contract
        from tests.test_jqmc_kernel.test_gdd_fd_comparison_contract import _receipt

        receipt = _receipt(receipt.obligation_id, load_fd_contract()[receipt.obligation_id])
        payload["pytest_node_id"] = receipt.pytest_node_id
    if "|matrix|" in receipt.obligation_id and payload["lane"] in {"xla-full", "mixed"}:
        # Deliberately synthetic aggregator input; real object observation is
        # checked independently by the observer/route negative controls.
        from gdd.engine.evidence import RouteFixtureReceipt
        from gdd.engine.fixture_observations import (
            AO_SIGNATURES,
            ComponentBoundaryReceipt,
            DTNCuspReceipt,
            HamiltonianFixtureReceipt,
            NuclearPotentialReceipt,
            orbital_boundary_roles,
        )

        ao, jas, ecp, boundary, gem = receipt.obligation_id.split("|")[2].split("__")
        signature = AO_SIGNATURES.get(ao) or ("pgto-plane-wave-complex" if boundary == "twist" else "pgto-plane-wave-real")
        components = {
            "none": (),
            "J1": ("J1", "J2"),
            "J2": ("J1", "J2"),
            "J3poly": ("J1", "J2", "J3"),
            "DTN": ("J1", "J2", "DTN"),
        }[jas]
        boundary_roles = ["hamiltonian", "coulomb"]
        for spin in ("up", "down"):
            boundary_roles.extend(orbital_boundary_roles(f"determinant.{spin}", signature))
        boundary_roles.extend(f"jastrow.{channel}" for channel in ("J1", "J2", "DTN") if channel in components)
        if "J3" in components:
            boundary_roles.extend(orbital_boundary_roles("jastrow.J3", signature))
        cell_vectors = () if boundary == "mol" else ((8.0, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 8.0))
        fixture = HamiltonianFixtureReceipt(
            content_snapshot="sha256:" + "0" * 64,
            determinant_kind="AGP" if gem == "AGP_MO" else gem,
            representation="AO" if gem == "AGP" else "MO",
            orbital_signatures=(signature, signature),
            electron_counts=(1, 0),
            jastrow_components=components,
            j1_form="exp" if "J1" in components else "off",
            j2_form="pade" if "J2" in components else "off",
            j3_representation="MO" if "J3" in components else "off",
            j3_orbital_signature=signature if "J3" in components else "not-applicable",
            dtn_channels=("u", "chi", "f") if "DTN" in components else (),
            periodic=boundary != "mol",
            cell_metric="not-applicable" if boundary == "mol" else "cubic",
            k_points=((0.25 if boundary == "twist" else 0.0, 0.0, 0.0),) * 2,
            ecp=ecp == "ECP",
            component_boundaries=tuple(
                ComponentBoundaryReceipt(role, boundary != "mol", cell_vectors) for role in boundary_roles
            ),
            nuclei=(NuclearPotentialReceipt(1, 0.0, 1 if ecp == "ECP" else 0, 0),),
            dtn_cusp=DTNCuspReceipt(True, (0,), (1.0,)) if "DTN" in components else None,
        )
        route = receipt.route_capture
        receipt = replace(
            receipt, route_capture=replace(route, fixture_calls=(RouteFixtureReceipt(fixture, route.coordinate_snapshots),))
        )
    payload.setdefault("evidence", encode_case_receipt(receipt))
    return payload


@pytest.mark.parametrize(
    "mutation", ["missing", "different-cell", "different-coordinates", "different-population", "boundary-cell", "dtn-charge"]
)
def test_matrix_completion_requires_consumed_fixture_facts(mutation):
    """Valid IDs and numerical payloads cannot hide absent or mismatched matrix inputs.

    ``boundary-cell``: a serialized internal component boundary from another cell;
    ``dtn-charge``: a DTN cusp target with another nuclear charge.
    """
    from gdd.engine.receipt_records import decode_case_receipt, encode_case_receipt

    if mutation in {"boundary-cell", "dtn-charge"}:
        payload = _completion_payload(obligation_id="ln_Psi|matrix|cart_pgto__DTN__AE__pbc_gamma__SD|axis=r")
        validate_record(payload)
        fixture = payload["evidence"]["fields"]["route_capture"]["fields"]["fixture_calls"][0]["fields"]["fixture"]["fields"]
        if mutation == "boundary-cell":
            fixture["component_boundaries"][-1]["fields"]["cell"][0][0] *= 2
        else:
            fixture["dtn_cusp"]["fields"]["effective_charges"] = [0.0]
        with pytest.raises((ValueError, AssertionError), match="boundary|DTN|nuclear"):
            validate_record(payload)
        return
    payload = _completion_payload(obligation_id="ln_Psi|matrix|cart_gto__none__AE__mol__SD|axis=r")
    validate_record(payload)
    receipt = decode_case_receipt(payload["evidence"])
    route = receipt.route_capture
    call = route.fixture_calls[0]
    if mutation == "missing":
        route = replace(route, fixture_calls=())
    elif mutation == "different-cell":
        route = replace(route, fixture_calls=(replace(call, fixture=replace(call.fixture, determinant_kind="AGPn")),))
    elif mutation == "different-coordinates":
        route = replace(route, fixture_calls=(replace(call, coordinate_snapshots=("not-consumed",)),))
    else:
        route = replace(route, fixture_calls=(replace(call, fixture=replace(call.fixture, electron_counts=(2, 0))),))
    if mutation == "different-coordinates":
        with pytest.raises(AssertionError, match="fixture observations differ"):
            encode_case_receipt(replace(receipt, route_capture=route))
    else:
        payload["evidence"] = encode_case_receipt(replace(receipt, route_capture=route))
        with pytest.raises(AssertionError, match=r"Hamiltonian|determinant|electron counts"):
            validate_record(payload)


@pytest.mark.parametrize("contract", ["cond-g", "generation-pattern", "attempt-accounting"])
def test_gdd_cond_g_receipts_enforce_bounds_ranges_and_sigma_min(contract):
    """A benign configuration cannot claim an ill-conditioned regime.

    ``cond-g``: bounds, ranges, attempts and sigma_min are enforced on receipts;
    ``generation-pattern``: a rejection-screen binding without a constrained cond(G) is rejected;
    ``attempt-accounting``: realized plus rejected configurations must equal the attempts.
    """
    if contract == "generation-pattern":
        binding = {
            "binding_id": "GENERATION-PATTERN-CONTRACT",
            "mode": "required",
            "position_source_kind": "seeded-random",
            "generator_operation": "contract-test.draw",
            "generator_arguments": (),
            "random_algorithm": "numpy.random.PCG64",
            "seed": 0,
            "configuration_count": 1,
            "generation_pattern": "rejection-screen",
            "geminal_conditioning_mode": "upper-bounded",
            "maximum_cond_g": 700.0,
            "upper_bound_comparison": "less-than",
            "conditioning_violation_action": "deterministic-resample",
            "maximum_generation_attempts": 10,
        }
        ElectronConfigurationBinding(**binding).validate()
        unconstrained = {
            **binding,
            "geminal_conditioning_mode": "not-limited",
            "maximum_cond_g": None,
            "upper_bound_comparison": None,
            "conditioning_violation_action": None,
            "maximum_generation_attempts": None,
        }
        with pytest.raises(AssertionError, match=r"constrained cond\(G\)"):
            ElectronConfigurationBinding(**unconstrained).validate()
        return
    if contract == "attempt-accounting":
        direct = ElectronConfigurationBinding(
            binding_id="PATTERN-TOKEN",
            mode="required",
            generation_pattern="direct",
            position_source_kind="explicit",
            configuration_count=1,
            position_set_id="POSITIONS-001",
            coordinates_source="contract-test",
            content_snapshot="sha256:positions",
            geminal_conditioning_mode="not-limited",
        )
        receipt = observe_electron_configurations(direct, np.asarray([[0.0, 0.0, 0.0]]), np.empty((0, 3)))
        receipt.validate()
        with pytest.raises(AssertionError, match="realized plus rejected"):
            replace(receipt, generation_attempts=3, rejected_configuration_count=0).validate()
        return
    r_up = np.asarray([[[0.0, 0.0, 0.0]], [[0.1, 0.0, 0.0]]])
    r_dn = np.empty((2, 0, 3))
    common = {
        "binding_id": "COND-G-CONTRACT-TEST",
        "mode": "required",
        "generation_pattern": "rejection-screen",
        "position_source_kind": "seeded-random",
        "generator_operation": "contract-test.draw",
        "generator_arguments": (),
        "random_algorithm": "numpy.random.PCG64",
        "seed": 0,
        "configuration_count": 2,
    }
    upper = ElectronConfigurationBinding(
        **common,
        geminal_conditioning_mode="upper-bounded",
        maximum_cond_g=700.0,
        upper_bound_comparison="less-than",
        conditioning_violation_action="deterministic-resample",
        maximum_generation_attempts=300,
    )
    observe_electron_configurations(
        upper,
        r_up,
        r_dn,
        cond_g_per_configuration=(10.0, 699.0),
        generation_attempts=3,
        rejected_configuration_count=1,
    )
    with pytest.raises(AssertionError, match=r"cond\(G\).*violates"):
        observe_electron_configurations(upper, r_up, r_dn, cond_g_per_configuration=(10.0, 700.0))

    target = replace(
        upper,
        geminal_conditioning_mode="target-range",
        generation_pattern="condition-target",
        generator_arguments=(("control_parameter", "same-spin-separation-delta"), ("search_method", "declared-log-grid")),
        minimum_cond_g=1.0e3,
        maximum_cond_g=1.0e6,
        lower_bound_comparison="greater-than",
        minimum_singular_value=1.0e-8,
        conditioning_violation_action="fail-case",
        maximum_generation_attempts=2,
    )
    target_receipt = observe_electron_configurations(
        target,
        r_up,
        r_dn,
        cond_g_per_configuration=(8.2e3, 2.0e4),
        sigma_min_g_per_configuration=(1.0e-5, 1.0e-6),
        sigma_max_g_per_configuration=(8.2e-2, 2.0e-2),
        machine_epsilon_per_configuration=(np.finfo(np.float64).eps,) * 2,
        cond_times_epsilon_per_configuration=(
            8.2e3 * np.finfo(np.float64).eps,
            2.0e4 * np.finfo(np.float64).eps,
        ),
    )
    with pytest.raises(AssertionError, match="exceeded its configured maximum"):
        replace(target_receipt, generation_attempts=3, rejected_configuration_count=1).validate()
    with pytest.raises(AssertionError, match="canonical control/search"):
        replace(target_receipt, canonical_generator_arguments=()).validate()
    with pytest.raises(AssertionError, match=r"sigma_min\(G\)"):
        observe_electron_configurations(
            target,
            r_up,
            r_dn,
            cond_g_per_configuration=(8.2e3, 2.0e4),
            sigma_min_g_per_configuration=(1.0e-5, 1.0e-9),
            sigma_max_g_per_configuration=(8.2e-2, 2.0e-5),
            machine_epsilon_per_configuration=(np.finfo(np.float64).eps,) * 2,
            cond_times_epsilon_per_configuration=(
                8.2e3 * np.finfo(np.float64).eps,
                2.0e4 * np.finfo(np.float64).eps,
            ),
        )


@pytest.mark.parametrize("probe", ["fixture-probes", "qe-engine-harness"])
def test_gdd_required_fd_fixture_availability_probes_are_live(probe):
    """Required fixtures are available, so no required case is skipped green.

    ``qe-engine-harness``: the engine's self-computed QE sentinel, which decides
    whether QE cases are required, agrees with the harness that runs them.
    """
    if probe == "qe-engine-harness":
        from gdd.engine.derivative_matrix import _QE_AVAILABLE, _QE_STEM

        from tests.test_jqmc_kernel import _grad_fidelity_harness as harness

        assert _QE_STEM == harness._QE_STEM
        assert _QE_AVAILABLE == harness._QE_AVAILABLE
        return
    fd_fixture = next(fixture for fixture in _CONTRACT["fixture_sets"] if fixture["id"] == "DERIV-FIX-FD-AUTODIFF-001")
    probes = {
        node["fixture"]["availability_probe"]
        for node in fd_fixture["reviewed_evidence"]["nodes"]
        if "availability_probe" in node["fixture"]
    }

    assert probes, "conditional required fixtures must expose at least one availability probe"
    unavailable = []
    for probe in sorted(probes):
        module_name, attribute = probe.rsplit(".", 1)
        if not bool(getattr(import_module(module_name), attribute)):
            unavailable.append(probe)
    assert not unavailable, f"required GDD fixtures are unavailable: {unavailable}"


@pytest.mark.parametrize("registry", ["ao-jastrow", "specialized-ao", "type-registries"])
def test_gdd_engine_matrix_vocabulary_matches_production_introspection(registry):
    """The engine's DECLARED vocabularies must equal production introspection.

    The requirement-side engine must stay importable without jqmc, so its axis
    vocabularies are declared contract data.  This gate keeps the declaration
    honest: a new production AO leaf family or Jastrow channel fails here until
    the declared matrix or an explicit frozen-adapter derivative contract is extended.
    ``specialized-ao``: every frozen adapter outside the matrix has required full/mixed
    AD evidence. ``type-registries``: the geminal, Jastrow-channel and DTN-flag
    registries equal introspection.
    """
    import dataclasses

    if registry == "specialized-ao":
        from gdd.engine.derivative_matrix import _AO_FAMILY_TOKENS, _SPECIALIZED_AO_FAMILY_CONTRACTS

        assert set(_AO_FAMILY_TOKENS).isdisjoint(_SPECIALIZED_AO_FAMILY_CONTRACTS)
        for module, guarantee_id in _SPECIALIZED_AO_FAMILY_CONTRACTS.values():
            document = load_module(_ROOT / "gdd/ssot/guarantees" / module)
            guarantee = next(row for row in document["guarantees"] if row["id"] == guarantee_id)
            assert guarantee["kind"] == "mathematical"
            nodes = [
                node
                for evidence in guarantee["required_evidence"]
                if evidence["verdict_method"] == "VM-MATH-AD-ORACLE-001"
                for node in evidence["nodes"]
            ]
            assert nodes
            for node in nodes:
                assert set(node["required_precision_modes"]) == {"full", "mixed"}
                assert node["required_parameter_cases"]
                assert (_ROOT / node["pytest_node"].split("::")[0]).is_file()
        return
    if registry == "type-registries":
        from jqmc.determinant import _GeminalBase
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_DTN_data

        from gdd.engine.derivative_inventory import (
            PRODUCTION_DTN_CHANNEL_FLAGS,
            PRODUCTION_GEMINAL_CLASSES,
            PRODUCTION_JASTROW_CHANNEL_FIELDS,
        )

        geminals = sorted(cls.__name__ for cls in _GeminalBase.__subclasses__() if cls.__module__ == "jqmc.determinant")
        assert tuple(geminals) == PRODUCTION_GEMINAL_CLASSES
        fields = sorted(
            field.name
            for field in dataclasses.fields(Jastrow_data)
            if field.name.startswith("jastrow_") and field.name.endswith("_data")
        )
        assert tuple(fields) == PRODUCTION_JASTROW_CHANNEL_FIELDS
        flags = sorted(field.name for field in dataclasses.fields(Jastrow_DTN_data) if field.name.startswith("use_"))
        assert tuple(flags) == PRODUCTION_DTN_CHANNEL_FLAGS
        return

    import jqmc.atomic_orbital as atomic_orbital
    from jqmc.jastrow_factor import Jastrow_data

    from gdd.engine.derivative_matrix import _PRODUCTION_AO_FAMILY_TOKENS, _JASTROW_CHANNEL_TOKENS

    introspected = sorted(
        name[len("AOs_") : -len("_data")]
        for name in dir(atomic_orbital)
        if name.startswith("AOs_")
        and name.endswith("_data")
        and "_base_data" not in name
        and isinstance(getattr(atomic_orbital, name), type)
    )
    assert tuple(introspected) == _PRODUCTION_AO_FAMILY_TOKENS

    field_token = {
        "jastrow_one_body_data": "J1",
        "jastrow_two_body_data": "J2",
        "jastrow_three_body_data": "J3poly",
        "jastrow_dtn_data": "DTN",
    }
    channels = {field_token[field.name] for field in dataclasses.fields(Jastrow_data) if field.name in field_token}
    assert tuple(sorted(channels | {"none"})) == _JASTROW_CHANNEL_TOKENS


def test_gdd_route_capture_and_screening_order_are_mechanically_enforced():
    """Anti-vacuity rules 1 and 3 must have real teeth in CaseReceipt.validate.

    Rule 1: a declared production route demands a capture receipt proving every
    declared electron configuration reached the wrapped production call.
    Rule 3: every comparison receipt must carry the finalized electron receipt's
    sequence token, so a comparison cannot precede screening.
    """
    from dataclasses import replace as dc_replace

    from jqmc._setting import ATOL_BIT_EXACT, RTOL_BIT_EXACT

    from gdd.engine.evidence import (
        CaseReceipt,
        ComparisonBinding,
        ElectronConfigurationBinding,
        InputBinding,
        ProductionRouteCapture,
        observe_allclose,
        observe_electron_configurations,
    )

    route = "unit.test.production-route.v1"
    binding = ElectronConfigurationBinding(
        binding_id="UNIT::route-capture",
        mode="required",
        generation_pattern="direct",
        position_source_kind="seeded-random",
        generator_operation="unit.generator",
        random_algorithm="numpy.random.PCG64",
        seed=7,
        configuration_count=2,
        geminal_conditioning_mode="not-limited",
        production_route_capture=route,
    )
    rng = np.random.default_rng(7)
    r_up = rng.normal(size=(2, 2, 3))
    r_dn = rng.normal(size=(2, 2, 3))
    electron_receipt = observe_electron_configurations(binding, r_up, r_dn)

    capture = ProductionRouteCapture(route, lambda args, kwargs: [(args[0], args[1])])
    production = capture.wrap(lambda up, dn: np.asarray(up) + np.asarray(dn))
    value = production(r_up, r_dn)

    comparison_binding = ComparisonBinding(
        binding_id="UNIT::comparison",
        policy_id="CMP-DERIV-AUTODIFF-ANALYTICAL-FULL-001",
        precision_mode="full",
        actual_role="xla-analytical",
        reference_role="xla-autodiff",
    )
    comparison = observe_allclose(
        comparison_binding, [(value, value)], atol=ATOL_BIT_EXACT, rtol=RTOL_BIT_EXACT, electron_receipt=electron_receipt
    )
    input_binding = InputBinding(
        binding_id="UNIT::input",
        kind="parameter-set",
        builder_operation="unit.builder",
        transformations=(),
        parameter_set_id="UNIT-PS",
        parameter_source="yaml-inline",
        content_snapshot="sha256:unit",
    )
    receipt = CaseReceipt(
        guarantee_id="UNIT-GUARANTEE",
        obligation_id="UNIT|case",
        pytest_node_id="unit::node",
        input_binding=input_binding,
        electron_configuration=electron_receipt,
        comparison=comparison,
        status="passed",
        route_capture=capture.receipt(),
    )
    receipt.validate()

    # Rule 1a: a declared route without a capture receipt is rejected.
    with pytest.raises(AssertionError, match="no route-capture receipt"):
        dc_replace(receipt, route_capture=None).validate()

    # Rule 1b: a configuration that never reached the production route is rejected.
    partial_capture = ProductionRouteCapture(route, lambda args, kwargs: [(args[0], args[1])])
    partial_production = partial_capture.wrap(lambda up, dn: np.asarray(up) + np.asarray(dn))
    partial_production(r_up[:1], r_dn[:1])
    with pytest.raises(AssertionError, match="never reached the production route"):
        dc_replace(receipt, route_capture=partial_capture.receipt()).validate()

    # Rule 1c: a capture from a different route is rejected.
    wrong_route = ProductionRouteCapture("unit.test.other-route.v1", lambda args, kwargs: [(args[0], args[1])])
    wrong_route.wrap(lambda up, dn: None)(r_up, r_dn)
    with pytest.raises(AssertionError, match="does not match"):
        dc_replace(receipt, route_capture=wrong_route.receipt()).validate()

    # Rule 3: a comparison receipt without the finalized electron token (or with a
    # token from different screening) is rejected.
    with pytest.raises(AssertionError, match="screening-order proof missing"):
        dc_replace(receipt, comparison=dc_replace(comparison, electron_sequence_token=None)).validate()
    other_screening = dc_replace(electron_receipt, generation_attempts=99)
    with pytest.raises(AssertionError, match="screening-order proof missing"):
        dc_replace(
            receipt, comparison=dc_replace(comparison, electron_sequence_token=other_screening.sequence_token())
        ).validate()
