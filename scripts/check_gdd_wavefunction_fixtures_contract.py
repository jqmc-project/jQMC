#!/usr/bin/env python3
"""Static CI gate for the shared wavefunction case contract."""

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

import ast
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdd.engine.coverage import missing_requirements
from gdd.engine.pinned_fixtures import fixture_exists
from gdd.engine.wavefunction_fixtures import classify_cell_metric

CONTRACT_PATH = ROOT / "gdd/ssot/fixtures/wavefunctions.yaml"
SOURCE_PATH = ROOT / "gdd/ssot/fixtures/sources.yaml"
ELECTRON_CONFIG_PATH = ROOT / "gdd/ssot/fixtures/electron_configurations.yaml"

SIGNATURE_FIELDS = {
    "determinant": {"kind", "representation", "orbitals"},
    "jastrow": {
        "j1_form",
        "j2_form",
        "j3_representation",
        "j3_orbital_signature",
        "dtn_channel_mask",
        "dtn_en_cusp_target",
    },
    "electrons": {"shell_structure", "spin_restriction"},
}
TOP_SIGNATURE_FIELDS = {
    *SIGNATURE_FIELDS,
    "periodicity",
    "bloch_k_regime",
    "cell_metric_family",
}
VOCABULARY_PATHS = {
    "determinant_kind": ("determinant", "kind"),
    "determinant_representation": ("determinant", "representation"),
    "j1_form": ("jastrow", "j1_form"),
    "j2_form": ("jastrow", "j2_form"),
    "j3_representation": ("jastrow", "j3_representation"),
    "dtn_channel_mask": ("jastrow", "dtn_channel_mask"),
    "dtn_en_cusp_target": ("jastrow", "dtn_en_cusp_target"),
    "shell_structure": ("electrons", "shell_structure"),
    "spin_restriction": ("electrons", "spin_restriction"),
    "periodicity": ("periodicity",),
    "bloch_k_regime": ("bloch_k_regime",),
    "cell_metric_family": ("cell_metric_family",),
}


def _load_yaml(path: Path, errors: list[str]) -> dict[str, Any]:
    from gdd.engine.guarantee_store import is_module_ledger, load_module

    try:
        value = load_module(path) if is_module_ledger(path) else yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        errors.append(f"cannot load {path.relative_to(ROOT)}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path.relative_to(ROOT)} must be a YAML mapping")
        return {}
    return value


def _class_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def _class_fields(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        raise ValueError(f"{path.relative_to(ROOT)} must define exactly one {class_name}")
    return {node.target.id for node in classes[0].body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}


def _at_path(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for part in path:
        current = current[part]
    return current


def _check_exact_keys(value: Any, expected: set[str], label: str, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{label} must be a mapping")
        return False
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        errors.append(f"{label} fields differ: missing={missing}, extra={extra}")
        return False
    return True


def _validate_document(contract: dict[str, Any], errors: list[str]) -> None:
    required = {
        "schema_version",
        "document",
        "enforcement_map",
        "shared_contracts",
        "production_bindings",
        "orbital_signatures",
        "vocabularies",
        "validity_rules",
        "cell_metric_classifier",
        "canonical_suite",
        "evidence_required_from_each_subscribing_feature",
        "enforcement",
    }
    missing = sorted(required - set(contract))
    extra = sorted(set(contract) - required)
    if missing or extra:
        errors.append(f"wavefunction top-level fields differ: missing={missing}, extra={extra}")
    if contract.get("schema_version") != 1:
        errors.append("wavefunction schema_version must be 1")
    # A document header has no decisions field.
    if "decisions" in contract.get("document", {}):
        errors.append("document header must not contain a decisions field (document.decisions)")


def _validate_vocabularies(contract: dict[str, Any], errors: list[str]) -> None:
    vocabularies = contract["vocabularies"]
    if set(vocabularies) != set(VOCABULARY_PATHS):
        errors.append(
            "vocabulary names differ: "
            f"missing={sorted(set(VOCABULARY_PATHS) - set(vocabularies))}, "
            f"extra={sorted(set(vocabularies) - set(VOCABULARY_PATHS))}"
        )
    for name, values in vocabularies.items():
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            errors.append(f"{name}: values must be a nonempty unique list")
    if vocabularies.get("dtn_en_cusp_target") != ["not-applicable", "ecp-zero", "ae-nonzero"]:
        errors.append("DTN cusp target must distinguish physical AE/ECP targets, not enable/disable a constraint")
    dtn = contract["validity_rules"]["dtn"]
    if dtn["off"].get("en_cusp_target") != ["not-applicable"] or dtn["enabled"].get("en_cusp_target") != [
        "ecp-zero",
        "ae-nonzero",
    ]:
        errors.append("DTN validity rules must retain the physical cusp targets")
    if dtn.get("constraint_guard") != {
        "apply_en_cusp": True,
        "unsafe_unconstrained_data": "rejected",
        "guarantee_ref": "IMPL-JAS-DTN-CUSP-VALIDATION-001",
    }:
        errors.append("DTN cusp constraints must remain enforced for AE and ECP")
    if type(dtn.get("constraint_guard", {}).get("apply_en_cusp")) is not bool:
        errors.append("DTN constraint flag must be boolean")
    ledger = _load_yaml(ROOT / "gdd/ssot/guarantees/jastrow_factor.yaml", errors)
    rejection = next((row for row in ledger.get("guarantees", ()) if row.get("id") == "IMPL-JAS-DTN-CUSP-VALIDATION-001"), {})
    nodes = {
        node.get("pytest_node")
        for group in rejection.get("required_evidence", ())
        if group.get("verdict_method") == "VM-ERROR-DISPOSITION-001"
        for node in group.get("nodes", ())
    }
    prefix = "tests/test_jqmc_kernel/test_jastrow_dtn.py::"
    required = {
        prefix + "TestDTNElectronNucleusCuspPerSpecies::test_cusp_constraints_cannot_be_disabled",
    }
    if not required <= nodes:
        errors.append("DTN unconstrained-input rejection requires blocking evidence for AE/ECP species")


def _validate_production_bindings(contract: dict[str, Any], errors: list[str]) -> None:
    bindings = contract["production_bindings"]
    orbital_path = ROOT / bindings["orbital_module"]
    orbital_classes = _class_names(orbital_path)
    discovered_orbitals = {
        name
        for name in orbital_classes
        if name.startswith("AOs_")
        and name.endswith("_data")
        and name not in {"AOs_base_data", "AOs_GTO_base_data", "AOs_pGTO_base_data"}
    }
    signatures = contract["orbital_signatures"]
    declared_orbitals = {row.get("evaluator") for row in signatures.values()}
    if declared_orbitals != discovered_orbitals:
        errors.append(
            "orbital evaluator vocabulary differs from production classes: "
            f"missing={sorted(discovered_orbitals - declared_orbitals)}, "
            f"extra={sorted(declared_orbitals - discovered_orbitals)}"
        )

    determinant_classes = _class_names(ROOT / bindings["determinant_module"])
    declared_determinants = bindings["determinant_classes"]
    if set(declared_determinants) != set(contract["vocabularies"]["determinant_kind"]):
        errors.append("determinant class bindings must cover determinant_kind exactly")
    missing_determinants = set(declared_determinants.values()) - determinant_classes
    if missing_determinants:
        errors.append(f"production determinant classes are missing: {sorted(missing_determinants)}")

    try:
        jastrow_fields = _class_fields(ROOT / bindings["jastrow_module"], bindings["jastrow_container"])
    except (OSError, SyntaxError, ValueError) as exc:
        errors.append(str(exc))
    else:
        missing_fields = set(bindings["jastrow_fields"]) - jastrow_fields
        if missing_fields:
            errors.append(f"Jastrow container is missing fields: {sorted(missing_fields)}")


def _validate_orbital_registry(contract: dict[str, Any], errors: list[str]) -> None:
    signatures = contract["orbital_signatures"]
    allowed_families = {"gto", "pgto", "plane-wave", "bspline", "composite"}
    allowed_angular = {"cartesian", "spherical", "not-applicable"}
    for signature_id, row in signatures.items():
        label = f"orbital_signatures.{signature_id}"
        required = {"evaluator", "family", "angular_representation", "scalar_field"}
        allowed = required | {"children"}
        if not isinstance(row, dict) or not required <= set(row) or not set(row) <= allowed:
            errors.append(f"{label} must contain {sorted(required)} and optional children only")
            continue
        if row["family"] not in allowed_families:
            errors.append(f"{label}: unknown family {row['family']}")
        if row["angular_representation"] not in allowed_angular:
            errors.append(f"{label}: unknown angular representation {row['angular_representation']}")
        if row["scalar_field"] not in {"real", "complex"}:
            errors.append(f"{label}: scalar_field must be real or complex")
        children = row.get("children")
        if row["family"] == "composite":
            if not isinstance(children, list) or len(children) < 2 or len(children) != len(set(children)):
                errors.append(f"{label}: composite children must be an ordered unique list with at least two entries")
                continue
            unknown = sorted(set(children) - set(signatures))
            if unknown:
                errors.append(f"{label}: unknown composite children {unknown}")
            child_fields = {signatures[child].get("scalar_field") for child in children if child in signatures}
            if child_fields and child_fields != {row["scalar_field"]}:
                errors.append(f"{label}: child scalar fields do not match the composite")
        elif children is not None:
            errors.append(f"{label}: only composite signatures may declare children")


def _orbital_family(contract: dict[str, Any], signature_id: str) -> str | None:
    if signature_id == "not-applicable":
        return None
    return contract["orbital_signatures"][signature_id]["family"]


def _validate_case(contract: dict[str, Any], case: dict[str, Any], errors: list[str]) -> None:
    label = case.get("id", "<missing-case-id>")
    signature = case.get("signature")
    if not _check_exact_keys(signature, TOP_SIGNATURE_FIELDS, f"{label}.signature", errors):
        return
    for section, fields in SIGNATURE_FIELDS.items():
        if not _check_exact_keys(signature[section], fields, f"{label}.{section}", errors):
            return
    if not _check_exact_keys(signature["determinant"]["orbitals"], {"up", "down"}, f"{label}.orbitals", errors):
        return

    vocabularies = contract["vocabularies"]
    for vocabulary, path in VOCABULARY_PATHS.items():
        value = _at_path(signature, path)
        if value not in vocabularies[vocabulary]:
            errors.append(f"{label}: {'.'.join(path)}={value} is not in {vocabulary}")

    orbital_ids = set(contract["orbital_signatures"])
    orbital_refs = {
        "determinant.orbitals.up": signature["determinant"]["orbitals"]["up"],
        "determinant.orbitals.down": signature["determinant"]["orbitals"]["down"],
        "jastrow.j3_orbital_signature": signature["jastrow"]["j3_orbital_signature"],
    }
    for field, value in orbital_refs.items():
        if value != "not-applicable" and value not in orbital_ids:
            errors.append(f"{label}: {field} references unknown orbital signature {value}")
            return
    if orbital_refs["determinant.orbitals.up"] == "not-applicable":
        errors.append(f"{label}: the up-spin determinant orbital cannot be not-applicable")

    rules = contract["validity_rules"]
    determinant = signature["determinant"]
    allowed_representation = rules["determinant_representation_by_kind"][determinant["kind"]]
    if determinant["representation"] not in allowed_representation:
        errors.append(f"{label}: {determinant['kind']} forbids representation {determinant['representation']}")

    jastrow = signature["jastrow"]
    j3_representation = jastrow["j3_representation"]
    j3_signature = jastrow["j3_orbital_signature"]
    j3_rule = rules["j3"][j3_representation]
    if j3_representation == "off":
        if j3_signature not in j3_rule["orbital_signature"]:
            errors.append(f"{label}: disabled J3 requires j3_orbital_signature=not-applicable")
    elif j3_signature == "not-applicable" or _orbital_family(contract, j3_signature) not in j3_rule["allowed_orbital_families"]:
        errors.append(f"{label}: J3 {j3_representation} forbids orbital signature {j3_signature}")

    dtn_mask = jastrow["dtn_channel_mask"]
    dtn_cusp = jastrow["dtn_en_cusp_target"]
    dtn_rules = rules["dtn"]
    if dtn_mask == "off":
        if dtn_cusp not in dtn_rules["off"]["en_cusp_target"]:
            errors.append(f"{label}: disabled DTN requires dtn_en_cusp_target=not-applicable")
    elif dtn_mask not in dtn_rules["enabled"]["channel_masks"] or dtn_cusp not in dtn_rules["enabled"]["en_cusp_target"]:
        errors.append(f"{label}: enabled DTN has invalid mask/cusp combination")

    electrons = signature["electrons"]
    electron_rules = rules["electron_structure"]
    shell = electrons["shell_structure"]
    restriction = electrons["spin_restriction"]
    down = determinant["orbitals"]["down"]
    shell_rule = electron_rules["shell_structure"][shell]
    if shell_rule["down_orbital_signature"] == "required" and down == "not-applicable":
        errors.append(f"{label}: {shell} requires a down-spin orbital")
    if restriction not in shell_rule["spin_restriction"]:
        errors.append(f"{label}: {shell} forbids spin restriction {restriction}")
    restriction_rule = electron_rules["spin_restriction"][restriction]
    down_rule = restriction_rule["down_orbital_signature"]
    if down_rule == "required" and down == "not-applicable":
        errors.append(f"{label}: {restriction} requires a down-spin orbital")
    if isinstance(down_rule, list) and down not in down_rule:
        errors.append(f"{label}: {restriction} forbids down-spin orbital {down}")
    if down_rule == "required-and-same-evaluator-as-up" and determinant["orbitals"]["up"] != down:
        errors.append(f"{label}: spin-restricted orbitals require identical evaluator signatures")

    boundary_rule = rules["boundary_conditions"][signature["periodicity"]]
    if signature["bloch_k_regime"] not in boundary_rule["bloch_k_regime"]:
        errors.append(f"{label}: invalid Bloch regime for {signature['periodicity']}")
    if signature["cell_metric_family"] not in boundary_rule["cell_metric_family"]:
        errors.append(f"{label}: invalid cell metric for {signature['periodicity']}")
    for field, orbital_id in orbital_refs.items():
        family = _orbital_family(contract, orbital_id)
        if family is not None and family not in boundary_rule["allowed_orbital_families"]:
            errors.append(f"{label}: {signature['periodicity']} forbids {field} family {family}")


def _validate_suite(contract: dict[str, Any], errors: list[str]) -> None:
    suite = contract["canonical_suite"]
    cases = suite.get("cases", [])
    case_ids = [case.get("id") for case in cases]
    if not cases or None in case_ids or len(case_ids) != len(set(case_ids)):
        errors.append("canonical case IDs must be present and unique")
    for case in cases:
        _validate_case(contract, case, errors)

    for vocabulary, path in VOCABULARY_PATHS.items():
        observed = {_at_path(case["signature"], path) for case in cases}
        missing = set(contract["vocabularies"][vocabulary]) - observed
        if missing:
            errors.append(f"canonical suite misses {vocabulary} values {sorted(missing)}")

    observed_up = {case["signature"]["determinant"]["orbitals"]["up"] for case in cases}
    missing_orbitals = set(contract["orbital_signatures"]) - observed_up
    if missing_orbitals:
        errors.append(f"canonical suite misses up-determinant orbital signatures {sorted(missing_orbitals)}")

    compositions: set[str] = set()
    for case in cases:
        jastrow = case["signature"]["jastrow"]
        j123 = any(jastrow[name] != "off" for name in ("j1_form", "j2_form", "j3_representation"))
        dtn = jastrow["dtn_channel_mask"] != "off"
        compositions.add(
            {(False, False): "none", (True, False): "j123", (False, True): "dtn", (True, True): "hybrid"}[(j123, dtn)]
        )
    if compositions != {"none", "j123", "dtn", "hybrid"}:
        errors.append(f"canonical suite misses Jastrow targeted compositions: observed={sorted(compositions)}")

    coverage = suite.get("coverage", {})
    if coverage.get("each_vocabulary_value_at_least_once") is not True:
        errors.append("canonical suite must require each vocabulary value")
    if coverage.get("each_orbital_signature_as_up_once") is not True:
        errors.append("canonical suite must require each orbital signature as an up determinant")
    if coverage.get("full_cartesian_product") != "forbidden":
        errors.append("full Cartesian product must be explicitly forbidden")
    if suite.get("selection", {}).get("rows_are_requirements_not_evidence") is not True:
        errors.append("canonical rows must remain requirements, not evidence")

    # Selected interactions are explicit requirements, not an accidental full
    # Cartesian product. Every tuple must occur in one canonical case; values
    # found separately in different cases do not satisfy it.
    def fact(signature: dict[str, Any], axis: str) -> Any:
        value: Any = signature
        for part in axis.split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(axis)
            value = value[part]
        return value

    interactions = coverage.get("required_interactions")
    if not isinstance(interactions, list) or not interactions:
        errors.append("canonical suite must declare selected required_interactions")
    else:
        interaction_ids: set[str] = set()
        for interaction in interactions:
            interaction_id = interaction.get("id")
            axes = interaction.get("axes")
            tuples = interaction.get("tuples")
            if not isinstance(interaction_id, str) or not interaction_id or interaction_id in interaction_ids:
                errors.append("canonical required_interactions IDs must be present and unique")
                continue
            interaction_ids.add(interaction_id)
            if not isinstance(axes, list) or not 2 <= len(axes) <= 3 or len(axes) != len(set(axes)):
                errors.append(f"{interaction_id}: axes must contain two or three unique paths")
                continue
            if not isinstance(tuples, list) or not tuples or any(len(row) != len(axes) for row in tuples):
                errors.append(f"{interaction_id}: tuples must be a nonempty list matching its axes")
                continue
            requirements = [dict(zip(axes, row, strict=True)) for row in tuples]
            if len({json.dumps(row, sort_keys=True, separators=(",", ":")) for row in requirements}) != len(requirements):
                errors.append(f"{interaction_id}: required tuples must be unique")
                continue
            try:
                realized = [{axis: fact(case["signature"], axis) for axis in axes} for case in cases]
            except KeyError as exc:
                errors.append(f"{interaction_id}: unknown axis path {exc.args[0]}")
                continue
            missing = missing_requirements(requirements, realized)
            if missing:
                errors.append(f"{interaction_id}: canonical suite misses same-case tuples {missing}")

    # Required signatures and interactions above are the contract. A digest of
    # that same authored list added only a manual re-pin when fixtures changed.


def _validate_cell_classifier(contract: dict[str, Any], errors: list[str]) -> None:
    import math

    classifier = contract["cell_metric_classifier"]
    periodic_families = set(contract["validity_rules"]["boundary_conditions"]["periodic"]["cell_metric_family"])
    precedence = classifier.get("precedence", [])
    classes = classifier.get("classes", {})
    if len(precedence) != len(set(precedence)) or set(precedence) != periodic_families:
        errors.append("cell classifier precedence must contain each periodic metric family exactly once")
    if set(classes) != periodic_families:
        errors.append("cell classifier definitions must contain each periodic metric family exactly once")
    expected_tolerances = {"length_rtol", "length_atol", "cosine_atol"}
    tolerances = classifier.get("tolerance_symbols", {})
    if set(tolerances) != expected_tolerances:
        errors.append("cell classifier must declare length_rtol, length_atol, and cosine_atol symbols")
    expected_symbols = {
        "length_rtol": "gdd.engine.wavefunction_fixtures.CELL_METRIC_LENGTH_RTOL",
        "length_atol": "gdd.engine.wavefunction_fixtures.CELL_METRIC_LENGTH_ATOL",
        "cosine_atol": "gdd.engine.wavefunction_fixtures.CELL_METRIC_COSINE_ATOL",
    }
    if tolerances != expected_symbols:
        errors.append("cell classifier tolerance symbols do not name the executable GDD classifier policy")
    if classifier.get("implementation_state") != "statically-verified":
        errors.append("cell classifier must remain statically verified")
    root_three_over_two = math.sqrt(3.0) / 2.0
    samples = {
        "cubic": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "tetragonal": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
        "orthorhombic": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        "hexagonal": [[1.0, 0.0, 0.0], [-0.5, root_three_over_two, 0.0], [0.0, 0.0, 2.0]],
        "rhombohedral": [
            [1.0, 0.0, 0.0],
            [0.5, root_three_over_two, 0.0],
            [0.5, 1.0 / (2.0 * math.sqrt(3.0)), math.sqrt(2.0 / 3.0)],
        ],
        "monoclinic": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.5, 0.0, 3.0]],
        "triclinic": [[1.0, 0.0, 0.0], [0.2, 1.0, 0.0], [0.3, 0.4, 1.0]],
    }
    for expected, cell in samples.items():
        actual = classify_cell_metric(cell)
        if actual != expected:
            errors.append(f"cell classifier returned {actual!r} for reviewed {expected!r} metric")
    try:
        classify_cell_metric([[0.0, 0.0, 0.0]] * 3)
    except ValueError:
        pass
    else:
        errors.append("cell classifier must reject singular cells")


def _validate_sources(errors: list[str]) -> int:
    registry = _load_yaml(SOURCE_PATH, errors)
    if not registry:
        return 0
    roots = registry.get("roots", {})
    sources = registry.get("sources", {})
    if not isinstance(roots, dict) or not roots:
        errors.append("source registry roots must be a nonempty mapping")
        return len(sources) if isinstance(sources, dict) else 0
    if not isinstance(sources, dict) or not sources:
        errors.append("source registry sources must be a nonempty mapping")
        return 0
    root_paths: dict[str, Path] = {}
    for root_id, root in roots.items():
        path = ROOT / root["path"]
        root_paths[root_id] = path
        if not path.is_dir():
            errors.append(f"source root {root_id} is missing: {root['path']}")
    for source_id, source in sources.items():
        root_id = source.get("root")
        if root_id not in root_paths:
            errors.append(f"{source_id}: unknown source root {root_id}")
            continue
        path = root_paths[root_id] / source["path"]
        if not fixture_exists(path):
            errors.append(f"{source_id}: committed source is missing: {path.relative_to(ROOT)}")
    return len(sources)


def _class_string_constants(path: Path, class_name: str) -> set[str]:
    """Every string literal inside one class body of a source file (AST, no import)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        raise ValueError(f"{path.relative_to(ROOT)} must define exactly one {class_name}")
    return {node.value for node in ast.walk(classes[0]) if isinstance(node, ast.Constant) and isinstance(node.value, str)}


def _validate_electron_configuration_contract(contract: dict[str, Any], errors: list[str]) -> int:
    electron = _load_yaml(ELECTRON_CONFIG_PATH, errors)
    if not electron:
        return 0
    expected_top = {
        "schema_version",
        "document",
        "enforcement_map",
        "binding_contract",
        "geminal_matrix_definition",
        "configuration_generation_guarantees",
        "anti_vacuity_rules",
        "enforcement",
    }
    if set(electron) != expected_top:
        errors.append(
            "electron-configuration top-level fields differ: "
            f"missing={sorted(expected_top - set(electron))}, extra={sorted(set(electron) - expected_top)}"
        )
        return 0
    if electron["schema_version"] != 1:
        errors.append("electron-configuration schema_version must be 1")

    binding = electron["binding_contract"]
    if binding.get("field") != "electron_configuration":
        errors.append("electron configurations must bind through the electron_configuration field")
    if binding.get("cardinality") != "exactly-one" or binding.get("implicit_default") != "forbidden":
        errors.append("electron configurations must have exactly one explicit binding per case")

    # The binding/receipt SHAPES live in gdd.engine.evidence; here the declared
    # policy vocabularies are verified against that engine source (AST, no import).
    expected_enums = {
        "modes": {"not-required", "required"},
        "position_sources": {"explicit", "seeded-random"},
        "generation_patterns": {"direct", "rejection-screen", "condition-target", "rank-deficient-target"},
        "conditioning_modes": {"not-applicable", "not-limited", "upper-bounded", "target-range"},
        "violation_actions": {"fail-case", "deterministic-resample"},
        "conditioning_subjects": {"geminal-matrix-G", "polynomial-evaluation-kappa"},
    }
    try:
        engine_constants = _class_string_constants(ROOT / "gdd/engine/evidence.py", "ElectronConfigurationBinding")
    except (OSError, SyntaxError, ValueError) as exc:
        errors.append(str(exc))
        engine_constants = set()
    for field, expected in expected_enums.items():
        declared = binding.get(field, [])
        if not isinstance(declared, list) or set(declared) != expected or len(declared) != len(expected):
            errors.append(f"electron-configuration {field} must be exactly {sorted(expected)}")
            continue
        if engine_constants and not expected <= engine_constants:
            errors.append(
                f"electron-configuration {field} values are unknown to "
                f"gdd.engine.evidence.ElectronConfigurationBinding: {sorted(expected - engine_constants)}"
            )
    policy = binding.get("policy", {})
    if policy.get("unseeded-randomness") != "forbidden":
        errors.append("electron configurations must forbid unseeded randomness")
    if policy.get("hidden-positions-in-pytest-fixture") != "forbidden":
        errors.append("electron positions hidden in a pytest fixture body must be forbidden")
    axes = binding.get("independent_axes", {})
    if axes.get("position_source_kind") != "coordinate-origin-form":
        errors.append("position_source_kind must remain the coordinate-origin axis")
    if axes.get("generation_pattern") != "generation-algorithm-semantics":
        errors.append("generation_pattern must remain the generation-algorithm axis")
    if axes.get("geminal_conditioning_mode") != "accepted-numerical-region":
        errors.append("geminal_conditioning_mode must remain the accepted-region axis")
    pattern_rules = binding.get("generation_pattern_rules", [])
    if (
        not isinstance(pattern_rules, list)
        or {row.get("pattern") for row in pattern_rules if isinstance(row, dict)} != expected_enums["generation_patterns"]
    ):
        errors.append("every electron generation pattern must have one reviewed rule")

    geminal = electron["geminal_matrix_definition"]
    if geminal.get("symbol") != "G" or geminal.get("production-operation") != ("jqmc.determinant.compute_geminal_all_elements"):
        errors.append("cond(G) must name the production geminal matrix G operation")
    condition_definition = geminal.get("condition-number", {})
    if condition_definition.get("name") != "cond_2(G)" or condition_definition.get("definition") != ("spectral-2-norm"):
        errors.append("cond(G) must be the spectral 2-norm condition number cond_2(G)")
    if condition_definition.get("formula") != "sigma_max(G) / sigma_min(G)":
        errors.append("cond_2(G) must be defined from the extreme singular values")
    evaluation_dtype = geminal.get("evaluation-dtype", {})
    if evaluation_dtype != {
        "real-G": "float64",
        "complex-G": "complex128",
        "reported-condition-number": "float64",
    }:
        errors.append("cond_2(G) must preserve full real or complex precision")
    expected_exclusions = {
        "atomic-orbital-overlap-matrix",
        "molecular-orbital-overlap-matrix",
        "optimizer-and-linear-method-matrices",
    }
    if set(geminal.get("exclusions", [])) != expected_exclusions:
        errors.append("cond(G) exclusions must prevent unrelated condition numbers from migrating")

    generation_guarantees = electron["configuration_generation_guarantees"]
    if not isinstance(generation_guarantees, list) or len(generation_guarantees) != 1:
        errors.append("electron configurations must declare exactly one initial generation guarantee")
    else:
        generation = generation_guarantees[0]
        declaration = generation.get("declaration", {})
        expected_node = "tests/test_jqmc_kernel/test_analytic_coverage_evidence.py::test_condition_target_generation_guarantee"
        if generation.get("id") != "CFG-ELECTRON-CONDITION-TARGET-001":
            errors.append("the initial condition-target generation guarantee ID is not pinned")
        if generation.get("evidence", {}).get("pytest_node") != expected_node:
            errors.append("condition-target generation guarantee must pin its executable pytest node")
        if declaration.get("generation_pattern") != "condition-target":
            errors.append("near-singular generation evidence must use condition-target")
        if declaration.get("geminal_conditioning_mode") != "target-range":
            errors.append("near-singular generation evidence must declare a target range")
        required_observations = set(generation.get("required_observations", []))
        expected_observations = {
            "coordinate-snapshot",
            "cond_2(G)",
            "sigma_max(G)",
            "sigma_min(G)",
            "float64-machine-epsilon",
            "cond_2(G)-times-epsilon",
            "generation-attempts",
            "rejected-configuration-count",
            "production-G-route-capture",
        }
        if not expected_observations <= required_observations:
            errors.append("condition-target generation guarantee is missing required runtime observations")

    if contract["shared_contracts"].get("electron_configuration_binding") != ("gdd/ssot/fixtures/electron_configurations.yaml"):
        errors.append("wavefunction contract must reference the electron-configuration contract")
    registration_fields = set(contract["evidence_required_from_each_subscribing_feature"].get("registration_fields", []))
    if "electron-configuration-binding" not in registration_fields:
        errors.append("subscribing feature registration must include an electron-configuration binding")

    static = electron["enforcement"]["static"]
    wavefunction_static = contract["enforcement"]["static"]
    if static["pre_commit_hook"] != wavefunction_static["pre_commit_hook"]:
        errors.append("electron-configuration contract must share the wavefunction pre-commit gate")
    return len(binding.get("position_sources", []))


def _validate_enforcement(contract: dict[str, Any], errors: list[str]) -> None:
    static = contract["enforcement"]["static"]
    precommit = _load_yaml(ROOT / ".pre-commit-config.yaml", errors)
    hook_ids = {hook.get("id") for repo in precommit.get("repos", []) for hook in repo.get("hooks", [])}
    if static["pre_commit_hook"] not in hook_ids:
        errors.append(f"pre-commit does not register {static['pre_commit_hook']}")


def check() -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    contract = _load_yaml(CONTRACT_PATH, errors)
    if not contract:
        return errors, {}
    _validate_document(contract, errors)
    if errors:
        return errors, {}
    _validate_vocabularies(contract, errors)
    _validate_production_bindings(contract, errors)
    _validate_orbital_registry(contract, errors)
    _validate_cell_classifier(contract, errors)
    _validate_suite(contract, errors)
    source_count = _validate_sources(errors)
    electron_source_count = _validate_electron_configuration_contract(contract, errors)
    _validate_enforcement(contract, errors)
    # Keep this gate runnable in the PyYAML-only pre-commit environment.
    from gdd.engine.canonical_contract import ledger_subscription_errors
    from gdd.engine.ledger import load_ledger

    try:
        for ledger_path in sorted((ROOT / "gdd/ssot/guarantees").glob("*.yaml")):
            errors.extend(ledger_subscription_errors(load_ledger(ledger_path), contract["canonical_suite"], ledger_path))
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(f"canonical subscription: {exc}")
    return errors, {
        "cases": len(contract["canonical_suite"]["cases"]),
        "orbitals": len(contract["orbital_signatures"]),
        "sources": source_count,
        "electron_sources": electron_source_count,
    }


def main() -> int:
    errors, counts = check()
    if errors:
        print("Wavefunction fixture GDD contract check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(
        "Wavefunction fixture GDD contract check passed "
        f"({counts['cases']} cases, {counts['orbitals']} orbital signatures, "
        f"{counts['sources']} committed sources, {counts['electron_sources']} electron-position source kinds)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
