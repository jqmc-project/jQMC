"""Derivative requirement expansion (GDD engine, requirement side).

Implements ``derivatives.required_cases.v1`` and the reviewed P01-P15 leaf,
R01-R10 numerical-regime, B20/B24/B31 and C08/C09/C10 disposition inventories:
the complete 1019-obligation analytical-derivative universe, its reviewed
matrix/GFMC digests, and the requirement lookup.  Pytest-side evidence
discovery lives in ``tests/test_jqmc_kernel/analytic_coverage_registration.py``
and is deliberately not imported here: this module must stay importable
without pytest, jqmc, JAX, or any test module.

GFMC requirement rows are derived by PARSING the reviewed evidence source
(``test_gfmc_force_analytic.py``) with a small inert ``pytest.param`` stand-in;
the module is never imported.  The reviewed digests pin the result.
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

import ast
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from pathlib import Path

from gdd.engine.derivative_matrix import _cell_id, _generate_matrix, _matrix_hash

_ROOT = Path(__file__).resolve().parents[2]


class _MarkDecoratorShim:
    """Inert stand-in for ``pytest.mark.*`` inside evaluated case declarations."""

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, _name):
        return self


@dataclass(frozen=True)
class _ParamShim:
    """Inert stand-in for a ``pytest.param`` value: only ``values`` is consumed."""

    values: tuple


class _PytestShim:
    """The only names case-list declarations may resolve from ``pytest``."""

    mark = _MarkDecoratorShim()

    @staticmethod
    def param(*values, marks=(), id=None):  # pytest API shape
        return _ParamShim(tuple(values))


_PYTEST_SHIM = _PytestShim()


class ExpectedOutcome(StrEnum):
    """Production behavior required for one coverage obligation."""

    VALUE = "value"
    NOT_APPLICABLE = "n/a"
    ERROR = "error"


class ProofKind(StrEnum):
    """Primary proof for a value-producing requirement."""

    DIRECT_AD = "direct_ad"
    IDENTITY = "identity"


class SentinelKind(StrEnum):
    """Supplemental independent checks; never a primary coverage proof."""

    DIRECTIONAL_FD = "directional_fd"


@dataclass(frozen=True)
class Requirement:
    """One reviewed obligation that must receive explicit pytest evidence."""

    id: str
    family: str
    category: str
    source_id: str
    description: str
    expected_outcome: ExpectedOutcome
    proof: ProofKind | None
    required_precision: tuple[str, ...]
    expected_leaves: tuple[str, ...] = ()
    liveness_required: bool = False
    sentinels: tuple[SentinelKind, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in ("ln_Psi", "e_L", "e_L_t", "e_L_n"):
            raise ValueError(f"unsupported analytical-derivative requirement family: {self.family}")
        if self.expected_outcome is ExpectedOutcome.VALUE and self.proof is None:
            raise ValueError(f"value requirement {self.id} needs a primary proof")
        if self.expected_outcome is not ExpectedOutcome.VALUE and self.proof is not None:
            raise ValueError(f"non-value requirement {self.id} cannot carry a value proof")
        if self.liveness_required and self.expected_outcome is not ExpectedOutcome.VALUE:
            raise ValueError(f"non-value requirement {self.id} cannot require liveness")
        if not self.required_precision:
            raise ValueError(f"requirement {self.id} has no required precision lane")


@dataclass(frozen=True)
class NumericalCase:
    """One actual pytest parameter combination registered for a requirement."""

    requirement_id: str
    # The registered pytest definition and the callspec values of the case; the
    # exact node id is pytest's own and is never synthesized (issue 789).
    pytest_definition: str
    pytest_parameters: tuple[tuple[str, str], ...]
    builder_available: bool
    registered_precision: tuple[str, ...]
    proof: ProofKind | None
    liveness_checked: bool = False
    input_binding_id: str | None = None
    electron_configuration_binding_id: str | None = None
    comparison_binding_id: str | None = None
    runtime_receipt_operation: str | None = None

    @property
    def pytest_evidence(self) -> str:
        """Human-readable registration: the definition and its callspec values."""
        if not self.pytest_parameters:
            return self.pytest_definition
        return f"{self.pytest_definition} [{', '.join(f'{name}={value}' for name, value in self.pytest_parameters)}]"


@dataclass(frozen=True)
class RegistrationDiff:
    """Static required-vs-registered comparison."""

    missing: tuple[str, ...]
    extra: tuple[str, ...]
    duplicates: tuple[str, ...]
    unavailable: tuple[str, ...]
    precision_mismatch: tuple[str, ...]
    proof_mismatch: tuple[str, ...]
    missing_liveness: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not (
            self.missing
            or self.extra
            or self.duplicates
            or self.unavailable
            or self.precision_mismatch
            or self.proof_mismatch
            or self.missing_liveness
        )


# This is the human-review lock for the existing pairwise/selective-strength-3
# matrix.  Changing the matrix is allowed, but must update this digest explicitly.
APPROVED_MATRIX_SHA256 = "96397978efbb1c3ad288a94346f4c3358cc2f2e7c7a00af95d35b39733c3b735"

# Review lock for the expanded GFMC pytest cases below.  Unlike the ln_Psi/e_L
# covering array, GFMC has several independently parametrized direct-AD and
# estimator-identity functions.  The digest covers function name, every actual
# parameter value and each consumed output axis.  Updating a pytest mark cannot
# silently redefine the reviewed e_L_t/e_L_n universe.
APPROVED_GFMC_MATRIX_SHA256 = "2bdd5dbe8d3942bb4d0f37077f546b0a33f9ee60d368dd2d1f55c13006131f7d"

_QUANTITY_FAMILY = {"ln_Psi": "ln_Psi", "e_L": "e_L"}
_FAMILY_QUANTITY = {family: quantity for quantity, family in _QUANTITY_FAMILY.items()}
_PHYSICAL_AXES = ("c", "r", "R", "eps")
_MATRIX = _generate_matrix()


def matrix_snapshot_matches() -> bool:
    """Whether the reviewed matrix digest still matches production test generation."""
    return _matrix_hash(_MATRIX) == APPROVED_MATRIX_SHA256


@dataclass(frozen=True)
class _GfmcFunctionSpec:
    family: str
    function_name: str
    targets: tuple[str, ...]
    proof: ProofKind


_GFMC_FUNCTION_SPECS = (
    # e_L_t: direct AD of the novar=0 continuum measurement e_L^t forward;
    # the clipped lattice propagation energy is deliberately outside this family.
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_force_analytic_vs_autograd", ("positions",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_stress_analytic_vs_autograd", ("strain",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_force_analytic_vs_autograd_withJ", ("positions",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_stress_analytic_vs_autograd_withJ", ("strain",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_force_analytic_vs_autograd_twist", ("positions",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_stress_analytic_vs_autograd_twist", ("strain",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_force_analytic_vs_autograd_twist_dtn", ("positions",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_stress_analytic_vs_autograd_twist_dtn", ("strain",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_force_analytic_vs_autograd_ecp", ("positions",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_stress_analytic_vs_autograd_ecp", ("strain",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_force_analytic_vs_autograd_pbc", ("positions",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_stress_analytic_vs_autograd_pbc", ("strain",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_force_analytic_vs_autograd_pbc_dltmove", ("positions",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_stress_analytic_vs_autograd_pbc_dltmove", ("strain",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_stress_analytic_vs_autograd_ecp_dtn", ("strain",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_force_analytic_vs_autograd_twist_ae", ("positions",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_stress_analytic_vs_autograd_twist_ae", ("strain",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_force_analytic_vs_autograd_dtn", ("positions",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_stress_analytic_vs_autograd_dtn", ("strain",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_t", "test_gfmc_t_force_analytic_vs_autograd_dtn_ecp", ("positions",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec(
        "e_L_t", "test_gfmc_t_force_stress_analytic_vs_autograd_dtn_pbc", ("positions", "strain"), ProofKind.DIRECT_AD
    ),
    # e_L_n: independent direct-AD representatives.
    _GfmcFunctionSpec("e_L_n", "test_gfmc_n_force_analytic_vs_autograd", ("positions",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_n", "test_gfmc_n_stress_analytic_vs_autograd", ("strain",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_n", "test_gfmc_n_force_analytic_vs_autograd_withJ", ("positions",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_n", "test_gfmc_n_stress_analytic_vs_autograd_withJ", ("strain",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_n", "test_gfmc_n_force_analytic_vs_autograd_ecp", ("positions",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_n", "test_gfmc_n_stress_analytic_vs_autograd_ecp", ("strain",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_n", "test_gfmc_n_force_analytic_vs_autograd_twist_dtn", ("positions",), ProofKind.DIRECT_AD),
    _GfmcFunctionSpec("e_L_n", "test_gfmc_n_stress_analytic_vs_autograd_twist_dtn", ("strain",), ProofKind.DIRECT_AD),
    # B31 transitive coverage: e_L_n and e_L_t share the differentiated
    # use_fast_update=False local-energy foundation; these tests pin the public
    # n/t equality on molecular, complex-twist and DTN representatives.
    _GfmcFunctionSpec("e_L_n", "test_gfmc_n_equals_t_force_and_stress", ("positions", "strain"), ProofKind.IDENTITY),
    _GfmcFunctionSpec("e_L_n", "test_gfmc_n_equals_t_twist", ("positions", "strain"), ProofKind.IDENTITY),
    _GfmcFunctionSpec("e_L_n", "test_gfmc_n_force_stress_dtn_equals_t", ("positions", "strain"), ProofKind.IDENTITY),
)


def _stable_parameter_value(value):
    """Convert the scalar/tuple values used by GFMC parametrizations to JSON."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_stable_parameter_value(item) for item in value]
    return repr(value)


def _parameter_rows_from_marks(marks, function_name: str) -> tuple[dict[str, object], ...]:
    """Expand pytest parametrizations, including stacked marks."""
    rows: list[dict[str, object]] = [{}]
    for mark in marks:
        names = tuple(name.strip() for name in mark[0].split(","))
        values = []
        for raw_value in mark[1]:
            parameter_values = getattr(raw_value, "values", None)
            if parameter_values is not None:
                unpacked = tuple(parameter_values)
            elif len(names) == 1:
                unpacked = (raw_value,)
            else:
                unpacked = tuple(raw_value)
            if len(unpacked) != len(names):
                raise ValueError(f"{function_name}: {names} cannot unpack {unpacked!r}")
            values.append(dict(zip(names, unpacked, strict=True)))
        rows = [{**left, **right} for left, right in product(rows, values)]
    return tuple(
        sorted(
            rows,
            key=lambda row: json.dumps(
                {key: _stable_parameter_value(value) for key, value in sorted(row.items())}, sort_keys=True
            ),
        )
    )


_GFMC_SOURCE = _ROOT / "tests" / "test_jqmc_kernel" / "test_gfmc_force_analytic.py"
_GFMC_PARAMETER_CONSTANTS = {
    "_CASES",
    "_JCASES",
    "_JIDS",
    "_TWIST_KINDS",
    "_ECP_CASES",
    "_ECP_JCASES",
    "_ECP_JIDS",
    "_PBC_CASES",
    "_PBC_DLT_CASES",
    "_H_AE_TWIST_KINDS",
    "_DTN_KINDS",
    "_DTN_AE_CASES",
    "_DTN_ECP_CASES",
    "_DTN_PBC_IDS",
}


def _gfmc_source_tree(source_path: Path | None = None) -> ast.Module:
    source_path = source_path or _GFMC_SOURCE
    return ast.parse(source_path.read_text(), filename=str(source_path))


def _assigned_name(node: ast.Assign) -> str | None:
    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def _gfmc_parameter_namespace(source_path: Path | None = None) -> dict[str, object]:
    """Evaluate only inert case-list declarations, never the JAX test module."""
    namespace: dict[str, object] = {"pytest": _PYTEST_SHIM}
    source_path = source_path or _GFMC_SOURCE
    for node in _gfmc_source_tree(source_path).body:
        keep = isinstance(node, ast.Assign) and _assigned_name(node) in _GFMC_PARAMETER_CONSTANTS
        keep = keep or (isinstance(node, ast.FunctionDef) and node.name == "_dtn_lane")
        if keep:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)
    missing = _GFMC_PARAMETER_CONSTANTS - namespace.keys()
    if missing:
        raise ValueError(f"GFMC parameter declarations missing from source: {sorted(missing)}")
    return namespace


def _gfmc_parameter_rows(function_name: str, source_path: Path | None = None) -> tuple[dict[str, object], ...]:
    source_path = source_path or _GFMC_SOURCE
    namespace = _gfmc_parameter_namespace(source_path)
    functions = {node.name: node for node in _gfmc_source_tree(source_path).body if isinstance(node, ast.FunctionDef)}
    function = functions.get(function_name)
    if function is None:
        raise ValueError(f"GFMC evidence function is absent: {function_name}")
    marks = []
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        try:
            decorator_name = ast.unparse(decorator.func)
        except Exception:
            continue
        if decorator_name != "pytest.mark.parametrize":
            continue
        if len(decorator.args) < 2:
            raise ValueError(f"malformed parametrization on {function_name}")
        names = eval(compile(ast.Expression(decorator.args[0]), str(source_path), "eval"), namespace)
        values = eval(compile(ast.Expression(decorator.args[1]), str(source_path), "eval"), namespace)
        marks.append((names, values))
    return _parameter_rows_from_marks(marks, function_name)


def _gfmc_case_payload(spec: _GfmcFunctionSpec, parameters: dict[str, object], target: str) -> dict[str, object]:
    return {
        "family": spec.family,
        "function": spec.function_name,
        "parameters": {key: _stable_parameter_value(value) for key, value in sorted(parameters.items())},
        "proof": spec.proof.value,
        "target": target,
    }


def _gfmc_case_id(spec: _GfmcFunctionSpec, parameters: dict[str, object], target: str) -> str:
    payload = _gfmc_case_payload(spec, parameters, target)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    category = "direct" if spec.proof is ProofKind.DIRECT_AD else "identity"
    return f"{spec.family}|gfmc-{category}|{spec.function_name}|{target}|case={digest}"


def gfmc_matrix_payload() -> tuple[dict[str, object], ...]:
    payload = []
    for spec in _GFMC_FUNCTION_SPECS:
        for parameters, target in product(_gfmc_parameter_rows(spec.function_name), spec.targets):
            payload.append(_gfmc_case_payload(spec, parameters, target))
    return tuple(sorted(payload, key=lambda item: json.dumps(item, sort_keys=True)))


def gfmc_matrix_hash() -> str:
    encoded = json.dumps(gfmc_matrix_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def gfmc_matrix_snapshot_matches() -> bool:
    return gfmc_matrix_hash() == APPROVED_GFMC_MATRIX_SHA256


def gfmc_function_inventory_diff() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return missing/extra assembled GFMC evidence functions."""
    expected = {spec.function_name for spec in _GFMC_FUNCTION_SPECS}
    discovered = {
        node.name
        for node in _gfmc_source_tree().body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("test_gfmc_")
        and ("analytic_vs_autograd" in node.name or "equals_t" in node.name)
    }
    return tuple(sorted(expected - discovered)), tuple(sorted(discovered - expected))


def gfmc_broad_matrix_covers_both_precisions() -> bool:
    """Whether the e_L_t/e_L_n analytical-vs-autograd matrix runs in both precision lanes."""
    tree = _gfmc_source_tree()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        ):
            continue
        return "skip_mixed_precision" not in ast.unparse(node.value)
    return False


def _gfmc_matrix_requirements() -> tuple[Requirement, ...]:
    requirements = []
    for spec in _GFMC_FUNCTION_SPECS:
        for parameters, target in product(_gfmc_parameter_rows(spec.function_name), spec.targets):
            params = ", ".join(f"{key}={value!r}" for key, value in sorted(parameters.items())) or "unparametrized"
            if spec.function_name == "test_gfmc_t_stress_analytic_vs_autograd_ecp_dtn":
                if parameters.get("nlm") == "dltmove":
                    description = (
                        "molecular ECP+DTN dltmove strain: production tmove and "
                        "production (dltmove - tmove) each equal same-precision AD"
                    )
                else:
                    description = "molecular ECP+DTN tmove strain equals same-precision AD"
                description += f" [{params}]"
            else:
                description = f"{spec.function_name}[{params}]: {target}"
            requirements.append(
                Requirement(
                    id=_gfmc_case_id(spec, parameters, target),
                    family=spec.family,
                    category="gfmc-direct" if spec.proof is ProofKind.DIRECT_AD else "gfmc-identity",
                    source_id="B24" if spec.proof is ProofKind.DIRECT_AD else "B31",
                    description=description,
                    expected_outcome=ExpectedOutcome.VALUE,
                    proof=spec.proof,
                    required_precision=("full", "mixed"),
                )
            )
    return tuple(requirements)


def _matrix_requirement_id(family: str, cell: tuple[str, ...], axis: str) -> str:
    return f"{family}|matrix|{_cell_id(cell)}|axis={axis}"


def _matrix_requirements() -> tuple[Requirement, ...]:
    requirements = []
    for (cell, _heavy), family, axis in product(_MATRIX, _FAMILY_QUANTITY, _PHYSICAL_AXES):
        quantity = _FAMILY_QUANTITY[family]
        requirements.append(
            Requirement(
                id=_matrix_requirement_id(family, tuple(cell), axis),
                family=family,
                category="matrix",
                source_id="fixture-matrix",
                description=f"{_cell_id(cell)}: {quantity}/{axis} analytical == autograd",
                expected_outcome=ExpectedOutcome.VALUE,
                proof=ProofKind.DIRECT_AD,
                required_precision=("full", "mixed"),
            )
        )
    return tuple(requirements)


@dataclass(frozen=True)
class _LeafContract:
    source_id: str
    input_flag: str
    condition: str
    expected_leaves: tuple[str, ...]


_LEAF_CONTRACTS = (
    _LeafContract("P01", "opt_geminal", "SD", ("mo_coefficients_up", "mo_coefficients_dn[unrestricted]")),
    _LeafContract(
        "P02",
        "opt_geminal",
        "AGPn",
        ("mo_coefficients_up", "mo_coefficients_dn[unrestricted]", "lambda_diag", "lambda_unpaired[open-shell]"),
    ),
    _LeafContract("P03", "opt_geminal", "AGP", ("lambda_matrix",)),
    _LeafContract("P04", "opt_lambda_basis_exp", "geminal radial basis", ("lambda_basis_exp",)),
    _LeafContract("P05", "opt_lambda_basis_coeff", "geminal radial basis", ("lambda_basis_coeff",)),
    _LeafContract("P06", "opt_J1_param", "J1", ("j1_param",)),
    _LeafContract("P07", "opt_J2_param", "J2", ("j2_param",)),
    _LeafContract("P08", "opt_J3_param", "J3", ("j3_matrix",)),
    _LeafContract("P09", "opt_J3_basis_exp", "single-leaf J3 radial basis", ("j3_basis_exp",)),
    _LeafContract("P10", "opt_J3_basis_coeff", "single-leaf J3 radial basis", ("j3_basis_coeff",)),
    _LeafContract("P11", "opt_JDTN_u_param", "DTN u", ("j3_poly_alpha",)),
    _LeafContract("P12", "opt_JDTN_chi_param", "DTN chi", ("j3_poly_beta",)),
    _LeafContract("P13", "opt_JDTN_f_param", "DTN f", ("j3_poly_gamma",)),
    _LeafContract(
        "P14",
        "opt_J3_basis_exp",
        "composite J3 GTO/pGTO child",
        ("children[i].radial.exponents", "j3_child{i}_basis_exp"),
    ),
    _LeafContract(
        "P15",
        "opt_J3_basis_coeff",
        "composite J3 GTO/pGTO child",
        ("children[i].radial.coefficients", "j3_child{i}_basis_coeff"),
    ),
)


def _leaf_requirements() -> tuple[Requirement, ...]:
    return tuple(
        Requirement(
            id=f"{family}|leaf|{contract.source_id}",
            family=family,
            category="leaf",
            source_id=contract.source_id,
            description=f"{contract.condition}: {contract.input_flag} publishes every expected leaf",
            expected_outcome=ExpectedOutcome.VALUE,
            proof=ProofKind.DIRECT_AD,
            required_precision=("full", "mixed"),
            expected_leaves=contract.expected_leaves,
            liveness_required=True,
        )
        for family, contract in product(_FAMILY_QUANTITY, _LEAF_CONTRACTS)
    )


_REGIME_FAMILIES = {
    "R01": ("ln_Psi", "e_L", "e_L_t", "e_L_n"),
    "R02": ("ln_Psi", "e_L", "e_L_t", "e_L_n"),
    "R03": ("ln_Psi", "e_L"),
    "R04": ("e_L", "e_L_t", "e_L_n"),
    "R05": ("e_L_t", "e_L_n"),
    "R06": ("e_L_t", "e_L_n"),
    "R07": ("e_L_t", "e_L_n"),
    "R08": ("e_L", "e_L_t", "e_L_n"),
    "R09": ("ln_Psi", "e_L", "e_L_t", "e_L_n"),
    "R10": ("ln_Psi", "e_L", "e_L_t", "e_L_n"),
}


def _regime_requirements() -> tuple[Requirement, ...]:
    return tuple(
        Requirement(
            id=f"{family}|regime|{source_id}",
            family=family,
            category="regime",
            source_id=source_id,
            description=(
                (
                    "R02 condition-target uses the active-precision running inverse and same-precision AD"
                    if source_id == "R02"
                    else f"explicit non-vacuous numerical precondition for {source_id}"
                )
                if family in _FAMILY_QUANTITY
                else (
                    "R02 condition-target: electron gradients equal same-precision AD and the one-nucleus "
                    "gradient satisfies translation invariance"
                    if source_id == "R02"
                    else f"{source_id} precondition plus its GFMC broad-matrix parity/identity companion"
                )
            ),
            expected_outcome=ExpectedOutcome.VALUE,
            proof=ProofKind.DIRECT_AD,
            required_precision=("full", "mixed"),
        )
        for source_id, families in _REGIME_FAMILIES.items()
        for family in families
    )


def _disposition_requirements() -> tuple[Requirement, ...]:
    requirements = [
        Requirement(
            id="ln_Psi|identity|B20|potential-independent",
            family="ln_Psi",
            category="identity",
            source_id="B20",
            description="continuous ln_Psi is identical for the same wavefunction under AE/ECP potential data",
            expected_outcome=ExpectedOutcome.VALUE,
            proof=ProofKind.IDENTITY,
            required_precision=("full", "mixed"),
        )
    ]
    for family in _FAMILY_QUANTITY:
        requirements.extend(
            [
                Requirement(
                    id=f"{family}|n-a|C09|geminal-parameterless-basis",
                    family=family,
                    category="validity",
                    source_id="C09",
                    description="PW/B-spline geminal leaves expose no radial basis parameter block",
                    expected_outcome=ExpectedOutcome.NOT_APPLICABLE,
                    proof=None,
                    required_precision=("full", "mixed"),
                ),
                Requirement(
                    id=f"{family}|n-a|C09|j3-parameterless-basis",
                    family=family,
                    category="validity",
                    source_id="C09",
                    description="PW/B-spline J3 leaves expose no radial basis parameter block",
                    expected_outcome=ExpectedOutcome.NOT_APPLICABLE,
                    proof=None,
                    required_precision=("full", "mixed"),
                ),
                Requirement(
                    id=f"{family}|error|C08|unknown-j1-form",
                    family=family,
                    category="validity",
                    source_id="C08",
                    description="an unknown J1 functional form is rejected explicitly",
                    expected_outcome=ExpectedOutcome.ERROR,
                    proof=None,
                    required_precision=("full", "mixed"),
                ),
                Requirement(
                    id=f"{family}|error|C08|unknown-j2-form",
                    family=family,
                    category="validity",
                    source_id="C08",
                    description="an unknown J2 functional form is rejected explicitly",
                    expected_outcome=ExpectedOutcome.ERROR,
                    proof=None,
                    required_precision=("full", "mixed"),
                ),
            ]
        )
    return tuple(requirements)


def _gfmc_contract_requirements() -> tuple[Requirement, ...]:
    requirements = []
    for family in ("e_L_t", "e_L_n"):
        requirements.extend(
            [
                Requirement(
                    id=f"{family}|output|B24",
                    family=family,
                    category="output",
                    source_id="B24",
                    description="positions and strain APIs publish finite, live dR/dr_up/dr_dn/eps outputs",
                    expected_outcome=ExpectedOutcome.VALUE,
                    proof=ProofKind.DIRECT_AD,
                    required_precision=("full", "mixed"),
                    expected_leaves=("dR", "dr_up", "dr_dn", "eps"),
                    liveness_required=True,
                ),
                Requirement(
                    id=f"{family}|error|C10|unknown-non-local-move",
                    family=family,
                    category="validity",
                    source_id="C10",
                    description="unknown non_local_move values are rejected by both public derivative entries",
                    expected_outcome=ExpectedOutcome.ERROR,
                    proof=None,
                    required_precision=("full", "mixed"),
                ),
            ]
        )
    return tuple(requirements)


REQUIRED_CASES = (
    _matrix_requirements()
    + _leaf_requirements()
    + _regime_requirements()
    + _disposition_requirements()
    + _gfmc_matrix_requirements()
    + _gfmc_contract_requirements()
)


_VALIDITY_REQUIREMENT_PARTS = {
    "geminal-parameterless-basis": ("n-a", "C09"),
    "j3-parameterless-basis": ("n-a", "C09"),
    "unknown-j1-form": ("error", "C08"),
    "unknown-j2-form": ("error", "C08"),
}


def requirements_by_id() -> dict[str, Requirement]:
    """Return the unique requirement lookup, raising on duplicate reviewed IDs."""
    lookup = {requirement.id: requirement for requirement in REQUIRED_CASES}
    if len(lookup) != len(REQUIRED_CASES):
        duplicates = sorted(
            requirement_id
            for requirement_id, count in Counter(requirement.id for requirement in REQUIRED_CASES).items()
            if count != 1
        )
        raise ValueError(f"duplicate required coverage IDs: {duplicates}")
    return lookup
