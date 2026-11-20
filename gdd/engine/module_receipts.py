"""Module evidence contracts and runtime observations of ordinary GDD evidence.

The module ledger is authoritative for the expected case.  This module loads
one contract per registered (guarantee, pytest node) edge, binds a collected
pytest item to its case, and can observe the materialized parameter set,
source artifacts, physics objects, electron coordinates and executed
contractual assertion sites of one call.  Completion of the required cases
uses ``gdd.engine.execution_records``.
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

import hashlib
import dis
import importlib
import json
import os
import re
import sys
import weakref
from dataclasses import dataclass, field
from enum import Enum
from functools import cache, lru_cache
from pathlib import Path
from types import CodeType, FrameType
from typing import Any

import numpy as np
import yaml

from gdd.engine.ledger import load_ledger
from gdd.engine.pinned_fixtures import checkout_path, fixture_digest, fixture_exists
from gdd.engine.condition_checks import capture_condition_inputs, condition_iteration_errors
from gdd.engine.evaluation_conditioning import is_evaluation_kappa_name
from gdd.engine.comparison_paths import bind_helper_call_parents, comparison_scope_contracts
from gdd.engine.tolerance_runtime import _COMPARATORS as _TOLERANCE_COMPARATORS, ToleranceRuntime
from gdd.engine.tolerance_provenance import observation_location
from gdd.engine.canonical_contract import (
    STREAMING_OPERATION,
    canonical_comparisons,
    canonical_precision_coverage_errors,
    canonical_requirement_errors,
)
from gdd.engine.canonical_chain import CanonicalStreamingCapture
from gdd.engine.canonical_use import CanonicalKineticCapture
from gdd.engine.canonical_ecp import CanonicalECPCapture


_TOLERANCE_COMPARATOR_FILES = frozenset(code.co_filename for code in _TOLERANCE_COMPARATORS)
ROOT = Path(__file__).resolve().parents[2]
GUARANTEE_DIR = ROOT / "gdd/ssot/guarantees"
_MISSING = object()
# Identity of a cached value -> (weak reference, committed artifacts it was
# built from, physics-object summaries observed while it was built).  A cache
# hit runs neither the file-opening frame nor the frames that receive the
# Hamiltonian, so the value the cache returns carries both to the later test
# that uses it (the same weak identity-registry pattern as
# ``_RUNTIME_COORDINATE_OBJECTS``).
_RUNTIME_OBJECT_PROVENANCE: dict[int, tuple[weakref.ReferenceType[Any], dict[str, str], dict[str, dict[str, Any]]]] = {}


def _clear_runtime_object_provenance() -> None:
    """Clear process-local cached-object provenance (used by self-tests)."""
    _RUNTIME_OBJECT_PROVENANCE.clear()


def _remember_object_provenance(
    value: Any,
    artifacts: dict[str, str],
    physics: dict[str, dict[str, Any]] | None = None,
) -> None:
    # Enum members are process-wide configuration singletons, not materialized
    # inputs. Tagging a CLI choice with its first file contaminates every later
    # case that uses the same choice, even when the physics objects differ.
    if (not artifacts and not physics) or isinstance(value, Enum):
        return
    identity = id(value)

    def discard(reference: weakref.ReferenceType[Any], *, key: int = identity) -> None:
        current = _RUNTIME_OBJECT_PROVENANCE.get(key)
        if current is not None and current[0] is reference:
            _RUNTIME_OBJECT_PROVENANCE.pop(key, None)

    try:
        reference = weakref.ref(value, discard)
    except TypeError:
        # Holding a strong reference to a large JAX/equinox object for the
        # entire pytest worker would change the suite's memory behaviour.  An
        # unweakrefable object therefore cannot act as a cross-test carrier.
        return
    current = _RUNTIME_OBJECT_PROVENANCE.get(identity)
    alive = current is not None and current[0]() is value
    merged = dict(current[1]) if alive else {}
    merged.update(artifacts)
    merged_physics = dict(current[2]) if alive else {}
    merged_physics.update(physics or {})
    _RUNTIME_OBJECT_PROVENANCE[identity] = (reference, merged, merged_physics)


def _object_provenance(value: Any) -> dict[str, str]:
    if isinstance(value, Enum):
        return {}
    current = _RUNTIME_OBJECT_PROVENANCE.get(id(value))
    if current is None:
        return {}
    if current[0]() is value:
        return dict(current[1])
    _RUNTIME_OBJECT_PROVENANCE.pop(id(value), None)
    return {}


def _object_physics(value: Any) -> dict[str, dict[str, Any]]:
    """Physics-object summaries observed while a cached value was built (empty when unknown)."""
    if isinstance(value, Enum):
        return {}
    current = _RUNTIME_OBJECT_PROVENANCE.get(id(value))
    if current is None or current[0]() is not value:
        return {}
    return dict(current[2])


@lru_cache(maxsize=1 << 16)
def _is_runtime_coordinate_name(name: str) -> bool:
    """Classify electron/evaluation coordinates with the static SSoT rules.

    Pure function of ``name``.  The runtime tracer asks this for every local of
    every traced production call (millions of times while JAX traces a GFMC
    driver), so the result is memoized.
    """
    leaf = name.rsplit(".", 1)[-1].split("=", 1)[0].lstrip("_")
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    normalized_leaf = re.sub(r"[^a-z0-9]+", "_", leaf.lower()).strip("_")
    if normalized_leaf.startswith(("grad_", "gradient_", "interaction_", "interactions_", "mesh_")):
        return False
    if re.search(r"(?:^|_)r_(?:up|dn|cart|carts|elec)(?:_carts?)?_(?:min|max)$", normalized_leaf):
        return False
    if re.search(r"(?:^|_)(?:n|num|number|total)(?:_of)?_(?:electrons?|walkers?)(?:_|$)", normalized):
        return False
    if "core_electron" in normalized or "position_deriv" in normalized:
        return False
    tokens = set(normalized.split("_"))
    if tokens.intersection({"atom", "atomic", "nuclear", "nucleus", "ion", "ionic"}) and not tokens.intersection(
        {"electron", "electrons", "walker", "walkers"}
    ):
        return False
    if leaf.startswith(("R_cart", "R_carts")):
        return False
    if leaf[:1].islower() and re.fullmatch(r"r|ru\d*|rd\d*", leaf.lower()):
        return True
    if normalized_leaf == "r_pbc":
        return True
    if any(re.search(rf"(?:^|_){token}s?(?:_|$)", normalized_leaf) for token in ("r_up", "r_dn", "r_cart", "r_elec", "coords")):
        return True
    if re.search(r"(?:^|_)walker(?:s)?(?:_|$)", normalized_leaf):
        walker_tokens = set(normalized_leaf.split("_"))
        return bool(walker_tokens.intersection({"batch", "state", "projected", "coordinates", "positions"}))
    tokens = set(normalized_leaf.split("_"))
    return bool(
        tokens.intersection({"electron", "electrons"})
        and tokens.intersection({"position", "positions", "configuration", "configurations", "coords"})
    )


# Identity of every array this worker has recorded as electron /
# evaluation coordinates.  A cached builder (``functools.cache``, a module dict
# cache) runs, and is observed, only for the first test that requests it; a
# later test receives the *same* array object under its own local name (for
# example ``ham, up, dn, rt = _inputs(case)``).  Holding that exact object is
# runtime evidence that the later test consumes the recorded coordinates, so
# the receipt is re-recorded for it.  Only arrays recorded from test-frame
# locals are registered (see ``_observe_coordinates(remember=...)``).  Weak
# references only: the registry never extends an array's lifetime, and a dead
# entry can never match a new object.
_RUNTIME_COORDINATE_OBJECTS: dict[int, weakref.ReferenceType[Any]] = {}


def _clear_runtime_coordinate_objects() -> None:
    """Clear the process-local coordinate identity registry (used by self-tests)."""
    _RUNTIME_COORDINATE_OBJECTS.clear()


def _remember_coordinate_object(value: Any) -> None:
    identity = id(value)

    def discard(reference: weakref.ReferenceType[Any], *, key: int = identity) -> None:
        if _RUNTIME_COORDINATE_OBJECTS.get(key) is reference:
            _RUNTIME_COORDINATE_OBJECTS.pop(key, None)

    try:
        _RUNTIME_COORDINATE_OBJECTS[identity] = weakref.ref(value, discard)
    except TypeError:
        return


def _is_remembered_coordinate_object(value: Any) -> bool:
    reference = _RUNTIME_COORDINATE_OBJECTS.get(id(value))
    return reference is not None and reference() is value


# Largest coordinate array (in elements) that is hashed into a receipt.  The
# finest committed evaluation grid is the 2 x 12-node knot Gauss-Legendre rule
# of test_composite_overlap_blocks.py (144 x 168 x 192 points, 13.9M elements);
# larger arrays are not coordinate inputs of any governed test.
_COORDINATE_RECEIPT_MAX_ELEMENTS = 1 << 24


def _is_physics_object(value: Any) -> bool:
    if isinstance(value, Enum):
        return False
    return _is_physics_type(type(value))


@lru_cache(maxsize=1 << 14)
def _is_physics_type(cls: type) -> bool:
    """Type-level classification behind :func:`_is_physics_object` (memoized)."""
    if issubclass(cls, Enum):
        return False
    module = cls.__module__
    identity = f"{module}.{cls.__qualname__}".lower()
    return module.startswith("jqmc") and any(
        token in identity
        for token in (
            "hamiltonian",
            "wavefunction",
            "geminal",
            "determinant",
            "jastrow",
            "orbital",
            "coulomb",
            "structure",
            "gfmc",
            "mcmc",
        )
    )


def _is_observable_container(value: Any) -> bool:
    return _is_physics_object(value) or _is_test_module_type(type(value))


@lru_cache(maxsize=1 << 14)
def _is_test_module_type(cls: type) -> bool:
    return cls.__module__.startswith(("tests.", "test_"))


@lru_cache(maxsize=1 << 14)
def _is_inert_leaf_type(cls: type) -> bool:
    """Whether every structural receipt observer ignores instances of ``cls``.

    For such a value the artifact scan, the physics-object summary, the
    provenance-object carrier and the recursive used-object walk all return
    without recording anything; only identity-keyed provenance lookups (and
    name-keyed coordinate / condition observation) can apply.  Classifying the
    type once lets the tracer skip half a dozen no-op recursive calls per local
    of every traced production call.  The predicate is deliberately
    conservative: strings, paths, containers, enums, h5py handles and every
    ``jqmc`` / test-module type take the full path.
    """
    if issubclass(cls, (str, os.PathLike, dict, tuple, list, set, frozenset, Enum)):
        return False
    if cls in (CanonicalKineticCapture, CanonicalStreamingCapture, CanonicalECPCapture):
        return False
    module = cls.__module__
    if not isinstance(module, str):
        return False
    return not module.startswith(("jqmc", "tests.", "test_", "h5py"))


@lru_cache(maxsize=1 << 14)
def _supports_weakref_type(cls: type) -> bool:
    """Whether ``weakref.ref`` accepts instances of ``cls`` (CPython tp_weaklistoffset).

    Only weak-referenceable objects can carry process-wide provenance in
    ``_RUNTIME_OBJECT_PROVENANCE`` (``_remember_object_provenance`` requires a
    weak reference), so the lookup is skipped for the other leaf types.
    """
    return cls.__weakrefoffset__ != 0


def _is_identity_shared_value(value: Any) -> bool:
    """Whether ``value``'s identity is shared by unrelated code, so it cannot carry provenance.

    Provenance is keyed by object identity.  ``None``, booleans, numbers and
    empty immutable containers are interpreter singletons or freely shared
    constants: a helper returning ``None`` after reading a file would otherwise
    tag every ``None`` local of the case (and, through them, every object it
    builds) with that file.
    """
    if value is None or value is Ellipsis or value is NotImplemented:
        return True
    if isinstance(value, (bool, int, float, complex)) and not isinstance(value, Enum):
        return True
    return isinstance(value, (str, bytes, tuple, frozenset)) and len(value) == 0


def _executes_module_body(frame: Any) -> bool:
    """Whether ``frame`` is a module body, or a ``<genexpr>``/comprehension/lambda frame it is evaluating."""
    current = frame
    while current is not None:
        name = getattr(getattr(current, "f_code", None), "co_name", "")
        if name == "<module>":
            return True
        if not name.startswith("<"):
            return False
        current = getattr(current, "f_back", None)
    return False


def _is_inert_leaf(value: Any) -> bool:
    cls = type(value)
    # ``isinstance`` also consults ``__class__``; objects that spoof it (mocks,
    # weakref proxies) keep the exact per-observer path.
    return value.__class__ is cls and _is_inert_leaf_type(cls)


def _is_abstract_jax_tracer(value: Any) -> bool:
    """Whether ``value`` is a JAX tracer whose ``__array__`` unconditionally raises."""
    jax_core = sys.modules.get("jax._src.core")
    if jax_core is None:
        return False
    tracer_type = getattr(jax_core, "Tracer", None)
    if tracer_type is None or not isinstance(value, tracer_type):
        return False
    return _raises_on_array_conversion(type(value), tracer_type)


@lru_cache(maxsize=1 << 10)
def _raises_on_array_conversion(cls: type, tracer_type: type) -> bool:
    import inspect

    array_method = getattr(cls, "__array__", None)
    if array_method is not getattr(tracer_type, "__array__", None):
        return False
    # Guard the invariant this shortcut relies on instead of assuming it for
    # future JAX versions: the base implementation must only raise.
    try:
        source = inspect.getsource(array_method)
    except (OSError, TypeError):
        return False
    body = [line.strip() for line in source.splitlines()[1:] if line.strip()]
    return body == ["raise TracerArrayConversionError(self)"]


@lru_cache(maxsize=1 << 16)
def _line_global_loads(code: CodeType) -> dict[int, tuple[str, ...]]:
    """Global names each source line of ``code`` loads (``LOAD_GLOBAL``/``LOAD_NAME``)."""
    loads: dict[int, set[str]] = {}
    for instruction in dis.get_instructions(code):
        if instruction.opname not in {"LOAD_GLOBAL", "LOAD_NAME"}:
            continue
        line = instruction.positions.lineno if instruction.positions is not None else None
        if line is None:
            continue
        loads.setdefault(line, set()).add(instruction.argval)
    return {line: tuple(sorted(names)) for line, names in loads.items()}


def _case_locals(frame: FrameType) -> dict[str, Any]:
    """A test frame's locals without memo caches captured from an enclosing scope.

    A factory fixture commonly returns a closure over a mutable cache
    (``cache = {}``; ``def get(case): ... cache[case] = build(case)``).  That
    cache holds the objects of every case built so far in the worker, so its
    provenance is not this call's input: scanning it would bind the union of
    all cached cases' sources to the value returned for one case.  The cached
    value the closure returns (and anything else it computes) is still
    observed through its own locals and return value.
    """
    values = frame.f_locals
    captured = getattr(frame.f_code, "co_freevars", ())
    if not captured:
        return values
    return {name: value for name, value in values.items() if not (name in captured and isinstance(value, (dict, list, set)))}


def _is_catalog_name(name: str) -> bool:
    normalized_name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return any(token in normalized_name.split("_") for token in ("fixtures", "catalog", "registry"))


@lru_cache(maxsize=1 << 16)
def _is_condition_name(name: str) -> bool:
    lowered = name.lstrip("_").lower()
    return re.search(r"(?:^|_)(?:cond|condition|sigma_min)(?:$|_)", lowered) is not None


@cache
def contract_snapshot() -> str:
    """Digest the exact machine-readable contract universe used by a run."""
    digest = hashlib.sha256()
    contract_roots = (ROOT / "gdd/ssot",)
    for path in sorted(
        item
        for contract_root in contract_roots
        for item in contract_root.rglob("*")
        if item.is_file() and item.suffix.lower() != ".md"
    ):
        relative = path.relative_to(ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def file_digest(path: Path) -> str:
    # Committed bytes, or the pinned sha256 of a manifest-pinned pseudopotential.
    return fixture_digest(path)


def canonical_value(value: Any) -> Any:
    """Lossless-enough JSON form for pytest parameter and fixture identity."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return {"float": repr(value)}
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((canonical_value(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, dict):
        return {str(key): canonical_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    try:
        array = np.asarray(value)
    except Exception:
        array = None
    if array is not None and array.dtype != object:
        contiguous = np.ascontiguousarray(array)
        return {
            "array": {
                "shape": list(contiguous.shape),
                "dtype": str(contiguous.dtype),
                "sha256": f"sha256:{hashlib.sha256(contiguous.view(np.uint8)).hexdigest()}",
            }
        }
    if hasattr(value, "__dict__"):
        public = {
            name: canonical_value(item)
            for name, item in vars(value).items()
            if not name.startswith("_") and isinstance(item, (str, int, float, bool, type(None)))
        }
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "public_fields": public}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def _resolved_selector(selector: dict[str, Any], namespace: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve source-bound selector leaves against the collected test module."""
    scope = dict(namespace or {})
    for module_name in ("jqmc", "numpy", "os"):
        if module_name not in scope:
            try:
                scope[module_name] = importlib.import_module(module_name)
            except ImportError:
                pass

    def resolve(value: Any) -> Any:
        if isinstance(value, dict) and set(value) == {"source_expression"}:
            expression = value["source_expression"]
            # The expression is repository-owned test syntax; the test module
            # has already executed it during pytest collection.
            realized = eval(expression, {"__builtins__": {}}, scope)
            return canonical_value(realized)
        if isinstance(value, dict):
            return {key: resolve(child) for key, child in value.items()}
        if isinstance(value, list):
            return [resolve(child) for child in value]
        return value

    return resolve(selector)


def split_nodeid(nodeid: str) -> tuple[str, str | None]:
    """Split a pytest node ID at its FIRST ``[``: the parameter ID may contain ``::`` and ``[``.

    A file path, class name or function name never contains ``[``, while a
    parameter ID may be arbitrary text such as ``fn bad() { rayon::join(a, b); }``
    or ``#[cfg(test)] ...``.  Anchoring the base at the last ``::`` once split such IDs inside the parameter, found no contract for
    the bogus base, and silently emitted no receipt for 58 governed cases.
    """
    match = re.match(r"^([^\[]*::[^\[]+)(?:\[(.*)\])?$", nodeid, flags=re.DOTALL)
    if match is None:
        return nodeid, None
    return match.group(1), match.group(2)


@dataclass(frozen=True)
class EvidenceNodeContract:
    guarantee_id: str
    ledger: str
    pytest_node: str
    claim_refs: tuple[str, ...]
    case_binding: dict[str, Any]
    executable_checks: dict[str, Any] | None = None
    canonical_fixture: dict[str, Any] | None = None
    condition_check_iterations: dict[str, Any] | None = None

    @property
    def cases(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.case_binding["fixture"]["concrete_cases"])

    @property
    def binding_digest(self) -> str:
        return canonical_digest(self.case_binding)

    def case_for(
        self,
        parameter_id: str | None,
        selector: dict[str, Any],
        namespace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if parameter_id is not None:
            matches = [case for case in self.cases if case.get("parameter_id") == parameter_id]
            if len(matches) == 1:
                declared = _resolved_selector(matches[0].get("selector", {}), namespace)
                if declared == selector:
                    return matches[0]
                raise AssertionError(
                    f"{self.pytest_node}: runtime selector differs from the declared case "
                    f"for parameter_id={parameter_id!r}: declared={declared!r}, runtime={selector!r}"
                )
            raise AssertionError(
                f"{self.pytest_node}: collected pytest parameter_id={parameter_id!r} is absent from the SSoT; "
                "a selector-only fallback is forbidden because it would hide collection-ID drift"
            )
        matches = [case for case in self.cases if _resolved_selector(case.get("selector", {}), namespace) == selector]
        if len(matches) == 1:
            return matches[0]
        if len(self.cases) == 1 and selector == self.cases[0].get("selector", {}):
            return self.cases[0]
        raise AssertionError(
            f"{self.pytest_node}: runtime parameter case did not match exactly: "
            f"parameter_id={parameter_id!r}, selector={selector!r}"
        )


def _receipt_precision_modes(contract: Any, case: dict[str, Any]) -> tuple[str, ...]:
    """Return test modes that have an executable comparison in this contract.

    The case binding describes every precision in which pytest executes the
    scientific test.  A single test can, however, provide different evidence
    to different guarantees in those runs.  When a contract authors numerical
    checks for only one precision, only that precision produces receipt
    obligations for the contract; the test still executes in every mode from
    its case binding.
    """
    execution = contract.case_binding["execution"]
    test_modes = tuple(case.get("precision_modes", execution["precision_modes"]))
    checks = getattr(contract, "executable_checks", None)
    if checks is None:
        return test_modes
    if not isinstance(checks, dict) or not checks:
        raise ValueError("executable checks must be a nonempty mapping")
    comparison_modes = [set(check.get("precision", ())) if isinstance(check, dict) else set() for check in checks.values()]
    if not comparison_modes[0] or any(modes != comparison_modes[0] for modes in comparison_modes[1:]):
        raise ValueError("executable checks must share one explicit precision scope")
    receipt_modes = tuple(mode for mode in test_modes if mode in comparison_modes[0])
    if not receipt_modes:
        raise ValueError("executable-check precision scope does not overlap test execution modes")
    return receipt_modes


@cache
def load_node_contracts() -> dict[str, tuple[EvidenceNodeContract, ...]]:
    """Return every module contract of each pytest node, one per guarantee.

    The test/guarantee relation is many-to-many (``gdd/README.md``): a test
    registered under several guarantees has one contract for each of them,
    each with its own guarantee id, case binding and obligations.  Only the
    same node registered twice under one guarantee is a malformed duplicate.
    """
    from gdd.engine.comparison_requirements import case_zone_coverage_errors
    from gdd.engine.parameter_requirements import parameter_case_errors

    contracts: dict[str, list[EvidenceNodeContract]] = {}

    def harvest(value: Any, guarantee_id: str, ledger: str) -> None:
        if isinstance(value, dict):
            node = value.get("pytest_node")
            binding = value.get("case_binding")
            if isinstance(node, str) and isinstance(binding, dict):
                owners = contracts.setdefault(node, [])
                if any(contract.guarantee_id == guarantee_id for contract in owners):
                    raise ValueError(f"pytest node is registered more than once under {guarantee_id}: {node}")
                contract = EvidenceNodeContract(
                    guarantee_id=guarantee_id,
                    ledger=ledger,
                    pytest_node=node,
                    claim_refs=tuple(binding.get("claim_refs", ())),
                    case_binding=binding,
                    executable_checks=value.get("executable_checks"),
                    canonical_fixture=value.get("canonical_fixture"),
                    condition_check_iterations=value.get("condition_check_iterations"),
                )
                owners.append(contract)
                requirement_errors = canonical_requirement_errors(
                    value.get("canonical_fixture"),
                    contract.cases,
                    value.get("executable_checks"),
                    execution=binding.get("execution"),
                )
                requirement_errors.extend(parameter_case_errors(value, contract.cases, execution=binding.get("execution")))
                requirement_errors.extend(case_zone_coverage_errors(value.get("executable_checks"), contract.cases))
                requirement_errors.extend(
                    condition_iteration_errors(
                        value.get("condition_check_iterations"),
                        value.get("executable_checks"),
                        binding.get("comparison", {}).get("typed_observations", ()),
                        binding.get("electron_configuration", {}).get("conditioning_assertions", ()),
                    )
                )
                if requirement_errors:
                    raise ValueError(f"{node}: {'; '.join(requirement_errors)}")
            for child in value.values():
                harvest(child, guarantee_id, ledger)
        elif isinstance(value, list):
            for child in value:
                harvest(child, guarantee_id, ledger)

    # These generated ledgers are large and every xdist worker loads them.
    # ``CSafeLoader`` has the same safe schema as ``SafeLoader`` but avoids
    # making contract collection dominate each scientific pytest process.
    for path in sorted(GUARANTEE_DIR.glob("*.yaml")):
        document = load_ledger(path)
        guarantees = document.get("module_evidence", ()) if path.name == "derivatives.yaml" else document.get("guarantees", ())
        for guarantee in guarantees:
            harvest(guarantee.get("required_evidence", ()), guarantee["id"], path.name)
    canonical_nodes = [
        {
            "pytest_node": contract.pytest_node,
            "canonical_fixture": contract.canonical_fixture,
            "executable_checks": contract.executable_checks,
            "case_binding": contract.case_binding,
        }
        for owners in contracts.values()
        for contract in owners
        if contract.canonical_fixture is not None
    ]
    coverage_errors = canonical_precision_coverage_errors(canonical_nodes)
    if coverage_errors:
        raise ValueError("; ".join(coverage_errors))
    return {node: tuple(owners) for node, owners in contracts.items()}


@cache
def committed_artifact_universe() -> dict[str, str]:
    """Return every registered committed input and its authoritative digest.

    Runtime observation must not be limited to the artifacts that one binding
    already expects: doing so makes an undeclared extra fixture invisible and
    turns the final purported exact-set comparison into a subset check.  The
    generated ledgers already pin nearly every registry digest, so reuse those
    values and hash only a source that is not currently selected by any node.
    """
    declared_digests: dict[str, str] = {}
    for contract in (contract for owners in load_node_contracts().values() for contract in owners):
        for row in contract.case_binding.get("sources", {}).get("artifacts", ()):
            path = row.get("path")
            digest = row.get("sha256")
            if isinstance(path, str) and isinstance(digest, str):
                previous = declared_digests.setdefault(path, digest)
                if previous != digest:
                    raise ValueError(f"conflicting committed-artifact digests for {path}")

    document = yaml.safe_load((ROOT / "gdd/ssot/fixtures/sources.yaml").read_text(encoding="utf-8"))
    roots = document["roots"]
    universe: dict[str, str] = {}
    for row in document["sources"].values():
        path = (Path(roots[row["root"]]["path"]) / row["path"]).as_posix()
        universe[path] = declared_digests.get(path) or file_digest(ROOT / path)
    return universe


@cache
def _current_artifact_digest(path: str) -> str:
    """Hash one materialized registered artifact at most once per worker."""
    return file_digest(ROOT / path)


@cache
def _artifact_registry_index() -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    paths = tuple(sorted(committed_artifact_universe(), key=len, reverse=True))
    by_basename: dict[str, list[str]] = {}
    for path in paths:
        by_basename.setdefault(Path(path).name, []).append(path)
    return paths, {name: tuple(matches) for name, matches in by_basename.items()}


@dataclass
class RuntimeObservation:
    expected_assertion_ids: set[str]
    expected_artifacts: dict[str, str]
    condition_required: bool = False
    assertion_specs: tuple[dict[str, Any], ...] = ()
    condition_expressions: tuple[str, ...] = ()
    executable_checks: dict[str, Any] | None = None
    canonical_fixture: dict[str, Any] | None = None
    condition_check_iterations: dict[str, Any] | None = None
    # The static case binding's electron/evaluation coordinate
    # symbols.  The static engine derives them from the consuming API's
    # coordinate slot (``compute_AOs(aos, points)`` -> ``points``), which a
    # jitted evaluator never exposes as a Python frame at runtime.
    coordinate_symbols: tuple[str, ...] = ()
    canonical_uses: dict[CanonicalKineticCapture | CanonicalStreamingCapture | CanonicalECPCapture, list[dict[str, Any]]] = (
        field(default_factory=dict)
    )
    executed_assertion_ids: set[str] = field(default_factory=set)
    observed_artifacts: dict[str, str] = field(default_factory=dict)
    coordinate_receipts: dict[str, dict[str, Any]] = field(default_factory=dict)
    physics_objects: dict[str, dict[str, Any]] = field(default_factory=dict)
    cond_g_observations: dict[str, Any] = field(default_factory=dict)
    condition_checks: list[dict[str, Any]] = field(default_factory=list)
    # A declared evaluation-kappa screen is observed separately from
    # cond(G); its checks never enter ``condition_checks``.
    evaluation_condition_required: bool = False
    evaluation_condition_expressions: tuple[str, ...] = ()
    evaluation_kappa_observations: dict[str, Any] = field(default_factory=dict)
    evaluation_condition_checks: list[dict[str, Any]] = field(default_factory=list)
    comparison_invocations: list[dict[str, Any]] = field(default_factory=list)
    _relative_test_paths: dict[str, str | None] = field(default_factory=dict, init=False, repr=False)
    _production_paths: dict[str, bool] = field(default_factory=dict, init=False, repr=False)
    _path_kinds: dict[str, tuple[str | None, bool, bool]] = field(default_factory=dict, init=False, repr=False)
    _observation_ids_by_line: dict[tuple[str, int], set[str]] = field(default_factory=dict, init=False, repr=False)
    _assertion_specs_by_id: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _pending_assertions: dict[int, dict[str, tuple[int, int]]] = field(default_factory=dict, init=False, repr=False)
    _pending_condition_inputs: dict[int, dict[str, dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _invocation_scopes: set[str] = field(default_factory=set, init=False, repr=False)
    _active_invocations: dict[int, tuple[str, set[str]]] = field(default_factory=dict, init=False, repr=False)
    _comparison_scopes: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _comparison_conditions: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _condition_lines: dict[tuple[str, int], list[str]] = field(default_factory=dict, init=False, repr=False)
    _branch_roots: dict[int, list[dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _branch_paths: dict[int, list[dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _call_lines: dict[tuple[str, int], list[str]] = field(default_factory=dict, init=False, repr=False)
    _call_roots: dict[int, list[dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _pending_calls: dict[int, dict[str, dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _call_parents: dict[int, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _invocation_exceptions: dict[int, tuple[dict[str, Any], BaseException]] = field(
        default_factory=dict, init=False, repr=False
    )
    _expected_contexts: dict[int, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _propagating: dict[int, tuple[str, BaseException]] = field(default_factory=dict, init=False, repr=False)
    _expected_exit_code: Any = field(default=None, init=False, repr=False)
    _code_positions: dict[Any, tuple] = field(default_factory=dict, init=False, repr=False)
    _physics_object_refs: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _provenance_object_refs: dict[int, weakref.ReferenceType[Any]] = field(default_factory=dict, init=False, repr=False)
    _used_provenance_object_refs: dict[int, weakref.ReferenceType[Any]] = field(default_factory=dict, init=False, repr=False)
    _local_value_provenance: dict[int, tuple[Any, dict[str, str]]] = field(default_factory=dict, init=False, repr=False)
    _artifact_scanned_object_ids: set[int] = field(default_factory=set, init=False, repr=False)
    _frame_artifacts: dict[int, dict[str, str]] = field(default_factory=dict, init=False, repr=False)
    _frame_physics_start: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _global_loads_by_line: dict[int, tuple[CodeType, dict[int, tuple[str, ...]]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _runtime_phase: str = field(default="call", init=False, repr=False)
    _tolerance_runtime: Any = field(default=None, init=False, repr=False)
    _opcode_events_armed: bool = field(default=False, init=False, repr=False)
    _declared_coordinate_names: frozenset[str] = field(default=frozenset(), init=False, repr=False)

    def __post_init__(self) -> None:
        # Only plain identifiers name a test-frame local; keyword and dotted
        # symbols are covered by the name classifier and the member walk.
        self._declared_coordinate_names = frozenset(
            symbol for symbol in self.coordinate_symbols if symbol.isidentifier() and not _is_runtime_coordinate_name(symbol)
        )
        errors = condition_iteration_errors(
            self.condition_check_iterations, self.executable_checks, self.assertion_specs, self.condition_expressions
        )
        if errors:
            raise ValueError("; ".join(errors))
        if self.executable_checks is not None:
            snapshot_sites = (
                tuple(self.executable_checks[name]["site"] for name in canonical_comparisons(self.canonical_fixture).values())
                if self.canonical_fixture
                else ()
            )
            self._tolerance_runtime = ToleranceRuntime(
                self.executable_checks,
                self.assertion_specs,
                snapshot_sites=snapshot_sites,
                chain_sites=(
                    {
                        self.executable_checks[name]["site"]: quantity
                        for quantity, name in canonical_comparisons(self.canonical_fixture).items()
                    }
                    if self.canonical_fixture and self.canonical_fixture["operation"] == STREAMING_OPERATION
                    else {}
                ),
            )
        self._assertion_specs_by_id = {row["id"]: row for row in self.assertion_specs}
        if any(row.get("expected_error_context") for row in self.assertion_specs):
            try:
                from _pytest.raises import RaisesExc as RaisesContext
            except ImportError:
                from _pytest.python_api import RaisesContext

            self._expected_exit_code = RaisesContext.__exit__.__code__
        self._comparison_scopes, self._comparison_conditions = comparison_scope_contracts(self.assertion_specs)
        self._invocation_scopes = {row["function"] for row in self._comparison_scopes.values()}
        for scope in self._comparison_scopes.values():
            for identity, call in scope["calls"].items():
                self._call_lines.setdefault((scope["function"], call["position"][0]), []).append(identity)
        for identity, condition in self._comparison_conditions.items():
            function = self._comparison_scopes[condition["parent"]]["function"]
            for line in range(condition["condition"][0], condition["condition"][1] + 1):
                self._condition_lines.setdefault((function, line), []).append(identity)
        for observation_id in self.expected_assertion_ids:
            spec = self._assertion_specs_by_id.get(observation_id, {})
            if spec.get("expected_error_context"):
                # Entering pytest.raises is not its successful verdict. Only
                # the real context manager's accepting exit certifies it.
                continue
            if "::" in observation_id:
                source, line, _ = observation_location({**spec, "id": observation_id})
                if not spec.get("helper_call"):
                    key = (source, line)
                    self._observation_ids_by_line.setdefault(key, set()).add(observation_id)
                continue
            # Compatibility for explicitly authored legacy scratch specs only.
            # Materialized ledgers are checked against the new source index;
            # a legacy receipt cannot satisfy the new expected identity set.
            if self._assertion_specs_by_id.get(observation_id, {}).get("helper_call"):
                continue
            source, separator, remainder = observation_id.partition(":")
            line_number, line_separator, _claim = remainder.partition(":")
            if not separator or not line_separator:
                continue
            try:
                key = (source, int(line_number))
            except ValueError:
                continue
            self._observation_ids_by_line.setdefault(key, set()).add(observation_id)

    def _complete_assertions(self, frame: FrameType, event: str, arg: Any) -> None:
        """Record normal statement completion, never mere entry into its line.

        In particular, caught assertion failures and multi-line comparison
        calls must not become passing evidence. Pending sites are frame-local
        so recursive/helper calls cannot complete their caller's statement.
        """
        pending = self._pending_assertions.get(id(frame))
        if not pending:
            return
        if event == "exception":
            if isinstance(arg[1], AssertionError):
                rejected = [identity for identity in pending if self._assertion_specs_by_id[identity].get("expected_rejection")]
                if len(rejected) == 1:
                    # The comparator itself rejected inside its pytest.raises
                    # suite; only the accepting exit completes the context.
                    self._expected_contexts[id(frame)] = {
                        "observation": self._assertion_specs_by_id[rejected[0]]["expected_rejection"],
                        "exception": arg[1],
                    }
            self._pending_assertions.pop(id(frame), None)
            self._pending_condition_inputs.pop(id(frame), None)
            return
        for observation_id, (start, end) in tuple(pending.items()):
            if event == "return" or (event == "line" and not start <= frame.f_lineno <= end):
                spec = self._assertion_specs_by_id.get(observation_id, {})
                # A returned predicate is evidence only if it returned true;
                # its caller could catch/ignore a false predicate afterwards.
                if not spec.get("truthy_return") or (event == "return" and bool(arg)):
                    self.executed_assertion_ids.add(observation_id)
                    self._mark_executed(frame, observation_id)
                    check = self._pending_condition_inputs.get(id(frame), {}).pop(observation_id, None)
                    if check is not None:
                        if (
                            check["expression"] in self.evaluation_condition_expressions
                            and check["expression"] not in self.condition_expressions
                        ):
                            self.evaluation_condition_checks.append(check)
                        else:
                            self.condition_checks.append(check)
                del pending[observation_id]
        if not pending:
            self._pending_assertions.pop(id(frame), None)
            self._pending_condition_inputs.pop(id(frame), None)

    def _mark_executed(self, frame: FrameType, observation_id: str) -> None:
        """Add a completed observation to its own suite of the frame's invocation."""
        spec = self._assertion_specs_by_id.get(observation_id, {})
        invocation = self._active_invocations.get(id(frame))
        if invocation is not None and spec.get("required_in_invocation") == invocation[0]:
            invocation[1].add(observation_id)
        conditional = spec.get("conditional_execution")
        if conditional:
            for branch in reversed(self._branch_paths.get(id(frame), ())):
                if f"{branch['condition']}/{branch['arm']}" == conditional["scope"]:
                    branch["executed"].add(observation_id)
                    break

    def _observe_invocation_exception(self, frame: FrameType, arg: Any) -> None:
        """Remember why an invocation frame is unwinding, at the exception's own source site.

        ``failed`` names the declared numerical comparison that raised an
        AssertionError, or the linked comparison-helper call an exceptional
        child invocation propagated through. Any other exception (an error
        raised by the code under test, or while evaluating an operand) has
        no failed comparison and records the line it passed through.
        """
        frame_id = id(frame)
        if frame_id not in self._active_invocations:
            return
        exception_type, exception = arg[0], arg[1]
        propagating = self._propagating.pop(frame_id, None)
        previous = self._invocation_exceptions.get(frame_id)
        if propagating is not None and propagating[1] is exception:
            failed, line = propagating[0], None
        elif previous is not None and previous[1] is exception:
            # Re-raised from a cleanup handler: keep the original site.
            return
        else:
            pending = self._pending_assertions.get(frame_id, {})
            numerical = [
                identity
                for identity in pending
                if self._assertion_specs_by_id[identity].get("kind")
                in {"tolerance-comparison", "exact-comparison", "numerical-bound"}
                and not self._assertion_specs_by_id[identity].get("expected_rejection")
            ]
            if len(numerical) == 1 and isinstance(exception, AssertionError):
                failed, line = numerical[0], None
            else:
                failed, line = None, frame.f_lineno
        record = {
            "type": f"{exception_type.__module__}.{exception_type.__qualname__}",
            "message": str(exception),
            "failed": failed,
        }
        if failed is None:
            record["line"] = line
            # Where the exception was raised: the test-tree frames
            # of its traceback, outermost first (library and comparator frames
            # are omitted). The closure accepts an AssertionError kill only when
            # the innermost of them on a declared observation is a decision site.
            origin = []
            traceback = arg[2] if len(arg) > 2 else None
            while traceback is not None:
                relative = self._relative_test_path(traceback.tb_frame.f_code.co_filename)
                if relative is not None:
                    origin.append({"path": relative, "line": traceback.tb_lineno})
                traceback = traceback.tb_next
            if origin:
                record["origin"] = origin
        self._invocation_exceptions[frame_id] = (record, exception)
        # An expected-error call whose argument evaluation raised
        # before its procedure was entered. The attempt records the outcome;
        # only pytest's accepting exit completes the raises context.
        if failed is None and id(frame) not in self._expected_contexts:
            for identity, attempt in tuple(self._pending_calls.get(frame_id, {}).items()):
                expected_error = self._assertion_specs_by_id[identity]["helper_call"].get("expected_error")
                if expected_error is None or attempt.get("invocation") is not None or "raised" in attempt:
                    continue
                attempt["raised"] = {
                    "type": record["type"],
                    "message": record["message"],
                    "origin": record.get("origin", []),
                }
                self.executed_assertion_ids.add(identity)
                self._mark_executed(frame, identity)
                self._expected_contexts[frame_id] = {"observation": expected_error["observation"], "exception": exception}
                break

    def _observe_comparison_invocation(self, frame: FrameType, event: str, relative: str) -> None:
        """Keep normally returned helper invocations separate, including recursion."""
        if not self._invocation_scopes:
            return
        frame_id = id(frame)
        if event == "call":
            scope = f"{relative}:{frame.f_code.co_firstlineno}:{frame.f_code.co_name}"
            # Harness teardown (fixture finalizers after the verdict) may call a
            # governed predicate for its own bookkeeping; that is not evidence
            # of this test and must not become a comparison invocation.
            if scope in self._invocation_scopes and self._runtime_phase != "teardown":
                self._active_invocations[frame_id] = (scope, set())
                self._branch_roots[frame_id] = []
                self._branch_paths[frame_id] = []
                self._call_roots[frame_id] = []
            # Bind the actual callee to the source call expression in its
            # nearest tracked caller. Wrappers may add untracked frames.
            caller = frame.f_back
            while caller is not None and id(caller) not in self._active_invocations:
                caller = caller.f_back
            if caller is not None and self._pending_calls.get(id(caller)):
                if caller.f_code not in self._code_positions:
                    self._code_positions[caller.f_code] = tuple(caller.f_code.co_positions())
                positions = self._code_positions[caller.f_code]
                position = positions[caller.f_lasti // 2] if 0 <= caller.f_lasti // 2 < len(positions) else None
                for identity, attempt in self._pending_calls.get(id(caller), {}).items():
                    spec = self._assertion_specs_by_id[identity]["helper_call"]
                    if spec["callee"] == scope and tuple(spec["position"]) == position:
                        self._call_parents[frame_id] = attempt
                        break
        elif event == "line" and frame_id in self._active_invocations:
            function = self._active_invocations[frame_id][0]
            path = self._branch_paths[frame_id]
            line = frame.f_lineno

            def contains(bounds):
                return bool(bounds) and bounds[0] <= line <= bounds[1]

            # Complete the previous decision and leave suites before entering
            # the next one. A comparison on the last suite line has already
            # been completed by _complete_assertions above this observer.
            while path:
                branch = path[-1]
                condition = self._comparison_conditions[branch["condition"]]
                if branch["arm"] is None:
                    if contains(condition["condition"]):
                        break
                    branch["arm"] = next((arm for arm in ("body", "orelse") if contains(condition[arm])), "fallthrough")
                if branch["arm"] in {"body", "orelse"} and contains(condition[branch["arm"]]):
                    break
                path.pop()
            parent = f"{path[-1]['condition']}/{path[-1]['arm']}" if path else function
            for identity in self._condition_lines.get((function, line), ()):
                condition = self._comparison_conditions[identity]
                if condition["parent"] != parent:
                    continue
                branch = {"condition": identity, "arm": None, "executed": set(), "branches": [], "calls": []}
                children = path[-1]["branches"] if path else self._branch_roots[frame_id]
                children.append(branch)
                path.append(branch)
                break
        elif event == "exception":
            path = self._branch_paths.get(frame_id)
            if path and path[-1]["arm"] is None:
                # An exception evaluating the guard is not a false guard.
                path[-1]["arm"] = "raised"
        elif event == "return":
            invocation = self._active_invocations.pop(frame_id, None)
            if invocation is None:
                return
            tolerance_calls = self._tolerance_runtime.finish_invocation(frame) if self._tolerance_runtime else []
            roots = self._branch_roots.pop(frame_id)
            self._branch_paths.pop(frame_id)
            calls = self._call_roots.pop(frame_id)
            parent_call = self._call_parents.pop(frame_id, None)
            exceptional = self._invocation_exceptions.pop(frame_id, None)
            expected_error = (
                self._assertion_specs_by_id[parent_call["call"]]["helper_call"].get("expected_error")
                if parent_call is not None
                else None
            )
            self._expected_contexts.pop(frame_id, None)
            self._propagating.pop(frame_id, None)
            self._pending_calls.pop(frame_id, None)
            # A trace return also occurs when unwinding an exception through
            # finally. Checking the opcode avoids counting that as success.
            opcode = dis.opname[frame.f_code.co_code[frame.f_lasti]]
            normal_return = opcode in {
                "RETURN_VALUE",
                "RETURN_CONST",
                "INSTRUMENTED_RETURN_VALUE",
                "INSTRUMENTED_RETURN_CONST",
            }
            if normal_return or (parent_call is not None and exceptional is not None):
                # An exceptional invocation is recorded only when it was
                # attempted from a linked comparison-helper call: either its
                # expected-error call (accepted by pytest.raises) or a caller
                # it propagated into, which must itself be exceptional.

                def serialize(branch):
                    arm = branch["arm"] or "fallthrough"
                    result = {"condition": branch["condition"], "arm": arm, "executed": sorted(branch["executed"])}
                    scope = self._comparison_scopes.get(f"{branch['condition']}/{arm}", {})
                    if scope.get("conditions"):
                        result["branches"] = [serialize(child) for child in branch["branches"]]
                    if scope.get("calls"):
                        result["calls"] = branch["calls"]
                    return result

                record = {"scope": invocation[0], "executed": sorted(invocation[1])}
                if not normal_return:
                    record["exception"] = exceptional[0]
                if any(
                    frame.f_lineno == line and self._comparison_scopes[f"{condition}/{arm}"]["comparisons"]
                    for condition in self._comparison_scopes[invocation[0]]["conditions"]
                    for arm, line in self._comparison_conditions[condition].get("terminal_returns", {}).items()
                ):
                    record["terminal_return"] = frame.f_lineno
                if tolerance_calls:
                    record["tolerance_calls"] = tolerance_calls
                if self._comparison_scopes[invocation[0]]["conditions"]:
                    record["branches"] = [serialize(branch) for branch in roots]
                if self._comparison_scopes[invocation[0]]["calls"]:
                    record["calls"] = calls
                if parent_call is not None:
                    record["caller"] = {"call": parent_call["call"], "invocation": None}
                    parent_call["invocation"] = len(self.comparison_invocations)
                    identity = parent_call["call"]
                    caller = frame.f_back
                    while caller is not None and id(caller) not in self._active_invocations:
                        caller = caller.f_back
                    if normal_return or expected_error is not None:
                        # A completed call, or an expected-error call whose
                        # exception its pytest.raises context must accept.
                        self.executed_assertion_ids.add(identity)
                        if caller is not None:
                            self._mark_executed(caller, identity)
                    if caller is not None and not normal_return:
                        if expected_error is not None:
                            self._expected_contexts[id(caller)] = {
                                "observation": expected_error["observation"],
                                "exception": exceptional[1],
                            }
                        else:
                            self._propagating[id(caller)] = (identity, exceptional[1])
                bind_helper_call_parents(record, self.comparison_invocations, len(self.comparison_invocations))
                self.comparison_invocations.append(record)

    def _observe_expected_exit(self, frame: FrameType, event: str, arg: Any):
        """Witness acceptance by pytest's actual raises context, not a caught error."""
        caller = frame.f_back
        pending = self._expected_contexts.get(id(caller))
        if pending is None:
            return None
        if event == "return":
            self._expected_contexts.pop(id(caller), None)
            spec = self._assertion_specs_by_id[pending["observation"]]
            _, start, _ = observation_location(spec)
            opcode = dis.opname[frame.f_code.co_code[frame.f_lasti]]
            if (
                arg is True
                and opcode in {"RETURN_VALUE", "RETURN_CONST", "INSTRUMENTED_RETURN_VALUE", "INSTRUMENTED_RETURN_CONST"}
                and caller.f_lineno == start
                and any(value is pending["exception"] for value in frame.f_locals.values())
                and id(caller) in self._active_invocations
            ):
                self.executed_assertion_ids.add(pending["observation"])
                self._mark_executed(caller, pending["observation"])
        return self.tracer

    def _observe_helper_call_attempts(self, frame: FrameType, event: str) -> None:
        """Record every attempt, including builtin/noop replacements with no callee trace."""
        frame_id = id(frame)
        if frame_id not in self._active_invocations:
            return
        pending = self._pending_calls.setdefault(frame_id, {})
        if event == "exception":
            pending.clear()
        elif event == "line":
            for identity in tuple(pending):
                start, end, _, _ = self._assertion_specs_by_id[identity]["helper_call"]["position"]
                if not start <= frame.f_lineno <= end:
                    del pending[identity]
            function = self._active_invocations[frame_id][0]
            for identity in self._call_lines.get((function, frame.f_lineno), ()):
                if identity in pending:
                    continue
                attempt = {"call": identity, "invocation": None}
                pending[identity] = attempt
                spec = self._assertion_specs_by_id[identity]
                conditional = spec.get("conditional_execution")
                if conditional:
                    for branch in reversed(self._branch_paths[frame_id]):
                        if f"{branch['condition']}/{branch['arm']}" == conditional["scope"]:
                            branch["calls"].append(attempt)
                            break
                else:
                    self._call_roots[frame_id].append(attempt)

    def begin_phase(self, phase: str) -> None:
        """Select whether observations belong to setup, call, or teardown.

        Setup is allowed to build an eager catalog containing many committed
        fixtures.  Those files are construction provenance, not evidence that
        the current parameter case used every catalog entry.  Test-call
        observations, in contrast, are realized inputs and may be reported
        directly.
        """
        if phase not in {"setup", "call", "teardown"}:
            raise ValueError(f"unsupported runtime receipt phase: {phase}")
        self._runtime_phase = phase

    def tracer(self, frame: FrameType, event: str, arg: Any):
        """Trace only test-source frames without resolving a path per event.

        ``sys.settrace`` invokes its global callback for every Python call made
        by a governed test, including calls in JAX, NumPy and pytest.  Calling
        ``Path.resolve()`` in that hot path made scientific tests tens of times
        slower.  Code filenames are immutable, so cache the inexpensive
        lexical test-tree classification and keep line events local to test
        frames. Assertions are recorded after normal statement completion;
        test/helper locals are observed when entering their assertion sites.
        """
        code = frame.f_code
        if code is self._expected_exit_code:
            return self._observe_expected_exit(frame, event, arg)
        # One cached lookup per event classifies the test-tree, production and
        # comparator paths (under ``sys.settrace`` this front end
        # runs for every Python call of a governed test, JAX internals
        # included, so it must stay a single dict hit).
        kind = self._path_kinds.get(code.co_filename)
        if kind is None:
            kind = self._path_kind(code.co_filename)
        relative = kind[0]
        if event == "opcode":
            # Nested comparisons (e.g. all(isclose(...))) do not necessarily
            # begin at a line event. Only registered test/helper invocations
            # enable these events; scientific production frames never do.
            for observation_id in self._tolerance_runtime.comparison_lines.get((relative, frame.f_lineno), ()):
                self._tolerance_runtime.begin_statement(frame, observation_id)
            return self.tracer
        if relative is None:
            if event != "call":
                # Neither the comparator observer nor production observation
                # acts on a non-call event of a non-test frame.
                return None
            if kind[2] and self._tolerance_runtime is not None and code in _TOLERANCE_COMPARATORS:
                # ``ToleranceRuntime.observe_call`` returns ``None`` for every
                # other code object.  Code objects hash by content (not
                # cached), so membership is only tested for frames whose file
                # defines a registered comparator; every other traced JAX/NumPy
                # frame skips both the hash and the method call.
                callback = self._tolerance_runtime.observe_call(frame, event)
                if callback is not None:
                    return callback
            if kind[1]:
                # Observe construction arguments once, but do not request line
                # tracing inside production/JAX code.  Returning ``None`` here
                # keeps the scientific hot path out of the trace loop.
                artifacts = self._observe_locals(
                    frame.f_locals,
                    artifact_recursive=True,
                    mark_used=self._runtime_phase == "call",
                )
                if artifacts:
                    caller = getattr(frame, "f_back", None)
                    if caller is not None and self._relative_test_path(caller.f_code.co_filename) is not None:
                        self._frame_artifacts.setdefault(id(caller), {}).update(artifacts)
            return None
        if event == "call":
            # Physics objects are recorded in insertion order; the ones added
            # while this test frame is active (its arguments included) are the
            # ones it built or consumed (see _bind_return_physics).
            self._frame_physics_start[id(frame)] = len(self.physics_objects)
        if event == "exception":
            self._observe_invocation_exception(frame, arg)
        self._complete_assertions(frame, event, arg)
        if event == "call" and self._tolerance_runtime is not None:
            self._tolerance_runtime.observe_helper_call(frame)
        self._observe_comparison_invocation(frame, event, relative)
        if event == "call" and self._tolerance_runtime is not None and id(frame) in self._active_invocations:
            frame.f_trace_opcodes = True
            if not self._opcode_events_armed:
                # CPython 3.12+ samples active local traces when settrace is
                # called. On the first call event, CPython 3.13 has not yet
                # installed this callback's return value as f_trace. Publish
                # the local callback before re-arming, preserving the global
                # installed callback (including an external wrapper).
                # Production frames still never request opcode events.
                installed_trace = sys.gettrace()
                if installed_trace is not None:
                    frame.f_trace = self.tracer
                    sys.settrace(installed_trace)
                    self._opcode_events_armed = True
        self._observe_helper_call_attempts(frame, event)
        frame_artifacts: dict[str, str] = {}
        if event == "call":
            # A cached fixture loader may not execute again in this governed
            # case.  Its successfully entered wrapper/builder still names the
            # concrete scalar source it selected; retain that direct (never
            # recursive catalog) provenance on the returned fixture value.
            frame_artifacts.update(self._observe_referenced_globals(frame, observe_artifacts=self._runtime_phase == "setup"))
            helper_name = getattr(frame.f_code, "co_name", "").lower()
            if self._runtime_phase == "call" and any(
                token in helper_name for token in ("build", "load", "read", "fixture", "ham")
            ):
                # A cache wrapper can bypass the file-opening implementation,
                # but entering a concrete test builder with a selected
                # path-valued argument is still runtime provenance.  Keep the
                # scan direct so a catalog argument cannot claim every entry.
                frame_artifacts.update(
                    self._observe_locals(
                        _case_locals(frame),
                        artifact_recursive=False,
                        mark_used=True,
                        test_frame=True,
                    )
                )
        if event == "line":
            observation_ids = self._observation_ids_by_line.get((relative, frame.f_lineno), ())
            for observation_id in observation_ids:
                if self._tolerance_runtime is not None and not frame.f_trace_opcodes:
                    self._tolerance_runtime.begin_statement(frame, observation_id)
                if getattr(code, "co_name", "").startswith("<") and getattr(code, "co_name", "") != "<module>":
                    # A comprehension/lambda frame on the assertion's line is
                    # part of evaluating the statement, not the statement:
                    # its return must not complete an assertion that later
                    # fails in the enclosing frame.
                    continue
                spec = self._assertion_specs_by_id.get(observation_id, {})
                end = spec.get("end_lineno", frame.f_lineno)
                self._pending_assertions.setdefault(id(frame), {})[observation_id] = (frame.f_lineno, end)
                expression = spec.get("expression")
                if expression in self.condition_expressions or expression in self.evaluation_condition_expressions:
                    check = {"observation_id": observation_id, "expression": expression}
                    try:
                        check["inputs"] = capture_condition_inputs(expression, frame.f_locals, getattr(frame, "f_globals", {}))
                        if observation_id in (self.condition_check_iterations or {}):
                            linked = self.condition_check_iterations[observation_id]["comparison"]
                            variable = self.executable_checks[linked]["iterations"]["variable"]
                            value = frame.f_locals.get(variable)
                            if not isinstance(value, str):
                                raise ValueError("conditioning iteration label is not an observed local string")
                            check["iteration"] = {"variable": variable, "value": value}
                    except (ValueError, TypeError, SyntaxError) as exc:
                        check["capture_error"] = str(exc)
                    self._pending_condition_inputs.setdefault(id(frame), {})[observation_id] = check
            if observation_ids:
                # At an assertion site the fixture has necessarily been
                # materialized.  Observing here catches objects created in the
                # test body without tracing every scientific source line.
                frame_artifacts.update(
                    self._observe_locals(
                        _case_locals(frame),
                        artifact_recursive=False,
                        mark_used=self._runtime_phase == "call",
                        test_frame=True,
                    )
                )
                frame_artifacts.update(self._observe_referenced_globals(frame, observe_artifacts=True))
        if event in {"call", "return"}:
            # A path nested inside a static parameter table is a declaration,
            # not proof that the file was materialized.  Test-frame locals are
            # therefore direct-only; production call arguments remain recursive.
            frame_artifacts.update(
                self._observe_locals(
                    _case_locals(frame),
                    artifact_recursive=False,
                    mark_used=False,
                    observe_artifacts=False,
                    test_frame=True,
                )
            )
            if event == "return":
                self._observe_physics_object("return", arg, members=True)
                self._observe_provenance_object(arg)
                if self._runtime_phase == "call":
                    self._observe_used_local_physics_objects(frame)
        if event == "line" and self._runtime_phase == "call":
            frame_artifacts.update(self._observe_line_global_paths(frame))
        if frame_artifacts:
            self._frame_artifacts.setdefault(id(frame), {}).update(frame_artifacts)
        if event == "return":
            self._bind_return_provenance(arg, self._frame_artifacts.pop(id(frame), {}))
            start = self._frame_physics_start.pop(id(frame), None)
            if start is not None and len(self.physics_objects) > start:
                names = list(self.physics_objects)[start:]
                self._bind_return_physics(arg, {name: self.physics_objects[name] for name in names})
        return self.tracer

    def _observe_line_global_paths(self, frame: FrameType) -> dict[str, str]:
        """Record a committed path held by a module global that the executing line loads.

        A cached builder or a test helper may name its committed source only
        through a module constant (``shutil.copy(_FIXTURE, dst)``).  The line
        that loads it executing in this case is the runtime evidence; a
        constant named on a line that does not run (an untaken branch) is not.
        Only direct ``str``/``PathLike`` values are inspected, never tables, and
        only a line that loads exactly one such constant identifies its source.

        A module body is excluded, together with the generator
        expressions and comprehensions it evaluates (``_OK = all(isfile(p)
        for p in ...)``).  It runs under this tracer only when the case imports
        a test module for the first time in the worker (``from tests.x import
        _build_ham`` inside a test), so its lines define path constants for
        every later user; crediting them to whichever case happened to import
        the module first is order-dependent attribution.
        """
        code = frame.f_code
        if not isinstance(code, CodeType) or _executes_module_body(frame):
            return {}
        # Keyed by identity (code objects hash by content); the entry keeps the
        # code object alive so the id cannot be reused while it is cached.
        entry = self._global_loads_by_line.get(id(code))
        if entry is None or entry[0] is not code:
            entry = (code, _line_global_loads(code))
            self._global_loads_by_line[id(code)] = entry
        names = entry[1].get(frame.f_lineno)
        if not names:
            return {}
        namespace = getattr(frame, "f_globals", {})
        paths = [value for name in names if isinstance(value := namespace.get(name, _MISSING), (str, os.PathLike))]
        if len(paths) != 1:
            # Several path constants on one line (``{"a": _A, "b": _B}[key]``,
            # ``_A if flag else _B``) do not say which one this case used.
            return {}
        return self._observe_artifact(paths[0], recursive=False)

    def _observe_used_local_physics_objects(self, frame: FrameType) -> None:
        """Mark physics objects bound directly to a returning test frame's locals as used.

        A ``functools.cache`` hit returns a cached Hamiltonian without running
        any Python frame, and a cached ``jax.jit`` call consumes it without a
        traced production frame.  The helper that bound it to a local name is
        the remaining runtime witness that this case used it.  Containers are
        not descended (a catalog of every case's objects is not a use).
        """
        for value in _case_locals(frame).values():
            if _is_inert_leaf(value) or not _is_physics_object(value) or not _object_provenance(value):
                continue
            try:
                self._used_provenance_object_refs[id(value)] = weakref.ref(value)
            except TypeError:
                continue

    def _path_kind(self, filename: str) -> tuple[str | None, bool, bool]:
        kind = self._path_kinds.get(filename)
        if kind is None:
            kind = (
                self._relative_test_path(filename),
                self._is_production_path(filename),
                filename in _TOLERANCE_COMPARATOR_FILES,
            )
            self._path_kinds[filename] = kind
        return kind

    def needs_call_events(self, filename: str) -> bool:
        """Whether :meth:`tracer` can act on a ``call`` event of code in ``filename``.

        For every other file the tracer returns ``None`` without side effects,
        so a trace backend may stop delivering those events
        (:func:`gdd.engine.runtime_trace.trace_events`).

        The tracer also acts on pytest's raises-context ``__exit__``
        (:meth:`_observe_expected_exit` witnesses an accepted expected
        exception there). Its file is neither test-tree, production nor
        comparator source, so it is named here; otherwise the CPython 3.12
        bridge disables that call event and no expected-error observation of
        a governed ``pytest.raises`` suite can ever execute.
        """
        if self._expected_exit_code is not None and filename == self._expected_exit_code.co_filename:
            return True
        relative, production, comparator = self._path_kind(filename)
        return relative is not None or production or comparator

    def _relative_test_path(self, filename: str) -> str | None:
        cached = self._relative_test_paths.get(filename, _MISSING)
        if cached is not _MISSING:
            return cached
        normalized = filename.replace(os.sep, "/")
        root = ROOT.as_posix().rstrip("/")
        prefix = f"{root}/tests/"
        if normalized.startswith(prefix):
            relative: str | None = normalized[len(root) + 1 :]
        elif normalized.startswith("tests/"):
            relative = normalized
        else:
            relative = None
        self._relative_test_paths[filename] = relative
        return relative

    def _is_production_path(self, filename: str) -> bool:
        cached = self._production_paths.get(filename)
        if cached is not None:
            return cached
        normalized = filename.replace(os.sep, "/")
        root = ROOT.as_posix().rstrip("/")
        is_production = normalized.startswith((f"{root}/jqmc/", "jqmc/"))
        self._production_paths[filename] = is_production
        return is_production

    def _observe_referenced_globals(
        self,
        frame: FrameType,
        *,
        observe_artifacts: bool = False,
        record_artifacts: bool = True,
    ) -> dict[str, str]:
        """Observe only globals whose names the executing function references.

        This captures ``_XML``, ``_FIXTURE_DIR`` and explicit coordinate arrays
        such as ``_R_PBC`` without recursively treating a fixture catalog whose
        entries merely *name* files as evidence that every file was opened.
        """
        artifacts: dict[str, str] = {}
        for name in getattr(frame.f_code, "co_names", ()):
            if name not in frame.f_globals:
                continue
            value = frame.f_globals[name]
            if observe_artifacts:
                artifacts.update(
                    self._observe_artifact(
                        value,
                        recursive=False,
                        record=record_artifacts,
                    )
                )
            if _is_runtime_coordinate_name(name):
                self._observe_coordinates(name, value)
            self._observe_physics_object(name, value)
            self._observe_provenance_object(value)
        return artifacts

    def _observe_callable_globals(
        self,
        function: Any,
        *,
        observe_artifacts: bool = False,
        record_artifacts: bool = True,
    ) -> dict[str, str]:
        function = getattr(function, "__func__", function)
        code = getattr(function, "__code__", None)
        namespace = getattr(function, "__globals__", None)
        if code is None or not isinstance(namespace, dict):
            return {}
        frame = type("_ReferencedGlobals", (), {"f_code": code, "f_globals": namespace})()
        return self._observe_referenced_globals(
            frame,
            observe_artifacts=observe_artifacts,
            record_artifacts=record_artifacts,
        )

    def observe_pytest_item(self, item: Any) -> None:
        """Bind cached pytest fixtures to the item that actually requested them.

        A module/session fixture may have been constructed for an earlier item,
        so its Python frame does not necessarily execute again.  Inspecting only
        the definitions of fixtures in this item's resolved fixture closure keeps
        the provenance case-local while avoiding an order-dependent receipt.
        """
        self._observe_callable_globals(getattr(item, "function", getattr(item, "obj", None)))
        funcargs = getattr(item, "funcargs", {})
        fixture_info = getattr(item, "_fixtureinfo", None)
        definitions = getattr(fixture_info, "name2fixturedefs", {})
        for fixture_name in getattr(item, "fixturenames", ()):
            fixture_definitions = definitions.get(fixture_name, ())
            if not fixture_definitions:
                continue
            # Pytest resolves the last definition after fixture overrides.
            # When a module/session fixture was cached before this governed
            # item, its setup frame cannot attach source provenance to the
            # value again.  A *single* scalar registered artifact referenced
            # directly by the selected fixture definition is nevertheless an
            # exact binding: this item requested that definition and pytest
            # supplied its resolved value.  Do not infer from containers or
            # from definitions naming multiple possible sources.
            fixture_definition = fixture_definitions[-1]
            artifacts = self._observe_callable_globals(
                getattr(fixture_definition, "func", None),
                observe_artifacts=True,
                record_artifacts=False,
            )
            if isinstance(funcargs, dict) and fixture_name in funcargs and len(artifacts) == 1:
                self._bind_return_provenance(funcargs[fixture_name], artifacts)
                self.observed_artifacts.update(artifacts)
        if isinstance(funcargs, dict):
            # These are pytest's resolved values for this exact item, not the
            # fixture catalog.  Recover provenance attached while their setup
            # functions ran, including immutable temporary path values.
            self.observed_artifacts.update(
                self._observe_locals(
                    funcargs,
                    artifact_recursive=False,
                    mark_used=False,
                    observe_artifacts=False,
                )
            )

    def _observe_locals(
        self,
        values: dict[str, Any],
        *,
        artifact_recursive: bool = True,
        mark_used: bool = True,
        observe_artifacts: bool = True,
        test_frame: bool = False,
    ) -> dict[str, str]:
        artifacts: dict[str, str] = {}
        record_used = mark_used and self._runtime_phase == "call"
        declared = self._declared_coordinate_names if test_frame else frozenset()
        for name, value in values.items():
            if _is_inert_leaf(value):
                # Exact fast path: for an inert leaf the artifact
                # scan, physics summary and provenance carrier are no-ops, and
                # the provenance/used-object walks reduce to the identity-keyed
                # lookups below.  Name-keyed observers still run unchanged.
                provenance = _object_provenance(value) if _supports_weakref_type(type(value)) else {}
                if provenance and record_used:
                    try:
                        self._used_provenance_object_refs[id(value)] = weakref.ref(value)
                    except TypeError:
                        pass
                local = self._local_value_provenance.get(id(value))
                if local is not None and local[0] is value:
                    provenance.update(local[1])
                artifacts.update(provenance)
                if _is_runtime_coordinate_name(name) or name in declared or _is_remembered_coordinate_object(value):
                    self._observe_coordinates(name, value, remember=test_frame)
                if self.condition_required:
                    if _is_condition_name(name):
                        self._observe_condition(name, value)
                    self._observe_condition_object(name, value)
                if self.evaluation_condition_required and is_evaluation_kappa_name(name):
                    self._observe_evaluation_kappa(name, value)
                continue
            if observe_artifacts:
                artifacts.update(self._observe_artifact(value, recursive=artifact_recursive))
            artifacts.update(self._provenance_from_value(value))
            if _is_runtime_coordinate_name(name) or name in declared or _is_remembered_coordinate_object(value):
                self._observe_coordinates(name, value, remember=test_frame)
            if self.condition_required and _is_condition_name(name):
                self._observe_condition(name, value)
            if self.condition_required:
                self._observe_condition_object(name, value)
            if self.evaluation_condition_required and is_evaluation_kappa_name(name):
                self._observe_evaluation_kappa(name, value)
            self._observe_physics_object(name, value, members=test_frame)
            self._observe_provenance_object(value)
            if record_used:
                self._observe_used_provenance_object(name, value)
        return artifacts

    def _observe_artifact(
        self,
        value: Any,
        *,
        recursive: bool = True,
        record: bool = True,
    ) -> dict[str, str]:
        candidates: list[str] = []
        visited: set[int] = set()

        def collect(candidate: Any, depth: int = 0) -> None:
            if _is_inert_leaf(candidate):
                return
            if isinstance(candidate, (str, os.PathLike)):
                candidates.append(os.fspath(candidate))
                return
            module = type(candidate).__module__
            if module.startswith("h5py"):
                try:
                    owner = getattr(candidate, "file", candidate)
                    filename = getattr(owner, "filename", None)
                except (AttributeError, RuntimeError, ValueError):
                    filename = None
                if isinstance(filename, (str, os.PathLike)):
                    candidates.append(os.fspath(filename))
                return
            if not recursive:
                return
            if depth >= 4:
                return
            identity = id(candidate)
            if identity in visited:
                return
            visited.add(identity)
            if _is_observable_container(candidate):
                if identity in self._artifact_scanned_object_ids:
                    return
                self._artifact_scanned_object_ids.add(identity)
            if isinstance(candidate, dict) and len(candidate) <= 32:
                for key, item in candidate.items():
                    collect(key, depth + 1)
                    collect(item, depth + 1)
            elif isinstance(candidate, (tuple, list, set, frozenset)) and len(candidate) <= 32:
                for item in candidate:
                    collect(item, depth + 1)
            elif _is_observable_container(candidate):
                try:
                    attributes = vars(candidate)
                except TypeError:
                    attributes = {}
                if len(attributes) <= 64:
                    for item in attributes.values():
                        collect(item, depth + 1)
                # setup_class commonly stores the fixture on the class rather
                # than the instance.  Read data attributes only; never invoke
                # descriptors or properties during receipt collection.
                if type(candidate).__module__.startswith(("tests.", "test_")):
                    class_attributes = vars(type(candidate))
                    if len(class_attributes) <= 128:
                        for item in class_attributes.values():
                            if not callable(item) and not isinstance(item, (staticmethod, classmethod, property)):
                                collect(item, depth + 1)

        collect(value)
        if not candidates:
            return {}
        registry_paths, basenames = _artifact_registry_index()
        observed: dict[str, str] = {}

        root = os.path.normpath(os.fspath(ROOT))
        for candidate in candidates:
            # A path in the session mirror of tests/qe_example_files names the checkout path it mirrors.
            lexical = checkout_path(os.path.normpath(candidate))
            if os.path.isabs(lexical):
                try:
                    common = os.path.commonpath((root, lexical))
                except ValueError:
                    common = ""
                relative = os.path.relpath(lexical, root).replace(os.sep, "/") if common == root else None
            else:
                relative = lexical.replace(os.sep, "/").removeprefix("./")

            matched = next(
                (
                    registered
                    for registered in registry_paths
                    if relative == registered or (relative is not None and relative.startswith(f"{registered}/"))
                ),
                None,
            )
            if matched is None and "/" not in lexical and "\\" not in lexical:
                unique = basenames.get(Path(lexical).name, ())
                matched = unique[0] if len(unique) == 1 else None
            if matched is None:
                continue
            if fixture_exists(ROOT / matched):
                digest = _current_artifact_digest(matched)
                observed[matched] = digest
                if record and self._runtime_phase == "call":
                    self.observed_artifacts[matched] = digest
        return observed

    def _provenance_from_value(
        self,
        value: Any,
        depth: int = 0,
        visited: set[int] | None = None,
    ) -> dict[str, str]:
        """Collect already-bound provenance without declaring the value used."""
        if depth > 4:
            return {}
        if _is_inert_leaf(value):
            artifacts = _object_provenance(value) if _supports_weakref_type(type(value)) else {}
            local = self._local_value_provenance.get(id(value))
            if local is not None and local[0] is value:
                artifacts.update(local[1])
            return artifacts
        if visited is None:
            visited = set()
        identity = id(value)
        if identity in visited:
            return {}
        visited.add(identity)
        artifacts = _object_provenance(value)
        local = self._local_value_provenance.get(identity)
        if local is not None and local[0] is value:
            artifacts.update(local[1])
        if isinstance(value, dict):
            if len(value) <= 64:
                for child in value.values():
                    artifacts.update(self._provenance_from_value(child, depth + 1, visited))
            return artifacts
        if isinstance(value, (tuple, list, set, frozenset)):
            if len(value) <= 64:
                for child in value:
                    artifacts.update(self._provenance_from_value(child, depth + 1, visited))
            return artifacts
        if _is_observable_container(value):
            try:
                attributes = vars(value)
            except TypeError:
                attributes = {}
            if len(attributes) <= 64:
                for child in attributes.values():
                    artifacts.update(self._provenance_from_value(child, depth + 1, visited))
        return artifacts

    def _bind_return_physics(self, value: Any, physics: dict[str, dict[str, Any]]) -> None:
        """Carry the physics objects a test helper's own frame observed on the value it returns.

        A cache builder (``_TWO_TWIST_CACHE[kind] = _run_two_twist_gfmc(kind)``)
        constructs and consumes its Hamiltonians once; a later test receives only
        the cached result.  The summaries ride on the returned jqmc objects (and
        test-defined records) through the provenance registry and are recovered
        in :meth:`finalize` when the later test uses that object.
        """
        if not physics or isinstance(value, Enum):
            return
        visited: set[int] = set()

        def bind(candidate: Any, depth: int = 0) -> None:
            if depth > 4 or id(candidate) in visited:
                return
            visited.add(id(candidate))
            if type(candidate).__module__.startswith("jqmc") or _is_observable_container(candidate):
                _remember_object_provenance(candidate, {}, physics)
            if isinstance(candidate, dict):
                children = list(candidate.values()) if len(candidate) <= 32 else []
            elif isinstance(candidate, (tuple, list)):
                children = list(candidate) if len(candidate) <= 32 else []
            else:
                children = []
            for child in children:
                bind(child, depth + 1)

        bind(value)

    def _bind_return_provenance(self, value: Any, artifacts: dict[str, str]) -> None:
        """Attach one helper frame's sources to its returned runtime value.

        A leaf loader normally returns one object from one committed source.
        An eager fixture-catalog builder returns many objects from many sources;
        binding that catalog-wide union to every child is exactly the
        cross-case contamination this receipt exists to reject.  Child objects
        already retain their leaf-loader provenance, so only small source sets
        are propagated recursively.
        """
        if not artifacts or isinstance(value, Enum) or _is_identity_shared_value(value):
            return
        # Immutable fixture results (most notably a temporary HDF5 path) are
        # not weak-referenceable.  Keep only those values for this observation;
        # retaining a weak-referenceable production object here would change
        # its lifetime and invalidate optimizer batch-release guarantees.
        try:
            weakref.ref(value)
        except TypeError:
            self._local_value_provenance[id(value)] = (value, dict(artifacts))
        _remember_object_provenance(value, artifacts)
        if len(artifacts) > 3:
            return

        visited: set[int] = set()

        def bind(candidate: Any, depth: int = 0) -> None:
            if depth > 4:
                return
            identity = id(candidate)
            if identity in visited:
                return
            visited.add(identity)
            if type(candidate).__module__.startswith("jqmc") or _is_observable_container(candidate):
                _remember_object_provenance(candidate, artifacts)
            if isinstance(candidate, dict):
                if len(candidate) <= 32:
                    for child in candidate.values():
                        bind(child, depth + 1)
                return
            if isinstance(candidate, (tuple, list, set, frozenset)):
                if len(candidate) <= 32:
                    for child in candidate:
                        bind(child, depth + 1)
                return
            if _is_observable_container(candidate):
                try:
                    attributes = vars(candidate)
                except TypeError:
                    attributes = {}
                if len(attributes) <= 64:
                    for child in attributes.values():
                        bind(child, depth + 1)

        bind(value)

    def _observe_used_provenance_object(
        self,
        name: str,
        value: Any,
        depth: int = 0,
        visited: set[int] | None = None,
    ) -> None:
        """Remember objects selected by the test call, excluding eager catalogs."""
        if isinstance(value, Enum):
            return
        if depth > 5:
            return
        if _is_inert_leaf(value):
            if _supports_weakref_type(type(value)) and _object_provenance(value):
                try:
                    self._used_provenance_object_refs[id(value)] = weakref.ref(value)
                except TypeError:
                    pass
            return
        if visited is None:
            visited = set()
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)

        provenance = _object_provenance(value)
        if provenance or type(value).__module__.startswith("jqmc") or _is_physics_object(value):
            try:
                self._used_provenance_object_refs[identity] = weakref.ref(value)
            except TypeError:
                pass

        if depth == 0 and _is_catalog_name(name):
            return
        if isinstance(value, dict):
            if len(value) <= 16:
                for key, child in value.items():
                    self._observe_used_provenance_object(f"{name}.{key}", child, depth + 1, visited)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            if len(value) <= 16:
                for index, child in enumerate(value):
                    self._observe_used_provenance_object(f"{name}[{index}]", child, depth + 1, visited)
            return
        if _is_observable_container(value):
            try:
                attributes = vars(value)
            except TypeError:
                attributes = {}
            if len(attributes) <= 64:
                for attribute, child in attributes.items():
                    self._observe_used_provenance_object(f"{name}.{attribute}", child, depth + 1, visited)

    def _observe_provenance_object(
        self,
        value: Any,
        depth: int = 0,
        visited: set[int] | None = None,
    ) -> None:
        """Retain weakrefable jqmc objects as cached-fixture provenance carriers."""
        if isinstance(value, Enum) or _is_inert_leaf(value):
            return
        if depth > 4:
            return
        if visited is None:
            visited = set()
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(value, dict):
            if len(value) <= 64:
                for child in value.values():
                    self._observe_provenance_object(child, depth + 1, visited)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            if len(value) <= 64:
                for child in value:
                    self._observe_provenance_object(child, depth + 1, visited)
            return
        if type(value).__module__.startswith("jqmc"):
            try:
                weakref.ref(value)
            except TypeError:
                return
            # Runtime receipt collection must be observational.  A strong
            # reference here changes the lifetime of short-lived production
            # values (for example an OptimizationBatch that must be released
            # before the next sampling step).  Cached fixture objects remain
            # reachable through their real owner and can still carry
            # provenance through this weak reference.
            self._provenance_object_refs[identity] = weakref.ref(value)

    def _observe_coordinates(
        self,
        name: str,
        value: Any,
        depth: int = 0,
        visited: set[int] | None = None,
        remember: bool = False,
    ) -> None:
        if depth > 4:
            return
        if visited is None:
            visited = set()
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(value, dict):
            for key, child in value.items():
                self._observe_coordinates(f"{name}.{key}", child, depth + 1, visited, remember)
            return
        if _is_abstract_jax_tracer(value):
            # ``np.asarray`` of a JAX tracer always raises
            # ``TracerArrayConversionError`` (caught below as "no array"), and
            # a tracer is neither a tuple/list nor an observable container, so
            # the observation is a no-op.  Skipping it avoids building that
            # exception for every coordinate-named local of every traced
            # production call.
            return
        try:
            array = np.asarray(value)
        except Exception:
            array = None
        if array is None or array.dtype == object:
            if isinstance(value, (tuple, list)) and len(value) <= 64:
                for index, child in enumerate(value):
                    self._observe_coordinates(f"{name}[{index}]", child, depth + 1, visited, remember)
            elif _is_observable_container(value):
                try:
                    attributes = vars(value)
                except TypeError:
                    attributes = {}
                for attribute, child in attributes.items():
                    self._observe_coordinates(f"{name}.{attribute}", child, depth + 1, visited, remember)
            return
        if array.ndim < 1 or array.shape[-1] != 3 or array.size > _COORDINATE_RECEIPT_MAX_ELEMENTS:
            return
        contiguous = np.ascontiguousarray(array)
        self.coordinate_receipts[name] = {
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "finite": bool(np.all(np.isfinite(contiguous))),
            "sha256": f"sha256:{hashlib.sha256(contiguous.view(np.uint8)).hexdigest()}",
        }
        if remember:
            # Only a test-frame coordinate is carried by identity; a production
            # helper that also wraps nuclear positions (``_wrap_into_cell(r_cart)``)
            # must not turn those positions into electron coordinates elsewhere.
            _remember_coordinate_object(value)

    def _observe_condition(self, name: str, value: Any) -> None:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return
        if array.dtype == object or array.size == 0 or not np.all(np.isfinite(array)):
            return
        if array.size == 1:
            self.cond_g_observations[name] = float(array.reshape(-1)[0])
            return
        contiguous = np.ascontiguousarray(array)
        self.cond_g_observations[name] = {
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "minimum": float(np.min(contiguous)),
            "maximum": float(np.max(contiguous)),
            "sha256": f"sha256:{hashlib.sha256(contiguous.view(np.uint8)).hexdigest()}",
        }

    def _observe_evaluation_kappa(self, name: str, value: Any) -> None:
        """Aggregate every observed value of a declared evaluation-kappa metric."""
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return
        if array.dtype.kind not in "iuf" or array.size == 0 or not np.all(np.isfinite(array)):
            return
        values = array.reshape(-1).astype(np.float64)
        previous = self.evaluation_kappa_observations.get(name)
        if previous is None:
            self.evaluation_kappa_observations[name] = {
                "dtype": str(array.dtype),
                "observations": int(values.size),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
            return
        previous["observations"] += int(values.size)
        previous["minimum"] = min(previous["minimum"], float(np.min(values)))
        previous["maximum"] = max(previous["maximum"], float(np.max(values)))
        if previous["dtype"] != str(array.dtype):
            previous["dtype"] = "mixed"

    def _observe_condition_object(self, name: str, value: Any) -> None:
        """Observe explicit condition-target records without knowing their test helper type."""
        for attribute in (
            "cond_g",
            "sigma_min_g",
            "sigma_max_g",
            "machine_epsilon",
            "cond_times_epsilon",
            "matrix_dimension",
            "generation_attempts",
            "rejected_configuration_count",
            "accepted_draw",
            "draw_index",
        ):
            if hasattr(value, attribute):
                self._observe_condition(f"{name}.{attribute}", getattr(value, attribute))
        if hasattr(value, "matrix_dtype"):
            self.cond_g_observations[f"{name}.matrix_dtype"] = str(value.matrix_dtype)
        for attribute in ("G", "g", "geminal", "geminal_matrix"):
            if not hasattr(value, attribute):
                continue
            try:
                matrix = np.asarray(getattr(value, attribute))
            except Exception:
                continue
            if matrix.dtype == object or matrix.ndim != 2:
                continue
            self.cond_g_observations[f"{name}.{attribute}.metadata"] = {
                "dimension": list(matrix.shape),
                "dtype": str(matrix.dtype),
            }

    def _observe_physics_object(
        self,
        name: str,
        value: Any,
        depth: int = 0,
        visited: set[int] | None = None,
        members: bool = False,
    ) -> None:
        if (
            type(value) in (CanonicalKineticCapture, CanonicalStreamingCapture, CanonicalECPCapture)
            and self.canonical_fixture is not None
        ):
            # Keep each explicit test capture distinct even if a temporary
            # capture would otherwise die and its Python id be reused.
            self.canonical_uses[value] = json.loads(json.dumps(value.records))
            return
        if _is_inert_leaf(value):
            return
        if depth > 5:
            return
        if visited is None:
            visited = set()
        identity_value = id(value)
        if identity_value in visited:
            return
        visited.add(identity_value)
        if isinstance(value, dict):
            if len(value) <= 64:
                # Snapshot: classifying a member may allocate (and collect),
                # and a collected object's callback may resize a live cache.
                for key, child in list(value.items()):
                    if members:
                        self._observe_coordinate_member(f"{name}.{key}", child)
                    self._observe_physics_object(f"{name}.{key}", child, depth + 1, visited, members)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            if len(value) <= 64:
                for index, child in enumerate(value):
                    self._observe_physics_object(f"{name}[{index}]", child, depth + 1, visited, members)
            return
        module = type(value).__module__
        class_name = type(value).__qualname__
        if _is_physics_object(value):
            if identity_value in self._physics_object_refs:
                return
            summary: dict[str, Any] = {"type": f"{module}.{class_name}"}
            for attribute in (
                "num_electron_up",
                "num_electron_dn",
                "jastrow_1b_type",
                "jastrow_2b_type",
                "is_complex",
                "pbc_flag",
            ):
                if hasattr(value, attribute):
                    summary[attribute] = canonical_value(getattr(value, attribute))
            for attribute in ("_latest_r_up_carts", "_latest_r_dn_carts"):
                if hasattr(value, attribute):
                    self._observe_coordinates(f"{name}.{attribute}", getattr(value, attribute))
            wavefunction = getattr(value, "wavefunction_data", None)
            if wavefunction is not None:
                geminal = getattr(wavefunction, "geminal_data", None)
                jastrow = getattr(wavefunction, "jastrow_data", None)
                if geminal is not None:
                    summary["electron_counts"] = [
                        int(getattr(geminal, "num_electron_up", 0)),
                        int(getattr(geminal, "num_electron_dn", 0)),
                    ]
                    summary["determinant_type"] = type(geminal).__qualname__
                if jastrow is not None:
                    summary["jastrow_components"] = [
                        label
                        for label, attribute in (
                            ("J1", "jastrow_one_body_data"),
                            ("J2", "jastrow_two_body_data"),
                            ("J3", "jastrow_three_body_data"),
                            ("J_DTN", "jastrow_dtn_data"),
                        )
                        if getattr(jastrow, attribute, None) is not None
                    ]
            self.physics_objects[name] = summary
            self._physics_object_refs[identity_value] = value

        if not _is_observable_container(value):
            return
        try:
            attributes = vars(value)
        except TypeError:
            attributes = {}
        # A physics object's own coordinate state is observed explicitly above
        # (``_latest_r_*_carts``); only members of test-defined records and of
        # plain containers are classified by name.
        members = members and not _is_physics_object(value)
        if len(attributes) <= 64:
            for attribute, child in list(attributes.items()):
                if members:
                    self._observe_coordinate_member(f"{name}.{attribute}", child)
                self._observe_physics_object(f"{name}.{attribute}", child, depth + 1, visited, members)
        if type(value).__module__.startswith(("tests.", "test_")):
            class_attributes = vars(type(value))
            if len(class_attributes) <= 128:
                for attribute, child in class_attributes.items():
                    if not callable(child) and not isinstance(child, (staticmethod, classmethod, property)):
                        self._observe_physics_object(f"{name}.{attribute}", child, depth + 1, visited, members)

    def _observe_coordinate_member(self, name: str, value: Any) -> None:
        """Observe a coordinate-named member of a test record or plain container.

        A cached capture (a test dataclass or dict holding ``r_up`` / ``r_dn``)
        is consumed by every later test through that member; the static engine
        already declares such dotted symbols (``capture.r_up``) as coordinates.
        The member name is classified exactly like a local of that name.
        """
        if _is_runtime_coordinate_name(name):
            self._observe_coordinates(name, value, remember=True)

    def finalize(self) -> None:
        """Resolve and publish provenance carried by cached physics objects.

        The first test that builds a cached Hamiltonian observes both its input
        path and its object identity.  Later tests may enter only the cache
        wrapper, so they recover the path/hash from the same live object.  This
        remains runtime evidence: no expected artifact is copied unless some
        executed test in this worker previously observed that exact path/hash.
        """
        recovered: dict[str, str] = {}
        carriers = {
            id(value): value for reference in self._used_provenance_object_refs.values() if (value := reference()) is not None
        }
        for value in carriers.values():
            recovered.update(
                {
                    path: digest
                    for path, digest in _object_provenance(value).items()
                    if committed_artifact_universe().get(path) == digest
                }
            )
            # Physics objects a cache hit did not rebuild (see _bind_return_physics).
            for name, summary in _object_physics(value).items():
                if not any(existing == summary for existing in self.physics_objects.values()):
                    self.physics_objects[f"cached.{name}"] = summary
        self.observed_artifacts.update(recovered)
        carried = {
            path: digest
            for path, digest in self.observed_artifacts.items()
            if committed_artifact_universe().get(path) == digest
        }
        # A single-source builder may be observed directly without a traced
        # helper return (the unit-test and simple-fixture path).  Preserve that
        # provenance without smearing a multi-source comparison across every
        # object participating in the comparison.
        if len(carried) == 1 or len(carriers) == 1:
            for value in carriers.values():
                _remember_object_provenance(value, carried)


def expected_artifacts(contract: EvidenceNodeContract, case: dict[str, Any]) -> dict[str, str]:
    rows = contract.case_binding.get("sources", {}).get("cases", ())
    selected = next((row for row in rows if row.get("case") == case.get("case")), None)
    artifacts = selected.get("artifacts", ()) if selected is not None else ()
    return {row["path"]: row["sha256"] for row in artifacts}


def runtime_selector(item: Any) -> dict[str, Any]:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return {}
    return {name: canonical_value(value) for name, value in callspec.params.items()}
