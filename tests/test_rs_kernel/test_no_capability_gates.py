"""MECHANICAL GUARD: no capability gate may reappear.

``_rs_pd_native_ok``-shaped booleans -- conjunctions of "is this configuration
something Rust can do?" predicates whose ``False`` branch quietly advances a REQUESTED
observable on the JAX kernel instead -- are forbidden outright: an unimplemented cell must
raise ``NotImplementedError``.

A prose rule decays. This module parses ``jqmc/jqmc_mcmc.py`` and ``jqmc/jqmc_gfmc.py``
with ``ast`` and fails if the SHAPE comes back.

What the guard catches (the gate shape, not a keyword):

  1. Any assignment whose value COMBINES two or more CAPABILITY PREDICATES -- ``not
     <complex flag>``, ``not <mpc flag>``, a ``*_leaf_ok`` / ``*_config_ok`` /
     ``*_basis_active`` / ``*_native`` style name, or a call to a ``*_leaf_ok`` /
     ``*_ok`` predicate function. That is exactly the
     ``(not _rs_complex) and (not self.__mpc) and ... and _rs_param_deriv_leaf_ok(...)``.
  2. A module-level ``def`` whose name matches the leaf-gate predicate shape
     (``_rs_*_leaf_ok`` / ``_rs_*_native_ok``) -- the function half of the same pattern.

A matcher that recognises only a TOP-LEVEL ``ast.And`` whose direct operands are
predicates, with a suffix list that omits the bare ``_native``, reports "no violation"
for four gate expressions; all four are pinned as regression rows of
``test_guard_detects_a_reintroduced_gate``. The matcher therefore flattens:

  * ``or`` as well as ``and`` (``a or b`` gates just as hard as ``a and b``);
  * NESTED boolean operators (``a and (b and not c)``);
  * ``all(...)`` / ``any(...)`` over a tuple / list / set / comprehension;
  * ``bool(...)`` wrappers, at any depth;
  * names ending in ``_native`` (``_rs_hf_native``, ``_rs_stress_hf_native``, ...).

What it deliberately does NOT catch, so it stays precise rather than broad:

  * the STRICT RAISES that replaced them (``if ... : raise NotImplementedError``): they are
    the sanctioned replacement, and they are guarded for content by
    ``test_raise_contract.py`` / ``test_mcmc_twist_mpc_position_deriv_raises``;
  * a single native-service predicate consumed by an immediate strict raise; this
    module guards the shape that invents a new conjunction;
  * GFMC's ``_rs_gfmc_force_chunk_ok`` chunk-FUSION gate, which selects between two paths
    that compute the same observable (performance, not capability) -- it is allowed by name
    below, and only by name, so a new gate cannot hide behind it;
  * ``_DERIVED_COMPOSITES`` -- names that are a pure ``and`` of predicates ALREADY declared
    (and already sanctioned) elsewhere in the same function, adding no new capability
    question of their own. See the block comment on that constant: they are allowed by
    EXACT name, they must each still exist (the ``stale-exception`` scan
    fails if one is deleted or renamed, so the allowlist cannot rot into a licence), and
    their ATOMS remain the tracked fallbacks;
  * ``_WARMUP_MIRRORS`` -- the ham-only copies of the force/stress gates that decide only
    which JAX kernels to PRE-COMPILE. They never select a compute kernel, so they cannot
    produce a wrong number in either direction; see the comment on that constant.

Mutation-verified: re-introducing a two-predicate gate expression fails this module, in
every one of the five shapes above.
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
import pathlib

import pytest

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_TARGETS = ("jqmc/jqmc_mcmc.py", "jqmc/jqmc_gfmc.py")

# Chunk-FUSION gates: both branches compute the same observable (one fused native call vs the
# per-step dispatch), so they are a performance choice, not a capability fallback. Allowed by
# EXACT name only.
_ALLOWED_NAMES = frozenset({"_rs_gfmc_force_chunk_ok", "_rs_obs_chunk_force_ok", "_rs_energy_chunk"})

# --------------------------------------------------------------------------- #
# DERIVED COMPOSITES.
#
# A pure ``and`` of predicates already sanctioned elsewhere in the same function
# would need an explicit entry here. Every periodic AO leaf is native, so there is
# no such composite and the allowlist is intentionally empty.  Any remaining
# force-family predicates are separately consumed by strict raises.
#
# The allowlist is EXACT-NAME and cannot rot: the ``stale-exception`` scan of
# ``test_no_capability_gate_expression_reappears`` fails if an entry stops existing in the sources, so a rename or a removal forces this
# comment to be re-read rather than leaving a standing licence behind.
_DERIVED_COMPOSITES = frozenset()

# WARM-UP MIRRORS. ``jqmc_mcmc.run()`` recomputes a ham-only copy of the force / stress
# gates BEFORE the main statics block, for one purpose: deciding which JAX kernels to
# PRE-COMPILE (a DTN(+B-spline) strained-energy warm-up alone peaks tens of GB per rank).
# These booleans never choose which kernel
# COMPUTES anything -- the authoritative gates below do -- so an over-eager skip costs a
# lazy recompile and can never produce a wrong number. That is stated in the production
# comment at the block head, and it is why they are excused here: they are not a
# capability decision in either direction.
_WARMUP_MIRRORS = frozenset({"_warm_stress_hf_native", "_warm_stress_pp_native"})

_PREDICATE_NAME_SUFFIXES = ("_leaf_ok", "_config_ok", "_native_ok", "_basis_active", "_family_ok", "_native")
_CAPABILITY_TOKENS = ("complex", "mpc")


def _sources():
    root = pathlib.Path(__file__).resolve().parents[2]
    return [(t, (root / t).read_text()) for t in _TARGETS]


def _target_name(node) -> str:
    """Assignment target name (``x = ...`` / ``self.__x = ...``), or ``''``."""
    tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
    for t in tgts:
        if isinstance(t, ast.Name):
            return t.id
        if isinstance(t, ast.Attribute):
            return t.attr
    return ""


def _is_capability_predicate(node) -> str | None:
    """Describe ``node`` if it is a capability predicate, else ``None``.

    * ``not <name containing 'complex'/'mpc'>`` -- the ``not _rs_complex`` /
      ``not self.__mpc`` terms;
    * a bare name / attribute whose suffix is a gate-predicate suffix;
    * a call to a function whose name has a gate-predicate suffix.
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner = node.operand
        text = ""
        if isinstance(inner, ast.Name):
            text = inner.id
        elif isinstance(inner, ast.Attribute):
            text = inner.attr
        if any(tok in text.lower() for tok in _CAPABILITY_TOKENS):
            return f"not {text}"
        return None
    name = ""
    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Attribute):
        name = node.attr
    elif isinstance(node, ast.Call):
        f = node.func
        name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
    if name and name.endswith(_PREDICATE_NAME_SUFFIXES):
        return name
    return None


def _combinator_operands(node) -> list | None:
    """Flatten a boolean COMBINATOR into its leaf operands, or ``None`` if ``node`` is
    not a combinator at all.

    The original version handled exactly one shape -- a top-level
    ``ast.And`` -- so ``a or b``, ``a and (b and c)`` and ``all((a, b, c))`` were all
    invisible. Every combinator below is flattened RECURSIVELY, so nesting one inside
    another does not launder a gate:

      * ``ast.BoolOp`` with either ``and`` or ``or``;
      * ``all(...)`` / ``any(...)`` over a tuple / list / set literal, or over a
        generator / list comprehension (in which case the comprehension's element
        expression is the operand);
      * ``bool(...)`` -- a transparent wrapper, unwrapped at any depth.
    """
    if isinstance(node, ast.BoolOp):
        out = []
        for v in node.values:
            out.extend(_combinator_operands(v) or [v])
        return out
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
        if node.func.id == "bool":
            inner = node.args[0]
            return _combinator_operands(inner) or [inner]
        if node.func.id in ("all", "any"):
            arg = node.args[0]
            if isinstance(arg, (ast.Tuple, ast.List, ast.Set)):
                out = []
                for e in arg.elts:
                    out.extend(_combinator_operands(e) or [e])
                return out
            if isinstance(arg, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
                return _combinator_operands(arg.elt) or [arg.elt]
            return [arg]
    return None


def _gate_assignments(tree):
    """(lineno, target, [predicates]) for every assignment COMBINING >= 2 capability
    predicates, in any of the shapes ``_combinator_operands`` recognises."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None:
            continue
        operands = _combinator_operands(value)
        if operands is None:
            continue
        preds = [p for p in (_is_capability_predicate(v) for v in operands) if p]
        if len(preds) >= 2:
            found.append((node.lineno, _target_name(node), preds))
    return found


def _expression_offenders():
    """Assignments combining >= 2 capability predicates outside the exact-name exceptions."""
    offenders = []
    for path, src in _sources():
        for lineno, target, preds in _gate_assignments(ast.parse(src)):
            if target in _ALLOWED_NAMES or target in _DERIVED_COMPOSITES or target in _WARMUP_MIRRORS:
                continue
            offenders.append(f"{path}:{lineno}: {target} combines {preds}")
    return offenders


def _predicate_function_offenders():
    """Module-level ``_rs_*_leaf_ok`` / ``_rs_*_native_ok`` gate predicates.

    ``_rs_param_deriv_leaf_ok`` is exactly this shape: a helper answering "can Rust do
    this wavefunction?" whose only consumer is a gate conjunction. A predicate called
    directly in an ``if`` escapes the assignment scans, so its definition is rejected.
    """
    offenders = []
    for path, src in _sources():
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_rs_"):
                if node.name.endswith(("_leaf_ok", "_native_ok", "_config_ok")) and node.name not in _ALLOWED_NAMES:
                    offenders.append(f"{path}:{node.lineno}: def {node.name}")
    return offenders


def _single_predicate_offenders():
    """A one-term capability gate is just as capable of hiding an unported cell.

    It is permitted only when the same predicate is consumed by a strict
    ``NotImplementedError`` guard, or when it is the exact MPC exception.
    Constant ``True`` bookkeeping and chunk-fusion performance switches are not
    capability decisions.
    """
    offenders = []
    for path, src in _sources():
        tree = ast.parse(src)
        strict_guarded: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if not any(isinstance(child, ast.Raise) for stmt in node.body for child in ast.walk(stmt)):
                continue
            for child in ast.walk(node.test):
                if isinstance(child, ast.Name) and child.id.endswith(_PREDICATE_NAME_SUFFIXES):
                    strict_guarded.add(child.id)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            target = _target_name(node)
            if not target.endswith(_PREDICATE_NAME_SUFFIXES):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
                continue
            if target.startswith(("_warm_", "_w_")):
                continue
            if target == "_rs_pd_j3_basis_active":
                # Detection input to ``_rs_pd_j3_gto``; that derived predicate is
                # consumed only by the strict raise, never by a dispatch branch.
                continue
            if target in _ALLOWED_NAMES or target in _WARMUP_MIRRORS or target in strict_guarded:
                continue
            # Exception for an MPC-dependent ``_rs_hf_native`` assignment in jqmc_gfmc.py.
            if path.endswith("jqmc_gfmc.py") and target == "_rs_hf_native" and "mpc" in ast.unparse(node.value).lower():
                continue
            offenders.append(f"{path}:{node.lineno}: {target} = {ast.unparse(node.value)}")
    return offenders


def _stale_exception_offenders():
    """Exact-name exceptions not assigned in a production driver.

    An allowlist that outlives the thing it excuses becomes a standing licence for a
    future gate to reuse the name.
    """
    assigned = set()
    for _path, src in _sources():
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                name = _target_name(node)
                if name:
                    assigned.add(name)
    return sorted((_ALLOWED_NAMES | _DERIVED_COMPOSITES | _WARMUP_MIRRORS) - assigned)


_LIVE_SCANS = {
    "expression": _expression_offenders,
    "predicate-function": _predicate_function_offenders,
    "single-predicate": _single_predicate_offenders,
    "stale-exception": _stale_exception_offenders,
}


@pytest.mark.parametrize("scan", sorted(_LIVE_SCANS))
def test_no_capability_gate_expression_reappears(scan):
    """No capability gate exists in the production drivers, in any of the guarded shapes.

    ``expression``: a conjunction of capability predicates (the ``_rs_pd_native_ok`` shape);
    ``predicate-function``: a ``_rs_*_leaf_ok``-style gate predicate definition;
    ``single-predicate``: a one-term gate not consumed by a strict raise;
    ``stale-exception``: an exact-name exception that excuses nothing.
    None may exist: port the cell, or raise NotImplementedError.
    """
    offenders = _LIVE_SCANS[scan]()
    assert not offenders, f"capability gate ({scan}) re-introduced or licensed:\n  " + "\n  ".join(offenders)


# The conjunction gate shape, plus the four expressions a top-level-And-only
# matcher reports as "no violation". Each row exercises a
# different flattening branch, so a narrowing of the matcher names the exact shape
# it stopped recognising.
_GATE_SHAPES = (
    # (id, source, expected target, minimum predicates)
    (
        "phase4-conjunction",
        "_rs_pd_native_ok = (\n"
        "    (not _rs_complex)\n"
        "    and (not self.__mpc)\n"
        "    and (not _rs_pd_basis_active)\n"
        "    and _rs_pd_config_ok\n"
        "    and _rs_param_deriv_leaf_ok(self.__hamiltonian_data)\n"
        ")\n",
        "_rs_pd_native_ok",
        4,
    ),
    # A bare '_native' suffix is a predicate name.
    ("and_native_suffix", "_rs_stress_hf_native = _rs_hf_native and _rs_stress_leaf_ok\n", "_rs_stress_hf_native", 2),
    # ast.Or is a gate shape too.
    ("or_combinator", "_rs_force_native = _rs_hf_native or _rs_pp_native\n", "_rs_force_native", 2),
    # all()/any() over a tuple is a BoolOp in meaning but an ast.Call in the tree.
    ("all_call", "x = all((not _rs_complex, not self.__mpc, _rs_leaf_ok))\n", "x", 3),
    # Operands nested below the top-level And are inspected too.
    ("nested_boolop", "x = _rs_hf_native and (_rs_stress_leaf_ok and not _rs_complex)\n", "x", 3),
)


@pytest.mark.parametrize("case_id, src, target, min_preds", _GATE_SHAPES, ids=[c[0] for c in _GATE_SHAPES])
def test_guard_detects_a_reintroduced_gate(case_id, src, target, min_preds):
    """The guard's own detector recognises every gate shape (negative control for the live scans)."""
    hits = _gate_assignments(ast.parse(src))
    assert hits, f"[{case_id}] the matcher does not recognise `{src.strip()}`"
    _lineno, got_target, preds = hits[0]
    assert got_target == target, f"[{case_id}] matched target {got_target!r}, expected {target!r}"
    assert len(preds) >= min_preds, f"[{case_id}] recognised only {preds}, expected >= {min_preds} capability predicates"


def test_rs_tests_do_not_turn_exceptions_into_skip_or_xfail():
    """A broken fixture/oracle/native call is a failure, never a green omission."""
    root = pathlib.Path(__file__).resolve().parent
    offenders = []
    for path in sorted(root.glob("test_*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ExceptHandler):
                continue
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "pytest"
                    and child.func.attr in {"skip", "xfail"}
                ):
                    offenders.append(f"{path.name}:{child.lineno}")
    assert not offenders, "exceptions converted to green skip/xfail:\n  " + "\n  ".join(offenders)


# --------------------------------------------------------------------------- #
# Reviewed predicate shapes in the Rust sources.
#
# The Rust half cannot raise NotImplementedError for an unsupported leaf the way
# the Python drivers do; its capability predicates select a slower route that
# computes the same observable. They are therefore reviewed only by EXACT
# (file, owner, name) with a reviewed witness: the route-counter index whose
# registered, both-precision test read proves the decline did not fire on the
# covered fixtures (gdd/engine/route_counter_evidence.py), or ``None`` with a
# reason for a same-observable performance route that has no counter of its own
# (listed as an open issue, not a licence: a new unwitnessed entry needs review).
# Numerical-safety predicates are classified separately: they select between
# equivalent arithmetic forms rather than between backend capabilities.
# A witness ``("predicate", i)`` names index ``i`` of the separate
# ``jqmc_rs.route_predicate_counters()`` array (the predicate's DECLINE counter),
# whose reads are reviewed in ``PREDICATE_COUNTER_READ_EVIDENCE``.
# Matcher: gdd.engine.rust_sources.masked_route_predicates. The classifications are
# enforced by scripts/check_gdd_rust_inventory.py in the static preflight; the tests
# below prove that entry point runs the predicate check and that the matcher has teeth.
# --------------------------------------------------------------------------- #


def test_rust_inventory_public_validator_rejects_unreviewed_predicate(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """The static preflight entry point must delegate to the Rust predicate inventory."""
    from scripts import check_gdd_rust_inventory as checker

    source = tmp_path / "jqmc/rs/src/audit.rs"
    source.parent.mkdir(parents=True)
    source.write_text(
        "fn audit_unreviewed_ok(a: f64) -> bool { a.is_finite() && a > 0.0 }\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        checker.rust_inventory,
        "load_rust_inventory",
        lambda _root: {
            "RUST_TESTS": {},
            "PYTHON_EXPORTS": {},
            "TEST_KINDS": frozenset(),
            "EXPORT_STATUSES": frozenset(),
        },
    )
    predicate_inventory_errors = checker.predicate_inventory_errors
    monkeypatch.setattr(
        checker,
        "predicate_inventory_errors",
        lambda root: predicate_inventory_errors(root, route_reviews={}, numeric_reviews={}),
    )

    assert checker.validation_errors(tmp_path) == [
        "Rust predicate ('jqmc/rs/src/audit.rs', 'fn', 'audit_unreviewed_ok') at line 1 has no exact reviewed classification"
    ]


def test_rust_route_predicate_detector_catches_a_new_gate():
    """Negative control: each Rust shape is detected; comments and strings are not code."""
    from gdd.engine.rust_sources import mask, masked_route_predicates

    source = (
        "fn new_leaf_ok(orb: &OrbData) -> bool {\n    orb.pw.is_none() && orb.bspline.is_none()\n}\n"
        "fn fast_new_route(g: &G) -> bool { g.a }\n"
        "fn classify(o: &OrbData) -> CandidateBasisClass { CandidateBasisClass::Unsupported }\n"
        "fn host(x: Option<u8>, y: bool) {\n    let new_fast = x.is_some() && y;\n"
        "    let plain_ok = compute(x);\n    // let hidden_ok = a && b;\n"
        '    let s = "let quoted_ok = a && b;";\n}\n'
    )
    found = {(owner, name) for _path, owner, name, _line in masked_route_predicates("x.rs", mask(source))}
    assert found == {("fn", "new_leaf_ok"), ("fn", "fast_new_route"), ("fn", "classify"), ("host", "new_fast")}, found
