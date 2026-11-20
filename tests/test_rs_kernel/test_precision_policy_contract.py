"""Native precision-policy and route-dispatch contract.

Numerical XLA--Rust agreement cannot prove that Rust selected the same scalar
type: an all-f64 kernel can sit within a few float32 ULPs.  These tests compare
the immutable Python-resolved precision-zone snapshot with Rust's independently
reported *implemented arithmetic support*.  The contract protects the four
mixed f32 zones against regression to an all-f64 implementation and verifies
that every production family resolves every zone it owns.

The suite is run in separate ``--precision-mode=full`` and ``mixed`` processes.
Both modes are blocking and must be green.
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

import numpy as np
import pytest

import jqmc_rs

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]

from jqmc._precision import (
    PRECISION_ZONE_ORDER,
    configure,
    get_tolerance_xla_rust_consistency,
    is_mixed_precision_enabled,
    precision_policy_snapshot,
)
from jqmc._setting import (
    ATOL_XLA_RUST_CONSISTENCY_MEDIUM,
    RTOL_XLA_RUST_CONSISTENCY_MEDIUM,
)

from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg, rs_ham
from tests._precision_fidelity import mixed_vs_full


# Call-site-complete production-family manifest at the precision-zone boundary.
# A family may have several concrete real/complex/batch entry points, but those
# twins own the same arithmetic zones.  The manifest deliberately includes the
# four leaf owners as their own rows and every fused/driver consumer downstream.
_PRODUCTION_ROUTE_ZONES = {
    "ao_value": (
        "ao_eval",
        "ao_bloch_eval",
    ),
    # Standalone block VGL has its own contraction zones. This row does not
    # claim that tiled orbitals are already supported by native drivers.
    "tiled_orbital_vgl": ("ao_eval", "ao_bloch_eval", "ao_grad_lap", "tiling_eval", "tiling_grad", "tiling_lap"),
    "jastrow_value": ("jastrow_eval",),
    "jastrow_grad_lap": ("jastrow_grad_lap",),
    "jastrow_ratio": ("jastrow_ratio",),
    "determinant_value": ("ao_eval", "ao_bloch_eval", "mo_eval", "det_eval"),
    "wavefunction_value": ("ao_eval", "ao_bloch_eval", "mo_eval", "det_eval", "jastrow_eval", "wf_eval"),
    "mcmc_transition": ("ao_eval", "ao_bloch_eval", "det_ratio", "jastrow_ratio", "wf_ratio"),
    "local_energy": (
        "ao_eval",
        "ao_bloch_eval",
        "ao_grad_lap",
        "mo_eval",
        "mo_grad",
        "mo_lap",
        "det_eval",
        "det_grad_lap",
        "jastrow_grad_lap",
        "coulomb",
        "ewald",
        "wf_kinetic",
        "local_energy",
    ),
    "mcmc_observables": (
        "ao_eval",
        "ao_bloch_eval",
        "jastrow_eval",
        "jastrow_grad_lap",
        "det_eval",
        "wf_kinetic",
        "coulomb",
        "ewald",
        "swct",
    ),
    "gfmc_projection": (
        "ao_eval",
        "ao_bloch_eval",
        "jastrow_eval",
        "jastrow_grad_lap",
        "jastrow_grad_lap_carry",
        "jastrow_ratio",
        "det_ratio",
        "wf_ratio",
        "wf_kinetic",
        "local_energy",
    ),
    "force_stress": (
        "ao_eval",
        "ao_bloch_eval",
        "ao_grad_lap",
        "mo_grad",
        "det_grad_lap",
        "jastrow_grad_lap",
        "wf_kinetic",
        "coulomb",
        "ewald",
        "swct",
    ),
    "parameter_derivatives": (
        "ao_eval",
        "ao_bloch_eval",
        "ao_grad_lap",
        "mo_eval",
        "mo_grad",
        "det_eval",
        "det_grad_lap",
        "jastrow_eval",
        "jastrow_grad_lap",
        "wf_kinetic",
        "coulomb",
    ),
}


def _parts():
    ham = rs_ham("mol_cart_sd_j12")
    return ham.wavefunction_data.geminal_data, ham.wavefunction_data.jastrow_data, ham.coulomb_potential_data


def _ctx(policy):
    gem, jas, cp = _parts()
    return jqmc_rs.build_obs_ctx(
        gem,
        jas,
        cp,
        False,
        False,
        False,
        False,
        False,
        False,
        precision_policy=policy,
    )


def test_precision_zone_schema_matches_rust():
    """Precision zone schema matches rust."""
    assert tuple(jqmc_rs.native_precision_zones()) == PRECISION_ZONE_ORDER


def test_force_medium_tolerance_only_widens_mixed():
    """Force medium tolerance only widens mixed."""
    strict = get_tolerance_xla_rust_consistency(("ao_eval", "wf_kinetic"))
    medium = get_tolerance_xla_rust_consistency(("ao_eval", "wf_kinetic"), level="medium")
    if precision_policy_snapshot()["ao_eval"] == "float32":
        assert medium == (ATOL_XLA_RUST_CONSISTENCY_MEDIUM, RTOL_XLA_RUST_CONSISTENCY_MEDIUM)
        assert medium[0] > strict[0]
        assert medium[1] > strict[1]
    else:
        assert medium == strict


@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Native precision-policy metadata only; no evaluation coordinates."
)
def test_native_context_owns_exact_python_snapshot():
    """Native context owns exact python snapshot."""
    expected = precision_policy_snapshot()
    ctx = _ctx(expected)
    assert ctx.precision_policy() == expected


def _full_and_mixed_policies():
    full = {zone: "float64" for zone in PRECISION_ZONE_ORDER}
    mixed = dict(full)
    for zone in ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "jastrow_ratio"):
        mixed[zone] = "float32"
    return full, mixed


@pytest.mark.parametrize("outcome", ["success", "error"])
@pytest.mark.parametrize(("python_mode", "native_mode"), [("full", "mixed"), ("mixed", "full")])
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Precision-policy restoration only; no evaluation coordinates."
)
def test_mixed_vs_full_restores_python_and_native_entry_policies(outcome, python_mode, native_mode):
    """The fidelity helper restores initially divergent backend policies independently."""
    suite_python_mode = "mixed" if is_mixed_precision_enabled() else "full"
    suite_native_policy = dict(jqmc_rs.native_precision_support())
    full_policy, mixed_policy = _full_and_mixed_policies()
    entry_native_policy = {"full": full_policy, "mixed": mixed_policy}[native_mode]

    configure(python_mode)
    jqmc_rs.configure_native_precision(entry_native_policy)

    def evaluate(_case):
        if outcome == "error":
            raise RuntimeError("deliberate evaluator failure")
        return np.zeros(1)

    try:
        if outcome == "error":
            with pytest.raises(RuntimeError, match="deliberate evaluator failure"):
                mixed_vs_full(
                    lambda: None,
                    evaluate,
                    configure_native=jqmc_rs.configure_native_precision,
                    native_policy_snapshot=jqmc_rs.native_precision_support,
                )
        else:
            mixed_vs_full(
                lambda: None,
                evaluate,
                configure_native=jqmc_rs.configure_native_precision,
                native_policy_snapshot=jqmc_rs.native_precision_support,
            )

        assert ("mixed" if is_mixed_precision_enabled() else "full") == python_mode
        assert jqmc_rs.native_precision_support() == entry_native_policy
    finally:
        configure(suite_python_mode)
        jqmc_rs.configure_native_precision(suite_native_policy)


# Routes whose arithmetic once followed the process-wide
# standalone setting (``configure_native_precision``) instead of the policy of the
# context or adapter that owns them.  Production never configures that setting.
# "context": NativeObsCtx e_L (plus the force on periodic cells); "adapter": the
# production complex-twist parameter-derivative adapter, whose free kernels parse
# their data at call time.
_CONTEXT_LOCAL_ROUTES = {
    "mol_cart_sd_j12": "context",  # molecular Cartesian AO value kernels
    "mol_ecp_sphe_j12": "context",  # molecular spherical AO value kernels + ECP
    "pbc_composite_j12": "context",  # periodic pGTO + plane-wave value kernels
    "pw_twist_jnone": "context",  # complex plane-wave value kernels
    "bspline_twist_jnone": "context",  # complex B-spline value kernels
    "twist_sphepgto_j123": "adapter",  # complex free kernels (MO / J3 O_k)
}


def _context_local_outputs(case, policy):
    """Build the owner once with ``policy`` and return an evaluator of its outputs."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
    from jqmc.determinant import compute_geminal_all_elements
    from jqmc.jqmc_mcmc import _RsComplexPdCtx

    ham = rs_ham(case)
    wf = ham.wavefunction_data
    gem, jas, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    pbc = bool(ham.structure_data.pbc_flag)
    r_up, r_dn = draw_well_conditioned_cfg(ham, seed=7)
    geminal = np.asarray(compute_geminal_all_elements(gem, r_up, r_dn))
    complex_ctx = bool(np.iscomplexobj(geminal))
    ctx = jqmc_rs.build_obs_ctx(gem, jas, cp, complex_ctx, pbc, pbc, precision_policy=policy)
    if _CONTEXT_LOCAL_ROUTES[case] == "adapter":
        adapter = _RsComplexPdCtx(gem, jas, cp, ctx)

        def evaluate():
            # Bloch AO values remain float64 in both policies. Their O_k
            # can therefore agree bitwise. The J3 energy derivative also
            # consumes the selectable Jastrow-gradient zone, so it supplies
            # the policy-sensitive control while retaining every O_k check.
            return [
                np.asarray(x)
                for x in (
                    *adapter.pd_ok_mo(r_up, r_dn),
                    adapter.pd_ok_j3(r_up, r_dn),
                    adapter.pd_el_j3(r_up, r_dn),
                )
            ]

        return evaluate
    up, dn = r_up[None], r_dn[None]
    inverse = np.linalg.inv(geminal)[None]
    weights = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    points = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)[None]
    suffix = "_complex_batch" if complex_ctx else "_batch"

    def evaluate():
        out = [np.asarray(getattr(ctx, "e_L" + suffix)(up, dn, weights, points, inverse))]
        if pbc:
            out += [np.asarray(x) for x in getattr(ctx, "grad_e_L_pbc" + suffix)(up, dn, weights, points, inverse)]
        return out

    return evaluate


@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_native_context_policy_is_immutable_and_context_local():
    """Native context policy is immutable and context local.

    Two contexts with different policies coexist in one process and keep their
    snapshot when the caller mutates its dictionaries.  Every output of
    a full-policy and of a mixed-policy owner (the context's e_L, plus the force on
    a periodic cell; the production complex-twist adapter's O_k and J3 energy derivative) is bitwise the
    same whether the process-wide standalone setting is configured to full
    (= never configured, as in production) or to mixed, and the two owners
    differ from each other (the owner's own policy is the one applied).
    """
    full, mixed = _full_and_mixed_policies()
    full_ctx = _ctx(full)
    mixed_ctx = _ctx(mixed)
    # Mutating the caller-owned dictionaries after construction must not mutate
    # either Rust context, and two policies must coexist in one process.
    full["ao_eval"] = "float32"
    mixed["ao_eval"] = "float64"
    assert full_ctx.precision_policy()["ao_eval"] == "float64"
    assert mixed_ctx.precision_policy()["ao_eval"] == "float32"

    # A half-constructed scoped-kernel wrapper (as seen by a tracer inside
    # ``__init__``) must raise AttributeError, never recurse.
    from jqmc.jqmc_mcmc import _NativeFreeKernels

    half_built = _NativeFreeKernels.__new__(_NativeFreeKernels)
    with pytest.raises(AttributeError):
        getattr(half_built, "_module")
    with pytest.raises(AttributeError):
        getattr(half_built, "compute_aos")

    full, mixed = _full_and_mixed_policies()
    python_policy = precision_policy_snapshot()
    for case in sorted(_CONTEXT_LOCAL_ROUTES):
        evaluators = {"full": _context_local_outputs(case, full), "mixed": _context_local_outputs(case, mixed)}
        outputs = {}
        try:
            for standalone_name, standalone in (("full", full), ("mixed", mixed)):
                jqmc_rs.configure_native_precision(standalone)
                for owner, evaluate in evaluators.items():
                    outputs[owner, standalone_name] = evaluate()
        finally:
            jqmc_rs.configure_native_precision(python_policy)
        for owner in evaluators:
            pairs = zip(outputs[owner, "full"], outputs[owner, "mixed"], strict=True)
            for k, (unconfigured, configured) in enumerate(pairs):
                assert np.all(np.isfinite(unconfigured)), (case, owner, k)
                assert np.array_equal(unconfigured, configured), (
                    f"{case}: output {k} of the {owner}-policy owner follows the process-wide standalone "
                    f"setting (max |delta| = {np.max(np.abs(unconfigured - configured)):.3g})"
                )
        assert any(not np.array_equal(a, b) for a, b in zip(outputs["full", "full"], outputs["mixed", "full"], strict=True)), (
            f"{case}: the full- and mixed-policy owners agree bitwise, so the owner policy was not applied"
        )


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("missing", "exactly 23 zones"),
        ("unknown", "unknown native precision zone"),
        ("dtype", "invalid native precision dtype"),
    ],
)
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Malformed precision policy rejected at context construction."
)
def test_native_context_rejects_malformed_policy(mutation, match):
    """Native context rejects malformed policy."""
    policy = precision_policy_snapshot()
    if mutation == "missing":
        policy.pop("ao_eval")
    elif mutation == "unknown":
        policy["not_a_zone"] = policy.pop("ao_eval")
    else:
        policy["ao_eval"] = "bfloat16"
    with pytest.raises((TypeError, ValueError), match=match):
        _ctx(policy)


@pytest.mark.parametrize("omission", ["omitted", "explicit_none"])
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Missing precision policy rejected at context construction."
)
def test_native_context_rejects_missing_policy(omission):
    """A context built without a precision policy raises instead of defaulting to full.

    A silent all-float64 default would let a harness that forgets the policy run
    fp64 Rust inside a mixed-precision session.  Both spellings of the omission
    must raise ``TypeError`` naming the missing argument.
    """
    gem, jas, cp = _parts()
    args = (gem, jas, cp, False, False, False, False, False, False)
    with pytest.raises(TypeError, match="requires an explicit precision_policy"):
        if omission == "omitted":
            jqmc_rs.build_obs_ctx(*args)
        else:
            jqmc_rs.build_obs_ctx(*args, precision_policy=None)


@pytest.mark.parametrize("zone", PRECISION_ZONE_ORDER)
def test_native_zone_dispatch_matches_python_policy(zone):
    """One-zone contract: selected native dtype matches the Python policy."""
    expected = precision_policy_snapshot()[zone]
    actual = jqmc_rs.native_precision_support()[zone]
    assert actual == expected, (
        f"Rust precision dispatch is not implemented for zone={zone!r}: "
        f"Python selected {expected}, Rust arithmetic is still {actual}"
    )


@pytest.mark.parametrize("route", tuple(_PRODUCTION_ROUTE_ZONES))
def test_native_production_route_dispatch_matches_python_policy(route):
    """Fused/driver contract: every route selects all of its requested zones."""
    expected = precision_policy_snapshot()
    actual = jqmc_rs.native_precision_support()
    mismatches = {
        zone: {"python": expected[zone], "rust": actual[zone]}
        for zone in _PRODUCTION_ROUTE_ZONES[route]
        if actual[zone] != expected[zone]
    }
    assert not mismatches, f"Rust route {route!r} does not execute Python's precision policy: {mismatches}"


def test_route_manifest_covers_all_precision_zones():
    """Route manifest covers all precision zones."""
    covered = {zone for zones in _PRODUCTION_ROUTE_ZONES.values() for zone in zones}
    assert covered == set(PRECISION_ZONE_ORDER)
