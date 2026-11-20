"""Analytic GFMC_t / LRDMC Hellmann--Feynman force
(``jqmc.derivatives.gfmc.grad_e_L_t_wrt_positions``) versus an independent
``jax.grad`` of the novar=0 continuum measurement energy, for molecular
all-electron jnone.

The oracle ``_e_L_t_ae_jnone`` below is a standalone reproduction of
the continuum measurement definition restricted to the AE (``ecp_flag=False``)
branch. The analytic continuum-measurement force must match it to machine
precision. Propagation mesh/sign-flip state is witnessed separately as a
precondition and is not differentiated by this oracle.
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

import os

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

import functools

from gdd.engine.evidence import ComparisonBinding, ObservationBinding, observe_allclose, observe_exact, observe_non_numerical
from jqmc._precision import get_tolerance_min as _resolve_tolerance_min
from jqmc.coulomb_potential import (
    compute_coulomb_potential_ee_dispatch,
    compute_coulomb_potential_ei_dispatch,
    compute_coulomb_potential_ii_dispatch,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors,
    compute_ecp_non_local_parts_nearest_neighbors_fast_update,
)
from jqmc.derivatives import ecp_force, gfmc
from jqmc.determinant import _geminal_for_grads_lap, compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_DTN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    _phase_free_j3_orbitals,
    compute_Jastrow_part,
)
from jqmc.jqmc_mcmc import _extract_active_positions_from_ham, _inject_active_positions
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import (
    Wavefunction_data,
    compute_kinetic_energy_all_elements,
)
from tests._test_data_cache import qe_example_files_root
from tests._ecp_core_electrons import core_electrons_of
from tests.test_jqmc_kernel._gdd_adapters import (
    case_receipt,
    committed_input,
    current_precision_mode,
    guarantee_id_for_manifest_family,
    seeded_single_electron_receipt,
)

_TREXIO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")

_GDD_CONTEXT = None
_NP_ASSERT_ALLCLOSE = np.testing.assert_allclose
_NP_ASSERT_ARRAY_EQUAL = np.testing.assert_array_equal
_GDD_SOURCE_BY_STEM = {
    "N_ae_ccpvdz_cart.h5": "WF-SRC-N-AE-CART-MOL",
    "Li_ae_ccpvdz_cart.h5": "WF-SRC-LI-AE-CART-MOL",
    "H2_ae_ccpvdz_cart.h5": "WF-SRC-H2-AE-CART-MOL",
    "H2_ae_ccpvdz_sphe.h5": "WF-SRC-H2-AE-SPHE-MOL",
    "H2_ecp_ccpvdz_cart.h5": "WF-SRC-H2-ECP-CART-MOL",
    "N2_ecp_ccpvtz_cart.h5": "WF-SRC-N2-ECP-CART-MOL",
    "Cl2_ecp_ccpvtz_cart.h5": "WF-SRC-CL2-ECP-CART-MOL",
    "lithium.h5": "WF-SRC-LI-ECP-SPHE-GAMMA",
    "lithium_k025.h5": "WF-SRC-LI-ECP-SPHE-NON-GAMMA",
}


def _gdd_get_tolerance_min(zones, level):
    # Analytical and primal-AD routes execute under the same precision policy.
    # Mixed uses the central ``medium`` policy; full remains on the central
    # ``strict`` tier. Both lanes remain blocking correctness evidence.
    resolved_level = "medium" if level == "strict" and current_precision_mode() == "mixed" else level
    atol, rtol = _resolve_tolerance_min(zones, resolved_level)
    if _GDD_CONTEXT is not None:
        _GDD_CONTEXT["tolerance"] = (tuple(zones), resolved_level, float(atol), float(rtol))
    return atol, rtol


#: Optional diagnostic payload. When a comparison fails, whatever the running test put
#: here is printed at full precision before the assertion propagates. These tests compare
#: at configurations found by a rejection search whose acceptance is evaluated in XLA
#: (cond_2(G) < 700 and a non-empty S'), so a different machine can accept a different
#: attempt and land on a different physical point -- so a failing array in one machine's
#: log may not reproduce on another, which compares somewhere else. The
#: dump makes the point itself recoverable from the log.
_DIAG_CONTEXT: dict | None = None


def _gdd_assert_allclose(actual, reference, *args, **kwargs):
    try:
        _NP_ASSERT_ALLCLOSE(actual, reference, *args, **kwargs)
    except AssertionError:
        if _DIAG_CONTEXT:
            np.set_printoptions(precision=17, threshold=10_000, linewidth=200)
            print("\n[DIAG] comparison failed; the configuration it failed at:")
            for key, value in _DIAG_CONTEXT.items():
                if hasattr(value, "shape"):
                    print(f"[DIAG] {key} = np.array({np.asarray(value).tolist()!r})")
                else:
                    print(f"[DIAG] {key} = {value!r}")
            print(f"[DIAG] actual    = np.array({np.asarray(actual).tolist()!r})")
            print(f"[DIAG] reference = np.array({np.asarray(reference).tolist()!r})", flush=True)
        raise
    if _GDD_CONTEXT is not None:
        tolerance = _GDD_CONTEXT.get("tolerance")
        if tolerance is None:
            raise AssertionError("GFMC GDD comparison did not resolve a registered tolerance")
        _GDD_CONTEXT["assertions"].append(("allclose", actual, reference, tolerance))


def _gdd_assert_array_equal(actual, reference, *args, **kwargs):
    _NP_ASSERT_ARRAY_EQUAL(actual, reference, *args, **kwargs)
    if _GDD_CONTEXT is not None:
        _GDD_CONTEXT["assertions"].append(("exact", actual, reference, None))


def _gdd_source_for_case(function_name, parameters):
    stem = parameters.get("stem")
    if stem is not None:
        return (_GDD_SOURCE_BY_STEM[stem],), "trexio"
    if "twist_ae" in function_name:
        return ("WF-SRC-H-AE-NON-GAMMA",), "qe-save"
    if "twist" in function_name:
        return ("WF-SRC-LI-ECP-SPHE-NON-GAMMA",), "trexio"
    if "dtn_pbc" in function_name:
        return ("WF-SRC-LI-ECP-SPHE-GAMMA",), "trexio"
    if function_name in {"test_gfmc_n_equals_t_force_and_stress", "test_gfmc_n_force_stress_dtn_equals_t"}:
        return ("WF-SRC-LI-AE-CART-MOL",), "trexio"
    raise AssertionError(f"GFMC GDD source mapping is missing for {function_name}")


def _gdd_assertions_for_target(assertions, targets, target):
    if len(targets) == 1:
        return assertions
    if targets == ("positions", "strain"):
        return assertions[:3] if target == "positions" else assertions[3:]
    raise AssertionError(f"unsupported GFMC target partition: {targets}")


def _finalize_gdd_gfmc_case(request, spec, context):
    from tests.test_jqmc_kernel import analytic_coverage_registration as manifest

    parameters = dict(getattr(getattr(request.node, "callspec", None), "params", {}))
    assertions = context["assertions"]
    if not assertions or not context["route_calls"]:
        raise AssertionError(f"{spec.function_name}: GFMC GDD route/comparison witness is empty")
    source_ids, artifact_format = _gdd_source_for_case(spec.function_name, parameters)
    first_call = context["route_calls"][0]
    _route_name, _ham, r_up, r_dn = first_call
    generation = context.get("generation", {})
    seed = int(generation.get("seed", parameters.get("seed", 0)))
    attempts = int(generation.get("attempts", 1))
    conditioning_mode = "upper-bounded" if "cond_g" in generation else "not-limited"
    for target in spec.targets:
        obligation_id = manifest._gfmc_case_id(spec, parameters, target)
        selected = _gdd_assertions_for_target(assertions, spec.targets, target)
        methods = {row[0] for row in selected}
        if methods == {"allclose"}:
            zones, level, atol, rtol = selected[0][3]
            if any(row[3] != selected[0][3] for row in selected):
                raise AssertionError(f"{obligation_id}: inconsistent tolerance resolution")
            policy_id = "CMP-ZONE-ALLCLOSE-MEDIUM-001" if level == "medium" else "CMP-ZONE-ALLCLOSE-STRICT-001"
            primary = observe_allclose(
                ComparisonBinding(
                    binding_id=f"COMPARISON::{obligation_id}",
                    policy_id=policy_id,
                    precision_mode=current_precision_mode(),
                    actual_role="xla-gfmc-analytical",
                    reference_role="xla-gfmc-autodiff-same-precision",
                    zones=zones,
                    level=level,
                ),
                tuple((row[1], row[2]) for row in selected),
                atol=atol,
                rtol=rtol,
            )
        elif methods == {"exact"}:
            primary = observe_exact(
                ComparisonBinding(
                    binding_id=f"COMPARISON::{obligation_id}",
                    policy_id="CMP-EXACT-VALUES-001",
                    precision_mode=current_precision_mode(),
                    actual_role="xla-gfmc-n-analytical",
                    reference_role="xla-gfmc-t-analytical",
                ),
                tuple((row[1], row[2]) for row in selected),
            )
        else:
            raise AssertionError(f"{obligation_id}: mixed or missing comparison methods {methods}")
        route_observation = observe_non_numerical(
            ObservationBinding(
                binding_id=f"ROUTE::{obligation_id}",
                policy_id="OBS-GFMC-ROUTE-001",
                precision_mode=current_precision_mode(),
                subject_role="gfmc-analytical-and-oracle-routes",
                required_observations=(
                    "route-executed",
                    "oracle-route-executed",
                    "propagation-sprime-precondition-observed",
                ),
            ),
            {
                "route-executed": bool(context["route_calls"]),
                "oracle-route-executed": bool(selected),
                "propagation-sprime-precondition-observed": bool(context["sprime_states"]),
            },
        )
        electrons = seeded_single_electron_receipt(
            obligation_id,
            r_up,
            r_dn,
            generator_operation=f"tests.test_jqmc_kernel.test_gfmc_force_analytic.{spec.function_name}",
            seed=seed,
            generator_arguments=tuple(
                (key, manifest._stable_parameter_value(value)) for key, value in sorted(parameters.items())
            ),
            generation_pattern="rejection-screen" if conditioning_mode == "upper-bounded" else "direct",
            geminal_conditioning_mode=conditioning_mode,
            maximum_cond_g=700.0 if conditioning_mode == "upper-bounded" else None,
            upper_bound_comparison="less-than" if conditioning_mode == "upper-bounded" else None,
            conditioning_violation_action="deterministic-resample" if conditioning_mode == "upper-bounded" else None,
            maximum_generation_attempts=300 if conditioning_mode == "upper-bounded" else None,
            cond_g_per_configuration=((float(generation["cond_g"]),) if conditioning_mode == "upper-bounded" else ()),
            generation_attempts=attempts,
            rejected_configuration_count=attempts - 1,
        )
        case_receipt(
            guarantee_id=guarantee_id_for_manifest_family(spec.family),
            obligation_id=obligation_id,
            pytest_node_id=request.node.nodeid,
            input_binding=committed_input(
                obligation_id,
                source_ids=source_ids,
                artifact_format=artifact_format,
                builder_operation=f"tests.test_jqmc_kernel.test_gfmc_force_analytic.{spec.function_name}",
                transformations=(f"derivatives.gfmc.fixture.{spec.function_name}.v1",),
            ),
            electron_configuration=electrons,
            primary_check=primary,
            additional_checks=(route_observation,),
        )


@pytest.fixture(autouse=True)
def _clear_jax_caches(request, monkeypatch):
    """Uniform belt (GFMC oracle side): drain jax's compile cache after EVERY test so no
    residual compiled programs accumulate across the ~190 rows a light-tier xdist worker runs -- the
    XLA-accumulation SIGABRT class that crashes xdist workers.  The analytic kernels
    (jitted in ``gfmc.py``) and the oracles (jitted via ``_jit_grad_pos`` / ``_jit_jacrev`` below) keep
    per-test compiles single-digit; this fixture clears even those between tests.  Supersedes the
    per-test manual ``jax.clear_caches()`` calls a few rows still carry (harmless double-clear)."""
    global _GDD_CONTEXT
    from tests.test_jqmc_kernel import analytic_coverage_registration as manifest

    spec = next((row for row in manifest._GFMC_FUNCTION_SPECS if row.function_name == request.function.__name__), None)
    if spec is None:
        yield
        jax.clear_caches()
        return
    context = {"assertions": [], "route_calls": [], "sprime_states": []}
    _GDD_CONTEXT = context
    sprime_operation = gfmc._sprime_data
    for function_name in (
        "grad_e_L_t_wrt_positions",
        "grad_e_L_t_wrt_strain",
        "grad_e_L_n_wrt_positions",
        "grad_e_L_n_wrt_strain",
    ):
        original = getattr(gfmc, function_name)

        @functools.wraps(original)
        def observed_route(*args, _original=original, _name=function_name, **kwargs):
            result = _original(*args, **kwargs)
            if _GDD_CONTEXT is not None and len(args) >= 5:
                ham, r_up, r_dn, RT, alat = args[:5]
                _GDD_CONTEXT["route_calls"].append((_name, ham, r_up, r_dn))
                weights, _ = sprime_operation(ham, r_up, r_dn, RT, alat)
                _GDD_CONTEXT["sprime_states"].append(bool(np.any(np.asarray(weights))))
            return result

        monkeypatch.setattr(gfmc, function_name, observed_route)
    yield
    if context["assertions"]:
        _finalize_gdd_gfmc_case(request, spec, context)
    _GDD_CONTEXT = None
    jax.clear_caches()


def _jit_grad_pos(f, active0, r_up, r_dn):
    """Jitted position-gradient HF-force oracle: lower the whole ``jax.grad`` as ONE XLA program
    instead of thousands of eager micro-compiles (the compile-cache flood behind the SIGABRT).
    ``f(active, r_up, r_dn)`` is the standalone ``e_L^{t,n}`` reproduction; returns
    ``(dR (n_atom,3), dr_up, dr_dn)`` -- identical numerics to the eager form, one XLA program."""
    gR = jax.jit(jax.grad(lambda a: f(a, r_up, r_dn)))(active0)["positions"]
    gru, grd = jax.jit(jax.grad(lambda a, b: f(active0, a, b), argnums=(0, 1)))(r_up, r_dn)
    return gR, gru, grd


def _jit_molecular_affine_stress_oracle(f, active0, r_up, r_dn):
    """Independent molecular stress oracle from position AD and the affine chain rule.

    For an open-system scalar ``E(R, r)``, applying ``F = I + eps`` to every
    nuclear and electron coordinate gives

    ``dE/deps_cd = sum_e (dE/dr_e,c) r_e,d + sum_I (dE/dR_I,c) R_I,d``.

    Differentiating positions first avoids a compiler-dependent mixed-precision
    cancellation in the monolithic strained graph without changing either the
    mathematical oracle or its element-wise tolerance.
    """
    g_active, gru, grd = jax.jit(jax.grad(f, argnums=(0, 1, 2)))(active0, r_up, r_dn)
    r_all = jnp.concatenate([r_up, r_dn], axis=0)
    grad_all = jnp.concatenate([gru, grd], axis=0)
    return jnp.einsum("ec,ed->cd", grad_all, r_all) + jnp.einsum("ic,id->cd", g_active["positions"], active0["positions"])


def _jit_strain_oracle(function, eps0):
    """Differentiate a primal strain function in the active precision lane."""
    return jax.jit(jax.jacrev(function))(eps0)


def _draw_well_conditioned_nonempty_sprime_cfg(ham, seed, scale, RT, alat, *, cond_max=700.0, max_tries=300):
    """Draw from the normal fixture family, rejecting near-node and empty-S' inputs.

    Acceptance uses only XLA-side physical preconditions before either
    derivative implementation is evaluated: the geminal must be reasonably
    conditioned and the test's S' channel must be active.  This does not select
    on analytic--autograd agreement and retains the deterministic distribution
    and seed origin.
    """
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    for attempt in range(max_tries):
        r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
        r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
        condition = float(np.linalg.cond(np.asarray(compute_geminal_all_elements(gd, r_up, r_dn))))
        if np.isfinite(condition) and condition < cond_max:
            sprime_weights, _ = gfmc._sprime_data(ham, r_up, r_dn, RT, alat)
            if float(np.sum(np.asarray(sprime_weights))) > 0.0:
                if _GDD_CONTEXT is not None:
                    _GDD_CONTEXT["generation"] = {
                        "seed": seed,
                        "attempts": attempt + 1,
                        "cond_g": condition,
                    }
                return r_up, r_dn
    raise AssertionError(
        f"XLA could not draw cond(G) < {cond_max:g} with non-empty S' from seed={seed} after {max_tries} candidates"
    )


# The analytic continuum-measurement force/stress is compared here against a ``jax.grad`` /
# ``jax.jacrev`` autograd oracle of ``_compute_local_energy_t``. The two are
# independent expression trees of the same derivative. Full precision uses the
# strict tier. Mixed mode evaluates both the production analytical route and the
# primal-AD oracle under the same mixed policy, using the central medium tier.
# The mixed comparisons remain blocking correctness evidence; smoke evidence is
# only supplemental.


def _build_ham(stem: str) -> Hamiltonian_data:
    (structure_data, aos_data, mos_up, mos_dn, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
        os.path.join(_TREXIO, stem), store_tuple=True
    )
    jastrow_data = Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None)  # jnone
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    return Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)


def _build_ham_gem(stem: str, gem: str) -> Hamiltonian_data:
    """Molecular AE jnone Hamiltonian whose geminal is rewritten to ``gem``.

    Production branches on the geminal representation inside the
    analytic GFMC force/stress path (the non-SD lambda leg of
    ``derivatives/kinetic.py``), but every analytic force/stress oracle used a
    plain ``SD`` geminal, so that branch had no autodiff-oracle evidence. The
    reviewed constructors are IMPORTED from the generated-matrix base and the
    optimizer helpers rather than re-encoded here.
    """
    from tests.test_jqmc_kernel._generated_matrix_base import _make_dense_mo_agp
    from tests.test_jqmc_kernel.optimizer_metadata_helpers import build_explicit_common_rank_agpn_from_sd

    ham = _build_ham(stem)
    base = ham.wavefunction_data.geminal_data
    if gem == "AGPn":
        new_gem = build_explicit_common_rank_agpn_from_sd(base)
    elif gem == "AGP_MO":
        new_gem = _make_dense_mo_agp(base, ion_positions=ham.structure_data.positions)
    else:
        raise ValueError(gem)
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=new_gem))


def _make_j3_ao(aos_data) -> Jastrow_three_body_data:
    """An ACTIVE three-body Jastrow on the raw AO leaf (``j123ao``): a random SYMMETRIC coupling
    block + a linear final column so every J3 term is nonzero (mirrors the 8-set harness
    ``_make_j3``).  A single real cart / sphe GTO leaf is the analytic J3 basis; a complex-twist
    leaf is first made the phase-free basis shared by all twists (no twisted J3), leaving the determinant's
    Bloch twist unchanged (a no-op for a real Gamma/molecular leaf)."""
    rng = np.random.default_rng(5)
    n = int(aos_data._num_orb)
    jm = rng.normal(size=(n, n + 1)) * 0.1
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)  # bosonic coupling block must be symmetric
    return Jastrow_three_body_data(orb_data=_phase_free_j3_orbitals(aos_data), j_matrix=jm)


def _build_ham_jastrow(stem: str, kind: str) -> Hamiltonian_data:
    """Molecular AE Hamiltonian with an analytic-polynomial Jastrow: ``j12`` = J1(exp) + J2(exp),
    ``j123ao`` = + an AO-leaf three-body Jastrow."""
    (structure_data, aos_data, mos_up, mos_dn, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
        os.path.join(_TREXIO, stem), store_tuple=True
    )
    n_atom = len(structure_data.positions)
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure_data
    )
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=core_electrons_of(coulomb_potential_data, n_atom),
        jastrow_1b_type="exp",
    )
    j3 = _make_j3_ao(aos_data) if kind == "j123ao" else None
    jastrow_data = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=j3)
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    return Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)


def _rot(seed: int):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return jnp.asarray(q)


def _e_L_t_ae_jnone(ham, r_up, r_dn, RT, alat):
    """Standalone continuum measurement energy returned by GFMC ``novar=0``.

    ``RT`` and ``alat`` remain in the signature because the force/stress test
    matrix shares one oracle interface with ECP/dltmove cases.  The all-electron
    measurement estimator itself contains no lattice Hamiltonian, sign-flip
    selector, or mesh rotation dependence.
    """
    wf = ham.wavefunction_data
    Tc_up, Tc_dn = compute_kinetic_energy_all_elements(wf, r_up, r_dn)
    Vee = compute_coulomb_potential_ee_dispatch(ham.coulomb_potential_data, r_up, r_dn)
    Vii = compute_coulomb_potential_ii_dispatch(ham.coulomb_potential_data)
    ei_up, ei_dn = compute_coulomb_potential_ei_dispatch(ham.coulomb_potential_data, r_up, r_dn)
    return jnp.real(Vee + Vii + jnp.sum(ei_up + Tc_up) + jnp.sum(ei_dn + Tc_dn))


# (stem, alat, seed, scale) chosen so the S' set is NON-empty (exercises the
# discretised correction), spanning single-atom multi-electron (N: 5up/2dn, both
# up- and dn-spin S' electrons), Li (3 electrons), and a two-atom molecule (H2).
_CASES = [
    ("N_ae_ccpvdz_cart.h5", 0.8, 0, 0.9),
    ("N_ae_ccpvdz_cart.h5", 0.5, 0, 0.9),
    ("Li_ae_ccpvdz_cart.h5", 0.5, 0, 0.9),
    ("H2_ae_ccpvdz_cart.h5", 1.0, 0, 0.9),
    ("H2_ae_ccpvdz_sphe.h5", 1.0, 0, 0.9),
]


@pytest.mark.light
@pytest.mark.parametrize("stem, alat, seed, scale", _CASES)
def test_gfmc_t_force_analytic_vs_autograd(stem, alat, seed, scale):
    ham = _build_ham(stem)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    # --- driver-equivalent jax.grad oracle (same mesh RT/alat) ---
    active0 = _extract_active_positions_from_ham(ham)

    def f_active(active, a, b):
        return _e_L_t_ae_jnone(_inject_active_positions(ham, active), a, b, RT, alat)

    gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)

    # --- analytic continuum-measurement force ---
    dR, dr_up, dr_dn = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat)

    atol, rtol = _gdd_get_tolerance_min(("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb"), "strict")
    _gdd_assert_allclose(np.asarray(dR), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_up), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_dn), np.asarray(grd_oracle), atol=atol, rtol=rtol)


def _e_L_t_ae_jnone_strained(ham, r_up, r_dn, RT, alat, strain):
    """Strained AE-jnone local energy, mirroring the driver's
    ``jqmc_gfmc._real_compute_local_energy_t_strained`` (F = I + eps applied to the
    Hamiltonian via ``_apply_strain`` + to the electrons/mesh via ``@ F.T``)."""
    F = jnp.eye(3, dtype=jnp.float64) + strain
    new_h = ham._apply_strain(F)
    return _e_L_t_ae_jnone(new_h, r_up @ F.T, r_dn @ F.T, RT @ F.T, alat)


@pytest.mark.light
@pytest.mark.parametrize("stem, alat, seed, scale", _CASES)
def test_gfmc_t_stress_analytic_vs_autograd(stem, alat, seed, scale):
    """Compare molecular AE GFMC_t stress with the independent position-AD affine virial."""
    ham = _build_ham(stem)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    # sanity: the multi-electron atom meshes (N: 4/7 or 2/7, Li: 2/3) have a NON-empty
    # S' set, so the discretised Coulomb/mesh/kinetic strain corrections are exercised
    # (not just the continuous stress). H2 (2 electrons near two protons) never produces
    # a sign-flip at physical scales -> empty S', a valid continuous-only multi-center
    # stress check (the S' correction must vanish there).
    w, _ = gfmc._sprime_data(ham, r_up, r_dn, RT, alat)
    if not stem.startswith("H2_"):
        assert float(np.sum(np.asarray(w))) > 0.0

    active0 = _extract_active_positions_from_ham(ham)

    def f_active(active, a, b):
        return _e_L_t_ae_jnone(_inject_active_positions(ham, active), a, b, RT, alat)

    stress_oracle = _jit_molecular_affine_stress_oracle(f_active, active0, r_up, r_dn)

    # --- analytic continuum-measurement stress ---
    stress = gfmc.grad_e_L_t_wrt_strain(ham, r_up, r_dn, RT, alat)

    atol, rtol = _gdd_get_tolerance_min(("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb"), "strict")
    _gdd_assert_allclose(np.asarray(stress), np.asarray(stress_oracle), atol=atol, rtol=rtol)


# --- WITH-Jastrow (j12 = J1+J2, j123ao = +AO-leaf J3) force + stress, molecular AE. ---
# Same (stem, alat, seed, scale) as the jnone cases -> non-empty S' (Li/N) + empty-S' H2.
# COST TIER IS PER ROW, not per function. Measured call-phase seconds on one
# reference box (full precision, the same host and load for every row, so the
# ratios -- not the absolute numbers -- are what the tier is read off):
#     j123ao rows   171 - 199 s   <- the AO-leaf three-body Jastrow Jacobian
#     j12 rows       47 -  60 s
# The ``j123ao`` rows are 3-4x the ``j12`` ones and dominate this file's wall
# clock; run as light rows they would make this file's light tier longer than
# its heavy tier. They are ``heavy``; the coverage is not lost, it is in the
# heavy tier, which is sharded.
_JCASES = [
    pytest.param(
        stem,
        alat,
        seed,
        scale,
        kind,
        marks=(pytest.mark.heavy if kind == "j123ao" else pytest.mark.light),
    )
    for (stem, alat, seed, scale) in _CASES
    for kind in ("j12", "j123ao")
]
# Derived from ``_CASES`` directly, not from ``_JCASES``: the latter now holds
# ``pytest.param`` objects, which do not unpack.
_JIDS = [f"{stem}-a{alat}-{kind}" for (stem, alat, seed, scale) in _CASES for kind in ("j12", "j123ao")]


@pytest.mark.parametrize("stem, alat, seed, scale, kind", _JCASES, ids=_JIDS)
def test_gfmc_t_force_analytic_vs_autograd_withJ(stem, alat, seed, scale, kind):
    """Analytic GFMC_t force with an analytic-polynomial Jastrow (j12 / j123ao) vs the SAME
    ``jax.grad(_compute_local_energy_t)`` oracle (the standalone e_L^t reproduction below uses the
    FULL wavefunction continuum kinetic energy so it carries the Jastrow automatically)."""
    ham = _build_ham_jastrow(stem, kind)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    active0 = _extract_active_positions_from_ham(ham)

    def f_active(active, a, b):
        return _e_L_t_ae_jnone(_inject_active_positions(ham, active), a, b, RT, alat)

    gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)

    dR, dr_up, dr_dn = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat)

    atol, rtol = _gdd_get_tolerance_min(
        ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap"), "strict"
    )
    _gdd_assert_allclose(np.asarray(dR), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_up), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_dn), np.asarray(grd_oracle), atol=atol, rtol=rtol)


@pytest.mark.parametrize("stem, alat, seed, scale, kind", _JCASES, ids=_JIDS)
def test_gfmc_t_stress_analytic_vs_autograd_withJ(stem, alat, seed, scale, kind):
    """Analytic GFMC_t stress with an analytic-polynomial Jastrow (j12 / j123ao) vs the SAME
    position-AD affine-virial oracle of the standalone continuum energy."""
    ham = _build_ham_jastrow(stem, kind)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    w, _ = gfmc._sprime_data(ham, r_up, r_dn, RT, alat)
    if not stem.startswith("H2_"):
        assert float(np.sum(np.asarray(w))) > 0.0

    active0 = _extract_active_positions_from_ham(ham)

    def f_active(active, a, b):
        return _e_L_t_ae_jnone(_inject_active_positions(ham, active), a, b, RT, alat)

    stress_oracle = _jit_molecular_affine_stress_oracle(f_active, active0, r_up, r_dn)

    stress = gfmc.grad_e_L_t_wrt_strain(ham, r_up, r_dn, RT, alat)

    atol, rtol = _gdd_get_tolerance_min(
        ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap"), "strict"
    )
    _gdd_assert_allclose(np.asarray(stress), np.asarray(stress_oracle), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------- #
# COMPLEX-TWIST periodic ECP force + stress (k != 0 Bloch phase).
#
# ``lithium_k025.h5`` is a 2-electron Li dimer at fractional twist k = (0.25, 0.25, 0.25) with an ECP.
# The Bloch phase exp(i k.r) makes the geminal G, the discretised hop weight t_{i,s} and the continuous
# kinetic Tc COMPLEX per sample; the forward fixed-node sign-flip / max clamp is lexicographic on the
# complex values and the observable is Re(d e_L^t / dX) (the driver Re-wraps e_L^t before grad/jacrev,
# jqmc_gfmc.py:1718-1719/1781-1782). The novar=0 measurement derivative runs
# holomorphically in the complex orbitals and Re-projects only at the boundary.
#
# Even though |Psi| is a nodeless pseudo-valence, a large-alat hop rotates the ratio Psi(cfg)/Psi(base)
# into the negative-real half-plane, so the propagation sign-flip predicate is
# non-empty for jnone/j123ao. This remains an independent adversarial-state pin;
# S-prime is not consumed by the measurement derivative. Marked heavy because
# the twist kinetic gradient is expensive to compile.
# ---------------------------------------------------------------------------- #
def _build_ham_twist(kind: str) -> Hamiltonian_data:
    """Complex-twist (k != 0) PBC-ECP Hamiltonian on ``lithium_k025.h5``: jnone / j12 / j123ao."""
    return _build_ham("lithium_k025.h5") if kind == "jnone" else _build_ham_jastrow("lithium_k025.h5", kind)


def _twist_config(ham, *, return_metadata: bool = False):
    """Return an adversarial twist configuration and propagation-membership flag.

    The scan prefers non-empty S-prime membership to keep propagation coverage
    live independently of the novar=0 measurement derivative.
    """
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    pos = np.asarray(ham.structure_data.positions)
    last = None
    for seed in range(12):
        rng = np.random.default_rng(seed)
        r_up = jnp.asarray(pos[:nu] + rng.normal(scale=0.7, size=(nu, 3)))
        r_dn = jnp.asarray(pos[:nd] + rng.normal(scale=0.7, size=(nd, 3)))
        for alat in (1.5, 2.0, 2.5):
            RT = _rot(seed + 2)
            w, _ = gfmc._sprime_data(ham, r_up, r_dn, RT, alat)
            last = (r_up, r_dn, RT, alat)
            if float(np.sum(np.asarray(w))) > 0.0:
                if _GDD_CONTEXT is not None:
                    _GDD_CONTEXT["generation"] = {"seed": 0, "attempts": seed + 1}
                result = (r_up, r_dn, RT, alat, True)
                return (*result, {"seed": 0, "attempts": seed + 1}) if return_metadata else result
    if _GDD_CONTEXT is not None:
        _GDD_CONTEXT["generation"] = {"seed": 0, "attempts": 12}
    result = (*last, False)
    return (*result, {"seed": 0, "attempts": 12}) if return_metadata else result


_TWIST_KINDS = ["jnone", "j12", "j123ao"]


@pytest.mark.heavy
@pytest.mark.parametrize("kind", _TWIST_KINDS)
def test_gfmc_t_force_analytic_vs_autograd_twist(kind):
    """Analytic GFMC_t FORCE, COMPLEX-TWIST periodic ECP (jnone / j12 / j123ao) vs
    ``jax.grad(jnp.real(_compute_local_energy_t))`` (the periodic-ECP tmove reproduction ``_e_L_t_ecp``,
    which Re-wraps the complex e_L^t like the driver).  The HF force observable is real: compare the
    analytic Re(d e_L/dX) to the oracle's d(Re e_L)/dX."""
    ham = _build_ham_twist(kind)
    r_up, r_dn, RT, alat, nonempty = _twist_config(ham)
    if kind != "j12":
        assert nonempty, f"expected live propagation S-prime membership for adversarial {kind} fixture"

    active0 = _extract_active_positions_from_ham(ham)

    def f_active(a, x, y):
        return _e_L_t_ecp(_inject_active_positions(ham, a), x, y, RT, alat)

    gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)

    dR, dr_up, dr_dn = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat)

    # The continuous complex derivative is Re-projected at the observable boundary.
    assert np.max(np.abs(np.imag(np.asarray(dR)))) == 0.0
    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb")
    if kind != "jnone":
        zones = zones + ("jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.real(np.asarray(dR)), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.real(np.asarray(dr_up)), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.real(np.asarray(dr_dn)), np.asarray(grd_oracle), atol=atol, rtol=rtol)


@pytest.mark.heavy
@pytest.mark.parametrize("kind", _TWIST_KINDS)
def test_gfmc_t_stress_analytic_vs_autograd_twist(kind):
    """Analytic GFMC_t STRESS, COMPLEX-TWIST periodic ECP (jnone / j12 / j123ao) vs
    ``jacrev(jnp.real(_compute_local_energy_t_strained))``. The novar=0
    measurement derivative includes the continuous periodic cell terms and
    Re-projects at the observable boundary.
    """
    ham = _build_ham_twist(kind)
    r_up, r_dn, RT, alat, nonempty = _twist_config(ham)
    if kind != "j12":
        assert nonempty

    eps0 = jnp.zeros((3, 3), dtype=jnp.float64)
    stress_oracle = _jit_strain_oracle(lambda e: _e_L_t_ecp_strained(ham, r_up, r_dn, RT, alat, e), eps0)

    stress = gfmc.grad_e_L_t_wrt_strain(ham, r_up, r_dn, RT, alat)

    assert np.max(np.abs(np.imag(np.asarray(stress)))) == 0.0
    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb")
    if kind != "jnone":
        zones = zones + ("jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.real(np.asarray(stress)), np.asarray(stress_oracle), atol=atol, rtol=rtol)


def _build_ham_twist_dtn() -> Hamiltonian_data:
    """Complex-twist (k != 0) PBC-ECP Hamiltonian with a DTN (u/chi/f cutoff-poly) Jastrow -- now IN
    scope (the DTN kinetic S'-reverse is holomorphic in the complex Bloch drift + Re-projected)."""
    (structure_data, aos_data, mos_up, mos_dn, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
        os.path.join(_TREXIO, "lithium_k025.h5"), store_tuple=True
    )
    r_ins = structure_data._wigner_seitz_inscribed_radius()
    dtn = Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=(1.0,),  # Li ccECP: Z=3, two core electrons
        pseudized_per_species=(True,),
        use_u=True,
        use_chi=True,
        use_f=True,
        cutoff_smoothness=4,
        n_order_u=4,
        n_order_chi=4,
        n_order_eN=2,
        n_order_ee=2,
        cutoff_ratio_u=3.0 / r_ins,
        cutoff_ratio_chi=3.0 / r_ins,
        cutoff_ratio_f=1.6 / (0.5 * r_ins),
        random_init=True,
        seed=3,
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=dtn
    )
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    return Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)


@pytest.mark.heavy
def test_gfmc_t_force_analytic_vs_autograd_twist_dtn():
    """Analytic GFMC_t FORCE, COMPLEX-TWIST periodic ECP + DTN (u/chi/f) Jastrow vs
    ``jax.grad(jnp.real(_compute_local_energy_t))`` (the periodic-ECP tmove reproduction ``_e_L_t_ecp``,
    which carries DTN through the continuous full-wavefunction kinetic). The
    measurement derivative is holomorphic and Re-projected at the boundary.
    """
    ham = _build_ham_twist_dtn()
    r_up, r_dn, RT, alat, nonempty = _twist_config(ham)
    assert nonempty, "expected live propagation S-prime membership on the adversarial DTN fixture"

    active0 = _extract_active_positions_from_ham(ham)

    def f_active(a, x, y):
        return _e_L_t_ecp(_inject_active_positions(ham, a), x, y, RT, alat)

    gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)

    dR, dr_up, dr_dn = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat)

    # The continuous complex derivative is Re-projected at the observable boundary.
    assert np.max(np.abs(np.imag(np.asarray(dR)))) == 0.0
    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb", "jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.real(np.asarray(dR)), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.real(np.asarray(dr_up)), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.real(np.asarray(dr_dn)), np.asarray(grd_oracle), atol=atol, rtol=rtol)


@pytest.mark.heavy
def test_gfmc_t_stress_analytic_vs_autograd_twist_dtn():
    """Analytic GFMC_t STRESS, COMPLEX-TWIST periodic ECP + DTN Jastrow vs
    ``jacrev(jnp.real(_compute_local_energy_t_strained))``. The continuous DTN
    measurement stress is holomorphic in the complex Bloch drift and
    Re-projected at the boundary.
    """
    ham = _build_ham_twist_dtn()
    r_up, r_dn, RT, alat, nonempty = _twist_config(ham)
    assert nonempty

    eps0 = jnp.zeros((3, 3), dtype=jnp.float64)
    stress_oracle = _jit_strain_oracle(lambda e: _e_L_t_ecp_strained(ham, r_up, r_dn, RT, alat, e), eps0)

    stress = gfmc.grad_e_L_t_wrt_strain(ham, r_up, r_dn, RT, alat)

    assert np.max(np.abs(np.imag(np.asarray(stress)))) == 0.0
    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb", "jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.real(np.asarray(stress)), np.asarray(stress_oracle), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------- #
# Molecular ECP force + stress.  With an ECP the ``opt`` selector uses the
# CONTINUOUS ``ei`` (not the discretised ``eid``), so the ``eid - ei`` S' Coulomb correction
# VANISHES and the ECP local + non-local potential lives entirely in the continuous ``e_L``
# (the fixed-node min/max split of the non-local grid reassembles to the full continuous sum).
# The analytic force/stress therefore reuse the verified VMC ECP continuous kernels and add
# ONLY the mesh (``S_i``) + weighted-kinetic (``Tc_i``) S' corrections.  ``non_local_move``
# defaults to ``tmove`` (the GFMC_t driver default), for which the non-local grid carries the
# FULL wavefunction ratio and the split reassembles to the standard continuous non-local energy.
# ---------------------------------------------------------------------------- #
def _build_ham_ecp(stem: str, kind: str) -> Hamiltonian_data:
    """Molecular ECP Hamiltonian: ``jnone`` (bare geminal) or an analytic-polynomial Jastrow
    (``j12`` = J1+J2, ``j123ao`` = +AO-leaf J3)."""
    return _build_ham(stem) if kind == "jnone" else _build_ham_jastrow(stem, kind)


def _e_L_t_ecp(ham, r_up, r_dn, RT, alat, non_local_move="tmove", rank_one_ratios=False, fused_base_inverse=False):
    """Standalone molecular-ECP continuum measurement energy (``novar=0``).

    The ECP local term and the localized non-local t-move/dltmove contribution
    remain part of the measurement estimator.  Lattice kinetic, sign-flip and
    max-clip terms are propagation-only and deliberately absent.

    ``rank_one_ratios`` selects how the non-local mesh ratios Psi(r'_g)/Psi(r) are
    formed; the energy is the same.  By default every mesh configuration rebuilds
    its geminal and takes ``det(G'_g)/det(G)``.  With ``rank_one_ratios=True`` the
    ratios come from the determinant lemma ``b_g^T G^{-1}`` with ``G^{-1}`` taken at
    the base configuration, through the production one-electron-move ratio forward.
    Only the moved electron is evaluated at the mesh point, so the electrons that
    did not move contribute one shared set of AO values.  Use this form only where
    the SSoT declares a conditioned reference (``conditioned-direct-ad-cases``).
    The Jastrow ratios of this form come from the production one-move ratio in the
    ``jastrow_ratio`` zone, not from a difference of two whole ``compute_Jastrow_part``
    values rounded in ``jastrow_eval`` (fp32 in the mixed lane).

    ``fused_base_inverse`` (requires ``rank_one_ratios``) takes the base ``G`` whose
    inverse forms the determinant-lemma ratios from the fused value/gradient/Laplacian
    orbital table (``determinant._geminal_for_grads_lap``), the same ``G`` the kinetic
    term of this energy inverts.  The kinetic and non-local terms then differentiate one
    wavefunction; with the value-only table they would describe two different mixed-lane
    geminal matrices (the rule stated in ``determinant._orbital_value_grad_lap_tables``).
    Full precision is unchanged up to round-off.
    """
    if fused_base_inverse and not rank_one_ratios:
        raise ValueError("fused_base_inverse selects the base inverse of the rank-one ratio form")
    wf = ham.wavefunction_data
    cp = ham.coulomb_potential_data
    Tc_up, Tc_dn = compute_kinetic_energy_all_elements(wf, r_up, r_dn)
    Vee = compute_coulomb_potential_ee_dispatch(cp, r_up, r_dn)
    Vii = compute_coulomb_potential_ii_dispatch(cp)
    ei_up, ei_dn = compute_coulomb_potential_ei_dispatch(cp, r_up, r_dn)
    ecp_local = compute_ecp_local_parts_all_pairs(cp, r_up, r_dn)
    if rank_one_ratios:
        base_geminal = _geminal_for_grads_lap if fused_base_inverse else compute_geminal_all_elements
        geminal_inv = jnp.linalg.inv(base_geminal(wf.geminal_data, r_up, r_dn))
        r_up_mesh, r_dn_mesh, V_nonlocal, _ = compute_ecp_non_local_parts_nearest_neighbors_fast_update(
            cp, wf, r_up, r_dn, RT, geminal_inv, flag_determinant_only=(non_local_move == "dltmove")
        )
    else:
        r_up_mesh, r_dn_mesh, V_nonlocal, _ = compute_ecp_non_local_parts_nearest_neighbors(
            cp, wf, r_up, r_dn, flag_determinant_only=(non_local_move == "dltmove"), RT=RT
        )
    ecp_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
    if non_local_move == "dltmove":
        # Driver ``dltmove`` (jqmc_gfmc._compute_V_elements_t): the non-local hop uses the
        # DETERMINANT-only amplitude ``V^det`` (flag_determinant_only=True above); the localised
        # (SP, ``V^det >= 0``) part goes to the diagonal WITHOUT the Jastrow ratio, while the
        # accepted (FN, ``V^det < 0``) part is re-weighted by ``jr_g = exp(J_mesh_g - J_ref)``.
        J_ref = compute_Jastrow_part(wf.jastrow_data, r_up, r_dn)
        J_mesh = jax.vmap(compute_Jastrow_part, in_axes=(None, 0, 0))(wf.jastrow_data, r_up_mesh, r_dn_mesh)
        jr = jnp.exp(J_mesh - J_ref)
        nondiag_ecp = jnp.sum(jnp.minimum(V_nonlocal, 0.0) * jr)
    else:
        nondiag_ecp = jnp.sum(jnp.minimum(V_nonlocal, 0.0))
    return jnp.real(Vee + Vii + jnp.sum(ei_up + Tc_up) + jnp.sum(ei_dn + Tc_dn) + ecp_local + ecp_SP + nondiag_ecp)


def _e_L_t_ecp_strained(ham, r_up, r_dn, RT, alat, strain, non_local_move="tmove"):
    """Strained molecular-ECP local energy (F = I + eps applied to the Hamiltonian via
    ``_apply_strain`` + to the electrons via ``@ F.T``; the quadrature rotation RT stays
    fixed), mirroring the driver's ``_real_compute_local_energy_t_strained``."""
    F = jnp.eye(3, dtype=jnp.float64) + strain
    new_h = ham._apply_strain(F)
    return _e_L_t_ecp(new_h, r_up @ F.T, r_dn @ F.T, RT, alat, non_local_move=non_local_move)


# (stem, alat, seed, scale): H2 (2 valence e, empty S' -> continuous-only ECP check), N2
# (non-empty S' for every Jastrow kind, multi-atom), Cl2 (2nd element; non-empty S' with J).
_ECP_CASES = [
    ("H2_ecp_ccpvdz_cart.h5", 1.0, 0, 0.9),
    ("N2_ecp_ccpvtz_cart.h5", 1.0, 0, 0.9),
    ("Cl2_ecp_ccpvtz_cart.h5", 0.8, 0, 0.9),
]
# ``nlm`` extends every ECP row over the LRDMC non-local move type: ``tmove`` (continuous
# reassembly) AND ``dltmove`` (determinant-clamped hop, Jastrow-reweighted FN part -- the closed-form
# ``delta`` correction ``gfmc.grad_e_L_t_wrt_{positions,strain}(non_local_move="dltmove")``).  For
# ``jnone`` and every empty-S' config ``dltmove == tmove`` (``jr = 1`` / no SP candidates); N2/Cl2 with
# a Jastrow are the NON-trivial ``delta`` rows (the un-clamped tmove form diverged ~1e4).
# Same per-row cost tier as ``_JCASES`` above. Measured call-phase seconds:
#     j123ao rows   213 - 238 s
#     j12 / jnone    47 -  68 s
_ECP_JCASES = [
    pytest.param(
        stem,
        alat,
        seed,
        scale,
        kind,
        nlm,
        marks=(pytest.mark.heavy if kind == "j123ao" else pytest.mark.light),
    )
    for (stem, alat, seed, scale) in _ECP_CASES
    for kind in ("jnone", "j12", "j123ao")
    for nlm in ("tmove", "dltmove")
]
# Derived from ``_ECP_CASES`` directly -- see the ``_JIDS`` note above.
_ECP_JIDS = [
    f"{stem}-a{alat}-{kind}-{nlm}"
    for (stem, alat, seed, scale) in _ECP_CASES
    for kind in ("jnone", "j12", "j123ao")
    for nlm in ("tmove", "dltmove")
]


@pytest.mark.parametrize("stem, alat, seed, scale, kind, nlm", _ECP_JCASES, ids=_ECP_JIDS)
def test_gfmc_t_force_analytic_vs_autograd_ecp(stem, alat, seed, scale, kind, nlm):
    """Analytic GFMC_t FORCE, molecular ECP (jnone / j12 / j123ao) x (tmove / dltmove) vs
    ``jax.grad(_compute_local_energy_t)`` (the ECP tmove/dltmove reproduction above)."""
    ham = _build_ham_ecp(stem, kind)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    # N2 has a NON-empty S' for every Jastrow kind -> the mesh/kinetic S' correction is exercised
    # (not just the continuous ECP force).  H2 is empty-S' (a continuous-only ECP check).
    w, _ = gfmc._sprime_data(ham, r_up, r_dn, RT, alat)
    if stem.startswith("N2_"):
        assert float(np.sum(np.asarray(w))) > 0.0

    active0 = _extract_active_positions_from_ham(ham)

    if stem == "N2_ecp_ccpvtz_cart.h5" and kind == "jnone":
        # Conditioned reference (SSoT ``conditioned-direct-ad-cases``, issue #541): the same
        # energy with determinant-lemma mesh ratios.  Re-evaluating every electron at each
        # of the Ne*Nv mesh configurations makes the AD derivative a cancellation of
        # O(sum |V_nl|) = O(10^2) terms, so fp32 AO round-off reaches 1e-4 in the mixed lane.
        def f_rank_one(active, a, b):
            return _e_L_t_ecp(_inject_active_positions(ham, active), a, b, RT, alat, non_local_move=nlm, rank_one_ratios=True)

        gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_rank_one, active0, r_up, r_dn)
    else:

        def f_active(active, a, b):
            return _e_L_t_ecp(_inject_active_positions(ham, active), a, b, RT, alat, non_local_move=nlm)

        gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)

    dR, dr_up, dr_dn = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat, non_local_move=nlm)

    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb")
    if kind != "jnone":
        zones = zones + ("jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.asarray(dR), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_up), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_dn), np.asarray(grd_oracle), atol=atol, rtol=rtol)


@pytest.mark.parametrize("stem, alat, seed, scale, kind, nlm", _ECP_JCASES, ids=_ECP_JIDS)
def test_gfmc_t_stress_analytic_vs_autograd_ecp(stem, alat, seed, scale, kind, nlm):
    """Analytic GFMC_t STRESS, molecular ECP (jnone / j12 / j123ao) x (tmove / dltmove) vs
    ``jacrev(_compute_local_energy_t_strained)`` (the ECP tmove/dltmove reproduction above)."""
    ham = _build_ham_ecp(stem, kind)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    w, _ = gfmc._sprime_data(ham, r_up, r_dn, RT, alat)
    if stem.startswith("N2_"):
        assert float(np.sum(np.asarray(w))) > 0.0

    eps0 = jnp.zeros((3, 3), dtype=jnp.float64)
    stress_oracle = _jit_strain_oracle(lambda e: _e_L_t_ecp_strained(ham, r_up, r_dn, RT, alat, e, non_local_move=nlm), eps0)

    stress = gfmc.grad_e_L_t_wrt_strain(ham, r_up, r_dn, RT, alat, non_local_move=nlm)

    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb")
    if kind != "jnone":
        zones = zones + ("jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.asarray(stress), np.asarray(stress_oracle), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------- #
# PERIODIC real-Gamma ECP force + stress.  The oracle ``_e_L_t_ecp`` /
# ``_e_L_t_ecp_strained`` (defined above) uses ONLY dispatch-level Coulomb/ECP kernels, which are
# periodic-aware, so it doubles as the PBC-ECP forward: Ewald ``V_ee``/``V_ii``, min-image ``ei``,
# periodic discretised kinetic mesh (home-cell wrap centralised in ``compute_AOs``), min-image ECP
# non-local mesh.  With an ECP the ``opt`` selector uses the CONTINUOUS ``ei`` (not ``eid``), so the
# ``eid - ei`` S' Coulomb correction VANISHES; the analytic PBC path is therefore the verified
# periodic VMC continuous force/stress (``assemble.compute_grad_e_L`` PBC-ECP branch) PLUS the mesh
# (``S_i``) + weighted-kinetic (``Tc_i``) S' corrections, using the periodic kinetic kernels
# (``_orbital_side_pbc`` / ``_orbital_side_r_pbc`` + periodic J1/J2/J3) and the periodic Pulay /
# kinetic STRESS (``pbc_stress``, S'-weighted).
#
# * ``lithium.h5`` (2e, ECP sphe-pgto, real Gamma): the smooth pseudo-valence never produces a
#   discretised sign-flip -> S' is ALWAYS empty, i.e. a continuous-only PBC-ECP check that ALSO
#   verifies the S' correction correctly vanishes.  Well-conditioned -> strict fp64.
# * no silicon.h5 rows (32e autograd oracle ~33+ GB each; see the _PBC_CASES note).
#   Non-empty-S' periodic coverage lives in the lithium_k025 twist rows + the arbitrary-weight
#   component tests below.
# ---------------------------------------------------------------------------- #
def _pbc_positions(ham, stem, seed, scale=0.5):
    """Spin-resolved electron coords for the PBC fixtures (lithium: plain random; its S' is empty
    regardless -- nodeless pseudo-valence)."""
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = rng.normal(scale=1.0, size=(nu, 3))
    r_dn = rng.normal(scale=1.0, size=(nd, 3))
    return jnp.asarray(r_up), jnp.asarray(r_dn)


# There are no silicon end-to-end rows: one 32e autograd-oracle test costs ~33+ GB RSS (measured;
# the j123ao stress jacrev is larger still), which exceeds the test memory budget, and running the
# suite at the edge of the memory budget is fragile.
# The coverage silicon carried is held by cheaper committed molecular and twist
# fixtures spanning the same continuous measurement components. Propagation
# membership liveness is pinned independently by the R05 regime test.
_PBC_CASES = [
    pytest.param("lithium.h5", 0.5, 0, kind, marks=pytest.mark.light, id=f"Li_pbc_gamma-{kind}")
    for kind in ("jnone", "j12", "j123ao")
]


@pytest.mark.parametrize("stem, alat, seed, kind", _PBC_CASES)
def test_gfmc_t_force_analytic_vs_autograd_pbc(stem, alat, seed, kind):
    """Analytic GFMC_t FORCE, periodic real-Gamma ECP (jnone / j12 / j123ao) vs
    ``jax.grad(_compute_local_energy_t)`` (the periodic-aware ECP tmove reproduction ``_e_L_t_ecp``)."""
    ham = _build_ham_ecp(stem, kind)
    r_up, r_dn = _pbc_positions(ham, stem, seed)
    RT = _rot(seed + 2)

    # lithium's S' is always empty (continuous-only PBC-ECP check -- the S' correction must
    # vanish there); non-empty-S' periodic coverage lives in the twist + component tests.
    active0 = _extract_active_positions_from_ham(ham)

    def f_active(active, a, b):
        return _e_L_t_ecp(_inject_active_positions(ham, active), a, b, RT, alat)

    gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)

    dR, dr_up, dr_dn = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat)

    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb")
    if kind != "jnone":
        zones = zones + ("jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.asarray(dR), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_up), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_dn), np.asarray(grd_oracle), atol=atol, rtol=rtol)


@pytest.mark.parametrize("stem, alat, seed, kind", _PBC_CASES)
def test_gfmc_t_stress_analytic_vs_autograd_pbc(stem, alat, seed, kind):
    """Analytic GFMC_t STRESS, periodic real-Gamma ECP (jnone / j12 / j123ao) vs
    ``jacrev(_compute_local_energy_t_strained)``.  The periodic strain deforms the cell, the atoms,
    the electrons AND the LRDMC mesh; the S' mesh/kinetic corrections carry the periodic cell terms
    (mesh via ``pbc_stress.stress_lnPsi_pbc`` per config, kinetic via the S'-weighted periodic
    kinetic stress ``pbc_stress.grad_T_wrt_strain_*``)."""
    ham = _build_ham_ecp(stem, kind)
    r_up, r_dn = _pbc_positions(ham, stem, seed)
    RT = _rot(seed + 2)

    eps0 = jnp.zeros((3, 3), dtype=jnp.float64)
    stress_oracle = _jit_strain_oracle(lambda e: _e_L_t_ecp_strained(ham, r_up, r_dn, RT, alat, e), eps0)

    stress = gfmc.grad_e_L_t_wrt_strain(ham, r_up, r_dn, RT, alat)

    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb")
    if kind != "jnone":
        zones = zones + ("jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.asarray(stress), np.asarray(stress_oracle), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------- #
# PERIODIC-Gamma dltmove: the ``dltmove`` ECP non-local Jastrow-reweight
# correction ``delta`` is now analytic for PBC too -- the FORCE reuses the min-image-aware molecular
# ``ecp_force.grad_V_ecp_nl_dltmove`` r/R branches verbatim, and the STRESS uses the periodic-Pulay-
# stress ``gfmc._dltmove_delta_strain_pbc`` (the only eps piece needing the cell terms).  ``jnone +
# dltmove == tmove`` (no correction), so only j12 / j123ao are non-trivial ``delta`` rows.  Oracle =
# the SAME driver ``jax.grad`` / ``jax.jacrev`` of ``_e_L_t_ecp`` with ``non_local_move="dltmove"``
# used for the molecular dltmove rows (periodic-aware: min-image ECP non-local + Ewald).
# ---------------------------------------------------------------------------- #
_PBC_DLT_CASES = [
    pytest.param("lithium.h5", 0.5, 0, kind, marks=pytest.mark.light, id=f"Li_pbc_dltmove-{kind}") for kind in ("j12", "j123ao")
]


@pytest.mark.parametrize("stem, alat, seed, kind", _PBC_DLT_CASES)
def test_gfmc_t_force_analytic_vs_autograd_pbc_dltmove(stem, alat, seed, kind):
    """Analytic GFMC_t FORCE, periodic real-Gamma ECP + Jastrow, ``dltmove`` (the closed carve-out)."""
    ham = _build_ham_ecp(stem, kind)
    r_up, r_dn = _pbc_positions(ham, stem, seed)
    RT = _rot(seed + 2)
    active0 = _extract_active_positions_from_ham(ham)

    def f_active(active, a, b):
        return _e_L_t_ecp(_inject_active_positions(ham, active), a, b, RT, alat, non_local_move="dltmove")

    gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)
    dR, dr_up, dr_dn = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat, non_local_move="dltmove")

    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb", "jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.asarray(dR), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_up), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_dn), np.asarray(grd_oracle), atol=atol, rtol=rtol)


@pytest.mark.parametrize("stem, alat, seed, kind", _PBC_DLT_CASES)
def test_gfmc_t_stress_analytic_vs_autograd_pbc_dltmove(stem, alat, seed, kind):
    """Analytic GFMC_t STRESS, periodic real-Gamma ECP + Jastrow, ``dltmove`` (the closed carve-out):
    the periodic ``_dltmove_delta_strain_pbc`` (full periodic Pulay stress) must reproduce the driver
    ``jacrev`` of the strained discretised dltmove e_L."""
    ham = _build_ham_ecp(stem, kind)
    r_up, r_dn = _pbc_positions(ham, stem, seed)
    RT = _rot(seed + 2)

    eps0 = jnp.zeros((3, 3), dtype=jnp.float64)
    stress_oracle = _jit_strain_oracle(
        lambda e: _e_L_t_ecp_strained(ham, r_up, r_dn, RT, alat, e, non_local_move="dltmove"), eps0
    )
    stress = gfmc.grad_e_L_t_wrt_strain(ham, r_up, r_dn, RT, alat, non_local_move="dltmove")

    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb", "jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.asarray(stress), np.asarray(stress_oracle), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------- #
# Molecular ECP + DTN GFMC_t STRESS: the stale guard (which raised NotImplementedError on
# the premise that ``assemble.compute_grad_e_L(wrt="eps")`` routed molecular ECP to ``_auto``, which
# NaN'd through DTN's higher-order AD) is REMOVED -- the continuous molecular-ECP e_L stress is now
# closed-form analytic (``_stress_eL`` + ``ecp_force.grad_V_ecp_wrt_strain_mol`` +
# ``dtn.grad_T_dtn_wrt_strain``), so the whole molecular-ECP-DTN stress composes.
# ---------------------------------------------------------------------------- #
def _build_ham_ecp_dtn(stem: str) -> Hamiltonian_data:
    """Molecular ECP Hamiltonian with a DTN (Drummond--Towler--Needs cutoff-poly u/chi/f) Jastrow."""
    (structure_data, aos_data, mos_up, mos_dn, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
        os.path.join(_TREXIO, stem), store_tuple=True
    )
    dtn = Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=tuple(
            float(
                np.mean(
                    (np.asarray(structure_data.atomic_numbers) - np.asarray(coulomb_potential_data.z_cores))[
                        np.asarray(structure_data.atomic_numbers) == Z
                    ]
                )
            )
            for Z in np.unique(structure_data.atomic_numbers)
        ),
        pseudized_per_species=tuple(
            bool(np.all(coulomb_potential_data.pseudized_per_atom[np.asarray(structure_data.atomic_numbers) == Z]))
            for Z in np.unique(structure_data.atomic_numbers)
        ),
        use_u=True,
        use_chi=True,
        use_f=True,
        cutoff_smoothness=4,
        n_order_u=4,
        n_order_chi=4,
        n_order_eN=2,
        n_order_ee=2,
        L_u=3.0,
        L_chi=3.0,
        L_f=1.6,
        random_init=True,
        seed=3,
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=dtn
    )
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    return Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)


@pytest.mark.light
@pytest.mark.parametrize("stem, alat", [("H2_ecp_ccpvdz_cart.h5", 1.0), ("N2_ecp_ccpvtz_cart.h5", 1.0)])
@pytest.mark.parametrize("nlm", ["tmove", "dltmove"])
def test_gfmc_t_stress_analytic_vs_autograd_ecp_dtn(stem, alat, nlm):
    """Analytic GFMC_t STRESS, molecular ECP + DTN Jastrow (the removed stale-guard case).

    ``tmove`` is compared directly with ``jacrev(_compute_local_energy_t_strained)``.  For
    ``dltmove`` the same-precision oracle is deliberately conditioned as two independently
    differentiated terms: the continuous ``tmove`` energy and the closed-form scalar
    ``dltmove - tmove`` correction.  Comparing the two terms separately avoids judging the
    mixed-precision round-off of two O(10--100) derivatives against an O(0.1) cancellation
    residual, while still requiring both public production routes and the correction itself
    to agree with independent AD under the centrally resolved tolerance.
    """
    ham = _build_ham_ecp_dtn(stem)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(0)
    r_up = jnp.asarray(rng.normal(scale=0.9, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=0.9, size=(nd, 3)))
    RT = _rot(1)

    eps0 = jnp.zeros((3, 3), dtype=jnp.float64)
    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    stress_t_oracle = _jit_strain_oracle(
        lambda e: _e_L_t_ecp_strained(ham, r_up, r_dn, RT, alat, e, non_local_move="tmove"), eps0
    )
    stress_t = gfmc.grad_e_L_t_wrt_strain(ham, r_up, r_dn, RT, alat, non_local_move="tmove")
    assert not np.isnan(np.asarray(stress_t_oracle)).any()
    _gdd_assert_allclose(np.asarray(stress_t), np.asarray(stress_t_oracle), atol=atol, rtol=rtol)

    if nlm == "dltmove":

        def strained_delta(strain):
            F = jnp.eye(3, dtype=jnp.float64) + strain
            return ecp_force._dltmove_delta(
                ham._apply_strain(F),
                r_up @ F.T,
                r_dn @ F.T,
                RT,
            )

        delta_oracle = _jit_strain_oracle(strained_delta, eps0)
        stress_dlt = gfmc.grad_e_L_t_wrt_strain(ham, r_up, r_dn, RT, alat, non_local_move="dltmove")
        assert not np.isnan(np.asarray(delta_oracle)).any()
        _gdd_assert_allclose(
            np.asarray(stress_dlt - stress_t),
            np.asarray(delta_oracle),
            atol=atol,
            rtol=rtol,
        )


# ---------------------------------------------------------------------------- #
# PERIODIC ALL-ELECTRON force + stress, real Gamma and complex twist.
#
# The novar=0 estimator differentiates the continuous kinetic + Ewald
# measurement energy; the discretized electron-ion clamp and S-prime terms are
# propagation-only. The committed QE hydrogen fixture below checks the full
# periodic-AE measurement derivative against an independent autograd oracle.
# Its non-empty propagation S-prime set is retained only as an adversarial-state
# invariant and does not enter force/stress.
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# GENUINE periodic ALL-ELECTRON complex-twist FORCE + STRESS (committed QE hydrogen fixture).
#
# ``tests/qe_example_files/H-ae_k025`` is a committed Quantum-Espresso .save for an 8-hydrogen cell
# at fractional twist k = (1/4, 1/4, 1/4) with the ALL-ELECTRON ``H.coulomb-ae.UPF`` (bare Coulomb,
# NO pseudopotential).  We build the jQMC Hamiltonian AT TEST TIME (temporary, discarded): load the
# QE reference, project the occupied bands onto the hardcoded cc-pVDZ-derived H all-electron pGTO
# basis (``generate_ae_even_tempered_shells``), and wrap with a bare-Coulomb (``ecp_flag=False``,
# full-Z) periodic ``Coulomb_potential_data`` + Ewald table via ``build_hamiltonian_from_wavefunction
# (ecp_by_element=None)``.  This is a GENUINE periodic all-electron system (real physics, charge
# neutral, complex Bloch orbitals), not a synthetic all-electron construction.
#
# The twist makes the geminal and continuous kinetic intermediates complex; the
# observable remains ``Re(d e_L/dX)``. The committed fixture is self-contained,
# and its non-empty propagation membership is checked independently.
# ---------------------------------------------------------------------------- #
_QE_H_AE_SAVE = os.path.join(qe_example_files_root(), "H-ae_k025", "out", "h_k025.save")


@functools.lru_cache(maxsize=1)
def _h_ae_twist_pieces():
    """(structure, complex pGTO AO leaf, MO geminal) from the committed H-ae_k025 QE output, built
    ONCE per process (the projection is a few seconds).  All-electron (bare Coulomb) H at k =
    (1/4,1/4,1/4) -> a complex Bloch pGTO leaf + complex128 MO coefficients."""
    from jqmc.pw_pgto_projection import project_pw_to_pgto
    from jqmc.qe_to_jqmc import build_wavefunction_from_projection
    from jqmc.qe_wrapper import load_qe_reference
    from jqmc.upf_wrapper import generate_ae_even_tempered_shells

    ref = load_qe_reference(_QE_H_AE_SAVE)
    g_target = float(np.sqrt(2.0 * ref.ecutwfc))
    shells = [generate_ae_even_tempered_shells(g_target=g_target, upf_lmax=0, atomic_number=1) for _ in ref.atom_labels]
    proj = project_pw_to_pgto(ref, shells)
    return build_wavefunction_from_projection(ref, proj, kind="pgto")


def _build_ham_h_ae_twist(kind: str) -> Hamiltonian_data:
    """GENUINE periodic all-electron complex-twist Hamiltonian on the committed H-ae_k025 QE fixture:
    ``jnone`` (bare geminal), ``j12`` = exp J1 + exp J2, ``j123ao`` = + an AO-leaf three-body Jastrow.
    ``ecp_by_element=None`` -> bare-Coulomb full-Z AE Coulomb + Ewald (H core_electrons = 0)."""
    from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction

    sd, aos, gem = _h_ae_twist_pieces()
    if kind == "jnone":
        return build_hamiltonian_from_wavefunction(sd, aos, gem, j1_param=None, j2_param=None, ecp_by_element=None)
    j3 = _make_j3_ao(aos) if kind == "j123ao" else None
    return build_hamiltonian_from_wavefunction(
        sd, aos, gem, j1_param=1.0, j1_type="exp", j2_param=1.0, j2_type="exp", ecp_by_element=None, jastrow_three_body_data=j3
    )


_H_AE_TWIST_KINDS = ["jnone", "j12", "j123ao"]


@pytest.mark.heavy
@pytest.mark.parametrize("kind", _H_AE_TWIST_KINDS)
def test_gfmc_t_force_analytic_vs_autograd_twist_ae(kind):
    """Analytic GFMC_t FORCE, GENUINE complex-twist periodic ALL-ELECTRON (jnone / j12 / j123ao) on the
    committed QE hydrogen fixture, vs ``jax.grad(jnp.real(_compute_local_energy_t))`` (the periodic-aware
    AE reproduction ``_e_L_t_ae_jnone``, which Re-wraps the complex e_L^t like the driver).  The HF force
    observable is real: the analytic continuum-measurement derivative Re-projects at the boundary."""
    ham = _build_ham_h_ae_twist(kind)
    r_up, r_dn, RT, alat, nonempty = _twist_config(ham)
    # Keep the propagation membership live independently of measurement parity.
    assert nonempty, f"expected live propagation S-prime membership for twisted-AE {kind}"

    active0 = _extract_active_positions_from_ham(ham)

    def f_active(a, x, y):
        return _e_L_t_ae_jnone(_inject_active_positions(ham, a), x, y, RT, alat)

    gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)

    dR, dr_up, dr_dn = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat)

    # The continuous complex derivative is Re-projected at the observable boundary.
    assert np.max(np.abs(np.imag(np.asarray(dR)))) == 0.0
    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb")
    if kind != "jnone":
        zones = zones + ("jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.real(np.asarray(dR)), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.real(np.asarray(dr_up)), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.real(np.asarray(dr_dn)), np.asarray(grd_oracle), atol=atol, rtol=rtol)


@pytest.mark.heavy
@pytest.mark.parametrize("kind", _H_AE_TWIST_KINDS)
def test_gfmc_t_stress_analytic_vs_autograd_twist_ae(kind):
    """Analytic GFMC_t STRESS, GENUINE complex-twist periodic ALL-ELECTRON (jnone / j12 / j123ao) on the
    committed QE hydrogen fixture, vs
    ``jacrev(jnp.real(_compute_local_energy_t_strained))``. The strain deforms
    the continuous periodic measurement geometry; propagation mesh derivatives
    are absent.
    """
    ham = _build_ham_h_ae_twist(kind)
    r_up, r_dn, RT, alat, nonempty = _twist_config(ham)
    assert nonempty

    eps0 = jnp.zeros((3, 3), dtype=jnp.float64)
    stress_oracle = _jit_strain_oracle(lambda e: _e_L_t_ae_jnone_strained(ham, r_up, r_dn, RT, alat, e), eps0)

    stress = gfmc.grad_e_L_t_wrt_strain(ham, r_up, r_dn, RT, alat)

    assert np.max(np.abs(np.imag(np.asarray(stress)))) == 0.0
    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb")
    if kind != "jnone":
        zones = zones + ("jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.real(np.asarray(stress)), np.asarray(stress_oracle), atol=atol, rtol=rtol)


# ============================================================================ #
# GFMC_n / LRDMC number-estimator.
#
# CRUX: GFMC_n and GFMC_t differ ONLY in the driver-level branching weight ``w_L``
# (GFMC_t uses ``w_L`` directly; GFMC_n uses the fixed-``n`` number projection with a
# ``V_diag / E`` division -- jqmc_gfmc.py:2286).  The differentiated Hellmann--Feynman
# LOCAL ENERGY is identical: ``jqmc_gfmc._compute_local_energy_n(use_fast_update=False)``
# is byte-for-byte ``_compute_local_energy_t`` (the ``use_fast_update=True`` SVD-geminal
# path is value-identical and used only in the hot move loop; the production GFMC_n FORCE
# kernel is called with ``use_fast_update=False`` at jqmc_gfmc.py:7012-7021 and the STRESS
# kernel hard-codes ``use_fast_update=False`` at jqmc_gfmc.py:6776).  So the ``_e_L_n_*``
# oracles below reproduce ``_compute_local_energy_n`` == the ``_t`` forward, and the analytic
# ``gfmc.grad_e_L_n_*`` delegate to the verified ``_t`` continuum-measurement implementation. These tests give the
# GFMC_n public API its OWN analytic-vs-autograd coverage + an explicit ``_n == _t`` identity.
# ============================================================================ #
def _e_L_n_ae_jnone(ham, r_up, r_dn, RT, alat):
    """Standalone AE-jnone reproduction of jqmc_gfmc._compute_local_energy_n
    (use_fast_update=False).  That forward is byte-for-byte ``_compute_local_energy_t``
    (only the fast-update SVD geminal-inverse move-loop path differs, and the production
    GFMC_n force/stress kernels use use_fast_update=False), so this equals ``_e_L_t_ae_jnone``."""
    return _e_L_t_ae_jnone(ham, r_up, r_dn, RT, alat)


def _e_L_n_ae_jnone_strained(ham, r_up, r_dn, RT, alat, strain):
    """Strained AE-jnone GFMC_n local energy == the ``_t`` strained forward (mirrors
    jqmc_gfmc._real_compute_local_energy_n_strained, which hard-codes use_fast_update=False)."""
    return _e_L_t_ae_jnone_strained(ham, r_up, r_dn, RT, alat, strain)


def _e_L_n_ecp(ham, r_up, r_dn, RT, alat, non_local_move="tmove", rank_one_ratios=False):
    """Standalone molecular-ECP reproduction of jqmc_gfmc._compute_local_energy_n (ECP branch,
    use_fast_update=False) == the ``_t`` ECP forward."""
    return _e_L_t_ecp(ham, r_up, r_dn, RT, alat, non_local_move=non_local_move, rank_one_ratios=rank_one_ratios)


def _e_L_n_ecp_strained(ham, r_up, r_dn, RT, alat, strain, non_local_move="tmove"):
    """Strained molecular-ECP GFMC_n local energy == the ``_t`` strained ECP forward."""
    return _e_L_t_ecp_strained(ham, r_up, r_dn, RT, alat, strain, non_local_move=non_local_move)


@pytest.mark.light
@pytest.mark.parametrize("stem, alat, seed, scale", _CASES)
def test_gfmc_n_force_analytic_vs_autograd(stem, alat, seed, scale):
    """Analytic GFMC_n FORCE (molecular AE jnone) vs ``jax.grad(_compute_local_energy_n)``."""
    ham = _build_ham(stem)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    active0 = _extract_active_positions_from_ham(ham)

    def f_active(active, a, b):
        return _e_L_n_ae_jnone(_inject_active_positions(ham, active), a, b, RT, alat)

    gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)

    dR, dr_up, dr_dn = gfmc.grad_e_L_n_wrt_positions(ham, r_up, r_dn, RT, alat)

    atol, rtol = _gdd_get_tolerance_min(("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb"), "strict")
    _gdd_assert_allclose(np.asarray(dR), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_up), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_dn), np.asarray(grd_oracle), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem, alat, seed, scale", _CASES)
def test_gfmc_n_stress_analytic_vs_autograd(stem, alat, seed, scale):
    """Analytic GFMC_n molecular AE stress vs the independent position-AD affine virial."""
    ham = _build_ham(stem)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    w, _ = gfmc._sprime_data(ham, r_up, r_dn, RT, alat)
    if not stem.startswith("H2_"):
        assert float(np.sum(np.asarray(w))) > 0.0

    active0 = _extract_active_positions_from_ham(ham)

    def f_active(active, a, b):
        return _e_L_n_ae_jnone(_inject_active_positions(ham, active), a, b, RT, alat)

    stress_oracle = _jit_molecular_affine_stress_oracle(f_active, active0, r_up, r_dn)

    stress = gfmc.grad_e_L_n_wrt_strain(ham, r_up, r_dn, RT, alat)

    atol, rtol = _gdd_get_tolerance_min(("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb"), "strict")
    _gdd_assert_allclose(np.asarray(stress), np.asarray(stress_oracle), atol=atol, rtol=rtol)


@pytest.mark.parametrize("stem, alat, seed, scale, kind", _JCASES, ids=_JIDS)
def test_gfmc_n_force_analytic_vs_autograd_withJ(stem, alat, seed, scale, kind):
    """Analytic GFMC_n FORCE with a polynomial Jastrow (j12 / j123ao), molecular AE."""
    ham = _build_ham_jastrow(stem, kind)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    active0 = _extract_active_positions_from_ham(ham)

    def f_active(active, a, b):
        return _e_L_n_ae_jnone(_inject_active_positions(ham, active), a, b, RT, alat)

    gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)

    dR, dr_up, dr_dn = gfmc.grad_e_L_n_wrt_positions(ham, r_up, r_dn, RT, alat)

    atol, rtol = _gdd_get_tolerance_min(
        ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap"), "strict"
    )
    _gdd_assert_allclose(np.asarray(dR), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_up), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_dn), np.asarray(grd_oracle), atol=atol, rtol=rtol)


@pytest.mark.parametrize("stem, alat, seed, scale, kind", _JCASES, ids=_JIDS)
def test_gfmc_n_stress_analytic_vs_autograd_withJ(stem, alat, seed, scale, kind):
    """Analytic GFMC_n molecular AE+Jastrow stress vs the position-AD affine virial."""
    ham = _build_ham_jastrow(stem, kind)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    w, _ = gfmc._sprime_data(ham, r_up, r_dn, RT, alat)
    if not stem.startswith("H2_"):
        assert float(np.sum(np.asarray(w))) > 0.0

    active0 = _extract_active_positions_from_ham(ham)

    def f_active(active, a, b):
        return _e_L_n_ae_jnone(_inject_active_positions(ham, active), a, b, RT, alat)

    stress_oracle = _jit_molecular_affine_stress_oracle(f_active, active0, r_up, r_dn)

    stress = gfmc.grad_e_L_n_wrt_strain(ham, r_up, r_dn, RT, alat)

    atol, rtol = _gdd_get_tolerance_min(
        ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap"), "strict"
    )
    _gdd_assert_allclose(np.asarray(stress), np.asarray(stress_oracle), atol=atol, rtol=rtol)


@pytest.mark.parametrize("stem, alat, seed, scale, kind, nlm", _ECP_JCASES, ids=_ECP_JIDS)
def test_gfmc_n_force_analytic_vs_autograd_ecp(stem, alat, seed, scale, kind, nlm):
    """Analytic GFMC_n FORCE, molecular ECP (jnone / j12 / j123ao) x (tmove / dltmove) vs
    ``jax.grad(_compute_local_energy_n)`` (the ECP tmove/dltmove reproduction)."""
    ham = _build_ham_ecp(stem, kind)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    w, _ = gfmc._sprime_data(ham, r_up, r_dn, RT, alat)
    if stem.startswith("N2_"):
        assert float(np.sum(np.asarray(w))) > 0.0

    active0 = _extract_active_positions_from_ham(ham)

    if stem == "N2_ecp_ccpvtz_cart.h5" and kind == "jnone":
        # Conditioned reference (SSoT ``conditioned-direct-ad-cases``, issue #541): the same
        # energy with determinant-lemma mesh ratios.  Re-evaluating every electron at each
        # of the Ne*Nv mesh configurations makes the AD derivative a cancellation of
        # O(sum |V_nl|) = O(10^2) terms, so fp32 AO round-off reaches 1e-4 in the mixed lane.
        def f_rank_one(active, a, b):
            return _e_L_n_ecp(_inject_active_positions(ham, active), a, b, RT, alat, non_local_move=nlm, rank_one_ratios=True)

        gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_rank_one, active0, r_up, r_dn)
    else:

        def f_active(active, a, b):
            return _e_L_n_ecp(_inject_active_positions(ham, active), a, b, RT, alat, non_local_move=nlm)

        gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)

    dR, dr_up, dr_dn = gfmc.grad_e_L_n_wrt_positions(ham, r_up, r_dn, RT, alat, non_local_move=nlm)

    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb")
    if kind != "jnone":
        zones = zones + ("jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.asarray(dR), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_up), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_dn), np.asarray(grd_oracle), atol=atol, rtol=rtol)


@pytest.mark.parametrize("stem, alat, seed, scale, kind, nlm", _ECP_JCASES, ids=_ECP_JIDS)
def test_gfmc_n_stress_analytic_vs_autograd_ecp(stem, alat, seed, scale, kind, nlm):
    """Analytic GFMC_n STRESS, molecular ECP (jnone / j12 / j123ao) x (tmove / dltmove) vs
    ``jacrev(_compute_local_energy_n_strained)`` (the ECP tmove/dltmove reproduction)."""
    ham = _build_ham_ecp(stem, kind)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    w, _ = gfmc._sprime_data(ham, r_up, r_dn, RT, alat)
    if stem.startswith("N2_"):
        assert float(np.sum(np.asarray(w))) > 0.0

    eps0 = jnp.zeros((3, 3), dtype=jnp.float64)
    stress_oracle = _jit_strain_oracle(lambda e: _e_L_n_ecp_strained(ham, r_up, r_dn, RT, alat, e, non_local_move=nlm), eps0)

    stress = gfmc.grad_e_L_n_wrt_strain(ham, r_up, r_dn, RT, alat, non_local_move=nlm)

    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb")
    if kind != "jnone":
        zones = zones + ("jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.asarray(stress), np.asarray(stress_oracle), atol=atol, rtol=rtol)


@pytest.mark.light
def test_gfmc_n_equals_t_force_and_stress():
    """The GFMC_n and GFMC_t Hellmann--Feynman FORCE + STRESS are bit-identical: the two
    estimators differ only in the driver branching weight, not in the differentiated e_L.
    Documents the estimator-independence and pins the delegation."""
    ham = _build_ham_jastrow("Li_ae_ccpvdz_cart.h5", "j123ao")
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(0)
    r_up = jnp.asarray(rng.normal(scale=0.9, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=0.9, size=(nd, 3)))
    RT = _rot(1)
    alat = 0.5

    dR_n, dru_n, drd_n = gfmc.grad_e_L_n_wrt_positions(ham, r_up, r_dn, RT, alat)
    dR_t, dru_t, drd_t = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat)
    _gdd_assert_array_equal(np.asarray(dR_n), np.asarray(dR_t))
    _gdd_assert_array_equal(np.asarray(dru_n), np.asarray(dru_t))
    _gdd_assert_array_equal(np.asarray(drd_n), np.asarray(drd_t))

    s_n = gfmc.grad_e_L_n_wrt_strain(ham, r_up, r_dn, RT, alat)
    s_t = gfmc.grad_e_L_t_wrt_strain(ham, r_up, r_dn, RT, alat)
    _gdd_assert_array_equal(np.asarray(s_n), np.asarray(s_t))


@pytest.mark.heavy
@pytest.mark.parametrize("kind", _TWIST_KINDS)
def test_gfmc_n_equals_t_twist(kind):
    """GFMC_n and GFMC_t HF FORCE + STRESS are bit-identical on the COMPLEX-TWIST periodic-ECP path too
    (``e_L^n == e_L^t``; the ``_n`` entry points delegate to the ``_t`` continuum-measurement implementation). Pins the
    delegation at k != 0 and, transitively with ``test_gfmc_t_*_twist``, gives GFMC_n its twist
    analytic-vs-autograd coverage."""
    ham = _build_ham_twist(kind)
    r_up, r_dn, RT, alat, _ = _twist_config(ham)

    dR_n, dru_n, drd_n = gfmc.grad_e_L_n_wrt_positions(ham, r_up, r_dn, RT, alat)
    dR_t, dru_t, drd_t = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat)
    _gdd_assert_array_equal(np.asarray(dR_n), np.asarray(dR_t))
    _gdd_assert_array_equal(np.asarray(dru_n), np.asarray(dru_t))
    _gdd_assert_array_equal(np.asarray(drd_n), np.asarray(drd_t))

    s_n = gfmc.grad_e_L_n_wrt_strain(ham, r_up, r_dn, RT, alat)
    s_t = gfmc.grad_e_L_t_wrt_strain(ham, r_up, r_dn, RT, alat)
    _gdd_assert_array_equal(np.asarray(s_n), np.asarray(s_t))


@pytest.mark.heavy
def test_gfmc_n_force_analytic_vs_autograd_twist_dtn():
    """Analytic GFMC_n FORCE, COMPLEX-TWIST periodic ECP + DTN, vs the INDEPENDENT autograd oracle
    ``jax.grad(jnp.real(_e_L_t_ecp))`` -- NOT a comparison to GFMC_t.  The GFMC_n force differentiates
    the fixed-node Hamiltonian local energy ``_compute_local_energy_n(use_fast_update=False)``, which is
    line-for-line ``_compute_local_energy_t`` (gfmc.py module note; the driver-level number-estimator
    e_L^n with its V_diag/E division + E_scf is NOT what the force differentiates), so ``_e_L_t_ecp``
    reproduces exactly the local energy this force differentiates.  The HF force is real (Im == 0)."""
    ham = _build_ham_twist_dtn()
    r_up, r_dn, RT, alat, nonempty = _twist_config(ham)
    assert nonempty, "expected live propagation S-prime membership on the adversarial DTN fixture"

    active0 = _extract_active_positions_from_ham(ham)

    def f_active(a, x, y):
        return _e_L_t_ecp(_inject_active_positions(ham, a), x, y, RT, alat)

    gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)

    dR, dr_up, dr_dn = gfmc.grad_e_L_n_wrt_positions(ham, r_up, r_dn, RT, alat)

    assert np.max(np.abs(np.imag(np.asarray(dR)))) == 0.0
    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb", "jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.real(np.asarray(dR)), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.real(np.asarray(dr_up)), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.real(np.asarray(dr_dn)), np.asarray(grd_oracle), atol=atol, rtol=rtol)


@pytest.mark.heavy
def test_gfmc_n_stress_analytic_vs_autograd_twist_dtn():
    """Analytic GFMC_n STRESS, COMPLEX-TWIST periodic ECP + DTN, vs the INDEPENDENT autograd oracle
    ``jacrev(jnp.real(_e_L_t_ecp_strained))`` (the fixed-node Hamiltonian local energy the force
    differentiates == ``_compute_local_energy_n(use_fast_update=False)``).  Stress is real (Im == 0)."""
    ham = _build_ham_twist_dtn()
    r_up, r_dn, RT, alat, nonempty = _twist_config(ham)
    assert nonempty

    eps0 = jnp.zeros((3, 3), dtype=jnp.float64)
    stress_oracle = _jit_strain_oracle(lambda e: _e_L_t_ecp_strained(ham, r_up, r_dn, RT, alat, e), eps0)

    stress = gfmc.grad_e_L_n_wrt_strain(ham, r_up, r_dn, RT, alat)

    assert np.max(np.abs(np.imag(np.asarray(stress)))) == 0.0
    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb", "jastrow_eval", "jastrow_grad_lap")
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.real(np.asarray(stress)), np.asarray(stress_oracle), atol=atol, rtol=rtol)


# ============================================================================ #
# DTN (Drummond--Towler--Needs cutoff-poly u/chi/f) Jastrow.
#
# DTN is an additive ln|Psi| channel. The continuous measurement force/stress
# carries it through the verified DTN VMC kernels. The independent forward
# oracles use the same full wave-function kinetic composition, while their
# expression trees remain independent of the analytic derivative. Cases compose
# DTN with jnone / j12 / j123ao across molecular AE/ECP and periodic ECP.
# ============================================================================ #
def _make_dtn(
    structure_data,
    z_eff_per_species,
    pseudized_per_species,
    *,
    preset="u_chi_f",
    C=4,
    seed=13,
    L=3.0,
):
    """A random ACTIVE DTN cutoff-poly Jastrow (u = e-e, chi = e-N, f = e-e-N) on the structure.
    ``L=None`` lets the factory auto-resolve the cutoff from the WS-inscribed radius (needed for the
    periodic case, where the hard cutoff must pin a unique min-image)."""
    use_u = "u" in preset
    use_chi = "chi" in preset
    use_f = "f" in preset
    cutoff_kwargs = {}
    if structure_data.pbc_flag and L is not None:
        r_ins = structure_data._wigner_seitz_inscribed_radius()
        cutoff_kwargs = {
            "cutoff_ratio_u": L / r_ins,
            "cutoff_ratio_chi": L / r_ins,
            "cutoff_ratio_f": min(L, 1.6) / (0.5 * r_ins),
        }
    elif not structure_data.pbc_flag:
        cutoff_kwargs = {
            "L_u": L,
            "L_chi": L,
            "L_f": None if L is None else min(L, 1.6),
        }
    return Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=z_eff_per_species,
        pseudized_per_species=pseudized_per_species,
        use_u=use_u,
        use_chi=use_chi,
        use_f=use_f,
        cutoff_smoothness=C,
        n_order_u=4,
        n_order_chi=4,
        n_order_eN=2,
        n_order_ee=2,
        random_init=True,
        seed=seed,
        **cutoff_kwargs,
    )


def _build_ham_dtn(stem: str, kind: str, *, dtn_L: float = 3.0) -> Hamiltonian_data:
    """Hamiltonian (molecular AE/ECP or periodic ECP) with a DTN Jastrow COMPOSED with the base
    Jastrow: ``dtn`` = DTN only, ``j12_dtn`` = J1(exp)+J2(exp)+DTN, ``j123ao_dtn`` = +AO-leaf J3."""
    (structure_data, aos_data, mos_up, mos_dn, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
        os.path.join(_TREXIO, stem), store_tuple=True
    )
    atomic_numbers = np.asarray(structure_data.atomic_numbers, dtype=np.float64)
    z_cores = (
        np.asarray(coulomb_potential_data.z_cores, dtype=np.float64)
        if coulomb_potential_data.ecp_flag
        else np.zeros_like(atomic_numbers)
    )
    z_eff_per_species = tuple(
        float(np.mean((atomic_numbers - z_cores)[atomic_numbers == Z])) for Z in np.unique(atomic_numbers)
    )
    pseudized_per_species = tuple(
        bool(np.all(coulomb_potential_data.pseudized_per_atom[atomic_numbers == Z])) for Z in np.unique(atomic_numbers)
    )
    dtn = _make_dtn(structure_data, z_eff_per_species, pseudized_per_species, L=dtn_L)
    if kind == "dtn":
        j1 = j2 = j3 = None
    else:
        j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure_data
        )
        j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            # The REAL core count, not zero. This builder serves ECP stems too, and J1's
            # strength follows the effective charge Z - core: telling it every atom is
            # all-electron makes a pseudized chlorine behave like bare Z=17. Measured at
            # the test configuration of Cl2/ccECP that single mistake puts J1 alone at
            # -188.3, which drags the composed Jastrow to exp(J) ~ 1e-65 and
            # leaves the wavefunction unphysical. AE stems keep zeros, since z_cores is
            # zero there -- the value comes from the Hamiltonian rather than an
            # assumption about it.
            core_electrons=tuple(float(z) for z in z_cores),
            jastrow_1b_type="exp",
        )
        j3 = _make_j3_ao(aos_data) if kind == "j123ao_dtn" else None
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=j3, jastrow_dtn_data=dtn
    )
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    return Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)


_DTN_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap")
_DTN_KINDS = ["dtn", "j12_dtn", "j123ao_dtn"]


def _dtn_lane(kind):
    # DTN autograd-oracle rows run in the ordinary light tier.  The jitted analytic entry and
    # the vectorized dtn.py compile once per axis for one DTN e_L (an eager analytic path would
    # flood jax's compile cache with hundreds of micro-compiles per axis, the XLA-accumulation
    # SIGABRT class of shared xdist workers), so these small (<=7e) oracles are light-appropriate.
    del kind
    return pytest.mark.light


# molecular AE: N (5up/2dn, non-empty S'), Li (non-empty S'), H2 (empty S' -> continuous-only check).
_DTN_AE_CASES = [
    pytest.param(stem, alat, seed, scale, kind, marks=_dtn_lane(kind), id=f"{stem}-a{alat}-{kind}")
    for (stem, alat, seed, scale) in [
        ("N_ae_ccpvdz_cart.h5", 0.8, 0, 0.9),
        ("Li_ae_ccpvdz_cart.h5", 0.5, 0, 0.9),
        ("H2_ae_ccpvdz_cart.h5", 1.0, 0, 0.9),
    ]
    for kind in _DTN_KINDS
]


@pytest.mark.parametrize("stem, alat, seed, scale, kind", _DTN_AE_CASES)
def test_gfmc_t_force_analytic_vs_autograd_dtn(stem, alat, seed, scale, kind):
    """Analytic GFMC_t FORCE, molecular AE with a DTN Jastrow (composed with jnone / j12 / j123ao)
    vs ``jax.grad(_compute_local_energy_t)`` (the AE reproduction ``_e_L_t_ae_jnone``, whose FULL
    wavefunction kinetic carries the DTN channel)."""
    ham = _build_ham_dtn(stem, kind)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    # Retain propagation-membership liveness as an orthogonal adversarial-state
    # assertion; it is not consumed by the measurement derivative.
    w, _ = gfmc._sprime_data(ham, r_up, r_dn, RT, alat)
    if not stem.startswith("H2_"):
        assert float(np.sum(np.asarray(w))) > 0.0

    active0 = _extract_active_positions_from_ham(ham)

    def f_active(active, a, b):
        return _e_L_t_ae_jnone(_inject_active_positions(ham, active), a, b, RT, alat)

    gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)

    dR, dr_up, dr_dn = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat)

    atol, rtol = _gdd_get_tolerance_min(_DTN_ZONES, "strict")
    _gdd_assert_allclose(np.asarray(dR), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_up), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_dn), np.asarray(grd_oracle), atol=atol, rtol=rtol)
    jax.clear_caches()


@pytest.mark.parametrize("stem, alat, seed, scale, kind", _DTN_AE_CASES)
def test_gfmc_t_stress_analytic_vs_autograd_dtn(stem, alat, seed, scale, kind):
    """Analytic GFMC_t STRESS, molecular AE with a DTN Jastrow vs
    the independent position-AD affine virial of ``_e_L_t_ae_jnone``. This is
    the exact open-system strain chain rule, including the DTN kinetic channel."""
    ham = _build_ham_dtn(stem, kind)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    w, _ = gfmc._sprime_data(ham, r_up, r_dn, RT, alat)
    if not stem.startswith("H2_"):
        assert float(np.sum(np.asarray(w))) > 0.0

    active0 = _extract_active_positions_from_ham(ham)

    def f_active(active, a, b):
        return _e_L_t_ae_jnone(_inject_active_positions(ham, active), a, b, RT, alat)

    stress_oracle = _jit_molecular_affine_stress_oracle(f_active, active0, r_up, r_dn)

    stress = gfmc.grad_e_L_t_wrt_strain(ham, r_up, r_dn, RT, alat)

    atol, rtol = _gdd_get_tolerance_min(_DTN_ZONES, "strict")
    _gdd_assert_allclose(np.asarray(stress), np.asarray(stress_oracle), atol=atol, rtol=rtol)
    jax.clear_caches()


# molecular ECP: N2 (non-empty S'), Cl2 (2nd element).  H2-ecp = empty S' (continuous-only DTN).
_DTN_ECP_CASES = [
    pytest.param(stem, alat, seed, scale, kind, marks=_dtn_lane(kind), id=f"{stem}-a{alat}-{kind}")
    for (stem, alat, seed, scale) in [
        ("N2_ecp_ccpvtz_cart.h5", 1.0, 0, 0.9),
        ("Cl2_ecp_ccpvtz_cart.h5", 0.8, 0, 0.9),
    ]
    for kind in _DTN_KINDS
]


@pytest.mark.parametrize("stem, alat, seed, scale, kind", _DTN_ECP_CASES)
def test_gfmc_t_force_analytic_vs_autograd_dtn_ecp(stem, alat, seed, scale, kind):
    """Compare molecular ECP+DTN measurement force with autograd.

    The novar=0 derivative contains the continuous kinetic and ECP tmove
    measurement terms; propagation clipping is outside the estimator.
    """
    ham = _build_ham_dtn(stem, kind)
    RT = _rot(seed + 1)
    r_up, r_dn = _draw_well_conditioned_nonempty_sprime_cfg(ham, seed, scale, RT, alat)

    w, _ = gfmc._sprime_data(ham, r_up, r_dn, RT, alat)
    assert float(np.sum(np.asarray(w))) > 0.0  # both N2 and Cl2 have a non-empty S' at this config

    active0 = _extract_active_positions_from_ham(ham)

    # Conditioned reference (SSoT ``conditioned-direct-ad-cases``, issue #721) for the Cl2 dtn and
    # j123ao_dtn rows: the same energy with determinant-lemma mesh ratios off the fused-table base
    # inverse.  The full-recompute form subtracts two whole Jastrow values rounded in the fp32 mixed
    # ``jastrow_eval`` zone at every mesh point; there its own mixed AD force is 1.7e-2 / 5.0e-2 off
    # fp64, far more than the production force.  The two branches differ only in their selector.
    if stem == "Cl2_ecp_ccpvtz_cart.h5" and kind == "dtn":
        gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(
            lambda active, a, b: _e_L_t_ecp(
                _inject_active_positions(ham, active), a, b, RT, alat, rank_one_ratios=True, fused_base_inverse=True
            ),
            active0,
            r_up,
            r_dn,
        )
    elif stem == "Cl2_ecp_ccpvtz_cart.h5" and kind == "j123ao_dtn":
        gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(
            lambda active, a, b: _e_L_t_ecp(
                _inject_active_positions(ham, active), a, b, RT, alat, rank_one_ratios=True, fused_base_inverse=True
            ),
            active0,
            r_up,
            r_dn,
        )
    else:

        def f_active(active, a, b):
            return _e_L_t_ecp(_inject_active_positions(ham, active), a, b, RT, alat)

        gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)

    dR, dr_up, dr_dn = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat)

    # Fingerprint the INPUTS as well as the point. r_up/r_dn already came back
    # bitwise identical between two machines while the results differed by
    # 1.4%, so the next question is whether the Hamiltonian itself is identical:
    # the DTN parameters, the J3 coupling block, the geminal, the ECP tables and the
    # local energy at this very configuration.
    import hashlib as _hashlib

    def _fp(x):
        a = np.asarray(x, dtype=np.float64)
        return f"sha256={_hashlib.sha256(a.tobytes()).hexdigest()[:16]} shape={a.shape} max={np.max(np.abs(a)):.17e}"

    _wf = ham.wavefunction_data
    _jd = _wf.jastrow_data
    _diag_inputs = {
        "e_L_t_at_config": float(np.real(_e_L_t_ecp(ham, r_up, r_dn, RT, alat))),
        "fp_mo_up": _fp(_wf.geminal_data.orb_data_up_spin.mo_coefficients),
        "fp_positions": _fp(ham.structure_data.positions),
        "fp_j3_matrix": _fp(_jd.jastrow_three_body_data.j_matrix) if _jd.jastrow_three_body_data else None,
        "fp_dtn_alpha": _fp(_jd.jastrow_dtn_data.alpha_indep) if _jd.jastrow_dtn_data else None,
        "fp_dtn_beta": _fp(_jd.jastrow_dtn_data.beta_indep) if _jd.jastrow_dtn_data else None,
        "fp_dtn_gamma": _fp(_jd.jastrow_dtn_data.gamma_indep) if _jd.jastrow_dtn_data else None,
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": getattr(__import__("jaxlib"), "__version__", "n/a"),
    }
    global _DIAG_CONTEXT
    _DIAG_CONTEXT = {
        "stem": stem,
        "alat": alat,
        "seed": seed,
        "scale": scale,
        "kind": kind,
        "r_up": r_up,
        "r_dn": r_dn,
        "RT": RT,
        **_diag_inputs,
    }
    atol, rtol = _gdd_get_tolerance_min(_DTN_ZONES, "strict")
    _gdd_assert_allclose(np.asarray(dR), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_up), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_dn), np.asarray(grd_oracle), atol=atol, rtol=rtol)
    jax.clear_caches()


# Periodic real-Gamma ECP+DTN on lithium.h5. ``L=None`` selects the
# WS-inscribed cutoff so the DTN hard cutoff pins a unique minimum image.
_DTN_PBC_IDS = [f"Li_pbc_gamma-{kind}" for kind in _DTN_KINDS]


@pytest.mark.light
@pytest.mark.parametrize("kind", _DTN_KINDS, ids=_DTN_PBC_IDS)
def test_gfmc_t_force_stress_analytic_vs_autograd_dtn_pbc(kind):
    """Analytic GFMC_t FORCE + STRESS, periodic real-Gamma ECP with a DTN Jastrow (lithium.h5) vs
    ``jax.grad`` / ``jax.jacrev`` of the periodic-aware continuous ECP
    reproduction. DTN uses the minimum-image displacement throughout.
    """
    ham = _build_ham_dtn("lithium.h5", kind, dtn_L=None)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(0)
    r_up = jnp.asarray(rng.normal(scale=1.0, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=1.0, size=(nd, 3)))
    RT = _rot(2)
    alat = 0.5

    active0 = _extract_active_positions_from_ham(ham)

    def f_active(active, a, b):
        return _e_L_t_ecp(_inject_active_positions(ham, active), a, b, RT, alat)

    gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)
    dR, dr_up, dr_dn = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat)

    eps0 = jnp.zeros((3, 3), dtype=jnp.float64)
    stress_oracle = _jit_strain_oracle(lambda e: _e_L_t_ecp_strained(ham, r_up, r_dn, RT, alat, e), eps0)
    stress = gfmc.grad_e_L_t_wrt_strain(ham, r_up, r_dn, RT, alat)

    zones = _DTN_ZONES + ("ewald",)
    atol, rtol = _gdd_get_tolerance_min(zones, "strict")
    _gdd_assert_allclose(np.asarray(dR), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_up), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_dn), np.asarray(grd_oracle), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(stress), np.asarray(stress_oracle), atol=atol, rtol=rtol)
    jax.clear_caches()


@pytest.mark.light
def test_gfmc_n_force_stress_dtn_equals_t():
    """GFMC_n and GFMC_t FORCE + STRESS are bit-identical with a DTN Jastrow (the estimators differ
    only in the driver branching weight), and the GFMC_n public API accepts DTN (no NotImplementedError)."""
    ham = _build_ham_dtn("Li_ae_ccpvdz_cart.h5", "j123ao_dtn")
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(0)
    r_up = jnp.asarray(rng.normal(scale=0.9, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=0.9, size=(nd, 3)))
    RT = _rot(1)
    alat = 0.5

    dR_n, dru_n, drd_n = gfmc.grad_e_L_n_wrt_positions(ham, r_up, r_dn, RT, alat)
    dR_t, dru_t, drd_t = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat)
    _gdd_assert_array_equal(np.asarray(dR_n), np.asarray(dR_t))
    _gdd_assert_array_equal(np.asarray(dru_n), np.asarray(dru_t))
    _gdd_assert_array_equal(np.asarray(drd_n), np.asarray(drd_t))

    s_n = gfmc.grad_e_L_n_wrt_strain(ham, r_up, r_dn, RT, alat)
    s_t = gfmc.grad_e_L_t_wrt_strain(ham, r_up, r_dn, RT, alat)
    _gdd_assert_array_equal(np.asarray(s_n), np.asarray(s_t))
