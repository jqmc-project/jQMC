"""Orthogonal liveness/regime/validity evidence for e_L_t/e_L_n.

The broad GFMC file supplies the fixture matrix.  These tests assert the
value-dependent preconditions that a plain analytical==AD comparison cannot
prove, and are registered independently by ``analytic_coverage_registration``.
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

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gdd.engine.evidence import (
    ComparisonBinding,
    ElectronConfigurationBinding,
    ObservationBinding,
    observe_allclose,
    observe_electron_configuration_sequence,
    observe_exact,
    observe_non_numerical,
)
from jqmc._precision import get_tolerance_min
from jqmc._setting import ATOL_PYTHON_FP64_BZ_MEAN, RTOL_PYTHON_FP64_ZERO
from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors
from jqmc.derivatives import assemble, ecp_force, gfmc
from jqmc.jastrow_factor import Jastrow_data
from jqmc.jqmc_gfmc import _sprime_data_from_projection_state
from jqmc.jqmc_mcmc import _extract_active_positions_from_ham, _inject_active_positions
from jqmc.jqmc_mcmc import _geminal_inv_single
from jqmc.wavefunction import (
    _init_discretized_kinetic_mesh_streaming_state,
    _init_kinetic_energy_all_elements_streaming_state,
)
from tests.test_jqmc_kernel import test_gfmc_force_analytic as _broad
from tests.test_jqmc_kernel._gdd_adapters import (
    case_receipt,
    committed_input,
    current_precision_mode,
    guarantee_id_for_manifest_family,
    seeded_single_electron_receipt,
)
from tests.test_jqmc_kernel.test_analytic_coverage_evidence import (
    _geminal_singular_values,
    _moderately_conditioned_configuration,
)

pytestmark = [
    pytest.mark.heavy,
    pytest.mark.without_chdir,
    pytest.mark.design_pin,
]

_FAMILIES = ("e_L_t", "e_L_n")
_REGIME_REQUIREMENTS = tuple(
    f"{family}|regime|{regime}"
    for regime in ("R01", "R02", "R04", "R05", "R06", "R07", "R08", "R09", "R10")
    for family in _FAMILIES
)


def output_obligation_id(family: str) -> str:
    """The B24 output-liveness obligation id; shared with the derivative registry."""
    return f"{family}|output|B24"


def invalid_move_obligation_id(family: str) -> str:
    """The C10 unknown non-local move obligation id; shared with the derivative registry."""
    return f"{family}|error|C10|unknown-non-local-move"


_ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "det_eval",
    "wf_kinetic",
    "coulomb",
    "ewald",
    "jastrow_eval",
    "jastrow_grad_lap",
)


@pytest.fixture(autouse=True)
def _clear_jax_caches(request):
    """Reuse the shared t/n compilation within a regime pair, then drain it."""
    yield
    parameters = getattr(getattr(request.node, "callspec", None), "params", {})
    family = parameters.get("family")
    requirement_id = parameters.get("requirement_id", "")
    if family == "e_L_n" or requirement_id.startswith("e_L_n|") or not parameters:
        jax.clear_caches()


def _coordinates(ham, *, seed=0, scale=0.9):
    gd = ham.wavefunction_data.geminal_data
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(int(gd.num_electron_up), 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(int(gd.num_electron_dn), 3)))
    return r_up, r_dn


def _position_function(family):
    return gfmc.grad_e_L_t_wrt_positions if family == "e_L_t" else gfmc.grad_e_L_n_wrt_positions


def _strain_function(family):
    return gfmc.grad_e_L_t_wrt_strain if family == "e_L_t" else gfmc.grad_e_L_n_wrt_strain


def _forward(ham, r_up, r_dn, RT, alat, non_local_move):
    if getattr(ham.coulomb_potential_data, "ecp_flag", False):
        return _broad._e_L_t_ecp(ham, r_up, r_dn, RT, alat, non_local_move=non_local_move)
    return _broad._e_L_t_ae_jnone(ham, r_up, r_dn, RT, alat)


def _assert_position_ad(
    family,
    ham,
    r_up,
    r_dn,
    RT,
    alat,
    *,
    non_local_move="tmove",
    compare_nuclear_with_ad=True,
    return_receipt_pairs=False,
):
    active0 = _extract_active_positions_from_ham(ham)

    def f_active(active, a, b):
        return _forward(_inject_active_positions(ham, active), a, b, RT, alat, non_local_move)

    expected = _broad._jit_grad_pos(f_active, active0, r_up, r_dn)
    actual = _position_function(family)(ham, r_up, r_dn, RT, alat, non_local_move=non_local_move)
    level = "medium" if current_precision_mode() == "mixed" else "strict"
    atol, rtol = get_tolerance_min(_ZONES, level)
    pairs = []
    start = 0 if compare_nuclear_with_ad else 1
    labels = ("dR", "dr_up", "dr_dn")
    for label, left, right in zip(labels[start:], actual[start:], expected[start:], strict=True):
        array = np.asarray(left)
        assert array.shape == np.asarray(right).shape, label
        assert np.all(np.isfinite(array)), label
        np.testing.assert_allclose(array, np.asarray(right), atol=atol, rtol=rtol, err_msg=label)
        pairs.append((array, np.asarray(right)))
    return (actual, tuple(pairs)) if return_receipt_pairs else actual


def _assert_n_t_identity(ham, r_up, r_dn, RT, alat, *, non_local_move="tmove", return_receipt_pairs=False):
    n_outputs = gfmc.grad_e_L_n_wrt_positions(ham, r_up, r_dn, RT, alat, non_local_move=non_local_move)
    t_outputs = gfmc.grad_e_L_t_wrt_positions(ham, r_up, r_dn, RT, alat, non_local_move=non_local_move)
    pairs = []
    for left, right in zip(n_outputs, t_outputs, strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        pairs.append((np.asarray(left), np.asarray(right)))
    return (n_outputs, tuple(pairs)) if return_receipt_pairs else n_outputs


def _gfmc_jastrow_channel_hamiltonians():
    """Li: non-empty S' with isolated J1/J2/J3/DTN and full composition."""
    base = _broad._build_ham("Li_ae_ccpvdz_cart.h5")
    legacy = _broad._build_ham_jastrow("Li_ae_ccpvdz_cart.h5", "j123ao").wavefunction_data.jastrow_data
    dtn = _broad._build_ham_dtn("Li_ae_ccpvdz_cart.h5", "dtn").wavefunction_data.jastrow_data

    def with_channels(*, j1=False, j2=False, j3=False, dtn_enabled=False):
        jastrow = Jastrow_data(
            jastrow_one_body_data=legacy.jastrow_one_body_data if j1 else None,
            jastrow_two_body_data=legacy.jastrow_two_body_data if j2 else None,
            jastrow_three_body_data=legacy.jastrow_three_body_data if j3 else None,
            jastrow_dtn_data=dtn.jastrow_dtn_data if dtn_enabled else None,
        )
        return base.replace(wavefunction_data=base.wavefunction_data.replace(jastrow_data=jastrow))

    return base, {
        "J1": with_channels(j1=True),
        "J2": with_channels(j2=True),
        "J3": with_channels(j3=True),
        "DTN": with_channels(dtn_enabled=True),
        "all": with_channels(j1=True, j2=True, j3=True, dtn_enabled=True),
    }


def _continuous_position_outputs(ham, r_up, r_dn, RT):
    dR = assemble.compute_grad_e_L(ham, r_up, r_dn, RT, wrt="R")
    dr_up, dr_dn = assemble.compute_grad_e_L(ham, r_up, r_dn, RT, wrt="r")
    return dR, dr_up, dr_dn


def _subtract_outputs(left, right):
    return tuple(np.asarray(a) - np.asarray(b) for a, b in zip(left, right, strict=True))


def _output_difference_norm(left, right):
    return sum(float(np.linalg.norm(np.asarray(a) - np.asarray(b))) for a, b in zip(left, right, strict=True))


@pytest.mark.parametrize("family", _FAMILIES)
def test_gfmc_output_liveness(family):
    """Every public e_L_t/e_L_n output leaf is finite, shaped and non-vacuous."""
    ham = _broad._build_ham_jastrow("H2_ae_ccpvdz_cart.h5", "j12")
    r_up, r_dn = _coordinates(ham)
    RT = _broad._rot(1)
    positions = _position_function(family)(ham, r_up, r_dn, RT, 1.0)
    stress = _strain_function(family)(ham, r_up, r_dn, RT, 1.0)
    outputs = (*positions, stress)
    for label, value in zip(("dR", "dr_up", "dr_dn"), positions, strict=True):
        assert float(np.linalg.norm(np.asarray(value))) > 1.0e-10, label
    assert float(np.linalg.norm(np.asarray(stress))) > 1.0e-10
    obligation_id = output_obligation_id(family)
    observation = observe_non_numerical(
        ObservationBinding(
            binding_id=f"LIVENESS::{obligation_id}",
            policy_id="OBS-ROUTE-LIVENESS-001",
            precision_mode=current_precision_mode(),
            subject_role="gfmc-public-derivative-outputs",
            required_observations=("route-executed", "expected-output-leaves-accounted", "finite", "nonzero"),
        ),
        {
            "route-executed": bool(outputs),
            "expected-output-leaves-accounted": len(outputs) == 4,
            "finite": all(np.all(np.isfinite(np.asarray(value))) for value in outputs),
            "nonzero": all(float(np.linalg.norm(np.asarray(value))) > 1.0e-10 for value in outputs),
        },
    )
    electrons = seeded_single_electron_receipt(
        obligation_id,
        r_up,
        r_dn,
        generator_operation="tests.test_jqmc_kernel.test_gfmc_analytic_coverage_evidence._coordinates",
        seed=0,
        generator_arguments=(("distribution", "normal"), ("scale", 0.9)),
        generation_pattern="direct",
    )
    case_receipt(
        guarantee_id=guarantee_id_for_manifest_family(family),
        obligation_id=obligation_id,
        pytest_node_id=(f"tests/test_jqmc_kernel/test_gfmc_analytic_coverage_evidence.py::test_gfmc_output_liveness[{family}]"),
        input_binding=committed_input(
            obligation_id,
            source_ids=("WF-SRC-H2-AE-CART-MOL",),
            builder_operation="tests.test_jqmc_kernel.test_gfmc_force_analytic._build_ham_jastrow",
            transformations=("derivatives.targeted.attach-j1-j2.v1",),
        ),
        electron_configuration=electrons,
        primary_check=observation,
    )


@pytest.mark.parametrize("family", _FAMILIES)
def test_gfmc_invalid_non_local_move(family):
    """C10 rejects invalid/ECP-missing moves while preserving AE ``None``."""
    ecp_ham = _broad._build_ham_ecp("H2_ecp_ccpvdz_cart.h5", "jnone")
    ecp_r_up, ecp_r_dn = _coordinates(ecp_ham)
    ae_ham = _broad._build_ham_jastrow("H2_ae_ccpvdz_cart.h5", "j12")
    ae_r_up, ae_r_dn = _coordinates(ae_ham)
    np.testing.assert_array_equal(np.asarray(ecp_r_up), np.asarray(ae_r_up))
    np.testing.assert_array_equal(np.asarray(ecp_r_dn), np.asarray(ae_r_dn))
    observed_errors = []
    exact_pairs = []
    for function in (_position_function(family), _strain_function(family)):
        with pytest.raises(ValueError, match="unsupported non_local_move") as typo_error:
            function(ecp_ham, ecp_r_up, ecp_r_dn, jnp.eye(3), 1.0, non_local_move="typo")
        observed_errors.append(typo_error.value)
        with pytest.raises(ValueError, match="unsupported non_local_move=None") as none_error:
            function(ecp_ham, ecp_r_up, ecp_r_dn, jnp.eye(3), 1.0, non_local_move=None)
        observed_errors.append(none_error.value)

        ae_none = function(ae_ham, ae_r_up, ae_r_dn, jnp.eye(3), 1.0, non_local_move=None)
        ae_tmove = function(ae_ham, ae_r_up, ae_r_dn, jnp.eye(3), 1.0, non_local_move="tmove")
        for none_value, tmove_value in zip(
            jax.tree_util.tree_leaves(ae_none), jax.tree_util.tree_leaves(ae_tmove), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(none_value), np.asarray(tmove_value))
            exact_pairs.append((none_value, tmove_value))

    obligation_id = invalid_move_obligation_id(family)
    error_observation = observe_non_numerical(
        ObservationBinding(
            binding_id=f"ERROR::{obligation_id}",
            policy_id="OBS-EXPECTED-ERROR-001",
            precision_mode=current_precision_mode(),
            subject_role="gfmc-invalid-non-local-move",
            required_observations=("expected-error-observed", "exception-type-matched", "message-matched"),
        ),
        {
            "expected-error-observed": len(observed_errors) == 4,
            "exception-type-matched": all(isinstance(error, ValueError) for error in observed_errors),
            "message-matched": all("unsupported non_local_move" in str(error) for error in observed_errors),
        },
    )
    ae_identity = observe_exact(
        ComparisonBinding(
            binding_id=f"EXACT-AE-NONE::{obligation_id}",
            policy_id="CMP-EXACT-VALUES-001",
            precision_mode=current_precision_mode(),
            actual_role="ae-none-non-local-move",
            reference_role="ae-tmove-non-local-move",
        ),
        exact_pairs,
    )
    electrons = seeded_single_electron_receipt(
        obligation_id,
        ecp_r_up,
        ecp_r_dn,
        generator_operation="tests.test_jqmc_kernel.test_gfmc_analytic_coverage_evidence._coordinates",
        seed=0,
        generator_arguments=(("distribution", "normal"), ("scale", 0.9)),
        generation_pattern="direct",
    )
    case_receipt(
        guarantee_id=guarantee_id_for_manifest_family(family),
        obligation_id=obligation_id,
        pytest_node_id=(
            f"tests/test_jqmc_kernel/test_gfmc_analytic_coverage_evidence.py::test_gfmc_invalid_non_local_move[{family}]"
        ),
        input_binding=committed_input(
            obligation_id,
            source_ids=("WF-SRC-H2-ECP-CART-MOL", "WF-SRC-H2-AE-CART-MOL"),
            builder_operation="tests.test_jqmc_kernel.test_gfmc_analytic_coverage_evidence.test_gfmc_invalid_non_local_move",
            transformations=("derivatives.targeted.attach-j1-j2-to-ae-control.v1",),
        ),
        electron_configuration=electrons,
        primary_check=error_observation,
        additional_checks=(ae_identity,),
    )


def _record_gfmc_regime_receipt(
    requirement_id,
    *,
    source_ids,
    configurations,
    compared_pairs,
    proof,
    generation_pattern,
    seed=0,
    generation_attempts=None,
    transformations=(),
    conditioning=None,
    identity_pairs=(),
):
    family, _category, regime = requirement_id.split("|")
    if proof == "direct-ad":
        level = "medium" if current_precision_mode() == "mixed" else "strict"
        reference_role = "xla-gfmc-autodiff-regime-result-same-precision"
        atol, rtol = get_tolerance_min(_ZONES, level)
        comparison = observe_allclose(
            ComparisonBinding(
                binding_id=f"COMPARISON::{requirement_id}",
                policy_id=f"CMP-ZONE-ALLCLOSE-{level.upper()}-001",
                precision_mode=current_precision_mode(),
                actual_role="xla-gfmc-analytical-regime-result",
                reference_role=reference_role,
                zones=_ZONES,
                level=level,
            ),
            compared_pairs,
            atol=atol,
            rtol=rtol,
        )
    else:
        comparison = observe_exact(
            ComparisonBinding(
                binding_id=f"COMPARISON::{requirement_id}",
                policy_id="CMP-EXACT-VALUES-001",
                precision_mode=current_precision_mode(),
                actual_role="xla-gfmc-n-regime-result",
                reference_role="xla-gfmc-t-regime-result",
            ),
            compared_pairs,
        )
    conditioning = {} if conditioning is None else conditioning
    electron_binding = ElectronConfigurationBinding(
        binding_id=f"ELECTRONS::{requirement_id}",
        mode="required",
        generation_pattern=generation_pattern,
        position_source_kind="seeded-random",
        generator_operation="tests.test_jqmc_kernel.test_gfmc_analytic_coverage_evidence.test_gfmc_numerical_regime_evidence",
        generator_arguments=conditioning.get("generator_arguments", (("regime", regime),)),
        random_algorithm="numpy.random.PCG64",
        seed=seed,
        configuration_count=len(configurations),
        geminal_conditioning_mode=conditioning.get("mode", "not-limited"),
        minimum_cond_g=conditioning.get("minimum_cond_g"),
        maximum_cond_g=conditioning.get("maximum_cond_g"),
        lower_bound_comparison=conditioning.get("lower_bound_comparison"),
        upper_bound_comparison=conditioning.get("upper_bound_comparison"),
        minimum_singular_value=conditioning.get("minimum_singular_value"),
        conditioning_violation_action=conditioning.get("violation_action"),
        maximum_generation_attempts=conditioning.get("maximum_generation_attempts"),
    )
    attempts = len(configurations) if generation_attempts is None else generation_attempts
    electrons = observe_electron_configuration_sequence(
        electron_binding,
        configurations,
        cond_g_per_configuration=conditioning.get("cond_g", ()),
        sigma_min_g_per_configuration=conditioning.get("sigma_min_g", ()),
        sigma_max_g_per_configuration=conditioning.get("sigma_max_g", ()),
        machine_epsilon_per_configuration=conditioning.get("machine_epsilon", ()),
        cond_times_epsilon_per_configuration=conditioning.get("cond_times_epsilon", ()),
        generation_attempts=attempts,
        rejected_configuration_count=max(0, attempts - len(configurations)),
    )
    precondition = observe_non_numerical(
        ObservationBinding(
            binding_id=f"REGIME::{requirement_id}",
            policy_id="OBS-NUMERICAL-REGIME-001",
            precision_mode=current_precision_mode(),
            subject_role=f"gfmc-numerical-regime-{regime}",
            required_observations=("numerical-precondition-observed", "route-executed"),
        ),
        {"numerical-precondition-observed": True, "route-executed": bool(compared_pairs)},
    )
    additional_checks = [precondition]
    if identity_pairs:
        level = "medium" if current_precision_mode() == "mixed" else "strict"
        atol, rtol = get_tolerance_min(_ZONES, level)
        additional_checks.append(
            observe_allclose(
                ComparisonBinding(
                    binding_id=f"TRANSLATION-IDENTITY::{requirement_id}",
                    policy_id=f"CMP-ZONE-ALLCLOSE-{level.upper()}-001",
                    precision_mode=current_precision_mode(),
                    actual_role="xla-gfmc-analytical-nuclear-derivative",
                    reference_role="negative-sum-of-xla-gfmc-analytical-electron-derivatives",
                    zones=_ZONES,
                    level=level,
                ),
                identity_pairs,
                atol=atol,
                rtol=rtol,
            )
        )
    return case_receipt(
        guarantee_id=guarantee_id_for_manifest_family(family),
        obligation_id=requirement_id,
        pytest_node_id=(
            "tests/test_jqmc_kernel/test_gfmc_analytic_coverage_evidence.py::"
            f"test_gfmc_numerical_regime_evidence[{requirement_id}]"
        ),
        input_binding=committed_input(
            requirement_id,
            source_ids=source_ids,
            builder_operation=(
                "tests.test_jqmc_kernel.test_gfmc_analytic_coverage_evidence.test_gfmc_numerical_regime_evidence"
            ),
            transformations=transformations,
        ),
        electron_configuration=electrons,
        primary_check=comparison,
        additional_checks=tuple(additional_checks),
    )


@pytest.mark.parametrize("requirement_id", _REGIME_REQUIREMENTS)
def test_gfmc_numerical_regime_evidence(requirement_id):
    """Assert each R01-R10 precondition paired with the broad GFMC parity rows.

    R02 adds a moderately conditioned, active-precision-resolved point: every
    electron gradient is compared with AD and the one-nucleus gradient is
    checked by exact translation invariance. R08 adds the missing identity-RT
    AD point; the other regimes supply liveness/precondition evidence orthogonal
    to direct AD rows already registered from ``test_gfmc_force_analytic``.
    """
    family, category, regime = requirement_id.split("|")
    assert category == "regime"

    if regime == "R01":
        ham = _broad._build_ham_jastrow("H2_ae_ccpvdz_cart.h5", "j12")
        r_up, r_dn = _coordinates(ham)
        singular_values = _geminal_singular_values(ham, r_up, r_dn)
        condition_number = float(singular_values[0] / singular_values[-1])
        assert condition_number < 1.0e3
        _outputs, pairs = _assert_position_ad(
            family,
            ham,
            r_up,
            r_dn,
            _broad._rot(1),
            0.8,
            return_receipt_pairs=True,
        )
        _record_gfmc_regime_receipt(
            requirement_id,
            source_ids=("WF-SRC-H2-AE-CART-MOL",),
            configurations=((r_up, r_dn),),
            compared_pairs=pairs,
            proof="direct-ad",
            generation_pattern="direct",
            conditioning={
                "mode": "upper-bounded",
                "maximum_cond_g": 1.0e3,
                "upper_bound_comparison": "less-than",
                "violation_action": "fail-case",
                "cond_g": (condition_number,),
            },
        )
        return

    if regime == "R02":
        ham, target = _moderately_conditioned_configuration()
        r_up, r_dn = target.r_up, target.r_dn
        minimum_cond_g, maximum_cond_g, minimum_singular_value = (1.0e3, 1.0e6, 1.0e-8)
        assert minimum_cond_g < target.cond_g < maximum_cond_g
        assert target.sigma_min_g > minimum_singular_value
        # e_L_n is verified against autodiff like e_L_t,
        # not by asserting that grad_e_L_n equals grad_e_L_t BITWISE. The n/t
        # bitwise identity is a claim about two implementations agreeing to the
        # last bit, which is stronger than the physics requires and which the
        # production routes do not satisfy in these regimes.
        outputs, pairs = _assert_position_ad(
            family,
            ham,
            r_up,
            r_dn,
            _broad._rot(1),
            0.8,
            compare_nuclear_with_ad=False,
            return_receipt_pairs=True,
        )
        # The one-nucleus molecular witness is exactly invariant under a
        # uniform translation.  Thus dE/dR = -sum_e dE/dr_e.  Record that
        # independent identity for dR; the electron gradients in ``pairs``
        # remain direct same-precision AD checks.
        translated_dR = -(jnp.sum(outputs[1], axis=0) + jnp.sum(outputs[2], axis=0))[None, :]
        identity_pairs = ((outputs[0], translated_dR),)
        proof = "direct-ad"
        _record_gfmc_regime_receipt(
            requirement_id,
            source_ids=("WF-SRC-N-AE-CART-MOL",),
            configurations=((r_up, r_dn),),
            compared_pairs=pairs,
            proof=proof,
            generation_pattern="condition-target",
            transformations=("derivatives.regime.make-moderately-conditioned-g.v1",),
            conditioning={
                "mode": "target-range",
                "minimum_cond_g": minimum_cond_g,
                "maximum_cond_g": maximum_cond_g,
                "lower_bound_comparison": "greater-than",
                "upper_bound_comparison": "less-than",
                "minimum_singular_value": minimum_singular_value,
                "violation_action": "fail-case",
                "maximum_generation_attempts": 11,
                "generator_arguments": (
                    ("control_parameter", "same-spin-separation-delta"),
                    ("search_method", "declared-log-grid"),
                    ("regime", regime),
                ),
                "cond_g": (target.cond_g,),
                "sigma_min_g": (target.sigma_min_g,),
                "sigma_max_g": (target.sigma_max_g,),
                "machine_epsilon": (target.machine_epsilon,),
                "cond_times_epsilon": (target.cond_times_epsilon,),
            },
            generation_attempts=target.generation_attempts,
            identity_pairs=identity_pairs,
        )
        return

    if regime == "R04":
        ham = _broad._build_ham_twist("j123ao")
        r_up, r_dn, RT, alat, nonempty, metadata = _broad._twist_config(ham, return_metadata=True)
        assert nonempty
        ratio = np.asarray(ecp_force._ecp_nl_common(ham, r_up, r_dn, RT, None)["ratio"])
        assert np.iscomplexobj(ratio)
        assert float(np.max(np.abs(np.real(ratio)))) > 1.0e-8
        assert float(np.max(np.abs(np.imag(ratio)))) > 1.0e-8
        _outputs, pairs = _assert_position_ad(family, ham, r_up, r_dn, RT, alat, return_receipt_pairs=True)
        _record_gfmc_regime_receipt(
            requirement_id,
            source_ids=("WF-SRC-LI-ECP-SPHE-NON-GAMMA",),
            configurations=((r_up, r_dn),),
            compared_pairs=pairs,
            proof="direct-ad",
            generation_pattern="direct",
            seed=metadata["seed"],
            generation_attempts=metadata["attempts"],
        )
        return

    if regime == "R05":
        # Keep ``ndk = sum(t_s + 1/(2a^2)) = -1`` at a=1, so zv=1
        # lies strictly between the synthetic eid values 0 and 2, so the rows
        # span all four predicate truth pairs.
        no_flip_row = np.full(6, -4.0 / 6.0, dtype=np.float64)
        flip_row = np.asarray([0.0, -0.8, -0.8, -0.8, -0.8, -0.8])
        kinetic_mesh = jnp.asarray(np.stack([no_flip_row, no_flip_row, flip_row, flip_row]))
        zero_pair = (jnp.zeros(2), jnp.zeros(2))
        discretized_coulomb = (jnp.asarray([0.0, 2.0]), jnp.asarray([0.0, 2.0]))
        synthetic_ham = SimpleNamespace(
            wavefunction_data=object(),
            coulomb_potential_data=SimpleNamespace(ecp_flag=False),
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(gfmc, "compute_kinetic_energy_all_elements", lambda *args: zero_pair)
            monkeypatch.setattr(
                gfmc,
                "compute_discretized_kinetic_energy",
                lambda *args: (None, None, kinetic_mesh),
            )
            monkeypatch.setattr(gfmc, "compute_coulomb_potential_ei_dispatch", lambda *args: zero_pair)
            monkeypatch.setattr(
                gfmc,
                "compute_coulomb_potential_ei_discretized_dispatch",
                lambda *args: discretized_coulomb,
            )
            membership, returned_mesh = gfmc._sprime_data(
                synthetic_ham,
                jnp.zeros((2, 3)),
                jnp.zeros((2, 3)),
                jnp.eye(3),
                jnp.asarray(1.0),
            )
        np.testing.assert_array_equal(np.asarray(membership), np.asarray([0.0, 0.0, 0.0, 1.0]))
        np.testing.assert_array_equal(np.asarray(returned_mesh), np.asarray(kinetic_mesh))
        nonempty_ham = _broad._build_ham("N_ae_ccpvdz_cart.h5")
        empty_ham = _broad._build_ham("H2_ae_ccpvdz_cart.h5")
        configurations = []
        for ham, expected_nonempty in ((nonempty_ham, True), (empty_ham, False)):
            r_up, r_dn = _coordinates(ham)
            configurations.append((r_up, r_dn))
            RT = _broad._rot(1)
            alat = 0.8
            w, elements = gfmc._sprime_data(ham, r_up, r_dn, RT, alat)
            assert bool(np.any(np.asarray(w))) is expected_nonempty
            _, A_inv, _, _ = _geminal_inv_single(ham.wavefunction_data.geminal_data, None, r_up, r_dn)
            kinetic_state = _init_kinetic_energy_all_elements_streaming_state(
                ham.wavefunction_data,
                r_up,
                r_dn,
                A_inv,
            )
            mesh_state = _init_discretized_kinetic_mesh_streaming_state(
                alat,
                ham.wavefunction_data,
                r_up,
                r_dn,
                RT,
            )
            state_w, state_elements = _sprime_data_from_projection_state(
                ham,
                r_up,
                r_dn,
                A_inv,
                kinetic_state,
                mesh_state,
                alat,
            )
            np.testing.assert_array_equal(np.asarray(state_w), np.asarray(w))
            atol, rtol = get_tolerance_min(_ZONES, "strict")
            np.testing.assert_allclose(np.asarray(state_elements), np.asarray(elements), atol=atol, rtol=rtol)
        r_up, r_dn = configurations[0]
        _outputs, pairs = _assert_position_ad(family, nonempty_ham, r_up, r_dn, _broad._rot(1), 0.8, return_receipt_pairs=True)
        _record_gfmc_regime_receipt(
            requirement_id,
            source_ids=("WF-SRC-N-AE-CART-MOL", "WF-SRC-H2-AE-CART-MOL"),
            configurations=tuple(configurations),
            compared_pairs=pairs,
            proof="direct-ad",
            generation_pattern="direct",
        )
        return

    if regime == "R06":
        ham = _broad._build_ham_ecp("N2_ecp_ccpvtz_cart.h5", "j123ao")
        r_up, r_dn = _coordinates(ham)
        RT = _broad._rot(1)
        _, _, values, _ = compute_ecp_non_local_parts_nearest_neighbors(
            ham.coulomb_potential_data,
            ham.wavefunction_data,
            r_up,
            r_dn,
            flag_determinant_only=True,
            RT=RT,
        )
        values = np.asarray(values)
        assert float(np.min(values)) < -1.0e-8
        assert float(np.max(values)) > 1.0e-8
        _outputs, pairs = _assert_position_ad(family, ham, r_up, r_dn, RT, 0.8, return_receipt_pairs=True)
        _record_gfmc_regime_receipt(
            requirement_id,
            source_ids=("WF-SRC-N2-ECP-CART-MOL",),
            configurations=((r_up, r_dn),),
            compared_pairs=pairs,
            proof="direct-ad",
            generation_pattern="direct",
        )
        return

    if regime == "R07":
        ham = _broad._build_ham("N_ae_ccpvdz_cart.h5")
        r_up, r_dn = _coordinates(ham)
        RT = _broad._rot(1)
        outputs = [_position_function(family)(ham, r_up, r_dn, RT, alat) for alat in (0.5, 0.8)]
        # novar=0 force/stress differentiate the continuum measurement energy;
        # lattice spacing belongs only to propagation and must drop out.
        atol, rtol = get_tolerance_min(_ZONES, "strict")
        for left, right in zip(outputs[0], outputs[1], strict=True):
            np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=atol, rtol=rtol)
        _checked, pairs = _assert_position_ad(family, ham, r_up, r_dn, RT, 0.8, return_receipt_pairs=True)
        _record_gfmc_regime_receipt(
            requirement_id,
            source_ids=("WF-SRC-N-AE-CART-MOL",),
            configurations=((r_up, r_dn),),
            compared_pairs=pairs,
            proof="direct-ad",
            generation_pattern="direct",
        )
        return

    if regime == "R08":
        ham = _broad._build_ham("N_ae_ccpvdz_cart.h5")
        r_up, r_dn = _coordinates(ham)
        rotations = (jnp.eye(3), _broad._rot(1))
        np.testing.assert_allclose(
            np.asarray(rotations[1].T @ rotations[1]),
            np.eye(3),
            atol=ATOL_PYTHON_FP64_BZ_MEAN,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        assert not np.array_equal(np.asarray(rotations[0]), np.asarray(rotations[1]))
        checked, pairs = _assert_position_ad(family, ham, r_up, r_dn, rotations[0], 0.8, return_receipt_pairs=True)
        outputs = [checked, _position_function(family)(ham, r_up, r_dn, rotations[1], 0.8)]
        # This committed fixture is AE. Its continuum measurement estimator
        # contains no mesh, so its derivatives are rotation-invariant. ECP
        # measurement derivatives legitimately depend on RT through the
        # non-local quadrature and are deliberately outside this assertion.
        atol, rtol = get_tolerance_min(_ZONES, "strict")
        for left, right in zip(outputs[0], outputs[1], strict=True):
            np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=atol, rtol=rtol)
        _record_gfmc_regime_receipt(
            requirement_id,
            source_ids=("WF-SRC-N-AE-CART-MOL",),
            configurations=((r_up, r_dn),),
            compared_pairs=pairs,
            proof="direct-ad",
            generation_pattern="direct",
        )
        return

    if regime == "R09":
        configurations = []
        pairs = []
        for stem, tied in (("H2_ae_ccpvdz_cart.h5", True), ("N_ae_ccpvdz_cart.h5", False)):
            ham = _broad._build_ham(stem)
            assert bool(ham.wavefunction_data.geminal_data.time_reversal) is tied
            r_up, r_dn = _coordinates(ham)
            configurations.append((r_up, r_dn))
            assert r_dn.shape[0] > 0
            outputs, checked_pairs = _assert_position_ad(
                family, ham, r_up, r_dn, _broad._rot(1), 0.8, return_receipt_pairs=True
            )
            pairs.extend(checked_pairs)
            assert float(np.linalg.norm(np.asarray(outputs[2]))) > 1.0e-10
        _record_gfmc_regime_receipt(
            requirement_id,
            source_ids=("WF-SRC-H2-AE-CART-MOL", "WF-SRC-N-AE-CART-MOL"),
            configurations=tuple(configurations),
            compared_pairs=tuple(pairs),
            proof="direct-ad",
            generation_pattern="direct",
        )
        return

    if regime == "R10":
        base, channels = _gfmc_jastrow_channel_hamiltonians()
        r_up, r_dn = _coordinates(base)
        RT = _broad._rot(1)
        base_w, _ = gfmc._sprime_data(base, r_up, r_dn, RT, 0.5)
        assert bool(np.any(np.asarray(base_w)))
        base_output, pairs = _assert_position_ad(family, base, r_up, r_dn, RT, 0.5, return_receipt_pairs=True)
        proof = "direct-ad"
        pairs = list(pairs)
        atol, rtol = get_tolerance_min(_ZONES, "strict")
        for actual, continuous in zip(base_output, _continuous_position_outputs(base, r_up, r_dn, RT), strict=True):
            np.testing.assert_allclose(np.asarray(actual), np.asarray(continuous), atol=atol, rtol=rtol)
        for channel, ham in channels.items():
            w, _ = gfmc._sprime_data(ham, r_up, r_dn, RT, 0.5)
            assert bool(np.any(np.asarray(w))), channel
            output, checked_pairs = _assert_position_ad(family, ham, r_up, r_dn, RT, 0.5, return_receipt_pairs=True)
            pairs.extend(checked_pairs)
            for actual, continuous in zip(output, _continuous_position_outputs(ham, r_up, r_dn, RT), strict=True):
                np.testing.assert_allclose(np.asarray(actual), np.asarray(continuous), atol=atol, rtol=rtol, err_msg=channel)
        _record_gfmc_regime_receipt(
            requirement_id,
            source_ids=("WF-SRC-LI-AE-CART-MOL",),
            configurations=((r_up, r_dn),),
            compared_pairs=tuple(pairs),
            proof=proof,
            generation_pattern="direct",
            transformations=("derivatives.regime.compose-gfmc-jastrow-channels.v1",),
        )
        return

    raise AssertionError(f"unknown GFMC regime requirement: {requirement_id}")
