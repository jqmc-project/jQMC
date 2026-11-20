"""Targeted numerical evidence for ln_Psi/e_L coverage requirements.

The generated route matrix proves broad analytical-vs-autograd parity.  This
module supplies the orthogonal contracts that a broad tree comparison cannot:

* exact published c-axis block names;
* analytical and autograd equality on each named block;
* finite, nonzero signal on every required block; and
* raw-child checks orthogonal to the production reducer's per-child flat blocks.

The parameter values on the test functions are inspected independently by
``analytic_coverage_registration.py``.  Removing a family/scenario from pytest
therefore creates a registration gap.
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

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gdd.engine.evidence import (
    CaseReceipt,
    ComparisonBinding,
    CompositeCaseReceipt,
    ElectronConfigurationBinding,
    ObservationBinding,
    ProductionRouteCapture,
    emit_completion_record,
    observe_allclose,
    observe_electron_configuration_sequence,
    observe_exact,
    observe_non_numerical,
)
from gdd.engine.electron_configurations import ConditionTargetResult, generate_same_spin_condition_target
import gdd.engine.electron_configurations as electron_generators
from jqmc import derivatives
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import ATOL_PYTHON_FP64_BZ_MEAN, RTOL_PYTHON_FP64_ZERO
from jqmc.derivatives import bridge, ecp_force
from jqmc.derivatives.bridge import _reduce_c_param_grads
from jqmc.determinant import AGPn_data, compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.jqmc_mcmc import _geminal_inv_batched
from jqmc.molecular_orbital import MOs_data
from jqmc.wavefunction import Wavefunction_data
from tests.test_jqmc_kernel._generated_matrix_base import (
    _apply_gem,
    _make_j3,
)
from tests.test_jqmc_kernel._gdd_adapters import (
    case_receipt,
    committed_input,
    current_precision_mode,
    guarantee_id_for_manifest_family,
    parameter_input,
    seeded_single_electron_receipt,
)
from tests.test_jqmc_kernel._grad_fidelity_harness import (
    _build_ham,
    _draw,
    _flags_for_axis,
)
from tests.test_jqmc_kernel.test_param_grad_mask_production_path import (
    _build_pbc_ham,
    _pick_electron_positions,
)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir, pytest.mark.design_pin]

_FAMILIES = ("ln_Psi", "e_L")
_LEAF_SCENARIOS = (
    "direct-mo",
    "agpn",
    "dense-agp",
    "lambda-basis",
    "legacy-jastrow",
    "dtn",
    "composite-j3",
)
_VALIDITY_DISPOSITIONS = (
    "geminal-parameterless-basis",
    "j3-parameterless-basis",
    "unknown-j1-form",
    "unknown-j2-form",
)
_REGIME_REQUIREMENTS = (
    "ln_Psi|regime|R01",
    "e_L|regime|R01",
    "ln_Psi|regime|R02",
    "e_L|regime|R02",
    "ln_Psi|regime|R03",
    "e_L|regime|R03",
    "e_L|regime|R04",
    "e_L|regime|R08",
    "ln_Psi|regime|R09",
    "e_L|regime|R09",
    "ln_Psi|regime|R10",
    "e_L|regime|R10",
)

_LEAF_SOURCE_IDS_BY_SCENARIO = {
    "direct-mo": ("P01",),
    "agpn": ("P02",),
    "dense-agp": ("P03",),
    "lambda-basis": ("P04", "P05"),
    "legacy-jastrow": ("P06", "P07", "P08", "P09", "P10"),
    "dtn": ("P11", "P12", "P13"),
    "composite-j3": ("P14", "P15"),
}

IDENTITY_OBLIGATION_ID = "ln_Psi|identity|B20|potential-independent"


def leaf_obligation_id(family: str, source_id: str) -> str:
    """The P01-P15 leaf obligation id; shared with the derivative registry and the Rust parity suite."""
    return f"{family}|leaf|{source_id}"


def validity_obligation_id(family: str, disposition: str) -> str:
    """The C08/C09 validity obligation id of one disposition (shared, as above)."""
    outcome = "n-a" if disposition.endswith("parameterless-basis") else "error"
    source_id = "C09" if outcome == "n-a" else "C08"
    return f"{family}|{outcome}|{source_id}|{disposition}"


_LEAF_BLOCKS_BY_SOURCE_ID = {
    "P01": ("mo_coefficients_up", "mo_coefficients_dn"),
    "P02": ("mo_coefficients_up", "mo_coefficients_dn", "lambda_diag", "lambda_unpaired"),
    "P03": ("lambda_matrix",),
    "P04": ("lambda_basis_exp",),
    "P05": ("lambda_basis_coeff",),
    "P06": ("j1_param",),
    "P07": ("j2_param",),
    "P08": ("j3_matrix",),
    "P09": ("j3_basis_exp",),
    "P10": ("j3_basis_coeff",),
    "P11": ("j3_poly_alpha",),
    "P12": ("j3_poly_beta",),
    "P13": ("j3_poly_gamma",),
}

_LEAF_SEEDS = {
    "direct-mo": 17,
    "agpn": 19,
    "dense-agp": 23,
    "lambda-basis": 29,
    "legacy-jastrow": 31,
    "dtn": 37,
    "composite-j3": 41,
}

_STRICT_ZONES = ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic")

_SCENARIO_BLOCKS = {
    "direct-mo": {"mo_coefficients_up", "mo_coefficients_dn"},
    "agpn": {"mo_coefficients_up", "mo_coefficients_dn", "lambda_diag", "lambda_unpaired"},
    "dense-agp": {"lambda_matrix"},
    "lambda-basis": {"lambda_basis_exp", "lambda_basis_coeff"},
    "legacy-jastrow": {"j1_param", "j2_param", "j3_matrix", "j3_basis_exp", "j3_basis_coeff"},
    "dtn": {"j3_poly_alpha", "j3_poly_beta", "j3_poly_gamma"},
}

_SCENARIO_FLAGS = {
    "direct-mo": {"opt_geminal": True},
    "agpn": {"opt_geminal": True},
    "dense-agp": {"opt_geminal": True},
    "lambda-basis": {"opt_lambda_basis_exp": True, "opt_lambda_basis_coeff": True},
    "legacy-jastrow": {
        "opt_J1_param": True,
        "opt_J2_param": True,
        "opt_J3_param": True,
        "opt_J3_basis_exp": True,
        "opt_J3_basis_coeff": True,
    },
    "dtn": {
        "opt_JDTN_u_param": True,
        "opt_JDTN_chi_param": True,
        "opt_JDTN_f_param": True,
    },
    "composite-j3": {"opt_J3_basis_exp": True, "opt_J3_basis_coeff": True},
}


def _resolved_flags(overrides: dict) -> dict:
    flags = dict.fromkeys(derivatives._PARAM_FLAG_KEYS, False)
    flags.update(overrides)
    return flags


@pytest.fixture(autouse=True)
def _clear_derivative_caches():
    yield
    bridge.cache_clear()


def _with_j3(ham: Hamiltonian_data, orb_data) -> Hamiltonian_data:
    old = ham.wavefunction_data.jastrow_data
    jastrow = Jastrow_data(
        jastrow_one_body_data=old.jastrow_one_body_data,
        jastrow_two_body_data=old.jastrow_two_body_data,
        jastrow_three_body_data=_make_j3(orb_data),
        jastrow_dtn_data=old.jastrow_dtn_data,
    )
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jastrow))


def _leaf_hamiltonian(scenario: str, *, return_attempts: bool = False):
    attempts = 1
    if scenario == "direct-mo":
        ham = _build_ham("N_ae_ccpvdz_cart.h5", True)
        r_up, r_dn = _draw(ham, seed=17)
    elif scenario == "agpn":
        ham = _apply_gem(_build_ham("N_ae_ccpvdz_cart.h5", True), "AGPn")
        geminal = ham.wavefunction_data.geminal_data
        assert isinstance(geminal, AGPn_data)
        # ``to_AGPn`` initially represents the source SD exactly
        # ([1]*occupied + [0]*virtual).  On that special surface e_L can be
        # invariant to every diagonal occupation at the sampled point, making
        # analytical==autograd a zero-vs-zero check.  Move to a valid correlated
        # AGPn with distinct nonzero occupations so lambda_diag is demonstrably
        # live (the regression that motivated P02/R03).
        correlated_diag = np.linspace(1.25, 0.35, geminal.num_mo, dtype=np.float64)
        geminal = geminal.replace(lambda_diag=correlated_diag)
        ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=geminal))
        r_up, r_dn = _draw(ham, seed=19)
    elif scenario == "dense-agp":
        ham = _build_ham("H2_ae_ccpvdz_cart.h5", False, {"dense_agp": True})
        r_up, r_dn = _draw(ham, seed=23)
    elif scenario == "lambda-basis":
        ham = _build_ham("H2_ae_ccpvdz_cart.h5", False)
        r_up, r_dn = _draw(ham, seed=29)
    elif scenario == "legacy-jastrow":
        ham = _with_j3(
            _build_ham("H2_ae_ccpvdz_cart.h5", True),
            _build_ham("H2_ae_ccpvdz_cart.h5", False).wavefunction_data.geminal_data.orb_data_up_spin.aos_data,
        )
        r_up, r_dn = _draw(ham, seed=31)
    elif scenario == "dtn":
        ham = _build_ham("H2_ae_ccpvdz_cart.h5", False, {"dtn": True})
        r_up, r_dn = _draw(ham, seed=37)
    elif scenario == "composite-j3":
        ham = _build_pbc_ham("pgto+pw", include_dtn=False)
        aos = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
        mo = MOs_data(num_mo=aos.num_ao, aos_data=aos, mo_coefficients=np.eye(aos.num_ao))
        ham = _with_j3(ham, mo)
        r_up, r_dn, attempts = _pick_electron_positions(ham, "pbc_pgto_pw", seed=41, return_attempts=True)
    else:  # pragma: no cover - parametrization and manifest pin the vocabulary
        raise AssertionError(f"unknown leaf scenario: {scenario}")
    result = (ham, r_up, r_dn)
    return (*result, attempts) if return_attempts else result


def _correlated_closed_agpn_hamiltonian() -> Hamiltonian_data:
    """Build a genuine multi-natural-orbital AGPn, not an SD-equivalent AGPn."""
    ham = _build_ham("H2_ae_ccpvdz_cart.h5", True)
    source_sd = ham.wavefunction_data.geminal_data
    aos = source_sd.orb_data_up_spin.aos_data
    occupied = np.asarray(source_sd.orb_data_up_spin.mo_coefficients)
    num_mo = min(3, int(aos.num_ao))
    assert num_mo > int(source_sd.num_electron_up)

    rng = np.random.default_rng(61)
    virtual = rng.normal(size=(num_mo - occupied.shape[0], int(aos.num_ao)))
    coefficients = np.vstack([occupied, virtual])
    mos = MOs_data(num_mo=num_mo, aos_data=aos, mo_coefficients=coefficients)
    mos.sanity_check()
    geminal = AGPn_data.build(
        num_electron_up=int(source_sd.num_electron_up),
        num_electron_dn=int(source_sd.num_electron_dn),
        mos_up=mos,
        lambda_diag=np.linspace(1.3, 0.2, num_mo, dtype=np.float64),
        time_reversal=True,
    )
    assert isinstance(geminal, AGPn_data)
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=geminal))


def _compute_c_pair(ham: Hamiltonian_data, family: str, r_up, r_dn, flags: dict):
    if family == "ln_Psi":
        analytical = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)["c"]
        automatic = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=False)["c"]
    else:
        analytical = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=True)["c"]
        automatic = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=False)["c"]
    return analytical, automatic


def _compute_axis_pair(
    ham: Hamiltonian_data,
    family: str,
    axis: str,
    r_up,
    r_dn,
    RT=None,
    *,
    geminal_inv=None,
):
    flags = {f"opt_{axis}": True}
    if family == "ln_Psi":
        analytical = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True, geminal_inv=geminal_inv)[axis]
        automatic = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=False, geminal_inv=geminal_inv)[axis]
    else:
        RT = jnp.eye(3) if RT is None else RT
        analytical = derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, flags, analytic=True, geminal_inv=geminal_inv)[axis]
        automatic = derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, flags, analytic=False, geminal_inv=geminal_inv)[axis]
    return analytical, automatic


def _batched(tree):
    return jax.tree_util.tree_map(lambda value: jnp.expand_dims(value, 0), tree)


def _declared_blocks(wf: Wavefunction_data, flags: dict) -> set[str]:
    return {block.name for block in wf.get_variational_blocks(state_token=("test", 0), **flags)}


def _stored_blocks(wf: Wavefunction_data, grad, flags: dict) -> dict[str, np.ndarray]:
    declared = _declared_blocks(wf, flags)
    reduced = _reduce_c_param_grads(wf, _batched(grad), 1, flags)
    return {name: np.asarray(value)[0] for name, value in reduced.items() if name in declared}


def _assert_live_equal(name: str, analytical, automatic, *, require_live: bool = True):
    analytical = np.asarray(analytical)
    automatic = np.asarray(automatic)
    assert analytical.size > 0, f"{name}: empty required block"
    assert np.all(np.isfinite(analytical)), f"{name}: analytical block is not finite"
    assert np.all(np.isfinite(automatic)), f"{name}: autograd block is not finite"
    if require_live:
        assert float(np.max(np.abs(automatic))) > 1.0e-12, f"{name}: autograd signal is vacuously zero"
    level = "medium" if current_precision_mode() == "mixed" else "strict"
    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic"),
        level,
    )
    np.testing.assert_allclose(analytical, automatic, atol=atol, rtol=rtol, err_msg=name)
    return analytical, automatic


def _assert_tree_allclose(analytical, automatic, label: str):
    leaves_a = jax.tree_util.tree_leaves(analytical)
    leaves_b = jax.tree_util.tree_leaves(automatic)
    assert len(leaves_a) == len(leaves_b)
    level = "medium" if current_precision_mode() == "mixed" else "strict"
    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic"),
        level,
    )
    pairs = []
    for index, (leaf_a, leaf_b) in enumerate(zip(leaves_a, leaves_b, strict=True)):
        array_a = np.asarray(leaf_a)
        array_b = np.asarray(leaf_b)
        assert np.all(np.isfinite(array_a)), f"{label} analytical leaf {index} is not finite"
        assert np.all(np.isfinite(array_b)), f"{label} autograd leaf {index} is not finite"
        np.testing.assert_allclose(array_a, array_b, atol=atol, rtol=rtol, err_msg=f"{label} leaf {index}")
        pairs.append((array_a, array_b))
    return tuple(pairs)


def _tree_norm(tree) -> float:
    leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(tree)]
    return float(np.sqrt(sum(np.vdot(leaf, leaf).real for leaf in leaves)))


def _tree_difference_norm(left, right) -> float:
    leaves_left = jax.tree_util.tree_leaves(left)
    leaves_right = jax.tree_util.tree_leaves(right)
    assert len(leaves_left) == len(leaves_right)
    return float(
        np.sqrt(
            sum(
                np.vdot(np.asarray(a) - np.asarray(b), np.asarray(a) - np.asarray(b)).real
                for a, b in zip(leaves_left, leaves_right, strict=True)
            )
        )
    )


def _geminal_singular_values(ham: Hamiltonian_data, r_up, r_dn) -> np.ndarray:
    matrix = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, r_up, r_dn))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    assert np.all(np.isfinite(singular_values))
    assert singular_values[-1] > 0.0
    return singular_values


def _condition_target_configuration(
    minimum_cond_g: float,
    maximum_cond_g: float,
    *,
    filename: str = "N_ae_ccpvdz_cart.h5",
    seed: int = 71,
    separation_grid: tuple[float, ...] | None = None,
) -> tuple[Hamiltonian_data, ConditionTargetResult]:
    """Search one declared separation grid using production G evaluations."""
    ham = _build_ham(filename, True)
    r_up, r_dn = _draw(ham, seed=seed)
    result = generate_same_spin_condition_target(
        ham.wavefunction_data.geminal_data,
        r_up,
        r_dn,
        target_minimum_cond_g=minimum_cond_g,
        target_maximum_cond_g=maximum_cond_g,
        separation_grid=(
            separation_grid if separation_grid is not None else tuple(float(value) for value in np.logspace(-3, -13, 11))
        ),
    )
    return ham, result


def _moderately_conditioned_configuration() -> tuple[Hamiltonian_data, ConditionTargetResult]:
    """R02: deterministic, invertible witness in the declared support band."""
    return _condition_target_configuration(1.0e3, 1.0e6)


@pytest.mark.skip_mixed_precision(
    reason="fp64-scale conditioning contract: the generator reports cond_2(G) against "
    "np.finfo(np.float64).eps and searches an fp64 condition band, but the production "
    "geminal call it uses as its oracle returns fp32 values in mixed mode, so the band "
    "is unreachable by construction rather than by defect"
)
def test_condition_target_generation_guarantee(monkeypatch):
    """Certify generation mechanics and measured near-singular G, not derivative accuracy."""
    ham = _build_ham("N_ae_ccpvdz_cart.h5", True)
    initial_r_up, initial_r_dn = _draw(ham, seed=71)
    separation_grid = tuple(float(value) for value in np.logspace(-3, -13, 11))
    route = "jqmc.determinant.compute_geminal_all_elements"
    capture = ProductionRouteCapture(route, lambda args, _kwargs: ((args[1], args[2]),))
    production_operation = electron_generators.compute_geminal_all_elements
    monkeypatch.setattr(electron_generators, "compute_geminal_all_elements", capture.wrap(production_operation))

    arguments = {
        "target_minimum_cond_g": 1.0e11,
        "target_maximum_cond_g": 1.0e12,
        "separation_grid": separation_grid,
    }
    first = generate_same_spin_condition_target(
        ham.wavefunction_data.geminal_data,
        initial_r_up,
        initial_r_dn,
        **arguments,
    )
    second = generate_same_spin_condition_target(
        ham.wavefunction_data.geminal_data,
        initial_r_up,
        initial_r_dn,
        **arguments,
    )
    np.testing.assert_array_equal(first.r_up, second.r_up)
    np.testing.assert_array_equal(first.r_dn, second.r_dn)
    assert first.cond_g == second.cond_g
    assert first.sigma_min_g == second.sigma_min_g
    assert first.generation_attempts == first.rejected_configuration_count + 1

    binding = ElectronConfigurationBinding(
        binding_id="ELECTRONS::ECG-CONDITION-TARGET-001",
        mode="required",
        generation_pattern="condition-target",
        position_source_kind="seeded-random",
        generator_operation="gdd.engine.electron_configurations.generate_same_spin_condition_target",
        generator_arguments=(
            ("control_parameter", "same-spin-separation-delta"),
            ("search_method", "declared-log-grid"),
            ("target_bin", "1e11<cond_2(G)<1e12"),
        ),
        random_algorithm="numpy.random.PCG64",
        seed=71,
        configuration_count=1,
        geminal_conditioning_mode="target-range",
        minimum_cond_g=1.0e11,
        maximum_cond_g=1.0e12,
        lower_bound_comparison="greater-than",
        upper_bound_comparison="less-than",
        minimum_singular_value=1.0e-13,
        conditioning_violation_action="fail-case",
        maximum_generation_attempts=len(separation_grid),
        production_route_capture=route,
    )
    electrons = observe_electron_configuration_sequence(
        binding,
        ((first.r_up, first.r_dn),),
        cond_g_per_configuration=(first.cond_g,),
        sigma_min_g_per_configuration=(first.sigma_min_g,),
        sigma_max_g_per_configuration=(first.sigma_max_g,),
        machine_epsilon_per_configuration=(first.machine_epsilon,),
        cond_times_epsilon_per_configuration=(first.cond_times_epsilon,),
        generation_attempts=first.generation_attempts,
        rejected_configuration_count=first.rejected_configuration_count,
    )
    generation_check = observe_non_numerical(
        ObservationBinding(
            binding_id="GENERATION::ECG-CONDITION-TARGET-001",
            policy_id="OBS-NUMERICAL-REGIME-001",
            precision_mode=current_precision_mode(),
            subject_role="condition-target-electron-configuration-generation",
            required_observations=("declared-pattern-used", "production-g-reached", "target-bin-observed"),
        ),
        {
            "declared-pattern-used": True,
            "production-g-reached": True,
            "target-bin-observed": 1.0e11 < first.cond_g < 1.0e12,
        },
        electron_receipt=electrons,
    )
    receipt = CaseReceipt(
        guarantee_id="CFG-ELECTRON-CONDITION-TARGET-001",
        obligation_id="ECG-CONDITION-TARGET-001",
        pytest_node_id=(
            "tests/test_jqmc_kernel/test_analytic_coverage_evidence.py::test_condition_target_generation_guarantee"
        ),
        input_binding=committed_input(
            "ECG-CONDITION-TARGET-001",
            source_ids=("WF-SRC-N-AE-CART-MOL",),
            builder_operation="tests.test_jqmc_kernel._grad_fidelity_harness._build_ham",
            transformations=("electron-configurations.same-spin-condition-target.v1",),
        ),
        electron_configuration=electrons,
        comparison=generation_check,
        status="passed",
        route_capture=capture.receipt(),
    )
    receipt.validate()


def test_condition_target_generator_cannot_bypass_production_g(monkeypatch):
    """A hard-coded coordinate return cannot satisfy the production-G route."""
    ham = _build_ham("N_ae_ccpvdz_cart.h5", True)
    initial_r_up, initial_r_dn = _draw(ham, seed=71)

    def refuse_production_call(*_args, **_kwargs):
        raise RuntimeError("production G sentinel")

    monkeypatch.setattr(electron_generators, "compute_geminal_all_elements", refuse_production_call)
    with pytest.raises(RuntimeError, match="production G sentinel"):
        generate_same_spin_condition_target(
            ham.wavefunction_data.geminal_data,
            initial_r_up,
            initial_r_dn,
            target_minimum_cond_g=1.0e11,
            target_maximum_cond_g=1.0e12,
            separation_grid=tuple(float(value) for value in np.logspace(-3, -13, 11)),
        )


def _jastrow_channel_hamiltonians() -> tuple[
    Hamiltonian_data,
    dict[str, Hamiltonian_data],
    jax.Array,
    jax.Array,
]:
    """Determinant baseline plus isolated and fully composed Jastrow channels."""
    base = _build_ham("H2_ae_ccpvdz_cart.h5", False)
    legacy = _build_ham("H2_ae_ccpvdz_cart.h5", True).wavefunction_data.jastrow_data
    dtn = _build_ham("H2_ae_ccpvdz_cart.h5", False, {"dtn": True}).wavefunction_data.jastrow_data
    aos = base.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
    j3 = _make_j3(aos)

    def with_channels(*, j1=False, j2=False, j3_enabled=False, dtn_enabled=False):
        jastrow = Jastrow_data(
            jastrow_one_body_data=legacy.jastrow_one_body_data if j1 else None,
            jastrow_two_body_data=legacy.jastrow_two_body_data if j2 else None,
            jastrow_three_body_data=j3 if j3_enabled else None,
            jastrow_dtn_data=dtn.jastrow_dtn_data if dtn_enabled else None,
        )
        return base.replace(wavefunction_data=base.wavefunction_data.replace(jastrow_data=jastrow))

    channels = {
        "J1": with_channels(j1=True),
        "J2": with_channels(j2=True),
        "J3": with_channels(j3_enabled=True),
        "DTN": with_channels(dtn_enabled=True),
        "all": with_channels(j1=True, j2=True, j3_enabled=True, dtn_enabled=True),
    }
    r_up, r_dn = _draw(base, seed=89)
    return base, channels, r_up, r_dn


def _assert_composite_j3_children(ham: Hamiltonian_data, analytical, automatic, flags: dict) -> None:
    expected = _declared_blocks(ham.wavefunction_data, flags)
    assert expected
    assert all(name.startswith("j3_child") for name in expected), expected

    ga = analytical.jastrow_data.jastrow_three_body_data.orb_data.aos_data
    gb = automatic.jastrow_data.jastrow_three_body_data.orb_data.aos_data
    observed: set[str] = set()
    for index, (child_a, child_b) in enumerate(zip(ga.children, gb.children, strict=True)):
        radial_a = getattr(child_a, "radial", None)
        radial_b = getattr(child_b, "radial", None)
        if radial_a is None:
            assert radial_b is None
            continue
        for suffix, field in (("basis_exp", "exponents"), ("basis_coeff", "coefficients")):
            name = f"j3_child{index}_{suffix}"
            observed.add(name)
            _assert_live_equal(name, getattr(radial_a, field), getattr(radial_b, field))
    assert observed == expected


def _leaf_input_binding(obligation_id: str, scenario: str):
    builder = "tests.test_jqmc_kernel.test_analytic_coverage_evidence._leaf_hamiltonian"
    if scenario == "composite-j3":
        return parameter_input(
            obligation_id,
            parameter_set_id="DERIV-LEAF-COMPOSITE-J3-001",
            builder_operation=builder,
            declaration={"orbital-kind": "pgto+pw", "builder-seed": 2025, "jastrow": "composite-j3"},
            transformations=("derivatives.targeted.attach-composite-j3.v1",),
        )
    source_id = "WF-SRC-N-AE-CART-MOL" if scenario in {"direct-mo", "agpn"} else "WF-SRC-H2-AE-CART-MOL"
    transformations = {
        "direct-mo": (),
        "agpn": ("derivatives.targeted.convert-sd-to-correlated-agpn.v1",),
        "dense-agp": ("derivatives.targeted.convert-sd-to-dense-agp.v1",),
        "lambda-basis": (),
        "legacy-jastrow": ("derivatives.targeted.attach-legacy-j1-j2-j3.v1",),
        "dtn": ("derivatives.targeted.attach-dtn.v1",),
    }[scenario]
    return committed_input(
        obligation_id,
        source_ids=(source_id,),
        builder_operation=builder,
        transformations=transformations,
    )


def _record_leaf_receipts(
    family,
    scenario,
    r_up,
    r_dn,
    attempts,
    named_pairs_by_source,
    *,
    emit_completion: bool,
):
    receipts = []
    precision_mode = current_precision_mode()
    level = "medium" if precision_mode == "mixed" else "strict"
    atol, rtol = get_tolerance_min(_STRICT_ZONES, level)
    for source_id in _LEAF_SOURCE_IDS_BY_SCENARIO[scenario]:
        obligation_id = leaf_obligation_id(family, source_id)
        named_pairs = named_pairs_by_source[source_id]
        comparison = observe_allclose(
            ComparisonBinding(
                binding_id=f"COMPARISON::{obligation_id}",
                policy_id=f"CMP-ZONE-ALLCLOSE-{level.upper()}-001",
                precision_mode=precision_mode,
                actual_role="xla-analytical-parameter-leaf",
                reference_role="xla-autodiff-parameter-leaf-same-precision",
                zones=_STRICT_ZONES,
                level=level,
            ),
            tuple((actual, reference) for _name, actual, reference in named_pairs),
            atol=atol,
            rtol=rtol,
        )
        finite = all(
            np.all(np.isfinite(np.asarray(value))) for _name, actual, reference in named_pairs for value in (actual, reference)
        )
        nonzero_by_name = {
            name: float(np.max(np.abs(np.asarray(reference)))) > 1.0e-12 for name, _actual, reference in named_pairs
        }
        if source_id == "P02" and not nonzero_by_name.get("lambda_diag", True):
            nonzero_by_name.pop("lambda_diag")
        liveness = observe_non_numerical(
            ObservationBinding(
                binding_id=f"LIVENESS::{obligation_id}",
                policy_id="OBS-ROUTE-LIVENESS-001",
                precision_mode=current_precision_mode(),
                subject_role="published-parameter-leaves",
                required_observations=(
                    "route-executed",
                    "expected-output-leaves-accounted",
                    "finite",
                    "nonzero",
                ),
            ),
            {
                "route-executed": bool(named_pairs),
                "expected-output-leaves-accounted": bool(named_pairs),
                "finite": finite,
                "nonzero": bool(nonzero_by_name) and all(nonzero_by_name.values()),
            },
        )
        electrons = seeded_single_electron_receipt(
            obligation_id,
            r_up,
            r_dn,
            generator_operation=(
                "tests.test_jqmc_kernel.test_param_grad_mask_production_path._pick_electron_positions"
                if scenario == "composite-j3"
                else "tests.test_jqmc_kernel._grad_fidelity_harness._draw"
            ),
            seed=_LEAF_SEEDS[scenario],
            generator_arguments=(
                (("selection", "node-far"), ("scale", 0.6))
                if scenario == "composite-j3"
                else (("distribution", "normal"), ("scale", 0.7))
            ),
            generation_pattern="direct",
            generation_attempts=attempts,
            rejected_configuration_count=attempts - 1,
        )
        receipts.append(
            case_receipt(
                guarantee_id=guarantee_id_for_manifest_family(family),
                obligation_id=obligation_id,
                pytest_node_id=(
                    "tests/test_jqmc_kernel/test_analytic_coverage_evidence.py::"
                    f"test_parameter_leaf_evidence[{scenario}-{family}]"
                ),
                input_binding=_leaf_input_binding(obligation_id, scenario),
                electron_configuration=electrons,
                primary_check=comparison,
                additional_checks=(liveness,),
                emit_completion=emit_completion,
            )
        )
    return tuple(receipts)


def _run_parameter_leaf_evidence(family, scenario, *, emit_completion: bool = True):
    """Every P01-P15 leaf is present, finite, live, and equal to autograd."""
    ham, r_up, r_dn, attempts = _leaf_hamiltonian(scenario, return_attempts=True)
    flags = _resolved_flags(_SCENARIO_FLAGS[scenario])
    analytical, automatic = _compute_c_pair(ham, family, r_up, r_dn, flags)

    if scenario == "composite-j3":
        expected = _declared_blocks(ham.wavefunction_data, flags)
        analytical_blocks = _stored_blocks(ham.wavefunction_data, analytical, flags)
        automatic_blocks = _stored_blocks(ham.wavefunction_data, automatic, flags)
        assert set(analytical_blocks) == expected
        assert set(automatic_blocks) == expected
        for name in sorted(expected):
            _assert_live_equal(name, analytical_blocks[name], automatic_blocks[name])
        _assert_composite_j3_children(ham, analytical, automatic, flags)

        # Publication and consumption are one contract: every emitted child
        # block must also route an optimizer update back to that same child.
        blocks = ham.wavefunction_data.get_variational_blocks(state_token=("test", 0), **flags)
        delta = np.ones(sum(block.size for block in blocks), dtype=np.float64)
        updated = ham.wavefunction_data.apply_block_updates(blocks, delta, learning_rate=1.0e-4)
        updated_blocks = {
            block.name: np.asarray(block.values) for block in updated.get_variational_blocks(state_token=("test", 1), **flags)
        }
        assert set(updated_blocks) == expected
        for block in blocks:
            np.testing.assert_array_equal(updated_blocks[block.name], np.asarray(block.values) + 1.0e-4)
        named_pairs_by_source = {
            "P14": [
                (name, analytical_blocks[name], automatic_blocks[name])
                for name in sorted(expected)
                if name.endswith("basis_exp")
            ],
            "P15": [
                (name, analytical_blocks[name], automatic_blocks[name])
                for name in sorted(expected)
                if name.endswith("basis_coeff")
            ],
        }
        receipts = _record_leaf_receipts(
            family,
            scenario,
            r_up,
            r_dn,
            attempts,
            named_pairs_by_source,
            emit_completion=emit_completion,
        )
        assert len(receipts) == 2
        return receipts

    expected = _SCENARIO_BLOCKS[scenario]
    assert _declared_blocks(ham.wavefunction_data, flags) == expected
    analytical_blocks = _stored_blocks(ham.wavefunction_data, analytical, flags)
    automatic_blocks = _stored_blocks(ham.wavefunction_data, automatic, flags)
    assert set(analytical_blocks) == expected
    assert set(automatic_blocks) == expected
    for name in sorted(expected):
        _assert_live_equal(
            name,
            analytical_blocks[name],
            automatic_blocks[name],
            # The open-shell N determinant occupies the complete up-spin MO
            # space, so its e_L/lambda_diag signal is exactly invariant.  P02
            # also runs the correlated closed-shell control below, where this
            # leaf is required to be live.
            require_live=not (scenario == "agpn" and family == "e_L" and name == "lambda_diag"),
        )

    named_pairs_by_source = {
        source_id: [
            (name, analytical_blocks[name], automatic_blocks[name])
            for name in _LEAF_BLOCKS_BY_SOURCE_ID[source_id]
            if name in analytical_blocks
        ]
        for source_id in _LEAF_SOURCE_IDS_BY_SCENARIO[scenario]
    }

    if scenario == "agpn" and family == "e_L":
        closed_agpn = _correlated_closed_agpn_hamiltonian()
        geminal = closed_agpn.wavefunction_data.geminal_data
        assert isinstance(geminal, AGPn_data)
        r_up_closed, r_dn_closed = _draw(closed_agpn, seed=59)
        a_closed, b_closed = _compute_c_pair(closed_agpn, family, r_up_closed, r_dn_closed, flags)
        a_blocks = _stored_blocks(closed_agpn.wavefunction_data, a_closed, flags)
        b_blocks = _stored_blocks(closed_agpn.wavefunction_data, b_closed, flags)
        assert set(a_blocks) == {"mo_coefficients_up", "lambda_diag"}
        assert set(b_blocks) == {"mo_coefficients_up", "lambda_diag"}
        _assert_live_equal("lambda_diag[correlated-closed-shell]", a_blocks["lambda_diag"], b_blocks["lambda_diag"])
        named_pairs_by_source["P02"].append(
            ("lambda_diag[correlated-closed-shell]", a_blocks["lambda_diag"], b_blocks["lambda_diag"])
        )

    receipts = _record_leaf_receipts(
        family,
        scenario,
        r_up,
        r_dn,
        attempts,
        named_pairs_by_source,
        emit_completion=emit_completion,
    )
    assert len(receipts) == len(_LEAF_SOURCE_IDS_BY_SCENARIO[scenario])
    return receipts


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("scenario", _LEAF_SCENARIOS)
def test_parameter_leaf_evidence(family, scenario):
    """Every P01-P15 leaf is present, finite, live, and equal to autograd."""
    _run_parameter_leaf_evidence(family, scenario)


def _parameterless_hamiltonian(disposition: str, *, return_attempts: bool = False):
    ham = _build_pbc_ham("pw", include_dtn=False)
    if disposition == "geminal-parameterless-basis":
        flags = _resolved_flags({"opt_lambda_basis_exp": True, "opt_lambda_basis_coeff": True})
    else:
        aos = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
        mo = MOs_data(num_mo=aos.num_ao, aos_data=aos, mo_coefficients=np.eye(aos.num_ao))
        ham = _with_j3(ham, mo)
        flags = _resolved_flags({"opt_J3_basis_exp": True, "opt_J3_basis_coeff": True})
    r_up, r_dn, attempts = _pick_electron_positions(ham, "pbc_pw", seed=43, return_attempts=True)
    result = (ham, r_up, r_dn, flags)
    return (*result, attempts) if return_attempts else result


def _invalid_jastrow_hamiltonian(disposition: str) -> tuple[Hamiltonian_data, jax.Array, jax.Array, dict]:
    ham = _build_ham("H2_ae_ccpvdz_cart.h5", True)
    jd = ham.wavefunction_data.jastrow_data
    if disposition == "unknown-j1-form":
        jd = jd.replace(jastrow_one_body_data=jd.jastrow_one_body_data.replace(jastrow_1b_type="unknown"))
        flags = _resolved_flags({"opt_J1_param": True})
    else:
        jd = jd.replace(jastrow_two_body_data=jd.jastrow_two_body_data.replace(jastrow_2b_type="unknown"))
        flags = _resolved_flags({"opt_J2_param": True})
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))
    r_up, r_dn = _draw(ham, seed=47)
    return ham, r_up, r_dn, flags


def _validity_input_binding(obligation_id: str, disposition: str):
    if disposition.endswith("parameterless-basis"):
        return parameter_input(
            obligation_id,
            parameter_set_id=f"DERIV-VALIDITY-{disposition.upper()}-001",
            builder_operation="tests.test_jqmc_kernel.test_analytic_coverage_evidence._parameterless_hamiltonian",
            declaration={"orbital-kind": "pbc-pw", "disposition": disposition, "builder-seed": 2025},
        )
    return committed_input(
        obligation_id,
        source_ids=("WF-SRC-H2-AE-CART-MOL",),
        builder_operation="tests.test_jqmc_kernel.test_analytic_coverage_evidence._invalid_jastrow_hamiltonian",
        transformations=(f"derivatives.targeted.inject-{disposition}.v1",),
    )


def _record_validity_receipt(family, disposition, r_up, r_dn, primary_check, *, attempts=1):
    outcome = "n-a" if disposition.endswith("parameterless-basis") else "error"
    obligation_id = validity_obligation_id(family, disposition)
    electrons = seeded_single_electron_receipt(
        obligation_id,
        r_up,
        r_dn,
        generator_operation=(
            "tests.test_jqmc_kernel.test_param_grad_mask_production_path._pick_electron_positions"
            if outcome == "n-a"
            else "tests.test_jqmc_kernel._grad_fidelity_harness._draw"
        ),
        seed=43 if outcome == "n-a" else 47,
        generator_arguments=(
            (("selection", "node-far"), ("scale", 0.6)) if outcome == "n-a" else (("distribution", "normal"), ("scale", 0.7))
        ),
        generation_pattern="direct",
        generation_attempts=attempts,
        rejected_configuration_count=attempts - 1,
    )
    return case_receipt(
        guarantee_id=guarantee_id_for_manifest_family(family),
        obligation_id=obligation_id,
        pytest_node_id=(
            f"tests/test_jqmc_kernel/test_analytic_coverage_evidence.py::test_validity_disposition[{disposition}-{family}]"
        ),
        input_binding=_validity_input_binding(obligation_id, disposition),
        electron_configuration=electrons,
        primary_check=primary_check,
    )


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("disposition", _VALIDITY_DISPOSITIONS)
def test_validity_disposition(family, disposition):
    """Parameterless owners are explicit n/a; invalid J1/J2 forms raise."""
    if disposition.endswith("parameterless-basis"):
        ham, r_up, r_dn, flags, attempts = _parameterless_hamiltonian(disposition, return_attempts=True)
        declared = _declared_blocks(ham.wavefunction_data, flags)
        assert declared == set()
        analytical, _automatic = _compute_c_pair(ham, family, r_up, r_dn, flags)
        stored = _stored_blocks(ham.wavefunction_data, analytical, flags)
        assert stored == {}
        observation = observe_non_numerical(
            ObservationBinding(
                binding_id=f"N-A::{family}::{disposition}",
                policy_id="OBS-REVIEWED-N-A-001",
                precision_mode=current_precision_mode(),
                subject_role="parameter-owner-applicability",
                required_observations=("applicability-rule-matched", "prohibited-output-absent"),
                applicability_rule_id=f"C09::{disposition}",
            ),
            {"applicability-rule-matched": not declared, "prohibited-output-absent": not stored},
        )
        _record_validity_receipt(family, disposition, r_up, r_dn, observation, attempts=attempts)
        return

    ham, r_up, r_dn, flags = _invalid_jastrow_hamiltonian(disposition)
    with pytest.raises(ValueError, match=r"unknown|Unknown") as exc_info:
        _compute_c_pair(ham, family, r_up, r_dn, flags)
    observation = observe_non_numerical(
        ObservationBinding(
            binding_id=f"ERROR::{family}::{disposition}",
            policy_id="OBS-EXPECTED-ERROR-001",
            precision_mode=current_precision_mode(),
            subject_role="invalid-jastrow-production-call",
            required_observations=("expected-error-observed", "exception-type-matched", "message-matched"),
        ),
        {
            "expected-error-observed": exc_info.value is not None,
            "exception-type-matched": isinstance(exc_info.value, ValueError),
            "message-matched": re.search(r"unknown", str(exc_info.value), re.IGNORECASE) is not None,
        },
    )
    _record_validity_receipt(family, disposition, r_up, r_dn, observation)


def _assert_tree_exact(left, right, label: str) -> None:
    leaves_left = jax.tree_util.tree_leaves(left)
    leaves_right = jax.tree_util.tree_leaves(right)
    assert len(leaves_left) == len(leaves_right)
    for index, (leaf_left, leaf_right) in enumerate(zip(leaves_left, leaves_right, strict=True)):
        np.testing.assert_array_equal(np.asarray(leaf_left), np.asarray(leaf_right), err_msg=f"{label} leaf {index}")


def test_v1_potential_independence_identity():
    """ln_Psi depends on the wavefunction/geometry, not on AE versus ECP potential data."""
    ae = _build_ham("H2_ae_ccpvdz_cart.h5", True)
    ecp = _build_ham("H2_ecp_ccpvdz_cart.h5", True)
    same_wavefunction_ecp = ae.replace(coulomb_potential_data=ecp.coulomb_potential_data)
    r_up, r_dn = _draw(ae, seed=53)
    compared = []
    for axis in ("c", "r", "R", "eps"):
        flags = _flags_for_axis(axis)
        left = derivatives.compute_grad_ln_Psi(ae, r_up, r_dn, flags, analytic=True)[axis]
        right = derivatives.compute_grad_ln_Psi(same_wavefunction_ecp, r_up, r_dn, flags, analytic=True)[axis]
        _assert_tree_exact(left, right, axis)
        compared.extend(zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True))
    obligation_id = IDENTITY_OBLIGATION_ID
    precision_mode = "mixed" if is_mixed_precision_enabled() else "full"
    exact = observe_exact(
        ComparisonBinding(
            binding_id=f"EXACT::{obligation_id}::{precision_mode}",
            policy_id="CMP-EXACT-VALUES-001",
            precision_mode=precision_mode,
            actual_role="v1-analytical-with-ae-potential",
            reference_role="v1-analytical-with-substituted-ecp-potential",
        ),
        compared,
    )
    electrons = seeded_single_electron_receipt(
        obligation_id,
        r_up,
        r_dn,
        generator_operation="tests.test_jqmc_kernel._grad_fidelity_harness._draw",
        seed=53,
        generator_arguments=(("distribution", "normal"), ("scale", 0.7)),
        generation_pattern="direct",
    )
    case_receipt(
        guarantee_id=guarantee_id_for_manifest_family("ln_Psi"),
        obligation_id=obligation_id,
        pytest_node_id=("tests/test_jqmc_kernel/test_analytic_coverage_evidence.py::test_v1_potential_independence_identity"),
        input_binding=committed_input(
            obligation_id,
            source_ids=("WF-SRC-H2-AE-CART-MOL", "WF-SRC-H2-ECP-CART-MOL"),
            builder_operation="tests.test_jqmc_kernel.test_analytic_coverage_evidence.test_v1_potential_independence_identity",
            transformations=("derivatives.targeted.substitute-ecp-potential-data.v1",),
        ),
        electron_configuration=electrons,
        primary_check=exact,
    )


def _record_continuous_regime_receipt(
    requirement_id,
    *,
    source_ids,
    configurations,
    compared_pairs,
    seed,
    generation_pattern,
    transformations=(),
    conditioning=None,
    generation_attempts=None,
):
    family, _category, regime = requirement_id.split("|")
    precision_mode = current_precision_mode()
    level = "medium" if precision_mode == "mixed" else "strict"
    atol, rtol = get_tolerance_min(_STRICT_ZONES, level)
    comparison = observe_allclose(
        ComparisonBinding(
            binding_id=f"COMPARISON::{requirement_id}",
            policy_id=f"CMP-ZONE-ALLCLOSE-{level.upper()}-001",
            precision_mode=precision_mode,
            actual_role="xla-analytical-regime-result",
            reference_role="xla-autodiff-regime-result-same-precision",
            zones=_STRICT_ZONES,
            level=level,
        ),
        compared_pairs,
        atol=atol,
        rtol=rtol,
    )
    conditioning = {} if conditioning is None else conditioning
    electron_binding = ElectronConfigurationBinding(
        binding_id=f"ELECTRONS::{requirement_id}",
        mode="required",
        generation_pattern=generation_pattern,
        position_source_kind="seeded-random",
        generator_operation="tests.test_jqmc_kernel.test_analytic_coverage_evidence.test_numerical_regime_evidence",
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
    electrons = observe_electron_configuration_sequence(
        electron_binding,
        configurations,
        cond_g_per_configuration=conditioning.get("cond_g", ()),
        sigma_min_g_per_configuration=conditioning.get("sigma_min_g", ()),
        sigma_max_g_per_configuration=conditioning.get("sigma_max_g", ()),
        machine_epsilon_per_configuration=conditioning.get("machine_epsilon", ()),
        cond_times_epsilon_per_configuration=conditioning.get("cond_times_epsilon", ()),
        generation_attempts=(len(configurations) if generation_attempts is None else generation_attempts),
        rejected_configuration_count=(0 if generation_attempts is None else generation_attempts - len(configurations)),
    )
    precondition = observe_non_numerical(
        ObservationBinding(
            binding_id=f"REGIME::{requirement_id}",
            policy_id="OBS-NUMERICAL-REGIME-001",
            precision_mode=current_precision_mode(),
            subject_role=f"numerical-regime-{regime}",
            required_observations=("numerical-precondition-observed", "route-executed"),
        ),
        {"numerical-precondition-observed": True, "route-executed": bool(compared_pairs)},
    )
    return case_receipt(
        guarantee_id=guarantee_id_for_manifest_family(family),
        obligation_id=requirement_id,
        pytest_node_id=(
            f"tests/test_jqmc_kernel/test_analytic_coverage_evidence.py::test_numerical_regime_evidence[{requirement_id}]"
        ),
        input_binding=committed_input(
            requirement_id,
            source_ids=source_ids,
            builder_operation="tests.test_jqmc_kernel.test_analytic_coverage_evidence.test_numerical_regime_evidence",
            transformations=transformations,
        ),
        electron_configuration=electrons,
        primary_check=comparison,
        additional_checks=(precondition,),
    )


@pytest.mark.parametrize("requirement_id", _REGIME_REQUIREMENTS)
def test_numerical_regime_evidence(requirement_id):
    """Assert that every named numerical precondition is live, then compare AD."""
    family, category, regime = requirement_id.split("|")
    assert category == "regime"

    if regime == "R01":
        ham = _build_ham("N_ae_ccpvdz_cart.h5", True)
        r_up, r_dn = _draw(ham, seed=71)
        singular_values = _geminal_singular_values(ham, r_up, r_dn)
        condition_number = float(singular_values[0] / singular_values[-1])
        assert singular_values[-1] / singular_values[0] > 1.0e-3
        assert condition_number < 1.0e3
        analytical, automatic = _compute_axis_pair(ham, family, "r", r_up, r_dn)
        pairs = _assert_tree_allclose(analytical, automatic, requirement_id)
        _record_continuous_regime_receipt(
            requirement_id,
            source_ids=("WF-SRC-N-AE-CART-MOL",),
            configurations=((r_up, r_dn),),
            compared_pairs=pairs,
            seed=71,
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
        # R02 qualifies the production continuous-observable route, which
        # consumes the same value-only running inverse as MCMC/GFMC.  Testing
        # the fresh fused-inverse route here would leave the actual high-cond(G)
        # production contract uncovered.
        _geminal, geminal_inv, _sign, _logabsdet = _geminal_inv_batched(
            ham.wavefunction_data.geminal_data,
            r_up[None, ...],
            r_dn[None, ...],
        )
        analytical, automatic = _compute_axis_pair(
            ham,
            family,
            "r",
            r_up,
            r_dn,
            geminal_inv=geminal_inv[0],
        )
        pairs = _assert_tree_allclose(analytical, automatic, requirement_id)
        _record_continuous_regime_receipt(
            requirement_id,
            source_ids=("WF-SRC-N-AE-CART-MOL",),
            configurations=((r_up, r_dn),),
            compared_pairs=pairs,
            seed=71,
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
        )
        return

    if regime == "R03":
        # R03 is the conjunction of P01-P15, not a static declaration.  Re-run
        # every leaf assertion here so this requirement has executable evidence
        # even when selected independently of the leaf tests.
        component_receipts = []
        for scenario in _LEAF_SCENARIOS:
            component_receipts.extend(_run_parameter_leaf_evidence(family, scenario, emit_completion=False))
            bridge.cache_clear()
        completion = observe_non_numerical(
            ObservationBinding(
                binding_id=f"COMPOSITE::{requirement_id}",
                policy_id="OBS-COMPOSITE-CLOSURE-001",
                precision_mode=current_precision_mode(),
                subject_role="P01-through-P15-leaf-receipts",
                required_observations=("required-components-complete", "component-receipts-valid"),
            ),
            {
                "required-components-complete": len(component_receipts) == 15,
                "component-receipts-valid": all(receipt.status == "passed" for receipt in component_receipts),
            },
        )
        composite = CompositeCaseReceipt(
            guarantee_id=guarantee_id_for_manifest_family(family),
            obligation_id=requirement_id,
            pytest_node_id=(
                f"tests/test_jqmc_kernel/test_analytic_coverage_evidence.py::test_numerical_regime_evidence[{requirement_id}]"
            ),
            component_receipts=tuple(component_receipts),
            completion_check=completion,
            status="passed",
        )
        composite.validate()
        emit_completion_record(composite, record_kind="composite")
        return

    if regime == "R04":
        assert family == "e_L"
        ham = _build_ham("lithium_k025.h5", True)
        r_up, r_dn = _draw(ham, seed=73)
        common = ecp_force._ecp_nl_common(ham, r_up, r_dn, jnp.eye(3), None)
        ratio = np.asarray(common["ratio"])
        assert np.iscomplexobj(ratio)
        assert float(np.max(np.abs(np.real(ratio)))) > 1.0e-8
        assert float(np.max(np.abs(np.imag(ratio)))) > 1.0e-8
        analytical, automatic = _compute_axis_pair(ham, family, "R", r_up, r_dn)
        pairs = _assert_tree_allclose(analytical, automatic, requirement_id)
        _record_continuous_regime_receipt(
            requirement_id,
            source_ids=("WF-SRC-LI-ECP-SPHE-NON-GAMMA",),
            configurations=((r_up, r_dn),),
            compared_pairs=pairs,
            seed=73,
            generation_pattern="direct",
        )
        return

    if regime == "R08":
        assert family == "e_L"
        ham = _build_ham("lithium.h5", True)
        # This fixture has a 1x1 geminal, so cond(G) is identically one and cannot
        # detect inverse amplification near a node.  Screen on sigma_min using only
        # the XLA geminal value, before either derivative backend is evaluated.
        maximum_generation_attempts = 32
        minimum_singular_value = 2.0e-3
        for attempt in range(maximum_generation_attempts):
            r_up, r_dn = _draw(ham, seed=79 + attempt)
            singular_values = _geminal_singular_values(ham, r_up, r_dn)
            condition_number = float(singular_values[0] / singular_values[-1])
            if 1.0 <= condition_number < 2.0 and float(singular_values[-1]) >= minimum_singular_value:
                break
        else:
            raise AssertionError(
                "R08 could not draw a 1x1 geminal with "
                f"sigma_min(G) >= {minimum_singular_value:g} after {maximum_generation_attempts} candidates"
            )
        theta = 0.43
        rotated = jnp.asarray(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(
            np.asarray(rotated.T @ rotated),
            np.eye(3),
            atol=ATOL_PYTHON_FP64_BZ_MEAN,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        assert not np.array_equal(np.asarray(rotated), np.eye(3))
        eye_analytical, eye_automatic = _compute_axis_pair(ham, family, "R", r_up, r_dn, jnp.eye(3))
        rot_analytical, rot_automatic = _compute_axis_pair(ham, family, "R", r_up, r_dn, rotated)
        pairs = _assert_tree_allclose(eye_analytical, eye_automatic, f"{requirement_id}[identity]")
        pairs += _assert_tree_allclose(rot_analytical, rot_automatic, f"{requirement_id}[rotated]")
        assert _tree_difference_norm(eye_automatic, rot_automatic) > 1.0e-8
        _record_continuous_regime_receipt(
            requirement_id,
            source_ids=("WF-SRC-LI-ECP-SPHE-GAMMA",),
            configurations=((r_up, r_dn),),
            compared_pairs=pairs,
            seed=79,
            generation_pattern="rejection-screen",
            conditioning={
                "mode": "target-range",
                "minimum_cond_g": 1.0,
                "maximum_cond_g": 2.0,
                "lower_bound_comparison": "greater-than-or-equal",
                "upper_bound_comparison": "less-than",
                "minimum_singular_value": minimum_singular_value,
                "violation_action": "deterministic-resample",
                "maximum_generation_attempts": maximum_generation_attempts,
                "generator_arguments": (
                    ("candidate_seed", "base-seed-plus-attempt"),
                    ("minimum_sigma_g", f"{minimum_singular_value:g}"),
                    ("regime", regime),
                ),
                "cond_g": (condition_number,),
                "sigma_min_g": (float(singular_values[-1]),),
            },
            generation_attempts=attempt + 1,
        )
        return

    if regime == "R09":
        flags = _resolved_flags({"opt_geminal": True})
        configurations = []
        pairs = []
        for stem, expected_tie, expected_blocks in (
            ("H2_ae_ccpvdz_cart.h5", True, {"mo_coefficients_up"}),
            ("N_ae_ccpvdz_cart.h5", False, {"mo_coefficients_up", "mo_coefficients_dn"}),
        ):
            ham = _build_ham(stem, True)
            assert bool(ham.wavefunction_data.geminal_data.time_reversal) is expected_tie
            r_up, r_dn = _draw(ham, seed=83)
            configurations.append((r_up, r_dn))
            analytical, automatic = _compute_c_pair(ham, family, r_up, r_dn, flags)
            ga = analytical.geminal_data
            gb = automatic.geminal_data
            pairs.append(
                _assert_live_equal(
                    f"{requirement_id}[{stem}:raw-up]",
                    ga.orb_data_up_spin.mo_coefficients,
                    gb.orb_data_up_spin.mo_coefficients,
                )
            )
            pairs.append(
                _assert_live_equal(
                    f"{requirement_id}[{stem}:raw-down]",
                    ga.orb_data_dn_spin.mo_coefficients,
                    gb.orb_data_dn_spin.mo_coefficients,
                )
            )
            analytical_blocks = _stored_blocks(ham.wavefunction_data, analytical, flags)
            automatic_blocks = _stored_blocks(ham.wavefunction_data, automatic, flags)
            assert set(analytical_blocks) == expected_blocks
            assert set(automatic_blocks) == expected_blocks
            for name in expected_blocks:
                pairs.append(_assert_live_equal(name, analytical_blocks[name], automatic_blocks[name]))
        _record_continuous_regime_receipt(
            requirement_id,
            source_ids=("WF-SRC-H2-AE-CART-MOL", "WF-SRC-N-AE-CART-MOL"),
            configurations=tuple(configurations),
            compared_pairs=tuple(pairs),
            seed=83,
            generation_pattern="direct",
        )
        return

    if regime == "R10":
        base, channels, r_up, r_dn = _jastrow_channel_hamiltonians()
        base_analytical, base_automatic = _compute_axis_pair(base, family, "r", r_up, r_dn)
        pairs = _assert_tree_allclose(base_analytical, base_automatic, f"{requirement_id}[determinant]")
        assert _tree_norm(base_automatic) > 1.0e-12
        for channel, ham in channels.items():
            analytical, automatic = _compute_axis_pair(ham, family, "r", r_up, r_dn)
            pairs += _assert_tree_allclose(analytical, automatic, f"{requirement_id}[{channel}]")
            assert _tree_norm(automatic) > 1.0e-12
            assert _tree_difference_norm(automatic, base_automatic) > 1.0e-10
        _record_continuous_regime_receipt(
            requirement_id,
            source_ids=("WF-SRC-H2-AE-CART-MOL",),
            configurations=((r_up, r_dn),),
            compared_pairs=pairs,
            seed=89,
            generation_pattern="direct",
            transformations=("derivatives.regime.compose-jastrow-channels.v1",),
        )
        return

    raise AssertionError(f"unknown regime requirement: {requirement_id}")
