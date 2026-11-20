"""Small real-numerics sentinels for independent GDD failure mechanisms."""

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

import numpy as np
import pytest


pytestmark = [pytest.mark.light, pytest.mark.without_chdir, pytest.mark.gdd_runs_science]


def test_independent_oracle_rebuilds_a_shared_wrong_matrix(monkeypatch):
    """A self-consistent but wrong determinant matrix cannot certify its own inverse."""
    from jax import random

    from tests import _vmc_fixed_input_oracle as oracle

    geminal = SimpleNamespace(num_electron_up=1, num_electron_dn=1)
    jastrow = object()
    nuclei, charges = np.zeros((1, 3)), np.ones(1)
    hamiltonian = SimpleNamespace(
        wavefunction_data=SimpleNamespace(geminal_data=geminal, jastrow_data=jastrow),
        structure_data=SimpleNamespace(positions=nuclei, atomic_numbers=charges),
        coulomb_potential_data=SimpleNamespace(z_cores=np.zeros(1)),
    )
    rebuilt = []
    wrong_matrix = np.array([[101.0]])
    wrong_inverse = np.linalg.inv(wrong_matrix)

    def fresh(actual_geminal, r_up, r_dn):
        assert actual_geminal is geminal
        rebuilt.append((r_up.copy(), r_dn.copy()))
        return np.array([[2.0]])

    def wrong_update(actual_hamiltonian, r_up, r_dn, matrix, inverse, dt, eps, key, normal):
        assert actual_hamiltonian is hamiltonian
        return r_up, r_dn, wrong_matrix.copy(), wrong_inverse.copy(), key, False

    monkeypatch.setattr(oracle, "compute_geminal_all_elements", fresh)
    monkeypatch.setattr(oracle, "_one_move", wrong_update)
    monkeypatch.setattr(oracle, "compute_local_energy", lambda *_: 3.0)
    rng = np.random.default_rng(0)
    r_up = nuclei + rng.normal(scale=0.5, size=(1, 3))
    r_dn = nuclei + rng.normal(scale=0.5, size=(1, 3))
    key = np.asarray(random.PRNGKey(0), dtype=np.uint32)
    matrix, inverse = np.array([[2.0]]), np.array([[0.5]])
    sweeps, evaluations = [], []
    for index in range(2):
        inputs = (
            geminal,
            jastrow,
            nuclei,
            charges,
            r_up,
            r_dn,
            matrix,
            inverse,
            1.5,
            0.0,
            *map(int, key),
            rng.normal(size=2),
        )
        output = {
            "r_up": r_up,
            "r_dn": r_dn,
            "geminal": wrong_matrix.copy(),
            "geminal_inv": wrong_inverse.copy(),
            "accepted": 0,
            "final_key": key,
        }
        sweeps.append((inputs, output))
        mesh = oracle.icosahedron_sym_mesh_Nv12
        evaluations.append((index, hamiltonian, r_up, r_dn, np.asarray(mesh.weights), np.asarray(mesh.grid_points)))
        matrix, inverse = wrong_matrix.copy(), wrong_inverse.copy()

    reference = oracle.vmc_reference(
        hamiltonian,
        seed=0,
        dt=1.5,
        eps=0.0,
        warmup=0,
        measurements=2,
        sweeps=sweeps,
        evaluations=evaluations,
    )
    actual_states = np.asarray([oracle.state_vector((r_up, r_dn, row["geminal"])) for _, row in sweeps])
    np.testing.assert_array_equal(
        np.asarray([row["geminal_inv"] for _, row in sweeps]),
        reference["inverses"],
    )
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        np.testing.assert_array_equal(actual_states, reference["states"])
    assert len(rebuilt) == 3


def test_state_handoff_rejects_a_frozen_carried_inverse(monkeypatch):
    """A mutually stale AO cache and inverse cannot certify the final positions."""
    import jax.numpy as jnp

    from tests.test_jqmc_kernel import test_determinant as evidence

    monkeypatch.setattr(evidence, "_advance_det_ratio_streaming_state", lambda geminal, state, *args: state)

    def zero_update(*, geminal_data, **kwargs):
        return jnp.zeros(geminal_data.num_electron_up, dtype=jnp.float64)

    monkeypatch.setattr(evidence, "_compute_v_up_move_from_det_ratio_state", zero_update)
    monkeypatch.setattr(evidence, "_compute_u_dn_move_from_det_ratio_state", zero_update)
    with pytest.raises(AssertionError, match="fresh final-position geminal"):
        evidence.TestDeterminantStreamingState().test_conditioned_sm_inverse_chain_matches_fresh_cached_state()


def test_runtime_consumption_rejects_a_missing_real_kinetic_call():
    """A real canonical input is insufficient when its kinetic consumer record is absent."""
    from gdd.engine.canonical_use import CanonicalKineticCapture, KINETIC_OPERATION, canonical_use_errors
    from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian

    hamiltonian = build_canonical_hamiltonian("WF-PATTERN-001")
    wavefunction = hamiltonian.wavefunction_data
    rng = np.random.Generator(np.random.PCG64(864))
    r_up, r_dn = rng.normal(scale=0.6, size=(2, 3)), rng.normal(scale=0.6, size=(2, 3))
    capture = CanonicalKineticCapture(hamiltonian)
    capture.evaluate(wavefunction, r_up, r_dn)
    requirement = {
        "suite": "WF-SUITE-CORE-001",
        "selector": "pattern_id",
        "operation": KINETIC_OPERATION,
        "comparison": "numerical-value",
        "electron_configuration": {
            "algorithm": "numpy.random.PCG64.normal",
            "seed": 864,
            "scale": 0.6,
            "configurations": 1,
            "cond_G": "finite-not-limited",
        },
    }
    output = capture.records[0]["output"]
    checks = {"numerical-value": {"site": "sentinel"}}
    invocations = [
        {
            "tolerance_calls": [
                {
                    "site": "sentinel",
                    "calls": [
                        {
                            "outcome": "normal",
                            "operand_snapshots": {"actual": output, "reference": output},
                        }
                    ],
                }
            ]
        }
    ]
    selector = {"pattern_id": "WF-PATTERN-001"}
    assert canonical_use_errors(requirement, selector, capture.records, checks, invocations) == []
    assert canonical_use_errors(requirement, selector, [], checks, invocations)


def test_aggregation_rejects_dropped_force_history(monkeypatch):
    """Production/debug agreement cannot hide a missing fixed-ensemble history."""
    from tests.test_jqmc_kernel import test_gfmc_estimator_reference as science

    science.test_gfmc_force_matches_fixed_ensemble_energy_derivative("conventional", True, "history-warmup")
    original = science._stored_driver

    def without_history(*args, **kwargs):
        driver = original(*args, **kwargs)
        driver._num_gfmc_collect_steps = 0
        return driver

    monkeypatch.setattr(science, "_stored_driver", without_history)
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        science.test_gfmc_force_matches_fixed_ensemble_energy_derivative("conventional", True, "history-warmup")


def test_component_activity_exposes_the_second_composite_child():
    """Structural and numerical sentinels expose inactive composite/Jastrow leaves."""
    import copy
    from dataclasses import replace

    from jqmc._precision import get_tolerance_min
    from gdd.engine.canonical_use import CanonicalKineticCapture
    from gdd.engine.fixture_observations import observe_hamiltonian
    from gdd.engine.kinetic_activation import activation_errors
    from gdd.engine.orbital_activation import orbital_activation_errors
    from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
    from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian, canonical_signature

    hamiltonian = _build_ham_for_cell("QEPROJ:pgto+pw:Li_k025:ECP", True, {"j3": "mo"}, "SD")
    wavefunction = hamiltonian.wavefunction_data
    j3 = wavefunction.jastrow_data.jastrow_three_body_data
    composite = j3.orb_data.aos_data
    first, second = composite.children
    structure = second.structure_data
    changed_cell = replace(
        structure,
        vec_a=2 * structure.vec_a,
        vec_b=2 * structure.vec_b,
        vec_c=2 * structure.vec_c,
    )
    changed_orbitals = replace(
        j3.orb_data,
        aos_data=replace(composite, children=(first, replace(second, structure_data=changed_cell))),
    )
    changed_j3 = replace(j3, orb_data=changed_orbitals)
    changed_wavefunction = replace(
        wavefunction,
        jastrow_data=replace(wavefunction.jastrow_data, jastrow_three_body_data=changed_j3),
    )
    with pytest.raises(AssertionError, match=r"component boundary differs.*\.1"):
        observe_hamiltonian(replace(hamiltonian, wavefunction_data=changed_wavefunction))

    # A structurally correct ablation receipt is still vacuous when the
    # component has no numerical effect. Keep one real composite/Jastrow case
    # so deleting either difference>band validator turns this sentinel red.
    pattern_id = "WF-PATTERN-011"
    activity_hamiltonian = build_canonical_hamiltonian(pattern_id)
    activity_wavefunction = activity_hamiltonian.wavefunction_data
    geminal = activity_wavefunction.geminal_data
    rng = np.random.default_rng(864)
    r_up = rng.normal(scale=0.6, size=(geminal.num_electron_up, 3))
    r_dn = rng.normal(scale=0.6, size=(geminal.num_electron_dn, 3))
    capture = CanonicalKineticCapture(activity_hamiltonian, activation=True, orbital_activation=True)
    capture.evaluate(activity_wavefunction, r_up, r_dn)
    signature = canonical_signature(pattern_id)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "jastrow_grad_lap", "wf_kinetic"), "strict")
    record = capture.records[0]
    assert activation_errors(record, signature, atol=atol, rtol=rtol) == []
    assert orbital_activation_errors(record, signature, atol=atol, rtol=rtol) == []

    zero_jastrow_effect = copy.deepcopy(record)
    zero_jastrow_effect["activation"]["ablations"][0]["output"] = copy.deepcopy(zero_jastrow_effect["activation"]["baseline"])
    assert "no effect outside" in activation_errors(zero_jastrow_effect, signature, atol=atol, rtol=rtol)[0]

    zero_orbital_effect = copy.deepcopy(record)
    zero_orbital_effect["orbital_activation"]["ablations"][0]["output"] = copy.deepcopy(
        zero_orbital_effect["orbital_activation"]["baseline"]
    )
    assert "no finite effect outside" in orbital_activation_errors(zero_orbital_effect, signature, atol=atol, rtol=rtol)[0]
