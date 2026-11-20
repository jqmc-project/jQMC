"""Fixed-input production GFMC interval batches across actual native pool widths."""

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

import multiprocessing
import os
import traceback

import jax
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min, precision_policy_snapshot
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_gfmc_projection import _cfg

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]

_CASES = ["mol_ae_sphe", "mol_ecp_tmove", "mol_ecp_dltmove", "pbc_gamma", "twist_j12_tmove", "twist_j12_dltmove"]
# Keep the source identity selector-local for GDD case provenance. These are
# the same recipes/seeds as the existing interval fixtures; _cfg owns positions.
_CELLS = {
    "mol_ae_sphe": {"pattern": "mol_sphe_sd_jnone", "seed": 1, "dltmove": False},
    "mol_ecp_tmove": {"pattern": "mol_ecp_cart_j12", "seed": 3, "dltmove": False},
    "mol_ecp_dltmove": {"pattern": "mol_ecp_cart_j12", "seed": 3, "dltmove": True},
    "pbc_gamma": {"pattern": "pbc_gamma_sphepgto_j12", "seed": 2, "dltmove": False},
    "twist_j12_tmove": {"pattern": "twist_sphepgto_j12", "seed": 4, "dltmove": False},
    "twist_j12_dltmove": {"pattern": "twist_sphepgto_j12", "seed": 4, "dltmove": True},
}
_ZONES = (
    "ao_eval",
    "det_eval",
    "det_ratio",
    "wf_eval",
    "wf_kinetic",
    "coulomb",
    "local_energy",
    "jastrow_eval",
    "jastrow_grad_lap",
    "ewald",
)
_DISCRETE_FIELDS = {"final_key", "n_substeps", "n_pot_updates"}
_NUMERICAL_FIELDS = ["e_L", "w_L", "r_up", "r_dn", "hf_dR", "hf_dr_up", "hf_dr_dn", "pp_dR", "pp_dr_up", "pp_dr_dn"]


def _assert_width(actual, requested):
    """Reject a process that silently retained a different global Rayon pool."""
    assert actual == requested, ("requested/live native pool width", requested, actual)


def _assert_fields(actual, expected, fields):
    """Require the complete API result, not just the intersection of two outputs."""
    assert set(actual) == set(expected) == set(fields), (set(actual), set(expected), set(fields))


def _assert_numerical(actual, expected):
    """Use the same central policy for detector controls and production comparison."""
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)


def _assert_exact_keys(actual, expected):
    """Keep the expected failing RNG comparison in its own invocation."""
    np.testing.assert_array_equal(actual, expected)


def _interval_worker(connection, width, ham, up, down, keys, complex_, dltmove, policy):
    """Run four real native interval calls on a fresh, independently sized pool."""
    try:
        import jqmc_rs

        os.environ["JQMC_RS_NUM_THREADS"] = str(width)
        jqmc_rs.configure_threads()
        live_width = jqmc_rs.current_num_threads()
        jqmc_rs.configure_native_precision(policy)
        wf = ham.wavefunction_data
        periodic = bool(ham.structure_data.pbc_flag)
        ctx = jqmc_rs.build_obs_ctx(
            wf.geminal_data,
            wf.jastrow_data,
            ham.coulomb_potential_data,
            complex_,
            periodic,
            periodic,
            want_force=True,
            want_gfmc=True,
            precision_policy=policy,
        )
        weights = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
        points = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)
        outputs = {}
        for driver in ("t", "n"):
            method = getattr(ctx, f"run_gfmc_projection_{driver}{'_complex' if complex_ else ''}_batch")
            for random_mesh in (False, True):
                if driver == "t":
                    args = (up, down, 0.30, 0.20, keys, weights, points)
                else:
                    args = (up, down, 0.30, -1.0, 3, keys, np.empty((0, 0, 3, 3)), weights, points)
                result = method(
                    *args,
                    random_mesh=random_mesh,
                    dltmove=dltmove,
                    nn=1,
                    want_force=True,
                    want_stress=periodic,
                )
                outputs[driver, random_mesh] = {key: np.asarray(value) for key, value in result.items()}
        connection.send(("ok", live_width, outputs))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _evaluate_intervals(width, ham, up, down, keys, complex_, dltmove, policy):
    """Transport native results; make all pass/fail decisions in the parent process."""
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_interval_worker, args=(child, width, ham, up, down, keys, complex_, dltmove, policy))
    process.start()
    child.close()
    try:
        assert parent.poll(180), "native interval thread comparison timed out"
        message = parent.recv()
        process.join(30)
        assert process.exitcode == 0 and message[0] == "ok", message
        _assert_width(message[1], width)
        return message[2]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(30)
        parent.close()


def _interval_inputs(cell):
    """Reuse committed interval fixtures, without conditioning or acceptance filters.

    Existing _cell seeds are 1 (H2 AE), 3 (H2 ECP) and 2 (Gamma Li).
    Non-Gamma Li uses _cfg seed 4. Five explicit offsets preserve walker
    identity; uint32 keys (0,7+i) supply independent deterministic streams.
    The active-mode Hamiltonian is frozen once before any child is spawned.
    """
    specification = _CELLS[cell]
    complex_ = cell.startswith("twist_")
    ham = rs_ham(specification["pattern"])
    up0, down0 = _cfg(ham, seed=specification["seed"])
    up = np.stack([up0 + np.array([0.03 * i, -0.02 * i, 0.04 * i]) for i in range(5)])
    down = np.stack([down0 + np.array([-0.01 * i, 0.04 * i, 0.02 * i]) for i in range(5)])
    keys = np.array([[0, 7 + i] for i in range(5)], dtype=np.uint32)
    frozen = jax.tree_util.tree_map(lambda value: np.asarray(value) if isinstance(value, jax.Array) else value, ham)
    return frozen, up, down, keys, complex_, specification["dltmove"]


def _compare_interval_batches(actual, reference, periodic, r_up, r_dn, keys):
    """Compare every ordered output; reject omitted fields, permutation and RNG drift."""
    scenarios = {(driver, mesh) for driver in ("t", "n") for mesh in (False, True)}
    assert set(actual) == set(reference) == scenarios
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    for driver, random_mesh in sorted(scenarios):
        got, expected = actual[driver, random_mesh], reference[driver, random_mesh]
        numerical_fields = list(_NUMERICAL_FIELDS)
        if driver == "n":
            numerical_fields.append("V_diag")
        required = set(numerical_fields) | _DISCRETE_FIELDS
        if periodic:
            required.update(("stress_HF", "stress_PP"))
        _assert_fields(got, expected, required)
        for field in required:
            assert got[field].shape == expected[field].shape and got[field].shape[0] == 5
            assert got[field].dtype == expected[field].dtype and got[field].size
            assert np.isfinite(got[field]).all() and np.isfinite(expected[field]).all(), field
        for field in _DISCRETE_FIELDS:
            assert np.issubdtype(expected[field].dtype, np.integer)
            np.testing.assert_array_equal(got[field], expected[field], err_msg=field)
        assert expected["final_key"].dtype == np.uint32
        assert expected["final_key"].shape == keys.shape
        assert not np.array_equal(expected["final_key"], keys)
        for field in numerical_fields:
            actual_value, expected_value = got[field], expected[field]
            np.testing.assert_allclose(actual_value, expected_value, atol=atol, rtol=rtol, equal_nan=False, err_msg=field)
        if periodic:
            for field in ("stress_HF", "stress_PP"):
                actual_value, expected_value = got[field], expected[field]
                np.testing.assert_allclose(actual_value, expected_value, atol=atol, rtol=rtol, equal_nan=False, err_msg=field)
        if driver == "n":
            np.testing.assert_array_equal(expected["n_substeps"], np.full(5, 3))
        else:
            assert np.all(expected["n_substeps"] > 1)
            assert np.unique(expected["n_substeps"]).size > 1
        assert not np.array_equal(expected["r_up"], r_up) or not np.array_equal(expected["r_dn"], r_dn)
        with pytest.raises(AssertionError):
            _assert_fields({key: value for key, value in got.items() if key != "e_L"}, expected, required)
        with pytest.raises(AssertionError):
            _assert_numerical(got["e_L"][::-1], expected["e_L"])
        changed_key = got["final_key"].copy()
        changed_key[0, 0] ^= np.uint32(1)
        with pytest.raises(AssertionError):
            _assert_exact_keys(changed_key, expected["final_key"])


@pytest.mark.parametrize("cell", _CASES)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_native_intervals_match_across_thread_counts(cell):
    """Actual 1/2/4-thread pools preserve five ordered GFMC_t/n walkers in each mode.

    Committed H2 AE spherical/no-J, H2 ECP Cartesian/J1+J2 exp (tmove and
    dltmove), and Gamma/non-Gamma Li pGTO ECP SD/MO fixtures exercise the real
    and complex batch APIs. Li uses J1+J2 exp; non-Gamma covers both ECP move
    modes. Seeds 1/3/2/4 and five explicit linear offsets define positions,
    without rejection. Every child receives the same frozen Hamiltonian,
    coordinates, uint32 keys (0,7+i), and native precision-policy snapshot.
    With alat=.30, tau=.20 or nsub=3/E_scf=-1, the native 12-point ECP mesh
    is both fixed and random; empty injected rotations select native caches.
    Compare all coordinates, energy/weight, HF/Pulay forces, periodic stresses
    and n-only V_diag at same-mode central strict tolerance, finite and in
    original walker order. RNG and integer counters are exact. Reject live
    width mismatch, missing fields, reversed energies, and a flipped RNG bit.
    Full and mixed are independent required lanes; no cross-backend RNG or
    physical-oracle claim is made and no floating-point bit contract is added.
    """
    ham, r_up, r_dn, keys, complex_, dltmove = _interval_inputs(cell)
    policy = precision_policy_snapshot()
    reference = _evaluate_intervals(1, ham, r_up, r_dn, keys, complex_, dltmove, policy)
    with pytest.raises(AssertionError):
        _assert_width(1, 4)
    for width in (2, 4):
        actual = _evaluate_intervals(width, ham, r_up, r_dn, keys, complex_, dltmove, policy)
        _compare_interval_batches(actual, reference, bool(ham.structure_data.pbc_flag), r_up, r_dn, keys)
