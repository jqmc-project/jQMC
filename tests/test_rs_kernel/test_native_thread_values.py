"""Fixed-input native observable batches across independent Rayon pool widths."""

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
from jqmc.determinant import compute_geminal_all_elements
from tests.test_jqmc_kernel.test_mcmc_acceptance_reference import _input

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]


def _worker(connection, width, ham, up, down, inverse, policy):
    try:
        import jqmc_rs

        os.environ["JQMC_RS_NUM_THREADS"] = str(width)
        jqmc_rs.configure_threads()
        live_width = jqmc_rs.current_num_threads()
        # The parent compares this actual native width with its request;
        # keep the verdict outside this worker's exception-transport handler.
        jqmc_rs.configure_native_precision(policy)
        gd, jd = ham.wavefunction_data.geminal_data, ham.wavefunction_data.jastrow_data
        cp = ham.coulomb_potential_data
        complex_, pbc = bool(np.iscomplexobj(inverse)), bool(ham.structure_data.pbc_flag)
        ctx = jqmc_rs.build_obs_ctx(gd, jd, cp, complex_, pbc, pbc, precision_policy=policy)
        weights = np.asarray(icosahedron_sym_mesh_Nv12.weights)
        points = np.tile(np.asarray(icosahedron_sym_mesh_Nv12.grid_points)[None], (len(up), 1, 1))
        energy = getattr(ctx, "e_L_complex_batch" if complex_ else "e_L_batch")(up, down, weights, points, inverse)
        if pbc:
            force = getattr(ctx, "grad_e_L_pbc_complex_batch" if complex_ else "grad_e_L_pbc_batch")(
                up, down, weights, points, inverse
            )
            pulay = getattr(ctx, "grad_ln_psi_pbc_complex_batch" if complex_ else "grad_ln_psi_pbc_batch")(up, down, inverse)
        else:
            force = (
                ctx.grad_e_L_ecp_batch(up, down, weights, points, inverse)
                if cp.ecp_flag
                else ctx.grad_e_L_ae_batch(up, down, inverse)
            )
            pulay = ctx.grad_ln_psi_batch(up, down, inverse)
        outputs = {"energy": np.asarray(energy)}
        for label, arrays in (("force", force), ("pulay", pulay)):
            for coordinate, array in zip(("R", "up", "down"), arrays, strict=True):
                outputs[label + "-" + coordinate] = np.asarray(array)
        if pbc:
            outputs["stress"] = np.asarray(
                getattr(ctx, "grad_e_L_eps_pbc_complex_batch" if complex_ else "grad_e_L_eps_pbc_batch")(
                    up, down, weights, points, inverse
                )
            )
            outputs["stress-pulay"] = np.asarray(
                getattr(ctx, "grad_ln_psi_eps_pbc_complex_batch" if complex_ else "grad_ln_psi_eps_pbc_batch")(
                    up, down, inverse
                )
            )
        connection.send(("ok", live_width, outputs))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _evaluate(width, ham, up, down, inverse):
    ham = jax.tree_util.tree_map(lambda leaf: np.asarray(leaf) if isinstance(leaf, jax.Array) else leaf, ham)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(child, width, ham, up, down, inverse, precision_policy_snapshot()))
    process.start()
    child.close()
    try:
        assert parent.poll(180), "native thread comparison timed out"
        message = parent.recv()
        process.join(30)
        assert process.exitcode == 0 and message[0] == "ok", message
        assert message[1] == width, "the requested native pool was not used"
        return message[2]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(30)
        parent.close()


def _inputs(filename):
    ham, state = _input(filename)
    up = np.stack([np.asarray(state[0]) + np.array([0.03 * i, -0.02 * i, 0.04 * i]) for i in range(5)])
    down = np.stack([np.asarray(state[1]) + np.array([-0.01 * i, 0.04 * i, 0.02 * i]) for i in range(5)])
    matrices = np.stack(
        [np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, u, d)) for u, d in zip(up, down)]
    )
    return ham, up, down, np.linalg.inv(matrices)


def _assert_observable(actual, expected, field):
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "wf_eval", "wf_kinetic", "coulomb", "jastrow_eval"), "strict")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)


def _compare_thread_batches(filename, periodic):
    """Actual 1/4-thread native pools preserve each fixed walker's energy, force and stress.

    H2 AE and water ECP are molecular; Gamma and non-Gamma Li ECP exercise
    periodic real/complex dispatch. Each uses SD/MO and J2 exp b=0.5. Five
    explicit linear offsets of PCG64 seed-2025 normal*0.5 positions fix all
    inputs without cond(G) filtering. An unrotated 12-point ECP mesh and
    identical dense inverses feed both native contexts. Separate spawned
    processes configure JQMC_RS_NUM_THREADS=1/4 and verify the live pool width.
    Every walker retains its original order. Compare energy and all nuclear,
    up/down HF and Pulay derivatives, plus both periodic stress tensors,
    at same-mode central strict tolerances. This is thread-count parity,
    not an independent physical oracle or statistical trajectory comparison.
    """
    ham, up, down, inverse = _inputs(filename)
    assert bool(ham.structure_data.pbc_flag) == periodic
    reference = _evaluate(1, ham, up, down, inverse)
    actual = _evaluate(4, ham, up, down, inverse)
    fields = ["energy", "force-R", "force-up", "force-down", "pulay-R", "pulay-up", "pulay-down"]
    if periodic:
        fields += ["stress", "stress-pulay"]
    assert set(reference) == set(actual) == set(fields)
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "wf_eval", "wf_kinetic", "coulomb", "jastrow_eval"), "strict")
    for field in fields:
        assert actual[field].shape == reference[field].shape and actual[field].shape[0] == 5
        assert actual[field].size and np.isfinite(reference[field]).all()
        actual_value, expected_value = actual[field], reference[field]
        np.testing.assert_allclose(actual_value, expected_value, atol=atol, rtol=rtol, equal_nan=False)
    assert np.any(reference["energy"] != reference["energy"][0]), "walker permutation needs nonconstant energies"
    with pytest.raises(AssertionError):
        _assert_observable(actual["energy"][::-1], reference["energy"], "fault-walker-permutation")


@pytest.mark.parametrize("filename", ["H2_ae_ccpvdz_cart.h5", "water_ccecp_ccpvqz.h5"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_molecular_native_observables_match_across_thread_counts(filename):
    """Native 1/4-thread pools agree on all five walkers' molecular energy and HF/Pulay forces.

    Committed H2 AE Cartesian (1up/1down) and water ECP spherical (4up/4down)
    use SD/MO and J2 exp b=0.5. PCG64 seed 2025 normal*0.5 positions plus
    five deterministic linear offsets define the input, with no cond(G)
    filtering. Each spawned process uses the same rounded Hamiltonian,
    positions, dense inverse, unrotated 12-point ECP mesh and active precision
    policy, then verifies its actual Rayon width. Same-mode central strict
    tolerances compare energy and every R/up/down HF/Pulay component. Reversing
    walker order must fail. This is numerical thread parity, not physics accuracy.
    """
    _compare_thread_batches(filename, False)


@pytest.mark.parametrize("filename", ["lithium.h5", "lithium_k025.h5"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_periodic_native_observables_match_across_thread_counts(filename):
    """Native 1/4-thread pools agree on periodic energy, HF/Pulay forces and both stress tensors.

    Committed Gamma/non-Gamma Li ECP spherical pGTO SD/MO has 1up/1down and
    J2 exp b=0.5. PCG64 seed 2025 normal*0.5 positions plus five deterministic
    linear offsets define the input, with no cond(G) filtering. Both spawned
    native contexts receive identical rounded inputs, dense inverses and an
    unrotated 12-point ECP mesh in the active precision mode; live Rayon pool
    widths must be 1/4. Compare all five walkers in order, including complex
    energy and every stress component, at same-mode central strict tolerances.
    Reversing walker order must fail. This is thread parity, not a physics oracle.
    """
    _compare_thread_batches(filename, True)
