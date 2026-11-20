"""Fixed-input native VMC composition: every sweep, measurement and aggregate.

Keep the original molecular ECP fixtures, two seeds and all iterations.
Compare same-input references, not independently sampled energy means.
The Cl2/N2 case names denote H2/cart and water/sphe fixtures;
they are not claims about which molecules were actually built.
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

pytestmark = pytest.mark.with_chdir

from jax import random  # noqa: E402
from jqmc._precision import get_tolerance_xla_rust_consistency, precision_policy_snapshot  # noqa: E402
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12  # noqa: E402
from jqmc.determinant import compute_geminal_all_elements  # noqa: E402

import jqmc_rs  # noqa: E402

from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests._vmc_fixed_input_oracle import state_vector, vmc_reference

_EL_ZONES = ("local_energy", "coulomb", "det_ratio", "ao_eval")
_STATE_ZONES = ("det_ratio", "det_eval", "ao_eval")
_INVERSE_ZONES = ("det_ratio", "det_eval")


def _charges(ham):
    cp = ham.coulomb_potential_data
    Z = np.asarray(ham.structure_data.atomic_numbers, dtype=np.float64)
    if cp.ecp_flag:
        Z = Z - np.asarray(cp.z_cores, dtype=np.float64)
    return Z


def _init_positions(ham, rng):
    gem = ham.wavefunction_data.geminal_data
    n_up, n_dn = gem.num_electron_up, gem.num_electron_dn
    sd = ham.structure_data
    if sd.pbc_flag:
        cell = np.asarray(sd.cell, dtype=np.float64)
        r_up = (rng.random((n_up, 3)) @ cell).astype(np.float64)
        r_dn = (rng.random((n_dn, 3)) @ cell).astype(np.float64)
    else:
        R = np.asarray(sd.positions, dtype=np.float64)
        r_up = (R[rng.integers(0, len(R), n_up)] + rng.normal(scale=0.5, size=(n_up, 3))).astype(np.float64)
        r_dn = (R[rng.integers(0, len(R), n_dn)] + rng.normal(scale=0.5, size=(n_dn, 3))).astype(np.float64)
    return r_up, r_dn


def native_vmc(ham, *, seed=0, dt=1.0, eps=0.0, moves_per_block=None, n_warmup=50, n_blocks=400):
    """Run a single-walker native VMC chain. Returns a dict with the energy
    estimate, its (autocorrelation-naive) standard error, acceptance rate, and
    the per-block local-energy samples."""
    gem = ham.wavefunction_data.geminal_data
    jas = ham.wavefunction_data.jastrow_data
    sd = ham.structure_data
    pos = np.asarray(sd.positions, dtype=np.float64)
    Z = _charges(ham)
    n_up, n_dn = gem.num_electron_up, gem.num_electron_dn
    if moves_per_block is None:
        moves_per_block = n_up + n_dn  # one "sweep" per block

    pbc = bool(sd.pbc_flag)
    cell_args = ()
    if pbc:
        cell_args = (np.asarray(sd.cell, dtype=np.float64), int(sd._n_max))

    w = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
    gp = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)  # RT = I

    rng = np.random.default_rng(seed)
    r_up, r_dn = _init_positions(ham, rng)
    G = np.asarray(compute_geminal_all_elements(gem, r_up, r_dn), dtype=np.float64)
    # Seed from NUMPY's inverse, as production does (the driver is
    # handed a JAX-computed G^-1), rather than from the Rust kernel under test.
    G_inv = np.asarray(np.linalg.inv(G), dtype=np.float64)
    key = np.asarray(random.PRNGKey(seed), dtype=np.uint32)

    energies = []
    total_acc = 0
    total_moves = 0
    for blk in range(n_warmup + n_blocks):
        normals = rng.normal(size=moves_per_block).astype(np.float64)
        res = jqmc_rs._mcmc_sweep(
            gem,
            jas,
            pos,
            Z,
            r_up,
            r_dn,
            G,
            G_inv,
            dt,
            eps,
            int(key[0]),
            int(key[1]),
            normals,
            *cell_args,
        )
        r_up = np.asarray(res["r_up"], dtype=np.float64)
        r_dn = np.asarray(res["r_dn"], dtype=np.float64)
        G = np.asarray(res["geminal"], dtype=np.float64)
        G_inv = np.asarray(res["geminal_inv"], dtype=np.float64)
        key = np.asarray(res["final_key"], dtype=np.uint32)
        total_acc += int(res["accepted"])
        total_moves += moves_per_block
        if blk >= n_warmup:
            eL = float(jqmc_rs.compute_local_energy(ham, r_up, r_dn, w, gp))
            energies.append(eL)

    energies = np.asarray(energies, dtype=np.float64)
    mean = float(energies.mean())
    # This is the declared finite-sample formula, not a convergence criterion.
    stderr = float(energies.std(ddof=1) / np.sqrt(len(energies))) if len(energies) > 1 else float("inf")
    return {
        "energy": mean,
        "stderr": stderr,
        "accept_rate": total_acc / total_moves,
        "samples": energies,
        "final_r_up": r_up,
        "final_r_dn": r_dn,
    }


# Preserve the original committed ECP + J12 Cartesian/spherical GTO inputs.
SYSTEMS = {
    "Cl2": "mol_ecp_cart_j12",
    "N2": "mol_ecp_sphe_j12",
}


@pytest.mark.heavy
@pytest.mark.parametrize("name", list(SYSTEMS))
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("dt,eps,warmup,measurements", [(1.5, 0.0, 120, 700)])
def test_native_vmc_molecular(monkeypatch, name, seed, dt, eps, warmup, measurements):
    """Every VMC sweep and measurement matches a deterministic same-input reference.

    Cl2 means committed tests/trexio_example_files/H2_ecp_ccpvdz_cart.h5
    (Cartesian GTO, one up and one down electron); N2 means committed
    tests/trexio_example_files/water_ccecp_ccpvqz.h5 (spherical GTO,
    four up and four down electrons). Both are molecular ECP SD/J12, closed-shell.
    PCG64 seeds 0/1 select initial nuclei and scale .5 displacements, with
    no cond(G) selection, then inject the original Gaussian moves. Keep
    dt=1.5, eps=0, 120 warmup, all 700 measurements and Ne moves per sweep.
    Check initial inputs, every handoff and every measurement position;
    independently replay each sweep from its actual input, checking exact
    accept/key decisions and numerical coordinates. Rebuild G independently at
    every endpoint; separately compare the saved inverse with a fresh NumPy
    inversion of the same saved G, using the fp64 determinant zones in both
    modes. Both checks are required: a consistently wrong G must fail the first,
    and a drifting inverse must fail the second. Evaluate
    XLA local energy at every measured configuration, then check the whole
    sample vector, mean, finite-sample SE, counted acceptance rate and final
    coordinates separately. Use existing backend-medium pairs within each
    precision. Neither a sigma band nor ground-state convergence is a verdict.
    """
    assert jqmc_rs.native_precision_support() == precision_policy_snapshot(), "native and XLA precision policies differ"
    ham = rs_ham(SYSTEMS[name])
    assert ham.coulomb_potential_data.ecp_flag and not ham.structure_data.pbc_flag
    sweeps, evaluations = [], []
    original_sweep, original_energy = jqmc_rs._mcmc_sweep, jqmc_rs.compute_local_energy

    def record_sweep(*args):
        inputs = (*args[:2], *(np.array(value, copy=True) for value in args[2:]))
        output = original_sweep(*args)
        sweeps.append((inputs, {key: np.array(value, copy=True) for key, value in output.items()}))
        return output

    def record_energy(actual_ham, up, down, weights, grid):
        evaluations.append((len(sweeps) - 1, actual_ham, *(np.array(value, copy=True) for value in (up, down, weights, grid))))
        return original_energy(actual_ham, up, down, weights, grid)

    monkeypatch.setattr(jqmc_rs, "_mcmc_sweep", record_sweep)
    monkeypatch.setattr(jqmc_rs, "compute_local_energy", record_energy)
    result = native_vmc(ham, seed=seed, dt=dt, eps=eps, n_warmup=warmup, n_blocks=measurements)
    assert len(sweeps) == warmup + measurements and len(evaluations) == measurements
    reference = vmc_reference(
        ham, seed=seed, dt=dt, eps=eps, warmup=warmup, measurements=measurements, sweeps=sweeps, evaluations=evaluations
    )
    actual_states = np.asarray([state_vector(tuple(row[key] for key in ("r_up", "r_dn", "geminal"))) for _, row in sweeps])
    actual_inverses = np.asarray([row["geminal_inv"] for _, row in sweeps])
    actual_decisions = np.asarray([(int(row["accepted"]), *map(int, row["final_key"])) for _, row in sweeps], dtype=np.uint64)
    final_positions = np.concatenate([result["final_r_up"].ravel(), result["final_r_dn"].ravel()])
    assert all(
        np.all(np.isfinite(value))
        for value in (
            actual_states,
            actual_inverses,
            result["samples"],
            reference["states"],
            reference["inverses"],
            reference["energies"],
        )
    )
    state_atol, state_rtol = get_tolerance_xla_rust_consistency(_STATE_ZONES, level="medium")
    inverse_atol, inverse_rtol = get_tolerance_xla_rust_consistency(_INVERSE_ZONES, level="medium")
    atol, rtol = get_tolerance_xla_rust_consistency(_EL_ZONES, level="medium")
    np.testing.assert_array_equal(reference["wiring"], np.ones_like(reference["wiring"]))
    np.testing.assert_array_equal(actual_decisions, reference["decisions"])
    np.testing.assert_allclose(actual_states, reference["states"], atol=state_atol, rtol=state_rtol)
    np.testing.assert_allclose(result["samples"], reference["energies"], atol=atol, rtol=rtol)
    np.testing.assert_allclose(result["energy"], reference["mean"], atol=atol, rtol=rtol)
    np.testing.assert_allclose(result["stderr"], reference["stderr"], atol=atol, rtol=rtol)
    np.testing.assert_array_equal(result["accept_rate"], reference["accept_rate"])
    np.testing.assert_allclose(final_positions, reference["final_positions"], atol=state_atol, rtol=state_rtol)
    np.testing.assert_allclose(actual_inverses, reference["inverses"], atol=inverse_atol, rtol=inverse_rtol)
