"""GFMC_t measurement-step force: same-input check against the public derivative entry points.

A non-altering spy records the exact state the production ``GFMC_t`` driver
holds when it draws its branching random number at each measurement step:
the walker positions, the interval mesh rotation and the per-walker raw force
leaves it has just evaluated.  One walker is used (multi-walker GFMC_t RNG
parity is not covered here), so every stored force buffer is
that walker's per-walker force.

At each captured input the node recomputes, with fresh inverses,

* the raw leaves with ``jqmc.derivatives.gfmc.grad_e_L_t_wrt_positions`` and
  ``jqmc.derivatives.compute_grad_ln_Psi(analytic=True)``;
* the no-SWCT and SWCT stored forces from those leaves, with the SWCT weights
  and weight gradients taken from ``jqmc.swct.evaluate_swct_omega`` /
  ``evaluate_swct_domega`` and contracted explicitly (the driver uses the fused
  ``evaluate_swct_omega_dot_batched`` route);
* the SWCT correction, compared with the difference of the stored forces of
  two runs that differ only in ``use_swct`` (their trajectories must be
  bit-identical, since SWCT does not feed back into the walk).

The derivative formulas themselves are certified by their own derivative
guarantees; this node checks the driver's inputs, cached inverses, SWCT
contraction and storage at a real measurement step.
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

from jqmc import derivatives
from jqmc import jqmc_gfmc as implementation
from jqmc._precision import get_tolerance_min
from jqmc._setting import ATOL_REDUCTION_ORDER, RTOL_REDUCTION_ORDER
from jqmc.derivatives import gfmc as gfmc_derivatives
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_three_body_data, Jastrow_two_body_data
from jqmc.jqmc_gfmc import GFMC_t
from jqmc.swct import evaluate_swct_domega, evaluate_swct_omega
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._ecp_core_electrons import core_electrons_of

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_TREXIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")

# Leaves the driver holds when it draws the branching number of a measurement step.
_CAPTURED = (
    "latest_RTs",
    "grad_e_L_R",
    "grad_e_L_r_up",
    "grad_e_L_r_dn",
    "grad_ln_Psi_dR",
    "grad_ln_Psi_r_up",
    "grad_ln_Psi_r_dn",
    "e_L_list",
    "w_L_list",
)
_STORED = ("_stored_force_HF", "_stored_force_PP", "_stored_E_L_force_PP", "_stored_e_L", "_stored_w_L")
_FORCE_ZONES = ("ao_eval", "ao_grad_lap", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic")


def _hamiltonian(trexio_file: str) -> Hamiltonian_data:
    structure, _aos, mos, _mos_dn, geminal, coulomb = read_trexio_file(
        trexio_file=os.path.join(_TREXIO_DIR, trexio_file), store_tuple=True
    )
    jastrow = Jastrow_data(
        jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure,
            core_electrons=core_electrons_of(coulomb, len(structure.atomic_numbers)),
            jastrow_1b_type="exp",
        ),
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure
        ),
        jastrow_three_body_data=Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=mos, random_init=True, random_scale=1.0e-3, seed=123
        ),
    )
    wavefunction = Wavefunction_data(jastrow_data=jastrow, geminal_data=geminal)
    wavefunction.sanity_check()
    hamiltonian = Hamiltonian_data(structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=wavefunction)
    hamiltonian.sanity_check()
    return hamiltonian


class _BranchingDrawSpy:
    """Delegate every call to the driver's own RandomState; record the caller's state on ``random()``.

    ``GFMC_t.run`` draws exactly one branching number per measurement step, after
    the observables and before the force assembly.  Reading the caller frame does
    not change any value; the test separately requires the observed driver to be
    bit-identical to an unobserved one.
    """

    def __init__(self, driver: GFMC_t):
        self._driver = driver
        self._rng = driver._branching_rng
        self.records: list[dict[str, np.ndarray]] = []

    def random(self, *args, **kwargs):
        frame = sys._getframe(1)
        if frame.f_code is not implementation.GFMC_t.run.__code__:
            raise AssertionError("branching draw outside GFMC_t.run")
        local = frame.f_locals
        if local["self"] is not self._driver:
            raise AssertionError("branching draw from another driver")
        row = {name: np.array(local[name], copy=True) for name in _CAPTURED}
        row["r_up"] = np.array(self._driver._latest_r_up_carts, copy=True)
        row["r_dn"] = np.array(self._driver._latest_r_dn_carts, copy=True)
        row["alat"] = float(self._driver._alat)
        row["non_local_move"] = self._driver._non_local_move
        self.records.append(row)
        return self._rng.random(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._rng, name)


def _driver_buffers(driver: GFMC_t) -> dict[str, np.ndarray]:
    names = [name for name in vars(driver) if name.startswith("_stored_")]
    names += ["_latest_r_up_carts", "_latest_r_dn_carts", "_jax_PRNG_key_list", "_mcmc_counter"]
    return {name: np.array(getattr(driver, name), copy=True) for name in names}


def _recompute(hamiltonian: Hamiltonian_data, row: dict) -> dict[str, np.ndarray]:
    """Raw leaves at the captured input from the public entry points (fresh inverses)."""
    up, dn, rt = jnp.asarray(row["r_up"][0]), jnp.asarray(row["r_dn"][0]), jnp.asarray(row["latest_RTs"][0])
    hf_R, hf_up, hf_dn = gfmc_derivatives.grad_e_L_t_wrt_positions(
        hamiltonian, up, dn, rt, row["alat"], non_local_move=row["non_local_move"]
    )
    score = derivatives.compute_grad_ln_Psi(hamiltonian, up, dn, {"opt_r": True, "opt_R": True}, analytic=True)
    structure = hamiltonian.structure_data
    omega_up, omega_dn = evaluate_swct_omega(structure, up), evaluate_swct_omega(structure, dn)
    domega = evaluate_swct_domega(structure, up) + evaluate_swct_domega(structure, dn)
    correction_hf = omega_up @ hf_up + omega_dn @ hf_dn
    correction_pp = omega_up @ score["r"][0] + omega_dn @ score["r"][1] + 0.5 * domega
    return {
        name: np.asarray(value)
        for name, value in {
            "grad_e_L_R": hf_R,
            "grad_e_L_r_up": hf_up,
            "grad_e_L_r_dn": hf_dn,
            "grad_ln_Psi_dR": score["R"],
            "grad_ln_Psi_r_up": score["r"][0],
            "grad_ln_Psi_r_dn": score["r"][1],
            "correction_hf": correction_hf,
            "correction_pp": correction_pp,
        }.items()
    }


def _assert_close(actual, reference, atol, rtol, label):
    # Every caller passes the reviewed strict force budget; verify the incoming
    # pair instead of trusting it, then compare with the same local resolver.
    expected_tolerance = get_tolerance_min(_FORCE_ZONES, "strict")
    assert (atol, rtol) == expected_tolerance, f"{label}: unexpected comparison budget"
    atol, rtol = get_tolerance_min(_FORCE_ZONES, "strict")
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert np.all(np.isfinite(actual)), f"{label}: non-finite production value"
    assert np.all(np.isfinite(reference)), f"{label}: non-finite reference value"
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=label)


@pytest.mark.parametrize(
    "trexio_file,non_local_move",
    [
        pytest.param("H2_ae_ccpvdz_cart.h5", "tmove", id="h2-ae-molecular", marks=pytest.mark.light),
        pytest.param("N2_ecp_ccpvtz_cart.h5", "tmove", id="n2-ecp-molecular", marks=pytest.mark.heavy),
        pytest.param("N2_ecp_ccpvtz_cart.h5", "dltmove", id="n2-ecp-molecular-dltmove", marks=pytest.mark.heavy),
        pytest.param("lithium.h5", "tmove", id="li-ecp-periodic", marks=pytest.mark.heavy),
    ],
)
def test_gfmc_t_measurement_force_matches_public_entry_points(trexio_file, non_local_move):
    """Stored GFMC_t forces (with and without SWCT) equal their same-input public-entry reassembly."""
    hamiltonian = _hamiltonian(trexio_file)
    num_steps = 3
    settings = dict(
        hamiltonian_data=hamiltonian,
        num_walkers=1,
        num_gfmc_collect_steps=1,
        mcmc_seed=3446,
        tau=0.1,
        alat=0.30,
        random_discretized_mesh=True,
        non_local_move=non_local_move,
        comput_position_deriv=True,
    )
    assert implementation._ANALYTIC_GRAD_ENABLED, "the declared driver force backend is the analytic one"

    baseline = GFMC_t(**settings, use_swct=True)
    baseline.run(num_mcmc_steps=num_steps)

    observed = {}
    for use_swct in (True, False):
        driver = GFMC_t(**settings, use_swct=use_swct)
        spy = _BranchingDrawSpy(driver)
        driver._branching_rng = spy
        driver.run(num_mcmc_steps=num_steps)
        driver._branching_rng = spy._rng
        assert len(spy.records) == num_steps, "one branching draw per measurement step"
        observed[use_swct] = (driver, spy.records)

    # Observation does not change the real driver. The force accumulators are compared with
    # the reduction-order pair: on GPU the ECP force adds with XLA scatter-add atomics in a
    # run-dependent order, so two runs differ in the last bits (#529). All else is bitwise.
    swct_driver, swct_records = observed[True]
    plain_driver, plain_records = observed[False]
    before, after = _driver_buffers(baseline), _driver_buffers(swct_driver)
    assert before.keys() == after.keys()
    for name, value in before.items():
        if "force" in name:
            np.testing.assert_allclose(
                after[name], value, atol=ATOL_REDUCTION_ORDER, rtol=RTOL_REDUCTION_ORDER, err_msg=f"spy changed {name}"
            )
        else:
            np.testing.assert_array_equal(after[name], value, err_msg=f"spy changed {name}")

    # SWCT does not feed back into the walk: same positions, keys, weights and energies.
    for name in ("_latest_r_up_carts", "_latest_r_dn_carts", "_jax_PRNG_key_list", "_stored_e_L", "_stored_w_L"):
        np.testing.assert_array_equal(getattr(plain_driver, name), getattr(swct_driver, name), err_msg=name)
    # The captured force leaves (_CAPTURED[1:7]) carry the same ECP scatter-add order as the
    # force accumulators above; positions, the mesh rotation, e_L and w_L stay bitwise.
    for plain_row, swct_row in zip(plain_records, swct_records, strict=True):
        for name in ("r_up", "r_dn", "latest_RTs", *_CAPTURED[7:]):
            np.testing.assert_array_equal(plain_row[name], swct_row[name], err_msg=f"trajectory {name}")
        for name in _CAPTURED[1:7]:
            np.testing.assert_allclose(
                plain_row[name],
                swct_row[name],
                atol=ATOL_REDUCTION_ORDER,
                rtol=RTOL_REDUCTION_ORDER,
                err_msg=f"trajectory {name}",
            )

    atol, rtol = get_tolerance_min(_FORCE_ZONES, "strict")
    stored = {flag: {name: np.asarray(getattr(driver, name)) for name in _STORED} for flag, (driver, _) in observed.items()}
    for flag in (True, False):
        for name in _STORED:
            assert len(stored[flag][name]) == num_steps, f"{name}: one stored row per measurement step"

    for step, row in enumerate(swct_records):
        reference = _recompute(hamiltonian, row)
        for name in _CAPTURED[1:7]:
            _assert_close(row[name][0], reference[name], atol, rtol, f"{trexio_file} step {step} raw leaf {name}")
        e_L = float(np.real(row["e_L_list"][0]))
        hf = reference["grad_e_L_R"]
        pp = reference["grad_ln_Psi_dR"]
        hf_swct = hf + reference["correction_hf"]
        pp_swct = pp + reference["correction_pp"]
        _assert_close(stored[False]["_stored_force_HF"][step, 0], hf, atol, rtol, f"{trexio_file} step {step} no-SWCT force_HF")
        _assert_close(stored[False]["_stored_force_PP"][step, 0], pp, atol, rtol, f"{trexio_file} step {step} no-SWCT force_PP")
        _assert_close(
            stored[False]["_stored_E_L_force_PP"][step, 0], e_L * pp, atol, rtol, f"{trexio_file} step {step} no-SWCT E_L*PP"
        )
        _assert_close(
            stored[True]["_stored_force_HF"][step, 0], hf_swct, atol, rtol, f"{trexio_file} step {step} SWCT force_HF"
        )
        _assert_close(
            stored[True]["_stored_force_PP"][step, 0], pp_swct, atol, rtol, f"{trexio_file} step {step} SWCT force_PP"
        )
        _assert_close(
            stored[True]["_stored_E_L_force_PP"][step, 0], e_L * pp_swct, atol, rtol, f"{trexio_file} step {step} SWCT E_L*PP"
        )
        # The stored SWCT correction (difference of the two otherwise identical runs).
        for key, name in (("correction_hf", "_stored_force_HF"), ("correction_pp", "_stored_force_PP")):
            difference = stored[True][name][step, 0] - stored[False][name][step, 0]
            _assert_close(difference, reference[key], atol, rtol, f"{trexio_file} step {step} stored SWCT {key}")
            # Non-vacuity: the correction is not within the comparison tolerance of zero.
            scale = atol + rtol * np.abs(reference[key])
            assert np.max(np.abs(reference[key]) - scale) > 0.0, f"{trexio_file} step {step} {key} is vacuous"
        for name in ("grad_e_L_R", "grad_ln_Psi_dR"):
            scale = atol + rtol * np.abs(reference[name])
            assert np.max(np.abs(reference[name]) - scale) > 0.0, f"{trexio_file} step {step} {name} is vacuous"

    jax.clear_caches()


@pytest.mark.heavy
@pytest.mark.parametrize("non_local_move", ["tmove", "dltmove"])
def test_gfmc_t_public_force_keeps_mixed_fidelity_at_measurement_input(non_local_move, tmp_path):
    """Public GFMC_t HF force at a real N2 ccECP measurement input: mixed against an independent full rebuild.

    The input is the first measurement step of a full-precision production GFMC_t run (the
    same fixture, seed and settings as above, one walker), so it does not depend on the lane.
    The physical input is serialized once in full precision and reloaded under each policy.
    ``grad_e_L_t_wrt_positions`` is called without an inverse, as a public user would.  Its ECP
    non-local base fields must then assemble the geminal from the same fused orbital table as the
    kinetic reverse pass, not from the fp32 value-only table.
    """
    from jqmc._precision import configure, is_mixed_precision_enabled
    from jqmc._setting import get_tolerance_mixed_fidelity
    from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

    trexio_file = "N2_ecp_ccpvtz_cart.h5"
    entry_mode = "mixed" if is_mixed_precision_enabled() else "full"
    source = tmp_path / "full_physical_source.h5"
    try:
        configure("full")
        full_source = _hamiltonian(trexio_file)
        full_source.save_to_hdf5(source)
        driver = GFMC_t(
            hamiltonian_data=full_source,
            num_walkers=1,
            num_gfmc_collect_steps=1,
            mcmc_seed=3446,
            tau=0.1,
            alat=0.30,
            random_discretized_mesh=True,
            non_local_move=non_local_move,
            comput_position_deriv=True,
            use_swct=False,
        )
        spy = _BranchingDrawSpy(driver)
        driver._branching_rng = spy
        driver.run(num_mcmc_steps=1)
        driver._branching_rng = spy._rng
        assert len(spy.records) == 1
        row = spy.records[0]
    finally:
        configure(entry_mode)
    up, dn, rt = (np.asarray(row[name][0]).copy() for name in ("r_up", "r_dn", "latest_RTs"))

    def evaluate(hamiltonian):
        assert hamiltonian.coulomb_potential_data.ecp_flag
        force_R, force_up, force_dn = gfmc_derivatives.grad_e_L_t_wrt_positions(
            hamiltonian, jnp.asarray(up), jnp.asarray(dn), jnp.asarray(rt), row["alat"], non_local_move=non_local_move
        )
        return {"nuclear": force_R, "electron-up": force_up, "electron-down": force_dn}

    mixed, full = mixed_vs_full(lambda: Hamiltonian_data.load_from_hdf5(source), evaluate)
    assert set(mixed) == set(full) == {"nuclear", "electron-up", "electron-down"}
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    compared = 0
    for field in full:
        for mixed_block, full_block in mixed_fidelity_blocks(mixed[field], full[field]):
            compared += 1
            assert np.max(np.abs(mixed_block - full_block)) <= atol + rtol * np.max(np.abs(full_block)), (
                f"{non_local_move} {field}: {np.max(np.abs(mixed_block - full_block)):.3e} > "
                f"{atol + rtol * np.max(np.abs(full_block)):.3e}"
            )
            changed = out_of_budget_mixed_block(mixed_block, full_block, atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                assert np.max(np.abs(changed - full_block)) <= atol + rtol * np.max(np.abs(full_block))
    assert compared == 3
    jax.clear_caches()


@pytest.mark.heavy
def test_gfmc_t_autograd_fallback_preserves_masked_position_forces(monkeypatch):
    """Actual masked fallback storage agrees with independent same-input analytic forces.

    Three real H2 measurement steps use all J1/J2/J3 channels and no SWCT,
    isolating nuclear derivatives from the electron-force correction. The
    existing branching spy observes actual inputs without changing the walk;
    reference kernels do not call with_param_grad_mask.
    """
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    monkeypatch.setattr(implementation, "_ANALYTIC_GRAD_ENABLED", False)
    calls = []
    original_mask = Wavefunction_data.with_param_grad_mask

    def observed_mask(wf, **flags):
        calls.append(dict(flags))
        return original_mask(wf, **flags)

    monkeypatch.setattr(Wavefunction_data, "with_param_grad_mask", observed_mask)
    hamiltonian = _hamiltonian("H2_ae_ccpvdz_cart.h5")
    driver = GFMC_t(
        hamiltonian_data=hamiltonian,
        num_walkers=1,
        num_gfmc_collect_steps=1,
        mcmc_seed=3446,
        tau=0.1,
        alat=0.30,
        random_discretized_mesh=True,
        non_local_move="tmove",
        comput_position_deriv=True,
        use_swct=False,
    )
    spy = _BranchingDrawSpy(driver)
    driver._branching_rng = spy
    driver.run(num_mcmc_steps=3)
    driver._branching_rng = spy._rng
    assert len(spy.records) == 3
    assert any(
        all(flags.get(name) is False for name in ("opt_J1_param", "opt_J2_param", "opt_J3_param", "opt_geminal"))
        for flags in calls
    ), "the actual position-gradient fallback must trace its frozen-parameter mask"
    atol, rtol = get_tolerance_min(_FORCE_ZONES, "strict")
    stored = {name: np.asarray(getattr(driver, name)) for name in _STORED}
    for step, row in enumerate(spy.records):
        reference = _recompute(hamiltonian, row)
        energy = float(np.real(row["e_L_list"][0]))
        for key, name in (("grad_e_L_R", "_stored_force_HF"), ("grad_ln_Psi_dR", "_stored_force_PP")):
            _assert_close(row[key][0], reference[key], atol, rtol, f"fallback step {step} raw {key}")
            _assert_close(stored[name][step, 0], reference[key], atol, rtol, f"fallback step {step} stored {name}")
        _assert_close(
            stored["_stored_E_L_force_PP"][step, 0],
            energy * reference["grad_ln_Psi_dR"],
            atol,
            rtol,
            f"fallback step {step} stored E_L*PP",
        )
    jax.clear_caches()
