"""Real composite/DTN and B-spline/J123 production driver coverage."""

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

import pytest

from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel import _rs_fixtures as fixtures
from tests.test_rs_kernel.test_chain_modes import test_rust_vs_jax_energy as _mcmc_energy
from tests.test_rs_kernel.test_gfmc_chain_modes import (
    _STEPS_STAT,
    _assert_deterministic_driver_parity,
    _assert_gfmc_force_stress_driver_pair,
    _install_ctx_spy,
    _run_gfmc,
)

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]


def _install_patterns(monkeypatch):
    monkeypatch.setitem(fixtures._PATTERNS, "pbc_composite_jdtn", ("QE:Li_gamma_pgto+pw", True, {"dtn": True}, "SD"))
    monkeypatch.setitem(fixtures._PATTERNS, "bspline_gamma_j123", ("QEPROJ:bspline:Li_gamma:ECP", True, {"j3": "mo"}, "SD"))


@pytest.mark.parametrize("pattern", ["pbc_composite_jdtn", "bspline_gamma_j123"])
def test_richer_mcmc_driver_matches_xla(pattern, monkeypatch):
    """Fixed-seed production MCMC walks and energies agree for real composite/DTN and B-spline/J123.

    The committed QE Li Gamma source is built as pGTO+PW with active DTN or
    B-spline with exp J1/J2 and identity-MO-wrapped active J3. The existing
    production driver parity oracle supplies identical inputs and checks
    native/XLA walks and observables using its unchanged precision policy.
    """
    _install_patterns(monkeypatch)
    _mcmc_energy("rs", pattern)


def test_composite_dtn_gfmc_driver_matches_xla(monkeypatch):
    """Fixed-seed real composite/DTN GFMC_n force and stress chains agree with XLA.

    The QE Li Gamma pGTO+PW SD fixture has live DTN terms. Forty deterministic
    load-balanced production steps use one walker, seed34356, a fixed mesh,
    and SWCT-enabled force/stress. Walks, energies, weights and every stored
    force/stress buffer use the existing driver parity oracle and policy.
    """
    _install_patterns(monkeypatch)
    pattern = "pbc_composite_jdtn"
    _assert_gfmc_force_stress_driver_pair("n", pattern, fixtures.rs_ham(pattern), monkeypatch)


def test_composite_dtn_conventional_gfmc_driver_matches_xla(monkeypatch):
    """Fixed-seed composite/DTN conventional GFMC force and stress match XLA.

    The existing QE Li Gamma pGTO+PW SD plus DTN input uses one walker,
    seed34356, forty fixed-mesh steps and SWCT. The shared driver oracle
    compares energies, weights and every stored force/stress buffer under
    the existing same-precision policy, with exact final positions and keys.
    This adds the conventional algorithm without changing the original
    load-balanced test or its required parameter case.
    """
    _install_patterns(monkeypatch)
    pattern = "pbc_composite_jdtn"
    _assert_gfmc_force_stress_driver_pair("t", pattern, fixtures.rs_ham(pattern), monkeypatch)


def test_bspline_j123_gfmc_values_match_xla(monkeypatch):
    """B-spline/J123 GFMC_t cached values and exact walks agree, without claiming force/stress coverage.

    The committed QE Li Gamma B-spline SD fixture has live exp J1/J2 and
    identity-MO-wrapped J3. Forty conventional production steps use one
    walker, seed34356 and the fixed quadrature mesh. Stored energies, squared
    energies and weights plus final positions and keys use the existing
    deterministic driver oracle; this contract intentionally concerns values.
    """
    _install_patterns(monkeypatch)
    ham = fixtures.rs_ham("bspline_gamma_j123")
    xla = _run_gfmc("t", ham, 34356, "jax", _STEPS_STAT, random_mesh=False, num_walkers=1)
    counts = _install_ctx_spy(monkeypatch)
    rust = _run_gfmc("t", ham, 34356, "rust", _STEPS_STAT, random_mesh=False, num_walkers=1)
    assert counts["run_gfmc_projection_t"] >= _STEPS_STAT
    _assert_deterministic_driver_parity(rust, xla, "t", periodic=True)


@pytest.mark.parametrize("basis", ["bspline", "pw"])
@pytest.mark.parametrize("driver", ["t", "n"])
def test_grid_j3_gfmc_force_stress_matches_xla(basis, driver, monkeypatch):
    """Both GFMC drivers preserve active PW/B-spline J123 force, stress and SWCT buffers.

    The committed QE Li Gamma ccECP fixture becomes PW or B-spline SD with
    exp J1/J2 and active identity-MO-wrapped J3. One walker, seed 34356 and
    forty fixed-mesh steps are identical between backends. The existing
    production oracle checks stored energies, weights, force/stress and SWCT
    buffers within the central same-mode backend policy, final coordinates
    and RNG keys exactly, and native projection/Pulay inverse engagement.
    No statistical acceptance window or independent chain is used.
    """
    pattern = f"{basis}_gamma_j123"
    ham = _build_ham_for_cell(f"QEPROJ:{basis}:Li_gamma:ECP", True, {"j3": "mo"}, "SD")
    _assert_gfmc_force_stress_driver_pair(driver, pattern, ham, monkeypatch)


# Twist (k != Gamma) rows. Complex
# B-spline with active DTN, and complex pGTO+PW composite with active MO J3.
_TWIST_PATTERNS = {
    "bspline_twist_jdtn": ("QEPROJ:bspline:Li_k025:ECP", True, {"dtn": True}, "SD"),
    "pgto_pw_twist_j123": ("QEPROJ:pgto+pw:Li_k025:ECP", True, {"j3": "mo"}, "SD"),
}
# Route witnesses (``jqmc_rs.gfmc_route_counters()`` indices): (must be > 0, must be == 0).
# Common: complex carried state (14) engaged, state gate miss (23), complex det
# candidate slow route (21) and complex det derivative slow route (30) silent.
# B-spline+DTN: complex B-spline candidate build/refresh (17/18) and derivative
# build/refresh (26/27), complex DTN grad cache build/refresh (31/32) with its slow
# route (33) silent, ECP-mesh DTN table hits (63) with the per-point fallback (34) silent.
# pGTO+PW+J3: complex pGTO+PW candidate build/refresh (19/20) and derivative
# build/refresh (28/29), complex J3 grad cache build/refresh (38/39) with the J3 grad
# slow route (40) and the complex J3 candidate slow fallback (22) silent.
_TWIST_ROUTE_WITNESSES = {
    "bspline_twist_jdtn": ((14, 17, 18, 26, 27, 31, 32, 63), (21, 23, 30, 33, 34)),
    "pgto_pw_twist_j123": ((14, 19, 20, 28, 29, 38, 39), (21, 22, 23, 30, 40)),
}


@pytest.mark.parametrize("driver", ["n", "t"])
@pytest.mark.parametrize("pattern", sorted(_TWIST_PATTERNS))
def test_richer_twist_gfmc_values_match_xla(pattern, driver, monkeypatch):
    """Complex-twist B-spline/DTN and pGTO+PW/J123 GFMC values agree with XLA on a fixed input.

    The committed QE Li k=(1/4,1/4,1/4) ccECP source is realized as a complex
    B-spline SD with exp J1/J2 and an active DTN, or as a complex pGTO+PW
    composite SD with exp J1/J2 and an identity-MO-wrapped active J3. One-walker
    production GFMC_n and GFMC_t (seed 34356, fixed quadrature mesh, energy
    only) must run the native complex projection chunk, carry the complex fast
    state and engage the basis-specific complex candidate/derivative caches plus
    the DTN or J3 derivative cache with every slow-route counter at zero. Stored
    weights, energies and squared energies, the public energy and the final
    positions/keys then use the existing deterministic driver oracle and its
    within-mode policy. Values only: complex ECP force/stress is out of scope
    (an open limitation), and whether J3 on a complex Bloch orbital is physically correct is open;
    this row certifies Rust/XLA parity of the shared definition.
    """
    monkeypatch.setitem(fixtures._PATTERNS, pattern, _TWIST_PATTERNS[pattern])
    ham = fixtures.rs_ham(pattern)
    jastrow = ham.wavefunction_data.jastrow_data
    assert bool(ham.structure_data.pbc_flag)
    if pattern == "bspline_twist_jdtn":
        assert jastrow.jastrow_dtn_data is not None and jastrow.jastrow_three_body_data is None
    else:
        assert jastrow.jastrow_three_body_data is not None and jastrow.jastrow_dtn_data is None
    xla = _run_gfmc(driver, ham, 34356, "jax", _STEPS_STAT, random_mesh=False, num_walkers=1)
    counts = _install_ctx_spy(monkeypatch)
    import jqmc_rs

    jqmc_rs.gfmc_route_counters_reset()
    rust = _run_gfmc(driver, ham, 34356, "rust", _STEPS_STAT, random_mesh=False, num_walkers=1)
    assert counts[f"run_gfmc_projection_{driver}"] >= _STEPS_STAT
    route_counts = list(jqmc_rs.gfmc_route_counters())
    positive, silent = _TWIST_ROUTE_WITNESSES[pattern]
    idle = [index for index in positive if route_counts[index] == 0]
    fired = [index for index in silent if route_counts[index] != 0]
    assert not idle, f"{pattern} GFMC_{driver}: complex route witnesses idle {idle}"
    assert not fired, f"{pattern} GFMC_{driver}: complex slow routes fired {fired}"
    _assert_deterministic_driver_parity(rust, xla, driver, periodic=True)


# MCMC twist rows. The complex MCMC energy chunk runs only on the
# native random stream (it refuses injected inputs), so a Rust/XLA walk twin is not
# available. The production chunk still stores, for every step, e_L at the walker it
# has just swept to; at the end of each production call the stored last-step e_L of
# every walker must equal the XLA local energy evaluated from scratch at the
# returned walker (no carried state on the XLA side).
#
# This parametrized row keeps its B-spline/DTN case. The ``pgto_pw_twist_j123`` cell
# (phase-free shared J3) is its own node,
# test_pgto_pw_j3_twist_mcmc_chunk_energies_match_xla_recompute.
_MCMC_TWIST_SEGMENTS = (6, 6, 6)
_MCMC_TWIST_PATTERNS = ("bspline_twist_jdtn",)


def _complex_chunk_spy(monkeypatch):
    import jqmc_rs

    calls = []
    original_build = jqmc_rs.build_obs_ctx

    class _Ctx:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            attr = getattr(self._inner, name)
            if name != "run_energy_chunk_batch":
                return attr

            def _run(*args, **kwargs):
                calls.append((bool(kwargs.get("complex", False)), bool(kwargs.get("native_rng", False))))
                return attr(*args, **kwargs)

            return _run

    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", lambda *a, **k: _Ctx(original_build(*a, **k)))
    return calls


@pytest.mark.parametrize("pattern", _MCMC_TWIST_PATTERNS)
def test_richer_twist_mcmc_chunk_energies_match_xla_recompute(pattern, monkeypatch):
    """Production complex-twist MCMC stores e_L equal to an XLA from-scratch e_L at its walkers.

    The QE Li k=(1/4,1/4,1/4) ccECP complex B-spline SD with exp J1/J2 and an
    active DTN (the B-spline/DTN twist cell of the GFMC rows above; the
    pGTO+PW/J3 cell is excluded, see the note above). Two walkers, seed 34356, fixed quadrature mesh,
    energy only, three production ``run`` calls of six steps. The native complex
    energy chunk must be the route taken (``complex=True``, native random stream).
    After each call the last stored e_L of every walker (complex, Re and Im)
    must equal ``compute_local_energy`` of the unchanged Hamiltonian at the
    returned walker positions, within the MCMC Rust/XLA driver policy of
    ``test_chain_modes``. Complex ECP force/stress stays out of scope.
    """
    import jax.numpy as jnp
    import numpy as np

    from jqmc.hamiltonians import compute_local_energy
    from jqmc.jqmc_mcmc import MCMC
    from tests.test_rs_kernel.test_chain_modes import _assert_value_parity

    monkeypatch.setitem(fixtures._PATTERNS, pattern, _TWIST_PATTERNS[pattern])
    monkeypatch.delenv("JQMC_FLAG_RUST_ON_CPU", raising=False)
    ham = fixtures.rs_ham(pattern)
    calls = _complex_chunk_spy(monkeypatch)
    mcmc = MCMC(
        hamiltonian_data=ham,
        Dt=2.0,
        mcmc_seed=34356,
        epsilon_AS=1.0e-6,
        num_walkers=2,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        random_discretized_mesh=False,
    )
    compared = 0
    moved = False
    previous = None
    for steps in _MCMC_TWIST_SEGMENTS:
        mcmc.run(num_mcmc_steps=steps)
        r_up = np.asarray(mcmc._MCMC__latest_r_up_carts, dtype=np.float64)
        r_dn = np.asarray(mcmc._MCMC__latest_r_dn_carts, dtype=np.float64)
        stored = np.asarray(mcmc.e_L)[-1]
        assert np.iscomplexobj(stored) and np.all(np.isfinite(stored)), (pattern, stored)
        for w in range(r_up.shape[0]):
            ref = complex(np.asarray(compute_local_energy(ham, jnp.asarray(r_up[w]), jnp.asarray(r_dn[w]), jnp.eye(3))))
            assert np.isfinite(ref), (pattern, w, ref)
            _assert_value_parity(complex(stored[w]), ref, f"{pattern} MCMC walker {w} stored e_L vs XLA")
            compared += 1
        if previous is not None and not np.array_equal(previous, r_up):
            moved = True
        previous = r_up
    assert compared == 2 * len(_MCMC_TWIST_SEGMENTS)
    assert moved, f"{pattern}: walkers did not move between production calls"
    assert calls and all(c == (True, True) for c in calls), (pattern, calls)


@pytest.mark.parametrize(
    "pattern,complex_chunk",
    [("mol_ecp_sphe_j12", False), ("twist_sphepgto_j12", True)],
)
def test_native_mcmc_random_mesh_energies_match_haar_recompute(pattern, complex_chunk, monkeypatch):
    """Native real/complex chunks apply the prescribed Haar mesh at their returned keys.

    No normal or rotated-grid injection is permitted. The same returned positions
    are recomputed independently through XLA, with SciPy's extrinsic Euler
    composition anchoring Rz(gamma) Ry(beta) Rx(alpha). Invariant Haar measure is
    d(alpha) d(gamma) d(sin(beta)); the pitch therefore uses asin(2*u-1).
    The rotated and identity-mesh energies must be distinguishable at the actual
    comparison tolerance, so an all-electron or orientation-insensitive fixture
    cannot certify the nonlocal-ECP rotation.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import jqmc_rs
    from scipy.spatial.transform import Rotation

    from jqmc._precision import get_tolerance_xla_rust_consistency
    from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE
    from jqmc.hamiltonians import compute_local_energy
    from jqmc.jqmc_mcmc import MCMC, _generate_rotation_matrix

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    ham = fixtures.rs_ham(pattern)
    calls = []
    original_build = jqmc_rs.build_obs_ctx

    class _Ctx:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            attr = getattr(self._inner, name)
            if name != "run_energy_chunk_batch":
                return attr

            def _run(*args, **kwargs):
                assert kwargs.get("native_rng") is True
                assert kwargs.get("random_mesh") is True
                assert not kwargs.get("test_injected_inputs", False)
                assert np.asarray(args[9]).size == np.asarray(args[10]).size == 0
                calls.append((bool(kwargs.get("complex", False)), int(kwargs["n_steps_fast"])))
                return attr(*args, **kwargs)

            return _run

    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", lambda *a, **k: _Ctx(original_build(*a, **k)))
    driver = MCMC(
        hamiltonian_data=ham,
        Dt=2.0,
        mcmc_seed=34356,
        epsilon_AS=1.0e-6,
        num_walkers=2,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        random_discretized_mesh=True,
    )
    driver.run(num_mcmc_steps=6)
    assert calls == [(complex_chunk, 6)], (pattern, calls)
    stored = np.asarray(driver.e_L)[-1]
    assert np.all(np.isfinite(stored))
    assert np.iscomplexobj(stored) == complex_chunk
    r_up = np.asarray(driver._MCMC__latest_r_up_carts)
    r_dn = np.asarray(driver._MCMC__latest_r_dn_carts)
    keys = np.asarray(driver._MCMC__jax_PRNG_key_list)
    atol, rtol = get_tolerance_xla_rust_consistency(
        ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"), level="strict"
    )
    reference = []
    unrotated = []
    for ru, rd, key in zip(r_up, r_dn, keys, strict=True):
        subkey = jax.random.split(jnp.asarray(key))[1]
        u = np.asarray(jax.random.uniform(subkey, shape=(3,), minval=0.0, maxval=1.0))
        angles = [2 * np.pi * u[0], np.arcsin(2 * u[1] - 1), 2 * np.pi * u[2]]
        rotation = Rotation.from_euler("xyz", angles).as_matrix().T
        np.testing.assert_allclose(
            _generate_rotation_matrix(jnp.asarray(key)),
            rotation,
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_LOOSE,
            equal_nan=False,
        )
        reference.append(compute_local_energy(ham, jnp.asarray(ru), jnp.asarray(rd), jnp.asarray(rotation)))
        unrotated.append(compute_local_energy(ham, jnp.asarray(ru), jnp.asarray(rd), jnp.eye(3)))
    reference = np.asarray(reference)
    unrotated = np.asarray(unrotated)
    assert np.all(np.isfinite(reference)) and np.all(np.isfinite(unrotated))
    assert not np.allclose(reference, unrotated, atol=atol, rtol=rtol, equal_nan=False), (
        pattern,
        "nonlocal-ECP mesh orientation had no detectable effect",
        reference,
        unrotated,
    )
    np.testing.assert_allclose(
        stored,
        reference,
        atol=atol,
        rtol=rtol,
        equal_nan=False,
        err_msg=f"{pattern}: native random-mesh energy vs independent Haar/XLA recompute",
    )


def test_pgto_pw_j3_twist_mcmc_chunk_energies_match_xla_recompute(monkeypatch):
    """Production complex-twist pGTO+PW/J123 MCMC stores e_L equal to an XLA from-scratch e_L at its walkers.

    The QE Li k=(1/4,1/4,1/4) ccECP source realized as the complex pGTO+PW
    composite SD with exp J1/J2 and an identity-MO-wrapped active J3 (the
    phase-free shared J3 convention), i.e. the pGTO+PW/J3 twist cell
    of the GFMC rows above. The oracle and inputs of the B-spline/DTN MCMC row
    are reused unchanged: two walkers, seed 34356, fixed quadrature mesh,
    energy only, three production calls of six steps on the native complex
    chunk (complex=True, native random stream); after each call the last stored
    complex e_L of every walker equals compute_local_energy at the returned
    walker within the MCMC Rust/XLA driver policy, and the walkers move.
    """
    test_richer_twist_mcmc_chunk_energies_match_xla_recompute("pgto_pw_twist_j123", monkeypatch)


@pytest.mark.parametrize("driver", ["n", "t"])
@pytest.mark.parametrize("pattern", sorted(_TWIST_PATTERNS))
def test_richer_twist_gfmc_force_stress_match_xla(pattern, driver, monkeypatch):
    """Complex-twist B-spline/DTN and pGTO+PW/J123 GFMC force, stress and SWCT buffers agree with XLA.

    The two twist cells of test_richer_twist_gfmc_values_match_xla (QE Li
    k=(1/4,1/4,1/4) ccECP complex B-spline SD with exp J1/J2 and an active DTN,
    and complex pGTO+PW composite SD with exp J1/J2 and an identity-MO-wrapped
    active J3) run one-walker forty-step fixed-mesh GFMC_n and GFMC_t with seed
    34356, SWCT-enabled atomic force and stress, through the shared force/stress
    driver oracle: stored energies, weights, force/stress and SWCT buffers within
    the unchanged same-mode backend policy, final coordinates and RNG keys exact,
    and native projection plus carried-inverse Pulay engagement.
    """
    monkeypatch.setitem(fixtures._PATTERNS, pattern, _TWIST_PATTERNS[pattern])
    _assert_gfmc_force_stress_driver_pair(driver, pattern, fixtures.rs_ham(pattern), monkeypatch)


# The two B-spline/DTN twist value rows above carry the XLA DTN
# candidate cache over GFMC_t tau=0.05 / GFMC_n six-projection intervals, where a
# stale cache moves the stored values by less than the mixed-precision envelope.
# The same oracle over longer intervals keeps more hops on one carried cache.
_LONG_INTERVAL = {"t": {"tau": 0.3}, "n": {"nsub": 15}}


@pytest.mark.parametrize("driver", ["n", "t"])
def test_bspline_dtn_twist_long_interval_gfmc_values_match_xla(driver, monkeypatch):
    """Complex-twist B-spline/DTN GFMC values agree with XLA over long projection intervals.

    Same input, route witnesses and deterministic driver oracle as the
    bspline_twist_jdtn rows of test_richer_twist_gfmc_values_match_xla (QE Li
    k=(1/4,1/4,1/4) ccECP complex B-spline SD, exp J1/J2, active DTN, one
    walker, seed 34356, fixed quadrature mesh, energy only), with longer
    projection intervals: GFMC_t tau=0.3 and GFMC_n 15 projections per
    measurement. Each interval then advances the carried DTN candidate and
    derivative caches over many hops before the stored weights, energies and
    squared energies are compared within the unchanged same-mode policy, with
    final positions and keys exact.
    """
    import jqmc_rs

    pattern = "bspline_twist_jdtn"
    monkeypatch.setitem(fixtures._PATTERNS, pattern, _TWIST_PATTERNS[pattern])
    ham = fixtures.rs_ham(pattern)
    jastrow = ham.wavefunction_data.jastrow_data
    assert bool(ham.structure_data.pbc_flag)
    assert jastrow.jastrow_dtn_data is not None and jastrow.jastrow_three_body_data is None
    interval = _LONG_INTERVAL[driver]
    xla = _run_gfmc(driver, ham, 34356, "jax", _STEPS_STAT, random_mesh=False, num_walkers=1, **interval)
    counts = _install_ctx_spy(monkeypatch)
    jqmc_rs.gfmc_route_counters_reset()
    rust = _run_gfmc(driver, ham, 34356, "rust", _STEPS_STAT, random_mesh=False, num_walkers=1, **interval)
    assert counts[f"run_gfmc_projection_{driver}"] >= _STEPS_STAT
    route_counts = list(jqmc_rs.gfmc_route_counters())
    positive, silent = _TWIST_ROUTE_WITNESSES[pattern]
    idle = [index for index in positive if route_counts[index] == 0]
    fired = [index for index in silent if route_counts[index] != 0]
    assert not idle, f"{pattern} GFMC_{driver}: complex route witnesses idle {idle}"
    assert not fired, f"{pattern} GFMC_{driver}: complex slow routes fired {fired}"
    _assert_deterministic_driver_parity(rust, xla, driver, periodic=True)
