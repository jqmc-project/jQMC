"""collections of unit tests."""

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

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from gdd.engine.evidence import (
    ElectronConfigurationBinding,
    ElectronConfigurationReceipt,
    observe_electron_configurations,
)

pytestmark = [pytest.mark.with_chdir]
from mpi4py import MPI

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import (
    configure,
    get_dtype_jnp,
    get_tolerance_cancelling_mean,
    get_tolerance_min,
    get_tolerance_squared_observable,
    is_mixed_precision_enabled,
)
from jqmc._setting import (
    ATOL_PYTHON_FP64_MEDIUM,
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_STEPS,
    GFMC_MIN_WARMUP_STEPS,
    RTOL_PYTHON_FP64_ZERO,
)
from jqmc.determinant import Geminal_data, compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t, _GFMC_n_debug, _GFMC_t_debug
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._ecp_core_electrons import core_electrons_of
from tests._mpi_collective_check import collective_check

# MPI related
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# (trexio_file, with_1b_jastrow, with_2b_jastrow, with_3b_jastrow, non_local_move)
# Rows marked ``heavy`` exceed the local 60s/test budget under
# JIT warm-up (full Jastrow + PBC ECP + complex Bloch JIT compile).
_heavy = pytest.mark.heavy
_light = pytest.mark.light


def _full_precision_cond_g(
    geminal_data: Geminal_data,
    r_up_batch: jax.Array | npt.ArrayLike,
    r_dn_batch: jax.Array | npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluate the test trajectory's ``cond_2(G)`` in fp64."""
    configure("full")
    try:
        values = []
        for r_up, r_dn in zip(np.asarray(r_up_batch), np.asarray(r_dn_batch), strict=True):
            G = compute_geminal_all_elements(
                geminal_data=geminal_data,
                r_up_carts=jnp.asarray(r_up, dtype=jnp.float64),
                r_dn_carts=jnp.asarray(r_dn, dtype=jnp.float64),
            )
            values.append(float(np.linalg.cond(np.asarray(G), p=2)))
        return np.asarray(values, dtype=np.float64)
    finally:
        configure("mixed")


def _observe_well_conditioned_gfmc_boundary(
    boundary: str,
    r_up_batch: jax.Array | npt.ArrayLike,
    r_dn_batch: jax.Array | npt.ArrayLike,
    cond_g: npt.NDArray[np.float64],
    *,
    maximum_cond_g: float,
    num_steps: int,
    tau: float,
    seed: int,
) -> ElectronConfigurationReceipt:
    binding = ElectronConfigurationBinding(
        binding_id=f"ELECTRONS::MIXED-XLA-GFMC-T::{boundary}",
        mode="required",
        generation_pattern="direct",
        position_source_kind="seeded-random",
        generator_operation="jqmc.jqmc_gfmc.GFMC_t",
        generator_arguments=(("boundary", boundary), ("num_steps", num_steps), ("tau", tau)),
        random_algorithm="jax.random.PRNGKey+numpy.random.RandomState",
        seed=seed,
        configuration_count=int(np.asarray(r_up_batch).shape[0]),
        geminal_conditioning_mode="upper-bounded",
        maximum_cond_g=maximum_cond_g,
        upper_bound_comparison="less-than-or-equal",
        conditioning_violation_action="fail-case",
    )
    return observe_electron_configurations(
        binding,
        r_up_batch,
        r_dn_batch,
        cond_g_per_configuration=tuple(float(value) for value in cond_g),
    )


@pytest.mark.light
@pytest.mark.skip_full_precision(reason="mixed-XLA production/debug trajectory witness")
def test_mixed_xla_gfmc_t_debug_equals_production_in_well_conditioned_domain():
    """Actual mixed XLA ``GFMC_t`` equals ``_GFMC_t_debug`` end to end.

    The Li all-electron fixture exercises a non-trivial 2x2 open-shell
    determinant with J1+J2+J3.  Realized initial/final geminal matrices are
    screened into the declared well-conditioned domain; final walkers,
    projection counters, weights, and local-energy observables pin the full
    stochastic trajectory.

    Near-node derived-state robustness and full-vs-mixed stochastic replay are
    separate numerical claims.
    """
    assert get_dtype_jnp("ao_eval") is jnp.float32
    assert os.environ.get("JQMC_FLAG_RUST_ON_CPU") == "0", "this oracle is specifically the XLA route"

    structure_data, _, mos_data, _, geminal_data, coulomb_potential_data = read_trexio_file(
        trexio_file=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            "Li_ae_ccpvdz_cart.h5",
        ),
        store_tuple=True,
    )
    jastrow = Jastrow_data(
        jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
            jastrow_1b_type="pade",
        ),
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.0,
            jastrow_2b_type="pade",
            structure_data=structure_data,
        ),
        jastrow_three_body_data=Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=mos_data,
            random_init=True,
            random_scale=1.0e-3,
            seed=123,
        ),
    )
    hamiltonian = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow, geminal_data=geminal_data),
    )
    hamiltonian.sanity_check()

    mcmc_seed = 3446
    tau = 0.003
    num_steps = 4
    kwargs = dict(
        hamiltonian_data=hamiltonian,
        num_walkers=2,
        num_gfmc_collect_steps=1,
        mcmc_seed=mcmc_seed,
        tau=tau,
        alat=0.30,
        random_discretized_mesh=True,
        comput_position_deriv=False,
    )
    debug = _GFMC_t_debug(**kwargs)
    production = GFMC_t(**kwargs)

    max_cond_g = 1.0e4
    initial_debug_cond = _full_precision_cond_g(
        geminal_data,
        debug._latest_r_up_carts,
        debug._latest_r_dn_carts,
    )
    initial_production_cond = _full_precision_cond_g(
        geminal_data,
        production._latest_r_up_carts,
        production._latest_r_dn_carts,
    )
    np.testing.assert_array_equal(initial_debug_cond, initial_production_cond)
    _observe_well_conditioned_gfmc_boundary(
        "initial-production",
        production._latest_r_up_carts,
        production._latest_r_dn_carts,
        initial_production_cond,
        maximum_cond_g=max_cond_g,
        num_steps=num_steps,
        tau=tau,
        seed=mcmc_seed,
    )
    assert np.all(initial_production_cond <= max_cond_g), (
        f"precondition failed before correctness evaluation: initial cond_2(G) "
        f"{initial_production_cond} exceeds {max_cond_g:.1e}"
    )

    debug.run(num_mcmc_steps=num_steps)
    production.run(num_mcmc_steps=num_steps)

    final_debug_cond = _full_precision_cond_g(
        geminal_data,
        debug._latest_r_up_carts,
        debug._latest_r_dn_carts,
    )
    final_production_cond = _full_precision_cond_g(
        geminal_data,
        production._latest_r_up_carts,
        production._latest_r_dn_carts,
    )
    _observe_well_conditioned_gfmc_boundary(
        "final-production",
        production._latest_r_up_carts,
        production._latest_r_dn_carts,
        final_production_cond,
        maximum_cond_g=max_cond_g,
        num_steps=num_steps,
        tau=tau,
        seed=mcmc_seed,
    )
    assert np.all(final_debug_cond <= max_cond_g), (
        f"debug trajectory left the declared well-conditioned domain: {final_debug_cond}"
    )
    assert np.all(final_production_cond <= max_cond_g), (
        f"production trajectory left the declared well-conditioned domain: {final_production_cond}"
    )

    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"),
        "medium",
    )
    np.testing.assert_allclose(
        debug._latest_r_up_carts,
        production._latest_r_up_carts,
        atol=ATOL_PYTHON_FP64_MEDIUM,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    np.testing.assert_allclose(
        debug._latest_r_dn_carts,
        production._latest_r_dn_carts,
        atol=ATOL_PYTHON_FP64_MEDIUM,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    np.testing.assert_array_equal(debug._jax_PRNG_key_list, production._jax_PRNG_key_list)
    np.testing.assert_array_equal(debug.average_projection_counter, production.average_projection_counter)
    with collective_check(mpi_comm, "mixed GFMC_t prod-vs-debug w_L / e_L / e_L2"):
        if mpi_rank == 0:
            # GFMC reduces observables to rank 0; non-root ranks deliberately
            # keep their debug storage empty.  Walker, PRNG, and counter parity
            # above is still checked independently on every rank.
            np.testing.assert_allclose(debug.w_L, production.w_L, atol=atol, rtol=rtol)
            np.testing.assert_allclose(debug.e_L, production.e_L, atol=atol, rtol=rtol)
            np.testing.assert_allclose(debug.e_L2, production.e_L2, atol=atol, rtol=rtol)
    counters = np.asarray(production.average_projection_counter)
    assert np.all(np.isfinite(counters))
    assert np.all((counters > 0) & (counters < 1.0e3)), f"projection count escaped robustness envelope: {counters}"


param_grid_n = [
    # (trexio_file, with_1b, with_2b, with_3b, non_local_move)
    # -- ECP cases (molecular) --
    pytest.param("H_ecp_ccpvdz_cart.h5", True, False, False, "tmove", marks=_light),
    pytest.param("H_ecp_ccpvdz_cart.h5", False, False, False, "dltmove", marks=_light),
    pytest.param("H_ecp_ccpvdz_cart.h5", True, True, True, "dltmove", marks=_light),
    # -- AE cases (molecular, non_local_move=None) --
    pytest.param("H2_ae_ccpvdz_cart.h5", True, True, True, None, marks=_light),
    # AE open-shell (n_up != n_dn): Li atom (2 up, 1 dn)
    pytest.param("Li_ae_ccpvdz_cart.h5", True, True, True, None, marks=_light),
    # -- Periodic ECP fixture: lithium.h5 (2-atom Li ccECP) --
    # Used instead of silicon.h5 (same reason as ``test_jqmc_gfmc_tau``): silicon
    # force-grad runs out of memory at 256 GB. lithium keeps PBC + ECP + tmove +
    # full Jastrow code path coverage at ~6 GB peak for ewald.
    pytest.param("lithium.h5", True, True, True, "tmove", marks=_heavy),
    # -- Complex-Bloch fixture (twist at k_frac = (0.25, 0, 0)) --
    # Catches three bug classes (loader mo_dn=conj(mo_up), SVD pseudo-inverse
    # missing conj, body_fun
    # walker wrap post-SM drift). Bit-identical at Gamma. lithium_k025.h5
    # carries a live Li ccECP; non_local_move=None selects the default
    # tmove. Jastrow off keeps per-walker memory below the
    # _GFMC_n_debug OOM threshold.
    pytest.param("lithium_k025.h5", False, False, False, None, marks=_heavy),
    # -- QE-projection basis-coverage sentinels (Gamma real, Li ccECP, Ne=2) --
    # The determinant candidate caches in the production GFMC drivers dispatch
    # through compute_orb_api, so the B-spline and pGTO+PW leaves must be
    # compared against the debug oracle too (they were never covered before).
    # ``with_1b/with_2b`` map onto the builder's coupled J1+J2 recipe; J3 off.
    pytest.param("QEPROJ:bspline:Li_gamma:ECP", True, True, False, "tmove", marks=_light),
    pytest.param("QE:Li_gamma_pgto+pw", True, True, False, "tmove", marks=_light),
    # -- Force-wiring diagonal coverage gaps.  Together with the rows above,
    # every active Jastrow leg, determinant representation and supported AO
    # family reaches production=debug at least once without a Cartesian product.
    pytest.param("RSPATTERN:mol_ecp_cart_jdtn", True, True, False, "tmove", marks=_heavy),
    pytest.param("RSPATTERN:mol_sphe_agp_jnone", False, False, False, None, marks=_heavy),
    pytest.param("RSPATTERN:mol_cart_agpn_jnone", False, False, False, None, marks=_heavy),
    pytest.param("RSPATTERN:pw_gamma_jnone", False, False, False, "tmove", marks=_heavy),
    pytest.param("RSPATTERN:pgto_bspline_gamma_jnone", False, False, False, "tmove", marks=_heavy),
]


#: Two-nucleus molecular rows re-run with SWCT on (``TestGFMCBra.test_jqmc_gfmc_n_swct``).
#: A single nucleus makes the SWCT weight identically one and its gradient
#: identically zero, so the one-atom rows of ``param_grid_n`` cannot see the
#: domega/2 term; H2 all-electron and H2 ccECP (tmove and dltmove) can.
param_grid_n_swct = [
    pytest.param("H2_ae_ccpvdz_cart.h5", True, True, True, None, marks=_light),
    pytest.param("H2_ecp_ccpvdz_cart.h5", True, True, True, "tmove", marks=_light),
    pytest.param("H2_ecp_ccpvdz_cart.h5", True, True, True, "dltmove", marks=_light),
]


def _build_qe_sentinel_hamiltonian(stem: str, with_jastrow: bool, with_3b_jastrow: bool) -> Hamiltonian_data:
    """Hamiltonian for a QE-projection basis sentinel (B-spline / pGTO+PW).

    Delegates to the shared derivative-matrix builder so the wavefunction
    assembly (projection, ECP parse, coupled J1+J2 recipe) stays single-source.
    """
    from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

    spec = {"j3": "mo"} if with_3b_jastrow else None
    return _build_ham_for_cell(stem, with_jastrow, spec)


class TestGFMCBra:
    @pytest.mark.parametrize(
        "trexio_file,with_1b_jastrow,with_2b_jastrow,with_3b_jastrow,non_local_move",
        param_grid_n,
    )
    @pytest.mark.parametrize(
        "mcmc_seed,num_walkers,num_mcmc_steps,num_mcmc_per_measurement,alat,E_scf,num_gfmc_collect_steps,warmup,bins",
        [(3446, 2, 40, 10, 0.30, -1.00, 5, 25, 10)],
        ids=["original-protocol"],
    )
    def test_jqmc_gfmc_n(
        self,
        trexio_file,
        with_1b_jastrow,
        with_2b_jastrow,
        with_3b_jastrow,
        non_local_move,
        mcmc_seed,
        num_walkers,
        num_mcmc_steps,
        num_mcmc_per_measurement,
        alat,
        E_scf,
        num_gfmc_collect_steps,
        warmup,
        bins,
        use_swct=False,
    ):
        """Every load-balanced hop and measurement matches its same-input reference.

        Keep the declared TREXIO/QE/Rust-pattern fixtures, two walkers per rank,
        seed 3446, forty intervals of ten hops, rotating meshes and force collection.
        No cond(G) screening is added. Full and mixed each use their existing
        strict/medium policy, never cross-precision agreement or a sampling band.
        An unobserved real driver must bit-exactly match the observed driver's walker
        trajectories, PRNG keys, carried inverse and integer bookkeeping; its floating
        observable accumulators must match within the node's energy/force policies.
        Full-recompute primitives independently assemble every candidate, diagonal,
        branching factor and fresh inverse at each real input. Categorical choices
        must exactly invert their own checked CDF; keys, carries and rank-wide
        resampling are checked. Fresh-inverse derivatives use the existing analytic
        backend (not an independent derivative formula). Measurement moments,
        historical weight products, energy/variance/force means and jackknife errors
        are independently recomputed from checked fixed inputs. Separately evolving
        trajectories and stress buffers are not the correctness oracle of this node.
        ``use_swct`` is not a fixture (it has a default); ``test_jqmc_gfmc_n_swct``
        re-enters this body with ``use_swct=True``.
        """
        if trexio_file.startswith("RSPATTERN:"):
            from tests.test_rs_kernel._rs_fixtures import rs_ham

            hamiltonian_data = rs_ham(trexio_file.removeprefix("RSPATTERN:"))
        elif trexio_file.startswith(("QE:", "QEPROJ:")):
            hamiltonian_data = _build_qe_sentinel_hamiltonian(trexio_file, with_1b_jastrow, with_3b_jastrow)
        else:
            (
                structure_data,
                _,
                mos_data,
                _,
                geminal_mo_data,
                coulomb_potential_data,
            ) = read_trexio_file(
                trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
                store_tuple=True,
            )

            jastrow_onebody_data = None
            if with_1b_jastrow:
                jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
                    jastrow_1b_param=1.0,
                    structure_data=structure_data,
                    core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
                    jastrow_1b_type="exp",
                )

            jastrow_twobody_data = None
            if with_2b_jastrow:
                jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(
                    jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure_data
                )

            jastrow_threebody_data = None
            if with_3b_jastrow:
                jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
                    orb_data=mos_data, random_init=True, random_scale=1.0e-3, seed=123
                )

            jastrow_data = Jastrow_data(
                jastrow_one_body_data=jastrow_onebody_data,
                jastrow_two_body_data=jastrow_twobody_data,
                jastrow_three_body_data=jastrow_threebody_data,
            )

            wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
            wavefunction_data.sanity_check()

            hamiltonian_data = Hamiltonian_data(
                structure_data=structure_data,
                coulomb_potential_data=coulomb_potential_data,
                wavefunction_data=wavefunction_data,
            )
        hamiltonian_data.sanity_check()

        nlm_kwargs = {"non_local_move": non_local_move} if non_local_move is not None else {}

        from tests._gfmc_n_same_input import (
            ObserveGFMCN,
            driver_state,
            finite_sample_estimates,
            history_comparisons,
            measurement_comparisons,
            observer_neutrality_split,
            population_comparisons,
            projection_comparisons,
        )

        settings = dict(
            hamiltonian_data=hamiltonian_data,
            num_walkers=num_walkers,
            num_mcmc_per_measurement=num_mcmc_per_measurement,
            num_gfmc_collect_steps=num_gfmc_collect_steps,
            mcmc_seed=mcmc_seed,
            E_scf=E_scf,
            alat=alat,
            random_discretized_mesh=True,
            comput_position_deriv=True,
            use_swct=use_swct,
            **nlm_kwargs,
        )
        baseline = GFMC_n(**settings)
        baseline.run(num_mcmc_steps=num_mcmc_steps)
        gfmc_jax = GFMC_n(**settings)
        observer = ObserveGFMCN(gfmc_jax)
        with pytest.MonkeyPatch.context() as patch:
            observer.install(patch)
            gfmc_jax.run(num_mcmc_steps=num_mcmc_steps)

        # Keep the reviewed within-mode policies; no trajectory-dependent skip.
        parity_level = "medium" if is_mixed_precision_enabled() else "strict"
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"),
            parity_level,
        )
        force_atol, force_rtol = get_tolerance_min(
            ("ao_eval", "ao_grad_lap", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic"),
            parity_level,
        )
        with collective_check(mpi_comm, "GFMC_n same-input leaves and observer neutrality"):
            actual_state, baseline_state = driver_state(gfmc_jax), driver_state(baseline)
            assert actual_state.keys() == baseline_state.keys()
            # Bitwise for trajectories, keys,
            # inverse and bookkeeping; observables are judged at the end of this test.
            bitwise_names, observer_energy, observer_force = observer_neutrality_split(actual_state, baseline_state)
            observer_equal = np.asarray([np.array_equal(actual_state[name], baseline_state[name]) for name in bitwise_names])
            np.testing.assert_array_equal(observer_equal, np.ones_like(observer_equal))
            projection = projection_comparisons(
                observer.intervals,
                expected_intervals=num_mcmc_steps,
                expected_walkers=num_walkers,
                expected_hops=num_mcmc_per_measurement,
                expected_random_mesh=True,
                expected_non_local_move=non_local_move or "tmove",
                expected_alat=alat,
                expected_hamiltonian=observer.hamiltonian,
            )
            measurement, assembled = measurement_comparisons(observer.intervals, observer.branches, use_swct=gfmc_jax._use_swct)
            np.testing.assert_array_equal(projection["exact"][0], projection["exact"][1])
            np.testing.assert_allclose(
                projection["candidates"][0], projection["candidates"][1], atol=atol, rtol=rtol, equal_nan=False
            )
            np.testing.assert_allclose(projection["cdf"][0], projection["cdf"][1], atol=atol, rtol=rtol, equal_nan=False)
            np.testing.assert_allclose(
                projection["propagation"][0], projection["propagation"][1], atol=atol, rtol=rtol, equal_nan=False
            )
            np.testing.assert_allclose(
                projection["weights"][0], projection["weights"][1], atol=atol, rtol=rtol, equal_nan=False
            )
            np.testing.assert_allclose(
                projection["inverse"][0], projection["inverse"][1], atol=atol, rtol=rtol, equal_nan=False
            )
            np.testing.assert_allclose(
                projection["positions"][0], projection["positions"][1], atol=atol, rtol=rtol, equal_nan=False
            )
            np.testing.assert_array_equal(measurement["exact"][0], measurement["exact"][1])
            np.testing.assert_allclose(
                measurement["measurement_diagonal"][0],
                measurement["measurement_diagonal"][1],
                atol=atol,
                rtol=rtol,
                equal_nan=False,
            )
            np.testing.assert_allclose(
                measurement["measurement_energy"][0],
                measurement["measurement_energy"][1],
                atol=atol,
                rtol=rtol,
                equal_nan=False,
            )
            np.testing.assert_allclose(
                measurement["measurement_inverse"][0],
                measurement["measurement_inverse"][1],
                atol=atol,
                rtol=rtol,
                equal_nan=False,
            )
            np.testing.assert_allclose(
                measurement["force_leaves"][0],
                measurement["force_leaves"][1],
                atol=force_atol,
                rtol=force_rtol,
                equal_nan=False,
            )
            np.testing.assert_allclose(
                measurement["force_assembly"][0],
                measurement["force_assembly"][1],
                atol=force_atol,
                rtol=force_rtol,
                equal_nan=False,
            )

        # All ranks finish the local comparisons before entering collectives.
        payloads = mpi_comm.allgather(observer.population_payload(assembled))
        with collective_check(mpi_comm, "GFMC_n population handoff and fixed-sample moments"):
            population = population_comparisons(
                payloads, expected_intervals=num_mcmc_steps, expected_walkers=num_walkers, initial_energy_shift=E_scf
            )
            np.testing.assert_array_equal(population["exact"][0], population["exact"][1])
            np.testing.assert_allclose(
                population["branching_cdf"][0], population["branching_cdf"][1], atol=atol, rtol=rtol, equal_nan=False
            )
            np.testing.assert_allclose(
                population["weights"][0], population["weights"][1], atol=atol, rtol=rtol, equal_nan=False
            )
            np.testing.assert_allclose(
                population["energy_shift"][0], population["energy_shift"][1], atol=atol, rtol=rtol, equal_nan=False
            )
            e_L_atol, e_L_rtol = get_tolerance_cancelling_mean(atol, rtol, np.max(np.abs(population["moments_energy"][1])))
            np.testing.assert_allclose(
                population["moments_energy"][0], population["moments_energy"][1], atol=e_L_atol, rtol=e_L_rtol, equal_nan=False
            )
            e_L2_atol, e_L2_rtol = get_tolerance_squared_observable(atol, rtol)
            np.testing.assert_allclose(
                population["moments_energy_squared"][0],
                population["moments_energy_squared"][1],
                atol=e_L2_atol,
                rtol=e_L2_rtol,
                equal_nan=False,
            )
            np.testing.assert_allclose(
                population["moments_force"][0],
                population["moments_force"][1],
                atol=force_atol,
                rtol=force_rtol,
                equal_nan=False,
            )
            # Conditional-input aggregation also obeys the tighter unmodified
            # base pair. Keep the existing cancelling/squared comparisons above;
            # these extra checks expose both moments to the executable policy gate.
            np.testing.assert_allclose(
                population["moments_energy"][0], population["moments_energy"][1], atol=atol, rtol=rtol, equal_nan=False
            )
            np.testing.assert_allclose(
                population["moments_energy_squared"][0],
                population["moments_energy_squared"][1],
                atol=atol,
                rtol=rtol,
                equal_nan=False,
            )
            if mpi_rank == 0:
                histories = history_comparisons(gfmc_jax)
                np.testing.assert_allclose(
                    histories["weights"][0], histories["weights"][1], atol=atol, rtol=rtol, equal_nan=False
                )
                for name in ("e_L", "e_L2", "force_HF", "force_PP", "E_L_force_PP"):
                    np.testing.assert_array_equal(histories[name][0], histories[name][1])
            root_fields = payloads[0]["driver_fields"]
            expected_energy, expected_force = finite_sample_estimates(
                *(
                    root_fields[name][:, 0]
                    for name in (
                        "_stored_w_L",
                        "_stored_e_L",
                        "_stored_e_L2",
                        "_stored_force_HF",
                        "_stored_force_PP",
                        "_stored_E_L_force_PP",
                    )
                ),
                collect=num_gfmc_collect_steps,
                warmup=warmup,
                bins=bins,
            )
        actual_energy = gfmc_jax.get_E(warmup, bins)
        actual_force = gfmc_jax.get_aF(warmup, bins)
        np.testing.assert_allclose(actual_energy, expected_energy, atol=atol, rtol=rtol, equal_nan=False)
        np.testing.assert_allclose(actual_force, expected_force, atol=force_atol, rtol=force_rtol, equal_nan=False)
        with collective_check(mpi_comm, "GFMC_n observer neutrality of floating observables"):
            # The observed driver's observable accumulators equal the unobserved
            # driver's within the node's existing energy and force policies.
            np.testing.assert_allclose(observer_energy[0], observer_energy[1], atol=atol, rtol=rtol, equal_nan=False)
            np.testing.assert_allclose(observer_force[0], observer_force[1], atol=force_atol, rtol=force_rtol, equal_nan=False)

        jax.clear_caches()

    @pytest.mark.parametrize(
        "trexio_file,with_1b_jastrow,with_2b_jastrow,with_3b_jastrow,non_local_move",
        param_grid_n_swct,
    )
    @pytest.mark.parametrize("use_swct", [True], ids=["swct"])
    def test_jqmc_gfmc_n_swct(
        self,
        trexio_file,
        with_1b_jastrow,
        with_2b_jastrow,
        with_3b_jastrow,
        non_local_move,
        use_swct,
    ):
        """The GFMC_n same-input node above, re-run with the SWCT force contraction on.

        Together with ``test_jqmc_gfmc_n`` (``use_swct=False``) this is the
        ``use_swct`` in {False, True} parametrization of the same-input check.
        The body, seed 3446, two walkers, forty
        intervals of ten hops, rotating meshes and every comparison and
        tolerance are those of ``test_jqmc_gfmc_n``; only the driver's
        ``use_swct`` changes, so the stored force_HF / force_PP / E_L*force_PP
        carry the driver's SWCT Omega contraction and the domega/2 term,
        and the per-walker reference (``tests/_gfmc_n_same_input.py``
        ``_force_leaves`` / ``assemble_observed_forces``) contracts the jqmc.swct
        weights explicitly at the same inputs. Molecular rows only: SWCT under
        PBC is outside this node.
        """
        self.test_jqmc_gfmc_n(
            trexio_file,
            with_1b_jastrow,
            with_2b_jastrow,
            with_3b_jastrow,
            non_local_move,
            mcmc_seed=3446,
            num_walkers=2,
            num_mcmc_steps=40,
            num_mcmc_per_measurement=10,
            alat=0.30,
            E_scf=-1.00,
            num_gfmc_collect_steps=5,
            warmup=25,
            bins=10,
            use_swct=use_swct,
        )


# ---------------------------------------------------------------------------
# GFMC_t (tau)
# ---------------------------------------------------------------------------

param_grid_t = [
    # (trexio_file, with_1b, with_2b, with_3b, non_local_move)
    # -- ECP cases (molecular) --
    pytest.param("H_ecp_ccpvdz_cart.h5", False, False, False, "tmove", marks=_light),
    # ("H_ecp_ccpvdz_cart.h5", False, False, False, "dltmove"),
    pytest.param("H_ecp_ccpvdz_cart.h5", True, True, True, "tmove", marks=_light),
    # ("H_ecp_ccpvdz_cart.h5", True, True, True, "dltmove"),
    # -- AE cases (molecular, no non_local_move) --
    pytest.param("H2_ae_ccpvdz_cart.h5", True, True, True, None, marks=_light),
    # AE open-shell (n_up != n_dn): Li atom (2 up, 1 dn)
    pytest.param("Li_ae_ccpvdz_cart.h5", True, True, True, None, marks=_light),
    # -- Periodic ECP fixture: lithium.h5 (2-atom Li ccECP) --
    # Covers PBC + ECP + tmove + full Jastrow at ~6 GB peak for ewald.
    # silicon.h5 (8-atom ccECP-ccpvtz) is not used: its force-grad hits
    # ~67 GB per walker (95% AD tape) on a 256 GB host, and _GFMC_t_debug
    # piles on top -> OOM.
    # The mixed skip mirrors the param_grid_n lithium+J3 row (fp32 catastrophic
    # cancellation in the debug-side full-J exponent); the row runs in the
    # mixed precision lane.
    pytest.param("lithium.h5", True, True, True, "tmove", marks=_heavy),
    # -- Complex-Bloch fixture (twist at k_frac = (0.25, 0, 0)) --
    # Catches three bug classes (loader conj, SVD conj, post-SM walker wrap).
    # Bit-identical at Gamma. lithium_k025.h5 carries a live Li ccECP;
    # non_local_move=None selects the default tmove. Jastrow off
    # keeps per-walker memory below the _GFMC_t_debug OOM threshold.
    pytest.param("lithium_k025.h5", False, False, False, None, marks=_heavy),
    # -- QE-projection basis-coverage sentinels (Gamma real, Li ccECP, Ne=2) --
    # Same rationale as the param_grid_n rows: the production candidate caches
    # dispatch polymorphically, so B-spline and pGTO+PW leaves need their own
    # prod-vs-debug comparison.
    pytest.param("QEPROJ:bspline:Li_gamma:ECP", True, True, False, "tmove", marks=_light),
    pytest.param("QE:Li_gamma_pgto+pw", True, True, False, "tmove", marks=_light),
    # The n-driver matrix above owns the determinant/AO diagonal.  Add DTN here
    # as well because GFMC_t and GFMC_n have separate projection/force wiring.
    pytest.param("RSPATTERN:mol_ecp_cart_jdtn", True, True, False, "tmove", marks=_heavy),
]


class TestGFMCTau:
    @pytest.mark.parametrize(
        "trexio_file,with_1b_jastrow,with_2b_jastrow,with_3b_jastrow,non_local_move",
        param_grid_t,
    )
    # No ``skip_mixed_precision``: after the GFMC_t lax.cond miscompile fix all
    # light rows pass at the mixed medium tier (--precision-mode=mixed), so the
    # comparison is enabled in both precision lanes. The heavy lithium rows
    # keep their own documented mixed behaviour.
    def test_jqmc_gfmc_t(self, trexio_file, with_1b_jastrow, with_2b_jastrow, with_3b_jastrow, non_local_move):
        """LRDMC debug vs production comparison.

        The Ewald regression: the production driver and the ``_GFMC_t_debug``
        oracle must agree on the branching weights, ``e_L`` and the force/stress
        estimators.
        """
        if trexio_file.startswith("RSPATTERN:"):
            from tests.test_rs_kernel._rs_fixtures import rs_ham

            hamiltonian_data = rs_ham(trexio_file.removeprefix("RSPATTERN:"))
        elif trexio_file.startswith(("QE:", "QEPROJ:")):
            hamiltonian_data = _build_qe_sentinel_hamiltonian(trexio_file, with_1b_jastrow, with_3b_jastrow)
        else:
            (
                structure_data,
                _,
                mos_data,
                _,
                geminal_mo_data,
                coulomb_potential_data,
            ) = read_trexio_file(
                trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
                store_tuple=True,
            )

            jastrow_onebody_data = None
            if with_1b_jastrow:
                jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
                    jastrow_1b_param=1.0,
                    structure_data=structure_data,
                    core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
                    jastrow_1b_type="pade",
                )

            jastrow_twobody_data = None
            if with_2b_jastrow:
                jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(
                    jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure_data
                )

            jastrow_threebody_data = None
            if with_3b_jastrow:
                jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
                    orb_data=mos_data, random_init=True, random_scale=1.0e-3, seed=123
                )

            jastrow_data = Jastrow_data(
                jastrow_one_body_data=jastrow_onebody_data,
                jastrow_two_body_data=jastrow_twobody_data,
                jastrow_three_body_data=jastrow_threebody_data,
            )

            wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
            wavefunction_data.sanity_check()

            hamiltonian_data = Hamiltonian_data(
                structure_data=structure_data,
                coulomb_potential_data=coulomb_potential_data,
                wavefunction_data=wavefunction_data,
            )
        hamiltonian_data.sanity_check()

        # GFMC param
        num_mcmc_steps = GFMC_MIN_STEPS
        num_walkers = 2
        mcmc_seed = 3446
        alat = 0.30
        # Short imaginary-time projection for the test: tau only sets the
        # per-branching while-loop length (~tau/dtau substeps); debug and prod
        # use the same tau so the equality check is unaffected, while 0.03 vs
        # 0.20 cuts the dominant eager-substep cost of the _GFMC_t_debug path.
        tau = 0.03
        num_gfmc_collect_steps = GFMC_MIN_COLLECT_STEPS

        nlm_kwargs = {"non_local_move": non_local_move} if non_local_move is not None else {}

        # run LRDMC single-shots
        # Exercise the production cadence contract directly: one random mesh
        # rotation is drawn per projection interval and shared by all of its
        # escape-time substeps in both production and debug implementations.
        gfmc_debug = _GFMC_t_debug(
            hamiltonian_data=hamiltonian_data,
            num_walkers=num_walkers,
            num_gfmc_collect_steps=num_gfmc_collect_steps,
            mcmc_seed=mcmc_seed,
            tau=tau,
            alat=alat,
            random_discretized_mesh=True,
            comput_position_deriv=True,
            **nlm_kwargs,
        )
        gfmc_debug.run(num_mcmc_steps=num_mcmc_steps)

        # run LRDMC single-shots
        gfmc_jax = GFMC_t(
            hamiltonian_data=hamiltonian_data,
            num_walkers=num_walkers,
            num_gfmc_collect_steps=num_gfmc_collect_steps,
            mcmc_seed=mcmc_seed,
            tau=tau,
            alat=alat,
            random_discretized_mesh=True,
            comput_position_deriv=True,
            **nlm_kwargs,
        )
        gfmc_jax.run(num_mcmc_steps=num_mcmc_steps)

        # e_L / e_L2 / w_L cross ao_eval/jastrow_eval/det_eval/coulomb/wf_kinetic zones;
        # the achievable debug-vs-jax agreement is bounded by the weakest (fp32 in mixed).
        # In full mode the eager debug loop and compiled fast-update loop differ in
        # reduction order across supported JAX versions; the named medium fp64
        # envelope covers that deterministic aggregate without error-bar statistics.
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"),
            "medium",
        )

        with collective_check(mpi_comm, "GFMC_t prod-vs-debug w_L / e_L / e_L2"):
            if mpi_rank == 0:
                # w_L
                w_L_debug = gfmc_debug.w_L
                w_L_jax = gfmc_jax.w_L
                assert not np.any(np.isnan(np.asarray(w_L_debug))), "NaN detected in first argument"
                assert not np.any(np.isnan(np.asarray(w_L_jax))), "NaN detected in second argument"
                np.testing.assert_allclose(w_L_debug, w_L_jax, atol=atol, rtol=rtol)

                # e_L. Each element is a WEIGHTED MEAN over the per-walker local
                # energies, and those carry mixed signs (the LRDMC discretized
                # Laplacian makes individual walkers' e_L positive), so a step can
                # cancel far below the array's own scale. Measured on this file under
                # ``mpirun -np 2``: median |e_L| = 9.22 with one element at 0.0315,
                # i.e. 293x below scale, which handed that single element a 222x
                # tighter ABSOLUTE pin than its neighbours for no arithmetic reason.
                # Rebase the envelope onto the array's scale.
                e_L_debug = gfmc_debug.e_L
                e_L_jax = gfmc_jax.e_L
                assert not np.any(np.isnan(np.asarray(e_L_debug))), "NaN detected in first argument"
                assert not np.any(np.isnan(np.asarray(e_L_jax))), "NaN detected in second argument"
                e_L_atol, e_L_rtol = get_tolerance_cancelling_mean(atol, rtol, np.max(np.abs(np.asarray(e_L_debug))))
                np.testing.assert_allclose(e_L_debug, e_L_jax, atol=e_L_atol, rtol=e_L_rtol)

                # e_L2
                e_L2_debug = gfmc_debug.e_L2
                e_L2_jax = gfmc_jax.e_L2
                assert not np.any(np.isnan(np.asarray(e_L2_debug))), "NaN detected in first argument"
                assert not np.any(np.isnan(np.asarray(e_L2_jax))), "NaN detected in second argument"
                # e_L2 = e_L**2 doubles the relative error of e_L (d(x^2)/x^2 = 2 dx/x),
                # so the base envelope's rtol is widened by the central squared-observable
                # factor; judging the square against the base rtol fails at half headroom.
                # NOT rebased the way e_L is above: e_L2 is the same weighted mean
                # taken over ``e_L**2``, i.e. NON-NEGATIVE terms under positive
                # weights, so it cannot cancel (measured minima 5.69 at np=1 and
                # 26.09 at np=2 on this fixture). Its relative envelope is sound.
                e_L2_atol, e_L2_rtol = get_tolerance_squared_observable(atol, rtol)
                np.testing.assert_allclose(e_L2_debug, e_L2_jax, atol=e_L2_atol, rtol=e_L2_rtol)

        # average_projection_counter
        # Both GFMC_t and _GFMC_t_debug now store local averages per rank.
        apc_debug = gfmc_debug.average_projection_counter
        apc_jax = gfmc_jax.average_projection_counter
        assert not np.any(np.isnan(np.asarray(apc_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(apc_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(apc_debug, apc_jax, atol=atol, rtol=rtol)

        # E
        E_debug, E_err_debug, Var_debug, Var_err_debug = gfmc_debug.get_E(
            num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS,
            num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS,
        )
        E_jax, E_err_jax, Var_jax, Var_err_jax = gfmc_jax.get_E(
            num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS,
            num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS,
        )
        assert not np.any(np.isnan(np.asarray(E_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(E_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(E_debug, E_jax, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(E_err_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(E_err_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(E_err_debug, E_err_jax, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(Var_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(Var_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(Var_debug, Var_jax, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(Var_err_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(Var_err_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(Var_err_debug, Var_err_jax, atol=atol, rtol=rtol)

        # aF -- BOTH the production driver and the _GFMC_*_debug oracle use the single
        # ``_ANALYTIC_GRAD_ENABLED`` backend flag, so the HF force/stress parity compares
        # analytic-vs-analytic (fp64 ao_grad_lap on both sides even in mixed) and holds at medium in
        # BOTH precision modes. ``dltmove + Jastrow`` is analytic on both sides for MOLECULAR AND
        # PERIODIC (the closed-form ``delta`` correction ``gfmc.grad_e_L_t_wrt_*(non_local_move="dltmove")``
        # -- with no periodic carve-out: the FORCE reuses the min-image-aware
        # molecular delta kernel verbatim, the STRESS uses ``gfmc._dltmove_delta_strain_pbc``), so no
        # dltmove config is skipped.
        force_mean_debug, force_std_debug = gfmc_debug.get_aF(
            num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS,
            num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS,
        )
        force_mean_jax, force_std_jax = gfmc_jax.get_aF(
            num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS,
            num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS,
        )
        assert not np.any(np.isnan(np.asarray(force_mean_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(force_mean_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(force_mean_debug, force_mean_jax, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(force_std_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(force_std_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(force_std_debug, force_std_jax, atol=atol, rtol=rtol)

        jax.clear_caches()


class TestGFMCStressParity:
    """LRDMC stress (get_aT / get_aP) production-vs-debug parity.

    Stress counterpart of the ``get_aF`` checks in ``TestGFMCTau`` /
    ``TestGFMCBra``: the debug-vs-production parity route of the GFMC stress, next
    to the integration sum-rule / finite-diff tests. Production evaluates the
    strained kernels under ``vmap`` and the ``_debug`` classes loop per walker,
    so this guards the vmap/assembly path. Periodic fixture required.
    """

    @pytest.mark.heavy
    def test_gfmc_t_stress_debug_vs_production(self):
        """``_GFMC_t_debug.get_aT`` / ``get_aP`` must agree with ``GFMC_t``."""
        trexio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "lithium.h5")
        (
            structure_data,
            _,
            mos_data,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(trexio_file=trexio_file, store_tuple=True)

        jastrow_data = Jastrow_data(
            jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
                jastrow_1b_param=1.0,
                structure_data=structure_data,
                core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
                jastrow_1b_type="pade",
            ),
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
                jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure_data
            ),
            jastrow_three_body_data=Jastrow_three_body_data.init_jastrow_three_body_data(
                orb_data=mos_data, random_init=True, random_scale=1.0e-3, seed=123
            ),
        )
        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )

        num_mcmc_steps = GFMC_MIN_STEPS
        num_walkers = 2
        mcmc_seed = 3446
        alat = 0.30
        tau = 0.03
        num_gfmc_collect_steps = GFMC_MIN_COLLECT_STEPS

        gfmc_debug = _GFMC_t_debug(
            hamiltonian_data=hamiltonian_data,
            num_walkers=num_walkers,
            num_gfmc_collect_steps=num_gfmc_collect_steps,
            mcmc_seed=mcmc_seed,
            tau=tau,
            alat=alat,
            random_discretized_mesh=False,
            comput_position_deriv=True,
            comput_stress=True,
            non_local_move="tmove",
        )
        gfmc_debug.run(num_mcmc_steps=num_mcmc_steps)

        gfmc_jax = GFMC_t(
            hamiltonian_data=hamiltonian_data,
            num_walkers=num_walkers,
            num_gfmc_collect_steps=num_gfmc_collect_steps,
            mcmc_seed=mcmc_seed,
            tau=tau,
            alat=alat,
            random_discretized_mesh=False,
            comput_position_deriv=True,
            comput_stress=True,
            non_local_move="tmove",
        )
        gfmc_jax.run(num_mcmc_steps=num_mcmc_steps)

        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"),
            "strict",
        )

        with collective_check(mpi_comm, "GFMC_t prod-vs-debug get_aT / get_aP"):
            if mpi_rank == 0:
                sigma_mean_d, sigma_std_d = gfmc_debug.get_aT(
                    num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS, num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS
                )
                sigma_mean_p, sigma_std_p = gfmc_jax.get_aT(
                    num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS, num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS
                )
                assert sigma_mean_d.shape == (3, 3)
                assert not np.any(np.isnan(sigma_mean_d)), "NaN in debug sigma_mean"
                assert not np.any(np.isnan(sigma_mean_p)), "NaN in production sigma_mean"
                np.testing.assert_allclose(sigma_mean_d, sigma_mean_p, atol=atol, rtol=rtol)
                np.testing.assert_allclose(sigma_std_d, sigma_std_p, atol=atol, rtol=rtol)

                P_mean_d, P_std_d = gfmc_debug.get_aP(
                    num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS, num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS
                )
                P_mean_p, P_std_p = gfmc_jax.get_aP(
                    num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS, num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS
                )
                assert P_mean_d.shape == (3,)
                np.testing.assert_allclose(P_mean_d, P_mean_p, atol=atol, rtol=rtol)
                np.testing.assert_allclose(P_std_d, P_std_p, atol=atol, rtol=rtol)

        jax.clear_caches()

    @pytest.mark.heavy
    def test_gfmc_n_stress_debug_vs_production(self):
        """``_GFMC_n_debug.get_aT`` / ``get_aP`` must agree with ``GFMC_n``."""
        trexio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "lithium.h5")
        (
            structure_data,
            _,
            mos_data,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(trexio_file=trexio_file, store_tuple=True)

        jastrow_data = Jastrow_data(
            jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
                jastrow_1b_param=1.0,
                structure_data=structure_data,
                core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
                jastrow_1b_type="exp",
            ),
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
                jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure_data
            ),
            jastrow_three_body_data=Jastrow_three_body_data.init_jastrow_three_body_data(
                orb_data=mos_data, random_init=True, random_scale=1.0e-3, seed=123
            ),
        )
        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )

        num_mcmc_steps = GFMC_MIN_STEPS
        num_walkers = 2
        mcmc_seed = 3446
        E_scf = -1.00
        alat = 0.30
        num_mcmc_per_measurement = 10
        num_gfmc_collect_steps = GFMC_MIN_COLLECT_STEPS

        gfmc_debug = _GFMC_n_debug(
            hamiltonian_data=hamiltonian_data,
            num_walkers=num_walkers,
            num_mcmc_per_measurement=num_mcmc_per_measurement,
            num_gfmc_collect_steps=num_gfmc_collect_steps,
            mcmc_seed=mcmc_seed,
            E_scf=E_scf,
            alat=alat,
            random_discretized_mesh=True,
            comput_position_deriv=True,
            comput_stress=True,
            non_local_move="tmove",
        )
        gfmc_debug.run(num_mcmc_steps=num_mcmc_steps)

        gfmc_jax = GFMC_n(
            hamiltonian_data=hamiltonian_data,
            num_walkers=num_walkers,
            num_mcmc_per_measurement=num_mcmc_per_measurement,
            num_gfmc_collect_steps=num_gfmc_collect_steps,
            mcmc_seed=mcmc_seed,
            E_scf=E_scf,
            alat=alat,
            random_discretized_mesh=True,
            comput_position_deriv=True,
            comput_stress=True,
            non_local_move="tmove",
        )
        gfmc_jax.run(num_mcmc_steps=num_mcmc_steps)

        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"),
            "strict",
        )

        with collective_check(mpi_comm, "GFMC_n prod-vs-debug get_aT / get_aP"):
            if mpi_rank == 0:
                sigma_mean_d, sigma_std_d = gfmc_debug.get_aT(
                    num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS, num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS
                )
                sigma_mean_p, sigma_std_p = gfmc_jax.get_aT(
                    num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS, num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS
                )
                assert sigma_mean_d.shape == (3, 3)
                assert not np.any(np.isnan(sigma_mean_d)), "NaN in debug sigma_mean"
                assert not np.any(np.isnan(sigma_mean_p)), "NaN in production sigma_mean"
                np.testing.assert_allclose(sigma_mean_d, sigma_mean_p, atol=atol, rtol=rtol)
                np.testing.assert_allclose(sigma_std_d, sigma_std_p, atol=atol, rtol=rtol)

                P_mean_d, P_std_d = gfmc_debug.get_aP(
                    num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS, num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS
                )
                P_mean_p, P_std_p = gfmc_jax.get_aP(
                    num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS, num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS
                )
                assert P_mean_d.shape == (3,)
                np.testing.assert_allclose(P_mean_d, P_mean_p, atol=atol, rtol=rtol)
                np.testing.assert_allclose(P_std_d, P_std_p, atol=atol, rtol=rtol)

        jax.clear_caches()


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
