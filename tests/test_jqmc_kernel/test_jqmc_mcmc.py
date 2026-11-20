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

from jqmc._setting import ATOL_NONVACUITY_DISTINCTION, RTOL_NONVACUITY_DISTINCTION
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
# Add the project root directory to sys.path to allow executing this script directly
# This is necessary because relative imports (e.g. 'from ..jqmc') are not allowed
# when running a script directly (as __main__).
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import (
    configure,
    get_dtype_jnp,
    get_tolerance_min,
    get_tolerance_squared_observable,
)
from jqmc._setting import (
    ATOL_BIT_EXACT,
    ATOL_PYTHON_FP64_LOOSE,
    RTOL_PYTHON_FP64_LOOSE,
    ATOL_PYTHON_FP64_MEDIUM,
    RTOL_C9_GAMMA_SR_UPDATE,
    MCMC_MIN_BIN_BLOCKS,
    MCMC_MIN_STEPS,
    MCMC_MIN_WARMUP_STEPS,
    RTOL_BIT_EXACT,
    RTOL_PYTHON_FP64_ZERO,
    atol_consistency,
    rtol_consistency,
)
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.determinant import Geminal_data, compute_geminal_all_elements
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_DTN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.jqmc_mcmc import MCMC, _MCMC_debug
from jqmc.jqmc_optimizer_kernel import solve_linear_method
from jqmc.jqmc_optimizer import BlockDerivativeSource, ParameterSelection
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import ParameterTangentSpec, VariationalParameterBlock, Wavefunction_data
from tests._ecp_core_electrons import core_electrons_of
from tests._mpi_collective_check import collective_check

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


def _test_euclidean_block(name, values, *, state_token, sync_scope):
    arr = np.asarray(values)
    is_complex = np.issubdtype(arr.dtype, np.complexfloating)
    return VariationalParameterBlock.build(
        name=name,
        values=arr,
        field="complex" if is_complex else "real",
        derivative_policy="holomorphic" if is_complex else "direct_real",
        sync_scope=sync_scope,
        tangent_spec=ParameterTangentSpec.euclidean(state_token=state_token),
        state_token=state_token,
    )


def _test_exact_sources_from_legacy_matrix(self, *, blocks, num_warmup_steps=0, local_energy=False):
    """Adapt synthetic optimizer fixtures to the exact C8 source contract.

    The fixture matrix represents its chosen storage columns directly.  Tied
    blocks therefore publish ``(A, 0)`` so the private legacy adapter recovers
    the original synthetic matrix exactly; this is test data construction,
    not a production compatibility path.
    """
    del local_energy
    matrix = np.asarray(self.get_dln_WF(blocks=blocks, num_mcmc_warmup_steps=num_warmup_steps))
    sample_shape = matrix.shape[:-1]
    sources = []
    offset = 0
    for block in blocks:
        values = matrix[..., offset : offset + block.size].reshape(*sample_shape, *block.shape)
        offset += block.size
        if block.derivative_policy == "direct_real":
            source = BlockDerivativeSource.build(block=block, direct=values)
        elif block.derivative_policy == "holomorphic":
            source = BlockDerivativeSource.build(block=block, d_p=values)
        else:
            source = BlockDerivativeSource.build(block=block, d_p=values, d_pbar=np.zeros_like(values))
        sources.append(source)
    assert offset == matrix.shape[-1]
    return tuple(sources)


# (trexio_file, with_1b_jastrow, with_2b_jastrow, with_3b_jastrow)
# Rows marked ``heavy`` exceed the local 60s/test budget under
# JIT warm-up (3-body J3 + NN jastrow + complex Bloch JIT compile).
_heavy = pytest.mark.heavy
_light = pytest.mark.light
param_grid = [
    # (trexio_file, with_1b, with_2b, with_3b)
    pytest.param("H2_ae_ccpvdz_cart.h5", True, True, True, marks=_light),
    pytest.param("H_ae_ccpvdz_cart.h5", True, False, False, marks=_light),
    pytest.param("Li_ae_ccpvdz_cart.h5", False, False, False, marks=_light),
    # Open-shell (n_up=2, n_dn=1): exercises J2 num_up>1 dense pair path
    # under force evaluation (de_L/dr second-order AD).
    pytest.param("Li_ae_ccpvdz_cart.h5", True, True, True, marks=_light),
    pytest.param("N_ae_ccpvdz_cart.h5", False, False, False, marks=_light),
    # n_up=4, n_dn=3 with J2 only: J2 dense pair path on a larger
    # open-shell system (no J3/NN) for narrow regression detection.
    pytest.param("N_ae_ccpvdz_cart.h5", True, True, False, marks=_light),
    # -- Periodic ECP fixture: lithium.h5 (2-atom Li ccECP) --
    # Covers PBC + ECP + full Jastrow code path at a fraction of silicon's
    # memory (silicon force-grad ~67 GB/walker, OOMs on 256 GB host).
    pytest.param("lithium.h5", True, True, True, marks=_heavy),
    # -- Complex-Bloch fixture (twist at k_frac = (0.25, 0, 0)) --
    # Catches three bug classes (loader conj, SVD conj, post-SM walker wrap).
    # Bit-identical at Gamma, so a
    # twist fixture is the minimal addition exercising the complex code
    # paths. Jastrow off keeps per-walker memory below _MCMC_debug OOM.
    pytest.param("lithium_k025.h5", False, False, False, marks=_heavy),
]


# ---------------------------------------------------------------------------
# L3: VMC optimization loop -- symmetry preservation tests
# ---------------------------------------------------------------------------

# Test parameters: (j3_type, lambda_type)
_SYMMETRY_TEST_CASES = [
    # L3-1: baseline, both symmetric, all params
    ("sym", "square_sym"),
    # L3-5: j3 symmetric, lambda non-symmetric -> only j3 preserved
    ("sym", "square_nonsym"),
    # L3-6: j3 non-symmetric, lambda symmetric -> only lambda preserved
    ("nonsym", "square_sym"),
    # L3-7: both non-symmetric -> no symmetrization (no-op)
    ("nonsym", "square_nonsym"),
    # L3-8: j3 symmetric, rectangular lambda with paired-symmetric sub-block
    ("sym", "rect_paired_sym"),
]


# ---------------------------------------------------------------------------
# End-to-end optimization smoke tests (no monkeypatching)
# ---------------------------------------------------------------------------

# Each case is a dict of opt_* flags passed to run_optimize.
_E2E_OPT_CASES = [
    pytest.param(
        {"opt_J1_param": True, "opt_J2_param": False, "opt_J3_param": False, "opt_geminal": False},
        id="j1_only",
    ),
    pytest.param(
        {"opt_J1_param": True, "opt_J2_param": True, "opt_J3_param": True, "opt_geminal": True},
        id="j123_lambda",
    ),
    pytest.param(
        {
            "opt_J1_param": False,
            "opt_J2_param": False,
            "opt_J3_param": False,
            "opt_geminal": False,
            "opt_J3_basis_exp": True,
            "opt_J3_basis_coeff": True,
        },
        id="j3_basis_only",
    ),
    pytest.param(
        {
            "opt_J1_param": False,
            "opt_J2_param": False,
            "opt_J3_param": False,
            "opt_geminal": False,
            "opt_lambda_basis_exp": True,
            "opt_lambda_basis_coeff": True,
        },
        id="lambda_basis_only",
    ),
    pytest.param(
        {
            "opt_J1_param": True,
            "opt_J2_param": True,
            "opt_J3_param": True,
            "opt_geminal": True,
            "opt_J3_basis_exp": True,
            "opt_J3_basis_coeff": True,
            "opt_lambda_basis_exp": True,
            "opt_lambda_basis_coeff": True,
        },
        id="all_on",
    ),
]


# ---------------------------------------------------------------------------
# DTN-only SR end-to-end smoke tests
# ---------------------------------------------------------------------------

# (fixture, n_up, n_dn) -- parameterised to cover both Z=1 closed-shell (H2)
# and Z>1 open-shell (Li) all-electron paths through the MCMC + SR pipeline
# on a DTN-only Jastrow. Mirrors the cusp-preservation guarantees enforced
# in the pure DTN unit tests in test_jastrow_dtn.py, but exercises the full
# get_variational_blocks -> apply_block_updates -> SR-solver chain end to end.
_DTN_SR_FIXTURES = [
    pytest.param("H2_ae_ccpvtz_cart.h5", id="h2_ae"),
    pytest.param("Li_ae_ccpvdz_cart.h5", id="li_ae_open_shell"),
]


# ---------------------------------------------------------------------------
# LM end-to-end smoke test
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Debug vs Production: get_aH and solve_linear_method consistency
# ---------------------------------------------------------------------------


@pytest.mark.light
def test_ordinary_pytest_suite_forces_xla_dispatch():
    """The shared pytest policy must override even a hostile ambient Rust setup."""
    assert os.environ.get("JQMC_FLAG_RUST_ON_CPU") == "0"


@pytest.mark.light
def test_jax_coordinator_address_avoids_occupied_port_and_is_broadcast(monkeypatch):
    """The JAX coordinator address skips a held port and is identical on every rank.

    ``jax.distributed.initialize(cluster_detection_method="mpi4py")`` derives
    the port from ``hash(hostname)`` (4096 candidates), so two concurrent MPI
    jobs on one host could share a coordinator port and abort with
    ``Unexpected task registered``. The shared helper
    must instead return a port the OS reports free. Rank 0 therefore holds
    the first picked port open, like another job's live coordinator, and the
    next pick must differ while every rank receives the same address. Runs
    serially and under ``mpirun -np 2`` (broadcast across two ranks).
    """
    import errno
    import socket

    from mpi4py import MPI

    from jqmc._driver_helpers import _pick_jax_coordinator_address

    # The default path is the OS allocator; an explicit JAX pin bypasses it.
    monkeypatch.delenv("JAX_COORDINATOR_ADDRESS", raising=False)
    monkeypatch.delenv("JAX_COORDINATOR_PORT", raising=False)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank0_hostname = comm.bcast(socket.gethostname() if rank == 0 else None, root=0)

    first = _pick_jax_coordinator_address(comm)
    assert comm.allgather(first) == [first] * comm.Get_size(), "ranks received different coordinator addresses"
    host, port = first.rsplit(":", 1)
    assert host == rank0_hostname
    held_port = int(port)
    assert 0 < held_port < 65536

    occupant = None
    bind_errno = 0
    if rank == 0:
        occupant = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            occupant.bind(("", held_port))
            occupant.listen()
        except OSError as exc:
            bind_errno = exc.errno
    # EADDRINUSE: another process took the port first, so it is held either way.
    bind_errno = comm.bcast(bind_errno, root=0)
    assert bind_errno in (0, errno.EADDRINUSE), os.strerror(bind_errno)
    try:
        second = _pick_jax_coordinator_address(comm)
    finally:
        if occupant is not None:
            occupant.close()
    assert comm.allgather(second) == [second] * comm.Get_size(), "ranks received different coordinator addresses"
    second_host, second_port = second.rsplit(":", 1)
    assert second_host == rank0_hostname
    assert int(second_port) != held_port, (
        f"coordinator port {held_port} was handed out again while another listener held it; "
        "concurrent MPI jobs on this host would share one JAX coordination service"
    )


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


def _observe_well_conditioned_mcmc_boundary(
    boundary: str,
    r_up_batch: jax.Array | npt.ArrayLike,
    r_dn_batch: jax.Array | npt.ArrayLike,
    cond_g: npt.NDArray[np.float64],
    *,
    maximum_cond_g: float,
    num_steps: int,
    num_moves_per_measurement: int,
    Dt: float,
    seed: int,
) -> ElectronConfigurationReceipt:
    binding = ElectronConfigurationBinding(
        binding_id=f"ELECTRONS::MIXED-XLA-MCMC::{boundary}",
        mode="required",
        generation_pattern="direct",
        position_source_kind="seeded-random",
        generator_operation="jqmc.jqmc_mcmc.MCMC",
        generator_arguments=(
            ("boundary", boundary),
            ("num_steps", num_steps),
            ("num_mcmc_per_measurement", num_moves_per_measurement),
            ("Dt", Dt),
        ),
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
def test_mixed_xla_mcmc_debug_equals_production_in_well_conditioned_domain():
    """Actual mixed XLA ``MCMC`` equals ``_MCMC_debug`` end to end.

    This conditions production/debug parity on realized, explicitly screened
    geminal matrices.  The Li all-electron fixture exercises a 2x2 open-shell
    determinant with J1+J2+J3.  Exact PRNG and accept/reject counters, final
    walkers, and local-energy observables pin the stochastic trajectory.

    This same-policy witness does not claim full-vs-mixed deterministic
    proposal and decision replay, which remains a separate precision gap.
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

    num_steps = 4
    num_moves_per_measurement = 4
    mcmc_seed = 3446
    Dt = 0.5
    kwargs = dict(
        hamiltonian_data=hamiltonian,
        num_walkers=2,
        num_mcmc_per_measurement=num_moves_per_measurement,
        mcmc_seed=mcmc_seed,
        Dt=Dt,
        epsilon_AS=1.0e-6,
        random_discretized_mesh=True,
        comput_position_deriv=False,
    )
    debug = _MCMC_debug(**kwargs)
    production = MCMC(**kwargs)

    debug_r_up = debug._MCMC_debug__latest_r_up_carts
    debug_r_dn = debug._MCMC_debug__latest_r_dn_carts
    production_r_up = production._MCMC__latest_r_up_carts
    production_r_dn = production._MCMC__latest_r_dn_carts
    max_cond_g = 1.0e4
    initial_debug_cond = _full_precision_cond_g(geminal_data, debug_r_up, debug_r_dn)
    initial_production_cond = _full_precision_cond_g(geminal_data, production_r_up, production_r_dn)
    np.testing.assert_array_equal(initial_debug_cond, initial_production_cond)
    _observe_well_conditioned_mcmc_boundary(
        "initial-production",
        production_r_up,
        production_r_dn,
        initial_production_cond,
        maximum_cond_g=max_cond_g,
        num_steps=num_steps,
        num_moves_per_measurement=num_moves_per_measurement,
        Dt=Dt,
        seed=mcmc_seed,
    )
    assert np.all(initial_production_cond <= max_cond_g), (
        f"precondition failed before correctness evaluation: initial cond_2(G) "
        f"{initial_production_cond} exceeds {max_cond_g:.1e}"
    )

    debug.run(num_mcmc_steps=num_steps)
    production.run(num_mcmc_steps=num_steps)

    debug_r_up = debug._MCMC_debug__latest_r_up_carts
    debug_r_dn = debug._MCMC_debug__latest_r_dn_carts
    production_r_up = production._MCMC__latest_r_up_carts
    production_r_dn = production._MCMC__latest_r_dn_carts
    final_debug_cond = _full_precision_cond_g(geminal_data, debug_r_up, debug_r_dn)
    final_production_cond = _full_precision_cond_g(geminal_data, production_r_up, production_r_dn)
    _observe_well_conditioned_mcmc_boundary(
        "final-production",
        production_r_up,
        production_r_dn,
        final_production_cond,
        maximum_cond_g=max_cond_g,
        num_steps=num_steps,
        num_moves_per_measurement=num_moves_per_measurement,
        Dt=Dt,
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
        debug_r_up,
        production_r_up,
        atol=ATOL_PYTHON_FP64_MEDIUM,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    np.testing.assert_allclose(
        debug_r_dn,
        production_r_dn,
        atol=ATOL_PYTHON_FP64_MEDIUM,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    np.testing.assert_array_equal(debug._MCMC_debug__jax_PRNG_key_list, production._MCMC__jax_PRNG_key_list)
    np.testing.assert_array_equal(debug._MCMC_debug__accepted_moves, production._MCMC__accepted_moves)
    np.testing.assert_array_equal(debug._MCMC_debug__rejected_moves, production._MCMC__rejected_moves)
    np.testing.assert_allclose(debug.w_L, production.w_L, atol=atol, rtol=rtol)
    np.testing.assert_allclose(debug.e_L, production.e_L, atol=atol, rtol=rtol)
    np.testing.assert_allclose(debug.e_L2, production.e_L2, atol=atol, rtol=rtol)


class TestMCMCDriver:
    @pytest.mark.parametrize(
        "trexio_file,with_1b_jastrow,with_2b_jastrow,with_3b_jastrow",
        param_grid,
    )
    def test_jqmc_mcmc(self, trexio_file, with_1b_jastrow, with_2b_jastrow, with_3b_jastrow):
        """Test comparison with MCMC debug and MCMC production implementations.

        The Ewald regression: the production driver and the ``_MCMC_debug`` oracle
        must agree on ``w_L`` / ``e_L`` / the force and stress estimators.
        """
        # e_L / w_L cross ao_eval/jastrow_eval/det_eval/coulomb/wf_kinetic zones; the
        # achievable debug-vs-jax agreement is bounded by the weakest (fp32 in mixed).
        # BOTH drivers compute these from an IDENTICAL forward graph, so they agree to
        # fp64 even in mixed -> strict row suffices.
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"),
            "strict",
        )
        # The force (get_aF) parity. The _MCMC_debug oracle uses _ANALYTIC_GRAD_ENABLED,
        # the SAME analytic ao_grad_lap backend as production (autograd through the fp32
        # ao_eval forward would hit the fp32 representation limit in mixed): the parity is
        # analytic-vs-analytic, identical math differing only in loop structure (eager
        # per-walker vs jit(vmap)), and therefore must satisfy the strict tier in both
        # precision modes.
        force_atol, force_rtol = get_tolerance_min(
            ("ao_eval", "ao_grad_lap", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic"),
            "strict",
        )
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

        # Native-vs-XLA periodic e_L is also required to satisfy the strict tier;
        # do not mask an accumulation-order discrepancy with a broader tolerance.
        eL_atol, eL_rtol = atol, rtol

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
                jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure_data
            )

        jastrow_threebody_data = None
        jastrow_dtn_data = None
        if with_3b_jastrow:
            jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
                orb_data=mos_data, random_init=True, random_scale=1.0e-3, seed=123
            )
            # The molecular H2 all-Jastrow row is also the MCMC integration
            # oracle for DTN: production uses the cached fast update while
            # _MCMC_debug continues to recompute the full Jastrow value.
            if trexio_file == "H2_ae_ccpvdz_cart.h5":
                jastrow_dtn_data = Jastrow_DTN_data.build(
                    structure_data=structure_data,
                    z_eff_per_species=(1.0,),  # all-electron H2 fixture
                    pseudized_per_species=(False,),
                    use_u=True,
                    use_chi=True,
                    use_f=True,
                    cutoff_smoothness=4,
                    n_order_u=2,
                    n_order_chi=2,
                    n_order_eN=2,
                    n_order_ee=2,
                    L_u=2.0,
                    L_chi=2.0,
                    L_f=2.0,
                    random_init=True,
                    seed=124,
                )

        jastrow_data = Jastrow_data(
            jastrow_one_body_data=jastrow_onebody_data,
            jastrow_two_body_data=jastrow_twobody_data,
            jastrow_three_body_data=jastrow_threebody_data,
            jastrow_dtn_data=jastrow_dtn_data,
        )

        jastrow_data.sanity_check()

        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        wavefunction_data.sanity_check()

        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )
        hamiltonian_data.sanity_check()

        num_walkers = 2
        num_mcmc_steps = MCMC_MIN_STEPS
        mcmc_seed = 34356
        Dt = 2.0
        epsilon_AS = 1.0e-6

        # run VMC single-shot
        mcmc_debug = _MCMC_debug(
            hamiltonian_data=hamiltonian_data,
            Dt=Dt,
            mcmc_seed=mcmc_seed,
            epsilon_AS=epsilon_AS,
            num_walkers=num_walkers,
            comput_position_deriv=True,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )
        mcmc_debug.run(num_mcmc_steps=num_mcmc_steps)

        mcmc_jax = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=Dt,
            mcmc_seed=mcmc_seed,
            epsilon_AS=epsilon_AS,
            num_walkers=num_walkers,
            comput_position_deriv=True,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )
        mcmc_jax.run(num_mcmc_steps=num_mcmc_steps)

        # w_L
        w_L_debug = mcmc_debug.w_L
        w_L_jax = mcmc_jax.w_L
        assert not np.any(np.isnan(np.asarray(w_L_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(w_L_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(w_L_debug, w_L_jax, atol=atol, rtol=rtol)

        # e_L
        e_L_debug = mcmc_debug.e_L
        e_L_jax = mcmc_jax.e_L
        assert not np.any(np.isnan(np.asarray(e_L_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(e_L_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(e_L_debug, e_L_jax, atol=eL_atol, rtol=eL_rtol)

        # e_L2
        e_L2_debug = mcmc_debug.e_L2
        e_L2_jax = mcmc_jax.e_L2
        assert not np.any(np.isnan(np.asarray(e_L2_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(e_L2_jax))), "NaN detected in second argument"
        # e_L2 = e_L**2 doubles the relative error of e_L (d(x^2)/x^2 = 2 dx/x);
        # widen the base envelope's rtol by the central squared-observable factor.
        e_L2_atol, e_L2_rtol = get_tolerance_squared_observable(eL_atol, eL_rtol)
        np.testing.assert_allclose(e_L2_debug, e_L2_jax, atol=e_L2_atol, rtol=e_L2_rtol)

        # E
        E_debug, E_err_debug, Var_debug, Var_err_debug = mcmc_debug.get_E(
            num_mcmc_warmup_steps=MCMC_MIN_WARMUP_STEPS,
            num_mcmc_bin_blocks=MCMC_MIN_BIN_BLOCKS,
        )
        E_jax, E_err_jax, Var_jax, Var_err_jax = mcmc_jax.get_E(
            num_mcmc_warmup_steps=MCMC_MIN_WARMUP_STEPS,
            num_mcmc_bin_blocks=MCMC_MIN_BIN_BLOCKS,
        )
        assert not np.any(np.isnan(E_debug)), f"E_debug contains NaN: {E_debug}"
        assert not np.any(np.isnan(E_jax)), f"E_jax contains NaN: {E_jax}"
        assert not np.any(np.isnan(E_err_debug)), f"E_err_debug contains NaN: {E_err_debug}"
        assert not np.any(np.isnan(E_err_jax)), f"E_err_jax contains NaN: {E_err_jax}"
        assert not np.any(np.isnan(Var_debug)), f"Var_debug contains NaN: {Var_debug}"
        assert not np.any(np.isnan(Var_jax)), f"Var_jax contains NaN: {Var_jax}"
        assert not np.any(np.isnan(Var_err_debug)), f"Var_err_debug contains NaN: {Var_err_debug}"
        assert not np.any(np.isnan(Var_err_jax)), f"Var_err_jax contains NaN: {Var_err_jax}"
        assert not np.any(np.isnan(np.asarray(E_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(E_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(E_debug, E_jax, atol=eL_atol, rtol=eL_rtol)
        assert not np.any(np.isnan(np.asarray(E_err_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(E_err_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(E_err_debug, E_err_jax, atol=eL_atol, rtol=eL_rtol)
        assert not np.any(np.isnan(np.asarray(Var_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(Var_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(Var_debug, Var_jax, atol=eL_atol, rtol=eL_rtol)
        assert not np.any(np.isnan(np.asarray(Var_err_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(Var_err_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(Var_err_debug, Var_err_jax, atol=eL_atol, rtol=eL_rtol)

        # aF -- the _MCMC_debug oracle uses _ANALYTIC_GRAD_ENABLED (the SAME derivative
        # backend as production), so both drivers differentiate the identical forward graph
        # and the force/stress parity holds in mixed precision too.
        # aF
        force_mean_debug, force_std_debug = mcmc_debug.get_aF(
            num_mcmc_warmup_steps=MCMC_MIN_WARMUP_STEPS,
            num_mcmc_bin_blocks=MCMC_MIN_BIN_BLOCKS,
        )
        force_mean_jax, force_std_jax = mcmc_jax.get_aF(
            num_mcmc_warmup_steps=MCMC_MIN_WARMUP_STEPS,
            num_mcmc_bin_blocks=MCMC_MIN_BIN_BLOCKS,
        )
        assert not np.any(np.isnan(force_mean_debug)), f"force_mean_debug contains NaN: {force_mean_debug}"
        assert not np.any(np.isnan(force_mean_jax)), f"force_mean_jax contains NaN: {force_mean_jax}"
        assert not np.any(np.isnan(force_std_debug)), f"force_std_debug contains NaN: {force_std_debug}"
        assert not np.any(np.isnan(force_std_jax)), f"force_std_jax contains NaN: {force_std_jax}"
        assert not np.any(np.isnan(np.asarray(force_mean_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(force_mean_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(force_mean_debug, force_mean_jax, atol=force_atol, rtol=force_rtol)
        assert not np.any(np.isnan(np.asarray(force_std_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(force_std_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(force_std_debug, force_std_jax, atol=force_atol, rtol=force_rtol)

        jax.clear_caches()


class TestMCMCOptimize:
    @pytest.mark.parametrize(
        "regime,cg_flag",
        [
            ("wide", False),  # num_params < num_samples, direct solver
            ("wide", True),  # num_params < num_samples, CG solver
            ("tall", False),  # num_params >= num_samples, direct solver
            ("tall", True),  # num_params >= num_samples, CG solver
        ],
    )
    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvtz_cart.h5"])
    def test_sr_wide_and_tall_matrix(self, trexio_file, regime, cg_flag, monkeypatch):
        """SR optimization must run without error for both primal (wide) and dual (tall)
        matrix branches, with both the direct solver and CG solver.

        Wide matrix:  num_params < num_samples_total  (primal formulation)
        Tall matrix:  num_params >= num_samples_total  (dual / push-through identity)

        The test is MPI-aware: num_samples_total = num_mcmc * num_walkers * mpi_size,
        so the regime thresholds are computed accordingly.
        """
        from mpi4py import MPI as _MPI

        mpi_size = _MPI.COMM_WORLD.Get_size()

        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        # Minimal Jastrow (j1 + j2 only) to keep param count small and controllable.
        jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
            jastrow_1b_type="pade",
        )
        jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="pade")

        jastrow_data = Jastrow_data(
            jastrow_one_body_data=jastrow_onebody_data,
            jastrow_two_body_data=jastrow_twobody_data,
            jastrow_three_body_data=None,
        )
        jastrow_data.sanity_check()

        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        wavefunction_data.sanity_check()

        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )

        num_walkers = 2
        Dt = 2.0
        mcmc_seed = 12345
        epsilon_AS = 1.0e-6

        # Build base parameter registry -- j1, j2, lambda.
        # For the "tall" regime, pad lambda_matrix so that total_params stays
        # larger than num_samples_total even with multiple MPI ranks.
        base_params = {}
        if wavefunction_data.jastrow_data.jastrow_one_body_data is not None:
            base_params["j1_param"] = np.ones_like(
                np.array(wavefunction_data.jastrow_data.jastrow_one_body_data.jastrow_1b_param)
            )
        if wavefunction_data.jastrow_data.jastrow_two_body_data is not None:
            base_params["j2_param"] = np.ones_like(
                np.array(wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param)
            )

        num_mcmc = 1  # default for tall
        fixed_param_size = sum(v.size for v in base_params.values())

        if regime == "tall":
            # num_samples_total = num_mcmc * num_walkers * mpi_size
            # We need total_params >= num_samples_total, so pad lambda_matrix.
            min_samples_total = num_mcmc * num_walkers * mpi_size
            lambda_size_needed = max(1, min_samples_total - fixed_param_size + 1)
            base_params["lambda_matrix"] = np.ones(lambda_size_needed, dtype=float)
        else:
            if hasattr(wavefunction_data.geminal_data, "lambda_matrix"):
                base_params["lambda_matrix"] = np.ones_like(np.array(wavefunction_data.geminal_data.lambda_matrix))
            else:
                base_params["lambda_matrix"] = np.array([[2.0, -2.0], [3.0, -3.0]], dtype=float)

        total_params = sum(v.size for v in base_params.values())

        if regime == "wide":
            # num_samples_total = num_mcmc * num_walkers * mpi_size > total_params
            num_mcmc = total_params // (num_walkers * mpi_size) + 2

        num_samples_total = num_mcmc * num_walkers * mpi_size

        # Sanity-check the regime we configured (accounting for all MPI ranks).
        if regime == "wide":
            assert total_params < num_samples_total, (
                f"Expected wide (params < samples_total): {total_params} < {num_samples_total}"
            )
        else:
            assert total_params >= num_samples_total, (
                f"Expected tall (params >= samples_total): {total_params} >= {num_samples_total}"
            )

        # Deterministic fake data with non-trivial variance so X and F are non-zero.
        rng = np.random.default_rng(42)
        fake_w_L_data = np.ones((num_mcmc, num_walkers))
        fake_e_L_data = rng.standard_normal((num_mcmc, num_walkers)) * 0.1

        # -- monkeypatch helpers --------------------------------------------------
        params_registry: dict[int, dict[str, np.ndarray]] = {}

        def register_params(wf, params):
            params_registry[id(wf)] = params

        def lookup_params(wf):
            return params_registry[id(wf)]

        def fake_get_variational_blocks(
            self,
            opt_J1_param=True,
            opt_J2_param=True,
            opt_J3_param=True,
            opt_geminal=False,
            opt_J3_basis_exp=False,
            opt_J3_basis_coeff=False,
            opt_lambda_basis_exp=False,
            opt_lambda_basis_coeff=False,
            **_kwargs,
        ):
            blocks = []
            token = _kwargs["state_token"]
            pos = lookup_params(self)
            if opt_J1_param and "j1_param" in pos:
                arr = pos["j1_param"]
                blocks.append(_test_euclidean_block("j1_param", arr, state_token=token, sync_scope="world"))
            if opt_J2_param and "j2_param" in pos:
                arr = pos["j2_param"]
                blocks.append(_test_euclidean_block("j2_param", arr, state_token=token, sync_scope="world"))
            if opt_geminal and "lambda_matrix" in pos:
                arr = pos["lambda_matrix"]
                blocks.append(_test_euclidean_block("lambda_matrix", arr, state_token=token, sync_scope="sampler"))
            return blocks

        def fake_apply_optimization_updates(self, *, blocks, updates, expected_state_token):
            params = lookup_params(self)
            for block, update in zip(blocks, updates, strict=True):
                assert update.state_token == expected_state_token
                params[block.name] = params[block.name] + np.asarray(update.delta)
            return self

        def fake_run(self, num_mcmc_steps: int = 0, max_time=None):
            return None

        def fake_get_dln_WF(
            self,
            blocks,
            num_mcmc_warmup_steps=0,
            chosen_param_index=None,
        ):
            total = sum(block.size for block in blocks)
            rng_local = np.random.default_rng(123)
            return rng_local.standard_normal((num_mcmc, self.num_walkers, total)) * 0.01

        def fake_get_E(self, num_mcmc_warmup_steps: int = 0, num_mcmc_bin_blocks: int = 1):
            return (0.0, 0.0, 0.0, 0.0)

        def fake_get_gF(
            self,
            num_mcmc_warmup_steps,
            num_mcmc_bin_blocks,
            blocks,
            chosen_param_index=None,
        ):
            total = sum(block.size for block in blocks)
            return np.ones(total, dtype=float), np.ones(total, dtype=float)

        monkeypatch.setattr(Wavefunction_data, "get_variational_blocks", fake_get_variational_blocks, raising=False)
        monkeypatch.setattr(Wavefunction_data, "apply_optimization_updates", fake_apply_optimization_updates, raising=False)
        monkeypatch.setattr(MCMC, "run", fake_run, raising=False)
        monkeypatch.setattr(MCMC, "get_E", fake_get_E, raising=False)
        monkeypatch.setattr(MCMC, "get_gF", fake_get_gF, raising=False)
        monkeypatch.setattr(MCMC, "get_dln_WF", fake_get_dln_WF, raising=False)
        monkeypatch.setattr(MCMC, "_optimization_derivative_sources", _test_exact_sources_from_legacy_matrix)
        monkeypatch.setattr(MCMC, "w_L", property(lambda self: fake_w_L_data), raising=False)
        monkeypatch.setattr(MCMC, "e_L", property(lambda self: fake_e_L_data), raising=False)

        # -- run the test ---------------------------------------------------------
        mcmc = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=Dt,
            mcmc_seed=mcmc_seed,
            epsilon_AS=epsilon_AS,
            num_walkers=num_walkers,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )

        current_params = {k: v.copy() for k, v in base_params.items()}
        register_params(mcmc.hamiltonian_data.wavefunction_data, current_params)

        before = {k: v.copy() for k, v in current_params.items()}

        mcmc.run_optimize(
            num_mcmc_steps=num_mcmc,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=1,
            opt_J1_param=True,
            opt_J2_param=True,
            opt_J3_param=False,
            opt_geminal=True,
            optimizer_kwargs={
                "method": "sr",
                "delta": 1.0e-3,
                "epsilon": 1.0e-3,
                "cg_flag": cg_flag,
            },
        )

        # At least one param should have been updated (non-trivial SR solve).
        any_changed = any(not np.array_equal(before[k], current_params[k]) for k in before)
        assert any_changed, f"Expected at least one parameter to change (regime={regime}, cg_flag={cg_flag})"

        jax.clear_caches()

    @pytest.mark.parametrize(
        "regime,cg_flag",
        [
            ("wide", False),
            ("wide", True),
            ("tall", False),
            ("tall", True),
        ],
    )
    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvtz_cart.h5"])
    @pytest.mark.skip_mpi(reason="numerical-agreement test runs single-process only")
    def test_sr_device_matches_cpu(self, trexio_file, regime, cg_flag, monkeypatch):
        """Each of the four SR solve paths (wide/tall x direct/CG) must produce
        the same parameter update on the JAX-native device branch as on the
        legacy NumPy/SciPy/mpi4py CPU branch, given identical inputs.

        Single-process: ``psum`` is trivial and the device path reduces to its
        local computation; this test verifies numerical agreement only.
        Multi-rank validation requires actually running ``mpirun`` and is out of
        scope for the unit suite.
        """
        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
            jastrow_1b_type="pade",
        )
        jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="pade")
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=jastrow_onebody_data,
            jastrow_two_body_data=jastrow_twobody_data,
            jastrow_three_body_data=None,
        )
        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )

        num_walkers = 2
        Dt = 2.0
        mcmc_seed = 12345
        epsilon_AS = 1.0e-6

        # Build a parameter set sized for the requested regime. Single-process,
        # so num_samples_total = num_mcmc * num_walkers.
        base_params: dict[str, np.ndarray] = {
            "j1_param": np.ones_like(np.array(jastrow_onebody_data.jastrow_1b_param)),
            "j2_param": np.ones_like(np.array(jastrow_twobody_data.jastrow_2b_param)),
        }
        fixed_param_size = sum(v.size for v in base_params.values())

        if regime == "tall":
            num_mcmc = 1
            min_samples_total = num_mcmc * num_walkers
            # Pad lambda_matrix so num_params >= num_samples_total.
            lambda_size_needed = max(1, min_samples_total - fixed_param_size + 1)
            base_params["lambda_matrix"] = np.ones(lambda_size_needed, dtype=float)
        else:
            base_params["lambda_matrix"] = np.array([[2.0, -2.0], [3.0, -3.0]], dtype=float)
            total_params_tmp = sum(v.size for v in base_params.values())
            num_mcmc = total_params_tmp // num_walkers + 2

        total_params = sum(v.size for v in base_params.values())
        num_samples_total = num_mcmc * num_walkers
        if regime == "wide":
            assert total_params < num_samples_total, f"wide setup invalid: {total_params} >= {num_samples_total}"
        else:
            assert total_params >= num_samples_total, f"tall setup invalid: {total_params} < {num_samples_total}"

        rng = np.random.default_rng(42)
        fake_w_L_data = np.ones((num_mcmc, num_walkers))
        fake_e_L_data = rng.standard_normal((num_mcmc, num_walkers)) * 0.1

        params_registry: dict[int, dict[str, np.ndarray]] = {}

        def register_params(wf, params):
            params_registry[id(wf)] = params

        def lookup_params(wf):
            return params_registry[id(wf)]

        def fake_get_variational_blocks(
            self,
            opt_J1_param=True,
            opt_J2_param=True,
            opt_J3_param=True,
            opt_geminal=False,
            opt_J3_basis_exp=False,
            opt_J3_basis_coeff=False,
            opt_lambda_basis_exp=False,
            opt_lambda_basis_coeff=False,
            **_kwargs,
        ):
            blocks = []
            token = _kwargs["state_token"]
            pos = lookup_params(self)
            if opt_J1_param and "j1_param" in pos:
                arr = pos["j1_param"]
                blocks.append(_test_euclidean_block("j1_param", arr, state_token=token, sync_scope="world"))
            if opt_J2_param and "j2_param" in pos:
                arr = pos["j2_param"]
                blocks.append(_test_euclidean_block("j2_param", arr, state_token=token, sync_scope="world"))
            if opt_geminal and "lambda_matrix" in pos:
                arr = pos["lambda_matrix"]
                blocks.append(_test_euclidean_block("lambda_matrix", arr, state_token=token, sync_scope="sampler"))
            return blocks

        def fake_apply_optimization_updates(self, *, blocks, updates, expected_state_token):
            params = lookup_params(self)
            for block, update in zip(blocks, updates, strict=True):
                assert update.state_token == expected_state_token
                params[block.name] = params[block.name] + np.asarray(update.delta)
            return self

        def fake_run(self, num_mcmc_steps: int = 0, max_time=None):
            return None

        # Deterministic O matrix so both runs see identical inputs.
        def fake_get_dln_WF(
            self,
            blocks,
            num_mcmc_warmup_steps=0,
            chosen_param_index=None,
        ):
            total = sum(block.size for block in blocks)
            rng_local = np.random.default_rng(123)
            return rng_local.standard_normal((num_mcmc, self.num_walkers, total)) * 0.01

        def fake_get_E(self, num_mcmc_warmup_steps: int = 0, num_mcmc_bin_blocks: int = 1):
            return (0.0, 0.0, 0.0, 0.0)

        def fake_get_gF(
            self,
            num_mcmc_warmup_steps,
            num_mcmc_bin_blocks,
            blocks,
            chosen_param_index=None,
        ):
            total = sum(block.size for block in blocks)
            return np.ones(total, dtype=float), np.ones(total, dtype=float)

        monkeypatch.setattr(Wavefunction_data, "get_variational_blocks", fake_get_variational_blocks, raising=False)
        monkeypatch.setattr(Wavefunction_data, "apply_optimization_updates", fake_apply_optimization_updates, raising=False)
        monkeypatch.setattr(MCMC, "run", fake_run, raising=False)
        monkeypatch.setattr(MCMC, "get_E", fake_get_E, raising=False)
        monkeypatch.setattr(MCMC, "get_gF", fake_get_gF, raising=False)
        monkeypatch.setattr(MCMC, "get_dln_WF", fake_get_dln_WF, raising=False)
        monkeypatch.setattr(MCMC, "_optimization_derivative_sources", _test_exact_sources_from_legacy_matrix)
        monkeypatch.setattr(MCMC, "w_L", property(lambda self: fake_w_L_data), raising=False)
        monkeypatch.setattr(MCMC, "e_L", property(lambda self: fake_e_L_data), raising=False)

        def run_once(use_device_optimizer: bool):
            mcmc = MCMC(
                hamiltonian_data=hamiltonian_data,
                Dt=Dt,
                mcmc_seed=mcmc_seed,
                epsilon_AS=epsilon_AS,
                num_walkers=num_walkers,
                comput_position_deriv=False,
                comput_log_WF_param_deriv=True,
                comput_e_L_param_deriv=False,
                random_discretized_mesh=True,
            )
            params = {k: v.copy() for k, v in base_params.items()}
            register_params(mcmc.hamiltonian_data.wavefunction_data, params)
            mcmc.run_optimize(
                num_mcmc_steps=num_mcmc,
                num_opt_steps=1,
                num_mcmc_warmup_steps=0,
                num_mcmc_bin_blocks=1,
                opt_J1_param=True,
                opt_J2_param=True,
                opt_J3_param=False,
                opt_geminal=True,
                optimizer_kwargs={
                    "method": "sr",
                    "delta": 1.0e-3,
                    "epsilon": 1.0e-3,
                    "cg_flag": cg_flag,
                    "cg_max_iter": 200,
                    # CG iteration tolerance well below the project consistency
                    # tolerance, so the device-vs-CPU difference is dominated by
                    # float64 round-off, not CG residual.
                    "cg_tol": 1.0e-12,
                },
                use_device_optimizer=use_device_optimizer,
            )
            return params

        cpu_params = run_once(use_device_optimizer=False)
        dev_params = run_once(use_device_optimizer=True)

        # Both branches do exactly the same float64 SR arithmetic, just with
        # NumPy/SciPy/mpi4py vs JAX/shard_map; difference is round-off only.
        # Use the project's strict-float64 consistency tolerance.
        for key in cpu_params:
            cpu_v = cpu_params[key]
            dev_v = dev_params[key]
            # Sanity: CPU branch produced a non-trivial update.
            assert not np.array_equal(cpu_v, base_params[key]), f"baseline CPU update is trivial for {key}"
            np.testing.assert_allclose(
                dev_v,
                cpu_v,
                atol=atol_consistency,
                rtol=rtol_consistency,
                err_msg=f"device vs CPU mismatch for {key} (regime={regime}, cg={cg_flag})",
            )

        jax.clear_caches()

    @pytest.mark.parametrize(
        "regime,cg_flag",
        [
            ("wide", False),
            ("wide", True),
            ("tall", False),
            ("tall", True),
        ],
    )
    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvtz_cart.h5"])
    @pytest.mark.skip_serial(reason="multi-rank agreement test; the single-process variant is test_sr_device_matches_cpu")
    def test_sr_device_matches_cpu_multirank(self, trexio_file, regime, cg_flag, monkeypatch):
        """Multi-rank counterpart to ``test_sr_device_matches_cpu``.

        Verifies that under ``mpirun -n N>=2``:

        - The legacy CPU branch (``mpi_comm.Reduce`` / ``Allreduce`` / ``Alltoallv``
          via mpi4py) and
        - The device branch (``jax.lax.psum`` / ``all_gather`` via NCCL on GPU
          or Gloo on CPU, dispatched through ``shard_map``)

        produce numerically equivalent ``theta`` updates for all four SR
        paths (wide/tall x direct/CG).

        Each MPI rank is given *different* fake samples (via a rank-dependent
        seed) so that the cross-rank reduction has actual work to do; if the
        fixture-installed ``jax.distributed.initialize`` were missing, the
        device branch would silently produce per-rank-local results that
        wouldn't agree with the CPU branch's globally-aggregated result.

        Skipped on single-process runs (the single-process variant lives in
        ``test_sr_device_matches_cpu``).
        """
        from mpi4py import MPI as _MPI

        comm = _MPI.COMM_WORLD
        mpi_size = comm.Get_size()
        mpi_rank = comm.Get_rank()

        # Under mpirun -n >= 2, however, "JAX sees only one process" is a BROKEN RUN, not an
        # absent capability: the device branch would compute per-rank-local results and this
        # test's whole reason for existing would evaporate. It is therefore a
        # failure, not a pytest.skip: a skip would let a failed jax.distributed.initialize
        # (swallowed by the conftest fixture's bare ``except Exception: pass``) delete
        # the multi-rank verification silently.
        n_proc = jax.process_count()
        if n_proc < 2:
            pytest.fail(
                f"jax.process_count()={n_proc} under mpirun -n {mpi_size}: jax.distributed is not "
                "initialized, so jax.lax.psum / all_gather degenerate to per-rank no-ops. The "
                "conftest ``_jax_distributed_init`` fixture auto-inits under ``mpirun -n N pytest`` "
                "and hard-raises if it cannot -- reaching here means the fixture did not run at all "
                "(check that the conftest is loaded). Fix the init; do not skip."
            )

        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
            jastrow_1b_type="pade",
        )
        jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="pade")
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=jastrow_onebody_data,
            jastrow_two_body_data=jastrow_twobody_data,
            jastrow_three_body_data=None,
        )
        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )

        num_walkers = 2
        Dt = 2.0
        mcmc_seed = 12345
        epsilon_AS = 1.0e-6

        # Build a parameter set sized for the requested regime; use mpi_size in
        # the sample-count budget since the SR system sees all ranks' samples.
        base_params: dict[str, np.ndarray] = {
            "j1_param": np.ones_like(np.array(jastrow_onebody_data.jastrow_1b_param)),
            "j2_param": np.ones_like(np.array(jastrow_twobody_data.jastrow_2b_param)),
        }
        fixed_param_size = sum(v.size for v in base_params.values())

        if regime == "tall":
            num_mcmc = 1
            min_samples_total = num_mcmc * num_walkers * mpi_size
            lambda_size_needed = max(1, min_samples_total - fixed_param_size + 1)
            base_params["lambda_matrix"] = np.ones(lambda_size_needed, dtype=float)
        else:
            base_params["lambda_matrix"] = np.array([[2.0, -2.0], [3.0, -3.0]], dtype=float)
            total_params_tmp = sum(v.size for v in base_params.values())
            # Ensure num_mcmc * num_walkers * mpi_size > total_params_tmp.
            num_mcmc = max(1, total_params_tmp // (num_walkers * mpi_size) + 2)

        total_params = sum(v.size for v in base_params.values())
        num_samples_total = num_mcmc * num_walkers * mpi_size
        if regime == "wide":
            assert total_params < num_samples_total, f"wide setup invalid: {total_params} >= {num_samples_total}"
        else:
            assert total_params >= num_samples_total, f"tall setup invalid: {total_params} < {num_samples_total}"

        # Rank-dependent fake data: each rank sees different samples so the
        # cross-rank reduction is meaningful. Same seeds in both run_once calls
        # so CPU and device branches see identical inputs.
        fake_w_L_data = np.ones((num_mcmc, num_walkers))
        rng = np.random.default_rng(42 + mpi_rank)
        fake_e_L_data = rng.standard_normal((num_mcmc, num_walkers)) * 0.1

        params_holder: dict[str, dict[str, np.ndarray] | None] = {"params": None}

        def register_params(_wf, params):
            params_holder["params"] = params

        def lookup_params(_wf):
            return params_holder["params"]

        def fake_get_variational_blocks(
            self,
            opt_J1_param=True,
            opt_J2_param=True,
            opt_J3_param=True,
            opt_geminal=False,
            opt_J3_basis_exp=False,
            opt_J3_basis_coeff=False,
            opt_lambda_basis_exp=False,
            opt_lambda_basis_coeff=False,
            **_kwargs,
        ):
            blocks = []
            token = _kwargs["state_token"]
            pos = lookup_params(self)
            if opt_J1_param and "j1_param" in pos:
                arr = pos["j1_param"]
                blocks.append(_test_euclidean_block("j1_param", arr, state_token=token, sync_scope="world"))
            if opt_J2_param and "j2_param" in pos:
                arr = pos["j2_param"]
                blocks.append(_test_euclidean_block("j2_param", arr, state_token=token, sync_scope="world"))
            if opt_geminal and "lambda_matrix" in pos:
                arr = pos["lambda_matrix"]
                blocks.append(_test_euclidean_block("lambda_matrix", arr, state_token=token, sync_scope="sampler"))
            return blocks

        def fake_apply_optimization_updates(self, *, blocks, updates, expected_state_token):
            params = lookup_params(self)
            for block, update in zip(blocks, updates, strict=True):
                assert update.state_token == expected_state_token
                params[block.name] = params[block.name] + np.asarray(update.delta)
            return self

        def fake_run(self, num_mcmc_steps: int = 0, max_time=None):
            return None

        def fake_get_dln_WF(
            self,
            blocks,
            num_mcmc_warmup_steps=0,
            chosen_param_index=None,
        ):
            total = sum(block.size for block in blocks)
            rng_local = np.random.default_rng(123 + mpi_rank)
            return rng_local.standard_normal((num_mcmc, self.num_walkers, total)) * 0.01

        def fake_get_E(self, num_mcmc_warmup_steps: int = 0, num_mcmc_bin_blocks: int = 1):
            return (0.0, 0.0, 0.0, 0.0)

        def fake_get_gF(
            self,
            num_mcmc_warmup_steps,
            num_mcmc_bin_blocks,
            blocks,
            chosen_param_index=None,
        ):
            total = sum(block.size for block in blocks)
            return np.ones(total, dtype=float), np.ones(total, dtype=float)

        monkeypatch.setattr(Wavefunction_data, "get_variational_blocks", fake_get_variational_blocks, raising=False)
        monkeypatch.setattr(Wavefunction_data, "apply_optimization_updates", fake_apply_optimization_updates, raising=False)
        monkeypatch.setattr(MCMC, "run", fake_run, raising=False)
        monkeypatch.setattr(MCMC, "get_E", fake_get_E, raising=False)
        monkeypatch.setattr(MCMC, "get_gF", fake_get_gF, raising=False)
        monkeypatch.setattr(MCMC, "get_dln_WF", fake_get_dln_WF, raising=False)
        monkeypatch.setattr(MCMC, "_optimization_derivative_sources", _test_exact_sources_from_legacy_matrix)
        monkeypatch.setattr(MCMC, "w_L", property(lambda self: fake_w_L_data), raising=False)
        monkeypatch.setattr(MCMC, "e_L", property(lambda self: fake_e_L_data), raising=False)

        def run_once(use_device_optimizer: bool):
            mcmc = MCMC(
                hamiltonian_data=hamiltonian_data,
                Dt=Dt,
                mcmc_seed=mcmc_seed,
                epsilon_AS=epsilon_AS,
                num_walkers=num_walkers,
                comput_position_deriv=False,
                comput_log_WF_param_deriv=True,
                comput_e_L_param_deriv=False,
                random_discretized_mesh=True,
            )
            params = {k: v.copy() for k, v in base_params.items()}
            register_params(mcmc.hamiltonian_data.wavefunction_data, params)
            mcmc.run_optimize(
                num_mcmc_steps=num_mcmc,
                num_opt_steps=1,
                num_mcmc_warmup_steps=0,
                num_mcmc_bin_blocks=1,
                opt_J1_param=True,
                opt_J2_param=True,
                opt_J3_param=False,
                opt_geminal=True,
                optimizer_kwargs={
                    "method": "sr",
                    "delta": 1.0e-3,
                    "epsilon": 1.0e-3,
                    "cg_flag": cg_flag,
                    "cg_max_iter": 200,
                    "cg_tol": 1.0e-14,
                },
                use_device_optimizer=use_device_optimizer,
            )
            return params

        cpu_params = run_once(use_device_optimizer=False)
        dev_params = run_once(use_device_optimizer=True)

        # Both branches must agree to consistency tolerance on every rank.
        for key in cpu_params:
            cpu_v = cpu_params[key]
            dev_v = dev_params[key]
            assert not np.array_equal(cpu_v, base_params[key]), f"baseline CPU update is trivial for {key} (rank={mpi_rank})"
            np.testing.assert_allclose(
                dev_v,
                cpu_v,
                atol=atol_consistency,
                rtol=rtol_consistency,
                err_msg=(
                    f"device vs CPU multirank mismatch for {key} (regime={regime}, cg={cg_flag}, rank={mpi_rank}/{mpi_size})"
                ),
            )

        # Sanity: CPU branch's bcast / device branch's psum both replicate theta
        # across ranks, so the wf updates should agree across ranks too.
        rank0_cpu = comm.bcast({k: v.copy() for k, v in cpu_params.items()}, root=0)
        for key in cpu_params:
            np.testing.assert_allclose(
                cpu_params[key],
                rank0_cpu[key],
                atol=atol_consistency,
                rtol=rtol_consistency,
                err_msg=f"CPU branch theta differs across ranks for {key} (rank={mpi_rank})",
            )
        rank0_dev = comm.bcast({k: v.copy() for k, v in dev_params.items()}, root=0)
        for key in dev_params:
            np.testing.assert_allclose(
                dev_params[key],
                rank0_dev[key],
                atol=atol_consistency,
                rtol=rtol_consistency,
                err_msg=f"device branch theta differs across ranks for {key} (rank={mpi_rank})",
            )

        jax.clear_caches()

    @pytest.mark.parametrize(
        "lm_subspace_dim,cg_flag,num_mcmc,num_walkers",
        [
            # aSR (gamma scaling): smooth function, strict at any size.
            (0, False, 10, 2),
            (0, True, 10, 2),
            # Subspace LM (size 2 + SR collective = 3 dims): well-conditioned
            # once samples >> 3, so strict at 200 mcmc * 4 walkers = 800 samples.
            (2, False, 200, 4),
            (2, True, 200, 4),
        ],
    )
    @pytest.mark.light
    @pytest.mark.skip_mpi(reason="numerical-agreement test runs single-process only")
    def test_sr_lm_device_matches_cpu(self, lm_subspace_dim, cg_flag, num_mcmc, num_walkers, monkeypatch):
        """LM / aSR end-to-end optimization with ``use_device_optimizer``
        toggled.

        The device branch only replaces the SR direct/CG solve; everything
        downstream (``get_aH``, ``solve_linear_method``, aSR gamma) still runs
        on the CPU/mpi4py path.

        Tested LM modes (cf. ``run_optimize`` ``optimizer_kwargs``):
            - ``lm_subspace_dim = 0``: aSR (gamma from H_0/H_1/H_2/S_2)
            - ``lm_subspace_dim = N`` (positive small): subspace LM (top-N + SR collective)

        What is compared, and why:
            - aSR (``lm_subspace_dim = 0``): gamma scaling is a smooth function
              of the SR direction, so the final wf parameters depend
              continuously on ``theta_SR`` and can be compared at strict
              tolerance.
            - Subspace LM (``lm_subspace_dim != 0``): ``solve_linear_method``
              contains two argmax operations (dgelscut parameter elimination
              and ``argmax(|v_0|^2)`` eigenvector selection) that are
              discontinuous in their inputs -- a round-off-level perturbation
              can flip the selected mode and produce O(1e-3) jumps in the
              downstream output (final wf parameters, ``E_lm``, etc.) even
              though both branches run deterministically. Note in particular
              that ``E_lm = eigvals_lm[argmax(|v_0|^2)]`` is *not* a
              continuous function of ``H_bar``: the ranking by ``|v_0|^2``
              is unrelated to the eigenvalue ordering, so an argmax flip can
              jump ``E_lm`` by the gap between two arbitrary eigenvalues.
              Instead, compare the inputs to ``solve_linear_method``
              (``H_0, f_vec, S, K, B``) -- these depend continuously on
              ``theta_SR``, so they are the right boundary at which to
              verify that the device-branch SR solve agrees with the CPU
              branch. Whatever ``solve_linear_method`` does downstream
              (including any argmax flips) is shared CPU code and not part
              of what this test is meant to cover.

        Single optimization step only: ``num_opt_steps > 1`` would diverge the
        MCMC trajectories once round-off-level wf differences accumulate.
        """
        from jqmc import jqmc_optimizer

        trexio_file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ae_ccpvdz_cart.h5"
        )

        def build_mcmc():
            (
                structure_data,
                aos_data,
                _,
                _,
                geminal_mo_data,
                coulomb_potential_data,
            ) = read_trexio_file(trexio_file=trexio_file_path, store_tuple=True)

            jastrow_data = Jastrow_data(
                jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
                    jastrow_1b_param=1.0,
                    structure_data=structure_data,
                    core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
                    jastrow_1b_type="pade",
                ),
                jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
                    jastrow_2b_param=0.5, jastrow_2b_type="pade"
                ),
                jastrow_three_body_data=Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data),
            )
            wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
            hamiltonian_data = Hamiltonian_data(
                structure_data=structure_data,
                coulomb_potential_data=coulomb_potential_data,
                wavefunction_data=wavefunction_data,
            )
            return MCMC(
                hamiltonian_data=hamiltonian_data,
                Dt=2.0,
                mcmc_seed=12345,
                num_walkers=num_walkers,
                comput_position_deriv=False,
                comput_log_WF_param_deriv=True,
                comput_e_L_param_deriv=True,  # required by use_lm=True
            )

        # Pristine reference, captured before any monkeypatching so chained
        # spies (one per ``run_once`` call) all delegate to the real solver.
        orig_solve_lm_real_pencil = jqmc_optimizer.solve_lm_real_pencil

        # This test isolates device-vs-host solver agreement.  Some deliberately
        # aggressive subspace-LM fixtures leave the physical J1 domain, which the
        # atomic production commit rejects.  Apply the already-scaled typed
        # updates through an unconstrained test seam so that the
        # comparison still reaches the LM-pencil boundary it is intended to test.
        def apply_without_sanity(self, *, blocks, updates, expected_state_token):
            assert all(update.state_token == expected_state_token for update in updates)
            flat_update = np.concatenate([np.asarray(update.delta).reshape(-1) for update in updates])
            return self.apply_block_updates(list(blocks), flat_update, learning_rate=1.0)

        monkeypatch.setattr(
            Wavefunction_data,
            "apply_optimization_updates",
            apply_without_sanity,
        )

        def run_once(use_device_optimizer: bool):
            lm_inputs: list[dict] = []

            def spy(pencil, *, condition_cutoff):
                lm_inputs.append(
                    {
                        "H_0": float(pencil.H_0),
                        "H": np.asarray(pencil.H).copy(),
                        "S": np.asarray(pencil.S).copy(),
                    }
                )
                return orig_solve_lm_real_pencil(pencil, condition_cutoff=condition_cutoff)

            monkeypatch.setattr(jqmc_optimizer, "solve_lm_real_pencil", spy)

            mcmc = build_mcmc()
            mcmc.run_optimize(
                num_mcmc_steps=num_mcmc,
                num_opt_steps=1,
                num_mcmc_warmup_steps=0,
                num_mcmc_bin_blocks=1,
                opt_J1_param=True,
                opt_J2_param=True,
                opt_J3_param=True,
                opt_geminal=True,
                optimizer_kwargs={
                    "method": "sr",
                    "use_lm": True,
                    "lm_subspace_dim": lm_subspace_dim,
                    "lm_cond": 1.0e-3,
                    "delta": 0.1,
                    "epsilon": 1.0e-6,
                    "cg_flag": cg_flag,
                    # NB: cg_tol=1e-14 (near machine eps) is needed for the
                    # LM step to receive bit-comparable theta_SR from both
                    # branches. With cg_tol=1e-12, CG can early-terminate at
                    # mutually different points along the iteration trajectory,
                    # producing O(1e-3) differences that the LM step preserves.
                    "cg_max_iter": 2000,
                    "cg_tol": 1.0e-14,
                },
                use_device_optimizer=use_device_optimizer,
            )
            wf = mcmc.hamiltonian_data.wavefunction_data
            wf_params: dict[str, np.ndarray] = {}
            if wf.jastrow_data.jastrow_one_body_data is not None:
                wf_params["j1_param"] = np.asarray(wf.jastrow_data.jastrow_one_body_data.jastrow_1b_param)
            if wf.jastrow_data.jastrow_two_body_data is not None:
                wf_params["j2_param"] = np.asarray(wf.jastrow_data.jastrow_two_body_data.jastrow_2b_param)
            if wf.jastrow_data.jastrow_three_body_data is not None:
                wf_params["j3_matrix"] = np.asarray(wf.jastrow_data.jastrow_three_body_data.j_matrix)
            if wf.geminal_data is not None:
                geminal = wf.geminal_data
                if hasattr(geminal.orb_data_up_spin, "mo_coefficients"):
                    wf_params["mo_coefficients_up"] = np.asarray(geminal.orb_data_up_spin.mo_coefficients)
                    if not geminal.time_reversal:
                        wf_params["mo_coefficients_dn"] = np.asarray(geminal.orb_data_dn_spin.mo_coefficients)
                if hasattr(geminal, "lambda_diag"):
                    wf_params["lambda_diag"] = np.asarray(geminal.lambda_diag)
                    wf_params["lambda_unpaired"] = np.asarray(geminal.lambda_unpaired)
                elif hasattr(geminal, "lambda_matrix"):
                    wf_params["lambda_matrix"] = np.asarray(geminal.lambda_matrix)
            return wf_params, lm_inputs

        cpu_params, cpu_lm_inputs = run_once(use_device_optimizer=False)
        dev_params, dev_lm_inputs = run_once(use_device_optimizer=True)

        if lm_subspace_dim == 0:
            # aSR path: the GEP solver is not invoked; final wf params are
            # Lipschitz in theta_SR via gamma scaling.
            assert cpu_lm_inputs == [] and dev_lm_inputs == []
            for key in cpu_params:
                np.testing.assert_allclose(
                    dev_params[key],
                    cpu_params[key],
                    atol=atol_consistency,
                    rtol=rtol_consistency,
                    err_msg=(f"device vs CPU aSR mismatch for {key} (lm_subspace_dim={lm_subspace_dim}, cg_flag={cg_flag})"),
                )
        else:
            # Subspace LM: compare the centered real pencil at the GEP boundary.
            # These depend continuously on theta_SR, so they are the natural
            # boundary at which the device-branch SR solve can be verified
            # against the CPU branch. Anything past this point (E_lm, c_vec,
            # final wf params) goes through argmax operations inside
            # solve_linear_method and is not safe to compare strictly.
            assert len(cpu_lm_inputs) == len(dev_lm_inputs) > 0, (
                f"solve_lm_real_pencil was not invoked (cpu={len(cpu_lm_inputs)}, dev={len(dev_lm_inputs)})"
            )
            for step, (c_in, d_in) in enumerate(zip(cpu_lm_inputs, dev_lm_inputs)):
                for key in ("H_0", "H", "S"):
                    np.testing.assert_allclose(
                        d_in[key],
                        c_in[key],
                        atol=atol_consistency,
                        rtol=rtol_consistency,
                        err_msg=(
                            f"device vs CPU LM-input mismatch for {key} at step {step} "
                            f"(lm_subspace_dim={lm_subspace_dim}, cg_flag={cg_flag})"
                        ),
                    )

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("local_energy_dtype", ["real", "complex"])
    @pytest.mark.parametrize("regime", ["wide", "tall"])
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvtz_cart.h5"])
    @pytest.mark.skip_mpi(reason="numerical-agreement test runs single-process only")
    def test_sr_cg_run_optimize_device_regression(self, trexio_file, regime, local_energy_dtype, monkeypatch):
        """Exercise real and mixed-dtype device CG through ``run_optimize``.

        In the real cases, each iteration carries the previous step's solution as the
        initial guess (``sr_cg_warm_start_primal`` for wide, ``sr_cg_warm_start_dual``
        for tall). Both CPU and device branches must persist this state correctly
        so the final wf parameters agree to consistency tolerance.

        To make the warm-start path actually exercise the iteration-to-iteration
        carry, the fake ``O`` matrix is varied per call (using a counter that is
        reset between the two ``run_once`` invocations so both branches see the
        same input sequence). The one-step complex-local-energy cases pin the valid
        non-Gamma dtype regime where real-only parameter blocks give real ``X``
        while the per-sample ``F`` and the CG system are complex; like the real
        cases they run both branches and compare the device update with the CPU
        update at the same consistency tolerance, each precision mode
        judged within itself.
        """
        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
            jastrow_1b_type="pade",
        )
        jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="pade")
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=jastrow_onebody_data,
            jastrow_two_body_data=jastrow_twobody_data,
            jastrow_three_body_data=None,
        )
        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )

        num_walkers = 2
        Dt = 2.0
        mcmc_seed = 12345
        epsilon_AS = 1.0e-6
        num_opt_steps = 1 if local_energy_dtype == "complex" else 3

        base_params: dict[str, np.ndarray] = {
            "j1_param": np.ones_like(np.array(jastrow_onebody_data.jastrow_1b_param)),
            "j2_param": np.ones_like(np.array(jastrow_twobody_data.jastrow_2b_param)),
        }
        fixed_param_size = sum(v.size for v in base_params.values())

        if regime == "tall":
            num_mcmc = 1
            min_samples_total = num_mcmc * num_walkers
            lambda_size_needed = max(1, min_samples_total - fixed_param_size + 1)
            base_params["lambda_matrix"] = np.ones(lambda_size_needed, dtype=float)
        else:
            base_params["lambda_matrix"] = np.array([[2.0, -2.0], [3.0, -3.0]], dtype=float)
            total_params_tmp = sum(v.size for v in base_params.values())
            num_mcmc = total_params_tmp // num_walkers + 2

        fake_w_L_data = np.ones((num_mcmc, num_walkers))
        rng = np.random.default_rng(42)
        fake_e_L_data = rng.standard_normal((num_mcmc, num_walkers)) * 0.1
        if local_energy_dtype == "complex":
            fake_e_L_data = fake_e_L_data + 1j * rng.standard_normal((num_mcmc, num_walkers)) * 0.1

        # Single-slot holder for the live params dict. We can't key by ``id(wf)``
        # because ``MCMC.hamiltonian_data`` setter calls ``apply_diff_mask`` which
        # rewraps the wavefunction with a fresh instance every time it's reassigned
        # (i.e. at the end of every optimization iteration). The single-slot
        # approach assumes one MCMC instance is alive at a time inside this test.
        params_holder: dict[str, dict[str, np.ndarray] | None] = {"params": None}

        def register_params(_wf, params):
            params_holder["params"] = params

        def lookup_params(_wf):
            return params_holder["params"]

        # Counter that varies the fake O matrix per get_dln_WF call, so successive
        # SR systems differ and CG warm-start has actual work to do. Reset between
        # the two run_once invocations so both branches see identical input streams.
        call_idx = {"count": 0}

        def fake_get_variational_blocks(
            self,
            opt_J1_param=True,
            opt_J2_param=True,
            opt_J3_param=True,
            opt_geminal=False,
            opt_J3_basis_exp=False,
            opt_J3_basis_coeff=False,
            opt_lambda_basis_exp=False,
            opt_lambda_basis_coeff=False,
            **_kwargs,
        ):
            blocks = []
            token = _kwargs["state_token"]
            pos = lookup_params(self)
            if opt_J1_param and "j1_param" in pos:
                arr = pos["j1_param"]
                blocks.append(_test_euclidean_block("j1_param", arr, state_token=token, sync_scope="world"))
            if opt_J2_param and "j2_param" in pos:
                arr = pos["j2_param"]
                blocks.append(_test_euclidean_block("j2_param", arr, state_token=token, sync_scope="world"))
            if opt_geminal and "lambda_matrix" in pos:
                arr = pos["lambda_matrix"]
                blocks.append(_test_euclidean_block("lambda_matrix", arr, state_token=token, sync_scope="sampler"))
            return blocks

        def fake_apply_optimization_updates(self, *, blocks, updates, expected_state_token):
            params = lookup_params(self)
            for block, update in zip(blocks, updates, strict=True):
                assert update.state_token == expected_state_token
                delta = np.asarray(update.delta)
                if not np.iscomplexobj(params[block.name]) and np.iscomplexobj(delta):
                    delta = delta.real
                params[block.name] = params[block.name] + delta
            return self

        def fake_run(self, num_mcmc_steps: int = 0, max_time=None):
            return None

        def fake_get_dln_WF(
            self,
            blocks,
            num_mcmc_warmup_steps=0,
            chosen_param_index=None,
        ):
            call_idx["count"] += 1
            total = sum(block.size for block in blocks)
            rng_local = np.random.default_rng(123 + call_idx["count"])
            return rng_local.standard_normal((num_mcmc, self.num_walkers, total)) * 0.01

        def fake_get_E(self, num_mcmc_warmup_steps: int = 0, num_mcmc_bin_blocks: int = 1):
            return (0.0, 0.0, 0.0, 0.0)

        def fake_get_gF(
            self,
            num_mcmc_warmup_steps,
            num_mcmc_bin_blocks,
            blocks,
            chosen_param_index=None,
        ):
            total = sum(block.size for block in blocks)
            return np.ones(total, dtype=float), np.ones(total, dtype=float)

        monkeypatch.setattr(Wavefunction_data, "get_variational_blocks", fake_get_variational_blocks, raising=False)
        monkeypatch.setattr(Wavefunction_data, "apply_optimization_updates", fake_apply_optimization_updates, raising=False)
        monkeypatch.setattr(MCMC, "run", fake_run, raising=False)
        monkeypatch.setattr(MCMC, "get_E", fake_get_E, raising=False)
        monkeypatch.setattr(MCMC, "get_gF", fake_get_gF, raising=False)
        monkeypatch.setattr(MCMC, "get_dln_WF", fake_get_dln_WF, raising=False)
        monkeypatch.setattr(MCMC, "_optimization_derivative_sources", _test_exact_sources_from_legacy_matrix)
        monkeypatch.setattr(MCMC, "w_L", property(lambda self: fake_w_L_data), raising=False)
        monkeypatch.setattr(MCMC, "e_L", property(lambda self: fake_e_L_data), raising=False)

        def run_once(use_device_optimizer: bool):
            call_idx["count"] = 0  # reset so both branches see the same per-iter inputs
            mcmc = MCMC(
                hamiltonian_data=hamiltonian_data,
                Dt=Dt,
                mcmc_seed=mcmc_seed,
                epsilon_AS=epsilon_AS,
                num_walkers=num_walkers,
                comput_position_deriv=False,
                comput_log_WF_param_deriv=True,
                comput_e_L_param_deriv=False,
                random_discretized_mesh=True,
            )
            params = {k: v.copy() for k, v in base_params.items()}
            register_params(mcmc.hamiltonian_data.wavefunction_data, params)
            mcmc.run_optimize(
                num_mcmc_steps=num_mcmc,
                num_opt_steps=num_opt_steps,
                num_mcmc_warmup_steps=0,
                num_mcmc_bin_blocks=1,
                opt_J1_param=True,
                opt_J2_param=True,
                opt_J3_param=False,
                opt_geminal=True,
                optimizer_kwargs={
                    "method": "sr",
                    "delta": 1.0e-3,
                    "epsilon": 1.0e-3,
                    "cg_flag": True,
                    "cg_max_iter": 200,
                    "cg_tol": 1.0e-12,
                },
                use_device_optimizer=use_device_optimizer,
            )
            return params

        cpu_params = run_once(use_device_optimizer=False)
        dev_params = run_once(use_device_optimizer=True)

        for key in cpu_params:
            cpu_v = cpu_params[key]
            dev_v = dev_params[key]
            # Sanity: the warm-started CG produced a non-trivial param trail.
            assert not np.array_equal(cpu_v, base_params[key]), f"baseline CPU update is trivial for {key}"
            if local_energy_dtype == "complex":
                # The complex X / F / CG system must still give a real update of a real block.
                assert not np.array_equal(dev_v, base_params[key]), f"device update is trivial for {key}"
                assert not np.iscomplexobj(dev_v), f"real parameter block {key} received a complex update"
            np.testing.assert_allclose(
                dev_v,
                cpu_v,
                atol=atol_consistency,
                rtol=rtol_consistency,
                err_msg=(
                    f"device vs CPU CG warm-start mismatch for {key} (regime={regime}, local_energy_dtype={local_energy_dtype})"
                ),
            )

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("opt_flags", _E2E_OPT_CASES)
    def test_optimize_e2e_smoke(self, opt_flags):
        """End-to-end 1-step optimisation without monkeypatching.

        Verifies that the full pipeline (MCMC sampling -> gradient computation ->
        SR solve -> parameter update) runs without NaN/Inf for various flag combos.
        """
        trexio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ae_ccpvdz_cart.h5")
        (
            structure_data,
            aos_data,
            _,
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
                jastrow_2b_param=0.5, jastrow_2b_type="pade"
            ),
            jastrow_three_body_data=Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data),
        )

        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )

        num_walkers = 2
        mcmc = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=2.0,
            mcmc_seed=12345,
            num_walkers=num_walkers,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=False,
        )

        mcmc.run_optimize(
            num_mcmc_steps=10,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=1,
            optimizer_kwargs={"method": "sr", "delta": 1e-3, "epsilon": 1e-3},
            **opt_flags,
        )

        # Verify no NaN/Inf in any parameter after optimisation
        wf = mcmc.hamiltonian_data.wavefunction_data
        if wf.jastrow_data.jastrow_one_body_data is not None:
            assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_one_body_data.jastrow_1b_param)))
        if wf.jastrow_data.jastrow_two_body_data is not None:
            assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_two_body_data.jastrow_2b_param)))
        if wf.jastrow_data.jastrow_three_body_data is not None:
            assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_three_body_data.j_matrix)))
            assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_three_body_data.ao_exponents)))
            assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_three_body_data.ao_coefficients)))
        if wf.geminal_data is not None:
            # Canonical SD stores occupied MOs directly and has no lambda
            # coordinate.  AGP/AGPn retain their lambda fields, so validate
            # those only when the representation owns them and always pin the
            # direct MO coordinates shared by SD/AGPn.
            for spin in ("orb_data_up_spin", "orb_data_dn_spin"):
                orb = getattr(wf.geminal_data, spin)
                if hasattr(orb, "mo_coefficients"):
                    assert np.all(np.isfinite(np.asarray(orb.mo_coefficients)))
            for name in ("lambda_matrix", "lambda_diag", "lambda_unpaired"):
                if hasattr(wf.geminal_data, name):
                    assert np.all(np.isfinite(np.asarray(getattr(wf.geminal_data, name))))
            assert np.all(np.isfinite(np.asarray(wf.geminal_data.ao_exponents_up)))
            assert np.all(np.isfinite(np.asarray(wf.geminal_data.ao_exponents_dn)))
            assert np.all(np.isfinite(np.asarray(wf.geminal_data.ao_coefficients_up)))
            assert np.all(np.isfinite(np.asarray(wf.geminal_data.ao_coefficients_dn)))

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_fixture", _DTN_SR_FIXTURES)
    def test_dtn_sr_step_e2e(self, trexio_fixture):
        """End-to-end: 1 SR step on a DTN-only Jastrow.

        Asserts:
          * ``MCMC.run_optimize`` completes without raising on a DTN-only
            ``Jastrow_data`` (validates ``get_variational_blocks`` /
            ``apply_block_updates`` integration with the SR solver),
          * the DTN ``alpha_indep`` / ``beta_indep`` / ``gamma_indep`` arrays
            moved after the step (non-zero update),
          * the e-e cusp on the antiparallel (1/2) and parallel (1/4) channels
            still holds after SR -- the alpha block is universal in Z, so this
            cusp algebra must be invariant across both fixtures.
        """
        trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_fixture)
        if not os.path.exists(trexio_path):
            pytest.skip(f"TREXIO fixture {trexio_fixture} is not present.")

        structure_data, _aos, _mos_up, _mos_dn, geminal_mo_data, coulomb_potential_data = read_trexio_file(
            trexio_file=trexio_path,
            store_tuple=True,
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
        dtn = Jastrow_DTN_data.build(
            structure_data=structure_data,
            z_eff_per_species=z_eff_per_species,
            pseudized_per_species=pseudized_per_species,
            use_u=True,
            use_chi=True,
            use_f=True,
            cutoff_smoothness=4,
            n_order_u=4,
            n_order_chi=4,
            n_order_eN=2,
            n_order_ee=2,
            L_u=2.0,
            L_chi=2.0,
            L_f=2.0,
            random_init=True,
            seed=123,
        )
        wf = Wavefunction_data(
            geminal_data=geminal_mo_data,
            jastrow_data=Jastrow_data(jastrow_dtn_data=dtn),
        )
        ham = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wf,
        )

        alpha_pre = np.array(ham.wavefunction_data.jastrow_data.jastrow_dtn_data.alpha_indep)
        beta_pre = np.array(ham.wavefunction_data.jastrow_data.jastrow_dtn_data.beta_indep)
        gamma_pre = np.array(ham.wavefunction_data.jastrow_data.jastrow_dtn_data.gamma_indep)

        mcmc = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=42,
            epsilon_AS=1.0e-6,
            num_walkers=2,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=True,
            random_discretized_mesh=True,
        )
        mcmc.run_optimize(
            num_mcmc_steps=30,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=1,
            opt_J1_param=False,
            opt_J2_param=False,
            opt_J3_param=False,
            opt_geminal=False,
            optimizer_kwargs={
                "method": "sr",
                "delta": 1.0e-3,
                "epsilon": 1.0e-3,
                "cg_flag": True,
            },
        )

        dtn_post = mcmc.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_dtn_data
        assert dtn_post is not None, "DTN data should remain attached after SR step."
        alpha_post = np.array(dtn_post.alpha_indep)
        beta_post = np.array(dtn_post.beta_indep)
        gamma_post = np.array(dtn_post.gamma_indep)
        moved = (
            not np.allclose(alpha_pre, alpha_post, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION)
            or not np.allclose(beta_pre, beta_post, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION)
            or not np.allclose(gamma_pre, gamma_post, rtol=RTOL_NONVACUITY_DISTINCTION, atol=ATOL_NONVACUITY_DISTINCTION)
        )
        assert moved, "DTN independent coefficients did not change under SR."

        # Cusp algebra: alpha is the universal e-e Jastrow channel, so both
        # fixtures must satisfy u'_anti(0+) = 1/2 and u'_par(0+) = 1/4 after SR.
        M_alpha = np.asarray(dtn_post.M_alpha)
        b_alpha = np.asarray(dtn_post.b_alpha)
        alpha_full = np.einsum("ski,si->sk", M_alpha, alpha_post) + b_alpha
        L = float(dtn_post.L_u)
        C = dtn_post.cutoff_smoothness
        # Slope at 0+ of the evaluators' (L - r)^C sum_k a_k r^k form.
        cusp_anti = (L ** (C - 1)) * (L * alpha_full[1, 1] - C * alpha_full[1, 0])
        cusp_par = (L ** (C - 1)) * (L * alpha_full[0, 1] - C * alpha_full[0, 0])
        np.testing.assert_allclose(cusp_anti, 0.5, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(cusp_par, 0.25, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        jax.clear_caches()

    @pytest.mark.light
    def test_optimize_lm_e2e_smoke(self):
        """End-to-end LM optimisation -- verifies no NaN/Inf after 1 step."""
        trexio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ae_ccpvdz_cart.h5")
        (
            structure_data,
            aos_data,
            _,
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
                jastrow_2b_param=0.5, jastrow_2b_type="pade"
            ),
            jastrow_three_body_data=Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data),
        )

        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )

        mcmc = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=2.0,
            mcmc_seed=12345,
            num_walkers=2,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=True,
        )

        mcmc.run_optimize(
            num_mcmc_steps=10,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=1,
            opt_J1_param=True,
            opt_J2_param=True,
            opt_J3_param=True,
            opt_geminal=True,
            optimizer_kwargs={
                "method": "sr",
                "use_lm": True,
                "delta": 0.1,
                "epsilon": 1e-6,
                "lm_subspace_dim": 0,
            },
        )

        wf = mcmc.hamiltonian_data.wavefunction_data
        if wf.jastrow_data.jastrow_one_body_data is not None:
            assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_one_body_data.jastrow_1b_param)))
        if wf.jastrow_data.jastrow_two_body_data is not None:
            assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_two_body_data.jastrow_2b_param)))
        if wf.jastrow_data.jastrow_three_body_data is not None:
            assert np.all(np.isfinite(np.asarray(wf.jastrow_data.jastrow_three_body_data.j_matrix)))
        if wf.geminal_data is not None:
            for spin in ("orb_data_up_spin", "orb_data_dn_spin"):
                orb = getattr(wf.geminal_data, spin)
                if hasattr(orb, "mo_coefficients"):
                    assert np.all(np.isfinite(np.asarray(orb.mo_coefficients)))
            for name in ("lambda_matrix", "lambda_diag", "lambda_unpaired"):
                if hasattr(wf.geminal_data, name):
                    assert np.all(np.isfinite(np.asarray(getattr(wf.geminal_data, name))))

        jax.clear_caches()


# ---------------------------------------------------------------------------
# solve_linear_method unit tests
# ---------------------------------------------------------------------------


def _dense_original_basis_lm(H_0, f_vec, S_matrix, K_matrix, B_matrix):
    """Independent LM reference: the (p+1)x(p+1) generalized eigenproblem in the original basis.

    ``H_bar = [[H_0, -f^H/2], [-f/2, H]]`` with ``H`` the Hermitian part of
    ``K + B`` and ``S_bar = diag(1, S)``, solved densely by
    ``scipy.linalg.eigh(H_bar, S_bar)`` (no S-orthonormalization, no cutoff).
    Returns every eigenvalue, every ``|v_0|^2`` of the ``S_bar``-normalized
    eigenvectors and the max-``|v_0|^2`` index; ``c = v[1:] / v[0]`` is
    invariant under the eigenvector phase.
    """
    import scipy.linalg

    p = len(f_vec)
    H = K_matrix + B_matrix
    H = 0.5 * (H + H.conj().T)
    H_bar = np.zeros((p + 1, p + 1), dtype=np.result_type(H, f_vec))
    H_bar[0, 0] = H_0
    H_bar[1:, 0] = -0.5 * f_vec
    H_bar[0, 1:] = -0.5 * np.conj(f_vec)
    H_bar[1:, 1:] = H
    S_bar = np.zeros((p + 1, p + 1), dtype=S_matrix.dtype)
    S_bar[0, 0] = 1.0
    S_bar[1:, 1:] = S_matrix
    eigenvalues, vectors = scipy.linalg.eigh(H_bar, S_bar)
    v0_sq = np.abs(vectors[0, :]) ** 2
    best = int(np.argmax(v0_sq))
    return eigenvalues, v0_sq, best, vectors[1:, best] / vectors[0, best]


class TestSolveLinearMethod:
    """Unit tests for the isolated Hermitian LM numerical kernel."""

    @pytest.mark.light
    def test_trivial_2x2(self):
        """p=1: 2x2 eigenvalue problem. E_lm <= H_0."""
        H_0 = -1.0
        f_vec = np.array([0.2])
        S_matrix = np.array([[1.0]])
        K_matrix = np.array([[-0.5]])
        B_matrix = np.array([[-0.1]])
        c_vec, E_lm, v0_sq = solve_linear_method(H_0, f_vec, S_matrix, K_matrix, B_matrix, epsilon=1e-10)
        assert c_vec.shape == (1,)
        assert E_lm <= H_0 + 1e-10, f"E_lm={E_lm} should be <= H_0={H_0}"
        assert 0.0 <= v0_sq <= 1.0 + 1e-10

    @pytest.mark.light
    def test_diagonal_known_solution(self):
        """Diagonal H, S: the update and eigenvalue equal the dense original-basis LM solution."""
        p = 5
        H_0 = -2.0
        f_vec = np.random.default_rng(42).standard_normal(p)
        S_matrix = np.diag(np.linspace(0.1, 1.0, p))
        K_matrix = np.diag(np.linspace(-1.0, -0.1, p))
        B_matrix = np.diag(np.linspace(-0.5, -0.05, p))
        c_vec, E_lm, v0_sq = solve_linear_method(H_0, f_vec, S_matrix, K_matrix, B_matrix, epsilon=1e-10)
        assert c_vec.shape == (p,)
        assert np.all(np.isfinite(c_vec))
        assert np.isfinite(E_lm)
        assert 0.0 <= v0_sq <= 1.0 + 1e-10
        eigenvalues, v0_sq_reference, best, c_reference = _dense_original_basis_lm(H_0, f_vec, S_matrix, K_matrix, B_matrix)
        np.testing.assert_allclose(E_lm, eigenvalues[best], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(c_vec, c_reference, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(v0_sq, v0_sq_reference[best], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    def test_epsilon_cutoff(self):
        """S eigenvalues below epsilon are cut; p' < p."""
        p = 4
        H_0 = -1.0
        f_vec = np.ones(p) * 0.1
        S_matrix = np.diag([1.0, 0.5, 1e-8, 1e-10])
        K_matrix = np.eye(p) * (-0.5)
        B_matrix = np.eye(p) * (-0.1)
        c_vec, E_lm, v0_sq = solve_linear_method(H_0, f_vec, S_matrix, K_matrix, B_matrix, epsilon=1e-6)
        assert c_vec.shape == (p,)
        assert np.isfinite(E_lm)
        assert 0.0 <= v0_sq <= 1.0 + 1e-10

    @pytest.mark.light
    def test_all_diag_S_zero(self):
        """All diag(S) = 0 -> dgelscut removes all parameters -> zero update, E_lm == H_0."""
        p = 3
        H_0 = -1.5
        f_vec = np.ones(p) * 0.1
        S_matrix = np.zeros((p, p))
        K_matrix = np.eye(p) * (-0.5)
        B_matrix = np.eye(p) * (-0.1)
        c_vec, E_lm, v0_sq = solve_linear_method(H_0, f_vec, S_matrix, K_matrix, B_matrix, epsilon=1e-6)
        np.testing.assert_array_equal(c_vec, np.zeros(p))
        assert E_lm == H_0
        assert v0_sq == 0.0

    @pytest.mark.light
    def test_v0_max_selection(self):
        """The eigenvector with max |v_0|^2 is selected."""
        # Construct a case where the lowest eigenvector is not the one with max |v_0|^2
        p = 2
        H_0 = 0.0
        f_vec = np.array([0.01, 0.01])
        S_matrix = np.eye(p)
        K_matrix = np.diag([-10.0, -0.1])
        B_matrix = np.zeros((p, p))
        c_vec, E_lm, v0_sq = solve_linear_method(H_0, f_vec, S_matrix, K_matrix, B_matrix, epsilon=1e-10)
        assert c_vec.shape == (p,)
        assert np.isfinite(E_lm)
        assert 0.0 <= v0_sq <= 1.0 + 1e-10
        eigenvalues, v0_sq_reference, best, c_reference = _dense_original_basis_lm(H_0, f_vec, S_matrix, K_matrix, B_matrix)
        # The fixture is non-vacuous: the max-|v_0|^2 root is not the lowest eigenvalue.
        assert best != int(np.argmin(eigenvalues))
        np.testing.assert_allclose(E_lm, eigenvalues[best], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(c_vec, c_reference, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)


def _complex_force_reference(comm, weights, energies, log_derivatives):
    """Independent pinned complex generalized force and overlap over every rank of ``comm``.

    ``f_k = -2 <w conj(O_k - <O_k>) (e_L - <e_L>)>`` with ``<e_L>`` real-projected
    (the pinned convention of doc/notes/derivatives.md) and the Hermitian overlap
    ``S_kl = <w conj(O_k - <O_k>) (O_l - <O_l>)>``. Every weighted sum -- sum w,
    sum w e, sum w O and the centred covariance sums -- is reduced over ``comm`` with
    mpi4py's pickle-based object ``allreduce`` before ``f`` and ``S`` are formed, so the
    reference is the global estimator on any rank count and never uses production's
    buffer reduction helpers. On a one-rank communicator every reduction returns
    its local operand and the reference is the plain raw-sample formula.
    Returns ``(e_bar, f_ref, S_ref)``.
    """
    total_weight = comm.allreduce(np.real(np.sum(weights)))
    weighted_log_derivatives = comm.allreduce((weights[:, None] * log_derivatives).sum(0))
    weighted_energy = comm.allreduce((weights * energies).sum())
    log_derivative_mean = weighted_log_derivatives / total_weight
    e_bar = np.real(weighted_energy / total_weight)
    centred = log_derivatives - log_derivative_mean[None, :]
    covariance = comm.allreduce(np.einsum("i,i,ik->k", weights, (energies - e_bar), np.conj(centred)))
    overlap = comm.allreduce(np.einsum("i,ik,il->kl", weights, np.conj(centred), centred))
    return e_bar, -2.0 * covariance / total_weight, overlap / total_weight


class TestProdVsDebugLM:
    @pytest.mark.parametrize(
        "case",
        [
            pytest.param("molecular", marks=pytest.mark.light, id="molecular_H2_ae_ccpvdz"),
            # Complex-twist (k != 0) row: ``get_aH`` must keep Im on a
            # complex-twist wavefunction (no real cast, no real MPI.DOUBLE
            # buffers), or the LM optimizer is wrong at k != 0.
            # Heavy: complex-Bloch JIT compile + a well-sampled f-vs-get_gF run.
            pytest.param("twist", marks=pytest.mark.heavy, id="twist_H2_pbc_ae_k025"),
        ],
    )
    def test_get_aH_and_solve_lm_debug_vs_production(self, case):
        """Verify that _MCMC_debug.get_aH and solve_linear_method agree with MCMC versions.

        Both aSR (scalar) and LM (matrix) modes are tested, on a molecular
        (real / Gamma) fixture and on a complex-twist (k != 0) fixture whose
        ``e_L`` / ``de_L`` are complex. The twist row additionally pins the
        COMPLEX generalized force ``f = -2 <w conj(dO) de>`` (the pinned
        convention of doc/notes/derivatives.md -- no Re projection)
        against a raw-cache reconstruction and against ``MCMC.get_gF`` (an
        independent jackknife implementation), so a future Im-drop regression
        is caught even if the production and debug paths drift together.
        """
        if case == "molecular":
            trexio_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ae_ccpvdz_cart.h5"
            )
        else:
            trexio_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_pbc_ae_k025_cart.h5"
            )
        (
            structure_data,
            aos_data,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(trexio_file=trexio_file, store_tuple=True)

        # Molecular: J1 pade + J2 pade + J3. Twist: J1 pade + J2 pade (J3 dropped
        # -- J1 + J2 already drives the complex-``e_L`` reductions, and the
        # complex-``O_k`` cross term is exercised by the get_gF check below).
        jastrow_kwargs = dict(
            jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
                jastrow_1b_param=1.0,
                structure_data=structure_data,
                core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
                jastrow_1b_type="pade",
            ),
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
                jastrow_2b_param=0.5, jastrow_2b_type="pade", structure_data=structure_data
            ),
        )
        if case == "molecular":
            jastrow_kwargs["jastrow_three_body_data"] = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)
        jastrow_data = Jastrow_data(**jastrow_kwargs)

        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )

        num_walkers = 2
        num_mcmc_steps = 30
        mcmc_seed = 12345
        Dt = 2.0
        epsilon_AS = 1.0e-6
        warmup = 5

        # Create debug and production MCMC instances with derivative computation enabled
        mcmc_debug = _MCMC_debug(
            hamiltonian_data=hamiltonian_data,
            Dt=Dt,
            mcmc_seed=mcmc_seed,
            epsilon_AS=epsilon_AS,
            num_walkers=num_walkers,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=True,
        )
        mcmc_debug.run(num_mcmc_steps=num_mcmc_steps)

        mcmc_prod = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=Dt,
            mcmc_seed=mcmc_seed,
            epsilon_AS=epsilon_AS,
            num_walkers=num_walkers,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=True,
        )
        mcmc_prod.run(num_mcmc_steps=num_mcmc_steps)

        # Get variational blocks
        blocks = hamiltonian_data.wavefunction_data.get_variational_blocks(state_token=("test", 0))

        # C6 provider boundary: direct ``run`` retains both O and D as exact
        # source objects.  This covers the complex local-energy samples at the
        # twist without passing through a storage-space optimizer adapter.
        source_blocks = hamiltonian_data.wavefunction_data.get_variational_blocks(
            state_token=("c6-provider", 0),
            opt_geminal=True,
        )
        log_sources = mcmc_prod._optimization_derivative_sources(blocks=source_blocks)
        energy_sources = mcmc_prod._optimization_derivative_sources(blocks=source_blocks, local_energy=True)
        assert tuple(source.name for source in log_sources) == tuple(block.name for block in source_blocks)
        assert tuple(source.name for source in energy_sources) == tuple(block.name for block in source_blocks)
        mo_log_source = next(source for source in log_sources if source.name == "mo_coefficients_up")
        mo_energy_source = next(source for source in energy_sources if source.name == "mo_coefficients_up")
        if case == "twist":
            assert mo_log_source.direct is None
            assert mo_log_source.d_p is not None and mo_log_source.d_pbar is not None
            assert mo_energy_source.direct is None
            assert mo_energy_source.d_p is not None and mo_energy_source.d_pbar is not None
            assert np.any(np.abs(np.imag(mo_log_source.d_pbar)) > 0.0)
            assert np.any(np.abs(np.imag(mo_energy_source.d_pbar)) > 0.0)
            direct_energy_sources = [source.direct for source in energy_sources if source.direct is not None]
            assert direct_energy_sources
            assert any(np.any(np.abs(np.imag(values)) > 0.0) for values in direct_energy_sources)
        else:
            assert mo_log_source.direct is not None
            assert mo_energy_source.direct is not None

        # C8 provider boundary: the public optimization batch owns the exact
        # flattened O/D sources and the per-bin complex sufficient statistics.
        # Reconstruct every statistic directly from the native (step, walker)
        # grids so the test does not use the production binning helper as its
        # oracle.  Each walker remains a separate jackknife bin, matching the
        # established MCMC get_gF convention.
        batch_blocks = mcmc_prod.variational_blocks(ParameterSelection(opt_geminal=True))
        num_bin_blocks = 5
        batch = mcmc_prod.optimization_batch(
            num_warmup_steps=warmup,
            num_bin_blocks=num_bin_blocks,
            blocks=batch_blocks,
            need_local_energy_derivatives=True,
        )
        assert not hasattr(batch, "log_derivatives")
        assert batch.state_token == mcmc_prod.wavefunction_state_token
        assert batch.jackknife_bin_metadata.batch_token == batch.batch_token
        assert tuple(source.name for source in batch.log_derivative_sources) == tuple(block.name for block in batch_blocks)
        assert batch.local_energy_derivative_sources is not None
        assert tuple(source.name for source in batch.local_energy_derivative_sources) == tuple(
            block.name for block in batch_blocks
        )

        weights_grid = np.asarray(mcmc_prod.w_L[warmup:])
        energies_grid = np.asarray(mcmc_prod.e_L[warmup:])
        log_source_grids = mcmc_prod._optimization_derivative_sources(
            blocks=batch_blocks,
            num_warmup_steps=warmup,
        )
        energy_source_grids = mcmc_prod._optimization_derivative_sources(
            blocks=batch_blocks,
            num_warmup_steps=warmup,
            local_energy=True,
        )
        num_samples = weights_grid.size
        np.testing.assert_array_equal(batch.weights, weights_grid.reshape(num_samples))
        np.testing.assert_array_equal(batch.local_energies, energies_grid.reshape(num_samples))

        step_groups = np.array_split(np.arange(weights_grid.shape[0]), num_bin_blocks)
        expected_counts = np.concatenate([np.full(weights_grid.shape[1], group.size, dtype=np.int64) for group in step_groups])
        expected_weight_sums = np.concatenate([np.sum(weights_grid[group], axis=0) for group in step_groups])
        expected_weighted_energy_sums = np.concatenate(
            [np.sum(weights_grid[group] * energies_grid[group], axis=0) for group in step_groups]
        )
        metadata = batch.jackknife_bin_metadata
        np.testing.assert_array_equal(metadata.bin_sample_counts, expected_counts)
        np.testing.assert_allclose(
            metadata.bin_weight_sums,
            expected_weight_sums,
            atol=ATOL_BIT_EXACT,
            rtol=RTOL_BIT_EXACT,
        )
        np.testing.assert_allclose(
            metadata.bin_weighted_energy_sums,
            expected_weighted_energy_sums,
            atol=ATOL_BIT_EXACT,
            rtol=RTOL_BIT_EXACT,
        )

        def independently_binned(values, *, energy=False, conjugate=False):
            component = np.conj(values) if conjugate else values
            trailing_axes = component.ndim - 2
            weighted = weights_grid[(...,) + (None,) * trailing_axes] * component
            if energy:
                weighted = weighted * energies_grid[(...,) + (None,) * trailing_axes]
            return np.concatenate([np.sum(weighted[group], axis=0) for group in step_groups], axis=0)

        for block, log_grid, energy_grid, log_flat, energy_flat, moments in zip(
            batch_blocks,
            log_source_grids,
            energy_source_grids,
            batch.log_derivative_sources,
            batch.local_energy_derivative_sources,
            metadata.log_source_moments,
            strict=True,
        ):
            assert moments.name == block.name
            for component_name in ("direct", "d_p", "d_pbar"):
                raw_log = getattr(log_grid, component_name)
                raw_energy = getattr(energy_grid, component_name)
                flat_log = getattr(log_flat, component_name)
                flat_energy = getattr(energy_flat, component_name)
                if raw_log is None:
                    assert raw_energy is flat_log is flat_energy is None
                    continue
                np.testing.assert_array_equal(flat_log, raw_log.reshape(num_samples, *block.shape))
                np.testing.assert_array_equal(flat_energy, raw_energy.reshape(num_samples, *block.shape))
                np.testing.assert_allclose(
                    getattr(moments.source_sum, component_name),
                    independently_binned(raw_log),
                    atol=ATOL_BIT_EXACT,
                    rtol=RTOL_BIT_EXACT,
                )
                np.testing.assert_allclose(
                    getattr(moments.conjugate_source_energy_sum, component_name),
                    independently_binned(raw_log, energy=True, conjugate=True),
                    atol=ATOL_BIT_EXACT,
                    rtol=RTOL_BIT_EXACT,
                )

        next_batch = mcmc_prod.optimization_batch(
            num_warmup_steps=warmup,
            num_bin_blocks=num_bin_blocks,
            blocks=batch_blocks,
            need_local_energy_derivatives=False,
        )
        batch_identity, batch_sequence = batch.batch_token.rsplit(":", 1)
        next_identity, next_sequence = next_batch.batch_token.rsplit(":", 1)
        assert next_identity == batch_identity
        assert int(next_sequence) == int(batch_sequence) + 1
        assert next_batch.local_energy_derivative_sources is None

        # H_0/f/S/K/B cross the full e_L path + optimization assembly; bound by weakest zone.
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "local_energy"),
            "strict",
        )

        # --- Test 1: get_aH in LM mode (return_matrices=True) ---
        H_0_d, f_d, S_d, K_d, B_d, bra_d = mcmc_debug.get_aH(
            blocks=blocks,
            num_mcmc_warmup_steps=warmup,
            return_matrices=True,
        )
        H_0_p, f_p, S_p, K_p, B_p, bra_p = mcmc_prod.get_aH(
            blocks=blocks,
            num_mcmc_warmup_steps=warmup,
            return_matrices=True,
        )

        np.testing.assert_allclose(H_0_d, H_0_p, atol=atol, rtol=rtol)
        np.testing.assert_allclose(f_d, f_p, atol=atol, rtol=rtol)
        np.testing.assert_allclose(S_d, S_p, atol=atol, rtol=rtol)
        np.testing.assert_allclose(K_d, K_p, atol=atol, rtol=rtol)
        np.testing.assert_allclose(B_d, B_p, atol=atol, rtol=rtol)
        np.testing.assert_allclose(bra_d, bra_p, atol=atol, rtol=rtol)

        # --- Test 2: get_aH in aSR mode (return_matrices=False) ---
        # Use a simple direction vector g for the aSR scalar projection test
        K_params = len(f_d)
        g = np.random.default_rng(42).standard_normal(K_params)

        H_0_d2, H_1_d, H_2_d, S_2_d = mcmc_debug.get_aH(
            blocks=blocks,
            g=g,
            num_mcmc_warmup_steps=warmup,
            return_matrices=False,
        )
        H_0_p2, H_1_p, H_2_p, S_2_p = mcmc_prod.get_aH(
            blocks=blocks,
            g=g,
            num_mcmc_warmup_steps=warmup,
            return_matrices=False,
        )

        np.testing.assert_allclose(H_0_d2, H_0_p2, atol=atol, rtol=rtol)
        np.testing.assert_allclose(H_1_d, H_1_p, atol=atol, rtol=rtol)
        np.testing.assert_allclose(H_2_d, H_2_p, atol=atol, rtol=rtol)
        np.testing.assert_allclose(S_2_d, S_2_p, atol=atol, rtol=rtol)

        # --- Test 3: aSR scalars should be consistent with LM matrices ---
        # For the non-symmetric pencil, H_1 is the average of the bra-row and
        # ket-column projections. S_2 = g^T S g and H_2 = g^T (K+B) g.
        # The aSR scalars are PHYSICAL-REAL projections
        # (H_1/S_2/gKg/gBg = Re(...)); at a complex twist ``K`` is not Hermitian
        # (its ``e_L`` weight is complex), so ``g^T (K+B) g`` carries a finite Im
        # part that the scalar path drops. Compare against ``np.real`` of the
        # matrix forms -- identity on the real Gamma / molecular path.
        H_1_from_mat = 0.5 * np.real(np.dot(bra_d, g) - 0.5 * np.dot(g, f_d))
        S_2_from_mat = np.real(g @ S_d @ g)
        H_2_from_mat = np.real(g @ (K_d + B_d) @ g)
        np.testing.assert_allclose(H_1_d, H_1_from_mat, atol=atol, rtol=rtol)
        np.testing.assert_allclose(S_2_d, S_2_from_mat, atol=atol, rtol=rtol)
        np.testing.assert_allclose(H_2_d, H_2_from_mat, atol=atol, rtol=rtol)

        # --- Test 5 (twist only): the production generalized force ``f`` is the
        # COMPLEX covariance and matches MCMC.get_gF on a complex-``O_k`` block. ---
        # The debug-vs-prod parity above is NOT sufficient on its own: both
        # classes could drift together (both could drop Im the same way).
        # Under the pinned convention (doc/notes/derivatives.md)
        # ``f = -2 <w conj(dO)(e - <e>)>`` stays COMPLEX -- Im(f) drives the
        # imaginary DOF of complex parameters. This block pins ``get_aH``'s
        # ``f`` (a) bit-tight against a raw-cache reconstruction of that
        # formula and (b) against ``get_gF`` (independent jackknife estimator)
        # via a SELF-CALIBRATING Im guard: the two independent estimators must
        # agree on Im(f) better than Im(f) agrees with zero -- if either one
        # drops Im, the guard fails with no tuned constants involved. (The
        # deterministic, implementation-independent oracles live in
        # tests/test_jqmc_kernel/test_complex_opt_gt_*.py.)
        if case == "twist":
            mcmc_gf = MCMC(
                hamiltonian_data=hamiltonian_data,
                Dt=Dt,
                mcmc_seed=mcmc_seed,
                epsilon_AS=epsilon_AS,
                num_walkers=16,  # enough samples that the raw-vs-jackknife binning noise << cross term
                comput_position_deriv=False,
                comput_log_WF_param_deriv=True,
                comput_e_L_param_deriv=True,
            )
            mcmc_gf.run(num_mcmc_steps=200)
            warm_gf = 20
            # Include the complex-``O_k`` orbital block (opt_geminal=True).
            blocks_gf = hamiltonian_data.wavefunction_data.get_variational_blocks(
                state_token=("test", 0), opt_J1_param=True, opt_J2_param=True, opt_J3_param=False, opt_geminal=True
            )
            O = np.asarray(mcmc_gf.get_dln_WF(num_mcmc_warmup_steps=warm_gf, blocks=blocks_gf))
            # Non-vacuous: at least one genuinely complex-``O_k`` block is present.
            assert np.iscomplexobj(O) and float(np.max(np.abs(O.imag))) > atol, "expected a complex-O_k block at k != 0"

            # Production get_aH generalized force (COMPLEX covariance).
            _, f_aH, _, _, _, _ = mcmc_gf.get_aH(
                blocks=blocks_gf,
                num_mcmc_warmup_steps=warm_gf,
                return_matrices=True,
            )
            f_aH = np.asarray(f_aH).ravel()

            # Independent estimator: get_gF (complex, jackknife replicates).
            f_gF, _ = mcmc_gf.get_gF(num_mcmc_warmup_steps=warm_gf, num_mcmc_bin_blocks=10, blocks=blocks_gf)
            f_gF = np.asarray(f_gF).ravel()

            # Reconstruct the pinned COMPLEX covariance force directly from the
            # raw caches -- an independent code path (no optimizer assembly).
            # The caches are rank-local while ``get_aH`` returns the global
            # force, so the weighted sums are reduced over the driver's
            # communicator first (issue #860); one rank is unchanged.
            Of = O.reshape(O.shape[0] * O.shape[1], O.shape[2])
            w = np.ravel(mcmc_gf.w_L[warm_gf:])
            e = np.ravel(mcmc_gf.e_L[warm_gf:])
            _, f_ref, _ = _complex_force_reference(mcmc_gf.comm, w, e, Of)

            # Production get_aH must equal the complex reconstruction, Re AND Im:
            # a Re projection (the discarded convention) or a dropped conj fails
            # this immediately.
            assert np.iscomplexobj(f_aH) and np.iscomplexobj(f_gF), "generalized force lost its complex dtype"
            np.testing.assert_allclose(f_aH, f_ref, atol=atol, rtol=rtol)

            # Self-calibrating Im guard (data-derived, no tuned constants): the
            # two INDEPENDENT estimators (plain covariance vs jackknife) must
            # agree on Im(f) better than Im(f) agrees with zero. If either
            # estimator dropped Im, the left side would equal the right side
            # (or Im would vanish entirely) and the strict inequality fails.
            im_norm = float(np.linalg.norm(f_aH.imag))
            im_err = float(np.linalg.norm(f_aH.imag - f_gF.imag))
            assert im_norm > 0.0, "Im(f) vanished on a complex-O_k block"
            assert im_err < im_norm, (
                f"independent estimators disagree on Im(f) at the Im-vs-zero level: im_err={im_err:.2e} im_norm={im_norm:.2e}"
            )

        jax.clear_caches()

    @pytest.mark.light
    def test_get_aH_complex_twist_force_multirank_global_reference(self):
        """Two-rank ``get_aH`` complex force at k != 0 equals the global raw-sample reference.

        This light case is the two-rank evidence for the MPI reduction of the
        COMPLEX generalized force (the twist row above is heavy). It samples the committed
        H2 PBC all-electron k=(0.25,0.25,0.25) Cartesian MO wavefunction with
        J1 + J2 Pade (four walkers, twenty steps, five warmup steps, seed 12345)
        on every rank of COMM_WORLD (two ranks under ``mpirun -np 2``, where ranks
        draw different chains; one rank in a serial run). With the
        J1, J2 and complex MO-coefficient blocks, production ``MCMC.get_aH``
        (LM mode) must return ``H_0``, ``f`` and the overlap ``S`` equal to
        ``_complex_force_reference`` over the driver's communicator: sum w,
        sum w e, sum w O and the centred covariances are summed over ranks by
        mpi4py's object ``allreduce`` before ``f`` and ``S`` are formed. ``f``
        and ``S`` are compared as complex arrays, so Re and Im both count.
        Non-vacuity: a complex O_k block is present and Im(f) and Im(S) exceed
        the comparison tolerance; on more than one rank the rank-local
        references must also differ from the global ones on every rank, so a
        rank-local or Im-dropping reduction cannot pass. Same-mode strict
        tolerances of the e_L / O zones, as for the twist row. The two-rank run
        is the required evidence; on one rank the same comparisons check the
        single-communicator path.
        """
        from mpi4py import MPI as _MPI

        trexio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_pbc_ae_k025_cart.h5")
        (
            structure_data,
            _,
            _,
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
                jastrow_2b_param=0.5, jastrow_2b_type="pade", structure_data=structure_data
            ),
        )
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data),
        )

        warmup = 5
        mcmc = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=2.0,
            mcmc_seed=12345,
            epsilon_AS=1.0e-6,
            num_walkers=4,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=True,
        )
        mcmc.run(num_mcmc_steps=20)
        blocks = hamiltonian_data.wavefunction_data.get_variational_blocks(
            state_token=("test", 0), opt_J1_param=True, opt_J2_param=True, opt_J3_param=False, opt_geminal=True
        )

        # Production LM-mode H_0 and complex generalized force (MPI collectives).
        H_0, f_aH, S_aH, _, _, _ = mcmc.get_aH(blocks=blocks, num_mcmc_warmup_steps=warmup, return_matrices=True)
        f_aH = np.asarray(f_aH).ravel()
        S_aH = np.asarray(S_aH)

        # Independent references from the raw rank-local caches.
        O = np.asarray(mcmc.get_dln_WF(num_mcmc_warmup_steps=warmup, blocks=blocks))
        Of = O.reshape(O.shape[0] * O.shape[1], O.shape[2])
        w = np.ravel(mcmc.w_L[warmup:])
        e = np.ravel(mcmc.e_L[warmup:])
        e_bar, f_ref, S_ref = _complex_force_reference(mcmc.comm, w, e, Of)
        _, f_rank_local, S_rank_local = _complex_force_reference(_MPI.COMM_SELF, w, e, Of)
        max_im_O = mcmc.comm.allreduce(float(np.max(np.abs(np.imag(Of)))), op=_MPI.MAX)

        # Same zones and tier as the twist row's f_aH-vs-f_ref comparison.
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "local_energy"),
            "strict",
        )
        with collective_check(mcmc.comm, "multirank complex get_aH force"):
            assert np.iscomplexobj(Of) and max_im_O > atol, "expected a complex-O_k block at k != 0"
            assert np.iscomplexobj(f_aH), "generalized force lost its complex dtype"
            assert float(np.max(np.abs(f_ref.imag))) > atol + rtol * float(np.max(np.abs(f_ref))), (
                "Im(f) of the global reference is within tolerance of zero; the Im check would be vacuous"
            )
            assert float(np.max(np.abs(S_ref.imag))) > atol + rtol * float(np.max(np.abs(S_ref))), (
                "Im(S) of the global reference is within tolerance of zero; the Im check would be vacuous"
            )
            if mcmc.comm.Get_size() > 1:
                assert not np.allclose(f_rank_local, f_ref, atol=atol, rtol=rtol), (
                    "rank-local and global force references coincide; the multi-rank check would be vacuous"
                )
                assert not np.allclose(S_rank_local, S_ref, atol=atol, rtol=rtol), (
                    "rank-local and global overlap references coincide; the multi-rank check would be vacuous"
                )
            np.testing.assert_allclose(H_0, e_bar, atol=atol, rtol=rtol)
            np.testing.assert_allclose(f_aH, f_ref, atol=atol, rtol=rtol)
            np.testing.assert_allclose(S_aH, S_ref, atol=atol, rtol=rtol)

        jax.clear_caches()


class TestProdVsDebugStress:
    @pytest.mark.heavy
    def test_get_aT_aP_debug_vs_production(self):
        """Verify ``_MCMC_debug.get_aT`` / ``get_aP`` agree with the production MCMC.

        This is the stress counterpart of the force parity check
        (``get_aF`` debug-vs-production above) and closes a coverage gap
        (stress had no debug-vs-production
        parity route, only the integration sum-rule / finite-diff tests).

        Both routes compute the symmetric stress tensor by autodiff TODAY
        (production vmaps ``jax.jacrev`` of the strained kernels;
        ``_MCMC_debug`` loops the same ``jax.jacrev`` per walker), so this
        currently guards the vmap/assembly path. Once the production stress
        kernels are replaced by analytical forms, ``_MCMC_debug`` stays on
        autograd and this same test becomes the analytical-vs-auto whole-
        observable check. Periodic fixture required -- molecular structures
        short-circuit ``get_aT`` to zeros.
        """
        trexio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "lithium.h5")
        (
            structure_data,
            aos_data,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(trexio_file=trexio_file, store_tuple=True)

        # Li ccECP: 1 valence / 2 core per atom (matches the force sum-rule fixture).
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
                jastrow_1b_param=0.5,
                structure_data=structure_data,
                core_electrons=(2, 2),
                jastrow_1b_type="pade",
            ),
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
                jastrow_2b_param=0.5, jastrow_2b_type="exp", structure_data=structure_data
            ),
            jastrow_three_body_data=Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data),
        )
        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )

        num_walkers = 2
        num_mcmc_steps = 20
        mcmc_seed = 34356
        Dt = 2.0
        epsilon_AS = 1.0e-2
        warmup = 5
        bin_blocks = 5

        mcmc_debug = _MCMC_debug(
            hamiltonian_data=hamiltonian_data,
            Dt=Dt,
            mcmc_seed=mcmc_seed,
            epsilon_AS=epsilon_AS,
            num_walkers=num_walkers,
            comput_position_deriv=True,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
        )
        mcmc_debug.run(num_mcmc_steps=num_mcmc_steps)

        mcmc_prod = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=Dt,
            mcmc_seed=mcmc_seed,
            epsilon_AS=epsilon_AS,
            num_walkers=num_walkers,
            comput_position_deriv=True,
            comput_stress=True,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
        )
        mcmc_prod.run(num_mcmc_steps=num_mcmc_steps)

        # Stress crosses ao_eval/jastrow_eval/det_eval/coulomb/wf_kinetic zones
        # (same span as the force estimator); bound by the weakest zone.
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"),
            "strict",
        )

        sigma_mean_d, sigma_std_d = mcmc_debug.get_aT(num_mcmc_warmup_steps=warmup, num_mcmc_bin_blocks=bin_blocks)
        sigma_mean_p, sigma_std_p = mcmc_prod.get_aT(num_mcmc_warmup_steps=warmup, num_mcmc_bin_blocks=bin_blocks)

        assert sigma_mean_d.shape == (3, 3)
        assert sigma_mean_p.shape == (3, 3)
        assert not np.any(np.isnan(sigma_mean_d)), "NaN in debug sigma_mean"
        assert not np.any(np.isnan(sigma_mean_p)), "NaN in production sigma_mean"
        np.testing.assert_allclose(sigma_mean_d, sigma_mean_p, atol=atol, rtol=rtol)
        np.testing.assert_allclose(sigma_std_d, sigma_std_p, atol=atol, rtol=rtol)

        P_mean_d, P_std_d = mcmc_debug.get_aP(num_mcmc_warmup_steps=warmup, num_mcmc_bin_blocks=bin_blocks)
        P_mean_p, P_std_p = mcmc_prod.get_aP(num_mcmc_warmup_steps=warmup, num_mcmc_bin_blocks=bin_blocks)

        assert P_mean_d.shape == (3,)
        np.testing.assert_allclose(P_mean_d, P_mean_p, atol=atol, rtol=rtol)
        np.testing.assert_allclose(P_std_d, P_std_p, atol=atol, rtol=rtol)

        jax.clear_caches()


@pytest.mark.light
class TestMPCJointJackknife:
    """Regression: the MPC-corrected error bar must be covariance aware.

    ``E`` and ``dE_MPC`` are weighted ratio estimators over the SAME walkers and
    weights, so combining their separate error bars in quadrature is wrong. The
    driver exposes ``get_E_mpc``; because the two estimators share the
    denominator, its joint jackknife equals the jackknife of the summed
    numerator, which this test pins against an independent reference.
    """

    @staticmethod
    def _driver(e_L, w_L, mpc_dV):
        from mpi4py import MPI

        obj = MCMC.__new__(MCMC)
        obj._MCMC__stored_e_L = e_L
        obj._MCMC__stored_e_L2 = e_L**2
        obj._MCMC__stored_w_L = w_L
        obj._MCMC__stored_mpc_dV = mpc_dV
        obj._MCMC__mcmc_counter = e_L.shape[0]
        obj._MCMC__num_walkers = e_L.shape[1]
        obj._MCMC__mpc = True
        obj._mpi_comm = MPI.COMM_SELF
        return obj

    @staticmethod
    def _reference(e_L, w_L, mpc_dV, bins):
        def binned(x):
            return np.ravel([np.sum(a, axis=0) for a in np.array_split(x, bins, axis=0)])

        wb, web, wvb = binned(w_L), binned(w_L * e_L), binned(w_L * mpc_dV)
        M = wb.size
        e_jk = np.array([(web.sum() - web[m]) / (wb.sum() - wb[m]) for m in range(M)])
        v_jk = np.array([(wvb.sum() - wvb[m]) / (wb.sum() - wb[m]) for m in range(M)])
        joint = e_jk + v_jk
        std = lambda a: float(np.sqrt(M - 1) * np.std(a))  # noqa: E731
        return float(np.average(joint)), std(joint), float(np.hypot(std(e_jk), std(v_jk)))

    @pytest.mark.light
    @pytest.mark.parametrize("coupling", [-0.5, +0.5])
    def test_get_E_mpc_matches_joint_jackknife_and_rejects_quadrature(self, coupling):
        """Get E MPC matches joint jackknife and rejects quadrature."""
        rng = np.random.default_rng(7)
        steps, walkers, bins = 100, 4, 10
        e_L = rng.standard_normal((steps, walkers)) - 5.0
        # correlated correction: quadrature is wrong in BOTH directions
        mpc_dV = coupling * (e_L + 5.0) + 0.01 * rng.standard_normal((steps, walkers))
        w_L = rng.uniform(0.9, 1.1, (steps, walkers))

        driver = self._driver(e_L, w_L, mpc_dV)
        mean, std = driver.get_E_mpc(num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=bins)
        ref_mean, ref_std, quadrature = self._reference(e_L, w_L, mpc_dV, bins)

        np.testing.assert_allclose(mean, ref_mean, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_C9_GAMMA_SR_UPDATE)
        np.testing.assert_allclose(std, ref_std, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_C9_GAMMA_SR_UPDATE)
        # the quadrature combination must be clearly distinguishable
        assert abs(quadrature - ref_std) > 0.1 * ref_std, (quadrature, ref_std)

    @pytest.mark.light
    def test_get_E_mpc_refuses_without_mpc(self):
        """Get E MPC refuses without MPC."""
        rng = np.random.default_rng(3)
        e_L = rng.standard_normal((20, 2))
        driver = self._driver(e_L, np.ones((20, 2)), np.zeros((20, 2)))
        driver._MCMC__mpc = False
        with pytest.raises(ValueError, match="mpc=True"):
            driver.get_E_mpc(num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=5)


@pytest.mark.light
def test_acceptance_ratio_property():
    """MCMC.acceptance_ratio = accepted / (accepted + rejected); None when no moves.

    Prerequisite of the workflow's Dt auto setting: the workflow reads the global
    ratio from the ``Acceptance ratio is X.XX %`` log line, but this rank-local
    property is the programmatic/test-facing accessor. Uses ``__new__`` to avoid
    the heavy MCMC constructor -- only the two move counters are exercised.
    """
    obj = MCMC.__new__(MCMC)
    obj._MCMC__accepted_moves = 1
    obj._MCMC__rejected_moves = 3
    assert obj.acceptance_ratio == 0.25  # exactly representable, no tolerance needed

    obj._MCMC__accepted_moves = 0
    obj._MCMC__rejected_moves = 0
    assert obj.acceptance_ratio is None


@pytest.mark.light
def test_get_gF_requires_blocks():
    """MCMC.get_gF has no implicit block list: omitting ``blocks`` fails at the call.

    The buffers are populated so that a defaulted ``blocks=None`` would reach
    the block loop and fail there; the required keyword-only
    argument makes Python reject the call before the body runs.
    """
    obj = MCMC.__new__(MCMC)
    obj._MCMC__stored_w_L = np.ones((4, 1))
    obj._MCMC__stored_e_L = np.zeros((4, 1))
    obj._MCMC__stored_log_WF_param_grads = {}
    obj._MCMC__mcmc_counter = 4
    obj._MCMC__num_walkers = 1
    with pytest.raises(TypeError, match=r"get_gF\(\) missing 1 required keyword-only argument: 'blocks'"):
        obj.get_gF(num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=2)


@pytest.mark.heavy
def test_restart_policy_matches_checkpoint(tmp_path):
    """Drift guard: TOML_TO_CFG stays in sync with MCMC.save_to_hdf5's driver_config.

    Build a real H2 MCMC, serialise it, and cross-check the
    checkpoint's driver_config keys against TOML_TO_CFG so that
      (a) every mapped (non-sentinel) cfg target is a real driver_config key, and
      (b) every real driver_config key is either mapped or explicitly listed as
          internal/deferred -- a newly-added checkpoint field cannot silently
          escape restart classification.
    """
    import h5py

    from jqmc._restart_policy import TOML_TO_CFG

    trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ae_ccpvdz_cart.h5")
    (structure_data, _, _mos, _, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
        trexio_file=trexio_path, store_tuple=True
    )
    wavefunction_data = Wavefunction_data(
        jastrow_data=Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None),
        geminal_data=geminal_mo_data,
    )
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )
    mcmc = MCMC(hamiltonian_data=hamiltonian_data, num_walkers=2, mcmc_seed=1)

    ckpt = str(tmp_path / "_rank0.h5")
    mcmc.save_to_hdf5(ckpt)
    with h5py.File(ckpt, "r") as f:
        cfg_keys = set(f["driver_config"].attrs.keys())

    # (a) every mapped, non-sentinel cfg target exists in the real driver_config.
    #
    # Scoped to the driver actually built here. TOML_TO_CFG was extended from
    # MCMC-only to other job sections
    # because the frozen-key guard had been wired to MCMC alone; an MCMC
    # checkpoint has no `alat` or `tau`, so demanding every section's targets
    # from it asserts something that was never true. The LRDMC and VMC halves
    # need their own driver_config guard against their own checkpoints -- see
    # the follow-up below; they are NOT covered by this test.
    _MCMC_SECTIONS = ("control", "mcmc", "coulomb")
    checked = 0
    for (section, key), cfg_key in TOML_TO_CFG.items():
        if cfg_key.startswith("<") or section not in _MCMC_SECTIONS:
            continue
        checked += 1
        assert cfg_key in cfg_keys, f"TOML_TO_CFG[{section}.{key}]={cfg_key!r} absent from driver_config {sorted(cfg_keys)}"
    assert checked >= 7, f"the MCMC-section drift guard degenerated to {checked} checked keys"

    # (b) every real driver_config key is mapped or explicitly internal/deferred.
    mapped_targets = {v for v in TOML_TO_CFG.values() if not v.startswith("<")}
    known_unmapped = {
        "comput_e_L_param_deriv",  # internal: derived flag, no TOML key
        "retain_derivative_sources",  # internal: checkpoint-restored storage policy, no TOML key
        "random_discretized_mesh",  # internal
        "mcmc_counter",
        "accepted_moves",
        "rejected_moves",
        "i_opt",  # counters / run state
    }
    for cfg_key in cfg_keys:
        if cfg_key.startswith("param_grad_flag_"):
            continue  # per-block gradient flags, internal
        assert cfg_key in mapped_targets or cfg_key in known_unmapped, (
            f"driver_config key {cfg_key!r} is unclassified: add it to a *_restart_policy "
            f"+ TOML_TO_CFG, or to known_unmapped with a reason"
        )
