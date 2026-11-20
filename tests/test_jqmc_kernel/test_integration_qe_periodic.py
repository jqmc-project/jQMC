"""Short integration tests for QE-derived periodic Hamiltonians.

For each QE-projection kind (currently ``pw`` and ``pgto+pw``) this module
runs a tiny MCMC, GFMC_t, GFMC_n, and LRDMC (GFMC_n without forces) on the
Cu QE sample (1 atom, 19 valence electrons after the ccECP-soft 10-core
auto-deduction). The checks are intentionally weak: the driver completes
and the energy expectation value comes back finite. The aim is to catch
gross plumbing regressions in the QE -> projection -> jqmc -> driver
pipeline; numerical accuracy is the concern of the existing silicon.h5
tests and of the QE-side reference tests.

To keep wall time low we use only 2 walkers, very few MCMC / GFMC steps,
and skip the position-derivative path: ``comput_position_deriv``
requires ``with_param_grad_mask``, which is defined only for the GTO
family (it touches per-primitive exponents), so a pure-PW Hamiltonian
cannot drive the full LRDMC force estimator. The bare LRDMC E expectation
is still tested.
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
from pathlib import Path

import jax
import numpy as np
import pytest

from tests._test_data_cache import qe_example_files_root
from tests.test_jqmc_kernel._gdd_fd_adapters import fd_case_parameters, record_fd_case

pytestmark = [pytest.mark.with_chdir]
from jqmc._precision import get_tolerance_min

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

_SAVE_DIR = qe_example_files_root() / "Li_gamma" / "out" / "li.save"
_XML = _SAVE_DIR / "data-file-schema.xml"
_WFC = _SAVE_DIR / "wfc1.hdf5"
_UPF = _SAVE_DIR / "Li.ccECP.upf"

_available = _XML.is_file() and _WFC.is_file() and _UPF.is_file()
requires_qe = pytest.mark.skipif(not _available, reason="QE Li_gamma sample not present")


# ---------------------------------------------------------------------------
# fixtures: build Hamiltonian_data in-memory (no h5 round-trip)
# ---------------------------------------------------------------------------


_COMMON_HAM_KWARGS = dict(
    j1_param=None,
    j2_param=1.0,
    j2_type="pade",
    ewald_tol=1.0e-8,
    # Explicit per-atom core charges (2 Li atoms, ccECP z_core = 2) with no ECP
    # record: the core-charge-only block of these fixtures. It is not inferred
    # when the ECP source is missing (issue #99).
    ecp_z_cores=(2.0, 2.0),
)


def _build_ham_pw():
    """Build a Hamiltonian_data for kind='pw' (QE Li reference)."""
    from jqmc.pw_pgto_projection import project_pw_to_pw
    from jqmc.qe_to_jqmc import build_hamiltonian_from_projection
    from jqmc.qe_wrapper import load_qe_reference

    ref = load_qe_reference(str(_SAVE_DIR))
    proj = project_pw_to_pw(ref, g_cut=None)
    return build_hamiltonian_from_projection(ref, proj, kind="pw", **_COMMON_HAM_KWARGS)


def _build_ham_bspline():
    """Build a Hamiltonian_data for kind='bspline' (QE Li reference)."""
    from jqmc.pw_pgto_projection import project_pw_to_spline
    from jqmc.qe_to_jqmc import build_hamiltonian_from_projection
    from jqmc.qe_wrapper import load_qe_reference

    ref = load_qe_reference(str(_SAVE_DIR))
    proj = project_pw_to_spline(ref, mesh_factor=1.0, order=5)
    return build_hamiltonian_from_projection(ref, proj, kind="bspline", **_COMMON_HAM_KWARGS)


def _build_ham_pgto():
    """Build a Hamiltonian_data for kind='pgto' (QE Li reference)."""
    from jqmc.pw_pgto_projection import project_pw_to_pgto
    from jqmc.qe_to_jqmc import build_hamiltonian_from_projection
    from jqmc.qe_wrapper import load_qe_reference
    from jqmc.upf_wrapper import UPFExponentInit

    ref = load_qe_reference(str(_SAVE_DIR))
    strat = UPFExponentInit({"li": str(_UPF)}, g_target=float(np.sqrt(2.0 * ref.ecutwfc)))
    shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]
    proj = project_pw_to_pgto(ref, shells)
    return build_hamiltonian_from_projection(ref, proj, kind="pgto", **_COMMON_HAM_KWARGS)


def _build_ham_pgto_pw():
    """Build a Hamiltonian_data for kind='pgto+pw' (QE Li reference)."""
    from jqmc.pw_pgto_projection import project_pw_to_mixed
    from jqmc.qe_to_jqmc import build_hamiltonian_from_projection
    from jqmc.qe_wrapper import load_qe_reference
    from jqmc.upf_wrapper import UPFExponentInit

    ref = load_qe_reference(str(_SAVE_DIR))
    strat = UPFExponentInit({"li": str(_UPF)}, g_target=float(np.sqrt(2.0 * ref.ecutwfc)))
    shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]
    g_max = float(np.sqrt(2.0 * ref.ecutwfc))
    proj = project_pw_to_mixed(ref, shells, g_cut=0.5 * g_max, drop_diffuse=False)
    return build_hamiltonian_from_projection(ref, proj, kind="pgto+pw", **_COMMON_HAM_KWARGS)


def _build_ham_pgto_bspline():
    """Build a Hamiltonian_data for kind='pgto+bspline' (QE Li reference)."""
    from jqmc.pw_pgto_projection import project_pw_to_pgto_spline
    from jqmc.qe_to_jqmc import build_hamiltonian_from_projection
    from jqmc.qe_wrapper import load_qe_reference
    from jqmc.upf_wrapper import UPFExponentInit

    ref = load_qe_reference(str(_SAVE_DIR))
    strat = UPFExponentInit({"li": str(_UPF)}, g_target=float(np.sqrt(2.0 * ref.ecutwfc)))
    shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]
    proj = project_pw_to_pgto_spline(ref, shells, mesh_factor=0.5, order=5)
    return build_hamiltonian_from_projection(ref, proj, kind="pgto+bspline", **_COMMON_HAM_KWARGS)


_KIND_BUILDERS = {
    "pw": _build_ham_pw,
    "bspline": _build_ham_bspline,
    "pgto": _build_ham_pgto,
    "pgto+pw": _build_ham_pgto_pw,
    "pgto+bspline": _build_ham_pgto_bspline,
}
_ALL_KINDS = list(_KIND_BUILDERS.keys())
# LRDMC force cases: the sample-wise SWCT sum rule is a numerical verdict only for
# the atom-following GTO basis; on the cell-anchored kinds the force run is a reviewed
# finiteness-only smoke (every returned force output must be finite).
_SWCT_NON_GTO_REASON = (
    "The basis is cell-anchored, so Sum_I F_I vanishes only in expectation; a short LRDMC run "
    "cannot resolve it sample-wise and the case asserts finiteness of force_mean and force_std only"
)
_LRDMC_FORCE_KINDS = [
    kind
    if kind == "pgto"
    else pytest.param(kind, marks=pytest.mark.gdd_finiteness_evidence(mode="smoke", reason=_SWCT_NON_GTO_REASON))
    for kind in _ALL_KINDS
]


@pytest.fixture(scope="module")
def hamiltonian(request):
    """Module-scoped, parametrized Hamiltonian_data fixture.

    Dispatched by the ``kind`` indirect parameter; each kind builds once per
    module and is reused across all algorithm tests for that kind. Every
    consumer is already gated by ``requires_qe`` on the committed sample.
    """
    kind = request.param
    builder = _KIND_BUILDERS.get(kind)
    if builder is None:
        raise ValueError(f"unknown kind: {kind}")
    return builder()


@pytest.fixture(scope="module")
def hamiltonian_pw():
    """Standalone pw fixture for the h5 round-trip end-to-end test (consumer gated by ``requires_qe``)."""
    return _build_ham_pw()


def _assert_finite_e_L(driver):
    """Driver-agnostic NaN/Inf check on the most recent e_L sample buffer."""
    e_L = np.asarray(driver.e_L)
    w_L = np.asarray(driver.w_L)
    assert e_L.size > 0, "empty e_L buffer"
    assert np.all(np.isfinite(e_L)), f"non-finite e_L: {e_L}"
    assert np.all(np.isfinite(w_L)), f"non-finite w_L: {w_L}"


# ---------------------------------------------------------------------------
# MCMC
# ---------------------------------------------------------------------------


class TestQEPeriodicIntegration:
    @pytest.mark.heavy
    @requires_qe
    @pytest.mark.parametrize("hamiltonian", _ALL_KINDS, indirect=True, ids=_ALL_KINDS)
    def test_mcmc_runs(self, hamiltonian):
        """Each kind: a short MCMC run completes and yields finite E."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = MCMC(
            hamiltonian_data=hamiltonian,
            Dt=1.5,
            mcmc_seed=42,
            epsilon_AS=1.0e-6,
            num_walkers=2,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )
        mcmc.run(num_mcmc_steps=10)
        _assert_finite_e_L(mcmc)

        E, E_err, Var, _Var_err = mcmc.get_E(num_mcmc_warmup_steps=2, num_mcmc_bin_blocks=2)
        assert np.isfinite(E), f"MCMC E not finite: {E}"
        assert np.isfinite(E_err), f"MCMC E_err not finite: {E_err}"
        assert np.isfinite(Var), f"MCMC Var not finite: {Var}"
        jax.clear_caches()

    # ---------------------------------------------------------------------------
    # GFMC_t (continuous-time)
    # ---------------------------------------------------------------------------
    @pytest.mark.heavy
    @requires_qe
    @pytest.mark.parametrize("hamiltonian", _ALL_KINDS, indirect=True, ids=_ALL_KINDS)
    def test_gfmc_t_runs(self, hamiltonian):
        """Each kind: a short GFMC_t run completes and yields finite E."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = GFMC_t(
            hamiltonian_data=hamiltonian,
            num_walkers=2,
            num_gfmc_collect_steps=3,
            mcmc_seed=42,
            tau=0.1,
            alat=0.30,
            random_discretized_mesh=False,
            comput_position_deriv=False,
            non_local_move="tmove",
        )
        gfmc.run(num_mcmc_steps=10)
        _assert_finite_e_L(gfmc)

        E, E_err, Var, _Var_err = gfmc.get_E(num_mcmc_warmup_steps=2, num_mcmc_bin_blocks=2)
        assert np.isfinite(E), f"GFMC_t E not finite: {E}"
        assert np.isfinite(E_err), f"GFMC_t E_err not finite: {E_err}"
        assert np.isfinite(Var), f"GFMC_t Var not finite: {Var}"
        jax.clear_caches()

    # ---------------------------------------------------------------------------
    # GFMC_n (branching)
    # ---------------------------------------------------------------------------
    @pytest.mark.heavy
    @requires_qe
    @pytest.mark.parametrize("hamiltonian", _ALL_KINDS, indirect=True, ids=_ALL_KINDS)
    def test_gfmc_n_runs(self, hamiltonian):
        """Each kind: a short GFMC_n run completes and yields finite E."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = GFMC_n(
            hamiltonian_data=hamiltonian,
            num_walkers=2,
            num_mcmc_per_measurement=5,
            num_gfmc_collect_steps=3,
            mcmc_seed=42,
            E_scf=-200.0,
            alat=0.30,
            comput_position_deriv=False,
            non_local_move="tmove",
        )
        gfmc.run(num_mcmc_steps=10)

        E, E_err, Var, _Var_err = gfmc.get_E(num_mcmc_warmup_steps=2, num_mcmc_bin_blocks=2)
        assert np.isfinite(E), f"GFMC_n E not finite: {E}"
        assert np.isfinite(E_err), f"GFMC_n E_err not finite: {E_err}"
        assert np.isfinite(Var), f"GFMC_n Var not finite: {Var}"
        jax.clear_caches()

    # ---------------------------------------------------------------------------
    # LRDMC == GFMC_n + alat (without SWCT forces)
    # ---------------------------------------------------------------------------
    @pytest.mark.heavy
    @requires_qe
    @pytest.mark.parametrize("hamiltonian", _ALL_KINDS, indirect=True, ids=_ALL_KINDS)
    def test_lrdmc_runs(self, hamiltonian):
        """Each kind: a short LRDMC (GFMC_n) energy expectation is finite.

        ``comput_position_deriv=False`` -- E_L expectation only. The SWCT force
        path is exercised separately in ``test_lrdmc_force_runs`` below.
        """
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = GFMC_n(
            hamiltonian_data=hamiltonian,
            num_walkers=2,
            num_mcmc_per_measurement=5,
            num_gfmc_collect_steps=3,
            mcmc_seed=42,
            E_scf=-200.0,
            alat=0.30,
            comput_position_deriv=False,
            non_local_move="tmove",
        )
        gfmc.run(num_mcmc_steps=10)

        E, E_err, _Var, _Var_err = gfmc.get_E(num_mcmc_warmup_steps=2, num_mcmc_bin_blocks=2)
        assert np.isfinite(E), f"LRDMC E not finite: {E}"
        assert np.isfinite(E_err), f"LRDMC E_err not finite: {E_err}"
        jax.clear_caches()

    @pytest.mark.heavy
    @requires_qe
    @pytest.mark.parametrize("hamiltonian", _LRDMC_FORCE_KINDS, indirect=True, ids=_ALL_KINDS)
    def test_lrdmc_force_runs(self, hamiltonian):
        """Each kind: LRDMC with comput_position_deriv=True + SWCT runs and
        yields finite per-atom forces. For atom-following (GTO) bases, also
        verify the SWCT translational sum rule sample-wise.

        Exercises ``Wavefunction_data.with_param_grad_mask`` -- the basis-exp /
        basis-coeff masking blocks short-circuit to no-op on non-GTO AOs
        (PW/Bspline/composite), so the force estimator must complete on all
        five kinds (finite-force check).

        SWCT sum-rule note
        ------------------
        The estimator ``F_I = grad_R_I e_L + Sum_e omega_I(r_e) * grad_r_e e_L``
        is a partial derivative w.r.t. R_I at fixed MO coefficients ``c``.
        Whether ``Sum_I F_I == 0`` holds sample-wise depends on whether the
        AO basis follows atoms:

        * GTO (pgto here): ``chi(r - R)`` is atom-following, so the partial
          derivative coincides with physical translation. The local energy
          e_L is invariant under simultaneous shift of (r, R), and the sum
          rule holds sample-wise to round-off. We verify it here for the
          single-atom Cu fixture (omega_0 = 1 by partition of unity).

        * PW / Bspline / composite (pw, bspline, pgto+pw, pgto+bspline):
          the basis is cell-anchored, not atom-anchored. The partial
          derivative dE/dR_I at fixed c does NOT coincide with physical
          translation, so e_L is not (r, R)-shift-invariant sample-wise.
          ``Sum_I F_I`` averages to 0 in the MCMC expectation (total energy
          is rigorously translation-invariant -- this is the standard
          "PW has no Pulay force" picture), but each sample carries
          finite residual and the average converges only with much longer
          runs than this integration test affords. Skip the sample-wise
          assertion for non-GTO kinds.

        Tolerances follow the silicon sum-rule logic: ``check_sum_rule_force``
        scales atol by ``max(|F|)`` so it handles both full and mixed precision.
        """
        from jqmc.atomic_orbital import AOs_GTO_base_data
        from jqmc.jqmc_gfmc import GFMC_n
        from jqmc.molecular_orbital import MOs_data

        from tests.conftest import check_sum_rule_force

        gfmc = GFMC_n(
            hamiltonian_data=hamiltonian,
            num_walkers=2,
            num_mcmc_per_measurement=5,
            num_gfmc_collect_steps=3,
            mcmc_seed=42,
            E_scf=-200.0,
            alat=0.30,
            comput_position_deriv=True,
            use_swct=True,
            non_local_move="tmove",
            epsilon_PW=1.0e-2,
        )
        gfmc.run(num_mcmc_steps=10)

        force_mean, force_std = gfmc.get_aF(num_mcmc_warmup_steps=2, num_mcmc_bin_blocks=2)
        assert np.all(np.isfinite(np.asarray(force_mean))), f"LRDMC force_mean not finite: {force_mean}"
        assert np.all(np.isfinite(np.asarray(force_std))), f"LRDMC force_std not finite: {force_std}"

        # Sample-wise SWCT sum rule applies only when the AO basis is atom-
        # following (GTO). For non-GTO leaves the residual averages to zero
        # only in expectation and short MCMC runs cannot resolve that here.
        orb = hamiltonian.wavefunction_data.geminal_data.orb_data_up_spin
        if isinstance(orb, MOs_data):
            orb = orb.aos_data
        if isinstance(orb, AOs_GTO_base_data):
            atol, rtol = get_tolerance_min(
                ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"),
                "strict",
            )
            check_sum_rule_force(force_mean, atol, rtol)

        jax.clear_caches()

    # ---------------------------------------------------------------------------
    # Optional: h5 round-trip + MCMC (covers the "build, save, reload, run"
    # end-to-end recipe). Uses the pw kind only -- pgto+pw is exercised by the
    # in-memory MCMC test above and the round-trip plumbing is already covered
    # by test_qe_to_jqmc.test_build_hamiltonian_pw_h5_roundtrip.
    # ---------------------------------------------------------------------------
    @pytest.mark.heavy
    @requires_qe
    def test_mcmc_runs_from_h5_roundtrip(self, tmp_path, hamiltonian_pw):
        """Save -> load -> MCMC must work end-to-end for the QE PW Hamiltonian."""
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jqmc_mcmc import MCMC

        h5 = tmp_path / "qe_pw_int.h5"
        hamiltonian_pw.save_to_hdf5(str(h5))
        assert h5.is_file()

        loaded = Hamiltonian_data.load_from_hdf5(str(h5))
        loaded.sanity_check()

        mcmc = MCMC(
            hamiltonian_data=loaded,
            Dt=1.5,
            mcmc_seed=42,
            epsilon_AS=1.0e-6,
            num_walkers=2,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )
        mcmc.run(num_mcmc_steps=5)
        _assert_finite_e_L(mcmc)
        jax.clear_caches()

    # ---------------------------------------------------------------------------
    # VMC optimization (MCMC.run_optimize) -- short wavefunction parameter update
    # loop on the QE-derived Hamiltonian. Verifies that the production-style
    # VMC optimization pipeline (sampling -> param gradients -> SR/Adam update)
    # completes without NaN and improves (or at least does not destroy) the
    # Hamiltonian state for each of the five projection kinds.
    #
    # Rank-truncated MO optimization goes through ``fix_geminal_rank`` +
    # auto-convert to ``AGPn_data`` at ``run_optimize`` entry.
    # ---------------------------------------------------------------------------
    @pytest.mark.heavy
    @requires_qe
    @pytest.mark.parametrize("hamiltonian", _ALL_KINDS, indirect=True, ids=_ALL_KINDS)
    def test_mcmc_optimize_runs(self, hamiltonian):
        """Each kind: a short MCMC.run_optimize step completes and yields a
        finite post-optimization energy.

        Production-style end-to-end check: QE -> projection -> Hamiltonian_data
        -> VMC sampling + parameter optimization. Optimizes J2 only (J1/J3/NN
        are off in the QE-derived fixture; basis-exp / basis-coeff are off
        because PW/Bspline have no primitive exponents) for a single SR step
        with 5 walkers and a small number of MCMC samples per iteration.

        The test verifies:
        1. ``run_optimize`` completes without exception on all five kinds.
        2. The post-optimization E expectation is finite.
        3. For non-GTO kinds the basis-mask guard (added to
           ``with_param_grad_mask`` for PW/Spline/composite) is exercised
           implicitly via the optimization step.
        """
        from jqmc.jqmc_mcmc import MCMC

        mcmc = MCMC(
            hamiltonian_data=hamiltonian,
            Dt=1.5,
            mcmc_seed=42,
            epsilon_AS=1.0e-6,
            num_walkers=2,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=True,
            random_discretized_mesh=True,
        )

        mcmc.run_optimize(
            num_mcmc_steps=8,
            num_opt_steps=1,
            num_mcmc_warmup_steps=2,
            num_mcmc_bin_blocks=2,
            opt_J1_param=False,
            opt_J2_param=True,
            opt_J3_param=False,
            opt_geminal=False,
            opt_J3_basis_exp=False,
            opt_J3_basis_coeff=False,
            opt_lambda_basis_exp=False,
            opt_lambda_basis_coeff=False,
            optimizer_kwargs={"method": "sr", "delta": 1.0e-2, "epsilon": 1.0e-3},
        )

        # Post-optimization energy must be finite. Re-run sampling briefly and
        # check the energy expectation comes back finite (cheap proxy for "the
        # optimization did not corrupt the wavefunction state").
        mcmc.run(num_mcmc_steps=5)
        _assert_finite_e_L(mcmc)
        E, E_err, _Var, _Var_err = mcmc.get_E(num_mcmc_warmup_steps=1, num_mcmc_bin_blocks=2)
        assert np.isfinite(E), f"post-optimize MCMC E not finite: {E}"
        assert np.isfinite(E_err), f"post-optimize MCMC E_err not finite: {E_err}"
        jax.clear_caches()

    # ---------------------------------------------------------------------------
    # Structural Pulay-zero property: pure PW / B-spline AO bases are cell-anchored
    # (no R-dependence), so d(compute_AOs)/d(structure_data.positions) is exactly
    # zero -- this is the textbook "PW DFT has no Pulay force" property pinned at
    # the AO level. GTO control verifies the test logic isn't trivially passing.
    # No MCMC, no Hamiltonian -- pure autodiff structural assertion.
    # ---------------------------------------------------------------------------

    @requires_qe
    @pytest.mark.parametrize(
        "hamiltonian, expect_zero_R_grad",
        [
            pytest.param("pw", True, id="pw"),
            pytest.param("bspline", True, id="bspline"),
            pytest.param("pgto", False, id="pgto-control"),
        ],
        indirect=["hamiltonian"],
    )
    @pytest.mark.heavy
    def test_pure_cell_anchored_basis_has_zero_R_gradient(self, hamiltonian, expect_zero_R_grad):
        """Pure PW / B-spline AO bases have exactly zero R-gradient; GTO does not.

        Cell-anchored bases (``cos(G.r)``, ``B-spline(r)``) carry no dependence
        on atom positions, so ``d(compute_AOs)/d(structure_data.positions)``
        vanishes to round-off (in practice, exactly zero in fp64 because the
        autodiff path produces no nonzero contributions). Atom-anchored GTO
        bases (``chi(r - R)``) have non-trivial R-dependence and must produce
        a nonzero gradient -- a control check that prevents the assertion
        from silently passing on every kind via a refactor bug.

        This is the underlying reason "PW DFT has no Pulay force" (the basis
        derivative contribution to dE/dR is absent). Guards against future
        refactors that accidentally route the PW / B-spline kernels through
        ``structure_data.positions`` (e.g. wrong cell / position accessor).
        """
        import jax
        import jax.numpy as jnp

        from jqmc.atomic_orbital import compute_AOs
        from jqmc.molecular_orbital import MOs_data

        # Extract the AO leaf (unwrap MOs_data if present).
        orb = hamiltonian.wavefunction_data.geminal_data.orb_data_up_spin
        aos_data = orb.aos_data if isinstance(orb, MOs_data) else orb

        # Small probe configuration -- 3 electron sample points in cell.
        r_carts = jnp.array(
            [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [-0.3, 0.7, 1.2]],
            dtype=jnp.float64,
        )
        positions0 = jnp.asarray(aos_data.structure_data.positions, dtype=jnp.float64)

        def aos_at_positions(positions):
            new_struct = aos_data.structure_data.replace(positions=positions)
            new_aos = aos_data.replace(structure_data=new_struct)
            return compute_AOs(new_aos, r_carts)

        R_grad = np.asarray(jax.jacrev(aos_at_positions)(positions0))
        max_abs = float(np.max(np.abs(R_grad)))

        if expect_zero_R_grad:
            # Cell-anchored basis: d(AOs)/dR must be EXACTLY zero (no fp tolerance).
            assert max_abs == 0.0, (
                f"Expected zero R-gradient for cell-anchored basis "
                f"{type(aos_data).__name__}, but got max|.| = {max_abs:.3e}. "
                f"This indicates a refactor accidentally introduced an "
                f"R-dependence into the basis evaluator."
            )
        else:
            # Atom-anchored basis (GTO): must have non-trivial R-gradient.
            assert max_abs > 0.0, (
                f"Expected non-trivial R-gradient for atom-anchored basis "
                f"{type(aos_data).__name__}, but got max|.| = {max_abs:.3e}. "
                f"This indicates the GTO basis lost its R-dependence -- a "
                f"control failure means the cell-anchored assertion above "
                f"may be silently passing."
            )


# ---------------------------------------------------------------------------
# Per-walker HF stress: AD vs finite-difference (deterministic; catches the PW
# vgl FFI stress bug). Stress = d e_L / d(cell strain). For a plane-wave basis the
# strain enters through the reciprocal-lattice G vectors, so the AD path
# differentiates the PW AO value+grad+lap FFI w.r.t. the cell -- exactly the path
# that the (now-fixed) g/gk-closure bug crashed ("No constant handler"). Unlike the
# MCMC force sum rule, this is a single deterministic per-walker check (no statistics):
# SWCT delivers the per-sample force sum rule only for atom-anchored GTO bases, so for
# the cell-anchored PW/Spline leaves the sum rule averages to zero only in expectation
# (too noisy for a tight guard) -- the FD-vs-AD of the local energy is the right
# deterministic test (mirrors test_stress_HF_fd_vs_ad_per_walker on the GTO silicon).
# ---------------------------------------------------------------------------
class TestQEPeriodicStressFDvsAD:
    @pytest.mark.heavy
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-DERIV-LOCAL-ENERGY-001",
    )
    @requires_qe
    @pytest.mark.parametrize("kind", ["pgto", "pgto+pw", "bspline"], ids=["pgto", "pgto+pw", "bspline"])
    def test_stress_HF_fd_vs_ad_per_walker(self, kind):
        """sigma_HF[0,0]: jacrev AD vs central FD of e_L under an x-x strain, one fixed walker."""
        import jax.numpy as jnp

        from jqmc._precision import get_tolerance
        from jqmc.hamiltonians import compute_local_energy
        from jqmc.derivatives._auto import _jit_vmap_local_stress_HF

        hamiltonian = _KIND_BUILDERS[kind]()
        gem = hamiltonian.wavefunction_data.geminal_data
        n_up, n_dn = int(gem.num_electron_up), int(gem.num_electron_dn)
        rng = np.random.default_rng(2026)
        r_up = jnp.asarray(rng.uniform(0.0, 3.0, size=(n_up, 3)), dtype=jnp.float64)
        r_dn = jnp.asarray(rng.uniform(0.0, 3.0, size=(n_dn, 3)), dtype=jnp.float64)
        RT = jnp.eye(3, dtype=jnp.float64)

        sigma_HF = np.asarray(
            _jit_vmap_local_stress_HF(
                hamiltonian, r_up[None, ...], r_dn[None, ...], RT[None, ...], jnp.zeros((3, 3), dtype=jnp.float64)
            )
        )[0]  # (3, 3)

        # 4th-order central FD: matches the ``medium`` tolerance contract in
        # ``_setting.py`` (calibrated for h=1e-3, truncation O(h^4 * f^(6))).
        # 2nd-order central FD with h=1e-4 leaves ~1e-4 residuals near AO cusps
        # / PP shells, which busts the medium-fp64 tolerance gate.
        h_step = 1.0e-3

        def _e_at(eps_xx):
            F = jnp.eye(3, dtype=jnp.float64).at[0, 0].add(eps_xx)
            return float(compute_local_energy(hamiltonian._apply_strain(F), r_up @ F.T, r_dn @ F.T, RT))

        fd_xx = (-_e_at(2.0 * h_step) + 8.0 * _e_at(h_step) - 8.0 * _e_at(-h_step) + _e_at(-2.0 * h_step)) / (12.0 * h_step)
        atol, rtol = get_tolerance("local_energy", "medium")
        np.testing.assert_allclose(float(sigma_HF[0, 0]), fd_xx, atol=atol, rtol=rtol)
        record_fd_case(
            "fd-ad-qe-periodic-stress",
            pytest_node_id="tests/test_jqmc_kernel/test_integration_qe_periodic.py::TestQEPeriodicStressFDvsAD::test_stress_HF_fd_vs_ad_per_walker",
            fixture_parameters=fd_case_parameters("fd-ad-qe-periodic-stress", kind=kind),
            source_ids=("WF-SRC-LI-ECP-GAMMA",),
            artifact_format="qe-save",
            transformations=(f"derivatives.fd.project-qe-{kind}.v1",),
            coordinates=(r_up, r_dn),
            coordinate_generator="numpy.random.Generator.uniform",
            coordinate_seed=2026,
            compared_pairs=((sigma_HF[0, 0], fd_xx),),
            atol=atol,
            rtol=rtol,
            zones=("local_energy",),
            level="medium",
            stencil="central-first-order-4",
            step=h_step,
            direction="eps-00",
        )
        jax.clear_caches()


# ---------------------------------------------------------------------------
# Per-walker HF force: AD vs finite-difference (deterministic; first FD-vs-AD force
# test for any basis). The stored production force_HF is the SWCT-warped local force
#   force_HF_I = d e_L / d R_I + sum_e omega_I(r_e) * d e_L / d r_e
# (jqmc_mcmc.py), i.e. the directional derivative of e_L under the space warp that
# moves atom I by delta and each electron e by omega_I(r_e)*delta. We compare the AD
# value (jax.grad of compute_local_energy w.r.t. the atom positions + electrons,
# combined with the swct omega) against an INDEPENDENT central finite difference of
# the same warped displacement. Deterministic per-walker (no MCMC statistics) and
# meaningful for EVERY basis -- GTO/pGTO carry the atom-following AO Pulay-local term,
# PW/Spline are cell-anchored; both are captured identically by the e_L derivative.
# (Only the statistical force_PP covariance is excluded.)
# ---------------------------------------------------------------------------
class TestQEPeriodicForceFDvsAD:
    @pytest.mark.heavy
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-DERIV-LOCAL-ENERGY-001",
    )
    @requires_qe
    @pytest.mark.parametrize("kind", ["pgto", "pgto+pw", "bspline"], ids=["pgto", "pgto+pw", "bspline"])
    def test_force_HF_fd_vs_ad_per_walker(self, kind):
        """force_HF (SWCT-warped d e_L/d R): jax AD vs central FD of the warped displacement."""
        import jax.numpy as jnp

        from jqmc._precision import get_tolerance
        from jqmc.hamiltonians import compute_local_energy
        from jqmc.jqmc_mcmc import _extract_active_positions_from_ham, _inject_active_positions
        from jqmc.swct import evaluate_swct_omega

        ham = _KIND_BUILDERS[kind]()
        sd = ham.structure_data
        n_atom = len(sd.positions)
        gem = ham.wavefunction_data.geminal_data
        n_up, n_dn = int(gem.num_electron_up), int(gem.num_electron_dn)
        rng = np.random.default_rng(2026)
        r_up = jnp.asarray(rng.uniform(0.0, 3.0, size=(n_up, 3)), dtype=jnp.float64)
        r_dn = jnp.asarray(rng.uniform(0.0, 3.0, size=(n_dn, 3)), dtype=jnp.float64)
        RT = jnp.eye(3, dtype=jnp.float64)

        active = _extract_active_positions_from_ham(ham)
        omega_up = np.asarray(evaluate_swct_omega(sd, r_up))  # (n_atom, n_up)
        omega_dn = np.asarray(evaluate_swct_omega(sd, r_dn))  # (n_atom, n_dn)

        def _e_L(ap, ru, rd):
            return jnp.real(compute_local_energy(_inject_active_positions(ham, ap), ru, rd, RT))

        # AD: force_HF = d e_L/d R + Omega . d e_L/d r (the production formula).
        grad_R = np.asarray(jax.grad(_e_L, argnums=0)(active, r_up, r_dn)["positions"])  # (n_atom, 3)
        grad_r_up = np.asarray(jax.grad(_e_L, argnums=1)(active, r_up, r_dn))  # (n_up, 3)
        grad_r_dn = np.asarray(jax.grad(_e_L, argnums=2)(active, r_up, r_dn))  # (n_dn, 3)
        force_HF_ad = grad_R + np.einsum("ae,ed->ad", omega_up, grad_r_up) + np.einsum("ae,ed->ad", omega_dn, grad_r_dn)

        # FD: 4th-order central difference of e_L under the warped displacement of
        # one (atom, axis). Matches the ``medium`` tolerance contract in ``_setting.py``
        # (calibrated for h=1e-3, truncation O(h^4 * f^(6))). 2nd-order with h=1e-4
        # leaves ~1e-4 residuals near AO cusps / PP shells.
        h_step = 1.0e-3
        force_HF_fd = np.zeros((n_atom, 3), dtype=np.float64)
        for atom in range(n_atom):
            for axis in range(3):

                def _e_warp(delta, atom=atom, axis=axis):
                    ap2 = dict(active)
                    ap2["positions"] = active["positions"].at[atom, axis].add(delta)
                    ru2 = r_up.at[:, axis].add(jnp.asarray(omega_up[atom], dtype=jnp.float64) * delta)
                    rd2 = r_dn.at[:, axis].add(jnp.asarray(omega_dn[atom], dtype=jnp.float64) * delta)
                    return float(_e_L(ap2, ru2, rd2))

                force_HF_fd[atom, axis] = (
                    -_e_warp(2.0 * h_step) + 8.0 * _e_warp(h_step) - 8.0 * _e_warp(-h_step) + _e_warp(-2.0 * h_step)
                ) / (12.0 * h_step)

        atol, rtol = get_tolerance("local_energy", "medium")
        np.testing.assert_allclose(force_HF_ad, force_HF_fd, atol=atol, rtol=rtol)
        record_fd_case(
            "fd-ad-qe-periodic-force",
            pytest_node_id="tests/test_jqmc_kernel/test_integration_qe_periodic.py::TestQEPeriodicForceFDvsAD::test_force_HF_fd_vs_ad_per_walker",
            fixture_parameters=fd_case_parameters("fd-ad-qe-periodic-force", kind=kind),
            source_ids=("WF-SRC-LI-ECP-GAMMA",),
            artifact_format="qe-save",
            transformations=(f"derivatives.fd.project-qe-{kind}.v1",),
            coordinates=(r_up, r_dn),
            coordinate_generator="numpy.random.Generator.uniform",
            coordinate_seed=2026,
            compared_pairs=((force_HF_ad, force_HF_fd),),
            atol=atol,
            rtol=rtol,
            zones=("local_energy",),
            level="medium",
            stencil="central-first-order-4",
            step=h_step,
            direction="swct-warped-ion-position-cartesian-basis",
        )
        jax.clear_caches()


if __name__ == "__main__":
    # Allow running the file directly for ad-hoc debugging.
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
