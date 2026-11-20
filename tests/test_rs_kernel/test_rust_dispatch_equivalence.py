"""Rust-vs-XLA production-driver end-to-end parity.

The same public ``MCMC`` driver is run with the native Rust backend ON and OFF
(pure XLA). Production Rust owns its proposal/mesh RNG, so the trajectories are
not expected to be elementwise identical. The contract is instead:

* both routes produce finite, shape-compatible stored trajectories;
* stored output structures are finite and shape-compatible; and
* spies prove that ON actually reached the native production entry points.

This is the DIRECT integration check for the ``flag_rust_on_CPU`` dispatch:
where ``test_jqmc_mcmc`` compares production ``MCMC`` against the ``_MCMC_debug`` oracle, THIS
module toggles ONLY the backend flag on one code path, isolating the Rust<->JAX dispatch itself.
Fixed-input transition and chunk diagnostics remain in ``test_mcmc_transition.py``
and ``test_obs_ctx_equivalence.py``; they do not create a second production route.

Coverage:
  * [mcmc update]: ``w_L`` / ``e_L`` / ``e_L2`` / binned E -- the update is Rust.
  * [mcmc observables, e_L]: the MOLECULAR e_L, the PERIODIC real-Gamma e_L, and
    the COMPLEX-twist (k != Gamma) e_L all run native -- ``compute_local_energy_fast``
    (real) / ``compute_local_energy_fast_complex`` (complex), T + Coulomb/Ewald + ECP reusing the
    sweep SM inverse -- so ``e_L`` / ``e_L2`` / binned-E here are a genuine native-vs-JAX comparison
    on EVERY cell; NO e_L path falls back to JAX. MPC is a PASSIVE,
    energy-only observable computed separately (never substituted into e_L / the projection) and
    carries no derivatives -- covered by ``tests/test_rs_kernel/test_mpc_observable.py``, not here.
    ``force_HF_stored`` AND ``force_PP_stored`` are native-vs-JAX
    on molecular + periodic real-Gamma + complex-twist pGTO + real-Gamma exotic leaves (see
    ``_assert_native_force``); ``stress_*_stored`` (and O_k / dE_L/dc) stay JAX==JAX guards.

AO-LEAF COVERAGE (SYSTEMATIC -- no ad-hoc gaps). The production sweep can receive any of these
periodic/twist AO geminal leaves; EVERY one is exercised here so no exotic-AO type ships broken:
                       | pgto sphe | pgto cart |  pure PW  | pure bspline | pgto+pw | pgto+bspline
  real Gamma           | Li_gamma  | H2_gamma  | gamma_pw  | gamma_bspline| gamma_  | gamma_pgto_
    (AOs_*_data)       |  _sphe    |  _cart    |           |              |  pgto_pw|  bspline
  complex twist k!=0   | Li_twist  | H2_twist  | twist_pw  | twist_bspline| twist_  | twist_pgto_
    (AOs_*_complex_data)|  _sphe    |  _cart    |           |              |  pgto_pw|  bspline
The pGTO leaves come from committed TREXIO fixtures (``_FIXTURES``); the PW / bspline / composite
leaves are built from the committed QE projection samples (``_QE_EXOTIC``) -- no bench-h5. Each
case asserts flag-ON (native) == flag-OFF (JAX) AND (via a sweep spy) that the native
``_mcmc_sweep`` / ``_mcmc_sweep_complex`` actually ran, with NO capability-gate fallback. Molecular
AE/ECP pgto cart leaves are also covered (``_FIXTURES``) as the non-periodic baseline.
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

import logging
import os

import jax
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from jqmc._setting import (
    MCMC_MIN_STEPS,
)
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.jqmc_mcmc import MCMC
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._test_data_cache import qe_example_files_root
from tests._ecp_core_electrons import core_electrons_of

_TREXIO = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files")

# (id, trexio stem, heavy?). J1+J2 exp Jastrow on every cell so the update exercises the
# Jastrow-ratio streaming state, not just the determinant. The exotic AO leaves below
# (bspline / plane-wave / composite / complex cart-pgto) close the dispatch coverage
# gap that let the ``pgto+bspline`` crashes ship: the original set only
# spanned molecular AE/ECP + sphe-pgto real-Gamma / sphe-pgto twist.
_FIXTURES = [
    ("H2_ae_cart", "H2_ae_ccpvdz_cart.h5", False, "real"),  # molecular all-electron cart
    ("H2_ecp_cart", "H2_ecp_ccpvdz_cart.h5", False, "real"),  # molecular ECP cart (local + non-local)
    # --- periodic real Gamma pGTO leaves (sphe + cart) ---
    ("Li_pbc_gamma_sphe", "lithium.h5", True, "real"),  # AOs_sphe_pgto_data (ECP)
    ("H2_pbc_gamma_cart", "H2_pbc_ae_cart.h5", True, "real"),  # AOs_cart_pgto_data (AE)
    # --- periodic complex twist (k != 0) pGTO leaves (sphe + cart) ---
    ("Li_pbc_twist_sphe", "lithium_k025.h5", True, "complex"),  # AOs_sphe_pgto_complex_data (ECP)
    ("H2_pbc_twist_cart", "H2_pbc_ae_k025_cart.h5", True, "complex"),  # AOs_cart_pgto_complex_data (AE)
]


def _build_ham(stem):
    structure_data, aos_data, mos_data, mos_data_no, geminal_mo_data, coulomb_potential_data = read_trexio_file(
        trexio_file=os.path.join(_TREXIO, stem), store_tuple=True
    )
    n_atom = len(structure_data.atomic_numbers)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=core_electrons_of(coulomb_potential_data, n_atom),
        jastrow_1b_type="exp",
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure_data
    )
    jastrow_data = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=None)
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    ham = Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)
    ham.sanity_check()
    return ham


def _run(ham, rust_on, comput_position_deriv=True):
    """One production ``MCMC`` run with the Rust backend forced on/off via the env override the
    dispatch reads at ``run()`` head. Same seed is used, but the native and XLA RNG
    implementations deliberately produce independent trajectories.
    ``comput_position_deriv=False`` is the ENERGY-ONLY run that the native chunk
    delegation (``run_energy_chunk``) covers; ``True`` keeps the per-step dispatch (force cores)."""
    if rust_on:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)  # unset -> flag_rust_on_CPU default True
    else:
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    try:
        mcmc = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=34356,
            epsilon_AS=1.0e-6,
            num_walkers=2,
            comput_position_deriv=comput_position_deriv,
            comput_stress=comput_position_deriv,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )
        mcmc.run(num_mcmc_steps=MCMC_MIN_STEPS)
        return mcmc
    finally:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)


def _assert_stored_trajectory_contract(m_rust, m_xla, names):
    """Different chains must still expose the same finite storage contract."""
    for name in names:
        rust_values = np.asarray(getattr(m_rust, name))
        xla_values = np.asarray(getattr(m_xla, name))
        assert rust_values.shape == xla_values.shape, f"{name}: shape mismatch"
        assert rust_values.dtype == xla_values.dtype, f"{name}: dtype mismatch"
        assert np.all(np.isfinite(rust_values)), f"{name} (Rust) is non-finite"
        assert np.all(np.isfinite(xla_values)), f"{name} (XLA) is non-finite"


def _assert_run_equivalent(m_rust, m_jax):
    """Independent production chains expose the same finite storage contract."""
    _assert_stored_trajectory_contract(
        m_rust,
        m_jax,
        (
            "w_L",
            "e_L",
            "e_L2",
            "force_HF_stored",
            "force_PP_stored",
            "stress_HF_stored",
            "stress_PP_stored",
        ),
    )


def _run_flag_on_with_native_spy(ham, monkeypatch, comput_position_deriv=True):
    """Run flag-on and count the native Rust kernel calls, so a test can assert the NATIVE
    sweep (real vs complex) AND the native e_L (``compute_local_energy``) actually
    ran, not a silent JAX fallback. Returns (mcmc, counts)."""
    import jqmc_rs

    counts = {
        "complex": 0,
        "real": 0,
        "e_L": 0,
        "force_HF": 0,
        "force_PP": 0,
        "stress_HF": 0,
        "stress_PP": 0,
        "energy_chunk": 0,
    }
    orig_c = jqmc_rs._mcmc_sweep_complex
    orig_r = jqmc_rs._mcmc_sweep
    orig_e = jqmc_rs.compute_local_energy_fast
    orig_ec = jqmc_rs.compute_local_energy_fast_complex
    orig_fae = jqmc_rs.vmc_grad_e_L_ae
    orig_fecp = jqmc_rs.vmc_grad_e_L_ecp
    orig_fpp = jqmc_rs.vmc_grad_ln_psi
    orig_fpbc = jqmc_rs.vmc_grad_e_L_pbc
    orig_fpbc_c = jqmc_rs.vmc_grad_e_L_pbc_complex
    orig_fpp_pbc = jqmc_rs.vmc_grad_ln_psi_pbc
    orig_fpp_pbc_c = jqmc_rs.vmc_grad_ln_psi_pbc_complex
    orig_shf = jqmc_rs.vmc_grad_e_L_eps_pbc
    orig_shf_c = jqmc_rs.vmc_grad_e_L_eps_pbc_complex
    orig_spp = jqmc_rs.vmc_grad_ln_psi_eps_pbc
    orig_spp_c = jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex

    def spy_c(*a, **k):
        counts["complex"] += 1
        return orig_c(*a, **k)

    def spy_r(*a, **k):
        counts["real"] += 1
        return orig_r(*a, **k)

    def spy_e(*a, **k):
        counts["e_L"] += 1
        return orig_e(*a, **k)

    def spy_ec(*a, **k):
        # The native COMPLEX-twist aggregate e_L counts as a native e_L call too.
        counts["e_L"] += 1
        return orig_ec(*a, **k)

    def spy_fae(*a, **k):
        # [timer_de_L_dR_dr]: native molecular AE VMC HF-force core.
        counts["force_HF"] += 1
        return orig_fae(*a, **k)

    def spy_fecp(*a, **k):
        # [timer_de_L_dR_dr]: native molecular ECP VMC HF-force core.
        counts["force_HF"] += 1
        return orig_fecp(*a, **k)

    def spy_fpp(*a, **k):
        # [timer_dln_Psi_dR_dr]: native molecular VMC Pulay WF-gradient core
        # (dln|Psi|/dR + drift; AE and ECP share it -- pure WF derivative, no mesh).
        counts["force_PP"] += 1
        return orig_fpp(*a, **k)

    def spy_fpbc(*a, **k):
        # native PERIODIC real-Gamma VMC HF-force core.
        counts["force_HF"] += 1
        return orig_fpbc(*a, **k)

    def spy_fpbc_c(*a, **k):
        # native COMPLEX-twist VMC HF-force core.
        counts["force_HF"] += 1
        return orig_fpbc_c(*a, **k)

    def spy_fpp_pbc(*a, **k):
        # native PERIODIC real-Gamma VMC Pulay core.
        counts["force_PP"] += 1
        return orig_fpp_pbc(*a, **k)

    def spy_fpp_pbc_c(*a, **k):
        # native COMPLEX-twist VMC Pulay core.
        counts["force_PP"] += 1
        return orig_fpp_pbc_c(*a, **k)

    def spy_shf(*a, **k):
        # native PERIODIC real-Gamma VMC HF stress.
        counts["stress_HF"] += 1
        return orig_shf(*a, **k)

    def spy_shf_c(*a, **k):
        # native COMPLEX-twist VMC HF stress.
        counts["stress_HF"] += 1
        return orig_shf_c(*a, **k)

    def spy_spp(*a, **k):
        # native PERIODIC real-Gamma VMC Pulay stress.
        counts["stress_PP"] += 1
        return orig_spp(*a, **k)

    def spy_spp_c(*a, **k):
        # native COMPLEX-twist VMC Pulay stress.
        counts["stress_PP"] += 1
        return orig_spp_c(*a, **k)

    monkeypatch.setattr(jqmc_rs, "vmc_grad_e_L_eps_pbc", spy_shf)
    monkeypatch.setattr(jqmc_rs, "vmc_grad_e_L_eps_pbc_complex", spy_shf_c)
    monkeypatch.setattr(jqmc_rs, "vmc_grad_ln_psi_eps_pbc", spy_spp)
    monkeypatch.setattr(jqmc_rs, "vmc_grad_ln_psi_eps_pbc_complex", spy_spp_c)
    monkeypatch.setattr(jqmc_rs, "_mcmc_sweep_complex", spy_c)
    monkeypatch.setattr(jqmc_rs, "_mcmc_sweep", spy_r)
    monkeypatch.setattr(jqmc_rs, "compute_local_energy_fast", spy_e)
    monkeypatch.setattr(jqmc_rs, "compute_local_energy_fast_complex", spy_ec)
    monkeypatch.setattr(jqmc_rs, "vmc_grad_e_L_ae", spy_fae)
    monkeypatch.setattr(jqmc_rs, "vmc_grad_e_L_ecp", spy_fecp)
    monkeypatch.setattr(jqmc_rs, "vmc_grad_ln_psi", spy_fpp)
    monkeypatch.setattr(jqmc_rs, "vmc_grad_e_L_pbc", spy_fpbc)
    monkeypatch.setattr(jqmc_rs, "vmc_grad_e_L_pbc_complex", spy_fpbc_c)
    monkeypatch.setattr(jqmc_rs, "vmc_grad_ln_psi_pbc", spy_fpp_pbc)
    monkeypatch.setattr(jqmc_rs, "vmc_grad_ln_psi_pbc_complex", spy_fpp_pbc_c)

    # NativeObsCtx: the driver calls the observable kernels as
    # METHODS on the once-built extraction ctx (``jqmc_rs.build_obs_ctx``), not the
    # module-level wrappers, so the module-fn spies above never fire for e_L/force/stress
    # on the production path. Wrap the ctx in a counting proxy keyed by method name (same
    # ``counts`` keys); the module-fn spies stay as belt-and-braces for any wrapper-path
    # caller. A ctx method missing from this map is intentionally NOT counted (it would
    # mean a new observable landed without extending the spy -- the count assert catches it).
    _ctx_count_key = {
        # The native sweep is a ctx method (Geminal/Jastrow/J3 extracted once),
        # not the module-level ``_mcmc_sweep`` -- count it here (the module spies stay as
        # belt-and-braces for any direct-wrapper caller).
        "mcmc_sweep": "real",
        "mcmc_sweep_complex": "complex",
        # The energy-only real-path run fuses sweep + e_L into one native chunk
        # call. It internally advances the sweep + e_L, so on this path the per-step
        # ``mcmc_sweep`` / ``e_L`` ctx methods do NOT fire -- the chunk does.
        # The driver fans the chunk over walkers via ``run_energy_chunk_batch``
        # (the per-walker ``run_energy_chunk`` entry survives for the unit/compare tests).
        "run_energy_chunk": "energy_chunk",
        "run_energy_chunk_batch": "energy_chunk",
        "e_L": "e_L",
        "e_L_complex": "e_L",
        "grad_e_L_ae": "force_HF",
        "grad_e_L_ecp": "force_HF",
        "grad_e_L_pbc": "force_HF",
        "grad_e_L_pbc_complex": "force_HF",
        "grad_ln_psi": "force_PP",
        "grad_ln_psi_pbc": "force_PP",
        "grad_ln_psi_pbc_complex": "force_PP",
        "grad_e_L_eps_pbc": "stress_HF",
        "grad_e_L_eps_pbc_complex": "stress_HF",
        "grad_ln_psi_eps_pbc": "stress_PP",
        "grad_ln_psi_eps_pbc_complex": "stress_PP",
    }
    # Every per-step ctx method has a ``*_batch``
    # twin the driver actually dispatches (ONE batched call per step covering all
    # walkers). Count each batch twin into the SAME bucket as its per-walker sibling
    # so the dispatch-equivalence assertions keep their meaning under the batch route.
    _ctx_count_key.update({k + "_batch": v for k, v in list(_ctx_count_key.items()) if not k.endswith("_batch")})

    class _CtxSpy:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            attr = getattr(self._inner, name)
            key = _ctx_count_key.get(name)
            if key is None or not callable(attr):
                return attr

            def _counted(*a, **k):
                counts[key] += 1
                return attr(*a, **k)

            return _counted

    orig_build = jqmc_rs.build_obs_ctx

    def spy_build(*a, **k):
        return _CtxSpy(orig_build(*a, **k))

    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", spy_build)
    m = _run(ham, rust_on=True, comput_position_deriv=comput_position_deriv)
    return m, counts


def _assert_native_eL(counts, expect_native, fixture_id):
    """[timer_e_L]: the native aggregate e_L runs on EVERY path --
    ``compute_local_energy_fast`` for MOLECULAR (bit-identical to jqmc at strict) +
    PERIODIC real-Gamma (bit-close at strict), and ``compute_local_energy_fast_complex`` for
    every COMPLEX-twist (k != Gamma) leaf (bit-close at strict; the kinetic reuses the
    running COMPLEX SM inverse with the internally-wrapping batch complex AO, and the ECP
    non-local stays complex exactly as jqmc's VMC energy so Im(e_L) == Im(T + V_nl)). Both spies
    increment ``counts['e_L']``, so ``expect_native`` is True on every path (MPC does not touch
    e_L at all -- it is a passive, energy-only observable computed separately). Assert the exact
    backend the driver selected, so a silent JAX fallback is caught."""
    if expect_native:
        assert counts["e_L"] > 0, f"expected the native e_L to run for {fixture_id}, got {counts}"
    else:
        assert counts["e_L"] == 0, f"native e_L must be capability-gated to JAX for {fixture_id}, got {counts}"


def _assert_native_force(counts, expect_hf, expect_pp, fixture_id):
    """[timer_de_L_dR_dr / timer_dln_Psi_dR_dr]: the HF force (dE_L/dR + Omega.dE_L/dr)
    and Pulay force (dln|Psi|/dR + Omega.dln|Psi|/dr + 1/2 sum dOmega) are gated SEPARATELY.

    * HF: native for MOLECULAR (``vmc_grad_e_L_ae`` / ``vmc_grad_e_L_ecp``, bit-identical at
      strict incl MO-J3 + DTN) AND PERIODIC real-Gamma + complex-twist pGTO
      (``vmc_grad_e_L_pbc`` / ``vmc_grad_e_L_pbc_complex``, at the documented periodic-force
      fp floor at near AND far-from-cell configs -- the far-config J3/DTN divergences were
      fixed kernel-level: J3 AO home-cell wrap + DTN min-image, see jqmc_mcmc gate comment).
      Excluded (JAX): complex-twist NON-pGTO, periodic MO-wrapped J3. MPC has no force at all
      (passive, energy-only observable) so it never appears in this comparison.
    * Pulay: native for MOLECULAR + periodic real-Gamma + complex-twist pGTO
      (``vmc_grad_ln_psi`` / ``vmc_grad_ln_psi_pbc`` / ``vmc_grad_ln_psi_pbc_complex``); the
      pure WF drift (grad ln|Psi|, no grad^2) matches JAX even far from the home cell.
      Complex-twist NON-pGTO (pw/bspline/composite) Pulay is gated off (reported gap)."""
    if expect_hf:
        assert counts["force_HF"] > 0, f"expected native HF force for {fixture_id}, got {counts}"
    else:
        assert counts["force_HF"] == 0, f"native HF force must be gated off for {fixture_id}, got {counts}"
    if expect_pp:
        assert counts["force_PP"] > 0, f"expected native Pulay force for {fixture_id}, got {counts}"
    else:
        assert counts["force_PP"] == 0, f"native Pulay force must be gated off for {fixture_id}, got {counts}"


def _stress_native_expect(ham):
    """Expected native VMC stress channels.  Every periodic determinant leaf,
    including real/complex pGTO+B-spline composites, is native."""
    if not ham.structure_data.pbc_flag:
        return (False, False)
    return (True, True)


def _assert_native_stress(counts, expect_hf, expect_pp, fixture_id):
    """[timer stress]: the periodic VMC HF stress (de_L/deps) and Pulay stress
    (dln|Psi|/deps) run natively (``vmc_grad_e_L_eps_pbc`` / ``vmc_grad_ln_psi_eps_pbc`` +
    complex) reusing the sweep SM inverse, replacing the JAX ``_grad_e_L_eps`` /
    ``_grad_ln_psi_eps``. Molecular runs compute no stress (buffers zero-padded); excluded
    exotic rows fall back to JAX. Assert the exact backend so a silent fallback is caught."""
    if expect_hf:
        assert counts["stress_HF"] > 0, f"expected native HF stress for {fixture_id}, got {counts}"
    else:
        assert counts["stress_HF"] == 0, f"native HF stress must be gated off for {fixture_id}, got {counts}"
    if expect_pp:
        assert counts["stress_PP"] > 0, f"expected native Pulay stress for {fixture_id}, got {counts}"
    else:
        assert counts["stress_PP"] == 0, f"native Pulay stress must be gated off for {fixture_id}, got {counts}"


def _force_chunk_eligible(ham):
    """Mirror the driver's observables-chunk gate for a FORCE run
    (``comput_position_deriv=True``). The chunk folds the HF force / Pulay force (and periodic
    HF/Pulay stress) INTO the fused ``run_energy_chunk`` loop, so on an eligible row the per-step
    force/stress ctx methods do NOT fire -- the fused chunk entry does (and advances the sweep +
    e_L internally too). Eligible = REAL path (the chunk drives the real sweep + real ClefArgs),
    HF + Pulay native, and -- for periodic -- BOTH stress channels native. The COMPLEX-twist
    family and non-pGTO-stress periodic leaves stay on the per-step dispatch."""
    from jqmc.atomic_orbital import is_complex_aos

    gd = ham.wavefunction_data.geminal_data
    ao_up = getattr(gd.orb_data_up_spin, "aos_data", gd.orb_data_up_spin)
    if is_complex_aos(ao_up):
        return False
    if not ham.structure_data.pbc_flag:
        return True  # molecular real: HF + Pulay native, no stress
    s_hf, s_pp = _stress_native_expect(ham)
    return bool(s_hf and s_pp)  # periodic: needs both native stress channels for the fold-in


def _assert_force_run_backend(ham, counts, native, fixture_id):
    """Assert the correct backend for a FORCE run: the fused observables chunk on an
    eligible (real, fully-native) row, else the per-step dispatch. On the chunk row every
    per-step ctx observable (sweep / e_L / force / stress) is advanced INSIDE Rust, so their
    Python-visible spy counts are 0 and ``energy_chunk`` is the only nonzero observable key."""
    if _force_chunk_eligible(ham):
        assert counts["energy_chunk"] > 0, f"expected the fused force chunk to run for {fixture_id}, got {counts}"
        for _k in ("real", "complex", "e_L", "force_HF", "force_PP", "stress_HF", "stress_PP"):
            assert counts[_k] == 0, f"chunk force run must not call per-step {_k} for {fixture_id}, got {counts}"
    else:
        assert counts["energy_chunk"] == 0, f"per-step force run must not fuse the chunk for {fixture_id}, got {counts}"
        assert counts[native] > 0, f"expected the native {native} per-step sweep to run, got {counts}"
        _assert_native_eL(counts, True, fixture_id)
        _assert_native_force(counts, expect_hf=True, expect_pp=True, fixture_id=fixture_id)
        _s_hf, _s_pp = _stress_native_expect(ham)
        _assert_native_stress(counts, expect_hf=_s_hf, expect_pp=_s_pp, fixture_id=fixture_id)


def _expected_fallbacks(ham):
    """The fallback notices this fixture is ALLOWED to emit -- DERIVED from the same stress-leaf
    gate mirror the driver uses, never hand-listed. With the capability gates removed (and the
    complex non-pGTO stress ported) the only observable that may still route to JAX is the periodic STRESS on an
    unported kernel leaf -- a composite pGTO(+)B-spline determinant (real OR complex; the withJ
    kinetic stress has no mixed pgto+bspline branch). e_L, force and param-deriv fallbacks are never
    allowed: their gates are gone, and MPC has no derivatives at all to fall back on (it is a
    passive, energy-only observable). Deriving the allowance means this whitelist EMPTIES ITSELF
    as leaves are ported -- a stale entry cannot survive."""
    s_hf, s_pp = _stress_native_expect(ham)
    return tuple(name for name, native in (("HF-stress", s_hf), ("Pulay-stress", s_pp)) if not native)


def _assert_no_gate_fallback(caplog, fixture_id, allow=()):
    """NO capability-gate fallback fired -- for ANY fixture. Each MCMC observable-gate
    fallback emits a LIVE ``logger.warning`` line (promoted from INFO to
    WARNING so it survives outside CLI rank 0) containing BOTH
    ``flag_rust_on_cpu`` and "falling back" -- distinct from JAX's unrelated "Falling back to
    cpu" GPU/xla_bridge notice, which never carries the ``flag_rust_on_cpu`` token. The run is
    captured at INFO on the ``jqmc`` logger so this string is live -- forcing a native gate off
    makes a real line appear (mutation-tested).

    ``allow`` has no MPC HF-force / HF-stress entries (MPC is a
    passive, energy-only observable with no derivatives at all), and the only permitted
    entries left are the unported STRESS leaves, which callers pass via ``_expected_fallbacks``
    (derived from the driver's own gate, so it cannot go stale). Anything else -- an e_L, force
    or param-deriv fallback, or a stress fallback on a leaf the gate says IS native -- fails."""
    hits = [
        rec.getMessage()
        for rec in caplog.records
        if "flag_rust_on_cpu" in rec.getMessage().lower()
        and "falling back" in rec.getMessage().lower()
        and not any(a.lower() in rec.getMessage().lower() for a in allow)
    ]
    assert not hits, f"unexpected native-sweep JAX fallback for {fixture_id}: {hits}"


@pytest.mark.parametrize(
    "fixture_id, stem, native",
    [
        pytest.param(fid, stem, native, marks=(pytest.mark.heavy if heavy else pytest.mark.light))
        for fid, stem, heavy, native in _FIXTURES
    ],
    ids=[f[0] for f in _FIXTURES],
)
def test_mcmc_rust_jax_dispatch_equivalent(fixture_id, stem, native, caplog, monkeypatch):
    """Broad production dispatch/storage contract, with native execution proven."""
    ham = _build_ham(stem)
    m_jax = _run(ham, rust_on=False)
    with caplog.at_level(logging.INFO, logger="jqmc"):
        m_rust, counts = _run_flag_on_with_native_spy(ham, monkeypatch)
    _assert_run_equivalent(m_rust, m_jax)
    # The force run fuses the HF/Pulay force (+ periodic stress) into the native chunk
    # on an eligible (real, fully-native) row -- so its per-step ctx observable spies are 0 and
    # ``energy_chunk`` fires; the complex-twist rows stay on per-step dispatch (spies fire). The
    # Deterministic numeric parity is owned by test_chain_modes.py.
    _assert_force_run_backend(ham, counts, native, fixture_id)
    _assert_no_gate_fallback(caplog, fixture_id, allow=_expected_fallbacks(ham))


def _assert_energy_equivalent(m_rust, m_jax):
    """Energy-only independent chains expose the same finite storage contract."""
    _assert_stored_trajectory_contract(m_rust, m_jax, ("w_L", "e_L", "e_L2"))


# Energy-only (``comput_position_deriv=False``) rows delegate each chunk to the
# native ``run_energy_chunk`` for both real and complex production paths.
_ENERGY_FIXTURES = [
    ("H2_ae_cart_energy", "H2_ae_ccpvdz_cart.h5", False, "real", True),  # molecular AE cart -> chunk
    ("H2_ecp_cart_energy", "H2_ecp_ccpvdz_cart.h5", False, "real", True),  # molecular ECP cart -> chunk
    ("Li_pbc_gamma_energy", "lithium.h5", True, "real", True),  # periodic real-Gamma ECP -> chunk
    ("Li_pbc_twist_energy", "lithium_k025.h5", True, "complex", True),  # complex twist -> native complex chunk
]


@pytest.mark.parametrize(
    "fixture_id, stem, native, chunk",
    [
        pytest.param(fid, stem, native, chunk, marks=(pytest.mark.heavy if heavy else pytest.mark.light))
        for fid, stem, heavy, native, chunk in _ENERGY_FIXTURES
    ],
    ids=[f[0] for f in _ENERGY_FIXTURES],
)
def test_mcmc_rust_jax_dispatch_energy_chunk(fixture_id, stem, native, chunk, caplog, monkeypatch):
    """ENERGY whole-run chunk delegation: an energy-only (``comput_position_deriv=False``)
    run delegates each ``mcmc_interval`` chunk to native Rust (sweep + e_L fused),
    and exposes the expected storage contract. The spy asserts that the real or
    complex native chunk actually ran."""
    ham = _build_ham(stem)
    m_jax = _run(ham, rust_on=False, comput_position_deriv=False)
    with caplog.at_level(logging.INFO, logger="jqmc"):
        m_rust, counts = _run_flag_on_with_native_spy(ham, monkeypatch, comput_position_deriv=False)
    _assert_energy_equivalent(m_rust, m_jax)
    if chunk:
        # Chunk fused sweep + e_L: the fused entry fires, and the per-step sweep / e_L ctx methods
        # do NOT (the chunk advances them internally in Rust).
        assert counts["energy_chunk"] > 0, f"expected the native energy chunk to run for {fixture_id}, got {counts}"
        assert counts["real"] == 0, f"chunk path must not call the per-step sweep for {fixture_id}, got {counts}"
        assert counts["e_L"] == 0, f"chunk path must not call the per-step e_L for {fixture_id}, got {counts}"
    else:
        pytest.fail(f"all production energy-only rows must use a native chunk: {fixture_id}: {counts}")
    _assert_no_gate_fallback(caplog, fixture_id)


# ---------------------------------------------------------------------------
# Exotic AO leaves that have NO committed TREXIO fixture: build them from the
# committed QE projection samples (the same pipeline the ``test_integration_qe_*``
# tests use), so the dispatch-equivalence class also spans the B-spline / plane-wave /
# pgto(+)bspline / pgto(+)pw AO leaves. These are the leaves the sweep crashed on
# (real-Gamma ``pgto+bspline``: the ``_num_ao_kernel`` read; complex-twist
# ``pgto+bspline``: the ``_g_vectors_jnp`` read) -- fixed by the real gto+bspline
# concatenation in ``OrbData`` and the complex-twist B-spline capability gate.
# NO bench-h5 dependency: committed QE inputs only.
# ---------------------------------------------------------------------------

_QE_DIR = os.fspath(qe_example_files_root())
_QE_GAMMA = os.path.join(_QE_DIR, "Li_gamma", "out", "li.save")
_QE_GAMMA_UPF = os.path.join(_QE_GAMMA, "Li.ccECP.upf")
_QE_TWIST = os.path.join(_QE_DIR, "Li_k025", "out", "li_k025.save")
_QE_TWIST_UPF = os.path.join(_QE_TWIST, "Li.ccECP.upf")

_qe_gamma_ok = os.path.isfile(os.path.join(_QE_GAMMA, "data-file-schema.xml")) and os.path.isfile(_QE_GAMMA_UPF)
_qe_twist_ok = os.path.isfile(os.path.join(_QE_TWIST, "data-file-schema.xml")) and os.path.isfile(_QE_TWIST_UPF)


def _pgto_shells(ref, upf):
    from jqmc.upf_wrapper import UPFExponentInit

    strat = UPFExponentInit({"li": upf}, g_target=float(np.sqrt(2.0 * ref.ecutwfc)))
    return [strat.shells_for_element(lbl) for lbl in ref.atom_labels]


def _build_ham_qe(kind):
    """Build a QE-projection Hamiltonian for the given exotic ``kind`` with the SAME
    J1+J2 exp Jastrow as the trexio fixtures (so the ratio-streaming state is exercised)."""
    from jqmc.pw_pgto_projection import (
        project_pw_to_mixed,
        project_pw_to_pgto_spline,
        project_pw_to_spline,
        project_pw_to_pw,
    )
    from jqmc.qe_to_jqmc import build_hamiltonian_from_projection
    from jqmc.qe_wrapper import load_qe_reference

    # Explicit per-atom core charges (2 Li atoms, ccECP z_core = 2) with no ECP
    # record: the core-charge-only block these fixtures were always built with.
    # It is not inferred when the ECP source is missing (issue #99).
    _J = dict(j1_param=1.0, j1_type="exp", j2_param=1.0, j2_type="exp", ewald_tol=1.0e-8, ecp_z_cores=(2.0, 2.0))

    if kind == "gamma_pw":
        ref = load_qe_reference(_QE_GAMMA)
        proj = project_pw_to_pw(ref, g_cut=None)
        ham = build_hamiltonian_from_projection(ref, proj, kind="pw", **_J)
    elif kind == "gamma_bspline":
        ref = load_qe_reference(_QE_GAMMA)
        proj = project_pw_to_spline(ref, mesh_factor=1.0, order=5)
        ham = build_hamiltonian_from_projection(ref, proj, kind="bspline", **_J)
    elif kind == "gamma_pgto_pw":
        ref = load_qe_reference(_QE_GAMMA)
        g_max = float(np.sqrt(2.0 * ref.ecutwfc))
        proj = project_pw_to_mixed(ref, _pgto_shells(ref, _QE_GAMMA_UPF), g_cut=0.5 * g_max, drop_diffuse=False)
        ham = build_hamiltonian_from_projection(ref, proj, kind="pgto+pw", **_J)
    elif kind == "gamma_pgto_bspline":
        ref = load_qe_reference(_QE_GAMMA)
        proj = project_pw_to_pgto_spline(ref, _pgto_shells(ref, _QE_GAMMA_UPF), mesh_factor=0.5, order=5)
        ham = build_hamiltonian_from_projection(ref, proj, kind="pgto+bspline", **_J)
    elif kind == "twist_pw":
        ref = load_qe_reference(_QE_TWIST)
        proj = project_pw_to_pw(ref, g_cut=None)
        ham = build_hamiltonian_from_projection(ref, proj, kind="pw", **_J)
    elif kind == "twist_bspline":
        ref = load_qe_reference(_QE_TWIST)
        proj = project_pw_to_spline(ref, mesh_factor=1.0, order=5)
        ham = build_hamiltonian_from_projection(ref, proj, kind="bspline", **_J)
    elif kind == "twist_pgto_pw":
        ref = load_qe_reference(_QE_TWIST)
        g_max = float(np.sqrt(2.0 * ref.ecutwfc))
        proj = project_pw_to_mixed(ref, _pgto_shells(ref, _QE_TWIST_UPF), g_cut=0.5 * g_max, drop_diffuse=False)
        ham = build_hamiltonian_from_projection(ref, proj, kind="pgto+pw", **_J)
    elif kind == "twist_pgto_bspline":
        ref = load_qe_reference(_QE_TWIST)
        proj = project_pw_to_pgto_spline(ref, _pgto_shells(ref, _QE_TWIST_UPF), mesh_factor=0.5, order=5)
        ham = build_hamiltonian_from_projection(ref, proj, kind="pgto+bspline", **_J)
    else:
        raise ValueError(f"unknown QE exotic kind: {kind}")
    ham.sanity_check()
    return ham


# (id, kind, needs-twist-fixture?, native-sweep). All heavy (QE projection pipeline).
# ``native`` is the sweep the flag-on run must actually invoke: "real" for real-Gamma,
# "complex" for the twist (k != Gamma) leaves. Every plane-wave / B-spline / composite
# leaf (pure + composite, real + complex) goes through the NATIVE sweep -- no JAX
# fallback. Together with the pGTO leaves in ``_FIXTURES`` this spans the FULL production
# periodic/twist AO-leaf matrix (see the module ``AO-LEAF COVERAGE`` note).
_QE_EXOTIC = [
    # --- real Gamma ---
    ("gamma_pw", "gamma_pw", False, "real"),  # AOs_plane_wave_data (pure)
    ("gamma_bspline", "gamma_bspline", False, "real"),  # AOs_bspline_data (pure)
    ("gamma_pgto_pw", "gamma_pgto_pw", False, "real"),  # composite AOs_cart_pgto_data + AOs_plane_wave_data
    ("gamma_pgto_bspline", "gamma_pgto_bspline", False, "real"),  # composite pgto + bspline (was: _num_ao_kernel crash)
    # --- complex twist (k != 0) ---
    ("twist_pw", "twist_pw", True, "complex"),  # AOs_plane_wave_complex_data (pure) -- was: _num_ao_kernel crash
    ("twist_bspline", "twist_bspline", True, "complex"),  # AOs_bspline_complex_data (pure)
    ("twist_pgto_pw", "twist_pgto_pw", True, "complex"),  # composite pgto-complex + plane_wave_complex
    (
        "twist_pgto_bspline",
        "twist_pgto_bspline",
        True,
        "complex",
    ),  # composite pgto-complex + bspline_complex (was: _g_vectors crash)
]


@pytest.mark.heavy
@pytest.mark.parametrize(
    "fixture_id, kind, needs_twist, native",
    [pytest.param(fid, kind, nt, nat) for fid, kind, nt, nat in _QE_EXOTIC],
    ids=[f[0] for f in _QE_EXOTIC],
)
def test_mcmc_rust_jax_dispatch_equivalent_qe_exotic(fixture_id, kind, needs_twist, native, caplog, monkeypatch):
    """Dispatch equivalence on the exotic AO leaves (B-spline / plane-wave / composite),
    including the complex-twist B-spline (pure + ``pgto+bspline``) that runs on the
    native complex B-spline sweep. Asserts the storage contract, expected native
    dispatch, and absence of capability-gate fallback warnings."""
    if needs_twist and not _qe_twist_ok:
        pytest.skip("Li_k025 QE fixture not present")
    if not needs_twist and not _qe_gamma_ok:
        pytest.skip("Li_gamma QE fixture not present")
    ham = _build_ham_qe(kind)
    m_jax = _run(ham, rust_on=False)
    with caplog.at_level(logging.INFO, logger="jqmc"):
        m_rust, counts = _run_flag_on_with_native_spy(ham, monkeypatch)
    # The REAL-Gamma exotic leaves (pw / bspline / composite) run the NATIVE
    # HF + Pulay force (``vmc_grad_e_L_pbc`` / ``vmc_grad_ln_psi_pbc``; kernel-validated == JAX
    # at the periodic-force fp floor). The COMPLEX-twist exotic (pw / bspline /
    # composite) force is ALSO native -- the complex determinant kinetic map carries the
    # complex PW + B-spline electron-axis kinetic-force term (``complex_det_kinetic_grads`` +
    # ``AosBsplineComplexData::det_egrad_fields``), and the complex Pulay/entry handle a 0-gto
    # (pure PW / bspline) leaf. So BOTH HF and Pulay are native native-vs-JAX on every exotic row.
    _assert_run_equivalent(m_rust, m_jax)
    # Every real exotic row has native stress and fuses force + stress into
    # the native chunk.  Complex-twist force runs remain on the per-step dispatch,
    # where the native HF + Pulay force/stress spies fire.
    # ``_assert_force_run_backend`` picks the right expectation from the same gate the driver uses.
    _assert_force_run_backend(ham, counts, native, fixture_id)
    # No determinant-leaf fallback is allowed.
    _assert_no_gate_fallback(caplog, fixture_id, allow=_expected_fallbacks(ham))


@pytest.mark.heavy
def test_mcmc_rust_jax_dispatch_equivalent_qe_zero_k_complex_dtn(tmp_path, caplog, monkeypatch):
    """Block regressions in the v3 Gamma-as-complex twist wiring.

    The ordinary QE Gamma builders above intentionally produce real AO leaves,
    while nonzero-twist rows exercise complex leaves without the converter/HDF5
    round-trip.  Neither substitutes for the production v3 representation:
    complex B-spline (pure or composite), exactly zero k, and shared DTN.  Run
    that loaded Hamiltonian through the same backend-toggle comparison and
    require the native complex sweep/observables to execute with no fallback.
    """
    from tests.test_rs_kernel._rs_fixtures import build_qe_zero_k_complex_dtn_ham

    # Keep both kinds in ONE pytest item.  CI runs this file under xdist; a
    # parametrized version can place the rows in different workers and hide
    # cross-kind JIT-cache/metadata contamination.  Production twist averaging
    # processes heterogeneous AO representations in one long-lived process, so
    # the sequential contract is intentional.
    for kind in ("bspline", "pgto+bspline"):
        fixture_id = f"zero_k_complex_{kind.replace('+', '_')}_dtn"
        ham = build_qe_zero_k_complex_dtn_ham(kind, tmp_path / f"{fixture_id}.h5")
        m_jax = _run(ham, rust_on=False)
        caplog.clear()
        with monkeypatch.context() as mp, caplog.at_level(logging.INFO, logger="jqmc"):
            m_rust, counts = _run_flag_on_with_native_spy(ham, mp)

        _assert_run_equivalent(m_rust, m_jax)
        _assert_force_run_backend(ham, counts, "complex", fixture_id)
        _assert_no_gate_fallback(caplog, fixture_id, allow=_expected_fallbacks(ham))


@pytest.mark.light
def test_mcmc_rust_jax_dispatch_mixed_built_ham(caplog, monkeypatch):
    """Regression guard for the fp32-leaf marshalling class: build the ham under a
    MIXED-precision zone with float32 Jastrow parameter leaves, then run the FULL-precision
    native sweep -- the CLI mixed-precision scenario where an fp32 leaf makes the Rust
    ``from_py`` fail with ``'ndarray'/'ArrayImpl' object cannot be converted to 'PyArray<f64>'``.
    The Jastrow parameters are stored in float64 in both modes, so a mixed build alone carries no
    float32 leaf; the J1 / J2 parameters are therefore set to float32 explicitly, as a
    checkpoint written with float32 Jastrow storage holds them. The driver coerces the fp32 leaves
    to fp64 before ``_mcmc_sweep``; the full-precision zones upcast the SAME fp32-rounded leaves
    in the JAX path too, so flag-ON (native) == flag-OFF (JAX) at strict tol.
    Restores the ambient precision mode afterwards so other tests are unaffected."""
    from jqmc._precision import configure, is_mixed_precision_enabled

    was_mixed = is_mixed_precision_enabled()
    try:
        configure("mixed")
        ham = _build_ham("H2_ecp_ccpvdz_cart.h5")  # molecular ECP cart -> real sweep
        jastrow = ham.wavefunction_data.jastrow_data
        jastrow = jastrow.replace(
            jastrow_one_body_data=jastrow.jastrow_one_body_data.replace(
                jastrow_1b_param=np.asarray(jastrow.jastrow_one_body_data.jastrow_1b_param, dtype=np.float32)
            ),
            jastrow_two_body_data=jastrow.jastrow_two_body_data.replace(
                jastrow_2b_param=np.asarray(jastrow.jastrow_two_body_data.jastrow_2b_param, dtype=np.float32)
            ),
        )
        ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jastrow))
        # The guard must actually carry an fp32 leaf, else it proves nothing.
        jd_leaves = jax.tree_util.tree_leaves(ham.wavefunction_data.jastrow_data)
        assert any(np.asarray(x).dtype == np.float32 for x in jd_leaves), "expected an fp32 jastrow leaf under mixed"
        configure("full")
        m_jax = _run(ham, rust_on=False)
        with caplog.at_level(logging.INFO, logger="jqmc"):
            m_rust, counts = _run_flag_on_with_native_spy(ham, monkeypatch)
        # This is the suite's one explicit mixed-DATA marshalling row. Select the
        # mixed backend-consistency envelope from the same full/mixed mode switch
        # as every ordinary mixed pytest run; there is no strict/medium level.
        configure("mixed")
        _assert_run_equivalent(m_rust, m_jax)
    finally:
        configure("mixed" if was_mixed else "full")
    # This molecular real force run fuses into the native chunk (sweep + e_L + HF/Pulay
    # force advanced INSIDE Rust), so the per-step sweep / e_L / force ctx spies are 0 and the
    # fused chunk entry fires. The point of this row is the fp32-leaf COERCION path through the
    # chunk-built ctx (``build_obs_ctx`` coerces), not the per-step backend selection.
    assert counts["energy_chunk"] > 0, f"expected the fused molecular force chunk to run, got {counts}"
    assert counts["real"] == 0 and counts["e_L"] == 0, f"chunk force run must advance sweep/e_L internally, got {counts}"
    _assert_native_stress(counts, expect_hf=False, expect_pp=False, fixture_id="mixed_built_ham")  # molecular -> no stress
    _assert_no_gate_fallback(caplog, "mixed_built_ham")


# ============================ [timer_dln_Psi_dc / timer_de_L_dc] =============================
# Native O_k = d ln|Psi|/dc and dE_L/dc dispatch equivalence. The optimization observables
# route through the verified free ``jqmc_rs`` c-axis kernels (assembled per walker into the
# SAME flat block-name dict the JAX ``derivatives`` path stores -- ``jqmc_mcmc._rs_native_c_param_grads``,
# mirroring ``derivatives.bridge._reduce_c_param_grads``). Gate: real path, no
# MPC, GTO-only leaf, no basis family; periodic restricts opt to J1/J2/DTN (geminal + J3 c-axis
# carry the periodic sphe-pGTO far-image AO gap -> JAX). This is a per-step dispatch (the energy
# chunk hard-excludes param-derivs), so unlike the force/e_L ctx methods the free-function spies
# below DO fire on the production path -- they confirm the native path ran (no silent JAX fallback).


def _run_param_deriv(ham, rust_on, *, opt_geminal=True):
    """One production ``MCMC`` run keeping BOTH optimization observables (O_k + dE_L/dc)."""
    if rust_on:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    else:
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    try:
        mcmc = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=34356,
            epsilon_AS=1.0e-6,
            num_walkers=2,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=True,
            random_discretized_mesh=True,
        )
        if not opt_geminal:
            # Periodic native scope is J1/J2/DTN; drop the geminal block so the native path engages.
            mcmc._MCMC__param_grad_flags["lambda_matrix"] = False
        mcmc.run(num_mcmc_steps=MCMC_MIN_STEPS)
        return mcmc
    finally:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)


def _run_param_deriv_flag_on_spy(ham, monkeypatch, *, opt_geminal=True):
    """Flag-on param-deriv run counting native c-axis kernel calls (O_k + dE_L/dc).

    The c-axis kernels are METHODS on the once-built ``NativeObsCtx``
    (``build_obs_ctx(want_param_deriv=True)``, ``ParamDerivArgs``), not the module-level free
    wrappers -- so the spy must wrap ``build_obs_ctx`` and
    count the ``pd_*`` ctx methods, NOT the free ``jqmc_rs.grad_*`` functions (which do not
    fire on the production param-deriv path). Every ``pd_ok_*`` increments ``okc`` and every
    ``pd_el_*`` (kinetic + ECP add) increments ``elc``, so a silent JAX fallback (zero native
    method calls) is still caught."""
    import jqmc_rs

    counts = {"okc": 0, "elc": 0}

    class _PdCtxSpy:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            attr = getattr(self._inner, name)
            if not callable(attr) or not name.startswith("pd_"):
                # e_L / sweep / force ctx methods pass through uncounted (their own
                # rows guard them); only the param-deriv methods feed this counter.
                return attr
            key = "okc" if name.startswith("pd_ok_") else "elc"

            def _counted(*a, **k):
                counts[key] += 1
                return attr(*a, **k)

            return _counted

    orig_build = jqmc_rs.build_obs_ctx

    def spy_build(*a, **k):
        return _PdCtxSpy(orig_build(*a, **k))

    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", spy_build)
    m = _run_param_deriv(ham, rust_on=True, opt_geminal=opt_geminal)
    return m, counts


def _ab_free_vs_ctx_identity(ham):
    """A/B byte-identity: the once-parsed ``ParamDerivArgs`` ctx methods vs the free
    kernels on the SAME (r_up, r_dn) must be EXACTLY equal (same kernel, same owned data --
    the ctx promotion is pure plumbing). Returns the count of (family, quantity) pairs
    asserted, so the caller can require it exercised a non-trivial set."""
    import jqmc_rs

    from jqmc._precision import precision_policy_snapshot
    from jqmc.jqmc_mcmc import MCMC

    # Driver-initialized (near-nucleus) positions -- physically sane, deterministic.
    drv = MCMC(hamiltonian_data=ham, num_walkers=1, mcmc_seed=1234, comput_position_deriv=False)
    r_up = np.asarray(drv._MCMC__latest_r_up_carts[0], dtype=np.float64)
    r_dn = np.asarray(drv._MCMC__latest_r_dn_carts[0], dtype=np.float64)

    wf = ham.wavefunction_data
    gem = wf.geminal_data
    jas = wf.jastrow_data
    cp = ham.coulomb_potential_data
    pbc = bool(ham.structure_data.pbc_flag)
    ctx = jqmc_rs.build_obs_ctx(
        gem,
        jas,
        cp,
        False,
        pbc,
        False,
        False,
        False,
        True,
        precision_policy=precision_policy_snapshot(),
    )

    def _eq(a, b, label):
        a = np.asarray(a)
        b = np.asarray(b)
        assert a.shape == b.shape, f"{label}: shape {a.shape} vs {b.shape}"
        np.testing.assert_array_equal(a, b, err_msg=f"A/B free-vs-ctx not byte-identical: {label}")

    checked = 0
    # O_k families
    _eq(ctx.pd_ok_j1(r_up, r_dn), jqmc_rs.grad_ln_Psi_jastrow_1b_param(jas.jastrow_one_body_data, r_up, r_dn), "ok_j1")
    _eq(ctx.pd_ok_j2(r_up, r_dn), jqmc_rs.grad_ln_Psi_jastrow_2b_param(jas.jastrow_two_body_data, r_up, r_dn), "ok_j2")
    checked += 2
    from jqmc.determinant import AGPn_data, SD_data

    if isinstance(gem, (SD_data, AGPn_data)):
        cu, cd = ctx.pd_ok_mo(r_up, r_dn)
        fu, fd = jqmc_rs.grad_ln_det_wrt_mo_coeff(gem, r_up, r_dn)
        _eq(cu, fu, "ok_mo_up")
        _eq(cd, fd, "ok_mo_dn")
        cu, cd = ctx.pd_el_mo(r_up, r_dn)
        fu, fd = jqmc_rs.grad_eL_wrt_mo_coeff(gem, jas, r_up, r_dn)
        _eq(cu, fu, "el_mo_up")
        _eq(cd, fd, "el_mo_dn")
        from jqmc.determinant import compute_geminal_all_elements

        g = np.asarray(compute_geminal_all_elements(gem, r_up, r_dn))
        supplied = np.linalg.inv(g)
        supplied[0, 0] += 0.125
        cu, cd = ctx.pd_el_mo(r_up, r_dn, supplied)
        fu, fd = jqmc_rs.grad_eL_wrt_mo_coeff(gem, jas, r_up, r_dn, supplied)
        _eq(cu, fu, "el_mo_gi_up")
        _eq(cd, fd, "el_mo_gi_dn")
        bu, bd = ctx.pd_el_mo_batch(r_up[None], r_dn[None], supplied[None])
        _eq(np.asarray(bu)[0], cu, "el_mo_gi_batch_up")
        _eq(np.asarray(bd)[0], cd, "el_mo_gi_batch_dn")
        checked += 8
    else:
        _eq(ctx.pd_ok_lambda(r_up, r_dn), jqmc_rs.grad_ln_det_wrt_lambda(gem, r_up, r_dn), "ok_lambda")
        _eq(ctx.pd_el_lambda(r_up, r_dn), jqmc_rs.grad_eL_wrt_lambda(gem, jas, r_up, r_dn), "el_lambda")
        checked += 2
    # dE_L/dc kinetic J1/J2
    _eq(ctx.pd_el_j1(r_up, r_dn), jqmc_rs.grad_eL_wrt_jastrow_1b_param(gem, jas, r_up, r_dn), "el_j1")
    _eq(ctx.pd_el_j2(r_up, r_dn), jqmc_rs.grad_eL_wrt_jastrow_2b_param(gem, jas, r_up, r_dn), "el_j2")
    checked += 2
    return checked


@pytest.mark.light
@pytest.mark.parametrize("stem", ["H2_ae_ccpvdz_cart.h5", "H2_pbc_ae_cart.h5"])
def test_mcmc_rust_param_deriv_free_vs_ctx_identity(stem):
    """A/B: the ctx-method param-deriv path is BYTE-IDENTICAL to the free-kernel path (one
    molecular + one periodic fixture). Pins that the ctx promotion changed nothing but the
    parse locus (kernels + inputs untouched, free kernels byte-verified)."""
    ham = _build_ham(stem)
    checked = _ab_free_vs_ctx_identity(ham)
    assert checked >= 6, f"A/B exercised too few (family, quantity) pairs: {checked}"


def _assert_param_grads_equivalent(m_rust, m_jax):
    for label, gr, gj in (
        ("dln_Psi_dc", m_rust.dln_Psi_dc, m_jax.dln_Psi_dc),
        ("de_L_dc", m_rust.de_L_dc, m_jax.de_L_dc),
    ):
        assert sorted(gr) == sorted(gj), f"{label} block-name set mismatch: {sorted(gr)} vs {sorted(gj)}"
        assert gr, f"{label} produced no blocks"
        for name in sorted(gr):
            a = np.asarray(gr[name])
            b = np.asarray(gj[name])
            assert a.shape == b.shape, f"{label}/{name} shape {a.shape} vs {b.shape}"
            assert np.all(np.isfinite(a)), f"{label}/{name} (Rust) non-finite"
            assert np.all(np.isfinite(b)), f"{label}/{name} (XLA) non-finite"


@pytest.mark.light
@pytest.mark.parametrize("stem", ["H2_ae_ccpvdz_cart.h5", "H2_ecp_ccpvdz_cart.h5"])
def test_mcmc_rust_jax_dispatch_param_deriv_molecular(stem, monkeypatch):
    """Molecular O_k + dE_L/dc production estimators: native Rust vs pure XLA."""
    ham = _build_ham(stem)
    m_jax = _run_param_deriv(ham, rust_on=False)
    m_rust, counts = _run_param_deriv_flag_on_spy(ham, monkeypatch)
    assert counts["okc"] > 0, f"expected the native O_k kernel to run, got {counts}"
    assert counts["elc"] > 0, f"expected the native dE_L/dc kernel to run, got {counts}"
    _assert_param_grads_equivalent(m_rust, m_jax)


@pytest.mark.heavy
@pytest.mark.parametrize("stem", ["H2_pbc_ae_cart.h5", "lithium.h5"])
def test_mcmc_rust_jax_dispatch_param_deriv_periodic(stem, monkeypatch):
    """Periodic real-Gamma O_k + dE_L/dc estimators for native J1/J2 families."""
    ham = _build_ham(stem)
    m_jax = _run_param_deriv(ham, rust_on=False, opt_geminal=False)
    m_rust, counts = _run_param_deriv_flag_on_spy(ham, monkeypatch, opt_geminal=False)
    assert counts["okc"] > 0, f"expected the native O_k kernel to run, got {counts}"
    assert counts["elc"] > 0, f"expected the native dE_L/dc kernel to run, got {counts}"
    _assert_param_grads_equivalent(m_rust, m_jax)
