"""Production Rust dispatch, failure-contract, packaging, and CI invariants.

Scope of this module (the failure-contract layer, distinct from observable aggregation
and sample-generation coverage):

  * The ``flag_rust_on_CPU`` dispatch gate itself (``jqmc/_setting.py`` + the inline
    resolution at the head of ``MCMC.run`` / the ``GFMC_t``/``GFMC_n`` per-driver
    ``_resolve_native_gfmc_chunk``): env override, CPU-vs-GPU, full and mixed precision,
    extension import success/failure. Every branch must either (a) engage the native
    path, or (b) stay on JAX with **no exception and no silent state corruption**, or
    (c) raise loudly (``RuntimeError`` with the build instruction) -- never a quiet
    partial/mixed execution.
  * The OBSERVABLE-family gates inside ``MCMC.run`` (stress / non-pGTO HF-force
    leaves; see ``jqmc/jqmc_mcmc.py`` around ``_rs_hf_native`` / ``_rs_pp_native`` /
    ``_rs_stress_hf_native`` / ``_rs_stress_pp_native``). The original dispatch review found
    that these fell through to the JAX ``derivatives`` path with **zero logging**;
    that gap is CLOSED -- each gate emits a live fallback notice (mirroring
    ``GFMC``'s ``_resolve_native_gfmc_chunk``) at **WARNING** level so it is visible
    outside CLI rank 0 (MPI ranks / library use). The param-deriv gates (``_rs_okc_native`` /
    ``_rs_elc_native``) are gone entirely: those observables are
    unconditionally native, and an unimplemented family raises instead of falling
    back. ``test_periodic_opt_geminal_param_deriv_is_native_and_silent`` below pins
    the inverted contract (native + NO fallback notice), and
    ``test_rust_dispatch_equivalence._assert_no_gate_fallback`` greps the live
    string.
  * Boundary-crossing hygiene not already pinned by the observables suite: no cross-walker aliasing on
    repeated direct low-level ``jqmc_rs`` calls (a stale-view bug would show up as
    call-order-dependent results), and dtype (float64, never a silent float32/complex64
    leak) on both the low-level kernel and the production driver's stored arrays. Walker
    ORDER and dtype at the full MCMC-driver level are the subject of
    ``tests/test_rs_kernel/test_mcmc_observables.py::test_mcmc_walker_order_equivariance``
    / ``::test_mcmc_rust_on_off_handoff``; this module adds one direct-driver
    corroboration (``test_walker_order_and_dtype_preserved_across_rust_boundary``) scoped
    to the dispatch layer itself, not a full re-derivation of observables coverage.

Named tolerances only (the dedicated XLA--Rust consistency helper or a separately
documented ``jqmc._setting`` constant) -- no inline literals (enforced by
``scripts/check_no_inline_tolerances.py``).
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
import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

import jax

import jqmc_rs

from jqmc._precision import configure as _configure_precision
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import MCMC_MIN_STEPS
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from jqmc.determinant import compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.jqmc_mcmc import MCMC
from jqmc.jqmc_cli import _execution_backend_label
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._ecp_core_electrons import core_electrons_of

_TREXIO = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files")
_MESH = icosahedron_sym_mesh_Nv12

_ZONES_EL = ("wf_kinetic", "ao_eval", "det_eval", "coulomb", "jastrow_grad_lap")


def _build_ham(stem: str) -> Hamiltonian_data:
    """Small molecular AE/ECP fixture with a real (non-trivial) J1+J2 Jastrow -- mirrors
    ``test_rust_dispatch_equivalence._build_ham`` but kept local (rs-kernel test files are
    conventionally self-contained; no cross-imports between sibling rs-kernel modules)."""
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


def _tiny_config(ham, seed):
    """One deterministic (r_up, r_dn) configuration near the nuclei -- for direct
    low-level ``jqmc_rs`` kernel calls (no MCMC sampling involved)."""
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    r_up = R[np.arange(nu) % len(R)] + rng.normal(scale=0.3, size=(nu, 3))
    r_dn = R[np.arange(nd) % len(R)] + rng.normal(scale=0.3, size=(nd, 3)) if nd > 0 else np.zeros((0, 3))
    return r_up, r_dn


def _local_energy_fast(ham, r_up, r_dn):
    gd = ham.wavefunction_data.geminal_data
    w = np.asarray(_MESH.weights, dtype=np.float64)
    gp = np.asarray(_MESH.grid_points, dtype=np.float64)
    g_inv = np.linalg.inv(np.asarray(compute_geminal_all_elements(gd, r_up, r_dn)))
    return jqmc_rs.compute_local_energy_fast(ham, r_up, r_dn, w, gp, g_inv), g_inv


@pytest.fixture(autouse=True)
def _restore_env_and_precision():
    """Every test in this module pokes the dispatch env vars and/or the precision zone
    registry; restore both unconditionally so a failing assertion never leaks state into
    a sibling test (xdist runs tests from many files in the same worker process)."""
    saved_env = {"JQMC_FLAG_RUST_ON_CPU": os.environ.get("JQMC_FLAG_RUST_ON_CPU")}
    yield
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    _configure_precision("full")


# ================================ dispatch failure contract ================================


@pytest.mark.light
@pytest.mark.parametrize("precision", ["full", "mixed"])
def test_cli_reports_xla_plus_rust_for_active_native_cpu_backend(monkeypatch, precision):
    """A usable extension on either supported CPU precision route is visible in CLI metadata."""
    os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    _configure_precision(precision)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert _execution_backend_label() == "XLA + Rust"


@pytest.mark.light
def test_cli_does_not_report_rust_when_extension_import_fails(monkeypatch):
    """A requested but unusable native extension is never advertised as active."""
    os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    _configure_precision("full")
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setitem(sys.modules, "jqmc_rs", None)
    assert _execution_backend_label() == "XLA"


@pytest.mark.light
@pytest.mark.parametrize("disabled_by", ["flag", "gpu"])
def test_cli_reports_xla_when_native_cpu_backend_is_inactive(monkeypatch, disabled_by):
    """The CLI must not advertise Rust when the production dispatch cannot use it."""
    os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    _configure_precision("full")
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    if disabled_by == "flag":
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    else:
        monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert _execution_backend_label() == "XLA"


@pytest.mark.light
def test_env_flag_off_disables_native_sweep_no_error(monkeypatch):
    """``JQMC_FLAG_RUST_ON_CPU=0`` must keep the driver on the JAX path with NO exception
    and NO call into any native kernel -- the env override wins
    over the module default ``flag_rust_on_CPU=True`` unconditionally."""
    ham = _build_ham("H2_ecp_ccpvdz_cart.h5")
    os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    called = {"n": 0}
    orig = jqmc_rs.compute_local_energy_fast
    monkeypatch.setattr(
        jqmc_rs, "compute_local_energy_fast", lambda *a, **k: (called.__setitem__("n", called["n"] + 1), orig(*a, **k))[1]
    )
    mcmc = MCMC(hamiltonian_data=ham, Dt=2.0, mcmc_seed=1, epsilon_AS=1.0e-6, num_walkers=2, random_discretized_mesh=True)
    mcmc.run(num_mcmc_steps=MCMC_MIN_STEPS)
    assert called["n"] == 0, "flag_rust_on_CPU=0 (env) must never call the native e_L kernel"
    assert np.all(np.isfinite(mcmc.e_L))


@pytest.mark.light
def test_extension_import_failure_raises_runtime_error_with_build_instructions():
    """When flag + CPU + full precision all hold but ``import jqmc_rs`` fails, the driver
    must raise ``RuntimeError`` immediately, not silently continue on JAX.
    Simulate the import failure by
    setting ``sys.modules['jqmc_rs'] = None`` (the documented Python mechanism that makes
    a subsequent ``import jqmc_rs`` raise ``ImportError`` without touching the real,
    already-imported extension object used by every other test in this process)."""
    ham = _build_ham("H2_ecp_ccpvdz_cart.h5")
    os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    real_module = sys.modules.get("jqmc_rs")
    assert real_module is not None, "jqmc_rs must already be imported for this test to be meaningful"
    sys.modules["jqmc_rs"] = None
    try:
        mcmc = MCMC(hamiltonian_data=ham, Dt=2.0, mcmc_seed=1, epsilon_AS=1.0e-6, num_walkers=1, random_discretized_mesh=True)
        with pytest.raises(RuntimeError) as excinfo:
            mcmc.run(num_mcmc_steps=MCMC_MIN_STEPS)
        msg = str(excinfo.value)
        assert "jqmc_rs" in msg, f"RuntimeError message must name the missing extension: {msg!r}"
        assert "pip install -e ." in msg, f"RuntimeError message must carry the build instruction: {msg!r}"
        assert "JQMC_FLAG_RUST_ON_CPU" in msg, f"RuntimeError message must mention the escape-hatch env var: {msg!r}"
    finally:
        sys.modules["jqmc_rs"] = real_module


@pytest.mark.light
def test_mixed_precision_enables_native_dispatch(monkeypatch):
    """Mixed precision is a first-class native CPU route and reaches NativeObsCtx."""
    ham = _build_ham("H2_ecp_ccpvdz_cart.h5")
    os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    _configure_precision("mixed")
    from jqmc._precision import is_mixed_precision_enabled

    assert is_mixed_precision_enabled(), "test setup: mixed precision must actually be active"
    called = {"n": 0}
    orig = jqmc_rs.build_obs_ctx

    def _build(*args, **kwargs):
        called["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", _build)
    mcmc = MCMC(hamiltonian_data=ham, Dt=2.0, mcmc_seed=1, epsilon_AS=1.0e-6, num_walkers=2, random_discretized_mesh=True)
    mcmc.run(num_mcmc_steps=MCMC_MIN_STEPS)
    assert called["n"] > 0, "mixed precision must construct and dispatch through NativeObsCtx"
    assert _execution_backend_label() == "XLA + Rust"
    assert np.all(np.isfinite(mcmc.e_L))


@pytest.mark.light
def test_gpu_backend_disables_native_dispatch_no_error(monkeypatch):
    """The dispatch gate reads ``jax.default_backend() == "cpu"``; force it to report
    "gpu" (no GPU is present in this sandbox, but the branch condition is identical logic
    to a real GPU host) and confirm the SAME out-of-scope, no-error, no-native-call
    contract as the mixed-precision case."""
    ham = _build_ham("H2_ecp_ccpvdz_cart.h5")
    os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    called = {"n": 0}
    orig = jqmc_rs.compute_local_energy_fast
    monkeypatch.setattr(
        jqmc_rs, "compute_local_energy_fast", lambda *a, **k: (called.__setitem__("n", called["n"] + 1), orig(*a, **k))[1]
    )
    mcmc = MCMC(hamiltonian_data=ham, Dt=2.0, mcmc_seed=1, epsilon_AS=1.0e-6, num_walkers=2, random_discretized_mesh=True)
    mcmc.run(num_mcmc_steps=MCMC_MIN_STEPS)
    assert called["n"] == 0, "a non-cpu default_backend() must never dispatch to the native e_L kernel"
    assert np.all(np.isfinite(mcmc.e_L))


# ============================ documented silent-gate characterization ============================


@pytest.mark.light
def test_periodic_opt_geminal_param_deriv_is_native_and_silent(caplog):
    """A periodic ``opt_geminal`` run takes the native c-axis assembler and logs no fallback.

    A periodic run with ``opt_geminal`` active must not route ``dln_Psi/dc`` to JAX
    silently through a capability gate. The free c-axis kernels map their input onto the
    home-cell representatives jqmc's ``compute_AOs`` uses (``_rs_wrap_home_cell``; far-roamed
    parity pinned by ``test_ungate_periodic_config_param_deriv.py``), so raw-coordinate AO
    evaluation is no reason to leave the native route.

    The pin: the periodic ``opt_geminal`` O_k must be produced by the NATIVE c-axis assembler
    (spy on ``jqmc_mcmc._rs_native_c_param_grads``), and no fallback notice may fire."""
    import jqmc.jqmc_mcmc as _mcmc_mod

    ham = _build_ham("H2_pbc_ae_cart.h5")
    os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    calls = {"n": 0}
    _orig = _mcmc_mod._rs_native_c_param_grads

    def _spy(*a, **k):
        calls["n"] += 1
        return _orig(*a, **k)

    mcmc = MCMC(
        hamiltonian_data=ham,
        Dt=2.0,
        mcmc_seed=1,
        epsilon_AS=1.0e-6,
        num_walkers=1,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=False,
        random_discretized_mesh=True,
    )
    # opt_geminal on: leave every other block at its default (True) so ``lambda_matrix`` /
    # ``mo_coefficients_*`` (the opt_geminal blocks) are active -- the exact condition the
    # deleted ``_rs_pd_config_ok`` gated off on a periodic hamiltonian.
    assert mcmc._MCMC__param_grad_flags.get("lambda_matrix", True) is True
    _mcmc_mod._rs_native_c_param_grads = _spy
    try:
        with caplog.at_level(logging.DEBUG, logger="jqmc"):
            mcmc.run(num_mcmc_steps=MCMC_MIN_STEPS)
    finally:
        _mcmc_mod._rs_native_c_param_grads = _orig
    assert np.all(np.isfinite(mcmc.e_L))
    assert mcmc.dln_Psi_dc, "O_k must be populated"
    assert calls["n"] > 0, "periodic opt_geminal O_k must run on the NATIVE c-axis assembler"
    fallbacks = [
        rec.getMessage()
        for rec in caplog.records
        if "flag_rust_on_cpu" in rec.getMessage().lower() and "falling back" in rec.getMessage().lower()
    ]
    assert fallbacks == [], f"no param-deriv gate fallback may fire: {fallbacks}"


# ================================ boundary-crossing hygiene ================================


@pytest.mark.light
@pytest.mark.parametrize("seed,nw", [(90210, 4)])
def test_walker_order_and_dtype_preserved_across_rust_boundary(seed, nw):
    """Four walkers retain dtype, distinct histories and endpoint/energy correspondence.

    Run the original H2 ECP/J1+J2 fixture and seed 90210 with Rust on/off for
    MCMC_MIN_STEPS, without comparing separately evolving trajectories. On
    each run's actual final up/down coordinates, recompute each walker's
    local energy through the XLA full route and compare to its own final
    stored entry using the existing backend-consistency policy. The fixture
    has zero non-local coefficients, so its random quadrature orientation
    cannot affect this energy reference. Distinctness/finite/shape checks
    remain non-vacuity checks, not replacements for that numerical oracle.
    """
    ham = _build_ham("H2_ecp_ccpvdz_cart.h5")

    def _run(rust_on):
        if rust_on:
            os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
        else:
            os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
        m = MCMC(hamiltonian_data=ham, Dt=2.0, mcmc_seed=seed, epsilon_AS=1.0e-6, num_walkers=nw, random_discretized_mesh=True)
        m.run(num_mcmc_steps=MCMC_MIN_STEPS)
        return m

    m_rust = _run(True)
    m_jax = _run(False)

    e_rust = m_rust.e_L
    e_jax = m_jax.e_L
    assert e_rust.dtype == np.float64, f"native e_L must stay float64, got {e_rust.dtype}"
    assert e_jax.dtype == np.float64, f"JAX e_L must stay float64, got {e_jax.dtype}"
    assert e_rust.shape == (MCMC_MIN_STEPS, nw)

    # Structural distinctness (exact equality check, not a tolerance): walkers must not
    # have collapsed onto each other, else the order-sensitive comparison below would be
    # vacuous. Independent MCMC chains agreeing bit-for-bit at every step is not a
    # legitimate physical outcome (it would mean the per-walker RNG fold failed).
    for i in range(1, nw):
        assert not np.array_equal(e_rust[:, 0], e_rust[:, i]), f"walkers 0 and {i} are bitwise identical -- RNG fold broken"

    for values, backend in ((e_rust, "Rust"), (e_jax, "XLA")):
        assert np.all(np.isfinite(values)), f"{backend} walker output contains non-finite values"
    from jqmc.hamiltonians import compute_local_energy

    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="strict")
    for backend, driver, actual in (("rust", m_rust, e_rust[-1]), ("xla", m_jax, e_jax[-1])):
        r_up_carts = np.asarray(driver._MCMC__latest_r_up_carts)
        r_dn_carts = np.asarray(driver._MCMC__latest_r_dn_carts)
        reference = np.asarray(
            [compute_local_energy(ham, up, down, np.eye(3)) for up, down in zip(r_up_carts, r_dn_carts, strict=True)]
        )
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.light
def test_no_cross_walker_aliasing_on_repeated_direct_kernel_calls():
    """Direct, driver-independent probe of the ``jqmc_rs`` marshalling boundary itself:
    call ``compute_local_energy_fast`` for configuration A, then B (a totally different
    electron placement), then A again. If the PyO3 binding held a stale view/reference
    into the LAST-marshalled numpy buffer instead of copying/reading it fresh each call
    (the marshalling-copy failure class), the third call
    would silently reproduce B's result instead of A's, or raise from a use-after-free/
    aliasing conflict. Also pins float64 dtype end to end (no fp32 leak)."""
    ham = _build_ham("H2_ae_ccpvdz_cart.h5")
    r_up_a, r_dn_a = _tiny_config(ham, seed=101)
    r_up_b, r_dn_b = _tiny_config(ham, seed=202)
    # Configurations must actually differ for this probe to mean anything.
    assert not np.array_equal(r_up_a, r_up_b)

    e_a1, ginv_a1 = _local_energy_fast(ham, r_up_a, r_dn_a)
    e_b, _ = _local_energy_fast(ham, r_up_b, r_dn_b)
    e_a2, ginv_a2 = _local_energy_fast(ham, r_up_a, r_dn_a)

    assert isinstance(e_a1, float) or np.asarray(e_a1).dtype == np.float64
    assert np.asarray(ginv_a1).dtype == np.float64
    assert np.array_equal(np.asarray(ginv_a1), np.asarray(ginv_a2)), "geminal_inv for the SAME config must be reproducible"
    assert e_a1 == e_a2, f"call A, B, A must reproduce A's exact result (no cross-call aliasing): {e_a1} != {e_a2}"
    assert e_a1 != e_b, "sanity: A and B must be genuinely different configurations/results"

    # Mutating the numpy arrays AFTER the call must not retroactively change anything --
    # proves the Rust side does not hold a live view into caller-owned memory once the
    # call has returned (a stale-view/aliasing bug would corrupt a LATER read of the
    # SAME returned scalar only if it were lazily materialized, which floats are not;
    # the meaningful assertion is that a subsequent, independent call is unaffected).
    r_up_a[:] = 0.0
    e_a3, _ = _local_energy_fast(ham, r_up_b, r_dn_b)
    assert e_a3 == e_b, "an unrelated later call must be unaffected by mutating a PREVIOUS call's input array"
