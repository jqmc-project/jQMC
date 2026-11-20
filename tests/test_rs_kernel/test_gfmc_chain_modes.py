"""Native production GFMC end-to-end parity.

The GFMC drivers gained the whole-run native projection-interval chunk
(``NativeObsCtx.run_gfmc_projection_{n,t}``): one batched native call (all walkers) per
branching step replaces the JAX projection loop (GFMC_n ``fori_loop`` / GFMC_t
escape-time ``while_loop``) + the post-projection V-elements e_L, driven by the
SAME parity-pinned ``gfmc_lrdmc_{ae,ecp}`` kernels the reference chains use.
Python keeps the MPI branching / reconfiguration / E_scf update / binning.

Production has one Rust chain. End-to-end parity covers both the identity mesh and
the native Threefry/Haar random mesh. Fixed-input interval diagnostics, including
explicitly injected rotations and every projection prefix, live in
``test_gfmc_projection.py``.
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

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import (
    ATOL_PYTHON_FP64_TIGHT,
    ATOL_PYTHON_FP64_LOOSE,
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_STEPS,
    GFMC_MIN_WARMUP_STEPS,
    RTOL_PYTHON_FP64_TIGHT,
    RTOL_PYTHON_FP64_ZERO,
    get_tolerance_gfmc_fast_pbc,
)
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._ecp_core_electrons import core_electrons_of
from tests.test_rs_kernel._rs_fixtures import rs_ham

_TREXIO = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files")

# Enough branching steps to exercise storage and public-estimator assembly at
# bounded cost (GFMC steps are ~10-100x an MCMC step).
_STEPS_STAT = GFMC_MIN_STEPS
_STEPS_CHAIN = GFMC_MIN_STEPS  # chain-identity rows compare raw per-step arrays


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
        jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure_data
    )
    jastrow_data = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=None)
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    ham = Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)
    ham.sanity_check()
    return ham


class _GfmcCtxSpy:
    """Counting proxy around the once-built ``NativeObsCtx`` (the ``_CtxSpy``
    pattern of ``test_rust_dispatch_equivalence``): the drivers call the chunk as
    a METHOD on the ctx, so a module-fn monkeypatch would never fire.
    """

    counts = None  # set per test via _install

    def __init__(self, inner):
        self._inner = inner

    _TRACKED = (
        "run_gfmc_projection_n",
        "run_gfmc_projection_t",
        "run_gfmc_projection_n_complex",
        "run_gfmc_projection_t_complex",
        "run_gfmc_projection_n_batch",
        "run_gfmc_projection_t_batch",
        "run_gfmc_projection_n_complex_batch",
        "run_gfmc_projection_t_complex_batch",
    )

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name not in _GfmcCtxSpy._TRACKED or not callable(attr):
            return attr

        # Fold the complex AND batch variants into the base driver key so the tests can
        # assert on ``run_gfmc_projection_{n,t}`` regardless of (real vs complex) interval
        # or (per-walker vs rayon-batched) dispatch. NOTE post-rayon the
        # driver issues ONE batched call per branching step covering every walker, so the
        # step-count expectations below are per-step, not per-walker-per-step.
        _key = name.replace("_batch", "").replace("_complex", "")

        def _counted(*a, **k):
            _GfmcCtxSpy.counts[_key] += 1
            return attr(*a, **k)

        return _counted


def _install_ctx_spy(monkeypatch):
    import jqmc_rs

    counts = {"run_gfmc_projection_n": 0, "run_gfmc_projection_t": 0}
    _GfmcCtxSpy.counts = counts
    orig_build = jqmc_rs.build_obs_ctx

    def spy_build(*a, **k):
        return _GfmcCtxSpy(orig_build(*a, **k))

    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", spy_build)
    return counts


def _run_gfmc(
    driver,
    ham,
    seed,
    mode,
    num_steps,
    nsub=6,
    tau=0.05,
    pos_deriv=False,
    comput_stress=False,
    use_swct=False,
    random_mesh=True,
    num_walkers=2,
    *,
    non_local_move="tmove",
):
    """One production GFMC run using ``mode`` in {"jax", "rust"}.
    Returns the driver (stored per-step arrays + get_E / force buffers live on it).
    """
    if mode == "jax":
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    else:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)  # -> rust backend on (default True)
    try:
        common = dict(
            hamiltonian_data=ham,
            num_walkers=num_walkers,
            num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
            mcmc_seed=seed,
            alat=0.30,
            non_local_move=non_local_move,
            random_discretized_mesh=random_mesh,
            comput_position_deriv=pos_deriv,
            comput_stress=comput_stress,
            use_swct=use_swct,
        )
        if driver == "n":
            g = GFMC_n(**common, num_mcmc_per_measurement=nsub, E_scf=-1.0)
        else:
            g = GFMC_t(**common, tau=tau)
        g.run(num_mcmc_steps=num_steps)
        return g
    finally:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)


# Zones crossed by a deterministic end-to-end GFMC value comparison.
_PARITY_ZONES = ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic", "swct")


def _assert_value_parity(rust_value, xla_value, label, *, level="medium", periodic=False):
    """Compare deterministic end-to-end values at the propagated-state envelope.

    ``medium`` resolves to the strict tolerance in full precision and only
    widens mixed precision. Fixed-input GFMC kernel parity remains strict.

    ``periodic=True`` selects ``get_tolerance_gfmc_fast_pbc``: the same zone
    pair RAISED to the periodic fast-interval floor
    (``{ATOL,RTOL}_GFMC_FAST_PBC_FLOOR``). Under PBC the Rust GFMC
    wavefunction-derivative block runs the SAME Sherman-Morrison fast update
    XLA runs (no molecular-only gate excludes periodic leaves; realized by
    per-basis candidate routes). The
    two backends' SM carries accumulate in different fp environments, so the
    fp64 per-step diagonal ``e_L`` sits at ~1e-7 absolute instead of the
    full-recompute path's coincidental ~1e-8. It is a FLOOR, not an override:
    in mixed precision the fp32 zone envelope is larger and wins, because there
    each backend's own fp32 error already exceeds their mutual disagreement
    (see the ``_setting`` note for the measured numbers). The WALK is
    byte-identical in both lanes (the ``assert_array_equal`` legs below are
    UNCHANGED and pass), and the floor applies ONLY to periodic systems;
    molecular parity is unaffected.
    """
    get = get_tolerance_gfmc_fast_pbc if periodic else get_tolerance_xla_rust_consistency
    atol, rtol = get(_PARITY_ZONES, level=level)
    np.testing.assert_allclose(rust_value, xla_value, atol=atol, rtol=rtol, err_msg=label)


def _assert_product_buffer_parity(rust, xla, product_name, factor_name, label, *, periodic=False):
    """Verify a stored ``e_L * factor`` buffer without double-charging factor error.

    The primitive ``e_L`` and ``factor`` arrays are compared separately by the
    caller at the ordinary backend-parity tolerance.  Applying that same
    element-wise tolerance to their product is mathematically stricter: both
    admitted factor errors contribute to the product.  Instead, require each
    backend's stored buffer to equal its own product at numpy-fp64 roundoff,
    then propagate the unchanged primitive tolerance through multiplication.
    Thus an independently corrupted product buffer still fails, while valid
    backend differences already admitted for either primitive are not counted
    twice.
    """
    rust_e_l = np.asarray(rust._stored_e_L)
    xla_e_l = np.asarray(xla._stored_e_L)
    rust_factor = np.asarray(getattr(rust, factor_name))
    xla_factor = np.asarray(getattr(xla, factor_name))
    while rust_e_l.ndim < rust_factor.ndim:
        rust_e_l = rust_e_l[..., None]
        xla_e_l = xla_e_l[..., None]

    rust_product = np.asarray(getattr(rust, product_name))
    xla_product = np.asarray(getattr(xla, product_name))
    np.testing.assert_allclose(
        rust_product,
        rust_e_l * rust_factor,
        atol=ATOL_PYTHON_FP64_TIGHT,
        rtol=RTOL_PYTHON_FP64_TIGHT,
        err_msg=f"{label}: Rust stored product identity",
    )
    np.testing.assert_allclose(
        xla_product,
        xla_e_l * xla_factor,
        atol=ATOL_PYTHON_FP64_TIGHT,
        rtol=RTOL_PYTHON_FP64_TIGHT,
        err_msg=f"{label}: XLA stored product identity",
    )

    get = get_tolerance_gfmc_fast_pbc if periodic else get_tolerance_xla_rust_consistency
    atol, rtol = get(_PARITY_ZONES, level="medium")
    e_l_allowance = atol + rtol * np.abs(xla_e_l)
    factor_allowance = atol + rtol * np.abs(xla_factor)
    propagated_allowance = (
        np.abs(xla_factor) * e_l_allowance + np.abs(xla_e_l) * factor_allowance + e_l_allowance * factor_allowance
    )
    product_error = np.abs(rust_product - xla_product)
    assert np.all(product_error <= propagated_allowance), (
        f"{label}: stored product exceeds the tolerance analytically propagated "
        f"from e_L and {factor_name}; max ratio="
        f"{np.max(product_error / np.maximum(propagated_allowance, np.finfo(np.float64).tiny))}"
    )


def _assert_molecular_stress_exact_zero(run, label):
    """Molecular expected-zero contract: stored stress buffers and public stress accessors are exact zeros."""
    for name in ("_stored_stress_HF", "_stored_stress_PP", "_stored_E_L_stress_PP"):
        values = np.asarray(getattr(run, name))
        assert values.ndim == 4 and values.shape[1:] == (1, 3, 3) and values.shape[0] > 0, (
            f"{label} {name}: molecular stress buffer has shape {values.shape}"
        )
        assert np.all(values == 0), f"{label} {name}: molecular stress buffer is not exactly zero"
    args = (GFMC_MIN_WARMUP_STEPS, GFMC_MIN_BIN_BLOCKS)
    for accessor, expected in (
        ("get_aT", (np.zeros((3, 3)), np.zeros((3, 3)))),
        ("get_aP", (np.zeros(3), np.zeros(3))),
        ("get_P_hydrostatic", (0.0, 0.0)),
    ):
        observed = getattr(run, accessor)(*args)
        assert len(observed) == len(expected), f"{label} {accessor}: returned {len(observed)} values"
        for part, (value, reference) in enumerate(zip(observed, expected, strict=True)):
            value = np.asarray(value)
            assert value.shape == np.shape(reference) and np.all(value == 0), (
                f"{label} {accessor}[{part}]: molecular stress estimate is not the exact-zero {np.shape(reference)}"
            )


def _assert_deterministic_driver_parity(
    rust,
    xla,
    driver,
    *,
    position_deriv=False,
    stress=False,
    periodic=False,
    random_mesh=False,
):
    # Periodic runs engage the fast WF path on the Rust side; raise ONLY the
    # per-step / public diagonal value pins to the fast-interval floor. Threefry
    # keys remain byte-identical; the random-mesh coordinate ULP is handled below.
    vkw = dict(periodic=periodic)
    for name in ("_stored_w_L", "_stored_e_L", "_stored_e_L2"):
        _assert_value_parity(getattr(rust, name), getattr(xla, name), f"GFMC_{driver} {name}", **vkw)
    if random_mesh:
        # Threefry uniforms and every discrete decision are identical.  Native
        # libm vs XLA sin/cos can differ by one or two final ulps, which appears
        # in a selected rotated displacement even when the selected candidate is
        # the same.  Keep the established absolute fp64 accumulator pin and zero
        # relative tolerance; the fixed-mesh path below remains byte-exact.
        np.testing.assert_allclose(
            rust._latest_r_up_carts,
            xla._latest_r_up_carts,
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        np.testing.assert_allclose(
            rust._latest_r_dn_carts,
            xla._latest_r_dn_carts,
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
    else:
        np.testing.assert_array_equal(rust._latest_r_up_carts, xla._latest_r_up_carts)
        np.testing.assert_array_equal(rust._latest_r_dn_carts, xla._latest_r_dn_carts)
    np.testing.assert_array_equal(rust._jax_PRNG_key_list, xla._jax_PRNG_key_list)
    args = (GFMC_MIN_WARMUP_STEPS, GFMC_MIN_BIN_BLOCKS)
    _assert_value_parity(rust.get_E(*args), xla.get_E(*args), f"GFMC_{driver} public energy", **vkw)
    if position_deriv:
        for name in (
            "_stored_force_HF",
            "_stored_force_PP",
        ):
            _assert_value_parity(getattr(rust, name), getattr(xla, name), f"GFMC_{driver} {name}", level="medium", **vkw)
        _assert_product_buffer_parity(
            rust,
            xla,
            "_stored_E_L_force_PP",
            "_stored_force_PP",
            f"GFMC_{driver} _stored_E_L_force_PP",
            **vkw,
        )
        _assert_value_parity(rust.get_aF(*args), xla.get_aF(*args), f"GFMC_{driver} public force", level="medium", **vkw)
    if stress and not periodic:
        # Stress is defined only for periodic cells; production stores
        # exact zeros and short-circuits the public estimators for molecules.
        # Declare that contract separately from the parity below (a 0 == 0 parity
        # alone would not notice a backend that stopped zeroing).
        _assert_molecular_stress_exact_zero(rust, f"GFMC_{driver} rust")
        _assert_molecular_stress_exact_zero(xla, f"GFMC_{driver} xla")
    if stress:
        for name in ("_stored_stress_HF", "_stored_stress_PP"):
            _assert_value_parity(getattr(rust, name), getattr(xla, name), f"GFMC_{driver} {name}", level="medium", **vkw)
        _assert_product_buffer_parity(
            rust,
            xla,
            "_stored_E_L_stress_PP",
            "_stored_stress_PP",
            f"GFMC_{driver} _stored_E_L_stress_PP",
            **vkw,
        )
        _assert_value_parity(rust.get_aT(*args), xla.get_aT(*args), f"GFMC_{driver} public stress", level="medium", **vkw)


@pytest.mark.light
def test_gfmc_native_chunk_gate_matrix():
    """Driver-level dispatch pin for the native-chunk gate expressions (no runs needed --
    ``_resolve_native_gfmc_chunk`` is the single gate the run() loops consume):

    * GFMC_n and GFMC_t use the native interval whenever Rust is active.
    * ``comput_position_deriv=True`` (force/stress fold-in): ENGAGES the
      native chunk whenever the force is chunk-serviceable -- molecular ae/ecp +
      periodic-ecp AND periodic bare-AE (no family gate)
      with every real-Gamma stress leaf. Complex-twist, periodic-MPC and the
      explicit refusal rows live in
      ``test_gfmc_native_chunk_route_witness_complex_mpc_and_refusal``.
    """
    ham_mol = _build_ham("H2_ecp_ccpvdz_cart.h5")
    ham_pbc = _build_ham("lithium.h5")
    # Periodic bare-AE (ecp_flag=False) force is chunk-eligible too --
    # its cores are pinned against XLA by test_ungate_gfmc_pbc_ae_force.py, so a pgto
    # real-Gamma pGTO/PW/B-spline stress leaves are all native.
    ham_pbc_ae = _build_ham("H2_pbc_ae_cart.h5")
    assert not bool(ham_pbc_ae.coulomb_potential_data.ecp_flag), "bare-AE row must have ecp_flag=False"
    assert bool(ham_pbc_ae.structure_data.pbc_flag), "bare-AE row must be periodic"
    # Periodic-ECP-with-pGTO force is chunk-eligible; bare-AE equally so.
    _pbc_force_ok = bool(ham_pbc.coulomb_potential_data.ecp_flag)
    matrix = [
        # (driver, ham, position_deriv, expect_engaged)
        ("n", ham_mol, False, True),
        ("n", ham_mol, True, True),
        ("n", ham_pbc, False, True),
        ("n", ham_pbc, True, _pbc_force_ok),
        ("t", ham_mol, False, True),
        ("t", ham_pbc, False, True),
        ("t", ham_mol, True, True),
        ("t", ham_pbc, True, _pbc_force_ok),
        # Periodic bare-AE force ENGAGES the native chunk.
        ("n", ham_pbc_ae, True, True),
        ("t", ham_pbc_ae, True, True),
        ("n", ham_pbc_ae, False, True),
    ]
    for driver, ham, pos_deriv, expect in matrix:
        common = dict(
            hamiltonian_data=ham,
            num_walkers=2,
            num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
            mcmc_seed=11,
            alat=0.30,
            non_local_move="tmove",
            random_discretized_mesh=True,
            comput_position_deriv=pos_deriv,
        )
        if driver == "n":
            g = GFMC_n(**common, num_mcmc_per_measurement=3, E_scf=-1.0)
        else:
            g = GFMC_t(**common, tau=0.05)
        got = g._resolve_native_gfmc_chunk(driver)
        engaged = got is not None
        assert engaged == expect, (
            f"gate mismatch: driver={driver} pbc={ham.structure_data.pbc_flag} "
            f"pos_deriv={pos_deriv}: engaged={engaged}, expected {expect}"
        )


def _gate_driver(driver, ham, *, pos_deriv=False, stress=False, mpc=False):
    common = dict(
        hamiltonian_data=ham,
        num_walkers=2,
        num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
        mcmc_seed=11,
        alat=0.30,
        non_local_move="tmove",
        random_discretized_mesh=True,
        comput_position_deriv=pos_deriv,
        comput_stress=stress,
        mpc=mpc,
    )
    if driver == "n":
        return GFMC_n(**common, num_mcmc_per_measurement=3, E_scf=-1.0)
    return GFMC_t(**common, tau=0.05)


@pytest.mark.heavy
def test_gfmc_native_chunk_route_witness_complex_mpc_and_refusal(monkeypatch):
    """The native-chunk resolver's outcome is observable on every branch.

    Expected values are derived from the gate in ``_resolve_native_gfmc_chunk``:
    ``_rs_hf_native`` / ``_rs_pp_native`` are unconditionally true, so the real and
    the periodic-MPC rows engage with or without position derivatives; a complex AO
    leaf engages iff the structure is periodic, with or without force / stress and
    MPC. Each engaged row must return the statics dict with the requested
    ``complex`` / ``want_force`` / ``want_stress`` flags and record
    ``_native_gfmc_chunk_route == "native"``.

    Only explicitly inactive Rust dispatch returns ``None``:

    * Rust-on-CPU flag off (``JQMC_FLAG_RUST_ON_CPU=0``) -> ``"jax:rust-inactive"``;
    * Rust active but a complex AO leaf on a non-periodic structure. No production
      builder creates such a leaf (the complex Bloch leaves require ``pbc_flag``), so
      the driver's ``_wf_is_complex`` flag is set on a molecular run to reach the
      branch. Both drivers must raise ``RuntimeError`` and record
      ``"native-refused:complex-aos-without-pbc"``, without selecting XLA.
      Restoring the real flag recovers native dispatch. Explicit Rust-off and
      non-CPU dispatch remain XLA routes even for the otherwise refused input.
    """
    import jqmc.jqmc_gfmc as gfmc_module

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    ham_twist = rs_ham("twist_sphepgto_j12")
    ham_pbc = _build_ham("lithium.h5")
    ham_mol = _build_ham("H2_ecp_ccpvdz_cart.h5")
    assert bool(ham_twist.structure_data.pbc_flag), "complex row must be periodic"
    engaged_rows = [
        # (driver, ham, pos_deriv, stress, mpc, complex)
        (driver, ham_twist, pos_deriv, stress, mpc, True)
        for driver in ("t", "n")
        for pos_deriv, stress, mpc in ((False, False, False), (True, False, False), (True, True, False), (False, False, True))
    ] + [(driver, ham_pbc, pos_deriv, False, True, False) for driver in ("t", "n") for pos_deriv in (False, True)]
    for driver, ham, pos_deriv, stress, mpc, is_complex in engaged_rows:
        tag = f"driver={driver} complex={is_complex} pos_deriv={pos_deriv} stress={stress} mpc={mpc}"
        g = _gate_driver(driver, ham, pos_deriv=pos_deriv, stress=stress, mpc=mpc)
        assert bool(g._wf_is_complex) is is_complex, tag
        got = g._resolve_native_gfmc_chunk(driver)
        assert got is not None, f"native chunk refused: {tag} (route={getattr(g, '_native_gfmc_chunk_route', None)})"
        assert g._native_gfmc_chunk_route == "native", tag
        assert (got["complex"], got["want_force"], got["want_stress"]) == (is_complex, pos_deriv, stress), (tag, got)

    # Refusal: complex leaf without a periodic structure (branch reached by flag).
    for driver in ("t", "n"):
        g = _gate_driver(driver, ham_mol)
        g._wf_is_complex = True
        with pytest.raises(RuntimeError, match=f"GFMC_{driver}.*complex-aos-without-pbc.*JQMC_FLAG_RUST_ON_CPU=0"):
            g._resolve_native_gfmc_chunk(driver)
        assert g._native_gfmc_chunk_route == "native-refused:complex-aos-without-pbc", driver
        # The same driver object with the real flag restored engages, so the
        # refusal is caused by the complex/non-periodic combination alone.
        g._wf_is_complex = False
        assert g._resolve_native_gfmc_chunk(driver) is not None and g._native_gfmc_chunk_route == "native", driver
        # A non-CPU backend selects XLA before checking native eligibility.
        g._wf_is_complex = True
        with monkeypatch.context() as non_cpu:
            non_cpu.setattr(gfmc_module.jax, "default_backend", lambda: "gpu")
            assert g._resolve_native_gfmc_chunk(driver) is None
            assert g._native_gfmc_chunk_route == "jax:rust-inactive", driver

    # Rust inactive: the flag the run() head reads.
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    for driver in ("t", "n"):
        g = _gate_driver(driver, ham_pbc)
        assert g._resolve_native_gfmc_chunk(driver) is None
        assert g._native_gfmc_chunk_route == "jax:rust-inactive", driver
        g = _gate_driver(driver, ham_mol)
        g._wf_is_complex = True
        assert g._resolve_native_gfmc_chunk(driver) is None
        assert g._native_gfmc_chunk_route == "jax:rust-inactive", driver


@pytest.mark.heavy
def test_gfmc_zero_k_complex_bspline_dtn_force_engages_native(tmp_path, monkeypatch):
    """The v3 Gamma-as-complex B-spline force route must be production-native.

    Kernel parity lives in ``test_gfmc_complex_force_stress``.  This companion
    contract prevents the GFMC_n/GFMC_t drivers from retaining a type gate that
    rejects the same verified pure/composite B-spline Hamiltonians or silently
    executes the per-step JAX path.  Keep all combinations in one pytest item
    so xdist cannot mask cross-kind state/cache failures.
    """
    from tests.test_rs_kernel._rs_fixtures import build_qe_zero_k_complex_dtn_ham

    for ao_kind in ("bspline", "pgto+bspline"):
        tag = f"zero_k_complex_{ao_kind.replace('+', '_')}_dtn"
        ham = build_qe_zero_k_complex_dtn_ham(ao_kind, tmp_path / f"{tag}.h5")
        for driver in ("n", "t"):
            with monkeypatch.context() as mp:
                counts = _install_ctx_spy(mp)
                g = _run_gfmc(driver, ham, 3446, "rust", 1, nsub=1, pos_deriv=True, use_swct=True)
            key = f"run_gfmc_projection_{driver}"
            assert counts[key] >= 1, f"native {driver} force chunk did not run for {tag}: {counts}"
            for name in ("_stored_force_HF", "_stored_force_PP"):
                values = np.asarray(getattr(g, name))
                assert values.size and np.all(np.isfinite(values)), f"{name} not finite for {tag}/{driver}"


@pytest.mark.light
def test_gfmc_t_force_fold_in_engages_fast_molecular(monkeypatch):
    """GFMC_t force fold-in dispatch: a production force run engages the native
    chunk (ctx spy: one ``run_gfmc_projection_t`` per walker per step) and yields
    finite force_HF / force_PP / stress buffers. End-to-end estimator parity is
    tested below; fixed-input numeric parity lives in test_gfmc_forces.
    """
    ham = _build_ham("H2_ecp_ccpvdz_cart.h5")
    counts = _install_ctx_spy(monkeypatch)
    g = _run_gfmc("t", ham, 3446, "rust", _STEPS_CHAIN, pos_deriv=True, use_swct=True)
    assert counts["run_gfmc_projection_t"] >= _STEPS_CHAIN, f"native GFMC_t force chunk did not engage: {counts}"
    for name in ("_stored_force_HF", "_stored_force_PP"):
        a = np.asarray(getattr(g, name))
        assert a.size and np.all(np.isfinite(a)), f"{name} not finite on the native fold-in path"


@pytest.mark.parametrize(
    ("driver", "stem"),
    [
        pytest.param("n", "H2_ecp_ccpvdz_cart.h5", marks=pytest.mark.light, id="gfmc_n_H2_mol"),
        pytest.param("t", "H2_ecp_ccpvdz_cart.h5", marks=pytest.mark.light, id="gfmc_t_H2_mol"),
        pytest.param("n", "lithium.h5", marks=pytest.mark.heavy, id="gfmc_n_Li_pbc"),
        pytest.param("t", "lithium.h5", marks=pytest.mark.heavy, id="gfmc_t_Li_pbc"),
        # CARTESIAN periodic pGTO exercises the lifted WF fast path AND
        # the cross-hop periodic candidate cache at the INTERVAL level (the Li rows
        # above are spherical; the converter's production output is
        # Cartesian). Bare-AE (chunk-eligible).
        pytest.param("n", "H2_pbc_ae_cart.h5", marks=pytest.mark.heavy, id="gfmc_n_H2cart_pbc"),
        pytest.param("t", "H2_pbc_ae_cart.h5", marks=pytest.mark.heavy, id="gfmc_t_H2cart_pbc"),
    ],
)
@pytest.mark.parametrize("random_mesh", [False, True], ids=["fixed-mesh", "random-mesh"])
def test_gfmc_rust_vs_jax_energy(driver, stem, random_mesh, monkeypatch):
    """Production GFMC_n/t energy is deterministic across Rust and XLA.

    Both the identity mesh and the native Threefry/Haar random mesh must retain
    the same projection decisions. Comparison is per-step and uses only the
    XLA-Rust arithmetic tolerance.
    """
    ham = _build_ham(stem)
    seed = 34356
    xla = _run_gfmc(driver, ham, seed, "jax", _STEPS_STAT, random_mesh=random_mesh, num_walkers=1)
    counts = _install_ctx_spy(monkeypatch)
    rust = _run_gfmc(driver, ham, seed, "rust", _STEPS_STAT, random_mesh=random_mesh, num_walkers=1)

    key = f"run_gfmc_projection_{driver}"
    assert counts[key] >= _STEPS_STAT, f"native GFMC chunk did not engage: {counts}"
    _assert_deterministic_driver_parity(
        rust,
        xla,
        driver,
        periodic=bool(ham.structure_data.pbc_flag),
        random_mesh=random_mesh,
    )


@pytest.mark.heavy
@pytest.mark.parametrize(
    ("driver", "pattern"),
    [
        pytest.param("n", "pbc_gamma_sphepgto_j12", id="j12-real-pgto-sd"),
        pytest.param("t", "mol_ecp_cart_j123", id="j123-cart-sd"),
        pytest.param("n", "mol_ecp_cart_jdtn", id="jdtn-cart-sd"),
        pytest.param("t", "mol_sphe_agp_jnone", id="sphe-agp"),
        pytest.param("n", "mol_cart_agpn_jnone", id="cart-agpn"),
        pytest.param("t", "pw_gamma_jnone", id="real-pw"),
        pytest.param("n", "bspline_gamma_jnone", id="real-bspline"),
        pytest.param("t", "pbc_composite_jnone", id="real-pgto-pw"),
        pytest.param("n", "pgto_bspline_gamma_jnone", id="real-pgto-bspline"),
        pytest.param("t", "twist_sphepgto_jnone", id="complex-pgto"),
        # Complex twist WITH a Jastrow (withJ Pulay / SWCT kernels).
        pytest.param("n", "twist_sphepgto_j12", id="complex-pgto-j12"),
        pytest.param("t", "twist_sphepgto_j123", id="complex-pgto-j123"),
        # Cross the driver axis with the QE-projection leaves so each
        # driver-specific Rust interval meets the non-pGTO candidate basis classes.
        pytest.param("t", "bspline_gamma_jnone", id="real-bspline-t"),
        pytest.param("t", "pgto_bspline_gamma_jnone", id="real-pgto-bspline-t"),
        pytest.param("n", "pw_gamma_jnone", id="real-pw-n"),
        # This walk leaves the dn electron one lattice vector outside the
        # home cell inside the atom-1 ECP range (steps 21-37); the Rust composite mesh
        # force must not take the moved MO gradient at the raw mesh point there.
        pytest.param("n", "pbc_composite_jnone", id="real-pgto-pw-n"),
        # QE plane-wave / B-spline Gamma leaves with J1+J2.
        pytest.param("t", "pw_gamma_j12", id="real-pw-j12-t"),
        pytest.param("n", "bspline_gamma_j12", id="real-bspline-j12-n"),
        # Dense MO-AGP geminal on both drivers and the three
        # single-channel DTN masks.
        pytest.param("n", "mol_ecp_cart_agpmo_j12", id="agpmo-j12-n"),
        pytest.param("t", "mol_ecp_cart_agpmo_j12", id="agpmo-j12-t"),
        pytest.param("t", "mol_ecp_cart_jdtn_u", id="jdtn-u-t"),
        pytest.param("n", "mol_ecp_cart_jdtn_chi", id="jdtn-chi-n"),
        pytest.param("t", "mol_ecp_cart_jdtn_f", id="jdtn-f-t"),
    ],
)
def test_gfmc_rust_vs_xla_end_to_end_force_stress(driver, pattern, monkeypatch):
    """Production force/stress parity across the Jastrow/determinant/AO diagonal."""
    _assert_gfmc_force_stress_driver_pair(driver, pattern, rs_ham(pattern), monkeypatch)


def _assert_gfmc_force_stress_driver_pair(driver, pattern, ham, monkeypatch):
    """Run the same force/stress request through both production backends."""
    import jqmc_rs

    xla = _run_gfmc(
        driver,
        ham,
        34356,
        "jax",
        _STEPS_STAT,
        pos_deriv=True,
        comput_stress=True,
        use_swct=True,
        random_mesh=False,
        num_walkers=1,
    )
    counts = _install_ctx_spy(monkeypatch)
    jqmc_rs.gfmc_route_counters_reset()
    rust = _run_gfmc(
        driver,
        ham,
        34356,
        "rust",
        _STEPS_STAT,
        pos_deriv=True,
        comput_stress=True,
        use_swct=True,
        random_mesh=False,
        num_walkers=1,
    )
    method = f"run_gfmc_projection_{driver}"
    assert counts[method] >= _STEPS_STAT, f"{driver}/{pattern}: native interval did not run: {counts}"
    route_counts = jqmc_rs.gfmc_route_counters()
    assert route_counts[64] >= _STEPS_STAT, f"{driver}/{pattern}: projection winv was not handed to Pulay force"
    assert route_counts[65] == 0, f"{driver}/{pattern}: Pulay force rebuilt winv"
    _assert_deterministic_driver_parity(
        rust,
        xla,
        driver,
        position_deriv=True,
        stress=True,
        periodic=bool(ham.structure_data.pbc_flag),
    )


@pytest.mark.heavy
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated", seed=34356)
@pytest.mark.parametrize(
    ("driver", "pattern"),
    [
        pytest.param("t", "mol_ecp_cart_j123", id="j123-cart-sd"),
        pytest.param("n", "mol_ecp_cart_jdtn", id="jdtn-cart-sd"),
        pytest.param("t", "mol_sphe_agp_jnone", id="sphe-agp"),
        pytest.param("n", "mol_cart_agpn_jnone", id="cart-agpn"),
        pytest.param("n", "mol_ecp_cart_agpmo_j12", id="agpmo-j12-n"),
        pytest.param("t", "mol_ecp_cart_agpmo_j12", id="agpmo-j12-t"),
        pytest.param("t", "mol_ecp_cart_jdtn_u", id="jdtn-u-t"),
        pytest.param("n", "mol_ecp_cart_jdtn_chi", id="jdtn-chi-n"),
        pytest.param("t", "mol_ecp_cart_jdtn_f", id="jdtn-f-t"),
    ],
)
def test_molecular_stress_expected_zero_production_drivers(driver, pattern, monkeypatch):
    """Molecular Rust/XLA force-stress runs have exact-zero stress buffers and accessors.

    The same nine molecular fixture/driver pairs as the broader parity test
    execute native projection intervals and compare deterministic trajectories.
    Both backends must independently store zero HF, PP and energy-times-PP
    stress and return zero stress/pressure means and errors, not just agree.
    """
    ham = rs_ham(pattern)
    assert not bool(ham.structure_data.pbc_flag), "expected-zero evidence must use a molecular Hamiltonian"
    _assert_gfmc_force_stress_driver_pair(driver, pattern, ham, monkeypatch)


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["n", "t"])
def test_gfmc_rust_vs_xla_end_to_end_stress_only(driver, monkeypatch):
    """A periodic stress request must not require or retain atomic-force fields."""
    ham = rs_ham("pbc_gamma_sphepgto_j12")
    common = dict(
        seed=34356,
        mode="jax",
        num_steps=_STEPS_STAT,
        pos_deriv=False,
        comput_stress=True,
        use_swct=False,
        random_mesh=False,
        num_walkers=1,
    )
    xla = _run_gfmc(driver, ham, **common)
    counts = _install_ctx_spy(monkeypatch)
    common["mode"] = "rust"
    rust = _run_gfmc(driver, ham, **common)

    method = f"run_gfmc_projection_{driver}"
    assert counts[method] >= _STEPS_STAT, f"{driver}: native stress-only interval did not run: {counts}"
    for run in (rust, xla):
        assert run._stored_force_HF.size == 0
        assert run._stored_force_PP.size == 0
        assert run._stored_E_L_force_PP.size == 0
    _assert_deterministic_driver_parity(
        rust,
        xla,
        driver,
        stress=True,
        periodic=True,
    )


class _FrozenGfmcRun:
    """Stored-buffer stand-in for a finished GFMC run (negative control only)."""

    def __init__(self, **arrays):
        for name, value in arrays.items():
            setattr(self, name, np.array(value, copy=True))

    def get_E(self, num_mcmc_warmup_steps, num_mcmc_bin_blocks):
        e = self._stored_e_L[num_mcmc_warmup_steps:]
        return (float(np.mean(e)), float(np.std(e)), 0.0, 0.0)


def _frozen_pair():
    rng = np.random.default_rng(20260927)
    n_step = GFMC_MIN_STEPS
    base = dict(
        _stored_w_L=rng.uniform(0.5, 1.5, size=(n_step, 1)),
        _stored_e_L=rng.uniform(-8.0, -7.0, size=(n_step, 1)),
        _stored_e_L2=rng.uniform(49.0, 64.0, size=(n_step, 1)),
        _latest_r_up_carts=rng.uniform(-2.0, 2.0, size=(1, 2, 3)),
        _latest_r_dn_carts=rng.uniform(-2.0, 2.0, size=(1, 1, 3)),
        _jax_PRNG_key_list=rng.integers(0, 2**31, size=(1, 2), dtype=np.uint32),
    )
    return _FrozenGfmcRun(**base), _FrozenGfmcRun(**base)


@pytest.mark.light
@pytest.mark.parametrize("driver", ["n", "t"])
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Frozen stored-buffer stand-ins exercise the parity oracle only."
)
def test_gfmc_driver_parity_oracle_rejects_perturbed_backend(driver):
    """Negative control for ``_assert_deterministic_driver_parity``.

    Two frozen copies of the same stored run pass.  Shifting ONE stored
    ``e_L`` entry of the Rust copy by 0.1 Ha must fail on the ``_stored_e_L``
    pin, and moving one walker coordinate by one ulp must fail the
    byte-identical walk pin, in both precision lanes.
    """
    rust, xla = _frozen_pair()
    _assert_deterministic_driver_parity(rust, xla, driver, periodic=True)

    rust, xla = _frozen_pair()
    rust._stored_e_L[GFMC_MIN_WARMUP_STEPS, 0] += 0.1
    with pytest.raises(AssertionError, match=f"GFMC_{driver} _stored_e_L"):
        _assert_deterministic_driver_parity(rust, xla, driver, periodic=True)

    rust, xla = _frozen_pair()
    rust._latest_r_up_carts[0, 0, 0] = np.nextafter(rust._latest_r_up_carts[0, 0, 0], np.inf)
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        _assert_deterministic_driver_parity(rust, xla, driver, periodic=True)


class _FrozenMolecularStressRun:
    """Stored-buffer stand-in for a finished molecular stress run (negative control only)."""

    def __init__(self, stress_mean_offset=0.0):
        n_step = GFMC_MIN_STEPS
        self._stored_stress_HF = np.zeros((n_step, 1, 3, 3))
        self._stored_stress_PP = np.zeros((n_step, 1, 3, 3))
        self._stored_E_L_stress_PP = np.zeros((n_step, 1, 3, 3))
        self._stress_mean_offset = stress_mean_offset

    def get_aT(self, num_mcmc_warmup_steps, num_mcmc_bin_blocks):
        return np.full((3, 3), self._stress_mean_offset), np.zeros((3, 3))

    def get_aP(self, num_mcmc_warmup_steps, num_mcmc_bin_blocks):
        return -np.full(3, self._stress_mean_offset), np.zeros(3)

    def get_P_hydrostatic(self, num_mcmc_warmup_steps, num_mcmc_bin_blocks):
        return -self._stress_mean_offset, 0.0


@pytest.mark.light
def test_molecular_stress_expected_zero_contract_rejects_nonzero():
    """Negative control for the molecular expected-zero stress branch.

    Exact-zero stored buffers and accessors pass. One stored sigma_HF entry of
    1e-300, or a public get_aT mean of 1e-300, must fail with the specific
    expected-zero message.
    """
    _assert_molecular_stress_exact_zero(_FrozenMolecularStressRun(), "frozen")

    run = _FrozenMolecularStressRun()
    run._stored_stress_HF[GFMC_MIN_WARMUP_STEPS, 0, 0, 1] = 1e-300
    with pytest.raises(AssertionError, match="_stored_stress_HF: molecular stress buffer is not exactly zero"):
        _assert_molecular_stress_exact_zero(run, "frozen")

    with pytest.raises(AssertionError, match=r"get_aT\[0\]: molecular stress estimate is not the exact-zero"):
        _assert_molecular_stress_exact_zero(_FrozenMolecularStressRun(stress_mean_offset=1e-300), "frozen")


# Complex Jastrow derivative-cache route counters (jqmc/rs/src/gfmc.rs
# ``gfmc_route_counters``): (build, refresh, slow) per Jastrow family.
_CPLX_JASTROW_CACHE_COUNTERS = {"dtn": (31, 32, 33), "j12": (35, 36, 37), "j123": (38, 39, 40)}


def _assert_complex_driver_route_witnesses(route_counts, jastrow_kind):
    """Complex production-driver route witnesses.

    The carried complex fast state is built every interval and never declined
    ([14] > 0, [23] CPLX_STATE_GATE_MISS == 0); the pGTO complex det derivative
    cache builds and refreshes ([24], [25]) and never takes the counted slow route
    ([30]); the Jastrow family's complex derivative cache builds, refreshes and
    never falls back to the slow route; DTN serves the ECP mesh from its table ([63] > 0,
    [34] == 0).
    """
    assert route_counts[14] > 0 and route_counts[23] == 0, (
        f"complex {jastrow_kind}: complex fast state not carried "
        f"(CPLX_STATE_INTERVALS={route_counts[14]}, CPLX_STATE_GATE_MISS={route_counts[23]})"
    )
    assert route_counts[24] > 0 and route_counts[25] > 0, f"complex {jastrow_kind}: pGTO det derivative cache idle"
    assert route_counts[30] == 0, f"complex {jastrow_kind}: complex det derivative took the slow route"
    build, refresh, slow = _CPLX_JASTROW_CACHE_COUNTERS[jastrow_kind]
    assert route_counts[build] > 0, f"complex {jastrow_kind} derivative cache did not build"
    assert route_counts[refresh] > 0, f"complex {jastrow_kind} derivative cache did not refresh"
    assert route_counts[slow] == 0, f"complex {jastrow_kind} derivative cache took the slow route"
    if jastrow_kind == "dtn":
        # ECP-mesh DTN ratios are served by the tabulated ECP table ([63]); the per-point
        # cached-mesh fallback ([34] CPLX_DTN_MESH_HITS, bumped only when both tables miss,
        # gfmc.rs ``dtn ratios_for_group``) stays silent on this fixture.
        assert route_counts[63] > 0 and route_counts[34] == 0, (
            f"complex dtn: ECP mesh DTN ratios left the tabulated route "
            f"(TABPIP_ECP_HITS={route_counts[63]}, CPLX_DTN_MESH_HITS={route_counts[34]})"
        )
    if jastrow_kind == "j123":
        # J1/J2 ride alongside J3 on this fixture.
        assert route_counts[35] > 0 and route_counts[37] == 0, "complex j123: J1/J2 derivative cache misrouted"


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["n", "t"])
@pytest.mark.parametrize("jastrow_kind", ["j12", "dtn", "j123"])
def test_gfmc_complex_twist_rust_vs_jax_energy(driver, jastrow_kind, monkeypatch):
    """Complex J1/J2, DTN and J1/J2/J3 caches preserve deterministic production parity."""
    if jastrow_kind == "j12":
        ham = _build_ham("lithium_k025.h5")
    else:
        ham = rs_ham({"dtn": "twist_sphepgto_jdtn", "j123": "twist_sphepgto_j123"}[jastrow_kind])
    if jastrow_kind == "j123":
        assert ham.wavefunction_data.jastrow_data.jastrow_three_body_data is not None
    seed_a = 34356

    xla = _run_gfmc(driver, ham, seed_a, "jax", _STEPS_STAT, random_mesh=False, num_walkers=1)
    counts = _install_ctx_spy(monkeypatch)
    import jqmc_rs

    jqmc_rs.gfmc_route_counters_reset()
    rust = _run_gfmc(driver, ham, seed_a, "rust", _STEPS_STAT, random_mesh=False, num_walkers=1)

    key = f"run_gfmc_projection_{driver}"
    assert counts[key] >= _STEPS_STAT, f"native complex GFMC chunk did not engage in fast mode: {counts}"
    route_counts = list(jqmc_rs.gfmc_route_counters())
    _assert_complex_driver_route_witnesses(route_counts, jastrow_kind)
    # Negative control: a declined complex state must turn the witness red.
    mutated = list(route_counts)
    mutated[23] = 1
    with pytest.raises(AssertionError, match="complex fast state not carried"):
        _assert_complex_driver_route_witnesses(mutated, jastrow_kind)
    _assert_deterministic_driver_parity(rust, xla, driver)
