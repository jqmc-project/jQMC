"""Blocking Rust contract for the canonical, energy-only MPC observable.

The MPC design is ``physical = no``: Ewald remains the sampled/projected
Hamiltonian and MPC contributes only the passive observable

    dV_mpc = V_ee^MPC - V_ee^Ewald.

It is a production-domain test and has four independent layers:

* the Python production surface must be the energy-only API;
* the PyO3 surface must contain value/correction entries and no MPC derivatives;
* a compact pairwise fixture matrix and valid-table boundary rows must satisfy
  XLA == Rust for the correction on non-vacuous configurations;
* MCMC, GFMC-t and GFMC-n each have a driver-level route test
  (``test_mpc_correction_production_route_is_blocking``).

Do not turn a missing API, fixture-build error or numerical mismatch into a
skip/xfail/fallback: a missing or incomplete Rust port must fail this module.
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

import importlib
from pathlib import Path

import numpy as np
import pytest

import jqmc_rs

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import (
    ATOL_PYTHON_FP64_MEDIUM,
    ATOL_REDUCTION_ORDER,
    MAXULP_MPC_SENTINEL_HISTORY,
    RTOL_PYTHON_FP64_ZERO,
    RTOL_REDUCTION_ORDER,
)

from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = pytest.mark.without_chdir

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MPC_ZONES = ("coulomb", "ewald")
_FIXTURE_GRID = (6, 8, 10)


def _new_mpc_module():
    """Load the production module, failing rather than omitting the row."""
    try:
        module = importlib.import_module("jqmc.mpc")
    except Exception as exc:
        pytest.fail(
            "the MPC API (jqmc.mpc) is not importable: it is required "
            f"by the Rust observable contract ({type(exc).__name__}: {exc})"
        )
    required = ("MPC_data", "build_mpc_data", "compute_mpc_potential_ee", "compute_mpc_correction_ee")
    missing = [name for name in required if not callable(getattr(module, name, None)) and name != "MPC_data"]
    if not hasattr(module, "MPC_data"):
        missing.append("MPC_data")
    assert not missing, f"jqmc.mpc surface is incomplete: missing {missing}"
    return module


def _rust_value_entry(name: str):
    entry = getattr(jqmc_rs, name, None)
    if not callable(entry):
        pytest.fail(f"Rust MPC observable is not ported: jqmc_rs.{name} is required and must be callable")
    return entry


@pytest.mark.light
def test_mpc_python_production_surface_is_energy_only():
    """MPC must be absent from local-energy and every derivative production path."""
    _new_mpc_module()

    hamiltonians = (_REPO_ROOT / "jqmc" / "hamiltonians.py").read_text(encoding="utf-8")
    derivatives = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((_REPO_ROOT / "jqmc" / "derivatives").glob("*.py"))
    )
    mcmc = (_REPO_ROOT / "jqmc" / "jqmc_mcmc.py").read_text(encoding="utf-8")
    gfmc = (_REPO_ROOT / "jqmc" / "jqmc_gfmc.py").read_text(encoding="utf-8")

    forbidden = ("compute_local_energy_mpc", "compute_local_energy_fast_mpc", "mpc_ctx", "grad_Vmpc")
    for token in forbidden:
        assert token not in hamiltonians, f"legacy MPC local-energy contract remains in hamiltonians.py: {token}"
        assert token not in derivatives, f"legacy MPC derivative contract remains under jqmc/derivatives: {token}"
    assert "compute_mpc_correction_ee" in mcmc, "MCMC does not accumulate the passive MPC correction"
    assert "compute_mpc_correction_ee" in gfmc, "GFMC does not accumulate the passive MPC correction"


@pytest.mark.light
def test_mpc_rust_surface_is_value_only():
    """Require the correction entry and reject the obsolete derivative API."""
    _new_mpc_module()
    _rust_value_entry("_compute_mpc_potential_ee")
    _rust_value_entry("_compute_mpc_correction_ee")

    forbidden = (
        "_grad_mpc_ee_wrt_r",
        "_grad_mpc_ee_strain",
        "_mpc_dele_dc_geminal_mo",
        "_mpc_dele_dc_geminal_mo_complex",
        "_mpc_hf_force_dR_response",
    )
    present = [name for name in forbidden if hasattr(jqmc_rs, name)]
    assert not present, f"obsolete MPC analytical-derivative PyO3 entries remain: {present}"


# Six pairwise representatives, not a Cartesian product.  ``expected_axes`` is
# checked against the BUILT Hamiltonian below; it is not merely a self-declared
# coverage label.  Together the rows cover Gamma/twist, AE/ECP,
# cart/sphe-pGTO/PW/B-spline/composite, closed/open shells,
# multiple electrons, and SD/AGPn/AGP.  Jastrow is deliberately absent: it
# cannot affect MPC data or dV.
_FIXTURE_CASES = (
    dict(
        id="gamma_sphe_pgto_ecp_sd",
        builder="rs",
        args=("pbc_gamma_sphepgto_jnone",),
        expected_axes=frozenset({"gamma", "ecp", "sphe_pgto", "sd", "closed_shell"}),
    ),
    dict(
        id="gamma_cart_pgto_ae_sd",
        builder="cell",
        args=("H2_pbc_ae_cart.h5", False, None, "SD"),
        expected_axes=frozenset({"gamma", "ae", "cart_pgto", "sd", "closed_shell"}),
    ),
    dict(
        id="twist_pw_ecp_sd",
        builder="rs",
        args=("pw_twist_jnone",),
        expected_axes=frozenset({"twist", "ecp", "plane_wave", "sd", "closed_shell"}),
    ),
    dict(
        id="gamma_bspline_lsda_sd",
        builder="lsda_gamma_bspline",
        args=(),
        # ``_build_gamma`` is imported lazily from another test module, so keep
        # the committed source explicit in this case selector for GDD receipts.
        source="tests/qe_example_files/O-lsda_gamma/out/o.save",
        expected_axes=frozenset({"gamma", "ecp", "bspline", "sd", "open_shell", "multiple_electron"}),
    ),
    dict(
        id="gamma_composite_ecp_agp",
        builder="cell",
        args=("QE:Li_gamma_pgto+pw", False, None, "AGP"),
        expected_axes=frozenset({"gamma", "ecp", "composite", "cart_pgto", "plane_wave", "agp", "closed_shell"}),
    ),
    dict(
        id="twist_composite_ae_agpn",
        builder="cell",
        args=("QEPROJ:pgto+pw:H_k025:AE", False, None, "AGPn"),
        expected_axes=frozenset(
            {
                "twist",
                "ae",
                "composite",
                "cart_pgto",
                "plane_wave",
                "agpn",
                "closed_shell",
                "multiple_electron",
            }
        ),
    ),
)

_REQUIRED_FIXTURE_AXES = frozenset(
    {
        "gamma",
        "twist",
        "ae",
        "ecp",
        "cart_pgto",
        "sphe_pgto",
        "plane_wave",
        "bspline",
        "composite",
        "open_shell",
        "closed_shell",
        "multiple_electron",
        "sd",
        "agpn",
        "agp",
    }
)


@pytest.mark.light
@pytest.mark.gdd_no_committed_artifact(
    reason="metadata-only coverage audit; fixture paths are declarations and no committed artifact is materialized"
)
def test_mpc_fixture_pairwise_ledger_covers_the_reviewed_axes():
    """Deleting a representative must make the declared pairwise matrix fail."""
    ids = [case["id"] for case in _FIXTURE_CASES]
    assert len(ids) == len(set(ids)), f"duplicate MPC fixture ids: {ids}"
    covered = frozenset().union(*(case["expected_axes"] for case in _FIXTURE_CASES))
    assert covered == _REQUIRED_FIXTURE_AXES, (
        f"MPC fixture ledger mismatch: missing={sorted(_REQUIRED_FIXTURE_AXES - covered)}, "
        f"unexpected={sorted(covered - _REQUIRED_FIXTURE_AXES)}"
    )


def _build_case(case, tmp_path):
    if case["builder"] == "rs":
        return rs_ham(*case["args"])
    if case["builder"] == "cell":
        return _build_ham_for_cell(*case["args"])
    if case["builder"] == "lsda_gamma_bspline":
        # Reuse the production QE conversion already exercised by the LSDA Rust
        # suite; this supplies genuinely unequal, independent spin channels.
        from tests.test_rs_kernel.test_periodic_lsda import _build_gamma

        return _build_gamma("bspline", tmp_path)
    raise AssertionError(f"unknown MPC fixture builder {case['builder']!r}")


def _ao_axes(aos_data):
    """Return basis axes from the actual AO pytree, including composite children."""
    class_name = type(aos_data).__name__.lower()
    axes = set()
    if "composite" in class_name:
        axes.add("composite")
        for child in aos_data.children:
            axes.update(_ao_axes(child))
        return axes
    if "cart_pgto" in class_name:
        axes.add("cart_pgto")
    if "sphe_pgto" in class_name:
        axes.add("sphe_pgto")
    if "plane_wave" in class_name:
        axes.add("plane_wave")
    if "bspline" in class_name:
        axes.add("bspline")
    return axes


def _actual_fixture_axes(ham):
    """Derive every coverage label from a constructed Hamiltonian."""
    from jqmc.atomic_orbital import is_complex_aos

    geminal = ham.wavefunction_data.geminal_data
    orb_up = geminal.orb_data_up_spin
    orb_dn = geminal.orb_data_dn_spin
    aos_up = orb_up.aos_data if hasattr(orb_up, "aos_data") else orb_up
    aos_dn = orb_dn.aos_data if hasattr(orb_dn, "aos_data") else orb_dn

    axes = set(_ao_axes(aos_up)) | set(_ao_axes(aos_dn))
    axes.add("twist" if is_complex_aos(aos_up) or is_complex_aos(aos_dn) else "gamma")
    axes.add("ecp" if bool(ham.coulomb_potential_data.ecp_flag) else "ae")

    trial_name = type(geminal).__name__.lower()
    if trial_name.startswith("sd_"):
        axes.add("sd")
    elif trial_name.startswith("agpn_"):
        axes.add("agpn")
    elif trial_name.startswith("agp_"):
        axes.add("agp")
    else:
        raise AssertionError(f"unclassified MPC trial-density source: {type(geminal).__name__}")

    n_up = int(geminal.num_electron_up)
    n_dn = int(geminal.num_electron_dn)
    axes.add("closed_shell" if n_up == n_dn and n_dn > 0 else "open_shell")
    if n_up + n_dn > 2:
        axes.add("multiple_electron")
    return frozenset(axes)


def _config(ham, seed: int):
    gem = ham.wavefunction_data.geminal_data
    n_up = int(gem.num_electron_up)
    n_dn = int(gem.num_electron_dn)
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    origin = np.mean(np.asarray(ham.structure_data.positions, dtype=np.float64), axis=0)
    rng = np.random.default_rng(seed)
    frac = rng.uniform(-0.25, 1.25, size=(n_up + n_dn, 3))
    carts = origin + frac @ cell
    return np.asarray(carts[:n_up]), np.asarray(carts[n_up:])


def _assert_correction_parity(mpc_module, ham, mpc_data, r_up, r_dn, tag):
    ref_potential = float(mpc_module.compute_mpc_potential_ee(mpc_data, r_up, r_dn))
    rust_potential = _rust_value_entry("_compute_mpc_potential_ee")
    got_potential = float(rust_potential(mpc_data, r_up, r_dn))
    ref = float(
        mpc_module.compute_mpc_correction_ee(
            mpc_data,
            ham.coulomb_potential_data.ewald_data,
            r_up,
            r_dn,
        )
    )
    rust_correction = _rust_value_entry("_compute_mpc_correction_ee")
    got = float(rust_correction(mpc_data, ham.coulomb_potential_data.ewald_data, r_up, r_dn))
    assert np.isfinite(ref_potential) and np.isfinite(got_potential), (
        f"{tag}: non-finite MPC potential: XLA={ref_potential}, Rust={got_potential}"
    )
    assert np.isfinite(ref) and np.isfinite(got), f"{tag}: non-finite correction: XLA={ref}, Rust={got}"

    atol, rtol = get_tolerance_xla_rust_consistency(_MPC_ZONES, level="strict")
    np.testing.assert_allclose(
        got_potential,
        ref_potential,
        atol=atol,
        rtol=rtol,
        err_msg=f"{tag}: V_MPC XLA != Rust",
    )
    comparison_band = atol + rtol * max(1.0, abs(ref))
    assert abs(ref) > 10.0 * comparison_band, (
        f"{tag}: vacuous MPC power row, |dV|={abs(ref):.3e} is not above the comparison band {comparison_band:.3e}"
    )
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=f"{tag}: dV_mpc XLA != Rust")

    # A whole-lattice translation must leave BOTH minimum-image MPC and Ewald
    # unchanged.  Compare the correction itself, not just its two components.
    shift = np.asarray(ham.structure_data.cell, dtype=np.float64)[0]
    got_shift = float(rust_correction(mpc_data, ham.coulomb_potential_data.ewald_data, r_up + shift, r_dn + shift))
    np.testing.assert_allclose(got_shift, got, atol=atol, rtol=rtol, err_msg=f"{tag}: Rust dV lattice translation")


@pytest.mark.heavy
@pytest.mark.parametrize("case", _FIXTURE_CASES, ids=[case["id"] for case in _FIXTURE_CASES])
def test_mpc_correction_xla_rust_pairwise_fixtures(case, tmp_path):
    """Python-built MPC/Ewald data must give the same nonzero dV in XLA and Rust."""
    mpc_module = _new_mpc_module()
    ham = _build_case(case, tmp_path)
    assert bool(ham.structure_data.pbc_flag), f"{case['id']}: MPC fixture is not periodic"
    actual_axes = _actual_fixture_axes(ham)
    assert actual_axes == case["expected_axes"], (
        f"{case['id']}: built Hamiltonian no longer represents its MPC coverage row; "
        f"actual={sorted(actual_axes)}, expected={sorted(case['expected_axes'])}"
    )
    mpc_data = mpc_module.build_mpc_data(ham, grid=_FIXTURE_GRID)
    r_up, r_dn = _config(ham, seed=17)
    _assert_correction_parity(mpc_module, ham, mpc_data, r_up, r_dn, case["id"])


_BOUNDARY_ROWS = (
    pytest.param((6, 8, 10), False, False, id="noncubic_grid_outside_cell"),
    pytest.param((7, 9, 11), True, True, id="odd_grid_skew_cell_close_pair_single_spin"),
)


@pytest.mark.light
def test_mpc_boundary_ledger_keeps_the_empty_spin_channel():
    """The cheap kernel-boundary matrix must retain an ``n_dn == 0`` row."""
    assert any(bool(row.values[2]) for row in _BOUNDARY_ROWS), (
        "MPC boundary coverage lost the single-spin/empty-array row; unequal LSDA spin "
        "populations do not exercise the n_dn == 0 Rust loop/ABI boundary"
    )


@pytest.mark.heavy
@pytest.mark.parametrize("grid_shape, apply_skew, single_spin", _BOUNDARY_ROWS)
def test_mpc_correction_xla_rust_valid_table_boundaries(grid_shape, apply_skew, single_spin):
    """Exercise grid/interpolation/cell boundaries using physically built MPC tables."""
    import jax.numpy as jnp

    mpc_module = _new_mpc_module()
    ham = _build_ham_for_cell("H2_pbc_ae_cart.h5", False, None, "SD")
    if apply_skew:
        deformation = jnp.asarray([[1.0, 0.17, -0.08], [0.0, 1.0, 0.11], [0.0, 0.0, 1.0]])
        ham = ham._apply_strain(deformation)
    mpc_data = mpc_module.build_mpc_data(ham, grid=grid_shape)

    gem = ham.wavefunction_data.geminal_data
    n_up, n_dn = int(gem.num_electron_up), int(gem.num_electron_dn)
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    origin = np.mean(np.asarray(ham.structure_data.positions, dtype=np.float64), axis=0)
    base = origin + np.asarray([1.12, -0.07, 0.51]) @ cell
    all_r = np.repeat(base[None, :], n_up + n_dn, axis=0)
    all_r += np.arange(n_up + n_dn, dtype=np.float64)[:, None] * np.asarray([0.013, -0.009, 0.007])
    if single_spin:
        # MPC and Ewald e-e values are spin-blind, so repartition the same physical
        # electron coordinates into one populated channel plus an explicit (0, 3)
        # channel. This cheaply pins the Rust empty-array loop/ABI boundary without
        # introducing another expensive trial-density table build.
        r_up = np.ascontiguousarray(all_r)
        r_dn = np.empty((0, 3), dtype=np.float64)
        assert r_up.shape == (n_up + n_dn, 3) and r_dn.shape == (0, 3)
    else:
        r_up, r_dn = all_r[:n_up], all_r[n_up:]
    _assert_correction_parity(
        mpc_module,
        ham,
        mpc_data,
        r_up,
        r_dn,
        f"grid={grid_shape},skew={apply_skew},single_spin={single_spin}",
    )


# Points for the interpolator parity row: everything the periodic tricubic
# stencil can see -- nuclear clouds (where the sampled error concentrates),
# a uniform sample, exact grid nodes (t = 0), cell faces reached from both
# sides, and coordinates several cells away (index wrap in both directions).
_INTERP_PARITY_POINTS_PER_CLASS = 64


def _interp_parity_points(ham, grid_shape, seed):
    rng = np.random.default_rng(seed)
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    nuclei = np.asarray(ham.structure_data.positions, dtype=np.float64)
    n = _INTERP_PARITY_POINTS_PER_CLASS
    near = np.concatenate([q + 0.25 * rng.standard_normal((n, 3)) for q in nuclei])
    uniform = rng.uniform(0.0, 1.0, size=(n, 3)) @ cell
    idx = np.stack([rng.integers(0, g, size=n) for g in grid_shape], axis=1)
    nodes = (idx / np.asarray(grid_shape, dtype=np.float64)) @ cell
    faces = rng.uniform(0.0, 1.0, size=(n, 3))
    faces[np.arange(n), rng.integers(0, 3, size=n)] = np.where(rng.random(n) < 0.5, 0.0, 1.0)
    faces = faces @ cell
    far = rng.uniform(-2.7, 3.4, size=(n, 3)) @ cell
    return np.concatenate([near, uniform, nodes, faces, far])


@pytest.mark.light
def test_mpc_v_lr_interpolation_xla_rust_parity():
    """The tabulated ``V_LR`` interpolator itself must agree between XLA and Rust.

    ``_assert_correction_parity`` compares whole-configuration sums, where the
    minimum-image pair term dominates and a defect in the periodic tricubic
    stencil (wrong weight, off-by-one wrap, transposed axes) is diluted below
    the zone tolerance. Here the interpolator is isolated by the one-electron
    identity ``V_MPC({r}, {}) - V_const = V_LR(r)``: the Rust value minus
    ``v_const`` is compared point by point with the XLA ``evaluate_v_lr`` used
    by the Python fallback. Both sides run the same fp64 stencil in the same
    order, so the reduction-order tolerance applies, not a physics band.
    """
    mpc_module = _new_mpc_module()
    ham = _build_ham_for_cell("H2_pbc_ae_cart.h5", False, None, "SD")
    mpc_data = mpc_module.build_mpc_data(ham, grid=_FIXTURE_GRID)
    points = _interp_parity_points(ham, _FIXTURE_GRID, seed=5)

    ref = np.asarray(mpc_module.evaluate_v_lr(mpc_data, points), dtype=np.float64)
    rust_potential = _rust_value_entry("_compute_mpc_potential_ee")
    v_const = float(mpc_data.v_const)
    empty = np.empty((0, 3), dtype=np.float64)
    got = np.asarray([float(rust_potential(mpc_data, r[None, :], empty)) - v_const for r in points])

    assert np.all(np.isfinite(ref)) and np.all(np.isfinite(got))
    comparison_band = ATOL_REDUCTION_ORDER + RTOL_REDUCTION_ORDER * np.max(np.abs(ref))
    assert np.min(np.abs(ref)) > 10.0 * comparison_band, "vacuous row: V_LR is not above the comparison band"
    np.testing.assert_allclose(
        got,
        ref,
        atol=ATOL_REDUCTION_ORDER,
        rtol=RTOL_REDUCTION_ORDER,
        err_msg="V_LR tricubic interpolation XLA != Rust",
    )


_INVALID_ABI_ROWS = (
    pytest.param("cell_shape", id="cell_must_be_3x3"),
    pytest.param("zero_grid", id="grid_dimensions_must_be_positive"),
    pytest.param("coordinate_shape", id="electron_coordinates_must_be_n_by_3"),
)


@pytest.mark.light
@pytest.mark.parametrize("malformation", _INVALID_ABI_ROWS)
def test_mpc_rust_value_abi_rejects_malformed_shapes(malformation):
    """Malformed Python inputs must fail closed as ValueError, never panic."""
    import jax.numpy as jnp

    mpc_module = _new_mpc_module()
    mpc_data = mpc_module.MPC_data(
        cell=jnp.eye(3, dtype=jnp.float64) * 5.0,
        origin=jnp.zeros(3, dtype=jnp.float64),
        n_max=1,
        grid_shape=(2, 2, 2),
        v_lr_grid=jnp.zeros((2, 2, 2), dtype=jnp.float64),
        v_lr_coef=jnp.zeros((2, 2, 2), dtype=jnp.float64),
        v_const=jnp.asarray(0.0, dtype=jnp.float64),
    )
    r_up = np.zeros((1, 3), dtype=np.float64)
    r_dn = np.ones((1, 3), dtype=np.float64)
    if malformation == "cell_shape":
        mpc_data = mpc_data.replace(cell=jnp.eye(2, dtype=jnp.float64))
    elif malformation == "zero_grid":
        mpc_data = mpc_data.replace(
            grid_shape=(0, 2, 2),
            v_lr_grid=jnp.zeros((0, 2, 2), dtype=jnp.float64),
            v_lr_coef=jnp.zeros((0, 2, 2), dtype=jnp.float64),
        )
    elif malformation == "coordinate_shape":
        r_up = np.zeros((1, 2), dtype=np.float64)
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(malformation)

    with pytest.raises(ValueError):
        _rust_value_entry("_compute_mpc_potential_ee")(mpc_data, r_up, r_dn)


# Every public production driver is exercised on its single native chain.
_DRIVER_WORKLIST = tuple(pytest.param(driver, id=driver) for driver in ("mcmc", "gfmc_t", "gfmc_n"))
_DRIVER_STEPS = {"mcmc": 4, "gfmc_t": 6, "gfmc_n": 6}
_NUM_WALKERS = 2
_GFMC_COLLECT_STEPS = 2
_SENTINEL_BASE = 8192.0


def _build_run(driver, ham, mpc, seed, *, full_observables):
    """Construct and run one small but complete production trajectory."""
    if driver == "mcmc":
        from jqmc.jqmc_mcmc import MCMC

        d = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=seed,
            epsilon_AS=1.0e-6,
            num_walkers=_NUM_WALKERS,
            num_mcmc_per_measurement=2,
            comput_position_deriv=full_observables,
            comput_log_WF_param_deriv=full_observables,
            comput_e_L_param_deriv=full_observables,
            use_swct=full_observables,
            random_discretized_mesh=True,
            mpc=mpc,
        )
        d.run(num_mcmc_steps=_DRIVER_STEPS[driver])
        return d
    from jqmc.jqmc_gfmc import GFMC_n, GFMC_t

    cls = GFMC_t if driver == "gfmc_t" else GFMC_n
    nproj = {} if driver == "gfmc_t" else {"num_mcmc_per_measurement": 16}
    d = cls(
        hamiltonian_data=ham,
        num_walkers=_NUM_WALKERS,
        num_gfmc_collect_steps=_GFMC_COLLECT_STEPS,
        mcmc_seed=seed,
        alat=0.30,
        non_local_move="tmove",
        comput_position_deriv=full_observables,
        use_swct=full_observables,
        mpc=mpc,
        **nproj,
    )
    d.run(num_mcmc_steps=_DRIVER_STEPS[driver])
    return d


def _freeze_state(value):
    """Copy a nested production-state value into an equality-friendly tree."""
    if isinstance(value, dict):
        return {key: _freeze_state(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_state(item) for item in value)
    if isinstance(value, np.ndarray) or hasattr(value, "__array__"):
        return np.asarray(value).copy()
    return value


def _passive_ewald_state(driver, d):
    """All state/output that MPC physical=no is forbidden to change."""
    state = {}
    for name, value in vars(d).items():
        is_stored = "stored_" in name and "mpc_dV" not in name
        is_walker = name.endswith(("latest_r_up_carts", "latest_r_dn_carts"))
        is_counter = name.endswith(
            (
                "accepted_moves",
                "rejected_moves",
                "num_survived_walkers",
                "num_killed_walkers",
                "mcmc_counter",
            )
        )
        if is_stored or is_walker or is_counter:
            state[name] = _freeze_state(value)
    if driver == "mcmc":
        state["public_dln_Psi_dc"] = _freeze_state(d.dln_Psi_dc)
        state["public_de_L_dc"] = _freeze_state(d.de_L_dc)
        state["public_acceptance_ratio"] = d.acceptance_ratio
    return state


def _assert_state_tree_equal(got, expected, path="state"):
    if isinstance(expected, dict):
        assert got.keys() == expected.keys(), f"{path}: state keys changed: {got.keys()} != {expected.keys()}"
        for key in expected:
            _assert_state_tree_equal(got[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, tuple):
        assert len(got) == len(expected), f"{path}: sequence length changed"
        for index, (got_item, expected_item) in enumerate(zip(got, expected)):
            _assert_state_tree_equal(got_item, expected_item, f"{path}[{index}]")
        return
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(got, expected, err_msg=f"{path}: MPC changed passive Ewald state")
        return
    assert got == expected, f"{path}: MPC changed {expected!r} to {got!r}"


def _independent_dE_mpc(d, num_blocks):
    """Independent single-rank reconstruction of the documented estimator."""
    weights = np.asarray(d.w_L, dtype=np.float64)
    values = np.asarray(d.mpc_dV, dtype=np.float64)
    assert weights.shape == values.shape and values.ndim == 2
    w_blocks = np.array([np.sum(block, axis=0) for block in np.array_split(weights, num_blocks, axis=0)]).ravel()
    wv_blocks = np.array([np.sum(block, axis=0) for block in np.array_split(weights * values, num_blocks, axis=0)]).ravel()
    total_w = np.sum(w_blocks)
    total_wv = np.sum(wv_blocks)
    jackknife = np.array([(total_wv - wv) / (total_w - w) for w, wv in zip(w_blocks, wv_blocks)])
    mean = np.mean(jackknife)
    error = np.sqrt((jackknife.size - 1) * np.mean((jackknife - mean) ** 2))
    return float(mean), float(error)


def _expected_sentinel_history(driver, d):
    steps = np.arange(_DRIVER_STEPS[driver], dtype=np.float64) + _SENTINEL_BASE
    if driver == "mcmc":
        return np.repeat(steps[:, None], _NUM_WALKERS, axis=1)
    return steps[_GFMC_COLLECT_STEPS:, None]


@pytest.mark.heavy
@pytest.mark.parametrize("driver", _DRIVER_WORKLIST)
def test_mpc_correction_production_route_is_blocking(driver, monkeypatch):
    """Pin native provenance, all-step dataflow, estimator math and physical=no.

    The native spy returns a step-coded sentinel. Because both
    walkers receive the same code, *any* legitimate MCMC/GFMC weighting must
    preserve that code exactly; a decoy native call followed by a JAX result,
    a shifted step, or a broken storage path therefore fails.
    """
    mpc_module = _new_mpc_module()
    ham = _build_ham_for_cell("H2_pbc_ae_cart.h5", False, None, "SD")
    assert bool(ham.structure_data.pbc_flag), f"{driver}: route fixture is not periodic"
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")

    calls = []
    # The production route is the rayon walker-batch entry: one native call per
    # stored step over ALL walkers. The spy targets the batch
    # and unfolds it into one per-walker record so the downstream per-walker XLA=Rust
    # comparison and the step-coded sentinel logic below are unchanged.
    real_correction_batch = jqmc_rs._compute_mpc_correction_ee_batch
    _batch_step = [0]

    def _spy(mpc_data, ewald_data, r_up_b, r_dn_b):
        r_up_b = np.asarray(r_up_b)
        r_dn_b = np.asarray(r_dn_b)
        nw = r_up_b.shape[0]
        step = _batch_step[0]
        _batch_step[0] += 1
        native = np.asarray(real_correction_batch(mpc_data, ewald_data, r_up_b, r_dn_b), dtype=float)
        for w in range(nw):
            xla = float(mpc_module.compute_mpc_correction_ee(mpc_data, ewald_data, r_up_b[w], r_dn_b[w]))
            calls.append(
                {
                    "mpc_data": mpc_data,
                    "ewald_data": ewald_data,
                    "shape_up": np.asarray(r_up_b[w]).shape,
                    "shape_dn": np.asarray(r_dn_b[w]).shape,
                    "native": float(native[w]),
                    "xla": xla,
                }
            )
        return np.full(nw, _SENTINEL_BASE + step, dtype=float)

    monkeypatch.setattr(jqmc_rs, "_compute_mpc_correction_ee_batch", _spy)

    # Every driver executes the full position-derivative
    # surface, so MPC on/off byte-compares real force/stress storage rather than
    # two empty arrays. MCMC additionally covers both parameter-gradient maps.
    full_observables = True
    d_on = _build_run(driver, ham, mpc=True, seed=17, full_observables=full_observables)
    expected_calls = _DRIVER_STEPS[driver] * _NUM_WALKERS
    assert len(calls) == expected_calls, (
        f"{driver}: native MPC correction must run once per stored step/walker; "
        f"got {len(calls)}, expected {expected_calls}. JAX fallback and decoy calls are forbidden"
    )

    atol, rtol = get_tolerance_xla_rust_consistency(_MPC_ZONES, level="strict")
    comparison_bands = []
    for index, call in enumerate(calls):
        assert call["mpc_data"] is not None and call["ewald_data"] is not None
        assert len(call["shape_up"]) == 2 and call["shape_up"][1] == 3
        assert len(call["shape_dn"]) == 2 and call["shape_dn"][1] == 3
        assert np.isfinite(call["native"]) and np.isfinite(call["xla"])
        np.testing.assert_allclose(
            call["native"],
            call["xla"],
            atol=atol,
            rtol=rtol,
            err_msg=f"{driver} native call {index}: dV XLA != Rust",
        )
        comparison_bands.append(atol + rtol * max(1.0, abs(call["xla"])))
    assert max(abs(call["xla"]) for call in calls) > 10.0 * max(comparison_bands), (
        f"{driver}: all production dV values are vacuous at the comparison tolerance"
    )

    np.testing.assert_allclose(
        np.asarray(d_on.mpc_dV),
        _expected_sentinel_history(driver, d_on),
        # GFMC's weighted sum/divide can move an exactly integral sentinel
        # by one binary64 ulp; anything beyond four ulp is not that fold.
        atol=MAXULP_MPC_SENTINEL_HISTORY * np.spacing(_SENTINEL_BASE + _DRIVER_STEPS[driver]),
        rtol=RTOL_PYTHON_FP64_ZERO,
        err_msg=f"{driver}: native return values did not reach every saved MPC row",
    )

    # Reconstruct BOTH numbers returned by get_dE_mpc.  Finite-only checks are
    # forbidden because a constant (0, 0) implementation would pass them.
    got_estimator = d_on.get_dE_mpc(num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=2)
    expected_estimator = _independent_dE_mpc(d_on, num_blocks=2)
    np.testing.assert_allclose(
        got_estimator,
        expected_estimator,
        atol=ATOL_PYTHON_FP64_MEDIUM,
        rtol=RTOL_PYTHON_FP64_ZERO,
        err_msg=f"{driver}: get_dE_mpc mean/error does not match independent weighted jackknife",
    )
    assert got_estimator != (0.0, 0.0), f"{driver}: vacuous MPC estimator"

    # MPC on/off must use the SAME Rust execution mode and leave every Ewald
    # trajectory/branch/weight/force/stress/parameter-gradient output byte equal.
    d_off = _build_run(driver, ham, mpc=False, seed=17, full_observables=full_observables)
    _assert_state_tree_equal(
        _passive_ewald_state(driver, d_on),
        _passive_ewald_state(driver, d_off),
        path=driver,
    )
