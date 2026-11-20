"""Declared reference helpers that no test used, now used as oracles.

``_smooth_periodic_distance_debug`` is documented as the fp64 NumPy reference
for the smooth periodic distance of the periodic J1/J2, and the public
wrappers ``evaluate_jastrow`` / ``evaluate_determinant`` had no caller in
``jqmc/``, ``tests/``, ``doc/`` or ``examples/``. Here each is compared with an
independent path:

* the production ``_smooth_periodic_distance`` (which runs in its caller's
  dtype; the fp64 displacement keeps it fp64 in both modes) against
  ``_smooth_periodic_distance_debug``, and the closed-form
  ``_smooth_periodic_distance_grad_lap`` (``jastrow_grad_lap`` zone) against the
  same NumPy formula evaluated in the ``jastrow_grad_lap`` zone dtype, on the
  three ``CELL_FIXTURES`` cells (value);
* the closed-form gradient and Laplacian against the autodiff gradient and
  Laplacian of a jnp transcription of the NumPy formula evaluated in the
  ``jastrow_grad_lap`` zone dtype, and that autodiff reference evaluated in
  fp64 against central finite differences of ``_smooth_periodic_distance_debug``;
* ``evaluate_jastrow`` against ``exp`` of the NumPy Jastrow reference evaluated
  in the ``jastrow_eval`` zone dtype (whose fp64 evaluation reproduces
  ``_compute_Jastrow_part_debug`` bit for bit), ``evaluate_determinant`` against
  the NumPy ``_compute_det_geminal_all_elements_debug``, and their product
  against ``evaluate_wavefunction``, on a molecular all-electron fixture and on
  the Li ccECP fixture at the k = 0.25 twist (periodic J2, complex determinant).

Each precision mode is judged against its own zone policy and, per the
same-precision reference rule, against a reference in its own precision: an fp32-zone production output is
never compared with an fp64 reference (the finite differences, whose fp32
evaluation cannot resolve a derivative at the tight central FD tolerance, are
compared with the fp64 autodiff reference in both modes and, in full mode only,
where the production derivatives are fp64, also directly with the production
derivatives as before). Negative controls scale the
production distance and the production determinant by a small factor above
the mode's tolerance and must fail with the oracle's message.

``_compute_kinetic_energy_debug`` is covered separately in
``test_wavefunction_kinetic_fd_oracle.py`` with a caller-selected step.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import jastrow_factor
from jqmc import wavefunction as wf_module
from jqmc._precision import get_dtype_np, get_tolerance_min
from jqmc._setting import ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT
from jqmc.determinant import _compute_det_geminal_all_elements_debug, compute_geminal_all_elements
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    _compute_Jastrow_part_debug,
    _compute_Jastrow_three_body_debug,
    _smooth_periodic_distance_debug,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data, evaluate_wavefunction
from tests.conftest import CELL_FIXTURES

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")
_N_DISPLACEMENTS = 40
_FD_STEP = 1.0e-4
_DISTANCE_EPS = 1.0e-12


# --------------------------------------------------------------------------- #
# Smooth periodic distance.
# --------------------------------------------------------------------------- #
def _displacements(cell, seed):
    rng = np.random.default_rng(seed)
    frac = rng.uniform(-1.5, 1.5, size=(_N_DISPLACEMENTS, 3))
    # keep every displacement away from a lattice point (the distance is 0 there)
    frac = np.where(np.abs(frac - np.round(frac)) < 0.05, frac + 0.1, frac)
    return frac @ cell


def _distance_zone_reference(r_d_cart, cell, inv_cell, dtype):
    """``_smooth_periodic_distance_debug`` with its arithmetic in ``dtype``.

    The fp64 displacement, cell and inverse cell are cast to ``dtype`` -- as the
    ``jastrow_grad_lap`` closed form casts its inputs -- and the literal debug
    formula is evaluated in that dtype. With ``dtype=np.float64`` it is the debug
    oracle bit for bit.
    """
    x = np.asarray(r_d_cart, dtype=dtype)
    cell = np.asarray(cell, dtype=dtype)
    inv_cell = np.asarray(inv_cell, dtype=dtype)
    phase = dtype(np.pi) * (x @ inv_cell)
    vector = np.sin(dtype(2) * phase) @ cell
    height = dtype(2) * np.sin(phase) ** 2
    square = np.dot(vector, vector) + np.dot(np.sum(cell * cell, axis=1), height * height)
    return np.sqrt(square) / (dtype(2) * dtype(np.pi))


def _distance_jnp(x, cell, inv_cell):
    """jnp transcription of ``_smooth_periodic_distance_debug``; arithmetic in the dtype of its inputs."""
    phase = jnp.pi * (x @ inv_cell)
    vector = jnp.sin(2 * phase) @ cell
    height = 2 * jnp.sin(phase) ** 2
    square = jnp.dot(vector, vector) + jnp.dot(jnp.sum(cell * cell, axis=1), height * height)
    return jnp.sqrt(square) / (2 * jnp.pi)


def _distance_ad_grad_lap(disp, cell, inv, dtype):
    """Autodiff gradient and Laplacian of ``_distance_jnp`` with inputs cast to ``dtype``."""
    cell_d = jnp.asarray(cell, dtype=dtype)
    inv_d = jnp.asarray(inv, dtype=dtype)

    def one(r):
        f = lambda x: _distance_jnp(x, cell_d, inv_d)  # noqa: E731
        return jax.grad(f)(r), jnp.trace(jax.hessian(f)(r))

    grad, lap = jax.vmap(one)(jnp.asarray(disp, dtype=dtype))
    return np.asarray(grad), np.asarray(lap)


def _distance_mismatches(cell_name):
    """Messages for every displacement where production and its same-precision reference differ (empty = pass)."""
    cell = CELL_FIXTURES[cell_name]
    inv = np.linalg.inv(cell)
    disp = _displacements(cell, seed=17)
    ref = np.array([_smooth_periodic_distance_debug(r, cell, inv) for r in disp])
    assert np.all(np.isfinite(ref)) and np.min(ref) > 0.0
    # The closed-form route casts to the jastrow_grad_lap zone, so its
    # reference is the debug formula evaluated in that zone dtype. Its fp64
    # evaluation is the declared fp64 debug oracle bit for bit.
    gl_dtype = get_dtype_np("jastrow_grad_lap")
    ref_gl = np.array([_distance_zone_reference(r, cell, inv, gl_dtype) for r in disp])
    np.testing.assert_array_equal(np.array([_distance_zone_reference(r, cell, inv, np.float64) for r in disp]), ref)
    assert np.all(np.isfinite(ref_gl)) and np.min(ref_gl) > 0.0
    # The direct route runs in its caller's dtype: the fp64 displacement keeps
    # it fp64 in both modes, so the fp64 debug oracle is its same-precision
    # reference.
    direct = np.asarray(
        jax.vmap(lambda r: jastrow_factor._smooth_periodic_distance(r, jnp.asarray(cell), jnp.asarray(inv)))(jnp.asarray(disp))
    )
    d_gl, grad, _q, lap = (
        np.asarray(x)
        for x in jastrow_factor._smooth_periodic_distance_grad_lap(
            jnp.asarray(disp), jnp.asarray(cell), jnp.asarray(inv), _DISTANCE_EPS
        )
    )
    failures = []
    for label, got, want, zone in (
        ("distance", direct, ref, "jastrow_eval"),
        ("grad_lap distance", d_gl, ref_gl, "jastrow_grad_lap"),
    ):
        atol, rtol = get_tolerance_min((zone,), "strict")
        assert np.all(np.isfinite(got))
        bad = ~np.isclose(got, want, atol=atol, rtol=rtol, equal_nan=False)
        if np.any(bad):
            failures.append(
                f"{cell_name}: smooth periodic {label} differs from the NumPy reference at {int(bad.sum())} points "
                f"(max |diff| {np.max(np.abs(got - want)):.3e})"
            )

    def fd_grad_lap(r):
        g = np.zeros(3)
        lp = 0.0
        f0 = _smooth_periodic_distance_debug(r, cell, inv)
        for a in range(3):
            e = np.zeros(3)
            e[a] = _FD_STEP
            fp = _smooth_periodic_distance_debug(r + e, cell, inv)
            fm = _smooth_periodic_distance_debug(r - e, cell, inv)
            g[a] = (fp - fm) / (2.0 * _FD_STEP)
            lp += (fp - 2.0 * f0 + fm) / _FD_STEP**2
        return g, lp

    # The closed-form gradient / Laplacian (jastrow_grad_lap zone) are
    # judged against the autodiff derivatives of the NumPy formula evaluated in
    # that zone dtype; the fp64 finite differences of the fp64 debug oracle
    # validate the fp64 autodiff reference (fp64 vs fp64 in both modes). In
    # full mode both references are fp64, so the production derivatives remain
    # tied to the finite differences of the debug oracle.
    fd = [fd_grad_lap(r) for r in disp]
    fd_grad = np.array([g for g, _ in fd])
    fd_lap = np.array([lp for _, lp in fd])
    ad_grad, ad_lap = _distance_ad_grad_lap(disp, cell, inv, gl_dtype)
    ad64_grad, ad64_lap = _distance_ad_grad_lap(disp, cell, inv, np.float64)
    for label, got, want in (("gradient", grad, ad_grad), ("Laplacian", lap, ad_lap)):
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(want))
        if not np.allclose(got, want, atol=ATOL_FD_VS_ANALYTIC_TIGHT, rtol=RTOL_FD_VS_ANALYTIC_TIGHT, equal_nan=False):
            failures.append(
                f"{cell_name}: smooth periodic distance {label} differs from the zone-dtype autodiff reference "
                f"(max |diff| {np.max(np.abs(got - want)):.3e})"
            )
    if gl_dtype == np.float64:
        # Full mode: the production derivatives are fp64, so the original fp64
        # comparison against finite differences of the debug oracle is kept
        # verbatim. In mixed they are fp32 and an fp64 FD reference is not
        # same-precision; checks above cover that mode.
        for label, got, want in (("gradient", grad, fd_grad), ("Laplacian", lap, fd_lap)):
            assert np.all(np.isfinite(got)) and np.all(np.isfinite(want))
            if not np.allclose(got, want, atol=ATOL_FD_VS_ANALYTIC_TIGHT, rtol=RTOL_FD_VS_ANALYTIC_TIGHT, equal_nan=False):
                failures.append(
                    f"{cell_name}: smooth periodic distance {label} differs from finite differences of the NumPy reference "
                    f"(max |diff| {np.max(np.abs(got - want)):.3e})"
                )
    for label, got, want in (("gradient", ad64_grad, fd_grad), ("Laplacian", ad64_lap, fd_lap)):
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(want))
        if not np.allclose(got, want, atol=ATOL_FD_VS_ANALYTIC_TIGHT, rtol=RTOL_FD_VS_ANALYTIC_TIGHT, equal_nan=False):
            failures.append(
                f"{cell_name}: fp64 autodiff {label} of the NumPy reference differs from its finite differences "
                f"(max |diff| {np.max(np.abs(got - want)):.3e})"
            )
    return failures


@pytest.mark.parametrize("cell_name", sorted(CELL_FIXTURES))
def test_smooth_periodic_distance_matches_numpy_debug_reference(cell_name):
    """Production smooth periodic distance (both routes) equals its NumPy reference in the route's precision."""
    failures = _distance_mismatches(cell_name)
    assert failures == [], "; ".join(failures)


def test_smooth_periodic_distance_oracle_rejects_a_scaled_distance(monkeypatch):
    """Scaling the production distance by 1 + 10 (rtol + atol / d_min) must fail the value check."""
    cell = CELL_FIXTURES["triclinic_skewed"]
    ref = np.array([_smooth_periodic_distance_debug(r, cell, np.linalg.inv(cell)) for r in _displacements(cell, seed=17)])
    atol, rtol = get_tolerance_min(("jastrow_eval",), "strict")
    factor = 1.0 + 10.0 * (rtol + atol / float(np.min(ref)))
    original = jastrow_factor._smooth_periodic_distance

    def scaled(r_d_cart, cell_, inv_cell):
        return factor * original(r_d_cart, cell_, inv_cell)

    monkeypatch.setattr(jastrow_factor, "_smooth_periodic_distance", scaled)
    failures = _distance_mismatches("triclinic_skewed")
    monkeypatch.undo()
    assert any("smooth periodic distance differs from the NumPy reference" in f for f in failures), failures


# --------------------------------------------------------------------------- #
# evaluate_jastrow / evaluate_determinant.
# --------------------------------------------------------------------------- #
_WF_FIXTURES = ("H2_ae_ccpvdz_cart.h5", "lithium_k025.h5")


def _wavefunction(trexio_file):
    structure, aos, _mos_up, _mos_dn, geminal, _coulomb = read_trexio_file(
        trexio_file=os.path.join(_FIXTURE_DIR, trexio_file), store_tuple=True
    )
    periodic = bool(structure.pbc_flag)
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure if periodic else None
    )
    j3 = (
        None
        if periodic
        else Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos, random_init=True, random_scale=1.0e-2)
    )
    jastrow = Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=j2, jastrow_three_body_data=j3)
    return structure, Wavefunction_data(geminal_data=geminal, jastrow_data=jastrow)


def _configurations(structure, geminal, seed):
    rng = np.random.default_rng(seed)
    centers = np.asarray(structure.positions)
    n_up, n_dn = geminal.num_electron_up, geminal.num_electron_dn
    for _ in range(300):
        ru = centers[rng.integers(0, len(centers), size=n_up)] + rng.normal(scale=0.7, size=(n_up, 3))
        rd = centers[rng.integers(0, len(centers), size=n_dn)] + rng.normal(scale=0.7, size=(n_dn, 3))
        g = np.asarray(compute_geminal_all_elements(geminal_data=geminal, r_up_carts=ru, r_dn_carts=rd))
        if np.all(np.isfinite(g)) and np.linalg.cond(g) < 1.0e4:
            return ru, rd
    raise AssertionError("no well-conditioned configuration in 300 deterministic draws")


def _j2_zone_reference(j2_data, r_up_carts, r_dn_carts, dtype):
    """J2 with the distance from fp64 geometry and the pair law / sum in ``dtype``.

    Literal transcription of ``_compute_Jastrow_two_body_molecular_debug`` and
    ``_compute_Jastrow_two_body_pbc_debug`` (summation order included): the
    ``r_i - r_j`` difference and, under PBC, its smooth periodic distance are
    formed in fp64 as production forms them, then cast to ``dtype``. With
    ``dtype=np.float64`` it is the matching debug twin bit for bit.
    """
    import itertools

    a = dtype(j2_data.jastrow_2b_param)
    j2b_type = j2_data.jastrow_2b_type
    assert j2b_type in ("exp", "pade"), j2b_type
    groups = (
        itertools.product(r_up_carts, r_dn_carts),
        itertools.combinations(r_up_carts, 2),
        itertools.combinations(r_dn_carts, 2),
    )
    if j2_data.structure_data.pbc_flag:
        cell = np.asarray(j2_data.structure_data.cell, dtype=np.float64)
        inv_cell = np.linalg.inv(cell)
        total = dtype(0.0)
        for group in groups:
            for r_i, r_j in group:
                diff = np.asarray(r_i, dtype=np.float64) - np.asarray(r_j, dtype=np.float64)
                r_ij = dtype(_distance_zone_reference(diff, cell, inv_cell, np.float64))
                if j2b_type == "exp":
                    total += dtype(1.0) / (dtype(2.0) * a) * (dtype(1.0) - np.exp(-a * r_ij))
                else:
                    total += r_ij / (dtype(2.0) * (dtype(1.0) + a * r_ij))
        return total

    def u(r_i, r_j):
        r_ij = np.linalg.norm(np.asarray(np.asarray(r_i) - np.asarray(r_j), dtype=dtype))
        if j2b_type == "exp":
            return dtype(1.0) / (dtype(2.0) * a) * (dtype(1.0) - np.exp(-a * r_ij))
        return r_ij / dtype(2.0) * (dtype(1.0) + a * r_ij) ** dtype(-1.0)

    anti, up_up, dn_dn = (np.sum(np.asarray([u(r_i, r_j) for r_i, r_j in group], dtype=dtype)) for group in groups)
    return anti + up_up + dn_dn


def _jastrow_zone_reference(jastrow_data, r_up_carts, r_dn_carts):
    """``exp(J)`` with J2 and J3 evaluated in the ``jastrow_eval`` zone dtype.

    J2 is ``_j2_zone_reference``; J3 is ``_compute_Jastrow_three_body_debug`` in
    its default ``jastrow_eval`` zone (AO values in ``ao_eval``, accumulation in
    ``jastrow_eval``, as production). The fp64 J2 transcription plus that J3 is
    asserted to reproduce ``_compute_Jastrow_part_debug`` bit for bit, so in
    full mode it equals that fp64 debug reference.
    """
    assert jastrow_data.jastrow_one_body_data is None and jastrow_data.jastrow_dtn_data is None
    dtype = get_dtype_np("jastrow_eval")
    j2_data, j3_data = jastrow_data.jastrow_two_body_data, jastrow_data.jastrow_three_body_data
    j2 = _j2_zone_reference(j2_data, r_up_carts, r_dn_carts, dtype) if j2_data is not None else dtype(0.0)
    j3 = _compute_Jastrow_three_body_debug(j3_data, r_up_carts, r_dn_carts) if j3_data is not None else dtype(0.0)
    j2_64 = _j2_zone_reference(j2_data, r_up_carts, r_dn_carts, np.float64) if j2_data is not None else 0.0
    debug = _compute_Jastrow_part_debug(jastrow_data, r_up_carts, r_dn_carts)
    assert np.isfinite(debug)
    np.testing.assert_array_equal(0.0 + (0.0 + j2_64) + (0.0 + (j3 if j3_data is not None else 0.0)), debug)
    return np.exp(dtype(j2) + dtype(j3))


def _wrapper_mismatches(trexio_file):
    """Messages for every seed where a public wrapper differs from its NumPy reference (empty = pass)."""
    structure, wf = _wavefunction(trexio_file)
    failures = []
    for seed in (0, 1, 2):
        ru, rd = _configurations(structure, wf.geminal_data, seed)
        jas = complex(np.asarray(wf_module.evaluate_jastrow(wf, jnp.asarray(ru), jnp.asarray(rd))))
        det = complex(np.asarray(wf_module.evaluate_determinant(wf, jnp.asarray(ru), jnp.asarray(rd))))
        psi = complex(np.asarray(evaluate_wavefunction(wf, jnp.asarray(ru), jnp.asarray(rd))))
        jas_ref = complex(_jastrow_zone_reference(wf.jastrow_data, ru, rd))
        det_ref = complex(_compute_det_geminal_all_elements_debug(wf.geminal_data, ru, rd))
        values = (jas, det, psi, jas_ref, det_ref)
        assert all(np.isfinite(v.real) and np.isfinite(v.imag) for v in values), values
        assert abs(det_ref) > 0.0 and abs(jas_ref - 1.0) > 0.0
        checks = (
            ("evaluate_jastrow vs exp(Jastrow zone-dtype reference)", jas, jas_ref, ("jastrow_eval",)),
            ("evaluate_determinant vs determinant debug", det, det_ref, ("ao_eval", "mo_eval", "det_eval")),
            ("evaluate_jastrow * evaluate_determinant vs evaluate_wavefunction", jas * det, psi, ("wf_eval",)),
        )
        for label, got, want, zones in checks:
            atol, rtol = get_tolerance_min(zones, "strict")
            if not np.isclose(got, want, atol=atol * max(1.0, abs(want)), rtol=rtol, equal_nan=False):
                failures.append(f"{trexio_file} seed={seed}: public wrapper mismatch ({label}): {got!r} vs {want!r}")
    return failures


@pytest.mark.parametrize("trexio_file", _WF_FIXTURES)
def test_evaluate_jastrow_and_determinant_match_numpy_references(trexio_file):
    """``evaluate_jastrow`` and ``evaluate_determinant`` equal their NumPy references and factor ``evaluate_wavefunction``."""
    failures = _wrapper_mismatches(trexio_file)
    assert failures == [], "; ".join(failures)
    jax.clear_caches()


def test_evaluate_determinant_oracle_rejects_a_scaled_determinant(monkeypatch):
    """Scaling the determinant returned by ``evaluate_determinant`` must fail the determinant check."""
    atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "det_eval"), "strict")
    factor = 1.0 + 10.0 * (rtol + atol)
    original = wf_module.evaluate_determinant

    def scaled(wavefunction_data, r_up_carts, r_dn_carts):
        return factor * original(wavefunction_data, r_up_carts, r_dn_carts)

    monkeypatch.setattr(wf_module, "evaluate_determinant", scaled)
    failures = _wrapper_mismatches("H2_ae_ccpvdz_cart.h5")
    monkeypatch.undo()
    jax.clear_caches()
    det_failures = [f for f in failures if "public wrapper mismatch (evaluate_determinant vs determinant debug)" in f]
    assert len(det_failures) == 3, failures
