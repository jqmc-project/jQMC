"""Issue #127: the native MCMC sweep and inverse refresh after a huge determinant ratio.

* ``sm_apply_inplace`` (the accepted-move rank-1 update of the native sweep)
  evaluated the moved electron's new inverse column (row for a down move) as
  ``X e_k (1 - (R - 1) / R)``: for ``|R| > 1/eps`` it is exactly zero, the next
  proposal sees ``R = 1`` and a move with exact acceptance ~1e-28 (power model)
  or 0.16 at draw > 0.25 (water) is accepted.
* The native energy chunk's per-step inverse refresh inverted the carried geminal by
  Gauss-Jordan elimination, which returns ``inf``/``NaN`` for a singular carried
  geminal (and then NaN local energies), whereas the Python inverse refresh takes the
  equilibrated SVD pseudo-inverse (finite).

References are independent of the production ratio / inverse-update / AS /
proposal helpers: rebuilt-G ``slogdet`` or the closed-form orbital for the
decisions, the rebuilt G for the carried G, ``numpy.linalg.inv`` of the carried G
for the carried inverse, and ``numpy.linalg.pinv`` (plus the Python refresh) for
the refreshed inverse of a singular geminal.
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

import jax.numpy as jnp
import jqmc_rs
import numpy as np
import pytest

from jqmc import jqmc_mcmc as mcmc
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled, precision_policy_snapshot
from tests import _sm_large_ratio_cases as cases

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_G_ZONES = ("ao_eval", "det_eval")
_INVERSE_ZONES = ("det_eval", "det_ratio")
_KEY0 = 12345
# A bounded ``while`` (not ``for ... in range(N)``): the GDD case-binding extractor
# statically unrolls literal ``range`` loops once per value per case.
_KEY_SEARCH_LIMIT = 200000


def _ctx(ham, complex_=False):
    wf = ham.wavefunction_data
    return jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        complex_,
        complex_,
        False,
        want_force=False,
        want_gfmc=False,
        precision_policy=precision_policy_snapshot(),
    )


def _key(spin, n_up, n_dn):
    """First threefry key whose two moves both select electron 0 of ``spin`` along x, second draw with margin."""
    counter = 0
    while counter < _KEY_SEARCH_LIMIT:
        first = jqmc_rs._move_draws(_KEY0, counter, n_up, n_dn)
        second = jqmc_rs._move_draws(first[4], first[5], n_up, n_dn)
        if all(d[0] == (spin == "up") and d[1] == 0 and d[2] == 0 for d in (first, second)):
            if second[3] > cases.NATIVE_MIN_SECOND_DRAW:
                return (_KEY0, counter), (first[3], second[3])
        counter += 1
    raise AssertionError("no threefry key selects the requested moves")


def _case(model, spin):
    if model == "water":
        return cases.water_case(spin)
    return cases.power_case()


@pytest.mark.parametrize(
    "model,spin",
    [("water", "up"), ("water", "down"), ("power", "up")],
)
def test_native_large_ratio_move_keeps_exact_decisions_and_carried_inverse(model, spin):
    """The production native sweep (ObsCtx.mcmc_sweep), first move |R| > 2**48: exact decisions and state.

    Same inputs as the XLA twin (water 28 -> 12 -> 13 bohr, up and down; x^12 model 0.01 -> 0.3 -> 0.02).
    The threefry key is the first one whose two moves select electron 0 of the moved spin along x with a
    second uniform draw above 0.25. A one-move sweep must accept (exact acceptance 1) and hand back the
    rebuilt G (rows scaled; ao_eval/det_eval strict) and the dense inverse of that carried G (columns
    scaled; det_eval/det_ratio strict). The two-move sweep must accept exactly the moves the exact
    acceptance accepts (move 2 rejected) and hand back the same exact state.
    """
    ham, nuclei, charges, configurations = _case(model, spin)
    geminal = ham.wavefunction_data.geminal_data
    ctx = _ctx(ham)
    (k0, k1), draws = _key(spin, geminal.num_electron_up, geminal.num_electron_dn)
    held = configurations[0]
    normals, expected_accepts, helds = [], [], []
    for label, target, draw in (("move-1", configurations[1], draws[0]), ("move-2", configurations[2], draws[1])):
        normal = cases.injected_normal(held, target, spin, nuclei, charges)
        proposal = cases.proposed(held, normal, spin, nuclei, charges)
        if label == "move-1":
            assert cases.log_abs_ratio(ham, held, proposal, model) > np.log(cases.LARGE_RATIO_WITNESS), "precondition"
        probability = cases.exact_acceptance(ham, held, proposal, spin, nuclei, charges, model)
        expected_accepts.append(int(draw < probability))
        held = proposal if expected_accepts[-1] else held
        normals.append(normal)
        helds.append(held)
    assert expected_accepts == [1, 0], "precondition: move 1 accepted, move 2 rejected by the exact acceptance"

    matrix0 = cases.reference_geminal(ham, configurations[0], model)
    g_atol, g_rtol = get_tolerance_min(_G_ZONES, "strict")
    i_atol, i_rtol = get_tolerance_min(_INVERSE_ZONES, "strict")
    for count in (1, 2):
        out = ctx.mcmc_sweep(
            nuclei,
            charges,
            configurations[0][0],
            configurations[0][1],
            matrix0,
            np.linalg.inv(matrix0),
            cases.DT,
            0.0,
            k0,
            k1,
            np.asarray(normals[:count]),
            None,
            0,
        )
        label = f"{count}-move sweep"
        assert int(out["accepted"]) == sum(expected_accepts[:count]), label
        expected = helds[count - 1]
        actual_positions = np.concatenate([np.asarray(out["r_up"]), np.asarray(out["r_dn"])])
        np.testing.assert_allclose(actual_positions, np.concatenate(expected), atol=g_atol, rtol=g_rtol, err_msg=label)
        actual_matrix = np.asarray(out["geminal"])
        reference_matrix = cases.reference_geminal(ham, expected, model)
        actual_scaled, reference_scaled = cases.row_scaled(actual_matrix, reference_matrix)
        np.testing.assert_allclose(actual_scaled, reference_scaled, atol=g_atol, rtol=g_rtol, err_msg=label)
        actual_inverse = np.asarray(out["geminal_inv"])
        dense_inverse = np.linalg.inv(actual_matrix)
        actual_scaled, reference_scaled = cases.column_scaled(actual_inverse, dense_inverse)
        np.testing.assert_allclose(actual_scaled, reference_scaled, atol=i_atol, rtol=i_rtol, err_msg=label)


def test_native_zero_geminal_refresh_matches_python_equilibrated_fallback():
    """The real and complex native inverse refreshes stay finite when the whole geminal underflows to zero.

    This calls the production refresh functions used by the native energy chunks. A nonzero rank-one
    test would make the equilibrated generalized inverse backend-SVD-dependent; the production failure addressed
    here is the exact-zero tail state. Starting from a zero carried inverse makes the refresh residual trigger
    deterministically without constructing an out-of-range AO basis.
    """
    from jqmc._setting import SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD

    real = np.zeros((2, 2))
    atol, rtol = get_tolerance_min(("det_eval",), "strict")
    for matrix, native_refresh in (
        (real, jqmc_rs._refresh_running_inverse_if_needed),
        (real * np.exp(0.7j), jqmc_rs._refresh_running_inverse_if_needed_complex),
    ):
        refreshed, native_inverse = native_refresh(
            matrix,
            np.zeros_like(matrix),
            SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD,
        )
        assert refreshed, "the zero carried inverse must trigger the inverse refresh"
        native_inverse = np.asarray(native_inverse)
        assert np.all(np.isfinite(native_inverse)), native_inverse
        python_inverse, python_count = mcmc._refresh_running_geminal_inv(
            jnp.asarray(matrix[None]),
            jnp.zeros_like(jnp.asarray(matrix[None])),
        )
        assert python_count == 1
        np.testing.assert_allclose(native_inverse, np.asarray(python_inverse[0]), atol=atol, rtol=rtol)
        np.testing.assert_allclose(native_inverse, np.linalg.pinv(matrix), atol=atol, rtol=rtol)


def test_native_geminal_inverse_guard_constants_match_python():
    """The native seed / refresh inverse uses the Python drivers' guard constants.

    The LU guard residual and the equilibration-exponent clip are internal constants shared by the XLA and the
    native routes, so both backends pick the same walkers for the pseudo-inverse fallback.
    """
    from jqmc import _setting

    assert jqmc_rs.geminal_lu_guard_residual_threshold() == _setting.GEMINAL_LU_GUARD_RESIDUAL_THRESHOLD
    assert jqmc_rs.geminal_equilibration_max_exponent() == _setting.GEMINAL_EQUILIBRATION_MAX_EXPONENT


def _graded_geminal(a: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form graded geminal with structural zeros and its exact inverse.

    ``G = [[a, 0, 0], [0, 1, a], [0, 1, -a]]`` grades row 0 and column 2 by ``a``
    simultaneously; ``G^-1 = [[1/a, 0, 0], [0, 1/2, 1/2], [0, 1/(2a), -1/(2a)]]``.
    Every entry of both is a finite float64 for the ``a`` used below.
    """
    matrix = np.array([[a, 0.0, 0.0], [0.0, 1.0, a], [0.0, 1.0, -a]])
    inverse = np.array([[1.0 / a, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.5 / a, -0.5 / a]])
    return matrix, inverse


@pytest.mark.parametrize("a", [1.0e-140, 1.0e-160, 1.0e-200])
@pytest.mark.parametrize("kind", ["real", "complex"])
def test_native_refresh_of_row_and_column_graded_geminal_is_the_exact_inverse(kind, a):
    """The native inverse refresh unscales the pseudo-inverse without overflowing a scale product.

    The equilibrating row scale of row 0 and column scale of column 2 are both ~1/a; for a = 1e-160
    and 1e-200 their product exceeds the float64 range although the exact inverse is finite, and the
    structural zero at ``G^-1[2, 0]`` turned ``inf * 0`` into NaN in both native refreshes
    (``determinant::refresh_running_inverse_if_needed`` and its complex twin, run at the energy-chunk
    refresh boundaries). a = 1e-140 is the in-range control. Starting from a zero carried inverse with the
    production threshold, the refresh must trigger and return the closed-form inverse (det_eval strict,
    each row scaled by its exact max-norm; the structural zeros exactly zero). The complex case multiplies G by the
    phase exp(0.7 i), whose exact inverse is exp(-0.7 i) G^-1.
    """
    from jqmc._setting import SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD

    matrix, exact = _graded_geminal(a)
    if kind == "real":
        refresh = jqmc_rs._refresh_running_inverse_if_needed
    else:
        phase = np.exp(0.7j)
        matrix, exact = matrix * phase, exact / phase
        refresh = jqmc_rs._refresh_running_inverse_if_needed_complex
    assert np.all(np.isfinite(matrix)) and np.all(np.isfinite(exact)), "precondition: finite G and G^-1"
    refreshed, inverse = refresh(matrix, np.zeros_like(matrix), SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD)
    inverse = np.asarray(inverse)
    assert refreshed, "the zero carried inverse must trigger the inverse refresh"
    assert np.all(np.isfinite(inverse)), inverse
    assert np.all(inverse[exact == 0] == 0), inverse
    atol, rtol = get_tolerance_min(("det_eval",), "strict")
    actual_scaled, reference_scaled = cases.row_scaled(inverse, exact)
    np.testing.assert_allclose(actual_scaled, reference_scaled, atol=atol, rtol=rtol)


def _complex_sweep(ctx, ham, start, normals, k0, k1):
    """One production native complex sweep (NativeObsCtx.mcmc_sweep_complex) from the closed-form carried pair."""
    matrix0 = cases.reference_geminal(ham, start, "cpower")
    assert np.all(np.isfinite(matrix0)) and np.all(matrix0 != 0), "precondition: finite nonzero closed-form G"
    return ctx.mcmc_sweep_complex(
        np.asarray(ham.structure_data.positions, dtype=np.float64),
        np.array([float(cases.COMPLEX_Z)]),
        start[0],
        start[1],
        matrix0,
        1.0 / matrix0,
        cases.DT,
        0.0,
        k0,
        k1,
        np.asarray(normals, dtype=np.float64),
        np.asarray(ham.structure_data.cell, dtype=np.float64),
        int(ham.structure_data._n_max),
    )


def _assert_complex_state(out, ham, expected, label):
    """Carried positions, G (closed form, rows scaled) and G^-1 (dense inverse of the carried G, columns scaled)."""
    g_atol, g_rtol = get_tolerance_min(_G_ZONES, "strict")
    i_atol, i_rtol = get_tolerance_min(_INVERSE_ZONES, "strict")
    actual_positions = np.concatenate([np.asarray(out["r_up"]), np.asarray(out["r_dn"])])
    np.testing.assert_allclose(actual_positions, np.concatenate(expected), atol=g_atol, rtol=g_rtol, err_msg=label)
    actual_matrix = np.asarray(out["geminal"])
    reference_matrix = cases.reference_geminal(ham, expected, "cpower")
    actual_scaled, reference_scaled = cases.row_scaled(actual_matrix, reference_matrix)
    np.testing.assert_allclose(actual_scaled, reference_scaled, atol=g_atol, rtol=g_rtol, err_msg=label)
    actual_inverse = np.asarray(out["geminal_inv"])
    assert np.all(np.isfinite(actual_inverse)), (label, actual_inverse)
    dense_inverse = np.linalg.inv(actual_matrix)
    actual_scaled, reference_scaled = cases.column_scaled(actual_inverse, dense_inverse)
    np.testing.assert_allclose(actual_scaled, reference_scaled, atol=i_atol, rtol=i_rtol, err_msg=label)


@pytest.mark.parametrize("spin", ["up", "down"])
def test_native_complex_large_ratio_move_keeps_exact_decisions_and_carried_inverse(spin):
    """The production native COMPLEX sweep (NativeObsCtx.mcmc_sweep_complex) writes the exact moved column / row.

    This case detects the removal of only the exact moved-column (up) / moved-row (dn) overwrite of
    ``complex_det::sm_apply_inplace_complex``. Model ``cpower``: periodic neutral
    He (Z = 2, cubic cell 100 bohr, twist (0.25, 0.125, 0)), no Jastrow, time_reversal=False, one up and one
    down electron in phi = c N12 x^12 (1 + i x) exp(-r^2); the moved electron goes x = 0.01 -> 0.3
    (|R| ~ 5.07e17 > 1/eps, exact acceptance 1) -> 0.02 (exact acceptance ~6.8e-29), the other electron stays
    at (0.4, 0.3, 0.4); Dt = 2, epsilon_AS = 0. The threefry key is the first one whose two moves select the
    moved spin along x with a second uniform draw above 0.25; each normal draw is injected so that the proposal
    lands on the target. The sweep starts from the closed-form carried pair (G, 1/G). A one-move sweep must
    accept and hand back the closed-form G (ao_eval/det_eval strict) and the dense inverse of the carried G
    (det_eval/det_ratio strict); the two-move sweep must reject move 2 exactly as the closed-form acceptance
    does. Without the overwrite the moved column keeps no digit, the carried inverse is wrong and move 2 is
    accepted. The values stay in the float32 range, so both precision modes apply.
    """
    ham, nuclei, charges, configurations = cases.complex_power_case(spin)
    geminal = ham.wavefunction_data.geminal_data
    assert np.iscomplexobj(np.asarray(geminal.orb_data_up_spin.mo_coefficients)), "precondition: complex orbital"
    ctx = _ctx(ham, complex_=True)
    (k0, k1), draws = _key(spin, geminal.num_electron_up, geminal.num_electron_dn)
    held = configurations[0]
    normals, expected_accepts, helds = [], [], []
    for label, target, draw in (("move-1", configurations[1], draws[0]), ("move-2", configurations[2], draws[1])):
        normal = cases.injected_normal(held, target, spin, nuclei, charges)
        proposal = cases.proposed(held, normal, spin, nuclei, charges)
        if label == "move-1":
            assert cases.log_abs_ratio(ham, held, proposal, "cpower") > np.log(cases.LARGE_RATIO_WITNESS), "precondition"
        probability = cases.exact_acceptance(ham, held, proposal, spin, nuclei, charges, "cpower")
        expected_accepts.append(int(draw < probability))
        held = proposal if expected_accepts[-1] else held
        normals.append(normal)
        helds.append(held)
    assert expected_accepts == [1, 0], "precondition: move 1 accepted, move 2 rejected by the exact acceptance"
    for count in (1, 2):
        out = _complex_sweep(ctx, ham, configurations[0], normals[:count], k0, k1)
        label = f"{count}-move complex sweep ({spin})"
        assert int(out["accepted"]) == sum(expected_accepts[:count]), label
        _assert_complex_state(out, ham, helds[count - 1], label)


@pytest.mark.parametrize("x_start", [1.0e-11, 1.0e-14, 1.0e-17])
@pytest.mark.parametrize("spin", ["up", "down"])
def test_native_complex_update_with_ratio_beyond_sqrt_float_max_keeps_finite_exact_inverse(spin, x_start):
    """A finite complex ratio |R| > 1e154 leaves a finite, exact carried inverse (range-safe complex division).

    ``num_complex``'s plain ``a / R`` forms ``|R|^2`` and ``a conj(R)`` unscaled, so with it the rank-1 and
    exact-slice divisions of ``complex_det::sm_apply_inplace_complex`` return NaN once ``|R|`` exceeds
    ~1.3e154 although the quotient is finite. Model ``cpower`` (see the move-pair test): the moved electron
    goes x_start -> 0.3 with the other at (0.4, 0.3, 0.4); |R| = 5.07e125 (x_start = 1e-11,
    in-range control), 5.07e161 (1e-14) and 5.07e197 (1e-17), exact acceptance 1. The sweep starts from the
    closed-form carried pair (G, 1/G) -- a finite float64 det-zone state in both precision modes; the sweep
    evaluates orbitals only at the proposed x = 0.3, inside the float32 range -- and must accept, return the
    closed-form G (ao_eval/det_eval strict) and a finite carried inverse equal to the dense inverse of the
    carried G (det_eval/det_ratio strict).
    """
    ham, nuclei, charges, configurations = cases.complex_power_case(spin, (x_start, cases.COMPLEX_X[1]))
    geminal = ham.wavefunction_data.geminal_data
    ctx = _ctx(ham, complex_=True)
    (k0, k1), _draws = _key(spin, geminal.num_electron_up, geminal.num_electron_dn)
    start, target = configurations
    normal = cases.injected_normal(start, target, spin, nuclei, charges)
    proposal = cases.proposed(start, normal, spin, nuclei, charges)
    log_ratio = cases.log_abs_ratio(ham, start, proposal, "cpower")
    assert log_ratio > np.log(cases.LARGE_RATIO_WITNESS), "precondition: large ratio"
    assert cases.exact_acceptance(ham, start, proposal, spin, nuclei, charges, "cpower") == 1.0, "precondition"
    out = _complex_sweep(ctx, ham, start, [normal], k0, k1)
    label = f"complex sweep {spin} x {x_start} -> 0.3 (log10|R| = {log_ratio / np.log(10):.1f})"
    assert int(out["accepted"]) == 1, label
    _assert_complex_state(out, ham, proposal, label)


@pytest.mark.parametrize("x_start", [1.0e-11, 1.0e-14, 1.0e-17])
@pytest.mark.parametrize("spin", ["up", "down"])
def test_native_complex_2x2_update_beyond_sqrt_float_max_keeps_whole_exact_inverse(spin, x_start):
    """The range-safe RANK-1 division keeps the non-moved inverse column / row finite and exact (n = 2).

    Reverting only the rank-1 division ``ainv_u[i] / R`` of ``complex_det::sm_apply_inplace_complex`` to the
    plain complex ``/`` is visible only with n >= 2: a 1x1 inverse, which the exact moved column / row
    overwrites whole, hides it. Model ``cpair``:
    periodic neutral Be (Z = 4, cubic cell 100 bohr, twist (0.25, 0.125, 0)), no Jastrow, time_reversal=False,
    two up and two down electrons in phi0 = c N12 x^12 (1 + i x) exp(-r^2) and phi1 = N13 x^13 exp(-r^2). The
    moved electron 0 (up and down) goes x_start -> 0.3 at y = 0.3, z = 0.4, the other electron of its spin stays
    at (0.4, 0.3, 0.4) and the opposite spin at (0.5, 0.2, 0.1), (0.7, 0.1, 0.3); |R| ~ 1e125 (x_start = 1e-11,
    in-range control), 1e161 (1e-14) and 1e197 (1e-17), exact acceptance 1, Dt = 2, epsilon_AS = 0. The sweep
    starts from the closed-form carried pair (G = Phi_up^T Phi_dn and its adjugate inverse). In full precision
    it must accept, return the closed-form G (rows scaled; ao_eval/det_eval strict) and a carried inverse that is
    finite everywhere and equals the closed-form adjugate inverse of the target G entry by entry, the non-moved
    column (up) / row (down) included (columns scaled; det_eval/det_ratio strict). With the plain division the
    rank-1 factor of an up move is NaN for |R| > ~1.3e154 (|ainv_u| |R| overflows), and that of a down move is
    a spurious 0 for the non-moved rows (|R|^2 overflows alone); only the moved column / row is rewritten, so the
    non-moved one stays NaN (up) or wrong (down). The correctness comparison is full precision only; the mixed
    lane runs the same sweep (orbitals at x = 0.3 in the float32 ao_eval zone, the update float64) and
    requires the same acceptance and an entirely finite carried inverse, with no tolerance comparison.
    """
    ham, nuclei, charges, configurations = cases.complex_pair_case(spin, x_start)
    geminal = ham.wavefunction_data.geminal_data
    assert (geminal.num_electron_up, geminal.num_electron_dn) == (2, 2), "precondition: 2x2 geminal"
    assert np.iscomplexobj(np.asarray(geminal.orb_data_up_spin.mo_coefficients)), "precondition: complex orbitals"
    ctx = _ctx(ham, complex_=True)
    (k0, k1), _draws = _key(spin, geminal.num_electron_up, geminal.num_electron_dn)
    start, target = configurations
    normal = cases.injected_normal(start, target, spin, nuclei, charges)
    proposal = cases.proposed(start, normal, spin, nuclei, charges)
    log_ratio = cases.log_abs_ratio(ham, start, proposal, "cpair")
    assert log_ratio > np.log(cases.LARGE_RATIO_WITNESS), "precondition: large ratio"
    assert cases.exact_acceptance(ham, start, proposal, spin, nuclei, charges, "cpair") == 1.0, "precondition"
    matrix0, inverse0 = cases.complex_pair_geminal(start), cases.complex_pair_inverse(start)
    assert np.all(np.isfinite(matrix0)) and np.all(np.isfinite(inverse0)), "precondition: finite carried pair"
    out = ctx.mcmc_sweep_complex(
        np.asarray(ham.structure_data.positions, dtype=np.float64),
        np.array([float(cases.PAIR_Z)]),
        start[0],
        start[1],
        matrix0,
        inverse0,
        cases.DT,
        0.0,
        k0,
        k1,
        np.asarray([normal], dtype=np.float64),
        np.asarray(ham.structure_data.cell, dtype=np.float64),
        int(ham.structure_data._n_max),
    )
    label = f"2x2 complex sweep {spin} x {x_start} -> 0.3 (log10|R| = {log_ratio / np.log(10):.1f})"
    assert int(out["accepted"]) == 1, label
    actual_inverse = np.asarray(out["geminal_inv"])
    assert np.all(np.isfinite(actual_inverse)), (label, actual_inverse)
    if is_mixed_precision_enabled():
        return
    g_atol, g_rtol = get_tolerance_min(_G_ZONES, "strict")
    i_atol, i_rtol = get_tolerance_min(_INVERSE_ZONES, "strict")
    actual_positions = np.concatenate([np.asarray(out["r_up"]), np.asarray(out["r_dn"])])
    np.testing.assert_allclose(actual_positions, np.concatenate(proposal), atol=g_atol, rtol=g_rtol, err_msg=label)
    actual_matrix = np.asarray(out["geminal"])
    reference_matrix = cases.complex_pair_geminal(proposal)
    actual_scaled, reference_scaled = cases.row_scaled(actual_matrix, reference_matrix)
    np.testing.assert_allclose(actual_scaled, reference_scaled, atol=g_atol, rtol=g_rtol, err_msg=label)
    reference_inverse = cases.complex_pair_inverse(proposal)
    actual_scaled, reference_scaled = cases.column_scaled(actual_inverse, reference_inverse)
    np.testing.assert_allclose(actual_scaled, reference_scaled, atol=i_atol, rtol=i_rtol, err_msg=label)
