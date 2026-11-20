"""Multi-twist (2x2x2 shifted-MP grid) complex composite pgto(+)PW fixture tests.

Fixture: the committed espresso output ``tests/qe_example_files/Li_k_twist/out/li_twist.save``
(bcc Li, ccECP, nosym+noinv UNFOLDED shifted 2x2x2 Monkhorst-Pack grid -> 8 k-points at
(+-1/4, +-1/4, +-1/4), Gamma excluded -- the layout the twist-average driver iterates over,
matching the pyscf ``lithium_twist_222.h5`` fixture). Per the fixture policy only the QE
output is committed; every Hamiltonian here is built AT TEST TIME through the same
projection pipeline as ``_grad_fidelity_harness._qe_composite_pieces`` (exponent-windowed
pGTO (+) 0.25*g_max plane waves) and discarded.

Coverage:
  * every one of the 8 twists loads, projects, and builds a COMPLEX composite
    (``AOs_cart_pgto_complex_data`` (+) ``AOs_plane_wave_complex_data``) Hamiltonian -- the
    complex-twist composite fixture beyond the single-k Li_k025;
  * the analytic derivative backend serves the twist-grid composite for every analytic
    axis (c on the geminal + J1/J2 blocks, r, R, eps for BOTH ln_Psi and e_L),
    matching the ``_auto`` autograd twin at zone-derived tolerance on representative twists
    from opposite corners of the grid. The geminal lambda-basis c-axis blocks
    are compared and required to be live; this fixture has no J3.
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

import os
from functools import lru_cache

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jqmc import derivatives
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from tests._test_data_cache import qe_example_files_root
from tests._precision_fidelity import mixed_vs_full
from tests.test_jqmc_kernel._orbital_mixed_fidelity import assert_mixed_parameter_derivative_fidelity
from jqmc.hamiltonians import compute_local_energy
from jqmc.pw_pgto_projection import project_pw_to_mixed
from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction, build_wavefunction_from_projection
from jqmc.qe_wrapper import load_qe_reference
from jqmc.upf_wrapper import UPFExponentInit, parse_gaussinfo
from jqmc.wavefunction import evaluate_wavefunction

from ._grad_fidelity_harness import _QE_PGTO_ALPHA_WINDOW, _QE_PW_G_CUT_FRACTION, _window_shells

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.with_chdir]

_TWIST_SAVE = os.path.join(qe_example_files_root(), "Li_k_twist", "out", "li_twist.save")
_UPF = os.path.join(_TWIST_SAVE, "Li.ccECP.upf")
_N_TWISTS = 8
_TIME_REVERSAL_PAIRS = ((1, 8), (2, 7), (3, 6), (4, 5))

# Zone chain of the compared quantities (AO eval -> Jastrow -> det -> Coulomb/ECP -> kinetic).
_ZONES = ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic")


_AXIS_CASES = [
    # lambda-basis radial blocks included: the complex-twist COMPOSITE basis-opt is
    # analytic as of task #33/#34 (cart-pgto-complex child registered; PW child
    # contributes a zero basis grad).
    (
        {
            "opt_geminal": True,
            "opt_J1_param": True,
            "opt_J2_param": True,
            "opt_lambda_basis_exp": True,
            "opt_lambda_basis_coeff": True,
        },
        "c",
    ),
    ({"opt_r": True}, "r"),
    ({"opt_R": True}, "R"),
    ({"opt_eps": True}, "eps"),
]


@lru_cache(maxsize=_N_TWISTS)
def _twist_ham(ik: int):
    """Composite pgto(+)PW Hamiltonian for twist ``ik`` (1-based), built from the committed
    save exactly like ``_grad_fidelity_harness._qe_composite_pieces`` (windowed pGTO keeps
    the composite overlap well-conditioned) with an ACTIVE J1+J2 Jastrow so the Jastrow
    c-axis blocks and the Jastrow channels of the position axes are exercised."""
    ref = load_qe_reference(_TWIST_SAVE, ik=ik)
    g_max = float(np.sqrt(2.0 * ref.ecutwfc))
    strat = UPFExponentInit({"li": _UPF}, g_target=g_max)
    shells = _window_shells([strat.shells_for_element(lbl) for lbl in ref.atom_labels], *_QE_PGTO_ALPHA_WINDOW)
    proj = project_pw_to_mixed(ref, shells, g_cut=_QE_PW_G_CUT_FRACTION * g_max, drop_diffuse=False)
    structure_data, aos_data, geminal_mo_data = build_wavefunction_from_projection(ref, proj, kind="pgto+pw")
    return build_hamiltonian_from_wavefunction(
        structure_data,
        aos_data,
        geminal_mo_data,
        j1_param=1.0,
        j2_param=0.5,
        ecp_by_element={"Li": parse_gaussinfo(_UPF)},
    )


def _draw(ham, seed: int):
    gd = ham.wavefunction_data.geminal_data
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=1.0, size=(int(gd.num_electron_up), 3)))
    r_dn = jnp.asarray(rng.normal(scale=1.0, size=(int(gd.num_electron_dn), 3)))
    return r_up, r_dn


def _assert_param_blocks_allclose(ham, a, b, atol, rtol):
    """Compare the c-grad pytrees on the PARAMETER blocks only (``collect_param_grads``,
    the same reduction production SR/LM consumes). Raw tree-leaf comparison is WRONG here:
    ``_auto``'s ``jax.grad`` also deposits cotangents on non-parameter leaves embedded in
    the wavefunction pytree (e.g. the atom positions inside ``jastrow_one_body_data``'s
    structure) which the analytic path leaves at zero by convention."""
    expected_tolerance = get_tolerance_min(_ZONES, "strict")
    assert (atol, rtol) == expected_tolerance
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    pa = ham.wavefunction_data.collect_param_grads(a)
    pb = ham.wavefunction_data.collect_param_grads(b)
    assert set(pa) == set(pb)
    assert pa, "collect_param_grads returned no parameter blocks"
    basis_blocks = {"lambda_basis_exp", "lambda_basis_coeff"}
    assert basis_blocks <= set(pa), "twist composite lambda basis blocks missing"
    for name in sorted(pa):
        actual, expected = np.asarray(pa[name]), np.asarray(pb[name])
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), name
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=name)
        if name in basis_blocks:
            assert np.max(np.abs(actual)) > atol, f"inactive analytical twist composite {name}"
            assert np.max(np.abs(expected)) > atol, f"inactive autodiff twist composite {name}"


def _twist_fidelity_case(ik):
    # The original cache is keyed only by twist, not arithmetic policy.
    # Projection/model construction must run again after EACH policy switch.
    _twist_ham.cache_clear()
    ham = _twist_ham(ik)
    r_up, r_dn = _draw(ham, seed=ik)
    return ham, r_up, r_dn


def _twist_analytical_axis(case, quantity, flags, axis):
    ham, r_up, r_dn = case
    if quantity == "ln_Psi":
        raw = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)[axis]
    else:
        raw = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=True)[axis]
    if axis == "c":
        return {name: np.asarray(value) for name, value in ham.wavefunction_data.collect_param_grads(raw).items()}
    return raw


def _all_twist_analytical_axes(case, quantity):
    return {axis: _twist_analytical_axis(case, quantity, flags, axis) for flags, axis in _AXIS_CASES}


def _assert_param_blocks_fidelity(actual, reference):
    assert set(actual) == set(reference) and actual, "missing mixed/full parameter blocks"
    basis_blocks = {"lambda_basis_exp", "lambda_basis_coeff"}
    assert basis_blocks <= set(actual), "twist composite lambda basis blocks missing"
    atol, _ = get_tolerance_min(_ZONES, "strict")
    for name in sorted(actual):
        assert_mixed_parameter_derivative_fidelity(actual[name], reference[name], f"twist composite {name} mixed vs full")
        if name in basis_blocks:
            assert np.max(np.abs(actual[name])) > atol, f"inactive analytical twist composite {name}"
            assert np.max(np.abs(reference[name])) > atol, f"inactive full twist composite {name}"


@pytest.mark.light
def test_twist_grid_all_8_composite_build():
    """Every twist of the unfolded 2x2x2 grid loads and builds the COMPLEX composite ham:
    cart-pgto-complex (+) plane-wave-complex children, 2 valence electrons, ECP attached."""
    for ik in range(1, _N_TWISTS + 1):
        ham = _twist_ham(ik)
        aos = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
        assert type(aos).__name__ == "AOs_composite_data"
        children = {type(c).__name__ for c in aos.children}
        assert children == {"AOs_cart_pgto_complex_data", "AOs_plane_wave_complex_data"}, children
        gd = ham.wavefunction_data.geminal_data
        assert int(gd.num_electron_up) + int(gd.num_electron_dn) == 2


@pytest.mark.light
@pytest.mark.parametrize("plus_ik,minus_ik", _TIME_REVERSAL_PAIRS)
def test_twist_grid_wavefunction_and_local_energy_are_time_reversal_pairs(plus_ik: int, minus_ik: int):
    """At fixed coordinates, ``Psi(-k)=conj(Psi(+k))`` and ``e_L(-k)=conj(e_L(+k))``.

    The four pairs exhaust the committed shifted 2x2x2 grid.  Both Hamiltonians
    are independently built from their QE wavefunction files, so this checks the
    full pGTO+PW projection, J1+J2, kinetic and ECP/Coulomb production chain; it
    is not the existing within-one-geminal spin-pairing identity.
    """
    plus_ham = _twist_ham(plus_ik)
    minus_ham = _twist_ham(minus_ik)
    r_up, r_dn = _draw(plus_ham, seed=2026)

    plus_k = np.asarray(plus_ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data.children[0].k_point)
    minus_k = np.asarray(minus_ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data.children[0].k_point)
    np.testing.assert_array_equal(minus_k, -plus_k)

    psi_plus = np.asarray(evaluate_wavefunction(plus_ham.wavefunction_data, r_up, r_dn))
    psi_minus = np.asarray(evaluate_wavefunction(minus_ham.wavefunction_data, r_up, r_dn))
    e_plus = np.asarray(compute_local_energy(plus_ham, r_up, r_dn, jnp.eye(3)))
    e_minus = np.asarray(compute_local_energy(minus_ham, r_up, r_dn, jnp.eye(3)))

    atol, rtol = get_tolerance_min(_ZONES, "strict")
    np.testing.assert_allclose(psi_minus, np.conj(psi_plus), atol=atol, rtol=rtol)
    np.testing.assert_allclose(e_minus, np.conj(e_plus), atol=atol, rtol=rtol)


# Two representative twists from opposite corners of the shifted grid (QE materialises the
# unfolded points as wfc1..8; 1 and 8 differ in the sign pattern of (+-1/4,+-1/4,+-1/4)).
@pytest.mark.light
@pytest.mark.parametrize("ik", [1, 8])
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_twist_grid_composite_grad_vs_auto(quantity, ik):
    """Analytic vs ``_auto`` on the twist-grid composite for every analytic axis:
    c (geminal + J1/J2 parameter blocks), r, R, eps. The analytic path is called directly
    for each axis and must match the ``_auto`` twin (a regression fails here loudly).
    Full keeps the same strict AD comparisons. Mixed rebuilds the production
    route independently under full/mixed policies and uses central block-fidelity
    budgets; both lambda radial blocks keep their finite/nonzero liveness floor."""
    if is_mixed_precision_enabled():
        mixed, full = mixed_vs_full(lambda: _twist_fidelity_case(ik), lambda case: _all_twist_analytical_axes(case, quantity))
        for axis in mixed:
            if axis == "c":
                _assert_param_blocks_fidelity(mixed[axis], full[axis])
            else:
                assert_mixed_parameter_derivative_fidelity(
                    mixed[axis], full[axis], f"twist composite {quantity} {axis} mixed vs full"
                )
        jax.clear_caches()
        return
    ham = _twist_ham(ik)
    r_up, r_dn = _draw(ham, seed=ik)
    RT = jnp.eye(3)
    atol, rtol = get_tolerance_min(_ZONES, "strict")

    axis_cases = _AXIS_CASES
    for flags, axis in axis_cases:
        if quantity == "ln_Psi":
            a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)[axis]
            b = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=False)[axis]
        else:
            a = derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, flags, analytic=True)[axis]
            b = derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, flags, analytic=False)[axis]
        if axis == "c":
            _assert_param_blocks_allclose(ham, a, b, atol, rtol)
        else:
            for xa, xb in zip(
                np.atleast_1d(a) if not isinstance(a, tuple) else a, np.atleast_1d(b) if not isinstance(b, tuple) else b
            ):
                np.testing.assert_allclose(np.asarray(xa), np.asarray(xb), atol=atol, rtol=rtol)
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize("dead_block", ["lambda_basis_exp", "lambda_basis_coeff"])
def test_twist_composite_basis_liveness_rejects_both_sides_zero(dead_block):
    """A synthetic comparator control rejects a dead lambda block even when both routes return zero.

    This tests the exact comparator used by the real Li twist-grid gradient
    rows; synthetic arrays are not a substitute for their physical evidence.
    """
    from types import SimpleNamespace

    fake_ham = SimpleNamespace(wavefunction_data=SimpleNamespace(collect_param_grads=lambda grad: grad))
    blocks = {"lambda_basis_exp": np.ones(3), "lambda_basis_coeff": np.ones(3), "j1_param": np.ones(1)}
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    if is_mixed_precision_enabled():
        _assert_param_blocks_fidelity(blocks, blocks)
    else:
        _assert_param_blocks_allclose(fake_ham, blocks, blocks, atol, rtol)
    faulty = dict(blocks, **{dead_block: np.zeros(3)})
    with pytest.raises(AssertionError, match=f"inactive analytical twist composite {dead_block}"):
        if is_mixed_precision_enabled():
            _assert_param_blocks_fidelity(faulty, faulty)
        else:
            _assert_param_blocks_allclose(fake_ham, faulty, faulty, atol, rtol)


@pytest.mark.light
@pytest.mark.parametrize("ik", [1, 8])
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_twist_grid_composite_basis_is_live(quantity, ik):
    """Both lambda radial blocks stay live; full has AD and mixed has full-route fidelity.

    This focused c-axis witness uses exactly the existing all-axis row's
    committed Li_k_twist source, J1/J2, seed=ik positions and joint MO/basis
    flags. The original all-axis evidence stays registered in its own guarantee;
    this sibling supplies distinct evidence for the additional liveness contract.
    """
    if is_mixed_precision_enabled():
        flags, axis = _AXIS_CASES[0]
        mixed, full = mixed_vs_full(
            lambda: _twist_fidelity_case(ik),
            lambda case: _twist_analytical_axis(case, quantity, flags, axis),
        )
        _assert_param_blocks_fidelity(mixed, full)
        jax.clear_caches()
        return
    ham = _twist_ham(ik)
    r_up, r_dn = _draw(ham, seed=ik)
    flags = {
        "opt_geminal": True,
        "opt_J1_param": True,
        "opt_J2_param": True,
        "opt_lambda_basis_exp": True,
        "opt_lambda_basis_coeff": True,
    }
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    if quantity == "ln_Psi":
        actual = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)["c"]
        reference = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=False)["c"]
    else:
        actual = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=True)["c"]
        reference = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=False)["c"]
    _assert_param_blocks_allclose(ham, actual, reference, atol, rtol)
    jax.clear_caches()
