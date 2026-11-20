"""Independent complex VMC ECP and fixed-phase per-candidate LRDMC definitions."""

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

import ast
import inspect
import textwrap

import jax
import numpy as np
import pytest

from jqmc import coulomb_potential as coulomb
from jqmc import jqmc_gfmc
from jqmc._precision import get_tolerance_min
from jqmc.determinant import compute_geminal_all_elements
from jqmc.hamiltonians import compute_local_energy, compute_local_energy_fast
from tests.test_jqmc_kernel._complex_ecp_reference import complex_local_energy_reference, ecp_terms
from tests.test_jqmc_kernel.test_gfmc_forward_production_binding import _production_n
from tests.test_jqmc_kernel.test_periodic_invariants import _COND_MAX, _pattern_ham, _draw_well_conditioned_cfg, _ginv

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]
_ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "mo_eval",
    "jastrow_eval",
    "jastrow_grad_lap",
    "det_eval",
    "wf_kinetic",
    "coulomb",
    "ewald",
    "local_energy",
)


def _case():
    ham = _pattern_ham("li_twist_jdtn")
    up, down = _draw_well_conditioned_cfg(ham, seed=7)
    G = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, up, down))
    cond_G = np.linalg.cond(G)
    assert cond_G < _COND_MAX
    return ham, up, down, np.eye(3)


def _compare(actual, reference):
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)


def _activated(terms):
    _local, _up, _down, det_terms, jratio, full_terms = terms
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    assert np.max(abs(np.imag(full_terms))) > atol + rtol * np.max(abs(np.real(full_terms)))
    for values in (det_terms, full_terms):
        assert np.any(np.real(values) < -atol) and np.any(np.real(values) > atol)
    assert np.all(np.real(jratio) > 0) and np.all(np.imag(jratio) == 0)


@pytest.mark.parametrize("route", ["standard", "fast"])
def test_vmc_complex_ecp_matches_independent_absolute_energy(route):
    """Li twist DTN VMC preserves the independent complex H Psi/Psi, including Im(Vnl).

    lithium_k025.h5 with the existing li_twist_jdtn recipe, one up/one down,
    one PCG64(700000+attempt) image-offset configuration selected by cond(G)<1e4,
    seed7 and at most300 attempts; fixed identity ECP rotation, nearest nucleus,
    twelve-point quadrature. Independent primitive spherical Gaussians, image
    Bloch phases, MO determinant, DTN/Jastrow polynomial, complex Hessian kinetic,
    NumPy Ewald and explicit radial/Legendre/Psi-ratio sum supply the reference.
    No production numerical ECP/AO/Jastrow/energy helper supplies that reference.
    The standard/fresh-inverse and fast/exact-inverse consumers each use their
    own precision mode and existing strict zones; no FD or stochastic average.
    """
    ham, up, down, RT = _case()
    terms = ecp_terms(ham, up, down, RT)
    _activated(terms)
    expected = complex_local_energy_reference(ham, up, down, RT)
    actual = (
        compute_local_energy(ham, up, down, RT)
        if route == "standard"
        else compute_local_energy_fast(ham, up, down, RT, _ginv(ham, up, down))
    )
    _compare(actual, expected)


@pytest.mark.parametrize("route", ["standard", "fast"])
@pytest.mark.parametrize("determinant_only", [False, True])
def test_fixed_phase_ecp_candidates_match_independent_real_projection(route, determinant_only):
    """Every Li twist ECP candidate is Re of its independent complex ratio before sign splitting.

    The unchanged Li DTN fixture, fixed seed7 image-offset positions and identity
    twelve-point mesh have nonzero imaginary terms and both real signs. Explicit
    radial/Legendre weights and independent primitive-AO determinant/Jastrow
    ratios determine every proposed coordinate and contribution. Determinant-
    only and full-Psi paths retain the tmove/dltmove distinction. Both modes
    use their existing strict zone budget, not a cross-precision comparison.
    """
    ham, up, down, RT = _case()
    terms = ecp_terms(ham, up, down, RT)
    _activated(terms)
    _local, expected_up, expected_down, det_terms, _jratio, full_terms = terms
    cp, wf = ham.coulomb_potential_data, ham.wavefunction_data
    if route == "standard":
        actual = coulomb.compute_ecp_non_local_parts_nearest_neighbors(
            cp, wf, up, down, RT, flag_determinant_only=determinant_only
        )
    else:
        actual = coulomb.compute_ecp_non_local_parts_nearest_neighbors_fast_update(
            cp, wf, up, down, RT, _ginv(ham, up, down), flag_determinant_only=determinant_only
        )
    expected = np.real(det_terms if determinant_only else full_terms)
    assert not np.iscomplexobj(np.asarray(actual[2]))
    _compare(actual[0], expected_up)
    _compare(actual[1], expected_down)
    _compare(actual[2], expected)
    _compare(actual[3], np.sum(expected))


@pytest.mark.parametrize("algorithm", ["conventional", "load-balanced"])
@pytest.mark.parametrize("move", ["tmove", "dltmove"])
def test_lrdmc_ecp_sign_split_matches_independent_candidate_definition(algorithm, move):
    """LRDMC ECP diagonal/off-diagonal increments obey per-candidate real sign splitting.

    On the same fixed Li twist DTN input, compare the actual production t/n
    Hamiltonian closure with a copy whose nonlocal coefficients alone are zero.
    Shared kinetic, local ECP and bare terms cancel. The independent explicit
    quadrature supplies max(Re(full),0)/min(Re(full),0) for tmove, versus
    max(Re(det),0)/min(Re(det),0)*exp(deltaJ) for dltmove. Both signs and imaginary
    contributions must be active. This tests ECP increments, not an independent
    derivation of every kinetic/projection term, and uses no MC error criterion.
    """
    ham, up, down, RT = _case()
    terms = ecp_terms(ham, up, down, RT)
    _activated(terms)
    _local, _up, _down, det_terms, jratio, full_terms = terms
    cp = ham.coulomb_potential_data
    coefficients = list(cp.coefficients)
    for i, (atom, ell) in enumerate(zip(cp.nucleus_index, cp.ang_moms, strict=True)):
        if ell < cp.max_ang_mom_plus_1[atom]:
            coefficients[i] = 0
    zero = ham.replace(coulomb_potential_data=cp.replace(coefficients=tuple(coefficients)))

    def evaluate(h):
        if algorithm == "conventional":
            return np.asarray(jqmc_gfmc._compute_V_elements_t(h, up, down, RT, move, 0.3))
        return np.asarray(_production_n()(h, up, down, RT, RT, move, 0.3, False))

    real_terms = np.real(det_terms if move == "dltmove" else full_terms)
    positive = np.sum(np.maximum(real_terms, 0))
    negative = np.sum(np.minimum(real_terms, 0) * (np.real(jratio) if move == "dltmove" else 1))
    expected = np.asarray([positive, negative, positive + negative])
    _compare(evaluate(ham) - evaluate(zero), expected)


@pytest.mark.parametrize("fault", ["omit-real-projection", "absolute-instead-of-real"])
def test_fixed_phase_candidate_oracle_rejects_projection_fault(fault):
    """Independent candidate values reject missing Re and magnitude replacing signed Re.

    Mutate the real projector expression in an isolated compiled namespace;
    leave the fixture and independently assembled complex ratios unchanged.
    The ordinary comparison must pass first and fail after either mutation.
    """
    ham, up, down, RT = _case()
    terms = ecp_terms(ham, up, down, RT)
    _activated(terms)
    reference = np.real(terms[-1])
    args = (ham.coulomb_potential_data, ham.wavefunction_data, up, down, RT)
    _compare(coulomb.compute_ecp_non_local_parts_nearest_neighbors(*args)[2], reference)
    source = textwrap.dedent(inspect.getsource(coulomb._compute_ecp_non_local_parts_nearest_neighbors_impl))
    old = "P_l_chunk = jnp.real(P_l_chunk)"
    assert source.count(old) == 1
    replacement = "P_l_chunk = P_l_chunk" if fault == "omit-real-projection" else "P_l_chunk = jnp.abs(P_l_chunk)"
    namespace = dict(vars(coulomb))
    exec(compile(ast.parse(source.replace(old, replacement)), "<ecp-projection-fault>", "exec"), namespace)
    changed = namespace["_compute_ecp_non_local_parts_nearest_neighbors_impl"](*args)[2]
    with pytest.raises(AssertionError):
        _compare(changed, reference)


@pytest.mark.parametrize("fault", ["force-real", "conjugate"])
def test_vmc_absolute_oracle_rejects_complex_ecp_fault(fault, monkeypatch):
    """Dropping or conjugating only the production ECP imaginary value breaks the absolute anchor.

    The unchanged Li DTN fixture and independent primitive/Hessian/quadrature
    reference must first pass. Mutate only the ratio-bearing contraction in
    the shared production fast ECP kernel; the actual VMC fast-energy consumer
    must then fail the same strict comparison in either precision mode.
    """
    ham, up, down, RT = _case()
    _activated(ecp_terms(ham, up, down, RT))
    reference = complex_local_energy_reference(ham, up, down, RT)
    inverse = _ginv(ham, up, down)
    _compare(compute_local_energy_fast(ham, up, down, RT, inverse), reference)
    name = "_compute_ecp_non_local_parts_nearest_neighbors_fast_update_impl"
    source = textwrap.dedent(inspect.getsource(getattr(coulomb, name)))
    old = "if _is_complex_wf and project_real:"
    assert source.count(old) == 1
    if fault == "force-real":
        source = source.replace(old, "if _is_complex_wf:")
    else:
        old = 'V_nonlocal_chunk = jnp.einsum("lg,lg->g", V_l_chunk, P_l_chunk)'
        assert source.count(old) == 1
        source = source.replace(old, 'V_nonlocal_chunk = jnp.einsum("lg,lg->g", V_l_chunk, jnp.conj(P_l_chunk))')
    namespace = dict(vars(coulomb))
    exec(compile(ast.parse(source), "<vmc-complex-ecp-fault>", "exec"), namespace)
    monkeypatch.setattr(coulomb, name, namespace[name])
    jax.clear_caches()
    changed = compute_local_energy_fast(ham, up, down, RT, inverse)
    with pytest.raises(AssertionError):
        _compare(changed, reference)


@pytest.mark.parametrize("route", ["standard", "fast"])
def test_vmc_complex_ecp_reverse_mode_position_gradient_has_no_complex_cast(route):
    """Reverse-mode position gradients of the complex VMC ECP need no complex-to-real cast.

    The unchanged Li twist DTN fixture, fixed seed7 image-offset positions and
    identity twelve-point mesh feed the standard holomorphic nonlocal ECP
    kernel or its fast sibling with the production reused inverse. The retained complex matrix
    element makes the V_l x P_l contraction complex, so its backward pass must
    not cast a complex cotangent into the real radial V_l leaf (the suite-wide
    ComplexWarning guard turns that cast into an error on jax 0.7.x). Because
    Re of the holomorphic sum is the fixed-phase sum as a function of the
    positions, the reverse-mode gradient of Re with respect to the up and down
    electrons must equal the gradient of the separately projected fixed-phase
    kernel within the existing strict zone budget of each precision mode. The
    Im gradient must be finite and active so the complex branch is exercised.
    """
    ham, up, down, RT = _case()
    _activated(ecp_terms(ham, up, down, RT))
    cp, wf = ham.coulomb_potential_data, ham.wavefunction_data
    extra = () if route == "standard" else (_ginv(ham, up, down),)
    suffix = "" if route == "standard" else "_fast_update"
    holomorphic = getattr(coulomb, f"compute_ecp_non_local_parts_nearest_neighbors{suffix}_holomorphic")
    projected = getattr(coulomb, f"compute_ecp_non_local_parts_nearest_neighbors{suffix}")

    def nonlocal_sum(kernel, part):
        def value(r_up, r_dn):
            return part(kernel(cp, wf, r_up, r_dn, RT, *extra)[3])

        return jax.grad(value, argnums=(0, 1))(up, down)

    value = holomorphic(cp, wf, up, down, RT, *extra)[3]
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    assert np.iscomplexobj(np.asarray(value))
    assert abs(np.imag(value)) > atol + rtol * abs(np.real(value))
    fixed_phase = nonlocal_sum(projected, jax.numpy.real)
    real_part = nonlocal_sum(holomorphic, jax.numpy.real)
    imag_part = nonlocal_sum(holomorphic, jax.numpy.imag)
    for real_grad, projected_grad, imag_grad in zip(real_part, fixed_phase, imag_part, strict=True):
        assert np.all(np.isfinite(np.asarray(projected_grad))) and np.max(abs(np.asarray(projected_grad))) > atol
        assert np.all(np.isfinite(np.asarray(imag_grad))) and np.max(abs(np.asarray(imag_grad))) > atol
        _compare(real_grad, projected_grad)
