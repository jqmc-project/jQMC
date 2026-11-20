"""Real native consumers keep a J3 whose composite has a phase-free Gamma-complex pGTO child.

PR #93 (issues #88 / #89) made the Rust ``JastrowData`` gate treat an ``AOs_composite_data`` J3
orbital as complex whenever ANY child type name contained ``complex``, and then left the real
``three_body`` as ``None``. The determinant, not the J3 child type, selects the real or complex
native dispatch. ``build_AOs_cart_pgto_complex_data(k_point=None, ...)`` builds a phase-free
Gamma-complex pGTO leaf (unit image phases) that passes the J3 phase-free validation, so a REAL
determinant can carry a J3 whose composite has such a child. The real ``ClefArgs`` local energy
then silently returned the energy of the wavefunction without its J3 (the sweep parser still
accepted the J3, so the public sampler ran and stored the wrong energy).

Wavefunction (Li_gamma pGTO+PW cell, one up and one down electron):

* determinant: one constant orbital per spin (a ``G = 0`` real plane wave), so the determinant
  kinetic term is zero;
* J3 orbital ``phi(r) = 0.4 N sum_n exp(-|r - R0 - n.cell|^2) + 0.2``, ``N = (2/pi)^(3/4)``: a
  normalized periodic s Gaussian on atom 0 (``alpha = 1``) plus a constant real plane-wave child;
* ``J = 0.19 (phi_up + phi_dn) + 0.7 phi_up phi_dn`` (``j_matrix = [[0.7, 0.19]]``);
* explicit zero ECP coefficients, so the only wavefunction-dependent energy is the kinetic term.

Independent reference: the J3 kinetic energy ``T = -1/2 sum_e (Lap_e J + |grad_e J|^2)`` from the
analytic derivatives of the Gaussian lattice sum over the ``[-3, 3]^3`` images (converged: the
nearest omitted image is more than ``3 * 6.63`` Bohr away). No production AO, orbital, Jastrow or
energy helper supplies it. The bare Coulomb / Ewald potential is identical with and without the
J3 at a fixed configuration, so the J3 contribution is ``e_L(J3) - e_L(no J3)`` of the same native
route. At offsets ``(0.3, 0.2, 0.1)`` (up) and ``(0.5, -0.2, 0.1)`` (down) from atom 0 the
reference is ``0.5666084267722298`` Ha.
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

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import get_tolerance_min, precision_policy_snapshot
from jqmc.atomic_orbital import (
    AOs_cart_pgto_complex_data,
    AOs_cart_pgto_data,
    AOs_composite_data,
    AOs_plane_wave_complex_data,
    AOs_plane_wave_data,
    build_AOs_cart_pgto_complex_data,
)
from jqmc.determinant import SD_data
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data
from jqmc.jqmc_mcmc import _NV_BASE_MESH, MCMC, Nv_default
from jqmc.molecular_orbital import MOs_data
from jqmc.wavefunction import Wavefunction_data
from tests.test_rs_kernel._rs_fixtures import rs_ham

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.with_chdir

_WEIGHTS = np.asarray(_NV_BASE_MESH[Nv_default].weights, dtype=np.float64)
_GRID = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, dtype=np.float64)
_GAUSS_WEIGHT = 0.4
_CONST_WEIGHT = 0.2
_J_PAIR = 0.7
_J_ONE = 0.19
_OFFSET_UP = np.array([0.3, 0.2, 0.1])
_OFFSET_DN = np.array([0.5, -0.2, 0.1])
_IMAGES = 3
_MCMC_SEED = 34356
_MCMC_SEGMENTS = (6, 6)
#: Zones crossed by the J3 value / kinetic local energy.
_ZONES = ("ao_eval", "jastrow_eval", "det_eval", "wf_kinetic")


def _phi_value_grad_lap(r: np.ndarray, center: np.ndarray, cell: np.ndarray):
    """Independent value, gradient and Laplacian of ``phi`` from the analytic Gaussian lattice sum."""
    norm = (2.0 / np.pi) ** 0.75
    value, grad, lap = 0.0, np.zeros(3), 0.0
    for n in itertools.product(range(-_IMAGES, _IMAGES + 1), repeat=3):
        d = r - center - np.asarray(n, dtype=np.float64) @ cell
        g = norm * np.exp(-d @ d)
        value += g
        grad += -2.0 * d * g
        lap += (4.0 * (d @ d) - 6.0) * g
    return _GAUSS_WEIGHT * value + _CONST_WEIGHT, _GAUSS_WEIGHT * grad, _GAUSS_WEIGHT * lap


def _j3_kinetic_reference(r_up: np.ndarray, r_dn: np.ndarray, center: np.ndarray, cell: np.ndarray) -> float:
    """``T = -1/2 sum_e (Lap_e J + |grad_e J|^2)`` for one up and one down electron."""
    pu, gu, lu = _phi_value_grad_lap(r_up[0], center, cell)
    pd, gd, ld = _phi_value_grad_lap(r_dn[0], center, cell)
    cu = _J_ONE + _J_PAIR * pd
    cd = _J_ONE + _J_PAIR * pu
    return float(-0.5 * (cu * lu + cu**2 * (gu @ gu) + cd * ld + cd**2 * (gd @ gd)))


def _hamiltonians(gamma_complex_child: bool):
    """(J3 Hamiltonian, J3-free Hamiltonian) on the Li_gamma cell; see the module docstring."""
    base = rs_ham("pbc_composite_jnone")
    st = base.structure_data
    cp0 = base.coulomb_potential_data
    # Same cores, Ewald tables and ECP channels, with every ECP coefficient explicitly zero.
    coulomb = dataclasses.replace(cp0, coefficients=[0.0] * len(cp0.coefficients))
    pgto_kw = dict(
        structure_data=st,
        nucleus_index=(0,),
        num_ao=1,
        num_ao_prim=1,
        orbital_indices=(0,),
        exponents=np.array([1.0]),
        coefficients=np.array([1.0]),
        angular_momentums=(0,),
        polynominal_order_x=(0,),
        polynominal_order_y=(0,),
        polynominal_order_z=(0,),
    )
    if gamma_complex_child:
        gauss = build_AOs_cart_pgto_complex_data(k_point=None, **pgto_kw)
        assert isinstance(gauss, AOs_cart_pgto_complex_data)
    else:
        gauss = AOs_cart_pgto_data.build(**pgto_kw)
    const_pw = AOs_plane_wave_data.build(st, np.zeros((1, 3)), np.zeros(1))
    # The MO dtype follows the AO model: a complex AO child requires complex128 coefficients
    # (``MOs_data.sanity_check``); the values are the same real weights (zero imaginary part).
    mo_dtype = np.complex128 if gamma_complex_child else np.float64
    j3_orb = MOs_data(
        aos_data=AOs_composite_data.build((gauss, const_pw)),
        num_mo=1,
        mo_coefficients=np.array([[_GAUSS_WEIGHT, _CONST_WEIGHT]], dtype=mo_dtype),
    )
    j3_orb.sanity_check()
    j3 = Jastrow_three_body_data(orb_data=j3_orb, j_matrix=np.array([[_J_PAIR, _J_ONE]]))
    j3.sanity_check()
    det_orb = MOs_data(aos_data=const_pw, num_mo=1, mo_coefficients=np.array([[1.0]]))
    geminal = SD_data(num_electron_up=1, num_electron_dn=1, orb_data_up_spin=det_orb, orb_data_dn_spin=det_orb)

    def ham(jastrow):
        wf = Wavefunction_data(jastrow_data=jastrow, geminal_data=geminal)
        hamiltonian = Hamiltonian_data(structure_data=st, coulomb_potential_data=coulomb, wavefunction_data=wf)
        hamiltonian.sanity_check()
        return hamiltonian

    with_j3 = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=j3, jastrow_dtn_data=None
    )
    without = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=None
    )
    return ham(with_j3), ham(without)


def _native_e_l(ham: Hamiltonian_data, r_up: np.ndarray, r_dn: np.ndarray) -> float:
    wf = ham.wavefunction_data
    ctx = jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        False,
        True,
        False,
        want_force=False,
        precision_policy=precision_policy_snapshot(),
    )
    # One constant orbital per spin: the 1x1 geminal matrix is 1 and so is its inverse.
    return float(ctx.e_L(r_up, r_dn, _WEIGHTS, _GRID, np.ones((1, 1))))


def _geometry(ham: Hamiltonian_data):
    st = ham.structure_data
    return np.asarray(st.positions[0], dtype=np.float64), np.asarray(st.cell, dtype=np.float64)


_CHILDREN = ["gamma_complex_pgto", "real_pgto"]


@pytest.mark.light
@pytest.mark.parametrize("child", _CHILDREN)
def test_real_dispatch_gamma_complex_pgto_composite_j3_direct_e_l_keeps_j3(child):
    """Real native context e_L keeps a J3 whose composite has a Gamma-complex pGTO child.

    At the documented configuration, native ``e_L(J3) - e_L(no J3)`` of the real periodic
    ``build_obs_ctx`` context must equal the analytic J3 kinetic reference (0.5666084267722298 Ha),
    in the active precision policy. ``real_pgto`` is the mathematically identical control.
    """
    ham_j3, ham_bare = _hamiltonians(child == "gamma_complex_pgto")
    center, cell = _geometry(ham_j3)
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    r_up = (center + _OFFSET_UP)[None, :]
    r_dn = (center + _OFFSET_DN)[None, :]
    ref = _j3_kinetic_reference(r_up, r_dn, center, cell)
    # Pin the documented value of the independent reference itself.
    np.testing.assert_allclose(ref, 0.5666084267722298, atol=atol, rtol=rtol)
    direct = _native_e_l(ham_j3, r_up, r_dn) - _native_e_l(ham_bare, r_up, r_dn)
    np.testing.assert_allclose(direct, ref, atol=atol, rtol=rtol, err_msg=f"{child}: native direct J3 energy")


@pytest.mark.light
@pytest.mark.parametrize("child", _CHILDREN)
def test_real_dispatch_gamma_complex_pgto_composite_j3_public_mcmc_keeps_j3(child, monkeypatch):
    """Public native real MCMC stores the e_L with the Gamma-complex pGTO composite J3 included.

    Native real MCMC (``run_energy_chunk_batch`` with ``complex=False``) on the J3 Hamiltonian,
    two walkers, fixed quadrature mesh, energy only, two production ``run`` calls. After each
    call the stored e_L of every walker must equal the XLA J3-free e_L (bare potential only) at
    the returned walker plus the analytic J3 kinetic reference there, within the MCMC Rust/XLA
    driver policy of ``test_chain_modes``; at least one compared walker must carry a J3 energy
    larger than the strict tolerance. ``real_pgto`` is the mathematically identical control.
    """
    from tests.test_rs_kernel.test_chain_modes import _assert_value_parity
    from tests.test_rs_kernel.test_richer_production_drivers import _complex_chunk_spy

    ham_j3, ham_bare = _hamiltonians(child == "gamma_complex_pgto")
    center, cell = _geometry(ham_j3)
    monkeypatch.delenv("JQMC_FLAG_RUST_ON_CPU", raising=False)
    calls = _complex_chunk_spy(monkeypatch)
    mcmc = MCMC(
        hamiltonian_data=ham_j3,
        Dt=2.0,
        mcmc_seed=_MCMC_SEED,
        epsilon_AS=1.0e-6,
        num_walkers=2,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        random_discretized_mesh=False,
    )
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    compared = 0
    resolvable = 0
    for steps in _MCMC_SEGMENTS:
        mcmc.run(num_mcmc_steps=steps)
        w_up = np.asarray(mcmc._MCMC__latest_r_up_carts, dtype=np.float64)
        w_dn = np.asarray(mcmc._MCMC__latest_r_dn_carts, dtype=np.float64)
        stored = np.asarray(mcmc.e_L)[-1]
        assert not np.iscomplexobj(stored) and np.all(np.isfinite(stored)), (child, stored)
        for w in range(w_up.shape[0]):
            bare = float(np.real(compute_local_energy(ham_bare, jnp.asarray(w_up[w]), jnp.asarray(w_dn[w]), jnp.eye(3))))
            walker_ref = _j3_kinetic_reference(w_up[w], w_dn[w], center, cell)
            # Count the walkers whose J3 contribution exceeds the comparison tolerance.
            resolvable += int(abs(walker_ref) > atol + rtol * abs(bare))
            _assert_value_parity(float(stored[w]), bare + walker_ref, f"{child}: public MCMC walker {w} stored e_L")
            compared += 1
    assert compared == 2 * len(_MCMC_SEGMENTS)
    # Omitting the J3 cannot pass: at least one compared walker carries a resolvable J3 energy.
    assert resolvable > 0, (child, resolvable)
    # The native REAL energy chunk (complex=False, native random stream) was the route taken.
    assert calls and all(c == (False, True) for c in calls), (child, calls)


@pytest.mark.light
def test_real_dispatch_complex_only_composite_j3_is_rejected_not_dropped():
    """A J3 the real native kernel cannot represent is rejected loudly, never silently dropped.

    Same real determinant, with the J3 constant child replaced by a phase-free (``k = 0``)
    COMPLEX plane wave, which only the complex observable path evaluates. The real periodic
    context and the free real local-energy kernel must raise ``NotImplementedError`` instead of
    returning the energy of the wavefunction without its J3.
    """
    ham_j3, _ = _hamiltonians(True)
    st = ham_j3.structure_data
    wf = ham_j3.wavefunction_data
    j3 = wf.jastrow_data.jastrow_three_body_data
    gauss = j3.orb_data.aos_data.children[0]
    complex_pw = AOs_plane_wave_complex_data.build(st, np.zeros((1, 3)), k_point=(0.0, 0.0, 0.0))
    orb = MOs_data(
        aos_data=AOs_composite_data.build((gauss, complex_pw)),
        num_mo=1,
        mo_coefficients=np.array([[_GAUSS_WEIGHT, _CONST_WEIGHT]], dtype=np.complex128),
    )
    orb.sanity_check()
    j3_complex_only = Jastrow_three_body_data(orb_data=orb, j_matrix=np.array([[_J_PAIR, _J_ONE]]))
    j3_complex_only.sanity_check()
    ham = ham_j3.replace(
        wavefunction_data=wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=j3_complex_only))
    )
    ham.sanity_check()
    center, _ = _geometry(ham)
    r_up = (center + _OFFSET_UP)[None, :]
    r_dn = (center + _OFFSET_DN)[None, :]
    with pytest.raises(NotImplementedError, match="supported only by the complex native path"):
        _native_e_l(ham, r_up, r_dn)
    with pytest.raises(NotImplementedError, match="supported only by the complex native path"):
        jqmc_rs.compute_local_energy_fast(ham, r_up, r_dn, _WEIGHTS, _GRID, np.ones((1, 1)))


def _complex_only_hamiltonian(ham_j3: Hamiltonian_data) -> Hamiltonian_data:
    """``ham_j3`` with its J3 constant child replaced by a phase-free (``k = 0``) COMPLEX plane wave.

    The orbital is mathematically the same ``phi`` (a constant complex plane wave at ``k = 0`` is
    the constant 1), but only the complex observable path can evaluate that child: the real
    native parser cannot represent it.
    """
    st = ham_j3.structure_data
    wf = ham_j3.wavefunction_data
    gauss = wf.jastrow_data.jastrow_three_body_data.orb_data.aos_data.children[0]
    complex_pw = AOs_plane_wave_complex_data.build(st, np.zeros((1, 3)), k_point=(0.0, 0.0, 0.0))
    orb = MOs_data(
        aos_data=AOs_composite_data.build((gauss, complex_pw)),
        num_mo=1,
        mo_coefficients=np.array([[_GAUSS_WEIGHT, _CONST_WEIGHT]], dtype=np.complex128),
    )
    orb.sanity_check()
    j3_complex_only = Jastrow_three_body_data(orb_data=orb, j_matrix=np.array([[_J_PAIR, _J_ONE]]))
    j3_complex_only.sanity_check()
    ham = ham_j3.replace(
        wavefunction_data=wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=j3_complex_only))
    )
    ham.sanity_check()
    return ham


def _documented_configuration(ham: Hamiltonian_data):
    center, cell = _geometry(ham)
    return center, cell, (center + _OFFSET_UP)[None, :], (center + _OFFSET_DN)[None, :]


@pytest.mark.light
def test_real_free_compute_local_energy_keeps_supported_j3_and_rejects_complex_only_j3():
    """The exported free real ``jqmc_rs.compute_local_energy`` never returns an energy without an active J3.

    Supported controls: for the phase-free Gamma-complex pGTO child and the identical real pGTO
    child, the free kernel's ``e_L(J3) - e_L(no J3)`` must equal the analytic J3 kinetic reference
    (0.5666084267722298 Ha). Unsupported input: with the constant child a ``k = 0`` complex plane
    wave (only the complex native path evaluates it), the free kernel must raise
    ``NotImplementedError`` instead of returning the energy of the wavefunction without its J3
    (a J3 contribution of exactly 0).
    """
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    for gamma_complex_child in (True, False):
        ham_j3, ham_bare = _hamiltonians(gamma_complex_child)
        center, cell, r_up, r_dn = _documented_configuration(ham_j3)
        ref = _j3_kinetic_reference(r_up, r_dn, center, cell)
        np.testing.assert_allclose(ref, 0.5666084267722298, atol=atol, rtol=rtol)
        free_j3 = float(jqmc_rs.compute_local_energy(ham_j3, r_up, r_dn, _WEIGHTS, _GRID))
        free_bare = float(jqmc_rs.compute_local_energy(ham_bare, r_up, r_dn, _WEIGHTS, _GRID))
        np.testing.assert_allclose(
            free_j3 - free_bare,
            ref,
            atol=atol,
            rtol=rtol,
            err_msg=f"gamma_complex_child={gamma_complex_child}: free compute_local_energy J3 energy",
        )
    ham_complex_only = _complex_only_hamiltonian(_hamiltonians(True)[0])
    _, _, r_up, r_dn = _documented_configuration(ham_complex_only)
    with pytest.raises(NotImplementedError, match="supported only by the complex native path"):
        jqmc_rs.compute_local_energy(ham_complex_only, r_up, r_dn, _WEIGHTS, _GRID)


@pytest.mark.light
def test_real_free_compute_kinetic_energy_keeps_supported_j3_and_rejects_complex_only_j3():
    """The exported free real ``jqmc_rs.compute_kinetic_energy`` never returns a kinetic energy without an active J3.

    The determinant is one constant orbital per spin, so the whole kinetic energy is the J3 term.
    Supported controls (Gamma-complex pGTO child and the identical real pGTO child): the free
    kernel must equal the analytic J3 kinetic reference (0.5666084267722298 Ha) and the J3-free
    wavefunction must give zero. Unsupported input (``k = 0`` complex plane-wave child): the free
    kernel must raise ``NotImplementedError`` instead of returning ``-0.0``.
    """
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    for gamma_complex_child in (True, False):
        ham_j3, ham_bare = _hamiltonians(gamma_complex_child)
        center, cell, r_up, r_dn = _documented_configuration(ham_j3)
        ref = _j3_kinetic_reference(r_up, r_dn, center, cell)
        np.testing.assert_allclose(ref, 0.5666084267722298, atol=atol, rtol=rtol)
        for ham, expected, label in ((ham_j3, ref, "J3"), (ham_bare, 0.0, "no J3")):
            wf = ham.wavefunction_data
            kinetic = float(jqmc_rs.compute_kinetic_energy(wf.geminal_data, wf.jastrow_data, r_up, r_dn))
            np.testing.assert_allclose(
                kinetic,
                expected,
                atol=atol,
                rtol=rtol,
                err_msg=f"gamma_complex_child={gamma_complex_child}: free compute_kinetic_energy ({label})",
            )
    ham_complex_only = _complex_only_hamiltonian(_hamiltonians(True)[0])
    _, _, r_up, r_dn = _documented_configuration(ham_complex_only)
    wf = ham_complex_only.wavefunction_data
    with pytest.raises(NotImplementedError, match="supported only by the complex native path"):
        jqmc_rs.compute_kinetic_energy(wf.geminal_data, wf.jastrow_data, r_up, r_dn)


@pytest.mark.light
def test_real_free_ecp_nonlocal_rejects_complex_only_j3():
    """A further exported real J3 consumer, ``compute_ecp_nonlocal_parts_all_pairs``, rejects a complex-only J3.

    Every exported real native entry point that reads the J3 of the parsed ``JastrowData``
    goes through the same guard; the free non-local ECP kernel (whose wavefunction ratios
    include the J3) is the representative one. Supported controls: the phase-free Gamma-complex
    pGTO child and the identical real pGTO child are accepted and give the same finite value.
    Unsupported input (``k = 0`` complex plane-wave child): it must raise ``NotImplementedError``
    instead of evaluating the ratios without the J3.
    """
    atol, rtol = get_tolerance_min(_ZONES, "strict")

    def nonlocal_ecp(ham):
        wf = ham.wavefunction_data
        _, _, r_up, r_dn = _documented_configuration(ham)
        return jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(
            wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data, r_up, r_dn, _WEIGHTS, _GRID, False
        )

    controls = [float(nonlocal_ecp(_hamiltonians(gamma_complex_child)[0])) for gamma_complex_child in (True, False)]
    assert np.all(np.isfinite(controls)), controls
    np.testing.assert_allclose(controls[0], controls[1], atol=atol, rtol=rtol)
    with pytest.raises(NotImplementedError, match="supported only by the complex native path"):
        nonlocal_ecp(_complex_only_hamiltonian(_hamiltonians(True)[0]))


@pytest.mark.light
def test_real_native_parser_reads_zero_imaginary_complex128_j3_mo_exactly():
    """The real native parser takes a sanity-valid complex128 J3 MO with zero imaginary part exactly.

    ``MOs_data.sanity_check`` requires complex128 coefficients over the phase-free Gamma-complex
    pGTO child. A cast with ``np.asarray(C, float64)`` in the real ``OrbData`` parser would emit
    ``ComplexWarning`` (an error under the strict cast guard) even for a zero imaginary part. With ``ComplexWarning``
    promoted to an error here as well, on the sanity-valid fixture (MO, J3 and Hamiltonian
    checks pass, coefficient dtype complex128): the real periodic native context ``e_L``, the free
    ``compute_local_energy`` and the free ``compute_kinetic_energy`` must equal the same
    quantities of the identical real pGTO control (float64 coefficients) exactly, and the J3
    energy must equal the analytic reference (0.5666084267722298 Ha) in the active policy.
    Negative control: the same child with a nonzero imaginary MO coefficient is still rejected
    with ``NotImplementedError`` (the real part is never taken silently).
    """
    import warnings

    from numpy.exceptions import ComplexWarning

    atol, rtol = get_tolerance_min(_ZONES, "strict")
    ham_c, bare_c = _hamiltonians(True)
    ham_r, bare_r = _hamiltonians(False)
    c_mo = ham_c.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data
    assert c_mo.mo_coefficients.dtype == np.complex128 and not np.any(c_mo.mo_coefficients.imag)
    center, cell, r_up, r_dn = _documented_configuration(ham_c)
    ref = _j3_kinetic_reference(r_up, r_dn, center, cell)

    def energies(ham, bare):
        wf = ham.wavefunction_data
        return np.array(
            [
                _native_e_l(ham, r_up, r_dn) - _native_e_l(bare, r_up, r_dn),
                float(jqmc_rs.compute_local_energy(ham, r_up, r_dn, _WEIGHTS, _GRID))
                - float(jqmc_rs.compute_local_energy(bare, r_up, r_dn, _WEIGHTS, _GRID)),
                float(jqmc_rs.compute_kinetic_energy(wf.geminal_data, wf.jastrow_data, r_up, r_dn)),
            ]
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error", ComplexWarning)
        got = energies(ham_c, bare_c)
        control = energies(ham_r, bare_r)
    np.testing.assert_array_equal(got, control)
    np.testing.assert_allclose(got, np.full(3, ref), atol=atol, rtol=rtol)

    wf = ham_c.wavefunction_data
    j3 = wf.jastrow_data.jastrow_three_body_data
    imag_orb = MOs_data(
        aos_data=j3.orb_data.aos_data,
        num_mo=1,
        mo_coefficients=np.array([[_GAUSS_WEIGHT, _CONST_WEIGHT + 0.1j]], dtype=np.complex128),
    )
    imag_orb.sanity_check()
    ham_imag = ham_c.replace(
        wavefunction_data=wf.replace(
            jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=j3.replace(orb_data=imag_orb))
        )
    )
    ham_imag.sanity_check()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComplexWarning)
        with pytest.raises(NotImplementedError, match="supported only by the complex native path"):
            _native_e_l(ham_imag, r_up, r_dn)
        with pytest.raises(NotImplementedError, match="supported only by the complex native path"):
            jqmc_rs.compute_local_energy(ham_imag, r_up, r_dn, _WEIGHTS, _GRID)
