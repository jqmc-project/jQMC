"""Layer-1 equivalence test: jqmc_rs.compute_AOs vs jqmc.

Molecular GTO AO value evaluation, cartesian and real-spherical, across
angular momenta l = 0,1,2(,3). Autograd-free. The pGTO image-expansion /
Bloch path is covered elsewhere -- molecular only.

The tests build their own synthetic AOs_cart_gto_data / AOs_sphe_gto_data (no
Hamiltonian fixture involved) and take their tolerances from the zone-derived helper.
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

import numpy as np
import pytest

import jqmc_rs

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.atomic_orbital import (
    AOs_cart_gto_data,
    AOs_sphe_gto_data,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
)
from jqmc.structure import Structure_data

# Value evaluation only touches the ao_eval zone.
_ZONES_VALUE = ("ao_eval",)
# Analytic gradient + Laplacian is the unified ao_grad_lap zone.
_ZONES_GRAD_LAP = ("ao_grad_lap",)


def _structure(R_carts):
    n = len(R_carts)
    return Structure_data(
        pbc_flag=False,
        positions=np.asarray(R_carts, dtype=np.float64),
        atomic_numbers=tuple([6] * n),
        element_symbols=tuple(["X"] * n),
        atomic_labels=tuple(["X"] * n),
    )


def _make_sphe(R_carts, rng, lmax=2):
    """s/p/d(/f) sphe-GTO basis on each atom, 2 primitives per shell."""
    structure = _structure(R_carts)
    nucleus_index, orbital_indices, exponents, coefficients = [], [], [], []
    angular_momentums, magnetic = [], []
    ao_idx = 0
    for a in range(len(R_carts)):
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                nucleus_index.append(a)
                angular_momentums.append(l)
                magnetic.append(m)
                for _ in range(2):  # 2 primitives
                    orbital_indices.append(ao_idx)
                    exponents.append(0.5 + rng.random())
                    coefficients.append(0.3 + rng.random())
                ao_idx += 1
    return AOs_sphe_gto_data._from_full_basis(
        exponents=np.array(exponents, dtype=np.float64),
        coefficients=np.array(coefficients, dtype=np.float64),
        structure_data=structure,
        nucleus_index=tuple(nucleus_index),
        num_ao=ao_idx,
        num_ao_prim=len(exponents),
        orbital_indices=tuple(orbital_indices),
        angular_momentums=tuple(angular_momentums),
        magnetic_quantum_numbers=tuple(magnetic),
    )


# A small set of cartesian (nx,ny,nz) shells: s, p, and some d.
_CART_ORDERS = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (2, 0, 0),
    (1, 1, 0),
    (0, 1, 1),
]


def _make_cart(R_carts, rng):
    structure = _structure(R_carts)
    nucleus_index, orbital_indices, exponents, coefficients = [], [], [], []
    angular_momentums, px, py, pz = [], [], [], []
    ao_idx = 0
    for a in range(len(R_carts)):
        for nx, ny, nz in _CART_ORDERS:
            nucleus_index.append(a)
            angular_momentums.append(nx + ny + nz)
            px.append(nx)
            py.append(ny)
            pz.append(nz)
            for _ in range(2):
                orbital_indices.append(ao_idx)
                exponents.append(0.5 + rng.random())
                coefficients.append(0.3 + rng.random())
            ao_idx += 1
    return AOs_cart_gto_data._from_full_basis(
        exponents=np.array(exponents, dtype=np.float64),
        coefficients=np.array(coefficients, dtype=np.float64),
        structure_data=structure,
        nucleus_index=tuple(nucleus_index),
        num_ao=ao_idx,
        num_ao_prim=len(exponents),
        orbital_indices=tuple(orbital_indices),
        angular_momentums=tuple(angular_momentums),
        polynominal_order_x=tuple(px),
        polynominal_order_y=tuple(py),
        polynominal_order_z=tuple(pz),
    )


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 4, 19])
@pytest.mark.parametrize("lmax", [1, 2, 3])
def test_compute_aos_sphe_matches_jax(seed, lmax):
    """Compute AOs sphe matches jax."""
    rng = np.random.default_rng(seed)
    R = (rng.random((3, 3)) - 0.5) * 4.0
    aos = _make_sphe(R, rng, lmax=lmax)
    r_elec = (rng.random((6, 3)) - 0.5) * 3.0
    ref = np.asarray(compute_AOs(aos, r_elec), dtype=np.float64)
    got = jqmc_rs.compute_AOs(aos, r_elec)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_VALUE, level="strict")
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert np.allclose(got, ref, rtol=rtol, atol=atol), np.abs(got - ref).max()


@pytest.mark.light
@pytest.mark.parametrize("backend", ["xla", "rust"])
def test_compute_aos_sphe_l6_angular_matches_scipy_legendre(backend):
    """l=6 sphe AO angular shape matches a scipy associated-Legendre reference.

    The consistency test above stops at l=3, and at l=6 its XLA-vs-Rust fp32
    envelope is narrower than the legitimate fp32 rounding of the two
    polynomial evaluations, so the l=6 angular part is checked against an
    external reference instead. Electrons sit on a sphere around the AO centre,
    so the radial factor is one unknown constant: each AO's values must be a
    positive multiple of P_6^|m|(cos theta) * (cos|m|phi | 1 | sin|m|phi), with
    scipy's Condon-Shortley phase removed (real Wikipedia convention).
    """
    import scipy.special

    from jqmc._precision import get_tolerance

    l = 6
    rng = np.random.default_rng(6)
    center = np.array([[0.3, -0.2, 0.5]])
    theta = np.arccos(rng.uniform(-1.0, 1.0, 24))
    phi = rng.uniform(0.0, 2.0 * np.pi, 24)
    dirs = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=1)
    r_elec = center + 1.3 * dirs
    atol, rtol = get_tolerance("ao_eval", "strict")
    for m in range(-l, l + 1):
        aos = AOs_sphe_gto_data._from_full_basis(
            exponents=np.array([0.4], dtype=np.float64),
            coefficients=np.array([1.0], dtype=np.float64),
            structure_data=_structure(center),
            nucleus_index=(0,),
            num_ao=1,
            num_ao_prim=1,
            orbital_indices=(0,),
            angular_momentums=(l,),
            magnetic_quantum_numbers=(m,),
        )
        m_abs = abs(m)
        legendre = (-1.0) ** m_abs * scipy.special.lpmv(m_abs, l, np.cos(theta))
        azimuth = np.cos(m_abs * phi) if m > 0 else (np.ones_like(phi) if m == 0 else np.sin(m_abs * phi))
        ref = legendre * azimuth
        if backend == "rust":
            got = np.asarray(jqmc_rs.compute_AOs(aos, r_elec), dtype=np.float64)[0]
        else:
            got = np.asarray(compute_AOs(aos, r_elec), dtype=np.float64)[0]
        # Compare unit-free shapes: both sides scaled to max |value| = 1.
        got = got / np.abs(got).max()
        ref = ref / np.abs(ref).max()
        assert float(got @ ref) > 0.0, f"m={m}: sign flipped"
        np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=f"l={l}, m={m}")


@pytest.mark.light
@pytest.mark.parametrize("backend", ["xla", "rust"])
def test_mcmc_l6_m0_local_energy_matches_harmonic_closed_form(backend, monkeypatch):
    """Production MCMC e_L of one electron in an l=6, m=0 sphe GTO equals a closed form.

    The batch AO test above does not reach the f64 angular table in mixed
    precision (its f32 shell path bypasses it), while the native mixed MCMC
    local energy does. Here psi = S_6,0(r) exp(-a r^2) about a bare proton,
    where S_6,0 is a solid harmonic: harmonic and homogeneous of degree 6.
    Then Lap(psi)/psi = 4 a^2 r^2 - 2 a (2l + 3), so
    e_L = -(4 a^2 r^2 - 30 a)/2 - 1/r at each walker's final position. The
    reference uses only that identity (no production or scipy harmonics),
    and the route is asserted (native kernels ran / did not run).
    """
    from jqmc._precision import get_tolerance_min
    from jqmc.coulomb_potential import Coulomb_potential_data
    from jqmc.determinant import SD_data
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import Jastrow_data
    from jqmc.jqmc_mcmc import MCMC
    from jqmc.molecular_orbital import MOs_data
    from jqmc.wavefunction import Wavefunction_data

    from tests.test_rs_kernel.conftest import assert_native_ran, assert_no_native_ran, native_ctx_spy

    l, alpha, nw = 6, 0.63, 3
    structure = Structure_data(
        pbc_flag=False,
        positions=np.zeros((1, 3)),
        atomic_numbers=(1,),
        element_symbols=("H",),
        atomic_labels=("H",),
    )
    n_ao = 2 * l + 1
    aos = AOs_sphe_gto_data._from_full_basis(
        exponents=np.full(n_ao, alpha),
        coefficients=np.ones(n_ao),
        structure_data=structure,
        nucleus_index=(0,) * n_ao,
        num_ao=n_ao,
        num_ao_prim=n_ao,
        orbital_indices=tuple(range(n_ao)),
        angular_momentums=(l,) * n_ao,
        magnetic_quantum_numbers=tuple(range(-l, l + 1)),
    )
    mo_coefficients = np.zeros((1, n_ao))
    mo_coefficients[0, l] = 1.0  # the m = 0 member of the complete l=6 shell
    mos = MOs_data(num_mo=1, aos_data=aos, mo_coefficients=mo_coefficients)
    ham = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=Coulomb_potential_data(structure_data=structure, ecp_flag=False),
        wavefunction_data=Wavefunction_data(
            jastrow_data=Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None),
            geminal_data=SD_data.build(num_electron_up=1, num_electron_dn=0, mos_up=mos, mos_dn=mos),
        ),
    )
    ham.sanity_check()

    if backend == "rust":
        monkeypatch.delenv("JQMC_FLAG_RUST_ON_CPU", raising=False)
    else:
        monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    driver = MCMC(
        hamiltonian_data=ham,
        Dt=0.3,
        mcmc_seed=29,
        epsilon_AS=0.0,
        num_walkers=nw,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
    )
    with native_ctx_spy() as counts:
        driver.run(num_mcmc_steps=7)
    if backend == "rust":
        assert_native_ran(counts, "l6 m0 Rust MCMC")
    else:
        assert_no_native_ran(counts, "l6 m0 XLA MCMC")

    r = np.asarray(driver._MCMC__latest_r_up_carts, dtype=np.float64).reshape(nw, 3)
    r2 = np.sum(r * r, axis=1)
    expected = -0.5 * (4.0 * alpha**2 * r2 - 2.0 * alpha * (2 * l + 3)) - 1.0 / np.sqrt(r2)
    actual = np.asarray(driver.e_L, dtype=np.float64)[-1].reshape(nw)
    atol, rtol = get_tolerance_min(("wf_kinetic", "ao_grad_lap", "det_grad_lap", "coulomb"))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=f"{backend}: l=6 m=0 MCMC e_L")


@pytest.mark.light
@pytest.mark.parametrize("seed", [1, 8, 27])
def test_compute_aos_cart_matches_jax(seed):
    """Compute AOs cart matches jax."""
    rng = np.random.default_rng(seed)
    R = (rng.random((2, 3)) - 0.5) * 4.0
    aos = _make_cart(R, rng)
    r_elec = (rng.random((5, 3)) - 0.5) * 3.0
    ref = np.asarray(compute_AOs(aos, r_elec), dtype=np.float64)
    got = jqmc_rs.compute_AOs(aos, r_elec)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_VALUE, level="strict")
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert np.allclose(got, ref, rtol=rtol, atol=atol), np.abs(got - ref).max()


# --- Analytic gradient + laplacian ------------------------------------------


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 4, 19])
@pytest.mark.parametrize("lmax", [1, 2, 3])
def test_compute_aos_grad_sphe_matches_jax(seed, lmax):
    """Compute AOs grad sphe matches jax."""
    rng = np.random.default_rng(seed)
    R = (rng.random((3, 3)) - 0.5) * 4.0
    aos = _make_sphe(R, rng, lmax=lmax)
    r_elec = (rng.random((6, 3)) - 0.5) * 3.0
    gx, gy, gz = (np.asarray(g, dtype=np.float64) for g in compute_AOs_grad(aos, r_elec))
    rx, ry, rz = jqmc_rs.compute_AOs_grad(aos, r_elec)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD_LAP, level="strict")
    for axis, got, ref in [("x", rx, gx), ("y", ry, gy), ("z", rz, gz)]:
        assert np.allclose(got, ref, rtol=rtol, atol=atol), (axis, np.abs(got - ref).max())


@pytest.mark.light
@pytest.mark.parametrize("seed", [1, 8, 27])
def test_compute_aos_grad_cart_matches_jax(seed):
    """Compute AOs grad cart matches jax."""
    rng = np.random.default_rng(seed)
    R = (rng.random((2, 3)) - 0.5) * 4.0
    aos = _make_cart(R, rng)
    r_elec = (rng.random((5, 3)) - 0.5) * 3.0
    gx, gy, gz = (np.asarray(g, dtype=np.float64) for g in compute_AOs_grad(aos, r_elec))
    rx, ry, rz = jqmc_rs.compute_AOs_grad(aos, r_elec)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD_LAP, level="strict")
    for axis, got, ref in [("x", rx, gx), ("y", ry, gy), ("z", rz, gz)]:
        assert np.allclose(got, ref, rtol=rtol, atol=atol), (axis, np.abs(got - ref).max())


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 4, 19])
@pytest.mark.parametrize("lmax", [1, 2, 3])
def test_compute_aos_laplacian_sphe_matches_jax(seed, lmax):
    """Compute AOs laplacian sphe matches jax."""
    rng = np.random.default_rng(seed)
    R = (rng.random((3, 3)) - 0.5) * 4.0
    aos = _make_sphe(R, rng, lmax=lmax)
    r_elec = (rng.random((6, 3)) - 0.5) * 3.0
    ref = np.asarray(compute_AOs_laplacian(aos, r_elec), dtype=np.float64)
    got = jqmc_rs.compute_AOs_laplacian(aos, r_elec)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD_LAP, level="strict")
    assert np.allclose(got, ref, rtol=rtol, atol=atol), np.abs(got - ref).max()


@pytest.mark.light
@pytest.mark.parametrize("seed", [1, 8, 27])
def test_compute_aos_laplacian_cart_matches_jax(seed):
    """Compute AOs laplacian cart matches jax."""
    rng = np.random.default_rng(seed)
    R = (rng.random((2, 3)) - 0.5) * 4.0
    aos = _make_cart(R, rng)
    r_elec = (rng.random((5, 3)) - 0.5) * 3.0
    ref = np.asarray(compute_AOs_laplacian(aos, r_elec), dtype=np.float64)
    got = jqmc_rs.compute_AOs_laplacian(aos, r_elec)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD_LAP, level="strict")
    assert np.allclose(got, ref, rtol=rtol, atol=atol), np.abs(got - ref).max()
