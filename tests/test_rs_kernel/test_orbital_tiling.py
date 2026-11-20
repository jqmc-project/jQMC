"""Native parity for the representation-independent primitive block provider."""

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

import numpy as np
import pytest

import jqmc_rs
from jqmc._setting import get_tolerance_xla_rust_consistency
from jqmc.orbital_tiling import TiledOrbitalSet_data, compute_tiled_orbitals_value_grad_lap
from tests.test_jqmc_kernel.test_orbital_tiling import _backend_set

pytestmark = [pytest.mark.without_chdir]


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("pw", marks=pytest.mark.light),
        pytest.param("bspline", marks=pytest.mark.light),
        pytest.param("pgto_cart", marks=pytest.mark.heavy),
        pytest.param("pgto_sphe", marks=pytest.mark.heavy),
        pytest.param("composite_cart", marks=pytest.mark.heavy),
        pytest.param("composite_sphe", marks=pytest.mark.heavy),
    ],
)
def test_native_block_model_vgl_and_logical_conjugation(backend, monkeypatch):
    """Native ragged local contractions match XLA, including whole-orbital conjugation."""
    orbitals = _backend_set(backend)
    positions = np.array([[0.51, 0.62, 0.73], [1.13, 0.41, 1.21], [8.13, -3.41, 6.21]], dtype=np.float64)
    partner = TiledOrbitalSet_data.build(
        blocks=[block.replace(conjugate=True) for block in orbitals.blocks],
        tiling_matrix=orbitals.tiling_matrix,
        supertwist=-np.asarray(orbitals.supertwist),
        spin="down",
    )
    for provider in (orbitals, partner):
        native = jqmc_rs.compute_tiled_orbitals_vgl(provider, positions)
        with monkeypatch.context() as context:
            context.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
            reference = compute_tiled_orbitals_value_grad_lap(provider, positions)
        for index, (actual, expected) in enumerate(zip(native, reference, strict=True)):
            zones = ("ao_eval", "tiling_eval") if index == 0 else ("ao_grad_lap", "tiling_grad", "tiling_lap")
            atol, rtol = get_tolerance_xla_rust_consistency(zones)
            assert actual.shape == expected.shape == (provider.num_mo, len(positions))
            assert actual.dtype == np.complex128
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)

    with pytest.raises(ValueError, match="normalization"):
        jqmc_rs.compute_tiled_orbitals_vgl(orbitals.replace(normalization="unsupported-normalization"), positions)


@pytest.mark.heavy
@pytest.mark.parametrize("backend", ["pw", "bspline", "pgto_cart", "pgto_sphe", "composite_cart", "composite_sphe"])
@pytest.mark.parametrize("restricted", [True, False])
def test_native_tiled_sd_energy_with_independent_supercell_j3(backend, restricted, monkeypatch):
    """Native SD/kinetic/local-energy calls match XLA with finite cross-cell supercell J3."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
    from jqmc.determinant import compute_geminal_all_elements
    from jqmc.hamiltonians import compute_local_energy_fast
    from jqmc.wavefunction import compute_kinetic_energy
    from tests.test_jqmc_kernel.test_orbital_tiling_hdf5 import _hamiltonian
    from tests.test_jqmc_kernel.test_orbital_tiling_j3 import _positions, _supercell_j3

    ham = _hamiltonian(backend, restricted)
    j3 = _supercell_j3(ham.structure_data)
    wf = ham.wavefunction_data
    wf = wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=j3))
    ham = ham.replace(wavefunction_data=wf)
    up, down = _positions(ham.structure_data)
    with monkeypatch.context() as context:
        context.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
        g = np.asarray(compute_geminal_all_elements(wf.geminal_data, up, down))
        inverse = np.linalg.inv(g)
        kinetic = complex(compute_kinetic_energy(wf, up, down))
        energy = complex(compute_local_energy_fast(ham, up, down, np.eye(3), inverse, None))
    mesh = icosahedron_sym_mesh_Nv12

    def forbidden(*args, **kwargs):
        pytest.fail("native tiled SD evaluation called the XLA evaluator")

    with monkeypatch.context() as context:
        context.setattr("jqmc.orbital_tiling.compute_tiled_orbitals", forbidden)
        context.setattr("jqmc.orbital_tiling.compute_tiled_orbitals_value_grad_lap", forbidden)
        context.setattr("jqmc.determinant.compute_tiled_orbitals", forbidden)
        context.setattr("jqmc.determinant.compute_tiled_orbitals_value_grad_lap", forbidden)
        actual_g = jqmc_rs.compute_geminal_complex(wf.geminal_data, up, down)
        actual_t = complex(*jqmc_rs.compute_kinetic_withj_complex(wf.geminal_data, wf.jastrow_data, up, down))
        actual_e = complex(
            *jqmc_rs.compute_local_energy_fast_complex(
                ham, up, down, np.asarray(mesh.weights, float), np.asarray(mesh.grid_points, float), inverse
            )
        )
    atol, rtol = get_tolerance_xla_rust_consistency(
        ("tiling_eval", "ao_eval", "det_eval", "wf_kinetic", "coulomb", "jastrow_grad_lap")
    )
    np.testing.assert_allclose(actual_g, g, atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual_t, kinetic, atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual_e, energy, atol=atol, rtol=rtol)


@pytest.mark.heavy
@pytest.mark.parametrize("backend", ["pw", "bspline3", "pgto_cart", "pgto_sphe", "composite_cart", "composite_sphe"])
@pytest.mark.parametrize("conjugate", [False, True])
def test_native_tiled_physical_vgl_cotangents(backend, conjugate, monkeypatch):
    """Native physical atom/strain/electron VGL responses equal XLA analytic with no fallback."""
    from jqmc.derivatives.orbital_tiling import tiled_field_vjp
    from jqmc.orbital_tiling import conjugate_tiled_orbitals
    from tests.test_jqmc_kernel.test_orbital_tiling_supercell import _source
    from tests.test_jqmc_kernel.test_orbital_tiling_hdf5 import _hamiltonian

    source = (
        _hamiltonian("pw", True).wavefunction_data.geminal_data.orb_data_up_spin
        if backend == "pw"
        else _source(backend, "general")
    )
    if conjugate:
        source = conjugate_tiled_orbitals(source, spin="down")
    points = np.array([[0.6, 0.7, 0.8], [4.2, -1.3, 1.1], [8.1, 7.2, -0.3]])
    rng = np.random.default_rng(231)
    bars = rng.normal(size=(5, source.num_mo, len(points))) + 1j * rng.normal(size=(5, source.num_mo, len(points)))
    expected = [
        np.asarray(tiled_field_vjp(source, points, bars[0], bars[1:4], bars[4], wrt=coordinate))
        for coordinate in ("R", "eps", "r")
    ]

    def forbidden(*args, **kwargs):
        pytest.fail("native tiled derivative called XLA")

    with monkeypatch.context() as context:
        context.setattr("jqmc.derivatives.orbital_tiling.tiled_field_vjp", forbidden)
        context.setattr("jqmc.orbital_tiling.compute_tiled_orbitals", forbidden)
        context.setattr("jqmc.orbital_tiling.compute_tiled_orbitals_value_grad_lap", forbidden)
        actual = jqmc_rs.compute_tiled_orbitals_vjp(source, points, *bars)
    atol, rtol = get_tolerance_xla_rust_consistency(("tiling_grad", "tiling_lap", "ao_eval", "ao_grad_lap"), level="medium")
    for observed, wanted in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, wanted, atol=atol, rtol=rtol)


@pytest.mark.heavy
@pytest.mark.parametrize("backend", ["pw", "bspline", "pgto_cart", "pgto_sphe", "composite_cart", "composite_sphe"])
@pytest.mark.parametrize("restricted", [True, False])
def test_native_tiled_full_force_stress_with_supercell_j3(backend, restricted, monkeypatch):
    """Native full force/stress wrappers match XLA analytical with finite cross-cell J3."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
    from jqmc.determinant import compute_geminal_all_elements
    from tests.test_rs_kernel.test_periodic_lsda import _jax_force, _jax_stress
    from tests.test_jqmc_kernel.test_orbital_tiling_hdf5 import _hamiltonian
    from tests.test_jqmc_kernel.test_orbital_tiling_j3 import _positions, _supercell_j3

    ham = _hamiltonian(backend, restricted)
    wf = ham.wavefunction_data
    wf = wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=_supercell_j3(ham.structure_data)))
    ham = ham.replace(wavefunction_data=wf)
    up, dn = _positions(ham.structure_data)
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    inverse = np.asarray(np.linalg.inv(np.asarray(compute_geminal_all_elements(gd, up, dn))), dtype=complex)
    hf, pulay = _jax_force(ham, up, dn, inverse)
    stress, pulay_stress = _jax_stress(ham, up, dn, inverse)
    mesh = icosahedron_sym_mesh_Nv12
    weights, points = np.asarray(mesh.weights, float), np.asarray(mesh.grid_points, float)

    def forbidden(*args, **kwargs):
        pytest.fail("native full derivative called XLA")

    with monkeypatch.context() as context:
        context.setattr("jqmc.derivatives.orbital_tiling.tiled_field_vjp", forbidden)
        context.setattr("jqmc.orbital_tiling.compute_tiled_orbitals", forbidden)
        native_hf = jqmc_rs.vmc_grad_e_L_pbc_complex(gd, jd, cp, up, dn, weights, points, inverse)
        native_pulay = jqmc_rs.vmc_grad_ln_psi_pbc_complex(gd, jd, up, dn, inverse)
        native_stress = jqmc_rs.vmc_grad_e_L_eps_pbc_complex(gd, jd, cp, up, dn, weights, points, inverse)
        native_pulay_stress = jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(gd, jd, up, dn, inverse)
    atol, rtol = get_tolerance_xla_rust_consistency(
        ("ao_eval", "ao_grad_lap", "tiling_grad", "tiling_lap", "wf_kinetic", "coulomb", "jastrow_grad_lap"), level="medium"
    )
    for actual, expected in zip(
        (*native_hf, *native_pulay, native_stress, native_pulay_stress), (*hf, *pulay, stress, pulay_stress), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
