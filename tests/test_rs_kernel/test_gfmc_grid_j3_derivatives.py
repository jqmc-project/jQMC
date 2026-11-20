"""Same-input GFMC measurement derivatives for active plane-wave and B-spline J3."""

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

from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import get_tolerance_min, get_tolerance_xla_rust_consistency, is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from jqmc.hamiltonians import Hamiltonian_data
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from jqmc.derivatives import gfmc
from tests._gfmc_n_same_input import leaf_reference
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel import test_gfmc_forces as helpers

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]

_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb", "jastrow_eval", "jastrow_grad_lap")


def _grid_j3_hamiltonian(basis):
    """Committed QE Li Gamma ccECP PW/B-spline SD with exp J1/J2 and active identity-MO J3.

    The fixture J3 coupling is a seed-5 draw of O(0.1) that was sized for the
    raw QE band amplitude. The B-spline AO leaf carries
    the unit-cell normalization Omega^(-1/2), while the plane-wave leaf does not.
    For B-spline the pair block is therefore rescaled by Omega and the one-body
    column by Omega^(1/2), which restores exactly the J3 function this test
    was written for. Without it the J3 force contribution of that leaf drops
    below the medium mixed XLA/Rust tolerance and the comparison could not see
    a native kernel that ignores J3.
    """
    ham = _build_ham_for_cell(f"QEPROJ:{basis}:Li_gamma:ECP", True, {"j3": "mo"})
    if basis == "bspline":
        omega = abs(float(np.linalg.det(ham.structure_data.cell)))
        wf = ham.wavefunction_data
        j3 = wf.jastrow_data.jastrow_three_body_data
        jm = np.array(j3.j_matrix, dtype=np.float64)
        jm[:, :-1] *= omega
        jm[:, -1] *= np.sqrt(omega)
        jastrow = wf.jastrow_data.replace(jastrow_three_body_data=j3.replace(j_matrix=jm))
        ham = ham.replace(wavefunction_data=wf.replace(jastrow_data=jastrow))
    ham.sanity_check()
    return ham


def _full_physical_source_factory(basis):
    """Reload independent models from one fresh full-precision physical source."""
    directory = tempfile.TemporaryDirectory(prefix="jqmc_gfmc_grid_fidelity_")
    source = None

    def factory():
        nonlocal source
        if source is None:
            assert not is_mixed_precision_enabled()
            from tests.test_jqmc_kernel._generated_matrix_base import _qe_proj_pieces

            _qe_proj_pieces.cache_clear()
            source = Path(directory.name) / "full_physical_source.h5"
            _grid_j3_hamiltonian(basis).save_to_hdf5(source)
        return Hamiltonian_data.load_from_hdf5(source)

    return factory


def _analytical_outputs(ham, move):
    """Original analytical XLA measurement quantities at the same seed-0 configuration."""
    ru, rd = (jnp.asarray(value) for value in helpers._pbc_cfg(ham, 0))
    forces = gfmc.grad_e_L_t_wrt_positions(ham, ru, rd, jnp.eye(3), 0.30, non_local_move=move)
    strain = gfmc.grad_e_L_t_wrt_strain(ham, ru, rd, jnp.eye(3), 0.30, non_local_move=move)
    assert len(forces) == 3
    return {
        name: np.asarray(value)
        for name, value in zip(("nuclear", "electron-up", "electron-down", "strain"), (*forces, strain), strict=True)
    }


def assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Apply the central block budget at a source-qualified local receipt site."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


@pytest.fixture(autouse=True)
def _release_oracle_compilations():
    """Release the distinct fixed-input derivative programs between cases."""
    yield
    jax.clear_caches()


@pytest.mark.parametrize("basis", ["bspline", "pw"])
@pytest.mark.parametrize("dltmove", [False, True], ids=["tmove", "dltmove"])
def test_gfmc_grid_j3_derivatives_match_energy_ad(basis, dltmove):
    """Full GFMC gradients equal measurement-energy AD; mixed retains analytical full fidelity.

    Committed QE Li Gamma ccECP becomes a PW or B-spline SD with exp J1/J2
    and active identity-MO-wrapped J3 (see _grid_j3_hamiltonian). Seed 0 draws
    one near-nucleus electron configuration using _pbc_cfg, without cond(G)
    filtering. Identity Nv12 quadrature and alat=0.30 are shared; under
    strain the ECP quadrature rotation stays fixed, so only the
    kinetic mesh follows F.T. The
    independent leaf_reference composes the measurement energy, not a
    derivative kernel. Nuclear and both electron gradients and all nine
    affine-strain components retain strict full AD. Mixed compares the original analytical
    route to independent full analytical models from identical physical input under the
    central homogeneous-block parameter-derivative budget, with an isolated perturbation
    rejection. No chain averages or native/XLA parity policy are changed.
    """
    if is_mixed_precision_enabled():
        move = "dltmove" if dltmove else "tmove"

        def analytical(fresh):
            values = _analytical_outputs(fresh, move)
            assert set(values) == {"nuclear", "electron-up", "electron-down", "strain"}, values.keys()
            geminal = fresh.wavefunction_data.geminal_data
            shapes = {
                "nuclear": np.asarray(fresh.structure_data.positions).shape,
                "electron-up": (int(geminal.num_electron_up), 3),
                "electron-down": (int(geminal.num_electron_dn), 3),
                "strain": (3, 3),
            }
            for name, value in values.items():
                assert value.shape == shapes[name], (name, value.shape, shapes[name])
                assert np.all(np.isfinite(value)), (name, value)
            return values

        mixed, full = mixed_vs_full(
            _full_physical_source_factory(basis),
            analytical,
            configure_native=jqmc_rs.configure_native_precision,
            native_policy_snapshot=jqmc_rs.native_precision_support,
        )
        blocks = mixed_fidelity_blocks(mixed, full)
        atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
        for quantity, (mixed_block, full_block) in zip(sorted(mixed), blocks, strict=True):
            assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
        mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
        return

    ham = _grid_j3_hamiltonian(basis)
    ru, rd = (jnp.asarray(value) for value in helpers._pbc_cfg(ham, 0))
    positions = jnp.asarray(ham.structure_data.positions)
    move = "dltmove" if dltmove else "tmove"

    def energy(nuclei, up, down):
        return jnp.real(
            leaf_reference(helpers._inject(ham, nuclei), up, down, jnp.eye(3), jnp.eye(3), 0.30, non_local_move=move)[4]
        )

    def strained_energy(eps):
        deformation = jnp.eye(3) + eps
        return jnp.real(
            leaf_reference(
                ham._apply_strain(deformation),
                ru @ deformation.T,
                rd @ deformation.T,
                deformation.T,
                jnp.eye(3),
                0.30,
                non_local_move=move,
            )[4]
        )

    reference_force = jax.jit(jax.grad(energy, argnums=(0, 1, 2)))(positions, ru, rd)
    reference_stress = jax.jit(jax.jacrev(strained_energy))(jnp.zeros((3, 3)))
    xla_force = gfmc.grad_e_L_t_wrt_positions(ham, ru, rd, jnp.eye(3), 0.30, non_local_move=move)
    xla_stress = gfmc.grad_e_L_t_wrt_strain(ham, ru, rd, jnp.eye(3), 0.30, non_local_move=move)
    atol, rtol = get_tolerance_min(_ZONES, level="medium" if is_mixed_precision_enabled() else "strict")
    for quantity, actual, reference in zip(
        ("nuclear", "electron-up", "electron-down", "strain"),
        (*xla_force, xla_stress),
        (*reference_force, reference_stress),
        strict=True,
    ):
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=quantity)


@pytest.mark.parametrize("basis", ["bspline", "pw"])
@pytest.mark.parametrize("dltmove", [False, True], ids=["tmove", "dltmove"])
def test_native_gfmc_grid_j3_derivatives_match_xla(basis, dltmove):
    """Rust returns every XLA GFMC derivative leaf with active PW/B-spline J3.

    Committed QE Li Gamma ccECP uses PW/B-spline SD, exp J1/J2 and active
    identity-MO-wrapped J3 (see _grid_j3_hamiltonian). Both backends receive
    the same seed-0 _pbc_cfg configuration, no cond(G) screening, identity
    Nv12 quadrature and alat0.30.
    Nuclear, up/down electron and all nine strain entries use the existing
    medium XLA/Rust GFMC consistency policy within full and mixed separately.
    The companion energy-AD test checks the XLA reference independently;
    removing J3 must change the returned forces and strain derivatives at this
    same tolerance, so a native kernel that drops J3 cannot pass.
    """
    ham = _grid_j3_hamiltonian(basis)
    ru, rd = (jnp.asarray(value) for value in helpers._pbc_cfg(ham, 0))
    move = "dltmove" if dltmove else "tmove"
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    weights = np.asarray(icosahedron_sym_mesh_Nv12.weights)
    points = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)
    args = (wf.geminal_data, wf.jastrow_data, cp, np.asarray(ru), np.asarray(rd), 0.30, np.eye(3), weights, points, 1, dltmove)
    native_force = jqmc_rs._gfmc_grad_e_L_ecp_pbc(*args)
    native_stress = jqmc_rs._gfmc_stress_ecp_pbc_withj(*args)
    xla_force = gfmc.grad_e_L_t_wrt_positions(ham, ru, rd, jnp.eye(3), 0.30, non_local_move=move)
    xla_stress = gfmc.grad_e_L_t_wrt_strain(ham, ru, rd, jnp.eye(3), 0.30, non_local_move=move)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    for quantity, actual, reference in zip(
        ("nuclear", "electron-up", "electron-down", "strain"),
        (*native_force, native_stress),
        (*xla_force, xla_stress),
        strict=True,
    ):
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=quantity)
    no_j3 = wf.jastrow_data.replace(jastrow_three_body_data=None)
    without_j3 = jqmc_rs._gfmc_grad_e_L_ecp_pbc(args[0], no_j3, *args[2:])
    without_j3_stress = jqmc_rs._gfmc_stress_ecp_pbc_withj(args[0], no_j3, *args[2:])
    assert not np.allclose(np.concatenate(native_force), np.concatenate(without_j3), atol=atol, rtol=rtol, equal_nan=False)
    assert not np.allclose(native_stress, without_j3_stress, atol=atol, rtol=rtol, equal_nan=False)
