"""Native derivative, force and GFMC routes rank only nuclei carrying non-local ECP channels."""

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.atomic_orbital import AOs_sphe_gto_data
from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors, octahedron_sym_mesh_Nv6
from jqmc.determinant import SD_data
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.molecular_orbital import MOs_data
from jqmc.wavefunction import Wavefunction_data
from tests._mixed_ecp_ae_reference import MIXED_AE_R_UP, mixed_ecp_ae_system
from tests.test_rs_kernel.test_gfmc_forces import ALAT, _ZONES_MOL, _assert_close, _oracle_ecp_grads
from tests.test_rs_kernel.test_gfmc_projection import _EL_ZONES_J, _MESH_GP, _MESH_W, _rs_step_eL, _xla_step
from tests.test_rs_kernel.test_param_grad_el import _ECP_ZONES

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]


def _mixed_hamiltonian(ecp_atom):
    """Keep the authored mixed potential/positions; use rectangular MOs so dE_nl/dC is live."""
    cp, _ = mixed_ecp_ae_system(ecp_atom)
    aos = AOs_sphe_gto_data.build(
        structure_data=cp.structure_data,
        nucleus_index=(0, 0, 1, 1),
        num_ao=4,
        num_ao_prim=4,
        orbital_indices=(0, 1, 2, 3),
        exponents=np.array([0.8, 0.3, 0.9, 0.35]),
        coefficients=np.ones(4),
        angular_momentums=(0, 0, 0, 0),
        magnetic_quantum_numbers=(0, 0, 0, 0),
    )
    mos = MOs_data.build(aos_data=aos, mo_coefficients=np.array([[1.0, 0.0, 0.3, 0.1], [0.1, 1.0, 0.0, 0.2]]))
    geminal = SD_data.build(num_electron_up=2, num_electron_dn=0, mos_up=mos, mos_dn=mos, time_reversal=False)
    # Live J2 distinguishes the full-wavefunction T-move from determinant-localized T-move.
    jastrow = Jastrow_data(
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="pade")
    )
    ham = Hamiltonian_data(
        structure_data=cp.structure_data,
        coulomb_potential_data=cp,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=jastrow),
    )
    ru, rd = MIXED_AE_R_UP.copy(), np.empty((0, 3))
    nearest = np.argmin(np.linalg.norm(ru[:, None] - np.asarray(cp.structure_data.positions)[None], axis=-1), axis=1)
    assert set(cp._nucleus_index_non_local_part) == {ecp_atom}
    assert 1 - ecp_atom in nearest, "an all-electron nucleus must be nearer than every ECP centre for one electron"
    return ham, ru, rd


def _mo_nonlocal_value(ham, ru, rd, coefficients):
    wf = ham.wavefunction_data
    geminal = wf.geminal_data.replace(orb_data_up_spin=wf.geminal_data.orb_data_up_spin.replace(mo_coefficients=coefficients))
    return compute_ecp_non_local_parts_nearest_neighbors(
        ham.coulomb_potential_data, wf.replace(geminal_data=geminal), ru, rd, np.eye(3), NN=1, Nv=6
    )[3]


@pytest.mark.parametrize("ecp_atom", [0, 1], ids=["trailing-ae", "leading-ae"])
def test_mixed_ecp_ae_parameter_derivative_selects_ecp_centres(ecp_atom, monkeypatch):
    """Native MO derivatives equal XLA AD for live rectangular MOs on both mixed layouts.

    The two nuclei and authored s/p non-local channels come from the independently
    validated octahedral fixture. Four AOs/two MOs prevent the square determinant's
    coefficient-independent ratios; Pade J2 is active and the down-spin sector is empty.
    """
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    ham, ru, rd = _mixed_hamiltonian(ecp_atom)
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    coefficients = np.asarray(wf.geminal_data.orb_data_up_spin.mo_coefficients)
    reference = np.asarray(jax.grad(lambda c: _mo_nonlocal_value(ham, ru, rd, c))(coefficients))
    native, down = jqmc_rs.grad_eL_ecp_wrt_mo_coeff(
        wf.geminal_data,
        wf.jastrow_data,
        cp,
        ru,
        rd,
        np.asarray(octahedron_sym_mesh_Nv6.weights),
        np.asarray(octahedron_sym_mesh_Nv6.grid_points),
    )
    atol, rtol = get_tolerance_xla_rust_consistency(_ECP_ZONES, level="medium")
    assert np.isfinite(reference).all() and np.max(np.abs(reference)) > atol
    assert np.asarray(native).shape == reference.shape == (2, 4)
    np.testing.assert_allclose(native, reference, atol=atol, rtol=rtol)
    np.testing.assert_array_equal(down, np.zeros_like(down))


@pytest.mark.parametrize("ecp_atom", [0, 1], ids=["trailing-ae", "leading-ae"])
@pytest.mark.parametrize("dltmove", [False, True], ids=["tmove", "dltmove"])
def test_mixed_ecp_ae_gfmc_force_selects_ecp_centres(ecp_atom, dltmove, monkeypatch):
    """Native molecular GFMC nuclear/electron gradients match the existing differentiated XLA oracle.

    Fixed authored electron coordinates put an electron nearer the bare nucleus;
    NN=1 must still use the sole ECP centre, in both active-J2 T-move conventions.
    """
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    ham, ru, rd = _mixed_hamiltonian(ecp_atom)
    wf = ham.wavefunction_data
    reference = _oracle_ecp_grads(ham, ru, rd, dltmove=dltmove)
    native = jqmc_rs._gfmc_grad_e_L_ecp(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        ru,
        rd,
        ALAT,
        np.eye(3),
        _MESH_W,
        _MESH_GP,
        1,
        dltmove,
    )
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_MOL, level="medium")
    for axis, observed, expected in zip(("nuclei", "up", "down"), native, reference, strict=True):
        observed, expected = np.asarray(observed), np.asarray(expected)
        assert observed.shape == expected.shape and np.isfinite(observed).all() and np.isfinite(expected).all()
        if expected.size:
            assert np.max(np.abs(expected)) > atol
            _assert_close(observed, expected, atol, rtol, (ecp_atom, dltmove, axis))
        else:
            assert expected.shape == (0, 3)


@pytest.mark.parametrize("ecp_atom", [0, 1], ids=["trailing-ae", "leading-ae"])
@pytest.mark.parametrize("non_local_move", ["tmove", "dltmove"])
@pytest.mark.parametrize("fast", [False, True], ids=["rebuild", "fast-mesh"])
def test_mixed_ecp_ae_gfmc_projection_selects_ecp_centres(ecp_atom, non_local_move, fast, monkeypatch):
    """Native GFMC rebuild/fast-mesh e_L and b_L agree with the existing XLA step at two lattice spacings.

    Same two-up-electron mixed molecule, live Pade J2 and fixed identity rotation;
    the projection helper evaluates the production closure used by GFMC_n/GFMC_t.
    No stochastic trajectory or statistical error envelope is used.
    """
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    ham, ru, rd = _mixed_hamiltonian(ecp_atom)
    atol, rtol = get_tolerance_xla_rust_consistency(_EL_ZONES_J, level="medium")
    for alat in (0.3, 1.0):
        reference = _xla_step(ham, ru, rd, alat, np.eye(3), non_local_move)
        native = _rs_step_eL(ham, ru, rd, alat, np.eye(3), non_local_move, fast=fast)
        expected = np.asarray([reference["e_L"], reference["b_L"]])
        assert np.isfinite(expected).all() and np.max(np.abs(expected)) > atol
        np.testing.assert_allclose(native, expected, atol=atol, rtol=rtol)
