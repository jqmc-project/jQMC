"""Independent strain finite differences for the stored MCMC stress buffers.

The reference never calls ``Hamiltonian_data._apply_strain``,
``Wavefunction_data._apply_strain``, ``Jastrow_data._apply_strain``,
``_strain_structure_data``, ``accumulate_strain_grad`` or any stress kernel.
For each strain it rebuilds the whole periodic Hamiltonian from raw
geometry-free inputs through the public constructors, with the cell rows and
every nucleus scaled by ``F = I + eps`` in this file.  Electron coordinates
are scaled the same way.  The ECP quadrature rotation argument ``RT`` stays
the identity: the non-local quadrature points of the strained system lie on
the sphere of the strained electron-nucleus distance along fixed directions,
so the rotation average of the derivative is the exact strain derivative of
the angular integral.  Carrying the rotation as ``RT @ F.T`` would move the
points off that sphere and bias the stress.

What is shared with production is the forward functional itself
(``compute_local_energy`` / ``evaluate_ln_wavefunction``) and the public
constructors that derive the cell-dependent static tables from the rebuilt
cell: ``Structure_data.build`` (minimum-image box), ``AOs_sphe_pgto_data.build``
(the pGTO lattice-image set) and ``build_ewald_data`` (eta, real-space images,
G set, ion-ion energy).  ``_detile_aos_data_for_save`` is called once on the
loaded, unstrained leaf only to recover the geometry-free molecular basis
(exponents, coefficients, angular labels); it is not called on a strained
system.
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

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._setting import ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT
from jqmc.atomic_orbital import AOs_sphe_pgto_data
from jqmc.coulomb_potential import Coulomb_potential_data, build_ewald_data, compute_ecp_non_local_parts_nearest_neighbors
from jqmc.derivatives import _auto
from jqmc.determinant import SD_data
from jqmc.hamiltonians import Hamiltonian_data, _detile_aos_data_for_save, compute_local_energy
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.jqmc_mcmc import MCMC
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data, evaluate_ln_wavefunction
from tests._ecp_core_electrons import core_electrons_of
from tests.test_rs_kernel.conftest import assert_native_ran, assert_no_native_ran, native_ctx_spy
from tests.test_rs_kernel.test_gfmc_projection import _backend_env

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]

_REPO = next(path for path in Path(__file__).resolve().parents if (path / "tests" / "trexio_example_files").is_dir())
_FIXTURE = _REPO / "tests" / "trexio_example_files" / "lithium.h5"

#: Fixed non-symmetric deformation applied to the committed cubic bcc Li cell
#: (and its fractional nuclear positions) to make the reference cell
#: triclinic: all three cell angles differ from 90 degrees.
_TRICLINIC = np.eye(3) + np.array([[0.04, 0.09, -0.03], [0.02, -0.05, 0.07], [-0.06, 0.03, 0.08]])

#: Central-difference step in strain.  The measured error of the central
#: stencil against the analytic tensor on this fixture scales as h**2 with no
#: noise floor down to 1e-5: max 4.5e-4 (h=1e-3), 4.5e-6 (h=1e-4),
#: 4.5e-8 (h=1e-5) for |sigma| ~ 30 Ha, so truncation at 1e-5 is more than
#: two orders below ATOL_FD_VS_ANALYTIC_TIGHT and round-off is ~1e-11.
_STEP = 1.0e-5


@lru_cache(maxsize=1)
def _raw_inputs():
    """Geometry-free inputs of the committed ccECP Li Gamma fixture (read once)."""
    structure, aos, mo_up, mo_dn, geminal, coulomb = read_trexio_file(trexio_file=str(_FIXTURE), store_tuple=True)
    basis = _detile_aos_data_for_save(aos)
    cell = np.array([structure.vec_a, structure.vec_b, structure.vec_c]) @ _TRICLINIC.T
    positions = np.asarray(structure.positions) @ _TRICLINIC.T
    return structure, basis, mo_up, mo_dn, geminal, coulomb, cell, positions


def _build_hamiltonian(F):
    """Rebuild the triclinic Li Hamiltonian with cell and nuclei mapped by ``F`` in this file."""
    structure, basis, mo_up, mo_dn, geminal, coulomb, cell, positions = _raw_inputs()
    cell = cell @ F.T
    positions = positions @ F.T
    sd = Structure_data.build(
        positions=positions,
        pbc_flag=True,
        vec_a=cell[0],
        vec_b=cell[1],
        vec_c=cell[2],
        atomic_numbers=structure.atomic_numbers,
        element_symbols=structure.element_symbols,
        atomic_labels=structure.atomic_labels,
    )
    aos = AOs_sphe_pgto_data.build(
        structure_data=sd,
        nucleus_index=basis.nucleus_index,
        num_ao=basis.num_ao,
        num_ao_prim=basis.num_ao_prim,
        orbital_indices=basis.orbital_indices,
        exponents=np.asarray(basis.exponents),
        coefficients=np.asarray(basis.coefficients),
        angular_momentums=basis.angular_momentums,
        magnetic_quantum_numbers=basis.magnetic_quantum_numbers,
    )
    orbitals = [
        MOs_data(num_mo=mo.num_mo, mo_coefficients=np.asarray(mo.mo_coefficients), aos_data=aos) for mo in (mo_up, mo_dn)
    ]
    determinant = SD_data.build(
        num_electron_up=geminal.num_electron_up,
        num_electron_dn=geminal.num_electron_dn,
        mos_up=orbitals[0],
        mos_dn=orbitals[1],
        time_reversal=geminal.time_reversal,
    )
    potential = Coulomb_potential_data(
        structure_data=sd,
        ecp_flag=True,
        z_cores=coulomb.z_cores,
        max_ang_mom_plus_1=coulomb.max_ang_mom_plus_1,
        num_ecps=coulomb.num_ecps,
        ang_moms=coulomb.ang_moms,
        nucleus_index=coulomb.nucleus_index,
        exponents=coulomb.exponents,
        coefficients=coulomb.coefficients,
        powers=coulomb.powers,
    )
    n_electron = geminal.num_electron_up + geminal.num_electron_dn
    potential = potential.replace(ewald_data=build_ewald_data(sd, potential, tol=1e-10, n_electron=n_electron))
    jastrow = Jastrow_data(
        jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=0.9,
            structure_data=sd,
            core_electrons=core_electrons_of(potential, len(positions)),
            jastrow_1b_type="exp",
        ),
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.2, jastrow_2b_type="exp", structure_data=sd
        ),
        jastrow_three_body_data=None,
    )
    wavefunction = Wavefunction_data(jastrow_data=jastrow, geminal_data=determinant)
    return Hamiltonian_data.build(structure_data=sd, coulomb_potential_data=potential, wavefunction_data=wavefunction)


def _strain_fd_reference(r_up, r_dn):
    """Central FD of e_L and ln|Psi| over all nine strain entries; the quadrature rotation stays RT = I."""
    energy = float(compute_local_energy(_build_hamiltonian(np.eye(3)), r_up, r_dn, np.eye(3)))
    hf, pp = np.zeros((3, 3)), np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            values = []
            for sign in (1.0, -1.0):
                F = np.eye(3)
                F[i, j] += sign * _STEP
                ham = _build_hamiltonian(F)
                values.append(
                    (
                        float(compute_local_energy(ham, r_up @ F.T, r_dn @ F.T, np.eye(3))),
                        float(evaluate_ln_wavefunction(ham.wavefunction_data, r_up @ F.T, r_dn @ F.T)),
                    )
                )
            hf[i, j] = (values[0][0] - values[1][0]) / (2.0 * _STEP)
            pp[i, j] = (values[0][1] - values[1][1]) / (2.0 * _STEP)
    return energy, hf, pp


def _assert_fd_inputs(actual, reference, label):
    """Require finite 3x3 tensors and a non-vacuous reference in every entry."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape == (3, 3), label
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)), label
    atol, rtol = ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT
    assert np.all(np.abs(reference) > atol + rtol * np.abs(reference)), f"{label}: vacuous component {reference}"


def _assert_frozen_hf_matches_fd(actual, reference):
    """Compare the frozen-cell energy-strain control at the unchanged FD tolerance."""
    atol, rtol = ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg="frozen/stress_HF")


def _assert_frozen_pp_matches_fd(actual, reference):
    """Compare the frozen-cell log-wavefunction-strain control at the unchanged FD tolerance."""
    atol, rtol = ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg="frozen/stress_PP")


def _assert_fixture_is_nontrivial(ham, r_up, r_dn):
    """Non-orthogonal cell, active J1/J2 and a non-zero non-local ECP term."""
    sd = ham.structure_data
    cell = np.array([sd.vec_a, sd.vec_b, sd.vec_c])
    gram = cell @ cell.T
    assert np.all(np.abs(gram[np.triu_indices(3, 1)]) > 1.0e-2 * np.max(np.abs(np.diag(gram)))), "cell is not triclinic"
    jastrow = ham.wavefunction_data.jastrow_data
    assert jastrow.jastrow_one_body_data is not None and jastrow.jastrow_two_body_data is not None
    potential = ham.coulomb_potential_data
    assert potential.ecp_flag and potential.ewald_data is not None
    non_local = float(compute_ecp_non_local_parts_nearest_neighbors(potential, ham.wavefunction_data, r_up, r_dn, np.eye(3))[3])
    assert abs(non_local) > ATOL_FD_VS_ANALYTIC_TIGHT, f"non-local ECP term is vacuous: {non_local}"


@pytest.mark.parametrize("backend", ["xla", "rust"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
@pytest.mark.skip_mixed_precision(
    reason="FD / numerical quadrature comparison dominated by float32 round-off",
    alternative_guarantee_ref="MATH-DERIV-LOCAL-ENERGY-001",
)
def test_mcmc_stress_buffers_match_independent_strain_fd(backend):
    """Stored per-walker stress_HF, stress_PP and E_L_stress_PP equal an independent strain FD.

    Committed lithium.h5 (bcc Li, ccECP, spherical pGTO, Gamma, one up/down
    electron) mapped into a triclinic cell, with J1 (exp, 0.9) and J2 (exp,
    1.2).  Two walkers from seed 34 take two moves per measurement at Dt=1.0,
    AS=0, with the identity quadrature rotation.  Two successive
    one-measurement production runs keep their sampled positions.  At each
    stored walker position the reference is the central difference (step
    1e-5) of e_L and ln|Psi| over all nine entries of eps, each strained
    system rebuilt from raw inputs through public constructors (module
    docstring), never through the production strain helpers, with the ECP
    quadrature directions held fixed (the rotation stays the identity).  Every entry of
    the reference must be non-zero, and each is compared at
    CMP-FD-AUTODIFF-FULL-001.  XLA and Rust are both required.  Full
    precision only: the FD reference is float32-limited in mixed, the same
    governed exclusion as the existing stress FD anchor.

    Mixed has no FD verdict here. MATH-DERIV-LOCAL-ENERGY-001 supplies
    same-precision ECP derivative-family AD evidence, not these identical
    sampler buffers. MATH-MCMC-SAMPLER-STRESS-AD-REFERENCE-001 additionally
    checks AE/J2 buffers, not this ccECP Li/J1+J2 fixture.
    """
    ham = _build_hamiltonian(np.eye(3))
    atol, rtol = ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT
    with _backend_env("fast" if backend == "rust" else "jax"):
        driver = MCMC(
            hamiltonian_data=ham,
            num_walkers=2,
            mcmc_seed=34,
            num_mcmc_per_measurement=2,
            Dt=1.0,
            epsilon_AS=0.0,
            random_discretized_mesh=False,
            comput_position_deriv=False,
            comput_stress=True,
        )
        with native_ctx_spy() as counts:
            for step in range(2):
                driver.run(num_mcmc_steps=1)
                r_up_carts = np.asarray(driver._MCMC__latest_r_up_carts)
                r_dn_carts = np.asarray(driver._MCMC__latest_r_dn_carts)
                assert r_up_carts.shape == r_dn_carts.shape == (2, 1, 3)
                for walker in range(2):
                    r_up, r_dn = r_up_carts[walker], r_dn_carts[walker]
                    _assert_fixture_is_nontrivial(ham, r_up, r_dn)
                    energy, hf, pp = _strain_fd_reference(r_up, r_dn)
                    for field, reference in (("stress_HF", hf), ("stress_PP", pp), ("E_L_stress_PP", energy * pp)):
                        actual = np.real(np.asarray(getattr(driver, "_MCMC__stored_" + field))[step, walker])
                        label = f"{step}/{walker}/{field}"
                        _assert_fd_inputs(actual, reference, label)
                        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=label)
    if backend == "rust":
        assert_native_ran(counts, "MCMC stress FD reference")
    else:
        assert_no_native_ran(counts, "MCMC stress FD reference")


def _twin_stress(ham, r_up, r_dn):
    """Unjitted AD twin that differentiates through ``Hamiltonian_data._apply_strain``."""
    zero = jnp.zeros((3, 3))
    hf = jax.jacrev(_auto._real_local_energy_strained, argnums=4)(ham, r_up, r_dn, jnp.eye(3), zero)
    pp = jax.jacrev(_auto._local_log_psi_strained, argnums=3)(ham.wavefunction_data, r_up, r_dn, zero)
    return np.asarray(hf), np.asarray(pp)


@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.skip_mixed_precision(
    reason="FD / numerical quadrature comparison dominated by float32 round-off",
    alternative_guarantee_ref="MATH-DERIV-LOCAL-ENERGY-001",
)
def test_independent_strain_fd_rejects_unstrained_jastrow_cell(monkeypatch):
    """The FD reference does not move with ``_apply_strain``: freezing the Jastrow cell fails it.

    Same triclinic ccECP Li J1+J2 fixture and one seeded configuration
    (numpy PCG64 seed 7, electrons 0.8 bohr around the two nuclei).  First the
    unmutated AD twin, which strains through ``Hamiltonian_data._apply_strain``,
    must match the independent FD for both stress_HF and stress_PP.  Then
    ``Jastrow_data._apply_strain`` is replaced by the identity (the J1/J2 cell
    and nuclei stop following the strain), and the same FD comparison must
    fail for both tensors.  This is a red-making mutation, which
    every twin-vs-twin stress oracle survives.  Full precision only, as for
    the main FD node.

    Mixed has no FD verdict here. MATH-DERIV-LOCAL-ENERGY-001 supplies ECP
    derivative-family AD evidence, not this exact fixture or the frozen-cell
    fault control; AE/J2 sampler-buffer AD is separate evidence as above.
    """
    ham = _build_hamiltonian(np.eye(3))
    positions = np.asarray(ham.structure_data.positions)
    rng = np.random.default_rng(7)
    r_up = positions[[0]] + rng.normal(scale=0.8, size=(1, 3))
    r_dn = positions[[1]] + rng.normal(scale=0.8, size=(1, 3))
    _assert_fixture_is_nontrivial(ham, r_up, r_dn)
    _, fd_hf, fd_pp = _strain_fd_reference(r_up, r_dn)
    twin_hf, twin_pp = _twin_stress(ham, r_up, r_dn)
    atol, rtol = ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT
    for label, actual, reference in (
        ("unmutated/stress_HF", twin_hf, fd_hf),
        ("unmutated/stress_PP", twin_pp, fd_pp),
    ):
        _assert_fd_inputs(actual, reference, label)
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=label)

    monkeypatch.setattr(Jastrow_data, "_apply_strain", lambda self, F: self)
    frozen_hf, frozen_pp = _twin_stress(ham, r_up, r_dn)
    _assert_fd_inputs(frozen_hf, fd_hf, "frozen/stress_HF")
    with pytest.raises(AssertionError, match="frozen/stress_HF"):
        _assert_frozen_hf_matches_fd(frozen_hf, fd_hf)
    _assert_fd_inputs(frozen_pp, fd_pp, "frozen/stress_PP")
    with pytest.raises(AssertionError, match="frozen/stress_PP"):
        _assert_frozen_pp_matches_fd(frozen_pp, fd_pp)
