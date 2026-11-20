"""Shared builders for the driver-level non-local l >= 1 ECP rows.

The committed ``Cl2_ecp_ccpvtz_cart.h5`` ccECP carries non-zero non-local s AND p
projector coefficients on both atoms (the H/H2/Li/water ccECP fixtures used by the other
driver rows have an s-only or empty non-local part).  These helpers build the
J1+J2 Hamiltonian, state the fixture precondition from the raw ECP arrays, build the
l >= 1 ablation (every non-local l >= 1 coefficient set to zero, nothing else changed)
and evaluate the per-walker l >= 1 non-local contribution with the eager all-pairs
reference kernel. ``raw_array_nearest_nucleus_nonlocal`` restricts that raw-array
reference to the production nearest-nucleus convention, and the two ``assert_*_raw_arrays``
helpers let a driver row compare its own output (stored MCMC e_L) or the production
non-local ECP energy at its final walkers with it.
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

import numpy as np

from jqmc.coulomb_potential import _compute_ecp_non_local_parts_all_pairs_debug
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._ecp_core_electrons import core_electrons_of

HIGH_L_FIXTURE = "Cl2_ecp_ccpvtz_cart.h5"
_TREXIO = os.path.join(os.path.dirname(__file__), "trexio_example_files")


def high_l_hamiltonian(stem: str = HIGH_L_FIXTURE) -> Hamiltonian_data:
    """J1 (exp) + J2 (pade) Hamiltonian on the committed high-l ECP fixture."""
    structure_data, _, _, _, geminal_mo_data, coulomb_potential_data = read_trexio_file(
        trexio_file=os.path.join(_TREXIO, stem), store_tuple=True
    )
    jastrow = Jastrow_data(
        jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
            jastrow_1b_type="exp",
        ),
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure_data
        ),
        jastrow_three_body_data=None,
    )
    ham = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow, geminal_data=geminal_mo_data),
    )
    ham.sanity_check()
    return ham


def _high_l_nonlocal_mask(coulomb_potential_data) -> np.ndarray:
    """Raw-array mask of the non-local (l < l_max+1 of its atom) terms with l >= 1."""
    ang = np.asarray(coulomb_potential_data.ang_moms, dtype=int)
    nuc = np.asarray(coulomb_potential_data.nucleus_index, dtype=int)
    lmax1 = np.asarray(coulomb_potential_data.max_ang_mom_plus_1, dtype=int)
    return (ang < lmax1[nuc]) & (ang >= 1)


def assert_fixture_has_live_high_l_channel(ham: Hamiltonian_data) -> None:
    """Precondition from the raw ECP arrays: every atom has a non-zero non-local p term."""
    cpd = ham.coulomb_potential_data
    assert cpd.ecp_flag, "high-l row needs an ECP fixture"
    mask = _high_l_nonlocal_mask(cpd)
    coeff = np.asarray(cpd.coefficients, dtype=float)
    nuc = np.asarray(cpd.nucleus_index, dtype=int)
    live_atoms = set(nuc[mask & (coeff != 0.0)].tolist())
    assert live_atoms == set(range(len(cpd.max_ang_mom_plus_1))), (
        f"fixture precondition: every atom needs a non-zero non-local l>=1 coefficient, got atoms {sorted(live_atoms)}"
    )


def ablate_high_l(ham: Hamiltonian_data) -> Hamiltonian_data:
    """Same Hamiltonian with every non-local l >= 1 coefficient set to zero."""
    cpd = ham.coulomb_potential_data
    coeff = np.asarray(cpd.coefficients, dtype=float).copy()
    coeff[_high_l_nonlocal_mask(cpd)] = 0.0
    ablated_cpd = cpd.replace(coefficients=tuple(float(c) for c in coeff))
    ablated = Hamiltonian_data(
        structure_data=ham.structure_data,
        coulomb_potential_data=ablated_cpd,
        wavefunction_data=ham.wavefunction_data,
    )
    ablated.sanity_check()
    return ablated


def high_l_nonlocal_contribution(ham: Hamiltonian_data, r_up: np.ndarray, r_dn: np.ndarray) -> np.ndarray:
    """Per-walker V_NL(full) - V_NL(l>=1 ablated) at the given walkers (identity mesh)."""
    ablated = ablate_high_l(ham)
    out = []
    for ru, rd in zip(np.asarray(r_up, dtype=np.float64), np.asarray(r_dn, dtype=np.float64), strict=True):
        full = _compute_ecp_non_local_parts_all_pairs_debug(ham.coulomb_potential_data, ham.wavefunction_data, ru, rd)[3]
        abl = _compute_ecp_non_local_parts_all_pairs_debug(ablated.coulomb_potential_data, ham.wavefunction_data, ru, rd)[3]
        out.append(float(np.real(full)) - float(np.real(abl)))
    return np.asarray(out)


def raw_array_nearest_nucleus_nonlocal(
    ham: Hamiltonian_data, r_up: np.ndarray, r_dn: np.ndarray, rt: np.ndarray
) -> tuple[float, float]:
    """V_NL with the production nearest-nucleus convention (NN = 1), from the raw ECP arrays.

    Returns ``(V_NL, V_NL(l >= 1))`` for one molecular configuration. The terms come from
    the eager all-pairs reference kernel (terms selected from the raw ``ang_moms`` /
    ``max_ang_mom_plus_1`` arrays, scipy Legendre polynomials, full determinant and Jastrow
    ratios) on the mesh rotated by ``rt``; only the terms of the nucleus nearest to each
    electron are kept, as the production kernels do with ``NN = 1``. The kernel emits, for
    each atom and each ``l < max_ang_mom_plus_1`` of that atom, the up electrons and then
    the down electrons, ``Nv`` mesh points each; the term count is checked against that
    layout.
    """
    from jqmc._setting import Nv_default

    cpd, wf = ham.coulomb_potential_data, ham.wavefunction_data
    assert not ham.structure_data.pbc_flag, "the nearest-nucleus selection below is molecular"
    r_up = np.asarray(r_up, dtype=np.float64)
    r_dn = np.asarray(r_dn, dtype=np.float64)
    terms = np.asarray(_compute_ecp_non_local_parts_all_pairs_debug(cpd, wf, r_up, r_dn, RT=np.asarray(rt))[2])
    positions = np.asarray(ham.structure_data.positions, dtype=np.float64)
    electrons = np.concatenate([r_up, r_dn])
    nearest = np.argmin(np.linalg.norm(electrons[:, None, :] - positions[None, :, :], axis=2), axis=1)
    keep, high_l = [], []
    for atom, lmax1 in enumerate(np.asarray(cpd.max_ang_mom_plus_1, dtype=int)):
        for ang_mom in range(lmax1):
            for electron in range(len(electrons)):
                keep.extend([nearest[electron] == atom] * Nv_default)
                high_l.extend([ang_mom >= 1] * Nv_default)
    assert terms.shape == (len(keep),), f"unexpected reference term layout {terms.shape} != ({len(keep)},)"
    keep, high_l = np.asarray(keep), np.asarray(high_l)
    real = np.real(terms)
    return float(np.sum(real[keep])), float(np.sum(real[keep & high_l]))


def _production_nonlocal_nn(ham, r_up, r_dn, rt):
    """Production non-local ECP energy with the drivers' nearest-neighbour convention (NN = 1)."""
    import jax.numpy as jnp

    from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors

    cpd, wf = ham.coulomb_potential_data, ham.wavefunction_data
    value = compute_ecp_non_local_parts_nearest_neighbors(cpd, wf, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(rt))[3]
    return float(np.real(np.asarray(value)))


def _raw_reference(ham, r_up, r_dn, rt, atol, rtol, label):
    """Raw-array V_NL at one walker; its l >= 1 part must exceed the comparison envelope."""
    reference, high_l = raw_array_nearest_nucleus_nonlocal(ham, r_up, r_dn, rt)
    assert np.isfinite(reference) and np.isfinite(high_l), (label, reference, high_l)
    envelope = atol + rtol * abs(reference)
    assert abs(high_l) > envelope, f"{label}: raw-array l>=1 non-local part {high_l} is inside the envelope {envelope}"
    return reference


def assert_production_nonlocal_matches_raw_arrays(ham, r_up_walkers, r_dn_walkers, atol, rtol, label):
    """At every final driver walker, production V_NL (NN = 1, identity mesh) equals the raw-array reference."""
    rt = np.eye(3)
    for walker, (r_up, r_dn) in enumerate(zip(np.asarray(r_up_walkers), np.asarray(r_dn_walkers), strict=True)):
        reference = _raw_reference(ham, r_up, r_dn, rt, atol, rtol, f"{label} walker {walker}")
        value = _production_nonlocal_nn(ham, r_up, r_dn, rt)
        assert np.isfinite(value), (label, walker, value)
        if not np.isclose(value, reference, atol=atol, rtol=rtol, equal_nan=False):
            raise AssertionError(
                f"{label}: production non-local ECP at final walker {walker} misses the raw-array reference: "
                f"{value!r} vs {reference!r}"
            )


def assert_mcmc_stored_e_L_matches_raw_arrays(ham, run, atol, rtol, label):
    """Stored production e_L of the last MCMC step == kinetic + bare + local ECP + raw-array V_NL at each walker.

    The last step's e_L is evaluated at the final walkers with the rotation generated
    from each walker's final key (``_generate_rotation_matrix``, as the driver does).
    """
    import jax.numpy as jnp

    from jqmc.coulomb_potential import compute_bare_coulomb_potential, compute_ecp_local_parts_all_pairs
    from jqmc.jqmc_mcmc import _generate_rotation_matrix
    from jqmc.wavefunction import compute_kinetic_energy

    cpd, wf = ham.coulomb_potential_data, ham.wavefunction_data
    keys = run._MCMC__jax_PRNG_key_list
    stored = np.asarray(run.e_L)[-1]
    walkers = zip(np.asarray(run._MCMC__latest_r_up_carts), np.asarray(run._MCMC__latest_r_dn_carts), strict=True)
    for walker, (r_up, r_dn) in enumerate(walkers):
        rt = np.asarray(_generate_rotation_matrix(keys[walker]))
        up, dn = jnp.asarray(r_up), jnp.asarray(r_dn)
        rest = sum(
            float(np.real(np.asarray(value)))
            for value in (
                compute_kinetic_energy(wf, up, dn),
                compute_bare_coulomb_potential(cpd, up, dn),
                compute_ecp_local_parts_all_pairs(cpd, up, dn),
            )
        )
        reference = rest + _raw_reference(ham, r_up, r_dn, rt, atol, rtol, f"{label} walker {walker}")
        assert np.isfinite(stored[walker]), (label, walker, stored[walker])
        if not np.isclose(stored[walker], reference, atol=atol, rtol=rtol, equal_nan=False):
            raise AssertionError(
                f"{label}: stored e_L at final walker {walker} misses the raw-array reference: "
                f"{stored[walker]!r} vs {reference!r}"
            )
