"""Jastrow component periodicity must match the Hamiltonian structure (issue #147).

Every Jastrow component (J1, J2, J3 orbital basis, DTN) evaluates its
distances with its *own* ``structure_data``: ``pbc_flag`` selects the
molecular or the periodic kernel and ``vec_a|b|c`` the lattice. A periodic
Hamiltonian whose J2 kept the molecular default ``Structure_data()`` of
``Jastrow_two_body_data.init_jastrow_two_body_data`` would evaluate a
non-periodic J2 in memory, while ``load_from_hdf5`` / a checkpoint restart
back-fills the periodic structure, so the reloaded wavefunction would be a
different function. The reverse (a periodic component on a molecular
Hamiltonian) would apply minimum-image distances to a finite system. Both
are rejected.

Independent references: lattice-translation invariance of a Gamma-point
periodic wavefunction, ln Psi(r_i + a_k) = ln Psi(r_i) for every electron i
and lattice vector a_k (exact up to rounding), and identity of the
wavefunction across save/load and checkpoint restart.
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

import jax
import numpy as np
import pytest

from jqmc._checkpoint import merge_rank_checkpoints
from jqmc._precision import get_tolerance_min
from jqmc._setting import get_tol_hdf5_roundtrip
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_DTN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.jqmc_mcmc import MCMC
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data, evaluate_ln_wavefunction

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_TREXIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")
_PBC_FILE = "H2_pbc_ae_cart.h5"  # Gamma point, cubic cell
_MOL_FILE = "H2_ecp_ccpvdz_cart.h5"
_LN_PSI_ZONES = ("ao_eval", "jastrow_eval", "det_eval", "wf_eval")

#: Periodic candidates: (accepted?, builder). An accepted candidate is a
#: consistent input; a rejected one carries a component whose periodicity
#: differs from the periodic Hamiltonian.
_PERIODIC_CANDIDATES = (
    "j2_pade_periodic",
    "j2_exp_periodic",
    "j1_periodic",
    "dtn_periodic",
    "j2_pade_default_structure",
    "j2_exp_default_structure",
    "j1_molecular_structure",
    "dtn_molecular_structure",
)
_PERIODIC_ACCEPTED = {"j2_pade_periodic", "j2_exp_periodic", "j1_periodic", "dtn_periodic"}


def _read(name):
    return read_trexio_file(trexio_file=os.path.join(_TREXIO_DIR, name), store_tuple=True)


def _molecular_copy(structure_data: Structure_data) -> Structure_data:
    """The same nuclei without the lattice (``pbc_flag=False``)."""
    return Structure_data(
        positions=np.asarray(structure_data.positions),
        atomic_numbers=tuple(structure_data.atomic_numbers),
        element_symbols=tuple(structure_data.element_symbols),
        atomic_labels=tuple(structure_data.atomic_labels),
        pbc_flag=False,
    )


def _dtn(structure_data: Structure_data) -> Jastrow_DTN_data:
    dtn = Jastrow_DTN_data.build(structure_data=structure_data, z_eff_per_species=[1.0], pseudized_per_species=[False])
    rng = np.random.default_rng(147)
    return dtn.replace(
        alpha_indep=np.asarray(rng.normal(size=np.shape(dtn.alpha_indep))) * 0.05,
        beta_indep=np.asarray(rng.normal(size=np.shape(dtn.beta_indep))) * 0.05,
    )


def _periodic_jastrow(case: str, structure_data: Structure_data) -> Jastrow_data:
    """The Jastrow of a periodic candidate, built without any sanity_check."""
    molecular = _molecular_copy(structure_data)
    zero_core = tuple(0.0 for _ in structure_data.atomic_numbers)
    if case.startswith("j2_"):
        j2_type = "pade" if "_pade_" in case else "exp"
        j2_param = 0.2 if j2_type == "pade" else 1.0
        kwargs = {"structure_data": structure_data} if case.endswith("_periodic") else {}
        j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=j2_param, jastrow_2b_type=j2_type, **kwargs)
        return Jastrow_data(jastrow_two_body_data=j2)
    if case.startswith("j1_"):
        sd = structure_data if case == "j1_periodic" else molecular
        j1 = Jastrow_one_body_data.init_jastrow_one_body_data(jastrow_1b_param=1.0, structure_data=sd, core_electrons=zero_core)
        return Jastrow_data(jastrow_one_body_data=j1)
    if case.startswith("dtn_"):
        return Jastrow_data(jastrow_dtn_data=_dtn(structure_data if case == "dtn_periodic" else molecular))
    raise AssertionError(case)


def _periodic_hamiltonian_unchecked(case: str):
    structure_data, _aos, _mos_up, _mos_dn, geminal_data, coulomb_data = _read(_PBC_FILE)
    jastrow_data = _periodic_jastrow(case, structure_data)
    return Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_data,
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data),
    )


def _build(ham: Hamiltonian_data) -> Hamiltonian_data:
    return Hamiltonian_data.build(
        structure_data=ham.structure_data,
        coulomb_potential_data=ham.coulomb_potential_data,
        wavefunction_data=ham.wavefunction_data,
    )


def _cell(structure_data: Structure_data) -> np.ndarray:
    return np.asarray([structure_data.vec_a, structure_data.vec_b, structure_data.vec_c], dtype=np.float64)


def _configurations(structure_data: Structure_data, count: int = 4):
    rng = np.random.default_rng(7)
    cell = _cell(structure_data)
    out = []
    for _ in range(count):
        r = rng.random((2, 3)) @ cell
        out.append((r[:1].copy(), r[1:].copy()))
    return out


def _ln_psi(ham: Hamiltonian_data, r_up, r_dn) -> float:
    return float(evaluate_ln_wavefunction(ham.wavefunction_data, r_up, r_dn))


def _assert_rejected(ham: Hamiltonian_data, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _build(ham)
    with pytest.raises(ValueError, match=match):
        ham.sanity_check()


@pytest.mark.light
@pytest.mark.parametrize("case", _PERIODIC_CANDIDATES)
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="seeded-random",
    reason="PCG64 seed 7, four uniform fractional positions in the H2_pbc_ae_cart cell for each spin.",
)
def test_accepted_periodic_hamiltonian_jastrow_is_lattice_periodic_and_reload_invariant(tmp_path, case):
    """Every periodic Hamiltonian that build accepts is lattice periodic in memory and survives save/load.

    For each candidate (consistent periodic J1 / Pade J2 / exp J2 / DTN, and
    the inconsistent J2 with the default molecular structure, J1 and DTN on a
    molecular copy of the nuclei) on the Gamma-point H2_pbc_ae_cart.h5
    Hamiltonian, Hamiltonian_data.build either rejects the input with
    ValueError, or the built Hamiltonian satisfies ln Psi(r_i + a_k) = ln Psi(r_i)
    for both electrons and all three lattice vectors at four seeded points, in
    memory and after save_to_hdf5/load_from_hdf5, and the reloaded ln Psi equals
    the in-memory one. The consistent candidates must be accepted. An accepted
    default-structure Pade J2 would show a translation error of 0.477 and a
    reload change of 0.307.
    """
    ham = _periodic_hamiltonian_unchecked(case)
    try:
        built = _build(ham)
    except ValueError:
        assert case not in _PERIODIC_ACCEPTED, f"{case}: a consistent periodic Hamiltonian was rejected"
        return
    path = str(tmp_path / "ham.h5")
    built.save_to_hdf5(path)
    reloaded = Hamiltonian_data.load_from_hdf5(path)

    atol, rtol = get_tolerance_min(_LN_PSI_ZONES, "strict")
    rt_atol, rt_rtol = get_tol_hdf5_roundtrip("strict")
    cell = _cell(built.structure_data)
    for r_up, r_dn in _configurations(built.structure_data):
        base = _ln_psi(built, r_up, r_dn)
        assert np.isfinite(base), case
        np.testing.assert_allclose(_ln_psi(reloaded, r_up, r_dn), base, atol=rt_atol, rtol=rt_rtol, err_msg=f"{case}: reload")
        for k in range(3):
            for label, (up, dn) in (("up", (r_up + cell[k], r_dn)), ("dn", (r_up, r_dn + cell[k]))):
                for name, h in (("memory", built), ("reloaded", reloaded)):
                    np.testing.assert_allclose(
                        _ln_psi(h, up, dn),
                        _ln_psi(h, r_up, r_dn),
                        atol=atol,
                        rtol=rtol,
                        err_msg=f"{case}: {name} ln Psi not invariant under a_{k} on the {label} electron",
                    )


@pytest.mark.light
@pytest.mark.parametrize(
    "case",
    [
        "pbc_j2_default_structure",
        "pbc_j1_molecular_structure",
        "pbc_dtn_molecular_structure",
        "pbc_j3_molecular_basis",
        "pbc_j2_other_cell",
        "mol_j2_periodic",
        "mol_j1_periodic",
        "mol_dtn_periodic",
    ],
)
def test_hamiltonian_rejects_jastrow_component_with_mismatched_periodicity(case):
    """Hamiltonian_data.build and sanity_check reject a Jastrow component whose periodicity or lattice differs from the Hamiltonian.

    Periodic H2_pbc_ae_cart.h5 Hamiltonian with: the default molecular J2
    structure, J1 or DTN on a molecular copy of the nuclei, an AO J3 on the
    molecular H2_ecp_ccpvdz_cart basis, or a periodic J2 whose cell is scaled
    by 1.1. Molecular H2_ecp_ccpvdz_cart.h5 Hamiltonian with a J2, J1 or DTN
    carrying the H2_pbc cell. Each must raise ValueError from
    Hamiltonian_data.build and from sanity_check of the directly constructed
    Hamiltonian.
    """
    pbc_sd, _a, _mu, _md, pbc_gem, pbc_coul = _read(_PBC_FILE)
    mol_sd, mol_aos, _mu2, _md2, mol_gem, mol_coul = _read(_MOL_FILE)
    zero_core_pbc = tuple(0.0 for _ in pbc_sd.atomic_numbers)
    if case.startswith("pbc_"):
        sd, gem, coul = pbc_sd, pbc_gem, pbc_coul
        if case == "pbc_j2_default_structure":
            jas = _periodic_jastrow("j2_pade_default_structure", pbc_sd)
            match = "jastrow_two_body_data.*pass structure_data"
        elif case == "pbc_j1_molecular_structure":
            jas = _periodic_jastrow("j1_molecular_structure", pbc_sd)
            match = "jastrow_one_body_data.structure_data.pbc_flag=False"
        elif case == "pbc_dtn_molecular_structure":
            jas = _periodic_jastrow("dtn_molecular_structure", pbc_sd)
            match = "jastrow_dtn_data.structure_data.pbc_flag=False"
        elif case == "pbc_j3_molecular_basis":
            j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=mol_aos)
            jas = Jastrow_data(jastrow_three_body_data=j3)
            match = "jastrow_three_body_data.orb_data.structure_data.pbc_flag=False"
        else:
            cell = 1.1 * _cell(pbc_sd)
            other = pbc_sd.replace(vec_a=cell[0], vec_b=cell[1], vec_c=cell[2])
            j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.2, structure_data=other)
            jas = Jastrow_data(jastrow_two_body_data=j2)
            match = "lattice vectors"
    else:
        sd, gem, coul = mol_sd, mol_gem, mol_coul
        if case == "mol_j2_periodic":
            j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.2, structure_data=pbc_sd)
            jas = Jastrow_data(jastrow_two_body_data=j2)
            match = "jastrow_two_body_data.structure_data.pbc_flag=True"
        elif case == "mol_j1_periodic":
            j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
                jastrow_1b_param=1.0, structure_data=pbc_sd, core_electrons=zero_core_pbc
            )
            jas = Jastrow_data(jastrow_one_body_data=j1)
            match = "jastrow_one_body_data.structure_data.pbc_flag=True"
        else:
            jas = Jastrow_data(jastrow_dtn_data=_dtn(pbc_sd))
            match = "jastrow_dtn_data.structure_data.pbc_flag=True"
    ham = Hamiltonian_data(
        structure_data=sd,
        coulomb_potential_data=coul,
        wavefunction_data=Wavefunction_data(jastrow_data=jas, geminal_data=gem),
    )
    _assert_rejected(ham, match)


@pytest.mark.light
def test_jastrow_data_rejects_components_with_mismatched_periodicity():
    """Jastrow_data.build rejects populated components that disagree on periodicity or lattice.

    A periodic J1 (H2_pbc_ae_cart) combined with a J2 or a DTN on a molecular
    copy of the same nuclei, or with a periodic J2 on a 1.1-scaled cell, raises
    ValueError. The empty default J2 structure is left to the Hamiltonian-level
    check (the legacy-checkpoint back-fill resolves it on load), so periodic J1
    plus default J2 is accepted by Jastrow_data.build.
    """
    pbc_sd = _read(_PBC_FILE)[0]
    molecular = _molecular_copy(pbc_sd)
    zero_core = tuple(0.0 for _ in pbc_sd.atomic_numbers)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(jastrow_1b_param=1.0, structure_data=pbc_sd, core_electrons=zero_core)
    cell = 1.1 * _cell(pbc_sd)
    other = pbc_sd.replace(vec_a=cell[0], vec_b=cell[1], vec_c=cell[2])
    for kwargs, match in (
        ({"jastrow_two_body_data": Jastrow_two_body_data.init_jastrow_two_body_data(structure_data=molecular)}, "pbc_flag"),
        ({"jastrow_dtn_data": _dtn(molecular)}, "pbc_flag"),
        ({"jastrow_two_body_data": Jastrow_two_body_data.init_jastrow_two_body_data(structure_data=other)}, "lattice vectors"),
    ):
        with pytest.raises(ValueError, match=match):
            Jastrow_data.build(jastrow_one_body_data=j1, **kwargs)
    accepted = Jastrow_data.build(
        jastrow_one_body_data=j1, jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data()
    )
    assert accepted.jastrow_two_body_data.structure_data.pbc_flag is False


@pytest.mark.light
@pytest.mark.parametrize(
    "case",
    ["mo_control", "mo_other_cell_second_child", "mo_molecular_second_child"],
)
def test_composite_j3_basis_child_with_mismatched_periodicity_is_rejected(case):
    """Reject mismatched composite children through public and recursive checks.

    Public Jastrow/Hamiltonian builders identify the mismatched Jastrow child
    before validating its composite. Require that child-path rejection, with
    valid MO dimensions, on all three public routes. Independently call
    check_structure_consistency to detect a regression that inspects only the
    forwarded first-child structure; that check must name children[1]. The
    matching-cell control is accepted by every route.
    """
    from tests.canonical_wavefunction_fixtures import _mos, _orbital

    sd, _aos, _mu, _md, gem, coul = _read(_PBC_FILE)
    cell = 1.1 * _cell(sd)
    other = sd.replace(vec_a=cell[0], vec_b=cell[1], vec_c=cell[2])
    first = _orbital("pgto-cartesian-real", sd)
    if case == "mo_control":
        second = _orbital("plane-wave-real", sd)
    elif case == "mo_molecular_second_child":
        second = _orbital("gto-cartesian-real", _molecular_copy(sd))
    else:
        second = _orbital("plane-wave-real", other)
    # Build the MO matrix on a valid same-cell composite of the same size,
    # then replace only the candidate child. A molecular GTO has fewer rows
    # than the plane-wave fixture; its negative control must not fail shape.
    matching_second = (
        _orbital("pgto-cartesian-real", sd) if case == "mo_molecular_second_child" else _orbital("plane-wave-real", sd)
    )
    orb_data = _mos(first + matching_second, 953).replace(aos_data=first + second)
    assert orb_data.mo_coefficients.shape == (orb_data.num_mo, orb_data.aos_data.num_ao)
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=orb_data)
    zero_core = tuple(0.0 for _ in sd.atomic_numbers)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(jastrow_1b_param=1.0, structure_data=sd, core_electrons=zero_core)
    ham = Hamiltonian_data(
        structure_data=sd,
        coulomb_potential_data=coul,
        wavefunction_data=Wavefunction_data(jastrow_data=Jastrow_data(jastrow_three_body_data=j3), geminal_data=gem),
    )
    jas = ham.wavefunction_data.jastrow_data
    if case == "mo_control":
        Jastrow_data.build(jastrow_one_body_data=j1, jastrow_three_body_data=j3)
        _build(ham)
        ham.sanity_check()
        jas.check_structure_consistency(sd)
        return
    with pytest.raises(ValueError, match=r"jastrow_three_body_data\.orb_data.*children\[1\]"):
        jas.check_structure_consistency(sd)
    # The lower-level MO/composite validator remains independently effective.
    with pytest.raises(ValueError, match="AOs_composite_data children must share one molecular structure_data"):
        orb_data.sanity_check()
    match = r"jastrow_three_body_data\.orb_data.*children\[1\]"
    with pytest.raises(ValueError, match=match):
        Jastrow_data.build(jastrow_one_body_data=j1, jastrow_three_body_data=j3)
    _assert_rejected(ham, match)


def mcmc_restart_ln_psi(case: str, tmp_path):
    """Run 5 MCMC steps on a periodic candidate, checkpoint, restart; return (in-memory, restarted) ln Psi per walker.

    Returns ``None`` when the candidate is rejected (``ValueError``) by
    ``Hamiltonian_data`` / ``MCMC`` construction.
    """
    ham = _periodic_hamiltonian_unchecked(case)
    try:
        mcmc = MCMC(hamiltonian_data=ham, mcmc_seed=12345, num_walkers=4, num_mcmc_per_measurement=8, Dt=2.0, epsilon_AS=0.0)
    except ValueError:
        return None
    mcmc.run(num_mcmc_steps=5)
    td = str(tmp_path)
    mcmc.save_to_hdf5(os.path.join(td, "._restart_rank0.h5"))
    merge_rank_checkpoints(
        output_path=os.path.join(td, "restart.h5"),
        mpi_size=1,
        driver_type="MCMC",
        hamiltonian_data=ham,
        tmp_pattern=os.path.join(td, "._restart_rank{rank}.h5"),
    )
    restarted = MCMC.load_from_hdf5(os.path.join(td, "restart.h5"), rank=0)
    r_up = np.asarray(mcmc._MCMC__latest_r_up_carts)
    r_dn = np.asarray(mcmc._MCMC__latest_r_dn_carts)
    before = np.asarray([_ln_psi(ham, r_up[w], r_dn[w]) for w in range(r_up.shape[0])])
    after = np.asarray([_ln_psi(restarted.hamiltonian_data, r_up[w], r_dn[w]) for w in range(r_up.shape[0])])
    return before, after


@pytest.mark.heavy
@pytest.mark.parametrize("case", ["j2_pade_periodic", "j2_pade_default_structure", "j1_molecular_structure"])
def test_mcmc_checkpoint_restart_keeps_periodic_jastrow_or_rejects(tmp_path, case):
    """An MCMC run on a periodic Hamiltonian restarts with the same ln Psi, or is rejected up front.

    MCMC on H2_pbc_ae_cart.h5 (4 walkers, seed 12345, 5 steps) with a periodic
    Pade J2 must restart from its merged checkpoint with ln Psi equal to the
    in-memory value at the latest walker positions (HDF5 round-trip tolerance).
    The Pade J2 with the default molecular structure and the J1 on molecular
    nuclei must either be rejected by MCMC construction (ValueError) or restart
    with the same ln Psi.
    """
    result = mcmc_restart_ln_psi(case, tmp_path)
    if result is None:
        assert case not in _PERIODIC_ACCEPTED, f"{case}: a consistent periodic Hamiltonian was rejected"
        return
    before, after = result
    assert np.all(np.isfinite(before)), case
    atol, rtol = get_tol_hdf5_roundtrip("strict")
    np.testing.assert_allclose(after, before, atol=atol, rtol=rtol, err_msg=f"{case}: restarted ln Psi")
