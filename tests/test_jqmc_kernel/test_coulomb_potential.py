"""Unit tests for ``jqmc.coulomb_potential``.

Organised into four sections:

  (1) Function equivalence -- production vs ``_debug`` oracle, and
      variant-vs-reference (``fast`` vs default LU, streaming vs
      reference state path, ...). The bulk of the file.

  (2) Physical / numerical invariants -- properties that hold
      independently of the implementation (Madelung constant,
      Ewald eta-invariance, self-energy on single-electron limits,
      ...). Populated when the Ewald tests are merged in.

  (3) Error paths / preconditions -- public-API rejections that
      protect downstream assumptions (e.g. ``build_ewald_data``
      rejecting molecular inputs, ``compute_ecp_non_local_parts_all_pairs``
      raising in PBC). Populated by the Ewald and ECP-PBC merges.

  (4) Dispatch shim routing -- ``compute_coulomb_potential_*_dispatch``
      verifies the static ``pbc_flag`` branch selects the correct
      backend (Ewald vs bare). Populated by the Ewald merge.
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
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
from jax import numpy as jnp

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance, get_tolerance_min
from jqmc._setting import (
    AO_IMAGE_TRUNCATION_ATOL,
    ATOL_BIT_EXACT,
    ATOL_EWALD_VS_EXTERNAL_REFERENCE,
    ATOL_PYTHON_FP64_TIGHT,
    RTOL_EWALD_VS_PYMATGEN_CHARGED_SLAB,
    RTOL_EWALD_VS_EXTERNAL_REFERENCE,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_PYTHON_FP64_ZERO,
)
from jqmc.atomic_orbital import AOs_sphe_gto_data, _apply_image_expansion
from jqmc.coulomb_potential import (
    Coulomb_potential_data,
    Ewald_data,
    _EWALD_G_BUDGET,
    _compute_bare_coulomb_potential_debug,
    _compute_bare_coulomb_potential_el_ion_element_wise_debug,
    _compute_coulomb_potential_fast_debug,
    _compute_discretized_bare_coulomb_potential_el_ion_element_wise_debug,
    _compute_ecp_coulomb_potential_fast_debug,
    _compute_ecp_local_parts_all_pairs_debug,
    _compute_ecp_non_local_parts_all_pairs_debug,
    _compute_ecp_non_local_parts_nearest_neighbors_debug,
    _compute_ewald_potential_ee_debug,
    _compute_ewald_potential_ei_debug,
    _compute_ewald_potential_ei_discretized_debug,
    _compute_ewald_potential_ii_debug,
    _eta_iteration_heuristic,
    build_ewald_data,
    compute_bare_coulomb_potential,
    compute_bare_coulomb_potential_el_el,
    compute_bare_coulomb_potential_el_ion_element_wise,
    compute_bare_coulomb_potential_ion_ion,
    compute_coulomb_potential,
    compute_coulomb_potential_ee_dispatch,
    compute_coulomb_potential_ei_discretized_dispatch,
    compute_coulomb_potential_ei_dispatch,
    compute_coulomb_potential_fast,
    compute_coulomb_potential_ii_dispatch,
    compute_discretized_bare_coulomb_potential_el_ion_element_wise,
    compute_ecp_coulomb_potential,
    compute_ecp_coulomb_potential_fast,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors,
    compute_ecp_non_local_parts_nearest_neighbors_fast_update,
    compute_ewald_potential_ee,
    compute_ewald_potential_ei,
    compute_ewald_potential_ei_discretized,
    compute_ewald_potential_ii,
    icosahedron_sym_mesh_Nv12,
    octahedron_sym_mesh_Nv6,
    octahedron_sym_mesh_Nv18,
    tetrahedron_sym_mesh_Nv4,
)
from jqmc.determinant import (
    Geminal_data,
    _init_det_ratio_streaming_state,
    compute_det_geminal_all_elements,
    compute_geminal_all_elements,
)
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests.conftest import make_pbc_structure

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# params
angle_values = [
    (0.0, 0.0, 0.0),
    (+1.0 / 18.0 * np.pi, -1.0 / 12.0 * np.pi, +1.0 / 95.0 * np.pi),
    (+1.0 / 12.0 * np.pi, -1.0 / 17.0 * np.pi, +1.0 / 15.0 * np.pi),
    (+1.0 / 22.0 * np.pi, +1.0 / 54.0 * np.pi, +1.0 / 75.0 * np.pi),
    (-1.0 / 56.0 * np.pi, -1.0 / 13.0 * np.pi, +1.0 / 25.0 * np.pi),
    (+np.pi, +np.pi, +np.pi),
    (-np.pi, -np.pi, -np.pi),
    (+2.0 * np.pi, +2.0 * np.pi, +2.0 * np.pi),
    (-2.0 * np.pi, -2.0 * np.pi, -2.0 * np.pi),
]

# angle parameters
angle_params = [
    pytest.param(alpha, beta, gamma, id=f"(alpha,beta,gamma)=({alpha:.2f},{beta:.2f},{gamma:.2f})")
    for alpha, beta, gamma in angle_values
]

# Nv parameters
Nv_params = [pytest.param(Nv, id=f"Nv={Nv}") for Nv in (4, 6, 12, 18)]


@pytest.mark.light
@pytest.mark.parametrize(
    "mesh",
    [tetrahedron_sym_mesh_Nv4, octahedron_sym_mesh_Nv6, icosahedron_sym_mesh_Nv12, octahedron_sym_mesh_Nv18],
    ids=["Nv=4", "Nv=6", "Nv=12", "Nv=18"],
)
def test_spherical_quadrature_weights_sum_to_one(mesh):
    """Spherical quadrature weights sum to one."""
    np.testing.assert_allclose(
        np.sum(mesh.weights),
        1.0,
        rtol=RTOL_PYTHON_FP64_ZERO,
        atol=ATOL_PYTHON_FP64_TIGHT,
    )


# --- Ewald-related fixtures and references ---------------------------------

#: Madelung constant for NaCl-type rocksalt (literature; Borwein-Glasser-McPhedran).
#: Standard reference value, accurate to at least 10 significant digits.
ALPHA_M_NACL = 1.7475645946

#: Periodic TREXIO fixture name used by the silicon Ewald cross-checks.
PERIODIC_FIXTURE = "silicon.h5"
PERIODIC_FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "trexio_example_files",
    PERIODIC_FIXTURE,
)


def _make_nacl_sc_fixture() -> tuple[Structure_data, Coulomb_potential_data]:
    """8-atom NaCl-type SC fixture (Z = +-1, nearest-neighbor distance = 1 Bohr).

    Eight atoms on the corners and face centers of a 2-Bohr cubic cell,
    alternating charge: positive at the corners, negative at the edge
    midpoints. Total charge zero (Madelung sum convergent under
    tin-foil BC). Nearest-neighbor distance is 1 Bohr exactly, so the
    Madelung energy per ion is ``E = -alpha_M * Z^2 / R_NN = -alpha_M``
    and the total cell energy (8 ions) is ``(N / 2) E_per_ion = -4 alpha_M``.

    ``Coulomb_potential_data`` has ``ecp_flag=False`` and uses
    ``atomic_numbers`` directly as charges. To get unit charges with the
    right signs, encode ``Z = +1`` as ``atomic_number = 1`` (H) and
    ``Z = -1`` as ``atomic_number = -1`` --- ``_effective_charges`` then
    returns ``atomic_numbers`` verbatim.
    """
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    structure = Structure_data.build(
        pbc_flag=True,
        positions=positions,
        vec_a=(2.0, 0.0, 0.0),
        vec_b=(0.0, 2.0, 0.0),
        vec_c=(0.0, 0.0, 2.0),
        atomic_numbers=(1, 1, 1, 1, -1, -1, -1, -1),
        element_symbols=("X",) * 8,
        atomic_labels=tuple(f"X{i}" for i in range(8)),
    )
    coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
    return structure, coulomb


def _make_h2_box_fixture() -> tuple[Structure_data, Coulomb_potential_data, np.ndarray, np.ndarray]:
    """All-electron H2 singlet in a 10 Bohr cubic box.

    Two protons at +-0.7 Bohr along z (total ``Z = +2``) and one up-spin
    + one down-spin electron (total ``q = -2``); charge-neutral, which
    is the precondition under which the Ewald sum omits the uniform
    background term.
    """
    positions = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=np.float64)
    structure = Structure_data.build(
        pbc_flag=True,
        positions=positions,
        vec_a=(10.0, 0.0, 0.0),
        vec_b=(0.0, 10.0, 0.0),
        vec_c=(0.0, 0.0, 10.0),
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H1", "H2"),
    )
    coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
    r_up = np.array([[0.1, 0.2, -0.6]], dtype=np.float64)
    r_dn = np.array([[-0.1, -0.2, 0.5]], dtype=np.float64)
    return structure, coulomb, r_up, r_dn


def _make_h2_monoclinic_fixture() -> tuple[Structure_data, Coulomb_potential_data, np.ndarray, np.ndarray]:
    """All-electron H2 singlet in a monoclinic (non-orthorhombic) box.

    Same neutral 2-proton (``Z = +1``) + 1-up + 1-dn electron content as
    :py:func:`_make_h2_box_fixture`, but in a true monoclinic cell
    (``a = 10``, ``b = 11``, ``c = 12`` Bohr, ``beta = 110`` deg, b-unique so
    ``vec_c`` carries both an x- and a z-component). Where the cubic h2-box
    leaves the cell-shape handling untested, this fixture forces the e-i / e-e
    Ewald path through the non-orthogonal lattice everywhere it matters --
    :py:func:`_minimum_image_displacement`, the reciprocal ``g_vectors``, the
    real-image enumeration, and the ``eta`` heuristic -- mirroring what the CuO
    tenorite fixture does for the ion-ion channel. Atom and electron positions
    are defined in fractional coordinates and mapped through the lattice so
    both stay inside the cell.
    """
    a_bohr, b_bohr, c_bohr = 10.0, 11.0, 12.0
    beta_rad = float(np.deg2rad(110.0))
    vec_a = (a_bohr, 0.0, 0.0)
    vec_b = (0.0, b_bohr, 0.0)
    vec_c = (c_bohr * float(np.cos(beta_rad)), 0.0, c_bohr * float(np.sin(beta_rad)))
    lattice_matrix = np.array([vec_a, vec_b, vec_c], dtype=np.float64)

    ion_frac = np.array([[0.40, 0.40, 0.40], [0.60, 0.55, 0.55]], dtype=np.float64)
    elec_frac = np.array([[0.25, 0.30, 0.30], [0.70, 0.65, 0.70]], dtype=np.float64)  # r_up, r_dn
    positions = ion_frac @ lattice_matrix
    structure = Structure_data.build(
        pbc_flag=True,
        positions=positions,
        vec_a=vec_a,
        vec_b=vec_b,
        vec_c=vec_c,
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H1", "H2"),
    )
    coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
    r_up = (elec_frac[:1] @ lattice_matrix).astype(np.float64)
    r_dn = (elec_frac[1:] @ lattice_matrix).astype(np.float64)
    return structure, coulomb, r_up, r_dn


def _make_silicon_ewald_fixture() -> tuple[Structure_data, Coulomb_potential_data, np.ndarray, np.ndarray]:
    """Silicon TREXIO fixture with deterministic synthetic electrons inside the cell.

    Skips the calling test if the periodic TREXIO file is not present.
    Exercises the ECP ``_effective_charges`` path (``Z_eff = Z - z_core``).
    """
    if not os.path.exists(PERIODIC_FIXTURE_PATH):
        pytest.skip(f"Periodic TREXIO fixture {PERIODIC_FIXTURE} is not present.")
    structure, _aos, _mos_up, _mos_dn, _geminal, coulomb = read_trexio_file(trexio_file=PERIODIC_FIXTURE_PATH, store_tuple=True)
    cell = np.asarray(structure.cell, dtype=np.float64)
    rng = np.random.default_rng(0)
    n_pair = 3
    frac = rng.uniform(0.1, 0.9, size=(2 * n_pair, 3))
    r_carts = frac @ cell
    return structure, coulomb, r_carts[:n_pair], r_carts[n_pair:]


def _build_full_jastrow_data(structure_data, geminal_mo_data, coulomb_potential_data, rng: np.random.Generator) -> Jastrow_data:
    core_electrons = tuple(coulomb_potential_data.z_cores)

    jastrow_one_body_data = Jastrow_one_body_data(
        jastrow_1b_param=1.0,
        jastrow_1b_type="exp",
        structure_data=structure_data,
        core_electrons=core_electrons,
    )

    jastrow_two_body_data = Jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp")

    num_orb = geminal_mo_data.orb_data_up_spin._num_orb
    j_matrix = rng.random((num_orb, num_orb + 1))
    j_matrix[:, :-1] = 0.5 * (j_matrix[:, :-1] + j_matrix[:, :-1].T)  # symmetric J3 (bosonic)
    jastrow_three_body_data = Jastrow_three_body_data(
        orb_data=geminal_mo_data.orb_data_up_spin,
        j_matrix=j_matrix,
    )

    return Jastrow_data(
        jastrow_one_body_data=jastrow_one_body_data,
        jastrow_two_body_data=jastrow_two_body_data,
        jastrow_three_body_data=jastrow_three_body_data,
    )


_FAST_VS_REFERENCE_PARAMS = [
    pytest.param(
        "water_ccecp_ccpvqz.h5",
        Nv,
        a,
        b,
        c,
        id=f"water-Nv={Nv}-(a,b,g)=({a:.2f},{b:.2f},{c:.2f})",
    )
    for Nv in (4, 6, 12, 18)
    for (a, b, c) in angle_values
] + [
    # Silicon PBC: one (Nv, angle) smoke point covers the PBC top-level path.
    pytest.param("silicon.h5", 6, 0.0, 0.0, 0.0, id="silicon-Nv=6-(a,b,g)=(0,0,0)"),
]


# ---------------------------------------------------------------------------
# (2) Physical / numerical invariants.
# ---------------------------------------------------------------------------


def _build_nonsingular_agp(structure_data, atom_positions):
    """AGP-AO geminal with rank >= n_electron, kept away from the nodal surface.

    Each atom carries three s-shells with distinct exponents so the molecular
    AO space has rank ``3 * natom >= 2`` (the electron count), giving a
    determinant that is ``O(1e-4)`` rather than ``O(1e-18)``. A near-singular
    geminal (one s-shell per atom -> ``det ~ 0``) makes the ``Psi'/Psi`` ECP
    ratio blow up and destroys the molecular-vs-PBC agreement this oracle
    relies on (see the projector-unwrap PoC).

    The same molecular ``mo_coefficients`` / ``lambda_matrix`` are reused for
    the molecular and PBC structures, so the only difference between the two
    runs is the AO image expansion and the nearest-image nucleus selection.
    """
    natom = len(atom_positions)
    shell_exponents = (0.3, 0.8, 1.8)
    n_shell = len(shell_exponents)
    num_ao = natom * n_shell
    nucleus_index = []
    exps = []
    for i_atom in range(natom):
        for e in shell_exponents:
            nucleus_index.append(i_atom)
            exps.append(e)
    aos_mol = AOs_sphe_gto_data._from_full_basis(
        structure_data=structure_data,
        nucleus_index=tuple(nucleus_index),
        num_ao=num_ao,
        num_ao_prim=num_ao,
        orbital_indices=tuple(range(num_ao)),
        exponents=np.array(exps, dtype=np.float64),
        coefficients=np.array([1.0] * num_ao, dtype=np.float64),
        angular_momentums=tuple([0] * num_ao),
        magnetic_quantum_numbers=tuple([0] * num_ao),
    )
    if structure_data.pbc_flag:
        aos = _apply_image_expansion(aos_mol, structure_data, tol=AO_IMAGE_TRUNCATION_ATOL)
    else:
        aos = aos_mol
    rng = np.random.default_rng(1)
    num_mo = num_ao
    mo_coefficients = 0.2 * rng.standard_normal((num_mo, num_ao)) + np.eye(num_mo, num_ao)
    mos = MOs_data(num_mo=num_mo, aos_data=aos, mo_coefficients=mo_coefficients)
    lambda_matrix = 0.1 * rng.standard_normal((num_mo, num_mo)) + np.eye(num_mo)
    return Geminal_data(
        num_electron_up=2,
        num_electron_dn=2,
        orb_data_up_spin=mos,
        orb_data_dn_spin=mos,
        lambda_matrix=lambda_matrix,
    )


def _build_single_atom_ecp(structure_data):
    """Minimal one-atom ECP (local l=2 + non-local l=0, l=1).

    Matches :py:func:`_make_pbc_ecp_fixture`'s synthetic channel layout.
    """
    z_cores = (2.0,)
    max_ang_mom_plus_1 = (2,)
    ang_moms, nucleus_index, exponents, coefficients, powers = [], [], [], [], []
    for l, e, c in ((2, 1.0, 2.0), (0, 1.2, 1.5), (1, 0.8, 1.0)):
        ang_moms.append(l)
        nucleus_index.append(0)
        exponents.append(e)
        coefficients.append(c)
        powers.append(0)
    return Coulomb_potential_data(
        structure_data=structure_data,
        ecp_flag=True,
        z_cores=z_cores,
        max_ang_mom_plus_1=max_ang_mom_plus_1,
        num_ecps=len(ang_moms),
        ang_moms=tuple(ang_moms),
        nucleus_index=tuple(nucleus_index),
        exponents=tuple(exponents),
        coefficients=tuple(coefficients),
        powers=tuple(powers),
    )


def _ecp_non_local_sum_V(structure_data, geminal_data, r_up, r_dn, NN=1):
    """``sum_V`` of the production non-local NN-ECP kernel (no Jastrow)."""
    coulomb = _build_single_atom_ecp(structure_data)
    jastrow = Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None)
    wf = Wavefunction_data(geminal_data=geminal_data, jastrow_data=jastrow)
    _, _, _, sum_V = compute_ecp_non_local_parts_nearest_neighbors(
        coulomb_potential_data=coulomb,
        wavefunction_data=wf,
        r_up_carts=jnp.array(r_up),
        r_dn_carts=jnp.array(r_dn),
        Nv=6,
        NN=NN,
        RT=jnp.eye(3),
    )
    return float(np.asarray(sum_V))


#: computed offline with pymatgen (see
#: :py:func:`test_compute_ewald_potential_ii_vs_pymatgen_nacl_sc` docstring
#: for the regeneration script). Keeping the reference
#: hardcoded means the test environment does not require pymatgen.
PYMATGEN_E_II_NACL_SC_HA: float = -6.9902583738
#: pymatgen version used to compute ``PYMATGEN_E_II_NACL_SC_HA``, recorded
#: alongside the value above so the regeneration is auditable.
PYMATGEN_VERSION_USED_FOR_NACL_SC: str = "2024.6.10"

#: Independent ion-ion Ewald reference for the 8-atom MgO B1 (rocksalt)
#: fixture (``a = 4.211`` Angstrom experimental, ``Z = +-2``). Same B1
#: sublattice as NaCl SC but with doubled charges, so any linear-in-Z
#: bug in the implementation shows up as a wrong scaling between
#: NaCl and MgO references. See
#: :py:func:`test_compute_ewald_potential_ii_vs_pymatgen_mgo_b1` for the
#: regeneration script.
PYMATGEN_E_II_MGO_B1_HA: float = -7.0274717259
#: pymatgen version used to compute ``PYMATGEN_E_II_MGO_B1_HA``, recorded
#: alongside the value above so the regeneration is auditable.
PYMATGEN_VERSION_USED_FOR_MGO_B1: str = "2024.6.10"

#: Independent ion-ion Ewald reference for the 4-atom wurtzite ZnO
#: fixture (hexagonal, ``a = 3.2495`` Angstrom, ``c = 5.2069`` Angstrom,
#: ``u = 0.382``, ``Z = +-2``). The non-orthorhombic primitive cell tests
#: cell-shape handling end-to-end in :py:func:`_enumerate_real_images`,
#: :py:func:`_enumerate_g_vectors`, and :py:func:`_eta_iteration_heuristic`;
#: any cubic-only assumption along that chain would mis-converge here.
#: See
#: :py:func:`test_compute_ewald_potential_ii_vs_pymatgen_zno_wurtzite` for
#: the regeneration script.
PYMATGEN_E_II_ZNO_WURTZITE_HA: float = -3.5144999760
#: pymatgen version used to compute ``PYMATGEN_E_II_ZNO_WURTZITE_HA``,
#: recorded alongside the value above so the regeneration is auditable.
PYMATGEN_VERSION_USED_FOR_ZNO_WURTZITE: str = "2024.6.10"

#: Independent ion-ion Ewald reference for the 8-atom CuO tenorite
#: fixture (monoclinic C2/c, ``a = 4.6837`` Angstrom, ``b = 3.4226`` Angstrom,
#: ``c = 5.1288`` Angstrom, ``beta = 99.54`` deg, ``Z = +-2``). True
#: monoclinic geometry (``beta != 90``) is the most demanding test of the
#: cell-shape handling: ``vec_c`` has both an x- and a z-component, so
#: any place in the chain that assumes orthorhombic axes ``(a,b,c)`` show
#: up here. See
#: :py:func:`test_compute_ewald_potential_ii_vs_pymatgen_cuo_tenorite` for
#: the regeneration script.
PYMATGEN_E_II_CUO_TENORITE_HA: float = -7.0518914869
#: pymatgen version used to compute ``PYMATGEN_E_II_CUO_TENORITE_HA``,
#: recorded alongside the value above so the regeneration is auditable.
PYMATGEN_VERSION_USED_FOR_CUO_TENORITE: str = "2024.6.10"

#: Independent electron-ion / electron-electron Ewald references for the
#: all-electron H2-in-box fixture (:py:func:`_make_h2_box_fixture`; 2 protons
#: ``Z = +1`` plus 1 up + 1 dn electron in a 10 Bohr cubic cell, charge
#: neutral). pymatgen has no electron/ion channels of its own, so the
#: reference is built from ``EwaldSummation.total_energy_matrix`` (the N x N
#: pairwise matrix over ions *and* electrons modelled as ``Z = -1`` point
#: charges): the e-i / e-e / i-i blocks are summed and each gets jqmc's
#: per-channel jellium background added back with pymatgen's own ``eta``
#: (the matrix omits the charged-cell term, which is nonzero per block even
#: though the three cancel for the neutral whole). Because every channel is
#: eta-invariant after the background is restored, the reconstruction matches
#: jqmc despite the two codes picking different ``eta``. See
#: :py:func:`test_compute_ewald_potential_ei_vs_pymatgen_h2_box`
#: and :py:func:`test_compute_ewald_potential_ee_vs_pymatgen_h2_box` for the
#: shared regeneration script.
PYMATGEN_E_EI_H2_BOX_HA: float = -7.8654451453
PYMATGEN_E_EE_H2_BOX_HA: float = 0.2776687088
#: pymatgen version used to compute the H2-box e-i / e-e references, recorded
#: alongside the values above so the regeneration is auditable.
PYMATGEN_VERSION_USED_FOR_H2_BOX: str = "2024.6.10"

#: Independent electron-ion / electron-electron Ewald references for the
#: monoclinic (non-orthorhombic) H2 fixture (:py:func:`_make_h2_monoclinic_fixture`;
#: same neutral 2-proton + 2-electron content as the cubic h2-box but in a
#: ``beta = 110`` deg monoclinic cell). Built the same way as the cubic h2-box
#: references (``EwaldSummation.total_energy_matrix`` block decomposition plus
#: jqmc's per-channel jellium background restored with pymatgen's ``eta``),
#: but here the non-orthogonal lattice exercises the cell-shape handling in the
#: e-i / e-e Ewald path -- the analogue of what the CuO tenorite fixture does
#: for the ion-ion channel. See
#: :py:func:`test_compute_ewald_potential_ei_vs_pymatgen_h2_monoclinic` for the
#: shared regeneration script.
PYMATGEN_E_EI_H2_MONO_HA: float = -0.4511252915
PYMATGEN_E_EE_H2_MONO_HA: float = -0.3248239521
#: pymatgen version used to compute the monoclinic-H2 e-i / e-e references,
#: recorded alongside the values above so the regeneration is auditable.
PYMATGEN_VERSION_USED_FOR_H2_MONO: str = "2024.6.10"


# ---------------------------------------------------------------------------
# (3) Error paths / preconditions.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (4) Dispatch shim routing.
# ---------------------------------------------------------------------------


class TestBareCoulomb:
    """production-vs-debug guardrails for bare Coulomb kernels."""

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5"])
    def test_compute_bare_coulomb_potential_production_vs_debug(self, trexio_file: str):
        """Test the bare coulomb potential computation."""
        atol, rtol = get_tolerance("coulomb", "strict")
        (
            _,
            _,
            _,
            _,
            _,
            coulomb_potential_data,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        # check its input
        coulomb_potential_data.sanity_check()

        r_up_carts_np = np.array(
            [
                [0.64878536, -0.83275288, 0.33532629],
                [0.55271273, 0.72310605, 0.93443775],
                [0.66767275, 0.1206456, -0.36521208],
                [-0.93165236, -0.0120386, 0.33003036],
            ]
        )
        r_dn_carts_np = np.array(
            [
                [1.0347816, 1.26162081, 0.42301735],
                [-0.57843435, 1.03651987, -0.55091542],
                [-1.56091964, -0.58952149, -0.99268141],
                [0.61863233, -0.14903326, 0.51962683],
            ]
        )

        r_up_carts_jnp = jnp.array(r_up_carts_np)
        r_dn_carts_jnp = jnp.array(r_dn_carts_np)

        # bare coulomb
        vpot_bare_jax = compute_bare_coulomb_potential(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts_jnp,
            r_dn_carts=r_dn_carts_jnp,
        )

        vpot_bare_debug = _compute_bare_coulomb_potential_debug(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts_np,
            r_dn_carts=r_dn_carts_np,
        )

        assert not np.any(np.isnan(np.asarray(vpot_bare_jax))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(vpot_bare_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(vpot_bare_jax, vpot_bare_debug, atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5"])
    def test_compute_bare_coulomb_potential_el_ion_element_wise_production_vs_debug(self, trexio_file: str):
        """Test the bare couloumb potential computation."""
        atol, rtol = get_tolerance("coulomb", "strict")
        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        # check its input
        coulomb_potential_data.sanity_check()

        r_up_carts_np = np.array(
            [
                [0.64878536, -0.83275288, 0.33532629],
                [0.55271273, 0.72310605, 0.93443775],
                [0.66767275, 0.1206456, -0.36521208],
                [-0.93165236, -0.0120386, 0.33003036],
            ]
        )
        r_dn_carts_np = np.array(
            [
                [1.0347816, 1.26162081, 0.42301735],
                [-0.57843435, 1.03651987, -0.55091542],
                [-1.56091964, -0.58952149, -0.99268141],
                [0.61863233, -0.14903326, 0.51962683],
            ]
        )

        r_up_carts_jnp = jnp.array(r_up_carts_np)
        r_dn_carts_jnp = jnp.array(r_dn_carts_np)

        interactions_R_r_up_debug, interactions_R_r_dn_debug = _compute_bare_coulomb_potential_el_ion_element_wise_debug(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts_np,
            r_dn_carts=r_dn_carts_np,
        )

        interactions_R_r_up_jax, interactions_R_r_dn_jax = compute_bare_coulomb_potential_el_ion_element_wise(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts_jnp,
            r_dn_carts=r_dn_carts_jnp,
        )

        assert not np.any(np.isnan(np.asarray(interactions_R_r_up_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(interactions_R_r_up_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(interactions_R_r_up_debug, interactions_R_r_up_jax, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(interactions_R_r_dn_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(interactions_R_r_dn_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(interactions_R_r_dn_debug, interactions_R_r_dn_jax, atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5"])
    def test_compute_discretized_bare_coulomb_potential_el_ion_element_wise_production_vs_debug(self, trexio_file: str):
        """Test the bare couloumb potential computation."""
        atol, rtol = get_tolerance("coulomb", "strict")
        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        # check its input
        coulomb_potential_data.sanity_check()

        r_up_carts_np = np.array(
            [
                [0.64878536, -0.83275288, 0.33532629],
                [0.55271273, 0.72310605, 0.93443775],
                [0.66767275, 0.1206456, -0.36521208],
                [-0.93165236, -0.0120386, 0.33003036],
            ]
        )
        r_dn_carts_np = np.array(
            [
                [1.0347816, 1.26162081, 0.42301735],
                [-0.57843435, 1.03651987, -0.55091542],
                [-1.56091964, -0.58952149, -0.99268141],
                [0.61863233, -0.14903326, 0.51962683],
            ]
        )

        r_up_carts_jnp = jnp.array(r_up_carts_np)
        r_dn_carts_jnp = jnp.array(r_dn_carts_np)

        alat = 0.5

        interactions_R_r_up_debug, interactions_R_r_dn_debug = (
            _compute_discretized_bare_coulomb_potential_el_ion_element_wise_debug(
                coulomb_potential_data=coulomb_potential_data,
                r_up_carts=r_up_carts_np,
                r_dn_carts=r_dn_carts_np,
                alat=alat,
            )
        )

        interactions_R_r_up_jax, interactions_R_r_dn_jax = compute_discretized_bare_coulomb_potential_el_ion_element_wise(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts_jnp,
            r_dn_carts=r_dn_carts_jnp,
            alat=alat,
        )

        assert not np.any(np.isnan(np.asarray(interactions_R_r_up_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(interactions_R_r_up_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(interactions_R_r_up_debug, interactions_R_r_up_jax, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(interactions_R_r_dn_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(interactions_R_r_dn_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(interactions_R_r_dn_debug, interactions_R_r_dn_jax, atol=atol, rtol=rtol)


class TestEcpLocal:
    """production-vs-debug for ECP local part."""

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "silicon.h5"])
    def test_compute_ecp_local_parts_all_pairs_production_vs_debug(self, trexio_file: str):
        """Production ECP local-part agrees with the helper-free ``_debug`` oracle.

        Parametrized over molecular ``water_ccecp_ccpvqz.h5`` (no MIC) and
        periodic ``silicon.h5`` (PBC path uses ``_get_min_dist_rel_R_cart_jnp``
        for nearest-image distance). One bit-identical cross-check on each
        fixture pins both code paths against the same oracle.
        """
        atol, rtol = get_tolerance("coulomb", "strict")
        trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file)
        if not os.path.exists(trexio_path):
            pytest.skip(f"TREXIO fixture {trexio_file} is not present.")
        structure_data, _, _, _, geminal_mo_data, coulomb_potential_data = read_trexio_file(
            trexio_file=trexio_path, store_tuple=True
        )
        coulomb_potential_data.sanity_check()

        rng = np.random.default_rng(0)
        n_up = geminal_mo_data.num_electron_up
        n_dn = geminal_mo_data.num_electron_dn
        if structure_data.pbc_flag:
            cell = np.asarray(structure_data.cell, dtype=np.float64)
            frac = rng.uniform(0.1, 0.9, size=(n_up + n_dn, 3))
            r_carts = frac @ cell
        else:
            r_carts = rng.uniform(-1.0, 1.0, size=(n_up + n_dn, 3))
        r_up_carts_np = r_carts[:n_up]
        r_dn_carts_np = r_carts[n_up:]
        r_up_carts_jnp = jnp.array(r_up_carts_np)
        r_dn_carts_jnp = jnp.array(r_dn_carts_np)

        vpot_ecp_local_full_NN_jax = compute_ecp_local_parts_all_pairs(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts_jnp,
            r_dn_carts=r_dn_carts_jnp,
        )
        vpot_ecp_local_full_NN_debug = _compute_ecp_local_parts_all_pairs_debug(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts_np,
            r_dn_carts=r_dn_carts_np,
        )

        assert not np.any(np.isnan(np.asarray(vpot_ecp_local_full_NN_jax))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(vpot_ecp_local_full_NN_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(vpot_ecp_local_full_NN_jax, vpot_ecp_local_full_NN_debug, atol=atol, rtol=rtol)


class TestEcpNonLocal:
    """production-vs-debug + fast-update guardrails for ECP non-local part."""

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5"])
    @pytest.mark.parametrize("Nv", Nv_params)
    @pytest.mark.parametrize("alpha, beta, gamma", angle_params)
    def test_compute_ecp_non_local_parts_all_pairs_production_vs_debug(self, trexio_file: str, Nv, alpha, beta, gamma):
        """Test the non-local ECP potential computation with the full neibohrs."""
        atol, rtol = get_tolerance("coulomb", "strict")
        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        # check its input
        coulomb_potential_data.sanity_check()

        # n_atom
        n_atom = structure_data.natom

        # define data
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=None,
            jastrow_three_body_data=None,
        )  # no jastrow for the time-being.

        wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

        r_up_carts_np = np.array(
            [
                [0.64878536, -0.83275288, 0.33532629],
                [0.55271273, 0.72310605, 0.93443775],
                [0.66767275, 0.1206456, -0.36521208],
                [-0.93165236, -0.0120386, 0.33003036],
            ]
        )
        r_dn_carts_np = np.array(
            [
                [1.0347816, 1.26162081, 0.42301735],
                [-0.57843435, 1.03651987, -0.55091542],
                [-1.56091964, -0.58952149, -0.99268141],
                [0.61863233, -0.14903326, 0.51962683],
            ]
        )

        r_up_carts_jnp = jnp.array(r_up_carts_np)
        r_dn_carts_jnp = jnp.array(r_dn_carts_np)

        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_b, sin_b = np.cos(beta), np.sin(beta)
        cos_g, sin_g = np.cos(gamma), np.sin(gamma)

        R_np = np.array(
            [
                [cos_b * cos_g, cos_g * sin_a * sin_b - cos_a * sin_g, sin_a * sin_g + cos_a * cos_g * sin_b],
                [cos_b * sin_g, cos_a * cos_g + sin_a * sin_b * sin_g, cos_a * sin_b * sin_g - cos_g * sin_a],
                [-sin_b, cos_b * sin_a, cos_a * cos_b],
            ]
        )
        R_jnp = jnp.array(R_np)

        # ecp non-local (full_NN)
        (
            mesh_non_local_ecp_part_r_up_carts_full_NN_jax,
            mesh_non_local_ecp_part_r_dn_carts_full_NN_jax,
            V_nonlocal_full_NN_jax,
            sum_V_nonlocal_full_NN_jax,
        ) = compute_ecp_non_local_parts_all_pairs(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts_jnp,
            r_dn_carts=r_dn_carts_jnp,
            Nv=Nv,
            RT=R_jnp.T,
        )

        (
            mesh_non_local_ecp_part_r_up_carts_full_NN_debug,
            mesh_non_local_ecp_part_r_dn_carts_full_NN_debug,
            V_nonlocal_full_NN_debug,
            sum_V_nonlocal_full_NN_debug,
        ) = _compute_ecp_non_local_parts_all_pairs_debug(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts_np,
            r_dn_carts=r_dn_carts_np,
            Nv=Nv,
            RT=R_np.T,
        )

        assert not np.any(np.isnan(np.asarray(sum_V_nonlocal_full_NN_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(sum_V_nonlocal_full_NN_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(sum_V_nonlocal_full_NN_debug, sum_V_nonlocal_full_NN_jax, atol=atol, rtol=rtol)

        mesh_non_local_r_up_carts_max_full_NN_jax = mesh_non_local_ecp_part_r_up_carts_full_NN_jax[
            np.argmax(V_nonlocal_full_NN_jax)
        ]
        mesh_non_local_r_up_carts_max_full_NN_debug = mesh_non_local_ecp_part_r_up_carts_full_NN_debug[
            np.argmax(V_nonlocal_full_NN_debug)
        ]
        mesh_non_local_r_dn_carts_max_full_NN_jax = mesh_non_local_ecp_part_r_dn_carts_full_NN_jax[
            np.argmax(V_nonlocal_full_NN_jax)
        ]
        mesh_non_local_r_dn_carts_max_full_NN_debug = mesh_non_local_ecp_part_r_dn_carts_full_NN_debug[
            np.argmax(V_nonlocal_full_NN_debug)
        ]
        V_ecp_non_local_max_full_NN_jax = V_nonlocal_full_NN_jax[np.argmax(V_nonlocal_full_NN_jax)]
        V_ecp_non_local_max_full_NN_debug = V_nonlocal_full_NN_debug[np.argmax(V_nonlocal_full_NN_debug)]

        assert not np.any(np.isnan(np.asarray(V_ecp_non_local_max_full_NN_jax))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(V_ecp_non_local_max_full_NN_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(
            V_ecp_non_local_max_full_NN_jax,
            V_ecp_non_local_max_full_NN_debug,
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(mesh_non_local_r_up_carts_max_full_NN_jax))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mesh_non_local_r_up_carts_max_full_NN_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(
            mesh_non_local_r_up_carts_max_full_NN_jax,
            mesh_non_local_r_up_carts_max_full_NN_debug,
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(mesh_non_local_r_dn_carts_max_full_NN_jax))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mesh_non_local_r_dn_carts_max_full_NN_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(
            mesh_non_local_r_dn_carts_max_full_NN_jax,
            mesh_non_local_r_dn_carts_max_full_NN_debug,
            atol=atol,
            rtol=rtol,
        )

        # ecp non-local (NN, N=max)
        (
            mesh_non_local_ecp_part_r_up_carts_NN_check_jax,
            mesh_non_local_ecp_part_r_dn_carts_NN_check_jax,
            V_nonlocal_NN_check_jax,
            sum_V_nonlocal_NN_check_jax,
        ) = compute_ecp_non_local_parts_nearest_neighbors(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            NN=n_atom,
            Nv=Nv,
            RT=R_jnp.T,
            r_up_carts=r_up_carts_jnp,
            r_dn_carts=r_dn_carts_jnp,
        )

        (
            mesh_non_local_ecp_part_r_up_carts_NN_check_debug,
            mesh_non_local_ecp_part_r_dn_carts_NN_check_debug,
            V_nonlocal_NN_check_debug,
            sum_V_nonlocal_NN_check_debug,
        ) = _compute_ecp_non_local_parts_nearest_neighbors_debug(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            NN=n_atom,
            Nv=Nv,
            RT=R_np.T,
            r_up_carts=r_up_carts_np,
            r_dn_carts=r_dn_carts_np,
        )

        # debug, full-NN vs check-NN
        assert not np.any(np.isnan(np.asarray(sum_V_nonlocal_full_NN_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(sum_V_nonlocal_NN_check_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(
            sum_V_nonlocal_full_NN_debug,
            sum_V_nonlocal_NN_check_debug,
            atol=atol,
            rtol=rtol,
        )

        # jax, full-NN vs check-NN
        assert not np.any(np.isnan(np.asarray(sum_V_nonlocal_full_NN_jax))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(sum_V_nonlocal_NN_check_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(sum_V_nonlocal_full_NN_jax, sum_V_nonlocal_NN_check_jax, atol=atol, rtol=rtol)

        mesh_non_local_r_up_carts_max_NN_check_jax = mesh_non_local_ecp_part_r_up_carts_NN_check_jax[
            np.argmax(V_nonlocal_NN_check_jax)
        ]
        mesh_non_local_r_up_carts_max_NN_check_debug = mesh_non_local_ecp_part_r_up_carts_NN_check_debug[
            np.argmax(V_nonlocal_NN_check_debug)
        ]
        mesh_non_local_r_dn_carts_max_NN_check_jax = mesh_non_local_ecp_part_r_dn_carts_NN_check_jax[
            np.argmax(V_nonlocal_NN_check_jax)
        ]
        mesh_non_local_r_dn_carts_max_NN_check_debug = mesh_non_local_ecp_part_r_dn_carts_NN_check_debug[
            np.argmax(V_nonlocal_NN_check_debug)
        ]
        V_ecp_non_local_max_NN_check_jax = V_nonlocal_NN_check_jax[np.argmax(V_nonlocal_NN_check_jax)]
        V_ecp_non_local_max_NN_check_debug = V_nonlocal_NN_check_debug[np.argmax(V_nonlocal_NN_check_debug)]

        # debug, full-NN vs check-NN
        assert not np.any(np.isnan(np.asarray(V_ecp_non_local_max_full_NN_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(V_ecp_non_local_max_NN_check_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(
            V_ecp_non_local_max_full_NN_debug,
            V_ecp_non_local_max_NN_check_debug,
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(mesh_non_local_r_up_carts_max_full_NN_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mesh_non_local_r_up_carts_max_NN_check_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(
            mesh_non_local_r_up_carts_max_full_NN_debug,
            mesh_non_local_r_up_carts_max_NN_check_debug,
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(mesh_non_local_r_dn_carts_max_full_NN_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mesh_non_local_r_dn_carts_max_NN_check_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(
            mesh_non_local_r_dn_carts_max_full_NN_debug,
            mesh_non_local_r_dn_carts_max_NN_check_debug,
            atol=atol,
            rtol=rtol,
        )

        # jax, full-NN vs check-NN
        assert not np.any(np.isnan(np.asarray(V_ecp_non_local_max_full_NN_jax))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(V_ecp_non_local_max_NN_check_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(
            V_ecp_non_local_max_full_NN_jax,
            V_ecp_non_local_max_NN_check_jax,
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(mesh_non_local_r_up_carts_max_full_NN_jax))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mesh_non_local_r_up_carts_max_NN_check_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(
            mesh_non_local_r_up_carts_max_full_NN_jax,
            mesh_non_local_r_up_carts_max_NN_check_jax,
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(mesh_non_local_r_dn_carts_max_full_NN_jax))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mesh_non_local_r_dn_carts_max_NN_check_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(
            mesh_non_local_r_dn_carts_max_full_NN_jax,
            mesh_non_local_r_dn_carts_max_NN_check_jax,
            atol=atol,
            rtol=rtol,
        )

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5"])
    @pytest.mark.parametrize("Nv", Nv_params)
    @pytest.mark.parametrize("alpha, beta, gamma", angle_params)
    def test_compute_ecp_non_local_parts_nearest_neighbors_production_vs_debug(self, trexio_file: str, Nv, alpha, beta, gamma):
        """Production NN-ECP non-local agrees with helper-free ``_debug`` oracle.

        Currently molecular-only. Silicon parametrisation surfaces an apparent
        shape mismatch between the production and ``_debug`` returns on
        multi-atom PBC fixtures (``argmax(V_debug)`` out of bounds for
        ``mesh_debug``); separate investigation required before silicon can be
        added back here. PBC NN-ECP non-local coverage is still provided by
        :py:func:`test_ecp_non_local_pbc_lattice_translation_invariance` and
        :py:func:`test_ecp_non_local_pbc_projector_unwrapped_vs_molecular_image`.
        """
        atol, rtol = get_tolerance("coulomb", "strict")
        trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file)
        if not os.path.exists(trexio_path):
            pytest.skip(f"TREXIO fixture {trexio_file} is not present.")
        structure_data, _, _, _, geminal_mo_data, coulomb_potential_data = read_trexio_file(
            trexio_file=trexio_path, store_tuple=True
        )
        coulomb_potential_data.sanity_check()
        n_atom = structure_data.natom

        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=None,
            jastrow_three_body_data=None,
        )
        wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

        rng = np.random.default_rng(0)
        n_up = geminal_mo_data.num_electron_up
        n_dn = geminal_mo_data.num_electron_dn
        if structure_data.pbc_flag:
            cell = np.asarray(structure_data.cell, dtype=np.float64)
            frac = rng.uniform(0.1, 0.9, size=(n_up + n_dn, 3))
            r_carts = frac @ cell
        else:
            r_carts = rng.uniform(-1.0, 1.0, size=(n_up + n_dn, 3))
        r_up_carts_np = r_carts[:n_up]
        r_dn_carts_np = r_carts[n_up:]
        r_up_carts_jnp = jnp.array(r_up_carts_np)
        r_dn_carts_jnp = jnp.array(r_dn_carts_np)

        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_b, sin_b = np.cos(beta), np.sin(beta)
        cos_g, sin_g = np.cos(gamma), np.sin(gamma)

        R_np = np.array(
            [
                [cos_b * cos_g, cos_g * sin_a * sin_b - cos_a * sin_g, sin_a * sin_g + cos_a * cos_g * sin_b],
                [cos_b * sin_g, cos_a * cos_g + sin_a * sin_b * sin_g, cos_a * sin_b * sin_g - cos_g * sin_a],
                [-sin_b, cos_b * sin_a, cos_a * cos_b],
            ]
        )
        R_jnp = jnp.array(R_np)

        for NN in range(1, n_atom):
            # ecp non-local (NN, NN=NN)
            (
                mesh_non_local_ecp_part_r_up_carts_NN_jax,
                mesh_non_local_ecp_part_r_dn_carts_NN_jax,
                V_nonlocal_NN_jax,
                sum_V_nonlocal_NN_jax,
            ) = compute_ecp_non_local_parts_nearest_neighbors(
                coulomb_potential_data=coulomb_potential_data,
                wavefunction_data=wavefunction_data,
                r_up_carts=r_up_carts_jnp,
                r_dn_carts=r_dn_carts_jnp,
                Nv=Nv,
                NN=NN,
                RT=R_jnp.T,
            )

            (
                mesh_non_local_ecp_part_r_up_carts_NN_debug,
                mesh_non_local_ecp_part_r_dn_carts_NN_debug,
                V_nonlocal_NN_debug,
                sum_V_nonlocal_NN_debug,
            ) = _compute_ecp_non_local_parts_nearest_neighbors_debug(
                coulomb_potential_data=coulomb_potential_data,
                wavefunction_data=wavefunction_data,
                r_up_carts=r_up_carts_np,
                r_dn_carts=r_dn_carts_np,
                Nv=Nv,
                NN=NN,
                RT=R_np.T,
            )

            assert not np.any(np.isnan(np.asarray(sum_V_nonlocal_NN_debug))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(sum_V_nonlocal_NN_jax))), "NaN detected in second argument"
            np.testing.assert_allclose(sum_V_nonlocal_NN_debug, sum_V_nonlocal_NN_jax, atol=atol, rtol=rtol)

            mesh_non_local_r_up_carts_max_NN_jax = mesh_non_local_ecp_part_r_up_carts_NN_jax[np.argmax(V_nonlocal_NN_jax)]
            mesh_non_local_r_up_carts_max_NN_debug = mesh_non_local_ecp_part_r_up_carts_NN_debug[np.argmax(V_nonlocal_NN_debug)]
            mesh_non_local_r_dn_carts_max_NN_jax = mesh_non_local_ecp_part_r_dn_carts_NN_jax[np.argmax(V_nonlocal_NN_jax)]
            mesh_non_local_r_dn_carts_max_NN_debug = mesh_non_local_ecp_part_r_dn_carts_NN_debug[np.argmax(V_nonlocal_NN_debug)]
            V_ecp_non_local_max_NN_jax = V_nonlocal_NN_jax[np.argmax(V_nonlocal_NN_jax)]
            V_ecp_non_local_max_NN_debug = V_nonlocal_NN_debug[np.argmax(V_nonlocal_NN_debug)]

            assert not np.any(np.isnan(np.asarray(V_ecp_non_local_max_NN_jax))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(V_ecp_non_local_max_NN_debug))), "NaN detected in second argument"
            np.testing.assert_allclose(
                V_ecp_non_local_max_NN_jax,
                V_ecp_non_local_max_NN_debug,
                atol=atol,
                rtol=rtol,
            )
            assert not np.any(np.isnan(np.asarray(mesh_non_local_r_up_carts_max_NN_jax))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(mesh_non_local_r_up_carts_max_NN_debug))), "NaN detected in second argument"
            np.testing.assert_allclose(
                mesh_non_local_r_up_carts_max_NN_jax,
                mesh_non_local_r_up_carts_max_NN_debug,
                atol=atol,
                rtol=rtol,
            )
            assert not np.any(np.isnan(np.asarray(mesh_non_local_r_dn_carts_max_NN_jax))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(mesh_non_local_r_dn_carts_max_NN_debug))), "NaN detected in second argument"
            np.testing.assert_allclose(
                mesh_non_local_r_dn_carts_max_NN_jax,
                mesh_non_local_r_dn_carts_max_NN_debug,
                atol=atol,
                rtol=rtol,
            )

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "silicon.h5"])
    def test_compute_ecp_non_local_parts_nearest_neighbors_fast_update_vs_reference(self, trexio_file: str):
        """``fast_update`` variant matches the reference NN-ECP non-local kernel.

        Parametrised over molecular ``water_ccecp_ccpvqz.h5`` and periodic
        ``silicon.h5``. Both variants are JIT-compiled production paths, so the
        comparison is feasible on silicon after the
        ``_primitive_image_radius`` atol-tolerance fix (``n_image`` ~6,900 ->
        ~250). The PBC inner loop is capped at ``NN=1`` because each NN value
        triggers a fresh static-JIT compile and silicon's larger HLO would
        otherwise stack 8 multi-second compiles; the wrap / MIC paths
        exercised are identical for every NN value.

        Tolerance: the ``fast_update`` path computes ``Psi(R')/Psi(R)`` with the
        Jastrow ratio assembled in the ``jastrow_ratio`` zone (fp64 in mixed
        mode), while the independent reference path computes full ``J(R)`` and
        ``J(R')`` values in the ``jastrow_eval`` zone (fp32), then subtracts in
        the ``coulomb`` zone (fp64). The comparison is therefore bounded by
        ``jastrow_eval`` even though the production rank-1 ratio is fp64.
        """
        atol, rtol = get_tolerance_min(["coulomb", "jastrow_eval", "jastrow_ratio"], "strict")
        trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file)
        if not os.path.exists(trexio_path):
            pytest.skip(f"TREXIO fixture {trexio_file} is not present.")
        structure_data, _aos_data, _mos_data_up, _mos_data_dn, geminal_mo_data, coulomb_potential_data = read_trexio_file(
            trexio_file=trexio_path,
            store_tuple=True,
        )

        rng = np.random.default_rng(0)
        jastrow_data = _build_full_jastrow_data(structure_data, geminal_mo_data, coulomb_potential_data, rng)
        wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

        n_up = geminal_mo_data.num_electron_up
        n_dn = geminal_mo_data.num_electron_dn
        if structure_data.pbc_flag:
            # Distribute around the nuclei: silicon's 16+16 valence electrons
            # at arbitrary cell-uniform positions land deep in the periodic
            # AGP node basin (det ~ 1e-60) and amplify numerical noise.
            atoms = np.asarray(structure_data.positions, dtype=np.float64)
            natom = atoms.shape[0]
            up_per_atom = n_up // natom
            dn_per_atom = n_dn // natom
            r_up_carts = np.repeat(atoms, up_per_atom, axis=0) + rng.normal(scale=0.4, size=(n_up, 3))
            r_dn_carts = np.repeat(atoms, dn_per_atom, axis=0) + rng.normal(scale=0.4, size=(n_dn, 3))
        else:
            r_up_carts = rng.uniform(-1.0, 1.0, size=(n_up, 3))
            r_dn_carts = rng.uniform(-1.0, 1.0, size=(n_dn, 3))

        geminal_old = compute_geminal_all_elements(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        A_old_inv = np.linalg.inv(np.array(geminal_old))

        nn_max = 2 if structure_data.pbc_flag else structure_data.natom
        for NN in range(1, nn_max):
            # ecp non-local (NN, NN=NN)
            Nv = 6
            RT = np.eye(3)

            (
                mesh_up_ref,
                mesh_dn_ref,
                V_ref,
                sum_V_ref,
            ) = compute_ecp_non_local_parts_nearest_neighbors(
                coulomb_potential_data=coulomb_potential_data,
                wavefunction_data=wavefunction_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                NN=NN,
                Nv=Nv,
                RT=RT,
            )

            (
                mesh_up_fast,
                mesh_dn_fast,
                V_fast,
                sum_V_fast,
            ) = compute_ecp_non_local_parts_nearest_neighbors_fast_update(
                coulomb_potential_data=coulomb_potential_data,
                wavefunction_data=wavefunction_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                RT=RT,
                A_old_inv=A_old_inv,
                NN=NN,
                Nv=Nv,
                flag_determinant_only=False,
            )

            assert not np.any(np.isnan(np.asarray(np.asarray(sum_V_fast)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(sum_V_ref)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(sum_V_fast), np.asarray(sum_V_ref), atol=atol, rtol=rtol)
            assert not np.any(np.isnan(np.asarray(np.asarray(V_fast)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(V_ref)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(V_fast), np.asarray(V_ref), atol=atol, rtol=rtol)
            assert not np.any(np.isnan(np.asarray(np.asarray(mesh_up_fast)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(mesh_up_ref)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(mesh_up_fast), np.asarray(mesh_up_ref), atol=atol, rtol=rtol)
            assert not np.any(np.isnan(np.asarray(np.asarray(mesh_dn_fast)))), "NaN detected in first argument"
            assert not np.any(np.isnan(np.asarray(np.asarray(mesh_dn_ref)))), "NaN detected in second argument"
            np.testing.assert_allclose(np.asarray(mesh_dn_fast), np.asarray(mesh_dn_ref), atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "silicon.h5"])
    def test_compute_ecp_non_local_parts_nearest_neighbors_fast_update_streaming_vs_reference(self, trexio_file: str):
        """``compute_ecp_non_local_parts_nearest_neighbors_fast_update`` must
        return identical mesh ratios when called with a consistent
        ``det_ratio_state`` vs the reference ``det_ratio_state=None`` path.

        The streaming caller forwards ``Det_ratio_streaming_state`` (or
        ``Det_streaming_state``) so the kernel can skip the bulk-side AO eval
        and the ``lambda @ ao_*`` precontracts. Parametrised over molecular
        ``water_ccecp_ccpvqz.h5`` and periodic ``silicon.h5``; the PBC inner
        loop is capped at ``NN=1`` for JIT-recompile control.
        """
        atol, rtol = get_tolerance("coulomb", "strict")
        trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file)
        if not os.path.exists(trexio_path):
            pytest.skip(f"TREXIO fixture {trexio_file} is not present.")
        structure_data, _aos_data, _mos_data_up, _mos_data_dn, geminal_mo_data, coulomb_potential_data = read_trexio_file(
            trexio_file=trexio_path,
            store_tuple=True,
        )

        rng = np.random.default_rng(0)
        jastrow_data = _build_full_jastrow_data(structure_data, geminal_mo_data, coulomb_potential_data, rng)
        wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

        n_up = geminal_mo_data.num_electron_up
        n_dn = geminal_mo_data.num_electron_dn
        if structure_data.pbc_flag:
            atoms = np.asarray(structure_data.positions, dtype=np.float64)
            natom = atoms.shape[0]
            up_per_atom = n_up // natom
            dn_per_atom = n_dn // natom
            r_up_carts = jnp.asarray(np.repeat(atoms, up_per_atom, axis=0) + rng.normal(scale=0.4, size=(n_up, 3)))
            r_dn_carts = jnp.asarray(np.repeat(atoms, dn_per_atom, axis=0) + rng.normal(scale=0.4, size=(n_dn, 3)))
        else:
            r_up_carts = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n_up, 3)))
            r_dn_carts = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n_dn, 3)))

        geminal_old = compute_geminal_all_elements(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        A_old_inv = jnp.linalg.inv(geminal_old)

        # Build a fresh slim state at the reference config (consistent with r_up/r_dn).
        state = _init_det_ratio_streaming_state(
            geminal_data=geminal_mo_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        Nv = 6
        RT = jnp.eye(3)

        nn_max = 2 if structure_data.pbc_flag else structure_data.natom
        for NN in range(1, nn_max):
            # Legacy path (no state).
            (
                mesh_up_reference,
                mesh_dn_reference,
                V_reference,
                sum_V_reference,
            ) = compute_ecp_non_local_parts_nearest_neighbors_fast_update(
                coulomb_potential_data=coulomb_potential_data,
                wavefunction_data=wavefunction_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                RT=RT,
                A_old_inv=A_old_inv,
                NN=NN,
                Nv=Nv,
                flag_determinant_only=False,
                det_ratio_state=None,
            )

            # Streaming path (with state).
            (
                mesh_up_streaming,
                mesh_dn_streaming,
                V_streaming,
                sum_V_streaming,
            ) = compute_ecp_non_local_parts_nearest_neighbors_fast_update(
                coulomb_potential_data=coulomb_potential_data,
                wavefunction_data=wavefunction_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                RT=RT,
                A_old_inv=A_old_inv,
                NN=NN,
                Nv=Nv,
                flag_determinant_only=False,
                det_ratio_state=state,
            )

            np.testing.assert_allclose(np.asarray(sum_V_streaming), np.asarray(sum_V_reference), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(V_streaming), np.asarray(V_reference), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(mesh_up_streaming), np.asarray(mesh_up_reference), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(mesh_dn_streaming), np.asarray(mesh_dn_reference), atol=atol, rtol=rtol)


class TestEcpCoulombFast:
    """fast vs reference for combined ECP+Coulomb potential."""

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file, Nv, alpha, beta, gamma", _FAST_VS_REFERENCE_PARAMS)
    def test_compute_ecp_coulomb_potential_fast_vs_reference(
        self, trexio_file: str, Nv: int, alpha: float, beta: float, gamma: float
    ):
        """compute_ecp_coulomb_potential_fast matches compute_ecp_coulomb_potential when A_old_inv is exact.

        With a freshly computed LU inverse both functions must agree to full double
        precision because the only difference is which code path owns the LU solve.
        """
        atol, rtol = get_tolerance("coulomb", "strict")
        trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file)
        if not os.path.exists(trexio_path):
            pytest.skip(f"TREXIO fixture {trexio_file} is not present.")
        structure_data, _aos_data, _mos_data_up, _mos_data_dn, geminal_mo_data, coulomb_potential_data = read_trexio_file(
            trexio_file=trexio_path,
            store_tuple=True,
        )
        rng = np.random.default_rng(0)
        wavefunction_data = Wavefunction_data(
            geminal_data=geminal_mo_data,
            jastrow_data=Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None),
        )
        n_up = geminal_mo_data.num_electron_up
        n_dn = geminal_mo_data.num_electron_dn
        if structure_data.pbc_flag:
            atoms = np.asarray(structure_data.positions, dtype=np.float64)
            natom = atoms.shape[0]
            up_per_atom = n_up // natom
            dn_per_atom = n_dn // natom
            r_up_carts = jnp.array(np.repeat(atoms, up_per_atom, axis=0) + rng.normal(scale=0.4, size=(n_up, 3)))
            r_dn_carts = jnp.array(np.repeat(atoms, dn_per_atom, axis=0) + rng.normal(scale=0.4, size=(n_dn, 3)))
        else:
            r_up_carts = jnp.array(rng.uniform(-1.0, 1.0, size=(n_up, 3)))
            r_dn_carts = jnp.array(rng.uniform(-1.0, 1.0, size=(n_dn, 3)))
        G = compute_geminal_all_elements(geminal_data=geminal_mo_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
        A_old_inv = jnp.linalg.inv(G)

        cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
        cos_b, sin_b = jnp.cos(beta), jnp.sin(beta)
        cos_g, sin_g = jnp.cos(gamma), jnp.sin(gamma)
        R = jnp.array(
            [
                [cos_b * cos_g, cos_g * sin_a * sin_b - cos_a * sin_g, sin_a * sin_g + cos_a * cos_g * sin_b],
                [cos_b * sin_g, cos_a * cos_g + sin_a * sin_b * sin_g, cos_a * sin_b * sin_g - cos_g * sin_a],
                [-sin_b, cos_b * sin_a, cos_a * cos_b],
            ]
        )
        RT = R.T

        # Reference: compute_ecp_coulomb_potential performs its own LU internally
        V_ref = compute_ecp_coulomb_potential(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            RT=RT,
            Nv=Nv,
        )

        # Fast path: accepts pre-computed A_old_inv, skips LU
        V_fast = compute_ecp_coulomb_potential_fast(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            RT=RT,
            A_old_inv=A_old_inv,
            Nv=Nv,
        )

        assert not np.any(np.isnan(np.asarray(V_ref))), "NaN detected in V_ref"
        assert not np.any(np.isnan(np.asarray(V_fast))), "NaN detected in V_fast"
        np.testing.assert_allclose(np.asarray(V_fast), np.asarray(V_ref), atol=atol, rtol=rtol)


class TestCoulombFast:
    """fast vs reference for the full Coulomb potential."""

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file, Nv, alpha, beta, gamma", _FAST_VS_REFERENCE_PARAMS)
    def test_compute_coulomb_potential_fast_vs_reference(
        self, trexio_file: str, Nv: int, alpha: float, beta: float, gamma: float
    ):
        """compute_coulomb_potential_fast matches compute_coulomb_potential when A_old_inv is exact.

        Covers both bare-Coulomb and ECP branches.  For ECP systems the non-local
        Psi(r')/Psi(r) ratios must be identical between the two paths because the
        same A_old_inv (exact LU inverse) is used.
        """
        atol, rtol = get_tolerance("coulomb", "strict")
        trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file)
        if not os.path.exists(trexio_path):
            pytest.skip(f"TREXIO fixture {trexio_file} is not present.")
        structure_data, _aos_data, _mos_data_up, _mos_data_dn, geminal_mo_data, coulomb_potential_data = read_trexio_file(
            trexio_file=trexio_path,
            store_tuple=True,
        )
        rng = np.random.default_rng(0)
        wavefunction_data = Wavefunction_data(
            geminal_data=geminal_mo_data,
            jastrow_data=Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None),
        )
        n_up = geminal_mo_data.num_electron_up
        n_dn = geminal_mo_data.num_electron_dn
        if structure_data.pbc_flag:
            atoms = np.asarray(structure_data.positions, dtype=np.float64)
            natom = atoms.shape[0]
            up_per_atom = n_up // natom
            dn_per_atom = n_dn // natom
            r_up_carts = jnp.array(np.repeat(atoms, up_per_atom, axis=0) + rng.normal(scale=0.4, size=(n_up, 3)))
            r_dn_carts = jnp.array(np.repeat(atoms, dn_per_atom, axis=0) + rng.normal(scale=0.4, size=(n_dn, 3)))
        else:
            r_up_carts = jnp.array(rng.uniform(-1.0, 1.0, size=(n_up, 3)))
            r_dn_carts = jnp.array(rng.uniform(-1.0, 1.0, size=(n_dn, 3)))
        G = compute_geminal_all_elements(geminal_data=geminal_mo_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
        A_old_inv = jnp.linalg.inv(G)

        cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
        cos_b, sin_b = jnp.cos(beta), jnp.sin(beta)
        cos_g, sin_g = jnp.cos(gamma), jnp.sin(gamma)
        R = jnp.array(
            [
                [cos_b * cos_g, cos_g * sin_a * sin_b - cos_a * sin_g, sin_a * sin_g + cos_a * cos_g * sin_b],
                [cos_b * sin_g, cos_a * cos_g + sin_a * sin_b * sin_g, cos_a * sin_b * sin_g - cos_g * sin_a],
                [-sin_b, cos_b * sin_a, cos_a * cos_b],
            ]
        )
        RT = R.T

        # Reference: standard path, computes LU internally for ECP non-local part
        V_ref = compute_coulomb_potential(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            RT=RT,
            wavefunction_data=wavefunction_data,
            Nv=Nv,
        )

        # Fast path: accepts caller-supplied inverse
        V_fast = compute_coulomb_potential_fast(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            RT=RT,
            A_old_inv=A_old_inv,
            wavefunction_data=wavefunction_data,
            Nv=Nv,
        )

        assert not np.any(np.isnan(np.asarray(V_ref))), "NaN detected in V_ref"
        assert not np.any(np.isnan(np.asarray(V_fast))), "NaN detected in V_fast"
        np.testing.assert_allclose(np.asarray(V_fast), np.asarray(V_ref), atol=atol, rtol=rtol)


class TestFastVsFastDebug:
    """Prod/debug pair guardrails for the three Coulomb ``_fast`` variants.

    Each ``_fast_debug`` discards ``A_old_inv`` and delegates to the non-fast
    sibling (fresh LU from scratch); the assertion is that the fast
    Sherman-Morrison path reproduces that reference to strict double precision
    when ``A_old_inv`` is the exact LU inverse at the current configuration.
    """

    @staticmethod
    def _setup(trexio_file: str, alpha: float, beta: float, gamma: float):
        """Build (data, wavefunction, r_up, r_dn, A_old_inv, RT) for a fixture."""
        trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file)
        if not os.path.exists(trexio_path):
            pytest.skip(f"TREXIO fixture {trexio_file} is not present.")
        structure_data, _aos_data, _mos_data_up, _mos_data_dn, geminal_mo_data, coulomb_potential_data = read_trexio_file(
            trexio_file=trexio_path,
            store_tuple=True,
        )
        rng = np.random.default_rng(0)
        wavefunction_data = Wavefunction_data(
            geminal_data=geminal_mo_data,
            jastrow_data=Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None),
        )
        n_up = geminal_mo_data.num_electron_up
        n_dn = geminal_mo_data.num_electron_dn
        if structure_data.pbc_flag:
            atoms = np.asarray(structure_data.positions, dtype=np.float64)
            natom = atoms.shape[0]
            r_up_carts = jnp.array(np.repeat(atoms, n_up // natom, axis=0) + rng.normal(scale=0.4, size=(n_up, 3)))
            r_dn_carts = jnp.array(np.repeat(atoms, n_dn // natom, axis=0) + rng.normal(scale=0.4, size=(n_dn, 3)))
        else:
            r_up_carts = jnp.array(rng.uniform(-1.0, 1.0, size=(n_up, 3)))
            r_dn_carts = jnp.array(rng.uniform(-1.0, 1.0, size=(n_dn, 3)))
        G = compute_geminal_all_elements(geminal_data=geminal_mo_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
        A_old_inv = jnp.linalg.inv(G)
        cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
        cos_b, sin_b = jnp.cos(beta), jnp.sin(beta)
        cos_g, sin_g = jnp.cos(gamma), jnp.sin(gamma)
        R = jnp.array(
            [
                [cos_b * cos_g, cos_g * sin_a * sin_b - cos_a * sin_g, sin_a * sin_g + cos_a * cos_g * sin_b],
                [cos_b * sin_g, cos_a * cos_g + sin_a * sin_b * sin_g, cos_a * sin_b * sin_g - cos_g * sin_a],
                [-sin_b, cos_b * sin_a, cos_a * cos_b],
            ]
        )
        return coulomb_potential_data, wavefunction_data, r_up_carts, r_dn_carts, A_old_inv, R.T

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file, Nv, alpha, beta, gamma", _FAST_VS_REFERENCE_PARAMS)
    def test_compute_ecp_coulomb_potential_fast_vs_fast_debug(
        self, trexio_file: str, Nv: int, alpha: float, beta: float, gamma: float
    ):
        """compute_ecp_coulomb_potential_fast == _compute_ecp_coulomb_potential_fast_debug."""
        atol, rtol = get_tolerance("coulomb", "strict")
        data, wf, r_up, r_dn, A_old_inv, RT = self._setup(trexio_file, alpha, beta, gamma)
        V_fast = compute_ecp_coulomb_potential_fast(
            coulomb_potential_data=data,
            wavefunction_data=wf,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            RT=RT,
            A_old_inv=A_old_inv,
            Nv=Nv,
        )
        V_debug = _compute_ecp_coulomb_potential_fast_debug(
            coulomb_potential_data=data,
            wavefunction_data=wf,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            RT=RT,
            A_old_inv=A_old_inv,
            Nv=Nv,
        )
        assert not np.any(np.isnan(np.asarray(V_fast))), "NaN detected in V_fast"
        assert not np.any(np.isnan(np.asarray(V_debug))), "NaN detected in V_debug"
        np.testing.assert_allclose(np.asarray(V_fast), np.asarray(V_debug), atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file, Nv, alpha, beta, gamma", _FAST_VS_REFERENCE_PARAMS)
    def test_compute_coulomb_potential_fast_vs_fast_debug(
        self, trexio_file: str, Nv: int, alpha: float, beta: float, gamma: float
    ):
        """compute_coulomb_potential_fast == _compute_coulomb_potential_fast_debug (bare + ECP)."""
        atol, rtol = get_tolerance("coulomb", "strict")
        data, wf, r_up, r_dn, A_old_inv, RT = self._setup(trexio_file, alpha, beta, gamma)
        V_fast = compute_coulomb_potential_fast(
            coulomb_potential_data=data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            RT=RT,
            A_old_inv=A_old_inv,
            wavefunction_data=wf,
            Nv=Nv,
        )
        V_debug = _compute_coulomb_potential_fast_debug(
            coulomb_potential_data=data,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            RT=RT,
            A_old_inv=A_old_inv,
            wavefunction_data=wf,
            Nv=Nv,
        )
        assert not np.any(np.isnan(np.asarray(V_fast))), "NaN detected in V_fast"
        assert not np.any(np.isnan(np.asarray(V_debug))), "NaN detected in V_debug"
        np.testing.assert_allclose(np.asarray(V_fast), np.asarray(V_debug), atol=atol, rtol=rtol)


class TestEwaldIon:
    """production-vs-debug for ion-ion Ewald."""

    @pytest.mark.light
    @pytest.mark.parametrize("fixture_name", ["nacl_sc", "silicon"])
    def test_compute_ewald_potential_ii_production_vs_debug(self, fixture_name: str):
        """``_compute_ewald_potential_ii_debug`` agrees bit-identically with
        ``ewald_data.ii_energy`` on each fixture.

        The ``_debug`` re-implements the formula from scratch (no
        helper calls), so this single bit-identical comparison validates the
        entire production helper chain ``_eta_iteration_heuristic`` /
        ``_enumerate_real_images`` / ``_enumerate_g_vectors`` /
        ``_compute_ewald_ii_energy``.

        The silicon TREXIO fixture exercises the ECP ``_effective_charges``
        path (``Z_eff = Z - z_core``) which must thread through unchanged.
        """
        if fixture_name == "nacl_sc":
            structure, coulomb = _make_nacl_sc_fixture()
            tol = 1e-12
        else:
            structure, coulomb = _make_silicon_ewald_ion_fixture()
            tol = 1e-10
        ewald_data = build_ewald_data(structure, coulomb, tol=tol)
        coulomb_with_ewald = coulomb.replace(ewald_data=ewald_data)
        E_ii_prod = float(compute_ewald_potential_ii(coulomb_with_ewald, ewald_data))
        E_ii_debug = _compute_ewald_potential_ii_debug(coulomb_with_ewald, ewald_data)
        atol, rtol = get_tolerance("ewald", "strict")
        np.testing.assert_allclose(E_ii_prod, E_ii_debug, atol=atol, rtol=rtol)


class TestEwaldElec:
    """production-vs-debug for electron-electron Ewald."""

    @pytest.mark.light
    @pytest.mark.parametrize("fixture_name", ["h2_box", "silicon"])
    def test_compute_ewald_potential_ee_production_vs_debug(self, fixture_name: str):
        """``compute_ewald_potential_ee`` agrees bit-identically with the
        helper-free ``_compute_ewald_potential_ee_debug`` reference.

        Validates the full production path: real-space ``vmap`` over ``(i, j, n)``,
        reciprocal-space structure-factor evaluation, and per-electron
        self-energy ``-(eta/sqrt(pi)) N_e`` attribution.
        """
        if fixture_name == "h2_box":
            structure, coulomb, r_up, r_dn = _make_h2_box_fixture()
        else:
            structure, coulomb, r_up, r_dn = _make_silicon_ewald_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-10)
        E_prod = float(compute_ewald_potential_ee(r_up, r_dn, ewald_data))
        E_debug = _compute_ewald_potential_ee_debug(r_up, r_dn, ewald_data)
        atol, rtol = get_tolerance("ewald", "strict")
        np.testing.assert_allclose(E_prod, E_debug, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_compute_ewald_potential_ee_single_electron_production_vs_debug(self):
        """A single electron in the periodic box: ``E_ee`` matches the
        ``_debug`` reference, which includes per-electron self-energy
        ``-(eta/sqrt(pi))`` plus the lattice self-image and reciprocal-self
        contributions.

        The single-electron case is the self-energy attribution stress test:
        no pair contributions, so the entire ``E_ee`` is structurally the
        self-energy piece. Bit-identical agreement validates the attribution
        boundary between ``_ee`` / ``_ei`` / ``_ii`` dispatch.
        """
        structure, coulomb, _, _ = _make_h2_box_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-10)
        r_up = np.array([[0.5, 0.0, 0.0]], dtype=np.float64)
        r_dn = np.zeros((0, 3), dtype=np.float64)
        E_prod = float(compute_ewald_potential_ee(r_up, r_dn, ewald_data))
        E_debug = _compute_ewald_potential_ee_debug(r_up, r_dn, ewald_data)
        atol, rtol = get_tolerance("ewald", "strict")
        np.testing.assert_allclose(E_prod, E_debug, atol=atol, rtol=rtol)


class TestEwaldElecIon:
    """production-vs-debug for electron-ion Ewald (incl. discretized)."""

    @pytest.mark.light
    @pytest.mark.parametrize("fixture_name", ["h2_box", "silicon"])
    def test_compute_ewald_potential_ei_production_vs_debug(self, fixture_name: str):
        """``compute_ewald_potential_ei`` agrees bit-identically with
        ``_compute_ewald_potential_ei_debug``.

        Returns per-electron arrays of shape ``(N_up,)`` and ``(N_dn,)``;
        the comparison is element-wise. The silicon fixture exercises the
        ECP ``Z_eff = Z - z_core`` path.
        """
        if fixture_name == "h2_box":
            structure, coulomb, r_up, r_dn = _make_h2_box_fixture()
        else:
            structure, coulomb, r_up, r_dn = _make_silicon_ewald_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-10)
        E_up_prod, E_dn_prod = compute_ewald_potential_ei(coulomb, r_up, r_dn, ewald_data)
        E_up_debug, E_dn_debug = _compute_ewald_potential_ei_debug(coulomb, r_up, r_dn, ewald_data)
        atol, rtol = get_tolerance("ewald", "strict")
        assert tuple(E_up_prod.shape) == (r_up.shape[0],)
        assert tuple(E_dn_prod.shape) == (r_dn.shape[0],)
        np.testing.assert_allclose(np.asarray(E_up_prod), E_up_debug, atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(E_dn_prod), E_dn_debug, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_compute_ewald_potential_ei_discretized_production_vs_debug(self):
        """``compute_ewald_potential_ei_discretized`` agrees bit-identically
        with its helper-free reference on the H2-in-box fixture for an
        LRDMC-typical ``alat``.

        The ``alat`` floor only modifies the real-space short-range piece;
        a single bit-identical comparison validates the production path
        (real-space ``max(r, alat)`` reduction + reciprocal-space
        pass-through) against the numpy reference.
        """
        structure, coulomb, r_up, r_dn = _make_h2_box_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-10)
        alat = 0.40
        E_up_prod, E_dn_prod = compute_ewald_potential_ei_discretized(
            coulomb,
            r_up,
            r_dn,
            ewald_data,
            alat,
        )
        E_up_debug, E_dn_debug = _compute_ewald_potential_ei_discretized_debug(
            coulomb,
            r_up,
            r_dn,
            ewald_data,
            alat,
        )
        atol, rtol = get_tolerance("ewald", "strict")
        assert tuple(E_up_prod.shape) == (r_up.shape[0],)
        assert tuple(E_dn_prod.shape) == (r_dn.shape[0],)
        np.testing.assert_allclose(np.asarray(E_up_prod), E_up_debug, atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(E_dn_prod), E_dn_debug, atol=atol, rtol=rtol)


class TestEwaldPhysical:
    """lattice translation invariance + total-energy eta invariance."""

    @pytest.mark.light
    def test_compute_ewald_potential_ee_pbc_lattice_translation_invariance(self):
        """V_ee Gamma-point Bloch periodicity: shifting any electron by a lattice
        vector ``T`` must leave the electron-electron Ewald energy unchanged.

        Caught the production bug
        where ``compute_ewald_potential_ee`` built the real-space term as
        ``r_ij = r_i - r_j`` with the truncated ``real_images`` list
        (``r_real_cut = sqrt(-log tol)/eta`` is typically smaller than the
        smallest lattice vector, so ``real_images = [(0,0,0)]`` is the only
        element). When two electrons sit in different periodic copies (or
        equivalently when the home-cell representative of ``r_i - r_j`` is
        not the minimum-image displacement), the dominant short-range
        ``erfc(eta r_min) / r_min`` contribution is dropped because the
        lattice translation that would bring ``r_i - r_j + T`` close to zero
        is not in ``real_images``. The reciprocal-space sum is intrinsically
        periodic (``cos(G . (r + T)) = cos(G . r)`` for any lattice
        translation), so it does NOT compensate, and ``V_ee`` ends up
        lattice-non-invariant by an order-Ha amount per close pair.

        Setup: two singlet-coupled electrons 0.5 Bohr apart inside the 10
        Bohr cubic box (close enough that ``erfc(eta * 0.5)/0.5`` is on the
        order of 1 Ha). Shifting one of them by a primitive lattice vector
        moves it to a periodic copy 9.5 Bohr from the partner in the
        home-cell convention. The real-space contribution effectively
        vanishes after the shift unless the formula picks the minimum-image
        displacement.

        ``compute_ewald_potential_ee`` routes ``r_ij``
        through :py:func:`jqmc.structure._minimum_image_displacement`
        (Wigner-Seitz, exact for arbitrary cells) before adding
        ``real_images`` and feeding into ``erfc``.
        """
        structure, coulomb, _, _ = _make_h2_box_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-10)

        # Two electrons close together inside the box; significant real-space
        # ``erfc(eta * |r_ij|)/|r_ij|`` contribution.
        r_up = np.array([[1.0, 5.0, 5.0]], dtype=np.float64)
        r_dn = np.array([[1.5, 5.0, 5.0]], dtype=np.float64)
        E_orig = float(compute_ewald_potential_ee(r_up, r_dn, ewald_data))

        cell = np.asarray(structure.cell, dtype=np.float64)
        # Shift only the up electron by lattice vector ``a``; the displacement
        # ``r_up - r_dn`` is no longer its minimum image in the home cell.
        r_up_shift = r_up + cell[0]
        E_shift = float(compute_ewald_potential_ee(r_up_shift, r_dn, ewald_data))

        atol, rtol = get_tolerance("ewald", "strict")
        np.testing.assert_allclose(E_shift, E_orig, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_compute_ewald_potential_ei_pbc_lattice_translation_invariance(self):
        """V_ei Gamma-point Bloch periodicity: shifting any electron by a lattice
        vector must leave the electron-ion Ewald energy unchanged.

        Same root cause and same diagnostic as the V_ee variant: the
        real-space ``r_iI = r_e - R_I`` enters ``erfc(eta r)/r`` without
        being wrapped to the minimum image of the (electron, ion) pair, and
        the truncated ``real_images`` list (just T=0 for typical PBC cells
        at tol=1e-10) cannot recover the dropped near-image contribution.
        Production walkers usually sit inside the home cell so a missing wrap
        surfaces only via the *open-shell* density distribution where
        unpaired alpha electrons spend more time near boundaries -- but the
        invariant identity tested here is unconditional and fires on any
        configuration where a single electron lies close to a nucleus.

        Setup: one up electron at ``R_0 + (0.5, 0, 0)`` Bohr -- close enough
        to nucleus 0 that ``erfc(eta * 0.5)/0.5 * Z`` is order Ha -- then
        shift it by lattice vector ``a``. The home-cell ``r_iI`` ramps up to
        ``|a|`` so ``erfc(eta * |a|) ~ 0`` and the real-space contribution
        is silently dropped.
        """
        structure, coulomb, _, _ = _make_h2_box_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-10)

        positions = np.asarray(structure.positions, dtype=np.float64)
        # Up electron 0.5 Bohr from nucleus 0 (significant real-space e-i contrib).
        r_up = positions[:1] + np.array([[0.5, 0.0, 0.0]], dtype=np.float64)
        r_dn = np.zeros((0, 3), dtype=np.float64)
        E_up_orig, E_dn_orig = compute_ewald_potential_ei(coulomb, r_up, r_dn, ewald_data)
        sum_orig = float(jnp.sum(E_up_orig) + jnp.sum(E_dn_orig))

        cell = np.asarray(structure.cell, dtype=np.float64)
        r_up_shift = r_up + cell[0]
        E_up_shift, E_dn_shift = compute_ewald_potential_ei(coulomb, r_up_shift, r_dn, ewald_data)
        sum_shift = float(jnp.sum(E_up_shift) + jnp.sum(E_dn_shift))

        atol, rtol = get_tolerance("ewald", "strict")
        np.testing.assert_allclose(sum_shift, sum_orig, atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("eta_factor", [0.5, 1.0, 2.0])
    def test_compute_ewald_potential_total_energy_eta_invariance_h2_box(self, eta_factor: float):
        """``E_total = E_ii + sum(E_ei_up + E_ei_dn) + E_ee`` is invariant under
        the (non-physical) Ewald split parameter eta.

        The all-electron H2 fixture has 2 ions of charge ``+1`` and 2
        electrons of charge ``-1`` in a 10 Bohr cubic cell --- total charge
        zero, so the omitted uniform background term is genuinely zero and
        does not contaminate the eta-invariance check.
        """
        structure, coulomb, r_up, r_dn = _make_h2_box_fixture()
        ewald_default = build_ewald_data(structure, coulomb, tol=1e-10)
        coulomb_default = coulomb.replace(ewald_data=ewald_default)
        E_ii_def = float(compute_ewald_potential_ii(coulomb_default, ewald_default))
        E_ei_up_def, E_ei_dn_def = compute_ewald_potential_ei(coulomb_default, r_up, r_dn, ewald_default)
        E_ee_def = float(compute_ewald_potential_ee(r_up, r_dn, ewald_default))
        E_total_default = E_ii_def + float(np.sum(np.asarray(E_ei_up_def))) + float(np.sum(np.asarray(E_ei_dn_def))) + E_ee_def
        ewald_alt = build_ewald_data(
            structure,
            coulomb,
            tol=1e-10,
            eta_override=eta_factor * ewald_default.eta,
        )
        coulomb_alt = coulomb.replace(ewald_data=ewald_alt)
        E_ii_alt = float(compute_ewald_potential_ii(coulomb_alt, ewald_alt))
        E_ei_up_alt, E_ei_dn_alt = compute_ewald_potential_ei(coulomb_alt, r_up, r_dn, ewald_alt)
        E_ee_alt = float(compute_ewald_potential_ee(r_up, r_dn, ewald_alt))
        E_total_alt = E_ii_alt + float(np.sum(np.asarray(E_ei_up_alt))) + float(np.sum(np.asarray(E_ei_dn_alt))) + E_ee_alt
        atol, rtol = get_tolerance("ewald", "strict")
        np.testing.assert_allclose(E_total_alt, E_total_default, atol=atol, rtol=rtol)


class TestEcpPhysical:
    """ECP non-local PBC translation invariance + unwrapped projector."""

    @pytest.mark.light
    def test_ecp_non_local_pbc_lattice_translation_invariance(self):
        """Gamma-point non-local ECP energy is invariant under a lattice translation.

        Shifting every electron by a lattice vector ``T = (n1, n2, n3) . a_i``
        is the periodic-system identity transform: at the Gamma point the
        (real) wavefunction is strictly periodic, so the non-local ECP energy
        ``sum_V`` must be bit-for-bit invariant. The mesh-point
        ``_wrap_into_cell`` folds the shifted quadrature points back to the
        home cell before AO evaluation, so any wrap drop or mis-application
        in the NN-ECP kernel breaks this identity.

        Pins down the whole PBC non-local chain end to end:

        * ``_get_min_dist_rel_R_cart`` (nearest-image nucleus selection) must
          pick the same image geometry after the shift;
        * the quadrature-point ``_wrap_into_cell`` must fold ``r_q`` back to a
          position whose AO image-expansion value matches the un-shifted run;
        * the determinant / Jastrow ratios must be translation invariant.

        Uses the real ``silicon.h5`` PBC fixture (8-atom Si conventional cell,
        32 valence electrons, ccECP-ccpvtz). Tractable after the
        ``_primitive_image_radius`` atol-tolerance fix dropped ``n_image`` from
        ~6,900 to ~250 for diffuse Si p-orbitals.
        """
        trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "silicon.h5")
        if not os.path.exists(trexio_path):
            pytest.skip("Periodic TREXIO fixture silicon.h5 is not present.")
        atol, rtol = get_tolerance("coulomb", "strict")
        structure_data, _aos, _mos_up, _mos_dn, geminal_mo_data, coulomb_potential_data = read_trexio_file(
            trexio_file=trexio_path, store_tuple=True
        )
        cell_np = np.asarray(structure_data.cell, dtype=np.float64)
        T = np.array([1, 2, -1], dtype=np.float64) @ cell_np

        wavefunction_data = Wavefunction_data(
            geminal_data=geminal_mo_data,
            jastrow_data=Jastrow_data(
                jastrow_one_body_data=None,
                jastrow_two_body_data=None,
                jastrow_three_body_data=None,
            ),
        )
        # 2 up + 2 dn per Si: same node-avoidance pattern as the kinetic
        # invariance test. |Psi| ~ 1e-40 is fine -- only the ratio matters
        # for translation invariance (see the silicon kinetic-invariance
        # docstring for the normalisation explanation).
        n_up = geminal_mo_data.num_electron_up
        n_dn = geminal_mo_data.num_electron_dn
        atoms = np.asarray(structure_data.positions, dtype=np.float64)
        natom = atoms.shape[0]
        up_per_atom = n_up // natom
        dn_per_atom = n_dn // natom
        rng = np.random.default_rng(0)
        r_up_base = np.repeat(atoms, up_per_atom, axis=0) + rng.normal(scale=0.4, size=(n_up, 3))
        r_dn_base = np.repeat(atoms, dn_per_atom, axis=0) + rng.normal(scale=0.4, size=(n_dn, 3))
        r_up_tr = r_up_base + T
        r_dn_tr = r_dn_base + T

        RT = jnp.eye(3)

        def _sum_V(r_up_np, r_dn_np, NN):
            _mesh_up, _mesh_dn, _V, sum_V = compute_ecp_non_local_parts_nearest_neighbors(
                coulomb_potential_data=coulomb_potential_data,
                wavefunction_data=wavefunction_data,
                r_up_carts=jnp.array(r_up_np),
                r_dn_carts=jnp.array(r_dn_np),
                Nv=6,
                NN=NN,
                RT=RT,
            )
            return np.asarray(sum_V)

        # NN=1 only: each additional NN value triggers a JIT re-compile of the
        # NN-static kernel, and silicon's ccECP-ccpvtz HLO is large enough that
        # the full ``range(1, natom+1)`` sweep would stack 8 multi-minute
        # compilations. NN=1 is the dominant non-local contribution and the
        # wrap behaviour is identical for every NN.
        sum_base = _sum_V(r_up_base, r_dn_base, NN=1)
        sum_tr = _sum_V(r_up_tr, r_dn_tr, NN=1)

        assert not np.any(np.isnan(sum_base)), "NaN in base sum_V"
        assert not np.any(np.isnan(sum_tr)), "NaN in translated sum_V"
        np.testing.assert_allclose(sum_tr, sum_base, atol=atol, rtol=rtol)

        jax.clear_caches()

    @pytest.mark.light
    def test_ecp_non_local_pbc_projector_unwrapped_vs_molecular_image(self):
        """PBC non-local ECP keeps the projector rel-R UNWRAPPED (minimum image).

        Independent oracle for the rule that the ECP projector argument ``rels``
        (the nucleus->electron minimum-image vector feeding ``V_l``, ``cos_theta`` and
        the sphere radius) must NOT be folded into the home cell, while the
        quadrature evaluation point IS wrapped. The production-vs-debug test
        cannot catch a wrong wrap here because both paths share the same wrap
        convention; this test uses a wrap-independent molecular reference instead.

        Construction: a single ECP atom sits near the home-cell corner and the
        electrons sit near the OPPOSITE corner (frac ~ 0.95). The minimum-image
        nucleus is therefore the periodic image at ``R + (L,L,L)``, only ~0.09*L
        away, not the home atom ~0.86*L away. The correct (unwrapped-projector)
        PBC energy must equal the *molecular* energy of the same electrons with a
        single atom placed at that image position:

            sum_V_pbc(atom@home, min-image)  ==  sum_V_mol(atom@image)        (1)

        A kernel that wrongly wrapped ``rels`` into the home cell would instead
        reproduce the molecular energy of the atom at the HOME position:

            sum_V_pbc_buggy                  ==  sum_V_mol(atom@home)         (2)

        The two molecular references differ by orders of magnitude (the ECP
        ``|rel|^-2 exp(-a|rel|^2)`` radial factor), so we assert (1) holds at
        strict tolerance AND that the home-vs-image references are far apart ---
        i.e. the test has teeth and would fail under a projector wrap.
        """
        atol, rtol = get_tolerance("coulomb", "medium")

        L = 8.0
        cell = np.diag([L, L, L]).astype(np.float64)
        atom_home = np.array([0.05, 0.05, 0.05]) * L
        atom_image = atom_home + np.array([L, L, L])

        # Electrons sit near the OPPOSITE corner so their minimum-image nearest
        # nucleus is ``atom_image`` (~0.1*L*sqrt(3) away), not the home atom
        # (~0.86*L*sqrt(3) away). The two same-spin electrons are kept well
        # separated (no Pauli node) and a few tenths of a Bohr from the image so
        # Psi stays away from the nodal surface.
        e_anchor = np.array([0.95, 0.95, 0.95]) * L
        r_up = e_anchor + np.array([[0.45, -0.30, 0.15], [-0.40, 0.50, -0.35]], dtype=np.float64)
        r_dn = e_anchor + np.array([[0.30, 0.20, -0.45], [-0.35, -0.25, 0.40]], dtype=np.float64)

        # --- PBC system: atom at home corner, projector must pick the image ---
        pbc_structure = make_pbc_structure(
            cell,
            positions=np.array([atom_home], dtype=np.float64),
            atomic_numbers=(6,),
            element_symbols=("C",),
            atomic_labels=("C1",),
        )
        geminal_pbc = _build_nonsingular_agp(pbc_structure, [atom_home])
        # Node-avoidance guard (|Psi| away from 0): det must be O(1e-4), not O(1e-18).
        det_pbc = float(np.asarray(compute_det_geminal_all_elements(geminal_pbc, jnp.array(r_up), jnp.array(r_dn))))
        assert abs(det_pbc) > 1.0e-8, "PBC WF too close to node; adjust fixture"
        sum_V_pbc = _ecp_non_local_sum_V(pbc_structure, geminal_pbc, r_up, r_dn)

        # --- molecular reference: single atom at the IMAGE position (correct) ---
        mol_image_structure = Structure_data(
            pbc_flag=False,
            positions=np.array([atom_image], dtype=np.float64),
            atomic_numbers=(6,),
            element_symbols=("C",),
            atomic_labels=("C1",),
        )
        geminal_mol_image = _build_nonsingular_agp(mol_image_structure, [atom_image])
        sum_V_mol_image = _ecp_non_local_sum_V(mol_image_structure, geminal_mol_image, r_up, r_dn)

        # --- molecular reference: single atom at the HOME position (wrong-wrap) ---
        mol_home_structure = Structure_data(
            pbc_flag=False,
            positions=np.array([atom_home], dtype=np.float64),
            atomic_numbers=(6,),
            element_symbols=("C",),
            atomic_labels=("C1",),
        )
        geminal_mol_home = _build_nonsingular_agp(mol_home_structure, [atom_home])
        sum_V_mol_home = _ecp_non_local_sum_V(mol_home_structure, geminal_mol_home, r_up, r_dn)

        # (1) correct PBC == molecular-at-image (projector kept unwrapped).
        # ``medium`` (not ``strict``) tolerance because the PBC wavefunction sums a
        # truncated AO image list (``AO_IMAGE_TRUNCATION_ATOL``) while the molecular
        # reference has a single atom, so the agreement floor is the image-list
        # truncation, not machine precision.
        assert not np.isnan(sum_V_pbc) and not np.isnan(sum_V_mol_image)
        np.testing.assert_allclose(sum_V_pbc, sum_V_mol_image, atol=atol, rtol=rtol)

        # Teeth: the wrong-wrap reference (atom at home) must disagree with the PBC
        # result by far more than the (1) agreement margin, so a projector wrap
        # regression (which would make PBC match the home atom) could not slip past
        # assertion (1). Here ``mol_home`` is ~50 orders of magnitude smaller than
        # the PBC value because the home atom is ~0.86*L*sqrt(3) away.
        agreement_margin = atol + rtol * abs(sum_V_pbc)
        assert abs(sum_V_pbc - sum_V_mol_home) > 1.0e3 * agreement_margin, (
            "home-vs-image references are not well separated; the oracle has no teeth "
            f"(pbc={sum_V_pbc:.3e}, mol_home={sum_V_mol_home:.3e})"
        )

        jax.clear_caches()


class TestEwaldExternal:
    """external reference comparisons (pymatgen + Madelung constant)."""

    @pytest.mark.light
    def test_compute_ewald_potential_ii_vs_pymatgen_nacl_sc(self):
        """``ewald_data.ii_energy`` matches an independent pymatgen Ewald sum.

        The 8-atom NaCl-type SC fixture (``a = 2`` Bohr, ``Z = +-1``,
        nearest-neighbor distance ``1`` Bohr) is charge-neutral, so
        pymatgen's ``EwaldSummation.total_energy`` is directly comparable
        to jqmc's ``ewald_data.ii_energy`` after unit conversion. Unlike
        ``test_madelung_sc`` (validates against a literature *constant*),
        this test validates against an independent *implementation* of
        Ewald summation.

        The reference value ``PYMATGEN_E_II_NACL_SC_HA`` is regenerated
        offline. The test environment does not require pymatgen as a
        dependency; run the following script outside the test env and
        paste the printed value into ``PYMATGEN_E_II_NACL_SC_HA`` above::

            # The EwaldSummation API is stable across pymatgen 2024+; the
            # pin below is a known-good version. Newer releases also work.
            pip install pymatgen==2024.6.10

        .. code-block:: python

            import numpy as np
            from importlib.metadata import version as _pkg_version
            from pymatgen.core import Lattice, Structure
            from pymatgen.analysis.ewald import EwaldSummation

            # CODATA 2018.
            BOHR_TO_ANG = 0.529177210903
            HA_TO_EV = 27.211386245988

            a_bohr = 2.0
            positions_bohr = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            )
            species = ["Na"] * 4 + ["Cl"] * 4
            oxi_states = [+1.0] * 4 + [-1.0] * 4

            lattice = Lattice.cubic(a_bohr * BOHR_TO_ANG)
            struct = Structure(
                lattice,
                species,
                positions_bohr * BOHR_TO_ANG,
                coords_are_cartesian=True,
            )
            struct.add_oxidation_state_by_site(oxi_states)

            # ``acc_factor=18`` drives pymatgen's truncation error well below
            # the ``ATOL_EWALD_VS_EXTERNAL_REFERENCE`` floor of the comparison.
            ewald = EwaldSummation(struct, acc_factor=18.0)
            e_ha = float(ewald.total_energy) / HA_TO_EV
            print(f"PYMATGEN_E_II_NACL_SC_HA = {e_ha:.10f}")
            print(f"pymatgen version: {_pkg_version('pymatgen')}")

        Expected magnitude: the literature Madelung constant gives
        ``E_ii = -alpha_M * 4 = -6.9902583784`` Ha for this fixture, so
        pymatgen should print a value close to that.
        """

        structure, coulomb = _make_nacl_sc_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        np.testing.assert_allclose(
            float(ewald_data.ii_energy),
            PYMATGEN_E_II_NACL_SC_HA,
            atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE,
            rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE,
        )

    @pytest.mark.light
    def test_compute_ewald_potential_ii_vs_pymatgen_mgo_b1(self):
        """``ewald_data.ii_energy`` matches pymatgen on MgO B1 (rocksalt, Z=+-2).

        Same 8-atom rocksalt sublattice as
        :py:func:`test_compute_ewald_potential_ii_vs_pymatgen_nacl_sc` but with
        the experimental MgO lattice constant ``a = 4.211`` Angstrom and
        ``Z = +-2`` charges. The Z**2 dependence in both the self-energy
        (``-(eta/sqrt(pi)) * sum_I Z_I**2``) and the pair sum means a
        linear-in-Z bug would yield a wrong ratio against the NaCl reference;
        pymatgen's independent implementation pins it down.

        Reference regeneration (run outside the test env)::

            pip install pymatgen==2024.6.10

        .. code-block:: python

            from importlib.metadata import version as _pkg_version
            from pymatgen.core import Lattice, Structure
            from pymatgen.analysis.ewald import EwaldSummation

            HA_TO_EV = 27.211386245988

            a_ang = 4.211  # experimental MgO B1 lattice constant
            lattice = Lattice.cubic(a_ang)
            species = ["Mg"] * 4 + ["O"] * 4
            frac_coords = [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.5, 0.5],
            ]
            oxi_states = [+2.0] * 4 + [-2.0] * 4

            struct = Structure(lattice, species, frac_coords)
            struct.add_oxidation_state_by_site(oxi_states)

            ewald = EwaldSummation(struct, acc_factor=18.0)
            e_ha = float(ewald.total_energy) / HA_TO_EV
            print(f"PYMATGEN_E_II_MGO_B1_HA = {e_ha:.10f}")
            print(f"pymatgen version: {_pkg_version('pymatgen')}")
        """

        # CODATA 2018: 1 Bohr = 0.529177210903 Angstrom.
        BOHR_PER_ANG = 1.0 / 0.529177210903
        a_bohr = 4.211 * BOHR_PER_ANG
        half = a_bohr / 2.0
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [half, half, 0.0],
                [half, 0.0, half],
                [0.0, half, half],
                [half, 0.0, 0.0],
                [0.0, half, 0.0],
                [0.0, 0.0, half],
                [half, half, half],
            ],
            dtype=np.float64,
        )
        structure = Structure_data.build(
            pbc_flag=True,
            positions=positions,
            vec_a=(a_bohr, 0.0, 0.0),
            vec_b=(0.0, a_bohr, 0.0),
            vec_c=(0.0, 0.0, a_bohr),
            atomic_numbers=(2, 2, 2, 2, -2, -2, -2, -2),
            element_symbols=("Mg",) * 4 + ("O",) * 4,
            atomic_labels=("Mg1", "Mg2", "Mg3", "Mg4", "O1", "O2", "O3", "O4"),
        )
        coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        np.testing.assert_allclose(
            float(ewald_data.ii_energy),
            PYMATGEN_E_II_MGO_B1_HA,
            atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE,
            rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE,
        )

    @pytest.mark.light
    def test_compute_ewald_potential_ii_vs_pymatgen_zno_wurtzite(self):
        """``ewald_data.ii_energy`` matches pymatgen on wurtzite ZnO (Z=+-2).

        Wurtzite ZnO is a real binary ionic crystal with a hexagonal
        (non-orthorhombic) primitive cell of 4 atoms (2 Zn + 2 O). The
        test pins down cell-shape handling end-to-end: lattice-image
        enumeration, G-vector enumeration, and eta iteration must all
        handle non-orthogonal cell vectors correctly. Any cubic-only
        assumption (e.g. ``L_min = a`` shortcut) along that chain would
        give a different answer here.

        Lattice parameters used: experimental ZnO ``a = 3.2495`` Angstrom,
        ``c = 5.2069`` Angstrom, internal parameter ``u = 0.382``.

        Reference regeneration (run outside the test env)::

            pip install pymatgen==2024.6.10

        .. code-block:: python

            from importlib.metadata import version as _pkg_version
            from pymatgen.core import Lattice, Structure
            from pymatgen.analysis.ewald import EwaldSummation

            HA_TO_EV = 27.211386245988

            a_ang = 3.2495
            c_ang = 5.2069
            u = 0.382

            lattice = Lattice.hexagonal(a_ang, c_ang)
            species = ["Zn", "Zn", "O", "O"]
            frac_coords = [
                [1.0 / 3.0, 2.0 / 3.0, 0.0],
                [2.0 / 3.0, 1.0 / 3.0, 0.5],
                [1.0 / 3.0, 2.0 / 3.0, u],
                [2.0 / 3.0, 1.0 / 3.0, 0.5 + u],
            ]
            oxi_states = [+2.0, +2.0, -2.0, -2.0]

            struct = Structure(lattice, species, frac_coords)
            struct.add_oxidation_state_by_site(oxi_states)

            ewald = EwaldSummation(struct, acc_factor=18.0)
            e_ha = float(ewald.total_energy) / HA_TO_EV
            print(f"PYMATGEN_E_II_ZNO_WURTZITE_HA = {e_ha:.10f}")
            print(f"pymatgen version: {_pkg_version('pymatgen')}")
        """

        # CODATA 2018: 1 Bohr = 0.529177210903 Angstrom.
        BOHR_PER_ANG = 1.0 / 0.529177210903
        a_bohr = 3.2495 * BOHR_PER_ANG
        c_bohr = 5.2069 * BOHR_PER_ANG
        u = 0.382

        sqrt3 = float(np.sqrt(3.0))
        vec_a = (a_bohr, 0.0, 0.0)
        vec_b = (-a_bohr / 2.0, a_bohr * sqrt3 / 2.0, 0.0)
        vec_c = (0.0, 0.0, c_bohr)
        lattice_matrix = np.array([vec_a, vec_b, vec_c], dtype=np.float64)

        frac_coords = np.array(
            [
                [1.0 / 3.0, 2.0 / 3.0, 0.0],
                [2.0 / 3.0, 1.0 / 3.0, 0.5],
                [1.0 / 3.0, 2.0 / 3.0, u],
                [2.0 / 3.0, 1.0 / 3.0, 0.5 + u],
            ],
            dtype=np.float64,
        )
        positions = frac_coords @ lattice_matrix

        structure = Structure_data.build(
            pbc_flag=True,
            positions=positions,
            vec_a=vec_a,
            vec_b=vec_b,
            vec_c=vec_c,
            atomic_numbers=(2, 2, -2, -2),
            element_symbols=("Zn", "Zn", "O", "O"),
            atomic_labels=("Zn1", "Zn2", "O1", "O2"),
        )
        coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        np.testing.assert_allclose(
            float(ewald_data.ii_energy),
            PYMATGEN_E_II_ZNO_WURTZITE_HA,
            atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE,
            rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE,
        )

    @pytest.mark.light
    def test_compute_ewald_potential_ii_vs_pymatgen_cuo_tenorite(self):
        """``ewald_data.ii_energy`` matches pymatgen on monoclinic CuO tenorite.

        CuO (tenorite, space group C2/c) is a real binary ionic crystal with
        a truly monoclinic conventional cell: ``beta = 99.54`` deg means
        ``vec_c = (c cos beta, 0, c sin beta)`` has a non-zero x-component, so
        the lattice vectors are not pairwise orthogonal AND ``vec_c`` is not
        aligned with z. This is the most demanding cell-shape test in
        Section (2): any orthorhombic assumption surviving in the chain
        (e.g. ``vec_c = (0, 0, c)``, ``L_min == min(a,b,c)``) shows up here.

        Lattice parameters used: experimental tenorite ``a = 4.6837`` Angstrom,
        ``b = 3.4226`` Angstrom, ``c = 5.1288`` Angstrom, ``beta = 99.54`` deg.
        Atomic positions: 4 Cu at Wyckoff 4c, 4 O at Wyckoff 4e with internal
        parameter ``y = 0.4184``.

        Reference regeneration (run outside the test env)::

            pip install pymatgen==2024.6.10

        .. code-block:: python

            from importlib.metadata import version as _pkg_version
            from pymatgen.core import Lattice, Structure
            from pymatgen.analysis.ewald import EwaldSummation

            HA_TO_EV = 27.211386245988

            a_ang = 4.6837
            b_ang = 3.4226
            c_ang = 5.1288
            beta_deg = 99.54
            y = 0.4184

            lattice = Lattice.monoclinic(a_ang, b_ang, c_ang, beta_deg)
            species = ["Cu"] * 4 + ["O"] * 4
            frac_coords = [
                [0.25, 0.25, 0.00],
                [0.25, 0.75, 0.50],
                [0.75, 0.75, 0.00],
                [0.75, 0.25, 0.50],
                [0.00, y, 0.25],
                [0.00, 1.0 - y, 0.75],
                [0.50, y + 0.5, 0.25],
                [0.50, 0.5 - y, 0.75],
            ]
            oxi_states = [+2.0] * 4 + [-2.0] * 4

            struct = Structure(lattice, species, frac_coords)
            struct.add_oxidation_state_by_site(oxi_states)

            ewald = EwaldSummation(struct, acc_factor=18.0)
            e_ha = float(ewald.total_energy) / HA_TO_EV
            print(f"PYMATGEN_E_II_CUO_TENORITE_HA = {e_ha:.10f}")
            print(f"pymatgen version: {_pkg_version('pymatgen')}")
        """

        # CODATA 2018: 1 Bohr = 0.529177210903 Angstrom.
        BOHR_PER_ANG = 1.0 / 0.529177210903
        a_bohr = 4.6837 * BOHR_PER_ANG
        b_bohr = 3.4226 * BOHR_PER_ANG
        c_bohr = 5.1288 * BOHR_PER_ANG
        beta_rad = float(np.deg2rad(99.54))
        y = 0.4184

        # Monoclinic with b-unique convention: beta is between vec_a and vec_c.
        vec_a = (a_bohr, 0.0, 0.0)
        vec_b = (0.0, b_bohr, 0.0)
        vec_c = (c_bohr * float(np.cos(beta_rad)), 0.0, c_bohr * float(np.sin(beta_rad)))
        lattice_matrix = np.array([vec_a, vec_b, vec_c], dtype=np.float64)

        frac_coords = np.array(
            [
                [0.25, 0.25, 0.00],
                [0.25, 0.75, 0.50],
                [0.75, 0.75, 0.00],
                [0.75, 0.25, 0.50],
                [0.00, y, 0.25],
                [0.00, 1.0 - y, 0.75],
                [0.50, y + 0.5, 0.25],
                [0.50, 0.5 - y, 0.75],
            ],
            dtype=np.float64,
        )
        positions = frac_coords @ lattice_matrix

        structure = Structure_data.build(
            pbc_flag=True,
            positions=positions,
            vec_a=vec_a,
            vec_b=vec_b,
            vec_c=vec_c,
            atomic_numbers=(2, 2, 2, 2, -2, -2, -2, -2),
            element_symbols=("Cu",) * 4 + ("O",) * 4,
            atomic_labels=("Cu1", "Cu2", "Cu3", "Cu4", "O1", "O2", "O3", "O4"),
        )
        coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        np.testing.assert_allclose(
            float(ewald_data.ii_energy),
            PYMATGEN_E_II_CUO_TENORITE_HA,
            atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE,
            rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE,
        )

    @pytest.mark.light
    def test_compute_ewald_potential_ei_vs_pymatgen_h2_box(self):
        """``sum(compute_ewald_potential_ei)`` matches an independent pymatgen
        Ewald reference on the all-electron H2-in-box fixture.

        Unlike the ion-ion tests, pymatgen has no electron-ion channel of its
        own, so the reference is carved out of
        ``EwaldSummation.total_energy_matrix`` --- the N x N pairwise energy
        matrix over *both* the ions and the electrons (modelled as ``Z = -1``
        point charges). The electron-ion block (ions x electrons, both
        off-diagonal halves) is summed and jqmc's per-channel jellium
        background is added back using pymatgen's own ``eta``. The matrix
        deliberately omits the charged-cell term, which is nonzero for each
        block in isolation even though the e-i / e-e / i-i terms cancel for
        the neutral whole; restoring it makes the channel eta-invariant, so
        the reconstruction matches jqmc even though the two codes pick
        different ``eta`` (pymatgen ``eta`` is the *square* of jqmc's: pymatgen
        uses ``erfc(sqrt(eta) r)`` / ``exp(-G^2/(4 eta))``).

        The reference values ``PYMATGEN_E_EI_H2_BOX_HA`` /
        ``PYMATGEN_E_EE_H2_BOX_HA`` are regenerated offline; the test
        environment does not require pymatgen. Run the following outside the
        test env and paste the printed values into the constants above::

            pip install pymatgen==2024.6.10

        .. code-block:: python

            import numpy as np
            from importlib.metadata import version as _pkg_version
            from pymatgen.core import Lattice, Structure
            from pymatgen.analysis.ewald import EwaldSummation

            # CODATA 2018.
            BOHR_TO_ANG = 0.529177210903
            HA_TO_EV = 27.211386245988

            a_bohr = 10.0
            ion_pos_bohr = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
            elec_pos_bohr = np.array([[0.1, 0.2, -0.6], [-0.1, -0.2, 0.5]])  # r_up, r_dn
            positions_bohr = np.vstack([ion_pos_bohr, elec_pos_bohr])
            oxi_states = [+1.0, +1.0, -1.0, -1.0]  # ions then electrons
            n_ion = ion_pos_bohr.shape[0]
            species = ["H"] * len(positions_bohr)  # identity irrelevant; only charge matters

            lattice = Lattice.cubic(a_bohr * BOHR_TO_ANG)
            struct = Structure(lattice, species, positions_bohr * BOHR_TO_ANG, coords_are_cartesian=True)
            struct.add_oxidation_state_by_site(oxi_states)

            # ``acc_factor=18`` drives pymatgen's truncation error well below
            # the ``ATOL_EWALD_VS_EXTERNAL_REFERENCE`` floor of the comparison.
            ewald = EwaldSummation(struct, acc_factor=18.0)
            M = np.asarray(ewald.total_energy_matrix)  # eV, (recip + real), self on diagonal

            ion, ele = slice(0, n_ion), slice(n_ion, None)
            block_ii = M[ion, ion].sum()
            block_ee = M[ele, ele].sum()
            block_ei = M[ion, ele].sum() + M[ele, ion].sum()

            # jellium background (jqmc convention) with pymatgen's eta (= eta_jqmc^2).
            CONV_FACT = EwaldSummation.CONV_FACT
            bg_pref = -CONV_FACT / 2.0 * np.pi / struct.volume / ewald.eta  # eV per charge^2
            Q_ion = sum(oxi_states[:n_ion])
            N_e = -sum(oxi_states[n_ion:])
            bg_ii = bg_pref * Q_ion**2
            bg_ee = bg_pref * N_e**2
            bg_ei = -2.0 * bg_pref * Q_ion * N_e

            print(f"PYMATGEN_E_EI_H2_BOX_HA = {(block_ei + bg_ei) / HA_TO_EV:.10f}")
            print(f"PYMATGEN_E_EE_H2_BOX_HA = {(block_ee + bg_ee) / HA_TO_EV:.10f}")
            print(f"# E_ii (cross-check)    = {(block_ii + bg_ii) / HA_TO_EV:.10f}")
            print(f"# E_total (cross-check) = {M.sum() / HA_TO_EV:.10f}")
            print(f"pymatgen version: {_pkg_version('pymatgen')}")

        Expected magnitude: the total ``E_ii + sum(E_ei) + E_ee`` is the same
        neutral-cell Coulomb energy validated by
        :py:func:`test_compute_ewald_potential_total_energy_eta_invariance_h2_box`.
        """

        structure, coulomb, r_up, r_dn = _make_h2_box_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        E_ei_up, E_ei_dn = compute_ewald_potential_ei(coulomb, r_up, r_dn, ewald_data)
        E_ei = float(np.sum(np.asarray(E_ei_up)) + np.sum(np.asarray(E_ei_dn)))
        np.testing.assert_allclose(
            E_ei,
            PYMATGEN_E_EI_H2_BOX_HA,
            atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE,
            rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE,
        )

    @pytest.mark.light
    def test_compute_ewald_potential_ee_vs_pymatgen_h2_box(self):
        """``compute_ewald_potential_ee`` matches the pymatgen e-e block on the
        H2-in-box fixture.

        Same ``total_energy_matrix`` decomposition as
        :py:func:`test_compute_ewald_potential_ei_vs_pymatgen_h2_box` (see its
        docstring for the shared regeneration script); here the electron-
        electron block (with the per-electron self-energy on the diagonal) is
        summed and given jqmc's ``-(pi / (2 eta^2 V)) N_e^2`` jellium
        background. The single up + single dn electron makes this the
        minimal pair-plus-self e-e reference.
        """

        structure, coulomb, r_up, r_dn = _make_h2_box_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        E_ee = float(compute_ewald_potential_ee(r_up, r_dn, ewald_data))
        np.testing.assert_allclose(
            E_ee,
            PYMATGEN_E_EE_H2_BOX_HA,
            atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE,
            rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE,
        )

    @pytest.mark.light
    def test_compute_ewald_potential_ei_vs_pymatgen_h2_monoclinic(self):
        """``sum(compute_ewald_potential_ei)`` matches pymatgen on the monoclinic
        (non-orthorhombic) H2 fixture.

        Same ``total_energy_matrix`` block decomposition as
        :py:func:`test_compute_ewald_potential_ei_vs_pymatgen_h2_box`; the only
        change is the lattice -- a ``beta = 110`` deg monoclinic cell whose
        ``vec_c`` carries both an x- and a z-component. This drives the e-i
        Ewald path through the non-orthogonal cell-shape handling
        (:py:func:`_minimum_image_displacement`, ``g_vectors``, real-image
        enumeration, ``eta`` heuristic), the analogue of what
        :py:func:`test_compute_ewald_potential_ii_vs_pymatgen_cuo_tenorite`
        does for the ion-ion channel.

        Reference regeneration (run outside the test env)::

            pip install pymatgen==2024.6.10

        .. code-block:: python

            import numpy as np
            from importlib.metadata import version as _pkg_version
            from pymatgen.core import Lattice, Structure
            from pymatgen.analysis.ewald import EwaldSummation

            # CODATA 2018.
            BOHR_TO_ANG = 0.529177210903
            HA_TO_EV = 27.211386245988

            a_bohr, b_bohr, c_bohr = 10.0, 11.0, 12.0
            beta = np.deg2rad(110.0)
            L = np.array([
                [a_bohr, 0.0, 0.0],
                [0.0, b_bohr, 0.0],
                [c_bohr * np.cos(beta), 0.0, c_bohr * np.sin(beta)],
            ])
            ion_frac = np.array([[0.40, 0.40, 0.40], [0.60, 0.55, 0.55]])
            elec_frac = np.array([[0.25, 0.30, 0.30], [0.70, 0.65, 0.70]])  # r_up, r_dn
            positions_bohr = np.vstack([ion_frac @ L, elec_frac @ L])
            oxi_states = [+1.0, +1.0, -1.0, -1.0]  # ions then electrons
            n_ion = 2
            species = ["H"] * len(positions_bohr)  # identity irrelevant; only charge matters

            lattice = Lattice(L * BOHR_TO_ANG)
            struct = Structure(lattice, species, positions_bohr * BOHR_TO_ANG, coords_are_cartesian=True)
            struct.add_oxidation_state_by_site(oxi_states)

            ewald = EwaldSummation(struct, acc_factor=18.0)
            M = np.asarray(ewald.total_energy_matrix)  # eV, (recip + real), self on diagonal

            ion, ele = slice(0, n_ion), slice(n_ion, None)
            block_ii = M[ion, ion].sum()
            block_ee = M[ele, ele].sum()
            block_ei = M[ion, ele].sum() + M[ele, ion].sum()

            # jellium background (jqmc convention) with pymatgen's eta (= eta_jqmc^2).
            CONV_FACT = EwaldSummation.CONV_FACT
            bg_pref = -CONV_FACT / 2.0 * np.pi / struct.volume / ewald.eta  # eV per charge^2
            Q_ion = sum(oxi_states[:n_ion])
            N_e = -sum(oxi_states[n_ion:])
            bg_ii = bg_pref * Q_ion**2
            bg_ee = bg_pref * N_e**2
            bg_ei = -2.0 * bg_pref * Q_ion * N_e

            print(f"PYMATGEN_E_EI_H2_MONO_HA = {(block_ei + bg_ei) / HA_TO_EV:.10f}")
            print(f"PYMATGEN_E_EE_H2_MONO_HA = {(block_ee + bg_ee) / HA_TO_EV:.10f}")
            print(f"# E_ii (cross-check)     = {(block_ii + bg_ii) / HA_TO_EV:.10f}")
            print(f"# E_total (cross-check)  = {M.sum() / HA_TO_EV:.10f}")
            print(f"pymatgen version: {_pkg_version('pymatgen')}")
        """

        structure, coulomb, r_up, r_dn = _make_h2_monoclinic_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        E_ei_up, E_ei_dn = compute_ewald_potential_ei(coulomb, r_up, r_dn, ewald_data)
        E_ei = float(np.sum(np.asarray(E_ei_up)) + np.sum(np.asarray(E_ei_dn)))
        np.testing.assert_allclose(
            E_ei,
            PYMATGEN_E_EI_H2_MONO_HA,
            atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE,
            rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE,
        )

    @pytest.mark.light
    def test_compute_ewald_potential_ee_vs_pymatgen_h2_monoclinic(self):
        """``compute_ewald_potential_ee`` matches the pymatgen e-e block on the
        monoclinic (non-orthorhombic) H2 fixture.

        Same ``total_energy_matrix`` decomposition as
        :py:func:`test_compute_ewald_potential_ei_vs_pymatgen_h2_monoclinic`
        (see its docstring for the shared regeneration script); here the
        electron-electron block (with the per-electron self-energy on the
        diagonal) is summed and given jqmc's ``-(pi / (2 eta^2 V)) N_e^2``
        jellium background. Exercises the e-e Ewald path under a non-orthogonal
        lattice.
        """

        structure, coulomb, r_up, r_dn = _make_h2_monoclinic_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        E_ee = float(compute_ewald_potential_ee(r_up, r_dn, ewald_data))
        np.testing.assert_allclose(
            E_ee,
            PYMATGEN_E_EE_H2_MONO_HA,
            atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE,
            rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE,
        )

    @pytest.mark.light
    def test_compute_ewald_potential_ii_madelung_constant_nacl_sc(self):
        """Ion-ion Ewald on an 8-atom NaCl-type SC reproduces ``alpha_M = 1.7475645946``.

        ``R_NN = 1`` and ``Z = +-1`` give a Madelung energy per ion of
        ``-alpha_M`` and a total cell energy of ``-4 alpha_M`` (the 1/2
        factor from the unordered pair sum). Inverting:
        ``alpha_M_test = -E_ii / 4``.

        Tolerance ``1e-12`` in :py:func:`build_ewald_data` keeps per-channel
        truncation below ``1e-12``; literature ``alpha_M`` is accurate to
        >= 10 digits, so agreement is asserted at ``ATOL_EWALD_VS_EXTERNAL_REFERENCE``.
        """
        structure, coulomb = _make_nacl_sc_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        E_ii = float(compute_ewald_potential_ii(coulomb, ewald_data))
        alpha_M_test = -E_ii / 4.0
        np.testing.assert_allclose(
            alpha_M_test, ALPHA_M_NACL, atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE, rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE
        )


class TestEtaHeuristic:
    """eta selection heuristic + extreme anisotropy error."""

    @pytest.mark.light
    def test_eta_iteration_heuristic_30bohr_cubic_baseline(self):
        """Cost-minimising selector returns a consistent, uncapped ``eta`` on a
        30 Bohr cubic cell at ``tol = 1e-10``.

        New contract (QE-style rewrite): ``eta`` is chosen by minimising the
        estimated per-evaluation cost over a log-grid, and the cutoffs are
        *pinned* to it by the truncation tolerance,
        ``r_real_cut = sqrt(-ln tol) / eta`` and
        ``g_max = 2 eta sqrt(-ln tol)``. There is no ``r_real_cut <= L_min``
        cap and no scalar ``G``-budget gate. Asserts positivity and
        the two pinned relations rather than a specific numeric ``eta`` (which
        depends on the cost model and the ion/electron counts).
        """
        cell = np.diag([30.0, 30.0, 30.0]).astype(np.float64)
        tol = 1e-10
        eta, r_real_cut, g_max = _eta_iteration_heuristic(cell, tol=tol, eta_override=None)
        sqrt_log = float(np.sqrt(-np.log(tol)))
        assert eta > 0.0
        assert np.isfinite(r_real_cut) and r_real_cut > 0.0
        np.testing.assert_allclose(r_real_cut, sqrt_log / eta, atol=ATOL_BIT_EXACT, rtol=RTOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(g_max, 2.0 * eta * sqrt_log, atol=ATOL_BIT_EXACT, rtol=RTOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    def test_eta_iteration_heuristic_override_skips_selection(self):
        """``eta_override`` bypasses the cost scan and returns the caller's eta.

        Required by the eta-invariance test pattern and the checkpoint-reload
        path: the caller can dial ``eta`` and receive the cutoffs pinned to it
        by the truncation tolerance, so the total Ewald energy stays invariant.
        """
        cell = np.diag([10.0, 10.0, 10.0]).astype(np.float64)
        tol = 1e-10
        eta_target = 0.7
        eta, r_real_cut, g_max = _eta_iteration_heuristic(cell, tol=tol, eta_override=eta_target)
        assert eta == eta_target
        sqrt_log = float(np.sqrt(-np.log(tol)))
        np.testing.assert_allclose(r_real_cut, sqrt_log / eta_target, atol=ATOL_BIT_EXACT, rtol=RTOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(g_max, 2.0 * eta_target * sqrt_log, atol=ATOL_BIT_EXACT, rtol=RTOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    @pytest.mark.parametrize("eta_factor", [0.5, 2.0])
    def test_eta_iteration_heuristic_handles_extreme_anisotropy(self, eta_factor: float):
        """A strongly anisotropic (vacuum) cell BUILDS instead of raising,
        and the built Ewald energy is eta-invariant.

        ``_eta_iteration_heuristic`` has no ``r_real_cut <= L_min`` cap and no
        scalar ``G``-budget: the selector returns finite, consistent cutoffs for
        any positive ``eta``, and the real-space image enumeration captures
        every in-plane image the vacuum cell needs. On an aspect-ratio-10 cell
        the test asserts success + eta-invariance of ``E_ii + E_ei + E_ee`` on a
        neutral point-charge fixture.
        """
        a, c = 4.0, 40.0  # aspect ratio 10, vacuum along z
        structure = Structure_data.build(
            pbc_flag=True,
            positions=np.array([[0.0, 0.0, 0.0], [a / 2.0, a / 2.0, 0.3]], dtype=np.float64),
            vec_a=(a, 0.0, 0.0),
            vec_b=(0.0, a, 0.0),
            vec_c=(0.0, 0.0, c),
            atomic_numbers=(1, 1),
            element_symbols=("H", "H"),
            atomic_labels=("H1", "H2"),
        )
        coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
        r_up = np.array([[0.3, 0.4, -0.5]], dtype=np.float64)
        r_dn = np.array([[a / 2.0 - 0.2, a / 2.0 + 0.1, 0.6]], dtype=np.float64)

        # No raise at tol=1e-12 (the old failure point).
        eta0, r_cut0, g_max0 = _eta_iteration_heuristic(
            np.asarray(structure.cell, dtype=np.float64),
            tol=1e-12,
            eta_override=None,  # tol-ok: algorithm input, result asserted against named/pinned values
        )
        assert eta0 > 0.0 and np.isfinite(r_cut0) and np.isfinite(g_max0)

        ewald_default = build_ewald_data(structure, coulomb, tol=1e-10)
        coulomb_default = coulomb.replace(ewald_data=ewald_default)
        E_ii = float(compute_ewald_potential_ii(coulomb_default, ewald_default))
        E_ei_up, E_ei_dn = compute_ewald_potential_ei(coulomb_default, r_up, r_dn, ewald_default)
        E_ee = float(compute_ewald_potential_ee(r_up, r_dn, ewald_default))
        E_total_default = E_ii + float(np.sum(np.asarray(E_ei_up))) + float(np.sum(np.asarray(E_ei_dn))) + E_ee

        ewald_alt = build_ewald_data(structure, coulomb, tol=1e-10, eta_override=eta_factor * ewald_default.eta)
        coulomb_alt = coulomb.replace(ewald_data=ewald_alt)
        E_ii_alt = float(compute_ewald_potential_ii(coulomb_alt, ewald_alt))
        E_ei_up_alt, E_ei_dn_alt = compute_ewald_potential_ei(coulomb_alt, r_up, r_dn, ewald_alt)
        E_ee_alt = float(compute_ewald_potential_ee(r_up, r_dn, ewald_alt))
        E_total_alt = E_ii_alt + float(np.sum(np.asarray(E_ei_up_alt))) + float(np.sum(np.asarray(E_ei_dn_alt))) + E_ee_alt
        atol, rtol = get_tolerance("ewald", "strict")
        np.testing.assert_allclose(E_total_alt, E_total_default, atol=atol, rtol=rtol)


class TestEwaldEdgeCase:
    """edge cases: zero electrons + alat=0 fallback."""

    @pytest.mark.light
    def test_compute_ewald_potential_ee_zero_electrons_is_zero(self):
        """``N_e = 0`` (no electrons) trivially gives ``E_ee = 0``.

        Sanity check for the empty-input path: no pair contributions, no
        self-energy contribution (``-(eta/sqrt(pi)) * 0 = 0``), no
        reciprocal structure factor (``sum_i = 0``).
        """
        structure, coulomb, _, _ = _make_h2_box_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-10)
        r_up = np.zeros((0, 3), dtype=np.float64)
        r_dn = np.zeros((0, 3), dtype=np.float64)
        E_ee = float(compute_ewald_potential_ee(r_up, r_dn, ewald_data))
        atol, rtol = get_tolerance("ewald", "strict")
        np.testing.assert_allclose(E_ee, 0.0, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_compute_ewald_potential_ei_zero_electrons_returns_empty(self):
        """``N_e = 0`` returns per-electron arrays of shape ``(0,)``."""
        structure, coulomb, _, _ = _make_h2_box_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-10)
        r_up = np.zeros((0, 3), dtype=np.float64)
        r_dn = np.zeros((0, 3), dtype=np.float64)
        E_up, E_dn = compute_ewald_potential_ei(coulomb, r_up, r_dn, ewald_data)
        assert tuple(E_up.shape) == (0,)
        assert tuple(E_dn.shape) == (0,)
        np.testing.assert_array_equal(np.asarray(E_up), np.empty((0,), dtype=np.float64))
        np.testing.assert_array_equal(np.asarray(E_dn), np.empty((0,), dtype=np.float64))

    @pytest.mark.light
    def test_compute_ewald_potential_ei_discretized_alat_zero_matches_unregularized(self):
        """At ``alat = 0`` the discretised Ewald e-I reduces to
        :py:func:`compute_ewald_potential_ei` whenever no walker sits on a
        periodic image of a nucleus.

        The ``max(r, 0)`` reduction is a no-op for every ``r > 0``, and
        the H2 fixture electrons (``(0.1, 0.2, -0.6)`` / ``(-0.1, -0.2, 0.5)``)
        are not on any image of either proton (``+-0.7`` along z), so this
        cross-check pins the regularised path to the unregularised one.
        """
        structure, coulomb, r_up, r_dn = _make_h2_box_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-10)
        E_up_reg, E_dn_reg = compute_ewald_potential_ei_discretized(
            coulomb,
            r_up,
            r_dn,
            ewald_data,
            alat=0.0,
        )
        E_up_ref, E_dn_ref = compute_ewald_potential_ei(coulomb, r_up, r_dn, ewald_data)
        atol, rtol = get_tolerance("ewald", "strict")
        np.testing.assert_allclose(np.asarray(E_up_reg), np.asarray(E_up_ref), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(E_dn_reg), np.asarray(E_dn_ref), atol=atol, rtol=rtol)


class TestEwaldError:
    """Ewald-builder error paths."""

    @pytest.mark.light
    def test_build_ewald_data_rejects_molecular_input(self):
        """:py:func:`build_ewald_data` raises on ``pbc_flag=False`` inputs.

        Enforces the strict contract: Ewald is only defined for
        periodic systems. The rejection happens at setup time before any
        heuristic work, so the error path is cheap.
        """
        structure = Structure_data(
            pbc_flag=False,
            positions=np.zeros((1, 3), dtype=np.float64),
            atomic_numbers=(1,),
            element_symbols=("H",),
            atomic_labels=("H1",),
        )
        coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
        with pytest.raises(ValueError, match="pbc_flag=True"):
            build_ewald_data(structure, coulomb, tol=1e-10)

    @staticmethod
    def _two_proton_cell() -> tuple[Structure_data, Coulomb_potential_data]:
        """All-electron H2 in a 10 Bohr cube: total ionic charge is exactly +2."""
        structure = make_pbc_structure(
            np.diag([10.0, 10.0, 10.0]),
            positions=np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]),
            atomic_numbers=(1, 1),
            element_symbols=("H", "H"),
            atomic_labels=("H1", "H2"),
        )
        return structure, Coulomb_potential_data(structure_data=structure, ecp_flag=False)

    @staticmethod
    def _silicon_ecp_cell() -> tuple[Structure_data, Coulomb_potential_data]:
        """Committed silicon ccECP TREXIO cell; a missing fixture fails instead of skipping."""
        structure, _aos, _mos_up, _mos_dn, _geminal, coulomb = read_trexio_file(
            trexio_file=PERIODIC_FIXTURE_PATH, store_tuple=True
        )
        return structure, coulomb

    @staticmethod
    def _silicon_ecp_charges(structure: Structure_data, coulomb: Coulomb_potential_data) -> tuple[int, int]:
        """Return ``(sum Z, sum (Z - z_core))`` of the ECP cell by integer arithmetic."""
        atomic_numbers = [int(z) for z in structure.atomic_numbers]
        z_cores = [int(round(float(zc))) for zc in coulomb.z_cores]
        bare_charge = sum(atomic_numbers)
        valence_charge = bare_charge - sum(z_cores)
        # Precondition of the ECP cases: core removal must change the total,
        # otherwise the bare-Z and effective-charge sums are indistinguishable.
        assert coulomb.ecp_flag and valence_charge != bare_charge
        return bare_charge, valence_charge

    @pytest.mark.light
    def test_build_ewald_data_accepts_charge_neutral_cells(self):
        """Neutral periodic cells are accepted, with ECP atoms counted by ``Z - z_core``.

        All-electron control: two protons and two electrons. ECP control: the
        silicon TREXIO cell with ``n_electron`` equal to the valence charge
        ``sum(Z - z_core)``, which differs from the bare ``sum(Z)``.
        """
        structure, coulomb = self._two_proton_cell()
        ewald = build_ewald_data(structure, coulomb, n_electron=2)
        assert ewald.structure_data is structure
        Ewald_data.build(structure_data=structure, coulomb_potential_data=coulomb, n_electron=2)

        si_structure, si_coulomb = self._silicon_ecp_cell()
        _bare_charge, valence_charge = self._silicon_ecp_charges(si_structure, si_coulomb)
        si_ewald = build_ewald_data(si_structure, si_coulomb, n_electron=valence_charge)
        assert si_ewald.structure_data is si_structure

    @pytest.mark.light
    @pytest.mark.parametrize("n_electron", [1, 3, 4])
    def test_build_ewald_data_rejects_charged_all_electron_cell(self, n_electron):
        """Two protons with 1, 3 or 4 electrons (cell charge +1, -1, -2) are rejected.

        Reference: the integer cell charge ``2 - n_electron`` is nonzero. Both the
        free builder and the ``Ewald_data.build`` factory must fail fast.
        """
        structure, coulomb = self._two_proton_cell()
        with pytest.raises(ValueError, match=rf"charge-neutral simulation cell.*n_electron = {n_electron}\b"):
            build_ewald_data(structure, coulomb, n_electron=n_electron)
        with pytest.raises(ValueError, match="charge-neutral simulation cell"):
            Ewald_data.build(structure_data=structure, coulomb_potential_data=coulomb, n_electron=n_electron)

    @pytest.mark.light
    @pytest.mark.parametrize("electron_count", ["bare_z", "valence_minus_1", "valence_plus_1"])
    def test_build_ewald_data_rejects_charged_ecp_cell(self, electron_count):
        """An ECP cell is neutral only for ``n_electron == sum(Z - z_core)``.

        ``bare_z`` passes the all-electron count ``sum(Z)``, which would be
        neutral if the core charges were ignored; the two others charge the
        valence cell by +1 and -1 electron.
        """
        structure, coulomb = self._silicon_ecp_cell()
        bare_charge, valence_charge = self._silicon_ecp_charges(structure, coulomb)
        n_electron = {
            "bare_z": bare_charge,
            "valence_minus_1": valence_charge - 1,
            "valence_plus_1": valence_charge + 1,
        }[electron_count]
        with pytest.raises(ValueError, match=rf"charge-neutral simulation cell.*n_electron = {n_electron}\b"):
            build_ewald_data(structure, coulomb, n_electron=n_electron)

    @pytest.mark.light
    @pytest.mark.parametrize("charge_shift", [-1, 1])
    def test_read_trexio_file_rejects_charged_periodic_file(self, tmp_path, charge_shift):
        """The periodic TREXIO loader rejects a file whose valence charge disagrees with its electron count.

        A copy of the silicon ECP fixture keeps its electron numbers and
        occupations but shifts one ``nucleus_charge`` (the TREXIO valence
        charge) by ``charge_shift``, so the cell carries a net charge of
        ``charge_shift``. The unmodified copy loads.
        """
        import shutil

        import trexio

        control_path = tmp_path / "control.h5"
        charged_path = tmp_path / "charged.h5"
        shutil.copyfile(PERIODIC_FIXTURE_PATH, control_path)
        shutil.copyfile(PERIODIC_FIXTURE_PATH, charged_path)
        file_u = trexio.File(str(charged_path), mode="u", back_end=trexio.TREXIO_HDF5)
        try:
            charges = np.asarray(trexio.read_nucleus_charge(file_u), dtype=np.float64)
            n_electron = int(trexio.read_electron_up_num(file_u)) + int(trexio.read_electron_dn_num(file_u))
            assert int(round(float(np.sum(charges)))) == n_electron
            charges[0] += float(charge_shift)
            trexio.write_nucleus_charge(file_u, charges)
        finally:
            file_u.close()

        read_trexio_file(trexio_file=str(control_path), store_tuple=True)
        with pytest.raises(ValueError, match=rf"charge-neutral simulation cell.*n_electron = {n_electron}\b"):
            read_trexio_file(trexio_file=str(charged_path), store_tuple=True)


class TestEcpError:
    """ECP non-local error paths."""

    @pytest.mark.light
    def test_compute_ecp_non_local_parts_all_pairs_raises_in_pbc(self):
        """``compute_ecp_non_local_parts_all_pairs`` raises ``NotImplementedError`` in PBC.

        The all-pairs variant enumerates molecular atoms with no image
        awareness, so it cannot be applied to periodic systems where the
        nearest atom to an electron may be an image of a different molecular
        atom. The production function raises at trace time (the ``pbc_flag``
        branch is static), so callers must route through
        ``compute_ecp_non_local_parts_nearest_neighbors`` for PBC inputs.
        """
        if not os.path.exists(PERIODIC_FIXTURE_PATH):
            pytest.skip(f"Periodic TREXIO fixture {PERIODIC_FIXTURE} is not present.")
        structure, _aos, _mos_up, _mos_dn, geminal_mo_data, coulomb = read_trexio_file(
            trexio_file=PERIODIC_FIXTURE_PATH, store_tuple=True
        )
        wavefunction_data = Wavefunction_data(
            geminal_data=geminal_mo_data,
            jastrow_data=Jastrow_data(
                jastrow_one_body_data=None,
                jastrow_two_body_data=None,
                jastrow_three_body_data=None,
            ),
        )
        rng = np.random.default_rng(0)
        n_up = geminal_mo_data.num_electron_up
        n_dn = geminal_mo_data.num_electron_dn
        cell = np.asarray(structure.cell, dtype=np.float64)
        frac = rng.uniform(0.1, 0.9, size=(n_up + n_dn, 3))
        r_carts = jnp.array(frac @ cell)
        with pytest.raises(NotImplementedError, match="all_pairs.*unsupported"):
            compute_ecp_non_local_parts_all_pairs(
                coulomb_potential_data=coulomb,
                wavefunction_data=wavefunction_data,
                r_up_carts=r_carts[:n_up],
                r_dn_carts=r_carts[n_up:],
                RT=jnp.eye(3),
                Nv=6,
            )

    @staticmethod
    def _channel_test_structure(n_atom: int = 2) -> Structure_data:
        """Two well-separated atoms; only the ECP table matters here."""
        return Structure_data(
            pbc_flag=False,
            positions=tuple((0.0, 0.0, 4.0 * i) for i in range(n_atom)),
            atomic_numbers=tuple([8] * n_atom),
            element_symbols=tuple(["O"] * n_atom),
            atomic_labels=tuple(["O"] * n_atom),
        )

    @classmethod
    def _channel_test_ecp(cls, *, max_ang_mom_plus_1, ang_moms, nucleus_index, coefficients=None, n_atom=2):
        """Build a Coulomb_potential_data with a hand-written ECP table."""
        n = len(ang_moms)
        return Coulomb_potential_data(
            structure_data=cls._channel_test_structure(n_atom),
            ecp_flag=True,
            z_cores=tuple([2.0] * n_atom),
            max_ang_mom_plus_1=tuple(max_ang_mom_plus_1),
            num_ecps=n,
            ang_moms=tuple(ang_moms),
            nucleus_index=tuple(nucleus_index),
            exponents=tuple([1.0] * n),
            coefficients=tuple(coefficients if coefficients is not None else [1.0] * n),
            powers=tuple([0] * n),
        )

    @pytest.mark.light
    def test_sanity_check_rejects_declared_l_with_no_terms(self):
        """An over-declared l_max: l=1 exists only on paper.

        ``max_ang_mom_plus_1`` promises non-local channels l=0..l_max, but
        nothing here is tagged l=1: the atom's own metadata is
        self-contradictory. This is exactly the off-by-one shape that
        ``parse_gaussinfo`` (upf_wrapper.py) produced before its fix -- l_max
        was derived by counting the local channel as if it were an angular
        momentum, so every ccECP UPF with a single non-local channel (H/Li/O)
        over-declared max_ang_mom_plus_1 by one. Physics was unaffected (the
        empty channel padded to a zero coefficient), but a TREXIO file
        carrying these tags loses the local channel under any spec-conforming
        reader. This check is the
        safety net: nothing downstream would otherwise notice.
        """
        data = self._channel_test_ecp(
            max_ang_mom_plus_1=[2, 2],
            # local tagged 2, one non-local l=0 -- nothing carries l=1
            ang_moms=[2, 2, 2, 0, 2, 2, 2, 0],
            nucleus_index=[0, 0, 0, 0, 1, 1, 1, 1],
        )
        with pytest.raises(ValueError, match=r"l = \[1\] carry no terms"):
            data.sanity_check()

    @pytest.mark.light
    def test_sanity_check_rejects_missing_local_channel(self):
        """No term tagged l_max + 1 at all: the local channel is gone."""
        data = self._channel_test_ecp(
            max_ang_mom_plus_1=[1, 1],
            ang_moms=[0, 0],
            nucleus_index=[0, 1],
        )
        with pytest.raises(ValueError, match="local channel is missing"):
            data.sanity_check()

    @pytest.mark.light
    def test_sanity_check_rejects_tag_above_local_channel(self):
        """A tag above l_max + 1 has no meaning in the local-first layout."""
        data = self._channel_test_ecp(
            max_ang_mom_plus_1=[1, 1],
            ang_moms=[1, 0, 3, 1, 0],
            nucleus_index=[0, 0, 0, 1, 1],
        )
        with pytest.raises(ValueError, match="exceeds"):
            data.sanity_check()

    @pytest.mark.light
    def test_sanity_check_accepts_zero_coefficient_terms(self):
        """A 0.0 coefficient is DATA, not corruption -- PySCF writes it for H/He.

        Every declared l carries a term; one of them is simply zero. Rejecting
        on magnitude would break the very files PySCF produces, so the check
        only ever tests for the PRESENCE of a term, never its size.
        """
        data = self._channel_test_ecp(
            max_ang_mom_plus_1=[1, 1],
            ang_moms=[1, 1, 1, 0, 1, 1, 1, 0],
            nucleus_index=[0, 0, 0, 0, 1, 1, 1, 1],
            coefficients=[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        )
        data.sanity_check()

    @pytest.mark.light
    def test_sanity_check_accepts_mixed_l_max_across_atoms(self):
        """Atoms may legitimately disagree on l_max; padding handles the difference.

        Atom 0 is an l_max=1 species (non-local l=0,1; local tagged 2), atom 1
        an l_max=0 one (non-local l=0; local tagged 1). Each is internally
        consistent, which is exactly what the check must require -- per atom,
        not globally.
        """
        data = self._channel_test_ecp(
            max_ang_mom_plus_1=[2, 1],
            ang_moms=[2, 2, 0, 1, 1, 1, 0],
            nucleus_index=[0, 0, 0, 0, 1, 1, 1],
        )
        data.sanity_check()

    @pytest.mark.light
    @pytest.mark.parametrize("lmax_p1,tags", [(1, [1, 0]), (2, [2, 0, 1]), (3, [3, 0, 1, 2])])
    def test_sanity_check_accepts_contiguous_channels(self, lmax_p1, tags):
        """The canonical local-first layout at several l_max values."""
        n = len(tags)
        data = self._channel_test_ecp(
            max_ang_mom_plus_1=[lmax_p1, lmax_p1],
            ang_moms=tags + tags,
            nucleus_index=[0] * n + [1] * n,
        )
        data.sanity_check()

    @pytest.mark.light
    @pytest.mark.parametrize(
        "lmax_p1,tags",
        [
            pytest.param(1, [1, 1, 1, 0], id="H-Li-O-lmax0"),
            pytest.param(2, [2, 2, 2, 0, 0, 1, 1], id="Si-lmax1"),
        ],
    )
    def test_sanity_check_accepts_real_ccecp_shapes(self, lmax_p1, tags):
        """The shipped ccECP UPFs, through the fixed converter, must validate.

        H/Li/O have a single non-local channel (l_max=0, local tagged 1); Si
        has two (l_max=1, local tagged 2).
        """
        n = len(tags)
        data = self._channel_test_ecp(
            max_ang_mom_plus_1=[lmax_p1, lmax_p1],
            ang_moms=tags + tags,
            nucleus_index=[0] * n + [1] * n,
        )
        data.sanity_check()


class TestDispatch:
    """dispatch routing PBC/molecular for plain + LRDMC discretized."""

    @pytest.mark.light
    def test_compute_coulomb_potential_dispatch_pbc(self):
        """The ee/ei/ii Coulomb dispatch shims route to ``compute_ewald_potential_*``
        on PBC inputs and produce the same numerical result as a direct call.

        Validates the dispatch wiring: the shim's ``if pbc_flag:`` branch is
        selected at trace time, the Ewald entry point is invoked, and the
        output flows through unchanged. Covers ee + ei + ii in one fixture
        setup; per-channel breakdown is in
        :py:func:`test_compute_ewald_potential_ee_production_vs_debug` etc.
        """
        structure, coulomb, r_up, r_dn = _make_h2_box_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-10)
        coulomb_with_ewald = coulomb.replace(ewald_data=ewald_data)
        atol, rtol = get_tolerance("ewald", "strict")

        E_ee_shim = float(compute_coulomb_potential_ee_dispatch(coulomb_with_ewald, r_up, r_dn))
        E_ee_direct = float(compute_ewald_potential_ee(r_up, r_dn, ewald_data))
        np.testing.assert_allclose(E_ee_shim, E_ee_direct, atol=atol, rtol=rtol)

        E_ei_up_shim, E_ei_dn_shim = compute_coulomb_potential_ei_dispatch(
            coulomb_with_ewald,
            r_up,
            r_dn,
        )
        E_ei_up_direct, E_ei_dn_direct = compute_ewald_potential_ei(
            coulomb_with_ewald,
            r_up,
            r_dn,
            ewald_data,
        )
        np.testing.assert_allclose(np.asarray(E_ei_up_shim), np.asarray(E_ei_up_direct), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(E_ei_dn_shim), np.asarray(E_ei_dn_direct), atol=atol, rtol=rtol)

        E_ii_shim = float(compute_coulomb_potential_ii_dispatch(coulomb_with_ewald))
        E_ii_direct = float(compute_ewald_potential_ii(coulomb_with_ewald, ewald_data))
        np.testing.assert_allclose(E_ii_shim, E_ii_direct, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_compute_coulomb_potential_dispatch_molecular(self):
        """The ee/ei/ii Coulomb dispatch shims fall through to the bare 1/r
        implementations on molecular inputs and reproduce them bitwise.

        Regression guard for HLO byte-identity:
        rewiring call sites onto the dispatch shims must not perturb
        molecular outputs.
        """
        mol_structure = Structure_data(
            pbc_flag=False,
            positions=np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=np.float64),
            atomic_numbers=(1, 1),
            element_symbols=("H", "H"),
            atomic_labels=("H1", "H2"),
        )
        mol_coulomb = Coulomb_potential_data(structure_data=mol_structure, ecp_flag=False)
        r_up = np.array([[0.1, 0.2, -0.6]], dtype=np.float64)
        r_dn = np.array([[-0.1, -0.2, 0.5]], dtype=np.float64)
        atol, rtol = get_tolerance("coulomb", "strict")

        E_ee_shim = float(compute_coulomb_potential_ee_dispatch(mol_coulomb, r_up, r_dn))
        E_ee_direct = float(compute_bare_coulomb_potential_el_el(r_up, r_dn))
        np.testing.assert_allclose(E_ee_shim, E_ee_direct, atol=atol, rtol=rtol)

        E_ei_up_shim, E_ei_dn_shim = compute_coulomb_potential_ei_dispatch(mol_coulomb, r_up, r_dn)
        E_ei_up_direct, E_ei_dn_direct = compute_bare_coulomb_potential_el_ion_element_wise(
            mol_coulomb,
            r_up,
            r_dn,
        )
        np.testing.assert_allclose(np.asarray(E_ei_up_shim), np.asarray(E_ei_up_direct), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(E_ei_dn_shim), np.asarray(E_ei_dn_direct), atol=atol, rtol=rtol)

        E_ii_shim = float(compute_coulomb_potential_ii_dispatch(mol_coulomb))
        E_ii_direct = float(compute_bare_coulomb_potential_ion_ion(mol_coulomb))
        np.testing.assert_allclose(E_ii_shim, E_ii_direct, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_compute_coulomb_potential_ei_discretized_dispatch_pbc(self):
        """The LRDMC discretised Coulomb dispatch forwards to
        :py:func:`compute_ewald_potential_ei_discretized` on PBC inputs and
        produces the same numerical result as a direct call.

        Wiring guard: the periodic branch does not raise.
        """
        structure, coulomb, r_up, r_dn = _make_h2_box_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-10)
        coulomb_with_ewald = coulomb.replace(ewald_data=ewald_data)
        alat = 0.40
        E_up_shim, E_dn_shim = compute_coulomb_potential_ei_discretized_dispatch(
            coulomb_with_ewald,
            r_up,
            r_dn,
            alat,
        )
        E_up_direct, E_dn_direct = compute_ewald_potential_ei_discretized(
            coulomb_with_ewald,
            r_up,
            r_dn,
            ewald_data,
            alat,
        )
        atol, rtol = get_tolerance("ewald", "strict")
        np.testing.assert_allclose(np.asarray(E_up_shim), np.asarray(E_up_direct), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(E_dn_shim), np.asarray(E_dn_direct), atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_compute_coulomb_potential_ei_discretized_dispatch_molecular(self):
        """The LRDMC discretised dispatch falls through to the bare-1/r-with-floor
        implementation on molecular inputs and reproduces it bitwise.

        Regression guard for the molecular path: rewiring to the dispatch
        shim must not perturb molecular LRDMC e-I evaluation.
        """
        mol_structure = Structure_data(
            pbc_flag=False,
            positions=np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=np.float64),
            atomic_numbers=(1, 1),
            element_symbols=("H", "H"),
            atomic_labels=("H1", "H2"),
        )
        mol_coulomb = Coulomb_potential_data(structure_data=mol_structure, ecp_flag=False)
        r_up = np.array([[0.1, 0.2, -0.6]], dtype=np.float64)
        r_dn = np.array([[-0.1, -0.2, 0.5]], dtype=np.float64)
        alat = 0.40
        E_up_shim, E_dn_shim = compute_coulomb_potential_ei_discretized_dispatch(
            mol_coulomb,
            r_up,
            r_dn,
            alat,
        )
        E_up_direct, E_dn_direct = compute_discretized_bare_coulomb_potential_el_ion_element_wise(
            mol_coulomb,
            r_up,
            r_dn,
            alat,
        )
        atol, rtol = get_tolerance("coulomb", "strict")
        np.testing.assert_allclose(np.asarray(E_up_shim), np.asarray(E_up_direct), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(E_dn_shim), np.asarray(E_dn_direct), atol=atol, rtol=rtol)


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)


# ============================================================================
# Slab / vacuum-cell Ewald references.
#
# Motivation: a parameter heuristic that caps ``r_real_cut <= L_min`` and
# keeps only the home image leaves the real-space sum truncated in strongly
# anisotropic cells with vacuum (slabs): on the 19-atom slab
# (10.914 x 10.914 x 47.243 Bohr) the ion-ion energy then sits ~60 mHa below
# Quantum Espresso's `ewald contribution`, and the total Ewald energy
# DEPENDS ON eta (it must not). ``build_ewald_data`` enumerates the
# real-space images explicitly and bounds the reciprocal sum per direction
# with a spherical |G| filter; these tests pin that regime with independent
# references.
# ============================================================================

#: Ion-ion Ewald reference for an 8-atom NaCl-type bilayer slab (Z=+-1)
#: floating in vacuum: in-plane lattice 10.914393 Bohr, cell height
#: 47.243153 Bohr (the 19-atom slab cell below, aspect ratio 4.33).
#: Regeneration (any env with pymatgen)::
#:
#:     pip install pymatgen==2026.5.4
#:
#: .. code-block:: python
#:
#:     from importlib.metadata import version as _pkg_version
#:     from pymatgen.core import Lattice, Structure
#:     from pymatgen.analysis.ewald import EwaldSummation
#:
#:     HA_TO_EV = 27.211386245988
#:     ANG = 0.529177210903  # Bohr -> Angstrom
#:     a_b, c_b = 10.914393, 47.243153
#:     a, c = a_b * ANG, c_b * ANG
#:     lat = Lattice([[a, 0, 0], [0, a, 0], [0, 0, c]])
#:     h = 0.5
#:     z2 = (a_b / 4.0) / c_b
#:     sites = [("Na", [0, 0, 0]), ("Cl", [h, 0, 0]), ("Cl", [0, h, 0]),
#:              ("Na", [h, h, 0]), ("Cl", [0, 0, z2]), ("Na", [h, 0, z2]),
#:              ("Na", [0, h, z2]), ("Cl", [h, h, z2])]
#:     st = Structure(lat, [s for s, _ in sites], [f for _, f in sites])
#:     st.add_oxidation_state_by_site(
#:         [+1 if s == "Na" else -1 for s, _ in sites])
#:     e_ha = float(EwaldSummation(st, acc_factor=18.0).total_energy) / HA_TO_EV
#:     print(f"PYMATGEN_E_II_NACL_SLAB_VAC_HA = {e_ha:.10f}")
#:     print(f"pymatgen version: {_pkg_version('pymatgen')}")
#:
#: (acc_factor 12 and 18 agree to 1e-10, i.e. the reference itself is
#: converged.)
PYMATGEN_E_II_NACL_SLAB_VAC_HA: float = -1.6624465232
#: pymatgen version used to compute ``PYMATGEN_E_II_NACL_SLAB_VAC_HA``.
PYMATGEN_VERSION_USED_FOR_NACL_SLAB_VAC: str = "2026.5.4"

#: Ion-ion Ewald references for the two 19-atom slab geometries
#: (19 atoms, valence charges Li +1 / H +1 / O +6, cell
#: 10.914393 x 10.914393 x 47.243153 Bohr; net ionic charge +24 with the
#: uniform neutralising background). Computed with pymatgen; acc_factor
#: 12 and 18 agree to 1e-10, and Quantum Espresso 7.5's printed `ewald
#: contribution` (16.65909555 / 60.513906875 Ha) agrees to ~2e-8 relative
#: (QE's own convergence). Regeneration::
#:
#:     pip install pymatgen==2026.5.4
#:
#: .. code-block:: python
#:
#:     from importlib.metadata import version as _pkg_version
#:     from pymatgen.core import Lattice, Structure
#:     from pymatgen.analysis.ewald import EwaldSummation
#:
#:     HA_TO_EV = 27.211386245988
#:     ANG = 0.529177210903
#:     a_b, c_b = 10.914393, 47.243153
#:     lat = Lattice([[a_b * ANG, 0, 0], [0, a_b * ANG, 0], [0, 0, c_b * ANG]])
#:     # positions: _WATER_LIH_SHARED_16 + _WATER_LIH_WATER_{FAR,ADS} below,
#:     # converted Bohr -> Angstrom, coords_are_cartesian=True
#:     # species ["Li"]*8 + ["H"]*10 + ["O"], oxidation states [1]*18 + [6]
#:     e_ha = float(EwaldSummation(st, acc_factor=18.0).total_energy) / HA_TO_EV
#:     print(f"{e_ha:.10f}"); print(_pkg_version("pymatgen"))
PYMATGEN_E_II_WATER_LIH_FAR_HA: float = 16.6590958887
PYMATGEN_E_II_WATER_LIH_ADS_HA: float = 60.5139080806
#: pymatgen version used for the 19-atom slab ion-ion references.
PYMATGEN_VERSION_USED_FOR_WATER_LIH: str = "2026.5.4"

# --- Slab-cell e-e / e-i references ---------------------------------------
#
# Why these tests exist: the ion-ion channel is pinned against pymatgen on
# slab/vacuum cells (TestEwaldSlabVacuum above), but without these tests the
# e-e and e-i channels would have external references ONLY on the compact H2
# box / H2 monoclinic fixtures. On a slab the other coverage is eta-invariance
# plus Rust==XLA parity -- and neither can see an ETA-INDEPENDENT constant. A
# Madelung-sized convention error (this cell's Madelung constant is
# xi = -57.96 mHa; the 24-electron exxdiv term is +695.5 mHa) would pass both.
#
# Convention note (why the per-channel numbers differ from pymatgen while the
# total agrees exactly): pymatgen's total_energy_matrix attributes the
# neutralising background to the charged ion (+24) and electron (-24) blocks,
# jqmc splits it differently. Measured on both geometries the per-channel
# offsets are identical to all printed digits -- ii and ee each -5187.546 mHa,
# e-i +10375.093 mHa = -(ii + ee) -- so they cancel exactly in the neutral
# total, which agrees to 4e-11 Ha. The offsets are also IDENTICAL between the
# adsorbed and far geometries, hence they cancel in any energy difference too.
# We therefore assert on the TOTAL, which is convention-free.

#: Fixed 24-electron configuration (12 up + 12 dn, Bohr) used with the
#: 19-atom slab geometries. Drawn once from
#: ``np.random.default_rng(0).uniform(0, 1, (12, 3)) * L`` (L = the cell
#: diagonal) and frozen here so the reference is reproducible.
_WATER_LIH_SLAB_R_UP_BOHR = np.array(
    [
        [6.952050181369, 2.944558220197, 1.935718460267],
        [0.180389109519, 8.876351005836, 43.121451388935],
        [6.621061264583, 7.962012158728, 25.682558646425],
        [10.205747916682, 8.904546320129, 0.129375382529],
        [9.358047234557, 0.366566168015, 34.471223892973],
        [1.917174475915, 9.421073987843, 25.580335271795],
        [3.271173360098, 4.613374448229, 1.337910556835],
        [1.356476523044, 7.319458417361, 30.575273115298],
        [6.716554953055, 4.187607612493, 47.111341569610],
        [10.705222355692, 7.482274636622, 30.729747108990],
        [7.513978177016, 4.244841267428, 6.382384856539],
        [7.874607289796, 5.733923539749, 14.656804394039],
    ]
)
_WATER_LIH_SLAB_R_DN_BOHR = np.array(
    [
        [5.302598039586, 9.708219792804, 44.127160732979],
        [3.905117390395, 6.237901183808, 15.206124888618],
        [6.486424089511, 3.688095914296, 18.501316359659],
        [9.716804155601, 2.479287248758, 29.441325624036],
        [0.916976477888, 9.087805456639, 37.185005766729],
        [2.612572173016, 9.566293353371, 2.766938629211],
        [3.668513692800, 1.640209161521, 21.275451600535],
        [8.691396041354, 2.517319711346, 2.457650285445],
        [4.415437768685, 2.166649383401, 4.287460019400],
        [6.333975731288, 3.260086981166, 31.747156834509],
        [2.177589965039, 10.282592738520, 17.248955540246],
        [1.151416940874, 6.866333605408, 43.801704405232],
    ]
)

#: Total Ewald energy (ii + ei + ee, Ha) for the neutral 19-ion + 24-electron
#: 19-atom slab systems with the frozen electron configuration above.
#: pymatgen 2026.5.4, ``acc_factor=18.0``. Regeneration::
#:
#:     pip install pymatgen==2026.5.4
#:
#: .. code-block:: python
#:
#:     import numpy as np
#:     from pymatgen.core import Lattice, Structure
#:     from pymatgen.analysis.ewald import EwaldSummation
#:
#:     HA_TO_EV, ANG = 27.211386245988, 0.529177210903
#:     a_b, c_b = 10.914393, 47.243153
#:     lat = Lattice(np.diag([a_b, a_b, c_b]) * ANG)
#:     ions = np.array(_WATER_LIH_SHARED_16 + _WATER_LIH_WATER_FAR)  # or _ADS
#:     pos = np.vstack([ions, _WATER_LIH_SLAB_R_UP_BOHR,
#:                      _WATER_LIH_SLAB_R_DN_BOHR]) * ANG
#:     q = np.concatenate([[1.0] * 18 + [6.0], -np.ones(24)])
#:     st = Structure(lat, ["Li"] * 8 + ["H"] * 10 + ["O"] + ["H"] * 24, pos,
#:                    coords_are_cartesian=True)
#:     st.add_oxidation_state_by_site(q.tolist())
#:     M = np.asarray(EwaldSummation(st, acc_factor=18.0).total_energy_matrix)
#:     print(f"{M.sum() / HA_TO_EV:.8f}")
PYMATGEN_E_TOTAL_WATER_LIH_FAR_HA: float = 3.66398537
PYMATGEN_E_TOTAL_WATER_LIH_ADS_HA: float = 45.43330366
#: pymatgen version used for the slab e-e / e-i totals.
PYMATGEN_VERSION_USED_FOR_WATER_LIH_TOTAL: str = "2026.5.4"

#: Independent electron-ion / electron-electron Ewald references for the
#: neutral H4 layer in the vacuum slab cell
#: (:py:func:`_make_h4_slab_box_fixture`). Same ``total_energy_matrix``
#: block decomposition as the cubic H2-box references (see
#: :py:func:`TestEwald.test_compute_ewald_potential_ei_vs_pymatgen_h2_box`
#: for the recipe); these are the anisotropic-vacuum-cell analogues of the
#: cubic/monoclinic ee/ei references, next to the slab-cell ii references.
#: Shared regeneration script in
#: :py:func:`TestEwaldSlabVacuum.test_compute_ewald_potential_ei_vs_pymatgen_h4_slab`.
PYMATGEN_E_EI_H4_SLAB_HA: float = -10.1925160017
PYMATGEN_E_EE_H4_SLAB_HA: float = 1.8530010087
#: Same slab cell / same 4 protons, but one up-spin electron parked DEEP in
#: the vacuum region (z = 23.035 Bohr -- the H site of the far water
#: molecule in the 19-atom slab cell): the exact "isolated charge in vacuum"
#: regime where a home-image-only Ewald sum is wrong. The
#: fixture is charge-neutral, so the total is background-free.
PYMATGEN_E_EI_H4_SLAB_VACUUM_ELECTRON_HA: float = -6.6054880309
PYMATGEN_E_EE_H4_SLAB_VACUUM_ELECTRON_HA: float = 0.3694970928
PYMATGEN_E_TOTAL_H4_SLAB_VACUUM_ELECTRON_HA: float = -4.3463443391
#: pymatgen version used to compute the five H4-slab ee/ei/total references.
PYMATGEN_VERSION_USED_FOR_H4_SLAB: str = "2025.10.7"

_SLAB_A_BOHR = 10.914393
_SLAB_C_BOHR = 47.243153

#: 19-atom slab geometries (Bohr): an Li8H8 slab plus one water molecule,
#: order Li x8, H x10 (8 slab + 2 water), O; far = adsorbed with the water
#: displaced +8.1 Ang in z.
_WATER_LIH_SHARED_16 = [
    [0.000000000, 0.000000000, 0.000000000],
    [0.000000000, 5.457196568, 0.000000000],
    [5.457196568, 0.000000000, 0.000000000],
    [5.457196568, 5.457196568, 0.000000000],
    [2.728598283, 2.728598283, 3.858820746],
    [2.728598283, 8.185794850, 3.858820746],
    [8.185794850, 2.728598283, 3.858820746],
    [8.185794850, 8.185794850, 3.858820746],
    [2.728598283, 2.728598283, 0.000000000],
    [2.728598283, 8.185794850, 0.000000000],
    [8.185794850, 2.728598283, 0.000000000],
    [8.185794850, 8.185794850, 0.000000000],
    [0.000000000, 0.000000000, 3.858820746],
    [0.000000000, 5.457196568, 3.858820746],
    [5.457196568, 0.000000000, 3.858820746],
    [5.457196568, 5.457196568, 3.858820746],
]
_WATER_LIH_WATER_FAR = [
    [4.019989278, 1.274473141, 23.035387293],
    [4.019989278, 4.182723427, 23.035387293],
    [2.896407187, 2.728598283, 23.230915366],
]
_WATER_LIH_WATER_ADS = [
    [4.019989278, 1.274473141, 7.728605684],
    [4.019989278, 4.182723427, 7.728605684],
    [2.896407187, 2.728598283, 7.924133756],
]


def _make_water_lih_structure(which: str) -> Structure_data:
    """The 19-atom slab with one water molecule, with valence point charges."""
    water = {"far": _WATER_LIH_WATER_FAR, "ads": _WATER_LIH_WATER_ADS}[which]
    positions = np.array(_WATER_LIH_SHARED_16 + water, dtype=np.float64)
    return Structure_data.build(
        pbc_flag=True,
        positions=positions,
        vec_a=(_SLAB_A_BOHR, 0.0, 0.0),
        vec_b=(0.0, _SLAB_A_BOHR, 0.0),
        vec_c=(0.0, 0.0, _SLAB_C_BOHR),
        atomic_numbers=(1,) * 8 + (1,) * 10 + (6,),
        element_symbols=("Li",) * 8 + ("H",) * 10 + ("O",),
        atomic_labels=tuple(f"Li{i}" for i in range(1, 9)) + tuple(f"H{i}" for i in range(1, 11)) + ("O1",),
    )


def _make_nacl_slab_structure() -> Structure_data:
    """8-atom NaCl-type bilayer slab in the 19-atom slab vacuum cell."""
    a, c = _SLAB_A_BOHR, _SLAB_C_BOHR
    h, z2 = a / 2.0, a / 4.0
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [h, 0.0, 0.0],
            [0.0, h, 0.0],
            [h, h, 0.0],
            [0.0, 0.0, z2],
            [h, 0.0, z2],
            [0.0, h, z2],
            [h, h, z2],
        ],
        dtype=np.float64,
    )
    charges = (1, -1, -1, 1, -1, 1, 1, -1)
    return Structure_data.build(
        pbc_flag=True,
        positions=positions,
        vec_a=(a, 0.0, 0.0),
        vec_b=(0.0, a, 0.0),
        vec_c=(0.0, 0.0, c),
        atomic_numbers=charges,
        element_symbols=tuple("Na" if q > 0 else "Cl" for q in charges),
        atomic_labels=tuple(f"X{i}" for i in range(1, 9)),
    )


def _make_h4_slab_box_fixture() -> tuple[Structure_data, Coulomb_potential_data, np.ndarray, np.ndarray]:
    """All-electron H4 layer in the vacuum slab cell (charge neutral).

    Four protons near z = 0 plus 2 up + 2 dn electrons close by: total
    charge zero, so the eta-invariance of E_ii + E_ei + E_ee is exact and
    background-free -- the anisotropic-cell analogue of
    :py:func:`_make_h2_box_fixture`.
    """
    a = _SLAB_A_BOHR
    h = a / 2.0
    positions = np.array(
        [[0.0, 0.0, 0.0], [h, 0.0, 0.3], [0.0, h, -0.2], [h, h, 0.1]],
        dtype=np.float64,
    )
    structure = Structure_data.build(
        pbc_flag=True,
        positions=positions,
        vec_a=(a, 0.0, 0.0),
        vec_b=(0.0, a, 0.0),
        vec_c=(0.0, 0.0, _SLAB_C_BOHR),
        atomic_numbers=(1, 1, 1, 1),
        element_symbols=("H",) * 4,
        atomic_labels=("H1", "H2", "H3", "H4"),
    )
    coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
    r_up = np.array([[0.1, 0.2, -0.6], [h + 0.2, -0.1, 0.8]], dtype=np.float64)
    r_dn = np.array([[-0.1, h - 0.2, 0.5], [h - 0.3, h + 0.1, -0.4]], dtype=np.float64)
    return structure, coulomb, r_up, r_dn


def _make_h4_slab_vacuum_electron_fixture() -> tuple[Structure_data, Coulomb_potential_data, np.ndarray, np.ndarray]:
    """H4 slab layer with one up-spin electron isolated deep in the vacuum.

    Identical protons/cell to :py:func:`_make_h4_slab_box_fixture`, but the
    second up-spin electron sits at ``z = 23.035`` Bohr -- the H site of the
    far water molecule in the 19-atom slab cell, ~23 Bohr from the charged
    layer. Still 4 protons + 4 electrons (charge neutral). This is the
    minimal point-charge model of the ``far`` regime (isolated charge in
    vacuum), where a home-image-only slab-cell Ewald sum is wrong.
    """
    a = _SLAB_A_BOHR
    h = a / 2.0
    positions = np.array(
        [[0.0, 0.0, 0.0], [h, 0.0, 0.3], [0.0, h, -0.2], [h, h, 0.1]],
        dtype=np.float64,
    )
    structure = Structure_data.build(
        pbc_flag=True,
        positions=positions,
        vec_a=(a, 0.0, 0.0),
        vec_b=(0.0, a, 0.0),
        vec_c=(0.0, 0.0, _SLAB_C_BOHR),
        atomic_numbers=(1, 1, 1, 1),
        element_symbols=("H",) * 4,
        atomic_labels=("H1", "H2", "H3", "H4"),
    )
    coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
    r_up = np.array([[0.1, 0.2, -0.6], [4.019989278, 1.274473141, 23.035387293]], dtype=np.float64)
    r_dn = np.array([[-0.1, h - 0.2, 0.5], [h - 0.3, h + 0.1, -0.4]], dtype=np.float64)
    return structure, coulomb, r_up, r_dn


class TestEwaldSlabVacuum:
    """Ewald correctness in anisotropic vacuum (slab) cells.

    These pin the exact regime where the current heuristic breaks (see the
    block comment above). All four tests are expected to FAIL on the
    home-image-only implementation and must turn green with the QE-style
    rewrite.
    """

    @pytest.mark.light
    def test_compute_ewald_potential_ii_vs_pymatgen_nacl_slab_vacuum(self):
        """``ewald_data.ii_energy`` matches pymatgen on a NaCl bilayer slab
        floating in 36 Bohr of vacuum (aspect ratio 4.33)."""
        structure = _make_nacl_slab_structure()
        coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        np.testing.assert_allclose(
            float(ewald_data.ii_energy),
            PYMATGEN_E_II_NACL_SLAB_VAC_HA,
            atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE,
            rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE,
        )

    @pytest.mark.light
    @pytest.mark.parametrize(
        "which, reference",
        [
            ("far", PYMATGEN_E_II_WATER_LIH_FAR_HA),
            ("ads", PYMATGEN_E_II_WATER_LIH_ADS_HA),
        ],
    )
    def test_compute_ewald_potential_ii_vs_pymatgen_water_lih(self, which: str, reference: float):
        """``ewald_data.ii_energy`` matches pymatgen on the charged,
        background-neutralised 19-atom slabs with one water molecule."""
        structure = _make_water_lih_structure(which)
        coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        np.testing.assert_allclose(
            float(ewald_data.ii_energy),
            reference,
            atol=ATOL_BIT_EXACT,
            rtol=RTOL_EWALD_VS_PYMATGEN_CHARGED_SLAB,
        )

    @pytest.mark.light
    @pytest.mark.parametrize("eta_factor", [0.5, 1.0, 2.0])
    def test_compute_ewald_potential_total_energy_eta_invariance_slab(self, eta_factor: float):
        """eta-invariance of ``E_ii + E_ei + E_ee`` on a neutral H4 layer in
        the vacuum slab cell -- the anisotropic-cell analogue of
        ``test_compute_ewald_potential_total_energy_eta_invariance_h2_box``.
        A real-space sum truncated at the home image violates it here by
        ~1e-2 Ha."""
        structure, coulomb, r_up, r_dn = _make_h4_slab_box_fixture()
        ewald_default = build_ewald_data(structure, coulomb, tol=1e-10)
        coulomb_default = coulomb.replace(ewald_data=ewald_default)
        E_ii_def = float(compute_ewald_potential_ii(coulomb_default, ewald_default))
        E_ei_up_def, E_ei_dn_def = compute_ewald_potential_ei(coulomb_default, r_up, r_dn, ewald_default)
        E_ee_def = float(compute_ewald_potential_ee(r_up, r_dn, ewald_default))
        E_total_default = E_ii_def + float(np.sum(np.asarray(E_ei_up_def))) + float(np.sum(np.asarray(E_ei_dn_def))) + E_ee_def
        ewald_alt = build_ewald_data(
            structure,
            coulomb,
            tol=1e-10,
            eta_override=eta_factor * ewald_default.eta,
        )
        coulomb_alt = coulomb.replace(ewald_data=ewald_alt)
        E_ii_alt = float(compute_ewald_potential_ii(coulomb_alt, ewald_alt))
        E_ei_up_alt, E_ei_dn_alt = compute_ewald_potential_ei(coulomb_alt, r_up, r_dn, ewald_alt)
        E_ee_alt = float(compute_ewald_potential_ee(r_up, r_dn, ewald_alt))
        E_total_alt = E_ii_alt + float(np.sum(np.asarray(E_ei_up_alt))) + float(np.sum(np.asarray(E_ei_dn_alt))) + E_ee_alt
        atol, rtol = get_tolerance("ewald", "strict")
        np.testing.assert_allclose(E_total_alt, E_total_default, atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize(
        "which, reference",
        [
            ("far", PYMATGEN_E_TOTAL_WATER_LIH_FAR_HA),
            ("ads", PYMATGEN_E_TOTAL_WATER_LIH_ADS_HA),
        ],
    )
    def test_compute_ewald_potential_total_vs_pymatgen_water_lih_slab(self, which, reference):
        """``E_ii + E_ei + E_ee`` matches pymatgen on the 19-atom slab
        geometries with a frozen 24-electron configuration.

        This is the only external-reference check of the e-e and e-i channels
        on a vacuum slab cell: elsewhere they are pinned only on compact cells
        (H2 box / H2 monoclinic), and the slab coverage is eta-invariance plus
        Rust==XLA parity, neither of which can detect an eta-independent
        constant such as a Madelung-sized (xi = -57.96 mHa here) convention
        error. Asserted on the neutral total because the per-channel split of
        the neutralising background is convention-dependent (see the note above
        the reference constants); the total is not.
        """
        structure = _make_water_lih_structure(which)
        coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        r_up = jnp.asarray(_WATER_LIH_SLAB_R_UP_BOHR)
        r_dn = jnp.asarray(_WATER_LIH_SLAB_R_DN_BOHR)
        E_ii = float(compute_ewald_potential_ii(coulomb, ewald_data))
        E_ei_up, E_ei_dn = compute_ewald_potential_ei(coulomb, r_up, r_dn, ewald_data)
        E_ei = float(np.sum(np.asarray(E_ei_up))) + float(np.sum(np.asarray(E_ei_dn)))
        E_ee = float(compute_ewald_potential_ee(r_up, r_dn, ewald_data))
        np.testing.assert_allclose(
            E_ii + E_ei + E_ee,
            reference,
            atol=ATOL_BIT_EXACT,
            rtol=RTOL_EWALD_VS_PYMATGEN_CHARGED_SLAB,
        )

    @pytest.mark.light
    def test_compute_ewald_potential_ei_vs_pymatgen_h4_slab(self):
        """``sum(compute_ewald_potential_ei)`` matches pymatgen on the neutral
        H4 layer in the vacuum slab cell.

        The anisotropic-vacuum-cell analogue of the cubic H2-box e-i/e-e
        pymatgen references (same ``total_energy_matrix`` block decomposition
        + per-channel jellium restore; see
        ``test_compute_ewald_potential_ei_vs_pymatgen_h2_box`` for the
        conventions). Shared regeneration script for all five H4-slab
        references::

            pip install pymatgen==2025.10.7

        .. code-block:: python

            import numpy as np
            from importlib.metadata import version as _pkg_version
            from pymatgen.core import Lattice, Structure
            from pymatgen.analysis.ewald import EwaldSummation

            BOHR_TO_ANG = 0.529177210903
            HA_TO_EV = 27.211386245988
            A, C = 10.914393, 47.243153
            h = A / 2.0
            ions = np.array([[0.0, 0.0, 0.0], [h, 0.0, 0.3], [0.0, h, -0.2], [h, h, 0.1]])
            fixtures = {
                "H4_SLAB": {
                    "r_up": np.array([[0.1, 0.2, -0.6], [h + 0.2, -0.1, 0.8]]),
                    "r_dn": np.array([[-0.1, h - 0.2, 0.5], [h - 0.3, h + 0.1, -0.4]]),
                },
                "H4_SLAB_VACUUM_ELECTRON": {
                    "r_up": np.array([[0.1, 0.2, -0.6], [4.019989278, 1.274473141, 23.035387293]]),
                    "r_dn": np.array([[-0.1, h - 0.2, 0.5], [h - 0.3, h + 0.1, -0.4]]),
                },
            }
            lattice = Lattice(np.diag([A, A, C]) * BOHR_TO_ANG)
            for name, fx in fixtures.items():
                elec = np.vstack([fx["r_up"], fx["r_dn"]])
                pos = np.vstack([ions, elec])
                s = Structure(lattice, ["H"] * 8, pos * BOHR_TO_ANG, coords_are_cartesian=True)
                s.add_oxidation_state_by_site([1.0] * 4 + [-1.0] * 4)
                ew = EwaldSummation(s, acc_factor=18.0)
                M = np.asarray(ew.total_energy_matrix)
                ion, ele = slice(0, 4), slice(4, None)
                bg = -EwaldSummation.CONV_FACT / 2.0 * np.pi / s.volume / ew.eta
                Q = Ne = 4.0
                ee = (M[ele, ele].sum() + bg * Ne**2) / HA_TO_EV
                ei = (M[ion, ele].sum() + M[ele, ion].sum() - 2.0 * bg * Q * Ne) / HA_TO_EV
                print(f"{name}: EI={ei:.10f} EE={ee:.10f} TOTAL={M.sum() / HA_TO_EV:.10f}")
            print(_pkg_version("pymatgen"))
        """
        structure, coulomb, r_up, r_dn = _make_h4_slab_box_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        E_ei_up, E_ei_dn = compute_ewald_potential_ei(coulomb, r_up, r_dn, ewald_data)
        E_ei = float(np.sum(np.asarray(E_ei_up)) + np.sum(np.asarray(E_ei_dn)))
        np.testing.assert_allclose(
            E_ei,
            PYMATGEN_E_EI_H4_SLAB_HA,
            atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE,
            rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE,
        )

    @pytest.mark.light
    def test_compute_ewald_potential_ee_vs_pymatgen_h4_slab(self):
        """``compute_ewald_potential_ee`` matches pymatgen on the neutral H4
        layer in the vacuum slab cell (regeneration script shared with
        :py:func:`test_compute_ewald_potential_ei_vs_pymatgen_h4_slab`)."""
        structure, coulomb, r_up, r_dn = _make_h4_slab_box_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        E_ee = float(compute_ewald_potential_ee(r_up, r_dn, ewald_data))
        np.testing.assert_allclose(
            E_ee,
            PYMATGEN_E_EE_H4_SLAB_HA,
            atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE,
            rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE,
        )

    @pytest.mark.light
    def test_compute_ewald_potential_ei_vs_pymatgen_h4_slab_vacuum_electron(self):
        """e-i Ewald with one electron isolated 23 Bohr deep in the vacuum
        (the point-charge model of the 19-atom slab ``far`` regime) matches
        pymatgen. Regeneration script shared with
        :py:func:`test_compute_ewald_potential_ei_vs_pymatgen_h4_slab`."""
        structure, coulomb, r_up, r_dn = _make_h4_slab_vacuum_electron_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        E_ei_up, E_ei_dn = compute_ewald_potential_ei(coulomb, r_up, r_dn, ewald_data)
        E_ei = float(np.sum(np.asarray(E_ei_up)) + np.sum(np.asarray(E_ei_dn)))
        np.testing.assert_allclose(
            E_ei,
            PYMATGEN_E_EI_H4_SLAB_VACUUM_ELECTRON_HA,
            atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE,
            rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE,
        )

    @pytest.mark.light
    def test_compute_ewald_potential_ee_vs_pymatgen_h4_slab_vacuum_electron(self):
        """e-e Ewald with one electron isolated 23 Bohr deep in the vacuum
        matches pymatgen. Regeneration script shared with
        :py:func:`test_compute_ewald_potential_ei_vs_pymatgen_h4_slab`."""
        structure, coulomb, r_up, r_dn = _make_h4_slab_vacuum_electron_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        E_ee = float(compute_ewald_potential_ee(r_up, r_dn, ewald_data))
        np.testing.assert_allclose(
            E_ee,
            PYMATGEN_E_EE_H4_SLAB_VACUUM_ELECTRON_HA,
            atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE,
            rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE,
        )

    @pytest.mark.light
    def test_compute_ewald_potential_total_vs_pymatgen_h4_slab_vacuum_electron(self):
        """``E_ii + sum(E_ei) + E_ee`` (charge-neutral, hence background-free)
        matches pymatgen's ``total_energy_matrix`` sum with the vacuum
        electron -- the end-to-end gate for the slab regime, independent of
        how the neutralising background is attributed per block.
        Regeneration script shared with
        :py:func:`test_compute_ewald_potential_ei_vs_pymatgen_h4_slab`."""
        structure, coulomb, r_up, r_dn = _make_h4_slab_vacuum_electron_fixture()
        ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
        E_ii = float(compute_ewald_potential_ii(coulomb, ewald_data))
        E_ei_up, E_ei_dn = compute_ewald_potential_ei(coulomb, r_up, r_dn, ewald_data)
        E_total = (
            E_ii
            + float(np.sum(np.asarray(E_ei_up)) + np.sum(np.asarray(E_ei_dn)))
            + float(compute_ewald_potential_ee(r_up, r_dn, ewald_data))
        )
        np.testing.assert_allclose(
            E_total,
            PYMATGEN_E_TOTAL_H4_SLAB_VACUUM_ELECTRON_HA,
            atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE,
            rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE,
        )


def _make_silicon_ewald_ion_fixture() -> tuple[Structure_data, Coulomb_potential_data]:
    """Load only the inputs consumed by the ion-only Ewald identity.

    Keeping the electron-free adapter separate prevents deterministic electron
    coordinates used by the e-e/e-i siblings from becoming a fictitious part
    of the ion-ion GDD fixture contract.
    """
    if not os.path.exists(PERIODIC_FIXTURE_PATH):
        pytest.skip(f"Periodic TREXIO fixture {PERIODIC_FIXTURE} is not present.")
    structure, _aos, _mos_up, _mos_dn, _geminal, coulomb = read_trexio_file(
        trexio_file=PERIODIC_FIXTURE_PATH,
        store_tuple=True,
    )
    return structure, coulomb
