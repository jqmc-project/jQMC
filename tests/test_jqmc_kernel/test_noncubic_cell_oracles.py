"""Independent periodic-kernel oracles on NON-orthogonal cells.

Every committed periodic TREXIO/QE fixture has a diagonal cell, so a metric slip
(``cell`` vs ``cell.T``, an orthogonal-only rounding, a per-axis image bound) is
invisible to the file-driven suite. This module checks four quantities on
triclinic, primitive-FCC, hexagonal and monoclinic cells against references that
do not reuse production image lists or lattice cutoffs:

(a) periodic pGTO values, Gamma and a twist, against a brute-force image sum of the
    molecular AO kernel (itself pinned to a plain-numpy Cartesian GTO formula);
(b) the periodic Cartesian overlap matrix against a brute-force whole-space
    primitive overlap sum (Gauss-Hermite per axis);
(c) the local energy of a committed TREXIO system is unchanged when the same
    lattice is written in a different, skewed cell basis ``U @ cell``
    (``U`` integer, ``det U = 1``): Gamma all-electron, Gamma ccECP, twist ccECP,
    and a genuinely triclinic lattice (mildly and strongly skewed bases); the Ewald
    energy on the strongly skewed triclinic basis is also checked against an
    independent numpy Ewald sum, with a complete G-vector count;
(d) the non-local ECP energy and e_L of an electron placed just across a skewed
    face equal those of its wrapped image.

Checks (a)-(d) keep every nucleus strictly inside the home cell. Section (e)
moves nuclei outside the home cell, onto its boundary and more than one cell away,
and moves a corner nucleus through a face (the image set must be chosen and
placed for the same, unwrapped centre).

Judgement: fixed inputs, each precision mode against its own zone policy. The AO
comparison adds ``AO_IMAGE_TRUNCATION_ATOL``, the accuracy production states for
its truncated image sum. Negative controls at the end show each check fails, with
its own message: a halved AO or overlap image radius, a transposed AO image metric
(cell.T), an orthogonal-only minimum image, a transposed home-cell wrap, and a
non-local ECP projector without the minimum image, and an orthogonal-only reciprocal
index box.
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
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance, get_tolerance_min
from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL
from jqmc.atomic_orbital import (
    _apply_image_expansion_complex,
    compute_AOs,
    compute_AOs_laplacian,
    compute_overlap_matrix,
)
from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import Jastrow_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests.conftest import CELL_FIXTURES
from tests.test_jqmc_kernel import _noncubic_oracles as ref

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_CELLS = {
    "triclinic_skewed": CELL_FIXTURES["triclinic_skewed"],
    "primitive_fcc_si": CELL_FIXTURES["primitive_fcc_si"],
    "hexagonal": ref.HEXAGONAL_CELL,
    "monoclinic": ref.MONOCLINIC_CELL,
    "skewed_basis_cubic6": ref.SKEWED_BASIS_CUBIC6_CELL,
}
_TWIST = (0.25, -0.1, 0.4)
_RT = np.eye(3)
_TREXIO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")
_E_L_ZONES = ("ao_eval", "ao_grad_lap", "mo_eval", "det_eval", "det_grad_lap", "wf_kinetic", "coulomb", "ewald")
_E_L_ZONES += ("det_ratio", "wf_ratio", "local_energy")


def _assert_finite(*arrays):
    for a in arrays:
        assert np.all(np.isfinite(np.asarray(a))), "non-finite value in a compared operand"


# --------------------------------------------------------------------------- #
# (a) periodic pGTO values vs a brute-force image sum
# --------------------------------------------------------------------------- #
def _periodic_leaf(cell, kind, k):
    positions = ref.ATOM_FRACTIONS @ cell
    structure = ref.build_structure(cell, positions)
    base = ref.build_aos(structure, kind)
    return positions, _apply_image_expansion_complex(base, structure, k_point=k, tol=AO_IMAGE_TRUNCATION_ATOL)


def _electrons(cell, seed, n=6):
    # Fractional coordinates in [-0.3, 1.3): every electron is wrapped by production.
    return np.random.default_rng(seed).uniform(-0.3, 1.3, size=(n, 3)) @ cell


def check_aos_against_bruteforce(cell_name: str, kind: str, k):
    """Raise ``AssertionError('periodic AO brute-force mismatch ...')`` on disagreement.

    The brute-force image sum is reduced in the ao_eval zone dtype, as production folds its images.
    """
    cell = _CELLS[cell_name]
    positions, leaf = _periodic_leaf(cell, kind, k)
    r = _electrons(cell, 20260927)
    prod = np.asarray(compute_AOs(aos_data=leaf, r_carts=jnp.asarray(r)))
    brute, n_images = ref.bruteforce_periodic_aos(cell, positions, kind, r, k)
    _assert_finite(prod, brute)
    assert n_images > 27, f"brute-force sum is not a lattice sum ({n_images} images)"
    ao_atol, ao_rtol = get_tolerance("ao_eval", "strict")
    # Production states that the truncated image sum reproduces the AO values to
    # AO_IMAGE_TRUNCATION_ATOL; the zone budget is added on top.
    atol = AO_IMAGE_TRUNCATION_ATOL + ao_atol
    if not np.allclose(prod, brute, atol=atol, rtol=ao_rtol, equal_nan=False):
        raise AssertionError(
            f"periodic AO brute-force mismatch on {cell_name}/{kind}/k={k}: "
            f"max|prod-brute|={np.max(np.abs(prod - brute)):.3e}, max|brute|={np.max(np.abs(brute)):.3e}"
        )
    return prod, brute


@pytest.mark.parametrize("kind", ["cart", "sphe"])
@pytest.mark.parametrize("cell_name", list(_CELLS))
def test_periodic_pgto_values_match_bruteforce_image_sum(cell_name, kind):
    """Periodic pGTO values on a skewed cell equal a brute-force lattice sum, at Gamma and a twist.

    Two atoms at fractional (0.1,0.2,0.15) and (0.7,0.55,0.8), s(1.3,0.35)/p/d/f shells,
    six electrons at seeded fractional positions in [-0.3,1.3). The reference sums
    exp(2 pi i k.n) chi(r - R - n.cell) with the open-boundary AO kernel over every image
    inside a Gaussian cutoff derived from the smallest exponent (tail < e^-80), not over
    the production image list. The Cartesian molecular kernel is pinned to a plain-numpy
    normalized GTO first. Both references run in the ao_eval zone dtype (r - R
    formed in fp64 and cast; the numpy GTO, the Bloch phase and the image sum in
    float32/complex64 in mixed, fp64 in full), so mixed is not judged against fp64.
    Tolerance: the ao_eval strict zone plus AO_IMAGE_TRUNCATION_ATOL, the accuracy
    production states for its truncated sum.
    """
    cell = _CELLS[cell_name]
    # Non-orthogonality guard: the Gram matrix has off-diagonal entries.
    gram = cell @ cell.T
    assert np.max(np.abs(gram - np.diag(np.diag(gram)))) > 1.0, f"{cell_name} is orthogonal"
    if kind == "cart":
        positions = ref.ATOM_FRACTIONS @ cell
        pts = _electrons(cell, 7, n=40)
        mol = np.asarray(
            compute_AOs(aos_data=ref.build_aos(ref.build_structure(None, positions), "cart"), r_carts=jnp.asarray(pts))
        )
        numpy_vals = ref.numpy_cart_aos(positions, pts)
        _assert_finite(mol, numpy_vals)
        atol, rtol = get_tolerance("ao_eval", "strict")
        np.testing.assert_allclose(mol, numpy_vals, atol=atol, rtol=rtol, equal_nan=False)
    for k in (None, _TWIST):
        prod, brute = check_aos_against_bruteforce(cell_name, kind, k)
        if k is not None:
            assert np.max(np.abs(brute.imag)) > 1.0e-2, "twist reference is effectively real"
    jax.clear_caches()


# --------------------------------------------------------------------------- #
# (b) periodic overlap vs a brute-force analytic lattice sum
# --------------------------------------------------------------------------- #
def check_overlap_against_bruteforce(cell_name: str, k):
    """Raise ``AssertionError('periodic overlap brute-force mismatch ...')`` on disagreement."""
    cell = _CELLS[cell_name]
    positions, leaf = _periodic_leaf(cell, "cart", k)
    prod = np.asarray(compute_overlap_matrix(leaf))
    brute = ref.bruteforce_periodic_cart_overlap(cell, positions, k)
    _assert_finite(prod, brute)
    atol, rtol = get_tolerance("det_eval", "strict")
    if not np.allclose(prod, brute, atol=atol, rtol=rtol, equal_nan=False):
        raise AssertionError(
            f"periodic overlap brute-force mismatch on {cell_name}/cart/k={k}: max|prod-brute|={np.max(np.abs(prod - brute)):.3e}"
        )
    return prod, brute


@pytest.mark.parametrize("cell_name", list(_CELLS))
def test_periodic_overlap_matches_bruteforce_lattice_sum(cell_name):
    """The periodic Cartesian overlap on a skewed cell equals a brute-force whole-space overlap sum.

    Same basis and cells as the AO test, Gamma and k=(0.25,-0.1,0.4). The reference
    sums exp(2 pi i k.n) <chi_mu(r-R_mu)|chi_nu(r-R_nu-n.cell)> over every image inside
    its own product-decay cutoff, each term by 12-point Gauss-Hermite quadrature per
    axis (exact for these degrees), with its own double-factorial normalization.
    Tolerance: det_eval strict (the overlap zone). Spherical leaves reach the same
    Cartesian lattice sum through a cell-independent cart->sphe transform, so the
    lattice-sum claim is made on the Cartesian leaf only.
    """
    for k in (None, _TWIST):
        _, brute = check_overlap_against_bruteforce(cell_name, k)
        off = brute - np.diag(np.diag(brute))
        assert np.max(np.abs(off)) > 1.0e-2, "overlap has no inter-centre content"
    jax.clear_caches()


# --------------------------------------------------------------------------- #
# (c) local energy invariant under a unimodular change of cell basis
# --------------------------------------------------------------------------- #
#: Strong skew: b1 = a1, b2 = a1 + a2, b3 = a1 + a2 + a3.
_U = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.float64)
#: Mild skew: b3 = a1 + a3. The triclinic lattice is checked in both bases; the strong
#: basis there catches an Ewald G-enumeration bounded for orthogonal cells only.
_U_MILD = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=np.float64)
_TRICLINIC_LATTICE = np.array([[15.1178, 0.0, 0.0], [4.2, 13.9, 0.0], [3.1, 2.6, 13.4]], dtype=np.float64)
#: case -> (TREXIO stem, physical lattice override or None, unimodular basis change)
_BASIS_CASES = {
    "h4_gamma_cubic_lattice": ("H4_pbc_ae_cart.h5", None, _U),
    "li_gamma_ccecp": ("lithium.h5", None, _U),
    "li_twist_ccecp": ("lithium_k025.h5", None, _U),
    "h4_gamma_triclinic_lattice": ("H4_pbc_ae_cart.h5", _TRICLINIC_LATTICE, _U_MILD),
    "h4_twist_triclinic_lattice": ("H4_pbc_ae_k025_cart.h5", _TRICLINIC_LATTICE, _U_MILD),
    "h4_gamma_triclinic_lattice_strong_skew": ("H4_pbc_ae_cart.h5", _TRICLINIC_LATTICE, _U),
    "h4_twist_triclinic_lattice_strong_skew": ("H4_pbc_ae_k025_cart.h5", _TRICLINIC_LATTICE, _U),
}
_NUCLEUS_MARGIN = 0.05


def _jastrow_free(path):
    structure, _aos, _mu, _md, geminal, coulomb = read_trexio_file(path, store_tuple=True)
    wf = Wavefunction_data(
        jastrow_data=Jastrow_data(
            jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=None
        ),
        geminal_data=geminal,
    )
    return Hamiltonian_data(structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=wf)


@pytest.fixture(scope="module")
def basis_pairs(tmp_path_factory):
    """case -> (ham in basis L, ham in basis U @ L), built lazily through the production loader."""
    root = str(tmp_path_factory.mktemp("noncubic_basis"))
    cache = {}

    def get(case):
        if case not in cache:
            stem, lattice, unimodular = _BASIS_CASES[case]
            src = os.path.join(_TREXIO, stem)
            shift = ref.choose_nucleus_shift(src, unimodular, lattice, _NUCLEUS_MARGIN)
            same = ref.transformed_trexio_copy(src, root, f"{case}_L", np.eye(3), lattice=lattice, nucleus_shift_frac=shift)
            skew = ref.transformed_trexio_copy(src, root, f"{case}_UL", unimodular, lattice=lattice, nucleus_shift_frac=shift)
            cache[case] = (_jastrow_free(same), _jastrow_free(skew))
        return cache[case]

    return get


def _local_energy(ham, r_up, r_dn):
    return complex(np.asarray(compute_local_energy(ham, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(_RT))))


def check_basis_identity(h_same, h_skew, configs, label):
    """Raise ``AssertionError('unimodular cell-basis identity violated ...')`` on disagreement."""
    atol, rtol = get_tolerance_min(_E_L_ZONES, "strict")
    values = []
    for r_up, r_dn in configs:
        e_same = _local_energy(h_same, r_up, r_dn)
        e_skew = _local_energy(h_skew, r_up, r_dn)
        _assert_finite(e_same, e_skew)
        values.append((e_same, e_skew))
        if not np.isclose(e_skew, e_same, atol=atol, rtol=rtol, equal_nan=False):
            raise AssertionError(
                f"unimodular cell-basis identity violated for {label}: e_L(L)={e_same!r}, e_L(U L)={e_skew!r}, "
                f"|diff|={abs(e_skew - e_same):.3e}"
            )
    return values


def _basis_configs(ham, seed, n=3):
    geminal = ham.wavefunction_data.geminal_data
    nu, nd = int(geminal.num_electron_up), int(geminal.num_electron_dn)
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return [(rng.uniform(-0.5, 1.5, size=(nu, 3)) @ cell, rng.uniform(-0.5, 1.5, size=(nd, 3)) @ cell) for _ in range(n)]


@pytest.mark.parametrize("case", list(_BASIS_CASES))
def test_local_energy_invariant_under_unimodular_change_of_cell_basis(case, basis_pairs):
    """e_L is unchanged when the same lattice is written in a skewed basis U @ L (det U = 1).

    The committed TREXIO file is rewritten twice through h5py: once in its own basis L
    and once in U @ L (k_frac -> U k_frac, so every Bloch phase is unchanged), with
    b1=a1, b2=a1+a2, b3=a1+a2+a3 on the cubic lattices; the triclinic cases first move
    the system onto a triclinic lattice at the same fractional atom positions and use
    both b3=a1+a3 and the strong basis above (the strong triclinic rows catch an Ewald
    G-enumeration bounded for orthogonal cells only). Nuclei are shifted rigidly (same Cartesian
    coordinates in both files) so they lie inside both home cells. Both files go through
    the production loader (image truncation, Ewald tables, ECP); the Jastrow is omitted
    because the J1/J2 sin(pi s) distance is basis dependent by construction. Three
    seeded configurations with electrons outside the home cell; full complex e_L, zone
    strict policy.
    """
    h_same, h_skew = basis_pairs(case)
    cell = np.asarray(h_same.structure_data.cell, dtype=np.float64)
    skew = np.asarray(h_skew.structure_data.cell, dtype=np.float64)
    np.testing.assert_array_equal(skew, _BASIS_CASES[case][2] @ cell)
    gram = skew @ skew.T
    assert np.max(np.abs(gram[np.triu_indices(3, 1)])) > 1.0, "the rewritten basis is not skewed"
    values = check_basis_identity(h_same, h_skew, _basis_configs(h_same, 31), case)
    if "twist" in case:
        assert max(abs(e.imag) for e, _ in values) > 1.0e-3, "twist e_L is effectively real"
    jax.clear_caches()


# --------------------------------------------------------------------------- #
# (c') Ewald energy on a strongly skewed basis vs an independent numpy Ewald sum
# --------------------------------------------------------------------------- #
#: Fractional positions of the neutral Ewald fixture: four protons, two up and two
#: down electrons (all inside the home cell).
_EWALD_NUCLEI_FRAC = np.array([[0.1, 0.2, 0.15], [0.7, 0.55, 0.8], [0.35, 0.6, 0.3], [0.8, 0.1, 0.5]])
_EWALD_UP_FRAC = np.array([[0.2, 0.3, 0.1], [0.65, 0.5, 0.9]])
_EWALD_DN_FRAC = np.array([[0.4, 0.7, 0.35], [0.9, 0.2, 0.45]])
_EWALD_BUILD_TOL = 1.0e-12
_EWALD_BASES = {"triclinic_L": np.eye(3), "triclinic_strong_skew_UL": _U}


def check_ewald_against_numpy(basis_name):
    """Raise ``AssertionError('Ewald vs numpy Ewald mismatch ...')`` on disagreement.

    Returns ``(n_g_enumerated, n_g_complete, e_prod, e_ref)``.
    """
    from jqmc._setting import ATOL_EWALD_VS_EXTERNAL_REFERENCE, RTOL_EWALD_VS_EXTERNAL_REFERENCE
    from jqmc.coulomb_potential import (
        Coulomb_potential_data,
        build_ewald_data,
        compute_ewald_potential_ee,
        compute_ewald_potential_ei,
        compute_ewald_potential_ii,
    )

    lattice = _TRICLINIC_LATTICE
    cell = _EWALD_BASES[basis_name] @ lattice
    R, r_up, r_dn = _EWALD_NUCLEI_FRAC @ lattice, _EWALD_UP_FRAC @ lattice, _EWALD_DN_FRAC @ lattice
    x = np.concatenate([R, r_up, r_dn])
    q = np.concatenate([np.ones(len(R)), -np.ones(len(r_up) + len(r_dn))])
    e_ref = ref.numpy_ewald_energy(lattice, x, q)
    structure = ref.build_structure(cell, R)
    coulomb = Coulomb_potential_data(structure_data=structure)
    ewald = build_ewald_data(structure, coulomb, tol=_EWALD_BUILD_TOL, n_electron=len(r_up) + len(r_dn))
    e_ii = float(np.asarray(compute_ewald_potential_ii(coulomb, ewald)))
    ei_up, ei_dn = compute_ewald_potential_ei(coulomb, jnp.asarray(r_up), jnp.asarray(r_dn), ewald)
    e_ee = float(np.asarray(compute_ewald_potential_ee(jnp.asarray(r_up), jnp.asarray(r_dn), ewald)))
    e_prod = e_ii + float(np.sum(np.asarray(ei_up))) + float(np.sum(np.asarray(ei_dn))) + e_ee
    _assert_finite(e_prod, e_ref)
    # Independent count of the non-zero reciprocal vectors with |G| <= g_max.
    recip = 2.0 * np.pi * np.linalg.inv(cell).T
    idx = ref.lattice_images_within(recip, np.zeros((1, 3)), float(ewald.g_max))
    n_complete = int(np.sum(np.any(idx != 0, axis=1)))
    n_enum = int(np.asarray(ewald.g_vectors).shape[0])
    if n_enum != n_complete or not np.isclose(
        e_prod, e_ref, atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE, rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE
    ):
        raise AssertionError(
            f"Ewald vs numpy Ewald mismatch for {basis_name}: G enumerated {n_enum} of {n_complete}, "
            f"E_prod={e_prod!r}, E_ref={e_ref!r}, diff={e_prod - e_ref:.3e}"
        )
    return n_enum, n_complete, e_prod, e_ref


@pytest.mark.parametrize("basis_name", list(_EWALD_BASES))
def test_ewald_energy_on_strongly_skewed_basis_matches_numpy_ewald(basis_name):
    """Ewald ii+ei+ee on a triclinic lattice, in its own and a strongly skewed basis, equals a numpy Ewald sum.

    The same triclinic lattice is written as L and as U @ L (b1=a1, b2=a1+a2,
    b3=a1+a2+a3). A neutral set of four protons and four electrons goes through
    build_ewald_data (tol 1e-12) and the three production Ewald channels; the reference
    is ref.numpy_ewald_energy on the physical lattice (its own eta, r_cut and g_cut,
    image lists from a brute-force cube). The number of enumerated G vectors must equal
    an independent count of the non-zero reciprocal vectors with |G| <= g_max, and the
    energy must agree within ATOL/RTOL_EWALD_VS_EXTERNAL_REFERENCE.
    """
    n_enum, _n_complete, _e, _r = check_ewald_against_numpy(basis_name)
    assert n_enum > 100, f"reciprocal sum is trivially small ({n_enum} G vectors)"
    jax.clear_caches()


# --------------------------------------------------------------------------- #
# (d) non-local ECP with an electron across a skewed face
# --------------------------------------------------------------------------- #
def _across_face_config(ham, seed):
    """Electron configuration whose first up electron sits just outside a home-cell face.

    The electron is 0.35 Bohr beyond the face nearest to a nucleus (so the ECP
    projector of that nucleus is live), along the face's inward normal reversed.
    Returns the unwrapped configuration and the one with that electron moved back
    by the crossing lattice vector.
    """
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    s = ref.fractional(cell, R)
    recip_unit = np.linalg.inv(cell).T / np.linalg.norm(np.linalg.inv(cell).T, axis=1, keepdims=True)
    heights = abs(float(np.linalg.det(cell))) / np.array(
        [np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)]
    )
    dist_low, dist_high = s * heights, (1.0 - s) * heights
    atom, axis, high = min(
        ((a, i, h) for a in range(len(R)) for i in range(3) for h in (0, 1)),
        key=lambda t: (dist_high if t[2] else dist_low)[t[0], t[1]],
    )
    to_face = (dist_high if high else dist_low)[atom, axis]
    normal = recip_unit[axis] * (1.0 if high else -1.0)
    e = R[atom] + (to_face + 0.35) * normal
    crossing = cell[axis] * (1.0 if high else -1.0)
    rng = np.random.default_rng(seed)
    geminal = ham.wavefunction_data.geminal_data
    nu, nd = int(geminal.num_electron_up), int(geminal.num_electron_dn)
    r_up = rng.uniform(0.1, 0.9, size=(nu, 3)) @ cell
    r_dn = rng.uniform(0.1, 0.9, size=(nd, 3)) @ cell
    r_up[0] = e
    wrapped = r_up.copy()
    wrapped[0] = e - crossing
    s_out, s_in = ref.fractional(cell, e), ref.fractional(cell, wrapped[0])
    assert (s_out[axis] > 1.0 or s_out[axis] < 0.0) and 0.0 <= s_in[axis] < 1.0, "electron did not cross the face"
    return (r_up, r_dn), (wrapped, r_dn)


def check_ecp_wrap(ham, unwrapped, wrapped, label):
    """Raise ``AssertionError('wrapped/unwrapped non-local ECP mismatch ...')`` on disagreement."""
    atol, rtol = get_tolerance_min(_E_L_ZONES, "strict")
    out = []
    for r_up, r_dn in (unwrapped, wrapped):
        v_nl = complex(
            np.asarray(
                compute_ecp_non_local_parts_nearest_neighbors(
                    ham.coulomb_potential_data, ham.wavefunction_data, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(_RT)
                )[-1]
            )
        )
        out.append((v_nl, _local_energy(ham, r_up, r_dn)))
    (v0, e0), (v1, e1) = out
    _assert_finite(v0, e0, v1, e1)
    # One guard per quantity, so each is its own declared comparison (same message).
    if not np.isclose(v1, v0, atol=atol, rtol=rtol, equal_nan=False):
        raise AssertionError(
            f"wrapped/unwrapped non-local ECP mismatch for {label}: V_nl {v0!r} vs {v1!r}, e_L {e0!r} vs {e1!r}"
        )
    if not np.isclose(e1, e0, atol=atol, rtol=rtol, equal_nan=False):
        raise AssertionError(
            f"wrapped/unwrapped non-local ECP mismatch for {label}: V_nl {v0!r} vs {v1!r}, e_L {e0!r} vs {e1!r}"
        )
    return v0, e0


@pytest.mark.parametrize("case", ["li_gamma_ccecp", "li_twist_ccecp"])
def test_ecp_nonlocal_electron_across_skewed_face_equals_wrapped(case, basis_pairs):
    """V_nl and e_L of an electron just across a skewed face equal those of its wrapped image.

    Lithium ccECP (Gamma and k=0.25 twist) rewritten in the skewed basis U @ L as in the
    basis test. The first up electron sits 0.35 Bohr outside the face nearest to a
    nucleus; the reference is the same electron moved back by the crossing lattice
    vector. The non-local sum (nearest-neighbour projector) and the full complex e_L
    must agree within the zone strict policy; the projector must be live.
    """
    _, ham = basis_pairs(case)
    unwrapped, wrapped = _across_face_config(ham, 5)
    v_nl, _ = check_ecp_wrap(ham, unwrapped, wrapped, case)
    atol, _ = get_tolerance_min(_E_L_ZONES, "strict")
    assert abs(v_nl) > 1.0e3 * atol, f"non-local ECP is inert at the face configuration ({v_nl!r})"
    jax.clear_caches()


# --------------------------------------------------------------------------- #
# (e) nuclei outside the home cell, on its boundary, and moving through a face
# --------------------------------------------------------------------------- #
_PLACEMENT_CELLS = {
    "cubic_10bohr": CELL_FIXTURES["cubic_10bohr"],
    "triclinic_skewed": CELL_FIXTURES["triclinic_skewed"],
    "hexagonal": ref.HEXAGONAL_CELL,
}


def _placed_leaf(cell, positions, kind, k, tol=AO_IMAGE_TRUNCATION_ATOL):
    structure = ref.build_structure(cell, positions)
    base = ref.build_aos(structure, kind)
    return _apply_image_expansion_complex(base, structure, k_point=k, tol=tol)


def check_placed_aos_against_bruteforce(cell, positions, kind, k, label, quantity="value", tol=AO_IMAGE_TRUNCATION_ATOL):
    """Raise ``AssertionError('placed-nucleus AO brute-force mismatch ...')`` on disagreement.

    ``quantity`` is ``"value"`` (zone ao_eval) or ``"laplacian"`` (zone ao_grad_lap);
    ``tol`` is the image truncation tolerance the leaf is built with, added to the
    zone atol. The brute-force image sum is reduced in that zone's dtype.
    """
    leaf = _placed_leaf(cell, positions, kind, k, tol)
    r = _electrons(cell, 20260927)
    evaluate = compute_AOs if quantity == "value" else compute_AOs_laplacian
    prod = np.asarray(evaluate(aos_data=leaf, r_carts=jnp.asarray(r)))
    brute, n_images = ref.bruteforce_periodic_aos(cell, positions, kind, r, k, quantity=quantity)
    _assert_finite(prod, brute)
    assert n_images > 27, f"brute-force sum is not a lattice sum ({n_images} images)"
    zone_atol, zone_rtol = get_tolerance("ao_eval" if quantity == "value" else "ao_grad_lap", "strict")
    atol = tol + zone_atol
    if not np.allclose(prod, brute, atol=atol, rtol=zone_rtol, equal_nan=False):
        raise AssertionError(
            f"placed-nucleus AO brute-force mismatch on {label}/{kind}/k={k}/{quantity}: "
            f"max|prod-brute|={np.max(np.abs(prod - brute)):.3e}, max|brute|={np.max(np.abs(brute)):.3e}"
        )
    return prod, brute


@pytest.mark.parametrize("placement", ["outside", "boundary", "far"])
@pytest.mark.parametrize("kind", ["cart", "sphe"])
@pytest.mark.parametrize("cell_name", list(_PLACEMENT_CELLS))
def test_periodic_pgto_values_with_nuclei_off_the_home_cell_match_bruteforce(cell_name, kind, placement):
    """Periodic pGTO values equal the brute-force lattice sum when nuclei leave the home cell.

    Same basis, electrons and reference as test (a), on a cubic, a triclinic and a
    hexagonal cell, with the two nuclei at the fractional placements of
    ref.PLACEMENT_FRACTIONS: just outside a lower and an upper face (-0.1, 1.1),
    exactly on a corner and an upper face (0 and 1), and more than one cell away
    (-1.1, 2.15). The positions are passed to production unwrapped, as a user file
    may contain them. Gamma and k=(0.25,-0.1,0.4); ao_eval strict plus
    AO_IMAGE_TRUNCATION_ATOL.
    """
    cell = _PLACEMENT_CELLS[cell_name]
    positions = ref.PLACEMENT_FRACTIONS[placement] @ cell
    for k in (None, _TWIST):
        _, brute = check_placed_aos_against_bruteforce(cell, positions, kind, k, f"{cell_name}/{placement}")
        if k is not None:
            assert np.max(np.abs(brute.imag)) > 1.0e-2, "twist reference is effectively real"
    jax.clear_caches()


@pytest.mark.parametrize("placement", ["outside", "boundary", "far"])
@pytest.mark.parametrize("cell_name", list(_PLACEMENT_CELLS))
def test_periodic_overlap_with_nuclei_off_the_home_cell_matches_bruteforce(cell_name, placement):
    """The periodic Cartesian overlap equals the brute-force lattice sum when nuclei leave the home cell.

    Placements and cells as in the AO test above; reference and tolerance as in
    test (b) (det_eval strict), Gamma and k=(0.25,-0.1,0.4). The "far" placement puts
    the two nuclei more than one cell apart in fractional coordinates, beyond the
    cell-cell image box of a same-cell pair.
    """
    cell = _PLACEMENT_CELLS[cell_name]
    positions = ref.PLACEMENT_FRACTIONS[placement] @ cell
    for k in (None, _TWIST):
        leaf = _placed_leaf(cell, positions, "cart", k)
        prod = np.asarray(compute_overlap_matrix(leaf))
        brute = ref.bruteforce_periodic_cart_overlap(cell, positions, k)
        _assert_finite(prod, brute)
        atol, rtol = get_tolerance("det_eval", "strict")
        if not np.allclose(prod, brute, atol=atol, rtol=rtol, equal_nan=False):
            raise AssertionError(
                f"placed-nucleus overlap brute-force mismatch on {cell_name}/{placement}/k={k}: "
                f"max|prod-brute|={np.max(np.abs(prod - brute)):.3e}"
            )
    jax.clear_caches()


#: Displacements (Bohr) of a corner nucleus along -a / +a through the a-face.
_FACE_STEPS = (-1.0e-3, -5.0e-5, 0.0, 5.0e-5, 1.0e-3)
#: Image truncation for the Laplacian comparison. Production states its truncation
#: bound (AO_IMAGE_TRUNCATION_ATOL) for AO VALUES; the Laplacian of a dropped tail
#: term carries extra (2 alpha r)^2-type factors, so the Laplacian leaf is built
#: with this tighter truncation and the check isolates the image-set choice.
_FACE_LAPLACIAN_TOL = AO_IMAGE_TRUNCATION_ATOL**2
#: Cells of the face-crossing (seam) check: the placement cells plus the primitive FCC cell
#: (small inradius, strongly non-orthogonal metric).
_FACE_CROSSING_CELLS = {**_PLACEMENT_CELLS, "primitive_fcc_si": CELL_FIXTURES["primitive_fcc_si"]}


def check_face_crossing(cell_name, kind, k):
    """AO value and Laplacian vs displacement of a corner nucleus through a face, layout rebuilt each step.

    Raises ``AssertionError('placed-nucleus AO brute-force mismatch ...')`` at the
    first step that disagrees with the brute-force sum.
    """
    cell = _FACE_CROSSING_CELLS[cell_name]
    unit_a = cell[0] / np.linalg.norm(cell[0])
    base = np.array([[0.0, 0.0, 0.0], [0.7, 0.55, 0.8]]) @ cell
    kinetic = []
    for h in _FACE_STEPS:
        positions = base.copy()
        positions[0] = positions[0] + h * unit_a
        label = f"{cell_name}/face-step h={h:+.1e}"
        check_placed_aos_against_bruteforce(cell, positions, kind, k, label)
        lap, _ = check_placed_aos_against_bruteforce(
            cell, positions, kind, k, label, quantity="laplacian", tol=_FACE_LAPLACIAN_TOL
        )
        kinetic.append(-0.5 * np.sum(lap))
    return kinetic


@pytest.mark.parametrize("kind", ["cart", "sphe"])
@pytest.mark.parametrize("cell_name", ["cubic_10bohr", "triclinic_skewed", "primitive_fcc_si"])
def test_ao_value_and_laplacian_continuous_as_a_corner_nucleus_crosses_a_face(cell_name, kind):
    """A nucleus on the cell corner moved through a face gives AO values and Laplacians equal to the lattice sum.

    Nucleus 0 starts at fractional (0,0,0) and is displaced by h in
    (-1e-3, -5e-5, 0, 5e-5, 1e-3) Bohr along the unit a vector, so it leaves the
    home cell for h < 0; nucleus 1 stays at (0.7,0.55,0.8). At every h the pGTO leaf
    (its image layout included) is rebuilt from scratch, as a geometry update does.
    AO values (ao_eval strict plus AO_IMAGE_TRUNCATION_ATOL, production truncation)
    and AO Laplacians (ao_grad_lap strict plus the truncation tolerance, leaf built
    with the tighter truncation AO_IMAGE_TRUNCATION_ATOL**2 because the production
    bound is stated for values) must equal the brute-force lattice sum at every step,
    Gamma and k=(0.25,-0.1,0.4); the brute-force sum is smooth in h, so a jump in
    the kept image set at the face fails this check.
    """
    for k in (None, _TWIST):
        kinetic = check_face_crossing(cell_name, kind, k)
        assert len(kinetic) == len(_FACE_STEPS)
    jax.clear_caches()


# --------------------------------------------------------------------------- #
# (f) periodic ECP local energy on cubic, FCC and triclinic lattices: analytic vs _auto and debug
# --------------------------------------------------------------------------- #
#: Physical lattice of each row (None: the file's own cubic cell). The FCC row is the
#: primitive FCC cell of the Si fixture; the triclinic row is the lattice of the basis test.
_ECP_LATTICES = {
    "cubic": None,
    "primitive_fcc": CELL_FIXTURES["primitive_fcc_si"],
    "triclinic": _TRICLINIC_LATTICE,
}
#: Lattice translations (fractional) of the nucleus images the electrons are placed at;
#: every electron is therefore outside the home cell and inside an ECP core of an image.
_ECP_IMAGE_SHIFTS = np.array([[1, 0, 0], [0, -1, 1], [-1, 1, -1]], dtype=np.float64)
#: Distance (Bohr) of an electron from its nucleus image: inside the Li ccECP core.
_ECP_CORE_OFFSET = 0.6


@pytest.fixture(scope="module")
def ecp_lattice_ham(tmp_path_factory):
    """lattice -> Li ccECP (Gamma, no Jastrow) on that lattice through the production loader (lazy, cached)."""
    root = str(tmp_path_factory.mktemp("ecp_lattices"))
    cache = {}

    def get(lattice_name):
        if lattice_name not in cache:
            src = os.path.join(_TREXIO, "lithium.h5")
            lattice = _ECP_LATTICES[lattice_name]
            shift = ref.choose_nucleus_shift(src, np.eye(3), lattice, _NUCLEUS_MARGIN)
            path = ref.transformed_trexio_copy(
                src, root, f"li_{lattice_name}", np.eye(3), lattice=lattice, nucleus_shift_frac=shift
            )
            cache[lattice_name] = _jastrow_free(path)
        return cache[lattice_name]

    return get


def _check_ecp_lattice(ham, lattice_name):
    """The row's cell is the declared lattice: cubic, or non-orthogonal (FCC, triclinic)."""
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    if _ECP_LATTICES[lattice_name] is None:
        np.testing.assert_array_equal(cell, np.diag(np.diag(cell)))
        assert np.ptp(np.diag(cell)) == 0.0, "the cubic row is not cubic"
    else:
        np.testing.assert_array_equal(cell, _ECP_LATTICES[lattice_name])
        gram = cell @ cell.T
        assert np.max(np.abs(gram[np.triu_indices(3, 1)])) > 1.0, "the lattice is not non-orthogonal"
    return cell


def _ecp_image_configs(ham, seed):
    """Configurations with every electron 0.6 Bohr from a lattice image of a nucleus, outside the home cell."""
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    geminal = ham.wavefunction_data.geminal_data
    nu, nd = int(geminal.num_electron_up), int(geminal.num_electron_dn)
    rng = np.random.default_rng(seed)
    configs = []
    for c, image in enumerate(_ECP_IMAGE_SHIFTS):
        electrons = []
        for i in range(nu + nd):
            direction = rng.normal(size=3)
            atom = (i + c) % len(R)
            electrons.append(R[atom] + image @ cell + _ECP_CORE_OFFSET * direction / np.linalg.norm(direction))
        electrons = np.asarray(electrons)
        s = ref.fractional(cell, electrons)
        assert np.all(np.any((s < 0.0) | (s >= 1.0), axis=1)), "an electron is inside the home cell"
        configs.append((electrons[:nu], electrons[nu:]))
    return configs


def check_periodic_ecp_local_energy_routes(ham, configs, label):
    """Raise ``AssertionError('periodic ECP route mismatch ...')`` when a route disagrees.

    Per configuration: production ``compute_local_energy`` (analytic kinetic) against
    ``_compute_local_energy_auto`` (autodiff kinetic, same potential), and the production
    ECP energy ``compute_ecp_coulomb_potential`` (JAX minimum image, rank-1 ratios) against
    ``_compute_ecp_coulomb_potential_debug`` (numpy minimum image and nearest nucleus,
    all-pairs local part, full-determinant ratios). Returns the debug ECP energies.
    """
    from jqmc.coulomb_potential import _compute_ecp_coulomb_potential_debug, compute_ecp_coulomb_potential
    from jqmc.hamiltonians import _compute_local_energy_auto

    atol, rtol = get_tolerance_min(_E_L_ZONES, "strict")
    coulomb, wavefunction = ham.coulomb_potential_data, ham.wavefunction_data
    debug_values = []
    for index, (r_up, r_dn) in enumerate(configs):
        up, dn = jnp.asarray(r_up), jnp.asarray(r_dn)
        e_analytic = _local_energy(ham, r_up, r_dn)
        e_auto = complex(np.asarray(_compute_local_energy_auto(ham, up, dn, jnp.asarray(_RT))))
        v_prod = complex(np.asarray(compute_ecp_coulomb_potential(coulomb, wavefunction, up, dn, jnp.asarray(_RT))))
        v_debug = complex(np.asarray(_compute_ecp_coulomb_potential_debug(coulomb, wavefunction, r_up, r_dn, _RT)))
        _assert_finite(e_analytic, e_auto, v_prod, v_debug)
        debug_values.append(v_debug)
        if not np.isclose(e_analytic, e_auto, atol=atol, rtol=rtol, equal_nan=False):
            raise AssertionError(
                f"periodic ECP route mismatch (analytic vs _auto e_L) for {label} config {index}: "
                f"{e_analytic!r} vs {e_auto!r}, |diff|={abs(e_analytic - e_auto):.3e}"
            )
        if not np.isclose(v_prod, v_debug, atol=atol, rtol=rtol, equal_nan=False):
            raise AssertionError(
                f"periodic ECP route mismatch (production vs debug V_ecp) for {label} config {index}: "
                f"{v_prod!r} vs {v_debug!r}, |diff|={abs(v_prod - v_debug):.3e}"
            )
    return debug_values


def _check_ecp_route_row(ham_factory, lattice):
    ham = ham_factory(lattice)
    _check_ecp_lattice(ham, lattice)
    values = check_periodic_ecp_local_energy_routes(ham, _ecp_image_configs(ham, 47), f"li_gamma_ccecp_{lattice}")
    atol, _ = get_tolerance_min(_E_L_ZONES, "strict")
    assert min(abs(v) for v in values) > 1.0e3 * atol, f"ECP energy is inert at an image configuration ({values!r})"
    jax.clear_caches()


def test_periodic_ecp_local_energy_analytic_matches_auto_and_debug_on_triclinic_lattice(ecp_lattice_ham):
    """Periodic ccECP local energy on a genuinely triclinic lattice: analytic route vs ``_auto`` and debug.

    Lithium ccECP (Gamma, Jastrow-free) moved onto the triclinic lattice of the basis test
    (the physical lattice itself is non-orthogonal, not a skewed basis of a cubic lattice)
    through the production loader; the sibling node below runs the same check on the
    file's own cubic cell and on the primitive FCC cell of the Si fixture. At three configurations every electron sits
    0.6 Bohr from a lattice image of a nucleus (translations a1, -a2+a3, -a1+a2-a3), so
    each is outside the home cell and the ECP of that image is live. The production local
    energy must equal the autodiff-kinetic route, and the production ECP energy must equal
    the numpy debug ECP energy, in the strict zone policy of the other e_L checks. The
    ECP must be live at every configuration.
    """
    _check_ecp_route_row(ecp_lattice_ham, "triclinic")


@pytest.mark.parametrize("lattice", ["cubic", "primitive_fcc"])
def test_periodic_ecp_local_energy_analytic_matches_auto_and_debug_on_cubic_and_fcc_lattices(lattice, ecp_lattice_ham):
    """Sibling rows of the triclinic node above on the cubic cell and the primitive FCC cell (same check)."""
    _check_ecp_route_row(ecp_lattice_ham, lattice)


_ECP_ROUTE_FAULTS = ["transpose-ecp-minimum-image-metric", "drop-ecp-minimum-image", "unwrapped-fused-grad-lap"]


def _check_ecp_route_fault(ecp_lattice_ham, lattice, fault, monkeypatch):
    from jqmc import atomic_orbital as ao
    from jqmc import coulomb_potential as cp
    from jqmc import structure as st

    ham = ecp_lattice_ham(lattice)
    cell = _check_ecp_lattice(ham, lattice)
    configs = _ecp_image_configs(ham, 47)
    label = f"li_gamma_ccecp_{lattice}"
    jax.clear_caches()
    message = None
    if fault == "transpose-ecp-minimum-image-metric":
        original = st._minimum_image_displacement
        monkeypatch.setattr(
            st,
            "_minimum_image_displacement",
            lambda diff, cell, inv_cell, n_max: original(diff, jnp.asarray(cell).T, jnp.asarray(inv_cell).T, n_max),
        )
        symmetric = np.array_equal(cell, cell.T)
        assert symmetric == (lattice != "triclinic"), f"unexpected cell symmetry for {lattice}"
        if not symmetric:
            message = r"periodic ECP route mismatch \(production vs debug V_ecp\)"
    elif fault == "drop-ecp-minimum-image":

        def raw(structure_data, r_cart, i_atom, dtype=None):
            return (structure_data._positions_cart_jnp[i_atom] - r_cart).astype(dtype or jnp.float64)

        monkeypatch.setattr(cp, "_get_min_dist_rel_R_cart_jnp", raw)
        message = r"periodic ECP route mismatch \(production vs debug V_ecp\)"
    else:
        cart, sphe = ao._compute_AOs_value_grad_lap_cart, ao._compute_AOs_value_grad_lap_sphe
        monkeypatch.setattr(ao.AOs_cart_gto_data, "_compute_AOs_value_grad_lap", lambda self, r_carts: cart(self, r_carts))
        monkeypatch.setattr(ao.AOs_sphe_gto_data, "_compute_AOs_value_grad_lap", lambda self, r_carts: sphe(self, r_carts))
        message = r"periodic ECP route mismatch \(analytic vs _auto e_L\)"
    if message is None:
        check_periodic_ecp_local_energy_routes(ham, configs, label)
    else:
        with pytest.raises(AssertionError, match=message):
            check_periodic_ecp_local_energy_routes(ham, configs, label)
    monkeypatch.undo()
    jax.clear_caches()


@pytest.mark.parametrize("fault", _ECP_ROUTE_FAULTS)
def test_triclinic_ecp_local_energy_routes_reject_route_faults(fault, monkeypatch, ecp_lattice_ham):
    """The ECP route check fails, with the message of the route that disagrees, for three faults.

    transpose-ecp-minimum-image-metric: the production (JAX) Wigner-Seitz minimum image
    uses cell.T and inv(cell).T; the numpy debug minimum image is untouched, so on the
    triclinic lattice the production ECP energy must differ from the debug one. The cubic
    cell and the primitive FCC cell are symmetric matrices, on which this fault is exactly
    the identity: those rows assert that the cell is symmetric and that the check still
    passes (the fault is specific to a non-symmetric metric). drop-ecp-minimum-image: the
    production ECP projector displacement R - r skips the minimum image (must fail on every
    lattice). unwrapped-fused-grad-lap: the fused AO value/gradient/Laplacian kernel of the
    analytic kinetic route evaluates the image sum at the unwrapped electron (the value
    route used by ``_auto`` still wraps into the home cell), so the analytic local energy
    must differ from the autodiff one (must fail on every lattice). The unmutated inputs
    pass the same check (test above).
    """
    _check_ecp_route_fault(ecp_lattice_ham, "triclinic", fault, monkeypatch)


@pytest.mark.parametrize("fault", _ECP_ROUTE_FAULTS)
@pytest.mark.parametrize("lattice", ["cubic", "primitive_fcc"])
def test_cubic_and_fcc_ecp_local_energy_routes_reject_route_faults(lattice, fault, monkeypatch, ecp_lattice_ham):
    """Sibling rows of the triclinic fault node above on the cubic and primitive FCC cells (same faults).

    drop-ecp-minimum-image and unwrapped-fused-grad-lap must fail. Both cells are symmetric
    matrices, so transpose-ecp-minimum-image-metric is exactly the identity there: the row
    asserts the symmetry and that the check still passes.
    """
    _check_ecp_route_fault(ecp_lattice_ham, lattice, fault, monkeypatch)


# --------------------------------------------------------------------------- #
# Negative controls
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fault",
    [
        "halve-ao-image-radius",
        "transpose-ao-image-metric",
        pytest.param(
            "halve-overlap-image-radius",
            marks=pytest.mark.gdd_electron_configuration(
                mode="not-required",
                reason=(
                    "The overlap fault branch evaluates the analytic overlap only; the one coordinate-named "
                    "symbol this case reaches is the nucleus array 'coords' read while the basis_pairs fixture builds "
                    "its Hamiltonians, not an electron or evaluation coordinate"
                ),
            ),
        ),
        "orthogonal-only-minimum-image",
        "transpose-home-cell-wrap",
        "drop-ecp-minimum-image",
        "orthogonal-only-g-index-box",
    ],
)
def test_noncubic_oracles_reject_metric_and_truncation_faults(fault, monkeypatch, basis_pairs):
    """Each oracle above fails, with its own message, for a plausible skewed-cell fault.

    halve-ao-image-radius / halve-overlap-image-radius: the AO (per-atom) or overlap
    (primitive-pair) image cutoff radius is halved, dropping needed images.
    transpose-ao-image-metric: the per-atom image selection uses cell.T (Gram metric
    cell.T @ cell). orthogonal-only-minimum-image: the minimum image becomes
    d - round(d inv_cell) cell. transpose-home-cell-wrap: the home-cell wrap uses
    inv(cell).T. drop-ecp-minimum-image: the ECP projector displacement R - r skips the
    minimum image. orthogonal-only-g-index-box: the reciprocal index box is bounded by
    ceil(g_max / min|b_i|) + 1 (a bound valid only for an orthogonal
    reciprocal basis). The transposition faults leave a cubic cell untouched, and the
    wrap fault is lattice periodic (the wrapped/unwrapped check cannot see it), so the
    basis identity is the check that must catch it. The unmutated inputs pass the same
    checks (tests above).
    """
    import jqmc.coulomb_potential as cp
    from jqmc import atomic_orbital as ao
    from jqmc import structure as st

    ham_case = "li_gamma_ccecp"
    h_same, h_skew = basis_pairs(ham_case)  # built before any mutation
    jax.clear_caches()
    if fault == "halve-ao-image-radius":
        original = st._primitive_image_radius
        monkeypatch.setattr(st, "_primitive_image_radius", lambda e, L, tol: 0.5 * original(e, L, tol))
        with pytest.raises(AssertionError, match="periodic AO brute-force mismatch on triclinic_skewed"):
            check_aos_against_bruteforce("triclinic_skewed", "cart", None)
    elif fault == "transpose-ao-image-metric":
        original = ao._primitive_image_list_indices_per_atom
        monkeypatch.setattr(
            ao, "_primitive_image_list_indices_per_atom", lambda cell, *a, **kw: original(np.asarray(cell).T, *a, **kw)
        )
        with pytest.raises(AssertionError, match="periodic AO brute-force mismatch on skewed_basis_cubic6"):
            check_aos_against_bruteforce("skewed_basis_cubic6", "cart", None)
    elif fault == "halve-overlap-image-radius":
        original = ao._primitive_image_list_indices
        # r_cut scales as exponent**-1/2: a four-fold exponent halves the radius.
        monkeypatch.setattr(ao, "_primitive_image_list_indices", lambda cell, e, *a: original(cell, 4.0 * e, *a))
        ao._PERIODIC_PGTO_LATTICE_OVERLAP_CACHE.clear()
        with pytest.raises(AssertionError, match="periodic overlap brute-force mismatch on triclinic_skewed"):
            check_overlap_against_bruteforce("triclinic_skewed", _TWIST)
        ao._PERIODIC_PGTO_LATTICE_OVERLAP_CACHE.clear()
    elif fault == "orthogonal-only-minimum-image":

        def naive(diff, cell, inv_cell, n_max):
            return diff - jnp.round(diff @ inv_cell) @ cell

        monkeypatch.setattr(st, "_minimum_image_displacement", naive)
        with pytest.raises(AssertionError, match="unimodular cell-basis identity violated for li_gamma_ccecp"):
            check_basis_identity(h_same, h_skew, _basis_configs(h_same, 31), ham_case)
    elif fault == "transpose-home-cell-wrap":
        original = st._wrap_into_cell
        monkeypatch.setattr(st, "_wrap_into_cell", lambda r, cell, inv_cell: original(r, cell, jnp.asarray(inv_cell).T))
        with pytest.raises(AssertionError, match="unimodular cell-basis identity violated for li_gamma_ccecp"):
            check_basis_identity(h_same, h_skew, _basis_configs(h_same, 31), ham_case)
    elif fault == "orthogonal-only-g-index-box":
        original = cp._enumerate_g_vectors

        def orthogonal_box(cell, g_max, eta):
            g_vectors, g_factors, g_indices = original(cell, g_max, eta)
            recip = 2.0 * np.pi * np.linalg.inv(np.asarray(cell, dtype=np.float64)).T
            bound = int(np.ceil(g_max / np.min(np.linalg.norm(recip, axis=1)))) + 1
            keep = np.all(np.abs(g_indices) <= bound, axis=1)
            return g_vectors[keep], g_factors[keep], g_indices[keep]

        monkeypatch.setattr(cp, "_enumerate_g_vectors", orthogonal_box)
        with pytest.raises(AssertionError, match="Ewald vs numpy Ewald mismatch for triclinic_strong_skew_UL"):
            check_ewald_against_numpy("triclinic_strong_skew_UL")
    else:

        def raw(structure_data, r_cart, i_atom, dtype=None):
            return (structure_data._positions_cart_jnp[i_atom] - r_cart).astype(dtype or jnp.float64)

        monkeypatch.setattr(cp, "_get_min_dist_rel_R_cart_jnp", raw)
        unwrapped, wrapped = _across_face_config(h_skew, 5)
        with pytest.raises(AssertionError, match="wrapped/unwrapped non-local ECP mismatch for li_gamma_ccecp"):
            check_ecp_wrap(h_skew, unwrapped, wrapped, ham_case)
    monkeypatch.undo()
    jax.clear_caches()


@pytest.mark.parametrize("fault", ["wrap-ao-image-selection", "drop-overlap-cell-offset"])
def test_placed_nucleus_oracles_reject_wrapped_image_selection(fault, monkeypatch):
    """The section (e) checks fail, with their own message, for the image-selection faults they target.

    wrap-ao-image-selection: the per-atom AO image set is chosen for the wrapped
    nucleus mod(R inv(cell), 1) while the centres stay at the unwrapped R; the placed-nucleus
    AO check on the "outside" placement and the
    face-crossing check (first step h=-1e-3) must both fail. drop-overlap-cell-offset:
    the overlap image set ignores the cells the two centres sit in; the "far"
    placement overlap check must fail. The unmutated inputs pass the same checks
    (tests above).
    """
    from jqmc import atomic_orbital as ao

    jax.clear_caches()
    triclinic = CELL_FIXTURES["triclinic_skewed"]
    if fault == "wrap-ao-image-selection":
        original = ao._primitive_image_list_indices_per_atom
        monkeypatch.setattr(
            ao,
            "_primitive_image_list_indices_per_atom",
            lambda cell, e, L, frac, *a, **kw: original(cell, e, L, np.mod(np.asarray(frac), 1.0), *a, **kw),
        )
        positions = ref.PLACEMENT_FRACTIONS["outside"] @ triclinic
        with pytest.raises(AssertionError, match="placed-nucleus AO brute-force mismatch on triclinic_skewed/outside"):
            check_placed_aos_against_bruteforce(triclinic, positions, "cart", None, "triclinic_skewed/outside")
        with pytest.raises(AssertionError, match="placed-nucleus AO brute-force mismatch on cubic_10bohr/face-step h=-1.0e-03"):
            check_face_crossing("cubic_10bohr", "sphe", _TWIST)
    else:
        monkeypatch.setattr(ao, "_lattice_cell_offsets", lambda cell, centers: np.zeros((len(centers), 3), dtype=np.int64))
        ao._PERIODIC_PGTO_LATTICE_OVERLAP_CACHE.clear()
        with pytest.raises(AssertionError, match="placed-nucleus overlap brute-force mismatch on triclinic_skewed/far"):
            test_periodic_overlap_with_nuclei_off_the_home_cell_matches_bruteforce("triclinic_skewed", "far")
        ao._PERIODIC_PGTO_LATTICE_OVERLAP_CACHE.clear()
    monkeypatch.undo()
    jax.clear_caches()


# --------------------------------------------------------------------------- #
# (f) issue #66: AO image selection on a non-reduced cell basis
# --------------------------------------------------------------------------- #
#: Rectangular lattice of the issue #66 reproduction (Bohr).
NON_REDUCED_AO_LATTICE = np.diag([6.0, 7.0, 8.0])
#: Unimodular basis change (det +1). The Gram matrix of ``U @ L`` is so ill-conditioned
#: that 20 coordinate-descent sweeps do not converge, and the old selector dropped the
#: per-atom image (0, 5, 17) of the alpha = 1 s primitive, 0.75 Bohr from the home cell.
NON_REDUCED_AO_TRANSFORM = np.array([[1, -6, -7], [0, 7, 11], [0, -2, -3]], dtype=np.float64)
#: The hydrogen centre of the issue report (Cartesian, Bohr; inside both home cells).
NON_REDUCED_AO_ATOM = np.array([[5.219579463313249, 5.052817284651085, 3.6767133709885487]])
#: Upper bound of every primitive envelope radius used below (Bohr); BVLS is skipped for
#: boxes provably farther away.
NON_REDUCED_AO_MAX_RADIUS = 10.0
#: Electron count (seeded, uniform in the rectangular home cell).
NON_REDUCED_AO_ELECTRONS = 400


def non_reduced_ao_electrons() -> np.ndarray:
    """Seeded electrons in the rectangular home cell (production wraps them into its own cell)."""
    return np.random.default_rng(7).uniform(0.0, 1.0, size=(NON_REDUCED_AO_ELECTRONS, 3)) @ NON_REDUCED_AO_LATTICE


def non_reduced_ao_leaf(k_frac_reduced=None):
    """Image-expanded production leaf in the basis ``U @ L``; ``k`` is given in the reduced basis.

    A lattice vector ``n_U @ (U L)`` has reduced coefficients ``n_L = n_U @ U``, so the Bloch
    phase ``k_L . n_L`` equals ``(U @ k_L) . n_U``.
    """
    from jqmc.atomic_orbital import _apply_image_expansion

    cell = NON_REDUCED_AO_TRANSFORM @ NON_REDUCED_AO_LATTICE
    structure = ref.build_structure(cell, NON_REDUCED_AO_ATOM)
    base = ref.build_aos(structure, "cart")
    if k_frac_reduced is None:
        return _apply_image_expansion(base, structure, tol=AO_IMAGE_TRUNCATION_ATOL)
    k_basis = tuple(float(x) for x in NON_REDUCED_AO_TRANSFORM @ np.asarray(k_frac_reduced, dtype=np.float64))
    return _apply_image_expansion_complex(base, structure, k_point=k_basis, tol=AO_IMAGE_TRUNCATION_ATOL)


def exact_box_distances(cell: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Closest approach of each fractional box ``[low, high]`` to the origin (Bohr).

    Exact scipy bounded least squares (BVLS) per box. Boxes whose centre is farther
    than the reach of the box (half its longest corner-to-corner diagonal) plus the
    largest radius of interest are skipped as ``inf`` by the caller-independent
    triangle inequality ``dist >= |centre| - half_diagonal``.
    """
    from scipy.optimize import lsq_linear

    side = high[0] - low[0]
    corners = np.array([[i, j, k] for i in (-0.5, 0.5) for j in (-0.5, 0.5) for k in (-0.5, 0.5)]) * side
    half_diagonal = float(np.max(np.linalg.norm(corners @ cell, axis=1)))
    centres = np.linalg.norm((0.5 * (low + high)) @ cell, axis=1)
    lower_bound = centres - half_diagonal
    distances = np.full(len(low), np.inf)
    for i in np.flatnonzero(lower_bound <= NON_REDUCED_AO_MAX_RADIUS):
        solution = lsq_linear(cell.T, np.zeros(3), bounds=(low[i], high[i]), method="bvls").x
        distances[i] = float(np.linalg.norm(cell.T @ solution))
    return distances


def test_periodic_pgto_values_in_non_reduced_basis_match_bruteforce_image_sum():
    """Periodic AO values and image sets on a non-reduced cell basis are complete.

    The lattice diag(6, 7, 8) Bohr is written in the basis ``U @ diag(6, 7, 8)``
    (``det U = 1``, strongly ill-conditioned Gram matrix). One hydrogen with the
    s(1.3, 0.35)/p/d/f test shells.

    * Selector: for every primitive (alpha, l) of the basis, the per-atom image set
      equals the set of candidates whose fractional box comes within the envelope
      radius by an exact scipy BVLS closest approach; the cell-cell ``_debug`` oracle
      equals the same BVLS reference for the alpha = 1 s primitive.
    * Values: the production XLA AO values of the image-expanded leaf, at Gamma (real
      leaf) and at the twist (0.25, -0.1, 0.4) of the reduced basis (complex leaf), equal
      the brute-force lattice sum of the molecular AO kernel over the reduced cell,
      within the ao_eval strict zone plus AO_IMAGE_TRUNCATION_ATOL.
    """
    from jqmc.structure import (
        _image_candidate_grid,
        _primitive_image_list_debug,
        _primitive_image_list_indices_per_atom,
        _primitive_image_radius,
    )

    cell = NON_REDUCED_AO_TRANSFORM @ NON_REDUCED_AO_LATTICE
    s_atom = NON_REDUCED_AO_ATOM[0] @ np.linalg.inv(cell)
    wrapped = s_atom - np.floor(s_atom)
    primitives = sorted({(alpha, l) for l, exponents, _ in ref.SHELLS for alpha in exponents})
    for alpha, l in primitives:
        r_cut = _primitive_image_radius(alpha, l, AO_IMAGE_TRUNCATION_ATOL)
        assert r_cut <= NON_REDUCED_AO_MAX_RADIUS
        candidates = _image_candidate_grid(cell, r_cut)
        distances = exact_box_distances(cell, -candidates - wrapped, 1.0 - candidates - wrapped)
        reference = sorted(tuple(int(x) for x in n - np.floor(s_atom)) for n in candidates[distances <= r_cut])
        actual = sorted(map(tuple, _primitive_image_list_indices_per_atom(cell, alpha, l, s_atom).tolist()))
        assert actual == reference, f"per-atom image set differs from BVLS for alpha={alpha}, l={l}"

    r_cut = _primitive_image_radius(1.0, 0, AO_IMAGE_TRUNCATION_ATOL)
    candidates = _image_candidate_grid(cell, r_cut)
    distances = exact_box_distances(cell, candidates - 1.0, candidates + 1.0)
    reference_cell_cell = sorted(tuple(int(x) for x in n) for n in candidates[distances <= r_cut])
    debug = _primitive_image_list_debug(cell, 1.0, 0)
    debug_indices = sorted(tuple(int(x) for x in row) for row in np.rint(debug @ np.linalg.inv(cell)))
    assert debug_indices == reference_cell_cell, "cell-cell _debug image set differs from BVLS"

    r = non_reduced_ao_electrons()
    ao_atol, ao_rtol = get_tolerance("ao_eval", "strict")
    atol = AO_IMAGE_TRUNCATION_ATOL + ao_atol
    for k in (None, _TWIST):
        prod = np.asarray(compute_AOs(aos_data=non_reduced_ao_leaf(k), r_carts=jnp.asarray(r)))
        brute, n_images = ref.bruteforce_periodic_aos(NON_REDUCED_AO_LATTICE, NON_REDUCED_AO_ATOM, "cart", r, k)
        _assert_finite(prod, brute)
        assert n_images > 27, f"brute-force sum is not a lattice sum ({n_images} images)"
        np.testing.assert_allclose(prod, brute, atol=atol, rtol=ao_rtol, err_msg=f"k={k}")
    jax.clear_caches()


# --------------------------------------------------------------------------- #
# (g) issue #66: floating-point domain of the closest-approach selector
# --------------------------------------------------------------------------- #
#: Extreme but valid (finite, non-singular) cell bases, rows in Bohr. ``shear_1e8`` is the
#: det-1 basis whose float64 Gram matrix is singular (``1 + 1e16`` rounds to ``1e16``).
EXTREME_IMAGE_BASES = {
    "shear_1e8": np.array([[1.0, 1e8, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "shear_1e8_triclinic": np.array([[1.0, 1e8, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    @ np.array([[5.1, 0.0, 0.0], [1.3, 4.7, 0.0], [-0.8, 1.1, 6.2]]),
    "double_shear_1e6": np.array([[1.0, 0.0, 0.0], [1e6, 1.0, 0.0], [0.0, -1e6, 1.0]]) @ np.diag([6.0, 7.0, 8.0]),
    "anisotropic_1e6": np.diag([1e-3, 1.0, 1e3]),
    "non_reduced_ao_transform": NON_REDUCED_AO_TRANSFORM @ NON_REDUCED_AO_LATTICE,
    "unimodular_2106_triclinic": np.array([[274.0, 0.0, -7.0], [-2106.0, 1.0, 35.0], [-39.0, 0.0, 1.0]])
    @ np.array([[5.1, 0.0, 0.0], [1.3, 4.7, 0.0], [-0.8, 1.1, 6.2]]),
    "unimodular_13634_triclinic": np.array([[1.0, -9.0, 51.0], [18.0, -161.0, 909.0], [-270.0, 2415.0, -13634.0]])
    @ np.array([[5.1, 0.0, 0.0], [1.3, 4.7, 0.0], [-0.8, 1.1, 6.2]]),
}
#: Fractional atom offset of the per-atom boxes ``[-n - s, 1 - n - s]``.
EXTREME_IMAGE_ATOM_OFFSET = np.array([0.3, 0.7, 0.1])
#: Half-width of the integer candidate cube passed to the selector.
EXTREME_IMAGE_CANDIDATE_HALF_WIDTH = 3


def exact_rational_box_distance_sq(cell: np.ndarray, low: np.ndarray, high: np.ndarray):
    """Exact squared distance from the origin to ``{v @ cell : low <= v <= high}`` (a Fraction).

    Brute-force Cartesian geometry in exact rational arithmetic on the float64 inputs (no
    Gram solve, no active-set enumeration, no rounding): zero if the origin is inside,
    else the minimum over the six parallelogram faces of the perpendicular distance when
    its foot lies on the face, or the clamped point-to-segment distance to the four edges.
    """
    from fractions import Fraction

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def axpy(s, a, b):
        return [s * x + y for x, y in zip(a, b)]

    def segment(start, edge):
        t = min(Fraction(1), max(Fraction(0), -dot(start, edge) / dot(edge, edge)))
        point = axpy(t, edge, start)
        return dot(point, point)

    def parallelogram(corner, e1, e2):
        g11, g12, g22 = dot(e1, e1), dot(e1, e2), dot(e2, e2)
        r1, r2 = -dot(corner, e1), -dot(corner, e2)
        det = g11 * g22 - g12 * g12
        s, t = (r1 * g22 - g12 * r2) / det, (g11 * r2 - g12 * r1) / det
        if 0 <= s <= 1 and 0 <= t <= 1:
            point = axpy(s, e1, axpy(t, e2, corner))
            return dot(point, point)
        return min(segment(corner, e1), segment(corner, e2), segment(axpy(1, e1, corner), e2), segment(axpy(1, e2, corner), e1))

    rows = [[Fraction(float(x)) for x in row] for row in cell]
    lo = [Fraction(float(x)) for x in low]
    hi = [Fraction(float(x)) for x in high]
    if all(lo[k] <= 0 <= hi[k] for k in range(3)):
        return Fraction(0)
    best = None
    for axis in range(3):
        j, k = (axis + 1) % 3, (axis + 2) % 3
        e1 = [(hi[j] - lo[j]) * x for x in rows[j]]
        e2 = [(hi[k] - lo[k]) * x for x in rows[k]]
        for side in (lo[axis], hi[axis]):
            corner = [side * a + lo[j] * b + lo[k] * c for a, b, c in zip(rows[axis], rows[j], rows[k])]
            d = parallelogram(corner, e1, e2)
            best = d if best is None or d < best else best
    return best


def test_image_selection_on_extreme_bases_matches_exact_rational_closest_approach():
    """The closest-approach selector is exact on extreme but valid cell bases.

    For every basis of ``EXTREME_IMAGE_BASES`` (including the det-1 shear whose float64
    Gram matrix is singular) and for both box kinds (cell-cell ``[n - 1, n + 1]`` and
    per-atom ``[-n - s, 1 - n - s]``), ``_images_within_rcut_from_box`` on the candidate
    cube ``[-3, 3]^3`` must keep exactly the candidates whose exact rational
    closest-approach distance is within ``r_cut``. ``r_cut`` lies in the widest relative gap
    between consecutive distinct exact distances (nearer half), so the expected set is
    unambiguous.
    A numerically singular or non-finite cell must raise ``ValueError`` rather than
    return an image set.
    """
    from fractions import Fraction

    from jqmc.structure import _images_within_rcut_from_box

    half = EXTREME_IMAGE_CANDIDATE_HALF_WIDTH
    axis = np.arange(-half, half + 1, dtype=np.float64)
    n = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    boxes = {
        "cell-cell": (n - 1.0, n + 1.0),
        "per-atom": (-n - EXTREME_IMAGE_ATOM_OFFSET, 1.0 - n - EXTREME_IMAGE_ATOM_OFFSET),
    }
    for name, cell in EXTREME_IMAGE_BASES.items():
        for kind, (low, high) in boxes.items():
            d_sq = [exact_rational_box_distance_sq(cell, low[i], high[i]) for i in range(len(n))]
            levels = sorted(set(d_sq))
            # The widest relative gap among the nearer half of the distinct non-zero distances.
            k = max(range(1, len(levels) // 2), key=lambda i: levels[i + 1] / levels[i])
            r_cut = float((float(levels[k]) * float(levels[k + 1])) ** 0.25)
            r_cut_sq = Fraction(r_cut) ** 2
            assert levels[k] < r_cut_sq < levels[k + 1], f"{name}/{kind}: r_cut is not inside the gap"
            expected = sorted(tuple(int(x) for x in n[i]) for i in range(len(n)) if d_sq[i] <= r_cut_sq)
            assert 0 < len(expected) < len(n)
            got = sorted(tuple(int(x) for x in row) for row in _images_within_rcut_from_box(cell, r_cut, n, low, high))
            assert got == expected, (
                f"{name}/{kind}: {len(set(expected) - set(got))} missing, {len(set(got) - set(expected))} extra"
            )

    low, high = boxes["cell-cell"]
    for bad_cell in (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]), np.full((3, 3), np.nan)):
        with pytest.raises(ValueError, match="non-singular cell basis"):
            _images_within_rcut_from_box(bad_cell, 1.0, n, low, high)


#: Extreme-SCALE but valid cell bases (rows in Bohr): every one is finite with
#: ``volume / (|a| |b| |c|)`` of order one, yet some row square, row-pair Gram entry or
#: distance square under- or overflows float64. ``tiny_diag_audit`` is the issue #66
#: counterexample ``diag(1e-200, 1e100, 1e100)`` (volume 1, row-norm product 1).
_TRICLINIC_ROWS = np.array([[5.1, 0.0, 0.0], [1.3, 4.7, 0.0], [-0.8, 1.1, 6.2]])
EXTREME_SCALE_IMAGE_BASES = {
    "tiny_diag_audit": np.diag([1e-200, 1e100, 1e100]),
    "tiny_triclinic": 1e-170 * _TRICLINIC_ROWS,
    "huge_triclinic": 1e170 * _TRICLINIC_ROWS,
    "tiny_shear_1e8_triclinic": 1e-160 * EXTREME_IMAGE_BASES["shear_1e8_triclinic"],
    "huge_shear_1e8_triclinic": 1e150 * EXTREME_IMAGE_BASES["shear_1e8_triclinic"],
    "mixed_scale_triclinic": np.diag([1e-170, 1.0, 1e170]) @ _TRICLINIC_ROWS,
    "mixed_scale_skew": np.array([[1e-200, 0.0, 0.0], [1e-194, 1e-200, 0.0], [0.3e150, -0.7e150, 1e150]]),
    "range_edge_triclinic": np.diag([1e300, 1e-300, 1.0]) @ _TRICLINIC_ROWS,
}


def exact_rcut_inside_gap(lower_sq, upper_sq) -> float:
    """A float64 ``r_cut`` with ``lower_sq < r_cut**2 < upper_sq`` (exact Fractions), at any scale.

    The geometric mean of the two distances is taken in log space from the exact rationals
    (big-integer ``math.log``), so no float64 square of a tiny or huge distance is formed.
    """
    import math

    def log(q):
        return math.log(q.numerator) - math.log(q.denominator)

    return math.exp((log(lower_sq) + log(upper_sq)) / 4.0)


def test_image_selection_on_extreme_scale_bases_matches_exact_rational_closest_approach():
    """The closest-approach selector is scale safe on every finite cell of its domain.

    The counterexample ``diag(1e-200, 1e100, 1e100)`` (candidate ``(0, 0, 0)``, fractional
    box ``[1, 2]^3``) lies in the declared domain ``volume > eps |a| |b| |c|`` and must not
    raise ``LinAlgError`` although a squared tiny row underflows to zero. Its exact closest
    point is the lower corner at distance ``sqrt(2e200 + 1e-400)``: ``r_cut = 1e99`` keeps
    nothing and ``r_cut = 2e100`` keeps the candidate. For every basis of
    ``EXTREME_SCALE_IMAGE_BASES`` (tiny, huge and mixed scales, sheared and triclinic) and both
    box kinds over the candidate cube ``[-3, 3]^3``, ``_images_within_rcut_from_box`` must
    keep exactly the candidates whose exact rational closest-approach distance
    (:py:func:`exact_rational_box_distance_sq`) is within ``r_cut``, with ``r_cut`` inside the
    widest relative gap of the exact distances (nearer half), placed in log space.
    """
    from fractions import Fraction

    from jqmc.structure import _images_within_rcut_from_box

    cell = EXTREME_SCALE_IMAGE_BASES["tiny_diag_audit"]
    one, low, high = np.zeros((1, 3)), np.ones((1, 3)), np.full((1, 3), 2.0)
    exact = exact_rational_box_distance_sq(cell, low[0], high[0])
    assert exact == 2 * Fraction(1e100) ** 2 + Fraction(1e-200) ** 2
    for r_cut, expected in ((1e99, []), (2e100, [(0, 0, 0)])):
        assert (exact <= Fraction(r_cut) ** 2) == bool(expected)
        got = [tuple(int(x) for x in row) for row in _images_within_rcut_from_box(cell, r_cut, one, low, high)]
        assert got == expected, f"diag(1e-200, 1e100, 1e100) counterexample, r_cut={r_cut!r}: got {got}"

    half = EXTREME_IMAGE_CANDIDATE_HALF_WIDTH
    axis = np.arange(-half, half + 1, dtype=np.float64)
    n = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    boxes = {
        "cell-cell": (n - 1.0, n + 1.0),
        "per-atom": (-n - EXTREME_IMAGE_ATOM_OFFSET, 1.0 - n - EXTREME_IMAGE_ATOM_OFFSET),
    }
    for name, cell in EXTREME_SCALE_IMAGE_BASES.items():
        assert np.all(np.isfinite(cell))
        for kind, (low, high) in boxes.items():
            d_sq = [exact_rational_box_distance_sq(cell, low[i], high[i]) for i in range(len(n))]
            levels = sorted(set(d_sq))
            k = max(range(1, len(levels) // 2), key=lambda i: levels[i + 1] / levels[i])
            r_cut = exact_rcut_inside_gap(levels[k], levels[k + 1])
            r_cut_sq = Fraction(r_cut) ** 2
            assert levels[k] < r_cut_sq < levels[k + 1], f"{name}/{kind}: r_cut is not inside the gap"
            expected = sorted(tuple(int(x) for x in n[i]) for i in range(len(n)) if d_sq[i] <= r_cut_sq)
            assert 0 < len(expected) < len(n)
            got = sorted(tuple(int(x) for x in row) for row in _images_within_rcut_from_box(cell, r_cut, n, low, high))
            assert got == expected, (
                f"{name}/{kind}: {len(set(expected) - set(got))} missing, {len(set(got) - set(expected))} extra"
            )
