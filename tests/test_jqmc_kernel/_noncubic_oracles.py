"""Independent references for periodic kernels on non-orthogonal cells.

Shared by ``test_noncubic_cell_oracles.py`` and the absolute-value anchors in
``test_periodic_invariants.py`` (JAX) and its Rust mirror. Nothing here reuses a
production image list, lattice-sum cutoff, or Ewald table:

* :func:`bruteforce_periodic_aos` sums the molecular (open-boundary) AO kernel
  over every lattice image inside a generous Gaussian cutoff that the helper
  derives itself from the smallest exponent.
* :func:`numpy_cart_aos` is a plain-numpy normalized Cartesian GTO evaluator used
  to pin the molecular kernel that the brute-force sum is built from.
* :func:`free_electron_kinetic` evaluates the local kinetic energy of a
  single-plane-wave Slater determinant from plane waves it builds itself.

Precision: a reference that judges a mixed-precision quantity is evaluated
in the precision regime of the production zones that compute it, never in fp64
throughout. Geometry differences are formed in fp64 and cast to the zone dtype;
each piece then runs in the dtype of its zone (``jqmc._precision`` accessors).
In ``full`` mode every zone is fp64 and the references are fp64.
* :func:`bruteforce_periodic_cart_overlap` sums whole-space primitive overlaps
  (Gauss-Hermite quadrature per axis, exact for the polynomial degrees used) over
  every lattice image inside its own cutoff.
* :func:`numpy_ewald_energy` is a textbook Ewald sum for neutral point charges.
* :func:`transformed_trexio_copy` rewrites a committed periodic TREXIO file into
  an equivalent cell basis ``U @ cell`` (``U`` integer, ``det U = +-1``) or onto a
  different lattice, so the production loader sees a non-orthogonal cell.
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

import itertools
import math
import os
import shutil

import numpy as np

import jax.numpy as jnp

from jqmc._precision import get_complex_dtype_np, get_dtype_np
from jqmc.atomic_orbital import AOs_cart_gto_data, AOs_sphe_gto_data, compute_AOs, compute_AOs_laplacian
from jqmc.structure import Structure_data

#: Non-orthogonal cells. The triclinic and FCC cells are the shared ``CELL_FIXTURES``
#: rows; the hexagonal, monoclinic and skewed-basis cells are local.
HEXAGONAL_CELL = np.array([[6.0, 0.0, 0.0], [-3.0, 3.0 * np.sqrt(3.0), 0.0], [0.0, 0.0, 7.0]], dtype=np.float64)
MONOCLINIC_CELL = np.array([[6.5, 0.0, 0.0], [0.0, 7.0, 0.0], [-2.6, 0.0, 6.2]], dtype=np.float64)
#: A 6 Bohr simple-cubic lattice written in the strongly skewed basis
#: b1 = a1, b2 = a1 + a2, b3 = a1 + a2 + a3 (a cell.T slip changes its image set most).
SKEWED_BASIS_CUBIC6_CELL = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]) @ np.diag([6.0, 6.0, 6.0])

#: Fractional positions of the two basis centres, strictly inside the home cell.
ATOM_FRACTIONS = np.array([[0.1, 0.2, 0.15], [0.7, 0.55, 0.8]], dtype=np.float64)

#: Nucleus placements off the home cell interior: just outside a lower
#: and an upper face, exactly on a corner and an upper face, and more than one
#: cell away (a separation beyond the cell-cell image box).
PLACEMENT_FRACTIONS = {
    "outside": np.array([[-0.1, 0.2, 0.15], [0.7, 1.1, 0.8]], dtype=np.float64),
    "boundary": np.array([[0.0, 0.0, 0.0], [0.7, 0.55, 1.0]], dtype=np.float64),
    "far": np.array([[-1.1, 0.2, 2.15], [0.7, 0.55, 0.8]], dtype=np.float64),
}

#: (l, exponents, coefficients). One diffuse s exponent (0.35) makes the lattice
#: sum reach several cells on the small skewed cells.
SHELLS = ((0, (1.3, 0.35), (0.7, 0.4)), (1, (0.45,), (1.0,)), (2, (0.8,), (1.0,)), (3, (1.1,), (1.0,)))

#: Gaussian tail target of the brute-force sums: every omitted primitive term is
#: below exp(-BRUTE_FORCE_LOG_TAIL) times its polynomial prefactor.
BRUTE_FORCE_LOG_TAIL = 80.0


def _double_factorial(n: int) -> int:
    return 1 if n <= 0 else n * _double_factorial(n - 2)


def build_structure(cell: np.ndarray | None, positions: np.ndarray) -> Structure_data:
    """Hydrogen centres at ``positions``; periodic when ``cell`` is given."""
    positions = np.asarray(positions, dtype=np.float64)
    kwargs = dict(
        pbc_flag=cell is not None,
        positions=positions,
        atomic_numbers=tuple([1] * len(positions)),
        element_symbols=tuple(["H"] * len(positions)),
        atomic_labels=tuple(f"H{i}" for i in range(len(positions))),
    )
    if cell is not None:
        kwargs.update(vec_a=tuple(cell[0]), vec_b=tuple(cell[1]), vec_c=tuple(cell[2]))
    return Structure_data.build(**kwargs)


def _shell_components(kind: str, l: int):
    if kind == "sphe":
        return [(m, None) for m in range(-l, l + 1)]
    return [(None, "".join(p)) for p in itertools.combinations_with_replacement("xyz", l)]


def build_aos(structure: Structure_data, kind: str) -> AOs_cart_gto_data | AOs_sphe_gto_data:
    """Molecular-rank GTO leaf with :data:`SHELLS` on every atom of ``structure``."""
    exps, coefs, orbital, ang, mag, nx, ny, nz, nucleus = [], [], [], [], [], [], [], [], []
    ao = 0
    for atom in range(len(structure.positions)):
        for l, shell_exps, shell_coefs in SHELLS:
            for m, poly in _shell_components(kind, l):
                for e, c in zip(shell_exps, shell_coefs, strict=True):
                    exps.append(e)
                    coefs.append(c)
                    orbital.append(ao)
                ang.append(l)
                nucleus.append(atom)
                if kind == "sphe":
                    mag.append(m)
                else:
                    nx.append(poly.count("x"))
                    ny.append(poly.count("y"))
                    nz.append(poly.count("z"))
                ao += 1
    common = dict(
        structure_data=structure,
        nucleus_index=tuple(nucleus),
        num_ao=ao,
        num_ao_prim=len(exps),
        orbital_indices=tuple(orbital),
        exponents=np.asarray(exps, dtype=np.float64),
        coefficients=np.asarray(coefs, dtype=np.float64),
        angular_momentums=tuple(ang),
    )
    if kind == "sphe":
        return AOs_sphe_gto_data._from_full_basis(**common, magnetic_quantum_numbers=tuple(mag))
    return AOs_cart_gto_data._from_full_basis(
        **common, polynominal_order_x=tuple(nx), polynominal_order_y=tuple(ny), polynominal_order_z=tuple(nz)
    )


def _cart_primitives(positions: np.ndarray):
    """(centre, alpha, coefficient, (nx, ny, nz), ao index) for the Cartesian basis."""
    out = []
    ao = 0
    for centre in np.asarray(positions, dtype=np.float64):
        for l, shell_exps, shell_coefs in SHELLS:
            for _, poly in _shell_components("cart", l):
                powers = (poly.count("x"), poly.count("y"), poly.count("z"))
                for e, c in zip(shell_exps, shell_coefs, strict=True):
                    out.append((centre, e, c, powers, ao))
                ao += 1
    return out, ao


def _cart_norm(alpha: float, powers: tuple[int, int, int]) -> float:
    l = sum(powers)
    denom = np.prod([_double_factorial(2 * n - 1) for n in powers])
    return math.sqrt((2.0 * alpha / math.pi) ** 1.5 * (4.0 * alpha) ** l / denom)


def numpy_cart_aos(positions: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Normalized contracted Cartesian GTO values, shape ``(num_ao, len(r))``, in the ``ao_eval`` dtype.

    Precision: ``r - R`` is formed in fp64 and cast to the ``ao_eval`` dtype; the
    polynomial, the Gaussian, the exponent and normalized coefficient (fp64 basis
    data cast to that dtype) and the contraction sum all run in it (float32 in
    mixed, float64 in full), as the production ``ao_eval`` zone does.
    """
    dtype = get_dtype_np("ao_eval")
    prims, num_ao = _cart_primitives(positions)
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros((num_ao, len(r)), dtype=dtype)
    for centre, alpha, coef, powers, ao in prims:
        d = (r - centre).astype(dtype)
        poly = d[:, 0] ** powers[0] * d[:, 1] ** powers[1] * d[:, 2] ** powers[2]
        prefactor = dtype(coef * _cart_norm(alpha, powers))
        out[ao] += prefactor * poly * np.exp(-dtype(alpha) * np.sum(d * d, axis=1))
    return out


def _inradius(cell: np.ndarray) -> float:
    volume = abs(float(np.linalg.det(cell)))
    areas = [np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)]
    return volume / max(areas)


def _gaussian_cutoff(alpha: float, l_max: int) -> float:
    """Radius beyond which r^l_max exp(-alpha r^2) < exp(-BRUTE_FORCE_LOG_TAIL) (r >= 1)."""
    r = math.sqrt(BRUTE_FORCE_LOG_TAIL / alpha)
    for _ in range(20):
        r = math.sqrt((BRUTE_FORCE_LOG_TAIL + l_max * math.log(max(r, 1.0))) / alpha)
    return r


def lattice_images_within(cell: np.ndarray, displacements: np.ndarray, radius: float) -> np.ndarray:
    """Integer triples n with ``min_d |d - n @ cell| <= radius`` over the given displacements.

    Enumerates a cube whose half-width follows from the inradius, and asserts that
    no kept image touches the cube boundary (the cube is large enough).
    """
    cell = np.asarray(cell, dtype=np.float64)
    reach = radius + float(np.max(np.linalg.norm(displacements, axis=1)))
    half = int(math.ceil(reach / _inradius(cell))) + 1
    grid = np.arange(-half, half + 1)
    n = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    t = n.astype(np.float64) @ cell
    dist = np.min(np.linalg.norm(displacements[None, :, :] - t[:, None, :], axis=-1), axis=1)
    kept = n[dist <= radius]
    assert np.max(np.abs(kept)) < half, "brute-force image cube is too small"
    return kept


def bruteforce_periodic_aos(
    cell: np.ndarray,
    positions: np.ndarray,
    kind: str,
    r: np.ndarray,
    k_frac: tuple[float, float, float] | None,
    quantity: str = "value",
) -> tuple[np.ndarray, int]:
    """sum_n exp(2 pi i k.n) chi_mu(r - R_mu - n @ cell) from the molecular AO kernel.

    ``quantity="laplacian"`` sums the molecular AO Laplacian instead. Returns the
    values ``(num_ao, len(r))`` and the number of images summed.

    Precision: the image points ``r - n @ cell`` are formed in fp64 and the molecular
    kernel casts ``r - R`` to its zone (``ao_eval`` for values, ``ao_grad_lap``
    for the Laplacian). The Bloch phase (fp64 from the integer images, then cast)
    and the image sum run in that same zone's dtype (float32 / complex64 for mixed
    values; fp64 for the Laplacian and in full), as the production image fold does.
    """
    kernel = {"value": compute_AOs, "laplacian": compute_AOs_laplacian}[quantity]
    molecular = build_aos(build_structure(None, positions), kind)
    alpha_min = min(min(e) for _, e, _ in SHELLS)
    l_max = max(l for l, _, _ in SHELLS)
    radius = _gaussian_cutoff(alpha_min, l_max)
    displacements = (np.asarray(r)[:, None, :] - np.asarray(positions)[None, :, :]).reshape(-1, 3)
    images = lattice_images_within(cell, displacements, radius)
    shifts = images.astype(np.float64) @ cell
    points = (np.asarray(r)[None, :, :] - shifts[:, None, :]).reshape(-1, 3)
    values = np.asarray(kernel(aos_data=molecular, r_carts=jnp.asarray(points)))
    values = values.reshape(values.shape[0], len(images), len(r))
    zone = "ao_eval" if quantity == "value" else "ao_grad_lap"
    if k_frac is None:
        dtype = get_dtype_np(zone)
        phase = np.ones(len(images), dtype=dtype)
    else:
        dtype = get_complex_dtype_np(zone)
        phase = np.exp(2j * np.pi * (images @ np.asarray(k_frac, dtype=np.float64))).astype(dtype)
    return np.einsum("n,anr->ar", phase, values.astype(dtype)), len(images)


_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(12)


def _overlap_1d(alpha: float, a: float, i: int, beta: float, b: np.ndarray, j: int) -> np.ndarray:
    """int (x-a)^i (x-b)^j exp(-alpha (x-a)^2 - beta (x-b)^2) dx by Gauss-Hermite (exact here)."""
    p = alpha + beta
    centre = (alpha * a + beta * b) / p
    prefactor = np.exp(-alpha * beta / p * (a - b) ** 2) / math.sqrt(p)
    x = centre[..., None] + _GH_NODES / math.sqrt(p)
    integrand = (x - a) ** i * (x - b[..., None]) ** j
    return prefactor * np.sum(_GH_WEIGHTS * integrand, axis=-1)


def bruteforce_periodic_cart_overlap(
    cell: np.ndarray, positions: np.ndarray, k_frac: tuple[float, float, float] | None
) -> np.ndarray:
    """S_mu,nu = sum_n exp(2 pi i k.n) <chi_mu(r - R_mu) | chi_nu(r - R_nu - n @ cell)>."""
    prims, num_ao = _cart_primitives(positions)
    positions = np.asarray(positions, dtype=np.float64)
    alpha_min = min(min(e) for _, e, _ in SHELLS)
    l_max = max(l for l, _, _ in SHELLS)
    # Product decay rho = alpha beta / (alpha + beta) >= alpha_min / 2.
    radius = _gaussian_cutoff(alpha_min / 2.0, 2 * l_max)
    displacements = (positions[:, None, :] - positions[None, :, :]).reshape(-1, 3)
    images = lattice_images_within(cell, displacements, radius)
    shifts = images.astype(np.float64) @ cell
    phase = np.ones(len(images)) if k_frac is None else np.exp(2j * np.pi * (images @ np.asarray(k_frac)))
    s = np.zeros((num_ao, num_ao), dtype=np.complex128)
    for ca, alpha, cc_a, pa, mu in prims:
        na = cc_a * _cart_norm(alpha, pa)
        for cb, beta, cc_b, pb, nu in prims:
            nb = cc_b * _cart_norm(beta, pb)
            centres = cb[None, :] + shifts
            term = np.ones(len(images))
            for axis in range(3):
                term = term * _overlap_1d(alpha, ca[axis], pa[axis], beta, centres[:, axis], pb[axis])
            s[mu, nu] += na * nb * np.sum(phase * term)
    return s if k_frac is not None else s.real


def numpy_ewald_energy(cell: np.ndarray, positions: np.ndarray, charges: np.ndarray) -> float:
    """Textbook Ewald energy of a NEUTRAL set of point charges (self images included).

    Precision: evaluated in the ``ewald`` zone dtype (float64 in both modes); the
    displacements ``x_i - x_j + T`` are formed in fp64 and cast to that dtype.
    """
    dtype = get_dtype_np("ewald")
    cell = np.asarray(cell, dtype=np.float64)
    x = np.asarray(positions, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    assert abs(float(np.sum(q))) == 0.0, "numpy_ewald_energy needs a neutral cell"
    from scipy.special import erfc

    volume = abs(float(np.linalg.det(cell)))
    eta = math.sqrt(math.pi) / volume ** (1.0 / 3.0)
    r_cut = 7.0 / eta  # erfc(7) ~ 4e-23
    g_cut = 2.0 * eta * 7.0  # exp(-49) ~ 5e-22
    d = (x[:, None, :] - x[None, :, :]).reshape(-1, 3)
    qq = (q[:, None] * q[None, :]).reshape(-1).astype(dtype)
    images = lattice_images_within(cell, d, r_cut).astype(np.float64) @ cell
    real = 0.0
    for t in images:
        v = np.linalg.norm((d + t[None, :]).astype(dtype), axis=1)
        mask = v > 0.0
        real += 0.5 * float(np.sum(qq[mask] * erfc(dtype(eta) * v[mask]) / v[mask]))
    recip = 2.0 * np.pi * np.linalg.inv(cell).T
    gidx = lattice_images_within(recip, np.zeros((1, 3)), g_cut)
    gidx = gidx[np.any(gidx != 0, axis=1)]
    g = (gidx.astype(np.float64) @ recip).astype(dtype)
    g2 = np.sum(g * g, axis=1)
    structure_factor = np.exp(1j * (g @ x.astype(dtype).T)) @ q.astype(dtype)
    recip_sum = (2.0 * np.pi / volume) * float(np.sum(np.exp(-g2 / (4.0 * eta**2)) / g2 * np.abs(structure_factor) ** 2))
    self_term = -eta / math.sqrt(math.pi) * float(np.sum(q * q))
    return real + recip_sum + self_term


def transformed_trexio_copy(
    src: str,
    dst_dir: str,
    tag: str,
    unimodular: np.ndarray,
    *,
    lattice: np.ndarray | None = None,
    nucleus_shift_frac: np.ndarray | None = None,
) -> str:
    """Copy a periodic TREXIO file, re-expressing it in the cell basis ``U @ lattice``.

    ``lattice`` (default: the file's own cell) is the physical lattice. The
    fractional k point is mapped ``k -> U k`` so ``exp(2 pi i k.n)`` is unchanged
    for every physical translation. Nuclei are shifted rigidly by
    ``nucleus_shift_frac @ lattice`` (identical Cartesian coordinates in every basis).
    """
    import h5py

    unimodular = np.asarray(unimodular, dtype=np.float64)
    assert abs(abs(round(float(np.linalg.det(unimodular)))) - 1) == 0, "basis change must be unimodular"
    dst = os.path.join(dst_dir, f"{tag}.h5")
    shutil.copy(src, dst)
    with h5py.File(dst, "r+") as f:
        own = np.stack([f["cell/cell_a"][()], f["cell/cell_b"][()], f["cell/cell_c"][()]])
        base = own if lattice is None else np.asarray(lattice, dtype=np.float64)
        new = unimodular @ base
        for i, name in enumerate(("cell_a", "cell_b", "cell_c")):
            f[f"cell/{name}"][...] = new[i]
        k = np.asarray(f["pbc/pbc_k_point"][()], dtype=np.float64)
        f["pbc/pbc_k_point"][...] = (unimodular @ k.reshape(-1, 3).T).T.reshape(k.shape)
        coords = np.asarray(f["nucleus/nucleus_coord"][()], dtype=np.float64)
        if lattice is not None:
            # Carry the atoms onto the new lattice at the same fractional positions.
            coords = (coords @ np.linalg.inv(own)) @ base
        if nucleus_shift_frac is not None:
            coords = coords + np.asarray(nucleus_shift_frac, dtype=np.float64) @ base
        f["nucleus/nucleus_coord"][...] = coords
    return dst


def fractional(cell: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.asarray(r, dtype=np.float64) @ np.linalg.inv(np.asarray(cell, dtype=np.float64))


def choose_nucleus_shift(src: str, unimodular: np.ndarray, lattice: np.ndarray | None, margin: float) -> np.ndarray:
    """First fractional shift (grid of 1/20) that puts every nucleus inside both home cells.

    "Inside" means fractional coordinates in ``[margin, 1 - margin]`` for the basis
    ``lattice`` and for ``U @ lattice``. Deterministic; raises if no shift exists.
    """
    import h5py

    with h5py.File(src, "r") as f:
        own = np.stack([f["cell/cell_a"][()], f["cell/cell_b"][()], f["cell/cell_c"][()]])
        coords = np.asarray(f["nucleus/nucleus_coord"][()], dtype=np.float64)
    base = own if lattice is None else np.asarray(lattice, dtype=np.float64)
    frac0 = coords @ np.linalg.inv(own)
    other = np.asarray(unimodular, dtype=np.float64) @ base
    steps = np.arange(20) / 20.0
    for shift in itertools.product(steps, steps, steps):
        cart = (frac0 + np.asarray(shift)) @ base
        ok = True
        for c in (base, other):
            s = fractional(c, cart)
            ok &= bool(np.all(s >= margin) and np.all(s <= 1.0 - margin))
        if ok:
            return np.asarray(shift)
    raise AssertionError("no rigid nucleus shift puts every nucleus inside both home cells")


def build_free_electron_pw_hamiltonian(
    cell: np.ndarray,
    nuclei_frac: np.ndarray,
    k_frac: tuple[float, float, float],
    millers_up: np.ndarray,
    millers_dn: np.ndarray,
):
    """All-electron H lattice with a Jastrow-free single-plane-wave Slater determinant.

    Up orbital j is exp(i (G_j + k) . r) on a +k leaf, down orbital j is
    exp(i (G_j - k) . r) on a -k leaf (identity MO coefficients, no time-reversal
    tie). For such a determinant sum_i -1/2 lap_i Psi / Psi equals
    sum_occ |G + k|^2 / 2 at EVERY configuration in exact arithmetic;
    :func:`free_electron_kinetic` evaluates it per precision zone.
    """
    from jqmc.atomic_orbital import AOs_plane_wave_complex_data
    from jqmc.coulomb_potential import Coulomb_potential_data, build_ewald_data
    from jqmc.determinant import SD_data
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import Jastrow_data
    from jqmc.molecular_orbital import MOs_data
    from jqmc.wavefunction import Wavefunction_data

    cell = np.asarray(cell, dtype=np.float64)
    positions = np.asarray(nuclei_frac, dtype=np.float64) @ cell
    structure = build_structure(cell, positions)
    n_up, n_dn = len(millers_up), len(millers_dn)
    assert n_up + n_dn == len(positions), "the H lattice must be neutral"
    coulomb = Coulomb_potential_data(structure_data=structure)
    coulomb = coulomb.replace(ewald_data=build_ewald_data(structure, coulomb, n_electron=n_up + n_dn))
    recip = 2.0 * np.pi * np.linalg.inv(cell).T
    k = tuple(float(x) for x in k_frac)
    leaf_up = AOs_plane_wave_complex_data.build(structure, np.asarray(millers_up, dtype=np.float64) @ recip, k_point=k)
    leaf_dn = AOs_plane_wave_complex_data.build(
        structure, np.asarray(millers_dn, dtype=np.float64) @ recip, k_point=tuple(-x for x in k)
    )
    mos_up = MOs_data.build(aos_data=leaf_up, mo_coefficients=np.eye(n_up, dtype=np.complex128))
    mos_dn = MOs_data.build(aos_data=leaf_dn, mo_coefficients=np.eye(n_dn, dtype=np.complex128))
    geminal = SD_data.build(num_electron_up=n_up, num_electron_dn=n_dn, mos_up=mos_up, mos_dn=mos_dn, time_reversal=False)
    wf = Wavefunction_data(
        geminal_data=geminal,
        jastrow_data=Jastrow_data(
            jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=None
        ),
    )
    return Hamiltonian_data.build(structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=wf)


def _plane_waves(q: np.ndarray, r: np.ndarray, zone: str) -> np.ndarray:
    """``A[i, j] = exp(i q_j . r_i)``: ``r`` and ``q`` cast to ``zone``, phase and exp in its dtype."""
    real = get_dtype_np(zone)
    theta = np.asarray(r, dtype=np.float64).astype(real) @ np.asarray(q, dtype=np.float64).astype(real).T
    return np.exp(get_complex_dtype_np(zone)(1j) * theta)


def _determinant_kinetic(q: np.ndarray, r: np.ndarray, fused_values: bool) -> complex:
    """-1/2 sum_i lap_i D / D of ``D = det[exp(i q_j . r_i)]``, one precision zone per piece.

    The matrix ``A`` holds plane-wave VALUES in the ``ao_eval`` complex dtype,
    carried through the identity MO contraction in ``mo_eval``. With
    ``fused_values=False`` they are evaluated in ``ao_eval`` arithmetic (the AO
    value route that builds the sweep geminal inverse); with ``fused_values=True``
    they are the ``ao_grad_lap`` evaluation stored in the ``ao_eval`` dtype (the
    fused value/gradient/Laplacian route). The gradients ``i q phi`` and
    Laplacians ``-|q|^2 phi`` are evaluated in ``ao_grad_lap``; ``A^-1``,
    ``grad ln D = A^-1 grad`` and ``lap ln D = A^-1 lap - (grad ln D)^2`` run in
    ``det_eval``; ``-1/2 (lap ln D + grad ln D . grad ln D)`` is assembled in
    ``wf_kinetic``. In mixed the float32 values are the denominator of the fp64
    derivatives, so the result departs from sum |q|^2 / 2 at the float32 level.
    """
    if fused_values:
        values = _plane_waves(q, r, "ao_grad_lap").astype(get_complex_dtype_np("ao_eval"))
    else:
        values = _plane_waves(q, r, "ao_eval")
    values = values.astype(get_complex_dtype_np("mo_eval"))
    phi = _plane_waves(q, r, "ao_grad_lap")
    qd = np.asarray(q, dtype=np.float64).astype(get_dtype_np("ao_grad_lap"))
    grad = (get_complex_dtype_np("ao_grad_lap")(1j) * qd)[None, :, :] * phi[:, :, None]  # (N_e, N_orb, 3)
    lap = -np.sum(qd * qd, axis=1)[None, :] * phi  # (N_e, N_orb)
    det_dtype = get_complex_dtype_np("det_eval")
    inv = np.linalg.inv(values.astype(det_dtype))  # inv[j, i]
    grad_ln = np.einsum("ji,ijx->ix", inv, grad.astype(det_dtype))
    lap_ln = np.einsum("ji,ij->i", inv, lap.astype(det_dtype)) - np.sum(grad_ln * grad_ln, axis=1)
    wf_dtype = get_complex_dtype_np("wf_kinetic")
    grad_ln = grad_ln.astype(wf_dtype)
    return complex(-0.5 * np.sum(lap_ln.astype(wf_dtype) + np.sum(grad_ln * grad_ln, axis=1)))


def free_electron_kinetic(
    cell: np.ndarray,
    k_frac,
    millers_up: np.ndarray,
    millers_dn: np.ndarray,
    r_up: np.ndarray,
    r_dn: np.ndarray,
    *,
    fused_values: bool = False,
) -> complex:
    """Local kinetic energy of the determinant of the builder above at ``(r_up, r_dn)``, per zone.

    The plane-wave vectors ``q = G + k`` (up) and ``G - k`` (down) are built here
    in fp64 from the cell, the Miller indices and ``k_frac`` (basis data), not
    from a production leaf; each spin determinant is evaluated by
    :func:`_determinant_kinetic`. ``fused_values`` selects the denominator route:
    False for a kinetic energy built on the sweep geminal inverse
    (``compute_local_energy_fast``), True for one whose inverse comes from the
    fused value/gradient/Laplacian tables (``compute_kinetic_energy``). In full
    mode every zone is fp64, both routes coincide and the value equals the closed
    form sum_occ |G + k|^2 / 2 up to fp64 round-off; in mixed it carries the
    float32 ``ao_eval`` values the production kinetic divides by.
    """
    recip = 2.0 * np.pi * np.linalg.inv(np.asarray(cell, dtype=np.float64)).T
    k = np.asarray(k_frac, dtype=np.float64) @ recip
    q_up = np.asarray(millers_up, dtype=np.float64) @ recip + k
    q_dn = np.asarray(millers_dn, dtype=np.float64) @ recip - k
    return _determinant_kinetic(q_up, r_up, fused_values) + _determinant_kinetic(q_dn, r_dn, fused_values)
