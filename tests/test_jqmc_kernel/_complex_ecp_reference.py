"""Independent primitive-AO, Jastrow, Ewald and explicit complex ECP references.

Promoted from the reviewed W51 Li diagnostic. Production numerical AO, Jastrow,
ECP, kinetic and wavefunction helpers are not called. Shared data are primitive
coefficients, declared image indices and MO coefficients; image enumeration and
basis truncation are outside this reference.
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

import itertools
import math
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import eval_legendre
from jqmc._precision import get_dtype_jnp, get_complex_dtype_jnp
from tests.test_jqmc_kernel._noncubic_oracles import numpy_ewald_energy


def plane_wave_value(aos, coordinates):
    """Explicit complex Bloch plane waves exp(i (G + k) . r) on the raw coordinates.

    G and k are rebuilt here from the stored integer Miller triples and the
    fractional twist with the reciprocal basis 2 pi inv(cell).T; no production
    reciprocal-vector property or AO kernel is called. No cell wrap is applied:
    the Bloch phase of an electron outside the home cell is part of the value.
    """
    dtype = get_dtype_jnp("ao_eval")
    cdtype = get_complex_dtype_jnp("ao_eval")
    cell = np.asarray(aos.structure_data.cell, dtype=np.float64)
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    miller = np.asarray(aos._g_indices, dtype=np.float64).reshape(-1, 3)
    assert miller.shape[0] == aos.num_ao
    wavevectors = (miller + np.asarray(aos.k_point, dtype=np.float64)[None, :]) @ reciprocal
    theta = jnp.asarray(coordinates, dtype=dtype) @ jnp.asarray(wavevectors.T, dtype=dtype)
    return jnp.exp(jnp.asarray(1j, cdtype) * theta.astype(cdtype)).T


def orbital_value(aos, coordinates):
    """Explicit AO values; no production numerical AO helper.

    Composite leaves concatenate their children in stored order, complex plane
    waves use :func:`plane_wave_value`, and Bloch pGTO leaves use the explicit
    primitive Gaussian/image sum below.
    """
    if hasattr(aos, "children"):
        return jnp.concatenate([orbital_value(child, coordinates) for child in aos.children], axis=0)
    if type(aos).__name__ == "AOs_plane_wave_complex_data":
        return plane_wave_value(aos, coordinates)
    return _bloch_pgto_value(aos, coordinates)


def _bloch_pgto_value(aos, coordinates):
    """Bloch pGTO values from the explicit primitive Gaussian/image sum (every call checks its shells)."""
    dtype = get_dtype_jnp("ao_eval")
    cdtype = get_complex_dtype_jnp("ao_eval")
    cell = np.asarray(aos.structure_data.cell)
    inv = np.linalg.inv(cell)
    frac = coordinates @ jnp.asarray(inv)
    hop = jnp.floor(frac)
    wrapped = (frac - hop) @ jnp.asarray(cell)
    home = np.asarray(aos._ragged_kernel_to_home, int)
    images = np.asarray(aos._image_indices, int)
    nuclei = np.asarray(aos.structure_data.positions)
    centers = nuclei[np.asarray(aos._ragged_kernel_atom)] + images @ cell
    displacement = (wrapped[None, :, :] - jnp.asarray(centers)[:, None, :]).astype(dtype)
    primitive_ao = np.asarray(aos.orbital_indices, int)
    angular = np.asarray(aos.angular_momentums, int)
    exponents = jnp.asarray(aos.exponents, dtype=dtype)
    coefficients = jnp.asarray(aos.coefficients, dtype=dtype)
    if hasattr(aos, "polynominal_order_x"):
        powers = np.asarray([aos.polynominal_order_x, aos.polynominal_order_y, aos.polynominal_order_z]).T
        factors = np.asarray([math.prod(math.factorial(int(n)) / math.factorial(2 * int(n)) for n in p) for p in powers])
        normalization = jnp.sqrt(
            (2 * exponents / jnp.asarray(np.pi, dtype)) ** jnp.asarray(1.5, dtype)
            * (8 * exponents) ** angular[primitive_ao]
            * jnp.asarray(factors[primitive_ao], dtype)
        )
        polynomial = jnp.prod(displacement ** jnp.asarray(powers[:, None, :]), axis=-1)
    else:
        magnetic = np.asarray(aos.magnetic_quantum_numbers, int)
        assert np.max(angular) <= 2
        x, y, z = (displacement[..., i] for i in range(3))
        polynomial = jnp.zeros_like(x)
        shapes = {
            (0, 0): jnp.ones_like(x),
            (1, -1): y,
            (1, 0): z,
            (1, 1): x,
            (2, -2): jnp.sqrt(jnp.asarray(3, dtype)) * x * y,
            (2, -1): jnp.sqrt(jnp.asarray(3, dtype)) * y * z,
            (2, 0): (2 * z * z - x * x - y * y) / 2,
            (2, 1): jnp.sqrt(jnp.asarray(3, dtype)) * x * z,
            (2, 2): jnp.sqrt(jnp.asarray(3, dtype)) * (x * x - y * y) / 2,
        }
        for (ell, m), value in shapes.items():
            polynomial += jnp.asarray(((angular == ell) & (magnetic == m))[:, None]) * value
        factor = np.asarray(
            [2.0 ** (2 * int(ell) + 3) * math.factorial(int(ell) + 1) / math.factorial(2 * int(ell) + 2) for ell in angular]
        )
        normalization = jnp.sqrt(
            jnp.asarray(factor[primitive_ao], dtype)
            * (2 * exponents) ** jnp.asarray(angular[primitive_ao] + 1.5, dtype)
            / jnp.sqrt(jnp.asarray(np.pi, dtype))
        )
        polynomial *= jnp.sqrt(jnp.asarray((2 * angular + 1)[:, None] / (4 * np.pi), dtype))
    radius2 = jnp.sum(displacement[primitive_ao] ** 2, axis=-1)
    primitives = (
        coefficients[:, None] * normalization[:, None] * jnp.exp(-exponents[:, None] * radius2) * polynomial[primitive_ao]
    )
    k = np.asarray(aos.k_point)
    phase = jnp.asarray(np.exp(2j * np.pi * (images @ k)), cdtype)
    weighted = primitives.astype(cdtype) * phase[primitive_ao, None]
    values = jnp.zeros((aos.num_ao, coordinates.shape[0]), dtype=cdtype).at[home[primitive_ao]].add(weighted)
    restore = jnp.exp((2j * jnp.pi * (hop @ jnp.asarray(k))).astype(cdtype))
    return values * restore[None, :]


def determinant_value(wf, up, down):
    gd = wf.geminal_data
    assert type(gd).__name__ == "SD_data"
    dtype = get_complex_dtype_jnp("mo_eval")

    def mos(orb, points):
        return jnp.asarray(orb.mo_coefficients, dtype=dtype) @ orbital_value(orb.aos_data, points).astype(dtype)

    u, d = mos(gd.orb_data_up_spin, up), mos(gd.orb_data_dn_spin, down)
    assert up.shape[0] == down.shape[0]
    return jnp.linalg.det(u.T @ d)


def jastrow_value(wf, up, down):
    jd = wf.jastrow_data
    dtype = get_dtype_jnp("jastrow_eval")
    electrons = jnp.concatenate((up, down)).astype(dtype)
    original_electrons = electrons
    total = jnp.asarray(0, dtype)
    dtn = jd.jastrow_dtn_data
    if dtn is not None:
        assert float(np.asarray(dtn.cutoff_scale)) == 1.0
        sd = dtn.structure_data
        cell = jnp.asarray(sd.cell, dtype=jnp.float64)
        inv = jnp.asarray(np.linalg.inv(np.asarray(sd.cell)))
        frac = electrons.astype(jnp.float64) @ inv
        electrons = ((frac - jnp.floor(frac)) @ cell).astype(dtype)
        nuclei = jnp.asarray(sd.positions, dtype=dtype)

        def distance(a, b):
            frac = (a - b).astype(jnp.float64) @ inv
            return jnp.linalg.norm(((frac - jnp.round(frac)) @ cell).astype(dtype))

        alpha = jnp.einsum("ski,si->sk", jnp.asarray(dtn.M_alpha, dtype), jnp.asarray(dtn.alpha_indep, dtype)) + jnp.asarray(
            dtn.b_alpha, dtype
        )
        beta = jnp.einsum("Iki,Ii->Ik", jnp.asarray(dtn.M_beta, dtype), jnp.asarray(dtn.beta_indep, dtype)) + jnp.asarray(
            dtn.b_beta, dtype
        )
        gamma = jnp.einsum("Ilmni,Ii->Ilmn", jnp.asarray(dtn.M_gamma, dtype), jnp.asarray(dtn.gamma_indep, dtype))
        C = dtn.cutoff_smoothness

        def cutoff(r, L):
            return jnp.maximum(jnp.asarray(L, dtype) - r, 0) ** C

        total = jnp.asarray(0, dtype)
        for i, j in itertools.combinations(range(len(electrons)), 2):
            r = distance(electrons[i], electrons[j])
            if dtn.use_u:
                spin = int((i < len(up)) != (j < len(up)))
                total += cutoff(r, dtn.L_u) * jnp.sum(alpha[spin] * r ** jnp.arange(dtn.n_order_u + 1))
            if dtn.use_f:
                for nucleus, species in zip(nuclei, np.asarray(dtn.species_index), strict=True):
                    ri, rj = distance(electrons[i], nucleus), distance(electrons[j], nucleus)
                    term = jnp.einsum(
                        "lmn,l,m,n->",
                        gamma[species],
                        ri ** jnp.arange(dtn.n_order_eN + 1),
                        rj ** jnp.arange(dtn.n_order_eN + 1),
                        r ** jnp.arange(dtn.n_order_ee + 1),
                    )
                    total += cutoff(ri, dtn.L_f[species]) * cutoff(rj, dtn.L_f[species]) * term
        if dtn.use_chi:
            for electron in electrons:
                for nucleus, species in zip(nuclei, np.asarray(dtn.species_index), strict=True):
                    r = distance(electron, nucleus)
                    total += cutoff(r, dtn.L_chi[species]) * jnp.sum(beta[species] * r ** jnp.arange(dtn.n_order_chi + 1))
        electrons = original_electrons

    def smooth(a, b, sd):
        cell = np.asarray(sd.cell)
        f = (a - b).astype(jnp.float64) @ jnp.asarray(np.linalg.inv(cell))
        return jnp.linalg.norm(jnp.asarray(np.linalg.norm(cell, axis=1) / np.pi) * jnp.sin(jnp.pi * f)).astype(dtype)

    one = jd.jastrow_one_body_data
    if one is not None:
        a = jnp.asarray(one.jastrow_1b_param, dtype)
        charges = np.asarray(one.structure_data.atomic_numbers) - np.asarray(one.core_electrons)
        for electron in electrons:
            for nucleus, Z in zip(np.asarray(one.structure_data.positions), charges, strict=True):
                r = smooth(electron, jnp.asarray(nucleus, dtype), one.structure_data)
                c = jnp.asarray((2 * Z) ** 0.25, dtype)
                pref = jnp.asarray(-((2 * Z) ** 0.75), dtype)
                total += (
                    pref * (-jnp.expm1(-a * c * r)) / (2 * a)
                    if one.jastrow_1b_type == "exp"
                    else pref * c * r / (2 * (1 + a * c * r))
                )
    two = jd.jastrow_two_body_data
    if two is not None:
        a = jnp.asarray(two.jastrow_2b_param, dtype)
        for i, j in itertools.combinations(range(len(electrons)), 2):
            r = smooth(electrons[i], electrons[j], two.structure_data)
            total += -jnp.expm1(-a * r) / (2 * a) if two.jastrow_2b_type == "exp" else r / (2 * (1 + a * r))
    return total


def psi_value(wf, up, down):
    return determinant_value(wf, up, down) * jnp.exp(jastrow_value(wf, up, down).astype(get_dtype_jnp("wf_eval")))


def kinetic_value(wf, up, down):
    positions = jnp.asarray(np.concatenate((up, down)))

    def psi(r):
        return psi_value(wf, r[: len(up)], r[len(up) :])

    hess = jax.jacfwd(jax.jacfwd(psi))(positions)
    return -0.5 * jnp.einsum("ijij->", hess) / psi(positions)


def ewald_value(ham, up, down):
    cp = ham.coulomb_potential_data
    charges = np.asarray(ham.structure_data.atomic_numbers, dtype=float)
    if cp.ecp_flag:
        charges -= np.asarray(cp.z_cores)
    return numpy_ewald_energy(
        np.asarray(ham.structure_data.cell),
        np.concatenate((ham.structure_data.positions, up, down)),
        np.concatenate((charges, -np.ones(len(up) + len(down)))),
    )


def ecp_terms(ham, up, down, RT):
    """Explicit nearest-nucleus Nv=12 contributions before any real projection."""
    cp = ham.coulomb_potential_data
    assert cp.ecp_flag
    dtype = get_dtype_jnp("coulomb")
    complex_dtype = get_complex_dtype_jnp("coulomb")
    cell = np.asarray(ham.structure_data.cell, dtype=dtype)
    inv = np.linalg.inv(cell)
    electrons = np.concatenate((up, down))
    nuclei = np.asarray(ham.structure_data.positions)
    candidates = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)), dtype=dtype)
    displacements = []
    for electron in electrons:
        row = []
        for nucleus in nuclei:
            f = np.asarray(electron - nucleus, dtype=dtype) @ inv
            images = (f - np.round(f) - candidates) @ cell
            row.append(images[np.argmin(np.sum(images * images, axis=1))])
        displacements.append(row)
    displacements = np.asarray(displacements)
    distances = np.linalg.norm(displacements, axis=-1)
    index = np.asarray(cp.nucleus_index)
    angular = np.asarray(cp.ang_moms)
    exponent = np.asarray(cp.exponents, dtype=dtype)
    coefficient = np.asarray(cp.coefficients, dtype=dtype)
    powers = np.asarray(cp.powers, dtype=dtype)

    def radial(atom, ell, r):
        selected = (index == atom) & (angular == ell)
        return np.sum(coefficient[selected] * r ** (powers[selected] - 2) * np.exp(-exponent[selected] * r * r))

    local = sum(radial(atom, cp.max_ang_mom_plus_1[atom], r) for row in distances for atom, r in enumerate(row))
    # Pole-oriented icosahedral quadrature, with all weights exactly 1/12.
    theta = np.asarray([0.0, np.pi] + [np.arctan(2.0)] * 5 + [np.pi - np.arctan(2.0)] * 5)
    phi = np.asarray([0.0, 0.0] + [2 * i * np.pi / 5 for i in range(5)] + [(2 * i + 1) * np.pi / 5 for i in range(5)])
    directions = np.asarray(
        np.stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), axis=1), dtype=dtype
    ) @ np.asarray(RT, dtype=dtype)
    configurations = []
    prefactors = []
    for electron, row in enumerate(distances):
        atom = int(np.argmin(row))
        r = row[atom]
        displacement = displacements[electron, atom]
        for direction in directions:
            moved = electrons.copy()
            # Retain input-coordinate precision; only the offset arithmetic is
            # in the Coulomb zone, as in the declared downstream use-site rule.
            moved[electron] = electrons[electron] + (-displacement + r * direction)
            cosine = np.dot(displacement / r, direction)
            prefactor = np.sum(
                np.asarray(
                    [
                        dtype(2 * ell + 1) * dtype(eval_legendre(ell, cosine)) * radial(atom, ell, r)
                        for ell in range(cp.max_ang_mom_plus_1[atom])
                    ],
                    dtype=dtype,
                ),
                dtype=dtype,
            ) / dtype(12)
            configurations.append(moved)
            prefactors.append(prefactor)

    wf = ham.wavefunction_data

    def determinant(points):
        return determinant_value(wf, points[: len(up)], points[len(up) :])

    def jastrow(points):
        return jastrow_value(wf, points[: len(up)], points[len(up) :])

    configurations = np.asarray(configurations)
    base = jnp.asarray(electrons)
    det_values = np.asarray(jax.jit(jax.vmap(determinant))(jnp.asarray(configurations)))
    j_values = np.asarray(jax.jit(jax.vmap(jastrow))(jnp.asarray(configurations)))
    det_ratio = np.asarray(det_values / np.asarray(determinant(base)), dtype=complex_dtype)
    jratio = np.asarray(np.exp(j_values - np.asarray(jastrow(base))), dtype=complex_dtype)
    prefactors = np.asarray(prefactors, dtype=dtype)
    det_terms = prefactors * det_ratio
    full_terms = det_terms * jratio
    return local, configurations[:, : len(up)], configurations[:, len(up) :], det_terms, jratio, full_terms


def complex_local_energy_reference(ham, up, down, RT):
    """Independent complex H Psi/Psi, not a correction to production energy."""
    local, _up, _down, _det, _j, full_terms = ecp_terms(ham, up, down, RT)
    return kinetic_value(ham.wavefunction_data, up, down) + ewald_value(ham, up, down) + local + np.sum(full_terms)
