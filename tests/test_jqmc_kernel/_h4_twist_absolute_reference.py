"""Independent H4 Cartesian Bloch/J1/J2 value and Hessian/Ewald references."""

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
from jqmc._precision import get_dtype_jnp, get_complex_dtype_jnp
from tests.test_jqmc_kernel._noncubic_oracles import numpy_ewald_energy


def orbital_value(aos, coordinates):
    """Explicit primitive Gaussian/image sum; no production numerical AO helper."""
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
    assert hasattr(aos, "polynominal_order_x")
    powers = np.asarray([aos.polynominal_order_x, aos.polynominal_order_y, aos.polynominal_order_z]).T
    factors = np.asarray([math.prod((math.factorial(int(n)) / math.factorial(2 * int(n)) for n in p)) for p in powers])
    normalization = jnp.sqrt(
        (2 * exponents / jnp.asarray(np.pi, dtype)) ** jnp.asarray(1.5, dtype)
        * (8 * exponents) ** angular[primitive_ao]
        * jnp.asarray(factors[primitive_ao], dtype)
    )
    polynomial = jnp.prod(displacement ** jnp.asarray(powers[:, None, :]), axis=-1)
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

    (u, d) = (mos(gd.orb_data_up_spin, up), mos(gd.orb_data_dn_spin, down))
    assert up.shape[0] == down.shape[0]
    return jnp.linalg.det(u.T @ d)


def jastrow_value(wf, up, down):
    jd = wf.jastrow_data
    dtype = get_dtype_jnp("jastrow_eval")
    electrons = jnp.concatenate((up, down)).astype(dtype)
    total = jnp.asarray(0, dtype)
    dtn = jd.jastrow_dtn_data
    assert dtn is None

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
                    pref * -jnp.expm1(-a * c * r) / (2 * a)
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
