"""Evaluate the building blocks of a jQMC wave function and Hamiltonian by hand.

This script walks through the low-level kernels that the MCMC and LRDMC
drivers call internally: atomic and molecular orbital values, the bare
Coulomb terms, and the local and non-local parts of the effective core
potential (ECP), all evaluated at one electron configuration. It reads the
TREXIO file written by ``run_pyscf.py`` in the same directory.
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

import jax
import numpy as np

from jqmc import (
    Jastrow_data,
    Jastrow_two_body_data,
    Wavefunction_data,
    compute_AOs,
    compute_bare_coulomb_potential_el_el,
    compute_bare_coulomb_potential_el_ion,
    compute_bare_coulomb_potential_ion_ion,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors,
    compute_MOs,
    read_trexio_file,
)

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# 1. Read the trial wave function from the TREXIO file
# ---------------------------------------------------------------------------
(
    structure_data,
    aos_data,
    mos_data_up,
    mos_data_dn,
    geminal_data,
    coulomb_potential_data,
) = read_trexio_file("water_ccecp_ccpvqz.h5")

print("Structure:")
for z, pos in zip(structure_data.atomic_numbers, structure_data.positions, strict=True):
    print(f"  Z = {int(z):2d}  R = {np.asarray(pos)}")
print(f"Number of AOs (Cartesian GTOs): {aos_data.num_ao}")
print(f"Number of occupied MOs (up / dn): {mos_data_up.num_mo} / {mos_data_dn.num_mo}")
print(f"Determinant type: {type(geminal_data).__name__}")
print(f"Number of electrons (up / dn): {geminal_data.num_electron_up} / {geminal_data.num_electron_dn}")
print(f"ECP present: {coulomb_potential_data.ecp_flag}")
print()

# ---------------------------------------------------------------------------
# 2. Assemble a wave function: two-body Jastrow x Slater determinant
# ---------------------------------------------------------------------------
jastrow_two_body_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)
jastrow_data = Jastrow_data(
    jastrow_one_body_data=None,
    jastrow_two_body_data=jastrow_two_body_data,
    jastrow_three_body_data=None,
)
wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)

# ---------------------------------------------------------------------------
# 3. Pick one electron configuration (4 up + 4 down electrons for water/ccECP)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=0)
num_ele_up = geminal_data.num_electron_up
num_ele_dn = geminal_data.num_electron_dn
center = np.asarray(structure_data.positions).mean(axis=0)
r_up_carts = center + rng.uniform(-1.0, 1.0, size=(num_ele_up, 3))
r_dn_carts = center + rng.uniform(-1.0, 1.0, size=(num_ele_dn, 3))

# ---------------------------------------------------------------------------
# 4. Atomic and molecular orbital values at the electron positions
# ---------------------------------------------------------------------------
ao_up_values = compute_AOs(aos_data=aos_data, r_carts=r_up_carts)  # (num_ao, N_up)
ao_dn_values = compute_AOs(aos_data=aos_data, r_carts=r_dn_carts)  # (num_ao, N_dn)
mo_up_values = compute_MOs(mos_data=mos_data_up, r_carts=r_up_carts)  # (num_mo, N_up)
mo_dn_values = compute_MOs(mos_data=mos_data_dn, r_carts=r_dn_carts)  # (num_mo, N_dn)
print(f"AO values: up {ao_up_values.shape}, dn {ao_dn_values.shape}")
print(f"MO values: up {mo_up_values.shape}, dn {mo_dn_values.shape}")
print()

# ---------------------------------------------------------------------------
# 5. Bare Coulomb terms (ion-ion, electron-electron, electron-ion)
# ---------------------------------------------------------------------------
V_ii = compute_bare_coulomb_potential_ion_ion(coulomb_potential_data=coulomb_potential_data)
V_ee = compute_bare_coulomb_potential_el_el(r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
V_ei = compute_bare_coulomb_potential_el_ion(
    coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
)
print(f"V_ii = {float(V_ii):+.8f} Ha")
print(f"V_ee = {float(V_ee):+.8f} Ha")
print(f"V_ei = {float(V_ei):+.8f} Ha")

# ---------------------------------------------------------------------------
# 6. ECP: local part (diagonal) and non-local part (angular projection)
# ---------------------------------------------------------------------------
V_local = compute_ecp_local_parts_all_pairs(
    coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
)
print(f"V_ecp_local = {float(V_local):+.8f} Ha")

# The non-local part needs the wave-function ratios on the angular quadrature
# mesh (Nv points around every nucleus); RT is the mesh rotation matrix. The
# identity is used here; production sampling draws a random rotation.
RT = np.eye(3)

# (a) every electron-nucleus pair
mesh_up, mesh_dn, V_nonlocal, sum_V_nonlocal = compute_ecp_non_local_parts_all_pairs(
    coulomb_potential_data=coulomb_potential_data,
    wavefunction_data=wavefunction_data,
    r_up_carts=r_up_carts,
    r_dn_carts=r_dn_carts,
    RT=RT,
    Nv=6,
)
print("Non-local ECP, all electron-nucleus pairs:")
print(f"  quadrature configurations: up {np.asarray(mesh_up).shape}, dn {np.asarray(mesh_dn).shape}")
print(f"  matrix elements: {np.asarray(V_nonlocal).shape}, sum = {float(sum_V_nonlocal):+.8f} Ha")

# (b) only the NN nearest nuclei of every electron (the production path)
mesh_up, mesh_dn, V_nonlocal, sum_V_nonlocal = compute_ecp_non_local_parts_nearest_neighbors(
    coulomb_potential_data=coulomb_potential_data,
    wavefunction_data=wavefunction_data,
    r_up_carts=r_up_carts,
    r_dn_carts=r_dn_carts,
    RT=RT,
    NN=1,
    Nv=6,
)
print("Non-local ECP, nearest neighbour only (NN = 1):")
print(f"  quadrature configurations: up {np.asarray(mesh_up).shape}, dn {np.asarray(mesh_dn).shape}")
print(f"  matrix elements: {np.asarray(V_nonlocal).shape}, sum = {float(sum_V_nonlocal):+.8f} Ha")
