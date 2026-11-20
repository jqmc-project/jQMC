# jqmc-example11: Python API for orbital and Hamiltonian kernels

This example shows how to call the low-level building blocks of jQMC from
Python: reading a TREXIO file into the jQMC data containers, evaluating
atomic and molecular orbitals at given electron positions, and evaluating
the bare Coulomb and effective-core-potential (ECP) terms of the Hamiltonian
for one electron configuration. These are the kernels that the `MCMC` and
`GFMC_t` / `GFMC_n` drivers call internally. For the driver-level Python API
(VMC optimization, sampling, LRDMC) see `jqmc-example12` and
`jqmc-example13`.

## Before you start

The example needs `jqmc` and PySCF with TREXIO export (for example
`pip install pyscf pyscf-forge trexio`; check with
`python -c "from pyscf.tools import trexio"`). Both scripts are serial
and need no MPI or GPU.

## Step 1: Generate the trial wave function with PySCF

Run a Hartree-Fock calculation for a water molecule with the ccECP
pseudopotential and the matching cc-pVQZ basis, and export it to TREXIO:

<!-- include: run_pyscf.py -->
```python
from pyscf import gto, scf
from pyscf.tools import trexio

filename = "water_ccecp_ccpvqz.h5"

mol = gto.Mole()
mol.verbose = 5
mol.atom = """
               O    5.00000000   7.14707700   7.65097100
               H    4.06806600   6.94297500   7.56376100
               H    5.38023700   6.89696300   6.80798400
               """
mol.basis = "ccecp-ccpvqz"
mol.unit = "A"
mol.ecp = "ccecp"
mol.charge = 0
mol.spin = 0
mol.symmetry = False
mol.cart = True
mol.output = "water.out"
mol.build()

mf = scf.HF(mol)
mf.max_cycle = 200
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)
```

```bash
python run_pyscf.py
```

This writes `water_ccecp_ccpvqz.h5` (and the PySCF log `water.out`). The
script prints only `output file: water.out`; the SCF result is at the end of
`water.out`:

```bash
grep "converged SCF energy" water.out
```

```text
converged SCF energy = -16.9450309201805
```

## Step 2: Evaluate the wave-function and Hamiltonian kernels

```bash
python how_to_use_modules.py
```

<!-- include: how_to_use_modules.py -->
```python
"""Evaluate the building blocks of a jQMC wave function and Hamiltonian by hand.

This script walks through the low-level kernels that the MCMC and LRDMC
drivers call internally: atomic and molecular orbital values, the bare
Coulomb terms, and the local and non-local parts of the effective core
potential (ECP), all evaluated at one electron configuration. It reads the
TREXIO file written by ``run_pyscf.py`` in the same directory.
"""

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
```

Output of our run (one process on an AMD EPYC 7713 node, about 10 s wall time
including JIT compilation):

```text
Structure:
  Z =  8  R = [ 9.44863062 13.50601812 14.45823978]
  Z =  1  R = [ 7.6875306  13.12032124 14.29343676]
  Z =  1  R = [10.16717442 13.03337116 12.86522522]
Number of AOs (Cartesian GTOs): 139
Number of occupied MOs (up / dn): 4 / 4
Determinant type: SD_data
Number of electrons (up / dn): 4 / 4
ECP present: True

AO values: up (139, 4), dn (139, 4)
MO values: up (4, 4), dn (4, 4)

V_ii = +6.97782447 Ha
V_ee = +21.48923885 Ha
V_ei = -51.94521475 Ha
V_ecp_local = -0.03000664 Ha
Non-local ECP, all electron-nucleus pairs:
  quadrature configurations: up (144, 4, 3), dn (144, 4, 3)
  matrix elements: (144,), sum = -3.48859644 Ha
Non-local ECP, nearest neighbour only (NN = 1):
  quadrature configurations: up (48, 4, 3), dn (48, 4, 3)
  matrix elements: (48,), sum = -3.48859495 Ha
```

The positions are printed in bohr (the PySCF input is in angstrom). The two
non-local sums agree to about 1.5e-6 Ha here: the nearest-neighbour variant
evaluates only one nucleus per electron (48 instead of 144 quadrature
configurations), and the contributions of the farther nuclei are negligible
because the non-local ECP projectors are short-ranged.

## What the script shows

- `read_trexio_file` returns the six jQMC data containers: `Structure_data`,
  the AO basis (`AOs_cart_gto_data` here, because PySCF exported Cartesian
  GTOs), the occupied MOs for each spin, the Slater determinant (`SD_data`),
  and `Coulomb_potential_data` including the ECP tables. A complete summary of
  a Hamiltonian file can be printed with `jqmc-tool hamiltonian show-info`.
- `compute_AOs` / `compute_MOs` return arrays of shape `(num_orbitals, N_e)`.
  The gradient and Laplacian variants (`compute_AOs_grad`,
  `compute_AOs_laplacian`, `compute_MOs_grad`, `compute_MOs_laplacian`) have
  the same call signature.
- The bare Coulomb energy is split into the ion-ion, electron-electron and
  electron-ion channels; on ECP atoms the electron-ion channel uses the
  effective charge.
- The ECP local part is a diagonal term. The non-local part is evaluated by
  the angular quadrature with `Nv` points around each nucleus and returns the
  displaced electron configurations, the corresponding matrix elements, and
  their sum. The `nearest_neighbors` variant, which restricts the sum to the
  `NN` closest nuclei of every electron, is what the production drivers use.
  It is the only variant for periodic systems, because the `all_pairs`
  variant loops over the atoms of one cell without periodic images and
  raises an error when the structure is periodic; for molecules it serves
  as a reference.
- The script uses `Nv = 6` quadrature points to keep the printed arrays
  small; the production default is `Nv = 12`.

The electron configuration is drawn with a fixed seed
(`np.random.default_rng(seed=0)`), so for a given TREXIO file the printed
numbers are reproducible; they still depend on the PySCF orbitals and are
not physical observables, only an illustration of how each term is
obtained. The printed array shapes follow from the system: 4 up and 4 down
electrons for water with ccECP. The technical notes on the Hamiltonian in
the manual describe the formulas behind every kernel called here.
