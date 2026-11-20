# jqmc-example12: Serial Python API for water VMC and LRDMC

This example uses jQMC as a Python library instead of the `jqmc` /
`jqmc-tool` commands. One script builds the wave function from a TREXIO file,
optimizes the Jastrow factor, runs a VMC production run, runs LRDMC at three
lattice spacings and extrapolates the LRDMC energy to $a \to 0$. Working in
Python makes it easy to chain steps, loop over parameters, or post-process
results without intermediate files. The MPI version of the same script is
`jqmc-example13`; the command-line version of the same calculation is
`jqmc-example01`.

## Before you start

* PySCF with TREXIO export (for example `pip install pyscf pyscf-forge trexio`;
  `python -c "from pyscf.tools import trexio"` must succeed).
* `jqmc` installed in the same environment.

All commands are run inside `examples/jqmc-example12`. The script runs in one
process (no MPI). JAX uses the threads of the CPU it runs on, so a single
process still benefits from a multi-core node.

The numbers below come from one run on one 128-core CPU node (AMD EPYC 7713),
one Python process with JAX allowed to use all 128 cores. The whole script took
61 min of wall time: 34 min for the
optimization, 10 s for the VMC run, and 3.5, 5.7 and 17.3 min for the LRDMC
runs at `alat` = 0.3, 0.2 and 0.1. To make a quick trial, reduce
`num_opt_steps` and `num_lrdmc_steps` in `run_jqmc.py`, at the cost of a less
converged Jastrow factor and larger error bars. The
settings in the script are tutorial values, not converged production choices.

## Step 1: Generate the trial wave function with PySCF

`run_pyscf.py` runs a Hartree-Fock calculation for water with the ccECP
pseudopotential and the cc-pVTZ basis and exports it to TREXIO:

<!-- include: run_pyscf.py -->
```python
from pyscf import gto, scf
from pyscf.tools import trexio

filename = "water_ccecp_ccpvtz.h5"

mol = gto.Mole()
mol.verbose = 5
mol.atom = """
               O    5.00000000   7.14707700   7.65097100
               H    4.06806600   6.94297500   7.56376100
               H    5.38023700   6.89696300   6.80798400
               """
mol.basis = "ccecp-ccpvtz"
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
grep "converged SCF energy" water.out
```

```text
converged SCF energy = -16.9437549661298
```

This writes `water_ccecp_ccpvtz.h5` (and the PySCF log `water.out`).

## Step 2: Run the QMC workflow with the jQMC API

```bash
python run_jqmc.py > run_jqmc.log 2>&1
```

<!-- include: run_jqmc.py -->
```python
import logging
import os
import subprocess
import sys

import jax
import numpy as np

from jqmc import (
    GFMC_t,
    Hamiltonian_data,
    Jastrow_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    MCMC,
    Wavefunction_data,
    read_trexio_file,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("jqmc")
logger.setLevel(logging.INFO)

# JAX configuration
jax.config.update("jax_enable_x64", True)

# =============================================================================
# 0. Settings
# =============================================================================
trexio_file = "water_ccecp_ccpvtz.h5"
hamiltonian_file = "hamiltonian_data.h5"

if not os.path.exists(trexio_file):
    logger.error(f"{trexio_file} not found. Please run run_pyscf.py first.")
    sys.exit(1)

# =============================================================================
# 1. Convert TREXIO to Hamiltonian
# =============================================================================
logger.info(f"Converting {trexio_file} to {hamiltonian_file}...")

(structure_data, aos_data, mos_data, _, geminal_data, coulomb_potential_data) = read_trexio_file(trexio_file, store_tuple=True)

# Jastrow factor J2 + J3, as `jqmc-tool trexio convert-to ... -j2 1.0 -j3 mo`.
# No one-body Jastrow: the J1 form imposes the electron-nucleus cusp with the
# effective charge, which the smooth ccECP pseudopotentials do not need.
# Jastrow 2-body (Pade, initial parameter 1.0)
jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)

# Jastrow 3-body in the occupied-MO basis (J matrix initialized to zero)
jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=mos_data)

jastrow_data = Jastrow_data(
    jastrow_one_body_data=None,
    jastrow_two_body_data=jastrow_twobody_data,
    jastrow_three_body_data=jastrow_threebody_data,
)

wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)

hamiltonian_data = Hamiltonian_data(
    structure_data=structure_data,
    coulomb_potential_data=coulomb_potential_data,
    wavefunction_data=wavefunction_data,
)

hamiltonian_data.save_to_hdf5(hamiltonian_file)
logger.info(f"Saved Hamiltonian data to {hamiltonian_file}")


# =============================================================================
# 2. VMC Optimization
# =============================================================================
logger.info("Starting VMC Optimization...")

mcmc_opt = MCMC(
    hamiltonian_data=hamiltonian_data,
    Dt=2.0,
    mcmc_seed=42,
    num_walkers=2000,
    num_mcmc_per_measurement=40,
    epsilon_AS=0.0,
    comput_position_deriv=False,
    comput_log_WF_param_deriv=True,  # needed by the SR optimizer
    comput_e_L_param_deriv=True,  # needed by use_lm=True
)

mcmc_opt.run_optimize(
    num_mcmc_steps=200,  # Measurements per optimization step
    num_opt_steps=50,
    wf_dump_freq=1,  # dump every step: the last 20 are averaged below
    num_mcmc_warmup_steps=30,
    num_mcmc_bin_blocks=10,
    opt_J1_param=False,
    opt_J2_param=True,
    opt_J3_param=True,
    opt_geminal=False,
    max_time=3600,
    optimizer_kwargs={"method": "sr", "delta": 0.15, "epsilon": 0.001, "use_lm": True},
)

# Hand the sampling runs the uniform average of the last 20 optimization steps,
# as the command-line tutorials do (copy_optimized_hamiltonian.py runs
# `jqmc-tool vmc average-wf`); mcmc_opt.hamiltonian_data is only the last step.
num_averaged_steps = 20
averaged_file = "hamiltonian_data_averaged.h5"
if os.path.exists(averaged_file):
    os.remove(averaged_file)  # left over from a previous run of this script
helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "copy_optimized_hamiltonian.py")
subprocess.run([sys.executable, helper, "--average-last", str(num_averaged_steps), ".", averaged_file], check=True)
optimized_hamiltonian = Hamiltonian_data.load_from_hdf5(averaged_file)


# =============================================================================
# 3. VMC Sampling
# =============================================================================
logger.info("Starting VMC Sampling...")

mcmc_prod = MCMC(
    hamiltonian_data=optimized_hamiltonian,
    Dt=2.0,
    mcmc_seed=123,
    num_walkers=2000,
    num_mcmc_per_measurement=40,
    epsilon_AS=0.0,
    comput_position_deriv=False,
    comput_log_WF_param_deriv=False,
)

num_vmc_steps = 1000
mcmc_prod.run(num_mcmc_steps=num_vmc_steps, max_time=3600)

vmc_E_mean, vmc_E_std, _, _ = mcmc_prod.get_E(
    num_mcmc_warmup_steps=int(num_vmc_steps * 0.1),
    num_mcmc_bin_blocks=10,
)

logger.info(f"VMC Energy: {vmc_E_mean:.5f} +/- {vmc_E_std:.5f} Ha")


# =============================================================================
# 4. LRDMC with multiple alats
# =============================================================================
alats = [0.3, 0.2, 0.1]
lrdmc_results = []

for alat in alats:
    logger.info(f"Starting LRDMC with alat={alat}...")

    lrdmc = GFMC_t(
        hamiltonian_data=optimized_hamiltonian,
        num_walkers=2000,
        tau=0.10,  # imaginary time between two branchings (1/Ha)
        num_gfmc_collect_steps=20,
        mcmc_seed=456,
        alat=alat,
        non_local_move="tmove",
        comput_position_deriv=False,
    )

    num_lrdmc_steps = 3000
    lrdmc.run(num_mcmc_steps=num_lrdmc_steps, max_time=3600)

    lrdmc_E_mean, lrdmc_E_std, _, _ = lrdmc.get_E(
        num_mcmc_warmup_steps=int(num_lrdmc_steps * 0.1),
        num_mcmc_bin_blocks=10,
    )

    logger.info(f"LRDMC Energy (alat={alat}): {lrdmc_E_mean:.5f} +/- {lrdmc_E_std:.5f} Ha")
    lrdmc_results.append((alat, lrdmc_E_mean, lrdmc_E_std))


# =============================================================================
# 5. Extrapolation
# =============================================================================
# Simple linear regression E(a) = E_0 + k * a^2 (usually LRDMC error scales as a^2)

x = np.array([r[0] ** 2 for r in lrdmc_results])
y = np.array([r[1] for r in lrdmc_results])
w = np.array([1.0 / r[2] ** 2 for r in lrdmc_results])  # weights

# Weighted linear regression
# y = c0 + c1 * x
A = np.vstack([x, np.ones(len(x))]).T
W = np.diag(w)
cov = np.linalg.inv(A.T @ W @ A)  # covariance of (c1, c0) from the input error bars
c1, c0 = cov @ (A.T @ W @ y)
c0_err = np.sqrt(cov[1, 1])
residuals = y - (c0 + c1 * x)
chi2 = float(np.sum(w * residuals**2))
dof = len(x) - 2  # data points minus fitted parameters

logger.info("-" * 40)
logger.info("Extrapolation Results (E vs a^2):")
for a, E, err in lrdmc_results:
    logger.info(f"alat={a:.2f}, a^2={a**2:.4f}, E={E:.5f} +/- {err:.5f}")

logger.info(f"Extrapolated Energy (a->0): {c0:.5f} +/- {c0_err:.5f} Ha")
logger.info(f"Fit quality: chi^2 = {chi2:.2f} for {dof} degree(s) of freedom")
logger.info("-" * 40)
```

The script performs these steps:

1. **Build the Hamiltonian.** `read_trexio_file` returns the structure, the AO
   and MO data, the Slater determinant and the Coulomb/ECP data. The script
   adds a two-body Pade Jastrow (initial parameter 1.0) and a three-body
   Jastrow in the basis of the occupied MOs (J matrix initialized to zero),
   the same J2 + J3 form that `jqmc-tool trexio convert-to ... -j2 1.0 -j3 mo`
   builds in `jqmc-example01`, and saves the result as `hamiltonian_data.h5`.
   No one-body Jastrow is used: the J1 form imposes the electron-nucleus cusp
   for the effective charge of every atom, which a smooth ccECP
   pseudopotential does not have; started from the same initial value 1.0 it
   raised the VMC energy of this system to about -4.4 Ha in a test.
2. **Optimize the Jastrow factor** with `MCMC.run_optimize`: 2000 walkers,
   50 steps of 200 measurements (40 moves each, 30 discarded, 10 bins), SR
   with `delta = 0.15` and `use_lm = True`, which needs
   `comput_e_L_param_deriv=True` in the `MCMC` constructor. With
   `wf_dump_freq=1` it writes `hamiltonian_data_opt_step_{1,...,50}.h5`;
   the script then averages the last 20 of them (steps 31-50, on the energy
   plateau, see Step 3) with `../copy_optimized_hamiltonian.py
   --average-last 20` into `hamiltonian_data_averaged.h5` and continues with
   that file.
3. **VMC sampling**: 2000 walkers, 1000 measurements, 10% discarded,
   10 bins.
4. **LRDMC** (`GFMC_t`, LRDMC with a fixed projection time between
   branchings) for `alat = 0.3, 0.2, 0.1` bohr: 2000 walkers, 3000 branching
   steps with `tau = 0.10` Ha$^{-1}$ of imaginary time each,
   correcting-factor history `num_gfmc_collect_steps = 20`, T-moves for the
   non-local ECP, 10% discarded, 10 bins. Because `tau` is fixed, every
   branching step covers the same imaginary time at every `alat`; a smaller
   `alat` needs more lattice moves (projections) to cover it, so the cost of
   a branching step grows roughly as `1/alat**2`.
5. **Extrapolation**: weighted linear fit $E(a) = E_0 + k a^2$ with weights
   $1/\sigma^2$; the error of $E_0$ comes from the fit covariance, and
   $\chi^2$ is printed for the one degree of freedom of a three-point fit.

The `MCMC` and `GFMC_t` class defaults are not the command-line defaults (for
example `MCMC` uses `epsilon_AS = 0.1` and computes no parameter derivatives
by default), so the script passes every setting explicitly.

## Step 3: Check the results

The optimization log reports, for every step, the energy before the update
and the largest generalized force with its signal-to-noise ratio (S/N). From
our `run_jqmc.log` (`grep -E "^E = |signal-to-noise" run_jqmc.log`), first
and last steps:

```text
Optimization step = 1/50.
E = -16.61667 +- 0.00391 Ha
Max generalized force signal-to-noise = 272.182.
...
Optimization step = 10/50.
E = -17.04076 +- 0.00291 Ha
Max generalized force signal-to-noise = 101.297.
...
Optimization step = 20/50.
E = -17.08903 +- 0.00282 Ha
Max generalized force signal-to-noise = 19.459.
...
Optimization step = 50/50.
E = -17.08782 +- 0.00276 Ha
Max generalized force signal-to-noise = 3.672.
```

The energy stops drifting after about 20 steps, and from step 30 on the S/N
stays between about 1 and 7: the remaining forces are mostly noise. If the
S/N is still large at the end of your run, increase `num_opt_steps`.

> [!NOTE]
> The script hands the VMC and LRDMC runs the uniform average of the last 20
> optimization steps (31-50), written by `../copy_optimized_hamiltonian.py
> --average-last 20` with `jqmc-tool vmc average-wf`; see
> [Averaging the last optimization steps](../../doc/notes/wavefunction_optimization.md).
> The VMC, LRDMC and extrapolation numbers below were not sampled from this
> average: the reference run sampled the
> in-memory last step (`mcmc_opt.hamiltonian_data`, step 50) and dumped
> every 10 steps. The
> optimization lines are unaffected (the dump frequency does not change the
> optimization).

The VMC, LRDMC and extrapolation lines at the end of `run_jqmc.log`:

```text
VMC Energy: -17.09003 +/- 0.00122 Ha
...
LRDMC Energy (alat=0.3): -17.26532 +/- 0.00110 Ha
...
LRDMC Energy (alat=0.2): -17.24753 +/- 0.00076 Ha
...
LRDMC Energy (alat=0.1): -17.24187 +/- 0.00095 Ha
----------------------------------------
Extrapolation Results (E vs a^2):
alat=0.30, a^2=0.0900, E=-17.26532 +/- 0.00110
alat=0.20, a^2=0.0400, E=-17.24753 +/- 0.00076
alat=0.10, a^2=0.0100, E=-17.24187 +/- 0.00095
Extrapolated Energy (a->0): -17.23730 +/- 0.00093 Ha
Fit quality: chi^2 = 8.92 for 1 degree(s) of freedom
----------------------------------------
```

* The extrapolated LRDMC energy, -17.2373(9) Ha, is 0.147 Ha below the VMC
  energy: the fixed-node projection removes most of the error of the J2 + J3
  Jastrow factor, while the nodes stay those of the Hartree-Fock determinant.
* The LRDMC energy rises as `alat` decreases: the lattice error lowers the
  energy at large `alat` here.
* $\chi^2 = 8.92$ for one degree of freedom is large: at this sub-mHa
  precision the three points are not described well by a pure $a^2$ line
  (the `alat = 0.2` point lies about 2 mHa above it). The MPI run of
  `jqmc-example13` with the same settings gave $\chi^2 = 0.00$ and an
  extrapolated energy within one error bar of this one, so the extrapolated
  value is robust at the 1-2 mHa level; for a sharper result use more
  `alat` values, smaller ones, or a fit with an additional $a^4$ term.
* Each branching step covers the same imaginary time (`tau = 0.10`), so the
  LRDMC log reports a growing number of lattice moves per branching step as
  `alat` decreases (`Average of the number of projections` = 27, 60 and 240)
  and the cost grows with it (68, 113 and 344 ms of projection per branching
  step in our run). The `Survived walkers ratio` stays at about 96.3-96.5%
  for all three `alat` values.
