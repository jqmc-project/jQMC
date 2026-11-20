# jqmc-example13: MPI Python API for water VMC and LRDMC

This example is the MPI version of `jqmc-example12`: the same water
calculation (Hartree-Fock trial wave function with a J2 + J3 Jastrow factor,
Jastrow optimization, VMC, LRDMC at three lattice spacings and the
$a \to 0$ extrapolation) driven from Python, with the walkers distributed over
MPI processes. The physics settings, seeds and step counts are those of
`jqmc-example12`; only the parallel setup differs:

| | `jqmc-example12/run_jqmc.py` | `jqmc-example13/run_jqmc_mpi.py` |
|---|---|---|
| Parallelism | one process | `mpirun -np N`; `jax.distributed.initialize` after `mpi4py` |
| Walkers | 2000 | 2000 in total, `2000 // N` per MPI process (`N` must divide 2000) |
| Logging and file writes | single process | user-level logs and `save_to_hdf5` only on rank 0 |

Because the random-number streams differ, the two scripts give energies that
agree within their error bars, not identical numbers.

## Before you start

* PySCF with TREXIO export (for example `pip install pyscf pyscf-forge trexio`;
  `python -c "from pyscf.tools import trexio"` must succeed).
* `jqmc`, `mpi4py` and an MPI implementation (for example Open MPI).

All commands are run inside `examples/jqmc-example13`. This example has no
TOML files: every setting is in `run_jqmc_mpi.py`.

The numbers below come from one run with 125 MPI processes (16 walkers each,
one thread per process) on one 128-core CPU node (AMD EPYC 7713). The script
took 32 min of wall time: 8 min for the
optimization, 10 s for the VMC run, and 2.4, 4.6 and 16.4 min for the LRDMC
runs at `alat` = 0.3, 0.2 and 0.1. To make a quick trial, reduce
`num_lrdmc_steps` (and `num_opt_steps`) in the script, at the cost of larger
error bars.

## Step 1: Generate the trial wave function with PySCF

`run_pyscf.py` is the same serial script as in `jqmc-example12` (HF, ccECP,
cc-pVTZ):

```bash
python run_pyscf.py
grep "converged SCF energy" water.out
```

```text
converged SCF energy = -16.9437549661298
```

This writes `water_ccecp_ccpvtz.h5`.

## Step 2: Run the QMC workflow with MPI

```bash
mpirun -np 125 python run_jqmc_mpi.py > run_jqmc_mpi.log 2>&1
```

Replace `125` with the number of MPI processes to use; it must divide the
total of 2000 walkers, otherwise the script stops with an error message.

<!-- include: run_jqmc_mpi.py -->
```python
import logging
import os
import subprocess
import sys

import jax

# Enable 64-bit floats right after importing JAX, before jQMC is imported and
# before any JAX array or distributed runtime is created, so that every array
# jQMC builds is float64.
jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

from jqmc import (  # noqa: E402
    GFMC_t,
    Hamiltonian_data,
    Jastrow_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    MCMC,
    Wavefunction_data,
    read_trexio_file,
)

# MPI Init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("jqmc")
if rank == 0:
    logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.ERROR)  # Suppress logs from other ranks

# JAX configuration. The drivers use JAX collectives across MPI ranks, so the
# JAX distributed runtime must know about the MPI communicator before any
# driver is constructed (the ``jqmc`` command does this automatically).
# HTTP(S) proxy variables can make the gRPC connection of the JAX coordination
# service hang (the ``jqmc`` command removes them too).
for _proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_proxy_var, None)
jax.distributed.initialize(cluster_detection_method="mpi4py")

# =============================================================================
# 0. Settings
# =============================================================================
trexio_file = "water_ccecp_ccpvtz.h5"
hamiltonian_file = "hamiltonian_data.h5"

# The total number of walkers is the same as in the serial jqmc-example12;
# it is split evenly over the MPI ranks.
num_walkers_total = 2000
if num_walkers_total % size != 0:
    if rank == 0:
        logger.error(f"num_walkers_total = {num_walkers_total} is not divisible by {size} MPI processes.")
    sys.exit(1)
num_walkers_per_rank = num_walkers_total // size

if rank == 0:
    logger.info(f"Running on {size} MPI processes, {num_walkers_per_rank} walkers per process.")

if not os.path.exists(trexio_file):
    if rank == 0:
        logger.error(f"{trexio_file} not found. Please run run_pyscf.py first.")
    sys.exit(1)

# =============================================================================
# 1. Convert TREXIO to Hamiltonian
# =============================================================================
if rank == 0:
    logger.info(f"Converting {trexio_file} to {hamiltonian_file}...")

# All ranks read TREXIO (it's read-only)
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

if rank == 0:
    hamiltonian_data.save_to_hdf5(hamiltonian_file)
    logger.info(f"Saved Hamiltonian data to {hamiltonian_file}")


# =============================================================================
# 2. VMC Optimization
# =============================================================================
if rank == 0:
    logger.info("Starting VMC Optimization...")

mcmc_opt = MCMC(
    hamiltonian_data=hamiltonian_data,
    Dt=2.0,
    mcmc_seed=42,
    num_walkers=num_walkers_per_rank,
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
# Rank 0 wrote the dumps, so rank 0 averages them in a separate non-MPI process
# (the MPI launcher variables are removed from its environment); then every
# rank reads the result.
num_averaged_steps = 20
averaged_file = "hamiltonian_data_averaged.h5"
status = 0
if rank == 0:
    if os.path.exists(averaged_file):
        os.remove(averaged_file)  # left over from a previous run of this script
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "copy_optimized_hamiltonian.py")
    launcher = ("PMI_", "PMIX_", "HYDRA_", "OMPI_", "MPI_LOCAL")
    environment = {key: value for key, value in os.environ.items() if not key.startswith(launcher)}
    command = [sys.executable, helper, "--average-last", str(num_averaged_steps), ".", averaged_file]
    status = subprocess.run(command, env=environment).returncode
if comm.bcast(status, root=0) != 0:
    sys.exit(1)
optimized_hamiltonian = Hamiltonian_data.load_from_hdf5(averaged_file)


# =============================================================================
# 3. VMC Sampling
# =============================================================================
if rank == 0:
    logger.info("Starting VMC Sampling...")

mcmc_prod = MCMC(
    hamiltonian_data=optimized_hamiltonian,
    Dt=2.0,
    mcmc_seed=123,
    num_walkers=num_walkers_per_rank,
    num_mcmc_per_measurement=40,
    epsilon_AS=0.0,
    comput_position_deriv=False,
    comput_log_WF_param_deriv=False,
)

num_vmc_steps = 1000
mcmc_prod.run(num_mcmc_steps=num_vmc_steps, max_time=3600)

# get_E performs MPI reduction internally
vmc_E_mean, vmc_E_std, _, _ = mcmc_prod.get_E(
    num_mcmc_warmup_steps=int(num_vmc_steps * 0.1),
    num_mcmc_bin_blocks=10,
)

if rank == 0:
    logger.info(f"VMC Energy: {vmc_E_mean:.5f} +/- {vmc_E_std:.5f} Ha")


# =============================================================================
# 4. LRDMC with multiple alats
# =============================================================================
alats = [0.3, 0.2, 0.1]
lrdmc_results = []

for alat in alats:
    if rank == 0:
        logger.info(f"Starting LRDMC with alat={alat}...")

    lrdmc = GFMC_t(
        hamiltonian_data=optimized_hamiltonian,
        num_walkers=num_walkers_per_rank,
        tau=0.10,  # imaginary time between two branchings (1/Ha)
        num_gfmc_collect_steps=20,
        mcmc_seed=456,
        alat=alat,
        non_local_move="tmove",
        comput_position_deriv=False,
    )

    num_lrdmc_steps = 3000
    lrdmc.run(num_mcmc_steps=num_lrdmc_steps, max_time=3600)

    # get_E performs MPI reduction internally
    lrdmc_E_mean, lrdmc_E_std, _, _ = lrdmc.get_E(
        num_mcmc_warmup_steps=int(num_lrdmc_steps * 0.1),
        num_mcmc_bin_blocks=10,
    )

    if rank == 0:
        logger.info(f"LRDMC Energy (alat={alat}): {lrdmc_E_mean:.5f} +/- {lrdmc_E_std:.5f} Ha")
    lrdmc_results.append((alat, lrdmc_E_mean, lrdmc_E_std))


# =============================================================================
# 5. Extrapolation (Only Rank 0)
# =============================================================================
if rank == 0:
    x = np.array([r[0] ** 2 for r in lrdmc_results])
    y = np.array([r[1] for r in lrdmc_results])
    w = np.array([1.0 / r[2] ** 2 for r in lrdmc_results])

    A = np.vstack([x, np.ones(len(x))]).T
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    c1, c0 = cov @ (A.T @ W @ y)
    # Standard error of the intercept from the weighted least-squares covariance.
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

What differs from the serial script:

* **MPI and JAX initialization.** The script enables 64-bit floats right
  after importing JAX, initializes `mpi4py`, removes the `http_proxy` /
  `https_proxy` variables, and calls
  `jax.distributed.initialize(cluster_detection_method="mpi4py")`. The
  drivers use JAX collectives across ranks; without this call jQMC stops with
  `RuntimeError: MPI/JAX rank-count mismatch` for `-np > 1`. The `jqmc`
  command does both steps itself; a Python script must do them. In our first
  attempt, without removing the proxy variables, the ranks failed to register
  with the JAX coordination service and the run aborted after 5 min with
  `DEADLINE_EXCEEDED`.
* **Walkers.** `num_walkers` in `MCMC` and `GFMC_t` is the number of walkers
  per MPI process; the script sets it to `2000 // size`.
* **Output.** The drivers write their own files (for example
  `hamiltonian_data_opt_step_*.h5`, one per step with `wf_dump_freq=1`) from
  rank 0 only. User-level output in the script (`logger.info`,
  `save_to_hdf5`, the extrapolation) is guarded by `rank == 0`; the logger of
  the other ranks is set to `ERROR`.
* **Averaged hand-off.** After the optimization rank 0 averages the last 20
  dumps (steps 31-50) with `../copy_optimized_hamiltonian.py --average-last 20`
  into `hamiltonian_data_averaged.h5`. The helper runs as a separate non-MPI
  process, so the script removes the MPI launcher variables (`PMI_*`,
  `PMIX_*`, `HYDRA_*`, `OMPI_*`) from its environment. Every rank then reads
  the averaged file, which must be on a file system shared by the ranks.
* **Statistics.** All ranks build the same `Hamiltonian_data` from the TREXIO
  file. `run_optimize` combines the gradients of all ranks, and `get_E`
  returns the global mean and error bar on every rank.

## Step 3: Check the results

From our `run_jqmc_mpi.log` (rank 0 output only):

```text
Running on 125 MPI processes, 16 walkers per process.
...
The number of MPI process = 125.
The number of walkers assigned for each MPI process = 16.
```

Optimization, first and last steps
(`grep -E "^E = |signal-to-noise" run_jqmc_mpi.log`):

```text
Optimization step = 1/50.
E = -16.60316 +- 0.00390 Ha
Max generalized force signal-to-noise = 276.241.
...
Optimization step = 10/50.
E = -17.04067 +- 0.00289 Ha
Max generalized force signal-to-noise = 104.512.
...
Optimization step = 20/50.
E = -17.09022 +- 0.00275 Ha
Max generalized force signal-to-noise = 19.726.
...
Optimization step = 50/50.
E = -17.08653 +- 0.00280 Ha
Max generalized force signal-to-noise = 3.352.
```

As in `jqmc-example12`, the energy is flat after about 20 steps and the S/N of
the largest generalized force stays between about 2 and 7 from step 30 on.

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

VMC, LRDMC and the extrapolation:

```text
VMC Energy: -17.08735 +/- 0.00124 Ha
...
LRDMC Energy (alat=0.3): -17.26274 +/- 0.00112 Ha
...
LRDMC Energy (alat=0.2): -17.24785 +/- 0.00105 Ha
...
LRDMC Energy (alat=0.1): -17.23903 +/- 0.00105 Ha
----------------------------------------
Extrapolation Results (E vs a^2):
alat=0.30, a^2=0.0900, E=-17.26274 +/- 0.00112
alat=0.20, a^2=0.0400, E=-17.24785 +/- 0.00105
alat=0.10, a^2=0.0100, E=-17.23903 +/- 0.00105
Extrapolated Energy (a->0): -17.23604 +/- 0.00105 Ha
Fit quality: chi^2 = 0.00 for 1 degree(s) of freedom
----------------------------------------
```

As in `jqmc-example12`, every branching step covers the same imaginary time
`tau = 0.10`; the average number of projections per branching step grows
from 27 (`alat = 0.3`) to 60 (0.2) and 240 (0.1), and so does the cost
(40, 81 and 303 ms of projection per branching step in our run).

Comparison with the serial run of `jqmc-example12` (same settings, different
random streams):

| | `jqmc-example12` (1 process) | `jqmc-example13` (125 MPI processes) |
|---|---|---|
| VMC | -17.09003(122) | -17.08735(124) |
| LRDMC `alat` = 0.3 | -17.26532(110) | -17.26274(112) |
| LRDMC `alat` = 0.2 | -17.24753(76) | -17.24785(105) |
| LRDMC `alat` = 0.1 | -17.24187(95) | -17.23903(105) |
| Extrapolated | -17.23730(93) | -17.23604(105) |
| $\chi^2$ of the $a^2$ fit | 8.92 | 0.00 |
| Wall time (whole script) | 61 min | 32 min |

The extrapolated energies agree within one combined error bar; the individual
LRDMC points differ by up to 2.8 mHa (2.0 combined error bars, at
`alat = 0.1`). The extrapolated LRDMC energy lies about 0.15 Ha below the VMC
energy, and the LRDMC energy rises as `alat` decreases, as in
`jqmc-example12`. The very different $\chi^2$ values of the two runs show
that, at this precision, a three-point $a^2$ fit is only a rough check (see
`jqmc-example12`). The MPI run is faster mainly in the optimization (8 min
against 34 min for the serial run); the LRDMC runs took 23 min in total
against 26 min for the serial run.
