# jqmc-example07: Silicon (primitive cell) from Quantum ESPRESSO, pure B-spline basis, J_DTN without cusp

Total energy of bulk **silicon** at the Gamma point. This tutorial shows how to
take a Quantum ESPRESSO (QE) PW LDA wavefunction, convert it into a jQMC
Hamiltonian on a **pure B-spline single-particle basis**, attach a DTN
polynomial Jastrow (no e-N cusp), optimise the Jastrow only, and then compute
VMC / LRDMC total energies.

## System

- 2-atom diamond-structure silicon primitive cell (FCC; `a_cubic = 5.381 Ang`,
  a test geometry, not the equilibrium lattice constant).
- ccECP pseudopotential on Si (4 valence electrons / atom = 8 electrons / cell).
- LDA ([Perdew-Zunger 1981](https://doi.org/10.1103/PhysRevB.23.5048)) from Quantum ESPRESSO, Gamma point only.

## Why this example exists

This example exercises the **pure B-spline** branch of the QE -> jQMC
converter (`--kind bspline`):

- **Fast local evaluation.** Evaluating a single-particle orbital at one
  electron position costs a few local B-spline blends instead of a sum over
  all plane waves.
- **Controlled interpolation, not an identity.** The spline reproduces the
  sampled QE orbital on its construction mesh, but a finite mesh and finite
  spline order introduce interpolation error between knots. Values,
  gradients, Laplacians, forces, and stress therefore belong to the spline
  approximant, not exactly to the original plane-wave determinant. Converge
  the mesh before treating the converted wavefunction as equivalent.

The Jastrow is the **DTN polynomial Jastrow** (`u_chi_f` preset):

- channel `u`   -- e-e two-body         (J2 analogue)
- channel `chi` -- e-N one-body         (J1 analogue)
- channel `f`   -- factorised e-e-N     (J3 analogue)

Because silicon is treated with a non-local ECP, there is **no all-electron
e-N cusp to enforce**. The converter derives that fact from the ECP records
and constrains the DTN chi slope to zero automatically.

The Slater determinant from QE LDA stays **frozen** -- we only optimise the
Jastrow parameters. This is the cheapest non-trivial QMC refinement of a
DFT wavefunction.

## Setup

All paths below are relative to `examples/jqmc-example07`; follow the displayed `cd`
commands. Keep the supplied scripts and TOML files; do not generate default
TOMLs over them. Where serial CPU, MPI CPU and MPI GPU launch lines are shown,
they are alternatives: choose one.

The outputs shown below come from one reference run of every step: 128 MPI
ranks on one 128-core CPU node (CPU only, `OMP_NUM_THREADS=1`). Wall times on
that node:

| Step | Wall time (128 ranks) |
|---|---|
| 1. QE `pw.x` (1 rank) | 4 s |
| 2. `jqmc-tool qe --kind bspline` | 2 s |
| 3. VMC optimisation, 200 steps | 80 min |
| 4. MCMC, 5000 steps | 1 min |
| 5. LRDMC `lrdmc-tau`, 5000 branchings | 7 min |
| 6. Mixed-precision MPC MCMC, 5000 steps | 3 min (23 s of it is the one-time MPC table setup) |

QMC numbers are stochastic: with a different number of ranks, walkers or
seeds you will get statistically compatible but not identical values.

## 1. Generate a trial WF (Quantum ESPRESSO)

The QE run that produced the trial WF used here is in `01DFT/`:

- `01DFT/pw.in`, `01DFT/pw.out` -- QE input and output (2-Si primitive cell,
  ccECP, LDA-PZ, `ecutwfc = 100 Ry`, Gamma point, 8 bands).
- `01DFT/silicon.save/` -- QE output bundle (`data-file-schema.xml`,
  `wfc1.hdf5`).
- The ccECP pseudopotential `Si.ccECP.upf` is not shipped. Download it from
  <https://pseudopotentiallibrary.org/recipes/Si/ccECP/Si.ccECP.upf> and place it in `01DFT/` (read by `pw.x`) and in
  `01DFT/silicon.save/` (the converter reads the UPF named in
  `data-file-schema.xml` from the save directory):

  ```bash
  curl -L -o 01DFT/Si.ccECP.upf https://pseudopotentiallibrary.org/recipes/Si/ccECP/Si.ccECP.upf
  cp 01DFT/Si.ccECP.upf 01DFT/silicon.save/
  ```

  `jqmc-example09`, `jqmc-example10`, `jqmc-workflow-example04` and
  `jqmc-workflow-example06` use the same file from `01DFT/`.

These QE outputs are shipped, so you can skip `pw.x` and go to step 2. To
regenerate them you need a `pw.x` built with HDF5 support (the converter
reads `wfc1.hdf5`); the run overwrites `01DFT/silicon.save/` and `pw.out`:

```bash
cd 01DFT
pw.x -in pw.in > pw.out          # or: mpirun -np N pw.x -in pw.in > pw.out
cd ..
```

The shipped files come from QE 7.5 on one MPI rank (4 s wall time):

```text
          k = 0.0000 0.0000 0.0000 (  2239 PWs)   bands (ev):

    -4.9848   7.6750   7.6750   7.6750   9.7498   9.7498   9.7498  11.2485
...
!    total energy              =     -14.35365371 Ry
...
     convergence has been achieved in  11 iterations
```

i.e. -7.17683 Ha per 2-Si primitive cell (ccECP, LDA-PZ, Gamma point only).
A rerun on 16 ranks gave the same total energy to all printed digits.

## 2. Convert QE -> jQMC (pure B-spline basis + DTN Jastrow)

```bash
cd 02vmc_JSD
jqmc-tool qe ../01DFT/silicon.save \
      --kind bspline \
      --output hamiltonian_data.h5 \
      --jastrow-dtn u_chi_f
```

```text
Auto-discovered 1 UPF entries from ../01DFT/silicon.save/data-file-schema.xml: Si=../01DFT/silicon.save/Si.ccECP.upf
Loaded QE reference from ../01DFT/silicon.save: 2 atoms, 8 bands, 4477 G-vectors, ecutwfc=50.000 Ha.

PW->B-spline: order 5, grid (36, 36, 36) from QE FFT grid (48, 48, 48) (mesh_factor 0.750), 4 bands, control-point table 0.003 GiB (complex128), mean residual 3.945e-12, max 5.894e-12.

build_wavefunction_from_projection(kind=bspline): 4 AOs, 4 MOs, ne_up=4, ne_dn=4.
build_hamiltonian_from_wavefunction: pbc=True, ne_up=4, ne_dn=4, j1=off, j2=off.
build_hamiltonian_from_wavefunction: DTN polynomial Jastrow attached (use_u=True, use_chi=True, use_f=True, C=4).

Hamiltonian data is saved in hamiltonian_data.h5.
```

What each flag does:

- `--kind bspline` selects the **pure B-spline** representation. Only the 4
  occupied bands are splined. The defaults are `--mesh-factor 0.75`
  (spline grid = 0.75 x the QE FFT grid, here 36^3) and `--spline-order 5`.
  The tiny residual above only checks the spline at its own nodes; converge
  energy and variance against the mesh (`--mesh-factor`) before trusting it.
- `--output hamiltonian_data.h5` writes the jQMC Hamiltonian HDF5 under the
  name used by every downstream TOML (`hamiltonian_h5 = "hamiltonian_data.h5"`).
- `--jastrow-dtn u_chi_f` attaches the full DTN polynomial preset
  (e-e + e-N + factorised e-e-N). Use `u_chi` to drop the f-channel,
  or `u` for e-e only.
- We do **not** pass any `-j1` / `-j2` / `-j3` flag -- those build the
  J1 + J2 + J3 Jastrow family used in examples 01-05, which is mutually
  exclusive with the DTN preset. The CLI refuses if you mix them.

You can inspect the resulting Hamiltonian with:

```bash
jqmc-tool hamiltonian show-info hamiltonian_data.h5
```

```text
...
**SD_data
  occupied MO rows (up, dn) = (4, 4)
  num_electron_up = 4
  num_electron_dn = 4
  time_reversal = True
**MOs_data
  Number of MOs = 4
  dim. of MOs coeff = (4, 4)
**AOs_bspline_data
  Number of AOs = 4
  spline grid shape (N0, N1, N2) = (36, 36, 36)
  spline order (degree) = 5
  control points per AO = 46656
  total stored coefficients (num_ao * cp_per_ao) = 186624
  complex (k != 0) path = False
...
**Jastrow_DTN_data
  terms: use_u=True, use_chi=True, use_f=True; electron-nucleus cusp constrained; C=4
  orders: N_u=8, N_chi=8, N_eN=4, N_ee=4
  cutoffs: L_u=2.7887, L_chi=[2.7886553386790753], L_f=[1.3943276693395377]
  cutoff ratios: u=0.9500, chi=[0.95], f=[0.95] (comoving volumetric stress policy)
  indep coefs: alpha=(2, 8), beta=(1, 8), gamma=(1, 57)
==================================================================
```

The `MOs_data` / `AOs_bspline_data` pair is printed twice, for the up- and
the down-spin orbitals (identical here); the second copy is elided above.

### The other QE orbital representation: pGTO+PW

The same `01DFT/silicon.save` can instead be converted to localized periodic
Gaussians plus a residual plane-wave component:

```bash
jqmc-tool qe ../01DFT/silicon.save \
      --kind pgto+pw --ecutwfc auto \
      --output hamiltonian_data.h5 \
      --jastrow-dtn u_chi_f
```

On this save it selects a residual PW cutoff of 7.5 Ry and builds 147 AOs
(58 pGTO + 89 PW):

```text
Auto pgto+pw cutoff: target=2.00e-05 MET, selected g_cut=2.739 Bohr^-1 (ecutwfc=7.500 Ry), n_pw=89, mean residual=2.282e-06.

build_wavefunction_from_projection(kind=pgto+pw): 147 AOs, 4 MOs, ne_up=4, ne_dn=4.
```

Choose **one** representation per directory: do not overwrite a Hamiltonian
beside existing optimization or restart files. This tutorial continues with
the B-spline Hamiltonian only. For a complete **pGTO+PW force** calculation,
follow `jqmc-example08`.

## 3. Optimise the Jastrow (VMC)

The VMC input `02vmc_JSD/vmc.toml` is set up so that **only DTN Jastrow
channels are optimised** (81 parameters); everything else (Slater
determinant, J1 / J2 / J3, lambda basis exponents / coefficients, DTN
cutoffs) is frozen. Each optimization step samples 500 measurements per
walker, discards the first 30 and estimates errors from 10 bins (the minimum
warmup and bin counts the code accepts).

<!-- include: 02vmc_JSD/vmc.toml -->
```toml
[control]
job_type = "vmc"          # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456         # Random seed for MCMC.
number_of_walkers = 4     # Number of walkers per MPI process.
max_time = 86400          # Maximum time in sec.
restart = false
restart_chk = "restart.h5"
hamiltonian_h5 = "hamiltonian_data.h5"
verbosity = "high"

[vmc]
num_mcmc_steps = 500
num_mcmc_per_measurement = 40
num_mcmc_warmup_steps = 30       # Initial measurements discarded per optimization step (code minimum: 30).
num_mcmc_bin_blocks = 10         # Bins per rank and walker for the error estimate (code minimum: 10).
Dt = 1.5                  # MCMC step size (bohr).
epsilon_AS = 0.0
num_opt_steps = 200
wf_dump_freq = 1
optimizer_kwargs = { method = "sr", delta = 0.15, epsilon = 0.001, cg_flag = true, cg_max_iter = 10000, use_lm = true, lm_subspace_dim = 0, lm_cond = 0.001 }

# --- Jastrow-only optimisation -------------------------------------------------
# J1 / J2 / J3 are all OFF: this wavefunction was built with the DTN
# polynomial Jastrow preset (--jastrow-dtn u_chi_f), which is mutually exclusive
# with the J1 + J2 + J3 family.
opt_J1_param = false
opt_J2_param = false
opt_J3_param = false

# DTN polynomial channels (this is the active Jastrow):
#   u   : e-e channel        (J2 analogue)
#   chi : e-N channel        (J1 analogue)
#   f   : factorised e-e-N   (J3 analogue)
opt_JDTN_u_param = true
opt_JDTN_chi_param = true
opt_JDTN_f_param = true

# --- Slater determinant FROZEN -------------------------------------------------
# QE -> jQMC handed us a B-spline determinant; we keep it frozen and only let
# the Jastrow relax.
opt_geminal = false
fix_geminal_rank = 0
opt_J3_basis_exp = false
opt_J3_basis_coeff = false
opt_lambda_basis_exp = false
opt_lambda_basis_coeff = false
```

Launch the VMC optimisation. You should still be in `02vmc_JSD/`:

```bash
jqmc vmc.toml > out_vmc 2> out_vmc.e               # serial CPU
mpirun -np 128 jqmc vmc.toml > out_vmc 2> out_vmc.e  # CPU MPI
mpiexec -n 4 -map-by ppr:4:node jqmc vmc.toml \
      > out_vmc 2> out_vmc.e                         # GPU MPI
```

With 128 ranks one optimization step took about 24 s (80 min for 200
steps). Monitor the optimisation with:

```bash
jqmc-tool vmc analyze-output out_vmc
jqmc-tool vmc analyze-output out_vmc -p -s vmc.jpg
```

```text
------------------------------------------------------
Iter     E (Ha)     Max generalized force (Ha/a.u.)  Max force S/N
------------------------------------------------------
   1  -7.0652(26)  -6.498(28)   229.056
   2  -7.1127(23)  -5.584(24)   237.377
   3  -7.1566(21)  -4.714(19)   242.426
...
  50  -7.3290(12)  -0.0950(70)    16.118
...
 100  -7.3294(13)  -0.0600(70)     9.154
...
 196  -7.3296(13)  -0.0260(70)     3.563
 197  -7.3293(13)  -0.0300(70)     4.841
 198  -7.3297(13)  -0.0170(50)     3.519
 199  -7.3284(13)  -0.0210(70)     2.833
 200  -7.3298(13)  -0.0180(50)     3.906
------------------------------------------------------
```

The energy drops by about 0.26 Ha in the first ~50 steps and then stays flat
within its error bar, while the maximum generalized force keeps shrinking.
By the last ~20 steps `Max force S/N` is about 3-5, i.e. the remaining
forces are near the noise level. If your run does not get there, continue it
with `restart = true` or adjust `delta` in `optimizer_kwargs` (see
`jqmc-example01`).

Optimization writes `02vmc_JSD/hamiltonian_data_opt_step_N.h5`; it does not
overwrite the converter's input. After checking convergence, use the handoff
helper below to select an actual saved dump.

## 4. Compute the VMC energy (MCMC sampling)

```bash
cd ..
cd 03mcmc_JSD
python ../../copy_optimized_hamiltonian.py --average-last 20 ../02vmc_JSD hamiltonian_data.h5
```

> [!NOTE]
> The helper averages the wave-function parameters of the last 20 dumps (here steps 181-200) with
> `jqmc-tool vmc average-wf`; see [Averaging the last optimization steps](../../doc/notes/wavefunction_optimization.md). The numbers quoted below were not sampled from this average: the reference run handed over the single last dump with
> `python ../../copy_optimized_hamiltonian.py ../02vmc_JSD hamiltonian_data.h5`, which printed
> `Copied optimized trial: ../02vmc_JSD/hamiltonian_data_opt_step_200.h5 -> hamiltonian_data.h5`.

<!-- include: 03mcmc_JSD/mcmc.toml -->
```toml
[control]
job_type = "mcmc"         # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456         # Random seed for MCMC.
number_of_walkers = 4     # Number of walkers per MPI process.
max_time = 86400          # Maximum time in sec.
restart = false
restart_chk = "restart.h5"
hamiltonian_h5 = "hamiltonian_data.h5"
verbosity = "high"

[mcmc]
num_mcmc_steps = 5000
num_mcmc_per_measurement = 40
num_mcmc_warmup_steps = 30
num_mcmc_bin_blocks = 10
Dt = 1.5
epsilon_AS = 0.0
atomic_force = false
compute_stress = false
use_swct = true
parameter_derivatives = false
```

`atomic_force = false` and `compute_stress = false`: this tutorial does
**not** compute forces or stress. See `jqmc-example08` for force-only
hydrogen and `jqmc-example09` for stress (with forces) on twist-averaged
silicon.

```bash
mpirun -np 128 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e
jqmc-tool mcmc compute-energy restart.h5 -b 10 -w 30
```

End of `out_mcmc` (128 ranks, 57 s):

```text
Final output(s):
  Total Energy: E = -7.32927 +- 0.00040 Ha.
  Variance: Var = 0.31764 +- 0.00054 Ha^2.
```

`compute-energy` recomputes the energy from the checkpoint with the same
warmup (`-w 30`) and bin count (`-b 10`) as `mcmc.toml`; use it to re-bin
without rerunning:

```text
Read restart checkpoint file(s) from restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 5120
E = -7.3292702927407944 +- 0.0003965920698677399 Ha.
```

This is the VMC energy per 2-Si cell at Gamma (Ewald interaction, no
finite-size correction).

## 5. Compute the LRDMC energy

LRDMC uses the `lrdmc-tau` driver (GFMC_t): the walkers are propagated for a
fixed imaginary time `tau` (Ha^-1, atomic units of imaginary time) between
branchings. This tutorial uses `tau = 0.10` and `alat = 0.30`. Run it in its
own subdirectory, so that a later scan over `alat` does not overwrite the
checkpoint:

```bash
cd ..
cd 04lrdmc_JSD
mkdir -p alat_0.30
cp ../03mcmc_JSD/hamiltonian_data.h5 alat_0.30/
cp lrdmc_tau.toml alat_0.30/
```

<!-- include: 04lrdmc_JSD/lrdmc_tau.toml -->
```toml
[control]
job_type = "lrdmc-tau"    # Imaginary-time (GFMC_t) flavour.
mcmc_seed = 34468
number_of_walkers = 4
max_time = 86400
restart = false
restart_chk = "restart.h5"
hamiltonian_h5 = "hamiltonian_data.h5"
verbosity = "high"

[lrdmc-tau]
num_mcmc_steps = 5000
tau = 0.10                # Imaginary-time step (Ha^-1, atomic units of imaginary time).
alat = 0.30               # Lattice discretisation grid (bohr).
non_local_move = "tmove"
num_gfmc_warmup_steps = 50
num_gfmc_bin_blocks = 50
num_gfmc_collect_steps = 20
atomic_force = false
compute_stress = false
use_swct = false
epsilon_PW = 0.0
```


```bash
cd alat_0.30
mpirun -np 128 jqmc lrdmc_tau.toml > out_lrdmc_tau 2> out_lrdmc_tau.e
jqmc-tool lrdmc compute-energy restart.h5 -b 50 -w 50 -c 20
```

End of `out_lrdmc_tau` (128 ranks, 7 min) and the `compute-energy` result:

```text
Survived walkers ratio = 98.33 %
Average of the number of projections  = 27
...
Final output(s):
  Total Energy: E = -7.38236 +- 0.00070 Ha.
  Variance: Var = 0.30771 +- 0.00062 Ha^2.
```

```text
Read restart checkpoint file(s) from restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 50
E = -7.382363176890383 +- 0.0006970338616170723 Ha.
```

`lrdmc compute-energy` uses the same warmup (`-w 50`), bin count (`-b 50`)
and weight-collection length (`-c 20`) as the TOML. `Average of the number
of projections` is the mean number of lattice moves per `tau = 0.10`.

Summary of the reference run (Ha per 2-Si cell, Gamma point, Ewald):

| Method | Energy (Ha) | Variance (Ha^2) |
|---|---|---|
| VMC (step 4) | -7.32927(40) | 0.31764(54) |
| LRDMC (`alat = 0.30`) | -7.38236(70) | 0.30771(62) |

LRDMC lies about 53 mHa below VMC.

How the reported energy is measured: the walkers are propagated with a
lattice-regularized Hamiltonian, but the energy that is printed is the
ordinary (continuum) local energy averaged over those walkers. This removes
some lattice-only terms from the reported number, but the walker distribution
itself still depends on `alat`, so the energy keeps an O(`alat^2`) bias. See
the section "Propagation and measurement local energies (`novar=0`)" of the
technical note `doc/notes/lrdmc.md`.

For a publication-grade total energy, copy `lrdmc_tau.toml` and the trial
into `alat_0.10/`, `alat_0.15/`, `alat_0.20/`, `alat_0.25/` subdirectories as
well, change only `alat` in each, and extrapolate with
`jqmc-tool lrdmc extrapolate-energy` (see `jqmc-example01` for the full
extrapolation recipe and `jqmc-example06` for a copy loop). Smaller `alat`
needs more lattice moves per unit of imaginary time and therefore costs
considerably more.

> [!NOTE]
> `non_local_move` selects the treatment of the ECP non-local channels in
> jQMC's GFMC drivers; this tutorial sets T-move (`"tmove"`, also the
> default). The determinant-locality T-move (`"dltmove"`) is the alternative.

## 6. Mixed-precision MPC energy recipe

The following variant uses the same optimized, single-Gamma Hamiltonian but
turns on two production features: mixed precision and the passive Model
Periodic Coulomb (MPC) energy correction.

Mixed precision (`[precision] mode = "mixed"`) is an opt-in speed setting;
the default `full` (all float64) used in steps 3-5 is the reference. Compare
the mixed-precision energy with the step-4 value before relying on it, and see
the technical note `doc/notes/mixed_precision.md`.

```bash
cd ../..
cd 05mcmc_JSD_mixed_mpc
python ../../copy_optimized_hamiltonian.py --average-last 20 ../02vmc_JSD hamiltonian_data.h5
mpirun -np 128 jqmc mcmc.toml > out_mcmc_mpc 2> out_mcmc_mpc.e
jqmc-tool mcmc compute-energy restart.h5 -b 10 -w 30
```

> [!NOTE]
> The helper averages the wave-function parameters of the last 20 dumps (here steps 181-200) with
> `jqmc-tool vmc average-wf`; see [Averaging the last optimization steps](../../doc/notes/wavefunction_optimization.md). The numbers quoted below were not sampled from this average: the reference run handed over the single last dump with
> `python ../../copy_optimized_hamiltonian.py ../02vmc_JSD hamiltonian_data.h5`.

<!-- include: 05mcmc_JSD_mixed_mpc/mcmc.toml -->
```toml
[control]
job_type = "mcmc"
mcmc_seed = 34478
number_of_walkers = 4
max_time = 86400
restart = false
restart_chk = "restart.h5"
hamiltonian_h5 = "hamiltonian_data.h5"
verbosity = "high"

[mcmc]
num_mcmc_steps = 5000
num_mcmc_per_measurement = 40
num_mcmc_warmup_steps = 30
num_mcmc_bin_blocks = 10
Dt = 1.5
epsilon_AS = 0.0
atomic_force = false
compute_stress = false
use_swct = true
parameter_derivatives = false

[coulomb]
mpc = true
# Convergence parameter in Hartree; repeat with larger values.
mpc_cutoff = 30.0

[precision]
mode = "mixed"
```

The log confirms the mixed-precision zones (`ao_eval` and the Jastrow
evaluation in float32, everything else float64):

```text
Precision: Mixed Precision (FP32 + FP64)
Zone detail:
  ao_eval: float32
  ao_grad_lap: float64
...
```

Before sampling starts, rank 0 builds the MPC tables and broadcasts them to
the other ranks; the log reports each stage:

```text
MPC setup: building the tables on rank 0 and broadcasting them to 128 ranks.
MPC setup: cutoff = 30.0 Ha, grid = (64, 64, 64), n_max = 3; g_G is extrapolated from the N, 2N and 4N grids.
  MPC setup: g_G on the 1N grid (64, 64, 64) done in 1.4 sec.
  MPC setup: g_G on the 2N grid (128, 128, 128) done in 3.0 sec.
  MPC setup: g_G on the 4N grid (256, 256, 256) done in 14.0 sec.
  MPC setup: trial density on the (64, 64, 64) grid done in 3.0 sec.
MPC setup: done in 23.3 sec.
```

In the reference run the whole job took 3 min; the 5000 MCMC steps
themselves took 95 s. End of `out_mcmc_mpc`:

```text
Final output(s):
  Total Energy: E = -7.32971 +- 0.00040 Ha.
  Variance: Var = 0.31777 +- 0.00043 Ha^2.
  MPC correction: dE_MPC = +0.15725 +- 0.00011 Ha (cutoff = 30.0 Ha).
  Total Energy (MPC-corrected): E_MPC = -7.17246 +- 0.00046 Ha.
```

and `compute-energy` from the checkpoint:

```text
Read restart checkpoint file(s) from restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 5120
E = -7.32971212970956 +- 0.00039500091424805135 Ha.
MPC correction: dE_MPC = 0.1572482284955083 +- 0.00010566904218679545 Ha.
Total Energy (MPC-corrected): E_MPC = -7.172463901214052 +- 0.00045765355075302357 Ha.
```

The mixed-precision energy, -7.32971(40) Ha, agrees with the full-precision
step-4 energy, -7.32927(40) Ha, within the combined error bar.

`E` is the energy with the Ewald electron-electron interaction; `E_MPC`
replaces it by the model periodic Coulomb interaction, which reduces the
finite-size error. For this 8-electron cell the correction is large
(+0.157 Ha), a reminder that a single 2-atom cell at Gamma is far from the
thermodynamic limit. MPC is passive: sampling, branching, forces, stress and
parameter gradients remain on the Ewald Hamiltonian. `mpc_cutoff` is a
numerical convergence parameter in Hartree, so repeat the run with a larger
cutoff until the MPC correction is stable.

## Forces / stress

The main VMC/LRDMC path deliberately keeps the scope minimal: **no forces, no
stress**. The optional section above adds mixed-precision MPC energy only.
It is not a force-equivalence reference between the QE plane-wave orbital and
the spline approximant. The force machinery is exercised in:

- `jqmc-example08` -- hydrogen (PBC) forces.
- `jqmc-example09` -- silicon stress / pressure (twist-averaged).

## Orbital tiling

The shipped Gamma-only primitive input describes this tutorial's cell.
A nontrivial Gamma-supercell tiling requires multiple primitive k points;
see the [tiling recipe](../../doc/examples.md#jqmc-example14-frozen-orbitals-in-a-tiled-supercell) for the source mesh and
frozen-node options. Its larger physical cell has different reference data.
