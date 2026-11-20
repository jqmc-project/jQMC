# jqmc-example08: Periodic all-electron hydrogen from QE pGTO+PW with atomic forces

This tutorial walks through the **forces workflow** for a fully periodic,
all-electron (AE) system: a dense packed-H2 cell solved with Quantum
ESPRESSO LDA, converted to jQMC with the **pGTO + PW mixed basis**, then
optimized with VMC and evaluated with MCMC and LRDMC with **atomic forces
turned on**.

All commands below are run inside `examples/jqmc-example08`; follow the
displayed `cd` commands. Keep the supplied TOML files; do not generate default
TOMLs over them.

## Reference run

Every number and log excerpt in this README comes from one run of the
shipped inputs on one 128-core CPU node (no GPU), except that the MCMC and
LRDMC steps sampled the single last optimization dump instead of the averaged
trial (see the notes at the two hand-offs):

| Step | Launch | Wall time |
|------|--------|-----------|
| 1 QE `pw.x` (QE 7.5, HDF5) | 32 MPI ranks | 21 s |
| 2 `jqmc-tool qe` conversion | serial | 26 s |
| 3 VMC optimization (200 steps) | 128 MPI ranks | 81 min |
| 4 MCMC with forces | 128 MPI ranks | 2.5 min (147 s sampling) |
| 5 LRDMC (GFMC_t) with forces | 128 MPI ranks | 6.5 min (388 s branching) |

`number_of_walkers = 1`, so the total sample count scales with the number of
MPI ranks. With fewer ranks, expect proportionally larger error bars (or
increase `num_mcmc_steps`).

## Why this example?

- **All-electron 1/r Coulomb** for hydrogen (`H.coulomb-ae.UPF` is a true
  1/r potential, `Z_valence = 1`). Neither plane waves nor Gaussians
  reproduce the electron-nucleus cusp exactly, so it is put back by the
  Jastrow: the **DTN polynomial Jastrow** (`--jastrow-dtn u_chi`) enforces
  the Kato e-N cusp in its `chi` channel.
- **General twist** `k = (0.25, 0.25, 0.25)` (crystal coordinates). This is
  not a Gamma-only run; the MOs are complex.
- **pGTO + PW mixed basis** with `--ecutwfc auto`: periodic Gaussians resolve
  the nuclear-localized orbital structure and a small plane-wave block keeps
  the residual. The pGTO block gives smooth analytic derivatives with respect
  to nuclear positions, which is why this basis is the one to use for forces.
- **Optimization scope**: DTN Jastrow + **MO coefficients** (`opt_geminal = true`).

> [!NOTE]
> For the non-force, Jastrow-only counterpart see `jqmc-example07`
> (silicon, ECP + B-spline). `jqmc-workflow-example05` drives the same
> system through the `jqmc_workflow` API.

> [!NOTE]
> Stress is not computed here. Setting `compute_stress = true` in `[mcmc]` /
> `[lrdmc-tau]` adds the stress tensor and pressure; it is independent of
> `atomic_force`. See the technical note `doc/notes/derivatives.md`.

## Directory layout

```text
jqmc-example08/
  README.md
  01DFT/                       <- QE input, UPF and the reference QE log
    h_scf.in
    h_scf.out
    H.coulomb-ae.UPF           <- downloaded, not shipped (see Step 1)
    out/h_k025.save/           <- written by pw.x (not shipped)
  02vmc_JSD_geminal/           <- conversion + VMC opt of Jastrow + MOs
    vmc.toml
  03mcmc_JSD_force/            <- MCMC with atomic forces
    mcmc.toml
  04lrdmc_JSD_force/           <- LRDMC (GFMC_t) with atomic forces
    lrdmc_tau.toml
```

## System

Dense packed H2 in a simple-orthorhombic cell:

- `ibrav = 8`, `a = 5.00`, `b = 6.00`, `c = 7.00` bohr (V = 210 bohr^3).
- 4 H atoms (two H2 molecules, bond length 1.4 bohr), 4 electrons
  (`nspin = 1`). The geometry is not relaxed, so the atomic forces are not zero.
- All-electron 1/r Coulomb potential (`H.coulomb-ae.UPF`).
- One explicit k-point at crystal `(1/4, 1/4, 1/4)` (`nosym = .true.`,
  `noinv = .true.` keep it from being folded into a star).
- `ecutwfc = 800 Ry`, `ecutrho = 3200 Ry`, `input_dft = 'pz'` (LDA-PZ).

## Step 1: DFT with Quantum ESPRESSO (`01DFT/`)

The QE wavefunction and density files are not shipped; run `pw.x` (built with
HDF5 support, so that it writes `wfc1.hdf5`) to create them. The
pseudopotential `H.coulomb-ae.UPF` is not shipped either; download it from
<https://pseudopotentials.quantum-espresso.org/upf_files/H.coulomb-ae.UPF> into `01DFT/` first:

```bash
cd 01DFT
curl -L -O https://pseudopotentials.quantum-espresso.org/upf_files/H.coulomb-ae.UPF
mpirun -np 32 pw.x -in h_scf.in > h_scf.out
cd ..
```

This writes `01DFT/out/h_k025.save/` (`data-file-schema.xml`, `wfc1.hdf5`,
`charge-density.hdf5`, `H.coulomb-ae.UPF`), which the conversion reads.
Key lines of `h_scf.out` from the reference run:

```text
     IMPORTANT: XC functional enforced from input :
     Exchange-correlation= PZ
...
     highest occupied, lowest unoccupied level (ev):    -5.3611    2.4430

!    total energy              =      -4.48559671 Ry
...
     convergence has been achieved in   8 iterations
...
     PWSCF        :      1.76s CPU     21.15s WALL
```

The LDA total energy, -2.2428 Ha per cell, is only a mean-field reference; it
is not a bound for the QMC energies below.

## Step 2: QE -> jQMC conversion (pGTO + PW, auto cutoff)

```bash
cd 02vmc_JSD_geminal
jqmc-tool qe \
      --kind pgto+pw \
      --ecutwfc auto \
      --exp-tail-eps 1e-2 \
      --jastrow-dtn u_chi \
      --output hamiltonian_data.h5 \
      ../01DFT/out/h_k025.save
```

- `--kind pgto+pw`: periodic Gaussians plus a plane-wave remainder.
- `--ecutwfc auto`: the converter sweeps the PW handoff cutoff, optimizes the
  pGTO exponents, and stops when the mean projection residual meets its
  built-in target (2e-5).
- `--exp-tail-eps 1e-2`: bounds every optimized pGTO primitive so that at most
  1% of its momentum-space norm lies outside the QE wavefunction sphere. This
  prevents the exponent optimizer from moving into a high-momentum region
  that the spilling objective cannot see. This is the CLI default; it is
  written explicitly here because this all-electron, high-cutoff example
  optimizes pGTO exponents. Pass `--exp-tail-eps 0` to disable the bound.
- `--jastrow-dtn u_chi`: DTN polynomial Jastrow with the e-e (`u`) and e-N
  (`chi`) terms, no `f` term. DTN is mutually exclusive with J1/J2/J3. The AE
  Kato cusp is imposed in `chi` automatically for the bare-Coulomb UPF.
- The UPF mapping is auto-discovered from `data-file-schema.xml`, so no
  `--upf` is needed.

Output (abridged):

```text
Auto-discovered 1 UPF entries from ../01DFT/out/h_k025.save/data-file-schema.xml: H=../01DFT/out/h_k025.save/H.coulomb-ae.UPF
Loaded QE reference from ../01DFT/out/h_k025.save: 4 atoms, 6 bands, 80267 G-vectors, ecutwfc=400.000 Ha.
...
pGTO exponent optimisation: scipy_success=True, accepted=True, nit=31; autograd spilling 3.887156e-05 -> 8.597459e-06, deployed spilling 3.887147e-05 -> 8.597297e-06, search full cond(S)=9.990e+07, deployed full cond(S)=9.990e+07.
...
Auto pgto+pw cutoff: target=2.00e-05 MET, selected g_cut=2.739 Bohr^-1 (ecutwfc=7.500 Ry), n_pw=72, mean residual=8.597e-06.

QE -> jQMC: non-Gamma twist loaded, xk (1/bohr Cartesian) = [0.3141592653589793, 0.2617993877991494, 0.2243994752564138], k_point (fractional reciprocal-lattice) = (0.25, 0.25, 0.25)
build_wavefunction_from_projection(kind=pgto+pw): 104 AOs, 2 MOs, ne_up=2, ne_dn=2.
build_hamiltonian_from_wavefunction: pbc=True, ne_up=2, ne_dn=2, j1=off, j2=off.
build_hamiltonian_from_wavefunction: DTN polynomial Jastrow attached (use_u=True, use_chi=True, use_f=False, C=4).

Hamiltonian data is saved in hamiltonian_data.h5.
```

So each MO is expanded in 32 pGTOs (8 per H) plus 72 plane waves.
Inspect the result with `jqmc-tool hamiltonian show-info hamiltonian_data.h5`;
its last block summarizes the Jastrow:

```text
**Jastrow_DTN_data
  terms: use_u=True, use_chi=True, use_f=False; electron-nucleus cusp constrained; C=4
  orders: N_u=8, N_chi=8, N_eN=4, N_ee=4
  cutoffs: L_u=2.3750, L_chi=[2.375], L_f=[1.1875]
  ...
```

## Step 3: VMC optimization of Jastrow + MOs (`02vmc_JSD_geminal/`)

Still in `02vmc_JSD_geminal/`:

```bash
mpirun -np 128 jqmc vmc.toml > out_vmc 2> out_vmc.e
```

<!-- include: 02vmc_JSD_geminal/vmc.toml -->
```toml
[control]
job_type = "vmc" # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 1 # Number of walkers per MPI process; add MPI ranks for more samples.
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "high" # Verbosity level. "low", "high", "devel", "mpi-low", "mpi-high", "mpi-devel"

[vmc]
num_mcmc_steps = 300 # Number of observable measurement steps per MPI and Walker.
num_mcmc_per_measurement = 40 # Number of MCMC updates per measurement.
num_mcmc_warmup_steps = 30 # Number of initial observable measurements discarded as warmup.
num_mcmc_bin_blocks = 10 # Number of blocks for binning per MPI and Walker.
Dt = 1.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # Epsilon parameter for Attaccalite-Sorella regularization.
num_opt_steps = 200 # Number of optimization steps.
wf_dump_freq = 1 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
optimizer_kwargs = { method = "sr", delta = 0.15, epsilon = 0.001, cg_flag = true, cg_max_iter = 10000, use_lm = true } # SR optimizer configuration.

# --- J1 / J2 / J3 family: OFF (mutually exclusive with the DTN polynomial Jastrow). ---
opt_J1_param = false
opt_J2_param = false
opt_J3_param = false

# --- DTN polynomial Jastrow (preset 'u_chi', chosen in the Step 2 conversion): u + chi only, no f. ---
opt_JDTN_u_param = true   # e-e channel (alpha block, J2 analogue).
opt_JDTN_chi_param = true # e-N channel (beta block, J1 analogue) -- carries the AE Kato cusp.
opt_JDTN_f_param = false  # preset 'u_chi' has no f block -- keep false.

# --- Geminal (MO coefficient) optimization: ON. ---
# fix_geminal_rank = 0 => 'not set' = full-rank ambient opt on the source class
# (SD_data here, since QE -> jQMC ships a Slater determinant).
opt_geminal = true
fix_geminal_rank = 0

# --- Lambda/AO basis: keep frozen (we only refine the Slater MO coefficients). ---
opt_lambda_basis_exp = false
opt_lambda_basis_coeff = false

# --- J3 AO basis optimization: off (optional keys; J3 is not used here, shown for completeness). ---
opt_J3_basis_exp = false
opt_J3_basis_coeff = false

[coulomb]
mpc = false # Periodic AE hydrogen; MPC finite-size correction not used in this tutorial.

# [precision]
# mode = "full"  # "full" (default, all float64; reference setting) or "mixed" (opt-in speed setting; see doc/notes/mixed_precision.md).
```

Key settings:

- 200 optimization steps, 300 measurements per rank per step (30 discarded as
  warmup, 10 bins), SR with `delta = 0.15` and `use_lm = true` (accelerated SR).
- `opt_JDTN_u_param = true`, `opt_JDTN_chi_param = true`
  (`opt_JDTN_f_param = false`: the `u_chi` preset has no `f` block).
- `opt_geminal = true`, `fix_geminal_rank = 0`: full MO-coefficient
  optimization of the Slater determinant from QE.
- J1/J2/J3 and AO-basis optimization are off.

The run optimizes 232 parameters:

```text
Number of variational parameters = 232.
Optimization batch: route=host, local samples=270, sampler ranks=128, real coordinates=440, storage parameters=232.
```

Each step took about 24 s, almost all of it sampling (which includes the
parameter derivatives of `ln Psi` and `e_L` needed by the optimizer). Analyse the
optimization:

```bash
jqmc-tool vmc analyze-output out_vmc -s vmc.jpg
```

```text
------------------------------------------------------
Iter     E (Ha)     Max generalized force (Ha/a.u.)  Max force S/N
------------------------------------------------------
   1  -2.1945(76)   +3.05(11)    49.845
   2  -2.1937(66)  +2.934(92)    54.591
   3  -2.2047(57)  +3.137(77)    64.883
   4  -2.2058(52)  +2.979(67)    65.437
   5  -2.2306(44)  +3.112(52)    70.431
...
  49  -2.3625(19)  +0.694(23)    31.074
  50  -2.3586(19)  +0.716(23)    33.358
...
  99  -2.3649(19)  +0.394(21)    19.121
 100  -2.3624(19)  -0.431(38)    20.678
...
 149  -2.3630(19)  -0.366(43)    15.560
 150  -2.3632(19)  -0.314(40)    14.504
...
 196  -2.3640(19)  -0.277(59)     9.143
 197  -2.3676(19)  -0.291(39)     9.966
 198  -2.3618(19)  +0.256(59)     7.700
 199  -2.3670(19)  -0.257(57)     8.523
 200  -2.3622(18)  +0.208(54)     7.952
------------------------------------------------------
Graph is saved in vmc.jpg.
```

(`-p` opens the same plot interactively.) The energy drops by about 0.17 Ha
from the initial wavefunction and is flat within error bars after roughly 50
steps. The maximum generalized force keeps decreasing, but its
signal-to-noise ratio is still about 8-10 at step 200 (the largest components
are a few `chi` coefficients), i.e. the parameters are not fully converged
even though the energy is. For a production run, continue the optimization
(`restart = true`) until `Max force S/N` falls below about 4-5; for this
tutorial the average of steps 181-200 is used as is.

## Step 4: MCMC with atomic forces (`03mcmc_JSD_force/`)

```bash
cd ../03mcmc_JSD_force
python ../../copy_optimized_hamiltonian.py --average-last 20 ../02vmc_JSD_geminal hamiltonian_data.h5
mpirun -np 128 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e
```

> [!NOTE]
> The helper averages the wave-function parameters of the last 20 dumps (here steps 181-200) with
> `jqmc-tool vmc average-wf`; see [Averaging the last optimization steps](../../doc/notes/wavefunction_optimization.md). The numbers quoted below were not sampled from this average: the reference run handed over the single last dump with
> `python ../../copy_optimized_hamiltonian.py ../02vmc_JSD_geminal hamiltonian_data.h5`.

`copy_optimized_hamiltonian.py --average-last 20` averages the dumps
`hamiltonian_data_opt_step_181.h5` to `hamiltonian_data_opt_step_200.h5`.

<!-- include: 03mcmc_JSD_force/mcmc.toml -->
```toml
[control]
job_type = "mcmc" # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 1 # Number of walkers per MPI process; add MPI ranks for more samples.
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "high" # Verbosity level. "low", "high", "devel", "mpi-low", "mpi-high", "mpi-devel"

[mcmc]
num_mcmc_steps = 20000 # Number of observable measurement steps per MPI and Walker.
num_mcmc_per_measurement = 40 # Number of MCMC updates per measurement.
num_mcmc_warmup_steps = 30 # Number of initial observable measurements discarded as warmup.
num_mcmc_bin_blocks = 10 # Number of blocks for binning per MPI and Walker.
Dt = 1.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # Epsilon parameter for Attaccalite-Sorella regularization.

# --- Atomic forces: this is the point of jqmc-example08. ---
atomic_force = true       # compute atomic forces in addition to e_L.
use_swct = true           # Space Warp Coordinate Transformation -- noise reduction for VMC forces.
compute_stress = false    # Stress is NOT computed in this example (see README for the one-liner).
parameter_derivatives = false

[coulomb]
mpc = false

# [precision]
# mode = "full"  # "full" (default, all float64; reference setting) or "mixed" (opt-in speed setting; see doc/notes/mixed_precision.md).
```

Key settings: 20000 measurements per rank (30 warmup, 10 bins),
**`atomic_force = true`** (Hellmann-Feynman + Pulay forces), and
**`use_swct = true`** (space-warp coordinate transformation, which reduces the
force variance). The end of `out_mcmc`:

```text
Final output(s):
  Total Energy: E = -2.36482 +- 0.00022 Ha.
  Variance: Var = 0.11950 +- 0.00013 Ha^2.
  Atomic Forces:
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  H       +0.0147(6)   -0.0462(5)   -0.0111(4)
  H       +0.0104(6)   -0.0212(6)   -0.0113(5)
  H       -0.0220(6)   +0.0328(6)   +0.0150(6)
  H       -0.0024(5)   +0.0346(6)   +0.0069(5)
  ------------------------------------------------
```

Recompute (or re-bin) the energy and forces from the checkpoint with the same
warmup and bin count as `mcmc.toml`:

```bash
jqmc-tool mcmc compute-energy restart.h5 -b 10 -w 30
jqmc-tool mcmc compute-force restart.h5 -b 10 -w 30
```

```text
Read restart checkpoint file(s) from restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 1280
E = -2.364819031772833 +- 0.00021723870495838572 Ha.
```

```text
Read restart checkpoint file(s) from restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 1280
Atomic Forces:
------------------------------------------------
Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
------------------------------------------------
H       +0.0147(6)   -0.0462(5)   -0.0111(4)
H       +0.0104(6)   -0.0212(6)   -0.0113(5)
H       -0.0220(6)   +0.0328(6)   +0.0150(6)
H       -0.0024(5)   +0.0346(6)   +0.0069(5)
------------------------------------------------
```

Values are mean(error in the last digits). Most components are tens of error
bars away from zero because the geometry is not an equilibrium structure; they
sum to zero over the four atoms within the error bars
((+0.0007, 0.0000, -0.0005) Ha/bohr), as translational invariance requires.

## Step 5: LRDMC (GFMC_t) with atomic forces (`04lrdmc_JSD_force/`)

```bash
cd ../04lrdmc_JSD_force
python ../../copy_optimized_hamiltonian.py --average-last 20 ../02vmc_JSD_geminal hamiltonian_data.h5
mpirun -np 128 jqmc lrdmc_tau.toml > out_lrdmc 2> out_lrdmc.e
```

> [!NOTE]
> The helper averages the wave-function parameters of the last 20 dumps (here steps 181-200) with
> `jqmc-tool vmc average-wf`; see [Averaging the last optimization steps](../../doc/notes/wavefunction_optimization.md). The numbers quoted below were not sampled from this average: the reference run handed over the single last dump with
> `python ../../copy_optimized_hamiltonian.py ../02vmc_JSD_geminal hamiltonian_data.h5`.

<!-- include: 04lrdmc_JSD_force/lrdmc_tau.toml -->
```toml
[control]
job_type = "lrdmc-tau" # GFMC_t (imaginary-time-step) driver. Atomic forces are supported here.
mcmc_seed = 34467
number_of_walkers = 1 # Walkers per MPI process; add MPI ranks for more samples.
max_time = 86400
restart = false
restart_chk = "restart.h5"
hamiltonian_h5 = "hamiltonian_data.h5"
verbosity = "high"

[lrdmc-tau]
num_mcmc_steps = 20000 # Number of observable measurement steps per MPI and Walker.
tau = 0.10            # Imaginary-time step between projections (Ha^-1, atomic units of imaginary time).
alat = 0.30           # Lattice discretization parameter (bohr).
non_local_move = "tmove" # T-move scheme for the non-local ECP term.
                         # AE hydrogen has no non-local channel from H.coulomb-ae.UPF (true 1/r),
                         # so this setting ("tmove", the default, or "dltmove") has no effect here.
num_gfmc_warmup_steps = 50
num_gfmc_bin_blocks = 50
num_gfmc_collect_steps = 20

# --- Atomic forces. ---
atomic_force = true       # compute atomic forces during the GFMC_t projection.
use_swct = false          # SWCT is off by default for LRDMC (matches the LRDMC force convention).
compute_stress = false    # Stress is NOT computed in this example.
epsilon_PW = 0.0          # Pathak-Wagner nodal regularization (off).

[coulomb]
mpc = false

# [precision]
# mode = "full"  # "full" (default, all float64; reference setting) or "mixed" (opt-in speed setting; see doc/notes/mixed_precision.md).
```

Key settings:

- `job_type = "lrdmc-tau"`: the GFMC_t driver, `tau = 0.10` Ha^-1 between
  branchings, lattice spacing `alat = 0.30` bohr.
- 20000 branching steps per rank; 50 warmup steps discarded, 50 bins, 20
  weight-collection steps.
- `non_local_move = "tmove"` is the default; the AE potential is local, so it
  has no non-local term to act on.
- **`atomic_force = true`**, `use_swct = false`, `compute_stress = false`.

The end of `out_lrdmc`:

```text
Survived walkers ratio = 99.03 %
Average of the number of projections  = 14
...
Final output(s):
  Total Energy: E = -2.38152 +- 0.00039 Ha.
  Variance: Var = 0.11695 +- 0.00017 Ha^2.
  Atomic Forces:
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  H       +0.0121(10)  -0.0452(9)   -0.0093(12)
  H       +0.0145(10)  -0.0188(13)  -0.0077(15)
  H       -0.0226(9)   +0.0294(11)  +0.0116(12)
  H       +0.0005(12)  +0.0382(11)  +0.0092(14)
  ------------------------------------------------
```

Recompute from the checkpoint with the TOML settings:

```bash
jqmc-tool lrdmc compute-energy restart.h5 -b 50 -w 50 -c 20
jqmc-tool lrdmc compute-force restart.h5 -b 50 -w 50 -c 20
```

```text
Read restart checkpoint file(s) from restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 50
E = -2.3815181662616034 +- 0.0003897520679547314 Ha.
```

```text
Read restart checkpoint file(s) from restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 50
Atomic Forces:
------------------------------------------------
Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
------------------------------------------------
H       +0.0121(10)  -0.0452(9)   -0.0093(12)
H       +0.0145(10)  -0.0188(13)  -0.0077(15)
H       -0.0226(9)   +0.0294(11)  +0.0116(12)
H       +0.0005(12)  +0.0382(11)  +0.0092(14)
------------------------------------------------
```

LRDMC lowers the energy by 16.7(5) mHa relative to VMC (-2.3815(4) vs
-2.3648(2) Ha), and the LRDMC forces agree with the VMC forces to within a few
mHa/bohr. The reported LRDMC energy and forces are measured with the continuum
local energy on walkers propagated with the lattice-regularized Hamiltonian;
see "Propagation and measurement local energies (`novar=0`)" in
`doc/notes/lrdmc.md`.

LRDMC energies and forces depend on `alat`. `jqmc-tool lrdmc
extrapolate-energy` extrapolates energies only; for forces, run several `alat`
values in sibling directories, compute each with `lrdmc compute-force`, and fit
each component against `alat^2` yourself.

## Summary

| Step | Directory | What it does | Result (reference run) |
|------|-----------|--------------|------------------------|
| 1 | `01DFT/` | QE LDA-PZ, k = (1/4, 1/4, 1/4) | E_LDA = -4.48559671 Ry |
| 2 | `02vmc_JSD_geminal/` | `jqmc-tool qe --kind pgto+pw --ecutwfc auto --jastrow-dtn u_chi` | 32 pGTO + 72 PW AOs |
| 3 | `02vmc_JSD_geminal/` | VMC opt of DTN Jastrow + MO coefficients | E ~ -2.363(2) Ha per step |
| 4 | `03mcmc_JSD_force/` | MCMC energy and forces (SWCT) | E = -2.3648(2) Ha |
| 5 | `04lrdmc_JSD_force/` | LRDMC (GFMC_t, alat = 0.30) energy and forces | E = -2.3815(4) Ha |

## Orbital tiling

This tutorial optimizes an untiled composite pGTO+PW determinant.
PW+pGTO tiling is available for a frozen determinant through
`--kind pgto+pw --tiling "2 1 1"`, provided the source contains a complete
primitive k coset for each requested supertwist. This tutorial's determinant
optimization remains an untiled calculation. A tiled variant must use
`opt_geminal = false` and an independent ordinary supercell Jastrow; see the
[tiling recipe](../../doc/examples.md#jqmc-example14-frozen-orbitals-in-a-tiled-supercell).
