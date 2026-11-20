# jqmc-example06: Periodic ice Ih from PySCF pGTOs (non-Gamma complex twist)

End-to-end VMC and LRDMC of crystalline ice from a single-k PySCF RHF trial,
demonstrating the PySCF -> TREXIO -> jQMC periodic-Gaussian (pGTO) route on a
multi-atom cell with complex Bloch orbitals.

## Before you start

In addition to jQMC, this route needs PySCF (`pyscf.pbc`), its
`pyscf.tools.trexio` exporter and the TREXIO Python package with HDF5 support
(`python -c "from pyscf.tools import trexio"` must succeed). The supplied
script generates the periodic mean-field state and its TREXIO file locally. No
Quantum ESPRESSO calculation or plane-wave projection is involved: jQMC imports
the Gaussian basis and the MO coefficients from TREXIO. For a QE plane-wave
starting point instead, see example07 (B-spline) or example08 (pGTO+PW).

All paths below are relative to `examples/jqmc-example06`; follow the displayed
`cd` commands. Keep the supplied scripts and TOML files; do not generate default
TOMLs over them. Where serial CPU, MPI CPU and MPI GPU launch lines are shown,
they are alternatives: choose one.

The outputs shown below come from one reference run of every step: 128 MPI
ranks on one 128-core CPU node (CPU only, `OMP_NUM_THREADS=1`; PySCF ran as a
single process with 128 OpenMP threads).
Wall times on that node:

| Step | Wall time |
|---|---|
| 1. PySCF RHF + TREXIO export | 19 s |
| 2. `jqmc-tool trexio convert-to` (login node) | 2 s |
| 2b. Optional check: determinant-only MCMC, 5000 steps | 6 min |
| 3. VMC optimisation, 150 steps | 1 h 41 min |
| 4. MCMC, 5000 steps | 8 min |
| 5. LRDMC `lrdmc-tau`, 5000 branchings at alat = 0.30 / 0.25 / 0.20 | 53 min / 1 h 10 min / 1 h 42 min |

QMC numbers are stochastic: with a different number of ranks, walkers or seeds
you will get statistically compatible but not identical values.

## What this example is

- System: ice Ih primitive cell, 4 H2O (proton-ordered), 32 valence electrons
  (ccECP on O and H).
- Mean field: single-k restricted Hartree-Fock (`scf.RHF(cell, kpt=...)`, not a
  multi-k `KRHF`) with `pyscf.pbc` and density fitting (GDF). The ccECP
  cc-pVDZ basis is written out inline in `run_pyscf.py` (`gto.basis.parse`) and
  used as Cartesian GTOs (`cell.cart = True`).
- k point: the general twist k = (0.25, 0.25, 0.25) in scaled crystal
  coordinates, chosen because it is **not** Gamma: the MO coefficients are
  complex, so the Slater determinant is evaluated in complex arithmetic (see
  "Notes on the complex-Bloch path" below).
- Trial wave function: the RHF Slater determinant (frozen) times a two-body
  (J2) and a three-body (J3) Jastrow factor; only the Jastrow is optimized.

## Layout

```text
jqmc-example06/
  01DFT/run_pyscf.py          # PySCF single-k RHF + TREXIO dump
  02vmc_JSD/vmc.toml          # Jastrow optimization (J2 + J3)
  03mcmc_JSD/mcmc.toml        # VMC energy with the optimized Jastrow
  04lrdmc_JSD/lrdmc_tau.toml  # LRDMC, lrdmc-tau (GFMC_t, fixed-tau) driver
```

The `[precision]` block is left commented in every `.toml`, so the default
`mode = "full"` (all float64, the reference setting) is used. `mode = "mixed"`
is an opt-in speed setting (see `doc/notes/mixed_precision.md`); it is
demonstrated in `jqmc-example07`, `jqmc-example09` and `jqmc-example10`.

---

## Step 1: Generate the trial wavefunction (PySCF single-k RHF -> TREXIO)

`01DFT/run_pyscf.py` builds the ice Ih primitive cell, runs RHF at the chosen
twist and writes `ice_ccecp_ccpvdz_cart.h5` (TREXIO) plus the PySCF log
`ice.out`.

```bash
cd 01DFT
python run_pyscf.py
```

The end of `ice.out`:

```text
cycle= 8 E= -67.911366408466  delta_E= -2.08e-08  |g|= 2.77e-05  |ddm|= 0.000648
...
converged SCF energy = -67.9113664088955
```

PySCF uses `exxdiv = ewald` (its default) for the exchange divergence, so this
RHF energy is directly comparable with the Ewald-summed energy jQMC evaluates
(see the optional check in Step 2).

You can inspect the TREXIO content with `jqmc-tool` (abridged):

```bash
jqmc-tool trexio show-info ice_ccecp_ccpvdz_cart.h5
```

```text
Non-Gamma twist loaded, k_point (fractional) = (0.2499999999999999, 0.2500000000000001, 0.25)
pbc_flag = True
num_ele_up = 16, num_ele_dn = 16, spin_polarized = False
spin_dependent MOs = False
Complex MO coefficients loaded (down-spin built as conj(C) downstream). Gauge-fixed 96 bands.
Cartesian basis functions.
**Structure_data
  PBC flag = True
  vec A = [7.25474552 4.18852882 0.        ] Bohr
  vec B = [-7.25474741  4.18852882  0.        ] Bohr
  vec C = [ 0.          0.         13.67135026] Bohr
...
**AOs_cart_pgto_complex_data**
  Number of AOs = 96
  Number of primitive AOs = 3195
  Angular part is the polynomial (Cartesian) function.
  Number of lattice image cells = 56
  max |n_a|, |n_b|, |n_c| over image indices = 2
  image truncation tolerance = 1.000e-07
  ------------------------------------
  **basis set for atom index 1: H**
  ------------------------------------
  H s
    9.569723 0.0425061
    1.669254 0.2520934
    0.405502 0.7898456
  H s
    0.116087 1.0000000
  H p
    0.666515 1.0000000
...
**MOs_data
  Number of MOs = 16
  dim. of MOs coeff = (16, 96)
...
**Ewald_data
  eta = 0.589162 Bohr^-1
  r_real_cut = 8.1447 Bohr
  g_max = 5.6542 Bohr^-1
  N_real_images = 281
  N_g = 2538
  ii_energy = -27.7761312403 Ha
```

---

## Step 2: Convert the TREXIO file into a jQMC Hamiltonian

`jQMC` reads a `hamiltonian_data.h5` file that bundles the structure, ECP,
trial determinant and Jastrow factors. Build it from the TREXIO file:

```bash
jqmc-tool trexio convert-to ice_ccecp_ccpvdz_cart.h5 -o hamiltonian_data.h5 -j2 1.0 -j3 ao-small
```

```text
Non-Gamma twist loaded, k_point (fractional) = (0.2499999999999999, 0.2500000000000001, 0.25)
...
Hamiltonian data is saved in hamiltonian_data.h5.
```

- `-j2 1.0` sets the initial two-body Jastrow parameter.
- `-j3 ao-small` builds the three-body Jastrow on a reduced, uncontracted
  atomic basis (3s on H, 3s1p on O): 48 functions, i.e. a 48 x 49 J3 matrix,
  2353 variational parameters together with J2. The full uncontracted basis
  (`-j3 ao`, 176 functions, 31153 parameters) is also accepted, but it makes
  every optimization step much more expensive and needs many more samples to
  optimize stably.
- The J3 basis is always built at k = 0 (real, phase-free), independent of the
  determinant twist; this is automatic.

Copy the initial trial only into VMC. MCMC and LRDMC must use an optimized dump
from that VMC run, as shown below.

```bash
cp hamiltonian_data.h5 ../02vmc_JSD/
```

### Step 2b (optional): check the determinant against PySCF

A wave function without any Jastrow factor reproduces the RHF energy, which
checks the whole PySCF -> TREXIO -> jQMC import (basis, complex MOs, ECP and
Ewald sum). Still in `01DFT/`, convert again without Jastrow and sample it with
the Step 4 `mcmc.toml`:

```bash
mkdir -p hf_check
jqmc-tool trexio convert-to ice_ccecp_ccpvdz_cart.h5 -o hf_check/hamiltonian_data.h5 -j3 none
cp ../03mcmc_JSD/mcmc.toml hf_check/
cd hf_check
mpirun -np 128 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e
jqmc-tool mcmc compute-energy restart.h5 -b 100 -w 50
cd ..
```

```text
Found 128 MPI ranks.
Total number of binned samples = 12800
E = -67.90511500503192 +- 0.00927147021766685 Ha.
```

This agrees with the PySCF RHF energy (-67.91137 Ha) within the error bar.

```bash
cd ..
```

---

## Step 3: Optimize the Jastrow factor (VMC)

`vmc.toml` optimizes J2 and J3 only (the determinant is frozen at the RHF
solution) for 150 steps. Each step uses 300 measurements per rank (1 walker per
rank; statistics come from the MPI ranks), discards the first 30 and uses 10
bins (the minimums the code accepts without a warning are 30 warmup
measurements and 10 bins). A dump is written
every step (`wf_dump_freq = 1`), so that the hand-off below can average the last 20 steps. The reference run behind the quoted numbers wrote a dump every 10 steps; the dump frequency does not change the optimization.

The optimizer is plain stochastic reconfiguration (`use_lm = false`) with a
small step, `delta = 0.02`. In the reference run, `delta = 0.05` (and also
`delta = 0.05, epsilon = 0.01`) diverged within five steps on this system.
Accelerated SR (`use_lm = true`, `delta = 0.15`) did not diverge in a short
test, but it needs the local-energy parameter derivatives, which raised the
cost of sampling from about 0.09 s to about 1.8 s per measurement here.

<!-- include: 02vmc_JSD/vmc.toml -->
```toml
[control]
job_type = "vmc" # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 1 # Number of walkers per MPI process (this example uses 1 walker per MPI rank and gains statistics from MPI ranks).
max_time = 10000 # Maximum time in sec.
restart = false
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low", "high", "devel", "mpi-low", "mpi-high", "mpi-devel"

[vmc]
num_mcmc_steps = 300 # Number of observable measurement steps per MPI process and walker. The global total is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates between successive observable measurements.
num_mcmc_warmup_steps = 30 # Number of initial observable measurements discarded as warmup.
num_mcmc_bin_blocks = 10 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 2.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # Epsilon parameter for Attaccalite-Sorella regularization.
num_opt_steps = 150 # Number of optimization steps.
wf_dump_freq = 1 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
optimizer_kwargs = { method = "sr", delta = 0.02, epsilon = 0.001, cg_flag = true, cg_max_iter = 10000, use_lm = false } # SR optimizer configuration (method plus step/regularization).
opt_J1_param = false
opt_J2_param = true
opt_J3_param = true
opt_geminal = false
opt_J3_basis_exp = false
opt_J3_basis_coeff = false
opt_lambda_basis_exp = false
opt_lambda_basis_coeff = false

# [precision]
# mode = "full"  # "full" (default, all float64; reference setting) or "mixed" (opt-in speed setting; see doc/notes/mixed_precision.md).
```

```bash
cd 02vmc_JSD
jqmc vmc.toml > out_vmc 2> out_vmc.e                    # serial CPU
mpirun -np 128 jqmc vmc.toml > out_vmc 2> out_vmc.e     # MPI CPU
mpiexec -n 4 -map-by ppr:4:node jqmc vmc.toml > out_vmc 2> out_vmc.e  # MPI GPU
```

Inspect the convergence (abridged):

```bash
jqmc-tool vmc analyze-output out_vmc
jqmc-tool vmc analyze-output out_vmc -s vmc.jpg   # also save a plot
```

```text
------------------------------------------------------
Iter     E (Ha)     Max generalized force (Ha/a.u.)  Max force S/N
------------------------------------------------------
   1  -67.399(37)  +2.413(61)    41.421
   2  -67.766(33)  +2.416(63)    41.679
   3  -68.004(30)  +2.578(65)    40.860
...
  10  -68.657(20)  +1.933(59)    32.940
...
  20  -68.911(15)  +0.899(43)    20.815
...
  30  -68.945(15)  +0.448(40)    11.164
...
  40  -68.983(14)  +0.325(37)     8.672
...
  50  -68.997(14)  +0.103(35)     3.809
...
 100  -69.005(14)  +0.063(23)     3.312
...
 148  -68.984(14)  +0.092(26)     3.647
 149  -68.983(14)  +0.065(32)     3.558
 150  -68.998(14)  -0.071(22)     3.286
------------------------------------------------------
```

The Jastrow lowers the energy by about 1.6 Ha from the first step (J3 = 0) and
the energy is flat from about step 50 on. The maximum generalized force is
then about 0.05-0.1 Ha/a.u. with `Max force S/N` around 3-4, i.e. the remaining
force is mostly noise. Per optimization step the run spent about 27 s on
sampling and about 13 s computing the generalized forces and solving the SR
equations.

If you need more steps, set `restart = true` and rerun; analyze both outputs
together with `jqmc-tool vmc analyze-output out_vmc out_vmc_cont`.

The optimized wave functions are saved as `hamiltonian_data_opt_step_N.h5`
(N = 1, 2, ..., 150, one per step with `wf_dump_freq = 1`). The original `hamiltonian_data.h5` is not overwritten.

---

## Step 4: VMC energy with the optimized Jastrow (MCMC)

Average the last 20 optimized dumps into the MCMC directory and sample the result.
`mcmc.toml` runs 5000 measurement steps per rank, discards the first 50 and
uses 100 bins per rank and walker (a bin count, not a bin length).

<!-- include: 03mcmc_JSD/mcmc.toml -->
```toml
[control]
job_type = "mcmc" # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 1 # Number of walkers per MPI process (PBC complex-Bloch run; scale samples via MPI ranks).
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low", "high", "devel", "mpi-low", "mpi-high", "mpi-devel"

[mcmc]
num_mcmc_steps = 5000 # Number of observable measurement steps per MPI process and walker. The global total is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates between successive observable measurements.
num_mcmc_warmup_steps = 50 # Number of initial observable measurements discarded as warmup.
num_mcmc_bin_blocks = 100 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 2.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # Epsilon parameter for Attaccalite-Sorella regularization.
atomic_force = false # If true, compute atomic forces.
compute_stress = false # If true, compute the stress tensor / pressure (periodic only).

# [precision]
# mode = "full"  # "full" (default, all float64; reference setting) or "mixed" (opt-in speed setting; see doc/notes/mixed_precision.md).
```

```bash
cd ..
python ../copy_optimized_hamiltonian.py --average-last 20 02vmc_JSD 03mcmc_JSD/hamiltonian_data.h5
cd 03mcmc_JSD
mpirun -np 128 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e
jqmc-tool mcmc compute-energy restart.h5 -b 100 -w 50
```

> [!NOTE]
> The helper averages the wave-function parameters of the last 20 dumps (here steps 131-150) with
> `jqmc-tool vmc average-wf`; see [Averaging the last optimization steps](../../doc/notes/wavefunction_optimization.md). The numbers quoted below were not sampled from this average: the reference run handed over the single last dump with
> `python ../copy_optimized_hamiltonian.py 02vmc_JSD 03mcmc_JSD/hamiltonian_data.h5`, which printed
> `Copied optimized trial: 02vmc_JSD/hamiltonian_data_opt_step_150.h5 -> 03mcmc_JSD/hamiltonian_data.h5`.
> The reference VMC run also wrote a dump only every 10 steps; the shipped VMC input sets `wf_dump_freq = 1` because this hand-off needs one dump per step (the dump frequency does not change the optimization).

End of `out_mcmc`:

```text
Final output(s):
  Total Energy: E = -68.99793 +- 0.00337 Ha.
  Variance: Var = 2.05601 +- 0.01767 Ha^2.
```

`compute-energy` re-bins the stored local energies with the same settings
(`-w 50`, `-b 100`):

```text
Found 128 MPI ranks.
Total number of binned samples = 12800
E = -68.99792544148387 +- 0.0033675243776161933 Ha.
```

This is the VMC total energy of the 4-molecule cell with the optimized JSD
trial at this single twist (so it still contains the one-body finite-size error
of that twist). Compared with Step 2b, the Jastrow lowers the energy by
1.09 Ha and the variance from 13.0 to 2.06 Ha^2.

---

## Step 5: LRDMC

LRDMC projects the trial wave function toward the fixed-phase ground state
(the complex-twist analogue of fixed-node); the non-local ECP part is treated
with `non_local_move = "tmove"`. The `lrdmc-tau` driver (GFMC_t) used here
propagates the walkers for a fixed imaginary time `tau = 0.10` Ha^-1 between
branchings. The energy carries an `O(alat^2)` lattice bias, so the final
number is extrapolated over three lattice spacings, `alat` = 0.30, 0.25 and
0.20 bohr. `lrdmc_tau.toml` runs 5000 measurements, discards 50 and uses 50
bins and 20 weight-collection steps:

<!-- include: 04lrdmc_JSD/lrdmc_tau.toml -->
```toml
[control]
job_type = "lrdmc-tau" # GFMC_t (fixed imaginary-time step) driver.
mcmc_seed = 34467
number_of_walkers = 1 # PBC complex-Bloch run; scale samples via MPI ranks.
max_time = 86400
restart = false
restart_chk = "restart.h5"
hamiltonian_h5 = "hamiltonian_data.h5"
verbosity = "low"

[lrdmc-tau]
num_mcmc_steps = 5000 # Number of observable measurement steps per MPI and Walker.
tau = 0.10 # Imaginary time between branchings (Ha^-1, atomic units of imaginary time).
alat = 0.30 # Lattice discretization parameter (bohr). LRDMC energy carries an O(alat^2) bias.
non_local_move = "tmove" # T-move scheme for the non-local ECP terms (the default).
num_gfmc_warmup_steps = 50
num_gfmc_bin_blocks = 50
num_gfmc_collect_steps = 20
atomic_force = false
compute_stress = false

# [precision]
# mode = "full"  # "full" (default, all float64; reference setting) or "mixed" (opt-in speed setting; see doc/notes/mixed_precision.md).
```


Start with `alat = 0.30` (the value in the shipped TOML), in its own directory
with the optimized trial:

```bash
cd ../04lrdmc_JSD
mkdir -p alat_0.30
cp ../03mcmc_JSD/hamiltonian_data.h5 alat_0.30/
cp lrdmc_tau.toml alat_0.30/lrdmc.toml
cd alat_0.30
mpirun -np 128 jqmc lrdmc.toml > out_lrdmc 2> out_lrdmc.e
jqmc-tool lrdmc compute-energy restart.h5 -b 50 -w 50 -c 20
cd ..
```

End of `out_lrdmc` and the `compute-energy` result:

```text
Survived walkers ratio = 95.97 %
Average of the number of projections  = 104

Final output(s):
  Total Energy: E = -69.15842 +- 0.00283 Ha.
  Variance: Var = 4.05580 +- 0.35169 Ha^2.
```

```text
Read restart checkpoint file(s) from restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 50
E = -69.1584242969248 +- 0.002830157538941747 Ha.
```

`lrdmc compute-energy` re-bins the stored weights and local energies with the
TOML settings (`-w 50` warmup measurements, `-b 50` bins, `-c 20`
weight-collection steps). `Average of the number of projections` is the mean
number of lattice moves per `tau`. The LRDMC energy lies 0.16 Ha below the
Step 4 VMC energy (-68.998 Ha).

Prepare the two smaller spacings from `04lrdmc_JSD`; only `alat` changes:

```bash
mkdir -p alat_0.25 alat_0.20
for a in 0.25 0.20; do
  sed -e "s/^alat = 0.30/alat = $a/" lrdmc_tau.toml > alat_$a/lrdmc.toml
  cp ../03mcmc_JSD/hamiltonian_data.h5 alat_$a/
done
```

Then enter each directory and run `jqmc lrdmc.toml` and the same
`compute-energy` command as above:

```text
alat_0.25:  E = -69.1534264416355 +- 0.0029467917979851214 Ha.
alat_0.20:  E = -69.14975258523849 +- 0.0029448126202329763 Ha.
```

A smaller `alat` makes each lattice move shorter, so more moves are needed per
`tau` (on average 104, 150 and 237 at `alat` = 0.30, 0.25 and 0.20) and the
run costs more (53 min, 1 h 10 min and 1 h 42 min here).

Extrapolate from `04lrdmc_JSD`:

```bash
jqmc-tool lrdmc extrapolate-energy alat_0.20/restart.h5 alat_0.25/restart.h5 alat_0.30/restart.h5 -p 1 -b 50 -w 50 -c 20 -s lrdmc_ext.jpg
```

```text
------------------------------------------------------------------------
Read restart checkpoint files from ['alat_0.20/restart.h5', 'alat_0.25/restart.h5', 'alat_0.30/restart.h5'].
Found 128 MPI ranks in alat_0.20/restart.h5.
  Total number of binned samples = 50
  For a = 0.2 bohr: E = -69.14975258523849 +- 0.0029448126202329763 Ha.
Found 128 MPI ranks in alat_0.25/restart.h5.
  Total number of binned samples = 50
  For a = 0.25 bohr: E = -69.1534264416355 +- 0.0029467917979851214 Ha.
Found 128 MPI ranks in alat_0.30/restart.h5.
  Total number of binned samples = 50
  For a = 0.3 bohr: E = -69.1584242969248 +- 0.002830157538941747 Ha.
------------------------------------------------------------------------
Extrapolation of the energy with respect to a^2.
  Polynomial order = 1.
  For a -> 0 bohr: E = -69.14271960078443 +- 0.005542139494657294 Ha.
------------------------------------------------------------------------
Graph is saved in lrdmc_ext.jpg.
```

The energy rises slightly as `alat` decreases (by 9(4) mHa from 0.30 to
0.20 bohr). The extrapolated energy, -69.143(6) Ha, is 0.14 Ha below the VMC
energy. The fit and its error are computed in closed form, so rerunning the
command on the same checkpoints prints the same numbers.

`-p 1` fits $E(a) = E_0 + c_2 a^2$. The tool default `-p 2` adds an $a^4$
term, which leaves no degree of freedom with only three spacings; use it only
with four or more. `-s` saves the fit as a plot (`-g` opens it interactively).

---

## Notes on the complex-Bloch path

- The MO coefficients at k = (0.25, 0.25, 0.25) are complex. jQMC evaluates
  the Slater determinant in complex arithmetic (the down-spin orbitals are the
  complex conjugates of the up-spin ones); the reported energies are real.
  The treatment of complex wave functions and their parameter derivatives is
  described in `doc/notes/complex_gradients.md`.
- The Jastrow factor is real. The J3 AO basis is always built at k = 0.
- A single twist keeps the one-body finite-size error of that twist; twist
  averaging is shown in `jqmc-example09`.

## Orbital tiling

For a larger frozen-node supercell, export a complete multi-k primitive
coset from PySCF and use the [tiling recipe](../../doc/examples.md#jqmc-example14-frozen-orbitals-in-a-tiled-supercell).
The single-k input here is not enough for nontrivial tiling. Tiled J3 uses
an independent ordinary supercell GTO basis.
