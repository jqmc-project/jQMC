# jqmc-example05: All-electron H2 energies and atomic forces

Energy and atomic forces of a hydrogen molecule ($R = 0.74\;\text{\AA}$)
with Cartesian GTOs (cc-pVTZ, all electron). The example compares the JSD
and JAGP ansatzes at the VMC (MCMC) and LRDMC levels. Forces are computed
with jQMC's analytic derivatives. The pioneering application of algorithmic
differentiation in ab initio QMC was introduced by S. Sorella and
L. Capriotti in 2010 [^2010SORjcp].

H$_2$ is the textbook check: the exact non-relativistic Born-Oppenheimer
energy at $R = 1.4$ bohr is $-1.174475$ Ha ($R = 0.74$ Angstrom = 1.398 bohr
is close enough that the energy difference, about $10^{-6}$ Ha, is far below
the error bars here). Because the two-electron singlet ground state has no
nodes, fixed-node LRDMC is exact for it apart from the lattice-spacing bias
and statistics, and every VMC energy must lie above it.

## Before you start

- You need PySCF with TREXIO export
  (`python -c "from pyscf.tools import trexio"` must succeed).
- The numbers below were obtained on **one 128-core CPU node with 128 MPI
  ranks** (4 walkers per rank, i.e.
  512 walkers). Wall times of that setup: JSD VMC 9 min, JAGP VMC 23 min,
  each MCMC run about 1 min, each LRDMC run 8 min. The job is small enough
  for a workstation; with fewer ranks the error bars grow as
  $1/\sqrt{\text{ranks}}$ at fixed input.
- The analysis commands, the meaning of the main TOML keys and the restart
  rules are summarized in the examples index (`examples/README.md`).

All paths below are relative to `examples/jqmc-example05`; follow the displayed `cd`
commands.

## Generate a trial WF

The first step of ab initio QMC is to generate a trial wave function with a
mean-field method such as DFT or Hartree-Fock. jQMC reads mean-field results
through the TREXIO file format. The following PySCF script runs a DFT-LDA
calculation of the hydrogen molecule at $R = 0.74\;\text{\AA}$ and writes a
TREXIO file.

```bash
cd 01DFT
```

<!-- include: 01DFT/run_pyscf.py -->
```python
from pyscf import gto, scf
from pyscf.tools import trexio

R = 0.74  # angstrom
filename = f"H2_R_{R:.2f}.h5"

mol = gto.Mole()
mol.verbose = 5
mol.atom = f"""
H    0.000000000   0.000000000  {-R / 2}
H    0.000000000   0.000000000  {+R / 2}
"""
mol.basis = "ccpvtz"
mol.unit = "A"
mol.ecp = None
mol.charge = 0
mol.spin = 0
mol.symmetry = False
mol.cart = True
mol.output = f"H2_R_{R:.2f}.out"
mol.build()

mf = scf.KS(mol).density_fit()
mf.max_cycle = 200
mf.xc = "LDA_X,LDA_C_PZ"
mf_scf = mf.kernel()

trexio.to_trexio(mf, filename)
```

Run it (a few seconds):

```bash
python run_pyscf.py
grep "converged SCF energy" H2_R_0.74.out
cd ..
```

```text
converged SCF energy = -1.13700890763025
```

## Convert TREXIO to jQMC format and optimize a trial WF: JSD (VMC)

Next, convert the TREXIO file to the jQMC format using `jqmc-tool`, and then
optimize the Jastrow factor (J1, J2 and J3) **and** the MO coefficients of the
determinant (`opt_geminal = true` in `vmc.toml`). Optimizing the orbitals
removes the self-consistency bias that fixed DFT orbitals would leave in the
forces (see the note in the MCMC section).

```bash
cd 02vmc_JSD
cp ../01DFT/H2_R_0.74.h5 .
jqmc-tool trexio convert-to H2_R_0.74.h5 -j1 1.0 -j2 1.0 -j3 mo
```

```text
...
Cartesian basis functions.
Hamiltonian data is saved in hamiltonian_data.h5.
```

The generated `hamiltonian_data.h5` is a wave-function file in the jQMC
format. `-j1` and `-j2` give the initial one- and two-body Jastrow
parameters, and `-j3` the basis (`ao`: atomic orbitals, `mo`: molecular
orbitals) of the three-body Jastrow factor.

Use the supplied VMC input. Directly edit `vmc.toml` if you want to change a parameter.

<!-- include: 02vmc_JSD/vmc.toml -->
```toml
[control]
job_type = "vmc" # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low", "high", "devel", "mpi-low", "mpi-high", "mpi-devel"

[vmc]
num_mcmc_steps = 300 # Number of observable measurement steps per MPI process and walker. The global total is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates between successive observable measurements.
num_mcmc_warmup_steps = 30 # Number of initial observable measurements discarded as warmup.
num_mcmc_bin_blocks = 10 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 1.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # Epsilon parameter for Attaccalite-Sorella regularization.
num_opt_steps = 300 # Number of optimization steps.
wf_dump_freq = 1 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
opt_J1_param = true
opt_J2_param = true
opt_J3_param = true
opt_geminal = true
fix_geminal_rank = 0
opt_J3_basis_exp = false
opt_J3_basis_coeff = false
opt_lambda_basis_exp = false
opt_lambda_basis_coeff = false
optimizer_kwargs = { method = "sr", delta = 0.15, epsilon = 0.001, cg_flag = true, cg_max_iter = 10000, use_lm = true }
```

Launch the job (we used 128 MPI ranks):

```bash
mpirun -np 128 jqmc vmc.toml > out_vmc 2> out_vmc.e
```

The 300 optimization steps took 507 s (`Total elapsed time for optimization
300 steps. = 507.07 sec.`). Inspect the optimization with `jqmc-tool`:

```bash
jqmc-tool vmc analyze-output out_vmc
```

```text
------------------------------------------------------
Iter     E (Ha)     Max generalized force (Ha/a.u.)  Max force S/N
------------------------------------------------------
   1  -0.9569(44)  +1.104(12)   120.764
   2  -0.9840(37)  +1.041(10)   137.957
   3  -1.0032(31)  +0.9840(90)   157.280
   4  -1.0201(26)  +0.9090(80)   179.558
   5  -1.0429(21)  +0.8310(60)   208.532
 ...
 291  -1.17230(25)  -0.0030(10)     4.260
 292  -1.17221(25)  -0.0040(10)     5.816
 293  -1.17229(25)  -0.0020(10)     2.169
 294  -1.17281(26)  -0.0030(10)     3.933
 295  -1.17249(26)  -0.0020(10)     3.061
 296  -1.17245(26)  -0.0020(10)     2.950
 297  -1.17242(25)  -0.0030(10)     3.383
 298  -1.17238(26)  -0.0030(10)     3.260
 299  -1.17215(25)  -0.0020(10)     2.734
 300  -1.17246(26)  -0.0030(10)     4.589
------------------------------------------------------
```

The energy is converged after about 150 steps. The important criteria are
`Max generalized force` and `Max force S/N`. `Max generalized force` (the
largest energy derivative with respect to a variational parameter) should be
zero within its error bar; a practical criterion is `Max force S/N` < 4-5,
reached here in the last ~100 steps. The optimizer log of the reference run
printed the force error with three fixed decimals, so the `Max generalized
force` errors in these tables carry one significant digit (`-0.0030(10)`
is an error of 0.001); the current log prints at least two significant digits of the
error.

```bash
cd ..
```

## Compute Energy and Atomic forces: JSD (MCMC)

Using the optimized wavefunction, compute the energy and atomic forces via
MCMC. Hand over the average of the last 20 optimized `hamiltonian_data` dumps
from the previous step and use the supplied TOML file.

```bash
cd 03mcmc_JSD
python ../../copy_optimized_hamiltonian.py --average-last 20 ../02vmc_JSD hamiltonian_data.h5
```

> [!NOTE]
> The helper averages the wave-function parameters of the last 20 dumps (here steps 281-300) with
> `jqmc-tool vmc average-wf`; see [Averaging the last optimization steps](../../doc/notes/wavefunction_optimization.md). The numbers quoted below were not sampled from this average: the reference run handed over the single last dump with
> `python ../../copy_optimized_hamiltonian.py ../02vmc_JSD hamiltonian_data.h5`, which printed
> `Copied optimized trial: ../02vmc_JSD/hamiltonian_data_opt_step_300.h5 -> hamiltonian_data.h5`.
> The reference VMC run also wrote a dump only every 10 steps; the shipped VMC input sets `wf_dump_freq = 1` because this hand-off needs one dump per step (the dump frequency does not change the optimization).

<!-- include: 03mcmc_JSD/mcmc.toml -->
```toml
[control]
job_type = "mcmc" # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low", "high", "devel", "mpi-low", "mpi-high", "mpi-devel"

[mcmc]
num_mcmc_steps = 10000 # Number of observable measurement steps per MPI process and walker. The global total is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates between successive observable measurements.
num_mcmc_warmup_steps = 30 # Number of initial observable measurements discarded as warmup.
num_mcmc_bin_blocks = 10 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 1.2 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # Epsilon parameter for Attaccalite-Sorella regularization.
atomic_force = true
use_swct = true # Apply Space Warp Coordinate Transformation (SWCT) to atomic forces.
```

`atomic_force = true` adds the atomic forces; `use_swct = true` applies the
space-warp coordinate transformation, which reduces the variance of the
force estimator.

```bash
mpirun -np 128 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e
```

### Analyze (MCMC)

The `Final output(s):` block of `out_mcmc` prints the energy, the variance
and the forces:

```bash
grep -A10 "Final output" out_mcmc
```

```text
Final output(s):
  Total Energy: E = -1.17246 +- 0.00004 Ha.
  Variance: Var = 0.00879 +- 0.00001 Ha^2.
  Atomic Forces:
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  H       +0.8(2.7)e-05 -0.4(2.7)e-05 +0.00173(14)
  H       -0.8(2.7)e-05 +0.4(2.7)e-05 -0.00173(14)
  ------------------------------------------------
```

To re-analyze the checkpoint with explicit warmup and bin values (here the
TOML's):

```bash
jqmc-tool mcmc compute-energy restart.h5 -w 30 -b 10
jqmc-tool mcmc compute-force restart.h5 -w 30 -b 10
```

```text
Read restart checkpoint file(s) from restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 5120
E = -1.17245845275111 +- 4.2375311507098714e-05 Ha.
Read restart checkpoint file(s) from restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 5120
Atomic Forces:
------------------------------------------------
Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
------------------------------------------------
H       +0.8(2.7)e-05 -0.4(2.7)e-05 +0.00173(14)
H       -0.8(2.7)e-05 +0.4(2.7)e-05 -0.00173(14)
------------------------------------------------
```

The number of binned samples is 10 bins x 4 walkers x 128 ranks. The forces
are $F = -dE/dR$ in Ha/bohr. The atoms sit at $z = \mp R/2$, so a positive
$F_z$ on the first atom (at $-z$) and a negative one on the second pull the
atoms together (the bond wants to shorten); the opposite signs push them
apart.

> [!NOTE]
> If one were to optimize only the Jastrow factor while keeping the determinant
> fixed (`opt_geminal = false`), the atomic forces would contain a
> self-consistency bias[^2022NAKjcp][^2021TIHjcp] because the DFT orbitals are
> not stationary with respect to the VMC energy. Here `opt_geminal = true`
> also optimizes the MO coefficients.

```bash
cd ..
```

## Compute Energy and Atomic forces: JSD (LRDMC)

Using the same optimized wave function, compute the energy and atomic forces
via LRDMC. The single lattice spacing `alat = 0.10` bohr is small for H$_2$,
so the remaining $O(a^2)$ bias is expected to be small, but it is not
extrapolated away here; `non_local_move` has no effect in this all-electron
calculation. The `lrdmc-tau` driver projects the walkers for a fixed
imaginary time `tau = 0.10` Ha$^{-1}$ between two branching (population
control) steps, so `num_mcmc_steps = 50000` corresponds to a total projection
time of 5000 Ha$^{-1}$ per walker.

```bash
cd 04lrdmc_JSD
python ../../copy_optimized_hamiltonian.py --average-last 20 ../02vmc_JSD hamiltonian_data.h5
```

> [!NOTE]
> The helper averages the wave-function parameters of the last 20 dumps (here steps 281-300) with
> `jqmc-tool vmc average-wf`; see [Averaging the last optimization steps](../../doc/notes/wavefunction_optimization.md). The numbers quoted below were not sampled from this average: the reference run handed over the single last dump with
> `python ../../copy_optimized_hamiltonian.py ../02vmc_JSD hamiltonian_data.h5`.
> The reference VMC run also wrote a dump only every 10 steps; the shipped VMC input sets `wf_dump_freq = 1` because this hand-off needs one dump per step (the dump frequency does not change the optimization).

<!-- include: 04lrdmc_JSD/lrdmc.toml -->
```toml
[control]
job_type = "lrdmc-tau" # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low", "high", "devel", "mpi-low", "mpi-high", "mpi-devel"

[lrdmc-tau]
num_mcmc_steps = 50000 # Number of observable measurement steps per MPI process and walker. The global total is num_mcmc_steps * mpi_size * number_of_walkers.
tau = 0.10 # Imaginary-time step between branchings (Ha^-1).
alat = 0.10 # The lattice discretization parameter (i.e. grid size) used for discretized the Hamiltonian and potential. The lattice spacing is alat * a0, where a0 is the Bohr radius.
non_local_move = "tmove" # The treatment of the non-local term in the Effective core potential. tmove (T-move) and dltmove (Determinant locality approximation with T-move) are available.
num_gfmc_warmup_steps = 25 # Number of initial observable measurements discarded as warmup.
num_gfmc_bin_blocks = 10 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_gfmc_bin_blocks, not num_gfmc_bin_blocks * mpi_size * number_of_walkers.
num_gfmc_collect_steps = 5 # Number of measurement (before binning) for collecting the weights.
atomic_force = true
use_swct = false # Apply Space Warp Coordinate Transformation (SWCT) to atomic forces. Default is false for LRDMC.
```

```bash
mpirun -np 128 jqmc lrdmc.toml > out_lrdmc 2> out_lrdmc.e
```

The 50000 branching steps took 447 s (`Total GFMC time for 50000 branching
steps =  446.540 sec.`, `Survived walkers ratio = 99.70 %`).

### Analyze (LRDMC)

```bash
grep -A10 "Final output" out_lrdmc
jqmc-tool lrdmc compute-energy restart.h5 -w 25 -b 10 -c 5
jqmc-tool lrdmc compute-force restart.h5 -w 25 -b 10 -c 5
```

```text
Final output(s):
  Total Energy: E = -1.17450 +- 0.00002 Ha.
  Variance: Var = 0.00878 +- 0.00001 Ha^2.
  Atomic Forces:
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  H       -0(9)e-05    -6(7)e-05    +0.00036(17)
  H       +0.00022(9)  -0(9)e-05    -0.00039(20)
  ------------------------------------------------
...
Total number of binned samples = 10
E = -1.1745032717407375 +- 2.200266246610315e-05 Ha.
...
Total number of binned samples = 10
Atomic Forces:
------------------------------------------------
Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
------------------------------------------------
H       -0(9)e-05    -6(7)e-05    +0.00036(17)
H       +0.00022(9)  -0(9)e-05    -0.00039(20)
------------------------------------------------
```

For LRDMC, `Total number of binned samples` equals the number of bins (10).
Without SWCT (`use_swct = false`) the two atoms' forces are estimated
separately, so they are not exactly opposite: here $F_z$ is +0.00036(17)
Ha/bohr on the first atom and -0.00039(20) Ha/bohr on the second, opposite
within the error bars. Entries such as `-0(9)e-05` are zero with an error
of $9\times10^{-5}$ Ha/bohr.

> [!NOTE]
> The LRDMC forces are intrinsically biased because the so-called Reynolds approximation[^1986REYijqc] is employed. See benchmark papers[^2022NAKjcp] [^2021TIHjcp].

```bash
cd ..
```

## Convert JSD to JAGP and optimize: JAGP (VMC)

Convert the optimized JSD ansatz to the JAGP ansatz using `jqmc-tool`, and
then optimize all variational parameters including the geminal (lambda)
parameters. For a two-electron singlet the AGP pairing function is a general
symmetric two-electron function in the basis set (it contains the full
configuration-interaction wave function in that basis), so it is much more
flexible than one doubly occupied orbital and the JAGP energy should be
lower.

```bash
cd 05vmc_JAGP
python ../../copy_optimized_hamiltonian.py ../02vmc_JSD hamiltonian_data_JSD.h5
jqmc-tool hamiltonian conv-wf --convert-to jagp hamiltonian_data_JSD.h5
```

```text
Convert SD to AGP.
Hamiltonian data is saved in hamiltonian_data_conv.h5.
```

```bash
mv hamiltonian_data_conv.h5 hamiltonian_data.h5
```

<!-- include: 05vmc_JAGP/vmc.toml -->
```toml
[control]
job_type = "vmc" # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low", "high", "devel", "mpi-low", "mpi-high", "mpi-devel"

[vmc]
num_mcmc_steps = 500 # Number of observable measurement steps per MPI process and walker. The global total is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates between successive observable measurements.
num_mcmc_warmup_steps = 30 # Number of initial observable measurements discarded as warmup.
num_mcmc_bin_blocks = 10 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 1.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # Epsilon parameter for Attaccalite-Sorella regularization.
num_opt_steps = 200 # Number of optimization steps.
wf_dump_freq = 1 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
opt_J1_param = true
opt_J2_param = true
opt_J3_param = true
opt_geminal = true
fix_geminal_rank = 0
opt_J3_basis_exp = false
opt_J3_basis_coeff = false
opt_lambda_basis_exp = false
opt_lambda_basis_coeff = false
optimizer_kwargs = { method = "sr", delta = 0.15, epsilon = 0.001, cg_flag = false, cg_max_iter = 10000, use_lm = true }
```

The JAGP wave function has 904 variational parameters (900 of them in the
lambda matrix). `cg_flag = false` makes the SR step use the direct linear
solver: with 128 ranks it was about six times faster per optimization step
than the conjugate-gradient solver (which needed ~480 iterations per step)
and gave the same energies.

```bash
mpirun -np 128 jqmc vmc.toml > out_vmc 2> out_vmc.e
jqmc-tool vmc analyze-output out_vmc
```

The 200 steps took 1325 s. Output (excerpt). Entries such as `+0.053(0)`
are not zero errors: the log of the reference run printed the force error
with three fixed decimals, so an error below 5e-4 was written as `0.000`:

```text
------------------------------------------------------
Iter     E (Ha)     Max generalized force (Ha/a.u.)  Max force S/N
------------------------------------------------------
   1  -1.17255(20)   +0.053(0)   145.059
   2  -1.17279(17)   +0.047(0)   143.578
   3  -1.17301(16)    +0.04(0)   133.849
   4  -1.17290(15)  -0.0350(10)   128.190
   5  -1.17335(14)   +0.031(0)   120.800
 ...
 191  -1.17427(11)   -0.001(0)     3.280
 192  -1.17402(11)   -0.001(0)     3.434
 193  -1.17447(11)  +0.0020(10)     4.485
 194  -1.17420(11)   -0.001(0)     3.288
 195  -1.17408(11)  +0.0020(10)     4.292
 196  -1.17415(11)  -0.0020(10)     4.966
 197  -1.17398(11)   +0.001(0)     3.851
 198  -1.17414(11)   -0.001(0)     4.415
 199  -1.17418(11)  -0.0010(10)     4.753
 200  -1.17430(11)   -0.001(0)     5.742
------------------------------------------------------
```

The energy drops by about 1.7 mHa with respect to JSD and is converged after
about 50 steps; `Max force S/N` stays around 3-6.

```bash
cd ..
```

## Compute Energy and Atomic forces: JAGP (MCMC)

Using the optimized JAGP wavefunction, compute the energy and atomic forces via MCMC.

```bash
cd 06mcmc_JAGP
python ../../copy_optimized_hamiltonian.py --average-last 20 ../05vmc_JAGP hamiltonian_data.h5
```

> [!NOTE]
> The helper averages the wave-function parameters of the last 20 dumps (here steps 181-200) with
> `jqmc-tool vmc average-wf`; see [Averaging the last optimization steps](../../doc/notes/wavefunction_optimization.md). The numbers quoted below were not sampled from this average: the reference run handed over the single last dump with
> `python ../../copy_optimized_hamiltonian.py ../05vmc_JAGP hamiltonian_data.h5`.
> The reference VMC run also wrote a dump only every 10 steps; the shipped VMC input sets `wf_dump_freq = 1` because this hand-off needs one dump per step (the dump frequency does not change the optimization).

<!-- include: 06mcmc_JAGP/mcmc.toml -->
```toml
[control]
job_type = "mcmc" # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low", "high", "devel", "mpi-low", "mpi-high", "mpi-devel"

[mcmc]
num_mcmc_steps = 10000 # Number of observable measurement steps per MPI process and walker. The global total is num_mcmc_steps * mpi_size * number_of_walkers.
num_mcmc_per_measurement = 40 # Number of MCMC updates between successive observable measurements.
num_mcmc_warmup_steps = 30 # Number of initial observable measurements discarded as warmup.
num_mcmc_bin_blocks = 10 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.
Dt = 1.2 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # Epsilon parameter for Attaccalite-Sorella regularization.
atomic_force = true
use_swct = true # Apply Space Warp Coordinate Transformation (SWCT) to atomic forces.
```

```bash
mpirun -np 128 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e
grep -A10 "Final output" out_mcmc
jqmc-tool mcmc compute-energy restart.h5 -w 30 -b 10
jqmc-tool mcmc compute-force restart.h5 -w 30 -b 10
```

```text
Final output(s):
  Total Energy: E = -1.17416 +- 0.00002 Ha.
  Variance: Var = 0.00291 +- 0.00001 Ha^2.
  Atomic Forces:
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  H       -1.4(2.1)e-05 +3.3(2.1)e-05 -0.00087(13)
  H       +1.4(2.1)e-05 -3.3(2.1)e-05 +0.00087(13)
  ------------------------------------------------
...
Total number of binned samples = 5120
E = -1.1741571784058846 +- 2.3993188171638907e-05 Ha.
...
Total number of binned samples = 5120
Atomic Forces:
------------------------------------------------
Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
------------------------------------------------
H       -1.4(2.1)e-05 +3.3(2.1)e-05 -0.00087(13)
H       +1.4(2.1)e-05 -3.3(2.1)e-05 +0.00087(13)
------------------------------------------------
```

The variance is three times smaller than with JSD (0.00291 vs 0.00879
Ha$^2$).

```bash
cd ..
```

## Compute Energy and Atomic forces: JAGP (LRDMC)

Using the same optimized JAGP wavefunction, compute the energy and atomic forces via LRDMC.

```bash
cd 07lrdmc_JAGP
python ../../copy_optimized_hamiltonian.py --average-last 20 ../05vmc_JAGP hamiltonian_data.h5
```

> [!NOTE]
> The helper averages the wave-function parameters of the last 20 dumps (here steps 181-200) with
> `jqmc-tool vmc average-wf`; see [Averaging the last optimization steps](../../doc/notes/wavefunction_optimization.md). The numbers quoted below were not sampled from this average: the reference run handed over the single last dump with
> `python ../../copy_optimized_hamiltonian.py ../05vmc_JAGP hamiltonian_data.h5`.
> The reference VMC run also wrote a dump only every 10 steps; the shipped VMC input sets `wf_dump_freq = 1` because this hand-off needs one dump per step (the dump frequency does not change the optimization).

<!-- include: 07lrdmc_JAGP/lrdmc.toml -->
```toml
[control]
job_type = "lrdmc-tau" # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 4 # Number of walkers per MPI process
max_time = 86400 # Maximum time in sec.
restart = false
restart_chk = "restart.h5" # Restart checkpoint file. If restart is True, this file is used.
hamiltonian_h5 = "hamiltonian_data.h5" # Hamiltonian checkpoint file. If restart is False, this file is used.
verbosity = "low" # Verbosity level. "low", "high", "devel", "mpi-low", "mpi-high", "mpi-devel"

[lrdmc-tau]
num_mcmc_steps = 50000 # Number of observable measurement steps per MPI process and walker. The global total is num_mcmc_steps * mpi_size * number_of_walkers.
tau = 0.10 # Imaginary-time step between branchings (Ha^-1).
alat = 0.10 # The lattice discretization parameter (i.e. grid size) used for discretized the Hamiltonian and potential. The lattice spacing is alat * a0, where a0 is the Bohr radius.
non_local_move = "tmove" # The treatment of the non-local term in the Effective core potential. tmove (T-move) and dltmove (Determinant locality approximation with T-move) are available.
num_gfmc_warmup_steps = 25 # Number of initial observable measurements discarded as warmup.
num_gfmc_bin_blocks = 10 # Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_gfmc_bin_blocks, not num_gfmc_bin_blocks * mpi_size * number_of_walkers.
num_gfmc_collect_steps = 5 # Number of measurement (before binning) for collecting the weights.
atomic_force = true
use_swct = false # Apply Space Warp Coordinate Transformation (SWCT) to atomic forces. Default is false for LRDMC.
```

The input is the same as for JSD. The run below took 495 s (`Total GFMC
time for 50000 branching steps =  494.565 sec.`, `Survived walkers ratio =
99.87 %`).

```bash
mpirun -np 128 jqmc lrdmc.toml > out_lrdmc 2> out_lrdmc.e
grep -A10 "Final output" out_lrdmc
jqmc-tool lrdmc compute-energy restart.h5 -w 25 -b 10 -c 5
jqmc-tool lrdmc compute-force restart.h5 -w 25 -b 10 -c 5
```

```text
Final output(s):
  Total Energy: E = -1.17448 +- 0.00002 Ha.
  Variance: Var = 0.00290 +- 0.00000 Ha^2.
  Atomic Forces:
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  H       +2(6)e-05    -4(6)e-05    -0.00058(17)
  H       +4(7)e-05    -7(5)e-05    +0.00060(17)
  ------------------------------------------------
...
Total number of binned samples = 10
E = -1.1744791809736885 +- 1.6340639081171998e-05 Ha.
...
Total number of binned samples = 10
Atomic Forces:
------------------------------------------------
Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
------------------------------------------------
H       +2(6)e-05    -4(6)e-05    -0.00058(17)
H       +4(7)e-05    -7(5)e-05    +0.00060(17)
------------------------------------------------
```

> [!NOTE]
> The LRDMC forces are intrinsically biased because the so-called Reynolds approximation[^1986REYijqc] is employed. See benchmark papers[^2022NAKjcp] [^2021TIHjcp].

```bash
cd ..
```

## Summary

H$_2$ at $R = 0.74\;\text{\AA}$, cc-pVTZ (Cartesian), all electron; 128 MPI
ranks x 4 walkers; exact energy $-1.174475$ Ha. $F_z$ is the force on the
first atom (at $-z$).

| Ansatz | Method | E (Ha)       | Var (Ha$^2$) | $F_z$ (Ha/bohr) |
|--------|--------|--------------|--------------|-----------------|
| JSD    | VMC    | -1.17246(4)  | 0.00879(1)   | +0.00173(14)    |
| JSD    | LRDMC  | -1.17450(2)  | 0.00878(1)   | +0.00036(17)    |
| JAGP   | VMC    | -1.17416(2)  | 0.00291(1)   | -0.00087(13)    |
| JAGP   | LRDMC  | -1.17448(2)  | 0.00290(0)   | -0.00058(17)    |

In both the JSD and JAGP calculations, `opt_geminal = true`, so the
determinantal parameters are optimized together with the Jastrow parameters.

How to read the energies. Both VMC energies lie above the exact value, the
JSD one by 2.0 mHa and the JAGP one by 0.3 mHa. Both LRDMC energies agree
with the exact value within statistics (JSD 1.3 error bars below it, JAGP
0.2 error bars below it; `jqmc-tool` values -1.174503(22) and -1.174479(16) Ha), as
expected for a nodeless ground state; the JAGP trial wave function, with a
three times smaller variance, also gives a smaller LRDMC error bar for the
same run length.

How to read the forces. $R = 0.74$ Angstrom is slightly shorter than the
experimental bond length (0.7414 Angstrom), so the exact force is small and
pushes the atoms apart: with the harmonic force constant of H$_2$
($k \approx 0.37$ Ha/bohr$^2$) one estimates $F_z \approx -0.0010$ Ha/bohr
on the first atom. The JAGP VMC value, -0.00087(13) Ha/bohr, agrees with this
estimate. The JSD VMC value, +0.00173(14) Ha/bohr, has the opposite sign: it
is the force of the JSD trial at this geometry, whose own equilibrium bond
length differs from the exact one. The LRDMC forces are biased by the
Reynolds approximation (see the notes above), so they still depend on the
trial wave function: with the JAGP trial they push the atoms apart, like the
estimate (-0.00058(17) and +0.00060(17) Ha/bohr on the first and second
atom), while with the JSD trial they keep the sign of the JSD VMC force but
are smaller (+0.00036(17) and -0.00039(20) Ha/bohr). The better JAGP trial
gives the more reliable LRDMC force.

If one were to keep the determinant fixed (`opt_geminal = false`), the DFT
orbitals would not be stationary with respect to the VMC energy and a finite
self-consistency bias could appear in the atomic forces[^2024SLOjctc]. An
alternative correction without full orbital optimization is described in
Ref. [^2024NAKprb].

[^2010SORjcp]: S. Sorella and L. Capriotti, J. Chem. Phys. **133** 234111 (2010) [https://doi.org/10.1063/1.3516208](https://doi.org/10.1063/1.3516208)
[^2022NAKjcp]: [K. Nakano et al., J. Chem. Phys. **156**, 034101 (2022)](https://doi.org/10.1063/5.0076302).
[^2021TIHjcp]: [J. Tiihonen et al., J. Chem. Phys. **154**, 204111 (2021)](https://doi.org/10.1063/5.0052266).
[^1986REYijqc]: [P. J. Reynolds et al., Int. J. Quantum Chem. **29**, 589 (1986)](https://doi.org/10.1002/qua.560290403).
[^2024SLOjctc]: E. Slootman et al. J. Chem. Theory Comput. **20**, 6020-6027 (2024), [https://doi.org/10.1021/acs.jctc.4c00498](https://doi.org/10.1021/acs.jctc.4c00498)
[^2024NAKprb]: K. Nakano et al. Phys. Rev. B **109**, 205151 (2024), [https://doi.org/10.1103/PhysRevB.109.205151](https://doi.org/10.1103/PhysRevB.109.205151)
