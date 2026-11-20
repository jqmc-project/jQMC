# jqmc-example03: Water with geminal and Jastrow optimization

Optimization of the determinant (MO coefficients) together with the Jastrow
factors for the water molecule, using the `opt_geminal` interface. Starting
from the same PySCF Hartree-Fock (HF) calculation as `jqmc-example01`, the
example first optimizes a Jastrow-only trial wave function (HF orbitals
fixed), then switches on `opt_geminal = true` and optimizes the orbitals as
well, and finally compares the VMC energies of the two trial wave functions
with the same MCMC setup.

## Before you start

- You need PySCF with TREXIO export
  (`python -c "from pyscf.tools import trexio"` must succeed).
- All numbers below come from runs with 128 MPI ranks on one 128-core CPU
  node. Wall times on that node:
  Jastrow-only VMC 19 min (150 steps), geminal VMC 83 min (200 steps),
  each MCMC about 9 min.
- The analysis commands, the meaning of the main TOML keys and the restart
  rules are summarized in the examples index (`examples/README.md`).

All paths below are relative to `examples/jqmc-example03`; follow the displayed `cd`
commands.

## Generate a trial WF

jQMC reads mean-field results through the TREXIO file format. The following
script (identical to `jqmc-example01`) runs an HF calculation for the water
molecule with PySCF and writes a TREXIO file.

<!-- include: 01DFT/run_pyscf.py -->
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
cd 01DFT
python run_pyscf.py
cd ..
```

`01DFT/water.out` reports the HF energy:

```text
converged SCF energy = -16.9437549661298
```

## Step 1: optimize the Jastrow factors only (VMC)

Convert the TREXIO file into a jQMC wave function with a two-body Jastrow
(`-j2 1.0`, initial parameter) and a three-body Jastrow on a compact AO
basis (`-j3 ao-small`, e.g. `3s` for H, `3s1p` for O; see the command-line
reference for `ao`, `ao-medium`, `ao-large`, `ao-full`, `mo`).

```bash
cd 02vmc_jastrow
cp ../01DFT/water_ccecp_ccpvtz.h5 .
jqmc-tool trexio convert-to water_ccecp_ccpvtz.h5 -j2 1.0 -j3 ao-small
```

```text
pbc_flag = False
num_ele_up = 4, num_ele_dn = 4, spin_polarized = False
spin_dependent MOs = False
Cartesian basis functions.
Hamiltonian data is saved in hamiltonian_data.h5.
```

In this step only the J2 parameter and the J3 matrix are optimized
(`opt_geminal = false`); the determinant keeps the HF orbitals.
Every optimization step samples 500 measurements per walker (4 walkers per
rank), discards 30 and uses 10 bins; `use_lm = true` (with the default
`lm_subspace_dim = 0`) selects accelerated SR (aSR).

<!-- include: 02vmc_jastrow/vmc.toml -->
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
Dt = 2.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # Epsilon parameter for Attaccalite-Sorella regularization.
num_opt_steps = 150 # Number of optimization steps.
wf_dump_freq = 1 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
optimizer_kwargs = { method = "sr", delta = 0.15, epsilon = 0.001, cg_flag = true, cg_max_iter = 10000, use_lm = true } # SR optimizer configuration (method plus step/regularization).
opt_J1_param = false
opt_J2_param = true
opt_J3_param = true
opt_geminal = false
fix_geminal_rank = 0
opt_J3_basis_exp = false
opt_J3_basis_coeff = false
opt_lambda_basis_exp = false
opt_lambda_basis_coeff = false
```

```bash
mpirun -np 128 jqmc vmc.toml > out_vmc 2> out_vmc.e
jqmc-tool vmc analyze-output out_vmc
```

```text
------------------------------------------------------
Iter     E (Ha)     Max generalized force (Ha/a.u.)  Max force S/N
------------------------------------------------------
   1  -16.6225(48)  +1.3170(60)   254.362
   2  -16.7217(43)  +1.1840(50)   245.330
   3  -16.8193(39)  +1.0580(50)   227.848
...
 146  -17.1949(16)  +0.0060(40)     2.494
 147  -17.1937(16)  +0.0120(60)     3.264
 148  -17.1963(16)  +0.0030(10)     2.255
 149  -17.1928(15)  -0.0110(40)     2.520
 150  -17.1954(16)  +0.0050(30)     3.104
------------------------------------------------------
```

The log reports `Number of variational parameters = 157.` (1 J2 + 156 J3).
The energy is flat within the error bar over the last steps and
`Max force S/N` is around 2-3 (< 4-5), so the Jastrow-only optimization is
converged. `jqmc-tool vmc analyze-output out_vmc -s vmc.jpg` saves a plot of
the curve.

## Step 2: optimize the orbitals and the Jastrow factors (VMC, `opt_geminal`)

Start from the last Jastrow-only wave function and switch on the
determinant optimization.

```bash
cd ../03vmc_geminal
python ../../copy_optimized_hamiltonian.py ../02vmc_jastrow hamiltonian_data.h5
```

```text
Copied optimized trial: ../02vmc_jastrow/hamiltonian_data_opt_step_150.h5 -> hamiltonian_data.h5
Selection is by saved step, not a test of scientific convergence.
```

The only differences from `02vmc_jastrow/vmc.toml` are `opt_geminal = true`
and `num_opt_steps = 200`:

- `opt_geminal = true` optimizes the determinant's MO coefficients together
  with J2 and J3. The TREXIO conversion produces a single Slater determinant;
  with `fix_geminal_rank = 0` the number of occupied orbitals stays fixed and
  only their coefficients change. This moves the nodal surface of the trial
  wave function, which a Jastrow-only optimization cannot do. (Older inputs
  that used `opt_with_projected_MOs` or `opt_lambda_param` must be migrated
  to `opt_geminal`.)

<!-- include: 03vmc_geminal/vmc.toml -->
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
Dt = 2.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # Epsilon parameter for Attaccalite-Sorella regularization.
num_opt_steps = 200 # Number of optimization steps.
wf_dump_freq = 1 # Frequency of wavefunction (i.e. hamiltonian_data) dump.
optimizer_kwargs = { method = "sr", delta = 0.15, epsilon = 0.001, cg_flag = true, cg_max_iter = 10000, use_lm = true } # SR optimizer configuration (method plus step/regularization).
opt_J1_param = false
opt_J2_param = true
opt_J3_param = true
opt_geminal = true
fix_geminal_rank = 0
opt_J3_basis_exp = false
opt_J3_basis_coeff = false
opt_lambda_basis_exp = false
opt_lambda_basis_coeff = false
```

```bash
mpirun -np 128 jqmc vmc.toml > out_vmc 2> out_vmc.e
jqmc-tool vmc analyze-output out_vmc
```

```text
------------------------------------------------------
Iter     E (Ha)     Max generalized force (Ha/a.u.)  Max force S/N
------------------------------------------------------
   1  -17.1943(16)  +0.126(12)    15.294
   2  -17.1949(15)  +0.112(39)    17.083
   3  -17.2020(15)  +0.1060(70)    16.886
   4  -17.2029(14)  +0.100(17)    16.052
   5  -17.2065(14)  -0.083(27)    14.396
...
 196  -17.2171(13)  +0.026(12)     3.034
 197  -17.2197(14)  +0.0190(90)     2.599
 198  -17.2179(13)  +0.024(11)     2.769
 199  -17.2174(13)  +0.042(32)     3.626
 200  -17.2167(13)  -0.038(22)     2.874
------------------------------------------------------
```

The number of variational parameters grows from 157 to 413 (the log
reports `Number of variational parameters = 413.`): the extra 256 are the
MO coefficients, which the log shows as the update block
`block=mo_coefficients_up size=256` (the conversion reported
`spin_dependent MOs = False`, so there is no separate down-spin block).
Step 1 of this run starts at the Jastrow-only energy (-17.1943(16) Ha); the
energy drops by about 20 mHa within the first ~10 steps and then stays
around -17.217 Ha. `Max force S/N` falls from about 15 to about 3 by the
end of the run.

> If the optimization does not converge well, try the following:
> - Adjust `delta` in `optimizer_kwargs`. A smaller `delta` (e.g., `0.05`) is more conservative but stable; a larger one (e.g., `0.30`) is more aggressive but may cause instabilities.
> - Set `use_lm = false` in `optimizer_kwargs` to switch off aSR and use plain SR with the fixed step `delta`.
> - Determinant parameters have derivatives that diverge near the nodal surface. If the optimization is noisy, a small `epsilon_AS > 0` (Attaccalite-Sorella reweighting; `jqmc-example04` uses `0.05` for its JAGP step) can stabilize it. This example runs with `epsilon_AS = 0.0`.

If an optimization has not converged, restart it with `restart = true` in
`[control]`; `num_opt_steps` then counts **additional** steps (see
"Restarting a run" in the examples index), and both outputs are analyzed
together with `jqmc-tool vmc analyze-output out_vmc out_vmc_cont`.

## Step 3: compare the energies (MCMC)

Run the same MCMC input (100 walkers per rank, 10000 measurements, 30 warmup,
10 bins) for both trial wave functions.

<!-- include: 04mcmc_jastrow/mcmc.toml -->
```toml
[control]
job_type = "mcmc" # Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".
mcmc_seed = 34456 # Random seed for MCMC
number_of_walkers = 100 # Number of walkers per MPI process
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
Dt = 2.0 # Step size for the MCMC update (bohr).
epsilon_AS = 0.0 # Epsilon parameter for Attaccalite-Sorella regularization.
```

`05mcmc_geminal/mcmc.toml` is identical.

Jastrow-only trial (HF orbitals):

```bash
cd ../04mcmc_jastrow
python ../../copy_optimized_hamiltonian.py --average-last 20 ../02vmc_jastrow hamiltonian_data.h5
mpirun -np 128 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e
grep -A2 "Final output" out_mcmc
jqmc-tool mcmc compute-energy restart.h5 -w 30 -b 10
```

> [!NOTE]
> The helper averages the wave-function parameters of the last 20 dumps (here steps 131-150) with
> `jqmc-tool vmc average-wf`; see [Averaging the last optimization steps](../../doc/notes/wavefunction_optimization.md). The numbers quoted below were not sampled from this average: the reference run handed over the single last dump with
> `python ../../copy_optimized_hamiltonian.py ../02vmc_jastrow hamiltonian_data.h5`.

```text
Final output(s):
  Total Energy: E = -17.19387 +- 0.00007 Ha.
  Variance: Var = 0.42656 +- 0.00078 Ha^2.
```

```text
Read restart checkpoint file(s) from restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 128000
E = -17.1938684065841 +- 6.838095707855326e-05 Ha.
```

Trial with optimized orbitals:

```bash
cd ../05mcmc_geminal
python ../../copy_optimized_hamiltonian.py --average-last 20 ../03vmc_geminal hamiltonian_data.h5
mpirun -np 128 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e
grep -A2 "Final output" out_mcmc
jqmc-tool mcmc compute-energy restart.h5 -w 30 -b 10
```

> [!NOTE]
> The helper averages the wave-function parameters of the last 20 dumps (here steps 181-200) with
> `jqmc-tool vmc average-wf`; see [Averaging the last optimization steps](../../doc/notes/wavefunction_optimization.md). The numbers quoted below were not sampled from this average: the reference run handed over the single last dump with
> `python ../../copy_optimized_hamiltonian.py ../03vmc_geminal hamiltonian_data.h5`.

```text
Final output(s):
  Total Energy: E = -17.21680 +- 0.00006 Ha.
  Variance: Var = 0.35156 +- 0.00081 Ha^2.
```

```text
Read restart checkpoint file(s) from restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 128000
E = -17.216803234544454 +- 5.8779783969021e-05 Ha.
```

The `jqmc-tool` line re-analyzes the checkpoint with the TOML's warmup and
bin values; the number of binned samples is 10 bins x 100 walkers x 128
ranks = 128000.

| Trial wave function | VMC energy (Ha) | Variance (Ha^2) |
|---|---|---|
| HF orbitals + J2/J3 (`04mcmc_jastrow`) | -17.19387(7) | 0.42656(78) |
| optimized orbitals + J2/J3 (`05mcmc_geminal`) | -17.21680(6) | 0.35156(81) |

With the same Jastrow form (J2 + J3 `ao-small`) and the same MCMC setup,
optimizing the orbitals with `opt_geminal = true` lowers the VMC energy by
22.9(1) mHa and reduces the variance by about 18%. Both trial wave functions
can be used as the starting point of an LRDMC calculation (see
`jqmc-example01`).
