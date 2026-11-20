# jqmc-example10: unrestricted open-shell UHF and LSDA trials

This example covers the unrestricted open-shell paths. The main path uses a
molecular PySCF UHF nitrogen atom (quartet, ccECP pseudopotential); the
final section gives a periodic Quantum ESPRESSO LSDA counterpart (an
isolated Si atom in a box, B-spline orbitals). jQMC keeps independent
alpha/beta determinant blocks in both cases.

## Before you start

All paths below are relative to `examples/jqmc-example10`; follow the
displayed `cd` commands. Keep the supplied script and TOML files; do not
generate default TOMLs over them.

Requirements specific to this example:

- PySCF with its TREXIO interface for step 1, and a Quantum ESPRESSO `pw.x`
  built with HDF5 for step 5.
- The launch lines are written with `mpirun -np N`. Every TOML uses 64
  walkers per MPI process, so the number of walkers and samples grows with
  `N`.
- All outputs quoted below were obtained with 128 MPI ranks on one 128-core
  CPU node. Wall times on that node: PySCF
  UHF 1 s (serial), VMC optimization 39 min, nitrogen MCMC 1.5 min,
  `pw.x` 1.5 min, Si LSDA MCMC 6 min.

Precision policy: `[precision].mode = "full"` (fp64 everywhere) is the jQMC
default and the reference setting. The VMC optimization in step 3 runs in
`full`; both sampling runs (steps 4 and 5) opt into `mode = "mixed"` to show
that the open-shell path samples in the cheaper mode. In mixed mode the
orbital (AO/MO) evaluation may run in fp32, while determinant evaluation,
accumulation of sums and the final estimators stay in fp64; see
[`doc/notes/mixed_precision.md`](../../doc/notes/mixed_precision.md).

## 1. Generate the UHF TREXIO file

```bash
cd 01SCF
python run_pyscf.py
```

<!-- include: 01SCF/run_pyscf.py -->
```python
"""Generate an unrestricted open-shell nitrogen TREXIO trial (ccECP)."""

from pyscf import gto, scf
from pyscf.tools import trexio

mol = gto.Mole()
mol.verbose = 5
mol.atom = "N 0.0 0.0 0.0"
mol.basis = "ccecp-ccpvdz"
mol.ecp = "ccecp"
mol.unit = "angstrom"
mol.charge = 0
# Five valence electrons; N_up - N_down = 3: the atomic 4S ground-state spin sector.
mol.spin = 3
mol.symmetry = False
mol.cart = True
mol.output = "nitrogen_uhf.out"
mol.build()

mf = scf.UHF(mol)
mf.max_cycle = 200
mf.kernel()
if not mf.converged:
    raise RuntimeError("PySCF UHF did not converge")

trexio.to_trexio(mf, "nitrogen_uhf_ccecp_ccpvdz.h5")
```

The important choices are `mol.spin = 3` and `scf.UHF(mol)`. This is not a
restricted open-shell calculation: the alpha and beta orbitals are stored as
separate TREXIO spin rows and remain variationally independent in jQMC.
`mol.ecp = "ccecp"` replaces the two 1s core electrons of nitrogen by the
ccECP pseudopotential, and `ccecp-ccpvdz` is the matching cc-pVDZ-quality
valence basis. Five valence electrons with `mol.spin = 3` give four up-spin
and one down-spin electron.

`nitrogen_uhf.out` ends with

```text
  alpha nocc = 4  HOMO = -0.571848050256278  LUMO = 0.985117853470698
  beta  nocc = 1  HOMO = -0.716969247137816  LUMO = 0.160918657925062
...
converged SCF energy = -9.63736792340325  <S^2> = 3.7552436  2S+1 = 4.002621
```

(PySCF also prints a `WARN: HOMO ... >= LUMO ...` line on the screen for an
early SCF iteration; the converged state above is the 4-up/1-down quartet.)

## 2. Convert to a jQMC Hamiltonian

```bash
cd ../02vmc_open_shell
jqmc-tool trexio convert-to ../01SCF/nitrogen_uhf_ccecp_ccpvdz.h5 \
      --output hamiltonian_data.h5 \
      --jastrow-2b-parameter 1.0
jqmc-tool hamiltonian show-info hamiltonian_data.h5
```

```text
pbc_flag = False
num_ele_up = 4, num_ele_dn = 1, spin_polarized = True
spin_dependent MOs = True
Cartesian basis functions.
Hamiltonian data is saved in hamiltonian_data.h5.
```

`show-info` confirms the pseudopotential and the two independent determinant
blocks (abridged):

```text
**Coulomb_potential_data
  ecp_flag = True
**SD_data
  occupied MO rows (up, dn) = (4, 1)
  num_electron_up = 4
  num_electron_dn = 1
  time_reversal = False
**MOs_data
  Number of MOs = 4
  dim. of MOs coeff = (4, 14)
...
**MOs_data
  Number of MOs = 1
  dim. of MOs coeff = (1, 14)
...
**Jastrow_two_body_data
  Jastrow 2b param = 1.0
  2b Jastrow functional form is the pade type.
```

The ccECP is finite at the nucleus, so there is no electron-nucleus cusp to
impose and the trial needs only the electron-electron Jastrow factor (J2).
A three-body Jastrow (`-j3 ao-small` or `-j3 mo`, see jqmc-example03 and
jqmc-example01) can be added the same way; it is left out here to keep the
optimization small (71 parameters).

J2 is spin-independent in jQMC: the same Pade function is used for parallel-
and antiparallel-spin pairs, so it imposes the antiparallel electron-electron
cusp (1/2) on every pair; the parallel-spin cusp (1/4) is not enforced. This
does not make the local energy diverge, because the antisymmetric
determinant already vanishes when two same-spin electrons meet.

## 3. Optimize J2 and both UHF determinant blocks

<!-- include: 02vmc_open_shell/vmc.toml -->
```toml
[control]
job_type = "vmc"
mcmc_seed = 34456
number_of_walkers = 64
max_time = 86400
restart = false
restart_chk = "restart.h5"
hamiltonian_h5 = "hamiltonian_data.h5"
verbosity = "high"

[vmc]
num_mcmc_steps = 500
num_mcmc_per_measurement = 20
num_mcmc_warmup_steps = 30
num_mcmc_bin_blocks = 10
Dt = 1.5
epsilon_AS = 0.0
num_opt_steps = 100
wf_dump_freq = 1
optimizer_kwargs = { method = "sr", delta = 0.05, epsilon = 0.001, cg_flag = true, cg_max_iter = 10000, use_lm = false }

opt_J1_param = false
opt_J2_param = true
opt_J3_param = false
opt_JDTN_u_param = false
opt_JDTN_chi_param = false
opt_JDTN_f_param = false

# UHF has independent alpha and beta MO blocks; both are optimized.
opt_geminal = true
fix_geminal_rank = 0
opt_J3_basis_exp = false
opt_J3_basis_coeff = false
opt_lambda_basis_exp = false
opt_lambda_basis_coeff = false

[precision]
mode = "full"
```

```bash
mpirun -np N jqmc vmc.toml > out_vmc 2> out_vmc.e
jqmc-tool vmc analyze-output out_vmc -s vmc.jpg
```

`opt_geminal = true` exposes both `mo_coefficients_up` and
`mo_coefficients_dn` because this trial is unrestricted; the log lists them
as separate derivative sources:

```text
Number of variational parameters = 71.
Optimization batch: route=host, local samples=30080, sampler ranks=128, real coordinates=71, storage parameters=71.
  Derivative source j2_param: O shape=(30080,), D shape=None, policy=direct_real, tangent=euclidean.
  Derivative source mo_coefficients_up: O shape=(30080, 4, 14), D shape=None, policy=direct_real, tangent=s_stiefel_rows.
  Derivative source mo_coefficients_dn: O shape=(30080, 1, 14), D shape=None, policy=direct_real, tangent=s_stiefel_rows.
```

A restricted open-shell (ROHF) TREXIO source instead ties the paired core
and does not have two independent spin blocks to optimize.

`analyze-output` prints one row per step and saves the plot to `vmc.jpg`
(abridged). The `(0)` force errors of the first steps are not zero errors:
the optimizer log of the reference run printed the force error with three
fixed decimals, so an error below 5e-4 was written as `0.000` (and
`-0.0180(10)` is an error of 0.001); the current log prints at least two significant digits of the
error.

```text
------------------------------------------------------
Iter     E (Ha)     Max generalized force (Ha/a.u.)  Max force S/N
------------------------------------------------------
   1  -9.63028(57)    +0.16(0)   465.612
   2  -9.65883(50)   +0.181(0)   428.366
   3  -9.67525(46)   +0.173(0)   381.813
   4  -9.68716(44)   +0.157(0)   338.335
   5  -9.69397(42)   +0.139(0)   296.003
...
  10  -9.70969(39)   +0.069(0)   157.487
...
  20  -9.71285(37)  -0.0180(10)    42.358
...
  30  -9.71413(37)  -0.0080(10)    10.891
...
  40  -9.71380(37)  +0.011(10)     3.726
...
  50  -9.71428(37)  +0.0050(30)     2.453
...
  96  -9.71444(37)  +0.0060(20)     2.831
  97  -9.71443(37)  -0.0060(30)     3.318
  98  -9.71440(37)  -0.0100(50)     2.179
  99  -9.71385(37)  -0.0060(20)     2.868
 100  -9.71430(37)  -0.0050(20)     2.297
------------------------------------------------------
Graph is saved in vmc.jpg.
```

The energy drops by about 80 mHa in the first 20 steps (orbital relaxation
plus the electron-electron correlation from J2) and is flat within about
1 mHa afterwards. `Max force S/N` falls from about 470 to 2-4 after step
40, i.e. the remaining generalized forces are at the noise level of the
500-step sampling. The 100 steps took 2316 s (about 23 s per step: 10 s
sampling and 13 s SR solve).

## 4. Sample the optimized open-shell trial (mixed precision)

```bash
cd ../03mcmc_open_shell
python ../../copy_optimized_hamiltonian.py --average-last 20 ../02vmc_open_shell hamiltonian_data.h5
mpirun -np N jqmc mcmc.toml > out_mcmc 2> out_mcmc.e
jqmc-tool mcmc compute-energy restart.h5 -b 10 -w 30
```

> [!NOTE]
> The helper averages the wave-function parameters of the last 20 dumps (here steps 81-100) with
> `jqmc-tool vmc average-wf`; see [Averaging the last optimization steps](../../doc/notes/wavefunction_optimization.md). The numbers quoted below were not sampled from this average: the reference run handed over the single last dump with
> `python ../../copy_optimized_hamiltonian.py ../02vmc_open_shell hamiltonian_data.h5`, which printed
> `Copied optimized trial: ../02vmc_open_shell/hamiltonian_data_opt_step_100.h5 -> hamiltonian_data.h5`.

<!-- include: 03mcmc_open_shell/mcmc.toml -->
```toml
[control]
job_type = "mcmc"
mcmc_seed = 34467
number_of_walkers = 64
max_time = 86400
restart = false
restart_chk = "restart.h5"
hamiltonian_h5 = "hamiltonian_data.h5"
verbosity = "high"

[mcmc]
num_mcmc_steps = 5000
num_mcmc_per_measurement = 20
num_mcmc_warmup_steps = 30
num_mcmc_bin_blocks = 10
Dt = 1.5
epsilon_AS = 0.0
atomic_force = false
compute_stress = false
use_swct = true
parameter_derivatives = false

[precision]
mode = "mixed"
```

```text
Total elapsed time for MCMC 5000 steps. = 40.85 sec.
...
Acceptance ratio is 62.89 %.  Ideal is ~ 50.00%. Adjust Dt.
...
Final output(s):
  Total Energy: E = -9.71384 +- 0.00014 Ha.
  Variance: Var = 0.29258 +- 0.00115 Ha^2.
```

```text
Read restart checkpoint file(s) from restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 81920
E = -9.713837700886586 +- 0.00013728704840062876 Ha.
```

`compute-energy` reads `restart.h5`, discards the first 30 measurements per
walker (`-w 30`, equal to `num_mcmc_warmup_steps`) and forms 10 bins per
walker and process (`-b 10`): 10 x 64 walkers x 128 processes = 81920 binned
samples.

The VMC energy, -9.71384(14) Ha, is 76 mHa below the UHF energy
(-9.63737 Ha, step 1). The variance is 0.293 Ha^2. For comparison, the
same `mcmc.toml` (with `mode = "full"`) on the unoptimized trial of step 2
gave

```text
  Total Energy: E = -9.63068 +- 0.00018 Ha.
  Variance: Var = 0.35614 +- 0.00022 Ha^2.
```

so the optimization of J2 and both determinant blocks lowers the energy by
83 mHa and the variance by about 18%.

As a precision check, the optimized run with `mode = "full"` (same seed and
walkers) gave

```text
  Total Energy: E = -9.71402 +- 0.00011 Ha.
  Variance: Var = 0.29192 +- 0.00113 Ha^2.
```

which agrees with the mixed-precision energy within 1.0 combined standard
error.

## 5. Periodic QE LSDA counterpart (B-spline, mixed precision)

`04QE_LSDA/si_atom_lsda.in` is a complete `nspin = 2` input for an isolated
Si atom in a periodic box. It reads the ccECP UPF from
`jqmc-example07/01DFT/Si.ccECP.upf`, which is not shipped: download it from
<https://pseudopotentiallibrary.org/recipes/Si/ccECP/Si.ccECP.upf> into `jqmc-example07/01DFT/` first. The four
valence electrons and `tot_magnetization = 2` give three up-spin and one
down-spin electron.

<!-- include: 04QE_LSDA/si_atom_lsda.in -->
```text
&CONTROL
    calculation   = 'scf'
    prefix        = 'si_atom_lsda'
    outdir        = './out'
    pseudo_dir    = '../../jqmc-example07/01DFT'
    verbosity     = 'high'
    disk_io       = 'medium'
/
&SYSTEM
    ! Isolated Si atom in a periodic box. The ccECP leaves four valence
    ! electrons; tot_magnetization=2 gives N_up=3 and N_down=1.
    ibrav         = 1
    celldm(1)     = 12.0
    nat           = 1
    ntyp          = 1
    nbnd          = 6
    ecutwfc       = 100.0
    ecutrho       = 400.0
    occupations   = 'fixed'
    nspin         = 2
    tot_magnetization = 2.0
    input_dft     = 'pz'
/
&ELECTRONS
    conv_thr         = 1.0d-9
    mixing_beta      = 0.4
    electron_maxstep = 200
/
ATOMIC_SPECIES
    Si  28.085  Si.ccECP.upf

ATOMIC_POSITIONS {bohr}
    Si  6.0  6.0  6.0

K_POINTS {gamma}
```

```bash
cd ../04QE_LSDA
mkdir -p out
pw.x -in si_atom_lsda.in > si_atom_lsda.out
```

(We ran `pw.x` with `mpirun -np 128`.)

The SCF needs 90 iterations (the up-spin p shell is split between two
occupied and one empty, nearly degenerate, orbital) and ends with

```text
 ------ SPIN UP ------------
          k = 0.0000 0.0000 0.0000 ( 14520 PWs)   bands (ev):
   -10.2514  -3.3238  -3.3238  -3.3101  -0.2622   1.8459
     occupation numbers
     1.0000   1.0000   1.0000   0.0000   0.0000   0.0000

 ------ SPIN DOWN ----------
          k = 0.0000 0.0000 0.0000 ( 14520 PWs)   bands (ev):
    -8.7691  -2.5273  -1.7446  -1.7446   0.0457   2.3187
     occupation numbers
     1.0000   0.0000   0.0000   0.0000   0.0000   0.0000

     highest occupied, lowest unoccupied level (ev):    -3.3238   -3.3101
!    total energy              =      -7.44981687 Ry
     total magnetization       =     2.00 Bohr mag/cell
     convergence has been achieved in  90 iterations
```

i.e. an LSDA total energy of -3.72491 Ha. QE writes `wfcup1.hdf5` and
`wfcdw1.hdf5` into `out/si_atom_lsda.save`; the converter reads them as
independent determinant channels:

```bash
jqmc-tool qe out/si_atom_lsda.save \
      --kind bspline \
      --jastrow-2b-parameter 1.0 \
      --output hamiltonian_data.h5
```

```text
Auto-discovered 1 UPF entries from out/si_atom_lsda.save/data-file-schema.xml: Si=out/si_atom_lsda.save/Si.ccECP.upf
Loaded QE reference from out/si_atom_lsda.save (spin=up): 1 atoms, 6 bands, 29039 G-vectors, ecutwfc=50.000 Ha.
Loaded QE reference from out/si_atom_lsda.save (spin=dw): 1 atoms, 6 bands, 29039 G-vectors, ecutwfc=50.000 Ha.
LSDA (nspin=2) save detected: n_occ_up=3, n_occ_dn=1; occupied-only source ranges are 3 up bands and 1 down bands.

PW->B-spline: order 5, grid (60, 60, 60) from QE FFT grid (80, 80, 80) (mesh_factor 0.750), 3 bands, control-point table 0.010 GiB (complex128), mean residual 6.867e-13, max 7.674e-13.
PW->B-spline: order 5, grid (60, 60, 60) from QE FFT grid (80, 80, 80) (mesh_factor 0.750), 1 bands, control-point table 0.003 GiB (complex128), mean residual 5.244e-13, max 5.244e-13.

build_wavefunction_from_projection(kind=bspline, LSDA): 3 AOs, 3 MOs, ne_up=3, ne_dn=1.
build_hamiltonian_from_wavefunction: pbc=True, ne_up=3, ne_dn=1, j1=off, j2=pade@1.0.

Hamiltonian data is saved in hamiltonian_data.h5.
```

`--kind bspline` interpolates the occupied QE orbitals on a periodic
quintic B-spline mesh (default `mesh-factor = 0.75`); `--kind pgto+pw` is
the other recommended QE target. No J1 is attached because Si is
described by an ECP. Fixed integer occupations are mandatory: fractional
occupations are rejected because a fixed-electron-number Slater determinant
cannot represent them.

Then sample:

```bash
mpirun -np N jqmc mcmc.toml > out_mcmc 2> out_mcmc.e
jqmc-tool mcmc compute-energy restart.h5 -b 10 -w 30
```

<!-- include: 04QE_LSDA/mcmc.toml -->
```toml
[control]
job_type = "mcmc"
mcmc_seed = 34478
number_of_walkers = 64
max_time = 86400
restart = false
restart_chk = "restart.h5"
hamiltonian_h5 = "hamiltonian_data.h5"
verbosity = "high"

[mcmc]
num_mcmc_steps = 5000
num_mcmc_per_measurement = 20
num_mcmc_warmup_steps = 30
num_mcmc_bin_blocks = 10
Dt = 3.0
epsilon_AS = 0.0
atomic_force = false
compute_stress = false
use_swct = true
parameter_derivatives = false

[precision]
mode = "mixed"
```

```text
Total elapsed time for MCMC 5000 steps. = 296.08 sec.
...
Acceptance ratio is 54.53 %.  Ideal is ~ 50.00%. Adjust Dt.
...
Final output(s):
  Total Energy: E = -3.72552 +- 0.00005 Ha.
  Variance: Var = 0.06539 +- 0.00003 Ha^2.
```

```text
Read restart checkpoint file(s) from restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 81920
E = -3.725518980367447 +- 5.10337363497993e-05 Ha.
```

There is no VMC step on this path: the sampled trial is the converter's
**unoptimized** LSDA determinant with a Pade J2 (`-j2 1.0`). Its energy,
-3.72552(5) Ha, lies only 0.6 mHa below the LSDA total energy, so this run
is a check that the LSDA conversion and mixed-precision sampling work, not
a physics result; optimize the trial (as in step 3) before using it. The
12-bohr box is tutorial-scale: for an isolated-atom result, increase it
until image interactions are converged.

## Orbital tiling

For periodic LSDA tiling, the [tiling recipe](../../doc/examples.md#jqmc-example14-frozen-orbitals-in-a-tiled-supercell)
retains independent spin coefficients and explicitly uses the supercell spin
sectors (+k_s,-k_s). The atomic/open-boundary optimization here stays untiled;
a nontrivial periodic tiling needs a complete primitive k coset.
