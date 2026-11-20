# jqmc-example09: twist-averaged silicon stress and MPC in mixed precision

This tutorial runs an explicit eight-point twist average for bulk silicon,
first with MCMC and then with LRDMC, and computes the BZ-averaged energy,
stress, pressure and the passive MPC finite-size energy correction in mixed
precision. It finishes with an `alat -> 0` extrapolation of the
twist-averaged LRDMC energy and a restart of a completed twist directory.

The trial wavefunction is the converter's **unoptimized** QE determinant on
a B-spline basis with a single Pade J2 (`-j2 1.0`, no J1 or J3). That keeps
the tutorial focused on the twist-averaged route; the energies, forces and
stresses below are therefore not variationally optimized results and are not
converged physical numbers for Si. A twist-averaged optimization of the same
system (shared Jastrow, per-twist determinants) is shown in
`jqmc-workflow-example06`.

## Before you start

All paths below are relative to `examples/jqmc-example09`; follow the
displayed `cd` commands. Keep the supplied TOML files; do not generate default
TOMLs over them.

Requirements specific to this example:

- **MPI with a multiple of eight ranks.** An eight-twist Hamiltonian needs an
  MPI world size that is a multiple of eight (each twist gets an equal group
  of ranks), so a serial run is not possible.
- **Quantum ESPRESSO `pw.x` built with HDF5**, so that it writes the
  `wfc*.hdf5` files the converter reads.

The outputs shown below come from one reference run of every step: 128 MPI
ranks on one 128-core CPU node (16 ranks x 8 walkers = 128 walkers per twist;
CPU only, `OMP_NUM_THREADS=1`). Wall times:

| Step | Wall time (128 ranks) | of which sampling |
|---|---|---|
| 1. QE `pw.x`, 8 k points | 1.5 min | |
| 2. `jqmc-tool qe --kind bspline --twist-average` (1 process) | 3 s | |
| 3. MCMC, energy + force + stress, 1000 steps | 19 min | 17 min |
| 4. LRDMC `alat = 0.30`, 2000 branching steps | 18 min | 16 min |
| 5. LRDMC `alat = 0.20` / `0.40`, 2000 steps each | 22 / 15 min | 21 / 14 min |
| Restart of the completed MCMC directory | 10 s | |

The rest of each job is start-up (imports, reading the Hamiltonian, JIT
compilation) and the one-time MPC table setup for `mpc_cutoff = 30.0` Ha,
which rank 0 of each twist group builds and broadcasts; the log reports it
(`MPC setup: done in 26.2 sec.` in the MCMC run).

Precision policy: `[precision].mode = "full"` (fp64 everywhere) is the jQMC
default and the reference setting. This example opts into `mode = "mixed"` to
reduce the cost of the eight-twist sampling; see
[`doc/notes/mixed_precision.md`](../../doc/notes/mixed_precision.md) for which
parts stay in fp64. To produce a reference value, rerun with `mode = "full"`
and compare.

## What this example covers

- QE multi-k output to one multi-twist jQMC Hamiltonian (B-spline orbitals).
- Eight complex, non-Gamma Bloch twists with their QE weights.
- MPI decomposition into one equal-size communicator per twist.
- `[precision].mode = "mixed"` on a production-style input.
- MCMC and `lrdmc-tau` stress/pressure with `compute_stress = true`.
- Per-twist MPC estimates and their BZ-weighted correction and total energy.
- Per-twist checkpoint directories, post-processing and restart.

The system is the same two-atom diamond-Si primitive cell and ccECP UPF as
`jqmc-example07`, with the same pure B-spline orbital representation
(`--kind bspline`); this example uses a Pade J2 instead of the DTN Jastrow.

## Directory layout

```text
jqmc-example09/
  01DFT/si_twist.in
  02mcmc_twist_stress/mcmc.toml
  03lrdmc_twist_stress/lrdmc_tau.toml
```

QE wavefunction files and generated jQMC HDF5 files are not shipped; the
steps below write them.

## 1. Generate all eight QE twists

Run from `jqmc-example09/01DFT`. The input reads
`jqmc-example07/01DFT/Si.ccECP.upf`, which is not shipped: download it from
<https://pseudopotentiallibrary.org/recipes/Si/ccECP/Si.ccECP.upf> into `jqmc-example07/01DFT/` first (see `jqmc-example07`).

```bash
cd 01DFT
mkdir -p out
mpirun -np 128 pw.x -in si_twist.in > si_twist.out
ls out/silicon_twist.save/wfc*.hdf5
```

```text
!    total energy              =     -15.66056909 Ry
     PWSCF        :     12.88s CPU   1m25.50s WALL
```

`nosym = .true.` and `noinv = .true.` retain the full shifted 2x2x2 grid, so
the save directory contains `wfc1.hdf5` through `wfc8.hdf5`. The calculation
uses fixed occupations; jQMC does not convert fractionally occupied bands.

Two properties of this grid matter for the QMC results below:

- It contains time-reversal pairs (k and -k), which give the same energy
  (compare twists 0 and 7 in the tables below), so half of the eight twists
  add no new information but still double the QMC cost. The tutorial accepts
  this to get a fixed twist count and a deterministic MPI launch size.
- The shifted 2x2x2 grid of the fcc reciprocal lattice is **not** invariant
  under the full cubic point group: two twists lie on the [111] axis and
  the other six do not. The twist-averaged system therefore only keeps the
  three-fold symmetry around [111], and the atomic forces are not forced to
  vanish by symmetry. Running the same QE input with `tprnfor = .true.`
  gives `+-0.00132 Ry/bohr` along [111] on the two atoms; the QMC forces in
  step 3 also point along [111]. Use a symmetric twist set for production.

## 2. Convert the QE grid to one twist-average input

```bash
cd ../02mcmc_twist_stress
jqmc-tool qe ../01DFT/out/silicon_twist.save \
      --twist-average \
      --kind bspline \
      --jastrow-2b-parameter 1.0 \
      --output hamiltonian_twists.h5 > out_convert 2>&1
head -4 out_convert; tail -2 out_convert
```

```text
Auto-discovered 1 UPF entries from ../01DFT/out/silicon_twist.save/data-file-schema.xml: Si=../01DFT/out/silicon_twist.save/Si.ccECP.upf
QE multi-twist convert: 8 k-points found in ../01DFT/out/silicon_twist.save/data-file-schema.xml
Loaded QE reference from ../01DFT/out/silicon_twist.save: 2 atoms, 8 bands, 4450 G-vectors, ecutwfc=50.000 Ha.
PW->B-spline: order 5, grid (34, 34, 34) from QE FFT grid (45, 45, 45) (mesh_factor 0.750), 8 bands, control-point table 0.005 GiB (complex128), mean residual 4.831e-12, max 2.057e-11.
QE multi-twist convert: wrote combined HDF5 (n_twist=8, sum_weights=2.000000) -> hamiltonian_twists.h5
Twist-average Hamiltonian data is saved in hamiltonian_twists.h5.
```

The `PW->B-spline` line is repeated for every twist; the mean and maximum
residuals (here of order 1e-11) measure how well the spline reproduces the QE
orbitals on the construction mesh.

The output is a multi-twist file with shared structure/Jastrow data and one
determinant payload per twist.

Why `--kind bspline`: it interpolates the QE orbitals on a real-space mesh
instead of projecting them onto Gaussians, so no fit is involved, the
conversion of all eight twists takes seconds, and one orbital evaluation is a
local B-spline blend. The spline is still an interpolation: the default
`--mesh-factor 0.75` keeps the spline Nyquist frequency at 1.5x the PW cutoff,
and you should converge the VMC energy and variance against the mesh before
treating the spline determinant as equivalent to the QE one (see
`jqmc-example07`). `--kind pgto+pw` is the other lossless choice; it is more
expensive per evaluation.

```bash
jqmc-tool hamiltonian show-info hamiltonian_twists.h5
```

```text
Twist-averaged input: 8 twists (sum of BZ weights = 2.000000).
  index  k-point (reduced or Cartesian, as written by the converter)   weight
      0  ( 0.250000,  0.250000,  0.250000)                           0.250000
      1  ( 0.250000,  0.250000, -0.250000)                           0.250000
      2  ( 0.250000, -0.250000,  0.250000)                           0.250000
      3  ( 0.250000, -0.250000, -0.250000)                           0.250000
      4  (-0.250000,  0.250000,  0.250000)                           0.250000
      5  (-0.250000,  0.250000, -0.250000)                           0.250000
      6  (-0.250000, -0.250000,  0.250000)                           0.250000
      7  (-0.250000, -0.250000, -0.250000)                           0.250000
Showing twist 0:
==================================================================
...
**SD_data
  occupied MO rows (up, dn) = (4, 4)
  num_electron_up = 4
  num_electron_dn = 4
  time_reversal = True
...
**AOs_bspline_complex_data
  Number of AOs (bands) = 8
  k_point (fractional) = (0.25000000000000006, 0.25000000000000006, 0.25000000000000006)
  spline grid shape (N0, N1, N2) = (34, 34, 34)
  spline order (degree) = 5
...
**Jastrow_two_body_data
  Jastrow 2b param = 1.0
  2b Jastrow functional form is the pade type.
==================================================================
```

`show-info` prints the twist table (index, k point and weight of each twist)
followed by the summary of one twist, by default twist 0. Add `--twist N` to
inspect twist `N` instead. The converter stores the QE k-point weights
unchanged (here 0.25 each, sum 2); the driver normalizes them only when
forming the BZ mean.

Do not add `twist_average` or `twist_index` to the calculation TOML. The
`/twists` schema in `hamiltonian_twists.h5` selects the twist-averaged driver
automatically; an ordinary Hamiltonian file always selects a single-k
calculation. The `--twist-average` option above belongs to the converter: it
chooses which kind of Hamiltonian file to create.

## 3. MCMC energy, force and stress in mixed precision

<!-- include: 02mcmc_twist_stress/mcmc.toml -->
```toml
[control]
job_type = "mcmc"
mcmc_seed = 34456
number_of_walkers = 8
max_time = 86400
restart = false
# A directory, not one HDF5 file, on the twist-averaged route.
restart_chk = "restart_twists"
hamiltonian_h5 = "hamiltonian_twists.h5"
# The /twists schema in hamiltonian_h5 selects twist averaging.
verbosity = "high"
[mcmc]
num_mcmc_steps = 1000
num_mcmc_per_measurement = 40
num_mcmc_warmup_steps = 30
num_mcmc_bin_blocks = 10
Dt = 1.5
epsilon_AS = 0.0
atomic_force = true
compute_stress = true
use_swct = true
parameter_derivatives = false

[coulomb]
mpc = true
mpc_cutoff = 30.0

[precision]
mode = "mixed"
```

`atomic_force = true` (with `use_swct = true`) adds the atomic forces to the
stress. A stress-only run (`atomic_force = false`, `compute_stress = true`) is
also supported on complex twists.

```bash
mpirun -np 128 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e
```

With 128 ranks each twist receives 16 ranks (`The number of MPI process = 16.`
in the log); with 8 ranks each rank would own one twist.

```bash
grep -A50 "Final output" out_mcmc
```

```text
Final output(s):
  BZ-averaged Energy: E_TA = -7.96597 +- 0.00051 Ha (s_twist = 0.16480 Ha).
  BZ-averaged MPC correction: dE_MPC = +0.15950 +- 0.00016 Ha (cutoff = 30.0 Ha).
  BZ-averaged Total Energy (MPC-corrected): E_MPC = -7.80647 +- 0.00062 Ha.
  Per-twist energy (Level-1 outputs):
  ------------------------------------------------
  twist  k_point                 E_k (Ha)
  ------------------------------------------------
      0  (+0.250,+0.250,+0.250)  -7.6814(15)
      1  (+0.250,+0.250,-0.250)  -8.0580(15)
      2  (+0.250,-0.250,+0.250)  -8.0598(14)
      3  (+0.250,-0.250,-0.250)  -8.0622(14)
      4  (-0.250,+0.250,+0.250)  -8.0626(14)
      5  (-0.250,+0.250,-0.250)  -8.0620(14)
      6  (-0.250,-0.250,+0.250)  -8.0620(14)
      7  (-0.250,-0.250,-0.250)  -7.6797(15)
  ------------------------------------------------
  BZ-averaged Atomic Forces:
  ------------------------------------------------
  atom    Fx(Ha/bohr)  Fy(Ha/bohr)  Fz(Ha/bohr)
  ------------------------------------------------
     0    -0.0142(23)  -0.0142(23)  -0.0127(23)
     1    +0.0121(23)  +0.0185(23)  +0.0105(23)
  ------------------------------------------------
  Stress Tensor (sigma):
  ------------------------------------------------
  Component   Value (Ha/bohr^3)
  ------------------------------------------------
  sigma_xx    +0.002169(8)
  sigma_yy    +0.002175(8)
  sigma_zz    +0.002167(8)
  sigma_xy    +3(9)e-06
  sigma_xz    +7(9)e-06
  sigma_yz    -4(9)e-06
  ------------------------------------------------
  Pressure (P = -diag(sigma)):
  ------------------------------------------------
  Component   Value (Ha/bohr^3)
  ------------------------------------------------
  P_x         -0.002169(8)
  P_y         -0.002175(8)
  P_z         -0.002167(8)
  P_hydro     -0.002170(5)
  ------------------------------------------------
```

### Reading the result

- `E_TA` is the BZ-weighted Ewald energy. `s_twist` (0.165 Ha here) is the
  physical dispersion among twists; it is reported separately from the
  Monte Carlo error bar. The two [111] twists (0 and 7) lie 0.38 Ha above the
  other six.
- `dE_MPC` and `E_MPC` are the BZ-averaged MPC correction and corrected total
  energy. The corrected-energy error is formed from the joint
  `E_Ewald + dV_MPC` jackknife inside each twist before the per-twist error
  bars are combined at the BZ level; it is not reconstructed from two
  separate error bars.
- The forces are equal and opposite on the two atoms and lie along [111],
  as allowed by the reduced symmetry of this twist set (step 1).
- The stress is in Ha/bohr^3; `P = -diag(sigma)`. Its value belongs to the
  unoptimized J2 trial and is not a physical pressure of Si.

To recompute the BZ-averaged energy and the forces from the checkpoint
directory (for example with different binning), pass the directory itself,
using the warmup and bin counts of `mcmc.toml`:

```bash
jqmc-tool mcmc compute-energy restart_twists -b 10 -w 30
```

```text
Read restart checkpoint file(s) from restart_twists.
Found 8 twist checkpoints.
E = -7.965973450963581 +- 0.0005121241783141448 Ha.
MPC correction: dE_MPC = 0.1595034843082801 +- 0.0001639166589259954 Ha.
Total Energy (MPC-corrected): E_MPC = -7.806469966655301 +- 0.0006178066278773538 Ha.
```

```bash
jqmc-tool mcmc compute-force restart_twists -b 10 -w 30
```

```text
Read restart checkpoint file(s) from restart_twists.
Atomic Forces:
------------------------------------------------
Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
------------------------------------------------
Si      -0.0142(23)  -0.0142(23)  -0.0127(23)
Si      +0.0121(23)  +0.0185(23)  +0.0105(23)
------------------------------------------------
Hydrostatic Pressure:
(per-twist pressure summaries stored by the run at its own warmup/bin settings; -b/-w/-c do not apply)
P_hydro = -0.002170158179977045 +- 5.2789990630819e-06 Ha/bohr^3.
```

The full stress tensor is printed only in the live log above.

MPC is a passive energy observable. Sampling, acceptance, forces, and stress
remain on the Ewald Hamiltonian even when `mpc = true`. `mpc_cutoff` is a
reciprocal-space convergence parameter in Hartree, so repeat the calculation
at larger cutoffs until the BZ-averaged correction is stable. The checkpoint
stores all per-twist MPC summaries and the cutoff; changing `mpc_cutoff`
makes those summaries stale, so use a fresh checkpoint directory for the
cutoff-convergence scan.

### Restart

`restart_chk = "restart_twists"` names a directory on this route. It contains
one file per twist (`twist_000.h5` ... `twist_007.h5`) plus
`shared_state.h5`, not one monolithic checkpoint. To resume, set
`restart = true` in `mcmc.toml` and launch with the same twists, the same BZ
weights and a compatible MPI world size. A resume refuses a directory written
for another twist count or other weights (jqmc-tool would combine its stored
weights); use a fresh directory instead. Completed twist groups are skipped; when all
eight are complete, the run only re-reads their summaries. Doing that on a
copy of the finished directory above (`restart = true`, `mpirun -np 128`)
took 10 s and
printed the same `Final output(s):` block (same energies, forces and stress).

## 4. LRDMC energy and stress in mixed precision

Copy the Hamiltonian and run the GFMC_t input:

```bash
cd ../03lrdmc_twist_stress
cp ../02mcmc_twist_stress/hamiltonian_twists.h5 .
mpirun -np 128 jqmc lrdmc_tau.toml > out_lrdmc 2> out_lrdmc.e
```

<!-- include: 03lrdmc_twist_stress/lrdmc_tau.toml -->
```toml
[control]
job_type = "lrdmc-tau"
mcmc_seed = 34467
number_of_walkers = 8
max_time = 86400
restart = false
# A directory, not one HDF5 file, on the twist-averaged route.
restart_chk = "restart_twists"
hamiltonian_h5 = "hamiltonian_twists.h5"
# The /twists schema in hamiltonian_h5 selects twist averaging.
verbosity = "high"
[lrdmc-tau]
num_mcmc_steps = 2000
tau = 0.10
alat = 0.30
non_local_move = "tmove"
num_gfmc_warmup_steps = 50
num_gfmc_bin_blocks = 50
num_gfmc_collect_steps = 20
atomic_force = false
compute_stress = true
use_swct = false
epsilon_PW = 0.0

[coulomb]
mpc = true
mpc_cutoff = 30.0

[precision]
mode = "mixed"
```

`out_lrdmc` ends with the same kind of `Final output(s):` block as the MCMC
step:

```text
Survived walkers ratio = 98.53 %
Average of the number of projections  = 27
...
Final output(s):
  BZ-averaged Energy: E_TA = -8.03658 +- 0.00095 Ha (s_twist = 0.16252 Ha).
  BZ-averaged MPC correction: dE_MPC = +0.14445 +- 0.00027 Ha (cutoff = 30.0 Ha).
  BZ-averaged Total Energy (MPC-corrected): E_MPC = -7.89213 +- 0.00115 Ha.
  Per-twist energy (Level-1 outputs):
  ------------------------------------------------
  twist  k_point                 E_k (Ha)
  ------------------------------------------------
      0  (+0.250,+0.250,+0.250)  -7.7555(25)
      1  (+0.250,+0.250,-0.250)  -8.1302(27)
      2  (+0.250,-0.250,+0.250)  -8.1298(25)
      3  (+0.250,-0.250,-0.250)  -8.1340(24)
      4  (-0.250,+0.250,+0.250)  -8.1308(23)
      5  (-0.250,+0.250,-0.250)  -8.1274(28)
      6  (-0.250,-0.250,+0.250)  -8.1302(33)
      7  (-0.250,-0.250,-0.250)  -7.7547(29)
  ------------------------------------------------
  Stress Tensor (sigma):
  ------------------------------------------------
  Component   Value (Ha/bohr^3)
  ------------------------------------------------
  sigma_xx    +0.002092(11)
  sigma_yy    +0.002064(12)
  sigma_zz    +0.002052(11)
  sigma_xy    +7.5(1.4)e-05
  sigma_xz    +5.6(1.4)e-05
  sigma_yz    +7.2(1.4)e-05
  ------------------------------------------------
  Pressure (P = -diag(sigma)):
  ------------------------------------------------
  Component   Value (Ha/bohr^3)
  ------------------------------------------------
  P_x         -0.002092(11)
  P_y         -0.002064(12)
  P_z         -0.002052(11)
  P_hydro     -0.002069(8)
  ------------------------------------------------
```

The LRDMC energy `E_TA = -8.0366(10)` Ha lies 70 mHa below the MCMC value
`-7.9660(5)` Ha of the same trial wavefunction. To recompute the energy from
the checkpoint directory with the binning of `lrdmc_tau.toml`:

```bash
jqmc-tool lrdmc compute-energy restart_twists -b 50 -w 50 -c 20
```

```text
Read restart checkpoint file(s) from restart_twists.
Found 8 twist checkpoints.
E = -8.03658310177796 +- 0.0009519215412618834 Ha.
MPC correction: dE_MPC = 0.14445307766413712 +- 0.00026869966004806336 Ha.
Total Energy (MPC-corrected): E_MPC = -7.892130024113824 +- 0.001149129979850813 Ha.
```

`-b`, `-w` and `-c` are the bin count, the discarded warmup measurements and
`num_gfmc_collect_steps` (the number of measurements over which the
branching weights are collected before binning); use the values of the run.

A single `alat = 0.30` result is not a continuum-limit result: the walkers
are propagated with the lattice-regularized Hamiltonian, whose error vanishes
only as `alat -> 0`, while the energy and stress are measured with the
continuum local energy. See "Propagation and measurement local energies" in
[`doc/notes/lrdmc.md`](../../doc/notes/lrdmc.md). The LRDMC MPC correction is
the passive mixed estimator sampled by the Ewald propagation; branching,
forces, and stress are not changed to MPC.

## 5. Extrapolate the twist-averaged energy

Prepare two more **fresh** runs from the same Hamiltonian, at `alat = 0.20`
and `0.40`; the `alat = 0.30` run of step 4 is the third point. Starting in
`03lrdmc_twist_stress`, the following creates a new scan directory and refuses
to overwrite an existing one. Each run has a different seed fixed before
sampling; do not choose seeds after inspecting the energies.

```bash
cd ..
python - <<'PY'
from pathlib import Path
import shutil
import tomlkit

source = Path("03lrdmc_twist_stress")
scan = Path("04lrdmc_alat_scan")
scan.mkdir(exist_ok=False)
for index, alat in enumerate((0.20, 0.40)):
    run = scan / f"alat_{alat:.2f}"
    run.mkdir()
    job = tomlkit.parse((source / "lrdmc_tau.toml").read_text())
    job["control"]["restart"] = False
    job["control"]["mcmc_seed"] = 34468 + index
    job["lrdmc-tau"]["alat"] = alat
    (run / "lrdmc_tau.toml").write_text(tomlkit.dumps(job))
    shutil.copy2(source / "hamiltonian_twists.h5", run / "hamiltonian_twists.h5")
PY
```

Run them one after the other, or submit each directory as a separate job:

```bash
for spacing in 0.20 0.40; do
    (cd "04lrdmc_alat_scan/alat_${spacing}" && \
        mpirun -np 128 jqmc lrdmc_tau.toml > out_lrdmc 2> out_lrdmc.e) || break
done
```

The smaller spacing needs more projections per branching step
(`Average of the number of projections` = 61, 27 and 16 at `alat` = 0.20,
0.30 and 0.40) and is correspondingly slower.

After **all twists in all three runs have completed**, post-process in a
single process. Pass the checkpoint directories themselves, not a wildcard
over their `twist_*.h5` files:

```bash
cd 04lrdmc_alat_scan
jqmc-tool lrdmc extrapolate-energy \
    alat_0.20/restart_twists ../03lrdmc_twist_stress/restart_twists alat_0.40/restart_twists \
    -p 1 -b 50 -w 50 -c 20 -s lrdmc_extrapolation.png > out_extrapolate 2>&1
grep -E "For a|Polynomial|Extrapolation of|Graph" out_extrapolate
```

```text
  For a = 0.2 bohr: E = -8.035272817402424 +- 0.000893204409924052 Ha.
  For a = 0.2 bohr: E_MPC = -7.890853310901468 +- 0.0010921847106981838 Ha.
  For a = 0.3 bohr: E = -8.03658310177796 +- 0.0009519215412618834 Ha.
  For a = 0.3 bohr: E_MPC = -7.892130024113824 +- 0.001149129979850813 Ha.
  For a = 0.4 bohr: E = -8.036250580835276 +- 0.0009312791912728943 Ha.
  For a = 0.4 bohr: E_MPC = -7.89191983428457 +- 0.0011484160857156906 Ha.
Extrapolation of the energy with respect to a^2.
  Polynomial order = 1.
  For a -> 0 bohr: E = -8.03532792942767 +- 0.0011529000684979266 Ha.
  For a -> 0 bohr: E_MPC = -7.890849060993272 +- 0.0014095976411083947 Ha.
Graph is saved in lrdmc_extrapolation.png.
```

The `a -> 0` value is the constant term of a least-squares polynomial fit in
`alat**2`, and its error bar is propagated from the per-spacing error bars in
closed form, so repeated calls give identical output.

The CLI computes the independently weighted BZ energy and error at each
`alat`, then fits those estimates versus `alat**2`; it rejects incomplete
twist sets or mismatched k points/weights. Here the three energies agree
within their error bars, so a linear fit in `alat**2` (`-p 1`) is used; the
default `-p 2` (quadratic in `alat**2`) has as many parameters as three points
and leaves no check of the model. Add spacings before trusting a fit order.

Because all inputs enable MPC with the same `mpc_cutoff = 30.0` Ha, the
command extrapolates **both the Ewald energy `E` and the MPC-corrected energy
`E_MPC`**, with separate labelled outputs and plot series. The corrected
error bars include the covariance of the sampled energy and MPC correction.
Partial MPC histories or inconsistent cutoffs are errors. If MPC is disabled
in every scan input, only `E` is extrapolated.

This does not extrapolate forces or stress. Lattice-spacing extrapolation
also does not replace convergence checks in MPC cutoff, twist grid, supercell
size, orbital representation, trial-wavefunction optimization and Monte Carlo
statistics; no converged Si result is claimed by this tutorial.

## Orbital tiling

Twist averaging in this tutorial keeps the primitive physical cell.
For a larger physical supercell, the [tiling recipe](../../doc/examples.md#jqmc-example14-frozen-orbitals-in-a-tiled-supercell)
groups complete primitive k cosets into supertwists before averaging. The
number and weights of the resulting sectors differ from the input k list.
