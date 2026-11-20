# jqmc-workflow-example05: All-electron Hydrogen via QE -> VMC -> MCMC -> LRDMC

End-to-end **workflow-API** counterpart to the manual TOML tutorial in
`examples/jqmc-example08`: same QE input, same conversion
(`pgto+pw`, `ecutwfc = "auto"`, DTN `u_chi` Jastrow), same optimization scope
(DTN Jastrow + MO coefficients) and the same production settings for forces.
Here `jqmc_workflow` (Launcher + state machine + pilot runs + automatic
continuation + output parsing) generates and submits the jQMC inputs instead
of hand-written TOML files.

Differences from `jqmc-example08`:

- VMC: two production runs of 100 optimization steps each (the second restarts
  from the first one's checkpoint), i.e. 200 steps as in example08, but with
  SR `delta = 0.10` (example08: `0.15`) and the number of MCMC samples per step
  chosen by a pilot run for a target error of 5 mHa per step (example08: fixed
  300 measurements per rank, about 2 mHa per step).
- MCMC/LRDMC: the number of steps is chosen by a pilot run so that the energy
  error reaches `TARGET_PROD_ERROR = 5e-4` Ha, instead of a fixed 20000 steps;
  LRDMC uses 30 warmup steps and 10 bins (example08: 50 and 50).

## Reference run

Every number below comes from one run of the shipped
`run_pipelines.py` and `qe-pbc/h_scf.in`, except for the VMC dump frequency
and the hand-off of the optimized trial (see the note after the VMC stage), with `SERVER`/`QUEUE_LABEL` pointing
at a PBS queue that gives one whole 128-core CPU node (no GPU) per job, i.e.
**128 MPI ranks per job**, 3 h wall-time limit. Wall times of the individual
jobs (from the PBS accounting files the workflow saves as
`job_accounting_*.txt`):

| Stage | Job | Wall time |
|-------|-----|-----------|
| QE `pw.x` (QE 7.5, HDF5; run by hand) | 32 MPI ranks | 17 s |
| `QE_Workflow` (`jqmc-tool qe`, local) | serial, login node | about 30 s |
| `VMC_Workflow` pilot (20 opt steps x 50 MCMC steps) | 128 ranks | 2 min |
| `VMC_Workflow` production run 1 (steps 1-100, 78 MCMC steps/step) | 128 ranks | 11.5 min |
| `VMC_Workflow` production run 2 (steps 101-200) | 128 ranks | 11.5 min |
| `MCMC_Workflow` pilot (100 steps) + production (3652 steps) | 128 ranks | 0.6 + 1.3 min |
| `LRDMC_Workflow` pilot (100 steps) + production (8189 steps) | 128 ranks | 0.5 + 3.3 min |

MCMC and LRDMC run concurrently (both depend only on VMC). The whole pipeline
took 2 h 15 min from launch to the summary table on a shared cluster, most of
it queue waiting; the node time is about 31 min. `number_of_walkers = 1`, so
with fewer MPI ranks the pilot runs ask for proportionally more steps.

## System

| Property        | Value                                       |
|-----------------|---------------------------------------------|
| Atoms           | 4 all-electron H (two H2 molecules, bond 1.4 bohr) |
| Cell            | Simple orthorhombic, a = 5.00, b = 6.00, c = 7.00 bohr |
| k-point         | (0.25, 0.25, 0.25) crystal -- general twist (complex MOs) |
| Potential       | 1/r AE Coulomb (`H.coulomb-ae.UPF`)         |
| DFT             | LDA-PZ from Quantum ESPRESSO, `ecutwfc = 800` Ry |
| Basis           | pGTO + PW (`ecutwfc = "auto"`)              |
| Trial Jastrow   | DTN polynomial, `u_chi` preset (no f)       |
| AE e-N cusp     | Enforced by the DTN `chi` term              |

The all-electron 1/r Coulomb potential demands a Kato e-N cusp that neither
Gaussians nor plane waves supply, so the DTN polynomial Jastrow puts it back.
DTN is mutually exclusive with the J1/J2/J3 family (see
`doc/notes/wavefunction.md`, section "Drummond-Towler-Needs (DTN) polynomial
Jastrow"), so the J1/J2/J3 converter options are `None` and all
`opt_J*_param` flags are `False`. The geometry is not relaxed, so the atomic
forces are not zero.

## Pipeline

```text
pw.x (by hand, in qe-pbc/)
    |  qe-pbc/out/h_k025.save/
    v
QE_Workflow(kind="pgto+pw", ecutwfc="auto", jastrow_dtn="u_chi")     01_wf/
    |  hamiltonian_data.h5
    v
VMC_Workflow(opt_geminal=True,                                       02_vmc/
             opt_JDTN_u_param=True, opt_JDTN_chi_param=True,
             opt_JDTN_f_param=False, opt_J1/J2/J3_param=False)
    |  FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian")) -> hamiltonian_data.h5
    +--> MCMC_Workflow(atomic_force=True)                            03_mcmc/
    |
    +--> LRDMC_Workflow(alat=0.30, non_local_move="tmove",           04_lrdmc/
                        atomic_force=True)
```

`LRDMC_Workflow` uses the GFMC_t driver (`job_type = "lrdmc-tau"`) whenever
neither `target_survived_walkers_ratio` nor `num_projection_per_measurement`
is passed, with `time_projection_tau = 0.10` Ha^-1 by default.
`non_local_move = "tmove"` is the default; the AE potential is local, so it
has no non-local term to act on.

## Files in this example

```text
jqmc-workflow-example05/
  README.md
  run_pipelines.py         # 4-stage chain driven by Launcher
  qe-pbc/
    h_scf.in               # QE input (same as jqmc-example08/01DFT/h_scf.in)
    h_scf.out              # QE log of the reference run
    H.coulomb-ae.UPF       # all-electron 1/r Coulomb pseudo (downloaded, not shipped)
```

## Step 1: QE (by hand)

`QE_Workflow` only converts an existing QE save directory; it does not run
`pw.x`. Run QE first (built with HDF5 support, so that it writes
`wfc1.hdf5`). Download the pseudopotential `H.coulomb-ae.UPF` from
<https://pseudopotentials.quantum-espresso.org/upf_files/H.coulomb-ae.UPF> into `qe-pbc/` first; it is not shipped:

```bash
cd qe-pbc
curl -L -O https://pseudopotentials.quantum-espresso.org/upf_files/H.coulomb-ae.UPF
mpirun -np 32 pw.x -in h_scf.in > h_scf.out
cd ..
```

This writes `qe-pbc/out/h_k025.save/`. Key lines of `h_scf.out` from the
reference run:

```text
     IMPORTANT: XC functional enforced from input :
     Exchange-correlation= PZ
...
     highest occupied, lowest unoccupied level (ev):    -5.3611    2.4430

!    total energy              =      -4.48559671 Ry
...
     convergence has been achieved in   8 iterations
...
     PWSCF        :      1.89s CPU     16.93s WALL
```

## Step 2: Configure the machine and run the pipeline

The configuration block of `run_pipelines.py`:

<!-- include: run_pipelines.py#configuration -->
```python
SERVER = "cluster"
QUEUE_LABEL = "cores-128-mpi-128-omp-1-3h"
LABEL = "h-ae"

QE_SAVE_GLOB = "qe-pbc/**/*.save"

DT = 1.0
TARGET_PROD_ERROR = 5.0e-4
TARGET_VMC_ERROR = 5.0e-3
MAX_TIME = 10000
POLL_INTERVAL = 120
MAX_CONTINUATION = 2
NUM_WALKERS = 1

# VMC optimisation steps per production run (the VMC stage runs
# MAX_CONTINUATION production runs, i.e. 200 steps in total).
NUM_OPT_STEPS = 100
WF_DUMP_FREQ = 1  # dump every step: VMC_Workflow hands on the average of the last 20 steps

# DTN Jastrow preset: u + chi only (no f).
JASTROW_DTN = "u_chi"

# LRDMC lattice spacing (Bohr) and non-local move scheme.
ALAT = 0.30
NON_LOCAL_MOVE = "tmove"
```

`SERVER` and `QUEUE_LABEL` are placeholders: set them to a machine and a queue
label defined in `jqmc_setting_local/` (placed in this directory; the script
changes into its own directory before launching) or in `~/.jqmc_setting/`.
See the **Configuration** section of `doc/notes/workflows.md` for the file
layout. The reference run used a PBS queue label with `num_cores = 128`
(one whole node) and `max_time = '03:00:00'`; it differed from the shipped
script only in the VMC dump frequency and the hand-off (see the note after
the VMC stage). Keep `MAX_TIME` (seconds, passed to jQMC) below the queue's wall-time
limit so that jQMC stops and writes its checkpoint before the scheduler kills
it. `max_job_submit` in `queue_data.toml` counts this driver's jobs in the
queue; this pipeline has at most two jobs (MCMC and LRDMC) in flight at once.

Then launch the driver (it only submits and polls; run it where `qsub`/`sbatch`
works, e.g. in the background on the login node):

```bash
nohup python run_pipelines.py > driver.log 2>&1 &
```

The driver resumes from `workflow_state.toml` files if it is restarted.

### What the stages print

`QE_Workflow` (`driver.log`, abridged) -- identical to the `jqmc-tool qe` run
in `jqmc-example08`:

```text
Loaded QE reference from .../qe-pbc/out/h_k025.save: 4 atoms, 6 bands, 80267 G-vectors, ecutwfc=400.000 Ha.
...
Auto pgto+pw cutoff: target=2.00e-05 MET, selected g_cut=2.739 Bohr^-1 (ecutwfc=7.500 Ry), n_pw=72, mean residual=8.597e-06.
...
build_wavefunction_from_projection(kind=pgto+pw): 104 AOs, 2 MOs, ne_up=2, ne_dn=2.
build_hamiltonian_from_wavefunction: DTN polynomial Jastrow attached (use_u=True, use_chi=True, use_f=False, C=4).
```

so each MO is expanded in 32 pGTOs (8 per H) plus 72 plane waves.

`VMC_Workflow` first runs a pilot (in `02_vmc/_pilot/`) and sizes the
production runs from its last-step error:

```text
Pilot VMC last-step energy: -2.32783 +- 0.00779 Ha (from 50 MCMC steps/step)

-- VMC Step Estimation Summary -------------------
  pilot MCMC/step     = 50
  warmup steps        = 30
  pilot VMC steps     = 20
  pilot error (last)  = 0.00779 Ha
  target error        = 0.005 Ha
  ...
  estimated MCMC/step = 78
  production VMC steps= 100
```

With no convergence criterion set (`target_snr`,
`energy_slope_sigma_threshold`), `VMC_Workflow` runs all `MAX_CONTINUATION = 2`
production runs; run 2 restarts from `restart.h5`, so the optimization reaches
step 200 (232 variational parameters). The workflow hands the average of steps
181-200 (`hamiltonian_data_opt_averaged.h5`) to MCMC and LRDMC.

> [!NOTE]
> `VMC_Workflow` hands the average of the last 20 optimization steps downstream by default
> (`handoff_mode="average"`, `handoff_average_steps=20`, written by `jqmc-tool vmc average-wf` to
> `hamiltonian_data_opt_averaged.h5`), so the script dumps every step (`WF_DUMP_FREQ = 1`). The numbers
> quoted here were not sampled from this average: the reference run dumped every
> 20 steps and handed the single last dump (`hamiltonian_data_opt_step_200.h5`) to the downstream stages; `handoff_mode="last"` selects this single-step hand-off.

`MCMC_Workflow` and `LRDMC_Workflow` each run a 100-step pilot and then size
production for `TARGET_PROD_ERROR`:

```text
-- MCMC Step Estimation Summary ------------------
  pilot steps       = 100
  warmup steps      = 30
  pilot error       = 0.00359678 Ha
  target error      = 0.0005 Ha
  ...
  estimated steps   = 3652
...
-- LRDMC Step Estimation Summary (a=0.3) --------
  pilot steps       = 100
  warmup steps      = 30
  pilot error       = 0.00539827 Ha
  target error      = 0.0005 Ha
  mode              = GFMC_t
  tau               = 0.1
  ...
  estimated steps   = 8189
...
  Target error achieved: 0.000493609 <= 0.0006 Ha (target*1.2) (run 1/2)
...
  Target error achieved: 0.00050865 <= 0.000525 Ha (target*1.05) (run 1/2)
```

Both reached the target in the first production run, so no continuation run
was needed. (The "est. prod. time" lines in the log are based on the pilot's
wall time, which includes queue waiting; ignore them on a busy cluster.)

## Results

At the end, `driver.log` shows the DAG summary and the table printed by
`run_pipelines.py` (reference run):

```text
==================================================
  DAG execution summary
--------------------------------------------------
  Completed : 4
  Failed    : 0  (of which skipped due to upstream: 0)
  Pending   : 0
==================================================
...
====================================================================================================
  Hydrogen Energy & Force Summary  (AE, packed-H2 ortho cell, k=(1/4, 1/4, 1/4))
  LRDMC a = 0.3,  VMC opt steps = 100 x 2 runs,  DTN preset = u_chi
====================================================================================================

|    Pattern |       Energy (Ha) |  Max|F| (Ha/bohr) |  t_net (s) |
|------------|-------------------|-------------------|------------|
|      MCMC  |      -2.36357(51) |        0.0484(12) |       28.0 |
|      LRDMC |      -2.38105(49) |        0.0476(18) |      160.9 |
```

`Max|F|` is the largest per-atom force norm; `t_net` is the net sampling time
reported by jQMC. The state of every stage can be checked at any time with
`jqmc-jobmanager show`:

```text
   ID  Status       Label                Type                 Server       Job#
  --------------------------------------------------------------------------
    0  completed    wf                   QE_Workflow          ?            -
       dir: 01_wf
    1  completed    vmc                  VMC_Workflow         cluster      job-001
       dir: 02_vmc
       energy: -2.36832 +- 0.00421 Ha
    2  completed    mcmc                 MCMC_Workflow        cluster      job-002
       dir: 03_mcmc
       energy: -2.3635654066315324 +- 0.000508650056013467 Ha
    3  completed    lrdmc                LRDMC_Workflow       cluster      job-003
       dir: 04_lrdmc
       energy: -2.3810484409671457 +- 0.0004936088255894646 Ha
```

(The VMC entry is the energy of the last optimization step, not a production
value. Server and job identifiers are illustrative.)

### Re-analysis by hand

```bash
jqmc-tool vmc analyze-output 02_vmc/output_vmc-*_1_*.out 02_vmc/output_vmc-*_2_*.out
jqmc-tool mcmc compute-energy 03_mcmc/restart.h5 -w 30 -b 10
jqmc-tool mcmc compute-force 03_mcmc/restart.h5 -w 30 -b 10
jqmc-tool lrdmc compute-energy 04_lrdmc/restart.h5 -w 30 -b 10 -c 20
jqmc-tool lrdmc compute-force 04_lrdmc/restart.h5 -w 30 -b 10 -c 20
```

The `-w/-b/-c` values match the warmup, bin and weight-collection settings in
`run_pipelines.py`. VMC optimization (both production runs, abridged):

```text
------------------------------------------------------
Iter     E (Ha)     Max generalized force (Ha/a.u.)  Max force S/N
------------------------------------------------------
   1  -2.185(17)   +2.80(27)    23.038
   2  -2.185(16)   +3.10(23)    22.758
   3  -2.209(14)   +3.14(19)    26.785
...
  50  -2.3550(41)  +1.027(48)    21.474
...
 100  -2.3603(44)  +0.793(51)    16.405
 101  -2.3650(43)  +0.695(50)    13.797
...
 196  -2.3634(43)   -0.66(27)     8.997
 197  -2.3604(43)  -0.472(87)     8.241
 198  -2.3678(45)   -0.71(22)     8.356
 199  -2.3660(45)   +0.49(14)     8.387
 200  -2.3683(42)   -0.42(12)     6.286
------------------------------------------------------
```

The energy is flat within the error bars after about 50 steps; the maximum
generalized force keeps decreasing but its S/N is still 6-9 at step 200, so
the parameters are not fully converged (as in `jqmc-example08`). For a
production run, pass `target_snr` (e.g. 4) to `VMC_Workflow` and raise
`max_continuation`: it then stops adding continuation runs once the averaged
S/N is at or below the target, and marks the stage failed if the target is
not reached within `max_continuation` runs.

MCMC (`03_mcmc/`):

```text
E = -2.3635654066315324 +- 0.000508650056013467 Ha.
```

```text
Atomic Forces:
------------------------------------------------
Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
------------------------------------------------
H       +0.0119(14)  -0.0460(12)  -0.0092(11)
H       +0.0131(14)  -0.0206(13)  -0.0103(12)
H       -0.0216(14)  +0.0319(14)  +0.0140(13)
H       -0.0022(12)  +0.0332(13)  +0.0074(13)
------------------------------------------------
```

LRDMC (`04_lrdmc/`, `alat = 0.30`; survived walkers ratio 99.04 %):

```text
E = -2.3810484409671457 +- 0.0004936088255894646 Ha.
```

```text
Atomic Forces:
------------------------------------------------
Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
------------------------------------------------
H       +0.0106(12)  -0.0447(18)  -0.0126(15)
H       +0.0137(17)  -0.0184(23)  -0.0116(16)
H       -0.0231(12)  +0.0301(15)  +0.0089(16)
H       +0.0029(16)  +0.0352(15)  +0.0123(18)
------------------------------------------------
```

The same values are in the `Final output(s)` block at the end of
`03_mcmc/output_mcmc-*.out` and `04_lrdmc/output_lrdmc-*.out`.

- LRDMC lowers the energy by 17.5(7) mHa relative to VMC
  (-2.3811(5) vs -2.3636(5) Ha).
- The forces are tens of error bars away from zero (the geometry is not an
  equilibrium structure) and sum to zero over the four atoms within the error
  bars, as translational invariance requires. MCMC and LRDMC forces differ by
  at most 5 mHa/bohr.
- Compared with `jqmc-example08` (MCMC -2.36482(22), LRDMC -2.38152(39) Ha),
  the LRDMC energies agree within the error bars and the VMC energy is
  1.3(6) mHa higher, reflecting the slightly different optimized trial
  wavefunction; the forces agree to within about 4 mHa/bohr.

LRDMC energies and forces depend on `alat`; for an `alat -> 0` extrapolation
see `jqmc-workflow-example04` (energies) or run several `ALAT` values and fit
each force component against `alat^2`.

## Optimized-trial handoff and continuation

MCMC and LRDMC receive `FileFrom("vmc", ValueFrom("vmc",
"optimized_hamiltonian"))`, renamed to `hamiltonian_data.h5`. The published
value is `hamiltonian_data_opt_averaged.h5`, the average of the 20
largest-numbered `hamiltonian_data_opt_step_N.h5` in `02_vmc/` (here steps
181-200, including the continued run); it is not inferred from
`NUM_OPT_STEPS`.

Inspect the VMC convergence before accepting that trial. To try another
trial, use fresh downstream directories: an existing staged file is not
overwritten automatically, and its checkpoint must not be reused with a
different Hamiltonian.

## Related examples

| Example                            | Same system? | Drive style |
|------------------------------------|--------------|-------------|
| `examples/jqmc-example08`          | Yes          | Manual TOML |
| `examples/jqmc-workflow-example03` | Water        | `jqmc_workflow` API (PySCF -> VMC -> MCMC -> LRDMC) |
| `examples/jqmc-workflow-example06` | Silicon      | `jqmc_workflow` API (multi-twist optimization and handoff) |

## Orbital tiling

This workflow uses an untiled composite pGTO+PW determinant.
For a frozen tiled variant, `QE_Workflow(kind="pgto+pw", tiling=[2,1,1], ...)`
retains both primitive basis components inside each k block. Supply complete
primitive k cosets and set `opt_geminal = false`; this workflow's existing
Slater-optimization inputs remain untiled. The Jastrow basis belongs independently
to the physical supercell. See the
[tiling recipe](../../doc/examples.md#jqmc-example14-frozen-orbitals-in-a-tiled-supercell).
