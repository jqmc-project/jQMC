# jqmc-workflow-example06: twist-averaged VMC optimization and downstream MCMC

This example runs a twist-averaged pipeline with `jqmc_workflow`. It converts
the eight-twist silicon QE calculation of
`jqmc-example09` into one multi-twist
B-spline Hamiltonian, optimizes the shared J2 Jastrow and the per-twist
determinants in a twist-averaged VMC run, and passes the complete optimized
multi-twist Hamiltonian that the VMC stage publishes to a downstream MCMC
run:

```text
qe   (QE_Workflow,   local)  ->  01_qe_to_jqmc/hamiltonian_data.h5
vmc  (VMC_Workflow,  batch)  ->  02_vmc/hamiltonian_data_opt_averaged.h5  (steps 11-30)
mcmc (MCMC_Workflow, batch)  ->  03_mcmc/restart.h5/   E_TA
```

All numbers below come from one run of `run_twist_pipeline.py` with the
settings shown in step 2, except for the VMC dump frequency and the hand-off
of the optimized trial (see the note in step 3): 128 MPI
ranks on one 128-core CPU node per batch job (8 walkers per rank, 16 ranks
and 128 walkers per twist), CPU only, `OMP_NUM_THREADS=1`,
`precision_mode="mixed"`.

| Stage | Where | Wall time |
|---|---|---|
| QE `pw.x`, 8 k points (prerequisite) | batch, 128 ranks | 54 s |
| `qe`: `jqmc-tool qe --kind bspline --twist-average` | driver host | < 5 s |
| `vmc`: 30 optimization steps x 150 measurements | batch, 128 ranks | 95 min |
| `mcmc`: 10000 measurements | batch, 128 ranks | 5 min |

Each optimization step took about 186 s: 101 s of sampling plus 84 s that the
log reports as `Time for computing E`. Those 84 s are not energy evaluation.
The run landed on a node with a NUMA imbalance, so the twist groups
finished their sampling at different times, and this timer also absorbs the
wait for the slowest twist. The step and stage times above are therefore
longer than on a balanced node.

The trial wavefunction is the QE determinant on the B-spline basis
with a Pade J2 (`j2_parameter=1.0`). The step counts produce a visible energy
gain, but the optimization is not converged (see step 3); this is a
demonstration of the route, not a production setting.

## Why the published file matters

A twist-averaged VMC run writes one checkpoint per twist plus the shared
state. Despite its `.h5` suffix, `restart.h5` is then a **directory**:

```text
02_vmc/restart.h5/
  shared_state.h5
  twist_000.h5
  ...
  twist_007.h5
```

That directory resumes the optimizer and sampler; it is not a trial
wavefunction. Under twist averaging, the
`hamiltonian_data_opt_step_N.h5` file that jQMC writes every `wf_dump_freq`
steps is instead a complete multi-twist Hamiltonian. It keeps the
structure, B-spline orbitals, k points and BZ weights, carries the optimized
shared Jastrow, and carries each twist's own optimized determinant:

```text
Twist-averaged Hamiltonian data (8 twists) is dumped as an HDF5 file: hamiltonian_data_opt_step_30.h5.
Twist-averaged optimization complete. Shared Jastrow parameters were synchronized across twists; determinant parameters were optimized in their twist-local sampler communicators.
```

`VMC_Workflow` averages the last 20 such files with `jqmc-tool vmc average-wf`
(the shared Jastrow once, every twist's determinant separately; the result is
again a complete multi-twist Hamiltonian) and publishes the basename of the
average, `hamiltonian_data_opt_averaged.h5`, as
`output_values["optimized_hamiltonian"]`; the averaged steps are recorded in
`output_values["optimized_hamiltonian_steps"]`. For this the script dumps
every step (`WF_DUMP_FREQ = 1`). Single-twist runs use the same code path.
The MCMC container receives the file through

```python
FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian"))
```

and renames it to `hamiltonian_data.h5`, so the same pipeline connection
works for single-twist and twist-averaged VMC runs.

## Before you start

Requirements specific to this example:

- **MPI with a multiple of eight ranks** in the queue that runs the VMC and
  MCMC jobs. Each twist gets an equal group of ranks, so a serial run is not
  possible. The reference run used 128 ranks.
- **Quantum ESPRESSO `pw.x` built with HDF5** for the prerequisite step.
- A `jqmc_workflow` machine configuration with a batch queue, see the
  **Configuration** section of
  [`doc/notes/workflows.md`](../../doc/notes/workflows.md). Place it in
  `jqmc_setting_local/` inside this directory or in `~/.jqmc_setting/`.

## 1. Generate the eight QE twists

The QE input belongs to `jqmc-example09`. It reads the ccECP pseudopotential
from `jqmc-example07/01DFT/`, which is not shipped: download it there first
(see step 1 of `jqmc-example07`). Then run step 1 of `jqmc-example09` from
this directory:

```bash
curl -L -o ../jqmc-example07/01DFT/Si.ccECP.upf https://pseudopotentiallibrary.org/recipes/Si/ccECP/Si.ccECP.upf
cd ../jqmc-example09/01DFT
mkdir -p out
mpirun -np 128 pw.x -in si_twist.in > si_twist.out
cd ../../jqmc-workflow-example06
```

```text
!    total energy              =     -15.66056909 Ry
     PWSCF        :      2.62s CPU     54.37s WALL
```

The pipeline reads `../jqmc-example09/01DFT/out/silicon_twist.save`
(`data-file-schema.xml`, `wfc1.hdf5` ... `wfc8.hdf5` and the UPF copied by
QE). They are written by the step above; none of them is shipped.

## 2. Configure and run the pipeline

The settings at the top of `run_twist_pipeline.py`:

<!-- include: run_twist_pipeline.py#configuration -->
```python
SERVER = "cluster"  # machine name in jqmc_setting_local/ or ~/.jqmc_setting/
QUEUE_LABEL = "cores-128-mpi-128-omp-1-3h"  # MPI ranks must be a multiple of N_TWIST
N_TWIST = 8

NUMBER_OF_WALKERS = 8  # walkers per MPI rank
NUM_OPT_STEPS = 30  # VMC optimization steps
NUM_VMC_MCMC_STEPS = 150  # MCMC measurements per optimization step
WF_DUMP_FREQ = 1  # dump every step: VMC_Workflow hands on the average of the last 20 steps
NUM_MCMC_STEPS = 10000  # measurements of the downstream MCMC run
DT = 2.0  # MCMC step size (bohr)
PRECISION_MODE = "mixed"  # "full" (default, reference) or "mixed"
```

Set `SERVER` and `QUEUE_LABEL` to a machine and queue of your
configuration. The reference run used a PBS machine entry whose queue
launches 128 MPI ranks on one node with a 3 h wall-time limit; all other
values except `WF_DUMP_FREQ` (10 in the reference run) are the ones shown
above.

The three containers:

<!-- include: run_twist_pipeline.py#workflows -->
```python
    qe_save_dir = os.path.abspath(os.path.join(base_dir, "..", "jqmc-example09", "01DFT", "out", "silicon_twist.save"))

    conversion = Container(
        label="qe",
        dirname="01_qe_to_jqmc",
        workflow=QE_Workflow(
            qe_save_dir=qe_save_dir,
            kind="bspline",
            twist_average=True,
            hamiltonian_file="hamiltonian_data.h5",
            j2_parameter=1.0,
            verbosity="high",
        ),
    )

    vmc = Container(
        label="vmc",
        dirname="02_vmc",
        input_files=[FileFrom("qe", "hamiltonian_data.h5")],
        workflow=VMC_Workflow(
            server_machine_name=SERVER,
            queue_label=QUEUE_LABEL,
            jobname="si-twist-vmc",
            number_of_walkers=NUMBER_OF_WALKERS,
            num_opt_steps=NUM_OPT_STEPS,
            num_mcmc_steps=NUM_VMC_MCMC_STEPS,
            # Code minimums: 30 warmup measurements and 10 bins.
            num_mcmc_warmup_steps=30,
            num_mcmc_bin_blocks=10,
            Dt=DT,
            wf_dump_freq=WF_DUMP_FREQ,
            opt_J1_param=False,
            opt_J2_param=True,
            opt_J3_param=False,
            opt_geminal=True,
            optimizer_kwargs={
                "method": "sr",
                "delta": 0.1,
                "epsilon": 0.001,
                "use_lm": True,
            },
            precision_mode=PRECISION_MODE,
            verbosity="high",
        ),
    )

    mcmc = Container(
        label="mcmc",
        dirname="03_mcmc",
        input_files=[
            FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian")),
        ],
        rename_input_files=["hamiltonian_data.h5"],
        workflow=MCMC_Workflow(
            server_machine_name=SERVER,
            queue_label=QUEUE_LABEL,
            jobname="si-twist-mcmc",
            number_of_walkers=NUMBER_OF_WALKERS,
            num_mcmc_steps=NUM_MCMC_STEPS,
            num_mcmc_warmup_steps=30,
            num_mcmc_bin_blocks=10,
            Dt=DT,
            precision_mode=PRECISION_MODE,
            verbosity="high",
        ),
    )
```

Notes on these choices:

- `kind="bspline"` interpolates the QE orbitals on a real-space mesh (default
  mesh factor 0.75 of the QE FFT grid), as in `jqmc-example09`; no Gaussian
  fit is involved. `kind="pgto+pw"` is the other lossless choice.
- `num_mcmc_steps` puts both batch stages in fixed-step mode: no pilot run,
  and `target_error` is ignored.
- `opt_geminal=True` optimizes each twist's determinant (the MO coefficients
  over the eight B-spline bands) in that twist's own sampler group; the J2
  parameter is shared by all twists.
- `precision_mode="mixed"` keeps this demonstration cheap; `"full"` (the
  default) is the reference setting, see
  [`doc/notes/mixed_precision.md`](../../doc/notes/mixed_precision.md).
- No MPC correction and no forces or stress are requested in the MCMC stage;
  see `jqmc-example09` for those observables on the same system.

Run the driver on the host that can submit jobs. It only submits and polls,
so it can run in the background:

```bash
nohup python run_twist_pipeline.py > driver.log 2>&1 &
```

The QE conversion runs in the driver process itself (a local
`jqmc-tool qe` subprocess, a few seconds here); VMC and MCMC are batch jobs. The
pipeline is linear, so at most one batch job is queued at a time. With the
`graphviz` Python package installed, `draw_graph=True` also writes
`dependency_graph.png`; without it the driver prints
`graphviz not installed; skipping graph rendering.` and continues.

The resulting layout:

```text
jqmc-workflow-example06/
  01_qe_to_jqmc/hamiltonian_data.h5
  02_vmc/hamiltonian_data_opt_step_{1..30}.h5
  02_vmc/hamiltonian_data_opt_averaged.h5   (average of steps 11-30)
  02_vmc/restart.h5/
  02_vmc/output_si-twist-vmc_1_<id>.out
  03_mcmc/hamiltonian_data.h5          (copy of hamiltonian_data_opt_averaged.h5)
  03_mcmc/restart.h5/
  03_mcmc/output_si-twist-mcmc_1_<id>.out
```

## 3. Results

> [!NOTE]
> `VMC_Workflow` hands the average of the last 20 optimization steps downstream by default
> (`handoff_mode="average"`, `handoff_average_steps=20`, written by `jqmc-tool vmc average-wf` to
> `hamiltonian_data_opt_averaged.h5`), so the script dumps every step (`WF_DUMP_FREQ = 1`). The numbers
> quoted here were not sampled from this average: the reference run dumped every
> 10 steps and handed the single last dump (`hamiltonian_data_opt_step_30.h5`) to the downstream stages; `handoff_mode="last"` selects this single-step hand-off.

The end of `driver.log`:

```text
  Running: jqmc-tool mcmc compute-energy restart.h5 -b 10 -w 30
  MCMC energy: -7.981086927500488 +- 0.00012868953460768261 Ha
...
==================================================
  DAG execution summary
--------------------------------------------------
  Completed : 3
  Failed    : 0  (of which skipped due to upstream: 0)
  Pending   : 0
==================================================
Program ends ...
Published optimized 8-twist Hamiltonian: hamiltonian_data_opt_step_30.h5
Twist-averaged MCMC energy: -7.981086927500488 +- 0.00012868953460768261 Ha
```

### Conversion

`driver.log` repeats the converter output:

```text
  Running: jqmc-tool qe .../jqmc-example09/01DFT/out/silicon_twist.save -k bspline -o hamiltonian_data.h5 --twist-average -j2 1.0 --dtn-cutoff-ratio-u 0.95 --dtn-cutoff-ratio-chi 0.95 --dtn-cutoff-ratio-f 0.95 --verbosity high
QE multi-twist convert: 8 k-points found in .../silicon_twist.save/data-file-schema.xml
Loaded QE reference from .../silicon_twist.save: 2 atoms, 8 bands, 4450 G-vectors, ecutwfc=50.000 Ha.
PW->B-spline: order 5, grid (34, 34, 34) from QE FFT grid (45, 45, 45) (mesh_factor 0.750), 8 bands, control-point table 0.005 GiB (complex128), mean residual 4.831e-12, max 2.057e-11.
...
QE multi-twist convert: wrote combined HDF5 (n_twist=8, sum_weights=2.000000) -> hamiltonian_data.h5
Twist-average Hamiltonian data is saved in hamiltonian_data.h5.
```

The `--dtn-cutoff-ratio-*` flags are always forwarded by `QE_Workflow`; they
have no effect here because no DTN Jastrow is requested.

### VMC optimization history

```bash
jqmc-tool vmc analyze-output 02_vmc/output_si-twist-vmc_1_*.out
```

```text
------------------------------------------------------
Iter     E (Ha)     Max generalized force (Ha/a.u.)  Max force S/N
------------------------------------------------------
   1  -7.9669(14)  -0.0620(10)   107.788
   2  -7.9711(13)  -0.0590(10)   100.011
   3  -7.9725(13)  -0.0570(10)    93.950
   4  -7.9749(13)  -0.0540(10)    84.346
   5  -7.9728(12)  -0.0500(10)    76.250
   6  -7.9765(12)  -0.0460(10)    69.397
   7  -7.9785(12)  -0.0480(70)    62.264
   8  -7.9779(12)  -0.0400(10)    56.918
   9  -7.9788(12)  -0.0380(70)    49.984
  10  -7.9776(12)  -0.0600(70)    46.641
  11  -7.9786(12)  -0.0480(70)    40.456
  12  -7.9795(12)  -0.0500(70)    36.055
  13  -7.9792(12)  -0.0660(90)    32.618
  14  -7.9812(12)  -0.0520(70)    25.865
  15  -7.9810(12)  -0.0510(70)    23.794
  16  -7.9825(12)  -0.0570(70)    22.169
  17  -7.9795(12)  -0.0450(70)    20.696
  18  -7.9801(12)  -0.0550(70)    17.478
  19  -7.9809(12)  -0.0610(70)    15.836
  20  -7.9817(12)  -0.0630(70)    12.312
  21  -7.9799(12)  -0.0530(90)    12.326
  22  -7.9823(12)  -0.0530(70)     9.530
  23  -7.9794(12)  -0.0590(70)    10.897
  24  -7.9816(12)  -0.0490(80)    10.407
  25  -7.9822(12)  -0.0540(80)     7.444
  26  -7.9817(12)  -0.0510(70)     8.217
  27  -7.9818(12)  -0.047(11)     6.987
  28  -7.9794(12)  -0.0490(70)     8.544
  29  -7.9823(12)  -0.0620(70)     9.013
  30  -7.9818(12)  -0.0370(80)     6.363
------------------------------------------------------
```

`E` is the twist-averaged energy sampled before each update; step 1 is the
unoptimized trial. The energy drops by about 15 mHa in the first 15 steps and
then fluctuates around -7.981 Ha. The largest signal-to-noise ratio (in 24 of
the 30 steps that of the shared J2 parameter) falls from 108 to about 6-9, so
the optimization is approaching but has not reached convergence. The
`Max generalized force` column shows the largest force over all parameters;
from step 7 on it is mostly a determinant coefficient with a larger error
bar. Each step used
`150 - 30 = 120` measurements x 128 walkers per twist.

### The published file

`run_twist_pipeline.py` reads the `_meta` header of the published file and
stops unless it is a multi-twist file with eight twists. With the shipped
settings the published file is `hamiltonian_data_opt_averaged.h5`; the
reference run, which handed over the single last dump, printed
`Published optimized 8-twist Hamiltonian: hamiltonian_data_opt_step_30.h5`.
The listing below is of that step-30 dump; the same command works on
`02_vmc/hamiltonian_data_opt_averaged.h5`:

```bash
jqmc-tool hamiltonian show-info 02_vmc/hamiltonian_data_opt_step_30.h5
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
...
  Jastrow 2b param = 0.6800273060798645
...
```

The same command on `01_qe_to_jqmc/hamiltonian_data.h5` prints
`Jastrow 2b param = 1.0`: the optimization moved the shared J2 parameter
from 1.0 to 0.68. Add `--twist N` to show twist `N`.

### Downstream MCMC

In the reference run the MCMC stage sampled
`hamiltonian_data_opt_step_30.h5`, i.e. the wavefunction after the 30th
update (the VMC log entry for step 30 is the energy before that update);
with the shipped settings it samples `hamiltonian_data_opt_averaged.h5`. From `03_mcmc/output_si-twist-mcmc_1_*.out`:

```text
Final output(s):
  BZ-averaged Energy: E_TA = -7.98109 +- 0.00013 Ha (s_twist = 0.16353 Ha).
  Per-twist energy (Level-1 outputs):
  ------------------------------------------------
  twist  k_point                 E_k (Ha)
  ------------------------------------------------
      0  (+0.250,+0.250,+0.250)  -7.6983(4)
      1  (+0.250,+0.250,-0.250)  -8.07507(35)
      2  (+0.250,-0.250,+0.250)  -8.07500(35)
      3  (+0.250,-0.250,-0.250)  -8.0755(4)
      4  (-0.250,+0.250,+0.250)  -8.0754(4)
      5  (-0.250,+0.250,-0.250)  -8.0759(4)
      6  (-0.250,-0.250,+0.250)  -8.0761(4)
      7  (-0.250,-0.250,-0.250)  -7.6974(4)
  ------------------------------------------------
```

`E_TA = -7.98109(13)` Ha is 14 mHa below the unoptimized trial (step 1 of
the VMC history, -7.9669(14) Ha). `s_twist` is the spread of the energies
among the twists, not a statistical error; the two twists on the [111] axis
(0 and 7) lie 0.38 Ha above the other six, as in `jqmc-example09`.

The workflow obtained its energy with `jqmc-tool mcmc compute-energy`. To
repeat or rebin it, pass the checkpoint directory:

```bash
jqmc-tool mcmc compute-energy 03_mcmc/restart.h5 -w 30 -b 10
```

```text
Read restart checkpoint file(s) from 03_mcmc/restart.h5.
Found 8 twist checkpoints.
E = -7.981086927500488 +- 0.00012868953460768261 Ha.
```

The result is the BZ-weighted average over the eight twists. `-w` discards
the first measurements as warmup and `-b` sets the number of jackknife bins;
keep `-w` at least 30 and `-b` at least 10.

## Re-running

The launcher records each stage in `<stage>/workflow_state.toml`. Running
`python run_twist_pipeline.py` again after a complete run submits nothing:

```text
[vmc] Existing project dir with status='completed' found; will short-circuit or resume depending on workflow policy. Delete project dir to force a clean restart.
[vmc] Already completed, no re-run.
```

and it prints the same two result lines at the end. Delete the stage
directories (`01_qe_to_jqmc`, `02_vmc`, `03_mcmc`) to start from scratch.

## Orbital tiling

For tiled twist ensembles, see the [tiling recipe](../../doc/examples.md#jqmc-example14-frozen-orbitals-in-a-tiled-supercell).
`QE_Workflow` and `WF_Workflow` accept `tiling` with `twist_average=True`;
keep Slater optimization disabled and optimize the independent supercell
Jastrow only. The ensemble checkpoint retains every sector's ragged k blocks.
