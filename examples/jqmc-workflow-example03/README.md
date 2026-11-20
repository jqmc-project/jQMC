# jqmc-workflow-example03: Water energy and atomic forces

An eight-valence-electron water molecule with ccECP / Cartesian cc-pVTZ
orbitals. Local PySCF RHF supplies the TREXIO trial. The JSD has **Pade J2**
(initial parameter 1.0) and AO-small J3, with no J1. VMC optimizes J2 and J3
while freezing the determinant, using SR with the linear-method step
(`delta=0.350, epsilon=0.001, use_lm=True`, default `lm_subspace_dim = 0`).
MCMC and LRDMC then compute the energy and the atomic forces with the
optimized trial.

```text
PySCF RHF (local) -> WF -> VMC -> MCMC  energy and forces
                               -> LRDMC energy and forces
```

MCMC and LRDMC both depend on VMC, not on each other, so they run
concurrently. LRDMC uses the default `LRDMC_Workflow` driver, the time-step
projection `lrdmc-tau` (GFMC_t), at lattice spacing `ALAT` (bohr).

## Setup and launch

Requires jQMC with `jqmc_workflow` matching this checkout, PySCF with TREXIO
export, and a workflow machine configuration. The script changes into its
own directory, so the configuration is read from `jqmc_setting_local/`
placed **inside this example directory** (project-local override) or else
from `~/.jqmc_setting/`. Set `SERVER`, `QUEUE_LABEL` and `PILOT_QUEUE_LABEL`
in the script to a machine and queue labels that exist in that
configuration; see the **Configuration** section of the workflow notes
(`doc/notes/workflows.md`). The queue entry (`num_cores`, `max_time`,
`max_job_submit`, submit template) decides the MPI size and wall time.

`MAX_TIME` (seconds) is written to `[control] max_time` of every jQMC input,
so keep it below the queue wall time. `max_job_submit` limits the number of
this driver's jobs in that queue; a submission above the limit waits for a
free slot.

```bash
python run_pipelines.py > driver.log 2>&1
```

The driver only runs PySCF and the TREXIO conversion locally (seconds) and
then submits/polls the batch jobs; run it in the background
(`nohup ... &`) on a login node. Re-running it after an interruption resumes
from `workflow_state.toml` in each stage directory and skips finished stages.

## Settings taken from the executable script

Edit the script, not this generated block (it is refreshed from
`run_pipelines.py` by `examples/sync_examples.py`).

<!-- include: run_pipelines.py#configuration -->
```python
SERVER = "cluster"
QUEUE_LABEL = "cores-128-mpi-128-omp-1-1h"
PILOT_QUEUE_LABEL = "cores-128-mpi-128-omp-1-1h"

# VMC
NUM_OPT_STEPS = 100
WF_DUMP_FREQ = 1  # dump every step: VMC_Workflow hands on the average of the last 20 steps
TARGET_VMC_ERROR = 3.0e-3

# MCMC / LRDMC
Dt = 2.0
ALAT = 0.30
TARGET_PROD_ERROR = 5.0e-4

# Common
MAX_TIME = 3000
MAX_CONTINUATION = 2
NUMBER_OF_WALKERS = 1
POLL_INTERVAL = 60

# pySCF output
TREXIO_FILE = "water_ccecp_ccpvtz.h5"
PYSCF_OUTPUT = "water.out"
```

The results below were produced with these settings on a PBS queue entry with
`num_cores = 128`, `omp_num_threads = 1`,
`max_time = '01:00:00'`, i.e. 128 MPI ranks x 1 walker on one 128-core CPU
node per job (full precision, CPU backend), except for the VMC dump
frequency and the hand-off of the optimized trial (see the note in the VMC
section). Set the server and queue labels of the script to entries of your
local configuration.

| Stage | PBS jobs | Steps | Job wall time |
| --- | --- | --- | --- |
| VMC pilot | 1 | 20 opt x 50 MCMC | 0:32 |
| VMC production | 2 | 2 x (100 opt x 444 MCMC) | 4:50 + 4:27 |
| MCMC pilot / production | 2 | 100 / 17243 | 1:04 / 1:47 |
| LRDMC pilot / production | 2 | 100 / 42458 | 0:36 / 6:05 |

The whole pipeline took about 20 minutes of compute; the elapsed time of the
driver (3.5 h) was dominated by queue waiting on a shared cluster.

## Step 0 and WF: PySCF and conversion

The embedded PySCF script writes `water.out` and `water_ccecp_ccpvtz.h5`; the
WF stage runs

```text
jqmc-tool trexio convert-to water_ccecp_ccpvtz.h5 -o hamiltonian_data.h5 -j2 1.0 -j3 ao-small
```

The driver reports `HF energy = -16.943755 Ha`.

## VMC: pilot and optimization

The pilot (20 steps x 50 MCMC steps per step) estimates the MCMC steps per
optimization step needed for `TARGET_VMC_ERROR`:

```text
-- VMC Step Estimation Summary -------------------
  pilot MCMC/step     = 50
  warmup steps        = 30
  pilot VMC steps     = 20
  pilot error (last)  = 0.01366 Ha
  target error        = 0.003 Ha
  ...
  estimated MCMC/step = 444
  production VMC steps= 100
  ...
  est. production time= 12m 0s
```

`NUM_OPT_STEPS` counts the steps **per production run**. Because no
convergence criterion (`target_snr`, `energy_slope_sigma_threshold`) is set,
all `MAX_CONTINUATION = 2` runs are executed, the second restarting from the
first: 200 optimization steps in total. The workflow then publishes the
average of the last 20 saved steps (181-200), `hamiltonian_data_opt_averaged.h5`,
as `optimized_hamiltonian`.

> [!NOTE]
> `VMC_Workflow` hands the average of the last 20 optimization steps downstream by default
> (`handoff_mode="average"`, `handoff_average_steps=20`, written by `jqmc-tool vmc average-wf` to
> `hamiltonian_data_opt_averaged.h5`), so the script dumps every step (`WF_DUMP_FREQ = 1`). The numbers
> quoted here were not sampled from this average: the reference run dumped every
> 20 steps and handed the single last dump (`hamiltonian_data_opt_step_200.h5`) to the downstream stages; `handoff_mode="last"` selects this single-step hand-off.

```bash
jqmc-tool vmc analyze-output 02_vmc/*.out
```

```text
------------------------------------------------------
Iter     E (Ha)     Max generalized force (Ha/a.u.)  Max force S/N
------------------------------------------------------
   1  -16.611(11)  +1.315(13)   119.026
   2  -16.8409(79)  +1.003(11)   100.358
   ...
  10  -17.1448(39)  +0.3640(70)    51.893
  ...
  23  -17.1869(33)  +0.0600(40)    14.367
  ...
  38  -17.1952(33)  +0.0310(80)     6.662
  ...
  58  -17.1900(33)  -0.0060(90)     2.485
  ...
 100  -17.1929(31)  +0.0150(90)     3.230
 101  -17.1900(32)  -0.0100(70)     2.824
  ...
 199  -17.1985(33)  -0.0070(70)     2.072
 200  -17.1943(33)  +0.0130(90)     2.698
------------------------------------------------------
```

The energy flattens at about -17.194 Ha after ~40 steps and the maximum
force S/N stays at 2-4 from step ~55 on (with occasional spikes to ~7, e.g.
steps 115-120), so the trial is converged within the per-step error of
3.3 mHa; the second run mainly confirms it.

## MCMC and LRDMC energies and forces

Each sampling stage runs a 100-step pilot, estimates the steps for
`TARGET_PROD_ERROR = 5.0e-4` Ha, and then one production run (both reached
the target in run 1, so no continuation was needed):

```text
-- MCMC Step Estimation Summary ------------------
  pilot error       = 0.00784065 Ha
  target error      = 0.0005 Ha
  estimated steps   = 17243
...
-- LRDMC Step Estimation Summary (a=0.3) --------
  pilot error       = 0.0123098 Ha
  target error      = 0.0005 Ha
  mode              = GFMC_t
  tau               = 0.1
  estimated steps   = 42458
```

Final output of the MCMC production log
(`03_mcmc/output_mcmc-water_1_*.out`):

```text
Final output(s):
  Total Energy: E = -17.19378 +- 0.00052 Ha.
  Variance: Var = 0.42537 +- 0.00172 Ha^2.
  Atomic Forces:
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  O       -0.0058(13)  -0.0047(8)   -0.0074(11)
  H       +0.0098(11)  +0.0019(6)   -0.0001(5)
  H       -0.0040(7)   +0.0028(6)   +0.0076(10)
  ------------------------------------------------
```

Final output of the LRDMC production log
(`04_lrdmc/output_lrdmc-water_1_*.out`):

```text
Survived walkers ratio = 98.11 %
Average of the number of projections  = 27

Final output(s):
  Total Energy: E = -17.24087 +- 0.00036 Ha.
  Variance: Var = 0.41337 +- 0.00105 Ha^2.
  Atomic Forces:
  ------------------------------------------------
  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
  ------------------------------------------------
  O       -0.0044(29)  -0.001(6)    -0.0072(35)
  H       +0.0005(9)   -0.0007(7)   -0.0010(5)
  H       +0.0001(6)   +0.00011(27) -0.0012(11)
  ------------------------------------------------
```

Summary printed at the end of `run_pipelines.py` (`Max|F|` is the largest
per-atom force norm, `t_net` the net execution time from the log):

```text
  Water Energy & Force Summary  (ccECP / cc-pVTZ, Cartesian)
  LRDMC a = 0.3,  VMC opt steps = 100 x 2 runs
  HF energy = -16.943755 Ha
...
|      Pattern |       Energy (Ha) |  Max|F| (Ha/bohr) |  t_net (s) |
|--------------|-------------------|-------------------|------------|
|        MCMC  |     -17.19378(52) |        0.0105(11) |       62.4 |
|        LRDMC |     -17.24087(36) |        0.0085(34) |      335.4 |
```

Interpretation:

- LRDMC (a = 0.30) lies 47.1(6) mHa below VMC, the fixed-node projection gain
  over the JSD trial. A continuum LRDMC energy needs several `ALAT` values
  and an extrapolation.
- The VMC forces sum to zero component-wise within one error bar
  (0.0000, 0.0000, +0.0001 Ha/bohr), as they should for an isolated
  molecule; the residual forces of up to 0.01 Ha/bohr show that this
  geometry is not the VMC equilibrium.
- The LRDMC force errors are larger and rougher: `lrdmc compute-energy`
  reports only `Total number of binned samples = 10` (LRDMC bins the
  weighted walker population globally, whereas MCMC bins per rank:
  1280 samples), and the LRDMC force sum, (-0.0038, -0.0016, -0.0094)
  Ha/bohr, is within about 2.5 error bars of zero. Energy targets are not
  force targets; tighten `TARGET_PROD_ERROR` (or run longer) before using
  LRDMC forces quantitatively.

The workflow state can be inspected with the job manager:

```bash
jqmc-jobmanager show
```

```text
   ID  Status       Label                Type                 Server       Job#
  --------------------------------------------------------------------------
    0  completed    wf                   WF_Workflow          ?            -
       dir: 01_wf
    1  completed    vmc                  VMC_Workflow         cluster      job-001
       dir: 02_vmc
       energy: -17.19426 +- 0.00326 Ha
    2  completed    mcmc                 MCMC_Workflow        cluster      job-002
       dir: 03_mcmc
       energy: -17.19378148784333 +- 0.0005214038797723325 Ha
    3  completed    lrdmc                LRDMC_Workflow       cluster      job-003
       dir: 04_lrdmc
       energy: -17.240865063270075 +- 0.00035716445331944815 Ha
```

(The VMC entry is the last optimization step, not a production energy.)

## Optimized-trial handoff and continuation

Downstream nodes use
`FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian"))` and rename the
staged file to `hamiltonian_data.h5`. The published value is the average of
the 20 largest numeric saved steps including continued runs (here steps
181-200, `hamiltonian_data_opt_averaged.h5`); it is not inferred from
`NUM_OPT_STEPS`. The reference run quoted above handed over
`hamiltonian_data_opt_step_200.h5` itself.

Inspect convergence before accepting that trial. To compare another trial,
use fresh downstream directories: an existing staged file is not
automatically overwritten, and its checkpoint must not be reused with a
different Hamiltonian.

## Re-analyzing the checkpoints

The workflow runs these post-processing commands itself with the settings
above (30 warmup measurements, 10 bins, LRDMC 20-step weight collection);
repeat them to inspect or rebin a result:

```bash
jqmc-tool mcmc compute-energy 03_mcmc/restart.h5 -w 30 -b 10
jqmc-tool mcmc compute-force 03_mcmc/restart.h5 -w 30 -b 10
jqmc-tool lrdmc compute-energy 04_lrdmc/restart.h5 -w 30 -b 10 -c 20
jqmc-tool lrdmc compute-force 04_lrdmc/restart.h5 -w 30 -b 10 -c 20
```

```text
$ jqmc-tool mcmc compute-energy 03_mcmc/restart.h5 -w 30 -b 10
Read restart checkpoint file(s) from 03_mcmc/restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 1280
E = -17.19378148784333 +- 0.0005214038797723325 Ha.

$ jqmc-tool lrdmc compute-energy 04_lrdmc/restart.h5 -w 30 -b 10 -c 20
Read restart checkpoint file(s) from 04_lrdmc/restart.h5.
Found 128 MPI ranks.
Total number of binned samples = 10
E = -17.240865063270075 +- 0.00035716445331944815 Ha.
```

`compute-force` prints the same force tables as the final outputs above.
