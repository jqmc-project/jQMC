(workflows_tags)=
# Autonomous Pipelines (jqmc_workflow)

## Overview

The **jqmc_workflow** package provides an autonomous pipeline engine for
**jQMC** calculations.
Users define a *pipeline* -- a directed acyclic graph (DAG) of workflow steps --
and the engine takes care of:

- **Input generation** -- TOML input files are created automatically from
  explicit parameter values (with sensible defaults).
- **Data transfer** -- Files are uploaded to (and downloaded from) remote
  supercomputers via SSH/SFTP using [Paramiko](https://www.paramiko.org/).
- **Job submission, monitoring, and collection** -- Jobs are submitted through
  the site's scheduler (PBS / Slurm / local `bash`), polled until completion,
  and output files are fetched back.
- **Dependency resolution** -- The DAG-based `Launcher` identifies which
  workflow steps are *ready* (all dependencies satisfied) and executes them
  **in parallel** using Python `asyncio` tasks.
- **Target error-bar estimation** -- In the default automatic mode the
  workflow aims at a `target_error` (Ha, default `0.001`): a small *pilot*
  run is executed first; the statistical error is used to
  estimate the number of production steps required, together with the
  estimated wall time.

A single Python script can therefore express a full QMC pipeline -- from
wavefunction preparation (TREXIO -> `hamiltonian_data.h5`) through VMC
optimization, MCMC production sampling, and LRDMC extrapolation -- and run it
end-to-end with automatic restarts across interruptions.


## Architecture

```text
run_pipeline.py
      |
      v
  +----------+
  | Launcher |   DAG executor (asyncio)
  +----------+
     |  creates one asyncio.Task per ready node
     |
     +--> Container("vmc")
     |         +--> VMC_Workflow.configure() -> .run()
     |
     +--> Container("mcmc-prod")   <- starts when its dependencies are done
     |         +--> MCMC_Workflow.configure() -> .run()
     |
     +--> Container("lrdmc-ext")   <- independent nodes run in parallel
               +--> LRDMC_Ext_Workflow.configure() -> .run()
                       +--> LRDMC_Workflow (alat=0.50)  \
                       +--> LRDMC_Workflow (alat=0.40)   > in parallel
                       +--> LRDMC_Workflow (alat=0.25)  /
```

### Key components

| Class / Module | Role |
|---|---|
| `Launcher` | Executes a DAG of `Container` nodes in topological order, launching independent nodes in parallel. Provides `get_session_state()`, `get_current_job()`, `get_job_history()` for session introspection. |
| `Container` | Wraps any `Workflow` subclass in a dedicated project directory with state tracking (`workflow_state.toml`). |
| `FileFrom` / `ValueFrom` | Declare inter-workflow dependencies (files or computed values). |
| `WF_Workflow` | Converts a TREXIO file to `hamiltonian_data.h5` (runs `jqmc-tool trexio convert-to` locally). |
| `QE_Workflow` | Projects a Quantum ESPRESSO `prefix.save` directory to `hamiltonian_data.h5` (runs `jqmc-tool qe` locally); see "Twist-aware inputs" below. |
| `VMC_Workflow` | Jastrow / orbital optimization (`job_type=vmc`). |
| `MCMC_Workflow` | Production energy sampling (`job_type=mcmc`). |
| `LRDMC_Workflow` | Lattice-regularized diffusion Monte Carlo for a single $a$ value. |
| `LRDMC_Ext_Workflow` | Runs multiple `LRDMC_Workflow` instances at different lattice spacings and performs $a^2 \to 0$ extrapolation. |
| `ScientificPhase` | Enum defining the scientific phases of a workflow session (INIT -> SCF -> WF_BUILD -> VMC -> MCMC -> LRDMC -> COMPLETED). See [Phase management](#phase-management). |
| `WorkflowStatus` / `JobStatus` | Enums for workflow-level and per-job status values. See [Status enums](#status-enums). |
| `parse_vmc_output`, `parse_mcmc_output`, `parse_lrdmc_output`, `parse_lrdmc_ext_output`, `parse_input_params` | Parse the output of finished workflow directories into the result types `VMC_Diagnostic_Data` (with `VMC_Step_Data`), `MCMC_Diagnostic_Data`, `LRDMC_Diagnostic_Data`, `LRDMC_Ext_Diagnostic_Data` and `Input_Parameters`. |


### Inter-workflow data passing (`FileFrom` / `ValueFrom`)

`FileFrom` and `ValueFrom` declare inter-workflow dependencies inside a
`Container` definition.  At launch time, the `Launcher` resolves them
to actual paths or values.

#### `FileFrom(label, filename)`

Pass a **file** produced by an upstream workflow.  `filename` can be:

- A **static string** -- when the exact name is known at definition time:

  ```python
  FileFrom("vmc", "hamiltonian_data_opt_step_9.h5")
  ```

- A **`ValueFrom` object** -- when the name is determined at runtime
  (e.g. VMC early convergence produces a step number that cannot be
  predicted):

  ```python
  FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian"))
  ```

  Here the `ValueFrom` is resolved first, yielding the actual basename
  (e.g. `"hamiltonian_data_opt_step_91.h5"`), which is then used to
  locate the file in the upstream directory.

> **Important: file renaming.**  When a dynamic `FileFrom` resolves to
> a name that differs from what the downstream workflow expects (e.g.
> `hamiltonian_data_opt_step_91.h5` vs. `hamiltonian_data.h5`), use
> `rename_input_files` to map it to the expected name.  Entries of
> `None` keep the original name.  A pre-launch validation check will
> raise `FileNotFoundError` **before any job is submitted** if required
> files are missing.
>
> ```python
> Container(
>     label="mcmc",
>     dirname="02_mcmc",
>     input_files=[
>         FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian")),
>     ],
>     rename_input_files=["hamiltonian_data.h5"],
>     workflow=MCMC_Workflow(...),
> )
> ```
>
> `rename_input_files` is matched to `input_files` by position: entry
> *i* renames input *i*; a `None` entry, or a missing trailing entry,
> keeps the source basename.  With multiple input files, use `None` to
> skip renaming for specific entries.  For example, to stage an
> auxiliary file under its own name together with the optimized
> Hamiltonian:
>
> ```python
> input_files=[
>     "aux_settings.txt",                                       # index 0: kept as is
>     FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian")),  # index 1: renamed
> ],
> rename_input_files=[None, "hamiltonian_data.h5"],
> ```
>
> Do not stage the original trial `hamiltonian_data.h5` next to the
> optimized file (see [Pre-launch validation](#pre-launch-validation)).

#### `ValueFrom(label, key)`

Pass a **scalar value** from an upstream workflow's `output_values`
dict.  The available keys depend on the workflow class -- see the
table below.

#### Available `output_values` keys

Each workflow populates `output_values` on completion.  These keys
can be referenced by downstream workflows via
`ValueFrom("label", "key")`.

##### `VMC_Workflow`

| Key | Type | Description |
|---|---|---|
| `optimized_hamiltonian` | `str` | Basename of the hand-off: `"hamiltonian_data_opt_averaged.h5"`, the average of the last `handoff_average_steps` (20) optimization steps (default `handoff_mode="average"`), or the last `hamiltonian_data_opt_step_N.h5` (e.g. `"hamiltonian_data_opt_step_91.h5"`) with `handoff_mode="last"`; for single-twist and twist-averaged runs alike (see "Optimized-wavefunction hand-off" and "Twist-aware inputs" below). |
| `optimized_hamiltonian_handoff` | `str` | `"average"` or `"last"`: the `handoff_mode` that produced `optimized_hamiltonian`. |
| `optimized_hamiltonian_sources` | `list[str]` | Basenames of the dumps behind `optimized_hamiltonian`, in step order. |
| `optimized_hamiltonian_steps` | `list[int]` | Their optimization step numbers. |
| `checkpoint` | `str` | Basename of the restart checkpoint file. |
| `num_mcmc_steps` | `int` | MCMC steps per optimization step (fixed-step mode; equals the `num_mcmc_steps` argument). |
| `estimated_mcmc_steps` | `int` | Estimated MCMC steps per optimization step (automatic mode). |
| `energy` | `float` | Energy from the last optimization step (Ha). |
| `energy_error` | `float` | Statistical error on `energy` (Ha). |
| `signal_to_noise` | `float` | Average S/N over the trailing window (only when `target_snr` is set). |
| `signal_to_noise_last` | `float` | S/N of the last optimization step (only when `target_snr` is set). |
| `energy_slope` | `float` | Slope of energy vs. step (only when `energy_slope_sigma_threshold` is set). |
| `energy_slope_std` | `float` | Standard deviation of the energy slope (same condition). |
| `error` | `str` | Error message (only on failure). |

##### `MCMC_Workflow`

| Key | Type | Description |
|---|---|---|
| `energy` | `float` | VMC energy (Ha). |
| `energy_error` | `float` | Statistical error on `energy` (Ha). |
| `restart_chk` | `str` | Basename of the restart checkpoint file. |
| `forces` | `object` | Atomic forces (only when `atomic_force=True`). SWCT is applied when `use_swct=True`. |
| `stress` | `object` | Symmetric stress tensor sigma (Ha/bohr^3) (only when `compute_stress=True`; periodic systems only). Keys: `sigma_xx`/`sigma_xx_err` ... `sigma_yz`/`sigma_yz_err`. |
| `pressure` | `object` | Diagonal pressure P = -diag(sigma) plus hydrostatic mean (Ha/bohr^3) (only when `compute_stress=True`). Keys: `Px`/`Px_err`, `Py`/`Py_err`, `Pz`/`Pz_err`, `P_hydro`/`P_hydro_err`. |
| `num_mcmc_steps` | `int` | Measurement steps per production run (fixed-step mode; equals the `num_mcmc_steps` argument). |
| `estimated_steps` | `int` | Estimated total measurement steps (automatic mode). |
| `dE_mpc` / `dE_mpc_error` | `float` | MPC finite-size correction and its error (Ha), only for a run with MPC enabled (`mpc=True`). `energy` stays the Ewald value; the corrected energy is `energy + dE_mpc`. |
| `error` | `str` | Error message (only on failure). |

##### `LRDMC_Workflow`

| Key | Type | Description |
|---|---|---|
| `energy` | `float` | DMC energy (Ha). |
| `energy_error` | `float` | Statistical error on `energy` (Ha). |
| `alat` | `float` | Lattice spacing used for this run. |
| `restart_chk` | `str` | Basename of the restart checkpoint file. |
| `forces` | `object` | Atomic forces (only when `atomic_force=True`). SWCT is applied when `use_swct=True`. |
| `stress` | `object` | Symmetric stress tensor sigma (Ha/bohr^3) (only when `compute_stress=True`; periodic systems only). Keys: `sigma_xx`/`sigma_xx_err` ... `sigma_yz`/`sigma_yz_err`. Supported by both `lrdmc-tau` (GFMC_t) and `lrdmc-bra` (GFMC_n). Available through `LRDMC_Workflow` only; `LRDMC_Ext_Workflow` has no `compute_stress` argument. |
| `pressure` | `object` | Diagonal pressure P = -diag(sigma) plus hydrostatic mean (Ha/bohr^3) (only when `compute_stress=True`). Keys: `Px`/`Px_err`, `Py`/`Py_err`, `Pz`/`Pz_err`, `P_hydro`/`P_hydro_err`. |
| `estimated_steps` | `int` | Estimated total measurement steps (automatic mode), or `num_gfmc_projections` (fixed-step mode). |
| `num_projection_per_measurement` | `int` | GFMC projections per measurement: the given or calibrated value in GFMC_n mode; in GFMC_t mode the average number of projections measured in the run (when it can be parsed). |
| `time_projection_tau` | `float` | Imaginary-time projection step (GFMC_t mode only). |
| `dE_mpc` / `dE_mpc_error` | `float` | MPC finite-size correction and its error (Ha), only for a run with MPC enabled (`mpc=True`, `LRDMC_Workflow` only). |
| `error` | `str` | Error message (only on failure). |

##### `LRDMC_Ext_Workflow`

| Key | Type | Description |
|---|---|---|
| `extrapolated_energy` | `float` | Continuum-limit ($a^2 \to 0$) extrapolated energy (Ha). |
| `extrapolated_energy_error` | `float` | Statistical error on `extrapolated_energy` (Ha). |
| `per_alat_results` | `dict` | Per-alat energy/error results. |
| `nmpm_per_alat` | `list[dict]` | GFMC projections per measurement for each lattice spacing, as `{"alat": float, "nmpm": int}` records. Can be passed back as `num_projection_per_measurement` of another `LRDMC_Ext_Workflow`. |
| `errors` | `list[str]` | Error messages for alat runs that failed. |
| `error` | `str` | Top-level error message (only on failure). |

##### `WF_Workflow` and `QE_Workflow`

| Key | Type | Description |
|---|---|---|
| `hamiltonian_file` | `str` | Basename of the generated Hamiltonian file (the `hamiltonian_file` argument, default `"hamiltonian_data.h5"`). |


### Target error-bar estimation

`VMC_Workflow`, `MCMC_Workflow` and `LRDMC_Workflow` run in one of two
modes:

- **Automatic mode** (the default) -- the workflow aims at
  `target_error` (default `0.001` Ha).  A short *pilot* run is executed
  first and its error bar is used to estimate the number of production
  steps (see [Step estimation formula](#step-estimation-formula)).
- **Fixed-step mode** -- set `num_mcmc_steps` (`VMC_Workflow`,
  `MCMC_Workflow`) or `num_gfmc_projections` (`LRDMC_Workflow`,
  `LRDMC_Ext_Workflow`).  The error-bar pilot is skipped,
  `target_error` is ignored, and every one of the `max_continuation`
  production runs uses exactly that many steps.

A separate queue may be used for the pilot via `pilot_queue_label`
(defaults to `queue_label`).

`VMC_Workflow` and `MCMC_Workflow` also accept `Dt="auto"`. Before either
mode, a calibration stage (`_pilot_a/`) then runs one short MCMC job per trial
`Dt` on the pilot queue, fits the measured acceptance ratio against `1/Dt`,
and uses the `Dt` that reaches `target_acceptance_ratio` (default `0.5`) for
the pilot and production runs.

#### MCMC

1. **Pilot** (`_pilot/`) -- A short run of `pilot_steps` measurement
   steps (default 100).
2. **Production** -- Step count estimated from the pilot error bar.

#### VMC

1. **Pilot** (`_pilot/`) -- A short optimization of `pilot_vmc_steps`
   optimization steps (default 5) with `pilot_mcmc_steps` MCMC steps per
   optimization step (default 50).  The error bar of the last pilot step
   is used to estimate the MCMC steps per optimization step needed for
   `target_error`.
2. **Production** -- `num_opt_steps` optimization steps with the
   estimated MCMC steps per step.

#### LRDMC

`LRDMC_Workflow` has two projection modes:

- **GFMC_t** (the default, `job_type = "lrdmc-tau"`) -- continuous
  imaginary-time projection with step `time_projection_tau` (default
  `0.10`).  No calibration is performed; only the error-bar pilot runs.
- **GFMC_n** (`job_type = "lrdmc-bra"`) -- a fixed number of GFMC
  projections per measurement.  Selected when
  `target_survived_walkers_ratio` (a typical value is `0.97`) or
  `num_projection_per_measurement` is passed; `time_projection_tau` is
  then ignored.

In GFMC_n mode with `target_survived_walkers_ratio` (and no explicit
`num_projection_per_measurement`), an additional calibration stage
determines `num_projection_per_measurement` automatically:

1. **Calibration** (`_pilot_a/_pilot1` - `_pilot_a/_pilot3`, parallel) --
   Three short LRDMC runs of `pilot_steps` steps with
   `num_projection_per_measurement = int(Ne x k x (0.3/alat)^2)` (k=2,4,6;
   $N_e$ is the total electron count).  A straight line is fitted to the
   observed survived-walkers ratio as a function of
   `num_projection_per_measurement` and solved for the target ratio; the
   result is rounded up to an even integer (minimum 2).
2. **Error-bar pilot** (`_pilot_b`) -- A run of `pilot_steps` steps
   (default 100) with the calibrated `num_projection_per_measurement`;
   its error bar estimates the production step count.
3. **Production** (`_1`, `_2`, ...) -- Start from scratch, accumulate
   statistics until `target_error` is achieved.

In GFMC_t mode, or in GFMC_n mode with an explicit
`num_projection_per_measurement`, the calibration stage is skipped and
only the error-bar pilot (`_pilot_b`) and production are executed.
`num_projection_per_measurement` is written as `[lrdmc-bra]
num_mcmc_per_measurement`, which has no default in `jqmc`; in GFMC_n mode
the workflow never fills it in itself, so it is either passed explicitly or
calibrated as above.

The energy post-processing settings of the LRDMC workflows default to
`num_gfmc_warmup_steps = 25`, `num_gfmc_bin_blocks = 10` and
`num_gfmc_collect_steps = 5`.  These are also the enforced minimums:
smaller values raise `ValueError` when `LRDMC_Workflow` or
`LRDMC_Ext_Workflow` is constructed.

For `LRDMC_Ext_Workflow` (multi-alat extrapolation), every alat value
independently runs its own calibration (GFMC_n with
`target_survived_walkers_ratio` only), error-bar pilot, and production
in parallel.  There is no inter-alat interaction until the final
extrapolation step.

#### AO basis optimization

`VMC_Workflow` supports optimizing the atomic-orbital (AO) Gaussian
basis parameters alongside Jastrow / orbital coefficients.  Four
boolean flags control which basis parameters are included in the
optimization:

| Parameter | Default | Description |
|---|---|---|
| `opt_J3_basis_exp` | `None` | Optimize J3 (three-body Jastrow) AO Gaussian exponents |
| `opt_J3_basis_coeff` | `None` | Optimize J3 AO contraction coefficients |
| `opt_lambda_basis_exp` | `None` | Optimize Geminal AO Gaussian exponents |
| `opt_lambda_basis_coeff` | `None` | Optimize Geminal AO contraction coefficients |

For an `AGP_data` input, `opt_lambda_basis_exp` and
`opt_lambda_basis_coeff` cannot be combined with a positive
`fix_geminal_rank`: the AGP-to-AGPn rank conversion relies on an AO overlap
matrix that changing the radial basis would invalidate. J3 basis optimization
is independent of this geminal-rank restriction.

Primitives belonging to the same shell (same atom, same $l$, same
initial radial parameters) are constrained to share identical values
throughout optimization.  See {ref}`sec:vmc_basis_opt` for details on the
shell-sharing constraint mechanism.

When set to `True` or `False`, the value is written to the `[vmc]`
section of the generated input TOML.  When left as `None` (the
default), the `jqmc` TOML default (`false`) is filled in.

#### VMC convergence checks

After all production runs complete, the VMC workflow checks whether
the optimization has converged.  Two independent criteria are
available; when both are active, both must pass for convergence.

##### Signal-to-noise (S/N) check

Enabled when `target_snr` is set (not `None`).  The workflow
averages the signal-to-noise ratio (S/N = max(|f| / |std f|)) over
the last `snr_avg_window` optimization steps (default 5).  If there
are fewer steps than the window size, all available values are used.

The convergence criterion is:

$$
\overline{\text{S/N}}_{\text{last } W} \le \text{target\_snr}
$$

where $W$ is `snr_avg_window`.

##### Energy-slope check

Enabled when `energy_slope_sigma_threshold` is set (not `None`).
A weighted linear regression is fitted to the last
`energy_slope_window_size` optimization steps (default 5):

$$
E_k = a + b \cdot k + \varepsilon_k, \quad w_k = 1/\sigma_k^2
$$

The optimization is considered converged (plateau) when the slope
$b$ is not significantly negative:

$$
b \ge -\sigma_b \times \text{energy\_slope\_sigma\_threshold}
$$

If instead $b < -\sigma_b \times \text{threshold}$, the energy is
still decreasing and optimization has not yet plateaued.

##### Combined verdict

| `target_snr` | `energy_slope_sigma_threshold` | Behavior |
|:---:|:---:|:---|
| `None` | `None` | No convergence check; always succeeds |
| set | `None` | S/N check only |
| `None` | set | Energy-slope check only |
| set | set | Both must pass |

All numerical values (averaged S/N, slope, slope std) are recorded
in `output_values` for downstream inspection.

In fixed-step mode (`num_mcmc_steps` is set), the convergence checks
are not performed.

##### Early exit on convergence

When convergence criteria are set, the VMC production loop exits
early as soon as all active checks pass, instead of always running
all `max_continuation` production runs (each run performs
`num_opt_steps` optimization steps).  `max_continuation` defaults to 1
for `VMC_Workflow`, `MCMC_Workflow` and `LRDMC_Workflow` (5 for
`LRDMC_Ext_Workflow`), so raise it if you want the loop to be able to
continue until convergence.  This avoids wasting compute
time once the optimization has plateaued.

On re-runs with changed criteria (e.g. lowering `target_snr`),
convergence is re-evaluated from the fetched results before
submitting new jobs.  If the existing results already satisfy the
new criteria, the workflow completes immediately without launching
additional runs.

With the averaged hand-off (`handoff_mode="average"`, the default) the
loop does not exit early while fewer than `handoff_average_steps`
optimization dumps exist: it runs the next continuation instead, so
that the hand-off can average the requested number of steps
(`handoff_allow_fewer=True` lifts this).

When no convergence criterion is set (`target_snr = None` and
`energy_slope_sigma_threshold = None`), all `max_continuation`
runs are executed unconditionally.  If a convergence criterion is set
but convergence is **not** achieved after all `max_continuation` runs,
the workflow returns `FAILED`.

#### Step estimation formula

The required number of production steps is estimated via

$$
N_{\text{eff,prod}} = N_{\text{eff,pilot}}
  \times \left(\frac{\sigma_{\text{pilot}}}{\sigma_{\text{target}}}\right)^2
  \times \frac{W_{\text{pilot}}}{W_{\text{prod}}}
$$

where $N_{\text{eff}} = N_{\text{total}} - N_{\text{warmup}}$ is the
number of steps that actually contribute to statistics (warmup steps
are discarded during post-processing via `-w`), and
$W = \text{walkers\_per\_MPI} \times \text{num\_MPI}$ is the
total number of walkers.  The ratio
$W_{\text{pilot}} / W_{\text{prod}}$ (the *walker ratio*) accounts
for the pilot queue using fewer (or more) MPI processes than the
production queue.  The number of MPI processes is read from
`queue_data.toml` (`num_cores`, or `mpi_per_node x nodes` as
fallback).

$N_{\text{eff,prod}}$ is truncated to an integer and is never smaller
than the number of bin blocks (`num_mcmc_bin_blocks` or
`num_gfmc_bin_blocks`).  The total production steps are then
$N_{\text{prod}} = N_{\text{eff,prod}} + N_{\text{warmup}}$.

The estimated production wall time is based on the **net** computation
time parsed from the pilot output (``Net GFMC time`` for LRDMC,
``Net total time for MCMC`` for MCMC/VMC).  Only this net portion
scales with the step count; overhead (JIT compilation, file I/O,
queue wait) is treated as a constant:

$$
T_{\text{prod}} \approx (T_{\text{wall,pilot}} - T_{\text{net,pilot}})
  + T_{\text{net,pilot}} \times \frac{N_{\text{prod}}}{N_{\text{pilot}}}
$$

The workflow logs a summary of the estimate.  The following example
is illustrative (the inputs are chosen for round numbers, not taken
from a real run) and is consistent with the formulas above: a GFMC_n
pilot of 100 steps with 25 warm-up steps, a pilot error of 0.004 Ha,
48 pilot and 480 production MPI processes give
$N_{\text{eff,prod}} = 75 \times (0.004/0.001)^2 \times 0.1 = 120$ and
$N_{\text{prod}} = 120 + 25 = 145$; with a pilot wall time of 300 s of
which 120 s is net computation,
$T_{\text{prod}} \approx 180 + 120 \times 145/100 = 354$ s.

```text
-- LRDMC Step Estimation Summary (a=0.25) --------
  pilot steps       = 100
  warmup steps      = 25
  pilot error       = 0.004 Ha
  target error      = 0.001 Ha
  mode              = GFMC_n
  nmpm              = 32
  pilot MPI procs   = 48
  prod. MPI procs   = 480
  walker ratio      = 0.1
  estimated steps   = 145
  pilot wall time   = 5m 0s
  pilot net time    = 2m 0s
  est. prod. time   = 5m 54s
--------------------------------------------------
```

The first production run starts from scratch (no restart from the
pilot checkpoint, which lives in the ``_pilot_b/`` subdirectory).
After each production run, the error bar is re-evaluated.
If it exceeds the target, additional steps are estimated and a
continuation run is launched automatically, up to `max_continuation`
times.

#### Target error not met

If `max_continuation` runs complete but the statistical error still
exceeds `target_error`, the workflow logs a **warning** and returns
`COMPLETED` (not `FAILED`).  The calculation itself succeeded; only
the error-bar criterion was not fully met.  This applies to both
`MCMC_Workflow` and `LRDMC_Workflow`.

#### Continuation with tighter target error

When re-running a pipeline with a stricter `target_error` (e.g.
lowering from `1e-3` to `5e-4`), the workflow detects that the
cached error from previous runs exceeds the new target and
automatically re-estimates additional steps from the accumulated
data.  Continuation runs are launched from where the previous
execution left off, up to `max_continuation`.


### Optimized-wavefunction hand-off (`handoff_mode`)

`VMC_Workflow` publishes, as `output_values["optimized_hamiltonian"]`, the
Hamiltonian that downstream MCMC and LRDMC stages sample. By default it is
the uniform average of the wavefunction parameters of the last 20
optimization steps, the same hand-off as the command-line tutorials
(`examples/copy_optimized_hamiltonian.py --average-last 20`; see
{ref}`subsec:opt_step_averaging`).

| Argument | Default | Meaning |
|---|---|---|
| `handoff_mode` | `"average"` | `"average"`: average the last `handoff_average_steps` dumps into `hamiltonian_data_opt_averaged.h5`. `"last"`: publish the dump with the largest step number itself (the hand-off before averaging became the default). |
| `handoff_average_steps` | `20` | Number of last optimization steps to average. |
| `handoff_allow_fewer` | `False` | Average all available dumps when fewer than `handoff_average_steps` exist, instead of refusing. |

At the end of the production runs the workflow selects the
`handoff_average_steps` dumps `hamiltonian_data_opt_step_N.h5` with the
largest step numbers N in the VMC directory (a continuation numbers its dumps
after the completed steps, so a run and its continuations are averaged
together) and runs `jqmc-tool vmc average-wf` on them in that directory, with
the Python interpreter of the workflow. The command and its checks are
described under `average-wf` in {ref}`api_reference_cli_tool_link`.

- **Dump frequency.** Only dumped steps can be averaged. The default
  `wf_dump_freq` (unset, i.e. the jQMC default 1) dumps every step. With
  `handoff_mode="average"` any other `wf_dump_freq` raises `ValueError` when
  the workflow is constructed. The dump frequency does not change the
  optimization itself.
- **Too few steps.** When `num_opt_steps * max_continuation` is smaller than
  `handoff_average_steps`, the constructor raises `ValueError` (not checked
  with `external_restart_chk`, whose staged directory may hold earlier dumps).
  At the end of the run, fewer dumps than `handoff_average_steps` raise
  `RuntimeError`, unless `handoff_allow_fewer=True`.
- **Missing steps.** Selected dumps that are not consecutive steps (deleted
  dumps, or dumps from a run with another dump frequency) raise
  `RuntimeError`, also with `handoff_allow_fewer=True`.
- **Refusals of the tool.** When `jqmc-tool vmc average-wf` refuses the
  dumps (for example a non-finite parameter of a diverged step, an MO sign
  or phase jump, or files that are not steps of one optimization), the
  workflow raises `RuntimeError` with the tool's `Error:` message and
  publishes nothing; an averaged file left by an earlier attempt is removed
  first. The tool's warnings go to the workflow log.
- **No fallback.** None of these cases falls back to the single last dump.
  Choose `handoff_mode="last"` explicitly for that hand-off.
- **Failed runs.** A VMC run that ends `FAILED` before the hand-off (for
  example a non-finite energy) does not average its steps.
- **Provenance.** `optimized_hamiltonian_handoff`,
  `optimized_hamiltonian_sources` and `optimized_hamiltonian_steps` record the
  hand-off in `output_values` and in the `[result]` fields of
  `workflow_state.toml`. The averaged file also carries the root attributes
  `opt_step_average_sources` and `opt_step_average_steps` written by the tool.

The averaged file of a twist-averaged run is again a complete multi-twist
(combined) Hamiltonian. A completed VMC directory keeps its recorded
hand-off; use a fresh VMC directory (and fresh downstream directories) to
change `handoff_mode` or `handoff_average_steps`. A `cleanup_patterns` entry
such as `"hamiltonian_data_opt*.h5"` would also delete the published file;
use `"hamiltonian_data_opt_step_*.h5"` to remove only the step dumps.

### Pre-launch validation

Before any job is submitted, the engine verifies that all required
files are present in the project directory.  Every resolved entry in
`input_files` (after renaming) must exist.  Workflow-internal files
(e.g. `hamiltonian_file`) are **not** checked because some workflows
(e.g. `WF_Workflow`) *produce* them rather than consume them.

If any file is missing, a `FileNotFoundError` is raised immediately
with a message listing the missing files:

```text
FileNotFoundError: [mcmc-N2-0.80] Required file(s) missing in '05_mcmc/'
before workflow launch: ['hamiltonian_data.h5'].
Check that input_files and rename_input_files are configured correctly.
```

This checks staged-file existence, not whether a workflow consumes the
staged name. For a dynamic optimized dump, explicitly use
`rename_input_files=["hamiltonian_data.h5"]` and set
`hamiltonian_file="hamiltonian_data.h5"`. Stage only the optimized
`FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian"))`, not the
original trial as a second input. Otherwise an original trial can remain
present and silently be consumed instead. Use a fresh downstream directory
when changing this handoff; existing files are not refreshed automatically.

Additionally, when the project directory already exists from a
previous interrupted run, any input files that are missing (but
available from the source) are automatically copied in.  Existing
files are not overwritten.


### Continuing an externally produced optimization

`VMC_Workflow(external_restart_chk=...)` continues an optimization that a
*different* run produced.  Stage the checkpoint through the container's
`input_files` (a twist checkpoint **directory** is copied recursively) and
name it in the workflow:

```python
Container(
    label="vmc-continue",
    dirname="vmc_continue",
    input_files=[
        "/path/to/previous_run/hamiltonian_data.h5",
        "/path/to/previous_run/restart.h5",   # directory: shared_state.h5 + twist_NNN.h5
    ],
    workflow=VMC_Workflow(
        server_machine_name="my-cluster",
        external_restart_chk="restart.h5",
        num_opt_steps=100,          # ADDITIONAL steps beyond the checkpoint
        num_mcmc_steps=40,
        # remaining optimizer options exactly as in the original run
    ),
)
```

Production run 1 is then emitted with `restart = true` /
`restart_chk = "restart.h5"` and the checkpoint is uploaded with the job.
On a restart, `num_opt_steps` is the number of **additional** optimization
steps to run beyond those already stored in the checkpoint; this holds for
both single-twist and twist-averaged checkpoints.

Because the checkpoint pins the keys that are *frozen on restart*
(`mcmc_seed`, `number_of_walkers`, `num_mcmc_per_measurement`, `Dt`,
`epsilon_AS`), the workflow must be constructed with the same values the
original run used; otherwise the `jqmc` restart guard rejects the
generated TOML with a `ValueError`.

Without `external_restart_chk` the behavior is unchanged: production run 1
always starts from scratch and only runs 2..`max_continuation` restart.


### Post-completion cleanup (`cleanup_patterns`)

When large checkpoint files (e.g. `restart.h5`, `hamiltonian_data_opt_step_*.h5`)
are no longer needed after a successful run, the `cleanup_patterns`
parameter can be used to automatically delete them.

```python
MCMC_Workflow(
    server_machine_name="my-cluster",
    target_error=0.001,
    cleanup_patterns=["restart.h5", "hamiltonian_data_opt_step_*.h5"],
)
```

**Behavior:**

- `cleanup_patterns` accepts a list of glob patterns (e.g.
  `["restart.h5", "hamiltonian_data_opt_step_*.h5"]`).
- Matching is **recursive** -- patterns are applied to the workflow
  directory **and** all subdirectories (e.g. `_pilot/`, `_pilot_a/`,
  `_pilot_b/`).
- **Local files** matching the patterns are always deleted.
- **Remote files** are deleted only when the workflow targets a remote
  machine (`server_machine_name` is not `"localhost"`).
- Cleanup runs **after** `CompletionStatus.OK` is confirmed -- it never
  touches files while the workflow might still need them for
  continuation.
- Cleanup failures are logged as warnings and never cause a completed
  workflow to fail.
- The default (`None`) performs no cleanup.

This is especially useful for massively-parallel MCMC runs where
`restart.h5` can grow to tens of gigabytes.

All workflow classes (`VMC_Workflow`, `MCMC_Workflow`,
`LRDMC_Workflow`, `LRDMC_Ext_Workflow`) support this parameter.
For `LRDMC_Ext_Workflow`, the child `LRDMC_Workflow` runs do **not**
clean up (the extrapolation still needs every alat's `restart.h5`);
the patterns are applied once, recursively from the `LRDMC_Ext_Workflow`
directory, after the extrapolation has finished.
`workflow_state.toml` is never deleted, even if a pattern matches it.


### Restart behavior

Every job is recorded in `workflow_state.toml` with a lifecycle:

```text
submitted  ->  completed  ->  fetched
```

On restart, the engine checks each job's status:

- **`fetched`** -- Input generation *and* submission are both skipped.
- **`submitted`** / **`completed`** -- Input is not regenerated; the job
  is resumed (polled or fetched).
- **No record** -- A fresh input file is generated and the job is submitted.

This means a pipeline can be interrupted at any point (Ctrl-C, node
failure, wall-time limit) and simply re-run; it will pick up exactly
where it left off.

When a job leaves the scheduler queue, the engine automatically collects
**job accounting** data (if `jobacct` is configured in `machine_data.yaml`)
before fetching output files.  The accounting command and output file
path are stored in the corresponding `[[jobs]]` record.  See
[Job accounting](#job-accounting).


### Phase management

A workflow session progresses through a sequence of **scientific phases**
defined by the `ScientificPhase` enum (module `_phase`):

```text
INIT -> SCF -> WF_BUILD -> VMC_PILOT -> VMC -> MCMC_PILOT -> MCMC
                                                                |
                                                                v
                        COMPLETED <- LRDMC_FIT <- LRDMC <- LRDMC_PILOT
```

Not every pipeline uses every phase (e.g. a VMC-only pipeline skips
LRDMC phases).  The allowed transitions are defined in
`PHASE_TRANSITIONS` -- for example, from `VMC` you may move to
`VMC_PILOT`, `VMC`, `MCMC_PILOT`, `MCMC`, `LRDMC_PILOT`, `LRDMC`, or
`COMPLETED`.  Use `can_advance(current, target)` to test a transition.

Each phase has a list of **allowed actions** (`PHASE_ALLOWED_ACTIONS`)
that further depends on the current `WorkflowStatus`:

- When `status == RUNNING`, configuration actions (`configure_*`) are
  filtered out.
- When `status == FAILED` or `status == CANCELLED`, only recovery
  actions (`recover_*`) and `rollback_phase` are available.

A set of **always-allowed actions** (`advance_phase`, `rollback_phase`,
`close_session`, `register_artifact`, `mark_unhealthy`) is appended
regardless of phase/status.

These rules are meant for external tools that drive a workflow session.
`allowed_actions(phase, status)` returns the permitted actions.  The
helper `require_action(action, phase, status)` (import it from
`jqmc_workflow._phase`; it is not exported at package level) raises
`ValueError` immediately if the action is not permitted.


### Status enums

Workflow status and job status are represented by `str`-based enums
(`WorkflowStatus`, `JobStatus`) so they remain human-readable in
`workflow_state.toml` and can be compared directly with strings.

**WorkflowStatus** values:

| Value | Meaning |
|---|---|
| `pending` | Not yet started |
| `copying` | Input files being transferred |
| `submitted` | Job submitted to scheduler |
| `running` | Execution in progress |
| `completed` | Finished successfully |
| `failed` | Terminated with an error |
| `cancelled` | Manually cancelled |

**JobStatus** values (per-job, stored in `[[jobs]]` of
`workflow_state.toml`):

| Value | Meaning |
|---|---|
| `submitted` | Job submitted |
| `completed` | Job finished (output not yet fetched) |
| `fetched` | Output files retrieved |
| `failed` | Job failed |
| `cancelled` | Job cancelled |

Each `[[jobs]]` record contains:

| Field | Description |
|---|---|
| `input_file` | Basename of the generated TOML input file (`input_{jobname}_{step}_{run_id}.toml`) |
| `output_file` | Basename of the stdout capture file (`output_{jobname}_{step}_{run_id}.out`) |
| `job_id` | Scheduler job ID (or `"local"` for local runs) |
| `server_machine` | Machine name |
| `status` | One of the `JobStatus` values above |
| `submitted_at` | ISO 8601 timestamp |
| `step` | Step index (0 = pilot, 1, 2, ... = production) |
| `run_id` | Short hex identifier for the job |
| `completed_at` | ISO 8601 timestamp (set on completion) |
| `fetched_at` | ISO 8601 timestamp (set on fetch) |
| `job_stdout` | Scheduler stdout path (queuing systems only) |
| `job_stderr` | Scheduler stderr path (queuing systems only) |
| `job_acct_command` | Accounting command executed (queuing systems only) |
| `job_acct_file` | Path to raw accounting output file (queuing systems only) |


### Artifact registry

The engine records file lineage in the `[[artifacts]]` array of
`workflow_state.toml`.  Each entry tracks:

| Field | Description |
|---|---|
| `filename` | Basename of the artifact file |
| `produced_by_job` | Input file that produced this artifact |
| `produced_at` | ISO 8601 timestamp |
| `artifact_type` | `"file"` (default) |
| `upstream` | Optional list of `{label, file}` dicts tracing the dependency chain |

Use `register_artifact()` to add an entry, `get_artifact_lineage()` to
look up a single file, and `get_artifact_registry()` to list all
artifacts.


### Input staleness detection

When a `Container` completes successfully, it records the **SHA-256
content hash** of each input file in the `[input_fingerprints]`
section of `workflow_state.toml`.  On subsequent runs, if the container
is already `completed` or `running`, the engine compares the current
input files against the recorded fingerprints.

If any input has changed (e.g. an upstream VMC produced a new optimized
wavefunction), the engine logs a warning:

```text
[mcmc-prod] Inputs have changed since the previous run. Delete 'mcmc_prod/' to re-run with the updated inputs.
```

The container is **not** automatically re-run -- the user must
manually delete the stale directory.  This conservative approach
avoids the risk of mixing old and new job data on the remote server.

Staleness is tracked per-file by SHA-256 content hashing.  If an
upstream workflow re-runs but produces a byte-identical output file,
no warning is triggered.


### Error recording

When a workflow fails, the `[error]` section of `workflow_state.toml`
is populated via `set_error()`:

```toml
[error]
message = "Job killed after 86400s"
exception_type = "RuntimeError"
traceback = "..."
```

The engine records raw data only -- failure classification and recovery
strategy are responsibilities of external tooling.


### Job accounting

For queuing systems (PBS, Slurm, Fujitsu TCS, etc.), the engine can
collect scheduler accounting data after a job leaves the queue.  This
is configured via the optional `jobacct` field in `machine_data.yaml`
(see [machine parameters](#machine-parameters)).

The engine executes `{jobacct} {job_id}` and writes the raw output to
a separate file `job_accounting_{job_id}.txt`.  The accounting command
and file path are stored per-job in the `[[jobs]]` record:

```toml
[[jobs]]
input_file = "input_vmc-H2-0.74_1_aebf13bd.toml"
output_file = "output_vmc-H2-0.74_1_aebf13bd.out"
job_id = "12345"
server_machine = "my-cluster"
status = "fetched"
step = 1
run_id = "aebf13bd"
job_stdout = "job_vmc-H2-0.74_aebf13bd.o"
job_stderr = "job_vmc-H2-0.74_aebf13bd.e"
job_acct_command = "sacct --format=State,ExitCode,MaxRSS,Elapsed,Timelimit -P --noheader -j 12345"
job_acct_file = "job_accounting_12345.txt"
```

No parsing or interpretation is performed -- that responsibility belongs
to external tooling.  If `jobacct` is not configured, the
`job_acct_command` and `job_acct_file` fields are simply absent.


### Session state queries

The `Launcher` provides three methods for programmatic introspection
(useful for external drivers and monitoring tools):

| Method | Returns |
|---|---|
| `get_session_state()` | Dict with per-workflow summaries, dependency graph, and progress counters (completed / failed / running / pending / total). |
| `get_current_job()` | The first workflow with status `running` or `submitted`, or `None`. |
| `get_job_history()` | Flat, chronologically-sorted list of all jobs across all workflows. |


### Machine catalog

Two helper functions are available for machine discovery:

| Function | Description |
|---|---|
| `list_machines()` | Returns a list of dicts summarizing all machines defined in `machine_data.yaml` (name, machine_type, queuing, ssh_host, workspace_root). |
| `probe_environment(machine_name)` | Tests connectivity to the named machine (SSH for remote, always reachable for local). Returns `{"reachable": True/False, ...}`. |


## Configuration

Configuration files are managed in the following directory hierarchy.
The engine looks for a project-local override first, then falls back to
the user-global directory:

1. `./jqmc_setting_local/` -- project-local override (if it exists in CWD)
2. `~/.jqmc_setting/` -- user-global default

On the very first run, if neither directory exists, the template shipped
with the package is copied to `~/.jqmc_setting/` and the user is asked
to edit it.

### Directory structure

```text
~/.jqmc_setting/
+-- machine_data.yaml           # Server machine definitions
+-- localhost/                  # Settings for localhost
|   +-- queue_data.toml
|   +-- submit_mpi.sh           # (name is user-defined)
+-- my-cluster/                 # Settings for a remote cluster (nickname)
|   +-- queue_data.toml
|   +-- submit_mpi.sh
+-- ...
```

### `machine_data.yaml`

A YAML file that defines each compute machine. The top-level keys are
nicknames (arbitrary labels); they do **not** need to match the SSH host.
Remote machines use the `ssh_host` field to specify the SSH connection
target.

#### Example

```yaml
localhost:
  machine_type: local
  queuing: false
  workspace_root: /home/user/jqmc_work
  jobsubmit: bash                    # runs the generated submit script

my-cluster:                          # nickname (freely chosen)
  ssh_host: pbs-cluster              # Host alias in ~/.ssh/config
  machine_type: remote
  queuing: true
  workspace_root: /home/user/jqmc_work
  jobsubmit: /opt/pbs/bin/qsub
  jobcheck: /opt/pbs/bin/qstat
  jobdel: /opt/pbs/bin/qdel
  jobnum_index: 0
  jobacct: /opt/pbs/bin/qstat -xf

my-slurm:                            # Slurm example
  ssh_host: slurm-cluster
  machine_type: remote
  queuing: true
  workspace_root: /home/user/jqmc_work
  jobsubmit: sbatch
  jobcheck: squeue --noheader
  jobdel: scancel
  jobnum_index: 3
  jobacct: sacct --format=State,ExitCode,MaxRSS,Elapsed,Timelimit -P --noheader -j

my-fujitsu:                          # Fujitsu TCS example
  ssh_host: fujitsu
  machine_type: remote
  queuing: true
  workspace_root: /home/user/jqmc_work
  jobsubmit: /usr/local/bin/pjsub
  jobcheck: /usr/local/bin/pjstat
  jobdel: /usr/local/bin/pjdel
  jobnum_index: 5
  jobacct: pjstat -H -s --choose jid,st,ec,elp,pc --data
```

#### Machine parameters

| Key | Type | Required | Description |
|---|---|---|---|
| `ssh_host` | String | When `machine_type: remote` | SSH connection target. Typically a `Host` alias defined in `~/.ssh/config`. |
| `machine_type` | `"local"` or `"remote"` | Yes | `local`: execute on the same host. `remote`: connect via SSH (Paramiko). |
| `queuing` | Boolean | Yes | `true`: use a batch scheduler. `false`: direct execution. |
| `workspace_root` | String (path) | Yes | Root directory for file management. Upload/download paths are resolved relative to this directory. Must be set on **both** localhost and the remote server. |
| `jobsubmit` | String (command) | Yes | Command that runs or submits the generated job script (e.g. `qsub`, `sbatch`, `pjsub`). Required for every machine, also with `queuing: false`, where `bash` (or `sh`) runs the script directly. |
| `jobcheck` | String (command) | When `queuing: true` | Command that lists the jobs in the queue (e.g. `qstat`, `squeue --noheader`, `pjstat`). A job counts as running while its ID (or the ID's part before the first dot) appears as a whitespace-separated word in the output; the column layout does not matter. A non-zero exit status fails the workflow. An empty output counts as "job done" only after three consecutive polls (`poll_interval` apart) return it, so a transiently empty listing does not end a running job. |
| `jobdel` | String (command) | When `queuing: true` | Command to cancel a job (e.g. `qdel`, `scancel`, `pjdel`). |
| `jobnum_index` | Integer | When `queuing: true` | 0-based column index of the job ID in the output of `jobsubmit`. For PBS (`42.server`), use `0`. For Slurm (`Submitted batch job 42`), use `3`. A word at this index that contains no digit is not accepted as a job ID: the submission fails with the raw `jobsubmit` output. |
| `jobacct` | String (command) | No | Scheduler accounting command **with flags**. The engine executes `{jobacct} {job_id}` after a job leaves the queue and saves the raw output to `job_accounting_{job_id}.txt`. No parsing is performed. Because the job ID is appended at the end, a flag that takes the job ID must come last. Examples: `sacct --format=State,ExitCode,MaxRSS,Elapsed,Timelimit -P --noheader -j` (Slurm), `qstat -xf` (PBS), `pjstat -H -s --choose jid,st,ec,elp,pc --data` (Fujitsu TCS). If omitted, no accounting data is collected. |
| `ip` | String | No | IP address or hostname of the remote machine. Usually the SSH alias in `~/.ssh/config` is used instead. |

#### SSH connection

For `machine_type: remote`, the `ssh_host` field specifies the SSH
connection target.  This is typically a `Host` alias defined in
`~/.ssh/config`.  The engine reads `~/.ssh/config` via Paramiko to
resolve `HostName`, `User`, `IdentityFile`, `Port`, and `ProxyJump`
(multi-hop) settings.
Your SSH key must be accessible without a passphrase prompt (use
`ssh-agent` or a passphrase-less key).

> **Note:** `CanonicalizeHostname` in `~/.ssh/config` may trigger a
> Paramiko bug.
> The engine automatically works around this by removing the directive
> in-memory before connecting.


### `queue_data.toml`

Defines batch queue presets for each machine. Written in TOML format with
queue labels as table keys.

#### Example

```toml
[default]
    submit_template = 'submit_mpi.sh'
    max_job_submit = 10
    queue = 'batch'
    num_cores = 120
    omp_num_threads = 1
    nodes = 1
    mpi_per_node = 120
    max_time = '24:00:00'

[small]
    submit_template = 'submit_mpi.sh'
    max_job_submit = 10
    queue = 'batch'
    num_cores = 32
    omp_num_threads = 1
    nodes = 1
    mpi_per_node = 32
    max_time = '01:00:00'

[large]
    submit_template = 'submit_mpi.sh'
    max_job_submit = 10
    queue = 'batch'
    num_cores = 480
    omp_num_threads = 1
    nodes = 4
    mpi_per_node = 120
    max_time = '24:00:00'
```

#### Queue parameters

| Key | Type | Required | Description |
|---|---|---|---|
| `submit_template` | String | Yes | Name of the job submission script template file placed in the same directory (e.g. `"submit_mpi.sh"`, `"submit_gpu.sh"`). |
| `max_job_submit` | Integer | Yes | Maximum number of jobs this driver keeps in this scheduler queue at once (see below). A submission that would exceed it waits and retries every `poll_interval` seconds of the workflow until a slot frees; it does not fail the workflow. |
| `max_job_submit_scope` | `"driver"` or `"user"` | No (default `"driver"`) | What `max_job_submit` counts. `"driver"`: the jobs submitted (or resumed) by this driver process for this machine and `queue` that are still listed by `jobcheck`; jobs of other drivers or of manual submissions on the same account are not counted. `"user"`: every `jobcheck` line that shows both your login name and the `queue` value, i.e. all of the account's jobs in the queue; needs a `jobcheck` whose output contains the user name. |
| *custom keys* | Any | No | Any additional key-value pairs. These are substituted into job script templates as `_KEY_` (upper-case). Common keys: `num_cores`, `omp_num_threads`, `nodes`, `mpi_per_node`, `max_time`, `queue`, `account`, `partition`. |

#### Job limit

`max_job_submit` throttles a driver; it is not an error condition. All
workflows of one `Launcher` share the limit of a queue: the driver records the
job IDs it submits (and those it resumes from `workflow_state.toml`) and, before
each submission, counts the recorded IDs that `jobcheck` still lists, keyed by
machine and `queue` (the queue label when `queue` is not set). This works with
any `jobcheck` output that shows the job ID, including schedulers whose job
list shows the group instead of the user. When a driver restarts, the
`Launcher` records the jobs saved as `submitted` in its workflows' project
directories that `jobcheck` still lists before any workflow submits, so jobs
left in the queue by the previous run keep their slots. Two drivers that share
an account each keep their own limit; set `max_job_submit_scope = "user"` to count the
whole account's jobs in the queue instead. A job that `jobsubmit` rejects (for
example a site limit) still fails the workflow.

#### TOML format notes

- Strings must be quoted: `queue = "small"`.
- Booleans: lowercase `true` / `false` only.
- Time values (e.g. `max_time`) **must** be quoted to avoid TOML
  local-time parsing: `max_time = "24:00:00"`.


### Job script templates

Shell scripts placed in the per-machine directory, referenced by the
`submit_template` key in `queue_data.toml`.  Template variables
(written as `_KEY_` with underscores) are replaced at submission time.
You can name the file anything you like (e.g. `submit_gpu.sh`).

#### Predefined variables

| Variable | Description | Default |
|---|---|---|
| `_COMMAND_` | Executable + args, uniformly `jqmc <input>.toml` for the workflow types that go through the queue template (MCMC / VMC / LRDMC, `job_type = "mcmc" / "vmc" / "lrdmc-bra" / "lrdmc-tau"`). Constructed by the Workflow subclass's `_get_command(input_file)` hook. | (set by workflow) |
| `_INPUT_` | Path to the jqmc input TOML file. Retained for backward compatibility with custom templates; new templates should use `_COMMAND_` instead, which already embeds the input filename for the jqmc-based workflows. | (set by workflow) |
| `_OUTPUT_` | Path to the jqmc stdout+stderr capture file | `"out.o"` |
| `_JOBNAME_` | Job name | `"jqmc-wf"` |
| `_JOB_STDOUT_` | Path for the **scheduler** stdout file (e.g. PBS `-o`, Slurm `--output`) | `"job_{jobname}_{run_id}.o"` |
| `_JOB_STDERR_` | Path for the **scheduler** stderr file (e.g. PBS `-e`, Slurm `--error`) | `"job_{jobname}_{run_id}.e"` |

##### The `_COMMAND_` placeholder and `_get_command`

Templates use `_COMMAND_` as the single entry point for the
executable invocation. The submit script looks like:

```bash
mpirun -np _NUM_CORES_ _COMMAND_ > ${OUTPUT} 2>&1
```

Each `Workflow` subclass implements `_get_command(input_file) -> str`
which returns the executable + args (no `mpirun`, no redirection --
the template owns both). The currently shipped subclasses produce:

| Workflow | `_get_command` returns | `job_type` |
|---|---|---|
| `MCMC_Workflow` | `jqmc <input>.toml` | `"mcmc"` |
| `VMC_Workflow` | `jqmc <input>.toml` | `"vmc"` |
| `LRDMC_Workflow` | `jqmc <input>.toml` | `"lrdmc-bra"` / `"lrdmc-tau"` |
| `WF_Workflow` | (none -- runs `jqmc-tool trexio convert-to` via `subprocess.run` directly, no template) | -- |
| `QE_Workflow` | (none -- runs `jqmc-tool qe` via `subprocess.run` directly, no template; the Fourier-space projection in `jqmc.pw_pgto_projection` finishes in seconds so no MPI / cluster dispatch is needed) | -- |
| `LRDMC_Ext_Workflow` | (none -- orchestrator, defers to child `LRDMC_Workflow` jobs) | -- |

The TOML schema for each `job_type` is in `jqmc.jqmc_miscs.cli_parameters`.

`_JOB_STDOUT_` and `_JOB_STDERR_` allow the engine to track where the
scheduler writes its output, which is useful for failure diagnosis.
If these placeholders are not present in the template the scheduler's
default naming convention is used (backward-compatible).
The file paths are recorded per-job in the `[[jobs]]` records of
`workflow_state.toml` (as `job_stdout` and `job_stderr`).

> **Note:** `job_stdout` and `job_stderr` are treated as **optional**
> during fetch.  If the files do not exist on the remote server (e.g.
> the job script template does not include `#SBATCH --output` /
> `#SBATCH --error` directives), the engine logs a warning and
> continues instead of raising an error.  All other output files
> remain mandatory.

#### Custom variables

All keys from `queue_data.toml` are available in upper-case with
surrounding underscores. For example, `num_cores = 48` becomes
`_NUM_CORES_` in the template.

#### Example (`submit_mpi.sh` for PBS)

```bash
#!/bin/sh
#PBS -N _JOBNAME_
#PBS -q _QUEUE_
#PBS -l nodes=_NODES_:ppn=_MPI_PER_NODE_
#PBS -l walltime=_MAX_TIME_
#PBS -o _JOB_STDOUT_
#PBS -e _JOB_STDERR_

export OMP_NUM_THREADS=_OMP_NUM_THREADS_
OUTPUT=_OUTPUT_

cd ${PBS_O_WORKDIR}
mpirun -np _NUM_CORES_ _COMMAND_ > ${OUTPUT} 2>&1
```

#### Example (`submit_mpi.sh` for Slurm)

```bash
#!/bin/bash
#SBATCH --job-name=_JOBNAME_
#SBATCH --partition=_QUEUE_
#SBATCH --nodes=_NODES_
#SBATCH --ntasks=_NUM_CORES_
#SBATCH --time=_MAX_TIME_
#SBATCH --output=_JOB_STDOUT_
#SBATCH --error=_JOB_STDERR_

export OMP_NUM_THREADS=_OMP_NUM_THREADS_
OUTPUT=_OUTPUT_

srun _COMMAND_ > ${OUTPUT} 2>&1
```

#### Example (`submit_serial.sh`)

```bash
#!/bin/sh
#PBS -N _JOBNAME_
#PBS -q _QUEUE_
#PBS -l nodes=_NODES_
#PBS -l walltime=_MAX_TIME_
#PBS -o _JOB_STDOUT_
#PBS -e _JOB_STDERR_

export OMP_NUM_THREADS=_OMP_NUM_THREADS_
OUTPUT=_OUTPUT_

cd ${PBS_O_WORKDIR}
_COMMAND_ > ${OUTPUT} 2>&1
```


## Twist-aware inputs (non-$\Gamma$ PBC)

Twist selection is a property of the Hamiltonian data, not a constructor
argument of `VMC_Workflow`, `MCMC_Workflow`, or `LRDMC_Workflow`. A
single-twist Hamiltonian carries the twist on the determinant's k-dependent AO
leaf (`aos_data.k_point`); `Structure_data` owns only the geometry and cell.
Passing that file through the ordinary `hamiltonian_file` argument
automatically selects the real-Gamma or complex-Bloch numerical path.

`QE_Workflow` can convert either one QE k-point or every k-point in the save
directory. Set `twist_average=True` to produce one multi-twist HDF5 file with a
`/twists` group:

```python
from jqmc_workflow import QE_Workflow

qe = QE_Workflow(
    qe_save_dir="./qe_si.save",
    kind="pgto+pw",
    twist_average=True,
    hamiltonian_file="hamiltonian_data.h5",
)
```

The k-point mesh and weights come from the QE save directory; this workflow
does not generate a new Monkhorst-Pack mesh. With `twist_average=False` (the
default), the converter follows the single-k path and always uses the first
k-point of the save directory (`wfc1.hdf5`, or `wfcup1.hdf5` / `wfcdw1.hdf5`
for a spin-polarized save). Neither `QE_Workflow` nor `jqmc-tool qe` has an
option to select another k-point; to convert a different single twist, run
the QE calculation with only that k-point.

The core `jqmc` command determines dispatch from the Hamiltonian file itself.
A combined HDF5 file with the combined `/twists` schema always dispatches MCMC, VMC
optimization, or either LRDMC driver over its stored twists, including the
one-twist identity case. An ordinary Hamiltonian file always runs one k point.
There is no TOML override for this property.

Under twist averaging jQMC dumps a complete multi-twist
`hamiltonian_data_opt_step_N.h5` after every `wf_dump_freq` chunk: the input
file is retained as the structure template; the canonical shared
Jastrow from `restart.h5/shared_state.h5` is written once, and the optimized
AO bases for both spins and MO/lambda payload from every
`restart.h5/twist_NNN.h5` replace the corresponding twist payload. This
includes exponents and contraction coefficients optimized with
`opt_lambda_basis_exp` and `opt_lambda_basis_coeff`. The same path serves Jastrow-only and
Jastrow+geminal optimization, and `VMC_Workflow` publishes the average of the
last such dumps (or, with `handoff_mode="last"`, the last dump) through
`optimized_hamiltonian` exactly as it does for a single-k run. A
rank-0-twist-only dump is never written; a run that produced no dump at all
raises instead of publishing a substitute.

For MCMC or either LRDMC driver, `[coulomb].mpc = true` (the `mpc=True`
argument of `MCMC_Workflow` and `LRDMC_Workflow`; `LRDMC_Ext_Workflow` has no
`mpc` or `mpc_cutoff` argument) collects the passive
MPC correction separately at every twist and reports its BZ-weighted
correction and covariance-aware corrected total. The per-twist MPC summaries
and `mpc_cutoff` are stored in the twist checkpoint directory. MPC does not
change sampling, branching, force, or stress estimators.

Twist averaging needs no separate workflow class. Use
`QE_Workflow(twist_average=True)` for a Quantum ESPRESSO
save directory, or `WF_Workflow(twist_average=True)` for a multi-k TREXIO file
(see {doc}`trexio`); the resulting Hamiltonian automatically selects the
twist-averaged calculation in the core `jqmc` CLI.

## Pipeline example

A minimal pipeline script that runs VMC -> MCMC + LRDMC extrapolation:

```python
from jqmc_workflow import (
    Container,
    FileFrom,
    Launcher,
    LRDMC_Ext_Workflow,
    MCMC_Workflow,
    ValueFrom,
    VMC_Workflow,
)

server = "my-cluster"
h5 = "hamiltonian_data.h5"

vmc = Container(
    label="vmc",
    dirname="vmc",
    input_files=[h5],
    workflow=VMC_Workflow(
        server_machine_name=server,
        hamiltonian_file=h5,
        queue_label="default",
        pilot_queue_label="small",
        jobname="vmc",
        target_error=0.001,
    ),
)

mcmc = Container(
    label="mcmc-prod",
    dirname="mcmc_prod",
    input_files=[FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian"))],
    rename_input_files=["hamiltonian_data.h5"],
    workflow=MCMC_Workflow(
        server_machine_name=server,
        hamiltonian_file=h5,
        queue_label="default",
        pilot_queue_label="small",
        jobname="mcmc",
        target_error=0.001,
        cleanup_patterns=["restart.h5"],  # delete large checkpoint on success
    ),
)

lrdmc = Container(
    label="lrdmc-ext",
    dirname="lrdmc_ext",
    input_files=[FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian"))],
    rename_input_files=["hamiltonian_data.h5"],
    workflow=LRDMC_Ext_Workflow(
        server_machine_name=server,
        alat_list=[0.10, 0.20, 0.25, 0.30],
        hamiltonian_file=h5,
        queue_label="default",
        pilot_queue_label="small",
        jobname_prefix="lrdmc",
        target_survived_walkers_ratio=0.97,  # selects GFMC_n with calibration
        E_scf=ValueFrom("mcmc-prod", "energy"),
        target_error=0.001,
        num_gfmc_warmup_steps=25,  # default and minimum
        num_gfmc_bin_blocks=10,    # default and minimum
    ),
)

pipeline = Launcher(workflows=[vmc, mcmc, lrdmc])
pipeline.launch()
```

In this example, `mcmc-prod` and `lrdmc-ext` depend on `vmc` (via
`FileFrom`).  The optimized Hamiltonian filename is resolved
dynamically through `ValueFrom("vmc", "optimized_hamiltonian")`,
so the pipeline works without knowing in advance the step number of
the last optimized dump (e.g. when VMC stops early on convergence).  Additionally, `lrdmc-ext` depends on
`mcmc-prod` (via `ValueFrom` for `E_scf`), so the DAG becomes
VMC -> MCMC -> LRDMC-ext.  Passing `target_survived_walkers_ratio`
selects GFMC_n mode (`lrdmc-bra`) and triggers automatic calibration of
`num_projection_per_measurement` independently at each lattice spacing;
without it the default GFMC_t mode (`lrdmc-tau`, `time_projection_tau=0.10`)
is used and no calibration runs.  All alat values run their calibration,
error-bar pilot, and production phases in parallel.  The
`num_gfmc_warmup_steps` / `num_gfmc_bin_blocks` values shown are the
defaults and may be omitted; smaller values are rejected.

## Job Manager CLI (`jqmc-jobmanager`)

The `jqmc-jobmanager` command-line tool monitors and manages running
pipelines.  It recursively discovers `workflow_state.toml` files under
the current directory and displays a summary tree.

### Commands

| Command | Description |
|---|---|
| `show`  | Print the workflow tree.  Add `--id N` to display full detail for one job. |
| `check` | Print the tree **and** query the scheduler on the remote machine for live job status.  Use `--id N` to target a specific job. |
| `del`   | Cancel a queued/running job and mark the workflow as `cancelled`.  Requires `--id N`. |

### Common options

| Option | Description |
|---|---|
| `--id N` | Numeric job ID shown in the tree (0-based). |
| `-s` / `--server` | Server machine name (defaults to `localhost`). Used by `check` and `del` to connect to the remote scheduler. |
| `--log-level` | `INFO` (default) or `DEBUG`. |

### Usage examples

Show the full workflow tree:

```bash
jqmc-jobmanager show
```

Show tree and detailed info for job 2:

```bash
jqmc-jobmanager show --id 2
```

Check live queue status for job 4 on `my-cluster`:

```bash
jqmc-jobmanager check --id 4 -s my-cluster
```

Cancel job 4 on `my-cluster`:

```bash
jqmc-jobmanager del --id 4 -s my-cluster
```

### Example output

```text
==============================================================================
  Workflow Job Tree
  Root: /home/user/project
==============================================================================
   ID  Status       Label                Type                 Server       Job#
  --------------------------------------------------------------------------
    0  completed    wf                   WF_Workflow          ?            -
       dir: 00_wf
    1  completed    vmc                  VMC_Workflow         my-cluster   12345
       dir: 01_vmc
       energy: -17.17 +- 0.005 Ha
    2  running      mcmc-prod            MCMC_Workflow        my-cluster   12367
       dir: 02_mcmc
    3  running      lrdmc-ext            LRDMC_Ext_Workflow   ?            -
       dir: 03_lrdmc_ext
```

For the full API reference, see
{ref}`API reference for the pipeline (jqmc_workflow) <api_ref_workflow>`.
