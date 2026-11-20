(api_reference_cli_link)=

# API reference for the command-line interface (jqmc)

## Command-line `jqmc` usage

You can run `jqmc` tasks (`mcmc`, `vmc`, `lrdmc-bra`, and `lrdmc-tau`) from the command line:

```bash
# Serial run
jqmc mcmc.toml > output.log

# MPI parallel run
mpirun -np 4 jqmc mcmc.toml > output.log
```

The input file is a TOML document whose keys match the parameters listed below.
Keys that are not listed are not used: the CLI prints a warning for an unknown
section or key (for example a misspelled `num_mcmc_step`) and for a job section
that the selected `job_type` does not read. Read the warnings at the top of the
output; a misspelled key otherwise means its default is used.

### Which `job_type` do I need?

The job names follow the driver names, not the textbook method names:

| Goal | `job_type` | TOML section |
| ---- | ---------- | ------------ |
| Optimize the trial wavefunction (Jastrow, determinant) by VMC | `"vmc"` | `[vmc]` |
| Sample a fixed trial wavefunction: VMC energy, forces, stress | `"mcmc"` | `[mcmc]` |
| Lattice-regularized diffusion Monte Carlo (LRDMC) with a fixed number of projections per measurement | `"lrdmc-bra"` | `[lrdmc-bra]` |
| LRDMC with a fixed imaginary-time step `tau` per measurement | `"lrdmc-tau"` | `[lrdmc-tau]` |

A typical study runs `vmc` (optimization), then `mcmc` with the optimized
`hamiltonian_data_opt_step_N.h5`, then LRDMC at several `alat` values followed
by an `alat -> 0` extrapolation. The two LRDMC drivers converge to the same
fixed-node energy; `lrdmc-bra` keeps the walker load better balanced across
MPI ranks and is the one written by `jqmc-tool lrdmc generate-input`. See
{ref}`reference_lrdmc` and {doc}`notes/lrdmc`.

> **Note**
> Throughout this document, "per MPI process and per walker" means the quantity is counted for each MPI rank and for each walker on that rank. When relevant, the total across all ranks and walkers is indicated explicitly.

---

## Input parameters for the command-line `jqmc`

### `precision`

The optional precision table has one supported key. Which parts of the
calculation run in reduced precision under `"mixed"` is fixed by jQMC and
cannot be changed from the TOML; see {doc}`notes/mixed_precision`.

| Key    | Default  | Description |
| ------ | -------: | ----------- |
| `mode` | `"full"` | `"full"` uses fp64/complex128 arithmetic; `"mixed"` applies the reviewed mixed-precision zone policy. See {doc}`notes/mixed_precision`. |

Example:

```toml
[precision]
mode = "mixed"
```

---

### `control`

| Key                 |                  Default | Description                                                                                                                |
| ------------------- | -----------------------: | -------------------------------------------------------------------------------------------------------------------------- |
| `job_type`          |             **required** | Select the job: `"mcmc"`, `"vmc"`, `"lrdmc-bra"`, or `"lrdmc-tau"`.                                                            |
| `mcmc_seed`         |                  `34456` | Random seed for MCMC/GFMC chain.                                                                                           |
| `number_of_walkers` |                      `4` | Number of walkers **per MPI process**.                                                                                     |
| `max_time`          |                  `86400` | Time limit in seconds, measured from the start of the sampling or optimization loop (reading the input before the loop is not counted). When it is exceeded, the loop stops at the next measurement block or optimization step, and the run finishes normally. A `vmc` optimization also stops before an optimization step that would end after `max_time`, predicting the step duration by the mean duration of the steps already done in this run. In both cases the run finishes normally: it prints the final output from the data collected so far and writes `restart_chk`, so the job can be continued with `restart = true`. Set it below the scheduler wall time with a margin for this final step. |
| `restart`           |                  `false` | If `true`, restart from a checkpoint.                                                                                      |
| `restart_chk`       |           `"restart.h5"` | Path of the restart checkpoint. Every run **writes** this checkpoint when it finishes (overwriting an existing one), and a run with `restart = true` **reads** it first. For twist-averaged runs this is the checkpoint **directory** (`twist_NNN.h5` + `shared_state.h5` layout). A twist-averaged `vmc` job also needs it (even on a fresh run) to be able to dump the complete multi-twist `hamiltonian_data_opt_step_N.h5`. A twist-averaged run stops at startup if this path is an existing regular file (for example a single-twist checkpoint or a leftover `restart.h5`); choose a new directory name such as `restart_chk = "restart_twists"` (with the default, the directory is called `restart.h5/`, which is easy to mistake for a file). |
| `hamiltonian_h5`    |  `"hamiltonian_data.h5"` | Hamiltonian checkpoint file. When `restart=false`, this file is used to initialize/record Hamiltonian data as appropriate. |
| `verbosity`         |                  `"low"` | Verbosity level: `"low"`, `"high"`, `"devel"`, `"mpi-low"`, `"mpi-high"`, `"mpi-devel"`. Without the `mpi-` prefix only MPI rank 0 prints (to stdout). With the `mpi-` prefix every rank writes its own log file `log_jqmc_mpirank_N.log` in the working directory instead (useful for debugging one rank). Any other value is an error. |

For fresh runs, the layout of `hamiltonian_h5` selects dispatch. A combined
file containing `/twists` runs the twist-averaged driver, including when it
contains one twist. An ordinary Hamiltonian file runs the single-k driver.
On restart, an existing checkpoint file selects the single-run path and supplies
its Hamiltonian; the original `hamiltonian_h5` is unused and need not exist.
An existing checkpoint directory selects twist averaging and uses the combined
input for the twist/identity contract. If that directory does not yet exist,
the combined input still selects the twist path and the existing resume rules
start its missing twists. Read errors and malformed checkpoint payloads remain
errors; no alternate format is attempted.

Orbital tiling is chosen during QE/TREXIO conversion, using the
[converter options](api_reference_cli_tool.md). Set `hamiltonian_h5` to that
output; the QMC TOML has no `tiling` or `supertwist` key. The same dispatch and
checkpoint-directory rules apply to tiled twist ensembles. See
[example14](examples.md#jqmc-example14-frozen-orbitals-in-a-tiled-supercell).

---

### `mcmc` (i.e., a single-shot MCMC without WF optimization)

| Key                        |      Default | Description                                                                                                                                                                                                                                   |
| -------------------------- | -----------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_mcmc_steps`           | **required** | Number of **measurement** steps per MPI process and per walker. Local energy and other observables are measured `num_mcmc_steps` times in total per (rank x walker). The global total equals `num_mcmc_steps x mpi_size x number_of_walkers`. |
| `num_mcmc_per_measurement` |         `40` | MCMC updates between successive measurements. Observables are recorded every `num_mcmc_per_measurement` steps.                                                                                                                                |
| `num_mcmc_warmup_steps`    |          `0` | Number of initial **measurement** steps discarded as warm-up (equilibration) before the energy and other averages are formed. The default is a placeholder: use at least 30 for results you report. |
| `num_mcmc_bin_blocks`      |          `5` | Number of binning blocks per MPI process and per walker (total binned blocks = `num_mcmc_bin_blocks x mpi_size x number_of_walkers`). The error bar is estimated from these blocks (jackknife); use at least 10, and choose the value as in {ref}`sec:error_analysis_bins`. The run is refused unless `num_mcmc_steps >= num_mcmc_warmup_steps` and `num_mcmc_steps - num_mcmc_warmup_steps >= num_mcmc_bin_blocks`. |
| `Dt`                       |        `2.0` | MCMC step size (Bohr).                                                                                                                                                                                                                        |
| `epsilon_AS`               |        `0.0` | Attaccalite-Sorella regularization parameter. `0.0` samples $\lvert\Psi\rvert^2$ directly. A positive value samples a guiding distribution that does not vanish at the nodal surface and reweights the averages, which removes the infinite variance of force estimators; choose it so that the average weight is about 0.8 (see {doc}`notes/vmc`). |
| `atomic_force`             |      `false` | If `true`, compute atomic forces (printed in the final output and stored in `restart_chk` for `jqmc-tool mcmc compute-force`).                                                                                                                 |
| `compute_stress`          |      `false` | If `true`, also compute the symmetric stress tensor sigma (Ha/bohr^3) and the diagonal pressure P = -diag(sigma). Periodic systems only; for a molecular system no stress is reported. `atomic_force` and `compute_stress` independently enable position and strain derivatives; stress-only does not request atomic-force buffers. |
| `use_swct`                 |       `true` | If `true`, apply Space Warp Coordinate Transformation (SWCT) to atomic forces. Default is `true` for MCMC.                                                                                                                                    |
| `parameter_derivatives`    |      `false` | If `true`, also accumulate the derivatives of $\ln\Psi$ with respect to the variational parameters during sampling (stored in the checkpoint; the final output only reports that they were computed). Not needed for ordinary energy or force runs. |

---

### `vmc` (stochastic reconfiguration / natural-gradient optimization)

| Key                        |      Default | Description                                                                                                          |
| -------------------------- | -----------: | -------------------------------------------------------------------------------------------------------------------- |
| `num_mcmc_steps`           | **required** | Same definition as in `mcmc`, but applied **per optimization step**: every optimization step runs `num_mcmc_steps` measurement steps per MPI process and per walker. |
| `num_mcmc_per_measurement` |         `40` | Same as `mcmc`.                                                                                                      |
| `num_mcmc_warmup_steps`    |          `0` | Warm-up measurement steps discarded **in every optimization step**.                                                  |
| `num_mcmc_bin_blocks`      |          `5` | Binning blocks per MPI process and per walker, used in every optimization step. The same constraints as in `mcmc` apply: `num_mcmc_steps >= num_mcmc_warmup_steps` and `num_mcmc_steps - num_mcmc_warmup_steps >= num_mcmc_bin_blocks`. |
| `Dt`                       |        `2.0` | MCMC step size (Bohr).                                                                                               |
| `epsilon_AS`               |        `0.0` | Attaccalite-Sorella regularization parameter, as in `mcmc`.                                                          |
| `num_opt_steps`            | **required** | Number of optimization iterations. On a restart (`restart = true`) this is the number of **additional** iterations, for both single-twist and twist-averaged runs; the step counter in the output and in `hamiltonian_data_opt_step_N.h5` continues from the checkpoint. |
| `wf_dump_freq`             |          `1` | Write wavefunction/Hamiltonian checkpoint every this many optimization steps, as `hamiltonian_data_opt_step_N.h5` (`N` = optimization steps completed, including any resume offset). A twist-averaged run dumps a complete multi-twist file holding every twist (shared Jastrow + per-twist MO/lambda), assembled from the per-twist checkpoints written at the same cadence, so any dumped step is a valid `hamiltonian_h5` for a fresh run. |
| `optimizer_kwargs`         | see sub-table below | SR/LM optimizer configuration. `method` must be `"sr"`; every other method raises `ValueError`. `delta`/`epsilon` control step size and regularization, `cg_*` entries tune the conjugate-gradient solver, and `use_lm` selects the LM/aSR hierarchy. See the supported sub-keys below. |
| `opt_J1_param`             |       `true` | Optimize J1 (one-body Jastrow) parameters. No-op if the wavefunction has no J1.                                      |
| `opt_J2_param`             |       `true` | Optimize J2 (two-body Jastrow) parameters. No-op if the wavefunction has no J2.                                      |
| `opt_J3_param`             |       `true` | Optimize J3 (three-body Jastrow) parameters. No-op if the wavefunction has no J3.                                    |
| `opt_geminal`              |      `false` | Canonical geminal opt flag. If `true`, optimize the geminal (`lambda_matrix` for `AGP_data`, `mo_coefficients` + `lambda_diag` for `AGPn_data`, `mo_coefficients` only for `SD_data`).                                                  |
| `fix_geminal_rank`         |          `0` | Rank-control flag for the rank-truncated MO opt path. `0` (default) means "not set" = full-rank ambient opt on the source class. A positive int triggers `AGP -> AGPn` auto-convert at `run_optimize` entry when the current geminal is `AGP_data`; `AGPn` / `SD` inputs have structural rank, so mismatches warn (`AGPn`) or raise (`SD`).             |
| `opt_J3_basis_exp`         |      `false` | Optimize J3 AO Gaussian exponents.                                                                                   |
| `opt_J3_basis_coeff`       |      `false` | Optimize J3 AO contraction coefficients.                                                                             |
| `opt_lambda_basis_exp`     |      `false` | Optimize Geminal AO Gaussian exponents (up and down spins, concatenated).                                            |
| `opt_lambda_basis_coeff`   |      `false` | Optimize Geminal AO contraction coefficients (up and down spins, concatenated).                                      |
| `opt_JDTN_u_param`         |       `true` | Optimize the DTN polynomial e-e channel (`alpha_indep`, "u" block, J2 analogue). No-op unless the wavefunction's DTN preset includes `u` (e.g. `u_chi_f`). |
| `opt_JDTN_chi_param`       |       `true` | Optimize the DTN polynomial e-N channel (`beta_indep`, "chi" block, J1 analogue). No-op unless the wavefunction's DTN preset includes `chi`. |
| `opt_JDTN_f_param`         |       `true` | Optimize the DTN polynomial factorised e-e-N channel (`gamma_indep`, "f" block, J3 analogue). No-op unless the wavefunction's DTN preset includes `f`. |

For tiled Slater orbitals, keep `opt_geminal`, `opt_lambda_basis_exp` and
`opt_lambda_basis_coeff` false; enabling any of them is rejected. Jastrow
parameters remain independently optimizable, including the ordinary supercell
J3 matrix and its separately controlled basis parameters. See the
[independent supercell J3](notes/orbital_tiling.md#independent-supercell-j3).

`optimizer_kwargs` sub-keys and their defaults. A partial table is merged
onto these defaults: keys you do not name keep the values below. For example,
`optimizer_kwargs = { delta = 0.1 }` changes only the step size and keeps
`use_lm = true`, `lm_subspace_dim = 0` (aSR).

| Sub-key           |   Default | Description                                                                                                          |
| ----------------- | --------: | -------------------------------------------------------------------------------------------------------------------- |
| `method`          |    `"sr"` | Stochastic reconfiguration (natural gradient), the only supported optimizer method.                                  |
| `delta`           |    `0.15` | SR prefactor in `c_i <- c_i + delta * S^{-1} f` (acts as the learning rate when `use_lm = true`).                    |
| `epsilon`         |   `0.001` | Regularization strength added to the SR `S` matrix; finite and `>= 0`.                                               |
| `cg_flag`         |    `true` | Use the conjugate-gradient solver for the SR linear system.                                                          |
| `cg_max_iter`     |   `10000` | Maximum CG iterations, an integer `>= 1`. A solve that stops above its tolerance logs one warning and is applied.     |
| `cg_tol`          |   `1e-05` | Relative CG convergence tolerance, finite and `> 0`: CG stops once `norm(r) <= cg_tol * norm(b)`, where `b` is the right-hand side of the solved SR system. Earlier versions used the absolute rule `norm(r) <= cg_tol` with default `1e-08`. |
| `use_lm`          |    `true` | If `true`, enable the accelerated-SR / linear-method (aSR/LM) family selected by `lm_subspace_dim`; if `false`, use plain SR with step `delta`. |
| `lm_subspace_dim` |       `0` | Used only when `use_lm = true`. `0` = accelerated SR (aSR: the SR direction with an optimized step scale), `>0` = reduced linear method on the N most important parameters, `-1` = full linear method on all parameters. Other values raise `ValueError`. |
| `lm_cond`         |   `0.001` | Smallest eigenvalue kept in the correlation matrix of the LM coordinates (condition number <= `1/lm_cond`); must lie in the open interval `(0, 1)`. One pivoted Cholesky pass removes coordinates until the kept set satisfies it. |
| `asr_gamma_max`   |     `0.5` | Maximum accepted absolute aSR gamma root before the plain-SR fallback is used.                                       |
| `asr_fallback_gamma` | `0.1` | Absolute fallback gamma magnitude; the model root supplies its sign.                                                 |
| `lm_reference_weight_min` | `0.9` | Reduced and full LM only: minimum reference weight of an accepted LM eigenvector; a candidate below it is rejected and the aSR step is used instead. |

---

### `lrdmc-bra`

| Key                        |      Default | Description                                                                                                                            |
| -------------------------- | -----------: | -------------------------------------------------------------------------------------------------------------------------------------- |
| `num_mcmc_steps`           | **required** | Number of **measurement** steps per MPI process and per walker during LRDMC.                                                           |
| `num_mcmc_per_measurement` | **required** | Number of GFMC projections between measurements (observables recorded after each block of projections), an integer >= 1. There is no default: a fresh run without it stops before sampling. One projection advances imaginary time by about `alat^2 / (3 N_e)`, so the value sets the imaginary time per measurement and depends on the system and on `alat`. The value is used unchanged; a single-twist restart may omit it and then keeps the checkpoint value. |
| `alat`                     |       `0.30` | Lattice discretization parameter (grid spacing). The lattice spacing is `alat x a_0`, where `a_0` is the Bohr radius.                    |
| `non_local_move`           |    `"tmove"` | Treatment of non-local ECP terms: `"tmove"` (T-move) or `"dltmove"` (determinant-locality + T-move).                                   |
| `num_gfmc_warmup_steps`    |         `25` | Number of warm-up measurement steps to discard. Must be at least 25 (the energy shift used for population control is unreliable before that); smaller values are rejected. |
| `num_gfmc_bin_blocks`      |         `10` | Number of binning blocks for GFMC. **Total binned blocks = `num_gfmc_bin_blocks`** (not multiplied by `mpi_size x number_of_walkers`). Must be at least 10 (a minimum, not a production recommendation; see {ref}`sec:error_analysis_bins`), and `num_mcmc_steps - num_gfmc_warmup_steps >= num_gfmc_bin_blocks`. |
| `num_gfmc_collect_steps`   |          `5` | Length of the weight history: the number of previous measurements whose branching weights are multiplied into each measurement's correcting factor before binning. Must be at least 5. |
| `E_scf`                    |        `0.0` | Initial total-energy guess (Ha) used to set the initial GFMC energy shift. For a **fresh** run set it to a realistic value, for example the VMC energy from `mcmc`; the default `0.0` is far from any real total energy. On a restart the value is taken from the checkpoint, so leave it out of the continuation TOML. |
| `atomic_force`             |      `false` | If `true`, compute atomic forces (also available afterwards with `jqmc-tool lrdmc compute-force`).                                     |
| `compute_stress`          |      `false` | If `true`, compute the PBC stress tensor and pressure (`get_aT` / `get_aP`). Supported on both LRDMC drivers (`lrdmc-bra` / GFMC_n and `lrdmc-tau` / GFMC_t). Periodic systems only; for a molecular system no stress is reported. When `false`, a force-only run skips stress compilation and evaluation. |
| `use_swct`                 |      `false` | If `true`, apply Space Warp Coordinate Transformation (SWCT) to atomic forces. Default is `false` for LRDMC.                           |
| `epsilon_PW`               |        `0.0` | Pathak-Wagner regularization parameter (Bohr). When > 0, the force estimator is regularized near the nodal surface (no regularization by default). |

---

### `lrdmc-tau`

| Key                      |      Default | Description                                                            |
| ------------------------ | -----------: | ---------------------------------------------------------------------- |
| `num_mcmc_steps`         | **required** | Number of **measurement** steps per MPI process and per walker.        |
| `tau`                    |       `0.10` | Imaginary-time length (in Ha^-1) projected between successive measurements. |
| `alat`                   |       `0.30` | Lattice discretization parameter; lattice spacing `alat x a_0`.         |
| `non_local_move`         |    `"tmove"` | Non-local ECP treatment: `"tmove"` or `"dltmove"`.                     |
| `num_gfmc_warmup_steps`  |         `25` | Warm-up measurement steps to discard; must be at least 25.             |
| `num_gfmc_bin_blocks`    |         `10` | Binning blocks for GFMC (total binned blocks = `num_gfmc_bin_blocks`); must be at least 10 (see {ref}`sec:error_analysis_bins`), and `num_mcmc_steps - num_gfmc_warmup_steps >= num_gfmc_bin_blocks`. |
| `num_gfmc_collect_steps` |          `5` | Length of the weight history used for the correcting factors, as in `lrdmc-bra`; must be at least 5. |
| `atomic_force`           |      `false` | If `true`, compute atomic forces.                                      |
| `compute_stress`        |      `false` | If `true`, compute sigma (Ha/bohr^3) and P = -diag(sigma). Supported by both LRDMC drivers. Periodic systems only; for a molecular system no stress is reported. When `false`, a force-only run skips stress compilation and evaluation. |
| `use_swct`               |      `false` | If `true`, apply SWCT to atomic forces. Default is `false` for LRDMC.  |
| `epsilon_PW`             |        `0.0` | Pathak-Wagner regularization parameter (Bohr). When > 0, the force estimator is regularized near the nodal surface (no regularization by default). |

For both LRDMC drivers, the lattice/sign-flip local energy controls projection
weights and branching, while the energy reported by `get_E` and the CLI is the
continuum measurement local energy (the local energy without the sign-flip
clipping, eq. {eq}`lrdmc_novar0_measurement`). Atomic-force and stress kernels
differentiate that same reported measurement energy. The walker distribution
still depends on `alat`, so all three observables require an `alat -> 0`
extrapolation; see {ref}`lrdmc_tags`.

---

### `coulomb`

| Key          | Default | Description |
| ------------ | ------: | ----------- |
| `mpc`        | `false` | If `true`, accumulate the passive MPC finite-size correction $\langle V_{ee}^{MPC} - V_{ee}^{Ewald}\rangle$ on the sampled configurations and report $E_{MPC} = E_{Ewald} + \langle dV\rangle$. Single- and multi-twist MCMC and LRDMC sampling jobs report the result. VMC optimization rejects `mpc = true`; run a sampling job after optimization. Sampling, LRDMC branching, local energy, forces, stress, and parameter gradients remain on the Ewald channel. Periodic systems only. |
| `mpc_cutoff` |  `30.0` | Reciprocal-space kinetic-energy cutoff in Ha for the MPC kernel and density FFT grid. This is a convergence parameter; increase it until the correction is stable. Ignored when `mpc = false`. |

---

## Minimal schema example

```toml
[control]
job_type = "mcmc"
number_of_walkers = 4
verbosity = "low"

[mcmc]
num_mcmc_steps = 1000
num_mcmc_per_measurement = 40
num_mcmc_warmup_steps = 30
num_mcmc_bin_blocks = 10
Dt = 2.0
```

Run it with `jqmc mcmc.toml > out_mcmc`; the input also needs the default
`hamiltonian_h5 = "hamiltonian_data.h5"` in the working directory (written by
`jqmc-tool trexio convert-to` or `jqmc-tool qe`).

> **Tips**
>
> * The default `num_mcmc_warmup_steps = 0` and `num_mcmc_bin_blocks = 5` only make a run start; use at least 30 warm-up steps and 10 bins for results you report, and check that the estimate is stable when you change them.
> * For reproducibility across MPI runs, keep `mcmc_seed` fixed and the MPI topology unchanged.
> * For a fresh LRDMC run, set `E_scf` to a realistic total energy (for example the VMC energy) to reduce initial transients in the population control. Leave it out only when continuing with `restart = true`.

(api_reference_cli_restart)=

## Restart inheritance

`restart = true` reconstructs the sampler/propagator from the checkpoint
`restart_chk`. Some TOML values are therefore **frozen** (restored from the
checkpoint), while the remaining values are **mutable** (read again for the
continuation). Input files written by the workflow package (`jqmc_workflow`)
label every key as `[frozen on restart]` or `[mutable on restart]` in their
comments; the templates written by `jqmc-tool <job> generate-input` do not
carry these labels, so use the table below.

| Section | Frozen keys |
| ------- | ----------- |
| `control` | `mcmc_seed`, `number_of_walkers`, `hamiltonian_h5` |
| `mcmc` | `num_mcmc_per_measurement`, `Dt`, `epsilon_AS`, `atomic_force`, `compute_stress`, `use_swct`, `parameter_derivatives` |
| `vmc` | `num_mcmc_per_measurement`, `Dt`, `epsilon_AS` |
| `lrdmc-bra` | `num_mcmc_per_measurement`, `alat`, `non_local_move`, `num_gfmc_collect_steps`, `E_scf`, `atomic_force`, `compute_stress`, `use_swct`, `epsilon_PW` |
| `lrdmc-tau` | `tau`, `alat`, `non_local_move`, `num_gfmc_collect_steps`, `atomic_force`, `compute_stress`, `use_swct`, `epsilon_PW` |
| `coulomb` | `mpc`, `mpc_cutoff` (restored from the checkpoint together with the sampler) |

All registered keys not listed as frozen for their section are mutable, for
example `num_mcmc_steps`, the warm-up and bin settings, `max_time` and
`num_opt_steps`. What a continuation does with them:

* `mcmc`, `lrdmc-bra`, `lrdmc-tau`: `num_mcmc_steps` new measurement steps are
  **appended** to the stored history; the final output and the analysis tools
  use the whole history.
* `vmc`: `num_opt_steps` **additional** optimization steps are run, on both the
  single-twist and the twist-averaged route. A single-twist VMC restart can
  enable `optimizer_kwargs.use_lm` after an SR update when the checkpoint
  contains no retained samples. If the checkpoint contains samples recorded
  without local-energy parameter derivatives, enabling LM raises
  `ValueError` before sampling.

On the single-twist route, an explicit scalar frozen value that conflicts with
the checkpoint raises `ValueError`; omit that key from the continuation TOML
or start a fresh run. This applies to the MPC flag and cutoff, so a continued
MPC job reports the correction and cutoff from its saved sampling settings.
The embedded Hamiltonian has no scalar comparison. For twist-averaged
continuations, keep the same settings across attempts.
In particular, generated `lrdmc-bra` input
intentionally omits `E_scf` because the population-control value evolves
during a run and the checkpoint owns it on restart.

(api_reference_cli_outputs)=

## What a run prints and writes

At the end of the standard output, every job prints a block headed
`Final output(s):`. For a single-twist `mcmc` run it looks like

```text
Final output(s):
  Total Energy: E = <mean> +- <error> Ha.
  Variance: Var = <mean> +- <error> Ha^2.
```

followed, when requested, by the MPC lines
(`MPC correction: dE_MPC = ...` and `Total Energy (MPC-corrected): E_MPC = ...`),
an `Atomic Forces:` table in Ha/bohr (one row per atom, the error in
parentheses on the last digits) and, for periodic systems with
`compute_stress = true`, the stress tensor and pressure. The error bars are
jackknife estimates over the bins set by the warm-up and bin keys of this
TOML. The LRDMC jobs print the same block. A twist-averaged run instead
prints the BZ-averaged energy `E_TA` with its Monte Carlo error and the
spread between twists `s_twist`, followed by a per-twist energy table.

A `vmc` run prints, for every optimization step, a header
`Optimization step = i/N`, the energy `E = ... +- ... Ha` before the update,
and `Max generalized force = ... +- ...` together with its signal-to-noise
ratio. A signal-to-noise ratio that stays small (of order a few or less)
means the remaining parameter gradients are no longer resolved by the
sampling. `jqmc-tool vmc analyze-output` reads these lines and tabulates or
plots them.

Files written in the working directory:

| File | Written by | Content |
| ---- | ---------- | ------- |
| `restart_chk` (default `restart.h5`; a directory for twist-averaged runs) | every job, at the end | Sampler state and the full measurement history. Input of `restart = true` and of the `jqmc-tool mcmc/lrdmc` analysis commands. |
| `hamiltonian_data_opt_step_N.h5` | `vmc`, every `wf_dump_freq` steps, at the last step, and when the run stops early | The optimized wavefunction after `N` steps. Use it as `hamiltonian_h5` of the following `mcmc` or LRDMC run. It is not an optimizer-state checkpoint. |
| `._restart_rank<N>.h5` | every job, at the end | Per-rank temporary files merged into `restart_chk` and deleted afterwards. Files left behind indicate an interrupted run. |
| `log_jqmc_mpirank_<N>.log` | only with `verbosity = "mpi-*"` | Per-rank log. |

A run that reaches `max_time` stops early but still prints the final output
and writes `restart_chk`; continue it with `restart = true`.
