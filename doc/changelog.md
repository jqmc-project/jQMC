(changelog)=

# Change Log

## v0.3.0a1 (alpha)

The first alpha of the 0.3 series adds periodic-system calculations,
primitive-cell orbital tiling and a native Rust-CPU backend. The changes below
are relative to v0.2.2.

### Highlights

* **Periodic systems and complex wavefunctions.** Read Quantum ESPRESSO
  output directly, or import output from other packages (such as PySCF and
  CP2K) via TREXIO. Real Gamma-point and complex Bloch wavefunctions support
  plane-wave, B-spline, periodic Gaussian and composite orbital
  representations.
* **Primitive-cell orbital tiling.** Build a supercell trial wavefunction from
  primitive-cell Quantum ESPRESSO or multi-k TREXIO orbitals. Occupied orbitals
  remain in primitive k blocks instead of being expanded into a global dense
  supercell coefficient matrix. Plane-wave, B-spline, periodic Gaussian and
  PW+pGTO composite representations support Gamma and general supertwists.
* **Twist averaging.** Run VMC, MCMC and LRDMC across multiple twists,
  distribute work over MPI ranks, and combine results with their
  Brillouin-zone weights. Optimized multi-twist Hamiltonians can be passed
  from VMC to subsequent MCMC/LRDMC stages.
* **Periodic interactions, forces and stress.** Ewald and model periodic
  Coulomb (MPC) interactions and the Drummond--Towler--Needs (DTN) polynomial
  Jastrow extend the molecular functionality. Atomic forces, stress and
  pressure are available on supported periodic routes, including both
  LRDMC drivers.
* **Rust-CPU execution and analytic derivatives.** Supported CPU MCMC and
  LRDMC runs use the Rust backend by default in full and mixed precision.
  GPU execution remains on JAX. Wavefunction optimization, force and stress
  calculations use analytic derivatives by default.
* **Prebuilt packages for more platforms.** Release wheels include the
  Rust extension for Linux x86_64, Linux aarch64 and macOS arm64, for
  Python 3.11, 3.12 and 3.13.

### Input, workflow and analysis improvements

* **Unversioned scientific HDF5.** Hamiltonian inputs and restart checkpoints
  carry no schema or format version; restart checkpoints no longer store the
  `format_version` attribute of v0.2.2. Single-twist and combined-twist layouts remain separate. Readers require
  their prescribed payloads, with no legacy-format fallback; regenerate inputs
  that do not satisfy the current content and normalization contracts. Software
  version and Git commit provenance remain available.

* **Periodic input conversion.** `jqmc-tool qe` / `QE_Workflow` convert
  Quantum ESPRESSO output directly without using TREXIO. This route supports
  mixed all-electron/ECP structures when analytic UPF records are available.
  `--twist-average` or `twist_average=True` creates combined multi-twist
  inputs. The same option is available in `jqmc-tool trexio convert-to` /
  `WF_Workflow` for multi-k TREXIO files. Both routes accept the `ao*` J3
  bases: every twist shares one J3 basis without the Bloch phase and one J3
  matrix, which twist-averaged optimization updates for all twists together.
  See [AO conventions](notes/ao.md) for orbital representations and
  conversion options.
* **Tiled supercell conversion.** `jqmc-tool qe` and
  `jqmc-tool trexio convert-to` accept a quoted `--tiling` matrix containing
  three diagonal or nine row-major integers. Use `--supertwist` to select one
  complete primitive-k sector or `--twist-average` to convert all complete
  sectors; workflow APIs accept `tiling` as an integer list. Each requested
  supercell twist, including Gamma, requires its corresponding complete
  primitive k-point coset. The converted Hamiltonian, Jastrow and optional J3
  basis use the physical supercell. See [Orbital tiling](notes/orbital_tiling.md)
  for supported inputs and examples.
* **Twist-aware workflows and analysis.** Combined multi-twist Hamiltonians
  and checkpoint directories flow through VMC, MCMC and both LRDMC drivers,
  their workflow hand-offs and the corresponding energy, force and
  extrapolation tools. Results are combined with the stored Brillouin-zone
  weights, including compatible MPC corrections.
* **Public Python API and examples.** Package-level exports include orbital
  containers/evaluators and Hamiltonian interaction kernels;
  `jqmc-example11` demonstrates their use. `jqmc-workflow-example04` adds
  a silicon equation of state and LRDMC lattice-spacing extrapolation.
* **CLI and diagnostics.** `jqmc --help`, invocation without an input, and
  `jqmc --version` provide usage/version output. Missing input files give a
  concise error. Plotting works with headless backends, and Hamiltonian basis
  listings preserve shell and atom identity. MO diagnostics check
  normalization and orthogonality using the AO overlap where supported.
* **Local workflow execution.** Project paths containing spaces work, and
  nonzero command exit codes are reported without repeated retries. VMC step
  estimates respect the bin-block requirements enforced by the CLI.
* **Twist-averaged optimization diagnostics.** `Max generalized force`,
  `Max generalized force signal-to-noise` and the `verbosity = "high"` list
  of top coordinates cover every optimized variable: the shared Jastrow
  coordinates and each twist's own orbital coordinates. Top rows of a twist's
  own coordinates carry its twist index, e.g. `mo_coefficients_up[t3:q980]`
  for `twist_003.h5`. Each step also logs one line per twist with that twist's
  largest signal-to-noise. The counts of variational parameters and real
  optimization coordinates are given over all twists. With
  `verbosity = "high"`, the per-block lines give each twist's own orbital
  blocks, e.g. `mo_coefficients_up[t3]`, and the coordinate summary adds a
  line of totals over all twists.
* **Log parser.** `jqmc.output_parser` reads `jqmc` logs.
  `parse_system_info` reads the system information, including each spin
  channel's tiled-orbital summary (`tiled_orbitals`). `parse_vmc_steps` reads
  each optimization step, including the per-block `[SR update]`,
  `[aSR update]` and `[LM update]` lines (with the `theta=ALL ZERO (no update)`
  form) into `steps[].parameter_updates` with a `method` field; every aSR
  fallback line into `steps[].asr["fallbacks"]`, one record per line with a
  `reason` and the printed numbers, and the negative-discriminant clipping
  into `asr["clipped_discriminant"]`; each MO block's `S-Stiefel tangent`
  line into `blocks[].s_stiefel_tangent`; and the linear-method lines
  (`LM conditioning input`, `LM conditioning`, `LM selected state`,
  `LM subspace`, `LM decision` and the rejected candidate's reference weight)
  into `steps[].lm`. A `Derivative source` line that prints `D shape=None`,
  as plain SR does, gives `D_shape` set to `None`. Numeric lines it does not
  recognize are listed in `unparsed_numeric_lines`.
* **Opposite spin twists for every input.** At a complex Bloch twist the
  down-spin channel is the complex conjugate of its own `+k` orbitals, stored
  as conjugated coefficients on the `-k` basis: the conjugated up
  coefficients for restricted inputs and the conjugated beta coefficients for
  spin-polarized (LSDA / UHF) inputs from QE and TREXIO, tiled or not. This
  requires a real Hamiltonian per spin channel (no vector potential, no
  spin-orbit coupling). `time_reversal` states whether the down coefficients
  are tied to the up coefficients. Every geminal is checked on construction
  and on loading: a spin-polarized complex-twist `hamiltonian_data.h5`,
  checkpoint or twist-average file with both spins at one twist is refused
  with an error.

### Upgrading from v0.2.2

* **Python 3.11 or later is required.** Python 3.10 is no longer supported.
  Prebuilt wheels include `jqmc_rs`; source and editable installs require a
  Rust toolchain and compile the extension during installation. Local builds
  use host-native CPU tuning and should be rebuilt for a different CPU.
  MPI remains a runtime dependency; GPU users install the appropriate JAX
  CUDA extras separately. Use `JQMC_FLAG_RUST_ON_CPU=0` to select XLA on CPU.
* **Neural-network Jastrow support is removed.** `NNJastrow`,
  `Jastrow_NN_data`, their CLI/workflow options and `opt_JNN_param` are no
  longer supported. Remove obsolete keys and replace inputs that require
  this ansatz.
* **Optax optimizers are removed.** Use
  `optimizer_kwargs["method"] = "sr"` with the SR/aSR/LM options. The optax
  dependency is removed, and old optax checkpoint state is ignored.
* **Workflow submodule imports have changed.** Direct imports of
  `jqmc_workflow._state`, `_machine` and `_output_parser` move to `state`,
  `machine` and `output_parser`; force/stress parsing helpers are in
  `forces_parser`. `_setting`, `_header_footer` and `_lrdmc_calibration` were
  removed. Package-level workflow re-exports are unchanged.

### Convention changes

* **LRDMC measurement energy.** Both LRDMC drivers now use the continuum
  local energy for reported energies, checkpoints and force estimators.
  Projection and branching continue to use the sign-flip-clipped lattice
  local energy. Reported energies and forces can therefore differ from
  v0.2.2 at finite `alat`; an `alat -> 0` extrapolation is still required.
  See [LRDMC notes](notes/lrdmc.md) for the estimator definitions.
* **`lrdmc-bra` projection count is explicit.**
  `num_mcmc_per_measurement` has no CLI or Python default. Set it explicitly,
  or let `LRDMC_Workflow` determine it by setting
  `target_survived_walkers_ratio`. Checkpoint-restored values are retained.
  `lrdmc-tau` is unchanged.
* **Linear Method fallback policy.** Rejected LM steps now attempt centered
  aSR before falling back to a fixed SR step. Optimization paths can
  therefore differ from v0.2.2 even with the same starting wavefunction.
* **Optimized-wavefunction hand-off.** `VMC_Workflow` now passes the average
  of the last 20 optimization steps to downstream stages by default. Set
  `handoff_mode="last"` to retain the previous single-dump behavior; use
  `handoff_average_steps` to change the averaging window.
* **`cg_tol` is a relative tolerance.** Every SR conjugate-gradient solve
  (CPU NumPy, CPU Rust and GPU JAX; wide and tall) now stops once
  `norm(r) <= cg_tol * norm(b)`, where `b` is the right-hand side of the
  solved system, instead of `norm(r) <= cg_tol`. The default changes from
  `1e-8` to `1e-5`. The previous default was about 9 orders of magnitude
  below the statistical noise of `b`. Inputs that set `cg_tol` explicitly
  change meaning: an old absolute value `t` corresponds to `t / norm(b)`, and
  `norm(b)` varies between systems and steps.
  With `verbosity = "high"` the `SR solve:` line reports the absolute
  threshold `cg_tol * norm(b)` as `tolerance`.
* **`cg_max_iter` default in the Python API.** When `optimizer_kwargs` does
  not set `cg_max_iter`, `run_optimize` now caps every SR conjugate-gradient
  solve at `10000` iterations, the value the command-line input already used
  and documented. It was `1000000`. Set `cg_max_iter` explicitly to keep the
  previous cap.
* **`use_device_collectives` removed.** `MCMC.run_optimize` and
  `OptimizationConfig` no longer accept `use_device_collectives`; passing it
  raises `TypeError`. `use_device_optimizer` (default `True`) still selects
  the device optimizer, and its reduction route now follows from the MPI
  communicators: a sampler that spans every rank reduces with the device
  collectives, a twist sampler of a twist-averaged run with the host MPI
  conjugate-gradient solve. The `jqmc` command-line driver chose this value
  itself from the JAX backend; it was never an input option.

### Bug fixes

* **SciPy 1.16.x and 1.17.x are excluded.** SciPy 1.16.2-1.17.x bundle
  OpenBLAS 0.3.29.dev/0.3.30, whose multi-threaded LU factorization can hang
  forever. The device SR direct solve hung this way on many-core hosts. SciPy 1.16.0-1.16.1 fail the SLSQP solve of the pGTO exponent
  optimizer. jQMC now requires `scipy >= 1.12.0, != 1.16.*, != 1.17.*`.
  Python 3.11 resolves SciPy 1.15.3 and Python 3.12 or later resolves 1.18.
  The supported Python and JAX ranges are unchanged.
* **J1 Pade wavefunctions.** The one-body Pade Jastrow now uses the correct
  cusp normalization. Inputs selecting this form produce a different
  wavefunction, energy and derivatives than in v0.2.2. Reoptimize and rerun
  affected calculations; do not combine their old and new sampling
  statistics. The exponential J1 form is unaffected by this correction.
* **LRDMC lattice kinetic coefficients.** Corrected a factor-of-two error
  in the reconstructed diagonal and background terms of the lattice kinetic
  operator. Recheck finite-`alat` results with appreciable sign-flip clipping;
  historical continuum extrapolations are not automatically invalidated.
* **Short-distance Jastrow evaluation.** Exponential J1/J2 values and
  derivatives are evaluated more stably near coalescence, reducing
  cancellation in mixed precision.
* **SR/LM solver checks.** An SR conjugate-gradient solve that stops above
  its tolerance (at `cg_max_iter` or on a breakdown) is now reported with
  one warning per solve at the default verbosity; the step is still applied.
  Invalid `cg_tol`, `cg_max_iter`, `epsilon` or `lm_subspace_dim` values
  stop the run before sampling, and non-finite SR samples stop the GPU
  optimizer as they already stopped the CPU one. The `jqmc-example04` JAGP
  inputs now allow 2000 CG iterations instead of truncating every solve at
  200.
* **Force post-processing memory.** `jqmc-tool mcmc compute-force` and
  `jqmc-tool lrdmc compute-force` no longer load the per-atom force
  observables (`force_HF`, `force_PP`, `E_L_force_PP`) whole, and no longer
  make a full-size weighted copy of each. They now read one bin block of one
  observable at a time, as the new twist-directory force estimate does. Peak
  memory drops from several times the checkpoint's force data to about two
  bin blocks of one observable plus the 2-D energies and weights, so long
  runs no longer run out of memory. The printed forces and errors are unchanged, bit for bit.
* **Hamiltonian summary printed once for every job type.** In v0.2.2 an
  `lrdmc-bra` or `lrdmc-tau` run started without a restart checkpoint printed
  the Hamiltonian summary twice. The `jqmc` log now prints it exactly once per
  run for `mcmc`, `vmc`, `lrdmc-bra` and `lrdmc-tau`.
* **`jqmc-tool lrdmc` analysis defaults.** `compute-energy`, `compute-force`
  and `extrapolate-energy` default to `-w 25`, `-b 10` and `-c 5`, the minima
  that the commands accept, so they run without these flags. In v0.2.2 they
  defaulted to `-w 0` and `-b 5`, below these minima: `compute-energy`
  refused its own defaults, `compute-force` printed a warning, and
  `extrapolate-energy` did not check them.

### Known limitations of this alpha

* Periodic cells must be charge neutral; charged-cell corrections are not
  implemented.
* Tiled determinant orbitals are frozen: orbital/geminal optimization and
  conversion to a dense supercell orbital representation are not supported.
  A pGTO+B-spline composite cannot be tiled.
* QE and periodic TREXIO conversion require a fixed integer electron count at
  every twist; fractional band occupations from smearing are not supported.
  Metallic systems can be converted from orbitals generated with fixed
  occupations.
* Twist-average and tiled conversion accept `ao*` J3 bases but reject
  `-j3 mo`, whose orbitals depend on the twist. On the TREXIO route, both
  `--twist-average` and `--tiling` preserve the source angular form and
  cannot be combined with
  `--ao-conv-to`.
* All-electron J1 parameter optimization can remain frozen or overshoot,
  depending on the initial value.
* APIs and input fields may change in subsequent 0.3 alphas. Preserve the
  exact version, conversion inputs and Hamiltonians used for scientific
  results.

### Technical changes

* **Calculation kernels.** Analytic parameter, force and stress derivatives
  extend the analytic orbital derivatives available in v0.2.2. LRDMC reuses
  determinant derivative tables and projection state on supported routes;
  carried inverses are checked and rebuilt when needed to control numerical
  drift. Molecular AO overlap evaluation is vectorized.
* **Orbital tiling storage and execution.** Tiled occupied orbitals retain
  their primitive k blocks in Hamiltonian and checkpoint HDF5 files. The XLA
  and native Rust-CPU paths evaluate these blocks directly for energies,
  forces and stress, and MPC density is formed from occupied values without a
  dense supercell expansion. J3 remains an independent ordinary supercell
  basis and can couple functions across replicated cells.
* **Analytic extrapolation errors.** The 1,000-draw Gaussian resampling in
  v0.2.2 is replaced by analytic error propagation for the same unweighted
  polynomial fit and independent input errors. This removes resampling
  noise from the extrapolated energy and error bar while retaining the
  input statistical uncertainties. Both `E` and `E_MPC` use this method.
* **Validation infrastructure.** Guarantee-driven development (GDD) maps
  scientific requirements to tests and required precision/execution cases.
  The suite separates light/heavy tests and isolates tests that write files.
* **Packaging and release validation.** Wheels are built from the source
  distribution and tested as installed packages.
* **Documentation.** The manual adds periodic/complex conventions, orbital
  tiling overview and examples, public kernel APIs, solid-state workflow examples
  and GDD references.

## May-29-2026: v0.2.2

First stable release since v0.1.0. v0.2.2 ships everything accumulated across four alphas (v0.2.0a1, v0.2.1a1, v0.2.1a2, v0.2.2a1) plus a final round of polish. Per-alpha sections are preserved below; this entry is a roll-up of the highlights from v0.1.0 to v0.2.2.

### Highlights (v0.1.0 -> v0.2.2)

#### Optimization

* **Linear Method (LM) optimizer** integrated under `method="sr"` with a unified `use_lm` / `lm_subspace_dim` hierarchy (plain SR / aSR / LM). New `|v_0|^2 < 0.9` fallback to plain SR keeps non-linear-regime updates from producing NaN energies.
* **Adaptive learning rate** for Stochastic Reconfiguration.
* **MO optimization** for JSD via the projection method with Attaccalite-Sorella regularization, plus geminal AO -> MO projection.
* **AO basis optimization** (`opt_J3_basis_coeff/exp`, `opt_lambda_basis_coeff/exp`) with shell-shared constraint and dual symmetrization.
* **Distributed tall-CG SR** solver via `psum`, removing `mpi_size`-scaling memory in the SR solve.

#### Performance

* **Fast-update use** across MCMC / VMC / LRDMC, with mat-vec hot paths converted to GEMM for better GPU utilization.
* **On-GPU VMC optimization** with `use_device_collectives` auto-selected by JAX backend; multi-GPU `run_optimize` supported.
* **LU -> SVD** in determinant / geminal / GFMC_n / GFMC_t for ill-conditioned stability; Cartesian / Spherical AO conversion (Cartesian GTOs are substantially faster on GPU); ECP fast path (`compute_ecp_coulomb_potential_fast`).

#### Numerical precision

* **Mixed-precision support** with `"full"` / `"mixed"` modes and per-zone dtype control. Three explicit design principles. AGP/SD geminal stays fp64 to prevent `log|det|` amplification; electron-nucleus `r - R` differences are reconstructed in fp64 before downcast to avoid catastrophic cancellation. `ao_grad_lap` and `mo_grad_lap` zones are split for finer-grained control.

#### Features

* **LRDMC atomic forces** with the Pathak-Wagner regularization.
* **Runtime-selectable Jastrow forms**: `jastrow_1b_type` and `jastrow_2b_type` (`exp` / `pade`).
* **`use_swct` flag** to toggle Space Warp Coordinate Transformation in MCMC and GFMC_n / GFMC_t.

#### `jqmc_workflow` automation package

* **jqmc-workflow** is introduced as a multi-stage QMC pipeline orchestrator (WF conversion -> VMC opt -> MCMC / LRDMC production) with automatic step estimation, checkpointing, and remote job management.

#### bug fixes

* GFMC_n / GFMC_t spin-polarized (`n_up != n_dn`, `n_dn >= 1`) MPI bug.
* MPI deadlock in `max_time` / `stop_flag` checks; `Allreduce` vs `allreduce` for scalars.
* Optimizer step estimation; force NaN; MCMC memory overflow from `r_up_history` / `r_dn_history` storage.

#### Infrastructure

* **Restart files** migrated from pickle `.chk` to HDF5 `.h5` (no backward compatibility).
* **Ruff lint pipeline** and pre-commit updates; non-ASCII cleanup across code and docstrings.
* **Examples**: 11 end-to-end tutorials (`jqmc-example01` to `jqmc-example08`, `jqmc-workflow-example01` to `jqmc-workflow-example03`).
* **Project ownership** transferred to the `jqmc-project` GitHub organization; URLs updated.

### Breaking changes since v0.1.0

* Restart files: pickle `.chk` is no longer supported; HDF5 `.h5` is the only format.
* Optimizer API: `num_param_opt`, `opt_filter_min_SN_ratio`, `adaptive_learning_rate`, and `method="lm"` are all removed or replaced; the Linear Method is now accessed via `method="sr"` with `use_lm=true` (and the new `lm_subspace_dim` / `lm_cond` parameters).

See the per-alpha sections below for full details.

## May-18-2026: v0.2.2a1

This release brings configurable mixed-precision support, deep kernel-level performance work (AOs, Jastrow, det/Jastrow ratios, GFMC), on-GPU VMC optimization, and a project-wide lint/cleanup.

### New features

* **Mixed precision support**: Added a configurable precision system with per-zone dtype control (fp64 by default; fp32 in selected zones in `"mixed"` mode). Refactored as selectable-precision modules with three explicit design principles. The geminal/AGP/SD path remains in fp64 to prevent fp32 amplification of `log|det|`, and electron-nucleus `r - R` differences are computed in fp64 before downcasting to avoid catastrophic cancellation. `ao_grad_lap` and `mo_grad_lap` precision zones are split for finer-grained control. The public API is reduced to a single mode selector: `"full"` or `"mixed"`.
* **On-GPU VMC optimization**: VMC parameter optimization can now run entirely on GPU. Added `use_device_collectives` (auto-selected by the JAX backend: GPU=`True`, otherwise `False`) with an MPI/JAX consistency check; the `jqmc` command-line driver selects it from the JAX backend. Multi-GPU `run_optimize` is supported.
* **Ruff lint pipeline**: Updated `.pre-commit-config.yaml`. Applied auto-fixes and manual cleanups across the codebase, and removed non-ASCII characters from code and docstrings.

### Performance & memory

* **AO module (HLO-level)**: Reduced L1/L2/DRAM traffics. Replaced `segment_sum` with a bucketed reduce+gather scheme (including `V_l`). Unrolled `(8Z)**l` in the Cartesian kernels to avoid XLA `while` loops. Removed `eps` from Cartesian GTOs. Fused AO/MO value/grad/lap into a single dispatch on hot paths.
* **Streaming caches on hot paths**: Stream cached AO and paired tables into every det-ratio / jas-ratio hot paths. Improved the J3 streaming cache in `jastrow_factor.py`. Reused J3 streaming-state AOs in LRDMC mesh / ECP non-local ratios. Improved the K state carry in `jqmc/wavefunction.py`.
* **Jastrow ratios**: Optimized J2 ratio from an `O(N^2)` baseline to `O(N * N_grid)` per-grid sums. Introduced a slim J3 state carry in the MCMC wavefunction update. Polished Jastrow with a dense `(N, N)` up-up / dn-dn pair reduction and removed scatter-add `while` loops.
* **GFMC_t**: Added a streaming kinetic-state path (parity with GFMC_n) and replaced a Python `while` loop with `lax.while_loop` in the projection.
* **Misc**: Vectorized the electron-configuration generator and PRNG key initialization. Switched the jackknife standard deviation to a two-pass centered sum-of-squares for better numerical stability. Replaced `hessian()` with `jvp(grad)` for the NN-Jastrow Laplacian.

### Bug fixes

* **GFMC/MCMC logging**: Improved loggers in `jqmc/jqmc_gfmc.py` and `jqmc/jqmc_mcmc.py`.

### Workflow (`jqmc_workflow`)

* Refactored workflow modules (`vmc_workflow.py`, `mcmc_workflow.py`, `lrdmc_workflow.py`, `lrdmc_ext_workflow.py`, `workflow.py`, and the `_*.py` helpers) for readability, with no behavior change.

### Tests & infrastructure

* Polished tolerance control across the test suite; introduced a medium tolerance for numerical-Laplacian tests and removed the separate autodiff tolerance.
* Removed `test_kinetic_energy_analytic_and_numerical` and `test_numerial_and_auto_grads_and_laplacians_ln_Det` because the numerical references are intrinsically unstable; analytical versions are already validated against JAX autograd.
* Made the numerical-Laplacian debug functions more stable.

## Apr-24-2026: v0.2.1a2

Minor update focusing on workflow improvements, bug fixes, and new benchmark infrastructure.

### New features

* **Kernel benchmark suite**: Added benchmark modules and tests for profiling kernel performance.
* **`cleanup_patterns` option**: Added a `cleanup_patterns` configuration option to `jqmc_workflow` for automatic post-run file cleanup, with support for recursive matching in subdirectories.

### Bug fixes

* **MPI deadlock in `max_time` / `stop_flag`**: Fixed a deadlock that could occur during `max_time` and `stop_flag` checks in MPI runs.

## Apr-16-2026: v0.2.1a1

This release focuses on a major update of the VMC optimizer (Linear Method), extended AO basis optimization, memory/performance improvements of the `jqmc` kernel package, and substantial hardening of the `jqmc_workflow` automation package.

### New features

*   **Linear Method (LM) optimizer**: Implemented the Linear Method optimizer that solves the generalized eigenvalue problem $\bar{H} v = E \bar{S} v$ for optimal parameter updates, providing a powerful alternative to the naive Stochastic Reconfiguration (SR). The optimization is robust and fast.
    *   LM is now integrated into the `method="sr"` code path, controlled by the `use_lm` flag and `lm_subspace_dim` parameter (inspired by TurboRVB's `ncg=1` design).
    *   **Unified optimizer hierarchy** under `method="sr"`:
        *   `use_lm=false`: plain SR
        *   `use_lm=true, lm_subspace_dim=0`: adaptive SR (aSR) with gamma scaling
        *   `use_lm=true, lm_subspace_dim>0`: LM with SR collective variable + top-$p$ S/N ratio parameters
        *   `use_lm=true, lm_subspace_dim=-1`: LM with SR collective variable + all parameters
    *   SR collective variable ($g = S^{-1}f$) is used as the first LM basis vector for stability.
    *   `dgelscut`-based preconditioning with iterative eigenvalue conditioning on the correlation matrix (condition number $\leq 1/\epsilon$), inspired by TurboRVB's implementation.
    *   S-orthonormalization ($P = U \Lambda^{-1/2}$) converts the generalized eigenvalue problem to standard form.
    *   Symmetrization of $H = K + B$ before eigenvalue solve to suppress finite-sample noise.
    *   Eigenvector selection by $\max |v_0|^2$ criterion.
    *   Separate `epsilon` (SR regularization) and `lm_cond` (LM dgelscut threshold) parameters.
    *   Fallback mechanisms: aSR selects the smaller-$|\gamma|$, reference-connected stationary root and falls back to plain SR ($\gamma=0.1$) when that root is not a positive descent direction; LM falls back to $0.1 \cdot g_\mathrm{sr}$ when its candidate fails the acceptance gates.
*   **Extended AO basis optimization**: Implemented `opt_J3_basis_coeff`, `opt_J3_basis_exp`, `opt_lambda_basis_coeff`, and `opt_lambda_basis_exp` options for optimizing three-body Jastrow and geminal AO basis exponents and coefficients in VMC.
    *   **Shell-shared constraint**: Same-atom, same-shell primitives share exponents/coefficients via `symmetrize_metric` (size-preserving shell averaging), consistent with `j3_matrix`/`lambda_matrix` symmetrization.
    *   **Dual symmetrization strategy**: $O_k$ derivatives are symmetrized at source in `get_dln_WF` for accurate $f$ and $S$, and post-hoc symmetrization is applied after `apply_block_update` to prevent floating-point drift over hundreds of optimization steps.
    *   Improved AO basis exponent selection using a log-spaced median window with widened margin (`/2.5`).
*   **`use_swct` parameter**: Added `use_swct` flag to MCMC, GFMC_t, and GFMC_n classes to control Space Warp Coordinate Transformation (SWCT) on/off for atomic force calculations. Default is `True` for MCMC and `False` for GFMC (LRDMC). When disabled, zero arrays are used for `omega`/`grad_omega`, and force formulas reduce to bare Hellmann-Feynman and Pulay forces.
*   **S/N ratio filter**: Applied S/N ratio filtering before SR matrix construction to reduce the effective matrix dimension, improving both speed and numerical stability. `O_matrix_local` is sliced to selected parameters before building $S$, so all SR computations operate in the reduced space.
*   **Shape assertions**: Added rigorous shape assertions (using `mcmc_counter`, `num_walkers`, `n_atoms`) to `get_E`, `get_aF`, `get_gF`, `get_aH` in MCMC, GFMC_t, GFMC_n, and their debug counterparts.

### Performance & memory

*   **Buffer-based MPI reduce in SR**: Eliminated `list()` round-trips and switched to buffer-based MPI reduce in the SR optimizer for lower overhead.
*   **Pre-compute collective observable**: Compute $O_\mathrm{SR} = \delta O \cdot g$ while the full $O$-matrix is still in memory during the SR solve, avoiding a redundant `get_dln_WF` call in the LM path.
*   **Avoid redundant JIT compilation**: Refactored `run_optimize` in `jqmc_mcmc.py` to skip redundant computation via early `continue`, reducing unnecessary recompilation.
*   **`jax.clear_caches()` after optimization loop**: Added cache clearing after the optimization loop as an OOM workaround.
*   **Store `np.array` instead of `list(np.array)`**: Refactored internal data storage to use `np.array` directly, reducing memory fragmentation.
*   **Wrapped properties with `np.asarray()`**: Prevent accidental storage of JAX arrays in checkpoint data to avoid OOM.
*   **Avoid redundant energy/force post-processing**: Skip unnecessary re-computation of energy and force post-processing.
*   **Better memory management**: Improved memory handling in `jqmc_gfmc.py` and `jqmc_mcmc.py`.


### Bug fixes

*   **GFMC_n / GFMC_t spin-polarized MPI bug**: Fixed a critical bug for systems where `n_up != n_dn` and `n_dn >= 1` with MPI >= 2 processes.
*   **GFMC_t projection averaging**: Fixed incorrect averaging of the number of projections across MPI ranks in GFMC_t.
*   **SR with `num_params >= num_samples`**: Fixed MPI bug when the number of optimizable parameters exceeds the number of samples.
*   **MPI `Allreduce` for scalars**: Replaced `Allreduce` with `allreduce` for scalar `int` and `float` values in `jqmc_mcmc.py` and `jqmc_gfmc.py`, as `Allreduce` for scalars exhibits implementation-dependent behavior.
*   **Optimizer step estimation**: Fixed `estimate_required_steps` -- removed incorrect `ceil` rounding and `max` clamp that ignored `walker_ratio`; added `min_steps` parameter.
*   **SR stability near convergence**: Improved stability of SR with adaptive learning rate in the vicinity of convergence.
*   **Pytree inconsistency**: Fixed a JAX pytree structural mismatch.
*   **S/N ratio diagnostics**: Fixed averaging (last S/N ratio -> averaged S/N ratios) and trivial output bugs.


### Workflow (`jqmc_workflow`)

*   **Major refactoring**: Comprehensive overhaul of all workflow modules (`vmc_workflow.py`, `mcmc_workflow.py`, `lrdmc_workflow.py`, `workflow.py`) with improved robustness, cleaner code structure, and new `_phase.py` module for phase management.
*   **SSH / file-descriptor leak fixes**: Fixed SSH connection hangs and leaks; consolidated `Machine` objects to prevent resource exhaustion.
*   **Continuation behavior**: Changed and improved the behavior of workflow continuations with `SHA256`-based input fingerprinting for reliable restart detection.
*   **Step count accumulation**: Fixed a bug in accumulated step counts for VMC and LRDMC workflows.
*   **VMC convergence check**: Implemented a new VMC energy-slope-based convergence check.
*   **New VMC workflow parameters**: Introduced additional configurable parameters for `vmc_workflow.py`.
*   **Output parser fixes**: Fixed parsers for workflow output processing.
*   **`FileFrom` handling**: Fixed and polished `FileFrom` file-transfer logic.
*   **Job ID checks**: Updated and improved job ID check logic for remote execution.
*   **Error estimation in workflows**: Fixed error estimation methods used by workflows.


### Breaking changes

*   **Removed `num_param_opt` and `opt_filter_min_SN_ratio`**: These parameters have been removed from `run_optimize()`, CLI, workflow, and TOML config. SR and optax now always optimize all parameters; parameter selection is handled internally by the LM subspace mechanism.
*   **Replaced `adaptive_learning_rate` with `use_lm`**: The `adaptive_learning_rate` flag is replaced by the `use_lm` flag, which controls the unified LM/aSR optimizer hierarchy.
*   **Removed `method="lm"` as separate code path**: The Linear Method is now accessed via `method="sr"` with `use_lm=true`.
*   **New optimizer parameter names**: `lm_cond` (default 0.001) replaces the previous LM-specific delta/epsilon naming.


## Mar-10-2026: v0.2.0a1

This is a major update with drastic performance improvements, new features, and a new workflow automation package (`jqmc-workflow`).

### Performance

*   **Drastic speedups**: MCMC, VMC, and LRDMC are all significantly faster than the previous version thanks to pervasive use of fast-update algorithms throughout the code.
*   **LU -> SVD replacement**: Replaced LU factorizations with SVD across determinant, geminal, `GFMC_n`, and `GFMC_t` modules, greatly improving numerical stability for ill-conditioned matrices.
*   **GEMM optimization**: Converted matrix-vector operations to matrix-matrix (GEMM) operations in Coulomb potential, determinant, and Jastrow factor modules for better GPU utilization.
*   **Cartesian / Spherical AO conversion**: Implemented Cartesian AO <-> Spherical AO conversion. Cartesian GTOs are substantially faster than spherical GTOs on GPUs, so users can now exploit this for better throughput.
*   **ECP fast computation**: Implemented `compute_ecp_coulomb_potential_fast` for efficient pseudopotential evaluation.
*   **`vmap` + `jit` fix**: `vmap`-ed functions are now explicitly wrapped with `jit`, as `vmap` does not automatically JIT-compile the mapped function.
*   **Removed `mpi4jax` dependency for CG**: Conjugate gradient (CG) solver now uses pure MPI on CPUs, eliminating the `mpi4jax` dependency.

### Optimization

*   **Adaptive learning rate for Stochastic Reconfiguration**: Implemented a linear-method-inspired automatic learning-rate adjustment scheme, leading to dramatically faster optimization convergence.
*   **Molecular orbital optimization**: Added MO optimization for JSD wavefunctions via the projection method with Attacalite-Sorella regularization.
*   **Geminal AO -> MO projection**: Implemented AO overlap matrix computation and geminal AO -> MO projection for constrained optimization.

### New features

*   **LRDMC force calculations**: Implemented LRDMC atomic forces with the Pathak-Wagner regularization.
*   **Jastrow functions**: Added `jastrow_1b_type` (`'exp'` / `'pade'`) and `jastrow_2b_type` (`'pade'` / `'exp'`) fields to `Jastrow_one_body_data` and `Jastrow_two_body_data`, enabling runtime selection of the one-body and two-body Jastrow functional forms.
    *   Exponential form: $u(r) = \frac{1}{2b}(1 - e^{-br})$
    *   Pade form: $u(r) = \frac{r}{2(1 + br)}$

### Bug fixes

*   Fixed force calculations producing NaN values; added NaN checks in all tests.
*   Fixed MCMC memory overflow caused by storing `r_up_history` / `r_dn_history`.
*   Fixed wavefunction without Jastrow not working for MCMC.
*   Fixed missing NN-Jastrow derivatives in `_GFMC_n_debug`.

### Infrastructure

*   **Restart file format change**: Switched restart files from pickle-based `*.chk` to HDF5-based `*.h5`. **Note:** backward compatibility with old `*.chk` files is *not* maintained.
*   **`jqmc_workflow` package**: Introduced the `jqmc_workflow` automation package for orchestrating multi-stage QMC pipelines (WF conversion -> VMC optimization -> MCMC / LRDMC production) with automatic step estimation, checkpointing, and remote job management.
*   **Removed `SWCT_data`**: Cleaned up legacy `SWCT_data` class as part of codebase refactoring.
*   **More comprehensive tests**: Substantially expanded the test suite to cover the new features and improve overall reliability.
*   **Expanded examples**: Reorganized and enriched the `examples/` directory with 11 end-to-end tutorials (`jqmc-example01`-`jqmc-example08`, `jqmc-workflow-example01`-`jqmc-workflow-example03`) covering single-point VMC/LRDMC, force calculations, GPU walker-scaling benchmarks, interaction-energy workflows, and PES scans with automated `jqmc_workflow` pipelines.


## Feb-5-2026: v0.1.0

- Release of the first stable version of **jQMC**.

### Known Limitations

*   Periodic Boundary Condition (PBC) calculations are being implemented for the next major release.

## Jan-23-2026: v0.1.0a3

- Release of the third alpha version of **jQMC**.

### Key Features

*   **Analytical derivatives**:
    *   Implemented analytical gradients and Laplacians for atomic and molecular orbitals in both spherical and Cartesian GTO bases.
    *   JAX autograd is now used primarily for validating the analytical gradients.
    *   Logarithmic derivatives of the wavefunction and derivatives of atomic force calculations still use JAX autograd.
*   **Testing precision**:
    *   Tightened and systematized decimal controls in tests, improving overall reliability.
*   **Fast updates**:
    *   Expanded fast-update implementations to more functions, yielding significant speedups in both MCMC and GFMC modules.

## Jan-14-2026: v0.1.0a1

- Release of the second alpha version of **jQMC**.

### Key Features

*   **Neural Network Jastrow**:
    *   Introduced `NNJastrow`, a PauliNet-inspired neural network architecture for many-body Jastrow factors, enabling more accurate wavefunction ansatz.
*   **Optimization Control**:
    *   Implemented proper gradient masking mechanisms (e.g., `with_param_grad_mask`). This allows for selectively freezing or optimizing specific parameter blocks (One-body, Two-body, Three-body, NN, and Geminal coefficients) during the VMC optimizations.

### Enhancements & Fixes

*   **I/O**: Changed the storage format for `hamiltonian_data` from pickled binary files to HDF5 (`.h5`) for better portability and compatibility.
*   **Documentation**: Updated `README.md`, docstrings, and API references to reflect recent changes and fix Sphinx warnings.
*   **Code Quality**: Refactored code based on suggestions and improved type hinting.

## Aug-20-2025: v0.1.0a0

- Release of the first alpha version of **jQMC**.

We are pleased to announce the first alpha release of **jQMC**, a Python-based Quantum Monte Carlo package built on **JAX**.

### Key Features

*   **JAX-based Core**: Fully utilizes JAX's Just-In-Time (JIT) compilation and automatic vectorization (`vmap`) for high-performance simulations on GPUs and TPUs.
*   **Algorithms**:
    *   **Variational Monte Carlo (VMC)**: Supports wavefunction optimization via Stochastic Reconfiguration (SR) and Natural Gradient methods.
    *   **Lattice Regularized Diffusion Monte Carlo (LRDMC)**: A stable and efficient projection method for ground state calculations.
*   **Wavefunctions**:
    *   **Ansatz**: Supports Jastrow-Slater Determinant (JSD) and Jastrow-Antisymmetrized Geminal Power (JAGP).
    *   **Jastrow Factors**: Includes One-body, Two-body, Three/Four-body terms.
    *   **Determinant Types**: Single Determinant (SD), Antisymmetrized Geminal Power (AGP), and Number-constrained AGP (AGPn).
*   **I/O & Interoperability**:
    *   **TREXIO Support**: Interfaces with the [TREXIO](https://github.com/TREX-CoE/trexio) library (HDF5 backend) for standardized input of molecular structure and basis sets (Cartesian & Spherical GTOs).
*   **Parallelization**:
    *   **MPI Support**: Implements `mpi4py` for efficient parallelization across multiple nodes.
*   **Documentation**:
    *   Comprehensive technical notes on Wavefunctions, VMC, LRDMC, and JAX implementation details.
    *   Examples demonstrating usage for various systems (H2, N2, Water, etc.).

### Known Limitations (Alpha)

*   Periodic Boundary Conditions (PBC) are currently in development.
*   Atomic force calculations with spherical harmonics are computationally intensive on current JAX versions.
*   Complex wavefunctions are not yet supported.
