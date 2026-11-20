(execution_guide)=

# Execution and convergence guide

## Choose the backend and precision separately

JAX device selection and the CPU hot-loop backend are different settings.
Set environment variables **before launching** Python or `jqmc`:

| Device | CPU Rust flag | MCMC / LRDMC hot loop | Precision |
|---|---|---|---|
| CPU | unset, or `JQMC_FLAG_RUST_ON_CPU=1` | Native `jqmc_rs` by default | full and mixed |
| CPU | `JQMC_FLAG_RUST_ON_CPU=0` | JAX/XLA | full and mixed |
| GPU | either | JAX/XLA; the CPU flag is irrelevant | full and mixed |

An enabled CPU Rust backend with a missing extension raises an error; it
does not silently select XLA. Build/install the matching native extension
as described in {doc}`install`, or explicitly choose XLA:

```bash
# From a prepared calculation directory; these are alternatives.
JAX_PLATFORMS=cpu JQMC_FLAG_RUST_ON_CPU=1 JQMC_RS_NUM_THREADS=1 jqmc mcmc.toml
JAX_PLATFORMS=cpu JQMC_FLAG_RUST_ON_CPU=0 jqmc mcmc.toml
```

`JQMC_RS_NUM_THREADS=1` in the first line keeps a small test on one native
thread; do not copy it into production runs, where you normally leave it
unset (see the next section).

This selects the sampling backend, not an all-native Python-free program.
Setup, conversion, analysis, and some other operations still use Python/JAX.
`[precision].mode = "full"` or `"mixed"` controls numerical zones;
mixed does not mean every operation is fp32. See {doc}`notes/mixed_precision`.
Verify full/mixed agreement for the observable and trial you need; a smaller
dtype is not by itself a speed or accuracy guarantee.

The native extension includes AO/MO and determinant evaluation, periodic
Jastrow and Coulomb/ECP kernels, analytic parameter/position/strain response,
and real/complex MCMC and GFMC driver chunks with native random-number
handling. It is not limited to value-only, RNG-free kernels. PyO3 exposes
NumPy-array interfaces; native contexts also own prepared data and scratch
buffers, so zero-copy array views do not imply that an entire calculation
performs no copies. Python retains conversion, orchestration, optimization,
MPI coordination, persistence and analysis; JAX remains the GPU path and an
independent implementation/automatic-differentiation reference where available.

No universal bitwise-equivalence claim is made by selecting the Rust backend;
agreement of values on one input does not verify derivatives, updates or a
whole stochastic trajectory. A source checkout on `PYTHONPATH` does not
prove that `jqmc_rs` was built from it: rebuild the extension
(`pip install -e .`) after changing the Rust sources.

## MPI, walkers, and threads

`number_of_walkers` is per MPI rank. On CPU, keep the number of ranks times
native threads within your allocated cores and bind ranks accordingly.
`JQMC_RS_NUM_THREADS` is the Rust walker-batch thread count, not an MPI rank
count or a replacement for `OMP_NUM_THREADS` / BLAS controls. It takes effect
when the process first initializes the native thread pool. The low-level
precedence is explicit `configure_threads(n)`, then this environment
variable, then `RAYON_NUM_THREADS`, then the affinity-aware default (the
cores the rank is bound to). Changing it after initialization does not
resize the pool. The `jqmc` header prints the width actually used as
`Rust rayon threads = N (per MPI rank)`.

Set CPU and NUMA placement at launch through the MPI launcher or batch
scheduler. For a standalone process, tools such as `taskset` and `numactl`
set the corresponding policies before starting jQMC.

NUMA placement affects speed, not results. On a node with several NUMA
domains (several sockets, or one processor split into memory domains),
memory that sits far from the cores using it can make a run noticeably
slower. A single process whose threads span several domains is the most
exposed case, and the memory-bound SR/CG solve of the optimizer is the
most sensitive step. Keep placement in mind when you choose ranks, threads
and binding, and check it when a run is slower than expected.

Typical launch lines (the output goes to `out_mcmc`, errors to `out_mcmc.e`):

```bash
# CPU, 4 MPI ranks; each rank uses the cores it is bound to
mpirun -np 4 jqmc mcmc.toml > out_mcmc 2> out_mcmc.e

# GPU node with 4 GPUs: one MPI rank per GPU (the mapping option depends on
# the MPI library and the scheduler)
mpiexec -n 4 -map-by ppr:4:node jqmc mcmc.toml > out_mcmc 2> out_mcmc.e
```

What the run prints at the end and which files it writes is described in
{ref}`api_reference_cli_outputs`.

On GPU, run one MPI rank per GPU, configure visible devices in the job
template, and verify the device assigned to each rank (the header reports
whether the JAX distributed initialization succeeded). For twist averaging, the MPI size must be divisible
by the number of twists, with an equal-size sampler group per twist. The
workflow queue label is only a configuration key; inspect the actual submit
template and scheduler resource request.

## Warmup, compilation, and throughput

Expect the first steps of a run to be slow: JAX compiles (just-in-time, JIT)
every kernel the first time it is called, and the force and stress kernels
take much longer to compile than the energy-only ones. This start-up cost is
paid once per run and does not reflect the cost of later samples.

`atomic_force` and `compute_stress` are independent: energy-only sets both
false; force-only sets just the former; stress-only sets just the latter;
both true requests both estimators. Stress is periodic-only. Leave unused
observables off to avoid their work and memory.

### Benchmarking

When you compare timings, report conversion, start-up (including the first
JIT compilation), sampling, and postprocessing separately, and compare
steady-state cost on the same trial wavefunction, device, precision, number
of walkers, observable flags and CPU/NUMA placement. No universal
force-to-energy cost ratio applies across basis representations and walker
counts. To benchmark the
JAX/XLA path on CPU, disable the Rust backend explicitly
(`JQMC_FLAG_RUST_ON_CPU=0`); see {doc}`notes/analytic_derivatives`.

## From a successful smoke to a scientific result

1. Check the installation with the commands in {doc}`install`, then run a
   short calculation from one of the tutorials (for example the H2 molecule in
   `jqmc-example05`). A finite error bar there only checks the installation,
   input, sampling, checkpoint, and analysis route.
2. Optimize the trial and inspect energy/gradient convergence. Select an
   actual `hamiltonian_data_opt_step_N.h5`, not the unchanged VMC input.
   Keep input identity with each checkpoint and use fresh sampling directories
   when changing the trial. A dump is not an optimizer-state checkpoint.
3. Run independent sampling with enough equilibration and long time bins.
   Postprocessing does not reuse TOML options: specify warmup and the number
   of bins, and for LRDMC the weight history (`-c`, the number of past
   branching weights in each correcting factor). Check that every
   bin is populated and that estimates/errors are stable as these choices
   and run length change. See {doc}`notes/error_analysis` and
   {doc}`api_reference_cli_tool`.
4. Converge systematic approximations separately: basis/projection residual,
   spline mesh, Ewald/MPC cutoffs, cell size/twist sampling, LRDMC lattice
   spacing and population/history dependence. Statistical errors do not
   include these, the fixed-node/fixed-phase bias (LRDMC keeps the nodes, or
   for complex wavefunctions the phase, of the trial wavefunction), the
   localization error of pseudopotentials (ECPs) in LRDMC, or the
   approximations of the LRDMC force estimator.

MPC adds a passive energy correction measured on Ewald-sampled walkers;
it does not change sampling, branching, forces, or stress. Converge the
correction's cutoff (`mpc_cutoff`) and quote the corrected total `E_MPC`
printed by the run; its error bar already includes the correlation between
the energy and the correction. For twist-averaged runs, use the
Brillouin-zone (BZ) averaged values, which weight each twist with its k-point
weight. An energy error target is not a force or
stress convergence target.
