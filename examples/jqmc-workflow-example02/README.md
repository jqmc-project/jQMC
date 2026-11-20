# jqmc-workflow-example02: GPU walker-scaling benchmark

A water ccECP / Cartesian cc-pVTZ benchmark sweeping 8 through 8192 walkers
per MPI rank. PySCF DFT-LDA prepares the trial locally; VMC refines Pade J2
and AO-small J3 with a fixed determinant. The mean-field input and device
assignment must be identical when comparing timings.

## Setup and launch

Requires an installed jQMC matching this checkout, PySCF with TREXIO export,
and a workflow machine configuration. The script changes into its own
directory before launching, so the configuration is read either from
`jqmc_setting_local/` placed **inside this example directory** (a
project-local override) or from the user-global `~/.jqmc_setting/`.

Before launching, edit the server/queue names below and provide the matching
machine entry, queue definition, submit template, and job limits. They are
site-specific defaults, not a ready-to-run local configuration. See the
**Configuration** section of the workflow notes (`doc/notes/workflows.md`)
for complete local and remote examples. Inspect the resource request before
submitting; a queue label alone does not configure a machine.

```bash
python run_pipelines.py
```

This is a production-sized workflow, not an installation smoke. Time limits
and continuation caps can stop a run before the requested error is reached.
Check the final workflow status, optimization history, and statistical
convergence, not just the existence of an output file.

## Settings taken from the executable script

Edit the script, not this generated block. Refresh with
`python examples/sync_examples.py` from the checkout root.

<!-- include: run_pipelines.py#configuration -->
```python
SERVER = "cluster"
QUEUE_LABEL_s = "cores-4-mpi-4-gpu-4-omp-1-30m"
QUEUE_LABEL_l = "cores-4-mpi-4-gpu-4-omp-1-3h"

NUM_OPT_STEPS = 50  # VMC optimization steps
WF_DUMP_FREQ = 1  # dump every step: VMC_Workflow hands on the average of the last 20 steps
Dt = 2.0  # MCMC hopping distance (bohr)
TARGET_VMC_ERROR = 1.0e-3  # Target VMC optimization error (Ha)
ALAT = 0.30  # LRDMC lattice spacing
NUM_MCMC_PER_MEASUREMENT = 40  # LRDMC GFMC projections per measurement

# Explicit measurement-step counts for the benchmark
# (no target-error automatic convergence)
NUM_MCMC_STEPS_MCMC = 1000  # MCMC measurement steps per production
NUM_MCMC_STEPS_LRDMC = 1000  # LRDMC measurement steps per production

# Walker counts for vectorization benchmark
WALKER_COUNTS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# pySCF file names
TREXIO_FILE = "water_trexio.hdf5"
PYSCF_OUTPUT = "water.out"
```

## Dependency graph and measurement counts

The WF and VMC stages run once. Each walker count has its own pair of jobs:

```text
WF -> VMC -- optimized Hamiltonian --> MCMC(nw)
         -- optimized Hamiltonian --> LRDMC(nw)
                                      ^
                    MCMC(nw).energy --+  (E_scf)
```

Different walker-count pairs can run concurrently, but LRDMC waits for
its corresponding MCMC energy. The 22 sampling jobs are **not** all
independent. The production counts are supplied through
`num_mcmc_steps=NUM_MCMC_STEPS_MCMC` and
`num_gfmc_projections=NUM_MCMC_STEPS_LRDMC`, not `pilot_steps`.
Here the LRDMC count is the driver's number of observable measurements;
`num_projection_per_measurement=40` sets projections between measurements
and, because it is set, selects the `lrdmc-bra` (GFMC_n) driver.

VMC uses 256 walkers per rank and SR with
`delta=0.350, epsilon=0.001, use_lm=True` (accelerated SR, since the
subspace dimension is left at its default 0); it optimizes J2 and J3 with the
determinant fixed. VMC and MCMC use 30 warmup measurements and 10 bins; LRDMC
uses 25 warmup measurements, 10 bins and a 5-measurement correcting-factor
history (the code minimums). These settings are chosen for throughput
comparison, not for converged scientific estimates.

## Optimized-trial handoff and continuation

Downstream nodes use
`FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian"))` (with the
corresponding per-geometry VMC label in the PES example) and rename the
staged file to `hamiltonian_data.h5`. The published value is
`hamiltonian_data_opt_averaged.h5`, the average of the 20 largest numeric saved
steps (`handoff_average_steps`, including continued runs), so the script dumps
every step (`WF_DUMP_FREQ = 1`). It is not inferred from `NUM_OPT_STEPS`, and
the converter's original input is never staged beside it.
`handoff_mode="last"` hands over the last saved step instead. The benchmark
plot below was measured with a single-step hand-off of the last saved step,
not with the averaged trial.

Inspect convergence before accepting that trial. To compare another trial,
use fresh downstream directories: an existing staged file is not
automatically overwritten, and its checkpoint must not be reused with a
different Hamiltonian.

## Resources and output

The supplied queue labels describe four MPI ranks and four GPUs. Configure
one visible GPU per rank in the submit template. This can require substantial
device memory at the upper walker counts; reduce the sweep to fit your
allocation. The script does not discover or reserve GPUs by itself.

Results are stored under `03_mcmc/wNNNNN/` and `04_lrdmc/wNNNNN/`.
The summary reports net execution time from the logs, separately from
initialization/JIT warmup. Record jQMC, JAX, precision, hardware, MPI layout,
and both warmup and steady-state timings when comparing results.

The plot was measured on one node with four NVIDIA H100 GPUs (the GENKAI
system). It is an illustrative benchmark, not a performance guarantee for
your jQMC, JAX and CUDA versions or your hardware:

![Benchmark](jqmc_vectorization_benchmark.jpg)
