# Examples

The examples are grouped by interface and physical scope. Commands assume that
`jqmc` and `jqmc-tool` come from the same checkout/version as the example
files. Mean-field and workflow examples also require their explicitly listed
external programs and machine configuration.

## Running an example

Use a checkout or source archive matching the installed jQMC version: a
`pip install` alone does not provide this examples tree. Run each tutorial
inside its own directory, `examples/<example>`, and follow its `cd` commands
in order. Some tutorials read shared files from sibling directories (QE
inputs and outputs, the downloaded `Si.ccECP.upf` of `jqmc-example07`,
`copy_optimized_hamiltonian.py`), so keep the tree intact. Pseudopotential
(UPF) files are not shipped; each tutorial that needs one gives its download
URL and where to place it. Generated
files (HDF5 checkpoints, logs) are written next to the inputs; use `git clean`
or a separate checkout if you want to keep the tree pristine. The listings of
whole Python files in the tutorials omit the licence header that every shipped
`.py` file carries; the rest of each listing is the file as shipped.

VMC leaves its original `hamiltonian_data.h5` unchanged and writes
`hamiltonian_data_opt_step_N.h5`. Before MCMC or LRDMC the tutorials run
`copy_optimized_hamiltonian.py --average-last 20`, which selects the 20
dumps with the largest **numeric** step and writes their uniform parameter
average with `jqmc-tool vmc average-wf` to the downstream input name (see
"Averaging the last optimization steps" in
[`doc/notes/wavefunction_optimization.md`](../doc/notes/wavefunction_optimization.md)).
The VMC inputs that feed such a hand-off therefore dump every step
(`wf_dump_freq = 1`). The helper refuses when fewer than 20 dumps exist
(`--allow-fewer` accepts them), when the destination already exists, and
whenever `average-wf` refuses the dumps. Without `--average-last` it copies
the single largest numeric step; the tutorials still do that where an
optimized wave function only starts a further optimization or a conversion.
Both forms work after partial or continued runs. Review optimization
convergence first: the averaged steps must lie on the energy plateau. Use
`--step N` to end the window at (or, without `--average-last`, to copy) a
particular reviewed dump, for example the last step before an optimizer
rollback. Use a fresh downstream directory for a different trial; never mix
its input with an old `restart.h5`.

Each command-line tutorial ends its MCMC and LRDMC steps with an **Analyze**
block. Two places report the energy:

- the `Final output(s):` block at the end of the run log (`out_mcmc`,
  `out_lrdmc`), which prints `Total Energy: E = ... +- ... Ha.` and
  `Variance: Var = ... +- ... Ha^2.` (and the atomic forces when
  `atomic_force = true`), computed with the warmup and bin settings of the
  TOML;
- `jqmc-tool mcmc compute-energy` / `compute-force` and
  `jqmc-tool lrdmc compute-energy` / `compute-force` / `extrapolate-energy`,
  which re-analyze the saved `restart.h5` checkpoint with the settings you
  pass on the command line.

The `jqmc-tool` commands do **not** read the TOML, and their default
options are below the minimums, so always pass them explicitly: `-w`
(warmup measurements to discard), `-b` (number of bins) and, for LRDMC,
`-c` (the correcting-factor history `num_gfmc_collect_steps`: the number of
past measurements whose walker weights are multiplied together to correct
the bias introduced by population control). The minimums are 30 warmup /
10 bins for MCMC and 25 warmup / 10 bins / 5 history for LRDMC. Each
tutorial shows the values that match its TOML. `jqmc-tool` prints the
number of jackknife samples as `Total number of binned samples = M`; for
MCMC $M$ = bins x walkers x MPI ranks, for LRDMC $M$ = bins. A finite error
bar is not evidence that these settings are converged: check that the
energy and its error bar stay stable when you change `-w` and `-b`.

## Before you start (all command-line tutorials)

- Launch lines are written as `mpirun -np N jqmc ...`; replace them with the
  launcher of your MPI library and queueing system. A serial run
  (`jqmc input.toml`) uses the same inputs but takes N times longer.
- Keep the supplied TOML files. They are the inputs behind the quoted
  numbers; `jqmc-tool * generate-input` templates use different settings.
- `jqmc` warns about unknown TOML sections and keys, and about sections
  that the selected `job_type` ignores. Read these warnings and the
  `Input parameters are::` echo at the top of each log before trusting a run.
- Mean-field inputs: the molecular tutorials, example06 and examples 11 to 13
  need PySCF with TREXIO export (for example
  `pip install pyscf pyscf-forge trexio`; check with
  `python -c "from pyscf.tools import trexio"`). The QE tutorials need a
  `pw.x` built with HDF5 so that it writes `wfc*.hdf5`.
- Every quoted number was produced on one 128-core CPU node (AMD EPYC 7713)
  with 128 MPI ranks unless the tutorial says otherwise; the wall times in
  the table below refer to that setup.

## What the main input keys mean

| Key | Meaning and how to choose it |
|---|---|
| `number_of_walkers` | Independent Markov chains per MPI rank. More walkers give more samples per step; on GPUs they are evaluated in parallel. |
| `num_mcmc_steps` | Number of measurements per rank and walker (the run length). |
| `num_mcmc_per_measurement` | MCMC: electron moves between two measurements. Larger values reduce the correlation between successive measurements. |
| `tau` | LRDMC (`lrdmc-tau`): imaginary time between two branchings (Ha^-1). All tutorials use `tau = 0.10`; the number of projections per branching then follows from the system and `alat`. |
| `num_mcmc_warmup_steps`, `num_gfmc_warmup_steps` | Initial measurements discarded before averaging (equilibration). |
| `num_mcmc_bin_blocks`, `num_gfmc_bin_blocks` | Number of bins of the jackknife error estimate. Each bin must be longer than the autocorrelation time, so at fixed run length fewer, longer bins are safer. |
| `Dt` | MCMC trial-move length (bohr). Too small explores slowly; too large lowers the acceptance ratio. |
| `epsilon_AS` | Attaccalite-Sorella reweighting parameter. A value $> 0$ tames the infinite variance of the determinant-parameter derivatives near the nodal surface; it matters when the determinant is optimized (`opt_geminal = true`). |
| `optimizer_kwargs` | Stochastic reconfiguration (SR) settings: `delta` is the step prefactor and `epsilon` the regularization added to the overlap matrix. `use_lm = true` with `lm_subspace_dim = 0` (the default) is accelerated SR (aSR), which chooses the length of each SR step from a model of the energy along that step; `lm_subspace_dim > 0` selects a reduced linear method and `-1` the full linear method; `use_lm = false` gives plain SR with the fixed step `delta`. Keys that a TOML does not set keep their documented defaults. |
| `alat` | LRDMC lattice spacing $a$ (bohr). The energy has an $O(a^2)$ bias: extrapolate to $a \to 0$, or use the same $a$ for every term of an energy difference. |
| `non_local_move` | Treatment of the non-local ECP term in LRDMC: `tmove` (T-move) or `dltmove` (determinant locality approximation with T-move). It has no effect in an all-electron calculation. |
| `num_gfmc_collect_steps` | Correcting-factor history (see above). |
| `atomic_force`, `use_swct` | Compute atomic forces; the space-warp coordinate transformation (SWCT) reduces the variance of the force estimator. |

The `-j3` option of `jqmc-tool trexio convert-to` selects the basis of the
three-body Jastrow factor: `mo` uses the molecular orbitals; `ao-small`,
`ao-medium` and `ao-large` use increasingly large subsets of the atomic
orbitals; `ao` and `ao-full` use all of them; `none` disables the three-body term.

## Restarting a run

Set `restart = true` in `[control]` (with `restart_chk` naming the
checkpoint) and run `jqmc` again in the same directory. For a VMC
optimization, `num_opt_steps` is then the number of **additional** steps,
and the `hamiltonian_data_opt_step_N.h5` dumps continue the step numbering.
Keys restored from the checkpoint (`mcmc_seed`, `number_of_walkers`,
`hamiltonian_h5`, `num_mcmc_per_measurement`, `Dt`, `epsilon_AS`; for LRDMC
also `tau`, `alat`, `non_local_move`, `num_gfmc_collect_steps` and the force
and stress switches) must
not be changed: `jqmc` refuses a restart in which an explicitly written
frozen key differs from the checkpoint value.

## Choose a periodic input route: PySCF or Quantum ESPRESSO

| Mean-field input | Conversion to the jQMC determinant | Start here |
|---|---|---|
| PySCF periodic Gaussian orbitals | PySCF -> TREXIO -> pGTO, using `jqmc-tool trexio convert-to`; no QE calculation is needed. | `jqmc-example06`: ice Ih at a non-Gamma complex twist, VMC optimization, MCMC and LRDMC. |
| QE plane waves, converted to B-splines | QE `.save` -> `jqmc-tool qe --kind bspline`. | `jqmc-example07`: Gamma-point silicon, VMC optimization, MCMC, LRDMC and MPC. Also used by `jqmc-example09`, `jqmc-example10` (LSDA) and `jqmc-workflow-example04`/`06`. |
| QE or PySCF k cosets tiled into a supercell | Add `--tiling "2 1 1"` to QE PW/B-spline/pGTO/PW+pGTO or TREXIO GTO conversion; select a supertwist or weighted ensemble, with frozen Slater orbitals. | `jqmc-example14`: common primitive k blocks and independent supercell J3 or DTN. |
| QE plane waves, converted to pGTO+PW | QE `.save` -> `jqmc-tool qe --kind pgto+pw --ecutwfc auto`. | `jqmc-example08`: all-electron hydrogen at a non-Gamma twist, VMC optimization, MCMC and LRDMC with forces; `jqmc-workflow-example05` is its workflow counterpart. |

The fragments above only identify the converter options; use the complete
commands in the selected tutorial. A B-spline mesh or a pGTO+PW cutoff is an
approximation, so inspect the conversion diagnostics and converge observables,
not just the orbital-fit residual.

Example07 ships the QE save directory of its small run without the
pseudopotential: download `Si.ccECP.upf` from
<https://pseudopotentiallibrary.org/recipes/Si/ccECP/Si.ccECP.upf> into
`jqmc-example07/01DFT/` and copy it into `01DFT/silicon.save/` (step 1 of the
tutorial), and the conversion needs no local QE run. The other QE tutorials
run `pw.x` first. For **multiple twists**,
use `jqmc-example09` (CLI) or `jqmc-workflow-example06` (workflow); example06
and example08 are single-twist tutorials. A multi-k PySCF state exported to
TREXIO is converted with `jqmc-tool trexio convert-to --twist-average` or
`WF_Workflow(twist_average=True)`; the downstream steps are those of the QE
tutorials.

All periodic tutorials use insulating or fixed-occupation mean-field states.
jQMC has no grand-canonical twist averaging, and `jqmc-tool qe` rejects
smeared occupations, so metals are not a supported production target; see
"Fixed electron numbers and metals" in the wave-function technical note.

## Command-line tutorials

Each example lives in the directory of the same name under `examples/`
(browse them at <https://github.com/jqmc-project/jQMC/tree/main/examples>,
or in the source archive of your installed version).

| Example | System and purpose |
|---|---|
| `jqmc-example01` | Water (ccECP): PySCF to TREXIO, JSD optimization, MCMC, and LRDMC (`lrdmc-tau`) at five lattice spacings with continuum extrapolation. |
| `jqmc-example02` | Water with a two-body Jastrow only: VMC optimization (with a restart) and MCMC energy and atomic forces. |
| `jqmc-example03` | Water: Jastrow-only optimization followed by geminal/MO optimization through `opt_geminal`, and the MCMC energies of both trials. |
| `jqmc-example04` | S22 water-dimer binding energy (aug-cc-pVDZ) with JSD and JAGP: VMC, MCMC and LRDMC for two monomers and the dimer. |
| `jqmc-example05` | All-electron H2: energies and analytic atomic forces with JSD and JAGP, MCMC and LRDMC. |
| `jqmc-example06` | Periodic ice Ih from PySCF pGTOs at a non-Gamma complex twist: VMC, MCMC, and LRDMC at three lattice spacings with continuum extrapolation. |
| `jqmc-example07` | Periodic silicon from QE with a B-spline determinant and DTN Jastrow: VMC, MCMC, LRDMC, and mixed-precision MCMC with MPC. |
| `jqmc-example08` | Periodic all-electron hydrogen from QE (pGTO+PW, DTN cusp): VMC optimization, MCMC and LRDMC energies and forces. |
| `jqmc-example09` | Eight-twist silicon (QE, B-spline): twist-averaged MCMC and `lrdmc-tau` with stress, pressure and BZ-averaged MPC in mixed precision, extrapolation, and restart of a twist checkpoint directory. |
| `jqmc-example10` | Unrestricted open shells: nitrogen atom with ccECP (PySCF UHF) optimization and sampling, and a silicon atom from QE LSDA (B-spline) in mixed precision. |
| `jqmc-example14` | QE/PySCF orbital tiling with frozen Slater coefficients, Gamma/general supertwists and ensembles; independent supercell AO J3 or DTN. |

Size of the shipped inputs and wall time of the reference runs (one 128-core
node, 128 MPI ranks, per job):

| Example | External programs | Shipped size | Wall time of the reference run |
|---|---|---|---|
| `jqmc-example01` | PySCF | VMC 300 steps; MCMC 300 walkers/rank x 5000; LRDMC 5 spacings x 300 walkers/rank x 1500 | VMC 55 min, MCMC 20 min, LRDMC 20 min (a = 0.30) to 153 min (a = 0.10) |
| `jqmc-example02` | PySCF | VMC 200 steps; MCMC with forces 100 walkers/rank x 3000 | VMC 4 min, MCMC 19 min |
| `jqmc-example03` | PySCF | VMC 150 (Jastrow) + 200 (geminal) steps; two MCMC runs of 100 walkers/rank x 10000 | 19 min + 83 min, MCMC about 9 min each |
| `jqmc-example04` | PySCF | Three systems x (JSD, JAGP) x (VMC, MCMC, LRDMC) | about 3 h for both monomers, about 9 h for the dimer |
| `jqmc-example05` | PySCF | 4 walkers/rank; JSD and JAGP VMC, MCMC and LRDMC | about 50 min in total |
| `jqmc-example06` | PySCF (`pyscf.pbc`) | VMC 150 steps; MCMC 5000; LRDMC 3 spacings | VMC 1 h 41 min; MCMC 8 min; LRDMC 53 to 102 min each |
| `jqmc-example07` | QE (optional; save directory shipped, UPF downloaded) | VMC 200 steps; MCMC 5000; LRDMC 5000; MPC MCMC 5000 | VMC 80 min; the others 1 to 7 min |
| `jqmc-example08` | QE | VMC 200 steps; MCMC and LRDMC 20000 with forces | VMC 81 min; MCMC 2.5 min; LRDMC 6.5 min |
| `jqmc-example09` | QE | 8 twists x 8 walkers; MCMC 1000; LRDMC 2000 at three spacings | see the tutorial |
| `jqmc-example10` | PySCF, QE | N: VMC 100 steps, MCMC 5000; Si LSDA: MCMC 5000 | VMC 39 min; MCMC 2 to 6 min |

## Feature index

| Feature | Start here |
|---|---|
| Molecular all-electron calculation | `jqmc-example05` |
| Periodic Gamma calculation | `jqmc-example07` |
| Non-Gamma complex Bloch orbitals | `jqmc-example06` or `jqmc-example08` |
| Twist averaging | `jqmc-example09` (CLI) or `jqmc-workflow-example06` (workflow) |
| Supercell orbital tiling with Gamma/general supertwists or twist averaging | `jqmc-example14` |
| Mixed precision | `jqmc-example07/05mcmc_JSD_mixed_mpc`, `jqmc-example09`, or `jqmc-example10` |
| Atomic forces | `jqmc-example02`, `jqmc-example05` (molecules), `jqmc-example08` (periodic), `jqmc-workflow-example01`, `03`, `05` |
| Stress and pressure | `jqmc-example09` |
| MPC finite-size energy correction | `jqmc-example07/05mcmc_JSD_mixed_mpc` (single twist) or `jqmc-example09` (twist average) |
| Open-shell / unrestricted determinant (UHF and LSDA) | `jqmc-example10` |
| Jastrow optimization | `jqmc-example01`, `jqmc-example02`, `jqmc-example07` |
| Determinant (geminal/MO, JAGP) optimization | `jqmc-example03`, `jqmc-example04`, `jqmc-example05`, `jqmc-example08` |
| Multi-twist VMC optimization and downstream handoff | `jqmc-workflow-example06` |
| Equation of state (volume scan) | `jqmc-workflow-example04` |
| LRDMC continuum extrapolation | `jqmc-example01`, `jqmc-example06`, `jqmc-example09`, `jqmc-workflow-example04` |
| Restart and continuation | [Restarting a run](#restarting-a-run) above; VMC restart in `jqmc-example02` and `jqmc-example04`; twist checkpoint directory in `jqmc-example09` |
| Serial / MPI Python API | `jqmc-example12` / `jqmc-example13` |
| Orbital and Hamiltonian kernels from Python (AOs, MOs, Coulomb, ECP) | `jqmc-example11` |

## Python API tutorials

| Example | System and purpose | Wall time of the reference run |
|---|---|---|
| `jqmc-example11` | Low-level Python API: read a TREXIO file into the jQMC data containers and evaluate AOs, MOs, bare Coulomb, and ECP local/non-local terms at one electron configuration. | 10 s, serial |
| `jqmc-example12` | Serial end-to-end Python API for water: TREXIO conversion, VMC optimization and sampling, LRDMC at three spacings, and extrapolation (2000 walkers). | 61 min, one process |
| `jqmc-example13` | MPI version of `jqmc-example12` using `mpi4py` and `run_jqmc_mpi.py` (2000 walkers in total). | 32 min, 125 ranks |

## jqmc_workflow tutorials

| Example | System and purpose |
|---|---|
| `jqmc-workflow-example01` | H2 potential-energy surface at seven bond lengths and force/PES-derivative comparison. |
| `jqmc-workflow-example02` | GPU walker/MPI vectorization benchmark. |
| `jqmc-workflow-example03` | Molecular water VMC, MCMC, and LRDMC energy/force pipeline. |
| `jqmc-workflow-example04` | Silicon equation of state: QE, B-spline conversion, VMC, MCMC and LRDMC over five lattice constants with a Birch-Murnaghan fit, plus an LRDMC lattice-spacing extrapolation. |
| `jqmc-workflow-example05` | Periodic all-electron hydrogen QE-to-jQMC workflow with DTN and forces (workflow counterpart of `jqmc-example08`). |
| `jqmc-workflow-example06` | Eight-twist QE conversion, mixed-precision VMC optimization, publication of the complete optimized Hamiltonian, and downstream MCMC handoff. |

The workflow engine submits ready jobs until the number of the driver's jobs
in a scheduler queue reaches `max_job_submit`; further submissions wait and
retry every `poll_interval` seconds until a slot frees. By default only the
jobs submitted by this driver are counted; `max_job_submit_scope = "user"`
counts all of your jobs in that queue.

## Files at the top of `examples/`

- `copy_optimized_hamiltonian.py`: hands the optimized VMC trial to the
  next step, either the average of the last N dumps (`--average-last N`)
  or a copy of one `hamiltonian_data_opt_step_N.h5`; used by the
  command-line tutorials.
- `sync_examples.py`: copies the input files named by the README include
  markers into the code blocks that follow them; running the examples does
  not need it.

## About the quoted numbers

Energies and forces quoted in the tutorials are statistical results of one
reference run made with the shipped inputs, except where a note says
otherwise (see the setup recorded in each tutorial). Your run uses different random numbers (and possibly a
different number of MPI ranks), so expect agreement within a few error bars,
not digit-by-digit equality. The warmup, bin and history settings of the
shipped TOMLs are starting points, not evidence of decorrelation or
convergence; at fixed run length more bins means shorter bins.

The MCMC and LRDMC numbers quoted after a hand-off were not sampled from
the average of the last 20 optimization steps that the shown commands
write: those reference runs sampled the single last dump. Each such
hand-off carries a note naming the dump that was used. Where the reference
VMC run dumped only every 10, 20 or 50 steps, the shipped input dumps every
step, which the averaged hand-off needs; the dump frequency does not change
the optimization itself.
