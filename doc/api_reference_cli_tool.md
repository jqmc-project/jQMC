(api_reference_cli_tool_link)=

# API reference for the command-line tool (jqmc-tool)

`jqmc-tool` provides ancillary CLI utilities for TREXIO conversion, Hamiltonian inspection/conversion, and pre/post-processing of VMC/MCMC/LRDMC workflows. Typical invocation:

```text
jqmc-tool <group> <command> [options]
```

Use the installed console entry point. `python -m jqmc.jqmc_tool <group>
<command> [options]` runs the same CLI (useful when the console script is not
on `PATH`). Every command prints its options with `--help`.

Groups: `trexio`, `hamiltonian`, `qe`, `vmc`, `mcmc`, `lrdmc`.

---

## `trexio` group

### `show-info`
Show concise metadata stored in a TREXIO file.

```
jqmc-tool trexio show-info <filename>
```
- `filename` (arg): TREXIO file.

### `show-detail`
Dump full objects (structure, AOs, MOs, geminal, Coulomb).

```
jqmc-tool trexio show-detail <filename>
```
- `filename` (arg): TREXIO file.

### `convert-to`
Convert TREXIO to `hamiltonian_data.h5` with optional Jastrow setup.

```
jqmc-tool trexio convert-to <trexio_file> [options]
```
- `-o, --output` (str, default `hamiltonian_data.h5`): Output filename.
- `-j1, --jastrow-1b-parameter` (float|None): J1 parameter. If set and ECP is present, core electrons are taken from ECP; otherwise zeroed.
- `--jastrow-1b-type` (`exp|pade`, default `exp`): Functional form of the one-body Jastrow. `exp` uses $u(r) = \frac{1}{2b}(1 - e^{-br})$; `pade` uses $u(r) = \frac{r}{2(1 + br)}$.
- `-j2, --jastrow-2b-parameter` (float|None): J2 parameter.
- `--jastrow-2b-type` (`pade|exp`, default `pade`): Functional form of the two-body Jastrow. `pade` uses $v(r) = \frac{r}{2(1+br)}$; `exp` uses $v(r) = \frac{1}{2b}(1-e^{-br})$.
- `-j3, --jastrow-3b-basis-set-type` (`ao|mo|ao-full|ao-small|ao-medium|ao-large|none`, default `none`): Choose J3 basis. `ao` and `ao-full` (synonyms) uncontract the AO basis and keep every distinct primitive Gaussian (nucleus, angular momentum, angular component, exponent) once; a primitive shared by several contracted shells (general contraction, e.g. cc-pVXZ) appears once. `ao-small`, `ao-medium` and `ao-large` keep, for each atom and angular momentum, the number of uncontracted radial shells given in the table below (at most the number of distinct exponents available), spread around the median of the log-exponents. `mo` uses MOs, `none` disables J3.

  | element row | `ao-small` | `ao-medium` | `ao-large` |
  | --- | --- | --- | --- |
  | H, He | 3s | 3s1p | 4s2p1d |
  | Li-Ne | 3s1p | 4s2p1d | 5s3p2d1f |
  | Na-Ar | 4s2p | 5s3p1d | 6s4p3d1f |
  | K and heavier, main group | 4s2p1d | 5s3p2d1f | 6s4p3d2f |
  | K and heavier, transition metals, lanthanides, actinides | 4s2p2d1f | 5s3p3d2f1g | 6s4p4d3f2g1h |

- `--jastrow-dtn` (`none|u|u_chi|u_f|u_chi_f`, default `none`): DTN (Drummond-Towler-Needs) polynomial Jastrow preset; selects which of u (e-e), chi (e-n), f (e-e-n) channels are active. Mutually exclusive with the Pade j1 / j2 / AO j3 family when not `none`.
- `--dtn-C` (int, default 4): DTN cutoff polynomial smoothness order C (>= 3).
- `--dtn-Nu` (int, default 8): DTN u-term polynomial order N_u.
- `--dtn-Nchi` (int, default 8): DTN chi-term polynomial order N_chi.
- `--dtn-NeN` (int, default 4): DTN f-term electron-ion polynomial order N_eN.
- `--dtn-Nee` (int, default 4): DTN f-term electron-electron polynomial order N_ee.
- `--dtn-Lu` (float|None): Molecular-only DTN u-term cutoff (Bohr). Absolute PBC cutoffs are rejected.
- `--dtn-Lchi` (float|None): Molecular-only DTN chi-term cutoff (Bohr, broadcast across species).
- `--dtn-Lf` (float|None): Molecular-only DTN f-term cutoff (Bohr, broadcast across species).
- `--dtn-cutoff-ratio-u` (float|None): Periodic u cutoff divided by `R_ins`; the periodic default is 0.95.
- `--dtn-cutoff-ratio-chi` (float|None): Periodic chi cutoff divided by `R_ins`; the periodic default is 0.95.
- `--dtn-cutoff-ratio-f` (float|None): Periodic f cutoff divided by `R_ins/2`; the periodic default is 0.95.
  The reference length is the Wigner--Seitz inscribed-sphere radius (the QMCPACK
  convention) and 0.95 leaves a 5% margin from the minimum-image seam; the
  comoving cutoff policy keeps the direct stress consistent with the equation
  of state at any ratio.
- `--dtn-cutoff-abs-cap` (float, default 8.0): Auto-cutoff absolute cap (Bohr).
- `--twist-average` (flag, default off): Without tiling, convert each k-point of a periodic multi-k TREXIO file into a twist sector of one combined HDF5 file; the file needs at least two k-points. With tiling, group complete primitive k cosets into supercell sectors, and use the sum of primitive weights for each sector. J3 `ao*` bases use an independent ordinary periodic GTO basis without the Bloch phase, with the source spherical or Cartesian form; one J3 basis and matrix are shared by every twist (for the primitive cell without tiling, the physical supercell with tiling). `-j3 mo` is rejected because its orbitals depend on the twist. Not available with `--ao-conv-to`.
- `--tiling` (one quoted string, default off): Three diagonal factors (`"3 3 3"`) or nine row-major integers (`"2 1 1 1 1 1 3 2 1"`). Group a complete primitive GTO k coset into a physical supercell while retaining local orbital coefficient blocks. With `--twist-average`, convert all complete supercell sectors. J3 `ao*` bases are expanded onto all physical supercell atoms; `-j3 mo` is rejected.
- `--supertwist` (one quoted string, default automatic): Three fractional supercell reciprocal coordinates, for example `"0.2 -0.1 0"`. Requires `--tiling`; selects one complete sector from the input mesh and is exclusive with `--twist-average`. With no selector, exactly one complete sector must be present. See [orbital tiling](notes/orbital_tiling.md).

- `--ao-conv-to` (`cart|sphe|None`, default `None`): Convert AOs after building the Hamiltonian. `cart` converts all AO representations (Geminal and Jastrow three-body) to Cartesian GTOs; `sphe` converts them to real spherical-harmonic GTOs. From a Cartesian source, `sphe` projects the MOs onto the spherical subspace in the AO overlap metric and discards the rest; for example, the s-like r^2 trace of a d shell in a Cartesian SCF is lost. The written wavefunction is then this projection, not the source wavefunction. When the discarded relative S-norm of an occupied MO exceeds `AO_SPHERICAL_PROJECTION_RESIDUAL_THRESHOLD` (1e-8), the command logs a WARNING per spin with the maximum residual and its MO index; the conversion still completes. A non-finite residual (invalid source MOs or overlap) is an error. When omitted (`None`), the original representation from the TREXIO file is kept.

Outputs an HDF5 `Hamiltonian_data` with Jastrow/Geminal embedded.

---

## `hamiltonian` group

### `show-info`
Print metadata from an existing Hamiltonian HDF5/checkpoint.

```
jqmc-tool hamiltonian show-info <hamiltonian_data> [--twist N]
```

A twist-averaged file (written by `jqmc-tool qe --twist-average`, for
example with `-o hamiltonian_twists.h5`) is accepted: the command prints the table of
twists (k points and weights) and then the details of one twist, selected
with `--twist N` (default `0`).

### `to-xyz`
Export nuclear geometry to XYZ (Bohr -> A). A twist-averaged file is also
accepted (all twists share the same geometry).

```
jqmc-tool hamiltonian to-xyz <hamiltonian_data> [-o struct.xyz]
```

### `conv-wf`
Convert wavefunction ansatz inside a Hamiltonian file.

```
jqmc-tool hamiltonian conv-wf <hamiltonian_data> -c {jagp|jagpn} [-o output.h5]
```
- `-c, --convert-to` (**required**): `jagp` converts SD/AGPn to AO-basis AGP;
  `jagpn` converts a supported SD/AGP input to the natural-orbital diagonal
  form.
- `-o, --output` (str, default `hamiltonian_data_conv.h5`): Output filename.

---

## `qe` group

Convert a Quantum ESPRESSO `prefix.save` directory to a jQMC `hamiltonian_data.h5`. Unlike the other groups, `qe` is a single command (no sub-leaves).

```
jqmc-tool qe <qe_save_dir> [options]
```
- `qe_save_dir` (arg): QE `prefix.save` directory containing `data-file-schema.xml` and `wfc<ik>.hdf5`, or its parent directory holding exactly one `*.save`. Every band occupation must be 0 or 1: run QE with `occupations='fixed'` (for `nspin=2`, with an integer `tot_magnetization`). Smeared occupations are rejected; see {ref}`sec:periodic_fixed_electrons` for the consequence for metals. Only spin-restricted (`nspin=1`) and LSDA (`nspin=2`) saves are supported: noncollinear or spin-orbit saves (`noncolin`/`lspinorb`, two-component spinor wavefunctions with `npol=2`) are rejected with an error for every `--kind`.

### Projection / output options

- `-k, --kind` (`pw|bspline|pgto|pgto+pw|pgto+bspline`, default `pgto+pw`): Projection target.
- `-u, --upf` (repeatable `element=path/to/element.upf`): UPF mapping (repeat once per element/species). Default: auto-discovered from `<prefix>.save/data-file-schema.xml`. Every kind takes the analytic ECP from the UPF, so a pseudopotential save (core electrons removed) whose UPFs cannot be found is rejected instead of being converted without its ECP; the pGTO-bearing kinds also seed their basis from it. A UPF map that misses a species is rejected with the missing species named. The converter classifies ECP presence per atom from the parsed UPF record, so mixed all-electron/ECP structures and zero-core ECPs retain distinct Coulomb and J_DTN cusp policies.
- `--twist-average` (flag, default off): Without tiling, convert each QE k-point into a twist sector of one combined HDF5 file. With tiling, group complete primitive k cosets into supercell sectors, and use the sum of primitive weights for each sector. Sector weights are normalized for averaging. J3 `ao*` bases are built from the UPF exponent seeds as an independent ordinary periodic Cartesian GTO basis without the Bloch phase, for every `--kind`; one J3 basis and matrix are shared by every twist (for the primitive cell without tiling, the physical supercell with tiling). `-j3 mo` is rejected because its orbitals depend on the twist. Compatibility: every spin treatment stores the down-spin channel at the opposite twist (the conjugate of its own `+k` orbitals). Spin-polarized (LSDA / UHF) complex-twist Hamiltonians and twist-average files that keep both spins at one twist are refused at load time; regenerate them from the QE or TREXIO source with the current converter.
- `--tiling` (one quoted string, default off): Three diagonal factors (`"3 3 3"`) or nine row-major integers (`"2 1 1 1 1 1 3 2 1"`). Retain primitive k blocks for `pw`, `bspline`, `pgto` or `pgto+pw` in a physical supercell. Requires complete unreduced k cosets, including for Gamma. Combine with `--twist-average` to convert multiple complete supercell sectors. J3 `ao*` bases use UPF exponent seeds on every supercell atom; `-j3 mo` is rejected.
- `--supertwist` (one quoted string, default automatic): Select one fractional supercell twist, for example `"0.2 -0.1 0"`. Requires `--tiling` and is exclusive with `--twist-average`; omission infers the only complete sector. See [orbital tiling](notes/orbital_tiling.md) for the supported inputs, stored conventions and restrictions.
- `--mesh-factor` (float, default 0.75): B-spline grid size relative to the QE FFT grid (`round(mesh_factor * fft_grid)` points per axis); the spline table grows as `mesh_factor^3`. The default keeps the spline Nyquist frequency at 1.5 times the plane-wave cutoff. A value of 0.5 puts it exactly at the cutoff: the fitting residual stays small, but the kinetic energy of the determinant is biased. Below 0.625 the converter prints a warning; in that case converge the VMC energy **and** variance with respect to `mesh_factor` before trusting the grid, because the fitting residual alone does not certify it.
- `--spline-order` (int, default 5): B-spline polynomial degree: `5` (quintic, recommended for forces) or `3` (cubic).
- `--orthonormalize/--no-orthonormalize` (flag, default on): `kind=pgto` only. Orthonormalize the projected MOs of every (k, spin) block in the periodic AO metric by band-order Cholesky, which keeps the span of every lowest-N band subset (a determinant changes only by a constant). Ignored for the other kinds. All kinds emit MOs normalized over the unit cell.
- `--ecutwfc` (str|None): PW handoff cutoff. Numeric value in Rydberg or the sentinel `auto` (pgto+pw only). Omit for the kind default. With `auto` the converter raises the cutoff along a ladder until the occupation-weighted mean projection residual is at most `2e-5`, up to the smaller of the QE wavefunction cutoff and 40 Ry. If no trial reaches the target, conversion stops with an error and asks for a numeric `--ecutwfc` above 40 Ry. The automatic target and cap are fixed; use a numeric `--ecutwfc` for a different handoff.
- `--alpha-min` (float|None): Diffuse-pGTO exponent floor. Omit for the kind default.
- `--exp-tail-eps` (float, default `1e-2`): Exponent-tail bound of the `pgto+pw` exponent optimizer: no primitive may place more than this fraction of its momentum-space norm outside the QE wavefunction sphere (`alpha <= alpha_max(l, eps)`). `0` disables the bound.
- `--max-cond-number` (float|None): Upper bound on cond(S) for the pGTO fit.
- `-o, --output` (str, default `hamiltonian_data.h5`): Output HDF5 file.

### Jastrow options (Pade family)

- `-j1, --jastrow-1b-parameter` (float|None): Jastrow one-body parameter.
- `--jastrow-1b-type` (`exp|pade`, default `exp`): Jastrow one-body form.
- `-j2, --jastrow-2b-parameter` (float|None): Jastrow two-body parameter. Omit to disable; it must be omitted when `--jastrow-dtn != none`.
- `--jastrow-2b-type` (`pade|exp`, default `pade`): Jastrow two-body form.
- `-j3, --jastrow-3b-basis-set-type` (`none|ao|ao-small|ao-medium|ao-large|ao-full|mo`, default `none`): Jastrow three-body basis.

### Jastrow options (DTN family)

- `--jastrow-dtn` (`none|u|u_chi|u_f|u_chi_f`, default `none`): DTN preset. Mutually exclusive with j1/j2/j3.
- `--dtn-C` (int, default 4): DTN cutoff smoothness order C.
- `--dtn-Nu` (int, default 8): DTN u-term polynomial order.
- `--dtn-Nchi` (int, default 8): DTN chi-term polynomial order.
- `--dtn-NeN` (int, default 4): DTN f-term electron-ion polynomial order.
- `--dtn-Nee` (int, default 4): DTN f-term electron-electron polynomial order.
- `--dtn-cutoff-ratio-u` (float, default 0.95): u cutoff divided by `R_ins`.
- `--dtn-cutoff-ratio-chi` (float, default 0.95): chi cutoff divided by `R_ins`.
- `--dtn-cutoff-ratio-f` (float, default 0.95): f cutoff divided by `R_ins/2`.
  The reference length is the Wigner--Seitz inscribed-sphere radius (the QMCPACK
  convention) and 0.95 leaves a 5% margin from the minimum-image seam; the
  comoving cutoff policy keeps the direct stress consistent with the equation
  of state at any ratio.

### Misc

- `--ewald-tol` (float, default 1e-10): Convergence tolerance forwarded to `build_ewald_data`.
- `--ecp-z-core` (float, repeatable): Per-atom core-electron count; repeat the option once per atom.
- `--verbosity` (str|None): Logger verbosity: `low`, `high`, `devel`, `mpi-low`, `mpi-high`, `mpi-devel`, `warning` or `error`.

---

## `vmc` group (pre/post utilities)

### `generate-input`
Emit a VMC `toml` template (sets `control.job_type = "vmc"`).

The template is a starting point, not a runnable input. Required keys
without a default (`num_mcmc_steps`, `num_opt_steps`) are written as the
string `"None"` and must be replaced by integers; review the warm-up and bin
settings as well (see {doc}`api_reference_cli`).

```
jqmc-tool vmc generate-input --generate [--filename vmc.toml] [--without-comment]
```
- `-g, --generate` (flag): Generate the template. Without this flag the command only prints a reminder.
- `-f, --filename` (str, default `vmc.toml`): Output filename.
- `-nc, --without-comment` (flag): Omit explanatory inline comments.

### `analyze-output`
Parse VMC optimization logs (energies, |f|, signal-to-noise) and optionally plot.

```
jqmc-tool vmc analyze-output <filenames...> [-p] [-s file]
```
- `-p, --plot_graph` (flag): Open an interactive matplotlib window summarising the result. On a headless machine (non-interactive backend such as `MPLBACKEND=Agg`) the window is skipped with a message; use `-s` to save the figure instead.
- `-s, --save-graph` (str|None): Specify a graph filename.

Pass the standard-output files of one or more `vmc` runs (for example a run
and its restarts, in order). The command prints one row per optimization
step with the columns `Iter`, `E (Ha)`, `Max generalized force (Ha/a.u.)` and
`Max force S/N`. The energy should decrease and then fluctuate around a
plateau; the largest generalized force should fall to the size of its own
error bar, which shows up as a small signal-to-noise ratio. Take the
`hamiltonian_data_opt_step_N.h5` for the following sampling runs from the
plateau.

Each value stays on the row of the `Optimization step` header it follows. A
step without its energy, force or signal-to-noise line (a step that did not
finish) is shown with `-` and named on stderr. A non-finite energy or force
(a diverged optimization) is shown as `nan`, named on stderr, and the command
exits with status 1 without saving a graph; so does input that contains no
optimization step.

### `average-wf`
Average the wavefunction parameters of several optimization steps of one VMC
optimization with uniform weights (TurboRVB style), for example the last 20
steps of the plateau found with `analyze-output`. Why and when to average, the
MO and twist treatment and the limitations are explained in the technical note
({ref}`subsec:opt_step_averaging`); the tutorials call it through
`examples/copy_optimized_hamiltonian.py --average-last 20`, and
`jqmc_workflow.VMC_Workflow` runs it on the last 20 steps by default
(`handoff_mode`, `handoff_average_steps`; see the workflow notes).

```
jqmc-tool vmc average-wf <hamiltonian_data_opt_step_*.h5 ...> [-o hamiltonian_data_averaged.h5]
```
- `-o, --output` (str, default `hamiltonian_data_averaged.h5`): Output Hamiltonian file. It is written in the
  format of the inputs: single-twist files give a single-twist file, twist-averaged (combined layout) files give a
  combined file. It must not be one of the inputs; an existing file is replaced.

Pass the files explicitly (the shell may expand a glob); which steps to average,
and what to do around an optimizer rollback, is the caller's choice. Each file
enters the mean once.

What is averaged. Only the variational parameters that differ between the
inputs: J1/J2 parameters, the J3 matrix, the DTN independent coefficients, the
AGP `lambda_matrix`, the AGPn `lambda_diag`/`lambda_unpaired`, the determinant MO
coefficients (per twist for combined inputs) and the radial exponents/contraction
coefficients of the J3 and geminal AO bases. Each becomes the arithmetic mean of
the inputs, accumulated in float64/complex128 and stored in the input dtype.
Elements equal in every input, and every other leaf of the file (structure,
basis, ECP, k-points and weights, parameters that were not optimized), are copied
bitwise from the first input; with a single input the output equals it bitwise.
The averaged MO coefficients are not re-orthonormalized: a determinant needs only
the coefficients, and for a single determinant a row re-orthonormalization would
only rescale Psi by a constant. The reload may therefore log the MO
orthonormality warning of `hamiltonian show-info`; the next optimization step
restores orthonormality through its retraction.

Same optimization. The files must have the same layout and the same HDF5 tree
(groups, datasets, attributes, shapes and dtypes) and be bitwise identical outside
the variational parameters. Otherwise the command prints `Error: ...` on stderr,
exits with status 1 and writes nothing (an existing output is left untouched).
This refuses a different atomic structure, basis set, ECP, ansatz (for example SD
against AGPn), set of Jastrow channels, twist count, k-point or weight, a different
precision mode (stored dtypes), and a non-finite parameter (also one that is equal in
every input, and in a single input). The `opt_*` flags are
not stored in the files. A scalar J1/J2 parameter that was never updated is
stored as an attribute instead of a dataset, so mixing an optimization that
optimized it with one that froze it is refused. An array block frozen for some
steps cannot be told apart from steps of an optimization that did not optimize it:
when one averaged parameter is identical in two inputs while another differs, the
command warns on stderr and still writes the average. It also warns when two
inputs hold identical parameters, and when the inputs were written by different
jQMC builds.

MO gauge. The optimizer moves the MO coefficients continuously (a tangent step
followed by the S-metric polar retraction), so consecutive steps share the sign
and phase of every orbital. A sign or phase jump of an orbital row between the
inputs (for example files of different conversions) is refused: the row
coherence `|sum_i c_ik| / sum_i |c_ik|` must be at least
`OPT_STEP_AVERAGE_MIN_MO_ROW_COHERENCE` (0.99, `jqmc/_setting.py`); the smallest
value is printed.

Provenance. The output carries root attributes that the loaders ignore:
`opt_step_average_sources` (the input paths in the given order),
`opt_step_average_steps` (the `N` of `..._opt_step_N.h5`, `-1` when absent),
`opt_step_average_num_sources`, `opt_step_average_weighting` (`uniform`),
`opt_step_average_leaves` (the averaged HDF5 paths, `group@name` for an
attribute) and `opt_step_average_jqmc_version` / `opt_step_average_jqmc_git_commit`.
The file is written to a temporary name, reloaded with the loaders a `jqmc` run
uses (every twist for combined inputs), and then moved into place.

The comparison and the mean stream each dataset in blocks of about 16 MiB per
input, so memory does not grow with the number of files.

---

## Analysis settings are explicit

Postprocessing reads the checkpoint, **not the original TOML**. Use explicit
options even when a synopsis shows them as optional. For example, after a
sufficiently long run (these are starting points, not convergence proofs):

```bash
jqmc-tool mcmc compute-energy restart.h5 -b 10 -w 30
jqmc-tool lrdmc compute-energy restart.h5 -b 10 -w 25 -c 5
```

`-b` is a **number of time bins**, not the number of measurements per bin.
For MCMC the total delete-bin replicas are `b * ranks * walkers`; LRDMC
first combines the branching-correlated population and uses `b` global
time bins. After the warm-up and, for LRDMC, the weight history (`-c`, the
number of past steps whose branching weights enter each measurement's
correcting factor), retain enough measurements for every bin to be nonempty. At least two positive-weight
replicas are needed for a jackknife error; more are needed for a useful
uncertainty estimate. Check stability against warmup, bin count, run length,
and LRDMC history length. How to choose `-b` (a scan from long to short time
blocks, keeping the largest bin count inside the plateau of the error bar),
how uncertain the resulting error bar is, and when to trust it are described
in {ref}`sec:error_analysis_bins`.

The recommended MCMC minima (30 warm-up measurements, 10 bins) are not
enforced and produce no warning. `mcmc compute-energy`, and `mcmc compute-force` and
`lrdmc compute-force` on a single checkpoint file, refuse (non-zero exit status,
no result printed) settings that leave a bin empty, give a non-positive or
non-finite delete-one-bin weight, or produce a non-finite result, for example
from a non-finite sample in the analysed window. The option defaults
`-b 1 -w 0` are below the recommended minima, so always pass `-b` and `-w`. The LRDMC commands reject
warm-up below 25, fewer than 10 bins and a weight history shorter than 5;
their defaults `-w 25 -b 10 -c 5` are these minima, which are not production
recommendations. A successful exit status is not a proof of
scientific validity: check the stability of the estimate as described above.

## `mcmc` group (pre/post utilities)

### `compute-energy`
Jackknife estimator of VMC energy from an MCMC restart archive.

```
jqmc-tool mcmc compute-energy <restart_chk> [-b N] [-w W]
```
- `restart_chk` (arg): a single-twist checkpoint file, or a twist-averaged checkpoint directory. For a directory, the per-twist energies are combined with the BZ weights stored in the checkpoint.
- `-b, --num_mcmc_bin_blocks` (int, default 1): Binning blocks per MPI x walker; total blocks = `b * mpi_size * walkers`. Choosing it: {ref}`sec:error_analysis_bins`.
- `-w, --num_mcmc_warmup_steps` (int, default 0): Discarded warmup measurements.

### `compute-force`
Jackknife estimator of VMC atomic forces (Hellmann-Feynman + Pulay) from an MCMC restart archive.  Requires that the MCMC run was performed with `atomic_force = true`.

```
jqmc-tool mcmc compute-force <restart_chk> [-b N] [-w W]
```
- `-b, --num_mcmc_bin_blocks` (int, default 1): Binning blocks per MPI x walker; total blocks = `b * mpi_size * walkers`. Choosing it: {ref}`sec:error_analysis_bins`.
- `-w, --num_mcmc_warmup_steps` (int, default 0): Discarded warmup measurements.

Outputs a per-atom force table in Ha/bohr with jackknife error bars.
For a twist-sharded checkpoint directory, this command recomputes each twist's force jackknife from that twist's stored samples with the given `-b`/`-w` (with the same refusals as for a single file) and combines the per-twist forces with the embedded BZ weights. The hydrostatic pressure combines the per-twist pressure summaries that the run stored at its own warmup/bin settings (`-b`/`-w` do not apply to it, and the output says so); a non-finite stored pressure summary is refused. Directories without embedded weight/count metadata are not supported. A merged single-checkpoint stress post-processor is not provided; its stress tensor remains available in the original `jqmc` stdout.

### `generate-input`
Emit an MCMC `toml` template (sets `control.job_type = "mcmc"`).

As for `vmc generate-input`, replace `num_mcmc_steps = "None"` by an integer
and review the warm-up and bin settings before running.

```
jqmc-tool mcmc generate-input --generate [--filename mcmc.toml] [--without-comment]
```
- `-g, --generate` (flag): Generate the template. Without this flag the command only prints a reminder.
- `-f, --filename` (str, default `mcmc.toml`): Output filename.
- `-nc, --without-comment` (flag): Omit explanatory inline comments.

---

## `lrdmc` group (pre/post utilities)

### `compute-energy`
Jackknife estimator of LRDMC energy from an LRDMC restart archive.

```
jqmc-tool lrdmc compute-energy <restart_chk> [-b N] [-w W] [-c C]
```
- `restart_chk` (arg): a single-twist checkpoint file, or a twist-averaged checkpoint directory. For a directory, the per-twist energies are combined with the BZ weights stored in the checkpoint.
- `-b, --num_gfmc_bin_blocks` (int, default 10, the minimum): Global time-bin count (`b`, not multiplied by ranks x walkers). Must be >= 10; the minimum is not a production recommendation, see {ref}`sec:error_analysis_bins`.
- `-w, --num_gfmc_warmup_steps` (int, default 25, the minimum): Discarded warmup steps; must be >= 25.
- `-c, --num_gfmc_collect_steps` (int, default 5): Length of the weight history used for the correcting factors; must be >= 5.

### `compute-force`
Jackknife estimator of LRDMC atomic forces (Hellmann-Feynman + Pulay) from an LRDMC restart archive.  Requires that the LRDMC run was performed with `atomic_force = true`.

```
jqmc-tool lrdmc compute-force <restart_chk> [-b N] [-w W] [-c C]
```
- `-b, --num_gfmc_bin_blocks` (int, default 10, the minimum): Global time-bin count. Must be >= 10; see {ref}`sec:error_analysis_bins`.
- `-w, --num_gfmc_warmup_steps` (int, default 25, the minimum): Discarded warmup steps; must be >= 25.
- `-c, --num_gfmc_collect_steps` (int, default 5): Length of the weight history used for the correcting factors; must be >= 5.

Outputs a per-atom force table in Ha/bohr with jackknife error bars.
For a twist-sharded checkpoint directory, the force of each twist is recomputed with the given `-b`/`-w`/`-c` and the pressure follows the same stored-summary convention as MCMC. A merged single-checkpoint stress post-processor is not provided; both LRDMC drivers still emit their stress tensor to the original `jqmc` stdout.

### `extrapolate-energy`
Fit energy vs $a^2$ from multiple LRDMC restart archives and extrapolate $a\to 0$.

Pass one checkpoint **file** per lattice spacing for single-twist runs, or one
complete checkpoint **directory** per spacing for twist-averaged runs. For a
directory, the command first repeats `compute-energy`'s per-twist jackknife
with the requested binning, warmup, and weight-history length, then combines
means and independent-twist errors with the embedded normalized BZ weights.
It fits these BZ-averaged points, not individual twist files pooled together.
Complex histories use the real part of the energy estimator.

```
jqmc-tool lrdmc extrapolate-energy <restart_chks...> [-p order] [-b N] [-w W] [-c C] [-g] [-s file]
```
- `-p, --polynomial-order` (int, default 2): Fit $E(a^2) = E_0 + a^2 E_2 + a^4 E_4 + \dots$.
- `-g, --plot-graph` (flag): Open an interactive matplotlib window summarising the result. On a headless machine (non-interactive backend such as `MPLBACKEND=Agg`) the window is skipped with a message; use `-s` to save the figure instead.
- `-s, --save-graph` (str|None): Specify a graph filename.
- `-b, --num_gfmc_bin_blocks`, `-w, --num_gfmc_warmup_steps`, `-c, --num_gfmc_collect_steps`: as in `compute-energy`.

For example, with complete multi-twist checkpoints at three spacings:

```bash
jqmc-tool lrdmc extrapolate-energy \
    alat_0.20/restart_twists alat_0.30/restart_twists alat_0.40/restart_twists \
    -p 2 -b 50 -w 50 -c 20 -s lrdmc_extrapolation.png
```

All directories must contain the same k points and normalized weights;
twist indices may be reordered. Each directory must have one common positive
finite `alat` across all ranks and twists. Missing identity/weight metadata,
incomplete or unconverged twist sets, mixed file/directory arguments, and
duplicate spacings are rejected. Order `p` requires at least `p + 1` distinct
$a^2$ points; for two spacings, choose `-p 1` explicitly. Use independently
sampled runs of the same physical system and trial for the scan; matching
twist metadata alone does not establish these physical conditions.

The command always fits the **uncorrected energy `E`**. If MPC observables
are present, it also fits **`E_MPC = E + dE_MPC`**, printing both series at
each spacing and at $a\to0$; a saved graph includes both. No extra option is
needed. The `E_MPC` error uses the paired jackknife of energy and correction,
including their covariance, before BZ averaging and extrapolation. It is
not the quadrature sum of their separate errors.

MPC histories must be present for every twist and spacing and for every MPI
rank that contributes energy samples; empty ranks in rank-0-reduced files
need no histories. Missing/partial histories, conflicting enable flags,
malformed samples, and inconsistent cutoffs are rejected, not silently
treated as an uncorrected-only scan. Every rank's `driver_config` must store
the same positive finite `mpc_cutoff` (Ha); any twist-summary cutoff must
agree. MPC files without cutoff metadata cannot establish this
consistency and are rejected. A stored default cutoff with `mpc = false`
and no MPC histories does not enable `E_MPC`.

Neither forces nor stress are extrapolated. The fit is an unweighted
least-squares polynomial in `a**2`, so the extrapolated value is a fixed
linear combination of the input energies and its error bar is the exact
propagation of the (independent) input error bars through that combination.
The result is deterministic: rerunning on the same checkpoints prints the
same numbers. Error bars describe this statistical propagation for the
selected polynomial fit, not its truncation/model error or the covariance
between the two fitted series.
Check stability against the spacing range and polynomial order before
reporting a continuum limit; also converge the MPC cutoff separately.

### `generate-input`
Emit an LRDMC `toml` template (sets `control.job_type = "lrdmc-bra"`).

Replace `num_mcmc_steps = "None"` and `num_mcmc_per_measurement = "None"`
by integers; both are required and have no default. `num_mcmc_per_measurement`
is the number of GFMC projections per measurement, which sets the imaginary
time per measurement (about `num_mcmc_per_measurement * alat^2 / (3 N_e)`).
The template leaves out `E_scf` so that the same file can later be used for a
restart; for a fresh run add `E_scf` under `[lrdmc-bra]` (for example the VMC
energy in Ha). There is no `lrdmc-tau` template: to use the fixed-`tau`
driver, change `job_type` to `"lrdmc-tau"`, rename the section to
`[lrdmc-tau]` and add `tau`; `lrdmc-tau` has no `E_scf` or
`num_mcmc_per_measurement` key (see {doc}`api_reference_cli`).

```
jqmc-tool lrdmc generate-input --generate [--filename lrdmc.toml] [--without-comment]
```
- `-g, --generate` (flag): Generate the template. Without this flag the command only prints a reminder.
- `-f, --filename` (str, default `lrdmc.toml`): Output filename.
- `-nc, --without-comment` (flag): Omit explanatory inline comments.
