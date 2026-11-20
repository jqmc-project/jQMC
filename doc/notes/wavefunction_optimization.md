(sec:opt_wf)=

# Wave function optimization
## Why a plain gradient descent is not enough

For a real parameter, let $f_k=-\partial E/\partial\alpha_k$; for a
complex parameter, let $f_k=-2\partial E/\partial\bar c_k$.  These
forces are derived in {ref}`sec:derivatives`; note in particular
{ref}`subsec:force_local_energy_derivative`: the covariance-only
estimator of $f_k$ is unbiased **only under exact
$|\Psi|^{2}$ sampling**.  The SR force is this covariance-only
estimator, whereas the linear method keeps the finite
$\langle\partial_{c_k} e_L\rangle$ through the non-symmetric
pencil described below.  The most direct
optimization strategy is steepest descent,

$$
\alpha'_k \leftarrow \alpha_k + \delta \alpha_k,
\qquad
\delta \alpha_k = \Delta\,f_k,
$$ (steepest_descent)

with a small step $\Delta$ (the learning rate `optimizer_kwargs.delta`). This works for very loosely-coupled
parameters but fails as soon as the wave function depends
non-linearly on $\boldsymbol{\alpha}$: a small change in one
parameter may produce a very different $\Psi$ while another
parameter has almost no effect on $\Psi$, and a scalar $\Delta$
cannot match both scales at once. Off-the-shelf Newton-Raphson,
conjugate-gradient, or quasi-Newton schemes do not transfer
straightforwardly either, because the matrix elements they consume
are themselves Monte Carlo estimates with statistical noise that
deteriorates rapidly along ill-conditioned directions.

jQMC therefore implements two QMC-tailored optimizers:

- **Stochastic Reconfiguration (SR)** -- a first-order method that
  preconditions the gradient by the wave-function metric, a.k.a.
  natural gradient;
- **Linear Method (LM)** -- a second-order method that solves a
  generalized eigenvalue problem for the energy-minimizing
  linear combination of the trial state and its parameter
  derivatives.

They are combined into four optimizer modes, all selected through
`[vmc] optimizer_kwargs`: plain SR, accelerated SR (aSR, an SR direction
whose length is chosen by a one-dimensional LM model,
sec. {ref}`subsec:practical_asr`), reduced LM (an LM pencil on a few
selected parameters plus the SR direction,
sec. {ref}`subsec:lm_sr_collective`), and full LM on all parameters
(sec. {ref}`subsec:linear_method`).

(subsec:choosing_optimizer)=

## Choosing optimizer settings

A VMC optimization is run with `job_type = "vmc"` and a `[vmc]` section.
The optimizer is configured by the `optimizer_kwargs` table, for example

```toml
[vmc]
num_mcmc_steps = 1000
num_opt_steps = 100
optimizer_kwargs = { method = "sr", delta = 0.15, epsilon = 0.001, use_lm = true, lm_subspace_dim = 0 }
```

A partial table is merged onto the defaults below: keys that are not
named keep their default values. The full list of `[vmc]` keys is given
in {doc}`../api_reference_cli`. A value outside the domain stated in the
table stops the run with a `ValueError` before any sampling.

| Key | Default | Symbol in this note | Meaning |
|---|---|---|---|
| `method` | `"sr"` | -- | the only supported optimizer family |
| `delta` | `0.15` | $\Delta$ | learning rate; the final parameter displacement is `delta` times the SR, aSR or LM step in every mode |
| `epsilon` | `0.001` | $\varepsilon_{\rm SR}$ | diagonal shift of the scale-normalized SR metric, eq. {eq}`regularization_s`; finite and $\ge 0$ |
| `cg_flag` | `true` | -- | solve the SR equation with matrix-free conjugate gradients (`false`: direct dense solve) |
| `cg_max_iter` | `10000` | -- | maximum number of CG iterations; an integer $\ge 1$ |
| `cg_tol` | `1e-5` | -- | relative CG residual tolerance: CG stops once $\lVert r\rVert \le$ `cg_tol` $\lVert b\rVert$ for the right-hand side $b$ of the solved SR system; finite and $> 0$ |
| `use_lm` | `true` | -- | `false`: plain SR; `true`: aSR or LM, selected by `lm_subspace_dim` |
| `lm_subspace_dim` | `0` | $m$ | `0`: aSR; `> 0`: reduced LM on the $m$ coordinates with the largest force signal-to-noise ratio plus the SR direction; `-1`: full LM on all coordinates; other values are rejected |
| `lm_cond` | `0.001` | -- | smallest eigenvalue kept in the correlation matrix of the LM tangent metric (sec. {ref}`subsec:lm_stabilization`) |
| `asr_gamma_max` | `0.5` | $\gamma_{\max}$ | largest accepted aSR step scale $\lvert\gamma\rvert$ |
| `asr_fallback_gamma` | `0.1` | $\gamma_{\rm fb}$ | step scale used when the aSR model is rejected |
| `lm_reference_weight_min` | `0.9` | -- | minimum reference weight $w_0^2$ of an accepted LM eigenvector (reduced and full LM only) |

How to choose a mode:

- **aSR** (`use_lm = true`, `lm_subspace_dim = 0`, the default) is the
  recommended starting point. It uses the SR direction and chooses its
  length from the one-dimensional LM energy model at each step, at the cost
  of also computing the parameter derivatives of the local energy.
- **Plain SR** (`use_lm = false`) is the cheapest per step, because no
  local-energy derivatives are needed. The step length is fixed by `delta`,
  so a smaller `delta` is safer and a larger one faster but less stable.
- **Reduced LM** (`lm_subspace_dim = m > 0`) adds individual second-order
  steps for the $m$ statistically most significant coordinates. The pencil
  has dimension of about $m+2$, so its cost is small for moderate $m$.
- **Full LM** (`lm_subspace_dim = -1`) builds and diagonalizes the dense
  $(Q+1)\times(Q+1)$ pencil over all $Q$ real coordinates. Use it only
  when the number of optimized parameters is small.

If an LM candidate is rejected (sec. {ref}`subsec:lm_stabilization`),
the step falls back to aSR without additional sampling.

Which parameters are optimized is controlled by the `[vmc]` flags:

| `[vmc]` flag | Default | Optimized parameters |
|---|---|---|
| `opt_J1_param` | `true` | J1 parameter $b_{\rm ei}$ |
| `opt_J2_param` | `true` | J2 parameter $b_{\rm ee}$ |
| `opt_J3_param` | `true` | J1 inhomogeneous and J3/J4 matrix |
| `opt_geminal` | `false` | determinant part: `lambda_matrix` (AGP), MO coefficients and `lambda_diag` (AGPn), MO coefficients (SD) |
| `opt_J3_basis_exp`, `opt_J3_basis_coeff` | `false` | exponents / contraction coefficients of the Jastrow AO basis |
| `opt_lambda_basis_exp`, `opt_lambda_basis_coeff` | `false` | exponents / contraction coefficients of the determinant AO basis |
| `opt_JDTN_u_param`, `opt_JDTN_chi_param`, `opt_JDTN_f_param` | `true` | DTN u, chi and f polynomial coefficients |

`fix_geminal_rank` (default `0`, meaning "not set") selects the
rank-constrained geminal optimization of
sec. {ref}`subsec:geminal_optimizer_contract`.

The plain `jqmc` program always runs `num_opt_steps` optimization steps;
it has no automatic stopping rule. Each step prints the energy and the
maximum force signal-to-noise ratio (sec. {ref}`subsec:sr_signal_to_noise`),
which `jqmc-tool vmc analyze-output` collects into a table. Automatic
convergence checks based on this ratio are provided by the
`jqmc_workflow` VMC workflow (`target_snr`, see {doc}`workflows`).
Once the energy has reached its plateau, the wave function handed to
MCMC or LRDMC is the uniform average of the parameters of the last steps
(sec. {ref}`subsec:opt_step_averaging`).

(subsec:opt_step_averaging)=

## Averaging the last optimization steps

### Why average

Each optimization step estimates the forces from a finite sample, so on the
energy plateau the parameters do not settle at the optimum
$\boldsymbol{\alpha}^\ast$; they fluctuate around it,
$\boldsymbol{\alpha}_k = \boldsymbol{\alpha}^\ast + \boldsymbol{\delta}_k$,
with a noise $\boldsymbol{\delta}_k$ set by the sampling and the step size.
The last step is one draw of this noise. Near the minimum the energy is
quadratic in the deviation, $E(\boldsymbol{\alpha}) \simeq E^\ast +
\tfrac12 \boldsymbol{\delta}^{\mathsf T} H \boldsymbol{\delta}$, so the
expected excess energy of the adopted parameters is
$\tfrac12 \operatorname{tr}(H\,\operatorname{Cov}\boldsymbol{\delta})$.
The mean of the last $N$ steps,
$\bar{\boldsymbol{\alpha}} = N^{-1}\sum_k \boldsymbol{\alpha}_k$, has the
covariance $\operatorname{Cov}\boldsymbol{\delta}$ reduced by about
$N/(2\tau)$, where $\tau$ is the integrated autocorrelation time of the
parameters in optimization steps ($\tau = 1/2$ for uncorrelated steps;
consecutive steps are correlated because each step only moves the previous
parameters), valid for $N \gg \tau$. Averaging therefore lowers both
the excess energy and the run-to-run spread of the wave function that the
fixed-node LRDMC inherits. TurboRVB adopts the parameters averaged over the
last optimization iterations in the same way
([Nakano *et al.*, J. Chem. Phys. 152, 204121 (2020)](https://doi.org/10.1063/5.0005037)).
Averaging removes noise, not drift: on a trajectory that still drifts the mean
lags behind the last step.

### When, and how many steps

Average only steps on the plateau. Read it from
`jqmc-tool vmc analyze-output out_vmc [out_vmc_cont ...]` (add `-s vmc.jpg`
for a plot). Over the whole averaging window the `E (Ha)` column must
fluctuate within its error bars without a trend, and `Max force S/N` must stay
of order one, below about 4-5 (sec. {ref}`subsec:sr_signal_to_noise`). A window
that starts before the energy has stopped decreasing biases the average
upward. The tutorials average the last 20 steps of runs of 100-300 steps,
which lie well inside the plateau of each run; more steps reduce the noise
further as long as all of them are on the plateau. Only dumped steps can be
averaged, so the VMC input sets `wf_dump_freq = 1`; the dump frequency does
not change the optimization itself.

### The command and what it averages

```text
jqmc-tool vmc average-wf <hamiltonian_data_opt_step_*.h5 ...> -o hamiltonian_data_averaged.h5
python examples/copy_optimized_hamiltonian.py --average-last 20 <vmc_dir> <destination>
```

The second form, used by the tutorials, selects the 20 dumps with the
largest step numbers and calls the first; it refuses when fewer dumps exist
(`--allow-fewer` accepts them) and when the destination exists.
`jqmc_workflow.VMC_Workflow` hands the same average to its downstream stages
by default (`handoff_mode`, `handoff_average_steps`, `handoff_allow_fewer`). The full option
and error reference is in {doc}`../api_reference_cli_tool` (`vmc average-wf`).
In short:

- The weights are uniform. Every variational parameter that differs between the
  inputs becomes the arithmetic mean of the inputs, accumulated in
  float64/complex128 and stored in the input dtype: J1/J2, the J3 matrix, the
  DTN independent coefficients, the AGP `lambda_matrix`, the AGPn
  `lambda_diag`/`lambda_unpaired`, the determinant MO coefficients and the
  J3/geminal AO radial exponents and contraction coefficients.
- The output is a byte copy of the first input in which only those leaves are
  replaced. Elements equal in every input, and everything else (structure,
  basis, ECP, k-points and weights, parameters that were not optimized), keep
  the first input's bits; one input gives a bitwise copy.
- "Same optimization" is defined by the file content: the same layout, the
  same HDF5 tree (groups, datasets, attributes, shapes, dtypes) and identical
  values outside the variational parameters. A different structure, basis,
  ECP, ansatz, set of Jastrow channels, twist set or precision mode (stored
  dtype), a non-finite parameter (changed or not, also with a single input), a
  duplicated input or an output that is an
  input is refused with exit status 1, and nothing is written. The `opt_*`
  flags are not stored; a never-updated scalar J1/J2 is stored as an attribute
  instead of a dataset, so mixing an optimization that optimized it with one
  that froze it is refused as a structural mismatch.
- Warnings, not refusals: two inputs with identical parameters, one parameter
  identical in two inputs while another differs (a block frozen for some
  steps, for example by the SR diagonal floor, or steps of optimizations with
  different optimized parameters), and inputs written by different jQMC
  builds.
- Root attributes record the provenance: the input paths, their step numbers,
  their count, the uniform weighting, the averaged leaves and the jQMC
  version and commit of the tool. The loaders ignore them. The output is
  reloaded with the loaders a `jqmc` run uses before it replaces the
  destination.

### MO coefficients without re-orthonormalization

The optimizer keeps the occupied rows $C$ orthonormal in the AO overlap
metric $S$, $C^\ast S C^{\mathsf T} = I$
(sec. {ref}`sec:sd_occupied_orbitals`). The mean $\bar C$ of nearby steps
deviates from orthonormality only at second order in their spread. The
determinant needs only the coefficients: for any invertible matrix $G$ acting
on the occupied rows, $\det(G\bar C\,\phi) = \det G\,\det(\bar C\,\phi)$,
so re-orthonormalizing a single determinant would only multiply $\Psi$ by a
constant, which no QMC estimator sees. For AGPn the rows are paired with
`lambda_diag`, and re-orthonormalizing them alone would change the geminal;
the average is therefore kept as the point of parameter space it is. A
further optimization started from the average restores the orthonormality at
its first step. The load
may log the MO orthonormality diagnostic of `Hamiltonian_data.sanity_check`;
this has no physical effect for a single determinant.

### Complex twists: phase and coherence

The sign of a real MO row and the phase of a complex one are a gauge: they
change $\Psi$ by a constant. Within one optimization they do not jump,
because the MO update is continuous (no QR or eigenvector sign choice is
made). Files from different conversions can differ by
such a gauge, and averaging over a jump cancels the orbital instead of
averaging it. The command therefore computes the row coherence
$\rho_k = |\sum_i \mathbf c_{ik}| / \sum_i |\mathbf c_{ik}|$ of every
determinant MO row $k$ over the inputs $i$ (per twist for combined inputs) and
refuses the inputs when $\min_k \rho_k$ is below
`OPT_STEP_AVERAGE_MIN_MO_ROW_COHERENCE` = 0.99 (`jqmc/_setting.py`); the
smallest value is printed. For example, one sign flip among 20 files gives
$\rho = 0.9$ and a $\pi/4$ phase jump in one of 20 gives 0.986. A rotation that mixes nearly degenerate occupied orbitals
while keeping $\rho \ge 0.99$ is not detected; within one optimization the
same continuity argument excludes it.

### Rollbacks and restarts

The command checks only that its inputs belong together; which steps to
average is the caller's choice. If the optimizer rolled back or diverged,
pass only steps after the last rollback, or end the window at a reviewed
step with `copy_optimized_hamiltonian.py --average-last 20 --step S`. A
continuation (`restart = true`) numbers its dumps after the completed steps,
so dumps of a run and its continuation can be averaged together.

### Twist-averaged inputs (combined layout)

The twist-averaged optimizer dumps one combined file per step with the
shared Jastrow under `/shared` and the MO / lambda payload (and AO overrides)
of every twist under `/twists/twist_NNN`. The command averages the shared
Jastrow once and every twist's determinant parameters separately, refuses a
different twist count, k-point or weight, applies the coherence check per
twist, and writes a combined file that loads for every twist. The
time-reversal relation $C_\downarrow = C_\uparrow^\ast$ survives exactly,
because the mean of the conjugates is the conjugate of the mean.

### Cost

The inputs are compared and averaged one dataset at a time in blocks of
about 16 MiB per input, so memory does not grow with the number of files.

### Limitations

- The `opt_*` flags are not stored, so an array block that one set of inputs
  optimized and another froze produces only a warning, and two independent
  optimizations with the same flags and the same start cannot be told apart.
  Passing the steps of one run is the caller's responsibility.
- The weights are uniform. The command gives no error bar for the averaged
  parameters and does not test convergence; a window off the plateau gives a
  biased average.
- The averaged MO coefficients are not re-orthonormalized (see above).
- There is no public Python function; the Python API examples call the
  command-line helper.

(subsec:sr)=

## Stochastic Reconfiguration (natural-gradient optimization)

SR [Sorella, PRB 64, 024512 (2001)](https://doi.org/10.1103/PhysRevB.64.024512) replaces the bare force by its
natural gradient with respect to the wave-function metric.  Complex
wave functions do not by themselves determine whether the SR
unknown is real or complex: that is determined separately for each
variational parameter.  This section therefore fixes both the
sample-space convention and the parameter manifold before writing a
linear system.  The Wirtinger and JAX conventions used below are
defined in {ref}`sec:complex_gradients`.

(subsec:sr_log_derivative)=

### Sample dtype and parameter field are independent

The sample arithmetic is real for a Gamma-point real wave function and
generally complex at a non-Gamma twist.  Independently, each active
variational parameter is declared real or complex.  A real parameter
can therefore have complex walker-wise $O$ and $D=\partial e_L/\partial
p$, while promoting a sample array to `complex128` does not create an
extra parameter degree of freedom.

The optimizer needs the complex directional derivative along every
independent **real** optimizer coordinate.  Three cases occur.

For an independent real parameter $r$, the full complex derivative is used,

$$
R_i=\frac{\partial\ln\Psi(\mathbf{x}_i)}{\partial r}.
$$ (sr_real_parameter_derivative)

Taking only $\partial\operatorname{Re}\ln\Psi/\partial r$ is insufficient at
a twist: it loses the phase derivative and therefore the
$\operatorname{Im}R\,\operatorname{Im}e_L$ contribution to the force.

For an independent complex parameter $p=x+iy$, allow the ansatz to depend on
both $p$ and $\bar p$ and use the two Wirtinger components

$$
A_i=\left.\frac{\partial\ln\Psi(\mathbf{x}_i)}{\partial p}\right|_{\bar p},
\qquad
B_i=\left.\frac{\partial\ln\Psi(\mathbf{x}_i)}{\partial\bar p}\right|_p.
$$ (sr_wirtinger_derivative_pair)

The two real-coordinate columns are

$$
O_x=A+B,
\qquad
O_y=i(A-B).
$$ (sr_wirtinger_to_real_columns)

An ordinary independent holomorphic complex block has $B=0$ and therefore
reduces to the familiar pair $(A,iA)$.  A conjugate tie, such as
$C_\downarrow=\overline{C_\uparrow}$ or a Hermitian off-diagonal geminal
entry, generally has $B\ne0$.  In that case
`jax.grad(real(log(Psi)), p)` returns $A+\overline B$ under the convention in
{ref}`sec:complex_gradients`; that single complex number cannot reconstruct
both $O_x$ and $O_y$.  jQMC therefore keeps $(A,B)$ separately for tied
blocks.

The same applies to the local-energy derivative: the full complex
$D_r=\partial e_L/\partial r$ for a real block, and
$(D_p,D_{\bar p})$ for a complex block, so that
$D_x=D_p+D_{\bar p}$ and $D_y=i(D_p-D_{\bar p})$.

After exact ties have been resolved and constrained blocks have been mapped to their coordinates,
collect the resulting directional derivatives in
$O_q,D_q\in\mathbb C^{N\times Q}$.  Define

$$
(X_q)_{ai}=\frac{\sqrt{w_i}}{\sqrt W}
\bigl((O_q)_{ia}-\bar O_{q,a}\bigr)^*,
\qquad
F_i=-2\frac{\sqrt{w_i}}{\sqrt W}(e_{Li}-\bar e_L),
\qquad W=\sum_iw_i.
$$ (sr_X_F)

The universal coordinate-space covariance and force are

$$
G=\operatorname{Re}(X_qX_q^\dagger),
\qquad
b=\operatorname{Re}(X_qF).
$$ (sr_complex_metric_force)

$F$ remains complex at a non-Gamma twist even when every active
parameter is real.  Taking $\operatorname{Re}F$ before the contraction
would discard valid $\operatorname{Im}O\,\operatorname{Im}e_L$ terms.

(subsec:sr_parameter_manifold)=

### One real-coordinate representation for every case

Split every complex parameter into real and imaginary parts and collect
all optimizer unknowns in one real vector $q\in\mathbb R^Q$.  Let
$\mathcal E$ be the complete real-to-storage **tangent** embedding,

$$
\delta p=\mathcal E q.
$$ (sr_parameter_embedding)

For an unconstrained Euclidean block, a real entry contributes one
unit-vector column and a complex entry contributes the two columns $e_k$
and $ie_k$.  Thus its ambient coordinate count is $P_r+2P_c$, ordered by
block, C-order storage index, and then `(x, y)` for each complex scalar.
For a constrained factor block, the owning ansatz specifies its
coordinates; see {ref}`subsec:geminal_optimizer_contract`.  The resulting global
$\mathcal E$ is a real-linear operator and need not be stored as a dense
matrix.

Exact dependencies such as time reversal are resolved before the
coordinates are formed.  The coordinate order and grouping are fixed for an
optimization run; the coordinates of a constrained block may depend on the
current state and are rebuilt for each step.

The **real-coordinate SR metric** $G\in\mathbb R^{Q\times Q}$ and force
$b\in\mathbb R^Q$ are the sample-first quantities in
eq. {eq}`sr_complex_metric_force`.  Before stabilization, the common SR
equation is

$$
Gq=b.
$$ (sr_mixed_coordinate_system)

By construction, $G$ is real symmetric and positive semidefinite.  It is
the covariance metric seen by the independent real optimizer coordinates
$q$, formed after exact ties, realification and the coordinates of
constrained blocks have produced $O_q$ (not by taking the real part of a storage-coordinate
matrix).

The regularization shift is applied after scale normalization,
eq. {eq}`regularization_s`, and not in the original coordinates.  The same
definition covers:

- all-real: $\mathcal E=I$;
- all-complex and unconstrained: every storage entry contributes an
  $x/y$ pair; and
- mixed: the real columns and complex $x/y$ pairs are simply
  concatenated.

For example, for the storage vector $p=(r,c)$ with
$r\in\mathbb R$ and $c\in\mathbb C$, use

$$
q=(q_r,q_x,q_y)^T,
\qquad
\mathcal E=
\begin{bmatrix}1&0&0\\0&1&i\end{bmatrix},
\qquad
\delta p=(q_r,q_x+iq_y)^T.
$$ (sr_mixed_coordinate_example)

For an unconstrained holomorphic storage force
$f=(f_r,f_c)^T$, the corresponding real right-hand side is

$$
b=\left(\operatorname{Re}f_r,
         \operatorname{Re}f_c,
         \operatorname{Im}f_c\right)^T.
$$ (sr_mixed_force_example)

For a holomorphic Euclidean complex column, the conjugated sample row is
expanded as $(X_c,-iX_c)$.  For a tied block, use
eq. {eq}`sr_wirtinger_to_real_columns` before conjugation; there is no
single $X_c$ from which both rows can be recovered.

The un-conjugated derivative arrays always use the **forward directional
chain rule**, never a Hermitian pullback.  We abbreviate the complete
operation as

$$
O_q=\operatorname{forward}_{\mathcal E}(O_{\rm source}),
\qquad
D_q=\operatorname{forward}_{\mathcal E}(D_{\rm source}).
$$ (optimizer_derivative_realification)

Here `source` is a full complex real-parameter column or a Wirtinger pair as
specified above.  In the special holomorphic Euclidean case this becomes the
ordinary matrix product $(O\mathcal E,D\mathcal E)$ and produces
$(O,iO)$ and $(D,iD)$ for one complex scalar.  For a general real-linear
factor action it denotes the directional derivative along every encoded
tangent column.

The symbol $\mathcal E^{\operatorname{adj}_\mathbb R}$ denotes the adjoint
with respect to real inner products,

$$
\operatorname{Re}\langle u,\mathcal E q\rangle_\mathbb C
=\langle\mathcal E^{\operatorname{adj}_\mathbb R}u,q\rangle_\mathbb R.
$$ (optimizer_real_adjoint_definition)

For a complex-linear matrix acting on complex storage this reduces, after
the outer real projection, to the familiar Hermitian pullback.  Applying a
Hermitian adjoint directly to un-conjugated $O$ or $D$ would be a
transpose-versus-adjoint error.

The implementation therefore uses one real-coordinate, symmetric linear
solver rather than separate real, complex, and mixed solvers.  An
all-complex Hermitian solve would require its own equivalence proof and is
not used by the production optimizer.

The parameter blocks, the `[vmc]` flag that activates each of them, and their scalar field are:

| block | `[vmc]` flag | independent field | notes |
|---|---|---|---|
| `j1_param`, `j2_param` | `opt_J1_param`, `opt_J2_param` | real | one scalar each |
| `j3_matrix` | `opt_J3_param` | real | J1 inhomogeneous column and J3/J4 matrix |
| `j3_poly_alpha`, `j3_poly_beta`, `j3_poly_gamma` | `opt_JDTN_u_param`, `opt_JDTN_chi_param`, `opt_JDTN_f_param` | real | DTN polynomial coefficients (u, chi, f), stored as independent constrained coefficients |
| `j3_basis_exp`, `j3_basis_coeff` | `opt_J3_basis_exp`, `opt_J3_basis_coeff` | real | shell-reduced basis coordinates |
| `lambda_basis_exp`, `lambda_basis_coeff` | `opt_lambda_basis_exp`, `opt_lambda_basis_coeff` | real | shell-reduced basis coordinates |
| `lambda_diag` | `opt_geminal` | real | AGPn only; never exposed by SD |
| `lambda_unpaired` | `opt_geminal` | real at Gamma, complex on the non-Gamma path | AGPn only; canonical SD has no lambda block |
| `lambda_matrix` | `opt_geminal` | real at Gamma, complex on the non-Gamma path | dense `AGP_data` storage without rank truncation; `fix_geminal_rank > 0` converts it to `AGPn_data` before optimization |
| `mo_coefficients_up` | `opt_geminal` | real at Gamma, complex on the non-Gamma path | occupied rows stay orthonormal in the AO overlap metric |
| `mo_coefficients_dn` | `opt_geminal` | same as `mo_coefficients_up` | exposed only when `time_reversal=False` |

When `time_reversal=True`, `mo_coefficients_dn` is not an independent
storage block.  The up derivative is kept as $A$
and the derivative through the tied down leaf as $B$ (not combined into
$A+\overline B$).  The coordinate adapter combines them according to
eq. {eq}`sr_wirtinger_to_real_columns`, while the update reconstructs
$C_\downarrow=\overline{C_\uparrow}$.  Hermitian off-diagonal
`lambda_matrix` entries are treated in the same way.

### Diagonal regularization

The sampled metric frequently becomes ill-conditioned: parameters
with weak influence on $\Psi$ produce small eigenvalues that are
swamped by Monte Carlo noise [Sorella, PRB 71, 241103 (2005)](https://doi.org/10.1103/PhysRevB.71.241103). After
the scale normalization below, jQMC applies the Tikhonov shift

$$
G_{\mathrm{pc}}\longmapsto
G_{\mathrm{pc}}+\varepsilon_{\mathrm{SR}}I.
$$ (regularization_s)

Let $D=\operatorname{diag}(\sqrt{G_{aa}})$ on the surviving coordinates.
The operational equation and its original-coordinate equivalent are

$$
(G_{\mathrm{pc}}+\varepsilon_{\mathrm{SR}}I)q_{\mathrm{pc}}
=b_{\mathrm{pc}},
\qquad
(G+\varepsilon_{\mathrm{SR}}D^2)q=b,
\qquad q=D^{-1}q_{\mathrm{pc}}.
$$ (sr_preconditioned_shift_equivalence)

Thus the shift acts exactly once in the normalized coordinate system; writing
$(G+\varepsilon I)q=b$ for the original unnormalized $G$ would specify a
different regularizer.  The shift is applied in the declared ambient
real-coordinate basis.  For a
constrained factor block that basis may be redundant; its Euclidean norm is the
chosen Tikhonov metric and is therefore part of the optimizer definition.
Coordinates whose metric diagonal falls below a fixed absolute floor
($10^{-10}$) are frozen for that step before normalization rather than amplified by division through a
nearly zero scale.  If one member of an $(x,y)$ pair of a complex
parameter is frozen, both members are frozen.

### Matrix forms and the push-through identity

The coordinate system can be written as an ordinary real least-squares
problem.  Define

$$
Y=X_q\in\mathbb C^{Q\times N},\qquad
Z=\begin{bmatrix}\operatorname{Re}Y&\operatorname{Im}Y\end{bmatrix}
  \in\mathbb R^{Q\times 2N},\qquad
h=\begin{bmatrix}\operatorname{Re}F\\-\operatorname{Im}F\end{bmatrix}.
$$ (sr_realified_arrays)

For a real wave function, $Y$ and $F$ are real.  The imaginary block of $Z$ is
then identically zero and carries an exactly zero dual, so it is omitted:
$Z=Y\in\mathbb R^{Q\times N}$ and $h=F$.  The real dual length is therefore
$N_{\mathrm{dual}}=N$ for real samples and $N_{\mathrm{dual}}=2N$ for complex
samples.

Then, before stabilization,

$$
ZZ^T=\operatorname{Re}(X_qX_q^\dagger)=G,
\qquad
Zh=\operatorname{Re}(X_qF)=b,
$$ (sr_realified_identities)

For the production shifted solve, define $Z_{\mathrm{pc}}=D^{-1}Z$ after
freezing sub-floor groups.  The ordinary real wide/tall pair is

$$
q_{\mathrm{pc}}
=(Z_{\mathrm{pc}}Z_{\mathrm{pc}}^T+\varepsilon I_Q)^{-1}
  Z_{\mathrm{pc}}h
=Z_{\mathrm{pc}}
 (Z_{\mathrm{pc}}^TZ_{\mathrm{pc}}+\varepsilon I_{N_{\mathrm{dual}}})^{-1}h,
\qquad q=D^{-1}q_{\mathrm{pc}}.
$$ (sr_mixed_push_through)

The first expression is the **wide** real-coordinate solve and the
second is the **tall** realified-sample solve.  One regime decision serves
every route: it compares the number $Q$ of surviving coordinates with the
global real dual length $N_{\mathrm{dual}}$ ($N$ summed over all ranks of
the solve, doubled for complex samples), not the storage length with the
number of walkers, and assigns equality to the tall route.  The
real/complex decision behind $N_{\mathrm{dual}}$ is made once per
optimizer entry and is the same on every rank, so the regime is too.

The wide solves never form $Z$: they apply $Z_{\mathrm{pc}}Z_{\mathrm{pc}}^T$
through $Y$, and the host stores a wide $Y$ sample-major so that each CG
operator application is one pass over it.  The tall solves need $Z$.  On
the host it is formed (a view of $Y$ for real samples, $[\operatorname{Re}Y,
\operatorname{Im}Y]$ for complex samples), gathered from all ranks with one
`Allgatherv`, and every rank solves the complete dual system redundantly.
The device tall kernels instead realify $Y$ inside the compiled kernel and
keep $Z$ and the dual sharded by sample.

### Scale-invariance preconditioning

For the SR step to be invariant under rescalings of individual
coordinates [Becca-Sorella (2017)](https://doi.org/10.1017/9781316417041), the canonical metric is normalized
to a correlation matrix,

$$
G^{pc}_{a b} =
\frac{G_{a b}}{\sqrt{G_{a a}\,G_{b b}}},
$$ (sr_preconditioning)

with the corresponding rescaling of the force,

$$
b^{pc}_{a} = \frac{b_a}{\sqrt{G_{a a}}}.
$$ (sr_force_precond)

The natural-gradient direction $q^{pc}$ in the rescaled basis is
related to the original-coordinate direction by
$q_a=q^{pc}_a/\sqrt{G_{aa}}$.  No separate matrix path is needed:
$Y$ is rescaled row-wise,

$$
Y_{a i} \leftarrow \frac{Y_{a i}}{\sqrt{G_{aa}}}.
$$ (sr_X_normalization)

The residual vector $F$ is **not** rescaled, because $b$ is obtained by
contracting the rescaled $Y$ with $F$.  Preconditioning is applied after the
complete coordinate derivative construction.

### Matrix-free conjugate-gradient solve with MPI distribution

Neither the wide nor tall path needs to assemble an inverse.  CG always
uses real state vectors and ordinary real dot products: $q$ on the wide
path and an $N_{\mathrm{dual}}$-component vector on the tall path.  Its
right-hand side, warm start, residual norm, and while-loop carry are all
real.  Raw derivative sources and $F$ keep their sample dtype until $Y$,
$Z$, and $h$ have been formed: complex at a twist, and real for a real wave
function, whose transform and centering stay real.  When the ranks of one
solve hold both, every rank takes the complex dtype.

Samples are distributed over MPI ranks.  Rank $d$ holds the raw block
derivative sources for $N_d$ samples and
$F^{(d)}\in\mathbb C^{N_d}$.  It applies the declared derivative/tangent
policies locally and obtains $Y^{(d)}=X_q^{(d)}$.  The global coordinate
sample matrix is

$$
Y = \bigl[\,Y^{(1)},\,Y^{(2)},\,\dots,\,Y^{(D)}\,\bigr]
\in \mathbb{C}^{Q \times N},
\qquad N = \sum_{d=1}^{D} N_d.
$$ (sr_X_distributed)

All optimization estimators use finite **real** sample weights.

On the wide path, a matrix-vector product is

$$
v\longmapsto
\operatorname{Re}\!\left[
\sum_d Y^{(d)}\bigl(Y^{(d)\dagger}v\bigr)\right]
+\varepsilon_{\mathrm{SR}}v,
$$ (sr_local_matvec)

where the sum is an all-reduce and $v\in\mathbb R^Q$.  The tall path
uses the local realification $Z^{(d)}$ and reconstructs $q=Zy$.  Its
global dual vector has length $N_{\mathrm{dual}}$ ($\sum_dN_d$ for real
samples, $2\sum_dN_d$ for complex samples).  On a device, rank $d$ owns
the dual segment of its own samples and only local sample blocks and a
few state vectors are resident per rank.  The host tall path gathers $Z$
and $h$ instead and solves the whole dual on every rank.  The dense
coordinate metric is never required by CG.

CG stops at the relative tolerance, at `cg_max_iter`, or on a breakdown
(a non-finite or vanishing $p^TAp$; that iteration takes no step and is not
counted).  A solve that ends above `cg_tol` $\lVert b\rVert$ is
reported with one `SR CG did not converge` warning at the default
verbosity, giving the regime, the iteration count, the residual, the target
and whether CG broke down; the unconverged step is then applied.  In a
twist-averaged run there is at most one such line for the shared scope and
one for the twist-local scopes, which states how many twists did not converge
and shows the worst of them.  Non-finite SR samples stop the run with an
error on every route and every rank.


### Twist ownership and parameter coordinates

In twist-averaged optimization, Jastrow parameters are shared across twists; determinant parameters (MO coefficients and lambda blocks) are twist-local. Every twist receives the same Jastrow update, while determinant updates remain twist-local. The shared state is checkpointed once and restored to every twist. The detailed twist-optimization formulation is work in progress.

`lm_subspace_dim` follows the same split. In a single-twist calculation, `lm_subspace_dim=d` selects at most d coordinates total from the combined Jastrow and determinant/MO coordinate set. In a twist-averaged calculation, positive `lm_subspace_dim=d` applies independently: up to d shared Jastrow coordinates and up to d determinant/MO coordinates per twist. It does not construct one joint 2d-coordinate pencil. The SR collective direction is additional to each budget. Real and imaginary parts of a complex parameter are selected together; a tie at the selection boundary may increase the realized coordinate count.

The per-step diagnostics cover every optimized variable: the shared Jastrow coordinates once and each twist's own determinant coordinates. `Max generalized force`, `Max generalized force signal-to-noise` (eq. {eq}`sr_signal_to_noise`) and the `verbosity = "high"` list of top coordinates are maxima over all of them, not averages over twists. A top row of a twist's own coordinate carries its twist index, e.g. `mo_coefficients_up[t3:q980]` for the twist stored in `twist_003.h5`. One line per twist gives that twist's largest signal-to-noise. `Number of variational parameters` and `Number of real optimization coordinates` count the parameters one twist holds; the `(all twists)` lines count the shared ones once plus those of every twist. With `verbosity = "high"`, the `[optimizer block]` lines give a shared block once and a twist's own block once per twist, as `mo_coefficients_up[t3]`. The `Coordinates (all twists)` line adds the real, storage, active and frozen counts of every twist to the shared ones.

(subsec:twist_optimization_energy)=

### Per-step twist-averaged energy and statistical error

During optimization, the reported twist-averaged energy uses the same Brillouin-zone weighting as production twist-average reporting. For normalized twist weights, the optimizer-level energy is the weighted sum of per-twist jackknife energy estimates, and its statistical error is the square root of the sum of squared weighted per-twist errors (using normalized twist weights). Twist dispersion is physical and is not added to the Monte Carlo error. Each twist contributes once regardless of MPI rank count; every rank logs the same twist-averaged energy and error.

The LM energy check in {ref}`subsec:lm_stabilization` uses the twist-averaged error for the shared Jastrow pencil and the per-twist error for the twist-local determinant pencil. Plain SR and aSR do not use this check; their per-step energy log reports the twist-averaged energy.

(subsec:sr_signal_to_noise)=

### Convergence criterion: signal-to-noise of the force

During an optimization, jQMC prints at every step the variational energy $E$
and the maximum signal-to-noise ratio among the coordinate forces
(`Max generalized force signal-to-noise` in the output),

$$
\bigl(\text{signal-to-noise ratio}\bigr) \;\equiv\;
\max_{a}\,\Bigl|\,\frac{b_a}{\sigma_{b_a}}\,\Bigr|,
$$ (sr_signal_to_noise)

where $\sigma_{b_a}$ is the jackknife error bar on $b_a$
(sec. {ref}`sec:error-analysis`), computed coordinate by coordinate from
jackknife replicates of the complete coordinate force.  When this ratio is
of order one, every force component is statistically consistent with zero.
The `jqmc` program itself does not stop on this criterion: it always runs
`num_opt_steps` steps.  The `jqmc_workflow` VMC workflow can stop
automatically when the ratio, averaged over the last few steps, falls to or
below its `target_snr` setting ({doc}`workflows`).

The ratio is built from the covariance (SR) force estimator.  It therefore
certifies the stationarity of that estimator; it does not include the
finite mean local-energy derivative $D_a$ that the linear method also uses
(sec. {ref}`subsec:linear_method`).

(subsec:linear_method)=

## Linear Method and accelerated SR

LM ([Sorella (2005)](https://doi.org/10.1103/PhysRevB.71.241103);
[Umrigar et al. (2007)](https://doi.org/10.1103/PhysRevLett.98.110201);
[Toulouse and Umrigar (2007)](https://doi.org/10.1063/1.2437215);
[Sorella, Casula, and Rocca (2007)](https://doi.org/10.1063/1.2746035)) uses exactly the same real coordinate vector
$q$ and embedding $\mathcal E$ as SR.  Therefore
real, complex, mixed, and fixed-rank geminal parameter sets share one real
generalized-eigenvalue solver.

### Coordinate derivative estimators

First construct the complete coordinate derivatives $O_q,D_q$ as described
above.  Center them only after tied Wirtinger components and
constrained-block coordinates have been resolved:

$$
\delta O_{ia}=(O_q)_{ia}-\bar O_{q,a},
\qquad
\delta D_{ia}=(D_q)_{ia}-\bar D_{q,a},
\qquad
\delta e_i=e_{Li}-\bar e_L.
$$ (lm_centered_observables)

For centering, $\bar e_L$ may be the full complex weighted mean.
jQMC uses the physical-real $H_0=\operatorname{Re}\langle
e_L\rangle$ in the force residual; the difference is a constant and
cancels exactly against centered $\delta O$ when identical weights are
used.

With every average weighted by $w_i/W$, construct

$$
\begin{aligned}
S^c_{ab}&=\langle\delta O_a^*\delta O_b\rangle,\\
f^c_a&=-2\langle\delta O_a^*\delta e\rangle,\\
D_a&=\langle \partial_{c_a} e_L\rangle,\\
K^c_{ab}&=\langle e_L\,\delta O_a^*\delta O_b\rangle,\\
B^c_{ab}&=\langle\delta D_a^*\delta O_b\rangle,\qquad
\tilde B^c_{ab}=(B^c)^\dagger_{ab}=\langle\delta O_a^*\delta D_b\rangle.
\end{aligned}
$$ (lm_sample_matrices)

Here $D_a=\langle\partial_{c_a} e_L\rangle$ is the sample mean of the
local-energy parameter derivative. For a parameter-independent Hermitian
Hamiltonian and a real coordinate $q_a$, exact sampling gives

$$
\langle\partial_{q_a}e_L\rangle
=\langle O_a(e_L^*-e_L)\rangle
=-2i\langle\delta O_a\,\delta\operatorname{Im}e_L\rangle.
$$ (lm_complex_mean_derivative)

This reduces to zero for a real state with real local energy; it need not
vanish for a complex state. For example, on a periodic interval $[0,2\pi]$
with $H=-\tfrac12\partial_x^2$ and $\Psi=\exp(iq\sin x)$,
$e_L=\tfrac12q^2\cos^2x+\tfrac{i}{2}q\sin x$ and
$\langle\partial_qe_L\rangle=q/2$. The same identity applies to the
holomorphic derivative convention in {ref}`subsec:force_local_energy_derivative`.
The linear method retains the derivative mean and evaluates its two
reference/tangent elements separately:

$$
H_{0a} = \langle\delta O_a\,\delta e\rangle + D_a,\qquad
H_{a0} = \langle\delta O_a^*\,\delta e\rangle = -\tfrac12 f^c_a,
$$

i.e. it is **not Hermitian-symmetrized** (note that the bra element uses
the unconjugated covariance).  Likewise the bulk block is
$K^c + \tilde B^c$, without symmetrization.  The per-sample
$\partial_{c_a} e_L$ is the same quantity used by $B^c$, so the LM path
pays no extra cost; the plain SR force is the covariance-only estimator.

$O_q$, $D_q$, $e_L$, $S^c$, $K^c$, $B^c$, and $f^c$ remain complex through these
contractions.  In particular, taking `real(D)` or `real(e_L)` per
walker loses terms at a non-Gamma twist.  Only the physical scalar
$H_0=\operatorname{Re}\langle e_L\rangle$ is projected immediately.

### Centered projective pencil

The LM tangent basis is
$\{\Psi,\delta O_{q,a}\Psi\}$ with
$\delta O_q=O_q-\bar O_q$.  Centering removes the component parallel to
the current wave function and therefore represents the tangent of the
normalized projective state.  It is applied after the real-coordinate
chain rule, so real, complex, mixed, and constrained parameter blocks all
obey the same definition.

Define the complex centered augmented matrices, **evaluated element by
element with their own estimators and NOT Hermitian-symmetrized**
(the non-symmetric linear method):

$$
H_c=
\begin{bmatrix}
H_0 & \langle\delta O_a\,\delta e\rangle + D_a\\[2pt]
\langle\delta O_a^*\,\delta e\rangle & K^c + \tilde B^c
\end{bmatrix},
\qquad
S_c=
\begin{bmatrix}1&0\\0&S^c\end{bmatrix},
$$ (lm_storage_augmented_matrices)

with $D_a=\langle\partial_{c_a} e_L\rangle$ and
$\tilde B^c_{ab}=\langle\delta O_a^*\,\delta(\partial_{c_b} e_L)\rangle$
(note the orientation: the local-energy derivative sits on the KET
index).  The top row is the bra estimator
$\langle\Psi|H|\delta O_a\Psi\rangle$ -- it carries the finite
$D_a$; the first column is the ket estimator
$\langle\delta O_a\Psi|H|\Psi\rangle$.  Under exact
$|\Psi|^{2}$ sampling the two are complex conjugates (Hermiticity,
{ref}`subsec:force_local_energy_derivative`) and $H_c$ becomes
Hermitian in expectation; on a finite sample they differ, and
the difference is precisely the information the optimizer needs.
The production pencil keeps these finite-sample estimators without
Hermitian symmetrization. Hermiticity can incorporate $D_a$ into correctly
conjugated complex covariances in exact expectations; it does not justify
setting $D_a=0$ for a complex state. Those covariance forms and the explicit
local-energy-derivative form can differ on a finite sample. In particular,
assigning the same complex number to both reference/tangent entries would
also violate their bra/ket conjugation relation. The real-coordinate pencil
below is formed only after the complex contractions are complete.

The optimization coefficients are real; the real-coordinate pencil is

$$
H_R=\operatorname{Re}H_c,
\qquad
S_R=\operatorname{Re}S_c
=\begin{bmatrix}1&0\\0&\operatorname{Re}S^c\end{bmatrix}.
$$ (lm_real_coordinate_transform)

$S_R$ is real symmetric; $H_R$ is real but **non-symmetric**.  The LM
equation is the general (non-symmetric) generalized eigenproblem

$$
H_Rw=E_{\rm LM}S_Rw,
\qquad w\in\mathbb R^{Q+1},
$$ (lm_generalized_evp)

solved with a general eigensolver (`scipy.linalg.eig`, right
eigenvectors).  Because $H_R$ is non-symmetric, complex-conjugate
eigenvalue pairs can occur; the physical root is selected among the
numerically real eigenvalues by the largest reference weight
(sec. {ref}`subsec:lm_stabilization`).  If no eigenvalue is numerically
real, no LM update is formed: a warning is printed and the LM step is
rejected (see the acceptance rule in sec. {ref}`subsec:lm_stabilization`).

The centered tangent update is

$$
q_{\rm LM}=\frac{w_{1:}}{w_0},
\qquad
\delta p=\mathcal E q_{\rm LM}.
$$ (lm_full_update)

The reference row of $S_R$ is exactly $(1,0,\ldots,0)$, and no additional
rational decode is applied.

The centered tangent family and a raw-derivative linear family have the same
first-order parameter direction.  Projectively rewriting
$\Psi+q^T\delta O\,\Psi$ in the raw basis changes the coefficient from $q$
to $q/(1-\bar O_q^Tq)=q+O(\lVert q\rVert^2)$.  LM applies the model step
directly, so this second-order rational decode is not applied.

This is valid without branching on the sampled wave-function dtype:

| sampled WF | parameters | optimizer representation |
|---|---|---|
| real or complex | all real | $\mathcal E=I$; real SR and real GEP |
| real or complex | all complex | one $x/y$ pair per entry; real SR and real GEP |
| real or complex | real + complex | concatenated real columns and $x/y$ pairs; the same real SR and real GEP |
| real or complex | `AGPn_data` or `SD_data` | the same real SR and real GEP; rank cap is structural, and the occupied MO rows stay orthonormal in the AO overlap metric |

The same centered pencil is used in all four rows.

(subsec:lm_sr_collective)=

### Reduced LM with the SR collective direction

Let $g\in\mathbb R^Q$ be the stabilized SR direction obtained from
eqs. {eq}`regularization_s` and
{eq}`sr_preconditioned_shift_equivalence`, de-preconditioned but not yet
multiplied by the learning rate.  Select real-coordinate indices
$a_1,\ldots,a_m$ and define

$$
A=\begin{bmatrix}g&e_{a_1}&\cdots&e_{a_m}\end{bmatrix}
\in\mathbb R^{Q\times(m+1)}.
$$ (lm_reduced_basis_map)

The first column is the SR collective direction; the remaining columns are
individual real coordinates.  Augment it without changing the reference
state,

$$
\mathcal A_0=
\begin{bmatrix}1&0\\0&A\end{bmatrix},
\qquad
H_A=\mathcal A_0^TH_R\mathcal A_0,
\qquad
S_A=\mathcal A_0^TS_R\mathcal A_0.
$$ (lm_reduced_matrices)

The one reduced GEP is

$$
H_Av=E_{\rm LM}S_Av,
$$ (lm_reduced_evp)

and the final update is

$$
q_{\rm LM}=A\frac{v_{1:}}{v_0},
\qquad
\delta p=\mathcal E q_{\rm LM}.
$$ (lm_sr_update)

In practice the pencil is built directly from the walker-wise collective
columns $\delta O_{\rm SR}=(O_q-\bar O_q)g$ and
$\delta D_{\rm SR}=(D_q-\bar D_q)g$ and the selected columns of $O_q,D_q$,
which is algebraically identical to the explicit $\mathcal A_0$ congruence
and avoids storing the full matrices.

The $m$ = `lm_subspace_dim` coordinates are selected after realification.
jQMC computes the signal-to-noise score $|b_a|/\sigma_{b_a}$ for every real coordinate.
The real and imaginary coordinates belonging to one complex storage
entry form one atomic selection group: rank the group by the maximum of
its two scores and include or exclude both coordinates together.  Real
parameters are one-coordinate groups.  Include complete groups in
descending score order and include all groups tied at the boundary,
even if that exceeds the requested coordinate count.  The SR collective
column is always present and is not counted in this budget.

(subsec:practical_asr)=

### Accelerated SR in real-coordinate space

aSR drops the selected coordinates and remains on the one real line

$$
q=\gamma g,\qquad \gamma\in\mathbb R.
$$ (asr_delta_alpha)

Use the centered projective one-dimensional family
$\Psi+\gamma\delta\partial_g\Psi$.  Partition the centered real matrices as

$$
H_R=\begin{bmatrix}H_0&h_{\rm bra}^T\\h_{\rm ket}&H_{qq}\end{bmatrix},
\qquad
S_R=\begin{bmatrix}1&0\\0&S_{qq}\end{bmatrix},
$$ (asr_coefficients)

where $h_{\rm bra}$ is the first row and $h_{\rm ket}$ the first column of
the non-symmetric pencil of eq. {eq}`lm_storage_augmented_matrices`, and
define

$$
H_1=\tfrac12\left(h_{\rm bra}+h_{\rm ket}\right)^Tg,\qquad H_2=g^TH_{qq}g,
\qquad S_2=g^TS_{qq}g.
$$ (asr_sample_coefficients)

Write $C_g=\operatorname{Re}\langle\delta O_g^*\,\delta e\rangle$ for the
covariance contraction along $g$ and
$D_g=\operatorname{Re}\langle\partial_g e_L\rangle$ for the mean local-energy
derivative along $g$.  For a real wave function the bra and ket covariances
coincide, and $H_1=C_g+D_g/2$; the finite $D_g$ is the information retained
by the non-symmetric pencil.

Equivalently,
$S_2=\operatorname{Re}\langle|\delta O_g|^2\rangle$ for
$\delta O_g=(O_q-\bar O_q)g$; the Hamiltonian coefficients come from the
matching centered $\delta O_g,\delta D_g$ contractions.  The energy model is

$$
E(\gamma)=
\frac{H_0+2\gamma H_1+\gamma^2H_2}
     {1+\gamma^2S_2},
$$ (asr_energy_gamma)

and its finite stationary points solve

$$
-H_1S_2\gamma^2
 +(H_2-H_0S_2)\gamma
 +H_1=0.
$$ (asr_quadratic)

$S_2$ is evaluated from the centered overlap matrix (not from $g^Tb$,
which is not an overlap contraction).  The two stationary roots are the two
projective eigenstates on the SR line.  aSR selects the root with smaller
$|\gamma|$, which is the branch with the larger reference-state overlap

$$
w_0^2=\frac{1}{1+\gamma^2S_2}.
$$ (asr_reference_weight)

The lower model-energy root is not used, because it can be an almost pure
tangent state with arbitrarily large $|\gamma|$.  **Negative reference-connected roots are accepted**: with
the non-symmetric $H_1 = C_g + D_g/2$
({ref}`subsec:force_local_energy_derivative`), a negative root means the
model has detected that the covariance (SR) direction points uphill on
the measurable landscape, and stepping *backward* along $g$ is the
model-optimal move; rejecting it and stepping $+\gamma_{\rm fb}$
forward would walk uphill.
Only the magnitude is guarded: if the reference-connected root is
non-finite, zero, or $|\gamma|$ exceeds the safety bound
$\gamma_{\max}$ (`asr_gamma_max`, default $0.5$), the
fallback magnitude $\gamma_{\rm fb}$ (`asr_fallback_gamma`, default $0.1$)
is used, with the model root's sign in the over-cap case.
Both roots, the selected branch and direction, and the final gamma are written
to the normal optimizer log.  The decoded update is always
$\delta p=\mathcal E(\gamma g)$; no parameter-type branch is needed inside
aSR.  For an internally consistent centered overlap estimator,
$S_2\ge0$, so the denominator is positive for every real $\gamma$:

$$
1+\gamma^2S_2\ge1.
$$ (asr_denominator_squares)

Values of $S_2$ that are negative beyond roundoff are rejected.  A tie in $|\gamma|$ is resolved in favor of
$\gamma_+$.

(subsec:lm_stabilization)=

### Stabilization and acceptance

Stabilization is performed **after** the real-coordinate pullback, so
it has one implementation for every parameter type:

1. Re-symmetrize the overlap $S_R$ and the conditioning metric
   $G_{\rm cond}$ numerically, $(S+S^T)/2$.  $H_R$ is **not** symmetrized.
2. Use the centered tangent metric
   $G_{\rm cond}=S_{1:,1:}=\operatorname{Re}S^c$.
3. Remove coordinates with negligible diagonal in $G_{\rm cond}$.
4. Normalize the survivors to a correlation matrix $C$ and select the
   active coordinates $A$ with one pivoted Cholesky factorization of
   $C-c\,I$ ($c$ = `lm_cond`, $0<c<1$), so that
   $\lambda_{\min}(C_A)\ge c$.  The log reports the active and removed
   counts and $\lambda_{\min}(C_A)$.

After the active coordinates are fixed, jQMC solves the corresponding
centered augmented submatrices of $(H_R,S_R)$ as a general
(non-symmetric, right-eigenvector) eigenproblem, reduced to standard form
with the Cholesky factor of the active overlap block; a failed Cholesky
factorization is reported as an error.  The reference-state row and column
are never removed.  The same host implementation serves the CPU and GPU
routes.

The generalized eigenvectors are normalized by $w^T S_Rw=1$.  Because the
centered tangent block is reference-orthogonal, the reference-state weight is

$$
w_{\rm ref}=w_0^2.
$$ (lm_physical_reference_weight)

This identity holds at Gamma and at a twist.  Among the eigenvectors with
numerically real eigenvalues ($|\operatorname{Im}\lambda|\le 10^{-8}\max(|\lambda|,1)$),
jQMC selects the one with the largest $w_0^2$,
breaking an exact tie by the lower eigenvalue, and decodes
$q=w_{1:}/w_0$.  Centering makes the complete pencil, conditioning metric,
selection, and decoded update invariant under a constant shift of a raw
derivative column.

The selected LM candidate is accepted only if

$$
w_{\rm ref}\ge w_{\min},
\qquad
E_{\rm LM}\le H_0+3\,\sigma_E,
$$ (lm_acceptance)

where $w_{\min}$ is `lm_reference_weight_min` (default $0.9$), $E_{\rm LM}$
is the selected eigenvalue, and $\sigma_E$ is the statistical error of the
current energy (sec. {ref}`subsec:twist_optimization_energy` for
twist-averaged runs).  The factor $3$ is fixed.  If either test fails, or
if the pencil has no numerically real eigenvalue, the
LM candidate is rejected and the aSR model of
sec. {ref}`subsec:practical_asr` is solved on the already available SR
direction.  A reduced LM pencil already carries this direction as its first
tangent coordinate; a full LM pencil is contracted with $q_{\rm SR}$.
Therefore the fallback requires no new samples.  Only if this aSR model
itself is invalid is the plain-SR fallback scale `asr_fallback_gamma` used.
Non-finite updates and failed post-update constraint checks stop the run
rather than triggering a fallback.  With `[control] verbosity = "high"`,
the accept/reject decision and its reason are written to the output.
(subsec:geminal_optimizer_contract)=
## Rank-constrained geminals: overview

Rank constraints are available for dense AGP and MO-based representations.
The dense AO approach was described in the earlier jQMC publication. The detailed
formulation of the new MO optimization is work in progress.

The MO representation reduces storage for a large AO basis by keeping the retained
MO coefficients instead of a dense AO geminal matrix. Equivalent retained factors
give the same wave function and observables up to round-off. Accepted MOs remain
orthonormal. `AGPn_data` also stores paired occupations and an unpaired block for
open shells; `SD_data` stores occupied MO rows only. The stored MO count caps the
paired rank; zero occupations can lower it.

### Optimizer choices and restrictions

| representation | setting | rank policy and restrictions |
|---|---|---|
| `AGP_data` | `fix_geminal_rank = 0` (default) | unconstrained dense AGP |
| `AGP_data` | `fix_geminal_rank = n > 0` | converted once to `AGPn_data` only when `opt_geminal = true`; cannot be combined with `opt_lambda_basis_exp` or `opt_lambda_basis_coeff`; n stored factors cap rank |
| `AGPn_data` | stored MO factors | paired rank at most stored MO count; exactly that count when every retained occupation is nonzero; a different positive `fix_geminal_rank` only warns and is ignored |
| `SD_data` | occupied MO rows | determinant rank is structural; no lambda or virtual-MO rows; a positive `fix_geminal_rank` different from N_down is an error |

With `time_reversal=True`, down-spin MOs are tied to the up-spin block. Otherwise the spin channels are independent.
