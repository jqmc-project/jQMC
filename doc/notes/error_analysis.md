(sec:error-analysis)=

# Error analysis
## Statistical estimators for QMC observables

Every QMC simulation -- MCMC sampling (used for VMC) and GFMC projection
(used for LRDMC) -- ultimately
estimates the mean and variance of a local observable
$\hat{O}(\mathbf{x})$ from a Monte Carlo sequence of electronic
configurations $\{\mathbf{x}_t\}_t$. Three issues turn this
seemingly elementary task into a numerical-statistics problem
that QMC codes have to handle explicitly:

1. consecutive Monte Carlo samples are **auto-correlated**, so the
   naive variance underestimates the true statistical error;
2. the most interesting estimators (variance itself, energy
   variance, forces, parameter gradients) are *non-linear*
   functions of the Monte Carlo samples, so plug-in error
   propagation is not exact;
3. GFMC and reweighted MCMC produce **weighted** samples
   $\{(\mathbf{x}_t, w_t)\}_t$, and the weighted-mean estimator
   $\bar{O} = \sum_t w_t O_t / \sum_t w_t$ is non-linear in $w_t$
   even before the observable is non-linear in $\mathbf{x}_t$.

jQMC handles (1)-(3) with the standard pair of techniques used in
the field: **reblocking** to decorrelate, followed by the
**jackknife** resampling to extract a consistent mean and variance.
This section documents the precise estimators jQMC implements,
following the treatment in [Becca and Sorella,
*Quantum Monte Carlo Approaches for Correlated Systems*, Cambridge
University Press (2017)](https://doi.org/10.1017/9781316417041).

## Notation: chains, walkers, and samples

MCMC sampling runs $N$ **independent** Markov chains in
parallel over the MPI ranks. Each rank typically carries one or
more random-walk *walkers* that are themselves uncoupled. We index
the MPI / walker dimension by $n=1,\dots,N$ and the in-chain
Monte Carlo step by $i=1,\dots,M$. The configuration at step $i$
of chain $n$ is $\mathbf{X}_{i,n}$ and the per-sample local
observable is

$$
O_{i,n} \equiv \hat{O}(\mathbf{X}_{i,n}).
$$ (sample_def)

Crucially, the in-chain dimension is *autocorrelated* (consecutive
$\mathbf{X}_{i,n}$ share much of the same electronic configuration)
while the cross-chain dimension is *not* (different chains were
seeded independently and never communicate). Every decorrelation
step below acts along the $i$ dimension only; the $n$ dimension is
preserved unchanged. This independence statement does not apply to LRDMC
walkers coupled by branching; their population-first time binning is
described in {ref}`sec:error_analysis_practice` below.

## Reblocking (binning)

Reblocking replaces the raw chain by a coarser chain of
*binned averages*, decreasing the autocorrelation time. If
$N_j$ consecutive samples are averaged into one bin, the binned
sequence for chain $n$ is

$$
O_{j,n} = \frac{1}{N_j}\sum_{i\,\in\,\text{bin}\,j} O_{i,n},
\qquad j=1,\dots,M_b,
$$ (binned_sample)

where $M_b$ is the resulting number of binned samples per chain.
The bin width $N_j$ must be large enough that consecutive bins are
effectively independent, i.e. several times the autocorrelation length of
the chain; the jackknife formulas below then operate on the
$\{O_{j,n}\}$ array as if it were an i.i.d. sample. jQMC does not compute
an autocorrelation time. You choose the number of bins (`num_mcmc_bin_blocks`
or `num_gfmc_bin_blocks`), and the bin width follows from the number of
retained measurements; how to choose the number of bins and check that they
are long enough is described in {ref}`sec:error_analysis_bins`.

Binning **never** crosses the MPI / walker dimension: each $n$
is reblocked independently, because the chains are already
independent and pooling them would not decrease autocorrelation. In
jQMC this is reflected in the data layout: the binning helper
reshapes only the in-chain axis and broadcasts over $n$.

For the unweighted estimator below it is convenient to *flatten*
the $(j, n)$ array into a single index $l = 1,\dots,N_J$ with
$N_J = M_b\,N$. The autocorrelation has already been absorbed by
the binning, and the resulting flat sample is treated as
i.i.d. by the jackknife.

## Unweighted jackknife

QMC estimators of interest are generally *non-linear* functions of
the Monte Carlo sample -- variance, ratios of expectation values,
forces, energy gradients. Plug-in error propagation through such
functions is not reliable at finite $N_J$. The jackknife is a
distribution-free resampling technique that produces a consistent
variance estimator without any specific assumption on the observable.

Given the flat sample $\{O_l\}_{l=1}^{N_J}$, define the **$m$-th
jackknife replica** by deleting the $m$-th sample from the mean,

$$
O_m^{\mathrm{J}}
= \frac{1}{N_J - 1}\sum_{l \ne m} O_l,
\qquad m=1,\dots,N_J.
$$ (jk_unweighted_sample)

The jackknife mean and variance of the estimator are then

$$
\mu = \mu^{\mathrm{J}} =
\frac{1}{N_J}\sum_{m=1}^{N_J} O_m^{\mathrm{J}},
$$ (jk_mean)

$$
\sigma^{2} = (N_J - 1)\,(\sigma^{\mathrm{J}})^{2}
=
\frac{N_J - 1}{N_J}
\sum_{m=1}^{N_J}
\left[
\bigl(O_m^{\mathrm{J}}\bigr)^{2}
- \biggl(\frac{1}{N_J}\sum_{m'=1}^{N_J} O_{m'}^{\mathrm{J}}\biggr)^{2}
\right],
$$ (jk_variance)

where $(\sigma^{\mathrm{J}})^{2} = \frac{1}{N_J}\sum_{m=1}^{N_J}\bigl(O_m^{\mathrm{J}} - \mu^{\mathrm{J}}\bigr)^{2}$
is the spread of the replicas. jQMC reports $\mu \pm \sigma$. It does not
apply the jackknife bias correction $N_J\hat\theta - (N_J - 1)\mu^{\mathrm{J}}$
($\hat\theta$ being the estimate from all samples): for a ratio estimator
such as eq. {eq}`weighted_mean` the remaining bias is of order $1/N_J$,
while the statistical error is of order $1/\sqrt{N_J}$.

The $(N_J - 1)$ inflation factor in eq. {eq}`jk_variance`
compensates the fact that each $O_m^{\mathrm{J}}$ already averages
$N_J - 1$ samples (which would otherwise make the spread of the
replicas $1/(N_J - 1)$ times the true spread). For a *linear*
observable the jackknife variance reduces to the usual
$\hat\sigma^{2}/N_J$ estimator; for a non-linear estimator it
remains consistent at order $1/N_J$.

A noteworthy property of the jackknife is that all replicas are
constructed from the *same* binned sample, so for linear or
near-linear observables many of the partial sums share factors and
the cost of producing all $N_J$ replicas is
$\mathcal{O}(N_J)$ rather than $\mathcal{O}(N_J^{2})$. jQMC
exploits this in its post-processing helpers, where the total
sum $\sum_l O_l$ is computed once and each replica obtained as
$\bigl(\sum_l O_l - O_m\bigr) / (N_J - 1)$.

## Weighted jackknife (GFMC and reweighted MCMC)

GFMC walkers carry *importance weights* $\{w_t\}_t$ that
encode the projector accumulated since the last branching. MCMC
samples carry the Attaccalite-Sorella weights
$w_{\mathrm{AS}} = |\Psi_T/\Psi_G|^{2}$ (see {doc}`vmc`); with the default
`epsilon_AS = 0.0` all of them equal 1. The estimator of interest in both
cases is the **weighted mean**

$$
\bar{O} = \frac{\sum_l w_l\,O_l}{\sum_l w_l},
$$ (weighted_mean)

which is non-linear in the joint sample $\{(w_l, O_l)\}_l$ -- the
denominator $\sum_l w_l$ is itself a Monte Carlo estimate, and the
ratio biases the plug-in variance.

The weighted jackknife replicates eq. {eq}`weighted_mean` after
deleting one sample:

$$
O_m^{\mathrm{J}}
= \frac{\sum_{l \ne m} w_l\,O_l}{\sum_{l \ne m} w_l}
= \frac{\sum_l w_l\,O_l - w_m\,O_m}{\sum_l w_l - w_m},
\qquad m=1,\dots,N_J.
$$ (jk_weighted_sample)

The right-hand form is the one used in code: the total numerator
$\sum_l w_l O_l$ and total weight $\sum_l w_l$ are pre-accumulated
once, and each replica is produced by a constant-time subtraction.
Once the replicas are in hand, the mean and variance are given by
the same eq. {eq}`jk_mean` and eq. {eq}`jk_variance` as in the
unweighted case:

$$
\mu = \mu^{\mathrm{J}} =
\frac{1}{N_J}\sum_{m=1}^{N_J} O_m^{\mathrm{J}},
$$ (jk_weighted_mean)

$$
\sigma^{2} = (N_J - 1)\,(\sigma^{\mathrm{J}})^{2}
=
\frac{N_J - 1}{N_J}
\sum_{m=1}^{N_J}
\left[
\bigl(O_m^{\mathrm{J}}\bigr)^{2}
- \biggl(\frac{1}{N_J}\sum_{m'=1}^{N_J} O_{m'}^{\mathrm{J}}\biggr)^{2}
\right].
$$ (jk_weighted_variance)

### Combining binning and weights

Bin along the time axis and retain the **sums of weights and weighted
observables**, not the separate arithmetic means of weights and observables:

$$
W_{j,n} = \sum_{i\,\in\,\text{bin}\,j} w_{i,n},
\qquad
A_{j,n} = \sum_{i\,\in\,\text{bin}\,j} w_{i,n}O_{i,n}.
$$ (binning_weights)

For a bin with nonzero weight, its weighted observable is $A_{j,n}/W_{j,n}$.
Keeping the sums also handles unequal bin lengths without changing sample
weights. For MCMC, flatten the independent rank/walker chains and their
time bins into $b=1,\ldots,N_J$. The delete-one-bin replica is

$$
O_b^{\mathrm{J}}
= \frac{A_{\mathrm{total}}-A_b}{W_{\mathrm{total}}-W_b},
\qquad
A_{\mathrm{total}}=\sum_b A_b,\quad W_{\mathrm{total}}=\sum_b W_b.
$$ (jk_weighted_block)

The replica mean and jackknife variance above then apply. Require at least
two nonempty bins and nonzero retained weight in every replica. LRDMC walkers
are correlated by branching: sum across ranks and walkers at each time step
**before** time binning, as described in {ref}`sec:error_analysis_practice`.

A minimal counterexample is two identical bins containing weights $(1,9)$
and observables $(0,10)$. Every correct delete-bin estimate is $90/10=9$.
Multiplying separately averaged weights and observables would instead give
$5$, losing their covariance:

```python
weights = [[1, 9], [1, 9]]
observables = [[0, 10], [0, 10]]
W = [sum(w) for w in weights]
A = [sum(wi * oi for wi, oi in zip(w, o)) for w, o in zip(weights, observables)]
replicas = [(sum(A) - a) / (sum(W) - w) for a, w in zip(A, W)]
assert replicas == [9.0, 9.0]
```

## Complex observables

When the determinant uses a complex Bloch AO basis at a non-$\Gamma$ twist
(`aos_data.k_point` is nonzero), the per-walker local observable
$\hat{O}(\mathbf{x})$ can be **complex** -- e.g., $e_L = \hat{H}\Psi/\Psi$,
$O_k = \partial(\ln\Psi)/\partial\alpha_k$, or atomic-force ingredients. The
expectation under $|\Psi|^{2}$ is real for any Hermitian observable, but the
per-sample complex value carries an imaginary part that averages to zero only
in expectation. The jackknife is extended as follows.

### What is reported

The jackknife replicas of eq. {eq}`jk_weighted_sample` (or
eq. {eq}`jk_weighted_block`) are formed with the complex $O_l$; the weights
are always real (they come from the Attaccalite-Sorella reweighting or from
the GFMC projector). Each replica is then split into its real and imaginary
parts, $O_m^{\mathrm{J}} = O_m^{\mathrm{J},\mathrm{Re}} + i\,O_m^{\mathrm{J},\mathrm{Im}}$.
For a Hermitian observable jQMC reports

```{math}
:label: jk_complex_report
\mu^{\mathrm{Re}} \pm \sigma_{\mathrm{Re}},
```

where $\mu^{\mathrm{Re}}$ and $\sigma_{\mathrm{Re}}$ follow from
eq. {eq}`jk_mean` and eq. {eq}`jk_variance` applied to the real parts
$O_m^{\mathrm{J},\mathrm{Re}}$ only. The imaginary part does not enter the
error bar: the reported quantity is the real number $\mu^{\mathrm{Re}}$, and
its error is the error of the real part. The mean $\mu^{\mathrm{Im}}$ of the
imaginary parts is used only as the diagnostic below. This applies to the
energy printed by the `mcmc`, `vmc`, `lrdmc-bra` and `lrdmc-tau` jobs and by
`jqmc-tool mcmc compute-energy` / `jqmc-tool lrdmc compute-energy`.

The energy variance reported next to the energy uses the modulus square of the
complex replica, $\mathrm{Var}_m = \langle |e_L|^{2}\rangle_m - |\langle e_L\rangle_m|^{2}$
in the MCMC driver (see {ref}`sec:error_analysis_practice`).

### Im $\langle E_L\rangle$ contamination guard

For a Hermitian operator and a correctly implemented complex $\Psi$,
$\langle\mathrm{Im}\, e_L\rangle$ is zero in expectation; at finite sample size it
is noise. The MCMC driver (jobs `mcmc` and `vmc`) compares it with the error of
the real part and logs a warning (it does not stop the run) when

```{math}
:label: im_eL_guard
\bigl|\,\mu^{\mathrm{Im}}\,\bigr| \,>\, 2\,\sigma_{\mathrm{Re}}.
```

The LRDMC drivers do not apply this check. Because a $2\sigma$ threshold is
crossed by chance in a noticeable fraction of short runs, a single warning is
not evidence of an error: check equilibration, the number of bins and the
sample size first. A warning that persists in long, equilibrated runs points to
an implementation problem, such as a wrong conjugation or a missing
per-electron Bloch wrap-restore phase in the complex AO evaluator; see
{doc}`ao` and {doc}`complex_gradients`. The guard cannot detect everything: a
silent cast to a real number erases the imaginary noise altogether, and every
check at the $\Gamma$ point, where all quantities are real, is blind to such
casts. Complex code paths therefore have to be tested at a non-$\Gamma$ twist.

(sec:error_analysis_practice)=

## In practice

The implementation differs between MCMC and GFMC because of the very
different correlation structure of the two algorithms.

- **MCMC (`job_type = "mcmc"` and `"vmc"`).** Each MPI rank and walker
  carries an independent chain. Within each chain the stored measurements
  after the first `num_mcmc_warmup_steps` are split into
  `num_mcmc_bin_blocks` bins (contiguous blocks of nearly equal length), and
  for every bin the sums $W_b$ and $A_b$ of eq. {eq}`binning_weights` are
  kept. The total number of binned samples used by the jackknife is
  $N_J = $ `num_mcmc_bin_blocks` $\cdot$ `mpi_size` $\cdot$
  `number_of_walkers`. The energy is the **weighted** delete-one-bin estimator
  eq. {eq}`jk_weighted_block` with the AS weights $w_{\mathrm{AS}}$
  (all equal to 1 when `epsilon_AS = 0.0`), followed by
  eq. {eq}`jk_weighted_mean` and eq. {eq}`jk_weighted_variance`. The energy
  variance is estimated per replica as
  $\mathrm{Var}_b = \bigl(A^{(2)}_{\mathrm{total}}-A^{(2)}_b\bigr)/\bigl(W_{\mathrm{total}}-W_b\bigr) - |O_b^{\mathrm{J}}|^{2}$,
  where $A^{(2)}$ are the binned sums of $w\,|e_L|^{2}$, and is averaged and
  given an error bar with the same jackknife formulas.

- **GFMC (`job_type = "lrdmc-bra"` and `"lrdmc-tau"`).** Walkers branch, so
  they are not independent. At every measurement the driver first reduces over
  all walkers and MPI ranks: it stores the mean weight $\bar w_n$ and the
  weighted population average $\bar e_L(x_n)$ of
  eq. {eq}`lrdmc_local_energy_weighted` (see {doc}`lrdmc`). Each stored
  $\bar e_L(x_n)$ is then weighted by the global population factor
  $\bar G_n^{\,p}$ of eq. {eq}`lrdmc_global_population_factor`, the product of
  the $p$ preceding mean weights, where $p$ is `num_gfmc_collect_steps`; the
  first $p$ measurements, which have no complete history, are dropped. Only
  then is the one-dimensional time series binned into `num_gfmc_bin_blocks`
  bins -- a **global** number, not multiplied by MPI size or walker count --
  and analysed with the weighted jackknife eq. {eq}`jk_weighted_block` with
  $w = \bar G_n^{\,p}$.

**Commands and keys.** The jobs print the energy at the end of a run using the
warm-up and bin settings of the input file. You can re-analyse a finished run
from its checkpoint with different settings, without rerunning:

```bash
jqmc-tool mcmc compute-energy restart.h5 -w 50 -b 20
jqmc-tool lrdmc compute-energy restart.h5 -w 50 -b 20 -c 5
```

Here `-w` is the number of discarded warm-up measurements, `-b` the number of
bins (per rank and walker for MCMC, global for LRDMC) and `-c` the value of
`num_gfmc_collect_steps`. The output line `E = <mean> +- <error> Ha.` is
$\mu^{\mathrm{Re}} \pm \sigma_{\mathrm{Re}}$ of eq. {eq}`jk_complex_report`.
`jqmc-tool mcmc compute-energy` refuses bin and warm-up settings that would
leave empty bins or an undefined result; `jqmc-tool lrdmc compute-energy`
refuses values below the minima in the table.

| Quantity | MCMC (`[mcmc]`, `[vmc]`) | GFMC (`[lrdmc-bra]`, `[lrdmc-tau]`) |
|---|---|---|
| Warm-up measurements | `num_mcmc_warmup_steps`; no minimum, chosen from the data | `num_gfmc_warmup_steps`; minimum 25 (default 25) |
| Number of bins | `num_mcmc_bin_blocks`, per rank and walker; no minimum, chosen from the data | `num_gfmc_bin_blocks`, global; minimum 10 (default 10) |
| History length $p$ | -- | `num_gfmc_collect_steps`; minimum 5 (default 5) |

**Checking the bin length.** If the time bins are shorter than the
autocorrelation length, the error bar is underestimated; if there are too few
of them, the error bar itself is noisy. Choose the bin count with the
long-block-to-short-block scan of {ref}`sec:error_analysis_bins`, which also
gives the uncertainty of the error bar and the checks to make before trusting
it. The minima in the table only prevent an undefined or extremely noisy
jackknife; they are not production recommendations. Select the warm-up `-w`
first, by increasing it until the mean no longer drifts: no choice of bins can
repair an incompletely equilibrated trajectory.

The same weighted delete-one-bin layout is used for forces, stress and the
reweighted estimators of the parameter optimization (see
{ref}`sec:opt_wf`). Keeping the total sums and subtracting one bin keeps the
post-processing cost $\mathcal{O}(N_J)$ even when $N_J$ is large.

(sec:error_analysis_bins)=

## Choosing the number of bins

The jackknife of the previous sections assumes that the time bins are
approximately independent. Whether they are depends on the bin length in the
units of the stored time series, and this length is set only indirectly, by
the number of bins. This section explains how to choose that number, how
uncertain the resulting error bar is, and when it can be trusted. jQMC does
not estimate an autocorrelation time and does not choose the bin count
automatically; the procedure below is done by re-analysing a checkpoint with
`jqmc-tool`.

### What is counted

Several different counts enter the error analysis. They must not be
confused, because the two failure modes below depend on different ones.

| Quantity | Symbol | MCMC (`mcmc`; one `vmc` optimization step) | GFMC (`lrdmc-bra`, `lrdmc-tau`) |
|---|---|---|---|
| updates between two stored measurements | -- | `num_mcmc_per_measurement` Metropolis steps | one branching interval: `num_mcmc_per_measurement` projections (`lrdmc-bra`) or imaginary time `tau` (`lrdmc-tau`) |
| stored measurements | $N_{\mathrm{st}}$ | in **each** chain, i.e. each MPI rank and walker: for `mcmc`, `num_mcmc_steps` summed over restarts; for `vmc`, the `num_mcmc_steps` of the current optimization step only | `num_mcmc_steps` (summed over restarts) values of **one** population-reduced series |
| retained measurements | $N$ | $N_{\mathrm{st}} - w$ per chain, $w$ = `-w` | $N_{\mathrm{st}} - p - w$, $p$ = `-c`, $w$ = `-w` |
| bin count (input) | $B$ | `-b` / `num_mcmc_bin_blocks`, per chain | `-b` / `num_gfmc_bin_blocks`, global |
| measurements per bin | $L$ | $\approx N/B$ | $\approx N/B$ |
| nominal jackknife blocks | $N_J$ | $B\,R\,W$ ($R$ MPI ranks, $W$ walkers per rank) | $B$ |

Bins are contiguous in time. When $B$ does not divide $N$, the first
$N \bmod B$ bins of each chain contain $\lceil N/B\rceil$ measurements and the
others $\lfloor N/B\rfloor$ (NumPy `array_split`). jQMC sums $w$ and $wO$ in
each bin, so bins of unequal length simply carry unequal weight. The number of
**effectively independent** blocks can be smaller than $N_J$: when the bin
weights $W_b$ are very unequal (large GFMC population-factor fluctuations, or
strongly varying Attaccalite-Sorella weights), a few bins dominate the
weighted mean and the error bar behaves as if fewer blocks had been used.

**Fixed-wavefunction histories and VMC batches.** An `mcmc`, `lrdmc-bra` or
`lrdmc-tau` run samples one fixed wavefunction, and a continuation with
`restart = true` appends its measurements to the stored history (see
{ref}`api_reference_cli_restart`); the whole history is one stationary
trajectory that can be re-binned from the checkpoint and extended. A `vmc`
optimization is different: every optimization step samples a new batch of
`num_mcmc_steps` measurements with the current wavefunction, and every
accepted parameter update starts a new batch with empty sample storage. The
counting and bin formulas of this section apply **within one optimization
step's batch** (its own warm-up, $B$, $L$ and $N_J$, as described for
`num_mcmc_bin_blocks` in {doc}`/api_reference_cli`); samples from different
optimization steps belong to different wavefunctions and must never be pooled
into one binning analysis. A `vmc` checkpoint written after an accepted
update holds no samples of the previous steps to re-bin, so the checkpoint
scan and the run-extension advice of the procedure below apply to `mcmc` and
LRDMC trajectories. Error bars of a final wavefunction are obtained from a
separate `mcmc` run with that wavefunction.

The time-block length $L$ controls the **autocorrelation bias**: it must be
long compared with the correlation time of the series, measured in stored
measurements. The block count $N_J$ controls the **statistical uncertainty of
the error bar** itself. More bins mean shorter blocks.

#### MCMC: blocks per chain

For `mcmc`, and within one `vmc` optimization step, every MPI rank and walker
carries an independent chain,
and each chain is cut into $B$ bins, so $N_J = B\,R\,W$ while
$L \approx N/B$ is set by $B$ alone. Adding walkers or ranks increases $N_J$
and therefore reduces the noise of the error bar, but it does **not** make a
too-short time block longer. Increasing `num_mcmc_per_measurement` inserts
more Metropolis updates between stored measurements and usually shortens the
autocorrelation time measured in stored measurements, but it does not prove
that successive measurements or bins are independent.

Worked example (counting only): 4 MPI ranks, `number_of_walkers = 8`,
`num_mcmc_steps = 1000` and `-w 50` give $R\,W = 32$ chains with
$N = 950$ retained measurements each.

| `-b` | measurements per bin $L$ | $N_J = B\,R\,W$ | ideal relative uncertainty of the error bar, eq. {eq}`jk_error_of_error` |
|---:|---:|---:|---:|
| 1 | 950 | 32 | 12.7% |
| 10 | 95 | 320 | 4.0% |
| 50 | 19 | 1600 | 1.8% |
| 100 | 50 bins of 10, 50 bins of 9 | 3200 | 1.3% |

At `-b 100` the error bar looks very precise, but it is meaningless if 9-10
measurements are shorter than the correlation time: the last column assumes
independent blocks. Conversely, `-b 1` (one block per chain, the whole
retained chain) has the longest possible blocks; with many walkers it still
gives a usable $N_J$, which makes it a useful long-block point of the scan
below, provided every chain is equilibrated after the warm-up.

#### GFMC: one global series

For `lrdmc-bra` and `lrdmc-tau` the walkers are coupled by branching. jQMC
therefore first reduces over all walkers and MPI ranks at every measurement
(see {ref}`sec:error_analysis_practice`), drops the first $p$ =
`num_gfmc_collect_steps` measurements, which have no complete weight history,
then the $w$ warm-up measurements, and bins the remaining **single** time
series into $B$ bins. Hence $N_J = B$: neither the MPI ranks nor the walkers
multiply the number of jackknife blocks (MPI only distributes the arithmetic).
The trade-off between autocorrelation bias and the noise of the error bar is
therefore much tighter than for MCMC.

Each retained measurement is weighted by the population factor
$\bar G_n^{\,p}$ of eq. {eq}`lrdmc_global_population_factor`, the product of
the $p$ preceding mean weights. Neighbouring factors share $p-1$ of these
weights, and branching correlates the population further. A bin must
therefore be long compared with the **effective** correlation length of the
weighted series $\bar G_n^{\,p}\,\bar e_L(x_n)$; a bin only slightly longer
than $p$ is not enough. Keep `-c` at the value used for the run while scanning
`-b`: changing $p$ changes the estimator (the population-control correction),
not only its error analysis.

The enforced minimum of 10 bins prevents an undefined or extremely small
jackknife sample. It does not make a 10-block error bar precise: under the
ideal Gaussian assumptions of eq. {eq}`jk_error_of_error`, 10 blocks already
mean about 24% relative uncertainty of the error bar, before weights,
branching or heavy tails are taken into account.

Worked example (counting only): 4 MPI ranks with 64 walkers each,
`num_mcmc_steps = 4000`, `-c 5` and `-w 50` give one series of
$N = 4000 - 5 - 50 = 3945$ retained measurements. The 256 walkers do not
enter $N_J$.

| `-b` | measurements per bin $L$ | $N_J$ | ideal relative uncertainty of the error bar |
|---:|---:|---:|---:|
| 10 | 395 (5 bins), 394 (5 bins) | 10 | 23.6% |
| 20 | 198 (5 bins), 197 (15 bins) | 20 | 16.2% |
| 40 | 99 (25 bins), 98 (15 bins) | 40 | 11.3% |
| 80 | 50 (25 bins), 49 (55 bins) | 80 | 8.0% |
| 160 | 25 (105 bins), 24 (55 bins) | 160 | 5.6% |

### Relation to conventional reblocking

Consider a linear observable, unit weights and $N_J$ bins of equal length $L$
with bin means $\bar x_b$ and overall mean
$\bar x = N_J^{-1}\sum_b \bar x_b$. The delete-one-bin replica of
eq. {eq}`jk_unweighted_sample` is then

$$
\bar x_{(b)} = \frac{N_J\,\bar x - \bar x_b}{N_J - 1},
\qquad
\bar x_{(b)} - \frac{1}{N_J}\sum_{b'}\bar x_{(b')}
= -\frac{\bar x_b - \bar x}{N_J - 1},
$$ (jk_block_replica)

and eq. {eq}`jk_variance` becomes

$$
\sigma^2
= \frac{N_J - 1}{N_J}\sum_{b=1}^{N_J}
  \frac{(\bar x_b - \bar x)^2}{(N_J - 1)^2}
= \frac{1}{N_J}\,\frac{1}{N_J - 1}\sum_{b=1}^{N_J}(\bar x_b - \bar x)^2
= \frac{s_{\bar x_b}^2}{N_J}.
$$ (jk_equals_reblocking)

For a linear mean the delete-one-bin jackknife is therefore exactly the
standard error computed from the bin means, i.e. the reblocking (binning or
blocking) method introduced by Flyvbjerg and Petersen [FP89] and used in the
binning analysis of Ambegaokar and Troyer [AT10]; Wolff [Wol04, App. B, eqs. (59)-(65)]
gives the same relation between ordinary and jackknife bins. The plateau
sought below is the ordinary reblocking plateau. The reason to keep the
jackknife is that the same delete-one-bin construction propagates the joint
fluctuations of numerator and denominator through non-linear estimators:
weighted means, the energy variance, forces and stress. With the unequal bin
lengths of `array_split` the unit-weight identity holds only approximately,
because each replica is then the length-weighted mean of the remaining bins.

### Why the error estimate reaches a plateau

For a stationary series with variance $\sigma_O^2$ and normalized
autocorrelation $\rho(t)$, the variance of the mean of $N$ consecutive
measurements is, for $N$ much longer than the correlation time,

$$
\operatorname{Var}(\bar O) \simeq \frac{2\tau_{\mathrm{int}}\,\sigma_O^2}{N},
\qquad
\tau_{\mathrm{int}} = \frac{1}{2} + \sum_{t=1}^{\infty}\rho(t),
$$ (tau_int_variance)

so that only about $N/(2\tau_{\mathrm{int}})$ measurements are effectively
independent [Wol04, eqs. (23)-(25)]; without correlations
$2\tau_{\mathrm{int}} = 1$. (Ambegaokar and Troyer [AT10, eqs. (17)-(18)]
write the same result as $(1 + 2\tau_A)\,\mathrm{Var}A/M$ with
$\tau_A = \tau_{\mathrm{int}} - 1/2$.) The same formula applied to one bin
gives $\operatorname{Var}(\bar O_b) \simeq 2\tau_{\mathrm{int}}\sigma_O^2/L$
once $L \gg \tau_{\mathrm{int}}$. The bins are then approximately independent,
and the jackknife estimate of the error of the overall mean is

$$
\sigma^2 \simeq \frac{\operatorname{Var}(\bar O_b)}{N_J}
\simeq \frac{2\tau_{\mathrm{int}}\,\sigma_O^2}{L\,N_J},
$$ (jk_plateau)

which does not depend on how the retained trajectory is cut, because $L\,N_J$
is the total number of retained measurements in all chains. Increasing $L$
while decreasing $B$ then no longer changes the estimate systematically: this
is the plateau. For shorter bins the within-bin sum misses part of the
correlation; for an exponentially decaying correlation with time scale $\tau$
the bin-mean variance, and hence $\sigma^2$, is too small by a relative amount
of about $\tau/L$ [Wol04, eq. (62)], i.e. the error bar by about $\tau/(2L)$.
More bins do not reduce this bias: they only reduce the statistical noise of
the estimate [Wol04, App. B]. A scan over bin lengths therefore shows the
error bar rising with $L$ and levelling off once $L$ is several correlation
times [AT10, Sec. IV.D].

This is an asymptotic argument. Weighted bins, non-linear estimators, unequal
bin lengths, population correlations in GFMC and a small number of blocks all
add finite-sample corrections, and the correlation time depends on the
observable [AT10]. The plateau is a diagnostic to be inspected, not a theorem
about a particular run.

### Uncertainty of the error bar

Even when the blocks are independent, the error bar is estimated from a
finite number of them and is itself a random quantity. For $N_J$ independent
Gaussian block means with $\nu = N_J - 1$, the sample variance $s^2$ of the
block means satisfies $\nu s^2/\sigma_b^2 \sim \chi^2_\nu$, so $s^2$ has
relative standard deviation $\sqrt{2/\nu}$. By the delta method the error
bar, which is proportional to $s$, has the approximate relative uncertainty

$$
\frac{\operatorname{sd}(\hat\sigma)}{\operatorname{E}(\hat\sigma)}
\simeq \frac{1}{\sqrt{2(N_J - 1)}}.
$$ (jk_error_of_error)

The same $1/\sqrt{2N_J}$ term appears as the statistical error of the
binning error estimate in Wolff's analysis [Wol04, eq. (46)]. Reference values
(the "exact" column is the Gaussian result
$\sqrt{1 - c_4^2}/c_4$ with $c_4 = \operatorname{E}(s)/\sigma_b$):

| independent blocks $N_J$ | eq. {eq}`jk_error_of_error` | exact for Gaussian blocks |
|---:|---:|---:|
| 10 | 23.6% | 23.9% |
| 20 | 16.2% | 16.3% |
| 30 | 13.1% | 13.2% |
| 50 | 10.1% | 10.1% |
| 100 | 7.1% | 7.1% |

For example, an error bar of $0.0010$ Ha obtained from 10 blocks is only
known to about $\pm 0.0002$ Ha; quote it with one significant digit and do
not read a difference of $2\sigma$ between two such results as decisive.

Read this as a scale for judging a plateau, not as an exact confidence
interval:

- error bars computed with different bin counts from the **same** trajectory
  are strongly correlated, so neighbouring points of a scan are not
  independent measurements, and their scatter understates the uncertainty of
  any one of them;
- the estimated standard deviation has a small finite-$N_J$ downward bias
  ($1 - c_4$, about 2.7% for 10 blocks);
- for weighted or non-linear estimators the delete-one-bin replicas need not
  be Gaussian, the jackknife bias need not vanish at finite $N_J$, and for
  non-linear functions of independent samples the jackknife variance tends to
  be biased upwards [ES81]; strong weight imbalance reduces the effective
  number of blocks below $N_J$;
- for heavy-tailed quantities eq. {eq}`jk_error_of_error` can be optimistic or
  inapplicable. Under plain $|\Psi|^2$ sampling, outliers of the energy
  estimate are more likely than a Gaussian suggests, and the sampled
  quantities behind the variance of the local energy and the local
  Hellmann-Feynman force have infinite variance, so the central limit theorem
  on which this section relies does not hold for them [Tra08]. The
  Attaccalite-Sorella reweighting (`epsilon_AS > 0`, see {doc}`vmc`) was
  introduced to obtain forces with finite variance [AS08].

The two failure modes can be balanced. With the autocorrelation bias
$\approx \tau/(2L)$ and the statistical term $\approx 1/\sqrt{2N_J}$, the
total relative error of a single-chain binning error bar is smallest at
$L \simeq \tau\,(2N/\tau)^{1/3}$, where it is
$\tfrac{3}{2}(2N/\tau)^{-1/3}$ [Wol04, eqs. (46)-(47)]. For a series of
$N = 100\tau$, $1000\tau$ and $10\,000\tau$ retained measurements this gives
about 17, 79 and 368 bins and a total uncertainty of the error bar of about
26%, 12% and 5.5%. These numbers assume exponential correlations and a known
$\tau$, which jQMC does not compute; they show why a longer run is the only
way to obtain both long blocks and many of them. In MCMC, independent chains
raise $N_J$ without shortening the blocks, so with many walkers longer time
blocks (fewer bins per chain) are affordable.

Alternatives that jQMC does not implement are the explicit summation of the
autocorrelation function (the $\Gamma$ method of Wolff [Wol04]) and automated
blocking with a statistical stopping test (Jonsson [Jon18]).

### Procedure

Use $B$ for the bin count passed with `-b` and $L$ for the measurements per
bin; more bins mean shorter blocks. The procedure re-analyses the checkpoint
of a fixed-wavefunction `mcmc`, `lrdmc-bra` or `lrdmc-tau` trajectory; it does
not apply across `vmc` optimization steps (see "Fixed-wavefunction histories
and VMC batches" above).

1. **Select the warm-up first.** Increase `-w` until the mean no longer shows
   a systematic drift. Changing the bin count cannot repair incomplete
   equilibration.
2. **Scan from long blocks to short blocks.** Re-analyse the same checkpoint
   with increasing $B$, starting from the smallest accepted bin count (the
   longest blocks), so that the scan moves from the safe side toward the side
   on which the error bar is biased low. A factor of 2 between steps is
   convenient:

   ```bash
   for b in 10 20 40 80 160; do
     echo "-b $b"
     jqmc-tool lrdmc compute-energy restart.h5 -w 50 -b $b -c 5 | grep -E "^(Total number|E =)"
   done
   ```

   For MCMC use `jqmc-tool mcmc compute-energy restart.h5 -w 50 -b $b`; with
   many walkers, bin counts below 10 (down to `-b 1`, one block per chain)
   can be added as long-block diagnostic points, because the nominal block
   count is still $B\,R\,W$ (see the MCMC example above). A bin count that
   would leave a bin empty is rejected with an error.
3. **Record, for every point**, the retained measurements $N$ (per chain for
   MCMC), $B$, the block length $L \approx N/B$, the printed
   `Total number of binned samples` ($N_J$), the mean and the error bar.
   Compare scans in terms of $L$, not of `-b`: the same `-b` means a different
   block length for a different run length or warm-up.
4. **Identify the plateau.** In the long-block region the error bars should
   agree within their uncertainty (eq. {eq}`jk_error_of_error`, evaluated
   with the $N_J$ of the longer-block point) and should show no downward trend
   as the blocks are shortened. A steady decrease toward short blocks is the
   autocorrelation bias.
5. **Choose the largest bin count that is still inside the plateau** (the
   shortest blocks that are still long enough). It keeps the most
   approximately independent blocks and therefore the least noisy error bar,
   without accepting the autocorrelation-driven underestimate. Never choose
   the bin count that happens to give the smallest error bar.
6. **If no plateau is resolved** before the number of blocks becomes too
   small to judge (the error bar still rises at the longest blocks you can
   form, or the points scatter by more than their expected uncertainty), the
   trajectory is too short to support a defensible error bar. Extend the run
   (an `mcmc` or LRDMC run can be continued from the checkpoint with
   `restart = true`, which appends to the same fixed-wavefunction history;
   for `vmc`, increase `num_mcmc_steps` per optimization step) and repeat
   the scan; Ambegaokar and Troyer draw the same conclusion when the binning
   analysis does not converge [AT10, Sec. IV.D].
7. **Repeat the check for each reported observable.** An energy plateau does
   not by itself validate the binning of the variance, a force or the stress,
   which can have a longer correlation time or heavier tails.

### Observables

- **Energy.** `jqmc-tool mcmc compute-energy` and
  `jqmc-tool lrdmc compute-energy` re-bin the stored local energies with any
  `-b`/`-w` (and `-c`); this is the observable to scan first.
- **Energy variance.** Printed by the `jqmc` drivers at the run's own bin
  setting, with the same delete-one-bin replicas as the energy. Its sampling
  distribution is heavy-tailed under $|\Psi|^2$ sampling [Tra08], so its error
  bar is the least reliable one; do not use it to validate the binning of the
  energy.
- **Forces.** `jqmc-tool mcmc compute-force` and
  `jqmc-tool lrdmc compute-force` accept the same `-b`/`-w` (and `-c`), so the
  scan can be repeated per force component. Force estimators can have heavy
  tails (see the regularizations `epsilon_AS` in {doc}`vmc` and `epsilon_PW`
  in {doc}`derivatives`); a plateau in the energy does not establish one in
  the forces.
- **Stress and pressure.** The drivers print the stress tensor and the
  hydrostatic pressure with the run's own `num_mcmc_bin_blocks` or
  `num_gfmc_bin_blocks`; `jqmc-tool` does not re-bin a merged single
  checkpoint for stress (see {doc}`/api_reference_cli_tool`). Choose these
  input values from an energy and force scan, and treat the stress error bar
  with the caveats above. For `mcmc` and LRDMC the bin settings are mutable
  on restart and a continuation reports the whole stored history (see
  {ref}`api_reference_cli_restart`), so a continuation run prints the stress
  of the extended history with its new bin setting.

### When to trust an error bar

An error bar is defensible when (i) the mean no longer drifts with `-w`;
(ii) a plateau is visible in a long-to-short block scan and the chosen $B$
lies inside it; (iii) the number of blocks at that point gives an
uncertainty of the error bar (eq. {eq}`jk_error_of_error`) that is acceptable
for the comparison you make; (iv) the weights are reasonably balanced (for
LRDMC see the survived-walkers ratio in {doc}`lrdmc`; for reweighted MCMC the
average Attaccalite-Sorella weight in {doc}`vmc`); and (v) for the quantity
actually reported, not only the energy. The most direct check is to repeat the
run with a different `mcmc_seed`: independent runs must agree within their
error bars [AT10, Sec. IV.A; Wol04, eqs. (26)-(28)].

### References

- [AS08] C. Attaccalite and S. Sorella, "Stable liquid hydrogen at high
  pressure by a novel ab initio molecular-dynamics calculation,"
  *Phys. Rev. Lett.* **100**, 114501 (2008).
  [doi:10.1103/PhysRevLett.100.114501](https://doi.org/10.1103/PhysRevLett.100.114501)
- [AT10] V. Ambegaokar and M. Troyer, "Estimating errors reliably in Monte
  Carlo simulations of the Ehrenfest model," *Am. J. Phys.* **78**, 150-157
  (2010). [doi:10.1119/1.3247985](https://doi.org/10.1119/1.3247985),
  [arXiv:0906.0943](https://arxiv.org/abs/0906.0943)
- [ES81] B. Efron and C. Stein, "The jackknife estimate of variance,"
  *Ann. Statist.* **9**, 586-596 (1981).
  [doi:10.1214/aos/1176345462](https://doi.org/10.1214/aos/1176345462)
- [FP89] H. Flyvbjerg and H. G. Petersen, "Error estimates on averages of
  correlated data," *J. Chem. Phys.* **91**, 461-466 (1989).
  [doi:10.1063/1.457480](https://doi.org/10.1063/1.457480)
- [Jon18] M. Jonsson, "Standard error estimation by an automated blocking
  method," *Phys. Rev. E* **98**, 043304 (2018).
  [doi:10.1103/PhysRevE.98.043304](https://doi.org/10.1103/PhysRevE.98.043304)
- [Tra08] J. R. Trail, "Heavy-tailed random error in quantum Monte Carlo,"
  *Phys. Rev. E* **77**, 016703 (2008).
  [doi:10.1103/PhysRevE.77.016703](https://doi.org/10.1103/PhysRevE.77.016703),
  [arXiv:0909.5505](https://arxiv.org/abs/0909.5505)
- [Wol04] U. Wolff, "Monte Carlo errors with less errors,"
  *Comput. Phys. Commun.* **156**, 143-153 (2004).
  [doi:10.1016/S0010-4655(03)00467-3](https://doi.org/10.1016/S0010-4655(03)00467-3),
  [arXiv:hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017)
