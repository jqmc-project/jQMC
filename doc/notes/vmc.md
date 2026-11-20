(vmc_tags)=

# Variational Monte Carlo (VMC)

**In jQMC.** The theory on this page is used by two job types of the `jqmc`
command. `job_type = "mcmc"` samples $|\Psi|^{2}$ for a fixed wave function
and measures the energy (and, on request, forces, stress or parameter
derivatives); its options are in the `[mcmc]` section. `job_type = "vmc"`
optimizes the wave-function parameters; each optimization step runs a short
MCMC sampling with the same sampler and then updates the parameters (see
{doc}`wavefunction_optimization`); its options are in the `[vmc]` section.
The sampler keys shared by both sections are:

| Symbol on this page | TOML key | Default | Meaning |
|---|---|---|---|
| $\Delta t$ | `Dt` | `2.0` (bohr) | step-size scale of the electron move |
| $\varepsilon$ | `epsilon_AS` | `0.0` | Attaccalite-Sorella cutoff; `0.0` switches the regularization off |
| -- | `num_mcmc_per_measurement` | `40` | Metropolis updates between two stored measurements |
| -- | `num_mcmc_warmup_steps` | `0` | stored measurements discarded as equilibration |
| -- | `num_mcmc_bin_blocks` | `5` | bins per MPI rank and walker for the error bar (see {ref}`sec:error-analysis`) |

## Variational Monte Carlo

The expectation value of the energy for a trial wave function $\Psi$ can be written as:

```{math}
:label: eq-vmc-energy
\langle E \rangle
= \frac{\displaystyle \int \mathrm{d}\mathbf{x}\,|\Psi(\mathbf{x})|^{2}\,\displaystyle\frac{\hat{\mathcal{H}}\Psi(\mathbf{x})}{\Psi(\mathbf{x})}}
       {\displaystyle \int \mathrm{d}\mathbf{x}\,|\Psi(\mathbf{x})|^{2}}
```

The squared-modulus form is used throughout: for real $\Psi$ it reduces to $\Psi^{2}$, and for complex $\Psi$ at non-$\Gamma$ twist (see {doc}`complex_gradients`) it is the correct probability weight. This is equivalently expressed as an average over the local energy $e_L(\mathbf{x})$ with probability density $\pi(\mathbf{x})$:

```{math}
\langle E \rangle = \int \mathrm{d}\mathbf{x}\;e_L(\mathbf{x})\,\pi(\mathbf{x})
```

Here:

* $\mathbf{x}=(\mathbf{r}_1\sigma_1,\mathbf{r}_2\sigma_2,\dots,\mathbf{r}_N\sigma_N)$ denotes all electron coordinates and spins.
* The **local energy** is defined by:

  ```{math}
  e_L(\mathbf{x}) = \frac{\hat{\mathcal{H}}\Psi(\mathbf{x})}{\Psi(\mathbf{x})}
  ```

  For Hermitian $\hat{\mathcal{H}}$ and real $\Psi$, $e_L$ is real. For complex $\Psi$, $e_L$ is **complex per walker**; only the expectation $\langle e_L\rangle$ under $|\Psi|^{2}$ is real, and the reported physical energy is $\mathrm{Re}\,\langle e_L\rangle$ (see eq. {eq}`eq-eval-observable` below).

* The **sampling probability** is:

  ```{math}
  \pi(\mathbf{x}) = \frac{|\Psi(\mathbf{x})|^{2}}{\displaystyle \int \mathrm{d}\mathbf{x}'\,|\Psi(\mathbf{x}')|^{2}}
  ```

To evaluate this multidimensional integral stochastically, one generates samples $\{\mathbf{x}_i\}$ according to $\pi(\mathbf{x})$ via Markov Chain Monte Carlo (MCMC) and computes the Monte Carlo estimator:

```{math}
:label: eq-eval-observable
E_{\mathrm{MCMC}} = \mathrm{Re}\bigl\langle e_L(\mathbf{x})\bigr\rangle_{\pi} \approx \mathrm{Re}\,\biggl(\frac{1}{M} \sum_{i=1}^M e_L(\mathbf{x}_i)\biggr)
```

The $\mathrm{Re}(\cdot)$ wrapper is the identity on the real path and projects out the finite-sample imaginary noise on the complex path. The complementary $\mathrm{Im}\langle e_L\rangle$ is kept as a diagnostic: the MCMC sampler (used by both `mcmc` and `vmc` jobs) logs a warning when $|\mathrm{Im}\langle e_L\rangle|$ exceeds twice the error bar of the real part (see {ref}`sec:error-analysis`). A short or poorly equilibrated run can trigger this warning by chance; if it persists in a long, equilibrated run, suspect the complex wave function or Hamiltonian (for example a wrong conjugation or a missing Bloch phase).

For $M$ independent samples the associated statistical error would be:

```{math}
\sqrt{\frac{\mathrm{Var}[e_L(\mathbf{x}_i)]}{M}}
```

with variance

```{math}
\mathrm{Var}[e_L] = \bigl\langle |e_L|^{2}\bigr\rangle - \bigl|\langle e_L\rangle\bigr|^{2}
```

(modulus-square form, reduces to the algebraic variance for real $e_L$). Consecutive MCMC samples are correlated, so this naive formula underestimates the error. jQMC does not estimate an autocorrelation time; it obtains the error bar by binning the samples and applying the jackknife, as described in {ref}`sec:error-analysis`.

If $\Psi$ is an exact eigenfunction of $\hat{\mathcal{H}}$ with eigenvalue $E_0$, then $e_L(\mathbf{x})=E_0$ for all $\mathbf{x}$, giving zero variance and $E_{\mathrm{MCMC}} = E_0$ exactly (the **zero-variance property**). For complex $\Psi$ this holds with $E_0 \in \mathbb{R}$ recovered as $\mathrm{Re}\,e_L$ and $\mathrm{Im}\,e_L \equiv 0$ at the exact eigenstate.

By the **variational theorem**, the estimator provides an upper bound to the true ground-state energy $E_0$.  Introducing variational parameters $\boldsymbol{\alpha}=(\alpha_1,\dots,\alpha_p)$ into the trial wave function $\Psi(\mathbf{x},\boldsymbol{\alpha})$, one defines:

```{math}
E_{\mathrm{VMC}}(\boldsymbol{\alpha}) = \int \mathrm{d}\mathbf{x}\;e_L(\mathbf{x},\boldsymbol{\alpha})\,\pi(\mathbf{x},\boldsymbol{\alpha}) \ge E_0
```

This constitutes the Variational Monte Carlo (VMC) framework.

The optimization of $\boldsymbol{\alpha}$ is challenging due to a complex energy landscape with statistical noise. jQMC evaluates the required logarithmic and local-energy derivatives analytically and updates the parameters with **stochastic reconfiguration** (SR); with the default `[vmc] optimizer_kwargs` (`use_lm = true`, `lm_subspace_dim = 0`) the SR step is accelerated by the linear method (see {doc}`wavefunction_optimization`). The analytic derivative backends are described in {doc}`analytic_derivatives`.

### MCMC Sampling and Metropolis-Hastings

jQMC uses a generalized Metropolis-Hastings algorithm to sample $\pi(\mathbf{x})$.  A proposed move from $\vec{x}$ to $\vec{x}'$ is accepted with probability:

```{math}
P(\vec{x}\to\vec{x}') = \min\Bigl[1, \frac{|\Psi_T(\vec{x}')|^2}{|\Psi_T(\vec{x})|^2} \frac{T(\vec{x}'\to\vec{x})}{T(\vec{x}\to\vec{x}')}\Bigr]
```

where $\Psi_T$ is the trial wave function and $T$ are transition kernels. When the AS regularization below is switched on, $\Psi_T$ in this ratio is replaced by the guiding function $\Psi_G$. The acceptance ratio is the modulus-square ratio $|\Psi_T(\vec{x}')/\Psi_T(\vec{x})|^{2}$, which is correct on both the real and the complex path. Concretely, in the source code the per-walker ratio assembly is

```{math}
R_{\mathrm{accept}} = \bigl|R_{\mathrm{AS}}\,R_{\mathrm{WF}}\bigr|^{2}
= \bigl(R_{\mathrm{AS}}\,R_{\mathrm{WF}}\bigr)\bigl(R_{\mathrm{AS}}\,R_{\mathrm{WF}}\bigr)^{*}
```

where $R_{\mathrm{AS}}$ is the AS regularization ratio (real, see below) and $R_{\mathrm{WF}} = \Psi_T(\vec{x}')/\Psi_T(\vec{x})$ is the wave-function ratio (complex for a complex Bloch wave function). For a real wave function the modulus-square form reduces to the algebraic square $R_{\mathrm{AS}}^{2}\,R_{\mathrm{WF}}^{2}$.

In the accelerated scheme, choose one electron and one Cartesian axis
uniformly. Its Gaussian displacement has standard deviation
$\sigma_l=f(\vec r_l)\Delta t$, where $\Delta t$ is the TOML key `Dt`, with

```{math}
f(\vec{r}_l) = \frac{1}{Z_{I(l)}^2} \frac{1 + Z_{I(l)}^2|\vec{r}_l-\vec{R}_{I(l)}|}{1 + |\vec{r}_l-\vec{R}_{I(l)}|}
```

and

```{math}
r'_{l,\gamma} = r_{l,\gamma} + g_l, \quad g_l\sim\mathcal{N}(0,\,[f(\vec{r}_l)\Delta t]^2).
```

The second Gaussian argument is the variance. Here $Z_{I(l)}$ is the
nearest atom's effective charge, and PBC distances use its nearest image.
For hydrogen ($Z=1$), $f=1$ at every distance, including the nuclear limit.
The other two electron coordinates stay unchanged. Detailed balance uses
the forward and reverse one-dimensional Gaussian kernels:

```{math}
T(\vec{x}\to\vec{x}') =
\frac{1}{\sqrt{2\pi (f(\vec{r}_l)\Delta t)^2}} \exp\Bigl[-\frac{|\vec{r}'_l-\vec{r}_l|^2}{2 (f(\vec{r}_l)\Delta t)^2}\Bigr]
```

and similarly for $T(\vec{x}'\to\vec{x})$, giving the ratio:

```{math}
\frac{T(\vec{x}'\to\vec{x})}{T(\vec{x}\to\vec{x}')} = \frac{f(\vec{r}_l)}{f(\vec{r}'_l)} \exp\Bigl[-|\vec{r}_l-\vec{r}'_l|^2 (\frac{1}{2(f(\vec{r}'_l)\Delta t)^2} - \frac{1}{2(f(\vec{r}_l)\Delta t)^2})\Bigr].
```

### Periodic walkers

For periodic systems, walker coordinates are stored **unwrapped**. Only the initial configuration is placed inside the cell; accepted moves are not mapped back. Interparticle distances use the minimum-image convention, and the atomic-orbital evaluator applies cell mapping internally. Do not wrap stored coordinates manually; doing so is unsupported. LRDMC uses the same convention; shared wrap and boundary-continuation conventions are summarized in {ref}`sec:pbc_wrap_conventions`. The detailed formulation is work in progress.

### Reweighting and AS Regularization

jQMC implements the Attaccalite-Sorella (AS) reweighting [C. Attaccalite and S. Sorella, Phys. Rev. Lett. 100, 114501 (2008)] to handle divergences of the local energy and of its derivatives near the nodal surface.  One samples a guiding distribution $\Pi_G \propto |\Psi_G|^{2}$ defined by:

```{math}
\Psi_G(\mathbf{x}) = \frac{R^\varepsilon(\mathbf{x})}{R(\mathbf{x})}\Psi_T(\mathbf{x})
```

with

```{math}
R(\mathbf{x}) = \Bigl(S \sum_{i,j} |(F^{-1})_{ij}|^2\Bigr)^{-\theta_R},
```

where $F$ is the geminal (pairing) matrix of {doc}`hamiltonian` and {doc}`wavefunction`, $\theta_R=3/8$, and $S$ is the smallest squared row or column norm of $F$,

```{math}
S = \min\Bigl(\min_i \sum_j |F_{ij}|^2,\; \min_j \sum_i |F_{ij}|^2\Bigr).
```

$R(\mathbf{x})$ goes to zero as $\mathbf{x}$ approaches the nodal surface, where $F$ becomes singular. The squared Frobenius norm $\sum_{ij}|(F^{-1})_{ij}|^2$ is computed from the inverse that the sampler already keeps up to date, so no extra inversion is needed.  The regularization

```{math}
R^\varepsilon(\mathbf{x}) = \max[R(\mathbf{x}),\varepsilon]
```

ensures non-vanishing guiding weights; $\Psi_G = \Psi_T$ wherever $R(\mathbf{x}) \ge \varepsilon$. Observables are reweighted with $w_{\mathrm{AS}}(\mathbf{x}) = |\Psi_T/\Psi_G|^2 = \bigl(R(\mathbf{x})/R^\varepsilon(\mathbf{x})\bigr)^2 \le 1$:

```{math}
\frac{\langle O(\mathbf{x})\,w_{\mathrm{AS}}(\mathbf{x})\rangle_{\Pi_G}}{\langle w_{\mathrm{AS}}(\mathbf{x})\rangle_{\Pi_G}}.
```

$\varepsilon$ is the TOML key `epsilon_AS` (default `0.0`, which gives $w_{\mathrm{AS}} = 1$ and plain sampling of $|\Psi_T|^{2}$). jQMC does not tune it automatically. At the end of a run the log prints the average weight (`Average of walker weights is ... Ideal is ~ 0.800. Adjust epsilon_AS.`); choose `epsilon_AS` so that $\langle w_{\mathrm{AS}}\rangle\approx0.8$. The regularization matters mainly when forces or parameter derivatives are measured, because their estimators diverge at the nodal surface.


(sec:vmc_basis_opt)=

### AO Basis Exponent and Coefficient Optimization

jQMC supports variational optimization of Gaussian-type orbital (GTO) basis exponents and contraction coefficients for AOs used by the three-body Jastrow factor and the geminal pairing function. The detailed formulation is work in progress.

The four optimization flags are:

| Flag | Target |
|---|---|
| `opt_J3_basis_exp` | Exponents of J3 AOs |
| `opt_J3_basis_coeff` | Contraction coefficients of J3 AOs |
| `opt_lambda_basis_exp` | Exponents of geminal AOs |
| `opt_lambda_basis_coeff` | Contraction coefficients of geminal AOs |

Radial parameters are shared per shell: the optimizer sees one exponent and one coefficient per shell primitive, not one per AO. Shell-mates remain equal after updates. All selected basis parameters are optimized simultaneously; there is no signal-to-noise filter or parameter-count selection.

#### Restrictions

* For an `AGP_data` input, `opt_lambda_basis_exp` and `opt_lambda_basis_coeff` cannot be combined with a positive `fix_geminal_rank`. The rank conversion uses an AO overlap matrix that would be invalidated by changing the AO radial basis. In a TOML input, `fix_geminal_rank = 0` is the sentinel for no rank conversion.
