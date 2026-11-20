(sec:derivatives)=

# Derivatives of total energy
## Overview

Two families of energy derivatives are required by the QMC workflow
implemented in jQMC:

- derivatives with respect to the **variational parameters**
  $\boldsymbol{\alpha}$ of the trial wave function -- the building
  block of the SR/LM parameter optimizer;
- derivatives with respect to **nuclear coordinates**
  $\mathbf{R}_{I}$ -- i.e. *atomic forces*, the building block of
  geometry optimization, ab-initio molecular dynamics, and the
  generation of training data for machine-learning interatomic
  potentials.

Both families are computed by applying the chain rule
systematically through the code that evaluates the trial wave
function and the local energy. Production calculations use closed-form
(analytic) derivative kernels. A combination of wave function and
Hamiltonian for which no analytic kernel exists raises an error; jQMC never
silently falls back to automatic differentiation. Automatic differentiation
(AD) with JAX defines the reference formulation below and is used to
validate the analytic kernels; see {doc}`analytic_derivatives`.

A third family, the derivative with respect to a homogeneous strain of the
simulation cell (stress tensor and pressure), is available for periodic
systems; see {ref}`sec:stress_pressure`.

**Corresponding input keys.** The derivatives on this page are switched on
by the following TOML keys (defaults in parentheses):

| Section | Key | Meaning |
|---|---|---|
| `[mcmc]` | `atomic_force` (`false`) | atomic forces, {ref}`subsec:forces_mcmc` |
| `[mcmc]` | `use_swct` (`true`) | space-warp terms in the MCMC force |
| `[mcmc]` | `compute_stress` (`false`) | stress tensor and pressure (periodic only) |
| `[mcmc]` | `parameter_derivatives` (`false`) | store the per-sample $O_k$ of eq. {eq}`O_k_def` |
| `[lrdmc-bra]`, `[lrdmc-tau]` | `atomic_force` (`false`) | atomic forces, {ref}`subsec:forces_lrdmc` |
| `[lrdmc-bra]`, `[lrdmc-tau]` | `use_swct` (`false`) | space-warp terms in the LRDMC force |
| `[lrdmc-bra]`, `[lrdmc-tau]` | `epsilon_PW` (`0.0`, off) | Pathak-Wagner width $\epsilon$ in bohr |
| `[lrdmc-bra]`, `[lrdmc-tau]` | `compute_stress` (`false`) | stress tensor and pressure (periodic only) |

The variational-parameter derivatives used by the optimizer are computed by
`job_type = "vmc"` automatically. Throughout this page $\Psi$ and $\Psi_T$
both denote the trial (guiding) wave function.

Algorithmic / automatic differentiation (AD) provides the reference
formulation and comes in two complementary modes
[Griewank and Walther, *Evaluating Derivatives*, SIAM (2008)](https://doi.org/10.1137/1.9780898717761):

- **Forward-mode AD (FAD)** propagates derivatives from inputs to
  outputs and computes a Jacobian-vector product (JVP) at the cost
  of one forward pass per input direction.
- **Adjoint-mode AD (AAD)** propagates derivatives from outputs to
  inputs and computes a vector-Jacobian product (VJP) at the cost
  of one forward + one reverse pass per output direction,
  *independent of the number of input directions*.

When the number of inputs (variational parameters, nuclear
coordinates) far exceeds the number of outputs (a single energy or
a single $\log\Psi$), AAD is preferred: it evaluates *all* gradients
in time comparable to a single function evaluation. AAD was first
introduced in the QMC context for atomic-force calculations
[Sorella and Capriotti, J. Chem. Phys. 133, 234111 (2010)](https://doi.org/10.1063/1.3516208) and is
the reference technique used by jQMC, leveraging the
`jax.grad` / `jax.value_and_grad` primitives of JAX
[Bradbury et al., JAX (2018)](https://github.com/jax-ml/jax).


## Derivatives with respect to variational parameters (MCMC)

The derivative of the total energy with respect to a real
variational parameter $\alpha_k$ is conventionally written as a
generalized *force*

$$
f_k = -\frac{\partial E(\boldsymbol{\alpha})}{\partial \alpha_k}
= -\frac{\partial}{\partial \alpha_k}
\frac{\langle\Psi_{\boldsymbol{\alpha}}|\hat{\mathcal{H}}|\Psi_{\boldsymbol{\alpha}}\rangle}
     {\langle\Psi_{\boldsymbol{\alpha}}|\Psi_{\boldsymbol{\alpha}}\rangle}.
$$ (general_force)

In variational Monte Carlo, $f_k$ is estimated from $M$ sampled
electronic configurations $\{\mathbf{x}_i\}_{i=1}^{M}$
[Becca and Sorella, *Quantum Monte Carlo Approaches for Correlated
Systems*, Cambridge University Press (2017)](https://doi.org/10.1017/9781316417041);
see [Sorella, Casula, and Rocca, *J. Chem. Phys.* **127**, 014105 (2007)](https://doi.org/10.1063/1.2746035)
for the complex generalization. The derivation makes the origin of the
final $2\,\mathrm{Re}$ factor explicit. Starting from
$E(\boldsymbol{\alpha}) = \langle\Psi_{\boldsymbol{\alpha}}|\hat{\mathcal{H}}|\Psi_{\boldsymbol{\alpha}}\rangle / \langle\Psi_{\boldsymbol{\alpha}}|\Psi_{\boldsymbol{\alpha}}\rangle$
and differentiating with respect to the **real** variational
parameter $\alpha_k$, two terms appear -- one from the **bra**
derivative $\partial\langle\Psi|/\partial\alpha_k = \langle\partial_{\alpha_k}\Psi|$,
and one from the **ket** derivative
$|\partial_{\alpha_k}\Psi\rangle$:

$$
\frac{\partial E}{\partial \alpha_k}
= \frac{\langle\partial_{\alpha_k}\Psi|\hat{\mathcal{H}}|\Psi\rangle
       + \langle\Psi|\hat{\mathcal{H}}|\partial_{\alpha_k}\Psi\rangle}
       {\langle\Psi|\Psi\rangle}
- E\,
\frac{\langle\partial_{\alpha_k}\Psi|\Psi\rangle
     + \langle\Psi|\partial_{\alpha_k}\Psi\rangle}
     {\langle\Psi|\Psi\rangle}.
$$

For a real parameter $\alpha_k$ and Hermitian $\hat{\mathcal{H}}$
the ket-side terms are the **complex conjugates** of the
corresponding bra-side terms,
$\langle\Psi|\hat{\mathcal{H}}|\partial_{\alpha_k}\Psi\rangle
= \overline{\langle\partial_{\alpha_k}\Psi|\hat{\mathcal{H}}|\Psi\rangle}$
and similarly for the overlap term. Writing the ket-side terms
explicitly as "$+\,\mathrm{c.c.}$" of the bra-side and recognizing
$z + \mathrm{c.c.} = 2\,\mathrm{Re}\,z$, the derivative collapses
to a single $\mathrm{Re}$-projected bra-side expression. The
bra-side ratios evaluate in Monte Carlo to

$$
\frac{\langle\partial_{\alpha_k}\Psi|\hat{\mathcal{H}}|\Psi\rangle}
{\langle\Psi|\Psi\rangle}
= \bigl\langle O_k^{*}(\mathbf{x})\,e_L(\mathbf{x})\bigr\rangle,
\qquad
\frac{\langle\partial_{\alpha_k}\Psi|\Psi\rangle}{\langle\Psi|\Psi\rangle}
= \bigl\langle O_k^{*}(\mathbf{x})\bigr\rangle,
$$

with $O_k(\mathbf{x})$ defined below in eq. {eq}`O_k_def` and
$\langle\cdot\rangle$ the $|\Psi|^{2}$-weighted average. The
conjugate on $O_k$ comes from
$(\partial_{\alpha_k}\Psi)^{*} = (\partial_{\alpha_k}\ln\Psi)^{*}\,\Psi^{*}$.
Substituting and using $E = \langle e_L\rangle$ gives

$$
\begin{aligned}
f_k
&= -\frac{\partial E}{\partial \alpha_k}
= -2\,\mathrm{Re}\,\Bigl[
\bigl\langle e_L(\mathbf{x})\,O_k^{*}(\mathbf{x})\bigr\rangle
- \bigl\langle e_L\bigr\rangle\,\bigl\langle O_k^{*}\bigr\rangle
\Bigr] \\
&= -2\,\mathrm{Re}\,\frac{\sum_{\mathbf{x}}
e_L(\mathbf{x})
\bigl(O_k(\mathbf{x}) - \bar O_k\bigr)^{*}
|\Psi_{\boldsymbol{\alpha}}(\mathbf{x})|^{2}}
{\sum_{\mathbf{x}} |\Psi_{\boldsymbol{\alpha}}(\mathbf{x})|^{2}} \\
&\;\simeq\; -2\,\mathrm{Re}\,\frac{\sum_{i=1}^{M} w_i\,
e_L(\mathbf{x}_i)\bigl(O_k(\mathbf{x}_i) - \bar O_k\bigr)^{*}}
{\sum_{i=1}^{M} w_i},
\end{aligned}
$$ (generalized_force_mc)

where $e_L(\mathbf{x})$ is the local energy and

$$
O_k(\mathbf{x}) =
\frac{\partial \ln \Psi_{\boldsymbol{\alpha}}(\mathbf{x})}
     {\partial \alpha_k}
$$ (O_k_def)

is the *logarithmic derivative* of the trial wave function. The
sample average is

$$
\bar O_k =
\frac{\sum_{i=1}^{M} w_i\,O_k(\mathbf{x}_i)}{\sum_{i=1}^{M} w_i}.
$$ (O_k_bar)

Equation {eq}`generalized_force_mc` is the force for a **real**
parameter $\alpha_k$.  It remains valid when the wave function and
the sample-level local energy are complex: the conjugate on $O_k$
comes from the bra, and the final $\operatorname{Re}$ restricts the
variation to the real parameter axis.

For an independent complex parameter $c_k$, define
$O_k=\partial\ln\Psi/\partial c_k$.  The packed force carrying both
real degrees of freedom is instead

$$
\begin{aligned}
f_k^{\mathbb C}
&\equiv -2\frac{\partial E}{\partial\bar c_k} \\
&= -2\,\Bigl\langle
  \bigl(O_k-\bar O_k\bigr)^{*}
  \bigl(e_L-\bar e_L\bigr)
\Bigr\rangle_w .
\end{aligned}
$$ (generalized_force_complex_mc)

There is no outer $\operatorname{Re}$ in
eq. {eq}`generalized_force_complex_mc`: its real part moves
$\operatorname{Re}c_k$ and its imaginary part moves
$\operatorname{Im}c_k$.  Applying a global real projection would
freeze half of every complex parameter.

The JAX and storage conventions are separate operations.  For a
holomorphic $\Psi(c)$,

$$
\operatorname{jax.grad}\bigl(\operatorname{Re}\ln\Psi, c\bigr)
=\frac{\partial\ln\Psi}{\partial c}=O_k,
$$

so `O_matrix` stores the **un-conjugated** logarithmic derivative.
The estimator constructs $O_k^{*}$ explicitly at its consumer.  See
{ref}`sec:complex_gradients` for the JAX convention and
{ref}`subsec:sr_log_derivative` for the optimizer's `X` convention.

The AD reference evaluates $O_k(\mathbf{x})$ for *all* $k$ in a
reverse-mode pass through the trial-wave-function code via `jax.grad`.
Production evaluates the same logarithmic derivatives with the analytic
kernels in `jqmc.derivatives`; the AD formulation specifies the derivative
and conjugation convention, not the production execution path.

Although the Hamiltonian does not depend on $\boldsymbol{\alpha}$,
the raw parameter derivative of $E$ **does** contain a
local-energy-derivative term,
$\langle \partial_{\alpha_k} e_L \rangle$, because
$e_L = \hat{\mathcal H}\Psi/\Psi$ depends on
$\boldsymbol{\alpha}$ through $\Psi$.  That term is removed from
eq. {eq}`generalized_force_mc` by the Hermiticity identity
$\langle \partial_{\alpha_k} e_L \rangle = 0$ (real parameters /
real $\Psi$), which holds **only as an expectation over the exact
$|\Psi|^{2}$ distribution**.  On a finite sample the identity holds
only up to statistical error; how the two optimizers of jQMC treat this
term is discussed in {ref}`subsec:force_local_energy_derivative`.  This is
in contrast to the atomic-force estimator below, where the Hamiltonian
itself depends on the parameter being differentiated and the
$\langle d e_L/d\mathbf{R}\rangle$ term is always kept explicitly.

On the complex path, $\bar O_k$, the sample-level $O_k$, and the
unprojected covariance may all be complex.  Whether a component is
restricted to a real axis or spans a complex plane is a property of
the parameter manifold, not of the sample vector.  That restriction
must therefore be applied by the optimizer in parameter coordinates;
globally replacing the local-energy residual or force covariance by
its real part is invalid for a mixed real/complex parameter set.  See
{ref}`subsec:sr_parameter_manifold`.

(subsec:force_local_energy_derivative)=
### The local-energy-derivative term on a finite sample

The derivation above invoked Hermiticity once -- the ket-side terms
were folded into "$+\,\mathrm{c.c.}$" of the bra side.  Doing so is
exact for the *integrals*, i.e. for expectations over the exact
$|\Psi|^{2}$ distribution.  A Monte Carlo estimate replaces those
integrals by averages over the sampled ensemble, and the two sides of
the Hermiticity identity then acquire **different** statistical errors.

The derivative that does not use Hermiticity is obtained by
differentiating the sampled functional directly.  For a holomorphic
$\Psi(c_k)$ and the measured energy $E=\langle \operatorname{Re}
e_L\rangle_w$, the Wirtinger derivative of the weighted average gives,
exactly,

$$
\frac{\partial E}{\partial c_k}
= \Bigl\langle
    \delta\!\bigl(\operatorname{Re} e_L\bigr)\,\delta O_k
  \Bigr\rangle_w
+ \tfrac12\bigl\langle \partial_{c_k} e_L \bigr\rangle_w ,
$$ (generalized_force_identity_free)

where $\delta A = A - \bar A$ and
$\partial_{c_k} e_L$ is the per-sample (holomorphic) local-energy
parameter derivative.  The two factors follow from the Wirtinger rules for
a holomorphic $\Psi$: $\partial_{c_k}\ln w = \partial_{c_k}\ln(\Psi\bar\Psi)
= O_k$ (the antiholomorphic $\bar\Psi$ does not contribute) and
$\partial_{c_k}\operatorname{Re} e_L = \tfrac12\,\partial_{c_k} e_L$.
Under exact sampling the Hermiticity identity

$$
\bigl\langle \partial_{c_k} e_L \bigr\rangle
= -\,2i\,\bigl\langle
    \delta\!\bigl(\operatorname{Im} e_L\bigr)\,\delta O_k
  \Bigr\rangle
\;\xrightarrow[\ \Psi\ \text{real}\ ]{}\; 0
$$ (hermiticity_delc_identity)

turns eq. {eq}`generalized_force_identity_free` back into the
conjugated covariance of
eq. {eq}`generalized_force_complex_mc`: substituting
{eq}`hermiticity_delc_identity` into
{eq}`generalized_force_identity_free` gives
$\partial E/\partial c_k = \langle \delta O_k\,\delta\bar e_L\rangle_w$,
and because $E$ is real,
$-2\,\partial E/\partial \bar c_k = -2\,\overline{\partial E/\partial c_k}$
reproduces $-2\langle \delta O_k^{*}\,\delta e_L\rangle_w$ with the
**complex** $e_L$.  The residual of eq. {eq}`hermiticity_delc_identity` on a
given sample is therefore a per-parameter diagnostic of sampling
quality: it vanishes for a perfectly equilibrated $|\Psi|^2$ ensemble.

For real independent optimization coordinates $q_a$, the same result can be
stated without Wirtinger notation. Set $O_a=\partial_{q_a}\ln\Psi$ and
$D_a=\partial_{q_a}e_L$. For a parameter-independent Hermitian Hamiltonian,
with derivatives remaining in its boundary-condition domain, the exact
normalized expectation obeys

$$
\begin{aligned}
\partial_{q_a}E
&=\langle\operatorname{Re}D_a\rangle
 +2\operatorname{Cov}(\operatorname{Re}e_L,\operatorname{Re}O_a)\\
&=2\operatorname{Re}\langle O_a^*(e_L-E)\rangle,\\
\langle\operatorname{Re}D_a\rangle
&=2\langle\operatorname{Im}O_a\,\operatorname{Im}e_L\rangle.
\end{aligned}
$$ (real_coordinate_energy_gradient_forms)

The last identity uses $\langle\operatorname{Im}e_L\rangle=0$ for the
Hermitian expectation. Thus the explicit derivative contribution is
incorporated into the **full complex covariance**. It is not generally zero,
and adding it again to that covariance would double count it. Replacing
$e_L$ by its real part before the complex covariance would lose it.
The zero-mean simplification holds for a real state with real local energy.

Analytical checks of the exact-expectation identities and the worked
periodic phase example are work in progress and are not reproduced here.

On a finite sample, the two expectation identities have statistical
residuals. The covariance-only SR estimator and the explicit-derivative
estimator can consequently differ, while converging to the same gradient
under the stated assumptions. The non-symmetric LM retains its own sampled
$D_a$ and bra/ket estimators rather than imposing their exact-expectation
relation on every finite ensemble.

The two optimizers of jQMC treat this term differently:

1. **The stochastic-reconfiguration force is the covariance-only
   estimator** of eq. {eq}`generalized_force_complex_mc`
   [Sorella, Phys. Rev. B 71, 241103 (2005)](https://doi.org/10.1103/PhysRevB.71.241103). It has the zero-variance
   property described below. It uses the Hermiticity identity with the full
   complex $e_L$, rather than assuming
   $\langle\partial_{c_k}e_L\rangle=0$. Only for a real state is that
   derivative mean a zero-expectation fluctuation; for a complex state its
   nonzero expectation is encoded by the imaginary-energy covariance in
   eq. {eq}`hermiticity_delc_identity`.
2. **The linear method keeps $\langle\partial_{c_k} e_L\rangle$ through
   its non-symmetric matrix pencil** ({ref}`sec:opt_wf`): the bra element
   $H_{0k}$ and the ket element $H_{k0}$ of the Hamiltonian matrix in the
   tangent space are evaluated by their own estimators and are not
   Hermitian-averaged, so the linear-method eigenvector is consistent with
   the sampled energy surface even when
   $D_k=\langle\partial_{c_k} e_L\rangle$ is finite.  The per-sample
   $\partial_{c_k} e_L$ is already required for the linear-method matrices,
   so this costs no extra sampling; plain-SR runs are unaffected.

### Zero-variance property

If the trial wave function is an exact eigenstate of
$\hat{\mathcal{H}}$, then $e_L(\mathbf{x})$ is independent of
$\mathbf{x}$ and the integrand of eq. {eq}`generalized_force_mc`
becomes proportional to
$\bigl(O_k(\mathbf{x}) - \bar O_k\bigr)$, whose Monte Carlo
average vanishes exactly. The parameter forces $f_k$ therefore have
the *zero-variance* property at the exact wave function: their
statistical error shrinks faster than $1/\sqrt{M}$ as
$\Psi \to \Psi_{\rm exact}$. This is what makes
eq. {eq}`generalized_force_mc` a good descent direction for an
optimizer that consumes $\{f_k\}_k$. jQMC uses it in the SR/LM hierarchy
(sec. {ref}`sec:opt_wf`).

### Limitations of variational-parameter forces

jQMC implements variational-parameter forces for MCMC only. LRDMC and GFMC
do not provide the corresponding estimator.

(subsec:forces_mcmc)=

## Atomic forces in MCMC: SWCT estimator

The atomic force on nucleus $I$,
$\mathbf{F}_{I} = -dE/d\mathbf{R}_{I}$, picks up two contributions
that the variational-parameter case did not have: (a) the
Hamiltonian itself depends on $\mathbf{R}_{I}$ via the
electron-nucleus Coulomb term and the ECP, so the local-energy
derivative $\partial e_L / \partial \mathbf{R}_{I}$ no longer
vanishes, and (b) the trial wave function depends on
$\mathbf{R}_{I}$ through the basis-set centers, contributing a
Pulay term. jQMC uses the *differential* form of the force estimator
combined with the *space-warp coordinate transformation* (SWCT) of
Umrigar [Umrigar, Int. J. Quantum Chem. Symp. 23, 217 (1989)](https://doi.org/10.1002/qua.560360826) to
reduce its statistical variance.

Starting from $E = \langle\Psi|\hat{\mathcal{H}}|\Psi\rangle / \langle\Psi|\Psi\rangle$
and differentiating w.r.t. $\mathbf{R}_{I}$ -- with the real
parameter $\mathbf{R}_{I}$ entering both the Hamiltonian and the
basis-set centers -- the bra-derivative side and the ket-derivative
side again contribute as complex-conjugate partners. Writing the
ket-side as "$+\,\mathrm{c.c.}$" of the bra-side:

$$
\mathbf{F}_{I} =
-\Bigl\langle \frac{d e_L}{d\mathbf{R}_{I}}\Bigr\rangle
+ \Bigl\{\,
\bigl\langle (e_L - \langle e_L\rangle)\,
\Bigl(\frac{d\,\ln(\mathcal{J}^{1/2}\Psi)}{d\mathbf{R}_{I}}\Bigr)^{*}
\bigr\rangle
+ \mathrm{c.c.}\,\Bigr\}\cdot(-1),
$$

where the first term is the Hellmann--Feynman piece (already real
in expectation for Hermitian $\hat{\mathcal{H}}$), the Pulay
bracket carries the c.c. inherited from
$\langle\Psi|\hat{\mathcal{H}}|\partial_{R_I}\Psi\rangle = \overline{\langle\partial_{R_I}\Psi|\hat{\mathcal{H}}|\Psi\rangle}$,
and the $\mathcal{J}^{1/2}$ factor in the log-derivative carries
the SWCT Jacobian rescaling (algebraically a real positive scalar,
so commutes with both $\mathrm{Re}$ and complex conjugation).
Using $z + \mathrm{c.c.} = 2\,\mathrm{Re}\,z$ collapses the
braces into a single $\mathrm{Re}$-projected expression:

$$
\mathbf{F}_{I} =
-\,\mathrm{Re}\,\Bigl\langle \frac{d}{d\mathbf{R}_{I}}\,e_L \Bigr\rangle
+ 2\,\mathrm{Re}\,\Bigl[
\langle e_L\rangle
\Bigl\langle \Bigl(\frac{d}{d\mathbf{R}_{I}}
\ln\!\bigl(\mathcal{J}^{1/2}\Psi\bigr)\Bigr)^{*}\Bigr\rangle
- \Bigl\langle e_L\,\Bigl(\frac{d}{d\mathbf{R}_{I}}
\ln\!\bigl(\mathcal{J}^{1/2}\Psi\bigr)\Bigr)^{*}\Bigr\rangle
\Bigr].
$$ (forces_aad)

The factor $+2$ multiplying the Pulay bracket therefore traces
back directly to the $z + \mathrm{c.c.}$ identity. The
$\mathrm{Re}(\cdot)$ wrapper on the Hellmann--Feynman term is the
identity for Hermitian $\hat{\mathcal{H}}$ (the imaginary part
averages to zero in expectation), and is written here for
notational consistency with the rest of the force estimator for
complex wave functions. $\mathcal{J}$ is the Jacobian of the
SWCT, and $\langle\cdot\rangle$ denotes the Monte Carlo average
over $|\Psi|^{2}$.

For real $\Psi$ the complex conjugate is the identity
and the two terms of the c.c. pair coincide; the factor 2 then
appears as "two equal contributions". For complex $\Psi$
(see {doc}`complex_gradients`) the two contributions are genuinely conjugate and the
factor $2\,\mathrm{Re}$ is the correct algebraic generalization
of the real factor of 2.

### Space-warp coordinate transformation

The SWCT mimics a rigid drag of the electronic charge around the
moving nucleus, so that the variance introduced by the Hamiltonian
$\mathbf{R}_{I}$-dependence is partially cancelled by an analogous
$\mathbf{R}_{I}$-dependence built into the sampling coordinates.
Concretely, every electron coordinate $\mathbf{r}_{i}$ is shifted
by

$$
\bar{\mathbf{r}}_{i} =
\mathbf{r}_{i} + \Delta \mathbf{R}_{I}\,\omega_{I}(\mathbf{r}_{i}),
$$ (swct_displacement)

$$
\omega_{I}(\mathbf{r}) =
\frac{\kappa(|\mathbf{r} - \mathbf{R}_{I}|)}
     {\sum_{J=1}^{N_a} \kappa(|\mathbf{r} - \mathbf{R}_{J}|)},
$$ (swct_omega)

where $\kappa(r)$ is a localizing function that decays fast enough
for large $r$ so that $\omega_{I} \to 0$ outside the immediate
neighborhood of nucleus $I$ -- charges far from every nucleus are
unaffected. jQMC follows Umrigar's original prescription and
adopts $\kappa(r) = 1/r^{4}$, which has been verified to behave
well across molecular and PBC systems.

### Differential form: AAD-friendly expansion

Expanding eq. {eq}`forces_aad` and applying the chain rule,
the VMC force can be written entirely in terms of partial
derivatives of the local energy and of $\ln\Psi$
[Sorella and Capriotti (2010)](https://doi.org/10.1063/1.3516208). The expansion produces, for each
electron $i$, a contribution

$$
\omega(\mathbf{R}_{I},\mathbf{r}_{i})\,
\frac{\partial\ln\Psi_T}{\partial\mathbf{r}_{i}}
+ \frac{1}{2}\,
\frac{\partial\omega(\mathbf{R}_{I},\mathbf{r}_{i})}
     {\partial\mathbf{r}_{i}}
$$

from the SWCT change of variables, alongside the direct
$\partial\ln\Psi_T/\partial\mathbf{R}_{I}$ term from the
basis-set centers. The same bra-side / ket-side pairing as in
eq. {eq}`forces_aad` applies to **every** chain-ruled
log-derivative below, so the Pulay-side bracket has the structure

$$
\Bigl\langle (e_L - E)\,\bigl(\text{sum of }\partial\ln\Psi_T\text{ pieces}\bigr)^{*}\Bigr\rangle + \mathrm{c.c.}
\;=\;
2\,\mathrm{Re}\,\Bigl\langle (e_L - E)\,\bigl(\text{sum of }\partial\ln\Psi_T\text{ pieces}\bigr)^{*}\Bigr\rangle,
$$

with the conjugate $(\cdot)^{*}$ inherited from $\langle\partial\Psi|$
(it lives on the log-derivative factor; the $\omega$-matrix and
$e_L - E$ stay un-conjugated). After this c.c. fold, the
expanded VMC force reads

$$
\begin{aligned}
\mathbf{F}_{I}^{\,\mathrm{VMC}}
=\;
&-\,\mathrm{Re}\,\Bigl\langle
\frac{\partial e_L}{\partial \mathbf{R}_{I}}
+ \sum_{i} \omega(\mathbf{R}_{I},\mathbf{r}_{i})\,
\frac{\partial e_L}{\partial \mathbf{r}_{i}}
\Bigr\rangle \\
&-\,2\,\mathrm{Re}\,\Bigl\langle
\bigl(e_L - E\bigr)
\Bigl(
\frac{\partial \ln \Psi_T}{\partial \mathbf{R}_{I}}
+ \sum_{i}
\Bigl[
\omega(\mathbf{R}_{I},\mathbf{r}_{i})\,
\frac{\partial \ln \Psi_T}{\partial \mathbf{r}_{i}}
+ \tfrac{1}{2}\,
\frac{\partial \omega(\mathbf{R}_{I},\mathbf{r}_{i})}
     {\partial \mathbf{r}_{i}}
\Bigr]
\Bigr)^{*}
\Bigr\rangle.
\end{aligned}
$$ (force_vmc_matrix)

Computing eq. {eq}`force_vmc_matrix` requires four derivative
fields per sample,

$$
\frac{\partial e_L}{\partial \mathbf{r}_{i}},\quad
\frac{\partial \ln \Psi}{\partial \mathbf{r}_{i}},\quad
\frac{\partial e_L}{\partial \mathbf{R}_{I}},\quad
\frac{\partial \ln \Psi}{\partial \mathbf{R}_{I}},
$$ (force_vmc_components)

totalling $6 N_e + 6 N_a$ components per configuration (the
angle-bracket averages are taken only afterwards, in
eq. {eq}`force_vmc_matrix`). Production
computes these fields with analytic local-energy and $\log\Psi$
derivatives, for both all-electron and pseudopotential calculations.
The independent AD reference differentiates the corresponding value
functions; it implements the same chain-rule expansion.

### Practical matrix layout

For an electronic configuration $\mathbf{x}$ split into up and down
electrons, jQMC stores the per-sample force ingredients as the
following arrays:

$$
\begin{aligned}
\partial E^{R} &\equiv \frac{\partial e_L}{\partial \mathbf{R}_{I}}
& &\text{dim. } (N_a, 3), \\
\partial E^{r,\uparrow} &\equiv \frac{\partial e_L}{\partial \mathbf{r}_{i}^{\uparrow}}
& &\text{dim. } (N_e^{\uparrow}, 3), \\
\partial E^{r,\downarrow} &\equiv \frac{\partial e_L}{\partial \mathbf{r}_{i}^{\downarrow}}
& &\text{dim. } (N_e^{\downarrow}, 3), \\
\partial \Psi^{R} &\equiv \frac{\partial \ln \Psi_T}{\partial \mathbf{R}_{I}}
& &\text{dim. } (N_a, 3), \\
\partial \Psi^{r,\uparrow} &\equiv \frac{\partial \ln \Psi_T}{\partial \mathbf{r}_{i}^{\uparrow}}
& &\text{dim. } (N_e^{\uparrow}, 3), \\
\partial \Psi^{r,\downarrow} &\equiv \frac{\partial \ln \Psi_T}{\partial \mathbf{r}_{i}^{\downarrow}}
& &\text{dim. } (N_e^{\downarrow}, 3), \\
\partial \Omega^{r,\uparrow} &\equiv
\sum_{i} \frac{\partial \omega(\mathbf{R}_{I},\mathbf{r}_{i}^{\uparrow})}
                {\partial \mathbf{r}_{i}^{\uparrow}}
& &\text{dim. } (N_a, 3), \\
\partial \Omega^{r,\downarrow} &\equiv
\sum_{i} \frac{\partial \omega(\mathbf{R}_{I},\mathbf{r}_{i}^{\downarrow})}
                {\partial \mathbf{r}_{i}^{\downarrow}}
& &\text{dim. } (N_a, 3), \\
\Omega^{\uparrow} &\equiv \omega(\mathbf{R}_{I},\mathbf{r}_{i}^{\uparrow})
& &\text{dim. } (N_a, N_e^{\uparrow}), \\
\Omega^{\downarrow} &\equiv \omega(\mathbf{R}_{I},\mathbf{r}_{i}^{\downarrow})
& &\text{dim. } (N_a, N_e^{\downarrow}).
\end{aligned}
$$ (force_components)

The $(N_a, 3)$ VMC force matrix is then obtained by the
matrix-matrix product

$$
\begin{aligned}
\mathbf{F}^{\mathrm{VMC}}
&=
-\,\mathrm{Re}\,\Bigl\langle
\partial E^{R}
+ \Omega^{\uparrow}\,\partial E^{r,\uparrow}
+ \Omega^{\downarrow}\,\partial E^{r,\downarrow}
\Bigr\rangle \\
&\quad
-\,2\,\mathrm{Re}\,\Bigl\langle
\bigl(e_L - E\bigr)
\Bigl(
\partial \Psi^{R}
+ \Omega^{\uparrow}\,\partial \Psi^{r,\uparrow}
+ \Omega^{\downarrow}\,\partial \Psi^{r,\downarrow}
+ \tfrac{1}{2}\,
\bigl(\partial \Omega^{r,\uparrow} + \partial \Omega^{r,\downarrow}\bigr)
\Bigr)^{*}
\Bigr\rangle \\
&\equiv\;
\mathbf{F}^{\mathrm{HF}} + \mathbf{F}^{\mathrm{Pulay}},
\end{aligned}
$$ (force_vmc_full)

splitting naturally into the Hellmann-Feynman part

$$
\mathbf{F}^{\mathrm{HF}} =
-\,\mathrm{Re}\,\Bigl\langle
\partial E^{R}
+ \Omega^{\uparrow}\,\partial E^{r,\uparrow}
+ \Omega^{\downarrow}\,\partial E^{r,\downarrow}
\Bigr\rangle
$$ (force_hf)

and the Pulay part

$$
\mathbf{F}^{\mathrm{Pulay}}
= -\,2\,\mathrm{Re}\,\Bigl\langle
\bigl(e_L - E\bigr)
\Bigl(
\partial \Psi^{R}
+ \Omega^{\uparrow}\,\partial \Psi^{r,\uparrow}
+ \Omega^{\downarrow}\,\partial \Psi^{r,\downarrow}
+ \tfrac{1}{2}\,
\bigl(\partial \Omega^{r,\uparrow} + \partial \Omega^{r,\downarrow}\bigr)
\Bigr)^{*}
\Bigr\rangle.
$$ (force_pulay)

For a real wave function every $\mathrm{Re}(\cdot)$ collapses to
the identity, every $(\cdot)^{*}$ to the identity, and the
"$+\,\mathrm{c.c.}$" partners coincide so the factor $2$ in front
of the Pulay bracket counts "two identical contributions". The
two forms above then reduce to the familiar real expressions of
eq. {eq}`force_vmc_full`. For a complex wave function the
$\omega$ matrix and $\partial\Omega$ stay **real** (structural /
geometric quantities, not $\Psi$-derived), but each product with
a complex $\partial \Psi$ or complex $\partial e_L$ becomes
complex; the bra-side and ket-side contributions are genuine
complex conjugates of each other, and the
$z + \mathrm{c.c.} = 2\,\mathrm{Re}\,z$ collapse is the algebraic
origin of the $2\,\mathrm{Re}$ factor. The outer
$\mathrm{Re}(\cdot)$ projects to the physical real force.

In production, the ingredients in eq. {eq}`force_components` are
provided by the analytic wave-function and Hamiltonian derivative kernels.
The assembly of eq. {eq}`force_vmc_full` consists of contractions and
reductions of these fields. The AD reference supplies an independent
evaluation of the derivatives for comparison.


### Complex wave functions: real projection of the force

For a complex (twisted Bloch) wave function the local energy is complex
sample by sample: $\mathrm{Im}\,e_L \neq 0$ on each configuration, while
$\langle \mathrm{Im}\,e_L\rangle = 0$ in expectation for a Hermitian
$\hat{\mathcal{H}}$. The atomic force is a real observable, so jQMC keeps
$e_L$ and its derivatives complex per sample and applies the outer
$\mathrm{Re}(\cdot)$ of eqs. {eq}`force_hf` and {eq}`force_pulay` to the
assembled estimator. Because $\mathbf{R}_I$ is real, the derivative and the
real projection commute,

$$
\frac{\partial\,\mathrm{Re}\,e_L}{\partial \mathbf{R}_{I}}
=
\mathrm{Re}\,\frac{\partial e_L}{\partial \mathbf{R}_{I}}
\qquad (\text{real }\mathbf{R}_{I}),
$$ (re_grad_commute)

which is also how the AD reference obtains a real scalar to differentiate
(see {doc}`complex_gradients` for the JAX convention). For a real wave
function the projection is the identity.

The $z + \mathrm{c.c.} = 2\,\mathrm{Re}\,z$ fold behind the
$2\,\mathrm{Re}$ factor of eqs. {eq}`forces_aad`-{eq}`force_vmc_full`
assumes a Hermitian $\hat{\mathcal{H}}$. jQMC implements only the
non-relativistic electronic Hamiltonian; spin-orbit coupling and other
relativistic Hamiltonians are not supported.

### Periodic bases, twists, and tiled orbitals

The Jastrow is real and positive at every twist; only the determinant carries the Bloch phase. Use one phase-free reference shared across twists, taken from a real Gamma basis or from the real cell-periodic functions of a complex MO, plane-wave, or B-spline reference. When Jastrow parameters are shared, the same reference must be shared; setup validates this. A retained nonzero twist or nonunit image phase is rejected. Detailed formulation of periodic twist/reference handling is work in progress.

Only atom-centered functions contribute a Pulay force. Plane-wave and B-spline blocks have no nuclear-center derivative term, so a determinant built only from them gives the Hellmann--Feynman part (including SWCT terms) plus the Pulay part of the atom-centered Jastrow. In a composite basis only periodic GTO children contribute. Basis data fixed at one geometry do not carry orbital response to a later displacement: regenerate plane-wave coefficients or B-spline tables for each geometry used in forces, optimization, or dynamics.

Orbital tiling changes storage only. Each supercell atom keeps an independent position derivative; primitive image contributions are assigned to their physical supercell atom before contraction, and dividing a primitive force by the number of replicas is not a derivative. Fixed plane-wave and B-spline data have no nuclear-center response. The native tiled derivative is defined at the reference geometry; continuation under independent atom displacements is outside its current scope. Detailed formulation of tiled derivatives is work in progress. See [orbital tiling](orbital_tiling.md).


### Translational sum rule

For a periodic system with more than two atoms, a useful sanity check of
the force output is the translational sum rule

$$
\sum_{I} \mathbf{F}_{I} = 0 .
$$ (force_sum_rule)

It follows from the invariance of $e_L$ and $\ln\Psi$ under a rigid
translation of all nuclei and electrons together. When SWCT is on, the
weights of eq. {eq}`swct_omega` sum to one, $\sum_I\omega_I(\mathbf{r})=1$,
so the sum rule holds **sample by sample**, to round-off, independently of
the statistical error. With SWCT off it holds only on average, within the
statistical error. It is not expected to hold when $\ln\Psi$ is not
translation invariant, which is the case for a plane-wave or B-spline
block (fixed in space while the nuclei move, see above); the sum then
measures the missing basis response. For a two-atom system the sum rule
reads $\mathbf{F}_1 = -\mathbf{F}_2$.

### Validation

The MCMC force implementation has been validated by reproducing the
potential-energy surface (PES) of the H$_2$ and N$_2$ dimers and
checking two diagnostics that are sensitive to bugs in either the
estimator or the SWCT algebra:

1. The numerical derivative of the variationally-converged PES
   with respect to bond distance agrees with the directly computed
   atomic forces (Hellmann-Feynman consistency).
2. With SWCT switched on, the forces on the two atoms of a
   symmetric dimer have *exactly* equal magnitudes and opposite
   directions, with no residual statistical bias
   [Nakano et al. (2022)](https://doi.org/10.1063/5.0076302).

For periodic systems, including a complex (non-$\Gamma$) twist, the
validation is limited to internal consistency checks in the test suite:
agreement of the analytic kernels with the AD reference and the
translational sum rule of eq. {eq}`force_sum_rule` with SWCT, in MCMC and
LRDMC. No comparison of periodic or complex-twist forces against an
independent code is reported here.

(subsec:forces_lrdmc)=

## Atomic forces in LRDMC

```{note}
**Complex wave functions in LRDMC.** The formulae below are written in real
form for clarity. Forces are also available for complex Bloch wave functions,
with the fixed-phase approximation of {ref}`sec:lrdmc_complex_phase`; the
detailed formulation is work in progress.
```

The LRDMC force estimator differs from the MCMC one
(sec. {ref}`subsec:forces_mcmc`) in three respects:

1. **SWCT is off by default** (`use_swct = false` in `[lrdmc-bra]` and
   `[lrdmc-tau]`, whereas `[mcmc]` defaults to `true`). The walkers sample
   the mixed distribution $\Psi_T\Psi_0$ rather than $|\Psi_T|^{2}$. In
   VMC the $\omega$ terms of eq. {eq}`force_vmc_full` average to zero
   exactly and only reduce the variance; on the mixed distribution they
   cancel only to the accuracy of the Reynolds approximation (item 2), so
   switching SWCT on can reduce the statistical error of the LRDMC force
   but may change its value by an amount of the order of the mixed-estimator
   bias. With `use_swct = true` the $\omega$ terms of
   eq. {eq}`force_vmc_full` are added to the estimators below exactly as
   in MCMC.
2. The Reynolds approximation
   [Reynolds et al., Int. J. Quantum Chem. 29, 589 (1986)](https://doi.org/10.1002/qua.560290403) is used to
   replace the inaccessible $\partial \Psi_0/\partial \mathbf{R}_{I}$
   by the available $\partial \Psi_T/\partial \mathbf{R}_{I}$.
3. The Pathak-Wagner regularization
   [Pathak and Wagner, AIP Advances 10, 085213 (2020)](https://doi.org/10.1063/5.0004008)
   can be switched on with `epsilon_PW > 0` to suppress divergent
   contributions to the local-energy derivative near the nodal surface
   of $\Psi_T$. It is off by default (`epsilon_PW = 0.0`).

### Mixed-distribution estimator (without regularization)

With SWCT off, all $\omega$-dependent terms in
eq. {eq}`force_vmc_full` vanish ($\omega_I = 0$). By the same bra/ket
pairing argument as the VMC case (eq. {eq}`forces_aad`), the
Pulay-bracket contribution carries a "$+\,\mathrm{c.c.}$" partner from
the ket-derivative side; using $z + \mathrm{c.c.} = 2\,\mathrm{Re}\,z$
collapses it to a single $\mathrm{Re}$-projected expression with the
conjugate $(\cdot)^{*}$ on the log-derivative. Re-using the shorthand of
eq. {eq}`force_components`, the force on atom $I$ sampled from the mixed
distribution $\Psi_T\Psi_0$ reads

$$
\begin{aligned}
\mathbf{F}_{I}^{\mathrm{mixed}}
&= -\,\mathrm{Re}\,\bigl\langle \partial E^{R} \bigr\rangle_{\Psi_T \Psi_0}
\;-\; 2\,\mathrm{Re}\,\bigl\langle
\bigl(e_L - E\bigr)\,\bigl(\partial \Psi^{R}\bigr)^{*}
\bigr\rangle_{\Psi_T \Psi_0} \\
&\equiv\;
\mathbf{F}^{\mathrm{HF}}_{\mathrm{mixed}} + \mathbf{F}^{\mathrm{Pulay}}_{\mathrm{mixed}},
\end{aligned}
$$ (force_lrdmc_mixed)

with

$$
\mathbf{F}^{\mathrm{HF}}_{\mathrm{mixed}}
= -\,\mathrm{Re}\,\bigl\langle \partial E^{R} \bigr\rangle_{\Psi_T \Psi_0},
$$ (force_lrdmc_hf)

$$
\mathbf{F}^{\mathrm{Pulay}}_{\mathrm{mixed}}
= -\,2\,\mathrm{Re}\,\bigl\langle
\bigl(e_L - E\bigr)\,\bigl(\partial \Psi^{R}\bigr)^{*}
\bigr\rangle_{\Psi_T \Psi_0}.
$$ (force_lrdmc_pulay)

Here $\langle\cdot\rangle_{\Psi_T \Psi_0}$ is the average over the
mixed distribution sampled by the LRDMC walker population (in
contrast to $\langle\cdot\rangle$ over $\Psi_T^{2}$ in MCMC). The
mixed estimator is *biased*: the sampling distribution
$\Psi_T \Psi_0$ differs from the pure distribution $\Psi_0^{2}$.
The bias vanishes when $\Psi_T = \Psi_0$ and is generally small
when $\Psi_T$ is well-optimized.

### Pathak-Wagner regularization

Near the nodal surface of $\Psi_T$, the local energy and its
derivatives can diverge, producing walkers with large
$|\partial e_L / \partial \mathbf{R}_{I}|$ that inflate the
statistical fluctuations of the force estimator. To suppress these
contributions, jQMC can apply the Pathak-Wagner reweighting
[Pathak and Wagner, 2020](https://doi.org/10.1063/5.0004008):

$$
\overline{O} =
\bigl\langle O(\mathbf{x})\,R_{\epsilon}(\mathbf{x}) \bigr\rangle,
$$ (regularized_estimator)

with the regularization kernel

$$
R_{\epsilon}(\mathbf{x}) =
\begin{cases}
7\,\bigl|\boldsymbol{\zeta}/\epsilon\bigr|^{6}
- 15\,\bigl|\boldsymbol{\zeta}/\epsilon\bigr|^{4}
+ 9\,\bigl|\boldsymbol{\zeta}/\epsilon\bigr|^{2},
& \bigl|\boldsymbol{\zeta}/\epsilon\bigr| \le 1, \\
1, & \bigl|\boldsymbol{\zeta}/\epsilon\bigr| > 1,
\end{cases}
$$ (f_epsilon)

and the *nodal-distance vector*

$$
\boldsymbol{\zeta} =
\frac{\Psi(\mathbf{x})\,\nabla\Psi(\mathbf{x})}
     {|\nabla\Psi(\mathbf{x})|^{2}}.
$$ (zeta_nodal_distance)

Here $\nabla$ denotes the many-body gradient operator that collects
all $3 N_e$ single-particle derivatives into a single vector.
Equivalently, defining the gradient matrix

$$
D_{i,k} \equiv
\frac{\partial \Psi(\mathbf{x})}{\partial r_{i,k}},
$$ (grad_psi_matrix)

regarded as an $(N_e, 3)$ array, one has

$$
|\nabla\Psi(\mathbf{x})|^{2} = \|\mathbf{D}\|_{F}^{2},
$$ (grad_psi_norm)

$$
\bigl|\boldsymbol{\zeta}/\epsilon\bigr| =
\frac{|\Psi(\mathbf{x})|}{\epsilon\,\|\mathbf{D}\|_{F}},
$$ (zeta_eta_ratio)

with $\|\cdot\|_{F}$ the Frobenius norm. Geometrically,
$\boldsymbol{\zeta}$ is the *signed normal distance* between
$\mathbf{x}$ and the nearest nodal point of $\Psi$ to first order,
and $\epsilon$ is a user-controlled distance from the nodal surface
within which the regularization is active. Outside the strip
$|\boldsymbol{\zeta}| \le \epsilon$ the regularization is the identity
($R_{\epsilon} = 1$), so the bulk of the walker population contributes
unchanged.

### Regularized mixed force

Applying eq. {eq}`regularized_estimator` to the two terms of
eq. {eq}`force_lrdmc_mixed`,

$$
\mathbf{F}^{\mathrm{HF}}_{\mathrm{mixed}} =
-\,\mathrm{Re}\,\bigl\langle
\partial E^{R}\,R_{\epsilon}(\mathbf{x})
\bigr\rangle_{\Psi_T \Psi_0},
$$ (force_lrdmc_hf_reg)

$$
\mathbf{F}^{\mathrm{Pulay}}_{\mathrm{mixed}} =
-\,2\,\mathrm{Re}\,\bigl\langle
\bigl(e_L - E\bigr)\,\bigl(\partial \Psi^{R}\bigr)^{*}\,R_{\epsilon}(\mathbf{x})
\bigr\rangle_{\Psi_T \Psi_0}.
$$ (force_lrdmc_pulay_reg)

The kernel $R_{\epsilon}(\mathbf{x})$ is built once per sample from
$\Psi(\mathbf{x})$ and the gradient matrix $\mathbf{D}$, both of
which are already available from the kinetic-energy estimator
(sec. {ref}`sec:kinetic_energy`). The same derivative fields
$\partial E^R$ and $\partial \Psi^R$ used by the unregularized
estimator therefore suffice for the regularized one;
eqs. {eq}`force_lrdmc_hf_reg`-{eq}`force_lrdmc_pulay_reg` add only
a multiplicative kernel per sample, with no additional
$\Psi$-evaluations.

The width $\epsilon$ is set with the key `epsilon_PW` (bohr) in
`[lrdmc-bra]` or `[lrdmc-tau]`. The default `epsilon_PW = 0.0` switches the
regularization off ($R_{\epsilon} = 1$ for every walker), which gives the
bare mixed estimator of eq. {eq}`force_lrdmc_mixed`. A larger
$\epsilon$ smooths the estimator over a wider nodal strip and lowers the
variance at the price of a larger bias; the bias vanishes as
$\epsilon \to 0$, where the force distribution regains its fat tails. A
practical check is to repeat the calculation with two or three values of
`epsilon_PW` and verify that the forces agree within error bars (or
extrapolate to $\epsilon \to 0$). When `atomic_force` and `compute_stress`
({ref}`sec:stress_pressure`) are both on, the same per-walker factor
$R_{\epsilon}$ also multiplies the LRDMC stress samples. The regularization is
not available in `[mcmc]`.

(sec:stress_pressure)=

## Stress tensor and pressure (PBC)

Set `compute_stress = true` in `[mcmc]`, `[lrdmc-bra]`, or `[lrdmc-tau]`
to estimate the stress tensor and pressure of a periodic system;
twist-averaged runs report the twist-averaged tensor. The switch is
independent of `atomic_force`. The detailed strain-response formulation is
work in progress and is not reproduced here.

Stress is the energy derivative with respect to uniform strain, divided by the cell volume. Pressure is minus one third of its trace. The reported tensor is symmetrized. Diagonal stress components are negative
under compression, and pressure is positive under compression; `P_hydro` is
the hydrostatic pressure. Units are Ha/bohr$^3$
($1\ \text{Ha/bohr}^3 \approx 2.9421\times 10^{4}$ GPa). The strain
derivative is taken at fixed trial-state parameters; periodic DTN cutoffs
use the comoving continuation.

### Estimator and output

LRDMC stress is a mixed estimator and has the same mixed bias as the LRDMC
force. With `epsilon_PW > 0` and `atomic_force = true`, the
Pathak--Wagner factor multiplies the LRDMC stress samples. No
space-warp-like transformation is applied.

Error bars use the same delete-one-bin jackknife as the energy and forces
(`num_mcmc_bin_blocks`, `num_gfmc_bin_blocks`). The pressure is evaluated
trace first on each jackknife replicate; the normal stresses are strongly
correlated, so combining the error bars of `P_x`, `P_y`, and `P_z` in
quadrature is wrong.

With `compute_stress = true` the output contains, after the atomic forces:

- `Stress Tensor (sigma):` `sigma_xx`, `sigma_yy`, `sigma_zz`, `sigma_xy`,
  `sigma_xz`, `sigma_yz` (Ha/bohr$^3$);
- `Pressure (P = -diag(sigma)):` `P_x`, `P_y`, `P_z`, and `P_hydro` with the
  trace-first error bar (Ha/bohr$^3$).

For a non-periodic system the key is ignored: no table is printed and the
Python methods return zeros. From Python, construct `MCMC`, `GFMC_t`, or
`GFMC_n` with `comput_stress=True` and call `get_aT`, `get_aP`, or
`get_P_hydrostatic`; these raise `ValueError` without that constructor flag.

### Checks and limitations

For an all-electron Coulomb system with a trial function that depends on
electron positions only through fractional coordinates (for example a
$\Gamma$-point plane-wave determinant without Jastrow), the virial relation
holds exactly. For a generic trial function, or with an ECP, it holds only
approximately and is a check of sign and magnitude.

Limitations: periodic systems only; the strain derivative is taken at fixed
trial-state parameters; the LRDMC stress is a mixed estimator; and no
variance reduction specific to the stress is applied.
