(lrdmc_tags)=

# Lattice Regularized Diffusion Monte Carlo (LRDMC)

> **PBC / twist scope.** Both GFMC drivers (`lrdmc-tau`/GFMC_t and
> `lrdmc-bra`/GFMC_n) accept periodic Gamma and non-Gamma complex-Bloch
> trials. Twist-averaged runs execute one GFMC chain per twist and combine the
> completed per-twist estimates with the stored Brillouin-zone weights;
> walker populations of different twists are never pooled.
> Atomic forces and stress are also available for non-Gamma complex trials.

## Overview

Lattice regularized diffusion Monte Carlo (LRDMC), initially proposed by Casula[^1], is a projection technique that systematically improves a variational Ansatz. It is based on Green's function Monte Carlo (GFMC)[^2][^3][^4], filtering the ground-state wavefunction from a given trial wavefunction.

The remainder of this section walks through the LRDMC formalism in
the form actually implemented in jQMC: the power-method projection,
the importance-sampling Green's function and its statistical
interpretation, the multi-walker branching scheme, the
continuous-time limit, and the *load-balanced* variant introduced
in jQMC[^9] that keeps every walker on the same number of
projection steps between two branchings.

**In jQMC.** The two algorithms correspond to two job types of the `jqmc`
command:

| Algorithm on this page | `job_type` | Driver class | Projection between branchings |
|---|---|---|---|
| Conventional (continuous time), {ref}`subsec:lrdmc_conventional` | `"lrdmc-tau"` | `GFMC_t` | imaginary time $\tau$, key `tau` |
| Load-balanced, {ref}`subsec:lrdmc_load_balanced` | `"lrdmc-bra"` | `GFMC_n` | $N_{\rm proj}$ projection steps, key `num_mcmc_per_measurement` |

The Python workflow `LRDMC_Workflow` uses `lrdmc-tau` by default. The input
keys are summarized in {ref}`subsec:lrdmc_parameters`.


## LRDMC formalism

(sec:lrdmc_complex_phase)=
### Complex phases and the real propagation weights

For complex trial states, propagation uses a fixed-phase approximation with real propagation weights; for a real trial state this reduces to the fixed-node approximation ([Ortiz, Ceperley and Martin, DOI: 10.1103/PhysRevLett.71.2777](https://doi.org/10.1103/PhysRevLett.71.2777)). Nonlocal ECPs use the same `tmove` and `dltmove` options as for real trial states. Both the XLA and the Rust GFMC routes implement it. The detailed formulation is work in progress.

### Power-method projection

The ground state $|\Upsilon_0\rangle$ of a Hamiltonian
$\hat{\mathcal{H}}$ is filtered from a guiding wave function
$|\Psi_{\rm G}\rangle$ by repeated application of the projector
$(\Lambda - \hat{\mathcal{H}})$. Expanding $|\Psi_{\rm G}\rangle$
on the (assumed complete) eigenbasis,

$$
|\Psi_{\rm G}\rangle = \sum_{i \ge 0} a_i\,|\Upsilon_i\rangle,
$$ (lrdmc_psiG_expansion)

one obtains

$$
\begin{aligned}
|\Upsilon_0\rangle
&\propto
\lim_{M\to\infty}\,(\Lambda - \hat{\mathcal{H}})^{M}\,|\Psi_{\rm G}\rangle \\
&=
\lim_{M\to\infty}\,(\Lambda - E_0)^{M}\,
\Bigl[
a_0\,|\Upsilon_0\rangle
+ \sum_{i\ne 0}
\Bigl(\frac{\Lambda - E_i}{\Lambda - E_0}\Bigr)^{M}\,
a_i\,|\Upsilon_i\rangle
\Bigr],
\end{aligned}
$$ (lrdmc_power_method)

with $\Lambda_{x,x'} = \lambda\,\delta_{x,x'}$ and $E_i$ the $i$-th
eigenvalue of $\hat{\mathcal{H}}$. Provided the ground state is
non-degenerate, $|(\Lambda - E_i)/(\Lambda - E_0)| < 1$ for all
$i \ge 1$, and the trial state is not orthogonal to the true
ground state ($a_0 = \langle \Psi_{\rm G} | \Upsilon_0 \rangle
\ne 0$), the projection converges to $|\Upsilon_0\rangle$. The
condition on the ratios is automatic on a lattice for large enough
$\lambda$; on the continuum, where the ab-initio
$\hat{\mathcal{H}}$ has an unbounded spectrum from above, it is
realized only in the $\lambda \to \infty$ limit (which is taken
later explicitly).

The projection is implemented iteratively,

$$
|\Psi_{n+1}\rangle = (\Lambda - \hat{\mathcal{H}})\,|\Psi_{n}\rangle,
$$ (lrdmc_iteration)

so that $|\Psi_n\rangle \to |\Upsilon_0\rangle$ up to a
normalization factor as $n \to \infty$.

### Importance-sampling Green's function

Expanding eq. {eq}`lrdmc_iteration` over a configuration basis and
multiplying through by $\Psi_{\rm G}(x') / \Psi_{\rm G}(x)$
introduces the *importance-sampling* Green's function used in
practice,

$$
\Psi_{\rm G}(x')\,\Psi_{n+1}(x') =
\sum_{x} \mathcal{G}_{x',x}\,\Psi_{\rm G}(x)\,\Psi_{n}(x),
$$ (lrdmc_master_eq)

$$
\mathcal{G}_{x',x} =
\langle x' | \Lambda - \hat{\mathcal{H}} | x \rangle\,
\frac{\langle x' | \Psi_{\rm G}\rangle}{\langle x | \Psi_{\rm G}\rangle}
=
\lambda\,\delta_{x',x}
- \mathcal{H}_{x',x}\,
\frac{\Psi_{\rm G}(x')}{\Psi_{\rm G}(x)},
$$ (lrdmc_greens_function)

with $\Psi_n(x) = \langle x | \Psi_n \rangle$. The mixed
distribution $\Pi_n(x) = \Psi_{\rm G}(x)\,\Psi_n(x)$ plays the
role of a probability and $\mathcal{G}$ of a transition kernel --
but with two caveats.

#### The Fermion sign problem

For fermions, $\mathcal{G}_{x',x}$ is not always non-negative: some
matrix elements have the wrong sign. jQMC circumvents this via the
fixed-node approximation (FNA) by defining an effective FN
Hamiltonian whose Green's function is non-negative by construction.
The technical details of the FNA and the associated lattice
regularization are out of scope here; in what follows we assume
non-negativity and refer to the FN Hamiltonian as
$\hat{\mathcal{H}}$ for brevity. See ref. [^1] and the textbook
[^10] for the full derivation.

#### Normalization and walker weights

Even when non-negative, $\mathcal{G}_{x',x}$ is not row-normalized.
Introducing the row sum

$$
b_x = \sum_{x'} \mathcal{G}_{x',x},
$$ (lrdmc_b_x)

one defines the normalized transition probability

$$
p_{x',x} = \frac{\mathcal{G}_{x',x}}{b_x},
$$ (lrdmc_normalization_probability)

which is non-negative and satisfies $\sum_{x'} p_{x',x} = 1$. The
Markov process then carries both a configuration and a weight,
$(x_n, w_n)$, evolving as

$$
x_{n+1} = x' \quad \text{with probability}\;p_{x',x_n},
$$ (lrdmc_projection_1)

$$
w_{n+1} = w_n\,b_{x_n}.
$$ (lrdmc_projection_2)

In master-equation form,

$$
\mathcal{P}_{n+1}(x', w') =
\sum_x \int dw\;K(x',w'\,|\,x,w)\,\mathcal{P}_n(x,w),
$$ (lrdmc_master_kernel)

with transition kernel
$K(x',w'\,|\,x,w) = p_{x',x}\,\delta(w' - w\,b_x)$. Multiplying by
$w'$ and integrating reproduces the importance-sampling Green's
function on the *weighted* marginal,

$$
\int dw'\,w'\,\mathcal{P}_{n+1}(x',w') =
\sum_x \mathcal{G}_{x',x}\,
\int dw\,w\,\mathcal{P}_n(x,w),
$$ (lrdmc_weighted_marginal)

so that

$$
\int dw\,w\,\mathcal{P}_n(x,w) = \Pi_n(x) =
\Psi_{\rm G}(x)\,\Psi_n(x).
$$ (lrdmc_marginal_pi)

That is, the *weighted* marginal of the $(x,w)$ process is the
mixed distribution that the power method projects onto the
ground state. The ground-state energy is estimated at projection
step $n$ (exactly in the limit $n\to\infty$, where
$|\Psi_n\rangle \propto |\Upsilon_0\rangle$) as

$$
E_0 =
\frac{\langle\Psi_{\rm G}|\hat{\mathcal{H}}|\Upsilon_0\rangle}
     {\langle\Psi_{\rm G}|\Upsilon_0\rangle}
= \frac{\sum_x e_L(x)\,\Psi_{\rm G}(x)\,\Psi_n(x)}
       {\sum_{x'}\Psi_{\rm G}(x')\,\Psi_n(x')},
$$ (lrdmc_E0_definition)

with the local energy

$$
e_L(x) =
\frac{\langle\Psi_{\rm G}|\hat{\mathcal{H}}|x\rangle}
     {\langle\Psi_{\rm G}|x\rangle}
= \sum_{x'} \mathcal{H}_{x,x'}\,
\frac{\Psi_{\rm G}(x')}{\Psi_{\rm G}(x)}.
$$ (lrdmc_local_energy)

### Multi-walker estimator

In practice, the marginal in eq. {eq}`lrdmc_marginal_pi` is
realized by an ensemble of $N_w$ *walkers* indexed by $\alpha$.
Each walker carries a dyad $(x_{\alpha,n}, w_{\alpha,n})$
distributed according to $\mathcal{P}_n(x,w)$, and the
ground-state energy is estimated as

$$
E_0 \;\simeq\;
\frac{\bigl\langle e_L(x_n)\,w_n \bigr\rangle}
     {\bigl\langle w_n \bigr\rangle},
$$ (lrdmc_E0_estimator)

with $\langle\cdot\rangle$ the statistical average over
*independent* Markov chains at step $n$
(sec. {ref}`sec:error-analysis`).

### Branching / correlated reconfiguration

Pure power iteration would let some walker weights diverge while
others vanish, and the variance of
eq. {eq}`lrdmc_E0_estimator` would blow up. jQMC, like every
GFMC / DMC code, mitigates this with a *branching* (a.k.a.
*reconfiguration*) step performed every $N_{\rm proj}$ projection
steps:

1. Pick the new walker configurations from the current ones with
   probability proportional to their pre-branching weights,

$$
p_{\alpha,n} = \frac{w_{\alpha,n}}{\sum_\beta w_{\beta,n}}.
$$ (lrdmc_branching_2)

2. Reset every walker weight to the population mean,

$$
w_{\alpha,n+1} = \bar w_n
= \frac{1}{N_w}\sum_\beta w_{\beta,n},
\qquad \forall\,\alpha.
$$ (lrdmc_branching_1)

The selection in step 1 is performed by the *correlated-random*
scheme of Buonaura and Sorella[^3] (see also ref. [^10]): the
$N_w$ random numbers

$$
Z_\alpha = \frac{\xi + \alpha - 1}{N_w},
\qquad \alpha = 1, \dots, N_w,
$$ (lrdmc_correlated_random_numbers)

are drawn from a single uniform variate $\xi \in [0,1)$ and
compared against the cumulative sums of the probabilities
$p_{\alpha,n}$ of eq. {eq}`lrdmc_branching_2`: walker $\beta$ is
selected for $Z_\alpha$ when
$\sum_{\gamma<\beta} p_{\gamma,n} \le Z_\alpha < \sum_{\gamma\le\beta} p_{\gamma,n}$. As proved in refs. [^3][^10], this
reconfiguration preserves the statistical averages while keeping
the total walker count fixed. It therefore alleviates the
population-control bias that affects the conventional DMC
branching scheme[^3]. The intrinsic exponential scaling of
projection methods with system size, originating from
walker-weight fluctuations[^11], persists but is not a problem
for typical applications[^11].

The *survived-walkers ratio* printed at the end of an LRDMC run is the
fraction of walkers that are selected at least once in step 1 (the number of
distinct parents divided by $N_w$), accumulated over all branchings. A ratio
close to 1 means that the weights are well balanced; a small ratio means that
many walkers are killed at each branching and the effective population is
small.

To make full use of the accumulated samples, observables are
evaluated using all post-equilibration data. After a
thermalization time $p$ (in branching-step units), the
configurations follow an equilibrium distribution $\pi_{\rm eq}(x)$.
Restarting the projection $p$ steps in the past with unit weights
and accumulating the average weights into the *global population
factor*

$$
\bar G_n^{\,p} = \prod_{j=1}^{p} \bar w_{n-j},
$$ (lrdmc_global_population_factor)

the ground-state energy is then estimated as

$$
E_0 \;\simeq\;
\frac{\sum_n \bar G_n^{\,p}\,\bar e_L(x_n)}
     {\sum_n \bar G_n^{\,p}},
$$ (lrdmc_E0_global_estimator)

with the population-averaged local energy

$$
\bar e_L(x_n) =
\frac{\sum_\alpha w_{\alpha,n}\,e_L(x_{\alpha,n})}
     {\sum_\alpha w_{\alpha,n}},
$$ (lrdmc_local_energy_weighted)

evaluated just before each reconfiguration, with weights
$w_{\alpha,n}$ accumulated only during the evolution since the
preceding branching step.

### Lattice regularization of the continuum Hamiltonian

The procedure above assumes a *lattice* Hamiltonian for which
$\mathcal{G}_{x',x}$ can be made non-negative for large enough
$\lambda$. The original ab-initio Hamiltonian is continuous and
its diagonal can be arbitrarily large because of the Coulomb
divergence at the nucleus. LRDMC therefore replaces the
continuous Hamiltonian by a *lattice* one $\hat{\mathcal{H}}^a$
of mesh size $a$,

$$
\hat{\mathcal{H}}^a \to \hat{\mathcal{H}}
\qquad\text{as}\qquad a \to 0,
$$ (lrdmc_lattice_limit)

in which each electron can hop only to a finite set of
nearest-neighbour grid points. The resulting hopping matrix
$\mathcal{H}_{x',x}^{a}$ is sparse and bounded. LRDMC energies
carry a $\mathcal{O}(a^2)$ bias and are extrapolated to $a \to 0$
by running the simulation at several mesh sizes. The mesh size $a$ is
the TOML key `alat` (bohr).

Each electron hops by $\pm a$ along the three axes of a Cartesian frame.
The frame is rotated by a uniformly distributed random rotation, which is
redrawn at the start of every projection interval, so that no direction of
space is privileged on average. The mesh is local to each walker: there is
no global grid, and in a periodic cell $a$ need not be commensurate with the
lattice vectors.

**Coulomb potential on the lattice.** Without ECPs, the electron-ion
attraction $-Z_I/r$ diverges at the nucleus. In the propagation local energy
(see below) the electron-ion distance is floored at the mesh size,
$r \to \max(r, a)$; for a molecule this gives $-Z_I/\max(r_{iI}, a)$. For a
periodic system the same floor is applied to the electron-ion distance inside
the real-space (screened) part of the Ewald sum (see {doc}`hamiltonian`), while
the reciprocal-space and uniform-background parts, which are bounded, are
unchanged. The floor only affects configurations with an electron closer than
$a$ to a nucleus (or to one of its periodic images), and it disappears as
$a\to 0$. With ECPs the bare electron-ion term is used, as in the continuum.

For periodic systems the walker coordinates are not wrapped into the cell
during the projection; see {ref}`sec:pbc_wrap_conventions`.

### Propagation and measurement local energies

jQMC keeps two local-energy compositions during LRDMC. They have different
roles and must not be interchanged:

- $e_L^{\mathrm{prop},a}$ is the fixed-node lattice local energy. It contains
  the six-point lattice hopping terms and the sign-flip maximum used to make
  the importance-sampled Green's function non-negative. It controls walker
  weights, projection, and branching only.
- $e_L^{\mathrm{meas}}$ is the continuum measurement local energy. It contains
  the continuum kinetic energy and the bare (unfloored) electron-ion potential, together with
  the electron-electron, ion-ion, local-ECP, and localized non-local-ECP
  contributions where applicable. It contains neither lattice hopping terms
  nor the propagation-only sign-flip maximum.

For the Cartesian six-point kinetic stencil, the propagation composition uses
$3/a^2$ on the diagonal for each electron. Each directed off-diagonal hopping
matrix element is $-\Psi(x_s)/(2a^2\Psi(x))$; equivalently, the zero-variance
reconstruction adds the $1/(2a^2)$ background for each of the six directions.
On a configuration where the sign-flip maximum is inactive, the zero-variance
identity makes the lattice and continuum compositions equal. They can differ
on clipped configurations; that difference is intentional.

The energy returned by `GFMC_t.get_E` or `GFMC_n.get_E`, written to a
checkpoint, and printed by the CLI is the weighted estimator of
$e_L^{\mathrm{meas}}$:

$$
E_{\mathrm{LRDMC}} =
\frac{\left\langle w\,e_L^{\mathrm{meas}}\right\rangle}
     {\left\langle w\right\rangle},
$$ (lrdmc_novar0_measurement)

where the weights $w$ were generated by $e_L^{\mathrm{prop},a}$. Atomic-force
and stress estimators differentiate the same continuum measurement energy that
is reported by `get_E`; the Pulay/covariance terms still use the projected
walker weights.

The measurement composition being continuous does **not** remove the lattice
spacing bias: the sampled distribution and weights still come from the lattice
propagator. Production energies, forces, and stresses therefore still require
an `alat -> 0` extrapolation.

### Zero-variance-zero-bias property near the node

The energy estimator eq. {eq}`lrdmc_E0_estimator` has the
*zero-variance-zero-bias* property: when
$|\Psi_{\rm G}\rangle = |\Upsilon_0\rangle$, the local-energy
fluctuation vanishes because $|\Upsilon_0\rangle$ is an eigenstate of
the (fixed-node) Hamiltonian. With a variational
$|\Psi_{\rm G}\rangle \ne |\Upsilon_0\rangle$, the energy variance
remains *finite* because the divergence of $e_L$ near the nodal
surface is cancelled by the vanishing of the trial and fixed-node
wave functions[^12].

Concretely, let $\delta$ denote the distance from the nodal
surface. Then $\Psi_{\rm G}, \Upsilon_0 \propto \delta$,
$e_L \propto \delta^{-1}$, and the sampling density behaves as
$P(x) \propto \delta^{2}$. The energy variance integral is

$$
\int_\epsilon^{1} d\delta\,P(\delta)\,e_L(\delta)^{2}
\sim
\int_\epsilon^{1} d\delta\,\delta^{2}\,\delta^{-2}
= \int_\epsilon^{1} d\delta,
$$ (lrdmc_variance_finite)

which is finite. *Derivatives* of $e_L$ behave as
$\partial e_L \propto \delta^{-2}$, however, and the variance of
the corresponding force estimators diverges as
$\int d\delta\,\delta^{-2}$ near the node. A reweighting[^13] or
regularization[^14] technique is therefore needed to cure the
divergence of force / parameter-derivative variances; for LRDMC forces
jQMC offers the Pathak-Wagner regularization (key `epsilon_PW`, off by
default) documented in sec. {ref}`subsec:forces_lrdmc`.


(subsec:lrdmc_conventional)=

## Conventional LRDMC algorithm

### Continuous-time formulation

A naive implementation of eqs. {eq}`lrdmc_projection_1` and
{eq}`lrdmc_projection_2` with a finite $\lambda$ is inefficient
on the continuum: for the Green's function to be non-negative,
$\lambda$ must dominate every diagonal Hamiltonian matrix element
$\mathcal{H}_{x,x}$, which makes $\mathcal{G}_{x,x}$ much larger
than $\mathcal{G}_{x'\ne x,x}$, so $p_{x,x} \gg p_{x'\ne x,x}$ and
the walker spends very long stretches at the same configuration
(long autocorrelation, inefficient sampling).

The standard cure is to take the limit
$\lambda \to \infty$ analytically and turn the projection into a
*continuous-time* stochastic process[^4][^15]. In this limit, the
algorithm draws the number $k$ of successive diagonal moves at a
configuration $x$ from a uniform variate $\xi \in [0,1)$,

$$
k = \Bigl\lfloor -\,\frac{\ln(1-\xi)}{1 - p_{x_n,x_n}} \Bigr\rfloor,
$$ (lrdmc_k_times_there)

which is, to leading order in $1 - p_{x_n,x_n}$, the geometric draw
$\lfloor \ln(1-\xi)/\ln p_{x_n,x_n} \rfloor$,

with weight update

$$
w_{n+1} = w_n\,b^{\,k}.
$$ (lrdmc_update_walker_weight_discretized)

Setting $\delta\tau \equiv 1/\lambda \to 0$ and keeping the
imaginary time $\tau = M/\lambda$ finite,
$(\Lambda - \hat{\mathcal{H}})^M \to \exp(-\tau \hat{\mathcal{H}})$
up to an irrelevant constant. To leading order in $\delta\tau$,

$$
1 - p_{x,x}
= \frac{\sum_{x'\ne x} \mathcal{G}_{x',x}}{\lambda - e_L(x)}
\;\to\; \delta\tau \sum_{x'\ne x} \mathcal{G}_{x',x}
+ \mathcal{O}\bigl((\delta\tau)^2\bigr),
$$ (lrdmc_small_tau_limit)

so the persistence time at configuration $x_n$ is the random
number

$$
\tau_\xi = -\,\frac{\ln(1-\xi)}
                   {\sum_{x' \ne x_n} \mathcal{G}_{x',x_n}},
$$ (lrdmc_tau_xi)

corresponding to $\tau_\xi = k\,\delta\tau$. The walker weight
update eq. {eq}`lrdmc_update_walker_weight_discretized` then
becomes

$$
w_{n+1} = w_n\,\exp\bigl(-\,\tau_\xi\,e_L(x_n)\bigr),
$$ (lrdmc_weight_update_conventional)

and the off-diagonal jump is drawn with the normalized
probability $\mathcal{G}_{x',x_n}/\sum_{x'\ne x_n}\mathcal{G}_{x',x_n}$.
The imaginary time remaining before the next branching is
decremented as
$\tau_{\rm left} \leftarrow \tau_{\rm left} - \tau_\xi$ until it
reaches zero. The branching projection interval $\tau$ between
two reconfigurations is an a-priori input (TOML key `tau` of
`[lrdmc-tau]`, default 0.10).

The advantage of the continuous-time formulation is that no
Suzuki-Trotter decomposition is needed -- $\exp(-\tau \hat{\mathcal{H}})$
is sampled exactly, with no Trotter error inherited from a
finite imaginary-time step. The price to pay is the lattice
discretization bias $\mathcal{O}(a^2)$, removed by the $a \to 0$
extrapolation discussed earlier.

### Intrinsic load-imbalance in parallel

The continuous-time formulation has a parallel-efficiency
drawback that the user does not see in serial runs but that
becomes acute on many GPUs / nodes. Because every walker
independently samples its persistence times $\tau_\xi$
(eq. {eq}`lrdmc_tau_xi`), the *number* of off-diagonal moves
needed to reach $\tau_{\rm left} = 0$ varies randomly from walker
to walker. The branching step must wait for the slowest walker
to finish; walkers that finish early sit idle. With $N_w$
walkers the maximum number of moves per branching grows
logarithmically as $\log(N_w)$, so the actual wall-clock per
branching step grows as $\log(N_w)$ even though the per-walker
work is constant on average.

Two seemingly simple fixes do not work in practice:

- Choosing a *large* projection time $\tau$ between branchings
  evens out the per-walker work, but at large $\tau$ the weight
  distribution becomes highly imbalanced, the survival rate at
  branching collapses, the effective sample size shrinks, and
  the statistical error inflates. There is a sharp tradeoff
  between load-imbalance reduction and statistical efficiency.
- Eq. {eq}`lrdmc_weight_update_conventional` makes the weights
  grow exponentially in $\tau$, which can overflow numerically;
  the global population factor
  eq. {eq}`lrdmc_global_population_factor` overflows for the
  same reason.

jQMC therefore offers a second algorithm in which the number of
projection steps between two branchings is set *a priori* and is
identical for every walker.


(subsec:lrdmc_load_balanced)=

## Load-balanced LRDMC algorithm

The load-balanced LRDMC algorithm[^9] removes the per-walker
variability in the number of projection steps by defining a
*new* projection process driven by an auxiliary off-diagonal
Hamiltonian -- one with zero diagonal -- so that every walker
hops to a different site at every step, never lingering.

### Auxiliary off-diagonal Hamiltonian

Split the FN LRDMC Hamiltonian into its off-diagonal and
diagonal parts,

$$
\mathcal{H}_{x',x} =
\bar{\mathcal{H}}_{x',x} + \delta_{x,x'}\,\mathcal{W}(x),
\qquad
\bar{\mathcal{H}}_{x,x} = 0,
$$ (lrdmc_lb_H_split)

with $\mathcal{W}(x) = \mathcal{H}_{x,x}$. The split is
well-defined for all $x$ thanks to the lattice regularization,
which cuts off the Coulomb divergence of the bare ab-initio
potential by the length scale $a$. The Schrodinger equation for
the FN ground state then reads

$$
\sum_x \bar{\mathcal{H}}_{x',x}\,\Upsilon_0(x) =
\bigl(E_0 - \mathcal{W}(x')\bigr)\,\Upsilon_0(x').
$$ (lrdmc_lb_schrodinger)

### Modified Green's function and stochastic process

By analogy with eq. {eq}`lrdmc_greens_function`, define the
*modified* importance-sampling Green's function

$$
\mathcal{G}_{x',x} =
-\,\frac{1}{\mathcal{W}(x) - E_0}\,
\bar{\mathcal{H}}_{x',x}\,
\frac{\Psi_{\rm G}(x')}{\Psi_{\rm G}(x)},
$$ (lrdmc_lb_modified_greens_function)

and decompose it as $\mathcal{G}_{x',x} = b_x\,p_{x',x}$ with

$$
\begin{aligned}
b_x &= \frac{1}{\mathcal{W}(x) - E_0}\,\bar b_x, \\
\bar b_x &= -\,\sum_{x'}
\Psi_{\rm G}(x')\,\bar{\mathcal{H}}_{x',x}/\Psi_{\rm G}(x), \\
p_{x',x} &=
-\,\frac{\Psi_{\rm G}(x')\,\bar{\mathcal{H}}_{x',x}/\Psi_{\rm G}(x)}
        {\bar b_x}.
\end{aligned}
$$ (lrdmc_lb_partition)

For $p_{x',x}$ to be a normalized transition probability and
$b_x$ a legitimate weight, both must be non-negative. The
non-negativity of $\bar b_x$ and $p_{x',x}$ is the standard FN
guarantee. The non-negativity of $b_x$ requires
$\mathcal{W}(x) > E_0$ for all $x$, which follows from a
variational argument: the trial state
$\phi(x) = \delta_{x,x_0}$ has variational energy
$\mathcal{W}(x_0)$, so $E_0 \le \mathcal{W}(x_0)$; since $x_0$ is
arbitrary, $E_0 \le \mathcal{W}(x)$ for every $x$ (with equality only in
the trivial case where $\phi$ is the ground state).

The walker dyad then evolves by the same rules as in the
conventional algorithm,

$$
x_{n+1} = x'\;\text{with probability}\;p_{x',x_n},
\qquad
w_{n+1} = w_n\,b_{x_n},
$$ (lrdmc_lb_lattice_update)

but now *every* step is an off-diagonal jump: the walker never
remains at the same configuration. The number of projection
steps between two branchings, $N_{\rm proj}$, is therefore
identical across walkers and is set as an a-priori input (TOML key
`num_mcmc_per_measurement` of `[lrdmc-bra]`, required, with no default).
One projection advances imaginary time by roughly $a^2/(3N_e)$, so a fixed
$N_{\rm proj}$ covers less imaginary time between branchings, and with it
shorter warmup and weight-history windows, as $a \to 0$ or as $N_e$ grows;
choose $N_{\rm proj}$ for the system and the mesh size.

### Stationary distribution and observable estimator

Because the auxiliary Hamiltonian carries an extra factor
$\mathcal{W}(x) - E_0$ in its denominator
(eq. {eq}`lrdmc_lb_modified_greens_function`), the weighted
marginal of the modified process is no longer
eq. {eq}`lrdmc_marginal_pi` but is modulated by the same factor:

$$
\int dw\,w\,P_n(x,w)
\;=\;
\bigl(\mathcal{W}(x) - E_0\bigr)\,\Psi_{\rm G}(x)\,\Upsilon_0(x).
$$ (lrdmc_lb_stationary)

For a guiding function $\Psi_{\rm G}(x)$ close enough to the FN ground
state $\Upsilon_0(x)$, the process converges to the distribution
eq. {eq}`lrdmc_lb_stationary` as the number of projections grows.

The factor $\mathcal{W}(x) - E_0$ in the stationary distribution
has to be compensated by an inverse factor in the weight when
averaging observables. The ground-state energy estimator becomes

$$
E_0 \;\simeq\;
\frac{\Bigl\langle e_L(x_n)\,
\dfrac{w_n}{\mathcal{W}(x_n) - E_0}\Bigr\rangle}
     {\Bigl\langle
\dfrac{w_n}{\mathcal{W}(x_n) - E_0}\Bigr\rangle},
$$ (lrdmc_lb_E0)

with $e_L(x) = -\bar b_x + \mathcal{W}(x)$ the local energy of
the FN Hamiltonian in eq. {eq}`lrdmc_lb_H_split` and
$\langle\cdot\rangle$ the average over independent Markov chains
at step $n$. The value of $E_0$ that enters
eq. {eq}`lrdmc_lb_modified_greens_function` is therefore
*self-consistent*: it is updated during the run from a running estimate
of eq. {eq}`lrdmc_lb_E0` (at every branching step at the beginning of the
run, less often later). The starting value is the TOML key `E_scf` of
`[lrdmc-bra]`; set it to a reasonable estimate of the energy, such as the
VMC energy.

The multi-walker / branching machinery
(eqs. {eq}`lrdmc_branching_2`-{eq}`lrdmc_E0_global_estimator`)
carries over verbatim to the load-balanced algorithm, with one
adjustment: the population-averaged local energy
eq. {eq}`lrdmc_local_energy_weighted` is replaced by

$$
\bar e_L(x_n) =
\frac{\displaystyle
\sum_\alpha \frac{w_{\alpha,n}}{\mathcal{W}(x_{\alpha,n}) - E_0}\,
e_L(x_{\alpha,n})}
{\displaystyle
\sum_\alpha \frac{w_{\alpha,n}}{\mathcal{W}(x_{\alpha,n}) - E_0}}.
$$ (lrdmc_lb_local_energy_weighted)

This is the same extra $1/(\mathcal{W}(x) - E_0)$ rescaling that
relates eq. {eq}`lrdmc_lb_E0` to its conventional counterpart
eq. {eq}`lrdmc_E0_estimator`.

### Behavior near the nodal surface

The load-balanced algorithm inherits the
zero-variance-zero-bias property of the conventional one, but
its variance behavior near the nodal surface is milder. With
$\delta$ the distance from the node,
$\Psi_{\rm G}, \Upsilon_0 \propto \delta$,
$e_L \propto \delta^{-1}$ as before, but the extra factor
$\mathcal{W}(x) - E_0$ diverges as $\delta^{-1}$ near the node,
so the sampling density in the load-balanced process scales as
$P(x) \propto \delta$ (one less power of $\delta$ than in the
conventional case).

The energy variance integral

$$
\int_\epsilon^{1} d\delta\,P(\delta)\,
\Bigl(\frac{e_L(\delta)}{\mathcal{W}(\delta) - E_0}\Bigr)^{2}
\sim
\int_\epsilon^{1} d\delta\,\delta\,(1)^{2}
= \int_\epsilon^{1} d\delta\,\delta,
$$ (lrdmc_lb_variance_finite)

is finite, as in the conventional algorithm. For derivatives
($\partial e_L \propto \delta^{-2}$),

$$
\int_\epsilon^{1} d\delta\,P(\delta)\,
\Bigl(\frac{\partial e_L(\delta)}{\mathcal{W}(\delta) - E_0}\Bigr)^{2}
\sim
\int_\epsilon^{1} d\delta\,\delta\,(\delta^{-2}/\delta^{-1})^{2}
= \int_\epsilon^{1} d\delta\,\delta^{-1},
$$ (lrdmc_lb_derivative_variance)

which still diverges but only logarithmically, i.e. as
$\int d\delta\,\delta^{-1}$ rather than the conventional
$\int d\delta\,\delta^{-2}$. A reweighting / regularization
technique is therefore still required for force / parameter
derivatives, but the load-balanced algorithm has intrinsically
smaller error bars on those quantities at fixed statistics,
because the operator fluctuations near the node are reduced.

### Equivalence to the conventional algorithm and load balance

Because the projection step count $N_{\rm proj}$ is fixed for
every walker, the per-walker compute is strictly uniform, and
the parallel efficiency does not degrade with $N_w$. Weak-scaling
benchmarks of the two algorithms in ref. [^9] show that the
conventional algorithm slows down as $\log(N_w)$ while the
load-balanced algorithm remains flat -- which is the practical
reason for adopting it on GPU clusters.

Numerically, the two algorithms give consistent results in the
limit $a \to 0$ provided the *average* number of projection
steps is matched, confirming their equivalence at the level of
ground-state observables[^9].


(subsec:lrdmc_parameters)=

## Input parameters

The symbols of this page map onto the `[lrdmc-tau]` and `[lrdmc-bra]` sections
of the `jqmc` input as follows (defaults as in the input template):

| Symbol / quantity | TOML key | Section | Default | Notes |
|---|---|---|---|---|
| mesh size $a$ (bohr) | `alat` | both | `0.30` | run several values and extrapolate to $a\to 0$ |
| projection time $\tau$ between branchings | `tau` | `[lrdmc-tau]` | `0.10` | |
| projection steps $N_{\rm proj}$ between branchings | `num_mcmc_per_measurement` | `[lrdmc-bra]` | required | no default; one projection advances imaginary time by about $a^2/(3N_e)$ |
| initial $E_0$ of eq. {eq}`lrdmc_lb_modified_greens_function` (Ha) | `E_scf` | `[lrdmc-bra]` | `0.0` | updated self-consistently during the run; start from the VMC energy |
| history length $p$ of eq. {eq}`lrdmc_global_population_factor` | `num_gfmc_collect_steps` | both | `5` | minimum 5 |
| discarded measurements | `num_gfmc_warmup_steps` | both | `25` | minimum 25 |
| number of bins for the error bar | `num_gfmc_bin_blocks` | both | `10` | global count, minimum 10 (not a production recommendation); see {ref}`sec:error_analysis_bins` |
| non-local ECP treatment | `non_local_move` | both | `"tmove"` | `"tmove"` or `"dltmove"`, see below |
| force regularization | `epsilon_PW` | both | `0.0` | see {doc}`derivatives` |

The number of walkers is set per MPI process in `[control]`
(`number_of_walkers`).

With ECPs the non-local part of the pseudopotential cannot be included in a
fixed-node lattice Hamiltonian directly. `non_local_move = "tmove"` (default)
treats the off-diagonal ECP elements with the T-move scheme of
refs. [^1][^8], which keeps the LRDMC energy variational. `"dltmove"` combines
the T-move with the determinant locality approximation of ref. [^7], in which
the ECP is localized with the determinant part of $\Psi_{\rm G}$ only, so that
the result depends much less on the Jastrow factor. Use the same choice for all
values of `alat` in an extrapolation.

## Practical points

1. No time-step error: unlike standard DMC, LRDMC does not rely on a Suzuki-Trotter decomposition. Instead, a systematic bias comes from the finite lattice spacing $a$[^1]. To obtain an unbiased fixed-node (FN) energy, extrapolate results to $a \to 0$ using several lattice spacings; the $a \to 0$ extrapolation is typically smooth and well captured by low-order polynomial fits.
2. Consistency with DMC: after removing the controllable $a \to 0$ extrapolation, FN energies from LRDMC agree with standard DMC calculations[^5].
3. Multiple mesh sizes: the LRDMC literature describes using two mesh sizes ($a$ and $a'$) so that regions near nuclei and valence regions are diffused appropriately[^1][^6]. jQMC v0.3.0a1 uses a single mesh size.
4. Variational principle with ECPs: LRDMC retains the variational principle even in the presence of effective core potentials, analogous to the T-move treatment in standard DMC[^1][^7][^8]. The choice is made with `non_local_move` (see {ref}`subsec:lrdmc_parameters`).

`jQMC` implements an LRDMC algorithm that maintains parallel efficiency for many walkers and many nodes via load-balancing across walkers[^9].

For further algorithmic details, please see the textbook "Quantum Monte Carlo Approaches for Correlated Systems"[^10].

[^1]: M. Casula, C. Filippi, and S. Sorella, Phys. Rev. Lett. 95, 100201 (2005). DOI: [10.1103/PhysRevLett.95.100201](https://doi.org/10.1103/PhysRevLett.95.100201)
[^2]: D. F. Ten Haaf, H. J. Van Bemmel, J. M. Van Leeuwen, W. Van Saarloos, and D. M. Ceperley, Phys. Rev. B 51, 13039 (1995). DOI: [10.1103/PhysRevB.51.13039](https://doi.org/10.1103/PhysRevB.51.13039)
[^3]: M. Calandra Buonaura and S. Sorella, Phys. Rev. B 57, 11446 (1998). DOI: [10.1103/PhysRevB.57.11446](https://doi.org/10.1103/PhysRevB.57.11446)
[^4]: S. Sorella and L. Capriotti, Phys. Rev. B 61, 2599 (2000). DOI: [10.1103/PhysRevB.61.2599](https://doi.org/10.1103/PhysRevB.61.2599)
[^5]: F. Della Pia, et al., J. Chem. Phys. 163, 104110 (2025). DOI: [10.1063/5.0272974](https://doi.org/10.1063/5.0272974)
[^6]: K. Nakano, R. Maezono, and S. Sorella, Phys. Rev. B 101, 155106 (2020). DOI: [10.1103/PhysRevB.101.155106](https://doi.org/10.1103/PhysRevB.101.155106)
[^7]: K. Nakano, S. Sorella, D. Alfe, A. Zen, J. Chem. Theory Comput. 20, 4591-4604 (2024). DOI: [10.1021/acs.jctc.4c00139](https://doi.org/10.1021/acs.jctc.4c00139)
[^8]: M. Casula, Phys. Rev. B 74, 161102 (2006). DOI: [10.1103/PhysRevB.74.161102](https://doi.org/10.1103/PhysRevB.74.161102)
[^9]: K. Nakano, S. Sorella, M. Casula, J. Chem. Phys. 163, 194117 (2025). DOI: [10.1063/5.0296986](https://doi.org/10.1063/5.0296986)
[^10]: "Quantum Monte Carlo Approaches for Correlated Systems", Cambridge University Press (2017). DOI: [10.1017/9781316417041](https://doi.org/10.1017/9781316417041)
[^11]: N. Nemec, Phys. Rev. B 81, 035119 (2010). DOI: [10.1103/PhysRevB.81.035119](https://doi.org/10.1103/PhysRevB.81.035119)
[^12]: A. Zen, S. Sorella, M. J. Gillan, A. Michaelides, and D. Alfe, Phys. Rev. B 93, 241118(R) (2016). DOI: [10.1103/PhysRevB.93.241118](https://doi.org/10.1103/PhysRevB.93.241118)
[^13]: C. Attaccalite and S. Sorella, Phys. Rev. Lett. 100, 114501 (2008). DOI: [10.1103/PhysRevLett.100.114501](https://doi.org/10.1103/PhysRevLett.100.114501)
[^14]: S. Pathak and L. K. Wagner, AIP Advances 10, 085213 (2020). DOI: [10.1063/5.0004008](https://doi.org/10.1063/5.0004008)
[^15]: N. Trivedi and D. M. Ceperley, Phys. Rev. B 41, 4552 (1990). DOI: [10.1103/PhysRevB.41.4552](https://doi.org/10.1103/PhysRevB.41.4552)
