# Wave functions

Both the accuracy and the computational efficiency of QMC approaches crucially depend on the WF ansatz.
In particular, as discussed in {doc}`vmc` and {doc}`lrdmc`, QMC algorithms, both at the variational and fixed-node level, imply several calculations of the local energy $e_L(\mathbf{x})$ and the ratio $\Psi(\mathbf{x})/\Psi(\mathbf{x}')$ for different electronic configurations $\mathbf{x}$ and $\mathbf{x}'$. The computational cost of these operations determines the overall efficiency of QMC and its scaling with systems size.

jQMC employs a many-body WF ansatz $\Psi$ which can be written as the product of two terms:

$$
\Psi  =  \Phi _\text{AS} \times \exp J \,,
$$

where the term $\exp J$, conventionally dubbed Jastrow factor, is symmetric under electron exchange, and the term $\Phi _\text{AS}$, also referred to as the determinant part of the WF, is antisymmetric.
The resulting WF $\Psi$ is antisymmetric, thus fermionic.

In the majority of QMC applications, the chosen $\Phi _\text{AS}$ is a single Slater determinant (SD) $\Phi _\text{SD}$, i.e., an antisymmetrized product of single-electron WFs.
Clearly, SD alone does not include any correlation other than the exchange. However, when a Jastrow factor, explicitly depending on the inter-electronic distances, is applied to $\Phi _\text{SD}$ the resulting ansatz $\Psi_\text{JSD} = \Phi _\text{SD} * \exp J$
often provides over 70\% of the correlation energy[^1] at the variational level.
Thus, the Jastrow factor proves very effective in describing the correlation, employing only a relatively small number of parameters, and therefore providing a very efficient way to improve the ansatz.[^2]
A Jastrow correlated SD (JSD) function yields a computational cost for QMC simulations  -- both VMC and FN -- about $\propto N^3$, namely  the same scaling of  most DFT codes. Therefore, although QMC has a much larger prefactor,  it represents an approach much cheaper than traditional quantum chemistry ones,
at  least for large enough systems.

[^1]: The correlation energy is typically defined as the difference between the exact energy and the Hartree-Fock energy, which is the variational minimum for a SD ansatz.
[^2]: However, the Jastrow factor makes $\Psi$ not factorizable when expectation values of quantum operators are evaluated. For this reason it is not a feasible route to traditional quantum chemistry approaches, as it requires  stochastic approaches to evaluate efficiently the corresponding multidimensional integrals.

While the JSD ansatz is quite satisfactory in several applications, there are situations where very high accuracy is required, and a more advanced
ansatz is necessary.
The route to improve JSD is not unique, and different approaches have been attempted within the QMC community.
First, it should be mentioned that improving the Jastrow factors is not an effective approach to achieve higher accuracy at the FN level, as the Jastrow is positive and cannot change the nodal surface.

The strategy employed in jQMC is inspired by the pioneering work of TurboRVB and is that the route toward an improved ansatz should not compromise the efficiency and good scaling of QMC. For this reason, neither backflow nor explicit multideterminant expansions are implemented in the code. Within the jQMC project, the main goal is instead to consider an ansatz that can be implicitly equivalent to a multideterminant expansion, but remains
in practice as efficient as a single determinant.

jQMC currently offers three alternatives for the choice of $\Phi_\text{AS}$, which correspond to

1. the Antisymmetrized Geminal Power (AGP),
2. the AGP with constrained number of molecular orbitals (AGPn), and
3. the single Slater determinant.

In QMC, the WFs are always meant to include the Jastrow factor, which proves fundamental to improve the properties of the overall WF. For instance, AGP carries more correlation than SD. However, it is not size-consistent unless it is multiplied by a Jastrow factor.
Thus, a fundamental step to take advantage of the WF ansatz  is the possibility  to perform reliable optimizations of the parameters. Optimization is discussed in {ref}`sec:opt_wf`.
The basis sets and atomic orbitals used in the Jastrow and antisymmetric parts are described in {ref}`sec:atomic_orbitals`.


## Two many-body Bloch conditions and the determinant

The simulation-cell twist $\mathbf k_s$ is set by the input and fixes the common many-body boundary condition; there is no separate primitive-cell boundary setting. A positive real Jastrow does not change the determinant boundary phases.

Slater determinants and balanced AGP are supported. AGP uses an AO geminal expansion by default; MO factorization is optional. Imported orbitals must provide the correct spin boundary phase for either assembly; see {doc}`ao` and {doc}`orbital_tiling`.

Paired jQMC uses spin twists $\mathbf k_{s\uparrow}=+\mathbf k_s$ and $\mathbf k_{s\downarrow}=-\mathbf k_s$ with a positive Bloch phase. Supported regimes are real-$\Gamma$ and complex Bloch; see {ref}`sec:periodic_determinants`. The detailed formulation of these many-body boundary conditions is work in progress.

(sec:Jastrow)=

## Jastrow factor (J)
The Jastrow factor ($\exp J$) plays an important role
in improving the correlation of the WF and in fulfilling Kato's cusp condition [T. Kato, Commun. Pure Appl. Math. 10, 151 (1957)](https://doi.org/10.1002/cpa.3160100201).
jQMC implements the Jastrow term composed of one-body, two-body, and three/four-body factors ($J = {J_1}+{J_2}+{J_{3/4}}$).
The one-body and two-body factors are used to fulfill the electron-ion and electron-electron cusp conditions, respectively, and the three/four-body factors are employed to consider a systematic expansion, in principle converging
to the most general  independent  electron pairs contribution.

The one-body Jastrow factor $J_1$ is the sum of two parts, the homogeneous part (enforcing the electron-ion cusp condition):

$$
J_1^{\rm h} \left( \mathbf{r}_1,\ldots,\mathbf{r}_N \right) = \sum_{i=1}^{N_{\rm{e}}} \sum_{\alpha=1}^{N_\text{a}} \left( { { - {{\left( {2{Z^{\rm eff}_{\alpha}}} \right)}^{3/4}}u\left( {(2{Z^{\rm eff}_{\alpha}})^{1/4}\left| {{\mathbf{r}_i} - {{\mathbf{R}}_{\alpha}}} \right|} \right)} } \right),
$$ (onebody_J_hom)

and the corresponding inhomogeneous part:

$$
{J_1^{\rm inh}}\left( {{{\mathbf{r}}_1}, \ldots, {{\mathbf{r}}_N}} \right) =  \sum_{i=1}^{N_{\rm{e}}} \sum_{\alpha=1}^{N_\text{a}} \left( {\sum\limits_{l} {M_{\alpha,l} \chi_{\alpha,l}\left( {{{\mathbf{r}}_i}} \right)} } \right) ,
$$ (onebody_J_inhom)

where ${{{\mathbf{r}}_i}}$ are the electron positions, ${{{\mathbf{R}}_{\alpha}}}$ are the positions of the $N_\text{a}$ atoms,
$Z^{\rm eff}_{\alpha} = Z_\alpha - z_{{\rm core},\alpha}$ is the effective charge of atom $\alpha$
(the nuclear charge $Z_\alpha$ for an all-electron atom, the nuclear charge minus the number of core electrons removed by the pseudopotential otherwise),
$l$ runs over the atomic orbitals $\chi _{\alpha,l}$ (e.g., GTOs) centered on atom $\alpha$, $\{ M_{\alpha,l} \}$ are variational parameters,
and $u(r)$ is a simple bounded function. jQMC supports two functional forms for $u$, selected by `jastrow_1b_type`
(`--jastrow-1b-type` of `jqmc-tool trexio convert-to` / `jqmc-tool qe`):

**Exponential form** (`jastrow_1b_type='exp'`, default):

$$
u\left( r \right) = \frac{ 1 }{2 b_{\text{ei}}} \left( {1 - {e^{ - r b_{\text{ei}}}}} \right) \,,
$$ (onebody_u)

**Pade form** (`jastrow_1b_type='pade'`):

$$
u\left( r \right) = \frac{r}{2(1 + b_{\text{ei}} \cdot r)} \,,
$$ (onebody_u_pade)

Both forms depend on a single variational parameter $b_{\text{ei}}$ (`jastrow_1b_param`, set with `-j1`).
Inserting {eq}`onebody_u_pade` into {eq}`onebody_J_hom` gives the per-atom term
$-Z^{\rm eff}_\alpha r/(1+b_{\rm ei}(2Z^{\rm eff}_\alpha)^{1/4} r)$.
For both forms the slope of the per-atom term at $r \to 0$ is $-Z^{\rm eff}_\alpha$, which is Kato's
electron-ion cusp condition for an all-electron atom.

The two-body Jastrow factor is defined as:

$$
{J_2}\left( {{{\mathbf{r}}_1}, \ldots, {{\mathbf{r}}_N}} \right) =  {\sum\limits_{i < j} {v\left( {\left| {{{\mathbf{r}}_i} - {{\mathbf{r}}_j}} \right|} \right)} },
$$ (twobody_jastrow)

where $v$ is another simple bounded function. jQMC supports two functional forms for $v$, selected by
`jastrow_2b_type` (`--jastrow-2b-type`):

**Pade form** (`jastrow_2b_type='pade'`, default):

$$
  v\left( {{r_{i,j}}} \right) =
    \cfrac{{{r_{i,j}}}}{2} \cdot {\left( {1 + b_{\rm{ee}} \cdot {{r_{i,j}}}} \right)^{ - 1}}
$$ (twobody_v)

**Exponential form** (`jastrow_2b_type='exp'`):

$$
  v\left( {{r_{i,j}}} \right) =
    \cfrac{1}{2 b_{\rm{ee}}} \left( 1 - e^{-b_{\rm{ee}} \cdot r_{i,j}} \right)
$$ (twobody_v_exp)

where  ${r_{i,j}} = \left| {{{\mathbf{r}}_i} - {{\mathbf{r}}_j}} \right|$,
and $b_{\rm{ee}}$ (`jastrow_2b_param`, set with `-j2`) is the variational parameter.
The same function $v$ is used for parallel-spin and antiparallel-spin pairs, so both have
the slope $v'(0)=1/2$: the antiparallel-spin cusp condition is satisfied, while the
parallel-spin cusp value $1/4$ is not imposed by $J_2$.

The three/four-body Jastrow factor reads:

$$
J_{3/4}\left( {{{\mathbf{r}}_1}, \ldots, {{\mathbf{r}}_N}} \right) =
\sum_{i < j}
\left(
\sum_{l} \sum_{l'}
M_{l,l'}
\chi _{l}( \mathbf{r}_i )
\chi _{l'}( \mathbf{r}_j )
\right),
$$ (threebody_jastrow)

where the indices $l$ and $l'$ indicate different orbitals and $\{ M_{l,l'} \}$ are variational parameters.

jQMC evaluates the inhomogeneous one-body ($J_1^{\rm inh}$) and three/four-body ($J_{3/4}$) Jastrow factors as matrix-vector and matrix-matrix products. The one-body term ${J_1^{\rm inh}}\left(\ldots, {{\mathbf{r}}_i}, \ldots \right)$ is computed as:

$$
\begin{aligned}
{J_1^{\rm inh}}\left(\ldots, {{\mathbf{r}}_i}, \ldots \right) &= \sum_{u=1}^{N_e^{\uparrow}} \sum_{l=1}^{L} \left( { {M_{1,l} \chi_{l} ( \vec{r}_u^{\uparrow})} } \right) + \sum_{d=1}^{N_e^{\downarrow}} \sum_{l=1}^{L} \left( { {M_{1,l} \chi_{l} ( \vec{r}_d^{\downarrow})} } \right) \\
&=
M_1 X^{\uparrow} \vec{e}^{N_e^{\uparrow}} + M_1 X^{\downarrow} \vec{e}^{N_e^{\downarrow}},
\end{aligned}
$$

where $\vec{e}^{N_e^{\uparrow}}$ and $\vec{e}^{N_e^{\downarrow}}$ are $(N_e^{\uparrow}, 1)$ and $(N_e^{\downarrow}, 1)$ unit column vectors, $X^{\uparrow}_{l,u} \equiv \chi _{l}( \vec{r}_{u}^{\uparrow} )$ is $(L, N_e^{\uparrow})$ matrix, $X^{\downarrow}_{l',d} \equiv \chi _{l'}( \vec{r}_{d}^{\downarrow} )$ is $(L, N_e^{\downarrow})$ matrix, and $M_1$ is the length-$L$ row vector of the coefficients $M_{\alpha,l}$ of eq. {eq}`onebody_J_inhom` (the index $l$ runs over all $L$ Jastrow AOs of all atoms).

The three-body term $J_{3/4}\left( {{{\mathbf{r}}_1}, \ldots, {{\mathbf{r}}_N}} \right)$ is computed as:

$$
\begin{aligned}
J_{3/4}\left( {{{\mathbf{r}}_1}, \ldots, {{\mathbf{r}}_N}} \right)
&=
\sum_{u' > u}^{N_e^{\uparrow}, N_e^{\uparrow}}
\left(
\sum_{l,l'}^{L, L}
\chi _{l}( \vec{r}_{u}^{\uparrow} )
M_{l,l'}
\chi _{l'}( \vec{r}_{u'}^{\uparrow} )
\right)
+
\sum_{d' > d}^{N_e^{\downarrow}, N_e^{\downarrow}}
\left(
\sum_{l,l'}^{L, L}
\chi _{l}( \vec{r}_{d}^{\downarrow} )
M_{l,l'}
\chi _{l'}( \vec{r}_{d'}^{\downarrow} )
\right)
+
\sum_{u,d}^{N_e^{\uparrow}, N_e^{\downarrow}}
\left(
\sum_{l,l'}^{L, L}
\chi _{l}( \vec{r}_{u}^{\uparrow} )
M_{l,l'}
\chi _{l'}( \vec{r}_{d}^{\downarrow} )
\right) \\
&=
\operatorname{tr}
\left(
(X^{\uparrow})^T
M
X^{\uparrow}
K^{\uparrow}
\right)
+
\operatorname{tr}
\left(
(X^{\downarrow})^T
M
X^{\downarrow}
K^{\downarrow}
\right)
+
(\vec{e}^{N_e^{\uparrow}})^T
(X^{\uparrow})^T
M
X^{\downarrow}
\vec{e}^{N_e^{\downarrow}},
\end{aligned}
$$

where $K^{\uparrow}$ and $K^{\downarrow}$ are $(N_e^{\uparrow}, N_e^{\uparrow})$ and $(N_e^{\downarrow}, N_e^{\downarrow})$ matrices, reading:

$$
\kappa_{i,j} =
\begin{cases}
1  & (i > j) \\
0 &(i \le j)
\end{cases}.
$$

The same symmetric matrix $M$ enters the parallel-spin and antiparallel-spin
three-body terms. The one-body coefficient vector $M_1$
($M_{\alpha,l}$ in eq. {eq}`onebody_J_inhom`) is stored as the last column
alongside $M$ in a single $(L, L+1)$ Jastrow matrix.

(sec:periodic_jastrow)=
## Periodic Jastrow factors

For a periodic cell the Jastrow factor is real and invariant under lattice translations. jQMC keeps the molecular forms, using periodic interparticle distances in $J_1^{\rm h}$/$J_2$ and lattice-summed periodic atomic orbitals in $J_1^{\rm inh}$/$J_{3/4}$. $J_1^{\rm h}$ and $J_2$ use a real lattice-periodic distance with the physical coalescence cusp; oblique cells are supported. Kernels and options follow {ref}`sec:Jastrow`: `jastrow_1b_type` (`exp` default, or `pade`), `jastrow_2b_type` (`pade` default, or `exp`), with `jastrow_1b_param`/`-j1` and `jastrow_2b_param`/`-j2`. $J_1^{\rm inh}$ and $J_{3/4}$ use lattice-summed periodic atomic orbitals ({ref}`sec:atomic_orbitals`, Periodic AO representations). The Jastrow basis is phase-free, real, and shared across twists. For complex MO/PW/spline references, construct once and share. Detailed periodic formulations are work in progress.

(sec:Jastrow_DTN)=
## Drummond-Towler-Needs (DTN) Jastrow

jQMC implements the DTN cutoff-polynomial family [Drummond, Towler, Needs, Phys. Rev. B 70, 235119 (2004)](https://doi.org/10.1103/PhysRevB.70.235119), with exponent $J_{\rm DTN}=J_u+J_\chi+J_f$, comprising one-body $\chi$, two-body $u$, and three-body $f$ terms. Exactly one Jastrow family is built per wave function; a DTN preset cannot be combined with `-j1`, `-j2`, or `-j3`. Select it when generating the Hamiltonian with `jqmc-tool trexio convert-to` or `jqmc-tool qe`:

| Option | Default | Meaning |
| --- | --- | --- |
| `--jastrow-dtn` | `none` | preset: `none`, `u`, `u_chi`, `u_f`, `u_chi_f` |
| `--dtn-C` | `4` | cutoff smoothness order ($C\ge3$ for continuous Laplacian) |
| `--dtn-Nu`, `--dtn-Nchi`, `--dtn-NeN`, `--dtn-Nee` | `8`, `8`, `4`, `4` | polynomial orders for $u$, $\chi$, and $f$ |
| `--dtn-cutoff-ratio-u`, `--dtn-cutoff-ratio-chi`, `--dtn-cutoff-ratio-f` | `0.95` | periodic cutoff ratios |
| `--dtn-Lu`, `--dtn-Lchi`, `--dtn-Lf` | geometry-derived | molecular absolute cutoffs in Bohr (`trexio convert-to` only) |
| `--dtn-cutoff-abs-cap` | `8.0` | cap on geometry-derived molecular cutoffs (`trexio convert-to` only) |

Presets without $u$ are low-level-only and leave the e-e cusp unsatisfied. Periodic cutoffs are dimensionless Wigner--Seitz inscribed-radius ratios, default `0.95` ($u$, $\chi$ use the bound; $f$ half); kernels must stay in the unique-minimum-image region, `sanity_check` rejects violations, and absolute periodic `L` inputs are rejected. Molecular cutoffs are absolute Bohr; if omitted, geometry-derived and capped at 8 Bohr; for a single atom, $L_\chi$ and $L_f$ fall back to $L_u$.

(sec:dtn_comoving_cutoff)=
### Strain and convergence

Periodic DTN cutoffs are part of the trial-wave-function definition. Under strain jQMC transports resolved cutoffs consistently and uses the same convention in MCMC and GFMC/LRDMC stress estimators; zero-strain cutoffs are those stored, and EOS comparisons must use the same volume-dependent continuation. The default ratio does not guarantee accuracy; check convergence or use larger cells; if an optimum reaches the allowed maximum, a boundary/policy response remains. Coefficient counts are fixed; ratios are not optimizer coordinates. VMC blocks use `[vmc]` flags `opt_JDTN_u_param`, `opt_JDTN_chi_param`, and `opt_JDTN_f_param` (default `true`; effective only when the term is present).

(sec:periodic_determinants)=
## Periodic determinants

The determinant part (SD, AGP, AGPn) inherits periodicity from the periodic AOs. A real Gamma point uses real periodic AOs with `k_point = None` and real $\Psi$ obeying strict periodic boundary conditions. A complex Bloch representation carries a fractional `k_point`; nonzero gives Bloch boundary conditions and complex $\Psi$. A Gamma twist inside a multi-twist input stays complex with `k_point = (0,0,0)` ({ref}`sec:atomic_orbitals`, Periodic AO representations). Up and down orbitals always use opposite twists $+\mathbf k$/$-\mathbf k$: the down-spin block is the complex conjugate of its own spin channel's $+\mathbf k$ orbitals, stored as conjugated coefficients on the $-\mathbf k$ AO basis. This requires a real Hamiltonian for each spin channel (no vector potential and no spin-orbit coupling; a collinear Zeeman-like field is allowed), so that conjugation maps a $+\mathbf k$ orbital to a $-\mathbf k$ orbital of the same channel; it does not require time-reversal symmetry of the spin state. `time_reversal` only selects coefficient tying: with `time_reversal=True` the down coefficients are the conjugate of the up coefficients and only $+\mathbf k$ is variational; with `time_reversal=False` (LSDA / UHF) the down coefficients are the conjugate of the independent beta coefficients and both blocks are variational. Expectation values are real parts of sampled complex local quantities ({doc}`hamiltonian`, {doc}`error_analysis`); spin-block storage is in {doc}`sd_occupied_orbitals`. The Jastrow is real and $\mathbf k$-independent.

(sec:periodic_fixed_electrons)=
### Fixed electron numbers and metals

Determinants have integer electron numbers, and jQMC uses the same $(N_\uparrow,N_\downarrow)$ at every twist; grand-canonical twist averaging is not implemented. `jqmc-tool qe` accepts only binary band occupations; smeared states and non-integer `tot_magnetization` are rejected. Use `occupations='fixed'` (for `nspin=2`, integer `tot_magnetization`). Metals are not a production target: fixed occupations pass conversion, but every twist fills the same number of lowest bands regardless of the Fermi surface, so the twist average does not sample it; tutorials cover insulators and a spin-polarized atom in a box. For periodic TREXIO, every k-point block must have finite integer occupations summing to the file's electron numbers; in a restricted block (2 = up and down, 1 = up only), doubly and singly occupied orbitals must also give $(N_\uparrow,N_\downarrow)$. Smeared and metallic states with occupied bands differing between k-points are rejected.

(sec:AGP)=

## Antisymmetrized Geminal Power (AGP)
jQMC implements the Antisymmetrized Geminal Power (AGP) ansatz for the antisymmetric part. M. Casula and S. Sorella first applied it to an *ab initio* calculation in 2003 [J. Chem. Phys. 119, 6500 (2003)](https://doi.org/10.1063/1.1604379), and it has since been implemented in other QMC codes. For simplicity, first consider a system with an even number $N$ of electrons.
The WF, written in terms of pairing functions, is:

$$
\Phi_\text{AS} (\mathbf{1},\ldots,\mathbf{N}) = {\cal A} \left\{ g(\mathbf{1},\mathbf{2}) g(\mathbf{3},\mathbf{4}) \ldots g(\mathbf{N-1},\mathbf{N}) \right\}
$$ (eq:wfPF0)

where ${\cal A}$ is the antisymmetrization operator:

$$
{\cal A} \equiv {1\over N!} \sum_{P\in S_N} \epsilon_P \hat P,
$$

$S_N$ the permutation group of $N$ elements, $\hat P$ the operator corresponding to  the generic permutation $P$, and $\epsilon_P$  its sign. We denote with  the generic bolded index $\mathbf{i}$ both the space ${\bf r}_i$ coordinates and the spin values $\sigma_i$:

$$
\mathbf{i} \leftrightarrow ({\bf r}_i, \sigma_i)
$$

corresponding to the $i^{th}$ electron.

Let us define $G$ the $N\times N$ matrix with elements $G_{i,j} = g(\mathbf{i},\mathbf{j})$.
Notice that

$$
g(\mathbf{i},\mathbf{j}) = -g(\mathbf{j},\mathbf{i}) \; (\text{and} \; G_{i,j} = -G_{j,i}),
$$

as a consequence of the statistics of fermionic particles, thus $G$ is skew-symmetric (i.e., $G^T = -G$, being $^T$ the transpose operator). This is an **algebraic** skew-symmetry expressing fermion exchange antisymmetry; it holds whether $G$ is real or complex, and the transpose, not the Hermitian conjugate, is the correct operation here.

We parameterize the pairing function $g$ with generalized spin-dependent orbitals $\phi(r, \sigma) \equiv \psi(r)\chi(\sigma)$, where $\psi$ is its spatial part and $\chi$ is the spin function. For 2$L$ orbitals,

$$
(\psi_1^{\alpha}(r)\alpha_1(\sigma) \cdots \psi_L^{\alpha}(r)\alpha_L(\sigma) \cdots \psi_1^{\beta}(r)\beta_1(\sigma) \cdots \psi_L^{\beta}(r)\beta_L(\sigma) ),
$$

the pairing function is parameterized as:

$$
g(i, j) = \sum_{\mu,\nu} A_{\mu,\nu} \phi_{\mu}(r_i, \sigma_i) \phi_{\nu}(r_j, \sigma_j) \equiv \sum_{\mu,\nu} A_{\mu,\nu} \psi_{\mu}(r_i)\psi_{\nu}(r_j)\chi_{\mu}(\sigma_i)\chi_{\nu}(\sigma_j)
$$

where $A_{\mu,\nu}$ is a $(2L,2L)$ skew-symmetric matrix as a consequence of $g(i,j) = - g(j,i)$.

jQMC currently considers only the pairing functions with opposite spins, thus, the antisymmetrization operator corresponds to the determinant of the up-down or down-up pairing matrix $G$, i.e.,

$$
\Phi(1 \cdots N) = \det(G_{u,d}) = (-1)^{N/2} \det(G_{d,u}),
$$

where the last equality follows from $G_{d,u} = -G_{u,d}^T$ ($G_{u,d}$ is an $N/2 \times N/2$ matrix).
We denote block matrices $\Lambda^{\uparrow, \downarrow}_{l,l'} \equiv A_{u,d}$ and $\Lambda^{\downarrow, \uparrow}_{l,l'} \equiv A_{d,u}$, where $l$ and $l'$ run over from 1 to $L$. Since $A$ is skew-symmetric, $\Lambda^{\uparrow, \downarrow}_{l,l'} = - \Lambda^{\downarrow, \uparrow}_{l',l}$. Indeed, one can consider up and down pairs without loss of generality. Say, we can omit the spin-dependent part in practice, and for a spatial generic pairing function $F_{i,j} \equiv f(r_i^{\uparrow}, r_j^{\downarrow})$ we have

$$
f\left( {{{\mathbf{r}}_i^{\uparrow}},{{\mathbf{r}}_j}^{\downarrow}} \right) = \sum\limits_{l,l'} {\Lambda_{l,l'}^{\uparrow, \downarrow} {\psi_{l}^{\uparrow}}\left( {{{\mathbf{r}}_i^{\uparrow}}} \right){\psi _{l'}^{\downarrow}}\left( {{{\mathbf{r}}_j^{\downarrow}}} \right)},
$$ (agp_expansion)

and

$$
\Phi(1 \cdots N) = \det(G_{u,d}) = \det(F).
$$

Hereafter, we think about cases with spin-independent orbitals $\psi^{\alpha} = \psi^{\beta}$. In such a case, the generalized pairing function for opposite spins is parametrized as

$$
\begin{aligned}
g(i, j) &=  \sum_{l,l'} \Lambda_{l,l'}^{\uparrow, \downarrow} \psi_{l}(r_i)\psi_{l'}(r_j)\alpha(\sigma_i)\beta(\sigma_j) + \sum_{l,l'} \Lambda_{l,l'}^{\downarrow, \uparrow} \psi_{l}(r_i)\psi_{l'}(r_j)\beta(\sigma_i)\alpha(\sigma_j) \\
&=  \sum_{l,l'} \Lambda_{l,l'}^{\uparrow, \downarrow} \psi_{l}(r_i)\psi_{l'}(r_j)\alpha(\sigma_i)\beta(\sigma_j) - \sum_{l,l'} \Lambda_{l',l}^{\uparrow, \downarrow} \psi_{l}(r_i)\psi_{l'}(r_j)\beta(\sigma_i)\alpha(\sigma_j) \\
\end{aligned}
$$

If $\Lambda_{l,l'}$ is symmetric, $ \Lambda_{l',l}^{\uparrow, \downarrow} =  \Lambda_{l,l'}^{\uparrow, \downarrow} \equiv \Lambda_{l,l'}^S $, leading to:

$$
\begin{aligned}
g(i, j) &=  \sum_{l,l'} \Lambda_{l,l'}^{\uparrow, \downarrow} \psi_{l}(r_i)\psi_{l'}(r_j)(\alpha(\sigma_i)\beta(\sigma_j) - \beta(\sigma_i)\alpha(\sigma_j)) \\
&=  \sum_{l,l'} \Lambda_{l,l'}^{S} \psi_{l}(r_i)\psi_{l'}(r_j)(\alpha(\sigma_i)\beta(\sigma_j) - \beta(\sigma_i)\alpha(\sigma_j)) \\
&\equiv f_{S}({\bf r}_i,{\bf r}_j) ({ {\left| \uparrow  \downarrow \right\rangle - \left| \downarrow  \uparrow \right\rangle}}),
\end{aligned}
$$

which is dubbed as AGP with spin-singlet pairs (AGPs).

If $\Lambda_{l,l'}$ is skew-symmetric, $ \Lambda_{l',l}^{\uparrow, \downarrow} =  - \Lambda_{l,l'}^{\uparrow, \downarrow} \equiv \Lambda_{l,l'}^A $, leading to:

$$
\begin{aligned}
g(i, j) &=  \sum_{l,l'} \Lambda_{l,l'}^{\uparrow, \downarrow} \psi_{l}(r_i)\psi_{l'}(r_j)(\alpha(\sigma_i)\beta(\sigma_j) + \beta(\sigma_i)\alpha(\sigma_j)) \\
&=  \sum_{l,l'} \Lambda_{l,l'}^{A} \psi_{l}(r_i)\psi_{l'}(r_j)(\alpha(\sigma_i)\beta(\sigma_j) + \beta(\sigma_i)\alpha(\sigma_j)) \\
&\equiv f_{T}({\bf r}_i,{\bf r}_j) ({ {\left| \uparrow  \downarrow \right\rangle + \left| \downarrow  \uparrow \right\rangle}}),
\end{aligned}
$$

which is dubbed as AGP with spin-triplet pairs (AGPt).

Even if $\Lambda^{l,l'}$ is neither symmetric nor skew-symmetric, the components corresponding to AGPs and AGPt can be uniquely recovered because any square matrix can be uniquely decomposed into a sum of a symmetric and a skew-symmetric matrix [G. Strang, *Linear Algebra and Its Applications*, 4th ed., Cengage (2006)](https://www.cengage.com/c/isbn/9780030105678/). In such a case, the resultant pairing function is the combination of AGPs and AGPt.

In this sense, if the basis is spin-dependent, the spatial and spin parts cannot be separated and the spin-singlet pairings are no longer eigenstates, introducing a spin-contaminated pairing function, dubbed as AGPc.

In the above formalism, AGPs, AGPt, AGPs+AGPt, AGPc are separately defined. However, in the actual implementation, we define only one matrix $\Lambda^{\uparrow,\downarrow}_{l,l'}$ and compute the spatial part of the pairing function as:

$$
f(\mathbf{r}_i, \mathbf{r}_j) = \sum_{l,l'} \Lambda_{l,l'}^{\uparrow, \downarrow} \, \phi_l^{\uparrow}(\mathbf{r}_i)\, \phi^{\downarrow}_{l'}(\mathbf{r}_j).
$$

```{note}
**Symmetry of $\Lambda$ with opposite spin twists: $\boldsymbol{\theta}^{\uparrow}=-\boldsymbol{\theta}^{\downarrow}$.**

For time-reversal-paired orbitals, $\psi_l^{\downarrow}=(\psi_l^{\uparrow})^{*}$,
with up-spin twist $+\boldsymbol{\theta}$ and down-spin twist
$-\boldsymbol{\theta}$ (sec. {ref}`sec:periodic_determinants`), jQMC enforces:

- paired block $\Lambda^{\uparrow,\downarrow}$ **Hermitian** ($\Lambda = \Lambda^{\dagger}$);
- unpaired block $\Lambda^{\uparrow}$ **general complex** (no constraint).

A complex-symmetric $\Lambda$ would preserve an exact spin singlet but gives a
less well-conditioned optimization. The Hermitian choice keeps the SR/LM
overlap matrix Hermitian positive semidefinite at the price of a small
singlet contamination, quantifiable by
$\|(\Lambda - \Lambda^{T})/2\|_F / \|\Lambda\|_F$. For a real wave function
the two constraints coincide (real symmetric = real Hermitian) and AGPs is
recovered exactly.
```


The AGP ansatz can be generalized to describe polarized systems, i.e., systems where the number $N_u$ of electrons of spin up is different from the number $N_d$ of electrons with spin down.
With no loss of generality, we can assume that $N_u>N_d$, thus the system is constituted by a number $p=N_d$ of electron pairs and a number $k=N_u-N_d$ of unpaired electrons (clearly, $N=N_u+N_d=2p+k$). For spin-polarized systems with unpaired orbitals,  we can add $k$ fictitious entries to $g(\mathbf{i},\mathbf{j})$, such that $g(\mathbf{i},\mathbf{N+l})=-g(\mathbf{N+l},\mathbf{i}) = \phi_l(\mathbf{i})$ for $l=1,\ldots,k$ and $i=1,\ldots,N_u$.

Thus, we need to evaluate:

$$
\Phi_\text{AGP} = \det(\tilde G) \,.
$$ (eq:wfAGPunpaired)

with the $N_u\times N_u$ matrix $\tilde G = \left[\begin{array}{c|c} G_{ud} & \Phi \end{array}\right]$, where the $N_u\times N_d$ matrix $G_{ud}$ describes the pairing between the $N_u$ spin up electrons and the $N_d$ spin down electrons, and the $N_u\times k$ matrix  $\Phi$ describes the $k$ unpaired orbitals.

Here, we describe a practical implementation of the pairing function calculation. As for the Jastrow factor, jQMC evaluates it as matrix-vector and matrix-matrix products. Suppose one evaluates the value of a many-body WF at $\vec{x}^{\uparrow} = (\vec{r}_1^{\uparrow}, \cdots, \vec{r}_i^{\uparrow}, \cdots, \vec{r}^{\uparrow}_{N_{e}^{\uparrow}})$ and $\vec{x}^{\downarrow} = (\vec{r}_1^{\downarrow}, \cdots, \vec{r}_j^{\downarrow}, \cdots, \vec{r}^{\downarrow}_{N_{e}^{\downarrow}})$.

The basis set expanding the AGP pairing function is arbitrary; jQMC supports the expansion with both atomic and molecular orbitals. Both AGP with Molecular Orbitals and AGP with Atomic Orbitals can be treated in the same way in practice.

For the paired parts with AOs, we can prepare for a lambda matrix $\Lambda^{\uparrow, \downarrow}$, i.e., $\lambda_{l,m}^{\uparrow, \downarrow}$, whose dimension is ($M_{ao}, M_{ao}$) and, atomic or molecular orbitals matrix, $\Psi_{l,i}^{\uparrow} = \psi^{\uparrow}_l(\vec{r}_i^{\uparrow})$, $\Psi_{m,j}^{\downarrow} = \psi^{\downarrow}_m(\vec{r}_j^{\downarrow})$, whose dimensions are $(M_{ao}, N_{e}^{\uparrow})$ and $(M_{ao}, N_{e}^{\downarrow})$. Here, we assume that the number of AOs is the same for up and down electrons such that the geminal matrix $\Lambda^{\uparrow, \downarrow}$ is square. Then, the pairing geminal matrix is $F(\vec{x}^{\uparrow}, \vec{x}^{\downarrow}) = (\Psi^{\uparrow})^T \Lambda^{\uparrow, \downarrow} \Psi^{\downarrow}$, where $F_{i,j} = \sum_{l,m} \Psi_{l,i}^{\uparrow} \lambda_{l,m}^{\uparrow, \downarrow} \Psi_{m,j}^{\downarrow}$. The dimension of $F$ matrix is $(N_{e}^{\uparrow}, N_{e}^{\downarrow})$. For the unpaired part, a lambda matrix $\Lambda^{\uparrow}$, whose dimension is ($M_{ao}, N_{e}^{\uparrow} - N_{e}^{\downarrow}$), and $\Psi_{l,i}^{\uparrow} = \psi^{\uparrow}_l(\vec{r}_i^{\uparrow})$, whose dimensions are $(M_{ao}, N_{e}^{\uparrow})$. The unpaired part can be computed as $F(\vec{x}^{\uparrow}) = (\Psi^{\uparrow})^T \Lambda^{\uparrow}$, where $F_{i,k} = \sum_{l} \Psi_{l,i}^{\uparrow} \lambda_{l,k}^{\uparrow}$, whose dimension is $(N_{e}^{\uparrow}, N_{e}^{\uparrow} - N_{e}^{\downarrow})$.

Then, the determinant part is computed as $\det (\tilde{F}(\vec{x}^{\uparrow}, \vec{x}^{\downarrow}))$, where $\tilde{F}(\vec{x}^{\uparrow}, \vec{x}^{\downarrow})$ is the concatenated matrix, whose dimension is $(N_{e}^{\uparrow}, N_{e}^{\uparrow})$, i.e., $\tilde{F}(\vec{x}^{\uparrow}, \vec{x}^{\downarrow}) = [(\Psi^{\uparrow})^T \Lambda^{\uparrow, \downarrow} \Psi^{\downarrow} | (\Psi^{\uparrow})^T \Lambda^{\uparrow} ]$. From the practical view point, we can set the input variational parameters as a concatenated matrix $\tilde{\Lambda} \equiv [\Lambda^{\uparrow, \downarrow} | \Lambda^{\uparrow}]$, whose dimension is $(M_{ao}, M_{ao} + N_{e}^{\uparrow} - N_{e}^{\downarrow})$.

MO cases are the same as the AO cases by setting $\Psi_{l,i}^{\uparrow} = \phi_l^{\uparrow}(\vec{r}_i^{\uparrow})$, $\Psi_{m,j}^{\downarrow} = \phi_m^{\downarrow}(\vec{r}_j^{\downarrow})$, whose dimensions are $(M_{mo}, N_{e}^{\uparrow})$ and $(M_{mo}, N_{e}^{\downarrow})$. The MOs can be different for up and down electrons in the case of UHF/UKS calculations.

Since MOs and AOs are connected via the following relation:

$$
\begin{gathered}
\Phi^{\uparrow} = C^{\uparrow} \Psi^{\uparrow} \\
\Phi^{\downarrow} = C^{\downarrow} \Psi^{\downarrow} \\
\end{gathered}
$$

Here $C^{\uparrow}$ and $C^{\downarrow}$ are the up- and down-spin MO
coefficient matrices, each with shape $(M_{mo},M_{ao})$. They are real for
a real wave function and complex for a complex Bloch wave function.

For time-reversal-paired spin channels with opposite twists,
$\boldsymbol{\theta}^{\uparrow}=-\boldsymbol{\theta}^{\downarrow}$,
only $C^{\uparrow}\equiv C^{+\mathbf{k}}$ is an independent variational
matrix. The down-spin block uses the $-\mathbf{k}$ AO basis and the
complex-conjugate coefficients

$$
C^{\downarrow}=C^{-\mathbf{k}}=(C^{+\mathbf{k}})^*=(C^{\uparrow})^*.
$$

This coefficient constraint is specific to time-reversal pairing, not to
general independent spin orbitals. Independent (LSDA / UHF) channels keep
the opposite twists with $C^{\downarrow}=(C_\beta^{+\mathbf{k}})^*$ on the
$-\mathbf{k}$ AO basis and no relation to $C^{\uparrow}$. The boundary phases are defined in
{ref}`sec:periodic_determinants`; storage of tied and independent spin
blocks is described in {doc}`sd_occupied_orbitals`.

The AO $\leftrightarrow$ MO transform reads:

$$
\begin{gathered}
\Lambda_A^{\uparrow, \downarrow} = (C^{\uparrow})^T \Lambda_M^{\uparrow, \downarrow} \,C^{\downarrow} \\
\Lambda_A^{\uparrow} = (C^{\uparrow})^T \Lambda_M^{\uparrow}.
\end{gathered}
$$

(unchanged in form -- both transposes are spatial-index swaps coming from the AO matrix orientation, not Hermitian conjugates of the basis transform). With $C^{\downarrow} = (C^{\uparrow})^{*}$ for a time-reversal-paired complex wave function the paired-block transform reduces to

$$
\Lambda_A^{\uparrow,\downarrow} = (C^{\uparrow})^{T}\,\Lambda_M^{\uparrow,\downarrow}\,(C^{\uparrow})^{*},
$$

which makes the Hermitian symmetry of $\Lambda_M^{\uparrow,\downarrow}$ (see the note above) propagate to the AO representation: if $\Lambda_M$ is Hermitian then $\Lambda_A^{\uparrow,\downarrow}$ is Hermitian as well, by the same algebra.

Indeed, the following matrices are the same, i.e., $\tilde{F}^{AO}(\vec{x}^{\uparrow}, \vec{x}^{\downarrow}) = \tilde{F}^{MO}(\vec{x}^{\uparrow}, \vec{x}^{\downarrow})$:

$$
\begin{gathered}
\tilde{F}^{AO}(\vec{x}^{\uparrow}, \vec{x}^{\downarrow}) = [(\Psi^{\uparrow})^T \Lambda_A^{\uparrow, \downarrow} \Psi^{\downarrow} | (\Psi^{\uparrow})^T \Lambda_A^{\uparrow} ] \\
\tilde{F}^{MO}(\vec{x}^{\uparrow}, \vec{x}^{\downarrow}) = [(\Phi^{\uparrow})^T \Lambda_M^{\uparrow, \downarrow} \Phi^{\downarrow} | (\Phi^{\uparrow})^T \Lambda_M^{\uparrow} ].
\end{gathered}
$$


## AGP with constrained number of molecular orbitals (AGPn)
If one expands the AGP pairing function with molecular orbitals and neglects the non-diagonal terms of the pairing matrix, the resultant pairing function reads:

$$
\tilde{F}^{MO}(\vec{x}^{\uparrow}, \vec{x}^{\downarrow}) = [(\Phi^{\uparrow})^T \Lambda_{M, {\rm diag}}^{\uparrow, \downarrow} \Phi^{\downarrow} | (\Phi^{\uparrow})^T \Lambda_M^{\uparrow} ],
$$

where $\Lambda_{M, {\rm diag}}^{\uparrow, \downarrow}$ is the diagonal matrix (i.e., the non-diagonal terms are set to zero, neglecting couplings between different molecular orbital indices). This is what we call the AGPn ansatz.

## Single Slater determinant (SD)
The SD is stored directly as the $N_e^\uparrow$ and $N_e^\downarrow$
occupied MO rows.  It does not retain the equivalent AGP selector
$\operatorname{diag}(1,\ldots,1,0,\ldots,0)$ or unoccupied MO rows.  If
$\Phi_\sigma$ denotes the occupied-orbital value table (rows: orbitals,
columns: spin-$\sigma$ electrons), and $\Phi_\uparrow[a:b]$ denotes its rows
$a+1,\ldots,b$ (0-based, end-exclusive slice), the open-shell
Slater matrix is

$$
G=\left[\Phi_\uparrow[0:N_e^\downarrow]^T\Phi_\downarrow
\;\middle|\;
\Phi_\uparrow[N_e^\downarrow:N_e^\uparrow]^T\right].
$$

The occupied rows still range over the complete AO columns, so ordinary
occupied--virtual orbital relaxation remains available.  Finite occupation
of additional geminal modes is instead an AGPn ansatz.  The options,
conversion and storage rules of this representation are summarized in
{ref}`sec:sd_occupied_orbitals`.

## Fast update of the determinant part

```{note}
**Real and complex wave functions.** The Sherman-Morrison rank-1 ratio algebra and the corresponding inverse update used throughout this section are valid for complex matrices without any algebraic modification: for a complex Bloch determinant the matrices $A$, $\vec{u}$, $\vec{v}$ and $A^{-1}$ are simply complex. The Metropolis acceptance probability is $|p|^{2}$ times the squared modulus of the Jastrow ratio in both cases (cf. {ref}`vmc_tags`, Metropolis section); the determinant ratio $p$ itself may be complex.
```

Suppose $l$-th electron is updated between the old and new electron coordinates,
the new matrix $A_{new}$ is

$$
A^{new}=
\begin{pmatrix}
A_{1,1}  & \cdots & A_{1,j}  & \cdots & A_{1,n}  \\
\vdots   & \ddots &          &        & \vdots   \\
A'_{l,1} &        & A'_{l,j} &        & A'_{l,n} \\
\vdots   &        &          & \ddots & \vdots   \\
A_{n,1}  & \cdots & A_{n,j}  & \cdots & A_{n,n}
\end{pmatrix}
,\:
A^{old}=
\begin{pmatrix}
A_{1,1} & \cdots & A_{1,j} & \cdots & A_{1,n}   \\
\vdots  & \ddots &         &        & \vdots    \\
A_{l,1} &        & A_{l,j} &        & A_{l,n}   \\
\vdots  &        &         & \ddots & \vdots    \\
A_{n,1} & \cdots & A_{n,j} & \cdots & A_{n,n}
\end{pmatrix}
$$

if $l$ is up-electron index, and

$$
A^{new}=
\begin{pmatrix}
A_{1,1} & \cdots & A'_{1,l} & \cdots & A_{1,n} \\
\vdots  & \ddots &          &        & \vdots  \\
A_{i,1} &        & A'_{i,l} &        & A_{i,n} \\
\vdots  &        &          & \ddots & \vdots  \\
A_{n,1} & \cdots & A'_{n,l} & \cdots & A_{n,n}
\end{pmatrix}
,\:
A^{old}=
\begin{pmatrix}
A_{1,1} & \cdots & A_{1,l} & \cdots & A_{1,n} \\
\vdots  & \ddots &         &        & \vdots  \\
A_{i,1} &        & A_{i,l} &        & A_{i,n} \\
\vdots  &        &         & \ddots & \vdots  \\
A_{n,1} & \cdots & A_{n,l} & \cdots & A_{n,n}
\end{pmatrix}
$$

if $l$ is down-electron index,

where $A_{i,j} = f(\vec{r}_{i}^{\uparrow}, \vec{r}_{j}^{\downarrow})$, $A'_{l,j} = f(\vec{r}_{l}^{\uparrow, new}, \vec{r}_{j}^{\downarrow})$, and $A'_{i,l} = f(\vec{r}_{i}^{\uparrow}, \vec{r}_{l}^{\downarrow, new})$. We notice that the Slater determinant is also treated like this since the SD is a special case of the AGP ansatz.

We can immediately derive:

$$
A^{new} = A^{old} + \vec{u} \cdot \vec{v}_l^T
$$

where,

if $l$ is up-electron index, $\vec{v}_l$ and $\vec{u}_l$ are

$$
\vec{v}_l=
\begin{pmatrix}
 \vdots \\ A'_{l,j} - A_{l,j} \\ \vdots
\end{pmatrix}
, \:
\vec{u}_l=
\begin{pmatrix}
 0 \\ \vdots \\ 1 \\ \vdots \\ 0
\end{pmatrix}
$$

if $l$ is down-electron index, $\vec{v}_l$ and $\vec{u}_l$ are

$$
\vec{v}_l=
\begin{pmatrix}
 0 \\ \vdots \\ 1 \\ \vdots \\ 0
\end{pmatrix}
, \:
\vec{u}_l=
\begin{pmatrix}
 \vdots \\ A'_{i,l} - A_{i,l} \\ \vdots
\end{pmatrix}
$$

The determinant ratio $p$ can be found via the matrix determinant lemma:

$$
\det(A^{new}) = \det(A^{old}+\vec{u}\vec{v}_l^T) = (1+\vec{v}_l^T (A^{old})^{-1} \vec{u})\det(A^{old}).
$$

The factor can be written as:

$$
(1+\vec{v}_l^T (A^{old})^{-1} \vec{u}_l) =
\begin{cases}
\sum_{j=1}^{n} A^{new}_{l,j} ({A^{old}})^{-1}_{j,l} & (l = \uparrow) \\
\sum_{i=1}^{n} ({A^{old}})^{-1}_{l,i} A^{new}_{i,l} & (l = \downarrow)  \\
\end{cases}
$$

Thus, the ratio $p$ can be computed via:

$$
p =
\frac{\det(A^{new})}{\det(A^{old})} = (1+\vec{v}_l^T ({A^{old}})^{-1} \vec{u}_l) =
\begin{cases}
\sum_{j=1}^{n} A^{new}_{l,j} ({A^{old}})^{-1}_{j,l} & (l = \uparrow) \\
\sum_{i=1}^{n} ({A^{old}})^{-1}_{l,i} A^{new}_{i,l} & (l = \downarrow)  \\
\end{cases}
$$

This equation shows that, to compute the ratio $p$, one does not have to compute $\det(A^{new})$ from scratch: one computes only the new row (or column) of $A^{new}$ for the new electron position and combines it with the stored inverse $(A^{old})^{-1}$. $\det(A^{new})$ itself is not needed, but $(A^{new})^{-1}$ is needed to compute the next ratio. Therefore, every time a proposed move is accepted, the inverse is updated by a rank-1 update with the *Sherman-Morrison* formula:

$$
(A^{new})^{-1} = (A^{old}+\vec{u}\vec{v}_l^T)^{-1} = (A^{old})^{-1} - (A^{old})^{-1}\vec{u}(1+\vec{v}_l^T (A^{old})^{-1} \vec{u})^{-1} \vec{v}_l^T (A^{old})^{-1}.
$$

The performance of the above standard procedure is dominated by matrix-vector
operations and memory bandwidth. For an alternative that batches inverse
updates into matrix-matrix operations, see the delayed-update method of
[McDaniel et al., J. Chem. Phys. 147, 174107 (2017)](https://doi.org/10.1063/1.4998616).

## Periodic Gaussian basis optimization

Periodic Gaussian basis optimization exposes physical exponents and contraction coefficients as optimization coordinates. Exponents must be finite and strictly positive; coefficients must be finite and may be positive or negative. Candidates outside these domains are rejected without clipping or changing the accepted wave function.

Image-truncation tolerance is part of the restart contract: HDF5 restart state records the tolerance and physical basis needed to reconstruct the same image envelope, so AO/J3 values and checkpoint continuation remain consistent after reload. Invalid-candidate rejection is collective across participating sampler and twist ranks.

The detailed formulation is work in progress.
