# Hamiltonian

(sec:kinetic_energy)=
## Kinetic

This section documents the kinetic term used in jQMC.

```{important}
**Notation in the kinetic section.** The same notation is used for real wave functions (molecules and $\Gamma$-point periodic systems) and for complex wave functions (complex Bloch determinants at a non-$\Gamma$ twist). The rule of thumb: **no bars inside the kinetic derivation**; anything written with bars is a modulus square and belongs elsewhere.

- **Algebraic square** of a 3-vector $\vec{v}$ is written as the dot product $\vec{v}\cdot\vec{v} = \sum_\gamma v_\gamma\,v_\gamma$ (no complex conjugation; bilinear, not sesquilinear), or as the shorthand $\vec{v}^{\,2} \equiv \vec{v}\cdot\vec{v}$ when no confusion arises. **No vertical bars.**
- **Modulus square** of a complex scalar $z$ is written $|z|^{2} = z\,\bar z$ (Hermitian norm, with vertical bars). It does **not** appear inside this kinetic section -- it shows up only in MCMC sampling weights $|\Psi|^{2}$ and Metropolis acceptance $|\Psi'/\Psi|^{2}$ (cf. {ref}`vmc_tags`).
- **Hadamard product squared** is written $A \odot A$, with entries $(A\odot A)_{ij} = A_{ij}\,A_{ij}$ (entry-wise algebraic square, no conjugation).
- **Wave-function log** is written $\ln\Psi$ (no bars), denoting the principal complex log of $\Psi$. For a real wave function, $\ln\Psi = \ln|\Psi| + i\pi\,[\Psi<0]$ on a single nodal pocket; the imaginary $i\pi$ is locally constant, so every $\mathbf{r}$- and $\boldsymbol{\alpha}$-derivative of $\ln\Psi$ coincides with the corresponding derivative of $\ln|\Psi|$. For a complex wave function, $\ln\Psi$ is the full holomorphic complex log including the smooth imaginary phase $i\,\arg\Psi$, and its derivatives are non-trivial (see {doc}`complex_gradients`).
```

### Laplacian

We assume that the trial wave function is composed of two elements, i.e., the determinant and Jastrow parts:

$$
\Psi_T(\vec{x}) = D(\vec{x}) e^{J(\vec{x})}.
$$

The local kinetic energy is

$$
K(\vec{x}) = -\frac{1}{2}\frac{\sum_l \left(\nabla_l^2 \Psi_T(\vec{x})\right)}{\Psi_T(\vec{x})}.
$$

In practice, instead of evaluating the Laplacian of $\Psi_T$ directly, one evaluates the same quantity as

$$
K(\vec{x}) = \frac{1}{2}\left[T(\vec{x}) - \vec{W}(\vec{x})\cdot\vec{W}(\vec{x})\right],
$$

obtained by the chain rule $\nabla^2\Psi/\Psi = \nabla^2\ln\Psi + (\nabla\ln\Psi)\cdot(\nabla\ln\Psi)$, where

$$
T(\vec{x}) = -\sum_l \nabla_l^2 \ln \Psi_T(\vec{x})
$$

and

$$
\vec{W}(\vec{x})\cdot\vec{W}(\vec{x}) = \sum_l \bigl(\vec{\nabla}_l \ln \Psi_T(\vec{x})\bigr)\cdot\bigl(\vec{\nabla}_l \ln \Psi_T(\vec{x})\bigr).
$$

The dot product on the right is the **algebraic** square (no complex conjugation): for each electron $l$,
$\bigl(\vec{\nabla}_l \ln\Psi_T\bigr)\cdot\bigl(\vec{\nabla}_l \ln\Psi_T\bigr) = \sum_\gamma \bigl(\partial_{l,\gamma}\ln\Psi_T\bigr)^{2}$.

We consider the specific form of the many-body trial wave function composed of exponential Jastrow and determinant terms:

$$
\Psi_T(\vec{x}) = e^{J(\vec{x})} D(\vec{x}),
$$

we have

$$
\vec{\nabla}_l \ln \Psi_T(\vec{x}) = \vec{\nabla}_l J(\vec{x}) + \vec{\nabla}_l \ln D(\vec{x})
$$

and

$$
\nabla_l^2 \ln \Psi_T(\vec{x}) = \nabla_l^2 J(\vec{x}) + \nabla_l^2 \ln D(\vec{x}).
$$

First, consider the determinant part. According to Jacobi's formula,

$$
\frac{\partial}{\partial p}\det A = (\det A)\,\operatorname{tr}\left(A^{-1}\frac{\partial A}{\partial p}\right)
= (\det A)\sum_{i,j}\left(A^{-1}_{i,j}\frac{\partial A_{j,i}}{\partial p}\right).
$$

Therefore, for

$$
D(\vec{x}) = \det F(\vec{x}) = \det\left(f_s(r_i^{\uparrow}, r_j^{\downarrow})\right),
$$

the gradient reads

$$
\vec{\nabla}_l \ln D(\vec{x}) = \frac{\vec{\nabla}_l \det F(\vec{x})}{\det F(\vec{x})} =
\begin{cases}
\sum_i\left(F^{-1}_{i,u}(\vec{x})\,\vec{\nabla}_u f(\vec{r}_u^{\uparrow}, \vec{r}_i^{\downarrow})\right) & (l=u),\\
\sum_j\left(F^{-1}_{d,j}(\vec{x})\,\vec{\nabla}_d f(\vec{r}_j^{\uparrow}, \vec{r}_d^{\downarrow})\right) & (l=d).
\end{cases}
$$

If one defines the following matrices,

$$
F^{(\nabla_{\gamma},\uparrow)}_{i,j} = \vec{\nabla}_{i,\gamma} f(\vec{r}_i^{\uparrow}, \vec{r}_j^{\downarrow}),
$$

and

$$
F^{(\nabla_{\gamma},\downarrow)}_{i,j} = \vec{\nabla}_{j,\gamma} f(\vec{r}_i^{\uparrow}, \vec{r}_j^{\downarrow}),
$$

where $\gamma=x,y,z$ and $F^{(\nabla_{\gamma},\downarrow)}_{i,j}=0$ for $j>N_e^{\downarrow}$, the gradients can be written (with $\sigma=(\uparrow,\downarrow)$) as

$$
\vec{\nabla}_{l,\gamma} \ln D(\vec{x}) = \frac{\vec{\nabla}_{l,\gamma}\det F(\vec{x})}{\det F(\vec{x})} =
\begin{cases}
\left(F^{(\nabla_{\gamma},\uparrow)}(\vec{x})F^{-1}(\vec{x})\right)_{l,l} & (l=u),\\
\left(F^{-1}(\vec{x})F^{(\nabla_{\gamma},\downarrow)}(\vec{x})\right)_{l,l} & (l=d).
\end{cases}
$$

Notice that $\vec{W}(\vec{x})\cdot\vec{W}(\vec{x})$ should be computed including the Jastrow part because

$$
\vec{W}(\vec{x})\cdot\vec{W}(\vec{x})
\ne
\sum_l \Bigl[
\bigl(\vec{\nabla}_l J(\vec{x})\bigr)\cdot\bigl(\vec{\nabla}_l J(\vec{x})\bigr)
+ \bigl(\vec{\nabla}_l \ln D(\vec{x})\bigr)\cdot\bigl(\vec{\nabla}_l \ln D(\vec{x})\bigr)
\Bigr],
$$

but

$$
\vec{W}(\vec{x})\cdot\vec{W}(\vec{x})
= \sum_l
\bigl(\vec{\nabla}_l J(\vec{x}) + \vec{\nabla}_l \ln D(\vec{x})\bigr)
\cdot
\bigl(\vec{\nabla}_l J(\vec{x}) + \vec{\nabla}_l \ln D(\vec{x})\bigr).
$$

For this reason, one can prepare intermediate vectors

$$
\vec{D}^{\Delta_{\gamma}^{\ln},\uparrow} \equiv \operatorname{diag}\left(F^{(\nabla_{\gamma},\uparrow)}(\vec{x})F^{-1}(\vec{x})\right),
$$

$$
\vec{D}^{\Delta_{\gamma}^{\ln},\downarrow} \equiv \operatorname{diag}\left(F^{-1}(\vec{x})F^{(\nabla_{\gamma},\downarrow)}(\vec{x})\right),
$$

for $\gamma=x,y,z$, whose lengths are $N_e^{\uparrow}$ and $N_e^{\downarrow}$, respectively. Then one can define intermediate matrices

$$
W^{D^{\uparrow}} \equiv \left(\vec{D}^{\Delta_x^{\ln},\uparrow}, \vec{D}^{\Delta_y^{\ln},\uparrow}, \vec{D}^{\Delta_z^{\ln},\uparrow}\right)^T,
$$

$$
W^{D^{\downarrow}} \equiv \left(\vec{D}^{\Delta_x^{\ln},\downarrow}, \vec{D}^{\Delta_y^{\ln},\downarrow}, \vec{D}^{\Delta_z^{\ln},\downarrow}\right)^T,
$$

where

$$
W^{D^{\uparrow}}_{l,\gamma} = \left(F^{(\nabla_{\gamma},\uparrow)}(\vec{x})F^{-1}(\vec{x})\right)_{l,l},
$$

$$
W^{D^{\downarrow}}_{l,\gamma} = \left(F^{-1}(\vec{x})F^{(\nabla_{\gamma},\downarrow)}(\vec{x})\right)_{l,l},
$$

and their dimensions are $(N_e^{\uparrow},3)$ and $(N_e^{\downarrow},3)$.

When a many-body WF is composed of only the determinant part, $\vec{W}(\vec{x})\cdot\vec{W}(\vec{x})$ can be computed as

$$
\vec{W}(\vec{x})\cdot\vec{W}(\vec{x})
= \sum_l \bigl(\vec{\nabla}_l \ln \Psi_T(\vec{x})\bigr)\cdot\bigl(\vec{\nabla}_l \ln \Psi_T(\vec{x})\bigr)
= \sum_{i,j}\left(W^{D^{\uparrow}} \odot W^{D^{\uparrow}}\right)_{i,j}
+ \sum_{i',j'}\left(W^{D^{\downarrow}} \odot W^{D^{\downarrow}}\right)_{i',j'},
$$

where $\odot$ is the Hadamard product. Both the dot product on the left and the Hadamard product on the right are **algebraic** (no complex conjugation): $(A\odot A)_{ij} = A_{ij}\,A_{ij}$, not $A_{ij}\,\overline{A_{ij}}$.

When a many-body WF is composed of both Jastrow and determinant parts, $\vec{W}(\vec{x})\cdot\vec{W}(\vec{x})$ can be computed as

$$
\begin{aligned}
\vec{W}(\vec{x})\cdot\vec{W}(\vec{x}) &= \sum_{i,j}\left[\left(W^{J^{\uparrow}} + W^{D^{\uparrow}}\right) \odot \left(W^{J^{\uparrow}} + W^{D^{\uparrow}}\right)\right]_{i,j} \\
&\quad + \sum_{i',j'}\left[\left(W^{J^{\downarrow}} + W^{D^{\downarrow}}\right) \odot \left(W^{J^{\downarrow}} + W^{D^{\downarrow}}\right)\right]_{i',j'}.
\end{aligned}
$$

where $W^{J^{\uparrow}}$ and $W^{J^{\downarrow}}$ are the gradients of the Jastrow factor.

The Laplacian reads:

$$
\nabla_l^2 \ln D(\vec{x}) =
\begin{cases}
\vec{\nabla}_u \sum_i\left(F^{-1}_{i,u}(\vec{x})\,\vec{\nabla}_u f(\vec{r}_u^{\uparrow},\vec{r}_i^{\downarrow})\right) \\
\quad = \sum_i\left(\vec{\nabla}_u F^{-1}(\vec{x})_{i,u}\,\vec{\nabla}_u f(\vec{r}_u^{\uparrow},\vec{r}_i^{\downarrow})\right)
+ \sum_i\left(F^{-1}_{i,u}(\vec{x})\,\nabla_u^2 f(\vec{r}_u^{\uparrow},\vec{r}_i^{\downarrow})\right) & (l=u), \\
\vec{\nabla}_d \sum_j\left(F^{-1}_{d,j}(\vec{x})\,\vec{\nabla}_d f(\vec{r}_j^{\uparrow},\vec{r}_d^{\downarrow})\right) \\
\quad = \sum_j\left(\vec{\nabla}_d F^{-1}(\vec{x})_{d,j}\,\vec{\nabla}_d f(\vec{r}_j^{\uparrow},\vec{r}_d^{\downarrow})\right)
+ \sum_j\left(F^{-1}_{d,j}(\vec{x})\,\nabla_d^2 f(\vec{r}_j^{\uparrow},\vec{r}_d^{\downarrow})\right) & (l=d).
\end{cases}
$$

The derivative of an inverse matrix is

$$
\frac{\partial A^{-1}}{\partial p} = -A^{-1}\frac{\partial A}{\partial p}A^{-1}.
$$

Thus, for example for $l=u$,

$$
\begin{aligned}
\sum_i \vec{\nabla}_u F^{-1}_{i,u}(\vec{x})\,\vec{\nabla}_u f(\vec{r}_u^{\uparrow},\vec{r}_i^{\downarrow})
&= -\sum_i\sum_{\mu,\nu} F^{-1}_{i,\mu}(\vec{x})\,\vec{\nabla}_u f(\vec{r}_{\mu}^{\uparrow},\vec{r}_{\nu}^{\downarrow})\,F^{-1}_{\nu,u}(\vec{x})\,\vec{\nabla}_u f(\vec{r}_u^{\uparrow},\vec{r}_i^{\downarrow}) \\
&= -\sum_{i,\nu} F^{-1}_{i,u}(\vec{x})\,\vec{\nabla}_u f(\vec{r}_u^{\uparrow},\vec{r}_{\nu}^{\downarrow})\,F^{-1}_{\nu,u}(\vec{x})\,\vec{\nabla}_u f(\vec{r}_u^{\uparrow},\vec{r}_i^{\downarrow}) \\
&= -\left(\sum_i F^{-1}_{i,u}(\vec{x})\,\vec{\nabla}_u f(\vec{r}_u^{\uparrow},\vec{r}_i^{\downarrow})\right)
\cdot
\left(\sum_{\nu} F^{-1}_{\nu,u}(\vec{x})\,\vec{\nabla}_u f(\vec{r}_u^{\uparrow},\vec{r}_{\nu}^{\downarrow})\right) \\
&= -\,\bigl(\vec{A}_u\bigr) \cdot \bigl(\vec{A}_u\bigr),
\qquad
\vec{A}_u \equiv \sum_i F^{-1}_{i,u}(\vec{x})\,\vec{\nabla}_u f(\vec{r}_u^{\uparrow},\vec{r}_i^{\downarrow}).
\end{aligned}
$$

The dot product on the last line is the **algebraic** square of the 3-vector $\vec{A}_u$ (polynomial multiplication of components, no complex conjugation). For real $\vec{A}_u$ it equals the modulus square $|\vec{A}_u|^{2}$; for complex $\vec{A}_u$ (complex Bloch wave function) the dot product retains the algebraic form $\sum_\gamma A_{u,\gamma}^{2}$.

Thus, finally, the Laplacian for the $l$-th electron becomes

$$
\nabla_l^2 \ln D(\vec{x}) =
\begin{cases}
-\left(\sum_i F^{-1}_{i,u}(\vec{x})\,\vec{\nabla}_u f(\vec{r}_u^{\uparrow},\vec{r}_i^{\downarrow})\right)
\cdot
\left(\sum_i F^{-1}_{i,u}(\vec{x})\,\vec{\nabla}_u f(\vec{r}_u^{\uparrow},\vec{r}_i^{\downarrow})\right) \\
\quad + \sum_i\left(F^{-1}_{i,u}(\vec{x})\,\nabla_u^2 f(\vec{r}_u^{\uparrow},\vec{r}_i^{\downarrow})\right) & (l=u),\\[1em]
-\left(\sum_j F^{-1}_{d,j}(\vec{x})\,\vec{\nabla}_d f(\vec{r}_j^{\uparrow},\vec{r}_d^{\downarrow})\right)
\cdot
\left(\sum_j F^{-1}_{d,j}(\vec{x})\,\vec{\nabla}_d f(\vec{r}_j^{\uparrow},\vec{r}_d^{\downarrow})\right) \\
\quad + \sum_j\left(F^{-1}_{d,j}(\vec{x})\,\nabla_d^2 f(\vec{r}_j^{\uparrow},\vec{r}_d^{\downarrow})\right) & (l=d).
\end{cases}
$$

Therefore, the total Laplacian of the determinant part can be written as follows. If one defines

$$
F^{(\nabla_{\gamma},\uparrow)}_{i,j} = \vec{\nabla}_{i,\gamma}f(\vec{r}_i^{\uparrow},\vec{r}_j^{\downarrow}), \quad
F^{(\nabla_{\gamma},\downarrow)}_{i,j} = \vec{\nabla}_{j,\gamma}f(\vec{r}_i^{\uparrow},\vec{r}_j^{\downarrow}),
$$

$$
F^{(\nabla^2,\uparrow)}_{i,j} = \nabla_i^2 f(\vec{r}_i^{\uparrow},\vec{r}_j^{\downarrow}), \quad
F^{(\nabla^2,\downarrow)}_{i,j} = \nabla_j^2 f(\vec{r}_i^{\uparrow},\vec{r}_j^{\downarrow}),
$$

with $F^{(\nabla_{\gamma},\downarrow)}_{i,j}=0$ and $F^{(\nabla^2,\downarrow)}_{i,j}=0$ for $j>N_e^{\downarrow}$, then (with $\sigma=(\uparrow,\downarrow)$):

$$
\begin{aligned}
\sum_l \nabla_l^2 \ln D(\vec{x})
&= -\operatorname{tr}\left[F^{(\nabla_x,\uparrow)}(\vec{x})F^{-1}(\vec{x}) \odot F^{(\nabla_x,\uparrow)}(\vec{x})F^{-1}(\vec{x})\right] \\
&\quad -\operatorname{tr}\left[F^{(\nabla_y,\uparrow)}(\vec{x})F^{-1}(\vec{x}) \odot F^{(\nabla_y,\uparrow)}(\vec{x})F^{-1}(\vec{x})\right] \\
&\quad -\operatorname{tr}\left[F^{(\nabla_z,\uparrow)}(\vec{x})F^{-1}(\vec{x}) \odot F^{(\nabla_z,\uparrow)}(\vec{x})F^{-1}(\vec{x})\right] \\
&\quad -\operatorname{tr}\left[F^{-1}(\vec{x})F^{(\nabla_x,\downarrow)}(\vec{x}) \odot F^{-1}(\vec{x})F^{(\nabla_x,\downarrow)}(\vec{x})\right] \\
&\quad -\operatorname{tr}\left[F^{-1}(\vec{x})F^{(\nabla_y,\downarrow)}(\vec{x}) \odot F^{-1}(\vec{x})F^{(\nabla_y,\downarrow)}(\vec{x})\right] \\
&\quad -\operatorname{tr}\left[F^{-1}(\vec{x})F^{(\nabla_z,\downarrow)}(\vec{x}) \odot F^{-1}(\vec{x})F^{(\nabla_z,\downarrow)}(\vec{x})\right] \\
&\quad +\operatorname{tr}\left[F^{(\nabla^2,\uparrow)}(\vec{x})F^{-1}(\vec{x})\right]
+ \operatorname{tr}\left[F^{-1}(\vec{x})F^{(\nabla^2,\downarrow)}(\vec{x})\right].
\end{aligned}
$$

Here $\odot$ represents the algebraic Hadamard product, $(A\odot A)_{ij} = A_{ij}\,A_{ij}$ (no complex conjugation). Notice that $\sum_l \nabla_l^2 \ln D(\vec{x})$ is a scalar value.

The gradients and Laplacians of geminals can be computed from those of atomic orbitals or molecular orbitals:

$$
\vec{\nabla}_i f(r_i,r_j) = \sum_{l,m} \lambda_{l,m}\left(\vec{\nabla}\psi_l(r_i)\right)\psi_m(r_j),
$$

$$
\nabla_i^2 f(r_i,r_j) = \sum_{l,m} \lambda_{l,m}\left(\nabla^2\psi_l(r_i)\right)\psi_m(r_j).
$$

Therefore,

$$
\bar{F}^{(\nabla_{\gamma},\uparrow)} = \left[(\Psi^{(\nabla_{\gamma},\uparrow)})^T\Lambda_A^{\uparrow,\downarrow}\Psi^{\downarrow}\,\middle|\,(\Psi^{(\nabla_{\gamma},\uparrow)})^T\Lambda_A^{\uparrow}\right],
$$

$$
\bar{F}^{(\nabla_{\gamma},\downarrow)} = \left[(\Psi^{\uparrow})^T\Lambda_A^{\uparrow,\downarrow}\Psi^{(\nabla_{\gamma},\downarrow)}\,\middle|\,0\right],
$$

$$
\bar{F}^{(\nabla^2,\uparrow)} = \left[(\Psi^{(\nabla^2,\uparrow)})^T\Lambda_A^{\uparrow,\downarrow}\Psi^{\downarrow}\,\middle|\,(\Psi^{(\nabla^2,\uparrow)})^T\Lambda_A^{\uparrow}\right],
$$

$$
\bar{F}^{(\nabla^2,\downarrow)} = \left[(\Psi^{\uparrow})^T\Lambda_A^{\uparrow,\downarrow}\Psi^{(\nabla^2,\downarrow)}\,\middle|\,0\right],
$$

with

$$
\Psi^{(\nabla_{\gamma},\uparrow)}_{l,i} = \vec{\nabla}_{\gamma}\psi_l^{\uparrow}(\vec{r}_i),\;
\Psi^{(\nabla^2,\uparrow)}_{l,i} = \nabla^2\psi_l^{\uparrow}(\vec{r}_i),
$$

$$
\Psi^{(\nabla_{\gamma},\downarrow)}_{m,j} = \vec{\nabla}_{\gamma}\psi_m^{\downarrow}(\vec{r}_j),\;
\Psi^{(\nabla^2,\downarrow)}_{m,j} = \nabla^2\psi_m^{\downarrow}(\vec{r}_j).
$$

Here $\Lambda_A^{\uparrow,\downarrow}$ is the paired block of the geminal coefficient matrix (it couples spin-up and spin-down orbitals) and $\Lambda_A^{\uparrow}$ is the unpaired block (it couples the spin-up orbitals to the $N_e^{\uparrow}-N_e^{\downarrow}$ unpaired columns of $F$). Both are defined in the AGP section of {doc}`wavefunction`; for a spin-balanced system the unpaired block is empty.

For practical implementations, one can use the Sherman-Morrison formula for derivatives and Laplacian updates of $F^{-1}(\vec{x})$ based on previous MCMC steps. However, when using AAD tools (e.g. JAX) for higher derivatives, one should implement a function with inputs $\vec{x}^{\uparrow}$ and $\vec{x}^{\downarrow}$ and outputs such as

$$
\frac{\partial}{\partial \vec{R}_{\alpha}}\nabla_l^2\ln D(\vec{x}),\quad
\frac{\partial}{\partial \vec{r}_{l'}}\nabla_l^2\ln D(\vec{x}),
$$

for, e.g., atomic forces (with space-warp). In this setting, one cannot always rely on Sherman-Morrison updates alone. Nevertheless, one can avoid explicit inversion by computing

$$
X = F^{-1}(\vec{x})F^{(\nabla,\downarrow)}(\vec{x})
$$

through solving

$$
F(\vec{x})X = F^{(\nabla,\downarrow)}(\vec{x}),
$$

e.g. via LU decomposition (`jax.scipy.linalg.solve`, `jax.scipy.linalg.lu_solve`).

Next, consider the Jastrow part. The derivation below is written for the two-body Jastrow $J_2$; the one-body, three-body and DTN Jastrow terms (see {doc}`wavefunction`) enter the gradient and Laplacian additively in the same way, and the other determinant types (SD, AGPn) follow the same algebra with their own $F$.

The two-body Jastrow factor is defined as

$$
J_2\left(\vec{r}_1\sigma_1,\ldots,\vec{r}_N\sigma_N\right)
= \sum_{i<j} v_{\sigma_i,\sigma_j}\left(|\vec{r}_i-\vec{r}_j|\right).
$$

jQMC uses a **spin-independent** pair function, $v_{\sigma_i,\sigma_j}(r) = v(r)$ for parallel and antiparallel pairs alike, with a single variational parameter $b_{\mathrm{ee}}$. Two forms are available, selected by `jastrow_2b_type` (`jqmc-tool ... --jastrow-2b-type`):

$$
v(r) =
\begin{cases}
\dfrac{r}{2\left(1+b_{\mathrm{ee}}\,r\right)} & (\texttt{pade},\ \text{default}),\\[1ex]
\dfrac{1}{2b_{\mathrm{ee}}}\left(1-e^{-b_{\mathrm{ee}}r}\right) & (\texttt{exp}).
\end{cases}
$$

Both forms satisfy $v'(0)=1/2$, i.e. the antiparallel electron-electron cusp condition; the same cusp $1/2$ is used for parallel pairs by design (the parallel-spin cusp $1/4$ is not imposed). For periodic systems $r$ is replaced by the smooth periodic distance of sec. {ref}`sec:periodic_jastrow`. Thus, the implementation of the two-body Jastrow is

$$
\begin{aligned}
J_2 &= \sum_{l=1}^{N_e^{\uparrow}}\sum_{l'=1}^{N_e^{\downarrow}} v_{\uparrow,\downarrow}\left(|\vec{r}_l^{\uparrow}-\vec{r}_{l'}^{\downarrow}|\right)
+ \sum_{l<l'}^{N_e^{\uparrow}} v_{\uparrow,\uparrow}\left(|\vec{r}_l^{\uparrow}-\vec{r}_{l'}^{\uparrow}|\right) \\
&\quad + \sum_{l<l'}^{N_e^{\downarrow}} v_{\downarrow,\downarrow}\left(|\vec{r}_l^{\downarrow}-\vec{r}_{l'}^{\downarrow}|\right),
\end{aligned}
$$

where all three terms use the same $v$ (the spin labels only mark which electrons enter each sum). The gradients of the two-body part read:

$$
\vec{\nabla}_l J_2\left(\ldots,\vec{r}_l\sigma_l,\ldots\right)=
\begin{cases}
\left[\sum_{l'\ne l}^{N_e^{\uparrow}} \vec{\nabla}_l v_{\uparrow,\uparrow}\left(|\vec{r}_l^{\uparrow}-\vec{r}_{l'}^{\uparrow}|\right)\right]
+ \left[\sum_{l''}^{N_e^{\downarrow}} \vec{\nabla}_l v_{\uparrow,\downarrow}\left(|\vec{r}_l^{\uparrow}-\vec{r}_{l''}^{\downarrow}|\right)\right] & (l=u),\\
\left[\sum_{l'\ne l}^{N_e^{\downarrow}} \vec{\nabla}_l v_{\downarrow,\downarrow}\left(|\vec{r}_l^{\downarrow}-\vec{r}_{l'}^{\downarrow}|\right)\right]
+ \left[\sum_{l''}^{N_e^{\uparrow}} \vec{\nabla}_l v_{\downarrow,\uparrow}\left(|\vec{r}_l^{\downarrow}-\vec{r}_{l''}^{\uparrow}|\right)\right] & (l=d).
\end{cases}
$$

One can store these values as three 1D vectors. The Laplacian reads

$$
\begin{aligned}
\nabla^2 J_2\left(\ldots,\vec{r}_l\sigma_l,\ldots\right)
&= \sum_l^{N_e^{\uparrow}}\Bigg(
\left[\sum_{l'\ne l}^{N_e^{\uparrow}} \nabla_l^2 v_{\uparrow,\uparrow}\left(|\vec{r}_l^{\uparrow}-\vec{r}_{l'}^{\uparrow}|\right)\right]
+ \left[\sum_{l''}^{N_e^{\downarrow}} \nabla_l^2 v_{\uparrow,\downarrow}\left(|\vec{r}_l^{\uparrow}-\vec{r}_{l''}^{\downarrow}|\right)\right]
\Bigg) \\
&\quad + \sum_l^{N_e^{\downarrow}}\Bigg(
\left[\sum_{l'\ne l}^{N_e^{\downarrow}} \nabla_l^2 v_{\downarrow,\downarrow}\left(|\vec{r}_l^{\downarrow}-\vec{r}_{l'}^{\downarrow}|\right)\right]
+ \left[\sum_{l''}^{N_e^{\uparrow}} \nabla_l^2 v_{\downarrow,\uparrow}\left(|\vec{r}_l^{\downarrow}-\vec{r}_{l''}^{\uparrow}|\right)\right]
\Bigg).
\end{aligned}
$$

This is a scalar value. jQMC evaluates the per-electron gradients and
Laplacians of the Jastrow terms in closed form; automatic differentiation
with JAX is used to cross-check them in the test suite.

### Kinetic-energy assembly for complex $\Psi$

When the determinant uses a complex Bloch AO basis at a non-$\Gamma$ twist
(`aos_data.k_point` is nonzero), $\Psi$ is **complex per walker**. The
algebraic notation introduced at the top of this section (dot products and
Hadamard products without bars) extends verbatim to this case, so the
equations of the previous subsections are unchanged in form. Three points
deserve attention.

#### 1. The complex phase enters through the analytic determinant derivatives

The kinetic-energy identity

$$
\frac{\nabla^{2}\Psi}{\Psi}
= \nabla^{2}\ln\Psi + \bigl(\nabla\ln\Psi\bigr)\cdot\bigl(\nabla\ln\Psi\bigr)
$$

is independent of whether $\Psi$ is real or complex (it is an algebraic identity in $\ln\Psi$). For a real wave function $\ln\Psi$ reduces to $\ln|\Psi|$ plus a locally constant $i\pi$, whose derivatives vanish, so only $\ln|\Psi|$ is needed. For a complex wave function jQMC still evaluates the real scalar $J + \ln|\det F|$ as the wave-function log; the missing complex phase $i\,\partial_r \arg(\det F)$ is supplied at the kinetic assembly site by the **analytic** holomorphic derivatives

$$
\nabla \ln(\det F), \qquad \nabla^{2} \ln(\det F),
$$

computed as $\mathrm{tr}(F^{-1}\,\partial F)$ and its derivative exactly as in the derivation above, with no modulus taken anywhere. The assembled $(\nabla \ln \Psi)\cdot(\nabla \ln \Psi)$ therefore retains the full complex phase without a complex $\ln\Psi$ scalar ever being formed. Where the determinant derivatives are evaluated through an SVD pseudo-inverse, the Hermitian conjugates of the singular-vector matrices (not the plain transposes) are used; for real matrices the two coincide.

#### 2. Algebraic squares, not modulus squares, in the kinetic energy

Every $\vec{v}\cdot\vec{v}$, $A\odot A$, and $\mathrm{tr}(A\odot A)$ written in the previous subsections is **bilinear** (not sesquilinear): it consumes its arguments without complex conjugation. The only change for a complex wave function is that the gradient of $\ln\Psi$ is complex. No modulus square is taken at the kinetic level -- that would be the wrong operator.

The assembled kinetic $T(\mathbf{x})$ and $\vec{W}\cdot\vec{W}$ are complex per walker; the physical kinetic energy is

$$
\langle K\rangle = \mathrm{Re}\,\bigl\langle K(\vec{x})\bigr\rangle_{|\Psi|^{2}},
$$

and $|\mathrm{Im}\langle K\rangle|$, which vanishes in expectation for a Hermitian operator, serves as a diagnostic (cf. {ref}`sec:error-analysis`).

#### 3. AS regularization: modulus square is correct here

The Attaccalite--Sorella regularization factor (defined in {doc}`vmc`, "Reweighting and AS Regularization") is a **distinct** site where the modulus square, not the algebraic square, is the correct operation: $R^{\varepsilon}$ uses Frobenius and row/column norms of $F^{-1}$ and of the geminal matrix $F$, which are inherently norms (sums of squared moduli). For a real wave function the algebraic square and the modulus square coincide; for a complex wave function only the modulus square keeps $R^{\varepsilon}$ real and positive, which is required for the fractional power in its definition to be well defined and for the guiding distribution to stay positive. jQMC therefore evaluates all AS regularization norms as $|\cdot|^{2}$ in every code path.

## Coulomb

Coulomb evaluation follows `structure_data.pbc_flag`: molecular systems use bare `1/r` kernels; periodic systems use Ewald kernels. Periodic runs require Ewald data built or loaded with the Hamiltonian.

### Charge-neutrality precondition

(ewald_charge_neutrality)=
A periodic cell must be neutral. If `n_electron` is supplied, construction checks that the effective nuclear charge sum equals `n_electron`; all-electron atoms use `Z_eff = Z`, ECP atoms use `Z_eff = Z - z_core`. A non-neutral periodic cell raises `ValueError`; charged-cell corrections are not implemented.

### Ewald settings

Ewald setup uses `eta`, real-space cutoff `r_real_cut`, and reciprocal cutoff `g_max`. Users normally set only `ewald_tol` (default `1e-10`); automatic selection derives the three scalars from it. Tighter tolerance increases cutoffs. For neutral cells `eta` is a cost parameter, not an energy parameter. Detailed truncation-error and automatic-selection formulation is work in progress.

### Save, reload and vacuum slabs

HDF5 save stores the Ewald setup scalars and structure reference; heavy derived arrays are not persisted. Reload reuses saved Ewald scalars when available, otherwise rebuilds with the default tolerance. A two-dimensional material is treated as 3D periodic Ewald with a vacuum supercell, not a strict 2D Coulomb operator. Converge vacuum width and in-plane finite size independently, especially for polar slabs.

### Where users control Ewald behavior

- QE conversion CLI (`jqmc_tool`): `--ewald-tol` (default `1e-10`).
- QE programmatic API (`build_hamiltonian_from_wavefunction` / `build_hamiltonian_from_projection`): `ewald_tol` keyword.
- Restart: saved Ewald scalars are reused automatically.

In normal user flows, tune `ewald_tol`; the remaining Ewald scalars are selected automatically.

### MPC finite-size correction

For periodic systems jQMC can report the Model Periodic Coulomb (MPC) finite-size correction of the electron-electron interaction [Williamson et al., Phys. Rev. B 55, R4851 (1997)](https://doi.org/10.1103/PhysRevB.55.R4851), in addition to the Ewald energy. MPC is a passive, energy-only observable: walkers propagate with the Ewald Hamiltonian, the local energy stays Ewald, and the correction is accumulated on the same configurations. Atomic forces, stresses and variational-parameter gradients stay on the Ewald channel and are not MPC-corrected. The detailed MPC formulation is work in progress.

`dE_MPC` reports the ensemble average of `V_MPC_ee - V_Ewald_ee`; `E_MPC = E_Ewald + dE_MPC`. Because these contributions are correlated, retain their covariance when estimating the error of the corrected total; do not add independent error bars.

Input keys are optional in the `[coulomb]` section of the `jqmc` TOML input:

| Key | Default | Meaning |
| --- | --- | --- |
| `mpc` | `false` | Accumulate the MPC correction. Requires a periodic Hamiltonian. |
| `mpc_cutoff` | `30.0` (Ha) | Reciprocal-space kinetic-energy cutoff of the MPC kernel and density grid. Convergence parameter: double it until `dE_MPC` stops moving. |

`mpc = true` is supported for `job_type = mcmc`, `lrdmc-bra` and `lrdmc-tau` (single twist and twist-averaged); it is rejected for `job_type = vmc`, because wavefunction optimization does not publish a final MPC estimate. Run an MCMC or LRDMC sampling job after optimization. Output reports `dE_MPC` and corrected total `E_MPC`; Python API getters are `get_dE_mpc` and `get_E_mpc` of the MCMC / LRDMC drivers, next to Ewald `get_E`.

(sec:Coulomb-and-ECP)=

## Effective Core Potential (ECP)

The previous section describes the bare $1/r$ Coulomb channel, which
is the only potential term in all-electron (AE) calculations. When
effective core potentials (ECPs) are used, the total electronic
potential acquires a second contribution that captures the
core-electron mean field. This section documents how jQMC handles
the ECP, in both the molecular and the periodic branches.

### Decomposition of the total potential

The total electronic potential is split into a Coulomb part and an
ECP part,

$$
V(\mathbf{x}) = V^{\mathrm{bare}}(\mathbf{x}) + V^{\mathrm{ecp}}(\mathbf{x}),
$$ (V_total_decomp)

with $\mathbf{x} = (\{\mathbf{r}_i\}_{i=1}^{N_e}, \{\mathbf{R}_I, Z_I\}_{I=1}^{N_a})$
collecting electron and nuclear data. The bare part further splits
into the standard three Coulomb channels,
$V^{\mathrm{bare}} = V^{\mathrm{ei}} + V^{\mathrm{ee}} + V^{\mathrm{ii}}$.

For molecular systems (`pbc_flag=False`) the three channels are
plain Cartesian sums,

$$
V^{\mathrm{ei}}(\mathbf{x}) = -\sum_{i,I}
\frac{Z_I^{\mathrm{eff}}}{|\mathbf{r}_i - \mathbf{R}_I|},
$$ (V_ei_mol)

$$
V^{\mathrm{ee}}(\mathbf{x}) = \sum_{j>i}
\frac{1}{|\mathbf{r}_i - \mathbf{r}_j|},
$$ (V_ee_mol)

$$
V^{\mathrm{ii}}(\mathbf{x}) = \sum_{J>I}
\frac{Z_I^{\mathrm{eff}}\,Z_J^{\mathrm{eff}}}
     {|\mathbf{R}_I - \mathbf{R}_J|},
$$ (V_ii_mol)

with the effective ionic charge

$$
Z_I^{\mathrm{eff}} =
\begin{cases}
Z_I & (\text{AE atom}), \\
Z_I - z_{\mathrm{core},I} & (\text{ECP atom}).
\end{cases}
$$ (Z_eff_def)

For periodic systems (`pbc_flag=True`) the same three channels are
evaluated by the Ewald decomposition documented in the Coulomb
section, again using $Z_I^{\mathrm{eff}}$ in place of $Z_I$ for ECP
atoms. The charge-neutrality precondition
$\sum_I Z_I^{\mathrm{eff}} = N_e$
(the {ref}`charge-neutrality condition <ewald_charge_neutrality>`) makes $V^{\mathrm{bare}}$ independent of $\eta$.

The ECP part adds the semi-local correction that represents the
removed core electrons. In jQMC the electron-ion *Coulomb*
$1/|\mathbf{r}_i - \mathbf{R}_I|$ term is kept inside $V^{\mathrm{bare}}$
even on ECP atoms (using $Z_I^{\mathrm{eff}}$), and $V^{\mathrm{ecp}}$
collects the remaining semi-local terms only. This separation matches
the `compute_bare_coulomb_potential` / `compute_ecp_coulomb_potential`
split in `jqmc.coulomb_potential` and keeps the two implementations
loosely coupled.

### Semi-local form

The non-Coulomb part of an ECP is the standard semi-local form
[Mitas, Shirley, Ceperley, J. Chem. Phys. 95, 3467 (1991)](https://doi.org/10.1063/1.460849):

$$
\hat{V}_{\mathrm{ecp}}^{i,I} = \hat{V}_{\mathrm{eloc}}^{i,I} +
\sum_{l=0}^{l_{\max}^{I}} \hat{V}_{l}^{i,I}\,\hat{P}_{l}^{i,I},
$$ (ecp_semilocal)

where $i$ and $I$ are electron and nucleus indices,

$$
\hat{P}_{l}^{i,I} = \sum_{m=-l}^{l} |Y_{lm}\rangle_{i,I}\langle Y_{lm}|_{i,I}
$$ (ecp_angular_proj)

projects the angular part of electron $i$'s coordinates relative to
nucleus $I$ onto angular momentum $l$, and
$V_{\mathrm{eloc}}^{I}(\mathbf{r}_i)$, $V_{l}^{I}(\mathbf{r}_i)$ are
purely radial functions of $r_{i,I} = |\mathbf{r}_i - \mathbf{R}_I|$.
The total ECP correction summed over electrons and nuclei is

$$
V^{\mathrm{ecp}}(\mathbf{x}) =
\sum_{i=1}^{N_e}\sum_{I=1}^{N_a}\,
\frac{\langle \mathbf{x}|\hat{V}_{\mathrm{ecp}}^{i,I}|\Psi\rangle}
     {\langle \mathbf{x}|\Psi\rangle}.
$$ (V_ecp_total)

The local part $\hat{V}_{\mathrm{eloc}}^{i,I}$ is diagonal in
$\mathbf{x}$ (no angular projection), while every $l$-channel of the
non-local part requires the angular integration discussed below.

### Radial parametrization

Both $V_{\mathrm{eloc}}^{I}$ and $V_{l}^{I}$ use a Gaussian-power
expansion,

$$
V_{\mathrm{eloc}}^{I}(\mathbf{r}_i) =
\frac{1}{r_{i,I}^{2}}\sum_{k=1}^{N_I}
a_{I,k}\,r_{i,I}^{\,n_{I,k}}\,
\exp\!\bigl(-b_{I,k}\,r_{i,I}^{2}\bigr),
$$ (V_eloc_param)

$$
V_{l}^{I}(\mathbf{r}_i) =
\frac{1}{r_{i,I}^{2}}\sum_{k=1}^{N_{I,l}}
a_{I,k,l}\,r_{i,I}^{\,n_{I,k,l}}\,
\exp\!\bigl(-b_{I,k,l}\,r_{i,I}^{2}\bigr),
$$ (V_l_param)

with per-term parameters $(a_{I,k}, n_{I,k}, b_{I,k})$ for the local
channel and $(a_{I,k,l}, n_{I,k,l}, b_{I,k,l})$ for each non-local
angular channel $l$. $N_I$ and $N_{I,l}$ are the number of Gaussian
terms per species and channel; in jQMC they are stored as flat ragged
arrays on `Coulomb_potential_data`. The expansion is the
ccECP / Trail-Needs convention used by the standard ECP libraries.

### Angular projection on the wave function

Following the standard treatment [Mitas et al., 1991](https://doi.org/10.1063/1.460849), the action of
the $l$-channel projector on a many-body wave function $\Psi$
reduces to an angular integration at fixed electron-nucleus distance
$r_{i,I}$:

$$
\begin{aligned}
\frac{\langle\mathbf{x}|\hat{V}_{l}^{i,I}\hat{P}_{l}^{i,I}|\Psi\rangle}
     {\langle\mathbf{x}|\Psi\rangle}
&= V_{l}^{I}(r_{i,I})
\sum_{m=-l}^{l} Y_{lm}(\Omega_{i,I})
\int d\Omega'_{i,I}\,Y_{lm}^{*}(\Omega'_{i,I})\,
\frac{\Psi(\dots,\mathbf{r}'_i,\dots)}{\Psi(\dots,\mathbf{r}_i,\dots)} \\
&= V_{l}^{I}(r_{i,I})\,\frac{2l+1}{4\pi}
\int d\Omega'_{i,I}\,
P_{l}\!\bigl(\mathbf{s}_{i,I}\cdot\mathbf{s}'_{i,I}\bigr)\,
\frac{\Psi(\dots,\mathbf{r}'_i,\dots)}{\Psi(\dots,\mathbf{r}_i,\dots)},
\end{aligned}
$$ (ecp_projection)

with unit vectors

$$
\mathbf{s}_{i,I} = \frac{\mathbf{r}_i - \mathbf{R}_I}
                          {|\mathbf{r}_i - \mathbf{R}_I|},
\qquad
\mathbf{s}'_{i,I} = \frac{\mathbf{r}'_i - \mathbf{R}_I}
                           {|\mathbf{r}'_i - \mathbf{R}_I|}.
$$ (ecp_unit_vectors)

The last identity uses the addition theorem for spherical harmonics,
$\sum_{m=-l}^{l}Y_{lm}(\Omega)Y_{lm}^{*}(\Omega')
= \tfrac{2l+1}{4\pi}\,P_{l}(\mathbf{s}_{i,I}\cdot\mathbf{s}'_{i,I})$,
where $P_l$ is the Legendre polynomial. The Coulomb / mean-field part
of the ECP -- the asymptotic $-Z_I^{\mathrm{eff}}/r$ tail of the
local channel -- is **not** repeated here: it is already accounted
for inside $V^{\mathrm{ei}}$ with $Z_I^{\mathrm{eff}}$
(eq. {eq}`V_ei_mol` and its Ewald counterpart).

### Mitas angular quadrature

jQMC follows Ref. [Mitas et al., 1991](https://doi.org/10.1063/1.460849) and replaces the angular
integral (eq. {eq}`ecp_projection`) by a single-quadrature rule on a
regular polyhedron with $N_V$ vertices. Four meshes are implemented
(`Nv` argument of the ECP kernels):

| `Nv` | Polyhedron        | Exact degree |
| ---- | ----------------- | ------------ |
| `4`  | tetrahedron       | 2 |
| `6`  | octahedron        | 3 |
| `12` | icosahedron       | 5 |
| `18` | extended octahedron | 5 |

The vertices and weights for each polyhedron are tabulated in
[Mitas et al., 1991](https://doi.org/10.1063/1.460849) and hard-coded in `jqmc.coulomb_potential` as
the `tetrahedron_sym_mesh_Nv4`, `octahedron_sym_mesh_Nv6`,
`icosahedron_sym_mesh_Nv12`, `octahedron_sym_mesh_Nv18` constants.
The default is `Nv = 12` (icosahedron). Neither `Nv` nor the
nearest-neighbor count $N_{\mathrm{NN}}$ (default `1`, see below) is
exposed as a TOML key of the `jqmc` CLI; they can only be changed
through the Python API.
The quadrature points around nucleus $I$ at distance $r_{i,I}$ are

$$
\mathbf{r}^{(m)}_i = \mathbf{R}_I + r_{i,I}\,\boldsymbol{\Delta}_m,
\qquad m=1,\dots,N_V,
\qquad \boldsymbol{\Delta}_m = \boldsymbol{\delta}_m / |\boldsymbol{\delta}_m|,
$$ (ecp_quad_points)

so the trial wave function only has to be evaluated on a sphere of
radius $r_{i,I}$ centered on $\mathbf{R}_I$ -- the radial coordinate
is unchanged. The polyhedron is randomly rotated at each MCMC /
LRDMC measurement. The Haar sampler draws three independent
$u_0,u_1,u_2\sim U[0,1)$ and forms
$R=R_z(\alpha)R_y(\beta)R_x(\gamma)$ with
$\alpha=2\pi u_0$, $\beta=\arcsin(2u_1-1)$, and $\gamma=2\pi u_2$;
the implementation returns $R_T=R^T$. Two uniformly sampled angles are
not a Haar rotation. LRDMC uses independent random keys for its kinetic
lattice rotation and ECP quadrature rotation. Randomization removes the bias introduced by the finite-degree quadrature when
acting on a generic many-body wave function.

The discretized projection reads

$$
\begin{aligned}
\frac{\langle\mathbf{x}|\hat{V}_{l}^{i,I}\hat{P}_{l}^{i,I}|\Psi\rangle}
     {\langle\mathbf{x}|\Psi\rangle}
&\;\simeq\;
\sum_{m=1}^{N_V} (2l+1)\,V_{l}^{I}(r_{i,I})\,w_{m}\,
P_{l}\!\bigl(\mathbf{s}_{i,I}\cdot\boldsymbol{\Delta}_{m}\bigr)\,
\frac{\Psi(\dots,\mathbf{r}^{(m)}_{i},\dots)}
     {\Psi(\dots,\mathbf{r}_{i},\dots)},
\end{aligned}
$$ (ecp_quadrature)

where $w_m$ are the quadrature weights. The wave-function ratios on
the $N_V$ quadrature points are produced by single-electron rank-1
(Sherman-Morrison) updates of the determinant inverse (see "Fast update of
the determinant part" in {doc}`wavefunction`) combined with a Jastrow
ratio evaluation, exactly as in the Metropolis step. Together they form the
`compute_ecp_non_local_parts_nearest_neighbors_fast_update` path
in `jqmc.coulomb_potential`.

### Molecular evaluation: all-pairs vs nearest-neighbors

Because $V_l^{I}$ and $V_{\mathrm{eloc}}^{I}$ decay as Gaussian
$\times r^{n}$ outside the core radius (typically beyond
$\sim 2$-$3$ Bohr), most $(i, I)$ pairs contribute negligibly to
$V^{\mathrm{ecp}}$. jQMC provides two complementary kernels:

- `compute_ecp_non_local_parts_all_pairs` -- evaluates every
  electron-nucleus pair. Reserved for molecular (or otherwise small)
  systems, debugging, and as a numerical reference.
- `compute_ecp_non_local_parts_nearest_neighbors[_fast_update]` --
  for each electron, evaluates only the $N_{\mathrm{NN}}$ closest
  nuclei (in the minimum-image sense for PBC; default $N_{\mathrm{NN}}=1$).
  This is the default path and is mandatory for PBC.

The local part is always evaluated over all $(i, I)$ pairs
(`compute_ecp_local_parts_all_pairs`); its cost is small because the
Gaussian-polynomial form (eq. {eq}`V_eloc_param`) is a single
scalar per pair, and no wave-function evaluation is required.

(sec:pbc_wrap_conventions)=
### PBC coordinate and wrap conventions

These conventions apply to VMC/MCMC and LRDMC.

- Walkers keep unwrapped Cartesian positions after moves; only the initial configuration is generated inside the cell.
- Periodic AO evaluation wraps evaluation points internally and returns the Bloch amplitude at unwrapped points, so downstream code needs no explicit phase handling.
- Difference-based terms use the minimum-image displacement. Jastrow distances use the smooth periodic distance in sec. {ref}`sec:periodic_jastrow`.
- At complex twists, test separately from Gamma; a Gamma-only check cannot reveal a missing phase.

### Periodic adaptation

The detailed formulation of the periodic ECP adaptation is work in progress.

Periodic ECP evaluation uses the minimum-image (MIC) displacement and distance to the nearest nuclear image for radial arguments, projector direction, and nearest-atom selection. MIC accuracy depends on potential range and cell size and must be checked.

- Local ECP: every electron-nucleus pair is evaluated with the radial local potential at its MIC distance, without an additional image sum. Check omitted-image contributions for small cells or diffuse potential components.
- Non-local ECP: for each electron, only the N_NN atoms with the smallest MIC distance contribute; default N_NN = 1. Selection is over atoms of the cell, each represented by its single nearest image. Increase N_NN or the supercell to check convergence of omitted projector terms; the configured count alone does not guarantee a contribution below the requested tolerance.
- Non-local projector directions use MIC displacements and are never wrapped.
- Non-local quadrature points are passed unwrapped; the periodic AO evaluator handles wrapping and Bloch-phase restoration internally. A bare wrap would drop the phase at nonzero twist.

The bare electron-ion sum on ECP atoms is folded into the Ewald `ei` channel using effective nuclear charges. The same Ewald kernel handles all-electron and ECP atoms; charge neutrality keeps the total independent of the Ewald parameter.

### J_DTN constraints with ECP atoms

The detailed formulation of the J_DTN constraints is work in progress.

J_DTN constraints are determined from the same ECP records; core charge alone is insufficient because a zero-core ECP still removes the nuclear Coulomb singularity. J_DTN Hamiltonians constrain chi to zero slope on ECP atoms and to the Kato slope on all-electron atoms. A serialized J_DTN block with `apply_en_cusp=False` is rejected during Hamiltonian `sanity_check`; regenerate the Hamiltonian and rerun VMC optimization before production sampling.

### Fast update of the wave-function ratios

The $N_V$ wave-function ratios of the quadrature points are not recomputed from scratch. Each point differs from the current configuration by a single-electron displacement, so the determinant ratio is a rank-1 (Sherman-Morrison) update and the Jastrow ratio comes from the analytic per-term helpers in `jqmc.jastrow_factor` (DTN ratios are described in sec. {ref}`sec:Jastrow_DTN`); the two factorize. This allows `compute_ecp_non_local_parts_nearest_neighbors_fast_update` to use $N_V N_{\mathrm{NN}}$ rank-1 ratios per electron, each costing $\mathcal{O}(N_e)$ once the new geminal row or column is formed, instead of $N_V N_{\mathrm{NN}}$ full determinant evaluations of $\mathcal{O}(N_e^{3})$ each. The determinant-update algebra is documented in "Fast update of the determinant part" in {doc}`wavefunction`.

### Relation to LRDMC: non-diagonal hoppings

In LRDMC (see {doc}`lrdmc`) the non-local potential acts as non-diagonal hoppings $\mathbf{x}\to\mathbf{x}_m$, where $\mathbf{x}_m$ differs from $\mathbf{x}$ only by moving electron $i$ to a quadrature point $\mathbf{r}^{(m)}_i$. Each contributes one term of the quadrature sum to the importance-sampled Green's function (eq. {eq}`lrdmc_greens_function`). Within the discretized sum, only hoppings that keep the electron-nucleus distance fixed contribute. The maximum number of non-diagonal matrix elements per LRDMC step is $N_V\,N_a\,N_e$, reduced in practice by the nearest-neighbour cutoff to about $N_V\,N_e\,N_{\mathrm{NN}}$. On PBC the hopped configuration uses the unwrapped quadrature point as the new electron position, as for any other move (sec. {ref}`sec:pbc_wrap_conventions`), and the matrix element uses the MIC projector direction.

### ECP non-local term for complex Bloch wave functions (non-Gamma twist)

For non-Gamma twists, the non-local ECP contribution is complex. The complex AO basis returns the true Bloch amplitude at evaluation points, including points outside the home cell; the ECP kernel uses the wave-function ratio at the two positions without an additional per-node phase factor. The projector ingredients and weights are the same as for a real wave function. Detailed formulation is work in progress.

The physical energy uses the real part of the per-walker non-local contribution when local-energy contributions are aggregated; no intermediate real cast discards the imaginary part. In expectation the imaginary part is zero. The absolute value of the imaginary expectation is a diagnostic; see {ref}`sec:error-analysis`.

Only the determinant/geminal factor carries the Bloch phase. The Jastrow factor is real and strictly periodic, so it requires no wrap correction. The Metropolis ratio uses the same position convention in numerator and denominator, so the per-walker phase cancels and the step is unaffected.
