(sec:atomic_orbitals)=
# Atomic orbitals (AOs) and Basis sets

## Atomic orbitals for the pairing function and the Jastrow factor

One of the most common choices for atomic orbitals in QMC is atom-centered Gaussian-type orbitals (GTOs). A primitive GTO $\psi_{l,m,\alpha}(\mathbf{r})$ can be constructed using either solid harmonics or Cartesian polynomial basis functions.

### Gaussian-Type Orbitals with Solid-Harmonic Basis

Primitive orbitals with regular solid harmonics are given by:

```{math}
\psi_{l,m,\alpha}(\mathbf{r}) = \mathcal{N}^{\rm solid}_{l,m,\alpha}\;\mathcal{R}_{\alpha}(\mathbf{r})\;\mathcal{S}_{l,m,\alpha}(\mathbf{r})
```

where the radial part is

```{math}
\mathcal{R}_{\alpha}(\mathbf{r}) = e^{-\alpha\,|\mathbf{r}-\mathbf{R}_A|^2}
```

and the solid harmonics are

```{math}
\mathcal{S}_{l,m,\alpha}(\mathbf{r}) = \sqrt{\frac{4\pi}{2l+1}}\;|\mathbf{r}-\mathbf{R}_A|^l\;\mathcal{Y}_{l,m}(\theta_A,\phi_A).
```

Here $\alpha$ is the Gaussian exponent, $\mathbf{R}_A=(X_A,Y_A,Z_A)$ is the center of atom $A$, $(\theta_A,\phi_A)$ are the polar angles of $\mathbf{r}-\mathbf{R}_A$, and $\mathcal{Y}_{l,m}(\theta,\phi)$ are real spherical harmonics. The Racah normalization ensures

```{math}
\int_0^\pi\!\int_0^{2\pi} \sin\theta\,\mathcal{S}_{l,m,\alpha}^*(\mathbf{r})\,\mathcal{S}_{l,m,\alpha}(\mathbf{r})\,\mathrm{d}\phi\,\mathrm{d}\theta = \frac{4\pi}{2l+1}\;r^{2l}.
```

The normalization constant is

```{math}
\mathcal{N}^{\rm solid}_{l,m,\alpha} = \sqrt{\frac{(2\alpha/\pi)^{3/2}\,(4\alpha)^l}{(2l-1)!!}}.
```

One may also write GTOs with spherical harmonics normalization

```{math}
\psi_{l,m,\alpha}(\mathbf{r}) = \mathcal{N}^{\rm sphe}_{l,m,\alpha}\;|\mathbf{r}-\mathbf{R}_A|^l\;\mathcal{R}_{\alpha}(\mathbf{r})\;\mathcal{Y}_{l,m}(\theta_A,\phi_A)
```

with

```{math}
\mathcal{N}^{\rm sphe}_{l,m,\alpha} = \sqrt{\frac{2^{2l+3}(l+1)!(2\alpha)^{l+3/2}}{(2l+2)!\,\sqrt{\pi}}}.
```

These two normalizations satisfy

```{math}
\frac{\mathcal{N}^{\rm solid}_{l,m,\alpha}}{\mathcal{N}^{\rm sphe}_{l,m,\alpha}} = \sqrt{\frac{2l+1}{4\pi}}.
```

### Gaussian-Type Orbitals with Cartesian Basis

Primitive orbitals in the Cartesian basis are

```{math}
\psi_{n_x,n_y,n_z,\alpha}(\mathbf{r}) = \mathcal{N}^{\rm cart}_{n_x,n_y,n_z,\alpha}\;e^{-\alpha\,|\mathbf{r}-\mathbf{R}_A|^2}\;(x-X_A)^{n_x}(y-Y_A)^{n_y}(z-Z_A)^{n_z}.
```

The normalization is

```{math}
\mathcal{N}^{\rm cart}_{n_x,n_y,n_z,\alpha} = \sqrt{\frac{(2\alpha/\pi)^{3/2}\,(4\alpha)^{n_x+n_y+n_z}}{(2n_x-1)!!\,(2n_y-1)!!\,(2n_z-1)!!}} = \sqrt{\frac{(2\alpha/\pi)^{3/2}\,(8\alpha)^{n_x+n_y+n_z}n_x!n_y!n_z!}{(2n_x)!(2n_y)!(2n_z)!}}.
```

We define the total angular momentum as $l=n_x+n_y+n_z$.  A basis of order $l$ includes all monomials of that degree (e.g., $l=2$ gives $d_{xx},d_{xy},\dots,d_{zz}$).  Note that normalization constants differ among Cartesian orbitals with the same $l$ (except for $s$ and $p$ shells).  For example:

```{math}
\mathcal{N}^{\rm cart}_{2,0,0,\alpha} = \sqrt{\frac{(2\alpha/\pi)^{3/2}(4\alpha)^2}{3\cdot1}},
```

```{math}
\mathcal{N}^{\rm cart}_{1,1,0,\alpha} = \sqrt{\frac{(2\alpha/\pi)^{3/2}(4\alpha)^2}{1\cdot1}}.
```

### Practical Tip

In jQMC (JAX), Cartesian GTOs are computationally faster than spherical ones when using `jit` and `grad`, since they avoid branching logic by varying only polynomial exponents rather than basis-function forms.


## Real Spherical and Solid Harmonics

The **real spherical harmonics** $\mathcal{Y}_{l,m}(\theta,\phi)$ are built from the complex spherical harmonics $Y_l^m(\theta,\phi)$ with the Condon-Shortley phase:

```{math}
:label: eq-real-spherical
\mathcal{Y}_{l,m}(\theta,\phi) =
\begin{cases}
\displaystyle
\frac{1}{\sqrt{2}}\bigl(Y_l^{-|m|}(\theta,\phi) + (-1)^m\,Y_l^{|m|}(\theta,\phi)\bigr), & m>0,\\[1em]
Y_l^{0}(\theta,\phi), & m=0,\\[0.75em]
\displaystyle
\frac{i}{\sqrt{2}}\bigl(Y_l^{-|m|}(\theta,\phi) - (-1)^m\,Y_l^{|m|}(\theta,\phi)\bigr), & m<0.
\end{cases}
```

Because the complex spherical harmonics satisfy

```{math}
:label: eq-ortho-complex
\int_{0}^{\pi}\!\!\int_{0}^{2\pi}
\bigl(Y_{l'}^{m'}(\theta,\phi)\bigr)^*\,Y_l^m(\theta,\phi)\,
\sin\theta\,d\phi\,d\theta
= \delta_{l,l'}\,\delta_{m,m'},
```

the real ones are also orthonormal:

```{math}
:label: eq-ortho-real
\int_{0}^{\pi}\!\!\int_{0}^{2\pi}
\mathcal{Y}_{l',m'}(\theta,\phi)\,\mathcal{Y}_{l,m}(\theta,\phi)\,
\sin\theta\,d\phi\,d\theta
= \delta_{l,l'}\,\delta_{m,m'}.
```

The spherical harmonics have singularities at the origin $(x,y,z)=(0,0,0)$. In practice one uses the **regular solid harmonics** centered at $\mathbf{R}_A$:

```{math}
:label: eq-solid-harmonic-def
\mathcal{S}_{l,m,\alpha}(\mathbf{r})
= \sqrt{\frac{4\pi}{2l+1}}\;
|\mathbf{r}-\mathbf{R}_A|^l\;
\mathcal{Y}_{l,m}(\theta_A,\phi_A).
```

### Polynomial Form of Solid Harmonics

One efficient way to compute $\mathcal{S}_{l,m}$ is via polynomials in $(x,y,z)$. Define:

```{math}
:label: eq-Cm
C_m(x,y) =
\begin{cases}
\displaystyle
\sum_{p=0}^{|m|}\binom{|m|}{p}x^p\,y^{|m|-p}\,\cos\!\bigl(\tfrac\pi2(|m|-p)\bigr),
& m\ge0,\\[0.75em]
\displaystyle
\sum_{p=0}^{|m|}\binom{|m|}{p}x^p\,y^{|m|-p}\,\sin\!\bigl(\tfrac\pi2(|m|-p)\bigr),
& m<0,
\end{cases}
```

```{math}
:label: eq-gamma-lmk
\gamma_{l,m,k}
= (-1)^k\,2^{-l}
\,\binom{l}{k}\,\binom{2l-2k}{l}\,
\frac{(l-2k)!}{(l-2k-|m|)!},
```

```{math}
:label: eq-Gamma-lm
\Gamma_{l,m}(\mathbf{r})
= \sum_{k=0}^{\lfloor (l-|m|)/2\rfloor}
\gamma_{l,m,k}\;
|\mathbf{r}|^{2k}\;z^{\,l-2k-|m|}.
```

Then

```{math}
:label: eq-solid-harmonic-poly
\mathcal{S}_{l,m,\alpha}(\mathbf{r})
= \sqrt{\frac{(2-\delta_{m,0})(l-|m|)!}{(l+|m|)!}}\;
C_m(x-\!X_A,\;y-\!Y_A)\;
\Gamma_{l,m}(\mathbf{r}-\mathbf{R}_A).
```

### Normalization of the Radial Part

For a Gaussian radial factor
$\psi_{l,m,\alpha}(\mathbf{r})=\mathcal{N}_{l,\alpha}\,r^l\,e^{-\alpha r^2}\,\mathcal{Y}_{l,m}(\theta,\phi)$,
the overall normalization requires

```{math}
:label: eq-normalization-integral
\int_{\mathbb{R}^3}
\bigl|\psi_{l,m,\alpha}(\mathbf{r})\bigr|^2\,d\mathbf{r}
=1.
```

Separating radial and angular parts:

```{math}
:label: eq-radial-integral
\mathcal{N}_{l,\alpha}^2
\int_{0}^{\infty}r^{2l+2}\,e^{-2\alpha r^2}\,dr
=1
\;\Longrightarrow\;
\mathcal{N}_{l,\alpha}^2
\frac{(2l+2)!\,\sqrt{\pi}}{2^{2l+3}(l+1)!\,(2\alpha)^{l+\frac{3}{2}}}
=1,
```

```{math}
:label: eq-angular-integral
\int_{0}^{\pi}\!\!\int_{0}^{2\pi}
\bigl|\mathcal{Y}_{l,m}(\theta,\phi)\bigr|^2\,
\sin\theta\,d\phi\,d\theta
=1.
```

Hence the **radial normalization constant** is

```{math}
:label: eq-normalization-factor
\mathcal{N}_{l,\alpha}
= \sqrt{
\frac{2^{2l+3}(l+1)!\,(2\alpha)^{\,l+\tfrac{3}{2}}}
     {(2l+2)!\,\sqrt{\pi}}
}.
```

This factor is *identical* for all $(l,m)$ in the same shell.

### Cartesian--spherical coefficient conversion

Both AO conventions are supported, and either can be converted into the other.
The directions are not exact inverses once $l\ge2$, because a general Cartesian
shell also contains $r^2\mathcal H_{l-2}$, $r^4\mathcal H_{l-4}$, ... trace
components in addition to the harmonic part $\mathcal H_l$.

* Spherical -> Cartesian is the exact per-shell solid-harmonic embedding: no fit
  and no pseudoinverse. Expansion coefficients are row vectors, so this is a
  right multiplication, `c_cart = c_sph * T`.
* Cartesian -> spherical projects the Cartesian expansion onto the spherical
  basis. For $l\ge2$ the spherical basis spans less than the Cartesian one, so
  this projection can lose information and is not the exact inverse of the
  embedding. With real AOs, or at $\Gamma$, the result is
  real; at a non-$\Gamma$ twist it may be complex. The detailed formulation is
  work in progress.

#### Component normalization and shell discovery

TREXIO may fold `ao_normalization` into every expanded Cartesian AO, so within
one physical contracted shell the primitive coefficient vectors are proportional
rather than identical. jQMC groups AOs by atom, $l$, primitive exponents and
proportional radial coefficient vectors, rescales them to a common Cartesian
convention, and rejects missing or duplicate Cartesian monomials; the physical
function is invariant under nonzero rescaling of Cartesian components. Distinct
contractions with the same exponents but non-proportional coefficient vectors
remain distinct shells.

#### Molecular and periodic implementation

Both conversions are available for molecular and periodic pGTO bases, at
$\Gamma$ and at non-$\Gamma$ twists; re-expansion preserves the periodic basis
kind, `k`, and the Bloch phases. Every coefficient consumer -- MO coefficients, the
AGP pairing matrices, the unpaired (AGPn) block and the AO-based Jastrow
couplings -- is converted consistently, so a re-expressed basis keeps the AGP and
three-body Jastrow channels consistent. The equivalence of the two AO conventions
at the value/gradient/Laplacian level is checked through $l=6$.

---

## Tables of Real Spherical Harmonics

Below are the explicit real spherical harmonics $\mathcal{Y}_{l,m}$ up to $l=6$, written in terms of the complex $Y_l^m$ and of Cartesian coordinates.  Let $r=\sqrt{x^2+y^2+z^2}$.  The corresponding solid harmonics follow from {eq}`eq-solid-harmonic-def` by multiplying with $\sqrt{4\pi/(2l+1)}\,r^l$.

### $l=0$ ($s$ orbital)

```{math}
:label: eq-l0
\mathcal{Y}_{0,0} = \frac{1}{2}\,\sqrt{\frac{1}{\pi}}.
```

### $l=1$ ($p$ orbitals)

```{math}
:label: eq-l1
\begin{aligned}
\mathcal{Y}_{1,-1} &= i\sqrt{\tfrac12}\bigl(Y_1^{-1}+Y_1^1\bigr)
= \sqrt{\tfrac{3}{4\pi}}\;\frac{y}{r},\\
\mathcal{Y}_{1,0}  &= \;\;Y_1^0
= \sqrt{\tfrac{3}{4\pi}}\;\frac{z}{r},\\
\mathcal{Y}_{1,1}  &= \sqrt{\tfrac12}\bigl(Y_1^{-1}-Y_1^1\bigr)
= \sqrt{\tfrac{3}{4\pi}}\;\frac{x}{r}.
\end{aligned}
```

### $l=2$ ($d$ orbitals)

```{math}
:label: eq-l2
\begin{aligned}
\mathcal{Y}_{2,-2} &= i\sqrt{\tfrac12}\bigl(Y_2^{-2}-Y_2^2\bigr)
= \tfrac12\sqrt{\tfrac{15}{\pi}}\;\frac{xy}{r^2},\\
\mathcal{Y}_{2,-1} &= i\sqrt{\tfrac12}\bigl(Y_2^{-1}+Y_2^1\bigr)
= \tfrac12\sqrt{\tfrac{15}{\pi}}\;\frac{yz}{r^2},\\
\mathcal{Y}_{2,0}  &= \;\;Y_2^0
= \tfrac14\sqrt{\tfrac{5}{\pi}}\;\frac{3z^2-r^2}{r^2},\\
\mathcal{Y}_{2,1}  &= \sqrt{\tfrac12}\bigl(Y_2^{-1}-Y_2^1\bigr)
= \tfrac12\sqrt{\tfrac{15}{\pi}}\;\frac{xz}{r^2},\\
\mathcal{Y}_{2,2}  &= \sqrt{\tfrac12}\bigl(Y_2^{-2}+Y_2^2\bigr)
= \tfrac14\sqrt{\tfrac{15}{\pi}}\;\frac{x^2-y^2}{r^2}.
\end{aligned}
```

### $l=3$ ($f$ orbitals)

```{math}
:label: eq-l3
\begin{aligned}
\mathcal{Y}_{3,-3} &= i\sqrt{\tfrac12}\bigl(Y_3^{-3}+Y_3^3\bigr)
= \tfrac14\sqrt{\tfrac{35}{2\pi}}\;\frac{y(3x^2-y^2)}{r^3},\\
\mathcal{Y}_{3,-2} &= i\sqrt{\tfrac12}\bigl(Y_3^{-2}-Y_3^2\bigr)
= \tfrac12\sqrt{\tfrac{105}{\pi}}\;\frac{xyz}{r^3},\\
\mathcal{Y}_{3,-1} &= i\sqrt{\tfrac12}\bigl(Y_3^{-1}+Y_3^1\bigr)
= \tfrac14\sqrt{\tfrac{21}{2\pi}}\;\frac{y(5z^2-r^2)}{r^3},\\
\mathcal{Y}_{3,0}  &= \;\;Y_3^0
= \tfrac14\sqrt{\tfrac{7}{\pi}}\;\frac{5z^3-3zr^2}{r^3},\\
\mathcal{Y}_{3,1}  &= \sqrt{\tfrac12}\bigl(Y_3^{-1}-Y_3^1\bigr)
= \tfrac14\sqrt{\tfrac{21}{2\pi}}\;\frac{x(5z^2-r^2)}{r^3},\\
\mathcal{Y}_{3,2}  &= \sqrt{\tfrac12}\bigl(Y_3^{-2}+Y_3^2\bigr)
= \tfrac14\sqrt{\tfrac{105}{\pi}}\;\frac{(x^2-y^2)z}{r^3},\\
\mathcal{Y}_{3,3}  &= \sqrt{\tfrac12}\bigl(Y_3^{-3}-Y_3^3\bigr)
= \tfrac14\sqrt{\tfrac{35}{2\pi}}\;\frac{x(x^2-3y^2)}{r^3}.
\end{aligned}
```

### $l=4$ ($g$ orbitals)

```{math}
:label: eq-l4
\begin{aligned}
\mathcal{Y}_{4,-4} &= i\sqrt{\tfrac12}\bigl(Y_4^{-4}-Y_4^4\bigr)
= \tfrac34\sqrt{\tfrac{35}{\pi}}\;\frac{xy(x^2-y^2)}{r^4},\\
\mathcal{Y}_{4,-3} &= i\sqrt{\tfrac12}\bigl(Y_4^{-3}+Y_4^3\bigr)
= \tfrac34\sqrt{\tfrac{35}{2\pi}}\;\frac{y(3x^2-y^2)z}{r^4},\\
\mathcal{Y}_{4,-2} &= i\sqrt{\tfrac12}\bigl(Y_4^{-2}-Y_4^2\bigr)
= \tfrac34\sqrt{\tfrac{5}{\pi}}\;\frac{xy(7z^2-r^2)}{r^4},\\
\mathcal{Y}_{4,-1} &= i\sqrt{\tfrac12}\bigl(Y_4^{-1}+Y_4^1\bigr)
= \tfrac34\sqrt{\tfrac{5}{2\pi}}\;\frac{y(7z^3-3zr^2)}{r^4},\\
\mathcal{Y}_{4,0}  &= \;\;Y_4^0
= \tfrac{3}{16}\sqrt{\tfrac{1}{\pi}}\;\frac{35z^4-30z^2r^2+3r^4}{r^4},\\
\mathcal{Y}_{4,1}  &= \sqrt{\tfrac12}\bigl(Y_4^{-1}-Y_4^1\bigr)
= \tfrac34\sqrt{\tfrac{5}{2\pi}}\;\frac{x(7z^3-3zr^2)}{r^4},\\
\mathcal{Y}_{4,2}  &= \sqrt{\tfrac12}\bigl(Y_4^{-2}+Y_4^2\bigr)
= \tfrac38\sqrt{\tfrac{5}{\pi}}\;\frac{(x^2-y^2)(7z^2-r^2)}{r^4},\\
\mathcal{Y}_{4,3}  &= \sqrt{\tfrac12}\bigl(Y_4^{-3}-Y_4^3\bigr)
= \tfrac34\sqrt{\tfrac{35}{2\pi}}\;\frac{x(x^2-3y^2)z}{r^4},\\
\mathcal{Y}_{4,4}  &= \sqrt{\tfrac12}\bigl(Y_4^{-4}+Y_4^4\bigr)
= \tfrac{3}{16}\sqrt{\tfrac{35}{\pi}}\;\frac{x^2(x^2-3y^2)-y^2(3x^2-y^2)}{r^4}.
\end{aligned}
```

### $l=5$ ($h$ orbitals)

```{math}
:label: eq-l5
\begin{aligned}
\mathcal{Y}_{5,-5} &= i\sqrt{\tfrac12}\bigl(Y_5^{-5}+Y_5^5\bigr)
= \tfrac{3}{16}\sqrt{\tfrac{77}{2\pi}}\;\frac{5x^4y-10x^2y^3+y^5}{r^5},\\
\mathcal{Y}_{5,-4} &= i\sqrt{\tfrac12}\bigl(Y_5^{-4}-Y_5^4\bigr)
= \tfrac{3}{16}\sqrt{\tfrac{385}{\pi}}\;\frac{4xyz(x^2-y^2)}{r^5},\\
\mathcal{Y}_{5,-3} &= i\sqrt{\tfrac12}\bigl(Y_5^{-3}+Y_5^3\bigr)
= \tfrac{1}{16}\sqrt{\tfrac{385}{2\pi}}\;\frac{-(y^3-3x^2y)(9z^2-r^2)}{r^5},\\
\mathcal{Y}_{5,-2} &= i\sqrt{\tfrac12}\bigl(Y_5^{-2}-Y_5^2\bigr)
= \tfrac{1}{8}\sqrt{\tfrac{1155}{\pi}}\;\frac{2xy(3z^3-zr^2)}{r^5},\\
\mathcal{Y}_{5,-1} &= i\sqrt{\tfrac12}\bigl(Y_5^{-1}+Y_5^1\bigr)
= \tfrac{1}{16}\sqrt{\tfrac{165}{\pi}}\;\frac{y(21z^4-14z^2r^2+r^4)}{r^5},\\
\mathcal{Y}_{5,0}  &= \;\;Y_5^0
= \tfrac{1}{16}\sqrt{\tfrac{11}{\pi}}\;\frac{63z^5-70z^3r^2+15zr^4}{r^5},\\
\mathcal{Y}_{5,1}  &= \sqrt{\tfrac12}\bigl(Y_5^{-1}-Y_5^1\bigr)
= \tfrac{1}{16}\sqrt{\tfrac{165}{\pi}}\;\frac{x(21z^4-14z^2r^2+r^4)}{r^5},\\
\mathcal{Y}_{5,2}  &= \sqrt{\tfrac12}\bigl(Y_5^{-2}+Y_5^2\bigr)
= \tfrac{1}{8}\sqrt{\tfrac{1155}{\pi}}\;\frac{(x^2-y^2)(3z^3-zr^2)}{r^5},\\
\mathcal{Y}_{5,3}  &= \sqrt{\tfrac12}\bigl(Y_5^{-3}-Y_5^3\bigr)
= \tfrac{1}{16}\sqrt{\tfrac{385}{2\pi}}\;\frac{(x^3-3xy^2)(9z^2-r^2)}{r^5},\\
\mathcal{Y}_{5,4}  &= \sqrt{\tfrac12}\bigl(Y_5^{-4}+Y_5^4\bigr)
= \tfrac{3}{16}\sqrt{\tfrac{385}{\pi}}\;\frac{x^2z(x^2-3y^2)-y^2z(3x^2-y^2)}{r^5},\\
\mathcal{Y}_{5,5}  &= \sqrt{\tfrac12}\bigl(Y_5^{-5}-Y_5^5\bigr)
= \tfrac{3}{16}\sqrt{\tfrac{77}{2\pi}}\;\frac{x^5-10x^3y^2+5xy^4}{r^5}.
\end{aligned}
```

### $l=6$ ($i$ orbitals)

```{math}
:label: eq-l6
\begin{aligned}
\mathcal{Y}_{6,-6} &= i\sqrt{\tfrac12}\bigl(Y_6^{-6}-Y_6^6\bigr)
= \tfrac{1}{64}\sqrt{\tfrac{6006}{\pi}}\;\frac{6x^5y-20x^3y^3+6xy^5}{r^6},\\
\mathcal{Y}_{6,-5} &= i\sqrt{\tfrac12}\bigl(Y_6^{-5}+Y_6^5\bigr)
= \tfrac{3}{32}\sqrt{\tfrac{2002}{\pi}}\;\frac{z(5x^4y-10x^2y^3+y^5)}{r^6},\\
\mathcal{Y}_{6,-4} &= i\sqrt{\tfrac12}\bigl(Y_6^{-4}-Y_6^4\bigr)
= \tfrac{3}{32}\sqrt{\tfrac{91}{\pi}}\;\frac{4xy(11z^2-r^2)(x^2-y^2)}{r^6},\\
\mathcal{Y}_{6,-3} &= i\sqrt{\tfrac12}\bigl(Y_6^{-3}+Y_6^3\bigr)
= \tfrac{1}{32}\sqrt{\tfrac{2730}{\pi}}\;\frac{-(11z^3-3zr^2)(y^3-3x^2y)}{r^6},\\
\mathcal{Y}_{6,-2} &= i\sqrt{\tfrac12}\bigl(Y_6^{-2}-Y_6^2\bigr)
= \tfrac{1}{64}\sqrt{\tfrac{2730}{\pi}}\;\frac{2xy(33z^4-18z^2r^2+r^4)}{r^6},\\
\mathcal{Y}_{6,-1} &= i\sqrt{\tfrac12}\bigl(Y_6^{-1}+Y_6^1\bigr)
= \tfrac{1}{16}\sqrt{\tfrac{273}{\pi}}\;\frac{y(33z^5-30z^3r^2+5zr^4)}{r^6},\\
\mathcal{Y}_{6,0}  &= \;\;Y_6^0
= \tfrac{1}{32}\sqrt{\tfrac{13}{\pi}}\;\frac{231z^6-315z^4r^2+105z^2r^4-5r^6}{r^6},\\
\mathcal{Y}_{6,1}  &= \sqrt{\tfrac12}\bigl(Y_6^{-1}-Y_6^1\bigr)
= \tfrac{1}{16}\sqrt{\tfrac{273}{\pi}}\;\frac{x(33z^5-30z^3r^2+5zr^4)}{r^6},\\
\mathcal{Y}_{6,2}  &= \sqrt{\tfrac12}\bigl(Y_6^{-2}+Y_6^2\bigr)
= \tfrac{1}{64}\sqrt{\tfrac{2730}{\pi}}\;\frac{(x^2-y^2)(33z^4-18z^2r^2+r^4)}{r^6},\\
\mathcal{Y}_{6,3}  &= \sqrt{\tfrac12}\bigl(Y_6^{-3}-Y_6^3\bigr)
= \tfrac{1}{32}\sqrt{\tfrac{2730}{\pi}}\;\frac{(11z^3-3zr^2)(x^3-3xy^2)}{r^6},\\
\mathcal{Y}_{6,4}  &= \sqrt{\tfrac12}\bigl(Y_6^{-4}+Y_6^4\bigr)
= \tfrac{3}{32}\sqrt{\tfrac{91}{\pi}}\;\frac{(11z^2-r^2)\bigl[x^2(x^2-3y^2)+y^2(y^2-3x^2)\bigr]}{r^6},\\
\mathcal{Y}_{6,5}  &= \sqrt{\tfrac12}\bigl(Y_6^{-5}-Y_6^5\bigr)
= \tfrac{3}{32}\sqrt{\tfrac{2002}{\pi}}\;\frac{z(x^5-10x^3y^2+5xy^4)}{r^6},\\
\mathcal{Y}_{6,6}  &= \sqrt{\tfrac12}\bigl(Y_6^{-6}+Y_6^6\bigr)
= \tfrac{1}{64}\sqrt{\tfrac{6006}{\pi}}\;\frac{x^6-15x^4y^2+15x^2y^4-y^6}{r^6}.
\end{aligned}
```

*$l\ge7$ harmonics are rarely needed in practice.*

## Periodic AO representations

Periodic AO objects supply determinant and AO-based Jastrow bases. `Structure_data` owns geometry and cell; a determinant AO object owns `k_point`. Determinant and Jastrow bases coexist as independent AO objects, and Jastrow coefficients are retained. Public API and CLI references give signatures, complete parameter lists, and examples.

### Supported representations

| Representation | Role | Main notes |
|---|---|---|
| pGTO | determinant and AO-based Jastrow | periodic contracted GTOs; real Gamma or complex Bloch; `num_ao` is home-cell count |
| Plane wave | determinant | retained inside `E_cut`; `num_ao` counts basis functions, not bands |
| B-spline | determinant | cubic or quintic; quintic default; `num_ao` is band count |
| Composite | determinant | `a + b`; order significant; children share one `Structure_data` |

pGTO details: Real single-k Gamma pGTO uses the real data classes with `k_point=None`; complex Bloch pGTO uses complex data classes with a fractional tuple. A Gamma twist in a multi-twist input keeps `(0,0,0)` and the complex form. Spin channels always use opposite twists, for restricted and LSDA inputs, untiled and tiled: the down-spin basis is the `with_negated_k()` partner (negated twist, conjugated image phases, negated plane-wave Miller indices or conjugated spline control points) of that channel's own `+k` basis -- the up basis for restricted input, beta's own basis for LSDA input, which need not equal alpha's. The pGTO truncation setting `AO_IMAGE_TRUNCATION_ATOL` (default `1e-7`) is not a global error bound; check overlap, gradient, and Laplacian convergence separately.

Plane waves: Gamma data are real; non-Gamma data are complex. Gamma-only QE half sets are expanded before projection; the Gamma converter emits real form. In cells with vacuum, conversion applies a taper automatically; there is no user option. Defaults are `PW_TAPER_VACUUM_RADIUS_BOHR` 10 bohr and `PW_TAPER_START_FRACTION` 0.8. Removed spectral weight is logged as `pw_taper_*` attributes. The taper can restore local kinetic energy where the physical tail exceeds the windowed remainder, but the G-set remains truncated and the basis must be converged as usual. Composite PW blocks are truncated at the residual orbital handoff cutoff and are not tapered.

B-spline: cubic and quintic orders are supported; quintic is default (`spline_order=5`, CLI `--spline-order 5`). Recommended `mesh_factor=0.75` (`jqmc-tool qe` default); control-point storage grows with its cube. The converter warns when per-axis headroom is below 1.25. Stored control points are interpolation coefficients, not orbital samples; knot agreement alone does not certify between-knot accuracy. Each stored spline band is one AO, so a pure B-spline MO matrix is the identity. Gamma control points are real; non-Gamma B-spline data are complex. Oblique cells require off-diagonal Laplacian terms. Converge VMC energy and variance with `mesh_factor`.

Composite: write `a + b`. The AO axis concatenates in order; `num_ao` is the sum; order is significant because MO columns and Jastrow AO matrices index by position. All children share one `Structure_data`. Molecular GTO objects may combine only with each other; periodic objects (pGTO, plane wave, B-spline, real or complex) may combine in any combination; mixing families is rejected. QE emits `pgto+pw` as `[pGTO, plane wave]` and `pgto+bspline` as `[pGTO, B-spline]`.

### Projection to hybrid representations

QE conversion supports `pw`, `pgto`, `bspline`, `pgto+pw`, and `pgto+bspline`. Projected coefficients and fidelity diagnostics are stored.

For PW -> pGTO + PW, each band has a Bloch pGTO block plus a low-`G` plane-wave block, so `pgto+pw` is bounded and not strictly lossless. At Gamma the PW block is real. `kind=pgto` omits the PW block. Important settings:

| Setting | Role | Default / CLI |
|---|---|---|
| `g_cut` | low-`G` handoff cutoff | `--ecutwfc` (Ry or `auto`); default `0.5*sqrt(2*ecutwfc)` of the QE run |
| `drop_diffuse` | automatic `alpha_min(g_cut)` | `True` |
| `alpha_min` | explicit exponent floor | `None`; `--alpha-min` overrides automatic |
| `svd_rcond` | orthogonalization floor | `1e-8`; `--max-cond-number` sets conditioning |
| `occ_only`, `occ_tol` | occupied bands and threshold | `True`, `1e-6` |
| `image_tol` | signature compatibility only | `1e-7` |

An omitted `--ecutwfc` sets `g_cut` at each twist to half of the QE plane-wave cutoff radius, `0.5*sqrt(2*ecutwfc)` (QE `ecutwfc` in Hartree). A numeric `--ecutwfc` (Ry) fixes `g_cut = sqrt(ecutwfc)` at every twist; `--ecutwfc auto` raises the cutoff per twist until the projection residual meets a fixed target. The cutoff, `--alpha-min` and `--exp-tail-eps` options are described in {doc}`../api_reference_cli_tool`. The projection residual is reported as a fit diagnostic; lower values indicate better fidelity.

For PW -> pGTO + B-spline, pGTO orbitals get a periodic B-spline correction. The per-band residual is a fit diagnostic for the combined orbital away from spline knots. Defaults: kernel mesh-factor 1.0, CLI `jqmc-tool qe` 0.75; `order` 5; `alpha_min` 0.0; `svd_rcond` 1e-8; `occ_only` True; `occ_tol` 1e-6; `image_tol` 1e-7 and should match emitted pGTO AOs. CLI maps `--mesh-factor`, `--spline-order`, `--alpha-min`, and `--max-cond-number`.

Convergence caution: the fit residual does not certify the spline grid. With a 300 Ry plane-wave cutoff, `mesh_factor=0.5` can keep the residual small while biasing kinetic energy by several mHa per twist; keep the default `0.75` and converge VMC energy and variance with `mesh_factor`.

### MO normalization and restrictions

All QE conversion kinds write unit-cell-normalized MOs. For `kind=pgto`, each twist-spin MO block is orthonormalized in the periodic AO metric of the emitted pGTO leaf; Gamma MOs stay real. Default `orthonormalize=True`; CLI `--orthonormalize/--no-orthonormalize`. `pgto+pw` and `pgto+bspline` are never orthonormalized: their metric requires the PW or spline block and cross overlap, so rescaling only the pGTO part would break the mixed orbital. Judge final orthonormality with an independently evaluated overlap, not the conversion residual alone.
