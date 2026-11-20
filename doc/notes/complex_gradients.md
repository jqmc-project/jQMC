(sec:complex_gradients)=

# Complex gradients and JAX conventions

This note fixes the notation used whenever a jQMC calculation differentiates
with respect to complex inputs.  It covers Wirtinger calculus and the value
returned by JAX.  Wave-function-specific objects such as the logarithmic
derivative, the SR metric, and the optimizer are defined separately in
{ref}`subsec:sr`.


## Wirtinger derivatives

Let $z=x+iy$.  For a function that is differentiable with respect to the two
real coordinates $x$ and $y$, define

$$
\frac{\partial}{\partial z}
= \frac{1}{2}\left(\frac{\partial}{\partial x}
-i\frac{\partial}{\partial y}\right),
\qquad
\frac{\partial}{\partial \bar z}
= \frac{1}{2}\left(\frac{\partial}{\partial x}
+i\frac{\partial}{\partial y}\right).
$$ (wirtinger_definitions)

The variables $z$ and $\bar z$ are treated as independent coordinates in
these derivatives.  For a general complex-valued function $h(z,\bar z)$,
both derivatives are needed:

$$
dh = \frac{\partial h}{\partial z}\,dz
   + \frac{\partial h}{\partial \bar z}\,d\bar z.
$$ (wirtinger_differential)

If $h$ is holomorphic, $\partial h/\partial\bar z=0$, and
$\partial h/\partial z$ is the ordinary complex derivative.

For a real-valued function $f:\mathbb{C}\to\mathbb{R}$,

$$
\frac{\partial f}{\partial\bar z}
= \overline{\frac{\partial f}{\partial z}},
$$ (real_wirtinger_conjugacy)

and the two real derivatives can be packed into the Euclidean gradient

$$
g_{\mathbb R}
\equiv \frac{\partial f}{\partial x}
+i\frac{\partial f}{\partial y}
=2\frac{\partial f}{\partial\bar z}.
$$ (complex_euclidean_gradient)

For a complex displacement $\delta z=\delta x+i\delta y$,

$$
df = \operatorname{Re}\!\left(g_{\mathbb R}^{*}\,\delta z\right).
$$ (complex_directional_derivative)

Consequently, $\delta z=-\eta g_{\mathbb R}$ is the steepest-descent step
for the ordinary Euclidean metric on $(x,y)$.  For vectors, the same equations
hold componentwise and the last expression becomes
$df=\operatorname{Re}(g_{\mathbb R}^{\dagger}\delta z)$.


## What `jax.grad` returns

For a real scalar output and a complex input, JAX returns

$$
g_{\mathrm{JAX}}
\equiv \operatorname{jax.grad}(f)(z)
= \frac{\partial f}{\partial x}
-i\frac{\partial f}{\partial y}
=2\frac{\partial f}{\partial z}
=\overline{g_{\mathbb R}}.
$$ (jax_complex_grad_convention)

This convention follows from JAX's complex VJP representation.  It is not the
packed Euclidean gradient in eq. {eq}`complex_euclidean_gradient`.  Therefore
optimization of a real loss uses

$$
z \leftarrow z-\eta\,\overline{\operatorname{jax.grad}(f)(z)}.
$$ (jax_complex_descent)

Omitting the conjugate reverses the sign of the imaginary-coordinate update.
This distinction disappears for real inputs, which is why a convention error
can remain hidden on the Gamma-point path.
The same conjugated-step rule is stated in the
[JAX complex differentiation documentation](https://docs.jax.dev/en/latest/complex-differentiation.html).

The following scalar identities pin the signs:

| $f(z)$ | `jax.grad(f)(z)` | packed Euclidean gradient $g_{\mathbb R}$ |
|---|---:|---:|
| $\lvert z\rvert^2$ | $2\bar z$ | $2z$ |
| $\operatorname{Re}z$ | $1$ | $1$ |
| $\operatorname{Im}z$ | $-i$ | $i$ |
| $\operatorname{Re}(a z)$ | $a$ | $\bar a$ |

These rows, together with the conjugated-descent rule, are the identities
to use when interpreting JAX gradients of complex parameters; the jQMC
optimizer applies them through its real-coordinate mapping.


## Holomorphic outputs and real parts

`jax.grad` normally requires a real scalar output.  If
$h:\mathbb C\to\mathbb C$ is known to be holomorphic, then

```python
jax.grad(h, holomorphic=True)(z)
```

returns the ordinary derivative $dh/dz$.  The flag is a promise made by the
caller; it does not test the Cauchy--Riemann equations and must not be used to
compress the full Jacobian of a non-holomorphic complex-output function.

There is a useful equivalent route for a holomorphic $h$.  Since
$f=\operatorname{Re}h$ is real-valued, the Cauchy--Riemann equations and
eq. {eq}`jax_complex_grad_convention` give

$$
\operatorname{jax.grad}(\operatorname{Re}h)(z)
=\frac{dh}{dz}.
$$ (jax_grad_real_holomorphic)

Thus `jax.grad(lambda z: real(log(Psi(z))))` returns the
**un-conjugated** holomorphic logarithmic derivative.  Any conjugation needed
by an estimator or metric must be visible in that consumer; it is not supplied
implicitly by this JAX call.  Tied or constrained parameterizations may make
the composed function non-holomorphic, in which case JAX differentiates the
actual real-coordinate computation and the relevant chain rule must be
specified by that parameterization.

For the corresponding wave-function storage and optimization conventions, see
{ref}`subsec:sr_log_derivative`; for force estimators, see {ref}`sec:derivatives`.
