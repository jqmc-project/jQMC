(jax_compatible_modules_tags)=

# Using jQMC kernels with JAX transformations

## JAX transformations

The numerical kernels in `jqmc` operate on `jax.numpy` arrays and use
`flax.struct.dataclass` containers whose differentiable fields are registered
as PyTree leaves. Supported low-level kernels can therefore be composed with
JAX transformations such as `jax.jit`, `jax.vmap`, `jax.grad`, and
`jax.jacrev`. CLI, file-I/O, workflow, logging, and other host-side helpers are
not transformation targets, and discrete control fields remain static.

Production wavefunction optimization, force, and stress calculations use the
closed-form kernels in `jqmc.derivatives`. JAX automatic differentiation is
used only as a test reference for these kernels; the XLA and Rust backends
both run the analytic kernels ({doc}`analytic_derivatives`).

When differentiating a low-level function directly, keep its static arguments
fixed and use only the continuous PyTree leaves documented by that function.
Passing a whole `Hamiltonian_data` object does not imply that every one of its
fields is differentiable.

## Complex-valued differentiation

Complex inputs require an explicit distinction between Wirtinger derivatives,
JAX's complex VJP convention, and the Euclidean descent direction.  The
authoritative definitions, sign checks, and `holomorphic=True` contract are in
{ref}`sec:complex_gradients`.  In particular, for a real scalar output JAX
returns $2\,\partial f/\partial z$, and optimization takes a step along the
conjugate of that returned value.
