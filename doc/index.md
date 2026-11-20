# Welcome to jQMC

**jQMC** implements two real-space *ab initio* quantum Monte Carlo (QMC)
methods for molecular and periodic systems: variational Monte Carlo (VMC) and
lattice-regularized diffusion Monte Carlo (LRDMC).

![jQMC logo](_static/logo_yoko2.jpg)

The following features of **jQMC** are highlighted:

- {ref}`Variational Monte Carlo: Wavefunction Optimization (VMC) <vmc_tags>`
- {ref}`Lattice Regularized Diffusion Monte Carlo (LRDMC) <lrdmc_tags>`
- {ref}`Periodic systems: Gamma-point and complex Bloch determinants, twist averaging <sec:periodic_determinants>`
- {ref}`Atomic force, pressure, and stress tensor calculations by VMC and LRDMC <sec:derivatives>`
- {ref}`JAX-compatible modules <jax_compatible_modules_tags>`
- {ref}`Interfaces with the TREXIO wavefunction format <trexio_tags>`
- {ref}`Autonomous pipelines (jqmc_workflow) <workflows_tags>`


```{toctree}
:hidden:
overview
install
execution
examples
notes/technical_notes
notes/workflows
gdd_audit/index
api_reference_cli
api_reference_cli_tool
api_reference
api_reference_mod/modules
citation
reference
changelog
```

For a first run, follow {doc}`install`, then choose a tutorial in
{doc}`examples`. Read {doc}`execution` before scaling to scientific runs.

New to quantum Monte Carlo? {ref}`overview_concepts` explains the order of
the calculation steps and the terms used in this manual.

## License

BSD-3-Clause license.

## Developer(s)

- Kosuke Nakano, National Institute for Materials Science (NIMS)

## Acknowledgements

**jQMC** development is supported by:

- National Institute for Materials Science (NIMS).
- Japan Science and Technology Agency (JST), PRESTO program.
- HANAMI (HPC Alliance for Applications and Supercomputing Innovation), the Europe-Japan collaboration.
