# Technical notes

User-facing notes for **jQMC**, in suggested reading order. These pages describe
supported methods, input parameters, conventions, practical limitations and
examples. Detailed exposition of the new periodic methods is work in progress.
Previously published molecular methods and technical JAX differentiation
conventions remain documented in detail.

- {doc}`ao`: atomic-orbital basis sets, including the periodic plane-wave,
  B-spline, pGTO and composite representations and their conversion from
  Quantum ESPRESSO.
- {doc}`hamiltonian`: the kinetic, Coulomb and effective-core-potential
  terms of the Hamiltonian in molecular and periodic systems.
- {doc}`vmc`: variational Monte Carlo sampling and energy estimation.
- {doc}`lrdmc`: lattice-regularized diffusion Monte Carlo (fixed-node
  projection).
- {doc}`wavefunction`: the wave-function ansatz: Jastrow factors (including
  the Drummond--Towler--Needs polynomial family and its comoving cutoff,
  {ref}`sec:dtn_comoving_cutoff`), periodic determinants at the $\Gamma$
  point and at complex Bloch twists, and the AGP / AGPn / SD determinant
  parts.
- {doc}`wavefunction_optimization`: SR, accelerated SR and the linear
  method, how to choose the optimizer settings
  ({ref}`subsec:choosing_optimizer`), averaging the last optimization steps
  before sampling ({ref}`subsec:opt_step_averaging`), and the twist-averaged
  optimization energy ({ref}`subsec:twist_optimization_energy`).
- {doc}`derivatives`: energy derivatives, forces and stress.
- {doc}`complex_gradients`: the JAX/Wirtinger sign convention for complex
  parameters ({ref}`sec:complex_gradients`).
- {doc}`sd_occupied_orbitals`: how a single Slater determinant is stored and
  optimized ({ref}`sec:sd_occupied_orbitals`).
- {doc}`orbital_tiling`: representation-independent primitive k blocks for
  frozen supercell orbitals, required k-point coverage, normalization conventions and supported physical derivatives.
- {doc}`error_analysis`: binning and jackknife error bars.
- {doc}`mixed_precision`: running selected parts of the calculation in
  float32.
- {doc}`analytic_derivatives`: the analytic derivative backend (Rust and XLA).
- {doc}`jax`: using jQMC kernels with JAX transformations.
- {doc}`trexio`: reading and converting TREXIO files.

```{toctree}
:hidden:
ao
hamiltonian
vmc
lrdmc
wavefunction
wavefunction_optimization
derivatives
complex_gradients
sd_occupied_orbitals
orbital_tiling
error_analysis
mixed_precision
analytic_derivatives
jax
trexio
```
