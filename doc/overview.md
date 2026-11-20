# Overview

**jQMC** is an *ab initio* quantum Monte Carlo (QMC) package implemented in
Python, JAX, and performance-critical Rust kernels. jQMC supports molecular
systems and periodic systems via plane-wave,
B-spline, and primitive-GTO atomic-orbital representations, and composite
representations that combine primitive GTOs with plane waves or B-splines. It implements Variational
Monte Carlo (VMC) and Lattice Regularized Diffusion Monte Carlo (LRDMC). JAX
just-in-time compilation and vectorized mapping support both CPU and GPU
execution.

```{admonition} v0.3.0a1 alpha -- stability disclaimer
:class: warning

This documentation describes **jQMC v0.3.0a1**, an *alpha* release. The public Python API and the CLI surface may still change in later 0.3 releases without backwards-compatibility shims. Pin the exact version in your environment when reproducing results.
```

(overview_concepts)=

## Basic concepts and workflow

A jQMC study usually follows these steps:

1. **Prepare a trial wavefunction.** A mean-field calculation (for example
   PySCF via TREXIO, or Quantum ESPRESSO) provides the orbitals;
   `jqmc-tool trexio convert-to` or `jqmc-tool qe` writes them, together with
   a Jastrow factor, to `hamiltonian_data.h5`.
2. **Optimize it with VMC** (`job_type = "vmc"`).
3. **Sample the optimized wavefunction with VMC** (`job_type = "mcmc"`) for the
   VMC energy, forces or stress.
4. **Project with LRDMC** (`job_type = "lrdmc-bra"` or `"lrdmc-tau"`) at several
   lattice spacings and extrapolate to zero spacing.

Each step is a `jqmc` run with a TOML input ({doc}`api_reference_cli`);
`jqmc-tool` prepares inputs and analyzes checkpoints
({doc}`api_reference_cli_tool`).

Terms used throughout this manual:

- **Trial wavefunction**: Jastrow factor times an antisymmetric part. **JSD**
  uses a Slater determinant, **JAGP** an antisymmetrized geminal power (a
  pairing function that can describe more correlation).
- **Jastrow factor**: a positive correlation factor $e^{J}$. **J1** (one-body)
  depends on electron-nucleus distances, **J2** (two-body) on electron-electron
  distances, **J3** (three-body) on electron-electron-nucleus terms expanded
  in a basis. **DTN** is an alternative polynomial Jastrow family (Drummond,
  Towler and Needs).
- **VMC (variational Monte Carlo)**: Metropolis sampling of
  $\lvert\Psi_T\rvert^2$; the average local energy is an upper bound of the
  ground-state energy. **Optimization** changes the wavefunction parameters
  to lower this energy (stochastic reconfiguration, SR, and its accelerated
  and linear-method variants).
- **LRDMC (lattice-regularized diffusion Monte Carlo)**: projects the trial
  wavefunction toward the ground state with the nodes fixed by the trial
  wavefunction (**fixed-node** approximation, **fixed-phase** for complex
  wavefunctions). The kinetic energy is discretized with lattice spacing
  `alat` (bohr), so results are extrapolated to `alat -> 0`.
- **Walker**: one electron configuration that is moved by the Monte Carlo
  algorithm. `number_of_walkers` walkers run on each MPI rank.
- **Measurement step, warm-up, bins**: observables are recorded once per
  measurement step; the first (warm-up) steps are discarded because the
  walkers are not yet equilibrated; the remaining steps are grouped into bins
  and the error bar is estimated from the bins by the jackknife method
  ({doc}`notes/error_analysis`).
- **ECP**: effective core potential (pseudopotential) that replaces the core
  electrons.
- **Twist averaging**: for periodic systems, averaging over several Bloch
  phases (k points, "twists") with their Brillouin-zone (BZ) weights to
  reduce one-body finite-size errors.
- **Units**: energies in hartree (Ha), lengths in bohr, forces in Ha/bohr.

## Development

Active development of jQMC, including the addition of new features, takes
place in the private `jQMC-forge` repository. If you would like to take part
in the development, please contact the developers directly. Feature requests
and suggestions for new functionality are welcome as
[GitHub Discussions](https://github.com/jqmc-project/jQMC/discussions) or
[GitHub Issues](https://github.com/jqmc-project/jQMC/issues) on the public
repository. See
[CONTRIBUTING.md](https://github.com/jqmc-project/jQMC/blob/main/CONTRIBUTING.md)
for how to report problems and propose changes.

![license](https://img.shields.io/github/license/jqmc-project/jQMC)
![tag](https://img.shields.io/github/v/tag/jqmc-project/jQMC)
![fork](https://img.shields.io/github/forks/jqmc-project/jQMC?style=social)
![stars](https://img.shields.io/github/stars/jqmc-project/jQMC?style=social)
![Public distribution check](https://github.com/jqmc-project/jQMC/actions/workflows/public-distribution-check.yml/badge.svg)
![Lint](https://github.com/jqmc-project/jQMC/actions/workflows/public-lint-ruff.yml/badge.svg)
[![Full Python coverage](https://raw.githubusercontent.com/jqmc-project/jQMC/ci-metadata/coverage.svg)](https://github.com/jqmc-project/jQMC/blob/ci-metadata/coverage-latest.json)
![DL](https://img.shields.io/pypi/dm/jqmc)
![python_version](https://img.shields.io/pypi/pyversions/jqmc)
![pypi_version](https://badge.fury.io/py/jqmc.svg)

The Full Python coverage badge reports the line coverage of `jqmc` and
`jqmc_workflow` measured for the latest source snapshot submitted to this
repository, possibly before that snapshot is merged. It is not Rust line
coverage or a measurement of every branch/release. Click the badge for the
tested commit and update date. Until the first snapshot is submitted, no
coverage value is published.
