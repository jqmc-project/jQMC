# jQMC

![jqmc_logo](https://raw.githubusercontent.com/jqmc-project/jQMC/main/logo/logo_yoko2.jpg)

**jQMC** is an ab initio quantum Monte Carlo (QMC) package for molecular
and periodic systems, implemented in Python, JAX, and performance-critical
Rust kernels. It provides Variational Monte Carlo (VMC) and Lattice
Regularized Diffusion Monte Carlo (LRDMC), including real and complex Bloch
wavefunctions, twist averaging, atomic forces, and stress for supported
configurations. JAX just-in-time compilation and vectorization target both
CPU and GPU execution. The user and developer manual is available from
[GitHub Pages](https://jqmc-project.github.io/jQMC/).

![license](https://img.shields.io/github/license/jqmc-project/jQMC)
![tag](https://img.shields.io/github/v/tag/jqmc-project/jQMC)
![fork](https://img.shields.io/github/forks/jqmc-project/jQMC?style=social)
![stars](https://img.shields.io/github/stars/jqmc-project/jQMC?style=social)
![Public distribution check](https://github.com/jqmc-project/jQMC/actions/workflows/public-distribution-check.yml/badge.svg)
![Lint](https://github.com/jqmc-project/jQMC/actions/workflows/public-lint-ruff.yml/badge.svg)
[![Docs](https://github.com/jqmc-project/jQMC/actions/workflows/public-deploy-gh-pages.yml/badge.svg?branch=rc-gh-pages)](https://jqmc-project.github.io/jQMC/)
[![Full Python coverage](https://raw.githubusercontent.com/jqmc-project/jQMC/ci-metadata/coverage.svg)](https://github.com/jqmc-project/jQMC/blob/ci-metadata/coverage-latest.json)
![DL](https://img.shields.io/pypi/dm/jqmc)
![python_version](https://img.shields.io/pypi/pyversions/jqmc)
![jax_version](https://img.shields.io/badge/JAX-0.7.0%20%E2%80%93%200.10.2-blue)
![pypi_version](https://badge.fury.io/py/jqmc.svg)

The Full Python coverage badge reports the line coverage of `jqmc` and
`jqmc_workflow` measured for the latest source snapshot submitted to this
repository, possibly before that snapshot is merged. It is not Rust line
coverage or a measurement of every branch/release. Click the badge for the
tested commit and update date. Until the first snapshot is submitted, no
coverage value is published.

What sets **jQMC** apart:

- It supports the standard Jastrow-Slater determinant (JSD) wavefunction and
  the resonating-valence-bond-type Jastrow Antisymmetrized Geminal Power
  (JAGP) wavefunction.
- It features a state-of-the-art optimization algorithm, stochastic reconfiguration, that enables stable optimization of both the amplitudes and nodal surfaces of many-body wave functions at the variational level.
- It implements the LRDMC method, providing a numerically stable approach to diffusion Monte Carlo calculations.
- Verified analytic derivative kernels provide wavefunction optimization,
  atomic forces, and stress; automatic differentiation is retained as an
  independent development oracle and explicit fallback mode.
- Calculations start seamlessly from [PySCF](https://github.com/pyscf/pyscf):
  molecular and periodic (`pyscf.pbc`) HF/DFT wavefunctions, including
  unrestricted and complex non-Gamma ones, are exported to TREXIO and turned
  into a jQMC trial wavefunction with a single `jqmc-tool trexio convert-to`
  command.
- Periodic calculations can be started directly from a
  [Quantum ESPRESSO](https://www.quantum-espresso.org) run and support
  Gaussian, primitive-periodic-GTO, plane-wave, B-spline, and composite
  one-particle representations, including non-Gamma complex twists and twist
  averaging where supported by the selected estimator.
- Written in `Python`, **jQMC** is designed to be user-friendly for executing simulations and easily extensible for developers implementing and testing new QMC methods.
- By leveraging `JAX` just-in-time (`jit`) compilation and vectorized mapping (`vmap`) functionalities, the code achieves high-performance computations **especially on GPUs** while remaining portable across CPUs and GPUs.
- MPI support enables the execution of large-scale computations on HPC facilities.
- **Automated workflows**: The `jqmc_workflow` package automates the entire simulation pipeline -- from pilot runs and step-count estimation through production runs and convergence monitoring -- allowing users to obtain publication-quality results with minimal manual intervention.
- To minimize bugs, the code is written in a loosely coupled manner and includes comprehensive unit tests and regression tests (managed by `pytest`).

This combination of features makes **jQMC** a versatile and powerful tool for both users and developers in the field of quantum Monte Carlo simulations.


## Known limitations

- Periodic cells must be charge neutral; a charged-cell correction is not
  currently implemented.
- For all-electron calculations with a one-body Jastrow (J1), the
  optimization of the J1 parameter is not yet stable: depending on its
  initial value it can stay frozen or overshoot. Pseudopotential (ECP)
  calculations, which need no J1, are the recommended route for now.
- MCMC with atomic forces (or stress) keeps the per-step, per-walker force
  (stress) values in memory and in the restart checkpoint, so both grow as
  steps x walkers x atoms: tens of GB for about 20 atoms, 64 walkers and
  10^6 steps. LRDMC keeps them averaged over walkers, so its growth is
  steps x atoms. Each continuation of either run copies the whole history
  and writes a full new checkpoint, so plan memory and disk space for long
  force runs.
- jQMC v0.3 is an alpha series. Pin the exact version used to produce
  scientific results because public APIs and input fields may still change.


## Development

Active development of jQMC, including the addition of new features, takes
place in the private `jQMC-forge` repository. External code contributions
are not merged directly through pull requests in the public jQMC repository.
We nevertheless welcome and appreciate pull requests as proposals: a
reproducer or suggested fix can help us understand a problem more clearly.

Such pull requests are not merged as submitted. Instead, the developers
investigate the problem, implement and test the changes in `jQMC-forge`,
and then integrate the reviewed and validated changes into the public jQMC
repository. Maintainer-managed pull requests that bring validated changes
from `jQMC-forge` into jQMC are part of this process.

If you would like to take part in development, please contact the developers
directly. Bug reports, feature requests, and suggestions are also welcome as
[GitHub Discussions](https://github.com/jqmc-project/jQMC/discussions) or
[GitHub Issues](https://github.com/jqmc-project/jQMC/issues) on this public
repository.


## Developer(s)
Kosuke Nakano (National Institute for Materials Science (NIMS), Japan)


## How to install jQMC

**The release version of jQMC** is available from `PyPI`. This README
describes the v0.3 alpha series. `pip install jqmc` without a version installs
the latest *stable* release instead (pip skips pre-releases unless asked), so
install the pre-release explicitly:

```bash
pip install "jqmc==0.3.0a1"
```

**The latest version of jQMC** can be installed via pip from the cloned GitHub repository.

```bash
git clone https://github.com/jqmc-project/jQMC
cd jQMC
pip install .
```

Building from source requires Python 3.11 or newer, an MPI implementation
with development headers, a C compiler, and a Rust toolchain. See the
[installation guide](https://jqmc-project.github.io/jQMC/install.html) for
CPU/GPU-specific JAX installation instructions.


## Examples

The complete, feature-indexed tutorial list is in
[`examples/README.md`](https://github.com/jqmc-project/jQMC/blob/main/examples/README.md).


## Supporting HF/DFT packages
`jQMC` can prepare a trial (guiding) wavefunction from a `TREXIO` file. Below is the list of HF/DFT packages that adopt `TREXIO` for writing wave functions:

- [Quantum Package](https://github.com/QuantumPackage/qp2)
- [PySCF](https://github.com/pyscf/pyscf)
- [FHI-aims](https://fhi-aims.org/)
- [CP2K](https://github.com/cp2k/cp2k)
- [Dirac](https://www.diracprogram.org)
- [pymolpro](https://molpro.github.io/pymolpro)

See the [TREXIO website](https://github.com/TREX-CoE/trexio) for the detail.

`jQMC` also reads the following package directly, without going through `TREXIO`:

- [Quantum ESPRESSO](https://www.quantum-espresso.org): `jqmc-tool qe` converts
  the `.save` directory of an HDF5-enabled `pw.x` run (single k point or a full
  twist grid) into a jQMC Hamiltonian.


## Documentation

**jQMC** user documentation is written using python sphinx. The source files are
stored in `doc` directory. Please see how to write the documentation at
`doc/README.md`.


## Branches

- `main`: main branch.
- `rc`: the latest stable version ready for package deployment.
- `rc-gh-pages`: the latest stable version ready for documentation deployment.

Every push to public `main` launches the public distribution check
(`public-distribution-check.yml`): the sdist and one Rust-enabled wheel per
platform are built, checked and installed, and the installed package runs a
short calculation. The public CI runs no numerical tests. The lint workflow
(`public-lint-ruff.yml`) runs separately on public PRs and on pushes to `main`,
`rc`, and `rc-gh-pages`: the ruff lint and format hooks,
`cargo fmt --check` and `cargo clippy` (warnings denied) on the `jqmc_rs`
crate, and the check that the third-party Rust licence list matches
`jqmc/rs/Cargo.lock`.

## How to deploy the package

Merging `main` into `rc` builds and checks the distributions with the same
distribution check, then deploys to TestPyPI. Tagging the resulting `rc`
commit with a valid canonical PEP 440 version
(for example, `v0.1.0b4`, `v0.1.1`, or `v1.0`) deploys the package to PyPI.

## How to deploy the documentation

Merging `main` into `rc-gh-pages` launches `public-deploy-gh-pages.yml`, which
builds the Sphinx manual and publishes it to
[GitHub Pages](https://jqmc-project.github.io/jQMC/).

## Contribution

Please see [CONTRIBUTING.md](https://github.com/jqmc-project/jQMC/blob/main/CONTRIBUTING.md) for contribution guidelines.


## Formatting

Formatting rules are written in `pyproject.toml`.


## Pre-commit

Pre-commit (https://pre-commit.com/) is mainly used for applying the formatting
rules automatically. Therefore, it is strongly encouraged to use it at or before
git-commit. Pre-commit is set-up and used in the following way:

- Installed by `pip install pre-commit`, `conda install pre_commit` or see
  https://pre-commit.com/#install.
- pre-commit hook is installed by `pre-commit install`.
- commit-msg hook (rejects non-ASCII bytes in commit messages) is
  installed by `pre-commit install --hook-type commit-msg`.
- pre-commit hook is run by `pre-commit run --all-files`.


## How to run tests

Install jQMC from source with `pip install -e ".[dev]"`; this also builds the
`jqmc_rs` Rust extension (a Rust toolchain is required) that the tests in
`tests/test_rs_kernel` exercise, and installs `pytest`, `pytest-xdist` and
`pre-commit`. Then:

```bash
pytest -n 8 -m light tests/                            # light tier, full precision
pytest -n 8 -m light --precision-mode mixed tests/     # the same tier, mixed precision
pytest -v tests/test_jqmc_kernel/test_jastrow.py       # one file, light and heavy
mpirun -np 2 python -m pytest tests/test_jqmc_kernel/test_checkpoint_restart_mpi.py  # MPI tests
```

Options: `--precision-mode {full,mixed}` (default `full`; each mode is judged
within its own precision), `--skip-heavy` (skip `heavy` tests) and
`--disable-jit`. Running every test, including `heavy`, takes many hours.

Every test carries exactly one cost marker (`light` or `heavy`) and one working
directory marker (`with_chdir` or `without_chdir`):

```python
import pytest


@pytest.mark.light
@pytest.mark.with_chdir
def test_something():
    ...
```

New tests must also be registered in the guarantee ledger of GDD
(guarantee-driven development, `gdd/README.md`), or the pre-commit hooks
reject them. See [CONTRIBUTING.md](https://github.com/jqmc-project/jQMC/blob/main/CONTRIBUTING.md) (Testing).

### Pseudopotentials in the test data

The pseudopotential data in the shipped test inputs come from two libraries:

- the [ccECP library](https://pseudopotentiallibrary.org): the ECP
  coefficients in the TREXIO files of `tests/trexio_example_files` that carry
  an ECP (H, Li, N, O, Si, Cl, Ti, Cu and Br) and in
  `tests/jqmc_hamiltonian_data/periodic_j3_basis_audit_initial.h5` (Li), and
  the `Li.ccECP.upf` and `O.ccECP.upf` files of the Li and O examples in
  `tests/qe_example_files`;
- the [Quantum ESPRESSO pseudopotential library](https://pseudopotentials.quantum-espresso.org):
  the all-electron `H.coulomb-ae.UPF` of the H examples in
  `tests/qe_example_files`.

The UPF files are not shipped. `tests/qe_example_files/pseudopotentials.toml`
lists each one with its download URL and checksum, and the tests download them
at run time.


## Citation of jQMC

If you used `jQMC` in your research project, please cite the following articles. This indeed helps the `jQMC` project to continue:

- "jQMC: A JAX-based ab initio quantum Monte Carlo package designed for GPU-accelerated computing",

  [K. Nakano and M. Casula, arXiv:2607.13781 (2026)](https://arxiv.org/abs/2607.13781)

  ```
  @article{nakano-jqmc,
      author  = {Nakano, Kousuke and Casula, Michele},
      title   = {jQMC: A JAX-based ab initio quantum Monte Carlo package designed for GPU-accelerated computing},
      journal = {arXiv preprint},
      eprint  = {2607.13781},
      archivePrefix = {arXiv},
      primaryClass  = {physics.chem-ph},
      year    = {2026},
      url     = {https://arxiv.org/abs/2607.13781}
  }
  ```

- "Load-Balanced Diffusion Monte Carlo Method with Lattice Regularization",

  [K. Nakano, S. Sorella, and M. Casula, J. Chem. Phys. 163, 194117 (2025)](https://doi.org/10.1063/5.0296986)

  ```
  @article{nakano-lrdmc,
      author = {Nakano, Kousuke and Sorella, Sandro and Casula, Michele},
      title = {Load-balanced diffusion Monte Carlo method with lattice regularization},
      journal = {J. Chem. Phys.},
      volume = {163},
      number = {19},
      pages = {194117},
      year = {2025},
      month = {11},
      doi = {10.1063/5.0296986}
  }
  ```

The methods implemented in `jQMC` are based on the works listed in the
[References](https://jqmc-project.github.io/jQMC/reference.html) page of the
manual; please also cite the original papers of the methods you use.
