# Contributing Guidelines

## Contribution Policy

Active development takes place in the private `jQMC-forge` repository. We do
not accept external code changes for direct merging in the public jQMC
repository. We nevertheless welcome and appreciate public pull requests as
proposals: a reproducer or suggested fix helps us understand the problem.

These pull requests are not merged as submitted, even if their CI checks pass.
The developers investigate the problem, implement and test the changes in
`jQMC-forge`, and then integrate the reviewed and validated changes into public
jQMC through maintainer-managed promotion pull requests. See also the
[README development policy](README.md#development).

Bug reports, feature requests, and suggestions are welcome through public
[GitHub Issues](https://github.com/jqmc-project/jQMC/issues) and
[GitHub Discussions](https://github.com/jqmc-project/jQMC/discussions). When
reporting a problem or proposing a fix, please include the observed behavior,
the expected behavior, and a minimal reproducer where possible. A complete
implementation is not required for a proposal. If you would like to take part
in development, please contact the developers directly.

The guidelines below apply to development in `jQMC-forge` and to the validated
changes promoted to public jQMC; they are not prerequisites for reporting a
problem or submitting an external proposal.

## Principles

In `jQMC`, our top priorities are:

1. **Sustainability** - maintainability of the codebase
2. **Ease of development** - simplicity of implementing new features or theories

We are willing to sacrifice some computational speed to achieve these goals. To that end, please follow these guidelines when contributing:

---

### Data and Algorithms

* `jQMC` is written in `Python` but uses a style that differs from conventional object-oriented programming (OOP).
* **Separation of concerns**:

  * **Data** are defined as static classes using `flax.struct.dataclass`.
  * **Algorithms** are implemented as standalone functions that accept these dataclass instances as arguments.
* Related dataclasses and their algorithms live in the same `Python` file and module, preserving the spirit of OOP while ensuring clarity.
* This design not only improves readability but also aligns with `JAX`'s requirement for side-effect-free functions.

---

### Testing

* Robust testing is central to `jQMC`. For every functionality, `jQMC` provides multiple implementations:

  1. A **`_debug`** version, written for human readability and easy tracing of the logic flow.
  2. A **production** version (no suffix), optimized and decorated with `@jit` for high performance -- using analytical derivatives where derivatives are involved (though its control flow may be less obvious).
  3. Where applicable, an **`_auto`** version, which is also `@jit`-optimized but relies on `JAX`'s automatic differentiation instead of analytical derivatives. This serves as an independent cross-check of the analytical implementation.
* In **`pytest`**, we verify that the `_debug`, production, and (when present) `_auto` variants produce numerically identical results.
* **When adding any new method**, you must implement both the `_debug` and production variants. If the method involves derivatives, an `_auto` variant should also be provided as a cross-check. Development pull requests in `jQMC-forge` lacking the required variants will not be approved.

#### Naming convention for production / debug variants

The suffix encodes which path a function implements.

* **production** -- no suffix (e.g. `compute_nodal_distance`, `compute_Jastrow_part`, `compute_AOs`).
* **`_debug`** -- leading-underscore, `_debug` suffix (e.g. `_compute_nodal_distance_debug`). Human-readable reference oracle.
* **`_fast`** -- a production variant that consumes precomputed running state (a running geminal inverse, accumulated streaming buffers) instead of recomputing it. Each `_fast` typically carries its own **`_fast_debug`** sibling that discards the running state and delegates to the non-fast / `_auto` reference, while the plain `_debug` recomputes from scratch. The `_fast`-vs-`_fast_debug` guardrail then asserts the Sherman-Morrison / streaming path reproduces the from-scratch reference (e.g. `compute_grads_and_laplacian_ln_Det_fast` vs `_compute_grads_and_laplacian_ln_Det_fast_debug`).
* **Exception -- AO grad/Laplacian** uses `analytic_*` (production) vs `autodiff` (debug) instead of the `_debug` suffix: the naming carries mathematical intent (closed-form derivative vs `JAX` autodiff) and is semantically equivalent, so it is **not** renamed to the `_debug` convention.

#### Running tests and registering new tests

* Run tests as described in `README.md` (How to run tests). Use `--precision-mode full` and `--precision-mode mixed`; each mode is judged within its own precision, and a mixed-precision result is never compared against a full-precision one.
* Tests under `tests/test_jqmc_kernel` always run the XLA path (`tests/conftest.py` disables Rust dispatch there). Tests that exercise `jqmc_rs` belong in `tests/test_rs_kernel` and need a `jqmc_rs` built from the current sources (`pip install -e .`).
* Files marked `mpi_only` need an MPI world: run them with `mpirun -np 2 python -m pytest <file>`.
* Every collected test carries exactly one of `light` / `heavy` and one of `with_chdir` / `without_chdir`. Mark each `pytest.param` row explicitly when rows differ in cost.
* Take tolerances from the central families in `jqmc/_setting.py` (or `jqmc._precision.get_tolerance_min`); inline tolerance literals in tests are rejected by a pre-commit hook.
* A test that is valid only in full precision uses `@pytest.mark.skip_mixed_precision(reason=..., alternative_guarantee_ref=...)`, naming the guarantee that covers the same capability in mixed precision. New skips need maintainer approval.
* A test that is valid only in mixed precision uses `@pytest.mark.skip_full_precision(reason=...)`. A test that is valid only under MPI uses `@pytest.mark.skip_serial(reason=...)`; one valid only in a single process uses `@pytest.mark.skip_mpi(reason=...)`. A test that needs a particular number of ranks checks it itself and fails on a mismatch. Outside its session a marked test is skipped.
* Every new test must be registered in the GDD SSoT (one file per guarantee, `gdd/ssot/guarantees/<module>/<ID>.yaml`; see `gdd/README.md`): add it as evidence of an existing guarantee or add a new guarantee. Then run `python scripts/gdd_sync.py` and `python scripts/gdd_sync.py --check`. The pre-commit hooks (orphan gate, fixture bindings, contract checks) reject unregistered tests and unapproved changes to protected contracts.
* Prefer extending an existing test (for example, one more `pytest.param` row) over adding a new test function.

---

### GitHub Actions

#### Public PR contribution-policy notice

`public-pr-policy-comment.yml` posts one English policy notice when a public
`jqmc-project/jQMC` PR is opened or reopened. It thanks the contributor, explains
that external PRs are references rather than directly merged changes, and links
to the [contribution policy](#contribution-policy) above. It does not close,
approve, or merge PRs and does not enforce merge restrictions.

#### Lint

The lint workflow (`public-lint-ruff.yml`) runs on pull requests and on pushes
to `main`, `rc` and `rc-gh-pages`. Its `ruff lint` job runs the `ruff`
and `ruff-format` pre-commit hooks on all files. Its `rust lint` job uses the
Rust toolchain pinned in the workflow (`RUST_TOOLCHAIN`) on the `jqmc_rs` crate
(`jqmc/rs`): `cargo fmt --check`, then
`cargo clippy --all-targets --locked -- -D warnings`, so the crate stays
rustfmt-formatted and free of rustc and clippy warnings; run both commands in
`jqmc/rs` before pushing Rust changes. The same job then runs
`python scripts/generate_third_party_licenses.py --check`, which fails when
`LICENSES/THIRD_PARTY_RUST_CRATES.txt` does not match `jqmc/rs/Cargo.lock`;
after a change of the Rust dependencies, regenerate the file with
`python scripts/generate_third_party_licenses.py`. The static GDD checks run in
the local pre-commit hooks (`python scripts/gdd_sync.py --check`).

#### Public package CI and deployment

The workflows of this repository are named `.github/workflows/public-*.yml`.
Every job is guarded by `github.repository == 'jqmc-project/jQMC'`, except the
build-only `public-build-distributions.yml`, which is also used to test
development snapshots before promotion. They use GitHub-hosted runners only.

The public distribution check (`public-distribution-check.yml`, check
`public-distribution-check`) runs on PRs and pushes to `main`, PRs
to `rc`, and manual dispatch. It is the final check before distribution and runs
no numerical tests. It builds an sdist and checks it with `twine check
--strict`, builds one Rust-enabled ABI3 wheel per platform FROM that sdist
(Linux x86_64, Linux aarch64, macOS arm64), validates each wheel against the
sdist and with `twine check`, then installs each wheel with its runtime
dependencies and MPI on Python 3.11, 3.12 and 3.13 with jax 0.7.2, pinned
near the bottom of the supported range (the package requires jax >= 0.7.0),
plus one Python 3.13 job per platform with the newest jax the package allows
(what a plain `pip install jqmc` resolves).

Each installation exercises the installed `jqmc`, `jqmc-tool` and
`jqmc-jobmanager` console scripts, generates and parses MCMC/VMC/LRDMC input
TOML, and reads and validates the bundled workflow templates; submission
scripts are syntax-checked only and no job is submitted. Every job also runs
`scripts/check_installed_jqmc_run.py`: it converts a small TREXIO file with the
installed `jqmc-tool`, runs short MCMC, VMC and LRDMC calculations with the
installed `jqmc` under `mpirun -np 2`, and requires "Program ends" and a finite
value in every `jqmc-tool` energy row. It checks that the shipped package runs
as a user runs it, not its numbers. Imports are isolated from the checkout.

`public-build-distributions.yml` is the single reusable build-and-check workflow
shared by the public distribution check, TestPyPI, PyPI and the wheel check of
`jQMC-forge`; it cannot publish. Deploy jobs consume only its
`public-distributions` artifact from the same workflow run, which is assembled
after every platform's wheel passed its checks.

| Public event | Result |
| --- | --- |
| Push to `rc` | Build, verify, publish to TestPyPI, verify the uploaded wheels |
| Push a canonical `v*` version tag whose commit is on `rc` | Build, verify, publish to PyPI, verify the uploaded wheels |
| Push to `rc-gh-pages` | Build current Sphinx docs and publish them to GitHub Pages |

After each upload, `public-verify-index.yml` downloads every platform's wheel
from the index it was uploaded to and requires it to be byte-identical (sha256)
to the tested file. On that platform it installs the wheel with its
dependencies from PyPI (never from TestPyPI) in a fresh environment and runs
`scripts/check_installed_jqmc_run.py`. Check the TestPyPI run before pushing a
release tag. A failure after the PyPI upload cannot undo the release, but it
reports at once that a published wheel does not work.

The PyPI/TestPyPI jobs use the `PYPI_JQMC_TOKEN` and `TEST_PYPI_JQMC_TOKEN`
secrets; no credentials are passed to the build or test jobs. Pages uses
branch-based publishing.
No deploy job runs for a PR or in forge.

Publishing runs are not cancelled by new pushes.


Binary wheels are built for the platforms that have jaxlib wheels: Linux
x86_64 and aarch64 (manylinux2014) and macOS arm64 (macOS 11 or later). Each is
one ABI3 wheel that includes `jqmc_rs` in the `jqmc` distribution (not a
separate Rust package), built from the published source distribution and
verified on its own platform with the checks above. The extension tag is
`abi3-py310`; package metadata requires Python >=3.11. The
source distribution, including all Rust build inputs, is also published; other
platforms build from it and need a Rust toolchain. jaxlib provides no macOS
x86_64, musllinux, ppc64le or s390x wheels, and Windows (MS-MPI, POSIX
submission templates) is untested, so there are no wheels for those. Release
builds override `target-cpu=native` with each platform's baseline (`x86-64`,
`generic` aarch64, `apple-m1`); local source builds keep native tuning. The wheel's portability tag covers the
extension, not JAX's independent system requirements. MPI remains a runtime
dependency, and users select JAX's CUDA extras separately for GPU operation.

---

### Examples and Tutorials

* Whenever you introduce a new computational feature, create a tutorial or example under the `examples/` directory to demonstrate its usage alongside your tests.

---

### Pull Requests and Integration

The following gates apply to developer changes, not to the external proposals
described in the [contribution policy](#contribution-policy).

* Developer changes need lint, the static GDD checks and maintainer review; the developer runs the new or changed tests locally.
* The `light` and `heavy` test markers classify each test's cost (see [Running tests and registering new tests](#running-tests-and-registering-new-tests)).
* Validated changes are brought into public jQMC through maintainer-managed promotion pull requests; the public pull request must also pass the required public CI checks and receive maintainer approval.
* Passing CI does not automatically merge a pull request.

---

### Dependent Modules

* To balance future maintainability with development efficiency, we strive to minimize external dependencies.
* The runtime dependencies are listed in `[project].dependencies` of `pyproject.toml`; the numerical core uses `numpy`, `scipy`, `jax` and `flax`. Test and development tools (`pytest`, `pytest-xdist`, `pre-commit`) are in the `dev` extra (`pip install -e ".[dev]"`).
* Other third-party packages should be avoided unless absolutely necessary. Any new dependency must be approved by @kousuke-nakano or another maintainer of @jqmc-project.

---

### Release Process

* All official package releases are performed by @kousuke-nakano or another maintainer of @jqmc-project as needed.

---

Thank you for considering contributing your work to the `jQMC` code!
