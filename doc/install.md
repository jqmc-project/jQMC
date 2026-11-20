(installation_link)=

# Installation

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11, 3.12 or 3.13 (the tested versions; `requires-python >= 3.11`) |
| JAX | >= 0.7.0, <= 0.10.2 |
| SciPy | >= 1.12.0, excluding 1.16.x and 1.17.x (1.16.2-1.17.x bundle an OpenBLAS whose multi-threaded LU can deadlock; 1.16.0-1.16.1 fail the pGTO exponent optimizer's SLSQP) |
| MPI runtime and headers | OpenMPI 4.x or MPICH 4.x (required for `mpi4py`) |
| C compiler | required to build `mpi4py`, `h5py` and `trexio` when no binary wheel fits |
| HDF5 library | required to build `h5py` and `trexio` from source |
| Rust | current stable toolchain, required only for a source build |

jQMC contains a compiled Rust extension (`jqmc_rs`). A release wheel, where one
exists for your platform (see below), already contains it. Otherwise, and from a
git checkout, it is built on your machine during `pip install`. In that case
install the prerequisites **before** running `pip`:

* macOS: the Xcode Command Line Tools (`xcode-select --install`, provides the
  C compiler), then `brew install open-mpi hdf5` and Rust via
  [`rustup`](https://rustup.rs/).
* Debian/Ubuntu: `sudo apt install build-essential libopenmpi-dev libhdf5-dev`
  (or `libmpich-dev` instead of `libopenmpi-dev`), and Rust via
  [`rustup`](https://rustup.rs/).
* HPC systems: load the MPI, compiler and HDF5 modules of the site, and
  install Rust with `rustup` in your home directory.

## Install

**Release version (PyPI):**

This manual describes the pre-release v0.3.0a1. `pip install jqmc` without
a version installs the latest *stable* release instead, which lacks many
features described here (pip skips pre-releases unless asked). Install the
matching version explicitly:

```bash
pip install "jqmc==0.3.0a1"
```

(`pip install --pre jqmc` installs the newest pre-release, which may be newer
than this manual.)

Release workflows build wheels for Linux x86_64 and aarch64 (manylinux2014) and
macOS arm64 (11 or later), containing the compiled
`jqmc_rs` extension together with jQMC's Python packages. When a compatible
wheel is available for the selected release, installation does not require a
Rust compiler. The extension uses CPython's stable ABI; the package still
requires Python >=3.11. MPI and JAX retain their own runtime requirements.
Other platforms use the source distribution and need a Rust toolchain; no
wheels are provided for Intel macOS, musl Linux or Windows.

**Latest development version (GitHub):**

```bash
git clone https://github.com/jqmc-project/jQMC
cd jQMC
pip install .
```

To get the source tree that matches this manual (including the `examples/`
directory), check out the release tag instead:
`git clone --branch v0.3.0a1 https://github.com/jqmc-project/jQMC`.

Building from a git checkout compiles `jqmc_rs` with
`target-cpu=native` (set in the repository's `.cargo/config.toml`). This lets
Rust use the instruction set of the build host for the hot kernels, but it
also makes the resulting extension machine-specific. Build jQMC on the
machine (or homogeneous compute-node CPU family) where it will run. Do not
copy a locally built wheel, virtual environment, or `jqmc_rs` shared library
to an older or otherwise incompatible CPU; rebuild it there instead.

To build this documentation locally, install the documentation extra from a
checkout: `pip install ".[docs]"` (see `doc/README.md`).

Published wheels use baseline code generation for each platform (`x86-64`,
`generic` on aarch64, `apple-m1`) instead of `target-cpu=native` so they can run
on different CPU models. For a CPU-tuned
build on a compute node, use the source-build procedure above.

## Selecting the right JAX backend

Installing jQMC pulls in the **CPU build** of JAX by default. To run on GPU you must install the matching JAX wheel **first** (or as part of the same `pip install` invocation), so that the GPU-aware `jaxlib` is the one selected for jQMC's `jax >= 0.7.0, <= 0.10.2` constraint.

```{important}
Always pin the JAX version explicitly when targeting a GPU. The default
`jax[cuda12]` or `jax[rocm7-local]` extras resolve to the latest published
wheel, which may be outside jQMC's supported range. Replace `0.X.Y` below with
a concrete version in the range declared by the installed jQMC release.
```

### NVIDIA GPU (CUDA 12)

```bash
# Step 1 -- install the CUDA-aware JAX wheel (Linux x86_64 + CUDA 12)
pip install --upgrade "jax[cuda12]==0.X.Y"

# Step 2 -- install jQMC (it will reuse the JAX wheel from step 1)
pip install "jqmc==0.3.0a1"
```

After step 1, verify the backend is GPU:

```bash
python -c "import jax; print(jax.devices())"
# Expect: [CudaDevice(id=0), CudaDevice(id=1), ...]
```

### AMD GPU (ROCm 7)

```bash
# Step 1 -- after installing a compatible ROCm 7 runtime, install its JAX plugin
pip install --upgrade "jax[rocm7-local]==0.X.Y"

# Step 2 -- install jQMC
pip install "jqmc==0.3.0a1"
```

After step 1, verify the backend is GPU:

```bash
python -c "import jax; print(jax.devices())"
# Expect: [RocmDevice(id=0), ...]
```

### Apple Silicon (Metal)

JAX's Metal backend is experimental and not currently validated against jQMC. Use the CPU build (the default install produces this on Apple Silicon).

### Verifying the install

```bash
python -c "from importlib.metadata import version; print(version('jqmc'))"
jqmc --help
jqmc-tool --help
jqmc-jobmanager --help
python -c "import jqmc_rs; print(jqmc_rs.__file__)"
```

`jqmc` runs a calculation from a TOML input, `jqmc-tool` converts inputs and
analyzes results, and `jqmc-jobmanager` manages jobs of the workflow package
(see {doc}`notes/workflows`). The last command checks the native extension
used by default on CPU.
If all five commands succeed, the installation is complete. Continue with the
tutorials in {doc}`examples`; see {doc}`execution` for CPU Rust versus XLA,
MPI/thread configuration, full/mixed precision, and scientific convergence.

## Notes

* **CUDA-aware MPI is NOT required.** jQMC uses MPI for host-side (CPU) reductions only -- walker-binning sums, jackknife two-pass variance, and similar post-kernel aggregation on numpy buffers. Multi-GPU runs use one MPI process per GPU; communication between the GPUs of these processes is handled by JAX (initialized with `jax.distributed.initialize`, with collectives such as `jax.lax.psum` running over NCCL). So a plain (non-CUDA-aware) `mpi4py` wheel against OpenMPI / MPICH is sufficient even for multi-GPU runs.
* For a CPU-only install (laptops, CI workers without GPU), the plain `pip install` above is sufficient -- no platform-specific JAX wheel needed.
