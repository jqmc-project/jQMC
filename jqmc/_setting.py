"""Setting module.

Contains physical constants, numerical stability parameters, and
test tolerance definitions used across jQMC.
"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from collections.abc import Callable, Iterable, Sequence

import jax
import numpy as np
import numpy.typing as npt


# Unit conversion
Bohr_to_Angstrom = 0.529177210903
Angstrom_to_Bohr = 1.0 / Bohr_to_Angstrom

# Default ECP integral parameters.  Nv=12 is the icosahedral spherical rule;
# it is the production default for MCMC, GFMC_t/GFMC_n, and their derivative
# routes.  Callers that need a lower-order diagnostic rule pass Nv explicitly.
Nv_default = 12
NN_default = 1

# =============================================================================
# Unified measured-budget streaming infrastructure.
#
# Every jit-side chunk width is derived from a byte budget, never from a fixed
# element count (a fixed count means 50 MB on one system and 5 GB on another):
#   width = measured_stream_batch(kernel, data, specs, n, BUDGET)
# which measures the per-element workspace with XLA's compile-time
# ``memory_analysis()`` (``measured_bytes_per_config``), picks a tentative width
# (``batch_size_for_budget``) and verifies it by compiling that width
# (``verified_stream_batch``). Probes run once per (kernel, shapes, backend) at
# first trace; startup-only cost, zero per-MC-step.
#
# There is NO fallback: if the measurement cannot be made (backend without
# ``memory_analysis``, or the probe compile fails), the helpers raise instead of
# silently switching to a different width, so a run never streams with a width
# nobody chose. ``memory_analysis()`` is available on CPU and GPU for every jax in
# the supported range (checked on jax 0.7.2 and 0.10.2).
#
# Two kinds of budget exist, and their semantics differ:
#   * PER-WALKER budgets (the axes INSIDE one walker's evaluation: electron pairs,
#     ECP quadrature configs, one-move mesh configs). These kernels run inside the
#     drivers' walker ``vmap``, so the live set is ~ budget x (walkers processed
#     at once). Changing them changes reduction grouping, i.e. results at fp64
#     round-off, so they are fixed constants (deterministic on every machine).
#   * The WALKER budget (``WALKER_STREAM_BUDGET_BYTES``): how many walkers one
#     device processes at once. Walkers are independent (no cross-walker
#     reduction), so this width does not change the mathematics; it bounds the
#     device memory of the whole batch. XLA may still select a different kernel
#     for a different batch shape, so equality is to fp64 round-off, not bitwise.
# Host-side setup budgets (PW_SPLINE_*, PW_PGTO_*, overlap tables) size host
# arrays from shapes and are outside this scheme.
# =============================================================================


def batch_size_for_budget(bytes_per_element: int, n_total: int, budget_bytes: int, min_batch: int) -> int:
    """Streaming batch width from a peak-byte budget with an occupancy floor.

    A streamed axis holds at
    most ``budget_bytes`` of per-element intermediates in flight, but never
    fewer than ``min_batch`` elements (GPU efficiency depends on the ABSOLUTE
    batch size, not the amplification ratio, so a unit amplification target is
    insufficient. ``bytes_per_element`` must be declared per site from the
    callee's dominant internal intermediates because input and output bytes alone
    under-count the working set. ``batch >= n_total``
    degenerates to the all-at-once vmap, the same contract as the fixed-batch
    knobs above.
    """
    if n_total <= 0:
        return 1
    batch = max(min_batch, int(budget_bytes // max(1, bytes_per_element)))
    if batch < n_total:
        # Round down to a power of two: divides typical walker counts evenly, so
        # ``lax.map`` avoids tracing a separate remainder chunk (halves compile
        # time and removes the ragged tail kernel). min_batch is itself a power
        # of two, so the floor survives the rounding.
        batch = 1 << (batch.bit_length() - 1)
    return min(n_total, max(min_batch, batch))


# There is neither an analytic model nor a fallback width: a
# fixture-calibrated pessimism factor is not an upper bound for unknown
# bases/backends, and a silent fallback hides the failure (see the header above).
# Memoized XLA-measured stream-kernel workspace, keyed by
# (kernel identity, data pytree structure + leaf avals, per-config arg
# shapes/dtypes, the ACTUAL jax.default_backend()) plus the probe batch size.
# The FIRST outer trace of a given shape pays the probe compiles; later traces
# (and later walkers/steps of the same shape) read the cache. Never cleared by
# ``jax.clear_caches()`` -- the measured bytes are a pure function of shapes and
# backend, independent of any budget global, so the cache stays valid across the
# budget-invariance tests that clear the JAX caches between runs.
_MEASURED_TOTAL_CACHE: dict = {}


def _measured_total_stream_bytes(
    kernel_fn: Callable[..., object],
    data: object,
    per_config_specs: Sequence[jax.ShapeDtypeStruct],
    batch: int,
    cache_identity: object = None,
) -> int:
    """XLA-compiled total (temp + output) bytes of ``vmap(kernel)`` at ``batch``.

    Lowers + compiles ``jax.jit(jax.vmap(kernel_fn))`` on ``ShapeDtypeStruct``
    args (the ``data`` pytree in axis ``None``, each per-config array batched in
    axis ``0``) and reads ``memory_analysis().temp_size_in_bytes`` +
    ``output_size_in_bytes``. Runs at TRACE time of an outer ``jit``: the probes
    are built from avals only (never tracer values), so the nested compile is
    pure host-side Python. Raises when the probe cannot be compiled or the
    backend reports no ``memory_analysis`` (see :func:`measured_bytes_per_config`).
    Memoized per (measurement family, batch). The family is keyed by the
    kernel's module and qualname unless ``cache_identity`` is given (callers that
    measure closure-converted kernels pass a digest of the kernel's jaxpr, since
    every such kernel shares one qualname).
    """
    import jax
    import jax.numpy as jnp

    # The cache identity must use the actual backend. A caller label could say
    # ``gpu`` while JAX compiles on CPU and would cache CPU measurements under
    # the wrong backend.
    leaves, treedef = jax.tree_util.tree_flatten(data)
    data_signature = tuple((tuple(jnp.shape(leaf)), jnp.result_type(leaf)) for leaf in leaves)
    per_config_signature = tuple((tuple(spec.shape), spec.dtype) for spec in per_config_specs)
    kernel_identity = (
        (getattr(kernel_fn, "__module__", ""), getattr(kernel_fn, "__qualname__", repr(kernel_fn)))
        if cache_identity is None
        else ("cache_identity", cache_identity)
    )
    measurement_family_key = (
        kernel_identity,
        treedef,
        data_signature,
        per_config_signature,
        jax.default_backend(),
    )
    key = (measurement_family_key, int(batch))
    cached = _MEASURED_TOTAL_CACHE.get(key)
    if cached is not None:
        return cached
    sds_data = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(jnp.shape(x), jnp.result_type(x)), data)
    vfn = jax.vmap(kernel_fn, in_axes=(None,) + (0,) * len(per_config_specs))
    batched = tuple(jax.ShapeDtypeStruct((int(batch),) + tuple(s.shape), s.dtype) for s in per_config_specs)
    comp = jax.jit(vfn).lower(sds_data, *batched).compile()
    ma = comp.memory_analysis()
    if ma is None:
        raise RuntimeError(f"jax.stages.Compiled.memory_analysis() returned None on backend {jax.default_backend()!r}")
    total = int(ma.temp_size_in_bytes) + int(ma.output_size_in_bytes)
    _MEASURED_TOTAL_CACHE[key] = total
    return total


def measured_bytes_per_config(
    kernel_fn: Callable[..., object],
    data: object,
    per_config_specs: Sequence[jax.ShapeDtypeStruct],
    cache_identity: object = None,
) -> int:
    r"""XLA-measured bytes of evaluator-internal workspace per streamed config.

    Hand-modeling every evaluator's internal temporaries (periodic
    images, primitives, B-spline, J3 orbital bases, DTN atom/polynomial
    workspace) keeps failing -- the hand-built sum under-counts the PERIODIC AO
    evaluator by up to 62.27x (silicon: 76,720 declared vs 4,776,984 measured).
    XLA's own memory analysis is therefore used as the estimator.

    Measurement: probe totals at batch 1 and 2 (see
    :func:`_measured_total_stream_bytes`); the per-config ESTIMATE is

        bpc = max(total(2) - total(1), total(1)),

    the supremum of ``total(n)/n`` over ``n >= 1`` for an affine
    ``total(n) = a + b n``. The affine fit is not a
    general upper bound -- XLA may switch algorithm/workspace across batch
    thresholds (GEMM/LU tiling, padding, autotuning) -- so this value picks the
    TENTATIVE batch only; the batch actually selected must then be verified by
    compiling it (:func:`verified_stream_batch`), which halves on excess.

    Args:
        kernel_fn: Pure callable ``(data, *per_config_arrays) -> Array`` with a
            stable module-level identity (used in the cache key). Do NOT pass a
            lambda closing over outer-trace tracers.
        data: The pytree threaded in vmap axis ``None``; only leaf avals are read.
        per_config_specs: ``ShapeDtypeStruct`` sequence for ONE config.
        cache_identity: Optional memoisation identity replacing the kernel's
            module/qualname (see :func:`_measured_total_stream_bytes`).

    Returns:
        Per-config workspace bytes (>= 1).

    Raises:
        RuntimeError: the probe could not be compiled or measured. There is no
            fallback width (see the module header): the error names the kernel.
    """
    try:
        total1 = _measured_total_stream_bytes(kernel_fn, data, per_config_specs, 1, cache_identity)
        total2 = _measured_total_stream_bytes(kernel_fn, data, per_config_specs, 2, cache_identity)
    except Exception as exc:
        name = f"{getattr(kernel_fn, '__module__', '')}.{getattr(kernel_fn, '__qualname__', repr(kernel_fn))}"
        raise RuntimeError(f"cannot measure the stream workspace of {name}: {exc!r}") from exc
    return max(1, int(max(total2 - total1, total1)))


def verified_stream_batch(
    kernel_fn: Callable[..., object],
    data: object,
    per_config_specs: Sequence[jax.ShapeDtypeStruct],
    tentative_batch: int,
    n_total: int,
    budget_bytes: int,
    min_batch: int,
    cache_identity: object = None,
) -> int:
    """Verify the selected batch by compiling it; halve until it fits.

    ``measured_bytes_per_config``'s batch-1/2 affine estimate picks a tentative
    batch, but only a compile of the batch actually chosen proves the XLA
    workspace at that batch fits the budget (algorithm/workspace can change
    across batch thresholds). Probes ``min(tentative, n_total)`` -- the shape
    production will really run -- and halves while the compiled total exceeds
    ``budget_bytes``, flooring at ``min_batch`` (the occupancy floor keeps
    memory-safety subordinate to it exactly as ``batch_size_for_budget`` does).
    The degenerate contract is preserved: when ``tentative >= n_total`` and the
    monolithic compile fits, the tentative value is returned unchanged so the
    caller's ``batch >= n`` plain-vmap branch still triggers. A failing probe
    raises (no fallback width; see the module header). Probe totals are memoized, and the production compile of the same shape hits the
    same XLA lowering, so the extra cost is bounded by a few cached compiles
    per (kernel, shape, backend).
    """
    n_total = int(n_total)
    tentative = int(tentative_batch)
    if n_total <= 0:
        return tentative
    batch = max(1, min(tentative, n_total))
    while True:
        try:
            total = _measured_total_stream_bytes(kernel_fn, data, per_config_specs, batch, cache_identity)
        except Exception as exc:
            name = f"{getattr(kernel_fn, '__module__', '')}.{getattr(kernel_fn, '__qualname__', repr(kernel_fn))}"
            raise RuntimeError(f"cannot verify stream width {batch} of {name}: {exc!r}") from exc
        if total <= int(budget_bytes) or batch <= int(min_batch):
            break
        batch = max(int(min_batch), batch // 2)
    if batch == min(tentative, n_total) and tentative >= n_total:
        return tentative
    return batch


# Occupancy floor shared by the budget-derived streaming sites: the
# B-spline sweep measured batch 64 -> 1.06x time but batch 16 -> 1.23x and
# below collapses; 16 is the floor where degradation stays acceptable.
STREAM_MIN_BATCH = 16


def measured_stream_batch(
    kernel_fn: Callable[..., object],
    data: object,
    per_config_specs: Sequence[jax.ShapeDtypeStruct],
    n_total: int,
    budget_bytes: int,
    min_batch: int = STREAM_MIN_BATCH,
    cache_identity: object = None,
) -> int:
    """Width of a budget-streamed ``jax.lax.map`` over ``n_total`` elements.

    The single implementation of the policy in the module header: measure the
    per-element workspace (:func:`measured_bytes_per_config`), pick a tentative
    width (:func:`batch_size_for_budget`), verify it by compiling it
    (:func:`verified_stream_batch`). Raises when the workspace cannot be measured.

    Degenerate contract (``width >= n_total`` == "hold all elements at once, no
    streaming"): the returned width is then ``n_total + 1``, strictly above the
    axis, so ``jax.lax.map(f, xs, batch_size=width)`` takes the plain-``vmap``
    remainder path with no ``scan`` primitive (``lax.map`` skips ``scan`` only
    when ``batch_size > len(xs)``; a scan-of-one hangs the jax-0.7.0 CPU lowering
    after many prior compiles). A streaming width (``< n_total``) is returned
    unchanged.
    """
    n_total = int(n_total)
    if n_total <= 0:
        return 1
    bpc = measured_bytes_per_config(kernel_fn, data, per_config_specs, cache_identity)
    tentative = batch_size_for_budget(bpc, n_total, budget_bytes, min_batch)
    batch = verified_stream_batch(
        kernel_fn, data, per_config_specs, tentative, n_total, budget_bytes, min_batch, cache_identity
    )
    return batch if batch < n_total else n_total + 1


def measured_closure_stream_batch(
    fn: Callable[..., object],
    one_element_args: tuple,
    n_total: int,
    budget_bytes: int,
    min_batch: int = STREAM_MIN_BATCH,
) -> int:
    """:func:`measured_stream_batch` for a closure ``fn(*element_args)``.

    The streamed callees of the derivative meshes and of the walker axis close
    over traced values (the Hamiltonian, per-walker state), which the measurement
    probes cannot lower directly. ``jax.closure_convert`` hoists those values into
    explicit arguments; the converted kernel is measured on their avals, and the
    memoisation identity is a digest of its jaxpr (all converted kernels share
    one qualname). ``one_element_args`` is one element's slice of the mapped
    arguments (pytrees allowed).
    """
    import hashlib

    import jax.numpy as jnp

    converted, consts = jax.closure_convert(fn, *one_element_args)
    flat_args, args_tree = jax.tree_util.tree_flatten(one_element_args)

    def kernel(data: tuple, *flat: jax.Array) -> object:
        return converted(*jax.tree_util.tree_unflatten(args_tree, list(flat)), *data)

    data = tuple(consts)
    specs = tuple(jax.ShapeDtypeStruct(jnp.shape(a), jnp.result_type(a)) for a in flat_args)
    data_specs = tuple(jax.ShapeDtypeStruct(jnp.shape(c), jnp.result_type(c)) for c in data)
    jaxpr_text = str(jax.make_jaxpr(kernel)(data_specs, *specs))
    identity = hashlib.sha256(jaxpr_text.encode()).hexdigest()
    return measured_stream_batch(kernel, data, specs, n_total, budget_bytes, min_batch, identity)


def budget_map(
    f: Callable[..., object],
    xs: tuple,
    budget_bytes: int,
    min_batch: int = 1,
) -> object:
    """``vmap(lambda x: f(*x))(xs)`` in measured-budget chunks of the leading axis.

    ``xs`` is a tuple of arrays/pytrees sharing the leading (mapped) axis; ``f``
    receives one element's slices and may close over traced values. The chunk
    width is :func:`measured_closure_stream_batch` against ``budget_bytes``; when
    every element fits, this is the plain ``vmap``. Otherwise an explicit
    ``lax.scan`` of ``vmap`` over padded chunks (padding repeats element 0 and is
    sliced off), not ``jax.lax.map(batch_size=...)``, whose collect step reshapes
    with a ``-1`` axis and raises ``ZeroDivisionError`` on zero-size output leaves
    (e.g. an empty optional parameter block of an O_k pytree). Each element's math
    is independent of the chunking.
    """
    import jax.numpy as jnp

    g = lambda x: f(*x)  # noqa: E731
    n = int(jax.tree_util.tree_leaves(xs)[0].shape[0])
    one = jax.tree_util.tree_map(lambda a: a[0], xs)
    b = measured_closure_stream_batch(g, (one,), n, budget_bytes, min_batch)
    if n <= b:
        return jax.vmap(g)(xs)
    n_pad = (-n) % b
    n_tot = n + n_pad
    n_ch = n_tot // b

    def _pad(a: jax.Array) -> jax.Array:
        if n_pad == 0:
            return a
        return jnp.concatenate([a, jnp.broadcast_to(a[:1], (n_pad,) + a.shape[1:])], axis=0)

    xs_r = jax.tree_util.tree_map(lambda a: _pad(a).reshape((n_ch, b) + a.shape[1:]), xs)
    _, ys = jax.lax.scan(lambda carry, chx: (carry, jax.vmap(g)(chx)), None, xs_r)
    return jax.tree_util.tree_map(lambda y: y.reshape((n_tot,) + y.shape[2:])[:n], ys)


def walker_map(f: Callable[..., object], *walker_args: object) -> object:
    """Map a per-walker kernel over the walker axis within ``WALKER_STREAM_BUDGET_BYTES``.

    ``f(*one_walker_args)`` may close over walker-independent traced values (the
    Hamiltonian, the lattice spacing). Equivalent to ``vmap(f)(*walker_args)``;
    walkers are processed in chunks only when the measured per-walker workspace
    times the walker count exceeds the device budget.
    """
    return budget_map(f, tuple(walker_args), WALKER_STREAM_BUDGET_BYTES)


# Device budget for the WALKER axis (see the module header): the drivers'
# per-walker kernels (MCMC electron-update sweep, analytic force / stress /
# parameter-derivative evaluation) are mapped over the walkers in widths of
# ``measured_stream_batch(kernel, ham, per_walker_specs, nw,
# WALKER_STREAM_BUDGET_BYTES, min_batch=1)``, so the transient workspace of one
# device stays within this budget at any walker count; when all walkers fit,
# the width exceeds nw and the map is the plain all-walker ``vmap``. Sized for
# the smallest supported GPU (A30, 24 GB) with room left for the persistent
# walker state and tables (B-spline coefficients, carried inverses).
WALKER_STREAM_BUDGET_BYTES = 8 * 1024**3

# Per-walker budget of the generic derivative mesh streaming
# (``derivatives.mesh_stream._mesh_map`` / ``_mesh_wsum``: ECP quadrature,
# GFMC S' lattice and O_k mesh reductions). Per-walker semantics as for the
# DTN budgets below; the walker axis is bounded by WALKER_STREAM_BUDGET_BYTES.
DERIVATIVE_MESH_BUDGET_BYTES = 1024**3

# DTN same-spin/anti-parallel pair axis (the streamed jastrow_factor sites and
# the closed-form DTN derivative kernels of ``derivatives.dtn``). SEMANTICS: this
# bounds ONE WALKER's chunk -- the probe measures the per-walker element workspace and the
# outer walker vmap multiplies the live set by nw afterward, so total device
# spend ~ budget x nw. Calibrated on the silicon+DTN N_e=432 / f-channel
# n_pair=93096 reference (probe bpc ~12 KB/pair): 8 MiB picks ~512 pairs, the
# measured optimum (28x memory for ~11% wall; 64 MiB landed 8x too high).
JASTROW_DTN_PAIR_BUDGET_BYTES = 8 * 1024**2

# DTN ECP quadrature-grid axis (the jastrow_factor sites). Per-walker semantics as for the pair budget above.
# Calibrated on the G=4096 reference (probe bpc 22-36 KB/config): 16 MiB picks
# ~256 configs, the measured working point (13.7x memory for 1.9x wall; the
# fixed count 64 was 5.1x slower, and 256 MiB never streamed at all).
JASTROW_DTN_ECP_GRID_BUDGET_BYTES = 16 * 1024**2

# DTN one-move MESH axis (the LRDMC kinetic lattice / ECP quadrature configs of
# ``derivatives.dtn.dtn_drift_mesh_dense`` and the DTN force / parameter meshes of
# ``derivatives.one_move_mesh``). Per-walker semantics as for the two budgets
# above -- and that is why these kernels do not use the generic
# DERIVATIVE_MESH_BUDGET_BYTES (1 GiB): they run INSIDE the walker
# vmap, so 1 GiB per walker is 16 GiB at NW=16 and OOMs a 24 GiB A30 (MEASURED
# on the calibration system: 16.2 GB single allocation at NW=16, and the
# unchunked form asked 19.2 GB at NW=8). Calibrated on that system (declared bpc
# ~3.1 MB/config from the 24 live ``(N_atom, N_e)`` f64 tensors): 256 MiB picks
# ~86 configs, which lands NW=16 near 4 GB with the config-independent legs
# hoisted out of the chunk loop.
JASTROW_DTN_ONE_MOVE_MESH_BUDGET_BYTES = 256 * 1024**2

# Peak-byte budget for the batched SWCT weight contraction
# (``swct.evaluate_swct_omega_dot_batched``, used by the driver force blocks).
# Per batch element the fused kernel holds one ``(N_e, N_a)`` kappa block --
# 373 KiB at N_e = 432, N_a = 108, trivial on its own. The problem
# is the BATCH: the chunked force block presents ``B = clen * nw`` elements at
# once, so the intermediate reaches ~9.6 GiB and OOMs (measured: both a
# materialise-then-host-einsum form AND the fused form die at B = 16384 on a
# 24 GB A30). Streaming B holds ``batch`` elements instead of all of them.
# Round-off only (the per-element result is independent of the grouping; only
# the batch's internal reduction order can shift).
SWCT_OMEGA_BATCH_BUDGET_BYTES = 1024**3

# Minimal warmup steps (VMC)
MCMC_MIN_WARMUP_STEPS = 30
MCMC_MIN_BIN_BLOCKS = 10

# Minimal warmup steps (GFMC)
GFMC_MIN_WARMUP_STEPS = 25
GFMC_MIN_BIN_BLOCKS = 10
GFMC_MIN_COLLECT_STEPS = 5

# Minimal total MCMC / GFMC steps for a get_E / get_aF run to satisfy the
# binning constraint ``(mcmc_counter - warmup) >= bin_blocks`` enforced in
# ``MCMC.get_E`` / ``GFMC_*.get_E`` (raises ValueError otherwise). For GFMC
# ``mcmc_counter = steps - collect_steps``, so the collect window is added.
# Derived from the floors above so the three stay mutually consistent if any
# floor is retuned -- never hard-code a bare step count in the tests.
MCMC_MIN_STEPS = MCMC_MIN_WARMUP_STEPS + MCMC_MIN_BIN_BLOCKS
GFMC_MIN_STEPS = GFMC_MIN_WARMUP_STEPS + GFMC_MIN_BIN_BLOCKS + GFMC_MIN_COLLECT_STEPS

# on the fly statistics param
GFMC_ON_THE_FLY_COLLECT_STEPS = 10
GFMC_ON_THE_FLY_BIN_BLOCKS = 10

# Analytic (autograd-free) gradient backend for the PRODUCTION MCMC / GFMC observable
# gradients (O_k = d ln|Psi|/dc, dE_L/dc, forces de_L/dR, stress de_L/deps), routed through
# the unified ``jqmc.derivatives`` interface with ``analytic=True``. DEFAULT TRUE: the analytic
# backend is the PRODUCTION DEFAULT. Set to False to revert production to the ``_auto`` autograd
# path. The ``_MCMC_debug`` / ``_GFMC_*_debug`` drivers use the SAME selected derivative
# backend while retaining their independent eager per-walker loops. Production-vs-debug parity
# therefore checks driver structure, not analytic-vs-autograd correctness; the dedicated
# ``jqmc.derivatives`` comparison matrix supplies the AD oracle.
ANALYTIC_GRAD_ENABLED = True

# Native (Rust) production dispatch on CPU.
# DEFAULT TRUE: on CPU, in both full and mixed precision, production MCMC / GFMC hot
# loops route through the ``jqmc_rs`` native kernels instead of the JAX vmap
# kernels. The dispatch is resolved INLINE at each driver's run head as
#   active = (env JQMC_FLAG_RUST_ON_CPU if set else flag_rust_on_CPU)
#            AND jax.default_backend() == "cpu"
#            AND import jqmc_rs succeeds
# GPU or an explicit disabled flag stay on JAX (active=False, no error). If the
# flag and CPU conditions hold but ``import jqmc_rs`` fails the driver raises RuntimeError with a
# build instruction; it never silently falls back to JAX. The
# ``_MCMC_debug`` / ``_GFMC_*_debug`` oracle drivers are UNAFFECTED (they stay
# on JAX), so the prod-vs-debug equivalence tests are the integration check.
# Native input packing may widen fp32/c64 storage without restoring discarded
# digits. The current zone policy is separately passed to the native context;
# fp64 transport alone does not select full-precision arithmetic.
flag_rust_on_CPU: bool = True

# --- Environment-variable ledger -----------------------------------------
#
# Every ``os.environ`` / ``std::env::var`` override read anywhere in
# ``jqmc/`` or ``jqmc/rs/src/`` (Python driver + Rust native extension),
# collected here so none can go undocumented. Every production override is
# listed alongside its precedence and numeric effect.
#
#   JQMC_FLAG_RUST_ON_CPU  -- {0,1}. Overrides ``flag_rust_on_CPU`` above
#       (documented in full at its definition). Python driver.
#   GFMC_T_OBS_FRESH_INV   -- {unset,"1"/truthy}. GFMC_t chunk observable
#       path: default reuses the per-step Sherman-Morrison ``G^-1`` already
#       computed by the sweep (no fresh per-step SVD-pinv); set to force a
#       fresh pinv recompute instead. Changes fp-level numerics (not just
#       performance). Read in ``jqmc_gfmc.py`` (``_obs_use_fresh_inv``).
#       Python driver, debug/perf lever.
#   JQMC_RS_ECPF_SLOW      -- {unset,present}. Rust GFMC/LRDMC ECP-force
#       and VMC periodic-stress kernels: default uses cached Jastrow mesh
#       candidates; any present value selects recomputation instead. Read in
#       ``jqmc/rs/src/gfmc_forces.rs``. Rust, debug/perf A-B lever (also
#       used as a test-only oracle switch in ``tests/test_rs_kernel``).
#   JQMC_RS_PROFILE        -- {unset,"1"/truthy}. Enables the Rust hot-path
#       phase profiler (``jqmc/rs/src/prof.rs``); prints ``timed!`` scope
#       timings via ``_prof_dump``. Rust, perf-debug only, no numeric effect.
#   JQMC_REQUIRE_MESH_CACHE -- {unset,"1"/truthy}. Makes any projection-scoped
#       mesh candidate cache that compiles its UNCACHED route raise
#       ``MeshCacheMissing`` at trace time instead of silently running slower.
#       Read in ``jqmc/_route_probe.py``. Python driver, test/diagnostic gate;
#       no numeric effect (both branches return the same values -- that is
#       precisely why the flag is needed). Off by default: production must not
#       refuse to run because a cache is missing.
#   JQMC_RS_NUM_THREADS    -- {unset,positive int}. Overrides the rayon global
#       thread-pool width for the Rust walker-batch harness.
#       Precedence inside ``jqmc_rs.configure_threads(n)`` is explicit ``n`` >
#       this env > rayon's affinity-aware default (``available_parallelism``,
#       which on Linux honors the cgroup / CPU-affinity mask so ``mpirun
#       --bind-to socket`` derives per-rank threads automatically). Read in
#       ``jqmc/rs/src/lib.rs`` (``configure_threads``); production-effective
#       because both drivers call ``configure_threads(None)`` at Rust-path
#       activation (jqmc_mcmc.py / jqmc_gfmc.py, right after the strict-raise
#       import). Rust, perf/affinity lever,
#       no numeric effect (per-walker threefry => thread-schedule-independent
#       results); the Rust kernel tests pin it to 4 in
#       ``tests/test_rs_kernel/conftest.py``.
#   JQMC_ECP_OK_FORCE_REBUILD -- {unset,any}. Rust ECP param-grad rebuild
#       guard: forces the fresh-rebuild path instead of the cheaper
#       incremental-update candidate check. Read in
#       ``jqmc/rs/src/param_grad_ecp.rs``. TEST-ONLY hook (set via
#       ``monkeypatch.setenv`` in ``tests/test_rs_kernel/test_param_grad_el.py``
#       to force-exercise the rebuild path); not intended for production use.

# Native-normal statistical sanity bounds (``test_chain_modes`` normal test /
# the rs-level ``native_normal_moments`` cargo test). Over N ~ 1e6 i.i.d. draws
# the first four sample moments of a true N(0,1) sit within ~a few sampling-error
# bars of (0, 1, 0, 0): std(mean)=1/sqrt(N)~1e-3, std(var)=sqrt(2/N)~1.4e-3,
# std(skew)=sqrt(6/N)~2.4e-3, std(exkurt)=sqrt(24/N)~4.9e-3. Bounds set at ~6-16
# sampling sigma: loose enough to never flake on a correct generator, tight
# enough that a broken transform (wrong scale, uniform-not-normal, sign bug)
# fails macroscopically. The native erfinv's ~1e-7 accuracy shifts these moments
# by << the bounds.
RS_NORMAL_SANITY_N = 1_000_000
RS_NORMAL_MEAN_ATOL = 1.0e-2
RS_NORMAL_VAR_ATOL = 1.5e-2
RS_NORMAL_SKEW_ATOL = 3.0e-2
RS_NORMAL_EXKURT_ATOL = 8.0e-2
# Kolmogorov-Smirnov statistic vs the standard-normal CDF over N ~ 1e6 draws.
# Critical value ~ c(alpha)/sqrt(N): at alpha=1e-3, c~1.95 -> ~2.0e-3. Bound set
# at ~2.5x critical: a correct generator never flakes; a wrong distribution
# (e.g. a residual uniform component) gives KS ~ 0.1+, orders above this.
RS_NORMAL_KS_ATOL = 5.0e-3

# Test tolerance settings (gradients, laplacians)
atol_auto_vs_numerical_deriv = 1.0e-1
rtol_auto_vs_numerical_deriv = 1.0e-4
atol_numerical_vs_analytic_deriv = 1.0e-1
rtol_numerical_vs_analytic_deriv = 1.0e-4
# auto (autograd of the forward) vs analytic (closed-form) GRADIENT agreement. These two
# routes evaluate the SAME mathematical quantity by completely different fp64 operation
# sequences, so they are never bit-identical; the only legitimate difference is fp64
# accumulation. Crucially this comparison is fp64-ONLY (the c-axis analytic-vs-_auto assert
# SKIPS mixed precision -- _auto differentiates the fp32 forward there), so a precision-ZONE
# tolerance (whose purpose is absorbing fp32/mixed error) is the WRONG instrument: it admits
# a systematic ~1e-6 relative derivation error. Calibrated from the measured
# analytic-vs-_auto discrepancy over all 858 matrix (cell, quantity, param-block) c-axis
# comparisons on committed fixtures: p50=1.5e-16, p90=2.9e-15, p99=5.1e-14, worst
# well-conditioned cell ~1.3e-13 (plane-wave / composite Det/mo, the most cancellation-heavy).
# atol/rtol set above that floor so a real systematic error (>= ~1e-6 rel) fails loudly while
# legitimate fp64 route differences pass. The ATOL floor is set by the WORST-CONDITIONED case,
# not the median: the open-shell (N_cart_J) down-spin ``mo_coefficients_dn`` e_L c-grad has a
# small-VALUE element (~7.5e-4 -- the sparsely-occupied down channel gives near-decoupled MO
# gradients) where rtol*|value| ~ 7.5e-14 is negligible, so the comparison is entirely
# atol-bound. There the analytic-vs-_auto ABSOLUTE agreement is the plain fp64 route floor
# (~5e-13 locally) but PLATFORM-DEPENDENT: CI runners accumulate up to ~4.8e-12 on that element
# (10x a local workstation). This is NOT a derivation bug -- analytic matches _auto (the exact autograd
# derivative) to that absolute floor; a real error would diverge macroscopically (a periodic-ECP
# derivation error is ~1e-3, 9 orders larger). atol=2e-11 covers the platform-worst fp64 floor with ~4x margin
# and is still 5 orders below any real derivation error (a 1e-6 rel error on that element = ~7.5e-10
# abs, 40x atol). rtol stays 1e-10 (governs the well-conditioned O(1)-valued blocks).
atol_auto_vs_analytic_deriv = 2.0e-11
rtol_auto_vs_analytic_deriv = 1.0e-10

# Test tolerance settings (others)
atol_debug_vs_production = 1.0e-8
rtol_debug_vs_production = 1.0e-6
atol_consistency = 1.0e-8
rtol_consistency = 1.0e-6

# --- Test tolerance dict (dtype-aware) ---
#
# Accessed via ``_precision.get_tolerance(zone, level)`` which resolves the
# zone's current dtype and returns ``(atol, rtol)``.
#
# Levels:
#   strict   -- two exact implementations of the same quantity (debug vs
#               production, analytic vs autodiff).  Difference is pure
#               floating-point round-off.
#   medium   -- analytic / autodiff vs 4th-order central finite differences
#               of a Laplacian (or higher-derivative-sensitive quantity),
#               *or* analytic vs autodiff of a quantity whose autodiff path
#               inherits a fp32 forward zone (e.g. ao_eval) and therefore
#               carries fp32 grad/lap noise the analytic path does not see.
#               4th-order FD has a round-off floor of ~eps_mach / h^2 with
#               h = 1e-3 (~5e-9) and a truncation term ~h^4 * f^(6) that can
#               grow O(1e-6) for sharp basis sets / cusp regions (e.g.
#               all-electron Z >= 7).  Use for skip_mixed_precision tests where
#               the FD side is a 2nd derivative.
_TOLERANCE: dict[str, dict[str, tuple[float, float]]] = {
    "strict": {"float64": (1e-8, 1e-6), "float32": (1e-6, 2e-4)},
    # Keep the fp32/fp64 scale identical to strict: atol x100, rtol x200.
    "medium": {"float64": (1e-7, 1e-5), "float32": (1e-5, 2e-3)},
}


# --- Mixed-versus-full numerical-fidelity tolerance ---
#
# This family answers a different question from ``_TOLERANCE``.  It compares
# one production quantity evaluated through the same route in mixed and full
# precision; it must not be used for analytical-vs-autodiff, XLA-vs-Rust, or
# other same-precision implementation comparisons.
#
# Orbital and parameter derivatives may cross fp32 forward zones before their
# fp64 analytical assembly. The reviewed fixtures contain products and
# reductions from those rounded values. Eight float32 eps provide the O(1)
# absolute floor and 32 float32 eps bound their block-scale propagation. The
# comparison is made once per homogeneous logical block against
# ``max(abs(full_block))``; a cancelling element therefore cannot tighten the
# budget by itself.
ATOL_MIXED_FIDELITY_PARAMETER_DERIVATIVE = 9.5367431640625e-7  # 8 * eps(float32)
RTOL_MIXED_FIDELITY_PARAMETER_DERIVATIVE = 3.814697265625e-6  # 32 * eps(float32)
ATOL_MIXED_FIDELITY_ORBITAL_DERIVATIVE = 9.5367431640625e-7  # 8 * eps(float32)
RTOL_MIXED_FIDELITY_ORBITAL_DERIVATIVE = 3.814697265625e-6  # 32 * eps(float32)
_MIXED_FIDELITY_TOLERANCE: dict[str, tuple[float, float]] = {
    "orbital_derivative": (
        ATOL_MIXED_FIDELITY_ORBITAL_DERIVATIVE,
        RTOL_MIXED_FIDELITY_ORBITAL_DERIVATIVE,
    ),
    "parameter_derivative": (
        ATOL_MIXED_FIDELITY_PARAMETER_DERIVATIVE,
        RTOL_MIXED_FIDELITY_PARAMETER_DERIVATIVE,
    ),
}


def get_tolerance_mixed_fidelity(quantity_class: str) -> tuple[float, float]:
    """Return the mixed-versus-full fidelity budget for a quantity class.

    The returned ``(atol, rtol)`` is applied to one homogeneous logical block
    as ``max(abs(mixed - full)) <= atol + rtol * max(abs(full))``.

    Args:
        quantity_class: Centrally reviewed class of the compared quantity.

    Returns:
        The absolute floor and block-relative mixed-fidelity budget.

    Raises:
        ValueError: If the quantity class has no reviewed fidelity model.
    """
    try:
        return _MIXED_FIDELITY_TOLERANCE[quantity_class]
    except KeyError as exc:
        raise ValueError(f"unknown mixed-fidelity quantity class: {quantity_class!r}") from exc


# --- XLA-vs-Rust implementation-consistency tolerance ---
#
# This table is deliberately separate from ``_TOLERANCE``. The latter answers
# whether a mixed-precision calculation remains numerically acceptable after
# errors have propagated through a complete physical calculation. It is much
# too loose to establish that two backend implementations execute the same
# zone-owned arithmetic: with ``_TOLERANCE`` an all-f64 Rust kernel compared
# with an fp32 XLA zone at rtol=1e-3 (strict) or 1e-2 (medium) would pass.
#
# XLA and Rust need not be bit-identical because reduction order, FMA
# contraction, conditioning, and library implementations may differ. Backend
# parity normally has one semantic level: two implementations of the same
# quantity. ``medium`` is a temporary, explicitly selected exception for
# conditionally amplified force/stress and parameter-derivative aggregates,
# plus deterministic MCMC/GFMC end-to-end state/observable parity. It does not
# widen fp64, and it must not be used by ordinary leaf-kernel/value parity.
# Temporary aggregate envelope for mixed backend parity.  The absolute floor
# retains force/stress sensitivity near zero; the relative side matches the
# existing mixed-precision propagation budget.  Full precision still resolves
# to the strict pair below.
ATOL_XLA_RUST_CONSISTENCY_MEDIUM = 1.0e-4
RTOL_XLA_RUST_CONSISTENCY_MEDIUM = 2.0e-3
# ``independent_fp32``: Rust and XLA evaluate the same fp32 zone with separate
# fp32 implementations (issue #679, DTN grad/lap parity). The strict float32
# pair is the envelope of ONE fp32 route against the exact value. Two separate
# routes each carry their own fp32 error of that size, and the two errors are
# independent and can have opposite signs. Measured at the TestDTNBroadParity
# 4u1d en-coalescence gradient element (fp64 value 0.19917939): Rust32 -3.95e-7,
# XLA32 +3.35e-7. Their difference is therefore bounded by the sum of the two
# single-route envelopes, twice the strict pair. Full precision resolves to the
# strict fp64 pair; this level does not widen fp64.
ATOL_XLA_RUST_INDEPENDENT_FP32 = 4.0e-7
RTOL_XLA_RUST_INDEPENDENT_FP32 = 4.0e-6

# --- Polynomial-evaluation conditioning cap ---
#
# A test that evaluates a polynomial-type quantity S(x) = sum_i t_i(x) in a
# zone dtype with unit roundoff u = eps / 2 has the classical forward-error
# bound |fl(S) - S| <= gamma_n * sum_i |t_i(x)|, so its relative error is at
# most kappa(x) * n * u with the evaluation condition number
#
#     kappa(x) = sum_i |t_i(x)| / |S(x)|  (>= 1).
#
# For the real solid harmonic S_lm = N * Lambda_lm(r, z) * A_m(x, y) of degree
# l, as summed by the debug oracle, each term carries the input cast to the
# zone dtype (<= l u), r = sqrt(x^2 + y^2 + z^2) raised to a power <= l
# (<= 2.25 l u), <= l further products and one coefficient rounding (<= (l + 1) u),
# and the two sums plus the final product add <= l + 2 roundings: in total at most
# (5.25 l + 3) u = (2.625 l + 1.5) eps <= (4 l + 4) eps for every l >= 0.
# Requiring kappa(x) * c_l * eps <= rtol with c_l = 4 l + 4 therefore keeps
# the guaranteed evaluation error of an accepted point inside the unchanged
# comparison rtol of the same zone/level. The cap is mode-dependent only through
# that zone's rtol and eps; full and mixed use the same rule.
EVALUATION_KAPPA_ROUNDING_SLOPE = 4
EVALUATION_KAPPA_ROUNDING_OFFSET = 4


def evaluation_kappa_max(rtol: float, eps: float, degree: int) -> float:
    """Return the largest accepted evaluation condition number for one degree.

    ``kappa_max = rtol / ((EVALUATION_KAPPA_ROUNDING_SLOPE * degree +
    EVALUATION_KAPPA_ROUNDING_OFFSET) * eps)``, i.e. the cap for which the
    documented forward-error bound ``kappa * c_l * eps`` equals ``rtol``.

    Args:
        rtol: Relative tolerance of the comparison the accepted points feed.
        eps: Machine epsilon of the dtype the quantity is evaluated in.
        degree: Total polynomial degree ``l`` (a nonnegative integer).

    Returns:
        The cap as a Python float.
    """
    if isinstance(degree, bool) or int(degree) != degree or degree < 0:
        raise ValueError("evaluation-kappa degree must be a nonnegative integer")
    if not (rtol > 0.0 and eps > 0.0):
        raise ValueError("evaluation-kappa cap requires positive rtol and eps")
    return float(rtol) / ((EVALUATION_KAPPA_ROUNDING_SLOPE * int(degree) + EVALUATION_KAPPA_ROUNDING_OFFSET) * float(eps))


# --- Same-precision reduction-order tolerance ---
#
# For two routes that execute the SAME arithmetic in the SAME precision and
# differ only in summation/reduction order (sparse atom-centric list vs dense
# all-pairs, fused mesh vs per-config stream), the achievable agreement is
# machine-epsilon class, NOT the propagation budget of a physical zone.
# Measured on the DTN sparse-vs-dense battery (f-channel value/grad/Laplacian,
# molecular + PBC, three seeds): max |delta| = 5.3e-15, max relative = 5.8e-16.
# The zone tolerances (jastrow_grad_lap strict 1e-8/1e-6, medium 1e-7/1e-5)
# sit 7-10 orders above that, so a 5e-6 Laplacian regression passes them under
# mutation. These values keep ~3 orders of headroom over the
# measured spread while catching that mutation by four orders.
ATOL_REDUCTION_ORDER = 1.0e-12
RTOL_REDUCTION_ORDER = 1.0e-10

# --- Rust fast-math approximations vs libm (jqmc/rs/src/fastmath.rs) ---
#
# Accuracy of the branch-free fp64 kernels the FAST Rust routes use in place of
# libm: ``erfc_scalar`` (Chebyshev fit of erfcx, degree 26; measured relative
# error below 5e-15 on z in [0, 6.5]) and ``exp_neg`` (measured within 1 ulp,
# relative error below 2.3e-16), and the Ewald real-space image sum
# ``ewald_image_sum`` built from them (seven O(1) terms). Read by the Rust unit
# tests through ``include_str!`` (jqmc/rs/src/lib.rs ``central_setting_f64``).
RTOL_FASTMATH_ERFC_VS_LIBM = 1.0e-14
RTOL_FASTMATH_EXP_VS_LIBM = 1.0e-15
# fp32 exp / sin-cos cores of the mixed ao_eval zone (fastmath::exp_f32, sincos_f32) versus
# libm's fp32 functions: 4 fp32 eps relative (exp) and 8 fp32 eps absolute (sin/cos, |x| <= 400).
RTOL_FASTMATH_F32_VS_LIBM = 4.76837158203125e-7  # 4 * eps(float32)
ATOL_FASTMATH_F32_SINCOS_VS_LIBM = 9.5367431640625e-7  # 8 * eps(float32)
ATOL_FASTMATH_EWALD_IMAGE_SUM_VS_LIBM = 1.0e-13

# --- FAST periodic J1/J2 value route (jqmc/rs/src/jastrow_factor.rs) ---
#
# The FAST native energy chunk (MCMC sweep ``ao_cutoff`` and ECP-mesh
# ``cutoff``) evaluates the periodic J1/J2 ratio with ``fastmath::sincos``
# (Cody-Waite + cephes minimax, |error| <= 2e-15 absolute versus libm on the
# PW test range) and ``fastmath::exp_neg`` instead of libm ``sin_cos`` /
# ``exp_m1``. Only the elementary functions and the summation order change.
# Distance: the smooth periodic distance is built from h_i = sin cos and
# z_i = sin^2 of three reduced angles, d^2 = |sum_i h_i a_i / pi|^2 + ...,
# so a per-angle error eps_sc propagates to |delta d| <~ 3 eps_sc max|a_i| / pi
# (~1e-14 for eps_sc = 2e-15 and the 5-6 bohr test cell); measured
# max |d_fast - d_exact| = 8.9e-16 bohr on 4000 displacements of a skewed
# 5 x 4.5 x 6 bohr cell (one quarter in the coalescence branch).
# Ratio: ln R is a sum of O(N_partner + N_atom) terms whose distance slope is
# bounded by |w| b (J1 prefactor ~ (2 Z_eff)^0.75 / (2a)); measured
# max |R_fast / R_exact - 1| = 3.1e-15 over 936 ratios of the 13-electron /
# 3-atom fixture (ECP-mesh and single-move twins, exp/pade J1 and J2).
# The bounds keep 110x and 320x headroom over these measurements and sit far
# below the statistical floor of the FAST chunk.
ATOL_FASTMATH_PERIODIC_SMOOTH_DIST = 1.0e-13
RTOL_FASTMATH_PERIODIC_J12_RATIO = 1.0e-12

# --- Coherent value-scale bias of a zone AO value route (issue #145) ---
#
# A value route that rounds each primitive product once to its zone dtype has
# per-point errors of either sign; their projection on the value itself,
#     kappa = sum_n v_ref (v - v_ref) / sum_n v_ref^2,
# averages to ~0 (statistical floor ~ relrms / sqrt(N_points), plus the small
# bias of the zone's exp implementation). A route that rounds an
# electron-independent factor (N_n, N_lm, c, or the exponent Z inside
# exp(-Z r^2)) to the zone dtype SEPARATELY carries that rounding into every
# value of the primitive: a coherent scale error of up to ~1 zone eps. Measured
# on the periodic Gamma sphe-pGTO fixture (pbc_gamma_sphepgto_j12_pade, s AOs,
# 4096 points, mixed): 0.9 eps for a route that rounds those factors separately,
# <= 0.1 eps for the XLA and Rust production routes. The fast-update kinetic energy pairs G^-1 of these values with
# the fp64 grad/lap, so a coherent bias shifts e_L systematically.
# The bound is a fraction of the machine epsilon of the zone dtype of the active
# mode plus a float64 allowance for coherent float64 effects that are not zone
# rounding: the association of the storage-precision constant factor
# (N_n N_lm (c e) vs (N_n N_lm c) e) and the polynomial fp64 exp of the Rust
# kernels. Measured in full precision on the same fixture: XLA <= 1 eps64,
# Rust <= 5.5 eps64. Full and mixed use the same rule; the float64 allowance
# (16 eps64 = 3.6e-15) is negligible against the mixed bound (3.0e-8).
AO_VALUE_COHERENT_BIAS_EPS_FRACTION = 0.25
AO_VALUE_COHERENT_BIAS_FP64_EPS_MULTIPLE = 16
# The statistic must be resolved: its standard error relrms / sqrt(N_points) is
# admitted up to this fraction of the bound, so sampling noise cannot decide it.
AO_VALUE_COHERENT_BIAS_STANDARD_ERROR_FRACTION = 0.25


def ao_value_coherent_bias_bounds(zone_eps: float) -> tuple[float, float]:
    """Return ``(kappa_bound, standard_error_bound)`` of the coherent AO value-scale check.

    ``kappa_bound = AO_VALUE_COHERENT_BIAS_EPS_FRACTION * zone_eps
    + AO_VALUE_COHERENT_BIAS_FP64_EPS_MULTIPLE * eps(float64)`` and
    ``standard_error_bound = AO_VALUE_COHERENT_BIAS_STANDARD_ERROR_FRACTION * kappa_bound``.

    Args:
        zone_eps: Machine epsilon of the zone dtype the value route evaluates in.

    Returns:
        The two bounds as Python floats.
    """
    if not zone_eps > 0.0:
        raise ValueError("coherent-bias bound requires a positive zone epsilon")
    kappa_bound = AO_VALUE_COHERENT_BIAS_EPS_FRACTION * float(zone_eps) + AO_VALUE_COHERENT_BIAS_FP64_EPS_MULTIPLE * float(
        np.finfo(np.float64).eps
    )
    return kappa_bound, AO_VALUE_COHERENT_BIAS_STANDARD_ERROR_FRACTION * kappa_bound


# --- Coherent value-scale bias of the periodic Cartesian pGTO value routes (issue #718) ---
#
# Same statistic kappa as above, judged on the Cartesian twin of the spherical
# fixture (lithium_cart.h5, bcc Li ccECP, s AOs, 32768 points uniform in the
# cell) against an fp64 textbook lattice sum, with two changes:
#
# * Resolution. The standard error is the weighted (delta-method) error of the
#   ratio estimator, sqrt(sum (v_ref e - kappa v_ref^2)^2) / sum v_ref^2, and
#   N_eff = (sum v_ref^2)^2 / sum v_ref^4 must reach a floor. The relrms /
#   sqrt(N) proxy above assumes near-uniform weights; for an all-electron 1s
#   core (Li_pbc_ae_ccpvdz_cart, 4096 points) N_eff is 14 and the true standard
#   error is ~0.12 float32 eps, ten times the proxy, so a seed-0 kappa of
#   +0.31 eps there was a sampling fluctuation (pooled over 32 seeds: +0.235).
#   On lithium_cart N_eff is ~1.2e3 for the tighter s shell and ~3.3e4 for the
#   diffuse one; the weighted standard error (0.011 / 0.004 eps) matches the
#   measured spread over 16 seeds (0.008-0.013 / 0.003-0.005 eps). The floor of
#   100 effective points keeps the normal approximation behind the standard
#   error meaningful and rejects core-dominated samples like the one above.
# * Bound 0.1 eps(ao_eval) (issue #718). Measured in mixed over
#   16 seeds (|kappa| of the worst s AO per seed): a route that rounds N, c
#   and Z to float32 separately gives 0.169-0.193 (XLA) and 0.160-0.186 (Rust),
#   16/16 seeds above 0.1 by >= 5 standard deviations; the production code gives
#   0.048-0.065 (XLA, a resolved -0.055 on the diffuse s shell is the largest
#   remaining term) and 0.012-0.045 (Rust), 16/16 seeds below 0.1 by >= 3.5
#   standard deviations. The bound sits between the two populations with a
#   margin of about 0.035 eps on the production side and 0.06 eps on the
#   separate-rounding side.
#   Full precision: |kappa| <= 2.2 eps64 (Rust polynomial fp64 exp), within
#   the same float64 allowance (16 eps64) as the spherical check.
AO_CART_PERIODIC_VALUE_COHERENT_BIAS_EPS_FRACTION = 0.1
AO_CART_PERIODIC_VALUE_COHERENT_BIAS_MIN_EFFECTIVE_POINTS = 100


_XLA_RUST_CONSISTENCY_TOLERANCE: dict[str, dict[str, tuple[float, float]]] = {
    "strict": {
        "float64": (1e-8, 1e-6),
        "float32": (2e-7, 2e-6),
    },
    "medium": {
        "float64": (1e-8, 1e-6),
        "float32": (ATOL_XLA_RUST_CONSISTENCY_MEDIUM, RTOL_XLA_RUST_CONSISTENCY_MEDIUM),
    },
    "independent_fp32": {
        "float64": (1e-8, 1e-6),
        "float32": (ATOL_XLA_RUST_INDEPENDENT_FP32, RTOL_XLA_RUST_INDEPENDENT_FP32),
    },
}


def get_tolerance_xla_rust_consistency(zones: str | Iterable[str], level: str = "strict") -> tuple[float, float]:
    """Return the backend-consistency ``(atol, rtol)`` for one or more zones.

    This helper is only for XLA/JAX-vs-Rust comparisons. Numerical derivatives,
    external references, stochastic integration, and ordinary mixed-precision
    acceptance tests must use their own tolerance families.

    Args:
        zones: A zone name or iterable of zone names crossed by the compared
            quantity. The loosest component dtype controls the result.
        level: ``"strict"`` for ordinary backend parity, or the temporary
            ``"medium"`` envelope for conditionally amplified force/stress or
            parameter-derivative aggregates and deterministic MCMC/GFMC
            end-to-end parity. ``"independent_fp32"`` is the strict pair
            summed over two separate fp32 implementations of one zone (see
            ``ATOL_XLA_RUST_INDEPENDENT_FP32``); in fp64 it equals ``"strict"``.
    Returns:
        The maximum ``(atol, rtol)`` required by the listed active zones.
    """
    from ._precision import _get_zone_str

    zone_names = (zones,) if isinstance(zones, str) else tuple(zones)
    if not zone_names:
        raise ValueError("XLA-Rust consistency tolerance requires at least one precision zone")
    if level not in _XLA_RUST_CONSISTENCY_TOLERANCE:
        raise ValueError(f"unknown XLA-Rust consistency tolerance level: {level!r}")
    pairs = [_XLA_RUST_CONSISTENCY_TOLERANCE[level][_get_zone_str(zone)] for zone in zone_names]
    return max(pair[0] for pair in pairs), max(pair[1] for pair in pairs)


# --- GFMC periodic fast-interval cross-backend floor ---
#
# Under PBC the Rust GFMC wavefunction-derivative block runs the same
# Sherman-Morrison fast update XLA runs, but the two SM carries accumulate in
# different fp environments (libm vs XLA transcendentals, different reduction
# order). Measured on lithium.h5 GFMC_t, 40 branching steps, full precision:
# per-step ``e_L`` max |delta| = 1.0e-7 (max rel 4.6e-7), i.e. an order above
# the ``medium`` fp64 pair (1e-8, 1e-6) at the absolute end. These two numbers
# are the minimum room a periodic fast-interval comparison needs on top of
# whatever the active zones already demand.
#
# They are a floor, never a ceiling: they are fp64-sized, and applying them as
# an unconditional override would make mixed precision stricter than the
# ordinary backend-consistency envelope it is measured against. In mixed
# precision both backends run ao_eval / jastrow_eval / jastrow_grad_lap at
# fp32, and each backend's own fp32-vs-fp64 error on the same walk (measured,
# same fixture: XLA 5.0e-6 abs / 6.2e-5 rel, Rust 1.7e-6 / 2.1e-5) is already
# larger than their mutual disagreement (6.7e-6 / 4.0e-5). No algorithm
# choice can close a gap the reference itself does not resolve. Hence
# ``get_tolerance_gfmc_fast_pbc`` takes the max with the zone tolerance.
ATOL_GFMC_FAST_PBC_FLOOR = 5.0e-7
RTOL_GFMC_FAST_PBC_FLOOR = 5.0e-6


def get_tolerance_gfmc_fast_pbc(
    zones: str | Iterable[str],
    level: str = "medium",
) -> tuple[float, float]:
    """Return the periodic GFMC fast-interval backend-consistency ``(atol, rtol)``.

    The zone-based :func:`get_tolerance_xla_rust_consistency` pair raised to the
    fp64-sized fast-interval floor above. In full precision the floor wins (the
    SM-carry residual exceeds the zone pair); in mixed precision the fp32 zone
    envelope wins, because it is the larger of the two and the floor was never
    meant to tighten it.

    Args:
        zones: A zone name or iterable of zone names crossed by the quantity.
        level: Forwarded to :func:`get_tolerance_xla_rust_consistency`.

    Returns:
        The element-wise maximum of the zone pair and the fast-interval floor.
    """
    atol, rtol = get_tolerance_xla_rust_consistency(zones, level=level)
    return max(atol, ATOL_GFMC_FAST_PBC_FLOOR), max(rtol, RTOL_GFMC_FAST_PBC_FLOOR)


# --- Project policy: every tolerance literal lives in this file ---
#
# NEVER inline ``abs=NUMBER`` / ``atol=NUMBER`` / ``rtol=NUMBER`` in a test
# (``tests/*.py``) or in production code (``jqmc/*.py``, ``jqmc_workflow/*.py``).
# Always import a named constant or call a named helper defined in
# ``jqmc/_setting.py``. The constants and helpers below are the single source
# of truth for every tolerance value used anywhere in the project.
#
# This rule applies even when the intent is bit-exact: define the bit-exact
# constant (``ATOL_BIT_EXACT = 0.0``) once here and import it.
#
# Why: HDF5 / save-load round-trip tests have been broken multiple times
# under ``--precision-mode=mixed`` because inline ``abs=1e-14`` cannot survive
# the post-load active-zone dtype alignment cast. Centralising in this file
# lets the precision-aware helpers (``get_tol_hdf5_roundtrip``,
# ``jqmc._precision.get_tolerance``) be swapped in without touching individual
# tests.

# --- Bit-exact identity tolerances ---
# Use when comparing the output of two implementations that MUST produce
# exactly the same floating-point bits (e.g. fused-vs-standalone JAX kernels
# that share the same backend ops; pure-Python numpy arithmetic where the
# two sides compute the same expression).
ATOL_BIT_EXACT = 0.0
RTOL_BIT_EXACT = 0.0

# --- Pure-Python aggregation math (numpy fp64 only) ---
# Use when comparing two numpy fp64 computations whose only difference
# is floating-point round-off accumulated over a small number of additions
# (BZ-mean weight normalisation, dual-path hand-computed weighted sum,
# constant-shift covariance, etc.). NOT precision-zone-aware -- numpy fp64
# arithmetic has ~e-15 ulp at value O(1) regardless of jQMC's precision mode.
ATOL_PYTHON_FP64_TIGHT = 1.0e-15  # for direct fp64 ulp comparisons at O(1)
ATOL_PYTHON_FP64_LOOSE = 1.0e-12  # for accumulators with multiple additions
ATOL_PYTHON_FP64_BZ_MEAN = 1.0e-14  # for BZ-mean weighted-sum aggregation
RTOL_PYTHON_FP64_ZERO = 0.0  # paired with the atol above; rtol=0 is intentional
# Round-trip bound for the two-significant-digit uncertainty formatting used by
# ``jqmc-tool compute-force``.  This checks the public text interface, not the
# unrounded internal force estimator.
ATOL_TOOL_FORCE_TEXT_ROUNDTRIP = 5.0e-3
# Existing absolute acceptance of the CLI's printed zero-spacing energy from
# fixed synthetic linear checkpoints; distinct from force/text parsing bounds.
ATOL_TOOL_EXTRAPOLATED_ENERGY_DISPLAY = 1.0e-3
# Existing inverse-acceptance least-squares calibration arithmetic check. This
# is a deterministic fit-result bound, not a Monte Carlo confidence interval.
ATOL_WORKFLOW_ACCEPTANCE_FIT = 1.0e-9
# UPF primitive-GTO fit of a radial function that lies exactly in the model
# space (two Gaussians times r^l): exponents, coefficients and the radial-L2
# residual must be recovered to this relative bound. It is the least-squares
# optimizer's convergence floor, not a fit-quality target; the observed worst
# case (l = 1, scipy trf defaults) is 2e-10.
RTOL_UPF_ANALYTIC_GAUSSIAN_FIT_RECOVERY = 1.0e-8
# Same-l radial Gaussian overlap condition number against cond of the overlap
# integrals evaluated by adaptive quadrature (epsrel = RTOL_PYTHON_FP64_LOOSE).
# cond(S) <= 1e6 here, which amplifies the per-entry quadrature error; the
# observed agreement is 4e-15.
RTOL_UPF_OVERLAP_QUADRATURE_REFERENCE = 1.0e-8
# Value-preserving production input-consistency budgets. These are matching /
# validation criteria, not scientific test acceptance or solver convergence.
RTOL_PGTO_INITIAL_EXPONENT_MATCH = 1.0e-12
ATOL_PGTO_REFERENCE_POSITION_MATCH = 1.0e-10
ATOL_PGTO_PRIMITIVE_CENTER_MATCH = 1.0e-6
ATOL_TWIST_CHECKPOINT_KPOINT_MATCH = 1.0e-12
# Integer supercell matrices and unfolded k-point congruences are constructed
# through fp64 determinants/matrix products. This is their setup-time
# integer-residue validation budget, not a runtime scientific tolerance.
ATOL_TILING_INTEGER_RESIDUE = 1.0e-10
# Fractional supercell coordinates within this fp64 roundoff distance of the
# upper face are snapped to its periodic representative at zero.
ATOL_TILING_FRACTIONAL_BOUNDARY_SNAP = 1.0e-13
ATOL_J3_COUPLING_SYMMETRY = 1.0e-12
RTOL_J3_COUPLING_SYMMETRY = 1.0e-9
ATOL_DRIVER_ELECTRON_POSITION_DISTINCTNESS = 1.0e-6
# The preceding placement predicate rejects near-duplicate ELECTRON positions
# during initialization; it does not match or merge nuclear centers.
ATOL_EWALD_CHARGE_NEUTRALITY = 1.0e-9
# Physical/fixture bounds, NOT residual tolerances or equality evidence. The
# MPC cap distinguishes a finite-size correction from replacing the leading
# interaction; the QE caps identify the committed twist-vector geometry.
MPC_FIXTURE_CORRECTION_MAGNITUDE_BOUND_HA = 1.0
QE_NON_GAMMA_FIXTURE_K_NORM_UPPER = 0.50
QE_MULTI_TWIST_FIXTURE_K_NORM_UPPER = 1.5
# Relative counterpart for direct fp64 comparisons at O(1).
RTOL_PYTHON_FP64_TIGHT = 1.0e-15
# ``rel=`` counterpart of ATOL_PYTHON_FP64_LOOSE, for ``pytest.approx(expected,
# rel=...)`` sites comparing a computed fp64 value to its closed-form expected
# value (e.g. ``Structure_data.cell_inradius()``, UPF exponent parsing) --
# same fp64-ulp-at-O(1) floor as the atol sibling, expressed relatively.
RTOL_PYTHON_FP64_LOOSE = 1.0e-12

# --- Explicit bounds equal to the NumPy/JAX allclose/isclose defaults ---
# Distinction predicates are fixture/non-vacuity checks, NOT numerical agreement
# oracles. In particular, a smaller bound would WEAKEN a negated comparison.
ATOL_NONVACUITY_DISTINCTION = 1.0e-8
RTOL_NONVACUITY_DISTINCTION = 1.0e-5
# Matching contraction columns while building the independent orbital reference
# is an input correspondence test, not a production value-comparison budget.
ATOL_ORBITAL_EMBEDDING_MATCH = 1.0e-8
RTOL_ORBITAL_EMBEDDING_MATCH = 1.0e-5
# The absolute FD bound is ATOL_COMPLEX_GRAD_FD; this is its relative
# component (the allclose default).
RTOL_COMPLEX_GRAD_FD = 1.0e-5

# --- LRDMC Haar sampler distribution evidence ---
# The committed Nv=12 icosahedron is defined analytically but represented in
# fp64.  Unit norms and its three off-diagonal dot products therefore use a
# small representation-roundoff band; equal weights remain bit exact.
ATOL_LRDMC_ICOSAHEDRON_GEOMETRY = 1.0e-14

# Deterministic second-moment check over 8192 fixed Threefry samples.  Each
# component has O(1/sqrt(N)) sampling error; 2.5e-2 is over six standard errors
# yet still cleanly rejects a uniform-pitch law (diag 1/4,1/4,1/2).
ATOL_LRDMC_HAAR_MOMENT = 2.5e-2
RTOL_LRDMC_HAAR_MOMENT = 0.0

# --- GFMC incremental-interval twin (IMPL-GFMC-INCREMENTAL-TWIN-001) ---
# Use when comparing the incremental projection interval (carried Ewald ee
# state + per-electron ei / local-ECP rows, one-electron delta per hop)
# against the full-recompute debug twin on the SAME backend and the SAME
# realized walk (the hop chain is route-invariant by construction: b_L and
# the hop cdf never touch the carried potential). The two routes are exact
# modulo fp reordering; the drift bound is ~eps_mach*sqrt(n_hops) relative
# (< 1e-11 even at 1e5 hops), so 1e-9 absolute is a >100x-margin pin that
# still catches any real algebra defect. rtol=0 is intentional (pure
# absolute pin on O(1..100 Ha) interval quantities).
ATOL_GFMC_INCREMENTAL_TWIN = 1.0e-9
RTOL_GFMC_INCREMENTAL_TWIN = 0.0

# --- Rust GFMC S-prime rank-one fast/full twin ---
# The production S' derivative route uses Sherman--Morrison/GEMM updates while
# the frozen debug oracle rebuilds every shifted determinant and Jastrow field.
# Candidate-level H4 Gamma/twist measurements stay below 1e-9. End-to-end
# complex force folds differ by up to 9.2e-9 across CPU implementations due to
# fp64 rank-one/GEMM reduction order; 1e-8 still exposes a missing candidate.
ATOL_RS_GFMC_SPRIME_FAST_VS_FULL = 1.0e-8
RTOL_RS_GFMC_SPRIME_FAST_VS_FULL = 1.0e-8

# --- Mixed-precision integration tolerances (LRDMC / GFMC end-to-end) ---
# Use when comparing end-to-end stochastic-driver outputs (energy, force)
# between two precision modes or between two equivalent drivers. Looser than
# ``strict`` because the integration path accumulates noise across thousands
# of MCMC steps.
ATOL_INTEGRATION_MIXED = 5.0e-3
RTOL_INTEGRATION_MIXED = 5.0e-2

# --- Numerical-derivative inheritance ---
# Upper-case aliases for tests. Definitions above
# (atol_auto_vs_numerical_deriv etc.) remain the canonical entry; the aliases
# below are convenience for tests that want the upper-case style.
ATOL_AUTO_VS_NUMERICAL_DERIV = atol_auto_vs_numerical_deriv
RTOL_AUTO_VS_NUMERICAL_DERIV = rtol_auto_vs_numerical_deriv
ATOL_NUMERICAL_VS_ANALYTIC_DERIV = atol_numerical_vs_analytic_deriv
RTOL_NUMERICAL_VS_ANALYTIC_DERIV = rtol_numerical_vs_analytic_deriv
ATOL_AUTO_VS_ANALYTIC_DERIV = atol_auto_vs_analytic_deriv
RTOL_AUTO_VS_ANALYTIC_DERIV = rtol_auto_vs_analytic_deriv

# --- Debug-vs-production / consistency ---
# NOTE: there are no upper-case ATOL_CONSISTENCY / RTOL_CONSISTENCY aliases:
# fixed fp64-calibrated values do not fit analytic-vs-_auto cross-graph parity
# asserts, where the achievable agreement is precision-mode dependent (two
# independent expression trees; fp32 round-off is path-dependent). Such
# comparisons must derive their tolerance from the zones the
# computation crosses via ``_precision.get_tolerance_min`` (see
# _grad_fidelity_harness._assert_axis_allclose). The lowercase ``atol_consistency`` /
# ``rtol_consistency`` remain: they guard fp64-by-construction STORAGE checks
# (matrix symmetry / orthonormality of loaded data), which are mode-independent.
ATOL_DEBUG_VS_PRODUCTION = atol_debug_vs_production
RTOL_DEBUG_VS_PRODUCTION = rtol_debug_vs_production

# --- Pure-Python aggregator invariants ---
# Use for aggregator invariant assertions whose numerical floor is set by the
# number of summands rather than the per-step round-off (charge neutrality of
# a grid sum, lattice-periodicity of an FFT-based potential, Hermitian
# magnitude of a real-symmetric reconstruction, identity-MO recovery, etc.).
# ~1e-10 / ~1e-9 atol covers grids up to ~1e5 summands at fp64 ulp.
ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT = 1.0e-10
RTOL_PYTHON_FP64_AGGREGATOR_INVARIANT = 1.0e-10

# All-real Gamma and multi-rank SR update checks.  The decoded target update
# must match the frozen reference update at these values.
ATOL_C9_GAMMA_SR_UPDATE = 1.0e-12
RTOL_C9_GAMMA_SR_UPDATE = 1.0e-10
ATOL_PYTHON_FP64_AGGREGATOR_LOOSE = 1.0e-9
RTOL_PYTHON_FP64_AGGREGATOR_LOOSE = 1.0e-9

# Canonical occupied-SD regression floors.  The direct determinant product
# accumulates two independent LU determinants; the MO directional derivative
# is bounded by the centred finite-difference cancellation/truncation error.
ATOL_SD_DIRECT_DET_PRODUCT = 2.0e-12
RTOL_SD_DIRECT_DET_PRODUCT = 2.0e-9
ATOL_SD_MO_DIRECTIONAL_FD = 2.0e-6
RTOL_SD_MO_DIRECTIONAL_FD = 2.0e-5

# --- Route-parity floor (ATOL_FLOOR_MEDIUM / RTOL_FLOOR_MEDIUM) ---
#
# ONE (atol, rtol) pair for EVERY periodic force / stress / translation-invariant
# comparison in the test suite (a single pair, not a per-usage zoo). Set to the
# LOOSEST measured floor across all its usages plus margin. Every other periodic
# comparison sits orders of magnitude inside it -- deliberately: a sub-floor
# regression in the tighter usages goes undetected in exchange for one constant to
# reason about. This is safe because every real bug
# this suite has ever caught violates its identity by >= ~3.5e-2 PER CONFIG (a
# dropped image / wrong metric / raw-vs-min-image displacement is an O(1) relative
# error, not a fp-floor nudge), 3 orders above ATOL_FLOOR_MEDIUM.
#
# Per-usage MEASURED floors (all << ATOL_FLOOR_MEDIUM unless noted):
#   * native-vs-JAX periodic HF force parity -- THE LOOSEST, sets the value. The HF
#     side inherits the periodic-kinetic fp-amplification floor (T = -1/2 grad^2 Psi
#     / Psi at low-|Psi| configs) through its position
#     gradient: <= ~1e-5 abs / ~1.6e-5 rel across seeds x J-ladders on production
#     lithium sphe-ECP MCMC configs. Making drift/Pulay, J3 drift, and the dR
#     channel share the cart-leaf full-image AO gradient does not shrink
#     this -- it is kinetic-amplification-bound, not an image-enumeration wiring gap.
#     On the image-offset-far invariants fixtures the parity re-measures tighter
#     (HF abs 2.4e-7 / large-value rel <= ~6e-7; Pulay abs 3.0e-8 / rel 3.7e-6), but
#     the production kinetic floor is retained as the governing worst case.
#   * native per-config translation SUM RULE, sum_I d./dR_I + sum_i d./dr_i == 0
#     (target zero; atol-bound). With the shared full-image AO gradient,
#     Pulay <= 2.1e-13 and HF <= 1.6e-13
#     (both channels, incl. J3, use the cart-leaf full-image grad; with the dR
#     channel on the full-image cart `pgto_deriv` and the dr drift on the
#     batch/sphe LEPSBAS-cutoff AO grad the residual is Pulay ~1.07e-8 / HF ~4.93e-8).
#   * JAX analytic sum rule (reference XLA assemble): HF max 4.2e-14, Pulay 4.8e-13.
#   * complex-twist Pulay dR vs central FD of Re(ln|Psi|) (Re-projection arbitration,
#     h=1e-5): abs 3.4e-8 / rel 2.3e-6 (FD h^2 truncation + periodic-kinetic floor).
#   * FFT-based PBC translation invariance / PW-reference (AO image sum, MPC kernel):
#     ~1e-9 fp64 FFT floor on a 32^3 grid; a dropped-image bug jumps it to rel ~1.
#
# rtol is inert against a zeros target (the sum rules) but governs the O(1)-valued
# force/stress/AO parity comparisons; keep atol == rtol so a single number covers
# both the small-value (atol-bound) and large-value (rtol-bound) components.
ATOL_FLOOR_MEDIUM = 2.0e-5
RTOL_FLOOR_MEDIUM = 2.0e-5

# --- Rust-vs-XLA VALUE parity tier (RS_PARITY_VALUE) ---
#
# Rust-vs-XLA parity uses a dedicated value tier. The ``get_tolerance_min`` zone
# LEVELS (strict = 1e-8/1e-6, medium = 1e-7/1e-5) are not calibrated against a
# Rust-vs-XLA measurement, and raising EVERY zone to the route-parity floor
# ``max(zone, FLOOR_MEDIUM)`` = 2e-5 on any periodic cell would also loosen
# channels that never touch an atomic orbital.
#
# MEASURED (instrumented run of tests/test_rs_kernel/test_mcmc_observables.py,
# 1892 recorded comparisons over all 26 fixture patterns; worst value per channel,
# ``absmax`` = max|rust - xla|, ``relmax`` = max elementwise |rust-xla|/|xla|):
#
#   AO-INDEPENDENT value / energy channels -- Ewald ee / ei / ei-elementwise, bare
#   Coulomb el-el / el-ion / ion-ion, ECP LOCAL, and the Jastrow values J1/J2/J3/
#   J_DTN -- measure the SAME on molecular and on EVERY periodic cell (real-Gamma,
#   twist, pGTO, PW, B-spline, composite):
#       V_ewald_ee              abs <= 2.22e-14   rel <= 1.30e-13   (the worst)
#       V_ewald_ei              abs <= 7.99e-15   rel <= 4.83e-15
#       V_ewald_ei_elementwise  abs <= 8.22e-15   rel <= 1.19e-14
#       V_ecp_local             abs <= 3.33e-16   rel <= 4.28e-16
#       J1 / J2 / J3 / J_DTN    abs <= 8.88e-16   rel <= 2.43e-16
#   and on the NON-pGTO periodic bases (pw_*, bspline_*, pbc_composite_*) the
#   AO-crossing channels join them:
#       T_kinetic               abs <= 2.49e-14   rel <= 2.13e-15
#       V_ecp_nonlocal          abs <= 7.99e-15   rel <= 1.38e-15
#       ln_det                  abs <= 8.88e-16   rel <= 8.77e-16
#
# Judged at 2e-5, this whole family would have NINE orders of magnitude of
# unused headroom, and a ~1e-7 systematic error (a wrong constant, a dropped
# low-order term) would pass everywhere. The tier below is calibrated to that
# measurement with ~45x (atol) / ~75x (rtol) honest headroom over the worst value.
#
# NOT applied to the AO-crossing channels on periodic sphe-pGTO cells: there the
# SAME instrumented run measures T_kinetic abs <= 1.37e-7 / rel <= 4.49e-7,
# V_ecp_nonlocal <= 8.28e-8, ln_det <= 4.71e-8, and the force / stress / param-grad
# channels reach 7.47e-6 abs -- the pGTO far-image AO truncation that FLOOR_MEDIUM
# exists for. That is the governing case and it keeps FLOOR_MEDIUM. The same
# channels on pw / bspline / composite periodic cells sit at 1e-14, which is what
# identifies the AO cutoff (not "being periodic") as the cause.
ATOL_RS_PARITY_VALUE = 1.0e-12
RTOL_RS_PARITY_VALUE = 1.0e-11

# --- fp64-exact identity floor (FLOOR_STRICT) ---
# For SINGLE-IMPLEMENTATION exact identities (e.g. the per-config translation
# sum rule Sum_I d/dR_I + Sum_i d/dr_i = 0): the comparison target is ZERO, no
# cross-implementation accumulation-order difference exists. Both drift channels
# use the same image set, so the identity closes at the fp64
# cancellation floor -- measured worst 2.1e-13 (rs) / 4.8e-13 (JAX). 1e-10
# gives ~400x margin over the measured floor while still detecting image-set
# asymmetries (which violate at ~1e-7..1e-8, proven by mutation
# testing: that class PASSES the 2e-5 FLOOR_MEDIUM undetected). NOTE the
# FLOOR_* family tiers (MEDIUM/STRICT) are named for their magnitude and are a
# DIFFERENT instrument from the get_tolerance_min zone LEVELS of the same
# words: FLOOR_MEDIUM = the route-parity floor (two implementations of the
# same quantity legitimately differ at ~1e-5 on periodic force because the
# periodic-kinetic fp amplification cannot be eliminated);
# FLOOR_STRICT = the tightest achievable for identity-vs-zero checks.
ATOL_FLOOR_STRICT = 1.0e-10
RTOL_FLOOR_STRICT = 0.0  # identity checks compare against zero; rtol is inert

# --- Non-vacuous e-e Jastrow-channel sanity floor (test_periodic_invariants.py) ---
# A lower bound on |two-body Jastrow value| at a deliberately-close e-e pair
# (~0.6-1.4 Bohr), used to assert a translation-invariance config is NOT
# accidentally vacuous (the lithium 1up/1dn fixtures' natural ~5 Bohr separation is
# exactly the gap that hid the far-image bugs this suite targets). Measured
# |J2| in [0.33, 2.52] on the close-pair configs actually drawn (H4-pbc / lithium
# sphe-ECP); 0.05 sits far below with wide margin while still catching a genuinely
# vacuous (near-zero-coupling) draw.
RS_PERIODIC_INVARIANCE_MIN_JASTROW_TWO_BODY = 0.05

# --- Intermediate fp64 tolerance (between TIGHT and LOOSE) ---
# Use when an intermediate fp64 quantity is compared to a reference computed
# with a slightly different but mathematically-equivalent algorithm
# (np vs jnp recompute, partial vs full sum, etc.). atol ~ 1e-13.
ATOL_PYTHON_FP64_MEDIUM = 1.0e-13

# --- PBC translation invariance / PW reference comparisons ---
# Use for "translate by a lattice vector and re-evaluate -> equal to original"
# assertions and for "evaluate at a PW reference point and check the orbital
# matches the analytic Bloch sum" assertions. The floor is set by the FFT
# precision (~1e-9 at fp64 on a 32^3 grid), a genuinely tighter, root-caused
# floor than the coarse ``ATOL_FLOOR_MEDIUM`` (2e-5) crossed-zone budget.
# Kept as a distinct named constant (NOT folded into the coarse floor) so the
# canonical-MPC translation-invariance test keeps its honest 1e-9 sensitivity: a
# trilinear-interp origin / minimum-image bug can leak an O(1e-9..1e-6) residual
# that the 2e-5 floor would silently pass.
ATOL_PBC_TRANSLATION_INVARIANT = 1.0e-9
RTOL_PBC_TRANSLATION_INVARIANT = 1.0e-9

# --- Periodic AO image-sum seam continuity ---
# The periodic AO image sum must keep ``chi`` continuous as the electron
# crosses the cell boundary (``chi(r)=chi(r+T)`` is trivial under the
# electron wrap and does NOT exercise the image set). Relative threshold on
# the seam mismatch ``|chi(seam-) - chi(seam+)| / max|chi|``: a COMPLETE
# image sum holds the seam to rel ~1e-4 (smooth O(eps) variation only), while
# DROPPING a needed neighbour image jumps it to rel ~1.0 (one side falls to
# ~0). ``1e-2`` cleanly separates the two.
RTOL_PBC_SEAM_CONTINUITY = 1.0e-2

# --- Per-atom vs cell-cell image-truncation equivalence ---
# The per-atom (``Omega - R_A``) image set is a subset of the cell-cell
# (``Omega - Omega``) set, and every dropped image is below
# ``AO_IMAGE_TRUNCATION_ATOL`` everywhere, so the folded AO difference is
# bounded by a small multiple of that envelope. The atol margin sits well
# below an O(1) dropped-NEEDED-image jump, so the equivalence assertion still
# fails loudly if the per-atom metric is over-aggressive.
ATOL_PBC_PER_ATOM_IMAGE_EQUIV = 1.0e-4
RTOL_PBC_PER_ATOM_IMAGE_EQUIV = 1.0e-6

# --- UPF / external-file text-format precision ---
# Use when comparing a parsed UPF / TREXIO / pseudopotential text-format
# field to an expected float literal stored at fixed text precision
# (typical UPF stores exponents to 6-9 decimal digits).
ATOL_TEXT_FORMAT_PARSE = 1.0e-6

# --- Parameter-block non-vacuity floor ---
# A parameter-derivative comparison (analytic vs autodiff) proves nothing when the
# block being compared is numerically zero: both sides return round-off and agree.
# This is the floor, RELATIVE to the largest parameter block of the same cell, that
# a block must reach for its comparison to carry evidence.
#
# Measured separation on the AGP_MO cells: the six well-conditioned GTO leaves sit at
# 1.4e-2 .. 4.6e-2, while the plane-wave / B-spline / composite leaves collapse to
# 2.4e-10 .. 1.4e-17 -- eight or more orders below. A floor of 1e-4 sits in that gap
# with several orders of headroom on both sides, so it separates "this fixture
# exercises the parameter" from "this fixture cannot see the parameter at all"
# without being sensitive to where exactly it is placed.
RTOL_PARAM_BLOCK_NONVACUITY = 1.0e-4

# --- Reported (short-form ufloat) output precision ---
# Use when reading a number BACK out of jqmc's own printed tables, where the
# text carries far fewer digits than the float that produced it. The
# ``ufloat(..., "+2uS")`` short form prints the uncertainty to two significant
# digits and rounds the value to that same place, so a round-trip through the
# text loses everything below it. These bound "the printed field reports the
# estimator value it claims", NOT numerical agreement between two computations:
# they must never be used to compare two computed quantities.
ATOL_REPORTED_VALUE_SHORT_FORM = 1.0e-4
# The uncertainty itself keeps only two significant digits, so its round-trip
# error is relative, not absolute. Still tight enough that a swapped or
# mis-scaled error bar fails: distinct components differ by factors of 2-6.
RTOL_REPORTED_ERROR_SHORT_FORM = 2.0e-1

# --- Finite-difference vs analytic / autodiff (tight band) ---
# Use for "compare the centred finite-difference of a smooth function to its
# analytic / autodiff derivative at small h (~1e-3 for first derivatives)".
# The achievable accuracy is bounded by the FD truncation error (h^2 ~ 1e-6
# for first derivatives) and the cancellation floor (eps_mach / h ~ 1e-13).
# Use for tests on the DTN polynomial kernel where the second derivative
# is bounded by the basis sharpness.
ATOL_FD_VS_ANALYTIC_TIGHT = 1.0e-5
RTOL_FD_VS_ANALYTIC_TIGHT = 5.0e-5

# --- Canonical SD direct-factor oracle ---
# The direct occupied-orbital contraction is compared against independently
# formed NumPy spin determinants.  The two paths use different reduction
# orders (JAX geminal factorization vs two NumPy determinants), so the check
# is not bit exact even in fp64.
ATOL_SD_DIRECT_FACTOR_ORACLE = 2.0e-12
RTOL_SD_DIRECT_FACTOR_ORACLE = 2.0e-9

# --- Complex-grad finite-difference vs autodiff ---
# Use for the JAX Wirtinger convention pin tests: FD step h in a complex
# direction has a higher floor (~1e-4) because the FD operates on
# Re(c) and Im(c) independently and the complex arithmetic doubles the
# cancellation cost.
ATOL_COMPLEX_GRAD_FD = 1.0e-4

# --- Optimizer ground-truth suite: production solver vs dense reference ---
# Used by the complex-optimizer ground-truth test suite to pin the complex
# SR/LM solvers against transparent dense numpy references.
# SR kernels (direct / CG, wide / tall) vs a dense Hermitian solve: bounded
# by the CG stopping tolerance + fp64 accumulation over ~1e2 samples. The
# ``*_SCALE`` constants multiply the max magnitude of the reference vector
# (a scaled atol -- the reference sets the problem's natural scale).
RTOL_SR_SOLVER_VS_DENSE = 5.0e-6
ATOL_SR_SOLVER_VS_DENSE_SCALE = 5.0e-6
# LM step (dgelscut + S-orthonormalization + eigh) vs a dense generalized
# eigensolve: the component ratio v[1:]/v[0] amplifies eigh round-off by the
# conditioning of the extended pencil, hence one decade looser than the raw
# eigenvalue agreement below.
RTOL_LM_STEP_VS_DENSE = 1.0e-6
ATOL_LM_STEP_VS_DENSE_SCALE = 1.0e-8
RTOL_LM_EIGVAL_VS_DENSE = 1.0e-8
ATOL_LM_EIGVAL_VS_DENSE = 1.0e-10
# Covariance quadratic forms (aSR H_1 / H_2 vs the pinned vdot references):
# plain fp64 reductions over the sample axis.
RTOL_QUADFORM_VS_DENSE = 1.0e-8
ATOL_QUADFORM_VS_DENSE = 1.0e-12
# Exact gauge (global phase) constraints on fixed samples
# (test_complex_opt_gt_gauge.py): C_up -> e^{i phi} C_up is an EXACT
# invariance of every w_i and e_L,i.
#   ROTATION -- E(theta) invariance under a finite gauge rotation. The test
#     compares two independently evaluated/reduced energies, so its numerical
#     floor is resolved from every crossed precision zone with
#     get_tolerance_min(..., "strict").
#   NULLSPACE_EXACT -- the gauge tangent i*C_up lies in the exact null space
#     of the centered dO (both the raw projection and its normalized
#     variant against ||f_ref||).
# Measured full-precision floors are 1.5e-17 (rotation, |E|~1.27 scale) and
# 1.07e-15 (nullspace). The nullspace calculation remains entirely fp64 and
# keeps its exact floor. The finite-rotation energy also crosses fp32 forward
# zones in mixed mode and therefore uses the normal dtype-aware strict tier.
ATOL_GAUGE_NULLSPACE_EXACT = 1.0e-12

# --- Interpolation / cusp extrapolation ---
# Use for tests of analytic vs autodiff agreement when the autodiff path
# inherits an interpolation noise floor that the analytic path does not.
# Examples: e-N cusp extrapolation to r=0 (B-spline / pGTO+spline residual);
# DTN polynomial cusp value at the saddle point.
ATOL_INTERPOLATION_RELAXED = 5.0e-3
RTOL_INTERPOLATION_RELAXED = 5.0e-2
ATOL_INTERPOLATION_TIGHT = 2.0e-3
# The DTN e-N cusp is an exact coefficient-space constraint.  Its independent
# NumPy finite-difference oracle reconstructs the fp64 storage coefficients, so
# the same absolute contract applies in full and mixed execution modes.
ATOL_DTN_EN_CUSP_CONSTRAINT = 1.0e-8
RTOL_DTN_EN_CUSP_CONSTRAINT = 0.0

# --- DTN comoving cutoff policy (``Jastrow_data._apply_strain``) ---
# ``cutoff_scale`` multiplies the stored cutoffs, so a scaled cutoff is a single
# fp64 product: one rounding, hence an ulp-level relative contract.
RTOL_DTN_COMOVING_CUTOFF_SCALE = 1.0e-14
# Scalar cutoff checks use an absolute (not relative) bound, separate from
# the array relative comparison above.
ATOL_DTN_COMOVING_CUTOFF_SCALE = 1.0e-14
# The inflated coefficients ride the minimum-norm constraint tangent to FIRST
# order in ln(cutoff_scale), so the cusp constraint at the SCALED cutoff holds
# only to second order: at |ln s| ~ 1e-3 the residual floor is ~1e-6.
ATOL_DTN_COMOVING_CUSP_TRANSPORT = 5.0e-6
# Comoving strain jacrev vs fixed-cutoff jacrev plus the closed-form policy term
# ``dJ/dln(ell_V) / 3 * I``.  In full precision the two autodiff traces and the
# analytic policy agree to accumulated fp64 round-off on an O(1) Jastrow value.
# In mixed mode the traces cross the fp32 ``jastrow_eval`` zone while the
# analytic policy remains fp64; compare their fp32 representations instead.
# The three diagonal policy contributions differ by exactly one float32 ULP on
# the committed seed-0 fixture, so one ULP is both the measured arithmetic floor
# and a substantially stronger contract than a generic mixed ``allclose``.
RTOL_DTN_COMOVING_POLICY_IDENTITY = 1.0e-10
ATOL_DTN_COMOVING_POLICY_IDENTITY = 1.0e-12
MAXULP_DTN_COMOVING_POLICY_IDENTITY_MIXED = 1

# Rank cutoff for the independent DTN constrained-polynomial test reference.
# Preserve its original 1e-10 pseudoinverse algorithm: this chooses the null
# space of the redundant constraint rows, not an actual/reference acceptance
# window. Production constraint construction and comparison budgets are unchanged.
RCOND_DTN_CONSTRAINT_REFERENCE = 1.0e-10

# Decimal thresholds of the TurboRVB regression tests: regression limits, not
# a physical-accuracy claim. The TurboRVB commit and recipe of the reference
# values are not recorded, so these limits must not be retuned.
DECIMAL_TURBORVB_AE_JASTROW_POTENTIAL = 2
DECIMAL_TURBORVB_AE_POTENTIAL = 3
DECIMAL_TURBORVB_ECP_POTENTIAL = 5
DECIMAL_TURBORVB_VALUES = 6
DECIMAL_COULOMB_ROUTE_PARITY = 10
DECIMAL_RECIPROCAL_CELL_DUALITY = 15

# Largest integer image box ``prod_k (2 N_k + 1)`` the static minimum-image
# helpers of jqmc.structure (DTN image shell, shortest-lattice-vector search)
# may expand; it is an allocation/compile-size limit, not a tolerance. A
# selected DTN plan with an LLL-reduced wrap basis has N_k <= 1 (27 candidates)
# and the standard cells keep at most that, so feasible cells are far below it.
# A larger box (an extreme unreduced basis whose reduction failed) raises a
# ValueError instead of attempting the allocation (PR #82).
MINIMUM_IMAGE_SHELL_MAX_CANDIDATES = 65536
# Full-precision central finite difference of the isotropic comoving strain.
# This retains the original absolute bound; mixed execution is excluded because
# differencing the fp32 Jastrow value at h=1e-5 is round-off dominated.
ATOL_DTN_COMOVING_POLICY_FD = 1.0e-8

# --- Algorithm-specific PW / pGTO projection tolerances ---
# Tolerances used by the PW / pGTO projection tests, tuned to specific
# algorithmic constants (PW cutoff, alpha_min sweep, spline order).
# These are NOT precision-zone-aware -- the PW projection runs in fp64 only.
ATOL_PW_PROJECTION_FP64 = 1.0e-10
ATOL_PW_PROJECTION_FP64_LOOSE = 1.0e-8
ATOL_PW_PROJECTION_FP64_RELAXED = 1.0e-7
ATOL_PW_PROJECTION_FP64_SPLINE_RESIDUAL = 1.0e-6
ATOL_PW_PROJECTION_FP64_GRID = 1.0e-9
# Existing literal factors/floors in independent overlap/projection checks.
# Factors multiply the same reference/target as before; they are NOT newly
# loosened bounds or replacements for element-wise comparisons.
ATOL_AO_OVERLAP_GRID_QUADRATURE = 5.0e-4
ATOL_AO_IMAGE_TRUNCATION_SUM_FACTOR = 1.0e3
RTOL_PGTO_LATTICE_OVERLAP_CONVERGENCE_FACTOR = 100.0
RTOL_PW_PROJECTION_RESIDUAL_PARITY_FACTOR = 1.05
RTOL_PW_PROJECTION_SPLINE_RESIDUAL_PARITY_FACTOR = 100.0
RTOL_COMPLEX_PGTO_TRANSLATION_ENVELOPE = 1.0e-8
RTOL_LRDMC_STREAMING_MESH_CHAIN = 1.0e-6
RTOL_RS_ECP_PARAMETER_SM_REBUILD = 1.0e-9
RTOL_RS_ECP_PARAMETER_SM_REBUILD_ILL_CONDITIONED = 1.0e-6
ATOL_GAUGE_REFERENCE_SCALE_FLOOR = 1.0e-30
ATOL_RELATIVE_COMPARISON_SCALE_FLOOR = 1.0e-300
ATOL_PERIODIC_EL_KINETIC_FLOOR = 3.0e-7
ATOL_EWALD_BACKEND_PARITY = 1.0e-10
RTOL_EWALD_BACKEND_PARITY = 1.0e-10
MAXULP_MPC_SENTINEL_HISTORY = 4.0
RTOL_PW_PROJECTION_FP64 = 1.0e-10
RTOL_PW_PROJECTION_FP64_LOOSE = 1.0e-7
RTOL_PW_PROJECTION_FP64_RELAXED = 5.0e-2
# PW-grid coefficient comparison: ``ATOL_PW_PROJECTION_FP64_GRID``-paired rtol
# for PW projection coefficient identities that compare against a reference
# computed from the same Phi^H Phi block but with the full S vs Schur-complement
# path. The ulp drift between paths sits at ~1e-9 relative.
RTOL_PW_PROJECTION_FP64_GRID = 1.0e-9
# PW reconstruction Plancherel-style norm assertion: the rtol band that the
# QE-normalized PW reconstruction `||psi||^2 dV` agrees with the cell volume
# to (Plancherel identity with a band-limited Miller set). Tighter than the
# residual band because the norm is a single scalar.
RTOL_PW_PROJECTION_FP64_NORM = 1.0e-5

# --- QE -> jQMC projection FIDELITY acceptance bands (test_pw_pgto_projection.py) ---
# Distinct from the ``*_PW_PROJECTION_FP64*`` family above (which pins two
# implementations of the SAME projection to fp64 round-off): these bound the
# projection's *physical* quality -- how well the emitted pGTO/PW/spline
# basis reproduces the QE reference bands -- against real UPF fixtures
# (Li_gamma / Cu). Per-tier distinctions the tests encode:
#   PGTO_ONLY_MEAN/MAX -- bare UPF-seeded pGTO fit (no PW/spline residual
#     channel), the loosest tier. Two independent code paths (the FFT/four
#     and the real-space/debug pGTO projector) are both held to this same
#     band so they can share the constant.
#   SWEEP_TARGET -- the ``optimize_mixed_basis`` design target (matches the
#     ``residual_target=1e-4`` default threaded through the same tests as a
#     builder kwarg): pgto+PW and pgto+spline combined models, and the
#     PW/spline dual-backend cross-check, all converge to this tier.
#   SPLINE_ONLY -- quintic spline alone on the native FFT mesh (no pGTO),
#     one order looser than the combined-model SWEEP_TARGET.
#   TIGHT_SWEEP -- the Cu fixture sweep's best trial, which reaches
#     near-machine-zero residual at the full QE Miller set; tighter than the
#     generic SWEEP_TARGET.
ATOL_QE_PROJECTION_FIDELITY_PGTO_ONLY_MEAN = 5.0e-3
ATOL_QE_PROJECTION_FIDELITY_PGTO_ONLY_MAX = 2.0e-2
ATOL_QE_PROJECTION_FIDELITY_SWEEP_TARGET = 1.0e-4
ATOL_QE_PROJECTION_FIDELITY_SPLINE_ONLY = 1.0e-3
ATOL_QE_PROJECTION_FIDELITY_TIGHT_SWEEP = 5.0e-4
# Whole-cell additive (pGTO real-space image-sum + spline residual)
# reconstruction vs the direct PW band value, relative L2 norm. An order of
# magnitude below the AO periodic image-truncation floor
# (``AO_IMAGE_TRUNCATION_ATOL = 1e-7``) that bounds it from above.
ATOL_QE_PROJECTION_RECONSTRUCTION_RESIDUAL = 1.0e-6

# --- QE PW/spline orbital RECONSTRUCTION relative-error tiers ---
# Used by test_qe_lsda.py, test_integration_qe_non_gamma.py and
# test_qe_to_jqmc.py to bound the relative L2-norm error between a projected
# orbital reconstruction (``compute_MOs`` on the emitted jQMC basis) and the
# direct QE PW band sum evaluated independently in the test. Three tiers:
#   RTOL_PW_PROJECTION_FP64 (existing, reused) -- ``kind="pw"`` projection,
#     lossless within fp64 round-off (no interpolation/truncation channel).
#   RTOL_QE_PW_RECONSTRUCTION_TIGHT -- whole-grid PW MO reconstruction
#     against the QE reference bands (LSDA Gamma path; Gamma pw-build path
#     in test_qe_to_jqmc.py). A looser tolerance for FFT-grid round-off is not
#     justified: the measured floors are 1.37e-15 (test_qe_lsda) and 2.66e-15
#     (test_qe_to_jqmc),
#     i.e. the grid_points() channel is invisible at fp64. Unified with the
#     RTOL_PW_PROJECTION_FP64 tier value (1e-10, >4 orders above the
#     measured floor), so both lossless PW reconstructions (qe_lsda and
#     non_gamma) gate at 1e-10.
#   RTOL_QE_PW_RECONSTRUCTION_LOOSE -- ``kind="spline"`` projection
#     (interpolation residual) or a +/-k twist reconstruction (extra
#     phase-convention bookkeeping over multiple bands).
RTOL_QE_PW_RECONSTRUCTION_TIGHT = 1.0e-10
RTOL_QE_PW_RECONSTRUCTION_LOOSE = 1.0e-5
# Per-band rescale-invariant regression gate for composite (pgto+pw /
# pgto+bspline) conversions vs the direct PW reference (test_qe_to_jqmc.py).
# 0.1% is a generous bound: every working backend measures well under it,
# while a ``project_pw_to_pgto_spline`` defect of the kind this test catches
# gives 0.5%-1.7% per-band residual even when its own self-measured fidelity
# residual reports ~1e-9 (a self-measurement cannot see
# composite-orbital-vs-PW divergence).
RTOL_QE_COMPOSITE_PER_BAND_REGRESSION_GATE = 1.0e-3

# --- QE Gamma-point pGTO projection realness gate (jqmc/qe_to_jqmc.py) ---
# At Gamma the projected pGTO MO coefficients should be real up to global
# phase; ``max|Im(mo_coeffs)|`` above this is logged as a WARNING (signal
# that the upstream projection went wrong, or the caller forgot to set
# ``k_point`` in structure_data) before the real part is taken. Same
# magnitude as the general UPF/text-format parse floor
# (``ATOL_TEXT_FORMAT_PARSE``) but kept as its own named constant since this
# one gates PRODUCTION control flow (a warning path), not a test assertion.
ATOL_QE_GAMMA_REALNESS_GATE = 1.0e-6

# --- External-reference Ewald / Madelung comparisons ---
# Use when comparing an internal Ewald (ion-ion or Madelung) result against
# an external independent implementation (pymatgen ``EwaldSummation``, the
# literature Madelung constant ``alpha_M``). Both reference values are
# truncation-converged well below ``1e-10`` (pymatgen ``acc_factor=18``,
# literature ``alpha_M`` accurate to 10+ digits), so the achievable agreement
# floor is the internal Ewald's own truncation tolerance (typically
# ``tol=1e-12`` passed to ``build_ewald_data`` in these tests). ``1e-8`` is a
# conservative atol that survives both ladder choices and is paired with
# ``rtol=0`` because the absolute energy scale is fixed by the fixture.
ATOL_EWALD_VS_EXTERNAL_REFERENCE = 1.0e-8
RTOL_EWALD_VS_EXTERNAL_REFERENCE = 0.0

# Tolerance against pymatgen for the CHARGED slab fixtures
# (net ionic charge +24 with a neutralising background; references span
# 16-60 Ha, so a relative tolerance is the honest metric). The measured
# jqmc-vs-pymatgen agreement is 6.8e-10 relative on both fixtures (the
# residual sits in the charged-cell background handling of either code);
# 5e-9 gives ~7x headroom while still catching the ~1e-3-relative slab
# truncation bug these fixtures were built for. QE 7.5's printed `ewald
# contribution` agrees with both to ~2e-8 relative (its own convergence).
RTOL_EWALD_VS_PYMATGEN_CHARGED_SLAB = 5.0e-9


# --- Test tolerance for HDF5 save/load round-trip equality (zone-aware) ---
#
# ``Hamiltonian_data.save_to_hdf5`` / ``load_from_hdf5`` does NOT guarantee
# bit-exact dtype preservation: the post-load
# ``_apply_{ao_basis,geminal,dtn}_dtype_alignment_post_load`` hooks cast the
# loaded values to the active precision zone (e.g. ``ao_eval`` is float32
# under ``--precision-mode=mixed``). HDF5 round-trip tests therefore cannot
# use bit-exact ``pytest.approx(abs=1e-14)`` / ``np.testing.assert_array_equal``;
# they must compare with the LOOSEST zone tolerance among the zones a
# Hamiltonian_data dataclass touches.
#
# Use ``get_tol_hdf5_roundtrip(kind)`` from this module everywhere.
_HDF5_ROUNDTRIP_ZONES: tuple[str, ...] = ("ao_eval", "jastrow_eval", "det_eval")


def get_tol_hdf5_roundtrip(kind: str = "strict") -> tuple[float, float]:
    """Return ``(atol, rtol)`` for HDF5 save/load round-trip equality.

    Resolves to ``get_tolerance_min`` over the ``_HDF5_ROUNDTRIP_ZONES`` set
    so the test passes in both ``--precision-mode=full`` (all zones fp64;
    atol ~ 1e-8) and ``--precision-mode=mixed`` (any zone fp32; atol ~ 1e-5).

    Args:
        kind: Tolerance level, one of ``"strict"`` or ``"medium"``. See the
            top-of-file ``_TOLERANCE`` documentation for the contract.

    Returns:
        Tuple ``(atol, rtol)`` in the loosest-zone semantics.
    """
    # Lazy import to avoid a circular import (``_precision`` reads
    # ``_TOLERANCE`` from this module).
    from ._precision import get_tolerance_min

    return get_tolerance_min(_HDF5_ROUNDTRIP_ZONES, kind)


# Native compile-time policy values are generated from these literal definitions.
# Keep the FP64 stabilization thresholds shared with the dtype-aware Python
# helpers below; the FP32 formulas retain their original dtype-specific values.
AS_LINEAR_MIN_SQ_NORM_FP64 = 1.0020841800044864e-292  # finfo(float64).tiny / eps
AS_NORM_CANCELLATION_THRESHOLD_FP64 = 1.4901161193847656e-8  # sqrt(eps64)
SM_LARGE_RATIO_THRESHOLD_FP64 = 67108864.0  # 1 / sqrt(eps64)

# SVD pseudoinverse cutoff. The dtype-aware float64 entry and Rust use this
# same literal; float32 retains its existing larger cutoff.
EPS_rcond_SVD = 1.0e-20

# Native per-image radial screen exp(-20.72), approximately 1e-9. The literal
# is used as is (not recomputed as -log(1e-9)) so results are exact.
NATIVE_AO_SCREEN_EXPONENT = 20.72

# Native force kernels omit singular zero-distance terms below this radius.
# This is a skip threshold, separate from the distance stabilization epsilon.
NATIVE_DISTANCE_SKIP_THRESHOLD = 1.0e-12

# Existing Rust unit-test budgets, centralized without changing their values.
# These are scalar fp64 oracle/reduction budgets, independent of Python's
# dtype-aware scientific comparison families.
ATOL_RS_UNIT_NEAR_MACHINE = 2.0e-15  # trig and DTN cached geometry tables
ATOL_RS_UNIT_REDUCTION = 2.0e-12  # DTN ratios and incremental J3 tables
ATOL_RS_UNIT_TIME_REVERSAL = 2.0e-14  # complex B-spline conjugation
RTOL_RS_UNIT_RANK1_FORCE = 1.0e-11  # max(1, |reference|)-scaled rank-1 forces
ATOL_RS_ERFINV_LEGACY_CORE = 1.0e-6  # rounded erfinv reference values
ATOL_RS_ERFINV_LEGACY_TAIL = 1.0e-5  # rounded tail reference values

# --- Dtype-aware EPS constants ---
#
# Some EPS values are tuned for float64 and break under float32 (underflow,
# loss of stabilization).  Use ``get_eps(name, dtype)`` to obtain the
# appropriate value for the current precision zone.
#
# Constants:
#   machine_precision -- floor for safe ratio in diagnostics.
#   rcond_svd         -- threshold for SVD pseudoinverse of the geminal matrix.
_EPS_DTYPE_AWARE: dict[str, dict[str, float]] = {
    "machine_precision": {"float64": 1e-38, "float32": 1e-38},
    "rcond_svd": {"float64": EPS_rcond_SVD, "float32": 1e-16},
}


def get_eps(name: str, dtype: npt.DTypeLike) -> float:
    """Return a dtype-aware numerical stability constant.

    Args:
        name: One of ``"machine_precision"``, ``"rcond_svd"``.
        dtype: A NumPy/JAX dtype (e.g. ``jnp.float32``, ``np.float64``).

    Returns:
        The appropriate epsilon value for the given dtype.

    Raises:
        KeyError: If *name* is not a known EPS constant.
    """
    dtype_key = "float32" if np.dtype(dtype) == np.float32 else "float64"
    return _EPS_DTYPE_AWARE[name][dtype_key]


def as_linear_min_sq_norm(dtype: npt.DTypeLike) -> float:
    """Return the smallest ``S`` for which the AS factor is evaluated as ``(S * F)**(-3/8)`` (issue #131).

    The Attaccalite-Sorella factor ``R_AS = (S F)^(-3/8)`` (``S`` the minimum
    squared row/column norm of ``G``, ``F = ||G^-1||_F^2``) is invariant under
    scaling one row of ``G``, but a far electron makes that row tiny: below
    ``sqrt(tiny)`` its squares underflow (``S = 0``) while ``F`` overflows, and
    the linear product ``0 * inf`` would return ``R_AS = 0``. The linear form is
    kept (bit-identical) while ``S >= tiny / eps`` of the real dtype -- every
    square contributing more than ``eps * S`` is then a normal number -- and ``F``
    and ``S * F`` are finite. Otherwise ``ln S`` and ``ln F`` are taken from
    max-abs-scaled sums and ``R_AS = exp(-3/8 (ln S + ln F))``. The native
    backend (``determinant::AS_LINEAR_MIN_SQ_NORM``) uses the float64 value.
    """
    if np.dtype(dtype) == np.float64:
        return AS_LINEAR_MIN_SQ_NORM_FP64
    info = np.finfo(np.dtype(dtype))
    return float(info.tiny / info.eps)


def as_incremental_norm_cancellation_threshold(dtype: npt.DTypeLike) -> float:
    """Return the relative size below which an incrementally updated AS norm is recomputed directly (issue #134).

    The native AS factor of a PROPOSED move updates the cached squared column
    (up move) or row (dn move) norms of ``G`` as ``n2 - G_kj^2 + G'_kj^2``. When
    the moved electron dominates that column / row and ``|G'_kj| << |G_kj|`` (a
    tiny determinant ratio), the result keeps only ``eps * n2`` of absolute
    accuracy and can come out zero or negative. A norm whose incremental value is
    below ``sqrt(eps) * n2`` of the real dtype -- so that at most about half of
    the digits would survive -- is recomputed from the proposed column / row,
    and the moved line itself is then rebuilt from the new orbital values rather
    than as ``G_k + (G'_k - G_k)``. The native backend
    (``determinant::AS_NORM_CANCELLATION_THRESHOLD``) uses the float64 value
    ``2**-26``; the XLA kernels always rebuild the moved line.
    """
    if np.dtype(dtype) == np.float64:
        return AS_NORM_CANCELLATION_THRESHOLD_FP64
    return float(np.finfo(np.dtype(dtype)).eps) ** 0.5


# Carried Sherman--Morrison inverse refresh.
# A running inverse seeded at a near-singular geminal keeps the seed's absolute
# error when the walker leaves the near-singular region; the rank-1 update never
# repairs it (exact algebra: E_new = E - (E u)(v^T X)/d with E = G X - I), so the
# residual relative to the now well-conditioned inverse can grow to 1e-5..1e-2.
# At every measurement boundary the MCMC driver evaluates the O(N^2) probe
# residual ``max|G (X P) - P|`` (two fixed probe columns P) and recomputes the
# inverse of the SAME carried G (the seed's SVD pseudo-inverse) only for walkers
# whose residual exceeds this threshold. ``float("inf")`` disables the refresh.
# The default 1e-10 balances accumulated inverse error against rebuild cost.
SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD = 1.0e-10

# Issue #605: seed/refresh inverses of a walker batch (MCMC seed and carried-inverse refresh, GFMC_t/GFMC_n
# seeds) are the equilibrated LU inverse ``C (R G C)^-1 R`` of every walker. A walker keeps it only
# when it is finite and its scale-free probe residual ``max|A (Y P) - P|`` (``A = R G C``, ``Y`` its
# LU inverse, ``P`` the two fixed probe columns) is at most this bound; otherwise it gets the
# equilibrated SVD pseudo-inverse. An invertible geminal stays at eps*cond (about 1e-10 at
# cond 1e6, 5e-7 at cond 1e10), while LU on a numerically rank-deficient geminal (coincident
# electrons, a linearly dependent row) returns finite garbage with a residual of order one.
# Mirrored by the Rust refresh. Not a user knob.
GEMINAL_LU_GUARD_RESIDUAL_THRESHOLD = 1.0e-6

# Issue #605: largest |exponent| of the power-of-two row/column equilibration scales of the geminal
# SVD pseudo-inverse (the LU inverse keeps the full 1021 range). The unscale factor of ``C A^+ R`` is
# then at most 2**956 and ``|A^+| <= 1/(rcond_svd s_max)`` is about 2**66, so the pseudo-inverse of
# any finite geminal is finite (with 1021, a far up and a far down electron made it overflow to NaN).
# Only rows/columns with an infinity norm below about 1e-144 are left partly unscaled. Mirrored by
# the Rust pseudo-inverse. Not a user knob.
GEMINAL_EQUILIBRATION_MAX_EXPONENT = 478


def sm_large_ratio_threshold(dtype: npt.DTypeLike) -> float:
    """Return the determinant-ratio magnitude above which the rank-1 inverse update is exact-column (issue #127).

    An accepted single-electron move replaces row ``k`` (up move) or column
    ``k`` (dn move) of ``G``; the new inverse's column ``k`` (up) or row ``k``
    (dn) is exactly the old one divided by the ratio ``R``. The generic rank-1
    form ``X - (X u)(v^T X) / R`` evaluates it as ``X e_k (1 - (R - 1) / R)``,
    whose relative error is about ``eps |R|``: all digits are lost once
    ``|R| > 1 / eps``, and the next move's ratio is then wrong. Above
    ``1 / sqrt(eps)`` of the inverse's (real) dtype -- where that error would
    exceed ``sqrt(eps)`` -- the production update writes the exact ``X e_k / R``
    (or ``e_k^T X / R``) instead. Below it the generic rank-1 form is
    used. The native backend (``determinant::SM_LARGE_RATIO_THRESHOLD``)
    uses the float64 value ``2**26``.
    """
    if np.dtype(dtype) == np.float64:
        return SM_LARGE_RATIO_THRESHOLD_FP64
    return float(np.finfo(np.dtype(dtype)).eps) ** -0.5


# Threshold for zero-division guards.
# Denominators with absolute value below this are treated as zero.
EPS_zero_division = 1.0e-30

# Small epsilon for safe distance / sqrt stabilization.
# Added to squared distances before sqrt to keep gradients finite
# when particles coincide (e.g. r_eN -> 0).
EPS_safe_distance = 1.0e-12

# Near-machine-precision floor for safe ratio in diagnostics.
EPS_machine_precision = 1.0e-300

# Relative-tolerance amplification for comparing a SQUARED observable (e_L2 =
# e_L**2, and any future x**2 pair) under the envelope derived for its BASE
# quantity. To first order d(x^2)/x^2 = 2 dx/x exactly, so two implementations
# that agree on x to rtol agree on x^2 only to 2*rtol -- judging both against
# the same rtol makes the squared quantity fail at exactly half the base
# quantity's headroom, structurally, on every fixture. Measured on the lithium
# ccECP GFMC debug-vs-production pair: e_L used
# 0.80 of its tolerance while e_L2 used 0.96 before the per-species cusp fix,
# and 0.67 vs 1.09 after it -- same ordering, amplification 1.2-1.6x, bounded
# by this factor. Applies to rtol ONLY: the first-order atol image is
# 2|x|*atol, which for |x| < 0.5 is TIGHTER than atol, so keeping the base
# atol is the conservative choice on both sides.
RTOL_AMPLIFICATION_SQUARED_OBSERVABLE = 2.0

# Margin below the bare atomic number at which a species counts as PSEUDIZED
# when the DTN electron-nucleus cusp condition is built. Z_eff and Z are both
# integers in every supported pseudopotential table, so the smallest real gap is
# 1 (one core electron removed); this margin only has to absorb the float
# round-trip through np.float64, hence a value many orders below that gap. A
# species at Z_eff == Z is all-electron and keeps chi'(0) = -Z_eff; a pseudized
# one is smooth at the nucleus, chi'(0) = 0.

# Absolute floor for diag(S) in scale-invariant SR.
# Parameters with diag_S below this threshold are considered converged
# (the wavefunction derivative has near-zero variance) and are frozen
# to prevent the scale-invariant normalization from amplifying noise
# into catastrophically large parameter updates.
min_S_diag_abs = 1.0e-10


# --- Wave-function optimizer numerical policy ---
#
# Keep every optimizer tolerance/cutoff policy here.  In particular, do not
# bury dtype-dependent validation bands or multiples of machine epsilon in
# ``wavefunction.py``, ``jqmc_optimizer.py``, or
# ``jqmc_optimizer_kernel.py``.  Solver inputs remain user-overridable; these
# are only their documented defaults and internal round-off guards.

# Public solver defaults used when the corresponding ``optimizer_kwargs``
# entry is absent.
# ``cg_tol`` is RELATIVE (issue #859): every SR CG
# stops once ||r|| <= cg_tol * ||b|| for the right-hand side b it solves.  At
# 1e-5 the SR direction differs from the exact solve by ~0.5 %, far below its
# statistical noise (the issue's N2 study: jackknife error of q ~120 %).
OPTIMIZER_DEFAULT_CG_TOL = 1.0e-5
# Iteration cap of every SR CG when ``cg_max_iter`` is absent.  One value for
# the Python API (``run_optimize``), the CLI default table and the
# documentation.
OPTIMIZER_DEFAULT_CG_MAX_ITER = 10000
OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF = 1.0e-3
# aSR |gamma| safety cap (together with the non-symmetric-pencil /
# negative-root policy). A cap of 1e6 is effectively
# no guard. Healthy runs use |gamma| ~ 0.17-0.48 (measured on a
# series of slab runs), so 0.5 admits the observed healthy range while
# rejecting outlier roots; beyond the cap the plain-SR fallback magnitude
# (below) is used with the model root's SIGN. Tighten per run via
# optimizer_kwargs["asr_gamma_max"] for delicate systems.
OPTIMIZER_DEFAULT_ASR_GAMMA_MAX = 0.5
OPTIMIZER_DEFAULT_ASR_FALLBACK_GAMMA = 0.1
OPTIMIZER_DEFAULT_LM_REFERENCE_WEIGHT_MIN = 0.9

# LM root classification (``solve_lm_real_pencil``): an eigenvalue of the
# Cholesky-reduced non-symmetric LM pencil counts as real when
# ``|Im lambda| <= OPTIMIZER_LM_REAL_ROOT_IMAG_RTOL * max(|lambda|, 1)``.
# Basis: a real root that is (nearly) defective splits under an O(eps)
# perturbation of the matrix into a complex pair of relative size
# O(sqrt(eps)) (Jordan-block perturbation, eigenvalue condition ~ eps**(1/2));
# 1e-8 ~ sqrt(eps_fp64) = 1.5e-8 accepts such round-off splits while a
# genuinely complex pair of the sampled pencil is excluded.
OPTIMIZER_LM_REAL_ROOT_IMAG_RTOL = 1.0e-8

# Full local-energy derivatives are staged on host for reduced LM. Stream this
# many samples back to the accelerator at a time for coordinate contraction;
# the reduced result has only ``lm_subspace_dim + 1`` columns. This bounds the
# extra HBM independently of the total walker/step count while retaining GPU
# throughput for the large parameter-axis contraction.
OPTIMIZER_DERIVATIVE_SAMPLE_BATCH = 1024
# Number of coordinates listed per optimization step, at verbosity = "high",
# in the "Top N coordinates by generalized-force signal-to-noise" block.
# Bounded on purpose: listing every coordinate does not scale with N.
OPTIMIZER_DIAGNOSTICS_TOP_COORDINATES = 50

# Absolute eigenvalue floor used consistently by every NumPy/JAX Hermitian
# square-root construction of the periodic AO overlap.  This is a numerical
# regularization parameter, not a proof that the resulting metric is the
# physical overlap on a rank-deficient basis.
AO_OVERLAP_EIGENVALUE_FLOOR = 1.0e-10

# Warning threshold of the MO orthonormality diagnostic run by
# ``Hamiltonian_data.sanity_check`` (``jqmc.hamiltonians._warn_mo_orthonormality``).
# It bounds both ``max_i |M_ii - 1|`` and ``max_{i!=j} |M_ij| / sqrt(M_ii M_jj)``
# of the MO Gram matrix ``M = conj(C) S C^T`` (fp64/complex128). It only selects
# a logger.warning; nothing is rejected. Chosen from measurements, identical in
# full and mixed precision:
#   * numerical floors: committed PySCF TREXIO fixtures (molecular, periodic
#     Gamma and twists) <= 3.8e-15 / 2.0e-13; QE B-spline and pGTO+B-spline
#     quadrature overlaps up to 7.9e-7 (mixed, fp32 samples);
#   * real defects in the committed QE fixtures: projected / truncated MOs are
#     non-orthogonal by 9.0e-4 (pGTO+B-spline twist) to 1.4e-2 (48-PW cap), and
#     every QE-derived set written before the unit-cell normalization convention
#     carries the common sqrt(Omega) scale (|M_ii - 1| ~ Omega = 2.9e2 to 8.6e2
#     bohr^3). QE conversions are unit-cell normalized and pure pGTO MOs are
#     orthonormalized; the threshold value does not depend on that.
# 1e-4 keeps two decades above the largest numerical floor and a factor ~9
# below the smallest measured real non-orthogonality.
MO_ORTHONORMALITY_WARNING_THRESHOLD = 1.0e-4

# Largest relative S-norm ``||r||_S / ||c||_S`` of the part of a Cartesian MO
# that ``MOs_data.to_spherical`` discards (``r = c - c R T``, the component
# outside the pure-spherical subspace, e.g. the r^2 trace of a d shell) for
# which ``jqmc-tool trexio convert-to --ao-conv-to sphe`` still counts as a
# representation change; above it the CLI logs a WARNING (issue #591). Host-side
# fp64/complex128 setup arithmetic, identical in full and mixed precision.
# Measured on the committed TREXIO fixtures:
#   * numerical floors (exact embeddings: Cartesian MOs without l >= 2 shells,
#     and spherical sources mapped by to_cartesian): <= 1.7e-16 molecular,
#     <= 5.2e-13 periodic (Li_pbc_ae_ccpvdz_sphe, Gamma) / 2.7e-15 (twists);
#   * native Cartesian SCF MOs with d/f shells: 4.5e-4 (Li cc-pVDZ) to
#     4.4e-2 (N2 cc-pVTZ), changing ln|Psi| by up to O(1).
# 1e-8 keeps over four decades above the largest floor and over four decades
# below the smallest measured real loss.
AO_SPHERICAL_PROJECTION_RESIDUAL_THRESHOLD = 1.0e-8

# Smallest row coherence rho_k = |sum_i c_ik| / sum_i |c_ik| (Euclidean norms of
# MO row k of input i, host fp64/complex128) that ``jqmc-tool vmc average-wf``
# accepts before averaging determinant MO coefficients (issue #749). A sign or
# phase (gauge) jump of an MO row between the averaged steps shrinks rho_k; an
# arithmetic mean over such a jump is not a mean of one continuous trajectory.
# The optimizer itself has no jump: an S-Stiefel tangent step followed by the
# polar retraction is continuous. Measured on real optimizer outputs
# (6 noisy early SR steps of H2 J1J2J3+MO; 8 steps of a 2-twist Li complex SD):
# rho >= 0.9985 (per-step row phase changes <= 1e-2 rad). One sign flip among N
# files gives rho = 1 - 2/N (0.9 for N = 20); a pi/4 phase jump in 1 of 20 gives
# 0.986. 0.99 refuses both while allowing a 14% RMS row spread
# (rho ~ 1 - sigma^2/2). Identical in full and mixed precision.
OPT_STEP_AVERAGE_MIN_MO_ROW_COHERENCE = 0.99

# Absolute band for recognizing exact SD/geminal representation structure
# during explicit conversion and load-time subclass dispatch.
GEMINAL_REPRESENTATION_ATOL = 1.0e-10

# S-Stiefel algebra is validated by a dimensionless normwise backward error,
# never by a storage-width-dependent absolute ``allclose`` band.  ``2*n`` is
# the dot-product operation count used in the standard gamma_k floating-point
# model; the safety factor covers the Hermitian eigensolver and reconstruction
# surrounding the final matrix product.  float64/complex128 therefore share
# the same real-component epsilon, as do float32/complex64.
S_STIEFEL_BACKWARD_ERROR_OPERATION_FACTOR = 2
S_STIEFEL_BACKWARD_ERROR_SAFETY_FACTOR = 8.0

# The optimizer metric must be nonsingular.  Eigenvalues below this fraction
# of lambda_max are lifted only after a PSD check has established that a
# negative value is round-off-sized; a materially negative overlap is a hard
# error rather than something hidden by regularization.
S_STIEFEL_METRIC_MIN_EIGENVALUE_RCOND = 1.0e-14

# The analytic periodic overlap is a one-lattice sum whose primitive-pair
# tail decays with rho = alpha*beta/(alpha+beta), generally more slowly than
# either AO envelope.  It therefore owns a cutoff separate from
# AO_IMAGE_TRUNCATION_ATOL. The key is the overlap's det-evaluation real
# precision, not the real/complex storage width of the resulting overlap.
_PGTO_LATTICE_OVERLAP_TERM_TOLERANCE_BY_REAL_DTYPE: dict[str, float] = {
    "float32": 1.0e-8,
    "float64": 1.0e-12,
}

# Composite AO overlap, pGTO x B-spline cross block:
# the exact reciprocal-space sum over m drops a term when
# prod_x sinc(m_x/N_x)^(n+1) * exp(-|q|^2/(4 alpha)) * (1 + |q|/(2 sqrt alpha))^l
# is below this value, i.e. below this fraction of its primitive group's m = 0
# scale. Set below the fp64 unit round-off so the block reaches fp64 accuracy in
# both precision modes (the overlap is a det_eval quantity).
COMPOSITE_CROSS_OVERLAP_SPECTRAL_TERM_TOLERANCE = 1.0e-17

# Regression-test tolerances for the exponent-only pGTO optimiser.  Recovery
# is limited by the finite reciprocal-space fit, whereas the condition comparison
# compares two fp64 evaluations of the same deployed overlap matrix.
RTOL_PGTO_EXPONENT_OPT_RECOVERY = 2.0e-3
RTOL_PGTO_CONDITION_AUDIT = 1.0e-10


def get_s_stiefel_backward_error_limit(dtype: npt.DTypeLike, dimension: int) -> float:
    """Return the dimensionless gamma_k bound for S-Stiefel algebra."""
    work = S_STIEFEL_BACKWARD_ERROR_OPERATION_FACTOR * int(dimension) * _real_finfo(dtype).eps
    if work >= 1.0:
        raise ValueError("S-Stiefel matrix dimension is too large for the selected floating-point dtype.")
    return float(S_STIEFEL_BACKWARD_ERROR_SAFETY_FACTOR * work / (1.0 - work))


def get_pgto_lattice_overlap_term_tolerance(dtype: npt.DTypeLike) -> float:
    """Return the primitive-pair tail target for the analytic lattice sum."""
    real_dtype_name = _real_finfo(dtype).dtype.name
    return _PGTO_LATTICE_OVERLAP_TERM_TOLERANCE_BY_REAL_DTYPE[real_dtype_name]


# Internal decisions whose numerical zero is defined relative to machine
# precision.  Store both multiplier and exponent here so scientific kernels
# contain no hidden ``100 * eps`` / ``eps**2`` policy.
_OPTIMIZER_ROUNDOFF_POLICIES: dict[str, tuple[float, int]] = {
    "lm_centered_overlap": (100.0, 1),
    "lm_active_diagonal": (1.0, 2),
    # Residual-pivot stop of the pivoted Cholesky factorization of the
    # shifted correlation matrix C - lm_cond * I (``scale`` = its dimension n).
    # Practical threshold scale, not a uniform round-off bound: the accepted
    # positive definite block has rows with sum of squares 1 - lm_cond <= 1,
    # so its residual diagonals carry rounding of order n * eps, but rejected
    # rows of the indefinite shifted matrix need not be bounded by 1.  The
    # scale is comparable to LAPACK's default rank-detection tolerance of
    # ?pstrf (N * eps * max diagonal, max diagonal <= 1 here).
    "lm_correlation_pivot": (1.0, 1),
    "asr_model": (100.0, 1),
    "force_standard_error": (1.0, 1),
}


def _real_finfo(dtype: npt.DTypeLike) -> np.finfo:
    """Return floating-point limits for the real component of *dtype*."""
    real_dtype = np.empty((), dtype=np.dtype(dtype)).real.dtype
    return np.finfo(real_dtype)


def get_optimizer_roundoff_tolerance(name: str, dtype: npt.DTypeLike, *, scale: float = 1.0) -> float:
    """Return one scale-aware optimizer round-off threshold."""
    try:
        multiplier, exponent = _OPTIMIZER_ROUNDOFF_POLICIES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown optimizer round-off policy: {name!r}.") from exc
    return float(multiplier * _real_finfo(dtype).eps ** exponent * scale)


# LM conditioning and pencil-solve verification (tests).  A computed
# correlation block or eigenpair can only be compared with an independent
# reference up to round-off, so the bounds below are derived from the same
# floating-point model as the production stop "lm_correlation_pivot".
#
# * lambda_min of a correlation block against lm_cond: the pivot stop itself
#   (n eps) plus the backward error of a symmetric eigensolver on a matrix with
#   ||C||_2 <= n (again n eps): 2 * n * eps.
# * LM eigenvector: the Cholesky reduction L^-1 H L^-T and the back-transform
#   x = L^-T y (and any independent dense generalized solver) carry a backward
#   error of order n eps kappa_2(S_A) relative to ||H||.  For a non-symmetric
#   pencil the resulting eigenvector error also depends on the non-normal
#   separation (LAPACK ``sep``), not only on the eigenvalue gap, so this is a
#   chosen comparison scale for the calibrated test pencils (well-separated,
#   mildly non-normal), not a uniform forward-error bound.  Tests express the
#   relative S-norm difference in units of n eps kappa_2(S_A)
#   (``get_lm_eigenvector_s_metric_unit``) and bound it by
#   ``ATOL_LM_EIGENVECTOR_S_METRIC_UNITS``; the factor 10 covers those pencils.
LM_CORRELATION_EIGENVALUE_SLACK_FACTOR = 2.0
ATOL_LM_EIGENVECTOR_S_METRIC_UNITS = 10.0


def get_lm_correlation_eigenvalue_slack(dtype: npt.DTypeLike, dimension: int) -> float:
    """Return the round-off slack for comparing lambda_min of a correlation block with ``lm_cond``."""
    return float(
        LM_CORRELATION_EIGENVALUE_SLACK_FACTOR
        * get_optimizer_roundoff_tolerance("lm_correlation_pivot", dtype, scale=float(dimension))
    )


def get_lm_eigenvector_s_metric_unit(dtype: npt.DTypeLike, dimension: int, overlap_condition_number: float) -> float:
    """Return ``n * eps * kappa_2(S_A)``, the round-off unit of a relative S-norm LM eigenvector difference."""
    return float(int(dimension) * _real_finfo(dtype).eps * float(overlap_condition_number))


def get_optimizer_breakdown_threshold(dtype: npt.DTypeLike) -> float:
    """Return the representability guard used for CG denominators."""
    return float(_real_finfo(dtype).tiny)


# Default truncation tolerance for the periodic AO image-list construction.
# Used by ``Structure_data.primitive_image_list``,
# ``_primitive_image_list*`` helpers (jqmc.structure), and
# ``_apply_image_expansion`` (jqmc.atomic_orbital)
# as the default ``tol`` for the per-primitive envelope cutoff
#     r_cut(alpha, L, tol)
# A primitive image T_n is kept when its closest approach to the home cell is
# at most r_cut, so this value controls the size of ``I_global`` (the global lattice-image set
# unioned over distinct ``(alpha_p, L_p)`` keys) and, by extension, the
# downstream kernel rank ``n_image * num_ao_molecular``.
#
# Tighter (smaller) tolerances grow ``n_image`` and increase memory /
# FLOP at AO evaluation; looser tolerances shrink it but truncate the
# periodic AO ``chi_tilde_mu = sum_n chi_mu(r - R_mu - T_n)`` at the
# corresponding floor. ``1e-7`` is the current default: it keeps ``n_image``
# (and the reverse-mode force-gradient memory footprint) tractable.
#
# Persisted per-instance as ``_image_truncation_tol`` (and the
# ``image_truncation_tol`` HDF5 group attr); the default below is what the
# loader falls back to when an old checkpoint omits the attr.
AO_IMAGE_TRUNCATION_ATOL = 1.0e-7

# Tiled source conversion compares different primitive/supercell image catalogs
# through third spatial derivatives and all raw strain components. The ordinary
# value-only cutoff does not bound those derivative tails: the Li TREXIO source
# has O(1e-6) strain tails at that cutoff. Retain a tighter per-instance catalog,
# persisted in HDF5, rather than loosening force/stress comparison tolerances.
TILING_PGTO_IMAGE_TRUNCATION_ATOL = AO_IMAGE_TRUNCATION_ATOL**2

# Auto-cutoff overlap-residual target for PW->pGTO+PW conversion
# (``--ecutwfc auto`` in ``jqmc-tool qe``). The geometric ladder of
# ``g_cut`` trials in :func:`jqmc.pw_pgto_projection.optimize_mixed_basis`
# stops at the first ``g_cut`` whose occupation-weighted mean overlap
# residual is at or below this value. Not exposed via the CLI on purpose:
# expert users who want a different fidelity target should set ``--ecutwfc``
# explicitly (in Rydberg) instead of tweaking this. Default ``2e-5`` --
# tight enough to push the residual below the sub-mHa VMC noise floor
# even on systems where the always-on pGTO tight extension
# (jqmc._setting.PGTO_TIGHT_*) does not fully absorb the high-G tail.
# A looser ``1e-4`` stop is adequate for many extended-basis cases but
# leaves residual VMC energy loss on some systems; the tighter target
# trades a few extra ladder steps for a more conservative fidelity
# guarantee.
PW_PGTO_AUTO_RESIDUAL_TARGET = 2.0e-5

# Default Gaussian-tail tolerance used by
# :func:`jqmc.pw_pgto_projection.alpha_min_from_gcut`. A pGTO primitive
# with exponent ``alpha`` has Fourier amplitude that decays as
# ``exp(-G^2 / (4 alpha))``; the alpha_min coupling drops primitives
# whose tail at ``|G| = g_cut`` is below this threshold, on the premise
# that the PW block at that ``g_cut`` already covers them. A looser
# (smaller) ``tail_tol`` keeps more primitives in the basis -- improving
# the projection residual at the cost of a larger pGTO basis and the
# corresponding VMC AO-evaluation overhead. Manual workflows that pin
# ``--alpha-min`` to a specific value bypass this constant; only callers
# that go through ``alpha_min_from_gcut`` (the auto path and any
# workflow that derives a fixed alpha_min via ``alpha_min_from_gcut``)
# see the change.
PW_PGTO_TAIL_TOL = 1.0e-3

# Exponent-tail bound of the exponent-only pGTO optimiser. A primitive of shell angular
# momentum l and exponent alpha places the fraction
# w_l(alpha) = Q(l + 3/2, g_max^2 / (2 alpha)) of its momentum-space norm
# outside the WFC sphere |G + k| <= g_max, where the spilling objective cannot
# see it.  ``optimize_pgto_exponents(exp_tail_eps=eps)`` bounds
# ln(alpha) <= ln(alpha_max(l, eps)) per shell l.
#
# PGTO_EXP_TAIL_CLIP_MARGIN: log-space margin below ln(alpha_max) to which an
#   infeasible seed exponent is clipped before SLSQP (trexio-tools value).
#   The post-optimisation audit compares ln(alpha) with ln(alpha_max) exactly
#   (the box clamp makes it exact), so it needs no tolerance.
PGTO_EXP_TAIL_CLIP_MARGIN = 1.0e-6
# PGTO_EXP_TAIL_CLIP_SPREAD_RATIO: when two or more seed primitives of the same
#   element shell exceed alpha_max, clipping all of them to the same value
#   makes the seed overlap exactly singular (e.g. Li ccECP s 190.08 and 5.938
#   at 30 Ha: cond 6e16).  They are instead placed on the geometric ladder
#   alpha_max * exp(-PGTO_EXP_TAIL_CLIP_MARGIN) / r^k, k = 0, 1, ..., with this
#   even-tempered ratio r (an algorithm input, not a tolerance), skipping a rung
#   within half a ladder step of an unclipped exponent of the shell.
PGTO_EXP_TAIL_CLIP_SPREAD_RATIO = 2.0

# Default
# exp_tail_eps of the user-facing entry points (``jqmc-tool qe
# --exp-tail-eps``, ``run_qe2jqmc`` qe_params, ``QE_Workflow``) is 1e-2;
# 0 (or None in run_qe2jqmc / QE_Workflow) disables the bound.
# ``optimize_pgto_exponents`` itself always defaults to None.
PGTO_EXP_TAIL_EPS_USER_DEFAULT: float | None = 1.0e-2

# When the bound clipped the
# seed and the SLSQP point is not adopted, return the clipped seed (True: it
# satisfies the bound) rather than the caller's original basis (False).
PGTO_EXP_TAIL_RETURN_CLIPPED_ON_REJECT = True

# Requested relative accuracy of the independent scipy.integrate.quad oracle
# of the exponent-tail weight test (an oracle input, not a comparison budget;
# the comparison uses RTOL_PYTHON_FP64_LOOSE and the achieved error on the
# test grid is <= 1e-14).
PGTO_EXP_TAIL_QUADRATURE_EPSREL = 1.0e-12

# Peak allocation budget for one analytic pGTO Fourier row chunk.  Production
# mixed-basis projection accumulates G^H G and G^H psi chunk by chunk instead
# of materialising the full ``n_fft * n_pgto`` matrix (67 GiB for 1024 pGTOs on a 160^3 grid).
# The overlap/rhs accumulators themselves are only O(n_pgto^2) / O(n_pgto*nmo).
PW_PGTO_FOURIER_CHUNK_BUDGET_BYTES = 256 * 1024**2

# Peak allocation budget for one block of primitive pairs in the molecular
# analytic Cartesian overlap (``atomic_orbital._compute_overlap_matrix_cart_analytic``).
# A block holds about ``ANALYTIC_OVERLAP_PAIR_WORK_ARRAYS`` live float64
# ``(bra primitives, ket primitives)`` arrays and is cut at AO boundaries, so
# the kernel's working set stays O(budget) independently of the AO count
# (measured: Ti2 cc-pVTZ, 902 primitives, peak 48 MiB).
ANALYTIC_OVERLAP_PAIR_BUDGET_BYTES = 64 * 1024**2
ANALYTIC_OVERLAP_PAIR_WORK_ARRAYS = 16

# Auto-sweep PW kinetic cutoff bounds, in Rydberg. The eight-point geometric
# ladder extends to the effective upper cutoff min(40 Ry, the QE cutoff) and
# starts at min(7.5 Ry, one quarter of the effective upper cutoff), so for QE
# cutoffs below 30 Ry the first trial is one quarter of the upper cutoff.
# The upper bound prevents unbounded mixed overlap matrices for high-cutoff
# QE references. If no trial meets the unchanged residual target, the CLI
# raises and requests an explicit numeric ecutwfc rather than silently
# accepting an insufficient basis. The 40 Ry ceiling covers the general-k
# conversion reported in issue #271.
PW_PGTO_AUTO_MIN_ECUTWFC_RY = 7.5
PW_PGTO_AUTO_MAX_ECUTWFC_RY = 40.0

# --- Tight-primitive extension for UPFExponentInit (always-on) -------------
# A UPF-seeded pGTO basis follows the PSWFC channels of the pseudopotential
# and typically tops out near ``alpha ~ 5`` -- which gives a Fourier reach
# of only ``|G| ~ 10`` Bohr^-1. The QE wfc carries content out to
# ``g_max_QE = sqrt(2 ecutwfc_QE)`` (often 15-25 Bohr^-1 for the meshes
# QMC consumes), so the un-extended pGTO basis cannot represent the high-G
# tail. Adding tight primitives via an even-tempered geometric extension
# at the tight end of every shell closes the gap. Applied unconditionally
# inside ``UPFExponentInit.shells_for_element``; all pGTO-bearing
# conversion kinds (``pgto``, ``pgto+pw``, ``pgto+bspline``) inherit the
# wider basis automatically.

# (NOTE) The target Fourier reach for the tight extension is NOT a
# global constant -- it is derived from each call site's QE reference
# (``g_target = sqrt(2 * ecutwfc_Ha)``) inside
# :class:`jqmc.upf_wrapper.UPFExponentInit`. Hardcoding a system-blind
# default would either over-extend (wasting basis for low-ecutwfc
# workflows) or under-extend (missing the high-G content for
# high-ecutwfc / AE workflows). Production code requires ``g_target``
# at ``UPFExponentInit.__init__``; tests pass explicit values.

# Even-tempered ratio between successive new primitives. 2.5 is the
# Schmidt-Ruedenberg standard observed in aug-cc-pV*Z and cc-pV*Z;
# empirically picked over the looser 4-5 commonly used for pure
# extension because (a) very short PSWFC ladders (e.g. H polarisation
# p with a single alpha ~ 0.3) need DENSER intermediate coverage to
# stay accurate at the high g_cut end of the auto sweep, and (b) the
# extra primitives (~16 across a small molecule) cost only a few
# percent in VMC AO evaluation while giving 3x better projection
# residual at cap=30 Ry. Methane benchmark:
#   ratio=2.5 -> +45 prims, trial-8 residual 3.5e-6 (MET);
#   ratio=5.0 -> +29 prims, trial-8 residual 1.05e-5 (just over target).
PGTO_TIGHT_RATIO = 2.5

# Cap on primitives added per shell. Prevents accidental basis-size
# explosion when the basis source already has a coarse alpha ladder; the
# extension is meant to refine the high-G end, not flood it.
PGTO_TIGHT_MAX_PER_SHELL = 3

# --- All-electron bare-Coulomb UPF (only H in practice) --------------------
# Bare-Coulomb AE UPFs carry only the 1/r local potential and an empty
# ``PP_PSWFC`` block, so the standard PSWFC-fit path
# (``fit_pswfc_channel_auto``) has nothing to fit. In practice the only
# element shipped this way is hydrogen (``H.coulomb-ae.UPF``); every
# other atom has a UPF with non-empty PSWFC channels and goes through the
# regular fit path. A Z-scaled tempered ladder that tries to be element-
# agnostic systematically lands wrong (the natural valence scale, the cusp
# tightness, and the diffuse tail vary independently with Z, so any single
# formula leaks energy through one shell or another), so we hardcode the
# H basis to a known-good chemistry-grade choice.
#
# Values are the cc-pVDZ uncontracted exponents for H: 3s + 1p =
# 6 cartesian AOs / atom. This matches the lower bound of chemistry-grade
# AE bases for QMC trial wavefunctions backed by PW residual.

# Cell-aware diffuse floor for the AE even-tempered ladder.
# Without this floor the ladder's most-diffuse primitives can have an
# e-fold extent that exceeds the simulation cell, producing per-cell
# image self-overlap of order ``2 * exp(-alpha * L^2)`` and driving the
# pgto LS overlap matrix toward singularity (cond(S) -> 1e8+). The
# floor caps the smallest alpha at
#   alpha_floor = AE_PGTO_CELL_SAFETY / L_min^2
# where ``L_min = min(|a|, |b|, |c|)`` is the shortest lattice vector.
# K_safety=4 gives one-image self-overlap ``2 exp(-4) ~ 3.7%`` at the
# floor (Gaussian e-fold = L_min / 2), so cross-cell mixing stays
# bounded and the LS conditioning is governed by the well-conditioned
# alpha range, not by the cell-wrapping tail. The ladder is then
# truncated (n_primitives shrinks) to respect the floor. Bigger cells
# automatically retain more diffuse primitives; tight cells trim more.
# Pass ``cell_min_edge = 0`` (or omit) at the call site to disable the
# floor and recover the original behaviour.
AE_PGTO_CELL_SAFETY = 4.0

# Eigenvalue floor for canonical-orthogonalisation pseudoinverse used
# by every projection path in :mod:`jqmc.pw_pgto_projection`
# (``_project_from_S_b``, ``_project_from_blocks``, the manual /
# auto / pgto-only entry points, and the CLI default in
# ``jqmc_tool``). Eigenvalues of the (n_gto + n_pw) overlap matrix
# below ``rcond * lambda_max`` are zeroed out so the projection
# pseudoinverse stays finite when the basis is near-redundant. CLI
# users can override per-call via ``--max-cond-number`` (the value
# fed there is ``1 / rcond``); this constant is the system-wide
# default if nothing is passed. 1e-8 corresponds to cond(S) <= 1e8 --
# matches the conventional pGTO basis quality criterion in
# fit_pswfc_channel_auto / _gaussian_radial_overlap_cond.
PW_PGTO_SVD_RCOND = 1.0e-8

# Shared worker count for the multi-core paths in the QE -> jQMC
# converter (:mod:`jqmc.pw_pgto_projection`):
#
#   * ``build_gto_design_matrix`` -- joblib thread pool over the
#     per-(center, alpha) group loop. Each group writes to a disjoint
#     output column slab; the dominant kernel (``np.exp`` + polynomial
#     accumulate) is pure-numpy element-wise so BLAS / OpenMP cannot
#     help, but the GIL is released and threads scale.
#   * ``reconstruct_pw_on_grid`` / ``_gto_fourier_at_millers`` /
#     ``bspline_prefilter_periodic`` --
#     ``scipy.fft.{fft, ifft, fftn, ifftn}(..., workers=...)`` replaces
#     the single-threaded ``np.fft.*``. Same FFT backend (pocketfft) so
#     numerical results match to round-off.
#
# Both call sites use the same ``-1 = all cores`` / ``1 = serial`` /
# ``N = N workers`` semantics. Override in user code via
# ``jqmc._setting.PW_PGTO_N_WORKERS = ...`` before importing the
# converter, or pass the explicit kwarg to ``build_gto_design_matrix``.
PW_PGTO_N_WORKERS = -1

# Byte budget for the LARGEST single per-band intermediate of the PW->B-spline
# fidelity diagnostic (``project_pw_to_spline`` / ``project_pw_to_pgto_spline``
# in ``pw_pgto_projection.py``) and of ``evaluate_spline_orbitals``'s internal
# axis-0 gather. The residual r_n is per-band independent, but an unchunked
# evaluation materialises ALL bands on the FULL evaluation grid at once: at
# 501 bands and FFT 300^3 that is a (501, 1800, 150, 150) complex128
# gather -- a 302 GiB numpy ArrayMemoryError -- plus three (27M, 501) grid
# arrays of ~216 GiB each, while the actual product (the control-point table)
# is only ~27 GiB. The band-chunk loop bounds the per-band intermediates at
#     chunk = clamp(BUDGET // bytes_per_band, 1, n_band)
# so peak scales with the budget, not with n_band. Host-side (numpy,
# conversion-time) knob. The FFT, the spline evaluation and the band-subset
# selection are bit-identical per band under chunking (measured); only the
# final ``einsum("pn,pn->n")`` reduction order depends on the column count,
# so chunked vs unchunked residuals agree to reduction round-off
# (~4e-14 relative measured on H-ae; asserted in
# ``test_pw_pgto_projection.py``). ``BUDGET`` large enough for one chunk
# degenerates to the all-bands evaluation.
#
# Default 256 MiB: an 8 GiB budget never binds in ANY of the 30 measured
# configurations -- the worst measured shape (n_band=32, spline 128^3)
# reached a 15.4 GiB host peak and still ran unchunked. And for THIS diagnostic
# chunking is NOT a memory<->speed tradeoff: the fidelity working set fits in
# cache when chunked, so smaller chunks are BOTH smaller and faster
# (4.5-13.5x less peak AND 1.05-2.30x faster across 7 real QE saves + 7
# synthetic shapes, bit-identical everywhere). 256 MiB puts every measured
# real save in that both-win regime while the clamp floor (chunk >= 1) keeps
# arbitrarily large per-band slabs runnable. The FIT-side knob
# (PW_SPLINE_FIT_CHUNK_BUDGET_BYTES) is kept at 8 GiB: chunking the fit
# measured ~0.83x SLOWER for only ~3x memory (the output table cannot be
# split).
PW_SPLINE_FIDELITY_CHUNK_BUDGET_BYTES = 256 * 1024**2

# Byte budget for the per-band intermediates of the PW->B-spline COEFFICIENT
# BUILD (``fit_pw_to_spline``: scatter + inverse FFT + periodic prefilter).
# Distinct from the fidelity budget above: that one bounds a DIAGNOSTIC whose
# result is a scalar per band, this one bounds the path that produces the
# control-point table the run actually needs.
#
# An unchunked chain holds every band at once and makes ~6 full-size copies of
# the (n_band, *spline_shape) cube. At 501 bands and spline 150^3 (mesh_factor
# 0.5 => 54 MB/band, 27 GiB/copy):
#     reconstruct_pw_on_grid : zero cube 27 + out-of-place ifftn 27   =  54 GiB
#     bspline_prefilter      : c + fft + divide + ifft, PER AXIS      = ~108 GiB
#     project_pw_to_spline   : all-band 27 + occupied copy 23 + astype 23
# i.e. a ~110 GiB peak for a 23 GiB product, which needs a >100 GB node.
# Bands are INDEPENDENT through the whole chain, so the
# loop bounds live memory at ``chunk`` bands:
#     chunk = clamp(BUDGET // (COPIES * 16 * prod(spline_shape)), 1, n_band)
# with COPIES the measured simultaneous-copy count of the prefilter (4).
# At this size an 8 GiB budget gives chunk ~= 37 bands, peak ~8 GiB of scratch on top of the
# 23 GiB output. Bit-exact: scipy's FFT batches bands independently (the
# per-band transform does not depend on how many rows share the call --
# verified in ``test_pw_pgto_projection.py``), and the prefilter is per-axis
# 1D deconvolution. A budget large enough for one chunk degenerates to the
# all-bands form. Host-side (numpy, conversion-time) knob.
PW_SPLINE_FIT_CHUNK_BUDGET_BYTES = 8 * 1024**3

# G-sphere taper of the QE plane-wave coefficients (conversion time only).
#
# QE stores each band as a SHARP sphere |G+k| <= G_max = sqrt(2 ecutwfc). The
# finite sum is entire in r, but its truncation remainder does not decay with
# the distance from the nuclei, so far from every nucleus the orbital's
# exponential tail falls below a truncation-ringing (Gibbs) floor that
# oscillates with wavenumber ~G_max, where the local kinetic energy
# -1/2 lap(psi)/psi is O(+-G_max^2/2) of either sign (~300 Ha at 600 Ry;
# measured median |T_loc| 280-310 Ha on a molecule-on-surface slab). LRDMC regularizes
# only nodal divergences, so one walker diffusing into such a region
# multiplies its weight by exp(+15 ... +54) per branching step and the
# population collapses (measured on that slab: e_L -7e3 Ha, weight
# 8e55, E = nan). The cure is smoothness of the spectrum at the edge, not a
# lower cutoff: a hard cut at 0.9 or 0.7 G_max RAISES the floor (5e-5 ->
# 1.4e-4 -> 8e-4) because the floor follows the coefficient size at the jump.
#
# ``taper_qe_reference`` multiplies every coefficient by the C-infinity smooth
# step w(|G+k|) that is 1 up to PW_TAPER_START_FRACTION * G_max and 0 at G_max
# with all derivatives (for a ``pw`` basis truncated at g_cut < G_max the edge
# is g_cut). The windowed remainder decays super-algebraically, not
# exponentially: the local kinetic energy is restored wherever the physical
# tail still exceeds that remainder (on that slab: the floor 3.6e-7 of the band
# maximum becomes a smooth 1.7e-8 background with k <= 1 bohr^-1), and nothing
# bounds it where an orbital is below ~1e-8 of its maximum. The occupied
# spectral weight above 0.8 G_max is ~1e-7 at 600 Ry (same slab); as a
# first-order, single-determinant diagnostic its G^4-weighted share of the
# Laplacian power is ~1e-4, consistent with the measured ~0.1 Ha spread of the
# per-configuration local energy (an L2 statement, not a pointwise bound), while
# the total energy moves far less. The per-band spectral loss is logged and
# recorded at conversion time.
PW_TAPER_START_FRACTION = 0.8

# The taper is applied ONLY when the cell has a vacuum region, i.e. a point
# farther than this from every nucleus: the covering radius of the periodic
# nuclei, computed exactly from the Voronoi vertices of the wrapped atoms and
# their images (``max_empty_distance``), so the decision does not depend on the
# origin, on the primitive basis or on the lattice representatives of the
# atoms. Cells without such a point (bulk Si/Li, layered bulk with ~3 bohr
# van der Waals gaps) keep their
# coefficients bit-identical: there the exponential orbital tail never reaches
# the ringing floor, which on that slab at 600 Ry starts ~10 bohr from the O nucleus
# for the lowest band and ~14 bohr for the HOMO: 10 bohr is where the floor
# begins to dominate the deepest bands. Slabs with >= 20 bohr of vacuum and
# molecules in large boxes are tapered; thin-vacuum slabs and small isolated-
# atom boxes (e.g. one atom in a 10 bohr cube, corner 8.66 bohr away) are not:
# there the tails still exceed the floor by >= 1e2 and the local-energy
# contamination stays at the Hartree level, far from a weight blow-up. The
# decision follows the geometry alone: there is no user option, so the same
# system is always converted the same way.
PW_TAPER_VACUUM_RADIUS_BOHR = 10.0

# Relative tolerance on the local kinetic energy -1/2 lap(psi)/psi of a tapered
# plane-wave sum against the exactly known value of a manufactured tail
# (tests only). The oracle is the periodized Gaussian-smoothed exp(-|z|)
# (L = 48 bohr, sigma = 0.15 bohr, G_max = 20 bohr^-1), whose local kinetic
# energy is exactly -1/2 Ha away from the source. Measured on a fine scan: the
# sharp-sphere sum exceeds 100 % error from 1.1e-3 of the maximum on and reaches
# tens to hundreds of Hartree below 3e-4; the tapered sum stays within this
# fraction of -1/2 Ha over the whole continuous range down to 2e-4 of the
# maximum (z <= 8.5 bohr) and at the sampled points 3.4e-4, 4.5e-5 and 6.1e-6
# (z = 8, 10, 12), but between 2e-5 and 6e-6 its deviation reaches 26-28 %, and
# beyond ~1e-7 the windowed remainder dominates and neither sum bounds the
# local energy.
RTOL_PW_TAPER_VACUUM_KINETIC = 0.1

# (NOTE) FFT-grid resolvability of tight primitives is checked at
# projection time inside :func:`jqmc.pw_pgto_projection.optimize_mixed_basis`
# against the actual cell + FFT grid (Gaussian width sigma = 1/sqrt(2 alpha)
# must clear the grid spacing dx; alpha_max_safe = 1/(2 dx^2)). A WARNING
# is emitted there when any primitive exceeds the limit. The check runs at
# projection time because an absolute alpha cap (e.g. 100) would be both
# system-blind (true safe alpha depends on cell/grid) AND always dominated
# by the G_target-derived alpha_target in practice.


# ---------------------------------------------------------------------------
# MPC (Model Periodic Coulomb) -- jqmc/mpc.py
# ---------------------------------------------------------------------------
# Reciprocal-space kinetic-energy cutoff (Ha) for the MPC f_G / density FFT
# grid. This is the *only* numerical knob of canonical MPC: the explicit
# two-body term is the minimum-image 1/r, so there is no Ewald split
# parameter to choose (an alpha-Ewald form has one, and its result depends
# on it).
#
# 30.0 matches CASINO's ``mpc_cutoff`` and QMCPACK's ``cutoff`` defaults, so
# numbers are comparable against those codes out of the box. It is a
# *convergence* parameter, not a tuning one: double it until the correction
# <V_ee^MPC - V_ee^Ewald> stops moving.
#
# Calibration: a water monomer in a 15.12 Bohr cell gives 44.27 / 42.76 /
# 42.20 mHa at 30 / 60 / 120 Ha -- monotone, with the increments shrinking
# geometrically (~2.7x per doubling) toward ~41.8 mHa. So the default
# carries ~6% (2.5 mHa) on that system and 120 Ha still leaves ~0.4 mHa.
# Systems targeting mHa accuracy should raise it and check convergence
# explicitly rather than trusting the default.
MPC_CUTOFF_HA = 30.0

# Real-space tabulation grid for V_LR, in units of the highest retained
# integer frequency m_d = ceil(sqrt(2 Ecut) / |b_d|) along each reciprocal
# axis. 2 m_d + 1 points would already REPRESENT the band-limited field;
# the tricubic B-spline that evaluates it between nodes needs the field
# oversampled to be accurate where the electrons actually are (the sharp
# V_LR dip at each nucleus). These are QMCPACK's choices (MPC.cpp:
# ``SplineDim = 4 * MaxDim`` in init_spline, ``N = max(64, 2 MaxDim + 1)``
# in init_f_G). With the Nyquist grid and trilinear interpolation the MPC
# correction of a 96-electron periodic cell is -147 mHa against QMCPACK's
# +103 mHa (near-nucleus bias -2.3 mHa/electron); at 4x + cubic the
# interpolation bias is below a micro-Hartree per electron.
MPC_GRID_POINTS_PER_MAX_FREQUENCY = 4
MPC_GRID_MIN_POINTS = 64

# Upper bound on the MEAN signed error (Ha per evaluation point) of the
# tabulated-V_LR interpolator against the exact trigonometric
# reconstruction of the same band-limited field, sampled where the
# electrons are (Gaussian clouds around the nuclei) and uniformly. The
# mean, not the RMS, is what matters: the sum over N_e electrons turns a
# per-point bias b into an N_e * b shift of <V_MPC - V_Ewald>, whereas
# a zero-mean scatter averages out in the Monte Carlo estimate. With
# the grid fixed at MPC_GRID_POINTS_PER_MAX_FREQUENCY points per
# retained frequency the node spacing times the highest wavenumber is
# a cutoff-independent pi / 2, so a low-cutoff measurement is
# representative of production.
#
# Calibration on H2_pbc_ae_cart.h5 (all-electron protons, 11.34 Bohr
# cube, Ecut = 9 Ha, grid 32^3 = 4 m): tricubic B-spline on the 4x grid
# +4.1e-7 Ha near the nuclei (rms 5.1e-7, uniform -8.5e-10); trilinear
# interpolation on the Nyquist grid (18^3) gives +5.6e-4, i.e. more
# than two orders of magnitude above this bound, and -2.3e-3 per
# electron on a ccECP molecular crystal. The bound is 5x the measured value.
ATOL_MPC_VLR_INTERPOLATION_MEAN_BIAS = 2.0e-6

# FIXTURE REGRESSION BOUNDS, NOT FIRST-PRINCIPLES ACCURACY BOUNDS. Maximum
# absolute error (Ha; paired with
# RTOL_PYTHON_FP64_ZERO) of the production MPC V_LR table against an
# independent reference, valid ONLY for the single fixed fixture of
# tests/test_jqmc_kernel/test_mpc_independent_reference.py: physical box
# diag(5, 6, 7) Bohr in its orthorhombic and integer-shear representations,
# one 3-plane-wave MO doubly occupied, cutoff 1.3 Ha, explicit 16^3 grid.
# The reference is a box-face Gauss-Legendre kernel integral plus a NumPy
# plane-wave density; it shares no Hood split, minimum-image scan, FFT
# extrapolation or B-spline lookup with production. Measured maxima (both
# precision modes, both representations): table 6.12e-7, off-grid lookup
# 1.29e-6; the bounds are those maxima rounded up to the next
# 1-2-5 value. No a-priori error bound exists for the table (Hood/FFT
# discretisation), so these numbers do not promise accuracy for any other
# cell, density or grid. They were adopted only because every injected
# table-construction fault measured (wrong Hood split, Hood sign
# or G=0 limit, dropped or truncated FFT extrapolation, wrong cell volume,
# disabled minimum image) exceeds them by at least 45x in both modes.
ATOL_MPC_TABLE_FIXTURE_REGRESSION = 1.0e-6
ATOL_MPC_OFFGRID_FIXTURE_REGRESSION = 2.0e-6

# FIXTURE REGRESSION BOUND, NOT A FIRST-PRINCIPLES ACCURACY BOUND (issue #803).
# Maximum absolute error (Ha; paired with RTOL_PYTHON_FP64_ZERO) of the
# production MPC V_LR table nodes, their lookup at table nodes with whole-cell
# offsets, and the canonical MPC e-e energy of four electrons on table nodes,
# against the independent reference of the OPEN-SHELL fixture of
# tests/test_jqmc_kernel/test_mpc_independent_reference.py only: the same
# diag(5, 6, 7) Bohr box (orthorhombic and integer-shear basis), box-face
# kernel, cutoff 1.3 Ha and explicit 16^3 grid as the table fixture above, but
# a 3 up / 1 down density of four orthonormal orbitals (complex twisted plane
# waves, real Gamma plane waves, or implicit-identity cubic B-splines). Its
# density contrast is larger than the doubly occupied table fixture, so the
# same Hood/FFT discretisation error is larger. Measured maxima over every case
# in both precision modes: table 1.91e-6, lookup 1.46e-6, energy 1.67e-6; the
# bound is the next 1-2-5 value above 2x the largest. Dropping the unpaired
# (singly occupied) orbitals from the MPC density (AGP, AGPn or identity-MO
# occupations) changes the table by >= 1.8e-2 and the lookup / energy by
# >= 8.8e-3 in both modes, i.e. >= 1700x the bound (issue #803).
ATOL_MPC_OPEN_SHELL_FIXTURE_REGRESSION = 5.0e-6

# Relative agreement required between canonical-MPC corrections built on nearby
# FFT grid sizes.  This deliberately loose bound is a discretisation-stability
# guard: the grids are not expected to agree at fp round-off, but no individual
# even/odd grid may produce a discontinuously different finite-size correction.
RTOL_MPC_GRID_DISCRETIZATION_STABILITY = 2.5e-1


# --- jaxlib persistent-compilation-cache load overflow ---
# jaxlib 0.9.1 and 0.9.2 load an executable from JAX's persistent compilation cache
# on a thread-pool worker with a 512 KiB stack, and ORC links the cached object
# files recursively (about eight frames per object file). A large executable
# overflows that stack: the process dies with SIGSEGV and no Python traceback.
# Measured (same executable, shared cache directory): 0.9.1 and 0.9.2 crash on the
# cache hit; 0.7.2, 0.8.0, 0.9.0, 0.10.0, 0.10.1 and 0.10.2 load it and pass.
# ``[first affected, first fixed)``. Single source of truth for the jqmc runtime
# warning below and for the test-harness gate in ``tests/conftest.py``.
JAXLIB_PERSISTENT_CACHE_LOAD_OVERFLOW = ((0, 9, 1), (0, 10, 0))

# JAX's boolean flag parser treats these environment strings as false.
_JAX_FALSE_STRINGS = ("0", "false", "f", "n", "no", "off")

_persistent_cache_warning_emitted = False


def jaxlib_persistent_cache_load_overflows(jaxlib_version: str | None) -> bool:
    """Whether this jaxlib release can overflow a worker stack loading a cached executable.

    ``jaxlib_version`` is a version string such as ``"0.9.2"``; local or dev
    suffixes are ignored. ``None`` or an unparsable string is not affected.
    """
    import re

    match = re.match(r"(\d+)\.(\d+)\.(\d+)", jaxlib_version or "")
    if match is None:
        return False
    release = tuple(int(part) for part in match.groups())
    first_affected, first_fixed = JAXLIB_PERSISTENT_CACHE_LOAD_OVERFLOW
    return first_affected <= release < first_fixed


def installed_jaxlib_version() -> str | None:
    """The installed jaxlib distribution version, read without importing jax."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("jaxlib")
    except PackageNotFoundError:
        return None


def persistent_compilation_cache_enabled(config=None, environ=None) -> bool:
    """Whether JAX's persistent compilation cache is enabled.

    Enabled means a cache directory is set (``jax_compilation_cache_dir`` in
    ``config`` or ``JAX_COMPILATION_CACHE_DIR`` in ``environ``) and the cache is
    not switched off (``jax_enable_compilation_cache`` false in ``config`` or
    ``JAX_ENABLE_COMPILATION_CACHE`` false in ``environ``). ``config`` defaults
    to ``jax.config`` and ``environ`` to ``os.environ``.
    """
    import os

    if environ is None:
        environ = os.environ
    if config is None:
        import jax

        config = jax.config
    cache_dir = getattr(config, "jax_compilation_cache_dir", None) or environ.get("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        return False
    if getattr(config, "jax_enable_compilation_cache", True) is False:
        return False
    return environ.get("JAX_ENABLE_COMPILATION_CACHE", "").strip().lower() not in _JAX_FALSE_STRINGS


def warn_if_persistent_cache_load_can_overflow(jaxlib_version=None, config=None, environ=None) -> bool:
    """Log one warning when the installed jaxlib can crash loading a cached executable.

    Fires at most once per process, only when ``jaxlib_version`` (default: the
    installed jaxlib) is in ``JAXLIB_PERSISTENT_CACHE_LOAD_OVERFLOW`` AND the
    persistent compilation cache is enabled (see
    :func:`persistent_compilation_cache_enabled`). Returns whether it warned.
    Never raises and changes no setting: it only reports.
    """
    global _persistent_cache_warning_emitted

    try:
        if _persistent_cache_warning_emitted:
            return False
        if jaxlib_version is None:
            jaxlib_version = installed_jaxlib_version()
        if not jaxlib_persistent_cache_load_overflows(jaxlib_version):
            return False
        if not persistent_compilation_cache_enabled(config=config, environ=environ):
            return False
        import logging

        affected = ".".join(map(str, JAXLIB_PERSISTENT_CACHE_LOAD_OVERFLOW[0]))
        fixed = ".".join(map(str, JAXLIB_PERSISTENT_CACHE_LOAD_OVERFLOW[1]))
        logging.getLogger(__name__).warning(
            "jaxlib %s with the JAX persistent compilation cache enabled: jaxlib >= %s and < %s can crash the "
            "process silently (SIGSEGV, no Python traceback) when it loads a large executable from the cache, "
            "because the cache-load thread overflows its stack. Upgrade to jaxlib >= %s, or set "
            "JAX_ENABLE_COMPILATION_CACHE=false.",
            jaxlib_version,
            affected,
            fixed,
            fixed,
        )
        _persistent_cache_warning_emitted = True
        return True
    except Exception:  # noqa: BLE001 -- a diagnostic must never break the import that runs it
        return False


# --- FAST-mode molecular AO screen bound (native Rust FAST chunk) ---
#
# Every FAST-mode molecular AO screen in the native backend (value screens:
# per-move orbital evaluation, ECP mesh AO contraction, GFMC candidate batch;
# derivative screens: FAST kinetic / J3 / GFMC derivative cache) may drop only
# AO values, gradient components and Laplacians below this ABSOLUTE bound, in
# AO units (normalization and contraction coefficients included; issue #100).
# This is a physical property of the
# screens, not a comparison tolerance; it mirrors FAST_SCREEN_ABS_BOUND in
# jqmc/rs/src/atomic_orbital.rs, and tests assert dropped terms against it.
FAST_AO_SCREEN_ABS_BOUND = 1.0e-12


# --- FAST-mode radial cut of the ECP non-local quadrature (native Rust FAST chunk) ---
#
# In the FAST real energy chunk (jqmc/rs/src/coulomb_potential.rs
# compute_ecp_nonlocal_parts_all_pairs with the mesh cutoff and the
# Sherman-Morrison inverse) an (electron, non-local term) pair whose radial
# factor |v_t(r)| = |c r^(p-2) exp(-e r^2)| is below this bound, in Ha, skips its
# whole quadrature. That evaluates exactly a modified Hamiltonian H' whose
# semilocal radial functions are truncated where |v_t| < EPS, so pointwise
# |dv_t| <= EPS; the rotation-averaged quadrature is the angular projector
# (norm <= 1), so ||H - H'|| <= N_e * n_t * EPS (n_t = non-local terms per
# nucleus) and every energy expectation (VMC, fixed-node) moves by at most that.
# Water 20 (160 electrons, one non-local term per O): <= 1.6e-10 Ha. The
# per-sample local energy is NOT bounded by EPS (the mesh psi ratio is
# unbounded near nodes). The exact route never cuts; XLA is unchanged. This is a
# physical property of the FAST route, not a comparison tolerance; it mirrors
# FAST_ECP_NONLOCAL_RADIAL_CUT in jqmc/rs/src/coulomb_potential.rs.
FAST_ECP_NONLOCAL_RADIAL_CUT = 1.0e-12


# --- Complex FAST chunk: ECP state on the carried determinant base ---
#
# The complex-twist FAST energy chunk seeds its per-e_L ECP non-local mesh state
# on the sweep's carried base (orbital tables + running Sherman-Morrison
# inverse) instead of a fresh orbital evaluation + geminal + inverse
# (IMPL-RS-CPLX-ECP-CARRIED-SEED-001). Same determinant, same positions; the
# two seeds differ only by the SM-carry rounding of num_moves rank-1 updates
# from a fresh per-step inverse. Comparison tolerance for carried vs fresh
# stored e_L on the same walk (full precision; mixed precision raises it to the
# ao_eval zone envelope). Measured on the QE Li k=(1/4,1/4,1/4) ccECP fixtures
# (pGTO J1/J2, J1/J2/J3, DTN; pGTO+PW and B-spline without Jastrow; two
# walkers, seven steps): max |delta e_L| = 2.4e-15, max relative = 8.3e-15.
# These values keep ~2.5 orders of headroom over that spread.
ATOL_CPLX_ECP_CARRIED_SEED = 1.0e-12
RTOL_CPLX_ECP_CARRIED_SEED = 1.0e-12


# --- Complex FAST chunk: carried kinetic tables and carried determinant base ---
#
# The complex-twist FAST energy chunk carries, across the steps of one chunk,
# (a) the kinetic derivative tables (complex GFMC det / J1-J2 / DTN / J3 caches,
# advanced by replaying the sweep's accepted moves, with their own fused-value
# inverse) instead of the fresh per-e_L kinetic, and (b) the determinant base
# (geminal, SM inverse, DetRatioStateC) instead of a per-step rebuild
# (IMPL-RS-CPLX-CHUNK-KIN-CARRY-001). Same quantities at the same positions; the
# routes differ only by rank-1 carry rounding and reduction order. Comparison
# tolerances for carried vs fresh stored e_L / w_L on the same walk (full
# precision; mixed precision raises them to the zone envelopes). Measured on the
# QE Li k=(1/4,1/4,1/4) ccECP fixtures (pGTO J1/J2, J1/J2/J3, DTN; pGTO+PW and
# B-spline without Jastrow; production driver, two walkers, seven steps):
#   kinetic carry vs fresh kinetic:   max |delta e_L| = 3.8e-15, max rel = 1.7e-14
#   base carry vs per-step rebuild:   max |delta e_L| = 6.0e-15, max rel = 1.7e-14,
#                                     w_L byte-identical, walk byte-identical
#   N-step chunk vs N 1-step chunks:  max |delta e_L| = 4.0e-16
# (mixed precision, for reference: kinetic carry max |delta e_L| = 4.0e-7, max rel =
# 1.1e-6 -- the fresh kinetic contracts fp64 derivative tables against the
# ao_eval-zone value-chain inverse, the carried tables against their own fp64
# inverse; inside the ao_eval zone envelope.) These values keep ~1.8 orders of
# headroom over the full-precision spread, leaving room for the longer rank-1
# chains of production chunks (up to 500 steps).
ATOL_CPLX_KIN_CARRY = 1.0e-12
RTOL_CPLX_KIN_CARRY = 1.0e-12
ATOL_CPLX_DET_BASE_CARRY = 1.0e-12
RTOL_CPLX_DET_BASE_CARRY = 1.0e-12


# --- Mixed-precision AO value range precondition (issue #111) ---
#
# In mixed precision the standalone AO value kernels (XLA ``_compute_AOs`` and
# the Rust f32 value kernel) evaluate every primitive as the product of
# electron-independent and coordinate factors in the ``ao_eval`` dtype:
# the normalization N (formed at storage precision and rounded once), the
# radial factor N c exp(-a r^2) and the angular polynomial (x^nx y^ny z^nz or
# the real solid harmonic, bounded by r^l). Issue #111: a basis whose factors leave the ``ao_eval`` range is
# rejected loudly when it is validated, instead of returning silently wrong
# values. The bounds below are derived from the ``ao_eval`` dtype itself:
#   ln_min / ln_max   -- natural logs of the smallest normal and largest finite
#                        value (subnormals are flushed or lose relative accuracy).
#   ln_depth          -- -ln(eps): the AO's "significant shell" is the radial
#                        range where |r^l exp(-a r^2)| is within a factor eps of
#                        its maximum; outside it the AO is below the dtype's
#                        resolution relative to its own peak.
#   ln_negligible     -- ln of the strict ``ao_eval`` absolute tolerance of the
#                        dtype (``_TOLERANCE``): a factor that underflows may drop
#                        only AO values below this absolute bound.


def get_ao_eval_value_range_bounds(dtype: npt.DTypeLike) -> tuple[float, float, float, float]:
    """Return ``(ln_min, ln_max, ln_depth, ln_negligible)`` for the AO evaluation dtype."""
    info = _real_finfo(dtype)
    negligible = _TOLERANCE["strict"][info.dtype.name][0]
    return (
        float(np.log(info.tiny)),
        float(np.log(info.max)),
        float(-np.log(info.eps)),
        float(np.log(negligible)),
    )
