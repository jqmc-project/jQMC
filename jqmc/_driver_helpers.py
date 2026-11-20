"""Shared driver helpers for the MCMC / GFMC / CLI loops.

Multi-caller utilities lifted out of the drivers: the JAX distributed
bootstrap over MPI, the MPI/JAX distribution consistency check, peak-memory logging, and initial electron-configuration
generation (production + debug).

Precision Zones:
    This module owns no Precision Zone.

See :mod:`jqmc._precision` for details.
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

import os
import resource
import socket
import sys
from collections.abc import Mapping
from logging import DEBUG, INFO, getLogger
from typing import Any

import jax
import numpy as np
import numpy.typing as npt
import toml
from mpi4py import MPI

from ._setting import ATOL_DRIVER_ELECTRON_POSITION_DISTINCTNESS

# set logger
logger = getLogger("jqmc").getChild(__name__)

# separator
num_sep_line = 66


def _init_external_control_file(toml_filename: str, world_comm: MPI.Intracomm) -> None:
    """Create the single per-invocation external-control stop file.

    The control file is a process-global artifact: exactly one process -- the
    root of the *world* communicator -- owns its lifecycle. This matters for
    twist-averaged runs, where ``world`` is split into one intra-comm per twist
    and every twist's intra-root would otherwise believe it is "rank 0" and
    race on the same shared path (two writers create it, two removers trip a
    ``FileNotFoundError`` TOCTOU). Gating on the world root keeps a single
    user-visible file written exactly once.

    Callers keep their own intra-comm ``Barrier()`` after this call (so every
    rank in a twist agrees the create attempt has happened); no world-level
    barrier is taken here, so this never deadlocks when ranks call ``run`` an
    unequal number of times. Readers tolerate the file's temporary absence via
    :func:`_read_external_stop_flag`.
    """
    if world_comm.Get_rank() == 0:
        if os.path.exists(toml_filename):
            logger.info(f"{toml_filename} exists, overwriting it.")
        with open(toml_filename, "w") as f:
            logger.info(f"{toml_filename} is generated. ")
            toml.dump({"external_control": {"stop": False}}, f)
        logger.info("")


def _read_external_stop_flag(toml_filename: str, comm: MPI.Intracomm) -> bool:
    """Return the external-control stop flag, shared across ``comm``.

    Only ``comm`` root reads the file (avoids filesystem-cache inconsistency
    between ranks), then broadcasts so every rank agrees on whether to break --
    a disagreement would otherwise deadlock the next collective. EAFP: a
    missing, partially written, or malformed file is treated as "no stop
    requested" (e.g. when the world root has already removed it while another
    twist is still looping).
    """
    if comm.Get_rank() == 0:
        try:
            with open(toml_filename) as f:
                stop_flag = bool(toml.load(f)["external_control"]["stop"])
        except (FileNotFoundError, KeyError, toml.TomlDecodeError):
            stop_flag = False
    else:
        stop_flag = False
    return comm.bcast(stop_flag, root=0)


def _remove_external_control_file(toml_filename: str, world_comm: MPI.Intracomm) -> None:
    """Remove the external-control file; only the world root removes it.

    Mirrors :func:`_init_external_control_file`: a single owner avoids the
    cross-process TOCTOU race where two twist roots both pass an ``isfile``
    check and then both call ``os.remove``. The removal is EAFP-guarded so a
    concurrent delete is harmless. Callers keep their own intra-comm
    ``Barrier()`` before this call.
    """
    if world_comm.Get_rank() == 0:
        try:
            os.remove(toml_filename)
            logger.info(f"Delete {toml_filename}")
        except FileNotFoundError:
            pass


def derive_rank_rng_streams(chain_seed: "int | tuple | list | np.ndarray", mpi_rank: int) -> tuple[np.ndarray, int]:
    """Hierarchical SeedSequence-based per-rank RNG derivation.

    This is the SINGLE derivation point for every sampler's RNG state.
    A stream is identified by the integer key ``[*chain_seed_words, mpi_rank]``:

    * non-twist / single-twist runs: ``chain_seed`` is the user's scalar
      ``mcmc_seed`` -> stream key = ``[mcmc_seed, rank]``;
    * twist-averaged runs: the twist drivers pass
      ``chain_seed = (mcmc_seed, twist_index)`` (see
      ``jqmc._twist_average._twist_sampler_seed``) -> stream key =
      ``[mcmc_seed, twist_index, intra_rank]``.

    Nothing is ever folded into a single uint32 along the way: the key words
    feed ``np.random.SeedSequence`` directly, whose spread function keeps
    distinct keys statistically independent.

    From ``SeedSequence(key)`` ten uint32 words are drawn once:

    * words 0..7 seed the NumPy legacy global RNG --
      ``np.random.seed(words[0:8])`` takes MT19937's ``init_by_array``
      path (256-bit seed space, no uint32 collapse);
    * words 8..9 form the JAX base key --
      ``jax_seed = int(words[8] & 0x7FFF_FFFF) << 32 | int(words[9])``
      (word 8 = high word with its top bit dropped so the integer always
      fits a non-negative int64 for ``jax.random.PRNGKey``; word 9 = low
      word).

    The actual ``SeedSequence`` entropy list is the key plus a fixed
    NON-ZERO terminator word ``1``: NumPy's ``SeedSequence`` absorbs
    trailing zero entropy words (``[s] == [s, 0] == [s, 0, 0]``), so
    without the terminator a 2-word single-run key ``[s, r]`` would alias
    the 3-word twist key ``[s, r, 0]`` (twist ``r``, intra rank 0). With
    the terminator no key is a trailing-zero extension of another, so keys
    of different lengths never collide.

    Args:
        chain_seed: Scalar seed or a sequence of integer key words
            (a tuple from the twist drivers, or an ndarray restored from a
            checkpoint's ``driver_config``).
        mpi_rank: Rank within the sampler's communicator (the intra-twist
            communicator for twist runs).

    Returns:
        ``(np_seed_words, jax_seed)`` -- a ``uint32[8]`` array for
        ``np.random.seed`` and the 63-bit Python int for
        ``jax.random.PRNGKey``.
    """
    if isinstance(chain_seed, (tuple, list, np.ndarray)):
        words = [int(w) for w in np.asarray(chain_seed).ravel()]
    else:
        words = [int(chain_seed)]
    ss = np.random.SeedSequence(words + [int(mpi_rank), 1])
    state = ss.generate_state(10, dtype=np.uint32)
    np_seed_words = state[:8].copy()
    jax_seed = (int(state[8] & 0x7FFFFFFF) << 32) | int(state[9])
    return np_seed_words, jax_seed


def _require(condition: bool, message: str, exc: type[Exception] = RuntimeError) -> None:
    """Raise ``exc(message)`` unless ``condition`` holds.

    Equivalent to ``assert condition, message`` but, unlike a bare ``assert``,
    survives ``python -O`` (assertions are stripped under ``-O``/``PYTHONOPTIMIZE``,
    silently deleting any validation expressed that way). Used throughout the
    ``MCMC`` / ``GFMC_t`` / ``GFMC_n`` production driver loops for shape and
    internal-consistency invariants that must not be optimized away.

    Args:
        condition: The invariant that must hold.
        message: Exact text to attach to the raised exception.
        exc: Exception type to raise on failure. Defaults to ``RuntimeError``
            (internal invariant); pass ``ValueError`` for user-input validation.
    """
    if not condition:
        raise exc(message)


def _validate_num_mcmc_per_measurement(value: Any, *, comm: MPI.Comm | None = None) -> None:
    """Require at least one sampler update between two measurements.

    ``num_mcmc_per_measurement`` counts the MCMC updates (``MCMC``) or GFMC
    projections (``GFMC_n``) between successive measurements. Zero would
    re-measure the unmoved configuration and a negative count has no meaning,
    so anything but an integer >= 1 is rejected before sampling. On checkpoint
    load ``comm`` makes every sampling rank raise the same error.
    """
    valid = isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 1
    error = None if valid else f"num_mcmc_per_measurement must be an integer >= 1 (got {value!r})."
    if comm is not None and comm.Get_size() > 1:
        for rank, rank_error in enumerate(comm.allgather(error)):
            if rank_error is not None:
                raise ValueError(f"checkpoint load sampling_rank={rank}: {rank_error}")
    elif error is not None:
        raise ValueError(error)


def _require_lrdmc_bra_num_mcmc_per_measurement(value: Any) -> None:
    """Refuse a GFMC_n run whose ``num_mcmc_per_measurement`` was not given.

    The number of GFMC projections between two measurements of ``GFMC_n``
    (``[lrdmc-bra] num_mcmc_per_measurement``) sets the imaginary time per
    measurement and with it the walker survival per branching, the warmup and
    collect windows and the cost; it depends on the system and on ``alat``. It
    is a required input with no default, so ``None`` (omitted) raises before any
    sampling. The value itself is checked by
    :func:`_validate_num_mcmc_per_measurement`.
    """
    if value is None:
        raise ValueError(
            "num_mcmc_per_measurement ([lrdmc-bra], GFMC projections per measurement) is required and has no "
            "default: set it to an integer >= 1."
        )


def _raise_collective_nonfinite_observables(
    comm: MPI.Intracomm,
    *,
    driver: str,
    stage: str,
    first_step: int,
    observables: Mapping[str, npt.ArrayLike],
    r_up_carts: npt.ArrayLike,
    r_dn_carts: npt.ArrayLike,
) -> None:
    """Fail every sampling rank when any rank owns a non-finite walker value.

    Each observable is scalar per walker and is accepted as ``(walkers,)`` or
    ``(samples, walkers)``.  The normal path performs one integer reduction and
    no object collection.  Detailed strings are gathered only after that
    reduction proves a failure exists, so a single bad rank cannot raise alone
    and strand its peers in a later MPI collective.

    Args:
        comm: Per-instance sampling communicator.
        driver: User-facing driver name (``MCMC``, ``GFMC_t`` or ``GFMC_n``).
        stage: Boundary being protected, such as ``measurement`` or
            ``checkpoint-load``.
        first_step: Global index of the first sample row in ``observables``.
        observables: Named scalar arrays whose last axis is the walker axis.
        r_up_carts: Current up-spin positions, ``(walkers, n_up, 3)``.
        r_dn_carts: Current down-spin positions, ``(walkers, n_dn, 3)``.

    Raises:
        ValueError: If an observable is not scalar-per-walker data.
        FloatingPointError: Collectively on every rank when any value is NaN or
            infinite.  The exception names rank, step, walker and field and
            refuses publication/checkpointing rather than dropping a walker.
    """
    sampling_rank = int(comm.Get_rank())
    world_rank = int(MPI.COMM_WORLD.Get_rank())
    r_up_shape = tuple(np.shape(r_up_carts))
    r_dn_shape = tuple(np.shape(r_dn_carts))
    num_walkers = int(r_up_shape[0]) if r_up_shape else 0
    if r_dn_shape and num_walkers == 0:
        num_walkers = int(r_dn_shape[0])

    local_records: list[str] = []
    local_bad_count = 0
    for field, raw_values in observables.items():
        values = np.asarray(raw_values)
        if values.ndim == 0:
            values = values.reshape(1, 1)
        elif values.ndim == 1:
            values = values.reshape(1, -1)
        elif values.ndim != 2:
            raise ValueError(
                f"{driver} {stage}: observable {field!r} must have shape (walkers,) or (samples, walkers), got {values.shape}."
            )
        if num_walkers and values.shape[1] != num_walkers:
            raise ValueError(
                f"{driver} {stage}: observable {field!r} has {values.shape[1]} walkers, "
                f"but restored/current positions have {num_walkers}."
            )
        bad_indices = np.argwhere(~np.isfinite(values))
        local_bad_count += int(bad_indices.shape[0])
        # Position materialization can transfer a large walker batch from a
        # GPU. Keep it strictly on the exceptional path; the normal-path gate
        # only reads static shapes plus the small host observable arrays.
        r_up = np.asarray(r_up_carts) if bad_indices.size else None
        r_dn = np.asarray(r_dn_carts) if bad_indices.size else None
        for sample_index, walker_index in bad_indices[:8]:
            position_blocks = []
            if r_up is not None and r_up.ndim == 3 and walker_index < r_up.shape[0]:
                position_blocks.append(np.ravel(r_up[walker_index]))
            if r_dn is not None and r_dn.ndim == 3 and walker_index < r_dn.shape[0]:
                position_blocks.append(np.ravel(r_dn[walker_index]))
            positions = np.concatenate(position_blocks) if position_blocks else np.empty(0)
            max_abs_position = float(np.max(np.abs(positions))) if positions.size else float("nan")
            local_records.append(
                f"sampling_rank={sampling_rank}, world_rank={world_rank}, "
                f"step={int(first_step) + int(sample_index)}, walker={int(walker_index)}, "
                f"field={field}, value={values[sample_index, walker_index]!r}, "
                f"max_abs_position={max_abs_position:.6g}"
            )

    global_bad_count = int(comm.allreduce(local_bad_count, op=MPI.SUM)) if comm.Get_size() > 1 else local_bad_count
    if global_bad_count == 0:
        return
    gathered: list[Any]
    if comm.Get_size() > 1:
        gathered = comm.allgather(tuple(local_records))
    else:
        gathered = [tuple(local_records)]
    records = [record for rank_records in gathered for record in rank_records]
    details = "\n  ".join(records[:16])
    omitted = global_bad_count - min(global_bad_count, 16)
    if omitted > 0:
        details += f"\n  ... and {omitted} more non-finite value(s)"
    from . import _setting
    from ._precision import is_mixed_precision_enabled

    env_rust = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
    rust_requested = (
        _setting.flag_rust_on_CPU if env_rust is None else env_rust.strip().lower() not in ("0", "false", "no", "off", "")
    )
    jax_backend = jax.default_backend()
    execution_backend = "XLA + Rust" if rust_requested and jax_backend == "cpu" else "XLA"
    precision_mode = "mixed" if is_mixed_precision_enabled() else "full"
    raise FloatingPointError(
        f"{driver} detected {global_bad_count} non-finite walker observable value(s) "
        f"at {stage} (precision={precision_mode}, backend={execution_backend}, platform={jax_backend}); "
        f"refusing to publish or checkpoint a biased/poisoned estimate.\n  {details}"
    )


def _os_assigned_free_tcp_port() -> int:
    """Return a TCP port the OS reports as free on this host right now.

    Binds a throw-away socket to port 0 on all IPv4 interfaces, reads back the
    port the kernel's ephemeral allocator chose, and closes the socket. The
    kernel never hands out a port that another socket currently holds, so the
    result cannot coincide with a live coordinator of another job on this host.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("", 0))
        return int(probe.getsockname()[1])


def _pick_jax_coordinator_address(comm: MPI.Intracomm) -> str:
    """Choose the JAX coordination-service address on rank 0 and broadcast it.

    Collective over ``comm`` (every rank must call it). Rank 0 builds
    ``"<hostname>:<port>"`` and ``comm.bcast`` hands the identical string to
    every rank.

    Why not JAX's own ``cluster_detection_method="mpi4py"`` address: its
    ``Mpi4pyCluster.get_coordinator_address`` uses
    ``hash(hostname) % 2**12 + 61440``, i.e. one of only 4096 ports chosen by
    the per-process string hash. Two concurrent MPI jobs on one host (CI
    runners, several jqmc jobs on one node) can pick the same port; one job's
    workers then register with the other job's coordinator and JAX aborts with
    ``INVALID_ARGUMENT: Unexpected task registered`` (or the job hangs until
    the init timeout). Here the port comes from the OS ephemeral allocator
    (:func:`_os_assigned_free_tcp_port`) instead.

    The probe socket is closed before JAX's coordinator binds the port, so a
    small time-of-check/time-of-use window remains in which another process
    could take it; the collision space is the OS ephemeral range with ports
    in use excluded, instead of 4096 hashed ports.

    An explicit JAX pin is respected unchanged: ``JAX_COORDINATOR_ADDRESS``
    (rank 0's value) is used verbatim, and ``JAX_COORDINATOR_PORT`` replaces
    the OS-assigned port, matching what JAX itself does with those variables.
    """
    address = None
    if comm.Get_rank() == 0:
        address = os.environ.get("JAX_COORDINATOR_ADDRESS")
        if not address:
            port = os.environ.get("JAX_COORDINATOR_PORT") or str(_os_assigned_free_tcp_port())
            address = f"{socket.gethostname()}:{port}"
    return comm.bcast(address, root=0)


def _initialize_jax_distributed() -> None:
    """Initialize ``jax.distributed`` over ``MPI.COMM_WORLD`` with a collision-free port.

    Single entry point shared by the CLI, the pytest MPI harness and the
    driver ``__main__`` blocks. The coordinator address comes from
    :func:`_pick_jax_coordinator_address`; everything else (process count,
    process id, host-local device ids) is still auto-detected by JAX's
    ``mpi4py`` cluster environment exactly as before. Exceptions from
    ``jax.distributed.initialize`` propagate unchanged so callers keep their
    existing handling.
    """
    coordinator_address = _pick_jax_coordinator_address(MPI.COMM_WORLD)
    jax.distributed.initialize(coordinator_address=coordinator_address, cluster_detection_method="mpi4py")


def _check_mpi4py_jax_distribution_consistency() -> None:
    """Raise ``RuntimeError`` if ``jax.process_count()`` differs from MPI world size.

    Required precondition for any production code path that relies on
    ``jax.shard_map`` + ``psum`` / ``all_gather`` to aggregate across MPI
    ranks (currently the device branch of ``MCMC.run_optimize``; future
    GFMC device migrations will share this requirement). Without
    ``jax.distributed.initialize(cluster_detection_method="mpi4py")``, each
    rank's JAX sees only its own local devices and the cross-rank
    collectives silently degenerate to per-rank-local computation, producing
    wrong results that don't match the global solution.

    Called at the entry point of every public driver loop (``MCMC.run``,
    ``MCMC.run_optimize``, ``GFMC_t.run``, ``GFMC_n.run``) so the failure
    surfaces immediately rather than after substantial work has been done
    on stale per-rank state.

    The CLI (``jqmc_cli.py``) initializes JAX distributed automatically.
    User scripts that import ``MCMC`` / ``GFMC`` directly must call
    ``jax.distributed.initialize(cluster_detection_method="mpi4py")``
    themselves before invoking these driver loops (see ``jqmc_cli.py``
    for the standard recipe, including the proxy-strip workaround).
    """
    mpi_size = MPI.COMM_WORLD.Get_size()
    if mpi_size != jax.process_count():
        raise RuntimeError(
            f"MPI/JAX rank-count mismatch: mpi_size={mpi_size} vs "
            f"jax.process_count()={jax.process_count()}. JAX cross-rank "
            "collectives (psum / all_gather) require ``jax.distributed.initialize("
            "cluster_detection_method='mpi4py')`` to be called before any "
            "production driver loop. The CLI does this automatically; user "
            "scripts importing MCMC / GFMC must call it themselves."
        )


# End-of-run stages whose peak-memory line is part of the default (INFO) output,
# so every run reports its peak memory. The other stages stay at DEBUG.
_MEMORY_INFO_STAGES = frozenset({"mcmc_post_run", "gfmc_t_post_run", "gfmc_n_post_run", "optimize_post_loop"})


def _log_memory_usage(stage: str, comm: MPI.Intracomm | None = None) -> dict:
    """Log the cluster-wide max peak RSS at a named stage.

    Cheap diagnostic intended to be called at a handful of stage boundaries
    inside the driver loops (post-setup, post-warmup, post-loop) so that
    users running on shared machines or under MPI can see where memory is
    being consumed without resorting to external profilers.

    Under MPI the local peak is Allreduce'd with ``MPI.MAX`` so every rank
    holds the same cluster-wide max; the logger itself controls who actually
    prints (rank-0-only by default, all ranks under the ``mpi-*`` verbosity
    labels). The ``sum`` aggregate is intentionally not reported -- it is
    misleading when ranks are spread across nodes.

    JAX device memory is only meaningful on GPU/TPU backends. On the CPU
    backend ``memory_stats`` is unreliable (often returns ``inf`` /
    placeholders), so the JAX line is suppressed entirely there.

    The end-of-run stages in ``_MEMORY_INFO_STAGES`` log at INFO, so the line
    appears in the default output (verbosity ``low`` and ``high``); the other
    stages log at DEBUG.

    Args:
        stage: Short label that identifies the call site (e.g. "post_setup").
        comm: Optional MPI intracommunicator. When supplied, the helper
            Allreduces across ranks before logging.

    Returns:
        Dict with the cluster-wide-max numbers so callers can record them
        in their own checkpoint / observable dicts.

    Notes:
        - ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` returns the high-water
          mark (peak) since process start; on Linux it is reported in kB, on
          macOS in bytes. Both are handled.
        - The function never raises (best-effort); diagnostic failures must
          not break a long-running production loop.
    """
    # ------ per-rank peak RSS ---------------------------------------------
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            peak_rss_gb = ru / (1024**3)
        else:
            peak_rss_gb = ru / (1024**2)
    except Exception:  # noqa: BLE001 -- diagnostic must not raise
        peak_rss_gb = float("nan")

    # ------ JAX device memory (skip on CPU -- numbers are not meaningful) -
    try:
        is_cpu = jax.default_backend() == "cpu"
    except Exception:  # noqa: BLE001
        is_cpu = True  # fail safe -- treat as CPU and skip
    jax_peak_gb = float("nan")
    if not is_cpu:
        try:
            stats = jax.devices()[0].memory_stats()
            if stats:
                jax_peak_gb = stats.get("peak_bytes_in_use", 0) / (1024**3)
        except Exception:  # noqa: BLE001
            pass

    # ------ Cluster-wide MAX via Allreduce so every rank holds it ---------
    size = 1
    if comm is not None:
        size = comm.Get_size()
    if comm is not None and size > 1:
        local = np.array([peak_rss_gb, jax_peak_gb], dtype=np.float64)
        # MPI.MAX skips NaN inconsistently across implementations; replace
        # NaN with -inf for the reduction and lift back to NaN afterwards.
        local_max = np.where(np.isnan(local), -np.inf, local)
        global_max = np.empty_like(local_max)
        comm.Allreduce(local_max, global_max, op=MPI.MAX)
        rss_max_gb = float(global_max[0]) if global_max[0] > -np.inf else float("nan")
        jax_max_gb = float(global_max[1]) if global_max[1] > -np.inf else float("nan")
    else:
        rss_max_gb = peak_rss_gb
        jax_max_gb = jax_peak_gb

    # ------ Single log line. The logger config (rank-0 vs all-ranks under
    # ``mpi-*`` verbosity) decides who actually emits.
    level = INFO if stage in _MEMORY_INFO_STAGES else DEBUG
    if not is_cpu and not np.isnan(jax_max_gb):
        logger.log(
            level,
            "[memory] stage=%s, RSS_peak_max=%.2f GB, JAX_peak_max=%.2f GB",
            stage,
            rss_max_gb,
            jax_max_gb,
        )
    else:
        logger.log(
            level,
            "[memory] stage=%s, RSS_peak_max=%.2f GB",
            stage,
            rss_max_gb,
        )

    return {
        "stage": stage,
        "n_ranks": size,
        "rss_peak_max_gb": rss_max_gb,
        "jax_peak_max_gb": jax_max_gb,
    }


def _generate_init_electron_configurations(
    tot_num_electron_up: int,
    tot_num_electron_dn: int,
    num_walkers: int,
    charges: np.ndarray,
    coords: np.ndarray,
    cell: np.ndarray | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Vectorized initial electron configuration generator for many walkers.

    Functionally equivalent to :func:`_generate_init_electron_configurations_debug`
    but avoids the per-walker Python loop. Designed for large walker counts
    (e.g. ``num_walkers = 16384``) where the reference version becomes the
    initialization bottleneck.

    Algorithm:
        1. Compute the deterministic atom-assignment templates for spin-up and
           spin-down electrons by replaying the reference state machine **once**.
           These templates depend only on (charges, coords) -- every walker
           shares them.
        2. Tile the templates to shape ``(num_walkers, ned)``.
        3. For the only branch in the reference that uses per-walker randomness
           -- trailing "extra up" electrons when
           ``tot_num_electron_up > sum(zeta - occup_dn)`` -- draw the random
           atom indices in one batched ``np.random.randint`` call.
        4. Draw all spherical random offsets in one batched call per spin and
           add them to the chosen atomic coordinates.

    The duplicate-avoidance retry loop in the reference implementation is
    omitted: spherical offsets are sampled from continuous uniform
    distributions, so the probability of two ``float64`` positions colliding
    within ``1e-6`` is effectively zero. The companion test
    ``test_init_electron_configurations.py`` verifies uniqueness across all
    generated positions.

    Parameters and returns are identical to
    :func:`_generate_init_electron_configurations_debug`.
    """
    min_dst = 0.1
    max_dst = 1.0
    dtype_np = np.float64

    nion = coords.shape[0]
    coords_np = np.asarray(coords, dtype=dtype_np)
    zeta = np.array([int(round(c)) for c in np.asarray(charges)], dtype=int)
    max_dn_per_atom = zeta // 2

    # 1) Build ion_seq (each next index is the atom farthest from the previous).
    ion_sel = np.ones(nion, dtype=bool)
    ion_seq = np.zeros(nion, dtype=int)
    ion_seq[0] = 0
    ion_sel[0] = False
    i_prev = 0
    for idx in range(1, nion):
        d2 = np.sum((coords_np[i_prev] - coords_np) ** 2, axis=1)
        d2_masked = np.where(ion_sel, d2, -1.0)
        best_i = int(np.argmax(d2_masked))
        ion_seq[idx] = best_i
        ion_sel[best_i] = False
        i_prev = best_i

    # 2) Replay the deterministic state machine ONCE to obtain owner templates.
    occup_total = np.zeros(nion, dtype=int)
    occup_dn = np.zeros(nion, dtype=int)
    occup_up = np.zeros(nion, dtype=int)

    # First place all spin-down electrons deterministically.
    ned_dn = tot_num_electron_dn
    dn_owner_template = np.empty(ned_dn, dtype=int)
    j_counter = 0
    for idn in range(ned_dn):
        while True:
            atom = ion_seq[j_counter % nion]
            if np.any(occup_dn < max_dn_per_atom):
                cond = occup_dn[atom] < max_dn_per_atom[atom]
            else:
                mask_zero = (max_dn_per_atom == 0) & (occup_total < zeta)
                if np.any(mask_zero):
                    cond = (max_dn_per_atom[atom] == 0) and (occup_total[atom] < zeta[atom])
                else:
                    cond = occup_total[atom] < zeta[atom]
            if cond:
                dn_owner_template[idn] = atom
                occup_dn[atom] += 1
                occup_total[atom] += 1
                j_counter += 1
                break
            j_counter += 1

    # Then place spin-up electrons; deterministic except for the
    # "extra" tail in Case 2 which is per-walker random.
    up_needed = zeta - occup_dn
    sum_up_needed = int(np.sum(up_needed))
    ned_up = tot_num_electron_up
    up_owner_template = np.empty(ned_up, dtype=int)
    n_random_extras = 0  # trailing electrons whose owner is random per walker

    if ned_up <= sum_up_needed:
        # Case 1: place exactly into the up_needed slots -- fully deterministic.
        ptr = 0
        for iup in range(ned_up):
            while True:
                atom = ion_seq[ptr % nion]
                if occup_up[atom] < up_needed[atom]:
                    up_owner_template[iup] = atom
                    occup_up[atom] += 1
                    occup_total[atom] += 1
                    ptr += 1
                    break
                ptr += 1
    else:
        # Case 2: first satisfy every atom's up_needed (deterministic), then
        # the remainder is sampled per walker.
        cnt = 0
        for atom in ion_seq:
            to_give = int(up_needed[atom])
            for _ in range(to_give):
                up_owner_template[cnt] = atom
                occup_up[atom] += 1
                occup_total[atom] += 1
                cnt += 1
        n_random_extras = ned_up - sum_up_needed

    # 3) Build per-walker owner arrays.
    if ned_dn > 0:
        dn_owner = np.broadcast_to(dn_owner_template, (num_walkers, ned_dn)).copy()
    else:
        dn_owner = np.empty((num_walkers, 0), dtype=int)

    up_owner = np.empty((num_walkers, ned_up), dtype=int)
    if n_random_extras > 0:
        det_count = ned_up - n_random_extras
        if det_count > 0:
            up_owner[:, :det_count] = up_owner_template[:det_count][None, :]
        # Per-walker random pick from ion_seq, matching the original
        #   idx = int(floor(np.random.rand() * nion))
        rand_idx = np.floor(np.random.rand(num_walkers, n_random_extras) * nion).astype(int)
        # Clip to nion-1 just in case np.random.rand() returns 1.0 (it shouldn't).
        np.clip(rand_idx, 0, nion - 1, out=rand_idx)
        up_owner[:, det_count:] = ion_seq[rand_idx]
    else:
        up_owner[:] = up_owner_template[None, :]

    # 4) Draw all spherical random offsets in batched form.
    def _spherical_offsets(shape: tuple[int, int]) -> np.ndarray:
        distance = np.random.uniform(min_dst, max_dst, size=shape)
        theta = np.random.uniform(0.0, np.pi, size=shape)
        phi = np.random.uniform(0.0, 2.0 * np.pi, size=shape)
        sin_t = np.sin(theta)
        return np.stack(
            [
                distance * sin_t * np.cos(phi),
                distance * sin_t * np.sin(phi),
                distance * np.cos(theta),
            ],
            axis=-1,
        ).astype(dtype_np, copy=False)

    offset_dn = _spherical_offsets((num_walkers, ned_dn)) if ned_dn > 0 else np.zeros((num_walkers, 0, 3), dtype=dtype_np)
    offset_up = _spherical_offsets((num_walkers, ned_up)) if ned_up > 0 else np.zeros((num_walkers, 0, 3), dtype=dtype_np)

    # 5) Assemble final positions: r = coords[owner] + offset.
    r_carts_up = (coords_np[up_owner] + offset_up).astype(dtype_np, copy=False)
    r_carts_dn = (coords_np[dn_owner] + offset_dn).astype(dtype_np, copy=False)

    # 6) Fold into the home cell for PBC (no-op for molecules, ``cell=None``).
    # The spherical offset can push an electron outside Omega when an atom sits
    # near a cell face; the first projection / MCMC step evaluates there, and
    # the finite AO periodic-image expansion is only accurate inside Omega.
    # Wrapping keeps the seed configuration well-defined.
    if cell is not None:
        cell_np = np.asarray(cell, dtype=dtype_np)
        inv_cell = np.linalg.inv(cell_np)
        frac_up = r_carts_up @ inv_cell
        frac_dn = r_carts_dn @ inv_cell
        r_carts_up = ((frac_up - np.floor(frac_up)) @ cell_np).astype(dtype_np, copy=False)
        r_carts_dn = ((frac_dn - np.floor(frac_dn)) @ cell_np).astype(dtype_np, copy=False)

    return r_carts_up, r_carts_dn, up_owner, dn_owner


def _generate_init_electron_configurations_debug(
    tot_num_electron_up: int,
    tot_num_electron_dn: int,
    num_walkers: int,
    charges: np.ndarray,
    coords: np.ndarray,
    cell: np.ndarray | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Reference (per-walker Python loop) initial electron configuration generator.

    This is the reference for cross-checks against the vectorized
    :func:`_generate_init_electron_configurations`. It runs an
    ``O(num_walkers)`` Python loop and is too slow for large walker counts
    (e.g. ``num_walkers = 16384``); use only for tests / debugging.

    Generate initial electron configurations (up/down positions) for a set of walkers,
    using the same ion_seq idea as the Fortran initconf routine. With ``cell``
    the positions are folded into the home cell; no other periodic boundary
    condition is applied.

    Args:
        tot_num_electron_up (int):
            Total number of spin-up electrons in the system.

        tot_num_electron_dn (int):
            Total number of spin-down electrons in the system.

        num_walkers (int):
            Number of independent walkers (configurations) to generate.

        charges (np.ndarray of shape (nion,)):
            Atomic charges (should reflect valence electron count, e.g.
            atomic_number or atomic_number - z_core). They will be rounded to
            integers internally.

        coords (np.ndarray of shape (nion, 3)):
            Cartesian coordinates of each atom in the system.

        cell (np.ndarray of shape (3, 3) or None):
            Lattice vectors as rows (Bohr) for PBC; the generated electron
            positions are folded into the home cell. ``None`` (default) for a
            molecule, in which case positions are returned unwrapped.

    Returns:
        r_carts_up (np.ndarray of shape (num_walkers, tot_num_electron_up, 3)):
            Generated positions of all spin-up electrons for each walker.

        r_carts_dn (np.ndarray of shape (num_walkers, tot_num_electron_dn, 3)):
            Generated positions of all spin-down electrons for each walker.

        up_owner (np.ndarray of shape (num_walkers, tot_num_electron_up), dtype=int):
            For each walker `iw` and each up-electron `k`, the atom-index it was assigned to.

        dn_owner (np.ndarray of shape (num_walkers, tot_num_electron_dn), dtype=int):
            For each walker `iw` and each down-electron `k`, the atom-index it was assigned to.
    """
    # Fixed random displacement range (+/-dst/2 in each coordinate)
    min_dst = 0.1
    max_dst = 1.0

    dtype_np = np.float64

    # 1) zeta[i] = integer valence count per atom
    nion = coords.shape[0]
    zeta = np.array([int(round(c)) for c in charges], dtype=int)

    # 2) max_dn_per_atom = floor(zeta[i]/2) for Hund's rule on down-electrons
    max_dn_per_atom = zeta // 2

    # 3) Build ion_seq so that each next index is the atom farthest from the previous
    ion_sel = np.ones(nion, dtype=bool)
    ion_seq = np.zeros(nion, dtype=int)
    ion_seq[0] = 0
    ion_sel[0] = False
    i_prev = 0
    for idx in range(1, nion):
        best_dist = -1.0
        best_i = -1
        for i in range(nion):
            if ion_sel[i]:
                d2 = np.sum((coords[i_prev] - coords[i]) ** 2)
                if d2 > best_dist:
                    best_dist = d2
                    best_i = i
        ion_seq[idx] = best_i
        ion_sel[best_i] = False
        i_prev = best_i

    # 4) Prepare storage for all walkers
    r_carts_up = np.zeros((num_walkers, tot_num_electron_up, 3), dtype=dtype_np)
    r_carts_dn = np.zeros((num_walkers, tot_num_electron_dn, 3), dtype=dtype_np)
    up_owner = np.zeros((num_walkers, tot_num_electron_up), dtype=int)
    dn_owner = np.zeros((num_walkers, tot_num_electron_dn), dtype=int)

    # 6) Loop over walkers
    for iw in range(num_walkers):
        # 6.1) Reset per-walker occupancy
        occup_total = np.zeros(nion, dtype=int)  # total electrons (up+dn) on each atom
        occup_dn = np.zeros(nion, dtype=int)  # how many down-electrons on each atom
        occup_up = np.zeros(nion, dtype=int)  # how many up-electrons on each atom
        cdown = 0
        cup = 0

        # 6.2) Compute any "extra" beyond sum(zeta)
        nel = tot_num_electron_up + tot_num_electron_dn
        ztot = int(np.sum(zeta))
        nelupeff = nel - ztot if nel > ztot else 0

        # -----------------------------------------
        # Place all down-electrons under Hund's limit first
        # -----------------------------------------
        ned_dn = tot_num_electron_dn
        down_positions = np.zeros((ned_dn, 3), dtype=dtype_np)
        j_counter = 0

        for idn in range(ned_dn):
            placed = False
            while not placed:
                atom = ion_seq[j_counter % nion]

                # If any atom still has occup_dn < max_dn_per_atom, restrict to those atoms.
                if np.any(occup_dn < max_dn_per_atom):
                    cond = occup_dn[atom] < max_dn_per_atom[atom]
                else:
                    # All atoms have occup_dn == max_dn_per_atom.  Next fallback:
                    #   1) If any atom has max_dn_per_atom == 0 (e.g. H) AND occup_total < zeta,
                    #      restrict to those atoms first.
                    mask_zero = (max_dn_per_atom == 0) & (occup_total < zeta)
                    if np.any(mask_zero):
                        cond = (max_dn_per_atom[atom] == 0) and (occup_total[atom] < zeta[atom])
                    else:
                        # Otherwise, any atom with occup_total < zeta can accept a down
                        cond = occup_total[atom] < zeta[atom]

                if cond:
                    # Place one dn-electron around coords[atom] + random_offset
                    x0, y0, z0 = coords[atom]
                    distance = np.random.uniform(min_dst, max_dst)
                    theta = np.random.uniform(0, np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    dx = distance * np.sin(theta) * np.cos(phi)
                    dy = distance * np.sin(theta) * np.sin(phi)
                    dz = distance * np.cos(theta)
                    new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])

                    # Avoid exact duplication among already-placed down-positions
                    ok = False
                    while not ok:
                        ok = True
                        for prev in down_positions[:cdown]:
                            if np.sum(np.abs(prev - new_pos)) < ATOL_DRIVER_ELECTRON_POSITION_DISTINCTNESS:
                                distance = np.random.uniform(min_dst, max_dst)
                                theta = np.random.uniform(0, np.pi)
                                phi = np.random.uniform(0, 2 * np.pi)
                                dx = distance * np.sin(theta) * np.cos(phi)
                                dy = distance * np.sin(theta) * np.sin(phi)
                                dz = distance * np.cos(theta)
                                new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])
                                ok = False
                                break

                    down_positions[idn] = new_pos
                    dn_owner[iw, idn] = atom
                    occup_dn[atom] += 1
                    occup_total[atom] += 1
                    cdown += 1
                    placed = True

                j_counter += 1

        # -----------------------------------------
        # Place up-electrons exactly to "fill to zeta" if possible
        # -----------------------------------------
        # Compute how many up each atom needs to reach zeta:
        up_needed = zeta - occup_dn  # array of length nion
        sum_up_needed = int(np.sum(up_needed))

        ned_up = tot_num_electron_up
        up_positions = np.zeros((ned_up, 3), dtype=dtype_np)

        # Case 1: ned_up <= sum_up_needed -> place ned_up among those up_needed slots
        if ned_up <= sum_up_needed:
            ptr = 0
            for iup in range(ned_up):
                placed = False
                while not placed:
                    atom = ion_seq[ptr % nion]
                    if occup_up[atom] < up_needed[atom]:
                        # Place one up-electron here
                        x0, y0, z0 = coords[atom]
                        distance = np.random.uniform(min_dst, max_dst)
                        theta = np.random.uniform(0, np.pi)
                        phi = np.random.uniform(0, 2 * np.pi)
                        dx = distance * np.sin(theta) * np.cos(phi)
                        dy = distance * np.sin(theta) * np.sin(phi)
                        dz = distance * np.cos(theta)
                        new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])

                        # Avoid duplication among already-placed up-positions
                        ok = False
                        while not ok:
                            ok = True
                            for prev in up_positions[:cup]:
                                if np.sum(np.abs(prev - new_pos)) < ATOL_DRIVER_ELECTRON_POSITION_DISTINCTNESS:
                                    distance = np.random.uniform(min_dst, max_dst)
                                    theta = np.random.uniform(0, np.pi)
                                    phi = np.random.uniform(0, 2 * np.pi)
                                    dx = distance * np.sin(theta) * np.cos(phi)
                                    dy = distance * np.sin(theta) * np.sin(phi)
                                    dz = distance * np.cos(theta)
                                    new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])
                                    ok = False
                                    break

                        up_positions[iup] = new_pos
                        up_owner[iw, iup] = atom
                        occup_up[atom] += 1
                        occup_total[atom] += 1
                        cup += 1
                        placed = True
                    ptr += 1

        # Case 2: ned_up > sum_up_needed -> give each atom its up_needed, then place extras
        else:
            # (a) first satisfy every atom's up_needed
            cnt = 0
            for atom in ion_seq:
                to_give = int(up_needed[atom])
                for _ in range(to_give):
                    x0, y0, z0 = coords[atom]
                    distance = np.random.uniform(min_dst, max_dst)
                    theta = np.random.uniform(0, np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    dx = distance * np.sin(theta) * np.cos(phi)
                    dy = distance * np.sin(theta) * np.sin(phi)
                    dz = distance * np.cos(theta)
                    new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])

                    ok = False
                    while not ok:
                        ok = True
                        for prev in up_positions[:cup]:
                            if np.sum(np.abs(prev - new_pos)) < ATOL_DRIVER_ELECTRON_POSITION_DISTINCTNESS:
                                distance = np.random.uniform(min_dst, max_dst)
                                theta = np.random.uniform(0, np.pi)
                                phi = np.random.uniform(0, 2 * np.pi)
                                dx = distance * np.sin(theta) * np.cos(phi)
                                dy = distance * np.sin(theta) * np.sin(phi)
                                dz = distance * np.cos(theta)
                                new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])
                                ok = False
                                break

                    up_positions[cnt] = new_pos
                    up_owner[iw, cnt] = atom
                    occup_up[atom] += 1
                    occup_total[atom] += 1
                    cnt += 1
                    cup += 1

            # (b) now place the "extra" up = ned_up - sum_up_needed on any atom (fallback)
            extra_up = ned_up - sum_up_needed
            for _ in range(extra_up):
                idx = int(np.floor(np.random.rand() * nion))
                atom = ion_seq[idx]
                x0, y0, z0 = coords[atom]
                distance = np.random.uniform(min_dst, max_dst)
                theta = np.random.uniform(0, np.pi)
                phi = np.random.uniform(0, 2 * np.pi)
                dx = distance * np.sin(theta) * np.cos(phi)
                dy = distance * np.sin(theta) * np.sin(phi)
                dz = distance * np.cos(theta)
                new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])

                ok = False
                while not ok:
                    ok = True
                    for prev in up_positions[:cup]:
                        if np.sum(np.abs(prev - new_pos)) < ATOL_DRIVER_ELECTRON_POSITION_DISTINCTNESS:
                            distance = np.random.uniform(min_dst, max_dst)
                            theta = np.random.uniform(0, np.pi)
                            phi = np.random.uniform(0, 2 * np.pi)
                            dx = distance * np.sin(theta) * np.cos(phi)
                            dy = distance * np.sin(theta) * np.sin(phi)
                            dz = distance * np.cos(theta)
                            new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])
                            ok = False
                            break

                up_positions[cnt] = new_pos
                up_owner[iw, cnt] = atom
                occup_up[atom] += 1
                occup_total[atom] += 1
                cnt += 1
                cup += 1

        # -----------------------------------------
        # Finally place any "extra" electrons that remain (nelupeff > 0).
        #          (This is almost never needed if tot_up+tot_dn == sum(zeta), but we include it for completeness.)
        # -----------------------------------------
        if nelupeff > 1:
            # 2a) extra down beyond Hund's limit
            sum_dn_assigned = int(np.sum(occup_dn))
            extra_dn = ned_dn - sum_dn_assigned
            for _ in range(extra_dn):
                idx = int(np.floor(np.random.rand() * nion))
                atom = ion_seq[idx]
                x0, y0, z0 = coords[atom]
                distance = np.random.uniform(min_dst, max_dst)
                theta = np.random.uniform(0, np.pi)
                phi = np.random.uniform(0, 2 * np.pi)
                dx = distance * np.sin(theta) * np.cos(phi)
                dy = distance * np.sin(theta) * np.sin(phi)
                dz = distance * np.cos(theta)
                new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])

                ok = False
                while not ok:
                    ok = True
                    for prev in down_positions[:cdown]:
                        if np.sum(np.abs(prev - new_pos)) < ATOL_DRIVER_ELECTRON_POSITION_DISTINCTNESS:
                            distance = np.random.uniform(min_dst, max_dst)
                            theta = np.random.uniform(0, np.pi)
                            phi = np.random.uniform(0, 2 * np.pi)
                            dx = distance * np.sin(theta) * np.cos(phi)
                            dy = distance * np.sin(theta) * np.sin(phi)
                            dz = distance * np.cos(theta)
                            new_pos = np.array([x0 + dx, y0 + dy, z0 + dz])
                            ok = False
                            break

                down_positions[cdown] = new_pos
                dn_owner[iw, cdown] = atom
                occup_dn[atom] += 1
                occup_total[atom] += 1
                cdown += 1

        # -----------------------------------------
        # 6.3) Final consistency check
        # -----------------------------------------
        if cup != tot_num_electron_up:
            raise RuntimeError(f"Walker {iw}: assigned up={cup}, expected {tot_num_electron_up}")
        if cdown != tot_num_electron_dn:
            raise RuntimeError(f"Walker {iw}: assigned dn={cdown}, expected {tot_num_electron_dn}")

        # -----------------------------------------
        # 6.4) Copy into outputs
        # -----------------------------------------
        r_carts_up[iw, :, :] = up_positions
        r_carts_dn[iw, :, :] = down_positions

    # Fold into the home cell for PBC (no-op for molecules, ``cell=None``);
    # mirrors the vectorized generator so the debug seed configuration matches.
    if cell is not None:
        cell_np = np.asarray(cell, dtype=dtype_np)
        inv_cell = np.linalg.inv(cell_np)
        frac_up = r_carts_up @ inv_cell
        frac_dn = r_carts_dn @ inv_cell
        r_carts_up = ((frac_up - np.floor(frac_up)) @ cell_np).astype(dtype_np, copy=False)
        r_carts_dn = ((frac_dn - np.floor(frac_dn)) @ cell_np).astype(dtype_np, copy=False)

    return r_carts_up, r_carts_dn, up_owner, dn_owner
