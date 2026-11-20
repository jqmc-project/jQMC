"""Checkpoint utilities for saving/loading QMC driver state to HDF5.

This module provides the infrastructure for the restart system.
All QMC driver state (MCMC, GFMC_t, GFMC_n) is saved to a single
HDF5 file with per-rank groups, and hamiltonian_data stored once
at the root level.

File layout::

    restart.h5
    |-- _meta/                     (driver_type, mpi_size, provenance, ...)
    |-- hamiltonian_data/          (shared, saved once)
    `-- rank_{R}/                  (per MPI rank)
        |-- driver_config/
        |-- rng_state/
        |-- walker_state/
        |-- observables/
        `-- optimizer_state/       (VMCopt only)

Write flow::

    1. Each rank writes a temporary HDF5: ``._restart_rank{R}.h5``
    2. MPI Barrier
    3. Rank 0 adds hamiltonian_data and consolidates the rank files. A
       multi-rank merge stages the new checkpoint beside its destination
       and publishes it before removing any rank files. For a single rank,
       its groups can be relinked in place instead of copied.
    4. MPI Barrier

Read flow::

    All ranks open the same file read-only and read their own group.
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

from __future__ import annotations

import datetime
import errno
import os
import shutil
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any

import h5py
import numpy as np

from .hamiltonians import (
    Hamiltonian_data,
    _apply_ao_basis_dtype_alignment_post_load,
    _apply_ewald_rebuild_post_load,
    _apply_image_expansion_post_load,
    _apply_jastrow_structure_backfill_post_load,
    _save_dataclass_to_hdf5,
)

logger = getLogger("jqmc").getChild(__name__)

# ---------------------------------------------------------------------------
# Per-rank checkpoint I/O (temporary file written by each rank)
# ---------------------------------------------------------------------------


def save_rank_checkpoint(
    filepath: str,
    *,
    driver_type: str,
    driver_config: dict[str, Any],
    rng_state: dict[str, Any],
    walker_state: dict[str, Any],
    observables: dict[str, Any],
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    """Write one MPI rank's data to a temporary HDF5 file.

    This file is later merged by :func:`merge_rank_checkpoints` on rank 0.

    Args:
        filepath: Path to write (e.g. ``._restart_rank0.h5``).
        driver_type: ``"MCMC"``, ``"GFMC_t"``, or ``"GFMC_n"``.
        driver_config: Scalar execution parameters (seed, Dt, etc.).
        rng_state: RNG arrays and seeds.
        walker_state: Latest walker positions / weights.
        observables: Accumulated measurements (e_L, w_L, ...).
        optimizer_state: VMCopt optimizer runtime (optional).
    """
    with h5py.File(filepath, "w") as f:
        # driver_config -- scalars as attrs
        cfg_grp = f.create_group("driver_config")
        for k, v in driver_config.items():
            if v is None:
                continue
            if isinstance(v, (np.ndarray,)):
                cfg_grp.create_dataset(k, data=v)
            elif isinstance(v, (int, float, bool, str)):
                cfg_grp.attrs[k] = v
            else:
                # fallback: try as attr
                try:
                    cfg_grp.attrs[k] = v
                except Exception:
                    # An unserializable driver_config attribute would otherwise
                    # be silently dropped and read back as missing on restart.
                    # Name the dropped key so this is discoverable in the log.
                    logger.warning(
                        f"checkpoint save: could not serialize driver_config[{k!r}] "
                        f"(type {type(v).__name__}); it will be MISSING on load.",
                        exc_info=True,
                    )

        # rng_state
        rng_grp = f.create_group("rng_state")
        for k, v in rng_state.items():
            if v is None:
                continue
            if isinstance(v, (np.ndarray,)):
                rng_grp.create_dataset(k, data=v)
            elif isinstance(v, (int, float, bool, str)):
                rng_grp.attrs[k] = v

        # walker_state
        ws_grp = f.create_group("walker_state")
        for k, v in walker_state.items():
            if v is None:
                continue
            if isinstance(v, np.ndarray):
                ws_grp.create_dataset(k, data=v)
            elif isinstance(v, (int, float, bool, str)):
                ws_grp.attrs[k] = v

        # observables
        obs_grp = f.create_group("observables")
        for k, v in observables.items():
            if v is None:
                continue
            if isinstance(v, dict):
                # param_grads: dict[str, ndarray]
                sub = obs_grp.create_group(k)
                for sk, sv in v.items():
                    if sv is not None and isinstance(sv, np.ndarray) and sv.size > 0:
                        sub.create_dataset(sk, data=sv)
            elif isinstance(v, np.ndarray) and v.size > 0:
                obs_grp.create_dataset(k, data=v)

        # optimizer_state (optional, VMCopt only)
        if optimizer_state is not None:
            opt_grp = f.create_group("optimizer_state")
            method = optimizer_state.get("method")
            if method is not None:
                opt_grp.attrs["method"] = method
            lineage_id = optimizer_state.get("wavefunction_lineage_id")
            if lineage_id is not None:
                opt_grp.attrs["wavefunction_lineage_id"] = lineage_id
            revision = optimizer_state.get("wavefunction_revision")
            if revision is not None:
                opt_grp.attrs["wavefunction_revision"] = int(revision)
            hyperparams = optimizer_state.get("hyperparameters")
            if hyperparams is not None:
                hp_grp = opt_grp.create_group("hyperparameters")
                for k, v in hyperparams.items():
                    if v is not None:
                        try:
                            hp_grp.attrs[k] = v
                        except Exception:
                            # Name an unserializable hyperparameter instead of
                            # silently dropping it from the restart state.
                            logger.warning(
                                f"checkpoint save: could not serialize optimizer hyperparameter {k!r} "
                                f"(type {type(v).__name__}); it will be MISSING on load.",
                                exc_info=True,
                            )


def load_rank_checkpoint(filepath: str, rank: int) -> dict[str, Any]:
    """Read one rank's data from a merged checkpoint HDF5.

    Args:
        filepath: Path to the merged checkpoint (e.g. ``restart.h5``).
        rank: The MPI rank whose data to load.

    Returns:
        A dict with keys: ``driver_config``, ``rng_state``, ``walker_state``,
        ``observables``, ``optimizer_state`` (may be ``None``).
    """
    result: dict[str, Any] = {}
    with h5py.File(filepath, "r") as f:
        rank_grp = f[f"rank_{rank}"]

        # driver_config
        cfg_grp = rank_grp["driver_config"]
        driver_config: dict[str, Any] = {}
        for k in cfg_grp.attrs:
            driver_config[k] = _convert_attr(cfg_grp.attrs[k])
        for k in cfg_grp:
            driver_config[k] = cfg_grp[k][()]
        result["driver_config"] = driver_config

        # rng_state
        rng_grp = rank_grp["rng_state"]
        rng_state: dict[str, Any] = {}
        for k in rng_grp.attrs:
            rng_state[k] = _convert_attr(rng_grp.attrs[k])
        for k in rng_grp:
            rng_state[k] = rng_grp[k][()]
        result["rng_state"] = rng_state

        # walker_state
        ws_grp = rank_grp["walker_state"]
        walker_state: dict[str, Any] = {}
        for k in ws_grp.attrs:
            walker_state[k] = _convert_attr(ws_grp.attrs[k])
        for k in ws_grp:
            walker_state[k] = ws_grp[k][()]
        result["walker_state"] = walker_state

        # observables
        obs_grp = rank_grp["observables"]
        observables: dict[str, Any] = {}
        for k in obs_grp:
            item = obs_grp[k]
            if isinstance(item, h5py.Group):
                # sub-dict (e.g. param_grads)
                sub: dict[str, np.ndarray] = {}
                for sk in item:
                    sub[sk] = item[sk][()]
                observables[k] = sub
            else:
                observables[k] = item[()]
        result["observables"] = observables

        # optimizer_state
        if "optimizer_state" in rank_grp:
            opt_grp = rank_grp["optimizer_state"]
            # The lineage id and revision are always written with the group;
            # an absent one is returned as ``None`` (never defaulted) so the
            # driver loader can reject the damaged rank collectively.
            revision = opt_grp.attrs.get("wavefunction_revision")
            optimizer_state: dict[str, Any] = {
                "method": opt_grp.attrs.get("method"),
                "hyperparameters": None,
                "wavefunction_lineage_id": _convert_attr(opt_grp.attrs.get("wavefunction_lineage_id")),
                "wavefunction_revision": None if revision is None else int(revision),
            }
            if "hyperparameters" in opt_grp:
                hp_grp = opt_grp["hyperparameters"]
                hyperparams = {}
                for k in hp_grp.attrs:
                    hyperparams[k] = _convert_attr(hp_grp.attrs[k])
                optimizer_state["hyperparameters"] = hyperparams
            result["optimizer_state"] = optimizer_state
        else:
            result["optimizer_state"] = None

    return result


def _convert_attr(val: Any) -> Any:
    """Convert HDF5 attribute values to native Python types."""
    if val is None:
        return None
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


# ---------------------------------------------------------------------------
# Merge / read top-level checkpoint
# ---------------------------------------------------------------------------


def merge_rank_checkpoints(
    output_path: str,
    *,
    mpi_size: int,
    driver_type: str,
    hamiltonian_data: Hamiltonian_data,
    tmp_pattern: str = "._restart_rank{rank}.h5",
    cleanup: bool = True,
) -> None:
    """Merge per-rank temporary HDF5 files into a single checkpoint.

    Called by rank 0 after the MPI barrier.

    Args:
        output_path: Destination path (e.g. ``restart.h5``).
        mpi_size: Total number of MPI ranks.
        driver_type: ``"MCMC"``, ``"GFMC_t"``, or ``"GFMC_n"``.
        hamiltonian_data: Shared Hamiltonian data (saved once at root).
        tmp_pattern: Filename pattern with ``{rank}`` placeholder.
        cleanup: Whether to remove temporary files after merging.
    """
    # Never publish a restartable checkpoint whose spins are not an
    # opposite-twist pair (the restart reader refuses it as well).
    from .determinant import require_opposite_spin_twists

    require_opposite_spin_twists(hamiltonian_data.wavefunction_data.geminal_data)
    use_in_place_merge = mpi_size == 1 and cleanup
    rank_zero_path = tmp_pattern.format(rank=0)
    with ExitStack() as staging:
        if use_in_place_merge:
            # A single-rank temporary already contains every large dataset
            # needed by the final checkpoint. Relinking its root groups
            # avoids copying the complete observable history.
            checkpoint_path = rank_zero_path
            checkpoint_mode = "r+"
        else:
            staging_dir = staging.enter_context(
                TemporaryDirectory(
                    dir=os.path.dirname(os.path.abspath(output_path)), prefix=".jqmc-merge-", ignore_cleanup_errors=True
                )
            )
            checkpoint_path = os.path.join(staging_dir, "checkpoint.h5")
            checkpoint_mode = "w"

        with h5py.File(checkpoint_path, checkpoint_mode) as out:
            if use_in_place_merge:
                rank_data_names = tuple(out.keys())
                out.create_group("rank_0")
                for group_name in rank_data_names:
                    out.move(group_name, f"rank_0/{group_name}")

            meta = out.create_group("_meta")
            meta.attrs["driver_type"] = driver_type
            meta.attrs["mpi_size"] = mpi_size
            from ._build_info import build_info

            info = build_info()
            meta.attrs["jqmc_version"] = info.version
            meta.attrs["jqmc_git_commit"] = info.git_commit
            meta.attrs["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

            hamiltonian_group = out.create_group("hamiltonian_data")
            _save_dataclass_to_hdf5(hamiltonian_group, hamiltonian_data)

            if not use_in_place_merge:
                for rank in range(mpi_size):
                    with h5py.File(tmp_pattern.format(rank=rank), "r") as tmp:
                        out.copy(tmp, f"rank_{rank}")

        # Closing HDF5 successfully is part of staging the complete file.
        if not use_in_place_merge:
            os.replace(checkpoint_path, output_path)

    if cleanup and not use_in_place_merge:
        for rank in range(mpi_size):
            os.remove(tmp_pattern.format(rank=rank))

    if use_in_place_merge and os.path.abspath(rank_zero_path) != os.path.abspath(output_path):
        try:
            os.replace(rank_zero_path, output_path)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            # A custom tmp_pattern may cross filesystems, where a physical
            # copy is unavoidable. Production keeps both paths together.
            shutil.copyfile(rank_zero_path, output_path)
            os.remove(rank_zero_path)


def load_hamiltonian_from_checkpoint(filepath: str) -> Hamiltonian_data:
    """Load only the hamiltonian_data from a checkpoint file.

    Useful for post-processing tools that need the wavefunction but not
    the per-rank QMC state. Reads the actual Hamiltonian payload and re-expands every
    periodic ``AOs_*_data`` instance to its image-tile form so the loaded
    Hamiltonian is immediately usable by AO evaluation kernels.

    Args:
        filepath: Path to the merged checkpoint HDF5.

    Returns:
        Hamiltonian_data instance.
    """
    from .hamiltonians import _load_dataclass_from_hdf5

    with h5py.File(filepath, "r") as f:
        obj = _load_dataclass_from_hdf5(Hamiltonian_data, f["hamiltonian_data"])
    obj = _apply_image_expansion_post_load(obj)
    obj = _apply_ewald_rebuild_post_load(obj)
    obj = _apply_jastrow_structure_backfill_post_load(obj)
    # AO basis exponents/coefficients are fp64 model storage in both precision
    # modes, as in Hamiltonian_data.load_from_hdf5 (issue #124): a checkpoint
    # written while the mixed loader rounded them to float32 is widened
    # exactly, and fp64 datasets are kept bit-for-bit.
    obj = _apply_ao_basis_dtype_alignment_post_load(obj)
    # Mirror Hamiltonian_data.load_from_hdf5's post-fixup validation.  In
    # particular, a tied-spin occupied SD must not carry an independently
    # modified down-spin leaf: the optimizer would otherwise replace it by
    # conj(C_up) at the first update and discontinuously change the restarted
    # wavefunction.
    # Every geminal class must also be an opposite-twist spin pair: a
    # checkpoint written with both spins at one twist is refused here, before
    # any restarted sampling.
    from .determinant import SD_data, require_opposite_spin_twists

    loaded_geminal = obj.wavefunction_data.geminal_data
    if isinstance(loaded_geminal, SD_data):
        loaded_geminal.sanity_check()
        if loaded_geminal.is_tiled_representation:
            obj.sanity_check()
    require_opposite_spin_twists(loaded_geminal)
    return obj


def load_checkpoint_meta(filepath: str) -> dict[str, Any]:
    """Read the ``_meta`` group from a checkpoint.

    Args:
        filepath: Path to the merged checkpoint HDF5.

    Returns:
        Dict with keys: ``driver_type``, ``mpi_size``,
        ``jqmc_version``, ``timestamp``; checkpoints written since issue #183
        also carry ``jqmc_git_commit``.
    """
    meta: dict[str, Any] = {}
    with h5py.File(filepath, "r") as f:
        grp = f["_meta"]
        for k in grp.attrs:
            meta[k] = _convert_attr(grp.attrs[k])
    return meta


def get_checkpoint_num_ranks(filepath: str) -> int:
    """Return the number of MPI ranks stored in a checkpoint.

    Args:
        filepath: Path to the merged checkpoint HDF5.

    Returns:
        Number of ``rank_*`` groups.
    """
    with h5py.File(filepath, "r") as f:
        return sum(1 for k in f.keys() if k.startswith("rank_"))


def _validate_restart_rank_topology(filepath: str, requested_size: int) -> None:
    """Reject restart topologies that would discard or duplicate rank state.

    ``requested_size`` belongs to the sampling communicator, not necessarily
    COMM_WORLD: a twist checkpoint stores one intra-twist communicator.
    Raw checkpoint readers and postprocessors may still read individual ranks.
    This check runs only at the driver/CLI restart boundary, never in a sampler
    loop. It validates group names and metadata without loading their payloads.
    """
    integer_types = (int, np.integer)
    if isinstance(requested_size, (bool, np.bool_)) or not isinstance(requested_size, integer_types) or requested_size < 1:
        raise ValueError("Restart sampling communicator size must be a positive integer.")
    with h5py.File(filepath, "r") as checkpoint:
        if "_meta" not in checkpoint or not isinstance(checkpoint["_meta"], h5py.Group):
            raise ValueError("Checkpoint rank metadata is missing or invalid.")
        stored_size = checkpoint["_meta"].attrs.get("mpi_size")
        if isinstance(stored_size, (bool, np.bool_)) or not isinstance(stored_size, integer_types) or stored_size < 1:
            raise ValueError("Checkpoint rank metadata mpi_size must be a positive integer.")
        rank_names = {name for name in checkpoint if name.startswith("rank_")}
        if len(rank_names) != stored_size:
            raise ValueError("Checkpoint rank metadata does not match the stored rank groups.")
        # Check the count first: corrupt, enormous mpi_size metadata must not
        # allocate an enormous range before being rejected.
        if rank_names != {f"rank_{rank}" for rank in range(len(rank_names))}:
            raise ValueError("Checkpoint rank groups must be contiguous rank_0 through rank_(mpi_size-1).")
        if any(not isinstance(checkpoint[name], h5py.Group) for name in rank_names):
            raise ValueError("Checkpoint rank entries must be HDF5 groups.")
        if stored_size != requested_size:
            raise ValueError(
                f"Restart MPI size mismatch: checkpoint has {stored_size} ranks, "
                f"sampling communicator has {requested_size}. Restart with the original sampling rank count."
            )


# ---------------------------------------------------------------------------
# Restart-state consistency across ranks (issues #805, #806, #807)
# ---------------------------------------------------------------------------


def missing_restart_keys(group: str, mapping: Mapping[str, Any], keys: Iterable[str]) -> list[str]:
    """Return ``group/key`` for each required restart key absent from ``mapping``.

    Restart state that the current writer always writes is required: a
    checkpoint lacking it is damaged or legacy and is rejected, never
    defaulted.
    """
    return [f"{group}/{key}" for key in keys if key not in mapping]


def history_length_errors(
    observables: Mapping[str, Any], names: Iterable[str], expected_rows: int, counter: int
) -> list[tuple[str, str]]:
    """Compare stored history row counts with what the restored step counter requires.

    An empty history is not written, so an absent observable has zero rows.
    """
    errors = []
    for name in names:
        rows = int(np.shape(observables[name])[0]) if name in observables and np.size(observables[name]) else 0
        if rows != expected_rows:
            errors.append(
                (
                    f"observables/{name}",
                    f"stores {rows} rows, but driver_config/mcmc_counter = {counter} requires {expected_rows}",
                )
            )
    return errors


def load_rank_checkpoint_for_restart(filepath: str, rank: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Read one rank group for a driver restart without raising on that rank alone.

    Returns ``(data, problems)``. When a required sub-group of ``rank_{rank}``
    is absent, ``data`` is ``None`` and ``problems`` names it, so the caller
    passes it to :func:`raise_on_restart_inconsistency` together with its
    peers instead of raising before them.
    """
    try:
        return load_rank_checkpoint(filepath, rank=rank), {}
    except KeyError as exc:
        return None, {"errors": [("", f"required checkpoint group is missing ({exc})")]}


def walker_shape_errors(
    walker_state: Mapping[str, Any],
    *,
    num_walkers: Any,
    num_electron_up: int,
    num_electron_dn: int,
    filepath: str,
) -> list[tuple[str, str]]:
    """:func:`validate_restored_walker_shapes` as a rank-local problem list (empty when valid)."""
    try:
        validate_restored_walker_shapes(
            walker_state["latest_r_up_carts"],
            walker_state["latest_r_dn_carts"],
            num_walkers=num_walkers,
            num_electron_up=num_electron_up,
            num_electron_dn=num_electron_dn,
            filepath=filepath,
        )
    except ValueError as exc:
        return [("walker_state", str(exc))]
    return []


def _agreement_value(value: Any) -> Any:
    """Return a picklable, ``==``-comparable form of a restored value."""
    if isinstance(value, np.ndarray):
        return ("ndarray", str(value.dtype), value.shape, value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def restart_state_problems(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the restart-state problems found in the gathered per-rank records.

    Pure function of the records every rank gathered, so every rank derives
    the same list. Each record holds ``rank`` (the checkpoint group index),
    ``missing`` (``group/key`` paths absent on that rank), ``errors``
    (``(group, message)`` pairs) and ``agreed`` (values that must be identical
    on every rank). A value absent on some rank is reported through ``missing``
    by the caller and is not compared.
    """
    problems = []
    for record in records:
        rank = record["rank"]
        problems.extend(f"rank_{rank}/{path}: required restart key is missing" for path in record["missing"])
        problems.extend(f"rank_{rank}{'/' + group if group else ''}: {message}" for group, message in record["errors"])
    keys = sorted({key for record in records for key in record["agreed"]})
    for key in keys:
        values = [(record["rank"], record["agreed"][key]) for record in records if key in record["agreed"]]
        if len(values) == len(records) and any(value != values[0][1] for _, value in values[1:]):
            listing = ", ".join(f"rank_{rank}={value!r}" for rank, value in values)
            problems.append(f"{key} differs across ranks: {listing}")
    return problems


def raise_on_restart_inconsistency(
    comm: Any,
    *,
    context: str,
    rank: int,
    missing: Sequence[str] = (),
    errors: Sequence[tuple[str, str]] = (),
    agreed: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Raise the same ``ValueError`` on every rank when any rank's restart state is unusable.

    Every rank of ``comm`` must call this at the same point, before any
    decision taken from restored state reaches a collective: one allgather of
    the per-rank records, then :func:`restart_state_problems` on the gathered
    list. No rank raises alone, and no default replaces a missing key.

    Args:
        comm: Sampling communicator whose ranks restore one group each.
        context: What is being restored, e.g. ``"MCMC restart from 'restart.h5'"``.
        rank: This rank's checkpoint group index (``rank_{rank}``).
        missing: This rank's absent required ``group/key`` paths.
        errors: This rank's other ``(group, message)`` problems.
        agreed: Values that must be identical on every rank.

    Returns:
        The gathered records (rank order), so that a caller can take a
        collective decision, e.g. a fresh optimizer lineage when every rank
        lacks one.
    """
    record = {
        "rank": int(rank),
        "missing": [str(path) for path in missing],
        "errors": [(str(group), str(message)) for group, message in errors],
        "agreed": {str(key): _agreement_value(value) for key, value in (agreed or {}).items()},
    }
    records = comm.allgather(record) if comm.Get_size() > 1 else [record]
    problems = restart_state_problems(records)
    if problems:
        raise ValueError(
            f"{context} rejected on every rank: the checkpoint is damaged, legacy, or inconsistent across ranks, "
            "and restart state is never defaulted. Restart from an intact checkpoint or start a fresh run.\n  "
            + "\n  ".join(problems)
        )
    return records


# ---------------------------------------------------------------------------
# Observable-only reading helpers (for post-processing)
# ---------------------------------------------------------------------------


def load_driver_config_from_checkpoint(
    filepath: str,
    rank: int = 0,
) -> dict[str, Any]:
    """Read ``driver_config`` attributes for a single rank.

    This is a lightweight way to inspect execution parameters (e.g.
    ``comput_position_deriv``, ``alat``, ``num_gfmc_collect_steps``)
    without deserialising the full driver object.

    Args:
        filepath: Path to the merged checkpoint HDF5.
        rank: MPI rank whose config to read (default 0).

    Returns:
        Dict of driver-config key->value pairs.
    """
    config: dict[str, Any] = {}
    with h5py.File(filepath, "r") as f:
        grp = f[f"rank_{rank}/driver_config"]
        for k in grp.attrs:
            config[k] = _convert_attr(grp.attrs[k])
    return config


def accumulated_weight_split(
    w_L: np.ndarray,
    num_gfmc_collect_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the accumulated GFMC weights as binary mantissas and exponents.

    Pure-NumPy equivalent of :func:`jqmc.jqmc_gfmc._compute_G_L_split`. For
    each index *i* the window product ``prod(w_L[i : i + k])`` (*k* =
    ``num_gfmc_collect_steps``) equals ``mantissa[i] * 2**exponent[i]``. Every
    factor is split by ``frexp`` and the running mantissa is renormalized after
    each multiplication, so the integer exponent carries the whole magnitude and
    no intermediate overflows or underflows. Scaling by powers of two is exact,
    so ``ldexp(mantissa, exponent)`` reproduces the sequential product bit for
    bit whenever that product is a normal float64 number.

    Args:
        w_L: Raw weight array, shape ``(A, x)``.
        num_gfmc_collect_steps: Window length *k*.

    Returns:
        ``(mantissa, exponent)``, each of shape ``(max(A - k, 0), x)``.
    """
    w_L = np.asarray(w_L, dtype=np.float64)
    A, x = w_L.shape
    k = num_gfmc_collect_steps
    n_out = max(A - k, 0)
    factor_mantissa, factor_exponent = np.frexp(w_L)
    mantissa = np.ones((n_out, x), dtype=np.float64)
    exponent = np.zeros((n_out, x), dtype=np.int64)
    for i in range(k if n_out > 0 else 0):
        mantissa, renormalization = np.frexp(mantissa * factor_mantissa[i : i + n_out])
        exponent += factor_exponent[i : i + n_out] + renormalization
    return mantissa, exponent


def scale_accumulated_weights(splits: Sequence[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """Form accumulated weights that share one power-of-two scale.

    ``splits`` holds the ``(mantissa, exponent)`` pairs of every weight block
    entering one estimate (for example the retained rows of each MPI rank).
    The largest exponent ``M`` among the nonzero finite mantissas of all blocks
    is subtracted before ``ldexp``, i.e. every weight is multiplied by the
    common exact factor ``2**-M``. The common factor cancels in every weighted
    mean and jackknife ratio, so those estimates equal the unscaled ones, but
    the weights stay below one however long the collect window is.

    Args:
        splits: Sequence of ``(mantissa, exponent)`` pairs.

    Returns:
        The scaled weights of every block, in the input order.
    """
    candidates = [exponent[(mantissa != 0) & np.isfinite(mantissa)] for mantissa, exponent in splits]
    nonempty = [values for values in candidates if values.size > 0]
    shift = max(int(np.max(values)) for values in nonempty) if nonempty else 0
    return [np.ldexp(mantissa, exponent - shift) for mantissa, exponent in splits]


def compute_accumulated_weights(
    w_L: np.ndarray,
    num_gfmc_collect_steps: int,
) -> np.ndarray:
    """Compute accumulated GFMC weights from raw per-step weights.

    Pure-NumPy equivalent of :func:`jqmc.jqmc_gfmc._compute_G_L`.
    For each index *i* the accumulated weight is
    ``prod(w_L[i : i + k])`` where *k* = ``num_gfmc_collect_steps``.
    This unscaled product leaves the float64 range once ``k * tau * |E|``
    exceeds about 709; estimators use
    :func:`compute_retained_accumulated_weights` instead.

    Args:
        w_L: Raw weight array, shape ``(A, x)``.
        num_gfmc_collect_steps: Window length *k*.

    Returns:
        Accumulated weights, shape ``(A - k, x)``.
    """
    mantissa, exponent = accumulated_weight_split(w_L, num_gfmc_collect_steps)
    return np.ldexp(mantissa, exponent)


def compute_retained_accumulated_weights(
    w_L_list: Sequence[np.ndarray],
    num_gfmc_collect_steps: int,
    num_gfmc_warmup_steps: int,
) -> list[np.ndarray]:
    """Accumulated weights of the retained rows of several histories, on one common scale.

    For every history the rows after ``num_gfmc_warmup_steps`` of
    :func:`accumulated_weight_split` are kept, and all of them are scaled by
    the single power of two of :func:`scale_accumulated_weights`. Use this for
    every estimate that combines the histories in one ratio.

    Args:
        w_L_list: Raw weight arrays, each of shape ``(A_r, x_r)``.
        num_gfmc_collect_steps: Window length *k*.
        num_gfmc_warmup_steps: Number of leading accumulated rows to discard.

    Returns:
        The scaled retained weights of every history, in the input order.
    """
    splits = []
    for w_L in w_L_list:
        mantissa, exponent = accumulated_weight_split(w_L, num_gfmc_collect_steps)
        splits.append((mantissa[num_gfmc_warmup_steps:], exponent[num_gfmc_warmup_steps:]))
    return scale_accumulated_weights(splits)


def load_observables_from_checkpoint(
    filepath: str,
    keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load observable datasets from all ranks without unpickling driver state.

    This is the fast-path for post-processing: only the requested HDF5
    datasets are read, avoiding full object deserialisation.

    Args:
        filepath: Path to the merged checkpoint HDF5.
        keys: Observable names to load (e.g. ``["e_L", "w_L"]``).
            If ``None``, all observables are loaded.

    Returns:
        A list of dicts, one per rank, each mapping observable name to ndarray.
    """
    result = []
    with h5py.File(filepath, "r") as f:
        num_ranks = sum(1 for k in f.keys() if k.startswith("rank_"))
        for rank in range(num_ranks):
            obs_grp = f[f"rank_{rank}/observables"]
            rank_obs: dict[str, Any] = {}
            iter_keys = keys if keys is not None else list(obs_grp.keys())
            for k in iter_keys:
                if k in obs_grp:
                    item = obs_grp[k]
                    if isinstance(item, h5py.Group):
                        sub: dict[str, np.ndarray] = {}
                        for sk in item:
                            sub[sk] = item[sk][()]
                        rank_obs[k] = sub
                    else:
                        rank_obs[k] = item[()]
            result.append(rank_obs)
    return result


@contextmanager
def open_observable_datasets(filepath: str, keys: Sequence[str]) -> Iterator[list[dict[str, h5py.Dataset]]]:
    """Open observable datasets of every rank without reading them.

    Lazy counterpart of :func:`load_observables_from_checkpoint` for arrays
    too large to hold whole, such as the per-atom force observables of a long
    run (issue 949): the caller reads row slices while the file is open.
    Ranks are enumerated and absent keys are omitted exactly as there.

    Args:
        filepath: Path to the merged checkpoint HDF5.
        keys: Observable dataset names to open.

    Yields:
        A list of dicts, one per rank, each mapping observable name to its
        open ``h5py.Dataset``. The handles are valid only inside the context.
    """
    with h5py.File(filepath, "r") as f:
        num_ranks = sum(1 for k in f.keys() if k.startswith("rank_"))
        result = []
        for rank in range(num_ranks):
            obs_grp = f[f"rank_{rank}/observables"]
            result.append({k: obs_grp[k] for k in keys if k in obs_grp})
        yield result


def validate_restored_walker_shapes(
    r_up_carts: Any,
    r_dn_carts: Any,
    *,
    num_walkers: int,
    num_electron_up: int,
    num_electron_dn: int,
    filepath: str,
) -> None:
    """Reject malformed walker arrays before restart execution.

    A checkpoint whose walker block was written for another electron count or
    walker count would otherwise be accepted and only fail (or silently
    misbehave) deep inside the first jitted kernel.

    Raises:
        ValueError: If either array is not ``(num_walkers, n_electrons, 3)``.
    """
    for label, arr, n_elec in (
        ("latest_r_up_carts", r_up_carts, num_electron_up),
        ("latest_r_dn_carts", r_dn_carts, num_electron_dn),
    ):
        shape = tuple(np.shape(arr))
        if n_elec == 0 and int(np.size(arr)) == 0:
            continue
        expected = (int(num_walkers), int(n_elec), 3)
        if shape != expected:
            raise ValueError(
                f"{filepath}: restored walker array {label} has shape {shape}, but the checkpoint's "
                f"Hamiltonian and driver config require {expected} (num_walkers, electrons, 3). "
                "The checkpoint is inconsistent; start a fresh run."
            )


def validate_restored_walker_finiteness_collectively(
    comm: Any,
    *,
    driver: str,
    filepath: str,
    r_up_carts: Any,
    r_dn_carts: Any,
) -> None:
    """Reject poisoned restart coordinates identically on every sampling rank.

    Shape validation must run first so each coordinate index has the documented
    ``(walker, electron, Cartesian component)`` meaning.  The common finite
    path performs one integer reduction and no object collection.  Detailed
    records are gathered only after that reduction proves at least one rank is
    poisoned; no rank raises before its peers reach the same boundary.

    Raises:
        FloatingPointError: Collectively when any restored coordinate is NaN
            or infinite.  The message identifies every affected sampling rank
            and the first coordinate indices on that rank.
    """
    sampling_rank = int(comm.Get_rank())
    local_bad_count = 0
    local_records: list[str] = []
    for field, raw_values in (
        ("latest_r_up_carts", r_up_carts),
        ("latest_r_dn_carts", r_dn_carts),
    ):
        values = np.asarray(raw_values)
        bad_indices = np.argwhere(~np.isfinite(values))
        local_bad_count += int(bad_indices.shape[0])
        for walker, electron, component in bad_indices[:8]:
            local_records.append(
                f"sampling_rank={sampling_rank}, field={field}, walker={int(walker)}, "
                f"electron={int(electron)}, component={int(component)}, "
                f"value={values[walker, electron, component]!r}"
            )

    global_bad_count = int(comm.allreduce(local_bad_count)) if comm.Get_size() > 1 else local_bad_count
    if global_bad_count == 0:
        return
    gathered = comm.allgather(tuple(local_records)) if comm.Get_size() > 1 else [tuple(local_records)]
    records = [record for rank_records in gathered for record in rank_records]
    details = "\n  ".join(records[:16])
    omitted = global_bad_count - len(records[:16])
    if omitted > 0:
        details += f"\n  ... and {omitted} more non-finite coordinate(s)"
    raise FloatingPointError(
        f"{driver} checkpoint load detected {global_bad_count} non-finite restored walker coordinate(s) "
        f"in {filepath}; refusing to continue a poisoned checkpoint. Start a fresh run from finite coordinates.\n  "
        f"{details}"
    )
