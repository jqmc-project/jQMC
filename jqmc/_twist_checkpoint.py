"""Per-twist checkpoint layout for a twist-averaged run.

Two files live under a single ``checkpoint_dir`` for a twist-averaged run:

* ``checkpoint_dir/twist_{idx:03d}.h5`` -- one per twist, written by that
  twist's ``intra_comm`` root. Same format as a regular single-twist
  ``restart.h5`` (the existing :func:`jqmc._checkpoint.merge_rank_checkpoints`
  output) plus checkpoint-manifest attrs under ``_meta``: ``twist_index``,
  ``target_steps`` (the ``num_mcmc_steps`` requested for this twist), and
  ``converged`` (boolean, set by the driver when the twist finishes).
  ``normalized_twist_weight`` and ``num_twists`` make the directory
  self-contained for scientifically correct BZ post-processing.
  Per-twist observable summary (``E_k``, ``sigma_k``, and optionally
  ``dE_mpc_k``, ``sigma_dE_mpc_k``, ``E_mpc_k``, ``sigma_E_mpc_k``,
  ``mpc_cutoff``,
  ``F_k``, ``sigma_F_k``, ``T_k``, ``sigma_T_k``, ``P_hydro_k``,
  ``sigma_P_hydro_k``) is stored under ``_meta`` so that on resume the
  BZ-mean aggregator does not have to re-run any completed twist.
* ``checkpoint_dir/shared_state.h5`` -- a single file written by the
  ``inter_comm`` root holding only the SHARED Jastrow block (under
  ``/jastrow_data``). For non-optimizing drivers (VMC twist averaging,
  GFMC) the file is redundant relative to the per-twist files (the
  Jastrow never moves) but is kept for layout symmetry and so the
  twist-averaged optimizer (where the Jastrow moves every SR step)
  has a canonical single-source-of-truth location for it.

Resume predicate (per twist): a twist is "complete" iff
``twist_{idx}.h5`` exists, its ``_meta/converged`` attr is ``True``,
and its ``_meta/target_steps`` attr is at least the caller's
``target_steps``. A driver resume also passes its analysis settings: the
stored ``target_steps``, ``num_mcmc_warmup_steps`` and
``num_mcmc_bin_blocks`` must then equal the request, because the stored
``E_k`` is one analysis of the stored chain. Partial files are NOT
auto-resumed here -- only the skip-completed-twists path is implemented;
mid-stream resume of a single twist (load the walker state +
``mcmc_counter`` and continue) is left to the individual drivers.
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

import os
import re
import shutil
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import numpy.typing as npt

from ._setting import ATOL_TWIST_CHECKPOINT_KPOINT_MATCH, RTOL_BIT_EXACT
from .hamiltonians import Hamiltonian_data
from .jastrow_factor import Jastrow_data

if TYPE_CHECKING:
    from mpi4py import MPI

    from .coulomb_potential import Coulomb_potential_data
    from .jqmc_gfmc import GFMC_n, GFMC_t
    from .jqmc_mcmc import MCMC

__all__ = [
    "twist_checkpoint_path",
    "shared_state_path",
    "is_twist_complete",
    "reject_twist_checkpoints_beyond",
    "invalidate_twist_checkpoints",
    "save_twist_checkpoint",
    "load_twist_checkpoint_summary",
    "load_twist_checkpoint_manifest",
    "save_shared_state",
    "load_shared_state",
    "write_twist_optimized_hamiltonian",
]

_TWIST_FMT = "twist_{idx:03d}.h5"
# The canonical per-twist file names that the directory manifest combines.
_TWIST_RE = re.compile(r"^twist_(\d{3})\.h5$")
_SHARED_NAME = "shared_state.h5"
_TMP_FMT = "._twist_{idx:03d}_rank{rank}.h5"

_ECP_FIELDS = (
    "ecp_flag",
    "z_cores",
    "max_ang_mom_plus_1",
    "num_ecps",
    "ang_moms",
    "nucleus_index",
    "exponents",
    "coefficients",
    "powers",
)


def twist_checkpoint_path(checkpoint_dir: str, twist_idx: int) -> str:
    """Path of the per-twist merged checkpoint file."""
    return os.path.join(checkpoint_dir, _TWIST_FMT.format(idx=int(twist_idx)))


def shared_state_path(checkpoint_dir: str) -> str:
    """Path of the shared (Jastrow-only) checkpoint file."""
    return os.path.join(checkpoint_dir, _SHARED_NAME)


def twist_identity_digest(hamiltonian_data: Hamiltonian_data) -> str:
    """Cheap strong identity of the physics behind a per-twist checkpoint.

    sha256 over the whole structure and the whole wavefunction: every Jastrow
    component (J1/J2/J3/DTN, parameters and static settings, presence or
    absence), the up- and dn-spin orbitals and, for AGP, the lambda matrix.
    Re-optimizing any variational parameter (or pointing resume at a
    different system) changes the digest, so stale per-twist energies can
    never be blended into a BZ average silently.

    Dataclasses (the flax ``struct`` containers) are walked field by field,
    static (``pytree_node=False``) fields included, with the class and field
    names mixed in, so adding or removing an optional component changes the
    digest. Arrays are normalized to a fixed-width dtype of the same kind
    (bool, int64, float64 or complex128) and hashed with their kind and
    shape, so a complex array and its real part never collide. The digest is
    a pure host-side function of the values, independent of backend and
    precision.
    """
    import dataclasses
    import hashlib

    digest = hashlib.sha256()

    def feed(value: Any) -> None:
        if value is None or isinstance(value, (str, bytes)):
            digest.update(f"<{type(value).__name__}:{value!r}>".encode())
            return
        if isinstance(value, (bool, int, float, complex)):
            # Python scalars hash like the equal 0-d array, so a parameter
            # that is a Python float in memory and a numpy scalar after an
            # HDF5 round-trip yields the same digest.
            value = np.asarray(value)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            digest.update(f"<dc:{type(value).__module__}.{type(value).__qualname__}>".encode())
            for field in dataclasses.fields(value):
                digest.update(f"<f:{field.name}>".encode())
                feed(getattr(value, field.name))
            digest.update(b"</dc>")
            return
        if isinstance(value, (tuple, list)):
            digest.update(f"<seq:{len(value)}>".encode())
            for item in value:
                feed(item)
            digest.update(b"</seq>")
            return
        if isinstance(value, dict):
            digest.update(f"<map:{len(value)}>".encode())
            for key in sorted(value, key=repr):
                feed(key)
                feed(value[key])
            digest.update(b"</map>")
            return
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            array = np.asarray(value)
            kind = array.dtype.kind
            if kind == "c":
                array = array.astype(np.complex128)
            elif kind == "f":
                array = array.astype(np.float64)
            elif kind in "iu":
                array = array.astype(np.int64)
            elif kind == "b":
                array = array.astype(np.bool_)
            else:
                # String / object arrays (e.g. element symbols after an HDF5
                # round-trip): hash element by element like a sequence.
                digest.update(f"<obj:{array.shape}>".encode())
                feed(array.ravel().tolist())
                return
            digest.update(f"<arr:{kind}:{array.shape}>".encode())
            digest.update(np.ascontiguousarray(array).tobytes())
            return
        raise TypeError(f"twist_identity_digest: unsupported value of type {type(value).__qualname__}")

    digest.update(b"jqmc-twist-identity-v2")
    feed(hamiltonian_data.structure_data)
    feed(hamiltonian_data.wavefunction_data)
    return digest.hexdigest()


def is_twist_complete(
    checkpoint_dir: str,
    twist_idx: int,
    target_steps: int,
    *,
    k_point: npt.ArrayLike | None = None,
    ham_digest: str | None = None,
    ecp: Coulomb_potential_data | None = None,
    mpc_cutoff: float | None = None,
    driver_type: str | None = None,
    sampler_config: Mapping[str, Any] | None = None,
    normalized_twist_weight: float | None = None,
    num_twists: int | None = None,
    analysis_config: Mapping[str, int] | None = None,
) -> bool:
    """Return True iff a previously written ``twist_{idx}.h5`` certifies
    that twist as complete vs ``target_steps``.

    The certification keys on three attrs under ``_meta``: ``converged``
    must be ``True`` and ``target_steps`` must be at least the
    caller's ``target_steps`` value (so re-running with a longer chain
    correctly invalidates the old checkpoint). A resume may also require
    matching driver and sampler settings before reusing a stored estimate.

    ``normalized_twist_weight`` and ``num_twists`` are the BZ weight and the
    twist count of the resuming run. A reused twist keeps its stored values,
    which the directory's BZ average uses, so they must equal the run's
    exactly. A checkpoint without them recomputes, like one without a k point
    or digest; a different stored value raises ``ValueError``.

    ``analysis_config`` maps ``num_mcmc_warmup_steps`` and
    ``num_mcmc_bin_blocks`` to the resuming run's values. A reused twist keeps
    its stored summary, which is the analysis of the stored chain with the
    stored settings, so a resume combines it with freshly analysed twists only
    when the stored ``target_steps`` and both settings equal the request
    (issue 862). A checkpoint without the settings recomputes; a longer stored
    chain or a different stored setting raises ``ValueError``.
    """
    path = twist_checkpoint_path(checkpoint_dir, twist_idx)
    if not os.path.isfile(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            if "_meta" not in f:
                return False
            attrs = f["_meta"].attrs
            converged = bool(attrs.get("converged", False))
            stored_target = int(attrs.get("target_steps", 0))
            if not (converged and stored_target >= int(target_steps)):
                return False
            # A checkpoint written for different
            # physics must never be blended into the BZ average. A checkpoint
            # WITHOUT identity attrs (pre-upgrade) is treated as incomplete
            # (recompute); a checkpoint WITH a mismatching identity is a hard
            # error -- the user pointed resume at the wrong/stale directory.
            if k_point is not None or ham_digest is not None:
                if k_point is not None and "k_point" not in attrs:
                    return False
                if ham_digest is not None and "ham_digest" not in attrs:
                    return False
                if k_point is not None:
                    stored_k = np.asarray(attrs["k_point"], dtype=np.float64)
                    if not np.allclose(
                        stored_k,
                        np.asarray(k_point, dtype=np.float64),
                        atol=ATOL_TWIST_CHECKPOINT_KPOINT_MATCH,
                        rtol=RTOL_BIT_EXACT,
                    ):
                        raise ValueError(
                            f"twist checkpoint {path} was written for k={stored_k.tolist()} but resume "
                            f"expects k={np.asarray(k_point, dtype=np.float64).tolist()}: refusing to blend a "
                            "stale per-twist energy into the BZ average. Clear the checkpoint directory."
                        )
                if ham_digest is not None and str(attrs["ham_digest"]) != str(ham_digest):
                    raise ValueError(
                        f"twist checkpoint {path} was written for a different wavefunction/structure "
                        "(identity digest mismatch): refusing to blend a stale per-twist energy into "
                        "the BZ average. Clear the checkpoint directory after re-optimization."
                    )
            # jqmc-tool combines the directory with the stored BZ weight and
            # twist count of every file; a reused twist keeps them unchanged.
            if num_twists is not None:
                if "num_twists" not in attrs:
                    return False
                stored_count = int(attrs["num_twists"])
                if stored_count != int(num_twists):
                    raise ValueError(
                        f"twist checkpoint {path} was written for num_twists={stored_count} but resume expects "
                        f"num_twists={int(num_twists)}: refusing to blend a stale per-twist energy into the BZ "
                        "average. Clear the checkpoint directory."
                    )
            if normalized_twist_weight is not None:
                if "normalized_twist_weight" not in attrs:
                    return False
                stored_weight = float(attrs["normalized_twist_weight"])
                if stored_weight != float(normalized_twist_weight):
                    raise ValueError(
                        f"twist checkpoint {path} was written for normalized BZ weight {stored_weight!r} but resume "
                        f"expects {float(normalized_twist_weight)!r}: refusing to blend a stale per-twist energy into "
                        "the BZ average. Clear the checkpoint directory."
                    )
            if ecp is not None:
                from .hamiltonians import _load_item

                # Merged checkpoints already store the input ECP, including
                # checkpoints written before this comparison was introduced.
                stored_ecp = f["hamiltonian_data/coulomb_potential_data"]
                for key in _ECP_FIELDS:
                    stored = _load_item(stored_ecp[key]) if key in stored_ecp else stored_ecp.attrs[key]
                    if not np.array_equal(stored, getattr(ecp, key)):
                        raise ValueError(
                            f"twist checkpoint {path} was written for a different ECP ({key} mismatch): "
                            "refusing to blend a stale per-twist energy into the BZ average."
                        )
            # MPC is passive with respect to sampling, but the stored
            # correction depends on its reciprocal-space cutoff.  A
            # pre-MPC checkpoint cannot satisfy an MPC resume, and a
            # different stored cutoff is a hard stale-result error.
            if mpc_cutoff is not None:
                mpc_summary_keys = {
                    "dE_mpc_k",
                    "sigma_dE_mpc_k",
                    "E_mpc_k",
                    "sigma_E_mpc_k",
                    "mpc_cutoff",
                }
                if not mpc_summary_keys <= attrs.keys():
                    return False
                stored_mpc_cutoff = float(attrs["mpc_cutoff"])
                if stored_mpc_cutoff != float(mpc_cutoff):
                    raise ValueError(
                        f"twist checkpoint {path} was written with mpc_cutoff={stored_mpc_cutoff} "
                        f"but resume expects mpc_cutoff={float(mpc_cutoff)}: refusing to reuse stale MPC observables."
                    )
                for key in mpc_summary_keys:
                    if not np.isfinite(float(attrs[key])):
                        raise ValueError(f"twist checkpoint {path} stores a non-finite {key}; refusing to resume from it.")
            if "E_k" in attrs and not np.isfinite(float(attrs["E_k"])):
                raise ValueError(f"twist checkpoint {path} stores a non-finite E_k; refusing to resume from it.")
            if driver_type is not None:
                if "driver_type" not in attrs:
                    return False
                if attrs["driver_type"] != driver_type:
                    raise ValueError(
                        f"twist checkpoint {path} was written by driver {attrs['driver_type']!r}, "
                        f"but resume expects {driver_type!r}: refusing to reuse stale per-twist estimates."
                    )
            if sampler_config is not None:
                config_attrs = f["rank_0/driver_config"].attrs
                for key, expected in sampler_config.items():
                    # GFMC_n evolves E_scf, so the rank config stores the final value.
                    stored = attrs.get("initial_E_scf") if key == "E_scf" else config_attrs.get(key)
                    if stored is None:
                        return False
                    if isinstance(stored, bytes):
                        stored = stored.decode()
                    if not np.array_equal(stored, expected):
                        raise ValueError(
                            f"twist checkpoint {path} has {key}={stored!r}, but resume expects "
                            f"{expected!r}: refusing to reuse stale per-twist estimates."
                        )
            if analysis_config is not None:
                if not set(analysis_config) <= attrs.keys():
                    return False
                stored_settings = {"num_mcmc_steps": stored_target}
                stored_settings.update((key, int(attrs[key])) for key in analysis_config)
                requested_settings = {"num_mcmc_steps": int(target_steps)}
                requested_settings.update((key, int(value)) for key, value in analysis_config.items())
                if stored_settings != requested_settings:
                    raise ValueError(
                        f"twist checkpoint {path} (twist {int(twist_idx)}) was analysed with {stored_settings}, "
                        f"but resume requests {requested_settings}: refusing to combine a per-twist estimate "
                        "analysed with other settings into the BZ average. Resume with the stored settings or "
                        "clear the checkpoint directory."
                    )
            return True
    except (OSError, KeyError):
        return False


def reject_twist_checkpoints_beyond(checkpoint_dir: str, num_twists: int) -> None:
    """Refuse a resume directory holding twist files the run does not own.

    A resume rewrites only ``twist_000.h5`` ... ``twist_{num_twists - 1}.h5``,
    but jqmc-tool combines every canonical ``twist_NNN.h5`` of the directory.
    A file with ``NNN >= num_twists`` (left by a run with more twists) would
    therefore stay in the directory's BZ average, so it raises ``ValueError``.
    """
    if not os.path.isdir(checkpoint_dir):
        return
    extra = sorted(
        name for name in os.listdir(checkpoint_dir) if (match := _TWIST_RE.match(name)) and int(match.group(1)) >= num_twists
    )
    if extra:
        raise ValueError(
            f"twist checkpoint directory {checkpoint_dir} holds {', '.join(extra)} beyond the {int(num_twists)} "
            "twist(s) of this run: refusing to resume, because the directory's BZ average would still combine "
            "them. Clear the checkpoint directory."
        )


def invalidate_twist_checkpoints(checkpoint_dir: str) -> list[str]:
    """Mark every canonical twist file of a fresh run's directory unconverged.

    A fresh run rewrites ``twist_NNN.h5`` only when that twist finishes, but
    jqmc-tool combines every canonical ``twist_NNN.h5`` of the directory. A
    fresh run killed between twist saves would therefore leave converged twists
    of an older run beside the new ones. Setting ``_meta/converged`` to
    ``False`` on every existing file before sampling makes the manifest refuse
    such a directory and a resume re-sample those twists (issue 812). Returns
    the sorted names of the files marked. A file that cannot be opened or
    written raises ``RuntimeError``.
    """
    if not os.path.isdir(checkpoint_dir):
        return []
    names = sorted(name for name in os.listdir(checkpoint_dir) if _TWIST_RE.match(name))
    for name in names:
        path = os.path.join(checkpoint_dir, name)
        try:
            with h5py.File(path, "r+") as f:
                f["_meta"].attrs["converged"] = False
        except (OSError, KeyError) as exc:
            raise RuntimeError(
                f"cannot mark twist checkpoint {path} unconverged before a fresh run ({exc}): refusing to sample, "
                "because the directory's BZ average would still combine it. Clear the checkpoint directory."
            ) from exc
    return names


def save_twist_checkpoint(
    checkpoint_dir: str,
    twist_idx: int,
    mcmc_or_gfmc: MCMC | GFMC_t | GFMC_n,
    *,
    intra_comm: MPI.Comm,
    inter_comm: MPI.Comm | None,
    driver_type: str,
    target_steps: int,
    converged: bool,
    initial_E_scf: float | None = None,
    E_k: float | None = None,
    sigma_k: float | None = None,
    dE_mpc_k: float | None = None,
    sigma_dE_mpc_k: float | None = None,
    E_mpc_k: float | None = None,
    sigma_E_mpc_k: float | None = None,
    mpc_cutoff: float | None = None,
    F_k: np.ndarray | None = None,
    sigma_F_k: np.ndarray | None = None,
    T_k: np.ndarray | None = None,
    sigma_T_k: np.ndarray | None = None,
    P_hydro_k: float | None = None,
    sigma_P_hydro_k: float | None = None,
    k_point: npt.ArrayLike | None = None,
    ham_digest: str | None = None,
    normalized_twist_weight: float | None = None,
    num_twists: int | None = None,
    num_mcmc_warmup_steps: int | None = None,
    num_mcmc_bin_blocks: int | None = None,
) -> None:
    """Write ``checkpoint_dir/twist_{idx}.h5`` for this twist and (on the
    ``inter_comm`` root only) ``checkpoint_dir/shared_state.h5``.

    Each intra rank writes a tmp file via the instance's
    ``save_to_hdf5``; the twist's intra rank 0 merges them via
    :func:`jqmc._checkpoint.merge_rank_checkpoints`. The inter-comm
    root then writes the Jastrow-only shared state. All ranks of the
    twist must call this routine (they all hold tmp data); the
    intra-merge + the shared write happen on rank 0 only.

    Args:
        checkpoint_dir: Existing directory. Created with ``exist_ok``.
        twist_idx: 0-based twist index.
        mcmc_or_gfmc: The per-twist driver instance with
            ``.save_to_hdf5`` and ``.hamiltonian_data``.
        intra_comm: The per-twist intra communicator (one group of
            ranks sampling this twist).
        inter_comm: The all-twists inter communicator (one rank per
            twist). May be ``None`` for ranks that are not twist roots.
        driver_type: ``"MCMC"`` / ``"GFMC_t"`` / ``"GFMC_n"`` -- passed
            through to ``merge_rank_checkpoints``.
        target_steps: ``num_mcmc_steps`` requested for this twist;
            stored as ``_meta/target_steps`` so resume can compare.
        converged: ``True`` iff the twist finished its requested steps
            cleanly. Required to be ``True`` for ``is_twist_complete``
            to certify the twist.
        initial_E_scf: Initial ``E_scf`` of the GFMC run, stored as
            ``_meta/initial_E_scf`` so a resume can compare it with the
            request (GFMC_n evolves ``E_scf``, so the driver config only
            holds the final value). Optional.
        E_k: Per-twist energy mean. Stored under ``_meta`` so resume can
            rebuild the BZ aggregate without re-running any completed
            twist's ``get_E``.
        sigma_k: Per-twist energy std, stored with ``E_k``.
        dE_mpc_k: Per-twist MPC correction (optional).
        sigma_dE_mpc_k: MC error of ``dE_mpc_k`` (optional).
        E_mpc_k: Per-twist covariance-aware MPC-corrected total energy
            (optional).
        sigma_E_mpc_k: Joint MC error of ``E_mpc_k`` (optional).
        mpc_cutoff: Reciprocal-space MPC cutoff associated with the stored
            MPC observables. It is part of resume identity when MPC is
            requested.
        F_k: Per-twist atomic forces (optional).
        sigma_F_k: MC error of ``F_k`` (optional).
        T_k: Per-twist stress tensor (optional).
        sigma_T_k: MC error of ``T_k`` (optional).
        P_hydro_k: Per-twist hydrostatic pressure ``-tr(sigma)/3``
            (optional). Stored SEPARATELY from the stress tensor because
            its error bar is the jackknife over the trace of the shared
            replicates and therefore cannot be reconstructed from
            ``sigma_T_k`` -- the three normal stresses are correlated.
        sigma_P_hydro_k: MC error of ``P_hydro_k`` (optional).
        k_point: Fractional k-point of this twist, stored as
            ``_meta/k_point`` identity attribute so resume refuses a
            checkpoint written for a different twist (optional).
        ham_digest: Digest of the twist's Hamiltonian
            (:func:`twist_identity_digest`), stored as ``_meta/ham_digest``
            identity attribute so resume refuses a checkpoint written for
            different physics (optional).
        normalized_twist_weight: Normalized BZ weight of this twist.  Stored
            so post-processing can reduce the sharded checkpoint without
            reopening the original combined Hamiltonian input.
        num_twists: Total number of twists in the checkpoint directory.
        num_mcmc_warmup_steps: Warmup steps of the stored summary's
            analysis. Resume compares it with the request together with
            ``target_steps``.
        num_mcmc_bin_blocks: Jackknife blocks of the stored summary's
            analysis. Resume compares it with the request together with
            ``target_steps``.
    """
    from ._checkpoint import merge_rank_checkpoints

    os.makedirs(checkpoint_dir, exist_ok=True)

    intra_rank = intra_comm.Get_rank()
    intra_size = intra_comm.Get_size()
    # Scope rank-local temporary files to one twist so concurrent merges do not collide.
    tmp_pattern = os.path.join(checkpoint_dir, _TMP_FMT.format(idx=int(twist_idx), rank="{rank}"))
    tmp_path = tmp_pattern.format(rank=intra_rank)
    mcmc_or_gfmc.save_to_hdf5(tmp_path)
    intra_comm.Barrier()

    out_path = twist_checkpoint_path(checkpoint_dir, twist_idx)
    if intra_rank == 0:
        merge_rank_checkpoints(
            output_path=out_path,
            mpi_size=intra_size,
            driver_type=driver_type,
            hamiltonian_data=mcmc_or_gfmc.hamiltonian_data,
            tmp_pattern=tmp_pattern,
        )
        with h5py.File(out_path, "a") as f:
            meta = f["_meta"]
            meta.attrs["twist_index"] = int(twist_idx)
            meta.attrs["target_steps"] = int(target_steps)
            meta.attrs["converged"] = bool(converged)
            if initial_E_scf is not None:
                meta.attrs["initial_E_scf"] = float(initial_E_scf)
            # Identity attributes make resume refuse checkpoints written for
            # different physics instead of blending stale energies.
            if k_point is not None:
                meta.attrs["k_point"] = np.asarray(k_point, dtype=np.float64)
            if ham_digest is not None:
                meta.attrs["ham_digest"] = str(ham_digest)
            if normalized_twist_weight is not None:
                meta.attrs["normalized_twist_weight"] = float(normalized_twist_weight)
            if num_twists is not None:
                meta.attrs["num_twists"] = int(num_twists)
            if num_mcmc_warmup_steps is not None:
                meta.attrs["num_mcmc_warmup_steps"] = int(num_mcmc_warmup_steps)
            if num_mcmc_bin_blocks is not None:
                meta.attrs["num_mcmc_bin_blocks"] = int(num_mcmc_bin_blocks)
            if E_k is not None:
                meta.attrs["E_k"] = float(E_k)
            if sigma_k is not None:
                meta.attrs["sigma_k"] = float(sigma_k)
            if dE_mpc_k is not None:
                meta.attrs["dE_mpc_k"] = float(dE_mpc_k)
            if sigma_dE_mpc_k is not None:
                meta.attrs["sigma_dE_mpc_k"] = float(sigma_dE_mpc_k)
            if E_mpc_k is not None:
                meta.attrs["E_mpc_k"] = float(E_mpc_k)
            if sigma_E_mpc_k is not None:
                meta.attrs["sigma_E_mpc_k"] = float(sigma_E_mpc_k)
            if mpc_cutoff is not None:
                meta.attrs["mpc_cutoff"] = float(mpc_cutoff)
            if F_k is not None:
                if "F_k" in meta:
                    del meta["F_k"]
                meta.create_dataset("F_k", data=np.asarray(F_k, dtype=np.float64))
            if sigma_F_k is not None:
                if "sigma_F_k" in meta:
                    del meta["sigma_F_k"]
                meta.create_dataset("sigma_F_k", data=np.asarray(sigma_F_k, dtype=np.float64))
            if T_k is not None:
                if "T_k" in meta:
                    del meta["T_k"]
                meta.create_dataset("T_k", data=np.asarray(T_k, dtype=np.float64))
            if sigma_T_k is not None:
                if "sigma_T_k" in meta:
                    del meta["sigma_T_k"]
                meta.create_dataset("sigma_T_k", data=np.asarray(sigma_T_k, dtype=np.float64))
            if P_hydro_k is not None:
                if "P_hydro_k" in meta:
                    del meta["P_hydro_k"]
                meta.create_dataset("P_hydro_k", data=float(P_hydro_k))
            if sigma_P_hydro_k is not None:
                if "sigma_P_hydro_k" in meta:
                    del meta["sigma_P_hydro_k"]
                meta.create_dataset("sigma_P_hydro_k", data=float(sigma_P_hydro_k))

    if inter_comm is not None and inter_comm.Get_rank() == 0:
        save_shared_state(checkpoint_dir, mcmc_or_gfmc.hamiltonian_data)

    intra_comm.Barrier()


def load_twist_checkpoint_summary(checkpoint_dir: str, twist_idx: int) -> dict[str, Any]:
    """Read back the per-twist summary written by
    :func:`save_twist_checkpoint` -- the ``_meta`` attrs and the
    optional ``F_k`` / ``sigma_F_k`` / ``T_k`` / ``sigma_T_k`` /
    ``P_hydro_k`` / ``sigma_P_hydro_k`` datasets. Returns a dict with keys
    ``twist_index``, ``target_steps``, ``converged``, and (when
    present) ``E_k`` / ``sigma_k`` / ``F_k`` / ``sigma_F_k``.
    """
    path = twist_checkpoint_path(checkpoint_dir, twist_idx)
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        meta = f["_meta"]
        attrs = meta.attrs
        out["twist_index"] = int(attrs.get("twist_index", twist_idx))
        out["target_steps"] = int(attrs.get("target_steps", 0))
        out["converged"] = bool(attrs.get("converged", False))
        if "normalized_twist_weight" in attrs:
            out["normalized_twist_weight"] = float(attrs["normalized_twist_weight"])
        if "num_twists" in attrs:
            out["num_twists"] = int(attrs["num_twists"])
        if "E_k" in attrs:
            out["E_k"] = float(attrs["E_k"])
        if "sigma_k" in attrs:
            out["sigma_k"] = float(attrs["sigma_k"])
        for key in ("dE_mpc_k", "sigma_dE_mpc_k", "E_mpc_k", "sigma_E_mpc_k", "mpc_cutoff"):
            if key in attrs:
                out[key] = float(attrs[key])
        if "F_k" in meta:
            out["F_k"] = np.asarray(meta["F_k"])
        if "sigma_F_k" in meta:
            out["sigma_F_k"] = np.asarray(meta["sigma_F_k"])
        if "T_k" in meta:
            out["T_k"] = np.asarray(meta["T_k"])
        if "sigma_T_k" in meta:
            out["sigma_T_k"] = np.asarray(meta["sigma_T_k"])
        if "P_hydro_k" in meta:
            out["P_hydro_k"] = float(np.asarray(meta["P_hydro_k"]))
        if "sigma_P_hydro_k" in meta:
            out["sigma_P_hydro_k"] = float(np.asarray(meta["sigma_P_hydro_k"]))
    return out


def load_twist_checkpoint_manifest(checkpoint_dir: str) -> list[tuple[str, float]]:
    """Return canonical per-twist files and normalized BZ weights.

    Every accepted ``twist_NNN.h5`` must store its normalized weight and the
    total twist count.  Pre-upgrade directories are deliberately unsupported:
    inferring or externally attaching weights could silently combine unrelated
    physics or corrupt symmetry-reduced BZ averages.
    """
    from ._twist_average import normalize_twist_weights

    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"{checkpoint_dir}: expected a twist checkpoint directory.")

    indexed_paths: list[tuple[int, str]] = []
    for name in os.listdir(checkpoint_dir):
        match = _TWIST_RE.match(name)
        if match is not None:
            indexed_paths.append((int(match.group(1)), os.path.join(checkpoint_dir, name)))
    indexed_paths.sort()
    if not indexed_paths:
        raise ValueError(f"{checkpoint_dir}: no canonical twist_NNN.h5 checkpoints were found.")

    summaries = [load_twist_checkpoint_summary(checkpoint_dir, idx) for idx, _path in indexed_paths]
    for (idx, path), summary in zip(indexed_paths, summaries, strict=True):
        if summary["twist_index"] != idx:
            raise ValueError(f"{path}: metadata twist_index={summary['twist_index']} does not match filename index {idx}.")
        if not summary["converged"]:
            raise ValueError(f"{path}: twist is not converged; refusing to form an incomplete BZ average.")

    has_weight = ["normalized_twist_weight" in summary for summary in summaries]
    has_count = ["num_twists" in summary for summary in summaries]
    if not all(has_weight) or not all(has_count):
        raise ValueError(
            f"{checkpoint_dir}: every twist checkpoint must store normalized_twist_weight and num_twists; "
            "pre-upgrade checkpoint directories are unsupported."
        )

    indices = [idx for idx, _path in indexed_paths]
    counts = {summary["num_twists"] for summary in summaries}
    if len(counts) != 1:
        raise ValueError(f"{checkpoint_dir}: inconsistent num_twists metadata: {sorted(counts)}.")
    expected_count = counts.pop()
    weights = np.asarray([summary["normalized_twist_weight"] for summary in summaries], dtype=np.float64)

    expected_indices = list(range(expected_count))
    if indices != expected_indices:
        raise ValueError(
            f"{checkpoint_dir}: incomplete twist checkpoint set; found indices {indices}, expected {expected_indices}."
        )
    if weights.shape != (expected_count,):
        raise ValueError(f"{checkpoint_dir}: expected {expected_count} twist weights, got shape {weights.shape}.")
    weights = normalize_twist_weights(weights)
    return [(path, float(weights[idx])) for idx, path in indexed_paths]


def save_shared_state(checkpoint_dir: str, hamiltonian_data: Hamiltonian_data) -> None:
    """Write the shared Jastrow block of ``hamiltonian_data`` to
    ``checkpoint_dir/shared_state.h5``.

    The file is overwritten atomically (write to tmp + rename) so a
    crash mid-write does not produce a corrupt shared state visible to
    a subsequent resume.
    """
    from .hamiltonians import _save_dataclass_to_hdf5

    hamiltonian_data.wavefunction_data.jastrow_data.sanity_check()
    os.makedirs(checkpoint_dir, exist_ok=True)
    out_path = shared_state_path(checkpoint_dir)
    tmp_path = out_path + ".tmp"
    with h5py.File(tmp_path, "w") as f:
        grp = f.create_group("jastrow_data")
        _save_dataclass_to_hdf5(grp, hamiltonian_data.wavefunction_data.jastrow_data)
    os.replace(tmp_path, out_path)


def write_twist_optimized_hamiltonian(
    source_hamiltonian: str,
    checkpoint_dir: str,
    output_hamiltonian: str,
) -> None:
    """Write the twist-averaged optimizer's combined wavefunction dump.

    A twist-averaged optimisation holds one optimized Hamiltonian per twist
    (``checkpoint_dir/twist_NNN.h5``) plus the world-synchronised shared Jastrow
    (``checkpoint_dir/shared_state.h5``).  A dump written by world rank 0 alone
    would carry ONE twist as a plain single-twist file, which a downstream MCMC
    / LRDMC job would silently consume as a single-k wavefunction.  This routine
    writes the complete artifact instead: the ``/shared`` template, k points and
    BZ weights of ``source_hamiltonian`` (the combined input the run started from),
    its ``/shared/wavefunction_data/jastrow_data`` replaced by the canonical
    shared Jastrow, and every ``/twists/twist_NNN`` AO basis and MO / lambda
    payload replaced by that twist's optimized values. The result is a valid ``hamiltonian_h5``
    input for a fresh run, so any dumped optimisation step is restartable.

    The per-twist checkpoints are used as they are, converged or not: an
    intermediate dump is a complete snapshot of the optimisation so far.  The
    file is written to a temporary path and ``os.replace``-d into place, so a
    crash mid-write never leaves a partial file under the final name.

    Per-twist ``aos_data_up`` / ``aos_data_dn`` override the shared geminal
    basis on load. Checkpoint serialization already de-tiles periodic Gaussian
    leaves, including composite children, so copying these persisted groups
    retains optimized radial parameters without duplicating periodic images.
    """
    from .hamiltonians import _require_twist_average_layout

    with h5py.File(source_hamiltonian, "r") as source:
        n_twist = _require_twist_average_layout(source)

    checkpoint_paths = [twist_checkpoint_path(checkpoint_dir, i) for i in range(n_twist)]
    missing = [path for path in checkpoint_paths if not os.path.isfile(path)]
    if missing:
        raise RuntimeError(
            "Cannot write the optimized twist Hamiltonian: "
            f"{len(missing)}/{n_twist} per-twist checkpoints are missing "
            f"(first missing: {missing[0]})."
        )

    shared_path = shared_state_path(checkpoint_dir)
    if not os.path.isfile(shared_path):
        raise RuntimeError(
            f"Cannot write the optimized twist Hamiltonian: canonical shared Jastrow is missing ({shared_path})."
        )

    def _class_name(group) -> str:
        name = group.attrs.get("_class_name", "")
        return name.decode("utf-8") if isinstance(name, bytes) else str(name)

    def _replace_dataset(group, name: str, values) -> None:
        if name in group:
            del group[name]
        group.create_dataset(name, data=values)

    geminal_path = "hamiltonian_data/wavefunction_data/geminal_data"
    mo_up_path = f"{geminal_path}/orb_data_up_spin/mo_coefficients"
    mo_dn_path = f"{geminal_path}/orb_data_dn_spin/mo_coefficients"

    tmp_output = f"{output_hamiltonian}.tmp"
    try:
        shutil.copyfile(source_hamiltonian, tmp_output)
        with (
            h5py.File(tmp_output, "r+") as output,
            h5py.File(shared_path, "r") as shared_state,
        ):
            shared_wavefunction = output["shared/wavefunction_data"]
            del shared_wavefunction["jastrow_data"]
            shared_state.copy(shared_state["jastrow_data"], shared_wavefunction, name="jastrow_data")
            # The shared geminal remains a template; each twist overrides its
            # full AO trees as well as its MO/lambda parameters below.
            template_geminal_type = _class_name(shared_wavefunction["geminal_data"])
            template_tiled = _class_name(shared_wavefunction["geminal_data/orb_data_up_spin"]) == "TiledOrbitalSet_data"

            for twist_index, checkpoint_path in enumerate(checkpoint_paths):
                with h5py.File(checkpoint_path, "r") as checkpoint:
                    checkpoint_geminal = checkpoint[geminal_path]
                    geminal_type = _class_name(checkpoint_geminal)
                    if geminal_type != template_geminal_type:
                        raise RuntimeError(
                            "Per-twist optimized geminal representations differ: "
                            f"the shared template is {template_geminal_type}, twist {twist_index} is {geminal_type}."
                        )
                    destination = output[f"twists/twist_{twist_index:03d}"]
                    tiled = _class_name(checkpoint_geminal["orb_data_up_spin"]) == "TiledOrbitalSet_data"
                    if tiled != template_tiled:
                        raise RuntimeError("Per-twist checkpoints cannot change the tiled orbital representation.")
                    if tiled:
                        # Copy the complete ragged provider, including local C and
                        # sharing references. No global MO or lambda is constructed.
                        if geminal_type != "SD_data":
                            raise RuntimeError("Tiled twist checkpoints require occupied SD_data.")
                        if "geminal" in destination:
                            del destination["geminal"]
                        checkpoint.copy(checkpoint_geminal, destination, name="geminal")
                        if twist_index == 0:
                            del shared_wavefunction["geminal_data"]
                            shared_wavefunction["geminal_data"] = destination["geminal"]
                        continue
                    for spin in ("up", "dn"):
                        name = f"aos_data_{spin}"
                        if name in destination:
                            del destination[name]
                        checkpoint.copy(checkpoint_geminal[f"orb_data_{spin}_spin/aos_data"], destination, name=name)
                    # The full checkpoint AO trees include the actual PW or
                    # spline children for BOTH spins. Remove source overrides
                    # so they cannot replace this authoritative snapshot.
                    for name in ("composite_child", "composite_child_dn"):
                        if name in destination:
                            del destination[name]
                    _replace_dataset(destination, "mo_coefficients_up", np.asarray(checkpoint[mo_up_path]))
                    _replace_dataset(destination, "mo_coefficients_dn", np.asarray(checkpoint[mo_dn_path]))
                    if "lambda_matrix" in destination:
                        del destination["lambda_matrix"]

                    if geminal_type == "AGP_data":
                        _replace_dataset(destination, "lambda_matrix", np.asarray(checkpoint_geminal["lambda_matrix"]))
                    elif geminal_type == "AGPn_data":
                        diagonal = np.diag(np.asarray(checkpoint_geminal["lambda_diag"]))
                        unpaired = np.asarray(checkpoint_geminal["lambda_unpaired"])
                        mo_dtype = np.asarray(checkpoint[mo_up_path]).dtype
                        if np.issubdtype(mo_dtype, np.complexfloating):
                            diagonal = diagonal.astype(np.complex128, copy=False)
                        lambda_matrix = diagonal if unpaired.shape[1] == 0 else np.hstack((diagonal, unpaired))
                        _replace_dataset(destination, "lambda_matrix", lambda_matrix)
                    elif geminal_type != "SD_data":
                        raise RuntimeError(
                            "Cannot write the optimized twist Hamiltonian: unsupported geminal "
                            f"representation {geminal_type!r}."
                        )
        if template_tiled:
            # Validate sector/cell/spin metadata before atomically publishing the
            # checkpoint ensemble. A mismatched sector must not become restartable.
            from ._twist_average import load_twist

            for twist_index in range(n_twist):
                load_twist(tmp_output, twist_index)
        os.replace(tmp_output, output_hamiltonian)
    except BaseException:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
        raise


def load_shared_state(checkpoint_dir: str) -> Jastrow_data:
    """Load the shared Jastrow block from
    ``checkpoint_dir/shared_state.h5`` and return it as a
    :class:`jqmc.jastrow_factor.Jastrow_data`.
    """
    from .hamiltonians import _load_dataclass_from_hdf5, _restore_shared_jastrow_after_load
    from .jastrow_factor import Jastrow_data

    path = shared_state_path(checkpoint_dir)
    with h5py.File(path, "r") as f:
        return _restore_shared_jastrow_after_load(_load_dataclass_from_hdf5(Jastrow_data, f["jastrow_data"]))
