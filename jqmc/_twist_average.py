"""Twist-averaged BZ-mean aggregator.

Pure-math helpers used by

* the in-driver Level-2 reduction (``run_twist_average`` on ``inter_comm``)
* the offline post-hoc aggregator (consumed for ``--resume`` summaries and
  the ``twist_average_summary.h5`` write)

Why a standalone module: the formula is identical for both call sites and
needs to round-trip byte-for-byte across them so a deterministic test can
compare live vs offline outputs without any custom plumbing. That dual-path
equivalence is what catches ``inter_comm`` reduction / weight-normalisation
bugs.

The estimator is

.. math::
    E_{\\rm TA}\\;=\\;\\sum_k W_k\\,\\bar E_k, \\qquad
    \\sigma(E_{\\rm TA})\\;=\\;\\sqrt{\\sum_k W_k^2\\,\\sigma_k^2},

with deterministic-grid weights :math:`W_k = w_k/\\sum_{k'} w_{k'}`. The
per-twist MC errors combine in *quadrature* because the twists are
**independent** chains; pooling per-twist jackknife bins into one global
jackknife would conflate the **physical twist dispersion**

.. math::
    s^2_{\\rm twist}\\;=\\;\\sum_k W_k\\,(\\bar E_k - E_{\\rm TA})^2

with the MC error and inflate the reported bar -- wrong. ``s_twist`` is a
finite-size diagnostic (the shell-filling oscillation TABC averages out)
and is reported separately, **never folded into** ``sigma(E_TA)``.

The same algebra applies to a vector observable :math:`\\bar F_k` (atomic
force) component-by-component; the optional ``forces`` helper handles
that shape (``n_atoms, 3``) without re-deriving the formula.
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

import inspect
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from logging import getLogger
from types import TracebackType
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from ._setting import MPC_CUTOFF_HA, RTOL_BIT_EXACT
from ._twist_checkpoint import shared_state_path, twist_checkpoint_path
from .atomic_orbital import AOs_composite_data
from .hamiltonians import Hamiltonian_data
from .jastrow_factor import Jastrow_data

if TYPE_CHECKING:
    from mpi4py import MPI

    from .atomic_orbital import AOs_base_data
    from .jqmc_optimizer import EnergyEstimate
    from .molecular_orbital import MOs_data
    from .structure import Structure_data

CompositeChildData: TypeAlias = tuple[Literal["pw_complex"], Sequence[int]] | tuple[Literal["bspline_complex"], npt.ArrayLike]

logger = getLogger("jqmc").getChild(__name__)

__all__ = [
    "TwistAverageEnergy",
    "TwistAverageForce",
    "TwistAverageMPC",
    "TwistAverageResult",
    "TwistCommSplit",
    "TwistInputMeta",
    "is_twist_average_input",
    "is_twist_run_complete",
    "load_shared_jastrow",
    "load_twist",
    "load_twist_average_input",
    "normalize_twist_weights",
    "bz_mean_energy",
    "bz_mean_force",
    "run_twist_average",
    "run_twist_average_from_hamiltonians",
    "run_twist_average_gfmc",
    "run_twist_average_optimize",
    "save_shared_jastrow",
    "save_twist_average_input",
    "shared_state_path",
    "split_world_for_twists",
    "twist_checkpoint_path",
]


# Required ``/_meta`` content attribute of a combined twist input: the spin
# twist pairing of its down-spin payload. Every writer pairs the spin channels
# at opposite twists (up ``+k``, down ``-k``; the down payload is the
# conjugate of its own channel's ``+k`` orbitals) and records that here. It is
# not a format version: the reader dispatches on nothing; it only requires the
# attribute where the payload itself cannot show the pairing -- the down rows
# of an independent-spin (LSDA / UHF) input at a complex twist, whose
# conjugated and unconjugated forms are both valid coefficient arrays and
# whose shared periodic-Gaussian template is undecidable at a Gamma twist.
# Restricted payloads are tied to the up rows, so they need no attribute.
SPIN_TWIST_PAIRING_ATTR = "spin_twist_pairing"
SPIN_TWIST_PAIRING_OPPOSITE = "opposite"


def normalize_twist_weights(weights: npt.ArrayLike) -> np.ndarray:
    """Normalise raw DFT k-point weights to ``W_k = w_k / sum_k w_k``.

    Args:
        weights: Per-twist raw weights ``w_k`` (DFT star multiplicities for a
            symmetry-reduced IBZ, or all-ones for an unreduced full mesh).
            ``None`` / empty / non-positive sum raises ``ValueError``.

    Returns:
        Normalised weights ``W_k`` summing to ``1.0`` (within float64
        round-off).

    Raises:
        ValueError: For empty / non-positive / non-finite weights.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size == 0:
        raise ValueError("normalize_twist_weights: empty weight array.")
    if not np.all(np.isfinite(w)):
        raise ValueError("normalize_twist_weights: weights must be finite.")
    if not np.all(w >= 0.0):
        raise ValueError("normalize_twist_weights: weights must be non-negative.")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("normalize_twist_weights: sum of weights must be positive.")
    return w / total


def _twist_sampler_seed(mcmc_seed: int, twist_index: int, n_twist: int) -> "int | tuple[int, int]":
    """Return the twist-scoped CHAIN-SEED KEY handed to the sampler.

    Hierarchical SeedSequence contract (see
    :func:`jqmc._driver_helpers.derive_rank_rng_streams`): the sampler appends
    its intra-communicator rank to this key and derives its RNG state from
    ``SeedSequence([*key, intra_rank])``. No value is ever collapsed to a
    single uint32 here: a ``generate_state(1)`` collapse produces
    full-range uint32 seeds whose in-sampler multiplicative rank fold
    overflows ``np.random.seed``'s 32-bit domain (ranks then fail
    deterministically and are missing at the Shutdown barrier).

    * ``n_twist > 1`` -> the pair ``(mcmc_seed, twist_index)``; full stream
      key = ``[mcmc_seed, twist_index, intra_rank]``.
    * ``n_twist == 1`` -> the bare ``mcmc_seed`` so the twist orchestrator
      remains byte-identical to a direct sampler call (stream key =
      ``[mcmc_seed, rank]``, same as every non-twist run).
    """
    if n_twist < 1:
        raise ValueError(f"_twist_sampler_seed: n_twist={n_twist} must be >= 1.")
    if not 0 <= twist_index < n_twist:
        raise ValueError(f"_twist_sampler_seed: twist_index={twist_index} must satisfy 0 <= twist_index < n_twist={n_twist}.")
    if n_twist == 1:
        return int(mcmc_seed)
    return (int(mcmc_seed), int(twist_index))


def _twist_sampler_options(driver, section: str, kwargs: Mapping) -> dict:
    """Resolve the frozen sampler values used by the constructor."""
    from ._restart_policy import TOML_TO_CFG

    # Observable upgrades are already handled by the summary availability check.
    observable_flags = {"comput_position_deriv", "comput_stress", "comput_log_WF_param_deriv"}
    keys = {cfg for (group, _), cfg in TOML_TO_CFG.items() if group == section} - observable_flags
    keys.add("random_discretized_mesh")
    parameters = inspect.signature(driver).parameters
    return {key: kwargs[key] if key in kwargs else parameters[key].default for key in sorted(keys)}


def _check_twist_resume(checkpoint_dir: str, twist_index: int, target_steps: int, *, intra_comm, world, **identity) -> bool:
    """Share the twist root's reuse decision and propagate errors before sampling."""
    from ._twist_checkpoint import is_twist_complete

    if world is None and intra_comm.Get_size() == 1:
        return is_twist_complete(checkpoint_dir, twist_index, target_steps, **identity)
    complete = False
    error = None
    if intra_comm.Get_rank() == 0:
        try:
            complete = is_twist_complete(checkpoint_dir, twist_index, target_steps, **identity)
        except ValueError as exc:
            error = str(exc)
    complete, error = intra_comm.bcast((complete, error), root=0)
    errors = (error,) if world is None else world.allgather(error)
    for error in errors:
        if error is not None:
            raise ValueError(error)
    return complete


def _decide_on_world_root(world, check, *args):
    """Run a directory check on world rank 0 before sampling and share its outcome with every rank.

    Every rank waits for the outcome. Any exception on rank 0 is broadcast,
    whether it is the check's own refusal or an unexpected filesystem or HDF5
    error such as an unlistable directory. Every rank then raises together
    instead of waiting in the broadcast for a root that failed alone. A
    ``ValueError`` stays a ``ValueError``; anything else is raised as
    ``RuntimeError`` naming the original type.
    """
    result, error, root_exc = None, None, None
    if world.Get_rank() == 0:
        try:
            result = check(*args)
        except Exception as exc:  # noqa: BLE001 -- re-raised on every rank below
            root_exc = exc
            message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else f"{type(exc).__name__}: {exc}"
            error = (isinstance(exc, ValueError), message)
    result, error = world.bcast((result, error), root=0)
    if error is not None:
        is_value_error, message = error
        raise (ValueError if is_value_error else RuntimeError)(message) from root_exc
    return result


def _reject_twist_checkpoints_beyond(checkpoint_dir: str, n_twist: int, world) -> None:
    """Refuse, on every rank and before sampling, a resume directory with twist files beyond ``n_twist``."""
    from ._twist_checkpoint import reject_twist_checkpoints_beyond

    _decide_on_world_root(world, reject_twist_checkpoints_beyond, checkpoint_dir, n_twist)


def _invalidate_twist_checkpoints(checkpoint_dir: str, world) -> None:
    """Mark the twist files of a fresh run's directory unconverged, on world rank 0 and before sampling.

    No twist is saved before the old files are marked, and a failure raises on every rank.
    """
    from ._twist_checkpoint import invalidate_twist_checkpoints

    names = _decide_on_world_root(world, invalidate_twist_checkpoints, checkpoint_dir)
    if world.Get_rank() == 0 and names:
        logger.info(f"Fresh twist run: marked {', '.join(names)} in {checkpoint_dir} unconverged before sampling.")


@dataclass(frozen=True)
class TwistAverageResult:
    """Aggregate output of :func:`run_twist_average_from_hamiltonians`.

    Attributes:
        energy: BZ-mean energy + dispersion (:class:`TwistAverageEnergy`).
        force: BZ-mean atomic force (:class:`TwistAverageForce` or ``None``
            when atomic forces were not requested).
        per_twist_E: Per-twist mean energy, shape ``(N_twist,)``.
        per_twist_E_std: Per-twist MC std, shape ``(N_twist,)``.
        per_twist_F: Per-twist mean force (or ``None``).
        per_twist_F_std: Per-twist MC std on force (or ``None``).
        k_points: List of per-twist k-points (each entry: tuple[float,
            float, float] or ``None`` at Gamma).
        mpc: Optional BZ-averaged MPC correction and covariance-aware
            corrected total, including the Level-1 per-twist estimates.
    """

    energy: "TwistAverageEnergy"
    force: "TwistAverageForce | None"
    per_twist_E: np.ndarray
    per_twist_E_std: np.ndarray
    per_twist_F: "np.ndarray | None"
    per_twist_F_std: "np.ndarray | None"
    k_points: list
    #: BZ-mean symmetric stress tensor (``get_aT`` per twist), reusing the
    #: :class:`TwistAverageForce` container with leaf shape ``(3, 3)``;
    #: ``None`` unless ``compute_stress=True``.
    stress: "TwistAverageForce | None" = None
    #: BZ-mean generalized (variational-parameter) force (``get_gF`` per
    #: twist), container leaf shape ``(n_param,)``. For SHARED parameters
    #: (the combined twist layout shares one Jastrow across twists) the BZ mean
    #: of the per-twist gradients IS the gradient of the BZ-mean energy:
    #: ``d E_TA / dc = Sum_k W_k dE_k/dc``. ``None`` unless
    #: ``compute_param_grad=True``.
    param_grad: "TwistAverageForce | None" = None
    #: BZ-mean hydrostatic pressure ``-tr(sigma)/3``, container leaf shape
    #: ``(1,)``. Carried SEPARATELY from ``stress`` because its error bar is
    #: the per-twist jackknife over the trace of the shared replicates: the
    #: three normal stresses are strongly correlated, so it cannot be
    #: recovered from ``stress.F_TA_std`` by quadrature.
    P_hydro: "TwistAverageForce | None" = None
    per_twist_T: "np.ndarray | None" = None
    per_twist_T_std: "np.ndarray | None" = None
    per_twist_P_hydro: "np.ndarray | None" = None
    per_twist_P_hydro_std: "np.ndarray | None" = None
    per_twist_gF: "np.ndarray | None" = None
    per_twist_gF_std: "np.ndarray | None" = None
    #: Optional MPC finite-size result. ``None`` keeps pure-Ewald runs
    #: byte-compatible and avoids manufacturing a zero correction.
    mpc: "TwistAverageMPC | None" = None


@dataclass(frozen=True)
class TwistAverageEnergy:
    """BZ-mean energy estimator output.

    Attributes:
        E_TA: BZ-averaged energy (Hartree).
        E_TA_std: MC error on E_TA (quadrature combine of per-twist sigma_k).
        s_twist: Twist dispersion -- a *physical* finite-size diagnostic, NOT
            an MC error. Reported separately, never folded into ``E_TA_std``.
        W: Normalised per-twist weights ``W_k`` (Sum_k W_k = 1).
    """

    E_TA: float
    E_TA_std: float
    s_twist: float
    W: np.ndarray


@dataclass(frozen=True)
class TwistAverageMPC:
    """BZ-averaged passive MPC observable and its Level-1 inputs.

    ``energy`` carries the covariance-aware corrected-total errors returned
    by each sampler's ``get_E_mpc``.  Those errors already include the
    same-walker Ewald/correction covariance; only independent twists are
    combined in quadrature by the BZ layer.
    """

    correction: TwistAverageEnergy
    energy: TwistAverageEnergy
    per_twist_correction: np.ndarray
    per_twist_correction_std: np.ndarray
    per_twist_energy: np.ndarray
    per_twist_energy_std: np.ndarray
    cutoff: float


@dataclass(frozen=True)
class TwistAverageForce:
    """BZ-mean atomic-force estimator output.

    Attributes:
        F_TA: BZ-averaged force, shape ``(n_atoms, 3)`` (Hartree / bohr).
        F_TA_std: MC error on F_TA, same shape (quadrature combine).
    """

    F_TA: np.ndarray
    F_TA_std: np.ndarray


def bz_mean_energy(
    E_per_twist: npt.ArrayLike,
    sigma_per_twist: npt.ArrayLike,
    weights: npt.ArrayLike,
) -> TwistAverageEnergy:
    """Two-level BZ mean of a per-twist scalar observable (energy).

    Args:
        E_per_twist: Per-twist mean ``E_bar_k``, shape ``(N_twist,)``.
        sigma_per_twist: Per-twist MC error ``sigma_k``, shape ``(N_twist,)``.
        weights: Per-twist raw DFT weights ``w_k`` (will be normalised).

    Returns:
        :class:`TwistAverageEnergy` with ``E_TA``, ``E_TA_std``, ``s_twist``,
        and the normalised weights ``W``.
    """
    E_k = np.asarray(E_per_twist, dtype=np.float64).ravel()
    sigma_k = np.asarray(sigma_per_twist, dtype=np.float64).ravel()
    if E_k.shape != sigma_k.shape:
        raise ValueError(f"bz_mean_energy: E_per_twist shape {E_k.shape} != sigma_per_twist {sigma_k.shape}")
    W = normalize_twist_weights(weights)
    if W.shape != E_k.shape:
        raise ValueError(f"bz_mean_energy: weights shape {W.shape} != E_per_twist {E_k.shape}")

    E_TA = float(np.sum(W * E_k))
    # Quadrature combine -- the twists are independent MC chains, so the
    # per-twist sigmas combine as sqrt(Sum_k W_k^2 sigma_k^2). A pooled
    # jackknife would conflate s_twist with the MC error.
    E_TA_std = float(np.sqrt(np.sum((W**2) * (sigma_k**2))))
    # Twist dispersion: physical finite-size diagnostic, NOT a MC error.
    s_twist = float(np.sqrt(np.sum(W * (E_k - E_TA) ** 2)))
    return TwistAverageEnergy(E_TA=E_TA, E_TA_std=E_TA_std, s_twist=s_twist, W=W)


def _aggregate_twist_mpc(
    per_twist_correction: npt.ArrayLike,
    per_twist_correction_std: npt.ArrayLike,
    per_twist_energy: npt.ArrayLike,
    per_twist_energy_std: npt.ArrayLike,
    weights: npt.ArrayLike,
    *,
    cutoff: float,
) -> TwistAverageMPC:
    """Build the common MCMC/GFMC MPC Level-2 result.

    ``per_twist_energy_std`` must come directly from each sampler's joint
    ``get_E_mpc`` estimator.  The BZ layer deliberately never reconstructs
    it from separate Ewald and correction error bars.
    """
    correction = np.asarray(per_twist_correction, dtype=np.float64)
    correction_std = np.asarray(per_twist_correction_std, dtype=np.float64)
    energy = np.asarray(per_twist_energy, dtype=np.float64)
    energy_std = np.asarray(per_twist_energy_std, dtype=np.float64)
    return TwistAverageMPC(
        correction=bz_mean_energy(correction, correction_std, weights),
        energy=bz_mean_energy(energy, energy_std, weights),
        per_twist_correction=correction,
        per_twist_correction_std=correction_std,
        per_twist_energy=energy,
        per_twist_energy_std=energy_std,
        cutoff=float(cutoff),
    )


def _aggregate_twist_optimization_energy(
    local_estimate: EnergyEstimate,
    *,
    twist_index: int,
    weights: npt.ArrayLike,
    intra_comm: MPI.Comm,
    inter_comm: MPI.Comm | None,
) -> EnergyEstimate:
    """Aggregate one optimization-step energy and broadcast it to all ranks.

    ``local_estimate`` has already been reduced within ``intra_comm`` by the
    sampler's ``get_E`` implementation.  Exactly one root per twist therefore
    participates in the inter-twist gather.  The inter root constructs the
    weighted BZ estimate, broadcasts it to the remaining twist roots, and each
    twist root then broadcasts the same immutable estimate within its sampler
    communicator.

    The energy error is the independent-chain quadrature
    ``sqrt(sum_k W_k**2 sigma_k**2)``.  Per-twist variances are averaged with
    ``W_k`` while their errors use the same independent-chain quadrature.
    """
    from .jqmc_optimizer import EnergyEstimate

    if not isinstance(local_estimate, EnergyEstimate):
        raise TypeError("local_estimate must be an EnergyEstimate.")
    W = normalize_twist_weights(weights)
    index = int(twist_index)
    if index < 0 or index >= W.size:
        raise ValueError(f"twist_index={index} is outside [0, {W.size}).")
    if W.size == 1:
        return local_estimate

    aggregate = None
    if intra_comm.Get_rank() == 0:
        if inter_comm is None:
            raise ValueError("a twist-root rank requires a valid inter_comm.")
        gathered = inter_comm.gather(local_estimate, root=0)
        if inter_comm.Get_rank() == 0:
            if gathered is None or len(gathered) != W.size:
                received = 0 if gathered is None else len(gathered)
                raise ValueError(f"inter-twist energy gather returned {received} estimates for {W.size} twists.")
            if not all(isinstance(estimate, EnergyEstimate) for estimate in gathered):
                raise TypeError("inter-twist energy gather must contain EnergyEstimate values.")
            means = np.asarray([estimate.mean for estimate in gathered], dtype=np.float64)
            errors = np.asarray([estimate.standard_error for estimate in gathered], dtype=np.float64)
            variances = np.asarray([estimate.variance for estimate in gathered], dtype=np.float64)
            variance_errors = np.asarray(
                [estimate.variance_standard_error for estimate in gathered],
                dtype=np.float64,
            )
            bz_energy = bz_mean_energy(means, errors, W)
            aggregate = EnergyEstimate(
                mean=bz_energy.E_TA,
                standard_error=bz_energy.E_TA_std,
                variance=float(np.sum(W * variances)),
                variance_standard_error=float(np.sqrt(np.sum((W * variance_errors) ** 2))),
            )
        aggregate = inter_comm.bcast(aggregate, root=0)
    return intra_comm.bcast(aggregate, root=0)


@dataclass(frozen=True)
class TwistCommSplit:
    """Output of :func:`split_world_for_twists`.

    Attributes:
        world_comm: The all-rank communicator (unchanged identity).
        intra_comm: This rank's twist-local sampling comm (one group per
            twist). All ranks in the same group sample the same Bloch
            twist.
        inter_comm: The twist-roots comm (one rank per twist, intra-rank
            ``0``). ``None`` on ranks whose ``intra_comm`` rank is nonzero --
            they do not participate in the BZ-mean reduction.
        twist_index: The 0-based index of the twist this rank is assigned
            to (``world_rank % n_twist`` for a contiguous-round-robin
            split, see ``key`` arg below).
        n_twist: Total number of twists in the BZ grid.
    """

    world_comm: MPI.Comm
    intra_comm: MPI.Comm
    inter_comm: MPI.Comm | None
    twist_index: int
    n_twist: int


def split_world_for_twists(world_comm: MPI.Comm, n_twist: int) -> TwistCommSplit:
    """Split a world communicator into per-twist intra-comms + a twist-roots
    inter-comm.

    Args:
        world_comm: An MPI communicator (typically ``MPI.COMM_WORLD``) to
            split into ``n_twist`` groups.
        n_twist: Number of twists in the BZ grid. Must be >= 1 and divide
            ``world_comm.Get_size()`` (one twist per group of ranks). If
            ``n_twist == 1`` the split is the identity (intra_comm =
            world_comm; inter_comm = world_comm with one rank-0 member).

    Returns:
        :class:`TwistCommSplit` -- a frozen dataclass holding the
        ``world_comm``, the per-rank ``intra_comm`` (= the twist's
        sampling comm), the ``inter_comm`` (twist-roots only; ``None`` on
        non-roots), the rank's ``twist_index``, and ``n_twist``.

    Raises:
        ValueError: If ``n_twist < 1`` or ``world_comm.Get_size() %
        n_twist != 0``: every twist gets equal compute, and ``W_k`` is
        applied explicitly in the Level-2 sum instead of by unequal
        sampling.
    """
    if n_twist < 1:
        raise ValueError(f"split_world_for_twists: n_twist={n_twist} must be >= 1.")
    world_size = world_comm.Get_size()
    if world_size % n_twist != 0:
        raise ValueError(
            f"split_world_for_twists: world_size={world_size} must be a multiple of "
            f"n_twist={n_twist} (one twist per equal-size group)."
        )
    world_rank = world_comm.Get_rank()
    intra_size = world_size // n_twist
    twist_index = world_rank // intra_size
    intra_key = world_rank % intra_size

    # MPI is imported lazily so the docstring / single-process tests don't
    # need an MPI runtime active (the tests stub world_comm with a fake).
    from mpi4py import MPI  # noqa: WPS433 -- lazy import for test ergonomics

    intra_comm = world_comm.Split(color=twist_index, key=intra_key)
    # inter_comm groups the rank-0-of-each-intra (the twist roots). All
    # other ranks pass ``MPI.UNDEFINED`` so they get a ``MPI.COMM_NULL``
    # back -- which mpi4py exposes as ``None`` (after .Get_rank() raises
    # if attempted). We normalise that to ``None`` for the caller.
    is_root = intra_key == 0
    inter_color = 0 if is_root else MPI.UNDEFINED
    raw_inter = world_comm.Split(color=inter_color, key=twist_index)
    if not is_root:
        # Free the COMM_NULL handle to keep mpi4py from accumulating one
        # per call. ``COMM_NULL`` does not need (and does not respond to)
        # ``Free``, so we just drop the reference.
        inter_comm = None
    else:
        inter_comm = raw_inter

    return TwistCommSplit(
        world_comm=world_comm,
        intra_comm=intra_comm,
        inter_comm=inter_comm,
        twist_index=twist_index,
        n_twist=n_twist,
    )


def bz_mean_force(
    F_per_twist: npt.ArrayLike,
    sigma_per_twist: npt.ArrayLike,
    weights: npt.ArrayLike,
) -> TwistAverageForce:
    """Two-level BZ mean of a per-twist array observable (force / stress / gF).

    The combine is component-wise identical to :func:`bz_mean_energy`.
    Atomic forces and stresses are physically real, while the raw
    generalized force returned by ``MCMC.get_gF`` remains complex until the
    optimizer's parameter-manifold map is applied.  Preserve that input dtype
    here; the Monte Carlo error remains real in either case.

    Args:
        F_per_twist: Per-twist mean observable with the twist on axis 0:
            atomic force ``(N_twist, n_atoms, 3)``, stress tensor
            ``(N_twist, 3, 3)``, or generalized (variational-parameter)
            force ``(N_twist, n_param)``.
        sigma_per_twist: Per-twist MC error, same shape.
        weights: Per-twist raw DFT weights ``w_k`` (will be normalised).

    Returns:
        :class:`TwistAverageForce` with ``F_TA`` and ``F_TA_std``, both of
        the observable shape (twist axis reduced).
    """
    F_k = np.asarray(F_per_twist)
    sigma_k = np.asarray(sigma_per_twist, dtype=np.float64)
    if F_k.shape != sigma_k.shape:
        raise ValueError(f"bz_mean_force: F_per_twist shape {F_k.shape} != sigma_per_twist {sigma_k.shape}")
    if F_k.ndim < 2:
        raise ValueError(
            f"bz_mean_force: per-twist observable must be at least 2-D (N_twist, ...observable axes); got shape {F_k.shape}"
        )
    W = normalize_twist_weights(weights)
    if W.size != F_k.shape[0]:
        raise ValueError(f"bz_mean_force: weights size {W.size} != F_per_twist N_twist={F_k.shape[0]}")

    # Broadcast W over the observable axes so every component receives the
    # same combine. Works for any per-twist array observable: atomic force
    # (N_twist, n_atoms, 3), stress tensor (N_twist, 3, 3), generalized
    # (variational-parameter) force (N_twist, n_param).
    W_b = W.reshape((W.size,) + (1,) * (F_k.ndim - 1))
    F_TA = np.sum(W_b * F_k, axis=0)
    F_TA_std = np.sqrt(np.sum((W_b**2) * (sigma_k**2), axis=0))
    return TwistAverageForce(F_TA=F_TA, F_TA_std=F_TA_std)


def _twist_k_point_from_hamiltonian(
    hamiltonian_data: Hamiltonian_data,
) -> tuple[float, float, float] | None:
    """Return the Bloch k-point stored by a twist's k-dependent AO leaf."""
    from .orbital_tiling import TiledOrbitalSet_data

    orbitals = hamiltonian_data.wavefunction_data.geminal_data.orb_data_up_spin
    if isinstance(orbitals, TiledOrbitalSet_data):
        return orbitals.supertwist
    aos_data = orbitals.aos_data
    if isinstance(aos_data, AOs_composite_data):
        return aos_data.children[1].k_point
    return aos_data.k_point


def _twist_k_points_from_list(hamiltonian_list: Sequence[Hamiltonian_data]) -> list:
    """Per-twist k-point table without materialising a lazy Hamiltonian list.

    The CLI hands the drivers a ``_LazyTwistHamiltonians`` view that loads a
    twist's Hamiltonian only on first access; iterating it here
    (``[_twist_k_point_from_hamiltonian(h) for h in ...]``) would force EVERY
    rank to deserialise EVERY twist before the communicator split. The lazy view
    carries the cheap ``/twists`` header k-points on ``twist_k_points``
    (identical values by construction: both ``load_twist`` and
    ``load_twist_average_input`` build ``tuple(float(x) for x in k)`` /
    ``None`` at Gamma), so prefer that and keep the full iteration only as
    the fallback for plain in-memory lists.
    """
    cheap = getattr(hamiltonian_list, "twist_k_points", None)
    if cheap is not None:
        k_points = list(cheap)
        if len(k_points) != len(hamiltonian_list):
            raise ValueError(
                f"_twist_k_points_from_list: header k_points length {len(k_points)} != n_twist {len(hamiltonian_list)}."
            )
        return k_points
    return [_twist_k_point_from_hamiltonian(hamiltonian) for hamiltonian in hamiltonian_list]


def run_twist_average_from_hamiltonians(
    hamiltonian_list: Sequence[Hamiltonian_data],
    weights: npt.ArrayLike,
    *,
    mcmc_seed: int = 34467,
    num_mcmc_steps: int,
    num_walkers: int = 40,
    num_mcmc_warmup_steps: int = 50,
    num_mcmc_bin_blocks: int = 10,
    max_time: float = float("inf"),
    world_comm: MPI.Comm | None = None,
    compute_force: bool = False,
    compute_stress: bool = False,
    compute_param_grad: bool = False,
    mcmc_kwargs: dict | None = None,
    checkpoint_dir: str | None = None,
    resume: bool = False,
) -> TwistAverageResult:
    """Run an MCMC chain per (Hamiltonian, twist) and BZ-average the results.

    This is the Hamiltonian-list-driven orchestrator. The file-path wrapper
    that reads ``hamiltonian_data.h5`` with ``/twists/twist_*`` and calls
    this helper is :func:`run_twist_average`.

    Behaviour at ``N_twist == 1``: the function constructs a single
    :class:`jqmc.jqmc_mcmc.MCMC`, runs it on the supplied ``world_comm``
    (default ``COMM_WORLD``), and returns the same ``E_mean / sigma`` the
    direct MCMC path would produce. This is a byte-identity exit -- the
    orchestrator must not perturb the single-twist chain in any way (no
    extra comm splits, no extra MCMC steps, no extra reductions).

    Behaviour at ``N_twist > 1``: the ``world_comm`` is split into
    ``N_twist`` groups via :func:`split_world_for_twists` (requires
    ``world_size`` divisible by ``N_twist``); each twist's MCMC reduces
    over its own intra-comm. The per-twist results are gathered via the
    inter-comm and combined via :func:`bz_mean_energy` /
    :func:`bz_mean_force`.

    Args:
        hamiltonian_list: List of ``Hamiltonian_data`` objects, one per
            twist. Each ``Hamiltonian_data``'s AO leaf ``aos_data.k_point``
            should be set to that twist's fractional k (``None`` at Gamma);
            mismatches with the BZ grid are not checked here -- the
            caller is responsible for handing in a consistent list.
        weights: Per-twist raw weights ``w_k`` (will be normalised
            internally via :func:`normalize_twist_weights`).
        mcmc_seed: Base RNG seed. Multi-twist runs derive a deterministic
            twist namespace before ``MCMC`` folds in the intra-rank.
        num_mcmc_steps: Per-twist MCMC step count (after warmup).
        num_walkers: Walkers per (rank, twist). Default 40.
        num_mcmc_warmup_steps: Warmup discarded before observable evaluation.
        num_mcmc_bin_blocks: Number of jackknife blocks.
        max_time: Wall-clock budget per twist (seconds).
        world_comm: Optional MPI comm. Defaults to ``MPI.COMM_WORLD``.
        compute_force: If True, also compute the atomic force per twist
            (``get_aF``) and the BZ-averaged force. Requires
            ``MCMC(comput_position_deriv=True)`` (set automatically).
        compute_stress: If True, also compute the symmetric stress tensor
            per twist (``get_aT``, leaf shape ``(3, 3)``) and its BZ mean
            (``TwistAverageResult.stress``). Sets
            ``MCMC(comput_stress=True)`` only; it is independent of
            ``compute_force`` and does NOT enable ``comput_position_deriv``
            (a stress-only run computes no atomic force).
        compute_param_grad: If True, also compute the generalized
            (variational-parameter) force per twist (``get_gF``, leaf
            shape ``(n_param,)``) and its BZ mean
            (``TwistAverageResult.param_grad``). For shared parameters the
            BZ mean is exactly ``d E_TA / dc``. Sets
            ``comput_log_WF_param_deriv=True`` and
            ``comput_e_L_param_deriv=True`` on the per-twist ``MCMC``.
            Not checkpointed yet: mutually exclusive with ``checkpoint_dir``.
        mcmc_kwargs: Extra kwargs forwarded to ``MCMC.__init__`` (e.g.
            ``epsilon_AS``, ``use_swct``, ``mpc``). When ``mpc=True``,
            both per-twist MPC estimators are persisted and BZ-aggregated
            into ``TwistAverageResult.mpc``. Default ``None``.
        checkpoint_dir: Optional directory for the per-twist checkpoint
            (``twist_{idx}.h5`` + ``shared_state.h5``).
            On a fresh run the existing ``twist_NNN.h5`` files are first
            marked unconverged, and each twist's checkpoint is written after
            that twist's chain finishes. On a resume run (``resume=True``)
            twists with a certified-complete checkpoint are skipped and
            their ``(E_k, sigma_k)`` are loaded back from disk.
        resume: If ``True``, skip twists whose checkpoint certifies them
            as complete via :func:`jqmc._twist_checkpoint.is_twist_complete`.
            A completed twist stored with another ``num_mcmc_steps``,
            ``num_mcmc_warmup_steps`` or ``num_mcmc_bin_blocks`` raises
            ``ValueError`` (a shorter stored chain is sampled again).
            Has no effect when ``checkpoint_dir`` is ``None``.

    Returns:
        :class:`TwistAverageResult` -- ``energy`` (always present),
        optional ``force``, plus per-twist arrays + k_points for logging /
        downstream summary writes.
    """
    if mcmc_kwargs is None:
        mcmc_kwargs = {}
    mpc_enabled = bool(mcmc_kwargs.get("mpc", False))
    mpc_cutoff = float(mcmc_kwargs.get("mpc_cutoff", MPC_CUTOFF_HA)) if mpc_enabled else None
    if not hamiltonian_list:
        raise ValueError("run_twist_average_from_hamiltonians: hamiltonian_list is empty.")
    if checkpoint_dir is not None and compute_param_grad:
        raise ValueError(
            "run_twist_average_from_hamiltonians: compute_param_grad is not "
            "persisted by the per-twist checkpoint yet; run without checkpoint_dir, "
            "or extend _twist_checkpoint first."
        )
    n_twist = len(hamiltonian_list)
    W = normalize_twist_weights(weights)
    if W.size != n_twist:
        raise ValueError(f"run_twist_average_from_hamiltonians: weights size {W.size} != n_twist {n_twist}.")
    # Immutable metadata, derived before communicator decomposition. Read
    # via the cheap header path when the list is lazy -- iterating the lazy
    # list here would force every rank to load every twist's Hamiltonian.
    k_points = _twist_k_points_from_list(hamiltonian_list)

    # Local import to avoid a module-load circular dependency
    # (jqmc.jqmc_mcmc imports from jqmc.*, this module is imported by tests
    # and would-be circular through MCMC's many transitive imports).
    from mpi4py import MPI

    from .jqmc_mcmc import MCMC

    world = world_comm if world_comm is not None else MPI.COMM_WORLD

    # ------------------------------------------------------------------
    # Comm split. Single-twist path is the identity (intra == world);
    # multi-twist requires divisibility (equal compute per twist).
    # ------------------------------------------------------------------
    if n_twist == 1:
        intra_comm = world
        inter_comm = world
        twist_indices = [0]
    else:
        split = split_world_for_twists(world, n_twist=n_twist)
        intra_comm = split.intra_comm
        inter_comm = split.inter_comm
        twist_indices = [split.twist_index]
        # Each rank only drives ITS twist's chain (one MCMC instance per
        # rank, not N_twist instances per rank). The aggregation happens
        # across inter_comm at the end.

    # ------------------------------------------------------------------
    # Per-twist run -- single twist for the multi-twist case, or all N
    # for the N_twist==1 case (where the loop is trivially one iteration).
    # ------------------------------------------------------------------
    per_twist_E = np.zeros(n_twist, dtype=np.float64)
    per_twist_E_std = np.zeros(n_twist, dtype=np.float64)
    # Steps done + 1 for each twist that max_time stopped (0: not stopped), reported once reduced.
    per_twist_stopped = np.zeros(n_twist, dtype=np.int64)
    per_twist_dE_mpc = np.zeros(n_twist, dtype=np.float64) if mpc_enabled else None
    per_twist_dE_mpc_std = np.zeros(n_twist, dtype=np.float64) if mpc_enabled else None
    per_twist_E_mpc = np.zeros(n_twist, dtype=np.float64) if mpc_enabled else None
    per_twist_E_mpc_std = np.zeros(n_twist, dtype=np.float64) if mpc_enabled else None
    per_twist_F = None
    per_twist_F_std = None
    per_twist_T = None
    per_twist_T_std = None
    per_twist_P = None
    per_twist_P_std = None
    per_twist_gF = None
    per_twist_gF_std = None
    if compute_force:
        # The shape of per-twist F is (n_atoms, 3); all twists share the same
        # structure, so resolve it from a twist this rank owns anyway -- using
        # index 0 would make every rank of every other twist load twist 0 too.
        n_atoms_0 = int(hamiltonian_list[twist_indices[0]].structure_data.natom)
        per_twist_F = np.zeros((n_twist, n_atoms_0, 3), dtype=np.float64)
        per_twist_F_std = np.zeros((n_twist, n_atoms_0, 3), dtype=np.float64)
    if compute_stress:
        per_twist_T = np.zeros((n_twist, 3, 3), dtype=np.float64)
        per_twist_T_std = np.zeros((n_twist, 3, 3), dtype=np.float64)
        # Leaf axis of size 1 so the reviewed ``bz_mean_force`` combine is
        # reused verbatim for the scalar pressure -- no second combine to
        # keep in sync.
        per_twist_P = np.zeros((n_twist, 1), dtype=np.float64)
        per_twist_P_std = np.zeros((n_twist, 1), dtype=np.float64)
    if compute_param_grad:
        # Every rank must allocate the same MPI buffer dtype even though a
        # Gamma twist can emit a real gF while a non-Gamma twist emits a
        # complex one. The determinant working dtype is the wave-function-
        # wide real/complex regime contract; SD derives it from occupied MOs,
        # while AGP/AGPn include their lambda leaves.
        if getattr(hamiltonian_list, "twist_k_points", None) is not None:
            # Lazy CLI list: probing every twist's dtype would force every
            # rank to deserialise every Hamiltonian. Each rank inspects only
            # the twist(s) it owns and the complex regime is agreed globally
            # over ``world`` -- same outcome as the full scan (complex128 iff
            # ANY twist's determinant regime is complex, float64 otherwise).
            local_complex = any(
                np.issubdtype(
                    np.dtype(hamiltonian_list[k].wavefunction_data.geminal_data._determinant_dtype_jnp),
                    np.complexfloating,
                )
                for k in twist_indices
            )
            any_complex = bool(world.allreduce(int(local_complex), op=MPI.MAX))
            param_grad_dtype = np.dtype(np.complex128) if any_complex else np.dtype(np.float64)
        else:
            param_grad_dtype = np.result_type(
                *(np.dtype(ham.wavefunction_data.geminal_data._determinant_dtype_jnp) for ham in hamiltonian_list),
                np.float64,
            )
    else:
        param_grad_dtype = None
    # per_twist_gF is allocated lazily after the first get_gF call -- the
    # parameter count is a property of the (shared) wavefunction layout and
    # is only known once a chain has been built. All twists share the
    # layout, so the shapes are uniform across ranks.

    # Per-twist checkpoint resume path. We import lazily so the helper module
    # isn't pulled in by every test that exercises the driver without
    # checkpointing.
    if checkpoint_dir is not None:
        from ._twist_checkpoint import (
            twist_identity_digest,
            load_twist_checkpoint_summary,
            save_twist_checkpoint,
        )
    sampler_options = _twist_sampler_options(MCMC, "mcmc", mcmc_kwargs) if checkpoint_dir is not None and resume else {}
    # The stored E_k is one analysis of the stored chain (issue 862).
    analysis_config = {"num_mcmc_warmup_steps": int(num_mcmc_warmup_steps), "num_mcmc_bin_blocks": int(num_mcmc_bin_blocks)}
    if checkpoint_dir is not None and resume:
        _reject_twist_checkpoints_beyond(checkpoint_dir, n_twist, world)

    for k_idx in twist_indices:
        ham_k = hamiltonian_list[k_idx]
        if (
            checkpoint_dir is not None
            and resume
            and _check_twist_resume(
                checkpoint_dir,
                k_idx,
                num_mcmc_steps,
                intra_comm=intra_comm,
                world=world if n_twist > 1 else None,
                k_point=k_points[k_idx],
                ham_digest=twist_identity_digest(ham_k),
                ecp=ham_k.coulomb_potential_data,
                mpc_cutoff=mpc_cutoff,
                driver_type="MCMC",
                sampler_config={**sampler_options, "num_walkers": num_walkers},
                normalized_twist_weight=float(W[k_idx]),
                num_twists=n_twist,
                analysis_config=analysis_config,
            )
        ):
            # Skip-completed path. The intra root reads the cached
            # (E_k, sigma_k) from the checkpoint and broadcasts so every
            # rank holds the same view before the BZ combine.
            if intra_comm.Get_rank() == 0:
                summary = load_twist_checkpoint_summary(checkpoint_dir, k_idx)
            else:
                summary = None
            summary = intra_comm.bcast(summary, root=0)
            # A checkpoint written before the hydrostatic replicates existed
            # has no ``P_hydro_k``; requiring it here makes such a twist
            # RE-RUN rather than silently produce a stress table whose
            # pressure row cannot carry an error bar.
            summary_has_requested_observables = (not compute_force or {"F_k", "sigma_F_k"} <= summary.keys()) and (
                not compute_stress or {"T_k", "sigma_T_k", "P_hydro_k", "sigma_P_hydro_k"} <= summary.keys()
            )
            summary_has_requested_observables = summary_has_requested_observables and (
                not mpc_enabled
                or {
                    "dE_mpc_k",
                    "sigma_dE_mpc_k",
                    "E_mpc_k",
                    "sigma_E_mpc_k",
                    "mpc_cutoff",
                }
                <= summary.keys()
            )
            if summary_has_requested_observables:
                per_twist_E[k_idx] = float(summary["E_k"])
                per_twist_E_std[k_idx] = float(summary["sigma_k"])
            if summary_has_requested_observables and compute_force:
                per_twist_F[k_idx] = np.asarray(summary["F_k"], dtype=np.float64)
                per_twist_F_std[k_idx] = np.asarray(summary["sigma_F_k"], dtype=np.float64)
            if summary_has_requested_observables and compute_stress:
                per_twist_T[k_idx] = np.asarray(summary["T_k"], dtype=np.float64)
                per_twist_T_std[k_idx] = np.asarray(summary["sigma_T_k"], dtype=np.float64)
                per_twist_P[k_idx, 0] = float(summary["P_hydro_k"])
                per_twist_P_std[k_idx, 0] = float(summary["sigma_P_hydro_k"])
            if summary_has_requested_observables and mpc_enabled:
                per_twist_dE_mpc[k_idx] = float(summary["dE_mpc_k"])
                per_twist_dE_mpc_std[k_idx] = float(summary["sigma_dE_mpc_k"])
                per_twist_E_mpc[k_idx] = float(summary["E_mpc_k"])
                per_twist_E_mpc_std[k_idx] = float(summary["sigma_E_mpc_k"])
            if summary_has_requested_observables:
                continue

        # Atomic force and periodic stress are independent observables.
        # Explicit entries in ``mcmc_kwargs`` are OR-ed in, never disabled.
        _init_kwargs = dict(mcmc_kwargs)
        _init_kwargs["comput_position_deriv"] = bool(compute_force or _init_kwargs.get("comput_position_deriv", False))
        _init_kwargs["comput_stress"] = bool(compute_stress or _init_kwargs.get("comput_stress", False))
        if compute_param_grad:
            _init_kwargs["comput_log_WF_param_deriv"] = True
            _init_kwargs["comput_e_L_param_deriv"] = True
        mcmc = MCMC(
            hamiltonian_data=ham_k,
            mcmc_seed=_twist_sampler_seed(mcmc_seed, k_idx, n_twist),
            num_walkers=num_walkers,
            mpi_comm_in=intra_comm,
            world_comm_in=world,
            **_init_kwargs,
        )
        # Every rank samples exactly one twist, so this runs once per rank.
        if checkpoint_dir is not None and not resume:
            _invalidate_twist_checkpoints(checkpoint_dir, world)
        mcmc.run(num_mcmc_steps=num_mcmc_steps, max_time=max_time)
        mcmc_complete = mcmc.mcmc_counter >= num_mcmc_steps
        if not mcmc_complete:
            per_twist_stopped[k_idx] = mcmc.mcmc_counter + 1
        E_mean, E_std, _, _ = mcmc.get_E(
            num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            num_mcmc_bin_blocks=num_mcmc_bin_blocks,
        )
        per_twist_E[k_idx] = float(E_mean)
        per_twist_E_std[k_idx] = float(E_std)
        dE_mpc_mean_k = dE_mpc_std_k = None
        E_mpc_mean_k = E_mpc_std_k = None
        if mpc_enabled:
            dE_mpc_mean, dE_mpc_std = mcmc.get_dE_mpc(
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            )
            E_mpc_mean, E_mpc_std = mcmc.get_E_mpc(
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            )
            per_twist_dE_mpc[k_idx] = float(dE_mpc_mean)
            per_twist_dE_mpc_std[k_idx] = float(dE_mpc_std)
            per_twist_E_mpc[k_idx] = float(E_mpc_mean)
            per_twist_E_mpc_std[k_idx] = float(E_mpc_std)
            dE_mpc_mean_k = float(dE_mpc_mean)
            dE_mpc_std_k = float(dE_mpc_std)
            E_mpc_mean_k = float(E_mpc_mean)
            E_mpc_std_k = float(E_mpc_std)
        F_mean_k = F_std_k = None
        if compute_force:
            F_mean, F_std = mcmc.get_aF(
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            )
            per_twist_F[k_idx] = np.asarray(F_mean, dtype=np.float64)
            per_twist_F_std[k_idx] = np.asarray(F_std, dtype=np.float64)
            F_mean_k = per_twist_F[k_idx]
            F_std_k = per_twist_F_std[k_idx]
        T_mean_k = T_std_k = None
        P_hydro_mean_k = P_hydro_std_k = None
        if compute_stress:
            T_mean, T_std = mcmc.get_aT(
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            )
            per_twist_T[k_idx] = np.asarray(T_mean, dtype=np.float64)
            per_twist_T_std[k_idx] = np.asarray(T_std, dtype=np.float64)
            T_mean_k = per_twist_T[k_idx]
            T_std_k = per_twist_T_std[k_idx]
            # Covariance-aware, taken from the SAME replicates as T above.
            P_hydro_mean_k, P_hydro_std_k = mcmc.get_P_hydrostatic(
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            )
            per_twist_P[k_idx, 0] = float(P_hydro_mean_k)
            per_twist_P_std[k_idx, 0] = float(P_hydro_std_k)
        if compute_param_grad:
            # Same default variational-parameter mask as ``run_optimize``
            # (Jastrow parameter blocks; geminal / basis blocks off). The
            # per-twist WFs share one parameter layout (the combined twist input
            # shares the Jastrow across twists), so the block list -- and
            # hence the gF length -- is uniform over twists.
            _gf_blocks = ham_k.wavefunction_data.get_variational_blocks(
                state_token=mcmc.wavefunction_state_token,
            )
            gF_mean, gF_std = mcmc.get_gF(
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
                blocks=_gf_blocks,
            )
            gF_mean = np.ravel(np.asarray(gF_mean)).astype(param_grad_dtype, copy=False)
            gF_std = np.ravel(np.asarray(gF_std, dtype=np.float64))
            if per_twist_gF is None:
                per_twist_gF = np.zeros((n_twist, gF_mean.size), dtype=param_grad_dtype)
                per_twist_gF_std = np.zeros((n_twist, gF_mean.size), dtype=np.float64)
            per_twist_gF[k_idx] = gF_mean
            per_twist_gF_std[k_idx] = gF_std

        if checkpoint_dir is not None:
            save_twist_checkpoint(
                checkpoint_dir,
                twist_idx=k_idx,
                mcmc_or_gfmc=mcmc,
                intra_comm=intra_comm,
                inter_comm=inter_comm,
                driver_type="MCMC",
                target_steps=int(num_mcmc_steps),
                # A twist stopped on max_time is not complete: a resume reruns it.
                converged=mcmc_complete,
                k_point=k_points[k_idx],
                ham_digest=twist_identity_digest(ham_k),
                E_k=float(E_mean),
                sigma_k=float(E_std),
                dE_mpc_k=dE_mpc_mean_k,
                sigma_dE_mpc_k=dE_mpc_std_k,
                E_mpc_k=E_mpc_mean_k,
                sigma_E_mpc_k=E_mpc_std_k,
                mpc_cutoff=mpc_cutoff,
                F_k=F_mean_k,
                sigma_F_k=F_std_k,
                T_k=T_mean_k,
                sigma_T_k=T_std_k,
                P_hydro_k=P_hydro_mean_k,
                sigma_P_hydro_k=P_hydro_std_k,
                normalized_twist_weight=float(W[k_idx]),
                num_twists=n_twist,
                **analysis_config,
            )

    # ------------------------------------------------------------------
    # Level-2 BZ-mean combine. At N_twist==1 the combine is trivial and
    # the orchestrator is byte-identical to the direct MCMC call. At
    # N_twist>1 the per-twist arrays must be inter-reduced
    # so every rank holds the same view before the combine.
    # ------------------------------------------------------------------
    if n_twist > 1:
        # Only twist-roots (intra rank 0) hold the per-twist E/E_std; the
        # other ranks have zeros. inter_comm.Allreduce(SUM) on twist-roots
        # gives every twist-root the full vector; we then broadcast back
        # via intra_comm so non-root ranks also know the BZ summary.
        if inter_comm is not None:
            # twist-root: gather everyone else's (E, std, F, F_std) via SUM
            per_twist_E = inter_comm.allreduce(per_twist_E, op=MPI.SUM)
            per_twist_E_std = inter_comm.allreduce(per_twist_E_std, op=MPI.SUM)
            per_twist_stopped = inter_comm.allreduce(per_twist_stopped, op=MPI.SUM)
            if mpc_enabled:
                per_twist_dE_mpc = inter_comm.allreduce(per_twist_dE_mpc, op=MPI.SUM)
                per_twist_dE_mpc_std = inter_comm.allreduce(per_twist_dE_mpc_std, op=MPI.SUM)
                per_twist_E_mpc = inter_comm.allreduce(per_twist_E_mpc, op=MPI.SUM)
                per_twist_E_mpc_std = inter_comm.allreduce(per_twist_E_mpc_std, op=MPI.SUM)
            if compute_force:
                per_twist_F = inter_comm.allreduce(per_twist_F, op=MPI.SUM)
                per_twist_F_std = inter_comm.allreduce(per_twist_F_std, op=MPI.SUM)
            if compute_stress:
                per_twist_T = inter_comm.allreduce(per_twist_T, op=MPI.SUM)
                per_twist_T_std = inter_comm.allreduce(per_twist_T_std, op=MPI.SUM)
                per_twist_P = inter_comm.allreduce(per_twist_P, op=MPI.SUM)
                per_twist_P_std = inter_comm.allreduce(per_twist_P_std, op=MPI.SUM)
            if compute_param_grad:
                per_twist_gF = inter_comm.allreduce(per_twist_gF, op=MPI.SUM)
                per_twist_gF_std = inter_comm.allreduce(per_twist_gF_std, op=MPI.SUM)
        # Broadcast root's view to non-root ranks of this twist.
        per_twist_E = intra_comm.bcast(per_twist_E, root=0)
        per_twist_E_std = intra_comm.bcast(per_twist_E_std, root=0)
        if mpc_enabled:
            per_twist_dE_mpc = intra_comm.bcast(per_twist_dE_mpc, root=0)
            per_twist_dE_mpc_std = intra_comm.bcast(per_twist_dE_mpc_std, root=0)
            per_twist_E_mpc = intra_comm.bcast(per_twist_E_mpc, root=0)
            per_twist_E_mpc_std = intra_comm.bcast(per_twist_E_mpc_std, root=0)
        if compute_force:
            per_twist_F = intra_comm.bcast(per_twist_F, root=0)
            per_twist_F_std = intra_comm.bcast(per_twist_F_std, root=0)
        if compute_stress:
            per_twist_T = intra_comm.bcast(per_twist_T, root=0)
            per_twist_T_std = intra_comm.bcast(per_twist_T_std, root=0)
            per_twist_P = intra_comm.bcast(per_twist_P, root=0)
            per_twist_P_std = intra_comm.bcast(per_twist_P_std, root=0)
        if compute_param_grad:
            per_twist_gF = intra_comm.bcast(per_twist_gF, root=0)
            per_twist_gF_std = intra_comm.bcast(per_twist_gF_std, root=0)

    if world.Get_rank() == 0:
        for k_idx in np.flatnonzero(per_twist_stopped):
            logger.warning(
                f"Twist {k_idx} stopped on max_time after {per_twist_stopped[k_idx] - 1} of {num_mcmc_steps} measurements: "
                "it is not complete, the BZ average below uses it as is, and a restart reruns it."
            )
    energy = bz_mean_energy(per_twist_E, per_twist_E_std, weights)
    force = None
    if compute_force:
        force = bz_mean_force(per_twist_F, per_twist_F_std, weights)
    stress = None
    P_hydro = None
    if compute_stress:
        stress = bz_mean_force(per_twist_T, per_twist_T_std, weights)
        # Independent twists -> the same weighted quadrature combine as every
        # other BZ observable; the WITHIN-twist correlation between the normal
        # stresses was already absorbed by the per-twist trace jackknife.
        P_hydro = bz_mean_force(per_twist_P, per_twist_P_std, weights)
    param_grad = None
    if compute_param_grad:
        param_grad = bz_mean_force(per_twist_gF, per_twist_gF_std, weights)
    mpc = None
    if mpc_enabled:
        mpc = _aggregate_twist_mpc(
            per_twist_dE_mpc,
            per_twist_dE_mpc_std,
            per_twist_E_mpc,
            per_twist_E_mpc_std,
            weights,
            cutoff=mpc_cutoff,
        )

    return TwistAverageResult(
        energy=energy,
        force=force,
        per_twist_E=per_twist_E,
        per_twist_E_std=per_twist_E_std,
        per_twist_F=per_twist_F,
        per_twist_F_std=per_twist_F_std,
        k_points=k_points,
        stress=stress,
        P_hydro=P_hydro,
        param_grad=param_grad,
        per_twist_T=per_twist_T,
        per_twist_T_std=per_twist_T_std,
        per_twist_P_hydro=per_twist_P,
        per_twist_P_hydro_std=per_twist_P_std,
        per_twist_gF=per_twist_gF,
        per_twist_gF_std=per_twist_gF_std,
        mpc=mpc,
    )


def run_twist_average_gfmc(
    hamiltonian_list: Sequence[Hamiltonian_data],
    weights: npt.ArrayLike,
    *,
    gfmc_kind: str = "tau",
    mcmc_seed: int = 34467,
    num_mcmc_steps: int,
    num_walkers: int = 40,
    num_mcmc_warmup_steps: int = 50,
    num_mcmc_bin_blocks: int = 10,
    max_time: float = float("inf"),
    world_comm: MPI.Comm | None = None,
    compute_force: bool = False,
    compute_stress: bool = False,
    gfmc_kwargs: dict | None = None,
    checkpoint_dir: str | None = None,
    resume: bool = False,
) -> TwistAverageResult:
    """GFMC mirror of :func:`run_twist_average_from_hamiltonians`.

    Drives one GFMC chain per (Hamiltonian, twist) and combines the
    per-twist ``(E_bar_k, F_bar_k)`` outputs into the BZ-averaged
    estimators via the same two-level scheme as the VMC driver -- the
    aggregator math (:func:`bz_mean_energy` / :func:`bz_mean_force`),
    the comm split (:func:`split_world_for_twists`), the per-``k``
    binning via intra-comm reductions, and the quadrature error combine
    are all reused verbatim. The GFMC-specific deltas are:

    * **Branching / reconfiguration stays within-twist.** Each chain
      reconfigures walkers over its own ``intra_comm``; there is no
      cross-twist branching (walkers of different twists sample
      different :math:`|\\Psi_k|^2`). Because each ``GFMC_*`` instance
      reduces through ``self._mpi_comm`` (= ``intra_comm`` once the
      driver injects it), this is enforced structurally.
    * **GFMC walker weight x twist weight stay separate.** The GFMC
      importance / branching weight is a per-walker, within-twist
      quantity; ``W_k`` enters only at the Level-2 BZ combine. The
      driver never folds ``W_k`` into the GFMC walker weights (which
      would distort the branching).

    Args:
        hamiltonian_list: One :class:`jqmc.hamiltonians.Hamiltonian_data`
            per twist.
        weights: Per-twist raw weights (will be normalised internally).
        gfmc_kind: ``"tau"`` -> :class:`jqmc.jqmc_gfmc.GFMC_t` (treatment
            GFMC; the standard projection at fixed ``tau``); ``"n"`` ->
            :class:`jqmc.jqmc_gfmc.GFMC_n` (number GFMC at fixed
            collect-step count). Other values raise.
        mcmc_seed: Base RNG seed. Multi-twist runs derive a deterministic
            twist namespace before the ``GFMC_*`` instance folds in the
            intra-rank.
        num_mcmc_steps: Per-twist GFMC step count.
        num_walkers: Walkers per (rank, twist).
        num_mcmc_warmup_steps: Warmup discarded before observable
            evaluation.
        num_mcmc_bin_blocks: Number of jackknife blocks for ``get_E``.
        max_time: Wall-clock budget per twist (seconds).
        world_comm: Optional MPI comm. Defaults to ``MPI.COMM_WORLD``.
        compute_force: If True, also compute the atomic force per twist
            and the BZ-averaged force.
        compute_stress: If True, also compute the symmetric stress tensor
            per twist (``get_aT``) and its BZ mean
            (``TwistAverageResult.stress``). Sets the GFMC
            ``comput_stress=True`` only; it is independent of
            ``compute_force`` and does NOT enable ``comput_position_deriv``.
        gfmc_kwargs: Extra kwargs forwarded to the GFMC constructor
            (e.g. ``tau`` / ``alat`` / ``non_local_move`` /
            ``epsilon_PW`` / ``mpc``). For ``gfmc_kind="n"`` / ``"n_debug"``
            it must carry ``num_mcmc_per_measurement`` (required, no
            default); without it a ``ValueError`` is raised before any
            sampling. When ``mpc=True``, the result carries
            the same MPC aggregate as the MCMC route.
        checkpoint_dir: Optional directory for the per-twist checkpoint
            (``twist_{idx}.h5``). On a fresh run the existing
            ``twist_NNN.h5`` files are first marked unconverged, and each
            twist's checkpoint is written after that twist's chain
            finishes; a twist stopped on ``max_time`` is saved as not
            converged. On a resume run (``resume=True``) twists with a
            certified-complete checkpoint are skipped and their per-twist
            estimators are loaded back from disk.
        resume: If ``True``, skip twists whose checkpoint certifies them
            as complete via :func:`jqmc._twist_checkpoint.is_twist_complete`.
            A complete twist whose stored summary lacks an observable
            requested by this run (force, stress or MPC) is sampled again.
            Has no effect when ``checkpoint_dir`` is ``None``.

    Returns:
        :class:`TwistAverageResult` -- same shape as the VMC twist
        driver, so downstream summary / aggregator code can stay
        single-codepath.
    """
    if gfmc_kwargs is None:
        gfmc_kwargs = {}
    mpc_enabled = bool(gfmc_kwargs.get("mpc", False))
    mpc_cutoff = float(gfmc_kwargs.get("mpc_cutoff", MPC_CUTOFF_HA)) if mpc_enabled else None
    if not hamiltonian_list:
        raise ValueError("run_twist_average_gfmc: hamiltonian_list is empty.")
    n_twist = len(hamiltonian_list)
    W = normalize_twist_weights(weights)
    if W.size != n_twist:
        raise ValueError(f"run_twist_average_gfmc: weights size {W.size} != n_twist {n_twist}.")
    # Match the MCMC driver: communicator decomposition applies to sampled
    # estimators, not to immutable metadata. Cheap header path for lazy lists.
    k_points = _twist_k_points_from_list(hamiltonian_list)

    from mpi4py import MPI

    # Lazy: ``_driver_helpers`` imports mpi4py.MPI at module level, which would
    # initialize MPI on a plain ``import jqmc._twist_average`` (#778).
    from ._driver_helpers import _require_lrdmc_bra_num_mcmc_per_measurement
    from .jqmc_gfmc import GFMC_n, GFMC_t, _GFMC_n_debug, _GFMC_t_debug

    # GFMC class dispatch table. The two ``_debug`` variants share
    # comm-injection and observable APIs with the production classes, so
    # the twist orchestrator routes to them via the same factory; the
    # surrounding ``run`` / ``get_E`` loop stays single-codepath. The
    # ``driver_type`` label fed into the checkpoint metadata is the
    # production-class name (the on-disk format is shared; the variant
    # used to produce the chain is recorded in ``_meta`` separately).
    _GFMC_CLS = {
        "tau": (GFMC_t, "GFMC_t"),
        "n": (GFMC_n, "GFMC_n"),
        "tau_debug": (_GFMC_t_debug, "GFMC_t"),
        "n_debug": (_GFMC_n_debug, "GFMC_n"),
    }
    if gfmc_kind not in _GFMC_CLS:
        raise ValueError(f"run_twist_average_gfmc: gfmc_kind={gfmc_kind!r} not in {sorted(_GFMC_CLS)}.")
    gfmc_cls, gfmc_driver_type = _GFMC_CLS[gfmc_kind]
    if gfmc_driver_type == "GFMC_n":
        # Required, no default: refuse before any twist is split or sampled.
        _require_lrdmc_bra_num_mcmc_per_measurement(gfmc_kwargs.get("num_mcmc_per_measurement"))

    world = world_comm if world_comm is not None else MPI.COMM_WORLD

    # Same comm-split contract as the VMC driver. Single-twist short-
    # circuits to the identity (intra == world) so the N_twist=1 path is
    # byte-identical to a plain ``GFMC_*`` run on the supplied
    # ``world_comm`` -- the GFMC analogue of the VMC byte-identity
    # guarantee.
    if n_twist == 1:
        intra_comm = world
        inter_comm = world
        twist_indices = [0]
    else:
        split = split_world_for_twists(world, n_twist=n_twist)
        intra_comm = split.intra_comm
        inter_comm = split.inter_comm
        twist_indices = [split.twist_index]

    per_twist_E = np.zeros(n_twist, dtype=np.float64)
    per_twist_E_std = np.zeros(n_twist, dtype=np.float64)
    # Steps done + 1 for each twist that max_time stopped (0: not stopped), reported once reduced.
    per_twist_stopped = np.zeros(n_twist, dtype=np.int64)
    per_twist_dE_mpc = np.zeros(n_twist, dtype=np.float64) if mpc_enabled else None
    per_twist_dE_mpc_std = np.zeros(n_twist, dtype=np.float64) if mpc_enabled else None
    per_twist_E_mpc = np.zeros(n_twist, dtype=np.float64) if mpc_enabled else None
    per_twist_E_mpc_std = np.zeros(n_twist, dtype=np.float64) if mpc_enabled else None
    per_twist_F = None
    per_twist_F_std = None
    per_twist_T = None
    per_twist_T_std = None
    per_twist_P = None
    per_twist_P_std = None
    if compute_force:
        # natom is twist-independent; index an owned twist (see MCMC driver).
        n_atoms_0 = int(hamiltonian_list[twist_indices[0]].structure_data.natom)
        per_twist_F = np.zeros((n_twist, n_atoms_0, 3), dtype=np.float64)
        per_twist_F_std = np.zeros((n_twist, n_atoms_0, 3), dtype=np.float64)
    if compute_stress:
        per_twist_T = np.zeros((n_twist, 3, 3), dtype=np.float64)
        per_twist_T_std = np.zeros((n_twist, 3, 3), dtype=np.float64)
        # Leaf axis of size 1 so the reviewed ``bz_mean_force`` combine is
        # reused verbatim for the scalar pressure -- no second combine to
        # keep in sync.
        per_twist_P = np.zeros((n_twist, 1), dtype=np.float64)
        per_twist_P_std = np.zeros((n_twist, 1), dtype=np.float64)

    if checkpoint_dir is not None:
        from ._twist_checkpoint import (
            load_twist_checkpoint_summary,
            save_twist_checkpoint,
            twist_identity_digest,
        )
    section = "lrdmc-bra" if gfmc_driver_type == "GFMC_n" else "lrdmc-tau"
    sampler_options = (
        _twist_sampler_options(gfmc_cls, section, gfmc_kwargs)
        if checkpoint_dir is not None and (resume or gfmc_driver_type == "GFMC_n")
        else {}
    )
    # The stored E_k is one analysis of the stored chain (issue 862).
    analysis_config = {"num_mcmc_warmup_steps": int(num_mcmc_warmup_steps), "num_mcmc_bin_blocks": int(num_mcmc_bin_blocks)}
    if checkpoint_dir is not None and resume:
        _reject_twist_checkpoints_beyond(checkpoint_dir, n_twist, world)

    for k_idx in twist_indices:
        ham_k = hamiltonian_list[k_idx]
        if (
            checkpoint_dir is not None
            and resume
            and _check_twist_resume(
                checkpoint_dir,
                k_idx,
                num_mcmc_steps,
                intra_comm=intra_comm,
                world=world if n_twist > 1 else None,
                k_point=k_points[k_idx],
                ham_digest=twist_identity_digest(ham_k),
                ecp=ham_k.coulomb_potential_data,
                mpc_cutoff=mpc_cutoff,
                driver_type=gfmc_driver_type,
                sampler_config={**sampler_options, "num_walkers": num_walkers},
                normalized_twist_weight=float(W[k_idx]),
                num_twists=n_twist,
                analysis_config=analysis_config,
            )
        ):
            if intra_comm.Get_rank() == 0:
                summary = load_twist_checkpoint_summary(checkpoint_dir, k_idx)
            else:
                summary = None
            summary = intra_comm.bcast(summary, root=0)
            # A checkpoint written before the hydrostatic replicates existed
            # has no ``P_hydro_k``; requiring it here makes such a twist
            # RE-RUN rather than silently produce a stress table whose
            # pressure row cannot carry an error bar.
            summary_has_requested_observables = (not compute_force or {"F_k", "sigma_F_k"} <= summary.keys()) and (
                not compute_stress or {"T_k", "sigma_T_k", "P_hydro_k", "sigma_P_hydro_k"} <= summary.keys()
            )
            summary_has_requested_observables = summary_has_requested_observables and (
                not mpc_enabled
                or {
                    "dE_mpc_k",
                    "sigma_dE_mpc_k",
                    "E_mpc_k",
                    "sigma_E_mpc_k",
                    "mpc_cutoff",
                }
                <= summary.keys()
            )
            if summary_has_requested_observables:
                per_twist_E[k_idx] = float(summary["E_k"])
                per_twist_E_std[k_idx] = float(summary["sigma_k"])
            if summary_has_requested_observables and compute_force:
                per_twist_F[k_idx] = np.asarray(summary["F_k"], dtype=np.float64)
                per_twist_F_std[k_idx] = np.asarray(summary["sigma_F_k"], dtype=np.float64)
            if summary_has_requested_observables and compute_stress:
                per_twist_T[k_idx] = np.asarray(summary["T_k"], dtype=np.float64)
                per_twist_T_std[k_idx] = np.asarray(summary["sigma_T_k"], dtype=np.float64)
                per_twist_P[k_idx, 0] = float(summary["P_hydro_k"])
                per_twist_P_std[k_idx, 0] = float(summary["sigma_P_hydro_k"])
            if summary_has_requested_observables and mpc_enabled:
                per_twist_dE_mpc[k_idx] = float(summary["dE_mpc_k"])
                per_twist_dE_mpc_std[k_idx] = float(summary["sigma_dE_mpc_k"])
                per_twist_E_mpc[k_idx] = float(summary["E_mpc_k"])
                per_twist_E_mpc_std[k_idx] = float(summary["sigma_E_mpc_k"])
            if summary_has_requested_observables:
                continue

        _gfmc_init_kwargs = dict(gfmc_kwargs)
        _gfmc_init_kwargs["comput_position_deriv"] = bool(
            compute_force or _gfmc_init_kwargs.get("comput_position_deriv", False)
        )
        _gfmc_init_kwargs["comput_stress"] = bool(compute_stress)
        gfmc = gfmc_cls(
            hamiltonian_data=ham_k,
            mcmc_seed=_twist_sampler_seed(mcmc_seed, k_idx, n_twist),
            num_walkers=num_walkers,
            mpi_comm_in=intra_comm,
            world_comm_in=world,
            **_gfmc_init_kwargs,
        )
        # The ``_debug`` variants of GFMC_t / GFMC_n omit the wall-clock
        # ``max_time`` kwarg (they are reference implementations meant for
        # bit-identity comparison against the production classes). Pass
        # ``max_time`` only when the run signature accepts it.
        _run_kwargs = {"num_mcmc_steps": num_mcmc_steps}
        if "max_time" in inspect.signature(gfmc.run).parameters:
            _run_kwargs["max_time"] = max_time
        # Every rank samples exactly one twist, so this runs once per rank.
        if checkpoint_dir is not None and not resume:
            _invalidate_twist_checkpoints(checkpoint_dir, world)
        gfmc.run(**_run_kwargs)
        # ``GFMC.mcmc_counter`` leaves out the collect steps; this is the raw step count.
        gfmc_complete = gfmc._mcmc_counter >= num_mcmc_steps
        if not gfmc_complete:
            per_twist_stopped[k_idx] = gfmc._mcmc_counter + 1
        E_mean, E_std, _, _ = gfmc.get_E(
            num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            num_mcmc_bin_blocks=num_mcmc_bin_blocks,
        )
        per_twist_E[k_idx] = float(E_mean)
        per_twist_E_std[k_idx] = float(E_std)
        dE_mpc_mean_k = dE_mpc_std_k = None
        E_mpc_mean_k = E_mpc_std_k = None
        if mpc_enabled:
            dE_mpc_mean, dE_mpc_std = gfmc.get_dE_mpc(
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            )
            E_mpc_mean, E_mpc_std = gfmc.get_E_mpc(
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            )
            per_twist_dE_mpc[k_idx] = float(dE_mpc_mean)
            per_twist_dE_mpc_std[k_idx] = float(dE_mpc_std)
            per_twist_E_mpc[k_idx] = float(E_mpc_mean)
            per_twist_E_mpc_std[k_idx] = float(E_mpc_std)
            dE_mpc_mean_k = float(dE_mpc_mean)
            dE_mpc_std_k = float(dE_mpc_std)
            E_mpc_mean_k = float(E_mpc_mean)
            E_mpc_std_k = float(E_mpc_std)
        F_mean_k = F_std_k = None
        if compute_force:
            F_mean, F_std = gfmc.get_aF(
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            )
            per_twist_F[k_idx] = np.asarray(F_mean, dtype=np.float64)
            per_twist_F_std[k_idx] = np.asarray(F_std, dtype=np.float64)
            F_mean_k = per_twist_F[k_idx]
            F_std_k = per_twist_F_std[k_idx]
        T_mean_k = T_std_k = None
        P_hydro_mean_k = P_hydro_std_k = None
        if compute_stress:
            T_mean, T_std = gfmc.get_aT(
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            )
            per_twist_T[k_idx] = np.asarray(T_mean, dtype=np.float64)
            per_twist_T_std[k_idx] = np.asarray(T_std, dtype=np.float64)
            T_mean_k = per_twist_T[k_idx]
            T_std_k = per_twist_T_std[k_idx]
            # Covariance-aware, taken from the SAME replicates as T above.
            P_hydro_mean_k, P_hydro_std_k = gfmc.get_P_hydrostatic(
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            )
            per_twist_P[k_idx, 0] = float(P_hydro_mean_k)
            per_twist_P_std[k_idx, 0] = float(P_hydro_std_k)

        if checkpoint_dir is not None:
            save_twist_checkpoint(
                checkpoint_dir,
                twist_idx=k_idx,
                mcmc_or_gfmc=gfmc,
                intra_comm=intra_comm,
                inter_comm=inter_comm,
                driver_type=gfmc_driver_type,
                target_steps=int(num_mcmc_steps),
                # A twist stopped on max_time is not complete: a resume reruns it.
                converged=gfmc_complete,
                initial_E_scf=sampler_options.get("E_scf"),
                k_point=k_points[k_idx],
                ham_digest=twist_identity_digest(ham_k),
                E_k=float(E_mean),
                sigma_k=float(E_std),
                dE_mpc_k=dE_mpc_mean_k,
                sigma_dE_mpc_k=dE_mpc_std_k,
                E_mpc_k=E_mpc_mean_k,
                sigma_E_mpc_k=E_mpc_std_k,
                mpc_cutoff=mpc_cutoff,
                F_k=F_mean_k,
                sigma_F_k=F_std_k,
                T_k=T_mean_k,
                sigma_T_k=T_std_k,
                P_hydro_k=P_hydro_mean_k,
                sigma_P_hydro_k=P_hydro_std_k,
                normalized_twist_weight=float(W[k_idx]),
                num_twists=n_twist,
                **analysis_config,
            )

    # Level-2 BZ-mean combine (same as the VMC driver). Inter-comm
    # gather + intra-comm broadcast for ``N_twist > 1``; trivial at
    # ``N_twist == 1``.
    if n_twist > 1:
        if inter_comm is not None:
            per_twist_E = inter_comm.allreduce(per_twist_E, op=MPI.SUM)
            per_twist_E_std = inter_comm.allreduce(per_twist_E_std, op=MPI.SUM)
            per_twist_stopped = inter_comm.allreduce(per_twist_stopped, op=MPI.SUM)
            if mpc_enabled:
                per_twist_dE_mpc = inter_comm.allreduce(per_twist_dE_mpc, op=MPI.SUM)
                per_twist_dE_mpc_std = inter_comm.allreduce(per_twist_dE_mpc_std, op=MPI.SUM)
                per_twist_E_mpc = inter_comm.allreduce(per_twist_E_mpc, op=MPI.SUM)
                per_twist_E_mpc_std = inter_comm.allreduce(per_twist_E_mpc_std, op=MPI.SUM)
            if compute_force:
                per_twist_F = inter_comm.allreduce(per_twist_F, op=MPI.SUM)
                per_twist_F_std = inter_comm.allreduce(per_twist_F_std, op=MPI.SUM)
            if compute_stress:
                per_twist_T = inter_comm.allreduce(per_twist_T, op=MPI.SUM)
                per_twist_T_std = inter_comm.allreduce(per_twist_T_std, op=MPI.SUM)
                per_twist_P = inter_comm.allreduce(per_twist_P, op=MPI.SUM)
                per_twist_P_std = inter_comm.allreduce(per_twist_P_std, op=MPI.SUM)
        per_twist_E = intra_comm.bcast(per_twist_E, root=0)
        per_twist_E_std = intra_comm.bcast(per_twist_E_std, root=0)
        if mpc_enabled:
            per_twist_dE_mpc = intra_comm.bcast(per_twist_dE_mpc, root=0)
            per_twist_dE_mpc_std = intra_comm.bcast(per_twist_dE_mpc_std, root=0)
            per_twist_E_mpc = intra_comm.bcast(per_twist_E_mpc, root=0)
            per_twist_E_mpc_std = intra_comm.bcast(per_twist_E_mpc_std, root=0)
        if compute_force:
            per_twist_F = intra_comm.bcast(per_twist_F, root=0)
            per_twist_F_std = intra_comm.bcast(per_twist_F_std, root=0)
        if compute_stress:
            per_twist_T = intra_comm.bcast(per_twist_T, root=0)
            per_twist_T_std = intra_comm.bcast(per_twist_T_std, root=0)
            per_twist_P = intra_comm.bcast(per_twist_P, root=0)
            per_twist_P_std = intra_comm.bcast(per_twist_P_std, root=0)

    if world.Get_rank() == 0:
        for k_idx in np.flatnonzero(per_twist_stopped):
            logger.warning(
                f"Twist {k_idx} stopped on max_time after {per_twist_stopped[k_idx] - 1} of {num_mcmc_steps} branching steps: "
                "it is not complete, the BZ average below uses it as is, and a restart reruns it."
            )
    energy = bz_mean_energy(per_twist_E, per_twist_E_std, weights)
    force = None
    if compute_force:
        force = bz_mean_force(per_twist_F, per_twist_F_std, weights)
    stress = None
    P_hydro = None
    if compute_stress:
        stress = bz_mean_force(per_twist_T, per_twist_T_std, weights)
        # Independent twists -> the same weighted quadrature combine as every
        # other BZ observable; the WITHIN-twist correlation between the normal
        # stresses was already absorbed by the per-twist trace jackknife.
        P_hydro = bz_mean_force(per_twist_P, per_twist_P_std, weights)
    mpc = None
    if mpc_enabled:
        mpc = _aggregate_twist_mpc(
            per_twist_dE_mpc,
            per_twist_dE_mpc_std,
            per_twist_E_mpc,
            per_twist_E_mpc_std,
            weights,
            cutoff=mpc_cutoff,
        )

    return TwistAverageResult(
        energy=energy,
        force=force,
        per_twist_E=per_twist_E,
        per_twist_E_std=per_twist_E_std,
        per_twist_F=per_twist_F,
        per_twist_F_std=per_twist_F_std,
        k_points=k_points,
        stress=stress,
        P_hydro=P_hydro,
        per_twist_T=per_twist_T,
        per_twist_T_std=per_twist_T_std,
        per_twist_P_hydro=per_twist_P,
        per_twist_P_hydro_std=per_twist_P_std,
        mpc=mpc,
    )


def run_twist_average_optimize(
    hamiltonian_list: Sequence[Hamiltonian_data],
    weights: npt.ArrayLike,
    *,
    mcmc_seed: int = 34467,
    num_walkers: int = 40,
    world_comm: MPI.Comm | None = None,
    mcmc_kwargs: dict | None = None,
    optimize_kwargs: dict | None = None,
    checkpoint_dir: str | None = None,
    resume: bool = False,
    source_hamiltonian_path: str | None = None,
    dump_dir: str | None = None,
) -> Hamiltonian_data:
    """Twist-averaged SR/LM optimization driver.

    Drives one :class:`jqmc.jqmc_mcmc.MCMC` instance per twist, then calls
    :meth:`MCMC.run_optimize` on it. The twist-decomposed real-coordinate
    optimizer reduces shared Jastrow blocks over the all-twist ``world_comm``
    and determinant blocks only over the twist-local sampler communicator. The
    driver separately gathers one energy estimate per twist root, constructs
    the weighted BZ estimate, and broadcasts it back to every rank.  Thus the
    shared update stays byte-synchronised while determinant updates remain
    twist-local.

    Two contracts the driver pins:

    * **byte-identity** at ``N_twist == 1``: with one
      Hamiltonian the orchestrator constructs a single ``MCMC`` on the
      supplied ``world_comm`` and forwards to ``run_optimize`` -- no extra
      comm split, no extra reductions, no extra MCMC steps. The optimized
      Hamiltonian must equal what a direct ``MCMC(...).run_optimize(...)``
      call would have produced.
    * **block-specific reduction routing** at ``N_twist > 1``:
      ``world_comm`` is split via
      :func:`split_world_for_twists` (requires divisibility); each rank
      runs ``run_optimize`` on its own twist with the all-rank
      ``world_comm`` passed in as ``world_comm_in``. The optimizer picks the
      communicator from each block's ``sync_scope`` metadata.

    Args:
        hamiltonian_list: One :class:`jqmc.hamiltonians.Hamiltonian_data`
            per twist (length == ``N_twist``).
        weights: Per-twist raw weights ``w_k``.  They define the BZ energy
            and independent-chain error used for every optimization-step log
            and for the world/shared LM energy gate.  The shared Jastrow block
            reduces through ``world_comm`` while the determinant block reduces
            through ``intra_comm``.
        mcmc_seed: Base RNG seed. Multi-twist runs derive a deterministic
            twist namespace before ``MCMC`` folds in the intra-rank.
        num_walkers: Walkers per (rank, twist).
        world_comm: Optional MPI comm. Defaults to ``MPI.COMM_WORLD``.
        mcmc_kwargs: Extra kwargs forwarded to ``MCMC.__init__``. The
            driver always enables ``comput_log_WF_param_deriv`` and
            ``comput_e_L_param_deriv`` (required by SR/LM); a caller-
            supplied value is honoured if explicitly set to ``True``.
        optimize_kwargs: Kwargs forwarded to ``MCMC.run_optimize`` (e.g.
            ``num_mcmc_steps``, ``num_opt_steps``, ``opt_geminal``,
            ``optimizer_kwargs={'method': 'sr', 'delta': 1e-2}``). Default
            empty -- the caller is expected to specify at least
            ``num_mcmc_steps`` / ``num_opt_steps`` for any non-trivial run.
            ``max_time`` is the wall-clock budget of this whole call, shared
            by all ``wf_dump_freq`` chunks and all ranks: like a single
            ``run_optimize`` call, the run stops after the optimization step
            at which world rank 0 sees it exceeded or predicts that the next
            step would end after it, or at which
            ``external_control_opt.toml`` holds ``stop = true``, and writes
            that chunk's checkpoint (``converged`` stays false).
        checkpoint_dir: Optional directory for the per-twist checkpoint.
            The orchestrator wraps
            :meth:`MCMC.run_optimize` in a chunked loop whose chunk size
            is the SR ``wf_dump_freq`` from ``optimize_kwargs`` (default
            = ``num_opt_steps`` -> single chunk = single save at the end).
            After every chunk the per-twist file ``twist_{idx}.h5`` and
            the canonical shared Jastrow file ``shared_state.h5`` are
            rewritten.
        resume: When ``True`` AND ``checkpoint_dir`` is set the driver
            looks for a partial per-twist file and, if found, restores
            the ``MCMC`` (walker state + RNG + ``i_opt`` counter) plus
            overrides the per-twist Jastrow with the canonical
            ``shared_state.h5`` block before continuing. The chunked
            outer loop then runs ``num_opt_steps`` *additional*
            iterations on top of the restored ``i_opt`` (the same
            semantics as a single-twist ``MCMC.run_optimize`` restart).
            All twists must resume from the same ``i_opt``; a mismatch
            (e.g. a kill between two per-twist saves) raises.
        source_hamiltonian_path: The combined ``hamiltonian_h5`` this run
            started from.  Required (together with ``checkpoint_dir``) for the
            multi-twist opt-step dump: after every chunk world rank 0 writes
            ``dump_dir/hamiltonian_data_opt_step_{i_opt}.h5`` holding the
            optimized state of EVERY twist, assembled from the just-written
            per-twist checkpoints by
            :func:`jqmc._twist_checkpoint.write_twist_optimized_hamiltonian`.
            That file is a complete combined input, so a fresh VMC / MCMC / LRDMC run
            can be started from any dumped step.  Without it a multi-twist run
            dumps NO opt-step Hamiltonian at all (a rank-0-twist-only file
            would be silently consumed downstream as a single-twist input).
        dump_dir: Directory for the multi-twist opt-step dumps.  Defaults to
            the current working directory, matching where the single-twist
            optimizer dump lands.

    Returns:
        The optimized :class:`Hamiltonian_data` for *this rank's* twist.
        Multi-rank callers must gather per twist if they need every
        optimized state on every rank (typical workflow: each twist-root
        writes its slot of the combined HDF5 layout after this call).
    """
    if mcmc_kwargs is None:
        mcmc_kwargs = {}
    if bool(mcmc_kwargs.get("mpc", False)):
        raise ValueError(
            "run_twist_average_optimize: mpc=True is not supported because optimization does not publish a final "
            "MPC estimate. Run an MCMC or LRDMC sampling job after optimization."
        )
    if optimize_kwargs is None:
        optimize_kwargs = {}
    if not hamiltonian_list:
        raise ValueError("run_twist_average_optimize: hamiltonian_list is empty.")
    n_twist = len(hamiltonian_list)
    W = normalize_twist_weights(weights)
    if W.size != n_twist:
        raise ValueError(f"run_twist_average_optimize: weights size {W.size} != n_twist {n_twist}.")

    from mpi4py import MPI

    from .jqmc_mcmc import MCMC
    from .jqmc_optimizer import OptimizationConfig, _optimization_control_file

    world = world_comm if world_comm is not None else MPI.COMM_WORLD

    # Same comm-split contract as the VMC / GFMC twist drivers (equal
    # compute per twist). Single-twist short-circuits to the identity.
    if n_twist == 1:
        intra_comm = world
        inter_comm = world
        twist_idx = 0
    else:
        split = split_world_for_twists(world, n_twist=n_twist)
        intra_comm = split.intra_comm
        inter_comm = split.inter_comm
        twist_idx = split.twist_index

    ham_k = hamiltonian_list[twist_idx]

    # SR/LM needs both derivative caches enabled; the user is allowed to
    # override only by *also* asking for them (forbidding disablement
    # makes the contract explicit).
    init_kwargs = dict(mcmc_kwargs)
    init_kwargs.setdefault("comput_log_WF_param_deriv", True)
    init_kwargs.setdefault("comput_e_L_param_deriv", True)
    if not init_kwargs["comput_log_WF_param_deriv"] or not init_kwargs["comput_e_L_param_deriv"]:
        raise ValueError(
            "run_twist_average_optimize: SR/LM requires both comput_log_WF_param_deriv=True and comput_e_L_param_deriv=True."
        )

    # Pull the outer-loop knobs from optimize_kwargs. ``num_opt_steps``
    # is the number of ADDITIONAL steps this call runs (on resume it is
    # added to the restored ``i_opt``); ``wf_dump_freq`` doubles as the chunk size so the inner
    # ``run_optimize`` call sees the same dump cadence semantics as
    # before (single chunk -> single dump at the end). The remaining
    # kwargs forward unchanged to the inner ``run_optimize``.
    opt_kwargs = dict(optimize_kwargs)
    if "energy_aggregator" in opt_kwargs:
        raise ValueError("run_twist_average_optimize owns energy_aggregator; callers must not override it.")
    if "sampler_weight" in opt_kwargs:
        raise ValueError("run_twist_average_optimize owns sampler_weight; callers must not override it.")
    if "twist_index" in opt_kwargs:
        raise ValueError("run_twist_average_optimize owns twist_index; callers must not override it.")
    requested_opt_steps = int(opt_kwargs.pop("num_opt_steps", 1))
    chunk_size = int(opt_kwargs.pop("wf_dump_freq", requested_opt_steps))
    # ``max_time`` is the budget of the whole chunked loop, not of each chunk:
    # every ``run_optimize`` call restarts its own clock, so each chunk only
    # receives what is left of it (see the loop below).
    max_time = float(opt_kwargs.pop("max_time", OptimizationConfig.max_time))
    if chunk_size <= 0:
        chunk_size = max(1, requested_opt_steps)

    # Resume vs fresh construction. ``MCMC.load_from_hdf5`` restores
    # walker state + RNG + ``i_opt``; the orchestrator then overrides the
    # per-twist Jastrow with the canonical ``shared_state.h5`` block so
    # the "shared Jastrow is single source of truth" invariant
    # holds across the kill / resume boundary.
    loaded_from_ckpt = False
    if checkpoint_dir is not None and resume:
        from dataclasses import replace as _dc_replace

        from ._twist_checkpoint import (
            load_shared_state,
            shared_state_path,
            twist_checkpoint_path,
        )

        ckpt_path = twist_checkpoint_path(checkpoint_dir, twist_idx)
        # Resume-or-fresh is one decision for the whole world: twists that
        # resumed beside twists that started fresh would enter different
        # collectives (#805). A partial set of per-twist checkpoints fails on
        # every rank, naming the twists that lack theirs.
        have_checkpoint = os.path.isfile(ckpt_path)
        twist_presence = (
            world.allgather((twist_idx, have_checkpoint)) if world.Get_size() > 1 else [(twist_idx, have_checkpoint)]
        )
        if any(present for _, present in twist_presence) and not all(present for _, present in twist_presence):
            lacking = sorted({index for index, present in twist_presence if not present})
            raise ValueError(
                f"Twist-averaged optimization resume from {checkpoint_dir!r}: the per-twist checkpoints of "
                f"twists {lacking} are missing while other twists have theirs. Restore the complete "
                "checkpoint directory or start a fresh run."
            )
        if have_checkpoint:
            shared_path = shared_state_path(checkpoint_dir)
            if not os.path.isfile(shared_path):
                raise FileNotFoundError(f"Cannot resume twist optimization without the canonical shared Jastrow: {shared_path}")
            shared_jastrow = load_shared_state(checkpoint_dir)
            mcmc = MCMC.load_from_hdf5(
                ckpt_path,
                rank=intra_comm.Get_rank(),
                mpi_comm_in=intra_comm,
                world_comm_in=world,
            )
            ham_resumed = mcmc.hamiltonian_data
            wf_resumed = ham_resumed.wavefunction_data
            mcmc.hamiltonian_data = _dc_replace(
                ham_resumed,
                wavefunction_data=_dc_replace(wf_resumed, jastrow_data=shared_jastrow),
            )
            loaded_from_ckpt = True

    if not loaded_from_ckpt:
        mcmc = MCMC(
            hamiltonian_data=ham_k,
            mcmc_seed=_twist_sampler_seed(mcmc_seed, twist_idx, n_twist),
            num_walkers=num_walkers,
            mpi_comm_in=intra_comm,
            world_comm_in=world,
            **init_kwargs,
        )

    if checkpoint_dir is not None:
        from ._twist_checkpoint import save_twist_checkpoint

    # Under twist averaging the single-twist optimizer dump is suppressed
    # outright: it would carry only the world-rank-0 twist, and a downstream
    # MCMC / LRDMC job reading it would silently sample a single-k
    # wavefunction. The complete multi-twist combined dump below replaces it with
    # exactly the same cadence (one dump per ``wf_dump_freq`` chunk, including
    # the final partial chunk and any early exit inside ``run_optimize``).
    dump_wavefunctions = n_twist > 1 and checkpoint_dir is not None and source_hamiltonian_path is not None
    if n_twist > 1 and not dump_wavefunctions:
        missing = []
        if checkpoint_dir is None:
            missing.append("checkpoint_dir")
        if source_hamiltonian_path is None:
            missing.append("source_hamiltonian_path")
        logger.warning(
            "run_twist_average_optimize: %s not given, so NO hamiltonian_data_opt_step_*.h5 "
            "will be dumped for this twist-averaged run. The single-twist dump is deliberately "
            "suppressed because it would contain only the rank-0 twist and would be silently "
            "consumed downstream as a single-twist input; pass both checkpoint_dir and "
            "source_hamiltonian_path to get the complete combined dump.",
            " and ".join(missing),
        )
    dump_directory = dump_dir if dump_dir is not None else os.getcwd()
    opt_kwargs["dump_wavefunction"] = n_twist == 1

    energy_aggregator = None
    if n_twist > 1:

        def aggregate_step_energy(local_estimate: EnergyEstimate) -> EnergyEstimate:
            return _aggregate_twist_optimization_energy(
                local_estimate,
                twist_index=twist_idx,
                weights=W,
                intra_comm=intra_comm,
                inter_comm=inter_comm,
            )

        energy_aggregator = aggregate_step_energy

    # Chunked outer loop. Each iteration:
    # 1. runs the next ``chunk`` SR iterations via ``run_optimize``;
    # 2. on the twist-root + inter-root writes the per-twist checkpoint
    #    + the shared Jastrow file;
    # 3. leaves the loop when ``run_optimize`` stopped early.
    # ``converged`` is set iff this is the final chunk.
    #
    # ``max_time`` and the external stop flag apply to the whole loop, as
    # for a single ``run_optimize`` call: the budget clock starts here and
    # each chunk gets only the remaining budget, and an early stop of a chunk
    # (budget exhausted or ``stop = true``) ends the run after its checkpoint
    # instead of starting another chunk. The wrapper owns the control file
    # through all chunk checkpoints and dumps. Only world rank 0's remaining
    # budget matters: ``run_optimize`` decides both stops there and broadcasts.
    completed = int(mcmc.i_opt)
    start_min = int(world.allreduce(completed, op=MPI.MIN))
    start_max = int(world.allreduce(completed, op=MPI.MAX))
    if start_min != start_max:
        raise RuntimeError(
            f"Twist-averaged optimization resume: the per-twist checkpoints disagree on the "
            f"completed step count (min {start_min}, max {start_max}); they must be identical "
            f"to continue. Restart from a consistent checkpoint directory."
        )
    total_opt_steps = completed + requested_opt_steps
    if completed >= total_opt_steps:
        return mcmc.hamiltonian_data
    opt_kwargs["_manage_external_control"] = False
    with _optimization_control_file(world):
        budget_start = time.perf_counter()
        while completed < total_opt_steps:
            chunk = min(chunk_size, total_opt_steps - completed)
            completed_before = completed
            remaining_time = max_time - (time.perf_counter() - budget_start)
            # Later chunks share this budget: the chunk's last step must also
            # stop the run when the next step would end after max_time.
            opt_kwargs["_more_steps_follow"] = completed + chunk < total_opt_steps
            if n_twist == 1:
                stop_reason = mcmc.run_optimize(num_opt_steps=chunk, wf_dump_freq=chunk, max_time=remaining_time, **opt_kwargs)
            else:
                stop_reason = mcmc.run_optimize(
                    num_opt_steps=chunk,
                    wf_dump_freq=chunk,
                    max_time=remaining_time,
                    energy_aggregator=energy_aggregator,
                    sampler_weight=float(W[twist_idx]),
                    twist_index=twist_idx,
                    **opt_kwargs,
                )
            completed = int(mcmc.i_opt)
            progressed_locally = int(completed > completed_before)
            progressed_world = int(world.allreduce(progressed_locally, op=MPI.MIN))
            if not progressed_world:
                raise RuntimeError(
                    "Twist-averaged optimization made no accepted progress on at "
                    "least one world rank; aborting atomically instead of retrying "
                    "the same chunk indefinitely."
                )
            if checkpoint_dir is not None:
                save_twist_checkpoint(
                    checkpoint_dir,
                    twist_idx=twist_idx,
                    mcmc_or_gfmc=mcmc,
                    intra_comm=intra_comm,
                    inter_comm=inter_comm,
                    driver_type="MCMC",
                    target_steps=total_opt_steps,
                    converged=(completed >= total_opt_steps),
                    normalized_twist_weight=float(W[twist_idx]),
                    num_twists=n_twist,
                )
            if dump_wavefunctions:
                from ._twist_checkpoint import write_twist_optimized_hamiltonian

                # Every twist root must have finished writing twist_{idx}.h5 (and
                # the inter-root shared_state.h5) before the assembler reads them.
                world.Barrier()
                if inter_comm is not None and inter_comm.Get_rank() == 0:
                    dump_name = f"hamiltonian_data_opt_step_{completed}.h5"
                    write_twist_optimized_hamiltonian(
                        source_hamiltonian_path,
                        checkpoint_dir,
                        os.path.join(dump_directory, dump_name),
                    )
                    logger.info(f"Twist-averaged Hamiltonian data ({n_twist} twists) is dumped as an HDF5 file: {dump_name}.")
                world.Barrier()
            if stop_reason is not None and completed < total_opt_steps:
                logger.info(
                    f"Twist-averaged optimization stopped early ({stop_reason}) after {completed} of {total_opt_steps} steps."
                )
                break

    return mcmc.hamiltonian_data


# =============================================================================
# Combined twist-averaged input HDF5 layout
#
# Combined input layout:
#
#   /_meta            attrs: n_twist, bz_weight_sum,
#                            spin_twist_pairing = "opposite" (required for
#                            independent-spin complex inputs; see
#                            SPIN_TWIST_PAIRING_ATTR)
#                     NO ``mesh`` / ``shift`` (jQMC enumerates no grid;
#                     the converter writes whatever DFT produced).
#   /shared/          _save_dataclass_to_hdf5 of the shared Hamiltonian_data
#                     template (structure, ECP / Coulomb, Jastrow, geminal
#                     with REAL AO basis -- mo_coefficients / lambda_matrix
#                     are placeholders that get overridden at load).
#   /twists/twist_{idx:03d}/   k_point (3,) fp64 (or empty for Gamma)
#                              weight  scalar fp64 (DFT w_k)
#                              mo_coefficients_up   complex128
#                              mo_coefficients_dn   complex128
#                              lambda_matrix        complex128 Hermitian
#                              aos_data_up/dn       optional full AO overrides
#                                                   (optimized dumps; de-tiled)
#
# The writer is intentionally simple: it takes a fully-assembled list of
# (Hamiltonian_data, k, w) and emits the layout. Building the per-twist
# arrays from a DFT output is the converter's job; that lives in
# jqmc/qe_to_jqmc.py and jqmc/trexio_wrapper.py.
# =============================================================================


@dataclass(frozen=True)
class TwistInputMeta:
    """Header read by :func:`load_twist_average_input` from a combined input.

    Attributes:
        n_twist: Number of twists in ``/twists``.
        k_points: List of ``tuple[float, float, float] | None`` per twist
            (``None`` if the dataset is empty, signalling Gamma).
        weights: Per-twist raw DFT weights (un-normalised) -- normalise
            with :func:`normalize_twist_weights`.
        bz_weight_sum: ``sum_k w_k`` as written by the converter, used to
            sanity-check completeness vs the symmetry star multiplicities.
    """

    n_twist: int
    k_points: list
    weights: np.ndarray
    bz_weight_sum: float


def _twist_group_name(idx: int) -> str:
    """Stable on-disk name: ``twist_000``, ``twist_001``, ..."""
    return f"twist_{int(idx):03d}"


def is_twist_average_input(filepath: str) -> bool:
    """Cheap predicate: does ``filepath`` contain a combined twist payload?

    Used by :py:mod:`jqmc.jqmc_cli` to auto-detect a twist-averaged
    input and route to the twist driver. Partial combined containers stay on
    that route so the reader can report missing content rather than silently
    selecting a single-Hamiltonian interpretation. HDF5/open errors propagate.
    """
    import h5py

    from .hamiltonians import _is_twist_average_layout, _require_twist_average_layout

    with h5py.File(filepath, "r") as f:
        if not _is_twist_average_layout(f):
            return False
        _require_twist_average_layout(f)
        return True


class TwistAverageWriter:
    """Incremental writer for the combined twist-averaged input HDF5.

    The batch entry point :func:`save_twist_average_input` needs every twist's
    payload in memory at once. For a B-spline twist-average that payload is the
    per-twist control-point table -- 25.2 GiB at 108 atoms, 501
    bands, mesh_factor 0.5 -- so a 4-twist run holds ~100 GiB of coefficients
    (twice that with an independent dn channel) on top of the template and the
    live projection buffers, which runs out of memory on a 251 GB node partway
    through twist 2.

    This writer emits ``/shared`` once up front and then one
    ``/twists/twist_<ik>/`` group per call, so the caller can drop each twist's
    arrays before projecting the next: peak holds ONE twist's coefficients
    instead of ``n_twist``. The bytes written, their layout and their values are
    identical to the batch writer -- only the order and lifetime change.

    Use as a context manager, or call :meth:`close` explicitly::

        with TwistAverageWriter(path, template_ham, n_twist) as w:
            for ik, ...:
                w.write_twist(ik, k_point, weight, mo_up, mo_dn, ...)

    ``/_meta/bz_weight_sum`` is finalised in :meth:`close` from the weights seen
    so far, which is why an unclosed writer produces an unusable file.
    """

    def __init__(self, filepath: str, template_hamiltonian: Hamiltonian_data, n_twist: int) -> None:
        import h5py

        from .determinant import SD_data
        from .hamiltonians import _save_dataclass_to_hdf5

        if int(n_twist) <= 0:
            raise ValueError("TwistAverageWriter: at least one twist is required.")
        # One phase-free correlation factor belongs to /shared, not to
        # the determinant twist stored in the template.
        template_hamiltonian.wavefunction_data.jastrow_data.sanity_check()
        self._n_twist = int(n_twist)
        self._template = template_hamiltonian
        self._is_sd = isinstance(template_hamiltonian.wavefunction_data.geminal_data, SD_data)
        self._weights: list[float] = []
        self._written: set[int] = set()
        self._f = h5py.File(filepath, "w")
        try:
            meta = self._f.create_group("_meta")
            meta.attrs["n_twist"] = self._n_twist
            meta.attrs[SPIN_TWIST_PAIRING_ATTR] = SPIN_TWIST_PAIRING_OPPOSITE
            # Placeholder; close() rewrites it with the real sum. A file whose
            # writer was never closed keeps this 0.0 and fails the loader's
            # weight checks rather than silently normalising to nonsense.
            meta.attrs["bz_weight_sum"] = 0.0
            shared_grp = self._f.create_group("shared")
            _save_dataclass_to_hdf5(shared_grp, template_hamiltonian)
            self._twists_grp = self._f.create_group("twists")
        except BaseException:
            self._f.close()
            raise

    def __enter__(self) -> "TwistAverageWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self._f.close()

    def record_attrs(self, attrs: Mapping[str, float]) -> None:
        """Store scalar conversion metadata (e.g. the PW G-sphere taper) on ``/_meta``."""
        meta = self._f["_meta"]
        for key, value in attrs.items():
            meta.attrs[str(key)] = value

    def write_twist(
        self,
        index: int,
        k_point: Sequence[float] | None,
        weight: float,
        mo_coefficients_up: npt.ArrayLike,
        mo_coefficients_dn: npt.ArrayLike,
        lambda_matrix: npt.ArrayLike | None = None,
        composite_child: CompositeChildData | None = None,
        composite_child_dn: CompositeChildData | None = None,
    ) -> None:
        """Write one ``/twists/twist_<index>/`` group.

        Args:
            index: 0-based twist index; each index may be written once.
            k_point: 3-tuple, or ``None`` for the real Gamma path.
            weight: Raw DFT weight for this twist.
            mo_coefficients_up: Occupied up-spin MO block for this twist.
            mo_coefficients_dn: Occupied dn-spin MO block for this twist.
            lambda_matrix: AGP/AGPn templates only.
            composite_child: Optional ``(kind_tag, payload)`` per-twist child.
            composite_child_dn: Optional dn-spin override of the above.

        Per-twist argument semantics match :func:`save_twist_average_input`;
        see its docstring for the encoding of the composite-child entries.
        """
        from ._setting import GEMINAL_REPRESENTATION_ATOL

        idx = int(index)
        if not 0 <= idx < self._n_twist:
            raise ValueError(f"TwistAverageWriter.write_twist: index {idx} outside [0, {self._n_twist}).")
        if idx in self._written:
            raise ValueError(f"TwistAverageWriter.write_twist: twist {idx} written twice.")
        if not self._is_sd and lambda_matrix is None:
            raise ValueError("TwistAverageWriter.write_twist: AGP/AGPn inputs require a lambda matrix per twist.")

        if self._is_sd:
            geminal = self._template.wavefunction_data.geminal_data
            expected_rows = {"up": int(geminal.num_electron_up), "dn": int(geminal.num_electron_dn)}
            for spin, coefficients in (("up", mo_coefficients_up), ("dn", mo_coefficients_dn)):
                shape = np.asarray(coefficients).shape
                if len(shape) != 2 or shape[0] != expected_rows[spin]:
                    raise ValueError(
                        "save_twist_average_input: canonical SD twist "
                        f"{idx} {spin}-spin MO coefficients must have exactly "
                        f"{expected_rows[spin]} occupied rows in a 2-D array; "
                        f"got {shape}."
                    )
            if bool(geminal.time_reversal) and not np.allclose(
                np.asarray(mo_coefficients_dn),
                np.conj(np.asarray(mo_coefficients_up)[: int(geminal.num_electron_dn)]),
                atol=GEMINAL_REPRESENTATION_ATOL,
                rtol=RTOL_BIT_EXACT,
            ):
                raise ValueError(
                    "save_twist_average_input: canonical SD twist "
                    f"{idx} with time_reversal=True requires down-spin "
                    "occupied coefficients to equal "
                    "conj(up[:num_electron_dn])."
                )

        g = self._twists_grp.create_group(_twist_group_name(idx))
        # ``is_gamma`` reflects the caller's choice -- ``None`` -> the real
        # Gamma-point path; a tuple (including ``(0, 0, 0)``) -> the complex
        # Bloch path. See the batch writer for why a tuple ``(0, 0, 0)`` must
        # NOT be collapsed to ``is_gamma = True``.
        if k_point is None:
            g.create_dataset("k_point", data=np.zeros(3, dtype=np.float64))
            g.attrs["is_gamma"] = True
        else:
            k_arr = np.asarray(k_point, dtype=np.float64).ravel()
            if k_arr.shape != (3,):
                raise ValueError(f"save_twist_average_input: k_points[{idx}] shape {k_arr.shape} != (3,)")
            g.create_dataset("k_point", data=k_arr)
            g.attrs["is_gamma"] = False
        g.create_dataset("weight", data=float(weight))
        # Preserve the caller's dtype (see the batch writer's note).
        g.create_dataset("mo_coefficients_up", data=np.asarray(mo_coefficients_up))
        g.create_dataset("mo_coefficients_dn", data=np.asarray(mo_coefficients_dn))
        if not self._is_sd:
            g.create_dataset("lambda_matrix", data=np.asarray(lambda_matrix))

        if composite_child_dn is not None and composite_child is None:
            raise ValueError(
                f"save_twist_average_input: twist {idx} has a dn-spin composite "
                "child but no up-spin one; the dn child is an LSDA override of "
                "the shared child, not a standalone."
            )
        for group_name, entry in (("composite_child", composite_child), ("composite_child_dn", composite_child_dn)):
            if entry is None:
                continue
            kind_tag, payload = entry
            cg = g.create_group(group_name)
            cg.attrs["kind"] = kind_tag
            if kind_tag == "pw_complex":
                cg.create_dataset("g_indices", data=np.asarray(payload, dtype=np.int64))
            elif kind_tag == "bspline_complex":
                cg.create_dataset("spline_coefficients", data=np.asarray(payload, dtype=np.complex128))
            else:
                raise ValueError(
                    f"save_twist_average_input: unknown {group_name} kind "
                    f"{kind_tag!r}; expected 'pw_complex' or 'bspline_complex'."
                )
        # Flush this twist before the caller frees its arrays, so a crash in
        # the next projection leaves a partial-but-coherent file instead of a
        # buffer-cached one.
        self._f.flush()
        self._weights.append(float(weight))
        self._written.add(idx)

    def write_tiled_twist(self, index: int, weight: float, geminal) -> None:
        """Store one ragged occupied k-block SD without global MO arrays."""
        from .atomic_orbital import _structure_data_equal
        from .hamiltonians import _save_dataclass_to_hdf5

        if not self._is_sd or not getattr(geminal, "is_tiled_representation", False):
            raise ValueError("tiled twist storage requires a canonical tiled SD")
        geminal.sanity_check()
        if not _structure_data_equal(geminal.orb_data_up_spin.structure_data, self._template.structure_data):
            raise ValueError("all tiled twists must use the same physical supercell")
        template_sd = self._template.wavefunction_data.geminal_data
        if not template_sd.is_tiled_representation:
            raise ValueError("tiled twist storage requires a tiled shared template")
        if (geminal.num_electron_up, geminal.num_electron_dn) != (template_sd.num_electron_up, template_sd.num_electron_dn):
            raise ValueError("all tiled twists must have the same spin populations")
        if not 0 <= index < self._n_twist or index in self._written:
            raise ValueError("invalid or duplicate tiled twist index")
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("tiled twist weight must be finite and positive")
        g = self._twists_grp.create_group(_twist_group_name(index))
        g.attrs["is_gamma"] = False
        g.create_dataset("k_point", data=geminal.orb_data_up_spin.supertwist)
        g.create_dataset("weight", data=weight)
        if index == 0 and geminal is template_sd:
            g["geminal"] = self._f["shared/wavefunction_data/geminal_data"]
        else:
            _save_dataclass_to_hdf5(g.create_group("geminal"), geminal)
        self._f.flush()
        self._weights.append(float(weight))
        self._written.add(index)

    def close(self) -> None:
        """Finalise ``/_meta`` and close the file."""
        try:
            missing = sorted(set(range(self._n_twist)) - self._written)
            if missing:
                raise ValueError(f"TwistAverageWriter.close: twists {missing} were never written.")
            self._f["_meta"].attrs["bz_weight_sum"] = float(np.sum(np.asarray(self._weights, dtype=np.float64)))
        finally:
            self._f.close()


def save_twist_average_input(
    filepath: str,
    template_hamiltonian: Hamiltonian_data,
    k_points: Sequence[npt.ArrayLike | None],
    weights: npt.ArrayLike,
    mo_coefficients_up_per_twist: Sequence[npt.ArrayLike],
    mo_coefficients_dn_per_twist: Sequence[npt.ArrayLike],
    lambda_matrix_per_twist: Sequence[npt.ArrayLike] | None = None,
    composite_child_per_twist: Sequence[CompositeChildData | None] | None = None,
    composite_child_dn_per_twist: Sequence[CompositeChildData | None] | None = None,
) -> None:
    """Write the combined twist-averaged input HDF5.

    Args:
        filepath: Destination ``hamiltonian_data.h5``.
        template_hamiltonian: A :class:`jqmc.hamiltonians.Hamiltonian_data` to
            serialise under ``/shared``. Its
            ``wavefunction_data.geminal_data.mo_coefficients_*`` fields are
            PLACEHOLDERS -- the per-twist
            datasets under ``/twists/twist_*/`` override them at load.
        k_points: Length-``N_twist`` list of per-twist k_point tuples (3-tuples)
            or ``None`` (Gamma). The writer emits a length-3 dataset per
            twist; ``None`` is encoded as a zero vector with a sentinel attr.
        weights: Length-``N_twist`` per-twist raw DFT weights.
        mo_coefficients_up_per_twist: Length-``N_twist`` list of complex
            MO arrays. For SD each has exactly ``N_up`` rows. The AO width
            may vary by twist for composite leaves and is checked against
            the reconstructed per-twist AO child by the loader.
        mo_coefficients_dn_per_twist: Same for the down-spin MOs, with
            exactly ``N_dn`` rows for SD.
            Each block is the conjugate of the down channel's own ``+k``
            coefficients; the loader puts the up leaf at ``+k`` and the
            down leaf at ``-k``. Under ``time_reversal=True`` the writer
            enforces ``dn == conj(up[:N_dn])``. Independent down orbitals
            (``conj(C_beta)`` of an LSDA / UHF input) require a
            ``time_reversal=False`` template.
        lambda_matrix_per_twist: For AGP/AGPn templates only, a
            length-``N_twist`` list of lambda matrices. Canonical SD inputs
            omit this argument and never persist a lambda dataset.
        composite_child_per_twist: Optional length-``N_twist`` list with the
            per-twist k-dependent child of the composite AO leaf, for
            ``kind="pgto+pw"`` / ``kind="pgto+bspline"`` multi-twist runs.
            Each entry is either ``None`` (pgto-only kinds; the AO leaf in
            ``/shared`` is k-independent and the loader needs no override)
            or one of:
              * ``("pw_complex", _g_indices_tuple)``: replaces the composite's
                PW child with an :class:`AOs_plane_wave_complex_data` rebuilt
                from the per-twist QE G-list. ``_g_indices_tuple`` is the
                flat 3*``num_ao_pw`` int tuple stored on the leaf.
              * ``("bspline_complex", spline_coefficients_complex)``:
                replaces the composite's B-spline child with an
                :class:`AOs_bspline_complex_data` rebuilt from per-twist
                complex spline coefficients (``(num_ao, prod(spline_shape))``
                complex128). Static fields (``spline_shape``, ``spline_order``)
                are inherited from the shared template.
            The pgto child and all other state (structure, Coulomb, Ewald,
            Jastrow) stay shared. When the parameter is ``None`` or the entry
            is ``None``, no per-twist composite child group is written.
        composite_child_dn_per_twist: Optional length-``N_twist`` list with a
            DOWN-SPIN-specific k-dependent child, same per-entry encoding as
            ``composite_child_per_twist``. Needed by LSDA (``nspin = 2``)
            saves whose child payload is spin-dependent (the per-band
            spline child encodes the bands themselves), where the up/dn
            channels cannot share one child. Written under
            ``/twists/twist_*/composite_child_dn``; when absent the loader
            falls back to the spin-shared ``composite_child`` for the dn
            leaf (the restricted convention). The payload is stored in the
            ``+k`` representation for BOTH groups; the loader applies the
            BCS / TR-pair ``-k`` flip (``with_negated_k``) to the dn leaf.

    Raises:
        ValueError: If the per-twist input lengths disagree.
    """
    from .determinant import SD_data

    n_twist = len(k_points)
    is_sd = isinstance(template_hamiltonian.wavefunction_data.geminal_data, SD_data)
    if not (
        len(weights) == n_twist
        and len(mo_coefficients_up_per_twist) == n_twist
        and len(mo_coefficients_dn_per_twist) == n_twist
    ):
        raise ValueError(
            "save_twist_average_input: all per-twist inputs must have the same length "
            f"(got n_twist={n_twist}, weights={len(weights)}, "
            f"mo_up={len(mo_coefficients_up_per_twist)}, "
            f"mo_dn={len(mo_coefficients_dn_per_twist)})."
        )
    if not is_sd and (lambda_matrix_per_twist is None or len(lambda_matrix_per_twist) != n_twist):
        raise ValueError("save_twist_average_input: AGP/AGPn inputs require one lambda matrix per twist.")
    if n_twist == 0:
        raise ValueError("save_twist_average_input: at least one twist is required.")

    # Thin loop over the incremental writer: identical bytes, identical layout.
    # Callers that can afford to hold every twist use this entry point; the
    # converter drives :class:`TwistAverageWriter` directly so it never holds
    # more than one twist (see its docstring for the memory bound).
    with TwistAverageWriter(filepath, template_hamiltonian, n_twist) as writer:
        for idx in range(n_twist):
            writer.write_twist(
                idx,
                k_points[idx],
                float(np.asarray(weights, dtype=np.float64)[idx]),
                mo_coefficients_up_per_twist[idx],
                mo_coefficients_dn_per_twist[idx],
                lambda_matrix=None if is_sd else lambda_matrix_per_twist[idx],
                composite_child=composite_child_per_twist[idx] if composite_child_per_twist is not None else None,
                composite_child_dn=(composite_child_dn_per_twist[idx] if composite_child_dn_per_twist is not None else None),
            )


def load_twist_average_input(filepath: str) -> TwistInputMeta:
    """Read the ``/_meta`` + per-twist ``k_point`` / ``weight`` header.

    Cheap O(N_twist) entry-point used by the driver to enumerate twists +
    plan the comm split before any heavy ``Hamiltonian_data`` deserialise.
    """
    import h5py

    from .hamiltonians import _require_twist_average_layout

    with h5py.File(filepath, "r") as f:
        n_twist = _require_twist_average_layout(f)
        bz_weight_sum = float(f["_meta"].attrs.get("bz_weight_sum", 0.0))

        twists = f["twists"]
        k_points: list = []
        weights = np.zeros(n_twist, dtype=np.float64)
        for idx in range(n_twist):
            g = twists[_twist_group_name(idx)]
            is_gamma = bool(g.attrs.get("is_gamma", False))
            k = np.asarray(g["k_point"]).reshape(-1)
            if is_gamma:
                k_points.append(None)
            else:
                k_points.append(tuple(float(x) for x in k))
            weights[idx] = float(np.asarray(g["weight"]).reshape(()))

    return TwistInputMeta(
        n_twist=n_twist,
        k_points=k_points,
        weights=weights,
        bz_weight_sum=bz_weight_sum,
    )


def _build_per_twist_ao_leaf(
    child_kind: str,
    child_data: npt.ArrayLike,
    new_struct: Structure_data,
    template_leaf: AOs_base_data,
    *,
    k_point: Sequence[float] | None,
) -> AOs_base_data:
    """Rebuild a per-twist k-dependent AO leaf from a saved payload.

    Used by :py:func:`_swap_composite_child` for both the composite-child
    case (``kind="pgto+pw"`` / ``"pgto+bspline"``) and the bare-leaf case
    (``kind="pw"`` / ``"bspline"``). ``template_leaf`` is the leaf at the
    swap slot in the shared template; it supplies the k-independent
    static fields (``spline_shape``, ``spline_order``) that the saved
    payload does not carry. ``k_point`` is the per-twist fractional
    Bloch twist that the rebuilt complex leaf must carry on its own
    ``k_point`` field (``k`` lives on the AO leaf, not on
    :class:`Structure_data`); ``None`` is allowed only when the saved
    twist is Gamma, in which case the complex leaf builders below would
    reject the rebuild via ``sanity_check``.
    """
    from .atomic_orbital import (
        AOs_bspline_complex_data,
        AOs_plane_wave_complex_data,
    )

    if child_kind == "pw_complex":
        g_indices_flat = np.asarray(child_data, dtype=np.int64).ravel()
        num_ao_pw = int(g_indices_flat.size // 3)
        new_leaf = AOs_plane_wave_complex_data(
            structure_data=new_struct,
            num_ao=num_ao_pw,
            k_point=k_point,
            _g_indices=tuple(int(x) for x in g_indices_flat.tolist()),
        )
        new_leaf.sanity_check()
        return new_leaf
    if child_kind == "bspline_complex":
        if not hasattr(template_leaf, "spline_shape") or not hasattr(template_leaf, "spline_order"):
            raise ValueError(
                "_build_per_twist_ao_leaf(bspline_complex): shared template's leaf at the "
                f"k-dependent slot is of type {type(template_leaf).__name__} and does not "
                "carry spline_shape / spline_order; cannot rebuild."
            )
        spline_shape = tuple(int(n) for n in template_leaf.spline_shape)
        spline_order = int(template_leaf.spline_order)
        c = np.asarray(child_data, dtype=np.complex128)
        n_band = int(c.shape[0])
        new_leaf = AOs_bspline_complex_data(
            structure_data=new_struct,
            num_ao=n_band,
            k_point=k_point,
            spline_coefficients=c.reshape(n_band, int(np.prod(spline_shape))),
            spline_shape=spline_shape,
            spline_order=spline_order,
        )
        new_leaf.sanity_check()
        return new_leaf
    raise ValueError(
        f"_build_per_twist_ao_leaf: unknown child_kind={child_kind!r}; expected 'pw_complex' or 'bspline_complex'."
    )


def _swap_composite_child(
    orb_data: MOs_data | AOs_base_data,
    child_kind: str,
    child_data: npt.ArrayLike,
    new_struct: Structure_data,
    *,
    k_point: Sequence[float] | None,
    negate_k: bool = False,
) -> MOs_data | AOs_base_data:
    """Replace the per-twist k-dependent AO state with the loaded payload.

    Two cases:

    * **Composite kinds** (``kind="pgto+pw"`` / ``"pgto+bspline"``):
      ``orb_data`` wraps an :class:`AOs_composite_data` of
      ``(pgto_child, k_dep_child)``. The pgto child is k-independent and
      stays put; only ``children[1]`` is replaced by a freshly rebuilt
      leaf using ``new_struct`` and the saved payload.
    * **Single-basis kinds** (``kind="pw"`` / ``"bspline"``): ``orb_data``
      wraps a bare :class:`AOs_plane_wave_complex_data` /
      :class:`AOs_bspline_complex_data` (no composite). The whole AO leaf
      is replaced by the rebuilt one.

    Args:
        orb_data: ``MOs_data`` or bare AO leaf from the shared template
            (already structure-patched by the caller).
        child_kind: ``"pw_complex"`` or ``"bspline_complex"``.
        child_data: For ``pw_complex``: an int64 ``(3*num_ao_pw,)`` flat
            G-indices array. For ``bspline_complex``: a complex128
            ``(num_ao, prod(spline_shape))`` coefficient block.
        new_struct: The per-twist :class:`Structure_data` (with the
            twist's ``k_point`` already applied) the rebuilt leaf should
            reference.
        k_point: The twist's fractional k-point (``+k``) the leaf is
            rebuilt at, or ``None``.
        negate_k: If ``True``, flip the rebuilt leaf to ``-k``
            (``with_negated_k``) after the rebuild; used for the spin-down
            leaf of a time-reversal pair, whose payload is stored in the
            ``+k`` representation. Default ``False``.

    Returns:
        ``orb_data`` with the per-twist AO state swapped.
    """
    from .atomic_orbital import (
        AOs_bspline_complex_data,
        AOs_composite_data,
        AOs_plane_wave_complex_data,
    )
    from .molecular_orbital import MOs_data

    if isinstance(orb_data, MOs_data):
        new_aos = _swap_composite_child(
            orb_data.aos_data, child_kind, child_data, new_struct, k_point=k_point, negate_k=negate_k
        )
        return orb_data.replace(aos_data=new_aos)
    if isinstance(orb_data, AOs_composite_data):
        children = list(orb_data.children)
        if len(children) < 2:
            raise ValueError(f"_swap_composite_child: composite has {len(children)} child(ren), expected >= 2.")
        template_leaf = children[1]
        new_child = _build_per_twist_ao_leaf(child_kind, child_data, new_struct, template_leaf, k_point=k_point)
        if negate_k:
            # BCS / TR-pair dn leaf: payload is stored in the +k
            # representation; flip after the rebuild.
            new_child = new_child.with_negated_k()
        children[1] = new_child
        return orb_data.replace(children=tuple(children))
    if isinstance(orb_data, (AOs_plane_wave_complex_data, AOs_bspline_complex_data)):
        # Single-basis kind: the whole AO leaf is the per-twist k-dep state.
        new_leaf = _build_per_twist_ao_leaf(child_kind, child_data, new_struct, orb_data, k_point=k_point)
        if negate_k:
            new_leaf = new_leaf.with_negated_k()
        return new_leaf
    raise ValueError(
        f"_swap_composite_child: expected AOs_composite_data, AOs_plane_wave_complex_data, "
        f"or AOs_bspline_complex_data; got {type(orb_data).__name__}."
    )


def load_twist(filepath: str, twist_index: int) -> Hamiltonian_data:
    """Build a :class:`jqmc.hamiltonians.Hamiltonian_data` for one twist.

    Reads the shared template from ``/shared`` and overrides the
    structure ``k_point`` plus the geminal's ``mo_coefficients`` /
    ``lambda_matrix`` from ``/twists/twist_{twist_index:03d}/``. Optimized dumps
    additionally override both spin AO trees using ``aos_data_up`` and
    ``aos_data_dn``; converter inputs without them retain the shared basis. Applies
    the same post-load fixups as :py:meth:`Hamiltonian_data.load_from_hdf5`
    (image expansion, Ewald rebuild, dtype alignment).

    Args:
        filepath: A twist-averaged ``hamiltonian_data.h5`` (combined layout).
        twist_index: 0-based twist selector.

    Returns:
        Hamiltonian_data: The per-twist Hamiltonian, ready to feed an MCMC
        instance.
    """
    import h5py

    from .hamiltonians import (
        Hamiltonian_data,
        _apply_ao_basis_dtype_alignment_post_load,
        _apply_dtn_dtype_alignment_post_load,
        _apply_ewald_rebuild_post_load,
        _apply_geminal_dtype_alignment_post_load,
        _apply_image_expansion_post_load,
        _apply_jastrow_structure_backfill_post_load,
        _load_dataclass_from_hdf5,
        _load_group,
        _require_twist_average_layout,
    )

    with h5py.File(filepath, "r") as f:
        n_twist = _require_twist_average_layout(f)
        spin_twist_pairing = f["_meta"].attrs.get(SPIN_TWIST_PAIRING_ATTR, "")
        if isinstance(spin_twist_pairing, bytes):
            spin_twist_pairing = spin_twist_pairing.decode("utf-8")
        if not (0 <= twist_index < n_twist):
            raise ValueError(f"twist_index={twist_index} out of range [0, {n_twist})")
        # Shared template -- the per-twist override path runs after this.
        ham = _load_dataclass_from_hdf5(Hamiltonian_data, f["shared"])
        # Per-twist overrides.
        g = f["twists"][_twist_group_name(twist_index)]
        template_sd = ham.wavefunction_data.geminal_data
        template_tiled = getattr(template_sd, "is_tiled_representation", False)
        if "geminal" in g and not template_tiled:
            raise ValueError("tiled twist payload has an incompatible shared template")
        if template_tiled:
            from .determinant import SD_data
            from ._setting import ATOL_TWIST_CHECKPOINT_KPOINT_MATCH

            if not template_tiled or any(name not in g for name in ("geminal", "k_point", "weight")):
                raise ValueError("incomplete tiled twist payload or incompatible shared template")
            weight = np.asarray(g["weight"])
            if weight.shape != () or not np.isfinite(weight) or weight <= 0:
                raise ValueError("tiled twist weight must be finite and positive")
            k = np.asarray(g["k_point"])
            sd = _load_dataclass_from_hdf5(SD_data, g["geminal"])
            if (
                not sd.is_tiled_representation
                or k.shape != (3,)
                or not np.allclose(
                    sd.orb_data_up_spin.supertwist,
                    k,
                    atol=ATOL_TWIST_CHECKPOINT_KPOINT_MATCH,
                    rtol=RTOL_BIT_EXACT,
                )
            ):
                raise ValueError("tiled twist metadata disagrees with its orbital sector")
            if (sd.num_electron_up, sd.num_electron_dn) != (template_sd.num_electron_up, template_sd.num_electron_dn):
                raise ValueError("tiled twist spin populations disagree with the shared template")
            obj = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=sd))
            obj = _apply_image_expansion_post_load(obj)
            obj = _apply_ewald_rebuild_post_load(obj)
            obj = _apply_jastrow_structure_backfill_post_load(obj)
            obj = _apply_ao_basis_dtype_alignment_post_load(obj)
            obj = _apply_geminal_dtype_alignment_post_load(obj)
            obj = _apply_dtn_dtype_alignment_post_load(obj)
            obj.sanity_check()
            return obj
        is_gamma = bool(g.attrs.get("is_gamma", False))
        k_arr = np.asarray(g["k_point"]).reshape(-1)
        if is_gamma:
            k_point = None
        else:
            k_point = tuple(float(x) for x in k_arr)
        # Preserve dataset dtype (float64 at Gamma, complex128 off-Gamma).
        # The post-load fixups -- in particular
        # ``_apply_geminal_dtype_alignment_post_load`` and
        # ``_apply_ao_basis_dtype_alignment_post_load`` -- enforce the
        # AO-leaf <-> MO / lambda dtype contract.
        mo_up = np.asarray(g["mo_coefficients_up"])
        mo_dn = np.asarray(g["mo_coefficients_dn"])
        has_lambda_matrix = "lambda_matrix" in g
        has_any_lambda_payload = any(name.startswith("lambda_") for name in g.keys())
        lambda_payload = np.asarray(g["lambda_matrix"]) if has_lambda_matrix else None
        has_aos_up = "aos_data_up" in g
        has_aos_dn = "aos_data_dn" in g
        if has_aos_up != has_aos_dn:
            raise ValueError(f"{filepath}: twist {twist_index} must supply both aos_data_up and aos_data_dn.")
        aos_up_override = _load_group(g["aos_data_up"]) if has_aos_up else None
        aos_dn_override = _load_group(g["aos_data_dn"]) if has_aos_dn else None

        # Optional per-twist composite-child override. Read raw data
        # here; we'll rebuild the child
        # below once the new structure_data is available. LSDA saves may
        # additionally carry a dn-spin-specific child (spin-dependent
        # payloads, e.g. per-band spline coefficients of the beta bands);
        # when absent, the dn leaf reuses the spin-shared child.
        def _read_child_group(group_name: str) -> tuple[str | None, np.ndarray | None]:
            if group_name not in g:
                return None, None
            cg = g[group_name]
            child_kind = cg.attrs["kind"]
            if isinstance(child_kind, bytes):
                child_kind = child_kind.decode("utf-8")
            if child_kind == "pw_complex":
                return child_kind, np.asarray(cg["g_indices"], dtype=np.int64)
            if child_kind == "bspline_complex":
                return child_kind, np.asarray(cg["spline_coefficients"], dtype=np.complex128)
            raise ValueError(
                f"{filepath}: unknown {group_name} kind {child_kind!r} "
                f"at twist_index={twist_index}; expected 'pw_complex' or 'bspline_complex'."
            )

        composite_child_kind, composite_child_data = _read_child_group("composite_child")
        composite_child_dn_kind, composite_child_dn_data = _read_child_group("composite_child_dn")
        if composite_child_dn_kind is not None and composite_child_dn_kind != composite_child_kind:
            raise ValueError(
                f"{filepath}: composite_child_dn kind {composite_child_dn_kind!r} differs "
                f"from composite_child kind {composite_child_kind!r} at twist_index={twist_index}."
            )

    # Normalise ``_n_max`` to a Python ``int``: the HDF5 round-trip turns
    # it into ``np.int64``, which JAX's pytree-metadata equality check
    # treats as distinct from the Python-int variant produced by
    # ``Structure_data.build``, silently invalidating JIT caches and
    # triggering a re-trace (sometimes raising ``Exception raised while
    # checking equality of metadata fields of pytree``). Casting to plain
    # ``int`` keeps the static field comparable with freshly-built
    # Structure_data instances throughout the JIT lifetime.
    new_struct = ham.structure_data.replace(
        _n_max=int(ham.structure_data._n_max),
    )
    # Patch geminal_data.mo_coefficients / lambda_matrix + per-twist
    # AO-leaf k_point. The MOs live under
    # wavefunction_data.geminal_data.orb_data_{up,dn}_spin. The per-twist
    # k flows into the leaf's own ``k_point`` field. The spin channels are
    # always at opposite twists: the up leaf gets ``+k`` and the dn leaf
    # ``-k``. The stored dn coefficients are the conjugate of the dn
    # channel's own ``+k`` orbitals -- ``conj(C_up)`` for a restricted
    # (``time_reversal=True``) file, ``conj(C_beta)`` for an independent
    # (``time_reversal=False``) LSDA / UHF file.
    geminal = ham.wavefunction_data.geminal_data
    if k_point is not None and not bool(geminal.time_reversal) and spin_twist_pairing != SPIN_TWIST_PAIRING_OPPOSITE:
        raise ValueError(
            f"{filepath}: a spin-polarized (independent-spin) twist-averaged input requires "
            f"/_meta/{SPIN_TWIST_PAIRING_ATTR}='{SPIN_TWIST_PAIRING_OPPOSITE}'; without it the down-spin "
            "payload may hold same-twist (+k, +k) rows, which cannot be loaded correctly. "
            "Regenerate the file from its QE / TREXIO source with the current converter."
        )
    _kp_up = tuple(float(c) for c in k_point) if k_point is not None else None
    _kp_dn = tuple(-float(c) for c in k_point) if k_point is not None else None
    # ``k_point`` lives on the AO leaf, not on
    # :class:`Structure_data` or on the :class:`AOs_composite_data`
    # wrapper. For a composite AO leaf the twist sits on the k-dependent
    # child (slot 1) -- the pgto child (slot 0) is k-independent, and
    # the composite shell itself has no ``k_point`` field. So:
    #   * bare leaf (``kind="pgto"`` / ``"pw"`` / ``"bspline"``): replace
    #     ``k_point`` directly here;
    #   * composite (``kind="pgto+pw"`` / ``"pgto+bspline"``): leave the
    #     top wrapper alone; the per-twist composite-child swap below
    #     (post image-expansion) rebuilds the child carrying the per-twist
    #     ``k_point``.
    _top_aos_up = aos_up_override if has_aos_up else geminal.orb_data_up_spin.aos_data
    _top_aos_dn = aos_dn_override if has_aos_dn else geminal.orb_data_dn_spin.aos_data

    def _retwist_pre_expansion(
        aos: AOs_base_data,
        kp: Sequence[float] | None,
    ) -> AOs_base_data:
        """Stamp THIS twist's k on a template AO leaf (pre image-expansion).

        The /shared template stores the ik=1 twist; every k-carrying leaf
        must be re-stamped with the per-twist k BEFORE
        ``_apply_image_expansion_post_load`` recomputes the k-derived
        image phases (the de-tiled gto child rebuilds its
        ``_image_phase_factors`` from its own ``k_point``). For a
        composite the k-dependent child (slot 1) is fully swapped below
        anyway, but the pgto child (slot 0) is re-stamped only here.
        """
        if isinstance(aos, AOs_composite_data):
            return aos.replace(children=tuple(_retwist_pre_expansion(c, kp) for c in aos.children))
        if not hasattr(aos, "k_point"):
            return aos
        retwisted = aos.replace(k_point=kp)
        # This load-only attribute is not a dataclass field, so replace()
        # would otherwise discard the saved periodic truncation policy.
        if hasattr(aos, "_image_truncation_tol"):
            object.__setattr__(retwisted, "_image_truncation_tol", aos._image_truncation_tol)
        return retwisted

    new_aos_up_pre = _retwist_pre_expansion(_top_aos_up, _kp_up)
    new_aos_dn_pre = _retwist_pre_expansion(_top_aos_dn, _kp_dn)
    # Swap only the AO leaf at this point -- keep the template's MO
    # coefficients so the intermediate geminal stays valid through
    # ``_apply_image_expansion_post_load`` (the per-twist MO + lambda
    # are dispatched below, AFTER the gto -> pgto re-expansion has
    # promoted the composite[0] (pgto child) to a periodic leaf, so the
    # composite does not mix a molecular gto with a periodic child).
    new_orb_up_pre = geminal.orb_data_up_spin.replace(aos_data=new_aos_up_pre)
    new_orb_dn_pre = geminal.orb_data_dn_spin.replace(aos_data=new_aos_dn_pre)
    geminal_pre = geminal.replace(orb_data_up_spin=new_orb_up_pre, orb_data_dn_spin=new_orb_dn_pre)
    wf_pre = ham.wavefunction_data.replace(geminal_data=geminal_pre)
    obj = ham.replace(structure_data=new_struct, wavefunction_data=wf_pre)

    # Image expansion FIRST: gto -> pgto for any AOs_*_gto_data leaf
    # (de-tiled persistence form). After this, composite children that
    # were gto become pgto (periodic), so the composite-mix invariant
    # (open-boundary gto cannot share a composite with a periodic leaf)
    # holds before we swap in the per-twist k-dependent child below.
    obj = _apply_image_expansion_post_load(obj)

    # Per-twist composite-child override.
    # At this point composite[0] is a periodic pgto and composite[1] is
    # the template's complex Bloch leaf; the rebuilt child carries the
    # per-twist ``k_point`` (up gets ``+k``, dn gets ``-k`` per BCS /
    # TR-pair convention; ``_build_per_twist_ao_leaf`` threads it down).
    geminal_expanded = obj.wavefunction_data.geminal_data
    orb_up_expanded = geminal_expanded.orb_data_up_spin
    orb_dn_expanded = geminal_expanded.orb_data_dn_spin
    if composite_child_kind is not None and not has_aos_up:
        orb_up_expanded = _swap_composite_child(
            orb_up_expanded, composite_child_kind, composite_child_data, new_struct, k_point=_kp_up
        )
        # dn: build the +k leaf from the payload (dn-specific when present,
        # else spin-shared), then flip it to -k via ``with_negated_k``
        # (negated G-list for PW; conjugated control points for splines).
        # Rebuilding directly at -k with the raw +k payload would yield
        # ``e^{-ik.r} u_k`` instead of ``conj(e^{ik.r} u_k)`` -- neither
        # phi^{+k} nor phi^{-k}.
        _dn_data = composite_child_dn_data if composite_child_dn_kind is not None else composite_child_data
        orb_dn_expanded = _swap_composite_child(
            orb_dn_expanded, composite_child_kind, _dn_data, new_struct, k_point=_kp_up, negate_k=True
        )

    from .determinant import SD_data, dispatch_loaded_geminal

    orb_up_final = orb_up_expanded.replace(mo_coefficients=mo_up)
    orb_dn_final = orb_dn_expanded.replace(mo_coefficients=mo_dn)
    if isinstance(geminal, SD_data):
        if has_any_lambda_payload:
            raise ValueError(
                f"{filepath}: twist {twist_index} is not a canonical occupied SD; "
                "regenerate the legacy lambda-bearing twist input."
            )
        expected_shapes = {
            "up": (
                int(geminal.num_electron_up),
                int(orb_up_expanded.aos_data.num_ao),
            ),
            "dn": (
                int(geminal.num_electron_dn),
                int(orb_dn_expanded.aos_data.num_ao),
            ),
        }
        for spin, coefficients in (("up", mo_up), ("dn", mo_dn)):
            shape = np.asarray(coefficients).shape
            if shape != expected_shapes[spin]:
                raise ValueError(
                    f"{filepath}: canonical SD twist {twist_index} {spin}-spin "
                    "MO coefficients must have exactly shape "
                    f"{expected_shapes[spin]} (occupied rows x AO columns); "
                    f"got {shape}."
                )
        new_geminal = SD_data.build(
            num_electron_up=int(geminal.num_electron_up),
            num_electron_dn=int(geminal.num_electron_dn),
            mos_up=orb_up_final,
            mos_dn=orb_dn_final,
            time_reversal=bool(geminal.time_reversal),
        )
    else:
        if not has_lambda_matrix:
            raise ValueError(f"{filepath}: twist {twist_index} is missing its lambda geminal payload.")
        new_geminal = dispatch_loaded_geminal(
            num_electron_up=int(geminal.num_electron_up),
            num_electron_dn=int(geminal.num_electron_dn),
            orb_data_up_spin=orb_up_final,
            orb_data_dn_spin=orb_dn_final,
            lambda_matrix=lambda_payload,
            time_reversal=bool(getattr(geminal, "time_reversal", int(geminal.num_electron_up) == int(geminal.num_electron_dn))),
        )
    from .determinant import require_opposite_spin_twists

    require_opposite_spin_twists(new_geminal)
    new_wf = obj.wavefunction_data.replace(geminal_data=new_geminal)
    obj = obj.replace(wavefunction_data=new_wf)

    # Remaining post-load fixups (image expansion already ran above).
    obj = _apply_ewald_rebuild_post_load(obj)
    obj = _apply_jastrow_structure_backfill_post_load(obj)
    obj = _apply_ao_basis_dtype_alignment_post_load(obj)
    obj = _apply_geminal_dtype_alignment_post_load(obj)
    obj = _apply_dtn_dtype_alignment_post_load(obj)
    obj.wavefunction_data.jastrow_data.sanity_check()
    return obj


# =============================================================================
# Per-twist + shared-state checkpoint layout
#
# Two HDF5 files per checkpoint directory:
#
#   checkpoint_dir/
#     twist_000.h5      per-twist walker state + that twist's current
#                       MO / lambda (uses the existing MCMC.save_to_hdf5
#                       layout under the hood). Written twist-root-only
#                       (intra_comm rank 0).
#     twist_001.h5      ... one per twist
#     shared_state.h5   the shared Jastrow_data (the cross-twist moving
#                       parameter block),
#                       written exactly ONCE per SR step by the inter-comm
#                       root (NOT duplicated N times across per-twist
#                       files -- those snapshot the Jastrow at write time
#                       but the authoritative version lives here).
#
# Resume contract:
#   - A per-twist run is "complete" iff its twist_NNN.h5 exists AND its
#     mcmc_counter >= target_steps AND /status/converged is set; a partial
#     sampling run is rerun from its first step, not skipped.
#   - On resume the shared Jastrow is loaded from shared_state.h5 and
#     replaces each per-twist file's snapshotted Jastrow so all twists
#     start the next SR step in lock-step.
# =============================================================================


def save_shared_jastrow(filepath: str, jastrow_data: Jastrow_data) -> None:
    """Write the shared Jastrow to ``shared_state.h5``.

    The payload kind and the actual Jastrow data identify this file.
    ``Jastrow_data`` is a flax struct dataclass,
    so the existing ``_save_dataclass_to_hdf5`` recursive writer handles
    every leaf (one-body, two-body, three-body, NN).

    Args:
        filepath: Destination ``shared_state.h5``.
        jastrow_data: :class:`jqmc.jastrow_factor.Jastrow_data` to persist.
    """
    import h5py

    from .hamiltonians import _save_dataclass_to_hdf5

    jastrow_data.sanity_check()
    with h5py.File(filepath, "w") as f:
        f.attrs["payload"] = "shared_jastrow"
        grp = f.create_group("jastrow_data")
        _save_dataclass_to_hdf5(grp, jastrow_data)


def load_shared_jastrow(filepath: str) -> Jastrow_data:
    """Load the shared Jastrow from ``shared_state.h5``.

    Args:
        filepath: Path to ``shared_state.h5``.

    Returns:
        Jastrow_data: The previously-saved shared Jastrow.

    Raises:
        ValueError: If the file is missing the expected ``payload`` attr
            (catches an accidental ``Hamiltonian_data`` mis-write).
    """
    import h5py

    from .hamiltonians import _load_dataclass_from_hdf5, _restore_shared_jastrow_after_load
    from .jastrow_factor import Jastrow_data

    with h5py.File(filepath, "r") as f:
        if f.attrs.get("payload") != "shared_jastrow":
            raise ValueError(
                f"{filepath}: missing or wrong ``payload`` attr (expected "
                f"``shared_jastrow``); refusing to load as a shared-state file."
            )
        return _restore_shared_jastrow_after_load(_load_dataclass_from_hdf5(Jastrow_data, f["jastrow_data"]))


def run_twist_average(
    twist_input_path: str,
    *,
    mcmc_seed: int = 34467,
    num_mcmc_steps: int,
    num_walkers: int = 40,
    num_mcmc_warmup_steps: int = 50,
    num_mcmc_bin_blocks: int = 10,
    max_time: float = float("inf"),
    world_comm: MPI.Comm | None = None,
    compute_force: bool = False,
    mcmc_kwargs: dict | None = None,
) -> TwistAverageResult:
    """File-path driver for a twist-averaged VMC run.

    Thin wrapper around :func:`run_twist_average_from_hamiltonians`. Reads
    the combined twist-averaged HDF5 input at ``twist_input_path``,
    builds one :class:`jqmc.hamiltonians.Hamiltonian_data` per twist via
    :func:`load_twist`, and forwards everything to the in-memory driver.

    Sample-allocation policy (equal compute per twist): every twist gets
    the same MCMC chain length / ``num_walkers`` regardless of its
    ``W_k``. Weights enter only at the Level-2 BZ combine.

    Args:
        twist_input_path: Combined ``hamiltonian_data.h5`` written by
            :func:`save_twist_average_input`.
        mcmc_seed: See :func:`run_twist_average_from_hamiltonians`.
        num_mcmc_steps: See :func:`run_twist_average_from_hamiltonians`.
        num_walkers: See :func:`run_twist_average_from_hamiltonians`.
        num_mcmc_warmup_steps: See
            :func:`run_twist_average_from_hamiltonians`.
        num_mcmc_bin_blocks: See
            :func:`run_twist_average_from_hamiltonians`.
        max_time: See :func:`run_twist_average_from_hamiltonians`.
        world_comm: See :func:`run_twist_average_from_hamiltonians`.
        compute_force: See :func:`run_twist_average_from_hamiltonians`.
        mcmc_kwargs: See :func:`run_twist_average_from_hamiltonians`.

    Returns:
        :class:`TwistAverageResult` with energy + per-twist diagnostics +
        (optional) atomic force.

    Raises:
        ValueError: If ``twist_input_path`` is not a combined input
        (use :func:`is_twist_average_input` to gate before calling).
    """
    meta = load_twist_average_input(twist_input_path)
    hamiltonian_list = [load_twist(twist_input_path, k) for k in range(meta.n_twist)]
    return run_twist_average_from_hamiltonians(
        hamiltonian_list=hamiltonian_list,
        weights=meta.weights,
        mcmc_seed=mcmc_seed,
        num_mcmc_steps=num_mcmc_steps,
        num_walkers=num_walkers,
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
        max_time=max_time,
        world_comm=world_comm,
        compute_force=compute_force,
        mcmc_kwargs=mcmc_kwargs,
    )


def is_twist_run_complete(filepath: str, target_steps: int) -> bool:
    """Resume predicate: has the per-twist MCMC reached its step count?

    Cheap predicate (no full ``MCMC.load_from_hdf5``): inspects the
    ``mcmc_counter`` scalar in the driver_config group of the per-twist
    checkpoint. Returns ``False`` for missing / unreadable files (the
    caller restarts from scratch) and for any counter strictly below the
    target.

    Args:
        filepath: ``<dir>/twist_NNN.h5`` path.
        target_steps: Required MCMC step count for this twist.

    Returns:
        ``True`` iff the file exists and ``mcmc_counter >= target_steps``.
    """
    import os

    import h5py

    if not os.path.exists(filepath):
        return False
    try:
        with h5py.File(filepath, "r") as f:
            # MCMC save_rank_checkpoint writes driver_config either under
            # /driver_config (post-merge) or under
            # /rank_NNN/driver_config (per-rank temporary). For the
            # twist-averaged per-twist file we accept either layout for
            # robustness.
            for grp_path in ("driver_config", "rank_0/driver_config"):
                if grp_path in f:
                    grp = f[grp_path]
                    if "mcmc_counter" in grp:
                        counter = int(np.asarray(grp["mcmc_counter"]).reshape(()))
                        return counter >= int(target_steps)
                    if "mcmc_counter" in grp.attrs:
                        return int(grp.attrs["mcmc_counter"]) >= int(target_steps)
            return False
    except (OSError, KeyError):
        return False
